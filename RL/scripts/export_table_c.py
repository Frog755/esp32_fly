#!/usr/bin/env python3
"""export_table_c.py —— 将 3D 翻转表 (flip_table_3d.npz) 导出为 ESP32-S3 适用的 C 头文件

生成的头文件包含:
1. flip_table_data.h:
   - 全仿真通过的有效格点数组 (只收录 passed==True)
   - 存储空间: ~170格 * (3 key + 1 tf + 40 knots) * 4 bytes ≈ 30 KB (完全可在 Flash 中作为 const 存储)
2. flip_table.h:
   - 单步查找函数 lookup_flip_cell(q, w)
   - 线性插值函数 interpolate_cell_throttle(cell, t, u_out)
"""
import os, sys
import numpy as np

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz_path = os.path.join(root_dir, 'models', 'flip_table_3d.npz')
    out_dir = os.path.join(root_dir, '..', 'code', 'connection_test', 'main')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(npz_path):
        print(f"错误: 找不到翻转表文件 {npz_path}")
        sys.exit(1)

    d = np.load(npz_path)
    keys = d['keys']          # (N, 3): [phi, tilt, w_par]
    passed = d['passed']      # (N,) bool
    tfs = d['tfs']            # (N,) float
    u_opts = d['u_opts']      # (N, 10, 4) float

    valid_idx = np.where(passed)[0]
    total_cells = len(keys)
    valid_count = len(valid_idx)
    print(f"读取翻转表: 总格数 {total_cells}, 全仿真合格格数 {valid_count} ({valid_count/total_cells*100:.1f}%)")

    # 1. 生成 flip_table_data.h
    data_header_path = os.path.join(out_dir, 'flip_table_data.h')
    with open(data_header_path, 'w', encoding='utf-8') as f:
        f.write("/*\n")
        f.write(" * flip_table_data.h —— 3D 翻转最优控制表数据 (常量表存放在 Flash)\n")
        f.write(f" * 自动生成于 export_table_c.py，有效格数: {valid_count}/{total_cells}\n")
        f.write(" */\n\n")
        f.write("#ifndef FLIP_TABLE_DATA_H\n")
        f.write("#define FLIP_TABLE_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define FLIP_TABLE_VALID_COUNT {valid_count}\n")
        f.write("#define FLIP_TABLE_KNOTS 10\n\n")
        f.write("typedef struct {\n")
        f.write("    float phi;       /* 方位角 [0, 360) deg */\n")
        f.write("    float tilt;      /* 倒扣倾角 [115, 175] deg */\n")
        f.write("    float w_par;     /* 回正轴向角速度 rad/s */\n")
        f.write("    float tf;        /* 轨迹时长秒 */\n")
        f.write("    float knots[FLIP_TABLE_KNOTS][4]; /* 10个时间节点的 4 电机油门 [0, 1] */\n")
        f.write("} flip_cell_t;\n\n")
        f.write("static const flip_cell_t FLIP_TABLE[FLIP_TABLE_VALID_COUNT] = {\n")

        for i, idx in enumerate(valid_idx):
            phi = float(keys[idx, 0])
            tilt = float(keys[idx, 1])
            w_par = float(keys[idx, 2])
            tf = float(tfs[idx])
            knots = u_opts[idx]  # (10, 4)

            f.write(f"    /* [{i:3d}] cell idx {idx:3d}: phi={phi:5.1f}, tilt={tilt:5.1f}, w={w_par:+5.1f}, tf={tf:.3f} */\n")
            f.write("    {\n")
            f.write(f"        .phi = {phi:.2f}f, .tilt = {tilt:.2f}f, .w_par = {w_par:.2f}f, .tf = {tf:.3f}f,\n")
            f.write("        .knots = {\n")
            for k in range(10):
                u0, u1, u2, u3 = knots[k]
                f.write(f"            {{{u0:.4f}f, {u1:.4f}f, {u2:.4f}f, {u3:.4f}f}},\n")
            f.write("        }\n")
            f.write("    },\n")

        f.write("};\n\n")
        f.write("#endif /* FLIP_TABLE_DATA_H */\n")

    print(f"写入成功: {data_header_path} ({os.path.getsize(data_header_path)/1024:.1f} KB)")

    # 2. 生成 flip_table.h (算法逻辑接口)
    algo_header_path = os.path.join(out_dir, 'flip_table.h')
    with open(algo_header_path, 'w', encoding='utf-8') as f:
        f.write("""/*
 * flip_table.h —— ESP32-S3 3D 翻转表快速检索与插值驱动
 *
 * 特性:
 *   1. 纯 C99 实现，零动态内存分配 (无 malloc)
 *   2. 欧氏加权球面就近匹配，O(N) 遍历耗时约 2 微秒 (N ~ 170)
 *   3. 匹配门限守护：超出覆盖范围返回 NULL，平滑退回 v5 规则兜底
 *   4. 时间轴线性插值，耗时 < 0.1 微秒
 */

#ifndef FLIP_TABLE_H
#define FLIP_TABLE_H

#include <math.h>
#include <string.h>
#include "flip_table_data.h"

/* 匹配容限门限 */
#define FLIP_GATE_DPHI_MAX  35.0f   /* 方位角最大容差 deg */
#define FLIP_GATE_DTILT_MAX 30.0f   /* 倾角最大容差 deg */
#define FLIP_GATE_DW_MAX    10.0f   /* 角速度最大容差 rad/s */

/* 权重缩放系数 */
#define FLIP_SCALE_PHI   (1.0f / 45.0f)
#define FLIP_SCALE_TILT  (1.0f / 20.0f)
#define FLIP_SCALE_W     (1.0f / 6.0f)

/**
 * 查表定位最近合法轨迹
 * @param q: 归一化机体四元数 [qw, qx, qy, qz]
 * @param w: 机体角速度 [wx, wy, wz] (rad/s)
 * @return: 命中则返回 const flip_cell_t*，超出容限或接近水平返回 NULL
 */
static inline const flip_cell_t* lookup_flip_cell(const float q[4], const float w[3])
{
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    if (qw < 0.0f) {
        qw = -qw; qx = -qx; qy = -qy; qz = -qz;
    }

    float qv_sq = qx * qx + qy * qy + qz * qz;
    if (qv_sq < 1e-8f) {
        return NULL; /* 接近正立，无需翻转 */
    }

    float qn = sqrtf(qv_sq);
    float ang_deg = 2.0f * atan2f(qn, qw) * (180.0f / 3.141592653589793f);
    if (ang_deg < 100.0f) {
        return NULL; /* 倾角小于 100°，不进入倒扣打靶 */
    }

    /* 归一化旋转轴 axis */
    float ax = qx / qn;
    float ay = qy / qn;
    float az = qz / qn;

    /* 方位角 phi [0, 360) */
    float phi_rad = atan2f(ay, ax);
    float phi_deg = phi_rad * (180.0f / 3.141592653589793f);
    if (phi_deg < 0.0f) {
        phi_deg += 360.0f;
    }

    /* 回正方向（-axis）角速度分量 */
    float w_par = -(w[0] * ax + w[1] * ay + w[2] * az);

    /* 遍历检索最小加权距离格点 */
    float best_dist = 1e9f;
    int best_i = -1;
    float best_dphi = 0.0f, best_dtilt = 0.0f, best_dw = 0.0f;

    for (int i = 0; i < FLIP_TABLE_VALID_COUNT; i++) {
        const flip_cell_t *c = &FLIP_TABLE[i];

        float dphi = fabsf(c->phi - phi_deg);
        if (dphi > 180.0f) {
            dphi = 360.0f - dphi;
        }
        float dtilt = fabsf(c->tilt - ang_deg);
        float dw = fabsf(c->w_par - w_par);

        float term_phi = dphi * FLIP_SCALE_PHI;
        float term_tilt = dtilt * FLIP_SCALE_TILT;
        float term_w = dw * FLIP_SCALE_W;
        float dist = term_phi * term_phi + term_tilt * term_tilt + term_w * term_w;

        if (dist < best_dist) {
            best_dist = dist;
            best_i = i;
            best_dphi = dphi;
            best_dtilt = dtilt;
            best_dw = dw;
        }
    }

    /* 门限过滤 */
    if (best_i < 0 ||
        best_dphi > FLIP_GATE_DPHI_MAX ||
        best_dtilt > FLIP_GATE_DTILT_MAX ||
        best_dw > FLIP_GATE_DW_MAX) {
        return NULL;
    }

    return &FLIP_TABLE[best_i];
}

/**
 * 根据轨迹耗时计算当前时刻 4 电机插值油门
 * @param cell: 选中的翻转格点
 * @param t_elapsed: 自打靶触发以来经过的物理秒数
 * @param throttle_out: 输出 4 电机油门 [u0, u1, u2, u3]，范围 [0, 1]
 * @return: 1 表示正在轨迹期内执行，0 表示已到达/超过 tf 需移交 v5
 */
static inline int interpolate_cell_throttle(const flip_cell_t *cell, float t_elapsed, float throttle_out[4])
{
    if (!cell) return 0;
    if (t_elapsed >= cell->tf) {
        return 0; /* 轨迹结束 */
    }

    float tau = t_elapsed / cell->tf; /* 归一化时间 [0, 1) */
    float knot_f = tau * (FLIP_TABLE_KNOTS - 1);
    int k = (int)knot_f;
    if (k >= FLIP_TABLE_KNOTS - 1) {
        k = FLIP_TABLE_KNOTS - 2;
    }
    float frac = knot_f - (float)k;

    for (int m = 0; m < 4; m++) {
        float u_low = cell->knots[k][m];
        float u_high = cell->knots[k + 1][m];
        float u = u_low + frac * (u_high - u_low);
        if (u < 0.0f) u = 0.0f;
        if (u > 1.0f) u = 1.0f;
        throttle_out[m] = u;
    }

    return 1;
}

#endif /* FLIP_TABLE_H */
""")

    print(f"写入成功: {algo_header_path} ({os.path.getsize(algo_header_path)/1024:.1f} KB)")
    print("ESP32-S3 C 代码导出管线就绪！")

if __name__ == "__main__":
    main()

