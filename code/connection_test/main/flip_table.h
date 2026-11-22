/*
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
