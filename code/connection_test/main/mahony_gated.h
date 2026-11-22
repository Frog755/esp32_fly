/**
 * @file mahony_gated.h
 * @brief 门控 Mahony 姿态估计器（纯 C，单精度）—— 翻转时只靠陀螺，近悬停时用加速度计拉回漂移
 *
 * 对齐 RL/env/att_est_wrap.py 的 mahony_update(mode='gated')，校验脚本 RL/scripts/verify_expert9_c_equivalence.py。
 *
 * 原理：加速度计测的是"比力"（推力/质量，沿机体 z 轴），只有接近悬停平衡时才指向"上"。
 * 翻转 / 失重 / 大机动时它不含重力方向信息，普通 Mahony 会被它带偏 18°~30°（Stage 18b 实测）。
 * 门控：修正增益 = KP × g_f × g_w
 *   g_f = clamp(1 − |‖a‖/g − 1| / F_BAND, 0, 1)      比力模长偏离 1g 超过 F_BAND（15%）→ 0
 *   g_w = clamp((W_HI − ‖ω‖) / (W_HI − W_LO), 0, 1)  ‖ω‖ 在 W_LO→W_HI（1→3 rad/s）之间线性降到 0
 *
 * 调用约定：
 *   - 以 IMU 采样率调用（仿真标定为 1 kHz，dt = 0.001 s）
 *   - gyro：机体系角速度 rad/s（建议上电静止时先减去陀螺零偏）
 *   - acc ：机体系比力 m/s²（静止水平放置时约为 [0, 0, +9.81]）
 *   - 输出 m->q：机体→世界四元数 [w, x, y, z]，直接作为 recovery_expert9_step() 的 q 输入
 *   - 偏航由陀螺积分得到、不受加速度计约束（回正控制只用倾角，不需要航向）
 */
#ifndef MAHONY_GATED_H
#define MAHONY_GATED_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAHONY_KP       2.0f      /* 修正增益（Kp=5 反而变差，见 Stage 18b） */
#define MAHONY_F_BAND   0.15f     /* 比力模长门限：偏离 1g 的比例 */
#define MAHONY_W_LO     1.0f      /* rad/s */
#define MAHONY_W_HI     3.0f      /* rad/s */
#define MAHONY_G        9.81f

typedef struct {
    float q[4];                   /* [w, x, y, z] */
} mahony_gated_t;

static inline float mahony_clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

static inline void mahony_gated_init(mahony_gated_t *m, const float q0[4])
{
    float n = sqrtf(q0[0] * q0[0] + q0[1] * q0[1] + q0[2] * q0[2] + q0[3] * q0[3]);
    float inv = (n > 1e-6f) ? 1.0f / n : 0.0f;
    if (inv == 0.0f) { m->q[0] = 1.0f; m->q[1] = m->q[2] = m->q[3] = 0.0f; return; }
    for (int i = 0; i < 4; i++) m->q[i] = q0[i] * inv;
}

/** 静止时由加速度计初始化倾角（偏航置 0）：取把机体系"上"方向 â 转到世界 z 轴的最短旋转。 */
static inline void mahony_gated_init_from_accel(mahony_gated_t *m, const float acc[3])
{
    float n = sqrtf(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]);
    if (n < 1e-6f) { m->q[0] = 1.0f; m->q[1] = m->q[2] = m->q[3] = 0.0f; return; }
    const float ax = acc[0] / n, ay = acc[1] / n, az = acc[2] / n;
    const float sxy = sqrtf(ax * ax + ay * ay);              /* = sin θ，θ = â 与 ez 的夹角 */
    if (sxy < 1e-9f) {                                       /* 正好竖直：正立 → 单位四元数；倒置 → 绕 x 轴 180° */
        if (az > 0.0f) { m->q[0] = 1.0f; m->q[1] = 0.0f; }
        else           { m->q[0] = 0.0f; m->q[1] = 1.0f; }
        m->q[2] = m->q[3] = 0.0f; return;
    }
    /* 旋转轴 (ay, −ax, 0)/sxy，转角 θ = atan2(sxy, az)；用半角三角函数而非 [1+az, …]，近倒置时不丢精度 */
    const float h = 0.5f * atan2f(sxy, az);
    const float s = sinf(h) / sxy;
    m->q[0] = cosf(h); m->q[1] = ay * s; m->q[2] = -ax * s; m->q[3] = 0.0f;
}

/** 一次更新，返回本次实际使用的修正增益 k（0 = 纯陀螺积分），便于遥测。 */
static inline float mahony_gated_update(mahony_gated_t *m, const float gyro[3], const float acc[3], float dt)
{
    const float qw = m->q[0], qx = m->q[1], qy = m->q[2], qz = m->q[3];
    /* 加速度计方向 */
    const float fn = sqrtf(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]);
    const float inv = 1.0f / ((fn > 1e-6f) ? fn : 1e-6f);
    const float ax = acc[0] * inv, ay = acc[1] * inv, az = acc[2] * inv;
    /* 估计的机体系"上"方向 v = R^T·ez = R 第 3 行 */
    const float vx = 2.0f * (qx * qz - qw * qy);
    const float vy = 2.0f * (qy * qz + qw * qx);
    const float vz = 1.0f - 2.0f * (qx * qx + qy * qy);
    /* 误差 e = â × v */
    const float ex = ay * vz - az * vy;
    const float ey = az * vx - ax * vz;
    const float ez = ax * vy - ay * vx;
    /* 门控增益 */
    const float gf = mahony_clampf(1.0f - fabsf(fn / MAHONY_G - 1.0f) / MAHONY_F_BAND, 0.0f, 1.0f);
    const float wn = sqrtf(gyro[0] * gyro[0] + gyro[1] * gyro[1] + gyro[2] * gyro[2]);
    const float gw = mahony_clampf((MAHONY_W_HI - wn) / (MAHONY_W_HI - MAHONY_W_LO), 0.0f, 1.0f);
    const float k = MAHONY_KP * gf * gw;
    /* 修正后角速度 → 旋转向量 → 四元数增量 q ⊗ exp(ωc·dt) */
    const float rx = (gyro[0] + k * ex) * dt;
    const float ry = (gyro[1] + k * ey) * dt;
    const float rz = (gyro[2] + k * ez) * dt;
    const float ang = sqrtf(rx * rx + ry * ry + rz * rz);
    const float ainv = 1.0f / ((ang > 1e-12f) ? ang : 1e-12f);
    const float h = 0.5f * ang;
    const float dw = cosf(h), sh = sinf(h);
    const float dx = rx * ainv * sh, dy = ry * ainv * sh, dz = rz * ainv * sh;
    float nw = qw * dw - qx * dx - qy * dy - qz * dz;
    float nx = qw * dx + qx * dw + qy * dz - qz * dy;
    float ny = qw * dy - qx * dz + qy * dw + qz * dx;
    float nz = qw * dz + qx * dy - qy * dx + qz * dw;
    const float nrm = 1.0f / sqrtf(nw * nw + nx * nx + ny * ny + nz * nz);
    m->q[0] = nw * nrm; m->q[1] = nx * nrm; m->q[2] = ny * nrm; m->q[3] = nz * nrm;
    return k;
}

#ifdef __cplusplus
}
#endif

#endif /* MAHONY_GATED_H */
