/**
 * alt_est.h —— 竖直通道互补滤波：加速度计（1 kHz 预测）+ 气压计（~50 Hz 修正）→ z / vz，并在线估计竖直加计零偏 ab
 *
 * 三阶互补（三个误差极点都放在 −ω，等价于固定增益卡尔曼）：
 *   预测：a = a_up − ab，z += vz·dt + ½·a·dt²，vz += a·dt，  a_up = (R·a_body)_z − g（世界系竖直比力去重力）
 *   修正：e = z_baro − z，z += 3ω·dtb·e，vz += 3ω²·dtb·e，ab −= ω³·dtb·e   （dtb = 两次气压计样本间隔）
 * ω = ALT_W rad/s 决定气压计信任带宽：低于 ω 听气压计（无漂移），高于 ω 听加速度计（无延迟、无气压噪声）。
 * 为什么要第三阶：二阶版在 0.2 m/s²（20 mg）加计零偏下 vz 稳态偏 2ζ·b/ω ≈ 0.27 m/s（主机单测实测 0.25），
 *   电机振动整流、标定残差都会产生这种量级的零偏；第三阶把它估掉，稳态 vz 偏差 → 0。
 * 纯 C、单精度、无 ESP 依赖，可在主机上单测（test/host_test.c）。
 */
#ifndef ALT_EST_H
#define ALT_EST_H

#include <math.h>

#define ALT_W        1.0f     /* rad/s：气压计修正带宽（1.5 时真机静止 vz σ 0.13 m/s，1.0 见 DEVLOG Stage 24） */
#define ALT_AB_MAX   2.0f     /* m/s²：零偏估计限幅 */
#define ALT_G        9.81f
#define ALT_A_CLAMP  40.0f    /* m/s²：撞击 / 读数异常时限幅 */

typedef struct {
    float z, vz;      /* m（相对开机气压基准）、m/s（向上为正） */
    float ab;         /* m/s²：竖直加计零偏估计 */
    int   init;
} alt_est_t;

static inline void alt_reset(alt_est_t *e, float z0) { e->z = z0; e->vz = 0.0f; e->ab = 0.0f; e->init = 1; }

/* q 为机体→世界四元数 [w,x,y,z]；a_body 为机体系比力 m/s²（静止水平时 ≈ (0,0,+g)） */
static inline float alt_up_accel(const float q[4], const float a_body[3])
{
    const float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    const float r20 = 2.0f * (qx * qz - qw * qy);
    const float r21 = 2.0f * (qy * qz + qw * qx);
    const float r22 = 1.0f - 2.0f * (qx * qx + qy * qy);
    float a = r20 * a_body[0] + r21 * a_body[1] + r22 * a_body[2] - ALT_G;
    return a > ALT_A_CLAMP ? ALT_A_CLAMP : (a < -ALT_A_CLAMP ? -ALT_A_CLAMP : a);
}

static inline void alt_predict(alt_est_t *e, float a_up, float dt)
{
    const float a = a_up - e->ab;
    e->z += e->vz * dt + 0.5f * a * dt * dt;
    e->vz += a * dt;
}

static inline void alt_correct(alt_est_t *e, float z_baro, float dtb)
{
    if (!e->init) { alt_reset(e, z_baro); return; }
    if (dtb > 0.2f) dtb = 0.2f;                       /* 气压计断档后不一次性猛拉 */
    const float err = z_baro - e->z;
    e->z += 3.0f * ALT_W * dtb * err;
    e->vz += 3.0f * ALT_W * ALT_W * dtb * err;
    e->ab -= ALT_W * ALT_W * ALT_W * dtb * err;
    if (e->ab > ALT_AB_MAX) e->ab = ALT_AB_MAX;
    if (e->ab < -ALT_AB_MAX) e->ab = -ALT_AB_MAX;
}

#endif /* ALT_EST_H */
