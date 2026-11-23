/**
 * flight_core.h —— 飞控核心：姿态估计 + 竖直估计 + 飞行状态机 + v9 控制器（纯 C，无 ESP 依赖）
 *
 * 固件 ctrl 任务（1 kHz）、主机单测、软件在环仿真（RL/scripts/sil_flight_v9.py）调用的是同一个 fc_step()。
 * 输入为已标定、已映射到机体 FLU 的 IMU 数据：加速度 m/s²（静止水平 ≈ (0,0,+g)）、角速度 rad/s（已扣零偏）。
 * 每 5 次调用（200 Hz）跑一次状态机 + 控制器，更新 u_out（只有 fsm_motors_on 时非零）。
 */
#ifndef FLIGHT_CORE_H
#define FLIGHT_CORE_H

#include <string.h>
#include <math.h>
#include "mahony_gated.h"
#include "alt_est.h"
#include "recovery_expert9.h"
#include "flight_fsm.h"

#define FC_BARO_STALE_S  0.2f
#define FC_LINK_TIMEOUT  0.6f     /* s：遥控心跳超过这么久没来 → 链路丢失（手机每 0.1 s 发一次） */

typedef struct {
    mahony_gated_t mg;
    alt_est_t alt;
    recovery9_state_t rc;
    fsm_t fsm;
    uint32_t tick;
    float t_baro;                 /* 距上次气压计样本 s */
    float t_link;                 /* 距上次遥控心跳 s */
    float vbatt;
    float k, anorm_g, w_norm, thrust_frac;
    float u_cmd[4], u_out[4];
    uint8_t mode;
    int pending_cmd;              /* 下一个 200 Hz 拍处理的指令（disarm 不会被别的指令覆盖） */
    uint32_t ev_seq; uint8_t ev_last;
} fc_t;

static inline void fc_init(fc_t *c, const float a0_body[3])
{
    memset(c, 0, sizeof(*c));
    mahony_gated_init_from_accel(&c->mg, a0_body);
    alt_reset(&c->alt, 0.0f);
    recovery9_reset(&c->rc);
    fsm_init(&c->fsm);
    c->vbatt = NAN;
    c->anorm_g = 1.0f;
    c->t_link = 1e9f;             /* 上电时没有链路 */
}

/* 收到一次遥控心跳（手机网页 / 串口 hb）时调用 */
static inline void fc_link_beat(fc_t *c) { c->t_link = 0.0f; }
static inline int fc_link_ok(const fc_t *c) { return c->t_link < FC_LINK_TIMEOUT; }

static inline void fc_command(fc_t *c, fsm_cmd_t cmd)
{
    if (cmd == FSM_CMD_NONE) return;
    if (c->pending_cmd == FSM_CMD_DISARM) return;
    c->pending_cmd = (int)cmd;
}

/* 1 kHz。baro_new=1 时 baro_z 为新样本（m，相对开机），baro_dtb 为与上一样本间隔；vbatt 可为 NaN。
 * 返回 1 表示本拍跑了 200 Hz 控制（u_out 已更新）。 */
static inline int fc_step(fc_t *c, const float a[3], const float g[3], float dt, int imu_ok,
                          int baro_new, float baro_z, float baro_dtb, float vbatt)
{
    c->tick++;
    if (!isnan(vbatt)) c->vbatt = vbatt;
    c->k = mahony_gated_update(&c->mg, g, a, dt);
    c->anorm_g = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) / MAHONY_G;
    c->w_norm = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    alt_predict(&c->alt, alt_up_accel(c->mg.q, a), dt);
    c->t_baro += dt;
    if (c->t_link < 1e8f) c->t_link += dt;
    if (baro_new) { alt_correct(&c->alt, baro_z, baro_dtb); c->t_baro = 0.0f; }
    if (c->tick % 5u) return 0;

    /* ---------------- 200 Hz ---------------- */
    const float *q = c->mg.q;
    const float qn = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    fsm_in_t in = {.vbatt = c->vbatt, .acc_norm_g = c->anorm_g, .r22 = 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]),
                   .w_norm = c->w_norm, .gate_k = c->k, .z = c->alt.z, .vz = c->alt.vz, .thrust_frac = c->thrust_frac,
                   .est_ok = imu_ok && isfinite(qn) && qn > 0.5f && c->t_baro < FC_BARO_STALE_S && isfinite(c->alt.z),
                   .link_ok = fc_link_ok(c)};
    const fsm_cmd_t cmd = (fsm_cmd_t)c->pending_cmd; c->pending_cmd = FSM_CMD_NONE;
    const int was_on = fsm_motors_on(&c->fsm);
    const fsm_event_t ev = fsm_step(&c->fsm, cmd, &in, 0.005f);
    if (ev != FSM_EV_NONE) { c->ev_seq++; c->ev_last = (uint8_t)ev; }
    if (fsm_motors_on(&c->fsm) && !was_on) recovery9_reset(&c->rc);

    const float vb = isfinite(c->vbatt) ? c->vbatt : EXP9_VREF;
    const float pos[3] = {0.0f, 0.0f, c->alt.z}, vel[3] = {0.0f, 0.0f, c->alt.vz}, tgt[3] = {0.0f, 0.0f, c->fsm.z_target};
    c->mode = recovery_expert9_step(&c->rc, q, g, pos, vel, c->alt.vz, c->alt.z - c->fsm.z_target, tgt, vb, c->u_cmd);
    const float vr = (vb > EXP9_VBATT_MIN ? vb : EXP9_VBATT_MIN) / EXP9_VREF;
    float T = 0.0f;
    for (int i = 0; i < 4; i++) T += c->u_cmd[i] * c->u_cmd[i];
    c->thrust_frac = T * EXP9_TMAX * vr * vr / (EXP9_MASS * EXP9_G);
    const int on = fsm_motors_on(&c->fsm);
    for (int i = 0; i < 4; i++) c->u_out[i] = on ? c->u_cmd[i] : 0.0f;
    return 1;
}

#endif /* FLIGHT_CORE_H */
