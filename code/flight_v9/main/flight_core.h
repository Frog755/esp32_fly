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
#include "acc_lpf.h"
#include "alt_est.h"
#include "recovery_expert9.h"
#include "flight_fsm.h"

#define FC_BARO_STALE_S  0.2f
#define FC_LINK_TIMEOUT  0.6f     /* s：遥控心跳超过这么久没来 → 链路丢失（手机每 0.1 s 发一次） */
/* ---- 推力→占空比换算（Stage 31 录音实测 + SIL）----
 * v9 控制器按仿真推力模型算指令：T = EXP9_TMAX·(u·Vb/Vref)²，模型悬停 u ≈ 0.85（3.7 V）。
 * 真机（30% 时转速 ~209 转/秒，40% 快离地）推力大得多：静止电压 FC_HOVER_VB 时悬停占空比约 FC_HOVER_DUTY。
 * 低油门有刷空心杯转速 ≈ 占空比 × 电压，推力 ∝ 转速² → 真机 T ≈ A·(d·V)²，与模型同形，只差一个常数：
 *   真实占空比 d = FC_DUTY_SCALE × c_k × 控制器指令 u
 *   FC_DUTY_SCALE = 真机悬停占空比 / 模型悬停占空比（同一电压下）；c_k = 空中在线修正（见下）。
 * 控制器本身不改（与 Python / C 等价校验、v9 调参不受影响）；u_cmd 仍是模型空间，u_out 是真实占空比。
 *
 * 1) 控制用电压 = 电机通电前最后一次读数（静止电压），通电期间保持不变：
 *    实测电机一转读数就掉 ~0.8 V（30% / 40% 一样，原因未明：测量偏差或接触电阻）。用这个读数做电压补偿，
 *    控制器会以为推力上限只剩重力的 ~1.1 倍，指令一直顶在上限、没有余量（SIL：悬停估计差 5% 就起不来）。
 *    一次飞行 ≤ 40 s，静止电压的真实下降 < 0.1 V，可以忽略。低电压保护仍用实时读数。
 * 2) 推力在线修正 c_k：机体 z 轴比力（加计，低通）/ 控制器以为的推力比 = 实际推力 / 以为的推力（不管怎么动都成立）。
 *    只在**确定离地**时更新（HOLD，或 TAKEOFF 且高度比起飞点高 0.3 m）：贴地时地面托着，比力永远 = 1 g，会把推力不够误判成够了。
 *    c_k ← c_k·(以为/实际)^(dt/2τ)，τ = FC_ADAPT_TAU；限幅 [FC_ADAPT_MIN, FC_ADAPT_MAX]；每次开机从 1 开始。
 * 3) 贴地推力不够时上调 c_k（Stage 34）：第 1 次系留首飞（Stage 33）推力 95% 时间饱和、离地只有 4 cm——
 *    控制器以为推力是重力 1.58 倍，机体比力实测 0.98 g；而 2) 要离地 0.3 m 才启动，推力不够就永远不启动。
 *    起飞中、还没离地、控制器以为的推力 > FC_GROUND_TF 倍重力、比力 < FC_GROUND_AZ（没在加速上升）时，
 *    c_k 以 FC_GROUND_RATE（/s）缓慢上调；一旦开始加速上升就停，离地后由 2) 双向修正（推力多了会拉回来）。
 *    贴地时地面托着，比力 ≤ 1 g，只能判断"不够"、不能判断"够了"，所以只往上调。
 * 悬停估计（Stage 34）：Stage 33 实测静止 4.14 V、贴地时占空比 0.53 推力约等于重力（读数掉到 3.0~3.1 V）。
 *   供电修好后（不再掉 1 V）悬停会低得多（按 T ∝ (d·V)² 约 0.42）；没修好、离地后约 0.55~0.6。
 *   取中间 0.50：c_k 0.75~1.30 覆盖 0.375~0.65，两种情况都在范围内。 */
#ifndef FC_HOVER_DUTY
#define FC_HOVER_DUTY    0.50f    /* 静止电压 FC_HOVER_VB 时的悬停占空比 */
#endif
#ifndef FC_HOVER_VB
#define FC_HOVER_VB      4.14f    /* V：Stage 33 首飞前静止电压 */
#endif
#define FC_MODEL_HOVER_U 0.85246f /* sqrt(EXP9_MASS·g / (4·EXP9_TMAX))：模型在 EXP9_VREF 下的悬停指令 */
#define FC_DUTY_SCALE    (FC_HOVER_DUTY * FC_HOVER_VB / (FC_MODEL_HOVER_U * EXP9_VREF))
#ifndef FC_THRUST_ADAPT
#define FC_THRUST_ADAPT  1
#endif
#define FC_ADAPT_TAU     1.5f     /* s：在线修正时间常数 */
#define FC_ADAPT_LP_TAU  0.25f    /* s：比力 / 推力比的低通 */
#define FC_ADAPT_MIN     0.75f    /* 修正范围：悬停占空比 ×0.75 ~ ×1.30 */
#define FC_ADAPT_MAX     1.30f
#define FC_ADAPT_CLIMB   0.30f    /* m：TAKEOFF 中高度比起飞点高这么多才算离地 */
#ifndef FC_GROUND_ADAPT
#define FC_GROUND_ADAPT  1
#endif
#define FC_GROUND_T0     0.5f     /* s：进入 TAKEOFF 这么久之后才开始判断（推力爬升中） */
#define FC_GROUND_TF     1.20f    /* 控制器以为的推力 / 重力 超过它 …… */
#define FC_GROUND_AZ     1.05f    /* …… 而机体比力低于它（g）→ 推力不够 */
#define FC_GROUND_RATE   0.15f    /* 1/s：c_k 上调速率（1.0 → 1.30 约 1.8 s） */

#ifndef FC_ACC_LPF_HZ
#define FC_ACC_LPF_HZ    10.0f    /* Hz：喂给门控 Mahony 的加计低通（Stage 30 真机振动回放：姿态误差 7.5° → 0.8°，见 acc_lpf.h） */
#endif

typedef struct {
    mahony_gated_t mg;
    acc_lpf_t alp;                /* 加计低通（只给 Mahony；状态机 / 竖直滤波用原始加计） */
    alt_est_t alt;
    recovery9_state_t rc;
    fsm_t fsm;
    uint32_t tick;
    float t_baro;                 /* 距上次气压计样本 s */
    float t_link;                 /* 距上次遥控心跳 s */
    float vbatt;
    float k, anorm_g, w_norm, thrust_frac;
    float u_cmd[4], u_out[4];
    float vb_ctrl;                /* 控制用电压：通电前最后一次读数，通电期间保持 */
    float c_k, az_lp, tf_lp;      /* 推力在线修正系数、机体 z 比力（g）与以为推力比的低通 */
    uint8_t adapting;             /* 本拍是否在更新 c_k（遥测） */
    uint8_t mode;
    int pending_cmd;              /* 下一个 200 Hz 拍处理的指令（disarm 不会被别的指令覆盖） */
    uint32_t ev_seq; uint8_t ev_last;
} fc_t;

static inline void fc_init(fc_t *c, const float a0_body[3])
{
    memset(c, 0, sizeof(*c));
    mahony_gated_init_from_accel(&c->mg, a0_body);
    acc_lpf_init(&c->alp, a0_body);
    alt_reset(&c->alt, 0.0f);
    recovery9_reset(&c->rc);
    fsm_init(&c->fsm);
    c->vbatt = NAN;
    c->vb_ctrl = NAN;
    c->c_k = 1.0f; c->az_lp = 1.0f; c->tf_lp = 1.0f;
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
    if (FC_ACC_LPF_HZ > 0.0f) { acc_lpf_step(&c->alp, a, dt, FC_ACC_LPF_HZ); c->k = mahony_gated_update(&c->mg, g, c->alp.y, dt); }
    else c->k = mahony_gated_update(&c->mg, g, a, dt);
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
                   .w_norm = c->w_norm, .wz = g[2], .gate_k = c->k, .z = c->alt.z, .vz = c->alt.vz, .thrust_frac = c->thrust_frac,
                   .est_ok = imu_ok && isfinite(qn) && qn > 0.5f && c->t_baro < FC_BARO_STALE_S && isfinite(c->alt.z),
                   .link_ok = fc_link_ok(c)};
    const fsm_cmd_t cmd = (fsm_cmd_t)c->pending_cmd; c->pending_cmd = FSM_CMD_NONE;
    const int was_on = fsm_motors_on(&c->fsm);
    const fsm_event_t ev = fsm_step(&c->fsm, cmd, &in, 0.005f);
    if (ev != FSM_EV_NONE) { c->ev_seq++; c->ev_last = (uint8_t)ev; }
    if (fsm_motors_on(&c->fsm) && !was_on) recovery9_reset(&c->rc);

    const int on = fsm_motors_on(&c->fsm);
    if (!on && isfinite(c->vbatt)) c->vb_ctrl = c->vbatt;                 /* 通电前一直跟随，通电后保持 */
    const float vb = isfinite(c->vb_ctrl) ? c->vb_ctrl : (isfinite(c->vbatt) ? c->vbatt : EXP9_VREF);
    const float pos[3] = {0.0f, 0.0f, c->alt.z}, vel[3] = {0.0f, 0.0f, c->alt.vz}, tgt[3] = {0.0f, 0.0f, c->fsm.z_target};
    c->mode = recovery_expert9_step(&c->rc, q, g, pos, vel, c->alt.vz, c->alt.z - c->fsm.z_target, tgt, vb, c->u_cmd);
    const float vr = (vb > EXP9_VBATT_MIN ? vb : EXP9_VBATT_MIN) / EXP9_VREF;
    float T = 0.0f;
    for (int i = 0; i < 4; i++) T += c->u_cmd[i] * c->u_cmd[i];
    const float tf_prev = c->thrust_frac;                                   /* 上一拍的指令 ≈ 本拍已产生的推力（电机滞后 ~30 ms） */
    c->thrust_frac = T * EXP9_TMAX * vr * vr / (EXP9_MASS * EXP9_G);
    /* ---- 推力在线修正 ---- */
    const float al = 0.005f / (FC_ADAPT_LP_TAU + 0.005f);
    c->az_lp += al * (c->alp.y[2] / MAHONY_G - c->az_lp);
    c->tf_lp += al * (tf_prev - c->tf_lp);
    const int airborne = (c->fsm.st == FSM_HOLD) ||
                         (c->fsm.st == FSM_TAKEOFF && c->alt.z > c->fsm.z_goal - FSM_TAKEOFF_H + FC_ADAPT_CLIMB);
    c->adapting = (uint8_t)(FC_THRUST_ADAPT && on && airborne && in.r22 > 0.9f && c->w_norm < 1.5f && c->tf_lp > 0.25f);   /* 推力过大时控制器会把指令压到下限 0.3·mg，门槛要低于它 */
    if (c->adapting) {
        float ratio = c->az_lp / c->tf_lp;                                  /* 实际推力 / 以为的推力 */
        ratio = ratio < 0.5f ? 0.5f : (ratio > 2.0f ? 2.0f : ratio);
        c->c_k *= powf(ratio, -0.005f / (2.0f * FC_ADAPT_TAU));
    } else if (FC_GROUND_ADAPT && on && c->fsm.st == FSM_TAKEOFF && !airborne && c->fsm.t_state > FC_GROUND_T0 &&
               in.r22 > 0.9f && c->tf_lp > FC_GROUND_TF && c->az_lp < FC_GROUND_AZ) {
        c->c_k *= 1.0f + FC_GROUND_RATE * 0.005f;                           /* 贴地推力不够：只往上调 */
        c->adapting = 2;
    }
    c->c_k = c->c_k < FC_ADAPT_MIN ? FC_ADAPT_MIN : (c->c_k > FC_ADAPT_MAX ? FC_ADAPT_MAX : c->c_k);
    for (int i = 0; i < 4; i++) c->u_out[i] = on ? c->u_cmd[i] * FC_DUTY_SCALE * c->c_k : 0.0f;
    return 1;
}

#endif /* FLIGHT_CORE_H */
