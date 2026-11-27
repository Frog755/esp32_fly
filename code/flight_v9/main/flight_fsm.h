/**
 * flight_fsm.h —— 飞行状态机：解锁 / 抛出或起飞 / 回正 / 悬停 ≥5 s / 自动降落 / 触地停桨 + 全部安全保护
 * （200 Hz 调用；纯 C、无 ESP 依赖，主机单测 test/host_test.c、软件在环仿真 RL/scripts/sil_flight_v9.py 用的是同一份代码）
 *
 *   DISARMED ──arm（电压 ≥3.6 V、倾角 <20°、静止）──▶ ARMED ─┬─ 失重 40 ms（抛出 / 松手）──▶ RECOVER ──稳定 0.25 s──▶ HOLD
 *       ▲                                                      └─ takeoff 指令（在地面）────▶ TAKEOFF ──到达 +0.8 m 且稳定──▶ HOLD
 *       │                                                                                                                 │ 悬停满 5 s / land 指令
 *       │                                                                                                                 ▼
 *       └──────────── 触地检测（低推力 + 竖直不动 0.3 s）◀────────── LAND（目标高度以 0.35 m/s 下降，不需要知道地面高度）
 *
 * 只在 TAKEOFF / RECOVER / HOLD / LAND 四个状态输出电机（fsm_motors_on）。
 * 停桨（→ DISARMED）：disarm 指令（任何状态立即生效）、触地、撞击 >6 g、估计器失效、LAND 超过 20 s、ARMED 20 s 未出手。
 * 转入降落（不停桨，空中停桨等于摔机）：悬停满 5 s、land 指令、低电压、回正 4 s 未稳定、起飞 6 s 未到高度、通电飞行累计 40 s、
 *   遥控链路丢失（手机心跳超时，见 flight_core.h FC_LINK_TIMEOUT；LAND 中丢失不影响继续降落）、
 *   起飞 / 悬停中偏航自转 > 半圈每秒持续 0.5 s（land_spin，Stage 34）。
 * 链路：解锁必须有链路；ARMED 中链路丢失 → 上锁（还没出手，最安全的处理）。
 * "稳定"判据与仿真评测一致（vec_recovery_env.STABLE_CRIT）：机体 z 轴夹角 <18°、|ω| <1.2 rad/s、|vz| <0.5 m/s。
 */
#ifndef FLIGHT_FSM_H
#define FLIGHT_FSM_H

#include <stdint.h>
#include <math.h>

/* ---- 解锁 / 出手 ---- */
#define FSM_ARM_VMIN        3.60f   /* V：解锁最低电压（耀坤作者实测"静态 >3.6 V 才能起飞"） */
#define FSM_ARM_R22_MIN     0.94f   /* 解锁时倾角 < 20° */
#define FSM_ARM_K_MIN       1.0f    /* 解锁时门控 Mahony 增益（加计在修正 = 静止） */
#define FSM_ARM_TIMEOUT     20.0f   /* s：解锁后未出手 / 未起飞 自动上锁 */
#define FSM_FREEFALL_G      0.35f   /* 比力模长 < 0.35 g 视为失重（已出手） */
#define FSM_FREEFALL_T      0.040f
/* ---- 任务 ---- */
#define FSM_HOLD_S          5.0f    /* s：稳定后悬停时长，之后自动降落 */
#define FSM_TAKEOFF_H       0.8f    /* m：起飞高度（相对起飞点） */
#define FSM_TAKEOFF_V       0.5f    /* m/s：起飞时目标高度上升速率 */
#define FSM_TAKEOFF_ZTOL    0.25f   /* m：到达判据 */
#define FSM_TAKEOFF_TMAX    6.0f    /* s */
#define FSM_RECOVER_TMAX    4.0f    /* s：回正后仍未稳定 → 降落 */
#define FSM_STABLE_R22      0.95f
#define FSM_STABLE_W        1.2f    /* rad/s */
#define FSM_STABLE_VZ       0.5f    /* m/s */
#define FSM_STABLE_T        0.25f   /* s */
#define FSM_ACTIVE_TMAX     40.0f   /* s：通电飞行累计上限 → 降落 */
/* ---- 降落 ---- */
#define FSM_LAND_V          0.35f   /* m/s：下降速率 */
#define FSM_LAND_LEAD_MAX   0.6f    /* m：目标最多领先当前高度这么多（触地后不无限下拉） */
#define FSM_LAND_MIN_T      1.0f    /* s：进入 LAND 1 s 后才开始触地检测 */
#define FSM_LAND_THRUST     0.6f    /* 总推力 < 0.6 × 重力 …… */
#define FSM_LAND_VZ         0.30f   /* …… 且 |vz| < 0.3 m/s …… */
#define FSM_LAND_T          0.30f   /* …… 持续 0.3 s → 判定触地 */
#define FSM_LAND_TMAX       20.0f   /* s：降落超时 → 停桨 */
/* ---- 保护 ---- */
#define FSM_LOWV            3.00f   /* V：飞行中低于此电压持续 0.5 s → 降落 */
#define FSM_LOWV_T          0.5f
#define FSM_IMPACT_G        6.0f    /* 比力模长 > 6 g 视为撞击 → 停桨 */
#define FSM_IMPACT_ARM_T    0.3f    /* s：开始飞行 0.3 s 后才启用撞击判据（出手瞬间的冲击） */
#ifndef FSM_SPIN_W
#define FSM_SPIN_W          3.14f   /* rad/s：起飞 / 悬停中偏航转速超过半圈每秒 …… */
#endif
#define FSM_SPIN_T          0.5f    /* …… 持续 0.5 s → 转入降落（Stage 33 首飞：推力饱和时自转加速到 2.5 圈/秒） */

typedef enum { FSM_DISARMED = 0, FSM_ARMED, FSM_TAKEOFF, FSM_RECOVER, FSM_HOLD, FSM_LAND, FSM__NST } fsm_state_t;
typedef enum { FSM_CMD_NONE = 0, FSM_CMD_ARM, FSM_CMD_DISARM, FSM_CMD_TAKEOFF, FSM_CMD_LAND } fsm_cmd_t;
typedef enum {
    FSM_EV_NONE = 0,
    FSM_EV_ARMED, FSM_EV_LAUNCH, FSM_EV_TAKEOFF, FSM_EV_STABLE,
    FSM_EV_LAND_AUTO, FSM_EV_LAND_CMD, FSM_EV_LAND_LOWV, FSM_EV_LAND_RECOVER_TO, FSM_EV_LAND_TAKEOFF_TO, FSM_EV_LAND_ACTIVE_TO,
    FSM_EV_LAND_LINK,
    FSM_EV_LANDED,
    FSM_EV_DISARM_CMD, FSM_EV_ARM_TIMEOUT, FSM_EV_LAND_TIMEOUT, FSM_EV_IMPACT, FSM_EV_EST_FAIL,
    FSM_EV_REFUSE_VBAT, FSM_EV_REFUSE_TILT, FSM_EV_REFUSE_EST, FSM_EV_REFUSE_STATE,
    FSM_EV_DISARM_LINK, FSM_EV_REFUSE_LINK,
    FSM_EV_LAND_SPIN,                 /* Stage 34：追加在末尾，已有事件编号不变（黑匣子 / 网页 / 脚本按编号解析） */
    FSM_EV__N
} fsm_event_t;

static const char *const FSM_EV_NAME[FSM_EV__N] = {
    "none", "armed", "launch", "takeoff", "stable_hold",
    "land_auto", "land_cmd", "land_low_voltage", "land_recover_timeout", "land_takeoff_timeout", "land_flight_timeout",
    "land_link_lost",
    "landed",
    "disarm_cmd", "arm_timeout", "land_timeout_kill", "impact", "estimator_fail",
    "refuse_vbatt", "refuse_tilt", "refuse_estimator", "refuse_state",
    "disarm_link_lost", "refuse_link", "land_spin"};
static const char *const FSM_STATE_NAME[FSM__NST] = {"DISARMED", "ARMED", "TAKEOFF", "RECOVER", "HOLD", "LAND"};

typedef struct {
    float vbatt;        /* V（NaN = 读数无效） */
    float acc_norm_g;   /* 比力模长 / g */
    float r22;          /* 机体 z 轴在世界 z 上的分量（1 = 水平，−1 = 倒扣） */
    float w_norm;       /* |ω| rad/s */
    float wz;           /* 机体偏航角速度 rad/s（自转保护） */
    float gate_k;       /* 门控 Mahony 当前修正增益 */
    float z, vz;        /* 竖直估计 m、m/s */
    float thrust_frac;  /* 上一拍控制器总推力 / 重力 */
    int   est_ok;       /* IMU 读数正常、四元数有限、气压计有数据 */
    int   link_ok;      /* 遥控链路（手机心跳 / 串口心跳）在线 */
} fsm_in_t;

typedef struct {
    fsm_state_t st;
    float t_state;      /* 当前状态已持续 s */
    float t_motor;      /* 本次通电飞行累计 s */
    float t_ff, t_lowv, t_stable, t_land, t_spin;
    float z_target;     /* 给控制器的目标高度 */
    float z_goal;       /* 起飞目标 */
    float t_hold_done;  /* HOLD 实际时长（遥测） */
} fsm_t;

static inline void fsm_init(fsm_t *f)
{
    f->st = FSM_DISARMED; f->t_state = f->t_motor = f->t_ff = f->t_lowv = f->t_stable = f->t_land = f->t_spin = 0.0f;
    f->z_target = f->z_goal = 0.0f; f->t_hold_done = 0.0f;
}
static inline int fsm_motors_on(const fsm_t *f) { return f->st >= FSM_TAKEOFF; }

static inline fsm_event_t fsm_go(fsm_t *f, fsm_state_t s, fsm_event_t ev, const fsm_in_t *in)
{
    if (s == FSM_LAND && f->z_target > in->z) f->z_target = in->z;      /* 降落从当前高度开始，不先爬升 */
    if (s >= FSM_TAKEOFF && f->st < FSM_TAKEOFF) f->t_motor = 0.0f;
    f->st = s; f->t_state = 0.0f; f->t_ff = 0.0f; f->t_stable = 0.0f; f->t_land = 0.0f; f->t_spin = 0.0f;
    if (s < FSM_TAKEOFF) f->t_lowv = 0.0f;
    return ev;
}

/* 每 200 Hz 调用一次；返回本步发生的事件（无则 FSM_EV_NONE） */
static inline fsm_event_t fsm_step(fsm_t *f, fsm_cmd_t cmd, const fsm_in_t *in, float dt)
{
    f->t_state += dt;
    if (cmd == FSM_CMD_DISARM)                                          /* 上锁永远优先、任何状态生效 */
        return f->st == FSM_DISARMED ? FSM_EV_NONE : fsm_go(f, FSM_DISARMED, FSM_EV_DISARM_CMD, in);

    if (f->st == FSM_DISARMED) {
        f->z_target = in->z;
        if (cmd == FSM_CMD_ARM) {
            if (!in->est_ok || in->gate_k < FSM_ARM_K_MIN) return FSM_EV_REFUSE_EST;
            if (!(in->vbatt >= FSM_ARM_VMIN)) return FSM_EV_REFUSE_VBAT;      /* NaN 也拒绝 */
            if (in->r22 < FSM_ARM_R22_MIN) return FSM_EV_REFUSE_TILT;
            if (!in->link_ok) return FSM_EV_REFUSE_LINK;
            return fsm_go(f, FSM_ARMED, FSM_EV_ARMED, in);
        }
        return (cmd == FSM_CMD_TAKEOFF || cmd == FSM_CMD_LAND) ? FSM_EV_REFUSE_STATE : FSM_EV_NONE;
    }

    if (f->st == FSM_ARMED) {
        f->z_target = in->z;
        if (!in->est_ok) return fsm_go(f, FSM_DISARMED, FSM_EV_EST_FAIL, in);
        if (!in->link_ok) return fsm_go(f, FSM_DISARMED, FSM_EV_DISARM_LINK, in);
        if (cmd == FSM_CMD_TAKEOFF) { f->z_goal = in->z + FSM_TAKEOFF_H; return fsm_go(f, FSM_TAKEOFF, FSM_EV_TAKEOFF, in); }
        if (cmd == FSM_CMD_ARM || cmd == FSM_CMD_LAND) return FSM_EV_REFUSE_STATE;
        f->t_ff = (in->acc_norm_g < FSM_FREEFALL_G) ? f->t_ff + dt : 0.0f;
        if (f->t_ff >= FSM_FREEFALL_T) return fsm_go(f, FSM_RECOVER, FSM_EV_LAUNCH, in);   /* 目标 = 出手高度 */
        if (f->t_state >= FSM_ARM_TIMEOUT) return fsm_go(f, FSM_DISARMED, FSM_EV_ARM_TIMEOUT, in);
        return FSM_EV_NONE;
    }

    /* ---------------- 通电飞行：TAKEOFF / RECOVER / HOLD / LAND ---------------- */
    f->t_motor += dt;
    if (!in->est_ok) return fsm_go(f, FSM_DISARMED, FSM_EV_EST_FAIL, in);
    if (f->t_motor >= FSM_IMPACT_ARM_T && in->acc_norm_g > FSM_IMPACT_G) return fsm_go(f, FSM_DISARMED, FSM_EV_IMPACT, in);
    if (cmd == FSM_CMD_ARM || cmd == FSM_CMD_TAKEOFF) return FSM_EV_REFUSE_STATE;

    if (f->st != FSM_LAND) {
        if (cmd == FSM_CMD_LAND) return fsm_go(f, FSM_LAND, FSM_EV_LAND_CMD, in);
        if (!in->link_ok) return fsm_go(f, FSM_LAND, FSM_EV_LAND_LINK, in);
        f->t_lowv = (in->vbatt < FSM_LOWV) ? f->t_lowv + dt : 0.0f;           /* NaN 不计入 */
        if (f->t_lowv >= FSM_LOWV_T) return fsm_go(f, FSM_LAND, FSM_EV_LAND_LOWV, in);
        if (f->t_motor >= FSM_ACTIVE_TMAX) return fsm_go(f, FSM_LAND, FSM_EV_LAND_ACTIVE_TO, in);
        /* 自转保护：只管起飞 / 悬停（回正时的快速翻转在 x/y 轴，抛出时的初始偏航转速由回正处理） */
        f->t_spin = ((f->st == FSM_TAKEOFF || f->st == FSM_HOLD) && fabsf(in->wz) > FSM_SPIN_W) ? f->t_spin + dt : 0.0f;
        if (f->t_spin >= FSM_SPIN_T) return fsm_go(f, FSM_LAND, FSM_EV_LAND_SPIN, in);
    }
    const int stable = in->r22 > FSM_STABLE_R22 && in->w_norm < FSM_STABLE_W && fabsf(in->vz) < FSM_STABLE_VZ;
    f->t_stable = stable ? f->t_stable + dt : 0.0f;

    switch (f->st) {
    case FSM_TAKEOFF:
        f->z_target += FSM_TAKEOFF_V * dt;
        if (f->z_target > f->z_goal) f->z_target = f->z_goal;
        if (f->z_target >= f->z_goal && fabsf(in->z - f->z_goal) < FSM_TAKEOFF_ZTOL && f->t_stable >= FSM_STABLE_T)
            return fsm_go(f, FSM_HOLD, FSM_EV_STABLE, in);
        if (f->t_state >= FSM_TAKEOFF_TMAX) return fsm_go(f, FSM_LAND, FSM_EV_LAND_TAKEOFF_TO, in);
        return FSM_EV_NONE;
    case FSM_RECOVER:
        if (f->t_stable >= FSM_STABLE_T) return fsm_go(f, FSM_HOLD, FSM_EV_STABLE, in);
        if (f->t_state >= FSM_RECOVER_TMAX) return fsm_go(f, FSM_LAND, FSM_EV_LAND_RECOVER_TO, in);
        return FSM_EV_NONE;
    case FSM_HOLD:
        f->t_hold_done = f->t_state;
        if (f->t_state >= FSM_HOLD_S) return fsm_go(f, FSM_LAND, FSM_EV_LAND_AUTO, in);
        return FSM_EV_NONE;
    case FSM_LAND: {
        f->z_target -= FSM_LAND_V * dt;
        if (f->z_target < in->z - FSM_LAND_LEAD_MAX) f->z_target = in->z - FSM_LAND_LEAD_MAX;
        const int touch = f->t_state >= FSM_LAND_MIN_T && in->thrust_frac < FSM_LAND_THRUST && fabsf(in->vz) < FSM_LAND_VZ;
        f->t_land = touch ? f->t_land + dt : 0.0f;
        if (f->t_land >= FSM_LAND_T) return fsm_go(f, FSM_DISARMED, FSM_EV_LANDED, in);
        if (f->t_state >= FSM_LAND_TMAX) return fsm_go(f, FSM_DISARMED, FSM_EV_LAND_TIMEOUT, in);
        return FSM_EV_NONE;
    }
    default:
        return FSM_EV_NONE;
    }
}

#endif /* FLIGHT_FSM_H */
