/**
 * flight_safety.h —— 飞控安全状态机（200 Hz 调用；纯 C，无 ESP 依赖，主机可单测）
 *
 *   DISARMED ──arm（电压/水平/估计器均合格）──▶ ARMED ──抛出检测到失重 或 go 指令──▶ ACTIVE
 *      ▲                                      │ 20 s 未抛出                         │
 *      └──────────────── disarm 指令（任何状态立即生效）/ 超时 / 低电压 / 撞击 / 估计器失效 ◀┘
 *
 * 只有 ACTIVE 时电机才有输出（flight_main.c 负责）；DISARMED / ARMED 输出恒为 0。
 * 本版（Stage 24）固件编译时不含电机驱动，状态机照常运行，用于在不转电机的前提下验证逻辑。
 */
#ifndef FLIGHT_SAFETY_H
#define FLIGHT_SAFETY_H

#include <stdint.h>

#define FS_ARM_VMIN        3.60f   /* V：解锁最低电压（耀坤作者实测"静态 >3.6 V 才能起飞"） */
#define FS_ARM_R22_MIN     0.94f   /* 解锁时机体 z 轴与竖直夹角 < 20° */
#define FS_ARM_K_MIN       1.0f    /* 解锁时门控 Mahony 增益（=加计在修正 = 静止） */
#define FS_ARM_TIMEOUT     20.0f   /* s：解锁后未抛出自动上锁 */
#define FS_FREEFALL_G      0.35f   /* 比力模长 < 0.35 g 视为失重（已出手） */
#define FS_FREEFALL_T      0.040f  /* s：持续时间 */
#define FS_ACTIVE_MAX      8.0f    /* s：ACTIVE 最长时间（控制器无降落逻辑，先保守） */
#define FS_LOWV            3.00f   /* V：ACTIVE 中低于此电压…… */
#define FS_LOWV_T          0.5f    /* s：……持续 0.5 s 停桨 */
#define FS_IMPACT_G        6.0f    /* 比力模长 > 6 g 视为撞击 */
#define FS_IMPACT_ARM_T    0.3f    /* s：进入 ACTIVE 0.3 s 后才启用撞击判据 */

typedef enum { FS_DISARMED = 0, FS_ARMED = 1, FS_ACTIVE = 2 } fs_state_t;
typedef enum { FS_CMD_NONE = 0, FS_CMD_ARM, FS_CMD_DISARM, FS_CMD_GO } fs_cmd_t;
typedef enum {
    FS_EV_NONE = 0,
    FS_EV_ARMED,            /* DISARMED → ARMED */
    FS_EV_LAUNCH,           /* ARMED → ACTIVE（失重） */
    FS_EV_GO,               /* ARMED → ACTIVE（指令） */
    FS_EV_DISARM_CMD,
    FS_EV_ARM_TIMEOUT,
    FS_EV_ACTIVE_TIMEOUT,
    FS_EV_LOWV,
    FS_EV_IMPACT,
    FS_EV_EST_FAIL,
    FS_EV_REFUSE_VBAT,      /* 拒绝解锁：电压 */
    FS_EV_REFUSE_TILT,      /* 拒绝解锁：不水平 */
    FS_EV_REFUSE_EST,       /* 拒绝解锁：估计器未就绪 / 在动 */
    FS_EV_REFUSE_STATE,     /* 指令与当前状态不符（如未解锁就 go） */
    FS_EV__N
} fs_event_t;

static const char *const FS_EV_NAME[FS_EV__N] = {
    "none", "armed", "launch", "go", "disarm_cmd", "arm_timeout", "active_timeout", "low_voltage",
    "impact", "estimator_fail", "refuse_vbatt", "refuse_tilt", "refuse_estimator", "refuse_state"};
static const char *const FS_STATE_NAME[3] = {"DISARMED", "ARMED", "ACTIVE"};

typedef struct {
    float vbatt;        /* V */
    float acc_norm_g;   /* 比力模长 / g */
    float r22;          /* 机体 z 轴在世界 z 上的分量（1 = 水平，−1 = 倒扣） */
    float gate_k;       /* 门控 Mahony 当前修正增益 */
    int   est_ok;       /* IMU 读数正常、四元数有限 */
} fs_in_t;

typedef struct {
    fs_state_t st;
    float t_state;      /* 当前状态已持续时间 s */
    float t_ff;         /* 失重累计 s */
    float t_lowv;       /* 低电压累计 s */
} fs_t;

static inline void fs_init(fs_t *f) { f->st = FS_DISARMED; f->t_state = 0; f->t_ff = 0; f->t_lowv = 0; }

static inline fs_event_t fs_go_state(fs_t *f, fs_state_t s, fs_event_t ev)
{
    f->st = s; f->t_state = 0; f->t_ff = 0; f->t_lowv = 0;
    return ev;
}

/* 每 200 Hz 调用一次；返回本步发生的事件（无则 FS_EV_NONE） */
static inline fs_event_t fs_step(fs_t *f, fs_cmd_t cmd, const fs_in_t *in, float dt)
{
    f->t_state += dt;
    if (cmd == FS_CMD_DISARM)                                  /* 上锁永远优先、任何状态生效 */
        return f->st == FS_DISARMED ? FS_EV_NONE : fs_go_state(f, FS_DISARMED, FS_EV_DISARM_CMD);

    switch (f->st) {
    case FS_DISARMED:
        if (cmd == FS_CMD_ARM) {
            if (!in->est_ok || in->gate_k < FS_ARM_K_MIN) return FS_EV_REFUSE_EST;
            if (!(in->vbatt >= FS_ARM_VMIN)) return FS_EV_REFUSE_VBAT;          /* NaN 也拒绝 */
            if (in->r22 < FS_ARM_R22_MIN) return FS_EV_REFUSE_TILT;
            return fs_go_state(f, FS_ARMED, FS_EV_ARMED);
        }
        if (cmd == FS_CMD_GO) return FS_EV_REFUSE_STATE;
        return FS_EV_NONE;

    case FS_ARMED:
        if (!in->est_ok) return fs_go_state(f, FS_DISARMED, FS_EV_EST_FAIL);
        if (cmd == FS_CMD_GO) return fs_go_state(f, FS_ACTIVE, FS_EV_GO);
        if (cmd == FS_CMD_ARM) return FS_EV_REFUSE_STATE;
        f->t_ff = (in->acc_norm_g < FS_FREEFALL_G) ? f->t_ff + dt : 0.0f;
        if (f->t_ff >= FS_FREEFALL_T) return fs_go_state(f, FS_ACTIVE, FS_EV_LAUNCH);
        if (f->t_state >= FS_ARM_TIMEOUT) return fs_go_state(f, FS_DISARMED, FS_EV_ARM_TIMEOUT);
        return FS_EV_NONE;

    case FS_ACTIVE:
        if (cmd == FS_CMD_ARM || cmd == FS_CMD_GO) return FS_EV_REFUSE_STATE;
        if (!in->est_ok) return fs_go_state(f, FS_DISARMED, FS_EV_EST_FAIL);
        if (f->t_state >= FS_IMPACT_ARM_T && in->acc_norm_g > FS_IMPACT_G) return fs_go_state(f, FS_DISARMED, FS_EV_IMPACT);
        f->t_lowv = (in->vbatt < FS_LOWV) ? f->t_lowv + dt : 0.0f;               /* NaN 不计入 */
        if (f->t_lowv >= FS_LOWV_T) return fs_go_state(f, FS_DISARMED, FS_EV_LOWV);
        if (f->t_state >= FS_ACTIVE_MAX) return fs_go_state(f, FS_DISARMED, FS_EV_ACTIVE_TIMEOUT);
        return FS_EV_NONE;
    }
    return FS_EV_NONE;
}

#endif /* FLIGHT_SAFETY_H */
