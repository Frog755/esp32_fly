/* host_test.c —— flight_v9 纯 C 部分的主机单测（gcc，不依赖 ESP-IDF）
 *   1. 控制器 + 电机映射方向：右侧朝下 / 机头朝下 / 逆时针自转 / 倒扣 → 期望哪几个电机加速（dryrun_check.py 的判据来源）
 *   2. 飞行状态机：解锁、抛出 / 起飞 → 回正 → 悬停 5 s → 降落 → 触地停桨，及全部保护
 *   3. 竖直互补滤波：悬停 / 爬升 / 自由落体 / 加计零偏下的 z、vz 误差
 * 编译运行：./run_host_test.sh
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "recovery_expert9.h"
#include "flight_fsm.h"
#include "alt_est.h"
#include "motor_out.h"
#include "flight_core.h"

static int n_pass, n_fail;
#define CHECK(cond, ...) do { if (cond) { n_pass++; printf("  ✓ "); } else { n_fail++; printf("  ✗ "); } printf(__VA_ARGS__); printf("\n"); } while (0)

/* ---------------- 1. 控制器方向 ---------------- */
typedef struct { float m[4]; uint8_t mode; } mot_t;    /* m[0..3] = M1..M4 */

static mot_t run_ctrl(const float q[4], const float w[3], int steps)
{
    recovery9_state_t st; recovery9_reset(&st);
    float pos[3] = {0, 0, 1.0f}, vel[3] = {0, 0, 0}, tgt[3] = {0, 0, 1.0f}, u[4];
    uint8_t mode = 0;
    for (int k = 0; k < steps; k++) mode = recovery_expert9_step(&st, q, w, pos, vel, 0.0f, 0.0f, tgt, 4.10f, u);
    mot_t r; r.mode = mode;
    for (int i = 0; i < 4; i++) r.m[i] = u[i];          /* MOT_NAME_SIM[i] = M(i+1)，见 motor_out.h */
    return r;
}
static float roll_diff(mot_t r)  { return (r.m[0] + r.m[1]) - (r.m[2] + r.m[3]); }   /* 右(M1,M2) − 左(M3,M4) */
static float pitch_diff(mot_t r) { return (r.m[0] + r.m[3]) - (r.m[1] + r.m[2]); }   /* 前(M1,M4) − 后(M2,M3) */
static float yaw_diff(mot_t r)   { return (r.m[0] + r.m[2]) - (r.m[1] + r.m[3]); }   /* 逆时针桨(M1,M3) − 顺时针桨(M2,M4) */
static void pr(const char *name, mot_t r)
{
    printf("    %-22s M1 %.3f  M2 %.3f  M3 %.3f  M4 %.3f | 右−左 %+.3f 前−后 %+.3f 逆桨−顺桨 %+.3f 模式 %d\n", name,
           r.m[0], r.m[1], r.m[2], r.m[3], roll_diff(r), pitch_diff(r), yaw_diff(r), r.mode);
}
static void q_axis(float q[4], float ax, float ay, float az, float deg)
{
    float h = 0.5f * deg * 3.14159265f / 180.0f, s = sinf(h);
    q[0] = cosf(h); q[1] = ax * s; q[2] = ay * s; q[3] = az * s;
}

static void test_controller(void)
{
    printf("[1] 控制器 + 电机映射方向（机体 FLU：x 机头 / y 左 / z 上；roll+ = 右侧朝下，pitch+ = 机头朝下）\n");
    printf("    映射：");
    for (int i = 0; i < 4; i++) printf("out_u[%d]→%s %s GPIO%d  ", i, MOT_NAME_SIM[i], MOT_POS_SIM[i], MOT_PIN_SIM[i]);
    printf("\n");
    float q[4], w0[3] = {0, 0, 0};
    q_axis(q, 1, 0, 0, 0);   mot_t lv = run_ctrl(q, w0, 40); pr("水平静止", lv);
    CHECK(fabsf(roll_diff(lv)) < 0.01f && fabsf(pitch_diff(lv)) < 0.01f && fabsf(yaw_diff(lv)) < 0.01f, "水平静止：四个电机相同");
    q_axis(q, 1, 0, 0, 30);  mot_t rr = run_ctrl(q, w0, 40); pr("右侧朝下 30°", rr);
    CHECK(roll_diff(rr) > 0.05f, "右侧朝下 → 右侧 M1+M2 加速（把右侧抬起来）");
    q_axis(q, 0, 1, 0, 30);  mot_t pp = run_ctrl(q, w0, 40); pr("机头朝下 30°", pp);
    CHECK(pitch_diff(pp) > 0.05f, "机头朝下 → 前方 M1+M4 加速（把机头抬起来）");
    q_axis(q, 1, 0, 0, -30); mot_t rl = run_ctrl(q, w0, 40); pr("左侧朝下 30°", rl);
    CHECK(roll_diff(rl) < -0.05f, "左侧朝下 → 左侧 M3+M4 加速");
    q_axis(q, 1, 0, 0, 0);
    float wccw[3] = {0, 0, +2.0f}; mot_t yc = run_ctrl(q, wccw, 40); pr("俯视逆时针自转 2 rad/s", yc);
    CHECK(yaw_diff(yc) > 0.02f, "逆时针自转 → 逆时针桨 M1+M3 加速（反扭矩为顺时针，制动）");
    float wcw[3] = {0, 0, -2.0f}; mot_t yw = run_ctrl(q, wcw, 40); pr("俯视顺时针自转 2 rad/s", yw);
    CHECK(yaw_diff(yw) < -0.02f, "顺时针自转 → 顺时针桨 M2+M4 加速");
    q_axis(q, 1, 0, 0, 180); mot_t inv = run_ctrl(q, w0, 1); pr("倒扣（绕 x 180°）", inv);
    CHECK(inv.mode == 1, "倒扣 → bang-bang 翻转模式");
    q_axis(q, 1, 0, 0, 170); mot_t inv2 = run_ctrl(q, w0, 1); pr("右侧朝下 170°", inv2);
    CHECK(inv2.mode == 1 && roll_diff(inv2) > 0.05f, "右侧朝下 170° → 翻转模式，且右侧加速（向左翻回）");
}

/* ---------------- 2. 飞行状态机 ---------------- */
static fsm_in_t ok_in(void)
{
    fsm_in_t in = {.vbatt = 4.1f, .acc_norm_g = 1.0f, .r22 = 1.0f, .w_norm = 0.0f, .gate_k = 2.0f, .z = 1.0f, .vz = 0.0f,
                   .thrust_frac = 1.0f, .est_ok = 1, .link_ok = 1};
    return in;
}
static fsm_event_t run_for(fsm_t *f, fsm_in_t in, float sec, fsm_event_t *first)
{
    fsm_event_t last = FSM_EV_NONE; *first = FSM_EV_NONE;
    int n = (int)(sec / 0.005f + 0.5f);
    for (int i = 0; i < n; i++) { fsm_event_t e = fsm_step(f, FSM_CMD_NONE, &in, 0.005f); if (e && !*first) *first = e; if (e) last = e; if (f->st == FSM_DISARMED) break; }
    return last;
}
static void armed(fsm_t *f) { fsm_in_t in = ok_in(); fsm_init(f); fsm_step(f, FSM_CMD_ARM, &in, 0.005f); }
static void launched(fsm_t *f) { fsm_event_t e; fsm_in_t in = ok_in(); armed(f); in.acc_norm_g = 0.1f; run_for(f, in, 0.05f, &e); }
static void holding(fsm_t *f) { fsm_event_t e; launched(f); run_for(f, ok_in(), 0.3f, &e); }

static void test_fsm(void)
{
    printf("[2] 飞行状态机\n");
    fsm_t f; fsm_in_t in; fsm_event_t e, first;
    fsm_init(&f); in = ok_in(); in.vbatt = 3.5f; e = fsm_step(&f, FSM_CMD_ARM, &in, 0.005f);
    CHECK(e == FSM_EV_REFUSE_VBAT && f.st == FSM_DISARMED, "电压 3.5 V 拒绝解锁");
    in.vbatt = NAN; e = fsm_step(&f, FSM_CMD_ARM, &in, 0.005f);
    CHECK(e == FSM_EV_REFUSE_VBAT, "电压读数 NaN 拒绝解锁");
    in = ok_in(); in.r22 = 0.9f; e = fsm_step(&f, FSM_CMD_ARM, &in, 0.005f);
    CHECK(e == FSM_EV_REFUSE_TILT, "倾斜 26° 拒绝解锁");
    in = ok_in(); in.gate_k = 0.2f; e = fsm_step(&f, FSM_CMD_ARM, &in, 0.005f);
    CHECK(e == FSM_EV_REFUSE_EST, "在动（门控增益低）拒绝解锁");
    in = ok_in(); e = fsm_step(&f, FSM_CMD_TAKEOFF, &in, 0.005f);
    CHECK(e == FSM_EV_REFUSE_STATE && f.st == FSM_DISARMED, "未解锁时 takeoff 被拒绝");
    e = fsm_step(&f, FSM_CMD_ARM, &in, 0.005f);
    CHECK(e == FSM_EV_ARMED && f.st == FSM_ARMED && !fsm_motors_on(&f), "条件合格 → 解锁（ARMED 电机不转）");
    run_for(&f, in, 19.9f, &first);
    CHECK(f.st == FSM_ARMED, "解锁 19.9 s 仍在等待");
    run_for(&f, in, 0.2f, &first);
    CHECK(f.st == FSM_DISARMED && first == FSM_EV_ARM_TIMEOUT, "解锁 20 s 未出手 → 自动上锁");

    /* 抛出 → 回正 → 悬停 5 s → 降落 → 触地 */
    armed(&f); in = ok_in(); in.z = 1.5f; in.acc_norm_g = 0.2f; run_for(&f, in, 0.030f, &first);
    CHECK(f.st == FSM_ARMED, "失重 30 ms 不触发（防误判）");
    in.acc_norm_g = 1.0f; run_for(&f, in, 0.01f, &first);
    in.acc_norm_g = 0.2f; run_for(&f, in, 0.030f, &first);
    CHECK(f.st == FSM_ARMED, "失重计时被打断后重新累计");
    run_for(&f, in, 0.015f, &first);
    CHECK(f.st == FSM_RECOVER && first == FSM_EV_LAUNCH && fsm_motors_on(&f), "连续失重 ≥40 ms → RECOVER（电机启动）");
    CHECK(fabsf(f.z_target - 1.5f) < 1e-4f, "回正目标高度 = 出手高度 1.5 m");
    in = ok_in(); in.z = 1.0f; in.r22 = 0.5f; in.w_norm = 8.0f; in.acc_norm_g = 8.0f; run_for(&f, in, 0.1f, &first);
    CHECK(f.st == FSM_RECOVER, "出手后 0.3 s 内的大加速度不算撞击");
    in = ok_in(); in.z = 1.0f; in.vz = 0.6f; run_for(&f, in, 0.5f, &first);
    CHECK(f.st == FSM_RECOVER, "|vz| 0.6 m/s（还在爬回）不算稳定");
    in.vz = 0.2f; run_for(&f, in, 0.2f, &first);
    CHECK(f.st == FSM_RECOVER, "稳定 0.2 s 还不进 HOLD");
    run_for(&f, in, 0.1f, &first);
    CHECK(f.st == FSM_HOLD && first == FSM_EV_STABLE, "稳定 0.25 s → HOLD");
    in = ok_in(); in.r22 = 0.8f; run_for(&f, in, 4.8f, &first);
    CHECK(f.st == FSM_HOLD, "HOLD 4.8 s（中途姿态晃动也不提前降落）");
    run_for(&f, in, 0.25f, &first);
    CHECK(f.st == FSM_LAND && first == FSM_EV_LAND_AUTO, "悬停满 5 s → 自动降落");
    float zt0 = f.z_target; in = ok_in(); in.vz = -0.35f; run_for(&f, in, 1.0f, &first);
    CHECK(fabsf((zt0 - f.z_target) - 0.35f) < 0.01f, "降落时目标高度以 0.35 m/s 下降（1 s 降 %.3f m）", zt0 - f.z_target);
    in.thrust_frac = 0.3f; in.vz = 0.0f; run_for(&f, in, 0.25f, &first);
    CHECK(f.st == FSM_LAND, "低推力 + 不动 0.25 s 还不停桨");
    run_for(&f, in, 0.1f, &first);
    CHECK(f.st == FSM_DISARMED && first == FSM_EV_LANDED, "低推力 + 不动 0.3 s → 触地停桨");

    launched(&f); in = ok_in(); in.r22 = 0.3f; run_for(&f, in, 3.9f, &first);
    CHECK(f.st == FSM_RECOVER, "回正 3.9 s 未稳定仍在回正");
    run_for(&f, in, 0.2f, &first);
    CHECK(f.st == FSM_LAND && first == FSM_EV_LAND_RECOVER_TO, "回正 4 s 未稳定 → 降落（不是空中停桨）");
    fsm_go(&f, FSM_LAND, FSM_EV_NONE, &in); in = ok_in(); in.thrust_frac = 0.3f; run_for(&f, in, 0.9f, &first);
    CHECK(f.st == FSM_LAND, "进入降落 1 s 内不做触地检测");
    in = ok_in(); in.vz = -0.35f; run_for(&f, in, 18.0f, &first);
    CHECK(f.st == FSM_LAND, "一直在下降（正常推力）不会误判触地");
    run_for(&f, in, 3.0f, &first);
    CHECK(f.st == FSM_DISARMED && first == FSM_EV_LAND_TIMEOUT, "降落 20 s 仍未触地 → 停桨");

    /* 起飞 */
    armed(&f); in = ok_in(); in.z = 0.0f; e = fsm_step(&f, FSM_CMD_TAKEOFF, &in, 0.005f);
    CHECK(e == FSM_EV_TAKEOFF && f.st == FSM_TAKEOFF && fabsf(f.z_goal - 0.8f) < 1e-4f, "takeoff → TAKEOFF，目标 = 起飞点 +0.8 m");
    run_for(&f, in, 1.0f, &first);
    CHECK(fabsf(f.z_target - 0.5f) < 0.01f, "起飞目标以 0.5 m/s 上升（1 s 后 %.3f m）", f.z_target);
    in.z = 0.75f; run_for(&f, in, 0.7f, &first);
    CHECK(f.st == FSM_HOLD && first == FSM_EV_STABLE, "到达 +0.8 m 附近且稳定 → HOLD");
    armed(&f); in = ok_in(); in.z = 0.0f; fsm_step(&f, FSM_CMD_TAKEOFF, &in, 0.005f); run_for(&f, in, 6.1f, &first);
    CHECK(f.st == FSM_LAND && first == FSM_EV_LAND_TAKEOFF_TO, "起飞 6 s 未离地（台架干跑 / 推力不足）→ 降落");
    run_for(&f, in, 0.5f, &first); in.thrust_frac = 0.3f; run_for(&f, in, 1.0f, &first);
    CHECK(f.st == FSM_DISARMED && first == FSM_EV_LANDED, "随后触地检测停桨");

    /* 保护 */
    holding(&f); in = ok_in(); in.acc_norm_g = 7.0f; fsm_step(&f, FSM_CMD_NONE, &in, 0.005f);
    CHECK(f.st == FSM_DISARMED, "悬停中 7 g 撞击 → 停桨");
    holding(&f); in = ok_in(); in.vbatt = 2.9f; run_for(&f, in, 0.45f, &first);
    CHECK(f.st == FSM_HOLD, "低电压 0.45 s 不动作（电机负载压降）");
    run_for(&f, in, 0.1f, &first);
    CHECK(f.st == FSM_LAND && first == FSM_EV_LAND_LOWV, "低电压持续 0.5 s → 降落（不在空中停桨）");
    holding(&f); in = ok_in(); e = fsm_step(&f, FSM_CMD_LAND, &in, 0.005f);
    CHECK(e == FSM_EV_LAND_CMD && f.st == FSM_LAND, "land 指令 → 立即降落");
    e = fsm_step(&f, FSM_CMD_TAKEOFF, &in, 0.005f);
    CHECK(e == FSM_EV_REFUSE_STATE && f.st == FSM_LAND, "飞行中 takeoff / arm 被忽略");
    holding(&f); in = ok_in(); in.est_ok = 0; e = fsm_step(&f, FSM_CMD_NONE, &in, 0.005f);
    CHECK(e == FSM_EV_EST_FAIL && f.st == FSM_DISARMED, "飞行中估计器失效 → 立即停桨");
    /* 遥控链路 */
    fsm_init(&f); in = ok_in(); in.link_ok = 0; e = fsm_step(&f, FSM_CMD_ARM, &in, 0.005f);
    CHECK(e == FSM_EV_REFUSE_LINK && f.st == FSM_DISARMED, "没有遥控链路拒绝解锁");
    armed(&f); in = ok_in(); in.link_ok = 0; e = fsm_step(&f, FSM_CMD_NONE, &in, 0.005f);
    CHECK(e == FSM_EV_DISARM_LINK && f.st == FSM_DISARMED, "解锁后（还没出手）链路丢失 → 上锁");
    holding(&f); in = ok_in(); in.link_ok = 0; e = fsm_step(&f, FSM_CMD_NONE, &in, 0.005f);
    CHECK(e == FSM_EV_LAND_LINK && f.st == FSM_LAND, "悬停中链路丢失 → 降落（不在空中停桨）");
    run_for(&f, in, 2.0f, &first);
    CHECK(f.st == FSM_LAND, "降落中链路一直丢失 → 继续降落");
    in.thrust_frac = 0.3f; run_for(&f, in, 1.0f, &first);
    CHECK(f.st == FSM_DISARMED && first == FSM_EV_LANDED, "链路丢失也照常触地停桨");
    armed(&f); in = ok_in(); in.z = 0.0f; fsm_step(&f, FSM_CMD_TAKEOFF, &in, 0.005f); in.link_ok = 0;
    e = fsm_step(&f, FSM_CMD_NONE, &in, 0.005f);
    CHECK(e == FSM_EV_LAND_LINK && f.st == FSM_LAND, "起飞中链路丢失 → 降落");

    holding(&f); in = ok_in(); in.est_ok = 0; in.vbatt = NAN; in.acc_norm_g = 0.1f;
    e = fsm_step(&f, FSM_CMD_DISARM, &in, 0.005f);
    CHECK(e == FSM_EV_DISARM_CMD && f.st == FSM_DISARMED && !fsm_motors_on(&f), "disarm 指令在任何输入下都立即生效");
}

/* ---------------- 3. 竖直互补滤波 ---------------- */
static double gauss(void) { double u = (rand() + 1.0) / (RAND_MAX + 2.0), v = (rand() + 1.0) / (RAND_MAX + 2.0); return sqrt(-2 * log(u)) * cos(6.2831853 * v); }

static void test_alt(void)
{
    printf("[3] 竖直互补滤波（气压计 50 Hz、σ 10 cm；加计 1 kHz、σ 0.3 m/s²）\n");
    const char *names[4] = {"静止", "1 m/s 匀速爬升 3 s", "自由落体 0.6 s 后悬停", "静止 + 加计零偏 0.2 m/s²"};
    for (int c = 0; c < 4; c++) {
        srand(42 + c);
        alt_est_t e; alt_reset(&e, 0.0f);
        double z = 0, vz = 0, se_z = 0, se_v = 0, max_v = 0; int n = 0;
        float bias = (c == 3) ? 0.2f : 0.0f;
        for (int k = 0; k < 8000; k++) {
            double t = k * 0.001, a = 0;
            if (c == 1) { a = (t > 2 && t < 2.1) ? 10 : ((t > 5 && t < 5.1) ? -10 : 0); }
            if (c == 2) { a = (t > 2 && t < 2.6) ? -9.81 : ((t >= 2.6 && t < 3.188) ? 10 : 0); }
            vz += a * 0.001; z += vz * 0.001;
            alt_predict(&e, (float)(a + bias + 0.3 * gauss()), 0.001f);
            if (k % 20 == 0) alt_correct(&e, (float)(z + 0.10 * gauss()), 0.02f);
            if (t > 1.0) { double ez = e.z - z, ev = e.vz - vz; se_z += ez * ez; se_v += ev * ev; n++; if (fabs(ev) > max_v) max_v = fabs(ev); }
        }
        double rz = sqrt(se_z / n), rv = sqrt(se_v / n);
        printf("    %-24s z 误差 RMS %.3f m   vz 误差 RMS %.3f m/s（最大 %.3f）  零偏估计 %+.3f m/s²\n", names[c], rz, rv, max_v, e.ab);
        CHECK(rz < 0.10 && rv < 0.10, "%s：z RMS < 0.10 m 且 vz RMS < 0.10 m/s", names[c]);
        if (c == 3) CHECK(fabsf(e.ab - bias) < 0.05f, "零偏估计收敛到 %.2f ± 0.05 m/s²", bias);
    }
}

/* ---------------- 4. flight_core 心跳超时（整链：估计 + 状态机） ---------------- */
static fc_t s_c; static int s_k; static uint32_t s_seq; static uint8_t s_ev;
static void core_run(float sec, int beat)
{
    static const float a0[3] = {0, 0, 9.81f}, w0[3] = {0, 0, 0};
    int n = (int)(sec * 1000.0f + 0.5f);
    for (int i = 0; i < n; i++, s_k++) {
        if (beat && s_k % 100 == 0) fc_link_beat(&s_c);
        fc_step(&s_c, a0, w0, 0.001f, 1, s_k % 20 == 0, 0.0f, 0.02f, 4.1f);
        if (s_c.ev_seq != s_seq) { s_seq = s_c.ev_seq; s_ev = s_c.ev_last; }
    }
}
static void test_core_link(void)
{
    printf("[4] flight_core 遥控心跳超时（静止 IMU 输入，手机每 0.1 s 一次心跳）\n");
    const float a0[3] = {0, 0, 9.81f};
    fc_init(&s_c, a0); s_k = 0; s_seq = 0; s_ev = 0;
    core_run(0.5f, 1); fc_command(&s_c, FSM_CMD_ARM); core_run(0.1f, 1);
    CHECK(s_c.fsm.st == FSM_ARMED, "有心跳 → 解锁成功（状态 %s，事件 %s）", FSM_STATE_NAME[s_c.fsm.st], FSM_EV_NAME[s_ev]);
    fc_command(&s_c, FSM_CMD_TAKEOFF); core_run(0.5f, 1);
    CHECK(s_c.fsm.st == FSM_TAKEOFF, "takeoff → TAKEOFF（状态 %s）", FSM_STATE_NAME[s_c.fsm.st]);
    core_run(0.45f, 0);                                    /* 上一次心跳在 0.1 s 前 → 心跳间隔 0.55 s */
    CHECK(s_c.fsm.st == FSM_TAKEOFF, "心跳间隔 0.55 s 仍在起飞（偶尔丢几个包不误判）（状态 %s）", FSM_STATE_NAME[s_c.fsm.st]);
    core_run(0.1f, 0);
    CHECK(s_c.fsm.st == FSM_LAND && s_ev == FSM_EV_LAND_LINK, "心跳间隔超过 0.6 s → 链路丢失 → 降落（状态 %s，事件 %s）", FSM_STATE_NAME[s_c.fsm.st], FSM_EV_NAME[s_ev]);
    fc_init(&s_c, a0); s_seq = 0; s_ev = 0;
    core_run(0.5f, 0); fc_command(&s_c, FSM_CMD_ARM); core_run(0.1f, 0);
    CHECK(s_c.fsm.st == FSM_DISARMED && s_ev == FSM_EV_REFUSE_LINK, "上电后从没收到心跳 → 拒绝解锁（事件 %s）", FSM_EV_NAME[s_ev]);
}

/* ---------------- 5. 推力换算：静止电压保持 + 空中推力在线修正（Stage 31） ---------------- */
static void core_run2(float sec, float az_g, float vb, float zb)
{
    const float a[3] = {0, 0, 9.81f * az_g}, w0[3] = {0, 0, 0};
    int n = (int)(sec * 1000.0f + 0.5f);
    for (int i = 0; i < n; i++, s_k++) {
        if (s_k % 100 == 0) fc_link_beat(&s_c);
        fc_step(&s_c, a, w0, 0.001f, 1, s_k % 20 == 0, zb, 0.02f, s_k % 100 == 0 ? vb : NAN);
        if (s_c.ev_seq != s_seq) { s_seq = s_c.ev_seq; s_ev = s_c.ev_last; }
    }
}
static float max_out(void) { float m = 0; for (int i = 0; i < 4; i++) m = s_c.u_out[i] > m ? s_c.u_out[i] : m; return m; }
static void test_core_thrust(void)
{
    printf("[5] 推力换算：静止电压保持、贴地不修正、空中在线修正（换算系数 %.3f）\n", (double)FC_DUTY_SCALE);
    const float a0[3] = {0, 0, 9.81f};
    fc_init(&s_c, a0); s_k = 0; s_seq = 0; s_ev = 0;
    core_run2(0.5f, 1.0f, 4.15f, 0.0f); fc_command(&s_c, FSM_CMD_ARM); core_run2(0.2f, 1.0f, 4.15f, 0.0f);
    fc_command(&s_c, FSM_CMD_TAKEOFF); core_run2(0.3f, 1.0f, 3.30f, 0.0f);     /* 一转读数就掉到 3.3 V */
    CHECK(s_c.fsm.st == FSM_TAKEOFF && fabsf(s_c.vb_ctrl - 4.15f) < 1e-3f, "通电后控制用电压保持通电前的 4.15 V（读数 %.2f V，控制用 %.2f V）",
          (double)s_c.vbatt, (double)s_c.vb_ctrl);
    float omax = 0; for (int i = 0; i < 1500; i++) { core_run2(0.001f, 1.0f, 3.30f, 0.0f); if (max_out() > omax) omax = max_out(); }
    CHECK(s_c.c_k == 1.0f && !s_c.adapting, "贴地起飞（高度不升、比力 1 g）不做在线修正（c_k %.3f）", (double)s_c.c_k);
    CHECK(omax <= FC_DUTY_SCALE * 1.0001f, "真实占空比不超过 换算系数 × 1（最大 %.3f）", (double)omax);
    core_run2(5.0f, 1.0f, 3.30f, 0.0f);
    CHECK(s_c.fsm.st == FSM_LAND || s_c.fsm.st == FSM_DISARMED, "贴地 6 s 没离地 → 起飞超时 → 降落（状态 %s）", FSM_STATE_NAME[s_c.fsm.st]);
    /* 空中闭环：真实推力 = s0 × c_k² × 控制器以为的推力（真机推力曲线与假设差一个常数 s0），竖直一维动力学 */
    const float s0s[3] = {1.25f, 0.80f, 3.0f};
    for (int t = 0; t < 3; t++) {
        const float s0 = s0s[t];
        fc_init(&s_c, a0); s_k = 0; s_seq = 0; s_ev = 0;
        core_run2(0.5f, 1.0f, 4.15f, 1.0f);
        s_c.fsm.st = FSM_HOLD; s_c.fsm.t_state = 0; s_c.fsm.z_target = 1.0f; s_c.fsm.z_goal = 1.0f;
        float z = 1.0f, vz = 0.0f, zmin = 1.0f, zmax = 1.0f;
        for (int i = 0; i < 6000; i++, s_k++) {
            if (s_k % 100 == 0) fc_link_beat(&s_c);
            if (i % 1000 == 0) s_c.fsm.t_state = 0.0f;                       /* 不让 5 s 自动降落打断测试 */
            float tr = 0; for (int j = 0; j < 4; j++) tr += s_c.u_out[j] * s_c.u_out[j];
            const float azg = s0 * tr / (4.0f * (FC_DUTY_SCALE * FC_MODEL_HOVER_U * EXP9_VREF / 4.15f) * (FC_DUTY_SCALE * FC_MODEL_HOVER_U * EXP9_VREF / 4.15f));
            vz += (azg - 1.0f) * 9.81f * 0.001f; z += vz * 0.001f;
            if (i > 3000) { zmin = z < zmin ? z : zmin; zmax = z > zmax ? z : zmax; }
            const float a[3] = {0, 0, 9.81f * azg}, w0[3] = {0, 0, 0};
            fc_step(&s_c, a, w0, 0.001f, 1, s_k % 20 == 0, z, 0.02f, s_k % 100 == 0 ? 4.15f : NAN);
        }
        const float want = 1.0f / sqrtf(s0), wc = want < FC_ADAPT_MIN ? FC_ADAPT_MIN : (want > FC_ADAPT_MAX ? FC_ADAPT_MAX : want);
        if (t < 2) CHECK(fabsf(s_c.c_k - wc) < 0.03f && zmax - zmin < 0.3f, "空中真实推力 = 以为的 %.2f 倍 → 6 s 后 c_k %.3f（应 %.3f），后 3 s 高度波动 %.2f m",
                         (double)s0, (double)s_c.c_k, (double)wc, (double)(zmax - zmin));
        else       CHECK(fabsf(s_c.c_k - FC_ADAPT_MIN) < 1e-4f, "推力大 3 倍（超出修正范围）→ c_k 停在下限 %.2f（%.3f）", (double)FC_ADAPT_MIN, (double)s_c.c_k);
    }
}

int main(void)
{
    test_controller();
    test_fsm();
    test_alt();
    test_core_link();
    test_core_thrust();
    printf("\n通过 %d / %d\n", n_pass, n_pass + n_fail);
    return n_fail ? 1 : 0;
}
