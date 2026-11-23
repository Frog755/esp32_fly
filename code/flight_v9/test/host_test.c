/* host_test.c —— flight_v9 纯 C 部分的主机单测（gcc，不依赖 ESP-IDF）
 *   1. 控制器 + 电机映射方向：右侧朝下 / 机头朝下 / 逆时针自转 / 倒扣 → 期望哪几个电机加速（dryrun_check.py 的判据来源）
 *   2. 安全状态机：解锁条件、超时、失重抛出、低电压、撞击、估计器失效、上锁优先
 *   3. 竖直互补滤波：悬停 / 爬升 / 自由落体 / 加计零偏下的 z、vz 误差
 * 编译运行：./run_host_test.sh
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "recovery_expert9.h"
#include "flight_safety.h"
#include "alt_est.h"
#include "motor_out.h"

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

/* ---------------- 2. 安全状态机 ---------------- */
static fs_in_t ok_in(void) { fs_in_t in = {.vbatt = 4.1f, .acc_norm_g = 1.0f, .r22 = 1.0f, .gate_k = 2.0f, .est_ok = 1}; return in; }
static fs_event_t run_for(fs_t *f, fs_in_t in, float sec, fs_event_t *first)
{
    fs_event_t last = FS_EV_NONE; *first = FS_EV_NONE;
    int n = (int)(sec / 0.005f + 0.5f);
    for (int i = 0; i < n; i++) { fs_event_t e = fs_step(f, FS_CMD_NONE, &in, 0.005f); if (e && !*first) *first = e; if (e) last = e; }
    return last;
}
static void test_safety(void)
{
    printf("[2] 安全状态机\n");
    fs_t f; fs_in_t in; fs_event_t e, first;
    fs_init(&f); in = ok_in(); in.vbatt = 3.5f;   e = fs_step(&f, FS_CMD_ARM, &in, 0.005f);
    CHECK(e == FS_EV_REFUSE_VBAT && f.st == FS_DISARMED, "电压 3.5 V 拒绝解锁");
    in.vbatt = NAN; e = fs_step(&f, FS_CMD_ARM, &in, 0.005f);
    CHECK(e == FS_EV_REFUSE_VBAT, "电压读数 NaN 拒绝解锁");
    in = ok_in(); in.r22 = 0.9f;                  e = fs_step(&f, FS_CMD_ARM, &in, 0.005f);
    CHECK(e == FS_EV_REFUSE_TILT, "倾斜 26° 拒绝解锁");
    in = ok_in(); in.gate_k = 0.2f;               e = fs_step(&f, FS_CMD_ARM, &in, 0.005f);
    CHECK(e == FS_EV_REFUSE_EST, "在动（门控增益低）拒绝解锁");
    in = ok_in();                                 e = fs_step(&f, FS_CMD_GO, &in, 0.005f);
    CHECK(e == FS_EV_REFUSE_STATE && f.st == FS_DISARMED, "未解锁时 go 被拒绝");
    e = fs_step(&f, FS_CMD_ARM, &in, 0.005f);
    CHECK(e == FS_EV_ARMED && f.st == FS_ARMED, "条件合格 → 解锁");
    run_for(&f, in, 19.9f, &first);
    CHECK(f.st == FS_ARMED, "解锁 19.9 s 仍在等待抛出");
    run_for(&f, in, 0.2f, &first);
    CHECK(f.st == FS_DISARMED && first == FS_EV_ARM_TIMEOUT, "解锁 20 s 未抛出 → 自动上锁");

    fs_init(&f); in = ok_in(); fs_step(&f, FS_CMD_ARM, &in, 0.005f);
    in.acc_norm_g = 0.2f; run_for(&f, in, 0.030f, &first);
    CHECK(f.st == FS_ARMED, "失重 30 ms 不触发（防误判）");
    in.acc_norm_g = 1.0f; run_for(&f, in, 0.01f, &first);
    in.acc_norm_g = 0.2f; run_for(&f, in, 0.030f, &first);
    CHECK(f.st == FS_ARMED, "失重计时被打断后重新累计");
    run_for(&f, in, 0.015f, &first);
    CHECK(f.st == FS_ACTIVE && first == FS_EV_LAUNCH, "连续失重 ≥40 ms → ACTIVE（抛出）");
    in = ok_in(); in.acc_norm_g = 8.0f; run_for(&f, in, 0.1f, &first);
    CHECK(f.st == FS_ACTIVE, "进入 ACTIVE 0.3 s 内的大加速度不算撞击（出手瞬间）");
    in = ok_in(); run_for(&f, in, 0.3f, &first); in.acc_norm_g = 7.0f; fs_step(&f, FS_CMD_NONE, &in, 0.005f);
    CHECK(f.st == FS_DISARMED, "ACTIVE 中 7 g 撞击 → 停桨");

    fs_init(&f); in = ok_in(); fs_step(&f, FS_CMD_ARM, &in, 0.005f); fs_step(&f, FS_CMD_GO, &in, 0.005f);
    CHECK(f.st == FS_ACTIVE, "go 指令 → ACTIVE");
    e = fs_step(&f, FS_CMD_ARM, &in, 0.005f);
    CHECK(e == FS_EV_REFUSE_STATE && f.st == FS_ACTIVE, "ACTIVE 中 arm 被忽略");
    in.vbatt = 2.9f; run_for(&f, in, 0.45f, &first);
    CHECK(f.st == FS_ACTIVE, "低电压 0.45 s 不停桨（电机启动压降）");
    run_for(&f, in, 0.1f, &first);
    CHECK(f.st == FS_DISARMED && first == FS_EV_LOWV, "低电压持续 0.5 s → 停桨");

    fs_init(&f); in = ok_in(); fs_step(&f, FS_CMD_ARM, &in, 0.005f); fs_step(&f, FS_CMD_GO, &in, 0.005f);
    run_for(&f, in, 7.9f, &first);
    CHECK(f.st == FS_ACTIVE, "ACTIVE 7.9 s 仍在运行");
    run_for(&f, in, 0.2f, &first);
    CHECK(f.st == FS_DISARMED && first == FS_EV_ACTIVE_TIMEOUT, "ACTIVE 8 s → 超时停桨");

    fs_init(&f); in = ok_in(); fs_step(&f, FS_CMD_ARM, &in, 0.005f); fs_step(&f, FS_CMD_GO, &in, 0.005f);
    in.est_ok = 0; e = fs_step(&f, FS_CMD_NONE, &in, 0.005f);
    CHECK(e == FS_EV_EST_FAIL && f.st == FS_DISARMED, "ACTIVE 中 IMU 失效 → 立即停桨");
    fs_init(&f); in = ok_in(); fs_step(&f, FS_CMD_ARM, &in, 0.005f); fs_step(&f, FS_CMD_GO, &in, 0.005f);
    in.est_ok = 0; in.vbatt = NAN; in.acc_norm_g = 0.1f;
    e = fs_step(&f, FS_CMD_DISARM, &in, 0.005f);
    CHECK(e == FS_EV_DISARM_CMD && f.st == FS_DISARMED, "disarm 指令在任何输入下都立即生效");
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

int main(void)
{
    test_controller();
    test_safety();
    test_alt();
    printf("\n通过 %d / %d\n", n_pass, n_pass + n_fail);
    return n_fail ? 1 : 0;
}
