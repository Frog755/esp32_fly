/**
 * flight_main.c —— 耀坤 K1.0 正式飞控（Stage 26：干跑版 + 手机无线控制；电机驱动不编译进固件）
 *
 * ★ 安全：默认 FLIGHT_MOTOR_OUTPUT=0 —— 不含任何 LEDC / PWM 代码，上电第一件事把 4 个电机引脚拉低并 gpio_hold 锁存；
 *   控制器、安全状态机照常运行，算出的电机指令只通过串口输出（"F" 行），用于在不转电机的前提下核对整条链路方向。
 *
 * Stage 27：电机驱动版（-DFLIGHT_MOTOR_OUTPUT=1）必须显式给油门上限 FLIGHT_U_MAX；"F" 行 out_sum 为实际写入电机的占空比之和；
 *   遥测任务看门狗：控制循环 60 ms 没有前进 → 电机强制停转并锁死（需重启）。
 *
 * 任务（核 1 只跑控制，所有打印 / 串口 / I2C 都在核 0）：
 *   ctrl  核 1，1 kHz：读 IMU → 标定 + 轴映射 → flight_core（门控 Mahony + 竖直互补滤波；每 5 拍跑飞行状态机 + v9 控制器）→ 电机
 *         任务：抛出 / takeoff → 回正 → 悬停 5 s → 以 0.35 m/s 降落 → 触地停桨（见 flight_fsm.h）
 *   baro  核 0，50 Hz：BMP280 → 相对高度；每 5 次读一次电池电压（10 Hz）
 *   cli   核 0：串口指令 arm / takeoff（= go）/ land / disarm / status（空格或 k = 立即 disarm）
 *   telem 核 0，50 Hz：输出 "F" 行与 "EVT" 事件行
 *   web   核 0：Wi-Fi 热点 YK-FLY-xxxx + 网页 http://192.168.4.1 + WebSocket（web_link.c）；手机每 0.1 s 心跳，
 *         心跳（手机或串口 "hb"）超时 0.6 s = 链路丢失 → 飞行中降落 / 解锁中上锁 / 不能解锁
 *
 * 串口格式：
 *   F t_ms state mode roll pitch yaw wz_dps z vz vbatt gate_k u_M1 u_M2 u_M3 u_M4 out_sum z_target thrust_frac
 *     （state 0..5 = DISARMED/ARMED/TAKEOFF/RECOVER/HOLD/LAND；mode 1 = bang-bang 翻转；角度为度；u 为控制器算出的占空比；
 *       out_sum = 实际写到电机的占空比之和，本版恒为 0）
 *   EVT t_ms 事件名 状态名
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_cpu.h"
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"

#include "yk_board.h"            /* 驱动（含 mahony_gated.h、imu_calib.h） */
#include "flight_core.h"         /* 估计 + 状态机 + v9 控制器（纯 C，仿真共用） */
#include "motor_out.h"
#include "web_link.h"
#include "blackbox.h"             /* 机载黑匣子（Stage 28，纯 C） */
#include "esp_attr.h"
#include "esp_cache.h"
#include "esp_random.h"

#define FV_STR2(x) #x
#define FV_STR(x) FV_STR2(x)
#if FLIGHT_MOTOR_OUTPUT
/* 固件名里带油门上限原文（如 "u_max=0.40f"）：烧录脚本在烧之前从镜像里核对，防止烧错上限 */
#define FW_NAME "flight_v9 (MOTOR OUTPUT u_max=" FV_STR(FLIGHT_U_MAX) ", acc_lpf=" FV_STR(FC_ACC_LPF_HZ) ", blackbox)"
#else
#define FW_NAME "flight_v9 (dry-run: phone control, acc_lpf=" FV_STR(FC_ACC_LPF_HZ) ", blackbox)"
#endif
#define CTRL_STALL_MS 60          /* 遥测任务看门狗：控制循环停摆超过这么久 → 强制停转并锁死（仅电机驱动版） */
#define R2D 57.29578f

/* ---------------- 共享状态 ---------------- */
typedef struct {
    int64_t t_us;
    float q[4], w[3], k, anorm_g;
    float z, vz, ab, vbatt, z_target, thrust_frac;
    float u_cmd[4], out_sum;
    uint8_t st, mode, link_ok;
    float t_state;
    uint32_t ev_seq;                              /* 事件环：ev_ring[seq % 8] */
    struct { uint32_t seq; uint8_t ev, st; uint32_t t_ms; } ev_ring[8];
    uint32_t loops, late, exec_max_us, ctrl_max_us, spi_fail;
    uint32_t late15, late20, late50, period_max_us;   /* 周期 >1.5 / >2 / >5 ms 次数、最长周期 */
} snap_t;

static snap_t s_snap;
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static int s_cmd = FSM_CMD_NONE;                 /* cli / web → ctrl（原子交换取走） */
static uint32_t s_link_seq = 0;                  /* 每收到一次心跳（手机 / 串口 hb）+1 */
static volatile float s_vbatt = NAN;             /* baro → ctrl */
static volatile uint32_t s_vbatt_seq = 0;
static volatile float s_baro_z = 0;              /* baro → ctrl */
static volatile uint32_t s_baro_seq = 0;
static volatile int64_t s_baro_t_us = 0;
/* 黑匣子：PSRAM noinit 段，开机不清零（复位后可读出上一次开机的记录；断电丢失） */
EXT_RAM_NOINIT_ATTR static bb_store_t s_bb_store;
static bb_writer_t s_bbw;
static volatile int s_bb_pause = 0;              /* cli → ctrl：读出期间暂停写入 */
static volatile int s_quiet = 0;                 /* cli → telem：读出期间暂停 F 行 */
static float s_gbias[3];
static float s_a0_body[3];
static double s_p0;

static void post_cmd(fsm_cmd_t c)
{
    if (c == FSM_CMD_DISARM) { __atomic_store_n(&s_cmd, (int)c, __ATOMIC_SEQ_CST); return; }
    int expect = FSM_CMD_NONE;                    /* 非 disarm 指令不覆盖还没被取走的 disarm */
    if (!__atomic_compare_exchange_n(&s_cmd, &expect, (int)c, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) && expect != FSM_CMD_DISARM)
        __atomic_store_n(&s_cmd, (int)c, __ATOMIC_SEQ_CST);
}

/* ---------------- 气压计 + 电池（核 0，50 Hz） ---------------- */
static void baro_task(void *arg)
{
    (void)arg;
    TickType_t last = xTaskGetTickCount();
    int n = 0;
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(20));
        double p, t;
        if (bmp_read(&p, &t)) {
            float z = (float)(44330.0 * (1.0 - pow(p / s_p0, 0.190295)));
            s_baro_z = z; s_baro_t_us = esp_timer_get_time(); s_baro_seq++;
        }
        if (++n % 5 == 0) { s_vbatt = vbatt_read(); s_vbatt_seq++; }
    }
}

/* ---------------- 控制（核 1，1 kHz）：读 IMU → flight_core（与软件在环仿真同一份代码）→ 电机 ---------------- */
static void ctrl_task(void *arg)
{
    (void)arg;
    static fc_t fc;
    fc_init(&fc, s_a0_body);
    uint32_t link_seen = __atomic_load_n(&s_link_seq, __ATOMIC_SEQ_CST);
    uint32_t late15 = 0, late20 = 0, late50 = 0, period_max = 0;
    uint32_t baro_seen = s_baro_seq, vb_seen = s_vbatt_seq, loops = 0, late = 0, exec_max = 0, ctrl_max = 0, spi_fail = 0;
    int64_t prev = esp_timer_get_time(), last_baro_t = s_baro_t_us;
    float out_sum = 0.0f;                          /* 实际写入电机的占空比之和（干跑版恒 0） */
    imu_raw_t r; float a[3], g[3];
    /* 黑匣子：每 5 ms 窗口内的振动（相邻 1 kHz 样本差分 RMS）、最长周期、最长执行 */
    float a_prev[3] = {0}, vib_ss = 0.0f; int have_prev = 0, vib_n = 0;
    uint32_t win_per = 0, win_exec = 0, bb_ev_seen = 0;
    TickType_t last = xTaskGetTickCount();

    for (;;) {
        vTaskDelayUntil(&last, 1);
        const int64_t now = esp_timer_get_time();
        const uint32_t c0 = esp_cpu_get_cycle_count();
        const uint32_t per = (uint32_t)(now - prev);
        float dt = per * 1e-6f; prev = now;
        if (per > 1100) late++;
        if (per > 1500) late15++;
        if (per > 2000) late20++;
        if (per > 5000) late50++;
        if (loops > 10 && per > period_max) period_max = per;
        if (dt > 0.005f) dt = 0.005f;
        loops++;

        imu_read(&r);
        const int raw_zero = (r.a[0] == 0 && r.a[1] == 0 && r.a[2] == 0 && r.g[0] == 0 && r.g[1] == 0 && r.g[2] == 0);
        if (raw_zero) spi_fail++;
        imu_to_body(&r, s_gbias, a, g);
        if (have_prev) { for (int j = 0; j < 3; j++) { const float d = a[j] - a_prev[j]; vib_ss += d * d; } vib_n++; }
        memcpy(a_prev, a, sizeof(a_prev)); have_prev = 1;
        if (per > win_per) win_per = per;

        int baro_new = 0; float baro_dt = 0.02f, baro_z = 0.0f;
        const uint32_t bs = s_baro_seq;
        if (bs != baro_seen) {
            const int64_t tb = s_baro_t_us;
            baro_new = 1; baro_z = s_baro_z; baro_dt = (tb - last_baro_t) * 1e-6f;
            last_baro_t = tb; baro_seen = bs;
        }
        float vb = NAN;
        const uint32_t vs = s_vbatt_seq;
        if (vs != vb_seen) { vb = s_vbatt; vb_seen = vs; }
        const int cmd = __atomic_exchange_n(&s_cmd, FSM_CMD_NONE, __ATOMIC_SEQ_CST);
        fc_command(&fc, (fsm_cmd_t)cmd);
        const uint32_t ls = __atomic_load_n(&s_link_seq, __ATOMIC_SEQ_CST);
        if (ls != link_seen) { fc_link_beat(&fc); link_seen = ls; }

        const uint32_t cc0 = esp_cpu_get_cycle_count();
        const int ticked = fc_step(&fc, a, g, dt, !raw_zero, baro_new, baro_z, baro_dt, vb);
        if (ticked) {
            out_sum = motor_out_write(fc.u_out);
            const uint32_t cu = (esp_cpu_get_cycle_count() - cc0) / 240;
            if (cu > ctrl_max) ctrl_max = cu;

            /* ---- 黑匣子：一条 64 字节记录写进 PSRAM，并立即回写 cache（复位时不丢最后几秒） ---- */
            bb_rec_t br = {0};
            br.t_ms = (uint32_t)(now / 1000); br.st = (uint8_t)fc.fsm.st; br.mode = fc.mode;
            br.ev = (fc.ev_seq != bb_ev_seen) ? fc.ev_last : 0; bb_ev_seen = fc.ev_seq;
            const int on = fsm_motors_on(&fc.fsm);
            br.flags = (uint8_t)((fc_link_ok(&fc) ? BB_F_LINK : 0) | (!raw_zero ? BB_F_IMU_OK : 0) | (on ? BB_F_ON : 0) |
                                 (motor_out_killed() ? BB_F_KILLED : 0) | (FLIGHT_MOTOR_OUTPUT ? BB_F_MOTOR : 0));
            br.k = bb_u8(fc.k * 100.0f);
            for (int j = 0; j < 4; j++) br.q[j] = bb_i16(fc.mg.q[j] * 32767.0f);
            for (int j = 0; j < 3; j++) { br.g[j] = bb_i16(g[j] * R2D * 10.0f); br.a[j] = bb_i16(a[j] / MAHONY_G * 1000.0f); }
            br.z = bb_i16(fc.alt.z * 1000.0f); br.vz = bb_i16(fc.alt.vz * 1000.0f); br.zt = bb_i16(fc.fsm.z_target * 1000.0f);
            br.ab = bb_i16(fc.alt.ab * 1000.0f); br.zb = bb_i16(s_baro_z * 1000.0f);
            br.vb = bb_u16(fc.vbatt * 1000.0f); br.thr = bb_u16(fc.thrust_frac * 1000.0f);
            for (int j = 0; j < 4; j++) br.u[j] = bb_u16(fc.u_cmd[j] * 10000.0f);
            br.out = bb_u16(out_sum * 10000.0f);
            br.per_max = bb_u8(win_per / 20.0f); br.exec_max = bb_u8(win_exec / 10.0f);
            br.vib = bb_u16(vib_n ? sqrtf(vib_ss / vib_n) / MAHONY_G * 1000.0f : 0.0f);
            win_per = 0; win_exec = 0; vib_ss = 0.0f; vib_n = 0;
            s_bbw.paused = s_bb_pause;
            bb_rec_t *dst = bb_write(&s_bbw, &br, br.st, on);
            if (dst) esp_cache_msync(dst, BB_REC_SIZE, ESP_CACHE_MSYNC_FLAG_DIR_C2M | ESP_CACHE_MSYNC_FLAG_TYPE_DATA);
        }
        const uint32_t ex = (esp_cpu_get_cycle_count() - c0) / 240;
        if (ex > exec_max) exec_max = ex;
        if (ex > win_exec) win_exec = ex;

        portENTER_CRITICAL(&s_mux);
        s_snap.t_us = now; memcpy(s_snap.q, fc.mg.q, sizeof(fc.mg.q)); memcpy(s_snap.w, g, sizeof(s_snap.w));
        s_snap.k = fc.k; s_snap.anorm_g = fc.anorm_g; s_snap.z = fc.alt.z; s_snap.vz = fc.alt.vz; s_snap.ab = fc.alt.ab;
        s_snap.vbatt = fc.vbatt; s_snap.z_target = fc.fsm.z_target; s_snap.thrust_frac = fc.thrust_frac;
        s_snap.st = (uint8_t)fc.fsm.st; s_snap.mode = fc.mode; s_snap.link_ok = (uint8_t)fc_link_ok(&fc); s_snap.t_state = fc.fsm.t_state;
        memcpy(s_snap.u_cmd, fc.u_cmd, sizeof(fc.u_cmd));
        s_snap.out_sum = out_sum;
        if (fc.ev_seq != s_snap.ev_seq) {
            s_snap.ev_seq = fc.ev_seq;
            const int i = (int)(fc.ev_seq % 8u);
            s_snap.ev_ring[i].seq = fc.ev_seq; s_snap.ev_ring[i].ev = fc.ev_last; s_snap.ev_ring[i].st = (uint8_t)fc.fsm.st;
            s_snap.ev_ring[i].t_ms = (uint32_t)(now / 1000);
        }
        s_snap.late15 = late15; s_snap.late20 = late20; s_snap.late50 = late50; s_snap.period_max_us = period_max;
        s_snap.loops = loops; s_snap.late = late; s_snap.exec_max_us = exec_max; s_snap.ctrl_max_us = ctrl_max; s_snap.spi_fail = spi_fail;
        portEXIT_CRITICAL(&s_mux);
    }
}

/* ---------------- 遥测（核 0，50 Hz） ---------------- */
static void euler(const float q[4], float *roll, float *pitch, float *yaw)
{
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    *roll = atan2f(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y)) * R2D;
    float sp = 2.0f * (w * y - z * x); sp = sp > 1.0f ? 1.0f : (sp < -1.0f ? -1.0f : sp);
    *pitch = asinf(sp) * R2D;
    *yaw = atan2f(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z)) * R2D;
}

static void print_status(void)
{
    snap_t s; portENTER_CRITICAL(&s_mux); s = s_snap; portEXIT_CRITICAL(&s_mux);
    printf("STATUS fw %s\n", FW_NAME);
    printf("STATUS motor_output %d\n", FLIGHT_MOTOR_OUTPUT);
    printf("STATUS u_max %.3f\n", FLIGHT_MOTOR_OUTPUT ? (double)FLIGHT_U_MAX : 0.0);
    printf("STATUS motor_killed %d\n", motor_out_killed());
    printf("STATUS motor_pins_level %d %d %d %d\n", gpio_get_level(16), gpio_get_level(6), gpio_get_level(5), gpio_get_level(15));
    printf("STATUS state %s\n", FSM_STATE_NAME[s.st]);
    printf("STATUS vbatt %.3f\n", s.vbatt);
    printf("STATUS calib_valid %d\n", IMU_CALIB_VALID);
    printf("STATUS level_trim %d %.3f %.3f\n", LEVEL_TRIM_VALID, LEVEL_TRIM_ROLL_DEG, LEVEL_TRIM_PITCH_DEG);
    printf("STATUS gyro_bias_dps %.3f %.3f %.3f\n", s_gbias[0], s_gbias[1], s_gbias[2]);
    printf("STATUS loops %lu late_gt1100us %lu exec_max_us %lu ctrl200_max_us %lu spi_fail %lu\n", (unsigned long)s.loops,
           (unsigned long)s.late, (unsigned long)s.exec_max_us, (unsigned long)s.ctrl_max_us, (unsigned long)s.spi_fail);
    printf("STATUS alt z %.3f vz %.3f acc_bias %.3f z_target %.3f thrust_frac %.3f\n", s.z, s.vz, s.ab, s.z_target, s.thrust_frac);
    printf("STATUS mission hold_s %.1f takeoff_h %.2f land_v %.2f\n", FSM_HOLD_S, FSM_TAKEOFF_H, FSM_LAND_V);
    web_stats_t w; web_link_stats(&w);
    printf("STATUS period late_gt1500us %lu late_gt2000us %lu late_gt5000us %lu period_max_us %lu\n", (unsigned long)s.late15,
           (unsigned long)s.late20, (unsigned long)s.late50, (unsigned long)s.period_max_us);
    printf("STATUS link_ok %d\n", s.link_ok);
    printf("STATUS blackbox boot %u slot %d n %lu frozen %d\n", (unsigned)s_bbw.boot, s_bbw.boot & 1, (unsigned long)s_bbw.n, s_bbw.frozen);
    printf("STATUS wifi %s ssid %s phones %d websockets %d hb_n %lu gap_max_ms %lu\n", w.started ? "on" : "off", w.ssid, w.n_sta, w.n_ws,
           (unsigned long)w.hb_n, (unsigned long)w.gap_all_ms);
    printf("STATUS mapping");
    for (int i = 0; i < 4; i++) printf(" u%d->%s(%s,GPIO%d)", i, MOT_NAME_SIM[i], MOT_POS_SIM[i], MOT_PIN_SIM[i]);
    printf("\nSTATUS_END\n");
}

static void telem_task(void *arg)
{
    (void)arg;
    uint32_t ev_seen = 0, loops_seen = 0;
    int64_t loops_t = esp_timer_get_time();
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(20));
        snap_t s; portENTER_CRITICAL(&s_mux); s = s_snap; portEXIT_CRITICAL(&s_mux);
        const int64_t tn = esp_timer_get_time();
        if (s.loops != loops_seen) { loops_seen = s.loops; loops_t = tn; }
        else if (FLIGHT_MOTOR_OUTPUT && !motor_out_killed() && tn - loops_t > CTRL_STALL_MS * 1000) {
            motor_out_kill();                     /* 控制循环停摆：电机不能停在最后一次写入的占空比上 */
            printf("EVT %lld motor_watchdog_kill CTRL_STALL\n", (long long)(tn / 1000));
        }
        if (s_quiet) continue;                    /* 黑匣子读出中：暂停遥测输出 */
        if (s.ev_seq != ev_seen) {
            const uint32_t from = (s.ev_seq - ev_seen > 8u) ? s.ev_seq - 7u : ev_seen + 1u;
            for (uint32_t q = from; q <= s.ev_seq; q++) {
                const int i = (int)(q % 8u);
                if (s.ev_ring[i].seq != q) continue;
                printf("EVT %lu %s %s\n", (unsigned long)s.ev_ring[i].t_ms, FSM_EV_NAME[s.ev_ring[i].ev], FSM_STATE_NAME[s.ev_ring[i].st]);
            }
            ev_seen = s.ev_seq;
        }
        float ro, pi, ya; euler(s.q, &ro, &pi, &ya);
        printf("F %lld %d %d %.1f %.1f %.1f %.1f %.3f %.3f %.2f %.2f %.3f %.3f %.3f %.3f %.3f %.3f %.2f\n", (long long)(s.t_us / 1000),
               s.st, s.mode, ro, pi, ya, s.w[2] * R2D, s.z, s.vz, s.vbatt, s.k,
               s.u_cmd[0], s.u_cmd[1], s.u_cmd[2], s.u_cmd[3], s.out_sum, s.z_target, s.thrust_frac);
    }
}

/* ---------------- 手机网页回调（核 0，httpd / 遥测任务里调用） ---------------- */
static int web_on_cmd(const char *c)
{
    if (!strcmp(c, "disarm")) post_cmd(FSM_CMD_DISARM);
    else if (!strcmp(c, "arm")) post_cmd(FSM_CMD_ARM);
    else if (!strcmp(c, "takeoff")) post_cmd(FSM_CMD_TAKEOFF);
    else if (!strcmp(c, "land")) post_cmd(FSM_CMD_LAND);
    else return 0;
    return 1;
}
static void web_on_beat(void) { __atomic_add_fetch(&s_link_seq, 1, __ATOMIC_SEQ_CST); }
static float fin(float x) { return isfinite(x) ? x : 0.0f; }
static int web_build_json(char *b, size_t n)
{
    snap_t s; portENTER_CRITICAL(&s_mux); s = s_snap; portEXIT_CRITICAL(&s_mux);
    web_stats_t w; web_link_stats(&w);
    float ro, pi, ya; euler(s.q, &ro, &pi, &ya);
    int len = snprintf(b, n, "{\"st\":%d,\"ts\":%.2f,\"mode\":%d,\"r\":%.1f,\"p\":%.1f,\"y\":%.1f,\"z\":%.2f,\"vz\":%.2f,\"zt\":%.2f,"
                       "\"vb\":%.2f,\"k\":%.2f,\"u\":[%.3f,%.3f,%.3f,%.3f],\"out\":%.3f,\"mo\":%d,\"um\":%.3f,\"mk\":%d,\"lk\":%d,\"hb\":%lu,\"gap\":%lu,\"evs\":%lu,\"ev\":\"%s\",\"evl\":[",
                       s.st, fin(s.t_state), s.mode, fin(ro), fin(pi), fin(ya), fin(s.z), fin(s.vz), fin(s.z_target), fin(s.vbatt), fin(s.k),
                       fin(s.u_cmd[0]), fin(s.u_cmd[1]), fin(s.u_cmd[2]), fin(s.u_cmd[3]), fin(s.out_sum), FLIGHT_MOTOR_OUTPUT,
                       FLIGHT_MOTOR_OUTPUT ? (double)FLIGHT_U_MAX : 0.0, motor_out_killed(), s.link_ok,
                       (unsigned long)w.hb_seq, (unsigned long)w.gap_win_ms, (unsigned long)s.ev_seq,
                       s.ev_seq ? FSM_EV_NAME[s.ev_ring[s.ev_seq % 8u].ev] : "");
    int first = 1;
    for (uint32_t q = (s.ev_seq > 3u ? s.ev_seq - 3u : 1u); q <= s.ev_seq && len < (int)n - 64; q++) {
        const int i = (int)(q % 8u);
        if (s.ev_ring[i].seq != q) continue;
        len += snprintf(b + len, n - len, "%s[%lu,\"%s\"]", first ? "" : ",", (unsigned long)q, FSM_EV_NAME[s.ev_ring[i].ev]);
        first = 0;
    }
    len += snprintf(b + len, n - len, "]}");
    return len;
}

/* ---------------- 黑匣子读出（核 0，只在 DISARMED 时允许） ---------------- */
/* 读出走 usb_serial_jtag_write_bytes 阻塞写：VFS（printf）层在电脑 50 ms 没取走数据时会静默丢字；
 * 驱动层只在环形缓冲满时等待，不丢。连续 5 s 写不出去（电脑没在读）就放弃本次读出。 */
static volatile int s_bb_abort = 0;
static void usb_write_all(const char *p, size_t n)
{
    int fails = 0;
    while (n && !s_bb_abort) {
        const int w = usb_serial_jtag_write_bytes(p, n, pdMS_TO_TICKS(200));
        if (w <= 0) { if (++fails > 25) s_bb_abort = 1; continue; }
        fails = 0; p += w; n -= (size_t)w;
    }
}
static void bb_out_stdout(void *ctx, const char *line)
{
    (void)ctx;
    usb_write_all(line, strlen(line)); usb_write_all("\n", 1);
}
static int bb_begin_dump(void)
{
    snap_t s; portENTER_CRITICAL(&s_mux); s = s_snap; portEXIT_CRITICAL(&s_mux);
    if (s.st != FSM_DISARMED) { printf("CLI bbdump refused: 只能在未解锁（DISARMED）时读出\n"); return 0; }
    s_bb_pause = 1; s_quiet = 1; s_bb_abort = 0;
    vTaskDelay(pdMS_TO_TICKS(30));                 /* 等 ctrl 拍和遥测停下 */
    fflush(stdout);
    usb_serial_jtag_wait_tx_done(pdMS_TO_TICKS(200));
    return 1;
}
static void bb_end_dump(void)
{
    usb_serial_jtag_wait_tx_done(pdMS_TO_TICKS(500));
    s_quiet = 0; s_bb_pause = 0;
}

static void bb_info(void)
{
    for (int si = 0; si < 2; si++) {
        const bb_slot_t *h = &s_bb_store.slot[si];
        const bb_range_t rg = bb_scan(h);
        if (!bb_slot_valid(h)) { printf("BBINFO slot %d invalid\n", si); continue; }
        printf("BBINFO slot %d boot %lu reset_reason %lu records %lu (%.1f s) idx %lu..%lu%s%s\n", si, (unsigned long)h->boot,
               (unsigned long)h->reset_reason, (unsigned long)rg.n_valid, rg.n_valid / (double)BB_HZ, (unsigned long)rg.lo,
               (unsigned long)rg.hi, h == s_bbw.cur ? " current" : "", (h == s_bbw.cur && s_bbw.frozen) ? " frozen" : "");
    }
}

static uint8_t s_bb_tmp[16 * BB_REC_SIZE];
static char s_bb_line[1400];

static void bb_dump(void)
{
    if (!bb_begin_dump()) return;
    char hdr[32]; snprintf(hdr, sizeof hdr, "BB_BEGIN %u", (unsigned)s_bbw.boot); bb_out_stdout(NULL, hdr);
    for (int si = 0; si < 2 && !s_bb_abort; si++)
        bb_dump_slot(&s_bb_store, si, &s_bb_store.slot[si] == s_bbw.cur, s_bbw.frozen, bb_out_stdout, NULL, s_bb_tmp, s_bb_line);
    bb_out_stdout(NULL, "BB_DONE");
    bb_end_dump();
}

static void bb_get(const char *args)                 /* bbget <槽> <lo> <hi>：补发读出时丢掉的记录 */
{
    unsigned si = 9, lo = 0, hi = 0;
    if (sscanf(args, "%u %u %u", &si, &lo, &hi) != 3 || si > 1 || hi < lo) { printf("CLI bbget usage: bbget <0|1> <lo> <hi>\n"); return; }
    if (!bb_begin_dump()) return;
    bb_dump_get(&s_bb_store, (int)si, lo, hi, bb_out_stdout, NULL, s_bb_tmp, s_bb_line);
    bb_end_dump();
}

/* ---------------- 串口指令（核 0） ---------------- */
static void cli_task(void *arg)
{
    (void)arg;
    char line[48]; int n = 0;
    for (;;) {
        uint8_t b[32];
        int got = usb_serial_jtag_read_bytes(b, sizeof(b), pdMS_TO_TICKS(20));
        for (int i = 0; i < got; i++) {
            const char c = (char)b[i];
            if (c == ' ' || c == 'k' || c == 'K') { if (n == 0) { post_cmd(FSM_CMD_DISARM); continue; } }   /* 单键急停 */
            if (c == '\r' || c == '\n') {
                line[n] = 0; n = 0;
                if (!strcmp(line, "arm")) post_cmd(FSM_CMD_ARM);
                else if (!strcmp(line, "disarm") || !strcmp(line, "stop")) post_cmd(FSM_CMD_DISARM);
                else if (!strcmp(line, "takeoff") || !strcmp(line, "go")) post_cmd(FSM_CMD_TAKEOFF);
                else if (!strcmp(line, "land")) post_cmd(FSM_CMD_LAND);
                else if (!strcmp(line, "status")) print_status();
                else if (!strcmp(line, "hb")) __atomic_add_fetch(&s_link_seq, 1, __ATOMIC_SEQ_CST);   /* 串口心跳（桌面测试用） */
                else if (!strcmp(line, "bbdump")) bb_dump();
                else if (!strncmp(line, "bbget ", 6)) bb_get(line + 6);
                else if (!strcmp(line, "bbinfo")) bb_info();
                else if (!strcmp(line, "reboot")) {                           /* 软件复位（测黑匣子跨复位保留；只在 DISARMED） */
                    snap_t s; portENTER_CRITICAL(&s_mux); s = s_snap; portEXIT_CRITICAL(&s_mux);
                    if (s.st != FSM_DISARMED) printf("CLI reboot refused: 只能在未解锁（DISARMED）时重启\n");
                    else { printf("CLI reboot\n"); fflush(stdout); vTaskDelay(pdMS_TO_TICKS(50)); esp_restart(); }
                }
                else if (line[0]) printf("CLI unknown '%s'（可用：arm / takeoff / land / disarm / status / hb / bbinfo / bbdump / bbget / reboot；空格或 k = 立即 disarm）\n", line);
            } else if (n < (int)sizeof(line) - 1) line[n++] = c;
        }
    }
}

/* ---------------- 开机 ---------------- */
static void gyro_calibrate(void)
{
    for (int attempt = 1; attempt <= 10; attempt++) {
        double sg[3] = {0}, sgg[3] = {0}, sa[3] = {0};
        const int N = 2000; imu_raw_t r;
        TickType_t last = xTaskGetTickCount();
        for (int i = 0; i < N; i++) {
            vTaskDelayUntil(&last, 1);
            imu_read(&r);
            for (int j = 0; j < 3; j++) { sg[j] += r.g[j]; sgg[j] += (double)r.g[j] * r.g[j]; sa[j] += r.a[j]; }
        }
        double smax = 0;
        for (int j = 0; j < 3; j++) { double m = sg[j] / N, v = sgg[j] / N - m * m; double sd = v > 0 ? sqrt(v) : 0; if (sd > smax) smax = sd; }
        printf("BOOT gyro_calib attempt %d std_max_dps %.3f\n", attempt, smax);
        if (smax < 0.6) {
            for (int j = 0; j < 3; j++) s_gbias[j] = (float)(sg[j] / N);
            imu_raw_t m = {{(float)(sa[0] / N), (float)(sa[1] / N), (float)(sa[2] / N)}, {0, 0, 0}, 0};
            float gb[3]; imu_to_body(&m, s_gbias, s_a0_body, gb);
            printf("BOOT gyro_bias_dps %.3f %.3f %.3f\n", s_gbias[0], s_gbias[1], s_gbias[2]);
            printf("BOOT acc_body_g %.4f %.4f %.4f\n", s_a0_body[0] / MAHONY_G, s_a0_body[1] / MAHONY_G, s_a0_body[2] / MAHONY_G);
            return;
        }
        printf("BOOT gyro_calib 在动，请把飞机放稳……\n");
    }
    fatal("gyro_calib_failed");
}

void app_main(void)
{
    motors_lock_low();                                                   /* ★ 第一件事 */
    const int used[] = {PIN_SPI_SCK, PIN_SPI_MOSI, PIN_SPI_MISO, PIN_IMU_CS, PIN_FLOW_CS, PIN_I2C_SDA, PIN_I2C_SCL, PIN_VBATT};
    const char *names[] = {"SCK", "MOSI", "MISO", "IMU_CS", "FLOW_CS", "SDA", "SCL", "VBATT"};
    for (int i = 0; i < 8; i++) check_pin_not_motor(used[i], names[i]);
    for (int i = 0; i < 4; i++)
        if (MOT_PIN_SIM[i] != MOTOR_PINS[0] && MOT_PIN_SIM[i] != MOTOR_PINS[1] && MOT_PIN_SIM[i] != MOTOR_PINS[2] && MOT_PIN_SIM[i] != MOTOR_PINS[3])
            fatal("mapping_pin_not_motor");

    usb_serial_jtag_driver_config_t uc = {.tx_buffer_size = 4096, .rx_buffer_size = 256};
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&uc));
    usb_serial_jtag_vfs_use_driver();
    vTaskDelay(pdMS_TO_TICKS(2500));

    printf("BOOT_BEGIN\nBOOT fw %s\nBOOT motor_output %d\nBOOT reset_reason %d\nBOOT calib_valid %d\n", FW_NAME, FLIGHT_MOTOR_OUTPUT,
           (int)esp_reset_reason(), IMU_CALIB_VALID);
    printf("BOOT motor_pins_level %d %d %d %d\n", gpio_get_level(16), gpio_get_level(6), gpio_get_level(5), gpio_get_level(15));
    printf("BOOT u_max %.3f\n", FLIGHT_MOTOR_OUTPUT ? (double)FLIGHT_U_MAX : 0.0);
    {   /* 黑匣子：选槽（上一次开机的槽保留），槽头立即回写到 PSRAM */
        bb_begin(&s_bbw, &s_bb_store, (uint32_t)esp_reset_reason(), esp_random());
        const int fl = ESP_CACHE_MSYNC_FLAG_DIR_C2M | ESP_CACHE_MSYNC_FLAG_TYPE_DATA;
        esp_cache_msync((uint8_t *)&s_bb_store + BB_GUARD, 64, fl);   /* 存储头（跳过开头 4 KB 时序调校区） */
        esp_cache_msync(&s_bb_store.slot[0], 64, fl); esp_cache_msync(&s_bb_store.slot[1], 64, fl);
        const bb_slot_t *prev = &s_bb_store.slot[(s_bbw.boot & 1) ^ 1];
        const bb_range_t rg = bb_scan(prev);
        printf("BOOT blackbox boot %u slot %d store %u KB; prev slot %s boot %lu records %lu (%.1f s)\n", (unsigned)s_bbw.boot,
               s_bbw.boot & 1, (unsigned)(sizeof(s_bb_store) / 1024), bb_slot_valid(prev) ? "valid" : "empty",
               (unsigned long)(bb_slot_valid(prev) ? prev->boot : 0), (unsigned long)rg.n_valid, rg.n_valid / (double)BB_HZ);
    }
    if (!IMU_CALIB_VALID) printf("BOOT ⚠ imu_calib.h 无效：请先运行 code/sensor_test/calib.sh 重新标定\n");

    vbatt_init();
    s_vbatt = vbatt_read();
    printf("BOOT vbatt_v %.3f\n", s_vbatt);
    i2c_scan();
    if (!bmp_init()) fatal("bmp280_init");
    {
        double ps = 0, t; int n = 0;
        for (int i = 0; i < 25; i++) { double p; if (bmp_read(&p, &t)) { ps += p; n++; } vTaskDelay(pdMS_TO_TICKS(20)); }
        if (n < 10) fatal("bmp280_read");
        s_p0 = ps / n;
        printf("BOOT baro_p0_pa %.1f\n", s_p0);
    }
    if (!imu_init()) fatal("imu_init");
    printf("BOOT gyro_calib 开始：请保持飞机静止 2 秒\n");
    gyro_calibrate();
    motor_out_init();
    static const web_cb_t wcb = {.on_cmd = web_on_cmd, .on_beat = web_on_beat, .build_json = web_build_json};
    web_link_start(&wcb);

    xTaskCreatePinnedToCore(baro_task, "baro", 4096, NULL, 6, NULL, 0);
    vTaskDelay(pdMS_TO_TICKS(60));
    xTaskCreatePinnedToCore(ctrl_task, "ctrl", 6144, NULL, configMAX_PRIORITIES - 2, NULL, 1);
    xTaskCreatePinnedToCore(cli_task, "cli", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(telem_task, "telem", 4096, NULL, 4, NULL, 0);
    vTaskDelay(pdMS_TO_TICKS(500));
    print_status();
    printf("FLT_READY\n");
}
