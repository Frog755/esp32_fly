/**
 * flight_main.c —— 耀坤 K1.0 正式飞控（Stage 24：干跑版，电机驱动不编译进固件）
 *
 * ★ 安全：默认 FLIGHT_MOTOR_OUTPUT=0 —— 不含任何 LEDC / PWM 代码，上电第一件事把 4 个电机引脚拉低并 gpio_hold 锁存；
 *   控制器、安全状态机照常运行，算出的电机指令只通过串口输出（"F" 行），用于在不转电机的前提下核对整条链路方向。
 *
 * 任务（核 1 只跑控制，所有打印 / 串口 / I2C 都在核 0）：
 *   ctrl  核 1，1 kHz：读 IMU → 标定 + 轴映射 → 门控 Mahony → 竖直互补滤波；每 5 拍（200 Hz）跑安全状态机 + v9 控制器 + 电机输出
 *   baro  核 0，50 Hz：BMP280 → 相对高度；每 5 次读一次电池电压（10 Hz）
 *   cli   核 0：串口指令 arm / disarm / go / status（空格或 k = 立即 disarm）
 *   telem 核 0，50 Hz：输出 "F" 行与 "EVT" 事件行
 *
 * 串口格式：
 *   F t_ms state mode roll pitch yaw wz_dps z vz vbatt gate_k u_M1 u_M2 u_M3 u_M4 out_sum
 *     （state 0/1/2 = DISARMED/ARMED/ACTIVE；mode 1 = bang-bang 翻转；角度为度；u 为控制器算出的占空比；
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
#include "recovery_expert9.h"
#include "alt_est.h"
#include "flight_safety.h"
#include "motor_out.h"

#define FW_NAME "flight_v9 (Stage 24 dry-run)"
#define R2D 57.29578f

/* ---------------- 共享状态 ---------------- */
typedef struct {
    int64_t t_us;
    float q[4], w[3], k, anorm_g;
    float z, vz, ab, vbatt;
    float u_cmd[4], out_sum;
    uint8_t st, mode;
    uint32_t ev_seq; uint8_t ev; float ev_t;
    /* 时序统计 */
    uint32_t loops, late, exec_max_us, ctrl_max_us, spi_fail;
} snap_t;

static snap_t s_snap;
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static volatile int s_cmd = FS_CMD_NONE;         /* cli → ctrl（ctrl 取走后清零） */
static volatile float s_vbatt = NAN;             /* baro → ctrl */
static volatile float s_baro_z = 0;              /* baro → ctrl */
static volatile uint32_t s_baro_seq = 0;
static volatile int64_t s_baro_t_us = 0;
static float s_gbias[3];
static float s_a0_body[3];
static double s_p0;

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
        if (++n % 5 == 0) s_vbatt = vbatt_read();
    }
}

/* ---------------- 控制（核 1，1 kHz） ---------------- */
static void ctrl_task(void *arg)
{
    (void)arg;
    mahony_gated_t mg; mahony_gated_init_from_accel(&mg, s_a0_body);
    alt_est_t alt; alt_reset(&alt, 0.0f);
    recovery9_state_t rc; recovery9_reset(&rc);
    fs_t fs; fs_init(&fs);
    float z_target = 0.0f, u_cmd[4] = {0}, k = 0, an = 1;
    uint8_t mode = 0;
    uint32_t baro_seq_seen = s_baro_seq, ev_seq = 0, loops = 0, late = 0, exec_max = 0, ctrl_max = 0, spi_fail = 0;
    uint8_t last_ev = 0; float last_ev_t = 0;
    int64_t prev = esp_timer_get_time(), last_baro_t = prev;
    imu_raw_t r; float a[3], g[3];
    TickType_t last = xTaskGetTickCount();

    for (;;) {
        vTaskDelayUntil(&last, 1);
        const int64_t now = esp_timer_get_time();
        const uint32_t c0 = esp_cpu_get_cycle_count();
        float dt = (now - prev) * 1e-6f; prev = now;
        if (dt > 1100e-6f) late++;
        if (dt > 0.005f) dt = 0.005f;
        loops++;

        imu_read(&r);
        const int raw_zero = (r.a[0] == 0 && r.a[1] == 0 && r.a[2] == 0 && r.g[0] == 0 && r.g[1] == 0 && r.g[2] == 0);
        if (raw_zero) spi_fail++;
        imu_to_body(&r, s_gbias, a, g);
        k = mahony_gated_update(&mg, g, a, dt);
        an = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) / MAHONY_G;
        alt_predict(&alt, alt_up_accel(mg.q, a), dt);
        const uint32_t bs = s_baro_seq;
        if (bs != baro_seq_seen) {
            const int64_t tb = s_baro_t_us;
            alt_correct(&alt, s_baro_z, (tb - last_baro_t) * 1e-6f);
            last_baro_t = tb; baro_seq_seen = bs;
        }

        uint32_t c_ctrl = 0;
        if (loops % 5 == 0) {                                   /* ---- 200 Hz ---- */
            const uint32_t cc0 = esp_cpu_get_cycle_count();
            const fs_cmd_t cmd = (fs_cmd_t)__atomic_exchange_n(&s_cmd, FS_CMD_NONE, __ATOMIC_SEQ_CST);   /* 取走指令（不丢 disarm） */
            const float qn = mg.q[0] * mg.q[0] + mg.q[1] * mg.q[1] + mg.q[2] * mg.q[2] + mg.q[3] * mg.q[3];
            const int baro_fresh = (now - last_baro_t) < 200000;
            fs_in_t in = {.vbatt = s_vbatt, .acc_norm_g = an,
                          .r22 = 1.0f - 2.0f * (mg.q[1] * mg.q[1] + mg.q[2] * mg.q[2]), .gate_k = k,
                          .est_ok = !raw_zero && isfinite(qn) && qn > 0.5f && baro_fresh};
            const fs_state_t st_before = fs.st;
            const fs_event_t ev = fs_step(&fs, cmd, &in, 0.005f);
            if (ev != FS_EV_NONE) { ev_seq++; last_ev = (uint8_t)ev; last_ev_t = now * 1e-6f; }
            if (fs.st == FS_ACTIVE && st_before != FS_ACTIVE) { recovery9_reset(&rc); z_target = alt.z; }
            if (fs.st != FS_ACTIVE) z_target = alt.z;           /* 未起飞时目标高度跟随当前高度（干跑时 z_est≈0） */

            const float pos[3] = {0.0f, 0.0f, alt.z}, vel[3] = {0.0f, 0.0f, alt.vz}, tgt[3] = {0.0f, 0.0f, z_target};
            const float vb = isfinite(s_vbatt) ? s_vbatt : EXP9_VREF;
            mode = recovery_expert9_step(&rc, mg.q, g, pos, vel, alt.vz, alt.z - z_target, tgt, vb, u_cmd);

            float u_out[4] = {0, 0, 0, 0};
            if (fs.st == FS_ACTIVE) memcpy(u_out, u_cmd, sizeof(u_out));
            motor_out_write(u_out);
            const float out_sum = FLIGHT_MOTOR_OUTPUT ? (u_out[0] + u_out[1] + u_out[2] + u_out[3]) : 0.0f;
            c_ctrl = esp_cpu_get_cycle_count() - cc0;
            if (c_ctrl / 240 > ctrl_max) ctrl_max = c_ctrl / 240;

            portENTER_CRITICAL(&s_mux);
            s_snap.st = (uint8_t)fs.st; s_snap.mode = mode;
            memcpy(s_snap.u_cmd, u_cmd, sizeof(u_cmd)); s_snap.out_sum = out_sum;
            s_snap.ev_seq = ev_seq; s_snap.ev = last_ev; s_snap.ev_t = last_ev_t;
            portEXIT_CRITICAL(&s_mux);
        }
        const uint32_t ex = (esp_cpu_get_cycle_count() - c0) / 240;
        if (ex > exec_max) exec_max = ex;

        portENTER_CRITICAL(&s_mux);
        s_snap.t_us = now; memcpy(s_snap.q, mg.q, sizeof(mg.q)); memcpy(s_snap.w, g, sizeof(s_snap.w));
        s_snap.k = k; s_snap.anorm_g = an; s_snap.z = alt.z; s_snap.vz = alt.vz; s_snap.ab = alt.ab; s_snap.vbatt = s_vbatt;
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
    printf("STATUS motor_pins_level %d %d %d %d\n", gpio_get_level(16), gpio_get_level(6), gpio_get_level(5), gpio_get_level(15));
    printf("STATUS state %s\n", FS_STATE_NAME[s.st]);
    printf("STATUS vbatt %.3f\n", s.vbatt);
    printf("STATUS calib_valid %d\n", IMU_CALIB_VALID);
    printf("STATUS gyro_bias_dps %.3f %.3f %.3f\n", s_gbias[0], s_gbias[1], s_gbias[2]);
    printf("STATUS loops %lu late_gt1100us %lu exec_max_us %lu ctrl200_max_us %lu spi_fail %lu\n", (unsigned long)s.loops,
           (unsigned long)s.late, (unsigned long)s.exec_max_us, (unsigned long)s.ctrl_max_us, (unsigned long)s.spi_fail);
    printf("STATUS alt z %.3f vz %.3f acc_bias %.3f\n", s.z, s.vz, s.ab);
    printf("STATUS mapping");
    for (int i = 0; i < 4; i++) printf(" u%d->%s(%s,GPIO%d)", i, MOT_NAME_SIM[i], MOT_POS_SIM[i], MOT_PIN_SIM[i]);
    printf("\nSTATUS_END\n");
}

static void telem_task(void *arg)
{
    (void)arg;
    uint32_t ev_seen = 0;
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(20));
        snap_t s; portENTER_CRITICAL(&s_mux); s = s_snap; portEXIT_CRITICAL(&s_mux);
        if (s.ev_seq != ev_seen) {
            ev_seen = s.ev_seq;
            printf("EVT %lld %s %s\n", (long long)(s.ev_t * 1000.0f), FS_EV_NAME[s.ev], FS_STATE_NAME[s.st]);
        }
        float ro, pi, ya; euler(s.q, &ro, &pi, &ya);
        printf("F %lld %d %d %.1f %.1f %.1f %.1f %.3f %.3f %.2f %.2f %.3f %.3f %.3f %.3f %.3f\n", (long long)(s.t_us / 1000),
               s.st, s.mode, ro, pi, ya, s.w[2] * R2D, s.z, s.vz, s.vbatt, s.k,
               s.u_cmd[0], s.u_cmd[1], s.u_cmd[2], s.u_cmd[3], s.out_sum);
    }
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
            if (c == ' ' || c == 'k' || c == 'K') { if (n == 0) { s_cmd = FS_CMD_DISARM; continue; } }   /* 单键急停 */
            if (c == '\r' || c == '\n') {
                line[n] = 0; n = 0;
                if (!strcmp(line, "arm")) s_cmd = FS_CMD_ARM;
                else if (!strcmp(line, "disarm") || !strcmp(line, "stop")) s_cmd = FS_CMD_DISARM;
                else if (!strcmp(line, "go")) s_cmd = FS_CMD_GO;
                else if (!strcmp(line, "status")) print_status();
                else if (line[0]) printf("CLI unknown '%s'（可用：arm / disarm / go / status；空格或 k = 立即 disarm）\n", line);
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

    xTaskCreatePinnedToCore(baro_task, "baro", 4096, NULL, 6, NULL, 0);
    vTaskDelay(pdMS_TO_TICKS(60));
    xTaskCreatePinnedToCore(ctrl_task, "ctrl", 6144, NULL, configMAX_PRIORITIES - 2, NULL, 1);
    xTaskCreatePinnedToCore(cli_task, "cli", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(telem_task, "telem", 4096, NULL, 4, NULL, 0);
    vTaskDelay(pdMS_TO_TICKS(500));
    print_status();
    printf("FLT_READY\n");
}
