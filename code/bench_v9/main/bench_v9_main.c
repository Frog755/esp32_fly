/**
 * bench_v9_main.c —— v9 控制器 + 门控 Mahony 上板实测（Stage 22）
 *
 *   T1 正确性：320 组真实飞行状态（gen_hil_vectors.py 从仿真闭环分层抽样）在芯片上跑 recovery_expert9_step，
 *              与 Python 期望值比较推力 / 油门 / 模式 / 积分器
 *   T2 正确性：512 组门控 Mahony 单次更新，比较四元数旋转角差
 *   T3 单次耗时：关中断逐次计时（CPU 周期），v9 / 门控 Mahony / v8 对比
 *   T4 实时循环：FreeRTOS 1 kHz 任务（每 tick 一次 Mahony，每 5 tick 一次 v9），10 s，统计唤醒抖动、执行时间、CPU 占用
 * 输出 "HIL <key> <value>" 行供主机解析，最后 "HIL_DONE"，之后每 3 s 重复打印一遍汇总。
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_timer.h"
#include "esp_chip_info.h"
#include "esp_idf_version.h"
#include "esp_system.h"
#include "sdkconfig.h"

#include "recovery_expert9.h"
#include "mahony_gated.h"
#include "recovery_expert8.h"
#include "hil_vectors.h"

#define MHZ CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static volatile float s_sink;
static char s_summary[4096];
static int s_sum_len;

#define OUT(...) do { printf(__VA_ARGS__); s_sum_len += snprintf(s_summary + s_sum_len, sizeof(s_summary) - s_sum_len, __VA_ARGS__); } while (0)

/* ---------------- T1 ---------------- */
static void t1_ctrl_correctness(void)
{
    double max_dT = 0, max_du = 0, max_dz = 0, max_du_nz = 0;
    int bb_mis = 0, nz_cnt = 0;
    for (int i = 0; i < HIL_NC; i++) {
        recovery9_state_t st = {hil_c_zin[i], hil_c_bbin[i]};
        float u[4];
        uint8_t bb = recovery_expert9_step(&st, &hil_c_q[4 * i], &hil_c_w[3 * i], &hil_c_pos[3 * i], &hil_c_vel[3 * i],
                                           hil_c_vz_est[i], hil_c_z_est[i], &hil_c_target[3 * i], hil_c_vb[i], u);
        if (bb != hil_c_bbout[i]) bb_mis++;
        double dz = fabs((double)st.z_int - (double)hil_c_zout[i]);
        if (dz > max_dz) max_dz = dz;
        double vr = (double)hil_c_vb[i] / (double)EXP9_VREF;
        for (int m = 0; m < 4; m++) {
            double T = (double)EXP9_TMAX * ((double)u[m] * vr) * ((double)u[m] * vr);
            double dT = fabs(T - (double)hil_c_T[4 * i + m]);
            double du = fabs((double)u[m] - (double)hil_c_u[4 * i + m]);
            if (dT > max_dT) max_dT = dT;
            if (u[m] > 1e-3f || hil_c_u[4 * i + m] > 1e-3f) { if (du > max_du) max_du = du; }
            else { if (du > max_du_nz) max_du_nz = du; if (du > 1e-4) nz_cnt++; }
        }
    }
    OUT("HIL t1_n %d\n", HIL_NC);
    OUT("HIL t1_max_dT_N %.3e\n", max_dT);
    OUT("HIL t1_max_du_live %.3e\n", max_du);
    OUT("HIL t1_max_du_nearzero %.3e\n", max_du_nz);
    OUT("HIL t1_nearzero_gt1e-4 %d\n", nz_cnt);
    OUT("HIL t1_max_dzint %.3e\n", max_dz);
    OUT("HIL t1_mode_mismatch %d\n", bb_mis);
}

/* ---------------- T2 ---------------- */
static double quat_angle_deg(const float a[4], const float b[4])
{
    /* r = conj(a) ⊗ b，角 = 2·atan2(|r_xyz|, |r_w|)，在 double 下算以免度量本身引入误差 */
    double aw = a[0], ax = -a[1], ay = -a[2], az = -a[3], bw = b[0], bx = b[1], by = b[2], bz = b[3];
    double rw = aw * bw - ax * bx - ay * by - az * bz;
    double rx = aw * bx + ax * bw + ay * bz - az * by;
    double ry = aw * by - ax * bz + ay * bw + az * bx;
    double rz = aw * bz + ax * by - ay * bx + az * bw;
    return 2.0 * atan2(sqrt(rx * rx + ry * ry + rz * rz), fabs(rw)) * 57.29577951308232;
}

static void t2_mahony_correctness(void)
{
    double mx = 0, sum = 0;
    for (int i = 0; i < HIL_NM; i++) {
        mahony_gated_t m;
        memcpy(m.q, &hil_m_q[4 * i], sizeof(m.q));
        mahony_gated_update(&m, &hil_m_w[3 * i], &hil_m_f[3 * i], 0.001f);
        double d = quat_angle_deg(m.q, &hil_m_qout[4 * i]);
        if (d > mx) mx = d;
        sum += d;
    }
    OUT("HIL t2_n %d\n", HIL_NM);
    OUT("HIL t2_max_deg %.3e\n", mx);
    OUT("HIL t2_mean_deg %.3e\n", sum / HIL_NM);
}

/* ---------------- T3 ---------------- */
typedef struct { uint32_t mn, mx; uint64_t sum; uint32_t n; } cyc_t;
static void cyc_add(cyc_t *c, uint32_t v) { if (v < c->mn) c->mn = v; if (v > c->mx) c->mx = v; c->sum += v; c->n++; }
static void cyc_print(const char *name, cyc_t *c, uint32_t overhead)
{
    double mn = (double)(c->mn - overhead), mx = (double)(c->mx - overhead), avg = (double)c->sum / c->n - overhead;
    OUT("HIL t3_%s_cycles_min %.0f\n", name, mn);
    OUT("HIL t3_%s_cycles_avg %.1f\n", name, avg);
    OUT("HIL t3_%s_cycles_max %.0f\n", name, mx);
    OUT("HIL t3_%s_us_avg %.3f\n", name, avg / MHZ);
    OUT("HIL t3_%s_us_max %.3f\n", name, mx / MHZ);
}

static void t3_timing(void)
{
    const int PASSES = 20;
    cyc_t c9 = {UINT32_MAX, 0, 0, 0}, cm = {UINT32_MAX, 0, 0, 0}, c8 = {UINT32_MAX, 0, 0, 0}, ce = {UINT32_MAX, 0, 0, 0};
    float u[4];
    for (int p = 0; p < PASSES; p++) {
        for (int i = 0; i < HIL_NC; i++) {
            portENTER_CRITICAL(&s_mux);
            uint32_t a = esp_cpu_get_cycle_count();
            uint32_t b = esp_cpu_get_cycle_count();
            portEXIT_CRITICAL(&s_mux);
            cyc_add(&ce, b - a);
        }
        for (int i = 0; i < HIL_NC; i++) {
            recovery9_state_t st = {hil_c_zin[i], hil_c_bbin[i]};
            portENTER_CRITICAL(&s_mux);
            uint32_t a = esp_cpu_get_cycle_count();
            recovery_expert9_step(&st, &hil_c_q[4 * i], &hil_c_w[3 * i], &hil_c_pos[3 * i], &hil_c_vel[3 * i],
                                  hil_c_vz_est[i], hil_c_z_est[i], &hil_c_target[3 * i], hil_c_vb[i], u);
            uint32_t b = esp_cpu_get_cycle_count();
            portEXIT_CRITICAL(&s_mux);
            s_sink += u[0];
            cyc_add(&c9, b - a);
        }
        for (int i = 0; i < HIL_NC; i++) {
            float zi = hil_c_zin[i];
            portENTER_CRITICAL(&s_mux);
            uint32_t a = esp_cpu_get_cycle_count();
            recovery_expert8_step(&hil_c_q[4 * i], &hil_c_w[3 * i], &hil_c_pos[3 * i], &hil_c_vel[3 * i],
                                  hil_c_vz_est[i], hil_c_z_est[i], &hil_c_target[3 * i], hil_c_vb[i], &zi, 0.005f, u);
            uint32_t b = esp_cpu_get_cycle_count();
            portEXIT_CRITICAL(&s_mux);
            s_sink += u[0];
            cyc_add(&c8, b - a);
        }
        for (int i = 0; i < HIL_NM; i++) {
            mahony_gated_t m;
            memcpy(m.q, &hil_m_q[4 * i], sizeof(m.q));
            portENTER_CRITICAL(&s_mux);
            uint32_t a = esp_cpu_get_cycle_count();
            mahony_gated_update(&m, &hil_m_w[3 * i], &hil_m_f[3 * i], 0.001f);
            uint32_t b = esp_cpu_get_cycle_count();
            portEXIT_CRITICAL(&s_mux);
            s_sink += m.q[0];
            cyc_add(&cm, b - a);
        }
    }
    uint32_t ov = ce.mn;
    OUT("HIL t3_overhead_cycles %u\n", (unsigned)ov);
    cyc_print("v9", &c9, ov);
    cyc_print("mahony", &cm, ov);
    cyc_print("v8", &c8, ov);
}

/* ---------------- T4 ---------------- */
#define T4_SECONDS 10
static volatile int s_t4_done;
static int64_t s_t4_period_min, s_t4_period_max, s_t4_period_sum; static int s_t4_period_n, s_t4_late;
static uint32_t s_t4_exec_max_cyc; static uint64_t s_t4_busy_cyc; static int s_t4_ctrl_calls;

static void t4_task(void *arg)
{
    (void)arg;
    mahony_gated_t m; mahony_gated_init(&m, &hil_m_q[0]);
    recovery9_state_t st; recovery9_reset(&st);
    float u[4];
    const int ticks = T4_SECONDS * 1000;
    TickType_t last = xTaskGetTickCount();
    int64_t prev = 0;
    s_t4_period_min = INT64_MAX; s_t4_period_max = 0;
    uint64_t t_start = 0;
    for (int k = 0; k < ticks; k++) {
        vTaskDelayUntil(&last, 1);
        int64_t now = esp_timer_get_time();
        if (k == 0) t_start = (uint64_t)now;
        if (k > 0) {
            int64_t per = now - prev;
            if (per < s_t4_period_min) s_t4_period_min = per;
            if (per > s_t4_period_max) s_t4_period_max = per;
            if (per > 1100) s_t4_late++;
            s_t4_period_sum += per; s_t4_period_n++;
        }
        prev = now;
        uint32_t a = esp_cpu_get_cycle_count();
        int i = k % HIL_NM;
        mahony_gated_update(&m, &hil_m_w[3 * i], &hil_m_f[3 * i], 0.001f);
        if (k % 5 == 0) {
            int j = (k / 5) % HIL_NC;
            recovery_expert9_step(&st, m.q, &hil_c_w[3 * j], &hil_c_pos[3 * j], &hil_c_vel[3 * j], hil_c_vz_est[j],
                                  hil_c_z_est[j], &hil_c_target[3 * j], hil_c_vb[j], u);
            s_sink += u[0];
            s_t4_ctrl_calls++;
        }
        uint32_t b = esp_cpu_get_cycle_count();
        uint32_t e = b - a;
        if (e > s_t4_exec_max_cyc) s_t4_exec_max_cyc = e;
        s_t4_busy_cyc += e;
    }
    int64_t total_us = esp_timer_get_time() - (int64_t)t_start;
    OUT("HIL t4_seconds %d\n", T4_SECONDS);
    OUT("HIL t4_freertos_hz %d\n", CONFIG_FREERTOS_HZ);
    OUT("HIL t4_ticks %d\n", ticks);
    OUT("HIL t4_ctrl_calls %d\n", s_t4_ctrl_calls);
    OUT("HIL t4_period_us_avg %.2f\n", (double)s_t4_period_sum / s_t4_period_n);
    OUT("HIL t4_period_us_min %lld\n", (long long)s_t4_period_min);
    OUT("HIL t4_period_us_max %lld\n", (long long)s_t4_period_max);
    OUT("HIL t4_late_gt1100us %d\n", s_t4_late);
    OUT("HIL t4_exec_us_max %.3f\n", (double)s_t4_exec_max_cyc / MHZ);
    OUT("HIL t4_cpu_load_pct %.4f\n", 100.0 * ((double)s_t4_busy_cyc / MHZ) / (double)total_us);
    s_t4_done = 1;
    vTaskDelete(NULL);
}

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(3000));       /* 给主机留出打开串口的时间 */
    esp_chip_info_t ci; esp_chip_info(&ci);
    OUT("HIL_BEGIN\n");
    OUT("HIL idf %s\n", esp_get_idf_version());
    OUT("HIL chip_model %d\n", (int)ci.model);
    OUT("HIL chip_rev %d\n", (int)ci.revision);
    OUT("HIL cores %d\n", (int)ci.cores);
    OUT("HIL cpu_mhz %d\n", MHZ);
    t1_ctrl_correctness();
    t2_mahony_correctness();
    t3_timing();
    xTaskCreatePinnedToCore(t4_task, "t4", 4096, NULL, configMAX_PRIORITIES - 2, NULL, 1);
    while (!s_t4_done) vTaskDelay(pdMS_TO_TICKS(100));
    OUT("HIL_DONE\n");
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(3000));
        printf("\n==== 汇总（每 3 s 重复）====\n%s", s_summary);
    }
}
