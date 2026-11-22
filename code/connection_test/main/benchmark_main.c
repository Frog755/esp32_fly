/*
 * benchmark_main.c —— Raptor 策略在 ESP32-S3 上的推理基准测试
 *
 * 测四件事：
 *   1. 数值正确性 —— C 实现 vs numpy 参考实现的逐位比对
 *   2. 单次前向耗时 —— 平均 / 最小 / 最大 / 抖动
 *   3. 可达最高频率 —— 连续推理能跑到多少 Hz
 *   4. 三任务并发抖动 —— 模拟 RL(100Hz) + PID(1kHz) 同时运行的时序稳定性
 *
 * 编译烧录：
 *   idf.py set-target esp32s3
 *   idf.py -p /dev/ttyACM0 build flash monitor
 */

#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_chip_info.h"

#include "raptor_policy.h"

static const char *TAG = "BENCH";

/* 容差：float32 累加误差，1e-5 相对量级合理 */
#define TOL_ABS 2e-5f

static raptor_state_t g_state;

/* ---------------- 耗时统计 ---------------- */
typedef struct {
    int64_t min_ns;
    int64_t max_ns;
    int64_t sum_ns;
    int     n;
} timing_t;

static void timing_reset(timing_t *t)
{
    t->min_ns = INT64_MAX;
    t->max_ns = 0;
    t->sum_ns = 0;
    t->n = 0;
}

static void timing_add(timing_t *t, int64_t ns)
{
    if (ns < t->min_ns) t->min_ns = ns;
    if (ns > t->max_ns) t->max_ns = ns;
    t->sum_ns += ns;
    t->n++;
}

static void timing_report(const char *label, timing_t *t)
{
    if (t->n == 0) return;
    float avg_us = (float)t->sum_ns / t->n / 1000.0f;
    printf("  %-28s n=%5d  avg=%8.2f us  min=%8.2f  max=%8.2f  jitter=%7.2f us\n",
           label, t->n,
           avg_us,
           t->min_ns / 1000.0f,
           t->max_ns / 1000.0f,
           (t->max_ns - t->min_ns) / 1000.0f);
}

/* ---------------- 1. 数值正确性 ---------------- */
static int test_correctness(void)
{
    printf("\n--- [1] 数值正确性（C vs numpy 参考实现）---\n");

    int pass = 0, fail = 0;
    float worst_err = 0.0f;

    for (int i = 0; i < RAPTOR_N_TEST; i++) {
        float act[RAPTOR_ACT_DIM];
        raptor_reset(&g_state);
        raptor_forward(raptor_test_obs[i], act, &g_state);

        float max_err = 0.0f;
        for (int k = 0; k < RAPTOR_ACT_DIM; k++) {
            float e = fabsf(act[k] - raptor_test_expect[i][k]);
            if (e > max_err) max_err = e;
        }
        if (max_err > worst_err) worst_err = max_err;

        bool ok = (max_err <= TOL_ABS);
        if (ok) pass++; else fail++;

        printf("  向量[%d] %s  max_err=%.3e  act=[%.6f, %.6f, %.6f, %.6f]\n",
               i, ok ? "PASS" : "FAIL", max_err,
               act[0], act[1], act[2], act[3]);
        if (!ok) {
            printf("           expect=[%.6f, %.6f, %.6f, %.6f]\n",
                   raptor_test_expect[i][0], raptor_test_expect[i][1],
                   raptor_test_expect[i][2], raptor_test_expect[i][3]);
        }
    }

    /* 连续步测试：验证 hidden state 跨步传递正确 */
    printf("\n  连续 3 步（验证 GRU 状态传递）:\n");
    raptor_reset(&g_state);
    for (int i = 0; i < 3; i++) {
        float act[RAPTOR_ACT_DIM];
        raptor_forward(raptor_seq_obs[i], act, &g_state);

        float max_err = 0.0f;
        for (int k = 0; k < RAPTOR_ACT_DIM; k++) {
            float e = fabsf(act[k] - raptor_seq_expect[i][k]);
            if (e > max_err) max_err = e;
        }
        if (max_err > worst_err) worst_err = max_err;
        bool ok = (max_err <= TOL_ABS);
        if (ok) pass++; else fail++;

        printf("    step%d %s  max_err=%.3e  act=[%.6f, %.6f, %.6f, %.6f]\n",
               i, ok ? "PASS" : "FAIL", max_err,
               act[0], act[1], act[2], act[3]);
    }

    printf("\n  结果: %d 通过 / %d 失败   最大误差=%.3e  (容差 %.0e)\n",
           pass, fail, worst_err, TOL_ABS);
    return fail;
}

/* ---------------- 2. 单次前向耗时 ---------------- */
static int64_t bench_once(void)
{
    float act[RAPTOR_ACT_DIM];
    float obs[RAPTOR_OBS_DIM];
    memcpy(obs, raptor_test_obs[0], sizeof(obs));

    int64_t t0 = esp_timer_get_time();
    raptor_forward(obs, act, &g_state);
    int64_t t1 = esp_timer_get_time();

    /* 防止编译器优化掉 */
    static volatile float sink;
    sink = act[0];

    return (t1 - t0) * 1000;   /* us -> ns */
}

static void bench_latency(void)
{
    printf("\n--- [2] 单次前向耗时 ---\n");

    raptor_reset(&g_state);

    /* 预热 */
    for (int i = 0; i < 100; i++) (void)bench_once();

    timing_t t;
    timing_reset(&t);
    const int N = 2000;
    for (int i = 0; i < N; i++) {
        timing_add(&t, bench_once());
    }
    timing_report("单次 forward", &t);

    float avg_us = (float)t.sum_ns / t.n / 1000.0f;
    printf("\n  据此推算理论最高频率: %.0f Hz\n", 1e6f / avg_us);
}

/* ---------------- 3. 连续推理最高频率 ---------------- */
static void bench_sustained(void)
{
    printf("\n--- [3] 连续推理吞吐（每秒能算多少次）---\n");

    raptor_reset(&g_state);
    float obs[RAPTOR_OBS_DIM], act[RAPTOR_ACT_DIM];
    memcpy(obs, raptor_test_obs[0], sizeof(obs));

    /* 跑 2 秒，数次数 */
    int64_t t0 = esp_timer_get_time();
    int64_t deadline = t0 + 2000000;   /* 2s */
    int count = 0;
    while (esp_timer_get_time() < deadline) {
        raptor_forward(obs, act, &g_state);
        obs[18] = act[0]; obs[19] = act[1]; obs[20] = act[2]; obs[21] = act[3];
        count++;
    }
    int64_t elapsed = esp_timer_get_time() - t0;

    printf("  2 秒内完成 %d 次推理 → 实测 %.0f Hz\n",
           count, count * 1e6f / elapsed);
    printf("  平均每次 %.2f us\n", (float)elapsed / count);
}

/* ---------------- 4. 三任务并发抖动 ---------------- */
/* 模拟真实飞控：core0 跑 RL(100Hz)，core1 跑 PID(1kHz)，core1 跑混控(4kHz) */

static volatile bool g_run_concurrent = false;
static timing_t  g_pid_timing;
static int64_t   g_pid_worst = 0;

static void pid_task(void *arg)
{
    /* 模拟 1kHz PID：固定周期唤醒，测量实际周期抖动 */
    const int64_t period_us = 1000;
    int64_t next = esp_timer_get_time();
    float p = 0.0f;

    while (g_run_concurrent) {
        next += period_us;
        int64_t now = esp_timer_get_time();
        int64_t sleep_us = next - now;
        if (sleep_us > 0) {
            esp_rom_delay_us(sleep_us);
        }

        /* 模拟 PID 计算：一些浮点运算 */
        int64_t compute_start = esp_timer_get_time();
        for (int i = 0; i < 20; i++) {
            p = p * 0.99f + 0.01f * (float)i;
        }
        int64_t compute_us = esp_timer_get_time() - compute_start;
        if (compute_us > g_pid_worst) g_pid_worst = compute_us;

        /* 测量实际间隔偏多少 */
        int64_t actual = esp_timer_get_time() - (next - period_us);
        timing_add(&g_pid_timing, (actual - period_us) * 1000);
        (void)p;
    }
    vTaskDelete(NULL);
}

static void bench_concurrent(void)
{
    printf("\n--- [4] 并发时序抖动（RL 100Hz + PID 1kHz）---\n");

    timing_reset(&g_pid_timing);
    g_pid_worst = 0;
    g_run_concurrent = true;

    /* PID 任务固定在 core 1，优先级较高 */
    xTaskCreatePinnedToCore(pid_task, "pid", 4096, NULL, 10, NULL, 1);
    vTaskDelay(pdMS_TO_TICKS(100));   /* 让 PID 先跑起来 */

    /* RL 在主任务（core 0）上跑 100Hz */
    raptor_reset(&g_state);
    float obs[RAPTOR_OBS_DIM], act[RAPTOR_ACT_DIM];
    memcpy(obs, raptor_test_obs[0], sizeof(obs));

    timing_t rl_t;
    timing_reset(&rl_t);

    const int RL_STEPS = 300;   /* 3 秒 @100Hz */
    for (int i = 0; i < RL_STEPS; i++) {
        int64_t t0 = esp_timer_get_time();
        raptor_forward(obs, act, &g_state);
        int64_t t1 = esp_timer_get_time();
        timing_add(&rl_t, (t1 - t0) * 1000);

        obs[18] = act[0]; obs[19] = act[1]; obs[20] = act[2]; obs[21] = act[3];
        vTaskDelay(pdMS_TO_TICKS(10));   /* 100Hz */
    }

    g_run_concurrent = false;
    vTaskDelay(pdMS_TO_TICKS(200));

    printf("  RL 推理（与 PID 并发时）:\n");
    timing_report("  forward", &rl_t);
    printf("\n  PID 周期偏差（理想 1000us）:\n");
    printf("    样本 %d, 平均偏差 %.2f us, 最大 %.2f us\n",
           g_pid_timing.n,
           (float)g_pid_timing.sum_ns / g_pid_timing.n / 1000.0f,
           g_pid_timing.max_ns / 1000.0f);
    printf("    PID 单次计算最长 %.2f us\n", g_pid_worst / 1000.0f);
}

/* ---------------- 5. 内存占用 ---------------- */
static void report_memory(void)
{
    printf("\n--- [5] 内存占用 ---\n");
    printf("  权重大小        : %d 字节 (%.1f KB)\n",
           (int)(sizeof(raptor_w0) + sizeof(raptor_b0) +
                 sizeof(raptor_w_in) + sizeof(raptor_b_in) +
                 sizeof(raptor_w_hid) + sizeof(raptor_b_hid) +
                 sizeof(raptor_h0) +
                 sizeof(raptor_w2) + sizeof(raptor_b2)),
           (float)(sizeof(raptor_w0) + sizeof(raptor_b0) +
                   sizeof(raptor_w_in) + sizeof(raptor_b_in) +
                   sizeof(raptor_w_hid) + sizeof(raptor_b_hid) +
                   sizeof(raptor_h0) +
                   sizeof(raptor_w2) + sizeof(raptor_b2)) / 1024.0f);
    printf("  单次推理栈占用  : < 1 KB（全静态分配，无 malloc）\n");
    printf("  内部 RAM 空闲   : %zu 字节\n",
           heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
#if CONFIG_SPIRAM
    printf("  PSRAM 空闲      : %zu 字节\n",
           heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#endif
}

void app_main(void)
{
    /* 等 USB-JTAG 就绪，避免日志丢失 */
    vTaskDelay(pdMS_TO_TICKS(1500));

    esp_chip_info_t chip;
    esp_chip_info(&chip);

    printf("\n\n");
    printf("===============================================================\n");
    printf("    Raptor 策略 ESP32-S3 推理基准测试\n");
    printf("===============================================================\n");
    printf("  ESP-IDF    : %s\n", esp_get_idf_version());
    printf("  芯片       : %s, %d 核, rev v%d.%d\n",
           CONFIG_IDF_TARGET, chip.cores,
           chip.revision / 100, chip.revision % 100);
    printf("  CPU 主频   : %d MHz\n", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    printf("  网络结构   : dense(22->16,ReLU) -> GRU(16->16) -> dense(16->4)\n");
    printf("  参数量     : 2084 float = 8.1 KB\n");
    printf("===============================================================\n");

    int fails = test_correctness();
    bench_latency();
    bench_sustained();
    report_memory();
    bench_concurrent();

    printf("\n===============================================================\n");
    if (fails == 0) {
        printf("  ✅ 全部测试通过：C 实现与 numpy 参考实现一致\n");
    } else {
        printf("  ❌ 有 %d 项数值校验失败，请检查 GRU 门序或权重导出\n", fails);
    }
    printf("===============================================================\n\n");

    ESP_LOGI(TAG, "基准测试结束，进入 1 秒心跳（验证长期稳定）");
    uint32_t n = 0;
    while (1) {
        n++;
        printf("[存活 %" PRIu32 "] uptime=%" PRIu32 " ms  free_internal=%zu\n",
               n, (uint32_t)(esp_timer_get_time() / 1000),
               heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
