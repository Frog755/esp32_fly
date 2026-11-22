/*
 * test_flip_table_c.c —— 宿主机 (Linux) 快速单元测试与比对验证
 *
 * 编译运行:
 *   gcc -O3 -Wall -Wextra -I. test_flip_table_c.c -lm -o test_flip_table_c
 *   ./test_flip_table_c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "flip_table.h"

int main()
{
    printf("=== 启动 ESP32-S3 3D 翻转表 C 驱动宿主机测试 ===\n");
    printf("有效格点总数: %d\n", FLIP_TABLE_VALID_COUNT);

    /* 1. 测试典型倒扣测试用例 (倾角 150°, 绕 X 轴) */
    float ang = 150.0f * (3.141592653589793f / 180.0f);
    float q_test[4] = {
        cosf(ang / 2.0f),
        sinf(ang / 2.0f),
        0.0f,
        0.0f
    };
    float w_test[3] = {0.0f, 0.0f, 0.0f};

    const flip_cell_t *cell = lookup_flip_cell(q_test, w_test);
    if (!cell) {
        printf("[FAIL] lookup_flip_cell 返回 NULL！\n");
        return 1;
    }

    printf("[PASS] 命中格点: phi=%.1f deg, tilt=%.1f deg, w_par=%.1f rad/s, tf=%.3f s\n",
           cell->phi, cell->tilt, cell->w_par, cell->tf);

    /* 2. 测试插值时序 */
    float dt = 0.005f;
    int steps = (int)(cell->tf / dt) + 5;
    float u[4];
    int active_steps = 0;

    for (int s = 0; s < steps; s++) {
        float t = s * dt;
        int active = interpolate_cell_throttle(cell, t, u);
        if (active) {
            active_steps++;
            for (int m = 0; m < 4; m++) {
                if (u[m] < 0.0f || u[m] > 1.0f) {
                    printf("[FAIL] 油门越界: step %d, u[%d] = %f\n", s, m, u[m]);
                    return 1;
                }
            }
        }
    }
    printf("[PASS] 插值时序验证成功: 有效执行步数 %d (%.3f s)\n", active_steps, active_steps * dt);

    /* 3. 测试吞吐量基准 (100,000 次连续查表测耗时) */
    clock_t t0 = clock();
    int niters = 100000;
    int hits = 0;
    for (int k = 0; k < niters; k++) {
        const flip_cell_t *dummy = lookup_flip_cell(q_test, w_test);
        if (dummy) hits++;
    }
    clock_t t1 = clock();
    double total_us = (double)(t1 - t0) / CLOCKS_PER_SEC * 1e6;
    double per_call_us = total_us / niters;

    printf("[PERF] %d 次查表总耗时: %.2f ms (单次查表平均: %.3f us, 频率: %.1f kHz, 命中: %d)\n",
           niters, total_us / 1000.0, per_call_us, 1000.0 / per_call_us, hits);

    /* 4. 测试接近水平态安全回退 */
    float q_level[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    if (lookup_flip_cell(q_level, w_test) != NULL) {
        printf("[FAIL] 水平姿态应返回 NULL (安全移交 v5 规则)！\n");
        return 1;
    }
    printf("[PASS] 水平姿态安全回退 (正确返回 NULL)\n");

    printf("\n=== 全部测试用例验证通过！ ===\n");
    return 0;
}
