/*
 * test_recovery_expert6_c.c —— v6 闭环专家纯 C 实现宿主机基准测试
 *
 * 编译与测试:
 *   gcc -O3 -Wall -Wextra -Werror -I. test_recovery_expert6_c.c -lm -o test_recovery_expert6_c
 *   ./test_recovery_expert6_c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "recovery_expert6.h"

int main()
{
    printf("=== 启动 ESP32-S3 v6 闭环自稳专家 C 驱动基准测试 ===\n");

    /* 1. 典型测试用例: 深倒扣 150°，高速角速度 12 rad/s */
    float ang = 150.0f * (3.141592653589793f / 180.0f);
    float q[4] = {
        cosf(ang / 2.0f),
        sinf(ang / 2.0f),
        0.0f,
        0.0f
    };
    float w[3] = {12.0f, -2.0f, 1.0f};
    float vz = -2.5f;
    float z = 0.5f;
    float v_batt = 3.7f;
    float u[4];

    recovery_expert6_step(q, w, vz, z, v_batt, u);
    printf("[PASS] 基础功能测试:\n");
    printf("  输入: tilt=150.0 deg, wx=12.0, vz=-2.5 m/s, vbatt=3.7V\n");
    printf("  输出: u0=%.4f, u1=%.4f, u2=%.4f, u3=%.4f\n", u[0], u[1], u[2], u[3]);

    for (int i = 0; i < 4; i++) {
        if (u[i] < 0.0f || u[i] > 1.0f) {
            printf("[FAIL] 油门越界: u[%d] = %f\n", i, u[i]);
            return 1;
        }
    }

    /* 2. 吞吐量与耗时基准 (连续 1,000,000 次推理) */
    int niters = 1000000;
    clock_t t0 = clock();
    float dummy_sum = 0.0f;
    for (int k = 0; k < niters; k++) {
        recovery_expert6_step(q, w, vz, z, v_batt, u);
        dummy_sum += u[0];
    }
    clock_t t1 = clock();
    double total_us = (double)(t1 - t0) / CLOCKS_PER_SEC * 1e6;
    double per_call_ns = (total_us * 1000.0) / niters;

    printf("\n[PERF] 性能基准测试 (%d 次连续调用):\n", niters);
    printf("  总耗时: %.2f ms\n", total_us / 1000.0);
    printf("  单步耗时: %.1f ns (0.0%d us)\n", per_call_ns, (int)(per_call_ns / 1000.0));
    printf("  最高吞吐率: %.2f MHz (万次/秒: %.1f)\n", 1000.0 / (per_call_ns), (1e9 / per_call_ns) / 10000.0);
    printf("  (防止优化桩: dummy_sum = %.2f)\n", dummy_sum);

    printf("\n=== 全部测试用例验证通过！ ===\n");
    return 0;
}

