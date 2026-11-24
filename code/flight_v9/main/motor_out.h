/**
 * motor_out.h —— 控制器输出（仿真电机顺序）→ 板上电机
 *
 * 映射依据（.devlog/STATE.md「电机映射」，2026-11-17 拆桨实测 + 仿真 rotor_pos/rotor_dir）：
 *   out_u[0] = FR 右前（逆时针）→ M1 / GPIO6
 *   out_u[1] = BR 右后（顺时针）→ M2 / GPIO5
 *   out_u[2] = BL 左后（逆时针）→ M3 / GPIO15
 *   out_u[3] = FL 左前（顺时针）→ M4 / GPIO16
 * 偏航无需额外取反（仿真 rotor_dir=(+1,−1,+1,−1) 与实物转向一致）。
 *
 * FLIGHT_MOTOR_OUTPUT（CMake：idf.py -B build_mc -DFLIGHT_MOTOR_OUTPUT=1 -DFLIGHT_U_MAX=0.06 build）：
 *   0（默认，干跑版）：不编译任何 LEDC 代码，电机引脚由 motors_lock_low() 拉低并 gpio_hold 锁存
 *   1（电机驱动版）：LEDC 20 kHz / 10 bit 驱动（耀坤 bench_test 同参数），**必须显式给出油门上限 FLIGHT_U_MAX**，
 *      否则编译报错——防止误编出不限油门的固件。Stage 27 拆桨台架用 0.06（11-17 实测 6% 不会触发 USB 复位，12% 会）。
 */
#ifndef MOTOR_OUT_H
#define MOTOR_OUT_H

#ifndef FLIGHT_MOTOR_OUTPUT
#define FLIGHT_MOTOR_OUTPUT 0
#endif
#if FLIGHT_MOTOR_OUTPUT && !defined(FLIGHT_U_MAX)
#error "电机驱动版必须显式指定油门上限：idf.py -B build_mc -DFLIGHT_MOTOR_OUTPUT=1 -DFLIGHT_U_MAX=0.06 build"
#endif
#ifndef FLIGHT_U_MAX
#define FLIGHT_U_MAX 1.0f          /* 干跑版不使用（不写电机），仅占位 */
#endif
#if FLIGHT_MOTOR_OUTPUT
_Static_assert(FLIGHT_U_MAX > 0.0f && FLIGHT_U_MAX <= 1.0f, "FLIGHT_U_MAX 必须在 (0, 1]");
#endif

static const int         MOT_PIN_SIM[4]  = {6, 5, 15, 16};
static const char *const MOT_NAME_SIM[4] = {"M1", "M2", "M3", "M4"};
static const char *const MOT_POS_SIM[4]  = {"右前", "右后", "左后", "左前"};

#if FLIGHT_MOTOR_OUTPUT
#include "driver/ledc.h"
#include "driver/gpio.h"

#define MOT_PWM_HZ    20000
#define MOT_PWM_BITS  LEDC_TIMER_10_BIT
#define MOT_PWM_MAX   1023

static volatile int s_motor_killed = 0;          /* 置 1 后永远写 0（看门狗 / 致命错误），需重启恢复 */

static void motor_out_init(void)
{
    ledc_timer_config_t t = {.speed_mode = LEDC_LOW_SPEED_MODE, .duty_resolution = MOT_PWM_BITS,
                             .timer_num = LEDC_TIMER_0, .freq_hz = MOT_PWM_HZ, .clk_cfg = LEDC_AUTO_CLK};
    ESP_ERROR_CHECK(ledc_timer_config(&t));
    for (int i = 0; i < 4; i++) {
        gpio_hold_dis(MOT_PIN_SIM[i]);
        ledc_channel_config_t c = {.gpio_num = MOT_PIN_SIM[i], .speed_mode = LEDC_LOW_SPEED_MODE, .channel = (ledc_channel_t)i,
                                   .timer_sel = LEDC_TIMER_0, .duty = 0, .hpoint = 0};
        ESP_ERROR_CHECK(ledc_channel_config(&c));
    }
}

/* 写 4 路占空比；返回实际写入的占空比之和（已按 FLIGHT_U_MAX 限幅、已挡掉 NaN） */
static inline float motor_out_write(const float u_sim[4])
{
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        float u = s_motor_killed ? 0.0f : u_sim[i];
        u = (u > FLIGHT_U_MAX) ? FLIGHT_U_MAX : (u > 0.0f ? u : 0.0f);   /* 同时挡掉 NaN */
        const uint32_t d = (uint32_t)(u * MOT_PWM_MAX + 0.5f);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)i, d);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)i);
        sum += (float)d / MOT_PWM_MAX;
    }
    return sum;
}

/* 强制停转并锁死（任何核心可调用）：控制循环卡住时由遥测任务调用 */
static void motor_out_kill(void)
{
    s_motor_killed = 1;
    for (int i = 0; i < 4; i++) ledc_stop(LEDC_LOW_SPEED_MODE, (ledc_channel_t)i, 0);   /* 输出固定低电平 */
}
static inline int motor_out_killed(void) { return s_motor_killed; }
#else
static inline void motor_out_init(void) {}                                 /* 引脚保持锁低 */
static inline float motor_out_write(const float u_sim[4]) { (void)u_sim; return 0.0f; }
static inline void motor_out_kill(void) {}
static inline int motor_out_killed(void) { return 0; }
#endif

#endif /* MOTOR_OUT_H */
