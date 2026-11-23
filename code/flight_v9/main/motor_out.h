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
 * FLIGHT_MOTOR_OUTPUT（CMake：idf.py -DFLIGHT_MOTOR_OUTPUT=1）：
 *   0（默认，Stage 24 唯一烧录过的配置）：不编译任何 LEDC 代码，电机引脚由 motors_lock_low() 拉低并 gpio_hold 锁存
 *   1：LEDC 20 kHz / 10 bit 驱动（耀坤 bench_test 同参数）——仅做过编译检查，未烧录、未上电验证
 */
#ifndef MOTOR_OUT_H
#define MOTOR_OUT_H

#ifndef FLIGHT_MOTOR_OUTPUT
#define FLIGHT_MOTOR_OUTPUT 0
#endif
#ifndef FLIGHT_U_MAX
#define FLIGHT_U_MAX 1.0f          /* 油门上限（系留测试时可调低） */
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

static inline void motor_out_write(const float u_sim[4])
{
    for (int i = 0; i < 4; i++) {
        float u = u_sim[i];
        u = (u > FLIGHT_U_MAX) ? FLIGHT_U_MAX : (u > 0.0f ? u : 0.0f);   /* 同时挡掉 NaN */
        ledc_set_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)i, (uint32_t)(u * MOT_PWM_MAX + 0.5f));
        ledc_update_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)i);
    }
}
#else
static inline void motor_out_init(void) {}                                 /* 引脚保持锁低 */
static inline void motor_out_write(const float u_sim[4]) { (void)u_sim; }
#endif

#endif /* MOTOR_OUT_H */
