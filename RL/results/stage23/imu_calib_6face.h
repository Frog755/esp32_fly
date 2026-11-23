/* 自动生成：calib_interactive.py 2026-11-23 13:52:14 —— 加速度计六面标定 + 芯片→机体 FLU 映射 */
#pragma once
#define IMU_CALIB_VALID 1
/* 校正：a_chip = (raw_g - OFF) / SCALE；然后 body = R · chip（陀螺只做 R 映射，零偏另行开机静止校准） */
static const float IMU_ACC_OFF_G[3] = {0.05986f, 0.00482f, 0.17065f};
static const float IMU_ACC_SCALE[3] = {1.00071f, 0.99790f, 1.00877f};
static const int8_t IMU_CHIP_TO_BODY[3][3] = {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}};
