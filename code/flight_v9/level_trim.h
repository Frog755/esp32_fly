/* level_trim.h —— IMU 安装水平修正，level_trim.py 2026-11-23 15:27:04 生成
 * 测量：(+1.500,+0.200) / 转 180° (+1.669,+0.821) / 复测 (+1.543,+0.373)；桌面坡度 (-0.084,-0.310)
 * 机体 = LEVEL_TRIM_R · (芯片→机体映射后的 IMU 读数)；生成新文件后需重新编译烧录。
 */
#pragma once
#define LEVEL_TRIM_VALID 1
#define LEVEL_TRIM_ROLL_DEG  1.584f
#define LEVEL_TRIM_PITCH_DEG 0.510f
static const float LEVEL_TRIM_R[3][3] = {{0.9999603f, 0.0002463f, 0.0089049f}, {0.0000000f, 0.9996177f, -0.0276490f}, {-0.0089083f, 0.0276479f, 0.9995780f}};
