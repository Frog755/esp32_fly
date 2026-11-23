cmake_minimum_required(VERSION 3.22)
# sensor_test —— 耀坤 K1.0 只读传感器实测（Stage 23）：不含任何 PWM 代码，电机引脚上电即拉低锁存
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(sensor_test)
