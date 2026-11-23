cmake_minimum_required(VERSION 3.22)
# bench_v9 —— v9 控制器 + 门控 Mahony 上板实测工程（Stage 22）
# 头文件直接引用 ../connection_test/main/ 下的 recovery_expert9.h / mahony_gated.h / recovery_expert8.h，不复制
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(bench_v9)
