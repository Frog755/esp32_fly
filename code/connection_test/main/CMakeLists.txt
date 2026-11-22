# 当前编译入口：
#   benchmark_main.c  —— Raptor 策略推理基准测试（当前使用）
#   conn_test.c       —— 开发板自检程序（保留，切换入口时改这里）
#
# 切换方式：把 SRCS 换成 "conn_test.c" 即可。
idf_component_register(SRCS "benchmark_main.c"
                    INCLUDE_DIRS ".")
