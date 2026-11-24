#!/usr/bin/env bash
# 主机单测：控制器+电机映射方向、安全状态机、竖直互补滤波（不需要板子）
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p ../build_host
gcc -O2 -Wall -Wextra -Wno-unused-function -I../main -I../../connection_test/main -o ../build_host/host_test host_test.c -lm && ../build_host/host_test && echo && echo "[5] 手机网页脚本" && node page_test.js && echo && echo "[6] 拆桨台架脚本判据" && python3 bench_logic_test.py
