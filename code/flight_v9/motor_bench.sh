#!/usr/bin/env bash
# Stage 27 拆桨电机台架（电机会真转，油门上限 6%）：用 ESP-IDF 自带的 Python（带 pyserial + esptool）运行 motor_bench.py
# 参数原样传给 motor_bench.py：[串口，默认 /dev/ttyACM0]  [--no-flash]
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=""
for c in "$HOME"/.espressif/tools/python/*/venv/bin/python python3; do
    if "$c" -c "import serial, esptool" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "找不到带 pyserial + esptool 的 Python（应为 ~/.espressif/tools/python/v6.1/venv/bin/python）"; exit 1; fi
exec "$PY" motor_bench.py "$@"
