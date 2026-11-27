#!/usr/bin/env bash
# Stage 33 系留首飞（装桨、真飞，油门上限 70%）：参数原样传给 first_flight.py：[串口]  [--no-flash]  [--no-audio]  [--analyse-file X.txt]
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=""
for c in "$HOME"/.espressif/tools/python/*/venv/bin/python python3; do
    if "$c" -c "import serial, esptool" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "找不到带 pyserial + esptool 的 Python（应为 ~/.espressif/tools/python/v6.1/venv/bin/python）"; exit 1; fi
exec "$PY" first_flight.py "$@"
