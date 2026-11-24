#!/usr/bin/env bash
# Stage 30 装桨振动检查（电机会真转 30%，飞机离不了地）：参数原样传给 vib_check.py：[串口]  [--no-flash]  [--analyse-file X.txt]
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=""
for c in "$HOME"/.espressif/tools/python/*/venv/bin/python python3; do
    if "$c" -c "import serial, esptool" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "找不到带 pyserial + esptool 的 Python（应为 ~/.espressif/tools/python/v6.1/venv/bin/python）"; exit 1; fi
exec "$PY" vib_check.py "$@"
