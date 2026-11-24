#!/usr/bin/env bash
# 读出机载黑匣子（Stage 28）：落地后插 USB（别拔电池）运行；参数原样传给 bb_read.py：[串口]  [--file 已保存的文本]
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=""
for c in "$HOME"/.espressif/tools/python/*/venv/bin/python python3; do
    if "$c" -c "import serial" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "找不到带 pyserial 的 Python（应为 ~/.espressif/tools/python/v6.1/venv/bin/python）"; exit 1; fi
exec "$PY" bb_read.py "$@"
