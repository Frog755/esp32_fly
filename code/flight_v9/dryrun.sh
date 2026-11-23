#!/usr/bin/env bash
# 用 ESP-IDF 自带的 Python（带 pyserial）运行飞控干跑检查；参数原样传给 dryrun_check.py（可选：串口，默认 /dev/ttyACM0）
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=""
for c in "$HOME"/.espressif/tools/python/*/venv/bin/python python3; do
    if "$c" -c "import serial" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "找不到带 pyserial 的 Python，请先 pip install --user pyserial"; exit 1; fi
exec "$PY" dryrun_check.py "$@"
