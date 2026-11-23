#!/usr/bin/env python3
"""hil_read.py —— 读取 bench_v9 固件的串口输出，解析 HIL 行，与主机侧阈值比较，写结果 JSON（Stage 22）。
用法：python3 hil_read.py /dev/ttyACM0 out.json [timeout_s]"""
import sys, time, json, re
import serial

port, out = sys.argv[1], sys.argv[2]
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 60
ser = serial.Serial(port, 115200, timeout=0.5)
buf = ''; t0 = time.time(); kv = {}; began = False
while time.time() - t0 < timeout:
    chunk = ser.read(4096).decode('utf-8', 'replace')
    if chunk:
        buf += chunk
        while '\n' in buf:
            line, buf = buf.split('\n', 1)
            line = line.strip('\r')
            print(line, flush=True)
            if line == 'HIL_BEGIN':
                began = True; kv = {}
            m = re.match(r'^HIL (\S+) (.+)$', line)
            if m and began:
                v = m.group(2)
                try:
                    v = float(v) if any(c in v for c in '.e') else int(v)
                except ValueError:
                    pass
                kv[m.group(1)] = v
            if line == 'HIL_DONE' and began:
                json.dump(kv, open(out, 'w'), ensure_ascii=False, indent=1)
                print(f'\n>>> 解析到 {len(kv)} 项，已写入 {out}')
                sys.exit(0)
print('>>> 超时：未收到 HIL_DONE'); sys.exit(1)
