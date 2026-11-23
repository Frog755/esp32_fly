#!/usr/bin/env python3
"""sensor_read.py —— 读取 sensor_test 固件串口：解析 SEN 汇总行（到 SEN_DONE 为止），
并可选继续录制 N 秒 50 Hz 流式数据（S 行）到 CSV，用于后续姿态轴标定。
用法：python3 sensor_read.py /dev/ttyACM0 out_prefix [record_s] [timeout_s] [--stream-only]
  --stream-only：板子已在流式输出阶段（打开串口不会复位它），直接从现在起录制 record_s 秒"""
import sys, time, json, re
import serial

port, prefix = sys.argv[1], sys.argv[2]
record_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
timeout = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0
ser = serial.Serial(port, 115200, timeout=0.2)
buf = ''; t0 = time.time(); kv = {}; began = done = False; t_done = None; rows = []
if '--stream-only' in sys.argv:
    done = True; t_done = time.time(); ser.reset_input_buffer()
log = open(prefix + '_serial.txt', 'w', encoding='utf-8')
while time.time() - t0 < timeout:
    chunk = ser.read(4096).decode('utf-8', 'replace')
    if not chunk:
        if done and time.time() - t_done >= record_s:
            break
        continue
    buf += chunk
    while '\n' in buf:
        line, buf = buf.split('\n', 1)
        line = line.strip('\r')
        log.write(line + '\n')
        if line.startswith('S ') and done:
            p = line.split()
            if len(p) == 8:
                rows.append([float(x) for x in p[1:]])
            continue
        print(line, flush=True)
        if line == 'SEN_BEGIN':
            began = True; kv = {}
        m = re.match(r'^SEN (\S+) (.+)$', line)
        if m and began:
            kv[m.group(1)] = m.group(2)
        if line == 'SEN_DONE' and began:
            done = True; t_done = time.time()
            json.dump(kv, open(prefix + '_summary.json', 'w'), ensure_ascii=False, indent=1)
            print(f'>>> 汇总 {len(kv)} 项 → {prefix}_summary.json', flush=True)
            if record_s <= 0:
                sys.exit(0)
    if done and time.time() - t_done >= record_s:
        break
if rows:
    with open(prefix + '_stream.csv', 'w') as f:
        f.write('t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps\n')
        for r in rows:
            f.write(','.join(f'{x:.5g}' for x in r) + '\n')
    print(f'>>> 录制 {len(rows)} 行 → {prefix}_stream.csv')
sys.exit(0 if done else 1)
