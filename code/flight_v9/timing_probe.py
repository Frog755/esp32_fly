#!/usr/bin/env python3
"""timing_probe.py —— 开发用：每 10 s 读一次 STATUS，统计 1 kHz 控制循环迟到情况"""
import serial, time, re, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
s = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)


def status():
    s.reset_input_buffer(); s.write(b'status\n'); buf = ''; t0 = time.time()
    while time.time() - t0 < 2.0:
        buf += s.read(4096).decode('utf-8', 'replace')
        if 'STATUS_END' in buf:
            break
    m = re.search(r'STATUS loops (\d+) late_gt1100us (\d+) exec_max_us (\d+)', buf)
    q = re.search(r'STATUS period late_gt1500us (\d+) late_gt2000us (\d+) late_gt5000us (\d+) period_max_us (\d+)', buf)
    return ([int(x) for x in m.groups()] + [int(x) for x in q.groups()]) if (m and q) else None


a = status()
for i in range(N):
    time.sleep(10); b = status()
    if not a or not b:
        print('读不到 STATUS'); a = b; continue
    n = b[0] - a[0]
    print(f'10 s：{n} 拍  >1.1ms {b[1]-a[1]}  >1.5ms {b[3]-a[3]}  >2ms {b[4]-a[4]}  >5ms {b[5]-a[5]}  最长周期 {b[6]} us  执行最长 {b[2]} us')
    a = b
