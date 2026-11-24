#!/usr/bin/env python3
"""bb_selftest.py —— 黑匣子上板自检（干跑版固件，电机不会转；飞机平放桌面即可，不需要动手）

1. 核对固件是 Stage 28 干跑版（motor_output 0、电机引脚 0 0 0 0）
2. 串口心跳 + arm + takeoff → 台上起飞超时 → 自动降落 → 触地（状态机按\"通电\"走一遍，干跑版不出力）
3. 等冻结（停转后 10 s）→ bbdump：当前槽应含通电段与 takeoff / land_takeoff_timeout / landed 事件、crc 通过
4. reboot（软件复位）→ 开机日志应报告上一槽有效 → 再 bbdump：上一槽与复位前逐条一致（跨复位保留）
5. 时序：黑匣子开启后 1 kHz 循环周期统计
用法：~/.espressif/tools/python/v6.1/venv/bin/python bb_selftest.py [/dev/ttyACM0]
"""
import sys, os, time, json, threading
import serial
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_read

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
res = {}
ok_all = True
def check(c, msg):
    global ok_all
    print(('  ✓ ' if c else '  ✗ ') + msg, flush=True); ok_all &= bool(c)


class B:
    def __init__(self):
        self.s = serial.Serial(PORT, 115200, timeout=0.05); time.sleep(0.2); self.s.reset_input_buffer()
        self.buf = ''; self.lines = []; self.lock = threading.Lock(); self.hb = False
        threading.Thread(target=self._hb, daemon=True).start()

    def _hb(self):
        while True:
            if self.hb:
                with self.lock:
                    try: self.s.write(b'hb\n')
                    except Exception: pass
            time.sleep(0.1)

    def send(self, t):
        with self.lock: self.s.write((t + '\n').encode())

    def pump(self, sec, until=None):
        t0 = time.time(); got = []
        while time.time() - t0 < sec:
            try: d = self.s.read(65536)
            except (serial.SerialException, OSError):
                time.sleep(0.5); self.reopen(); continue
            self.buf += d.decode('ascii', 'replace')
            while '\n' in self.buf:
                ln, self.buf = self.buf.split('\n', 1); ln = ln.strip(); got.append(ln)
                if until and until(ln): return got
        return got

    def reopen(self):
        for _ in range(40):
            try:
                self.s.close(); self.s = serial.Serial(PORT, 115200, timeout=0.05); return
            except Exception: time.sleep(0.25)

    def status(self):
        self.send('status'); st = {}
        for ln in self.pump(3, until=lambda l: l == 'STATUS_END'):
            p = ln.split()
            if p and p[0] == 'STATUS' and len(p) > 1: st[p[1]] = ' '.join(p[2:])
        return st

    def dump(self):
        """bbdump + 按缺失区间 bbget 补发（与 bb_read.py 同一套解析）；返回 (行, 用时, 首次缺失条数, 补发次数)"""
        t0 = time.time(); self.send('bbdump')
        ls = [l for l in self.pump(90, until=lambda l: l == 'BB_DONE') if l.startswith('BB')]
        sl = bb_read.parse(ls)
        miss0 = sum(b - a + 1 for d in sl.values() for a, b in bb_read.missing_ranges(d))
        nref = 0
        for _ in range(6):
            todo = [(si, a, b) for si, d in sorted(sl.items()) for a, b in bb_read.missing_ranges(d)]
            if not todo: break
            for si, a, b in todo:
                self.send(f'bbget {si} {a} {b}'); nref += 1
                ls += [l for l in self.pump(30, until=lambda l: l.startswith('BB_GET_END')) if l.startswith('BB')]
            sl = bb_read.parse(ls)
        return ls, time.time() - t0, miss0, nref


b = B()
print('等待飞控数据……')
b.pump(25, until=lambda l: l.startswith('F '))      # 开机约 8~10 s 才有 F 行；之前问 status 会没有回应
b.pump(1)
st = b.status()
if 'fw' not in st: b.pump(2); st = b.status()
print('   ' + st.get('fw', '?'))
check('dry-run' in st.get('fw', '') and st.get('motor_output') == '0' and st.get('motor_pins_level') == '0 0 0 0',
      '干跑版 Stage 28 固件、电机驱动未编译、电机引脚全低')
print('   黑匣子 ' + st.get('blackbox', '?'))

print('\n[1] 台上走一遍\"通电\"流程（干跑版不出力）')
b.hb = True; b.pump(1.0); b.send('arm')
ev = b.pump(3, until=lambda l: 'EVT' in l and 'armed' in l)
check(any('armed ARMED' in l for l in ev), '解锁')
b.send('takeoff')
ev = b.pump(15, until=lambda l: 'EVT' in l and 'landed' in l)
evs = [l for l in ev if l.startswith('EVT')]
print('   ' + ' | '.join(evs))
check(any('land_takeoff_timeout' in l for l in evs) and any('landed' in l for l in evs), '起飞超时 → 降落 → 触地')
print('   等 11 s 冻结……'); b.pump(11)
st = b.status(); print('   黑匣子 ' + st.get('blackbox', '?'))
check(st.get('blackbox', '').endswith('frozen 1'), '停转 10 s 后冻结')
b.hb = False; b.pump(1)

print('\n[2] 读出（bbdump）')
lines, dt, miss0, nref = b.dump()
sl = bb_read.parse(lines)
print(f'   首次读出传输缺失 {miss0} 条，补发 {nref} 次')
res['transfer'] = dict(missing_first=miss0, refetch=nref)
cur = next(d for d in sl.values() if d.get('valid') and d['is_current'])
cur_si = next(si for si, d in sl.items() if d is cur)
n_bytes = sum(len(l) for l in lines)
print(f'   用时 {dt:.1f} s，{n_bytes / 1024:.0f} KB 文本')
for l in bb_read.summary(cur_si, cur): print(l)
rs = [bb_read.scaled(r) for r in cur['recs']]
on = [s for s in rs if s['on']]
names = [s['ev'] for s in rs if s['ev']]
check(cur.get('complete'), '每行 crc32 通过、条数与槽头一致（完整）')
check(cur['frozen'] and 6.5 < len(on) / 200 < 9.0, f'当前槽已冻结、含通电段 {len(on) / 200:.2f} s')
check(names[-4:] == ['armed', 'takeoff', 'land_takeoff_timeout', 'landed'] or
      ['armed', 'takeoff', 'land_takeoff_timeout', 'landed'] == [n for n in names if n in ('armed', 'takeoff', 'land_takeoff_timeout', 'landed')][-4:],
      '事件序列 armed → takeoff → land_takeoff_timeout → landed')
gaps = sum(1 for a, c in zip(cur['recs'], cur['recs'][1:]) if c['idx'] != a['idx'] + 1)
check(gaps == 0, f'记录连续无断档（{len(rs)} 条）')
check(all(s['out'] == 0 for s in rs) and all(not s['motor_fw'] for s in rs), '干跑版：实际输出全程 0')
res['before'] = dict(boot=cur['boot'], n=len(rs), on_s=len(on) / 200, dump_s=dt)
before = cur['recs']

print('\n[3] 软件复位后上一槽是否保留')
b.send('reboot')
boot = b.pump(25, until=lambda l: l == 'FLT_READY')
bl = [l for l in boot if l.startswith('BOOT blackbox') or l.startswith('BOOT reset_reason')]
print('   ' + ' | '.join(bl))
check(any('prev slot valid' in l for l in bl), '开机日志报告上一槽有效')
b.pump(1)
lines2, _, miss2, nref2 = b.dump()
sl2 = bb_read.parse(lines2)
print(f'   复位后读出传输缺失 {miss2} 条，补发 {nref2} 次')
prev = next((d for d in sl2.values() if d.get('valid') and d['boot'] == cur['boot']), None)
now = next(d for d in sl2.values() if d.get('valid') and d['is_current'])
check(prev is not None and not prev['is_current'] and prev.get('complete'), '上一槽（复位前那次开机）完整读出')
same = prev is not None and prev['recs'] == before
check(same, f'上一槽与复位前读出逐条逐字段一致（{len(before)} 条）')
check(now['boot'] == cur['boot'] + 1 and now['reset_name'] == 'SW', f'新开机号 {now["boot"]}、复位原因 {now["reset_name"]}')
pi = next(si for si, d in sl2.items() if d is prev) if prev else None
if prev:
    for l in bb_read.summary(pi, prev, now['reset_name']): print(l)

def counters(st):
    p = (st.get('loops', '') + ' ' + st.get('period', '')).split()
    d = {'loops': int(p[0])} if p and p[0].isdigit() else {}
    for k, v in zip(p[1::2], p[2::2]):
        if v.isdigit(): d[k] = int(v)
    return d

print('\n[4] 时序：黑匣子正常记录时 30 s（计数器差值，不含读出期间）')
c_dump_end = counters(b.status())
b.pump(30)
c1 = counters(b.status())
d = {k: c1.get(k, 0) - c_dump_end.get(k, 0) for k in ('loops', 'late_gt1100us', 'late_gt1500us', 'late_gt2000us', 'late_gt5000us')}
print(f'   30 s：{d["loops"]} 拍，>1.1 ms {d["late_gt1100us"]}（{d["late_gt1100us"] / max(1, d["loops"]) * 100:.1f}%），'
      f'>1.5 ms {d["late_gt1500us"]}，>2 ms {d["late_gt2000us"]}，>5 ms {d["late_gt5000us"]}')
print(f'   （开机以来含两次读出：period_max {c1.get("period_max_us")} μs、>2 ms {c1.get("late_gt2000us")} 次——读出只在未解锁时允许）')
res['timing'] = dict(window_30s=d, since_boot=c1)
check(d['late_gt2000us'] == 0, '正常记录时没有 >2 ms 的控制周期')
rec_per = max(bb_read.scaled(r)['per_max_us'] for r in before)
check(rec_per < 2000, f'黑匣子自己记下的最长周期 {rec_per} μs < 2 ms（含台上通电流程）')
os.makedirs('blackbox', exist_ok=True)
with open('blackbox/selftest_dump_before_reboot.txt', 'w') as f: f.write('\n'.join(lines) + '\n')
with open('blackbox/selftest_dump_after_reboot.txt', 'w') as f: f.write('\n'.join(lines2) + '\n')
res['passed'] = ok_all
json.dump(res, open('blackbox/selftest.json', 'w'), ensure_ascii=False, indent=1)
print('\n黑匣子上板自检全部通过' if ok_all else '\n有失败项')
sys.exit(0 if ok_all else 1)
