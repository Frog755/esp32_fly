#!/usr/bin/env python3
"""dryrun_check.py —— 正式飞控干跑检查（电机不转）：用手摆姿态，核对控制器算出的电机指令方向

前提：板子运行 code/flight_v9 固件（FLIGHT_MOTOR_OUTPUT=0，电机驱动未编译进固件，引脚锁低）。
      上电 / 复位后约 8 s 开始输出（开机要静止 2 s 标定陀螺）。
用法（code/flight_v9 目录）：  ./dryrun.sh            交互检查 8 步
                               ./dryrun.sh --only=1,7  只跑指定步骤
判据来源：test/host_test.c（同一份控制器代码在电脑上算出的期望方向）
  右侧朝下 → 右侧 M1+M2 加速；机头朝下 → 前方 M1+M4 加速；
  俯视逆时针转 → 逆时针桨 M1+M3 加速（反扭矩制动）；倒扣 → 进入 bang-bang 翻转模式
  第 8 步：起飞→（台架上离不了地）起飞超时→自动降落→触地停桨；land 指令；空格急停
输出：dryrun_check.json、dryrun_raw.csv、dryrun_events.txt —— 把终端输出或 json 发给我。
"""
import sys, time, json, math, threading

try:
    import serial
except ImportError:
    sys.exit('缺少 pyserial：请用 ./dryrun.sh 运行')

args = [a for a in sys.argv[1:] if not a.startswith('--')]
PORT = args[0] if args else '/dev/ttyACM0'
ONLY = None
for a in sys.argv[1:]:
    if a.startswith('--only='):
        ONLY = {int(x) for x in a[7:].split(',') if x.strip()}
COLS = ['t', 'st', 'mode', 'roll', 'pitch', 'yaw', 'wz', 'z', 'vz', 'vb', 'k', 'm1', 'm2', 'm3', 'm4', 'out', 'zt', 'thr']
raw, events = [], []
STATE = ['DISARMED', 'ARMED', 'TAKEOFF', 'RECOVER', 'HOLD', 'LAND']


class Board:
    def __init__(self):
        self.ser = serial.Serial(PORT, 115200, timeout=0.1)
        time.sleep(0.2); self.ser.reset_input_buffer()
        self.buf = ''; self.last_t = None; self.resets = 0; self.status = {}; self.evts = []; self.out_nonzero = 0
        self.lock = threading.Lock(); self.hb_on = True
        threading.Thread(target=self._hb, daemon=True).start()     # 串口心跳：固件没有链路就不让解锁（同手机网页每 0.1 s 一次）

    def _hb(self):
        while True:
            if self.hb_on:
                try:
                    with self.lock:
                        self.ser.write(b'hb\n')
                except Exception:
                    pass
            time.sleep(0.1)

    def send(self, s):
        with self.lock:
            self.ser.write((s + '\n').encode())

    def kill(self):
        with self.lock:
            self.ser.write(b' ')

    def poll(self):
        out = []
        self.buf += self.ser.read(4096).decode('utf-8', 'replace')
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            line = line.strip(); p = line.split()
            if not p:
                continue
            if p[0] == 'F' and len(p) == 19:
                try:
                    v = dict(zip(COLS, [float(x) for x in p[1:]]))
                except ValueError:
                    continue
                if self.last_t is not None and v['t'] < self.last_t - 500:
                    self.note(f'⚠ 时间戳倒退 {self.last_t / 1e3:.1f}s → {v["t"] / 1e3:.1f}s：板子复位过')
                self.last_t = v['t']
                if abs(v['out']) > 1e-6:
                    self.out_nonzero += 1
                raw.append(v); out.append(v)
            elif p[0] == 'EVT':
                self.evts.append(line); events.append((time.strftime('%H:%M:%S'), line))
            elif p[0] == 'STATUS' and len(p) >= 2:
                self.status[p[1]] = ' '.join(p[2:])
            elif p[0] == 'BOOT' and len(p) >= 3 and p[1] == 'reset_reason':
                self.note('⚠ 板子重新开机，复位原因 ' + p[2])
            elif any(k in line for k in ('FATAL', 'Guru', 'abort', 'Backtrace', 'panic')):
                self.note('⚠ 板子输出错误信息：' + line[:160])
            else:
                events.append((time.strftime('%H:%M:%S'), line))
        return out

    def note(self, msg):
        self.resets += 1; events.append((time.strftime('%H:%M:%S'), msg)); print('\n   ' + msg, flush=True)

    def collect(self, sec):
        rows, t0 = [], time.time()
        self.ser.reset_input_buffer(); self.buf = ''
        while time.time() - t0 < sec:
            rows += self.poll()
        return rows

    def get_status(self):
        self.status = {}; self.send('status'); t0 = time.time()
        while time.time() - t0 < 2.0 and 'mapping' not in self.status:
            self.poll()
        return self.status

    def wait_evt(self, word, sec=1.5):
        t0 = time.time()
        while time.time() - t0 < sec:
            self.poll()
            for e in self.evts:
                if word in e:
                    return e
        return None


def mean(rows, f):
    return sum(f(r) for r in rows) / len(rows) if rows else float('nan')


rd = lambda r: (r['m1'] + r['m2']) - (r['m3'] + r['m4'])   # 右 − 左
pd = lambda r: (r['m1'] + r['m4']) - (r['m2'] + r['m3'])   # 前 − 后
yd = lambda r: (r['m1'] + r['m3']) - (r['m2'] + r['m4'])   # 逆时针桨 − 顺时针桨


def show(rows):
    print(f'   roll {mean(rows, lambda r: r["roll"]):+6.1f}°  pitch {mean(rows, lambda r: r["pitch"]):+6.1f}°  '
          f'模式 {mean(rows, lambda r: r["mode"]):.2f}  电机指令 M1 {mean(rows, lambda r: r["m1"]):.2f} M2 {mean(rows, lambda r: r["m2"]):.2f} '
          f'M3 {mean(rows, lambda r: r["m3"]):.2f} M4 {mean(rows, lambda r: r["m4"]):.2f}')
    print(f'   右−左 {mean(rows, rd):+.2f}   前−后 {mean(rows, pd):+.2f}   逆桨−顺桨 {mean(rows, yd):+.2f}')


def step_pose(b, prompt, judge, sec=2.0):
    input(f'→【{prompt}】，好了按回车 ')
    print(f'   采样 {sec:.0f} 秒……', flush=True)
    r0 = b.resets; rows = b.collect(sec)
    if b.resets > r0 or len(rows) < 25 * sec:
        return False, '板子复位或数据不足', {}
    show(rows)
    ok, rule = judge(rows)
    return ok, rule, dict(n=len(rows), roll=mean(rows, lambda r: r['roll']), pitch=mean(rows, lambda r: r['pitch']),
                          mode=mean(rows, lambda r: r['mode']), right_minus_left=mean(rows, rd), front_minus_back=mean(rows, pd),
                          ccw_minus_cw=mean(rows, yd))


def step_yaw(b):
    input('→【放平在桌面上，按回车后立刻在桌面上「俯视逆时针」连续转动约 3 秒（转一圈左右，别太慢）】 ')
    print('   开始转！采样 3 秒……', flush=True)
    r0 = b.resets; rows = b.collect(3.0)
    if b.resets > r0 or len(rows) < 75:
        return False, '板子复位或数据不足', {}
    spin = [r for r in rows if r['wz'] > 30]
    print(f'   逆时针转速 >30°/s 的样本 {len(spin)} 个，平均转速 {mean(spin, lambda r: r["wz"]):.0f}°/s')
    if len(spin) < 15:
        return False, '转得太慢或方向反了（需要俯视逆时针、>30°/s 持续约 0.3 s 以上）', dict(n_spin=len(spin))
    show(spin)
    ok = mean(spin, yd) > 0.05
    return ok, '要求转动时 逆时针桨 M1+M3 − 顺时针桨 M2+M4 > 0.05（反扭矩制动自转）', dict(n_spin=len(spin), wz=mean(spin, lambda r: r['wz']), ccw_minus_cw=mean(spin, yd))


def step_safety(b):
    input('→【安全逻辑 + 降落 + 链路丢失检查（电机不会转，约 30 秒）：飞机平放静止，按回车，然后别碰它】 ')
    res = {}; ok = True; out_max = 0.0

    def ev(cmd, word, sec=1.5):
        b.evts = []
        if cmd == ' ':
            b.kill()
        elif cmd:
            b.send(cmd)
        e = b.wait_evt(word, sec)
        return e

    # A：起飞 → 台架上不可能离地 → 6 s 起飞超时 → 自动降落 → 触地检测停桨
    e = ev('arm', 'armed'); print(f'   arm → {e}'); ok &= bool(e); res['arm'] = e
    e = ev('takeoff', 'takeoff'); print(f'   takeoff → {e}'); ok &= bool(e); res['takeoff'] = e
    t0 = time.time(); seen = []
    while time.time() - t0 < 10.0:
        rows = b.poll()
        out_max = max([out_max] + [abs(r['out']) for r in rows])
        for x in b.evts:
            if x not in seen:
                seen.append(x)
        if any('landed' in x or 'DISARMED' in x for x in seen):
            break
    for x in seen:
        print(f'     {x}')
    ok &= any('land_takeoff_timeout' in x for x in seen) and any(' landed ' in x + ' ' for x in seen)
    res['takeoff_timeout_then_landed'] = seen
    # B：起飞 1 s 后 land 指令 → 降落 → 触地停桨
    e = ev('arm', 'armed'); ok &= bool(e)
    e = ev('takeoff', 'takeoff'); ok &= bool(e)
    time.sleep(1.0)
    e = ev('land', 'land_cmd'); print(f'   land → {e}'); ok &= bool(e); res['land_cmd'] = e
    e = b.wait_evt('landed', 4.0); print(f'     → {e}'); ok &= bool(e); res['land_cmd_landed'] = e
    # C：空格急停
    e = ev('arm', 'armed'); ok &= bool(e)
    e = ev('takeoff', 'takeoff'); ok &= bool(e)
    e = ev(' ', 'disarm_cmd'); print(f'   空格急停 → {e}'); ok &= bool(e); res['space_kill'] = e
    # D：未解锁时 takeoff 被拒
    e = ev('takeoff', 'refuse_state'); print(f'   未解锁时 takeoff → {e}'); ok &= bool(e); res['takeoff_when_disarmed'] = e
    # E：起飞中心跳中断 → 降落 → 触地停桨
    e = ev('arm', 'armed'); ok &= bool(e)
    e = ev('takeoff', 'takeoff'); ok &= bool(e)
    b.hb_on = False; b.evts = []
    e = b.wait_evt('land_link_lost', 2.0); print(f'   起飞中停止心跳 → {e}'); ok &= bool(e); res['link_lost_in_flight'] = e
    e = b.wait_evt('landed', 4.0); print(f'     → {e}'); ok &= bool(e); res['link_lost_landed'] = e
    # F：没有心跳时拒绝解锁
    time.sleep(0.3)
    e = ev('arm', 'refuse_link'); print(f'   没有心跳时 arm → {e}'); ok &= bool(e); res['arm_without_link'] = e
    # G：解锁后（还没出手）心跳中断 → 上锁
    b.hb_on = True; time.sleep(0.5)
    e = ev('arm', 'armed'); ok &= bool(e)
    b.hb_on = False; b.evts = []
    e = b.wait_evt('disarm_link_lost', 2.0); print(f'   解锁后停止心跳 → {e}'); ok &= bool(e); res['link_lost_armed'] = e
    b.hb_on = True; time.sleep(0.3)
    rows = b.collect(0.5)
    out_max = max([out_max] + [abs(r['out']) for r in rows])
    print(f'   全程实际写电机的占空比之和最大 {out_max:.3f}（干跑版应恒为 0）')
    ok &= out_max == 0.0
    return ok, ('起飞超时→自动降落→触地停桨、land 指令→触地停桨、空格→立即上锁、未解锁 takeoff 被拒、'
                '心跳中断：起飞中→降落→触地停桨 / 无心跳拒绝解锁 / 解锁中→上锁；全程电机输出 0'), res


def main():
    print(__doc__.split('\n用法')[0])
    b = Board()
    t0 = time.time(); got = []
    while time.time() - t0 < 2.0:
        got += b.poll()
    if len(got) < 40:
        sys.exit(f'2 s 只收到 {len(got)} 行数据（应约 100 行）。请确认烧的是 flight_v9 固件，复位后静置约 8 s 再运行。')
    st = b.get_status()
    print(f'✓ 收到数据（{len(got) / 2:.0f} Hz）  固件 {st.get("fw", "?")}')
    print(f'   电机输出编译开关 {st.get("motor_output", "?")}   电机引脚电平 {st.get("motor_pins_level", "?")}   电池 {st.get("vbatt", "?")} V')
    print(f'   映射 {st.get("mapping", "?")}\n')
    if st.get('motor_output') != '0' or st.get('motor_pins_level') != '0 0 0 0':
        sys.exit('✗ 这个固件不是干跑版（或电机引脚不是低电平），停止检查。')
    steps = [
        ('平放静止', lambda: step_pose(b, '平放在桌面上，放稳别动',
            lambda rows: (abs(mean(rows, rd)) < 0.15 and abs(mean(rows, pd)) < 0.15 and mean(rows, lambda r: r['mode']) < 0.1,
                          '要求 |右−左|、|前−后| < 0.15，PD 模式'))),
        ('右侧朝下', lambda: step_pose(b, '右侧朝下倾斜约 30~45°（左侧抬起，靠住别动）',
            lambda rows: (mean(rows, lambda r: r['roll']) > 15 and mean(rows, rd) > 0.15, '要求 roll > 15° 且 右侧 M1+M2 比左侧大 0.15 以上'))),
        ('机头朝下', lambda: step_pose(b, '放回平放，再让机头朝下倾斜约 30~45°（机尾抬起，靠住别动）',
            lambda rows: (mean(rows, lambda r: r['pitch']) > 15 and mean(rows, pd) > 0.15, '要求 pitch > 15° 且 前方 M1+M4 比后方大 0.15 以上'))),
        ('左侧朝下', lambda: step_pose(b, '放回平放，再让左侧朝下倾斜约 30~45°（右侧抬起，靠住别动）',
            lambda rows: (mean(rows, lambda r: r['roll']) < -15 and mean(rows, rd) < -0.15, '要求 roll < −15° 且 左侧 M3+M4 比右侧大 0.15 以上'))),
        ('逆时针自转', lambda: step_yaw(b)),
        ('倒扣', lambda: step_pose(b, '翻过来倒扣在桌面上（机腹朝上），放稳别动',
            lambda rows: (mean(rows, lambda r: r['mode']) > 0.8, '要求进入 bang-bang 翻转模式（模式 1）'))),
        ('翻回平放', lambda: step_pose(b, '翻回来平放在桌面上，放稳别动',
            lambda rows: (mean(rows, lambda r: r['mode']) < 0.1 and abs(mean(rows, lambda r: r['roll'])) < 5 and abs(mean(rows, lambda r: r['pitch'])) < 5,
                          '要求回到 PD 模式、|roll|、|pitch| < 5°'))),
        ('安全逻辑', lambda: step_safety(b)),
    ]
    results = []
    for i, (name, fn) in enumerate(steps, 1):
        if ONLY and i not in ONLY:
            continue
        print(f'第 {i}/{len(steps)} 步：{name}')
        ok, rule, data = fn()
        print(f'   {"✓" if ok else "✗"} {rule}\n')
        results.append(dict(step=name, ok=bool(ok), rule=rule, **data))
    st = b.get_status()
    print('================ 结果 ================')
    for r in results:
        print(f'  {"✓" if r["ok"] else "✗"} {r["step"]}')
    n_ok = sum(r['ok'] for r in results)
    print(f'通过 {n_ok}/{len(results)}')
    print(f'全程实际电机输出非零的行数：{b.out_nonzero}（应为 0）；结束时电机引脚电平 {st.get("motor_pins_level", "?")}')
    print(f'时序：{st.get("loops", "?")}')
    if b.resets:
        print(f'⚠ 检测到 {b.resets} 次复位相关事件（见 dryrun_events.txt）')
    json.dump(dict(port=PORT, time=time.strftime('%Y-%m-%d %H:%M:%S'), passed=n_ok, total=len(results), steps=results,
                   out_nonzero_rows=b.out_nonzero, status_end=st, reset_events=[e for _, e in events if e.startswith('⚠')]),
              open('dryrun_check.json', 'w'), ensure_ascii=False, indent=1)
    with open('dryrun_raw.csv', 'w') as f:
        f.write(','.join(COLS) + '\n')
        for v in raw:
            f.write(','.join(f'{v[c]:g}' for c in COLS) + '\n')
    with open('dryrun_events.txt', 'w', encoding='utf-8') as f:
        for t, e in events:
            f.write(f'{t} {e}\n')
    print('已保存 dryrun_check.json、dryrun_raw.csv、dryrun_events.txt。把终端输出或 json 发给我就行。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已取消')
