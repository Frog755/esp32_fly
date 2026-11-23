#!/usr/bin/env python3
"""attitude_check.py —— 真机姿态估计检查（门控 Mahony + 六面标定），你自己控制节奏

前提：板子运行【已带 imu_calib.h 编译】的 sensor_test 固件（上电/复位后约 20 s 开始输出 "A" 行）。电机引脚全程锁低。
用法（code/sensor_test 目录）：
  ./attitude.sh            交互检查 7 步（每步摆好按回车，采 2 s 判定 ✓/✗）
  ./attitude.sh --live     只实时显示 roll / pitch / yaw / 门控增益（Ctrl+C 退出）
  ./attitude.sh --only=1,5,6,7   只跑指定步骤（第 1 步平放基准建议总带上）
板子若中途复位（时间戳倒退 / 收到开机信息 / 崩溃信息），脚本会当场提示并记下复位原因，存进 attitude_check_events.txt。
约定：机体 FLU（x 机头 / y 左 / z 上），ZYX 欧拉角
  机头朝下 → pitch 为正；右侧朝下 → roll 为正；俯视逆时针转 → yaw 增加
输出：attitude_check.json、attitude_check_raw.csv（全部 A 行），把终端输出或 json 发给我。
"""
import sys, time, json, math

try:
    import serial
except ImportError:
    sys.exit('缺少 pyserial：请用 ./attitude.sh 运行')

args = [a for a in sys.argv[1:] if not a.startswith('--')]
PORT = args[0] if args else '/dev/ttyACM0'
LIVE = '--live' in sys.argv
ONLY = None
for a in sys.argv[1:]:
    if a.startswith('--only='):
        ONLY = {int(x) for x in a[7:].split(',') if x.strip()}
raw_log = []
events = []          # (本机时间, 内容)：开机/复位/崩溃相关的非 A/S 行
RESET_NAMES = {0: '未知', 1: '上电', 2: '外部引脚', 3: '软件重启 esp_restart', 4: '崩溃(panic)', 5: '中断看门狗', 6: '任务看门狗',
               7: '其他看门狗', 8: '深睡唤醒', 9: '欠压(brownout)', 10: 'SDIO', 11: 'USB 外设复位(USB-Serial-JTAG)', 12: 'JTAG', 13: 'eFuse', 14: '电源毛刺', 15: 'CPU 锁死'}


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


class Reader:
    def __init__(self):
        self.ser = serial.Serial(PORT, 115200, timeout=0.1)
        time.sleep(0.2); self.ser.reset_input_buffer(); self.buf = ''
        self.last_t = None; self.resets = 0

    def poll(self):
        out = []
        self.buf += self.ser.read(4096).decode('utf-8', 'replace')
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            p = line.split()
            if len(p) == 7 and p[0] == 'A':
                try:
                    v = [float(x) for x in p[1:]]
                except ValueError:
                    continue
                if self.last_t is not None and v[0] < self.last_t - 500:
                    self.note(f'⚠ 时间戳倒退 {self.last_t / 1e3:.1f}s → {v[0] / 1e3:.1f}s：板子刚才复位过')
                self.last_t = v[0]
                raw_log.append(v); out.append(v)   # t_ms roll pitch yaw k anorm
            elif p and p[0] != 'S':
                txt = line.strip()
                if txt.startswith('SEN reset_reason'):
                    try:
                        code = int(txt.split()[-1]); txt += f'（{RESET_NAMES.get(code, "?")}）'
                    except ValueError:
                        pass
                    self.note('⚠ 板子重新开机，复位原因：' + txt.split(' ', 2)[-1])
                elif any(k in txt for k in ('Guru', 'abort', 'Backtrace', 'panic', 'Brownout', 'rst:', 'FATAL')):
                    self.note('⚠ 板子输出崩溃/复位信息：' + txt[:160])
                else:
                    events.append((time.strftime('%H:%M:%S'), txt))
        return out

    def note(self, msg):
        self.resets += 1
        events.append((time.strftime('%H:%M:%S'), msg))
        print('\n   ' + msg, flush=True)

    def wait_stream(self, max_s=40):
        """复位后等待重新出 A 行（开机自检需约 20 s，期间请保持静止，陀螺零偏在这时标定）"""
        print('   板子在重新开机自检，请把飞机【平放静止】别动，等待数据恢复……', flush=True)
        t0 = time.time()
        while time.time() - t0 < max_s:
            if self.poll():
                print('   数据已恢复。注意：复位时若飞机在动，陀螺零偏会标歪（偏航会漂），可以按 RST 再静置 20 s。'); return True
        print('   ✗ 40 s 内没有恢复数据'); return False

    def collect(self, sec):
        rows, t0 = [], time.time()
        self.ser.reset_input_buffer(); self.buf = ''
        while time.time() - t0 < sec:
            rows += self.poll()
        return rows


def summarize(rows):
    n = len(rows)
    # yaw 取圆均值，roll 也用圆均值（倒扣时在 ±180 附近）
    def cmean(i):
        s = sum(math.sin(math.radians(r[i])) for r in rows); c = sum(math.cos(math.radians(r[i])) for r in rows)
        return math.degrees(math.atan2(s, c))
    return dict(n=n, roll=cmean(1), pitch=sum(r[2] for r in rows) / n, yaw=cmean(3),
                k=sum(r[4] for r in rows) / n, anorm=sum(r[5] for r in rows) / n)


def fmt(s):
    return f'roll {s["roll"]:+7.1f}°  pitch {s["pitch"]:+6.1f}°  yaw {s["yaw"]:+7.1f}°  门控增益 {s["k"]:.2f}  |a| {s["anorm"]:.3f} g'


def live(rd):
    print('实时显示（Ctrl+C 退出）：门控增益 2.00 = 加计完全参与修正；0 = 纯陀螺（快速转动/加速时正常）')
    last = 0
    while True:
        for v in rd.poll():
            if time.time() - last > 0.1:
                last = time.time()
                print(f'\r roll {v[1]:+7.1f}°  pitch {v[2]:+6.1f}°  yaw {v[3]:+7.1f}°  门控 {v[4]:.2f}  |a| {v[5]:.3f} g   ', end='', flush=True)


STEPS = [
    # (名字, 提示, 判定函数(s, base) -> (ok, 说明))
    ('平放基准', '平放在桌面上（正常起飞姿态）',
     lambda s, b: (abs(s['roll']) < 5 and abs(s['pitch']) < 5 and s['k'] > 1.5 and abs(s['anorm'] - 1) < 0.02,
                   '要求 |roll|、|pitch| < 5°，门控增益 > 1.5，|a| 在 1±0.02 g')),
    ('机头朝下', '机头朝下倾斜约 30~45°（机尾抬起，靠住别动）',
     lambda s, b: (15 < s['pitch'] - b['pitch'] < 70 and abs(wrap(s['roll'] - b['roll'])) < 15, '要求 pitch 增加 15~70°，roll 基本不变')),
    ('右侧朝下', '放回平放，再让右侧朝下倾斜约 30~45°（左侧抬起，靠住别动）',
     lambda s, b: (15 < wrap(s['roll'] - b['roll']) < 70 and abs(s['pitch'] - b['pitch']) < 15, '要求 roll 增加 15~70°，pitch 基本不变')),
    ('俯视逆时针 90°', '放回平放，在桌面上【俯视逆时针】转约 90°（机头从前方转向左边），放平别动',
     lambda s, b: (50 < wrap(s['yaw'] - b['yaw']) < 130 and abs(s['roll']) < 8 and abs(s['pitch']) < 8, '要求 yaw 增加 50~130°（纯陀螺积分，验证偏航符号）')),
    ('倒扣', '翻过来倒扣在桌面上（机腹朝上），放稳别动',
     lambda s, b: (abs(wrap(s['roll'])) > 160 and abs(s['pitch']) < 20, '要求 |roll| ≈ 180°（倒扣时 roll 在 ±180 附近），pitch 接近 0')),
    ('倒扣静置 5 s', '保持倒扣不动（这一步采 5 秒）',
     lambda s, b: (abs(wrap(s['roll'])) > 170 and s['k'] > 1.5, '要求 |roll| > 170°、门控增益 > 1.5（倒扣时加计照常修正）')),
    ('翻回平放', '翻回来平放在桌面上，放稳别动（检验翻转后有无残留误差）',
     lambda s, b: (abs(s['roll']) < 5 and abs(s['pitch']) < 5, '要求 |roll|、|pitch| 回到 < 5°')),
]


def main():
    print(__doc__.split('\n用法')[0])
    rd = Reader()
    t0 = time.time(); got = []
    while time.time() - t0 < 2.0:
        got += rd.poll()
    if len(got) < 40:
        sys.exit(f'2 s 只收到 {len(got)} 行姿态数据（应约 100 行）。请确认烧的是带标定的新固件，复位后等约 20 s 再运行。')
    print(f'✓ 收到姿态数据（{len(got) / 2:.0f} Hz）\n')
    if LIVE:
        try:
            live(rd)
        except KeyboardInterrupt:
            print(); return
    base = None; results = []
    for i, (name, prompt, judge) in enumerate(STEPS, 1):
        if ONLY and i not in ONLY:
            continue
        r_wait = rd.resets
        rd.poll()
        if rd.resets > r_wait:
            rd.wait_stream()
        input(f'第 {i}/{len(STEPS)} 步 →【{prompt}】，好了按回车 ')
        sec = 5.0 if '5 s' in name else 2.0
        print(f'   采样 {sec:.0f} 秒，别碰飞机……', flush=True)
        r0 = rd.resets
        rows = rd.collect(sec)
        if rd.resets > r0 or len(rows) < 25 * sec:
            why = '板子在采样时复位了' if rd.resets > r0 else f'只收到 {len(rows)} 行，板子可能正在复位或串口断了'
            print(f'   ✗ {why}'); results.append(dict(step=name, ok=False, note=why))
            rd.wait_stream(); continue
        s = summarize(rows)
        if base is None:
            base = s
        ok, rule = judge(s, base)
        print('   ' + fmt(s))
        print(f'   {"✓" if ok else "✗"} {rule}\n')
        results.append(dict(step=name, ok=bool(ok), rule=rule, **s))
    n_ok = sum(r['ok'] for r in results)
    print('================ 结果 ================')
    for r in results:
        print(f'  {"✓" if r["ok"] else "✗"} {r["step"]}' + (f'（{r["note"]}）' if r.get('note') else ''))
    print(f'通过 {n_ok}/{len(results)}')
    if n_ok < len(results):
        print('有 ✗ 也没关系，把输出发给我分析；常见原因：动作方向弄反、没放稳、倾斜角度太小。')
    if rd.resets:
        print(f'⚠ 本次检测到 {rd.resets} 次复位相关事件（见 attitude_check_events.txt）')
    with open('attitude_check_events.txt', 'w', encoding='utf-8') as f:
        for t, e in events:
            f.write(f'{t} {e}\n')
    json.dump(dict(port=PORT, time=time.strftime('%Y-%m-%d %H:%M:%S'), passed=n_ok, total=len(results), steps=results,
                   reset_events=[e for _, e in events if e.startswith('⚠')]),
              open('attitude_check.json', 'w'), ensure_ascii=False, indent=1)
    with open('attitude_check_raw.csv', 'w') as f:
        f.write('t_ms,roll_deg,pitch_deg,yaw_deg,gate_k,anorm_g\n')
        for v in raw_log:
            f.write(','.join(f'{x:g}' for x in v) + '\n')
    print('已保存 attitude_check.json、attitude_check_raw.csv。把终端输出或 json 发给我就行。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已取消')
