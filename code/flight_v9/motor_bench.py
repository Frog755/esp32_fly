#!/usr/bin/env python3
"""motor_bench.py —— Stage 27 拆桨电机台架：第一次由本飞控真转电机（油门上限 6%）

★ 前提：4 个桨全部拆下；电池插着（≥3.9 V）；USB 连电脑；手机能连热点 YK-FLY-xxxx。
  飞机平放在桌面上，电机轴附近不要有线和杂物。全程随时 Ctrl+C = 串口急停 + 退出。
流程（脚本一步步提示，每步你按回车 / 在手机上操作）：
  0. 安全确认（要你输入「已拆桨」）→ 烧录电机驱动版（build_mc，油门上限 6%）→ 开机核对
  A. 起飞超时 → 自动降落 → 触地停桨：手机 按住解锁 → 按住起飞，然后不要碰，约 8 秒后自己停
  B. 急停：手机 解锁 → 起飞，脚本提示后按「急停」
  C. 链路丢失：手机 解锁 → 起飞，脚本提示后关掉手机 WLAN（不锁屏），约 1~2 秒内应自己停
  最后默认烧回干跑版（电机驱动不编译）。
输出：motor_bench.json、motor_bench_raw.csv、motor_bench_events.txt —— 把终端最后的结果或 json 发给我。
用法（code/flight_v9 目录）：./motor_bench.sh [串口，默认 /dev/ttyACM0]   --no-flash 跳过烧录（板上已是电机驱动版）
"""
import sys, os, time, json, glob, subprocess, statistics

try:
    import serial
except ImportError:
    sys.exit('缺少 pyserial：请用 ./motor_bench.sh 运行')

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith('--')]
PORT = args[0] if args else '/dev/ttyACM0'
NO_FLASH = '--no-flash' in sys.argv
U_MAX = 0.06
OUT_FULL = 4 * U_MAX                       # 4 路都打满上限时实际占空比之和
COLS = ['t', 'st', 'mode', 'roll', 'pitch', 'yaw', 'wz', 'z', 'vz', 'vb', 'k', 'm1', 'm2', 'm3', 'm4', 'out', 'zt', 'thr']
STATE = ['DISARMED', 'ARMED', 'TAKEOFF', 'RECOVER', 'HOLD', 'LAND']
raw, events = [], []


def log_event(line):
    events.append((time.strftime('%H:%M:%S'), line))


# ---------------------------------------------------------------- 串口
class Board:
    def __init__(self):
        self.ser = None; self.buf = ''; self.status = {}; self.evts = []; self.resets = []; self.last_t = None
        self.open()

    def open(self, wait=20.0):
        t0 = time.time()
        while time.time() - t0 < wait:
            try:
                self.ser = serial.Serial(PORT, 115200, timeout=0.05)
                time.sleep(0.2); self.ser.reset_input_buffer(); self.buf = ''
                return True
            except (serial.SerialException, OSError):
                time.sleep(0.3)
        return False

    def write(self, b):
        try:
            self.ser.write(b)
        except Exception:
            pass

    def send(self, s):
        self.write((s + '\n').encode())

    def estop(self):
        for _ in range(3):
            self.write(b' '); time.sleep(0.02)

    def poll(self):
        out = []
        try:
            data = self.ser.read(4096)
        except (serial.SerialException, OSError):
            self.note('⚠ 串口断开（可能是电机干扰导致 USB 复位 / 芯片复位），尝试重连……')
            try:
                self.ser.close()
            except Exception:
                pass
            if not self.open(15.0):
                raise SystemExit('✗ 15 秒内串口没有恢复，停止。电机此时应已因芯片复位而停转；请检查板子。')
            return out
        self.buf += data.decode('utf-8', 'replace')
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
                v['wall'] = time.time()
                raw.append(v); out.append(v)
            elif p[0] == 'EVT':
                self.evts.append(line); log_event(line)
            elif p[0] == 'STATUS' and len(p) >= 2:
                self.status[p[1]] = ' '.join(p[2:])
            elif p[0] == 'BOOT' and len(p) >= 3 and p[1] == 'reset_reason':
                if self.last_t is None:             # 烧录后 / 打开串口后第一次开机（esptool 硬复位 = 11 ESP_RST_USB），属预期
                    log_event(f'开机（烧录后的正常重启），复位原因 {p[2]}')
                else:
                    self.note('⚠ 板子重新开机，复位原因 ' + p[2])
            elif any(k in line for k in ('FATAL', 'Guru', 'abort', 'Backtrace', 'panic')):
                self.note('⚠ 板子输出错误信息：' + line[:160])
            else:
                log_event(line)
        return out

    def note(self, msg):
        self.resets.append(msg); log_event(msg); print('\n   ' + msg, flush=True)

    def get_status(self):
        self.status = {}; self.send('status'); t0 = time.time()
        while time.time() - t0 < 2.5 and 'mapping' not in self.status:
            self.poll()
        return self.status


def yes(prompt):
    while True:
        a = input(prompt + ' [y/n] ').strip().lower()
        if a in ('y', 'yes', '是', '对', '1'):
            return True
        if a in ('n', 'no', '否', '不', '0'):
            return False


# ---------------------------------------------------------------- 烧录
def flash(build_dir, must_contain, label):
    bdir = os.path.join(HERE, build_dir)
    fa = os.path.join(bdir, 'flash_args')
    app = os.path.join(bdir, 'flight_v9.bin')
    if not (os.path.exists(fa) and os.path.exists(app)):
        sys.exit(f'✗ 找不到 {build_dir}/flash_args 或 flight_v9.bin，请先让我编译')
    if must_contain.encode() not in open(app, 'rb').read():
        sys.exit(f'✗ {build_dir}/flight_v9.bin 里没有「{must_contain}」，不是预期的{label}固件，停止。')
    argv = []
    for ln in open(fa).read().split('\n'):
        argv += ln.split()
    cmd = [sys.executable, '-m', 'esptool', '--chip', 'esp32s3', '-p', PORT, '-b', '460800',
           '--before', 'default-reset', '--after', 'hard-reset', 'write-flash'] + argv
    print(f'   烧录{label}固件（{build_dir}）……', flush=True)
    r = subprocess.run(cmd, cwd=bdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_event(f'flash {build_dir} exit {r.returncode}')
    if r.returncode != 0:
        print(r.stdout[-2500:])
        sys.exit(f'✗ 烧录{label}固件失败')
    print(f'   ✓ 已烧录{label}固件，板子正在重启。')


def wait_boot(b, expect_motor):
    print('   等待开机（保持飞机平放、不要动，开机要静止 2 秒标定陀螺）……', flush=True)
    t0 = time.time(); got = []
    while time.time() - t0 < 25:
        got += b.poll()
        if len(got) > 50:
            break
    if len(got) < 50:
        sys.exit('✗ 25 s 内没收到飞控数据。请按一下板子复位键，静置 10 秒后用 --no-flash 重新运行。')
    st = b.get_status()
    mo, um, pins = st.get('motor_output', '?'), st.get('u_max', '?'), st.get('motor_pins_level', '?')
    print(f'   固件 {st.get("fw", "?")}')
    print(f'   电机驱动 {mo}  油门上限 {um}  电机引脚 {pins}  电池 {st.get("vbatt", "?")} V  状态 {st.get("state", "?")}')
    if expect_motor:
        if mo != '1' or abs(float(um or 0) - U_MAX) > 1e-4:
            sys.exit(f'✗ 板上不是「电机驱动版、油门上限 {U_MAX}」固件，停止。')
        if st.get('state') != 'DISARMED' or st.get('motor_killed', '0') != '0':
            sys.exit('✗ 开机状态不对（应为 DISARMED、未锁死），停止。')
    elif mo != '0':
        print('   ⚠ 板上仍是电机驱动版！请手动重烧干跑版：idf.py -p /dev/ttyACM0 flash')
    return st


# ---------------------------------------------------------------- 分析（纯函数，便于离线检查）
def evt_name(e):
    p = e.split()
    return p[2] if len(p) >= 3 else ''


def evt_t(e):
    try:
        return float(e.split()[1])
    except (IndexError, ValueError):
        return float('nan')


def std(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else float('nan')


def analyse(rows, evts):
    """rows：本步所有 F 行；evts：本步 EVT 行。返回电机通电段的统计。"""
    on = [r for r in rows if r['out'] > 1e-6]
    off_before = [r for r in rows if r['st'] == 0 and (not on or r['t'] < on[0]['t'])]
    res = dict(n_rows=len(rows), n_on=len(on), events=[evt_name(e) for e in evts])
    if on:
        res.update(on_s=(on[-1]['t'] - on[0]['t']) / 1e3, out_mean=statistics.mean(r['out'] for r in on),
                   out_max=max(r['out'] for r in on), on_states=sorted({STATE[int(r['st'])] for r in on}),
                   k_on=statistics.mean(r['k'] for r in on), roll_std_on=std([r['roll'] for r in on]),
                   pitch_std_on=std([r['pitch'] for r in on]), vb_min_on=min(r['vb'] for r in on),
                   t_on_ms=on[0]['t'], t_last_on_ms=on[-1]['t'])
        after = [r for r in rows if r['t'] > on[-1]['t']]
        res['zero_after'] = bool(after) and all(r['out'] == 0 for r in after)
        # 危险方向的不一致：状态机已不在通电状态（DISARMED / ARMED）却仍在写非 0（降落末段指令很小写成 0 属正常，不计）
        res['inconsistent_rows'] = sum(1 for r in rows if r['st'] < 2 and r['out'] > 1e-6)
    if off_before:
        res.update(k_off=statistics.mean(r['k'] for r in off_before), roll_std_off=std([r['roll'] for r in off_before]),
                   pitch_std_off=std([r['pitch'] for r in off_before]), vb_off=statistics.mean(r['vb'] for r in off_before))
    stop = [e for e in evts if evt_name(e) in ('landed', 'disarm_cmd', 'land_timeout_kill', 'impact', 'estimator_fail', 'motor_watchdog_kill')]
    if stop and on:
        res['stop_evt'] = evt_name(stop[0])
        res['stop_lag_ms'] = on[-1]['t'] - evt_t(stop[0])      # 最后一行通电 − 停转事件时刻（≤ 20 ms 遥测周期为正常）
    return res


def check_common(r):
    ok, why = True, []
    if r.get('n_on', 0) < 25:
        return False, ['电机几乎没通电（<0.5 s）']
    if r['out_max'] > OUT_FULL + 0.005:
        ok = False; why.append(f'实际占空比之和 {r["out_max"]:.3f} 超过上限 {OUT_FULL:.2f}')
    if not r.get('zero_after'):
        ok = False; why.append('停转后仍有非 0 输出')
    if r.get('inconsistent_rows', 0) > 0:
        ok = False; why.append(f'状态与实际输出不一致 {r["inconsistent_rows"]} 行')
    if 'stop_lag_ms' in r and r['stop_lag_ms'] > 40:
        ok = False; why.append(f'停转事件后 {r["stop_lag_ms"]:.0f} ms 仍在输出')
    return ok, why


# ---------------------------------------------------------------- 台架步骤
def wait_link(b):
    while True:
        st = b.get_status()
        wifi = st.get('wifi', ''); lk = st.get('link_ok', '0')
        if lk == '1':
            print(f'   ✓ 手机链路在线（{wifi}）'); return
        input(f'   手机链路不在线（{wifi or "?"}）。请手机连热点、打开 http://192.168.4.1 并保持亮屏，好了按回车 ')


def run_phone_step(b, title, how, mid_prompt=None, mid_after_s=2.0, max_s=45.0):
    """how：给用户的操作说明；mid_prompt：电机转起来 mid_after_s 秒后打印的动作提示（急停 / 锁屏）"""
    wait_link(b)
    input(f'→【{title}】{how}\n   准备好了按回车，然后在手机上操作 ')
    b.evts = []; rows = []; t_on = None; prompted = False; t0 = time.time(); t_stop = None
    print('   等待你在手机上 解锁 → 起飞 ……（60 秒内）', flush=True)
    while True:
        new = b.poll(); rows += new
        for r in new:
            if r['out'] > 1e-6 and t_on is None:
                t_on = time.time(); print('   电机已通电。', flush=True)
        if t_on is None and time.time() - t0 > 60:
            print('   60 秒内电机没有通电。'); break
        if mid_prompt and t_on and not prompted and time.time() - t_on > mid_after_s:
            print(f'\n   >>> {mid_prompt} <<<\n', flush=True); prompted = True
        if t_on and rows and rows[-1]['st'] == 0 and t_stop is None:
            t_stop = time.time()
        if t_stop and time.time() - t_stop > 1.0:
            break                                   # 停转后再多录 1 s，确认输出保持 0
        if t_on and time.time() - t_on > max_s:
            print(f'   ⚠ 电机通电超过 {max_s:.0f} s 仍未停，脚本发串口急停'); b.estop(); time.sleep(0.5); rows += b.poll(); break
    r = analyse(rows, b.evts)
    if r.get('n_on'):
        print(f'   通电 {r["on_s"]:.1f} s（状态 {"/".join(r["on_states"])}），实际占空比之和 平均 {r["out_mean"]:.3f} 最大 {r["out_max"]:.3f}'
              f'（上限 {OUT_FULL:.2f}），停转事件 {r.get("stop_evt", "?")}')
        print(f'   事件：{" → ".join(r["events"])}')
        print(f'   电机转时：门控增益 k {r["k_on"]:.2f}（停时 {r.get("k_off", float("nan")):.2f}）  roll 抖动 {r["roll_std_on"]:.2f}°'
              f'（停时 {r.get("roll_std_off", float("nan")):.2f}°）  电池最低 {r["vb_min_on"]:.2f} V')
    return r


def step_a(b):
    r = run_phone_step(b, 'A 起飞超时 → 自动降落 → 触地停桨',
                       '飞机平放在桌面。手机上「按住解锁」，再「按住起飞」，然后不要碰手机和飞机。\n'
                       '   预期：4 个电机一起低速转（拆桨、6%），约 6 秒后转入降落，再约 1~2 秒自己停。')
    ok, why = check_common(r)
    ev = r.get('events', [])
    if not ('takeoff' in ev and 'land_takeoff_timeout' in ev and 'landed' in ev):
        ok = False; why.append('事件应为 takeoff → land_takeoff_timeout → landed')
    if r.get('n_on') and not (OUT_FULL * 0.8 <= r['out_mean'] <= OUT_FULL + 0.005):
        ok = False; why.append(f'起飞段应 4 路打满上限（之和≈{OUT_FULL:.2f}），实测平均 {r["out_mean"]:.3f}')
    if r.get('n_on') and not (5.5 <= r['on_s'] <= 10.0):
        ok = False; why.append(f'通电时长 {r["on_s"]:.1f} s 不在 5.5~10 s')
    r['user_all4'] = yes('   你看到 / 听到 4 个电机都转了、转速差不多吗？')
    r['user_stopped'] = yes('   最后电机自己停了吗？')
    if not (r['user_all4'] and r['user_stopped']):
        ok = False; why.append('你观察到的不符合预期')
    return ok, why, r


def step_b(b):
    r = run_phone_step(b, 'B 急停', '手机上「按住解锁」→「按住起飞」，电机转起来后等脚本提示再按「急停」。',
                       mid_prompt='现在按手机上的「急停」')
    ok, why = check_common(r)
    if r.get('stop_evt') != 'disarm_cmd':
        ok = False; why.append(f'停转原因应为 disarm_cmd，实际 {r.get("stop_evt")}')
    r['user_stopped'] = yes('   按急停后电机是不是立刻停了？')
    if not r['user_stopped']:
        ok = False; why.append('你观察到急停后没有立刻停')
    return ok, why, r


def step_c(b):
    r = run_phone_step(b, 'C 链路丢失', '手机上「按住解锁」→「按住起飞」，电机转起来后等脚本提示，**不锁屏**，\n'
                       '   直接从屏幕顶部下拉快捷开关关掉 WLAN（或开飞行模式）——这样测的是纯粹的心跳超时。',
                       mid_prompt='现在关掉手机 WLAN（下拉快捷开关），不要锁屏')
    ok, why = check_common(r)
    ev = r.get('events', [])
    cause = 'land_link_lost' if 'land_link_lost' in ev else ('land_cmd' if 'land_cmd' in ev else None)
    r['land_cause'] = cause
    if cause == 'land_cmd':
        print('   注：这次是网页切到后台时主动发了「降落」（land_cmd），不是心跳超时；也是正确的保护路径。')
    if not (cause and r.get('stop_evt') == 'landed'):
        ok = False; why.append('事件应为 land_link_lost（或 land_cmd）→ landed')
    else:
        tl = [evt_t(e) for e in b.evts if evt_name(e) == cause][0]
        r['link_to_stop_s'] = (r['t_last_on_ms'] - tl) / 1e3
        print(f'   链路丢失 → 停转 {r["link_to_stop_s"]:.2f} s')
        if r['link_to_stop_s'] > 3.0:
            ok = False; why.append(f'链路丢失后 {r["link_to_stop_s"]:.1f} s 才停（应 < 3 s）')
    r['user_stopped'] = yes('   关掉 WLAN 后电机在 1~2 秒内停了吗？')
    if not r['user_stopped']:
        ok = False; why.append('你观察到断开后没有及时停')
    return ok, why, r


def save(results, st_end, restored):
    out = dict(port=PORT, time=time.strftime('%Y-%m-%d %H:%M:%S'), u_max=U_MAX, steps=results,
               passed=sum(s['ok'] for s in results), total=len(results), status_end=st_end,
               reset_events=b_global.resets if b_global else [], restored_dryrun=restored)
    json.dump(out, open(os.path.join(HERE, 'motor_bench.json'), 'w'), ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, 'motor_bench_raw.csv'), 'w') as f:
        f.write(','.join(COLS) + '\n')
        for v in raw:
            f.write(','.join(f'{v[c]:g}' for c in COLS) + '\n')
    with open(os.path.join(HERE, 'motor_bench_events.txt'), 'w', encoding='utf-8') as f:
        for t, e in events:
            f.write(f'{t} {e}\n')


b_global = None


def main():
    global b_global
    print(__doc__.split('\n用法')[0])
    print('=' * 60)
    print('⚠ 这一步电机会真转（油门上限 6%）。必须先拆下全部 4 个桨。')
    if input('   确认 4 个桨都已拆下，请输入「已拆桨」：').strip() != '已拆桨':
        sys.exit('未确认拆桨，退出（什么都没做）。')
    if not NO_FLASH:
        if not glob.glob(PORT):
            sys.exit(f'✗ 找不到串口 {PORT}，请插上 USB。')
        flash('build_mc', 'MOTOR OUTPUT u_max=0.06f', '电机驱动版（6%）')
    b = Board(); b_global = b
    wait_boot(b, expect_motor=True)
    results, restored = [], False
    try:
        for name, fn in (('A 起飞超时→自动降落→触地停桨', step_a), ('B 急停', step_b), ('C 链路丢失', step_c)):
            print('\n' + '-' * 60)
            ok, why, data = fn(b)
            print(f'   {"✓" if ok else "✗"} {name}' + ('' if ok else '：' + '；'.join(why)))
            results.append(dict(step=name, ok=bool(ok), why=why, **data))
            if not ok and not yes('   这一步没通过，还继续下一步吗？'):
                break
    finally:
        b.estop()
    st_end = b.get_status()
    print('\n================ 结果 ================')
    for r in results:
        print(f'  {"✓" if r["ok"] else "✗"} {r["step"]}' + ('' if r['ok'] else '：' + '；'.join(r['why'])))
    print(f'通过 {sum(r["ok"] for r in results)}/{len(results)}')
    print(f'结束时：状态 {st_end.get("state", "?")}  电机引脚 {st_end.get("motor_pins_level", "?")}  看门狗锁死 {st_end.get("motor_killed", "?")}')
    print(f'时序：{st_end.get("loops", "?")}')
    print(f'周期：{st_end.get("period", "?")}')
    if b.resets:
        print(f'⚠ 复位 / 串口异常 {len(b.resets)} 次：' + ' | '.join(b.resets))
    try:
        b.ser.close()
    except Exception:
        pass
    if yes('\n烧回干跑版固件（电机驱动不编译，推荐）？'):
        flash('build', 'dry-run: phone control', '干跑版')
        restored = True
    save(results, st_end, restored)
    print('已保存 motor_bench.json、motor_bench_raw.csv、motor_bench_events.txt。把上面的结果或 json 发给我就行。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if b_global:
            b_global.estop()
            print('\n已发串口急停并退出。⚠ 板上可能仍是电机驱动版，下次运行用 --no-flash 或让我烧回干跑版。')
        else:
            print('\n已取消')
