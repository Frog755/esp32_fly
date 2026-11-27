#!/usr/bin/env python3
"""vib_check.py —— Stage 30 装桨振动检查（电机会真转，油门上限 30%；飞机离不了地）

★ 前提：4 个桨已装好（方向见 FIRST_FLIGHT.md）、已系留、半径 2 m 内没人没东西；电池充满（≥ 4.10 V）；手机能连热点。
为什么 30%：第 1 次（40%）实测飞机已轻微离地、平移（真机推力远大于仿真模型），40% ≈ 推力等于重力；
  按推力 ∝ 油门² 粗估，30% ≈ 重力的 55~60%，不会离地。固件需含加计低通（acc_lpf，Stage 30 修复），脚本烧前核对。
流程：
  1. 输入「已装桨已系留」→ 核对镜像后烧 build_vib（u_max=0.30、含 acc_lpf）→ 开机核对
  2. 默认 **USB 不拔**（2026-11-27 用户反馈插拔 USB 有时会让飞机断电，疑似焊点问题）：脚本通过串口实时看状态、自动检测停转；
     电机干扰万一让 USB 复位 → 芯片复位 → 电机停（贴地 30%，安全），黑匣子上一槽保留复位前的数据。`--unplug` 用旧流程（拔 USB）。
  3. 手机：按住解锁 → 按住起飞 → 电机 30% 转约 6 s（应一直贴地）→ 自动降落 → 约 1~2 s 后自己停
  4. 停稳后（拔 USB 流程：插回 USB 按回车）→ 读出黑匣子 → 分析振动对姿态估计的影响 → 默认烧回干跑版
判据（飞机贴在地上不动，真实姿态不变，所以"电机转时的姿态估计 − 转之前"就是振动造成的误差）：
  姿态偏差 < 3° 正常；3~8° 需要处理振动后再飞；> 8° 不能飞。另报门控关闭占比、振动、陀螺抖动、偏航漂移。
Stage 31：同时用电脑麦克风录电机声音（默认开启，--no-audio 关闭）→ 测转速 → 估推力量级（audio_rpm.py）。
  电脑放在飞机 1 m 以内（USB 线够长即可）。
输出：blackbox/vib_<时间>.json + 原始读出 + CSV + 录音。把终端最后的结果发给我。
用法：./vib_check.sh [串口]   --no-flash 跳过烧录   --analyse-file X.txt 只离线分析一份已保存的读出
"""
import sys, os, time, json, glob, subprocess, statistics as stt, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bb_read
try:
    import audio_rpm
except Exception:
    audio_rpm = None

args = [a for a in sys.argv[1:] if not a.startswith('--')]
PORT = args[0] if args and not args[0].endswith('.txt') else '/dev/ttyACM0'
STATE = ['DISARMED', 'ARMED', 'TAKEOFF', 'RECOVER', 'HOLD', 'LAND']
U_MAX = 0.30
TAG = 'MOTOR OUTPUT u_max=0.30f, acc_lpf=10.0f'


def yes(prompt):
    while True:
        a = input(prompt + ' [y/n] ').strip().lower()
        if a in ('y', 'yes', '是', '对', '1'):
            return True
        if a in ('n', 'no', '否', '不', '0'):
            return False


def flash(build_dir, must_contain, label):
    bdir = os.path.join(HERE, build_dir)
    fa, app = os.path.join(bdir, 'flash_args'), os.path.join(bdir, 'flight_v9.bin')
    if not (os.path.exists(fa) and os.path.exists(app)):
        sys.exit(f'✗ 找不到 {build_dir}/flash_args 或 flight_v9.bin，请先让我编译')
    if must_contain.encode() not in open(app, 'rb').read():
        sys.exit(f'✗ {build_dir}/flight_v9.bin 里没有「{must_contain}」，不是预期的{label}固件，停止。')
    argv = open(fa).read().split()
    cmd = [sys.executable, '-m', 'esptool', '--chip', 'esp32s3', '-p', PORT, '-b', '460800',
           '--before', 'default-reset', '--after', 'hard-reset', 'write-flash'] + argv
    for attempt in (1, 2):
        print(f'   烧录{label}固件（{build_dir}）……', flush=True)
        r = subprocess.run(cmd, cwd=bdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0 and 'Hash of data verified' in r.stdout:
            print(f'   ✓ 已烧录{label}固件'); return
        print(r.stdout[-800:]); time.sleep(2)
    sys.exit(f'✗ 烧录{label}固件失败（重试 1 次后）')


def wait_port(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        if glob.glob(PORT):
            return True
        time.sleep(0.3)
    return False


def open_and_wait_boot(sec=25):
    """打开串口，等到有 F 行（开机约 8~10 s），返回 (serial, status dict)"""
    import serial
    if not wait_port(sec):
        sys.exit(f'✗ {sec} s 内没看到串口 {PORT}')
    ser = serial.Serial(PORT, 115200, timeout=0.1); time.sleep(0.2)
    buf, t0, got_f = '', time.time(), False
    while time.time() - t0 < sec and not got_f:
        buf += ser.read(8192).decode('utf-8', 'replace')
        got_f = '\nF ' in buf
    if not got_f:
        sys.exit('✗ 没收到飞控数据（固件没起来？按一下复位键再试）')
    time.sleep(0.5); ser.reset_input_buffer(); ser.write(b'status\n')
    st, buf, t0 = {}, '', time.time()
    while time.time() - t0 < 3 and 'STATUS_END' not in buf:
        buf += ser.read(8192).decode('utf-8', 'replace')
    for ln in buf.split('\n'):
        p = ln.strip().split()
        if len(p) >= 2 and p[0] == 'STATUS':
            st[p[1]] = ' '.join(p[2:])
    return ser, st


# ---------------------------------------------------------------- USB 不拔时的实时监视
class UsbWatch:
    """逐行喂串口输出，判断：电机开始转（on）、停转且稳定（done）、芯片复位（resets）。只读，不往板子发心跳——链路只来自手机。"""
    def __init__(self):
        self.on_seen = False; self.off_since = None; self.resets = 0; self.reset_reasons = []
        self.last = None; self.events = []; self.done = False; self.vb_min = None; self.out_max = 0.0; self.st = None

    def feed(self, line, now):
        p = line.strip().split()
        if not p:
            return None
        if p[0] == 'F' and len(p) == 19:
            try:
                st, out, vb = int(float(p[2])), float(p[16]), float(p[10])
                roll, pitch = float(p[4]), float(p[5])
            except ValueError:
                return None
            self.st = st; self.last = dict(st=st, out=out, vb=vb, roll=roll, pitch=pitch)
            if out > 1e-6 or st >= 2:
                self.on_seen = True; self.off_since = None
                self.out_max = max(self.out_max, out); self.vb_min = vb if self.vb_min is None else min(self.vb_min, vb)
            elif self.on_seen:
                self.off_since = now if self.off_since is None else self.off_since
                if now - self.off_since > 1.5:
                    self.done = True
            return 'F'
        if p[0] == 'EVT':
            self.events.append(line.strip()); return 'EVT'
        if p[0] == 'BOOT' and len(p) >= 3 and p[1] == 'reset_reason':
            self.resets += 1; self.reset_reasons.append(p[2])
            if self.on_seen:
                self.done = True          # 通电中复位：电机已停（开机先锁低引脚），这次就到这里
            return 'RESET'
        return None


# ---------------------------------------------------------------- 分析（纯函数，可离线跑）
def analyse(slots):
    """在两个槽里找最近一次通电段，统计振动对估计的影响"""
    cand = []
    for si, d in slots.items():
        if not d.get('valid') or not d['recs']:
            continue
        rs = [bb_read.scaled(r) for r in d['recs']]
        on_idx = [i for i, s in enumerate(rs) if s['on']]
        if on_idx:
            cand.append((d['boot'], si, d, rs, on_idx))
    if not cand:
        return dict(found=False, msg='黑匣子里没有电机通电段（手机上解锁 + 起飞了吗？）')
    boot, si, d, rs, on_idx = max(cand, key=lambda c: c[0])
    # 最后一段连续通电
    end = on_idx[-1]; start = end
    while start - 1 >= 0 and rs[start - 1]['on'] and rs[start - 1]['idx'] == rs[start]['idx'] - 1:
        start -= 1
    on = rs[start:end + 1]
    pre = [s for s in rs[max(0, start - 600):start] if not s['on'] and s['st'] in (0, 1)][-400:]   # 通电前最多 2 s
    post = rs[end + 1:end + 1 + 400]
    eu = lambda s: bb_read.euler(s)
    def m(xs): return stt.mean(xs) if xs else float('nan')
    def sd(xs): return stt.pstdev(xs) if len(xs) > 1 else float('nan')
    def pq(xs, q):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float('nan')
    r0 = m([eu(s)[0] for s in pre]); p0 = m([eu(s)[1] for s in pre]); y0 = m([eu(s)[2] for s in pre])
    droll = [eu(s)[0] - r0 for s in on]; dpitch = [eu(s)[1] - p0 for s in on]
    dyaw = [((eu(s)[2] - y0 + 180) % 360) - 180 for s in on]
    tilt_err = [math.hypot(a, b) for a, b in zip(droll, dpitch)]
    an = lambda s: math.sqrt(s['ax_g'] ** 2 + s['ay_g'] ** 2 + s['az_g'] ** 2)
    evs = [f'{s["t"]:.2f}s {s["ev"]}' for s in rs[max(0, start - 1200):min(len(rs), end + 400)] if s['ev']]
    res = dict(
        found=True, slot=si, boot=boot, reset_name=d['reset_name'], complete=d.get('complete'),
        on_s=len(on) / 200, events=evs, motor_fw=bool(on[0]['motor_fw']), killed=any(s['killed'] for s in on),
        out_mean=m([s['out'] for s in on]), out_max=max(s['out'] for s in on), out_cap=4 * U_MAX,
        tilt_err_max=max(tilt_err), tilt_err_p95=pq(tilt_err, 0.95), tilt_err_mean=m(tilt_err),
        roll_dev_mean=m(droll), pitch_dev_mean=m(dpitch), roll_std_on=sd([eu(s)[0] for s in on]), pitch_std_on=sd([eu(s)[1] for s in on]),
        yaw_drift_deg=dyaw[-1] if dyaw else float('nan'),
        k_med_on=stt.median(s['k'] for s in on), k_lt05_on=sum(s['k'] < 0.5 for s in on) / len(on),
        k_med_pre=stt.median(s['k'] for s in pre) if pre else float('nan'),
        vib_med_on=stt.median(s['vib_g'] for s in on), vib_p95_on=pq([s['vib_g'] for s in on], 0.95),
        vib_med_pre=stt.median(s['vib_g'] for s in pre) if pre else float('nan'),
        anorm_max_on=max(an(s) for s in on), anorm_med_on=stt.median(an(s) for s in on),
        gyro_std_on=[sd([s[k] for s in on]) for k in ('gx_dps', 'gy_dps', 'gz_dps')],
        gyro_std_pre=[sd([s[k] for s in pre]) for k in ('gx_dps', 'gy_dps', 'gz_dps')],
        z_pre=m([s['z'] for s in pre]), z_max_on=max(s['z'] for s in on), vb_pre=m([s['vb'] for s in pre]), vb_min_on=min(s['vb'] for s in on),
        per_max_us=max(s['per_max_us'] for s in on), exec_max_us=max(s['exec_max_us'] for s in on),
        u_cmd_mean=[m([s[k] for s in on]) for k in ('u1', 'u2', 'u3', 'u4')],
        post_out_zero=all(s['out'] == 0 for s in post) if post else None)
    # ---- 判定 ----
    names = [e.split()[1] for e in evs]
    seq_ok = all(n in names for n in ('armed', 'takeoff', 'land_takeoff_timeout', 'landed'))
    checks = [
        ('流程：armed → takeoff → land_takeoff_timeout → landed', seq_ok),
        (f'电机驱动版、实际输出不超过 4×{U_MAX:.0%}（之和 ≤ {4 * U_MAX:.2f}，实测最大 {res["out_max"]:.3f}）', res['motor_fw'] and res['out_max'] <= 4 * U_MAX + 0.005),
        ('停转后输出保持 0、看门狗未触发', bool(res['post_out_zero']) and not res['killed']),
        (f'1 kHz 最长周期 {res["per_max_us"]} μs < 2 ms', res['per_max_us'] < 2000),
    ]
    te = res['tilt_err_max']
    verdict = '正常' if te < 3 else ('需要处理振动' if te < 8 else '不能飞')
    res.update(seq_ok=seq_ok, checks=[(c, bool(ok)) for c, ok in checks], attitude_verdict=verdict,
               passed=all(ok for _, ok in checks) and te < 3)
    return res


def report(r):
    if not r['found']:
        print('✗ ' + r['msg']); return
    print(f'槽 {r["slot"]}（开机号 {r["boot"]}，复位原因 {r["reset_name"]}）  通电 {r["on_s"]:.2f} s  读出完整 {"✓" if r["complete"] else "✗"}')
    print('   事件：' + ' → '.join(r['events']))
    for c, ok in r['checks']:
        print(f'   {"✓" if ok else "✗"} {c}')
    print(f'\n   【姿态估计误差（飞机贴地不动，电机转时 − 转之前）】最大 {r["tilt_err_max"]:.2f}°  P95 {r["tilt_err_p95"]:.2f}°  平均 {r["tilt_err_mean"]:.2f}°'
          f'  → {r["attitude_verdict"]}（< 3° 正常，3~8° 先处理振动，> 8° 不能飞）')
    print(f'      平均偏移 roll {r["roll_dev_mean"]:+.2f}°  pitch {r["pitch_dev_mean"]:+.2f}°；抖动 roll {r["roll_std_on"]:.2f}°  pitch {r["pitch_std_on"]:.2f}°；'
          f'偏航漂移 {r["yaw_drift_deg"]:+.1f}°（{r["on_s"]:.1f} s 内）')
    print(f'   【门控 Mahony】k 中位 {r["k_med_pre"]:.2f}（转之前）→ {r["k_med_on"]:.2f}（转时），k<0.5（加计修正基本关闭）占 {r["k_lt05_on"] * 100:.0f}%')
    print(f'   【振动】中位 {r["vib_med_pre"]:.3f} g → {r["vib_med_on"]:.3f} g，P95 {r["vib_p95_on"]:.3f} g；比力模长 中位 {r["anorm_med_on"]:.2f} g 最大 {r["anorm_max_on"]:.2f} g')
    gp, go = r['gyro_std_pre'], r['gyro_std_on']
    print(f'   【陀螺抖动 °/s】转之前 {gp[0]:.2f} / {gp[1]:.2f} / {gp[2]:.2f} → 转时 {go[0]:.2f} / {go[1]:.2f} / {go[2]:.2f}')
    print(f'   【其他】实际占空比之和 平均 {r["out_mean"]:.3f}（上限 {r["out_cap"]:.2f}）；控制器指令 M1~M4 平均 '
          + ' / '.join(f'{u:.2f}' for u in r['u_cmd_mean']) + f'；电池 {r["vb_pre"]:.2f} → 最低 {r["vb_min_on"]:.2f} V；'
          f'高度估计 {r["z_pre"]:+.2f} → 最高 {r["z_max_on"]:+.2f} m（桨下气流会影响气压计，仅供参考）')
    print('\n' + ('✓ 振动检查通过（振动不影响姿态估计）。注意：首飞前仍需修正推力模型，现固件装桨不能真飞' if r['passed'] else '✗ 有项目没通过，先把结果发给我'))


def main():
    print(__doc__.split('\n用法')[0]); print('=' * 64)
    if '--analyse-file' in sys.argv:
        src = sys.argv[sys.argv.index('--analyse-file') + 1]
        r = analyse(bb_read.parse(open(src).read().split('\n'))); report(r); return
    print(f'⚠ 这一步电机会真转（{U_MAX:.0%}），桨是装着的。人离开 1 m 以上，手不要靠近桨。')
    if input('   确认 4 个桨已装好、已系留、周围 2 m 没人没东西，请输入「已装桨已系留」：').strip() != '已装桨已系留':
        sys.exit('未确认，退出（什么都没做）。')
    if not yes('   电池充满了吗（≥ 4.10 V）？桨的方向核对过吗（左前 / 右后 顺时针，右前 / 左后 逆时针，俯视）？'):
        sys.exit('先充电 / 核对桨向再来。')
    if '--no-flash' not in sys.argv:
        if not glob.glob(PORT):
            sys.exit(f'✗ 找不到串口 {PORT}，请插上 USB。')
        flash('build_vib', TAG, f'振动检查版（油门上限 {U_MAX:.0%}、含加计低通）')
    ser, st = open_and_wait_boot()
    print(f'   固件 {st.get("fw", "?")}')
    print(f'   电机驱动 {st.get("motor_output")}  油门上限 {st.get("u_max")}  状态 {st.get("state")}  电池 {st.get("vbatt")} V  黑匣子 {st.get("blackbox")}')
    if st.get('motor_output') != '1' or abs(float(st.get('u_max') or 0) - U_MAX) > 1e-4 or st.get('state') != 'DISARMED' or 'acc_lpf' not in st.get('fw', ''):
        sys.exit(f'✗ 板上不是「电机驱动版、油门上限 {U_MAX:.2f}、未解锁」，停止。')
    try:
        if float(st.get('vbatt', 0)) < 4.0:
            print(f'   ⚠ 电池 {st.get("vbatt")} V 偏低（< 4.0 V），建议先充电。')
    except ValueError:
        pass
    keep_usb = '--unplug' not in sys.argv
    if not keep_usb:
        ser.close()
    rec, wav = None, None
    if audio_rpm and '--no-audio' not in sys.argv:
        os.makedirs(os.path.join(HERE, 'blackbox'), exist_ok=True)
        wav = os.path.join(HERE, 'blackbox', 'vib_' + time.strftime('%Y%m%d_%H%M%S') + '_audio.wav')
        try:
            rec = audio_rpm.Recorder(wav, max_s=300)
            print(f'   🎙 已开始用电脑麦克风录音（{rec.src or "默认输入"}），电脑放在飞机 1 m 以内，录音期间别说话 / 别放音乐')
        except Exception as e:
            print(f'   ⚠ 录音没启动（{e}），继续，只做振动分析')
            rec, wav = None, None
    print('\n' + '-' * 64)
    watch = None
    if keep_usb:
        print('现在按顺序做（**USB 不用拔**）：')
        print('  1) 飞机放回系留点、放平，人退到 1 m 外；USB 线理顺，别靠近桨')
        print('  2) 手机连热点、打开 http://192.168.4.1：「按住解锁」→「按住起飞」')
        print(f'     电机以 {U_MAX:.0%} 转约 6 秒 → 自动降落 → 再约 1~2 秒自己停；飞机应一直贴在地上')
        print('  3) 什么都不用按：脚本看到电机停稳后自动读黑匣子')
        print('  异常时：手机「急停」，或在这里按 Ctrl+C（脚本通过串口急停）\n')
        watch = UsbWatch(); buf = ''; t0 = time.time(); t_print = 0.0
        try:
            while not watch.done and time.time() - t0 < 180:
                try:
                    data = ser.read(4096)
                except (serial.SerialException, OSError):
                    print('\n   ⚠ 串口断开（多半是电机干扰让 USB 复位）——等它恢复……', flush=True)
                    try:
                        ser.close()
                    except Exception:
                        pass
                    if not wait_port(20):
                        print('   ✗ 20 s 内串口没回来。电机应已因芯片复位停转；请检查飞机，再插拔 USB 后用 --no-flash 重跑读数'); break
                    time.sleep(0.5); ser = serial.Serial(PORT, 115200, timeout=0.1); watch.resets += 1; watch.reset_reasons.append('serial_lost')
                    if watch.on_seen:
                        watch.done = True
                    continue
                buf += data.decode('utf-8', 'replace')
                while '\n' in buf:
                    ln, buf = buf.split('\n', 1)
                    k = watch.feed(ln, time.time())
                    if k == 'EVT':
                        print('\n   ' + ln.strip(), flush=True)
                    elif k == 'RESET':
                        print(f'\n   ⚠ 板子重新开机（复位原因 {watch.reset_reasons[-1]}）'
                              + ('——电机已停；黑匣子上一槽保留了复位前的数据' if watch.on_seen else ''), flush=True)
                if watch.last and time.time() - t_print > 1.0:
                    L = watch.last; t_print = time.time()
                    print(f'\r   状态 {STATE[L["st"]] if L["st"] < 6 else L["st"]:<8} 输出 {L["out"]:.2f}  电池 {L["vb"]:.2f} V  '
                          f'roll {L["roll"]:+5.1f}° pitch {L["pitch"]:+5.1f}°   ', end='', flush=True)
                if not watch.on_seen and time.time() - t0 > 120:
                    print('\n   120 s 内电机没通电（手机上解锁 + 起飞了吗？）'); break
        except KeyboardInterrupt:
            for _ in range(3):
                try:
                    ser.write(b' '); time.sleep(0.02)
                except Exception:
                    pass
            print('\n   已通过串口急停')
        print()
        if rec:
            print(f'   🎙 录音结束（{rec.stop():.0f} s）')
        time.sleep(1.0)
        try:
            ser.close()
        except Exception:
            pass
        if watch.resets and watch.on_seen:
            print('   等板子重新开机完成……'); s2, _ = open_and_wait_boot(30); s2.close()
    else:
        print('现在按顺序做：')
        print('  1) **拔掉 USB 线**（电池不要拔！拔电池黑匣子就没了）')
        print('  2) 飞机放回系留点、放平，人退到 1 m 外')
        print('  3) 手机连热点、打开 http://192.168.4.1：「按住解锁」→「按住起飞」')
        print(f'     电机以 {U_MAX:.0%} 转约 6 秒 → 自动降落 → 再约 1~2 秒自己停；飞机应一直贴在地上；异常随时按「急停」')
        print('  4) 电机停稳后，**插回 USB**')
        input('做完第 4 步按回车 ')
        if rec:
            print(f'   🎙 录音结束（{rec.stop():.0f} s）')
        print('   等板子出数据……')
        ser, st2 = open_and_wait_boot(30)
        print(f'   状态 {st2.get("state")}  黑匣子 {st2.get("blackbox")}')
        ser.close(); time.sleep(0.3)
    lines, slots, nref = bb_read.read_serial(PORT)
    od = os.path.join(HERE, 'blackbox'); os.makedirs(od, exist_ok=True)
    tag = time.strftime('%Y%m%d_%H%M%S')
    open(os.path.join(od, f'vib_{tag}_raw.txt'), 'w').write('\n'.join(lines) + '\n')
    for si, d in slots.items():
        if d.get('valid') and d['recs']:
            bb_read.write_csv(os.path.join(od, f'vib_{tag}_slot{si}_boot{d["boot"]}.csv'), d)
    r = analyse(slots)
    print('\n================ 振动检查结果 ================')
    report(r)
    if watch is not None:
        r['usb_kept'] = True; r['usb_resets'] = watch.resets; r['usb_reset_reasons'] = watch.reset_reasons
        print(f'   【USB 没拔】通电期间芯片复位 {watch.resets} 次' + (f'（{", ".join(watch.reset_reasons)}）' if watch.resets else '')
              + (f'；串口看到电池最低 {watch.vb_min:.2f} V' if watch.vb_min is not None else ''))
    if wav and os.path.exists(wav) and audio_rpm:
        print('\n================ 电机转速（录音）================')
        csvs = [os.path.join(od, f'vib_{tag}_slot{r["slot"]}_boot{r["boot"]}.csv')] if r.get('found') else []
        try:
            ar = audio_rpm.analyse_external(wav, U_MAX, vb=r.get('vb_min_on') if r.get('found') else None,
                                            bb_csv=csvs[0] if csvs and os.path.exists(csvs[0]) else None) or {}
            if ar.get('seg') and r.get('found'):
                print(f'   （黑匣子通电 {r["on_s"]:.1f} s，录音里电机响 {ar["seg"]["dur"]:.1f} s）')
            r['audio'] = {k: v for k, v in ar.items() if k != 'path'}
        except Exception as e:
            print(f'   ⚠ 录音分析失败：{e}（录音文件已保存，发给我也行）')
        print(f'   录音文件：{os.path.relpath(wav, HERE)}')
    r['user_lifted'] = yes('\n   飞机离地了吗（哪怕一侧翘起来）？离地答 y，一直贴在地上答 n')
    r['user_note_ok'] = yes('   电机声音正常、4 个电机转速听起来差不多、没有异响 / 明显抖动吗？')
    if r['user_lifted']:
        print(f'   ⚠ {U_MAX:.0%} 也离地了：推力比粗估还大——先把结果发我。')
    json.dump(r, open(os.path.join(od, f'vib_{tag}.json'), 'w'), ensure_ascii=False, indent=1)
    print(f'   已保存 blackbox/vib_{tag}.json、原始读出与 CSV')
    if yes('\n烧回干跑版固件（电机驱动不编译，推荐）？'):
        flash('build', 'dry-run: phone control', '干跑版')
    print('把上面「振动检查结果」整段发给我就行。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已取消。⚠ 板上可能仍是振动检查版（电机驱动版），下次运行用 --no-flash 或让我烧回干跑版。')
