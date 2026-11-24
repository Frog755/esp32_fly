#!/usr/bin/env python3
"""vib_check.py —— Stage 30 装桨振动检查（电机会真转，油门上限 40%；飞机离不了地）

★ 前提：4 个桨已装好（方向见 FIRST_FLIGHT.md）、已系留、半径 2 m 内没人没东西；电池充满（≥ 4.10 V）；手机能连热点。
为什么 40% 不是 50%：仿真单电机最大推力 0.27 N 是估计值、没台架标定过。若真机推力比仿真大一倍，
  50% 时推力可达重力 ~90%（加上贴地效应可能离地）；40% 时仍 < 60%，留足余量。
流程：
  1. 输入「已装桨已系留」→ 核对镜像后烧 build_vib（u_max=0.40）→ 开机核对
  2. **拔掉 USB（电池别拔）**——电机 EMI 会让插着的 USB 复位，进而让芯片复位
  3. 手机：按住解锁 → 按住起飞 → 电机 40% 转约 6 s → 自动降落 → 约 1~2 s 后自己停
  4. 停稳后插回 USB，按回车 → 读出黑匣子 → 分析振动对姿态估计的影响 → 默认烧回干跑版
判据（飞机贴在地上不动，真实姿态不变，所以"电机转时的姿态估计 − 转之前"就是振动造成的误差）：
  姿态偏差 < 3° 正常；3~8° 需要处理振动后再飞；> 8° 不能飞。另报门控关闭占比、振动、陀螺抖动、偏航漂移。
输出：blackbox/vib_<时间>.json + 原始读出 + CSV。把终端最后的结果发给我。
用法：./vib_check.sh [串口]   --no-flash 跳过烧录   --analyse-file X.txt 只离线分析一份已保存的读出
"""
import sys, os, time, json, glob, subprocess, statistics as stt, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bb_read

args = [a for a in sys.argv[1:] if not a.startswith('--')]
PORT = args[0] if args and not args[0].endswith('.txt') else '/dev/ttyACM0'
U_MAX = 0.40
TAG = 'MOTOR OUTPUT u_max=0.40f'


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
    print('\n' + ('✓ 振动检查通过，可以进入系留首飞' if r['passed'] else '✗ 有项目没通过，先把结果发给我'))


def main():
    print(__doc__.split('\n用法')[0]); print('=' * 64)
    if '--analyse-file' in sys.argv:
        src = sys.argv[sys.argv.index('--analyse-file') + 1]
        r = analyse(bb_read.parse(open(src).read().split('\n'))); report(r); return
    print('⚠ 这一步电机会真转（40%），桨是装着的。人离开 1 m 以上，手不要靠近桨。')
    if input('   确认 4 个桨已装好、已系留、周围 2 m 没人没东西，请输入「已装桨已系留」：').strip() != '已装桨已系留':
        sys.exit('未确认，退出（什么都没做）。')
    if not yes('   电池充满了吗（≥ 4.10 V）？桨的方向核对过吗（左前 / 右后 顺时针，右前 / 左后 逆时针，俯视）？'):
        sys.exit('先充电 / 核对桨向再来。')
    if '--no-flash' not in sys.argv:
        if not glob.glob(PORT):
            sys.exit(f'✗ 找不到串口 {PORT}，请插上 USB。')
        flash('build_vib', TAG, '振动检查版（油门上限 40%）')
    ser, st = open_and_wait_boot()
    print(f'   固件 {st.get("fw", "?")}')
    print(f'   电机驱动 {st.get("motor_output")}  油门上限 {st.get("u_max")}  状态 {st.get("state")}  电池 {st.get("vbatt")} V  黑匣子 {st.get("blackbox")}')
    if st.get('motor_output') != '1' or abs(float(st.get('u_max') or 0) - U_MAX) > 1e-4 or st.get('state') != 'DISARMED':
        sys.exit('✗ 板上不是「电机驱动版、油门上限 0.40、未解锁」，停止。')
    try:
        if float(st.get('vbatt', 0)) < 4.0:
            print(f'   ⚠ 电池 {st.get("vbatt")} V 偏低（< 4.0 V），建议先充电。')
    except ValueError:
        pass
    ser.close()
    print('\n' + '-' * 64)
    print('现在按顺序做：')
    print('  1) **拔掉 USB 线**（电池不要拔！拔电池黑匣子就没了）')
    print('  2) 飞机放回系留点、放平，人退到 1 m 外')
    print('  3) 手机连热点、打开 http://192.168.4.1：「按住解锁」→「按住起飞」')
    print('     电机以 40% 转约 6 秒 → 自动降落 → 再约 1~2 秒自己停；异常随时按「急停」')
    print('  4) 电机停稳后，**插回 USB**')
    input('做完第 4 步按回车 ')
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
    r['user_lifted'] = yes('\n   飞机离地了吗（哪怕一侧翘起来）？离地答 y，一直贴在地上答 n')
    r['user_note_ok'] = yes('   电机声音正常、4 个电机转速听起来差不多、没有异响 / 明显抖动吗？')
    if r['user_lifted']:
        print('   ⚠ 离地了：说明真机推力比仿真大，首飞油门上限要重新定——先把结果发我。')
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
