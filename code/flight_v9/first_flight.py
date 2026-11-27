#!/usr/bin/env python3
"""first_flight.py —— Stage 33 系留首飞（装桨、真飞，油门上限 70%）

★ 这是第一次真正离地飞行。脚本会先逐项确认检查清单（FIRST_FLIGHT.md 第 2 节），任何一项答"否"就退出、什么都不做。
流程：
  1. 检查清单 → 输入「系留首飞」→ 核对镜像（MOTOR OUTPUT u_max=0.70f, acc_lpf=10.0f）后烧 build_ff → 开机核对（电机驱动、上限、推力换算、电压）
  2. 电脑开始录音（测悬停转速，--no-audio 关闭）
  3. **拔掉 USB（电池别拔）**——飞行要拴绳活动，USB 线会缠桨；拔完看手机网页是否还连着（没连着 = 拔 USB 时断电了，先查电源）
  4. 手机「按住解锁」→「按住起飞」：升到约 0.8 m → 悬停 5 s → 以 0.35 m/s 自动降落 → 触地停桨（共约 10 s）。异常立刻「急停」
  5. 停稳后**插回 USB（电池别拔）**按回车 → 读黑匣子 → 飞行分析 → 默认烧回干跑版
输出：blackbox/ff_<时间>.json + 原始读出 + CSV + 录音。把「首飞结果」整段发给我。
用法：./first_flight.sh [串口]   --no-flash 跳过烧录   --no-audio 不录音   --analyse-file X.txt 只离线分析
"""
import sys, os, time, json, math, statistics as stt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bb_read
import vib_check as V          # 复用：烧录（镜像核对 + 重试）、等开机读 STATUS、yes()
try:
    import audio_rpm
except Exception:
    audio_rpm = None

U_MAX = 0.70
TAG = 'MOTOR OUTPUT u_max=0.70f, acc_lpf=10.0f'
DUTY_SCALE = 0.5553            # FC_DUTY_SCALE（flight_core.h：0.42 × 4.17 / (0.85246 × 3.7)），开机时用 STATUS thrust_map 覆盖
HOVER_BELIEF = 0.42
EXPECT = ['armed', 'takeoff', 'stable_hold', 'land_auto', 'landed']


# ---------------------------------------------------------------- 飞行分析（纯函数，可离线跑）
def analyse_flight(slots, scale=DUTY_SCALE, umax=U_MAX):
    cand = []
    for si, d in slots.items():
        if not d.get('valid') or not d['recs']:
            continue
        rs = [bb_read.scaled(r) for r in d['recs']]
        on_idx = [i for i, s in enumerate(rs) if s['on']]
        if on_idx:
            cand.append((d['boot'], si, d, rs, on_idx))
    if not cand:
        return dict(found=False, msg='黑匣子里没有电机通电段（手机上解锁 + 起飞了吗？插回 USB 时板子是不是断电了？）')
    boot, si, d, rs, on_idx = max(cand, key=lambda c: c[0])
    end = on_idx[-1]; start = end
    while start - 1 >= 0 and rs[start - 1]['on'] and rs[start - 1]['idx'] == rs[start]['idx'] - 1:
        start -= 1
    on = rs[start:end + 1]
    pre = [s for s in rs[max(0, start - 600):start] if not s['on']][-400:]
    post = rs[end + 1:end + 1 + 400]
    win = rs[max(0, start - 1200):min(len(rs), end + 400)]
    evs = [(s['t'], s['ev']) for s in win if s['ev']]
    names = [e for _, e in evs]
    t_of = lambda name: next((t for t, e in evs if e == name), None)
    m = lambda xs: stt.mean(xs) if xs else float('nan')
    eu = [bb_read.euler(s) for s in on]
    hold = [s for s in on if s['st'] == 4]
    flying = [s for s in on if s['st'] in (2, 3, 4)]
    # 推力修正系数：out = scale × c_k × Σu（没有任何电机被上限截断时成立）
    def ck_of(s):
        su = s['u1'] + s['u2'] + s['u3'] + s['u4']
        if su < 0.2:
            return None
        ck = s['out'] / (scale * su)
        umx = max(s['u1'], s['u2'], s['u3'], s['u4'])
        return ck if umx * scale * ck < umax - 0.005 else None
    ck_hold = [c for c in (ck_of(s) for s in hold) if c is not None]
    ck_all = [c for c in (ck_of(s) for s in on) if c is not None]
    def sat(s):
        umx = max(s['u1'], s['u2'], s['u3'], s['u4'])
        return umx >= 0.999 or (ck_of(s) is None and s['out'] > 0)
    z0 = m([s['z'] for s in pre[-100:]]) if pre else 0.0
    res = dict(
        found=True, slot=si, boot=boot, reset_name=d['reset_name'], complete=d.get('complete'),
        on_s=len(on) / 200, events=[f'{t:.2f}s {e}' for t, e in evs], killed=any(s['killed'] for s in on),
        t_to_hold=(t_of('stable_hold') - t_of('takeoff')) if t_of('stable_hold') and t_of('takeoff') else None,
        hold_s=len(hold) / 200, z_rise_max=max(s['z'] for s in on) - z0, z_rise_hold=m([s['z'] for s in hold]) - z0 if hold else None,
        roll_max=max(abs(e[0]) for e in eu), pitch_max=max(abs(e[1]) for e in eu),
        roll_std_hold=stt.pstdev([bb_read.euler(s)[0] for s in hold]) if len(hold) > 1 else None,
        pitch_std_hold=stt.pstdev([bb_read.euler(s)[1] for s in hold]) if len(hold) > 1 else None,
        yaw_drift=(lambda a, b: ((b - a + 180) % 360) - 180)(eu[0][2], eu[-1][2]),
        ck_hold=stt.median(ck_hold) if ck_hold else None, ck_end=ck_all[-1] if ck_all else None,
        duty_hold=m([s['out'] / 4 for s in hold]) if hold else None,
        sat_frac=sum(sat(s) for s in flying) / len(flying) if flying else None,
        k_lt05=sum(s['k'] < 0.5 for s in on) / len(on), vib_med=stt.median(s['vib_g'] for s in on),
        vb_static=m([s['vb'] for s in pre]) if pre else None, vb_min=min(s['vb'] for s in on),
        per_max_us=max(s['per_max_us'] for s in rs), link_lost=any(not s['link'] for s in on),
        post_zero=all(s['out'] == 0 for s in post) if post else None)
    stop = next((e for e in names if e in ('landed', 'impact', 'estimator_fail', 'disarm_cmd', 'land_timeout_kill')), None)
    land_why = next((e for e in names if e.startswith('land_')), None)
    res.update(stop=stop, land_why=land_why)
    ck = res['ck_hold']
    checks = [
        ('离地并进入悬停（stable_hold）', 'stable_hold' in names),
        ('悬停满 5 s 后自动降落（land_auto）', land_why == 'land_auto'),
        ('触地停桨（landed，不是撞击 / 估计失效 / 急停）', stop == 'landed'),
        (f'姿态最大 roll {res["roll_max"]:.1f}° / pitch {res["pitch_max"]:.1f}° < 25°', res['roll_max'] < 25 and res['pitch_max'] < 25),
        ('推力修正系数没有顶到限幅（0.75~1.30）' + (f'：悬停时 {ck:.3f}' if ck else '：没算出来'), ck is not None and 0.77 < ck < 1.28),
        (f'起飞 + 悬停期间顶上限 < 20%' + (f'（{res["sat_frac"] * 100:.0f}%）' if res['sat_frac'] is not None else ''),
         res['sat_frac'] is not None and res['sat_frac'] < 0.2),
        ('停桨后输出保持 0、看门狗未触发、遥控链路没断', bool(res['post_zero']) and not res['killed'] and not res['link_lost']),
    ]
    res['checks'] = [(c, bool(ok)) for c, ok in checks]
    res['passed'] = all(ok for _, ok in checks)
    if ck:
        res['hover_duty_next'] = round(HOVER_BELIEF * ck, 3)      # 下次编译用的 FC_HOVER_DUTY
    return res


def report(r):
    if not r['found']:
        print('✗ ' + r['msg']); return
    print(f'槽 {r["slot"]}（开机号 {r["boot"]}，复位原因 {r["reset_name"]}）  通电 {r["on_s"]:.1f} s  读出完整 {"✓" if r["complete"] else "✗"}')
    print('   事件：' + ' → '.join(r['events']))
    for c, ok in r['checks']:
        print(f'   {"✓" if ok else "✗"} {c}')
    print(f'\n   【高度（气压计 + 加计估计，相对起飞前）】最高 {r["z_rise_max"]:+.2f} m' +
          (f'，悬停平均 {r["z_rise_hold"]:+.2f} m（目标 +0.80 m）' if r['z_rise_hold'] is not None else '') +
          (f'；起飞到稳定 {r["t_to_hold"]:.1f} s' if r['t_to_hold'] else '') + f'；悬停 {r["hold_s"]:.1f} s')
    if r['roll_std_hold'] is not None:
        print(f'   【姿态】悬停时抖动 roll {r["roll_std_hold"]:.2f}° / pitch {r["pitch_std_hold"]:.2f}°；偏航漂移 {r["yaw_drift"]:+.1f}°')
    if r['ck_hold']:
        print(f'   【推力】悬停真实占空比 {r["duty_hold"]:.3f}；推力修正系数 c_k {r["ck_hold"]:.3f}（1 = 换算正好）'
              f' → 下次换算可改为悬停 {r["hover_duty_next"]:.3f} @ 静止电压')
    print(f'   【其他】门控 k<0.5 占 {r["k_lt05"] * 100:.0f}%；振动中位 {r["vib_med"]:.2f} g；电池 静止 {r["vb_static"]:.2f} → 最低读数 {r["vb_min"]:.2f} V；'
          f'1 kHz 最长周期 {r["per_max_us"]} μs')
    print('\n' + ('✓ 首飞流程全部正常' if r['passed'] else '✗ 有项目没通过，先把结果发给我，别急着飞第二次'))


# ---------------------------------------------------------------- 主流程
CHECKLIST = [
    '电池充满了吗（静止 ≥ 4.10 V，手机网页上看）？',
    '电池焊点 / 插头查过了吗（不松、不虚焊；插拔 USB 会断电的问题处理过或确认不影响电池供电）？',
    '4 个桨方向对吗（俯视：左前 / 右后 顺时针，右前 / 左后 逆时针）、拧紧、没裂纹？',
    '系留好了吗（细线 1.5~2 m，系在机身底部中心，另一端拴 ≥ 1 kg 重物、放在起飞点正下方，线理顺离桨远）？',
    '场地：起飞点半径 3 m 内没有人和东西、地面平整硬、没风？',
    '手机连上热点、网页打开、锁屏时间调长；你站在 2 m 外，手指能随时按到「急停」？',
]


def main():
    if '--analyse-file' in sys.argv:
        src = sys.argv[sys.argv.index('--analyse-file') + 1]
        r = analyse_flight(bb_read.parse(open(src).read().split('\n'))); report(r); return
    print(__doc__.split('\n用法')[0]); print('=' * 64)
    print('⚠ 这一次飞机会真正离地（目标 0.8 m），桨是装着的。')
    for q in CHECKLIST:
        if not V.yes('   ' + q):
            sys.exit('有一项没准备好，退出（什么都没做）。准备好再来。')
    if input('   全部确认，请输入「系留首飞」：').strip() != '系留首飞':
        sys.exit('未确认，退出（什么都没做）。')
    PORT = V.PORT
    if '--no-flash' not in sys.argv:
        import glob
        if not glob.glob(PORT):
            sys.exit(f'✗ 找不到串口 {PORT}，请插上 USB。')
        V.flash('build_ff', TAG, '首飞版（油门上限 70%）')
    ser, st = V.open_and_wait_boot()
    tm = dict(zip(st.get('thrust_map', '').split()[::2], st.get('thrust_map', '').split()[1::2]))
    print(f'   固件 {st.get("fw", "?")}')
    print(f'   电机驱动 {st.get("motor_output")}  油门上限 {st.get("u_max")}  推力换算 {tm.get("scale", "?")}（假设悬停 {tm.get("hover_duty", "?")} @ {tm.get("at_vb", "?")} V，'
          f'在线修正 {"开" if tm.get("adapt") == "1" else "关"}）  电池 {st.get("vbatt")} V  状态 {st.get("state")}')
    bad = (st.get('motor_output') != '1' or abs(float(st.get('u_max') or 0) - U_MAX) > 1e-4 or st.get('state') != 'DISARMED'
           or 'acc_lpf' not in st.get('fw', '') or tm.get('adapt') != '1')
    if bad:
        sys.exit('✗ 板上不是「首飞版、油门上限 0.70、推力在线修正开、未解锁」，停止。')
    scale = float(tm.get('scale', DUTY_SCALE))
    try:
        vb = float(st.get('vbatt', 0))
        if vb < 4.10 and not V.yes(f'   ⚠ 电池 {vb:.2f} V < 4.10 V，推力余量会变小。仍要继续吗？'):
            sys.exit('先充电。')
    except ValueError:
        pass
    ser.close()
    rec, wav = None, None
    if audio_rpm and '--no-audio' not in sys.argv:
        os.makedirs(os.path.join(HERE, 'blackbox'), exist_ok=True)
        wav = os.path.join(HERE, 'blackbox', 'ff_' + time.strftime('%Y%m%d_%H%M%S') + '_audio.wav')
        try:
            rec = audio_rpm.Recorder(wav, max_s=600)
            print(f'   🎙 已开始录音（电脑放在离起飞点 1~2 m、不在飞机可能飞到的地方）')
        except Exception as e:
            print(f'   ⚠ 录音没启动（{e}），继续'); rec, wav = None, None
    print('\n' + '-' * 64)
    print('现在按顺序做：')
    print('  1) **拔掉 USB 线**（电池不要拔！）——然后看手机网页：还连着、状态「未解锁」= 正常；')
    print('     网页断了又重连 / 显示刚开机 = 拔 USB 时断电了 → 不要飞，先查电源，按 Ctrl+C 退出')
    print('  2) 飞机放到系留点、放平，静止 3 秒；人退到 2 m 外')
    print('  3) 手机：「按住解锁」→「按住起飞」')
    print('     升到约 0.8 m → 悬停 5 秒 → 自动降落 → 触地停桨（共约 10 秒）')
    print('     歪得厉害 / 打转 / 冲高超过 1.5 m / 飘向人或东西 → 立刻按「急停」（拴着绳，摔落 < 1.5 m）')
    print('  4) 停稳后 **插回 USB**（电池别拔），按回车')
    input('做完第 4 步按回车 ')
    if rec:
        print(f'   🎙 录音结束（{rec.stop():.0f} s）')
    print('   等板子出数据……')
    ser, st2 = V.open_and_wait_boot(30)
    print(f'   状态 {st2.get("state")}  黑匣子 {st2.get("blackbox")}')
    ser.close(); time.sleep(0.3)
    lines, slots, nref = bb_read.read_serial(PORT)
    od = os.path.join(HERE, 'blackbox'); tag = time.strftime('%Y%m%d_%H%M%S')
    open(os.path.join(od, f'ff_{tag}_raw.txt'), 'w').write('\n'.join(lines) + '\n')
    for si, d in slots.items():
        if d.get('valid') and d['recs']:
            bb_read.write_csv(os.path.join(od, f'ff_{tag}_slot{si}_boot{d["boot"]}.csv'), d)
    r = analyse_flight(slots, scale=scale)
    print('\n================ 首飞结果 ================')
    report(r)
    if wav and os.path.exists(wav) and audio_rpm and r.get('duty_hold'):
        print('\n================ 悬停转速（录音）================')
        try:
            ar = audio_rpm.analyse_external(wav, r['duty_hold'], vb=r.get('vb_static'), verdict=False) or {}
            r['audio'] = {k: v for k, v in ar.items() if k != 'path'}
        except Exception as e:
            print(f'   ⚠ 录音分析失败：{e}')
        print(f'   录音文件：{os.path.relpath(wav, HERE)}')
    r['user_note'] = input('\n   飞得怎么样？一句话描述（比如"很稳"/"往左飘"/"冲得很高"），回车跳过：').strip()
    json.dump(r, open(os.path.join(od, f'ff_{tag}.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    print(f'   已保存 blackbox/ff_{tag}.json、原始读出与 CSV')
    if V.yes('\n烧回干跑版固件（电机驱动不编译，推荐）？'):
        V.flash('build', 'dry-run: phone control', '干跑版')
    print('把上面「首飞结果」整段发给我就行。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已取消。⚠ 板上可能仍是首飞版（电机驱动版）。下次用 --no-flash 继续，或让我烧回干跑版。')
