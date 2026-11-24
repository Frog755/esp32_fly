#!/usr/bin/env python3
"""sweep_umax_v9.py —— Stage 29：首飞油门上限 FLIGHT_U_MAX 扫描（固件核心软件在环，同 sil_flight_v9.py）

油门上限加在与固件 motor_out_write() 相同的位置：控制器照常算指令，写电机时截断（控制器不知道有上限）。
推力 ∝ u²：上限 0.9 只剩 81% 最大推力。满电 4.15 V 悬停 u≈0.76（Stage 24~27 实测 0.77），半电 ~3.8 V 约 0.83。
用法：python3 sweep_umax_v9.py [--quick]  → RL/results/stage29/sweep_umax.json
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sil_flight_v9 as S

ROOT = S.ROOT
CAPS = [0.80, 0.85, 0.90, 0.95, 1.00]
# (场景, 噪声, 开局电量, 架数)
RUNS = [('takeoff', 'nominal', 1.0, 256), ('takeoff', 'pessimistic', 1.0, 256), ('takeoff', 'nominal', 0.5, 256),
        ('drop', 'nominal', 1.0, 256), ('throw', 'nominal', 1.0, 256)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'RL', 'results', 'stage29', 'sweep_umax.json'))
    a = ap.parse_args()
    L = S.build()
    out = []
    for cap in CAPS:
        for i, (sc, nz, soc, n) in enumerate(RUNS):
            n = 64 if a.quick else n
            t0 = time.time()
            r = S.run(L, sc, nz, n, seed=300 + i, u_max=cap, soc=soc)       # 同一场景各上限用同一种子 → 同一批初始条件
            r['wall_s'] = round(time.time() - t0, 1)
            out.append(r)
            print(f"上限 {cap:.2f}  {sc:<8}{nz:<12}电量 {soc:.1f}  N={n:<4} 成功 {r['ok_rate'] * 100:5.1f}%  进入悬停 {r['hold_reached'] * 100:5.1f}%  "
                  f"坠地 {r['crash_rate'] * 100:4.1f}%  空中停桨 {r['air_kill_rate'] * 100:3.1f}%  "
                  f"最高 中位 {r['zmax_med']:.2f} / P90 {r['zmax_p90']:.2f} / 最大 {r['zmax_max']:.2f} m  "
                  f"顶上限占比 中位 {r['sat_frac_med'] * 100:4.1f}% / P90 {r['sat_frac_p90'] * 100:4.1f}%  "
                  f"悬停峰值指令 {r['u_hold_peak_med']:.2f}  电压最低 {r['vb_min_med']:.2f} V  停桨 {r['off_events']}", flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print('→', a.out)


if __name__ == '__main__':
    main()
