#!/usr/bin/env python3
"""sweep_hover_v9.py —— Stage 31：固件推力换算（假设的悬停占空比）对真机推力曲线不确定性的鲁棒性

固件：真实占空比 = FC_DUTY_SCALE × 控制器指令，FC_DUTY_SCALE 由 FC_HOVER_DUTY（读数 3.4 V 时的悬停占空比）决定。
真机（SIL 被控对象）在实测工作条件下的悬停占空比 h ∈ 0.34~0.50（实测估计 0.41~0.43），三种电池 / 曲线情形：
  offset：电压读数一转就掉 0.8 V 是测量偏差（真实电压 ~4.1 V）；  offset+knee：同上 + 转速在高占空比变平（knee 0.55）；
  sag：掉 0.8 V 是真实压降（内阻 0.31 Ω）。
用法：python3 sweep_hover_v9.py [--quick]  → RL/results/stage31/sweep_hover.json
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sil_flight_v9 as S

# (标签, 编译宏)：v2 = 静止电压补偿 + 空中推力在线修正（Stage 31），对照 = 同样换算但关掉在线修正
BUILDS = [('修正开', {'FC_HOVER_DUTY': '0.42f'}), ('修正关', {'FC_HOVER_DUTY': '0.42f', 'FC_THRUST_ADAPT': '0'})]
HOVERS = [0.32, 0.36, 0.40, 0.44, 0.48, 0.52]
MODES = [('offset', dict(vb_mode='offset')), ('offset+knee', dict(vb_mode='offset', plant_knee=0.55)), ('sag', dict(plant_rint=0.31))]
SCENS = [('takeoff', 'nominal'), ('takeoff', 'pessimistic'), ('drop', 'nominal')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=os.path.join(S.ROOT, 'RL', 'results', 'stage31', 'sweep_hover_v2.json'))
    ap.add_argument('--umax', type=float, default=0.60, help='真实占空比上限（v2 结果用的是 0.60）')
    ap.add_argument('--hovers', default='', help='逗号分隔，覆盖默认真机悬停列表')
    ap.add_argument('--only-adapt', action='store_true', help='只跑"修正开"')
    a = ap.parse_args()
    global HOVERS, BUILDS
    if a.hovers:
        HOVERS = [float(x) for x in a.hovers.split(',')]
    if a.only_adapt:
        BUILDS = BUILDS[:1]
    N = 32 if a.quick else 128
    out = []
    for b, defs in BUILDS:
        L = S.build(defs)
        print(f'===== {b}：固件假设悬停 {L.sil_hover_duty():.2f} @ {L.sil_hover_vb():.2f} V（静止），换算系数 {L.sil_duty_scale():.4f} =====', flush=True)
        for mname, kw in MODES:
            for h in HOVERS:
                for i, (sc, nz) in enumerate(SCENS):
                    t0 = time.time()
                    r = S.run(L, sc, nz, N, seed=500 + i, u_max=a.umax, plant_hover=h, **kw)
                    r.update(belief=b, mode=mname, wall_s=round(time.time() - t0, 1)); out.append(r)
                    print(f"{b} {mname:<12} 真机悬停 {h:.2f}  {sc:<8}{nz:<12} 成功 {r['ok_rate']*100:5.1f}%  坠地 {r['crash_rate']*100:4.1f}%  "
                          f"没起来 {(1 - r['hold_reached'])*100:5.1f}%  最高 中位 {r['zmax_med']:.2f}/最大 {r['zmax_max']:.2f} m  "
                          f"顶上限 P90 {r['sat_frac_p90']*100:4.0f}%  悬停占空比 {r['duty_hold_med']:.3f}  c_k {r['ck_med']:.3f}  停桨 {r['off_events']}", flush=True)
                    os.makedirs(os.path.dirname(a.out), exist_ok=True)
                    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print('→', a.out)


if __name__ == '__main__':
    main()
