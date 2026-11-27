#!/usr/bin/env python3
"""sweep_stage34.py —— Stage 34：按第 1 次系留首飞（Stage 33）修改后的固件 SIL 验证

对照：
  旧 = 首飞版（FC_HOVER_DUTY 0.42 @ 4.17 V、油门上限 0.70、没有贴地上调、没有自转保护）
  新 = Stage 34（FC_HOVER_DUTY 0.50 @ 4.14 V、油门上限 0.85、贴地推力不够只上调 c_k、自转 > 半圈/秒 0.5 s → 降落）
被控对象（"真机"）：
  供电没修：线路电阻 0.25 Ω（Stage 33 实测占空比 0.53 时掉 1 V），悬停占空比 0.50~0.62
  供电修好：内阻默认，悬停占空比 0.36~0.48
  偏航扰动 0 / −3.5 rad/s²（Stage 33 拟合，满转时）
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sil_flight_v9 as S

BUILDS = [('旧（首飞版）', {'FC_HOVER_DUTY': '0.42f', 'FC_HOVER_VB': '4.17f', 'FC_GROUND_ADAPT': '0', 'FSM_SPIN_W': '1e9f'}, 0.70),
          ('新（Stage 34）', {}, 0.85)]
PLANTS = [('供电没修', dict(plant_rint=0.25), [0.50, 0.54, 0.58, 0.62]),
          ('供电修好', dict(), [0.36, 0.42, 0.48])]
YAWS = [0.0, 3.5]
SCENS = [('takeoff', 'nominal'), ('takeoff', 'pessimistic'), ('drop', 'nominal'), ('throw', 'nominal')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=os.path.join(S.ROOT, 'RL', 'results', 'stage34', 'sweep_stage34.json'))
    ap.add_argument('--only', default='', help='只跑名字含这个字符串的固件')
    a = ap.parse_args()
    N = 32 if a.quick else 96
    out = []
    for bname, defs, umax in BUILDS:
        if a.only and a.only not in bname:
            continue
        L = S.build(defs)
        print(f'===== {bname}：假设悬停 {L.sil_hover_duty():.2f} @ {L.sil_hover_vb():.2f} V，换算系数 {L.sil_duty_scale():.4f}，油门上限 {umax:.2f} =====', flush=True)
        for pname, kw, hovers in PLANTS:
            for h in hovers:
                for yd in YAWS:
                    cells = []
                    for i, (sc, nz) in enumerate(SCENS):
                        t0 = time.time()
                        r = S.run(L, sc, nz, N, seed=3400 + i, u_max=umax, plant_hover=h, yaw_dist=-yd, T=25.0, **kw)
                        r.update(build=bname, plant=pname, wall_s=round(time.time() - t0, 1)); out.append(r)
                        ev = r['off_events']; spin = ev.get('land_spin', 0); tko = ev.get('land_takeoff_timeout', 0)
                        cells.append(f"{sc[:4]}/{nz[:3]} {r['ok_rate'] * 100:3.0f}%（撞 {r['crash_rate'] * 100:2.0f} 转 {spin:2d} 超时 {tko:2d} 饱和 {r['sat_frac_med'] * 100:3.0f}% 自转 {r['wz_max_p90_dps']:4.0f}°/s c_k {r['ck_med']:.2f}）")
                    print(f'  {pname} 悬停 {h:.2f} 偏航扰动 {yd:.1f}： ' + ' | '.join(cells), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print('→', a.out)


if __name__ == '__main__':
    main()
