#!/usr/bin/env python3
"""
diagnose_omega_curve.py —— P2 的能力边界曲线（ω 扫描）
================================================================

要回答的唯一问题：
  P2 这个策略，在【交接转速 ω_tilt】多大时开始崩？
  把它和理想 PD 画在同一张表上，就知道差距是"策略不行"还是"物理不行"。

方法（受控实验，排除分布混杂）：
  固定其它一切变量，只扫 ω_tilt：
      up_z   = 0.85（倾角 31.8°，模拟刚翻过 0.7 的位置）
      z      = 2.5 m
      vz     = -1.0 m/s
      vxy    = 0
      ω_tilt = 目标值，绕机身 xy 平面内一个随机方向
  每个 ω 档跑 N 个随机方向 × 随机种子，统计：
      · 达成 P2 判据（up_z>0.98 & |vz|<0.5 & ω_tilt<1，连续 50 步）的比例
      · 最长连续保持步数中位
      · 平均油门（诊断推力病理）

  这样得到一条「P2 成功率 vs 交接 ω」的曲线，
  再叠加 PD 同曲线 → 直接看出 RL 与物理上限的差距。

用法：
  python3 scripts/diagnose_omega_curve.py --trials 24
"""

import argparse
import json
import math
import os
import sys

import numpy as np

_RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RL_DIR, "env"))
sys.path.insert(0, _RL_DIR)

import quad_sim_fast_v2 as qs  # noqa: E402

G = 9.81
HOVER = qs.QuadParams().hover_throttle


def pd_action(env, kp=0.030, kd=0.0030, kz=4.0, vz_ref=0.0):
    sim, p = env.sim, env.p
    qw, qx, qy, qz = sim.q
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)
    r10 = 2 * (qx * qy + qw * qz)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qw * qx)
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    zx, zy, zz = r02, r12, r22
    cx, cy, cz = zy, -zx, 0.0
    ex = r00 * cx + r10 * cy + r20 * cz
    ey = r01 * cx + r11 * cy + r21 * cz
    ez = r02 * cx + r12 * cy + r22 * cz
    wx, wy, wz = sim.w
    tx = kp * ex - kd * wx
    ty = kp * ey - kd * wy
    tz = -0.0008 * wz
    az_ref = kz * (vz_ref - sim.vel[2])
    Fz = p.mass * (G + az_ref) / max(zz, 0.5) * max(zz, 0.3)
    Fz = float(max(Fz, 0.25 * p.mass * G))
    rp = p.rotor_pos
    M = np.vstack([np.ones(4), rp[:, 1], -rp[:, 0], -p.rotor_dir * p.k_m_ratio])
    T = np.clip(np.linalg.solve(M, np.r_[Fz, tx, ty, tz]),
                0, p.thrust_max_per_motor)
    u = np.sqrt(T / p.thrust_max_per_motor) * (p.v_ref / max(sim.v_batt, 3.0))
    return np.clip(u * 2.0 - 1.0, -1, 1)


def up_z_of(sim):
    return 1 - 2 * (sim.q[1] ** 2 + sim.q[2] ** 2)


def om_tilt(sim):
    return math.sqrt(sim.w[0] ** 2 + sim.w[1] ** 2)


def make_state(up_z_target, om_target, ang_dir, z=2.5, vz=-1.0, vxy=0.0):
    """构造一个指定倾角与倾角角速度的初始状态。"""
    tilt = math.acos(max(-1.0, min(1.0, up_z_target)))
    ax = math.cos(ang_dir)
    ay = math.sin(ang_dir)
    q = np.r_[math.cos(tilt / 2), math.sin(tilt / 2) * ax,
              math.sin(tilt / 2) * ay, 0.0]
    # 角速度方向垂直于机身 z 轴（纯倾角旋转，不含自旋）
    omega = [om_target * ay, -om_target * ax, 0.0]
    return dict(pos=[0.0, 0.0, z], vel=[0.0, 0.0, vz], quat=list(q),
                omega=omega, soc=1.0)


def rollout(model, state, phase, use_pd=False, seed=0):
    env = qs.ThrowRecoveryEnv(difficulty=0.7, domain_rand=0.0,
                              episode_s=4.0, seed=seed, phase=phase)
    env.reset()
    env.sim.reset(**state)
    obs = env._raw_obs().astype(np.float32)
    hold = best = 0
    thr = []
    for _ in range(env.episode_steps):
        if use_pd:
            a = pd_action(env)
        else:
            a, _ = model.predict(obs, deterministic=True)
        thr.append(float(np.mean((np.asarray(a) + 1) / 2)))
        obs, _r, t, tr, _i = env.step(a)
        c1 = up_z_of(env.sim) > 0.98
        c2 = abs(env.sim.vel[2]) < 0.5
        c3 = om_tilt(env.sim) < 1.0
        if c1 and c2 and c3:
            hold += 1
            best = max(best, hold)
        else:
            hold = 0
        if t or tr:
            break
    return dict(best=best, thr=float(np.mean(thr)) if thr else 0.0,
                upz=float(up_z_of(env.sim)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2", default="models/p2_v18/model.zip")
    ap.add_argument("--p2-old", default="models/p2_v15/model.zip",
                    help="额外对照：上一代 P2（ω 上限 26 训练）")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--om", type=float, nargs="+",
                    default=[5, 10, 15, 20, 26, 32, 38, 44, 50, 56])
    ap.add_argument("--upz", type=float, default=0.85)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    m2 = PPO.load(args.p2, device="cpu")
    try:
        m_old = PPO.load(args.p2_old, device="cpu")
    except Exception:  # noqa: BLE001
        m_old = None

    print("=" * 86)
    print(f"  P2 能力边界曲线  |  固定 up_z={args.upz}  z=2.5m  vz=-1.0  vxy=0")
    print(f"  只扫「倾角角速度 ω_tilt」  |  悬停油门 {HOVER:.3f}")
    print("=" * 86)
    hdr = f"  {'ω_tilt':>7}{'P2 v18':>10}{'保持中位':>10}{'油门':>8}"
    if m_old is not None:
        hdr += f"{'P2 v15':>10}"
    hdr += f"{'理想PD':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 4))

    rows = []
    for om in args.om:
        n = args.trials
        r_new = [rollout(m2, make_state(args.upz, om, 2 * math.pi * i / n),
                         2, seed=1000 + i) for i in range(n)]
        ok_new = sum(1 for r in r_new if r["best"] >= 50)
        r_pd = [rollout(None, make_state(args.upz, om, 2 * math.pi * i / n),
                        2, use_pd=True, seed=1000 + i) for i in range(n)]
        ok_pd = sum(1 for r in r_pd if r["best"] >= 50)
        line = (f"  {om:>7.0f}{ok_new:>5}/{n:<4}{np.median([r['best'] for r in r_new]):>10.0f}"
                f"{np.median([r['thr'] for r in r_new]):>8.3f}")
        row = dict(om=om, new_ok=ok_new, new_rate=ok_new / n,
                   new_best_med=float(np.median([r["best"] for r in r_new])),
                   new_thr=float(np.median([r["thr"] for r in r_new])),
                   pd_ok=ok_pd, pd_rate=ok_pd / n)
        if m_old is not None:
            r_old = [rollout(m_old, make_state(args.upz, om, 2 * math.pi * i / n),
                             2, seed=1000 + i) for i in range(n)]
            ok_old = sum(1 for r in r_old if r["best"] >= 50)
            line += f"{ok_old:>5}/{n:<4}"
            row["old_ok"] = ok_old
        line += f"{ok_pd:>5}/{n:<4}"
        print(line)
        rows.append(row)

    print()
    print("  说明：P2 v18 训练时 ω 采样上限 26（v19 才改 60），v15 上限 26。")
    print("        ω_tilt=50 的交接状态对它们都是【训练分布外】。")
    print()
    print("  【解读】")
    edge = [r["om"] for r in rows if r["new_rate"] >= 0.8]
    if edge:
        print(f"    P2 v18 在 ω_tilt ≤ {max(edge):.0f} 时达成率 ≥80%，"
              f"超过后开始崩 → 这是它的能力边界")
    pd_edge = [r["om"] for r in rows if r["pd_rate"] >= 0.8]
    if pd_edge:
        print(f"    理想 PD 边界 ω_tilt ≤ {max(pd_edge):.0f} → 物理上限")
        if edge and max(pd_edge) > max(edge):
            gap = max(pd_edge) - max(edge)
            print(f"    RL 与物理上限差 {gap:.0f} rad/s → "
                  f"这部分是【策略/训练问题】，不是物理问题")
    else:
        print("    理想 PD 在扫描范围内未达 80% → 判据本身偏严")

    out = os.path.join(_RL_DIR, "results", "omega_curve.json")
    with open(out, "w") as f:
        json.dump(dict(rows=rows, hover=HOVER, upz=args.upz), f, indent=1)
    print(f"\n  原始数据: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
