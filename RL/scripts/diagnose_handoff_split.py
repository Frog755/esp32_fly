#!/usr/bin/env python3
"""
diagnose_handoff_split.py —— 交接瞬间角速度的「分量分解」
================================================================

动机（mj 的观察，2026-11-04）：
  「其实我最终只需要他能稳定姿态朝上…yaw 角速度怎么转都无所谓，
    但是如果有 z 轴角速度肯定不行」

  这句话需要区分两个「z 轴」：
    - 世界系 z 轴角速度 → 就是偏航（yaw），只转朝向，不改变倾角
    - 机体系 z 轴角速度 wz   → 是「绕机身自身对称轴自旋」，
                                 严格说也不改变 up_z（倾角）

  本脚本要回答的问题是：
  交接瞬间的 |ω| ≈ 42 rad/s（此前实测），其中
    有多少是【真正在改变倾角】的 wx,wy 分量？
    有多少是【不改变倾角】的 wz 分量？

  如果 |ω_tilt| = sqrt(wx²+wy²) 远小于 42，说明 P2 的 ω<1 判据
  把大量「无害自旋」也算进了难度，放宽 wz 就是合理降难。
  如果 |ω_tilt| 本身就 ≈42，那 P2 面对的是真实的倾角旋转，
  放宽 wz 救不了，只能靠 P1 刹车或改判据。

  另一件事：自旋在【有气动阻尼】时会慢慢衰减，但本仿真
  drag_ang z 分量只有 4e-6，几乎无阻尼，所以 wz 会长期保持。
  这也要量出来 —— 如果 P2 需要 50 步稳定，wz 从头到尾不衰减，
  那「ω<1 保持 50 步」在物理上就是等 wz 自己消失，几乎不可能。

用法：
  python3 scripts/diagnose_handoff_split.py --model models/p1_v18/model.zip --episodes 300
"""

import argparse
import json
import math
import os
import sys

import numpy as np

_RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RL_DIR, "env"))
sys.path.insert(0, _RL_DIR)   # env/quad_sim_fixed 内部 `from env.xxx import` 需要 RL 在路径上

import quad_sim_fast_v2 as qs  # noqa: E402

HANDOFF_UPZ = 0.7


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def run(args):
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")

    rows = []
    decay_rows = []
    for i in range(args.episodes):
        d = args.diffs[i % len(args.diffs)]
        seed = args.seed + i
        env = qs.ThrowRecoveryEnv(difficulty=d, domain_rand=1.0,
                                  episode_s=4.0, seed=seed, phase=1)
        obs, _ = env.reset()
        init_w = tuple(env.sim.w)
        flipped = False
        for k in range(env.episode_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if not flipped and info["up_z"] > HANDOFF_UPZ:
                flipped = True
                w = list(env.sim.w)
                w_tilt = math.sqrt(w[0] ** 2 + w[1] ** 2)
                w_yaw = abs(w[2])
                w_all = math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
                rows.append(dict(
                    seed=seed, d=d, step=k + 1,
                    omega=w_all, omega_tilt=w_tilt, omega_yaw=w_yaw,
                    upz=float(info["up_z"]),
                    vxy=float(math.hypot(env.sim.vel[0], env.sim.vel[1])),
                    wz=float(w[2]),
                    # 自旋占比：wz 在总模长中的能量占比
                    yaw_frac=(w_yaw ** 2 / (w_all ** 2 + 1e-12)),
                ))
                # 交接后继续跑，观察 wz 自然衰减速度（不给控制，纯被动）
                break
            if term or trunc:
                break

        # 被动衰减实验：把交接状态复制一份，给零动作（重力自由落体不够时间，
        # 所以用 reset 到该状态 + 悬停油门近似），只观察 ω 自身演化
        if flipped and len(decay_rows) < args.decay_samples:
            st = dict(pos=list(env.sim.pos), vel=list(env.sim.vel),
                      quat=list(env.sim.q), omega=list(env.sim.w),
                      soc=1.0)
            env2 = qs.ThrowRecoveryEnv(difficulty=d, domain_rand=0.0,
                                       episode_s=4.0, seed=seed, phase=2)
            env2.reset()
            env2.sim.reset(**st)
            # 零力矩动作：四电机同油门 → 只产生推力，不产生力矩
            wz_trace = []
            for k2 in range(150):
                a0 = np.zeros(4)
                # 用环境 step 但动作全 0 → u=0.5 → 各电机同推力，净力矩≈0
                obs2, r2, t2, tr2, info2 = env2.step(a0)
                wz_trace.append(abs(info2["omega_norm"]))
                if t2 or tr2:
                    break
            if len(wz_trace) > 60:
                decay_rows.append(dict(
                    d=d, seed=seed,
                    w0=wz_trace[0], w30=wz_trace[30],
                    w60=wz_trace[60],
                    ratio_60=wz_trace[60] / max(wz_trace[0], 1e-9),
                    steps_to_half=next((j for j, v in enumerate(wz_trace)
                                        if v < wz_trace[0] * 0.5), None),
                ))

    # ---- 汇总 ----
    print("=" * 78)
    print(f"  交接瞬间角速度分量分解  |  {os.path.basename(os.path.dirname(args.model))}")
    print("=" * 78)
    print(f"  样本数: {len(rows)} / {args.episodes} 回合（未翻正的不计入）")
    if not rows:
        print("  ❌ 没有交接样本")
        return 1

    om = [r["omega"] for r in rows]
    ot = [r["omega_tilt"] for r in rows]
    oy = [r["omega_yaw"] for r in rows]
    yf = [r["yaw_frac"] for r in rows]

    print()
    print(f"  {'量':<18}{'中位':>9}{'均值':>9}{'P90':>9}{'max':>9}")
    print("  " + "-" * 55)
    for name, arr in (("|ω| 总模长", om), ("|ω_tilt| √(wx²+wy²)", ot),
                      ("|ω_yaw| |wz|", oy)):
        print(f"  {name:<18}{pct(arr,50):>9.2f}{np.mean(arr):>9.2f}"
              f"{pct(arr,90):>9.2f}{max(arr):>9.2f}")
    print(f"  {'yaw 能量占比':<18}{pct(yf,50)*100:>8.1f}%"
          f"{np.mean(yf)*100:>8.1f}%{pct(yf,90)*100:>8.1f}%"
          f"{max(yf)*100:>8.1f}%")

    print()
    print("  【判据可行性】若放宽为 ω_tilt<1（忽略自旋 wz）：")
    for th in (1.0, 2.0, 4.0, 8.0, 15.0, 28.0):
        n = sum(1 for v in ot if v < th)
        print(f"    ω_tilt < {th:>5.1f}  已达标 {n:>4}/{len(ot)} = {n/len(ot)*100:5.1f}%")
    print()
    print("  【对照】若判据仍是 |ω|<1（含自旋）：")
    n = sum(1 for v in om if v < 1.0)
    print(f"    |ω| < 1.0     已达标 {n:>4}/{len(om)} = {n/len(om)*100:5.1f}%")

    if decay_rows:
        print()
        print("  【被动衰减】零力矩（四电机同油门）下总 ω 的演变：")
        print(f"    {'d':>5}{'w0':>9}{'w@30':>9}{'w@60':>9}{'60步后剩余':>12}{'半衰步数':>10}")
        for r in decay_rows:
            print(f"    {r['d']:>5.1f}{r['w0']:>9.2f}{r['w30']:>9.2f}"
                  f"{r['w60']:>9.2f}{r['ratio_60']*100:>11.1f}%"
                  f"{str(r['steps_to_half']):>10}")

    out = os.path.join(_RL_DIR, "results", "handoff_split.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(rows=rows, decay=decay_rows,
                       summary=dict(
                           omega=pct(om, 50), omega_tilt=pct(ot, 50),
                           omega_yaw=pct(oy, 50), yaw_frac=pct(yf, 50))), f,
                  indent=1)
    print(f"\n  原始数据: {out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/p1_v18/model.zip")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--diffs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--decay-samples", type=int, default=12)
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
