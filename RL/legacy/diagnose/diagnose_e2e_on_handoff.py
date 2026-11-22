#!/usr/bin/env python3
"""
diagnose_e2e_on_handoff.py —— 决定性对照：端到端模型能否处理 P1 的交接状态
================================================================

为什么这是决定性的：

  已知（实测）：
    · P1 v18 交接状态 |ω_tilt| 中位 42.4，P90 51.5
    · 理想 PD（甚至最弱的 kp=0.010）在这些状态上达到 54~71% 的 P2 判据达成率
    · P2 RL 模型在同状态上 0/24，平均油门 0.238（悬停需 0.707）

  这说明：判据不是难点，执行器不是瓶颈，P2 的 0% 是训练/分布问题。

  那么最后一个问题：如果不用 P2，直接用端到端模型 E-A2 接手这些状态，
  它能做成吗？
    能  → 说明「高转速下稳定姿态」这个能力端到端已经学会了，
          分阶段唯一缺的就是「把这个能力装进 P2」，目标明确。
    不能 → 说明这个状态空间本身对任何 RL 都难，
          那分阶段也不该背这个锅，问题在任务定义。

  这直接决定要不要继续修 P2，还是直接收尾走端到端。

用法：
  python3 scripts/diagnose_e2e_on_handoff.py
"""

# --- [Stage 20 legacy 迁移引导] 本文件原位于 RL/scripts/，现移至 RL/legacy/diagnose/ ---
# _LEGACY_ORIG_DIR 指回原目录：原有基于 __file__ 的相对路径（../env、../models 等）语义不变；
# 并把主线 env/、scripts/ 与各 legacy 子目录加入 sys.path，跨目录的旧导入照常可用。逻辑未改动。
import os as _lg_os, sys as _lg_sys
_LEGACY_RL = _lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.abspath(__file__))))
_LEGACY_ORIG_DIR = _lg_os.path.join(_LEGACY_RL, 'scripts')
for _lg_p in [_LEGACY_RL, _lg_os.path.join(_LEGACY_RL, 'env'), _lg_os.path.join(_LEGACY_RL, 'scripts')] + sorted(
        _lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d) for _lg_d in _lg_os.listdir(_lg_os.path.join(_LEGACY_RL, 'legacy'))
        if _lg_os.path.isdir(_lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d)) and not _lg_d.startswith(('.', '_'))):
    if _lg_p not in _lg_sys.path:
        _lg_sys.path.insert(1, _lg_p)
# --- [/Stage 20 legacy 迁移引导] ---
import argparse
import json
import math
import os
import sys

import numpy as np

_RL_DIR = os.path.dirname(_LEGACY_ORIG_DIR)
sys.path.insert(0, os.path.join(_RL_DIR, "env"))
sys.path.insert(0, _RL_DIR)

import quad_sim_fast_v2 as qs  # noqa: E402

G = 9.81


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


def collect_handoffs(m1, episodes, diffs, seed):
    out = []
    for i in range(episodes):
        d = diffs[i % len(diffs)]
        env = qs.ThrowRecoveryEnv(difficulty=d, domain_rand=1.0,
                                  episode_s=4.0, seed=seed + i, phase=1)
        obs, _ = env.reset()
        for k in range(env.episode_steps):
            a, _ = m1.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            if info["up_z"] > 0.7:
                out.append(dict(d=d, seed=seed + i,
                                state=dict(pos=list(env.sim.pos),
                                           vel=list(env.sim.vel),
                                           quat=list(env.sim.q),
                                           omega=list(env.sim.w), soc=1.0)))
                break
            if term or trunc:
                break
    return out


def rollout(hs, policy_fn, phase, history_len=0):
    """从给定交接状态起跑，返回 (最长保持, 存活步数, 末态up_z, 平均油门, 结局)"""
    env = qs.ThrowRecoveryEnv(difficulty=hs["d"], domain_rand=0.0,
                              episode_s=4.0, seed=hs["seed"], phase=phase,
                              history_len=history_len)
    env.reset()
    env.sim.reset(**hs["state"])
    obs = env._raw_obs().astype(np.float32)
    if history_len > 0:
        env._hist = [env._hist_feature()] * history_len
        obs = env._stack_obs(env._raw_obs()).astype(np.float32)
    hold = best = 0
    thr = []
    n = 0
    last = None
    for _ in range(env.episode_steps):
        a = policy_fn(obs, env)
        thr.append(float(np.mean((np.asarray(a) + 1) / 2)))
        obs, _r, t, tr, info = env.step(a)
        n += 1
        last = info
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
    return dict(best=best, steps=n,
                upz=float(up_z_of(env.sim)),
                thr=float(np.mean(thr)) if thr else 0.0,
                z=float(env.sim.pos[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default="models/p1_v18/model.zip")
    ap.add_argument("--p2", default="models/p2_v18/model.zip")
    ap.add_argument("--e2e", default="models/v6b_EA2_rot_small/model.zip")
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--diffs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    m1 = PPO.load(args.p1, device="cpu")
    m2 = PPO.load(args.p2, device="cpu")
    me = PPO.load(args.e2e, device="cpu")

    hs = collect_handoffs(m1, args.episodes, args.diffs, args.seed)[:args.samples]
    print("=" * 78)
    print(f"  决定性对照：从 P1 真实交接状态起跑，谁能稳住？ (n={len(hs)})")
    w0 = [math.sqrt(sum(v ** 2 for v in h["state"]["omega"])) for h in hs]
    print(f"  交接 |ω| 中位 {np.median(w0):.1f}  P90 {np.percentile(w0,90):.1f}  "
          f"max {max(w0):.1f}")
    print(f"  悬停油门 {qs.QuadParams().hover_throttle:.3f}")
    print("=" * 78)

    agents = {
        "P2 v18 (phase=2)": (lambda o, e: m2.predict(o, deterministic=True)[0], 2, 0),
        "E-A2 端到端":       (lambda o, e: me.predict(o, deterministic=True)[0], 0, 0),
        "理想 PD (kp=.03)":  (lambda o, e: pd_action(e), 2, 0),
    }

    print()
    print(f"  {'代理':<20}{'达成50步':>10}{'保持中位':>10}{'存活中位':>10}"
          f"{'末态up_z':>10}{'油门':>8}")
    print("  " + "-" * 68)
    summary = {}
    for name, (fn, phase, hl) in agents.items():
        rs = [rollout(h, fn, phase, hl) for h in hs]
        ok = sum(1 for r in rs if r["best"] >= 50)
        summary[name] = dict(
            ok=ok, n=len(rs), rate=ok / len(rs),
            best_med=float(np.median([r["best"] for r in rs])),
            steps_med=float(np.median([r["steps"] for r in rs])),
            upz_med=float(np.median([r["upz"] for r in rs])),
            thr_med=float(np.median([r["thr"] for r in rs])),
        )
        s = summary[name]
        print(f"  {name:<20}{ok:>6}/{len(rs):<3}{s['best_med']:>10.0f}"
              f"{s['steps_med']:>10.0f}{s['upz_med']:>10.2f}{s['thr_med']:>8.3f}")

    # 分 ω 档看谁在什么转速下崩
    print()
    print("  【按交接 ω 分档的达成率】")
    print(f"  {'ω 区间':<12}{'样本':>6}" + "".join(
        f"{n.split()[0]:>12}" for n in agents))
    print("  " + "-" * 60)
    buckets = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 61)]
    for lo, hi in buckets:
        idx = [i for i, w in enumerate(w0) if lo <= w < hi]
        if not idx:
            continue
        line = f"  [{lo:>2},{hi:>2})     {len(idx):>6}"
        for name, (fn, phase, hl) in agents.items():
            rs = [rollout(hs[i], fn, phase, hl) for i in idx]
            ok = sum(1 for r in rs if r["best"] >= 50)
            line += f"{ok}/{len(rs):<10}"
        print(line)

    out = os.path.join(_RL_DIR, "results", "e2e_on_handoff.json")
    with open(out, "w") as f:
        json.dump(dict(summary=summary,
                       w0=[float(x) for x in w0]), f, indent=1)
    print(f"\n  原始数据: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
