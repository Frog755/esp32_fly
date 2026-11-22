#!/usr/bin/env python3
"""
diagnose_pd_sweep.py —— P2 判据的「难度曲线」：扫描控制器强度
================================================================

为什么要做这个（2026-11-04）：

  前面我用 PD 验证 P2 判据可达性，得出过两个互相矛盾的结论：
    - 旧结论：用 env 内置 pd_controller（kp=0.020, kd=0.0020）
              → 从 ω>28 的交接状态几乎全失败 → 我宣称「ω>28 物理不可达」
    - 新结论：用更强的 ideal_pd（kp=0.030, kd=0.0030, 带高度 PD）
              → 70% 达成 → 判据其实可达

  两者矛盾说明：判据的「可达性」不是一个布尔值，而是**控制器强度**的函数。
  所以"降难度"这个决策必须建立在难度曲线上，而不是单点测量。

本脚本扫描 kp/kd 组合，回答：
  Q1 判据在 kp 多小的时候开始崩？→ 临界点在哪
  Q2 理想 PD 刹 ω=42 需要多少力矩（相对物理上限）→ 执行器余量
  Q3 RL 实际用的力矩 vs 理想 PD → 差多少（定位 P2 为什么学不出来）

用法：
  python3 scripts/diagnose_pd_sweep.py --model models/p1_v18/model.zip --samples 30
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


def pd_action(env, kp, kd, kz, tz_gain=0.0008, vz_ref=0.0):
    """参数化几何姿态 PD。额外返回力矩使用率，供诊断。"""
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
    tz = -tz_gain * wz

    vz = sim.vel[2]
    az_ref = kz * (vz_ref - vz)
    tilt_ok = max(zz, 0.0)
    Fz = p.mass * (G + az_ref) / max(zz, 0.5) * max(tilt_ok, 0.3)
    Fz = float(max(Fz, 0.25 * p.mass * G))

    rp = p.rotor_pos
    M = np.vstack([np.ones(4), rp[:, 1], -rp[:, 0], -p.rotor_dir * p.k_m_ratio])
    T = np.linalg.solve(M, np.r_[Fz, tx, ty, tz])
    T_raw = T.copy()
    T = np.clip(T, 0, p.thrust_max_per_motor)
    # 力矩使用率：实际施加 vs 需求（>1 表示饱和，控制余量耗尽）
    sat = float(np.max(np.abs(T_raw - T)) / p.thrust_max_per_motor)
    u = np.sqrt(T / p.thrust_max_per_motor) * (p.v_ref / max(sim.v_batt, 3.0))
    return np.clip(u * 2.0 - 1.0, -1, 1), sat


def up_z_of(sim):
    return 1 - 2 * (sim.q[1] ** 2 + sim.q[2] ** 2)


def om_tilt(sim):
    return math.sqrt(sim.w[0] ** 2 + sim.w[1] ** 2)


def collect_p1_handoffs(model, episodes, diffs, seed):
    """收集 P1 真实交接状态（up_z 首过 0.7）。"""
    states = []
    for i in range(episodes):
        d = diffs[i % len(diffs)]
        env = qs.ThrowRecoveryEnv(difficulty=d, domain_rand=1.0,
                                  episode_s=4.0, seed=seed + i, phase=1)
        obs, _ = env.reset()
        for k in range(env.episode_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            if info["up_z"] > 0.7:
                states.append(dict(
                    d=d, seed=seed + i,
                    state=dict(pos=list(env.sim.pos), vel=list(env.sim.vel),
                               quat=list(env.sim.q), omega=list(env.sim.w),
                               soc=1.0)))
                break
            if term or trunc:
                break
    return states


def run_pd(hs, kp, kd, kz, hold_need=50):
    env = qs.ThrowRecoveryEnv(difficulty=hs["d"], domain_rand=0.0,
                              episode_s=4.0, seed=hs["seed"], phase=2)
    env.reset()
    env.sim.reset(**hs["state"])
    hold = best = 0
    sat_sum = sat_n = 0.0
    for _ in range(env.episode_steps):
        act, sat = pd_action(env, kp, kd, kz)
        sat_sum += sat
        sat_n += 1
        _o, _r, t, tr, _i = env.step(act)
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
    return best, (sat_sum / max(sat_n, 1))


def run_rl(hs, model):
    """用 P2 模型跑同一交接状态，顺带记录等效油门均值。"""
    env = qs.ThrowRecoveryEnv(difficulty=hs["d"], domain_rand=0.0,
                              episode_s=4.0, seed=hs["seed"], phase=2)
    env.reset()
    env.sim.reset(**hs["state"])
    obs = env._raw_obs().astype(np.float32)
    hold = best = 0
    thr = []
    for _ in range(env.episode_steps):
        a, _ = model.predict(obs, deterministic=True)
        thr.append(float(np.mean((a + 1) / 2)))
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
    return best, (sum(thr) / max(len(thr), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/p1_v18/model.zip")
    ap.add_argument("--p2", default="models/p2_v18/model.zip")
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--diffs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    m1 = PPO.load(args.model, device="cpu")
    states = collect_p1_handoffs(m1, args.episodes, args.diffs, args.seed)
    states = states[:args.samples]
    print("=" * 78)
    print(f"  P2 判据难度曲线  |  交接样本 {len(states)} 个")
    w0 = [math.sqrt(sum(v ** 2 for v in h["state"]["omega"])) for h in states]
    print(f"  交接 |ω| 中位 {np.median(w0):.1f}  P90 {np.percentile(w0,90):.1f}  "
          f"max {max(w0):.1f}")
    print("=" * 78)

    print()
    print(f"  {'kp':>7}{'kd':>8}{'kz':>6}{'达成率':>9}{'保持中位':>10}{'力矩饱和率':>12}")
    print("  " + "-" * 52)
    curve = []
    for kp, kd, kz in [(0.010, 0.0012, 3.0), (0.015, 0.0016, 3.0),
                       (0.020, 0.0020, 3.0), (0.025, 0.0026, 4.0),
                       (0.030, 0.0030, 4.0), (0.040, 0.0040, 5.0),
                       (0.055, 0.0055, 6.0), (0.080, 0.0080, 8.0)]:
        bests, sats = [], []
        for hs in states:
            b, s = run_pd(hs, kp, kd, kz)
            bests.append(b)
            sats.append(s)
        ok = sum(1 for b in bests if b >= 50)
        curve.append(dict(kp=kp, kd=kd, kz=kz, rate=ok / len(states),
                          ok=ok, n=len(states),
                          best_med=float(np.median(bests)),
                          sat=float(np.mean(sats))))
        print(f"  {kp:>7.3f}{kd:>8.4f}{kz:>6.1f}"
              f"{ok/len(states)*100:>8.0f}%{np.median(bests):>10.0f}"
              f"{np.mean(sats)*100:>11.1f}%")

    # ---- RL 对照 ----
    print()
    print("  【RL 对照】同一个 P2 模型在这些状态上的表现：")
    try:
        m2 = PPO.load(args.p2, device="cpu")
        bests, thrs = [], []
        for hs in states:
            b, t = run_rl(hs, m2)
            bests.append(b)
            thrs.append(t)
        ok = sum(1 for b in bests if b >= 50)
        print(f"    达成率 {ok}/{len(states)}  保持中位 {np.median(bests):.0f} 步")
        print(f"    平均油门 {np.mean(thrs):.3f}  "
              f"（悬停需要 {qs.QuadParams().hover_throttle:.3f}）")
        rl_thr = float(np.mean(thrs))
        rl_ok = ok / len(states)
    except Exception as e:  # noqa: BLE001
        print(f"    ⚠️ 无法评估 P2: {e}")
        rl_thr, rl_ok = None, None

    # ---- 结论 ----
    print()
    print("=" * 78)
    print("  结论")
    print("=" * 78)
    good = [c for c in curve if c["rate"] >= 0.8]
    if good:
        crit = min(good, key=lambda c: c["kp"])
        print(f"  · 判据在 kp ≥ {crit['kp']:.3f} 时达成率 ≥80% "
              f"（kd={crit['kd']:.4f}, kz={crit['kz']:.1f}）")
        print(f"  · 该点力矩饱和率 {crit['sat']*100:.1f}%"
              f" → 执行器余量 {'充足' if crit['sat'] < 0.3 else '紧张'}")
    else:
        print("  · 扫描范围内没有配置达到 80% 达成率 → 判据可能确实过严")
    if rl_thr is not None:
        print(f"  · P2 RL 平均油门 {rl_thr:.3f} vs 悬停 "
              f"{qs.QuadParams().hover_throttle:.3f} → "
              f"{'推力不足（保不住高度）' if rl_thr < 0.65 else '推力基本够'}")

    out = os.path.join(_RL_DIR, "results", "pd_sweep.json")
    with open(out, "w") as f:
        json.dump(dict(curve=curve, rl_rate=rl_ok, rl_throttle=rl_thr,
                       w0=[float(x) for x in w0]), f, indent=1)
    print(f"\n  原始数据: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
