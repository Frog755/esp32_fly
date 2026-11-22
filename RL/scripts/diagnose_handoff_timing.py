#!/usr/bin/env python3
"""
diagnose_handoff_timing.py —— 交接时机的可达性分析（回答 mj 两个提议）
================================================================

mj 的提议（2026-11-04）：
  「P1 推出的姿态条件给宽松一点，给 P2 更多时间去减速稳定姿态」
  「我最终只需要他能稳定姿态朝上 高度尽量在原范围允许范围内就行了
    yaw 角速度怎么转都无所谓 但是 z 轴角速度肯定不行」

本脚本要回答三个问题，全部用【可测量的数】回答，不靠推理：

  Q1「提前交接」：沿 P1 轨迹，ω_tilt 随 up_z 怎么变？
     如果 up_z 从 0.7 降到 0.4 时 ω 明显更小，提前交接就是真收益。
     如果 ω 在 up_z 升高时反而上升，提前交接也救不了。

  Q2「时间预算」：从交接时刻到坠地还有多少步？够不够刹车？

  Q3「物理可达性」（最关键）：
     给一个【远超 RL 能力的理想 PD 控制器】（几何姿态控制 + 满力矩），
     从 P1 真实交接状态出发，看它能不能做到
        up_z>0.98 且 |vz|<0.5 且 ω_tilt<1，连续 50 步。
     如果理想 PD 都做不到（<8/10），那 P2 的 0% 不是训练问题，是判据不可达，
     任何 RL 都学不出来 —— 这是我在元规则里定的「判据必须先过 PD 验证」。

用法：
  python3 scripts/diagnose_handoff_timing.py --episodes 200
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


# ============================================================ 理想 PD 控制器
def ideal_pd(env, kp=0.030, kd=0.0030, kz=4.0, kxy=2.0, vz_ref=0.0):
    """几何姿态 PD + 位置保持（比 env 里内置的 pd_controller 更强）。

    思路：把当前机体系 z 轴转到世界 z 轴，用最短旋转；
    角速度误差项提供阻尼，力矩直接映射到电机混控。
    高度用 vz_ref 的 PD，水平速度也做阻尼（避免漂出界）。
    """
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

    # 期望：机体系 z 轴 → 世界 z 轴。误差轴 = z_b × e_z（机体系下）
    cx, cy, cz = zy * 1.0 - zz * 0.0, zz * 0.0 - zx * 1.0, 0.0
    ex = r00 * cx + r10 * cy + r20 * cz
    ey = r01 * cx + r11 * cy + r21 * cz
    ez = r02 * cx + r12 * cy + r22 * cz

    wx, wy, wz = sim.w
    tx = kp * ex - kd * wx
    ty = kp * ey - kd * wy
    tz = -0.0008 * wz          # 偏航只做阻尼，不跟踪朝向（mj：yaw 无所谓）

    # 高度：PD 控 vz（保高优先，避免坠地）
    vz = sim.vel[2]
    az_ref = kz * (vz_ref - vz)
    tilt_ok = max(zz, 0.0)
    Fz = p.mass * (G + az_ref) / max(zz, 0.5) * max(tilt_ok, 0.3)
    # 水平速度阻尼（在倾斜已可控时轻微用倾角换向，这里简化为不处理，
    # 因为 mj 的目标不含位置收敛）
    Fz = float(max(Fz, 0.25 * p.mass * G))

    rp = p.rotor_pos
    M = np.vstack([np.ones(4), rp[:, 1], -rp[:, 0], -p.rotor_dir * p.k_m_ratio])
    T = np.linalg.solve(M, np.r_[Fz, tx, ty, tz])
    T = np.clip(T, 0, p.thrust_max_per_motor)
    u = np.sqrt(T / p.thrust_max_per_motor) * (p.v_ref / max(sim.v_batt, 3.0))
    return np.clip(u * 2.0 - 1.0, -1, 1)


def omega_tilt(sim):
    w = sim.w
    return math.sqrt(w[0] ** 2 + w[1] ** 2)


def pctl(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


# ============================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/p1_v18/model.zip")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--diffs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--pd-trials", type=int, default=40)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")

    # ---------- Q1 + Q2：沿 P1 轨迹采样 ----------
    prof = {}          # up_z 桶 → ω_tilt 列表（整条轨迹）
    cross = {}         # 阈值 → 首次越过瞬间的 ω_tilt 列表（"提前交接"对应这个）
    cross_states = {}  # 阈值 → 该瞬间的完整状态
    handoff_states = []
    for i in range(args.episodes):
        d = args.diffs[i % len(args.diffs)]
        seed = args.seed + i
        env = qs.ThrowRecoveryEnv(difficulty=d, domain_rand=1.0,
                                  episode_s=4.0, seed=seed, phase=1)
        obs, _ = env.reset()
        flipped = False
        crossed = set()        # 本回合已越过的阈值（必须逐回合重置！）
        for k in range(env.episode_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            upz = float(info["up_z"])
            ot = omega_tilt(env.sim)
            # 按 up_z 分桶（只统计已翻过的过程，up_z 从负到正）
            b = round(upz * 10) / 10
            prof.setdefault(b, []).append(ot)
            # 首次越过各阈值（单调上升时才有意义）
            for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                if upz > th and th not in crossed:
                    crossed.add(th)
                    cross.setdefault(th, []).append(ot)
                    cross_states.setdefault(th, []).append(dict(
                        d=d, seed=seed, step=k + 1, omega=ot,
                        state=dict(pos=list(env.sim.pos),
                                   vel=list(env.sim.vel),
                                   quat=list(env.sim.q),
                                   omega=list(env.sim.w), soc=1.0),
                    ))
            if not flipped and upz > 0.7:
                flipped = True
                handoff_states.append(dict(
                    d=d, seed=seed, step=k + 1,
                    state=dict(pos=list(env.sim.pos), vel=list(env.sim.vel),
                               quat=list(env.sim.q), omega=list(env.sim.w),
                               soc=1.0),
                ))
                break
            if term or trunc:
                break

    print("=" * 78)
    print("  Q1  沿 P1 轨迹：ω_tilt 随 up_z 的变化（能否靠提前交交接降 ω）")
    print("=" * 78)
    print(f"  {'up_z':>7}{'样本':>8}{'ω_tilt中位':>13}{'P90':>9}{'max':>9}")
    print("  " + "-" * 48)
    for b in sorted(prof):
        if b < -0.2:
            continue
        arr = prof[b]
        if len(arr) < 20:
            continue
        print(f"  {b:>7.1f}{len(arr):>8}{pctl(arr,50):>13.2f}"
              f"{pctl(arr,90):>9.2f}{max(arr):>9.2f}")

    print()
    print("  【Q1b 关键】首次越过各 up_z 阈值瞬间的 ω_tilt —— 这才是\"提前交接\"的真实收益")
    print(f"  {'阈值':>7}{'样本':>8}{'ω_tilt中位':>13}{'P90':>9}{'<28占比':>10}{'<35占比':>10}")
    print("  " + "-" * 57)
    for th in sorted(cross):
        arr = cross[th]
        if len(arr) < 10:
            continue
        f28 = sum(1 for v in arr if v < 28) / len(arr) * 100
        f35 = sum(1 for v in arr if v < 35) / len(arr) * 100
        print(f"  {th:>7.1f}{len(arr):>8}{pctl(arr,50):>13.2f}"
              f"{pctl(arr,90):>9.2f}{f28:>9.1f}%{f35:>9.1f}%")

    # ---------- Q2：交接后时间预算 ----------
    print()
    print("=" * 78)
    print("  Q2  交接后剩余步数（自由落体到坠地的时间预算）")
    print("=" * 78)
    remains = []
    for hs in handoff_states[:80]:
        d, seed = hs["d"], hs["seed"]
        env2 = qs.ThrowRecoveryEnv(difficulty=d, domain_rand=0.0,
                                   episode_s=4.0, seed=seed, phase=2)
        env2.reset()
        env2.sim.reset(**hs["state"])
        n = 0
        for k2 in range(env2.episode_steps):
            u = ideal_pd(env2)      # 用理想 PD 探"最多能活多久"
            _o, _r, t2, tr2, _i = env2.step(u)
            n += 1
            if t2 or tr2:
                break
        remains.append(n)
    if remains:
        print(f"  理想 PD 下从交接能存活: 中位 {pctl(remains,50):.0f} 步 "
              f"P10 {pctl(remains,10):.0f} 步  min {min(remains)} 步")
        print("  （P2 需要 刹车 + 转身 + 稳 50 步；若中位 < 100 步步数就很紧）")

    # ---------- Q3：物理可达性 ----------
    print()
    print("=" * 78)
    print("  Q3  理想 PD 能否达成 P2 判据（up_z>0.98 & |vz|<0.5 & ω_tilt<1，连续50步）")
    print("=" * 78)
    results = []
    for hs in handoff_states[:args.pd_trials]:
        d, seed = hs["d"], hs["seed"]
        env2 = qs.ThrowRecoveryEnv(difficulty=d, domain_rand=0.0,
                                   episode_s=4.0, seed=seed, phase=2)
        env2.reset()
        env2.sim.reset(**hs["state"])
        hold = best = 0
        ok_up = ok_vz = ok_om = 0
        w0 = math.sqrt(sum(v ** 2 for v in hs["state"]["omega"]))
        for k2 in range(env2.episode_steps):
            u = ideal_pd(env2)
            _o, _r, t2, tr2, info = env2.step(u)
            zz = 1 - 2 * (env2.sim.q[1] ** 2 + env2.sim.q[2] ** 2)
            c1 = zz > 0.98
            c2 = abs(env2.sim.vel[2]) < 0.5
            c3 = omega_tilt(env2.sim) < 1.0
            ok_up += c1
            ok_vz += c2
            ok_om += c3
            if c1 and c2 and c3:
                hold += 1
                best = max(best, hold)
            else:
                hold = 0
            if t2 or tr2:
                break
        results.append(dict(d=d, w0=w0, best=best, up=ok_up, vz=ok_vz, om=ok_om))

    n_ok = sum(1 for r in results if r["best"] >= 50)
    print(f"  {'d':>5}{'交接ω':>9}{'最长保持':>10}{'up_z步':>9}{'vz步':>8}{'ω步':>8}")
    print("  " + "-" * 50)
    for r in results[:20]:
        print(f"  {r['d']:>5.1f}{r['w0']:>9.1f}{r['best']:>10}"
              f"{r['up']:>9}{r['vz']:>8}{r['om']:>8}")
    print()
    print(f"  ✅ 达成 50 步连续: {n_ok}/{len(results)} "
          f"= {n_ok/max(len(results),1)*100:.0f}%")
    print("  判定标准（元规则）：<8/10 即判据物理不可达，不该交给 RL 去学")

    # 分档统计：交接 ω 与能否达成的关系
    print()
    print("  【关键】交接 ω 分档 vs 理想 PD 达成率：")
    for lo, hi in ((0, 15), (15, 25), (25, 35), (35, 45), (45, 61)):
        grp = [r for r in results if lo <= r["w0"] < hi]
        if not grp:
            continue
        ok = sum(1 for r in grp if r["best"] >= 50)
        print(f"    ω∈[{lo:>2},{hi:>2})  {ok:>2}/{len(grp):<3} = "
              f"{ok/len(grp)*100:5.1f}%  最长保持中位 "
              f"{pctl([r['best'] for r in grp],50):.0f} 步")

    # ---------- Q4：分交接阈值 × 理想 PD 达成率 ----------
    print()
    print("=" * 78)
    print("  Q4  按不同交接阈值起跑，理想 PD 达成 P2 判据的成功率")
    print("  （直接回答 mj 的提议：P1 姿态条件放宽是否给 P2 减负）")
    print("=" * 78)
    print(f"  {'交接up_z':>9}{'样本':>7}{'交接ω中位':>11}{'达成率':>9}{'最长保持中位':>13}")
    print("  " + "-" * 51)
    q4 = {}
    for th in sorted(cross_states):
        grp = cross_states[th][:args.pd_trials]
        if len(grp) < 8:
            continue
        ok = 0
        bests = []
        for hs in grp:
            env2 = qs.ThrowRecoveryEnv(difficulty=hs["d"], domain_rand=0.0,
                                       episode_s=4.0, seed=hs["seed"], phase=2)
            env2.reset()
            env2.sim.reset(**hs["state"])
            hold = best = 0
            for k2 in range(env2.episode_steps):
                u = ideal_pd(env2)
                _o, _r, t2, tr2, _i = env2.step(u)
                zz = env2.sim.q[0] ** 2
                upnow = 1 - 2 * (env2.sim.q[1] ** 2 + env2.sim.q[2] ** 2)
                c1 = upnow > 0.98
                c2 = abs(env2.sim.vel[2]) < 0.5
                c3 = omega_tilt(env2.sim) < 1.0
                if c1 and c2 and c3:
                    hold += 1
                    best = max(best, hold)
                else:
                    hold = 0
                if t2 or tr2:
                    break
            bests.append(best)
            if best >= 50:
                ok += 1
        wmed = pctl([g["omega"] for g in grp], 50)
        q4[th] = dict(n=len(grp), ok=ok, rate=ok / len(grp), wmed=wmed,
                      best_med=pctl(bests, 50))
        print(f"  {th:>9.1f}{len(grp):>7}{wmed:>11.1f}"
              f"{ok/len(grp)*100:>8.0f}%{pctl(bests,50):>13.0f}")

    out = os.path.join(_RL_DIR, "results", "handoff_timing.json")
    with open(out, "w") as f:
        json.dump(dict(profile={str(k): [pctl(v, 50), pctl(v, 90), len(v)]
                                for k, v in prof.items()},
                       remains=[int(v) for v in remains],
                       q1b={str(k): dict(n=len(v), med=pctl(v, 50),
                                         p90=pctl(v, 90))
                            for k, v in cross.items()},
                       q4={str(k): v for k, v in q4.items()},
                       pd_results=[{k: (int(x) if k in ("best", "up", "vz", "om")
                                        else float(x))
                                    for k, x in r.items()} for r in results]),
                  f, indent=1)
    print(f"\n  原始数据: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
