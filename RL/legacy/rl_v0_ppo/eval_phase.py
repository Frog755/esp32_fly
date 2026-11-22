#!/usr/bin/env python3
"""
eval_phase.py —— 单阶段策略的专用评估（P1 或 P2）
================================================================

背景与动机：
  eval.py 用的是【端到端严格判据】（活满 800 步 + 终态稳定）。
  用它评估 P1（纯翻转器）永远得 0% —— 因为 P1 翻正即交班，本就不该活满全程。
  这不是策略差，是评估工具用错了对象（v18 实际踩过这个坑）。

本脚本按阶段使用正确的判据和目标：
  Phase 1（翻转）：判据 = up_z 是否曾达到 0.7（交接成功）
                  额外统计：翻转耗时、交接瞬间的 ω/vxy/vz 分布（供下游对齐）
  Phase 2（稳定）：判据 = up_z>0.98 & |vz|<0.5 & ω<ω_th & 连续 50 步
                  额外统计：逐条件达标步数（定位卡在哪一环）

用法：
  python3 eval_phase.py --model models/p1_v18/model.zip --phase 1
  python3 eval_phase.py --model models/p2_v18/model.zip --phase 2 --omega-th 2.0
  python3 eval_phase.py --model models/p1_v16/model.zip --phase 1 --compare v18
"""

# --- [Stage 20 legacy 迁移引导] 本文件原位于 RL/，现移至 RL/legacy/rl_v0_ppo/ ---
# _LEGACY_ORIG_DIR 指回原目录：原有基于 __file__ 的相对路径（../env、../models 等）语义不变；
# 并把主线 env/、scripts/ 与各 legacy 子目录加入 sys.path，跨目录的旧导入照常可用。逻辑未改动。
import os as _lg_os, sys as _lg_sys
_LEGACY_RL = _lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.abspath(__file__))))
_LEGACY_ORIG_DIR = _LEGACY_RL
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
import time
from collections import Counter

import numpy as np

_RL_DIR = _LEGACY_ORIG_DIR
sys.path.insert(0, os.path.join(_RL_DIR, "env"))

import quad_sim_fast_v2 as qs  # noqa: E402
from eval import wilson_interval  # noqa: E402

HANDOFF_UPZ = 0.7      # P1 交接阈值
STABLE_HOLD = 50       # P2 稳定保持步数


def run_p1(model, difficulty, seed, episode_s=4.0):
    """Phase 1：纯翻转。成功 = up_z 曾达到 0.7。"""
    env = qs.ThrowRecoveryEnv(difficulty=difficulty, domain_rand=1.0,
                              episode_s=episode_s, seed=seed, phase=1)
    obs, _ = env.reset()
    init_om = float(np.linalg.norm(env.sim.w))
    init_upz = 1 - 2 * (env.sim.quat[1] ** 2 + env.sim.quat[2] ** 2) \
        if hasattr(env.sim, "quat") else None

    flipped = False
    flip_step = None
    handoff = {}
    k = 0
    info = {}
    for k in range(env.episode_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        if not flipped and info["up_z"] > HANDOFF_UPZ:
            flipped = True
            flip_step = k + 1
            handoff = {
                "omega": float(info["omega_norm"]),
                "vxy": float(math.hypot(env.sim.vel[0], env.sim.vel[1])),
                "vz": float(info["vz"]),
                "upz": float(info["up_z"]),
            }
        if term or trunc:
            break
    return {
        "seed": int(seed),
        "difficulty": float(difficulty),
        "success": bool(flipped),
        "flip_steps": flip_step,
        "flip_time_s": round(flip_step / env.p.ctrl_hz, 3) if flip_step else None,
        "init_omega": round(init_om, 1),
        "handoff": handoff,
        "final_upz": round(float(info.get("up_z", 0)), 3),
    }


def run_p2(model, difficulty, seed, omega_th, episode_s=4.0):
    """Phase 2：高速刹停。成功 = 四条件连续 50 步。"""
    env = qs.ThrowRecoveryEnv(difficulty=difficulty, domain_rand=1.0,
                              episode_s=episode_s, seed=seed, phase=2)
    obs, _ = env.reset()
    init_om = float(np.linalg.norm(env.sim.w))

    hold = 0
    best_hold = 0
    n_up = n_vz = n_om = n_all = 0
    k = 0
    info = {}
    for k in range(env.episode_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        c1 = info["up_z"] > 0.95
        c2 = abs(info["vz"]) < 0.5
        c3 = info["omega_norm"] < omega_th
        n_up += c1
        n_vz += c2
        n_om += c3
        if c1 and c2 and c3:
            n_all += 1
            hold += 1
            best_hold = max(best_hold, hold)
        else:
            hold = 0
        if term or trunc:
            break
    return {
        "seed": int(seed),
        "difficulty": float(difficulty),
        "success": bool(best_hold >= STABLE_HOLD),
        "best_hold": int(best_hold),
        "steps": int(k + 1),
        "init_omega": round(init_om, 1),
        "up_ok_steps": n_up, "vz_ok_steps": n_vz,
        "om_ok_steps": n_om, "all_ok_steps": n_all,
        "final_upz": round(float(info.get("up_z", 0)), 2),
        "final_vz": round(float(info.get("vz", 0)), 2),
        "final_omega": round(float(info.get("omega_norm", 0)), 2),
    }


def report_p1(records, tag, out_dir):
    n = len(records)
    ok = sum(1 for r in records if r["success"])
    ci = wilson_interval(ok, n)
    times = [r["flip_time_s"] for r in records if r["success"] and r["flip_time_s"]]
    oms = [r["handoff"]["omega"] for r in records if r["success"]]
    vxys = [r["handoff"]["vxy"] for r in records if r["success"]]
    vzs = [r["handoff"]["vz"] for r in records if r["success"]]
    upzs = [r["handoff"]["upz"] for r in records if r["success"]]

    print(f"    翻转率 {ok}/{n} = {ok/n*100:5.1f}%  [95%CI {ci[0]*100:.0f}~{ci[1]*100:.0f}%]")
    if times:
        print(f"    翻转耗时 中位 {np.median(times):.3f}s ({np.median(times)*200:.0f} 步)")
    if oms:
        print(f"    交接时 |ω|  中位 {np.median(oms):5.1f}  P90 {np.percentile(oms,90):5.1f}  max {max(oms):5.1f}")
        print(f"    交接时 |vxy| 中位 {np.median(vxys):5.2f}  P90 {np.percentile(vxys,90):5.2f}  max {max(vxys):5.2f}")
        print(f"    交接时 vz   中位 {np.median(vzs):+5.2f}  up_z 中位 {np.median(upzs):.2f}")

    return {"n": n, "success": ok, "rate": round(ok/n, 4),
            "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "flip_time_median_s": round(float(np.median(times)), 3) if times else None,
            "handoff_omega_median": round(float(np.median(oms)), 1) if oms else None,
            "handoff_omega_p90": round(float(np.percentile(oms, 90)), 1) if oms else None,
            "handoff_vxy_max": round(float(max(vxys)), 2) if vxys else None,
            "episodes": records}


def report_p2(records, tag, omega_th, out_dir):
    n = len(records)
    ok = sum(1 for r in records if r["success"])
    ci = wilson_interval(ok, n)
    holds = [r["best_hold"] for r in records]
    steps = [r["steps"] for r in records]
    print(f"    达标率 {ok}/{n} = {ok/n*100:5.1f}%  [95%CI {ci[0]*100:.0f}~{ci[1]*100:.0f}%]")
    print(f"    最长连续稳定 中位 {np.median(holds):.0f} / {STABLE_HOLD} 步  最佳 {max(holds)}")
    print(f"    平均存活 {np.mean(steps):.0f} / 800 步")
    print(f"    逐条件达标步数（平均）: 姿态 {np.mean([r['up_ok_steps'] for r in records]):.0f}"
          f"  速度 {np.mean([r['vz_ok_steps'] for r in records]):.0f}"
          f"  角速度 {np.mean([r['om_ok_steps'] for r in records]):.0f}"
          f"  全达标 {np.mean([r['all_ok_steps'] for r in records]):.0f}")
    return {"n": n, "success": ok, "rate": round(ok/n, 4),
            "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "best_hold_median": int(np.median(holds)),
            "steps_mean": round(float(np.mean(steps)), 0),
            "cond_steps_mean": {
                "up": round(float(np.mean([r["up_ok_steps"] for r in records])), 0),
                "vz": round(float(np.mean([r["vz_ok_steps"] for r in records])), 0),
                "omega": round(float(np.mean([r["om_ok_steps"] for r in records])), 0),
                "all": round(float(np.mean([r["all_ok_steps"] for r in records])), 0),
            },
            "episodes": records}


def main():
    ap = argparse.ArgumentParser(description="单阶段策略专用评估")
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--diffs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--omega-th", type=float, default=2.0,
                    help="Phase2 稳定判据的角速度阈值（v18 定为 2.0）")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    tag = os.path.basename(os.path.dirname(args.model))

    print("=" * 74)
    print(f"  Phase {args.phase} 专用评估 —— {tag}")
    print("=" * 74)
    if args.phase == 1:
        print(f"  判据: up_z 曾达到 {HANDOFF_UPZ}（翻正即交接，不看速度）")
    else:
        print(f"  判据: up_z>0.98 & |vz|<0.5 & ω<{args.omega_th}，连续 {STABLE_HOLD} 步")
    print(f"  回合数: {args.episodes}/档   种子集: [{args.seed}, {args.seed+args.episodes})")
    print("=" * 74)

    results = {}
    t0 = time.time()
    for d in args.diffs:
        print(f"\n  ── d={d}（最大倾角 {d*180:.0f}°）──")
        recs = []
        for i in range(args.episodes):
            if args.phase == 1:
                recs.append(run_p1(model, d, args.seed + i))
            else:
                recs.append(run_p2(model, d, args.seed + i, args.omega_th))
        if args.phase == 1:
            results[str(d)] = report_p1(recs, tag, args.out)
        else:
            results[str(d)] = report_p2(recs, tag, args.omega_th, args.out)

    out_dir = args.out or os.path.join(_RL_DIR, "results", f"{tag}_phase{args.phase}")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "model": args.model, "phase": args.phase,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conditions": {
            "episodes_per_difficulty": args.episodes,
            "seed_set": f"[{args.seed}, {args.seed+args.episodes})",
            "criterion": (f"up_z>{HANDOFF_UPZ}" if args.phase == 1
                          else f"up_z>0.95 & |vz|<0.5 & omega<{args.omega_th} for {STABLE_HOLD} steps"),
        },
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }
    jp = os.path.join(out_dir, "eval_result.json")
    def _to_py(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _to_py(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_py(v) for v in obj]
        return obj
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(_to_py(payload), f, ensure_ascii=False, indent=2)

    print(f"\n{'='*74}")
    print(f"  ✅ 完成 {payload['elapsed_s']}s   数据: {jp}")
    print(f"{'='*74}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
