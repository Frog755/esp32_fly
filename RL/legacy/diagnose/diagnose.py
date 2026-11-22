#!/usr/bin/env python3
"""
失败模式诊断：RL 到底是在哪一步失败的？

对比 RL 策略 vs PD 基线，跑同样的难度，逐 step 记录：
  - 姿态 up_z 的变化
  - 存活步数（在哪一步终止）
  - 终止原因（触地 / 出界 / 超时）
  - 动作幅度统计

输出一张"死法分布表"，直接告诉你要改奖励的哪一项。

用法：
  QUAD_SIM_MODULE=quad_sim_fast python3 diagnose.py --model ../output/run_curriculum/model.zip
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
import os
import sys

import numpy as np

_ROOT = os.path.dirname(_LEGACY_ORIG_DIR)
sys.path.insert(0, os.path.join(_ROOT, "sim"))
_MOD = os.environ.get("QUAD_SIM_MODULE", "quad_sim_fast")
import importlib
qs = importlib.import_module(_MOD)

from stable_baselines3 import PPO  # noqa: E402


def rollout(env, policy_fn, max_steps=None):
    """跑一个 episode，返回诊断信息。"""
    obs, _ = env.reset()
    steps = max_steps or env.episode_steps
    up_zs, acts, omegas = [], [], []
    for k in range(steps):
        act = policy_fn(obs)
        obs, rew, term, trunc, info = env.step(act)
        up_zs.append(info["up_z"])
        omegas.append(info["omega_norm"])
        acts.append(act)
        if term or trunc:
            break
    s = env.sim
    final_z = float(s.pos[2])
    # 判定死法
    if final_z < 0.08:
        cause = "触地"
    elif abs(s.pos[0]) > 4.9 or abs(s.pos[1]) > 4.9 or s.pos[2] > 5.9:
        cause = "出界"
    else:
        cause = "超时"
    reached_up = max(up_zs) > 0.95
    return dict(steps=k + 1, cause=cause, reached_up=reached_up,
                max_up_z=float(max(up_zs)), final_up_z=float(up_zs[-1]),
                up_z_at_landing=float(up_zs[-1]),
                act_mean=float(np.mean(np.abs(acts))),
                act_var=float(np.var(acts)),
                mean_omega=float(np.mean(omegas)),
                final_z=final_z)


def summarize(name, results):
    n = len(results)
    causes = {}
    for r in results:
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    reached = sum(1 for r in results if r["reached_up"])
    alive = sum(1 for r in results if r["cause"] != "触地" or r["final_z"] > 0.05)
    print(f"\n[{name}]  n={n}")
    print(f"  曾转正(up_z>0.95): {reached}/{n} = {reached/n*100:.0f}%")
    print(f"  死法分布: " + ", ".join(f"{k} {v}次" for k, v in sorted(causes.items())))
    print(f"  平均存活步数: {np.mean([r['steps'] for r in results]):.0f} "
          f"({np.mean([r['steps'] for r in results])/200:.2f}s)")
    print(f"  平均动作幅度 |a|: {np.mean([r['act_mean'] for r in results]):.3f}")
    print(f"  动作方差: {np.mean([r['act_var'] for r in results]):.4f}")
    print(f"  平均角速度: {np.mean([r['mean_omega'] for r in results]):.2f} rad/s")
    # 触地但没转正的比例 —— 直接指向奖励设计问题
    crash_unrecovered = sum(1 for r in results
                            if r["cause"] == "触地" and not r["reached_up"])
    print(f"  触地且从未转正: {crash_unrecovered}/{n} = {crash_unrecovered/n*100:.0f}%"
          f"  ← 如果很高，说明策略根本没学会翻回来")
    return dict(n=n, reached=reached, causes=causes,
                mean_steps=float(np.mean([r["steps"] for r in results])),
                crash_unrecovered=crash_unrecovered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--difficulties", default="0.3,0.7,1.0")
    args = ap.parse_args()

    model = PPO.load(args.model)
    diffs = [float(x) for x in args.difficulties.split(",")]
    out = {}

    print("=" * 72)
    print(f"失败模式诊断  model={os.path.basename(args.model)}  sim={_MOD}")
    print("=" * 72)

    for d in diffs:
        # RL
        env = qs.ThrowRecoveryEnv(difficulty=d, domain_rand=1.0, seed=99)
        rl_res = []
        for _ in range(args.n):
            rl_res.append(rollout(env, lambda o: model.predict(o, deterministic=True)[0]))
        rl_sum = summarize(f"RL  difficulty={d}", rl_res)

        # PD
        env2 = qs.ThrowRecoveryEnv(difficulty=d, domain_rand=1.0, seed=99)
        pd_res = []
        for _ in range(args.n):
            pd_res.append(rollout(env2,
                                  lambda o, e=env2: np.clip(qs.pd_controller(e.sim, e.p) * 2 - 1, -1, 1)))
        pd_sum = summarize(f"PD  difficulty={d}", pd_res)

        out[str(d)] = dict(rl=rl_sum, pd=pd_sum)

    fp = os.path.join(_ROOT, "output", "diagnose.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n写入 {fp}")


if __name__ == "__main__":
    main()
