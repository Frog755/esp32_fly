#!/usr/bin/env python3
"""eval_pd_e2e.py —— 纯 PD 基线在端到端环境与 eval.py 严格判据下的成绩。
与 eval_hierarchical.py / eval.py 同判据同种子集，三者可直接对照。"""
import argparse, os, sys, time
from collections import Counter
import numpy as np
_RL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_RL_DIR, "env"))
import quad_sim_fast_v2 as qs
from eval import classify_outcome, wilson_interval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--diffs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--out", default=os.path.join(_RL_DIR, "results", "pd_e2e"))
    args = ap.parse_args()
    print("纯 PD 基线：端到端环境 + eval.py 严格判据")
    os.makedirs(args.out, exist_ok=True)
    results = {}
    t0 = time.time()
    for d in args.diffs:
        counter = Counter()
        for i in range(args.episodes):
            env = qs.ThrowRecoveryEnv(difficulty=d, domain_rand=1.0, episode_s=4.0,
                                      seed=args.seed + i)
            obs, _ = env.reset()
            info = {}
            k = 0
            for k in range(env.episode_steps):
                u = qs.pd_controller(env.sim, env.p)
                act = np.clip(u * 2 - 1, -1, 1)
                obs, _r, term, trunc, info = env.step(act)
                if term or trunc:
                    break
            counter[classify_outcome(info, trunc, k + 1, env.episode_steps)] += 1
        n = args.episodes
        n_ok = counter.get("success", 0)
        ci = wilson_interval(n_ok, n)
        print(f"\n  d={d:.1f}: {n_ok}/{n} = {n_ok/n*100:.1f}%  [95%CI {ci[0]*100:.0f}~{ci[1]*100:.0f}%]")
        for kk, v in counter.most_common():
            if kk != "success":
                print(f"      └ {kk:<22} {v:>4} ({v/n*100:.0f}%)")
        results[str(d)] = {"success_rate": n_ok/n, "counts": dict(counter)}
    import json
    with open(os.path.join(args.out, "eval_result.json"), "w") as f:
        json.dump({"policy": "PD-only", "elapsed_s": time.time()-t0, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s，结果写入 {args.out}")

if __name__ == "__main__":
    main()
