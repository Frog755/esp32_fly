#!/usr/bin/env python3
"""
eval_by_height.py —— 按初始高度分组统计成功率
================================================================

动机：
  从 180° 完全倒扣翻正，物理上必然要掉一段高度（自由落体 g）。
  若初始高度过低，无论策略多强都不可能成功。
  笼统的「总成功率」会把「物理不可能」的样本混进来，低估策略真实能力。

做法：
  用与 eval.py 完全相同的固定种子集，但额外记录每个回合的初始高度，
  然后按高度区间分组统计成功率。

  关键：种子集必须与 eval.py 一致，否则无法与其他 run 对比。

用法：
  python3 eval_by_height.py --model models/v6b_EA2_rot_small/model.zip
  python3 eval_by_height.py --model models/v6b_EA2_rot_small/model.zip --episodes 1000
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

_RL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_RL_DIR, "env"))

import quad_sim_fast_v2 as qs  # noqa: E402
from eval import classify_outcome, wilson_interval  # noqa: E402

# 高度分组（米）
HEIGHT_BINS = [(1.5, 1.8), (1.8, 2.1), (2.1, 2.4), (2.4, 2.6)]


def run_episode(policy_fn, difficulty, seed):
    """跑一个回合，额外返回初始高度。"""
    env = qs.ThrowRecoveryEnv(difficulty=difficulty, domain_rand=1.0,
                              episode_s=4.0, seed=seed)
    obs, _ = env.reset()
    init_z = float(env.sim.pos[2])      # ★ 记录初始高度

    done = False
    info = {}
    k = 0
    while not done:
        act = policy_fn(obs, env)
        obs, _r, term, trunc, info = env.step(act)
        k += 1
        if term or trunc:
            done = True

    outcome = classify_outcome(info, trunc, k, env.episode_steps)
    return outcome, init_z


def bucket_of(z):
    for lo, hi in HEIGHT_BINS:
        if lo <= z < hi:
            return f"{lo:.1f}-{hi:.1f}m"
    return "其他"


def main():
    ap = argparse.ArgumentParser(description="按初始高度分组评估")
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--diffs", type=float, nargs="+", default=[0.3, 0.7, 1.0])
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")

    def policy_fn(obs, env):
        a, _ = model.predict(obs, deterministic=True)
        return a

    print("=" * 78)
    print("  按初始高度分组的成功率评估")
    print("=" * 78)
    print(f"  模型     : {args.model}")
    print(f"  回合数   : {args.episodes} / 档")
    print(f"  种子集   : [{args.seed}, {args.seed + args.episodes})  （与 eval.py 一致）")
    print(f"  高度分组 : {[f'{lo}-{hi}m' for lo, hi in HEIGHT_BINS]}")
    print("=" * 78)

    all_results = {}

    for d in args.diffs:
        print(f"\n{'─'*78}")
        print(f"  难度 d={d}（最大倾角 {d*180:.0f}°）")
        print(f"{'─'*78}")

        # 按高度桶收集结果
        buckets = defaultdict(lambda: {"total": 0, "success": 0, "outcomes": []})

        for i in range(args.episodes):
            seed = args.seed + i
            outcome, init_z = run_episode(policy_fn, d, seed)
            b = bucket_of(init_z)
            buckets[b]["total"] += 1
            buckets[b]["outcomes"].append(outcome)
            if outcome == "success":
                buckets[b]["success"] += 1

        # 输出
        print(f"  {'高度区间':<12}{'样本数':>7}{'成功':>6}{'成功率':>9}{'95%CI':>16}")
        print(f"  {'-'*70}")

        result_d = {}
        for lo, hi in HEIGHT_BINS:
            b = f"{lo:.1f}-{hi:.1f}m"
            rec = buckets.get(b)
            if not rec or rec["total"] == 0:
                continue
            n, s = rec["total"], rec["success"]
            rate = s / n
            ci = wilson_interval(s, n)
            bar = "█" * int(rate * 30)
            print(f"  {b:<12}{n:>7}{s:>6}{rate*100:>8.1f}%"
                  f"   [{ci[0]*100:.0f},{ci[1]*100:.0f}]%  {bar}")
            result_d[b] = {
                "n": n, "success": s, "rate": round(rate, 4),
                "ci95": [round(ci[0], 4), round(ci[1], 4)],
                "outcome_counts": {k: rec["outcomes"].count(k)
                                   for k in set(rec["outcomes"])},
            }

        # 总计
        total_n = sum(r["total"] for r in buckets.values())
        total_s = sum(r["success"] for r in buckets.values())
        ci_all = wilson_interval(total_s, total_n)
        print(f"  {'-'*70}")
        print(f"  {'合计':<12}{total_n:>7}{total_s:>6}{total_s/total_n*100:>8.1f}%"
              f"   [{ci_all[0]*100:.0f},{ci_all[1]*100:.0f}]%")

        result_d["_total"] = {"n": total_n, "success": total_s,
                              "rate": round(total_s / total_n, 4)}
        all_results[str(d)] = result_d

    # ---------- 落盘 ----------
    out_dir = args.out or os.path.join(_RL_DIR, "results", "by_height",
                                       os.path.basename(os.path.dirname(args.model)))
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "model": args.model,
        "episodes_per_difficulty": args.episodes,
        "seed_set": f"[{args.seed}, {args.seed + args.episodes})",
        "height_bins": [f"{lo:.1f}-{hi:.1f}m" for lo, hi in HEIGHT_BINS],
        "results": all_results,
    }
    json_path = os.path.join(out_dir, "eval_by_height.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---------- Markdown ----------
    lines = ["# 按初始高度分组的成功率报告\n"]
    lines.append(f"- **模型**：`{args.model}`")
    lines.append(f"- **回合数**：{args.episodes} / 档")
    lines.append(f"- **种子集**：`{payload['seed_set']}`（与 eval.py 一致，可直接横向对比）\n")

    lines.append("## 为什么按高度分组\n")
    lines.append("从 180° 完全倒扣翻正，物理上必然要掉高度。自由落体估算：\n")
    lines.append("```")
    lines.append("h = 0.5 * g * t²")
    lines.append("翻转耗时 0.6s → 掉高 = 0.5 × 9.81 × 0.36 ≈ 1.77m")
    lines.append("翻转耗时 1.0s → 掉高 = 0.5 × 9.81 × 1.00 ≈ 4.91m")
    lines.append("```\n")
    lines.append("初始高度 1.5m 的样本，即便策略完美也来不及完成翻转。"
                 "这类样本混在总成功率里会系统性低估策略能力。\n")

    for d, bins in all_results.items():
        lines.append(f"## 难度 d={d}（最大倾角 {float(d)*180:.0f}°）\n")
        lines.append("| 高度区间 | 样本数 | 成功 | 成功率 | 95% CI |")
        lines.append("|---|---|---|---|---|")
        for b, r in bins.items():
            if b == "_total":
                continue
            ci = r["ci95"]
            lines.append(f"| {b} | {r['n']} | {r['success']} | "
                         f"**{r['rate']*100:.1f}%** | "
                         f"{ci[0]*100:.0f}~{ci[1]*100:.0f}% |")
        t = bins["_total"]
        lines.append(f"| **合计** | {t['n']} | {t['success']} | "
                     f"{t['rate']*100:.1f}% | — |")
        lines.append("")

    md_path = os.path.join(out_dir, "summary_by_height.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*78}")
    print(f"  ✅ 完成")
    print(f"  数据 : {json_path}")
    print(f"  报告 : {md_path}")
    print(f"{'='*78}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
