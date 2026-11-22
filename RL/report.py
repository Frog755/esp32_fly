#!/usr/bin/env python3
"""
report.py —— 多策略 A/B 对比报告（消融实验层）
================================================================

对标业界标准：
  - NVIDIA Isaac Lab   : 鼓励 baseline 与 candidate 在同一套 eval 条件下对比
  - AGILE              : 输出 HTML/Parquet 报告供快速诊断
  - arXiv 2409.09491   : 「A/B testing, 即交错评估不同策略，评估者应盲于策略名称」

设计：
  report.py 不自己跑仿真 —— 它只读取 eval.py 产出的 eval_result.json。
  这样保证：
    - 所有对比都建立在同一套固定种子集上
    - 报告可重复生成，不需要重新跑仿真
    - 训练中途也能随时补一个新 checkpoint 进来对比

用法：
  # 列出所有已评估的 run
  python3 report.py --list

  # 对比两个 run
  python3 report.py --runs baseline_run exp_v2_reward

  # 对比所有 run（含 PD 基线）
  python3 report.py --all

  # 指定难度档位对比
  python3 report.py --all --diff 1.0
"""

import argparse
import json
import os
import sys

_RL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_RL_DIR, "results")


# ================================================================ 读取
def load_result(run_name):
    path = os.path.join(RESULTS_DIR, run_name, "eval_result.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_runs():
    if not os.path.isdir(RESULTS_DIR):
        return []
    runs = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        p = os.path.join(RESULTS_DIR, d, "eval_result.json")
        if os.path.exists(p):
            runs.append(d)
    return runs


# ================================================================ 统计工具
def outcome_table(result):
    """从 eval_result 里抽取失败分类汇总。"""
    table = {}
    for d, r in result["results"].items():
        table[float(d)] = r["outcome_counts"]
    return table


def all_outcome_keys(results):
    keys = set()
    for r in results:
        for counts in outcome_table(r).values():
            keys.update(counts.keys())
    order = ["success", "crash_not_upright", "crash_upright",
             "timeout_not_upright", "timeout_unstable", "out_of_bounds"]
    return [k for k in order if k in keys] + sorted(keys - set(order))


# ================================================================ Markdown 报告
def build_markdown(runs_data, diff_filter=None):
    """runs_data: [(run_name, result_dict), ...]"""
    lines = []
    lines.append("# 策略 A/B 对比报告\n")
    lines.append("> 本报告由 `report.py` 生成，所有数据来自 `eval.py` 在**同一套固定种子集**上的评估。\n")

    # ---------- 评估条件一致性检查 ----------
    lines.append("## 评估条件一致性\n")
    conds = [r["evaluation_conditions"] for _, r in runs_data]
    same_seed = len({c["seed_set"] for c in conds}) == 1
    same_eps = len({c["episodes_per_difficulty"] for c in conds}) == 1

    lines.append("| 检查项 | 状态 |")
    lines.append("|---|---|")
    lines.append(f"| 种子集一致 | {'✅ 一致' if same_seed else '❌ **不一致，对比无效**'} |")
    lines.append(f"| 回合数一致 | {'✅ 一致' if same_eps else '❌ **不一致**'} |")
    if not same_seed:
        lines.append("\n⚠️ **警告：不同 run 使用了不同种子集，下面的对比不可信。**"
                     "请用相同 `--seed` 重跑 `eval.py`。\n")
    else:
        lines.append(f"| 种子集 | `{conds[0]['seed_set']}` |")
        lines.append(f"| 每档回合数 | {conds[0]['episodes_per_difficulty']} |\n")

    # ---------- 成功率对比 ----------
    all_diffs = sorted({float(d) for _, r in runs_data for d in r["results"]})
    if diff_filter is not None:
        all_diffs = [d for d in all_diffs if abs(d - diff_filter) < 1e-6]

    lines.append("## 成功率对比（含 95% 置信区间）\n")
    hdr = "| 难度 | " + " | ".join(n for n, _ in runs_data) + " |"
    lines.append(hdr)
    lines.append("|---|" + "---|" * len(runs_data))

    for d in all_diffs:
        cells = []
        for _n, r in runs_data:
            rec = r["results"].get(str(d))
            if rec is None:
                cells.append("—")
            else:
                ci = rec["success_ci95"]
                cells.append(f"**{rec['success_rate']*100:.0f}%** "
                             f"({rec['n_success']}/{rec['n_episodes']})"
                             f"<br><sub>{ci[0]*100:.0f}~{ci[1]*100:.0f}%</sub>")
        lines.append(f"| {d:.1f} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---------- 提升幅度 ----------
    if len(runs_data) == 2:
        base_name, base = runs_data[0]
        new_name, new = runs_data[1]
        lines.append(f"## 提升幅度（`{new_name}` vs `{base_name}`）\n")
        lines.append("| 难度 | 基线 | 新的 | 差值 | 判读 |")
        lines.append("|---|---|---|---|---|")
        for d in all_diffs:
            b = base["results"].get(str(d))
            n = new["results"].get(str(d))
            if b is None or n is None:
                continue
            delta = n["success_rate"] - b["success_rate"]
            # 置信区间是否重叠（判断差异是否显著）
            b_ci, n_ci = b["success_ci95"], n["success_ci95"]
            overlap = not (n_ci[0] > b_ci[1] or n_ci[1] < b_ci[0])
            if delta > 0.05 and not overlap:
                verdict = "✅ 显著提升"
            elif delta > 0.05 and overlap:
                verdict = "⚠️ 提升但不显著（CI 重叠）"
            elif delta < -0.05 and not overlap:
                verdict = "❌ 显著退化"
            elif delta < -0.05:
                verdict = "⚠️ 退化但不显著"
            else:
                verdict = "≈ 无变化"
            lines.append(f"| {d:.1f} | {b['success_rate']*100:.0f}% | "
                         f"{n['success_rate']*100:.0f}% | "
                         f"**{delta*100:+.0f}pp** | {verdict} |")
        lines.append("")
        lines.append("> `pp` = percentage point（百分点）。"
                     "置信区间不重叠才算「显著」—— 这是判断改进是否真实有效的标准做法。\n")

    # ---------- 失败分类对比 ----------
    lines.append("## 失败分类对比（每档难度失败次数）\n")
    keys = all_outcome_keys([r for _, r in runs_data])
    fail_keys = [k for k in keys if k != "success"]

    for d in all_diffs:
        lines.append(f"### 难度 d={d:.1f}\n")
        lines.append("| 结局类别 | " + " | ".join(n for n, _ in runs_data) + " |")
        lines.append("|---|" + "---|" * len(runs_data))
        for k in keys:
            cells = []
            for _n, r in runs_data:
                rec = r["results"].get(str(d))
                if rec is None:
                    cells.append("—")
                else:
                    v = rec["outcome_counts"].get(k, 0)
                    marker = " ⭐" if k == "success" and v > 0 else ""
                    cells.append(f"{v}{marker}")
            lines.append(f"| `{k}` | " + " | ".join(cells) + " |")
        lines.append("")

    # ---------- 回正耗时 ----------
    lines.append("## 回正耗时对比（中位数，秒）\n")
    lines.append("| 难度 | " + " | ".join(n for n, _ in runs_data) + " |")
    lines.append("|---|" + "---|" * len(runs_data))
    for d in all_diffs:
        cells = []
        for _n, r in runs_data:
            rec = r["results"].get(str(d))
            v = rec.get("recover_time_median_s") if rec else None
            cells.append("—" if v is None else f"{v:.2f}")
        lines.append(f"| {d:.1f} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---------- 结论建议 ----------
    lines.append("## 自动判读\n")
    if len(runs_data) >= 2:
        base_name, base = runs_data[0]
        new_name, new = runs_data[-1]
        # 找最大难度档
        d_max = max(all_diffs)
        b = base["results"].get(str(d_max))
        n = new["results"].get(str(d_max))
        if b and n:
            lines.append(f"在最高难度 d={d_max:.1f} 上：")
            lines.append(f"- `{base_name}`：{b['success_rate']*100:.0f}% "
                         f"({b['n_success']}/{b['n_episodes']})")
            lines.append(f"- `{new_name}`：{n['success_rate']*100:.0f}% "
                         f"({n['n_success']}/{n['n_episodes']})")
            # 主要失败模式
            top_fail_new = sorted(
                [(k, v) for k, v in n["outcome_counts"].items() if k != "success"],
                key=lambda x: -x[1])
            if top_fail_new:
                lines.append(f"\n`{new_name}` 在极限档的首要失败模式是 "
                             f"`{top_fail_new[0][0]}`（{top_fail_new[0][1]} 次）。")
                lines.append("→ 下一步优先针对这个失败类别调整奖励权重。\n")
    lines.append("")

    return "\n".join(lines)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="多策略 A/B 对比报告")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="要对比的 run 名称（按顺序，第一个作为基线）")
    ap.add_argument("--all", action="store_true", help="对比所有已评估的 run")
    ap.add_argument("--list", action="store_true", help="只列出已评估的 run")
    ap.add_argument("--diff", type=float, default=None,
                    help="只对比某个难度档")
    ap.add_argument("--out", default=None, help="输出 markdown 路径")
    args = ap.parse_args()

    runs = list_runs()

    if args.list or (not args.runs and not args.all):
        print("=" * 60)
        print("  已评估的 run")
        print("=" * 60)
        if not runs:
            print("  （无）请先运行 eval.py")
            return 0
        for r in runs:
            res = load_result(r)
            if res:
                # 汇总所有难度档的成功率
                rates = [v["success_rate"] for v in res["results"].values()]
                avg = sum(rates) / len(rates) * 100
                print(f"  {r:<28} 平均成功率 {avg:5.1f}%  "
                      f"评估于 {res['evaluated_at']}")
        print()
        print("用法: python3 report.py --runs <基线run> <对比run>")
        return 0

    targets = runs if args.all else args.runs
    runs_data = []
    for name in targets:
        res = load_result(name)
        if res is None:
            print(f"⚠️ 跳过 {name}：未找到 eval_result.json")
            continue
        runs_data.append((name, res))

    if not runs_data:
        print("❌ 没有可用的评估结果")
        return 1

    if len(runs_data) < 2 and not args.all:
        print("⚠️ 只有一个 run，无法对比。用 --all 或提供两个以上 run。")

    md = build_markdown(runs_data, args.diff)

    out = args.out or os.path.join(RESULTS_DIR, "comparison_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print()
    print("=" * 60)
    print(f"✅ 报告已保存: {out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
