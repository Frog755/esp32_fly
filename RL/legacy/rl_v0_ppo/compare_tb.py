#!/usr/bin/env python3
"""
compare_tb.py —— 两个训练 run 的 TensorBoard 曲线对比图
================================================================

对标业界做法：
  Isaac Lab / AGILE 在论文与周报里对比实验时，会把多个 run 的同名 scalar
  叠在同一张图上，用统一坐标轴、统一种子、统一 eval 频率对比。

为什么要脚本化：
  TensorBoard 网页可以手动勾选多个 run，但没法：
    - 导出成可放进报告的静态图
    - 保证两次实验的 eval 频率/坐标对齐
    - 在 CI 里自动生成
  所以正规做法是用脚本读 event 文件自己画。

用法：
  python3 compare_tb.py --runs baseline_run v3_reward
  python3 compare_tb.py --runs baseline_run v3_reward --metric bench/d1.0_success
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
import glob
import os
import sys

import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC",
                                   "Droid Sans Fallback", "AR PL UMing CN",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_RL_DIR = _LEGACY_ORIG_DIR

# 要对比的关键指标（顺序即图的顺序）
KEY_METRICS = [
    ("eval/success_rate", "评估成功率", "%"),
    ("eval/difficulty", "课程难度", ""),
    ("rollout/ep_rew_mean", "回合平均奖励", ""),
    ("rollout/ep_len_mean", "回合平均长度", "步"),
    ("train/explained_variance", "Critic 解释方差", ""),
    ("train/approx_kl", "策略更新幅度 KL", ""),
]

COLORS = ["#999999", "#d62728", "#1f77b4", "#2ca02c", "#9467bd"]


def find_run_dir(run_name):
    """定位 run 的 event 目录（sb3 会加 _1 后缀）"""
    patterns = [
        os.path.join(_RL_DIR, "models", run_name, "tb", f"{run_name}*"),
        os.path.join(_RL_DIR, "models", run_name, "tb", f"{run_name}_s*"),
    ]
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[-1]
    direct = os.path.join(_RL_DIR, "models", run_name, "tb")
    if os.path.isdir(direct):
        sub = [d for d in glob.glob(os.path.join(direct, "*")) if os.path.isdir(d)]
        if sub:
            return sorted(sub)[-1]
    return None


def load_scalars(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def plot_compare(run_names, out_png, metrics=None):
    metrics = metrics or KEY_METRICS
    loaded = []
    for name in run_names:
        d = find_run_dir(name)
        if d is None:
            print(f"⚠️ 找不到 {name} 的 event 目录，跳过")
            continue
        loaded.append((name, load_scalars(d)))

    if len(loaded) < 1:
        print("❌ 没有可用的 run")
        return None

    n_plots = len(metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=(11, 3.0 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (tag, label, unit) in zip(axes, metrics):
        any_data = False
        for i, (name, ea) in enumerate(loaded):
            if tag not in ea.Tags()["scalars"]:
                continue
            ev = ea.Scalars(tag)
            steps = np.array([e.step for e in ev], dtype=float)
            vals = np.array([e.value for e in ev], dtype=float)
            # 成功率转百分比显示
            if tag == "eval/success_rate":
                vals = vals * 100.0
            ax.plot(steps, vals, color=COLORS[i % len(COLORS)],
                    lw=1.8, label=name, marker="o" if len(ev) < 40 else None,
                    ms=3.5 if len(ev) < 40 else 0)
            any_data = True
        if not any_data:
            ax.text(0.5, 0.5, f"{label}（无数据）", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_ylabel(f"{label}\n{unit}" if unit else label, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("训练步数 (timesteps)", fontsize=11)
    fig.suptitle("训练曲线对比：" + " vs ".join(n for n, _ in loaded),
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_bench(run_names, out_png):
    """专门画固定难度基准（bench/dX.0_success），这是跨实验最可比的指标"""
    loaded = []
    for name in run_names:
        d = find_run_dir(name)
        if d is None:
            continue
        loaded.append((name, load_scalars(d)))

    diffs = ["0.3", "0.5", "0.7", "1.0"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.ravel()

    for ax, d in zip(axes, diffs):
        tag = f"bench/d{d}_success"
        any_data = False
        for i, (name, ea) in enumerate(loaded):
            if tag not in ea.Tags()["scalars"]:
                continue
            ev = ea.Scalars(tag)
            steps = np.array([e.step for e in ev], dtype=float)
            vals = np.array([e.value for e in ev], dtype=float) * 100
            ax.plot(steps, vals, color=COLORS[i % len(COLORS)], lw=1.8,
                    marker="o", ms=3.5, label=name)
            any_data = True
        if not any_data:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        ax.set_title(f"固定难度基准 d={d}", fontsize=11)
        ax.set_ylabel("成功率 (%)", fontsize=10)
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[2:]:
        ax.set_xlabel("训练步数", fontsize=10)
    fig.suptitle("固定难度基准对比（跨实验最可比）", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def print_table(run_names):
    """终端打印关键指标的首末值对照"""
    loaded = []
    for name in run_names:
        d = find_run_dir(name)
        if d is None:
            continue
        loaded.append((name, load_scalars(d)))

    print("=" * 82)
    print("  训练曲线关键指标对照（首次 → 末次）")
    print("=" * 82)
    for tag, label, unit in KEY_METRICS:
        row = [f"  {label:<18}"]
        for name, ea in loaded:
            if tag not in ea.Tags()["scalars"]:
                row.append(f"{name}: —")
                continue
            ev = ea.Scalars(tag)
            scale = 100.0 if tag == "eval/success_rate" else 1.0
            suf = "%" if tag == "eval/success_rate" else ""
            row.append(f"{name}: {ev[0].value*scale:>8.2f}{suf} → "
                       f"{ev[-1].value*scale:>8.2f}{suf}")
        print("   |   ".join(row))

    print()
    print("=" * 82)
    print("  固定难度基准对照（末次值）")
    print("=" * 82)
    header = f"  {'':<18}" + "".join(f"{n:>20}" for n, _ in loaded)
    print(header)
    for d in ["0.3", "0.5", "0.7", "1.0"]:
        tag = f"bench/d{d}_success"
        cells = []
        for name, ea in loaded:
            if tag in ea.Tags()["scalars"]:
                ev = ea.Scalars(tag)
                cells.append(f"{ev[-1].value*100:>19.1f}%")
            else:
                cells.append(f"{'—':>20}")
        print(f"  {'d=' + d:<18}" + "".join(cells))
    print("=" * 82)


def main():
    ap = argparse.ArgumentParser(description="训练曲线对比")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="要对比的 run 名称（第一个作为基线）")
    ap.add_argument("--out", default=None, help="输出 PNG 路径")
    ap.add_argument("--no-bench", action="store_true", help="不画固定难度基准图")
    args = ap.parse_args()

    print_table(args.runs)

    out_dir = os.path.join(_RL_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = "_vs_".join(args.runs)

    out1 = args.out or os.path.join(out_dir, f"tb_compare_{tag}.png")
    plot_compare(args.runs, out1)
    print(f"\n✅ 训练曲线对比图: {out1}")

    if not args.no_bench:
        out2 = os.path.join(out_dir, f"tb_bench_{tag}.png")
        plot_bench(args.runs, out2)
        print(f"✅ 固定难度基准图: {out2}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
