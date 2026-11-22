#!/usr/bin/env python3
"""
把 TensorBoard 的 event 文件直接读出来，画成静态图片。

用途：TensorBoard 网页打不开时的备用方案，也方便贴到报告/聊天里。

用法：
  python3 export_tb_plots.py                    # 画全部关键曲线
  python3 export_tb_plots.py --tags ep_rew_mean ep_len_mean approx_kl
"""
# --- [Stage 20 legacy 迁移引导] 本文件原位于 RL/scripts/，现移至 RL/legacy/rl_vec/ ---
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
import os
import sys

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = os.path.dirname(_LEGACY_ORIG_DIR)
_OUT = os.path.join(_ROOT, "output")

# 只显示这几个有意义的 run（过滤掉中断的）
# 自动发现 run：目录名形如 E1_v1_s0_1，直接去掉末尾的 _<序号> 作为图例
RUN_FILTER = {}

# 实验含义说明（只用于图例的可读性，不影响数据）
LABELS = {
    "E1_v1_s0": "E1 v1+curriculum",
    "E2_v1hard_s1": "E2 v1+hard(d=1.0)",
    "E3_v2_s0": "E3 v2+curriculum",
}


def label_for(run_rel):
    """从 run 相对路径推出图例名。"""
    base = os.path.basename(run_rel.rstrip("/"))
    key = re.sub(r"_\d+$", "", base)      # 去掉 sb3 的 _1 后缀
    return LABELS.get(key, base)


def load_all(logdir):
    """读所有 run 的所有标量，返回 {run: DataFrame}"""
    from tbparse import SummaryReader
    data = {}
    for root, dirs, files in os.walk(logdir):
        if not any(f.startswith("events.out.tfevents") for f in files):
            continue
        rel = os.path.relpath(root, logdir)
        # 只认 output/run_E*/tb/<run>/ 这一层，忽略杂项
        if not re.match(r"run_E[0-9]+/tb/[^/]+$", rel):
            continue
        RUN_FILTER[rel] = label_for(rel)
        try:
            reader = SummaryReader(root)
            df = reader.scalars
            if not df.empty:
                data[rel] = df
        except Exception as e:
            print(f"  [warn] {rel}: {e}")
    return data


def plot_tags(data, tags, outdir):
    made = []
    for tag in tags:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plotted = False
        for run, df in data.items():
            sub = df[df["tag"] == tag]
            if sub.empty:
                continue
            # 同一 run 可能有多个 event，按 step 排序去重
            sub = sub.sort_values("step").drop_duplicates("step", keep="last")
            ax.plot(sub["step"], sub["value"], label=RUN_FILTER.get(run, run),
                    linewidth=1.4, alpha=0.85)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("training steps")
        ax.set_ylabel(tag)
        ax.set_title(tag)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = tag.replace("/", "_").replace("\\", "_")
        fp = os.path.join(outdir, f"tb_{safe}.png")
        fig.savefig(fp, dpi=120)
        plt.close(fig)
        made.append(fp)
        print(f"  ✓ {fp}")
    return made


def plot_combo(data, outdir):
    """画一张 2x2 汇总图：ep_rew / ep_len / approx_kl / value_loss"""
    wanted = [("rollout/ep_rew_mean", "avg episode reward"),
              ("rollout/ep_len_mean", "avg episode length (steps)"),
              ("train/approx_kl", "approx KL (policy change)"),
              ("train/entropy_loss", "entropy loss (exploration)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (tag, title) in zip(axes.flat, wanted):
        plotted = False
        for run, df in data.items():
            sub = df[df["tag"] == tag].sort_values("step").drop_duplicates("step", keep="last")
            if sub.empty:
                continue
            ax.plot(sub["step"], sub["value"], label=RUN_FILTER.get(run, run),
                    linewidth=1.3, alpha=0.85)
            plotted = True
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if not plotted:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
        else:
            ax.legend(fontsize=7)
    fig.suptitle("PPO training: E1(v1 curriculum) vs E2(v1 hard) vs E3(v2 curriculum)",
                 fontsize=12)
    fig.tight_layout()
    fp = os.path.join(outdir, "tb_summary_2x2.png")
    fig.savefig(fp, dpi=125)
    plt.close(fig)
    print(f"  ✓ {fp}")
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=_OUT)
    ap.add_argument("--outdir", default=_OUT)
    ap.add_argument("--tags", nargs="*", default=None)
    args = ap.parse_args()

    print(f"读取 {args.logdir} ...")
    data = load_all(args.logdir)
    if not data:
        print("没有读到任何数据")
        return 1
    print(f"读到 {len(data)} 个 run:")
    for r, df in data.items():
        tags = df["tag"].nunique()
        print(f"  {r}: {len(df)} 条记录, {tags} 个 tag")

    os.makedirs(args.outdir, exist_ok=True)
    tags = args.tags or ["rollout/ep_rew_mean", "rollout/ep_len_mean",
                         "train/approx_kl", "train/entropy_loss",
                         "train/value_loss", "train/policy_gradient_loss",
                         "eval/success_rate", "eval/recovery_time_s",
                         "bench/d0.3_success", "bench/d0.5_success",
                         "bench/d0.7_success", "bench/d1.0_success"]
    print("\n画单曲线:")
    plot_tags(data, tags, args.outdir)
    print("\n画汇总图:")
    plot_combo(data, args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
