#!/usr/bin/env python3
"""
eval.py —— 标准化策略评估脚本（量化层）
================================================================

对标业界标准：
  - NVIDIA Isaac Lab  : scripts/reinforcement_learning/sb3/eval.py
  - Unitree rl_lab    : scripts/rsl_rl/play.py (eval 模式)
  - AGILE             : scripts/eval.py (Parquet/JSON 轨迹 + 指标)

核心设计原则（来自 Robot Learning as an Empirical Science, arXiv 2409.09491）：
  1. 报告绝对次数，不只报百分比（35% 是 7/20 还是 70/200？）
  2. 固定 seed 集，保证不同 checkpoint 之间 A/B 可比
  3. 失败必须分类计数，不只报总成功率
  4. 记录 evaluation conditions，任何人可复现
  5. 轨迹数据落盘，可事后分析、可画图

输出：
  results/<run_tag>/eval_<难度>.json   每档难度的完整结果
  results/<run_tag>/summary.md         人类可读汇总报告

用法：
  # 评估单个 checkpoint，默认 100 回合 × 4 档难度
  python3 eval.py --model models/baseline_run/model.zip

  # 指定回合数与难度
  python3 eval.py --model models/baseline_run/model.zip --episodes 50 --diffs 0.3 0.7 1.0

  # 只跑 PD 基线（对照组）
  python3 eval.py --baseline pd --episodes 100
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
from collections import Counter, defaultdict

import numpy as np

_RL_DIR = _LEGACY_ORIG_DIR
sys.path.insert(0, os.path.join(_RL_DIR, "env"))

import quad_sim_fast_v2 as qs  # noqa: E402


# ================================================================ 失败分类
def classify_outcome(info_last, truncated, steps, episode_steps):
    """把一回合的结果归入标准失败类别。

    业界做法：失败必须分类（perception/control/dynamics/...），
    这里针对空中回正任务，分成 4 类互斥结局。
    """
    up_z = info_last["up_z"]
    z = info_last["z"]
    vz = info_last["vz"]
    omega = info_last["omega_norm"]

    # 判定「已改平」：机体 z 轴竖直分量 > 0.9（约 25° 内）
    upright = up_z > 0.9

    if truncated:
        # 活满了整个 episode（没摔），再看状态质量
        if upright and abs(vz) < 1.0 and omega < 2.0:
            return "success"
        if upright:
            return "timeout_unstable"      # 立起来了但还在晃/没稳住速度
        return "timeout_not_upright"       # 活满了但姿态一直没翻正

    # 非 truncated = 提前终止（摔地或出界）
    if z < 0.08:
        if upright:
            return "crash_upright"         # 姿态正立但没止住下落，砸地
        return "crash_not_upright"         # 姿态没翻正，直接砸下去
    # 出界（水平 >5m 或高度 >6m）
    return "out_of_bounds"


# ================================================================ 单回合
def run_episode(policy_fn, difficulty, seed, episode_s=4.0, domain_rand=1.0,
                history_len=0):
    """跑一个完整回合，返回（结局类别, 详情字典）。

    policy_fn(obs) -> action  ；由调用方决定是 RL 模型还是 PD 控制器
    """
    env = qs.ThrowRecoveryEnv(
        difficulty=difficulty, domain_rand=domain_rand,
        episode_s=episode_s, seed=seed, history_len=history_len,
    )
    obs, _ = env.reset()

    recovered_at = None       # 首次稳定改平的步数
    min_up_z = 1.0
    max_height_drop = 0.0
    init_z = None
    actions_used = []
    upz_trace = []

    done = False
    info = {}
    k = 0
    while not done:
        act = policy_fn(obs, env)
        obs, _r, term, trunc, info = env.step(act)
        k += 1

        up_z = info["up_z"]
        upz_trace.append(float(up_z))
        min_up_z = min(min_up_z, up_z)
        if init_z is None:
            init_z = info["z"]
        max_height_drop = max(max_height_drop, init_z - info["z"])
        actions_used.append(float(np.abs(np.asarray(act)).max()))

        # 首次"改平"时刻（up_z>0.95 且角速度<1）
        if recovered_at is None and up_z > 0.95 and info["omega_norm"] < 1.0:
            recovered_at = k / env.p.ctrl_hz

        if term or trunc:
            done = True

    outcome = classify_outcome(info, trunc, k, env.episode_steps)

    return outcome, {
        "seed": seed,
        "difficulty": difficulty,
        "outcome": outcome,
        "steps": k,
        "duration_s": round(k / env.p.ctrl_hz, 3),
        "recover_time_s": None if recovered_at is None else round(recovered_at, 3),
        "final_up_z": round(float(info["up_z"]), 4),
        "final_z": round(float(info["z"]), 3),
        "final_vz": round(float(info["vz"]), 3),
        "final_omega": round(float(info["omega_norm"]), 3),
        "min_up_z": round(float(min_up_z), 4),
        "height_drop_m": round(float(max_height_drop), 3),
        "action_abs_max_mean": round(float(np.mean(actions_used)), 4),
        # 简化轨迹（每 10 步采一个点，够画图又不臃肿）
        "upz_trace": [round(v, 3) for v in upz_trace[::10]],
    }


# ================================================================ Wilson 置信区间
def wilson_interval(successes, n, z=1.96):
    """Wilson score interval —— 小样本成功率的标准置信区间算法。

    为什么不用朴素 p±sqrt(p(1-p)/n)：当 p 接近 0 或 1 时会算出负数/超 1 的荒谬区间。
    Wilson 区间在极端比例和小样本下依然稳健，是评估报告的标准做法。
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ================================================================ 主评估
def evaluate(policy_fn, policy_name, diffs, episodes, base_seed, out_dir,
             history_len=0):
    """对每个难度档跑 episodes 个回合，返回完整结果字典。"""
    all_results = {}
    t_start = time.time()

    for d in diffs:
        # ★ 关键：固定 seed 集 —— 每档难度用 [base_seed, base_seed+episodes) 这组种子
        #   这样不同 checkpoint 之间是严格 A/B 可比的（同样的初始局面）
        seeds = [base_seed + i for i in range(episodes)]
        records = []
        for s in seeds:
            outcome, detail = run_episode(policy_fn, d, s, history_len=history_len)
            records.append(detail)

        counter = Counter(r["outcome"] for r in records)
        n = len(records)
        n_success = counter.get("success", 0)
        ci_lo, ci_hi = wilson_interval(n_success, n)

        rec_times = [r["recover_time_s"] for r in records
                     if r["recover_time_s"] is not None]

        all_results[d] = {
            "difficulty": d,
            "n_episodes": n,
            "n_success": n_success,
            "success_rate": round(n_success / n, 4),
            "success_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            "outcome_counts": dict(counter),
            "recover_time_mean_s": round(float(np.mean(rec_times)), 3) if rec_times else None,
            "recover_time_median_s": round(float(np.median(rec_times)), 3) if rec_times else None,
            "episodes": records,
        }

        print(f"  d={d:.1f}  {n_success:>3}/{n:<3} = {n_success/n*100:5.1f}%  "
              f"[95%CI {ci_lo*100:.0f}~{ci_hi*100:.0f}%]  "
              f"回正耗时 {all_results[d]['recover_time_median_s']}s")
        # 失败分类明细
        for k, v in counter.most_common():
            if k != "success":
                print(f"        └ {k:<22} {v:>3} ({v/n*100:.0f}%)")

    elapsed = time.time() - t_start

    # ---------- 落盘 ----------
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "policy": policy_name,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_conditions": {
            "episodes_per_difficulty": episodes,
            "difficulties": diffs,
            "seed_set": f"[{base_seed}, {base_seed + episodes})",
            "domain_randomization": 1.0,
            "episode_seconds": 4.0,
            "control_hz": 200,
            "action_mode": "deterministic",
            "history_stacking_frames": history_len,
            "success_criterion": "up_z>0.9 & |vz|<1.0 & |omega|<2.0 at episode end",
        },
        "elapsed_s": round(elapsed, 1),
        "results": {str(k): v for k, v in all_results.items()},
    }
    json_path = os.path.join(out_dir, "eval_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload, json_path


# ================================================================ 报告
def write_summary_md(payload, path):
    """生成人类可读的 Markdown 汇总报告"""
    lines = []
    lines.append(f"# 策略评估报告\n")
    lines.append(f"- **策略**：`{payload['policy']}`")
    lines.append(f"- **评估时间**：{payload['evaluated_at']}")
    lines.append(f"- **耗时**：{payload['elapsed_s']} 秒\n")

    cond = payload["evaluation_conditions"]
    lines.append("## 评估条件（可复现性记录）\n")
    lines.append(f"| 项目 | 值 |")
    lines.append(f"|---|---|")
    lines.append(f"| 每档回合数 | {cond['episodes_per_difficulty']} |")
    lines.append(f"| 难度档位 | {cond['difficulties']} |")
    lines.append(f"| 种子集（固定） | {cond['seed_set']} |")
    lines.append(f"| 域随机化强度 | {cond['domain_randomization']} |")
    lines.append(f"| 控制频率 | {cond['control_hz']} Hz |")
    lines.append(f"| 动作模式 | {cond['action_mode']} |")
    lines.append(f"| 成功判定 | {cond['success_criterion']} |\n")

    lines.append("## 成功率与置信区间\n")
    lines.append("| 难度 | 成功/总数 | 成功率 | 95% 置信区间 | 回正耗时(中位) |")
    lines.append("|---|---|---|---|---|")
    for d, r in payload["results"].items():
        ci = r["success_ci95"]
        lines.append(f"| {float(d):.1f} | {r['n_success']}/{r['n_episodes']} | "
                     f"{r['success_rate']*100:.1f}% | "
                     f"{ci[0]*100:.0f}% ~ {ci[1]*100:.0f}% | "
                     f"{r['recover_time_median_s']}s |")
    lines.append("")

    lines.append("## 失败分类明细\n")
    all_keys = set()
    for r in payload["results"].values():
        all_keys.update(r["outcome_counts"].keys())
    order = ["success", "crash_not_upright", "crash_upright",
             "timeout_not_upright", "timeout_unstable", "out_of_bounds"]
    cols = [k for k in order if k in all_keys] + sorted(all_keys - set(order))

    lines.append("| 难度 | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for d, r in payload["results"].items():
        cells = [str(r["outcome_counts"].get(c, 0)) for c in cols]
        lines.append(f"| {float(d):.1f} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 类别含义\n")
    meaning = {
        "success": "活满全程且状态达标（已改平 + 速度/角速度收敛）",
        "crash_not_upright": "**姿态没翻正**，直接砸地",
        "crash_upright": "**姿态正立但没止住下落**，砸地",
        "timeout_not_upright": "活满全程但姿态一直没翻正",
        "timeout_unstable": "活满全程但状态不达标（在晃 / 速度太大）",
        "out_of_bounds": "飞出 5m 水平或 6m 高度边界",
    }
    for c in cols:
        lines.append(f"- `{c}` —— {meaning.get(c, '未知')}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ================================================================ 策略工厂
def make_rl_policy(model_path):
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")

    def fn(obs, env):
        act, _ = model.predict(obs, deterministic=True)
        return act
    return fn, f"RL:{os.path.basename(os.path.dirname(model_path))}/model.zip"


def make_pd_policy():
    def fn(obs, env):
        # PD 控制器的输出是 [0,1] 油门，环境要 [-1,1]，需反向映射
        u = qs.pd_controller(env.sim, env.p)
        return u * 2.0 - 1.0
    return fn, "PD_baseline"


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="标准化策略评估")
    ap.add_argument("--model", default=None, help="SB3 checkpoint 路径")
    ap.add_argument("--baseline", default=None, choices=["pd"],
                    help="跑传统基线控制器对照组")
    # 项目统一标准：1000 回合/档（约 ±3pp 的 95%CI 半宽，可区分 6pp 以上差异）
    ap.add_argument("--episodes", type=int, default=1000, help="每档难度回合数")
    ap.add_argument("--diffs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0],
                    help="难度档位（0~1）")
    ap.add_argument("--seed", type=int, default=20261029, help="种子集起点")
    ap.add_argument("--history", type=int, default=0,
                    help="History Stacking 帧数，必须与训练时一致（0=22维，3=43维）")
    ap.add_argument("--out", default=None, help="输出目录")
    args = ap.parse_args()

    if not args.model and not args.baseline:
        ap.error("必须指定 --model 或 --baseline pd")

    if args.baseline == "pd":
        policy_fn, name = make_pd_policy()
        default_out = os.path.join(_RL_DIR, "results", "pd_baseline")
    else:
        if not os.path.exists(args.model):
            print(f"❌ 模型不存在: {args.model}")
            return 1
        policy_fn, name = make_rl_policy(args.model)
        default_out = os.path.join(_RL_DIR, "results",
                                   os.path.basename(os.path.dirname(args.model)))

    out_dir = args.out or default_out

    print("=" * 70)
    print(f"  标准化策略评估 —— {name}")
    print("=" * 70)
    print(f"  回合数/档 : {args.episodes}")
    print(f"  难度档位  : {args.diffs}")
    print(f"  固定种子集: [{args.seed}, {args.seed + args.episodes})")
    print(f"  成功判定  : up_z>0.9 & |vz|<1.0 & |omega|<2.0")
    print("=" * 70)
    print()

    payload, json_path = evaluate(policy_fn, name, args.diffs,
                                  args.episodes, args.seed, out_dir,
                                  history_len=args.history)

    md_path = os.path.join(out_dir, "summary.md")
    write_summary_md(payload, md_path)

    print()
    print("=" * 70)
    print(f"  ✅ 评估完成，耗时 {payload['elapsed_s']}s")
    print(f"  原始数据 : {json_path}")
    print(f"  汇总报告 : {md_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
