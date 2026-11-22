#!/usr/bin/env python3
"""
eval_hierarchical.py —— 分层策略联合评估（分阶段方案的"期末考试"）
================================================================

背景：
  分阶段训练产出两个专用策略：
    - Phase 1（翻转）：只负责 up_z < 0.7 区间，把机身从倒扣翻正
    - Phase 2（稳定）：只负责 up_z >= 0.7 区间，稳住姿态与速度
  部署时两个模型都烧进 ESP32，飞控代码按当前姿态切换调用。
  本脚本在【端到端环境】里模拟这个切换逻辑，评估整体回正能力，
  与端到端单模型（E-A2）严格对照。

切换逻辑（带滞回，防止阈值附近抖动）：
  - 初始：up_z < 0.7 用 P1，否则用 P2
  - P1 → P2：up_z > 0.7 时切换
  - P2 → P1：up_z < 0.5 才切回（0.5~0.7 之间保持当前）

成功判据：与 eval.py 完全一致（端到端严格判据），可直接用 report.py 对比。

额外诊断：记录每次交接瞬间的角速度分布，
  用于验证「分布偏移」风险 —— Phase 2 训练时初始角速度只有 0.5~4 rad/s，
  如果交接时实际角速度远超此范围，Phase 2 会没见过这种状态。

用法：
  python3 eval_hierarchical.py \
    --phase1 models/p1_recovery/model.zip \
    --phase2 models/p2_stabilize/model.zip \
    --episodes 1000
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

_RL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_RL_DIR, "env"))

import quad_sim_fast_v2 as qs  # noqa: E402
from eval import classify_outcome, wilson_interval  # noqa: E402

SWITCH_TO_P2 = 0.7   # up_z 超过此值 且 ω 达标：P1 → P2
# v16 修正：回切阈值从 0.5 收紧到 0.2，并加"单回合最多回切 2 次"的限流。
#   原因：P2 不稳时 up_z 会在 0.65~0.95 间振荡，若阈值 0.5 太松，
#   会触发 P1↔P2 反复横跳；每次横跳 P1 都在已有高角速度上猛翻，
#   角速度正反馈累积到 784 rad/s（实测）→ 彻底失控。
SWITCH_BACK_P1 = 0.2  # 只有真的翻倒（up_z<0.2）才切回 P1
MAX_SWITCH_BACK = 2   # 单回合最多回切次数，超出则锁定在 P2
# v14：交接还要看角速度。v13 诊断：up_z>0.7 瞬间切换时 ω 中位 8.1~14.9 rad/s，
# 超出 P2 训练分布（≤6），P2 接不稳。改为"翻正且基本刹住才交接"，
# 让 P1 把残余旋转收一收再交出去（v13b 的 P1 正是这么训练的：
# 它的 env 在 up_z>0.7 且 ω≥4 时不终止、持续重罚，所以 P1 会过界后继续刹车）。
# v15b：阈值对齐 P1 训练时的终止条件（up_z>0.7 且 ω<4 才交接）。
#   实测交接 ω 中位 5.7~5.8 说明 6.0 的门槛"放手太早"——P1 还在它的
#   刹车惩罚区（4≤ω<6 时 env 不终止、持续重罚）就被夺走了控制权。
HANDOFF_OMEGA_MAX = 1e9   # v16：P1 纯翻转，交接不看 ω


class HierarchicalPolicy:
    """按姿态切换的双策略控制器（模拟 ESP32 部署逻辑）"""

    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
        self.reset_stats()

    def reset_stats(self):
        self.current = None          # 当前使用的阶段 1/2
        self.handoff_upz = []        # 每次 P1→P2 交接瞬间的 up_z
        self.handoff_omega = []      # 交接瞬间的角速度（分布偏移诊断）
        self.handoff_vz = []         # 交接瞬间的垂直速度
        self.p1_steps = 0
        self.p2_steps = 0
        self.n_handoffs = 0
        self.n_switchback = 0   # 回切次数（限流用）

    def __call__(self, obs, env):
        q = env.sim.q
        up_z = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        omega = float(np.linalg.norm(env.sim.w))

        # 滞回切换逻辑（v14：P1→P2 需同时满足翻正 + 角速度达标）
        if self.current is None:
            self.current = 1 if up_z < SWITCH_TO_P2 else 2
        elif self.current == 1 and up_z > SWITCH_TO_P2 and omega <= HANDOFF_OMEGA_MAX:
            # P1 已翻正且把旋转刹到 P2 的经验范围内，才交接
            self.current = 2
            self.n_handoffs += 1
            self.handoff_upz.append(float(up_z))
            self.handoff_omega.append(omega)
            self.handoff_vz.append(float(env.sim.vel[2]))
        elif (self.current == 2 and up_z < SWITCH_BACK_P1
              and self.n_switchback < MAX_SWITCH_BACK):
            self.current = 1
            self.n_switchback += 1

        model = self.m1 if self.current == 1 else self.m2
        if self.current == 1:
            self.p1_steps += 1
        else:
            self.p2_steps += 1
        act, _ = model.predict(obs, deterministic=True)
        return act


def run_episode(hp, difficulty, seed, episode_s=4.0):
    env = qs.ThrowRecoveryEnv(difficulty=difficulty, domain_rand=1.0,
                              episode_s=episode_s, seed=seed)
    obs, _ = env.reset()
    hp.reset_stats()

    done = False
    info = {}
    k = 0
    while not done:
        act = hp(obs, env)
        obs, _r, term, trunc, info = env.step(act)
        k += 1
        if term or trunc:
            done = True

    outcome = classify_outcome(info, trunc, k, env.episode_steps)
    return outcome, {
        "seed": seed,
        "outcome": outcome,
        "steps": k,
        "n_handoffs": hp.n_handoffs,
        "p1_steps": hp.p1_steps,
        "p2_steps": hp.p2_steps,
        "handoff_omega": [round(x, 2) for x in hp.handoff_omega],
        "handoff_vz": [round(x, 2) for x in hp.handoff_vz],
        "final_up_z": round(float(info["up_z"]), 3),
        "height_drop_m": round(max(0.0, 2.0 - float(info["z"])), 3),
    }


def main():
    ap = argparse.ArgumentParser(description="分层策略联合评估")
    ap.add_argument("--phase1", required=True, help="Phase 1 翻转模型")
    ap.add_argument("--phase2", required=True, help="Phase 2 稳定模型")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--diffs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--seed", type=int, default=20261029,
                    help="种子集起点（与 eval.py 一致才能对照）")
    ap.add_argument("--out", default=os.path.join(_RL_DIR, "results", "hierarchical"))
    args = ap.parse_args()

    from stable_baselines3 import PPO
    m1 = PPO.load(args.phase1, device="cpu")
    m2 = PPO.load(args.phase2, device="cpu")
    hp = HierarchicalPolicy(m1, m2)

    print("=" * 74)
    print("  分层策略联合评估（端到端环境 + 滞回切换 + 严格判据）")
    print("=" * 74)
    print(f"  Phase1: {args.phase1}")
    print(f"  Phase2: {args.phase2}")
    print(f"  切换逻辑: P1→P2 当 up_z>{SWITCH_TO_P2}；P2→P1 当 up_z<{SWITCH_BACK_P1}")
    print(f"  成功判据: 与 eval.py 相同（活满全程+up_z>0.9+|vz|<1.0+|ω|<2.0）")
    print(f"  种子集: [{args.seed}, {args.seed + args.episodes})")
    print("=" * 74)

    all_results = {}
    t0 = time.time()

    for d in args.diffs:
        records = []
        for i in range(args.episodes):
            oc, detail = run_episode(hp, d, args.seed + i)
            records.append(detail)

        n = len(records)
        counter = Counter(r["outcome"] for r in records)
        n_ok = counter.get("success", 0)
        ci = wilson_interval(n_ok, n)

        # 交接诊断
        omegas = [w for r in records for w in r["handoff_omega"]]
        no_handoff = sum(1 for r in records if r["n_handoffs"] == 0)

        print(f"\n  d={d:.1f}: {n_ok}/{n} = {n_ok/n*100:.1f}%  "
              f"[95%CI {ci[0]*100:.0f}~{ci[1]*100:.0f}%]")
        for k, v in counter.most_common():
            if k != "success":
                print(f"      └ {k:<22} {v:>4} ({v/n*100:.0f}%)")
        if omegas:
            print(f"      交接次数: 平均 {np.mean([r['n_handoffs'] for r in records]):.2f}/回合, "
                  f"从未交接 {no_handoff} 回合")
            print(f"      交接时角速度: 中位 {np.median(omegas):.1f} rad/s, "
                  f"P90 {np.percentile(omegas, 90):.1f}, max {max(omegas):.1f}")
            vzs = [v for r in records for v in r["handoff_vz"]]
            if vzs:
                print(f"      交接时垂直速度: 中位 {np.median(vzs):.1f} m/s, "
                      f"P90 {np.percentile(vzs, 90):.1f}")

        all_results[str(d)] = {
            "difficulty": d,
            "n_episodes": n,
            "n_success": n_ok,
            "success_rate": round(n_ok / n, 4),
            "success_ci95": [round(ci[0], 4), round(ci[1], 4)],
            "outcome_counts": dict(counter),
            "handoff_omega_median": round(float(np.median(omegas)), 2) if omegas else None,
            "handoff_omega_p90": round(float(np.percentile(omegas, 90)), 2) if omegas else None,
            "episodes": records,
        }

    elapsed = time.time() - t0

    # ---------- 落盘（兼容 report.py 格式） ----------
    os.makedirs(args.out, exist_ok=True)
    payload = {
        "policy": f"Hierarchical(P1={os.path.basename(os.path.dirname(args.phase1))},"
                  f"P2={os.path.basename(os.path.dirname(args.phase2))})",
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_conditions": {
            "episodes_per_difficulty": args.episodes,
            "difficulties": args.diffs,
            "seed_set": f"[{args.seed}, {args.seed + args.episodes})",
            "domain_randomization": 1.0,
            "episode_seconds": 4.0,
            "control_hz": 200,
            "action_mode": "deterministic",
            "success_criterion": "up_z>0.9 & |vz|<1.0 & |omega|<2.0 at episode end",
            "switching": f"P1->P2 up_z>{SWITCH_TO_P2}, P2->P1 up_z<{SWITCH_BACK_P1}",
        },
        "elapsed_s": round(elapsed, 1),
        "results": all_results,
    }
    json_path = os.path.join(args.out, "eval_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"\n{'='*74}")
    print(f"  ✅ 完成，耗时 {elapsed:.0f}s")
    print(f"  数据: {json_path}")
    print(f"  对比: python3 report.py --runs v6b_EA2 hierarchical")
    print(f"{'='*74}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
