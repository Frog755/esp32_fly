#!/usr/bin/env python3
"""
play.py —— 策略定性回放脚本（可视化层）
================================================================

对标业界标准：
  - NVIDIA Isaac Lab  : scripts/reinforcement_learning/sb3/play.py
  - MuJoCo UniLab     : uv run eval --render-mode {interactive|record|none}

用途：eval.py 告诉你「失败多少」，play.py 告诉你「失败长什么样」。
     这是诊断奖励设计的最后一步 —— 看到具体的姿态曲线和电机输出，
     你才能判断该改哪个奖励项。

三种模式（对齐 UniLab 的 render-mode）：
  --mode text    终端逐帧打印状态表（无需显示环境，SSH 也能用）
  --mode curve   生成 PNG 曲线图（姿态/高度/角速度/电机输出 四联图）
  --mode video   生成 MP4 动画（需要 ffmpeg，已安装）

用法：
  # 看第 5 号种子的失败回放（终端打印）
  python3 play.py --model models/baseline_run/model.zip --diff 0.3 --seed 20261034 --mode text

  # 生成曲线图（最推荐 —— 一图看清死在哪）
  python3 play.py --model models/baseline_run/model.zip --diff 0.3 --seed 20261034 --mode curve

  # 生成视频
  python3 play.py --model models/baseline_run/model.zip --diff 0.3 --seed 20261034 --mode video
"""

import argparse
import os
import sys

import numpy as np

_RL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_RL_DIR, "env"))

import quad_sim_fast_v2 as qs  # noqa: E402

# 中文字体（Noto CJK 已确认可用）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC",
                                   "Droid Sans Fallback", "AR PL UMing CN",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ================================================================ 跑回合并记录
def rollout(policy_fn, difficulty, seed, episode_s=4.0):
    """跑一个回合，逐帧记录所有关键量，供可视化使用。"""
    env = qs.ThrowRecoveryEnv(difficulty=difficulty, domain_rand=1.0,
                              episode_s=episode_s, seed=seed)
    obs, _ = env.reset()

    rec = {
        "t": [], "up_z": [], "z": [], "vz": [], "vxy": [], "omega": [],
        "throttle": [], "action": [], "reward": [],
    }
    done = False
    info = {}
    k = 0
    while not done:
        act = policy_fn(obs, env)
        obs, r, term, trunc, info = env.step(act)
        k += 1

        t = k / env.p.ctrl_hz
        rec["t"].append(t)
        rec["up_z"].append(float(info["up_z"]))
        rec["z"].append(float(info["z"]))
        rec["vz"].append(float(info["vz"]))
        rec["vxy"].append(float(np.linalg.norm(env.sim.vel[:2])))
        rec["omega"].append(float(info["omega_norm"]))
        # 动作 [-1,1] -> 油门 [0,1]
        u = (np.asarray(act, dtype=float) + 1.0) / 2.0
        rec["throttle"].append([float(x) for x in u])
        rec["action"].append([float(x) for x in np.asarray(act, dtype=float)])
        rec["reward"].append(float(r))

        if term or trunc:
            done = True

    for key in rec:
        rec[key] = np.array(rec[key])
    rec["seed"] = seed
    rec["difficulty"] = difficulty
    rec["terminated"] = bool(term)
    rec["truncated"] = bool(trunc)
    rec["info_final"] = info
    rec["env"] = env
    return rec


# ================================================================ 文本模式
def print_text(rec):
    info = rec["info_final"]
    up_z = info["up_z"]
    if rec["truncated"] and up_z > 0.9:
        verdict = "✅ 成功（活满全程且正立）"
    elif info["z"] < 0.08 and up_z > 0.9:
        verdict = "❌ 正立但没刹住高度，砸地"
    elif info["z"] < 0.08:
        verdict = "❌ 姿态没翻正，砸地"
    else:
        verdict = "⚠️ 超时/出界"

    print(f"\n{'='*78}")
    print(f"  回放：d={rec['difficulty']:.2f}  seed={rec['seed']}  结局：{verdict}")
    print(f"  总时长 {rec['t'][-1]:.2f}s，共 {len(rec['t'])} 步")
    print(f"{'='*78}")
    print(f"{'t(s)':>6} {'up_z':>7} {'高度':>7} {'vz':>7} {'|ω|':>7} "
          f"{'油门M1':>7} {'M2':>7} {'M3':>7} {'M4':>7}")
    print("-" * 78)

    n = len(rec["t"])
    step = max(1, n // 40)   # 最多打 40 行
    for i in range(0, n, step):
        th = rec["throttle"][i]
        mark = ""
        if rec["up_z"][i] > 0.9:
            mark += "★"
        if i == 0:
            mark += " (起始)"
        print(f"{rec['t'][i]:>6.2f} {rec['up_z'][i]:>7.3f} {rec['z'][i]:>7.2f} "
              f"{rec['vz'][i]:>7.2f} {rec['omega'][i]:>7.2f} "
              f"{th[0]:>7.3f} {th[1]:>7.3f} {th[2]:>7.3f} {th[3]:>7.3f} {mark}")
    # 最后一行必打
    i = n - 1
    th = rec["throttle"][i]
    print(f"{rec['t'][i]:>6.2f} {rec['up_z'][i]:>7.3f} {rec['z'][i]:>7.2f} "
          f"{rec['vz'][i]:>7.2f} {rec['omega'][i]:>7.2f} "
          f"{th[0]:>7.3f} {th[1]:>7.3f} {th[2]:>7.3f} {th[3]:>7.3f} (结束)")
    print("-" * 78)
    print(f"  终态：up_z={info['up_z']:.3f}  z={info['z']:.2f}m  "
          f"vz={info['vz']:.2f}  |ω|={info['omega_norm']:.2f}")
    print(f"  ★ 标记 = 该步姿态已改平（up_z>0.9）")


# ================================================================ 曲线模式
def plot_curves(rec, out_png):
    """四联图：姿态 / 高度+垂直速度 / 角速度 / 四电机油门。"""
    t = rec["t"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    # --- 1. 姿态 up_z ---
    ax = axes[0]
    ax.plot(t, rec["up_z"], color="#1f77b4", lw=2, label="up_z（机身z轴竖直分量）")
    ax.axhline(0.9, color="green", ls="--", lw=1, alpha=0.7, label="改平阈值 0.9")
    ax.axhline(0.0, color="gray", ls=":", lw=1, alpha=0.6, label="完全倒扣 0.0")
    ax.set_ylabel("up_z", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(f"姿态回正过程（d={rec['difficulty']:.2f}, seed={rec['seed']}）",
                 fontsize=12, fontweight="bold")

    # --- 2. 高度 + 垂直速度 ---
    ax = axes[1]
    ax.plot(t, rec["z"], color="#d62728", lw=2, label="高度 z (m)")
    ax.set_ylabel("高度 (m)", color="#d62728", fontsize=11)
    ax.tick_params(axis="y", labelcolor="#d62728")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, rec["vz"], color="#ff7f0e", lw=1.6, ls="--", label="垂直速度 vz (m/s)")
    ax2.set_ylabel("vz (m/s)", color="#ff7f0e", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=9)

    # --- 3. 角速度 ---
    ax = axes[2]
    ax.plot(t, rec["omega"], color="#9467bd", lw=2, label="|ω| 角速度模长 (rad/s)")
    ax.set_ylabel("|ω| (rad/s)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # --- 4. 四电机油门 ---
    ax = axes[3]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i in range(4):
        ax.plot(t, rec["throttle"][:, i], color=colors[i], lw=1.4,
                label=f"电机{i+1}")
    ax.axhline(0.707, color="black", ls=":", lw=1, alpha=0.6, label="悬停油门 0.707")
    ax.set_ylabel("油门 [0,1]", fontsize=11)
    ax.set_xlabel("时间 (s)", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9, ncol=5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ================================================================ 视频模式
def make_video(rec, out_mp4, fps=40, slowmo=1):
    """俯视 + 侧视双视角的简易动画（不依赖 3D 引擎，纯 matplotlib 绘制）。"""
    import imageio.v2 as imageio

    env = rec["env"]
    p = env.p
    t = rec["t"]
    n = len(t)

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(12, 6))
    frames = []

    # 回放需要重新积分姿态 —— 用记录下的 up_z 近似重建不可行，
    # 这里改为重跑一次环境，同步抓取四元数
    env2 = qs.ThrowRecoveryEnv(difficulty=rec["difficulty"], domain_rand=1.0,
                               episode_s=4.0, seed=rec["seed"])
    obs2, _ = env2.reset()
    quats = []
    positions = []
    done = False
    while not done:
        act = rec["action"][len(quats)] if len(quats) < n else np.zeros(4)
        obs2, _r, term, trunc, info = env2.step(act)
        quats.append(list(env2.sim.q))
        positions.append(list(env2.sim.pos))
        if term or trunc:
            done = True

    for i in range(0, min(n, len(quats)), max(1, int(rec["env"].p.ctrl_hz / fps))):
        q = quats[i]
        pos = positions[i]
        ax_top.clear(); ax_side.clear()

        # 机体坐标轴（3 条线）
        w, x, y, z = q
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])
        arm = p.wheelbase / 2 / np.sqrt(2)
        motors_body = np.array([[arm,-arm,0],[-arm,-arm,0],[-arm,arm,0],[arm,arm,0]])

        # 俯视图
        ax_top.set_xlim(-1.5, 1.5); ax_top.set_ylim(-1.5, 1.5)
        ax_top.set_aspect("equal"); ax_top.grid(alpha=0.3)
        ax_top.set_title(f"俯视  t={t[i]:.2f}s  up_z={rec['up_z'][i]:.2f}", fontsize=10)
        for mb in motors_body:
            mw = R @ mb + np.array([pos[0], pos[1], 0])
            ax_top.plot([pos[0], mw[0]], [pos[1], mw[1]], "k-", lw=2, alpha=0.6)
            ax_top.add_patch(plt.Circle((mw[0], mw[1]), 0.05,
                                        color="tab:blue", ec="k", zorder=3))
        ax_top.plot(pos[0], pos[1], "ko", ms=4)

        # 侧视图
        ax_side.set_xlim(-1.5, 1.5); ax_side.set_ylim(0, 3.0)
        ax_side.grid(alpha=0.3)
        ax_side.set_title(f"侧视  z={pos[2]:.2f}m  vz={rec['vz'][i]:.2f}m/s", fontsize=10)
        ax_side.axhline(0, color="brown", lw=2)
        for mb in motors_body:
            mw = R @ mb + np.array([0, 0, pos[2]])   # 用 x=0 平面投影
            ax_side.plot([0, mw[0]], [pos[2], mw[2]], "k-", lw=2, alpha=0.6)
        ax_side.plot(0, pos[2], "ko", ms=4)

        fig.suptitle(f"d={rec['difficulty']:.2f}  seed={rec['seed']}", fontsize=11)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(buf.copy())

    plt.close(fig)
    # 慢放：每帧重复 slowmo 次，等效把播放速度降为 1/slowmo
    if slowmo > 1:
        frames = [f for f in frames for _ in range(slowmo)]
    imageio.mimsave(out_mp4, frames, fps=fps)
    return out_mp4


# ================================================================ 策略工厂
def make_rl_policy(model_path):
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")

    def fn(obs, env):
        act, _ = model.predict(obs, deterministic=True)
        return act
    return fn


def make_pd_policy():
    def fn(obs, env):
        u = qs.pd_controller(env.sim, env.p)
        return u * 2.0 - 1.0
    return fn


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="策略定性回放")
    ap.add_argument("--model", default=None, help="SB3 checkpoint")
    ap.add_argument("--baseline", default=None, choices=["pd"])
    ap.add_argument("--diff", type=float, default=0.3, help="难度档位")
    ap.add_argument("--seed", type=int, default=20261029, help="随机种子")
    ap.add_argument("--mode", default="text", choices=["text", "curve", "video"],
                    help="text=终端表格 / curve=PNG曲线 / video=MP4动画")
    ap.add_argument("--out", default=None, help="输出文件路径")
    ap.add_argument("--slowmo", type=int, default=1,
                    help="慢放倍率（1=正常，4=4倍慢放，8=8倍慢放）")
    ap.add_argument("--trail", action="store_true",
                    help="视频模式：绘制运动轨迹残影，便于观察路径")
    ap.add_argument("--panorama", action="store_true",
                    help="panorama 模式：跑多个种子，输出一张对比总图")
    args = ap.parse_args()

    if not args.model and not args.baseline:
        ap.error("必须指定 --model 或 --baseline pd")

    if args.baseline == "pd":
        policy_fn = make_pd_policy()
        tag = "pd_baseline"
    else:
        policy_fn = make_rl_policy(args.model)
        tag = os.path.basename(os.path.dirname(args.model))

    out_base = os.path.join(_RL_DIR, "results", tag, "play")
    os.makedirs(out_base, exist_ok=True)

    # ---------- panorama：多样本对比总图 ----------
    if args.panorama:
        from matplotlib.gridspec import GridSpec
        n_seeds = 6
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
        for idx in range(n_seeds):
            s = args.seed + idx
            rec = rollout(policy_fn, args.diff, s)
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            ax.plot(rec["t"], rec["up_z"], lw=1.8, label="up_z")
            ax.plot(rec["t"], rec["z"] / 3.0, lw=1.4, ls="--", label="高度/3")
            ax.axhline(0.9, color="green", ls=":", lw=1)
            ax.set_ylim(-1.1, 1.1); ax.grid(alpha=0.3)
            ok = rec["truncated"] and rec["info_final"]["up_z"] > 0.9
            ax.set_title(f"seed={s}  {'✅成功' if ok else '❌失败'}",
                         fontsize=10, color="green" if ok else "crimson")
            ax.legend(fontsize=8, loc="lower right")
        fig.suptitle(f"{tag}  多种子姿态对比  d={args.diff:.2f}", fontsize=13)
        out = args.out or os.path.join(out_base, f"panorama_d{args.diff:.2f}.png")
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 对比总图: {out}")
        return 0

    rec = rollout(policy_fn, args.diff, args.seed)

    if args.mode == "text":
        print_text(rec)
        # 顺带存一份 JSON 便于脚本处理
        import json
        js = args.out or os.path.join(
            out_base, f"d{args.diff:.2f}_s{args.seed}.json")
        slim = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in rec.items() if k != "env"}
        with open(js, "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False, indent=2)
        print(f"\n  轨迹数据: {js}")

    elif args.mode == "curve":
        out = args.out or os.path.join(
            out_base, f"d{args.diff:.2f}_s{args.seed}.png")
        plot_curves(rec, out)
        print(f"✅ 曲线图: {out}")
        # 同时打印一行摘要
        info = rec["info_final"]
        ok = rec["truncated"] and info["up_z"] > 0.9
        print(f"   结局：{'✅成功' if ok else '❌失败'}  "
              f"终态 up_z={info['up_z']:.2f} z={info['z']:.2f} "
              f"vz={info['vz']:.2f} |ω|={info['omega_norm']:.2f}")

    elif args.mode == "video":
        out = args.out or os.path.join(
            out_base, f"d{args.diff:.2f}_s{args.seed}.mp4")
        print(f"  正在生成动画（慢放 {args.slowmo}x）...")
        make_video(rec, out, slowmo=args.slowmo)
        print(f"✅ 视频: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
