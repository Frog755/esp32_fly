#!/usr/bin/env python3
"""
PPO 训练脚本：70g 抛飞自稳四旋翼

两条训练线（--mode）：
  curriculum  课程学习：difficulty 从 start 起步，成功率 > promote_at 就升档，直到 1.0
  hard        极限开局：固定 difficulty（默认 1.0），用来验证「一上来全是 180°+大角速度」可行性

核心设计说明：
  - 用 quad_sim_fixed（修了 randomize 丢字段的 bug）
  - domain_rand=1.0，与作者建议一致
  - 评价标准：与 PD 基线同一套判据，便于横向对比
  - PPO + 64x64 MLP（先跑基线），支持 --net gru 用 GRU

用法：
  # A 线：课程学习
  python3 train_ppo.py --mode curriculum --steps 2000000 --out ../output/run_curriculum

  # B 线：极限开局
  python3 train_ppo.py --mode hard --difficulty 1.0 --steps 2000000 --out ../output/run_hard

  # 评估已有模型
  python3 train_ppo.py --eval ../output/run_curriculum/model.zip --episodes 100
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

import numpy as np

_RL_DIR = _LEGACY_ORIG_DIR
_SIM_DIR = os.path.join(_RL_DIR, "env")
sys.path.insert(0, _SIM_DIR)
os.environ.setdefault("QUAD_SIM_MODULE", "quad_sim_fast_v2")
_SIM_MOD = os.environ.get("QUAD_SIM_MODULE", "quad_sim_fast_v2")
import importlib
qs = importlib.import_module(_SIM_MOD)  # noqa: E402

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.policies import ActorCriticPolicy  # noqa: E402
import torch.nn as nn  # noqa: E402


# ============================================================ 输出压缩策略
class TanhActorCriticPolicy(ActorCriticPolicy):
    """在 action_net 之后追加 Tanh，确保网络输出严格落在 [-1, 1]。

    背景（v10 实验动机）：
      实测 E-A2 模型的原始输出范围是 [-9.73, +8.41]，
      1000 个状态采样中仅 30.9% 落在 [-1,1] 内，**69.1% 被 env 的 np.clip 截断**。
      原因是 action_net 是最后一层且没有激活函数，
      其权重绝对值之和 sum|W| ≈ 9，理论上限就是 ±9。
      后果：策略学成了「二值开关」而非「连续调节」，丧失精细控制能力。

    修复：给 action_net 套一层 Tanh（无参数，参数量不变）。
      输出天然 ∈ (-1,1)，不再依赖 clip 兜底，梯度也更平滑。
      对应 RLtools/Raptor 的做法（策略输出层即为有界输出）。
    """

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())


# ============================================================ 可选难度环境
class DifficultyEnv(gym.Env):
    """包一层，让 difficulty 可以在训练中途动态变化（课程学习用）。"""

    metadata = {"render_modes": []}

    def __init__(self, difficulty=1.0, domain_rand=1.0, seed=None,
                 history_len=0, phase=0, residual_pd=False):
        super().__init__()
        self.difficulty = difficulty
        self.phase = phase
        self.inner = qs.ThrowRecoveryEnv(
            difficulty=difficulty, domain_rand=domain_rand, seed=seed,
            history_len=history_len, phase=phase, residual_pd=residual_pd)
        obs_dim = self.inner.obs_dim   # 22 (无堆叠) 或 22+7*history_len
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    def set_difficulty(self, d):
        self.difficulty = float(np.clip(d, 0.0, 1.0))
        self.inner.difficulty = self.difficulty

    def reset(self, seed=None, options=None):
        self.inner.difficulty = self.difficulty
        return self.inner.reset(seed=seed, options=options)

    def step(self, action):
        return self.inner.step(action)


# ============================================================ 课程学习回调
class CurriculumCallback(BaseCallback):
    """每隔 eval_every 步跑一批评估，成功率 > promote_at 就把 difficulty 升一档。"""

    def __init__(self, n_eval=30, every=50000, promote_at=0.7, step_up=0.2,
                 max_diff=1.0, verbose=1, save_dir=None,
                 bench_diffs=(0.3, 0.5, 0.7, 1.0), bench_n=20,
                 smooth_window=3, max_std=0.20, history_len=0, phase=0,
                 residual_pd=False):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.history_len = history_len
        self.phase = phase
        self.residual_pd = bool(residual_pd)
        self.n_eval, self.every = n_eval, every
        self.promote_at, self.step_up, self.max_diff = promote_at, step_up, max_diff
        self.history = []
        self._last_eval = None
        self.bench_diffs = bench_diffs
        self.bench_n = bench_n
        self.smooth_window = smooth_window
        self.max_std = max_std
        self._bench_history = []
        self._tb_writer = None

    def attach_writer(self, tb_parent, run_name):
        """记录目标目录；真正的 SummaryWriter 在第一次评估时懒创建，
        这样 sb3 已经建好了 <tb_parent>/<run_name>/，不会撞名加 _1 后缀。"""
        self._tb_parent = tb_parent
        self._run_name = run_name

    def close(self):
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()

    def _on_step(self) -> bool:
        # 用 >= 而不是 ==，避免训练步长不整除 every 时永远不触发
        if self._last_eval is not None and self.num_timesteps - self._last_eval < self.every:
            return True
        if self.num_timesteps < self.every:
            return True
        self._last_eval = self.num_timesteps
        venv = self.training_env
        d = venv.get_attr("difficulty")[0]
        rate, t_rec = evaluate_policy(self.model, d, n=self.n_eval,
                                      history_len=self.history_len,
                                      phase=self.phase,
                                      residual_pd=self.residual_pd)
        self.history.append(dict(steps=self.num_timesteps, difficulty=float(d),
                                 success=float(rate), t_rec=t_rec))
        # ===== 关键：评估指标写进独立的 TensorBoard 目录 =====
        # 不依赖 sb3 内部 logger（它在 on_step 里的 dump 行为不稳定），
        # 自己开一个 SummaryWriter 写到 <tb>/eval_metrics/，TensorBoard 会当成独立 run 显示。
        if self._tb_writer is None and getattr(self, "_tb_parent", None):
            from torch.utils.tensorboard import SummaryWriter
            import glob as _glob
            # sb3 的命名固定是 <tb_log_name>_<n>，n 从 1 开始（见 configure_logger）。
            # 它已经建好目录了，我们直接把 eval 写进去 —— 一次实验就是一个 run。
            from stable_baselines3.common.utils import get_latest_run_id
            rid = get_latest_run_id(self._tb_parent, self._run_name)
            put = os.path.join(self._tb_parent, f"{self._run_name}_{rid if rid > 0 else 1}")
            os.makedirs(put, exist_ok=True)
            self._tb_writer = SummaryWriter(put)
        if self._tb_writer is not None:
            w = self._tb_writer
            st = self.num_timesteps
            w.add_scalar("eval/success_rate", float(rate), st)
            w.add_scalar("eval/difficulty", float(d), st)
            if t_rec is not None:
                w.add_scalar("eval/recovery_time_s", float(t_rec), st)
            w.add_scalar("eval/n_episodes", int(self.n_eval), st)
            # 固定难度基准：每次评估都用同一组难度，跨实验可对齐
            for _d in self.bench_diffs:
                _r, _t = evaluate_policy(self.model, _d, n=self.bench_n,
                                         history_len=self.history_len,
                                         phase=self.phase,
                                         residual_pd=self.residual_pd)
                w.add_scalar(f"bench/d{_d:.1f}_success", float(_r), st)
                if _t is not None:
                    w.add_scalar(f"bench/d{_d:.1f}_t_rec", float(_t), st)
                self._bench_history.append(dict(steps=st, difficulty=_d,
                                                success=_r, t_rec=_t))
            w.flush()
        # 每个 eval 周期都存一次档，避免训练被中断时丢失全部进度
        if self.save_dir:
            self.model.save(os.path.join(self.save_dir, "model.zip"))
        if self.verbose:
            print(f"  [eval] steps={self.num_timesteps:>8}  diff={d:.2f}  "
                  f"成功率={rate*100:.0f}%  转正={t_rec if t_rec is None else round(t_rec,2)}s")
        # 升档判据：用最近 3 次评估的滑动平均 + 稳定性约束，
        # 避免单次小样本评估的噪声导致误升档或永久卡档
        promote = False
        recent = [h["success"] for h in self.history[-self.smooth_window:]]
        if len(recent) >= self.smooth_window:
            mean_s = float(np.mean(recent))
            std_s = float(np.std(recent))
            promote = (mean_s >= self.promote_at) and (std_s <= self.max_std)
        elif rate >= self.promote_at:
            promote = True     # 历史不够时退回单次判据

        if promote and d < self.max_diff:
            nd = min(self.max_diff, d + self.step_up)
            venv.set_attr("difficulty", nd)
            if self.verbose:
                r = self.history[-self.smooth_window:]
                print(f"  ↑ 升档 difficulty {d:.2f} → {nd:.2f}  "
                      f"(近{len(r)}次均值 {np.mean([x['success'] for x in r])*100:.0f}%, "
                      f"std {np.std([x['success'] for x in r])*100:.0f}pp)")
        return True


# ============================================================ 评估（与 PD 同判据）
def evaluate_policy(model, difficulty, n=50, domain_rand=1.0, seed=1234,
                    policy_fn=None, history_len=0, phase=0, residual_pd=False):
    """返回 (成功率, 平均转正时间)。

    phase=0 用端到端判据（曾翻正+未坠地）
    phase=1/2 用对应阶段的判据（Phase1 看是否达成 up_z>0.7 交接）

    residual_pd: v30。必须与训练时的设置一致，否则评估会用错的动作合成方式
      （网络输出被当成绝对油门而非 PD 残差），结果无意义。
      当 policy_fn 给定时（纯 PD 基线对照），不启用残差。
    """
    env = qs.ThrowRecoveryEnv(difficulty=difficulty, domain_rand=domain_rand, seed=seed,
                              history_len=history_len, phase=phase,
                              residual_pd=(residual_pd and policy_fn is None))
    ok, t_recs = 0, []
    for _ in range(n):
        obs, _ = env.reset()
        rec = None
        reached_upright = False      # Phase1：是否曾达到 up_z>0.7（交接成功）
        stable_hold = 0              # Phase2：当前连续稳定步数
        max_stable_hold = 0          # Phase2：最大连续稳定步数
        for k in range(env.episode_steps):
            if policy_fn is not None:
                act = np.clip(policy_fn(env.sim, env.p) * 2 - 1, -1, 1)
            else:
                act, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(act)
            if rec is None and info["up_z"] > 0.95 and info["omega_norm"] < 1.0:
                rec = k / env.p.ctrl_hz
            if phase == 1 and info["up_z"] > 0.7:
                # v16: 与 env 一致 —— 纯翻转，翻正即交接（速度交给 P2）
                reached_upright = True
            if phase == 2:
                # v26: 与 env 判据严格同步（up_z > 0.95 且 |vz| < 0.5 且 omega < 1.2，连续 50 步达标）
                if (info["up_z"] > 0.95 and abs(info["vz"]) < 0.5
                        and info["omega_norm"] < 1.2):
                    stable_hold += 1
                    if stable_hold > max_stable_hold:
                        max_stable_hold = stable_hold
                else:
                    stable_hold = 0
            if term or trunc:
                break

        if phase == 1:
            # Phase1 判据：是否达成「翻正交接」（不要求活满，因为达标即终止是设计）
            if reached_upright:
                ok += 1
                if rec is not None:
                    t_recs.append(rec)
        elif phase == 2:
            # Phase2 判据：是否达成「稳定保持」（连续 50 步达标）
            if max_stable_hold >= 50:
                ok += 1
                if rec is not None:
                    t_recs.append(rec)
        else:
            alive = info["z"] > 0.05 or (info["up_z"] > 0.9 and abs(info["vz"]) < 0.6)
            if alive and rec is not None:
                ok += 1
                t_recs.append(rec)
    return ok / n, (float(np.mean(t_recs)) if t_recs else None)


# ============================================================ 主流程
def build_model(env, net, seed, tb_parent=None, run_name=None,
                net_arch=(64, 64), tanh_out=False, ent_coef=0.01):
    """构建 PPO 模型。

    net_arch : 隐层宽度，默认 (64,64)
    tanh_out : True 时使用 TanhActorCriticPolicy，把动作输出压到 (-1,1)
               （v10 实验：修复 69% 动作被 clip 的问题）
    ent_coef : 熵正则系数，默认 0.01；连续动作控制若策略梯度微弱，0.01 会导致动作方差单调发散
    """
    if net == "mlp":
        policy_kwargs = dict(net_arch=list(net_arch))
        policy = TanhActorCriticPolicy if tanh_out else "MlpPolicy"
    elif net == "gru":
        # 注意：此分支历史上只是占位，实际仍用 MlpPolicy，不是真正的 GRU
        policy_kwargs = dict(net_arch=list(net_arch))
        policy = TanhActorCriticPolicy if tanh_out else "MlpPolicy"
    else:
        raise ValueError(net)
    return PPO(policy, env, seed=seed, n_steps=1024, batch_size=256,
               learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
               clip_range=0.2, ent_coef=ent_coef, vf_coef=0.5,
               policy_kwargs=policy_kwargs, verbose=0, device="cpu",
               # 传 (父目录, run名)：sb3 会写到 <父目录>/<run名>/，
               # 我们的 eval SummaryWriter 也写同一个目录 → TensorBoard 合并成一张图
               tensorboard_log=tb_parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="curriculum", choices=["curriculum", "hard"])
    ap.add_argument("--difficulty", type=float, default=1.0, help="hard 模式的固定难度")
    ap.add_argument("--start", type=float, default=0.2, help="curriculum 起始难度")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--net", default="mlp", choices=["mlp", "gru"])
    ap.add_argument("--history", type=int, default=0,
                    help="History Stacking 帧数，0=不堆叠(22维)，3=堆3帧(43维)")
    ap.add_argument("--width", type=int, default=64,
                    help="隐层宽度，默认64。128 表示 [128,128]")
    ap.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                    help="分阶段训练：0=端到端（默认），1=翻转策略，2=稳定策略")
    ap.add_argument("--tanh-out", action="store_true",
                    help="在 action_net 后加 tanh，把动作输出压到 (-1,1)。"
                         "默认关闭以保持与历史实验可比。")
    ap.add_argument("--ent-coef", type=float, default=0.01,
                    help="PPO 熵损失系数，默认 0.01；Phase 2 推荐 0.001 避免噪声发散")
    ap.add_argument("--init-hover-bias", action="store_true",
                    help="是否将输出层偏置初始化为悬停油门动作 (a=0.4149 对应 u=0.7075)")
    ap.add_argument("--residual-pd", action="store_true",
                    help="v30：PD 残差控制模式。最终油门 = PD基线 + RESIDUAL_SCALE*网络输出。"
                         "网络此时学的是相对 PD 的修正量，而非从零学完整控制律。"
                         "实测：PD 在 P2 判据三项达标 95/93/88%%，而端到端 MLP 仅 9/44/8%%。"
                         "注意：奖励函数不因此改变，仍是 v24 那套。")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval", default=None, help="评估已有模型")
    ap.add_argument("--tag", default="", help="实验标签，会成为 TensorBoard 的 run 目录名，"
                    "例如 'v2_lr1e3_s0' → tb/v2_lr1e3_s0_1027_1430")
    args = ap.parse_args()

    out = args.out or os.path.join(_ROOT, "output", f"run_{args.mode}")
    os.makedirs(out, exist_ok=True)

    # ---------- 纯评估 ----------
    if args.eval:
        print(f"评估 {args.eval}")
        env = DifficultyEnv(difficulty=args.difficulty, history_len=args.history,
                            phase=args.phase, residual_pd=args.residual_pd)
        model = PPO.load(args.eval, env=env)
        for d in (0.3, 0.5, 0.7, 1.0):
            r, t = evaluate_policy(model, d, n=50, history_len=args.history,
                                  phase=args.phase, residual_pd=args.residual_pd)
            print(f"  difficulty={d:.1f}  成功率={r*100:5.1f}%  转正={t if t is None else round(t,2)}s")
        return

    # ===== 命名规范：<实验标签>_<seed>，时间戳移到子目录避免重名 =====
    # TensorBoard 左侧只显示这一层，所以名字必须简短且信息完整
    import datetime as _dt
    _stamp = _dt.datetime.now().strftime("%m%d_%H%M")
    _tag = args.tag or args.mode
    # run 名规范：<实验标签>_s<seed>  例：E3_v2_s0
    # 规则：标签里不放超参（超参看 MANIFEST.md），只放能区分实验的最少信息
    run_name = f"{_tag}_s{args.seed}"
    tb_parent = os.path.join(out, "tb")
    tb_dir = os.path.join(tb_parent, run_name)

    t0 = time.time()
    if args.mode == "curriculum":
        base = DifficultyEnv(difficulty=args.start, domain_rand=1.0, seed=args.seed,
                             history_len=args.history, phase=args.phase,
                             residual_pd=args.residual_pd)
        cb = CurriculumCallback(n_eval=30, every=50000, promote_at=0.70, step_up=0.2,
                                save_dir=out, smooth_window=3, max_std=0.20,
                                history_len=args.history, phase=args.phase,
                                residual_pd=args.residual_pd)
    else:
        base = DifficultyEnv(difficulty=args.difficulty, domain_rand=1.0, seed=args.seed,
                             history_len=args.history, phase=args.phase,
                             residual_pd=args.residual_pd)
        cb = CurriculumCallback(n_eval=30, every=50000, promote_at=99.0,
                                save_dir=out, smooth_window=3, max_std=0.20,
                                history_len=args.history, phase=args.phase,
                                residual_pd=args.residual_pd)

    env = Monitor(base)
    # 清理同名残留目录（必须在 PPO 构造前，否则 sb3 会加 _1 后缀）
    import shutil as _shutil, glob as _g
    for _d in _g.glob(os.path.join(tb_parent, run_name + "*")):
        _shutil.rmtree(_d, ignore_errors=True)

    model = build_model(env, args.net, args.seed,
                        net_arch=(args.width, args.width),
                        tanh_out=args.tanh_out,
                        ent_coef=args.ent_coef,
                        tb_parent=tb_parent, run_name=run_name)
    if args.init_hover_bias:
        # 将 policy.action_net 的偏置初始化为悬停油门动作 0.4149 (u=0.7075)
        # 消除开局因半重力自由下坠造成的零有效探索
        model.policy.action_net.bias.data.fill_(0.4149)
        print("★ [Prior] action_net.bias 已初始化为 0.4149 (对应悬停油门 u=0.7075)")

    print(f"模式={args.mode}  网络={args.net}  步数={args.steps}  起始难度="
          f"{args.start if args.mode=='curriculum' else args.difficulty}  ent_coef={args.ent_coef}")
    print(f"参数总量 {sum(p.numel() for p in model.policy.parameters())}")

    cb.attach_writer(tb_parent, run_name)
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False,
                tb_log_name=run_name)
    cb.close()
    model.save(os.path.join(out, "model.zip"))

    # 保存课程曲线
    with open(os.path.join(out, "curriculum.json"), "w", encoding="utf-8") as f:
        json.dump(cb.history, f, indent=2)
    with open(os.path.join(out, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(cb._bench_history, f, indent=2)

    print(f"\n训练耗时 {time.time()-t0:.0f}s，正在评估...")
    results = []
    print(f"{'难度':>6} {'成功率':>8} {'转正时间':>10}")
    for d in (0.3, 0.5, 0.7, 1.0):
        r, t = evaluate_policy(model, d, n=50, history_len=args.history,
                                  phase=args.phase, residual_pd=args.residual_pd)
        results.append(dict(difficulty=d, success=r, t_rec=t))
        print(f"{d:6.1f} {r*100:7.1f}% {('  --  ' if t is None else f'{t:.2f}s'):>10}")

    # PD 基线（同判据同随机种子）
    print("\nPD 基线对照：")
    pd_res = []
    for d in (0.3, 0.5, 0.7, 1.0):
        r, t = evaluate_policy(None, d, n=50, policy_fn=qs.pd_controller)
        pd_res.append(dict(difficulty=d, success=r, t_rec=t))
        print(f"{d:6.1f} {r*100:7.1f}% {('  --  ' if t is None else f'{t:.2f}s'):>10}")

    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(mode=args.mode, net=args.net, steps=args.steps,
                       rl=results, pd=pd_res), f, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
