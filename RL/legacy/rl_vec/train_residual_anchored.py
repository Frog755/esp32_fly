#!/usr/bin/env python3
"""train_residual_anchored.py —— BC 锚定残差训练（Self-Imitation / KL-锚定家族）

损失: L = L_PPO + λ_BC · E‖π_θ(s)−a*‖² ，(s,a*) 来自 model_2M 成功轨迹数据集。
目的: 对冲 on-policy 稀释——成功技能有恒定梯度维持，不再"2M 学会、60M 丢掉"。
附带: best-checkpoint 按评估自动保留（stable50 为主、eval_rew 破平）。
用法: python scripts/train_residual_anchored.py [total_steps] [out_name] [bc_coef]
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
import sys, os, time, json
HERE = _LEGACY_ORIG_DIR
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
import numpy as np
import torch as th
import torch
import torch.nn.functional as F
import gymnasium.spaces as spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecFrameStack
from vec_residual_env import VecResidualEnv
from yao_kun_params import YaoKunK10Params

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else 'residual_anchored'
BC_COEF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
DATA = os.path.join(HERE, '..', 'models', 'anchor_dataset.npz')
OUT = os.path.join(HERE, '..', 'models', OUT_NAME)
os.makedirs(OUT, exist_ok=True)


class AnchoredPPO(PPO):
    """PPO + BC 锚定损失（忠实抄写 SB3 2.9.2 train()，仅插入 BC 项）。"""

    def __init__(self, *args, bc_obs=None, bc_act=None, bc_coef=0.5, bc_batch=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_obs = th.as_tensor(np.asarray(bc_obs, np.float32))
        self.bc_act = th.as_tensor(np.asarray(bc_act, np.float32))
        self.bc_coef = bc_coef
        self.bc_batch = bc_batch
        self._bc_rng = np.random.default_rng(0)

    def _excluded_save_params(self) -> list:
        """bc 数据集不进 checkpoint（曾把 154MB 数据集序列化进每个 ckpt，50 个≈11GB 撑爆磁盘）。"""
        return super()._excluded_save_params() + ["bc_obs", "bc_act", "_bc_rng"]

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, clip_fractions, bc_losses = [], [], [], [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                # ---- BC 锚定项（本类唯一新增）----
                bc_idx = self._bc_rng.integers(0, len(self.bc_obs), size=self.bc_batch)
                bc_o = self.bc_obs[bc_idx].to(self.device)
                bc_a = self.bc_act[bc_idx].to(self.device)
                bc_dist = self.policy.get_distribution(bc_o)
                bc_loss = F.mse_loss(bc_dist.distribution.mean, bc_a)
                bc_losses.append(bc_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss \
                    + self.bc_coef * bc_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/bc_loss", float(np.mean(bc_losses)))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())


if __name__ == "__main__":
    torch.set_num_threads(2)
    d = np.load(DATA)
    bc_obs, bc_act = d['bc_obs'], d['bc_act']
    print(f"锚定数据集: {len(bc_obs)} 对  λ_BC={BC_COEF}", flush=True)

    base = VecResidualEnv(num_envs=256, seed=0, episode_s=5.0, params=YaoKunK10Params())
    venv = VecFrameStack(base, n_stack=8)
    ebase = VecResidualEnv(num_envs=64, seed=777, episode_s=5.0, params=YaoKunK10Params())
    evenv = VecFrameStack(ebase, n_stack=8)   # 评估升到 64 通道（降低 16 通道的小样本噪声）

    model = AnchoredPPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=3,
                        learning_rate=3e-4, gamma=0.999, gae_lambda=0.95,
                        ent_coef=0.001, verbose=0, device="cpu", seed=0,
                        policy_kwargs=dict(log_std_init=-1.0),
                        bc_obs=bc_obs, bc_act=bc_act, bc_coef=BC_COEF,
                        tensorboard_log=os.path.join(OUT, 'tb'))

    CHUNK = 2_000_000
    eval_log, best = [], dict(score=(-1, -1e18))
    t0 = time.time()
    done_steps = 0
    chunk = 0
    while done_steps < TOTAL:
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=(chunk == 0))
        done_steps += CHUNK
        chunk += 1
        model.save(os.path.join(OUT, f"model_{done_steps//1_000_000}M.zip"))
        obs = evenv.reset()
        ep_rew = np.zeros(64)
        stable_max = np.zeros(64)
        n_crash = 0
        for _ in range(1000):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = evenv.step(a)
            ep_rew += r
            for i in range(64):
                stable_max[i] = max(stable_max[i], infos[i]['stable_cnt'])
                if done[i]:
                    n_crash += int(infos[i].get('crash', False))
            if done.any():
                obs = evenv.reset()
        s50 = int((stable_max >= 50).sum())
        rec = dict(step=done_steps, eval_rew=float(ep_rew.mean()),
                   stable50=s50, stable_med=int(np.median(stable_max)),
                   stable_max=int(stable_max.max()), crash=n_crash,
                   elapsed=int(time.time() - t0))
        eval_log.append(rec)
        with open(os.path.join(OUT, 'eval_log.json'), 'w') as f:
            json.dump(eval_log, f, indent=2)
        model.logger.record("eval/stable50", s50)
        model.logger.record("eval/rew_mean", rec['eval_rew'])
        model.logger.dump(step=done_steps)
        # ---- best-checkpoint（B1：稳定50 为主，eval_rew 破平）----
        score = (s50, rec['eval_rew'])
        if score > best['score']:
            best = dict(score=score, step=done_steps)
            model.save(os.path.join(OUT, "model_best.zip"))
            with open(os.path.join(OUT, 'best.json'), 'w') as f:
                json.dump(best, f, indent=2)
        print(f"chunk {chunk}: steps {done_steps:>9,}  eval_rew={rec['eval_rew']:+9.1f}  "
              f"稳定≥50步 {s50}/64  最长稳定 {rec['stable_max']}  坠地 {n_crash}  "
              f"best@{best['step']//1_000_000}M  [{rec['elapsed']}s]", flush=True)
    print("TRAINING_COMPLETE", flush=True)
