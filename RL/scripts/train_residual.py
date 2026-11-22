#!/usr/bin/env python3
"""train_residual.py —— Round 5 主线：残差 PD+RL（PPO）

架构: VecResidualEnv —— PD 兜底悬停（96% 基线向量化版），RL 只学残差；
      全球面直接开局（PD 基座使梯度抵消不存在，无需课程）。
PPO 超参与 R1~R4 完全一致（单变量：只换环境架构）。
用法: python scripts/train_residual.py [total_steps] [out_name]
"""
import sys, os, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from vec_residual_env import VecResidualEnv
from yao_kun_params import YaoKunK10Params

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else 'residual_v1'
OUT = os.path.join(HERE, '..', 'models', OUT_NAME)
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    torch.set_num_threads(2)
    base = VecResidualEnv(num_envs=256, seed=0, episode_s=5.0, params=YaoKunK10Params())
    venv = VecFrameStack(base, n_stack=8)
    ebase = VecResidualEnv(num_envs=16, seed=777, episode_s=5.0, params=YaoKunK10Params())
    evenv = VecFrameStack(ebase, n_stack=8)

    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=3,
                learning_rate=3e-4, gamma=0.999, gae_lambda=0.95,
                ent_coef=0.001, verbose=0, device="cpu", seed=0,
                policy_kwargs=dict(log_std_init=-1.0),   # 初始 std 0.37：残差初值 |a|≈0.2，
                                                         # 不再一步就搅碎 PD 悬停（默认 std=1）
                tensorboard_log=os.path.join(OUT, 'tb'))

    CHUNK = 2_000_000
    eval_log = []
    t0 = time.time()
    done_steps = 0
    chunk = 0
    while done_steps < TOTAL:
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=(chunk == 0))
        done_steps += CHUNK
        chunk += 1
        model.save(os.path.join(OUT, f"model_{done_steps//1_000_000}M.zip"))
        # ---------- 确定性评估（全球面） ----------
        obs = evenv.reset()
        ep_rew = np.zeros(16)
        stable_max = np.zeros(16, dtype=np.int64)
        z_end = np.zeros(16)
        n_crash = 0
        for _ in range(1000):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = evenv.step(a)
            ep_rew += r
            for i in range(16):
                stable_max[i] = max(stable_max[i], infos[i]['stable_cnt'])
                if done[i]:
                    z_end[i] = infos[i]['z']
                    n_crash += int(infos[i].get('crash', False))
            if done.any():
                obs = evenv.reset()
        rec = dict(step=done_steps,
                   eval_rew=float(ep_rew.mean()),
                   stable50=int((stable_max >= 50).sum()),
                   stable_med=int(np.median(stable_max)),
                   stable_max=int(stable_max.max()),
                   z_end_mean=float(z_end.mean()),
                   z_end_gt2=int((z_end > 2.0).sum()),
                   crash=n_crash,
                   elapsed=int(time.time() - t0))
        eval_log.append(rec)
        with open(os.path.join(OUT, 'eval_log.json'), 'w') as f:
            json.dump(eval_log, f, indent=2)
        model.logger.record("eval/rew_mean", rec['eval_rew'])
        model.logger.record("eval/stable50", rec['stable50'])
        model.logger.record("eval/stable_med", rec['stable_med'])
        model.logger.record("eval/z_end_gt2", rec['z_end_gt2'])
        model.logger.record("eval/crash_end", rec['crash'])
        model.logger.dump(step=done_steps)
        print(f"chunk {chunk}: steps {done_steps:>9,}  eval_rew={rec['eval_rew']:+9.1f}  "
              f"稳定≥50步 {rec['stable50']}/16  最长稳定 {rec['stable_max']}  "
              f"z_end>2m {rec['z_end_gt2']}/16  坠地终局 {rec['crash']}  [{rec['elapsed']}s]", flush=True)
    print("TRAINING_COMPLETE", flush=True)
