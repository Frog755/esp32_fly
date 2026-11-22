#!/usr/bin/env python3
"""train_sac.py —— Raptor 配方回归：SAC（回放池）+ 全球面 + 稠密二次奖励 + 帧堆叠

背景: R1~R4 证明 PPO(on-policy) 在全球面分布下灾难性遗忘（R2 尸检）。
      Raptor 配方的承重件是 SAC 回放池——旧技能数据永久保留。
      本脚本 = 回归被验证的配方（GRU 暂以 8 帧堆叠近似，RecurrentPPO 实测 31 步/s 不可用），
      同时实测 CPU 上 SAC 真实吞吐（为租 4090 决策提供依据）。
用法: python scripts/train_sac.py [total_steps] [out_name]
"""
import sys, os, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecFrameStack
from vec_recovery_env import VecRecoveryEnv
from yao_kun_params import YaoKunK10Params

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else 'sac_v1'
OUT = os.path.join(HERE, '..', 'models', OUT_NAME)
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    torch.set_num_threads(4)
    # 训练 8 通道（off-policy 梯度更新主导耗时，通道数不必 256）
    base = VecRecoveryEnv(num_envs=8, seed=0, episode_s=5.0, params=YaoKunK10Params(),
                          tilt_cap_deg=180.0)          # 全球面直接开局（Raptor all-around）
    venv = VecFrameStack(base, n_stack=8)
    ebase = VecRecoveryEnv(num_envs=16, seed=777, episode_s=5.0, params=YaoKunK10Params(),
                           tilt_cap_deg=180.0)
    evenv = VecFrameStack(ebase, n_stack=8)

    model = SAC("MlpPolicy", venv, buffer_size=2_000_000, batch_size=128,
                learning_rate=3e-4, learning_starts=10_000,
                train_freq=1, gradient_steps=-1,      # 1:1 更新/转移（标准 SAC）
                ent_coef='auto', verbose=0, device="cpu", seed=0,
                tensorboard_log=os.path.join(OUT, 'tb'))

    CHUNK = 1_000_000
    eval_log = []
    t0 = time.time()
    done_steps = 0
    chunk = 0
    while done_steps < TOTAL:
        t_chunk = time.time()
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=(chunk == 0))
        done_steps += CHUNK
        chunk += 1
        rate = CHUNK / (time.time() - t_chunk)
        eta_h = (TOTAL - done_steps) / rate / 3600
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
                   steps_per_s=round(rate, 1),
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
              f"z_end>2m {rec['z_end_gt2']}/16  坠地终局 {rec['crash']}  "
              f"[{rate:.0f} steps/s, ETA {eta_h:.1f}h]", flush=True)
    print("TRAINING_COMPLETE", flush=True)
