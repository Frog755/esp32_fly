#!/usr/bin/env python3
"""train_rate_anchored.py —— C1 语义动作层 + SIL 自模仿锚定（迭代闭环第一轮）

组件（各自已被先前实验验证）:
  · VecRateEnv: rate 内环 + 语义指令（rate_v1 证明可从零学到 9.8% 大样本技能）
  · AnchoredPPO λ=5（residual_anchored_l5 证明 100M 不塌的唯一配方）
  · 锚定数据集 = rate model_12M 成功轨迹（Self-Imitation：17.4 万对，成功率过滤）
迭代逻辑: 本轮若产出新 best（>9.8%），可用新 best 重采数据集进入下一轮 SIL。
用法: python scripts/train_rate_anchored.py [total_steps] [out_name] [bc_coef]
"""
import sys, os, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
sys.path.insert(0, HERE)
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecFrameStack
from vec_rate_env import VecRateEnv
from yao_kun_params import YaoKunK10Params
from train_residual_anchored import AnchoredPPO

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else 'rate_sil'
BC_COEF = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
DATA = os.path.join(HERE, '..', 'models', 'anchor_dataset_rate.npz')
OUT = os.path.join(HERE, '..', 'models', OUT_NAME)
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    torch.set_num_threads(2)
    d = np.load(DATA)
    print(f"SIL 锚定数据集: {len(d['bc_obs'])} 对  λ_BC={BC_COEF}", flush=True)

    base = VecRateEnv(num_envs=256, seed=0, episode_s=5.0, params=YaoKunK10Params())
    venv = VecFrameStack(base, n_stack=8)
    ebase = VecRateEnv(num_envs=64, seed=777, episode_s=5.0, params=YaoKunK10Params())
    evenv = VecFrameStack(ebase, n_stack=8)

    model = AnchoredPPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=3,
                        learning_rate=3e-4, gamma=0.999, gae_lambda=0.95,
                        ent_coef=0.001, verbose=0, device="cpu", seed=0,
                        policy_kwargs=dict(log_std_init=-1.0),
                        bc_obs=d['bc_obs'], bc_act=d['bc_act'], bc_coef=BC_COEF,
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
