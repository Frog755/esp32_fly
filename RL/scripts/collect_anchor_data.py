#!/usr/bin/env python3
"""collect_anchor_data.py —— 采集 BC 锚定数据集（来自 model_2M 的成功轨迹）

只保留成功回合（达成稳定≥50步 且 首回合未坠地）的 (obs_128维, action) 对。
失败回合不进数据集——锚垃圾 = 把垃圾焊死在策略上。
用法: python scripts/collect_anchor_data.py [ckpt] [n_seeds] [max_pairs]
输出: models/anchor_dataset.npz  (bc_obs float32 [M,128], bc_act float32 [M,4])
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from vec_residual_env import VecResidualEnv
from yao_kun_params import YaoKunK10Params

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'models', 'residual_v1', 'model_2M.zip')
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_PAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 300_000
OUT = os.path.join(HERE, '..', 'models', 'anchor_dataset.npz')

if __name__ == "__main__":
    torch.set_num_threads(2)
    model = PPO.load(CKPT, device="cpu")
    all_obs, all_act = [], []
    for seed in range(7, 7 + N_SEEDS):
        base = VecResidualEnv(num_envs=256, seed=seed, episode_s=5.0, params=YaoKunK10Params())
        evenv = VecFrameStack(base, n_stack=8)
        obs = evenv.reset()
        obs_store = np.zeros((1000, 256, 128), np.float32)
        act_store = np.zeros((1000, 256, 4), np.float32)
        crashed = np.zeros(256, bool)
        stable_max = np.zeros(256)
        for t in range(1000):
            a, _ = model.predict(obs, deterministic=True)
            obs_store[t] = obs
            act_store[t] = a
            obs, r, done, infos = evenv.step(a)
            sc = np.array([inf['stable_cnt'] for inf in infos])
            stable_max = np.maximum(stable_max, sc)
            crashed |= np.array([inf['crash'] for inf in infos], bool) & (np.array([inf['k'] for inf in infos]) <= t + 1)
        ok = (stable_max >= 50) & (~crashed)
        o = obs_store[:, ok]                     # (1000, n_ok, 128)
        a = act_store[:, ok]                     # (1000, n_ok, 4)
        all_obs.append(o.transpose(1, 0, 2).reshape(-1, 128))
        all_act.append(a.transpose(1, 0, 2).reshape(-1, 4))
        print(f"seed {seed}: 成功通道 {int(ok.sum())}/256 → {int(ok.sum())*1000} 对", flush=True)
    bc_obs = np.concatenate(all_obs)
    bc_act = np.concatenate(all_act)
    if len(bc_obs) > MAX_PAIRS:
        idx = np.random.default_rng(0).choice(len(bc_obs), MAX_PAIRS, replace=False)
        bc_obs, bc_act = bc_obs[idx], bc_act[idx]
    np.savez_compressed(OUT, bc_obs=bc_obs, bc_act=bc_act)
    print(f"数据集: {len(bc_obs)} 对 → {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")
