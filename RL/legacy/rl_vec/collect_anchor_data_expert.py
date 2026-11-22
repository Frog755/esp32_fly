#!/usr/bin/env python3
"""collect_anchor_data_expert.py —— SIL 锚定数据集：v3 专家成功轨迹（目标动作=0）

优雅之处: 残差定义 u = u_expert + K·g·a，"专家会做的"的最优残差就是 a=0。
所以数据集 = 专家成功回合里出现的 (obs, 0) 对；失败态不采集——留给 RL 自由探索。
锚定损失 λ‖π(s)−0‖² 在成功态等价于"别插手"约束，在盲区无梯度污染。
用法: python scripts/collect_anchor_data_expert.py [n_seeds] [max_pairs]
输出: models/anchor_dataset_expert.npz
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
import sys, os
HERE = _LEGACY_ORIG_DIR
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecFrameStack
from vec_expert_residual_env import VecExpertResidualEnv
from yao_kun_params import YaoKunK10Params

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAX_PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
OUT = os.path.join(HERE, '..', 'models', 'anchor_dataset_expert.npz')

if __name__ == "__main__":
    torch.set_num_threads(4)
    all_o = []
    for seed in range(21, 21 + N_SEEDS):
        env = VecExpertResidualEnv(num_envs=256, seed=seed, episode_s=5.0,
                                   params=YaoKunK10Params())
        obs = env.reset()                       # 残差不用 a：直接取 16 维再手工堆
        o_store = np.zeros((1000, 256, 16), np.float32)
        smax = np.zeros(256); crashed = np.zeros(256, bool)
        for t in range(1000):
            o_store[t] = obs
            obs, r, done, infos = env.step(np.zeros((256, 4)))
            smax = np.maximum(smax, [i2['stable_cnt'] for i2 in infos])
            crashed |= np.array([i2['crash'] for i2 in infos], bool)
        ok = (smax >= 50) & (~crashed)
        pairs = o_store[:, ok]                  # (1000, n_ok, 16)
        # 展开成 8 帧堆叠观测: 每个成功通道取滑窗
        P = pairs.transpose(1, 0, 2)             # (n_ok, 1000, 16)
        n_ok = P.shape[0]
        stacked = np.lib.stride_tricks.sliding_window_view(P, 8, axis=1)  # (n_ok,993,16,8)
        stacked = np.moveaxis(stacked, -1, 2).reshape(n_ok, 993, 128)     # (n_ok,993,8×16)
        # 注意 sliding_window 帧序 = 时间从旧到新 → SB3 布局最新在最后，一致 ✓
        all_o.append(stacked.reshape(-1, 128).astype(np.float32)[::5])  # 每5帧取1, 控制规模
        print(f"seed {seed}: 成功通道 {n_ok}/256 → 样本 {len(stacked.reshape(-1,128))//5}", flush=True)
    bc_obs = np.concatenate(all_o)
    if len(bc_obs) > MAX_PAIRS:
        idx = np.random.default_rng(0).choice(len(bc_obs), MAX_PAIRS, replace=False)
        bc_obs = bc_obs[idx]
    bc_act = np.zeros_like(bc_obs[:, :4])       # 目标动作恒 0（专家区不插手）
    np.savez_compressed(OUT, bc_obs=bc_obs, bc_act=bc_act)
    print(f"数据集: {len(bc_obs)} 对 → {OUT}")
