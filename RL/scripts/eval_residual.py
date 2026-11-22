#!/usr/bin/env python3
"""eval_residual.py —— 残差策略尸检：固定倾角 rollout + 残差幅值分布

用法: python scripts/eval_residual.py <model.zip> [episode_s]
报告每档倾角：|残差|均值/峰值（验证"正立趋零、倒扣带输出"的架构契约）、终局姿态、最长稳定。
帧堆叠对齐：写入标称初态后用 8 步 a=0 热身冲掉旧帧（8 帧堆叠）。
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
from quad_sim_vec import up_z_of

CKPT = sys.argv[1]
EP_S = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
TILTS = [0, 30, 60, 90, 120, 150, 180]
N = len(TILTS)


def write_nominal(base):
    for i, tilt in enumerate(TILTS):
        a_ = np.deg2rad(tilt)
        base.dyn.quat[i] = [np.cos(a_/2), np.sin(a_/2), 0, 0]
        base.dyn.w[i] = 0.0
        base.dyn.vel[i] = 0.0
        base.dyn.pos[i] = [0.0, 0.0, 2.5]
    base._filt_init[:] = False


if __name__ == "__main__":
    torch.set_num_threads(2)
    base = VecResidualEnv(num_envs=N, seed=42, episode_s=EP_S, params=YaoKunK10Params())
    evenv = VecFrameStack(base, n_stack=8)
    model = PPO.load(CKPT, device="cpu")

    evenv.reset()
    write_nominal(base)
    for _ in range(8):                       # 热身：8 帧堆叠对齐（此段不计入统计）
        obs, *_ = evenv.step(np.zeros((N, 4)))
    write_nominal(base)                      # 回到标称初态，正式开始
    n_steps = int(EP_S * 200)
    A = np.zeros((n_steps, N, 4))
    sc = np.zeros(N)
    for t in range(n_steps):
        a, _ = model.predict(obs, deterministic=True)
        A[t] = a
        obs, r, done, infos = evenv.step(a)
        sc = np.maximum(sc, [inf['stable_cnt'] for inf in infos])

    print(f"\n=== {os.path.basename(CKPT)} 残差尸检（固定倾角 {EP_S}s，初速/初角速度=0）===")
    print(f"{'倾角':>5} | {'|残差|均值':>9} {'峰值':>6} | {'终局up_z':>8} {'终局z':>6} | 最长稳定步")
    for i, tilt in enumerate(TILTS):
        print(f"{tilt:>4}° | {np.abs(A[:, i]).mean():>9.3f} {np.abs(A[:, i]).max():>6.2f} | "
              f"{up_z_of(base.dyn.quat[i:i+1])[0]:>8.3f} {base.dyn.pos[i, 2]:>6.2f} | {sc[i]:.0f}")
