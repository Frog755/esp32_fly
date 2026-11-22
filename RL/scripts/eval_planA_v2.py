#!/usr/bin/env python3
"""eval_planA_v2.py —— 第二轮训练后状态覆盖度诊断

对照第一轮尸检结论（策略=常数油门 0.53，正立/倒扣输出差仅 0.01）：
在 7 档初始倾角（0°~180°）× 若干 yaw 下 rollout，报告：
  - 各档倾角的油门统计（应随状态显著变化——死锁解除的直接证据）
  - 各档倾角的翻正时间 / 终局高度 / 是否稳定
用法: python scripts/eval_planA_v2.py <model.zip> [episode_s]
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from vec_recovery_env import VecRecoveryEnv
from yao_kun_params import YaoKunK10Params
from quad_sim_vec import up_z_of

CKPT = sys.argv[1]
EP_S = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
TILTS = [0, 30, 60, 90, 120, 150, 180]

if __name__ == "__main__":
    torch.set_num_threads(2)
    base = VecRecoveryEnv(num_envs=len(TILTS), seed=42, obs_mode='deploy',
                          episode_s=EP_S, params=YaoKunK10Params(), tilt_cap_deg=180.0)
    evenv = VecFrameStack(base, n_stack=8)
    model = PPO.load(CKPT, device="cpu")

    # 定制初始化：每个通道一个固定倾角，yaw=0，零角速度，z=2.5
    base.reset()
    for i, tilt_deg in enumerate(TILTS):
        ang = np.deg2rad(tilt_deg)
        base.dyn.quat[i] = [np.cos(ang/2), np.sin(ang/2), 0, 0]  # 绕 x 轴倾 tilt
        base.dyn.w[i] = 0.0
        base.dyn.vel[i] = 0.0
        base.dyn.pos[i] = [0.0, 0.0, 2.5]
    obs = evenv.reset()   # 注意：reset 会重新采样；改为手动写入后再取观测
    # 手动重新写入（reset 后再覆盖）
    for i, tilt_deg in enumerate(TILTS):
        ang = np.deg2rad(tilt_deg)
        base.dyn.quat[i] = [np.cos(ang/2), np.sin(ang/2), 0, 0]
        base.dyn.w[i] = 0.0
        base.dyn.vel[i] = 0.0
        base.dyn.pos[i] = [0.0, 0.0, 2.5]
    base._prev_a = np.zeros((len(TILTS), 4))
    base._filt_init[:] = False
    obs = base._get_obs()
    # 手动展开帧堆叠（对齐 VecFrameStack 布局：最新帧在最前）
    n_stack = 8
    single = obs
    obs_stacked = np.concatenate([single] * n_stack, axis=1)

    n_steps = int(EP_S * 200)
    thr_log = np.zeros((n_steps, len(TILTS)))
    stable_cnt = np.zeros(len(TILTS), dtype=np.int64)
    stable_max = np.zeros(len(TILTS), dtype=np.int64)
    for t in range(n_steps):
        a, _ = model.predict(obs_stacked, deterministic=True)
        thr_log[t] = ((a + 1) / 2).mean(axis=1)   # 4 电机油门均值（反映总推力意图）
        obs, r, done, infos = base.step(a)
        single = base._get_obs()
        obs_stacked = np.concatenate([obs_stacked[:, single.shape[1]:], single], axis=1)  # SB3 布局：最新帧在最后
        for i in range(len(TILTS)):
            stable_cnt[i] = stable_cnt[i] + 1 if infos[i]['stable_cnt'] > 0 else 0
            stable_max[i] = max(stable_max[i], stable_cnt[i])

    print(f"\n=== {os.path.basename(CKPT)} 状态覆盖度诊断（每档倾角独立 rollout {EP_S}s）===")
    print(f"{'倾角':>5} | {'油门均值':>8} {'油门范围':>15} | {'终局up_z':>8} {'终局z':>7} | 最长稳定步")
    for i, tilt in enumerate(TILTS):
        u = thr_log[:, i]
        upz = up_z_of(base.dyn.quat[i:i+1])[0]
        z = base.dyn.pos[i, 2]
        print(f"{tilt:>4}° | {u.mean():>8.3f} [{u.min():.3f},{u.max():.3f}] | "
              f"{upz:>8.3f} {z:>7.2f} | {stable_max[i]}")
    # 死锁判据：倾角 0° 与 180° 的油门均值差（第一轮=0.01）
    d = abs(thr_log[:, 0].mean() - thr_log[:, -1].mean())
    print(f"\n正立/倒扣油门均值差: {d:.3f}（第一轮常数策略 = 0.01 → 越大越好）")
