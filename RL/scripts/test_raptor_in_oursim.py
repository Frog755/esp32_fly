#!/usr/bin/env python3
"""test_raptor_in_oursim.py —— Raptor 基础策略在本项目自研 sim 中的闭环测试
目的：回答"RL 到底能不能做大角度回正"——用 Raptor 权重直接验证。
观测映射 (Raptor 约定 22 维):
  [0:3]  pos (m, FLU world)
  [3:12] 旋转矩阵展平 (row-major)
  [12:15] lin_vel (m/s, world)
  [15:18] ang_vel (rad/s, body)
  [18:22] prev_action (RPM 归一化 [-1,1])
动作映射: a∈[-1,1] -> 油门 u=(a+1)/2 (RPM 归一化与油门的近似等效)
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))
import quad_sim_fast_v2 as qs
from raptor_policy_np import RaptorPolicy, RAPTOR_DEFAULT

def quat_to_R(q):
    qw,qx,qy,qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),     1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)]])

def build_obs(env, prev_a):
    s = env.sim
    R = quat_to_R(s.q)
    return np.concatenate([s.pos.copy(), R.flatten(), s.vel.copy(),
                           np.asarray(s.w), prev_a]).astype(np.float64)

def run_episode(pol, difficulty, seed, episode_s=4.0, phase=0, ctrl_every=1, verbose=False):
    env = qs.ThrowRecoveryEnv(difficulty=difficulty, domain_rand=1.0,
                              episode_s=episode_s, seed=seed, phase=phase)
    obs, _ = env.reset()
    pol.reset()
    prev_a = np.zeros(4)
    z0 = float(env.sim.pos[2]); zmin = z0
    flipped = False; flip_k = None
    k = 0
    for k in range(env.episode_steps):
        if k % ctrl_every == 0:
            a = pol.step(build_obs(env, prev_a))
            prev_a = np.clip(a, -1, 1)
        zmin = min(zmin, float(env.sim.pos[2]))
        obs, r, term, trunc, info = env.step(prev_a)
        if not flipped and info['up_z'] > 0.9 and info['omega_norm'] < 2.0:
            flipped = True; flip_k = k
        if term or trunc:
            break
    from eval import classify_outcome
    outcome = classify_outcome(info, trunc, k+1, env.episode_steps)
    return dict(outcome=outcome, steps=k+1, z0=z0, zmin=zmin, zend=float(env.sim.pos[2]),
                flip_k=flip_k, up_z=info['up_z'], omega=info['omega_norm'], vz=info['vz'])

def batch(pol, difficulty, n=50, phase=0, ctrl_every=1, seed0=20261029):
    from collections import Counter
    cnt = Counter(); zs=[]; flips=[]
    for i in range(n):
        r = run_episode(pol, difficulty, seed0+i, phase=phase, ctrl_every=ctrl_every)
        cnt[r['outcome']] += 1
        zs.append((r['z0'], r['zmin'], r['zend']))
        flips.append(r['flip_k'])
    return cnt, zs, flips

if __name__ == "__main__":
    pol = RaptorPolicy(RAPTOR_DEFAULT)
    print("Raptor 基础策略 在自研 sim 中的闭环测试 (50 回合/档, 种子集同 PD 对照)")
    print("="*72)
    for phase, name in [(0, "端到端(任意姿态抛掷)")]:
        for d in [0.3, 0.7, 1.0]:
            cnt, zs, flips = batch(pol, d, n=50, phase=phase)
            n_ok = cnt.get('success', 0)
            zf = [z for z in zs if z[0] > 0]
            drop = np.mean([z0-zmin for z0, zmin, _ in zf])
            fk = [f for f in flips if f is not None]
            ft = np.median(fk)*0.005 if fk else float('nan')
            print(f"  phase={phase} d={d:.1f}: 成功 {n_ok}/50 = {n_ok*2:.0f}%  "
                  f"| 平均最大掉高 {drop:.2f}m | 翻正用时中位 {ft:.2f}s")
            for k2, v in cnt.most_common():
                if k2 != 'success':
                    print(f"      └ {k2:<22}{v:>4} ({v*2:.0f}%)")
