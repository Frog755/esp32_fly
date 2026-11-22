#!/usr/bin/env python3
"""train_planA.py —— Plan A 训练脚本（第四轮：性能门控课程 + 坠地终止 + 奖励再平衡 + TB）

第四轮修正（对应第三轮尸检：时间门控线性爬坡过快 → 分布追不上、悬停未巩固即被冲走）：
  1. 性能门控课程：cap 从 30° 起步，仅当当前壳层考核通过（稳定≥50步 ≥12/16）才 +15°，
     否则原地继续训练——强制每个壳层巩固后再扩大（Round2 实测定点 30° 训 10M 可达 15/16）；
  2. 保留已验证项：坠地终止（z<目标-3m）、W_POS=1.0、TensorBoard、渐进带采样；
  3. PPO 超参与前三轮完全一致（单变量归因：只改课程推进方式）。

用法: python scripts/train_planA.py [total_steps] [out_name]
默认: 100_000_000  planA_v4
"""
import sys, os, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from vec_recovery_env import VecRecoveryEnv, STABLE_CRIT
from yao_kun_params import YaoKunK10Params

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else 'planA_v4'
OUT = os.path.join(HERE, '..', 'models', OUT_NAME)
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    torch.set_num_threads(2)
    # 训练环境 256 通道（cap=30° 起步）
    base = VecRecoveryEnv(num_envs=256, seed=0, obs_mode='deploy',
                          episode_s=5.0, params=YaoKunK10Params(), tilt_cap_deg=30.0)
    venv = VecFrameStack(base, n_stack=8)
    # 评估环境 16 通道（独立种子；倾角上限跟随训练课程）
    ebase = VecRecoveryEnv(num_envs=16, seed=777, obs_mode='deploy',
                           episode_s=5.0, params=YaoKunK10Params(), tilt_cap_deg=30.0)
    evenv = VecFrameStack(ebase, n_stack=8)

    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=3,
                learning_rate=3e-4, gamma=0.999, gae_lambda=0.95,
                ent_coef=0.001, verbose=0, device="cpu", seed=0,
                tensorboard_log=os.path.join(OUT, 'tb'))

    CHUNK = 2_000_000          # 每 chunk 200 万步
    eval_log = []
    t0 = time.time()
    done_steps = 0
    chunk = 0
    cap_deg = 30.0                     # 性能门控课程：考核通过才前进
    GATE_STABLE50 = 12                 # 当前壳层考核线：稳定≥50步 ≥12/16
    CAP_STEP = 15.0                    # 每次前进 +15°
    while done_steps < TOTAL:
        base.set_tilt_cap(cap_deg)
        ebase.set_tilt_cap(cap_deg)
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=(chunk == 0))
        done_steps += CHUNK
        chunk += 1
        model.save(os.path.join(OUT, f"model_{done_steps//1_000_000}M.zip"))
        # ---------- 确定性评估（当前阶段分布） ----------
        obs = evenv.reset()
        ep_rew = np.zeros(16)
        stable_max = np.zeros(16, dtype=np.int64)
        z_end = np.zeros(16)
        n_crash = 0
        steps = 0
        for _ in range(1000):     # 5s
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = evenv.step(a)
            ep_rew += r
            steps += 1
            for i in range(16):
                stable_max[i] = max(stable_max[i], infos[i]['stable_cnt'])
                if done[i]:
                    z_end[i] = infos[i]['z']
                    n_crash += int(infos[i].get('crash', False))
            if done.any():
                obs = evenv.reset()
        rec = dict(step=done_steps, tilt_cap=round(cap_deg, 1),
                   eval_rew=float(ep_rew.mean()),
                   stable50=int((stable_max >= STABLE_CRIT['hold']).sum()),
                   stable_med=int(np.median(stable_max)),
                   stable_max=int(stable_max.max()),
                   z_end_mean=float(z_end.mean()),
                   z_end_gt2=int((z_end > 2.0).sum()),
                   crash=n_crash,
                   elapsed=int(time.time() - t0))
        eval_log.append(rec)
        with open(os.path.join(OUT, 'eval_log.json'), 'w') as f:
            json.dump(eval_log, f, indent=2)
        # ---------- TensorBoard：eval 指标 ----------
        model.logger.record("eval/rew_mean", rec['eval_rew'])
        model.logger.record("eval/stable50", rec['stable50'])
        model.logger.record("eval/stable_med", rec['stable_med'])
        model.logger.record("eval/stable_max", rec['stable_max'])
        model.logger.record("eval/z_end_gt2", rec['z_end_gt2'])
        model.logger.record("eval/crash_end", rec['crash'])
        model.logger.record("eval/tilt_cap", cap_deg)
        model.logger.dump(step=done_steps)
        # ---------- 性能门控：当前壳层考核通过才前进（否则原地巩固） ----------
        advanced = False
        if rec['stable50'] >= GATE_STABLE50 and cap_deg < 180.0:
            cap_deg = min(180.0, cap_deg + CAP_STEP)
            advanced = True
        print(f"chunk {chunk} [cap={rec['tilt_cap']:>5.1f}°{'→'+format(cap_deg,'.0f')+'°' if advanced else ''}]: "
              f"steps {done_steps:>9,}  eval_rew={rec['eval_rew']:+9.1f}  "
              f"稳定≥50步 {rec['stable50']}/16  最长稳定 {rec['stable_max']}  "
              f"z_end>2m {rec['z_end_gt2']}/16  坠地终局 {rec['crash']}  [{rec['elapsed']}s]", flush=True)
    print("TRAINING_COMPLETE", flush=True)
