#!/usr/bin/env python3
"""train_residual_omegafirst.py —— ω-first 课程：先学"刹停旋转"，再学"回正"

靶子（三方终评钉死的瓶颈）: ω0>14 rad/s 开局成功率仅 43%、倒扣×随机转速 15%。
机理: PD 姿态环 kd·ω 在 ω>11 rad/s 时需求力矩超过最大差动力矩 0.0225 —— 刹不住。
     刹停本身是已知物理可行动作（bang-bang 反向力矩），先把这一段学会。

Phase 1（0~20M）: 倾角 ≤30° 正立区 + ω0 ∈ U(10, 30)（超出任务分布，故意超训）
Phase 2（20M~） : 全球面 + ω0 ∈ U(0, 20)（标准任务分布）
评估环境: 始终 = 标准全球面任务（ω0 ≤ 20），直接衡量 Phase 1 技能向任务的迁移。
锚定: 沿用 λ=5（数据集来自 model_2M 成功轨迹；其中含高转速开局的成功样本）。
用法: python scripts/train_residual_omegafirst.py [total_steps] [out_name]
"""
import sys, os, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
sys.path.insert(0, HERE)
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecFrameStack
from vec_residual_env import VecResidualEnv
from yao_kun_params import YaoKunK10Params
from train_residual_anchored import AnchoredPPO

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else 'residual_omegafirst'
PHASE1_STEPS = 20_000_000
DATA = os.path.join(HERE, '..', 'models', 'anchor_dataset.npz')
OUT = os.path.join(HERE, '..', 'models', OUT_NAME)
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    torch.set_num_threads(2)
    d = np.load(DATA)
    print(f"锚定数据集: {len(d['bc_obs'])} 对  λ_BC=5.0", flush=True)

    base = VecResidualEnv(num_envs=256, seed=0, episode_s=5.0, params=YaoKunK10Params())
    venv = VecFrameStack(base, n_stack=8)
    # 评估环境: 固定为标准任务分布（全球面 + ω0≤20），不随训练相位变——直接量迁移
    ebase = VecResidualEnv(num_envs=64, seed=777, episode_s=5.0, params=YaoKunK10Params())
    evenv = VecFrameStack(ebase, n_stack=8)

    model = AnchoredPPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=3,
                        learning_rate=3e-4, gamma=0.999, gae_lambda=0.95,
                        ent_coef=0.001, verbose=0, device="cpu", seed=0,
                        policy_kwargs=dict(log_std_init=-1.0),
                        bc_obs=d['bc_obs'], bc_act=d['bc_act'], bc_coef=5.0,
                        tensorboard_log=os.path.join(OUT, 'tb'))

    CHUNK = 2_000_000
    eval_log, best = [], dict(score=(-1, -1e18))
    t0 = time.time()
    done_steps = 0
    chunk = 0
    while done_steps < TOTAL:
        phase = 1 if done_steps < PHASE1_STEPS else 2
        if phase == 1:
            base.set_tilt_cap(30.0)
            base.set_omega0_range(10.0, 30.0)   # 超出任务分布的刹停特训
        else:
            base.set_tilt_cap(180.0)
            base.set_omega0_range(0.0, 20.0)    # 回到标准任务分布
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=(chunk == 0))
        done_steps += CHUNK
        chunk += 1
        model.save(os.path.join(OUT, f"model_{done_steps//1_000_000}M.zip"))
        # ---- 评估（标准任务分布，64 通道）----
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
        rec = dict(step=done_steps, phase=phase, eval_rew=float(ep_rew.mean()),
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
        print(f"chunk {chunk} [P{phase}]: steps {done_steps:>9,}  eval_rew={rec['eval_rew']:+9.1f}  "
              f"稳定≥50步 {s50}/64  最长稳定 {rec['stable_max']}  坠地 {n_crash}  "
              f"best@{best['step']//1_000_000}M  [{rec['elapsed']}s]", flush=True)
    print("TRAINING_COMPLETE", flush=True)
