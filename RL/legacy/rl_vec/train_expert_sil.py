#!/usr/bin/env python3
"""train_expert_sil.py —— Stage 11 主线: v3 专家基座 + 门控残差 + SIL 零锚定

起点=80.5%（a≡0 时逐位等于 v3 专家，冒烟已验）；锚定把"专家会做的别插手"焊死；
RL 的全部任务 = 攻专家盲区（高初始转速的动量管理，ω≥14 带残余 28% 失败）。
预期形态: 曲线从 ~52/64(=80.5%×64) 起步，锚定使其不会下跌，峰值向 58-64/64 爬。
用法: python scripts/train_expert_sil.py [total_steps] [out_name] [bc_coef]
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
import sys, os, time, json
HERE = _LEGACY_ORIG_DIR
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
sys.path.insert(0, HERE)
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecFrameStack
from vec_expert_residual_env import VecExpertResidualEnv
from yao_kun_params import YaoKunK10Params
from train_residual_anchored import AnchoredPPO

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else 'expert_sil'
BC_COEF = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
DATA = os.path.join(HERE, '..', 'models', 'anchor_dataset_expert.npz')
OUT = os.path.join(HERE, '..', 'models', OUT_NAME)
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    torch.set_num_threads(2)
    d = np.load(DATA)
    print(f"SIL 零锚定数据集: {len(d['bc_obs'])} 对  λ_BC={BC_COEF}", flush=True)

    base = VecExpertResidualEnv(num_envs=256, seed=0, episode_s=5.0, params=YaoKunK10Params())
    venv = VecFrameStack(base, n_stack=8)
    ebase = VecExpertResidualEnv(num_envs=64, seed=777, episode_s=5.0, params=YaoKunK10Params())
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
        crash_mask = np.zeros(64, bool)
        # 正式协议: 不手工 reset（让通道各自 auto-reset），成功=从未坠地且稳定≥50
        for _ in range(1000):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = evenv.step(a)
            ep_rew += r
            for i in range(64):
                stable_max[i] = max(stable_max[i], infos[i]['stable_cnt'])
                crash_mask[i] |= bool(infos[i].get('crash', False))
        n_crash = int(crash_mask.sum())
        s50 = int(((~crash_mask) & (stable_max >= 50)).sum())
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
              f"正式稳定 {s50}/64  最长稳定 {rec['stable_max']}  坠地率 {n_crash}/64  "
              f"best@{best['step']//1_000_000}M  [{rec['elapsed']}s]", flush=True)
    print("TRAINING_COMPLETE", flush=True)
