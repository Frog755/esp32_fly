# RL 目录：仿真环境与控制器开发

本项目从**端到端强化学习**起步，经 18+ 轮实验后转向**规则控制器 + 离线最优轨迹**。
本目录保留了全部环境、训练脚本与实验数据，作为"RL 为什么不适用"的完整证据链。

> **结论先行**：PPO 全路线（课程学习/门控残差/BC 锚定/语义动作层/SIL）与 SAC 在 CPU 可及样本量下
> 全部收敛于基座天花板，无增长生态位。当前冠军是纯规则 v5（83.6%），正在做离线轨迹查表（目标 89%）。

---

## 目录结构

```
RL/
├── env/                          # 动力学引擎与任务环境
│   ├── quad_sim_vec.py           # ★ 批量刚体动力学（180k quad-steps/s，对拍 1.8e-12）
│   ├── vec_recovery_env.py       # 任务环境基类（奖励/互补滤波观测器/终止/钩子）
│   ├── vec_expert5_env.py        # ★ 冠军：v5 三段式规则 83.6%
│   ├── vec_expert_env.py         # v3 三段式 80.5%
│   ├── vec_residual_env.py       # PD + 门控残差（RL 路线，60% 天花板）
│   ├── vec_expert_residual_env.py# v3 + 残差 + SIL 零锚定（RL 净负，76.6%）
│   ├── vec_rate_env.py           # 语义动作层 rate 内环（RL，9.8%）
│   ├── quad_sim_fast_v2.py       # 旧标量环境（P1 时代，含 pd_controller 基线）
│   ├── quad_sim_fixed.py         # 基准校验版
│   └── yao_kun_params.py         # 耀坤 K1.0 实机参数台账
│
├── scripts/
│   ├── flip_lto.py               # ★ 单轴离线轨迹优化（对拍解析解 0.9%）
│   ├── flip_lto3d.py             # 三轴严格四元数求解器
│   ├── flip_table2.py            # 电机分组建表管线
│   ├── flip_table.py             # 早期"力矩反解"路线（已废弃）
│   ├── train_expert_sil.py       # v3 基座 + 门控残差 + SIL 零锚定
│   ├── train_residual_anchored.py# AnchoredPPO（PPO + BC 损失）
│   ├── train_rate_anchored.py    # 语义动作层 + SIL
│   ├── train_planA.py            # Plan A（渐进倾角课程）
│   ├── train_sac.py              # SAC 对照（回放池机制验证）
│   ├── collect_anchor_data*.py   # 锚定数据集采集
│   ├── eval_planA_v2.py          # 倾角扫描尸检
│   ├── eval_residual.py          # 残差幅值尸检
│   ├── raptor_policy_np.py       # Raptor 权重解析 + GRU numpy 前向
│   └── test_raptor_in_oursim.py  # Raptor 权重在本项目 sim 的闭环测试
│
├── models/                       # checkpoints 与查表数据
│   ├── expert_sil/               # Stage 11 (v3+RL) 全程 50 个 ckpt
│   ├── residual_v1/              # 门控残差（model_2M 是 60.5% 那个）
│   ├── rate_sil/                 # 语义动作层 SIL
│   ├── flip_table_axis_x.npz     # 42 格单轴表（已证不合格）
│   ├── anchor_dataset*.npz       # 锚定数据集
│   └── p1_*/ p2_*/ planA_*/      # 历史各轮
│
├── docs/                         # 实验报告
│   ├── 交接说明-Stage13查表控制器.md   # ★ 新对话必读
│   ├── 实验总览.md                    # 全部实验总表
│   ├── 奖励函数迭代实验日志.md         # 逐轮过程
│   ├── PlanA完整历程与第一轮训练报告.md # RL 路线尸检
│   └── v18~v21 诊断报告
│
├── eval.py / eval_phase.py / eval_by_height.py / eval_hierarchical.py
├── eval_pd_e2e.py                # 纯 PD 端到端基线（96/87.4/74.5/49.1%）
├── compare_tb.py / monitor_training.py / report.py / play.py
├── export_policy_c.py            # 权重 → C 头文件
└── train_ppo.py                  # 旧版训练入口（P1/P2 时代）
```

---

## 核心数据

| 策略 | 总分 | ω≥14 | ≥135° |
|---|---|---|---|
| 纯 PD | 60.2% | 40.0% | 15.2% |
| model_2M（RL 残差） | 60.5% | 42.7% | 15.2% |
| v3 三段式 | 80.5% | 72.0% | 54.3% |
| **v5 bang-bang** | **83.6%** | **76.0%** | **67.4%** |
| expert_sil（v3+RL） | 76.6% | 62.7% | 47.8% |

512 通道正式判据：成功 = 从未坠地 且 稳定悬停 ≥50 步。

---

## 关键脚本用法

```bash
# 单轴求解器三道 sanity（对拍解析解 / 扫 tf / 实跑验证）
python3 scripts/flip_lto.py

# 电机分组参数化建表测试
python3 scripts/flip_table2.py

# 评估 v5 冠军（512 通道正式判据）
python3 -c "
import sys; sys.path.insert(0,'env')
import numpy as np, torch
from vec_expert5_env import VecExpert5
from yao_kun_params import YaoKunK10Params
env = VecExpert5(num_envs=256, seed=2024, episode_s=5.0,
                 params=YaoKunK10Params(), tilt_cap_deg=180.0)
env.reset()
smax=np.zeros(256); crash=np.zeros(256,bool)
for t in range(1000):
    obs,r,done,infos = env.step(np.zeros((256,4)))
    smax=np.maximum(smax,[i['stable_cnt'] for i in infos])
    crash|=[i['crash'] for i in infos]
print('成功率', ((~crash)&(smax>=50)).mean())
"
```

---

## 环境依赖

numpy 2.x / scipy 1.15+ / stable-baselines3 2.9.2 / torch (CPU) / gymnasium / tensorboard
（`pip install --user scipy==1.15.3` 若系统 scipy 过旧会与 numpy 2.x 冲突）
