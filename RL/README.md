# RL 目录：v9 规则控制器的仿真、调参与验证

> 目录名沿用历史（项目最初走强化学习路线）。**当前主线不做 RL 训练，也不是 MPC**：
> v9 是手写的规则控制器 —— 翻转段按物理推出的时间最优 bang-bang，接近正立后交给 PD；每步只是一组固定公式，没有网络、没有在线优化。
> 所谓"训练" = **物理分析写控制律 + 512 通道仿真批量扫参数**，按成功率 / 掉高 / 耗时打分选最优。
> RL 时代的 PPO、奖励函数、残差网络等脚本已归档到 [legacy/](legacy/README.md)（Stage 20）。

## v9 主线链路

```
env/yao_kun_params.py          实机物理参数（质量、推力、惯量、电机时间常数）
        ↓
env/quad_sim_vec.py            批量刚体动力学，一次推进 512 架
        ↓
env/vec_recovery_env.py        环境外壳：随机初始姿态、气压计观测器、坠地判定、稳定悬停计数
        ↓                      （里面也算 RL 奖励，但主线评测传入的动作全为 0、没有任何代码读奖励）
env/vec_expert9_env.py         ★ v9 控制律本体，全部参数在 DEFAULT_CTRL
        ↓
scripts/eval_recovery_metrics.py   打分：成功率、掉高、翻正耗时、稳悬耗时
        ↓
scripts/sweep_v9.py            多进程参数扫描 → 最终参数 results/stage17/v9_final_ctrl.json
        ↓
验证   scripts/final_compare_v9.py   留出种子（防过拟合）
       scripts/robust_v9.py          模型参数失配
       scripts/eval_noise*.py        传感器噪声（env/noise_wrap.py）
       scripts/eval_att_est.py       真实姿态估计器（env/att_est_wrap.py）
        ↓
scripts/visualize_v8_v9.py     MuJoCo 同工况对比
        ↓
code/connection_test/main/recovery_expert9.h + mahony_gated.h   C 实现
        ↓   scripts/verify_expert9_c_equivalence.py 逐项校验（Stage 21）
        ↓   scripts/gen_hil_vectors.py → code/bench_v9/ 上板一致性与耗时（Stage 22）
code/sensor_test/   真机传感器 + 六面标定 calib.sh + 姿态检查 attitude.sh（Stage 23）
[待做] 拆桨电机测试（映射 / 转向 / 偏航符号）→ 完整飞控固件 → 台架自检
```

## 目录

```
RL/
├── env/
│   ├── yao_kun_params.py         实机参数
│   ├── quad_sim_vec.py           ★ 批量动力学
│   ├── quad_sim_fast_v2.py       旧标量仿真；主线只用其中的 QuadParams 参数类
│   ├── quad_sim_fixed.py         旧标量仿真校验版（quad_sim_fast_v2 懒加载其 quat_mul，故留在主线）
│   ├── vec_recovery_env.py       环境外壳
│   ├── vec_expert9_env.py        ★ v9（99.6%，C 版 recovery_expert9.h）
│   ├── vec_expert8_env.py        v8 对照版本（94.5%，有 C 头文件）
│   ├── vec_expert_env.py         v3 ┐
│   ├── vec_expert4~7_env.py      v4~v7 ┘ 历代规则控制器（对照）
│   ├── vec_table_expert_env.py   Stage 13 查表控制器（总榜脚本仍在用，故留在主线）
│   ├── noise_wrap.py             传感器噪声包装
│   └── att_est_wrap.py           IMU + Mahony 姿态估计器包装
├── scripts/
│   ├── eval_recovery_metrics.py  掉高 / 耗时专项评测
│   ├── sweep_v9.py               v9 参数扫描
│   ├── final_compare_v9.py       v8 vs v9 留出种子对比
│   ├── robust_v9.py              模型失配测试
│   ├── eval_noise.py / eval_noise_att.py / eval_att_est.py   噪声与估计器复测
│   ├── eval_all_controllers.py   历代控制器 512 通道总榜
│   ├── verify_expert9_c_equivalence.py   ★ v9 控制器 + 门控 Mahony 的 C 实现全量校验
│   ├── verify_expert{6,7,8}_c_equivalence.py   前代 C 头文件 vs Python 逐位校验
│   ├── gen_hil_vectors.py        生成上板测试向量 code/bench_v9/main/hil_vectors.h（Stage 22）
│   ├── visualize_v8_v9.py        ★ v8 vs v9 同工况 3D 对比
│   ├── visualize_recovery.py     旧版 3D 查看器（v5~v8）
│   └── generate_mujoco_visualizations.py   旧版批量导出视频
├── legacy/                       历史实验（RL 训练、查表、Raptor、诊断），逻辑未改，仍可运行
├── results/                      各阶段原始结果（stage17 调参、stage18 噪声、stage19 可视化、stage20 迁移核对、stage21 C 校验、stage22 上板、stage23 真机传感器）
├── models/                       checkpoints 与查表数据（大多不入库）
├── data/                         Raptor 参考权重
└── docs/                         历史实验报告（文中路径为迁移前路径，对照见 legacy/README.md）
```

## 常用命令（在 `RL/scripts/` 下）

```bash
python3 eval_recovery_metrics.py --ctrl vec_expert9_env:VecExpert9 vec_expert8_env:VecExpert8   # v9 / v8 打分
python3 eval_all_controllers.py                # 历代总榜
python3 final_compare_v9.py                    # 8 种子最终对比
python3 eval_att_est.py                        # 真实姿态估计器复测
python3 visualize_v8_v9.py                     # 3D 对比（或项目根目录 ./run_visualizer.sh v8v9）
python3 verify_expert9_c_equivalence.py        # v9 + 门控 Mahony C 实现全量校验（--quick 快速版）
python3 verify_expert8_c_equivalence.py        # v8 C 头文件校验
```

改 v9 参数：编辑 `env/vec_expert9_env.py` 里的 `DEFAULT_CTRL`，或在评测时传 `env_kwargs=dict(ctrl={...})`（`sweep_v9.py` 就是这么做的）。
**改完参数必须同步改 `recovery_expert9.h` 的 `#define`**，再跑 `verify_expert9_c_equivalence.py`——它的 A 项会逐个比对常数，不一致直接报错。
