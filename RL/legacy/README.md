# RL/legacy —— 历史实验脚本（Stage 20 归档）

这里是项目走到 v9 之前尝试过、但**已不在主线上**的脚本：强化学习路线、Stage 13 翻转查表方案、Raptor 对照、一次性诊断。
保留它们是为了能复现当时的结论（"端到端 RL 为什么不适用"的证据链）。当前主线见 [../README.md](../README.md)。

## 分组

| 目录 | 内容 | 原位置 |
|---|---|---|
| `rl_v0_ppo/` | 最早一代 PPO（70g 机型、标量仿真 quad_sim*、TensorBoard 工具链） | RL/ 根目录、env/ |
| `rl_vec/` | 向量化环境上的各轮 RL：Plan A、SAC、残差 PD+RL、语义动作层、SIL 锚定 | scripts/、env/ |
| `flip_table/` | Stage 13 离线最优翻转轨迹 + 查表控制器（已被 v6~v9 闭环规则取代） | scripts/ |
| `raptor/` | Raptor 外部基础策略的复现与在本项目仿真里的对照 | scripts/ |
| `diagnose/` | 各轮一次性诊断脚本 | scripts/、RL/ 根目录 |

另有：`OLD_RL_README.md` = 迁移前的 `RL/README.md`（RL 时代的目录说明，原文保留）；`rl_v0_ppo/run_baseline.sh` = 原 `RL/run_baseline.sh`。

## 迁移只改了路径，没改逻辑

每个 `.py` 开头加了一段"legacy 迁移引导"：
- `_LEGACY_ORIG_DIR` 指回文件**原来**所在目录，源码里的 `os.path.dirname(os.path.abspath(__file__))` 都换成了它，
  所以 `../env`、`../models`、`../results` 这类相对路径的含义和迁移前完全一样（结果仍读写到 `RL/models/`、`RL/results/`）；
- 把 `RL/`、`RL/env`、`RL/scripts` 和每个 legacy 子目录加进 `sys.path`，跨目录的旧导入（如 `from eval import ...`）照常可用。

迁移时逐个核对过（`RL/results/stage20/`）：46 个文件迁移前后加载结果、模块级路径变量、加载到的本地模块**全部一致**；
只读脚本 `flip_lto.py`、`flip_table2.py`、`eval_table_vs_v5.py`、`test_raptor_in_oursim.py`、`raptor_policy_np.py` 以 `__main__` 实跑通过。
**训练 / 采集 / 写文件类脚本没有实跑**（会覆盖 `RL/models/` 里已有的 checkpoint），只做了加载核对。

## 怎么运行

和迁移前一样，**在 `RL/` 目录下运行**，只是路径变了，例如：

```bash
cd RL
python3 legacy/rl_v0_ppo/train_ppo.py --help
python3 legacy/flip_table/flip_lto.py
bash legacy/rl_v0_ppo/run_baseline.sh        # 已改为自动定位 RL/ 根目录
```

`raptor/raptor_ref.py` 的默认参数 `--policy ../data/raptor_policy.h` 是相对**当前工作目录**的（原本假设在 `RL/scripts/` 下运行），
现在请显式传：`python3 legacy/raptor/raptor_ref.py --policy data/raptor_policy.h`（在 `RL/` 下运行）。

## 迁移前就无法加载的 4 个脚本（与迁移无关，未修）

| 脚本 | 原因 |
|---|---|
| `rl_vec/eval_residual.py`、`rl_vec/eval_planA_v2.py` | 模块顶层直接读 `sys.argv[1]`，不带参数就 IndexError（带参数运行正常的话不受影响） |
| `diagnose/diagnose.py` | 导入早已不存在的 `quad_sim_fast` 模块 |
| `diagnose/baseline_recovery.py` | 依赖未安装的 `l2f` 包（learning-to-fly 的 Python 接口） |

另：旧标量仿真 `quad_sim_fixed.py` **没有**迁移，仍在 `RL/env/`——主线的 `quad_sim_fast_v2.py` 在 `reset()` 里懒加载它的 `quat_mul`。

## 旧路径 → 新路径对照

`RL/docs/` 下的历史实验报告里写的仍是旧路径，按下表对应：

| 旧路径 | 新路径 |
|---|---|
| `RL/train_ppo.py` | `RL/legacy/rl_v0_ppo/train_ppo.py` |
| `RL/eval.py` | `RL/legacy/rl_v0_ppo/eval.py` |
| `RL/play.py` | `RL/legacy/rl_v0_ppo/play.py` |
| `RL/eval_phase.py` | `RL/legacy/rl_v0_ppo/eval_phase.py` |
| `RL/eval_hierarchical.py` | `RL/legacy/rl_v0_ppo/eval_hierarchical.py` |
| `RL/eval_by_height.py` | `RL/legacy/rl_v0_ppo/eval_by_height.py` |
| `RL/report.py` | `RL/legacy/rl_v0_ppo/report.py` |
| `RL/compare_tb.py` | `RL/legacy/rl_v0_ppo/compare_tb.py` |
| `RL/monitor_training.py` | `RL/legacy/rl_v0_ppo/monitor_training.py` |
| `RL/export_policy_c.py` | `RL/legacy/rl_v0_ppo/export_policy_c.py` |
| `RL/env/quad_sim.py` | `RL/legacy/rl_v0_ppo/quad_sim.py` |
| `RL/env/quad_sim_fast_v2_v28_backup.py` | `RL/legacy/rl_v0_ppo/quad_sim_fast_v2_v28_backup.py` |
| `RL/scripts/train_planA.py` | `RL/legacy/rl_vec/train_planA.py` |
| `RL/scripts/train_sac.py` | `RL/legacy/rl_vec/train_sac.py` |
| `RL/scripts/train_residual.py` | `RL/legacy/rl_vec/train_residual.py` |
| `RL/scripts/train_residual_anchored.py` | `RL/legacy/rl_vec/train_residual_anchored.py` |
| `RL/scripts/train_residual_omegafirst.py` | `RL/legacy/rl_vec/train_residual_omegafirst.py` |
| `RL/scripts/train_rate.py` | `RL/legacy/rl_vec/train_rate.py` |
| `RL/scripts/train_rate_anchored.py` | `RL/legacy/rl_vec/train_rate_anchored.py` |
| `RL/scripts/train_expert_sil.py` | `RL/legacy/rl_vec/train_expert_sil.py` |
| `RL/scripts/collect_anchor_data.py` | `RL/legacy/rl_vec/collect_anchor_data.py` |
| `RL/scripts/collect_anchor_data_expert.py` | `RL/legacy/rl_vec/collect_anchor_data_expert.py` |
| `RL/scripts/eval_residual.py` | `RL/legacy/rl_vec/eval_residual.py` |
| `RL/scripts/eval_planA_v2.py` | `RL/legacy/rl_vec/eval_planA_v2.py` |
| `RL/scripts/export_tb_plots.py` | `RL/legacy/rl_vec/export_tb_plots.py` |
| `RL/env/vec_residual_env.py` | `RL/legacy/rl_vec/vec_residual_env.py` |
| `RL/env/vec_rate_env.py` | `RL/legacy/rl_vec/vec_rate_env.py` |
| `RL/env/vec_expert_residual_env.py` | `RL/legacy/rl_vec/vec_expert_residual_env.py` |
| `RL/scripts/flip_lto.py` | `RL/legacy/flip_table/flip_lto.py` |
| `RL/scripts/flip_lto3d.py` | `RL/legacy/flip_table/flip_lto3d.py` |
| `RL/scripts/flip_table.py` | `RL/legacy/flip_table/flip_table.py` |
| `RL/scripts/flip_table2.py` | `RL/legacy/flip_table/flip_table2.py` |
| `RL/scripts/build_flip_table3d.py` | `RL/legacy/flip_table/build_flip_table3d.py` |
| `RL/scripts/export_table_c.py` | `RL/legacy/flip_table/export_table_c.py` |
| `RL/scripts/eval_table_vs_v5.py` | `RL/legacy/flip_table/eval_table_vs_v5.py` |
| `RL/scripts/raptor_ref.py` | `RL/legacy/raptor/raptor_ref.py` |
| `RL/scripts/raptor_policy_np.py` | `RL/legacy/raptor/raptor_policy_np.py` |
| `RL/scripts/test_raptor_in_oursim.py` | `RL/legacy/raptor/test_raptor_in_oursim.py` |
| `RL/scripts/diagnose.py` | `RL/legacy/diagnose/diagnose.py` |
| `RL/scripts/diagnose_e2e_on_handoff.py` | `RL/legacy/diagnose/diagnose_e2e_on_handoff.py` |
| `RL/scripts/diagnose_handoff_split.py` | `RL/legacy/diagnose/diagnose_handoff_split.py` |
| `RL/scripts/diagnose_handoff_timing.py` | `RL/legacy/diagnose/diagnose_handoff_timing.py` |
| `RL/scripts/diagnose_omega_curve.py` | `RL/legacy/diagnose/diagnose_omega_curve.py` |
| `RL/scripts/diagnose_pd_sweep.py` | `RL/legacy/diagnose/diagnose_pd_sweep.py` |
| `RL/scripts/baseline_recovery.py` | `RL/legacy/diagnose/baseline_recovery.py` |
| `RL/eval_pd_e2e.py` | `RL/legacy/diagnose/eval_pd_e2e.py` |
