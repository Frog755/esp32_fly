# Project DevLog (Milestones & Changes)

> 规则：
> 1. 按“任务/里程碑”聚合记录，禁止记单条 shell 命令或碎片尝试。
> 2. 每个条目必须包含：时间、动机、核心动作、最终影响与关联文件。

---

## [2026-10-30 23:50] v4_vxy 标准评估与失败分类

- **背景/动机**: 对最近训练产出的 v4_vxy checkpoint 做一次可复现的标准四步评估，本轮重点确认失败模式而非修改训练配置。
- **核心操作**:
  1. 使用固定种子 `[20261029, 20261129)`，四档难度各 100 回合，完成 400 回合 deterministic 评估。
  2. 统计成功率、Wilson 95% CI、终局指标和互斥失败类别，并与 v3_reward 做同条件对比。
- **影响/结果**: v4_vxy 成功率为 d=0.3/0.5/0.7/1.0 的 45%/32%/25%/22%。总体失败以 `out_of_bounds`（126/276）为主；极限难度 d=1.0 首要失败切换为 `crash_not_upright`（37/78）。相较 v3_reward，中低难度成功率下降 8~9 个百分点，当前不能判定为有效提升。
- **关联文件**:
  - `eval_result.json`
  - `summary.md`
  - `failure_classification.md`
  - `comparison_v3_vs_v4.md`
