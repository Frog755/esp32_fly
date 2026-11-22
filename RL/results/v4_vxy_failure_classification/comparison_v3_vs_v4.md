# 策略 A/B 对比报告

> 本报告由 `report.py` 生成，所有数据来自 `eval.py` 在**同一套固定种子集**上的评估。

## 评估条件一致性

| 检查项 | 状态 |
|---|---|
| 种子集一致 | ✅ 一致 |
| 回合数一致 | ✅ 一致 |
| 种子集 | `[20261029, 20261129)` |
| 每档回合数 | 100 |

## 成功率对比（含 95% 置信区间）

| 难度 | v3_reward | v4_vxy_failure_classification |
|---|---|---|
| 0.3 | **53%** (53/100)<br><sub>43~62%</sub> | **45%** (45/100)<br><sub>36~55%</sub> |
| 0.5 | **41%** (41/100)<br><sub>32~51%</sub> | **32%** (32/100)<br><sub>24~42%</sub> |
| 0.7 | **33%** (33/100)<br><sub>25~43%</sub> | **25%** (25/100)<br><sub>18~34%</sub> |
| 1.0 | **24%** (24/100)<br><sub>17~33%</sub> | **22%** (22/100)<br><sub>15~31%</sub> |

## 提升幅度（`v4_vxy_failure_classification` vs `v3_reward`）

| 难度 | 基线 | 新的 | 差值 | 判读 |
|---|---|---|---|---|
| 0.3 | 53% | 45% | **-8pp** | ⚠️ 退化但不显著 |
| 0.5 | 41% | 32% | **-9pp** | ⚠️ 退化但不显著 |
| 0.7 | 33% | 25% | **-8pp** | ⚠️ 退化但不显著 |
| 1.0 | 24% | 22% | **-2pp** | ≈ 无变化 |

> `pp` = percentage point（百分点）。置信区间不重叠才算「显著」—— 这是判断改进是否真实有效的标准做法。

## 失败分类对比（每档难度失败次数）

### 难度 d=0.3

| 结局类别 | v3_reward | v4_vxy_failure_classification |
|---|---|---|
| `success` | 53 ⭐ | 45 ⭐ |
| `crash_not_upright` | 5 | 1 |
| `crash_upright` | 1 | 7 |
| `timeout_not_upright` | 0 | 1 |
| `timeout_unstable` | 13 | 21 |
| `out_of_bounds` | 28 | 25 |

### 难度 d=0.5

| 结局类别 | v3_reward | v4_vxy_failure_classification |
|---|---|---|
| `success` | 41 ⭐ | 32 ⭐ |
| `crash_not_upright` | 9 | 3 |
| `crash_upright` | 3 | 5 |
| `timeout_not_upright` | 0 | 3 |
| `timeout_unstable` | 6 | 19 |
| `out_of_bounds` | 41 | 38 |

### 难度 d=0.7

| 结局类别 | v3_reward | v4_vxy_failure_classification |
|---|---|---|
| `success` | 33 ⭐ | 25 ⭐ |
| `crash_not_upright` | 19 | 14 |
| `crash_upright` | 3 | 6 |
| `timeout_not_upright` | 0 | 1 |
| `timeout_unstable` | 8 | 17 |
| `out_of_bounds` | 37 | 37 |

### 难度 d=1.0

| 结局类别 | v3_reward | v4_vxy_failure_classification |
|---|---|---|
| `success` | 24 ⭐ | 22 ⭐ |
| `crash_not_upright` | 36 | 37 |
| `crash_upright` | 5 | 9 |
| `timeout_not_upright` | 0 | 0 |
| `timeout_unstable` | 1 | 6 |
| `out_of_bounds` | 34 | 26 |

## 回正耗时对比（中位数，秒）

| 难度 | v3_reward | v4_vxy_failure_classification |
|---|---|---|
| 0.3 | 0.49 | 0.50 |
| 0.5 | 0.56 | 0.60 |
| 0.7 | 0.63 | 0.67 |
| 1.0 | 0.66 | 0.82 |

## 自动判读

在最高难度 d=1.0 上：
- `v3_reward`：24% (24/100)
- `v4_vxy_failure_classification`：22% (22/100)

`v4_vxy_failure_classification` 在极限档的首要失败模式是 `crash_not_upright`（37 次）。
→ 下一步优先针对这个失败类别调整奖励权重。

