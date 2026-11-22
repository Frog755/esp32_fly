# 策略 A/B 对比报告

> 本报告由 `report.py` 生成，所有数据来自 `eval.py` 在**同一套固定种子集**上的评估。

## 评估条件一致性

| 检查项 | 状态 |
|---|---|
| 种子集一致 | ✅ 一致 |
| 回合数一致 | ✅ 一致 |
| 种子集 | `[20261029, 20262029)` |
| 每档回合数 | 1000 |

## 成功率对比（含 95% 置信区间）

| 难度 | v6b_EA2 | hierarchical |
|---|---|---|
| 0.3 | **61%** (611/1000)<br><sub>58~64%</sub> | **63%** (631/1000)<br><sub>60~66%</sub> |
| 0.5 | **56%** (565/1000)<br><sub>53~60%</sub> | **51%** (509/1000)<br><sub>48~54%</sub> |
| 0.7 | **44%** (442/1000)<br><sub>41~47%</sub> | **38%** (385/1000)<br><sub>36~42%</sub> |
| 1.0 | **28%** (277/1000)<br><sub>25~31%</sub> | **24%** (243/1000)<br><sub>22~27%</sub> |

## 提升幅度（`hierarchical` vs `v6b_EA2`）

| 难度 | 基线 | 新的 | 差值 | 判读 |
|---|---|---|---|---|
| 0.3 | 61% | 63% | **+2pp** | ≈ 无变化 |
| 0.5 | 56% | 51% | **-6pp** | ⚠️ 退化但不显著 |
| 0.7 | 44% | 38% | **-6pp** | ⚠️ 退化但不显著 |
| 1.0 | 28% | 24% | **-3pp** | ≈ 无变化 |

> `pp` = percentage point（百分点）。置信区间不重叠才算「显著」—— 这是判断改进是否真实有效的标准做法。

## 失败分类对比（每档难度失败次数）

### 难度 d=0.3

| 结局类别 | v6b_EA2 | hierarchical |
|---|---|---|
| `success` | 611 ⭐ | 631 ⭐ |
| `crash_not_upright` | 127 | 5 |
| `crash_upright` | 74 | 10 |
| `timeout_not_upright` | 0 | 0 |
| `timeout_unstable` | 135 | 57 |
| `out_of_bounds` | 53 | 297 |

### 难度 d=0.5

| 结局类别 | v6b_EA2 | hierarchical |
|---|---|---|
| `success` | 565 ⭐ | 509 ⭐ |
| `crash_not_upright` | 200 | 10 |
| `crash_upright` | 67 | 13 |
| `timeout_not_upright` | 0 | 3 |
| `timeout_unstable` | 109 | 47 |
| `out_of_bounds` | 59 | 418 |

### 难度 d=0.7

| 结局类别 | v6b_EA2 | hierarchical |
|---|---|---|
| `success` | 442 ⭐ | 385 ⭐ |
| `crash_not_upright` | 314 | 30 |
| `crash_upright` | 69 | 23 |
| `timeout_not_upright` | 0 | 5 |
| `timeout_unstable` | 99 | 50 |
| `out_of_bounds` | 76 | 507 |

### 难度 d=1.0

| 结局类别 | v6b_EA2 | hierarchical |
|---|---|---|
| `success` | 277 ⭐ | 243 ⭐ |
| `crash_not_upright` | 512 | 90 |
| `crash_upright` | 80 | 62 |
| `timeout_not_upright` | 0 | 6 |
| `timeout_unstable` | 61 | 36 |
| `out_of_bounds` | 70 | 563 |

## 回正耗时对比（中位数，秒）

| 难度 | v6b_EA2 | hierarchical |
|---|---|---|
| 0.3 | 0.41 | — |
| 0.5 | 0.61 | — |
| 0.7 | 0.81 | — |
| 1.0 | 0.94 | — |

## 自动判读

在最高难度 d=1.0 上：
- `v6b_EA2`：28% (277/1000)
- `hierarchical`：24% (243/1000)

`hierarchical` 在极限档的首要失败模式是 `out_of_bounds`（563 次）。
→ 下一步优先针对这个失败类别调整奖励权重。

