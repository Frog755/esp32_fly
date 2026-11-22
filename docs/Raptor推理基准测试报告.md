# Raptor 策略 ESP32-S3 推理基准测试报告

测试日期：2026-10-29
测试平台：ESP32-S3 QFN56 rev v0.2，双核 240MHz，16MB Flash，8MB PSRAM
ESP-IDF：v6.1

---

## 一、结论先说

**你师兄的「ESP32 可能跑不动」这个担心可以彻底排除。**

单次推理 **189 μs**，跑 100Hz 只占用 **1.9%** CPU 时间。算力余量 **52 倍**。

| 指标 | 实测值 |
|---|---|
| 单次前向 | 189.4 μs |
| 理论最高频率 | 5,279 Hz |
| 实测连续吞吐 | 5,259 Hz |
| 跑 100Hz 的 CPU 占用 | 1.9% |
| 跑 200Hz 的 CPU 占用 | 3.8% |

---

## 二、网络结构（从 raptor_policy.h 实际解析）

```
输入 22 维 [pos(3), R展平(9), lin_vel(3), ang_vel(3), prev_action(4)]
  ↓
layer_0: Dense 22 → 16, ReLU          368 参数
  ↓
layer_1: GRU   16 → 16, 门序 [r|z|n] 1648 参数
  ↓
layer_2: Dense 16 → 4, Identity        68 参数
  ↓
输出 4 维动作（无 squash，值域可超出 [-1,1]）
```

**总参数 2084 float = 8,336 字节 (8.1 KB)**

### GRU 更新式（RLtools 实现，与 PyTorch 不同）

```
Wi = W_in·a + b_in          (48 维)
Wh = W_hid·h + b_hid        (48 维)
rz = sigmoid(Wi[0:32] + Wh[0:32])
r  = rz[0:16],  z = rz[16:32]
n  = tanh(r ⊙ Wh[32:48] + Wi[32:48])     ← r 只乘 hidden 分支
h' = z⊙h + (1-z)⊙n
```

⚠️ **移植最容易错的三点**：
1. 门序是 `[r | z | n]`，不是常见的 `[z | r | n]`
2. `r` 和 `z` 一起过 sigmoid
3. `n` 中 `r` 只作用在 `Wh` 分支上，不是乘整个和

---

## 三、测试结果详情

### [1] 数值正确性 —— 12/12 通过

C 实现与 numpy 参考实现逐位比对，**最大误差 4.768e-07**（单精度浮点累加误差量级，符合预期）。

| 测试项 | 通过/总数 |
|---|---|
| 单步向量（9 组，含 1 组静止悬停基准） | 9/9 |
| 连续 3 步（验证 GRU 状态跨步传递） | 3/3 |

### [2] 单次前向耗时

```
n=2000  avg=189.43 μs  min=189.00 μs  max=196.00 μs  jitter=7.00 μs
```

抖动仅 7μs，非常稳定。

### [3] 连续推理吞吐

```
2 秒完成 10519 次 → 5259 Hz，平均 190.15 μs/次
```

### [4] 并发时序抖动（RL 100Hz + PID 1kHz）

```
RL 推理（与 PID 并发时）: avg=189.96 μs  max=202.00 μs  jitter=13 μs
PID 周期偏差（理想 1000us）: 平均 3.10 μs，最大 9.00 μs
```

**关键结论**：PID 跑 1kHz 时周期偏差最大只 9μs（不到 1%），说明 RL 推理完全不会干扰底层控制环。

### [5] 内存占用

| 项目 | 值 |
|---|---|
| 权重（Flash 常量） | 8,336 字节 |
| 推理栈占用 | < 1 KB（全静态，无 malloc） |
| 内部 RAM 空闲 | 389,775 字节 |
| PSRAM 空闲 | 8,386,156 字节 |

---

## 四、本轮修掉的一个真实 bug

**现象**：首轮测试单步 9/9 全 PASS（误差 3e-7），但连续 3 步全 FAIL（误差 6e-2）。

**根因**：`export_policy_c.py` 中连续步测试用了 numpy 引用共享：

```python
# 错误写法
seq_obs.append(o)     # 存的是引用
a = policy.step(o)
o[18:22] = a          # 改了同一块内存，seq_obs[i] 也被污染
```

导致烧进板子里的观测是被污染的版本，而期望值是用原始观测算的，自然对不上。

**修复**：改用 `.copy()` 快照：

```python
o_snapshot = o.copy()
seq_obs.append(o_snapshot)
```

修复后 12/12 全过。

> 注意：这个问题**只在连续步测试中出现**，单步测试因为每组都新建 obs 且立即计算，没有受影响。C 实现的数学逻辑从一开始就是对的。

---

## 五、对分层控制架构的建议（已被实测数据支持）

| 层级 | 建议频率 | 实测依据 |
|---|---|---|
| RL 策略层 | 100 Hz | 189μs/次，占 1.9% CPU，余量 52 倍 |
| 姿态 PID | 1 kHz | 并发时周期偏差最大 9μs，稳定 |
| 混控/PWM | 1-4 kHz | 用 MCPWM 外设，CPU 占用极低 |

**架构可行，数据支撑充分。**

---

## 六、文件清单

| 文件 | 说明 |
|---|---|
| `main/raptor_weights.h` | 权重 + 校验向量（自动生成，46KB） |
| `main/raptor_policy.h` | 纯 C 前向实现（静态分配，无 malloc） |
| `main/benchmark_main.c` | 基准测试主程序 |
| `scripts/export_policy_c.py` | 权重导出脚本（从 raptor_policy.h 生成 C 头） |

### 重新生成 / 跑测试

```bash
# 导出权重（源策略在另一个项目目录）
cd /home/frog/embodied_ai/esp32_fly/code/connection_test/scripts
~/.venv/bin/python3 export_policy_c.py \
  --policy ~/ai_workspace/deepseek_workspace/esp32-rl-无人机回正/data/raptor_policy.h \
  --out ../main/

# 编译烧录
cd ..
source ~/.espressif/tools/activate_idf_v6.1.sh
idf.py -p /dev/ttyACM0 build flash monitor
```

---

## 七、待办

- [ ] int8 量化版本对比测试（你要求两种都测，目前只做了 float32）
- [ ] 加上真实传感器读取的端到端延迟测量
- [ ] 硬件到位后做实物闭环测试
