# 仓库核实记录（2026-10-26）

## 1. arplaboratory/learning-to-fly —— 论文复现版（已冻结）
- 593 stars / 63 forks / MIT / C++ / 最后 push 2025-05-22，默认分支 `master`
- 论⽂：Learning to Fly in Seconds（RA-L 2024, arXiv 2311.13081）
- README 首行明确：**L2F 已上游到 RLtools**，
  - Python 接口 → `rl-tools/l2f`（pip install l2f）
  - CUDA benchmark → `rl-tools/l2f-benchmark`
  - 环境本体 → `rl-tools/rl-tools` 的 `include/rl_tools/rl/environments/l2f`
  - 训练配方 → RLtools Zoo `src/rl/zoo/l2f/sac.h`，结果在 zoo.rl.tools
  - 本仓库仅用于"在最新 Ubuntu LTS 上复现论文数字"
- 内部实现要点（`include/learning_to_fly/simulator/`）：
  - 4 电机、每电机三个推力系数 `{a,b,c}: thrust = a + b*rpm + c*rpm²`（mrs.h 里 a=-1.7690, b=0.0038361, c=1.3298e-6）
  - 训练机型 MRS：mass=0.73kg，臂长 0.1202m，J=diag(0.0079,0.0079,…)，torque_const=0.016
  - Crazyflie 参数也在 `parameters/dynamics/crazy_flie.h`：mass=0.027kg，臂 0.028m，J=diag(3.85e-6,3.85e-6,5.97e-6)，c=3.16e-10
  - 积分步长 `Integration dt = 0.02`（子步细分在 operations_generic.h），**策略控制频率 100 Hz**（dt=10ms）
  - 动作 4 维（归一化 -1..1 → rpm），观测 = position(3)+orientation(四元数4)+lin_vel(3)+ang_vel(3)+action_history(32×4)
  - 默认初始化 `all_around_2`：姿态扰动 ±3.14 rad（整球随机）、线速度 1 m/s、角速度 1 rad/s
  - 默认奖励 `reward_squared_position_only_torque`：pos 5 / ori 5 / lin_vel 0.01 / action 0.01 / constant 2 / scale 0.5
  - 终止条件 `fast_learning`：位置 0.6m、线速度/角速度阈值 1000（放得很松，为了样本效率）
- 固件部署：submodule `controller` → arplaboratory/learning_to_fly_controller，编译成 **Crazyflie 2.x 的 cf2.bin**（make cf2_defconfig）。
  - 注意：官方固件 = 整机替换 Crazyflie 固件，**不是外挂 ESP32 方案**
  - 安全提示（原文）：先用手柄 hover button 当 dead man's switch；起飞用原控制器、空中切换策略；yaw 偏移大时会触发猛动作

## 2. rl-tools/raptor.rl.tools —— 只是项目主页
- **不是训练代码**。1 star，JavaScript，无 license，无 README 正文
- 文件构成：`index.html / raptor.js / l2f.js / controller.js / simulator.js / gamepad.html + wasm/{l2f.cpp,l2f_main.cpp,environment.h}`
- 本质：把 L2F 环境 + 策略编译成 **WASM**，在浏览器里用键盘/手柄实时驱动无人机
- 依赖 `./download_dependencies.sh`（拉 three.js / mathjs / stats / jsfive / gl-matrix / rltools.js）、`./build_emscripten.sh`、`./bundle.sh`（esbuild）
- `blob/registry` 由 `src/rl/environments/l2f/export_dynamics.cpp` 生成
- 结论：想跑，用它就行；想训练/部署，去下面三个仓库

## 3. 真正相关的主力仓库（用户没提但必须知道）
### rl-tools/raptor —— 232 stars，"A Foundation Policy for Quadrotor Control"
- Science Robotics 论文；**已经有预训练通用策略**
- `pip install foundation-policy==1.0.1` → `from foundation_policy import Raptor`
- 输入 22 维：position(3) + 旋转矩阵展平(9) + lin_vel(3) + ang_vel(3) + previous_action(4)
- **100 Hz 控制**；可以跑更高频率（用 `CONTROL_INTERVAL_INTERMEDIATE_NS` + `FORCE_SYNC_NATIVE` 每 N 步推进一次 latent）
- 坐标约定 **FLU**（x前 y左 z上），电机顺序 `[FR, BR, BL, FL]`（注意与 PX4 的 [FR,RR,RL,FL] 关系），action∈[-1,1]，`rpm = (max-min)*(a+1)/2+min`
- 部署参考实现已覆盖：**PX4 / Betaflight / Crazyflie / M5StampFly**，在 rl-tools 仓库 `embedded_platforms` 目录
- C++ 接口：`rl_tools/inference/applications/l2f/c_backend.h`，调用 `rl_tools_inference_applications_l2f_control(current_time_ns, &obs, &act)`
- 导出的策略头：`rl-tools/px4-blob/policy.h`，481 KB

### policy.h 实测结构（已下载核实，`data/policy_head.txt` 未存，但已解析）
- 3 层：`dense(22→16, RELU)` → `GRU(16→16, 无 fast_tanh)` → `dense(16→4, IDENTITY)`
- 权重张量 shape：`[16,22]`, `[16]`, 4×`[48,16]`, 2×`[48]`, `[16]`, `[4,16]`, `[4]`
- **参数量约 1.5k 个 float ≈ 6 KB**，加激活/缓冲 < 80 KB
- GRU 门序：`[r | z | n]`，`n = tanh(W_in·x + b_in + r ⊙ (W_hid·h + b_hid))`，`h' = (1-z)⊙n + z⊙h`
  （依据 rl-tools `nn/layers/gru/operations_generic.h` line 300-340）
- 结论：**ESP32-S3 跑这个策略毫无压力，单次前向 100µs 量级**

### rl-tools/esp32 —— ESP-IDF 推理基准
- 已有 `operations/esp32.h`（group_1/2/3）、`nn/layers/dense/operations_esp32/{dsp,opt}.h`
- `DefaultESP32Specification<Hardware::S3 | DEFAULT>`，靠 `esp-dsp` 组件
- 示例 `rl_tools_adapter.cpp`：`rlt::evaluate(device, model, input, output, buffer, rng)`，自带输出校验函数
- 坑：README 明说要**关掉 task watchdog**，否则会被中断重启；示例模型是 MLP，**不是 GRU**，GRU 需自己接 state 推进

### 其他可参考
- rl-tools/hyperdrone（68★，最快的 RL 用无人机仿真器）
- rl-tools/M5StampFly + M5StampFly-firmware（ESP32-S3 掌上无人机，**和用户方案最接近的现成硬件路径**）
- rl-tools/l2f（pip 包，24★）、l2f-benchmark（CUDA）、l2f-studio、zoo.rl.tools
