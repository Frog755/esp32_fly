# AGENTS.md —— 给在本仓库工作的 AI 编码助手

> 新对话先读：本文件 → [.devlog/STATE.md](.devlog/STATE.md)（当前状态与下一步）→ 需要细节时再查 [.devlog/DEVLOG.md](.devlog/DEVLOG.md)。
> 与用户沟通、文档、注释、提交信息一律用**中文**。最后更新：2026-11-24（Stage 27 准备）。

## 1. 项目是什么

耀坤 K1.0（立创 ESP32-S3R8N8，80 g，1S 电池，8520 空心杯 ×4）四旋翼：**任意姿态抛出 → 空中回正 → 悬停 → 自动降落**。

- 控制器是**手写规则控制器 v9**（bang-bang 翻转 + PD 悬停，"训练" = 物理分析 + 512 通道仿真扫参）。**不是 RL、不是 MPC**——RL 路线已被项目自身否定（见 `note.md`、`RL/legacy/`），不要再提议用 RL / PPO 训练。
- 链路：Python 仿真（`RL/env/vec_expert9_env.py`）→ 纯 C（`code/connection_test/main/recovery_expert9.h` + `mahony_gated.h`）→ 正式飞控固件 `code/flight_v9/`。
- ⚠ **推力模型未标定，不确定 1~5 倍**（用户看到 40% 有一点离地，但也可能是振动跳动；公开数据与原作者说法支持现模型）。SIL：真机推力 ≥ 3.3 倍模型时 v9 推力下限 0.3·mg 已大于重力 → 降不下来、飞走。**测出推力前装桨不能真飞**。
- 现状：板上默认是**干跑版**（电机驱动未编译进固件，电机不会转）；手机无线控制已完成并经用户手机实测；
  电机驱动版 `build_mc`（油门上限 6%）拆桨台架 3/3 通过（Stage 27，11-24，**本飞控第一次真转电机**）；**从未装桨、从未起飞**。

## 2. 安全规则（最重要）

1. **不得自行让电机转起来。** `FLIGHT_MOTOR_OUTPUT` 默认 0（不编译任何 LEDC 代码，上电即锁低 GPIO16/6/5/15）。改成 1、烧录带电机驱动的固件、或任何会让电机转的测试，都必须先得到用户明确同意，并确认**已拆桨**（首次）或**系留 + 空旷场地**（装桨）。
   电机驱动版必须显式给油门上限 `-DFLIGHT_U_MAX=<0~1>`（缺了编译失败）；电机版固件由用户运行脚本自己烧（`motor_bench.sh` 要求输入「已拆桨」、`vib_check.sh` 要求「已装桨已系留」），助手不要直接烧 `build_mc` / `build_vib`。固件名含油门上限原文（`u_max=0.40f`），脚本烧前核对。
2. 飞机常年**插着电池**（4.1~4.2 V），不是"没电所以安全"。
3. 空中出问题一律**转降落**，只有触地 / 撞击 / 估计器失效 / 急停才停桨（`flight_fsm.h`）。新增保护逻辑时保持这一原则。
4. 不要依赖 USB 串口做飞行控制：电机 12% 油门时 EMI 会触发 `ESP_RST_USB`（芯片复位）。飞行控制走手机 Wi-Fi（`web_link.c`），心跳超时 0.6 s → 降落。
5. 需要用户动手的测试（摆姿态、标定、看转向），**写成用户自己运行、按回车推进的脚本**，不要边在线采数据边口头指挥（用户明确要求过：线上同步无法判断节奏）。

## 3. 硬件事实（已实测，别再猜）

| 项 | 值 |
|---|---|
| 电机 | M4 左前 GPIO16 顺时针 / M1 右前 GPIO6 逆 / M2 右后 GPIO5 顺 / M3 左后 GPIO15 逆（俯视）；LEDC 20 kHz 10 bit，高电平导通 |
| 控制器输出 → 电机 | `out_u[0..3]`（仿真顺序 FR, BR, BL, FL）= **M1 / M2 / M3 / M4**，偏航**不取反**（`motor_out.h`，依据见 STATE.md「电机映射」） |
| IMU | MPU6500 SPI：SCK38 / MOSI18 / MISO21 / CS42（GPIO17 是光流片选，保持高）；±16 g / ±2000 dps，1 kHz |
| IMU 校准 | 加计六面标定 `code/sensor_test/imu_calib.h`（z 零偏 +0.17 g）；芯片→机体 FLU：x=芯片+y、y=芯片−x、z=芯片+z；安装倾斜 `code/flight_v9/level_trim.h`（roll +1.584°、pitch +0.510°） |
| 其他 | BMP280 I2C SDA9 / SCL8 @0x76；电池 ADC GPIO2 ×2 分压；LED 41/7/14；蜂鸣器 39；机头 = M4 与 M1 之间 |
| 坐标 | 机体 FLU（x 机头 / y 左 / z 上）；roll+ = 右侧朝下，pitch+ = 机头朝下，俯视逆时针 = yaw+ |

## 4. 目录地图

```
code/flight_v9/      ★ 正式飞控（ESP-IDF v6.1）——见 code/flight_v9/README.md
  main/flight_core.h   估计 + 飞行状态机 + v9 控制器打包（固件与仿真共用同一份代码）
  main/flight_fsm.h    DISARMED→ARMED→RECOVER/TAKEOFF→HOLD 5 s→LAND→触地停桨 + 全部保护
  main/alt_est.h       竖直三阶互补滤波（z、vz、加计零偏）
  main/web_link.c      Wi-Fi 热点 YK-FLY-xxxx + 网页 + WebSocket；main/web/index.html 手机页面
  main/motor_out.h     电机映射与 LEDC（FLIGHT_MOTOR_OUTPUT=1 才编译）
  main/yk_board.h      板级驱动（由 sensor_test 实测驱动抽出）
  test/                主机单测 host_test.c（62 项）+ 网页脚本测试 page_test.js
  dryrun.sh / level_trim.sh   用户自己运行的检查 / 标定脚本
  motor_bench.sh       Stage 27 拆桨电机台架（用户运行：烧 build_mc → 3 步台架 → 烧回干跑版）
  vib_check.sh         Stage 30 装桨振动检查（用户运行：烧 build_vib 上限 0.30 → 拔 USB 手机起飞 → 读黑匣子分析 → 烧回干跑版）
  audio_rpm.py         Stage 31 电脑麦克风录电机声音 → 转速 → 推力量级（vib_check 默认录音；--selftest 自检）
  FIRST_FLIGHT.md      Stage 29 首飞方案：油门上限 0.95 的依据、检查清单、飞行步骤
  main/acc_lpf.h       Stage 30 加计 10 Hz 低通，只喂门控 Mahony（装桨振动下门控选择偏差 → 姿态误差 7°，回放见 test/vib_replay.c）
  main/blackbox.h      Stage 28 机载黑匣子（PSRAM noinit，复位保留、断电丢失）；bb_read.sh 读出，bb_selftest.py 上板自检（干跑版）
code/sensor_test/    只读传感器固件 + 六面标定 calib.sh + 姿态检查 attitude.sh（imu_calib.h 在这里）
code/bench_v9/       上板一致性与耗时测试固件
code/connection_test/main/  recovery_expert9.h、mahony_gated.h（C 实现，被 flight_v9 引用）
RL/env/              仿真（quad_sim_vec.py 动力学、vec_expert9_env.py v9）
RL/scripts/          评测、C 等价校验、sil_flight_v9.py（固件核心的软件在环仿真）
RL/results/stageNN/  每个阶段的原始结果（只增不改）
RL/legacy/           历史 RL / 查表实验，不再维护
.devlog/             STATE.md 看板、DEVLOG.md 里程碑
```

## 5. 常用命令

```bash
# ESP-IDF 环境（必须在同一条 shell 命令里 source，见 §6）
source ~/.espressif/tools/activate_idf_v6.1.sh 1>build/.idf_env 2>&1; idf.py build
idf.py -p /dev/ttyACM0 flash
# 主机单测（不需要板子）
code/flight_v9/test/run_host_test.sh
# 固件核心软件在环仿真（约 3 分钟；--quick 约 1 分钟）
cd RL/scripts && python3 sil_flight_v9.py [--quick]
# C 实现 vs Python 全量校验
cd RL/scripts && python3 verify_expert9_c_equivalence.py [--quick]
# 1 kHz 控制循环时序（板子运行 flight_v9 时）
~/.espressif/tools/python/v6.1/venv/bin/python code/flight_v9/timing_probe.py 3
```
串口：`/dev/ttyACM0`（USB-Serial-JTAG），打开串口不会复位板子；开机约 8 s（含 2 s 陀螺静止标定）后才有数据。
带 pyserial 的 Python：`~/.espressif/tools/python/v6.1/venv/bin/python`（系统 `python3` 有 numpy 无 serial）。

## 6. 工具环境的坑（agentdock 远程 shell）

- **删除只能用回收站工具**（`recycle_delete`）；命令里出现 `rm`、`2>/dev/null`、`> /...`（包括正则里的 `>/`）会被拦截。需要丢弃输出时重定向到工作目录里的文件。
- 复杂的 Python 补丁：先写成 `_patch_xxx.py` 文件，**下一次调用**再运行，用完放回收站。**不要把"写文件"和"运行/删除它"放在同一批并行调用里**（会先执行后者，已踩过多次）。
- `source activate_idf_v6.1.sh` 放进 bash 脚本里会让脚本直接退出；要在 `run_command` 的命令行里内联 source。
- 长任务（编译、仿真）用后台运行 + 日志文件，再轮询日志；连接偶尔断开，断开后先查文件 / 进程状态再决定是否重跑。
- 改了 `sdkconfig.defaults` 后要把旧 `sdkconfig` 放回收站再 `idf.py reconfigure build`，否则新选项不生效。
- ESP32-S3 开机 MSPI 时序调校会写 PSRAM 物理地址 0 的 128 字节（黑匣子已留 4 KB 保护区）；USB-Serial-JTAG 的 printf 层在电脑 50 ms 没读时会静默丢字，大量数据要用 `usb_serial_jtag_write_bytes` 阻塞写并带校验。
- 不要让笔记本连飞机热点：笔记本 Wi-Fi 是远程连接的通道，断了就失联。热点只能用 `nmcli dev wifi list` 扫描确认（常要扫 3~5 轮）。

## 7. 工作约定

1. **两套实现必须逐项对拍**（Python ↔ C、仿真 ↔ 固件）；任何判据先验证它测的是你以为的东西（`note.md`「判据事故 5 次」）。
2. 固件逻辑尽量写成**无 ESP 依赖的纯 C 头文件**，同一份代码进主机单测和 `sil_flight_v9.py`，固件里只留驱动与任务调度。
3. 每个阶段：结果存 `RL/results/stageNN/`；同步更新 `.devlog/DEVLOG.md`（追加里程碑）、`.devlog/STATE.md`（重写当前状态与下一步）、`README.md`（摘要与目录）；再提交推送。
4. 实测数字写清楚条件与局限，不夸大（例如"仿真结果""干跑版""未起飞"）；推翻旧结论要在文档里明确更正。
5. Git：远程 `https://github.com/Frog755/esp32_fly`（HTTPS，gh 已登录）；仓库本地提交邮箱 `192193058+Frog755@users.noreply.github.com`。提交信息用中文，格式 `Stage NN: 摘要` + 要点列表。
6. 许可：本项目 MIT；`THIRD_PARTY_NOTICES.md` 记 RLtools / RAPTOR（MIT）。耀坤上游 ESP32-Drone 是 GPL-3.0，**只参考参数和引脚，不要复制其代码**。
7. 用户偏好：少说过程、多给结论；需要用户选择时给出明确选项；涉及电机和起飞的每一步先说清风险再等确认。
