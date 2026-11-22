#!/usr/bin/env python3
"""visualize_recovery.py —— 耀坤 K1.0 微型四旋翼 空中自稳 3D 原生交互测评程序

特性:
  1. 官方原生 MuJoCo 自由视角与全功能面板:
     - 鼠标左键拖动: 360° 自由旋转视角 (Orbit)
     - 鼠标右键拖动: 平移视角 (Pan)
     - 鼠标滚轮: 自由缩放 (Zoom，从 5cm 细看旋翼到 10m 全局纵览)
     - 双击机身: 锁定视角追踪
     - 官方侧边栏 (Tab 展开/折叠): 包含相机选择、线框模式、坐标轴、质心、接触力、碰撞体、传感器曲线等全量选项
  2. 1:1 绝对真实物理尺度参照系 (消除"2米过长"的视觉错觉):
     - 1:1 成人人体模型 (1.75m 身高，站立于 -0.5m 地面，头顶在 +1.25m，手臂上举标定 1.85m 抛掷高度)
     - 1:1 实验台工作桌 (0.75m 高 x 1.2m 长 x 0.6m 宽) 及实验设备
     - 3.0m 彩色分段高度标尺塔 (含 -0.5m 危险地表、0m 起始基准、1.0m 标尺环、2.0m 标尺环、2.5m 目标靶标)
     - 地面精准 1.0m x 1.0m 方格地砖 (数地砖即得实际物理米数)
  3. 真实物理动力学实时解算 (Real-Time Physics):
     - 200Hz 动力学与姿态控制闭环实时解算，非预录视频
     - 支持双机并排实时对决 (左侧 v5 基线坠毁 vs 右侧 v6 冠军平稳悬停)
     - 支持 149° 极限倒扣、15.4 rad/s 高转速、512 题库随机抽取

快捷键指南:
  [空格 / Space]  : 暂停 / 继续动力学仿真 (暂停时仍可 360° 鼠标自由旋转/缩放定格观察)
  [1]             : 载入 149° 极限深倒扣工况 (Env 85)
  [2]             : 载入 15.4 rad/s 极限高角速度工况 (Env 48)
  [3]             : 载入 116° 斜轴滚转工况 (Env 54)
  [N / n]         : 随机切换下一题 (来自 512 道官方基准测试题库)
  [R / r]         : 重新从 t=0 仿真播放当前工况
  [M / m]         : 切换控制器模式 (compare 双机对决 / v6 冠军 / v5 基线)
  [S / s]         : 切换 0.25x 慢动作特写 / 1.0x 正常速度
  [Tab]           : 展开 / 折叠官方原生 MuJoCo 侧边栏控制面板
  [Q / ESC]       : 退出测评查看器
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

from yao_kun_params import YaoKunK10Params
from vec_expert5_env import VecExpert5
from vec_expert6_env import VecExpert6
from vec_expert7_env import VecExpert7
from vec_expert8_env import VecExpert8

# 确保有 DISPLAY
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    if os.path.exists("/tmp/.X11-unix/X0"):
        os.environ["DISPLAY"] = ":0"

MUJOCO_XML = """
<mujoco model="yaokun_k10_interactive_arena">
  <compiler angle="degree" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.45 0.45 0.48" diffuse="0.65 0.65 0.68" specular="0.25 0.25 0.28"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.16 0.22 0.30" rgb2="0.05 0.07 0.10" width="512" height="512"/>
    <!-- 地面网格: 8m x 8m，每格刚好 1.0m x 1.0m -->
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.12 0.14 0.18" rgb2="0.18 0.21 0.26"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.12"/>
    <material name="mat_v6" rgba="0.15 0.85 0.95 1.0" shininess="0.9"/>
    <material name="mat_v5" rgba="0.95 0.25 0.20 1.0" shininess="0.9"/>
    <material name="mat_arm" rgba="0.30 0.30 0.35 1.0"/>
    <material name="prop_front" rgba="0.25 0.95 0.35 0.85"/>
    <material name="prop_rear" rgba="1.0 0.25 0.25 0.85"/>
    <material name="prop_v5" rgba="0.95 0.65 0.15 0.85"/>
    <material name="mat_target" rgba="0.2 0.95 0.4 0.35"/>
  </asset>
  <worldbody>
    <!-- 地面 (-0.5m 坠毁警戒网/地面，8m x 8m，每格严格 1m x 1m) -->
    <geom name="floor" type="plane" size="4 4 0.05" pos="0 0 -0.5" material="grid_mat"/>

    <!-- 0m 参考基准地平面环 (直径 2.0m) -->
    <site name="datum_0m" type="cylinder" size="1.0 0.002" pos="0 0 0.0" rgba="0.7 0.7 0.3 0.2"/>

    <!-- 2.5m 目标悬停靶标平面 (直径 0.8m) -->
    <site name="target_v6" type="cylinder" size="0.4 0.004" pos="0.4 0 2.5" material="mat_target"/>
    <site name="target_v5" type="cylinder" size="0.4 0.004" pos="-0.4 0 2.5" rgba="0.8 0.4 0.2 0.25"/>

    <!-- ==================== 1:1 绝对真实物理尺度参照系 ==================== -->
    <!-- 1. 物理高度标尺柱 (放置在 x=1.1, y=-0.8，高 3.0m，从 -0.5m 到 +2.5m) -->
    <body name="height_ruler" pos="1.1 -0.8 0">
      <geom name="r_base" type="cylinder" size="0.15 0.02" pos="0 0 -0.48" rgba="0.3 0.3 0.3 1"/>
      <!-- -0.5m ~ 0.0m: 红色触地危险段 (0.5m) -->
      <geom name="r_seg_danger" type="cylinder" size="0.02 0.25" pos="0 0 -0.25" rgba="0.9 0.2 0.2 1"/>
      <geom name="r_ring_0m" type="cylinder" size="0.08 0.006" pos="0 0 0.0" rgba="1.0 0.8 0.1 1"/>
      <!-- 0.0m ~ 1.0m: 标尺中下段 (1.0m) -->
      <geom name="r_seg_1m" type="cylinder" size="0.02 0.5" pos="0 0 0.5" rgba="0.85 0.75 0.2 1"/>
      <geom name="r_ring_1m" type="cylinder" size="0.10 0.006" pos="0 0 1.0" rgba="1.0 0.9 0.2 1"/>
      <!-- 1.0m ~ 2.0m: 标尺中高段 (1.0m) -->
      <geom name="r_seg_2m" type="cylinder" size="0.02 0.5" pos="0 0 1.5" rgba="0.2 0.65 0.9 1"/>
      <geom name="r_ring_2m" type="cylinder" size="0.12 0.006" pos="0 0 2.0" rgba="0.2 0.85 1.0 1"/>
      <!-- 2.0m ~ 2.5m: 目标顶段 (0.5m) -->
      <geom name="r_seg_top" type="cylinder" size="0.02 0.25" pos="0 0 2.25" rgba="0.2 0.9 0.4 1"/>
      <geom name="r_ring_2_5m" type="cylinder" size="0.15 0.008" pos="0 0 2.5" rgba="0.1 1.0 0.5 1"/>
    </body>

    <!-- 2. 1:1 成人参考人形 (高 1.75m，脚踩 -0.5m 地面，头顶在 +1.25m，手臂上举标定 1.85m 抛投点) -->
    <body name="human_reference" pos="-1.1 -0.7 -0.5">
      <geom name="h_leg_l" type="capsule" fromto="-0.10 0 0.05  -0.10 0 0.85" size="0.045" rgba="0.22 0.28 0.42 1"/>
      <geom name="h_leg_r" type="capsule" fromto=" 0.10 0 0.05   0.10 0 0.85" size="0.045" rgba="0.22 0.28 0.42 1"/>
      <geom name="h_torso" type="capsule" fromto="0 0 0.85  0 0 1.45" size="0.12" rgba="0.85 0.45 0.22 1"/>
      <geom name="h_head" type="sphere" pos="0 0 1.62" size="0.10" rgba="0.92 0.78 0.68 1"/>
      <geom name="h_arm_l" type="capsule" fromto="-0.18 0 1.40  -0.22 0 0.85" size="0.035" rgba="0.85 0.45 0.22 1"/>
      <geom name="h_arm_r" type="capsule" fromto=" 0.18 0 1.40   0.32 0 1.85" size="0.035" rgba="0.85 0.45 0.22 1"/>
    </body>

    <!-- 3. 实验台工作桌 (高 0.75m，长 1.2m，宽 0.6m) -->
    <body name="lab_desk" pos="-1.1 0.7 -0.5">
      <geom name="desk_top" type="box" size="0.6 0.3 0.02" pos="0 0 0.73" rgba="0.45 0.35 0.28 1"/>
      <geom name="d_leg1" type="cylinder" size="0.02 0.36" pos="-0.55 -0.25 0.36" rgba="0.2 0.2 0.2 1"/>
      <geom name="d_leg2" type="cylinder" size="0.02 0.36" pos=" 0.55 -0.25 0.36" rgba="0.2 0.2 0.2 1"/>
      <geom name="d_leg3" type="cylinder" size="0.02 0.36" pos="-0.55  0.25 0.36" rgba="0.2 0.2 0.2 1"/>
      <geom name="d_leg4" type="cylinder" size="0.02 0.36" pos=" 0.55  0.25 0.36" rgba="0.2 0.2 0.2 1"/>
      <geom name="inst" type="box" size="0.18 0.12 0.05" pos="0 0 0.79" rgba="0.7 0.75 0.8 1"/>
    </body>

    <!-- ==================== 四旋翼机体: v6 冠军控制器 (右侧通道) ==================== -->
    <body name="drone_v6" pos="0.4 0 2.5">
      <freejoint name="v6_root"/>
      <geom name="v6_body" type="box" size="0.025 0.025 0.005" material="mat_v6" mass="0.040"/>
      <geom name="v6_batt" type="box" size="0.015 0.035 0.006" pos="0 0 -0.008" rgba="0.12 0.12 0.15 1.0" mass="0.020"/>
      <geom name="v6_nose" type="box" size="0.006 0.010 0.003" pos="0.025 0 0.005" rgba="1.0 0.8 0.1 1.0"/>
      <geom name="v6_arm0" type="capsule" fromto="0 0 0  0.0417 -0.0417 0" size="0.002" material="mat_arm"/>
      <geom name="v6_arm1" type="capsule" fromto="0 0 0 -0.0417 -0.0417 0" size="0.002" material="mat_arm"/>
      <geom name="v6_arm2" type="capsule" fromto="0 0 0 -0.0417  0.0417 0" size="0.002" material="mat_arm"/>
      <geom name="v6_arm3" type="capsule" fromto="0 0 0  0.0417  0.0417 0" size="0.002" material="mat_arm"/>

      <geom name="v6_m0" type="cylinder" pos=" 0.0417 -0.0417 0.005" size="0.005 0.008" rgba="0.7 0.7 0.7 1" mass="0.005"/>
      <body name="v6_p0_b" pos=" 0.0417 -0.0417 0.012">
        <joint name="v6_p0" type="hinge" axis="0 0 1"/>
        <geom name="v6_prop0" type="box" size="0.0325 0.004 0.001" material="prop_rear"/>
      </body>

      <geom name="v6_m1" type="cylinder" pos="-0.0417 -0.0417 0.005" size="0.005 0.008" rgba="0.7 0.7 0.7 1" mass="0.005"/>
      <body name="v6_p1_b" pos="-0.0417 -0.0417 0.012">
        <joint name="v6_p1" type="hinge" axis="0 0 -1"/>
        <geom name="v6_prop1" type="box" size="0.0325 0.004 0.001" material="prop_front"/>
      </body>

      <geom name="v6_m2" type="cylinder" pos="-0.0417  0.0417 0.005" size="0.005 0.008" rgba="0.7 0.7 0.7 1" mass="0.005"/>
      <body name="v6_p2_b" pos="-0.0417  0.0417 0.012">
        <joint name="v6_p2" type="hinge" axis="0 0 1"/>
        <geom name="v6_prop2" type="box" size="0.0325 0.004 0.001" material="prop_rear"/>
      </body>

      <geom name="v6_m3" type="cylinder" pos=" 0.0417  0.0417 0.005" size="0.005 0.008" rgba="0.7 0.7 0.7 1" mass="0.005"/>
      <body name="v6_p3_b" pos=" 0.0417  0.0417 0.012">
        <joint name="v6_p3" type="hinge" axis="0 0 -1"/>
        <geom name="v6_prop3" type="box" size="0.0325 0.004 0.001" material="prop_front"/>
      </body>
    </body>

    <!-- ==================== 四旋翼机体: v5 基线控制器 (左侧通道，对比模式) ==================== -->
    <body name="drone_v5" pos="-0.4 0 2.5">
      <freejoint name="v5_root"/>
      <geom name="v5_body" type="box" size="0.025 0.025 0.005" material="mat_v5" mass="0.040"/>
      <geom name="v5_batt" type="box" size="0.015 0.035 0.006" pos="0 0 -0.008" rgba="0.12 0.12 0.15 1.0" mass="0.020"/>
      <geom name="v5_nose" type="box" size="0.006 0.010 0.003" pos="0.025 0 0.005" rgba="1.0 0.8 0.1 1.0"/>
      <geom name="v5_arm0" type="capsule" fromto="0 0 0  0.0417 -0.0417 0" size="0.002" material="mat_arm"/>
      <geom name="v5_arm1" type="capsule" fromto="0 0 0 -0.0417 -0.0417 0" size="0.002" material="mat_arm"/>
      <geom name="v5_arm2" type="capsule" fromto="0 0 0 -0.0417  0.0417 0" size="0.002" material="mat_arm"/>
      <geom name="v5_arm3" type="capsule" fromto="0 0 0  0.0417  0.0417 0" size="0.002" material="mat_arm"/>

      <geom name="v5_m0" type="cylinder" pos=" 0.0417 -0.0417 0.005" size="0.005 0.008" rgba="0.7 0.7 0.7 1" mass="0.005"/>
      <body name="v5_p0_b" pos=" 0.0417 -0.0417 0.012">
        <joint name="v5_p0" type="hinge" axis="0 0 1"/>
        <geom name="v5_prop0" type="box" size="0.0325 0.004 0.001" material="prop_v5"/>
      </body>

      <geom name="v5_m1" type="cylinder" pos="-0.0417 -0.0417 0.005" size="0.005 0.008" rgba="0.7 0.7 0.7 1" mass="0.005"/>
      <body name="v5_p1_b" pos="-0.0417 -0.0417 0.012">
        <joint name="v5_p1" type="hinge" axis="0 0 -1"/>
        <geom name="v5_prop1" type="box" size="0.0325 0.004 0.001" material="prop_v5"/>
      </body>

      <geom name="v5_m2" type="cylinder" pos="-0.0417  0.0417 0.005" size="0.005 0.008" rgba="0.7 0.7 0.7 1" mass="0.005"/>
      <body name="v5_p2_b" pos="-0.0417  0.0417 0.012">
        <joint name="v5_p2" type="hinge" axis="0 0 1"/>
        <geom name="v5_prop2" type="box" size="0.0325 0.004 0.001" material="prop_v5"/>
      </body>

      <geom name="v5_m3" type="cylinder" pos=" 0.0417  0.0417 0.005" size="0.005 0.008" rgba="0.7 0.7 0.7 1" mass="0.005"/>
      <body name="v5_p3_b" pos=" 0.0417  0.0417 0.012">
        <joint name="v5_p3" type="hinge" axis="0 0 -1"/>
        <geom name="v5_prop3" type="box" size="0.0325 0.004 0.001" material="prop_v5"/>
      </body>
    </body>

    <!-- 预设相机视角 (支持在原生 Viewer 的 Cameras 下拉菜单直接切换) -->
    <camera name="free_orbit" pos="2.2 -2.4 2.2" xyaxes="0.74 0.67 0 -0.28 0.31 0.91"/>
    <camera name="track_v6" mode="targetbody" target="drone_v6" pos="1.2 -1.2 2.6" xyaxes="0.707 0.707 0 -0.22 0.22 0.95"/>
    <camera name="side_view" pos="0 -3.4 1.5" xyaxes="1 0 0 0 0.25 0.97"/>
    <camera name="human_eye" pos="-1.1 -0.7 1.12" xyaxes="0.707 0.707 0 -0.15 0.15 0.98"/>
  </worldbody>
</mujoco>
"""

BENCHMARK_CASES = {
    "deep": {"idx": 85, "name": "149° 极限深倒扣工况 (Env 85)"},
    "fast": {"idx": 48, "name": "15.4 rad/s 极限高角速度工况 (Env 48)"},
    "oblique": {"idx": 54, "name": "116° 斜轴对角滚转 (Env 54)"}
}


def build_env(env_cls, seed=2024):
    p = YaoKunK10Params()
    env = env_cls(num_envs=512, seed=seed, params=p, tilt_cap_deg=180.0)
    env.reset()
    return env


def simulate_live():
    parser = argparse.ArgumentParser(description="耀坤 K1.0 微型四旋翼自稳 3D 原生交互测评查看器")
    parser.add_argument("--mode", type=str, default="compare", choices=["v8", "v7", "v6", "v5", "compare"],
                        help="运行模式: compare(v6漂移 vs v8定点对决), v8(陀螺解耦定点旗舰), v7(定点专家), v6(前代闭环专家), v5(基线)")
    parser.add_argument("--case", type=str, default="deep", choices=["deep", "fast", "oblique", "random"],
                        help="测试工况: deep(149°倒扣), fast(15.4 rad/s), oblique(116°斜轴), random(随机抽取)")
    parser.add_argument("--speed", type=float, default=1.0, help="初始播放速度倍率 (例如 1.0 或 0.25)")
    parser.add_argument("--save", type=str, default=None, help="离线导出 MP4 视频路径 (支持无显示界面环境)")
    args = parser.parse_args()

    # 导出视频模式
    if args.save:
        export_video_mp4(args.mode, args.case, args.save)
        return

    # 检查原生 viewer 支持
    import mujoco.viewer

    print("==========================================================================")
    print("   🚁 耀坤 K1.0 (ESP32-S3) 微型四旋翼 官方原生 MuJoCo 3D 动力学测评器")
    print("==========================================================================")
    print(f"当前模式: {args.mode.upper()} | 初始工况: {args.case.upper()}")
    print("--------------------------------------------------------------------------")
    print("🎮 官方原生自由视角操作指南:")
    print("  [鼠标左键拖动]   : 360° 自由旋转视角 (Orbit)")
    print("  [鼠标右键拖动]   : 平移视角 (Pan)")
    print("  [鼠标滚轮缩放]   : 自由拉近/拉远 (Zoom，从 5cm 细看旋翼到 10m 全局纵览)")
    print("  [鼠标双击机身]   : 锁定视角追踪")
    print("  [Tab 键]         : 展开 / 折叠官方侧边栏 (相机切换、线框、坐标轴、质心、力箭头)")
    print("--------------------------------------------------------------------------")
    print("⌨️ 仿真交互快捷键:")
    print("  [空格 / Space]  : 暂停 / 继续动力学仿真 (暂停时仍可 360° 鼠标自由旋转定格观察)")
    print("  [1]             : 载入 149° 极限深倒扣工况 (Env 85)")
    print("  [2]             : 载入 15.4 rad/s 极限高角速度工况 (Env 48)")
    print("  [3]             : 载入 116° 斜轴滚转工况 (Env 54)")
    print("  [N / n]         : 随机切换下一题 (来自 512 道官方题库)")
    print("  [R / r]         : 重新从 t=0 仿真播放当前工况")
    print("  [M / m]         : 切换模式 (compare 双机对决 / v6 冠军 / v5 基线)")
    print("  [S / s]         : 切换 0.25x 慢动作特写 / 1.0x 正常速度")
    print("  [Q / ESC]       : 退出测评查看器")
    print("==========================================================================\n")

    model = mujoco.MjModel.from_xml_string(MUJOCO_XML)
    data = mujoco.MjData(model)

    # 仿真状态
    state = {
        "mode": args.mode,
        "case": args.case,
        "env_idx": BENCHMARK_CASES[args.case]["idx"] if args.case in BENCHMARK_CASES else np.random.randint(0, 512),
        "case_name": BENCHMARK_CASES[args.case]["name"] if args.case in BENCHMARK_CASES else "随机工况",
        "paused": False,
        "speed": args.speed,
        "step": 0,
        "prop_angles_a": np.zeros(4),
        "prop_angles_b": np.zeros(4),
        "needs_reset": True,
        "should_quit": False
    }

    # 环境初始化: v8 旗舰, v7 定点, v6 前代, v5 基线
    env_v8 = build_env(VecExpert8)
    env_v7 = build_env(VecExpert7)
    env_v6 = build_env(VecExpert6)
    env_v5 = build_env(VecExpert5)

    def on_key(keycode):
        c = chr(keycode).upper() if 32 <= keycode <= 126 else ""
        if keycode == 32:  # 空格
            state["paused"] = not state["paused"]
            print(f">>> [状态] {'⏸️ 仿真暂停 (可使用鼠标 360° 自由旋转/缩放定格观察姿态)' if state['paused'] else '▶️ 仿真继续'}")
        elif keycode == 49 or c == '1':
            state["case"] = "deep"
            state["env_idx"] = BENCHMARK_CASES["deep"]["idx"]
            state["case_name"] = BENCHMARK_CASES["deep"]["name"]
            state["needs_reset"] = True
            print(f">>> [切换] 载入案例 1: {state['case_name']}")
        elif keycode == 50 or c == '2':
            state["case"] = "fast"
            state["env_idx"] = BENCHMARK_CASES["fast"]["idx"]
            state["case_name"] = BENCHMARK_CASES["fast"]["name"]
            state["needs_reset"] = True
            print(f">>> [切换] 载入案例 2: {state['case_name']}")
        elif keycode == 51 or c == '3':
            state["case"] = "oblique"
            state["env_idx"] = BENCHMARK_CASES["oblique"]["idx"]
            state["case_name"] = BENCHMARK_CASES["oblique"]["name"]
            state["needs_reset"] = True
            print(f">>> [切换] 载入案例 3: {state['case_name']}")
        elif c == 'N':
            state["env_idx"] = np.random.randint(0, 512)
            state["case_name"] = f"随机通道 #{state['env_idx']}"
            state["needs_reset"] = True
            print(f">>> [切换] 随机抽取新工况: {state['case_name']}")
        elif c == 'R':
            state["needs_reset"] = True
            print(f">>> [重放] 从 t=0 重启当前回合...")
        elif c == 'M':
            modes = ["compare", "v8", "v7", "v6", "v5"]
            nxt = modes[(modes.index(state["mode"]) + 1) % len(modes)]
            state["mode"] = nxt
            state["needs_reset"] = True
            print(f">>> [模式切换] 当前模式: {state['mode'].upper()}")
        elif c == 'S':
            state["speed"] = 0.25 if state["speed"] > 0.5 else 1.0
            print(f">>> [倍速] 仿真速度调整为: {state['speed']}x")
        elif c == 'Q' or keycode == 256:  # 256 = GLFW_KEY_ESCAPE
            state["should_quit"] = True

    # 启动官方原生 Viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key, show_left_ui=True, show_right_ui=True) as viewer:
        v6_root_qadr = model.jnt_qposadr[model.joint('v6_root').id]
        v6_p0_qadr = model.jnt_qposadr[model.joint('v6_p0').id]
        v5_root_qadr = model.jnt_qposadr[model.joint('v5_root').id]
        v5_p0_qadr = model.jnt_qposadr[model.joint('v5_p0').id]

        while viewer.is_running() and not state["should_quit"]:
            t_loop_start = time.time()

            if state["needs_reset"]:
                env_v8.reset()
                env_v7.reset()
                env_v6.reset()
                env_v5.reset()
                state["step"] = 0
                state["needs_reset"] = False
                state["prop_angles_a"][:] = 0.0
                state["prop_angles_b"][:] = 0.0

            if not state["paused"]:
                idx = state["env_idx"]
                cur_mode = state["mode"]

                # 步进物理
                env_v8.step(np.zeros((512, 4)))
                env_v7.step(np.zeros((512, 4)))
                env_v6.step(np.zeros((512, 4)))
                env_v5.step(np.zeros((512, 4)))

                s_v8, s_v7, s_v6, s_v5 = env_v8.dyn, env_v7.dyn, env_v6.dyn, env_v5.dyn
                pos_v8, quat_v8, u_v8 = s_v8.pos[idx].copy(), s_v8.quat[idx].copy(), env_v8._prev_a[idx].copy()
                pos_v7, quat_v7, u_v7 = s_v7.pos[idx].copy(), s_v7.quat[idx].copy(), env_v7._prev_a[idx].copy()
                pos_v6, quat_v6, u_v6 = s_v6.pos[idx].copy(), s_v6.quat[idx].copy(), env_v6._prev_a[idx].copy()
                pos_v5, quat_v5, u_v5 = s_v5.pos[idx].copy(), s_v5.quat[idx].copy(), env_v5._prev_a[idx].copy()

                if cur_mode == "compare":
                    # 右机: v8 旗舰 (陀螺解耦+定点), 左机: v6 前代 (开环水平漂移)
                    for m in range(4):
                        state["prop_angles_a"][m] += float(np.clip(u_v8[m], 0, 1)) * 2.5
                        state["prop_angles_b"][m] += float(np.clip(u_v6[m], 0, 1)) * 2.5

                    data.qpos[v6_root_qadr:v6_root_qadr+3] = np.array([0.4 + pos_v8[0], pos_v8[1], max(-0.48, pos_v8[2])])
                    data.qpos[v6_root_qadr+3:v6_root_qadr+7] = quat_v8
                    data.qpos[v6_p0_qadr:v6_p0_qadr+4] = state["prop_angles_a"]

                    data.qpos[v5_root_qadr:v5_root_qadr+3] = np.array([-0.4 + pos_v6[0], pos_v6[1], max(-0.48, pos_v6[2])])
                    data.qpos[v5_root_qadr+3:v5_root_qadr+7] = quat_v6
                    data.qpos[v5_p0_qadr:v5_p0_qadr+4] = state["prop_angles_b"]

                elif cur_mode == "v8":
                    for m in range(4):
                        state["prop_angles_a"][m] += float(np.clip(u_v8[m], 0, 1)) * 2.5
                    data.qpos[v6_root_qadr:v6_root_qadr+3] = np.array([pos_v8[0], pos_v8[1], max(-0.48, pos_v8[2])])
                    data.qpos[v6_root_qadr+3:v6_root_qadr+7] = quat_v8
                    data.qpos[v6_p0_qadr:v6_p0_qadr+4] = state["prop_angles_a"]
                    data.qpos[v5_root_qadr:v5_root_qadr+3] = np.array([0, 0, -10.0])

                elif cur_mode == "v7":
                    for m in range(4):
                        state["prop_angles_a"][m] += float(np.clip(u_v7[m], 0, 1)) * 2.5
                    data.qpos[v6_root_qadr:v6_root_qadr+3] = np.array([pos_v7[0], pos_v7[1], max(-0.48, pos_v7[2])])
                    data.qpos[v6_root_qadr+3:v6_root_qadr+7] = quat_v7
                    data.qpos[v6_p0_qadr:v6_p0_qadr+4] = state["prop_angles_a"]
                    data.qpos[v5_root_qadr:v5_root_qadr+3] = np.array([0, 0, -10.0])

                elif cur_mode == "v6":
                    for m in range(4):
                        state["prop_angles_a"][m] += float(np.clip(u_v6[m], 0, 1)) * 2.5
                    data.qpos[v6_root_qadr:v6_root_qadr+3] = np.array([pos_v6[0], pos_v6[1], max(-0.48, pos_v6[2])])
                    data.qpos[v6_root_qadr+3:v6_root_qadr+7] = quat_v6
                    data.qpos[v6_p0_qadr:v6_p0_qadr+4] = state["prop_angles_a"]
                    data.qpos[v5_root_qadr:v5_root_qadr+3] = np.array([0, 0, -10.0])

                elif cur_mode == "v5":
                    for m in range(4):
                        state["prop_angles_b"][m] += float(np.clip(u_v5[m], 0, 1)) * 2.5
                    data.qpos[v5_root_qadr:v5_root_qadr+3] = np.array([pos_v5[0], pos_v5[1], max(-0.48, pos_v5[2])])
                    data.qpos[v5_root_qadr+3:v5_root_qadr+7] = quat_v5
                    data.qpos[v5_p0_qadr:v5_p0_qadr+4] = state["prop_angles_b"]
                    data.qpos[v6_root_qadr:v6_root_qadr+3] = np.array([0, 0, -10.0])

                state["step"] += 1

                # 遥测打印
                if state["step"] % 20 == 0:
                    t_cur = state["step"] * 0.005
                    tilt_v8 = np.degrees(np.arccos(np.clip(1 - 2*(quat_v8[1]**2 + quat_v8[2]**2), -1, 1)))
                    drift_v8 = np.hypot(pos_v8[0], pos_v8[1])
                    drift_v6 = np.hypot(pos_v6[0], pos_v6[1])
                    if cur_mode == "v8":
                        sys.stdout.write(f"\r[t={t_cur:4.2f}s] v8旗舰: z={pos_v8[2]:+.2f}m xy漂移={drift_v8:.2f}m 倾角={tilt_v8:4.0f}°   ")
                    elif cur_mode == "v7":
                        sys.stdout.write(f"\r[t={t_cur:4.2f}s] v7定点: z={pos_v7[2]:+.2f}m xy漂移={drift_v7:.2f}m   ")
                    elif cur_mode == "v6":
                        sys.stdout.write(f"\r[t={t_cur:4.2f}s] v6前代: z={pos_v6[2]:+.2f}m xy漂移={drift_v6:.2f}m   ")
                    else:
                        sys.stdout.write(f"\r[t={t_cur:4.2f}s] v8(右/绿): z={pos_v8[2]:+.2f}m xy漂移={drift_v8:.2f}m | v6(左/蓝): z={pos_v6[2]:+.2f}m xy漂移={drift_v6:.2f}m   ")
                    sys.stdout.flush()

                if state["step"] >= 260:
                    time.sleep(0.5)
                    state["needs_reset"] = True

            # 更新 MuJoCo 场景
            mujoco.mj_forward(model, data)
            viewer.sync()

            # 速度同步
            dt_target = (0.005 / state["speed"]) if not state["paused"] else 0.02
            elapsed = time.time() - t_loop_start
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)

    print("\n\n>>> 用户退出 MuJoCo 测评查看器。")
    # 安全干净退出，规避 GLFW TLS 线程注销 segfault (code 139)
    os._exit(0)


def export_video_mp4(mode, case_str, out_path, fps=30):
    """离线录制并导出 MP4 视频 (支持无显示器的纯服务器环境)"""
    import cv2
    print(f">>> 正在将 3D 动力学测评导出至: {out_path} ...")
    model = mujoco.MjModel.from_xml_string(MUJOCO_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 720, 1280)

    env_idx = BENCHMARK_CASES[case_str]["idx"] if case_str in BENCHMARK_CASES else 85
    env_v6 = build_env(VecExpert6)
    env_v5 = build_env(VecExpert5)

    v6_root_qadr = model.jnt_qposadr[model.joint('v6_root').id]
    v6_p0_qadr = model.jnt_qposadr[model.joint('v6_p0').id]
    v5_root_qadr = model.jnt_qposadr[model.joint('v5_root').id]
    v5_p0_qadr = model.jnt_qposadr[model.joint('v5_p0').id]

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
    prop_v6 = np.zeros(4)
    prop_v5 = np.zeros(4)

    for step in range(240):
        # 动力学步进
        env_v6.step(np.zeros((512, 4)))
        env_v5.step(np.zeros((512, 4)))

        pos_v6 = env_v6.dyn.pos[env_idx].copy()
        quat_v6 = env_v6.dyn.quat[env_idx].copy()
        u_v6 = env_v6._prev_a[env_idx].copy()

        pos_v5 = env_v5.dyn.pos[env_idx].copy()
        quat_v5 = env_v5.dyn.quat[env_idx].copy()
        u_v5 = env_v5._prev_a[env_idx].copy()

        for m in range(4):
            prop_v6[m] += float(np.clip(u_v6[m], 0, 1)) * 2.5
            prop_v5[m] += float(np.clip(u_v5[m], 0, 1)) * 2.5

        if mode == "compare":
            data.qpos[v6_root_qadr:v6_root_qadr+3] = np.array([0.4 + pos_v6[0], pos_v6[1], max(-0.48, pos_v6[2])])
            data.qpos[v6_root_qadr+3:v6_root_qadr+7] = quat_v6
            data.qpos[v6_p0_qadr:v6_p0_qadr+4] = prop_v6

            data.qpos[v5_root_qadr:v5_root_qadr+3] = np.array([-0.4 + pos_v5[0], pos_v5[1], max(-0.48, pos_v5[2])])
            data.qpos[v5_root_qadr+3:v5_root_qadr+7] = quat_v5
            data.qpos[v5_p0_qadr:v5_p0_qadr+4] = prop_v5
        elif mode == "v6":
            data.qpos[v6_root_qadr:v6_root_qadr+3] = np.array([pos_v6[0], pos_v6[1], max(-0.48, pos_v6[2])])
            data.qpos[v6_root_qadr+3:v6_root_qadr+7] = quat_v6
            data.qpos[v6_p0_qadr:v6_p0_qadr+4] = prop_v6
            data.qpos[v5_root_qadr:v5_root_qadr+3] = np.array([0, 0, -10.0])
        elif mode == "v5":
            data.qpos[v5_root_qadr:v5_root_qadr+3] = np.array([pos_v5[0], pos_v5[1], max(-0.48, pos_v5[2])])
            data.qpos[v5_root_qadr+3:v5_root_qadr+7] = quat_v5
            data.qpos[v5_p0_qadr:v5_p0_qadr+4] = prop_v5
            data.qpos[v6_root_qadr:v6_root_qadr+3] = np.array([0, 0, -10.0])

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="free_orbit")
        img = renderer.render()
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 叠加简要水印
        cv2.putText(bgr, f"YaoKun K1.0 3D Arena - Step {step:3d} (t={step*0.005:4.2f}s)", (20, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"Left: v5 Baseline | Right: v6 Champion | 1.0m Grid Tiles", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 240), 1, cv2.LINE_AA)

        writer.write(bgr)

    writer.release()
    print(f"🎉 视频导出完成: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


if __name__ == "__main__":
    simulate_live()
