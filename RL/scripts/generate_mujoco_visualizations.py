#!/usr/bin/env python3
"""generate_mujoco_visualizations.py —— 生成 MuJoCo 真实渲染视频、动图与交互轨迹数据

产物:
1. v6_deep_inverted_recovery.mp4 & .gif: 深倒扣 (149°) 极速回正与稳定悬停
2. v6_high_speed_recovery.mp4 & .gif: 极限高角速度 (15.4 rad/s) 阻尼制动
3. v5_vs_v6_split_screen.mp4 & .gif: 相同初始状态下 v5 (坠毁) vs v6 (成功) 分屏对比
4. eval_trajectories.json: 供交互式 HTML5 Canvas 播放器渲染的轨迹点云
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
import numpy as np
import cv2
import mujoco
from yao_kun_params import YaoKunK10Params
from vec_expert5_env import VecExpert5
from vec_expert6_env import VecExpert6

MUJOCO_XML = """
<mujoco model="yaokun_k10">
  <compiler angle="degree" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.12 0.16 0.22" rgb2="0.04 0.06 0.09" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.10 0.12 0.15" rgb2="0.14 0.17 0.22"/>
    <material name="grid_mat" texture="grid" texrepeat="12 12" reflectance="0.15"/>
    <material name="body_mat" rgba="0.15 0.75 0.95 1.0" shininess="0.9"/>
    <material name="arm_mat" rgba="0.35 0.35 0.40 1.0"/>
    <material name="prop_ccw" rgba="1.0 0.25 0.25 0.85"/>
    <material name="prop_cw" rgba="0.25 0.85 0.35 0.85"/>
    <material name="target_mat" rgba="0.2 0.9 0.4 0.25"/>
    <material name="crash_mat" rgba="0.9 0.2 0.2 0.20"/>
  </asset>
  <worldbody>
    <!-- 碰撞坠毁警戒地面 (-0.5m) -->
    <geom name="floor" type="plane" size="8 8 0.1" pos="0 0 -0.5" material="grid_mat"/>
    <!-- 目标悬停指示面 (2.5m) -->
    <site name="target" type="cylinder" size="0.45 0.005" pos="0 0 2.5" material="target_mat"/>
    <!-- 坠毁警戒网 (-0.45m) -->
    <site name="crash_floor" type="cylinder" size="0.6 0.005" pos="0 0 -0.48" material="crash_mat"/>

    <body name="drone" pos="0 0 2.5">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size="0.025 0.025 0.005" material="body_mat" mass="0.040"/>
      <geom name="battery" type="box" size="0.015 0.035 0.006" pos="0 0 -0.008" rgba="0.12 0.12 0.15 1.0" mass="0.020"/>
      <geom name="nose" type="box" size="0.006 0.010 0.003" pos="0.025 0 0.005" rgba="1.0 0.8 0.1 1.0"/>

      <geom name="arm0" type="capsule" fromto="0 0 0  0.0417 -0.0417 0" size="0.002" material="arm_mat"/>
      <geom name="arm1" type="capsule" fromto="0 0 0 -0.0417 -0.0417 0" size="0.002" material="arm_mat"/>
      <geom name="arm2" type="capsule" fromto="0 0 0 -0.0417  0.0417 0" size="0.002" material="arm_mat"/>
      <geom name="arm3" type="capsule" fromto="0 0 0  0.0417  0.0417 0" size="0.002" material="arm_mat"/>

      <geom name="motor0" type="cylinder" pos=" 0.0417 -0.0417 0.005" size="0.005 0.008" rgba="0.75 0.75 0.75 1.0" mass="0.005"/>
      <body name="prop0_body" pos=" 0.0417 -0.0417 0.012">
        <joint name="prop0_j" type="hinge" axis="0 0 1"/>
        <geom name="prop0" type="box" size="0.0325 0.004 0.001" material="prop_ccw"/>
      </body>

      <geom name="motor1" type="cylinder" pos="-0.0417 -0.0417 0.005" size="0.005 0.008" rgba="0.75 0.75 0.75 1.0" mass="0.005"/>
      <body name="prop1_body" pos="-0.0417 -0.0417 0.012">
        <joint name="prop1_j" type="hinge" axis="0 0 -1"/>
        <geom name="prop1" type="box" size="0.0325 0.004 0.001" material="prop_cw"/>
      </body>

      <geom name="motor2" type="cylinder" pos="-0.0417  0.0417 0.005" size="0.005 0.008" rgba="0.75 0.75 0.75 1.0" mass="0.005"/>
      <body name="prop2_body" pos="-0.0417  0.0417 0.012">
        <joint name="prop2_j" type="hinge" axis="0 0 1"/>
        <geom name="prop2" type="box" size="0.0325 0.004 0.001" material="prop_ccw"/>
      </body>

      <geom name="motor3" type="cylinder" pos=" 0.0417  0.0417 0.005" size="0.005 0.008" rgba="0.75 0.75 0.75 1.0" mass="0.005"/>
      <body name="prop3_body" pos=" 0.0417  0.0417 0.012">
        <joint name="prop3_j" type="hinge" axis="0 0 -1"/>
        <geom name="prop3" type="box" size="0.0325 0.004 0.001" material="prop_cw"/>
      </body>
    </body>
    <camera name="track_cam" mode="targetbody" target="drone" pos="1.4 -1.4 2.8" xyaxes="0.707 0.707 0 -0.22 0.22 0.95"/>
    <camera name="side_cam" pos="0 -2.6 1.8" xyaxes="1 0 0 0 0.28 0.96"/>
  </worldbody>
</mujoco>
"""


def rollout_episode(env_cls, env_idx, seed=2024, max_steps=500):
    p = YaoKunK10Params()
    env = env_cls(num_envs=512, seed=seed, params=p, tilt_cap_deg=180.0)
    env.reset()

    traj = []
    for step in range(max_steps):
        s = env.dyn
        pos = s.pos[env_idx].copy()
        vel = s.vel[env_idx].copy()
        quat = s.quat[env_idx].copy()
        w = s.w[env_idx].copy()

        # 计算 r22 与倾角
        qv = quat[1:4]
        r22 = 1.0 - 2.0 * (qv[0]**2 + qv[1]**2)
        tilt = float(np.degrees(np.arccos(np.clip(r22, -1, 1))))

        obs, r, done, infos = env.step(np.zeros((512, 4)))
        info = infos[env_idx]
        u = env._prev_a[env_idx].copy()  # 上步执行的油门映射

        # 状态机阶段
        if r22 < -0.70:
            stage = "FLIP_ACCEL"
        elif r22 >= -0.70 and vel[2] < -1.5:
            stage = "BRAKE_DAMPING"
        elif r22 > 0.8:
            stage = "STABLE_HOVER"
        else:
            stage = "PD_LEVELING"

        traj.append(dict(
            step=step, t=step*0.005,
            pos=pos.tolist(), vel=vel.tolist(), quat=quat.tolist(),
            w=w.tolist(), tilt=tilt, r22=float(r22),
            u=u.tolist(), stage=stage, crash=bool(info['crash']),
            stable=int(info['stable_cnt'])
        ))
        if info['crash'] and step > 40:
            # 记录坠毁后继续录制 20 帧供展示触地撞击
            if len([x for x in traj if x['crash']]) > 25:
                break

    return traj


def render_trajectory_to_frames(model, traj, width=640, height=480, title="YaoKun K1.0"):
    d = mujoco.MjData(model)
    r = mujoco.Renderer(model, height, width)

    frames = []
    prop_angles = np.zeros(4)

    for item in traj:
        pos = item['pos']
        quat = item['quat']
        u = item['u']
        t = item['t']
        tilt = item['tilt']
        vz = item['vel'][2]
        stage = item['stage']
        crash = item['crash']
        stable = item['stable']

        # 螺旋桨随油门转动
        for m in range(4):
            prop_angles[m] += float(u[m]) * 1.5

        # 更新 MuJoCo 自由关节位姿
        d.qpos[0:3] = pos
        d.qpos[3:7] = quat  # [qw, qx, qy, qz]
        d.qpos[7] = prop_angles[0]
        d.qpos[8] = prop_angles[1]
        d.qpos[9] = prop_angles[2]
        d.qpos[10] = prop_angles[3]

        mujoco.mj_forward(model, d)
        r.update_scene(d, camera='track_cam')
        img = r.render()  # RGB (H, W, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 绘制抬头显示 HUD
        # 1. 顶部半透明状态栏
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (width, 50), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)

        # 标题与时间
        cv2.putText(img_bgr, f"{title}", (15, 22), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, f"Time: {t:4.2f}s (Step {item['step']:3d})", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 210), 1, cv2.LINE_AA)

        # 阶段徽章
        stage_colors = {
            "FLIP_ACCEL": (40, 40, 230),       # 红
            "BRAKE_DAMPING": (30, 180, 230),   # 橙黄
            "PD_LEVELING": (220, 160, 50),     # 蓝
            "STABLE_HOVER": (50, 220, 80)      # 绿
        }
        badge_color = (30, 30, 220) if crash else stage_colors.get(stage, (200, 200, 200))
        badge_text = "CRASHED" if crash else stage
        cv2.rectangle(img_bgr, (width - 170, 12), (width - 15, 38), badge_color, -1)
        cv2.putText(img_bgr, badge_text, (width - 162, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. 遥测状态数据仪表 (左侧)
        y_hud = 80
        cv2.putText(img_bgr, f"Altitude z:  {pos[2]:+.2f} m", (15, y_hud), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (50, 220, 50) if pos[2] > 0.0 else (50, 50, 240), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, f"Velocity vz: {vz:+.2f} m/s", (15, y_hud + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, f"Tilt angle:  {tilt:5.1f} deg", (15, y_hud + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (50, 220, 50) if tilt < 25 else (220, 180, 50), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, f"Stable hold: {stable:3d} / 50", (15, y_hud + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (50, 220, 80) if stable >= 50 else (180, 180, 180), 1, cv2.LINE_AA)

        # 3. 电机油门直方条 (右侧)
        cv2.putText(img_bgr, "Motor PWM:", (width - 140, y_hud), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        for m in range(4):
            bar_y = y_hud + 14 + m * 14
            val = float(np.clip(u[m], 0.0, 1.0))
            cv2.rectangle(img_bgr, (width - 140, bar_y), (width - 40, bar_y + 8), (40, 50, 60), -1)
            cv2.rectangle(img_bgr, (width - 140, bar_y), (width - 140 + int(val * 100), bar_y + 8), (220, 160, 40), -1)
            cv2.putText(img_bgr, f"M{m}", (width - 32, bar_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        frames.append(img_bgr)

    return frames


def save_video_and_gif(frames, out_mp4, out_gif, fps=30):
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    # MP4
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    # 使用 ffmpeg 导出高质量、高压缩比动图 (GIF)
    palette = out_gif + ".palette.png"
    os.system(f"ffmpeg -y -i {out_mp4} -vf 'fps=15,scale={w//2}:-1:flags=lanczos,palettegen' {palette} >/dev/null 2>&1")
    os.system(f"ffmpeg -y -i {out_mp4} -i {palette} -filter_complex 'fps=15,scale={w//2}:-1:flags=lanczos[x];[x][1:v]paletteuse' {out_gif} >/dev/null 2>&1")
    if os.path.exists(palette):
        os.remove(palette)
    print(f"产物生成: {out_mp4} ({os.path.getsize(out_mp4)/1024:.1f} KB), {out_gif} ({os.path.getsize(out_gif)/1024:.1f} KB)")


def main():
    art_dir = "/home/frog/.gemini/antigravity/brain/c657968a-2168-45fe-b7a5-caf6f840015b"
    os.makedirs(art_dir, exist_ok=True)

    print("=== 初始化 MuJoCo 物理与相机渲染管线 ===")
    model = mujoco.MjModel.from_xml_string(MUJOCO_XML)

    # 1. 案例 1: 深倒扣 (149°) 极速回正 (Env 85, v6 冠军)
    print("\n>>> 录制案例 1: 深倒扣 149° 极速回正 (v6)...")
    traj_v6_deep = rollout_episode(VecExpert6, 85, seed=2024, max_steps=240)
    frames_v6_deep = render_trajectory_to_frames(model, traj_v6_deep, title="v6 Champion: Deep Inverted (149 deg)")
    save_video_and_gif(frames_v6_deep,
                       os.path.join(art_dir, "v6_deep_inverted.mp4"),
                       os.path.join(art_dir, "v6_deep_inverted.gif"))

    # 2. 案例 2: 极限高角速度 (15.4 rad/s) 阻尼刹车 (Env 48, v6 冠军)
    print("\n>>> 录制案例 2: 高角速度 15.4 rad/s 阻尼制动 (v6)...")
    traj_v6_fast = rollout_episode(VecExpert6, 48, seed=2024, max_steps=240)
    frames_v6_fast = render_trajectory_to_frames(model, traj_v6_fast, title="v6 Champion: High Angular Rate (15.4 rad/s)")
    save_video_and_gif(frames_v6_fast,
                       os.path.join(art_dir, "v6_high_speed.mp4"),
                       os.path.join(art_dir, "v6_high_speed.gif"))

    # 3. 案例 3: 同一极限深倒扣初始状态下 v5 坠毁 vs v6 成功对比分屏
    print("\n>>> 录制案例 3: v5 vs v6 相同状态分屏对决 (Env 85)...")
    traj_v5 = rollout_episode(VecExpert5, 85, seed=2024, max_steps=180)
    traj_v6 = rollout_episode(VecExpert6, 85, seed=2024, max_steps=180)

    frames_v5 = render_trajectory_to_frames(model, traj_v5, width=480, height=360, title="v5 Baseline (CRASH)")
    frames_v6 = render_trajectory_to_frames(model, traj_v6, width=480, height=360, title="v6 Champion (SUCCESS)")

    min_len = min(len(frames_v5), len(frames_v6))
    split_frames = []
    for i in range(min_len):
        f5 = frames_v5[i]
        f6 = frames_v6[i]
        # 水平拼合
        combined = np.hstack([f5, f6])
        split_frames.append(combined)

    save_video_and_gif(split_frames,
                       os.path.join(art_dir, "v5_vs_v6_comparison.mp4"),
                       os.path.join(art_dir, "v5_vs_v6_comparison.gif"))

    # 4. 导出交互式 JSON 轨迹点云
    json_path = os.path.join(art_dir, "eval_trajectories.json")
    export_payload = {
        "deep_inverted_v6": traj_v6_deep,
        "high_speed_v6": traj_v6_fast,
        "v5_crash": traj_v5,
        "v6_success": traj_v6
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f)
    print(f"\n交互数据点云已导出: {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")


if __name__ == "__main__":
    main()

