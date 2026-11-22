#!/usr/bin/env python3
"""
Baseline 测试：Raptor 出厂策略在 L2F 仿真里的「回正」能力

测什么：
  1. 从不同初始倾角 roll=0..180° 释放，策略多久把姿态拉回水平？
  2. 回正成功率 vs 初始倾角曲线
  3. 不同初始角速度（失控翻滚）下的鲁棒性

关键约束（已核实）：
  - l2f 的 observation 是 82 维（含 action_history 等），Raptor 只吃前 22 维
  - 控制频率 100 Hz，dt = 10 ms
  - 坐标系 FLU；Raptor 的 22 维 = pos(3) + R展平(9) + lin_vel(3) + ang_vel(3) + prev_action(4)
  - 仿真器内部状态用四元数，需要转成旋转矩阵给策略

用法：
  python3 baseline_recovery.py                       # 默认扫 roll 0..180
  python3 baseline_recovery.py --angles 0,45,90,135,180 --omega 0
  python3 baseline_recovery.py --trials 5 --out ../output/baseline.json
"""
import argparse
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l2f import vector8 as vector  # noqa: E402
import l2f  # noqa: E402
from foundation_policy import Raptor  # noqa: E402


# ---------------- 四元数 → 旋转矩阵 ----------------
def quat_to_R(q):
    """l2f 的四元数布局是 [w, x, y, z]（需按实际校验）。返回 3x3 row-major 展平。"""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3).flatten()
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    R = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])
    return R.flatten()


def tilt_deg(R_flat):
    """从旋转矩阵求机体 z 轴与世界 z 轴的夹角（0° = 水平，180° = 倒扣）。"""
    R = R_flat.reshape(3, 3)
    # R 的第三列 = 机体 z 轴在世界系下的方向（FLU）
    zb = R[:, 2]
    c = np.clip(zb[2], -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def build_obs22(state_i, action_prev):
    """从 l2f 的单个 state 结构拼 Raptor 的 22 维观测。"""
    pos = np.asarray(state_i.position, dtype=np.float32).reshape(-1)[:3]
    quat = np.asarray(state_i.orientation, dtype=np.float32).reshape(-1)[:4]
    # l2f/RLtools 四元数约定：[w, x, y, z]
    R = quat_to_R(quat)
    # l2f 的 linear_velocity / angular_velocity 通常在 state 里；退化时置零
    lv = np.asarray(getattr(state_i, "linear_velocity", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
    av = np.asarray(getattr(state_i, "angular_velocity", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
    obs = np.concatenate([pos, R, lv, av, np.asarray(action_prev, dtype=np.float32).reshape(-1)[:4]])
    return obs.astype(np.float32)


def probe_state_fields():
    """打印 state 对象有哪些字段，用于确认四元数布局与速度字段名。"""
    dev = l2f.Device()
    rng = vector.VectorRng()
    env = vector.VectorEnvironment()
    params = vector.VectorParameters()
    state = vector.VectorState()
    vector.initialize_rng(dev, rng, 0)
    vector.initialize_environment(dev, env)
    vector.sample_initial_parameters(dev, env, params, rng)
    vector.sample_initial_state(dev, env, params, state, rng)
    s0 = state.states[0]
    return type(s0).__name__, [a for a in dir(s0) if not a.startswith("_")], s0


def run_episode(policy, dev, env, params, state, rng, init_roll_deg, init_omega, max_steps=300):
    """跑一个 episode，返回 (回正步数 或 None, 轨迹)。"""
    # 采样初始状态
    vector.sample_initial_state(dev, env, params, state, rng)

    # 强制把 0 号环境的姿态改成指定倾角
    s = state.states[0]
    r = np.radians(init_roll_deg) / 2.0
    # 绕 x 轴 roll 的四元数 [w,x,y,z]
    q = np.array([np.cos(r), np.sin(r), 0.0, 0.0], dtype=np.float32)
    s.orientation = q.tolist()
    if init_omega:
        s.angular_velocity = [init_omega, 0.0, 0.0]
    # ⚠️ 关键坑：L2F 默认 termination 是「位置偏离原点 > 0.6 m 即终止」，
    # 所以初始位置必须留在 0.6 m 球内，否则第一步就判出界（实测验证）。
    # 这里统一放到原点附近，只控制姿态/角速度变量。
    s.position = [0.0, 0.0, 0.0]
    s.linear_velocity = [0.0, 0.0, 0.0]

    policy.reset()
    act_prev = np.zeros(4, dtype=np.float32)
    traj = []
    recovered_at = None

    for step in range(max_steps):
        cur = state.states[0]
        obs22 = build_obs22(cur, act_prev)
        tilt = tilt_deg(obs22[3:12])
        traj.append({"step": step, "tilt_deg": tilt,
                     "pos_z": float(np.asarray(cur.position).reshape(-1)[2])})

        if recovered_at is None and tilt < 15.0:
            recovered_at = step

        # 全 batch 一起喂（其余环境随便动，只看 0 号）
        obs_all = np.zeros((env.N_ENVIRONMENTS, env.OBSERVATION_DIM), dtype=np.float32)
        vector.observe(dev, env, params, state, obs_all, rng)
        obs_all[0, :22] = obs22
        action = policy.evaluate_step(obs_all[:, :22])   # [N,4]
        action[1:] = 0.0                                  # 其他环境不参与

        next_state = vector.VectorState()
        vector.step(dev, env, params, state, action, next_state, rng)
        state.assign(next_state)

        terminated_flags = np.zeros(env.N_ENVIRONMENTS, dtype=bool)
        vector.terminated(dev, env, params, state, terminated_flags, rng)
        if terminated_flags[0]:
            break
        act_prev = np.clip(action[0], -1, 1).astype(np.float32)

    # 回正时间（秒）
    t_rec = None if recovered_at is None else recovered_at * 0.01
    return t_rec, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", default="0,15,30,45,60,75,90,105,120,135,150,165,180")
    ap.add_argument("--omega", type=float, default=0.0, help="附加初始角速度 rad/s")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-traj", action="store_true")
    args = ap.parse_args()

    print("=== l2f state 字段探测 ===")
    cls, fields, s0 = probe_state_fields()
    print(f"state class: {cls}")
    print(f"fields: {fields}")
    for f in ["position", "orientation", "linear_velocity", "angular_velocity", "rotor_speeds"]:
        if hasattr(s0, f):
            v = getattr(s0, f)
            print(f"  {f} = {v if not hasattr(v, '__len__') or len(v) < 6 else str(v)[:80]}")
    print()

    policy = Raptor()
    angles = [float(a) for a in args.angles.split(",")]
    rows = []

    print(f"=== 回正测试 (omega0={args.omega} rad/s, trials={args.trials}) ===")
    print(f"{'初始倾角':>8} {'回正时间':>10} {'成功率':>8}")
    for ang in angles:
        dev = l2f.Device()
        rng = vector.VectorRng()
        env = vector.VectorEnvironment()
        params = vector.VectorParameters()
        state = vector.VectorState()
        vector.initialize_rng(dev, rng, 0)
        vector.initialize_environment(dev, env)
        vector.sample_initial_parameters(dev, env, params, rng)

        times = []
        traj_last = None
        for _ in range(args.trials):
            t, traj = run_episode(policy, dev, env, params, state, rng,
                                  ang, args.omega, args.max_steps)
            times.append(t)
            traj_last = traj
        ok = sum(1 for t in times if t is not None)
        avg = np.mean([t for t in times if t is not None]) if ok else None
        print(f"{ang:8.0f} {(f'{avg:.2f}s' if avg else '  --  '):>10} {ok}/{args.trials:>6}")
        rows.append({"angle_deg": ang, "omega0": args.omega,
                     "times": [None if t is None else round(t, 3) for t in times],
                     "success": ok, "trials": args.trials,
                     "avg_time": None if avg is None else round(float(avg), 3)})
        if args.dump_traj and traj_last:
            rows[-1]["trajectory"] = traj_last

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"\n结果写入 {args.out}")


if __name__ == "__main__":
    main()
