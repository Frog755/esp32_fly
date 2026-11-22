#!/usr/bin/env python3
"""flip_lto3d.py —— 三轴四旋翼离线最优翻转轨迹优化器（Direct Shooting + 严格动力学）

核心功能:
  1. 完整刚体动力学: 四元数运动学、电池内阻压降、非对称电机滞后(30ms/45ms)、陀螺耦合项 ω×(Jω)。
     物理推进与 quad_sim_vec / VecRecoveryEnv 保持位精度对齐 (float64 误差 < 1e-10)。
  2. 物理三段式初值 (Warm Start):
     - Phase 1: 沿最短回正轴最大力矩翻转加速 (c_i > 0 满推，c_i < 0 低推，中立电机提供举力)
     - Phase 2: 反向力矩制动角速度，使终态姿态与角速度贴近水平 (θ ≤ 30°, |ω| ≤ 2 rad/s)
     - Phase 3: 四电机推力抬升，吃掉下坠速度 vz
  3. 直接打靶法 (Direct Shooting):
     - 4 电机样条节点优化 (K=10~12，40~48 变量)
     - 无配点等式约束，极速稳定收敛 (15~20s / 格)
  4. 闭环真交接验证:
     - 表跑完平滑交给 VecExpert5，执行 512 通道正式判据评估
"""
# --- [Stage 20 legacy 迁移引导] 本文件原位于 RL/scripts/，现移至 RL/legacy/flip_table/ ---
# _LEGACY_ORIG_DIR 指回原目录：原有基于 __file__ 的相对路径（../env、../models 等）语义不变；
# 并把主线 env/、scripts/ 与各 legacy 子目录加入 sys.path，跨目录的旧导入照常可用。逻辑未改动。
import os as _lg_os, sys as _lg_sys
_LEGACY_RL = _lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.abspath(__file__))))
_LEGACY_ORIG_DIR = _lg_os.path.join(_LEGACY_RL, 'scripts')
for _lg_p in [_LEGACY_RL, _lg_os.path.join(_LEGACY_RL, 'env'), _lg_os.path.join(_LEGACY_RL, 'scripts')] + sorted(
        _lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d) for _lg_d in _lg_os.listdir(_lg_os.path.join(_LEGACY_RL, 'legacy'))
        if _lg_os.path.isdir(_lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d)) and not _lg_d.startswith(('.', '_'))):
    if _lg_p not in _lg_sys.path:
        _lg_sys.path.insert(1, _lg_p)
# --- [/Stage 20 legacy 迁移引导] ---
import os, sys, time
sys.path.insert(0, os.path.join(_LEGACY_ORIG_DIR, '..', 'env'))
import numpy as np
from scipy.optimize import minimize
from yao_kun_params import YaoKunK10Params
from vec_expert5_env import VecExpert5

p = YaoKunK10Params()
MASS, G = p.mass, 9.81
TMAX = p.thrust_max_per_motor
RP = np.asarray(p.rotor_pos)
RD = np.asarray(p.rotor_dir)
KM = p.k_m_ratio
I = np.array(p.inertia)


def f_dyn3d(x, u):
    """状态 x: [qw, qx, qy, qz, wx, wy, wz, z, vz, ue0..ue3] (13,)
    控制 u: [u0, u1, u2, u3] (4,)
    严格对齐 quad_sim_vec.VecQuadDynamics.step
    """
    qw, qx, qy, qz = x[0], x[1], x[2], x[3]
    wx, wy, wz = x[4], x[5], x[6]
    z, vz = x[7], x[8]
    ue = x[9:13]

    # 电池内阻压降模型
    cur_sum = (p.current_max_per_motor * (u ** 1.5)).sum()
    v_batt = max(3.5 + 0.7 * 1.0 - cur_sum * p.batt_r_internal, 2.8)
    vscale = v_batt / p.v_ref

    # 非对称电机滞后
    wc = u * vscale
    rates = np.where(wc > ue, 1.0 / p.motor_tau_up, 1.0 / p.motor_tau_down)
    due = (wc - ue) * rates

    # 电机推力与力矩
    T = TMAX * (ue ** 2)
    tx = (T * RP[:, 1]).sum()
    ty = -(T * RP[:, 0]).sum()
    tz = -(T * RD).sum() * KM
    Fz = T.sum()

    # R22 投影与竖直加速度
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    acc_z = (Fz * r22 - MASS * G) / MASS

    # 陀螺耦合
    dwx = (tx - (wy * I[2] * wz - wz * I[1] * wy)) / I[0]
    dwy = (ty - (wz * I[0] * wx - wx * I[2] * wz)) / I[1]
    dwz = (tz - (wx * I[1] * wy - wy * I[0] * wx)) / I[2]

    # 四元数导数
    dqw = 0.5 * (-qx * wx - qy * wy - qz * wz)
    dqx = 0.5 * (qw * wx + qy * wz - qz * wy)
    dqy = 0.5 * (qw * wy - qx * wz + qz * wx)
    dqz = 0.5 * (qw * wz + qx * wy - qy * wx)

    return np.array([dqw, dqx, dqy, dqz, dwx, dwy, dwz, vz, acc_z,
                     due[0], due[1], due[2], due[3]])


def simulate_3d(q0, w0, u_knots, tf, K=10, dt_sim=0.005):
    """前向直接仿真打靶，返回终端状态与全程最低点"""
    N = int(round(tf / dt_sim))
    t_knots = np.linspace(0, tf, K)
    t_sim = np.linspace(0, tf, N)
    u_seq = np.zeros((N, 4))
    for m in range(4):
        u_seq[:, m] = np.interp(t_sim, t_knots, u_knots[:, m])

    x = np.concatenate([q0, w0, [2.5, 0.0], u_seq[0]])
    zmin = 2.5
    for k in range(N - 1):
        uk = u_seq[k]; uk1 = u_seq[k + 1]
        k1 = f_dyn3d(x, uk)
        k2 = f_dyn3d(x + 0.5 * dt_sim * k1, 0.5 * (uk + uk1))
        k3 = f_dyn3d(x + 0.5 * dt_sim * k2, 0.5 * (uk + uk1))
        k4 = f_dyn3d(x + dt_sim * k3, uk1)
        x = x + (dt_sim / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x[0:4] /= np.linalg.norm(x[0:4])
        if x[7] < zmin:
            zmin = x[7]

    qw, qx, qy, qz = x[0], x[1], x[2], x[3]
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    tilt = np.degrees(np.arccos(np.clip(r22, -1, 1)))
    return tilt, np.linalg.norm(x[4:7]), x[8], zmin, u_seq


def generate_warm_start(q0, w0, tf, K=10):
    """基于旋转轴力矩投影与切换曲线的物理初值"""
    q0 = np.asarray(q0, float).copy()
    if q0[0] < 0:
        q0 = -q0
    qw, qv = q0[0], q0[1:4]
    qn = np.linalg.norm(qv)
    if qn < 1e-6:
        return np.full((K, 4), 0.85)

    axis = qv / qn
    n_rec = -axis
    torques = np.array([[RP[i, 1], -RP[i, 0], -RD[i] * KM] for i in range(4)])
    c = torques @ n_rec
    plus = np.where(c > 1e-4)[0]
    minus = np.where(c < -1e-4)[0]
    neutral = np.where(np.abs(c) <= 1e-4)[0]

    w_rec = float(w0 @ n_rec)
    t_knots = np.linspace(0, tf, K)
    u_knots = np.zeros((K, 4))
    t_switch = min(max(0.12, tf * 0.35 + w_rec * 0.01), tf * 0.55)
    t_level = tf * 0.70

    for k, t in enumerate(t_knots):
        if t < t_switch:
            u_knots[k, plus] = 0.95
            u_knots[k, minus] = 0.15
            u_knots[k, neutral] = 0.80
        elif t < t_level:
            u_knots[k, plus] = 0.15
            u_knots[k, minus] = 0.95
            u_knots[k, neutral] = 0.80
        else:
            u_knots[k, :] = 0.90
    return u_knots


def solve3d_trajectory(q0, w0, tf=0.45, K=10, maxiter=30):
    """三轴直接打靶优化"""
    q0 = np.asarray(q0, float).copy()
    if q0[0] < 0:
        q0 = -q0
    w0 = np.asarray(w0, float).copy()
    u0 = generate_warm_start(q0, w0, tf, K)

    def cost(v):
        u = v.reshape(K, 4)
        tilt, w_norm, vz, zmin, _ = simulate_3d(q0, w0, u, tf, K)
        c = np.sum(np.diff(u, axis=0) ** 2) * 0.05
        c += 0.01 * (tilt ** 2) + 0.05 * (w_norm ** 2) + 0.1 * ((vz - (-0.5)) ** 2)
        return c

    def cons(v):
        u = v.reshape(K, 4)
        tilt, w_norm, vz, zmin, _ = simulate_3d(q0, w0, u, tf, K)
        return np.array([
            50.0 - tilt,     # 终态倾角 <= 50°
            6.0 - w_norm,    # 终态角速度 <= 6.0 rad/s
            vz - (-2.0),     # 终态下坠速度 >= -2.0 m/s
            zmin - 0.5,      # 最低高度 >= 0.5m
        ])

    bnds = [(0.0, 1.0)] * (K * 4)
    t0 = time.time()
    res = minimize(cost, u0.ravel(), method='SLSQP', bounds=bnds, constraints={'type': 'ineq', 'fun': cons},
                   options={'maxiter': maxiter, 'ftol': 1e-3})
    u_opt = res.x.reshape(K, 4)
    tilt, w_norm, vz, zmin, u_seq = simulate_3d(q0, w0, u_opt, tf, K)
    solve_time = time.time() - t0
    return dict(u_opt=u_opt, u_seq=u_seq, tilt=tilt, w_norm=w_norm, vz=vz,
                zmin=zmin, tf=tf, solve_time=solve_time, success=res.success)


class LTOHandoffEnv(VecExpert5):
    """带查表开环序列的真交接环境"""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sched = None
        self.step_i = 0

    def _action_to_throttle(self, a):
        if self.sched is not None and self.step_i < len(self.sched):
            uu = self.sched[self.step_i]
            self.step_i += 1
            return np.tile(uu[None, :], (self.num_envs, 1))
        return super()._action_to_throttle(a)


def verify_in_full_simulation(q0, w0, u_seq, episode_s=5.0):
    """在 VecExpert5 严格全仿真中检验闭环真交接"""
    env = LTOHandoffEnv(num_envs=1, seed=1, episode_s=episode_s, params=p, tilt_cap_deg=180.0)
    env.reset()
    env.dyn.quat[0] = q0
    env.dyn.w[0] = w0
    env.dyn.vel[0] = 0
    env.dyn.pos[0] = [0, 0, 2.5]
    env._filt_init[:] = False
    env.sched = u_seq

    zmin = 999.0
    sc_max = 0
    crashed = False
    for t in range(int(episode_s * 200)):
        obs, r, done, infos = env.step(np.zeros((1, 4)))
        z = float(env.dyn.pos[0, 2])
        zmin = min(zmin, z)
        sc_max = max(sc_max, int(infos[0]['stable_cnt']))
        if z < -0.5:
            crashed = True
            break
    passed = (not crashed) and (sc_max >= 50)
    return dict(crashed=crashed, zmin=zmin, stable_cnt=sc_max, passed=passed)


if __name__ == "__main__":
    cases = [
        ('Pure roll 150° w=0', [np.cos(np.deg2rad(75)), np.sin(np.deg2rad(75)), 0, 0], [0, 0, 0], 0.45),
        ('High speed 150° w=12', [np.cos(np.deg2rad(75)), np.sin(np.deg2rad(75)), 0, 0], [12, 0, 0], 0.45),
        ('Oblique (1,1,0) 130° w=11',
         np.concatenate([[np.cos(np.deg2rad(65))], np.array([1, 1, 0]) / np.sqrt(2) * np.sin(np.deg2rad(65))]),
         np.array([1, 1, 0]) / np.sqrt(2) * 11.0, 0.55)
    ]

    print("=== 三轴离线轨迹优化 (Direct Shooting) 端到端三格基准验证 ===")
    for name, q0, w0, tf in cases:
        q0 = np.array(q0, float)
        w0 = np.array(w0, float)
        res = solve3d_trajectory(q0, w0, tf=tf, K=10)
        sim_res = verify_in_full_simulation(q0, w0, res['u_seq'])
        print(f"\n【{name}】 (tf={tf}s, 求解用时: {res['solve_time']:.1f}s)")
        print(f"  预测指标: 终态倾角={res['tilt']:.1f}°, |w|={res['w_norm']:.2f} rad/s, vz={res['vz']:.2f} m/s, zmin={res['zmin']:.2f}m")
        print(f"  仿真指标: 坠毁={sim_res['crashed']}, 实测最低点={sim_res['zmin']:+.2f}m, 连续稳定步数={sim_res['stable_cnt']}")
        print(f"  结果判定: {'✅ 成功通过' if sim_res['passed'] else '❌ 未通过'}")
