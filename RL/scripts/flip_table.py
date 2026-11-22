#!/usr/bin/env python3
"""flip_table.py —— 翻转查表生成管线（轴对齐单轴解 → mixer 反解 → 三轴精修）

三步（每格）:
  1) 轴对齐单轴 LTO: 入口 (θ0, ω_par) 解出 (θ(t), Fz(t), τ(t))
     · 等效惯量 I_eff = nᵀIn（绕翻转轴的转动惯量，不是各轴独立的近似）
  2) mixer 反解: (Fz, τ·n) → 4 电机推力（正确处理斜轴力矩分摊）
  3) 三轴 SLSQP 精修: 用反解结果当初值，严格四元数动力学下修正
产物: data/flip_table.npz  (θ0, ω_par) → 电机时间序列
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
import numpy as np
from scipy.optimize import minimize
from yao_kun_params import YaoKunK10Params

p = YaoKunK10Params()
MASS, G = p.mass, 9.81
VSCALE = 1.135
TMAX = p.thrust_max_per_motor * VSCALE ** 2
I_DIAG = np.array([5.7e-5, 6.6e-5, 1.14e-4])
RP = np.asarray(p.rotor_pos); RD = np.asarray(p.rotor_dir); KM = p.k_m_ratio
ARM = abs(RP[0, 0])
TAU_LAG = 0.037
THETA_HAND, OMEGA_HAND, VZ_HAND = np.deg2rad(60.0), 6.0, -3.0
MIX = np.vstack([np.ones(4), RP[:, 1], -RP[:, 0], -RD * KM])
MIX_INV = np.linalg.inv(MIX)
TAU_MAX_RP = 2 * TMAX * ARM


def tau_axis_max(axis):
    """绕任意方向的最大可用力矩：枚举 4 电机 16 个开关顶点，取沿 axis 投影最大值。
    物理事实: 斜轴力矩由 x/y 两轴分摊 → 可用力矩 = 纯 roll 的 66~71%。"""
    ax = np.asarray(axis, float); ax = ax / np.linalg.norm(ax)
    best = -1e9
    for m in range(16):
        T = np.array([TMAX if (m >> i) & 1 else 0.0 for i in range(4)])
        tau = np.array([(T * RP[:, 1]).sum(), -(T * RP[:, 0]).sum(), -(T * RD).sum() * KM])
        best = max(best, float(tau @ ax))
    return best


# ---------------- 三轴动力学（与 quad_sim_vec 同约定） ----------------
def f_dyn3(x, u, tau=TAU_LAG):
    qw, qx, qy, qz = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    wx, wy, wz = x[..., 4], x[..., 5], x[..., 6]
    vz = x[..., 8]
    T = TMAX * x[..., 9:13] ** 2
    r22 = 1 - 2 * (qx * qx + qy * qy)
    dq = np.stack([0.5 * (-qx * wx - qy * wy - qz * wz),
                   0.5 * (qw * wx + qy * wz - qz * wy),
                   0.5 * (qw * wy + qz * wx - qx * wz),
                   0.5 * (qw * wz + qx * wy - qy * wx)], axis=-1)
    tx = (T * RP[:, 1]).sum(axis=-1)
    ty = -(T * RP[:, 0]).sum(axis=-1)
    tz = -(T * RD).sum(axis=-1) * KM
    dw = np.stack([tx / I_DIAG[0], ty / I_DIAG[1], tz / I_DIAG[2]], axis=-1)
    accz = (T.sum(axis=-1) * r22 - MASS * G) / MASS
    due = (u - x[..., 9:13]) / tau if tau > 0 else np.zeros_like(x[..., 9:13])
    return np.concatenate([dq, dw, vz[..., None], accz[..., None], due], axis=-1)


def tilt_deg(x):
    r22 = 1 - 2 * (x[..., 1] ** 2 + x[..., 2] ** 2)
    return np.degrees(np.arccos(np.clip(r22, -1, 1)))


def quat_axis(axis, ang):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    return np.concatenate([[np.cos(ang / 2)], a * np.sin(ang / 2)])


# ---------------- 1) 轴对齐单轴 LTO ----------------
def solve_single_axis(tilt, omega_par, axis, zfloor=0.3, tfs=(0.35, 0.45, 0.6, 0.8)):
    """θ0=tilt(rad), ω_par 沿"回正方向"为正。返回 (tf, θ(t), Fz(t), τ(t))"""
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    I_eff = float(axis ** 2 @ I_DIAG)                 # 绕该轴转动惯量 nᵀIn
    tau_avail = tau_axis_max(axis)                    # 该轴真实可用力矩（斜轴受限）
    alpha = tau_avail / I_eff
    for tf in tfs:
        N = 36
        dt = tf / (N - 1)
        # 初值: 切换曲线
        x = np.array([tilt, omega_par, 2.5, 0.0, 0.42, 0.42])
        st0, u0 = [x.copy()], []
        for k in range(N):
            sw = st0[-1][0] * (1 if st0[-1][1] >= 0 else 1) + st0[-1][1] * abs(st0[-1][1]) / (2 * alpha)
            uu = np.array([0.60, 0.20]) if sw > 0 else np.array([0.20, 0.60])
            u0.append(uu)
            if k < N - 1:
                xx = st0[-1]
                tau_here = tau_avail * (xx[4] - xx[5])
                Fz_body = 4 * TMAX * ((xx[4] + xx[5]) * 0.5) ** 2
                d = np.array([xx[1], tau_here / I_eff, xx[3],
                              (Fz_body * np.cos(xx[0]) - MASS * G) / MASS,
                              (uu[0] - xx[4]) / TAU_LAG,
                              (uu[1] - xx[5]) / TAU_LAG])
                st0.append(xx + dt * d)
        st0 = np.array(st0); u0 = np.array(u0)

        def unpack(v):
            return v[:N * 6].reshape(N, 6), v[N * 6:].reshape(N, 2)

        def f1d(xx, uu, tau=TAU_LAG):
            th, w, z, vz, ue_a, ue_b = [xx[..., i] for i in range(6)]
            # u∈[0,1] 表示"沿 axis 的四电机不对称度"，力矩 = u*τ_avail；推力项按平均值近似
            tau_here = tau_avail * (ue_a - ue_b)
            Fz_body = 4 * TMAX * ((ue_a + ue_b) * 0.5) ** 2
            return np.stack([w, tau_here / I_eff, vz,
                             (Fz_body * np.cos(th) - MASS * G) / MASS,
                             (uu[..., 0] - ue_a) / tau, (uu[..., 1] - ue_b) / tau], axis=-1)

        def con_dyn(v):
            st, u = unpack(v)
            return (st[1:] - st[:-1] - 0.5 * dt * (f1d(st[:-1], u[:-1]) + f1d(st[1:], u[1:]))).ravel()

        cons = [
            {'type': 'eq', 'fun': con_dyn},
            {'type': 'eq', 'fun': lambda v: unpack(v)[0][0] - np.array([tilt, omega_par, 2.5, 0, .42, .42])},
            {'type': 'ineq', 'fun': lambda v: THETA_HAND - abs(unpack(v)[0][-1, 0])},
            {'type': 'ineq', 'fun': lambda v: OMEGA_HAND - abs(unpack(v)[0][-1, 1])},
            {'type': 'ineq', 'fun': lambda v: unpack(v)[0][-1, 3] - VZ_HAND},
            {'type': 'ineq', 'fun': lambda v: (unpack(v)[0][:, 2] - zfloor).ravel()},
        ]
        v0 = np.concatenate([st0.ravel(), u0.ravel()])
        bnds = [(None, None)] * (N * 6) + [(0, 1)] * (N * 2)
        res = minimize(lambda v: float(np.sum(np.diff(unpack(v)[1], axis=0) ** 2) * .01), v0,
                       method='SLSQP', bounds=bnds, constraints=cons,
                       options={'maxiter': 120, 'ftol': 1e-7})
        st, u = unpack(res.x)
        ok = (np.max(np.abs(con_dyn(res.x))) < 5e-3
              and abs(st[-1, 0]) <= THETA_HAND + 1e-3 and abs(st[-1, 1]) <= OMEGA_HAND + 1e-3
              and st[:, 2].min() >= zfloor - 1e-3)
        if ok:
            u_asym = st[:, 4] - st[:, 5]                # 不对称度 → 沿 axis 力矩
            u_sum = 0.5 * (st[:, 4] + st[:, 5])         # 平均油门 → 总推力
            Fz = 4 * TMAX * u_sum ** 2
            tau = tau_avail * u_asym
            return dict(tf=tf, th=st[:, 0], Fz=Fz, tau=tau, I_eff=I_eff,
                        alpha=alpha, u_asym=u_asym, u_sum=u_sum, zmin=st[:, 2].min())
    return None


# ---------------- 2) mixer 反解 ----------------
def motors_from_Fz_tau(Fz, tau_vec):
    """(Fz, [τx,τy,τz]) → 4 电机推力 → 油门。负推力=不可行（返回 None）"""
    rhs = np.stack([np.atleast_1d(Fz), np.atleast_1d(tau_vec[..., 0]),
                    np.atleast_1d(tau_vec[..., 1]), np.atleast_1d(tau_vec[..., 2])], axis=-1)
    T = rhs @ MIX_INV.T
    if np.any(T < -1e-6) or np.any(T > TMAX + 1e-6):
        return None
    return np.sqrt(np.clip(T, 0, TMAX) / TMAX)


# ---------------- 3) 三轴精修 ----------------
def refine3d(q0, w0, motors, tf, zfloor=0.3, maxiter=120):
    N = len(motors)
    dt = tf / (N - 1)
    st0 = np.zeros((N, 13))
    st0[:, 0:4] = q0
    st0[:, 4:7] = w0
    st0[:, 7] = np.linspace(2.5, 2.5 - 0.3, N)
    st0[:, 8] = 0
    st0[:, 9:13] = 0.42
    st0[0, 0:4] = q0; st0[0, 4:7] = w0
    u0 = motors.copy()
    nst, nu = N * 13, N * 4

    def unpack(v):
        return v[:nst].reshape(N, 13), v[nst:].reshape(N, 4)

    def con_dyn(v):
        st, u = unpack(v)
        return (st[1:] - st[:-1] - 0.5 * dt * (f_dyn3(st[:-1], u[:-1]) + f_dyn3(st[1:], u[1:]))).ravel()

    cons = [
        {'type': 'eq', 'fun': con_dyn},
        {'type': 'eq', 'fun': lambda v: unpack(v)[0][0] - np.concatenate([q0, w0, [2.5, 0], np.full(4, .42)])},
        {'type': 'eq', 'fun': lambda v: (np.sum(unpack(v)[0][:, 0:4] ** 2, axis=1) - 1) * 50},
        {'type': 'ineq', 'fun': lambda v: THETA_HAND - tilt_deg(unpack(v)[0][-1])},
        {'type': 'ineq', 'fun': lambda v: OMEGA_HAND - np.linalg.norm(unpack(v)[0][-1, 4:7])},
        {'type': 'ineq', 'fun': lambda v: unpack(v)[0][-1, 8] - VZ_HAND},
        {'type': 'ineq', 'fun': lambda v: (unpack(v)[0][:, 7] - zfloor).ravel()},
    ]
    v0 = np.concatenate([st0.ravel(), u0.ravel()])
    bnds = [(None, None)] * nst + [(0, 1)] * nu
    res = minimize(lambda v: float(np.sum(np.diff(unpack(v)[1], axis=0) ** 2) * .01), v0,
                   method='SLSQP', bounds=bnds, constraints=cons,
                   options={'maxiter': maxiter, 'ftol': 1e-7})
    st, u = unpack(res.x)
    ok = (np.max(np.abs(con_dyn(res.x))) < 5e-3
          and tilt_deg(st[-1]) <= np.degrees(THETA_HAND) + 1e-2
          and np.linalg.norm(st[-1, 4:7]) <= OMEGA_HAND + 1e-2
          and st[:, 7].min() >= zfloor - 1e-3)
    return st, u, ok


if __name__ == "__main__":
    # 端到端一格: 绕 (1,1,0)/√2 倾 130°，ω 沿翻转轴分量 11
    axis = np.array([1., 1, 0]) / np.sqrt(2)
    tilt = np.deg2rad(130)
    om_par = 11.0
    print(f"入口: tilt=130°, ω_par={om_par}, axis={np.round(axis,2)}")
    r = solve_single_axis(tilt, om_par, axis)
    if r is None:
        print("❌ 单轴解失败"); raise SystemExit
    print(f"① 单轴解: tf={r['tf']}s zmin={r['zmin']:+.2f}m I_eff={r['I_eff']:.2e} α={r['alpha']:.0f}")

    # 沿轴的力矩矢量
    tau_vec = np.stack([r['tau'] * axis[0], r['tau'] * axis[1], r['tau'] * axis[2]], axis=1)
    mot = motors_from_Fz_tau(r['Fz'], tau_vec)
    if mot is None:
        print("❌ mixer 反解不可行（斜轴力矩超出 4 电机能力）")
        # 退化: 只用一半力矩
        for scale in [0.8, 0.6, 0.4]:
            mot = motors_from_Fz_tau(r['Fz'], tau_vec * scale)
            if mot is not None:
                print(f"   降级力矩 ×{scale} 后可行")
                break
    if mot is None:
        raise SystemExit
    print(f"② mixer 反解成功，电机均值={np.round(mot.mean(axis=0),2)}")

    q0 = quat_axis(axis, tilt)
    w0 = om_par * axis * (-1 if False else 1)
    st, u, ok = refine3d(q0, w0, mot, r['tf'])
    print(f"③ 三轴精修: ok={ok} zmin={st[:,7].min():+.2f}m tilt_end={tilt_deg(st[-1]):.0f}° |ω_end|={np.linalg.norm(st[-1,4:7]):.1f}")
    if ok:
        np.savez(HERE + '/../models/flip_table_demo.npz',
                 axis=axis, tilt=tilt, om_par=om_par, tf=r['tf'], u=u)
        print("   已存 models/flip_table_demo.npz")
