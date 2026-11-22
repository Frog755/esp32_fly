#!/usr/bin/env python3
"""flip_table2.py —— 翻转查表管线 v2: 电机分组参数化（无需求逆，天然可行）

核心洞察: 绕任意轴 n 翻转 = 把 4 电机按"对 n 的力矩贡献符号"分两组，做两组差动。
  组P: c_i = (τ_i · n) > 0 ;  组M: c_i < 0
  控制 (uP, uM) 就是真实电机油门 → 力矩 = TMAX(uP²·Σc_P − uM²·Σ|c_M|)，推力 = TMAX(ΣuP² + ΣuM²)
  这样单轴模型与真实四电机严格一一对应，不存在"反解不可行"。
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
import os, sys
HERE = _LEGACY_ORIG_DIR
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
TAU_LAG = 0.037
THETA_HAND, OMEGA_HAND, VZ_HAND = np.deg2rad(60.0), 6.0, -3.0


def motor_torque_unit(i):
    """第 i 个电机单位推力产生的力矩向量 (tx,ty,tz)"""
    return np.array([RP[i, 1], -RP[i, 0], -RD[i] * KM])


def group_by_axis(axis):
    """按对 axis 力矩贡献的正负分两组，返回 (plus_idx, minus_idx, c)"""
    ax = np.asarray(axis, float); ax = ax / np.linalg.norm(ax)
    c = np.array([motor_torque_unit(i) @ ax for i in range(4)])
    plus = np.where(c > 1e-9)[0]
    minus = np.where(c < -1e-9)[0]
    return plus, minus, c


def quat_axis(axis, ang):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    return np.concatenate([[np.cos(ang / 2)], a * np.sin(ang / 2)])


def make_dyn(axis, nP, nM, cP, cM):
    """单轴动力学: x=(θ,ω,z,vz,uP_eff,uM_eff), u=(uP_cmd,uM_cmd)"""
    def f(x, u, tau=TAU_LAG):
        th, w, z, vz, upe, ume = [x[..., i] for i in range(6)]
        tau_net = TMAX * (nP * upe ** 2 * cP - nM * ume ** 2 * cM)
        Fz_body = TMAX * (nP * upe ** 2 + nM * ume ** 2)
        return np.stack([w, tau_net / I_DIAG[0], vz,
                         (Fz_body * np.cos(th) - MASS * G) / MASS,
                         (u[..., 0] - upe) / tau, (u[..., 1] - ume) / tau], axis=-1)
    return f


def solve_single(axis, tilt, om_par, zfloor=0.3, tfs=(0.35, 0.45, 0.6, 0.8, 1.0), N=36):
    ax = np.asarray(axis, float); ax = ax / np.linalg.norm(ax)
    plus, minus, c = group_by_axis(ax)
    nP, nM = len(plus), len(minus)
    if nP == 0 or nM == 0:
        return None                                    # 该轴无法产生力矩（如纯 yaw 可能只有单向）
    cP = float(c[plus].sum()); cM = float(-c[minus].sum())
    alpha = TMAX * cM / I_DIAG[0] if nP >= nM else TMAX * cP / I_DIAG[0]
    f = make_dyn(ax, nP, nM, cP, cM)

    for tf in tfs:
        dt = tf / (N - 1)
        st0 = np.zeros((N, 6)); u0 = np.zeros((N, 2))
        x = np.array([tilt, om_par, 2.5, 0.0, 0.5, 0.5])
        for k in range(N):
            st0[k] = x
            sw = x[0] + x[1] * abs(x[1]) / (2 * max(alpha, 1))
            u = np.array([0.62, 0.18]) if sw > 0 else np.array([0.18, 0.62])
            u0[k] = u
            if k < N - 1:
                x = x + dt * f(x, u)

        def unpack(v):
            return v[:N * 6].reshape(N, 6), v[N * 6:].reshape(N, 2)

        def con_dyn(v):
            st, u = unpack(v)
            return (st[1:] - st[:-1] - 0.5 * dt * (f(st[:-1], u[:-1]) + f(st[1:], u[1:]))).ravel()

        cons = [
            {'type': 'eq', 'fun': con_dyn},
            {'type': 'eq', 'fun': lambda v: unpack(v)[0][0] - np.array([tilt, om_par, 2.5, 0, .5, .5])},
            {'type': 'ineq', 'fun': lambda v: THETA_HAND - abs(unpack(v)[0][-1, 0])},
            {'type': 'ineq', 'fun': lambda v: OMEGA_HAND - abs(unpack(v)[0][-1, 1])},
            {'type': 'ineq', 'fun': lambda v: unpack(v)[0][-1, 3] - VZ_HAND},
            {'type': 'ineq', 'fun': lambda v: (unpack(v)[0][:, 2] - zfloor).ravel()},
        ]
        v0 = np.concatenate([st0.ravel(), u0.ravel()])
        bnds = [(None, None)] * (N * 6) + [(0, 1)] * (N * 2)
        res = minimize(lambda v: float(np.sum(np.diff(unpack(v)[1], axis=0) ** 2) * .01), v0,
                       method='SLSQP', bounds=bnds, constraints=cons,
                       options={'maxiter': 130, 'ftol': 1e-7})
        st, u = unpack(res.x)
        ok = (np.max(np.abs(con_dyn(res.x))) < 5e-3
              and abs(st[-1, 0]) <= THETA_HAND + 1e-3 and abs(st[-1, 1]) <= OMEGA_HAND + 1e-3
              and st[:, 2].min() >= zfloor - 1e-3)
        if ok:
            uP = np.zeros((N, 4)); uM = np.zeros((N, 4))
            uP[:, plus] = st[:, 4:5]; uM[:, minus] = st[:, 5:6]
            motors = uP + uM
            return dict(tf=tf, st=st, motors=motors, plus=plus, minus=minus,
                        cP=cP, cM=cM, alpha=alpha, zmin=st[:, 2].min())
    return None


if __name__ == "__main__":
    import time
    for name, axis, tilt_deg_, om in [
        ("x轴180°静止", [1, 0, 0], 180, 0.0),
        ("x轴150° ω=12", [1, 0, 0], 150, 12.0),
        ("斜(1,1,0)130° ω=11", np.array([1., 1, 0]) / np.sqrt(2), 130, 11.0),
        ("斜(1,1,1)90° ω=13", np.array([1., 1, 1]) / np.sqrt(3), 90, 13.0),
    ]:
        t0 = time.time()
        r = solve_single(np.asarray(axis, float), np.deg2rad(tilt_deg_), om)
        if r is None:
            print(f"❌ {name}: 无解")
            continue
        print(f"✅ {name}: tf={r['tf']}s zmin={r['zmin']:+.2f}m "
              f"分组 P{r['plus']}/M{r['minus']} α={r['alpha']:.0f} [{time.time()-t0:.1f}s]")
