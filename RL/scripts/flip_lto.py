#!/usr/bin/env python3
"""flip_lto.py —— 翻转轨迹离线优化器（单轴版 v2：含电机滞后 + 交接班终端）

v1 三处硬伤及修复（来自步骤① 实跑闸门）：
  ① 终端只约束 (θ,ω)=(0,0)、不管终端力矩 → 翻正后又歪回去死
     → v2 终端 = 交接班姿势（|θ|≤60°, |ω|≤6, vz≥−3），收尾交给 v5 规则
  ② 模型无电机滞后（真机 30/45ms）→ v2 状态加 (uA_eff, uB_eff) 一阶滞后
  ③ C 检查把最后一段油门无限保持 → v2 过 tf 后真交给 VecExpert5 规则接盘
模型（与 quad_sim_vec 同约定）:
  组A=电机2,3(力臂+ARM), 组B=电机0,1(力臂−ARM)
  T_i = TMAX·VSCALE²·u_eff_i²;  tx = 2·ARM·(TA−TB);  Fv = 2(TA+TB)·cosθ − mg
"""
import numpy as np
from scipy.optimize import minimize

MASS, G, TMAX, ARM, I_ROLL = 0.08, 9.81, 0.27, 0.0417, 5.7e-5
VSCALE = 1.135
TMAX_EFF = TMAX * VSCALE ** 2
ALPHA = 2 * ARM * TMAX_EFF / I_ROLL
TAU_LAG = 0.037
THETA_HAND = np.deg2rad(60.0)
OMEGA_HAND = 6.0
VZ_HAND = -3.0


def f_dyn(x, u, tau=TAU_LAG):
    """x=(θ,ω,z,vz,uA_eff,uB_eff)，u=(uA_cmd,uB_cmd)"""
    th, w, z, vz, ua, ub = [x[..., i] for i in range(6)]
    TA = TMAX_EFF * ua ** 2
    TB = TMAX_EFF * ub ** 2
    dw = 2 * ARM * (TA - TB) / I_ROLL
    Fv = 2 * (TA + TB) * np.cos(th) - MASS * G
    if tau > 0:
        d_ua = (u[..., 0] - ua) / tau
        d_ub = (u[..., 1] - ub) / tau
    else:
        d_ua = np.zeros_like(ua); d_ub = np.zeros_like(ub)
    return np.stack([w, dw, vz, Fv / MASS, d_ua, d_ub], axis=-1)


def warm_start(theta0, om0, tf, N=40, tau=TAU_LAG):
    dt = tf / (N - 1)
    x = np.array([theta0, om0, 2.5, 0.0, 0.0, 0.0], float)
    states, acts = [x.copy()], []
    for k in range(N):
        sw = x[0] + x[1] * abs(x[1]) / (2 * ALPHA)
        u = np.array([0.15, 0.55]) if sw > 0 else np.array([0.55, 0.15])
        acts.append(u)
        if k < N - 1:
            x = x + dt * f_dyn(x, u, tau)
            states.append(x.copy())
    return np.array(states), np.array(acts)


def solve_feasible(theta0, om0, tf, N=40, zfloor=-0.45, tau=TAU_LAG, warm=None,
                   maxiter=200):
    """固定时界可行性 OCP，终端=交接班姿势。"""
    dt = tf / (N - 1)
    if warm is None:
        st0, u0 = warm_start(theta0, om0, tf, N, tau)
    else:
        st0, u0 = warm
        idx = np.linspace(0, len(st0) - 1, N).astype(int)
        st0 = st0[idx] if len(st0) >= N else np.vstack(
            [st0, np.repeat(st0[-1:], N - len(st0), axis=0)])
        u0 = u0[idx]
        if st0.shape[1] != 6:
            st0 = np.hstack([st0, np.repeat(st0[:, 4:6], 1, axis=1)]) if st0.shape[1] >= 6 else st0

    nst, nu = N * 6, N * 2

    def unpack(v):
        return v[:nst].reshape(N, 6), v[nst:].reshape(N, 2)

    def con_dyn(v):
        st, u = unpack(v)
        fk = f_dyn(st[:-1], u[:-1], tau)
        fk1 = f_dyn(st[1:], u[1:], tau)
        return (st[1:] - st[:-1] - 0.5 * dt * (fk + fk1)).ravel()

    def con_init(v):
        st, _ = unpack(v)
        base = np.array([theta0, om0, 2.5, 0.0, 0.0, 0.0])
        return st[0] - base

    x0 = np.concatenate([st0.ravel(), u0.ravel()])
    cons = [
        {'type': 'eq', 'fun': con_dyn},
        {'type': 'eq', 'fun': con_init},
        {'type': 'ineq', 'fun': lambda v: THETA_HAND - np.abs(unpack(v)[0][-1, 0])},
        {'type': 'ineq', 'fun': lambda v: OMEGA_HAND - np.abs(unpack(v)[0][-1, 1])},
        {'type': 'ineq', 'fun': lambda v: unpack(v)[0][-1, 3] - VZ_HAND},
        {'type': 'ineq', 'fun': lambda v: (unpack(v)[0][:, 2] - zfloor).ravel()},
    ]
    bnds = [(None, None)] * nst + [(0.0, 1.0)] * nu
    obj = lambda v: float(np.sum(np.diff(unpack(v)[1], axis=0) ** 2) * 0.01)
    res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons,
                   options={'maxiter': maxiter, 'ftol': 1e-7})
    st, u = unpack(res.x)
    dmax = float(np.max(np.abs(con_dyn(res.x))))
    eps = 1e-4
    ok = (dmax < 2e-3
          and abs(st[-1, 0]) <= THETA_HAND + eps
          and abs(st[-1, 1]) <= OMEGA_HAND + eps
          and st[-1, 3] >= VZ_HAND - eps
          and st[:, 2].min() >= zfloor - eps)
    return st, u, ok, dmax


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))

    # ---------- A: 无滞后/无坠地约束，对拍解析解 ----------
    th0 = np.pi
    a_nolag = 2 * ARM * TMAX / I_ROLL
    t_star = 2 * np.sqrt(th0 / a_nolag)
    print(f"A 解析解 t*={t_star:.4f}s (alpha={a_nolag:.0f})")
    feas_t = None
    for tf in [0.14, 0.16, 0.18, 0.20, 0.24]:
        st, u, ok, dmax = solve_feasible(th0, 0.0, tf, zfloor=-100.0, tau=0.0)
        print(f"   tf={tf:.2f}s ok={ok} err={dmax:.1e}")
        if ok and feas_t is None:
            feas_t = tf
    print(f"A 最小可行 tf={feas_t}，差 {abs(feas_t-t_star)/t_star*100:.1f}%" if feas_t else "A 失败")

    # ---------- B: 含滞后 + 坠地约束，warm-start 链 ----------
    print("B 含滞后扫 tf:")
    best, warm = None, None
    for tf in [0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.9]:
        st, u, ok, dmax = solve_feasible(th0, 0.0, tf, zfloor=-0.45, warm=warm)
        print(f"   tf={tf:.2f}s ok={ok} zmin={st[:,2].min():+.2f} err={dmax:.1e}")
        if ok:
            warm = (st, u)
            if best is None:
                best = (tf, st, u)
    if best is None:
        print("B 全灭"); raise SystemExit
    tf, st, u = best
    print(f"B 选定 tf={tf}s zmin={st[:,2].min():+.2f}m")

    # ---------- C: 真交接，表跑完交给 v5 ----------
    import torch
    torch.set_num_threads(4)
    from vec_expert5_env import VecExpert5
    from yao_kun_params import YaoKunK10Params

    class LTOHandoffEnv(VecExpert5):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.sched = None; self.step_i = 0
        def _action_to_throttle(self, a):
            if self.sched is not None and self.step_i < len(self.sched):
                uu = self.sched[self.step_i]; self.step_i += 1
                return np.tile(uu[None, :], (self.num_envs, 1))
            return super()._action_to_throttle(a)

    p = YaoKunK10Params()
    env = LTOHandoffEnv(num_envs=1, seed=1, episode_s=9.9, params=p, tilt_cap_deg=180.0)
    env.reset()
    h = np.pi / 2
    env.dyn.quat[0] = [np.cos(h), np.sin(h), 0, 0]
    env.dyn.w[0] = 0; env.dyn.vel[0] = 0; env.dyn.pos[0] = [0, 0, 2.5]
    env._filt_init[:] = False
    ts = np.linspace(0, tf, len(u))
    nctl = int(tf * 200)
    sched = np.zeros((nctl, 4))
    for t in range(nctl):
        ua = float(np.interp(t / 200, ts, u[:, 0])); ub = float(np.interp(t / 200, ts, u[:, 1]))
        sched[t] = [ub, ub, ua, ua]
    env.sched = sched
    Ttot = int((tf + 2.5) * 200)
    zmin, track, sc_max = 9.0, [], 0
    for t in range(Ttot):
        obs, r, done, infos = env.step(np.zeros((1, 4)))
        r22 = 1 - 2 * (env.dyn.quat[0, 1] ** 2 + env.dyn.quat[0, 2] ** 2)
        th = float(np.degrees(np.arccos(np.clip(r22, -1, 1))))
        z = float(env.dyn.pos[0, 2]); zmin = min(zmin, z)
        sc_max = max(sc_max, int(infos[0]['stable_cnt']))
        track.append((t / 200, th, z, float(env.dyn.w[0, 0])))
    arrive = next((tt for tt, thh, z, w in track if thh < 20 and abs(w) < 3), None)
    print("C 真交接结果:")
    print(f"   zmin={zmin:+.2f}m (预测 {st[:,2].min():+.2f}m, 坠线 −0.50m)")
    print(f"   到位(θ<20°,|ω|<3): {arrive if arrive is None else round(arrive,2)}s")
    print(f"   最长稳定={sc_max} 步 (≥50 完全恢复)")
    print(f"   终态 θ={track[-1][1]:.1f}° z={track[-1][2]:.2f}m")
    print(f"   {'✅ 实跑存活' if zmin > -0.49 else '❌ 实跑坠毁'}")
