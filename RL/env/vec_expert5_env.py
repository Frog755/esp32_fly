#!/usr/bin/env python3
"""vec_expert5_env.py —— v5 专家（当前冠军）: 最小时间姿态 bang-bang 翻转段

演进链: PD(60.2) → v3 三段式(80.5) → v4 锁存/动量(73.8, 被否) → 因子矩阵证伪 v4 假设
        → v5 = 双积分器最小时间切换曲线 s = ψ + ω|ω|/(2α)，u = −τmax·sign(s)
v4 教训: "方向打架"是症状不是病——逐误差重决策的优势其实是"永远走当前最短路径"；
真正的病是无视旋转动能（该刹车时不刹=过冲、硬掰=振荡）。切换曲线一次解决。
实测(256ch 正式判据): 总 83.6% | ω≥14: 76.0% | ≥135°: 67.4%（对 v3 +3.1/+4.0/+13.1）
"""
import numpy as np

from vec_recovery_env import VecRecoveryEnv, G

PD_KP, PD_KD, PD_TZ_K = 0.020, 0.0020, 0.0004
FLIP_ENTER, BRAKE_VZ, BRAKE_TFRAC = -0.25, -1.0, 0.95
FLIP_THRUST = 0.5          # 翻转段保留 50% 总推力（力矩余量与下推的折中）


class VecExpert5(VecRecoveryEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = self.params
        TMAX = p.thrust_max_per_motor
        ARM = abs(p.rotor_pos[0][0])
        self._tau_rp = 2*TMAX*ARM
        self._tau_y = 2*TMAX*p.k_m_ratio
        I = getattr(p, 'inertia', np.array([5.7e-5, 6.6e-5, 1.14e-4]))
        self._alpha = np.array([self._tau_rp/I[0], self._tau_rp/I[1], self._tau_y/I[2]])
        self._TMAX = TMAX

    def step_async(self, actions):
        self._actions = actions

    def _action_to_throttle(self, a):
        s = self.dyn
        p = self.params
        TMAX = self._TMAX
        qw, qv = s.quat[:, 0], s.quat[:, 1:4]
        qn = np.linalg.norm(qv, axis=1) + 1e-12
        ang = 2*np.arctan2(qn, np.clip(qw, -1, 1))
        ang2 = np.where(qw < 0, 2*np.pi - ang, ang)          # 最短路径角
        psi = -(ang2[:, None] * qv / qn[:, None])            # 到正立所需本体旋转矢量
        sw = psi + s.w * np.abs(s.w) / (2*self._alpha[None, :])   # 切换曲线
        r22 = 1 - 2*(qv[:, 0]**2 + qv[:, 1]**2)
        flip = r22 < FLIP_ENTER
        brake = (r22 >= FLIP_ENTER) & (self._vz_est < BRAKE_VZ)
        # PD 误差（交接段用）
        r02 = 2*(qv[:, 0]*qv[:, 2] + qw*qv[:, 1]); r12 = 2*(qv[:, 1]*qv[:, 2] - qw*qv[:, 0])
        r00 = 1-2*(qv[:, 1]**2+qv[:, 2]**2); r11 = 1-2*(qv[:, 0]**2+qv[:, 2]**2)
        r10 = 2*(qv[:, 0]*qv[:, 1] + qw*qv[:, 2]); r01 = 2*(qv[:, 0]*qv[:, 1] - qw*qv[:, 2])
        ex = r00*r12 + r10*(-r02); ey = r01*r12 + r11*(-r02)
        tau_ff = -self._tau_rp*np.sign(sw[:, 0])
        tau_fp = -self._tau_rp*np.sign(sw[:, 1])
        tx = np.where(flip, tau_ff, PD_KP*ex - PD_KD*s.w[:, 0])
        ty = np.where(flip, tau_fp, PD_KP*ey - PD_KD*s.w[:, 1])
        tz = np.where(flip, -self._tau_y*np.sign(sw[:, 2]), -PD_TZ_K*s.w[:, 2])
        vz_ref = np.clip(-0.8*self._z_est, -0.5, 1.5)
        Fz_pd = np.maximum(p.mass*(G + 3.0*(vz_ref - self._vz_est))
                           / np.maximum(r22, 0.5) * np.maximum(r22, 0.0),
                           0.3*p.mass*G)
        Fz = np.where(brake, 4*TMAX*BRAKE_TFRAC,
                      np.where(flip, 4*TMAX*FLIP_THRUST, Fz_pd))
        rhs = np.stack([Fz, tx, ty, tz], axis=1)
        M = np.vstack([np.ones(4), s.rp[:, 1], -s.rp[:, 0], -s.rd*p.k_m_ratio])
        t_max = (s.kf * s.wm**2 * s.mg[:, 0])[:, None]
        T = np.clip(rhs @ np.linalg.inv(M).T, 0.0, t_max)
        return np.sqrt(T / t_max) * (p.v_ref / s.v_batt)[:, None]
