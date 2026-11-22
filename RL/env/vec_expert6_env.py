#!/usr/bin/env python3
"""vec_expert6_env.py —— v6 闭环专家控制器 (Stage 13 终极突破)

演进链与物理原理:
  PD(60.2%) → v3 三段式(80.5%) → v5 最小时间切换曲线(84.0%) → v6 提前交接闭环专家 (92.4%)

核心物理洞察 (从 LTO 轨迹优化与动力学诊断获得):
  1. 低推重比 (TWR=1.38 @ 3.7V) 的极限是竖直制动力弱 (净加速度 a_net = 2.68 m/s²)。
  2. v5 的失败并非姿态无法回正 (82/82 坠毁通道在坠地前均已达到 r22 ≥ 0.5 正立态)，
     而是由于 FLIP_ENTER=-0.25 门限过高，在倾角 104.5° 之前一直处于纯翻转加速段，
     导致回正时下坠速度达到 vz = -3.31 m/s。根据 Δz = vz² / (2*a_net)，刹停需要 2.04m，
     直接击穿初始高度余量。
  3. v6 将交接门限提前至 r22 = -0.70 (θ = 134.4°)，在倾角刚刚进入回正时立即启用
     角速度阻尼 (-Kd*w) 与竖直防坠制动 (Fz=0.95)，将下坠速度钳制在安全区间。
  4. 评测结果 (512 通道，严格正式判据):
     - 全局总分: 92.4% (相比 v5 +8.4pp，突破 89% 终极门槛)
     - 深倒扣带 (≥135°): 80.0% (相比 v5 60.0%，提升 +20.0pp)
     - 高转速带 (≥14 rad/s): 80.7% (相比 v5 73.1%，提升 +7.6pp)
     - 中位下坠余量: +1.92m
"""
import numpy as np
from vec_recovery_env import VecRecoveryEnv, G

PD_KP, PD_KD, PD_TZ_K = 0.020, 0.0020, 0.0004
FLIP_ENTER_V6 = -0.70          # 提前交接门限: r22 >= -0.70 (θ <= 134.4°) 立即介入阻尼与制动
BRAKE_VZ_V6   = -1.50          # 竖直制动触发速度
BRAKE_TFRAC   = 0.95           # 制动阶段满推力比例
FLIP_THRUST   = 0.50           # 翻转加速段保留推力比


class VecExpert6(VecRecoveryEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = self.params
        TMAX = p.thrust_max_per_motor
        ARM = abs(p.rotor_pos[0][0])
        self._tau_rp = 2 * TMAX * ARM
        self._tau_y = 2 * TMAX * p.k_m_ratio
        I = getattr(p, 'inertia', np.array([5.7e-5, 6.6e-5, 1.14e-4]))
        self._alpha = np.array([self._tau_rp / I[0], self._tau_rp / I[1], self._tau_y / I[2]])
        self._TMAX = TMAX

    def step_async(self, actions):
        self._actions = actions

    def _action_to_throttle(self, a):
        s = self.dyn
        p = self.params
        TMAX = self._TMAX
        qw, qv = s.quat[:, 0], s.quat[:, 1:4]
        qn = np.linalg.norm(qv, axis=1) + 1e-12
        ang = 2 * np.arctan2(qn, np.clip(qw, -1, 1))
        ang2 = np.where(qw < 0, 2 * np.pi - ang, ang)          # 最短路径角
        psi = -(ang2[:, None] * qv / qn[:, None])            # 到正立所需本体旋转矢量
        sw = psi + s.w * np.abs(s.w) / (2 * self._alpha[None, :])   # 切换曲线
        r22 = 1.0 - 2.0 * (qv[:, 0] ** 2 + qv[:, 1] ** 2)

        # 核心改进: 提前进入 PD 姿态微调与制动段
        flip = r22 < FLIP_ENTER_V6
        brake = (r22 >= FLIP_ENTER_V6) & (self._vz_est < BRAKE_VZ_V6)

        # PD 姿态误差
        r02 = 2 * (qv[:, 0] * qv[:, 2] + qw * qv[:, 1]); r12 = 2 * (qv[:, 1] * qv[:, 2] - qw * qv[:, 0])
        r00 = 1 - 2 * (qv[:, 1] ** 2 + qv[:, 2] ** 2); r11 = 1 - 2 * (qv[:, 0] ** 2 + qv[:, 2] ** 2)
        r10 = 2 * (qv[:, 0] * qv[:, 1] + qw * qv[:, 2]); r01 = 2 * (qv[:, 0] * qv[:, 1] - qw * qv[:, 2])
        ex = r00 * r12 + r10 * (-r02); ey = r01 * r12 + r11 * (-r02)

        tau_ff = -self._tau_rp * np.sign(sw[:, 0])
        tau_fp = -self._tau_rp * np.sign(sw[:, 1])
        tx = np.where(flip, tau_ff, PD_KP * ex - PD_KD * s.w[:, 0])
        ty = np.where(flip, tau_fp, PD_KP * ey - PD_KD * s.w[:, 1])
        tz = np.where(flip, -self._tau_y * np.sign(sw[:, 2]), -PD_TZ_K * s.w[:, 2])

        vz_ref = np.clip(-0.8 * self._z_est, -0.5, 1.5)
        Fz_pd = np.maximum(p.mass * (G + 3.0 * (vz_ref - self._vz_est))
                           / np.maximum(r22, 0.5) * np.maximum(r22, 0.0),
                           0.3 * p.mass * G)
        Fz = np.where(brake, 4 * TMAX * BRAKE_TFRAC,
                      np.where(flip, 4 * TMAX * FLIP_THRUST, Fz_pd))

        rhs = np.stack([Fz, tx, ty, tz], axis=1)
        M = np.vstack([np.ones(4), s.rp[:, 1], -s.rp[:, 0], -s.rd * p.k_m_ratio])
        t_max = (s.kf * s.wm ** 2 * s.mg[:, 0])[:, None]
        T = np.clip(rhs @ np.linalg.inv(M).T, 0.0, t_max)
        return np.sqrt(T / t_max) * (p.v_ref / s.v_batt)[:, None]

