#!/usr/bin/env python3
"""vec_rate_env.py —— C1 分层语义动作环境：rate 内环 + RL 指令层

架构（对应实验矩阵收官后的结论：PD+残差族天花板=60%，盲区=高转速刹车与倒扣带恢复）：
  RL 输出 4 维语义指令（不再是电机油门）：
      a[0]   → 垂直推力指令 c∈[-1,1] → Fz = mg·(0.8 + 0.7c)   （0.1~1.5mg，c=0≈悬停）
      a[1:4] → 期望角速度 ω_des = a·25 rad/s（跨过 PD 姿态环的力矩极限包线）
  内环（不需要学习）：τ = clip(Kp_rate·(ω_des − ω), ±τ_max)  → 现成电机混合矩阵
      · 刹停 = ω_des=0 → 内环自动满力矩反打（PD 姿态环结构上做不到的事）
      · 电机滞后/死区/压降全部由内环吸收，RL 的输出在语义层保持干净
  观测/奖励/终止与 VecRecoveryEnv 完全一致（deploy-16 维，稠密二次代价，坠地终止）。
  无门控无锚定——本实验检验"语义动作层"本身的价值（单变量）。
"""
import numpy as np

from quad_sim_vec import G
from vec_recovery_env import VecRecoveryEnv

RATE_KP = 0.003        # rate 环增益：误差 8 rad/s → 满力矩（τ_c = I/Kp ≈ 20ms，与电机滞后同量级）
OMEGA_DES_MAX = 25.0   # 期望角速度上限（跨过 PD 姿态环 kd·ω 力矩极限 ≈11 rad/s）
FZ_MID, FZ_SPAN = 0.8, 0.7   # Fz = mg·(0.8 + 0.7c)：c=-1→0.1mg（倒扣减推）~ c=1→1.5mg


class VecRateEnv(VecRecoveryEnv):
    """rate 内环 + RL 指令层的批量环境。动作 a∈[-1,1]⁴ = [爬升, ω_des xyz]。"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('tilt_cap_deg', 180.0)   # 全球面直接开局
        super().__init__(*args, **kwargs)
        p = self.params
        rp = self.dyn.rp
        rd = self.dyn.rd
        self._M = np.vstack([np.ones(4), rp[:, 1], -rp[:, 0], -rd * p.k_m_ratio])
        self._M_inv = np.linalg.inv(self._M)
        self._tau_max = 2 * p.thrust_max_per_motor * abs(p.rotor_pos[0][0])       # 0.0225
        self._tau_max_yaw = 2 * p.thrust_max_per_motor * p.k_m_ratio              # 0.0065

    def _action_to_throttle(self, a):
        s = self.dyn
        # --- 指令层：语义 → 物理目标 ---
        w_des = a[:, 1:4] * OMEGA_DES_MAX
        Fz = self.params.mass * G * (FZ_MID + FZ_SPAN * a[:, 0])
        # --- rate 内环：P 控制器 + 力矩限幅（吸收电机滞后/死区/压降）---
        tau = RATE_KP * (w_des - s.w)
        tau[:, 0:2] = np.clip(tau[:, 0:2], -self._tau_max, self._tau_max)
        tau[:, 2] = np.clip(tau[:, 2], -self._tau_max_yaw, self._tau_max_yaw)
        # --- 电机混合（同旧版 PD 的 mixer）---
        rhs = np.stack([Fz, tau[:, 0], tau[:, 1], tau[:, 2]], axis=1)
        T = rhs @ self._M_inv.T
        t_max = (self.dyn.kf * self.dyn.wm ** 2 * self.dyn.mg[:, 0])[:, None]
        T = np.clip(T, 0.0, t_max)
        return np.sqrt(T / t_max) * (self.params.v_ref / s.v_batt)[:, None]
