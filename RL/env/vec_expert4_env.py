#!/usr/bin/env python3
"""vec_expert4_env.py —— v4 专家：方向锁存 + 动量顺水（失败指纹驱动的工程迭代）

v3 残余失败 100% 命中"方向打架"桶（θ̇ 每步翻转，ω0 中位 10.4，0.73s 烧光预算）：
翻转方向每步用瞬时姿态误差符号决定 → 高转速/近死点时误差趋零且抖 → bang-bang 原地摇摆。
v4 两条修复（Stage 12，纯传统控制）:
  1. 锁存: 进入翻转段的瞬间定一次方向，直到翻转段退出都不再改（迟滞模式）;
  2. 动量顺水: 定向时若误差主轴上有 |ω|≥2 rad/s → 顺着已转方向继续翻（最短路径是圆，
     往哪边转都能到正立；顺着转零刹车成本，逆着转要先杀掉角动量）。
"""
import numpy as np

from vec_expert_env import (PD_KP, PD_KD, PD_TZ_K, FLIP_ENTER, BRAKE_VZ,
                            BRAKE_TFRAC, G)
from vec_recovery_env import VecRecoveryEnv


class VecExpert4Env(VecRecoveryEnv):
    USE_LATCH = True
    USE_MOMENTUM = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latched = np.zeros((self.num_envs, 2), dtype=np.int8)   # [axis, dir]
        self._was_flip = np.zeros(self.num_envs, dtype=bool)

    def step_async(self, actions):
        self._actions = actions

    def _action_to_throttle(self, a):
        s = self.dyn
        N = self.num_envs
        p = self.params
        TMAX = p.thrust_max_per_motor
        qw, qx, qy, qz = s.quat[:, 0], s.quat[:, 1], s.quat[:, 2], s.quat[:, 3]
        r00 = 1 - 2*(qy*qy + qz*qz); r01 = 2*(qx*qy - qw*qz)
        r10 = 2*(qx*qy + qw*qz);     r11 = 1 - 2*(qx*qx + qz*qz)
        r02 = 2*(qx*qz + qw*qy);     r12 = 2*(qy*qz - qw*qx)
        r22 = 1 - 2*(qx*qx + qy*qy)
        ex = r00*r12 + r10*(-r02)
        ey = r01*r12 + r11*(-r02)
        flip = r22 < FLIP_ENTER
        brake = (r22 >= FLIP_ENTER) & (self._vz_est < BRAKE_VZ)
        # ---- 锁存定向: 进入翻转段的沿触发 ----
        enter = (flip & ~self._was_flip) if self.USE_LATCH else flip
        if enter.any():
            idx = np.where(enter)[0]
            roll_axis = np.abs(ex[idx]) >= np.abs(ey[idx])              # 0=roll 1=pitch
            err_dir = np.where(roll_axis, np.where(ex[idx] > 0, 1, -1),
                                             np.where(ey[idx] > 0, 1, -1))
            om_comp = np.where(roll_axis, s.w[idx, 0], s.w[idx, 1])
            momentum_go = (np.abs(om_comp) >= 2.0) & self.USE_MOMENTUM
            dir_ = np.where(momentum_go, np.sign(np.where(om_comp == 0, 1, om_comp)), err_dir)
            self._latched[idx, 0] = roll_axis.astype(np.int8)
            self._latched[idx, 1] = dir_.astype(np.int8)
        self._was_flip = flip
        # ---- 翻转段: 按锁存直控电机 ----
        axis_is_roll = self._latched[:, 0] == 0
        dpos = self._latched[:, 1] > 0
        Tflip = np.zeros((N, 4))
        # roll+: 电机2,3  roll-: 0,1   pitch+: 1,2   pitch-: 0,3
        def setm(mask, cols):
            if mask.any():
                Tflip[np.ix_(np.where(mask)[0], cols)] = TMAX
        setm(flip & axis_is_roll & dpos, [2, 3])
        setm(flip & axis_is_roll & ~dpos, [0, 1])
        setm(flip & ~axis_is_roll & dpos, [1, 2])
        setm(flip & ~axis_is_roll & ~dpos, [0, 3])
        # ---- 制动/正常段: mixer+PD ----
        tx = PD_KP*ex - PD_KD*s.w[:, 0]
        ty = PD_KP*ey - PD_KD*s.w[:, 1]
        tz = -PD_TZ_K*s.w[:, 2]
        vz_ref = np.clip(-0.8*self._z_est, -0.5, 1.5)
        Fz_pd = np.maximum(p.mass*(G + 3.0*(vz_ref - self._vz_est))
                           / np.maximum(r22, 0.5) * np.maximum(r22, 0.0),
                           0.3*p.mass*G)
        Fz = np.where(brake, 4*TMAX*BRAKE_TFRAC, Fz_pd)
        rhs = np.stack([Fz, tx, ty, tz], axis=1)
        M = np.vstack([np.ones(4), s.rp[:, 1], -s.rp[:, 0], -s.rd*p.k_m_ratio])
        t_max = (s.kf * s.wm**2 * s.mg[:, 0])[:, None]
        Tpd = np.clip(rhs @ np.linalg.inv(M).T, 0.0, t_max)
        T = np.where(flip[:, None], Tflip, Tpd)
        return np.sqrt(T / t_max) * (p.v_ref / s.v_batt)[:, None]
