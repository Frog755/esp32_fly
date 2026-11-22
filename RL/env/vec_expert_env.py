#!/usr/bin/env python3
"""vec_expert_env.py —— 三段式手搓救援专家（v3，2026-11-09 诊断产物）

背景（推翻了 model_2M 时代的结论）:
  · 严格判据（首坠前计数）复测发现: model_2M/恒定nudge/PD 在静态 ≥150° 全部坠毁，
    此前"静态倒扣 100% 翻正"是 auto-reset 污染的测量假象（已收回）。
  · 轨迹诊断定位真凶: PD 油门几何——倒扣时 tilt_ok=0 却保留 0.3mg 对称【向下】推力
    (1.3g 加速坠落)；翻过 90° 时 vz 已 −6~−9 m/s，Fz 需求超过满油门 → mixer 饱和，
    力矩权限归零，机体卡死在水平面坠地。不是死点问题，是动能管理问题。

三段式规则（零学习）:
  翻转段 (θ>104°):  电机直控 bang-bang——误差主轴一对满推、另一对归零（τmax=395 rad/s²
                    快翻，~0.13s 扫完 165°；mixer 死区的替代方案）
  制动段 (θ<104° 且 vz_est<−1): 全推 (0.95×4·Tmax) + PD 姿态力矩——最短时间内吃掉下坠动能
  正常段:          原 PD（姿态环+油门环+高度外环）

实测（256 通道全球面，seed 2024，成功=首坠前稳定≥50步）:
  总 80.5% | ω0≥14: 72.0% | ≥135°: 54.3%   （对照 PD/model_2M: 60.2-60.5 / 40-42.7 / 15.2）
  静态 150-180°: 全部 ✓（最低点损失 0.5~1.4m，预算内）
残余失败: 高初始转速与 bang-bang 翻转方向打架（非动量感知）——留给 RL 残差的新靶子。
"""
import numpy as np

from vec_recovery_env import VecRecoveryEnv, G

PD_KP, PD_KD = 0.020, 0.0020
PD_TZ_K = 0.0004
FLIP_ENTER = -0.25      # r22 < 此值(θ>~104°) 进翻转段
BRAKE_VZ = -1.0         # 出翻转段后 vz_est 低于此值持续制动
BRAKE_TFRAC = 0.95      # 制动段油门占满推比例


class VecExpertEnv(VecRecoveryEnv):
    def step_async(self, actions):
        self._actions = actions

    def _action_to_throttle(self, a):
        s = self.dyn
        N = self.num_envs
        TMAX = self.params.thrust_max_per_motor
        qw, qx, qy, qz = s.quat[:, 0], s.quat[:, 1], s.quat[:, 2], s.quat[:, 3]
        r00 = 1 - 2*(qy*qy + qz*qz); r01 = 2*(qx*qy - qw*qz)
        r10 = 2*(qx*qy + qw*qz);     r11 = 1 - 2*(qx*qx + qz*qz)
        r02 = 2*(qx*qz + qw*qy);     r12 = 2*(qy*qz - qw*qx)
        r22 = 1 - 2*(qx*qx + qy*qy)
        ex = r00*r12 + r10*(-r02)
        ey = r01*r12 + r11*(-r02)
        flip = r22 < FLIP_ENTER
        brake = (r22 >= FLIP_ENTER) & (self._vz_est < BRAKE_VZ)
        # --- 翻转段: 电机直控（mixer 死区的替代） ---
        roll_axis = np.abs(ex) >= np.abs(ey)
        roll_pos = flip & roll_axis & (ex > 0)
        roll_neg = flip & roll_axis & (ex <= 0)
        pitch_pos = flip & ~roll_axis & (ey > 0)
        pitch_neg = flip & ~roll_axis & (ey <= 0)
        Tflip = np.zeros((N, 4))
        Tflip[:, 0] = np.where(roll_neg | pitch_neg, TMAX, 0.0)
        Tflip[:, 1] = np.where(roll_neg | pitch_pos, TMAX, 0.0)
        Tflip[:, 2] = np.where(roll_pos | pitch_neg, TMAX, 0.0)
        Tflip[:, 3] = np.where(roll_pos | pitch_pos, TMAX, 0.0)
        # --- 制动段/正常段: mixer + PD（制动段仅替换 Fz 为满推） ---
        tx = PD_KP*ex - PD_KD*s.w[:, 0]
        ty = PD_KP*ey - PD_KD*s.w[:, 1]
        tz = -PD_TZ_K*s.w[:, 2]
        vz_ref = np.clip(-0.8*self._z_est, -0.5, 1.5)
        Fz_pd = np.maximum(self.params.mass*(G + 3.0*(vz_ref - self._vz_est))
                           / np.maximum(r22, 0.5) * np.maximum(r22, 0.0),
                           0.3*self.params.mass*G)
        Fz = np.where(brake, 4*TMAX*BRAKE_TFRAC, Fz_pd)
        rhs = np.stack([Fz, tx, ty, tz], axis=1)
        M = np.vstack([np.ones(4), s.rp[:, 1], -s.rp[:, 0], -s.rd*self.params.k_m_ratio])
        t_max = (s.kf * s.wm**2 * s.mg[:, 0])[:, None]
        Tpd = np.clip(rhs @ np.linalg.inv(M).T, 0.0, t_max)
        T = np.where(flip[:, None], Tflip, Tpd)
        return np.sqrt(T / t_max) * (self.params.v_ref / s.v_batt)[:, None]
