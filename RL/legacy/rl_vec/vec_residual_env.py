#!/usr/bin/env python3
"""vec_residual_env.py —— 残差 PD+RL 训练环境（Round 5 主线）

架构（对应 R2~R4 三轮课程失败的尸检结论）：
  纯端到端 PPO 在任何含 >45~60° 倾角的分布下都会把悬停精度冲刷掉（梯度互扰 + 无回放）。
  本环境让 PD 兜底悬停/姿态保持（实机 96% 正立基线的向量化版），RL 只输出【门控残差】：
      u_total = clip(u_pd + K_RES · g · a, 0, 1),  a ∈ [-1,1]⁴
      g = max(clip((0.7-up_z)/0.2), clip((|ω|max-6)/6)) ∈ [0,1]   ← IMU 可得，ESP32 几行 C
  关键性质：
    · 策略输出 a=0（tanh 初始化中心）≈ 纯 PD → 训练第 0 步就有 96% 正立能力，
      "悬停原语"不再需要学习，梯度互扰的根源被移除；
    · PD 倒扣时 tilt_ok=0 → 最小油门下坠（基座行为有界且可预测），翻转恢复交给残差学；
    · PD 全部用量可部署：姿态+陀螺(IMU)、vz_est/z_est(互补滤波)、v_batt(ADC)，无 xy 位置。
  初始姿态：全球面直接开局（Raptor all-around，PD 基座使梯度抵消不再存在）。
"""
# --- [Stage 20 legacy 迁移引导] 本文件原位于 RL/env/，现移至 RL/legacy/rl_vec/ ---
# _LEGACY_ORIG_DIR 指回原目录：原有基于 __file__ 的相对路径（../env、../models 等）语义不变；
# 并把主线 env/、scripts/ 与各 legacy 子目录加入 sys.path，跨目录的旧导入照常可用。逻辑未改动。
import os as _lg_os, sys as _lg_sys
_LEGACY_RL = _lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.abspath(__file__))))
_LEGACY_ORIG_DIR = _lg_os.path.join(_LEGACY_RL, 'env')
for _lg_p in [_LEGACY_RL, _lg_os.path.join(_LEGACY_RL, 'env'), _lg_os.path.join(_LEGACY_RL, 'scripts')] + sorted(
        _lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d) for _lg_d in _lg_os.listdir(_lg_os.path.join(_LEGACY_RL, 'legacy'))
        if _lg_os.path.isdir(_lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d)) and not _lg_d.startswith(('.', '_'))):
    if _lg_p not in _lg_sys.path:
        _lg_sys.path.insert(1, _lg_p)
# --- [/Stage 20 legacy 迁移引导] ---
import numpy as np

from quad_sim_vec import G, up_z_of
from vec_recovery_env import VecRecoveryEnv

K_RES = 0.5           # 残差增益：a=±1 可把任一电机拉满/归零（u_pd±0.5 覆盖全部包线）
W_RES = 0.05          # 残差 ℓ2 幅值惩罚（第一版尸检：只罚 Δa² 时初始噪声残差 |a|≈0.58 永不收缩）
GATE_UZ = 0.7         # 门控-姿态：up_z < 0.7（倾角>45.6°）开始开门，<0.5 全开
GATE_OMEGA = 6.0      # 门控-转速：|ω| > 6 rad/s 开始开门（kd·ω=0.012 接近差动力矩极限 0.0225）
GATE_RAMP = 0.2       # 门控过渡带宽（防硬开关抖振）
PD_KP, PD_KD = 0.020, 0.0020
PD_TZ_K = 0.0004
PD_VZ_KP = 3.0        # 油门环 vz P 增益（与 96% 基线一致）
Z_KP = 0.8            # 高度外环：vz_ref = clip(Z_KP·(z_tgt - z_est), …)（可部署：用 z_est）
VZ_REF_MIN, VZ_REF_MAX = -0.5, 1.5


class VecResidualEnv(VecRecoveryEnv):
    """PD 基座 + RL 残差的批量环境。动作 a∈[-1,1]⁴ = 残差；观测与父类相同（16 维 deploy）。"""

    def __init__(self, *args, k_res=K_RES, **kwargs):
        kwargs.setdefault('tilt_cap_deg', 180.0)   # 全球面直接开局
        super().__init__(*args, **kwargs)
        self.k_res = k_res
        # 电机混合矩阵（同标量版 pd_controller：[Fz,tx,ty,tz] = M·u_thrust）
        rp = self.dyn.rp
        rd = self.dyn.rd
        # 按【行】堆叠（同标量版 np.vstack）：曾误用 np.stack(axis=1) 按列堆叠
        #   → 求了转置的逆，推力全打到单电机
        self._M = np.vstack([
            np.ones(4), rp[:, 1], -rp[:, 0], -rd * self.params.k_m_ratio
        ])                                          # (4,4)，各通道共享
        self._M_inv = np.linalg.inv(self._M)

    # ---------------- 向量化 PD 基座 ----------------
    def _pd_u(self):
        """批量 PD：返回 u_pd (N,4)。数学与标量 pd_controller 一致（姿态误差=推力方向误差）。"""
        s = self.dyn
        qw, qx, qy, qz = s.quat[:, 0], s.quat[:, 1], s.quat[:, 2], s.quat[:, 3]
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qw * qz)
        r02 = 2 * (qx * qz + qw * qy)
        r10 = 2 * (qx * qy + qw * qz)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r12 = 2 * (qy * qz - qw * qx)
        r20 = 2 * (qx * qz - qw * qy)
        r21 = 2 * (qy * qz + qw * qx)
        r22 = 1 - 2 * (qx * qx + qy * qy)
        zx, zy, zz = r02, r12, r22                  # 机体系 z 轴在世界系的指向
        # e_rot = R^T @ (z_b × e_z) = R^T @ [zy, -zx, 0]
        cx, cy = zy, -zx
        ex = r00 * cx + r10 * cy
        ey = r01 * cx + r11 * cy
        ez = r02 * cx + r12 * cy
        tx = PD_KP * ex - PD_KD * s.w[:, 0]
        ty = PD_KP * ey - PD_KD * s.w[:, 1]
        tz = -PD_TZ_K * s.w[:, 2]
        tilt_ok = np.maximum(zz, 0.0)               # 倒扣时姿态门 → 油门落到下限
        # 高度外环（可部署：用互补滤波估计值，不用真值）。
        #   注意 z_est 是【相对目标高度】的估计（气压计模型 = z - 2.5 + 偏置），故 vz_ref = -Z_KP·z_est
        vz_ref = np.clip(-Z_KP * self._z_est, VZ_REF_MIN, VZ_REF_MAX)
        Fz = self.params.mass * (G + PD_VZ_KP * (vz_ref - self._vz_est)) \
            / np.maximum(zz, 0.5) * tilt_ok
        Fz = np.maximum(Fz, 0.3 * self.params.mass * G)
        # u = M⁻¹·[Fz,tx,ty,tz]；批量形式 T_i = M⁻¹·rhs_i → T = rhs @ (M⁻¹)ᵀ
        rhs = np.stack([Fz, tx, ty, tz], axis=1)                    # (N,4)
        T = rhs @ self._M_inv.T                                     # (N,4)
        t_max = (self.dyn.kf * self.dyn.wm ** 2 * self.dyn.mg[:, 0])[:, None]
        T = np.clip(T, 0.0, t_max)
        return np.sqrt(T / t_max) * (self.params.v_ref / s.v_batt)[:, None]

    # ---------------- 动作→油门：PD 基座 + 门控残差 ----------------
    def _gate(self):
        """残差门控 g∈[0,1]（全部 IMU 可得量，ESP32 几行 C）：
        PD 全能区（正立+低转速）g=0 → 残差物理归零，96% 基线分毫不动；
        倒扣/高转速（PD 无力区）g=1 → RL 全权接管。"""
        up_z = up_z_of(self.dyn.quat)
        g_att = np.clip((GATE_UZ - up_z) / GATE_RAMP, 0.0, 1.0)
        g_omg = np.clip((np.abs(self.dyn.w).max(axis=1) - GATE_OMEGA) / GATE_RAMP, 0.0, 1.0)
        return np.maximum(g_att, g_omg)

    def _action_to_throttle(self, a):
        return np.clip(self._pd_u() + self.k_res * self._gate()[:, None] * a, 0.0, 1.0)

    def _action_cost(self, a):
        """残差 ℓ2 幅值惩罚：把"PD 已经够好的状态"里的大残差变贵。
        量级校验：翻正机动 |a|²≈1 × ~300 步 × W_RES=0.05 → 代价 1.5，
        远小于姿态收益（e_att 4→0 × W_ATT=3 × 0.1 奖励尺度 × 300 步 ≈ 360）——不会扼杀翻转。"""
        return W_RES * (a ** 2).sum(axis=1)
