#!/usr/bin/env python3
"""vec_expert_residual_env.py —— v3 三段式专家基座 + 门控 RL 残差（Stage 11）

u_total = clip( u_expert3 + K_RES · g · a , 0, 1 ),  a ∈ [-1,1]⁴
g = max( clip((GATE_UZ−up_z)/0.2), clip((|ω|max−6)/0.2) )   ← 全 IMU 可得

与 residual_v1 的区别（为什么这轮值得跑）:
  · 基座从 PD(60.2%) 换成 v3 专家(80.5%)——起点即历史最高；
  · 锚定目标 = a→0 在专家成功态（SIL 的解析形态：专家会做的，策略只需学会不插手）；
  · RL 的真实任务被压缩到专家盲区：高初始转速下的翻转方向选择与动量管理（ω≥14 带 28% 失败）。
验证契约: a≡0 时本环境必须逐位复现 v3 的 80.5%（冒烟测试强制检查）。
"""
import numpy as np

from quad_sim_vec import up_z_of
from vec_expert_env import VecExpertEnv

K_RES = 0.5
W_RES = 0.05          # 残差 ℓ2（锚定之外的连续小惩罚）
GATE_UZ = 0.5         # 倾角>60° 全开门（bang-bang/制动区=专家可能需动量帮助的地方）
GATE_OMEGA = 6.0      # |ω|>6 rad/s 开门（PD 阻尼极限区，v3 的 bang-bang 在此与惯性打架）
GATE_RAMP = 0.2


class VecExpertResidualEnv(VecExpertEnv):
    def __init__(self, *args, k_res=K_RES, **kwargs):
        kwargs.setdefault('tilt_cap_deg', 180.0)
        super().__init__(*args, **kwargs)
        self.k_res = k_res

    def _gate(self):
        up_z = up_z_of(self.dyn.quat)
        g_att = np.clip((GATE_UZ - up_z) / GATE_RAMP, 0.0, 1.0)
        g_omg = np.clip((np.abs(self.dyn.w).max(axis=1) - GATE_OMEGA) / GATE_RAMP, 0.0, 1.0)
        return np.maximum(g_att, g_omg)

    def _action_to_throttle(self, a):
        u_exp = super()._action_to_throttle(a)     # v3 专家规则（不吃 a）
        return np.clip(u_exp + self.k_res * self._gate()[:, None] * a, 0.0, 1.0)

    def _action_cost(self, a):
        return W_RES * (a ** 2).sum(axis=1)
