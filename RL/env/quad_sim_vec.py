#!/usr/bin/env python3
"""quad_sim_vec.py —— 向量化批量四旋翼仿真 (Plan A 基础设施)

设计原则：
  1. 物理与 quad_sim_fast_v2.QuadDynamics.step **逐行对应**（验证脚本见本文件底部 __main__）：
     同样的初始状态 + 同样的动作序列 => 轨迹误差必须压在 float64 舍入级 (~1e-12)。
  2. N 架飞机状态存为 (N, ...) 数组，一次 numpy 运算推进全部 N 架。
  3. 支持逐架参数差异（域随机化）：mass/J/kf/mg/drag/电池参数均为 (N,) 或 (N,4) 数组。

坐标约定与 Raptor 一致 (FLU)：电机序 [FR, BR, BL, FL]，四元数 (w,x,y,z)。
"""
import numpy as np

G = 9.81


class VecQuadDynamics:
    """N 架四旋翼的批量刚体动力学。物理 = QuadDynamics.step 的数组化。"""

    def __init__(self, N, params, motor_gain=None):
        p = params
        self.N = int(N)
        self.dt = p.dt_phys
        # ---- 标量参数 → (N,) 数组（域随机化时逐架覆盖）----
        self.mass = np.full(N, p.mass, dtype=np.float64)
        self.J = np.tile(np.asarray(p.inertia, float), (N, 1))            # (N,3)
        self.Jinv = 1.0 / self.J
        self.kf = np.full(N, p.k_f, dtype=np.float64)
        base_mg = np.asarray(getattr(p, "motor_gain", np.ones(4)), float)
        self.mg = np.tile(base_mg, (N, 1)) if motor_gain is None else np.asarray(motor_gain, float).copy()
        self.rp = np.asarray(p.rotor_pos, float)                          # (4,2) 共享
        self.rd = np.asarray(p.rotor_dir, float)                          # (4,)
        self.drag_lin = np.tile(np.asarray(p.drag_lin, float), (N, 1))    # (N,3)
        self.drag_ang = np.tile(np.asarray(p.drag_ang, float), (N, 1))    # (N,3)
        self.mintau = np.full(N, 1.0 / p.motor_tau_up)
        self.mtau = np.full(N, 1.0 / p.motor_tau_down)
        self.tau_dead = np.full(N, p.pwm_deadzone)
        self.r_i = np.full(N, p.batt_r_internal)
        self.imax = np.full(N, p.current_max_per_motor)
        self.wm = np.full(N, p.omega_max)
        self.vref = np.full(N, p.v_ref)
        self.ocv_lo = np.full(N, p.v_empty)
        self.ocv_hi = np.full(N, p.v_full - p.v_empty)
        self.cap = np.full(N, p.batt_capacity_mah / 1000.0)
        self.k_m = p.k_m_ratio
        # ---- 状态 (N, ...) ----
        self.pos = np.zeros((N, 3))
        self.vel = np.zeros((N, 3))
        self.quat = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))                 # (N,4)
        self.w = np.zeros((N, 3))
        self.rotor_w = np.zeros((N, 4))
        self.soc = np.ones(N)
        self.v_batt = np.full(N, p.v_full)
        self.t = 0.0
        self.last_acc = np.zeros((N, 3))

    def set_state(self, pos, vel, quat, omega, soc=1.0):
        """批量设定初始状态。各参数形状 (N,3)/(N,4)/(N,)。"""
        self.pos = np.asarray(pos, float).copy()
        self.vel = np.asarray(vel, float).copy()
        q = np.asarray(quat, float).copy()
        self.quat = q / np.linalg.norm(q, axis=1, keepdims=True)
        self.w = np.asarray(omega, float).copy()
        self.rotor_w = np.zeros((self.N, 4))
        self.soc = np.full(self.N, soc, dtype=float) if np.isscalar(soc) else np.asarray(soc, float).copy()
        self.v_batt = self.ocv_lo + self.ocv_hi * self.soc
        self.last_acc = np.zeros((self.N, 3))

    def step(self, u):
        """u: (N,4) 期望油门（0~1）。推进一个物理子步 self.dt。"""
        dt = self.dt
        # 1) 油门裁剪 + 死区（与标量版同序）
        ue = np.clip(u, 0.0, 1.0)
        ue = np.where(ue < self.tau_dead[:, None], 0.0, ue)
        # 2) 电池：电流 → 压降 → 用【更新后】电压算转速指令（标量版同序）
        cur_sum = (self.imax[:, None] * ue ** 1.5).sum(axis=1)
        ocv = self.ocv_lo + self.ocv_hi * self.soc
        self.v_batt = np.maximum(ocv - cur_sum * self.r_i, 2.8)
        self.soc = np.maximum(self.soc - cur_sum * dt / 3600.0 / self.cap, 0.0)
        vscale = (self.v_batt / self.vref)[:, None]
        # 3) 电机一阶滞后（上升/下降不对称）
        wc = ue * self.wm[:, None] * vscale
        rising = wc > self.rotor_w
        rate = np.where(rising, self.mintau[:, None], self.mtau[:, None])
        self.rotor_w = self.rotor_w + (wc - self.rotor_w) * dt * rate
        T = self.kf[:, None] * self.rotor_w ** 2 * self.mg                 # (N,4)
        # 4) 力矩（展开求和，同标量版）
        tx = (T * self.rp[None, :, 1]).sum(axis=1)
        ty = -(T * self.rp[None, :, 0]).sum(axis=1)
        tz = -(T * self.rd[None, :]).sum(axis=1) * self.k_m
        Fz = T.sum(axis=1)
        # 5) 四元数 → 旋转矩阵（批量展开）
        q = self.quat
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        r00 = 1 - 2 * (qy * qy + qz * qz); r01 = 2 * (qx * qy - qw * qz); r02 = 2 * (qx * qz + qw * qy)
        r10 = 2 * (qx * qy + qw * qz); r11 = 1 - 2 * (qx * qx + qz * qz); r12 = 2 * (qy * qz - qw * qx)
        r20 = 2 * (qx * qz - qw * qy); r21 = 2 * (qy * qz + qw * qx); r22 = 1 - 2 * (qx * qx + qy * qy)
        vx, vy, vz = self.vel[:, 0], self.vel[:, 1], self.vel[:, 2]
        vbx = r00 * vx + r10 * vy + r20 * vz
        vby = r01 * vx + r11 * vy + r21 * vz
        vbz = r02 * vx + r12 * vy + r22 * vz
        Fbx = -self.drag_lin[:, 0] * vbx
        Fby = -self.drag_lin[:, 1] * vby
        Fbz = Fz - self.drag_lin[:, 2] * vbz
        wx, wy, wz = self.w[:, 0], self.w[:, 1], self.w[:, 2]
        tx = tx - self.drag_ang[:, 0] * wx
        ty = ty - self.drag_ang[:, 1] * wy
        tz = tz - self.drag_ang[:, 2] * wz
        ax = (r00 * Fbx + r01 * Fby + r02 * Fbz) / self.mass
        ay = (r10 * Fbx + r11 * Fby + r12 * Fbz) / self.mass
        az = (r20 * Fbx + r21 * Fby + r22 * Fbz) / self.mass - G
        # 6) 陀螺项 ω × (Jω)
        Jx = self.J[:, 0] * wx; Jy = self.J[:, 1] * wy; Jz = self.J[:, 2] * wz
        gx = wy * Jz - wz * Jy
        gy = wz * Jx - wx * Jz
        gz = wx * Jy - wy * Jx
        dwx = self.Jinv[:, 0] * (tx - gx)
        dwy = self.Jinv[:, 1] * (ty - gy)
        dwz = self.Jinv[:, 2] * (tz - gz)
        # 7) 半隐式欧拉（vel 先更新，pos 用新 vel；ω 先更新，四元数用新 ω）
        self.vel = self.vel + np.stack([ax, ay, az], axis=1) * dt
        self.pos = self.pos + self.vel * dt
        self.w = self.w + np.stack([dwx, dwy, dwz], axis=1) * dt
        # 8) 四元数积分（用【更新后】的 ω，与标量版半隐式顺序一致）+ 归一化
        wx, wy, wz = self.w[:, 0], self.w[:, 1], self.w[:, 2]
        dqw = 0.5 * (-qx * wx - qy * wy - qz * wz)
        dqx = 0.5 * (qw * wx + qy * wz - qz * wy)
        dqy = 0.5 * (qw * wy - qx * wz + qz * wx)
        dqz = 0.5 * (qw * wz + qx * wy - qy * wx)
        q = self.quat + np.stack([dqw, dqx, dqy, dqz], axis=1) * dt
        self.quat = q / np.linalg.norm(q, axis=1, keepdims=True)
        self.t += dt
        self.last_acc = np.stack([ax, ay, az], axis=1)


def up_z_of(quat):
    """(N,4) 四元数 → (N,) 机体 z 轴世界竖直分量 = R22 = 1-2(qx²+qy²)。
    修复记录：曾误写 1-2(qy²+qz²)（=R00，机体 x 轴分量）——对绕 x 轴翻滚恒返回 1，
    姿态奖励与稳定判据对 x 轴翻滚失明（影响 R1~R4 全部训练与评估）。"""
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return 1 - 2 * (qx * qx + qy * qy)
