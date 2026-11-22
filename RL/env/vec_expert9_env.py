#!/usr/bin/env python3
"""vec_expert9_env.py —— v9 低掉高快速回正控制器（Stage 17 仿真冠军）

512 通道 8-seed 均值：成功率 99.6%（v8 94.3%），坠毁 1.9/512（v8 29.0），
掉高中位 0.87m（v8 1.17m）、P90 1.57m（v8 2.54m），稳悬耗时中位 1.35s（v8 2.25s）。
⚠️ 仅仿真验证（理想传感器、满电开局），尚未移植 C。

相对 v8 的改动（括号 = 单独关掉后的坠毁数，完整 v9 为 2 架；均可经 ctrl 字典开关做消融）：
  1. 自旋超前补偿 T_DIR（→36）：输出前把 (τx,τy) 按 −ωz·T_DIR 预先反向旋转，
     抵消电机滞后期间机体偏航自旋造成的力矩方向偏转（v8 剩余坠毁的主因）。
  2. 约简姿态测地线 bang-bang（→14）：只把机体 z 轴转回竖直、不纠偏航；沿测地线方向 n 单轴 bang-bang，
     X 机架沿 n 的最大力矩 τ_rp/(|nx|+|ny|)，垂直分量只阻尼；切换曲线含电机滞后补偿 T_LAG。
  3. 电压感知推力上限 vbatt_aware（→11）：按实时 v_batt 换算 Tmax。
  4. 力矩优先分配 prio_mixer（→6）：先 roll/pitch 力矩，再集体推力，偏航最后。
  5. 正立半球降低翻转力矩（K_TAU_UP）腾出升力；翻转段正立半球满推力；bang-bang ↔ PD 迟滞 105°/120°。
  6. 垂直通道更积极（KV_Z 9.0 / KZ_REF 3.0 / VZ_UP 1.8）：不影响成功率，缩短掉高与稳悬耗时。
其余（PD 段水平定点、陀螺前馈、高度积分门控、竖直刹车）沿用 v8。
默认参数 = 最终版（同 RL/results/stage17/v9_final_ctrl.json）。详见 .devlog/DEVLOG.md Stage 17。
"""
import numpy as np
from vec_recovery_env import VecRecoveryEnv, G

DEFAULT_CTRL = dict(
    reduced_att=True,     # 1. 约简姿态误差
    prio_mixer=True,      # 2. 力矩优先分配
    vbatt_aware=True,     # 4. 电压感知推力上限
    TH_H=np.deg2rad(105),  # bang-bang → PD 交接倾角
    TH_HYS=np.deg2rad(15),
    W_H=10.0,             # |ω_xy| 超过此值也走 bang-bang（高转速先刹停）
    T_LAG=0.09,          # 电机滞后超前补偿 (s)
    K_ALPHA=0.6,         # 切换曲线使用的有效角加速度比例
    R_LO=0.0,             # 翻转段 r22 高于此值才加集体推力
    FLIP_FZ_LO=0.0,       # 倒扣半球额外集体推力（占 4Tmax 比例；0=仅力矩所需）
    YAW_IN_FLIP=False,    # 翻转段是否做偏航 bang-bang
    KD_PERP=0.002,        # 测地线 bang-bang 时垂直方向角速度阻尼
    T_DIR=0.035,           # 自旋超前补偿时间 (s)
    K_TAU_INV=1.0,        # 倒扣半球 (r22<R_LO) bang-bang 力矩比例
    K_TAU_UP=0.6,         # 正立半球 bang-bang 力矩比例
    YAW_RES=0.0,          # 近正立且仍在自旋时，为偏航阻尼预留的单电机推力余量比例
    KZ_FLIP=0.002,         # 翻转段偏航阻尼增益
    KP_ATT=0.025, KD_ATT=0.0027, KZ_ATT=0.002,
    KP_POS_XY=1.20, KD_VEL_XY=1.80, MAX_TILT_RAD=0.20,
    BRAKE_VZ=-1.50, BRAKE_TFRAC=0.95,
    KV_Z=9.0, KI_Z=4.0, VZ_UP=1.8, VZ_DN=-0.8, KZ_REF=3.0,
)


class VecExpert9(VecRecoveryEnv):
    def __init__(self, *args, ctrl=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = dict(DEFAULT_CTRL)
        if ctrl:
            self.c.update(ctrl)
        p = self.params
        self._TMAX = p.thrust_max_per_motor
        self._ARM = abs(p.rotor_pos[0][0])
        I = np.asarray(getattr(p, 'inertia', (5.7e-5, 6.6e-5, 1.14e-4)), float)
        self._I = I
        self._z_int = np.zeros(self.num_envs)
        self._bb = np.zeros(self.num_envs, bool)
        rp = np.asarray(p.rotor_pos, float); rd = np.asarray(p.rotor_dir, float)
        M = np.vstack([np.ones(4), rp[:, 1], -rp[:, 0], -rd * p.k_m_ratio])
        self._Minv = np.linalg.inv(M)            # (4,4): T = Minv @ [Fz,tx,ty,tz]

    def reset(self):
        self._z_int[:] = 0.0
        self._bb[:] = False
        return super().reset()

    def _reset_done(self, dones):
        super()._reset_done(dones)
        self._z_int[dones] = 0.0
        self._bb[dones] = False

    def step_async(self, actions):
        self._actions = actions

    # ---------------- 力矩优先分配器 ----------------
    def _allocate(self, Fz, tx, ty, tz, tmax, yaw_res=None):
        Mi = self._Minv
        d = tx[:, None] * Mi[None, :, 1] + ty[:, None] * Mi[None, :, 2]      # 纯 roll/pitch 分量
        span = d.max(1) - d.min(1)
        sc = np.minimum(1.0, tmax / np.maximum(span, 1e-12))
        d *= sc[:, None]
        c_lo = -d.min(1); c_hi = tmax - d.max(1)
        if yaw_res is not None:
            c_hi = np.maximum(c_hi - yaw_res, c_lo)   # 给偏航阻尼预留余量
        c = np.clip(Fz / 4.0, c_lo, c_hi)
        T = d + c[:, None]
        dy = tz[:, None] * Mi[None, :, 3]
        with np.errstate(divide='ignore', invalid='ignore'):
            lim = np.where(dy > 0, (tmax[:, None] - T) / dy, np.where(dy < 0, T / (-dy), np.inf))
        sy = np.clip(lim.min(1), 0.0, 1.0)
        T = T + sy[:, None] * dy
        return np.clip(T, 0.0, tmax[:, None])

    def _action_to_throttle(self, a):
        c = self.c
        s = self.dyn; p = self.params; I = self._I
        qw, qx, qy, qz = s.quat[:, 0], s.quat[:, 1], s.quat[:, 2], s.quat[:, 3]
        qv = s.quat[:, 1:4]
        r00 = 1 - 2 * (qy ** 2 + qz ** 2); r11 = 1 - 2 * (qx ** 2 + qz ** 2)
        r01 = 2 * (qx * qy - qw * qz); r10 = 2 * (qx * qy + qw * qz)
        r02 = 2 * (qx * qz + qw * qy); r12 = 2 * (qy * qz - qw * qx)
        r20 = 2 * (qx * qz - qw * qy); r21 = 2 * (qy * qz + qw * qx)
        r22 = 1 - 2 * (qx ** 2 + qy ** 2)
        w = s.w

        tmax = (s.kf * s.wm ** 2 * s.mg[:, 0])
        if c['vbatt_aware']:
            tmax_eff = tmax * (s.v_batt / p.v_ref) ** 2
        else:
            tmax_eff = tmax.copy()
        tau_rp = 2 * tmax_eff * self._ARM
        alpha_x = c['K_ALPHA'] * tau_rp / I[0]
        alpha_y = c['K_ALPHA'] * tau_rp / I[1]
        tau_y = 2 * tmax_eff * p.k_m_ratio

        # ---- 姿态误差（切换曲线坐标 x，满足 dx/dt ≈ ω）----
        theta = np.arccos(np.clip(r22, -1.0, 1.0))
        if c['reduced_att']:
            nx, ny = -r21, r20
            nn = np.hypot(nx, ny)
            wxy = np.hypot(w[:, 0], w[:, 1])
            sing = nn < 0.15
            fbx = np.where(wxy > 0.5, w[:, 0] / np.maximum(wxy, 1e-9), 1.0)
            fby = np.where(wxy > 0.5, w[:, 1] / np.maximum(wxy, 1e-9), 0.0)
            nx = np.where(sing, fbx, nx / np.maximum(nn, 1e-9))
            ny = np.where(sing, fby, ny / np.maximum(nn, 1e-9))
            ex_ = -theta * nx; ey_ = -theta * ny
            ez_ = np.zeros_like(theta)
        else:
            qn = np.linalg.norm(qv, axis=1) + 1e-12
            ang = 2 * np.arctan2(qn, np.clip(qw, -1, 1))
            ang2 = np.where(qw < 0, 2 * np.pi - ang, ang)
            psi = -(ang2[:, None] * qv / qn[:, None])
            ex_, ey_, ez_ = psi[:, 0], psi[:, 1], psi[:, 2]
        TL = c['T_LAG']
        sz = ez_ + w[:, 2] * np.abs(w[:, 2]) / (2 * tau_y / I[2])
        if c['reduced_att']:
            # 沿测地线方向 n 的单轴 bang-bang：标量坐标 p=-θ（目标 0），速度 v=ω·n
            # X 型机架 roll/pitch 力矩可行域为 L1 球 |tx|+|ty| ≤ 2·Tmax·arm → 沿 n 的最大力矩
            tau_n = tau_rp / (np.abs(nx) + np.abs(ny))
            # 力矩/推力取舍：倒扣半球降力矩 → 少往下推；正立半球降力矩 → 腾出集体推力向上
            tau_n = tau_n * np.where(r22 < c['R_LO'], c['K_TAU_INV'], c['K_TAU_UP'])
            I_n = I[0] * nx ** 2 + I[1] * ny ** 2
            alpha_n = c['K_ALPHA'] * tau_n / I_n
            v_par = w[:, 0] * nx + w[:, 1] * ny
            s_par = -theta + v_par * TL + v_par * np.abs(v_par) / (2 * alpha_n)
            u_par = -np.sign(s_par) * tau_n
            wpx = w[:, 0] - v_par * nx; wpy = w[:, 1] - v_par * ny      # 垂直分量只阻尼
            tx_bb = u_par * nx - c['KD_PERP'] * wpx
            ty_bb = u_par * ny - c['KD_PERP'] * wpy
        else:
            sx = ex_ + w[:, 0] * TL + w[:, 0] * np.abs(w[:, 0]) / (2 * alpha_x)
            sy = ey_ + w[:, 1] * TL + w[:, 1] * np.abs(w[:, 1]) / (2 * alpha_y)
            tx_bb = -tau_rp * np.sign(sx)
            ty_bb = -tau_rp * np.sign(sy)

        # ---- 模式：bang-bang（大倾角/高转速） ↔ PD（近正立），带迟滞 ----
        wxy = np.hypot(w[:, 0], w[:, 1])
        enter = (theta > c['TH_H'] + c['TH_HYS']) | (wxy > c['W_H'] + 5.0)
        leave = (theta < c['TH_H']) & (wxy < c['W_H'])
        self._bb = np.where(enter, True, np.where(leave, False, self._bb))
        bb = self._bb

        # ---- PD 段（沿用 v8：平飞误差 + 航向解耦水平定点 + 安全门控）----
        ex_level = r00 * r12 + r10 * (-r02)
        ey_level = r01 * r12 + r11 * (-r02)
        pe = self._target[:, 0:2] - s.pos[:, 0:2]; vw = s.vel[:, 0:2]
        axw = c['KP_POS_XY'] * pe[:, 0] - c['KD_VEL_XY'] * vw[:, 0]
        ayw = c['KP_POS_XY'] * pe[:, 1] - c['KD_VEL_XY'] * vw[:, 1]
        abx = r00 * axw + r10 * ayw; aby = r01 * axw + r11 * ayw
        d_ey = np.clip(abx / G, -c['MAX_TILT_RAD'], c['MAX_TILT_RAD'])
        d_ex = np.clip(-aby / G, -c['MAX_TILT_RAD'], c['MAX_TILT_RAD'])
        gate = (r22 > 0.65) & (s.pos[:, 2] > 0.8) & (s.vel[:, 2] > -0.5)
        beta = np.where(gate, np.clip((r22 - 0.65) / 0.25, 0.0, 1.0), 0.0)
        ex = ex_level + beta * d_ex; ey = ey_level + beta * d_ey
        tx_pd = c['KP_ATT'] * ex - c['KD_ATT'] * w[:, 0]
        ty_pd = c['KP_ATT'] * ey - c['KD_ATT'] * w[:, 1]
        tz_pd = -c['KZ_ATT'] * w[:, 2]

        tz_bb = (-tau_y * np.sign(sz)) if c['YAW_IN_FLIP'] else tz_pd

        tx = np.where(bb, tx_bb, tx_pd); ty = np.where(bb, ty_bb, ty_pd); tz = np.where(bb, tz_bb, tz_pd)
        # 自旋超前补偿：机体以 ωz 自旋时，电机滞后 T_DIR 内力矩方向会被转偏 → 预先反向旋转 (tx,ty)
        if c['T_DIR'] != 0.0:
            phi = -w[:, 2] * c['T_DIR']
            cph, sph = np.cos(phi), np.sin(phi)
            tx, ty = tx * cph - ty * sph, tx * sph + ty * cph
        # 高自旋时偏航阻尼（翻转段）
        if c['KZ_FLIP'] > 0.0:
            tz = np.where(bb, -c['KZ_FLIP'] * w[:, 2], tz)
        # 欧拉陀螺力矩前馈
        tx = tx + (I[2] - I[1]) * w[:, 1] * w[:, 2]
        ty = ty + (I[0] - I[2]) * w[:, 0] * w[:, 2]
        tz = tz + (I[1] - I[0]) * w[:, 0] * w[:, 1]

        # ---- 垂直通道 ----
        z_err = self._target[:, 2] - s.pos[:, 2]
        near = np.abs(z_err) < 0.35
        self._z_int = np.where(near & (r22 > 0.75), np.clip(self._z_int + z_err * 0.005, -0.15, 0.15),
                               self._z_int * 0.98)
        vz_ref = np.clip(-c['KZ_REF'] * self._z_est, c['VZ_DN'], c['VZ_UP'])
        Fz_pd = np.maximum(p.mass * (G + c['KV_Z'] * (vz_ref - self._vz_est) + c['KI_Z'] * self._z_int)
                           / np.maximum(r22, 0.5) * np.maximum(r22, 0.0), 0.3 * p.mass * G)
        Fmax = 4 * tmax_eff
        brake = (~bb) & (self._vz_est < c['BRAKE_VZ'])
        Fz_pd = np.where(brake, Fmax * c['BRAKE_TFRAC'], Fz_pd)
        Fz_bb = np.where(r22 > c['R_LO'], Fmax, Fmax * c['FLIP_FZ_LO'])
        Fz = np.where(bb, Fz_bb, Fz_pd)

        if c['prio_mixer']:
            yres = np.where((~bb) & (np.abs(w[:, 2]) > 2.0) & (r22 > 0.8), c['YAW_RES'] * tmax_eff, 0.0)
            T = self._allocate(Fz, tx, ty, tz, tmax_eff, yres)
        else:
            rhs = np.stack([Fz, tx, ty, tz], axis=1)
            T = np.clip(rhs @ self._Minv.T, 0.0, tmax_eff[:, None])
        u = np.sqrt(T / tmax_eff[:, None]) * np.sqrt(tmax_eff / tmax)[:, None] * (p.v_ref / s.v_batt)[:, None]
        return np.clip(u, 0.0, 1.0)
