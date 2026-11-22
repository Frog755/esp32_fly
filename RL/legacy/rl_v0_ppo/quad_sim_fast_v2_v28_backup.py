#!/usr/bin/env python3
"""
quad_sim 快速版：与 quad_sim_fixed.py 数学等价，但去掉 numpy 小数组开销。

原始每步 813us，瓶颈：np.cross（26%）、np.sum、np.array 反复构造、quat_to_rot 分配。
本版用纯 Python 标量 + 预分配 numpy 数组，目标 < 100us/步（8 倍以上加速）。

一致性验证：verify_fast.py 会跑 10000 步逐位对比固定版与快速版的 pos/vel/quat/omega。
"""
from __future__ import annotations

# --- [Stage 20 legacy 迁移引导] 本文件原位于 RL/env/，现移至 RL/legacy/rl_v0_ppo/ ---
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
import math

import numpy as np

G = 9.81

__all__ = ["QuadParams", "QuadDynamics", "ThrowRecoveryEnv", "quat_to_rot",
           "randomize", "pd_controller"]


class QuadParams:
    """与 quad_sim_fixed.QuadParams 字段完全一致（dataclass 字段是热路径瓶颈，改普通类）。"""

    __slots__ = ("mass", "inertia", "wheelbase", "rotor_pos", "rotor_dir",
                 "thrust_max_per_motor", "omega_max", "k_m_ratio",
                 "motor_tau_up", "motor_tau_down", "v_ref", "pwm_deadzone",
                 "v_full", "v_empty", "batt_capacity_mah", "batt_r_internal",
                 "current_max_per_motor", "drag_lin", "drag_ang",
                 "gyro_noise_std", "acc_noise_std", "gyro_bias_std",
                 "acc_bias_std", "obs_delay_steps", "alt_noise_std",
                 "dt_phys", "ctrl_hz", "k_f", "motor_gain")

    def __init__(self, mass=0.070, inertia=(5.0e-5, 5.8e-5, 1.0e-4),
                 wheelbase=0.122, rotor_pos=None, rotor_dir=(+1, -1, +1, -1),
                 thrust_max_per_motor=0.343, omega_max=3140.0, k_m_ratio=0.008,
                 motor_tau_up=0.030, motor_tau_down=0.045, v_ref=3.7,
                 pwm_deadzone=0.05, v_full=4.2, v_empty=3.5,
                 batt_capacity_mah=400.0, batt_r_internal=0.08,
                 current_max_per_motor=2.5, drag_lin=(0.02, 0.02, 0.03),
                 drag_ang=(2e-6, 2e-6, 4e-6), gyro_noise_std=0.01,
                 acc_noise_std=0.05, gyro_bias_std=0.02, acc_bias_std=0.1,
                 obs_delay_steps=1, alt_noise_std=0.02, dt_phys=0.001, ctrl_hz=200,
                 motor_gain=None):
        self.mass = mass
        self.inertia = tuple(inertia)
        self.wheelbase = wheelbase
        self.rotor_dir = tuple(rotor_dir)
        self.thrust_max_per_motor = thrust_max_per_motor
        self.omega_max = omega_max
        self.k_m_ratio = k_m_ratio
        self.motor_tau_up = motor_tau_up
        self.motor_tau_down = motor_tau_down
        self.v_ref = v_ref
        self.pwm_deadzone = pwm_deadzone
        self.v_full = v_full
        self.v_empty = v_empty
        self.batt_capacity_mah = batt_capacity_mah
        self.batt_r_internal = batt_r_internal
        self.current_max_per_motor = current_max_per_motor
        self.drag_lin = tuple(drag_lin)
        self.drag_ang = tuple(drag_ang)
        self.gyro_noise_std = gyro_noise_std
        self.acc_noise_std = acc_noise_std
        self.gyro_bias_std = gyro_bias_std
        self.acc_bias_std = acc_bias_std
        self.obs_delay_steps = obs_delay_steps
        self.alt_noise_std = alt_noise_std
        self.dt_phys = dt_phys
        self.ctrl_hz = ctrl_hz
        self.motor_gain = np.ones(4) if motor_gain is None else np.asarray(motor_gain, float)
        self._post()

    def _post(self):
        if self.wheelbase:
            a = self.wheelbase / 2 / math.sqrt(2)
            self.rotor_pos = np.array([[a, -a, 0.0], [-a, -a, 0.0],
                                       [-a, a, 0.0], [a, a, 0.0]])
        self.rotor_dir = np.asarray(self.rotor_dir, float)
        self.k_f = self.thrust_max_per_motor / self.omega_max ** 2

    @property
    def hover_throttle(self):
        return float(math.sqrt(self.mass * G / 4 / self.thrust_max_per_motor))

    @property
    def thrust_to_weight(self):
        return 4 * self.thrust_max_per_motor / (self.mass * G)


def randomize(p, rng, scale=1.0):
    """深拷贝再扰动，不丢字段（与修复版一致）。"""
    import copy
    q = copy.deepcopy(p)

    def r(x, pct):
        return x * (1 + scale * rng.uniform(-pct, pct))

    q.mass = r(p.mass, 0.10)
    q.inertia = tuple(r(i, 0.20) for i in p.inertia)
    q.k_m_ratio = r(p.k_m_ratio, 0.30)
    q.motor_tau_up = r(p.motor_tau_up, 0.40)
    q.motor_tau_down = r(p.motor_tau_down, 0.40)
    q.batt_r_internal = r(p.batt_r_internal, 0.50)
    q.drag_lin = tuple(r(d, 0.50) for d in p.drag_lin)
    q.drag_ang = tuple(r(d, 0.50) for d in p.drag_ang)
    if scale > 0:
        q.obs_delay_steps = int(rng.integers(0, 3))
    q.motor_gain = 1 + scale * rng.uniform(-0.08, 0.08, 4)
    q.k_f = q.thrust_max_per_motor / q.omega_max ** 2
    return q


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class QuadDynamics:
    """标量化的 6DoF 刚体。状态存普通 list，避免 numpy 小数组分配。"""

    __slots__ = ("p", "J", "Jinv", "motor_gain", "pos", "vel", "q", "w",
                 "rotor_w", "soc", "v_batt", "t", "_last_acc", "rp", "rd",
                 "drag_lin", "drag_ang", "kf", "mg", "m", "mintau", "mtau",
                 "tau_dead", "r_i", "cap", "imax", "wm", "vref", "ocv_lo",
                 "ocv_hi", "vf")

    def __init__(self, p):
        self.p = p
        self.J = list(p.inertia)
        self.Jinv = [1.0 / j for j in self.J]
        self.motor_gain = list(np.asarray(getattr(p, "motor_gain", np.ones(4)), float))
        self.rp = p.rotor_pos
        self.rd = list(p.rotor_dir)
        self.drag_lin = list(p.drag_lin)
        self.drag_ang = list(p.drag_ang)
        self.kf = p.k_f
        self.mg = self.motor_gain
        self.m = p.mass
        self.mintau = 1.0 / p.motor_tau_up
        self.mtau = 1.0 / p.motor_tau_down
        self.tau_dead = p.pwm_deadzone
        self.r_i = p.batt_r_internal
        self.cap = p.batt_capacity_mah / 1000.0
        self.imax = p.current_max_per_motor
        self.wm = p.omega_max
        self.vref = p.v_ref
        self.vf = p.v_full
        self.ocv_lo = p.v_empty
        self.ocv_hi = p.v_full - p.v_empty
        self.reset()

    def reset(self, pos=None, vel=None, quat=None, omega=None, soc=1.0):
        self.pos = [0.0, 0.0, 0.0] if pos is None else [float(x) for x in pos]
        self.vel = [0.0, 0.0, 0.0] if vel is None else [float(x) for x in vel]
        self.q = [1.0, 0.0, 0.0, 0.0] if quat is None else [float(x) for x in quat]
        self.w = [0.0, 0.0, 0.0] if omega is None else [float(x) for x in omega]
        self.rotor_w = [0.0, 0.0, 0.0, 0.0]
        self.soc = soc
        self.v_batt = self.ocv_lo + self.ocv_hi * soc
        self.t = 0.0
        self._last_acc = [0.0, 0.0, 0.0]

    def step(self, u, dt=None):
        dt = self.p.dt_phys if dt is None else dt
        rpm = self.rp
        kf = self.kf
        mg = self.mg
        wnow = self.rotor_w
        dead = self.tau_dead
        # 电池
        cur = 0.0
        ue = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            ui = u[i]
            if ui < 0.0:
                ui = 0.0
            elif ui > 1.0:
                ui = 1.0
            if ui < dead:
                ui = 0.0
            ue[i] = ui
            cur += self.imax * ui ** 1.5
        ocv = self.ocv_lo + self.ocv_hi * self.soc
        self.v_batt = max(ocv - cur * self.r_i, 2.8)
        self.soc = max(self.soc - cur * dt / 3600.0 / self.cap, 0.0)
        # 注意：原版用的是【更新后】的 v_batt 计算转速指令，这里必须一致
        vscale = self.v_batt / self.vref

        # 电机一阶滞后
        mintau, mtau, wm = self.mintau, self.mtau, self.wm
        T0 = T1 = T2 = T3 = 0.0
        for i in range(4):
            wc = ue[i] * wm * vscale
            wi = wnow[i]
            if wc > wi:
                wi += (wc - wi) * dt * mintau
            else:
                wi += (wc - wi) * dt * mtau
            wnow[i] = wi
            T = kf * wi * wi * mg[i]
            if i == 0:
                T0 = T
            elif i == 1:
                T1 = T
            elif i == 2:
                T2 = T
            else:
                T3 = T

        # 力矩（展开求和，避免 np.sum）
        y0, y1, y2, y3 = rpm[0, 1], rpm[1, 1], rpm[2, 1], rpm[3, 1]
        x0, x1, x2, x3 = rpm[0, 0], rpm[1, 0], rpm[2, 0], rpm[3, 0]
        tx = y0 * T0 + y1 * T1 + y2 * T2 + y3 * T3
        ty = -(x0 * T0 + x1 * T1 + x2 * T2 + x3 * T3)
        rd = self.rd
        tz = -(rd[0] * T0 + rd[1] * T1 + rd[2] * T2 + rd[3] * T3) * self.p.k_m_ratio
        Fz = T0 + T1 + T2 + T3

        # 四元数→旋转矩阵
        qw, qx, qy, qz = self.q
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qw * qz)
        r02 = 2 * (qx * qz + qw * qy)
        r10 = 2 * (qx * qy + qw * qz)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r12 = 2 * (qy * qz - qw * qx)
        r20 = 2 * (qx * qz - qw * qy)
        r21 = 2 * (qy * qz + qw * qx)
        r22 = 1 - 2 * (qx * qx + qy * qy)

        vx, vy, vz = self.vel
        # v_body = R^T @ v
        vbx = r00 * vx + r10 * vy + r20 * vz
        vby = r01 * vx + r11 * vy + r21 * vz
        vbz = r02 * vx + r12 * vy + r22 * vz

        dl = self.drag_lin
        Fbx = -dl[0] * vbx
        Fby = -dl[1] * vby
        Fbz = Fz - dl[2] * vbz

        wx, wy, wz = self.w
        da = self.drag_ang
        tx -= da[0] * wx
        ty -= da[1] * wy
        tz -= da[2] * wz

        # 世界系加速度 R @ F / m
        ax = (r00 * Fbx + r01 * Fby + r02 * Fbz) / self.m
        ay = (r10 * Fbx + r11 * Fby + r12 * Fbz) / self.m
        az = (r20 * Fbx + r21 * Fby + r22 * Fbz) / self.m - G

        # 陀螺项 ω × (J ω)
        J = self.J
        Jx, Jy, Jz = J[0] * wx, J[1] * wy, J[2] * wz
        gx = wy * Jz - wz * Jy
        gy = wz * Jx - wx * Jz
        gz = wx * Jy - wy * Jx
        Ji = self.Jinv
        dwx = Ji[0] * (tx - gx)
        dwy = Ji[1] * (ty - gy)
        dwz = Ji[2] * (tz - gz)

        # 半隐式欧拉
        self.vel = [vx + ax * dt, vy + ay * dt, vz + az * dt]
        self.pos = [self.pos[0] + self.vel[0] * dt,
                    self.pos[1] + self.vel[1] * dt,
                    self.pos[2] + self.vel[2] * dt]
        wx += dwx * dt
        wy += dwy * dt
        wz += dwz * dt
        self.w = [wx, wy, wz]

        # 四元数积分
        dqw = 0.5 * (-qx * wx - qy * wy - qz * wz)
        dqx = 0.5 * (qw * wx + qy * wz - qz * wy)
        dqy = 0.5 * (qw * wy - qx * wz + qz * wx)
        dqz = 0.5 * (qw * wz + qx * wy - qy * wx)
        qw += dqw * dt
        qx += dqx * dt
        qy += dqy * dt
        qz += dqz * dt
        n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        self.q = [qw / n, qx / n, qy / n, qz / n]

        self.t += dt
        self._last_acc = [ax, ay, az]
        return (T0, T1, T2, T3)

    def imu(self, rng, gyro_bias, acc_bias):
        qw, qx, qy, qz = self.q
        r02 = 2 * (qx * qz + qw * qy)
        r12 = 2 * (qy * qz - qw * qx)
        r22 = 1 - 2 * (qx * qx + qy * qy)
        awx, awy, awz = self._last_acc
        ax, ay, az = awx, awy, awz + G
        # specific force = R^T @ a
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qw * qz)
        r10 = 2 * (qx * qy + qw * qz)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r20 = 2 * (qx * qz - qw * qy)
        r21 = 2 * (qy * qz + qw * qx)
        sx = r00 * ax + r10 * ay + r20 * az
        sy = r01 * ax + r11 * ay + r21 * az
        sz = r02 * ax + r12 * ay + r22 * az
        g = self.p.gyro_noise_std
        a = self.p.acc_noise_std
        return ([self.w[0] + gyro_bias[0] + rng.normal(0, g),
                 self.w[1] + gyro_bias[1] + rng.normal(0, g),
                 self.w[2] + gyro_bias[2] + rng.normal(0, g)],
                [sx + acc_bias[0] + rng.normal(0, a),
                 sy + acc_bias[1] + rng.normal(0, a),
                 sz + acc_bias[2] + rng.normal(0, a)])


class ThrowRecoveryEnv:
    """与 quad_sim_fixed.ThrowRecoveryEnv 接口一致。"""

    metadata = {"render_modes": []}

    # Phase 2 奖励整体缩放系数（v19）。
    #   实测依据：相同随机策略 20 回合的回合回报均值
    #     phase=0 = -970.8，phase=1 = -405.0，phase=2 = -5536.2
    #   P2 是 phase0 的 5.7 倍 → value_loss 27088（p1_v18 的 262 倍）→
    #   policy_gradient_loss≈0 → 策略不更新 → 熵项把 std 从 1.0 推到 3.55。
    #   取 1/6 后 P2 量级 ≈ -923，与 phase=0 的 -971 基本对齐。
    #   等比例缩放不改变最优策略，只修数值可学性。
    P2_REWARD_SCALE = 1.0 / 6.0

    def __init__(self, params=None, domain_rand=1.0, episode_s=4.0,
                 descend_rate=0.3, difficulty=1.0, seed=None,
                 history_len=0, phase=0):
        """
        history_len: 历史堆叠帧数（0=不堆叠，保持原 22 维观测）
          实验 v8（History Stacking）用 history_len=3：
            额外堆叠最近 3 帧的 [ang_vel(3), last_action(4)] = 21 维
            总观测 = 22 + 21 = 43 维
          目的：让无记忆 MLP 也能看到时间趋势（电机惯性、观测延迟、当前翻滚阶段），
                而不引入 GRU/LSTM 的训练复杂度与推理开销。

        phase: 分阶段训练模式（0=端到端，保持历史行为）
          1 = Phase 1 翻转策略：
              初始姿态只采样在倒扣区（up_z ∈ [-1, +0.3]，约 tilt > 72°）
              奖励只关注姿态改善，不罚高度/水平速度
              终止条件：up_z > 0.7（成功交接）或坠地
          2 = Phase 2 稳定策略：
              初始姿态采样在接近正立区（up_z ∈ [0.6, 0.95]）
              带一定的下坠速度与角速度，模拟翻转结束时的状态
              奖励重罚角速度与 vz 偏差，鼓励精细控制
        """
        self.difficulty = difficulty
        self.base = params or QuadParams()
        self.dr = domain_rand
        self.episode_steps = int(episode_s * self.base.ctrl_hz)
        self.descend_rate = descend_rate
        self.rng = np.random.default_rng(seed)
        self.observation_space = None
        self.action_space = None
        # ---- 分阶段训练配置 ----
        self.phase = int(phase)
        # ---- History Stacking 配置 ----
        self.history_len = int(history_len)
        self.obs_dim = 22 + 7 * self.history_len   # 7 = ang_vel(3) + action(4)
        self._hist = None                          # 环形历史缓冲，reset 时清空

    def _throw_state(self):
        from env.quad_sim_fixed import quat_mul
        r, d = self.rng, self.difficulty

        def _rand_vec3(rng, mag_min, mag_max):
            """按「模长均匀 + 随机单位方向」采样三维向量。

            为什么不用 per-axis uniform(-m, m)：
              那样每个轴最大 m，但向量模长可达 m*sqrt(3)≈1.73m，
              会远超目标范围（实测 ω max 26，per-axis 采样会到 31.9）。
            """
            mag = rng.uniform(mag_min, mag_max)
            v = rng.normal(size=3)
            n = math.sqrt(float(v[0]**2 + v[1]**2 + v[2]**2)) + 1e-9
            return (v / n) * mag

        def _rand_omega(rng, scale):
            if scale == "MAG_COVERAGE":
                # P1 v18：模长 2~26 全覆盖（对齐端到端真实开局 max 24.7）
                return _rand_vec3(rng, 2.0, 26.0)
            if scale == "P2_CURRICULUM":
                # P2 v21 课程学习：角速度上限随难度 d 提升
                # d=0.2 时上限 14 rad/s；d=1.0 时上限 60 rad/s（覆盖真实交接极限）
                om_max = 2.0 + 58.0 * d
                return _rand_vec3(rng, 1.0, om_max)
            if scale is None:
                # P2 默认：直接按弧长均匀，模长 1~60 rad/s
                return _rand_vec3(rng, 1.0, 60.0)
            return rng.uniform(-1, 1, 3) * scale

        # ===== 分阶段采样 =====
        if self.phase == 1:
            # Phase 1 翻转：初始姿态只在倒扣区（tilt 从 72° 到 180°）
            tilt_min = math.radians(72.0)
            tilt_max = math.pi
            ang = r.uniform(tilt_min, tilt_max)
            omega_scale = "MAG_COVERAGE"
        elif self.phase == 2:
            # Phase 2 v21 课程学习重构：
            #
            # 病理（v18~v20 长期 0% 的核心阻碍之一）：
            #   此前 Phase 2 完全跳过了 difficulty 课程梯度，硬编码锁死在极限分布：
            #   ω 直接采样到 60 rad/s，倾角 32°~50°，下坠 2m/s，漂移 4.4m/s。
            #   四旋翼是开环极度不稳定系统，未训练的网络没有主动阻尼先验，
            #   面对 60 rad/s 的剧烈自旋，随机探索连续 50 步达标的概率严格为 0。
            #
            # 修正（按 d 平滑铺设能力阶梯）：
            #   d=0.2：近乎正立（tilt 8°~18°），ω≤14 rad/s，微弱漂移与下坠。
            #          让策略先学会【悬停定高与基础速率阻尼】（产生 ~0.7075 稳态推力）。
            #   d 逐步升高：倾角放宽至 32°~50°，ω 扩展至 60 rad/s，平移扩展至 4.4 m/s。
            tilt_min_deg = 5.0 + 26.8 * d    # d=0.2->10.4° (up_z~0.98); d=1.0->31.8° (up_z~0.85)
            tilt_max_deg = 10.0 + 39.5 * d   # d=0.2->17.9° (up_z~0.95); d=1.0->49.5° (up_z~0.65)
            ang = r.uniform(math.radians(tilt_min_deg), math.radians(tilt_max_deg))
            omega_scale = "P2_CURRICULUM"
        else:
            # 端到端（原行为）：tilt 从 0 到 difficulty*180°
            ang = r.uniform(0, d * math.pi)
            omega_scale = (2 + 13 * d)

        axis = r.normal(size=3)
        axis[2] = 0
        na = math.sqrt(axis[0] ** 2 + axis[1] ** 2) + 1e-9
        axis = axis / na
        q_tilt = np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]
        yaw = r.uniform(-math.pi, math.pi)
        q = quat_mul(np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), q_tilt)

        # Phase 2 需要在翻转结束时的高度（已掉了一段）开始
        # v17：P2 初始高度提到 2.0~3.0m —— 自由落体预算从 314 步增至 ~450 步，
        # 给"刹旋转+稳姿态+保高"三件事留出足够时间窗（v16 只活 118 步就坠地）
        z_lo, z_hi = (2.0, 3.0) if self.phase == 2 else (1.5, 2.5)
        # Phase 2 初始带下坠速度（翻转刚结束，通常 vz 为负；随难度从 -0.4 扩展至 -2.0）
        vz = r.uniform(-2.0 * d, 0.0) if self.phase == 2 else r.uniform(0.0, 2.5)
        xy = 0.3
        if self.phase == 2:
            # P2 v21 课程学习：水平速度按模长采样，上限 4.4*d（从 d=0.2 的 0.88m/s 扩展至 d=1.0 的 4.4m/s）
            vxy_max = max(0.2, 4.4 * d)
            vxy_vec = _rand_vec3(r, 0.0, vxy_max)[:2]
            vel = np.r_[vxy_vec, vz]
        else:
            vel = np.r_[r.uniform(-2, 2, 2) * d, vz]

        return dict(
            pos=[r.uniform(-xy, xy), r.uniform(-xy, xy), r.uniform(z_lo, z_hi)],
            vel=vel,
            quat=q,
            omega=_rand_omega(r, omega_scale),
            soc=r.uniform(0.5, 1.0))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.p = randomize(self.base, self.rng, self.dr)
        self.sim = QuadDynamics(self.p)
        self.sim.reset(**self._throw_state())
        self.gyro_bias = self.rng.normal(0, self.p.gyro_bias_std, 3)
        self.acc_bias = self.rng.normal(0, self.p.acc_bias_std, 3)
        self.last_action = np.zeros(4)
        self._prev_vz = 0.0
        # E-A/E-B 新增：记录上一帧姿态与倒扣驻留时间
        self._prev_up_z = float(1 - 2 * (self.sim.q[1] ** 2 + self.sim.q[2] ** 2))
        self._prev_vxy = float(math.sqrt(self.sim.vel[0] ** 2 + self.sim.vel[1] ** 2))
        self._tilt_dwell = 0          # 连续处于"未立起"状态的步数
        # ★ v19：P2 稳定达标计数与一次性奖励标志，必须逐回合复位。
        #   否则 +20 只会在【进程内第一回合】发放，之后全部哑掉 —— 这与 v8
        #   History Stacking 漏清 _hist 是同一类 bug（跨回合状态泄漏）。
        self._stable_steps = 0
        self._success_paid = False
        self.k = 0
        self.substeps = max(1, int(round(1.0 / (self.p.ctrl_hz * self.p.dt_phys))))
        self._dt_phys_eff = (1.0 / self.p.ctrl_hz) / self.substeps
        o = self._raw_obs()
        self.obs_buf = [o] * (self.p.obs_delay_steps + 1)
        # ★ History Stacking：reset 时必须清空历史，否则会串到上一个 episode 的帧
        if self.history_len > 0:
            self._hist = [self._hist_feature()] * self.history_len
            return self._stack_obs(self.obs_buf[0]).astype(np.float32), {}
        return self.obs_buf[0].astype(np.float32), {}

    # ---------- History Stacking 辅助 ----------
    def _hist_feature(self):
        """提取要堆叠的 7 维特征：ang_vel(3) + last_action(4)"""
        w = np.asarray(self.sim.w, dtype=np.float32) / 10.0   # 角速度归一化
        a = np.asarray(self.last_action, dtype=np.float32)
        return np.concatenate([w, a])

    def _raw_obs_safe(self):
        """失控时的安全观测：把 NaN/Inf 替换成 0，避免污染策略输入。"""
        o = np.asarray(self._raw_obs(), dtype=np.float32)
        o = np.nan_to_num(o, nan=0.0, posinf=10.0, neginf=-10.0)
        return o

    def _stack_obs(self, base_obs):
        """把当前帧特征压入历史，并拼出完整观测"""
        self._hist.pop(0)
        self._hist.append(self._hist_feature())
        return np.concatenate([base_obs] + self._hist)

    def _raw_obs(self):
        s = self.sim
        gyro, acc = s.imu(self.rng, self.gyro_bias, self.acc_bias)
        qw, qx, qy, qz = s.q
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qw * qz)
        r10 = 2 * (qx * qy + qw * qz)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r20 = 2 * (qx * qz - qw * qy)
        r21 = 2 * (qy * qz + qw * qx)
        R6 = np.array([r00, r10, r20, r01, r11, r21]) + self.rng.normal(0, 0.01, 6)
        vel = np.array(s.vel) + self.rng.normal(0, 0.05, 3)
        z = s.pos[2] + self.rng.normal(0, self.p.alt_noise_std)
        vb = (s.v_batt - 3.7) / 0.5
        return np.concatenate([R6, np.array(gyro) / 10, vel / 3, [z / 2],
                               self.last_action, [vb],
                               [0.0 if self.phase == 2 else self.descend_rate],
                               np.array(acc) / G])

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        u = (a + 1) / 2
        for _ in range(self.substeps):
            self.sim.step(u, dt=self._dt_phys_eff)
        self.k += 1

        # v18 物理稳定性守卫：角速度超过 60 rad/s 判定失控终止。
        #   原因：环境用线性力矩模型 + 微小气动阻尼（drag_ang≈2e-6），
        #   ω 一旦超过 ~200 rad/s 就会数值爆炸（实测 NaN）。
        #   真实四旋翼在此转速下桨叶已失速、推力模型失效，
        #   所以"超过 60 rad/s"在物理上等价于"已经失控坠机"，
        #   终止它是合理建模，而不是掩盖问题。
        _om_guard = math.sqrt(self.sim.w[0]**2 + self.sim.w[1]**2 + self.sim.w[2]**2)
        if _om_guard > 60.0 or not math.isfinite(_om_guard):
            self.obs_buf.append(self._raw_obs_safe())
            obs = self.obs_buf.pop(0)
            info = dict(up_z=0.0, omega_norm=60.0, z=float(self.sim.pos[2]),
                        vz=float(self.sim.vel[2]), v_batt=float(self.sim.v_batt))
            return obs.astype(np.float32), -10.0, True, False, info

        s = self.sim
        qw, qx, qy, qz = s.q
        up_z = 1 - 2 * (qx * qx + qy * qy)
        wx, wy, wz = s.w
        omega_norm = math.sqrt(wx * wx + wy * wy + wz * wz)
        vx, vy, vz = s.vel
        # ===== v3 奖励：修正「躺平比翻转更划算」的收益倒挂 =====
        #
        # 诊断依据（eval.py + report.py 固定种子集 100 回合实测）：
        #   RL 成功率仅 1~3%，其中 42~61% 是 crash_not_upright（根本没翻正）。
        #   旧奖励账本：躺平 -2.8/步 vs 努力翻转 -5.1/步 → 策略理性选择躺平。
        #
        # 三处改动：
        #   1) r_up 由「正立给奖」改为「倒扣给罚」，躺平代价 -3 → -6
        #   2) r_rate 按 up_z 分档：躺着罚得轻（允许暴力翻转），立起重罚（要求稳住）
        #      注意：三档系数全部为负，只调「扣多少」，绝不改成正奖励，
        #      否则策略会学出「保持倒扣+高速旋转」来白刷分。
        #   3) r_vz 触发条件由「前 0.5s 不罚」改为「up_z < 0.7 不罚」
        #      姿态没立起来时苛责高度没有意义，立起来后才真正要求控速。

        # --- 1) 姿态项：躺着就是错 ---
        r_up = -3.0 * (1.0 - up_z)

        # --- 2) 角速度项：分档惩罚，姿态越差越宽容 ---
        if up_z < 0.3:
            k_rate = 0.003     # 倒扣/大角度：几乎不痛，鼓励果断翻转
        elif up_z < 0.8:
            k_rate = 0.01      # 翻转中途：原力度
        else:
            k_rate = 0.03      # 已立起：罚 3 倍，逼它稳住别晃
        r_rate = -k_rate * omega_norm ** 2

        r_bonus = 1.0 if up_z > 0.9 else 0.0               # 明确的"已回正"里程碑

        # --- 4) 水平速度项（v5 改动：全局惩罚 → 按姿态分档）---
        # v4 的失败教训：直接把系数 0.2→0.6（全局惩罚）不仅没降 out_of_bounds，
        #   反而让 timeout_unstable 从 13/6/8/1 暴涨到 21/19/17/6。
        # 原因：倒扣翻滚时物理上必然伴随水平速度，全局惩罚等于给「翻转」这条路加税，
        #   策略被迫在「冒险翻转」和「保守不动」之间又倒向后者。
        # v5 修正：只在姿态已经立起来（up_z>0.7）之后才要求收敛水平速度。
        #   这与 r_vz 的设计哲学一致 —— 姿态没立起来时先别管平移。
        if up_z < 0.7:
            r_vxy = 0.0
        else:
            r_vxy = -0.6 * math.sqrt(vx * vx + vy * vy)

        # --- 3) vz 项：按姿态触发，而非按时间 ---
        if up_z < 0.7:
            r_vz = 0.0         # 还没立起来，不管高度
        else:
            r_vz = -0.3 * abs(vz + self.descend_rate)
        r_act = -0.03 * float(np.sum((a - self.last_action) ** 2))   # 略放宽，允许果断动作

        # --- 5) E-A 有效旋转奖励（v6）---
        # 设计目标：消除"倒扣时策略犹豫不决"的问题，同时彻底堵死刷分。
        #
        # 为什么不直接奖励 |ω|（"角加速度最大化"）：
        #   给正奖励策略就会无限刷 —— 学出"永远高速旋转"的反行为。
        #   这正是 v4 的教训：任何正奖励都可能被找到漏洞。
        #
        # 核心思路：奖励的是「有效旋转」= 角速度中真正让姿态变正的那部分。
        #   delta_up_z = 本帧 up_z - 上帧 up_z
        #   只有当 up_z 在增大（机身正在翻正）时才给奖励，且奖励大小与改善速率成正比。
        #
        # 这个设计自动排除了所有无效旋转：
        #   - 绕 z 轴（yaw）转 → 不改变倾角 → delta_up_z≈0 → 拿不到分
        #   - 绕 xy 轴但方向错 → 越转越歪 → delta_up_z<0 → 拿不到分
        #   - 高速乱转 → 倾角没改善 → 拿不到分
        # 因此根本不需要"按轴区分奖励大小"，有效性判断已内含其中。
        dt_ctrl = 1.0 / self.p.ctrl_hz
        delta_up_z = up_z - self._prev_up_z

        if up_z < 0.7 and delta_up_z > 0:
            # 只在"尚未立起"且"正在变正"时奖励，奖励量 = 每秒改善多少 up_z
            #
            # ⚠️ E-A 第一次尝试用了 0.8 * min(..., 8.0)，单步上限高达 +6.4，
            #    与 r_up 满罚 -6.0 同级 → 策略学会"只要翻得快，高度掉光也无所谓"，
            #    1000 回合严格评估成功率从 54% 崩到 2.3%，crash_upright 占 69%。
            # 修正：系数降到 1/10（0.08）、上限降到 3.0，单步最大 +0.24。
            #    这样它只能起"引导翻转方向"的作用，无法抵消放弃高度控制的代价。
            r_effective_rot = 0.08 * min(delta_up_z / dt_ctrl, 3.0)
        else:
            r_effective_rot = 0.0

        # 倒扣驻留计数（供 E-B 使用）
        if up_z < 0.7:
            self._tilt_dwell += 1
        else:
            self._tilt_dwell = 0

        # ===== 分阶段奖励（phase 1/2 覆盖端到端逻辑）=====
        if self.phase == 1:
            # Phase 1 = 纯翻转（v16 架构重构：mj 定调的职责边界）
            #
            # 架构原则（重要）：分阶段的职责必须按"动作类型"划分，不能按姿态阈值划分。
            #   P1 = 加速/转向类（施加力矩改变姿态方向）→ 只管姿态
            #   P2 = 阻尼/减速类（刹旋转 + 刹平移 + 收姿态）→ 只管速度收敛
            # 这样才能避免 P1 内部出现"既加速翻正、又减速刹车"的矛盾目标。
            #
            # 演进历史（为什么最终回到纯姿态）：
            #   v12: up_z>0.7 即终止 → 交接 ω 达 26 rad/s，而 P2 只见过 0.5~4 → 联合崩到 8%
            #        归因错误：当时以为"P1 甩太快"，实际是"P2 训练分布太窄"
            #   v13/v13b: 逼 P1 必须把 ω 刹到 <4 才交接
            #        → P1 背上矛盾目标；极端 ω 下永远刹不住 → 握着控制权漂移出界
            #        → 联合 out_of_bounds 占 56%
            #   v16（本版本）: P1 回归纯翻转，翻正即交（up_z>0.7）；
            #        速度全部交给 P2 处理，P2 训练分布改为【实测交接分布】覆盖 ω≤26
            #        —— 下游适配上游的真实输出，而不是上游阉割自己迁就下游
            reward = r_up + r_effective_rot + r_bonus + 0.5

            # 翻正即交接：不看 ω、不看水平速度（这些是 P2 的活）
            if up_z > 0.7:
                reward += 20.0
                terminated = True
            else:
                terminated = False
        elif self.phase == 2:
            # Phase 2 稳定（v26 姿态死区放宽 + 水平刹车闭环 + 2米安全区阻尼）：
            #
            # 1. 姿态死区放宽至 up_z > 0.95 (倾角 <= 18.2°)：
            #    - 允许机身倾斜 15°~18° 打出 3m/s² 反向加速度进行主动刹车，且升力仅损耗 5%；
            #    - up_z >= 0.95 时仅微弱二次惩罚不苛责，给刹车留出充分角度自由；
            #    - up_z < 0.95 时强力回正。
            if up_z >= 0.95:
                r_up = -1.0 * (1.0 - up_z)
            else:
                r_up = -5.0 * (0.95 - up_z) - 0.05

            # 2. 暂态翻正有效旋转补偿 (保持开启)
            delta_up_z = up_z - self._prev_up_z
            dt_ctrl = 1.0 / self.p.ctrl_hz
            if up_z < 0.95 and delta_up_z > 0:
                r_effective_rot = 0.15 * min(delta_up_z / dt_ctrl, 4.0)
            else:
                r_effective_rot = 0.0

            # 3. 角速度阻尼（v28：连续插值，但保留大倾角区间的强阻尼）
            #
            # 【v27 的错误】把 k_quad 在大倾角压到 0.005、k_lin 压到 0，
            #   本意是"别卡住翻转"。实测代价：d=0.3 的「角速度达标步数」从 v24 的 25
            #   掉到 11，d=1.0 甚至掉到 0 —— 大倾角下完全不管自旋，机身带着 10+ rad/s
            #   一路翻下去，永远等不到姿态稳定。
            #   教训：大倾角确实要宽容，但绝不能"零阻尼"，ω 越界就是失控。
            #
            # 【v28】保留插值形式（保证 C0 连续、无断点陷阱），但把区间压回
            #   [0.85, 0.95]，且两端都保留足够的阻尼强度：
            #     up_z <= 0.85：-(0.05*ω + 0.008*ω²)  —— 仍能压住 10 rad/s 自旋
            #     up_z >= 0.95：-(0.30*ω + 0.020*ω²)  —— 立直后强力稳压到 ω<1.2
            #   相比 v26 的硬分档，这里没有跳变（不会再诱发"卡在 0.89 高速旋转"）。
            _t = min(1.0, max(0.0, (up_z - 0.85) / 0.10))
            k_lin = 0.05 + 0.25 * _t
            k_quad = 0.008 + 0.012 * _t
            r_rate_stab = -(k_lin * omega_norm + k_quad * omega_norm ** 2)

            # 4. 回正里程碑奖励
            r_bonus = 1.0 if up_z > 0.95 else 0.0

            # 5. 稳态收油门定高阻尼（v27：加入"推力可达性"门控，破除满油门力矩死锁）
            #
            # 【病灶】v23~v26 一律对下坠施加强惩罚，从不检查当前姿态下油门还剩多少余量。
            #   物理硬约束：维持高度所需总推力 = mg / up_z，而四电机最大总推力 4*Tmax = 2mg。
            #      up_z = 0.85 (倾角 32°) → 需平均油门 0.767，力矩余量 23%；
            #      up_z = 0.60 (倾角 53°) → 需平均油门 0.913，力矩余量 8.7%；
            #      up_z = 0.50 (倾角 60°) → 需平均油门 1.000，力矩余量 0%；
            #      up_z < 0.50 (倾角 >60°) → 需推力已超物理上限，必然下坠。
            #   在大倾角区间继续重罚 vz，等于逼策略去完成一件物理上做不到的事。
            #   它唯一还能拿到的"最优解"就是把四个电机全部推到 1.0
            #   （实测 v26 姿态崩溃瞬间 max(u)=1.000 占 100%，mean(u)=0.926）。
            #   而四电机同时贴顶 ⇒ 差动力矩裕度归零 ⇒ up_z 再也爬不回来 ⇒
            #   100% 推力指向水平 ⇒ 6 m/s 横冲 ⇒ out_of_bounds（占 v26 失败回合 88%）。
            #
            # 【修复】按"推力可达性"给出 vz_gate：
            #   up_z >= 0.95（倾角<=18°）：门控 1.0，全力定高（余量 27%，绰绰有余）；
            #   0.40 <= up_z < 0.95       ：线性衰减，越歪越不罚，把油门让给差动力矩；
            #   up_z < 0.40（倾角>66°）   ：门控 0，完全豁免 —— 此时唯一有效功就是回正。
            #   原则：物理做不到的事不扣分，先把油门裕度留给力矩，姿态救回来再谈定高。
            # 【v27 → v28 关键修正：门控必须留保底，绝不可归零】
            #
            #  v27 的错误（实测 crash 从 8% 暴涨到 96%，vz_ok 从 143 跌到 22）：
            #    把 vz_gate 在 up_z<0.40 直接降到 0，本意是"物理做不到就不要罚"。
            #    但门控归零等于【彻底删掉了油门的托底梯度】：
            #      · v26 大倾角下坠 -2m/s：r_vz = -0.333/步（强梯度，逼它推油门）
            #      · v27 大倾角下坠 -2m/s：r_vz = -0.000/步（无梯度，推不推都一样）
            #    策略于是学到的不是"先回正再定高"，而是"大倾角时可以不管高度"
            #    → 自由落体 → 砸地 48/50。
            #    教训：豁免惩罚 ≠ 降低要求；它同时抹掉了"该往哪个方向使劲"的信号。
            #
            #  v28 修正：门控只降低倍数、保留方向性梯度，下限钉在 VZ_GATE_FLOOR。
            #    任何姿态下坠都必须有一个"推油门能减分"的斜率存在，否则策略必然放弃保高。
            VZ_GATE_FLOOR = 0.40
            vz_gate = max(VZ_GATE_FLOOR, min(1.0, max(0.0, (up_z - 0.40) / 0.55)))

            # 【关键：惩罚封顶，保证单调性】
            #   若直接写 r_vz = vraw * gate，则 gate 随 up_z 上升而变大，
            #   惩罚幅度也同步变大，其斜率 |vraw|*∂gate/∂up_z 会超过 r_up 的斜率 5.0，
            #   造成"姿态变好反而掉分"的局部最优（实测 10 个非单调点）。
            #   对幅度封顶 CAP=2.0 后，斜率上界 = 2.0*(1/0.55) = 3.64 < 5.0，严格单调。
            VZ_PEN_CAP = 2.0
            if vz >= 0:
                vz_mag = 0.3 * vz
            else:
                vz_mag = 0.8 * abs(vz) + 1.2 * max(0.0, -vz - 0.5)
            if up_z >= 0.95:
                vz_mag = 0.8 * abs(vz) + 1.2 * max(0.0, abs(vz) - 0.5)
            r_vz_stab = -vz_gate * min(vz_mag, VZ_PEN_CAP)

            # 6. 水平漂移阻尼与稳定阶段有效刹车补偿（v27：改成连续淡入，消除 up_z=0.90 断点）
            #
            # 【病灶】v22~v26 是 if up_z >= 0.90: r_vxy = -(...) else: 0.0 的硬开关。
            #   姿态刚跨过 0.90 时 r_vxy 会从 0 突然跳到 -0.267（scaled），
            #   造成"姿态变好反而掉分"的非单调台阶，策略会故意停在 0.89x 规避。
            #   实测该台阶在 ω=6 rad/s 时导致 10 个非单调点。
            #
            # 【修复】用 vxy_gate 从 up_z=0.85 到 0.95 线性淡入，全程 C0 连续。
            #   注意：r_effective_brake（刹车正奖励）保留严格前置锁 ——
            #   它只有在"已立直 + 真有漂移 + 真在减速"时才发放，避免大倾角乱晃刷分。
            vxy_gate = min(1.0, max(0.0, (up_z - 0.85) / 0.10))
            vxy_now = math.sqrt(vx * vx + vy * vy)
            delta_vxy = vxy_now - self._prev_vxy
            r_vxy_stab = vxy_gate * (-(1.2 * vxy_now + 0.8 * max(0.0, vxy_now - 0.5)))
            if up_z >= 0.90 and vxy_now > 0.3 and delta_vxy < 0:
                r_effective_brake = 0.10 * min(-delta_vxy / dt_ctrl, 2.5)
            else:
                r_effective_brake = 0.0

            # 7. 2米安全区软势场阻尼（同样连续淡入，避免边界台阶）
            r_xy = math.sqrt(s.pos[0] ** 2 + s.pos[1] ** 2)
            if r_xy > 2.0:
                r_boundary = vxy_gate * (-0.5 * (r_xy - 2.0) ** 2)
            else:
                r_boundary = 0.0

            # ★ 全局缩放 1/6
            reward = (r_up + r_rate_stab + r_effective_rot + r_bonus +
                      r_vz_stab + r_vxy_stab + r_effective_brake + r_boundary +
                      r_act + 0.5) * self.P2_REWARD_SCALE

            # 稳定达标判据：up_z > 0.95 且 |vz| < 0.5 且 omega < 1.2，持续 50 步
            stable_now = (up_z > 0.95 and abs(vz) < 0.5 and omega_norm < 1.2)
            self._stable_steps = self._stable_steps + 1 if stable_now else 0
            if self._stable_steps >= 50 and not self._success_paid:
                reward += 20.0 * self.P2_REWARD_SCALE
                self._success_paid = True      # 一次性奖励，不重复发放
            terminated = False
        else:
            reward = (r_up + r_rate + r_bonus + r_vxy + r_vz + r_act
                      + r_effective_rot + 0.5)
            terminated = False
        self._prev_up_z = up_z
        self._prev_vxy = float(math.sqrt(vx * vx + vy * vy))

        self.last_action = a
        z, vz_now = float(s.pos[2]), float(vz)
        # 坠地判定（所有模式共享）
        # v19：phase=2 的坠地奖惩也乘同一缩放系数，否则它相对 per-step 奖励
        #   会被放大 6 倍（坠落 -25 变成"超重惩罚"），改变最优策略。
        #   原则：同一 phase 内所有奖励项必须用同一个缩放系数。
        _scale = self.P2_REWARD_SCALE if self.phase == 2 else 1.0
        if z < 0.08 and self._prev_vz < 0:
            approach_vz = abs(self._prev_vz)
            soft = up_z > 0.9 and approach_vz < 0.8 and abs(vz_now) < 2.0
            reward += (15.0 if soft else -25.0) * _scale
            terminated = True
        self._prev_vz = vz_now
        if math.sqrt(s.pos[0] ** 2 + s.pos[1] ** 2) > 5 or s.pos[2] > 6:
            terminated = True
        truncated = self.k >= self.episode_steps

        self.obs_buf.append(self._raw_obs())
        obs = self.obs_buf.pop(0)
        info = dict(up_z=up_z, omega_norm=omega_norm, z=z, vz=vz_now,
                    v_batt=float(s.v_batt))
        if self.history_len > 0:
            obs = self._stack_obs(obs)
        return obs.astype(np.float32), float(reward), terminated, truncated, info


def pd_controller(sim, p, vz_ref=-0.3):
    """与修复版一致的 PD（仅用于基线对照）。"""
    qw, qx, qy, qz = sim.q
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)
    r10 = 2 * (qx * qy + qw * qz)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qw * qx)
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    zx, zy, zz = r02, r12, r22
    # e_rot = R^T @ (z_b × e_z)
    cx = zy * 1.0 - zz * 0.0
    cy = zz * 0.0 - zx * 1.0
    cz = 0.0
    ex = r00 * cx + r10 * cy + r20 * cz
    ey = r01 * cx + r11 * cy + r21 * cz
    ez = r02 * cx + r12 * cy + r22 * cz
    kp, kd = 0.020, 0.0020
    wx, wy, wz = sim.w
    tx = kp * ex - kd * wx
    ty = kp * ey - kd * wy
    tz = -0.0004 * wz
    tilt_ok = max(zz, 0.0)
    Fz = p.mass * (G + 3.0 * (vz_ref - sim.vel[2])) / max(zz, 0.5) * tilt_ok
    Fz = float(max(Fz, 0.3 * p.mass * G))
    rp = p.rotor_pos
    M = np.vstack([np.ones(4), rp[:, 1], -rp[:, 0], -p.rotor_dir * p.k_m_ratio])
    T = np.linalg.solve(M, np.r_[Fz, tx, ty, tz])
    T = np.clip(T, 0, p.thrust_max_per_motor)
    return np.sqrt(T / p.thrust_max_per_motor) * (p.v_ref / max(sim.v_batt, 3.0))
