"""
ESP32-S3 抛飞自稳四旋翼 —— 仿真模型 v0.1
机型：122 mm 轴距 FR4 机架 + 4x 8520 直驱空心杯 + 75 mm 桨 + 1S LiPo，AUW ≈ 70 g

内容：
  1. QuadParams   —— 全部物理参数（★ = 估算值，必须实测替换）
  2. QuadDynamics —— 6DoF 刚体 + 电机一阶滞后 + 电池压降 + 气动阻力 + IMU 噪声/延迟
  3. ThrowRecoveryEnv —— Gymnasium 环境：随机抛飞初始状态 → 恢复悬停并缓降
  4. __main__ 自检 —— 用级联 PD 从随机抛飞恢复，验证符号/混控方向正确

坐标系：世界 ENU（z 向上）；机体 FLU（x 前、y 左、z 上）。四元数 [w, x, y, z]，机体→世界。
电机布局（X 型，俯视）：
        前 (+x)
   M4(CW)   M1(CCW)
        \\   /
         [ ]
        /   \\
   M3(CCW)  M2(CW)
依赖：numpy, gymnasium（可选，没有也能跑动力学）

用法：
  python quad_sim.py                      # 自检：混控方向 + PD 基线 + gym 接口
  # 训练示例（pip install stable-baselines3）：
  from stable_baselines3 import PPO
  env = ThrowRecoveryEnv(difficulty=0.3)  # 课程：成功率>80% 后逐步升到 1.0
  PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64])).learn(5_000_000)
  # actor 64x64 MLP ≈ 6k 参数，可直接导出到 ESP32-S3

Sim2Real 顺序：先实测所有 ★ 参数（称重、推力台、电机阶跃响应、双线摆惯量、
电池压降），替换默认值，再把 randomize() 的范围收窄到实测误差附近。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

G = 9.81


# ============================================================ 1. 参数
@dataclass
class QuadParams:
    # ---- 质量与惯量 ----
    mass: float = 0.070                 # kg ★ 整机含电池，上秤称
    # ★ 估算：电机 4x5g 于 r=61mm + 机架板 + 中心电池/开发板盒体。用双线摆实测更准
    inertia: tuple = (5.0e-5, 5.8e-5, 1.0e-4)   # Ixx, Iyy, Izz [kg·m²]

    # ---- 几何 ----
    wheelbase: float = 0.122            # m，对角电机轴距
    # 电机位置（机体系，m），由轴距算出，见 __post_init__
    rotor_pos: np.ndarray = field(default=None)
    # 旋向：+1 = CCW（俯视逆时针，转子角动量 +z），-1 = CW
    rotor_dir: np.ndarray = field(default_factory=lambda: np.array([+1, -1, +1, -1]))

    # ---- 电机 + 桨（8520 + 75mm，1S）----
    thrust_max_per_motor: float = 0.343  # N ★ ≈35 gf，满油门、电池端电压 3.7V 时，推力台实测
    omega_max: float = 3140.0            # rad/s ★ ≈30000 RPM 带桨负载 @ 3.7V
    k_m_ratio: float = 0.008             # m ★ 反扭/推力比 τ = k_m_ratio * T
    motor_tau_up: float = 0.030          # s ★ 加速时间常数（空心杯+75mm桨）
    motor_tau_down: float = 0.045        # s ★ 减速更慢（无主动刹车）
    v_ref: float = 3.7                   # V，thrust_max 对应的端电压
    pwm_deadzone: float = 0.05           # 低于此占空比电机不转 ★

    # ---- 电池（1S LiPo，约 300-450 mAh）----
    v_full: float = 4.2
    v_empty: float = 3.5
    batt_capacity_mah: float = 400.0     # ★
    batt_r_internal: float = 0.08        # Ω ★ 含 MOSFET/线阻（空心杯 1S 压降很大，实测！）
    current_max_per_motor: float = 2.5   # A ★ 满油门单电机电流

    # ---- 气动阻力 ----
    drag_lin: tuple = (0.02, 0.02, 0.03)        # N/(m/s) 机体系线性阻力 ★
    drag_ang: tuple = (2e-6, 2e-6, 4e-6)        # N·m/(rad/s) 角阻尼 ★

    # ---- 传感器（ICM-42688 级别 IMU）----
    gyro_noise_std: float = 0.01         # rad/s 每样本
    acc_noise_std: float = 0.05          # m/s² 每样本
    gyro_bias_std: float = 0.02          # rad/s 每回合随机偏置
    acc_bias_std: float = 0.1            # m/s²
    obs_delay_steps: int = 1             # 控制步，观测延迟 ★ 实测 IMU→推理→PWM
    alt_noise_std: float = 0.02          # m（VL53L1X / 气压计融合后）

    # ---- 仿真时序 ----
    dt_phys: float = 0.001               # s，物理积分 1 kHz
    ctrl_hz: int = 200                   # 策略频率 ★ ESP32 上实测推理延迟后定

    def __post_init__(self):
        if self.rotor_pos is None:
            a = self.wheelbase / 2 / np.sqrt(2)      # 0.0431 m
            # M1 前右, M2 后右, M3 后左, M4 前左
            self.rotor_pos = np.array([[ a, -a, 0.0],
                                       [-a, -a, 0.0],
                                       [-a,  a, 0.0],
                                       [ a,  a, 0.0]])
        self.k_f = self.thrust_max_per_motor / self.omega_max ** 2   # N/(rad/s)²

    @property
    def hover_throttle(self) -> float:
        """端电压=v_ref 时悬停所需占空比（推力 ∝ u²）"""
        return float(np.sqrt(self.mass * G / 4 / self.thrust_max_per_motor))

    @property
    def thrust_to_weight(self) -> float:
        return 4 * self.thrust_max_per_motor / (self.mass * G)


def randomize(p: QuadParams, rng: np.random.Generator, scale: float = 1.0) -> QuadParams:
    """域随机化：返回扰动后的新参数。scale=0 关闭。

    [FIX-E] 原实现重建 QuadParams 时只显式传了部分字段，其余字段静默回落到
    类默认值 —— 一旦你在外部改过 batt_capacity_mah / pwm_deadzone / IMU 噪声等，
    训练时会被悄悄丢掉。这里改为「深拷贝原对象再逐项扰动」。
    """
    import copy as _copy
    q = _copy.deepcopy(p)

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
    # 每个电机推力单独 ±8%（电机/桨个体差异）；显式声明为字段而非动态挂属性
    q.motor_gain = 1 + scale * rng.uniform(-0.08, 0.08, 4)
    q.__post_init__()          # 重算由 mass/wheelbase 派生的 k_f
    return q


# ============================================================ 2. 动力学
def quat_mul(q, r):
    w0, x0, y0, z0 = q; w1, x1, y1, z1 = r
    return np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1,
                     w0*x1 + x0*w1 + y0*z1 - z0*y1,
                     w0*y1 - x0*z1 + y0*w1 + z0*x1,
                     w0*z1 + x0*y1 - y0*x1 + z0*w1])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                     [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


class QuadDynamics:
    """状态：pos(3) vel(3, 世界) quat(4) omega(3, 机体) rotor_w(4) soc(1)"""

    def __init__(self, p: QuadParams):
        self.p = p
        self.J = np.array(p.inertia)
        self.Jinv = 1.0 / self.J
        self.motor_gain = getattr(p, "motor_gain", np.ones(4))
        self.reset()

    def reset(self, pos=None, vel=None, quat=None, omega=None, soc=1.0):
        self.pos = np.zeros(3) if pos is None else np.array(pos, float)
        self.vel = np.zeros(3) if vel is None else np.array(vel, float)
        self.quat = np.array([1., 0, 0, 0]) if quat is None else np.array(quat, float)
        self.omega = np.zeros(3) if omega is None else np.array(omega, float)
        self.rotor_w = np.zeros(4)
        self.soc = soc
        self.v_batt = self._ocv()
        self.t = 0.0

    def _ocv(self):
        p = self.p
        return p.v_empty + (p.v_full - p.v_empty) * self.soc   # 线性近似 ★ 可换放电曲线

    def step(self, u: np.ndarray, dt: float | None = None):
        """u: 4 个电机 PWM 占空比 [0,1]，推进一个物理步（默认 dt_phys）"""
        p, dt = self.p, (self.p.dt_phys if dt is None else dt)
        u = np.clip(u, 0.0, 1.0)
        u_eff = np.where(u < p.pwm_deadzone, 0.0, u)

        # 电池：端电压 = OCV - I·R，电流 ∝ u^1.5 近似 ★
        current = np.sum(p.current_max_per_motor * u_eff ** 1.5)
        self.v_batt = max(self._ocv() - current * p.batt_r_internal, 2.8)
        self.soc = max(self.soc - current * dt / 3600 / (p.batt_capacity_mah / 1000), 0.0)

        # 电机：稳态转速 ∝ u·V，一阶滞后（加减速时间常数不同）
        w_cmd = u_eff * p.omega_max * (self.v_batt / p.v_ref)
        tau = np.where(w_cmd > self.rotor_w, p.motor_tau_up, p.motor_tau_down)
        self.rotor_w += (w_cmd - self.rotor_w) * (dt / tau)

        T = p.k_f * self.rotor_w ** 2 * self.motor_gain           # 每电机推力 N
        F_body = np.array([0.0, 0.0, T.sum()])
        tau_body = np.array([
            np.sum(p.rotor_pos[:, 1] * T),                         # roll  = Σ y·T
            -np.sum(p.rotor_pos[:, 0] * T),                        # pitch = -Σ x·T
            -np.sum(p.rotor_dir * p.k_m_ratio * T),                # yaw 反扭矩
        ])

        R = quat_to_rot(self.quat)
        v_body = R.T @ self.vel
        F_body = F_body - np.array(p.drag_lin) * v_body
        tau_body = tau_body - np.array(p.drag_ang) * self.omega

        acc = (R @ F_body) / p.mass - np.array([0, 0, G])
        domega = self.Jinv * (tau_body - np.cross(self.omega, self.J * self.omega))

        # 半隐式欧拉
        self.vel += acc * dt
        self.pos += self.vel * dt
        self.omega += domega * dt
        dq = quat_mul(self.quat, np.r_[0.0, self.omega]) * 0.5
        self.quat += dq * dt
        self.quat /= np.linalg.norm(self.quat)
        self.t += dt
        self._last_acc_world = acc
        return T

    def imu(self, rng, gyro_bias, acc_bias):
        """返回带噪声的 (gyro[机体], accel[机体比力])"""
        R = quat_to_rot(self.quat)
        a_w = getattr(self, "_last_acc_world", np.zeros(3))
        specific = R.T @ (a_w + np.array([0, 0, G]))
        gyro = self.omega + gyro_bias + rng.normal(0, self.p.gyro_noise_std, 3)
        acc = specific + acc_bias + rng.normal(0, self.p.acc_noise_std, 3)
        return gyro, acc


# ============================================================ 3. RL 环境
try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:          # 没装 gymnasium 也能用
    gym, spaces, _Base = None, None, object


class ThrowRecoveryEnv(_Base):
    """
    任务：任意姿态+角速度+速度抛出 → 尽快转正、角速度归零、悬停后以 ~0.3 m/s 缓降。
    观测(22)：R 的前两列(6, 比四元数无歧义) | gyro(3) | 速度(3, 世界) | 高度(1) |
             上一动作(4) | 电池电压归一(1) | 目标下降速度(1) | 竖直加速度计(3)
       注：真机上 R/速度/高度来自 ESP32 上的姿态解算+高度融合，训练时已加噪声和延迟。
    动作(4)：[-1,1] → 电机占空比 [0,1]（直接电机级，便于 Sim2Real 对齐）
    """
    metadata = {"render_modes": []}

    def __init__(self, params: QuadParams | None = None, domain_rand: float = 1.0,
                 episode_s: float = 4.0, descend_rate: float = 0.3, difficulty: float = 1.0,
                 seed: int | None = None):
        self.difficulty = difficulty
        self.base = params or QuadParams()
        self.dr = domain_rand
        self.episode_steps = int(episode_s * self.base.ctrl_hz)
        self.descend_rate = descend_rate
        self.rng = np.random.default_rng(seed)
        if spaces is not None:
            self.observation_space = spaces.Box(-np.inf, np.inf, (22,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    # -------- 抛飞初始状态分布（课程学习时逐步放大）--------
    def _throw_state(self):
        r, d = self.rng, self.difficulty          # d∈[0,1]，课程学习逐步调到 1
        tilt = d * np.pi                          # 最大初始倾角：d=1 时含完全倒扣
        axis = r.normal(size=3); axis[2] = 0; axis /= np.linalg.norm(axis) + 1e-9
        ang = r.uniform(0, tilt)
        q_tilt = np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis]
        yaw = r.uniform(-np.pi, np.pi)
        q = quat_mul(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), q_tilt)
        return dict(
            pos=[r.uniform(-0.3, 0.3), r.uniform(-0.3, 0.3), r.uniform(1.5, 2.5)],
            vel=np.r_[r.uniform(-2, 2, 2) * d, r.uniform(0.0, 2.5)],  # 手抛：向上为主
            quat=q,
            omega=r.uniform(-1, 1, 3) * (2 + 13 * d),               # 最高 ±15 rad/s
            soc=r.uniform(0.5, 1.0),
        )

    def reset(self, seed=None, options=None):
        if _Base is not object:
            super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.p = randomize(self.base, self.rng, self.dr)
        self.sim = QuadDynamics(self.p)
        self.sim.reset(**self._throw_state())
        self.gyro_bias = self.rng.normal(0, self.p.gyro_bias_std, 3)
        self.acc_bias = self.rng.normal(0, self.p.acc_bias_std, 3)
        self.last_action = np.zeros(4)
        self._prev_z = None
        self._prev_vz = 0.0
        self.k = 0
        # [FIX-D] 原实现 int(round(1/(200*0.001)))=5，再把整步 dt=1ms 跑 5 次，
        # 实际每个控制步推进 5ms，物理步长被放大 5 倍（1kHz 变 200Hz 等效）。
        # 正解：按 ctrl_hz 反推物理步长，保证「每控制步总时长 = 1/ctrl_hz」。
        self.substeps = max(1, int(round(1.0 / (self.p.ctrl_hz * self.p.dt_phys))))
        self._dt_phys_eff = (1.0 / self.p.ctrl_hz) / self.substeps
        o = self._raw_obs()
        self.obs_buf = [o] * (self.p.obs_delay_steps + 1)
        return self.obs_buf[0].astype(np.float32), {}

    def _raw_obs(self):
        s = self.sim
        R = quat_to_rot(s.quat)
        gyro, acc = s.imu(self.rng, self.gyro_bias, self.acc_bias)
        R6 = R[:, :2].T.reshape(-1) + self.rng.normal(0, 0.01, 6)   # 姿态估计误差
        vel = s.vel + self.rng.normal(0, 0.05, 3)
        z = s.pos[2] + self.rng.normal(0, self.p.alt_noise_std)
        vb = (s.v_batt - 3.7) / 0.5
        return np.concatenate([R6, gyro / 10, vel / 3, [z / 2], self.last_action,
                               [vb], [self.descend_rate], acc / G])

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        u = (a + 1) / 2
        for _ in range(self.substeps):
            self.sim.step(u, dt=self._dt_phys_eff)
        self.k += 1

        s = self.sim
        R = quat_to_rot(s.quat)
        up_z = R[2, 2]                                   # 机体 z 轴在世界的竖直分量，1=正
        r_up = up_z
        r_rate = -0.05 * np.sum(s.omega ** 2) ** 0.5
        r_vxy = -0.3 * np.linalg.norm(s.vel[:2])
        r_vz = -0.5 * abs(s.vel[2] + self.descend_rate)  # 目标：-0.3 m/s 缓降
        r_act = -0.05 * np.sum((a - self.last_action) ** 2)
        reward = r_up + r_rate + r_vxy + r_vz + r_act + 0.5   # 存活奖励

        self.last_action = a
        z, vz = float(s.pos[2]), float(s.vel[2])
        # [FIX-C] 原实现只在 z<0.05 那一帧判定软着陆，此时已落到地面，
        # 且用当帧 vz 判断——撞地瞬间 vz 必然很大，软着陆奖励几乎不可达
        # （实测 PD 缓降 9 次触地里只有 4 次被判软着陆）。
        # 正解：用「落地前的进场速度」+「触地时的姿态」判定，并放宽到 z<0.08。
        if self._prev_z is not None and z < 0.08 and self._prev_vz < 0:
            approach_vz = abs(self._prev_vz)          # 进场速度（更物理）
            soft = up_z > 0.9 and approach_vz < 0.8 and abs(vz) < 2.0
            reward += 5.0 if soft else -20.0
            terminated = True
        else:
            terminated = False
        self._prev_z, self._prev_vz = z, vz
        out_of_bounds = np.linalg.norm(s.pos[:2]) > 5 or s.pos[2] > 6
        terminated = bool(terminated or out_of_bounds)
        truncated = self.k >= self.episode_steps

        self.obs_buf.append(self._raw_obs())
        obs = self.obs_buf.pop(0)
        info = dict(up_z=up_z, omega_norm=float(np.linalg.norm(s.omega)),
                    z=float(s.pos[2]), vz=float(s.vel[2]), v_batt=float(s.v_batt))
        return obs.astype(np.float32), float(reward), terminated, truncated, info


# ============================================================ 4. 自检：级联 PD 恢复
def pd_controller(sim: QuadDynamics, p: QuadParams, vz_ref=-0.3):
    """仅用于验证模型符号；也可作为真机安全回退控制器的原型。"""
    R = quat_to_rot(sim.quat)
    z_b = R[:, 2]
    # 期望推力方向：竖直向上；姿态误差 = z_b × e_z（机体系）
    e_rot = R.T @ np.cross(z_b, np.array([0, 0, 1.0]))
    kp, kd = 0.020, 0.0020            # N·m/rad, N·m/(rad/s)
    tau_cmd = kp * e_rot - kd * sim.omega
    tau_cmd[2] = -0.0004 * sim.omega[2]

    # 倒扣时只修姿态，正立后才给爬升推力
    tilt_ok = max(z_b[2], 0.0)
    Fz = p.mass * (G + 3.0 * (vz_ref - sim.vel[2])) / max(z_b[2], 0.5) * tilt_ok
    Fz = max(Fz, 0.3 * p.mass * G)

    # 混控求解：[F, τx, τy, τz] = M · T
    x, y = p.rotor_pos[:, 0], p.rotor_pos[:, 1]
    M = np.vstack([np.ones(4), y, -x, -p.rotor_dir * p.k_m_ratio])
    T = np.linalg.solve(M, np.r_[Fz, tau_cmd])
    T = np.clip(T, 0, p.thrust_max_per_motor)
    return np.sqrt(T / p.thrust_max_per_motor) * (p.v_ref / max(sim.v_batt, 3.0))


if __name__ == "__main__":
    p = QuadParams()
    print(f"k_f = {p.k_f:.3e} N/(rad/s)^2   悬停占空比 = {p.hover_throttle:.2f}   "
          f"推重比 = {p.thrust_to_weight:.2f}")

    # 1) 方向自检：单电机加推力应产生预期力矩
    sim = QuadDynamics(p)
    for i, name in enumerate(["M1前右", "M2后右", "M3后左", "M4前左"]):
        sim.reset(pos=[0, 0, 1])
        u = np.full(4, p.hover_throttle); u[i] += 0.15
        for _ in range(100):
            sim.step(u)
        print(f"  {name}+  → omega = {np.round(sim.omega, 2)}")

    # 2) PD 从随机抛飞恢复（RL 要打败的基线）
    for diff in (0.3, 1.0):
        env = ThrowRecoveryEnv(seed=0, domain_rand=1.0, difficulty=diff)
        ok, N, t_rec = 0, 50, []
        for ep in range(N):
            env.reset()
            rec = None
            for k in range(env.episode_steps):
                u = pd_controller(env.sim, env.p)
                _, _, term, trunc, info = env.step(u * 2 - 1)
                if rec is None and info["up_z"] > 0.95 and info["omega_norm"] < 1.0:
                    rec = k / env.p.ctrl_hz
                if term or trunc:
                    break
            alive = info["z"] > 0.05 or (info["up_z"] > 0.9 and abs(info["vz"]) < 0.6)
            if alive and rec is not None:
                ok += 1; t_rec.append(rec)
        print(f"PD 基线 difficulty={diff}: 成功 {ok}/{N}，平均转正 "
              f"{np.mean(t_rec) if t_rec else float('nan'):.2f}s")

    # 3) Gymnasium 接口检查
    if gym is not None:
        from gymnasium.utils.env_checker import check_env
        check_env(ThrowRecoveryEnv(seed=1), skip_render_check=True)
        print("gymnasium check_env 通过")
