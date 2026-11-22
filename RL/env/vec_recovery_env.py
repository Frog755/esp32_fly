#!/usr/bin/env python3
"""vec_recovery_env.py —— Plan A 新配方训练环境（向量化 + SB3 VecEnv 接口）

与旧 ThrowRecoveryEnv 的配方差异（每项对应一个实测失败）：
  1. 奖励 = 单一均质二次代价（含【高度项】——旧配方 9 个版本从未有过的项），无门限无阶梯；
  2. 初始姿态两阶段课程（第二轮）：Phase A ±30° 正立区学悬停原语 → Phase B 全球面 ±π（Raptor all-around）；
  3. 终止 = 回合超时 + 坠地（z < 目标-3m，第二轮新增），大姿态/大角速度不再被杀；
  4. 观测 = Raptor 22 维约定（rel_pos + R + vel + ω + prev_action），
     为后续 GRU / 与 Raptor 对照铺路；
  5. 成功判据（连续 50 步稳定悬停）只作为【评估指标】记录，不进奖励。
"""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from quad_sim_vec import VecQuadDynamics, up_z_of
from quad_sim_fast_v2 import QuadParams
from yao_kun_params import YaoKunK10Params
from quad_sim_vec import G

# ---------------- 新配方超参数（集中放这里，便于台账审查） ----------------
# 第二轮修正（第一轮尸检结论，见 docs/PlanA完整历程与第一轮训练报告.md）：
#   W_POS 3.0→1.0：深坠时位置代价放大到 ~2700/步，淹没姿态/速度项梯度
W_POS = 1.0      # 位置(相对目标)误差² —— 高度项所在
W_ATT = 3.0      # (1-up_z)² —— 姿态误差
W_VEL = 0.5      # 速度²
W_OMEGA = 0.01   # 角速度²
W_DA = 0.02      # 动作变化²（平滑）
REWARD_SCALE = 0.1

TARGET = np.array([0.0, 0.0, 2.5])   # 目标悬停点（用户目标：高度 2m+）
EPISODE_S = 5.0                       # 回合时长（TWR=1.38 下恢复+爬回2m 约需 3~4s，留裕量）
Z_CRASH_MARGIN = 3.0                  # 坠地终止：z < TARGET_z - 3m → done（第二轮新增，
                                      #   第一轮 z 最低 -75m：深坠区零梯度样本占 ~80% 时间）
PHASE_A_TILT = np.deg2rad(30.0)       # Phase A：正立区姿态扰动上限（±30°）
PHASE_A_OMEGA = 5.0                   # Phase A：初始角速度上限（Phase B 仍 20）
Z0_LO, Z0_HI = 2.2, 3.2               # 初始高度范围
OMEGA0_MAX = 20.0                     # 初始角速度上限（全球面）
STABLE_CRIT = dict(up_z=0.95, vz=0.5, omega=1.2, z=2.0, hold=50)  # 评估用，不进奖励


def random_quats(N, rng):
    """SO(3) 均匀采样：4 维高斯归一化。"""
    q = rng.normal(size=(N, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class VecRecoveryEnv(VecEnv):
    """N 架四旋翼"倒扣恢复+稳定悬停"批量环境。

    动作: (N,4) 油门 [0,1] 连续（网络 tanh 输出经 (a+1)/2 映射）。
    观测: (N,22) [rel_pos(3), R(9), vel(3), ω(3), prev_action(4)]，Raptor 约定。
    奖励: -REWARD_SCALE·(二次代价和)，每步稠密。
    终止: 仅回合超时（done=True 时自动重置）。
    """

    OBS_DEPLOY = 16   # 部署级：gyro(3)+acc(3)+重力方向(3)+v_batt(1)+vz_est(1)+z_est(1)+prev_action(4)
    #   vz/z 来自互补滤波（加速度计积分为主 + 气压计低频校正）——真机 ESP32 几行 C 即可实现，
    #   这是行业标准做法：手写"好理解的估计器"，把学习留给"难的动作策略"。
    # 气压计模型：BMP280 噪声 σ=0.08m + 慢偏置随机游走；有效更新率低由 GRU 补间
    OBS_RAPTOR = 22   # Raptor 约定：rel_pos(3)+R(9)+vel(3)+ω(3)+prev_action(4)（含不可部署的 pos/vel）

    def __init__(self, num_envs=256, episode_s=EPISODE_S, seed=0,
                 domain_rand=0.0, params=None, obs_mode='deploy',
                 tilt_cap_deg=30.0):
        assert obs_mode in ('deploy', 'raptor')
        assert 0.0 < tilt_cap_deg <= 180.0
        self.obs_mode = obs_mode
        # 渐进课程（第三轮）：初始倾角上限 cap。30°=Phase A 正立区；180°=全球面。
        #   cap 随训练线性爬升，分布始终覆盖已掌握区域 → 旧技能有持续梯度维持（第二轮
        #   实测二值切换会灾难性遗忘：正立油门 0.833→0.596）
        self.tilt_cap = np.deg2rad(tilt_cap_deg)
        self._om_up = (0.0, PHASE_A_OMEGA)   # 正立分支初始角速度范围（默认同旧版）
        self._om_g = (0.0, OMEGA0_MAX)       # 全球面分支初始角速度范围（默认同旧版）
        obs_dim = self.OBS_DEPLOY if obs_mode == 'deploy' else self.OBS_RAPTOR
        self.num_envs = int(num_envs)
        self.episode_steps = int(episode_s * 200)   # ctrl 200Hz
        self.rng = np.random.default_rng(seed)
        self.params = params or YaoKunK10Params()   # 默认实机参数（80g / TWR@3.7V=1.38）
        self.domain_rand = domain_rand
        self.dyn = VecQuadDynamics(self.num_envs, self.params)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self._prev_a = np.zeros((self.num_envs, 4))
        self._stable_cnt = np.zeros(self.num_envs, dtype=np.int64)
        self._k = np.zeros(self.num_envs, dtype=np.int64)
        self._target = np.tile(TARGET, (self.num_envs, 1))
        self._seed = seed
        # 气压计误差模型：每回合随机偏置 + 缓慢随机游走（桨扰/天气），噪声每步刷新
        self._baro_bias = np.zeros(self.num_envs)
        self._baro_noise = 0.08
        self._baro_bias_walk = 0.01
        # 互补滤波状态（部署时这段逻辑在 ESP32 上用 C 实现）
        self._vz_est = np.zeros(self.num_envs)
        self._z_est = np.zeros(self.num_envs)
        self._filt_init = np.zeros(self.num_envs, dtype=bool)
        self._kv = 4.0        # 位置误差→速度校正增益（=ωn², ωn=2 rad/s）
        self._kz = 3.6        # 位置误差→位置校正增益（=2ζωn, ζ=0.9）

    # ---------------- VecEnv 接口 ----------------
    def set_tilt_cap(self, cap_deg):
        """课程进度推进（训练脚本按步数调用）。"""
        assert 0.0 < cap_deg <= 180.0
        self.tilt_cap = np.deg2rad(cap_deg)

    def set_omega0_range(self, lo, hi):
        """初始角速度范围（两个分支共用；ω-first 课程用：先只练高转速刹停）。"""
        assert 0.0 <= lo <= hi
        self._om_up = (lo, hi)
        self._om_g = (lo, hi)

    def _sample_init(self, n):
        """按当前倾角上限采样初始状态 (pos, vel, quat, omega)。"""
        rng = self.rng
        prog = float(np.clip((self.tilt_cap - PHASE_A_TILT) / (np.pi - PHASE_A_TILT), 0.0, 1.0))
        if prog <= 0.0:
            # 起点：±30° 正立区。单盆地探索，先学悬停原语——避开全球面下
            #   "倒扣→油门↓ / 正立→油门↑" 的梯度抵消死锁（第一轮尸检根因）
            axis = rng.normal(size=(n, 3))
            axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
            ang = rng.uniform(0.0, PHASE_A_TILT, size=(n, 1))
            quat = np.concatenate([np.cos(ang / 2), axis * np.sin(ang / 2)], axis=1)
            omega = axis * rng.uniform(self._om_up[0], self._om_up[1], size=(n, 1))  # 扰动轴=倾斜轴
            vel = np.column_stack([rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n),
                                   rng.uniform(-0.5, 0.0, n)])
        elif prog >= 1.0:
            # 终点：全球面（与第一轮完全一致的 SO(3) 均匀采样）
            quat = random_quats(n, rng)
            wdir = rng.normal(size=(n, 3))
            wdir /= np.linalg.norm(wdir, axis=1, keepdims=True) + 1e-9
            omega = wdir * rng.uniform(self._om_g[0], self._om_g[1], size=(n, 1))
            vel = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
                                   rng.uniform(-1.5, 0.0, n)])
        else:
            # 渐进带：倾角 U(0, cap)；ω 上限与速度范围随 cap 线性放大（分布无跳变）
            axis = rng.normal(size=(n, 3))
            axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
            ang = rng.uniform(0.0, self.tilt_cap, size=(n, 1))
            quat = np.concatenate([np.cos(ang / 2), axis * np.sin(ang / 2)], axis=1)
            omega_cap = PHASE_A_OMEGA + (OMEGA0_MAX - PHASE_A_OMEGA) * prog
            omega = axis * rng.uniform(0.0, omega_cap, size=(n, 1))
            v_rng = 0.5 + 0.5 * prog
            vel = np.column_stack([rng.uniform(-v_rng, v_rng, n), rng.uniform(-v_rng, v_rng, n),
                                   rng.uniform(-v_rng, 0.0, n)])
        pos = np.column_stack([rng.uniform(-0.3, 0.3, n), rng.uniform(-0.3, 0.3, n),
                               rng.uniform(Z0_LO, Z0_HI, n)])
        quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        return pos, vel, quat, omega

    def reset(self):
        rng = self.rng
        N = self.num_envs
        pos, vel, quat, omega = self._sample_init(N)
        self.dyn.set_state(pos, vel, quat, omega, soc=1.0)
        self._prev_a = np.zeros((N, 4))
        self._stable_cnt = np.zeros(N, dtype=np.int64)
        self._k = np.zeros(N, dtype=np.int64)
        self._baro_bias = rng.normal(0, 0.05, N)
        self._vz_est = np.zeros(N)
        self._z_est = pos[:, 2] - TARGET[2]   # z_est 语义=相对目标高度（与气压计模型一致；
                                              #   曾误用绝对高度，首步 obs 语义跳变 ~2.5m）
        self._filt_init[:] = False
        return self._get_obs()

    def step_async(self, actions):
        self._actions = np.clip(np.asarray(actions, float), -1.0, 1.0)

    def _action_to_throttle(self, a):
        """动作→油门映射（钩子：残差子类覆写为 PD 基座 + 残差混合）。"""
        return (a + 1.0) / 2.0

    def _action_cost(self, a):
        """动作代价钩子（父类无；残差子类返回 ℓ2 幅值惩罚）。
        背景：只罚 Δa² 时大常值残差零成本，PPO 初始噪声(|a|≈0.5)永不收缩。"""
        return np.zeros(self.num_envs)

    def step_wait(self):
        # 动作 [-1,1] → 油门 [0,1]
        u = self._action_to_throttle(self._actions)
        prev_a = self._prev_a.copy()
        for _ in range(5):                       # 5 个物理子步（1kHz 物理 / 200Hz 控制）
            self.dyn.step(u)
        self._k += 1
        # ---- 稠密二次代价（含高度项！）----
        rel_pos = self.dyn.pos - self._target
        up_z = up_z_of(self.dyn.quat)
        e_att = (1.0 - up_z) ** 2
        cost = (W_POS * (rel_pos ** 2).sum(axis=1)
                + W_ATT * e_att
                + W_VEL * (self.dyn.vel ** 2).sum(axis=1)
                + W_OMEGA * (self.dyn.w ** 2).sum(axis=1)
                + W_DA * ((self._actions - prev_a) ** 2).sum(axis=1)
                + self._action_cost(self._actions))
        reward = -REWARD_SCALE * cost
        # ---- 评估指标：稳定悬停连续计数（不进奖励）----
        vz = self.dyn.vel[:, 2]
        stable_now = ((up_z > STABLE_CRIT['up_z']) & (np.abs(vz) < STABLE_CRIT['vz'])
                      & (np.linalg.norm(self.dyn.w, axis=1) < STABLE_CRIT['omega'])
                      & (self.dyn.pos[:, 2] > STABLE_CRIT['z']))
        self._stable_cnt = np.where(stable_now, self._stable_cnt + 1, 0)
        # ---- 互补滤波：加速度计积分为主，气压计低频校正 ----
        dt_ctrl = 1.0 / 200
        up_z_now = up_z
        qw2, qx2, qy2, qz2 = (self.dyn.quat[:, 0], self.dyn.quat[:, 1],
                              self.dyn.quat[:, 2], self.dyn.quat[:, 3])
        # 世界系垂直方向比力 f_z_world = R 第3行 · f_body（f_body 用观测缓存）
        f_body = self._last_f_body
        # R 第 3 行（世界 z 分量的投影行）：r20=2(qx·qz-qw·qy), r21=2(qy·qz+qw·qx), r22=1-2(qx²+qy²)
        f_z_world = (2*(qx2*qz2 - qw2*qy2) * f_body[:, 0]
                     + 2*(qy2*qz2 + qw2*qx2) * f_body[:, 1]
                     + (1 - 2*(qx2*qx2 + qy2*qy2)) * f_body[:, 2])
        a_z_world = f_z_world - G
        baro_z = (self.dyn.pos[:, 2] - self._target[:, 2]) + self._baro_bias \
                 + self.rng.normal(0, self._baro_noise, self.num_envs)
        init_mask = ~self._filt_init
        if init_mask.any():
            self._vz_est[init_mask] = 0.0
            self._z_est[init_mask] = baro_z[init_mask]
            self._filt_init[init_mask] = True
        # Luenberger 观测器（对含噪气压计【不做差分】——差分会把 0.08m 噪声放大成
        #   16 m/s 的微分噪声，实测误差 4.6 m/s。改用位置误差反馈校正速度：
        #   数学上等价于二阶观测器，噪声以 1/ωn² 衰减而不是被微分放大）
        err_z = baro_z - self._z_est
        self._vz_est += a_z_world * dt_ctrl          # 1) 加速度计积分（预测）
        self._vz_est += self._kv * err_z * dt_ctrl   # 2) 位置误差 → 速度校正
        self._z_est += self._vz_est * dt_ctrl        # 3) 速度积分（预测）
        self._z_est += self._kz * err_z * dt_ctrl    # 4) 位置误差 → 位置校正
        # ---- 观测 ----
        obs = self._get_obs()
        # ---- 终止：超时 + 坠地（第二轮新增），自动重置 ----
        trunc = self._k >= self.episode_steps
        crash = self.dyn.pos[:, 2] < (TARGET[2] - Z_CRASH_MARGIN)
        dones = trunc | crash
        infos = [dict(stable_cnt=int(self._stable_cnt[i]), k=int(self._k[i]),
                      up_z=float(up_z[i]), z=float(self.dyn.pos[i, 2]),
                      crash=bool(crash[i])) for i in range(self.num_envs)]
        if dones.any():
            self._reset_done(dones)
        self._prev_a = self._actions.copy()
        return obs, reward.astype(np.float64), dones, infos

    def _get_obs(self):
        """两种观测模式：
        deploy(14维)——耀坤基座版实机可得量：
            gyro(3) + 加速度计比力(3) + 重力方向·机体系(3) + v_batt(1) + prev_action(4)
            （MPU6500 + 电池 ADC 分压，全部真实存在；无 pos/vel/高度）
        raptor(22维)——Raptor 约定（含仿真特权量 rel_pos/vel），仅用于与 Raptor 对照。
        """
        s = self.dyn
        qw, qx, qy, qz = s.quat[:, 0], s.quat[:, 1], s.quat[:, 2], s.quat[:, 3]
        R = np.stack([
            np.stack([1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)], axis=1),
            np.stack([2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)], axis=1),
            np.stack([2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)], axis=1),
        ], axis=2)                                # (N,3,3)
        if self.obs_mode == 'deploy':
            # 重力方向·机体系 = R^T @ [0,0,-1]（比力计静止时读到的方向）
            grav_body = np.stack([-R[:, 0, 2], -R[:, 1, 2], -R[:, 2, 2]], axis=1)
            # 加速度计比力 = R^T @ (a_world - g) = R^T @ a_world + [0,0,+g]
            acc_body = np.stack([
                (R[:, 0, 0]*s.last_acc[:, 0] + R[:, 1, 0]*s.last_acc[:, 1] + R[:, 2, 0]*s.last_acc[:, 2]),
                (R[:, 0, 1]*s.last_acc[:, 0] + R[:, 1, 1]*s.last_acc[:, 1] + R[:, 2, 1]*s.last_acc[:, 2]),
                (R[:, 0, 2]*s.last_acc[:, 0] + R[:, 1, 2]*s.last_acc[:, 1] + R[:, 2, 2]*s.last_acc[:, 2]) + G], axis=1)
            vb = (s.v_batt / 4.2)[:, None]        # 归一化到满电
            self._last_f_body = acc_body          # 供互补滤波计算世界系垂直比力
            return np.concatenate([s.w, acc_body, grav_body, vb,
                                   self._vz_est[:, None], self._z_est[:, None],
                                   self._prev_a], axis=1).astype(np.float32)
        rel_pos = s.pos - self._target
        return np.concatenate([rel_pos, R.reshape(self.num_envs, 9), s.vel, s.w,
                               self._prev_a], axis=1).astype(np.float32)

    def _reset_done(self, dones):
        """被重置的通道按当前课程阶段重新初始化（等效 Raptor 的 auto-reset）。"""
        idx = np.where(dones)[0]
        pos, vel, quat, omega = self._sample_init(len(idx))
        self.dyn.pos[idx] = pos
        self.dyn.vel[idx] = vel
        self.dyn.quat[idx] = quat
        self.dyn.w[idx] = omega
        self.dyn.rotor_w[idx] = 0.0
        self.dyn.soc[idx] = 1.0
        self.dyn.v_batt[idx] = self.dyn.ocv_lo[idx] + self.dyn.ocv_hi[idx]
        self._stable_cnt[idx] = 0
        self._k[idx] = 0
        self._baro_bias[idx] = self.rng.normal(0, 0.05, len(idx))
        self._vz_est[idx] = 0.0
        self._z_est[idx] = pos[:, 2] - TARGET[2]   # 修复：原误写 pos[idx, 2]（部分 done 会 IndexError）
                                                   #   并统一为相对目标高度语义
        self._filt_init[idx] = False

    def close_extras(self, **kwargs):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._seed = seed

    def env_is_wrapped(self, wrapper_class):
        return [False for _ in range(self.num_envs)]

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs)
                for _ in range(self.num_envs)]
