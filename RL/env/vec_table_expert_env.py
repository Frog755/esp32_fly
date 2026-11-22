#!/usr/bin/env python3
"""vec_table_expert_env.py —— 3D 最优翻转查表控制器 + v5 规则接盘闭环环境

核心架构:
  任意倒扣态 (r22 < -0.25)
     │
     ├─► [查表命中 & 验证合格] ──► 开环执行离线最优 4 电机打靶轨迹 (0.35~0.55s)
     │                                     │ (终态: θ≤20°, |ω|≤1, vz≈0)
     │                                     ▼
     │                           交接给 v5 PD/制动段 ──► 稳定悬停
     │
     └─► [表外/极端角速度/未通过点] ──► 退回 v5 最小时间切换曲线 bang-bang 兜底
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from vec_expert5_env import VecExpert5


class FlipTable3D:
    """三轴翻转表查找器 (高效最近邻与插值)"""

    def __init__(self, table_path=None):
        if table_path is None:
            table_path = os.path.join(HERE, '..', 'models', 'flip_table_3d.npz')
        self.table_path = table_path
        self.loaded = False
        if os.path.exists(table_path):
            self._load(table_path)

    def _load(self, path):
        d = np.load(path)
        self.keys = d['keys']          # (N, 3): [phi, tilt, w_par]
        self.passed = d['passed']      # (N,) bool
        self.tfs = d['tfs']            # (N,) float
        self.u_opts = d['u_opts']      # (N, 10, 4) float

        # 仅索引全仿真已通过的合格格点
        valid_idx = np.where(self.passed)[0]
        if len(valid_idx) == 0:
            print("[FlipTable3D] 警告: 表中无全仿真通过的格点，全部回退 v5")
            return

        self.valid_keys = self.keys[valid_idx]        # (M, 3)
        self.valid_tfs = self.tfs[valid_idx]          # (M,)
        self.valid_u_opts = self.u_opts[valid_idx]    # (M, 10, 4)

        # 预先插值至 200Hz 序列，避免在线插值开销
        self.valid_scheds = []
        for i in range(len(valid_idx)):
            tf = self.valid_tfs[i]
            n_steps = int(round(tf / 0.005))
            t_knots = np.linspace(0, tf, 10)
            t_sim = np.linspace(0, tf, n_steps)
            u_seq = np.zeros((n_steps, 4), dtype=np.float32)
            for m in range(4):
                u_seq[:, m] = np.interp(t_sim, t_knots, self.valid_u_opts[i, :, m])
            self.valid_scheds.append(u_seq)

        # 归一化查找权重 (phi: 1/45, tilt: 1/20, w_par: 1/6)
        self.scale = np.array([1.0 / 45.0, 1.0 / 20.0, 1.0 / 6.0], dtype=np.float32)
        self.loaded = True
        print(f"[FlipTable3D] 加载完成: 合格格点 {len(valid_idx)}/{len(self.keys)} ({len(valid_idx)/len(self.keys)*100:.1f}%)")

    def lookup(self, q, w):
        """输入四元数 q(4,) 与机体角速度 w(3,)，返回最近合格轨迹 u_seq(n_steps, 4) 或 None"""
        if not self.loaded or len(self.valid_scheds) == 0:
            return None

        # 1. 规范化四元数
        q = np.asarray(q, float)
        if q[0] < 0:
            q = -q
        qw, qv = q[0], q[1:4]
        qn = np.linalg.norm(qv)
        if qn < 1e-6:
            return None  # 接近水平，无需翻转表

        ang_deg = np.degrees(2.0 * np.arctan2(qn, qw))
        axis = qv / qn  # 旋转轴

        # 2. 计算方位角 phi (mod 360)
        phi_deg = np.degrees(np.arctan2(axis[1], axis[0])) % 360.0

        # 3. 计算沿回正轴（-axis）的角速度分量
        w_par = float(-np.dot(w, axis))

        # 4. 环形方位角差值
        d_phi = np.abs(self.valid_keys[:, 0] - phi_deg)
        d_phi = np.minimum(d_phi, 360.0 - d_phi)
        d_tilt = np.abs(self.valid_keys[:, 1] - ang_deg)
        d_w = np.abs(self.valid_keys[:, 2] - w_par)

        dist = (d_phi * self.scale[0]) ** 2 + (d_tilt * self.scale[1]) ** 2 + (d_w * self.scale[2]) ** 2
        best_i = int(np.argmin(dist))

        # 匹配容限: 方位角允许 ±30°，倾角允许 ±25°，角速度允许 ±8 rad/s
        if d_phi[best_i] > 35.0 or d_tilt[best_i] > 30.0 or d_w[best_i] > 10.0:
            return None  # 超出表的有效覆盖球，安全回退 v5

        return self.valid_scheds[best_i]


class VecTableExpert(VecExpert5):
    """查表 + v5 混合专家控制器"""

    def __init__(self, *args, table_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.table = FlipTable3D(table_path)
        self._sched = [None] * self.num_envs
        self._step_i = np.zeros(self.num_envs, dtype=np.int32)
        self._table_triggered = np.zeros(self.num_envs, dtype=bool)

    def reset(self):
        obs = super().reset()
        self._sched = [None] * self.num_envs
        self._step_i[:] = 0
        self._table_triggered[:] = False
        return obs

    def reset_at(self, idx):
        super().reset_at(idx)
        for i in idx:
            self._sched[i] = None
            self._step_i[i] = 0
            self._table_triggered[i] = False

    def _action_to_throttle(self, a):
        # 先由基线 v5 规则算出默认油门 (保证任何时候有后备)
        u_default = super()._action_to_throttle(a)
        u_out = u_default.copy()

        s = self.dyn
        qv = s.quat[:, 1:4]
        r22 = 1.0 - 2.0 * (qv[:, 0] ** 2 + qv[:, 1] ** 2)

        for i in range(self.num_envs):
            # 1. 正在执行查表轨迹
            if self._sched[i] is not None:
                if self._step_i[i] < len(self._sched[i]):
                    u_out[i] = self._sched[i][self._step_i[i]]
                    self._step_i[i] += 1
                else:
                    # 表执行完毕，交班给 v5
                    self._sched[i] = None
                continue

            # 2. 检查是否触发倒扣翻转段且尚未执行过表
            if r22[i] < -0.25 and not self._table_triggered[i]:
                traj = self.table.lookup(s.quat[i], s.w[i])
                if traj is not None:
                    self._sched[i] = traj
                    self._step_i[i] = 0
                    self._table_triggered[i] = True
                    u_out[i] = traj[0]
                    self._step_i[i] = 1

        return u_out

