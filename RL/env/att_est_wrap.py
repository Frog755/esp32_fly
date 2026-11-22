#!/usr/bin/env python3
"""att_est_wrap.py —— 真实姿态估计器包装（Stage 18b）

与 noise_wrap.py 的区别：noise_wrap 直接给姿态加"人为设定"的误差；这里在仿真里真的跑一个
IMU → 姿态融合 的估计器，控制器读到的是估计器输出，误差由物理过程自然产生。

IMU 模型（1kHz，挂在每个物理子步之后）：
  - 陀螺：真值 + 每回合零偏 (σ gyro_bias) + 白噪声 (σ gyro_sigma，代表振动)；±2000°/s (34.9 rad/s) 限幅（MPU6500 最大量程）
  - 加速度计：机体系比力真值 R^T(a_world + g·ez) + 白噪声 (σ acc_sigma，代表振动)
    注意：多旋翼的比力≈推力/质量，沿机体 z 轴——只有在悬停平衡时才指向"上"，翻转/失重时不含重力方向信息。

估计器（Mahony 互补滤波，q̇ = ½ q ⊗ (ω_meas + Kp·gate·e)，e = â × v̂，v̂ = 估计的机体系"上"方向）：
  - mode='truth' : 不用估计器，控制器读真值（基线）
  - mode='gyro'  : 纯陀螺积分（Kp=0）
  - mode='plain' : 普通 Mahony，加速度计全程按固定 Kp 修正
  - mode='gated' : 门控 Mahony —— 修正增益 × gate，gate = 比力模长接近 1g 的程度 × 角速度足够小的程度；
                   翻转中 gate≈0（只靠陀螺），回到近悬停后 gate→1 拉回漂移
初始估计 = 真值 ⊗ 随机倾角误差 (σ init_err_deg)，代表被扰动前估计器已在正常工作。
控制器读到的角速度 = 限幅后的陀螺测量值。位置/速度仍用真值（上一轮已证明其噪声影响可忽略）。
已知简化：基类垂直通道的气压计/加速度计互补滤波仍用真实姿态投影。
"""
import numpy as np

G = 9.81
GYRO_FS = np.deg2rad(2000.0)

DEFAULT_EST = dict(mode='truth', kp=2.0, gyro_sigma=0.05, gyro_bias=0.01, acc_sigma=1.0,
                   init_err_deg=2.0, f_band=0.15, w_lo=1.0, w_hi=3.0, sat=True)


def qmul(p, q):
    w1, x1, y1, z1 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)


def qexp(rv):
    ang = np.linalg.norm(rv, axis=1)
    ax = rv / np.maximum(ang, 1e-12)[:, None]
    h = 0.5 * ang
    return np.concatenate([np.cos(h)[:, None], ax * np.sin(h)[:, None]], axis=1)


def body_z_world(q):
    """机体 z 轴在世界系的方向 = R 第 3 列 (r02, r12, r22)。"""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)], axis=1)


def up_in_body(q):
    """世界"上"方向在机体系的表示 = R 第 3 行 (r20, r21, r22)。"""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)], axis=1)


def make_estimated(base_cls):
    class Est(base_cls):
        def __init__(self, *args, est=None, est_seed=1, est_stats=None, **kwargs):
            self._ec = dict(DEFAULT_EST)
            self._ec.update(est or {})
            self._erng = np.random.default_rng(est_seed)
            super().__init__(*args, **kwargs)
            N = self.num_envs
            self._qe = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))
            self._gb = np.zeros((N, 3))
            self._wm = np.zeros((N, 3))
            self._first = np.ones(N, bool)
            self._err_max = np.zeros(N); self._err_sum = np.zeros(N); self._err_n = np.zeros(N)
            self._wmax = np.zeros(N); self._sat_hit = np.zeros(N, bool)
            orig = self.dyn.step

            def step_hook(u, _orig=orig):
                _orig(u)
                self._est_update()
            self.dyn.step = step_hook
            if est_stats is not None:
                est_stats['env'] = self

        # ---------------- 估计器 ----------------
        def _gyro_meas(self, idx=None):
            s = self.dyn; c = self._ec
            w = s.w if idx is None else s.w[idx]
            gb = self._gb if idx is None else self._gb[idx]
            wm = w + gb + self._erng.normal(0.0, c['gyro_sigma'], size=w.shape)
            return np.clip(wm, -GYRO_FS, GYRO_FS) if c['sat'] else wm

        def _init_est(self, mask):
            idx = np.where(mask)[0]
            if len(idx) == 0:
                return
            c = self._ec; r = self._erng
            rv = r.normal(0.0, np.deg2rad(c['init_err_deg']), size=(len(idx), 3)); rv[:, 2] = 0.0
            q = qmul(qexp(rv), self.dyn.quat[idx])
            self._qe[idx] = q / np.linalg.norm(q, axis=1, keepdims=True)
            self._gb[idx] = r.normal(0.0, c['gyro_bias'], size=(len(idx), 3))
            self._wm[idx] = self._gyro_meas(idx)

        def _est_update(self):
            s = self.dyn; c = self._ec; N = self.num_envs; dt = s.dt
            awt = np.abs(s.w).max(axis=1)
            self._wmax = np.where(self._first, np.maximum(self._wmax, awt), self._wmax)
            self._sat_hit |= self._first & (awt > GYRO_FS)
            wm = self._gyro_meas()
            self._wm = wm
            if c['mode'] == 'truth':
                return
            # 加速度计：机体系比力 = R^T (a_world + g ez)
            q = s.quat
            qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            fw = s.last_acc + np.array([0.0, 0.0, G])
            R0 = np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)], axis=1)
            R1 = np.stack([2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)], axis=1)
            R2 = np.stack([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)], axis=1)
            fb = (R0 * fw[:, 0:1] + R1 * fw[:, 1:2] + R2 * fw[:, 2:3])
            fb = fb + self._erng.normal(0.0, c['acc_sigma'], size=(N, 3))
            fn = np.linalg.norm(fb, axis=1)
            a = fb / np.maximum(fn, 1e-6)[:, None]
            v = up_in_body(self._qe)
            e = np.cross(a, v)
            if c['mode'] == 'gyro':
                k = np.zeros(N)
            elif c['mode'] == 'plain':
                k = np.full(N, c['kp']) * (fn > 1e-3)
            else:  # gated
                gf = np.clip(1.0 - np.abs(fn / G - 1.0) / c['f_band'], 0.0, 1.0)
                gw = np.clip((c['w_hi'] - np.linalg.norm(wm, axis=1)) / (c['w_hi'] - c['w_lo']), 0.0, 1.0)
                k = c['kp'] * gf * gw
            wc = wm + k[:, None] * e
            qn = qmul(self._qe, qexp(wc * dt))
            self._qe = qn / np.linalg.norm(qn, axis=1, keepdims=True)
            # 倾角估计误差（机体 z 轴夹角）
            err = np.arccos(np.clip((body_z_world(self._qe) * body_z_world(q)).sum(1), -1.0, 1.0))
            f = self._first
            self._err_max = np.where(f, np.maximum(self._err_max, err), self._err_max)
            self._err_sum += np.where(f, err, 0.0); self._err_n += f

        # ---------------- 环境钩子 ----------------
        def reset(self):
            out = super().reset()
            if hasattr(self, '_qe'):
                self._init_est(np.ones(self.num_envs, bool))
            return out

        def _reset_done(self, dones):
            super()._reset_done(dones)
            d = np.asarray(dones, bool)
            self._first &= ~d
            self._init_est(d)

        def _action_to_throttle(self, a):
            if self._ec['mode'] == 'truth':
                return super()._action_to_throttle(a)
            s = self.dyn
            true = (s.quat, s.w)
            try:
                s.quat, s.w = self._qe.copy(), self._wm.copy()
                return super()._action_to_throttle(a)
            finally:
                s.quat, s.w = true

        def est_summary(self):
            m = self._err_n > 0
            base = dict(sat_pct=float(self._sat_hit.mean() * 100), wmax99=float(np.percentile(self._wmax, 99)),
                        wmax_max=float(self._wmax.max()))
            if not m.any():
                base.update(err_max50=0.0, err_max90=0.0, err_mean=0.0)
                return base
            emax = np.degrees(self._err_max[m]); emean = np.degrees(self._err_sum[m] / self._err_n[m])
            base.update(err_max50=float(np.median(emax)), err_max90=float(np.percentile(emax, 90)),
                        err_mean=float(np.mean(emean)))
            return base

    Est.__name__ = 'Est' + base_cls.__name__
    return Est
