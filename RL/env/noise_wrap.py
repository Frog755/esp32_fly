#!/usr/bin/env python3
"""noise_wrap.py —— 传感器噪声包装器（Stage 18）

给任意控制器环境类套一层：控制器每步读到的是【带噪测量值】，仿真推进与评测仍用真值。
做法：在 _action_to_throttle 调用期间，临时把 dyn.quat / dyn.w / dyn.pos / dyn.vel 换成测量值，算完立刻换回。
这样 v8 / v9 代码一行不改，噪声模型完全一致。

噪声模型（每个通道独立；"偏置"每回合重采样并保持，"白噪声"每控制步 200Hz 重采样）：
  - 陀螺：白噪声 σ_gyro (rad/s，代表滤波后的振动噪声) + 每回合零偏 σ_gyro_bias (rad/s)
  - 姿态估计：世界系水平轴小角度旋转误差 = 每回合固定偏置 σ_att_bias (deg) + 白噪声 σ_att (deg)
           （模拟互补滤波/Mahony 在剧烈翻转中的倾角漂移与抖动）
  - 位置：水平白噪声 σ_pos (m)，z 通道 σ_posz (m，气压计级)
  - 速度：白噪声 σ_vel (m/s)
垂直通道的 _z_est / _vz_est 本来就来自基类的气压计噪声 + 加速度计互补滤波，不另加。
"""
import numpy as np

LEVELS = {
    'none': dict(),
    'L1':   dict(gyro=0.05, gyro_bias=0.02, att=0.5, att_bias=1.0, pos=0.02, posz=0.05, vel=0.05),
    'L2':   dict(gyro=0.15, gyro_bias=0.05, att=1.0, att_bias=3.0, pos=0.05, posz=0.08, vel=0.15),
    'L3':   dict(gyro=0.30, gyro_bias=0.10, att=2.0, att_bias=5.0, pos=0.10, posz=0.15, vel=0.30),
    'L4':   dict(gyro=0.60, gyro_bias=0.20, att=4.0, att_bias=10.0, pos=0.20, posz=0.30, vel=0.60),
    'L5':   dict(gyro=1.20, gyro_bias=0.40, att=8.0, att_bias=20.0, pos=0.40, posz=0.60, vel=1.20),
}


def _quat_from_rotvec(rv):
    ang = np.linalg.norm(rv, axis=1)
    axis = rv / np.maximum(ang, 1e-12)[:, None]
    h = 0.5 * ang
    return np.concatenate([np.cos(h)[:, None], axis * np.sin(h)[:, None]], axis=1)


def _qmul(p, q):
    w1, x1, y1, z1 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)


def _tilt_rotvec(n, sigma_rad, rng):
    """世界系水平轴上的随机小角度旋转（只扰动倾角估计；偏航估计误差对回正无关紧要）。"""
    v = rng.normal(0.0, sigma_rad, size=(n, 3))
    v[:, 2] = 0.0
    return v


def make_noisy(base_cls):
    class Noisy(base_cls):
        def __init__(self, *args, noise=None, noise_seed=12345, **kwargs):
            self._nz = dict(gyro=0.0, gyro_bias=0.0, att=0.0, att_bias=0.0, pos=0.0, posz=0.0, vel=0.0)
            self._nz.update(noise or {})
            self._nrng = np.random.default_rng(noise_seed)
            super().__init__(*args, **kwargs)
            self._gb = np.zeros((self.num_envs, 3))
            self._ab = np.zeros((self.num_envs, 3))
            self._resample_bias(np.ones(self.num_envs, bool))

        def _resample_bias(self, mask):
            if not hasattr(self, '_gb'):
                return
            n = int(mask.sum())
            if n == 0:
                return
            r = self._nrng
            self._gb[mask] = r.normal(0.0, self._nz['gyro_bias'], size=(n, 3))
            self._ab[mask] = _tilt_rotvec(n, np.deg2rad(self._nz['att_bias']), r)

        def reset(self):
            out = super().reset()
            self._resample_bias(np.ones(self.num_envs, bool))
            return out

        def _reset_done(self, dones):
            super()._reset_done(dones)
            self._resample_bias(np.asarray(dones, bool))

        def _action_to_throttle(self, a):
            s = self.dyn; z = self._nz; r = self._nrng; N = self.num_envs
            true = (s.quat, s.w, s.pos, s.vel)
            if z['att'] or z['att_bias']:
                rv = self._ab + _tilt_rotvec(N, np.deg2rad(z['att']), r)
                qm = _qmul(_quat_from_rotvec(rv), s.quat)
                qm = qm / np.linalg.norm(qm, axis=1, keepdims=True)
            else:
                qm = s.quat
            wm = s.w + self._gb + r.normal(0.0, z['gyro'], size=(N, 3))
            pm = s.pos + np.column_stack([r.normal(0.0, z['pos'], N), r.normal(0.0, z['pos'], N),
                                          r.normal(0.0, z['posz'], N)])
            vm = s.vel + r.normal(0.0, z['vel'], size=(N, 3))
            try:
                s.quat, s.w, s.pos, s.vel = qm, wm, pm, vm
                return super()._action_to_throttle(a)
            finally:
                s.quat, s.w, s.pos, s.vel = true

    Noisy.__name__ = 'Noisy' + base_cls.__name__
    return Noisy
