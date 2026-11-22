#!/usr/bin/env python3
"""eval_recovery_metrics.py —— 回正"掉高量 + 耗时"专项评测（Stage 17）

在官方 512 通道判据（eval_all_controllers.py：成功 = 从未坠地 z>-0.5 且连续稳定 ≥50 步）之外，
额外统计：
  - drop      : 掉高量 = 初始高度 z0 - 坠地/重置前全程最低点 zmin（m）
  - t_up      : 姿态翻正耗时 = 首次 up_z > 0.9 且 |ω| < 3 rad/s（s，含偏航自旋）
  - t_upxy    : 倾斜翻正耗时 = 首次 up_z > 0.9 且 |ω_xy| < 3 rad/s（s，不计偏航自旋）
  - t_stable  : 进入稳定悬停耗时 = 第一段 ≥50 步稳定区间的起点（s），即判据开始计数时刻
  - t_climb   : 下坠被止住的时刻 = 首次达到 zmin 的时刻（s）
所有逐通道统计只取"首个回合"（坠毁通道在坠毁前截断，避免 auto-reset 污染——项目铁律）。
"""
import os, sys, time, argparse, importlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
import numpy as np
from yao_kun_params import YaoKunK10Params
from quad_sim_vec import up_z_of

DT = 0.005
HOLD = 50


def run_metrics(env_cls, num_envs=512, seed=2024, steps=1000, env_kwargs=None, perturb=None):
    p = YaoKunK10Params()
    env = env_cls(num_envs=num_envs, seed=seed, episode_s=steps * DT, params=p,
                  tilt_cap_deg=180.0, **(env_kwargs or {}))
    if perturb:
        perturb(env)   # 只改仿真真值（env.dyn），控制器仍用名义参数 → 模型失配测试
    env.reset()
    N = num_envs
    q0 = env.dyn.quat.copy(); w0 = env.dyn.w.copy()
    z0 = env.dyn.pos[:, 2].copy()
    tilt0 = np.degrees(np.arccos(np.clip(1 - 2 * (q0[:, 1] ** 2 + q0[:, 2] ** 2), -1, 1)))
    wn0 = np.linalg.norm(w0, axis=1)

    alive = np.ones(N, bool)            # 首回合仍在进行
    crashed = np.zeros(N, bool)
    zmin = z0.copy(); t_zmin = np.zeros(N)
    t_up = np.full(N, np.nan)
    t_upxy = np.full(N, np.nan)
    t_stable = np.full(N, np.nan)
    stable_run = np.zeros(N, np.int64)
    t0 = time.time()
    for k in range(steps):
        env.step(np.zeros((N, 4)))
        t = (k + 1) * DT
        s = env.dyn
        # 坠毁判定用 env 内部 crash（z < -0.5），先读 info 前的状态：env.step 已 auto-reset，
        # 所以用 _k==0 识别被重置的通道
        reset_now = (env._k == 0) & alive
        if k < steps - 1:
            crashed |= reset_now
            alive &= ~reset_now
        z = s.pos[:, 2]
        upd = alive & (z < zmin)
        zmin = np.where(upd, z, zmin); t_zmin = np.where(upd, t, t_zmin)
        upz = up_z_of(s.quat); wn = np.linalg.norm(s.w, axis=1)
        hit_up = alive & np.isnan(t_up) & (upz > 0.9) & (wn < 3.0)
        t_up[hit_up] = t
        hit_xy = alive & np.isnan(t_upxy) & (upz > 0.9) & (np.hypot(s.w[:, 0], s.w[:, 1]) < 3.0)
        t_upxy[hit_xy] = t
        stable_now = alive & (upz > 0.95) & (np.abs(s.vel[:, 2]) < 0.5) & (wn < 1.2) & (z > 2.0)
        stable_run = np.where(stable_now, stable_run + 1, 0)
        hit_st = np.isnan(t_stable) & (stable_run >= HOLD)
        t_stable[hit_st] = t - (HOLD - 1) * DT
    # 坠毁通道 zmin 取坠地线（被 reset 前最后一帧未记录时）
    zmin = np.where(crashed, np.minimum(zmin, -0.5), zmin)
    succ = (~crashed) & ~np.isnan(t_stable)
    drop = z0 - zmin
    return dict(N=N, elapsed=time.time() - t0, succ=succ, crashed=crashed, drop=drop,
                zmin=zmin, t_zmin=t_zmin, t_up=t_up, t_upxy=t_upxy, t_stable=t_stable,
                tilt0=tilt0, wn0=wn0, z0=z0)


def _q(a, qs=(50, 90, 99)):
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return [float('nan')] * (len(qs) + 1)
    return [float(np.percentile(a, q)) for q in qs] + [float(a.max())]


def summarize(r, name):
    succ, cr = r['succ'], r['crashed']
    deep = r['tilt0'] >= 135; fast = r['wn0'] >= 14
    d50, d90, d99, dmx = _q(r['drop'])
    dd50, dd90, _, _ = _q(r['drop'][deep])
    tu50, tu90, tu99, _ = _q(r['t_up'])
    ts50, ts90, ts99, _ = _q(r['t_stable'])
    tx50, tx90, tx99, _ = _q(r['t_upxy'])
    row = dict(name=name, rate=succ.mean() * 100, deep=succ[deep].mean() * 100,
               fast=succ[fast].mean() * 100, crashes=int(cr.sum()),
               drop50=d50, drop90=d90, drop99=d99, dropmax=dmx,
               drop_deep50=dd50, drop_deep90=dd90,
               tup50=tu50, tup90=tu90, tup99=tu99, txy50=tx50, txy90=tx90, txy99=tx99, tst50=ts50, tst90=ts90, tst99=ts99,
               drop_mean=float(r['drop'].mean()))
    return row


def print_row(row):
    print(f"【{row['name']}】 成功 {row['rate']:5.1f}%  深倒扣 {row['deep']:5.1f}%  高转速 {row['fast']:5.1f}%  坠毁 {row['crashes']}")
    print(f"   掉高 m   : 中位 {row['drop50']:.2f} | P90 {row['drop90']:.2f} | P99 {row['drop99']:.2f} | 最大 {row['dropmax']:.2f} | 均值 {row['drop_mean']:.2f}"
          f" | 深倒扣中位 {row['drop_deep50']:.2f} P90 {row['drop_deep90']:.2f}")
    print(f"   翻正 s   : 中位 {row['tup50']:.3f} | P90 {row['tup90']:.3f} | P99 {row['tup99']:.3f}   （含偏航 |ω|<3）")
    print(f"   倾斜翻正 s: 中位 {row['txy50']:.3f} | P90 {row['txy90']:.3f} | P99 {row['txy99']:.3f}   （|ω_xy|<3）")
    print(f"   稳悬停 s : 中位 {row['tst50']:.3f} | P90 {row['tst90']:.3f} | P99 {row['tst99']:.3f}")


def load_cls(spec):
    mod, cls = spec.split(':')
    return getattr(importlib.import_module(mod), cls)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ctrl', nargs='+', default=['vec_expert8_env:VecExpert8'])
    ap.add_argument('--envs', type=int, default=512)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2024])
    a = ap.parse_args()
    for spec in a.ctrl:
        cls = load_cls(spec)
        for sd in a.seeds:
            r = run_metrics(cls, a.envs, sd)
            row = summarize(r, f"{spec.split(':')[1]} seed={sd}")
            print_row(row)
            print(f"   (用时 {r['elapsed']:.1f}s)")
