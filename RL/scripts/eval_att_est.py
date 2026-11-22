#!/usr/bin/env python3
"""eval_att_est.py —— 真实姿态估计器下的 v8 / v9 复测（Stage 18b）

对比：理想姿态 / 纯陀螺积分 / 普通 Mahony / 门控 Mahony；常规噪声与悲观噪声；5s 与 10s 回合。
常规噪声：陀螺白噪 0.05 rad/s @1kHz、零偏 0.01 rad/s (≈0.6°/s，校准后)、加速度计振动 1.0 m/s²、初始估计误差 2°
悲观噪声：陀螺白噪 0.2、零偏 0.05 (≈2.9°/s)、加速度计 3.0 m/s²、初始估计误差 5°
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
import numpy as np
from multiprocessing import Pool
from eval_recovery_metrics import run_metrics, summarize
from att_est_wrap import make_estimated

PESS = dict(gyro_sigma=0.2, gyro_bias=0.05, acc_sigma=3.0, init_err_deg=5.0)
CASES = [
    ('理想姿态', dict(mode='truth'), 1000),
    ('纯陀螺积分', dict(mode='gyro'), 1000),
    ('普通Mahony Kp0.5', dict(mode='plain', kp=0.5), 1000),
    ('普通Mahony Kp2', dict(mode='plain', kp=2.0), 1000),
    ('门控Mahony Kp2', dict(mode='gated', kp=2.0), 1000),
    ('门控Mahony Kp5', dict(mode='gated', kp=5.0), 1000),
    ('悲观·纯陀螺', dict(mode='gyro', **PESS), 1000),
    ('悲观·普通Kp2', dict(mode='plain', kp=2.0, **PESS), 1000),
    ('悲观·门控Kp2', dict(mode='gated', kp=2.0, **PESS), 1000),
    ('悲观·纯陀螺 10s', dict(mode='gyro', **PESS), 2000),
    ('悲观·门控Kp2 10s', dict(mode='gated', kp=2.0, **PESS), 2000),
]
SEEDS = [2024, 42, 100, 2026, 7]


def job(arg):
    ctrl, case, est, steps, seed = arg
    if ctrl == 'v8':
        from vec_expert8_env import VecExpert8 as B
    else:
        from vec_expert9_env import VecExpert9 as B
    st = {}
    r = run_metrics(make_estimated(B), 512, seed, steps=steps,
                    env_kwargs=dict(est=est, est_seed=seed * 104729 + 3, est_stats=st))
    row = summarize(r, f"{ctrl} {case}")
    row.update(st['env'].est_summary())
    row.update(ctrl=ctrl, case=case, seed=seed, nostable=int((~r['succ'] & ~r['crashed']).sum()))
    return row


K = ['rate', 'crashes', 'nostable', 'drop50', 'drop90', 'txy50', 'tst50', 'err_max50', 'err_max90', 'err_mean', 'sat_pct', 'wmax99', 'wmax_max']

if __name__ == '__main__':
    jobs = [(c, case, est, steps, sd) for c in ['v8', 'v9'] for case, est, steps in CASES for sd in SEEDS]
    with Pool(16) as pool:
        rows = pool.map(job, jobs)
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['case'], r['ctrl'])].append(r)
    print(f"{'工况':<18}{'控制器':<5}{'成功率':>7}{'坠毁':>6}{'未稳悬':>6}{'掉高中位':>8}{'P90':>6}{'倾斜翻正':>8}{'稳悬中位':>8} | {'估计误差峰值中位':>10}{'P90':>7}{'均值':>6} | {'饱和%':>6}{'|ω|P99':>7}{'最大':>6}")
    for case, _, _ in CASES:
        for c in ['v8', 'v9']:
            m = {k: float(np.mean([r[k] for r in g[(case, c)]])) for k in K}
            print(f"{case:<18}{c:<5}{m['rate']:6.1f}%{m['crashes']:6.1f}{m['nostable']:6.1f}{m['drop50']:8.2f}{m['drop90']:6.2f}{m['txy50']:8.3f}{m['tst50']:8.2f} | "
                  f"{m['err_max50']:9.1f}°{m['err_max90']:6.1f}°{m['err_mean']:5.1f}° | {m['sat_pct']:6.2f}{m['wmax99']:7.1f}{m['wmax_max']:6.1f}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'stage18', 'att_est.json')
    json.dump(rows, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
