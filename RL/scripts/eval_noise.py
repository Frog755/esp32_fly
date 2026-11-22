#!/usr/bin/env python3
"""eval_noise.py —— v8 vs v9 传感器噪声复测（Stage 18）

工况：无噪声 / L1~L5 全通道，以及 L4 下单通道隔离（只陀螺 / 只姿态 / 只位置速度）。
每个工况 × 控制器 × 种子 跑一次 512 通道（首回合统计，同 eval_recovery_metrics）。
噪声随机数独立于初态随机数（noise_seed 固定派生），v8 与 v9 在同一工况下拿到同一组初态与同分布噪声。
"""
import os, sys, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
import numpy as np
from multiprocessing import Pool
from eval_recovery_metrics import run_metrics, summarize
from noise_wrap import LEVELS, make_noisy

L4 = LEVELS['L4']
CASES = [
    ('无噪声', {}),
    ('L1 全通道', LEVELS['L1']),
    ('L2 全通道', LEVELS['L2']),
    ('L3 全通道', LEVELS['L3']),
    ('L4 全通道', LEVELS['L4']),
    ('L5 全通道', LEVELS['L5']),
    ('L4 只陀螺', dict(gyro=L4['gyro'], gyro_bias=L4['gyro_bias'])),
    ('L4 只姿态', dict(att=L4['att'], att_bias=L4['att_bias'])),
    ('L4 只位置速度', dict(pos=L4['pos'], posz=L4['posz'], vel=L4['vel'])),
]


def job(arg):
    ctrl, case, noise, seed, extra = arg
    if ctrl == 'v8':
        from vec_expert8_env import VecExpert8 as B; kw = {}
    else:
        from vec_expert9_env import VecExpert9 as B; kw = dict(ctrl=extra) if extra else {}
    C = make_noisy(B)
    kw.update(noise=noise, noise_seed=seed * 7919 + 17)
    r = run_metrics(C, 512, seed, env_kwargs=kw)
    row = summarize(r, f"{ctrl} {case}")
    row.update(ctrl=ctrl, case=case, seed=seed)
    return row


K = ['rate', 'deep', 'fast', 'crashes', 'drop50', 'drop90', 'drop_deep50', 'txy50', 'txy90', 'tst50', 'tst90']


def report(rows, ctrls, cases):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['case'], r['ctrl'])].append(r)
    print(f"{'工况':<12}{'控制器':<6}{'成功率':>7}{'深倒扣':>7}{'坠毁':>6}{'掉高中位':>8}{'P90':>6}{'深倒扣掉高':>9}{'倾斜翻正中位':>10}{'P90':>7}{'稳悬中位':>8}{'P90':>6}")
    out = {}
    for case, _ in cases:
        for c in ctrls:
            rs = g[(case, c)]
            if not rs:
                continue
            m = {k: float(np.nanmean([r[k] for r in rs])) for k in K}
            out[(case, c)] = m
            print(f"{case:<12}{c:<6}{m['rate']:6.1f}%{m['deep']:6.1f}%{m['crashes']:6.1f}{m['drop50']:8.2f}{m['drop90']:6.2f}{m['drop_deep50']:9.2f}"
                  f"{m['txy50']:10.3f}{m['txy90']:7.3f}{m['tst50']:8.2f}{m['tst90']:6.2f}")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[2024, 42, 100, 2026, 7])
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'stage18', 'noise.json'))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    jobs = [(c, case, nz, sd, None) for c in ['v8', 'v9'] for case, nz in CASES for sd in a.seeds]
    with Pool(16) as pool:
        rows = pool.map(job, jobs)
    report(rows, ['v8', 'v9'], CASES)
    json.dump(rows, open(a.out, 'w'), ensure_ascii=False, indent=1, default=float)
