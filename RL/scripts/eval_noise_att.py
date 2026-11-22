#!/usr/bin/env python3
"""eval_noise_att.py —— 姿态估计误差拆分扫描：每回合固定偏置 vs 每步白噪声，找失效点。"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from eval_noise import job, report

CASES = [('无噪声', {})]
for b in [5, 10, 15, 20]:
    CASES.append((f'姿态偏置{b}°', dict(att_bias=float(b))))
for w in [4, 8, 12]:
    CASES.append((f'姿态白噪{w}°', dict(att=float(w))))

if __name__ == '__main__':
    seeds = [2024, 42, 100, 2026, 7]
    jobs = [(c, case, nz, sd, None) for c in ['v8', 'v9'] for case, nz in CASES for sd in seeds]
    with Pool(16) as pool:
        rows = pool.map(job, jobs)
    report(rows, ['v8', 'v9'], CASES)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'stage18', 'noise_att.json')
    json.dump(rows, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
