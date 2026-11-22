#!/usr/bin/env python3
"""final_compare_v9.py —— v8 vs v9 最终对比（调参种子 + 留出种子）"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
import numpy as np
from multiprocessing import Pool
from eval_recovery_metrics import run_metrics, summarize

TUNE = [2024, 42, 100]
HOLD = [2026, 7, 123, 999, 31415]
CTRL = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'stage17', 'v9_final_ctrl.json')))


def job(arg):
    name, seed = arg
    if name == 'v8':
        from vec_expert8_env import VecExpert8 as C; kw = None
    else:
        from vec_expert9_env import VecExpert9 as C; kw = dict(ctrl=CTRL)
    r = run_metrics(C, 512, seed, env_kwargs=kw)
    row = summarize(r, name); row['seed'] = seed
    return row


K = ['rate', 'deep', 'fast', 'crashes', 'drop50', 'drop90', 'drop99', 'drop_mean', 'drop_deep50', 'drop_deep90',
     'tup50', 'tup90', 'txy50', 'txy90', 'tst50', 'tst90']

if __name__ == '__main__':
    jobs = [(n, s) for n in ['v8', 'v9'] for s in TUNE + HOLD]
    with Pool(16) as pool:
        rows = pool.map(job, jobs)
    out = {}
    for grp, seeds in [('调参种子', TUNE), ('留出种子', HOLD), ('全部 8 种子', TUNE + HOLD)]:
        print(f"\n===== {grp} {seeds} =====")
        for n in ['v8', 'v9']:
            rs = [r for r in rows if r['name'] == n and r['seed'] in seeds]
            m = {k: float(np.mean([r[k] for r in rs])) for k in K}
            out[(grp, n)] = m
            print(f"{n}: 成功 {m['rate']:5.1f}% 深倒扣 {m['deep']:5.1f}% 高转速 {m['fast']:5.1f}% 坠毁 {m['crashes']:5.1f}/512")
            print(f"    掉高: 中位 {m['drop50']:.2f}  P90 {m['drop90']:.2f}  P99 {m['drop99']:.2f}  均值 {m['drop_mean']:.2f} | 深倒扣 中位 {m['drop_deep50']:.2f} P90 {m['drop_deep90']:.2f}")
            print(f"    倾斜翻正: 中位 {m['txy50']:.3f}s P90 {m['txy90']:.3f}s | 含偏航翻正: 中位 {m['tup50']:.3f}s P90 {m['tup90']:.3f}s | 稳定悬停: 中位 {m['tst50']:.2f}s P90 {m['tst90']:.2f}s")
    print("\n逐种子成功率/坠毁:")
    for s in TUNE + HOLD:
        a = [r for r in rows if r['seed'] == s]
        print(s, '  '.join(f"{r['name']} {r['rate']:.1f}% ({r['crashes']})" for r in a))
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'stage17', 'final_compare.json'), 'w'),
              ensure_ascii=False, indent=1, default=float)
