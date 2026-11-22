#!/usr/bin/env python3
"""sweep_v9.py —— v9 参数/消融并行扫描（多进程，每个配置一次 512 通道评测）"""
import os, sys, json, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
import numpy as np
from multiprocessing import Pool
from eval_recovery_metrics import run_metrics, summarize


def job(arg):
    name, ctrl, seed, n = arg
    from vec_expert9_env import VecExpert9
    r = run_metrics(VecExpert9, n, seed, env_kwargs=dict(ctrl=ctrl))
    row = summarize(r, name)
    row['ctrl'] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in ctrl.items()}
    row['seed'] = seed
    return row


def fmt(r):
    return (f"{r['name']:<38} 成功{r['rate']:5.1f}% 深{r['deep']:5.1f}% 坠{r['crashes']:3d} | 掉高 中位{r['drop50']:.2f} P90 {r['drop90']:.2f} 均{r['drop_mean']:.2f} 深中位{r['drop_deep50']:.2f}"
            f" | 翻正 中位{r['tup50']:.3f} P90 {r['tup90']:.3f} | 稳悬 中位{r['tst50']:.2f} P90 {r['tst90']:.2f}")


def run_jobs(jobs, procs=16):
    with Pool(procs) as pool:
        return pool.map(job, jobs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='json 文件: [[name, ctrl_dict], ...]')
    ap.add_argument('--seeds', type=int, nargs='+', default=[2024])
    ap.add_argument('--envs', type=int, default=512)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    cfgs = json.load(open(a.cfg))
    jobs = [(nm, ct, sd, a.envs) for nm, ct in cfgs for sd in a.seeds]
    rows = run_jobs(jobs)
    for r in rows:
        print(fmt(r))
    if a.out:
        json.dump(rows, open(a.out, 'w'), ensure_ascii=False, indent=1, default=float)
