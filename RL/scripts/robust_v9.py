#!/usr/bin/env python3
"""robust_v9.py —— 模型失配鲁棒性测试：只改仿真真值（env.dyn），控制器仍用名义参数。
对比 v8 与 v9 在电机滞后 / 质量 / 惯量失配下的表现，检查 v9 的增益是否依赖于"恰好知道"名义模型。"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
import numpy as np
from multiprocessing import Pool
from eval_recovery_metrics import run_metrics, summarize


def make_perturb(kind, k):
    def f(env):
        d = env.dyn
        if kind == 'lag':          # 电机时间常数 ×k
            d.mintau = d.mintau / k; d.mtau = d.mtau / k
        elif kind == 'mass':       # 质量 ×k（等效推力不足）
            d.mass = d.mass * k
        elif kind == 'inertia':    # 转动惯量 ×k
            d.J = d.J * k; d.Jinv = 1.0 / d.J
    return None if kind == 'nominal' else f


def job(arg):
    ctrl_name, kind, k, seed, ctrl = arg
    if ctrl_name == 'v8':
        from vec_expert8_env import VecExpert8 as C; kw = None
    else:
        from vec_expert9_env import VecExpert9 as C; kw = dict(ctrl=ctrl)
    r = run_metrics(C, 512, seed, env_kwargs=kw, perturb=make_perturb(kind, k))
    row = summarize(r, f"{ctrl_name} {kind}x{k}")
    row.update(ctrl_name=ctrl_name, kind=kind, k=k, seed=seed)
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ctrl', required=True, help='v9 ctrl json 文件')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024])
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    ctrl = json.load(open(a.ctrl))
    cases = [('nominal', 1.0), ('lag', 0.7), ('lag', 1.4), ('mass', 1.1), ('mass', 0.9), ('inertia', 1.2), ('inertia', 0.8)]
    jobs = [(c, kind, k, sd, ctrl) for c in ['v8', 'v9'] for kind, k in cases for sd in a.seeds]
    with Pool(16) as pool:
        rows = pool.map(job, jobs)
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['kind'], r['k'], r['ctrl_name'])].append(r)
    K = ['rate', 'crashes', 'drop50', 'drop90', 'drop_deep50', 'txy50', 'txy90', 'tst50', 'tst90']
    print(f"{'工况':<14}{'控制器':<5}{'成功率':>7}{'坠毁':>6}{'掉高中位':>8}{'掉高P90':>8}{'深倒扣掉高':>9}{'倾斜翻正中位':>10}{'P90':>7}{'稳悬中位':>8}{'P90':>7}")
    for kind, k in cases:
        for c in ['v8', 'v9']:
            rs = g[(kind, k, c)]
            m = {x: np.mean([r[x] for r in rs]) for x in K}
            print(f"{kind+' x'+str(k):<14}{c:<5}{m['rate']:6.1f}%{m['crashes']:6.1f}{m['drop50']:8.2f}{m['drop90']:8.2f}{m['drop_deep50']:9.2f}"
                  f"{m['txy50']:10.3f}{m['txy90']:7.3f}{m['tst50']:8.2f}{m['tst90']:7.2f}")
    if a.out:
        json.dump(rows, open(a.out, 'w'), ensure_ascii=False, indent=1, default=float)
