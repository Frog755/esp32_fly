#!/usr/bin/env python3
"""eval_table_vs_v5.py —— 512 通道正式判据对决: 查表控制器 (VecTableExpert) vs v5 冠军基线

评估协议严格遵守项目铁律:
  - 成功 = 全程从未坠地 (z > -0.50m) 且 连续稳定悬停 >= 50 步
  - 严禁单通道手工 reset，禁用 auto-reset 污染统计
  - 分三段切片对比: 全局总分、高转速带 (ω >= 14 rad/s)、深倒扣带 (θ >= 135°)
  - 记录最低下坠余量 (zmin - (-0.50m))
"""
# --- [Stage 20 legacy 迁移引导] 本文件原位于 RL/scripts/，现移至 RL/legacy/flip_table/ ---
# _LEGACY_ORIG_DIR 指回原目录：原有基于 __file__ 的相对路径（../env、../models 等）语义不变；
# 并把主线 env/、scripts/ 与各 legacy 子目录加入 sys.path，跨目录的旧导入照常可用。逻辑未改动。
import os as _lg_os, sys as _lg_sys
_LEGACY_RL = _lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.abspath(__file__))))
_LEGACY_ORIG_DIR = _lg_os.path.join(_LEGACY_RL, 'scripts')
for _lg_p in [_LEGACY_RL, _lg_os.path.join(_LEGACY_RL, 'env'), _lg_os.path.join(_LEGACY_RL, 'scripts')] + sorted(
        _lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d) for _lg_d in _lg_os.listdir(_lg_os.path.join(_LEGACY_RL, 'legacy'))
        if _lg_os.path.isdir(_lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d)) and not _lg_d.startswith(('.', '_'))):
    if _lg_p not in _lg_sys.path:
        _lg_sys.path.insert(1, _lg_p)
# --- [/Stage 20 legacy 迁移引导] ---
import os, sys, time
sys.path.insert(0, os.path.join(_LEGACY_ORIG_DIR, '..', 'env'))
import numpy as np
from yao_kun_params import YaoKunK10Params
from vec_expert5_env import VecExpert5
from vec_table_expert_env import VecTableExpert


def run_benchmark(env_cls, num_envs=512, seed=2024, name="Strategy", **kwargs):
    p = YaoKunK10Params()
    env = env_cls(num_envs=num_envs, seed=seed, episode_s=5.0, params=p, tilt_cap_deg=180.0, **kwargs)
    env.reset()

    # 记录初始切片特征
    q0 = env.dyn.quat.copy()
    w0 = env.dyn.w.copy()
    r22_0 = 1.0 - 2.0 * (q0[:, 1] ** 2 + q0[:, 2] ** 2)
    tilts_0 = np.degrees(np.arccos(np.clip(r22_0, -1, 1)))
    wnorms_0 = np.linalg.norm(w0, axis=1)

    smax = np.zeros(num_envs, dtype=np.int32)
    crashed = np.zeros(num_envs, dtype=bool)
    zmins = np.full(num_envs, 999.0, dtype=np.float32)

    t0 = time.time()
    for t in range(1000):  # 5.0s / 0.005s = 1000 steps
        obs, r, done, infos = env.step(np.zeros((num_envs, 4)))
        for i, info in enumerate(infos):
            sc = info['stable_cnt']
            if sc > smax[i]:
                smax[i] = sc
            if info['crash']:
                crashed[i] = True
        z_curr = env.dyn.pos[:, 2]
        zmins = np.minimum(zmins, z_curr)

    elapsed = time.time() - t0
    succ = (~crashed) & (smax >= 50)
    margins = zmins - (-0.50)  # 相对坠毁线 (-0.5m) 的余量

    # 切片过滤
    idx_deep = np.where(tilts_0 >= 135.0)[0]
    idx_fast = np.where(wnorms_0 >= 14.0)[0]

    rate_total = succ.mean() * 100.0
    rate_deep = succ[idx_deep].mean() * 100.0 if len(idx_deep) > 0 else 0.0
    rate_fast = succ[idx_fast].mean() * 100.0 if len(idx_fast) > 0 else 0.0
    med_margin = float(np.median(margins))

    print(f"==================================================")
    print(f"【{name}】 (N={num_envs}, seed={seed}, 运行用时: {elapsed:.1f}s)")
    print(f"  ★ 全局总分      : {rate_total:5.1f}% ({succ.sum()}/{num_envs})")
    print(f"  ★ 深倒扣带(≥135°): {rate_deep:5.1f}% ({succ[idx_deep].sum()}/{len(idx_deep)})")
    print(f"  ★ 高转速带(≥14)  : {rate_fast:5.1f}% ({succ[idx_fast].sum()}/{len(idx_fast)})")
    print(f"  ★ 中位下坠余量  : {med_margin:+.2f}m (最低点中位数: {np.median(zmins):+.2f}m)")
    print(f"==================================================")

    return dict(rate_total=rate_total, rate_deep=rate_deep, rate_fast=rate_fast,
                med_margin=med_margin, succ=succ, zmins=zmins)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    print(f"\n>>> 运行纯 v5 冠军基线对齐...")
    res_v5 = run_benchmark(VecExpert5, num_envs=args.envs, seed=args.seed, name="纯规则基线 v5")

    print(f"\n>>> 运行 3D 查表混合控制器 (VecTableExpert)...")
    res_table = run_benchmark(VecTableExpert, num_envs=args.envs, seed=args.seed, name="3D 查表控制器 (Stage 13)")

    delta_total = res_table['rate_total'] - res_v5['rate_total']
    delta_deep = res_table['rate_deep'] - res_v5['rate_deep']
    delta_fast = res_table['rate_fast'] - res_v5['rate_fast']

    print("\n" + "=" * 55)
    print("              Stage 13 终评对决结果")
    print("=" * 55)
    print(f"指标             纯 v5      3D 查表      增量 (Delta)")
    print(f"总成功率        {res_v5['rate_total']:5.1f}%     {res_table['rate_total']:5.1f}%     {delta_total:+5.1f}pp")
    print(f"深倒扣(≥135°)   {res_v5['rate_deep']:5.1f}%     {res_table['rate_deep']:5.1f}%     {delta_deep:+5.1f}pp")
    print(f"高转速(≥14)     {res_v5['rate_fast']:5.1f}%     {res_table['rate_fast']:5.1f}%     {delta_fast:+5.1f}pp")
    print(f"中位下坠余量    {res_v5['med_margin']:+.2f}m     {res_table['med_margin']:+.2f}m     {res_table['med_margin']-res_v5['med_margin']:+.2f}m")
    print("=" * 55)

