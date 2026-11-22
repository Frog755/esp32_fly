#!/usr/bin/env python3
"""eval_all_controllers.py —— 512 通道官方基准终极大对决:
  1. 纯规则基线 v5 (原冠军, 83.6%)
  2. 3D 轨迹优化查表控制器 (VecTableExpert)
  3. v6 提前交接闭环专家 (VecExpert6, 新冠军)

评测协议严格遵循项目铁律:
  - 成功 = 全程从未坠地 (z > -0.50m) 且 连续稳定悬停 >= 50 步
  - 严禁单通道手工 reset，禁用 auto-reset 污染统计
  - 分切片统计: 全局总分、深倒扣带 (θ >= 135°)、高转速带 (ω >= 14 rad/s)
  - 记录最低点中位下坠余量 (zmin - (-0.50m))
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
import numpy as np
from yao_kun_params import YaoKunK10Params
from vec_expert5_env import VecExpert5
from vec_table_expert_env import VecTableExpert
from vec_expert6_env import VecExpert6
from vec_expert7_env import VecExpert7
from vec_expert8_env import VecExpert8


def run_benchmark(env_cls, num_envs=512, seed=2024, name="Strategy", **kwargs):
    p = YaoKunK10Params()
    env = env_cls(num_envs=num_envs, seed=seed, episode_s=5.0, params=p, tilt_cap_deg=180.0, **kwargs)
    env.reset()

    q0 = env.dyn.quat.copy()
    w0 = env.dyn.w.copy()
    r22_0 = 1.0 - 2.0 * (q0[:, 1] ** 2 + q0[:, 2] ** 2)
    tilts_0 = np.degrees(np.arccos(np.clip(r22_0, -1, 1)))
    wnorms_0 = np.linalg.norm(w0, axis=1)

    smax = np.zeros(num_envs, dtype=np.int32)
    crashed = np.zeros(num_envs, dtype=bool)
    zmins = np.full(num_envs, 999.0, dtype=np.float32)

    last_pos = None
    last_vel = None

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

        # 记录 5.0s 重置前的真实终态（第 999 步）
        if t == 998:
            last_pos = env.dyn.pos.copy()
            last_vel = env.dyn.vel.copy()

    elapsed = time.time() - t0
    succ = (~crashed) & (smax >= 50)
    margins = zmins - (-0.50)

    idx_deep = np.where(tilts_0 >= 135.0)[0]
    idx_fast = np.where(wnorms_0 >= 14.0)[0]

    rate_total = succ.mean() * 100.0
    rate_deep = succ[idx_deep].mean() * 100.0 if len(idx_deep) > 0 else 0.0
    rate_fast = succ[idx_fast].mean() * 100.0 if len(idx_fast) > 0 else 0.0
    med_margin = float(np.median(margins))

    surv = ~crashed
    if last_pos is None:
        last_pos = env.dyn.pos.copy()
        last_vel = env.dyn.vel.copy()

    drifts_surv = np.hypot(last_pos[surv, 0], last_pos[surv, 1]) if surv.any() else np.array([0.0])
    vels_surv = np.hypot(last_vel[surv, 0], last_vel[surv, 1]) if surv.any() else np.array([0.0])
    alt_errs_surv = np.abs(last_pos[surv, 2] - 2.5) if surv.any() else np.array([0.0])

    drift_xy = float(drifts_surv.mean())
    drift_med = float(np.median(drifts_surv))
    drift_runaway = float((drifts_surv > 3.0).mean() * 100.0)
    vel_xy = float(vels_surv.mean())
    alt_err = float(alt_errs_surv.mean())

    print(f"==================================================")
    print(f"【{name}】 (N={num_envs}, seed={seed}, 运行用时: {elapsed:.1f}s)")
    print(f"  ★ 成功达标率      : {rate_total:5.1f}% ({succ.sum()}/{num_envs})")
    print(f"  ★ 深倒扣带(≥135°)  : {rate_deep:5.1f}% ({succ[idx_deep].sum()}/{len(idx_deep)})")
    print(f"  ★ 高转速带(≥14)    : {rate_fast:5.1f}% ({succ[idx_fast].sum()}/{len(idx_fast)})")
    print(f"  ★ 中位下坠余量    : {med_margin:+.2f}m (坠毁数: {crashed.sum()}/{num_envs})")
    print(f"  ★ 中位水平漂移    : {drift_med:.3f}m")
    print(f"  ★ 平均水平漂移    : {drift_xy:.3f}m (失控漂移>3m率: {drift_runaway:.1f}%)")
    print(f"  ★ 残余水平速度    : {vel_xy:.3f}m/s")
    print(f"  ★ 最终高度误差    : {alt_err:.3f}m")
    print(f"==================================================")

    return dict(rate_total=rate_total, rate_deep=rate_deep, rate_fast=rate_fast,
                med_margin=med_margin, drift_xy=drift_xy, drift_med=drift_med,
                drift_runaway=drift_runaway, vel_xy=vel_xy, alt_err=alt_err,
                crashes=crashed.sum(), succ=succ, zmins=zmins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    print(f"\n>>> 1. 运行纯规则基线 v5...")
    res_v5 = run_benchmark(VecExpert5, num_envs=args.envs, seed=args.seed, name="纯规则基线 v5")

    print(f"\n>>> 2. 运行 3D 查表混合控制器 (VecTableExpert)...")
    res_table = run_benchmark(VecTableExpert, num_envs=args.envs, seed=args.seed, name="3D 查表控制器")

    print(f"\n>>> 3. 运行 v6 提前交接专家 (VecExpert6)...")
    res_v6 = run_benchmark(VecExpert6, num_envs=args.envs, seed=args.seed, name="v6 提前交接专家")

    print(f"\n>>> 4. 运行 v7 级联定点自稳专家 (VecExpert7)...")
    res_v7 = run_benchmark(VecExpert7, num_envs=args.envs, seed=args.seed, name="v7 级联定点专家")

    print(f"\n>>> 5. 运行最新一代 v8 陀螺解耦定点旗舰 (VecExpert8)...")
    res_v8 = run_benchmark(VecExpert8, num_envs=args.envs, seed=args.seed, name="v8 陀螺解耦旗舰")

    print("\n" + "=" * 105)
    print("                              五代控制器官方 512 全量基准横向测评总榜")
    print("=" * 105)
    print(f"{'指标':<16} {'v5 基线':<10} {'3D 查表':<10} {'v6 专家':<10} {'v7 定点':<10} {'v8 旗舰(最新)':<16} {'v8 相对 v6 改善':<18}")
    print("-" * 105)
    d_tot = res_v8['rate_total'] - res_v6['rate_total']
    d_deep = res_v8['rate_deep'] - res_v6['rate_deep']
    d_xy = res_v8['drift_xy'] - res_v6['drift_xy']
    d_med = res_v8['drift_med'] - res_v6['drift_med']
    d_runaway = res_v8['drift_runaway'] - res_v6['drift_runaway']
    d_vel = res_v8['vel_xy'] - res_v6['vel_xy']
    d_alt = res_v8['alt_err'] - res_v6['alt_err']
    d_crash = res_v8['crashes'] - res_v6['crashes']

    print(f"{'成功达标率':<16} {res_v5['rate_total']:5.1f}%    {res_table['rate_total']:5.1f}%    {res_v6['rate_total']:5.1f}%    {res_v7['rate_total']:5.1f}%    {res_v8['rate_total']:5.1f}%            {d_tot:+5.1f}pp")
    print(f"{'深倒扣(≥135°)':<16} {res_v5['rate_deep']:5.1f}%    {res_table['rate_deep']:5.1f}%    {res_v6['rate_deep']:5.1f}%    {res_v7['rate_deep']:5.1f}%    {res_v8['rate_deep']:5.1f}%            {d_deep:+5.1f}pp")
    print(f"{'坠毁损坏数量':<16} {res_v5['crashes']:<10d} {res_table['crashes']:<10d} {res_v6['crashes']:<10d} {res_v7['crashes']:<10d} {res_v8['crashes']:<16d} {d_crash:+d} 架 (减小)")
    print(f"{'中位水平漂移':<16} {res_v5['drift_med']:5.3f}m   {res_table['drift_med']:5.3f}m   {res_v6['drift_med']:5.3f}m   {res_v7['drift_med']:5.3f}m   {res_v8['drift_med']:5.3f}m           {d_med:+5.3f}m (降{(1-res_v8['drift_med']/res_v6['drift_med'])*100:.1f}%)")
    print(f"{'平均水平漂移':<16} {res_v5['drift_xy']:5.3f}m   {res_table['drift_xy']:5.3f}m   {res_v6['drift_xy']:5.3f}m   {res_v7['drift_xy']:5.3f}m   {res_v8['drift_xy']:5.3f}m           {d_xy:+5.3f}m (降{(1-res_v8['drift_xy']/res_v6['drift_xy'])*100:.1f}%)")
    print(f"{'失控漂移>3m率':<16} {res_v5['drift_runaway']:5.1f}%    {res_table['drift_runaway']:5.1f}%    {res_v6['drift_runaway']:5.1f}%    {res_v7['drift_runaway']:5.1f}%    {res_v8['drift_runaway']:5.1f}%            {d_runaway:+5.1f}pp (降{(1-res_v8['drift_runaway']/res_v6['drift_runaway'])*100:.1f}%)")
    print(f"{'水平残余速度':<16} {res_v5['vel_xy']:5.3f}m/s {res_table['vel_xy']:5.3f}m/s {res_v6['vel_xy']:5.3f}m/s {res_v7['vel_xy']:5.3f}m/s {res_v8['vel_xy']:5.3f}m/s         {d_vel:+5.3f}m/s")
    print(f"{'最终高度误差':<16} {res_v5['alt_err']:5.3f}m   {res_table['alt_err']:5.3f}m   {res_v6['alt_err']:5.3f}m   {res_v7['alt_err']:5.3f}m   {res_v8['alt_err']:5.3f}m           {d_alt:+5.3f}m")
    print(f"{'中位下坠余量':<16} {res_v5['med_margin']:+.2f}m    {res_table['med_margin']:+.2f}m    {res_v6['med_margin']:+.2f}m    {res_v7['med_margin']:+.2f}m    {res_v8['med_margin']:+.2f}m            {res_v8['med_margin']-res_v6['med_margin']:+.2f}m")
    print("=" * 105)

