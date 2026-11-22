#!/usr/bin/env python3
"""build_flip_table3d.py —— 批量生成三轴四旋翼最优翻转轨迹表 (Stage 13 核心管线)

网格划分:
  - 方位角 phi (8 向): 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° (覆盖两主轴与两对角轴全向)
  - 倒扣倾角 tilt (4 档): 115°, 135°, 155°, 175° (覆盖 r22 < -0.25 的倒扣激活区)
  - 平行回正角速度 w_par (6 档): -12, -6, 0, 6, 12, 16 rad/s (覆盖低速与极限高速)
  总格数: 8 × 4 × 6 = 192 格。

质量保障铁律:
  - 每一格优化完成后，立即在 VecExpert5 严格全仿真中闭环实测
  - 记录 passed, zmin, stable_cnt，只有实跑通过的格才标记为合格 (valid=True)
  - 存储产物: RL/models/flip_table_3d.npz
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np
from flip_lto3d import solve3d_trajectory, verify_in_full_simulation


def solve_one_cell(item):
    idx, phi_deg, tilt_deg, w_par = item
    axis = np.array([np.cos(np.deg2rad(phi_deg)), np.sin(np.deg2rad(phi_deg)), 0.0])
    ang = np.deg2rad(tilt_deg)
    q0 = np.concatenate([[np.cos(ang / 2.0)], axis * np.sin(ang / 2.0)])
    # 沿回正方向（-axis）的角速度分量为 w_par => 初始角速度 w0 = -axis * w_par
    w0 = -axis * float(w_par)

    # 动态时界设计
    tf = 0.45 if tilt_deg <= 140 else 0.50
    if w_par < -4.0:
        tf += 0.05
    elif w_par > 8.0:
        tf -= 0.03
    tf = float(np.clip(tf, 0.38, 0.55))

    t0 = time.time()
    try:
        opt_res = solve3d_trajectory(q0, w0, tf=tf, K=10, maxiter=25)
        sim_res = verify_in_full_simulation(q0, w0, opt_res['u_seq'], episode_s=5.0)
        elapsed = time.time() - t0
        passed = bool(sim_res['passed'])
        zmin = float(sim_res['zmin'])
        stable = int(sim_res['stable_cnt'])
        u_opt = opt_res['u_opt']  # (10, 4)
        u_seq = opt_res['u_seq']  # (N, 4)
    except Exception as e:
        elapsed = time.time() - t0
        passed = False
        zmin = -9.99
        stable = 0
        u_opt = np.full((10, 4), 0.5)
        u_seq = np.full((int(round(tf / 0.005)), 4), 0.5)

    return dict(idx=idx, phi=phi_deg, tilt=tilt_deg, w_par=w_par, tf=tf,
                passed=passed, zmin=zmin, stable=stable, elapsed=elapsed,
                u_opt=u_opt, u_seq=u_seq)


def main():
    phis = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    tilts = [115.0, 135.0, 155.0, 175.0]
    w_pars = [-12.0, -6.0, 0.0, 6.0, 12.0, 16.0]

    tasks = []
    idx = 0
    for phi in phis:
        for tilt in tilts:
            for w in w_pars:
                tasks.append((idx, phi, tilt, w))
                idx += 1

    total = len(tasks)
    print(f"=== 启动 Stage 13 三轴翻转表并行构建 (共 {total} 格) ===")
    start_time = time.time()

    n_workers = min(10, mp.cpu_count())
    print(f"使用并行进程数: {n_workers}")

    results = []
    passed_count = 0

    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(solve_one_cell, tasks):
            results.append(r)
            if r['passed']:
                passed_count += 1
            done_cnt = len(results)
            pct = done_cnt / total * 100
            print(f"[{done_cnt:3d}/{total:3d}] ({pct:5.1f}%) "
                  f"phi={r['phi']:5.1f}° tilt={r['tilt']:5.1f}° w={r['w_par']:+5.1f} | "
                  f"{'PASS' if r['passed'] else 'FAIL'} (zmin={r['zmin']:+.2f}m, stable={r['stable']:3d}) "
                  f"用时: {r['elapsed']:.1f}s | 通过率: {passed_count}/{done_cnt} ({passed_count/done_cnt*100:.1f}%)")

    total_time = time.time() - start_time
    print(f"\n=== 建表全部完成！总用时: {total_time:.1f}s ({total_time/60:.1f} min) ===")
    print(f"全仿真通过数: {passed_count}/{total} ({passed_count/total*100:.1f}%)")

    # 按 idx 重新排序
    results.sort(key=lambda x: x['idx'])

    # 打包保存
    keys = np.array([[r['phi'], r['tilt'], r['w_par']] for r in results])
    passed = np.array([r['passed'] for r in results], dtype=bool)
    zmin = np.array([r['zmin'] for r in results], dtype=np.float32)
    stable = np.array([r['stable'] for r in results], dtype=np.int32)
    tfs = np.array([r['tf'] for r in results], dtype=np.float32)
    u_opts = np.array([r['u_opt'] for r in results], dtype=np.float32)  # (192, 10, 4)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'flip_table_3d.npz')
    np.savez_compressed(save_path,
                        keys=keys, passed=passed, zmin=zmin, stable=stable,
                        tfs=tfs, u_opts=u_opts, phis=phis, tilts=tilts, w_pars=w_pars)
    print(f"产物已保存至: {save_path} ({os.path.getsize(save_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

