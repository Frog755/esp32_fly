#!/usr/bin/env python3
"""calib_fit.py —— 加速度计椭球拟合（每轴零偏 + 比例）+ 芯片→机体轴映射 + 陀螺一致性校验（Stage 23）

与 calib_poses.py 的区别：不要求严格静止（手持也行）。
  1) 取 0.4 s 窗口均值，窗口内 |ω|max < 40 °/s 且加速度标准差 < 40 mg 视为"准静止"
  2) 拟合 a_true = (a_raw − o) / s，最小化 Σ (|a_true| − 1)²（6 参数：o_xyz、s_xyz）
  3) 准静止窗口按主导轴聚类，按首次出现顺序对应 平放 / 倒扣 / 机头朝上 / 机头朝下 / 左侧朝下 / 右侧朝下 → 芯片→机体 FLU 矩阵
  4) 陀螺一致性：对相邻两个准静止段之间积分陀螺（芯片系），预测重力方向的变化，与加速度计实测比较 → 验证陀螺比例与符号
用法：python3 calib_fit.py stream.csv out.json [起始秒，默认 0]
"""
import sys, json
import numpy as np
from scipy.optimize import least_squares

d = np.genfromtxt(sys.argv[1], delimiter=',', names=True)
t0_cut = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
T = (d['t_ms'] - d['t_ms'][0]) / 1e3
A = np.stack([d['ax_g'], d['ay_g'], d['az_g']], 1); W = np.stack([d['gx_dps'], d['gy_dps'], d['gz_dps']], 1)
WIN = 20
wins = []
for i in range(0, len(A) - WIN, WIN // 2):
    s = slice(i, i + WIN)
    if T[i] < t0_cut:
        continue
    if np.linalg.norm(W[s], axis=1).max() < 40 and A[s].std(0).max() < 0.04:
        wins.append((T[i], A[s].mean(0)))
M = np.array([m for _, m in wins]); TW = np.array([t for t, _ in wins])
print(f'准静止窗口 {len(M)} 个')


def resid(p):
    o, s = p[:3], p[3:]
    return np.linalg.norm((M - o) / s, axis=1) - 1.0


fit = least_squares(resid, x0=[0, 0, 0, 1, 1, 1])
o, s = fit.x[:3], fit.x[3:]
r = resid(fit.x)
print(f'零偏 o (g) = {np.round(o, 4)}   比例 s = {np.round(s, 4)}')
print(f'校正前 |a| 偏离 1g：均方根 {np.sqrt(np.mean((np.linalg.norm(M, axis=1) - 1) ** 2)) * 1e3:.0f} mg；校正后 {np.sqrt(np.mean(r ** 2)) * 1e3:.1f} mg（含手持晃动）')

# 主导轴聚类
C = (M - o) / s
keys = []
for c in C:
    ax = int(np.argmax(np.abs(c))); oth = np.delete(np.abs(c), ax)
    keys.append((ax, int(np.sign(c[ax]))) if abs(c[ax]) > 0.85 and oth.max() < 0.45 else None)
order = []; clusters = {}
for k, c, t in zip(keys, C, TW):
    if k is None:
        continue
    clusters.setdefault(k, []).append((t, c))
    if not order or order[-1] != k:
        order.append(k)
lab = lambda k: f'{"xyz"[k[0]]}{"+" if k[1] > 0 else "-"}'
print('朝向出现顺序：', ' → '.join(lab(k) for k in order))
for k, v in clusters.items():
    ts = [t for t, _ in v]
    print(f'  芯片 {lab(k)} 朝上：{len(v):3d} 个窗口，t={min(ts):.0f}~{max(ts):.0f}s，校正后均值 {np.round(np.mean([c for _, c in v], 0), 3)}')
POSES = [('平放', (2, +1)), ('倒扣', (2, -1)), ('机头朝上', (0, +1)), ('机头朝下', (0, -1)), ('左侧朝下', (1, -1)), ('右侧朝下', (1, +1))]
first = []
for k in order:
    if k not in first:
        first.append(k)
res = dict(acc_offset_g=o.tolist(), acc_scale=s.tolist(), fit_rms_mg=float(np.sqrt(np.mean(r ** 2)) * 1e3),
           n_windows=int(len(M)), orientation_order=[lab(k) for k in order])
if len(first) >= 6:
    R = np.zeros((3, 3))
    for (name, (b_ax, b_sg)), k in zip(POSES, first[:6]):
        R[b_ax] = 0; R[b_ax, k[0]] = k[1] * b_sg
        print(f'  {name:<5}：芯片 {lab(k)} 朝上 → 机体 {"xyz"[b_ax]}{"+" if b_sg > 0 else "-"}')
    det = float(np.linalg.det(R))
    perm_ok = np.allclose(np.abs(R).sum(0), 1) and np.allclose(np.abs(R).sum(1), 1)
    res.update(chip_to_body=R.astype(int).tolist(), det=det, mapping_valid=bool(perm_ok and det > 0.5))
    print('芯片 → 机体 FLU：\n', R.astype(int), f'\n行列式 {det:+.0f}', '✓ 是真旋转（不是镜像）' if res['mapping_valid'] else '✗ 无效')
else:
    print('⚠️ 不同朝向不足 6 个：', [lab(k) for k in first])

# 陀螺一致性：在两个准静止窗口之间积分芯片系陀螺，预测"重力方向"在芯片系里的变化
idx_w = [int(np.searchsorted(T, t)) for t in TW]
checks = []
for a in range(len(idx_w) - 1):
    i0, i1 = idx_w[a] + WIN // 2, idx_w[a + 1] + WIN // 2
    if i1 - i0 < 25 or i1 - i0 > 600:
        continue
    g0 = C[a] / np.linalg.norm(C[a]); g1 = C[a + 1] / np.linalg.norm(C[a + 1])
    ang_meas = np.degrees(np.arccos(np.clip(g0 @ g1, -1, 1)))
    if ang_meas < 45:
        continue
    g = g0.copy()
    for k in range(i0, i1):
        dt = (T[k + 1] - T[k])
        w = np.radians(W[k])
        g = g - np.cross(w, g) * dt                     # 世界固定矢量在旋转机体系中：dg/dt = −ω × g
        g /= np.linalg.norm(g)
    err = np.degrees(np.arccos(np.clip(g @ g1, -1, 1)))
    checks.append((TW[a], ang_meas, err))
if checks:
    print('陀螺积分 vs 加速度计（大角度转动段）：')
    for t, am, e in checks[:12]:
        print(f'  t≈{t:5.1f}s  实测转过 {am:5.1f}°  陀螺预测终点误差 {e:5.1f}°')
    errs = np.array([e for _, _, e in checks]); angs = np.array([a for _, a, _ in checks])
    res.update(gyro_check_n=len(checks), gyro_check_err_med_deg=float(np.median(errs)), gyro_check_rel_err_med=float(np.median(errs / angs)))
    print(f'  中位误差 {np.median(errs):.1f}°（相对转角 {np.median(errs / angs) * 100:.0f}%）——若陀螺符号或比例错，误差会接近转角本身（~100%）')
json.dump(res, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print('→', sys.argv[2])
