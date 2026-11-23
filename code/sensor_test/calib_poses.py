#!/usr/bin/env python3
"""calib_poses.py —— 六面静置标定：从 50 Hz 流式数据里自动切出静止段，求
  1) 加速度计每轴零偏 + 比例因子（六面法：每根芯片轴朝上 / 朝下各一次）
  2) 芯片坐标系 → 机体 FLU（x 前 / y 左 / z 上）的带符号轴置换
     平放 → 机体 +z 朝上；机头朝上 → 机体 +x 朝上；左侧朝下 → 机体 −y 朝上（即右侧朝上）
用法：python3 calib_poses.py stream.csv out.json
姿态顺序（每个静止约 5 s）：平放、倒扣、机头朝上、机头朝下、左侧朝下、右侧朝下"""
import sys, json
import numpy as np

POSES = ['平放', '倒扣', '机头朝上', '机头朝下', '左侧朝下', '右侧朝下']
BODY_UP = {'平放': (2, +1), '倒扣': (2, -1), '机头朝上': (0, +1), '机头朝下': (0, -1), '左侧朝下': (1, -1), '右侧朝下': (1, +1)}

d = np.genfromtxt(sys.argv[1], delimiter=',', names=True)
A = np.stack([d['ax_g'], d['ay_g'], d['az_g']], 1); W = np.stack([d['gx_dps'], d['gy_dps'], d['gz_dps']], 1)
wn = np.linalg.norm(W, axis=1)
# 静止判据：0.6 s 窗口内陀螺模长 < 3 dps 且加速度各轴标准差 < 15 mg
win = 30
still = np.zeros(len(A), bool)
for i in range(len(A) - win):
    seg = slice(i, i + win)
    if wn[seg].max() < 3.0 and A[seg].std(0).max() < 0.015:
        still[seg] = True
# 连续静止段，丢弃 < 2 s 的段，每段取中间 80%；只保留"某根芯片轴明显朝上/朝下"的段（斜放的段丢弃）
raw = []; i = 0; T = d['t_ms']
while i < len(still):
    if still[i]:
        j = i
        while j < len(still) and still[j]:
            j += 1
        if j - i >= 100:
            k = (j - i) // 10
            m = A[i + k:j - k].mean(0)
            ax = int(np.argmax(np.abs(m)))
            others = np.delete(np.abs(m), ax)
            ok = abs(m[ax]) > 0.7 and others.max() < 0.35
            raw.append((m, ax, int(np.sign(m[ax])), (T[i] - T[0]) / 1e3, (j - i) / 50.0, ok))
        i = j
    else:
        i += 1
print(f'静止段 {len(raw)} 个：')
for m, ax, sg, t, dur, ok in raw:
    print(f'  t={t:6.1f}s 持续 {dur:4.1f}s  均值 {np.round(m, 3)}  → 芯片 {"xyz"[ax]}{"+" if sg > 0 else "-"} {"" if ok else "（斜放，丢弃）"}')
# 相邻同朝向合并
segs = []; seg_keys = []
for m, ax, sg, t, dur, ok in raw:
    if not ok:
        continue
    if seg_keys and seg_keys[-1] == (ax, sg):
        segs[-1] = (segs[-1] + m) / 2
    else:
        segs.append(m); seg_keys.append((ax, sg))
print(f'有效朝向段（合并相邻同朝向后）{len(segs)} 个：', [f'{"xyz"[a]}{"+" if g > 0 else "-"}' for a, g in seg_keys])
# 按"哪根芯片轴、正负号"归类；同类多段取平均
groups = {}
for m in segs:
    ax = int(np.argmax(np.abs(m))); sg = int(np.sign(m[ax]))
    groups.setdefault((ax, sg), []).append(m)
for (ax, sg), ms in sorted(groups.items()):
    print(f'  芯片 {"xyz"[ax]}{"+" if sg > 0 else "-"} 朝上：{len(ms)} 段，均值 {np.round(np.mean(ms, 0), 4)}')
res = dict(n_segments=len(segs), groups={f'{"xyz"[a]}{"+" if s > 0 else "-"}': np.mean(v, 0).tolist() for (a, s), v in groups.items()})
missing = [f'{"xyz"[a]}{"+" if s > 0 else "-"}' for a in range(3) for s in (1, -1) if (a, s) not in groups]
if missing:
    print('⚠️ 缺少这些朝向：', missing, '——无法完成六面标定'); res['missing'] = missing
else:
    off = np.zeros(3); scl = np.zeros(3)
    for a in range(3):
        up = np.mean(groups[(a, 1)], 0)[a]; dn = np.mean(groups[(a, -1)], 0)[a]
        off[a] = (up + dn) / 2; scl[a] = (up - dn) / 2          # 读数 = scl·真值 + off（单位 g）
    res.update(acc_offset_g=off.tolist(), acc_scale=scl.tolist())
    print('加速度计零偏 (g):', np.round(off, 4), ' 比例因子:', np.round(scl, 4))
    # 校正后各静止段模长
    nrm = [float(np.linalg.norm((m - off) / scl)) for m in segs]
    res['corrected_norm_g'] = nrm
    print('校正后各静止段 |a| (g):', np.round(nrm, 4))
# 轴映射：按姿态顺序取前 6 个静止段
# 姿态顺序：按"首次出现"的 6 个不同朝向
first = []; seen = set()
for m, key in zip(segs, seg_keys):
    if key not in seen:
        seen.add(key); first.append(m)
if len(first) >= 6:
    R = np.zeros((3, 3))     # body = R @ chip
    ok = True
    for name, m in zip(POSES, first[:6]):
        ax = int(np.argmax(np.abs(m))); sg = int(np.sign(m[ax]))
        b_ax, b_sg = BODY_UP[name]
        if R[b_ax].any() and not (R[b_ax, ax] == sg * b_sg):
            ok = False
        R[b_ax] = 0; R[b_ax, ax] = sg * b_sg
        print(f'  {name:<5}：芯片 {"xyz"[ax]}{"+" if sg > 0 else "-"} 朝上 → 机体 {"xyz"[b_ax]}{"+" if b_sg > 0 else "-"}')
    det = float(np.linalg.det(R))
    valid = ok and abs(abs(det) - 1) < 1e-9 and np.allclose(np.abs(R).sum(0), 1) and np.allclose(np.abs(R).sum(1), 1)
    res.update(chip_to_body=R.astype(int).tolist(), det=det, mapping_valid=bool(valid and det > 0))
    print('芯片 → 机体 FLU 矩阵：\n', R.astype(int), f'\n行列式 {det:+.0f}（必须为 +1，否则是镜像，说明姿态做反了或顺序不对）')
json.dump(res, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print('→', sys.argv[2])
