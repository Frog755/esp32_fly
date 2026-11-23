#!/usr/bin/env python3
"""gen_hil_vectors.py —— 生成上板测试向量头文件 code/bench_v9/main/hil_vectors.h（Stage 22）

从 Python 仿真闭环里分层抽样真实飞行状态（bang-bang / 模式切换 / 竖直刹车 / 方向退化 / 普通 PD），
输入先舍入到 float32（与固件拿到的一致），期望输出由 Python（float64）在同一份输入上计算。
另抽样门控 Mahony 的单次更新（悲观噪声，含翻转中与近悬停），以及一条 v8 输入用于对比耗时。
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env')); sys.path.insert(0, HERE)
import numpy as np
import verify_expert9_c_equivalence as V
import att_est_wrap as AE
from eval_recovery_metrics import run_metrics

OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'code', 'bench_v9', 'main', 'hil_vectors.h'))
N_CTRL_PER = 64           # 每类抽 64 个
N_MAH = 512
rng = np.random.default_rng(20261122)
F32 = np.float32

# ---------------- 控制器向量 ----------------
pools = {k: [] for k in ('bb', 'trans', 'brake', 'sing', 'pd')}
env = V.VecExpert9(num_envs=512, seed=2024, episode_s=5.0, params=V.P, tilt_cap_deg=180.0)
orig = env._action_to_throttle


def hook(a):
    inp = V.env_inputs(env); z0 = env._z_int.astype(F32); bb0 = env._bb.copy()
    u = orig(a)
    bb1 = env._bb
    q = inp['q'].astype(np.float64)
    r20 = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2]); r21 = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    cats = dict(bb=bb1 & (bb1 == bb0), trans=bb1 != bb0, brake=(~bb1) & (inp['vz_est'] < V.DEFAULT_CTRL['BRAKE_VZ']),
                sing=np.hypot(r20, r21) < 0.15, pd=(~bb1))
    if env._k[0] % 7 == 0 or cats['trans'].any():
        for k, m in cats.items():
            for i in np.where(m)[0][:6]:
                pools[k].append(({kk: vv[i].copy() for kk, vv in inp.items()}, float(z0[i]), bool(bb0[i])))
    return u


env._action_to_throttle = hook
env.reset()
for _ in range(1000):
    env.step(np.zeros((512, 4)))

sel = []
for k, lst in pools.items():
    idx = rng.choice(len(lst), size=min(N_CTRL_PER, len(lst)), replace=False)
    sel += [lst[i] for i in idx]
    print(f'{k:<6} 候选 {len(lst):6d}，抽 {len(idx)}')
NC = len(sel)
inp = {k: V.c32(np.stack([s[0][k] for s in sel])) for k in sel[0][0]}
z_in = np.array([s[1] for s in sel], F32); bb_in = np.array([s[2] for s in sel], bool)
u_exp, z_exp, bb_exp = V.py_ctrl(inp, z_in.copy(), bb_in)
T_exp = V.P.k_f * V.P.omega_max ** 2 * (u_exp * inp['vb'].astype(np.float64)[:, None] / V.P.v_ref) ** 2

# ---------------- 门控 Mahony 向量 ----------------
mah = []


def mhook(qe, wm, fb, dt, c):
    out = AE.mahony_update(qe, wm, fb, dt, c)
    if rng.random() < 0.15:
        wn = np.linalg.norm(wm, axis=1)
        for pick in (np.where(wn > 5)[0], np.where(wn < 2)[0], np.arange(len(qe))):
            if len(pick):
                i = int(rng.choice(pick)); mah.append((qe[i].copy(), wm[i].copy(), fb[i].copy()))
    return out


est = dict(mode='gated', kp=2.0, update_fn=mhook, **V.PESS)
run_metrics(AE.make_estimated(V.VecExpert9), 512, 42, steps=600, env_kwargs=dict(est=est, est_seed=42 * 104729 + 3))
mi = rng.choice(len(mah), size=N_MAH, replace=False)
mq = V.c32(np.stack([mah[i][0] for i in mi])); mw = V.c32(np.stack([mah[i][1] for i in mi])); mf = V.c32(np.stack([mah[i][2] for i in mi]))
mq_exp = AE.mahony_update(mq.astype(np.float64), mw.astype(np.float64), mf.astype(np.float64), 0.001, dict(AE.DEFAULT_EST, mode='gated', kp=2.0))
mk_gate = None


def arr(name, a, ctype='float'):
    a = np.asarray(a).ravel()
    if ctype == 'float':
        body = ', '.join(f'{float(F32(x))!r}f'.replace('inf', 'INFINITY') for x in a)
    else:
        body = ', '.join(str(int(x)) for x in a)
    lines = []
    parts = body.split(', ')
    for j in range(0, len(parts), 8):
        lines.append('    ' + ', '.join(parts[j:j + 8]))
    return f'static const {ctype} {name}[{len(a)}] = {{\n' + ',\n'.join(lines) + '\n};\n'


os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('/* 自动生成：RL/scripts/gen_hil_vectors.py —— 不要手改。\n'
            f' * 控制器向量 {NC} 组（bang-bang / 模式切换 / 竖直刹车 / 方向退化 / PD 各 ≤{N_CTRL_PER}），门控 Mahony 向量 {N_MAH} 组。\n'
            ' * 输入已舍入为 float32；期望值由 Python（float64）在同一份输入上计算。 */\n'
            '#pragma once\n#include <stdint.h>\n')
    f.write(f'#define HIL_NC {NC}\n#define HIL_NM {N_MAH}\n')
    for k in ('q', 'w', 'pos', 'vel', 'vz_est', 'z_est', 'target', 'vb'):
        f.write(arr(f'hil_c_{k}', inp[k]))
    f.write(arr('hil_c_zin', z_in)); f.write(arr('hil_c_bbin', bb_in.astype(int), 'uint8_t'))
    f.write(arr('hil_c_u', u_exp)); f.write(arr('hil_c_T', T_exp)); f.write(arr('hil_c_zout', z_exp))
    f.write(arr('hil_c_bbout', bb_exp.astype(int), 'uint8_t'))
    f.write(arr('hil_m_q', mq)); f.write(arr('hil_m_w', mw)); f.write(arr('hil_m_f', mf)); f.write(arr('hil_m_qout', mq_exp))
print(f'写出 {OUT}：控制器 {NC} 组（期望 bb=1 {int(bb_exp.sum())} 组、模式切换 {int((bb_exp != bb_in).sum())} 组），Mahony {N_MAH} 组，'
      f'{os.path.getsize(OUT) / 1024:.0f} KB')
