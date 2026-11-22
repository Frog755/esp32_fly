#!/usr/bin/env python3
"""verify_expert9_c_equivalence.py —— v9 控制器 + 门控 Mahony 的 C 实现 vs Python 逐项校验（Stage 21）

  A. 常数核对      ：C 头文件 #define 与 YaoKunK10Params / DEFAULT_CTRL / DEFAULT_EST 逐个比对，电机符号与 Minv 一致
  B. 控制器逐步核对：Python v9 闭环飞 512 通道 × 1000 步，每一步把【同一份 float32 输入 + 同一份内部状态】
                     同时喂给 C 和 Python，比较输出油门 / 模式 / 积分器；超差样本做"扰动测试"判断是否落在开关边界
  C. 控制器闭环    ：C 控制器直接在仿真里飞 8 个种子，与 Python v9 的成功率 / 掉高 / 耗时对比
  D. 门控 Mahony   ：D1 静止初始化；D2 10 s 悲观噪声轨迹上逐次更新核对；D3 C 估计器 + C 控制器整链闭环
  E. ESP32-S3 交叉编译：xtensa-esp32s3-elf-gcc 编译，检查无双精度软浮点调用、代码体积
  F. 主机单步耗时

用法：python3 verify_expert9_c_equivalence.py [--quick]
"""
import os, sys, re, json, time, glob, ctypes, argparse, subprocess, tempfile, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
CODE = os.path.abspath(os.path.join(RL, '..', 'code', 'connection_test', 'main'))
sys.path.insert(0, os.path.join(RL, 'env')); sys.path.insert(0, HERE)
import numpy as np
from multiprocessing import Pool
from yao_kun_params import YaoKunK10Params
from vec_recovery_env import VecRecoveryEnv, G
from vec_expert9_env import VecExpert9, DEFAULT_CTRL
import att_est_wrap as AE
from eval_recovery_metrics import run_metrics, summarize

P = YaoKunK10Params()
TOL = 1e-4                     # 单电机油门差超过此值记为"显著差异"
F32 = np.float32
OUT_DIR = os.path.join(RL, 'results', 'stage21')
REPORT = {}

WRAP = r'''
#include <stdint.h>
#include <time.h>
#include "recovery_expert9.h"
#include "mahony_gated.h"
void exp9_batch(int n, const float* q, const float* w, const float* pos, const float* vel,
                const float* vz_est, const float* z_est, const float* target, const float* vb,
                float* z_int, uint8_t* bb, float* out_u) {
    for (int i = 0; i < n; i++) {
        recovery9_state_t st; st.z_int = z_int[i]; st.bb = bb[i];
        recovery_expert9_step(&st, q + 4 * i, w + 3 * i, pos + 3 * i, vel + 3 * i, vz_est[i], z_est[i],
                              target + 3 * i, vb[i], out_u + 4 * i);
        z_int[i] = st.z_int; bb[i] = st.bb;
    }
}
void mahony_batch(int n, float* q, const float* gyro, const float* acc, float dt) {
    for (int i = 0; i < n; i++) {
        mahony_gated_t m; for (int j = 0; j < 4; j++) m.q[j] = q[4 * i + j];
        mahony_gated_update(&m, gyro + 3 * i, acc + 3 * i, dt);
        for (int j = 0; j < 4; j++) q[4 * i + j] = m.q[j];
    }
}
void mahony_init_acc_batch(int n, const float* acc, float* q) {
    for (int i = 0; i < n; i++) {
        mahony_gated_t m; mahony_gated_init_from_accel(&m, acc + 3 * i);
        for (int j = 0; j < 4; j++) q[4 * i + j] = m.q[j];
    }
}
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
double bench_exp9(int n, const float* q, const float* w, const float* pos, const float* vel,
                  const float* vz_est, const float* z_est, const float* target, const float* vb, int reps) {
    volatile float sink = 0.0f; float u[4]; recovery9_state_t st = {0.0f, 0u};
    double t0 = now_ns();
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < n; i++) {
            recovery_expert9_step(&st, q + 4 * i, w + 3 * i, pos + 3 * i, vel + 3 * i, vz_est[i], z_est[i],
                                  target + 3 * i, vb[i], u);
            sink += u[0];
        }
    return (now_ns() - t0) / ((double)reps * n);
}
double bench_mahony(int n, const float* gyro, const float* acc, int reps) {
    volatile float sink = 0.0f; mahony_gated_t m = {{1.0f, 0.0f, 0.0f, 0.0f}};
    double t0 = now_ns();
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < n; i++) { sink += mahony_gated_update(&m, gyro + 3 * i, acc + 3 * i, 0.001f); }
    return (now_ns() - t0) / ((double)reps * n);
}
'''

LIB = None
BUILD_DIR = tempfile.mkdtemp(prefix='verify_exp9_')


def fp(nd=None):
    return np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')


def build():
    global LIB
    c = os.path.join(BUILD_DIR, 'wrap.c'); so = os.path.join(BUILD_DIR, 'libexp9.so')
    open(c, 'w').write(WRAP)
    cmd = ['gcc', '-O2', '-std=c99', '-D_POSIX_C_SOURCE=199309L', '-Wall', '-Wextra', '-Wdouble-promotion', '-Wfloat-conversion',
           '-Werror=double-promotion', '-Werror=float-conversion', '-shared', '-fPIC', '-I', CODE, '-o', so, c, '-lm']
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    REPORT['host_build_warnings'] = r.stderr.strip()
    LIB = ctypes.CDLL(so)
    u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    LIB.exp9_batch.argtypes = [ctypes.c_int] + [fp()] * 8 + [fp(), u8, fp()]
    LIB.mahony_batch.argtypes = [ctypes.c_int, fp(), fp(), fp(), ctypes.c_float]
    LIB.mahony_init_acc_batch.argtypes = [ctypes.c_int, fp(), fp()]
    LIB.bench_exp9.argtypes = [ctypes.c_int] + [fp()] * 8 + [ctypes.c_int]
    LIB.bench_exp9.restype = ctypes.c_double
    LIB.bench_mahony.argtypes = [ctypes.c_int, fp(), fp(), ctypes.c_int]
    LIB.bench_mahony.restype = ctypes.c_double
    return so


def c32(x):
    return np.ascontiguousarray(x, dtype=F32)


def c_ctrl(inp, z_int, bb):
    """C 控制器批量调用；z_int / bb 为 float32 / uint8 数组，原地更新。"""
    N = len(z_int); u = np.zeros((N, 4), F32)
    LIB.exp9_batch(N, inp['q'], inp['w'], inp['pos'], inp['vel'], inp['vz_est'], inp['z_est'], inp['target'], inp['vb'], z_int, bb, u)
    return u


def env_inputs(env):
    s = env.dyn
    return {k: c32(v) for k, v in dict(q=s.quat, w=s.w, pos=s.pos, vel=s.vel, vz_est=env._vz_est, z_est=env._z_est,
                                         target=env._target, vb=s.v_batt).items()}


_PROBES = {}


def py_ctrl(inp, z0, bb0):
    """Python v9 在给定（float32 舍入后的）输入与内部状态上算一步。"""
    N = len(z0)
    if N not in _PROBES:
        _PROBES[N] = VecExpert9(num_envs=N, seed=0, episode_s=5.0, params=P, tilt_cap_deg=180.0)
    e = _PROBES[N]; d = e.dyn
    d.quat = inp['q'].astype(np.float64); d.w = inp['w'].astype(np.float64)
    d.pos = inp['pos'].astype(np.float64); d.vel = inp['vel'].astype(np.float64); d.v_batt = inp['vb'].astype(np.float64)
    e._z_est = inp['z_est'].astype(np.float64); e._vz_est = inp['vz_est'].astype(np.float64)
    e._target = inp['target'].astype(np.float64)
    e._z_int = np.asarray(z0, np.float64).copy(); e._bb = np.asarray(bb0, bool).copy()
    u = e._action_to_throttle(None)
    return u, e._z_int.copy(), e._bb.copy()


# ============================ A. 常数 ============================
def section_A():
    h = open(os.path.join(CODE, 'recovery_expert9.h'), encoding='utf-8').read()
    hm = open(os.path.join(CODE, 'mahony_gated.h'), encoding='utf-8').read()
    defs = {m.group(1): float(m.group(2)) for m in re.finditer(r'#define\s+((?:EXP9|MAHONY)_\w+)\s+\(?(-?[0-9.]+(?:e-?[0-9]+)?)f\)?', h + hm)}
    c = DEFAULT_CTRL
    rp = np.asarray(P.rotor_pos, float); rd = np.asarray(P.rotor_dir, float)
    Mi = np.linalg.inv(np.vstack([np.ones(4), rp[:, 1], -rp[:, 0], -rd * P.k_m_ratio]))
    exp = dict(EXP9_MASS=P.mass, EXP9_G=G, EXP9_TMAX=P.k_f * P.omega_max ** 2 * float(np.asarray(getattr(P, 'motor_gain', [1.0]))[0]),
               EXP9_VREF=P.v_ref, EXP9_VBATT_MIN=2.8, EXP9_ARM=abs(rp[0, 0]), EXP9_KM=P.k_m_ratio,
               EXP9_IX=P.inertia[0], EXP9_IY=P.inertia[1], EXP9_IZ=P.inertia[2],
               EXP9_MI_XY=abs(Mi[0, 1]), EXP9_MI_Z=abs(Mi[0, 3]), EXP9_DT=0.005,
               MAHONY_KP=AE.DEFAULT_EST['kp'], MAHONY_F_BAND=AE.DEFAULT_EST['f_band'], MAHONY_W_LO=AE.DEFAULT_EST['w_lo'],
               MAHONY_W_HI=AE.DEFAULT_EST['w_hi'], MAHONY_G=AE.G)
    for k in ['TH_H', 'TH_HYS', 'W_H', 'T_LAG', 'K_ALPHA', 'R_LO', 'FLIP_FZ_LO', 'KD_PERP', 'T_DIR', 'K_TAU_INV', 'K_TAU_UP',
              'YAW_RES', 'KZ_FLIP', 'KP_ATT', 'KD_ATT', 'KZ_ATT', 'KP_POS_XY', 'KD_VEL_XY', 'MAX_TILT_RAD', 'BRAKE_VZ',
              'BRAKE_TFRAC', 'KV_Z', 'KI_Z', 'VZ_UP', 'VZ_DN', 'KZ_REF']:
        exp['EXP9_' + k] = float(c[k])
    bad = []
    for k, v in exp.items():
        if k not in defs:
            bad.append(f'{k} 头文件里没有'); continue
        if not (F32(defs[k]) == F32(v) or abs(defs[k] - v) <= 1e-6 * max(1.0, abs(v))):
            bad.append(f'{k}: 头文件 {defs[k]} vs Python {v}')
    sg = {}
    for nm in ('X', 'Y', 'Z'):
        m = re.search(rf'EXP9_SGN_{nm}\[4\]\s*=\s*\{{([^}}]*)\}}', h)
        sg[nm] = np.array([float(x.strip().rstrip('f')) for x in m.group(1).split(',')])
    for nm, col in (('X', 1), ('Y', 2), ('Z', 3)):
        if not np.array_equal(sg[nm], np.sign(Mi[:, col])):
            bad.append(f'EXP9_SGN_{nm} {sg[nm]} ≠ sign(Minv[:,{col}]) {np.sign(Mi[:, col])}')
    if not np.allclose(Mi[:, 0], 0.25):
        bad.append('Minv 第 0 列不是 1/4')
    for k, want in (('reduced_att', True), ('prio_mixer', True), ('vbatt_aware', True), ('YAW_IN_FLIP', False)):
        if bool(c[k]) != want:
            bad.append(f'DEFAULT_CTRL[{k}]={c[k]}，C 实现假设为 {want}')
    if not c['KZ_FLIP'] > 0:
        bad.append('C 实现假设 KZ_FLIP > 0')
    REPORT['A'] = dict(n_checked=len(exp) + 3 + 5, bad=bad)
    print(f'[A] 常数核对：{len(exp)} 个常数 + 3 组电机符号 + 5 项开关假设 → ' + ('全部一致' if not bad else f'{len(bad)} 处不一致：{bad}'))
    return not bad


# ============================ B. 控制器逐步核对 ============================
def boundary_test(inp1, z0, bb0, K=24, eps=4.0 * 2 ** -24, seed=0):
    """把单个样本的输入做 float32 舍入量级的扰动，看 Python 自己的输出是否跳变（= 处在开关面上）。"""
    rng = np.random.default_rng(seed)
    rep = {k: np.repeat(v[None], K, 0).astype(np.float64) for k, v in inp1.items()}
    for k in ('q', 'w', 'pos', 'vel', 'vz_est', 'z_est', 'vb'):
        rep[k] = rep[k] * (1.0 + eps * rng.standard_normal(rep[k].shape)) + 1e-9 * rng.standard_normal(rep[k].shape)
    rep = {k: c32(v) if k != 'q' else c32(v) for k, v in rep.items()}
    u, _, bbk = py_ctrl(rep, np.full(K, z0), np.full(K, bb0))
    u0, _, _ = py_ctrl({k: v[None] for k, v in inp1.items()}, np.array([z0]), np.array([bb0]))
    return float(np.abs(u - u0).max()), bool(bbk.any() != bbk.all())


def section_B(seeds, steps):
    tot = dict(n=0, bb=0, trans=0, brake=0, sing=0, sig=0, sig_boundary=0, bb_mis=0, bb_mis_boundary=0, nearzero_only=0)
    max_cont = 0.0; max_live = 0.0; max_dz = 0.0; max_dT = 0.0; max_nz = 0.0; examples = []
    TM = P.k_f * P.omega_max ** 2
    for sd in seeds:
        env = VecExpert9(num_envs=512, seed=sd, episode_s=steps * 0.005, params=P, tilt_cap_deg=180.0)
        orig = env._action_to_throttle

        def hook(a, _orig=orig, _env=env):
            nonlocal max_cont, max_live, max_dz, max_dT, max_nz
            inp = env_inputs(_env)
            z0 = _env._z_int.copy(); bb0 = _env._bb.copy()
            u_live = _orig(a)
            z_in = z0.astype(F32)                          # 两边拿到同一份 float32 积分器初值
            zc = z_in.copy(); bc = bb0.astype(np.uint8)     # C 原地更新 zc / bc
            uc = c_ctrl(inp, zc, bc)
            up, zp, bp = py_ctrl(inp, z_in.copy(), bb0)
            # 油门 u ∝ √推力：推力本应为 0 的电机，float32 多出 1 个 ulp 推力（~3e-8 N）就会被放大成 ~3e-4 油门差，
            # 且远低于电机死区（pwm_deadzone=0.05）——这类"近零推力"差异在推力域比较、单独计数
            vbv = inp['vb'].astype(np.float64)[:, None]
            Tc = TM * (uc * vbv / P.v_ref) ** 2; Tp = TM * (up * vbv / P.v_ref) ** 2
            dT = np.abs(Tc - Tp).max(1)
            du_raw = np.abs(uc - up)
            live = (uc > 1e-3) | (up > 1e-3)
            du = np.where(live, du_raw, 0.0).max(1)
            nzo = (du_raw.max(1) > TOL) & ~(du > TOL)
            tot['nearzero_only'] += int(nzo.sum())
            if nzo.any():
                max_nz = max(max_nz, float(du_raw.max(1)[nzo].max()))
            max_dT = max(max_dT, float(dT.max()))
            tot['n'] += len(du); tot['bb'] += int(bp.sum()); tot['trans'] += int((bp != bb0).sum())
            q = inp['q'].astype(np.float64)
            r20 = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2]); r21 = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
            tot['sing'] += int((np.hypot(r20, r21) < 0.15).sum())
            tot['brake'] += int(((~bp) & (inp['vz_est'] < DEFAULT_CTRL['BRAKE_VZ'])).sum())
            bbm = (bc.astype(bool) != bp)
            sig = (du > TOL) | bbm
            for i in np.where(sig)[0]:
                one = {k: v[i] for k, v in inp.items()}
                spread, bb_flip = boundary_test(one, float(z_in[i]), bool(bb0[i]), seed=int(i))
                is_b = spread > TOL / 2 or bb_flip
                if du[i] > TOL:
                    tot['sig'] += 1; tot['sig_boundary'] += int(is_b)
                if bbm[i]:
                    tot['bb_mis'] += 1; tot['bb_mis_boundary'] += int(is_b)
                if len(examples) < 8:
                    examples.append(dict(seed=sd, step=int(_env._k[i]), ch=int(i), du=float(du[i]), bb_mismatch=bool(bbm[i]),
                                         python_spread=spread, boundary=bool(is_b)))
            ok = ~sig
            if ok.any():
                max_cont = max(max_cont, float(du[ok].max()))
                max_live = max(max_live, float(np.where(live, np.abs(uc - u_live), 0.0).max(1)[ok].max()))
            max_dz = max(max_dz, float(np.abs(zc - zp).max()))
            return u_live
        env._action_to_throttle = hook
        env.reset()
        for _ in range(steps):
            env.step(np.zeros((512, 4)))
    nb = tot['sig'] - tot['sig_boundary'] + tot['bb_mis'] - tot['bb_mis_boundary']
    REPORT['B'] = dict(seeds=list(seeds), steps=steps, **tot, max_cont_du=max_cont, max_live_du=max_live, max_dz=max_dz,
                       max_dT_N=max_dT, max_nearzero_du=max_nz, non_boundary=nb, examples=examples)
    print(f"[B] 逐步核对：{tot['n']:,} 个样本（{len(seeds)} 种子 × {steps} 步 × 512 通道；bang-bang {tot['bb'] / tot['n'] * 100:.1f}%、"
          f"模式切换 {tot['trans']:,} 次、竖直刹车 {tot['brake']:,}、方向退化分支 {tot['sing']:,}）")
    print(f"    连续误差：单电机油门最大差 {max_cont:.2e}（同一 float32 输入）；对未舍入的 float64 实况 {max_live:.2e}；"
          f"单电机推力最大差 {max_dT:.2e} N（悬停约 0.196 N/电机）；高度积分器最大差 {max_dz:.2e}")
    print(f"    近零推力电机（u<1e-3，低于死区 0.05）：{tot['nearzero_only']} 个样本油门差 >{TOL:g}，最大 {max_nz:.2e}"
          f"（= √放大 1 个 float32 ulp 推力，推力差已含在上面的推力最大差里）")
    print(f"    显著差异（>{TOL:g} 或模式不同）：油门 {tot['sig']} 个 / 模式 {tot['bb_mis']} 个，"
          f"其中落在开关边界上的（float32 舍入量级扰动就会让 Python 自己跳变）：{tot['sig_boundary']} / {tot['bb_mis_boundary']}；非边界差异 {nb} 个")
    return nb == 0 and max_cont < TOL and max_dT < 1e-6


# ============================ C. 控制器闭环 ============================
class VecExpert9C(VecRecoveryEnv):
    """用 C 实现（recovery_expert9.h）驱动的 v9；内部状态以 float32 / uint8 保存，与固件一致。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._zc = np.zeros(self.num_envs, F32); self._bc = np.zeros(self.num_envs, np.uint8)

    def reset(self):
        self._zc[:] = 0; self._bc[:] = 0
        return super().reset()

    def _reset_done(self, dones):
        super()._reset_done(dones)
        self._zc[dones] = 0; self._bc[dones] = 0

    def step_async(self, actions):
        self._actions = actions

    def _action_to_throttle(self, a):
        return c_ctrl(env_inputs(self), self._zc, self._bc).astype(np.float64)


def _job_closed(arg):
    kind, seed, steps, est = arg
    B = VecExpert9C if kind == 'C' else VecExpert9
    if est is None:
        r = run_metrics(B, 512, seed, steps=steps)
    else:
        e = dict(est)
        if kind == 'C':
            e['update_fn'] = c_mahony_update
        r = run_metrics(AE.make_estimated(B), 512, seed, steps=steps, env_kwargs=dict(est=e, est_seed=seed * 104729 + 3))
    row = summarize(r, kind)
    row.update(kind=kind, seed=seed, succ_vec=r['succ'].astype(int).tolist(), drop_vec=np.round(r['drop'], 4).tolist())
    return row


K_STAT = ['rate', 'deep', 'fast', 'crashes', 'drop50', 'drop90', 'txy50', 'txy90', 'tst50', 'tst90']


def compare_closed(rows, label):
    res = {}
    for kind in ('Py', 'C'):
        rs = [r for r in rows if r['kind'] == kind]
        res[kind] = {k: float(np.nanmean([r[k] for r in rs])) for k in K_STAT}
    seeds = sorted({r['seed'] for r in rows})
    same = []; ddrop = []
    for sd in seeds:
        a = next(r for r in rows if r['kind'] == 'Py' and r['seed'] == sd); b = next(r for r in rows if r['kind'] == 'C' and r['seed'] == sd)
        same.append(np.mean(np.array(a['succ_vec']) == np.array(b['succ_vec'])))
        ddrop.append(np.abs(np.array(a['drop_vec']) - np.array(b['drop_vec'])))
    ddrop = np.concatenate(ddrop)
    res['same_outcome'] = float(np.mean(same)); res['drop_diff_med'] = float(np.median(ddrop)); res['drop_diff_p99'] = float(np.percentile(ddrop, 99)); res['drop_diff_max'] = float(ddrop.max())
    res['per_seed'] = {sd: (next(r for r in rows if r['kind'] == 'Py' and r['seed'] == sd)['rate'],
                            next(r for r in rows if r['kind'] == 'C' and r['seed'] == sd)['rate']) for sd in seeds}
    p, c = res['Py'], res['C']
    print(f"    {label}（{len(seeds)} 种子 × 512）")
    print(f"      Python: 成功 {p['rate']:5.1f}%  坠毁 {p['crashes']:4.1f}  掉高中位 {p['drop50']:.3f} P90 {p['drop90']:.3f}  倾斜翻正 {p['txy50']:.3f}s  稳悬 {p['tst50']:.3f}s")
    print(f"      C     : 成功 {c['rate']:5.1f}%  坠毁 {c['crashes']:4.1f}  掉高中位 {c['drop50']:.3f} P90 {c['drop90']:.3f}  倾斜翻正 {c['txy50']:.3f}s  稳悬 {c['tst50']:.3f}s")
    print(f"      逐通道成败相同 {res['same_outcome'] * 100:.2f}%，逐通道掉高差 中位 {res['drop_diff_med']:.1e} m / P99 {res['drop_diff_p99']:.1e} m / 最大 {res['drop_diff_max']:.1e} m")
    return res


def section_C(seeds):
    with Pool(16) as pool:
        rows = pool.map(_job_closed, [(k, sd, 1000, None) for k in ('Py', 'C') for sd in seeds])
    print('[C] 控制器闭环：C 控制器直接飞')
    REPORT['C'] = compare_closed(rows, '理想传感器 5 s')
    r = REPORT['C']
    return abs(r['C']['rate'] - r['Py']['rate']) <= 0.3 and abs(r['C']['drop50'] - r['Py']['drop50']) <= 0.02


# ============================ D. 门控 Mahony ============================
def c_mahony_update(qe, wm, fb, dt, c):
    assert c['mode'] == 'gated'
    q = c32(qe)
    LIB.mahony_batch(len(q), q, c32(wm), c32(fb), float(dt))
    return q.astype(np.float64)


def quat_angle_deg(a, b):
    """两四元数之间的旋转角（°）：r = conj(a)⊗b，角 = 2·atan2(|r_xyz|, |r_w|)（比 arccos(点积) 在小角度下准确）。"""
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    ac = a * np.array([1.0, -1.0, -1.0, -1.0])
    r = AE.qmul(ac, b)
    return np.degrees(2 * np.arctan2(np.linalg.norm(r[:, 1:], axis=1), np.abs(r[:, 0])))


PESS = dict(gyro_sigma=0.2, gyro_bias=0.05, acc_sigma=3.0, init_err_deg=5.0)


def section_D(quick):
    ok = True
    # D1 静止初始化
    rng = np.random.default_rng(7)
    a = rng.standard_normal((20000, 3)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    a[:4] = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 0.6, -0.8]]
    acc = c32(a * 9.81); q = np.zeros((len(a), 4), F32)
    LIB.mahony_init_acc_batch(len(a), acc, q)
    up = AE.up_in_body(q.astype(np.float64)); up /= np.linalg.norm(up, axis=1, keepdims=True)
    ad = acc.astype(np.float64); ad /= np.linalg.norm(ad, axis=1, keepdims=True)     # 与 C 拿到的同一份 float32 输入比
    err = np.degrees(np.arctan2(np.linalg.norm(np.cross(up, ad), axis=1), (up * ad).sum(1)))
    yaw_ok = np.abs(q[:, 3]).max()
    REPORT['D1'] = dict(n=len(a), max_err_deg=float(err.max()), max_qz=float(yaw_ok))
    print(f"[D1] 静止初始化：{len(a):,} 个随机方向（含完全倒置）→ 估计\"上\"方向与加速度计方向最大偏差 {err.max():.2e}°，偏航分量 |qz| 最大 {yaw_ok:.1e}")
    ok &= err.max() < 1e-3
    # D2 逐次更新核对
    steps = 600 if quick else 2000
    stats = dict(n=0, max_deg=0.0, sum_deg=0.0, max_deg_live=0.0)

    def hook(qe, wm, fb, dt, c):
        out = AE.mahony_update(qe, wm, fb, dt, c)                         # Python 原样驱动仿真
        q32, w32, f32 = c32(qe), c32(wm), c32(fb)
        qc = q32.copy(); LIB.mahony_batch(len(qc), qc, w32, f32, float(dt))
        qp = AE.mahony_update(q32.astype(np.float64), w32.astype(np.float64), f32.astype(np.float64), dt, c)
        d = quat_angle_deg(qc.astype(np.float64), qp)
        stats['n'] += len(d); stats['max_deg'] = max(stats['max_deg'], float(d.max())); stats['sum_deg'] += float(d.sum())
        stats['max_deg_live'] = max(stats['max_deg_live'], float(quat_angle_deg(qc.astype(np.float64), out).max()))
        return out
    est = dict(mode='gated', kp=2.0, update_fn=hook, **PESS)
    run_metrics(AE.make_estimated(VecExpert9), 512, 2024, steps=steps, env_kwargs=dict(est=est, est_seed=2024 * 104729 + 3))
    REPORT['D2'] = dict(steps=steps, **stats, mean_deg=stats['sum_deg'] / stats['n'])
    print(f"[D2] 门控 Mahony 逐次更新：悲观噪声 {steps * 0.005:.0f} s 轨迹，{stats['n']:,} 次更新（1 kHz × 512 通道，含全部翻转过程）")
    print(f"     单次更新后四元数差：最大 {stats['max_deg']:.2e}°、平均 {stats['sum_deg'] / stats['n']:.2e}°（同一 float32 输入）；对 float64 实况最大 {stats['max_deg_live']:.2e}°")
    ok &= stats['max_deg'] < 1e-3
    # D3 整链闭环
    seeds = [2024, 42] if quick else [2024, 42, 100, 2026, 7]
    REPORT['D3'] = {}
    print('[D3] 整链闭环：C 门控 Mahony + C 控制器 vs Python 门控 Mahony + Python 控制器')
    for label, est, steps in [('常规噪声 5 s', dict(mode='gated', kp=2.0), 1000),
                              ('悲观噪声 10 s', dict(mode='gated', kp=2.0, **PESS), 2000)]:
        with Pool(16) as pool:
            rows = pool.map(_job_closed, [(k, sd, steps, est) for k in ('Py', 'C') for sd in seeds])
        r = compare_closed(rows, label)
        REPORT['D3'][label] = r
        ok &= abs(r['C']['rate'] - r['Py']['rate']) <= 0.5
    return ok


# ============================ E. ESP32-S3 交叉编译 ============================
def section_E():
    cands = sorted(glob.glob(os.path.expanduser('~/.espressif/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin/xtensa-esp32s3-elf-gcc')))
    if not cands:
        REPORT['E'] = dict(skipped='未找到 xtensa-esp32s3-elf-gcc'); print('[E] 跳过：未找到 ESP32-S3 工具链'); return True
    gcc = cands[-1]; bindir = os.path.dirname(gcc)
    src = os.path.join(BUILD_DIR, 'esp_tu.c'); obj = os.path.join(BUILD_DIR, 'esp_tu.o')
    open(src, 'w').write('''#include "recovery_expert9.h"
#include "mahony_gated.h"
recovery9_state_t g_st; mahony_gated_t g_m;
void fc_imu_1khz(const float gyro[3], const float acc[3]) { mahony_gated_update(&g_m, gyro, acc, 0.001f); }
uint8_t fc_ctrl_200hz(const float gyro[3], const float pos[3], const float vel[3], float vz, float ze,
                      const float tgt[3], float vb, float u[4]) {
    return recovery_expert9_step(&g_st, g_m.q, gyro, pos, vel, vz, ze, tgt, vb, u);
}
''')
    cmd = [gcc, '-O2', '-std=c99', '-mlongcalls', '-ffunction-sections', '-Wall', '-Wextra', '-Wdouble-promotion',
           '-Werror=double-promotion', '-I', CODE, '-c', src, '-o', obj]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        REPORT['E'] = dict(error=r.stderr[-800:]); print('[E] 交叉编译失败：', r.stderr[-800:]); return False
    nm = os.path.join(bindir, 'xtensa-esp32s3-elf-nm')
    und = sorted(set(l.split()[-1] for l in subprocess.run([nm, '-u', obj], capture_output=True, text=True).stdout.splitlines() if l.strip()))
    sizes = {}
    for l in subprocess.run([nm, '-S', '--size-sort', obj], capture_output=True, text=True).stdout.splitlines():
        p = l.split()
        if len(p) == 4 and p[2] in 'tT':
            sizes[p[3]] = int(p[1], 16)
    dbl = [s for s in und if (s.startswith('__') and 'df' in s)                 # 双精度软浮点助手：__adddf3 / __extendsfdf2 / __truncdfsf2 ...
           or s in ('sin', 'cos', 'atan2', 'sqrt', 'fabs', 'acos', 'exp', 'pow', 'tan', 'hypot')]
    REPORT['E'] = dict(toolchain=gcc, undefined=und, double_calls=dbl, func_bytes=sizes, warnings=r.stderr.strip())
    tot = sum(v for k, v in sizes.items() if k.startswith('fc_'))
    print(f"[E] ESP32-S3 交叉编译（{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(gcc))))}，-O2）：通过；"
          f"外部调用 {und}；双精度调用 {dbl if dbl else '无'}；代码 {tot} 字节（{', '.join(f'{k} {v} B' for k, v in sizes.items() if k.startswith('fc_'))}）")
    return not dbl


# ============================ F. 主机耗时 ============================
def section_F():
    env = VecExpert9(num_envs=512, seed=2024, episode_s=5.0, params=P, tilt_cap_deg=180.0); env.reset()
    for _ in range(40):
        env.step(np.zeros((512, 4)))
    inp = env_inputs(env)
    ns_c = LIB.bench_exp9(512, inp['q'], inp['w'], inp['pos'], inp['vel'], inp['vz_est'], inp['z_est'], inp['target'], inp['vb'], 400)
    rng = np.random.default_rng(1)
    g = c32(rng.standard_normal((512, 3)) * 5); a = c32(rng.standard_normal((512, 3)) * 3 + [0, 0, 9.81])
    ns_m = LIB.bench_mahony(512, g, a, 400)
    REPORT['F'] = dict(ns_ctrl=ns_c, ns_mahony=ns_m, cpu=subprocess.run(['sh', '-c', "grep -m1 'model name' /proc/cpuinfo"], capture_output=True, text=True).stdout.strip())
    print(f"[F] 主机单步耗时（gcc -O2）：v9 控制器 {ns_c:.0f} ns/次，门控 Mahony {ns_m:.0f} ns/次（{REPORT['F']['cpu'].split(':')[-1].strip()}）")
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); a = ap.parse_args()
    t0 = time.time()
    build()
    res = {}
    res['A'] = section_A()
    res['B'] = section_B([2024] if a.quick else [2024, 42], 1000)
    res['C'] = section_C([2024, 42] if a.quick else [2024, 42, 100, 2026, 7, 123, 999, 31415])
    res['D'] = section_D(a.quick)
    res['E'] = section_E()
    res['F'] = section_F()
    REPORT['pass'] = res; REPORT['elapsed_s'] = time.time() - t0
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'verify_expert9_quick.json' if a.quick else 'verify_expert9.json')
    json.dump(REPORT, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print('\n' + '  '.join(f"{k}:{'✅' if v else '❌'}" for k, v in res.items()) + f"   用时 {time.time() - t0:.0f} s   → {out}")
    sys.exit(0 if all(res.values()) else 1)
