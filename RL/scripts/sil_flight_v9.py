#!/usr/bin/env python3
"""sil_flight_v9.py —— 正式飞控核心的软件在环仿真（Stage 25）

被测对象：code/flight_v9/main/flight_core.h —— 固件 ctrl 任务里跑的同一份 C 代码
          （门控 Mahony + 竖直三阶互补滤波 + 飞行状态机 + v9 控制器），编译成 .so 由本脚本驱动。
仿真侧：VecQuadDynamics（与 Stage 17~21 评测同一套动力学，1 kHz）+ 地面接触（弹簧阻尼 + 摩擦）
        + 传感器模型（陀螺 / 加计噪声与零偏、IMU 安装倾斜、气压计白噪声 + 慢漂移、电池电压）。
场景：
  throw    任意姿态抛出（全球面姿态、ω ≤ 20 rad/s、2.2~3.2 m，与仿真评测同分布）→ 回正 → 悬停 5 s → 降落 → 触地停桨
  drop     手持水平松手（1.5 m、静止）→ 同上（首飞推荐动作）
  takeoff  地面解锁 → takeoff 指令 → 升到 +0.8 m → 悬停 5 s → 降落 → 触地停桨
用法：python3 sil_flight_v9.py [--quick] [--out RL/results/stage25/sil_flight_v9.json]
"""
import os, sys, json, time, ctypes, argparse, subprocess, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'RL', 'env'))
from quad_sim_vec import VecQuadDynamics          # noqa: E402
from yao_kun_params import YaoKunK10Params         # noqa: E402

G = 9.81
FW = os.path.join(ROOT, 'code', 'flight_v9', 'main')
INC = [FW, os.path.join(ROOT, 'code', 'connection_test', 'main')]
ST = ['DISARMED', 'ARMED', 'TAKEOFF', 'RECOVER', 'HOLD', 'LAND']
EV = ["none", "armed", "launch", "takeoff", "stable_hold", "land_auto", "land_cmd", "land_low_voltage", "land_recover_timeout",
      "land_takeoff_timeout", "land_flight_timeout", "landed", "disarm_cmd", "arm_timeout", "land_timeout_kill", "impact",
      "estimator_fail", "refuse_vbatt", "refuse_tilt", "refuse_estimator", "refuse_state"]
CMD = dict(none=0, arm=1, disarm=2, takeoff=3, land=4)

WRAP = r'''
#include <stdlib.h>
#include "flight_core.h"
static fc_t *C = 0;
int sil_create(int n) { free(C); C = (fc_t *)calloc((size_t)n, sizeof(fc_t)); return C != 0; }
void sil_init(int n, const float *a0) { for (int i = 0; i < n; i++) fc_init(&C[i], a0 + 3 * i); }
void sil_set_est(int n, const float *q, const float *z, const float *vz) {
    for (int i = 0; i < n; i++) { for (int j = 0; j < 4; j++) C[i].mg.q[j] = q[4 * i + j];
        C[i].alt.z = z[i]; C[i].alt.vz = vz[i]; C[i].alt.init = 1; } }
void sil_set_alt(int n, const float *z, const float *vz) { for (int i = 0; i < n; i++) { C[i].alt.z = z[i]; C[i].alt.vz = vz[i]; C[i].alt.init = 1; } }
void sil_force_armed(int n) { for (int i = 0; i < n; i++) { C[i].fsm.st = FSM_ARMED; C[i].fsm.t_state = 0; } }
void sil_cmd(int n, const int *cmd) { for (int i = 0; i < n; i++) fc_command(&C[i], (fsm_cmd_t)cmd[i]); }
void sil_step(int n, const float *a, const float *g, float dt, const unsigned char *bn, const float *bz, const float *bdt,
              const float *vb, float *u, unsigned char *st, unsigned char *ev, float *est) {
    for (int i = 0; i < n; i++) {
        fc_t *c = &C[i];
        fc_step(c, a + 3 * i, g + 3 * i, dt, 1, bn[i], bz[i], bdt[i], vb[i]);
        for (int j = 0; j < 4; j++) u[4 * i + j] = c->u_out[j];
        st[i] = (unsigned char)c->fsm.st; ev[i] = c->ev_last;
        float *e = est + 8 * i;
        e[0] = c->mg.q[0]; e[1] = c->mg.q[1]; e[2] = c->mg.q[2]; e[3] = c->mg.q[3];
        e[4] = c->alt.z; e[5] = c->alt.vz; e[6] = c->fsm.z_target; e[7] = c->thrust_frac;
    }
}
'''


def build():
    d = tempfile.mkdtemp(prefix='sil_fc_')
    c = os.path.join(d, 'wrap.c'); so = os.path.join(d, 'libfc.so')
    open(c, 'w').write(WRAP)
    cmd = ['gcc', '-O2', '-shared', '-fPIC', '-std=c99', '-Wall', '-Wno-unused-function'] + sum([['-I', p] for p in INC], []) + [c, '-o', so, '-lm']
    subprocess.check_call(cmd)
    L = ctypes.CDLL(so)
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    L.sil_create.argtypes = [ctypes.c_int]
    L.sil_init.argtypes = [ctypes.c_int, f32]
    L.sil_set_est.argtypes = [ctypes.c_int, f32, f32, f32]
    L.sil_force_armed.argtypes = [ctypes.c_int]
    L.sil_set_alt.argtypes = [ctypes.c_int, f32, f32]
    L.sil_cmd.argtypes = [ctypes.c_int, i32]
    L.sil_step.argtypes = [ctypes.c_int, f32, f32, ctypes.c_float, u8, f32, f32, f32, f32, u8, u8, f32]
    return L


def qmul(p, q):
    w1, x1, y1, z1 = p.T; w2, x2, y2, z2 = q.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)


def qexp(rv):
    ang = np.linalg.norm(rv, axis=1); ax = rv / np.maximum(ang, 1e-12)[:, None]
    return np.concatenate([np.cos(ang / 2)[:, None], ax * np.sin(ang / 2)[:, None]], axis=1)


def rotmat(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)], 1)   # (N,3,3) body→world


def random_quats(n, rng):
    u = rng.random((n, 3))
    return np.stack([np.sqrt(1 - u[:, 0]) * np.sin(2 * np.pi * u[:, 1]), np.sqrt(1 - u[:, 0]) * np.cos(2 * np.pi * u[:, 1]),
                     np.sqrt(u[:, 0]) * np.sin(2 * np.pi * u[:, 2]), np.sqrt(u[:, 0]) * np.cos(2 * np.pi * u[:, 2])], 1)[:, [3, 0, 1, 2]]


NOISE = {
    # 陀螺 σ / 零偏残差 rad/s；加计 σ / 零偏残差 m/s²；IMU 安装倾斜 °；气压计白噪 / 慢漂幅度 m；电机增益散布
    # 噪声 σ 为电机满转时的振动水平，按转速比例缩放；静止噪声取真机实测（加计 ~2 mg、陀螺 ~0.1 °/s）
    'nominal':     dict(g_sig=0.03, g_bias=0.005, a_sig=0.5, a_bias=0.05, mount_deg=0.0, b_sig=0.08, b_drift=0.20, mgain=0.0),
    'pessimistic': dict(g_sig=0.05, g_bias=0.010, a_sig=1.0, a_bias=0.10, mount_deg=1.5, b_sig=0.12, b_drift=0.35, mgain=0.08),
}


def run(L, scen, noise, N, seed, T=30.0, mount_deg=None):
    rng = np.random.default_rng(seed)
    nz = dict(NOISE[noise]);
    if mount_deg is not None:
        nz['mount_deg'] = mount_deg
    p = YaoKunK10Params()
    mg = None if nz['mgain'] == 0 else rng.uniform(1 - nz['mgain'], 1 + nz['mgain'], (N, 4))
    dyn = VecQuadDynamics(N, p, motor_gain=mg)
    dt = dyn.dt
    # ---------------- 初始状态 ----------------
    if scen == 'throw':
        quat = random_quats(N, rng)
        wdir = rng.normal(size=(N, 3)); wdir /= np.linalg.norm(wdir, axis=1, keepdims=True)
        omega = wdir * rng.uniform(0, 20.0, (N, 1))
        vel = np.column_stack([rng.uniform(-1, 1, N), rng.uniform(-1, 1, N), rng.uniform(-1.5, 0, N)])
        pos = np.column_stack([np.zeros(N), np.zeros(N), rng.uniform(2.2, 3.2, N)])
    elif scen == 'drop':
        quat = np.tile([1.0, 0, 0, 0], (N, 1)); omega = np.zeros((N, 3)); vel = np.zeros((N, 3))
        pos = np.column_stack([np.zeros(N), np.zeros(N), np.full(N, 1.5)])
    else:  # takeoff
        quat = np.tile([1.0, 0, 0, 0], (N, 1)); omega = np.zeros((N, 3)); vel = np.zeros((N, 3))
        pos = np.column_stack([np.zeros(N), np.zeros(N), np.full(N, -0.08 * G / 800.0)])   # 静置在地面（弹簧压缩量）
    dyn.set_state(pos, vel, quat, omega, soc=1.0)
    # ---------------- 传感器参数 ----------------
    gb = rng.normal(0, nz['g_bias'], (N, 3)); ab = rng.normal(0, nz['a_bias'], (N, 3))
    mrv = rng.normal(size=(N, 3)); mrv[:, 2] = 0; mrv /= np.linalg.norm(mrv, axis=1, keepdims=True) + 1e-12
    Rm = rotmat(qexp(mrv * np.deg2rad(nz['mount_deg'])))                       # IMU 安装倾斜（固件不知道）
    b_ph = rng.uniform(0, 2 * np.pi, (N, 2)); b_T = np.column_stack([rng.uniform(8, 20, N), rng.uniform(30, 60, N)])
    b_off = rng.normal(0, 0.05, N)

    def baro_err(t):
        return b_off + nz['b_drift'] * (0.6 * np.sin(2 * np.pi * t / b_T[:, 0] + b_ph[:, 0]) + 0.4 * np.sin(2 * np.pi * t / b_T[:, 1] + b_ph[:, 1])) \
            + rng.normal(0, nz['b_sig'], N)

    def sense(t):
        R = rotmat(dyn.quat)
        fw = dyn.last_acc + np.array([0, 0, G])
        fb = np.einsum('nji,nj->ni', R, fw)
        vib = (dyn.rotor_w.mean(axis=1) / dyn.wm)[:, None]                  # 0 = 电机停，1 = 满转
        fi = np.einsum('nji,nj->ni', Rm, fb) + ab + rng.normal(0, 1, (N, 3)) * (0.02 + nz['a_sig'] * vib)
        wi = np.einsum('nji,nj->ni', Rm, dyn.w) + gb + rng.normal(0, 1, (N, 3)) * (0.002 + nz['g_sig'] * vib)
        return np.ascontiguousarray(fi, np.float32), np.ascontiguousarray(wi, np.float32)

    # ---------------- 核心初始化 ----------------
    L.sil_create(N)
    a0 = np.einsum('nji,nj->ni', rotmat(dyn.quat), np.tile([0, 0, G], (N, 1)))   # 静止时的比力 → 经安装倾斜与零偏后为加计读数
    a0m = np.einsum('nji,nj->ni', Rm, a0) + ab
    L.sil_init(N, np.ascontiguousarray(a0m, np.float32))                          # 手持 / 地面静止：和固件一样由加计初始化姿态
    z0 = pos[:, 2] + baro_err(0.0)
    if scen == 'throw':
        rv = rng.normal(0, np.deg2rad(2.0), (N, 3)); rv[:, 2] = 0      # 抛出过程中估计器一直在跑：初始倾角误差 σ 2°
        qe = qmul(qexp(rv), dyn.quat)
        L.sil_set_est(N, np.ascontiguousarray(qe, np.float32), np.ascontiguousarray(z0, np.float32),
                      np.ascontiguousarray(vel[:, 2] + rng.normal(0, 0.1, N), np.float32))
        L.sil_force_armed(N)
    elif scen == 'drop':
        L.sil_set_alt(N, np.ascontiguousarray(z0, np.float32), np.zeros(N, np.float32))
        L.sil_force_armed(N)
    else:
        L.sil_set_alt(N, np.ascontiguousarray(z0, np.float32), np.zeros(N, np.float32))
    # ---------------- 仿真 ----------------
    K = int(T / dt)
    u = np.zeros((N, 4), np.float32); st = np.zeros(N, np.uint8); ev = np.zeros(N, np.uint8); est = np.zeros((N, 8), np.float32)
    bn = np.zeros(N, np.uint8); bz = np.zeros(N, np.float32); bdt = np.full(N, 0.02, np.float32); vb = np.full(N, np.nan, np.float32)
    kg, cg, mu = 800.0, 8.0, 20.0
    # 记录
    t_on = np.full(N, np.nan); t_hold = np.full(N, np.nan); t_land = np.full(N, np.nan); t_off = np.full(N, np.nan)
    ev_off = np.zeros(N, np.int64); z_off = np.full(N, np.nan); xy0 = np.full((N, 2), np.nan)
    first_contact = np.full(N, np.nan); c_vz = np.full(N, np.nan); c_tilt = np.full(N, np.nan); c_state = np.full(N, -1)
    zmin = np.full(N, np.inf); vxy_hold_max = np.zeros(N); xy_land = np.full((N, 2), np.nan); ztrue_hold = [[] for _ in range(N)]
    prev_st = np.full(N, 1 if scen != 'takeoff' else 0); airborne = np.zeros(N, bool)
    estz_err = []
    for k in range(K):
        t = k * dt
        if scen == 'takeoff':
            if k == 300:
                L.sil_cmd(N, np.full(N, CMD['arm'], np.int32))
            if k == 500:
                L.sil_cmd(N, np.full(N, CMD['takeoff'], np.int32))
        dyn.step(u.astype(np.float64))
        # 地面：弹簧阻尼 + 摩擦
        pen = -dyn.pos[:, 2]; m = pen > 0
        if m.any():
            F = np.maximum(kg * pen[m] - cg * dyn.vel[m, 2], 0.0); acc = F / dyn.mass[m]
            newc = m & np.isnan(first_contact) & airborne
            if newc.any():
                tilt = np.degrees(np.arccos(np.clip(1 - 2 * (dyn.quat[newc, 1] ** 2 + dyn.quat[newc, 2] ** 2), -1, 1)))
                first_contact[newc] = t; c_vz[newc] = dyn.vel[newc, 2]; c_tilt[newc] = tilt; c_state[newc] = st[newc]
            dyn.vel[m, 2] += acc * dt; dyn.last_acc[m, 2] += acc
            fr = np.exp(-mu * dt); dyn.vel[m, 0] *= fr; dyn.vel[m, 1] *= fr; dyn.w[m] *= np.exp(-30 * dt)
        fi, wi = sense(t)
        if k % 20 == 0:
            bn[:] = 1; bz[:] = (dyn.pos[:, 2] + baro_err(t)).astype(np.float32)
        else:
            bn[:] = 0
        vb[:] = (dyn.v_batt + rng.normal(0, 0.01, N)).astype(np.float32) if k % 100 == 0 else np.nan
        L.sil_step(N, fi, wi, dt, bn, bz, bdt, vb, u, st, ev, est)
        # ---- 记录 ----
        on = st >= 2
        airborne |= on & (dyn.pos[:, 2] > 0.05)
        newon = on & np.isnan(t_on); t_on[newon] = t; xy0[newon] = dyn.pos[newon, :2]
        nh = (st == 4) & np.isnan(t_hold); t_hold[nh] = t
        nl = (st == 5) & np.isnan(t_land); t_land[nl] = t
        off = (~on) & (prev_st >= 2) & np.isnan(t_off)
        if off.any():
            t_off[off] = t; ev_off[off] = ev[off]; z_off[off] = dyn.pos[off, 2]; xy_land[off] = dyn.pos[off, :2]
        fl = on | (~np.isnan(t_on) & np.isnan(t_off))
        zmin = np.where(fl, np.minimum(zmin, dyn.pos[:, 2]), zmin)
        hold = st == 4
        vxy_hold_max = np.where(hold, np.maximum(vxy_hold_max, np.linalg.norm(dyn.vel[:, :2], axis=1)), vxy_hold_max)
        if k % 50 == 0 and hold.any():
            estz_err.append(np.abs(est[hold, 4] - (dyn.pos[hold, 2] + 0)).mean())
        prev_st = st.astype(np.int64).copy()
        if k % 1000 == 0 and np.all(~np.isnan(t_off) | (st <= 1) & (k > 2000)):
            if np.all(~np.isnan(t_off)):
                break
    # ---------------- 判定 ----------------
    landed_ev = np.isin(ev_off, [EV.index('landed'), EV.index('impact')])
    gentle = (c_state == 5) & (np.abs(c_vz) < 1.0) & (c_tilt < 20)
    ok = landed_ev & ~np.isnan(t_hold) & gentle & (z_off < 0.05)
    air_kill = ~np.isnan(t_off) & (z_off > 0.10)
    early_ground = ~np.isnan(first_contact) & (c_state != 5)
    crash = early_ground | (~np.isnan(first_contact) & ((np.abs(c_vz) >= 1.0) | (c_tilt >= 20)))
    never_on = np.isnan(t_on)
    never_off = ~never_on & np.isnan(t_off)
    drift = np.linalg.norm(xy_land - xy0, axis=1)
    det_delay = t_off - first_contact
    def pct(x, q):
        x = x[np.isfinite(x)]; return float(np.percentile(x, q)) if len(x) else float('nan')
    ev_hist = {}
    for e in ev_off[~np.isnan(t_off)]:
        ev_hist[EV[e]] = ev_hist.get(EV[e], 0) + 1
    res = dict(scenario=scen, noise=noise, mount_deg=nz['mount_deg'], N=N, seed=seed,
               ok_rate=float(ok.mean()), hold_reached=float((~np.isnan(t_hold)).mean()),
               crash_rate=float(crash.mean()), early_ground_rate=float(early_ground.mean()),
               air_kill_rate=float(air_kill.mean()), never_off_rate=float(never_off.mean()), never_on_rate=float(never_on.mean()),
               t_hold_med=pct(t_hold - t_on, 50), t_hold_p90=pct(t_hold - t_on, 90),
               zmin_med=pct(np.where(np.isfinite(zmin), zmin, np.nan), 50), zmin_p10=pct(np.where(np.isfinite(zmin), zmin, np.nan), 10),
               touch_vz_med=pct(np.where(c_state == 5, c_vz, np.nan), 50), touch_vz_p10=pct(np.where(c_state == 5, c_vz, np.nan), 10),
               detect_delay_med=pct(np.where(c_state == 5, det_delay, np.nan), 50), detect_delay_p90=pct(np.where(c_state == 5, det_delay, np.nan), 90),
               flight_time_med=pct(t_off - t_on, 50), drift_med=pct(drift, 50), drift_p90=pct(drift, 90),
               vxy_hold_max_med=pct(vxy_hold_max, 50), vxy_hold_max_p90=pct(vxy_hold_max, 90),
               hold_z_est_err_mean=float(np.mean(estz_err)) if estz_err else float('nan'), off_events=ev_hist)
    return res


def show(r):
    print(f"  {r['scenario']:<8} {r['noise']:<11} 安装倾斜 {r['mount_deg']:.1f}°  N={r['N']}")
    print(f"    完整成功（回正→悬停 5 s→降落→触地停桨）{r['ok_rate'] * 100:5.1f}%   进入悬停 {r['hold_reached'] * 100:5.1f}%   "
          f"坠地/硬着陆 {r['crash_rate'] * 100:4.1f}%（降落前触地 {r['early_ground_rate'] * 100:4.1f}%）   空中停桨 {r['air_kill_rate'] * 100:4.1f}%   未停桨 {r['never_off_rate'] * 100:4.1f}%   没起飞 {r['never_on_rate'] * 100:4.1f}%")
    print(f"    进入悬停 中位 {r['t_hold_med']:.2f} s / P90 {r['t_hold_p90']:.2f} s   最低高度 中位 {r['zmin_med']:.2f} m / P10 {r['zmin_p10']:.2f} m   "
          f"触地速度 中位 {r['touch_vz_med']:+.2f} / P10 {r['touch_vz_p10']:+.2f} m/s   触地→停桨 中位 {r['detect_delay_med']:.2f} s / P90 {r['detect_delay_p90']:.2f} s")
    print(f"    全程 中位 {r['flight_time_med']:.1f} s   水平漂移 中位 {r['drift_med']:.2f} m / P90 {r['drift_p90']:.2f} m   悬停时最大水平速度 中位 {r['vxy_hold_max_med']:.2f} / P90 {r['vxy_hold_max_p90']:.2f} m/s")
    print(f"    停桨原因 {r['off_events']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'RL', 'results', 'stage25', 'sil_flight_v9.json'))
    a = ap.parse_args()
    L = build()
    N = 128 if a.quick else 512
    runs = [('throw', 'nominal', None), ('throw', 'pessimistic', None), ('drop', 'nominal', None), ('drop', 'pessimistic', None),
            ('takeoff', 'nominal', None), ('takeoff', 'pessimistic', None),
            ('drop', 'nominal', 0.0), ('drop', 'nominal', 1.5), ('drop', 'nominal', 3.0)]
    if a.quick:
        runs = runs[:1] + runs[2:3] + runs[4:5]
    out = []
    for i, (sc, nzn, md) in enumerate(runs):
        t0 = time.time()
        n = N if sc == 'throw' else max(64, N // 2)
        r = run(L, sc, nzn, n, seed=100 + i, mount_deg=md)
        r['wall_s'] = round(time.time() - t0, 1)
        show(r); out.append(r); sys.stdout.flush()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print('→', a.out)


if __name__ == '__main__':
    main()
