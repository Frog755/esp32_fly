#!/usr/bin/env python3
"""visualize_v8_v9.py —— v8 vs v9 同工况并排 3D 对比（Stage 19）

左（红）= v8，右（青）= v9。两机来自同一 seed、同一通道 → 初始姿态 / 角速度 / 高度 / 气压计偏置完全相同，
逐步结果与 eval_recovery_metrics.py 一致（--check 可核对）。
每架飞机额外绘制：机体 z 轴（推力方向）箭头、飞行轨迹尾迹、最低点高度标记盘。

用法：
  python3 visualize_v8_v9.py                       # 交互 3D 查看器（需桌面显示）
  python3 visualize_v8_v9.py --case 249            # 交互查看器从指定通道开始
  python3 visualize_v8_v9.py --export out.mp4      # 离线导出对比视频（EGL 无窗口渲染，H.264）
  python3 visualize_v8_v9.py --plot out.png        # 导出精选工况高度曲线对比图
  python3 visualize_v8_v9.py --check               # 核对精选工况指标与评测脚本一致

交互快捷键：
  [1]~[6] 精选工况   [N] 下一个精选   [B] 随机通道   [空格] 暂停/继续   [R] 重播
  [S] 1x / 0.25x 慢放   [Q / ESC] 退出
  鼠标：左键拖动旋转，右键拖动平移，滚轮缩放（与 MuJoCo 原生查看器一致）
"""
import os, sys, time, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
sys.path.insert(0, HERE)
import numpy as np

SEED = 2024
N_ENVS = 512
DT = 0.005
STEPS = 999                      # 与评测一致：第 1000 步回合超时会自动重置，只看前 999 步
OFF = {'v8': -0.4, 'v9': 0.4}    # 显示时的横向偏移（与场景里两个目标靶标位置一致）
SLOT = {'v8': 'v5', 'v9': 'v6'}  # 复用 visualize_recovery 场景中的两架机体：左槽（红）/ 右槽（青）
RGBA = {'v8': (0.95, 0.30, 0.25), 'v9': (0.20, 0.85, 0.95)}
C255 = {k: tuple(int(c * 255) for c in v) for k, v in RGBA.items()}
SIDE = {'v8': '左·红', 'v9': '右·青'}

CASES = [
    dict(idx=85,  note='深倒扣，几乎无初始转速'),
    dict(idx=249, note='偏航高速自旋，v8 坠毁'),
    dict(idx=241, note='倒扣 + 中等转速，掉高差最大'),
    dict(idx=146, note='近完全倒扣 + 高转速，v8 坠毁'),
    dict(idx=48,  note='高转速翻滚'),
    dict(idx=396, note='倒扣 + 偏航自旋 19 rad/s，两者都坠毁（v9 剩余失败）'),
]

FONTS = [('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 2),
         ('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc', 2),
         ('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 2),
         ('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', 0)]
BOLD = ('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 2)


def body_z(q):
    qw, qx, qy, qz = q
    return np.array([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)])


# ============================ 双机仿真 ============================
class Track:
    def __init__(self, pos, quat):
        self.pos = pos.copy(); self.quat = quat.copy(); self.u = np.zeros(4)
        self.z0 = float(pos[2]); self.zmin = self.z0; self.t_zmin = 0.0
        self.crashed = False; self.t_crash = None
        self.t_up = None; self.t_stable = None
        self.trail = [pos.copy()]; self.zs = []

    @property
    def drop(self):
        return self.z0 - self.zmin

    def status(self):
        if self.crashed:
            return f'坠毁 @ {self.t_crash:.2f} s'
        if self.t_stable is not None:
            return '稳定悬停'
        return '已翻正，回升中' if self.t_up is not None else '翻转中'


class Pair:
    def __init__(self):
        from yao_kun_params import YaoKunK10Params
        from vec_expert8_env import VecExpert8
        from vec_expert9_env import VecExpert9
        self.envs = {'v8': VecExpert8(num_envs=N_ENVS, seed=SEED, params=YaoKunK10Params(), tilt_cap_deg=180.0),
                     'v9': VecExpert9(num_envs=N_ENVS, seed=SEED, params=YaoKunK10Params(), tilt_cap_deg=180.0)}
        for e in self.envs.values():           # 截获每步实际油门（用于桨叶转速显示）
            orig = e._action_to_throttle

            def hook(a, _o=orig, _e=e):
                u = _o(a); _e._last_u = u
                return u
            e._action_to_throttle = hook
        self.reset(CASES[0]['idx'])

    def reset(self, idx):
        self.idx = int(idx); self.k = 0; self.tr = {}
        for name, e in self.envs.items():
            e.seed(SEED)                        # 每次重播都重新播种 → 初态与评测完全相同
            e.reset()
            self.tr[name] = Track(e.dyn.pos[self.idx], e.dyn.quat[self.idx])
        q = self.envs['v8'].dyn.quat[self.idx]; w = self.envs['v8'].dyn.w[self.idx]
        self.tilt0 = float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))
        self.w0 = float(np.linalg.norm(w)); self.wz0 = float(w[2])

    @property
    def t(self):
        return self.k * DT

    @property
    def done(self):
        return self.k >= STEPS or all(tr.crashed for tr in self.tr.values())

    def step(self):
        if self.k >= STEPS:
            return
        self.k += 1
        t = self.t; i = self.idx
        for name, e in self.envs.items():
            tr = self.tr[name]
            if tr.crashed:
                continue
            e.step(np.zeros((N_ENVS, 4)))
            if e._k[i] == 0:                    # 该通道被自动重置 → 首回合已坠地（z < -0.5）
                tr.crashed = True; tr.t_crash = t
                tr.pos = np.array([tr.pos[0], tr.pos[1], -0.5]); tr.zmin = -0.5
                tr.zs.append(-0.5); tr.trail.append(tr.pos.copy())
                continue
            s = e.dyn
            tr.pos = s.pos[i].copy(); tr.quat = s.quat[i].copy(); tr.u = np.asarray(e._last_u[i]).copy()
            z = float(tr.pos[2]); tr.zs.append(z)
            if z < tr.zmin:
                tr.zmin = z; tr.t_zmin = t
            upz = 1 - 2 * (tr.quat[1] ** 2 + tr.quat[2] ** 2)
            if tr.t_up is None and upz > 0.9 and np.hypot(s.w[i, 0], s.w[i, 1]) < 3.0:
                tr.t_up = t
            if tr.t_stable is None and e._stable_cnt[i] >= 50:
                tr.t_stable = t - 49 * DT
            if self.k % 4 == 0:
                tr.trail.append(tr.pos.copy())

    def run_to_end(self):
        while not self.done:
            self.step()

    def desc(self):
        return f'初始倾角 {self.tilt0:.0f}°   |ω| {self.w0:.1f} rad/s（偏航 {self.wz0:+.1f}）   初始高度 {self.tr["v8"].z0:.2f} m'

    def desc_en(self):
        return f'tilt0 {self.tilt0:.0f} deg  |w0| {self.w0:.1f} rad/s (yaw {self.wz0:+.1f})  z0 {self.tr["v8"].z0:.2f} m'


def case_note(idx):
    for c in CASES:
        if c['idx'] == idx:
            return c['note']
    return '随机通道'


# ============================ 场景与叠加图形 ============================
def scene_ids(model):
    ids = {}
    for slot in ('v5', 'v6'):
        ids[slot] = dict(root=model.jnt_qposadr[model.joint(f'{slot}_root').id],
                         prop=model.jnt_qposadr[model.joint(f'{slot}_p0').id])
    return ids


def disp_pos(pair, name):
    p = pair.tr[name].pos
    return np.array([OFF[name] + p[0], p[1], max(-0.48, p[2])])


def apply_state(mujoco, model, data, pair, ids, props):
    for name in ('v8', 'v9'):
        tr = pair.tr[name]; sl = ids[SLOT[name]]
        data.qpos[sl['root']:sl['root'] + 3] = disp_pos(pair, name)
        data.qpos[sl['root'] + 3:sl['root'] + 7] = tr.quat
        if not tr.crashed:
            props[name] += np.clip(tr.u, 0, 1) * 0.9
        data.qpos[sl['prop']:sl['prop'] + 4] = props[name]
    mujoco.mj_forward(model, data)


def _add(mujoco, scn, gtype, size, pos, rgba):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, np.float64), np.asarray(pos, np.float64),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def _arrow(mujoco, scn, p0, p1, width, rgba):
    g = _add(mujoco, scn, mujoco.mjtGeom.mjGEOM_ARROW, [0, 0, 0], [0, 0, 0], rgba)
    if g is None:
        return
    p0 = np.asarray(p0, np.float64); p1 = np.asarray(p1, np.float64)
    if hasattr(mujoco, 'mjv_connector'):
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, p0, p1)
    else:
        mujoco.mjv_makeConnector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, *p0, *p1)


def draw_overlays(mujoco, scn, pair, arrow_len=0.45):
    G = mujoco.mjtGeom
    for name in ('v8', 'v9'):
        tr = pair.tr[name]; c = RGBA[name]; off = np.array([OFF[name], 0.0, 0.0])
        for q in tr.trail[-260:]:
            _add(mujoco, scn, G.mjGEOM_SPHERE, [0.009, 0, 0], q + off, (*c, 0.55))
        _add(mujoco, scn, G.mjGEOM_CYLINDER, [0.16, 0.003, 0], [OFF[name], 0.0, tr.z0], (0.85, 0.85, 0.85, 0.25))
        _add(mujoco, scn, G.mjGEOM_CYLINDER, [0.16, 0.005, 0], [OFF[name], 0.0, tr.zmin], (*c, 0.75))
        p = disp_pos(pair, name)
        if tr.crashed:
            _add(mujoco, scn, G.mjGEOM_SPHERE, [0.06, 0, 0], p, (1.0, 0.1, 0.1, 0.6))
        else:
            _arrow(mujoco, scn, p, p + arrow_len * body_z(tr.quat), 0.016, (*c, 1.0))


# ============================ 交互查看器（预计算 + 回放） ============================
# 卡顿根因：每一帧都要把 v8 / v9 两个 512 通道批量环境各推进一步（7.3 ms/步，上限 0.68x 实时），
# 且大量 Python 计算一直占着 GIL，查看器的事件/按键处理被拖住，看起来像卡死。
# 现在：轨迹在后台进程里整段算好（每个工况约 8 s，6 个精选工况并行），查看器进程只按时间索引回放，
# 每帧开销 < 1 ms；数值与评测完全一致（回放的就是同一段仿真记录）。
_WORKER_PAIR = None


def _init_worker():
    global _WORKER_PAIR
    import warnings
    warnings.filterwarnings('ignore')
    _WORKER_PAIR = Pair()


def _sim_worker(idx):
    pair = _WORKER_PAIR
    pair.reset(idx)
    names = ('v8', 'v9')
    rec = {n: dict(pos=[], quat=[], u=[]) for n in names}

    def snap():
        for n in names:
            tr = pair.tr[n]
            rec[n]['pos'].append(tr.pos.copy()); rec[n]['quat'].append(tr.quat.copy()); rec[n]['u'].append(np.asarray(tr.u, float).copy())
    snap()
    while not pair.done:
        pair.step(); snap()
    out = dict(idx=int(idx), K=int(pair.k), tilt0=pair.tilt0, w0=pair.w0, wz0=pair.wz0,
               desc=pair.desc(), desc_en=pair.desc_en(), rec={}, meta={})
    for n in names:
        tr = pair.tr[n]
        out['rec'][n] = {k: np.asarray(v) for k, v in rec[n].items()}
        out['meta'][n] = dict(z0=tr.z0, t_up=tr.t_up, t_stable=tr.t_stable, t_crash=tr.t_crash)
    return out


class Replay:
    """与 Pair 同接口的回放对象（apply_state / draw_overlays 可直接使用）。"""

    def __init__(self, res):
        self.res = res; self.idx = res['idx']; self.K = res['K']; self.k = 0
        self.tilt0, self.w0, self.wz0 = res['tilt0'], res['w0'], res['wz0']
        self.tr = {}; self._zmin = {}
        for n in ('v8', 'v9'):
            r = res['rec'][n]
            self.tr[n] = Track(r['pos'][0], r['quat'][0])
            self._zmin[n] = np.minimum.accumulate(r['pos'][:, 2])
        self.seek(0)

    @property
    def t(self):
        return self.k * DT

    @property
    def done(self):
        return self.k >= self.K

    def desc(self):
        return self.res['desc']

    def desc_en(self):
        return self.res['desc_en']

    def seek(self, k):
        self.k = int(min(max(k, 0), self.K)); k = self.k; t = k * DT; eps = 1e-9
        for n, tr in self.tr.items():
            r = self.res['rec'][n]; m = self.res['meta'][n]
            tr.pos = r['pos'][k]; tr.quat = r['quat'][k]; tr.u = r['u'][k]
            tr.z0 = m['z0']; tr.zmin = float(self._zmin[n][k])
            tr.t_crash = m['t_crash']
            tr.crashed = m['t_crash'] is not None and t >= m['t_crash'] - eps
            tr.t_up = m['t_up'] if (m['t_up'] is not None and t >= m['t_up'] - eps) else None
            ts = m['t_stable']
            tr.t_stable = ts if (ts is not None and t >= ts + 49 * DT - eps) else None
            tr.trail = r['pos'][0:k + 1:4]
            tr.zs = r['pos'][1:k + 1, 2]


def run_live(args):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    # 先 fork 计算进程，再初始化 OpenGL / 查看器（避免 fork 继承 GL 上下文）
    pool = ProcessPoolExecutor(max_workers=min(len(CASES), os.cpu_count() or 2),
                               mp_context=mp.get_context('fork'), initializer=_init_worker)
    futs = {}

    def request(idx):
        if idx not in futs:
            futs[idx] = pool.submit(_sim_worker, int(idx))

    start = args.case if args.case is not None else CASES[0]['idx']
    request(start)
    for c in CASES:
        request(c['idx'])

    import mujoco, mujoco.viewer
    from visualize_recovery import MUJOCO_XML
    model = mujoco.MjModel.from_xml_string(MUJOCO_XML); data = mujoco.MjData(model)
    ids = scene_ids(model)
    ci0 = next((i for i, c in enumerate(CASES) if c['idx'] == start), 0)
    st = dict(ci=ci0, idx=start, paused=False, speed=1.0, reset=True, quit=False)

    def pick(ci):
        st['ci'] = ci % len(CASES); st['idx'] = CASES[st['ci']]['idx']; st['reset'] = True

    def on_key(kc):
        ch = chr(kc).upper() if 32 <= kc <= 126 else ''
        if kc == 32:
            st['paused'] = not st['paused']
        elif ch and ch in '123456':
            pick(int(ch) - 1)
        elif ch == 'N':
            pick(st['ci'] + 1)
        elif ch == 'B':
            st['idx'] = int(np.random.randint(0, N_ENVS)); st['reset'] = True
        elif ch == 'R':
            st['reset'] = True
        elif ch == 'S':
            st['speed'] = 0.25 if st['speed'] > 0.5 else 1.0
        elif ch == 'Q' or kc == 256:
            st['quit'] = True

    print(__doc__, flush=True)
    props = {'v8': np.zeros(4), 'v9': np.zeros(4)}
    rep = None; pending = None; hold_until = None; sim_t = 0.0
    frames = 0; fps_t0 = time.time()
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as v:
        v.cam.lookat[:] = [0.0, 0.0, 1.2]; v.cam.distance = 4.6; v.cam.azimuth = 90; v.cam.elevation = -6
        last = time.time()
        while v.is_running() and not st['quit']:
            now = time.time(); dt_wall = now - last; last = now
            if st['reset']:
                st['reset'] = False
                if rep is not None and rep.idx == st['idx']:
                    sim_t = 0.0; rep.seek(0); hold_until = None; props = {'v8': np.zeros(4), 'v9': np.zeros(4)}
                else:
                    pending = st['idx']; request(pending); rep = None
            if rep is None and pending is not None and futs[pending].done():
                try:
                    rep = Replay(futs[pending].result())
                    print(f'\n=== 通道 #{rep.idx}：{case_note(rep.idx)} ===\n    {rep.desc()}', flush=True)
                except Exception as e:
                    print(f'\n通道 #{pending} 计算失败：{e!r}', flush=True)
                pending = None; sim_t = 0.0; hold_until = None; props = {'v8': np.zeros(4), 'v9': np.zeros(4)}
            if rep is not None and not st['paused'] and hold_until is None:
                sim_t += dt_wall * st['speed']
                rep.seek(int(sim_t / DT))
                if rep.done:
                    hold_until = now + 2.5
                    for nm in ('v8', 'v9'):
                        tr = rep.tr[nm]
                        print(f'    {nm}：掉高 {tr.drop:.2f} m，翻正 {"—" if tr.t_up is None else f"{tr.t_up:.2f} s"}，'
                              f'稳定悬停 {"—" if tr.t_stable is None else f"{tr.t_stable:.2f} s"}，{tr.status()}', flush=True)
            if hold_until is not None and now > hold_until:       # 播完停 2.5 s：精选工况自动切到下一个，随机通道原地重播
                hold_until = None
                if any(c['idx'] == rep.idx for c in CASES):
                    pick(st['ci'] + 1)
                else:
                    sim_t = 0.0; rep.seek(0); props = {'v8': np.zeros(4), 'v9': np.zeros(4)}
            with v.lock():
                v.user_scn.ngeom = 0
                if rep is not None:
                    apply_state(mujoco, model, data, rep, ids, props)
                    draw_overlays(mujoco, v.user_scn, rep)
                if hasattr(v, 'set_texts'):
                    try:
                        if rep is None:
                            txt = (f'Computing case #{pending} ... (~10 s)', 'left red = v8 | right cyan = v9')
                        else:
                            lines = []
                            for nm in ('v8', 'v9'):
                                tr = rep.tr[nm]
                                ss = 'CRASH' if tr.crashed else ('hover %.2fs' % tr.t_stable if tr.t_stable is not None else '...')
                                lines.append(f'{nm}: z {max(tr.pos[2], -0.5):+.2f}m  drop {tr.drop:.2f}m  {ss}')
                            txt = (f'Case #{rep.idx}  t={rep.t:.2f}s  {st["speed"]}x' + ('  PAUSED' if st['paused'] else ''),
                                   rep.desc_en() + '\nleft red = v8 | right cyan = v9\n' + '\n'.join(lines))
                        v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, *txt)])
                    except Exception:
                        pass
            v.sync()
            frames += 1
            if now - fps_t0 > 10.0:
                print(f'[查看器] 主循环 {frames / (now - fps_t0):.0f} fps', flush=True)
                frames = 0; fps_t0 = now
            time.sleep(0.012)
    pool.shutdown(wait=False, cancel_futures=True)
    os._exit(0)


# ============================ 视频导出 ============================
def _font(size, bold=False):
    from PIL import ImageFont
    for path, ix in ([BOLD] if bold else []) + FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=ix)
            except Exception:
                pass
    return ImageFont.load_default()


def _hud(img3d, closeups, pair, note, speed, W, H, fonts, banner=None):
    from PIL import Image, ImageDraw
    f_t, f_m, f_s, f_b = fonts
    frame = Image.new('RGB', (W, H), (12, 15, 21))
    frame.paste(Image.fromarray(img3d), (0, 0))
    d = ImageDraw.Draw(frame, 'RGBA')
    h3 = img3d.shape[0]
    # 顶部标题条
    d.rectangle([0, 0, W, 58], fill=(8, 10, 16, 190))
    d.text((16, 6), f'通道 #{pair.idx}', font=f_t, fill=(240, 244, 250))
    d.text((16, 34), f'{note}   ·   {pair.desc()}', font=f_s, fill=(175, 188, 205))
    d.text((W - 16, 10), f't = {pair.t:4.2f} s', font=f_t, fill=(240, 244, 250), anchor='ra')
    d.text((W - 16, 36), f'{speed:g}× 播放', font=f_s, fill=(175, 188, 205), anchor='ra')
    d.text((W // 2 - 12, 6), '● v8（左·红）', font=f_m, fill=C255['v8'], anchor='ra')
    d.text((W // 2 + 12, 6), '● v9（右·青）', font=f_m, fill=C255['v9'], anchor='la')
    d.text((16, h3 - 26), '箭头 = 推力方向（机体 z 轴）   彩色圆盘 = 最低点   灰色圆盘 = 初始高度', font=f_s,
           fill=(185, 195, 210), anchor='la')
    if banner:
        d.rectangle([0, h3 // 2 - 40, W, h3 // 2 + 40], fill=(8, 10, 16, 200))
        d.text((W // 2, h3 // 2), banner, font=f_b, fill=(250, 250, 250), anchor='mm')
    # 底部仪表区
    y0 = h3; ch = H - h3
    cw = closeups['v8'].shape[1]
    frame.paste(Image.fromarray(closeups['v8']), (0, y0))
    frame.paste(Image.fromarray(closeups['v9']), (W - cw, y0))
    d.text((6, y0 + 4), 'v8 近景', font=f_s, fill=C255['v8'])
    d.text((W - cw + 6, y0 + 4), 'v9 近景', font=f_s, fill=C255['v9'])
    sw = 270
    for name, x in (('v8', cw + 14), ('v9', W - cw - sw + 4)):
        tr = pair.tr[name]
        d.text((x, y0 + 8), f'{name}（{SIDE[name]}）', font=f_m, fill=C255[name])
        rows = [('高度', f'{max(tr.pos[2], -0.5):+.2f} m'), ('掉高', f'{tr.drop:.2f} m'),
                ('翻正', '—' if tr.t_up is None else f'{tr.t_up:.2f} s'),
                ('稳定悬停', '—' if tr.t_stable is None else f'{tr.t_stable:.2f} s')]
        for j, (k_, v_) in enumerate(rows):
            d.text((x, y0 + 40 + j * 25), k_, font=f_s, fill=(160, 172, 190))
            d.text((x + 100, y0 + 40 + j * 25), v_, font=f_s, fill=(235, 240, 248))
        stc = (255, 110, 100) if tr.crashed else ((120, 230, 140) if tr.t_stable is not None else (230, 210, 120))
        d.text((x, y0 + 142), tr.status(), font=f_m, fill=stc)
    # 中间高度曲线
    px0, px1 = cw + sw + 40, W - cw - sw - 16
    py0, py1 = y0 + 10, H - 26
    d.rectangle([px0 - 32, y0 + 2, px1 + 8, H - 4], fill=(20, 24, 33))
    zlo, zhi, tmax = -0.6, 3.4, STEPS * DT
    X = lambda t: px0 + (px1 - px0) * t / tmax
    Y = lambda z: py1 - (py1 - py0) * (z - zlo) / (zhi - zlo)
    for z in (0, 1, 2, 3):
        d.line([(px0, Y(z)), (px1, Y(z))], fill=(48, 56, 70))
        d.text((px0 - 8, Y(z)), f'{z}', font=f_s, fill=(140, 150, 166), anchor='rm')
    d.line([(px0, Y(2.5)), (px1, Y(2.5))], fill=(70, 180, 100), width=1)
    d.line([(px0, Y(-0.5)), (px1, Y(-0.5))], fill=(190, 70, 70), width=1)
    for s in range(0, 6):
        d.text((X(s), py1 + 3), f'{s}s', font=f_s, fill=(140, 150, 166), anchor='ma')
    for name in ('v8', 'v9'):
        zs = pair.tr[name].zs
        if len(zs) > 1:
            stp = max(1, len(zs) // 250)
            pts = [(X((j + 1) * DT), Y(max(z, -0.5))) for j, z in list(enumerate(zs))[::stp]]
            d.line(pts, fill=C255[name], width=2)
    d.line([(X(pair.t), py0), (X(pair.t), py1)], fill=(150, 150, 160, 160))
    d.text((px0 + 6, py0), '高度 z（m）', font=f_s, fill=(170, 180, 196))
    d.text((px1 - 4, Y(2.5) - 2), '目标', font=f_s, fill=(90, 190, 115), anchor='rb')
    d.text((px1 - 4, Y(-0.5) - 2), '坠地', font=f_s, fill=(200, 90, 90), anchor='rb')
    return np.asarray(frame)


def export_video(args):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    from visualize_recovery import MUJOCO_XML
    W, H = args.width, args.height
    H3 = int(round(H * 0.75)); CH = H - H3
    model = mujoco.MjModel.from_xml_string(MUJOCO_XML); data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    ids = scene_ids(model)
    for gi in range(model.ngeom):          # 固定机位下人形 / 桌子挡在画面中央 → 视频里隐藏，保留高度标尺
        gname = model.geom(gi).name or ''
        if gname.startswith(('h_', 'd_', 'desk', 'inst')):
            model.geom_rgba[gi, 3] = 0.0
    r_main = mujoco.Renderer(model, H3, W, max_geom=4000)
    r_cl = mujoco.Renderer(model, CH, CH, max_geom=500)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 1.2]; cam.distance = 4.6; cam.azimuth = 90; cam.elevation = -4
    camc = mujoco.MjvCamera(); camc.type = mujoco.mjtCamera.mjCAMERA_FREE
    camc.distance = 0.42; camc.azimuth = 90; camc.elevation = -18
    sc = H / 720.0
    fonts = (_font(int(22 * sc), True), _font(int(19 * sc)), _font(int(15 * sc)), _font(int(34 * sc), True))
    fps = args.fps
    spf = max(1, int(round(args.speed / (fps * DT))))
    eff_speed = spf * fps * DT
    out = args.export
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                           '-crf', str(args.crf), '-preset', 'medium', '-movflags', '+faststart', out],
                          stdin=subprocess.PIPE)
    pair = Pair()
    sel = [c for c in CASES if (not args.cases or c['idx'] in args.cases)]
    t0 = time.time()
    for n, c in enumerate(sel):
        pair.reset(c['idx']); props = {'v8': np.zeros(4), 'v9': np.zeros(4)}

        def frame(banner=None):
            apply_state(mujoco, model, data, pair, ids, props)
            r_main.update_scene(data, camera=cam)
            draw_overlays(mujoco, r_main.scene, pair)
            img = r_main.render()
            cl = {}
            for nm in ('v8', 'v9'):
                camc.lookat[:] = disp_pos(pair, nm)
                r_cl.update_scene(data, camera=camc)
                p = disp_pos(pair, nm)
                if not pair.tr[nm].crashed:
                    _arrow(mujoco, r_cl.scene, p, p + 0.12 * body_z(pair.tr[nm].quat), 0.004, (*RGBA[nm], 1.0))
                cl[nm] = r_cl.render()
            return _hud(img, cl, pair, c['note'], eff_speed, W, H, fonts, banner)

        f = frame(banner=f'工况 {n + 1}/{len(sel)}：{c["note"]}')
        for _ in range(int(fps * 1.6)):
            ff.stdin.write(f.tobytes())
        while not pair.done:
            for _ in range(spf):
                pair.step()
            ff.stdin.write(frame().tobytes())
        a, b = pair.tr['v8'], pair.tr['v9']
        if a.crashed and not b.crashed:
            summ = f'v8 坠毁，v9 掉高 {b.drop:.2f} m 后恢复'
        elif a.crashed and b.crashed:
            summ = '两者都坠毁'
        else:
            summ = f'掉高 v8 {a.drop:.2f} m → v9 {b.drop:.2f} m'
            if a.t_stable is not None and b.t_stable is not None:
                summ += f'，稳定悬停 {a.t_stable:.2f} s → {b.t_stable:.2f} s'
        f = frame(banner=summ)
        for _ in range(int(fps * 2.4)):
            ff.stdin.write(f.tobytes())
        print(f'[{n + 1}/{len(sel)}] #{c["idx"]} {summ}  ({time.time() - t0:.0f}s)', flush=True)
    ff.stdin.close(); ff.wait()
    print(f'视频导出完成：{out}（{os.path.getsize(out) / 1e6:.2f} MB）')


# ============================ 曲线图 / 核对 ============================
def export_plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    cjk = None
    for path in ['/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'] + [p for p, _ in FONTS]:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            cjk = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams['font.family'] = ['DejaVu Sans'] + ([cjk] if cjk else [])   # 拉丁字母/数字用 DejaVu，中文回退到 CJK 字体
    plt.rcParams['axes.unicode_minus'] = False
    pair = Pair()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), CASES):
        pair.reset(c['idx']); pair.run_to_end()
        for nm in ('v8', 'v9'):
            tr = pair.tr[nm]; zs = np.array(tr.zs); ts = np.arange(1, len(zs) + 1) * DT
            col = np.array(RGBA[nm])
            lab = f'{nm}：掉高 {tr.drop:.2f} m，' + (f'坠毁 @{tr.t_crash:.2f}s' if tr.crashed else
                                                   (f'稳悬 {tr.t_stable:.2f}s' if tr.t_stable is not None else '未稳悬'))
            ax.plot(ts, np.maximum(zs, -0.5), color=col * 0.85, lw=2, label=lab)
            ax.plot([tr.t_zmin], [max(tr.zmin, -0.5)], 'v', color=col * 0.85, ms=7)
            if tr.t_stable is not None:
                ax.axvline(tr.t_stable, color=col * 0.85, ls=':', lw=1.2)
        ax.axhline(2.5, color='#3a9d57', lw=1, ls='--'); ax.axhline(-0.5, color='#c0392b', lw=1)
        ax.set_title(f'#{c["idx"]} {c["note"]}\n倾角 {pair.tilt0:.0f}°，|ω| {pair.w0:.1f} rad/s（偏航 {pair.wz0:+.1f}）', fontsize=10.5)
        ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel('时间 t（s）')
    for ax in axes[:, 0]:
        ax.set_ylabel('高度 z（m）')
    axes[0, 0].set_ylim(-0.7, 3.4); axes[0, 0].set_xlim(0, STEPS * DT)
    fig.suptitle('v8 vs v9 同一初始状态的高度曲线（绿虚线 = 目标 2.5 m，红线 = 坠地 -0.5 m，▼ = 最低点，点线 = 进入稳定悬停）', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=100)
    print(f'曲线图导出完成：{out}')


def check():
    from eval_recovery_metrics import run_metrics
    from vec_expert8_env import VecExpert8
    from vec_expert9_env import VecExpert9
    ref = {'v8': run_metrics(VecExpert8, N_ENVS, SEED), 'v9': run_metrics(VecExpert9, N_ENVS, SEED)}
    pair = Pair(); ok = True
    for c in CASES:
        pair.reset(c['idx']); pair.run_to_end(); i = c['idx']
        for nm in ('v8', 'v9'):
            tr = pair.tr[nm]; r = ref[nm]
            same = (tr.crashed == bool(r['crashed'][i])) and abs(tr.drop - r['drop'][i]) < 1e-6
            ok &= same
            print(f'#{i:3d} {nm}: 掉高 {tr.drop:.3f} (评测 {r["drop"][i]:.3f}) 坠毁 {tr.crashed} (评测 {bool(r["crashed"][i])}) '
                  f'稳悬 {tr.t_stable} (评测 {r["t_stable"][i]:.3f})  {"✓" if same else "✗"}')
    print('全部一致' if ok else '存在不一致')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='v8 vs v9 同工况并排 3D 对比')
    ap.add_argument('--case', type=int, default=None, help='交互查看器起始通道（0~511）')
    ap.add_argument('--export', type=str, default=None, help='导出 MP4 路径')
    ap.add_argument('--cases', type=int, nargs='*', default=None, help='导出时只导出这些通道（默认全部精选）')
    ap.add_argument('--width', type=int, default=1280); ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=25); ap.add_argument('--speed', type=float, default=0.5)
    ap.add_argument('--crf', type=int, default=23)
    ap.add_argument('--plot', type=str, default=None, help='导出高度曲线对比图 PNG 路径')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.check:
        check()
    elif a.plot:
        export_plot(a.plot)
    elif a.export:
        export_video(a)
    else:
        if not os.environ.get('DISPLAY') and os.path.exists('/tmp/.X11-unix/X0'):
            os.environ['DISPLAY'] = ':0'
        run_live(a)
