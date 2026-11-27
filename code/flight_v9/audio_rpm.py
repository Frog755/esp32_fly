#!/usr/bin/env python3
"""audio_rpm.py —— Stage 31：用电脑麦克风录电机声音，测转速 → 估推力量级（区分"推力模型对"与"推力大 5 倍"）

原理：桨每转一圈，不平衡 / 电刷换向 / 两片桨叶各产生一次周期声音 → 声谱里有一串等间隔的谐波，间隔 = 转速 f_r（转/秒），
  2 叶桨的叶片通过频率 = 2·f_r。推力 T = Ct·ρ·n²·D⁴（螺旋桨基本关系，n 为转/秒，D 为桨径）。
  Ct（推力系数）小桨约 0.08~0.12，用 0.10，误差约 ±30%；两种假设推力差 5 倍（转速差 2.3 倍），这个精度足够区分。
做法：录音全程（电机停 → 转 → 停），取"电机转"那段的平均谱与"电机停"那段的平均谱相减（dB），只保留电机带来的新谱线，
  再找谐波梳（同一个 f_r 的 1~8 次谐波都抬高）——电脑风扇、环境声在两段里都有，会被减掉。
用法：
  python3 audio_rpm.py --record 20 out.wav        录 20 s（测试麦克风）
  python3 audio_rpm.py --analyse out.wav [--u 0.30 --vb 3.34]   分析
  python3 audio_rpm.py --selftest                 用合成谐波 + 真实环境噪声验证分析算法
"""
import sys, os, time, wave, json, subprocess, shutil
try:
    import numpy as np          # 分析要 numpy；录音（Recorder）不需要——ESP-IDF 的 Python 没装 numpy，只用它录音
except ImportError:
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
RHO, D_PROP, CT = 1.2, 0.065, 0.10          # 空气密度 kg/m³、桨径 m（65 mm）、推力系数（估计）
MASS, G = 0.080, 9.81
SIM_TMAX, SIM_WMAX, SIM_VREF = 0.27, 2500.0, 3.7   # 仿真模型（yao_kun_params.py）：单电机满推力 N、满转速 rad/s、参考电压


# ---------------------------------------------------------------- 录音
def default_source():
    try:
        s = subprocess.run(['pactl', 'get-default-source'], capture_output=True, text=True, timeout=5).stdout.strip()
        return s or None
    except Exception:
        return None


class Recorder:
    def __init__(self, path, max_s=240):
        self.path, self.p = path, None
        if not shutil.which('parecord'):
            raise RuntimeError('没有 parecord')
        src = default_source()
        cmd = ['timeout', str(max_s), 'parecord', '--rate=48000', '--channels=1', '--format=s16le', '--file-format=wav']
        if src:
            cmd.append(f'--device={src}')
        cmd.append(path)
        self.src = src
        self.p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.t0 = time.time()

    def stop(self):
        if self.p and self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(3)
            except subprocess.TimeoutExpired:
                self.p.kill()
        return time.time() - self.t0


def load(path):
    w = wave.open(path)
    fs = w.getframerate(); ch = w.getnchannels()
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x - x.mean(), fs


def analyse_external(path, u, vb=None, bb_csv=None, python='/usr/bin/python3', verdict=True):
    """在带 numpy 的系统 Python 里跑分析（vib_check 用 ESP-IDF 的 Python 运行，那边没有 numpy）；打印报告，返回结果 dict"""
    out = path + '.json'
    cmd = [python, os.path.abspath(__file__), '--analyse', path, '--u', str(u), '--json-out', out]
    if vb:
        cmd += ['--vb', f'{vb:.3f}']
    if bb_csv:
        cmd += ['--bb', bb_csv]
    if not verdict:
        cmd.append('--no-verdict')
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    print(r.stdout.rstrip())
    if r.returncode != 0:
        print(r.stderr[-800:]); return None
    return json.load(open(out))


# ---------------------------------------------------------------- 分析
def spectrogram(x, fs, win_s=0.25):
    n = int(win_s * fs); hop = n // 2
    w = np.hanning(n)
    frames = [x[i:i + n] * w for i in range(0, len(x) - n, hop)]
    S = np.abs(np.fft.rfft(np.array(frames), axis=1)) ** 2
    t = (np.arange(len(frames)) * hop + n / 2) / fs
    return t, np.fft.rfftfreq(n, 1 / fs), S


def _smooth(e, k):
    k = max(1, int(k)); pad = np.pad(e, (k, k), mode='edge')
    return np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(e))])


def find_on_segment(t, f, S, min_s=3.0, max_s=15.0):
    """电机转的那段：80~2000 Hz 频带能量（1 s 中值平滑）高于 安静水平 与 响的水平 中点的最长连续段"""
    band = (f > 80) & (f < 2000)
    e = _smooth(10 * np.log10(S[:, band].sum(axis=1) + 1e-9), 4)
    base, loud = np.percentile(e, 15), np.percentile(e, 95)
    if loud - base < 4:
        return None
    hot = e > base + 0.5 * (loud - base)
    best, cur = (0, -1), None
    for i, h in enumerate(hot):
        if h and cur is None:
            cur = i
        if (not h or i == len(hot) - 1) and cur is not None:
            jj = i if h else i - 1
            if jj - cur > best[1] - best[0]:
                best = (cur, jj)
            cur = None
    a, b = best
    if b < a:
        return None
    dur = t[b] - t[a]
    return dict(i0=a, i1=b, t0=float(t[a]), t1=float(t[b]), dur=float(dur), rise_db=float(loud - base),
                ok=min_s <= dur <= max_s)


def tonality(f, P, rel=0.08):
    """谱线突出度 dB：每个频点比它 ±8% 频带内的中位数高多少（去掉宽带噪声的整体抬升）"""
    L = 10 * np.log10(P + 1e-12)
    df = f[1] - f[0]
    out = np.zeros_like(L)
    for i in range(1, len(f)):
        w = max(4, int(rel * f[i] / df))
        lo, hi = max(1, i - w), min(len(f), i + w + 1)
        out[i] = L[i] - np.median(L[lo:hi])
    return out


def harmonic_comb(f, score, fmin=30.0, fmax=450.0, kmax=8, step=0.25):
    """在 [fmin, fmax] 里找基频 f_r：f_r 的 1..kmax 次谐波处"新谱线突出度"的平均值最大（缺谐波会拉低平均 → 抑制次谐波）；
    再查八度：若 f_r/2 的奇次谐波（0.5, 1.5, 2.5 … 倍）也明显突出，说明真正的基频是 f_r/2。"""
    df = f[1] - f[0]
    def at(fc):
        w = max(2, int(0.015 * fc / df)); i = int(round(fc / df))
        lo, hi = max(0, i - w), min(len(f) - 1, i + w)
        return float(score[lo:hi + 1].max()) if hi > lo else 0.0
    cands = []
    for f0 in np.arange(fmin, fmax, step):
        ks = [k for k in range(1, kmax + 1) if k * f0 < 2500]
        if len(ks) < 3:
            continue
        vals = [min(max(at(k * f0), 0.0), 20.0) for k in ks]
        cands.append((float(np.mean(vals)), float(f0), vals))
    cands.sort(key=lambda c: -c[0])
    score0, f0, vals = cands[0]
    odd_half = [at((k + 0.5) * f0) for k in range(0, 4)]          # f0/2 的奇次谐波
    octave_down = float(np.mean([min(max(v, 0), 20) for v in odd_half]))
    if octave_down > 0.6 * score0 and octave_down > 4:
        f0 = f0 / 2; vals = [min(max(at(k * f0), 0.0), 20.0) for k in range(1, kmax + 1)]
    return f0, (score0, f0, vals, octave_down)


def analyse(path, u=0.30, vb=None, bb_csv=None):
    x, fs = load(path)
    t, f, S = spectrogram(x, fs)
    seg = find_on_segment(t, f, S)
    out = dict(path=path, fs=fs, sec=len(x) / fs)
    if not seg:
        out.update(ok=False, msg='录音里没找到电机转的那一段（能量没有明显升高）'); return out
    a, b = seg['i0'], seg['i1']
    trim = max(2, int((b - a) * 0.1))
    on = S[a + trim:b - trim + 1].mean(axis=0)
    quiet_idx = [i for i in range(len(t)) if i < a - 4 or i > b + 8]          # 去掉起转 / 停转过渡
    if len(quiet_idx) < 8:
        out.update(ok=False, msg='电机转之前 / 之后的安静段太短，没法扣掉环境噪声'); return out
    quiet = S[quiet_idx].mean(axis=0)
    # 新谱线 = 电机转时突出、安静时不突出（电脑风扇等环境谱线两段都有，被扣掉）
    ratio = tonality(f, on) - np.maximum(tonality(f, quiet), 0.0)
    band = (f > 40) & (f < 3000)
    # 最强的新谱线
    idx = [i for i in np.argsort(ratio * band)[::-1] if band[i]]
    peaks = []
    for i in idx:
        if all(abs(f[i] - f[j]) > 0.03 * f[i] + 4 for j in peaks):
            peaks.append(i)
        if len(peaks) == 8:
            break
    f_r, top = harmonic_comb(f, ratio)
    n = f_r                                      # 转/秒
    T_motor = CT * RHO * n ** 2 * D_PROP ** 4
    W4 = MASS * G / 4
    frac = T_motor / W4                          # 单电机推力 / 悬停所需（重量的 1/4）
    u_hover = u / np.sqrt(frac) if frac > 0 else float('nan')          # T ∝ u²（转速 ∝ 油门）粗外推
    # 仿真模型在同一油门 / 电压下预测的转速
    vr = (vb / SIM_VREF) if vb else 1.0
    n_sim = u * vr * SIM_WMAX / (2 * np.pi)
    out.update(ok=bool(seg['ok']), seg=seg, f_rot=float(f_r), f_bpf=float(2 * f_r), rpm=float(60 * f_r),
               comb_score=round(float(top[0]), 1), octave_down=round(float(top[3]), 1), comb_vals=[round(float(v), 1) for v in top[2]],
               peaks=[(round(float(f[i]), 1), round(float(ratio[i]), 1)) for i in peaks],
               T_motor=float(T_motor), thrust_frac=float(frac), u_hover_est=float(u_hover), n_sim=float(n_sim),
               speed_ratio_vs_sim=float(f_r / n_sim), thrust_ratio_vs_sim=float((f_r / n_sim) ** 2), u=u, vb=vb)
    if bb_csv:
        try:
            out['imu'] = imu_check(f_r, imu_peaks(bb_csv))
        except Exception as e:
            out['imu_err'] = str(e)
    return out


# ---------------------------------------------------------------- 与黑匣子交叉核对
def imu_peaks(csv_path, n_peaks=3):
    """黑匣子 200 Hz 记录里，电机转时加计 / 陀螺的最强振动谱线（频率已经混叠到 0~100 Hz）"""
    import csv
    rs = list(csv.DictReader(open(csv_path)))
    on = np.array([float(r['on']) for r in rs]) == 1
    if on.sum() < 600:
        return []
    i0, i1 = np.where(on)[0][[0, -1]]
    sl = slice(i0 + 100, i1 - 20)
    tot = None
    for k in ('ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps'):
        x = np.array([float(r[k]) for r in rs])[sl]; x = x - x.mean()
        n = 512
        P = np.mean([np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n))) ** 2 for i in range(0, len(x) - n, n // 2)], axis=0)
        P = P / np.median(P[5:])
        tot = P if tot is None else tot + P
    f = np.fft.rfftfreq(512, 1 / 200.0)
    order = [i for i in np.argsort(tot)[::-1] if f[i] > 2]
    pk = []
    for i in order:
        if all(abs(f[i] - f[j]) > 2.5 for j in pk):
            pk.append(i)
        if len(pk) == n_peaks:
            break
    return [float(f[i]) for i in pk]


def alias200(fr):
    a = fr % 200.0
    return a if a <= 100 else 200.0 - a


def imu_check(f_rot, peaks):
    """f_rot 的 1~4 次谐波混叠到 200 Hz 采样后，离黑匣子最强谱线最近的是哪个（误差 Hz）"""
    if not peaks:
        return None
    best = min(((abs(alias200(k * f_rot) - p), k, p) for k in (1, 2, 3, 4) for p in peaks))
    return dict(err_hz=round(best[0], 1), k=best[1], peak=best[2], peaks=peaks,
                consistent=bool(best[0] < max(2.5, 0.02 * best[1] * f_rot)))


def report(r):
    if not r.get('seg'):
        print('✗ ' + r.get('msg', '分析失败')); return
    s = r['seg']
    print(f'   录音 {r["sec"]:.1f} s；电机转的那段 {s["t0"]:.1f} ~ {s["t1"]:.1f} s（{s["dur"]:.1f} s，比安静时响 {s["rise_db"]:.0f} dB）'
          + ('' if s['ok'] else '  ⚠ 时长不像 6~9 s 的通电段，结果仅供参考'))
    print('   电机带来的新谱线（频率 Hz / 突出 dB）：' + '  '.join(f'{p[0]:.0f}/{p[1]:+.0f}' for p in r['peaks']))
    print(f'   谐波梳：转速 f_r ≈ {r["f_rot"]:.1f} 转/秒（{r["rpm"]:.0f} rpm），叶片通过频率 {r["f_bpf"]:.0f} Hz；'
          f'1~8 次谐波突出 ' + ' / '.join(f'{v:.0f}' for v in r['comb_vals']) + ' dB')
    if r.get('no_verdict'):
        print(f'   按 Ct={CT}（±30%）：单电机推力约 {r["T_motor"] * 1000 / G:.0f} g，= 悬停所需的 {r["thrust_frac"] * 100:.0f}%（悬停时应接近 100%）')
        return
    print(f'   仿真模型在 {r["u"]:.0%} 油门' + (f'、{r["vb"]:.2f} V' if r['vb'] else '') + f' 下预测转速 {r["n_sim"]:.0f} 转/秒 → 实测 / 仿真 = {r["speed_ratio_vs_sim"]:.2f} 倍'
          f'（推力约 {r["thrust_ratio_vs_sim"]:.1f} 倍）')
    print(f'   按 Ct={CT}（±30%）：单电机推力约 {r["T_motor"] * 1000 / G:.0f} g，= 悬停所需的 {r["thrust_frac"] * 100:.0f}%；'
          f'粗外推悬停油门约 {r["u_hover_est"]:.0%}')
    ic = r.get('imu')
    if ic:
        print(f'   黑匣子交叉核对：IMU 振动最强谱线 {" / ".join(f"{p:.1f}" for p in ic["peaks"])} Hz（200 Hz 采样已混叠）；'
              f'{ic["k"]}× 转速混叠后 {alias200(ic["k"] * r["f_rot"]):.1f} Hz，差 {ic["err_hz"]} Hz → {"对得上" if ic["consistent"] else "⚠ 对不上"}'
              '（混叠后频率挤在 0~100 Hz，这项核对比较弱，仅供参考）')
    # 结论按 Ct 估出的推力（不依赖仿真的转速模型）；仿真模型 30% 时约为悬停所需的 10%，"40% ≈ 悬停"时约 55%
    fr = r['thrust_frac']
    if fr < 0.25:
        print('   → 结论：推力与仿真模型**基本一致**（40% 时的"离地"多半是振动跳动），现控制器的油门换算可以沿用')
    elif fr > 0.40:
        print('   → 结论：推力**远大于**仿真模型（接近"40% 就能悬停"），必须按实测修正油门换算后才能飞')
    else:
        print('   → 结论：推力比仿真模型大（介于两种假设之间），需要修正油门换算')


# ---------------------------------------------------------------- 自检：合成谐波 + 真实环境噪声
def selftest():
    noise_path = os.path.join(HERE, 'build', 'audio', 'dmic.wav')
    fs = 48000
    if os.path.exists(noise_path):
        nz, fs = load(noise_path)
    else:
        nz = np.random.default_rng(0).normal(0, 150, fs * 5)
    rng = np.random.default_rng(1)
    ok_all = True
    for f_r, a1 in ((95.0, 0.5), (108.0, 0.5), (226.0, 0.5), (300.0, 0.5), (108.0, 0.08), (226.0, 0.08)):
        T = 30.0; n = int(T * fs); t = np.arange(n) / fs
        x = np.tile(nz, int(np.ceil(n / len(nz))))[:n].copy()
        on = (t > 10) & (t < 18.3)
        env = np.clip((t - 10) / 0.3, 0, 1) * np.clip((18.3 - t) / 0.3, 0, 1)
        sig = np.zeros(n)
        for m, fm in enumerate((1.0, 1.006, 0.994, 1.012)):          # 4 个电机转速差 ~1%
            ph = rng.uniform(0, 2 * np.pi, 9)
            for k, amp in ((1, a1), (2, 1.0), (3, 0.35), (4, 0.5), (5, 0.2), (6, 0.25)):
                wm = 2 * np.pi * 0.7                                       # 转速 ±0.3% 慢波动：相位 = ∫ω dt（不能写成 f(t)·t）
                sig += amp * np.sin(2 * np.pi * k * f_r * fm * (t - 0.003 / wm * np.cos(wm * t)) + ph[k])
        x += 120 * env * sig + env * rng.normal(0, 200, n)            # 谐波 + 宽带气流噪声
        p = os.path.join(HERE, 'build', 'audio', f'_selftest_{int(f_r)}_{a1}.wav'); os.makedirs(os.path.dirname(p), exist_ok=True)
        with wave.open(p, 'w') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs)
            w.writeframes(np.clip(x, -32767, 32767).astype(np.int16).tobytes())
        r = analyse(p, u=0.30, vb=3.34)
        good = r.get('seg') and abs(r['f_rot'] / f_r - 1) < 0.03 and 7.5 < r['seg']['dur'] < 9.5
        ok_all &= bool(good)
        print(f'  {"✓" if good else "✗"} 合成转速 {f_r:.0f} 转/秒（基频幅度 {a1}）→ 测得 {r.get("f_rot", float("nan")):.1f}，'
              f'通电段 {r["seg"]["t0"]:.1f}~{r["seg"]["t1"]:.1f} s' if r.get('seg') else f'  ✗ {f_r}: {r.get("msg")}')
        os.remove(p)
    ic = imu_check(103.2, [6.3, 24.6, 12.1])
    good = ic['consistent'] and ic['k'] == 2
    ok_all &= good
    print(f'  {"✓" if good else "✗"} 混叠核对：103.2 转/秒 的 2 倍 206.4 Hz 混叠到 {alias200(206.4):.1f} Hz，对上 6.3 Hz 谱线')
    print('音频分析自检全部通过' if ok_all else '有失败项')
    return ok_all


if __name__ == '__main__':
    a = sys.argv[1:]
    if '--selftest' in a:
        sys.exit(0 if selftest() else 1)
    elif '--record' in a:
        sec = float(a[a.index('--record') + 1]); path = a[a.index('--record') + 2]
        rec = Recorder(path); time.sleep(sec); rec.stop(); print('录好了', path, '设备', rec.src)
    elif '--analyse' in a:
        path = a[a.index('--analyse') + 1]
        u = float(a[a.index('--u') + 1]) if '--u' in a else 0.30
        vb = float(a[a.index('--vb') + 1]) if '--vb' in a else None
        bb = a[a.index('--bb') + 1] if '--bb' in a else None
        r = analyse(path, u, vb, bb)
        if '--no-verdict' in a:
            r['no_verdict'] = True
        report(r)
        if '--json-out' in a:
            json.dump(r, open(a[a.index('--json-out') + 1], 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    else:
        print(__doc__)
