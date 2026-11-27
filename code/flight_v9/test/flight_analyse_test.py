#!/usr/bin/env python3
"""first_flight.analyse_flight 离线测试：合成一次标准首飞的黑匣子读出（与固件同格式），检查分析结果；再用真实的贴地振动检查数据检查"没离地"会被判失败"""
import sys, os, struct, base64, zlib, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import bb_read, first_flight as FF

ok_all = True
def check(c, msg):
    global ok_all
    print(('  ✓ ' if c else '  ✗ ') + msg); ok_all &= bool(c)

EV = {n: i for i, n in enumerate(bb_read.EV)}
BOOT = 4242


def rec(idx, t, st, ev, on, z, vb, u, out, roll_deg=0.0, k=1.9, vib=0.45):
    h = math.radians(roll_deg) / 2
    q = (math.cos(h), math.sin(h), 0.0, 0.0)
    flags = 1 | 2 | (4 if on else 0) | 16
    vals = [idx, int(t * 1000), BOOT, st, 0, ev, flags, int(k * 100), 0,
            *[int(round(x * 32767)) for x in q], 0, 0, 0, 0, 0, 1000,
            int(z * 1000), 0, int(z * 1000), 0, int(z * 1000), int(vb * 1000), 1000,
            *[int(round(x * 1e4)) for x in u], int(round(out * 1e4)), 75, 50, int(vib * 1000)]
    b = bytearray(bb_read.REC.pack(*vals))
    x = 0xA5
    for i, c in enumerate(b):
        if i != 15:
            x ^= c
    b[15] = x
    return bytes(b)


def dump(recs):
    lines = [f'BB_SLOT 0 {BOOT} 11 0 {len(recs) - 1} {len(recs)} 1 1']
    allc = 0
    for i in range(0, len(recs), 16):
        data = b''.join(recs[i:i + 16]); allc = zlib.crc32(data, allc)
        lines.append(f'BB 0 {i} {len(recs[i:i + 16])} {zlib.crc32(data):08x} {base64.b64encode(data).decode()}')
    lines.append(f'BB_END 0 {len(recs)} {allc:08x}')
    return lines


def flight(ck=0.95, ucmd=0.75, roll=2.0, land_ev='land_auto', stop_ev='landed'):
    sc = FF.DUTY_SCALE
    recs, t, idx = [], 0.0, 0
    def add(n, st, first_ev, on, zf, vb, u):
        nonlocal t, idx
        for j in range(n):
            z = zf(j / n)
            out = min(sc * ck * u, FF.U_MAX) * 4 if on else 0.0
            recs.append(rec(idx, t, st, EV[first_ev] if (j == 0 and first_ev) else 0, on, z, vb, [u] * 4 if on else [0] * 4, out,
                            roll if on else 0.0))
            idx += 1; t += 0.005
    add(400, 0, None, 0, lambda f: 0.0, 4.15, 0)
    add(200, 1, 'armed', 0, lambda f: 0.0, 4.15, 0)
    add(400, 2, 'takeoff', 1, lambda f: 0.8 * f, 3.40, ucmd)
    add(1000, 4, 'stable_hold', 1, lambda f: 0.8, 3.40, ucmd)
    add(500, 5, land_ev, 1, lambda f: 0.8 * (1 - f), 3.40, ucmd * 0.9)
    add(400, 0, stop_ev, 0, lambda f: 0.0, 4.10, 0)
    return dump(recs)


r = FF.analyse_flight(bb_read.parse(flight()))
check(r['found'] and r['passed'], '标准首飞（c_k 0.95）：全部判定通过')
check(abs(r['ck_hold'] - 0.95) < 0.005 and abs(r['hover_duty_next'] - FF.HOVER_BELIEF * 0.95) < 0.002,
      f'从黑匣子反算推力修正系数 {r["ck_hold"]:.3f}（真值 0.95），建议下次悬停 {r["hover_duty_next"]}')
check(abs(r['z_rise_hold'] - 0.8) < 0.02 and abs(r['hold_s'] - 5.0) < 0.05 and abs(r['t_to_hold'] - 2.0) < 0.02,
      f'悬停高度 {r["z_rise_hold"]:.2f} m、悬停 {r["hold_s"]:.1f} s、起飞到稳定 {r["t_to_hold"]:.2f} s')
check(abs(r['roll_max'] - 2.0) < 0.1 and abs(r['duty_hold'] - FF.DUTY_SCALE * 0.95 * 0.75) < 0.002,
      f'姿态最大 {r["roll_max"]:.1f}°、悬停真实占空比 {r["duty_hold"]:.3f}')
r = FF.analyse_flight(bb_read.parse(flight(ck=1.30, ucmd=1.0)))
check(not r['passed'] and r['sat_frac'] > 0.9, f'推力不够（指令顶到 1、实际被 70% 截断）→ 判失败，顶上限 {r["sat_frac"] * 100:.0f}%')
r = FF.analyse_flight(bb_read.parse(flight(roll=30.0, stop_ev='impact')))
check(not r['passed'] and r['stop'] == 'impact', '姿态 30° + 撞击停桨 → 判失败')
r = FF.analyse_flight(bb_read.parse(flight(land_ev='land_link_lost')))
check(not r['passed'] and r['land_why'] == 'land_link_lost', '链路丢失降落（不是悬停满 5 s）→ 判失败')
real = os.path.join(HERE, '..', 'blackbox', 'vib_20261127_144814_raw.txt')
if os.path.exists(real):
    r = FF.analyse_flight(bb_read.parse(open(real).read().split('\n')))
    check(r['found'] and not r['passed'] and not dict(r['checks'])['离地并进入悬停（stable_hold）'],
          '真实贴地 30% 振动检查数据：没有 stable_hold → 判"没离地"')
print('首飞分析离线测试全部通过' if ok_all else '有失败项')
sys.exit(0 if ok_all else 1)
