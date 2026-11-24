#!/usr/bin/env python3
"""motor_bench 判据的离线自检：用合成的遥测行模拟 A / B / C 三步的正常与故障情形（不需要板子）"""
import sys, os, types
sys.modules.setdefault('serial', types.SimpleNamespace(SerialException=Exception, Serial=None))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import motor_bench as mb

COLS = mb.COLS


def row(t, st, out, k=1.9, roll=0.0, vb=4.1):
    v = dict(zip(COLS, [t, st, 0, roll, 0, 0, 0, 0, 0, vb, k, .9, .9, .9, .9, out, 0, 1]))
    return v


def seq(spans):
    """spans：[(时长 ms, 状态, out)]，50 Hz"""
    rows, t = [], 1000.0
    for dur, st, out in spans:
        for _ in range(int(dur / 20)):
            rows.append(row(t, st, out)); t += 20
    return rows


ok_all = True
def check(c, msg):
    global ok_all
    print(('  ✓ ' if c else '  ✗ ') + msg); ok_all &= bool(c)

# A 正常：解锁 1 s → TAKEOFF 6 s 满上限 → LAND 1.3 s 半上限 → DISARMED
rows = seq([(1000, 1, 0), (6000, 2, 0.24), (1300, 5, 0.12), (1000, 0, 0)])
t_land = rows[50 + 300 + 65 - 1]['t']
evts = ['EVT 2000 armed ARMED', 'EVT 2000 takeoff TAKEOFF', 'EVT 8000 land_takeoff_timeout LAND', f'EVT {t_land + 10:.0f} landed DISARMED']
r = mb.analyse(rows, evts); ok, why = mb.check_common(r)
check(ok and r['zero_after'] and abs(r['on_s'] - 7.3) < 0.1 and r['stop_evt'] == 'landed', f'A 正常：通过、通电 {r["on_s"]:.1f} s、停转滞后 {r["stop_lag_ms"]:.0f} ms')
# A 故障：超上限
rows2 = seq([(1000, 1, 0), (2000, 2, 0.30), (1000, 0, 0)])
ok, why = mb.check_common(mb.analyse(rows2, ['EVT 4000 landed DISARMED']))
check(not ok and any('超过上限' in w for w in why), 'A 故障：输出超过 4×6% → 判失败：' + '；'.join(why))
# 停转后仍有输出
rows3 = seq([(1000, 1, 0), (2000, 2, 0.24), (500, 0, 0), (100, 0, 0.05), (500, 0, 0)])
ok, why = mb.check_common(mb.analyse(rows3, ['EVT 3990 disarm_cmd DISARMED']))
check(not ok, '故障：DISARMED 后又出现输出 → 判失败：' + '；'.join(why))
# B 急停滞后过长：disarm 事件后还输出 200 ms
rows4 = seq([(1000, 1, 0), (2000, 2, 0.24), (1000, 0, 0)])
ok, why = mb.check_common(mb.analyse(rows4, ['EVT 3780 disarm_cmd DISARMED']))      # 最后通电 3980 ms
check(not ok and any('仍在输出' in w for w in why), '故障：急停事件后 200 ms 仍在输出 → 判失败：' + '；'.join(why))
# B 正常
ok, why = mb.check_common(mb.analyse(rows4, ['EVT 3970 disarm_cmd DISARMED']))
check(ok, '急停正常（事件后 ≤ 1 个遥测周期停）→ 通过')
# 电机几乎没通电
ok, why = mb.check_common(mb.analyse(seq([(3000, 1, 0), (1000, 0, 0)]), []))
check(not ok, '电机没通电 → 判失败：' + '；'.join(why))
print('判据自检全部通过' if ok_all else '有失败项')
sys.exit(0 if ok_all else 1)
