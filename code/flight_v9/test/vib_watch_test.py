#!/usr/bin/env python3
"""vib_check.UsbWatch 离线测试：正常停转、通电中芯片复位、一直没通电"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import vib_check as V

ok_all = True
def check(c, msg):
    global ok_all
    print(('  ✓ ' if c else '  ✗ ') + msg); ok_all &= bool(c)

def F(st, out, vb=4.1):
    return f"F 1000 {st} 0 0.1 -0.2 0.0 0.0 0.0 0.0 {vb:.2f} 1.9 0.9 0.9 0.9 0.9 {out:.3f} 0.0 1.0"

# 正常：解锁 → 通电 6 s → 降落 → 停转 2 s
w = V.UsbWatch(); t = 0.0
for st, out, n in ((1, 0, 50), (2, 1.2, 300), (5, 1.2, 65), (0, 0, 100)):
    for _ in range(n):
        w.feed(F(st, out, 3.34 if out else 4.13), t); t += 0.02
check(w.on_seen and w.done and w.resets == 0 and abs(w.out_max - 1.2) < 1e-9 and abs(w.vb_min - 3.34) < 1e-9,
      '正常流程：检测到通电 → 停转 1.5 s 后结束；最大输出 1.20、电池最低 3.34 V')
# 停转但还没满 1.5 s
w = V.UsbWatch(); t = 0.0
for st, out, n in ((2, 1.2, 100), (0, 0, 50)):
    for _ in range(n):
        w.feed(F(st, out), t); t += 0.02
check(not w.done, '停转不足 1.5 s 不结束')
# 通电中复位
w = V.UsbWatch(); t = 0.0
for _ in range(100):
    w.feed(F(2, 1.2), t); t += 0.02
k = w.feed('BOOT reset_reason 11', t)
check(k == 'RESET' and w.done and w.resets == 1 and w.reset_reasons == ['11'], '通电中芯片复位（USB 复位 = 11）→ 结束，记 1 次复位')
# 开机日志里的复位（还没通电）不算结束
w = V.UsbWatch()
w.feed('BOOT reset_reason 3', 0.0)
check(not w.done and w.resets == 1, '通电前的开机日志不结束')
# 杂行、EVT
w = V.UsbWatch()
check(w.feed('EVT 25600 takeoff TAKEOFF', 0) == 'EVT' and w.events and w.feed('garbage ### F 1 2', 0) is None, 'EVT 行记录、杂行忽略')
print('USB 不拔监视逻辑测试全部通过' if ok_all else '有失败项')
sys.exit(0 if ok_all else 1)
