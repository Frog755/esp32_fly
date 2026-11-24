#!/usr/bin/env python3
"""bb_test.py —— 用 bb_read.py 解码 bb_test.c 输出的读出文本，与 C 侧期望值逐条对拍"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import bb_read

B = os.path.join(HERE, '..', 'build_host')
ok_all = True
def check(c, msg):
    global ok_all
    print(('  ✓ ' if c else '  ✗ ') + msg); ok_all &= bool(c)

slots = bb_read.parse(open(os.path.join(B, 'bb_dump.txt')).read().split('\n'))
exp = [tuple(int(x) for x in l.split()) for l in open(os.path.join(B, 'bb_expect.txt')) if l.strip()]
check(set(slots) == {0, 1} and all(d['valid'] for d in slots.values()), '解析出两个有效槽（夹杂的非 BB 行被忽略）')
check(all(d['crc_ok'] for d in slots.values()), 'crc32 与条数校验通过（C 的 base64 / crc32 与 Python 一致）')
check(slots[0]['boot'] == 2 and slots[0]['is_current'] and slots[1]['boot'] == 1 and slots[1]['reset_name'] == 'POWERON'
      and slots[0]['reset_name'] == 'PANIC', '槽头：开机号、当前槽、复位原因')
got = [(si, r['idx'], r['t_ms'], r['st'], r['qx'], r['gx'], r['gz'], r['u4'], r['vib']) for si in (0, 1) for r in slots[si]['recs']]
check(got == exp, f'逐条逐字段与 C 一致（{len(exp)} 条，含环形回绕后的顺序）')
s0 = bb_read.scaled(slots[1]['recs'][-1])
check(s0['state'] == 'RECOVER' and s0['on'] == 1 and abs(s0['vb'] - 4.15) < 1e-9 and abs(s0['u4'] - 6.5535) < 1e-9
      and abs(s0['ax_g'] + 0.012) < 1e-9 and s0['per_max_us'] == 1500 and s0['exec_max_us'] == 670, '换算：状态名、电压、指令、比力、时序单位')
summ = bb_read.summary(1, slots[1], slots[0]['reset_name'])
check(any('通电' in l for l in summ) and any('意外复位' in l for l in summ), '摘要包含通电段统计与\"上一次开机\"提示')
# 模拟 USB 丢行 / 坏行：删掉 3 行、改坏 1 行 → 不完整；合并 bbget 补发后完整且与 C 一致
lines = open(os.path.join(B, 'bb_dump.txt')).read().split('\n')
bb_idx = [i for i, l in enumerate(lines) if l.startswith('BB ')]
drop = {bb_idx[1], bb_idx[7], bb_idx[-2]}
bad = bb_idx[4]
lossy = [(l[:-6] + 'AAAA==' if i == bad else l) for i, l in enumerate(lines) if i not in drop]
lossy = [l for l in lossy if not l.startswith('BB_END')] + [lines[bb_idx[-1]][:40]]   # 结尾截断的半行
sl = bb_read.parse(lossy)
miss = {si: bb_read.missing_ranges(d) for si, d in sl.items()}
n_miss = sum(b - a + 1 for v in miss.values() for a, b in v)
check(not all(d['complete'] for d in sl.values()) and n_miss == 64 and sum(d['bad_lines'] for d in sl.values()) == 2,
      f'丢 3 行 + 改坏 1 行 + 截断半行：判不完整，缺 {n_miss} 条（4 × 16）、坏行 2（改坏 1 + 截断 1）')
sl = bb_read.parse(open(os.path.join(B, 'bb_get.txt')).read().split('\n'), sl)
got2 = [(si, r['idx'], r['t_ms'], r['st'], r['qx'], r['gx'], r['gz'], r['u4'], r['vib']) for si in (0, 1) for r in sl[si]['recs']]
check(all(d['complete'] for d in sl.values()) and got2 == exp and not any(bb_read.missing_ranges(d) for d in sl.values()),
      '合并 bbget 补发后完整，逐条与 C 一致')
inv = bb_read.parse(open(os.path.join(B, 'bb_dump_invalid.txt')).read().split('\n'))
check(inv[0]['valid'] is False, '无效槽解析为空')
print('黑匣子 Python 对拍全部通过' if ok_all else '有失败项')
sys.exit(0 if ok_all else 1)
