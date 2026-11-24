#!/usr/bin/env python3
"""bb_read.py —— 读出机载黑匣子（Stage 28）

用法（code/flight_v9 目录）：
  ./bb_read.sh                 落地后插 USB（不要拔电池！断电会丢数据），读出两个槽 → blackbox/bb_<时间>_slot<N>.csv + 摘要
  ./bb_read.sh --file X.txt    离线解码一份已保存的串口输出（主机单测也用这个）
板子要处于未解锁（DISARMED）状态；读出时暂停记录、暂停遥测，约 10~20 s。
"""
import sys, os, time, struct, base64, zlib, json

REC = struct.Struct('<IIH6B4h3h3h3hhhHH4HHBBH')
assert REC.size == 64
RAW = ['idx', 't_ms', 'boot', 'st', 'mode', 'ev', 'flags', 'k', 'chk', 'qw', 'qx', 'qy', 'qz', 'gx', 'gy', 'gz', 'ax', 'ay', 'az',
       'z', 'vz', 'zt', 'ab', 'zb', 'vb', 'thr', 'u1', 'u2', 'u3', 'u4', 'out', 'per_max', 'exec_max', 'vib']
STATE = ['DISARMED', 'ARMED', 'TAKEOFF', 'RECOVER', 'HOLD', 'LAND']
EV = ["none", "armed", "launch", "takeoff", "stable_hold", "land_auto", "land_cmd", "land_low_voltage", "land_recover_timeout",
      "land_takeoff_timeout", "land_flight_timeout", "land_link_lost", "landed", "disarm_cmd", "arm_timeout", "land_timeout_kill",
      "impact", "estimator_fail", "refuse_vbatt", "refuse_tilt", "refuse_estimator", "refuse_state", "disarm_link_lost", "refuse_link"]
RESET = {0: 'UNKNOWN', 1: 'POWERON', 2: 'EXT', 3: 'SW', 4: 'PANIC', 5: 'INT_WDT', 6: 'TASK_WDT', 7: 'WDT', 8: 'DEEPSLEEP', 9: 'BROWNOUT',
         10: 'SDIO', 11: 'USB', 12: 'JTAG', 13: 'EFUSE', 14: 'PWR_GLITCH', 15: 'CPU_LOCKUP'}


def chk_ok(b):
    x = 0xA5
    for i, c in enumerate(b):
        if i != 15:
            x ^= c
    return x == b[15]


def parse(lines):
    """返回 {槽号: dict(meta…, recs=[原始字段 dict], crc_ok, n_end)}"""
    slots, cur = {}, None
    for ln in lines:
        p = ln.strip().split()
        if not p:
            continue
        if p[0] == 'BB_SLOT':
            s = int(p[1])
            if p[2] == 'invalid':
                slots[s] = dict(valid=False, recs=[]); continue
            b, rr, lo, hi, nv, ic, fz = map(int, p[2:9])
            slots[s] = dict(valid=True, boot=b, reset_reason=rr, reset_name=RESET.get(rr, str(rr)), lo=lo, hi=hi, n_valid=nv,
                            is_current=bool(ic), frozen=bool(fz), recs=[], crc=0, bad_chk=0)
            cur = slots[s]
        elif p[0] == 'BB' and len(p) == 3 and cur is not None:
            data = base64.b64decode(p[2])
            cur['crc'] = zlib.crc32(data, cur['crc'])
            for i in range(0, len(data) - 63, 64):
                b = data[i:i + 64]
                if not chk_ok(b):
                    cur['bad_chk'] += 1; continue
                cur['recs'].append(dict(zip(RAW, REC.unpack(b))))
        elif p[0] == 'BB_END':
            s = int(p[1]); d = slots.get(s)
            if d and d.get('valid'):
                d['n_end'] = int(p[2]); d['crc_ok'] = (d['crc'] & 0xFFFFFFFF) == int(p[3], 16) and d['n_end'] == len(d['recs'])
    return slots


def scaled(r):
    f = r['flags']
    return dict(
        t=r['t_ms'] / 1e3, idx=r['idx'], st=r['st'], state=STATE[r['st']] if r['st'] < 6 else '?', mode=r['mode'],
        ev=EV[r['ev']] if r['ev'] and r['ev'] < len(EV) else '', link=f & 1, imu_ok=(f >> 1) & 1, on=(f >> 2) & 1,
        killed=(f >> 3) & 1, motor_fw=(f >> 4) & 1, k=r['k'] / 100,
        qw=r['qw'] / 32767, qx=r['qx'] / 32767, qy=r['qy'] / 32767, qz=r['qz'] / 32767,
        gx_dps=r['gx'] / 10, gy_dps=r['gy'] / 10, gz_dps=r['gz'] / 10, ax_g=r['ax'] / 1000, ay_g=r['ay'] / 1000, az_g=r['az'] / 1000,
        z=r['z'] / 1000, vz=r['vz'] / 1000, zt=r['zt'] / 1000, ab=r['ab'] / 1000, zb=r['zb'] / 1000, vb=r['vb'] / 1000,
        thr=r['thr'] / 1000, u1=r['u1'] / 1e4, u2=r['u2'] / 1e4, u3=r['u3'] / 1e4, u4=r['u4'] / 1e4, out=r['out'] / 1e4,
        per_max_us=r['per_max'] * 20, exec_max_us=r['exec_max'] * 10, vib_g=r['vib'] / 1000)


def euler(s):
    import math
    w, x, y, z = s['qw'], s['qx'], s['qy'], s['qz']
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    return roll, math.degrees(math.asin(sp)), math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def summary(si, d, ended_by=None):
    """ended_by：这次开机是怎么结束的（= 下一次开机的复位原因名；None = 这是当前开机）"""
    if not d.get('valid'):
        return [f'槽 {si}：空（断电后还没写过）']
    rs = [scaled(r) for r in d['recs']]
    out = [f'槽 {si}：开机号 {d["boot"]}（这次开机的复位原因 {d["reset_name"]}）{"【当前】" if d["is_current"] else ""}'
           f'{"【已冻结】" if d["frozen"] else ""}  记录 {len(rs)} 条 / {len(rs) / 200:.1f} s  校验 {"✓" if d.get("crc_ok") else "✗"}'
           + (f'  坏记录 {d["bad_chk"]}' if d['bad_chk'] else '')]
    if not rs:
        return out
    gaps = sum(1 for a, b in zip(rs, rs[1:]) if b['idx'] != a['idx'] + 1)
    out.append(f'   时间 {rs[0]["t"]:.1f} ~ {rs[-1]["t"]:.1f} s（开机后），序号断档 {gaps} 处')
    evs = [f'{s["t"]:.2f}s {s["ev"]}' for s in rs if s['ev']]
    out.append('   事件：' + (' → '.join(evs) if evs else '无'))
    on = [s for s in rs if s['on']]
    if on:
        import statistics as st
        an = [((s['ax_g'] ** 2 + s['ay_g'] ** 2 + s['az_g'] ** 2) ** 0.5) for s in on]
        eu = [euler(s) for s in on]
        out.append(f'   通电 {len(on) / 200:.2f} s：实际占空比之和 平均 {st.mean(s["out"] for s in on):.3f} 最大 {max(s["out"] for s in on):.3f}；'
                   f'电池最低 {min(s["vb"] for s in on):.2f} V')
        out.append(f'   门控 k 中位 {st.median(s["k"] for s in on):.2f}，k<0.5 占 {sum(s["k"] < 0.5 for s in on) / len(on) * 100:.0f}%；'
                   f'振动 中位 {st.median(s["vib_g"] for s in on):.3f} g 最大 {max(s["vib_g"] for s in on):.3f} g；比力最大 {max(an):.2f} g')
        out.append(f'   姿态 |roll| 最大 {max(abs(e[0]) for e in eu):.1f}°  |pitch| 最大 {max(abs(e[1]) for e in eu):.1f}°；'
                   f'高度 {min(s["z"] for s in on):+.2f} ~ {max(s["z"] for s in on):+.2f} m')
    out.append(f'   1 kHz 最长周期 {max(s["per_max_us"] for s in rs)} μs，最长执行 {max(s["exec_max_us"] for s in rs)} μs；'
               f'看门狗锁死 {"是" if any(s["killed"] for s in rs) else "否"}')
    if ended_by:
        bad = ended_by in ('PANIC', 'INT_WDT', 'TASK_WDT', 'WDT', 'BROWNOUT', 'PWR_GLITCH', 'CPU_LOCKUP', 'UNKNOWN')
        out.append(f'   （这是上一次开机的记录，以复位 {ended_by} 结束'
                   + ('——**意外复位**，出事前的数据就在这里）' if bad else '——SW = reboot 指令，USB = 烧录 / USB 总线复位）'))
    return out


def write_csv(path, d):
    rs = [scaled(r) for r in d['recs']]
    if not rs:
        return
    keys = list(rs[0].keys()) + ['roll', 'pitch', 'yaw']
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for s in rs:
            e = euler(s)
            f.write(','.join(str(s[k]) for k in keys[:-3]) + f',{e[0]:.2f},{e[1]:.2f},{e[2]:.2f}\n')


def read_serial(port):
    import serial
    ser = serial.Serial(port, 115200, timeout=0.2)
    time.sleep(0.2); ser.reset_input_buffer()
    ser.write(b'bbdump\n')
    buf, lines, t0, last = '', [], time.time(), time.time()
    while time.time() - t0 < 120:
        data = ser.read(65536)
        if data:
            last = time.time()
            buf += data.decode('ascii', 'replace')
            while '\n' in buf:
                ln, buf = buf.split('\n', 1); ln = ln.strip()
                if ln.startswith('BB'):
                    lines.append(ln)
                    if ln.startswith('BB_SLOT'):
                        print('   ' + ln, flush=True)
                elif ln.startswith('CLI') and 'bbdump' in ln:
                    print('   ' + ln)
                if ln == 'BB_DONE':
                    return lines
        elif time.time() - last > 5:
            break
    if not any(l.startswith('BB_SLOT') for l in lines):
        sys.exit('✗ 板子没有回应 bbdump（固件是不是 Stage 28 以后的？是否已解锁中？）')
    print('⚠ 没收到 BB_DONE，数据可能不完整')
    return lines


def main():
    args = sys.argv[1:]
    if '--file' in args:
        src = args[args.index('--file') + 1]
        lines = open(src).read().split('\n')
    else:
        port = next((a for a in args if not a.startswith('--')), '/dev/ttyACM0')
        print(f'从 {port} 读出黑匣子……')
        lines = read_serial(port)
    slots = parse(lines)
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blackbox')
    os.makedirs(od, exist_ok=True)
    tag = time.strftime('%Y%m%d_%H%M%S')
    if '--file' not in args:
        with open(os.path.join(od, f'bb_{tag}_raw.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
    print('================ 黑匣子 ================')
    meta = {}
    cur = next((d for d in slots.values() if d.get('valid') and d['is_current']), None)
    for si in sorted(slots):
        d = slots[si]
        ended = cur['reset_name'] if cur and d.get('valid') and not d['is_current'] and d['boot'] + 1 == cur['boot'] else None
        for l in summary(si, d, ended):
            print(l)
        d = slots[si]
        if d.get('valid') and d['recs'] and '--no-csv' not in args:
            p = os.path.join(od, f'bb_{tag}_slot{si}_boot{d["boot"]}.csv'); write_csv(p, d); print(f'   → {os.path.relpath(p)}')
        meta[si] = {k: v for k, v in d.items() if k not in ('recs', 'crc')} | dict(n_recs=len(d['recs']))
    if '--no-csv' not in args:
        json.dump(meta, open(os.path.join(od, f'bb_{tag}_meta.json'), 'w'), ensure_ascii=False, indent=1)
    if '--json' in args:
        print('JSON ' + json.dumps(dict((si, [r for r in slots[si]['recs']]) for si in slots)))
    ok = all(d.get('crc_ok', True) for d in slots.values() if d.get('valid'))
    print('全部校验通过' if ok else '⚠ 有槽校验失败')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
