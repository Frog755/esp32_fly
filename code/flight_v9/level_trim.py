#!/usr/bin/env python3
"""level_trim.py —— 测 IMU 安装倾斜（相对机架），生成 level_trim.h（你自己控制节奏，电机不转）

为什么要测：只靠 IMU 时，悬停中加速度计测到的是推力方向，飞控会把"IMU 认为的水平"当成水平；
  IMU 相对机架歪 1°，飞机悬停时就真歪 1°，水平加速度 0.17 m/s²，5 秒就有约 2 m 漂移。
  桌面上直接读 roll/pitch 分不清是 IMU 歪还是桌子歪——原地转 180° 再读一次：桌子坡度变号，IMU 安装倾斜不变，平均就是安装倾斜。
前提：板子运行 code/flight_v9 固件（任何版本均可，电机驱动未编译），上电 / 复位后静置约 8 s。
用法（code/flight_v9 目录）：  ./level_trim.sh
  第 1 次：放在尽量平的硬桌面上，机头朝任意方向，按回车采 3 s
  第 2 次：在原地（同一块位置）水平转 180°（机头朝反方向），按回车采 3 s
  第 3 次：转回第 1 次的朝向，按回车采 3 s（检查重复性）
输出：level_trim.h（覆盖）、level_trim.json。把终端输出发给我，我重新编译烧录。
"""
import sys, time, json, math, re, os

try:
    import serial
except ImportError:
    sys.exit('缺少 pyserial：请用 ./level_trim.sh 运行')

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
HERE = os.path.dirname(os.path.abspath(__file__))


def read_old():
    try:
        s = open(os.path.join(HERE, 'level_trim.h'), encoding='utf-8').read()
        r = float(re.search(r'LEVEL_TRIM_ROLL_DEG\s+([-\d.]+)', s).group(1))
        p = float(re.search(r'LEVEL_TRIM_PITCH_DEG\s+([-\d.]+)', s).group(1))
        return r, p
    except Exception:
        return 0.0, 0.0


def collect(ser, sec):
    ser.reset_input_buffer(); buf = ''; rows = []; t0 = time.time()
    while time.time() - t0 < sec:
        buf += ser.read(4096).decode('utf-8', 'replace')
        while '\n' in buf:
            line, buf = buf.split('\n', 1); p = line.split()
            if p and p[0] == 'F' and len(p) >= 17:
                try:
                    rows.append((float(p[4]), float(p[5]), float(p[11])))     # roll pitch gate_k
                except ValueError:
                    pass
    return rows


def main():
    print(__doc__.split('\n用法')[0])
    ser = serial.Serial(PORT, 115200, timeout=0.1); time.sleep(0.2)
    if len(collect(ser, 1.5)) < 30:
        sys.exit('1.5 s 内没收到 flight_v9 的数据。请确认烧的是 flight_v9 固件，复位后静置约 8 s 再运行。')
    old_r, old_p = read_old()
    print(f'✓ 收到数据。当前固件里的修正量：roll {old_r:+.3f}°  pitch {old_p:+.3f}°（测到的是修正后的残差）\n')
    prompts = ['第 1 次：放在平整的硬桌面上，放稳别动', '第 2 次：在原地水平转 180°（机头朝反方向），放稳别动', '第 3 次：转回第 1 次的朝向，放稳别动']
    meas = []
    for pr in prompts:
        while True:
            input(f'→【{pr}】，好了按回车 ')
            print('   采样 3 秒……', flush=True)
            rows = collect(ser, 3.0)
            if len(rows) < 100:
                print(f'   ✗ 只收到 {len(rows)} 行，再试一次'); continue
            r = sum(x[0] for x in rows) / len(rows); p = sum(x[1] for x in rows) / len(rows)
            sr = max(abs(x[0] - r) for x in rows); k = min(x[2] for x in rows)
            print(f'   roll {r:+.3f}°  pitch {p:+.3f}°（波动 ±{sr:.2f}°，门控增益最小 {k:.2f}）')
            if sr > 0.5 or k < 1.0:
                print('   ✗ 不够静止，请放稳再试'); continue
            meas.append((r, p)); print(); break
    (r1, p1), (r2, p2), (r3, p3) = meas
    mr, mp = (r1 + r2) / 2, (p1 + p2) / 2
    sr, sp = (r1 - r2) / 2, (p1 - p2) / 2
    rep = math.hypot(r3 - r1, p3 - p1)
    new_r, new_p = old_r + mr, old_p + mp
    print('================ 结果 ================')
    print(f'IMU 安装倾斜（残差）：roll {mr:+.3f}°  pitch {mp:+.3f}°')
    print(f'桌面坡度          ：roll {sr:+.3f}°  pitch {sp:+.3f}°（与飞机无关）')
    print(f'重复性（第 3 次 vs 第 1 次）：{rep:.3f}°' + ('  ✓' if rep < 0.3 else '  ⚠ 偏大，建议重测（桌面不硬 / 没放回同一位置）'))
    print(f'新的总修正量：roll {new_r:+.3f}°  pitch {new_p:+.3f}°')
    ok = rep < 0.3 and abs(new_r) < 6 and abs(new_p) < 6
    # body = Ry(pitch)·Rx(roll)·imu ：把"IMU 认为的水平"转回机架水平
    a, b = math.radians(new_r), math.radians(new_p)
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    Rx = [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]; Ry = [[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]
    R = [[sum(Ry[i][k] * Rx[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    with open(os.path.join(HERE, 'level_trim.h'), 'w', encoding='utf-8') as f:
        f.write(f'/* level_trim.h —— IMU 安装水平修正，level_trim.py {time.strftime("%Y-%m-%d %H:%M:%S")} 生成\n')
        f.write(f' * 测量：({r1:+.3f},{p1:+.3f}) / 转 180° ({r2:+.3f},{p2:+.3f}) / 复测 ({r3:+.3f},{p3:+.3f})；桌面坡度 ({sr:+.3f},{sp:+.3f})\n')
        f.write(' * 机体 = LEVEL_TRIM_R · (芯片→机体映射后的 IMU 读数)；生成新文件后需重新编译烧录。\n */\n#pragma once\n')
        f.write(f'#define LEVEL_TRIM_VALID {1 if ok else 0}\n#define LEVEL_TRIM_ROLL_DEG  {new_r:.3f}f\n#define LEVEL_TRIM_PITCH_DEG {new_p:.3f}f\n')
        f.write('static const float LEVEL_TRIM_R[3][3] = {' + ', '.join('{' + ', '.join(f'{x:.7f}f' for x in row) + '}' for row in R) + '};\n')
    json.dump(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), meas=meas, mount=[mr, mp], table=[sr, sp], repeat_deg=rep,
                   old=[old_r, old_p], new=[new_r, new_p], valid=ok), open(os.path.join(HERE, 'level_trim.json'), 'w'), indent=1)
    print(('✓' if ok else '⚠') + ' 已写入 level_trim.h' + ('' if ok else '（标记为无效，固件不会使用）') + '、level_trim.json。把终端输出发给我，我重新编译烧录。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已取消')
