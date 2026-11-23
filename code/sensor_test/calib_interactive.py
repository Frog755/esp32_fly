#!/usr/bin/env python3
"""calib_interactive.py —— 加速度计六面标定 + 芯片→机体轴映射（交互式，你自己控制节奏）

前提：板子上正在运行 sensor_test 固件（上电 / 复位后约 20 s 进入 50 Hz 流式输出，电机引脚全程锁低）。
用法（在 code/sensor_test 目录下）：  ./calib.sh        或    python calib_interactive.py [/dev/ttyACM0]

流程：每个姿态 → 你摆好、放稳（最好靠在桌面/书本上，别一直手拿）→ 按回车 → 脚本采 3 s：
  · 不够静止会提示你重来；朝向和前面某个姿态重复、或明显斜放，也会提示重来
  · 6 个姿态都采完后计算：每轴零偏 + 比例因子、芯片→机体 FLU 矩阵（x 前 / y 左 / z 上）
输出：imu_calib.json（数据与结果）、imu_calib.h（可直接 #include 进固件）
只依赖 pyserial，不需要 numpy。
"""
import sys, time, json, math

try:
    import serial
except ImportError:
    sys.exit('缺少 pyserial：请用 ./calib.sh 运行（它会自动用 ESP-IDF 自带的 Python），或 pip install pyserial')

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
POSES = [  # (提示, 机体哪根轴朝上, 正负)
    ('平放（正常起飞姿态，机腹朝下）', 2, +1),
    ('倒扣（机腹朝上）', 2, -1),
    ('机头朝上竖起', 0, +1),
    ('机头朝下竖起', 0, -1),
    ('左侧朝下立起（右侧朝上）', 1, -1),
    ('右侧朝下立起（左侧朝上）', 1, +1),
]
SAMPLE_S = 3.0
GYRO_MAX_DPS = 5.0        # 采样期间陀螺模长上限
ACC_STD_MAX_G = 0.02      # 采样期间加速度每轴标准差上限
AXIS = 'xyz'


def open_port():
    try:
        s = serial.Serial(PORT, 115200, timeout=0.2)
    except serial.SerialException as e:
        sys.exit(f'打不开串口 {PORT}：{e}\n'
                 '  · 用 ls /dev/ttyACM* 确认板子枚举了（没有就换 USB 线/口）\n'
                 '  · 有别的程序占着就先关掉（idf.py monitor / 串口助手 / 上一次的 calib.sh）')
    time.sleep(0.2)
    s.reset_input_buffer()
    return s


def read_rows(ser, seconds):
    """读 seconds 秒的 S 行，返回 [(ax,ay,az,gx,gy,gz), ...]"""
    rows, buf, t0 = [], '', time.time()
    while time.time() - t0 < seconds:
        buf += ser.read(2048).decode('utf-8', 'replace')
        while '\n' in buf:
            line, buf = buf.split('\n', 1)
            p = line.strip().split()
            if len(p) == 8 and p[0] == 'S':
                try:
                    rows.append(tuple(float(x) for x in p[2:]))
                except ValueError:
                    pass
    return rows


def mean_std(rows, k):
    v = [r[k] for r in rows]
    m = sum(v) / len(v)
    return m, math.sqrt(max(0.0, sum((x - m) ** 2 for x in v) / len(v)))


def check_stream(ser, wait_s=40.0):
    """等板子进入 50 Hz 流式输出。

    固件上电/复位后要先跑约 20 s 自检，这段时间一行都不发，
    所以不能只等 1.5 s 就判定"没数据"。这里最多等 wait_s 秒。
    """
    t0, rows, tail, next_note = time.time(), 0, '', 3.0
    while True:
        el = time.time() - t0
        chunk = ser.read(4096)
        if chunk:
            text = chunk.decode('utf-8', 'replace')
            tail = (tail + text)[-300:]
            rows += sum(1 for ln in text.split('\n') if ln.startswith('S ') and len(ln.split()) == 8)
        if rows >= 40:
            n = len(read_rows(ser, 1.0))
            print(f'✓ 串口 {PORT} 有数据（{n} Hz）\n')
            return
        if el >= wait_s:
            msg = (f'等了 {wait_s:.0f} s 仍没收到 50 Hz 流式数据（S 行）。\n'
                   '可能原因：\n'
                   '  · 板子刚复位/还在自检：等它跑完约 20 s 再运行（本脚本已等这么久，说明确实没在发）\n'
                   '  · 串口被别的程序占着：idf.py monitor、串口助手、上一次没退干净的 calib.sh\n'
                   f'  · 串口选错了：当前用 {PORT}，用 ls /dev/ttyACM* 确认\n'
                   '  · 固件不对/没烧进去：重新 idf.py -p ' + PORT + ' flash 后再试')
            if tail.strip():
                msg += f'\n最近收到的串口内容：\n{tail.strip()}'
            else:
                msg += '\n这段时间串口一个字节都没收到（板子没在跑，或端口被占用/选错）'
            sys.exit(msg)
        if el >= next_note:
            print(f'   …已等 {el:.0f} s，还没看到流式数据（复位后要自检约 20 s）', flush=True)
            next_note += 5.0


def capture(ser, prompt, used):
    while True:
        input(f'→ 请把飞机摆成：【{prompt}】，放稳后按回车 ')
        ser.reset_input_buffer()
        print(f'   采样 {SAMPLE_S:.0f} 秒，别碰飞机……', flush=True)
        rows = read_rows(ser, SAMPLE_S)
        if len(rows) < 0.8 * 50 * SAMPLE_S:
            print(f'   ✗ 只收到 {len(rows)} 行数据，串口可能断了，再试一次'); continue
        a = [mean_std(rows, k) for k in range(3)]
        gmax = max(math.sqrt(r[3] ** 2 + r[4] ** 2 + r[5] ** 2) for r in rows)
        astd = max(s for _, s in a)
        m = [x for x, _ in a]
        ax = max(range(3), key=lambda i: abs(m[i])); sg = 1 if m[ax] > 0 else -1
        others = max(abs(m[i]) for i in range(3) if i != ax)
        print(f'   加速度均值 ({m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.3f}) g   陀螺最大 {gmax:.1f} °/s   抖动 {astd * 1e3:.0f} mg')
        if gmax > GYRO_MAX_DPS or astd > ACC_STD_MAX_G:
            print('   ✗ 不够静止，请把飞机靠稳在桌面/书本上再试'); continue
        if abs(m[ax]) < 0.7 or others > 0.3:
            print('   ✗ 看起来是斜放的（没有一根轴明显朝上），请摆正一点再试'); continue
        if (ax, sg) in used:
            print(f'   ✗ 这个朝向（芯片 {AXIS[ax]}{"+" if sg > 0 else "-"} 朝上）和"{used[(ax, sg)]}"一样，请检查姿态再试'); continue
        print(f'   ✓ 记录：芯片 {AXIS[ax]}{"+" if sg > 0 else "-"} 朝上\n')
        return m, ax, sg, gmax, astd


def main():
    print(__doc__.split('\n用法')[0])
    ser = open_port()
    check_stream(ser)
    used, caps = {}, []
    for prompt, b_ax, b_sg in POSES:
        m, ax, sg, gmax, astd = capture(ser, prompt, used)
        used[(ax, sg)] = prompt
        caps.append(dict(pose=prompt, body_axis=AXIS[b_ax], body_sign=b_sg, chip_axis=AXIS[ax], chip_sign=sg,
                         acc_mean_g=m, gyro_max_dps=gmax, acc_std_g=astd))
    # ---- 轴映射：body = R·chip ----
    R = [[0] * 3 for _ in range(3)]
    for c in caps:
        b = AXIS.index(c['body_axis']); ch = AXIS.index(c['chip_axis'])
        R[b] = [0, 0, 0]; R[b][ch] = c['chip_sign'] * c['body_sign']
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    perm_ok = all(sum(abs(R[i][j]) for j in range(3)) == 1 for i in range(3)) and all(sum(abs(R[i][j]) for i in range(3)) == 1 for j in range(3))
    # ---- 六面法零偏 / 比例（芯片坐标系）----
    up = {}; dn = {}
    for c in caps:
        ch = AXIS.index(c['chip_axis'])
        (up if c['chip_sign'] > 0 else dn)[ch] = c['acc_mean_g'][ch]
    off = [(up[i] + dn[i]) / 2 for i in range(3)]
    scl = [(up[i] - dn[i]) / 2 for i in range(3)]
    norms = []
    for c in caps:
        v = [(c['acc_mean_g'][i] - off[i]) / scl[i] for i in range(3)]
        norms.append(math.sqrt(sum(x * x for x in v)))
    print('================ 结果 ================')
    print(f'加速度计零偏 (g)：x {off[0]:+.4f}  y {off[1]:+.4f}  z {off[2]:+.4f}')
    print(f'加速度计比例   ：x {scl[0]:.4f}   y {scl[1]:.4f}   z {scl[2]:.4f}')
    print('校正后 6 个姿态的 |a| (应都≈1.000)：' + '  '.join(f'{n:.3f}' for n in norms))
    print('芯片 → 机体 FLU（x 前 / y 左 / z 上）：')
    for i in range(3):
        terms = [f'{"+" if R[i][j] > 0 else "-"}芯片{AXIS[j]}' for j in range(3) if R[i][j]]
        print(f'   机体 {AXIS[i]} = {" ".join(terms)}')
    ok = perm_ok and det == 1 and all(abs(n - 1) < 0.02 for n in norms)
    if not perm_ok or det != 1:
        print(f'✗ 映射无效（行列式 {det}）：通常是某个姿态摆反了（比如机头朝上/朝下、左/右侧弄反），请重新运行')
    elif not ok:
        print('⚠ 映射有效，但有姿态校正后 |a| 偏离 1 超过 2%，建议重新运行一次')
    else:
        print('✓ 映射是真旋转（行列式 +1），校正后各姿态 |a| 误差都在 2% 以内')
    res = dict(port=PORT, captures=caps, acc_offset_g=off, acc_scale=scl, corrected_norm_g=norms,
               chip_to_body=R, det=det, valid=ok, time=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(res, open('imu_calib.json', 'w'), ensure_ascii=False, indent=1)
    with open('imu_calib.h', 'w', encoding='utf-8') as f:
        f.write(f'/* 自动生成：calib_interactive.py {res["time"]} —— 加速度计六面标定 + 芯片→机体 FLU 映射 */\n#pragma once\n')
        f.write(f'#define IMU_CALIB_VALID {1 if ok else 0}\n')
        f.write('/* 校正：a_chip = (raw_g - OFF) / SCALE；然后 body = R · chip（陀螺只做 R 映射，零偏另行开机静止校准） */\n')
        f.write('static const float IMU_ACC_OFF_G[3] = {%.5ff, %.5ff, %.5ff};\n' % tuple(off))
        f.write('static const float IMU_ACC_SCALE[3] = {%.5ff, %.5ff, %.5ff};\n' % tuple(scl))
        f.write('static const int8_t IMU_CHIP_TO_BODY[3][3] = {{%d, %d, %d}, {%d, %d, %d}, {%d, %d, %d}};\n' % tuple(x for r in R for x in r))
    print('\n已保存 imu_calib.json 和 imu_calib.h（在当前目录）。把终端输出或 imu_calib.json 发给我就行。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已取消')
