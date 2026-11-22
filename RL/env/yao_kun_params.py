#!/usr/bin/env python3
"""yao_kun_params.py —— 耀坤 K1.0（立创 ESP32-S3 版）实机参数集

硬件来源（2026-11 核实）：
  - 立创开源硬件平台 oshwhub.com/lindahe/project_sulkgarf（耀坤无人机立创 ESP32S3 版）
  - GitHub YaoKunS/ESP32-Drone（自研飞控，GPL v3）
  - 动力：8520 空心杯有刷电机 ×4 + 65mm 桨，AO3416 MOSFET PWM 20kHz 驱动
  - 电池：1S 3.7V 520mAh 30C 锂电（作者实测"静态电压>3.6V 才能起飞悬停"→ 低 TWR 特征）
  - 全重 ~80g（用户口述），纯姿态自稳飞控，基础版无位置/速度/高度传感器

标定状态标注：
  [实测]   = 来自开源页/仓库明确数据
  [区间]   = 来自同类 8520+65mm 动力系统公开数据，取中值
  [待标定] = 必须实机台架/试飞校准的量（附标定方法）
"""
import numpy as np
from quad_sim_fast_v2 import QuadParams


class YaoKunK10Params(QuadParams):
    """耀坤 K1.0 参数集。与旧 70g 参数的差异：mass/TWR/电池/轴距/k_m。"""

    def __init__(self):
        super().__init__(
            # [实测] 全重 ~80g（含 1S 520mAh 电池、PCB 机架、65mm 桨、保护圈）
            mass=0.080,
            # [区间] 8520+65mm@3.7V 满油门单桨推力：公开数据 25~30g 区间，取 27.5g=0.27N。
            #   交叉验证：作者"3.6V 才能起飞"→ 3.6V 时单桨 0.256N(26g)，4 桨 104g vs 80g 全重
            #   → TWR=1.3，与"勉强起飞"描述吻合；4.2V 满电 TWR=1.73。
            #   [待标定] 电子秤+推力台架实测：满油门克数（最关键的单一未知量）。
            thrust_max_per_motor=0.27,
            # [区间] 8520 带桨负载转速 ~23000-25000 RPM @3.7V（无载 40k+）
            omega_max=2500.0,                      # rad/s ≈ 23873 RPM
            # [区间] 65mm 桨的转矩/推力等效臂 10~15mm，取 12mm（旧值 8mm 偏小）
            k_m_ratio=0.012,
            # [实测] PCB 即机架 10×10cm X 型；同类 8520 机型轴距 118mm → 臂 41.7mm
            wheelbase=0.118,
            rotor_pos=((+0.0417, -0.0417), (-0.0417, -0.0417),
                       (-0.0417, +0.0417), (+0.0417, +0.0417)),
            # [区间] 惯量按质量比 80/70 缩放旧估值（×1.14）。
            #   [待标定] 摆动周期法实测：绕轴摆动 T=2π√(I/mgd)，三轴各测一次。
            inertia=(5.7e-5, 6.6e-5, 1.14e-4),
            # [实测] 1S 520mAh 30C
            batt_capacity_mah=520.0,
            # [区间] 30C 小包内阻 25~40mΩ，取 35mΩ（压降直接进推力模型）
            batt_r_internal=0.035,
            # [区间] 8520 满油门电流 2.5~3A，取 2.5A（影响 v_batt 压降曲线）
            current_max_per_motor=2.5,
            # [区间] 空心杯+MOSFET PWM 20kHz：电气时间常数 ~1ms 可忽略，
            #   机械滞后由桨+转子惯量主导，30/45ms 与同类实测一致，暂保留
            motor_tau_up=0.030, motor_tau_down=0.045,
            # [实测] 1S 锂电 3.7V 标称 / 4.2V 满电
            v_ref=3.7, v_full=4.2, v_empty=3.5,
            # [区间] 65mm 双叶桨线性/角阻尼与旧机型同量级，暂保留
            drag_lin=(0.02, 0.02, 0.03), drag_ang=(2e-6, 2e-6, 4e-6),
        )
        # _post() 会重算 k_f = thrust_max / omega_max² = 0.27/2500² = 4.32e-8


def ledger(p=None):
    """参数台账：悬停点 / TWR / 电压边缘——训练前必看。"""
    p = p or YaoKunK10Params()
    mg = p.mass * 9.81
    u_hover = (mg / (4 * p.thrust_max_per_motor)) ** 0.5
    print(f"=== 耀坤 K1.0 参数台账 ===")
    print(f"全重 {p.mass*1000:.0f}g，自重 {mg:.3f}N")
    print(f"单桨最大推力 @3.7V: {p.thrust_max_per_motor*1000/9.81:.1f}g  "
          f"(ω_max={p.omega_max:.0f} rad/s ≈ {p.omega_max*60/(2*np.pi):.0f} RPM)")
    print(f"悬停油门 u = {u_hover:.3f}   (悬停电流 ≈ {p.current_max_per_motor*u_hover**1.5*4+0.2:.1f}A)")
    print(f"TWR @3.7V = {4*p.thrust_max_per_motor/mg:.2f}")
    print(f"TWR @4.2V = {4*p.thrust_max_per_motor*(p.v_full/p.v_ref)**2/mg:.2f}")
    print(f"TWR @3.6V = {4*p.thrust_max_per_motor*(3.6/p.v_ref)**2/mg:.2f}  ← 开源页'3.6V 才能起飞'检验")
    print(f"8A 悬停压降 = {8*p.batt_r_internal:.2f}V ({8*p.batt_r_internal/3.7*100:.0f}%)")
    sag = 10 * p.batt_r_internal
    loss = 1 - ((3.7 - sag) / 3.7) ** 2   # 推力∝ω²，ω指令∝v_batt
    print(f"满油门 10A 压降 = {sag:.2f}V → 推力瞬时损失 {loss*100:.0f}%")
    print(f"最大差动力矩 = 2*T_max*arm = {2*p.thrust_max_per_motor*p.rotor_pos[0][0]:.5f} N·m")
    hover_amp = (p.current_max_per_motor * u_hover**1.5 * 4 + 0.2) * 1000  # mA
    print(f"续航(悬停) ≈ {p.batt_capacity_mah/hover_amp*60:.1f} min   (悬停 {hover_amp/1000:.1f}A)")


if __name__ == "__main__":
    ledger()
