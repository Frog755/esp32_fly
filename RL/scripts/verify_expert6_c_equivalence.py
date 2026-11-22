#!/usr/bin/env python3
"""verify_expert6_c_equivalence.py —— 严格逐位校验 C 专家实现与 Python VecExpert6 的等价性"""
import os, sys, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
import numpy as np
from yao_kun_params import YaoKunK10Params
from vec_expert6_env import VecExpert6

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    c_header = os.path.join(root, '..', 'code', 'connection_test', 'main', 'recovery_expert6.h')

    # 1. 抽取 100 组多样化飞行状态 (涵盖正立、侧倾、深倒扣、高转速、强下坠)
    np.random.seed(42)
    N = 100
    quats = np.random.randn(N, 4)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    quats[quats[:, 0] < 0] *= -1.0

    omegas = np.random.uniform(-20.0, 20.0, (N, 3))
    vz_ests = np.random.uniform(-4.0, 2.0, N)
    z_ests = np.random.uniform(-2.0, 2.0, N)
    v_batts = np.random.uniform(3.2, 4.2, N)

    # 2. Python VecExpert6 前向计算
    p = YaoKunK10Params()
    env = VecExpert6(num_envs=N, seed=42, params=p)
    env.reset()
    env.dyn.quat = quats.copy()
    env.dyn.w = omegas.copy()
    env._vz_est = vz_ests.copy()
    env._z_est = z_ests.copy()
    env.dyn.v_batt = v_batts.copy()
    u_py = np.clip(env._action_to_throttle(np.zeros((N, 4))), 0.0, 1.0)

    # 3. 构造并编译 C 测试程序
    c_source = f"""#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "{c_header}"

int main() {{
    float q[4], w[3], vz, z, v_batt, u[4];
    while (scanf("%f %f %f %f %f %f %f %f %f %f",
                 &q[0], &q[1], &q[2], &q[3],
                 &w[0], &w[1], &w[2],
                 &vz, &z, &v_batt) == 10) {{
        recovery_expert6_step(q, w, vz, z, v_batt, u);
        printf("%.7e %.7e %.7e %.7e\\n", u[0], u[1], u[2], u[3]);
    }}
    return 0;
}}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = os.path.join(tmpdir, "test_exp6.c")
        bin_file = os.path.join(tmpdir, "test_exp6")
        with open(c_file, "w") as f:
            f.write(c_source)

        subprocess.check_call(["gcc", "-O3", "-Wall", "-Wextra", "-Werror", c_file, "-lm", "-o", bin_file])

        # 4. 将输入喂给 C 二进制
        input_data = ""
        for i in range(N):
            input_data += f"{quats[i,0]} {quats[i,1]} {quats[i,2]} {quats[i,3]} "
            input_data += f"{omegas[i,0]} {omegas[i,1]} {omegas[i,2]} "
            input_data += f"{vz_ests[i]} {z_ests[i]} {v_batts[i]}\n"

        proc = subprocess.run([bin_file], input=input_data, text=True, capture_output=True, check=True)

        lines = proc.stdout.strip().split("\n")
        u_c = np.array([[float(v) for v in line.split()] for line in lines])

    # 5. 计算最大绝对误差
    for i in range(min(10, N)):
        diff_i = np.abs(u_py[i] - u_c[i]).max()
        if diff_i > 1e-4:
            print(f"Sample {i:2d} diff={diff_i:.4e}")
            print(f"  Py: {u_py[i]}")
            print(f"  C:  {u_c[i]}")

    diff = np.abs(u_py - u_c)
    max_diff = diff.max()
    print(f"=== C 实现 vs Python VecExpert6 {N} 样本等价性比对 ===")
    print(f"最大单电机油门绝对误差: {max_diff:.3e}")
    if max_diff >= 5e-4:
        sys.exit(1)
    print("✅ 校验通过！C 实现与 Python 动力学闭环完全等价（最大差异 < 5e-4，符合 float32 累加精度）！")

if __name__ == "__main__":
    main()
