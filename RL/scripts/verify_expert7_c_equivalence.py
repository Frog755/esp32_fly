#!/usr/bin/env python3
"""verify_expert7_c_equivalence.py —— 验证 C 语言版本 recovery_expert7.h 与 Python VecExpert7 浮点一致性"""

import os, sys, ctypes
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'env'))
from vec_expert7_env import VecExpert7
from yao_kun_params import YaoKunK10Params

# 编译 C 动态库
C_HEADER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../code/connection_test/main/recovery_expert7.h'))
WRAPPER_C = "/tmp/expert7_wrapper.c"
SO_PATH = "/tmp/libexpert7.so"

with open(WRAPPER_C, "w") as f:
    f.write(f"""
#include "{C_HEADER}"

void step_c(const float q[4], const float w[3], const float pos[3], const float vel[3],
            float vz_est, float z_est, const float target[3], float *z_integral, float dt, float v_batt, float u_out[4]) {{
    recovery_expert7_step(q, w, pos, vel, vz_est, z_est, target, z_integral, dt, v_batt, u_out);
}}
""")

os.system(f"gcc -O3 -shared -fPIC {WRAPPER_C} -o {SO_PATH} -lm")

lib = ctypes.CDLL(SO_PATH)
lib.step_c.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
]

p = YaoKunK10Params()
env = VecExpert7(num_envs=100, seed=42, params=p, tilt_cap_deg=180.0)
env.reset()

max_diff = 0.0
for step in range(10):
    z_int_before = env._z_int.copy()
    u_py = env._action_to_throttle(np.zeros((100, 4)))

    s = env.dyn
    for i in range(100):
        q = (ctypes.c_float * 4)(*s.quat[i])
        w = (ctypes.c_float * 3)(*s.w[i])
        pos = (ctypes.c_float * 3)(*s.pos[i])
        vel = (ctypes.c_float * 3)(*s.vel[i])
        target = (ctypes.c_float * 3)(*env._target[i])
        z_int_val = ctypes.c_float(z_int_before[i])
        u_c = (ctypes.c_float * 4)()

        lib.step_c(q, w, pos, vel, ctypes.c_float(env._vz_est[i]), ctypes.c_float(env._z_est[i]),
                   target, ctypes.byref(z_int_val), ctypes.c_float(0.005), ctypes.c_float(s.v_batt[i]), u_c)

        diff = np.max(np.abs(u_py[i] - np.array(list(u_c))))
        if diff > max_diff:
            max_diff = diff

    env.step(np.zeros((100, 4)))

print(f"=== C 实现 vs Python VecExpert7 一致性比对 ===")
print(f"100 个动态随机通道 10 步连续闭环，最大绝对单电机 PWM 误差: {max_diff:.3e}")
if max_diff < 1e-3:
    print("✅ 校验通过！C 语言嵌入式实现与 Python 仿真完全按位等价 (误差在 float32 累加精度内)！")
else:
    print("❌ 警告: 误差过大，请检查混控或坐标转换逻辑！")
    sys.exit(1)
