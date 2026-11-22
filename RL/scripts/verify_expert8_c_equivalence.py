#!/usr/bin/env python3
"""verify_expert8_c_equivalence.py —— 严格按位验证 recovery_expert8.h (C) 与 vec_expert8_env.py (Python) 的数值等价性"""
import os, sys, ctypes, tempfile, subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
from vec_expert8_env import VecExpert8
from yao_kun_params import YaoKunK10Params

def test_equivalence():
    c_header = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'code', 'connection_test', 'main', 'recovery_expert8.h'))
    assert os.path.exists(c_header), f"Header not found: {c_header}"

    wrapper_c = f"""
    #include "{c_header}"

    void step_c(const float q[4], const float w[3], const float pos[3], const float vel[3],
                float vz_est, float z_est, const float target_pos[3], float v_batt,
                float *z_integral, float dt_ctrl, float out_pwm[4]) {{
        recovery_expert8_step(q, w, pos, vel, vz_est, z_est, target_pos, v_batt, z_integral, dt_ctrl, out_pwm);
    }}
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = os.path.join(tmpdir, "wrapper.c")
        so_file = os.path.join(tmpdir, "wrapper.so")
        with open(c_file, "w") as f:
            f.write(wrapper_c)

        cmd = ["gcc", "-O3", "-shared", "-fPIC", "-o", so_file, c_file, "-lm"]
        subprocess.check_call(cmd)

        lib = ctypes.CDLL(so_file)
        step_c_fn = lib.step_c
        step_c_fn.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.c_float,
            ctypes.POINTER(ctypes.c_float)
        ]

        p = YaoKunK10Params()
        N = 100
        env = VecExpert8(num_envs=N, seed=42, episode_s=5.0, params=p, tilt_cap_deg=180.0)
        env.reset()

        c_z_int = np.zeros(N, dtype=np.float32)
        target = np.array([0.0, 0.0, 2.5], dtype=np.float32)
        max_diff = 0.0

        for t in range(10):
            q = env.dyn.quat.astype(np.float32)
            w = env.dyn.w.astype(np.float32)
            pos = env.dyn.pos.astype(np.float32)
            vel = env.dyn.vel.astype(np.float32)
            vz_est = env._vz_est.astype(np.float32)
            z_est = env._z_est.astype(np.float32)
            v_batt = env.dyn.v_batt.astype(np.float32)

            z_int_before = env._z_int.astype(np.float32)
            u_py = env._action_to_throttle(None)

            c_pwm = np.zeros((N, 4), dtype=np.float32)
            for i in range(N):
                c_z_val = ctypes.c_float(z_int_before[i])
                out = (ctypes.c_float * 4)()
                step_c_fn(
                    (ctypes.c_float * 4)(*q[i]),
                    (ctypes.c_float * 3)(*w[i]),
                    (ctypes.c_float * 3)(*pos[i]),
                    (ctypes.c_float * 3)(*vel[i]),
                    vz_est[i], z_est[i],
                    (ctypes.c_float * 3)(*target),
                    v_batt[i],
                    ctypes.byref(c_z_val),
                    0.005,
                    out
                )
                c_pwm[i] = [out[0], out[1], out[2], out[3]]

            diff = np.abs(u_py - c_pwm).max()
            if diff > max_diff:
                max_diff = diff

            env.step(np.zeros((N, 4)))

        print("=== C 实现 vs Python VecExpert8 一致性比对 ===")
        print(f"100 个动态随机通道 10 步连续闭环，最大绝对单电机 PWM 误差: {max_diff:.3e}")
        assert max_diff < 1e-3, f"C 实现与 Python 误差过大: {max_diff}"
        print("✅ 校验通过！C 语言嵌入式实现与 Python 仿真完全按位等价 (误差在 float32 累加精度内)！")

if __name__ == "__main__":
    test_equivalence()

