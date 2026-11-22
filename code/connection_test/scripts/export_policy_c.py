#!/usr/bin/env python3
"""
从 raptor_policy.h 导出 ESP32 C 头文件 + 校验向量
=================================================
用途：把 Raptor 策略（22→16→GRU16→4, 2084 float）导出成纯 C 数组，
     供 ESP32-S3 做前向推理 + 耗时基准测试。

关键：权重直接转成 C 数组，同时生成一组「输入观测 → 期望动作」的校验向量，
     烧到板上后可以逐位比对，确保 C 实现和 numpy 参考实现完全一致。

用法：
  python3 export_policy_c.py \
      --policy ../../ai_workspace/deepseek_workspace/esp32-rl-无人机回正/data/raptor_policy.h \
      --out ../connection_test/main/
"""

import argparse
import os
import sys

import numpy as np

# raptor_ref.py 所在目录（可用 --ref-scripts 覆盖）
DEFAULT_REF_SCRIPTS = os.path.expanduser(
    "~/ai_workspace/deepseek_workspace/esp32-rl-无人机回正/scripts")

parse_policy = None
RaptorPolicy = None


def load_ref(ref_scripts):
    """延迟导入 raptor_ref，目录在参数解析后才确定"""
    global parse_policy, RaptorPolicy
    sys.path.insert(0, ref_scripts)
    from raptor_ref import parse_policy as _pp, RaptorPolicy as _rp
    parse_policy, RaptorPolicy = _pp, _rp


def emit_1d(f, name, arr):
    """输出一维 float 数组"""
    vals = ", ".join(f"{v:.9e}f" for v in arr.reshape(-1))
    f.write(f"static const float {name}[{arr.size}] = {{\n    {vals}\n}};\n\n")


def emit_2d(f, name, arr):
    """输出二维 float 数组（行主序）"""
    rows, cols = arr.shape
    f.write(f"static const float {name}[{rows}][{cols}] = {{\n")
    for r in range(rows):
        vals = ", ".join(f"{v:.9e}f" for v in arr[r])
        f.write(f"    {{ {vals} }},\n")
    f.write("};\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="raptor_policy.h 路径")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--n-vectors", type=int, default=8,
                    help="生成多少组校验向量")
    ap.add_argument("--ref-scripts", default=DEFAULT_REF_SCRIPTS,
                    help="raptor_ref.py 所在目录")
    args = ap.parse_args()

    load_ref(args.ref_scripts)
    L = parse_policy(args.policy)
    l0, l1, l2 = L["layer_0"], L["layer_1"], L["layer_2"]

    total = sum(a.size for l in L.values() for a in l.values())
    print(f"参数量: {total} float = {total*4/1024:.1f} KB")

    # ---------- 生成校验向量 ----------
    # 用确定性随机种子 + 合理范围的观测：
    #   pos(3) ±2m, R(9) 近似单位阵, lin_vel(3) ±1, ang_vel(3) ±2, prev_action(4) ±1
    rng = np.random.default_rng(20261029)
    policy = RaptorPolicy(L)

    cases = []
    for i in range(args.n_vectors):
        obs = np.zeros(22, dtype=np.float32)
        obs[0:3] = rng.uniform(-2.0, 2.0, 3)          # position
        # 用近似单位阵 + 小扰动，避免生成非正交矩阵
        base = np.eye(3).reshape(-1)
        obs[3:12] = base + rng.uniform(-0.15, 0.15, 9).astype(np.float32)
        obs[12:15] = rng.uniform(-1.0, 1.0, 3)        # linear velocity
        obs[15:18] = rng.uniform(-2.0, 2.0, 3)        # angular velocity
        obs[18:22] = rng.uniform(-1.0, 1.0, 4)        # previous action

        policy.reset()                                 # 每组都从初始 h 开始
        act = policy.step(obs)
        cases.append((obs, act, policy.h.copy()))

    # 另外加一组全零的静止悬停基准
    obs0 = np.zeros(22, dtype=np.float32)
    obs0[3:12] = np.eye(3).reshape(-1)
    policy.reset()
    act0 = policy.step(obs0)
    cases.insert(0, (obs0, act0, policy.h.copy()))

    # 连续 3 步测试（验证 hidden state 跨步传递）
    # 注意：必须 .copy()，否则 append 存的是引用，
    #      后面修改 o[18:22] 会污染已存入的观测，导致期望值对不上
    policy.reset()
    seq_obs = []
    seq_act = []
    for i in range(3):
        o = np.zeros(22, dtype=np.float32)
        o[3:12] = np.eye(3).reshape(-1)
        o[15] = 0.1 * (i + 1)
        o_snapshot = o.copy()          # ← 快照当前观测
        seq_obs.append(o_snapshot)
        a = policy.step(o)
        seq_act.append(a.copy())
        o[18:22] = a                   # 动作喂回作为下一步的 prev_action

    # ---------- 写 C 头文件 ----------
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "raptor_weights.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write("/*\n")
        f.write(" * Raptor 策略权重 —— 由 export_policy_c.py 自动生成，请勿手改\n")
        f.write(" *\n")
        f.write(" * 网络: dense(22->16, ReLU) -> GRU(16->16, 门序 r|z|n) -> dense(16->4, Identity)\n")
        f.write(f" * 参数: {total} float = {total*4/1024:.1f} KB\n")
        f.write(" *\n")
        f.write(" * GRU 更新式 (RLtools 实现):\n")
        f.write(" *   Wi = W_in*a + b_in        (48维)\n")
        f.write(" *   Wh = W_hid*h + b_hid      (48维)\n")
        f.write(" *   rz = sigmoid(Wi[0:32] + Wh[0:32])\n")
        f.write(" *   r  = rz[0:16],  z = rz[16:32]\n")
        f.write(" *   n  = tanh(r * Wh[32:48] + Wi[32:48])\n")
        f.write(" *   h' = z*h + (1-z)*n\n")
        f.write(" */\n\n")
        f.write("#ifndef RAPTOR_WEIGHTS_H\n#define RAPTOR_WEIGHTS_H\n\n")
        f.write("#define RAPTOR_OBS_DIM   22\n")
        f.write("#define RAPTOR_HIDDEN    16\n")
        f.write("#define RAPTOR_GRU_DIM   48   /* 3*HIDDEN: r|z|n */\n")
        f.write("#define RAPTOR_ACT_DIM   4\n\n")

        emit_2d(f, "raptor_w0", l0["weights"])
        emit_1d(f, "raptor_b0", l0["biases"])
        emit_2d(f, "raptor_w_in", l1["weights_input"])
        emit_1d(f, "raptor_b_in", l1["biases_input"])
        emit_2d(f, "raptor_w_hid", l1["weights_hidden"])
        emit_1d(f, "raptor_b_hid", l1["biases_hidden"])
        emit_1d(f, "raptor_h0", l1["initial_hidden_state"])
        emit_2d(f, "raptor_w2", l2["weights"])
        emit_1d(f, "raptor_b2", l2["biases"])

        # 校验向量
        f.write(f"#define RAPTOR_N_TEST {len(cases)}\n\n")
        f.write("static const float raptor_test_obs[RAPTOR_N_TEST][RAPTOR_OBS_DIM] = {\n")
        for obs, _, _ in cases:
            vals = ", ".join(f"{v:.9e}f" for v in obs)
            f.write(f"    {{ {vals} }},\n")
        f.write("};\n\n")

        f.write("static const float raptor_test_expect[RAPTOR_N_TEST][RAPTOR_ACT_DIM] = {\n")
        for _, act, _ in cases:
            vals = ", ".join(f"{v:.9e}f" for v in act)
            f.write(f"    {{ {vals} }},\n")
        f.write("};\n\n")

        # 连续步测试
        f.write("/* 连续 3 步，用于验证 hidden state 跨步传递 */\n")
        f.write("static const float raptor_seq_obs[3][RAPTOR_OBS_DIM] = {\n")
        for o in seq_obs:
            vals = ", ".join(f"{v:.9e}f" for v in o)
            f.write(f"    {{ {vals} }},\n")
        f.write("};\n\n")
        f.write("static const float raptor_seq_expect[3][RAPTOR_ACT_DIM] = {\n")
        for a in seq_act:
            vals = ", ".join(f"{v:.9e}f" for v in a)
            f.write(f"    {{ {vals} }},\n")
        f.write("};\n\n")

        f.write("#endif /* RAPTOR_WEIGHTS_H */\n")

    print(f"已生成 → {path}")

    # ---------- 打印校验向量摘要，方便人工核对 ----------
    print(f"\n校验向量 {len(cases)} 组（含 1 组静止悬停基准）:")
    for i, (obs, act, _) in enumerate(cases[:4]):
        print(f"  [{i}] action = {np.round(act, 6).tolist()}")
    print(f"\n连续 3 步动作:")
    for i, a in enumerate(seq_act):
        print(f"  step {i}: {np.round(a, 6).tolist()}")


if __name__ == "__main__":
    main()
