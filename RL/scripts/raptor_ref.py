#!/usr/bin/env python3
"""
Raptor Foundation Policy 参考实现 + policy.h 解析器（最小可跑版）

用途：
  1. 从 rl-tools/px4-blob/policy.h（C++ constexpr 导出头）里抠出权重
  2. 用纯 numpy 复现前向：dense(22->16, ReLU) -> GRU(16->16) -> dense(16->4, Identity)
  3. 打印动作统计，作为 ESP32 端逐层对齐的黄金参考

网络结构依据（已核实 policy.h）：
  layer_0 dense  CONFIG dim=16 act=RELU     INPUT_SHAPE [500,2,22]  weights [16,22]
  layer_1 GRU    CONFIG HIDDEN=16 fast_tanh=false  weights_input [48,16] -> 门序 [r|z|n]
  layer_2 dense  CONFIG dim=4  act=IDENTITY weights [4,16]

GRU 更新式依据 rl-tools nn/layers/gru/operations_generic.h:
  pre = (W_in · x + b_in) + r * (W_hid · h + b_hid)
  r,z = sigmoid(pre[0:16]), sigmoid(pre[16:32])
  n   = tanh(pre[32:48])
  h'  = (1 - z) * n + z * h

用法：
  python3 raptor_ref.py --policy ../data/raptor_policy.h
  python3 raptor_ref.py --policy ../data/raptor_policy.h --dump ../output/weights.txt
"""
import argparse
import re
import sys

import numpy as np

HEX_FLOAT = re.compile(r"[-+]?0[xX][0-9a-fA-F.]+[pP][-+]?[0-9]+f")


def parse_policy(path):
    """把 policy.h 拆成 {层名: {参数名: ndarray}}，形状从 SHAPE 行读取。"""
    src = open(path, "r", encoding="utf-8", errors="ignore").read()
    layers = {}
    # 逐块扫描: namespace layer_N { ... }
    seen = set()
    # 注意：每层被拆成 "parameters_memory 块" 和 "CONFIG/TYPE 块" 两个同名 namespace，
    # 一份带权重字面量、一份不带，所以要遍历全部并合并。
    for lm in re.finditer(r"namespace (layer_\d+)\s*\{", src):
        name = lm.group(1)
        # 找到匹配的右括号（简单按层名切片到下一个 layer_ 或文件尾）
        nxt = src.find("namespace layer_", lm.end())
        body = src[lm.start(): nxt if nxt != -1 else len(src)]
        layers.setdefault(name, {})
        for pm in re.finditer(r"namespace (weights|biases|initial_hidden_state|\w*(?:weights|biases)\w*)\s*\{", body):
            pname = pm.group(1)
            if pname in layers[name]:
                continue
            seg = body[pm.start():]
            shape_m = re.search(r"Shape<unsigned long,\s*([0-9,\s]+)>", seg)
            if not shape_m:
                continue
            shape = tuple(int(x) for x in shape_m.group(1).split(",") if x.strip())
            depth_m = re.search(r"float memory\[\] = \{", seg)
            if not depth_m:
                continue
            lit = seg[depth_m.end(): seg.find("}", depth_m.end())]
            vals = [float.fromhex(v.rstrip("f")) if v.rstrip("f").startswith(("0X", "0x", "-0X", "-0x")) else float(v)
                    for v in HEX_FLOAT.findall(lit)]
            arr = np.array(vals, dtype=np.float32)
            if arr.size != int(np.prod(shape)):
                print(f"  [warn] {name}/{pname}: 解析 {arr.size} 个浮点，但 SHAPE={shape} "
                      f"需要 {int(np.prod(shape))} — 权重可能是索引复用的，跳过", file=sys.stderr)
                continue
            layers[name][pname] = arr.reshape(shape)
    return layers


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RaptorPolicy:
    """单样本、有状态（需要跨步保留 h）。"""

    def __init__(self, layers):
        l0, l1, l2 = layers["layer_0"], layers["layer_1"], layers["layer_2"]
        self.w0, self.b0 = l0["weights"], l0["biases"]
        self.w_in, self.b_in = l1["weights_input"], l1["biases_input"]
        self.w_hid, self.b_hid = l1["weights_hidden"], l1["biases_hidden"]
        self.h0 = l1["initial_hidden_state"].reshape(-1)
        self.w2, self.b2 = l2["weights"], l2["biases"]
        self.reset()

    def reset(self):
        self.h = self.h0.copy()

    def step(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        assert x.size == self.w0.shape[1], f"观测维度 {x.size} != {self.w0.shape[1]}"
        a = np.maximum(self.w0 @ x + self.b0, 0.0)              # [16]
        # GRU（严格按 foundation_policy 官方实现对齐）：
        #   Wi = W_in·x + b_in, Wh = W_hid·h + b_hid，门序 [r | z | n]
        #   r,z = sigmoid(Wi_rz + Wh_rz)          <- 注意 rz 一起过 sigmoid
        #   n   = tanh( r*(Wh_n) + Wi_n )         <- r 只作用在 hidden 分支
        #   h'  = z*h + (1-z)*n
        H = self.h.shape[0]
        Wi = self.w_in @ a + self.b_in
        Wh = self.w_hid @ self.h + self.b_hid
        H2 = 2 * H
        rz = sigmoid((Wi[:H2] + Wh[:H2]).astype(np.float64)).astype(np.float32)
        r, z = rz[:H], rz[H:]
        n = np.tanh((r * Wh[H2:] + Wi[H2:]).astype(np.float64)).astype(np.float32)
        self.h = (z * self.h + (1.0 - z) * n).astype(np.float32)
        return self.w2 @ self.h + self.b2                       # 输出层 act=IDENTITY（squash 在训练侧）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="../data/raptor_policy.h")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    layers = parse_policy(args.policy)
    print("解析到的层:", {k: sorted(v) for k, v in layers.items()})
    p = RaptorPolicy(layers)

    total = sum(a.size for l in layers.values() for a in l.values())
    print(f"可导出浮点参数: {total} 个 → {total * 4 / 1024:.1f} KB")

    # 静止悬停位姿（FLU: 姿态=单位阵展平，全零速度与动作）
    obs = np.zeros(22, dtype=np.float32)
    obs[3:12] = np.eye(3).reshape(-1)   # 旋转矩阵展平
    print("\n初始观测 [x,y,z,R(9),v(3),w(3),a_prev(4)]:", obs.tolist())
    for i in range(args.steps):
        act = p.step(obs)
        print(f"  step {i}: action = {np.round(act, 4).tolist()}")
        obs[18:22] = act                 # 把动作塞回 previous_action

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for lname in sorted(layers):
                for pname in sorted(layers[lname]):
                    f.write(f"{lname}/{pname} shape={layers[lname][pname].shape}\n")
                    f.write(np.array2string(layers[lname][pname], precision=8,
                                            separator=", ", max_line_width=120) + "\n\n")
        print(f"\n权重已导出 → {args.dump}")


if __name__ == "__main__":
    main()
