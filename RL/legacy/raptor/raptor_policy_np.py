#!/usr/bin/env python3
"""raptor_policy_np.py —— 把 raptor_weights.h 解析成 numpy 前向策略
用于在自研 sim (quad_sim_fast_v2) 中闭环测试 Raptor 基础策略。
观测约定 (22维): pos(3) + R展平(9) + lin_vel(3) + ang_vel(3) + prev_action(4)
坐标: FLU, 电机序 [FR, BR, BL, FL], 动作 = RPM 归一化 [-1,1]
"""
# --- [Stage 20 legacy 迁移引导] 本文件原位于 RL/scripts/，现移至 RL/legacy/raptor/ ---
# _LEGACY_ORIG_DIR 指回原目录：原有基于 __file__ 的相对路径（../env、../models 等）语义不变；
# 并把主线 env/、scripts/ 与各 legacy 子目录加入 sys.path，跨目录的旧导入照常可用。逻辑未改动。
import os as _lg_os, sys as _lg_sys
_LEGACY_RL = _lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.abspath(__file__))))
_LEGACY_ORIG_DIR = _lg_os.path.join(_LEGACY_RL, 'scripts')
for _lg_p in [_LEGACY_RL, _lg_os.path.join(_LEGACY_RL, 'env'), _lg_os.path.join(_LEGACY_RL, 'scripts')] + sorted(
        _lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d) for _lg_d in _lg_os.listdir(_lg_os.path.join(_LEGACY_RL, 'legacy'))
        if _lg_os.path.isdir(_lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d)) and not _lg_d.startswith(('.', '_'))):
    if _lg_p not in _lg_sys.path:
        _lg_sys.path.insert(1, _lg_p)
# --- [/Stage 20 legacy 迁移引导] ---
import re
import numpy as np

def parse_weights(path):
    src = open(path).read()
    out = {}
    src = re.sub(r'#define\s+RAPTOR_\w+\s+\d+', '', src)   # 去掉宏定义，支持用宏做维度的数组
    for m in re.finditer(r'static const float (raptor_\w+)\[([^\]]*)\](?:\[([^\]]*)\])?\s*=\s*\{([^;]*)\};', src):
        name, d1s, d2s, body = m.group(1), m.group(2), m.group(3), m.group(4)
        dims = {}
        for dm in re.finditer(r'#define\s+(RAPTOR_\w+)\s+(\d+)', open(RAPTOR_DEFAULT).read()):
            dims[dm.group(1)] = int(dm.group(2))
        dims['RAPTOR_OBS_DIM'], dims['RAPTOR_ACT_DIM'] = 22, 4
        dims['RAPTOR_N_TEST'], dims['RAPTOR_HIDDEN'] = 9, 16
        def dim(tok):
            tok = tok.strip()
            return int(tok) if tok.isdigit() else dims[tok]
        vals = [float(x) for x in re.findall(r'[-+0-9.eE]+', body.replace('f', ' '))]
        if d2s:
            out[name] = np.array(vals, dtype=np.float64).reshape(dim(d1s), dim(d2s))
        else:
            out[name] = np.array(vals, dtype=np.float64)
    return out

class RaptorPolicy:
    """dense(22->16, ReLU) -> GRU(16->16, 门序 r|z|n) -> dense(16->4, Identity)"""
    def __init__(self, wpath):
        w = parse_weights(wpath)
        self.w0, self.b0 = w['raptor_w0'], w['raptor_b0']
        self.w_in, self.b_in = w['raptor_w_in'], w['raptor_b_in']
        self.w_hid, self.b_hid = w['raptor_w_hid'], w['raptor_b_hid']
        self.w2, self.b2 = w['raptor_w2'], w['raptor_b2']
        self.h = w['raptor_h0'].copy()          # 初始隐状态
    def reset(self):
        self.h = np.zeros(16)
    def step(self, obs22):
        x = np.asarray(obs22, dtype=np.float64)
        h0 = np.maximum(self.w0 @ x + self.b0, 0.0)          # ReLU
        Wi = self.w_in @ h0 + self.b_in                      # 48
        Wh = self.w_hid @ self.h + self.b_hid                # 48
        rz = 1.0 / (1.0 + np.exp(-(Wi[0:32] + Wh[0:32])))
        r, z = rz[0:16], rz[16:32]
        n = np.tanh(Wh[32:48] * r + Wi[32:48])
        self.h = z * self.h + (1.0 - z) * n
        return self.w2 @ self.h + self.b2                    # 4 动作

RAPTOR_DEFAULT = "/home/frog/embodied_ai/esp32_fly/code/connection_test/main/raptor_weights.h"

if __name__ == "__main__":
    import sys
    w = parse_weights(RAPTOR_DEFAULT)
    for k, v in w.items():
        print(f"{k:<16} {v.shape}")
    p = RaptorPolicy(RAPTOR_DEFAULT)
    # 校验：与 C 头文件里的 test_expect 对照
    obs0 = w['raptor_test_obs'][0]
    act = p.step(obs0)
    exp = w['raptor_test_expect'][0]
    print("单步校验 obs[0]:")
    print("  py  :", np.round(act, 6))
    print("  C参考:", np.round(exp, 6))
    print("  max_err:", np.max(np.abs(act - exp)))
