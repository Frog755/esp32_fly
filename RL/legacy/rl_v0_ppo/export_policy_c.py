#!/usr/bin/env python3
"""
export_policy_c.py —— 将训练好的 PyTorch/SB3 策略网络权重导出为嵌入式 C 头文件
模仿 SelfRisingRobot 的 export_policy_header.py 设计
"""
# --- [Stage 20 legacy 迁移引导] 本文件原位于 RL/，现移至 RL/legacy/rl_v0_ppo/ ---
# _LEGACY_ORIG_DIR 指回原目录：原有基于 __file__ 的相对路径（../env、../models 等）语义不变；
# 并把主线 env/、scripts/ 与各 legacy 子目录加入 sys.path，跨目录的旧导入照常可用。逻辑未改动。
import os as _lg_os, sys as _lg_sys
_LEGACY_RL = _lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.dirname(_lg_os.path.abspath(__file__))))
_LEGACY_ORIG_DIR = _LEGACY_RL
for _lg_p in [_LEGACY_RL, _lg_os.path.join(_LEGACY_RL, 'env'), _lg_os.path.join(_LEGACY_RL, 'scripts')] + sorted(
        _lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d) for _lg_d in _lg_os.listdir(_lg_os.path.join(_LEGACY_RL, 'legacy'))
        if _lg_os.path.isdir(_lg_os.path.join(_LEGACY_RL, 'legacy', _lg_d)) and not _lg_d.startswith(('.', '_'))):
    if _lg_p not in _lg_sys.path:
        _lg_sys.path.insert(1, _lg_p)
# --- [/Stage 20 legacy 迁移引导] ---
import argparse
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Export trained PPO policy to C header for ESP32-S3")
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 model.zip")
    parser.add_argument("--out", type=str, default="../code/connection_test/main/policy_weights.h", help="Output C header path")
    args = parser.parse_args()

    print(f"正在加载模型: {args.model}")
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    
    # 提取 Policy 网络参数
    policy = model.policy
    print(f"网络结构: {policy}")
    
    # 后续实战中将在这里根据具体网络架构（MLP 或 GRU）输出格式化 C 数组
    print("模型解析成功，准备导出 C 代码...")

if __name__ == "__main__":
    main()
