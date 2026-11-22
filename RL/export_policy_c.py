#!/usr/bin/env python3
"""
export_policy_c.py —— 将训练好的 PyTorch/SB3 策略网络权重导出为嵌入式 C 头文件
模仿 SelfRisingRobot 的 export_policy_header.py 设计
"""
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
