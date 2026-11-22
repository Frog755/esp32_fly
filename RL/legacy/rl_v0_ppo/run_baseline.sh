#!/usr/bin/env bash
# run_baseline.sh —— 一键启动 Baseline 训练并在后台拉起 TensorBoard
set -e

RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # Stage 20：脚本已移到 RL/legacy/rl_v0_ppo/
cd "$RL_DIR"

# 激活 Python 虚拟环境
VENV_PATH="/home/frog/ai_workspace/deepseek_workspace/esp32-rl-无人机回正/.venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "未找到虚拟环境: $VENV_PATH"
    exit 1
fi

OUT_DIR="$RL_DIR/models/baseline_run"
TB_DIR="$OUT_DIR/tb"
mkdir -p "$OUT_DIR"

# 检查是否已有 TensorBoard 在运行，没有则启动
if ! pgrep -f "tensorboard --logdir" > /dev/null; then
    echo "[1/2] 正在后台启动 TensorBoard 面板 (端口 6006)..."
    nohup tensorboard --logdir "$TB_DIR" --port 6006 --bind_all > "$RL_DIR/tb_logs/tb_server.log" 2>&1 &
    sleep 2
    echo "      TensorBoard 已就绪：http://localhost:6006"
else
    echo "[1/2] TensorBoard 已经在后台运行中 (端口 6006)"
fi

echo "[2/2] 准备启动 PPO 基线训练 (Curriculum 课程学习，100万步)..."
echo "      输出目录: $OUT_DIR"
echo "      日志将写入: $TB_DIR"
echo "--------------------------------------------------------"

# 启动训练
python3 legacy/rl_v0_ppo/train_ppo.py \
    --mode curriculum \
    --start 0.2 \
    --steps 1000000 \
    --net mlp \
    --seed 42 \
    --out "$OUT_DIR" \
    --tag "baseline_mlp"
