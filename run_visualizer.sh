#!/usr/bin/env bash
# ==============================================================================
# YaoKun K1.0 (ESP32-S3) 微型四旋翼自稳飞控 —— 交互式测评可视化与基准启动脚本
# ==============================================================================

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/RL:${PROJECT_ROOT}/RL/env:${PYTHONPATH}"

# 自动检查并绑定图形服务
if [ -z "$DISPLAY" ]; then
    if [ -e "/tmp/.X11-unix/X0" ]; then
        export DISPLAY=":0"
    fi
fi

show_help() {
    echo "=========================================================================="
    echo "   🚁 耀坤 K1.0 (ESP32-S3) 微型四旋翼 官方原生 MuJoCo 3D 自由视角测评程序"
    echo "=========================================================================="
    echo "使用方式:"
    echo "  ./run_visualizer.sh              # 默认启动 3D 原生对决 (左侧 v6 开环漂移 vs 右侧 v8 旗舰定点对决)"
    echo "  ./run_visualizer.sh v8v9         # v8 vs v9 同工况并排对比（左红 v8 / 右青 v9，按 1~6 切换精选工况）"
    echo "  ./run_visualizer.sh export-v8v9  # 离线导出 v8 vs v9 对比视频 + 高度曲线图到 RL/results/stage19/"
    echo "  ./run_visualizer.sh v8           # 启动 v8 陀螺解耦定点旗舰单机视口"
    echo "  ./run_visualizer.sh v7           # 启动 v7 级联定点自稳专家单机视口"
    echo "  ./run_visualizer.sh v6           # 启动 v6 前代姿态闭环专家单机视口"
    echo "  ./run_visualizer.sh deep         # 测评 149° 极限深倒扣工况"
    echo "  ./run_visualizer.sh fast         # 测评 15.4 rad/s 极限高角速度工况"
    echo "  ./run_visualizer.sh eval         # 运行 512 通道官方五代基准综合测评榜"
    echo "  ./run_visualizer.sh verify-c     # 运行 C 固件与 Python 浮点一致性校验"
    echo "  ./run_visualizer.sh export       # 离线批量导出高清对决视频与动图"
    echo "--------------------------------------------------------------------------"
    echo "3D 原生交互说明:"
    echo "  - 鼠标左键拖动: 360° 自由旋转视角 (Orbit)"
    echo "  - 鼠标右键拖动: 平移视野 (Pan)"
    echo "  - 鼠标滚轮: 自由缩放 (Zoom)"
    echo "  - Tab 键: 展开/折叠官方选项面板 (相机、线框、坐标轴、质心、力矢量)"
    echo "  - 空格键: 暂停/恢复 (暂停时可自由旋转定格细看姿态)"
    echo "=========================================================================="
}

CMD="${1:-compare}"

case "$CMD" in
    compare)
        echo ">>> 启动官方原生 MuJoCo 3D 交互测评 (左侧 v6 开环漂移 vs 右侧 v8 旗舰定点对决)..."
        python3 "${PROJECT_ROOT}/RL/scripts/visualize_recovery.py" --mode compare --case deep
        ;;
    v8v9)
        echo ">>> 启动 v8 vs v9 同工况并排 3D 对比（左红 v8 / 右青 v9）..."
        python3 "${PROJECT_ROOT}/RL/scripts/visualize_v8_v9.py" "${@:2}"
        ;;
    export-v8v9)
        echo ">>> 导出 v8 vs v9 对比视频与高度曲线图..."
        mkdir -p "${PROJECT_ROOT}/RL/results/stage19"
        python3 "${PROJECT_ROOT}/RL/scripts/visualize_v8_v9.py" --plot "${PROJECT_ROOT}/RL/results/stage19/v8_vs_v9_curves.png"
        python3 "${PROJECT_ROOT}/RL/scripts/visualize_v8_v9.py" --export "${PROJECT_ROOT}/RL/results/stage19/v8_vs_v9_6cases.mp4"
        ;;
    v8)
        echo ">>> 启动官方原生 MuJoCo 3D 交互测评 (v8 陀螺解耦定点旗舰单机视口)..."
        python3 "${PROJECT_ROOT}/RL/scripts/visualize_recovery.py" --mode v8 --case deep
        ;;
    v7)
        echo ">>> 启动官方原生 MuJoCo 3D 交互测评 (v7 级联定点自稳专家单机视口)..."
        python3 "${PROJECT_ROOT}/RL/scripts/visualize_recovery.py" --mode v7 --case deep
        ;;
    v6)
        echo ">>> 启动官方原生 MuJoCo 3D 交互测评 (v6 冠军控制器单机视口)..."
        python3 "${PROJECT_ROOT}/RL/scripts/visualize_recovery.py" --mode v6 --case deep
        ;;
    v5)
        echo ">>> 启动官方原生 MuJoCo 3D 交互测评 (v5 基线单机视口)..."
        python3 "${PROJECT_ROOT}/RL/scripts/visualize_recovery.py" --mode v5 --case deep
        ;;
    deep)
        echo ">>> 测评 149° 深倒扣极限工况..."
        python3 "${PROJECT_ROOT}/RL/scripts/visualize_recovery.py" --mode compare --case deep
        ;;
    fast)
        echo ">>> 测评 15.4 rad/s 高转速极限工况..."
        python3 "${PROJECT_ROOT}/RL/scripts/visualize_recovery.py" --mode compare --case fast
        ;;
    eval)
        echo ">>> 运行 512 通道官方五代全量基准测试..."
        python3 "${PROJECT_ROOT}/RL/scripts/eval_all_controllers.py"
        ;;
    verify-c)
        echo ">>> 校验 C 嵌入式头文件 recovery_expert8.h 与 Python 输出一致性..."
        python3 "${PROJECT_ROOT}/RL/scripts/verify_expert8_c_equivalence.py"
        ;;
    export)
        echo ">>> 导出全量 MuJoCo 评测高清视频与 GIF..."
        python3 "${PROJECT_ROOT}/RL/scripts/generate_mujoco_visualizations.py"
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo "未知选项: $CMD"
        show_help
        exit 1
        ;;
esac

