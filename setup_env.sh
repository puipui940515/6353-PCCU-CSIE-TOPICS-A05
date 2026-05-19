#!/bin/bash
# Dobot Magician 聲波震動定位專題 · 開發環境啟動腳本
#
# 使用方式:
#   source ~/dobot_project/setup_env.sh           # 預設:ROS 2 + venv 全載(訓練、跑 Python 用)
#   source ~/dobot_project/setup_env.sh --no-py   # 只載 ROS 2,跳過 venv(colcon build 用)
#
# 為何要雙模式?
#   colcon build 會呼叫 python3 跑內部 script(catkin_pkg 等),
#   若 venv 已 active,python3 指向 venv 而非系統 Python,catkin_pkg 找不到 → build 失敗。
#   所以 colcon build 前要 deactivate 或用 --no-py 啟動。
#
# 對應文件:docs/agents/domain.md §2、deployment.md §1

# ---- 防呆:確認用 source 執行 ----
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "❌ 請用 source 執行此腳本,不要直接執行:"
    echo "   source ~/dobot_project/setup_env.sh"
    exit 1
fi

# ---- 解析參數 ----
LOAD_VENV=1
for arg in "$@"; do
    case "$arg" in
        --no-py)
            LOAD_VENV=0
            ;;
        --help|-h)
            echo "用法:"
            echo "  source setup_env.sh           # ROS 2 + venv 全載(預設)"
            echo "  source setup_env.sh --no-py   # 只載 ROS 2(colcon build 用)"
            return 0
            ;;
        *)
            echo "⚠️  未知參數: $arg(使用 --help 看用法)"
            ;;
    esac
done

# ---- 專案根目錄(以本腳本位置推算) ----
DOBOT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DOBOT_PROJECT_ROOT

# ---- 若 venv 已 active,先 deactivate(避免雙重 source 衝突) ----
if [ -n "${VIRTUAL_ENV}" ] && command -v deactivate >/dev/null 2>&1; then
    deactivate
fi

# ---- 1. ROS 2 Humble ----
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo "❌ /opt/ros/humble/setup.bash 不存在,請先安裝 ROS 2 Humble"
    return 1
fi

# ---- 2. ROS 2 workspace(若已 colcon build) ----
if [ -f "${DOBOT_PROJECT_ROOT}/ros2_ws/install/setup.bash" ]; then
    source "${DOBOT_PROJECT_ROOT}/ros2_ws/install/setup.bash"
else
    echo "⚠️  ros2_ws/install/setup.bash 不存在,尚未 colcon build"
fi

# ---- 3. Python venv(可用 --no-py 跳過) ----
if [ "${LOAD_VENV}" = "1" ]; then
    if [ -f "${DOBOT_PROJECT_ROOT}/.venv/bin/activate" ]; then
        source "${DOBOT_PROJECT_ROOT}/.venv/bin/activate"
    else
        echo "❌ .venv 不存在,請先建立:"
        echo "   cd ${DOBOT_PROJECT_ROOT}"
        echo "   python3.10 -m venv .venv"
        echo "   source .venv/bin/activate"
        echo "   pip install -U pip wheel"
        echo "   pip install -r requirements.txt"
        return 1
    fi
fi

# ---- 4. 環境驗證 ----
if [ "${LOAD_VENV}" = "1" ]; then
    python3 -c "import mujoco, rclpy" 2>/dev/null && \
        echo "✅ Dobot 開發環境就緒 (ROS 2 + ros2_ws + .venv + MuJoCo) · 模式: 全載" || \
        echo "❌ 環境驗證失敗,缺少套件(mujoco / rclpy)"
else
    python3 -c "import rclpy" 2>/dev/null && \
        echo "✅ Dobot 開發環境就緒 (ROS 2 + ros2_ws) · 模式: 僅 ROS 2(colcon build 安全)" || \
        echo "❌ 環境驗證失敗,缺少 rclpy"
fi

# ---- 5. 快捷指令提示 ----
echo ""
echo "專案根目錄: ${DOBOT_PROJECT_ROOT}"
echo ""
if [ "${LOAD_VENV}" = "1" ]; then
    echo "常用指令:"
    echo "  python -m mujoco.viewer                              # MuJoCo 互動 viewer"
    echo "  tensorboard --logdir \$DOBOT_PROJECT_ROOT/runs/      # 訓練曲線"
    echo ""
    echo "要 colcon build 時:"
    echo "  source \$DOBOT_PROJECT_ROOT/setup_env.sh --no-py"
    echo "  cd \$DOBOT_PROJECT_ROOT/ros2_ws && colcon build --symlink-install"
else
    echo "常用指令:"
    echo "  cd \$DOBOT_PROJECT_ROOT/ros2_ws && colcon build --symlink-install"
    echo ""
    echo "build 完要訓練時:"
    echo "  source \$DOBOT_PROJECT_ROOT/setup_env.sh"
fi
