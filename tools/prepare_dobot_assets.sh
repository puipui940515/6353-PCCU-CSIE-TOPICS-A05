#!/bin/bash
# prepare_dobot_assets.sh
#
# 用途:從 magician_ros2/dobot_description/ 複製 URDF + meshes 到 assets/dobot/,
#       並把 package:// 路徑改成 MuJoCo 認得的相對路徑。
#
# 使用:在已啟用 .venv 的 terminal 跑
#   source ~/dobot_project/setup_env.sh
#   bash ~/dobot_project/tools/prepare_dobot_assets.sh

set -e  # 任一指令失敗就停

PROJECT_ROOT="${DOBOT_PROJECT_ROOT:-$HOME/dobot_project}"
SRC_DESC="${PROJECT_ROOT}/ros2_ws/src/magician_ros2/dobot_description"
DST_DOBOT="${PROJECT_ROOT}/assets/dobot"

echo "=== Dobot assets 準備 ==="
echo "來源: ${SRC_DESC}"
echo "目標: ${DST_DOBOT}"
echo ""

# ---- 1. 建目錄 ----
mkdir -p "${DST_DOBOT}/meshes/visual" "${DST_DOBOT}/meshes/collision"

# ---- 2. 複製 URDF(從 clean_model_no_macros.urdf 開始,因為它已展開 macro)----
SRC_URDF="${SRC_DESC}/model/clean_model_no_macros.urdf"
DST_URDF="${DST_DOBOT}/magician.urdf"

if [ ! -f "${SRC_URDF}" ]; then
    echo "❌ 找不到 ${SRC_URDF}"
    exit 1
fi

cp "${SRC_URDF}" "${DST_URDF}"
echo "✅ 複製 URDF: ${DST_URDF}"

# ---- 3. 改 mesh 路徑:package://dobot_description/meshes/dae/ → meshes/visual/ ----
#         並把 .dae 副檔名換成 .obj(MuJoCo 對 DAE 支援差)
sed -i 's|package://dobot_description/meshes/dae/|meshes/visual/|g' "${DST_URDF}"
sed -i 's|\.dae|\.obj|g' "${DST_URDF}"
echo "✅ 修改 mesh 路徑(package:// → meshes/visual/,.dae → .obj)"

# ---- 4. 複製 DAE meshes(之後會轉成 OBJ)----
if [ -d "${SRC_DESC}/meshes/dae" ]; then
    cp "${SRC_DESC}/meshes/dae/"*.dae "${DST_DOBOT}/meshes/visual/" 2>/dev/null || true
    DAE_COUNT=$(ls "${DST_DOBOT}/meshes/visual/"*.dae 2>/dev/null | wc -l)
    echo "✅ 複製 ${DAE_COUNT} 個 DAE mesh 到 meshes/visual/"
fi

# ---- 5. 複製 collision STL(若有)----
if [ -d "${SRC_DESC}/meshes/collision" ]; then
    find "${SRC_DESC}/meshes/collision" -name "*.stl" -exec cp {} "${DST_DOBOT}/meshes/collision/" \; 2>/dev/null || true
    STL_COUNT=$(ls "${DST_DOBOT}/meshes/collision/"*.stl 2>/dev/null | wc -l)
    echo "✅ 複製 ${STL_COUNT} 個 STL mesh 到 meshes/collision/"
fi

# ---- 6. 列出結果 ----
echo ""
echo "=== 結果 ==="
echo "URDF: ${DST_URDF}"
echo "  行數: $(wc -l < ${DST_URDF})"
echo "  mimic joint 數: $(grep -c 'mimic' ${DST_URDF})"
echo "  mesh references: $(grep -c '<mesh' ${DST_URDF})"
echo ""
echo "下一步:"
echo "  1. 跑 DAE → OBJ 轉檔: python tools/convert_dae_to_obj.py"
echo "  2. 載入測試: python tools/test_load_urdf.py"
