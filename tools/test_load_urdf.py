"""Dobot URDF 載入測試

目的:第一次嘗試讓 MuJoCo 載入 Dobot URDF,看會炸什麼。
這一輪不處理 mimic joint(會看到 warning,正常)。

執行:
    source ~/dobot_project/setup_env.sh
    python tools/test_load_urdf.py            # 只測載入
    python tools/test_load_urdf.py --viewer   # 載入 + 開 viewer 互動
"""

from pathlib import Path
import argparse
import os
import sys

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("❌ 缺少 mujoco,請先 source setup_env.sh")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="載入後開 MuJoCo viewer")
    args = parser.parse_args()

    project_root = Path.home() / "dobot_project"
    urdf_path = project_root / "assets" / "dobot" / "magician.urdf"

    if not urdf_path.exists():
        print(f"❌ 找不到 {urdf_path},請先跑 prepare_dobot_assets.sh")
        sys.exit(1)

    print(f"嘗試載入: {urdf_path}")
    print(f"工作目錄: {urdf_path.parent}")
    print("(URDF 的 mesh 路徑是相對的,所以要 cd 進 assets/dobot/)")
    print()

    # MuJoCo 載入 URDF 時,mesh 相對路徑是相對 URDF 所在目錄
    os.chdir(urdf_path.parent)

    try:
        model = mujoco.MjModel.from_xml_path(str(urdf_path.name))
        print(f"✅ URDF 載入成功!")
        print(f"   nq (自由度): {model.nq}")
        print(f"   nv (速度維度): {model.nv}")
        print(f"   nbody (body 數): {model.nbody}")
        print(f"   ngeom (geom 數): {model.ngeom}")
        print(f"   njnt (joint 數): {model.njnt}")
        print()

        # 列出 joints
        print("   Joints:")
        for i in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            jtype = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
            print(f"     [{i}] {jname} ({jtype})")
        print()

        # 列出 bodies
        print("   Bodies:")
        for i in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            print(f"     [{i}] {bname}")
        print()

        # 試跑一個 simulation step
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        print(f"✅ 第一個 simulation step 成功")
        print()

        if args.viewer:
            print("開啟 viewer...(關閉視窗結束)")
            mujoco.viewer.launch(model, data)
        else:
            print("提示: 加 --viewer 旗標可開互動視窗")

    except Exception as e:
        print(f"❌ 載入失敗: {type(e).__name__}")
        print(f"   {e}")
        print()
        print("常見原因:")
        print("  1. mesh 檔案找不到 → 確認 meshes/visual/ 內有 .obj 檔")
        print("  2. mimic joint 不支援 → 預期會炸,下一輪處理")
        print("  3. 其他語法錯誤 → 把錯誤訊息貼給 Claude")
        sys.exit(1)


if __name__ == "__main__":
    main()
