"""載入 magician_scene.mjcf 並開 MuJoCo viewer

執行:
    python ~/dobot_project/tools/test_scene.py            # 載入測試
    python ~/dobot_project/tools/test_scene.py --viewer   # 開互動視窗
"""

from pathlib import Path
import argparse
import os
import sys

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("❌ 缺少 mujoco,請 source setup_env.sh")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    project_root = Path.home() / "dobot_project"
    mjcf_path = project_root / "assets" / "dobot" / "magician_scene.mjcf"

    if not mjcf_path.exists():
        print(f"❌ 找不到 {mjcf_path}")
        print("   請先跑: python ~/dobot_project/tools/build_scene.py")
        sys.exit(1)

    os.chdir(mjcf_path.parent)
    print(f"載入: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path.name)
    data = mujoco.MjData(model)
    print(f"✅ 載入成功 (nu={model.nu}, neq={model.neq}, nbody={model.nbody})")

    print("Joints:")
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  [{i}] {jname}")
    print("Actuators:")
    for i in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {aname}")

    if args.viewer:
        print("\n開啟 viewer...(關閉視窗結束)")
        print("操作:")
        print("  Space         開始/暫停")
        print("  Ctrl + 右鍵   拖曳關節")
        print("  滑鼠           旋轉/縮放/平移視角")
        mujoco.viewer.launch(model, data)
    else:
        # 跑 100 步看會不會炸
        for _ in range(100):
            mujoco.mj_step(model, data)
        print(f"\n✅ 100 步 simulation 成功")
        print("提示: 加 --viewer 旗標可開互動視窗")


if __name__ == "__main__":
    main()
