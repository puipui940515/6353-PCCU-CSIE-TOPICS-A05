"""DAE → OBJ 轉檔器

MuJoCo 對 COLLADA (DAE) 支援很差(常見 normal / texture 解析失敗),
轉成 OBJ 最穩。本腳本用 trimesh 批次轉換。

執行前:
    pip install trimesh

執行:
    python tools/convert_dae_to_obj.py
"""

from pathlib import Path
import sys

try:
    import trimesh
except ImportError:
    print("❌ 缺少 trimesh,請先安裝:")
    print("   pip install trimesh")
    sys.exit(1)


def main() -> None:
    project_root = Path.home() / "dobot_project"
    visual_dir = project_root / "assets" / "dobot" / "meshes" / "visual"

    if not visual_dir.exists():
        print(f"❌ 找不到 {visual_dir},請先跑 prepare_dobot_assets.sh")
        sys.exit(1)

    dae_files = list(visual_dir.glob("*.dae"))
    if not dae_files:
        print(f"⚠️  {visual_dir} 沒有 DAE 檔案")
        return

    print(f"找到 {len(dae_files)} 個 DAE 檔案,開始轉換...")

    success_count = 0
    fail_count = 0
    for dae_path in dae_files:
        obj_path = dae_path.with_suffix(".obj")
        try:
            mesh = trimesh.load(str(dae_path), force="mesh")
            mesh.export(str(obj_path))
            print(f"  ✅ {dae_path.name} → {obj_path.name}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ {dae_path.name}: {e}")
            fail_count += 1

    print(f"\n完成: {success_count} 成功, {fail_count} 失敗")

    if fail_count == 0:
        print("\n下一步: python tools/test_load_urdf.py")


if __name__ == "__main__":
    main()
