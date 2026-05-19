"""建立完整 MuJoCo 場景:URDF + mimic 約束 + actuator + 地板 + 方塊 + 目標區

執行:
    cd ~/dobot_project/assets/dobot
    python ~/dobot_project/tools/build_scene.py

產出:
    ~/dobot_project/assets/dobot/magician_scene.mjcf

對應 docs/agents/domain.md §5.5(Curriculum)、§6.2(mimic 用 equality 處理)。

關鍵設計:
  1. block 一律用 freejoint + 重力(stage 3+ 標準狀態)
  2. 額外加一個 weld equality(預設 inactive),env 在 stage 1-2 reset 時動態啟用,
     把 block 焊在目標位置,行為等同 mocap。stage 3+ 保持 inactive 走真重力。
  3. 綠色目標區 plane(stage 4 視覺標記),contype=0/conaffinity=0,純標記不碰撞,
     env runtime 改 geom_pos 把它挪到隨機位置。

Collision group:
  - 機械臂內部 geom: contype=2, conaffinity=1
  - 地板: contype=1, conaffinity=2
  - 方塊: contype=1, conaffinity=3 (= 0b11)
  - 目標區: contype=0, conaffinity=0(純視覺)
"""

from pathlib import Path
import sys
import os

try:
    import mujoco
except ImportError:
    print("❌ 缺少 mujoco,請 source setup_env.sh")
    sys.exit(1)


# ---- 場景常數 ----
FLOOR_Z = -0.131
BLOCK_HALF_SIZE = 0.005   # 邊長 10 mm,小於夾爪最大開度 13.5 mm
BLOCK_Z = FLOOR_Z + BLOCK_HALF_SIZE
BLOCK_XY = (0.2, 0.0)

# 目標區(stage 4 用)
TARGET_ZONE_HALF_SIZE = 0.04           # 半邊長 4 cm
TARGET_ZONE_THICKNESS = 0.001          # 薄薄一片,純視覺
TARGET_ZONE_XY = (0.2, 0.1)            # 預設位置(env reset 會覆寫)
TARGET_ZONE_Z = FLOOR_Z + TARGET_ZONE_THICKNESS / 2 + 0.0005  # 略浮一點點避免 Z-fighting

# ---- Collision groups ----
ARM_CONTYPE = 2
ARM_CONAFFINITY = 1
FLOOR_CONTYPE = 1
FLOOR_CONAFFINITY = 2
BLOCK_CONTYPE = 1
BLOCK_CONAFFINITY = 3
TARGET_CONTYPE = 0      # 純視覺,完全不參與碰撞
TARGET_CONAFFINITY = 0

# ---- Joint range(取自 URDF)----
JOINT_RANGES = {
    "magician_joint_2": (-0.0873, 1.5708),
    "magician_joint_3": (-0.2618, 1.2217),
    "magician_joint_prismatic_l": (0.0, 0.0135),
}

# ---- Mimic 對應 ----
MIMIC_PAIRS = [
    ("parallel_link_1", "magician_joint_mimic_1", "magician_joint_2", -1.0),
    ("parallel_link_2", "magician_joint_mimic_2", "magician_joint_3", -1.0),
    ("gripper_sync",    "magician_joint_prismatic_r", "magician_joint_prismatic_l", -1.0),
]


def main() -> None:
    project_root = Path.home() / "dobot_project"
    assets_dir = project_root / "assets" / "dobot"
    urdf_path = assets_dir / "magician.urdf"
    out_path = assets_dir / "magician_scene.mjcf"

    if not urdf_path.exists():
        print(f"❌ 找不到 {urdf_path}")
        sys.exit(1)

    os.chdir(assets_dir)

    print(f"讀取 URDF: {urdf_path.name}")
    spec = mujoco.MjSpec.from_file(str(urdf_path.name))

    spec.option.timestep = 0.002
    spec.option.iterations = 50
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # 重力預設啟用(stage 3+ 標準),stage 1-2 由 env 用 weld 把 block 釘住模擬無重力
    spec.option.gravity = [0, 0, -9.81]

    # ---- 修 mimic / prismatic joint range ----
    print("修正 mimic / prismatic joint range...")
    for eq_name, mimic_name, source_name, mult in MIMIC_PAIRS:
        src_range = JOINT_RANGES[source_name]
        new_lo = min(mult * src_range[0], mult * src_range[1])
        new_hi = max(mult * src_range[0], mult * src_range[1])
        try:
            j = spec.joint(mimic_name)
            j.range = [new_lo, new_hi]
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            print(f"  ✅ {mimic_name}: range → [{new_lo:.4f}, {new_hi:.4f}]")
        except Exception as e:
            print(f"  ⚠️  {mimic_name}: {e}")

    # ---- 停用機械臂 self-collision ----
    print("停用機械臂內部 self-collision...")
    arm_geom_count = 0
    for geom in spec.geoms:
        geom.contype = ARM_CONTYPE
        geom.conaffinity = ARM_CONAFFINITY
        arm_geom_count += 1
    print(f"  ✅ 設定 {arm_geom_count} 個機械臂 geom")

    # ---- Equality constraints (mimic) ----
    # 重要:必須用 hard constraint,否則 mimic equality 會被 actuator 的 PD 力量壓掉。
    # 預設 solref=[0.02, 1.0]、solimp=[0.9, 0.95, ...] 太軟,會出現:
    #   J2=0.006、mimic_1=0.39(應該 = -0.006)的失敗案例。
    # 改成 solref[0]=0.001、solimp 接近 1 後,殘差 < 0.002 rad(實務可接受)。
    print("加 equality constraints (處理 mimic, hard constraint)...")
    for eq_name, j1, j2, mult in MIMIC_PAIRS:
        eq = spec.add_equality()
        eq.name = eq_name
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
        eq.name1 = j1
        eq.name2 = j2
        eq.data[:5] = [0, mult, 0, 0, 0]
        eq.solref[:] = [0.001, 1.0]
        eq.solimp[:] = [0.99, 0.999, 0.001, 0.5, 2]
        print(f"  ✅ {eq_name}: {j1} = {mult} * {j2}")

    # ---- Actuators ----
    print("加 actuators...")
    actuators = [
        ("act_joint_1", "magician_joint_1", 50, -2.0944, 2.0944),
        ("act_joint_2", "magician_joint_2", 50, 0.0, 1.4835),
        ("act_joint_3", "magician_joint_3", 50, -0.2618, 1.2217),
        ("act_joint_4", "magician_joint_4", 20, -1.5708, 1.5708),
        ("act_gripper", "magician_joint_prismatic_l", 100, 0.0, 0.0135),
    ]
    for name, joint, kp, lo, hi in actuators:
        kv = kp / 5
        a = spec.add_actuator()
        a.name = name
        a.target = joint
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gainprm[0] = kp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[0] = 0
        a.biasprm[1] = -kp
        a.biasprm[2] = -kv
        a.ctrlrange = [lo, hi]
        a.ctrllimited = True
        print(f"  ✅ {name} → {joint} (kp={kp}, kv={kv})")

    # ---- 場景 ----
    print(f"加場景 (地板 z={FLOOR_Z:.3f}, 方塊 xy={BLOCK_XY} z={BLOCK_Z:.3f})...")
    wb = spec.worldbody

    wb.add_light(pos=[0, 0, 2], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    wb.add_light(pos=[1, 1, 2], dir=[-0.5, -0.5, -1], diffuse=[0.4, 0.4, 0.4])

    floor = wb.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, FLOOR_Z],
        size=[1, 1, 0.1],
        rgba=[0.7, 0.7, 0.7, 1],
        friction=[1.0, 0.005, 0.0001],
    )
    floor.contype = FLOOR_CONTYPE
    floor.conaffinity = FLOOR_CONAFFINITY

    # ---- 方塊 (freejoint + 重力,stage 1-2 由 env weld 釘住) ----
    block = wb.add_body(
        name="test_block",
        pos=[BLOCK_XY[0], BLOCK_XY[1], BLOCK_Z],
    )
    block.add_freejoint(name="test_block_freejoint")
    block_geom = block.add_geom(
        name="test_block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BLOCK_HALF_SIZE, BLOCK_HALF_SIZE, BLOCK_HALF_SIZE],
        mass=0.05,
        rgba=[0.9, 0.3, 0.3, 1],
        friction=[1.0, 0.005, 0.0001],
    )
    block_geom.contype = BLOCK_CONTYPE
    block_geom.conaffinity = BLOCK_CONAFFINITY

    # ---- 目標區(stage 4,純視覺薄綠色板)----
    target = wb.add_body(
        name="target_zone",
        pos=[TARGET_ZONE_XY[0], TARGET_ZONE_XY[1], TARGET_ZONE_Z],
        mocap=True,   # 用 mocap body,env 透過 mocap_pos 動態挪位置(比改 geom 容易)
    )
    target_geom = target.add_geom(
        name="target_zone_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TARGET_ZONE_HALF_SIZE, TARGET_ZONE_HALF_SIZE, TARGET_ZONE_THICKNESS / 2],
        rgba=[0.2, 0.9, 0.3, 0.5],  # 半透明綠
    )
    target_geom.contype = TARGET_CONTYPE
    target_geom.conaffinity = TARGET_CONAFFINITY

    # ---- Weld equality(stage 1-2 用,預設 inactive)----
    # 把 block freejoint 焊在 world frame,等同 mocap。
    # env 在 stage 1-2 開啟此 constraint,在 reset 時改 eq_data 設定目標位姿。
    # Layout(11 維):anchor1(3) + anchor2(3) + relpose_quat(4) + torquescale(1)
    weld = spec.add_equality()
    weld.name = "block_weld"
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.objtype = mujoco.mjtObj.mjOBJ_BODY
    weld.name1 = "test_block"
    weld.name2 = "world"
    weld.active = False  # 預設關,env 在 stage 1-2 動態開啟
    # 預設 identity quat + torquescale=1,env runtime 改 anchor2(焊接位置)
    weld.data[:] = [0, 0, 0,   # anchor1(test_block frame,用 body 中心)
                    0, 0, 0,   # anchor2(world frame,env runtime 覆寫)
                    1, 0, 0, 0, # relpose quat
                    1]          # torquescale
    print("  ✅ 目標區(綠色板,純視覺) + block_weld equality(預設 inactive)")

    # ---- 編譯驗證 ----
    print("編譯驗證...")
    try:
        m = spec.compile()
    except Exception as e:
        print(f"❌ 編譯失敗: {e}")
        sys.exit(1)
    d = mujoco.MjData(m)
    print(f"  ✅ nu={m.nu}, neq={m.neq}, nbody={m.nbody}, njnt={m.njnt}")

    # 確認 weld 找得到
    weld_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "block_weld")
    print(f"  ✅ block_weld eq_id = {weld_id}(env 將用此 id runtime 啟用)")

    # 確認 target mocap id
    target_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    target_mocap = m.body_mocapid[target_bid]
    print(f"  ✅ target_zone mocap_id = {target_mocap}")

    # 跑 500 步測 J1 控制
    import numpy as np
    d.ctrl[0] = 1.0
    for _ in range(500):
        mujoco.mj_step(m, d)
    j1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "magician_joint_1")
    j1_qpos = d.qpos[m.jnt_qposadr[j1_id]]
    print(f"  ✅ 500 步後 J1 qpos = {j1_qpos:.4f}")

    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    print(f"  ✅ 初始接觸數 ncon = {d.ncon}")

    # ---- 匯出 ----
    mjcf_str = spec.to_xml()
    out_path.write_text(mjcf_str)
    print(f"\n✅ 已產出 {out_path}")
    print(f"\n下一步:")
    print(f"  python ~/dobot_project/tools/test_scene.py --viewer")


if __name__ == "__main__":
    main()
