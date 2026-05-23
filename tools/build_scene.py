"""建立完整 MuJoCo 場景:URDF + mimic 約束 + actuator + 地板 + 方塊 + 目標區

執行:
    cd ~/dobot_project/assets/dobot
    python ~/dobot_project/tools/build_scene.py

產出:
    ~/dobot_project/assets/dobot/magician_scene.mjcf

對應 docs/agents/domain.md §6.2(mimic 用 equality 處理)。

Collision group 設定(避免 self-collision 同時讓方塊接觸地板):
  - 機械臂內部 geom: contype=2, conaffinity=1
    → 機械臂內部互不偵測,但跟 group 1 的東西會碰
  - 地板: contype=1, conaffinity=2
    → 跟機械臂(conaffinity=1)會碰,跟同類(conaffinity=2 only)不會
  - 方塊: contype=1, conaffinity=3 (= 0b11)
    → 跟機械臂(2 & 3 != 0)會碰,跟地板(1 & 3 != 0)會碰

env 依賴項(mujoco_dobot_env.py 啟動會檢查,缺則 raise):
  - test_block (freejoint)      : 方塊本體
  - target_zone (mocap body)    : stage 4-5 目標投放區,位置由 data.mocap_pos 控制
  - block_weld (WELD equality)  : stage 1-2 把方塊焊在空中,env reset 控制 active/位置
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
FLOOR_Z = -0.131            # 對應實際機械臂底部 -0.131 + 1mm 容差
BLOCK_HALF_SIZE = 0.005     # 邊長 10mm,< 夾爪最大開度 13.5mm,必須與 env 的 BLOCK_HALF_SIZE 一致
BLOCK_Z = FLOOR_Z + BLOCK_HALF_SIZE   # 方塊中心 = -0.126,底部貼地板
BLOCK_XY = (0.2, 0.0)         # 預設位置(env reset 會覆寫)
TARGET_MARKER_RADIUS = 0.03   # 目標區視覺標記半徑

# ---- Collision groups ----
ARM_CONTYPE = 2
ARM_CONAFFINITY = 1
FLOOR_CONTYPE = 1
FLOOR_CONAFFINITY = 2
BLOCK_CONTYPE = 1
BLOCK_CONAFFINITY = 3   # 0b11:跟 group 1(地板)+ group 2(機械臂)都碰

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
    print(f"  ✅ 設定 {arm_geom_count} 個機械臂 geom 為 contype={ARM_CONTYPE}, conaffinity={ARM_CONAFFINITY}")

    # ---- 6 個麥克風 site(掛在 gripper_core 上)----
    print("加 6 個 mic site...")
    end_body = spec.body("magician_link_gripper_core")

    mic_positions = [
        ("mic0", [0.000,  0.000, 0.0]),
        ("mic1", [0.004,  0.000, 0.0]),
        ("mic2", [0.012,  0.000, 0.0]),
        ("mic3", [0.028,  0.000, 0.0]),
        ("mic4", [0.000,  0.012, 0.0]),
        ("mic5", [0.000, -0.012, 0.0]),
    ]
    for name, pos in mic_positions:
        end_body.add_site(name=name, pos=pos)
        print(f"  ✅ {name} @ {pos}")

    # ---- Equality constraints(mimic)----
    print("加 equality constraints (處理 mimic)...")
    for eq_name, j1, j2, mult in MIMIC_PAIRS:
        eq = spec.add_equality()
        eq.name = eq_name
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
        eq.name1 = j1
        eq.name2 = j2
        eq.data[:5] = [0, mult, 0, 0, 0]
        # 硬化約束,避免閉環機構解算殘差過大導致末端飄
        eq.solref = [0.001, 1]
        eq.solimp = [0.99, 0.999, 0.001, 0.5, 2]
        print(f"  ✅ {eq_name}: {j1} = {mult} * {j2}")

    # ---- Actuators ----
    print("加 actuators (含 damping)...")
    actuators = [
        ("act_joint_1", "magician_joint_1", 300, -2.0944, 2.0944),
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

    print(f"  ✅ 地板 (contype={FLOOR_CONTYPE}, conaffinity={FLOOR_CONAFFINITY})")
    print(f"  ✅ 方塊 (contype={BLOCK_CONTYPE}, conaffinity={BLOCK_CONAFFINITY})")

    # ---- 目標投放區(mocap body)----
    # env line 157-162 要求 target_zone 必須是 mocap body(位置由 data.mocap_pos 控制)。
    # 純視覺標記,不參與碰撞(contype=conaffinity=0)。預設藏遠處,env reset 會設位置。
    target = wb.add_body(name="target_zone", mocap=True, pos=[5.0, 5.0, FLOOR_Z])
    target_geom = target.add_geom(
        name="target_zone_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[TARGET_MARKER_RADIUS, 0.001, 0.0],
        rgba=[0.2, 0.8, 0.2, 0.4],
    )
    target_geom.contype = 0
    target_geom.conaffinity = 0
    print("  ✅ target_zone (mocap body, 非碰撞綠色標記)")

    # ---- block_weld(WELD equality)----
    # env line 165-169 要求名為 block_weld 的 weld eq。
    # 把 test_block 焊到 world;預設 inactive,env reset 依 stage(1-2 啟用)控制 active 與焊接座標。
    # eq_data layout 由 env reset 覆寫(world anchor / block anchor / quat / torquescale)。
    weld = spec.add_equality()
    weld.name = "block_weld"
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.objtype = mujoco.mjtObj.mjOBJ_BODY
    weld.name1 = "test_block"
    weld.name2 = ""                  # 焊到 world
    weld.active = False              # 預設關閉,env 依 stage 開
    weld.data[6:10] = [1, 0, 0, 0]   # identity quat 預設(env reset 會覆寫)
    weld.data[10] = 1.0              # torquescale 預設
    print("  ✅ block_weld (WELD eq, 預設 inactive)")

    # ---- 編譯驗證 ----
    print("編譯驗證...")
    try:
        m = spec.compile()
    except Exception as e:
        print(f"❌ 編譯失敗: {e}")
        sys.exit(1)
    d = mujoco.MjData(m)
    print(f"  ✅ nu={m.nu}, neq={m.neq}, nbody={m.nbody}, njnt={m.njnt}")

    # 跑 500 步測 J1 控制
    import numpy as np
    d.ctrl[0] = 1.0
    for _ in range(500):
        mujoco.mj_step(m, d)
    j1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "magician_joint_1")
    j1_qpos = d.qpos[m.jnt_qposadr[j1_id]]
    print(f"  ✅ 500 步後 J1 qpos = {j1_qpos:.4f}(預期接近 1.0)")

    # 用真實底部算最低點(box 中心 - 半邊長)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    lowest = float('inf')
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name in ("floor", "test_block_geom", "target_zone_geom"):
            continue
        if m.geom_type[i] != 7:  # box
            continue
        bottom = d.geom_xpos[i][2] - m.geom_size[i][2]
        if bottom < lowest:
            lowest = bottom

    print(f"  ✅ 機械臂真實最低 z={lowest:.4f}, 地板 z={FLOOR_Z}, 間隙={(lowest-FLOOR_Z)*1000:.1f} mm")

    # 看初始接觸
    print(f"  ✅ 初始接觸數 ncon = {d.ncon}")
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"
        print(f"     contact: {g1} ↔ {g2}, dist={c.dist:.4f}")

    # 驗證 mic site 世界座標
    mujoco.mj_forward(m, d)
    print("  mic site 世界座標:")
    for i, name in enumerate(["mic0","mic1","mic2","mic3","mic4","mic5"]):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        print(f"    {name}: {d.site_xpos[sid]}")

    # 驗證 env 依賴項(缺則 env 啟動會 raise)
    tz = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    tz_mocap = int(m.body_mocapid[tz]) if tz >= 0 else -1
    bw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "block_weld")
    print(f"  ✅ target_zone mocapid = {tz_mocap}(需 ≥ 0)")
    print(f"  ✅ block_weld eq id = {bw}(需 ≥ 0)")
    if tz_mocap < 0 or bw < 0:
        print("  ❌ env 依賴項缺失,eval/train 會 raise")
        sys.exit(1)

    # ---- 匯出 ----
    mjcf_str = spec.to_xml()
    out_path.write_text(mjcf_str)
    print(f"\n✅ 已產出 {out_path}")
    print(f"\n下一步:")
    print(f"  python ~/dobot_project/tools/test_scene.py --viewer")


if __name__ == "__main__":
    main()
