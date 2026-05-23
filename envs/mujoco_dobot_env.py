"""MuJoCoDobotEnv:Phase 1a curriculum 訓練用的 MuJoCo 環境

對應 docs/agents/domain.md §5.4(stage-aware reward)、§5.5(Curriculum 5 stage)。

Stage 1:對準方塊                    (mocap-like via weld,無重力效果)
Stage 2:抓取方塊(夾爪閉合)        (同上)
Stage 3:抓起 + 握住                 (freejoint + 重力)
Stage 4:放置到目標區                (freejoint + 重力 + 目標區順序狀態機)
Stage 5:形狀泛化                    (同 4,物體形狀隨機;尚未實作物體切換)
"""

from __future__ import annotations
from pathlib import Path
import os

import mujoco
import numpy as np

from .base_dobot_env import BaseDobotEnv, N_JOINTS, ACTION_DIM


POLICY_HZ = 50
PHYSICS_HZ = 500
SUBSTEPS = PHYSICS_HZ // POLICY_HZ

BLOCK_HALF_SIZE = 0.005   # 邊長 10 mm,小於夾爪最大開度 13.5 mm
GRASP_DISTANCE_THRESHOLD = 0.03
LIFT_THRESHOLD = 0.05            # stage 3+:方塊離地多少算「抓起」
PLACE_LANDED_TOLERANCE = 0.005   # stage 4:方塊 z 距地板多少算「落地」
GRIPPER_CLOSED_THRESHOLD = 0.3   # gripper_open 正規化值,< 此為閉合
RELEASE_THRESHOLD = 0.7          # > 此為張開
TARGET_ZONE_HALF_SIZE = 0.04     # 與 build_scene.py 對齊

# 工作空間
ARM_REACH = 0.32
BLOCK_RADIUS_RANGE = (0.20, 0.28)
BLOCK_ANGLE_RANGE = (-np.pi, np.pi)
BLOCK_MAX_HEIGHT = 0.20

# Stage 3+ 方塊初始懸空高度(讓它從這裡自由落下)
BLOCK_INITIAL_HOVER_HEIGHT = 0.02

# Stage 1-2 方塊浮空高度範圍(weld 焊住,不掉)
STAGE12_BLOCK_HEIGHT_RANGE = (0.02, 0.20)

# Stage 4 目標區隨機
TARGET_RADIUS_RANGE = (0.20, 0.28)
TARGET_ANGLE_RANGE = (-np.pi, np.pi)
MIN_BLOCK_TARGET_DISTANCE = 0.08
MAX_RESAMPLE_ATTEMPTS = 20

JOINT_VEL_SCALE_PER_JOINT = np.array([5.0, 3.0, 3.0, 3.0], dtype=np.float32)
GRIPPER_VEL_SCALE = 0.05
GRIPPER_MAX_OPEN = 0.0135

# ============================================================
# Reward 預設權重(與 domain.md §5.4 表格對齊)
# 各 stage config (configs/sac_stageN.yaml) 可覆寫
# ============================================================
DEFAULT_REWARD_WEIGHTS = {
    # 通用(W_HORIZONTAL/W_VERTICAL 已合併為 W_DISTANCE:直線歐氏距離,各方向等權)
    "W_DISTANCE": 1.5,
    "W_PROGRESS": 5.0,
    "W_ACTION_SMOOTH": 0.001,
    "W_TIME": 0.01,
    "R_TIMEOUT": -1.0,
    # Stage 1
    "R_SUCCESS_ALIGN": 5.0,
    # Stage 2
    "W_GRIPPER_CLOSE": 0.1,
    "GRIPPER_SHAPING_RADIUS": 0.05,
    "R_SUCCESS_GRASP": 10.0,
    # Stage 3(W_LIFT_HEIGHT:potential-based,只在握住時獎勵抬升、懲罰下降,握平穩為 0)
    "R_SUCCESS_LIFT": 10.0,
    "W_HOLDING": 0.05,
    "W_LIFT_HEIGHT": 2.0,
    # Stage 4
    "W_DIST_TO_TARGET": 2.0,
    "R_SUCCESS_PLACE": 20.0,
    "R_SUCCESS_RELEASE": 20.0,
}


# ---- 放置狀態機(stage 4)----
TASK_STATE_INITIAL = 0
TASK_STATE_GRASPED = 1
TASK_STATE_PLACED = 2
TASK_STATE_RELEASED = 3  # = SUCCESS


class MuJoCoDobotEnv(BaseDobotEnv):

    def __init__(
        self,
        mjcf_path: str | None = None,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
        stage: int = 1,
        reward_weights: dict | None = None,
    ) -> None:
        super().__init__()

        if stage not in (1, 2, 3, 4, 5):
            raise ValueError(f"stage 必須是 1~5,收到 {stage}")
        self.stage = stage

        # 合併 reward 權重
        self.w = dict(DEFAULT_REWARD_WEIGHTS)
        if reward_weights:
            self.w.update(reward_weights)

        if mjcf_path is None:
            mjcf_path = str(
                Path.home() / "dobot_project" / "assets" / "dobot" / "magician_scene.mjcf"
            )
        self.mjcf_path = mjcf_path

        original_cwd = os.getcwd()
        os.chdir(Path(mjcf_path).parent)
        try:
            self.model = mujoco.MjModel.from_xml_path(Path(mjcf_path).name)
        finally:
            os.chdir(original_cwd)

        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.render_mode = render_mode
        self._viewer = None

        # ---- 索引快取 ----
        self._joint_qpos_addr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"magician_joint_{i+1}")]
            for i in range(N_JOINTS)
        ]
        self._joint_qvel_addr = [
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"magician_joint_{i+1}")]
            for i in range(N_JOINTS)
        ]
        self._gripper_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "magician_joint_prismatic_l")
        ]

        self._tcp_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "magician_link_gripper_core"
        )
        self._block_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "test_block"
        )
        self._block_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "test_block_freejoint"
        )
        self._block_qpos_addr = self.model.jnt_qposadr[self._block_jid]
        self._block_qvel_addr = self.model.jnt_dofadr[self._block_jid]

        # 目標區 mocap
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone"
        )
        self._target_mocap_id = int(self.model.body_mocapid[self._target_body_id])
        if self._target_mocap_id < 0:
            raise RuntimeError("target_zone 不是 mocap body,build_scene.py 有問題")

        # Weld constraint(stage 1-2 用)
        self._weld_eq_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "block_weld"
        )
        if self._weld_eq_id < 0:
            raise RuntimeError("找不到 block_weld equality,build_scene.py 有問題")

        # 地板高度從 model 抓
        floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._floor_z = float(self.model.geom_pos[floor_gid, 2])

        self._act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_joint_{i+1}")
            for i in range(N_JOINTS)
        ] + [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
        ]

        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._prev_distance = None
        self._prev_block_height = None

        # Stage 3+ 用:首次抓起與 stage 4 狀態機
        self._has_lifted = False
        self._task_state = TASK_STATE_INITIAL

        # 當前 episode 的方塊與目標區 xyz(reset 時填)
        self._block_xyz = np.zeros(3, dtype=np.float64)
        self._target_xyz = np.zeros(3, dtype=np.float64)

    # ============================================================
    # Stage 設定
    # ============================================================

    def _stage_uses_weld(self) -> bool:
        """Stage 1-2 用 weld 把方塊釘住(模擬 mocap)"""
        return self.stage in (1, 2)

    def _stage_uses_target(self) -> bool:
        """Stage 4-5 啟用目標區"""
        return self.stage in (4, 5)

    # ============================================================
    # Reset
    # ============================================================

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # ---- 取樣方塊位置 ----
        block_xyz = self._sample_xyz_in_workspace(rng)

        # ---- 取樣目標區位置(stage 4-5)----
        if self._stage_uses_target():
            target_xyz = self._sample_target_xyz(rng, block_xyz)
        else:
            # 其他 stage 把目標區藏到遠處(視覺上看不到)
            target_xyz = np.array([5.0, 5.0, self._floor_z], dtype=np.float64)

        # ---- 設定方塊位置(freejoint qpos[0:3])----
        self.data.qpos[self._block_qpos_addr:self._block_qpos_addr+3] = block_xyz
        self.data.qpos[self._block_qpos_addr+3:self._block_qpos_addr+7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._block_qvel_addr:self._block_qvel_addr+6] = 0.0

        # ---- 設定目標區位置(mocap)----
        self.data.mocap_pos[self._target_mocap_id] = target_xyz
        self.data.mocap_quat[self._target_mocap_id] = [1.0, 0.0, 0.0, 0.0]

        # ---- Weld constraint:stage 1-2 啟用 ----
        if self._stage_uses_weld():
            # MuJoCo weld eq_data layout(11 維):
            #   [0:3]   anchor (在 body2=world frame 的位置)← 焊接的世界座標
            #   [3:6]   anchor (在 body1=test_block frame 的位置)← 用 body 中心
            #   [6:10]  relpose quat(wxyz)
            #   [10]    torquescale
            # 注意:MuJoCo 的 weld 文件裡 anchor1/anchor2 對 body1/body2 的對應與直覺相反,
            # 實測:把 world frame 的目標座標放 data[0:3],body 中心 (0,0,0) 放 data[3:6]。
            self.model.eq_data[self._weld_eq_id, 0:3] = block_xyz   # world frame: 目標焊接位置
            self.model.eq_data[self._weld_eq_id, 3:6] = [0, 0, 0]   # block frame: body 中心
            self.model.eq_data[self._weld_eq_id, 6:10] = [1, 0, 0, 0]
            self.model.eq_data[self._weld_eq_id, 10] = 1.0
            self.model.eq_active0[self._weld_eq_id] = 1
            self.data.eq_active[self._weld_eq_id] = 1
        else:
            self.model.eq_active0[self._weld_eq_id] = 0
            self.data.eq_active[self._weld_eq_id] = 0

        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # ---- 狀態重置 ----
        self.current_step = 0
        self._last_action[:] = 0
        self._prev_distance = None
        self._prev_block_height = None
        self._has_lifted = False
        self._task_state = TASK_STATE_INITIAL
        self._block_xyz = block_xyz.copy()
        self._target_xyz = target_xyz.copy()

        return self._build_obs(), {
            "stage": self.stage,
            "block_xyz": tuple(block_xyz),
            "target_xyz": tuple(target_xyz),
            "task_state": self._task_state,
        }

    def _sample_xyz_in_workspace(self, rng: np.random.Generator) -> np.ndarray:
        """工作空間內均勻取樣方塊位置

        - Stage 1-2:方塊浮在空中(weld 焊住,不掉),高度範圍 STAGE12_BLOCK_HEIGHT_RANGE
        - Stage 3+:方塊離地 BLOCK_INITIAL_HOVER_HEIGHT 起跳,自由落下(物理引擎接管)
        """
        r_min, r_max = BLOCK_RADIUS_RANGE
        radius = float(np.sqrt(rng.uniform(r_min**2, r_max**2)))
        angle = rng.uniform(*BLOCK_ANGLE_RANGE)
        bx = radius * float(np.cos(angle))
        by = radius * float(np.sin(angle))

        if self.stage in (1, 2):
            # 浮空隨機高度(weld 會把它焊在這裡)
            h_min, h_max = STAGE12_BLOCK_HEIGHT_RANGE
            # 額外限制:不超出臂展球殼
            vertical_reach = float(np.sqrt(max(ARM_REACH**2 - radius**2, 0.0)))
            h_max = min(h_max, vertical_reach)
            height = rng.uniform(h_min, max(h_max, h_min))
            bz = self._floor_z + BLOCK_HALF_SIZE + height
        else:
            # Stage 3+:離地 2 cm 起跳,自由落下
            bz = self._floor_z + BLOCK_HALF_SIZE + BLOCK_INITIAL_HOVER_HEIGHT

        return np.array([bx, by, bz], dtype=np.float64)

    def _sample_target_xyz(self, rng: np.random.Generator, block_xyz: np.ndarray) -> np.ndarray:
        """取樣目標區位置,確保與方塊距離 > MIN_BLOCK_TARGET_DISTANCE"""
        r_min, r_max = TARGET_RADIUS_RANGE
        for _ in range(MAX_RESAMPLE_ATTEMPTS):
            radius = float(np.sqrt(rng.uniform(r_min**2, r_max**2)))
            angle = rng.uniform(*TARGET_ANGLE_RANGE)
            tx = radius * float(np.cos(angle))
            ty = radius * float(np.sin(angle))
            tz = self._floor_z + 0.001  # 略浮一點點
            d_xy = float(np.hypot(tx - block_xyz[0], ty - block_xyz[1]))
            if d_xy >= MIN_BLOCK_TARGET_DISTANCE:
                return np.array([tx, ty, tz], dtype=np.float64)
        # Fallback:取最後一次,不卡住訓練
        return np.array([tx, ty, tz], dtype=np.float64)

    # ============================================================
    # Step
    # ============================================================

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        current_joint_pos = np.array([self.data.qpos[a] for a in self._joint_qpos_addr])
        current_gripper = self.data.qpos[self._gripper_qpos_addr]

        dt = 1.0 / POLICY_HZ
        target_joint_pos = current_joint_pos + action[:N_JOINTS] * JOINT_VEL_SCALE_PER_JOINT * dt
        target_gripper = current_gripper + action[N_JOINTS] * GRIPPER_VEL_SCALE * dt

        for i, act_id in enumerate(self._act_ids[:N_JOINTS]):
            lo, hi = self.model.actuator_ctrlrange[act_id]
            self.data.ctrl[act_id] = float(np.clip(target_joint_pos[i], lo, hi))
        lo, hi = self.model.actuator_ctrlrange[self._act_ids[-1]]
        self.data.ctrl[self._act_ids[-1]] = float(np.clip(target_gripper, lo, hi))

        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        if self._viewer is not None:
            self._viewer.sync()

        reward, success, reward_info = self._compute_reward(action)
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        terminated = success

        if truncated and not success:
            reward += self.w["R_TIMEOUT"]

        self._last_action = action

        info = {
            "success": success,
            "step": self.current_step,
            "stage": self.stage,
            "task_state": self._task_state,
            "block_pose": self._get_block_pose(),
            "target_xyz": tuple(self._target_xyz),
            **reward_info,
        }
        return self._build_obs(), float(reward), bool(terminated), bool(truncated), info

    # ============================================================
    # Observation
    # ============================================================

    def _build_obs(self) -> dict:
        joint_pos = np.array([self.data.qpos[a] for a in self._joint_qpos_addr], dtype=np.float32)
        joint_vel = np.array([self.data.qvel[a] for a in self._joint_qvel_addr], dtype=np.float32)
        tcp_xyz = self.data.xpos[self._tcp_body_id]
        tcp_quat = self.data.xquat[self._tcp_body_id]
        tcp_pose = np.concatenate([tcp_xyz, tcp_quat]).astype(np.float32)
        gripper = np.array([self.data.qpos[self._gripper_qpos_addr]], dtype=np.float32)

        return {
            "joint_position": joint_pos,
            "joint_velocity": joint_vel,
            "tcp_pose": tcp_pose,
            "gripper_state": gripper,
            "block_pose": self._get_block_pose(),
        }

    def _get_block_pose(self) -> np.ndarray:
        return np.concatenate([
            self.data.xpos[self._block_body_id],
            self.data.xquat[self._block_body_id],
        ]).astype(np.float32)

    # ============================================================
    # Reward dispatch
    # ============================================================

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool, dict]:
        if self.stage == 1:
            return self._reward_stage1(action)
        elif self.stage == 2:
            return self._reward_stage2(action)
        elif self.stage == 3:
            return self._reward_stage3(action)
        elif self.stage in (4, 5):
            return self._reward_stage4(action)
        else:
            raise ValueError(f"Unknown stage {self.stage}")

    # ---- Stage 1:對準 ----
    def _reward_stage1(self, action: np.ndarray) -> tuple[float, bool, dict]:
        tcp_xyz = self.data.xpos[self._tcp_body_id]
        block_xyz = self.data.xpos[self._block_body_id]
        distance = float(np.linalg.norm(tcp_xyz - block_xyz))

        reward = -self.w["W_DISTANCE"] * distance
        if self._prev_distance is not None:
            reward += self.w["W_PROGRESS"] * (self._prev_distance - distance)
        self._prev_distance = distance

        reward -= self.w["W_ACTION_SMOOTH"] * float(np.sum((action - self._last_action) ** 2))
        reward -= self.w["W_TIME"]

        success = distance < GRASP_DISTANCE_THRESHOLD
        if success:
            reward += self.w["R_SUCCESS_ALIGN"]

        return reward, success, {"distance": distance}

    # ---- Stage 2:抓取(對準 + 夾爪閉合) ----
    def _reward_stage2(self, action: np.ndarray) -> tuple[float, bool, dict]:
        tcp_xyz = self.data.xpos[self._tcp_body_id]
        block_xyz = self.data.xpos[self._block_body_id]
        distance = float(np.linalg.norm(tcp_xyz - block_xyz))

        reward = -self.w["W_DISTANCE"] * distance
        if self._prev_distance is not None:
            reward += self.w["W_PROGRESS"] * (self._prev_distance - distance)
        self._prev_distance = distance

        gripper_open = float(self.data.qpos[self._gripper_qpos_addr]) / GRIPPER_MAX_OPEN
        if distance < self.w["GRIPPER_SHAPING_RADIUS"]:
            reward += self.w["W_GRIPPER_CLOSE"] * (1.0 - gripper_open)

        reward -= self.w["W_ACTION_SMOOTH"] * float(np.sum((action - self._last_action) ** 2))
        reward -= self.w["W_TIME"]

        close_to_block = distance < GRASP_DISTANCE_THRESHOLD
        gripper_closed = gripper_open < GRIPPER_CLOSED_THRESHOLD
        success = bool(close_to_block and gripper_closed)
        if success:
            reward += self.w["R_SUCCESS_GRASP"]

        return reward, success, {"distance": distance, "gripper_open": gripper_open}

    # ---- Stage 3:抓起 + 握住 ----
    def _reward_stage3(self, action: np.ndarray) -> tuple[float, bool, dict]:
        tcp_xyz = self.data.xpos[self._tcp_body_id]
        block_xyz = self.data.xpos[self._block_body_id]
        distance = float(np.linalg.norm(tcp_xyz - block_xyz))

        reward = -self.w["W_DISTANCE"] * distance
        if self._prev_distance is not None:
            reward += self.w["W_PROGRESS"] * (self._prev_distance - distance)
        self._prev_distance = distance

        gripper_open = float(self.data.qpos[self._gripper_qpos_addr]) / GRIPPER_MAX_OPEN
        if distance < self.w["GRIPPER_SHAPING_RADIUS"]:
            reward += self.w["W_GRIPPER_CLOSE"] * (1.0 - gripper_open)

        # 抓起判定
        block_height_above_floor = block_xyz[2] - (self._floor_z + BLOCK_HALF_SIZE)
        is_lifted = block_height_above_floor > LIFT_THRESHOLD
        # 「在手中」近似:方塊離地 + 與 TCP 近(距離 < 抓取半徑稍寬)
        is_holding = is_lifted and (distance < GRASP_DISTANCE_THRESHOLD * 2.0) and \
                     (gripper_open < GRIPPER_CLOSED_THRESHOLD)

        # 首次抓起獎勵(只給一次)
        if is_lifted and not self._has_lifted:
            reward += self.w["R_SUCCESS_LIFT"]
            self._has_lifted = True

        # 持續握持獎勵 + potential-based 抬升(獎勵抬高、懲罰下降,握平穩時為 0)
        if is_holding:
            reward += self.w["W_HOLDING"]
            if self._prev_block_height is not None:
                reward += self.w["W_LIFT_HEIGHT"] * (block_height_above_floor - self._prev_block_height)
            self._prev_block_height = block_height_above_floor
        else:
            self._prev_block_height = None

        reward -= self.w["W_ACTION_SMOOTH"] * float(np.sum((action - self._last_action) ** 2))
        reward -= self.w["W_TIME"]

        # 成功:已抓起 + 仍在握(撐到 episode 結束就算)
        success = is_holding and self._has_lifted
        # Stage 3 不在 mid-episode 終結,讓 truncate 結束以驗證「持續握住」
        # 若想 episode 內就 terminate,把下面 return 改成 (reward, success, info)

        return reward, success, {
            "distance": distance,
            "block_height": float(block_height_above_floor),
            "is_lifted": bool(is_lifted),
            "is_holding": bool(is_holding),
        }

    # ---- Stage 4-5:放置(順序狀態機)----
    def _reward_stage4(self, action: np.ndarray) -> tuple[float, bool, dict]:
        tcp_xyz = self.data.xpos[self._tcp_body_id]
        block_xyz = self.data.xpos[self._block_body_id]
        target_xyz = self._target_xyz

        distance_tcp_block = float(np.linalg.norm(tcp_xyz - block_xyz))
        distance_block_target_xy = float(np.linalg.norm(block_xyz[:2] - target_xyz[:2]))

        # Phase A:還沒抓起 → 鼓勵靠近方塊(同 stage 2/3 shaping)
        # Phase B:已抓起 → 鼓勵把方塊靠近目標區
        if self._task_state == TASK_STATE_INITIAL:
            reward = -self.w["W_DISTANCE"] * distance_tcp_block
            shaping_target = distance_tcp_block
        else:
            # 已抓起後:shaping 換成方塊→目標區距離
            reward = -self.w["W_DIST_TO_TARGET"] * distance_block_target_xy
            shaping_target = distance_block_target_xy

        if self._prev_distance is not None:
            reward += self.w["W_PROGRESS"] * (self._prev_distance - shaping_target)
        self._prev_distance = shaping_target

        gripper_open = float(self.data.qpos[self._gripper_qpos_addr]) / GRIPPER_MAX_OPEN
        if self._task_state == TASK_STATE_INITIAL and distance_tcp_block < self.w["GRIPPER_SHAPING_RADIUS"]:
            reward += self.w["W_GRIPPER_CLOSE"] * (1.0 - gripper_open)

        # ---- 狀態機 ----
        block_height = block_xyz[2] - (self._floor_z + BLOCK_HALF_SIZE)
        block_landed = abs(block_xyz[2] - (self._floor_z + BLOCK_HALF_SIZE)) < PLACE_LANDED_TOLERANCE
        block_in_zone = distance_block_target_xy < TARGET_ZONE_HALF_SIZE
        is_gripper_open = gripper_open > RELEASE_THRESHOLD
        is_holding = (block_height > LIFT_THRESHOLD) and \
                     (distance_tcp_block < GRASP_DISTANCE_THRESHOLD * 2.0) and \
                     (gripper_open < GRIPPER_CLOSED_THRESHOLD)

        # 狀態轉移(順序、單向)
        if self._task_state == TASK_STATE_INITIAL:
            if is_holding and block_height > LIFT_THRESHOLD:
                self._task_state = TASK_STATE_GRASPED
                reward += self.w["R_SUCCESS_LIFT"]  # 抓起獎勵
        elif self._task_state == TASK_STATE_GRASPED:
            # 已抓起:等方塊落到目標區(放下)
            if block_in_zone and block_landed:
                self._task_state = TASK_STATE_PLACED
                reward += self.w["R_SUCCESS_PLACE"]
        elif self._task_state == TASK_STATE_PLACED:
            # 已放置:等夾爪張開
            if is_gripper_open:
                self._task_state = TASK_STATE_RELEASED
                reward += self.w["R_SUCCESS_RELEASE"]

        # 持續握持小獎勵(只在 Phase A→B 轉移後給)
        if self._task_state == TASK_STATE_GRASPED and is_holding:
            reward += self.w["W_HOLDING"]

        reward -= self.w["W_ACTION_SMOOTH"] * float(np.sum((action - self._last_action) ** 2))
        reward -= self.w["W_TIME"]

        success = self._task_state == TASK_STATE_RELEASED

        return reward, success, {
            "distance_tcp_block": distance_tcp_block,
            "distance_block_target": distance_block_target_xy,
            "block_height": float(block_height),
            "is_holding": bool(is_holding),
            "block_in_zone": bool(block_in_zone),
            "block_landed": bool(block_landed),
            "is_gripper_open": bool(is_gripper_open),
            "task_state": self._task_state,
        }

    # ============================================================
    # Misc
    # ============================================================

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def render(self):
        if self.render_mode == "human" and self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
