"""BaseDobotEnv:模擬與真機共用的抽象介面

對應 docs/agents/domain.md §4(Gym Env 介面契約)、§5.5(Curriculum)。
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from gymnasium import spaces


N_JOINTS = 4
GRIPPER_DIM = 1
TCP_DIM = 7
WRENCH_DIM = 6
IMU_DIM = 3

ACTION_DIM = N_JOINTS + GRIPPER_DIM


class BaseDobotEnv(gym.Env, ABC):
    """Dobot Magician 抽象環境(Phase 1a:Curriculum 訓練版本)

    Observation(dict):
      joint_position: (4,)   J1-J4 角度,rad
      joint_velocity: (4,)   J1-J4 角速度,rad/s
      tcp_pose:       (7,)   末端 xyz + quat
      gripper_state:  (1,)   夾爪當前開度
      block_pose:     (7,)   [DEPRECATED-Phase1a] 方塊 xyz + quat,oracle

    Action(Box):
      正規化關節速度 + 夾爪命令,維度 5,範圍 [-1, 1]

    Note:
      block_pose 是 oracle observation,違反 domain.md §1「無視覺」精神,
      Phase 1b 聲學模組上線後移除,改為 audio + 場景認知。

      Stage 4 的目標區位置不進入 obs space — env 內部存放,reward 函式可讀,
      但 policy 看不到。介面真的需要新欄位時(stage 4 進行中發現必要),
      會是有意識的 breaking change,屆時更新 domain.md 與本檔。
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self) -> None:
        super().__init__()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "joint_position": spaces.Box(-np.pi, np.pi, (N_JOINTS,), np.float32),
            "joint_velocity": spaces.Box(-10.0, 10.0, (N_JOINTS,), np.float32),
            "tcp_pose":       spaces.Box(-1.0, 1.0, (TCP_DIM,), np.float32),
            "gripper_state":  spaces.Box(0.0, 0.02, (GRIPPER_DIM,), np.float32),
            # DEPRECATED-Phase1a:見 class docstring
            "block_pose":     spaces.Box(-1.0, 1.0, (TCP_DIM,), np.float32),
        })

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
