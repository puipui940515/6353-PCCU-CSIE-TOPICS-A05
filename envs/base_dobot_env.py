"""BaseDobotEnv:模擬與真機共用的抽象介面

對應 docs/agents/domain.md §4(Gym Env 介面契約)、§5.5(Curriculum)。

放置位置:dobot_project/envs/base_dobot_env.py

2026-05-23 變更([breaking]):
  observation_space 改由 config.obs_spec() 驅動,不再 hardcode。
  移除 oracle block_pose,改聲學定位 source_azimuth/source_range + base_to_tcp_dist。
  詳見 requirement_localization_to_sac.md。
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from detect.config import LocalizationConfig, DEFAULT


N_JOINTS = 4
GRIPPER_DIM = 1
TCP_DIM = 7

ACTION_DIM = N_JOINTS + GRIPPER_DIM


class BaseDobotEnv(gym.Env, ABC):
    """Dobot Magician 抽象環境(聲學定位 obs 版)

    Observation(dict,實際內容由 config.obs 開關決定):
      joint_position:   (4,)   J1-J4 角度,rad
      joint_velocity:   (4,)   J1-J4 角速度,rad/s
      tcp_pose:         (7,)   末端 xyz + quat
      gripper_state:    (1,)   夾爪當前開度
      base_to_tcp_dist: (1,)   末端到基座距離(純 FK,m)
      source_azimuth:   (72,)  聲學方位機率分布(softmax)
      source_range:     (4,)   聲學遠近機率分布(softmax),權重無距離 head 時自動不出現

    Action(Box):
      正規化關節速度 + 夾爪命令,維度 5,範圍 [-1, 1]

    Note:
      上帝視角的 block_pose 已從 obs 移除(違反 domain.md §1「無視覺」)。
      reward 函式仍可讀方塊真值(訓練監督訊號,policy 不經手),見 requirement §3。
      若 config.obs.enable_oracle_block_pose=True 會把它塞回 obs,僅供 debug。
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # obs 各 key 的上下界來源(機率分布項固定 [0,1])
    _PROB_KEYS = ("source_azimuth", "source_range")

    def __init__(self, cfg: LocalizationConfig = DEFAULT) -> None:
        super().__init__()
        self.cfg = cfg

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = self._build_observation_space(cfg)

    def _build_observation_space(self, cfg: LocalizationConfig) -> spaces.Dict:
        """由 config.obs_spec() 決定有哪些 key、各多大,範圍取自 config.obs.ranges。"""
        spec = cfg.obs_spec()           # {key: dim}
        r = cfg.obs.ranges
        box: dict[str, spaces.Box] = {}
        for key, dim in spec.items():
            if key in self._PROB_KEYS:
                lo, hi = 0.0, 1.0       # softmax 機率
            elif key == "block_pose":
                lo, hi = -1.0, 1.0      # debug 用
            else:
                lo, hi = getattr(r, key)
            box[key] = spaces.Box(lo, hi, (dim,), np.float32)
        return spaces.Dict(box)

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
