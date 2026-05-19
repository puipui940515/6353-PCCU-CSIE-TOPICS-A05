"""Env smoke test:確認 MuJoCoDobotEnv 介面正確

執行:
    source ~/dobot_project/setup_env.sh
    python ~/dobot_project/training/test_env.py
"""

from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from envs import MuJoCoDobotEnv


def main() -> None:
    print("建立 env...")
    env = MuJoCoDobotEnv()

    print(f"  action_space:      {env.action_space}")
    print(f"  observation_space keys: {list(env.observation_space.spaces.keys())}")

    print("\nReset...")
    obs, info = env.reset(seed=42)
    for k, v in obs.items():
        print(f"  obs['{k}']: shape={v.shape}, dtype={v.dtype}, "
              f"range=[{v.min():.3f}, {v.max():.3f}]")

    print("\n跑 100 步隨機 action...")
    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if step % 20 == 0:
            tcp = obs["tcp_pose"][:3]
            block = obs["block_pose"][:3]
            dist = np.linalg.norm(tcp - block)
            print(f"  step {step:3d}: reward={reward:6.3f}, dist(tcp,block)={dist:.4f}, success={info.get('success')}")
        if term or trunc:
            print(f"  episode 結束 at step {step}")
            break

    print(f"\n總 reward: {total_reward:.2f}")
    print("✅ env smoke test 通過")
    env.close()


if __name__ == "__main__":
    main()
