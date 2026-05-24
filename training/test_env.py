"""Env smoke test:確認 MuJoCoDobotEnv 介面正確(聲學定位 obs 版)

執行:
    source ~/dobot_project/setup_env.sh
    python ~/dobot_project/training/test_env.py
    # 帶定位權重:
    python ~/dobot_project/training/test_env.py --weights ~/dobot_project/detect/runs/<tag>/checkpoints/best_eval.pt

放置位置:dobot_project/training/test_env.py
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from envs import MuJoCoDobotEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=None,
                    help="定位網路權重(detect 訓練的)。不給則隨機初始化")
    ap.add_argument("--stage", type=int, default=1)
    args = ap.parse_args()

    print("建立 env...")
    env = MuJoCoDobotEnv(stage=args.stage, perception_weights=args.weights)

    print(f"  action_space:           {env.action_space}")
    print(f"  observation_space keys: {list(env.observation_space.spaces.keys())}")
    print(f"  定位權重含距離 head:    {env.perception.has_range}")

    print("\nReset...")
    obs, info = env.reset(seed=42)
    for k, v in obs.items():
        print(f"  obs['{k}']: shape={v.shape}, dtype={v.dtype}, "
              f"range=[{v.min():.3f}, {v.max():.3f}]")

    # 聲學定位 argmax 解讀
    if "source_azimuth" in obs:
        az_bin = int(np.argmax(obs["source_azimuth"]))
        print(f"  → source_azimuth argmax bin {az_bin} ≈ {az_bin * 5}°")
    if "source_range" in obs:
        rg_bin = int(np.argmax(obs["source_range"]))
        labels = ["很近", "近", "遠", "很遠"]
        print(f"  → source_range argmax bin {rg_bin}"
              f"({labels[rg_bin] if rg_bin < len(labels) else rg_bin})")

    print("\n跑 100 步隨機 action...")
    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if step % 20 == 0:
            # distance 從 info 讀(reward 算的真值),不從 obs 讀(obs 已無上帝視角)
            dist = info.get("distance", info.get("distance_tcp_block", float("nan")))
            btt = obs.get("base_to_tcp_dist", [float("nan")])[0]
            print(f"  step {step:3d}: reward={reward:6.3f}, "
                  f"dist(true)={dist:.4f}, base_to_tcp={btt:.4f}, "
                  f"success={info.get('success')}")
        if term or trunc:
            print(f"  episode 結束 at step {step}")
            break

    print(f"\n總 reward: {total_reward:.2f}")
    print("✅ env smoke test 通過")
    env.close()


if __name__ == "__main__":
    main()
