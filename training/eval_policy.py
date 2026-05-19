"""載入訓練好的 SAC policy,開 MuJoCo viewer 看實際抓取效果

執行:
    source ~/dobot_project/setup_env.sh
    python ~/dobot_project/training/eval_policy.py \
        --model ~/dobot_project/runs/sac_<timestamp>/best/best_model.zip

    # 也可以跑多個 episode 看成功率
    python ~/dobot_project/training/eval_policy.py \
        --model <path> --episodes 20 --no-viewer
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import SAC

from envs import MuJoCoDobotEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="訓練好的 SAC zip 檔路徑")
    parser.add_argument("--episodes", type=int, default=5,
                        help="跑幾個 episode")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--no-viewer", action="store_true",
                        help="不開 viewer(只跑 evaluation 算成功率)")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        print(f"❌ 找不到 model: {model_path}")
        sys.exit(1)

    print(f"載入 model: {model_path}")
    model = SAC.load(str(model_path), device="cuda")

    env = MuJoCoDobotEnv(max_episode_steps=args.max_steps)
    if not args.no_viewer:
        env.render_mode = "human"
        env.render()
        print("Viewer 啟動,5 秒後開始 episode...")
        time.sleep(5)

    successes = 0
    total_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_steps = 0
        success = False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if info.get("success"):
                success = True
            if terminated or truncated:
                break
            if not args.no_viewer:
                # 不睡會跑太快看不清楚,睡 1/50 秒對應 policy 頻率
                time.sleep(0.02)

        successes += int(success)
        total_rewards.append(ep_reward)
        print(f"  Episode {ep+1:2d}: reward={ep_reward:7.2f}, steps={ep_steps:3d}, success={success}")

    print(f"\n=== 結果 ===")
    print(f"  成功率: {successes}/{args.episodes} = {successes/args.episodes*100:.1f}%")
    print(f"  平均 reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
