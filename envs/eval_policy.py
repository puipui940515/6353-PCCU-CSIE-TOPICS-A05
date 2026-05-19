"""載入訓練好的 SAC policy,開 MuJoCo viewer 看實際抓取效果

執行:
    source ~/dobot_project/setup_env.sh

    # 預設 stage 1(對準,weld 焊住方塊不會掉)
    python ~/dobot_project/training/eval_policy.py \
        --model ~/dobot_project/runs/sac_stage1_<ts>/best/best_model.zip

    # Stage 3+(freejoint + 重力,方塊真的會掉)
    python ~/dobot_project/training/eval_policy.py \
        --model ~/dobot_project/runs/sac_stage3_<ts>/best/best_model.zip \
        --stage 3

    # 多 episode 算成功率(不開 viewer)
    python ~/dobot_project/training/eval_policy.py \
        --model <path> --stage 4 --episodes 20 --no-viewer

注意:
    --stage 必須跟 model 訓練時用的 stage 一致,否則 env 物理設定不對,
    eval 出來的成功率沒意義。
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
    parser.add_argument("--stage", type=int, default=1,
                        help="要 eval 的 stage(必須跟訓練時一致),預設 1")
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
    print(f"Stage: {args.stage}")
    model = SAC.load(str(model_path), device="cuda")

    env = MuJoCoDobotEnv(
        max_episode_steps=args.max_steps,
        stage=args.stage,
    )
    if not args.no_viewer:
        env.render_mode = "human"
        env.render()
        print("Viewer 啟動,5 秒後開始 episode...")
        time.sleep(5)

    successes = 0
    total_rewards = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_steps = 0
        success = False
        # Stage 4 才追蹤狀態機進展
        max_task_state = info.get("task_state", 0)

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if info.get("success"):
                success = True
            max_task_state = max(max_task_state, info.get("task_state", 0))
            if terminated or truncated:
                break
            if not args.no_viewer:
                time.sleep(0.02)

        successes += int(success)
        total_rewards.append(ep_reward)
        extra = f", max_task_state={max_task_state}" if args.stage in (4, 5) else ""
        print(f"  Episode {ep+1:2d}: reward={ep_reward:7.2f}, steps={ep_steps:3d}, success={success}{extra}")

    print(f"\n=== 結果(Stage {args.stage}) ===")
    print(f"  成功率: {successes}/{args.episodes} = {successes/args.episodes*100:.1f}%")
    print(f"  平均 reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
