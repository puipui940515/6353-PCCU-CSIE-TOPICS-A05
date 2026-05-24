"""SAC 訓練 Dobot 抓取任務(Phase 1a Curriculum 版)

對應 docs/agents/domain.md §5.4(reward)、§5.5(curriculum)、§8(超參數)

使用 config:
    # Stage 1
    python train_sac.py --config ~/dobot_project/configs/sac_stage1.yaml

    # Stage 2(從 stage 1 best resume)
    python train_sac.py \
        --config ~/dobot_project/configs/sac_stage2.yaml \
        --resume ~/dobot_project/runs/sac_stage1_<ts>/best/best_model.zip

CLI 可覆寫 config 內任何 schedule 參數,例如 --total-steps、--n-envs、--seed。
"""

from __future__ import annotations
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from envs import MuJoCoDobotEnv
from training.sac_policy import LayerNormCombinedExtractor


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def make_env(stage: int, reward_weights: dict | None, seed: int = 0,
             perception_weights: str | None = None):
    def _init():
        env = MuJoCoDobotEnv(
            render_mode=None,
            stage=stage,
            reward_weights=reward_weights,
            perception_weights=perception_weights,
        )
        env = Monitor(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init


def parse_ent_coef(v):
    if isinstance(v, str) and v.startswith("auto"):
        return v
    return float(v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "configs" / "sac_stage1.yaml"))
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="run 目錄前綴,預設用 config 內 stage")
    parser.add_argument("--resume", type=str, default=None,
                        help="從 zip 繼續訓練(stage 切換時用)")
    parser.add_argument("--load-buffer", type=str, default=None,
                        help="同時載入 replay buffer pkl;stage 切換時建議不載(reward 變了)")
    parser.add_argument("--perception-weights", type=str, default=None,
                        help="定位網路權重(detect 訓練的 .pt)。不給則 env 用隨機初始化定位網路"
                             "(等於瞎眼,僅供管路測試)")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    stage = int(cfg.get("stage", 1))
    reward_weights = cfg.get("reward_weights", None)
    net_cfg = cfg["network"]
    sac_cfg = cfg["sac"]
    sched_cfg = cfg["schedule"]

    total_steps = args.total_steps or sched_cfg["total_steps"]
    n_envs = args.n_envs or sched_cfg["n_envs"]
    seed = args.seed if args.seed is not None else sched_cfg["seed"]
    tag = args.tag or f"sac_stage{stage}"

    hidden = list(net_cfg["hidden_layers"])
    activation_cls = ACTIVATION_MAP[net_cfg["activation"].lower()]
    use_layer_norm = bool(net_cfg.get("layer_norm", True))

    policy_kwargs = dict(
        net_arch=dict(pi=hidden, qf=hidden),
        activation_fn=activation_cls,
    )
    if use_layer_norm:
        policy_kwargs["features_extractor_class"] = LayerNormCombinedExtractor

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / f"{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run 目錄: {run_dir}")
    print(f"Stage: {stage}")
    print(f"Network: hidden={hidden}, activation={net_cfg['activation']}")
    if reward_weights:
        print(f"Reward weights 覆寫: {list(reward_weights.keys())}")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump({
            **cfg,
            "_runtime": {
                "total_steps": total_steps, "n_envs": n_envs, "seed": seed,
                "tag": tag, "resume": args.resume,
            },
        }, f, sort_keys=False)
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        (run_dir / "commit.txt").write_text(sha)
    except Exception:
        (run_dir / "commit.txt").write_text("UNKNOWN")

    print(f"建立 {n_envs} 個平行 env(stage={stage})...")
    if n_envs == 1:
        train_env = DummyVecEnv([make_env(stage, reward_weights, seed,
                                          args.perception_weights)])
    else:
        train_env = SubprocVecEnv([
            make_env(stage, reward_weights, seed + i, args.perception_weights)
            for i in range(n_envs)
        ])
    eval_env = DummyVecEnv([make_env(stage, reward_weights, seed + 999,
                                     args.perception_weights)])

    if args.resume:
        resume_path = Path(args.resume).expanduser()
        print(f"從 {resume_path} 繼續訓練(stage 切換)...")
        model = SAC.load(
            str(resume_path),
            env=train_env,
            device="cuda",
            tensorboard_log=str(run_dir / "tb"),
        )
        model.learning_starts = 0
        if args.load_buffer:
            buf_path = Path(args.load_buffer).expanduser()
            print(f"載入 replay buffer: {buf_path}")
            print("⚠️  注意:跨 stage 載 buffer,舊資料的 reward 結構已過時,慎用")
            model.load_replay_buffer(str(buf_path))
    else:
        print("建立新 SAC model...")
        model = SAC(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=float(sac_cfg["learning_rate"]),
            buffer_size=int(sac_cfg["buffer_size"]),
            batch_size=int(sac_cfg["batch_size"]),
            gamma=float(sac_cfg["gamma"]),
            tau=float(sac_cfg["tau"]),
            train_freq=int(sac_cfg["train_freq"]),
            gradient_steps=int(sac_cfg["gradient_steps"]),
            learning_starts=int(sac_cfg["learning_starts"]),
            ent_coef=parse_ent_coef(sac_cfg["ent_coef"]),
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            tensorboard_log=str(run_dir / "tb"),
            device="cuda",
        )

    checkpoint_every = int(sched_cfg["checkpoint_every"])
    eval_every = int(sched_cfg["eval_every"])
    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_every // n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="sac",
    )
    eval_cb = EvalCallback(
        eval_env=eval_env,
        eval_freq=max(eval_every // n_envs, 1),
        n_eval_episodes=5,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval_logs"),
        deterministic=True,
        render=False,
    )

    print(f"開始訓練 ({total_steps} steps)...")
    print(f"監控: tensorboard --logdir {run_dir}/tb")
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = run_dir / "final.zip"
    model.save(str(final_path))
    buffer_path = run_dir / "replay_buffer.pkl"
    model.save_replay_buffer(str(buffer_path))
    print(f"\n✅ Stage {stage} 訓練完成。")
    print(f"   Final:  {final_path}")
    print(f"   Best:   {run_dir / 'best' / 'best_model.zip'}")
    print(f"   Buffer: {buffer_path}")
    print(f"\n下一步(若要進下一 stage):")
    print(f"   python train_sac.py \\")
    print(f"       --config configs/sac_stage{stage+1}.yaml \\")
    print(f"       --resume {run_dir / 'best' / 'best_model.zip'}")


if __name__ == "__main__":
    main()
