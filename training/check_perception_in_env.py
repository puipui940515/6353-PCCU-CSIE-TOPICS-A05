"""Env 内定位准确度检查(接 SAC 前的眼睛体检)

目的:量「眼睛装进 MuJoCoDobotEnv 后」对方块的方位/距离命中率。
这跟 train_gpu 的 81% 不同 —— train_gpu 量的是 detect 自己的资料分布;
本脚本量的是 env 里真实的方块分布 + mic 世界座标(阵列装末端、会动),
确认没有座标系或介面造成的系统性偏差,再花时间训 SAC。

⚠️ 座标系(关键):
  定位网路输出的 source_azimuth 是「相对阵列」的方位(domain.md §4.1)。
  阵列装末端,所以真值方位要用 (block - mic_center) 算,不能用 block 的世界方位。
  这脚本就是按「相对阵列」比对,否则会误判眼睛坏掉。

用法:
    cd ~/dobot_project
    python training/check_perception_in_env.py \
        --weights detect/runs/range_v1/checkpoints/best_eval.pt --n 500
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from envs import MuJoCoDobotEnv
from detect.config import DEFAULT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True,
                    help="定位网路权重(detect 训练的 .pt)")
    ap.add_argument("--n", type=int, default=500, help="撒方块次数")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = DEFAULT
    n_az = cfg.task.n_azimuth_bins
    deg_per_bin = 360.0 / n_az
    hit_thr = cfg.task.hit_threshold_deg
    bin_edges = np.asarray(cfg.range_head.bin_edges_m)

    env = MuJoCoDobotEnv(stage=args.stage, perception_weights=args.weights)
    print(f"定位权重含距离 head: {env.perception.has_range}")

    az_hits = 0
    az_errs = []
    rg_hits = 0
    rg_has = 0
    rng = np.random.default_rng(args.seed)

    for ep in range(args.n):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        # --- 取真值:相对阵列(mic 中心)的方位与距离 ---
        # env 内部:_block_xyz 是方块世界座标,mic site 世界座标可从 data 取
        block_xyz = env._block_xyz
        mic_world = np.array([env.data.site_xpos[sid] for sid in env._mic_site_ids])
        mic_center = mic_world.mean(axis=0)
        rel = block_xyz - mic_center

        true_az = np.degrees(np.arctan2(rel[1], rel[0])) % 360.0
        true_range = float(np.linalg.norm(rel))
        true_rg_bin = int(np.searchsorted(bin_edges, true_range, side="right"))

        # --- 取预测:obs 里的 source_* argmax ---
        pred_az_bin = int(np.argmax(obs["source_azimuth"]))
        pred_az = pred_az_bin * deg_per_bin
        err = abs(pred_az - true_az) % 360.0
        err = min(err, 360.0 - err)
        az_errs.append(err)
        if err < hit_thr:
            az_hits += 1

        if "source_range" in obs:
            rg_has += 1
            pred_rg_bin = int(np.argmax(obs["source_range"]))
            if pred_rg_bin == true_rg_bin:
                rg_hits += 1

    env.close()

    print(f"\n=== 方位(相对阵列)===")
    print(f"  命中率(<{hit_thr:.0f}°) = {az_hits/args.n:.1%}")
    print(f"  平均误差        = {np.mean(az_errs):.1f}°")
    print(f"  中位误差        = {np.median(az_errs):.1f}°")
    if rg_has:
        print(f"=== 距离(4-bin)===")
        print(f"  命中率 = {rg_hits/rg_has:.1%}(随机 {1/cfg.range_head.n_range_bins:.0%})")
    else:
        print("=== 距离:此权重无距离 head,obs 无 source_range ===")

    print(f"\n判读:")
    print(f"  方位命中率应接近 train_gpu 的水准(~90%)。若明显更低,")
    print(f"  表示 env 里 mic 世界座标/方块分布与 detect 训练有偏差,需排查。")
    print(f"  距离命中率应明显 > 随机(25%)。接近随机则 source_range 不该进 obs。")


if __name__ == "__main__":
    main()
