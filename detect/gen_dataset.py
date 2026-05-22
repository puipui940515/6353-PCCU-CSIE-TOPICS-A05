"""預生成訓練資料(GPU 方案 B 的關鍵)

瓶頸不在網路(28k 參數),在「現合成 6 通道 192kHz 音訊 + 帶通 + FFT」。
把這步預先做完存盤,訓練時直接讀 → GPU 才餵得飽。

用法:
    python gen_dataset.py --n 200000 --seed 0 --out data/train.npz
    python gen_dataset.py --n 20000  --seed 999 --out data/eval.npz

產出 .npz:feats (N, feat_dim) + labels (N,) + meta(sig_type 供分類統計)。
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np

from config import DEFAULT
from env import LocalizationEnv, az_to_bin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="樣本數")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/train.npz")
    ap.add_argument("--use-v2", action="store_true", help="用 v2 特徵(預設 v1)")
    args = ap.parse_args()

    cfg = DEFAULT
    n_bins = cfg.task.n_azimuth_bins
    env = LocalizationEnv(cfg, seed=args.seed, use_v2=args.use_v2)

    feat_dim = (cfg.audio.n_mics - 1) * 3
    feats = np.zeros((args.n, feat_dim), dtype=np.float32)
    labels = np.zeros(args.n, dtype=np.int64)
    sig_types = []

    t0 = time.time()
    for i in range(args.n):
        f, az, meta = env.sample()
        feats[i] = f
        labels[i] = az_to_bin(az, n_bins)
        sig_types.append(meta["type"])
        if (i + 1) % max(1, args.n // 20) == 0:
            el = time.time() - t0
            print(f"  {i+1}/{args.n} ({(i+1)/args.n:.0%}) | {el:.0f}s "
                  f"| {(i+1)/el:.0f} 樣本/秒")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feats=feats, labels=labels,
                        sig_types=np.array(sig_types))
    print(f"\n✅ 存於 {out} | {args.n} 樣本 | feat_dim={feat_dim} | "
          f"耗時 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
