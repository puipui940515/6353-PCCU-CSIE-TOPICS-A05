"""預生成訓練資料(GPU 方案 B 的關鍵)· 多進程平行版

瓶頸不在網路(28k 參數),在「合成 6 通道 192kHz 音訊 + pyroomacoustics 渲染 + 帶通 + FFT」。
把這步預先做完存盤,訓練時直接讀 → GPU 才餵得飽。

用法:
    python gen_dataset.py --n 200000 --seed 0 --use-v2 --out data/train.npz
    python gen_dataset.py --n 20000  --seed 999 --use-v2 --out data/eval.npz
    # 指定核心數(預設用全部):
    python gen_dataset.py --n 200000 --use-v2 --workers 8 --out data/train.npz

平行化:把 n 樣本切給 W 個 worker,各自獨立 rng(seed+worker_id,可重現)。
產出 .npz:feats (N, feat_dim) + labels (N,) + range_labels (N,) + sig_types。
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from config import DEFAULT
from env import LocalizationEnv, az_to_bin, range_to_bin


def _fmt_eta(sec: float) -> str:
    """秒 → 人類可讀 ETA。"""
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _worker(task: tuple) -> tuple:
    """單一 worker:產 n_local 個樣本。回傳 (feats, labels, range_labels, sig_types)。

    每個 worker 用 seed+worker_id 建獨立 rng,確保平行下仍可重現、且不重複。
    """
    worker_id, n_local, seed, use_v2 = task
    cfg = DEFAULT
    n_bins = cfg.task.n_azimuth_bins
    bin_edges = cfg.range_head.bin_edges_m
    feat_dim = (cfg.audio.n_mics - 1) * 3

    env = LocalizationEnv(cfg, seed=seed + worker_id, use_v2=use_v2)
    feats = np.zeros((n_local, feat_dim), dtype=np.float32)
    labels = np.zeros(n_local, dtype=np.int64)
    range_labels = np.zeros(n_local, dtype=np.int64)
    sig_types = []
    render_types = []
    for i in range(n_local):
        f, az, meta = env.sample()
        feats[i] = f
        labels[i] = az_to_bin(az, n_bins)
        range_labels[i] = range_to_bin(meta["source_range_m"], bin_edges)
        sig_types.append(meta["type"])
        render_types.append(meta.get("render_type", "unknown"))
    return feats, labels, range_labels, sig_types, render_types


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="樣本數")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/train.npz")
    ap.add_argument("--use-v2", action="store_true", help="用 v2 特徵(預設 v1)")
    ap.add_argument("--workers", type=int, default=0,
                    help="平行進程數,預設 0 = 用全部 CPU 核心")
    ap.add_argument("--chunk-size", type=int, default=100,
                    help="每塊樣本數(越小進度更新越即時,排程開銷略增)")
    args = ap.parse_args()

    cfg = DEFAULT
    feat_dim = (cfg.audio.n_mics - 1) * 3
    n_workers = args.workers or os.cpu_count() or 1
    n_workers = min(n_workers, args.n)

    # 切塊:用固定小 chunk size,塊數多 → imap 高頻回收 → 進度條即時更新
    cs = max(1, args.chunk_size)
    n_chunks = (args.n + cs - 1) // cs
    chunk_sizes = [min(cs, args.n - k * cs) for k in range(n_chunks)]
    tasks = [(k, chunk_sizes[k], args.seed, args.use_v2)
             for k in range(n_chunks) if chunk_sizes[k] > 0]

    print(f"平行生成:{args.n} 樣本 | {n_workers} workers | {len(tasks)} chunks "
          f"(每塊 {cs}) | feat_dim={feat_dim} | {'v2' if args.use_v2 else 'v1'} 特徵")

    feats_all = np.zeros((args.n, feat_dim), dtype=np.float32)
    labels_all = np.zeros(args.n, dtype=np.int64)
    range_all = np.zeros(args.n, dtype=np.int64)
    sig_all: list[str] = []
    render_all: list[str] = []

    t0 = time.time()
    done = 0
    write_ptr = 0
    with Pool(processes=n_workers) as pool:
        # imap:依序回收結果,邊收邊更新進度
        for feats, labels, ranges, sig_types, render_types in pool.imap(_worker, tasks):
            k = len(labels)
            feats_all[write_ptr:write_ptr + k] = feats
            labels_all[write_ptr:write_ptr + k] = labels
            range_all[write_ptr:write_ptr + k] = ranges
            sig_all.extend(sig_types)
            render_all.extend(render_types)
            write_ptr += k
            done += k
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            eta = (args.n - done) / rate if rate > 0 else 0
            bar_n = int(30 * done / args.n)
            bar = "█" * bar_n + "·" * (30 - bar_n)
            # 單行刷新(\r),不洗版
            print(f"\r  [{bar}] {done}/{args.n} ({done/args.n:.0%}) "
                  f"| {rate:.0f} 樣本/秒 | 已用 {_fmt_eta(el)} | ETA {_fmt_eta(eta)}   ",
                  end="", flush=True)
    print()  # 換行收尾

    # 距離 bin 分布(檢查是否嚴重不均 → 影響距離 head 訓練)
    uniq, cnt = np.unique(range_all, return_counts=True)
    dist_str = ", ".join(f"bin{u}:{c/args.n:.0%}" for u, c in zip(uniq, cnt))
    # 實際混合比例(確認 pyroom_ratio 有生效;含 fallback 拆開看)
    runiq, rcnt = np.unique(np.array(render_all), return_counts=True)
    render_str = ", ".join(f"{u}:{c/args.n:.0%}" for u, c in zip(runiq, rcnt))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feats=feats_all, labels=labels_all,
                        range_labels=range_all,
                        sig_types=np.array(sig_all))
    print(f"✅ 存於 {out} | {args.n} 樣本 | feat_dim={feat_dim} | "
          f"耗時 {_fmt_eta(time.time()-t0)}")
    print(f"   距離 bin 分布: {dist_str}")
    print(f"   實際渲染比例: {render_str}(目標 pyroom≈{cfg.source_dr.pyroom_ratio:.0%})")
    print(f"   (bin 嚴重不均時,train_gpu 距離 CE 可考慮加 class weight)")


if __name__ == "__main__":
    main()
