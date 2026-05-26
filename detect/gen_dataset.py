"""預生成訓練資料(GPU 方案 B 的關鍵)· 多進程平行版 v2

瓶頸不在網路(28k 參數),在「合成 6 通道 192kHz 音訊 + pyroomacoustics 渲染 + 帶通 + FFT」。
把這步預先做完存盤,訓練時直接讀 → GPU 才餵得飽。

用法:
    python gen_dataset.py --n 200000 --seed 0 --use-v2 --out data/train.npz
    python gen_dataset.py --n 20000  --seed 999 --use-v2 --out data/eval.npz
    # 指定核心數(預設用全部):
    python gen_dataset.py --n 200000 --use-v2 --workers 32 --out data/train.npz
    python gen_dataset.py --n 200000 --use-v2 --workers 32 --out data/eval.npz

平行化:把 n 樣本切給 W 個 worker,各自獨立 rng(seed+worker_id,可重現)。
每個 worker 進程只建一次 env(關鍵優化:舊版每個 chunk 重建 env,白燒 0.1~1s × chunk 數)。
產出 .npz:feats (N, feat_dim) + labels (N,) + range_labels (N,) + sig_types。

--- v2 改動(不改變樣本欄位與品質,模型照樣讀)---
1. env 只建 W 次(舊版 ~n/chunk_size 次)。env 初始化 0.1~1s 時,這是最大的一刀。
2. 進度用共享計數器(Value)即時更新,不再靠細切塊 → 跨進程序列化次數 2000→W。
3. 存檔預設不壓縮(savez),單執行緒 zlib 是隱形瓶頸;要壓用 --compress。
4. 內建計時探針:回報 env 建構 / 純 sample / 存檔三段時間,判斷下一刀該不該砍進 env.py。
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from multiprocessing import Pool, Value

import numpy as np

from config import DEFAULT
from env import LocalizationEnv, az_to_bin, range_to_bin, height_to_bin


# 共享計數器:worker 累加,主進程輪詢顯示進度。
# 用 initializer 注入,避免每個 task 都序列化傳遞。
_PROGRESS = None


def _init_pool(counter) -> None:
    global _PROGRESS
    _PROGRESS = counter


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
    """單一 worker:產 n_local 個樣本。

    每個 worker 用 seed+worker_id 建獨立 rng,確保平行下仍可重現、且不重複。
    env 只在此建一次(整個 task 共用),這是 v2 的關鍵優化。
    """
    worker_id, n_local, seed, use_v2 = task
    cfg = DEFAULT
    n_bins = cfg.task.n_azimuth_bins
    bin_edges = cfg.range_head.bin_edges_m
    height_edges = cfg.height_head.bin_edges_m
    feat_dim = (cfg.audio.n_mics - 1) * 3

    # --- 計時探針:env 建構 ---
    t_env0 = time.time()
    env = LocalizationEnv(cfg, seed=seed + worker_id, use_v2=use_v2)
    env_build_s = time.time() - t_env0

    feats = np.zeros((n_local, feat_dim), dtype=np.float32)
    labels = np.zeros(n_local, dtype=np.int64)
    range_labels = np.zeros(n_local, dtype=np.int64)
    height_labels = np.zeros(n_local, dtype=np.int64)
    source_ranges = np.zeros(n_local, dtype=np.float32)
    source_heights = np.zeros(n_local, dtype=np.float32)
    f0s = np.zeros(n_local, dtype=np.float32)
    self_yaws = np.zeros(n_local, dtype=np.float32)
    obstacle_counts = np.zeros(n_local, dtype=np.int64)
    obstacle_gains = np.zeros((n_local, cfg.audio.n_mics), dtype=np.float32)
    sig_types = []
    render_types = []

    # --- 計時探針:純 sample (DSP) ---
    t_smp0 = time.time()
    # 每處理 report_every 個就累加共享計數器一次(降低鎖競爭)
    report_every = max(1, n_local // 200)
    pending = 0
    for i in range(n_local):
        f, az, meta = env.sample()
        feats[i] = f
        labels[i] = az_to_bin(az, n_bins)
        range_labels[i] = range_to_bin(meta["source_range_m"], bin_edges)
        height_labels[i] = height_to_bin(meta["source_height_m"], height_edges)
        source_ranges[i] = meta["source_range_m"]
        source_heights[i] = meta["source_height_m"]
        f0s[i] = meta["f0"]
        self_yaws[i] = meta["self_yaw_deg"]
        obstacle_counts[i] = meta["obstacle_count"]
        obstacle_gains[i] = np.asarray(meta["obstacle_channel_gains"], dtype=np.float32)
        sig_types.append(meta["type"])
        render_types.append(meta.get("render_type", "unknown"))
        pending += 1
        if pending >= report_every and _PROGRESS is not None:
            with _PROGRESS.get_lock():
                _PROGRESS.value += pending
            pending = 0
    if pending and _PROGRESS is not None:
        with _PROGRESS.get_lock():
            _PROGRESS.value += pending
    sample_s = time.time() - t_smp0

    return (feats, labels, range_labels, height_labels, sig_types, render_types,
            source_ranges, source_heights, f0s, self_yaws, obstacle_counts,
            obstacle_gains, env_build_s, sample_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="樣本數")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/train.npz")
    ap.add_argument("--use-v2", action="store_true", help="用 v2 特徵(預設 v1)")
    ap.add_argument("--workers", type=int, default=0,
                    help="平行進程數,預設 0 = 用全部 CPU 核心")
    ap.add_argument("--compress", action="store_true",
                    help="存檔壓縮(zlib,慢但檔案小);預設不壓縮以加速")
    args = ap.parse_args()

    cfg = DEFAULT
    feat_dim = (cfg.audio.n_mics - 1) * 3
    n_workers = args.workers or os.cpu_count() or 1
    n_workers = min(n_workers, args.n)

    # 切塊:一 worker 一個大 task。env 只建 n_workers 次(舊版建 ~n/100 次)。
    # 樣本盡量平均分配,餘數攤到前幾個 worker。
    base = args.n // n_workers
    rem = args.n % n_workers
    chunk_sizes = [base + (1 if k < rem else 0) for k in range(n_workers)]
    tasks = [(k, chunk_sizes[k], args.seed, args.use_v2)
             for k in range(n_workers) if chunk_sizes[k] > 0]

    print(f"平行生成:{args.n} 樣本 | {n_workers} workers | {len(tasks)} tasks "
          f"(一 worker 一塊) | feat_dim={feat_dim} | "
          f"{'v2' if args.use_v2 else 'v1'} 特徵")

    feats_all = np.zeros((args.n, feat_dim), dtype=np.float32)
    labels_all = np.zeros(args.n, dtype=np.int64)
    range_all = np.zeros(args.n, dtype=np.int64)
    height_all = np.zeros(args.n, dtype=np.int64)
    source_ranges_all = np.zeros(args.n, dtype=np.float32)
    source_heights_all = np.zeros(args.n, dtype=np.float32)
    f0_all = np.zeros(args.n, dtype=np.float32)
    yaw_all = np.zeros(args.n, dtype=np.float32)
    obstacle_count_all = np.zeros(args.n, dtype=np.int64)
    obstacle_gains_all = np.zeros((args.n, cfg.audio.n_mics), dtype=np.float32)
    sig_all: list[str] = []
    render_all: list[str] = []

    t0 = time.time()
    write_ptr = 0
    env_build_total = 0.0   # 各 worker env 建構耗時加總(平行,僅供參考)
    env_build_max = 0.0     # 最慢的 worker env 建構(這才是牆鐘上的阻塞)
    sample_total = 0.0      # 各 worker 純 DSP 耗時加總

    counter = Value("Q", 0)  # 無號 64-bit 共享計數器
    with Pool(processes=n_workers, initializer=_init_pool,
              initargs=(counter,)) as pool:
        # imap_unordered:哪個先好先收,進度由共享計數器即時反映
        result_iter = pool.imap_unordered(_worker, tasks)
        results = []
        n_done_tasks = 0
        # 邊收結果邊輪詢進度
        while n_done_tasks < len(tasks):
            # 嘗試非阻塞地收一個結果(用短 timeout 模擬輪詢)
            try:
                res = result_iter.__next__()
                results.append(res)
                n_done_tasks += 1
            except StopIteration:
                break
            # 不論是否剛收到結果,都刷新一次進度
            done = counter.value
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            eta = (args.n - done) / rate if rate > 0 else 0
            bar_n = int(30 * done / args.n)
            bar = "█" * bar_n + "·" * (30 - bar_n)
            print(f"\r  [{bar}] {done}/{args.n} ({done/args.n:.0%}) "
                  f"| {rate:.0f} 樣本/秒 | 已用 {_fmt_eta(el)} | ETA {_fmt_eta(eta)}   ",
                  end="", flush=True)

    # 收完所有結果後寫入(寫入順序與品質無關,因每個 worker 樣本獨立)
    for (feats, labels, ranges, heights, sig_types, render_types,
         source_ranges, source_heights, f0s, yaws, obstacle_counts,
         obstacle_gains, env_build_s, sample_s) in results:
        k = len(labels)
        feats_all[write_ptr:write_ptr + k] = feats
        labels_all[write_ptr:write_ptr + k] = labels
        range_all[write_ptr:write_ptr + k] = ranges
        height_all[write_ptr:write_ptr + k] = heights
        source_ranges_all[write_ptr:write_ptr + k] = source_ranges
        source_heights_all[write_ptr:write_ptr + k] = source_heights
        f0_all[write_ptr:write_ptr + k] = f0s
        yaw_all[write_ptr:write_ptr + k] = yaws
        obstacle_count_all[write_ptr:write_ptr + k] = obstacle_counts
        obstacle_gains_all[write_ptr:write_ptr + k] = obstacle_gains
        sig_all.extend(sig_types)
        render_all.extend(render_types)
        write_ptr += k
        env_build_total += env_build_s
        env_build_max = max(env_build_max, env_build_s)
        sample_total += sample_s

    gen_wall = time.time() - t0
    print()  # 換行收尾

    # 距離 bin 分布(檢查是否嚴重不均 → 影響距離 head 訓練)
    uniq, cnt = np.unique(range_all, return_counts=True)
    dist_str = ", ".join(f"bin{u}:{c/args.n:.0%}" for u, c in zip(uniq, cnt))
    huniq, hcnt = np.unique(height_all, return_counts=True)
    height_str = ", ".join(f"bin{u}:{c/args.n:.0%}" for u, c in zip(huniq, hcnt))
    # 實際混合比例(確認 pyroom_ratio 有生效;含 fallback 拆開看)
    runiq, rcnt = np.unique(np.array(render_all), return_counts=True)
    render_str = ", ".join(f"{u}:{c/args.n:.0%}" for u, c in zip(runiq, rcnt))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_save0 = time.time()
    save_fn = np.savez_compressed if args.compress else np.savez
    save_fn(
        out,
        feats=feats_all,
        labels=labels_all,
        range_labels=range_all,
        height_labels=height_all,
        sig_types=np.array(sig_all),
        render_types=np.array(render_all),
        source_ranges=source_ranges_all,
        source_heights=source_heights_all,
        f0_hz=f0_all,
        self_yaw_deg=yaw_all,
        obstacle_counts=obstacle_count_all,
        obstacle_gains=obstacle_gains_all,
    )
    save_s = time.time() - t_save0

    print(f"✅ 存於 {out} | {args.n} 樣本 | feat_dim={feat_dim} | "
          f"耗時 {_fmt_eta(time.time()-t0)}")
    print(f"   距離 bin 分布: {dist_str}")
    print(f"   高度 bin 分布: {height_str}")
    print(f"   實際渲染比例: {render_str}(目標 pyroom≈{cfg.source_dr.pyroom_ratio:.0%})")
    print(f"   f0 範圍: {f0_all.min():.0f}-{f0_all.max():.0f} Hz | "
          f"障礙物樣本: {(obstacle_count_all > 0).mean():.0%}")
    # --- 計時探針總結:這三行決定下一刀砍哪裡 ---
    print(f"   ⏱  生成牆鐘 {gen_wall:.1f}s | env 建構(最慢 worker){env_build_max:.2f}s "
          f"| 純 DSP sample 合計 {sample_total:.1f}s(平行,跨 {n_workers} workers)")
    print(f"   ⏱  存檔 {save_s:.1f}s ({'壓縮' if args.compress else '不壓縮'})")
    avg_sample_us = sample_total / max(1, args.n) * 1e6
    print(f"   ⏱  單樣本平均 DSP ≈ {avg_sample_us:.0f} µs "
          f"→ 若這項佔大頭,下一刀請看 env.sample()(RIR 快取 / FFT 後端 / 濾波器係數)")
    print(f"   (bin 嚴重不均時,train_gpu 距離 CE 可考慮加 class weight)")


if __name__ == "__main__":
    main()