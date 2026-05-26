"""
model_inspector.py
==================
兩大功能：

  ① 資料集相似度分析（只需 --train / --eval）
     - nearest_neighbor_dist.png   每筆 eval 樣本到最近 train 鄰居的距離分布
                                    （Euclidean + Cosine，判斷 eval 是否 OOD）
     - tsne_split.png              train + eval 聯合 t-SNE，以 split 著色
     - tsne_azimuth.png            同上，以 azimuth bin 著色
     - cosine_heatmap.png          抽樣 cosine 相似度熱圖（eval × train）

  ② 模型「特徵圖」等價物（加上 --weights）
     說明：本模型是純 MLP，不存在 CNN 意義上的空間特徵圖。
     但以下圖等價地展示了「網路學到了什麼」：
     - embedding_tsne_azimuth.png  backbone 128 維隱藏表示 → t-SNE，以 azimuth 著色
     - embedding_tsne_range.png    同上，以 range bin 著色
     - weight_layer1.png           第一層權重熱圖 (128 × 15)，每行是一個神經元對各 mic 特徵的偏好
     - confusion_azimuth.png       方位角預測混淆矩陣（eval set）
     - saliency_mean.png           輸入梯度顯著圖：每個 input feature 對預測貢獻多少

用法：
    # 只看資料相似度（不需要訓練完的模型）
    cd ~/dobot_project/detect
    python model_inspector.py --train data/backup2/train.npz --eval  data/backup2/eval.npz

    # 加入模型特徵分析
    python model_inspector.py --train data/backup2/train.npz --eval  data/backup2/eval.npz --weights runs/runs_舊backup/gpu_run/checkpoints/best_eval.pt

    # 自訂取樣數（t-SNE / NN 搜尋）與輸出目錄
    python model_inspector.py \\
        --train data/train.npz \\
        --eval  data/eval.npz \\
        --weights runs/gpu_run/checkpoints/best_eval.pt \\
        --n-tsne 3000 \\
        --n-nn   5000 \\
        --outdir inspect_model

需求：
    pip install matplotlib scikit-learn seaborn torch numpy scipy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm  # noqa: needed for colormaps compat
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 特徵維度標籤（對應 signal_processing.extract_features_v2）
# MIC_PAIRS = [(0,1),(0,2),(0,3),(0,4),(0,5)]
# 每 pair 產生 3 個特徵：sin(Δφ), cos(Δφ), energy_ratio
# ============================================================

MIC_PAIR_LABELS = [
    ("0-1", "4mm near"),
    ("0-2", "12mm"),
    ("0-3", "28mm far"),
    ("0-4", "12mm vertical"),
    ("0-5", "12mm vertical"),
]

FEAT_LABELS = []
for (pair, desc) in MIC_PAIR_LABELS:
    FEAT_LABELS += [
        f"sin phi ({pair})\n{desc}",
        f"cos phi ({pair})\n{desc}",
        f"E-ratio ({pair})\n{desc}",
    ]


# ============================================================
# 資料載入
# ============================================================

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    feats         = d["feats"].astype(np.float32)
    labels        = d["labels"].astype(np.int32)
    range_labels  = d["range_labels"].astype(np.int32)
    height_labels = d["height_labels"].astype(np.int32) if "height_labels" in d else None
    return feats, labels, range_labels, height_labels


def subsample(arrays: list[np.ndarray], n: int, seed: int = 42) -> list[np.ndarray]:
    """從多個同長度陣列中取相同的 n 筆隨機子集。"""
    rng  = np.random.default_rng(seed)
    size = len(arrays[0])
    if size <= n:
        return arrays
    idx = rng.choice(size, n, replace=False)
    return [a[idx] for a in arrays]


# ============================================================
# ① 資料集相似度
# ============================================================

def plot_nn_distance(
    tr_feats: np.ndarray,
    ev_feats: np.ndarray,
    out_dir: Path,
    n_train: int = 5000,
    seed: int = 42,
):
    """
    每筆 eval 樣本在 feature 空間中找最近的 train 鄰居，
    畫距離分布直方圖。
    - 距離很小 → eval 被 train 良好覆蓋。
    - 長尾或有峰值偏右 → eval 含 OOD 樣本，訓練資料未覆蓋到的情況。
    """
    print("  [NN Distance] Building nearest-neighbor index...")
    tr_sub = subsample([tr_feats], n_train, seed)[0]

    scaler = StandardScaler().fit(tr_sub)
    tr_sc  = scaler.transform(tr_sub)
    ev_sc  = scaler.transform(ev_feats)

    # Euclidean
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
    nn.fit(tr_sc)
    euc_dist, _ = nn.kneighbors(ev_sc)
    euc_dist = euc_dist[:, 0]

    # Cosine（越接近 0 越相似；這裡存 1 - cosine_sim 作為距離）
    cos_sim  = cosine_similarity(ev_sc, tr_sc)   # (n_eval, n_train)
    cos_dist = 1.0 - cos_sim.max(axis=1)          # 最近鄰的 1-cosine

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Eval vs Train nearest-neighbor distance distribution\n"
        "(left peak = eval is covered by train; right tail = possible OOD samples)",
        fontsize=13,
    )

    for ax, dist, title, xlabel in zip(
        axes,
        [euc_dist, cos_dist],
        ["Euclidean distance", "Cosine distance (1 - similarity)"],
        ["L2 distance (standardized feature space)", "1 - cosine similarity"],
    ):
        ax.hist(dist, bins=60, color="#4C8EFF", edgecolor="none", alpha=0.85)
        ax.axvline(np.median(dist), color="red",    linestyle="--", label=f"Median {np.median(dist):.3f}")
        ax.axvline(np.mean(dist),   color="orange", linestyle=":",  label=f"Mean {np.mean(dist):.3f}")
        p95 = np.percentile(dist, 95)
        ax.axvline(p95, color="grey", linestyle="-.", label=f"P95 {p95:.3f}")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Sample count")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_dir / "nearest_neighbor_dist.png", dpi=180)
    plt.close()
    print("  [ok] nearest_neighbor_dist.png")


def plot_joint_tsne(
    tr_feats: np.ndarray, tr_labels: np.ndarray,
    ev_feats: np.ndarray, ev_labels: np.ndarray,
    out_dir: Path,
    n_total: int = 3000,
    seed: int = 42,
):
    """
    Train + eval 聯合 t-SNE，產生兩張圖：
    1. 以 split（train/eval）著色 → 看 train/eval 分布是否重疊
    2. 以 azimuth bin 著色 → 看模型 input space 的可分性
    """
    print("  [t-SNE] Sampling and running t-SNE (may take 30-90 seconds)...")

    n_tr = int(n_total * 0.8)
    n_ev = n_total - n_tr

    tr_sub, tr_lbl_sub = subsample([tr_feats, tr_labels], n_tr, seed)
    ev_sub, ev_lbl_sub = subsample([ev_feats, ev_labels], n_ev, seed + 1)

    combined = np.vstack([tr_sub, ev_sub])
    split_tag = np.array(["train"] * len(tr_sub) + ["eval"] * len(ev_sub))
    az_tag    = np.concatenate([tr_lbl_sub, ev_lbl_sub])

    scaler   = StandardScaler()
    combined_sc = scaler.fit_transform(combined)

    tsne = TSNE(n_components=2, perplexity=40, random_state=seed, n_jobs=-1)
    emb  = tsne.fit_transform(combined_sc)

    # ---- 圖1: split 著色 ----
    fig, ax = plt.subplots(figsize=(9, 8))
    colors  = {"train": "#4C8EFF", "eval": "#FF6B6B"}
    alphas  = {"train": 0.35, "eval": 0.8}
    sizes   = {"train": 8, "eval": 18}
    for split in ["train", "eval"]:
        mask = split_tag == split
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            s=sizes[split], c=colors[split],
            alpha=alphas[split], label=split, edgecolors="none",
        )
    ax.set_title("t-SNE: Train vs Eval distribution overlap\n(more overlap = eval is better covered by train)", fontsize=12)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=3, fontsize=11)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_split.png", dpi=180)
    plt.close()
    print("  [ok] tsne_split.png")

    # ---- 圖2: azimuth bin 著色 ----
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=az_tag, cmap="hsv", s=8, alpha=0.5, edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Azimuth Bin (0-71 -> 0-355 deg)", fontsize=10)
    ax.set_title("t-SNE colored by azimuth label\n(continuous color bands mean neighboring directions stay nearby)", fontsize=12)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_azimuth.png", dpi=180)
    plt.close()
    print("  [ok] tsne_azimuth.png")


def plot_cosine_heatmap(
    tr_feats: np.ndarray,
    ev_feats: np.ndarray,
    out_dir: Path,
    n_sample: int = 300,
    seed: int = 42,
):
    """
    抽樣後計算 eval × train 的 cosine 相似度熱圖。
    每行是一筆 eval 樣本，每列是一筆 train 樣本。
    熱圖若有明確的對角線塊狀結構 = 資料有良好的 cluster 結構（可分性高）。
    """
    print("  [Cosine Heatmap] Computing...")

    tr_sub = subsample([tr_feats], n_sample, seed)[0]
    ev_sub = subsample([ev_feats], n_sample, seed + 1)[0]

    scaler = StandardScaler().fit(tr_sub)
    sim    = cosine_similarity(
        scaler.transform(ev_sub),
        scaler.transform(tr_sub),
    )  # (n_sample, n_sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns_h = sns.heatmap(
        sim, ax=ax, cmap="RdYlGn",
        vmin=-1, vmax=1, center=0,
        xticklabels=False, yticklabels=False,
        cbar_kws={"label": "Cosine Similarity"},
    )
    ax.set_title(
        f"Cosine similarity heatmap (eval {n_sample} x train {n_sample}, standardized)\n"
        "Rows = eval samples, columns = train samples; greener = more similar",
        fontsize=11,
    )
    ax.set_xlabel(f"Train samples ({n_sample} sampled)")
    ax.set_ylabel(f"Eval samples ({n_sample} sampled)")
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_heatmap.png", dpi=180)
    plt.close()
    print("  [ok] cosine_heatmap.png")


# ============================================================
# ② 模型特徵視覺化
# ============================================================

def load_model(ckpt_path: str):
    """載入 checkpoint，回傳 (net, cfg, device)。"""
    import torch
    from config import DEFAULT
    from model import build_net

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt["model"]
    has_range  = any("range"  in k for k in state.keys())
    has_height = any("height" in k for k in state.keys())

    net = build_net(DEFAULT, with_range=has_range, with_height=has_height).to(device)
    net.load_state_dict(state)
    net.eval()

    print(f"  Model loaded: has_range={has_range}, has_height={has_height}, device={device}")
    return net, DEFAULT, device


def extract_embeddings(
    net,
    feats: np.ndarray,
    device,
    batch_size: int = 1024,
) -> np.ndarray:
    """對 feats 跑 backbone，回傳 (N, hidden) 的 embedding。"""
    import torch

    net.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            x = torch.tensor(feats[i:i + batch_size], device=device)
            h = net.backbone(x)        # (B, hidden=128)
            embeddings.append(h.cpu().numpy())
    return np.vstack(embeddings)


def plot_embedding_tsne(
    net,
    tr_feats: np.ndarray, tr_labels: np.ndarray, tr_range_labels: np.ndarray,
    ev_feats: np.ndarray, ev_labels: np.ndarray, ev_range_labels: np.ndarray,
    device,
    out_dir: Path,
    n_total: int = 3000,
    seed: int = 42,
):
    """
    backbone embedding（128 維）→ t-SNE。
    這是 MLP 版本的「特徵圖」：展示網路學到的隱藏表示空間。
    分兩張圖：以 azimuth 著色 / 以 range bin 著色。
    好的 embedding 應該讓相鄰方向的點聚集在一起（色帶連續）。
    """
    print("  [Embedding t-SNE] Preparing backbone outputs...")

    n_tr = int(n_total * 0.7)
    n_ev = n_total - n_tr

    tr_sub, tr_lbl_sub, tr_rng_sub = subsample([tr_feats, tr_labels, tr_range_labels], n_tr, seed)
    ev_sub, ev_lbl_sub, ev_rng_sub = subsample([ev_feats, ev_labels, ev_range_labels], n_ev, seed + 1)

    combined = np.vstack([tr_sub, ev_sub])
    az_tags  = np.concatenate([tr_lbl_sub, ev_lbl_sub])
    rng_tags = np.concatenate([tr_rng_sub, ev_rng_sub])
    split_tags = np.array(["train"] * len(tr_sub) + ["eval"] * len(ev_sub))

    print("  [Embedding t-SNE] Extracting embeddings...")
    embs = extract_embeddings(net, combined, device)

    print("  [Embedding t-SNE] Running t-SNE (may take 30-90 seconds)...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=seed, n_jobs=-1)
    emb2d = tsne.fit_transform(embs)

    # ---- azimuth 著色 ----
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=az_tags, cmap="hsv",
                    s=8, alpha=0.55, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03)
    cbar.set_label("Azimuth Bin (0-71)", fontsize=10)
    ax.set_title(
        "Backbone embedding t-SNE colored by azimuth\n"
        "Smooth color bands mean the model learned a continuous direction representation",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_tsne_azimuth.png", dpi=180)
    plt.close()
    print("  [ok] embedding_tsne_azimuth.png")

    # ---- range bin 著色 ----
    n_range_bins = int(rng_tags.max()) + 1
    try:
        cmap_r = matplotlib.colormaps["plasma"].resampled(n_range_bins)
    except AttributeError:
        cmap_r = plt.cm.get_cmap("plasma", n_range_bins)  # noqa: deprecated in mpl ≥ 3.7
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=rng_tags, cmap=cmap_r,
                    s=8, alpha=0.55, edgecolors="none",
                    vmin=-0.5, vmax=n_range_bins - 0.5)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, ticks=range(n_range_bins))
    cbar.set_label("Range Bin", fontsize=10)
    ax.set_title(
        "Backbone embedding t-SNE colored by range bin\n"
        "Clear clusters mean range information is encoded in the backbone",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_tsne_range.png", dpi=180)
    plt.close()
    print("  [ok] embedding_tsne_range.png")


def plot_weight_layer1(net, out_dir: Path):
    """
    第一層權重熱圖：shape (128, 15) → 每行是一個隱藏神經元，
    每列是一個輸入特徵（15 = 5 mic pair × 3）。
    亮色/暗色 = 該神經元正/負偏好該特徵。
    按 5 個 mic pair 分組標示，方便看哪些 pair 被哪些神經元採用。
    """
    W = net.backbone[0].weight.detach().cpu().numpy()   # (128, 15)

    # 按每個 mic pair 的平均絕對權重排序神經元
    pair_importance = np.array([
        np.abs(W[:, i*3:(i+1)*3]).mean(axis=1) for i in range(5)
    ]).T  # (128, 5)
    neuron_order = np.argsort(-pair_importance.max(axis=1))
    W_sorted = W[neuron_order]

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(W_sorted, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(W).max(), vmax=np.abs(W).max())
    plt.colorbar(im, ax=ax, fraction=0.02, label="Weight")

    # x 軸：特徵名稱（縮短版）
    short_labels = []
    for i, (pair, _) in enumerate(MIC_PAIR_LABELS):
        short_labels += [f"sin\n({pair})", f"cos\n({pair})", f"E\n({pair})"]
    ax.set_xticks(range(15))
    ax.set_xticklabels(short_labels, fontsize=8)

    # 分組線
    for i in range(1, 5):
        ax.axvline(i * 3 - 0.5, color="yellow", linewidth=1.2, alpha=0.6)

    ax.set_xlabel("Input features (each 3 columns = one mic pair)", fontsize=11)
    ax.set_ylabel("Hidden neurons (sorted by max pair weight)", fontsize=11)
    ax.set_title(
        "First-layer weight heatmap (128 neurons x 15 features)\n"
        "Red = positive weight, blue = negative weight; yellow lines split mic pairs",
        fontsize=12,
    )

    # 標示 pair 分組
    pair_names = [f"pair {p}" for p, _ in MIC_PAIR_LABELS]
    for i, name in enumerate(pair_names):
        ax.text(i * 3 + 1, -3.5, name, ha="center", va="top", fontsize=8, color="gold")

    plt.tight_layout()
    plt.savefig(out_dir / "weight_layer1.png", dpi=180)
    plt.close()
    print("  [ok] weight_layer1.png")


def plot_confusion_azimuth(
    net,
    ev_feats: np.ndarray,
    ev_labels: np.ndarray,
    cfg,
    device,
    out_dir: Path,
    n_samples: int = 5000,
    seed: int = 42,
):
    """
    方位角混淆矩陣（eval set 上的預測 vs 真值）。
    對角線越亮 = 預測越準；對角線附近的亮帶 = 只差幾個 bin（可接受的誤差）。
    若有非對角線的亮塊 = 系統性誤差（如前/後混淆、相位模糊）。
    """
    import torch

    print("  [Confusion Matrix] Predicting...")

    sub_feats, sub_labels = subsample([ev_feats, ev_labels], n_samples, seed)
    n_bins = cfg.task.n_azimuth_bins

    all_preds = []
    with torch.no_grad():
        batch = 512
        for i in range(0, len(sub_feats), batch):
            x = torch.tensor(sub_feats[i:i+batch], device=device)
            out = net(x)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
    preds = np.concatenate(all_preds)

    # 計算混淆矩陣（正規化為 row-wise recall）
    conf = np.zeros((n_bins, n_bins), dtype=np.float32)
    for t, p in zip(sub_labels, preds):
        conf[t, p] += 1
    row_sum = conf.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    conf_norm = conf / row_sum

    # 計算 angular error（每個 true bin 的平均預測 bin 誤差，考慮循環）
    mean_err_per_bin = []
    for t in range(n_bins):
        row = preds[sub_labels == t]
        if len(row) == 0:
            mean_err_per_bin.append(0.0)
            continue
        diff = np.abs(row - t)
        diff = np.minimum(diff, n_bins - diff)
        mean_err_per_bin.append(diff.mean() * (360.0 / n_bins))

    fig = plt.figure(figsize=(15, 7))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)

    # 混淆矩陣
    ax0 = fig.add_subplot(gs[0])
    im  = ax0.imshow(conf_norm, aspect="auto", cmap="Blues",
                     interpolation="nearest", vmin=0, vmax=conf_norm.max())
    plt.colorbar(im, ax=ax0, fraction=0.03, label="Recall (row-normalized)")
    tick_step = 12   # 每 12 bin (=60°) 標一次
    ticks = list(range(0, n_bins, tick_step))
    tick_degs = [f"{int(t*360/n_bins)} deg" for t in ticks]
    ax0.set_xticks(ticks); ax0.set_xticklabels(tick_degs)
    ax0.set_yticks(ticks); ax0.set_yticklabels(tick_degs)
    ax0.set_xlabel("Predicted azimuth"); ax0.set_ylabel("True azimuth")
    ax0.set_title(
        f"Azimuth confusion matrix (eval {n_samples} samples, row-normalized)\n"
        "Bright diagonal = accurate; shifted bands = systematic offset",
        fontsize=11,
    )

    # 每個 bin 的平均誤差 bar chart
    ax1 = fig.add_subplot(gs[1])
    bin_degs = np.arange(n_bins) * (360.0 / n_bins)
    ax1.barh(bin_degs, mean_err_per_bin, height=360.0 / n_bins * 0.85,
             color="#4C8EFF", alpha=0.8)
    ax1.set_xlabel("Mean angular error (deg)")
    ax1.set_ylabel("True azimuth (deg)")
    ax1.set_title("Mean error per bin", fontsize=10)
    ax1.axvline(10, color="red", linestyle="--", linewidth=1, label="10 deg threshold")
    ax1.legend(fontsize=9)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_azimuth.png", dpi=180)
    plt.close()
    print("  [ok] confusion_azimuth.png")


def plot_saliency(
    net,
    ev_feats: np.ndarray,
    ev_labels: np.ndarray,
    device,
    out_dir: Path,
    n_samples: int = 2000,
    seed: int = 42,
):
    """
    輸入梯度顯著圖（Gradient-based Saliency）：
    對每筆 eval 樣本，計算「正確類別 logit 對輸入特徵的梯度」，
    取絕對值後對所有樣本取平均 → 每個輸入維度的「重要性」。

    這是 MLP 版的「feature importance map」——
    顯示模型在推論時最依賴哪些 mic pair 和哪種特徵（相位差 vs 能量比）。
    """
    import torch

    print("  [Saliency] Computing gradient saliency...")

    sub_feats, sub_labels = subsample([ev_feats, ev_labels], n_samples, seed)
    feat_dim = sub_feats.shape[1]

    saliency_sum = np.zeros(feat_dim, dtype=np.float64)
    count = 0

    batch = 256
    for i in range(0, len(sub_feats), batch):
        x_np = sub_feats[i:i + batch]
        y_np = sub_labels[i:i + batch]

        x = torch.tensor(x_np, device=device, requires_grad=True)
        out = net(x)
        logits = out[0] if isinstance(out, tuple) else out   # (B, n_bins)

        # 每筆取真正類別的 logit
        true_logits = logits[torch.arange(len(y_np)), torch.tensor(y_np, device=device)]
        true_logits.sum().backward()

        grad = x.grad.abs().detach().cpu().numpy()  # (B, feat_dim)
        saliency_sum += grad.sum(axis=0)
        count += len(x_np)

    mean_saliency = saliency_sum / count  # (feat_dim,)

    # 正規化到 [0,1]
    sal_norm = mean_saliency / (mean_saliency.max() + 1e-8)

    # 按 mic pair 分組著色
    pair_colors = ["#4C8EFF", "#FF6B6B", "#FFD93D", "#6BCB77", "#C77DFF"]
    bar_colors  = []
    for i in range(5):
        bar_colors += [pair_colors[i]] * 3

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(feat_dim), sal_norm, color=bar_colors, edgecolor="none", alpha=0.88)

    # 分組線與標籤
    for i in range(1, 5):
        ax.axvline(i * 3 - 0.5, color="grey", linewidth=1, linestyle="--", alpha=0.5)

    ax.set_xticks(range(feat_dim))
    short_labs = []
    for (pair, _) in MIC_PAIR_LABELS:
        short_labs += [f"sin\n{pair}", f"cos\n{pair}", f"E\n{pair}"]
    ax.set_xticklabels(short_labs, fontsize=8)

    ax.set_ylabel("Normalized mean absolute gradient (importance)", fontsize=11)
    ax.set_xlabel("Input features (each 3 columns = one mic pair)", fontsize=11)
    ax.set_title(
        "Input gradient saliency: which features matter for azimuth prediction?\n"
        "High bar = heavily used by the model; low bar = barely used",
        fontsize=12,
    )

    # pair 圖例
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=pair_colors[i], label=f"pair {MIC_PAIR_LABELS[i][0]} {MIC_PAIR_LABELS[i][1]}")
        for i in range(5)
    ]
    ax.legend(handles=legend_items, fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_dir / "saliency_mean.png", dpi=180)
    plt.close()
    print("  [ok] saliency_mean.png")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Dataset similarity analysis and model feature visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--train",   required=True,  help="training .npz path")
    ap.add_argument("--eval",    required=True,  help="evaluation .npz path")
    ap.add_argument("--weights", default=None,   help="model checkpoint .pt (optional)")
    ap.add_argument("--outdir",  default="inspect_model")
    ap.add_argument("--n-tsne",  type=int, default=3000,
                    help="total t-SNE samples (train + eval, default 3000)")
    ap.add_argument("--n-nn",    type=int, default=5000,
                    help="train subset size for NN distance search (default 5000)")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}\n")

    # ---- 載入資料 ----
    print("Loading training set...")
    tr_feats, tr_labels, tr_range_labels, tr_height_labels = load_npz(args.train)
    print(f"  train: {tr_feats.shape}  feat_dim={tr_feats.shape[1]}")

    print("Loading evaluation set...")
    ev_feats, ev_labels, ev_range_labels, ev_height_labels = load_npz(args.eval)
    print(f"  eval:  {ev_feats.shape}")

    # ──────────────────────────────────────────
    # ① 資料集相似度分析
    # ──────────────────────────────────────────
    print("\n=== 1. Dataset similarity analysis ===")

    plot_nn_distance(
        tr_feats, ev_feats, out_dir,
        n_train=args.n_nn, seed=args.seed,
    )
    plot_joint_tsne(
        tr_feats, tr_labels,
        ev_feats, ev_labels,
        out_dir, n_total=args.n_tsne, seed=args.seed,
    )
    plot_cosine_heatmap(
        tr_feats, ev_feats, out_dir, seed=args.seed,
    )

    # ──────────────────────────────────────────
    # ② 模型特徵視覺化
    # ──────────────────────────────────────────
    if args.weights is None:
        print("\nNo --weights provided; skipping model feature analysis.")
        print("  Add --weights <checkpoint.pt> to inspect model features.")
    else:
        print("\n=== 2. Model feature visualization ===")
        net, cfg, device = load_model(args.weights)

        plot_embedding_tsne(
            net,
            tr_feats, tr_labels, tr_range_labels,
            ev_feats, ev_labels, ev_range_labels,
            device, out_dir,
            n_total=args.n_tsne, seed=args.seed,
        )
        plot_weight_layer1(net, out_dir)
        plot_confusion_azimuth(
            net, ev_feats, ev_labels, cfg, device, out_dir, seed=args.seed,
        )
        plot_saliency(
            net, ev_feats, ev_labels, device, out_dir, seed=args.seed,
        )

    # ──────────────────────────────────────────
    # 完成
    # ──────────────────────────────────────────
    output_files = sorted(out_dir.glob("*.png"))
    print(f"\nDone. Generated {len(output_files)} plots:")
    for f in output_files:
        print(f"   {f.name}")
    print(f"\nOutput directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
