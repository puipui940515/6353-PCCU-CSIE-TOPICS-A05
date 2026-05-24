"""
dataset_inspector.py

可視化你的聲源定位資料集：
- 麥克風 / 聲源 幾何圖
- feature heatmap
- PCA 2D 分布
- label histogram
- range histogram
- signal type 分布

使用方式:
    python dataset_inspector.py --data data/train.npz

需求:
    pip install matplotlib scikit-learn seaborn

輸出:
    inspect_output/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================
# 載入資料
# =========================

def load_dataset(path: str):
    data = np.load(path, allow_pickle=True)

    feats = data["feats"]
    labels = data["labels"]
    range_labels = data["range_labels"]

    sig_types = None
    if "sig_types" in data:
        sig_types = data["sig_types"]

    return feats, labels, range_labels, sig_types


# =========================
# 幾何圖
# =========================

def plot_geometry(out_dir: Path):
    """
    範例 6 麥克風圓形陣列
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    # 麥克風位置
    radius = 0.08
    mic_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)

    mic_x = radius * np.cos(mic_angles)
    mic_y = radius * np.sin(mic_angles)

    # 聲源位置
    src_x = 0.8
    src_y = 0.4

    # 畫 mic
    ax.scatter(mic_x, mic_y, s=120, label="Microphones")

    for i, (x, y) in enumerate(zip(mic_x, mic_y)):
        ax.text(x, y, f"mic{i}", fontsize=10)

    # 畫 source
    ax.scatter(src_x, src_y, s=200, marker="*", label="Source")

    # 畫路徑
    for x, y in zip(mic_x, mic_y):
        ax.plot([src_x, x], [src_y, y], alpha=0.4)

    ax.set_title("Microphone Array Geometry")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.axis("equal")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "geometry.png", dpi=200)
    plt.close()


# =========================
# Feature Heatmap
# =========================

def plot_feature_heatmap(feats, out_dir: Path, n_samples=128):
    fig, ax = plt.subplots(figsize=(12, 6))

    sample_feats = feats[:n_samples]

    im = ax.imshow(
        sample_feats,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_title("Feature Heatmap")
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Sample Index")

    fig.colorbar(im)

    plt.tight_layout()
    plt.savefig(out_dir / "feature_heatmap.png", dpi=200)
    plt.close()


# =========================
# PCA
# =========================

def plot_pca(feats, labels, out_dir: Path):
    scaler = StandardScaler()
    feats_std = scaler.fit_transform(feats)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(feats_std)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        s=5,
        alpha=0.6
    )

    ax.set_title("PCA Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Azimuth Bin")

    plt.tight_layout()
    plt.savefig(out_dir / "pca_projection.png", dpi=200)
    plt.close()


# =========================
# Label Histogram
# =========================

def plot_label_hist(labels, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(labels, bins=len(np.unique(labels)))

    ax.set_title("Azimuth Label Distribution")
    ax.set_xlabel("Azimuth Bin")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_dir / "azimuth_distribution.png", dpi=200)
    plt.close()


# =========================
# Range Histogram
# =========================

def plot_range_hist(range_labels, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(range_labels, bins=len(np.unique(range_labels)))

    ax.set_title("Range Label Distribution")
    ax.set_xlabel("Range Bin")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_dir / "range_distribution.png", dpi=200)
    plt.close()


# =========================
# Signal Type
# =========================

def plot_signal_types(sig_types, out_dir: Path):
    if sig_types is None:
        return

    uniq, cnt = np.unique(sig_types, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(uniq.astype(str), cnt)

    ax.set_title("Signal Type Distribution")
    ax.set_ylabel("Count")

    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.savefig(out_dir / "signal_type_distribution.png", dpi=200)
    plt.close()


# =========================
# Feature Statistics
# =========================

def print_stats(feats, labels, range_labels):
    print("\n========== DATASET STATS ==========")

    print(f"Samples:        {len(feats)}")
    print(f"Feature Dim:    {feats.shape[1]}")

    print("\nFeature Stats:")
    print(f"  mean: {feats.mean():.4f}")
    print(f"  std : {feats.std():.4f}")
    print(f"  min : {feats.min():.4f}")
    print(f"  max : {feats.max():.4f}")

    print("\nAzimuth Labels:")
    uniq, cnt = np.unique(labels, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"  bin {u}: {c}")

    print("\nRange Labels:")
    uniq, cnt = np.unique(range_labels, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"  bin {u}: {c}")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="path to .npz dataset"
    )

    ap.add_argument(
        "--outdir",
        type=str,
        default="inspect_output"
    )

    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    feats, labels, range_labels, sig_types = load_dataset(args.data)

    print_stats(feats, labels, range_labels)

    print("\nGenerating plots...")

    plot_geometry(out_dir)
    plot_feature_heatmap(feats, out_dir)
    plot_pca(feats, labels, out_dir)
    plot_label_hist(labels, out_dir)
    plot_range_hist(range_labels, out_dir)
    plot_signal_types(sig_types, out_dir)

    print("\nDone.")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()