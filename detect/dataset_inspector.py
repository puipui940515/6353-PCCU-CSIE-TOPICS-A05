"""
dataset_inspector.py  (v2 — 適配 gen_dataset v2 輸出格式)

可視化你的聲源定位資料集：
  幾何類
  - geometry.png              6 麥克風圓形陣列 + 聲源示意（2D 俯視）
  - geometry_3d.png           麥克風陣列 + 聲源示意（3D，含高度資訊）
  特徵類
  - feature_heatmap.png       前 N 筆 feats 熱圖
  - pca_projection.png        PCA 2D，以方位角著色
  分布類（label）
  - azimuth_distribution.png  方位角 bin 計數
  - range_distribution.png    距離 bin 計數
  - height_distribution.png   高度 bin 計數（有 height_labels 才輸出）
  分布類（連續量，有才輸出）
  - source_range_m.png        聲源距離連續值分布
  - source_height_m.png       聲源高度連續值分布
  - f0_distribution.png       發聲基頻分布
  - self_yaw_distribution.png 末端 yaw 分布（DR 驗證）
  分布類（meta）
  - signal_type_distribution.png  信號類型
  - render_type_distribution.png  渲染類型（pyroom vs free_field）
  - obstacle_count_distribution.png 障礙物數量
  - obstacle_gain_heatmap.png     各通道衰減係數熱圖（抽樣）

使用方式:
    cd ~/dobot_project/detect
    python dataset_inspector.py --data data/train.npz
    python dataset_inspector.py --data data/train.npz --outdir inspect_output --n-heatmap 256

需求:
    pip install matplotlib scikit-learn seaborn numpy

輸出:
    inspect_output/  (預設)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================
# 載入資料
# =========================

def load_dataset(path: str) -> dict:
    """載入 .npz，回傳所有欄位的 dict（欄位不存在時值為 None）。"""
    data = np.load(path, allow_pickle=True)

    def _get(key, cast=None):
        if key not in data:
            return None
        arr = data[key]
        return arr.astype(cast) if cast is not None else arr

    d = {
        # 訓練必要
        "feats":          _get("feats",         np.float32),
        "labels":         _get("labels",        np.int64),
        "range_labels":   _get("range_labels",  np.int64),
        # v1 或 v2 均有
        "height_labels":  _get("height_labels", np.int64),
        "sig_types":      _get("sig_types"),
        # v2 新增
        "render_types":   _get("render_types"),
        "source_ranges":  _get("source_ranges",  np.float32),
        "source_heights": _get("source_heights", np.float32),
        "f0_hz":          _get("f0_hz",          np.float32),
        "self_yaw_deg":   _get("self_yaw_deg",   np.float32),
        "obstacle_counts":_get("obstacle_counts",np.int64),
        "obstacle_gains": _get("obstacle_gains", np.float32),
    }
    return d


# =========================
# 幾何圖（2D）
# =========================

def plot_geometry(out_dir: Path):
    """6 麥克風圓形陣列（2D 俯視）。"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # 真實 mic layout（來自 config.py mic_layout）
    mic_positions = np.array([
        [0.000,  0.000],
        [0.004,  0.000],
        [0.012,  0.000],
        [0.028,  0.000],
        [0.000,  0.012],
        [0.000, -0.012],
    ]) * 1000  # → mm

    ax.scatter(mic_positions[:, 0], mic_positions[:, 1],
               s=150, zorder=5, label="Microphone")
    for i, (x, y) in enumerate(mic_positions):
        ax.annotate(f"mic{i}", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=9)

    # 示意聲源（工作空間中隨機一點）
    src_x, src_y = 180.0, 80.0
    ax.scatter(src_x, src_y, s=250, marker="*", color="red", label="Source (example)")
    for x, y in mic_positions:
        ax.plot([src_x, x], [src_y, y], "r-", alpha=0.2, linewidth=0.8)

    # 工作空間圓
    theta = np.linspace(0, 2 * np.pi, 300)
    for r_mm in [50, 160, 300]:
        ax.plot(r_mm * np.cos(theta), r_mm * np.sin(theta),
                "--", alpha=0.25, color="grey", linewidth=0.8)
        ax.text(r_mm, 5, f"{r_mm}mm", fontsize=7, color="grey")

    ax.set_title("Microphone Array Geometry (Top View, mm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend(fontsize=9)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "geometry.png", dpi=200)
    plt.close()
    print("  [ok] geometry.png")


# =========================
# 幾何圖（3D）
# =========================

def plot_geometry_3d(out_dir: Path):
    """含高度的 3D 麥克風陣列示意圖。"""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    mic_positions = np.array([
        [0.000,  0.000, 0.0],
        [0.004,  0.000, 0.0],
        [0.012,  0.000, 0.0],
        [0.028,  0.000, 0.0],
        [0.000,  0.012, 0.0],
        [0.000, -0.012, 0.0],
    ]) * 1000  # → mm

    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2],
               s=80, c="steelblue", zorder=5, label="Microphone")
    for i, (x, y, z) in enumerate(mic_positions):
        ax.text(x, y, z + 3, f"m{i}", fontsize=7)

    # 示意聲源（含高度）
    src = np.array([120.0, 60.0, 90.0])
    ax.scatter(*src, s=200, marker="*", color="red", label="Source (example, z=90mm)")
    for mp in mic_positions:
        ax.plot([src[0], mp[0]], [src[1], mp[1]], [src[2], mp[2]],
                "r-", alpha=0.18, linewidth=0.7)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Microphone Array Geometry (3D)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "geometry_3d.png", dpi=200)
    plt.close()
    print("  [ok] geometry_3d.png")


# =========================
# Feature Heatmap
# =========================

def plot_feature_heatmap(feats: np.ndarray, out_dir: Path, n_samples: int = 128):
    fig, ax = plt.subplots(figsize=(12, 6))
    sample_feats = feats[:n_samples]
    im = ax.imshow(sample_feats, aspect="auto", interpolation="nearest")
    ax.set_title(f"Feature Heatmap (first {n_samples} samples)")
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Sample Index")

    # 標 mic pair 分組線
    for i in range(1, 5):
        ax.axvline(i * 3 - 0.5, color="yellow", linewidth=1, alpha=0.7)
    fig.colorbar(im)

    plt.tight_layout()
    plt.savefig(out_dir / "feature_heatmap.png", dpi=200)
    plt.close()
    print("  [ok] feature_heatmap.png")


# =========================
# PCA
# =========================

def plot_pca(feats: np.ndarray, labels: np.ndarray, out_dir: Path):
    scaler = StandardScaler()
    feats_std = scaler.fit_transform(feats)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(feats_std)

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1],
                         c=labels, s=4, alpha=0.5, cmap="hsv")
    ax.set_title("PCA Projection (colored by azimuth bin)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Azimuth Bin")

    plt.tight_layout()
    plt.savefig(out_dir / "pca_projection.png", dpi=200)
    plt.close()
    print("  [ok] pca_projection.png")


# =========================
# Label Histograms（bin 類）
# =========================

def plot_label_hist(labels: np.ndarray, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(labels, bins=len(np.unique(labels)), color="#4C8EFF", edgecolor="none", alpha=0.85)
    ax.set_title("Azimuth Label Distribution")
    ax.set_xlabel("Azimuth Bin (0 = 0 deg, max = 355 deg)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "azimuth_distribution.png", dpi=200)
    plt.close()
    print("  [ok] azimuth_distribution.png")


def plot_range_hist(range_labels: np.ndarray, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    uniq = np.unique(range_labels)
    ax.hist(range_labels, bins=len(uniq), color="#FF6B6B", edgecolor="white", alpha=0.85)
    ax.set_title("Range Label Distribution")
    ax.set_xlabel("Range Bin")
    ax.set_ylabel("Count")
    # 標邊界說明
    ax.text(0.01, 0.96, "bin0:<0.08m  bin1:0.08-0.16m  bin2:0.16-0.24m  bin3:>0.24m",
            transform=ax.transAxes, fontsize=8, va="top", color="grey")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "range_distribution.png", dpi=200)
    plt.close()
    print("  [ok] range_distribution.png")


def plot_height_hist(height_labels: np.ndarray | None, out_dir: Path):
    if height_labels is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(height_labels, bins=len(np.unique(height_labels)),
            color="#6BCB77", edgecolor="white", alpha=0.85)
    ax.set_title("Height Label Distribution")
    ax.set_xlabel("Height Bin")
    ax.set_ylabel("Count")
    ax.text(0.01, 0.96, "bin0:<0.04m  bin1:0.04-0.08m  bin2:0.08-0.12m  "
            "bin3:0.12-0.16m  bin4:>0.16m",
            transform=ax.transAxes, fontsize=7, va="top", color="grey")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "height_distribution.png", dpi=200)
    plt.close()
    print("  [ok] height_distribution.png")


# =========================
# 連續量分布（v2 新增）
# =========================

def plot_continuous(arr: np.ndarray | None, title: str, xlabel: str,
                    fname: str, out_dir: Path, color: str = "#4C8EFF", bins: int = 60):
    if arr is None:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(arr, bins=bins, color=color, edgecolor="none", alpha=0.85)
    ax.axvline(np.median(arr), color="red",    linestyle="--",
               label=f"Median {np.median(arr):.3f}")
    ax.axvline(np.mean(arr),   color="orange", linestyle=":",
               label=f"Mean {np.mean(arr):.3f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()
    print(f"  [ok] {fname}")


# =========================
# 類別分布（sig_types / render_types）
# =========================

def plot_categorical(arr: np.ndarray | None, title: str, fname: str,
                     out_dir: Path, color: str = "#C77DFF"):
    if arr is None:
        return
    uniq, cnt = np.unique(arr, return_counts=True)
    fig, ax = plt.subplots(figsize=(max(6, len(uniq) * 1.5), 4))
    bars = ax.bar(uniq.astype(str), cnt, color=color, alpha=0.85)
    for bar, c in zip(bars, cnt):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * cnt.max(),
                f"{c:,}\n({c/len(arr)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()
    print(f"  [ok] {fname}")


# =========================
# 障礙物通道衰減熱圖
# =========================

def plot_obstacle_gain_heatmap(obstacle_gains: np.ndarray | None,
                               obstacle_counts: np.ndarray | None,
                               out_dir: Path, n_samples: int = 300):
    if obstacle_gains is None:
        return
    # 只取有障礙物的樣本
    if obstacle_counts is not None:
        mask = obstacle_counts > 0
        gains = obstacle_gains[mask]
    else:
        gains = obstacle_gains
    if len(gains) == 0:
        return
    # 隨機取樣
    rng = np.random.default_rng(42)
    idx = rng.choice(len(gains), min(n_samples, len(gains)), replace=False)
    gains_sub = gains[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：熱圖
    im = axes[0].imshow(gains_sub.T, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn",
                        interpolation="nearest")
    axes[0].set_xlabel(f"Sample ({len(gains_sub)} obstacle samples)")
    axes[0].set_ylabel("Mic Channel")
    axes[0].set_yticks(range(gains_sub.shape[1]))
    axes[0].set_yticklabels([f"mic{i}" for i in range(gains_sub.shape[1])])
    axes[0].set_title("Obstacle Channel Gain Heatmap\n(1=no attenuation, <1=blocked)")
    fig.colorbar(im, ax=axes[0], label="Gain")

    # 右：各通道平均衰減
    mean_gain = gains_sub.mean(axis=0)
    axes[1].bar(range(len(mean_gain)), mean_gain, color="#FF6B6B", alpha=0.85)
    axes[1].set_xlabel("Mic Channel")
    axes[1].set_ylabel("Mean Gain (obstacle samples)")
    axes[1].set_xticks(range(len(mean_gain)))
    axes[1].set_xticklabels([f"mic{i}" for i in range(len(mean_gain))])
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(1.0, color="green", linestyle="--", alpha=0.5, label="No attenuation")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Per-Channel Mean Gain (obstacle samples)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "obstacle_gain_heatmap.png", dpi=200)
    plt.close()
    print("  [ok] obstacle_gain_heatmap.png")


# =========================
# Feature Statistics
# =========================

def print_stats(d: dict):
    feats         = d["feats"]
    labels        = d["labels"]
    range_labels  = d["range_labels"]
    height_labels = d["height_labels"]

    print("\n========== DATASET STATS ==========")
    print(f"Samples:        {len(feats):,}")
    print(f"Feature Dim:    {feats.shape[1]}")

    print("\nFeature Stats:")
    print(f"  mean: {feats.mean():.4f}")
    print(f"  std : {feats.std():.4f}")
    print(f"  min : {feats.min():.4f}")
    print(f"  max : {feats.max():.4f}")

    print("\nAzimuth Labels:")
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"  bins: {len(uniq)}  range: [{uniq.min()}, {uniq.max()}]")
    imbalance = cnt.max() / cnt.min() if cnt.min() > 0 else float("inf")
    print(f"  imbalance (max/min count): {imbalance:.2f}")

    print("\nRange Labels:")
    uniq, cnt = np.unique(range_labels, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"  bin {u}: {c:,} ({c/len(labels)*100:.1f}%)")

    if height_labels is not None:
        print("\nHeight Labels:")
        uniq, cnt = np.unique(height_labels, return_counts=True)
        for u, c in zip(uniq, cnt):
            print(f"  bin {u}: {c:,} ({c/len(labels)*100:.1f}%)")

    # v2 新增欄位摘要
    for key, label in [
        ("render_types",   "Render type"),
        ("sig_types",      "Signal type"),
    ]:
        arr = d.get(key)
        if arr is not None:
            print(f"\n{label}:")
            uniq, cnt = np.unique(arr, return_counts=True)
            for u, c in zip(uniq, cnt):
                print(f"  {u}: {c:,} ({c/len(labels)*100:.1f}%)")

    for key, label in [
        ("source_ranges",  "Source range (m)"),
        ("source_heights", "Source height (m)"),
        ("f0_hz",          "F0 (Hz)"),
        ("self_yaw_deg",   "Self yaw (deg)"),
    ]:
        arr = d.get(key)
        if arr is not None:
            print(f"\n{label}: min={arr.min():.3f}  mean={arr.mean():.3f}  "
                  f"max={arr.max():.3f}  std={arr.std():.3f}")

    obs = d.get("obstacle_counts")
    if obs is not None:
        print(f"\nObstacle samples: {(obs > 0).sum():,} ({(obs > 0).mean()*100:.1f}%)")

    print("====================================\n")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser(
        description="Acoustic localization dataset inspector (v2)"
    )
    ap.add_argument("--data",      type=str, required=True, help="path to .npz dataset")
    ap.add_argument("--outdir",    type=str, default="inspect_output")
    ap.add_argument("--n-heatmap", type=int, default=128, help="feature heatmap sample count")
    ap.add_argument("--no-pca",    action="store_true",  help="skip PCA for large datasets")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data}")
    d = load_dataset(args.data)
    feats         = d["feats"]
    labels        = d["labels"]
    range_labels  = d["range_labels"]

    if feats is None or labels is None or range_labels is None:
        raise ValueError("Dataset is missing required fields: feats / labels / range_labels")

    print_stats(d)
    print("Generating plots...")

    # 幾何
    plot_geometry(out_dir)
    plot_geometry_3d(out_dir)

    # 特徵
    plot_feature_heatmap(feats, out_dir, n_samples=args.n_heatmap)
    if not args.no_pca:
        plot_pca(feats, labels, out_dir)

    # Label 分布
    plot_label_hist(labels, out_dir)
    plot_range_hist(range_labels, out_dir)
    plot_height_hist(d["height_labels"], out_dir)

    # 連續量分布（v2）
    plot_continuous(d["source_ranges"],  "Source Range Distribution",
                    "Distance to source (m)", "source_range_m.png", out_dir,
                    color="#FF6B6B")
    plot_continuous(d["source_heights"], "Source Height Distribution",
                    "Source height in array frame (m)", "source_height_m.png", out_dir,
                    color="#6BCB77")
    plot_continuous(d["f0_hz"],          "Signal Fundamental Frequency (F0)",
                    "F0 (Hz)", "f0_distribution.png", out_dir,
                    color="#FFD93D")
    plot_continuous(d["self_yaw_deg"],   "Self Yaw Distribution (Domain Randomization)",
                    "Yaw (deg)", "self_yaw_distribution.png", out_dir,
                    color="#C77DFF")

    # 類別分布
    plot_categorical(d["sig_types"],    "Signal Type Distribution",
                     "signal_type_distribution.png", out_dir, color="#4C8EFF")
    plot_categorical(d["render_types"], "Render Type Distribution (pyroom vs free_field)",
                     "render_type_distribution.png", out_dir, color="#FF9F1C")
    plot_categorical(d["obstacle_counts"],
                     "Obstacle Count Distribution",
                     "obstacle_count_distribution.png", out_dir, color="#EF476F")

    # 障礙物通道衰減
    plot_obstacle_gain_heatmap(d["obstacle_gains"], d["obstacle_counts"], out_dir)

    print("\nDone.")
    output_files = sorted(out_dir.glob("*.png"))
    print(f"Generated {len(output_files)} plots:")
    for f in output_files:
        print(f"  {f.name}")
    print(f"\nOutput directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
