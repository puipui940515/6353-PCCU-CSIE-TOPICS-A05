from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DETECT = ROOT / "detect"
sys.path.insert(0, str(DETECT))

from config import DEFAULT  # noqa: E402
from model import build_net  # noqa: E402


CFG = DEFAULT


def bin_centers(edges: tuple[float, ...], low: float, high: float) -> np.ndarray:
    bounds = np.asarray((low, *edges, high), dtype=np.float32)
    return (bounds[:-1] + bounds[1:]) * 0.5


def range_centers() -> np.ndarray:
    return bin_centers(
        CFG.range_head.bin_edges_m,
        CFG.task.workspace_r_range[0],
        CFG.task.workspace_r_range[1],
    )


def height_centers() -> np.ndarray:
    return bin_centers(
        CFG.height_head.bin_edges_m,
        CFG.task.workspace_z_range[0],
        CFG.task.workspace_z_range[1],
    )


def load_ckpt(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    with_range = any(k.startswith("range_head.") for k in state)
    with_height = any(k.startswith("height_head.") for k in state)
    model = build_net(CFG, with_range=with_range, with_height=with_height)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, with_range, with_height, ckpt if isinstance(ckpt, dict) else {}


def evaluate(model, eval_npz: Path, with_range: bool, with_height: bool) -> dict:
    d = np.load(eval_npz)
    x = d["feats"].astype("float32")
    y_az = d["labels"].astype("int64")
    y_range = d["range_labels"].astype("int64") if "range_labels" in d.files else None
    y_height = d["height_labels"].astype("int64") if "height_labels" in d.files else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    preds_az, preds_range, preds_height = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            batch = torch.from_numpy(x[i : i + 4096]).to(device)
            out = model(batch)
            if isinstance(out, tuple):
                az_logits = out[0]
                r_logits = out[1] if len(out) > 1 else None
                h_logits = out[2] if len(out) > 2 else None
            else:
                az_logits, r_logits, h_logits = out, None, None
            preds_az.append(az_logits.argmax(dim=1).cpu().numpy())
            if r_logits is not None:
                preds_range.append(r_logits.argmax(dim=1).cpu().numpy())
            if h_logits is not None:
                preds_height.append(h_logits.argmax(dim=1).cpu().numpy())
    elapsed = time.perf_counter() - t0

    p_az = np.concatenate(preds_az)
    deg_per_bin = 360.0 / CFG.task.n_azimuth_bins
    pred_deg = p_az.astype("float32") * deg_per_bin
    true_deg = y_az.astype("float32") * deg_per_bin
    az_err = np.abs(pred_deg - true_deg) % 360.0
    az_err = np.minimum(az_err, 360.0 - az_err)
    out = {
        "eval_samples": int(len(x)),
        "device": device,
        "latency_ms_per_sample": float(elapsed * 1000.0 / max(1, len(x))),
        "azimuth_acc": float((p_az == y_az).mean()),
        "azimuth_hit_10deg": float((az_err < CFG.task.hit_threshold_deg).mean()),
        "azimuth_mean_err_deg": float(az_err.mean()),
    }
    if y_range is not None and preds_range:
        p_range = np.concatenate(preds_range)
        out["range_acc"] = float((p_range == y_range).mean())
        centers = range_centers()
        out["range_mae_m"] = float(np.abs(centers[p_range] - centers[y_range]).mean())
        out["azimuth_range_joint_acc"] = float(((p_az == y_az) & (p_range == y_range)).mean())
    if y_height is not None and preds_height:
        p_height = np.concatenate(preds_height)
        out["height_acc"] = float((p_height == y_height).mean())
        centers = height_centers()
        out["height_mae_m"] = float(np.abs(centers[p_height] - centers[y_height]).mean())
        joint = p_az == y_az
        if "range_acc" in out:
            joint = joint & (p_range == y_range)
        out["all_heads_joint_acc"] = float((joint & (p_height == y_height)).mean())
    if "obstacle_counts" in d.files:
        obs = d["obstacle_counts"] > 0
        out["azimuth_hit_clear"] = float((az_err[~obs] < CFG.task.hit_threshold_deg).mean())
        out["azimuth_hit_obstacle"] = float((az_err[obs] < CFG.task.hit_threshold_deg).mean())
    return out


def dataset_stats(path: Path) -> dict:
    d = np.load(path)
    stats = {"samples": int(len(d["feats"])), "feature_dim": int(d["feats"].shape[1])}
    for name in ["f0_hz", "source_ranges", "source_heights", "self_yaw_deg", "obstacle_counts"]:
        if name in d.files:
            arr = d[name].astype("float64")
            stats[name] = {
                "min": float(arr.min()),
                "mean": float(arr.mean()),
                "max": float(arr.max()),
            }
    if "obstacle_gains" in d.files:
        g = d["obstacle_gains"].astype("float64")
        stats["obstacle_gains"] = {"min": float(g.min()), "mean": float(g.mean()), "max": float(g.max())}
    return stats


def tensorboard_last_scalars(run_dir: Path) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        return {"error": f"tensorboard unavailable: {exc}"}
    result = {}
    for event in run_dir.glob("**/events.out.tfevents.*"):
        acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except Exception as exc:
            result[str(event.relative_to(ROOT))] = {"error": str(exc)}
            continue
        tags = acc.Tags().get("scalars", [])
        payload = {}
        for tag in tags:
            vals = acc.Scalars(tag)
            if vals:
                last = vals[-1]
                payload[tag] = {"step": int(last.step), "value": float(last.value)}
        result[str(event.relative_to(ROOT))] = payload
    return result


def main() -> None:
    run = DETECT / "runs" / "gpu" / "checkpoints" / "best_eval.pt"
    if not run.exists():
        run = DETECT / "runs" / "gpu_run" / "checkpoints" / "best_eval.pt"
    model, with_range, with_height, ckpt = load_ckpt(run)
    payload = {
        "checkpoint": str(run.relative_to(ROOT)),
        "with_range": with_range,
        "with_height": with_height,
        "checkpoint_keys": sorted([k for k in ckpt.keys() if k != "model"]),
        "eval": evaluate(model, DETECT / "data" / "eval.npz", with_range, with_height),
        "train_dataset": dataset_stats(DETECT / "data" / "train.npz"),
        "eval_dataset": dataset_stats(DETECT / "data" / "eval.npz"),
        "perception_tb": tensorboard_last_scalars((run.parents[1] / "tb")),
        "sac_tb": tensorboard_last_scalars(ROOT / "runs"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
