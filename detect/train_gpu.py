"""GPU 訓練(方案 B):從預生成資料讀,餵飽 GPU

先跑 gen_dataset.py 產生 data/train.npz 與 data/eval.npz,再跑本檔。
自動偵測 CUDA;無 GPU 則 fallback CPU。
checkpoint / tensorboard / resume 與 train.py 一致。

用法:
    python gen_dataset.py --n 200000 --seed 0   --out data/train.npz
    python gen_dataset.py --n 20000  --seed 999 --out data/eval.npz
    python train_gpu.py --steps 5000 --tag gpu_run
    python train_gpu.py --resume runs/gpu_run/checkpoints/latest.pt --steps 3000

監控:
    tensorboard --logdir runs/
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import DEFAULT
from model import build_net, count_params


def load_npz(path: str, device):
    d = np.load(path, allow_pickle=True)
    feats = torch.tensor(d["feats"], device=device)
    labels = torch.tensor(d["labels"], dtype=torch.long, device=device)
    # 距離 label:新資料才有;舊資料(無 range_labels)回 None,自動退回單 head 訓練
    if "range_labels" in d:
        range_labels = torch.tensor(d["range_labels"], dtype=torch.long, device=device)
    else:
        range_labels = None
    if "height_labels" in d:
        height_labels = torch.tensor(d["height_labels"], dtype=torch.long, device=device)
    else:
        height_labels = None
    # sig_types:分 signal_type 診斷用(字串陣列,留在 CPU);無則回 None
    meta = {
        "sig_types": d["sig_types"] if "sig_types" in d else None,
        "render_types": d["render_types"] if "render_types" in d else None,
        "f0_hz": d["f0_hz"] if "f0_hz" in d else None,
        "source_ranges": d["source_ranges"] if "source_ranges" in d else None,
        "source_heights": d["source_heights"] if "source_heights" in d else None,
        "self_yaw_deg": d["self_yaw_deg"] if "self_yaw_deg" in d else None,
        "obstacle_counts": d["obstacle_counts"] if "obstacle_counts" in d else None,
    }
    return feats, labels, range_labels, height_labels, meta


def _sha256_short(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _split_outputs(out, has_range: bool, has_height: bool):
    if not isinstance(out, tuple):
        return out, None, None
    az_logits = out[0]
    cursor = 1
    range_logits = out[cursor] if has_range else None
    if has_range:
        cursor += 1
    height_logits = out[cursor] if has_height else None
    return az_logits, range_logits, height_logits


def evaluate(feats, labels, range_labels, height_labels, net, cfg, meta=None,
             breakdown: bool = False, n_sectors: int = 8) -> dict:
    """評估定位網路。

    Args:
        breakdown: True 時額外計算分象限 / 分 signal_type 命中率(放進 m["sectors"]、
                   m["by_sig"]),供診斷用。訓練迴圈每步評估設 False(省開銷),
                   最終評估設 True。
        n_sectors: 方位分扇區數(預設 8 = 每 45°)。

    回傳 dict:整體 hit_rate / mean_err_deg(+ range_hit_rate),
    breakdown=True 時另含 sectors / by_sig 兩張表(list[dict])。
    """
    net.eval()
    n_bins = cfg.task.n_azimuth_bins
    deg_per_bin = 360.0 / n_bins
    hit_thresh = cfg.task.hit_threshold_deg
    with torch.no_grad():
        out = net(feats)
        has_range = range_labels is not None and getattr(net, "has_range", False)
        has_height = height_labels is not None and getattr(net, "has_height", False)
        az_logits, range_logits, height_logits = _split_outputs(out, has_range, has_height)
        pred_bin = az_logits.argmax(dim=-1)
        pred_deg = pred_bin.float() * deg_per_bin
        true_deg = labels.float() * deg_per_bin
        err = torch.abs(pred_deg - true_deg) % 360
        err = torch.minimum(err, 360 - err)
        hit = (err < hit_thresh)
        hit_rate = hit.float().mean().item()
        mean_err = err.mean().item()
        m = {"hit_rate": hit_rate, "mean_err_deg": mean_err}

        # 距離 head go/no-go 指標:4-bin 分類命中率(隨機猜 = 1/n_range_bins)
        rg_hit_mask = None
        if range_logits is not None:
            rg_pred = range_logits.argmax(dim=-1)
            rg_hit_mask = (rg_pred == range_labels)
            m["range_hit_rate"] = rg_hit_mask.float().mean().item()
            m["range_chance"] = 1.0 / cfg.range_head.n_range_bins
        ht_hit_mask = None
        if height_logits is not None:
            ht_pred = height_logits.argmax(dim=-1)
            ht_hit_mask = (ht_pred == height_labels)
            m["height_hit_rate"] = ht_hit_mask.float().mean().item()
            m["height_chance"] = 1.0 / cfg.height_head.n_height_bins

        if breakdown:
            # ---- 分方位扇區:把 true_deg 切成 n_sectors 個等寬扇區 ----
            # 目的:揪出系統性盲區(尤其前後 / 鏡像方位,RK-4)。
            sector_w = 360.0 / n_sectors
            sector_idx = (true_deg / sector_w).long().clamp(max=n_sectors - 1)
            sectors = []
            for s in range(n_sectors):
                mask = sector_idx == s
                cnt = int(mask.sum().item())
                row = {
                    "sector": f"{int(s*sector_w):3d}-{int((s+1)*sector_w):3d}°",
                    "n": cnt,
                    "hit_rate": hit[mask].float().mean().item() if cnt else float("nan"),
                    "mean_err": err[mask].mean().item() if cnt else float("nan"),
                }
                if rg_hit_mask is not None:
                    row["range_hit"] = rg_hit_mask[mask].float().mean().item() if cnt else float("nan")
                if ht_hit_mask is not None:
                    row["height_hit"] = ht_hit_mask[mask].float().mean().item() if cnt else float("nan")
                sectors.append(row)
            m["sectors"] = sectors

            # ---- 分 signal_type:cw / chirp / pulse_train 各自表現 ----
            # 目的:chirp/pulse 單 bin 相位意義弱(extract_features 自陳),
            #       若失敗集中在某型 → 是訊號設計問題而非幾何問題。
            sig_types = meta.get("sig_types") if meta else None
            if sig_types is not None:
                by_sig = []
                hit_cpu = hit.cpu().numpy()
                err_cpu = err.cpu().numpy()
                rg_cpu = rg_hit_mask.cpu().numpy() if rg_hit_mask is not None else None
                ht_cpu = ht_hit_mask.cpu().numpy() if ht_hit_mask is not None else None
                for st in np.unique(sig_types):
                    smask = (sig_types == st)
                    cnt = int(smask.sum())
                    row = {
                        "sig_type": str(st),
                        "n": cnt,
                        "hit_rate": float(hit_cpu[smask].mean()) if cnt else float("nan"),
                        "mean_err": float(err_cpu[smask].mean()) if cnt else float("nan"),
                    }
                    if rg_cpu is not None:
                        row["range_hit"] = float(rg_cpu[smask].mean()) if cnt else float("nan")
                    if ht_cpu is not None:
                        row["height_hit"] = float(ht_cpu[smask].mean()) if cnt else float("nan")
                    by_sig.append(row)
                m["by_sig"] = by_sig
            render_types = meta.get("render_types") if meta else None
            if render_types is not None:
                by_render = []
                hit_cpu = hit.cpu().numpy()
                err_cpu = err.cpu().numpy()
                for rt in np.unique(render_types):
                    rmask = (render_types == rt)
                    cnt = int(rmask.sum())
                    by_render.append({
                        "render_type": str(rt),
                        "n": cnt,
                        "hit_rate": float(hit_cpu[rmask].mean()) if cnt else float("nan"),
                        "mean_err": float(err_cpu[rmask].mean()) if cnt else float("nan"),
                    })
                m["by_render"] = by_render
            obstacle_counts = meta.get("obstacle_counts") if meta else None
            if obstacle_counts is not None:
                hit_cpu = hit.cpu().numpy()
                err_cpu = err.cpu().numpy()
                has_obs = obstacle_counts > 0
                m["obstacle"] = [
                    {
                        "group": "clear",
                        "n": int((~has_obs).sum()),
                        "hit_rate": float(hit_cpu[~has_obs].mean()) if (~has_obs).any() else float("nan"),
                        "mean_err": float(err_cpu[~has_obs].mean()) if (~has_obs).any() else float("nan"),
                    },
                    {
                        "group": "obstacle",
                        "n": int(has_obs.sum()),
                        "hit_rate": float(hit_cpu[has_obs].mean()) if has_obs.any() else float("nan"),
                        "mean_err": float(err_cpu[has_obs].mean()) if has_obs.any() else float("nan"),
                    },
                ]
    net.train()
    return m


def print_breakdown(m: dict) -> None:
    """把 evaluate(breakdown=True) 的分組結果印成對齊的表。"""
    has_rg = "range_hit_rate" in m
    has_ht = "height_hit_rate" in m
    if "sectors" in m:
        print("\n  ── 分方位扇區(揪鏡像盲區)──")
        hdr = f"  {'扇區':>10} {'樣本':>6} {'hit_rate':>9} {'mean_err':>9}"
        if has_rg:
            hdr += f" {'range_hit':>10}"
        if has_ht:
            hdr += f" {'height_hit':>10}"
        print(hdr)
        for r in m["sectors"]:
            line = (f"  {r['sector']:>10} {r['n']:>6} "
                    f"{r['hit_rate']:>8.1%} {r['mean_err']:>8.1f}°")
            if has_rg and "range_hit" in r:
                line += f" {r['range_hit']:>9.1%}"
            if has_ht and "height_hit" in r:
                line += f" {r['height_hit']:>9.1%}"
            print(line)
    if "by_sig" in m:
        print("\n  ── 分 signal_type ──")
        hdr = f"  {'類型':>12} {'樣本':>6} {'hit_rate':>9} {'mean_err':>9}"
        if has_rg:
            hdr += f" {'range_hit':>10}"
        if has_ht:
            hdr += f" {'height_hit':>10}"
        print(hdr)
        for r in m["by_sig"]:
            line = (f"  {r['sig_type']:>12} {r['n']:>6} "
                    f"{r['hit_rate']:>8.1%} {r['mean_err']:>8.1f}°")
            if has_rg and "range_hit" in r:
                line += f" {r['range_hit']:>9.1%}"
            if has_ht and "height_hit" in r:
                line += f" {r['height_hit']:>9.1%}"
            print(line)
    if "by_render" in m:
        print("\n  ── 分 render_type ──")
        for r in m["by_render"]:
            print(f"  {r['render_type']:>18} {r['n']:>6} "
                  f"{r['hit_rate']:>8.1%} {r['mean_err']:>8.1f}°")
    if "obstacle" in m:
        print("\n  ── 障礙物分組 ──")
        for r in m["obstacle"]:
            print(f"  {r['group']:>10} {r['n']:>6} "
                  f"{r['hit_rate']:>8.1%} {r['mean_err']:>8.1f}°")


def save_ckpt(path: Path, net, opt, step: int, best_hit: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "best_hit": best_hit}, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)       # GPU 可用大 batch
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="gpu")
    ap.add_argument("--train-data", type=str, default="data/train.npz")
    ap.add_argument("--eval-data", type=str, default="data/eval.npz")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置 = {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type=='cuda' else ''}")

    cfg = DEFAULT
    tr_feats, tr_labels, tr_range, tr_height, tr_meta = load_npz(args.train_data, device)
    ev_feats, ev_labels, ev_range, ev_height, ev_meta = load_npz(args.eval_data, device)
    print(f"訓練資料 {tuple(tr_feats.shape)} | 評估資料 {tuple(ev_feats.shape)}")

    # 資料含距離 label → 建雙 head;否則退回單 head(向後相容舊資料)
    with_range = tr_range is not None
    with_height = tr_height is not None
    if with_range:
        print(f"偵測到距離 label → 雙 head 訓練(方位 + {cfg.range_head.n_range_bins}-bin 距離)")
    else:
        print("無距離 label → 單 head 訓練(僅方位,舊行為)")
    if with_height:
        print(f"偵測到高度 label → 加高度 head({cfg.height_head.n_height_bins}-bin)")

    net = build_net(cfg, with_range=with_range, with_height=with_height).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    range_ce_weight = cfg.range_head.range_ce_weight
    height_ce_weight = cfg.height_head.height_ce_weight

    run_dir = Path("runs") / args.tag
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        **vars(args),
        "train_sha256": _sha256_short(args.train_data),
        "eval_sha256": _sha256_short(args.eval_data),
        "with_range": with_range,
        "with_height": with_height,
        "n_params": count_params(net),
        "model_memory": "stateless_mlp_no_temporal_memory",
    }
    (run_dir / "config.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False))
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    writer.add_text("run/config", json.dumps(run_info, indent=2, ensure_ascii=False), 0)
    writer.add_text(
        "model/memory",
        "Current LocalizationNet is a stateless MLP: every sample is predicted independently. "
        "Temporal memory would require sequence datasets plus GRU/LSTM/Transformer state.",
        0,
    )
    writer.add_histogram("data/train_labels_azimuth", tr_labels.float().cpu(), 0)
    if tr_range is not None:
        writer.add_histogram("data/train_labels_range", tr_range.float().cpu(), 0)
    if tr_height is not None:
        writer.add_histogram("data/train_labels_height", tr_height.float().cpu(), 0)
    for key in ("f0_hz", "source_ranges", "source_heights", "self_yaw_deg", "obstacle_counts"):
        if tr_meta.get(key) is not None:
            writer.add_histogram(f"data/{key}", torch.tensor(tr_meta[key]).float(), 0)

    start_step, best_hit = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step, best_hit = ck["step"], ck["best_hit"]
        print(f"從 {args.resume} 接續,step={start_step}, best_hit={best_hit:.2%}")

    print(f"模型參數量 = {count_params(net):,}")
    eval_every = args.eval_every or max(1, args.steps // 10)
    end_step = start_step + args.steps
    n_train = tr_feats.shape[0]
    g = torch.Generator(device=device).manual_seed(args.seed)

    for step in range(start_step + 1, end_step + 1):
        idx = torch.randint(0, n_train, (args.batch,), generator=g, device=device)
        out = net(tr_feats[idx])
        az_logits, rg_logits, ht_logits = _split_outputs(out, with_range, with_height)
        if with_range:
            loss_az = loss_fn(az_logits, tr_labels[idx])
            loss_rg = loss_fn(rg_logits, tr_range[idx])
            loss = loss_az + range_ce_weight * loss_rg
        else:
            loss = loss_fn(az_logits, tr_labels[idx])
            loss_az = loss
            loss_rg = None
        if with_height:
            loss_ht = loss_fn(ht_logits, tr_height[idx])
            loss = loss + height_ce_weight * loss_ht
        else:
            loss_ht = None
        opt.zero_grad()
        loss.backward()
        opt.step()
        writer.add_scalar("train/loss", loss.item(), step)
        if with_range:
            writer.add_scalar("train/loss_azimuth", loss_az.item(), step)
            writer.add_scalar("train/loss_range", loss_rg.item(), step)
        if with_height:
            writer.add_scalar("train/loss_height", loss_ht.item(), step)

        if step % eval_every == 0:
            m = evaluate(ev_feats, ev_labels, ev_range, ev_height, net, cfg, meta=ev_meta)
            writer.add_scalar("eval/hit_rate", m["hit_rate"], step)
            writer.add_scalar("eval/mean_err_deg", m["mean_err_deg"], step)
            msg = (f"step {step:6d} | loss {loss.item():.3f} | "
                   f"hit_rate {m['hit_rate']:.2%} | mean_err {m['mean_err_deg']:.1f}°")
            if "range_hit_rate" in m:
                writer.add_scalar("eval/range_hit_rate", m["range_hit_rate"], step)
                # go/no-go:距離命中率明顯 > 隨機猜(1/n_bins)才有資訊價值
                msg += (f" | range_hit {m['range_hit_rate']:.2%} "
                        f"(隨機 {m['range_chance']:.0%})")
            if "height_hit_rate" in m:
                writer.add_scalar("eval/height_hit_rate", m["height_hit_rate"], step)
                msg += (f" | height_hit {m['height_hit_rate']:.2%} "
                        f"(隨機 {m['height_chance']:.0%})")
            print(msg)
            if m["hit_rate"] > best_hit:
                best_hit = m["hit_rate"]
                save_ckpt(ckpt_dir / "best_eval.pt", net, opt, step, best_hit)

        if step % args.save_every == 0:
            save_ckpt(ckpt_dir / f"step_{step}.pt", net, opt, step, best_hit)
            save_ckpt(ckpt_dir / "latest.pt", net, opt, step, best_hit)

    final = evaluate(ev_feats, ev_labels, ev_range, ev_height, net, cfg,
                     meta=ev_meta, breakdown=True)
    for group_name in ("by_sig", "by_render", "obstacle"):
        if group_name in final:
            for row in final[group_name]:
                name = row.get("sig_type") or row.get("render_type") or row.get("group")
                writer.add_scalar(f"final/{group_name}/{name}_hit_rate", row["hit_rate"], end_step)
    save_ckpt(ckpt_dir / "latest.pt", net, opt, end_step, best_hit)
    writer.close()
    # 整體數字
    summary = {
        k: v for k, v in final.items()
        if k not in ("sectors", "by_sig", "by_render", "obstacle")
    }
    print(f"\n最終評估(整體): {summary}")
    # 分組診斷表(分象限揪鏡像盲區 + 分 signal_type)
    print_breakdown(final)
    print(f"\ncheckpoint: {ckpt_dir}")


if __name__ == "__main__":
    main()
