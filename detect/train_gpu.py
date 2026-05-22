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
    return feats, labels


def evaluate(feats, labels, net, cfg) -> dict:
    net.eval()
    n_bins = cfg.task.n_azimuth_bins
    deg_per_bin = 360.0 / n_bins
    with torch.no_grad():
        pred_bin = net(feats).argmax(dim=-1)
        pred_deg = pred_bin.float() * deg_per_bin
        true_deg = labels.float() * deg_per_bin
        err = torch.abs(pred_deg - true_deg) % 360
        err = torch.minimum(err, 360 - err)
        hit_rate = (err < cfg.task.hit_threshold_deg).float().mean().item()
        mean_err = err.mean().item()
    net.train()
    return {"hit_rate": hit_rate, "mean_err_deg": mean_err}


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
    tr_feats, tr_labels = load_npz(args.train_data, device)
    ev_feats, ev_labels = load_npz(args.eval_data, device)
    print(f"訓練資料 {tuple(tr_feats.shape)} | 評估資料 {tuple(ev_feats.shape)}")

    net = build_net(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    run_dir = Path("runs") / args.tag
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

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
        loss = loss_fn(net(tr_feats[idx]), tr_labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        writer.add_scalar("train/loss", loss.item(), step)

        if step % eval_every == 0:
            m = evaluate(ev_feats, ev_labels, net, cfg)
            writer.add_scalar("eval/hit_rate", m["hit_rate"], step)
            writer.add_scalar("eval/mean_err_deg", m["mean_err_deg"], step)
            print(f"step {step:6d} | loss {loss.item():.3f} | "
                  f"hit_rate {m['hit_rate']:.2%} | mean_err {m['mean_err_deg']:.1f}°")
            if m["hit_rate"] > best_hit:
                best_hit = m["hit_rate"]
                save_ckpt(ckpt_dir / "best_eval.pt", net, opt, step, best_hit)

        if step % args.save_every == 0:
            save_ckpt(ckpt_dir / f"step_{step}.pt", net, opt, step, best_hit)
            save_ckpt(ckpt_dir / "latest.pt", net, opt, step, best_hit)

    final = evaluate(ev_feats, ev_labels, net, cfg)
    save_ckpt(ckpt_dir / "latest.pt", net, opt, end_step, best_hit)
    writer.close()
    print(f"\n最終評估: {final}")
    print(f"checkpoint: {ckpt_dir}")


if __name__ == "__main__":
    main()
