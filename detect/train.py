"""定位網路訓練(監督學習)

Stage 0.5:在接 SAC 之前,先把定位能力練出來。
標籤來自模擬真值,loss = CrossEntropy on azimuth bin。

對齊 Instructions.md §5:
  - 接受 --seed,預設不隨機
  - run 資料夾 runs/<tag>/ 含 config / checkpoints / tb log
  - 永遠保留 best_eval.pt 與 latest.pt,定期存 checkpoint

執行:
    python train.py --steps 5000 --seed 0
    python train.py --steps 5000 --tag myrun --save-every 200
    python train.py --resume runs/myrun/checkpoints/latest.pt --steps 3000  # 接續訓練

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
from env import LocalizationEnv, az_to_bin
from model import build_net, count_params


def make_batch(env: LocalizationEnv, n_bins: int, batch: int):
    feats, labels = [], []
    for _ in range(batch):
        feat, true_az, _ = env.sample()
        feats.append(feat)
        labels.append(az_to_bin(true_az, n_bins))
    return (torch.tensor(np.stack(feats)),
            torch.tensor(labels, dtype=torch.long))


def evaluate(env: LocalizationEnv, net, cfg, n_eval: int = 500) -> dict:
    net.eval()
    hits, errs = 0, []
    n_bins = cfg.task.n_azimuth_bins
    with torch.no_grad():
        for _ in range(n_eval):
            feat, true_az, _ = env.sample()
            pred_az = net.predict_azimuth(
                torch.tensor(feat).unsqueeze(0), n_bins).item()
            err = min(abs(pred_az - true_az) % 360, 360 - abs(pred_az - true_az) % 360)
            errs.append(err)
            if err < cfg.task.hit_threshold_deg:
                hits += 1
    net.train()
    return {"hit_rate": hits / n_eval, "mean_err_deg": float(np.mean(errs))}


def save_ckpt(path: Path, net, opt, step: int, best_hit: float) -> None:
    """存 checkpoint:含模型/optimizer/步數/最佳成績,供 resume。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "best_hit": best_hit,
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000, help="本次訓練步數")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)          # 預設不隨機
    ap.add_argument("--tag", type=str, default="loc")       # run 資料夾名
    ap.add_argument("--save-every", type=int, default=200, help="每 N 步存一次 checkpoint")
    ap.add_argument("--eval-every", type=int, default=0, help="每 N 步評估;0=自動(steps/10)")
    ap.add_argument("--resume", type=str, default="", help="從 checkpoint 路徑接續訓練")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    cfg = DEFAULT
    cfg.task.seed = args.seed
    n_bins = cfg.task.n_azimuth_bins

    train_env = LocalizationEnv(cfg, seed=args.seed)
    eval_env = LocalizationEnv(cfg, seed=args.seed + 999)

    net = build_net(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # --- run 資料夾(對齊 Instructions.md §5)---
    run_dir = Path("runs") / args.tag
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    start_step, best_hit = 0, -1.0
    # --- resume ---
    if args.resume:
        ck = torch.load(args.resume)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        best_hit = ck["best_hit"]
        print(f"從 {args.resume} 接續,起始 step={start_step}, best_hit={best_hit:.2%}")

    print(f"模型參數量 = {count_params(net):,}")
    eval_every = args.eval_every or max(1, args.steps // 10)
    end_step = start_step + args.steps

    for step in range(start_step + 1, end_step + 1):
        feats, labels = make_batch(train_env, n_bins, args.batch)
        loss = loss_fn(net(feats), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        writer.add_scalar("train/loss", loss.item(), step)

        if step % eval_every == 0:
            m = evaluate(eval_env, net, cfg, n_eval=200)
            writer.add_scalar("eval/hit_rate", m["hit_rate"], step)
            writer.add_scalar("eval/mean_err_deg", m["mean_err_deg"], step)
            print(f"step {step:6d} | loss {loss.item():.3f} | "
                  f"hit_rate {m['hit_rate']:.2%} | mean_err {m['mean_err_deg']:.1f}°")
            # best:評估命中率最高的,永遠保留
            if m["hit_rate"] > best_hit:
                best_hit = m["hit_rate"]
                save_ckpt(ckpt_dir / "best_eval.pt", net, opt, step, best_hit)

        # 定期 checkpoint + 永遠更新 latest
        if step % args.save_every == 0:
            save_ckpt(ckpt_dir / f"step_{step}.pt", net, opt, step, best_hit)
            save_ckpt(ckpt_dir / "latest.pt", net, opt, step, best_hit)

    # 收尾:最終評估 + 存 latest
    final = evaluate(eval_env, net, cfg, n_eval=1000)
    save_ckpt(ckpt_dir / "latest.pt", net, opt, end_step, best_hit)
    writer.close()
    print(f"\n最終評估: {final}")
    print(f"checkpoint 存於: {ckpt_dir}")
    print(f"TensorBoard: tensorboard --logdir runs/")


if __name__ == "__main__":
    main()
