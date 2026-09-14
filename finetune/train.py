"""Contrastive fine-tuning: teach the backbone that a degraded copy is the same painting.

Objective is InfoNCE over in-batch negatives. For a batch of B images we build B
clean anchors and B degraded positives; positive i must be closer to anchor i
than to any of the other B-1 anchors. Those negatives are the crux -- the
gallery contains 9,000 artwork distractors, so the model has to separate *this*
painting from other paintings, not merely paintings from non-paintings.

The loss is symmetric (query->gallery and gallery->query). Only the
query->gallery direction is scored by the competition, but including both is a
cheap regulariser and standard practice.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import PairDataset
from model import RetrievalNet

DATA_DIR = Path.home() / "Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset"


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(len(anchor), device=anchor.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitl14")
    ap.add_argument("--size", type=int, default=392)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5, help="low: we are nudging, not retraining")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="the head is randomly initialised")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--trainable-blocks", type=int, default=4)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N images")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default="finetuned.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert args.size % 14 == 0, "DINOv2 uses 14px patches; size must be a multiple of 14"
    set_seed(args.seed)

    paths = sorted(Path(args.data_dir).glob("*.png"))
    if args.limit:
        paths = paths[: args.limit]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"images {len(paths)}  device {device}  backbone {args.backbone} @{args.size}")

    model = RetrievalNet(args.backbone, args.trainable_blocks).to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train/1e6:.1f}M / {n_all/1e6:.1f}M parameters")

    head_ids = {id(p) for p in model.head.parameters()}
    optim = torch.optim.AdamW(
        [
            {"params": [p for p in model.trainable_parameters() if id(p) not in head_ids],
             "lr": args.lr},
            {"params": list(model.head.parameters()), "lr": args.head_lr},
        ],
        weight_decay=0.05,
    )

    loader = DataLoader(
        PairDataset(paths, args.size, args.strength),
        batch_size=args.batch, shuffle=True, num_workers=args.workers,
        drop_last=True,  # InfoNCE needs a consistent negative count
    )
    steps_total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=[args.lr, args.head_lr], total_steps=steps_total, pct_start=0.1
    )

    scaler = torch.amp.GradScaler(device) if (args.fp16 and device == "cuda") else None
    t0 = time.time()
    step = 0

    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for anchor, positive, _ in loader:
            anchor, positive = anchor.to(device), positive.to(device)
            optim.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = info_nce(model(anchor), model(positive), args.temperature)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss = info_nce(model(anchor), model(positive), args.temperature)
                loss.backward()
                optim.step()
            sched.step()

            running += loss.item() * len(anchor)
            seen += len(anchor)
            step += 1
            if step % 50 == 0:
                rate = step / (time.time() - t0)
                eta = (steps_total - step) / rate / 60
                print(f"  epoch {epoch} step {step}/{steps_total}  "
                      f"loss {running/seen:.4f}  {rate:.2f} it/s  ETA {eta:.0f} min", flush=True)

        torch.save(
            {"model": model.state_dict(), "args": vars(args), "epoch": epoch},
            args.out,
        )
        print(f"epoch {epoch} done, mean loss {running/seen:.4f} -> saved {args.out}", flush=True)

    print(f"finished in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
