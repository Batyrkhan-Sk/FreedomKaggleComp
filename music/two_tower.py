"""Two-tower retrieval baseline for the music task.

This is deliberately self-contained so the same code can become a Kaggle
notebook cell.  It trains implicit user/item embeddings with sampled logistic
loss, retrieves from every track observed before `cut`, and evaluates the
competition's top-50 metric on a temporal holdout.

Example (Kaggle GPU):
    python two_tower.py --cut 2025-08-16 --eval --epochs 8 --device cuda
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

TOPK = 50


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user.weight, std=0.03)
        nn.init.normal_(self.item.weight, std=0.03)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.ib.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item(items)).sum(1) + self.ub(users).squeeze(1) + self.ib(items).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quantised_truth(df: pl.DataFrame, meta: pl.DataFrame, users: pl.Series) -> pl.DataFrame:
    return (
        df.filter(pl.col("user_id").is_in(users.implode()))
        .group_by(["user_id", "item_id"])
        .agg(pl.col("listened_duration").sum().alias("seconds"))
        .join(meta, on="item_id", how="left")
        .filter(pl.col("track_duration") > 0)
        .with_columns(((pl.col("seconds") / pl.col("track_duration")).clip(0, 1) * 4).round().truediv(4).alias("frac"))
        .select("user_id", "item_id", "frac")
    )


def metric(recs: pl.DataFrame, truth: pl.DataFrame, active: int) -> float:
    got = (recs.filter(pl.col("rank") <= TOPK)
            .join(truth, on=["user_id", "item_id"], how="left")
            .with_columns(pl.col("frac").fill_null(0.0)))
    return got["frac"].sum() / active / TOPK


def resolve_input_dir(requested: str) -> Path:
    """Use the requested folder, or find the attached Kaggle dataset by file."""
    path = Path(requested)
    if (path / "interactions.csv").is_file():
        return path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = [p.parent for p in kaggle_input.rglob("interactions.csv")
                   if (p.parent / "item_metadata.csv").is_file() and (p.parent / "test.csv").is_file()]
        if len(matches) == 1:
            print(f"input-dir auto-detected: {matches[0]}", flush=True)
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"multiple attached datasets contain interactions.csv: {matches}")
    raise FileNotFoundError(
        f"Could not find interactions.csv in {path}. Attach the competition dataset; "
        "available Kaggle inputs are under /kaggle/input.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="2025-08-16")
    ap.add_argument("--input-dir", default=".", help="directory containing the competition CSVs")
    ap.add_argument("--output", default="two_tower_submission.csv")
    ap.add_argument("--retrieval-k", type=int, default=500,
                    help="candidates per user to save; rank 1..50 remains the scored list")
    ap.add_argument("--eval", action="store_true", help="score the following 15-day window")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--max-events", type=int, default=8_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    inp = resolve_input_dir(args.input_dir)
    inter = pl.read_csv(inp / "interactions.csv", columns=["user_id", "item_id", "listened_duration", "listened_datetime"]).with_columns(
        pl.col("listened_datetime").str.slice(0, 10).alias("d")
    )
    meta = pl.read_csv(inp / "item_metadata.csv", columns=["item_id", "track_duration"])
    users = pl.read_csv(inp / "test.csv")["user_id"]
    hist = inter.filter(pl.col("d") < args.cut).filter(pl.col("user_id").is_in(users.implode()))
    future = inter.filter(pl.col("d") >= args.cut) if args.eval else None

    # Keep completed / meaningful listens; repeated events become importance
    # samples naturally, which matches the implicit-feedback objective.
    events = hist.filter(pl.col("listened_duration") >= 30).select("user_id", "item_id")
    if events.height > args.max_events:
        events = events.sample(n=args.max_events, with_replacement=False, seed=args.seed)
    user_ids = np.sort(users.to_numpy())
    item_ids = np.sort(hist["item_id"].unique().to_numpy())
    umap = {int(v): i for i, v in enumerate(user_ids)}
    imap = {int(v): i for i, v in enumerate(item_ids)}
    eu = np.fromiter((umap[int(v)] for v in events["user_id"].to_numpy()), dtype=np.int64, count=events.height)
    ei = np.fromiter((imap[int(v)] for v in events["item_id"].to_numpy()), dtype=np.int64, count=events.height)
    # Negative distribution follows item popularity^0.75, standard for
    # implicit retrieval; it avoids making the task trivially easy with only
    # obscure random negatives.
    pop = np.bincount(ei, minlength=len(item_ids)).astype(np.float64)
    neg_p = pop ** 0.75
    neg_p /= neg_p.sum()
    print(f"device={device} events={len(eu):,} users={len(user_ids):,} items={len(item_ids):,}", flush=True)

    model = TwoTower(len(user_ids), len(item_ids), args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    order = np.arange(len(eu))
    for epoch in range(args.epochs):
        np.random.default_rng(args.seed + epoch).shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            ix = order[start:start + args.batch_size]
            u = torch.as_tensor(eu[ix], device=device)
            pos = torch.as_tensor(ei[ix], device=device)
            neg = torch.as_tensor(np.random.default_rng(args.seed + epoch + start).choice(len(item_ids), size=len(ix), p=neg_p), device=device)
            loss = -F.logsigmoid(model.score(u, pos) - model.score(u, neg)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(ix)
        print(f"epoch {epoch + 1}/{args.epochs}: bpr_loss={total / len(order):.5f}", flush=True)

    # Full-catalogue retrieval in small user batches. Exclude items already
    # heard by that user; the final blend will add the GBM's history candidates.
    seen = hist.select("user_id", "item_id").group_by("user_id").agg(pl.col("item_id").alias("seen"))
    seen_map = {int(u): set(x) for u, x in seen.iter_rows()}
    model.eval()
    rows = []
    item_t = torch.arange(len(item_ids), device=device)
    with torch.no_grad():
        for start in range(0, len(user_ids), 128):
            uidx = torch.arange(start, min(start + 128, len(user_ids)), device=device)
            scores = model.user(uidx) @ model.item.weight.T + model.ib(item_t).squeeze(1)[None, :] + model.ub(uidx)
            for local, ui in enumerate(range(start, min(start + 128, len(user_ids)))):
                if seen_map.get(int(user_ids[ui])):
                    mask = np.fromiter((imap[x] for x in seen_map[int(user_ids[ui])] if x in imap), dtype=np.int64)
                    scores[local, torch.as_tensor(mask, device=device)] = -torch.inf
            _, top = torch.topk(scores, k=min(args.retrieval_k, len(item_ids)), dim=1)
            for local, ui in enumerate(range(start, min(start + 128, len(user_ids)))):
                rows.extend((int(user_ids[ui]), int(item_ids[j]), rank + 1) for rank, j in enumerate(top[local].cpu().numpy()))
    recs = pl.DataFrame(rows, schema=["user_id", "item_id", "rank"], orient="row")
    if args.eval:
        truth = quantised_truth(future, meta, users)
        active = truth["user_id"].n_unique()
        top50 = metric(recs, truth, active)
        pool = recs.join(truth, on=["user_id", "item_id"], how="left").with_columns(pl.col("frac").fill_null(0.0))
        pool_score = pool["frac"].sum() / active / TOPK
        pool.write_csv(args.output)
        print(f"two-tower new-only top50={top50:.5f} | top{args.retrieval_k} candidate value={pool_score:.5f} | active={active}", flush=True)
        print(f"wrote holdout candidates: {args.output}", flush=True)
    else:
        recs.filter(pl.col("rank") <= TOPK).with_row_index("id").write_csv(args.output)
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
