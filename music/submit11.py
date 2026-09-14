"""Submission from two specialist rankers at a fixed split.

Only run this if train11.py cleared the +0.005 offline bar -- the harness is not
decision-grade below that, and two sub-0.005 deltas have already reversed sign
on this task's leaderboard.

Trains on four stacked windows (matching submit7.py, the 0.37779 model) and
applies at 2025-08-31 for the real test window.
"""
import argparse

import numpy as np
import polars as pl

from evaluate import TOPK
from rank3 import target
from train4 import load_all
from train7 import FEATS, featurise
from train11 import HIST_FEATS, NEW_FEATS, fit

WINDOWS = [("2025-07-01", "2025-07-16"), ("2025-07-16", "2025-07-31"),
           ("2025-08-01", "2025-08-16"), ("2025-08-16", "2025-08-31")]
APPLY = "2025-08-31"

ap = argparse.ArgumentParser()
ap.add_argument("--split", type=int, default=34, help="history slots of 50")
ap.add_argument("--out", default="submission_specialist.csv")
args = ap.parse_args()

inter, meta, artists, genres, users = load_all()
parts = []
for cut, hi in WINDOWS:
    C = featurise(inter, meta, artists, genres, users, cut)
    parts.append(C.join(target(inter, meta, users, cut, hi), on=["user_id", "item_id"],
                        how="left").with_columns(pl.col("y").fill_null(0.0))
                 .select(FEATS + ["y"]))
    print(f"  window {cut}: {parts[-1].height:,}", flush=True)
tr = pl.concat(parts)

trh, trn = tr.filter(pl.col("is_hist") == 1), tr.filter(pl.col("is_hist") == 0)
mH = fit(trh.select(HIST_FEATS).to_numpy().astype(np.float32), trh["y"].to_numpy())
mN = fit(trn.select(NEW_FEATS).to_numpy().astype(np.float32), trn["y"].to_numpy())
print(f"fitted hist {mH.n_iter_} iters / new {mN.n_iter_} iters", flush=True)

C = featurise(inter, meta, artists, genres, users, APPLY)
Ch = C.filter(pl.col("is_hist") == 1)
Cn = C.filter(pl.col("is_hist") == 0)
Ch = Ch.with_columns(pl.Series("p", mH.predict(Ch.select(HIST_FEATS).to_numpy().astype(np.float32))))
Cn = Cn.with_columns(pl.Series("p", mN.predict(Cn.select(NEW_FEATS).to_numpy().astype(np.float32))))
Ch = Ch.sort(["user_id", "p"], descending=[False, True]).with_columns(
    pl.int_range(pl.len()).over("user_id").alias("r"))
Cn = Cn.sort(["user_id", "p"], descending=[False, True]).with_columns(
    pl.int_range(pl.len()).over("user_id").alias("r"))

h = Ch.filter(pl.col("r") < args.split)
short = (h.group_by("user_id").agg(pl.len().alias("nh"))
         .with_columns((TOPK - pl.col("nh")).alias("need")))
allu = (pl.DataFrame({"user_id": users.to_list()}).join(short, on="user_id", how="left")
        .with_columns(pl.col("need").fill_null(TOPK)))
n = (Cn.join(allu.select("user_id", "need"), on="user_id", how="left")
       .filter(pl.col("r") < pl.col("need")))

top = (pl.concat([h.select("user_id", "item_id", "p", "is_hist"),
                  n.select("user_id", "item_id", "p", "is_hist")])
       .sort(["user_id", "is_hist", "p"], descending=[False, True, True])
       .with_columns(pl.int_range(pl.len()).over("user_id").add(1).alias("rank")))
recs = top.select("user_id", "item_id", "rank").sort(["user_id", "rank"])

# QA -- a cheap pass over every generated submission caught a 14,464-row file before
cnt = recs.group_by("user_id").agg(pl.len().alias("n"))
assert cnt["n"].min() == TOPK == cnt["n"].max(), f"bad counts {cnt['n'].min()}..{cnt['n'].max()}"
assert recs.select("user_id", "item_id").is_duplicated().sum() == 0, "duplicate item per user"
assert set(recs["user_id"].unique()) == set(users), "user set mismatch"
assert recs.height == len(users) * TOPK, f"expected {len(users)*TOPK} rows, got {recs.height}"
assert recs["rank"].min() == 1 and recs["rank"].max() == TOPK
assert recs.null_count().sum_horizontal().item() == 0, "nulls present"

recs.with_row_index("id").select("id", "user_id", "item_id", "rank").write_csv(args.out)
pct = 100 * top.filter(pl.col("is_hist") == 0).height / top.height
print(f"wrote {args.out}: {recs.height:,} rows, new-slot {pct:.1f}%, split {args.split}/{TOPK-args.split}")
