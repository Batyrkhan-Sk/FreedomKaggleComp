"""Submission at the tuned capacity: 200 iters x lr 0.03 (was 600 x 0.06).

Measured +0.00508 offline against the shipping config, on a broad plateau
(40..150 iters all score +0.0045..+0.0051), so the choice is not sensitive to
where exactly on the plateau it lands. Control confirms the model is real: the
tuned GBM beats the best no-model ranker by +0.021, four times the tuning gain.

Diagnosis behind it: the target is ~98% zeros, so continued MSE fitting buys
accuracy on the near-zero mass and pays for it in the top-50 ordering that
actually scores. 600 iterations was inherited and never tuned against the
ranking metric. Confirmed on two independent axes -- iteration count and
learning rate reach the same plateau from opposite ends.

Trains on four stacked windows and applies at 2025-08-31, matching submit7.py
(the 0.37779 artifact) in everything except the two hyperparameters.
"""
import argparse

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK
from rank3 import target
from train4 import load_all
from train7 import FEATS, featurise

WINDOWS = [("2025-07-01", "2025-07-16"), ("2025-07-16", "2025-07-31"),
           ("2025-08-01", "2025-08-16"), ("2025-08-16", "2025-08-31")]
APPLY = "2025-08-31"

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="submission_tuned.csv")
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

m = HistGradientBoostingRegressor(
    max_iter=args.iters, learning_rate=args.lr, max_leaf_nodes=63,
    min_samples_leaf=200, l2_regularization=1.0, random_state=args.seed,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=40)
m.fit(tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy())
print(f"fitted {m.n_iter_}/{args.iters} iters at lr {args.lr}", flush=True)

C = featurise(inter, meta, artists, genres, users, APPLY)
C = C.with_columns(pl.Series("p", m.predict(C.select(FEATS).to_numpy().astype(np.float32))))
top = (C.sort(["user_id", "p"], descending=[False, True])
         .group_by("user_id", maintain_order=True).head(TOPK)
         .with_columns(pl.int_range(pl.len()).over("user_id").add(1).alias("rank"))
         .select("user_id", "item_id", "rank", "is_hist"))
recs = top.select("user_id", "item_id", "rank").sort(["user_id", "rank"])

cnt = recs.group_by("user_id").agg(pl.len().alias("n"))
assert cnt["n"].min() == TOPK == cnt["n"].max(), f"bad counts {cnt['n'].min()}..{cnt['n'].max()}"
assert recs.select("user_id", "item_id").is_duplicated().sum() == 0, "duplicate item per user"
assert set(recs["user_id"].unique()) == set(users), "user set mismatch"
assert recs.height == len(users) * TOPK, f"expected {len(users)*TOPK}, got {recs.height}"
assert recs["rank"].min() == 1 and recs["rank"].max() == TOPK
assert recs.null_count().sum_horizontal().item() == 0, "nulls present"

recs.with_row_index("id").select("id", "user_id", "item_id", "rank").write_csv(args.out)
pct = 100 * top.filter(pl.col("is_hist") == 0).height / top.height
print(f"wrote {args.out}: {recs.height:,} rows, new-slot {pct:.1f}%")
