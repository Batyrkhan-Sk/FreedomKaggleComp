"""Control: what does NO model score?

The capacity sweep keeps improving as capacity falls, and the best config now
sits at 40 iterations -- barely-trained. Two readings fit that curve equally:

  (a) regularisation genuinely helps, or
  (b) the sweep is walking toward "do not use the model at all", and the gain is
      measuring the GBM's REMOVAL rather than its improvement.

These have opposite consequences, and no amount of further sweeping separates
them. The control does: score a trivial ranker with no model at all.

  history -> decay_plays  (recency-weighted play count)
  new     -> cand_rank    (trending order, i.e. the pool's own ranking)
  history before new, matching the ~34/16 emergent split

If the trivial ranker lands near the swept configs, the GBM is contributing
close to nothing and the whole task needs re-framing. If it lands far below,
the sweep is real tuning and 40 iters is a genuine (if odd) optimum.
"""
import numpy as np
import polars as pl

from evaluate import TOPK, truth
from train4 import load_all
from train7 import APPLY
from train12 import cached

BASE = 0.29035

inter, meta, artists, genres, users = load_all()
C = cached("apply", None)
t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
active = t["user_id"].n_unique()


def score(df, label):
    hit = (df.select("user_id", "item_id", "is_hist")
             .join(t, on=["user_id", "item_id"], how="left")
             .with_columns(pl.col("frac").fill_null(0.0)))
    off = hit["frac"].sum() / active / TOPK
    hs = hit.filter(pl.col("is_hist") == 1)["frac"].sum()
    ns = hit.filter(pl.col("is_hist") == 0)["frac"].sum()
    print(f"{label:38s} offline {off:.5f}  LB~{off*1.29:.4f}  "
          f"({off-BASE:+.5f} vs GBM-600)  hist {hs:,.0f} new {ns:,.0f}")
    return off


print(f"GBM 600 baseline {BASE:.5f}   best swept (40 iters) 0.29545\n")

# history first by decay_plays, then new by trending rank -- no model anywhere
trivial = (C.with_columns(
    pl.when(pl.col("is_hist") == 1)
      .then(pl.col("decay_plays"))
      .otherwise(-pl.col("cand_rank").cast(pl.Float64) / 1e6)
      .alias("p"))
    .sort(["user_id", "is_hist", "p"], descending=[False, True, True])
    .group_by("user_id", maintain_order=True).head(TOPK))
score(trivial, "trivial: decay_plays | cand_rank")

# variants, to see how much of it is just "history first"
for col in ("plays", "decay_full", "mean_f", "recency"):
    d = pl.col(col) if col != "recency" else -pl.col(col)
    v = (C.with_columns(
        pl.when(pl.col("is_hist") == 1).then(d)
          .otherwise(-pl.col("cand_rank").cast(pl.Float64) / 1e6).alias("p"))
        .sort(["user_id", "is_hist", "p"], descending=[False, True, True])
        .group_by("user_id", maintain_order=True).head(TOPK))
    score(v, f"trivial: {col} | cand_rank")

# pure popularity, no personalisation at all -- the true floor
pure = (C.with_columns((-pl.col("cand_rank").cast(pl.Float64)).alias("p"))
        .sort(["user_id", "p"], descending=[False, True])
        .group_by("user_id", maintain_order=True).head(TOPK))
score(pure, "pure trending, no history at all")
