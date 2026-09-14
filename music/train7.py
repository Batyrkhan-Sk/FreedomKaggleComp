"""Two-stage ranker with co-visitation and latent-factor affinity."""

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from covis import covis_features
from evaluate import TOPK, truth
from latent import latent_features
from rank3 import target
from rank4 import FEATS as BASE_FEATS, build_candidates, load_genres
from train4 import load_all

FEATS = BASE_FEATS + ["covis", "covis_rank", "latent", "latent_rank"]
WINDOWS = [("2025-07-01", "2025-07-16"), ("2025-07-16", "2025-07-31"),
           ("2025-08-01", "2025-08-16")]
APPLY = "2025-08-16"
N_POOL = 1500


def featurise(inter, meta, artists, genres, users, cut):
    C = build_candidates(inter, meta, artists, genres, users, cut, n_pool=N_POOL)
    pool = C.filter(pl.col("is_hist") == 0)["item_id"].unique().to_list()
    C = (
        C.join(covis_features(inter, users, pool, cut), on=["user_id", "item_id"], how="left")
         .join(latent_features(inter, users, pool, cut), on=["user_id", "item_id"], how="left")
         .with_columns(pl.col("covis").fill_null(0.0), pl.col("latent").fill_null(0.0))
    )
    # rank("min"), not "ordinal": 42% of covis and 28% of latent are exactly
    # 0.0, and "ordinal" hands those ties distinct ranks by ROW POSITION -- a
    # median of 360 rows per user whose feature value is an artifact of polars
    # row emission rather than of the data. "min" gives ties the same rank:
    # deterministic, and semantically right.
    # Pin ROW ORDER too, not just values. HistGradientBoosting carves its
    # validation_fraction by POSITION (a seeded shuffle over row indices), so two
    # runs with identical features in a different order train on different data
    # and stop at different points. Measured cost: a no-op join reordering rows
    # moved the score by 0.0019 -- the same size as the effects being measured.
    return C.with_columns(
        pl.col("covis").rank("min", descending=True).over("user_id").alias("covis_rank"),
        pl.col("latent").rank("min", descending=True).over("user_id").alias("latent_rank"),
    ).sort(["user_id", "item_id"])


def main():
    inter, meta, artists, genres, users = load_all()
    parts = []
    for cut, hi in WINDOWS:
        print(f"window {cut}")
        C = featurise(inter, meta, artists, genres, users, cut)
        p = C.join(target(inter, meta, users, cut, hi), on=["user_id", "item_id"],
                   how="left").with_columns(pl.col("y").fill_null(0.0))
        print(f"  {p.height:,} cands, latent nonzero on {p.filter(pl.col('latent')!=0).height:,}")
        parts.append(p.select(FEATS + ["y"]))
    tr = pl.concat(parts)
    print(f"total {tr.height:,} rows")

    m = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, random_state=0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40)
    m.fit(tr.select(FEATS).to_numpy(), tr["y"].to_numpy())
    print(f"fitted {m.n_iter_} iters")

    C = featurise(inter, meta, artists, genres, users, APPLY)
    C = C.with_columns(pl.Series("p", m.predict(C.select(FEATS).to_numpy())))
    top = (C.sort(["user_id", "p"], descending=[False, True])
             .group_by("user_id", maintain_order=True).head(TOPK)
             .with_columns(pl.int_range(pl.len()).over("user_id").add(1).alias("rank"))
             .select("user_id", "item_id", "rank", "is_hist"))
    print(f"  NEW-item slots: {100*top.filter(pl.col('is_hist')==0).height/top.height:.1f}%")

    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()
    hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(pl.col("frac").fill_null(0.0))
    tot = hit["frac"].sum()
    print(f"\n  ACTIVE-only   {tot/active/TOPK:.5f}   (covis-only 0.29170)")
    print(f"  projected LB  ~{tot/active/TOPK*1.29:.3f}   (current real 0.37553)")


if __name__ == "__main__":
    main()
