"""Two-stage ranker plus co-visitation features.

Adds one signal to train4: how strongly each pool candidate co-occurs with the
user's current rotation. Standalone CF ranked at 0.04, but as a feature the
model can use it selectively -- only for the new-item slots where the pair-level
history features are all zero and it currently has nothing personal to go on.
"""

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from covis import covis_features
from evaluate import TOPK, truth
from rank3 import target
from rank4 import FEATS as BASE_FEATS, build_candidates, load_genres
from train4 import load_all

FEATS = BASE_FEATS + ["covis", "covis_rank"]
WINDOWS = [("2025-07-01", "2025-07-16"), ("2025-07-16", "2025-07-31"),
           ("2025-08-01", "2025-08-16"), ("2025-08-16", "2025-08-31")]
APPLY = "2025-08-31"
N_POOL = 1500


def with_covis(inter, meta, artists, genres, users, cut, n_pool=N_POOL):
    C = build_candidates(inter, meta, artists, genres, users, cut, n_pool=n_pool)
    pool_items = (
        C.filter(pl.col("is_hist") == 0)["item_id"].unique().to_list()
    )
    cv = covis_features(inter, users, pool_items, cut)
    C = C.join(cv, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("covis").fill_null(0.0)
    )
    # rank within user makes the raw magnitude comparable across users
    C = C.with_columns(
        pl.col("covis").rank("ordinal", descending=True).over("user_id").alias("covis_rank")
    )
    return C


def main():
    inter, meta, artists, genres, users = load_all()

    parts = []
    for cut, hi in WINDOWS:
        C = with_covis(inter, meta, artists, genres, users, cut)
        y = target(inter, meta, users, cut, hi)
        p = C.join(y, on=["user_id", "item_id"], how="left").with_columns(
            pl.col("y").fill_null(0.0)
        )
        nz = p.filter(pl.col("covis") > 0)
        print(f"  {cut}: {p.height:,} cands, covis nonzero on {nz.height:,}")
        parts.append(p.select(FEATS + ["y"]))
    tr = pl.concat(parts)
    print(f"total {tr.height:,} rows")

    m = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, random_state=0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40,
    )
    m.fit(tr.select(FEATS).to_numpy(), tr["y"].to_numpy())
    print(f"fitted {m.n_iter_} iters")

    C = with_covis(inter, meta, artists, genres, users, APPLY)
    C = C.with_columns(pl.Series("p", m.predict(C.select(FEATS).to_numpy())))
    top = (
        C.sort(["user_id", "p"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .head(TOPK)
        .with_columns(pl.int_range(pl.len()).over("user_id").add(1).alias("rank"))
        .select("user_id", "item_id", "rank", "is_hist")
    )
    print(f"  NEW-item slots: {100*top.filter(pl.col('is_hist')==0).height/top.height:.1f}%")

    recs = top.select("user_id", "item_id", "rank").sort(["user_id", "rank"])
    n = recs.group_by("user_id").agg(pl.len().alias("n"))
    assert n["n"].min() == 50 == n["n"].max(), f"counts {n['n'].min()}..{n['n'].max()}"
    assert recs.select("user_id", "item_id").is_duplicated().sum() == 0
    assert set(recs["user_id"].unique()) == set(users)
    out = recs.with_row_index("id").select("id", "user_id", "item_id", "rank")
    out.write_csv("submission_covis.csv")
    print(f"wrote submission_covis.csv: {out.height:,} rows, {out['user_id'].n_unique()} users")


if __name__ == "__main__":
    main()
