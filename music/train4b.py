"""Train the two-stage ranker and evaluate on the Aug 16-30 holdout."""

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from rank4 import FEATS, build_candidates, load_genres
from rank3 import target

WINDOWS = [("2025-07-01", "2025-07-16"), ("2025-07-16", "2025-07-31"),
           ("2025-08-01", "2025-08-16")]
APPLY = "2025-08-16"
N_POOL = 4000


def load_all():
    inter = pl.read_csv(
        "interactions.csv",
        columns=["user_id", "item_id", "listened_duration", "listened_datetime"],
    ).with_columns(pl.col("listened_datetime").str.slice(0, 10).alias("d"))
    meta = pl.read_csv("item_metadata.csv", columns=["item_id", "track_duration"])
    artists = pl.read_csv("item_metadata.csv", columns=["item_id", "artist_name"])
    genres = load_genres()
    users = pl.read_csv("test.csv")["user_id"]
    return inter, meta, artists, genres, users


def main():
    inter, meta, artists, genres, users = load_all()

    parts = []
    for cut, hi in WINDOWS:
        C = build_candidates(inter, meta, artists, genres, users, cut, n_pool=N_POOL)
        y = target(inter, meta, users, cut, hi)
        p = C.join(y, on=["user_id", "item_id"], how="left").with_columns(
            pl.col("y").fill_null(0.0)
        )
        pos = p.filter(pl.col("y") > 0)
        print(f"  {cut}: {p.height:,} candidates, {pos.height:,} positive "
              f"({pos.filter(pl.col('is_hist')==0).height:,} of them NEW)")
        parts.append(p.select(FEATS + ["y"]))
    tr = pl.concat(parts)
    print(f"total training rows {tr.height:,}")

    m = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, random_state=0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40,
    )
    m.fit(tr.select(FEATS).to_numpy(), tr["y"].to_numpy())
    print(f"fitted {m.n_iter_} iters")

    C = build_candidates(inter, meta, artists, genres, users, APPLY, n_pool=N_POOL)
    C = C.with_columns(pl.Series("p", m.predict(C.select(FEATS).to_numpy())))
    top = (
        C.sort(["user_id", "p"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .head(TOPK)
        .with_columns(pl.int_range(pl.len()).over("user_id").add(1).alias("rank"))
        .select("user_id", "item_id", "rank", "is_hist")
    )
    print(f"  recommended slots that are NEW items: "
          f"{100*top.filter(pl.col('is_hist')==0).height/top.height:.1f}%")

    holdout = inter.filter(pl.col("d") >= APPLY)
    t = truth(holdout, meta, users)
    active = t["user_id"].n_unique()
    hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("frac").fill_null(0.0)
    )
    tot = hit["frac"].sum()
    print(f"\n  all-users     {tot/1500/TOPK:.5f}")
    print(f"  ACTIVE-only   {tot/active/TOPK:.5f}   (prev best 0.28943)")
    print(f"  projected LB  ~{tot/active/TOPK*1.25:.3f}   (current real 0.36087)")


if __name__ == "__main__":
    main()
