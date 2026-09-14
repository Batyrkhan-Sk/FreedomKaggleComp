"""Train the stacked-window ranker and evaluate on the Aug 16-30 holdout."""

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from rank3 import FEATS, featurise, target
from v3 import trend_list
import v2

# Stacked training windows: features before `cut`, target in the 15 days after.
WINDOWS = [("2025-06-15", "2025-06-30"), ("2025-07-01", "2025-07-16"),
           ("2025-07-16", "2025-07-31"), ("2025-08-01", "2025-08-16")]
APPLY = "2025-08-16"


def load_all():
    inter = pl.read_csv(
        "interactions.csv",
        columns=["user_id", "item_id", "listened_duration", "listened_datetime"],
    ).with_columns(pl.col("listened_datetime").str.slice(0, 10).alias("d"))
    meta = pl.read_csv("item_metadata.csv", columns=["item_id", "track_duration"])
    artists = pl.read_csv("item_metadata.csv", columns=["item_id", "artist_name"])
    users = pl.read_csv("test.csv")["user_id"]
    return inter, meta, artists, users


def build_training(inter, meta, artists, users):
    parts = []
    for cut, hi in WINDOWS:
        X = featurise(inter, meta, artists, users, cut)
        y = target(inter, meta, users, cut, hi)
        p = X.join(y, on=["user_id", "item_id"], how="left").with_columns(
            pl.col("y").fill_null(0.0)
        )
        parts.append(p.select(FEATS + ["y"]))
        print(f"  window {cut}: {p.height:,} pairs, {p.filter(pl.col('y')>0).height:,} positive")
    return pl.concat(parts)


def main():
    inter, meta, artists, users = load_all()
    print("building stacked training windows...")
    tr = build_training(inter, meta, artists, users)
    print(f"  total {tr.height:,} rows")

    m = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
    )
    m.fit(tr.select(FEATS).to_numpy(), tr["y"].to_numpy())
    print(f"  fitted, {m.n_iter_} iterations")

    print("applying to holdout...")
    X = featurise(inter, meta, artists, users, APPLY)
    X = X.with_columns(pl.Series("p", m.predict(X.select(FEATS).to_numpy())))
    hist = (
        X.sort(["user_id", "p"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .head(TOPK)
    )
    by_user = {}
    for u, i in hist.select("user_id", "item_id").iter_rows():
        by_user.setdefault(u, []).append(i)

    v2.CUT_DT = pl.lit(APPLY).str.to_date()
    pop = trend_list(inter.filter(pl.col("d") < APPLY), meta, 14, "completion", TOPK * 4)

    rows = []
    for u in users.to_list():
        chosen, seen = [], set()
        for src in (by_user.get(u, []), pop):
            for it in src:
                if len(chosen) >= TOPK:
                    break
                if it not in seen:
                    seen.add(it); chosen.append(it)
        for r, it in enumerate(chosen[:TOPK], 1):
            rows.append((u, it, r))
    recs = pl.DataFrame(rows, schema=["user_id", "item_id", "rank"], orient="row")

    holdout = inter.filter(pl.col("d") >= APPLY)
    t = truth(holdout, meta, users)
    active = t["user_id"].n_unique()
    hit = recs.join(t, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("frac").fill_null(0.0)
    )
    tot = hit["frac"].sum()
    print(f"\n  all-users     {tot/1500/TOPK:.5f}   (prev best 0.21079)")
    print(f"  ACTIVE-only   {tot/active/TOPK:.5f}   (prev best 0.27080, real LB 0.33862)")
    print(f"  projected LB  ~{tot/active/TOPK*1.25:.3f}")


if __name__ == "__main__":
    main()
