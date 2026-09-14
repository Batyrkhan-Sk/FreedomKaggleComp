"""Better trending lists, and the right split between repeat and discovery slots.

CF lost badly (0.04 vs trending's 0.14), so discovery here is a global-trends
problem, not a taste-neighbour problem. What remains is choosing *which* global
tracks: the metric pays for how much of a track gets listened to, so a track
people finish is worth more than a track people merely start.
"""

import polars as pl

from evaluate import CUT, TOPK, load, score, truth
from v2 import hist_recency
import v2


def trend_list(train, meta, days=14, mode="users", n=200):
    """Candidate global tracks, ranked several different ways."""
    lo = (v2.CUT_DT - pl.duration(days=days)).cast(pl.Utf8)
    w = train.filter(pl.col("d") >= lo).join(meta, on="item_id", how="left").filter(
        pl.col("track_duration") > 0
    )
    if mode == "users":                       # distinct listeners (the v2 baseline)
        agg = w.group_by("item_id").agg(pl.col("user_id").n_unique().alias("s"))
    elif mode == "plays":
        agg = w.group_by("item_id").agg(pl.len().alias("s"))
    elif mode == "completion":
        # total listened time expressed in "full plays" -- rewards tracks people finish
        agg = w.with_columns(
            (pl.col("listened_duration") / pl.col("track_duration")).clip(0, 1).alias("f")
        ).group_by("item_id").agg(pl.col("f").sum().alias("s"))
    elif mode == "mean_completion":
        # average completion, but only for tracks with real traction
        agg = (
            w.with_columns(
                (pl.col("listened_duration") / pl.col("track_duration")).clip(0, 1).alias("f")
            )
            .group_by("item_id")
            .agg(pl.col("f").mean().alias("m"), pl.col("user_id").n_unique().alias("u"))
            .filter(pl.col("u") >= 50)
            .with_columns((pl.col("m") * pl.col("u").log1p()).alias("s"))
        )
    return agg.sort("s", descending=True).head(n)["item_id"].to_list()


def build(train, meta, users, halflife=21.0, n_hist=50, days=14, mode="completion",
          topk=TOPK):
    hist = hist_recency(train, users, halflife, topk).filter(pl.col("rank") <= n_hist)
    by_user = {}
    for u, i, r in hist.iter_rows():
        by_user.setdefault(u, []).append(i)
    pop = trend_list(train, meta, days, mode, topk * 4)

    rows = []
    for u in users.to_list():
        chosen, seen = [], set()
        for it in by_user.get(u, []):
            if it not in seen:
                seen.add(it); chosen.append(it)
        for it in pop:
            if len(chosen) >= topk:
                break
            if it not in seen:
                seen.add(it); chosen.append(it)
        for r, it in enumerate(chosen[:topk], 1):
            rows.append((u, it, r))
    return pl.DataFrame(rows, schema=["user_id", "item_id", "rank"], orient="row")


if __name__ == "__main__":
    train, holdout, meta, users = load()
    t = truth(holdout, meta, users)
    print("baseline blend = 0.20812\n")

    print("=== trending ranking mode (full 50 history slots allowed) ===")
    for mode in ("users", "plays", "completion", "mean_completion"):
        s = score(build(train, meta, users, mode=mode), t, users, verbose=False)
        print(f"  mode={mode:<16} -> {s:.5f}")
