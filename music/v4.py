"""Ranking a user's own catalogue.

Most users have far more than 50 distinct tracks in history, so trending rarely
gets a slot and the whole score hinges on picking the right 50 from what they
already play. The question is which signal best predicts "will replay, and
finish, in the next fortnight": raw frequency, recency, or completion.
"""

import polars as pl

from evaluate import CUT, TOPK, load, score, truth
from v3 import trend_list
import v2

CUT_DT = None  # set in main


def user_scores(train, meta, users, halflife=21.0, mode="decay_plays"):
    w = (
        train.filter(pl.col("user_id").is_in(users.implode()))
        .join(meta, on="item_id", how="left")
        .filter(pl.col("track_duration") > 0)
        .with_columns(
            (v2.CUT_DT - pl.col("d").str.to_date()).dt.total_days().alias("age"),
            (pl.col("listened_duration") / pl.col("track_duration")).clip(0, 1).alias("f"),
        )
        .with_columns((0.5 ** (pl.col("age") / halflife)).alias("decay"))
    )
    g = w.group_by(["user_id", "item_id"]).agg(
        pl.len().alias("plays"),
        pl.col("decay").sum().alias("decay_plays"),
        pl.col("f").sum().alias("full_plays"),
        (pl.col("f") * pl.col("decay")).sum().alias("decay_full"),
        pl.col("f").mean().alias("mean_f"),
        pl.col("age").min().alias("recency"),
        pl.col("listened_duration").sum().alias("secs"),
    )
    if mode == "recency":
        return g.with_columns((-pl.col("recency")).alias("s"))
    if mode == "decay_full":
        return g.with_columns(pl.col("decay_full").alias("s"))
    if mode == "full_plays":
        return g.with_columns(pl.col("full_plays").alias("s"))
    if mode == "decay_x_meanf":
        return g.with_columns((pl.col("decay_plays") * pl.col("mean_f")).alias("s"))
    if mode == "decay_plays_recency":
        # frequency, tie-broken hard toward things played recently
        return g.with_columns(
            (pl.col("decay_plays") * (1.0 / (1.0 + pl.col("recency") / 30.0))).alias("s")
        )
    return g.with_columns(pl.col("decay_plays").alias("s"))


def build(train, meta, users, halflife=21.0, mode="decay_plays", days=14, topk=TOPK):
    g = user_scores(train, meta, users, halflife, mode)
    hist = (
        g.sort(["user_id", "s", "secs"], descending=[False, True, True])
        .group_by("user_id", maintain_order=True)
        .head(topk)
    )
    by_user = {}
    for row in hist.select("user_id", "item_id").iter_rows():
        by_user.setdefault(row[0], []).append(row[1])
    pop = trend_list(train, meta, days, "completion", topk * 4)

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
    print("baseline (decay_plays, hl=21) = 0.20812\n")
    for mode in ("decay_plays", "recency", "decay_full", "full_plays",
                 "decay_x_meanf", "decay_plays_recency"):
        s = score(build(train, meta, users, mode=mode), t, users, verbose=False)
        print(f"  {mode:<22} -> {s:.5f}")
