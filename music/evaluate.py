"""Offline scorer for the music-recommender task.

Unlike task 1, this competition has a real temporal structure, so we can
reproduce the host's metric exactly: hold out the last 15 days of training
(2025-08-16..08-30), which mirrors the 15-day test window, and score
recommendations against what users actually listened to.

Metric (from the competition Overview):
  per recommended track, the listening fraction quantised to {0, .25, .5, .75, 1}
  summed over the user's 50 recommendations   -> user score in [0, 50]
  averaged over ALL test users (inactive ones score 0)
  divided by 50                               -> final score in [0, 1]
"""

import polars as pl

CUT = "2025-08-16"
TOPK = 50


def load(cut=CUT):
    df = pl.read_csv(
        "interactions.csv",
        columns=["user_id", "item_id", "listened_duration", "listened_datetime"],
    ).with_columns(pl.col("listened_datetime").str.slice(0, 10).alias("d"))
    meta = pl.read_csv("item_metadata.csv", columns=["item_id", "track_duration"])
    users = pl.read_csv("test.csv")["user_id"]
    return df.filter(pl.col("d") < cut), df.filter(pl.col("d") >= cut), meta, users


def truth(holdout, meta, users):
    """Per (user, item) listening fraction in the holdout window.

    Replays do not count extra, so total listened time is capped at one full
    play of the track before quantising.
    """
    t = (
        holdout.filter(pl.col("user_id").is_in(users.implode()))
        .group_by(["user_id", "item_id"])
        .agg(pl.col("listened_duration").sum().alias("secs"))
        .join(meta, on="item_id", how="left")
        .filter(pl.col("track_duration") > 0)
        .with_columns(
            (pl.col("secs") / pl.col("track_duration")).clip(0, 1).alias("raw")
        )
        .with_columns(((pl.col("raw") * 4).round() / 4).alias("frac"))
        .select("user_id", "item_id", "frac")
    )
    return t


def score(recs, truth_df, users, topk=TOPK, verbose=True):
    """recs: DataFrame with user_id, item_id, rank (1..50)."""
    recs = recs.filter(pl.col("rank") <= topk)
    hit = recs.join(truth_df, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("frac").fill_null(0.0)
    )
    per_user = hit.group_by("user_id").agg(pl.col("frac").sum().alias("s"))
    # Users with no recommendations still count as 0 -- divide by the full test set.
    total = per_user["s"].sum()
    final = total / len(users) / topk
    if verbose:
        covered = per_user.filter(pl.col("s") > 0).height
        print(f"  users with any hit: {covered}/{len(users)}   "
              f"mean user score {total/len(users):.2f}/50")
    return final
