"""Stronger learned ranker for the repeat-listening slots.

Calibration against the first real submission (offline-active 0.271 -> real
0.339) projects the history-only oracle at ~0.46 real, and the top leaderboard
score is 0.452. So the winning signal is not discovery -- it is ranking a user's
own catalogue almost perfectly. This module invests there.

Improvements over rank_model.py:
  * trains on several stacked time windows instead of one
  * artist-affinity, trend and user-activity features
  * reports the active-user-only metric, which tracks the leaderboard
"""

import numpy as np
import polars as pl

FEATS = [
    "plays", "decay_plays", "full_plays", "decay_full", "mean_f", "max_f",
    "recency", "first_age", "span", "secs", "n_days", "plays_7", "plays_30",
    "u_plays", "u_items", "u_plays_7", "u_recency",
    "item_pop", "item_mean_f", "item_trend",
    "share_of_user", "track_duration",
    "artist_aff", "artist_share", "artist_n_items",
]


def featurise(inter, meta, artists, users, cut, halflife=21.0):
    cut_dt = pl.lit(cut).str.to_date()
    base = (
        inter.filter(pl.col("d") < cut)
        .filter(pl.col("user_id").is_in(users.implode()))
        .join(meta, on="item_id", how="left")
        .filter(pl.col("track_duration") > 0)
        .with_columns(
            (cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"),
            (pl.col("listened_duration") / pl.col("track_duration")).clip(0, 1).alias("f"),
        )
        .with_columns((0.5 ** (pl.col("age") / halflife)).alias("decay"))
    )

    g = base.group_by(["user_id", "item_id"]).agg(
        pl.len().alias("plays"),
        pl.col("decay").sum().alias("decay_plays"),
        pl.col("f").sum().alias("full_plays"),
        (pl.col("f") * pl.col("decay")).sum().alias("decay_full"),
        pl.col("f").mean().alias("mean_f"),
        pl.col("f").max().alias("max_f"),
        pl.col("age").min().alias("recency"),
        pl.col("age").max().alias("first_age"),
        pl.col("listened_duration").sum().alias("secs"),
        pl.col("d").n_unique().alias("n_days"),
        (pl.col("age") <= 7).sum().alias("plays_7"),
        (pl.col("age") <= 30).sum().alias("plays_30"),
        pl.col("track_duration").first().alias("track_duration"),
    ).with_columns((pl.col("first_age") - pl.col("recency")).alias("span"))

    u = base.group_by("user_id").agg(
        pl.len().alias("u_plays"),
        pl.col("item_id").n_unique().alias("u_items"),
        (pl.col("age") <= 7).sum().alias("u_plays_7"),
        pl.col("age").min().alias("u_recency"),
    )

    pop_all = (
        inter.filter(pl.col("d") < cut)
        .join(meta, on="item_id", how="left")
        .filter(pl.col("track_duration") > 0)
        .with_columns(
            (cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"),
            (pl.col("listened_duration") / pl.col("track_duration")).clip(0, 1).alias("f"),
        )
        .group_by("item_id")
        .agg(
            pl.col("user_id").n_unique().alias("item_pop"),
            pl.col("f").mean().alias("item_mean_f"),
            (pl.col("age") <= 14).sum().alias("recent14"),
            pl.len().alias("alltime"),
        )
        # rising vs fading: share of a track's plays that are recent
        .with_columns((pl.col("recent14") / (pl.col("alltime") + 1)).alias("item_trend"))
        .select("item_id", "item_pop", "item_mean_f", "item_trend")
    )

    ua = (
        base.join(artists, on="item_id", how="left")
        .filter(pl.col("artist_name").is_not_null())
        .group_by(["user_id", "artist_name"])
        .agg(
            pl.col("decay").sum().alias("artist_aff"),
            pl.col("item_id").n_unique().alias("artist_n_items"),
        )
    )

    g = (
        g.join(u, on="user_id", how="left")
        .join(pop_all, on="item_id", how="left")
        .join(artists, on="item_id", how="left")
        .join(ua, on=["user_id", "artist_name"], how="left")
        .with_columns(
            (pl.col("plays") / pl.col("u_plays")).alias("share_of_user"),
            pl.col("artist_aff").fill_null(0.0),
            pl.col("artist_n_items").fill_null(0),
        )
        .with_columns(
            (pl.col("artist_aff") / (pl.col("decay_plays") + 1e-6)).alias("artist_share")
        )
    )
    return g


def target(inter, meta, users, lo, hi):
    return (
        inter.filter((pl.col("d") >= pl.lit(lo)) & (pl.col("d") < pl.lit(hi)))
        .filter(pl.col("user_id").is_in(users.implode()))
        .group_by(["user_id", "item_id"])
        .agg(pl.col("listened_duration").sum().alias("s2"))
        .join(meta, on="item_id", how="left")
        .filter(pl.col("track_duration") > 0)
        .with_columns((pl.col("s2") / pl.col("track_duration")).clip(0, 1).alias("raw"))
        .with_columns(((pl.col("raw") * 4).round() / 4).alias("y"))
        .select("user_id", "item_id", "y")
    )
