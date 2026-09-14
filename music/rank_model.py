"""Learned ranker for the repeat-listening slots.

Heuristic orderings of a user's catalogue have plateaued around 0.211 against a
history-only ceiling of 0.284, so the remaining gap is a prediction problem:
given how a user has engaged with a track, how much of it will they listen to
in the next fortnight?

Training uses an earlier temporal split so the model never sees its own target:
    features from  < FEAT_CUT
    target from    [FEAT_CUT, FEAT_CUT + 15d)
and is then applied with features from < CUT to rank the real window.
"""

import numpy as np
import polars as pl

FEATS = [
    "plays", "decay_plays", "full_plays", "decay_full", "mean_f", "max_f",
    "recency", "first_age", "span", "secs", "n_days",
    "u_plays", "u_items", "item_pop", "item_mean_f", "share_of_user",
    "track_duration",
]


def featurise(inter, meta, users, cut, halflife=21.0):
    """Per (user, item) features from interactions strictly before `cut`."""
    cut_dt = pl.lit(cut).str.to_date()
    w = (
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
    g = w.group_by(["user_id", "item_id"]).agg(
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
        pl.col("track_duration").first().alias("track_duration"),
    ).with_columns((pl.col("first_age") - pl.col("recency")).alias("span"))

    u = g.group_by("user_id").agg(
        pl.col("plays").sum().alias("u_plays"),
        pl.len().alias("u_items"),
    )
    # Item-level popularity from the whole population, not just test users.
    it = (
        inter.filter(pl.col("d") < cut)
        .join(meta, on="item_id", how="left")
        .filter(pl.col("track_duration") > 0)
        .with_columns(
            (pl.col("listened_duration") / pl.col("track_duration")).clip(0, 1).alias("f")
        )
        .group_by("item_id")
        .agg(
            pl.col("user_id").n_unique().alias("item_pop"),
            pl.col("f").mean().alias("item_mean_f"),
        )
    )
    g = (
        g.join(u, on="user_id", how="left")
        .join(it, on="item_id", how="left")
        .with_columns((pl.col("plays") / pl.col("u_plays")).alias("share_of_user"))
    )
    return g


def target(inter, meta, users, lo, hi):
    cut_dt_lo, cut_dt_hi = pl.lit(lo), pl.lit(hi)
    return (
        inter.filter((pl.col("d") >= cut_dt_lo) & (pl.col("d") < cut_dt_hi))
        .filter(pl.col("user_id").is_in(users.implode()))
        .group_by(["user_id", "item_id"])
        .agg(pl.col("listened_duration").sum().alias("secs2"))
        .join(meta, on="item_id", how="left")
        .filter(pl.col("track_duration") > 0)
        .with_columns((pl.col("secs2") / pl.col("track_duration")).clip(0, 1).alias("raw"))
        .with_columns(((pl.col("raw") * 4).round() / 4).alias("y"))
        .select("user_id", "item_id", "y")
    )
