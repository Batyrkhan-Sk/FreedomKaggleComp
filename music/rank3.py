"""Feature set v3 -- aimed squarely at "will this user replay this track soon?".

Calibration is now confirmed (offline-active x1.25 = leaderboard), and the gap
to the leaders is entirely ranking precision. The features added here target
things the v2 set could not express:

  * consistency  -- a track played across many distinct weeks is core rotation,
                    not a one-off; raw play counts cannot tell those apart.
  * momentum     -- plays in the last 3/7 days versus the user's baseline.
  * stickiness   -- some tracks are globally replayed, others are heard once;
                    a track's average plays-per-listener captures that.
  * user habits  -- how repetitive this listener is overall, which sets how much
                    weight their own history deserves.
"""

import polars as pl

FEATS = [
    # pair-level volume
    "plays", "decay_plays", "full_plays", "decay_full", "secs",
    "mean_f", "max_f", "std_f",
    # pair-level timing
    "recency", "first_age", "span", "n_days", "n_weeks", "week_consistency",
    "plays_3", "plays_7", "plays_14", "plays_30", "momentum",
    # user-level
    "u_plays", "u_items", "u_plays_7", "u_recency", "u_repeat_rate", "u_mean_f",
    # item-level
    "item_pop", "item_mean_f", "item_trend", "item_stickiness", "item_age",
    # relative
    "share_of_user", "rank_in_user", "track_duration",
    # artist-level
    "artist_aff", "artist_share", "artist_n_items",
]


def featurise(inter, meta, artists, users, cut, halflife=21.0):
    cut_dt = pl.lit(cut).str.to_date()
    hist = inter.filter(pl.col("d") < cut)

    base = (
        hist.filter(pl.col("user_id").is_in(users.implode()))
        .join(meta, on="item_id", how="left")
        .filter(pl.col("track_duration") > 0)
        .with_columns(
            (cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"),
            (pl.col("listened_duration") / pl.col("track_duration")).clip(0, 1).alias("f"),
        )
        .with_columns(
            (0.5 ** (pl.col("age") / halflife)).alias("decay"),
            (pl.col("age") // 7).alias("wk"),
        )
    )

    g = base.group_by(["user_id", "item_id"]).agg(
        pl.len().alias("plays"),
        pl.col("decay").sum().alias("decay_plays"),
        pl.col("f").sum().alias("full_plays"),
        (pl.col("f") * pl.col("decay")).sum().alias("decay_full"),
        pl.col("listened_duration").sum().alias("secs"),
        pl.col("f").mean().alias("mean_f"),
        pl.col("f").max().alias("max_f"),
        pl.col("f").std().fill_null(0.0).alias("std_f"),
        pl.col("age").min().alias("recency"),
        pl.col("age").max().alias("first_age"),
        pl.col("d").n_unique().alias("n_days"),
        pl.col("wk").n_unique().alias("n_weeks"),
        (pl.col("age") <= 3).sum().alias("plays_3"),
        (pl.col("age") <= 7).sum().alias("plays_7"),
        (pl.col("age") <= 14).sum().alias("plays_14"),
        (pl.col("age") <= 30).sum().alias("plays_30"),
        pl.col("track_duration").first().alias("track_duration"),
    ).with_columns(
        (pl.col("first_age") - pl.col("recency")).alias("span"),
    ).with_columns(
        # fraction of the weeks since first play in which it was played again
        (pl.col("n_weeks") / ((pl.col("span") / 7) + 1)).alias("week_consistency"),
        # recent burst relative to lifetime rate
        (pl.col("plays_7") / (pl.col("plays") / (pl.col("span") + 1) * 7 + 0.5)).alias("momentum"),
    )

    u = base.group_by("user_id").agg(
        pl.len().alias("u_plays"),
        pl.col("item_id").n_unique().alias("u_items"),
        (pl.col("age") <= 7).sum().alias("u_plays_7"),
        pl.col("age").min().alias("u_recency"),
        pl.col("f").mean().alias("u_mean_f"),
    ).with_columns(
        # >1 means the user replays; ~1 means they consume each track once
        (pl.col("u_plays") / pl.col("u_items")).alias("u_repeat_rate")
    )

    it = (
        hist.join(meta, on="item_id", how="left")
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
            pl.col("age").max().alias("item_age"),
        )
        .with_columns(
            (pl.col("recent14") / (pl.col("alltime") + 1)).alias("item_trend"),
            # average plays per distinct listener -- how "sticky" the track is
            (pl.col("alltime") / (pl.col("item_pop") + 1)).alias("item_stickiness"),
        )
        .select("item_id", "item_pop", "item_mean_f", "item_trend",
                "item_stickiness", "item_age")
    )

    ua = (
        base.join(artists, on="item_id", how="left")
        .filter(pl.col("artist_name").is_not_null())
        .group_by(["user_id", "artist_name"])
        .agg(pl.col("decay").sum().alias("artist_aff"),
             pl.col("item_id").n_unique().alias("artist_n_items"))
    )

    g = (
        g.join(u, on="user_id", how="left")
        .join(it, on="item_id", how="left")
        .join(artists, on="item_id", how="left")
        .join(ua, on=["user_id", "artist_name"], how="left")
        .with_columns(
            (pl.col("plays") / pl.col("u_plays")).alias("share_of_user"),
            pl.col("artist_aff").fill_null(0.0),
            pl.col("artist_n_items").fill_null(0),
        )
        .with_columns(
            (pl.col("artist_aff") / (pl.col("decay_plays") + 1e-6)).alias("artist_share"),
            # where this track sits inside the user's own ordering
            pl.col("decay_plays").rank("ordinal", descending=True).over("user_id").alias("rank_in_user"),
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
