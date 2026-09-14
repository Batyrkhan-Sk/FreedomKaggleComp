"""Two-stage recommender: wide candidate pool, then one model ranks everything.

The diagnostic that motivated this: new-to-the-user tracks carry 36.7 of the
50 available points per active user, against 26.8 for repeats -- yet every
earlier attempt bolted discovery on as an afterthought and lost. New-item value
is spread over ~15k tracks, so a 50-item trending list reaches only 10.6% of it
while a 1000-item pool reaches 44%.

So: candidates = everything the user has played + the top-N trending tracks they
have not, and a single model scores both kinds side by side. For a new track the
pair-level history features are all zero, and the model has to lean on artist
affinity, genre affinity and the track's own trend -- which is exactly the
personalisation the trending pad could not express.
"""

import numpy as np
import polars as pl

PAIR = [
    "plays", "decay_plays", "full_plays", "decay_full", "secs",
    "mean_f", "max_f", "recency", "first_age", "span", "n_days", "n_weeks",
    "plays_7", "plays_30", "share_of_user",
]
CTX = [
    "is_hist", "u_plays", "u_items", "u_plays_7", "u_recency", "u_repeat_rate",
    "u_mean_f", "item_pop", "item_mean_f", "item_trend", "item_stickiness",
    "track_duration", "artist_aff", "artist_n_items", "genre_aff", "cand_rank",
]
FEATS = PAIR + CTX


def _prep(inter, meta, cut, halflife):
    cut_dt = pl.lit(cut).str.to_date()
    return (
        inter.filter(pl.col("d") < cut)
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


def item_stats(all_hist):
    return (
        all_hist.group_by("item_id")
        .agg(
            pl.col("user_id").n_unique().alias("item_pop"),
            pl.col("f").mean().alias("item_mean_f"),
            (pl.col("age") <= 14).sum().alias("r14"),
            pl.len().alias("alltime"),
            pl.col("track_duration").first().alias("track_duration"),
        )
        .with_columns(
            (pl.col("r14") / (pl.col("alltime") + 1)).alias("item_trend"),
            (pl.col("alltime") / (pl.col("item_pop") + 1)).alias("item_stickiness"),
        )
        .select("item_id", "item_pop", "item_mean_f", "item_trend",
                "item_stickiness", "track_duration")
    )


def build_candidates(inter, meta, artists, genres, users, cut,
                     halflife=21.0, n_pool=1500):
    """History items plus the top-`n_pool` trending tracks the user lacks."""
    all_hist = _prep(inter, meta, cut, halflife)
    ustats = item_stats(all_hist)

    base = all_hist.filter(pl.col("user_id").is_in(users.implode()))

    pair = base.group_by(["user_id", "item_id"]).agg(
        pl.len().alias("plays"),
        pl.col("decay").sum().alias("decay_plays"),
        pl.col("f").sum().alias("full_plays"),
        (pl.col("f") * pl.col("decay")).sum().alias("decay_full"),
        pl.col("listened_duration").sum().alias("secs"),
        pl.col("f").mean().alias("mean_f"),
        pl.col("f").max().alias("max_f"),
        pl.col("age").min().alias("recency"),
        pl.col("age").max().alias("first_age"),
        pl.col("d").n_unique().alias("n_days"),
        pl.col("wk").n_unique().alias("n_weeks"),
        (pl.col("age") <= 7).sum().alias("plays_7"),
        (pl.col("age") <= 30).sum().alias("plays_30"),
    ).with_columns((pl.col("first_age") - pl.col("recency")).alias("span"))

    u = base.group_by("user_id").agg(
        pl.len().alias("u_plays"),
        pl.col("item_id").n_unique().alias("u_items"),
        (pl.col("age") <= 7).sum().alias("u_plays_7"),
        pl.col("age").min().alias("u_recency"),
        pl.col("f").mean().alias("u_mean_f"),
    ).with_columns((pl.col("u_plays") / pl.col("u_items")).alias("u_repeat_rate"))

    # user -> artist and user -> genre affinity, both recency-decayed
    ua = (
        base.join(artists, on="item_id", how="left")
        .filter(pl.col("artist_name").is_not_null())
        .group_by(["user_id", "artist_name"])
        .agg(pl.col("decay").sum().alias("artist_aff"),
             pl.col("item_id").n_unique().alias("artist_n_items"))
    )
    ug = (
        base.join(genres, on="item_id", how="left")
        .filter(pl.col("genre").is_not_null())
        .group_by(["user_id", "genre"])
        .agg(pl.col("decay").sum().alias("genre_aff"))
    )

    # trending pool, ranked; cand_rank lets the model know how mainstream it is
    pool = (
        all_hist.filter(pl.col("age") <= 14)
        .group_by("item_id")
        .agg(pl.col("f").sum().alias("s"))
        .sort("s", descending=True)
        .head(n_pool)
        .with_row_index("cand_rank")
        .select("item_id", "cand_rank")
    )

    hist_c = pair.select("user_id", "item_id").with_columns(pl.lit(1).alias("is_hist"))
    pool_c = users.to_frame().join(pool, how="cross").with_columns(
        pl.lit(0).alias("is_hist")
    )
    cand = (
        pl.concat([hist_c.with_columns(pl.lit(None, dtype=pl.UInt32).alias("cand_rank")),
                   pool_c.select("user_id", "item_id", "is_hist", "cand_rank")],
                  how="diagonal")
        .unique(subset=["user_id", "item_id"], keep="first")
    )

    out = (
        cand.join(pair, on=["user_id", "item_id"], how="left")
        .join(u, on="user_id", how="left")
        .join(ustats, on="item_id", how="left")
        .join(artists, on="item_id", how="left")
        .join(ua, on=["user_id", "artist_name"], how="left")
        .join(genres, on="item_id", how="left")
        .join(ug, on=["user_id", "genre"], how="left")
        .with_columns([pl.col(c).fill_null(0.0) for c in PAIR if c != "share_of_user"])
        .with_columns(
            pl.col("artist_aff").fill_null(0.0),
            pl.col("artist_n_items").fill_null(0),
            pl.col("genre_aff").fill_null(0.0),
            pl.col("cand_rank").fill_null(99999),
        )
        .with_columns(
            (pl.col("plays") / (pl.col("u_plays") + 1e-6)).alias("share_of_user")
        )
        .unique(subset=["user_id", "item_id"], keep="first")
    )
    return out


def load_genres():
    """One row per (item, primary genre) -- the list column is a stringified list."""
    g = pl.read_csv("item_metadata.csv", columns=["item_id", "track_genres_list"])
    return (
        g.with_columns(
            pl.col("track_genres_list")
            .str.replace_all(r"[\[\]']", "")
            .str.split(",")
            .list.first()
            .str.strip_chars()
            .alias("genre")
        )
        .select("item_id", "genre")
    )
