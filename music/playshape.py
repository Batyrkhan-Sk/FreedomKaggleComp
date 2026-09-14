"""Play-shape features: the DISTRIBUTION of listening fractions, not just its mean.

The model sees `plays`, `full_plays` (sum of fractions) and `mean_f` -- but
nothing about the SHAPE of a user's listening on a track. mean_f = 0.5 is
identical for:
   * a track always listened to halfway  (a bad replay candidate)
   * a track half skipped, half completed (a good one -- they finish it when
     they are in the mood, and the metric pays for completions)

Under a metric that scores the listening FRACTION, those are very different, and
history ranking is where we sit at 78% of oracle with the largest absolute gap.

Six counts/rates over the user's plays of that track: true completions (f>.9),
skips (f<.1), partials, their rates, and the spread of f.
"""
import polars as pl

SHAPE_FEATS = ["n_full", "n_skip", "n_part", "full_rate", "skip_rate", "f_std"]


def shape_features(inter, meta, users, cut, min_secs=0):
    return (
        inter.filter(pl.col("d") < cut)
        .filter(pl.col("user_id").is_in(users.implode()))
        .join(meta, on="item_id", how="left")
        .filter(pl.col("track_duration") > 0)
        .with_columns((pl.col("listened_duration") / pl.col("track_duration"))
                      .clip(0, 1).alias("f"))
        .group_by(["user_id", "item_id"])
        .agg((pl.col("f") > 0.9).sum().alias("n_full"),
             (pl.col("f") < 0.1).sum().alias("n_skip"),
             ((pl.col("f") >= 0.1) & (pl.col("f") <= 0.9)).sum().alias("n_part"),
             pl.col("f").std().fill_null(0.0).alias("f_std"),
             pl.len().alias("_n"))
        .with_columns((pl.col("n_full") / pl.col("_n")).alias("full_rate"),
                      (pl.col("n_skip") / pl.col("_n")).alias("skip_rate"))
        .drop("_n")
    )


def add_shape(C, sh):
    return (C.join(sh, on=["user_id", "item_id"], how="left")
            .with_columns([pl.col(c).fill_null(0.0) for c in SHAPE_FEATS]))
