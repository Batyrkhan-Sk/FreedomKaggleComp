"""Does this user play THIS track on a rhythm, and is it due?

The pair features encode how much (`plays`, `full_plays`, `secs`) and how
recently (`recency`), but nothing encodes the user's own cadence for the track.
Playing something 3 days after a 3-day habit is a different state from 3 days
after a 60-day habit, and the model currently cannot tell them apart.

Measured on the holdout (periodicity.py, 75,294 history pairs with >=3 distinct
play days): within a FIXED recency band the replay rate moves 15-33 points with
the gap ratio -- 0-1d early 58% vs due 91%, 8-21d early 18% vs stale 33%. That is
not flat, so the signal is real and unexploited.

Everything here is derived from columns the candidate frame already has. The
point is not new information -- it is that a gradient-boosted tree splits on one
axis at a time and cannot form `recency / (span / (n_days - 1))` by itself.
Handing a tree the interaction it cannot build is the whole trick.
"""
import polars as pl

CADENCE_FEATS = ["cadence", "gap_ratio", "overdue", "log_gap_ratio"]


def add_cadence(C):
    """Add cadence features to a candidate frame from rank4.build_candidates."""
    return C.with_columns(
        # average days between distinct listening days for this (user, track)
        (pl.col("span") / pl.max_horizontal(pl.col("n_days") - 1, pl.lit(1)))
        .alias("cadence")
    ).with_columns(
        # how many of this pair's own cycles have elapsed since the last play.
        # New tracks have span 0 and n_days 0, so cadence is 0 -- guard the
        # division rather than letting it become inf and poison the split finder.
        (pl.col("recency") / pl.max_horizontal(pl.col("cadence"), pl.lit(0.5)))
        .alias("gap_ratio"),
        (pl.col("recency") - pl.col("cadence")).alias("overdue"),
    ).with_columns(
        # the raw ratio is heavy-tailed (stale tracks reach 100x); the log keeps
        # the informative 0-3x range from being squeezed into one leaf
        (pl.col("gap_ratio") + 1.0).log().alias("log_gap_ratio")
    )
