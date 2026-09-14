"""Is there a replay-TIMING signal the current features miss?

Current pair features encode recency (days since last play), n_days, span --
all "how much / how recently". None encode the user's own CADENCE for that
track: if someone plays a track every 3 days and it has been 4 days, that is a
different state from a 3-day gap on a track they play every 60 days.

Test: among history pairs, does gap_ratio = recency / mean_inter_play_gap
separate replayers from non-replayers ONCE recency and play count are
controlled for? If it carries nothing beyond recency, do not build it.
"""
import polars as pl
from evaluate import CUT, load, truth

train, hold, meta, users = load(CUT)
t = truth(hold, meta, users)
print(f"holdout truth rows {t.height:,}", flush=True)

base = (train.filter(pl.col("user_id").is_in(users.implode()))
        .with_columns((pl.lit(CUT).str.to_date() - pl.col("d").str.to_date())
                      .dt.total_days().alias("age")))

pair = (base.group_by(["user_id", "item_id"])
        .agg(pl.len().alias("plays"),
             pl.col("age").min().alias("recency"),
             pl.col("age").max().alias("first_age"),
             pl.col("d").n_unique().alias("n_days"))
        .filter(pl.col("n_days") >= 3)            # need >=3 distinct days for a cadence
        .with_columns(((pl.col("first_age") - pl.col("recency"))
                       / (pl.col("n_days") - 1)).alias("cadence"))
        .with_columns((pl.col("recency") / pl.col("cadence")).alias("gap_ratio")))

p = (pair.join(t, on=["user_id", "item_id"], how="left")
     .with_columns(pl.col("frac").fill_null(0.0)))
print(f"history pairs with cadence: {p.height:,}  replay rate {(p['frac']>0).mean():.3f}", flush=True)

# Control for recency: within each recency bucket, does gap_ratio still separate?
p = p.with_columns(
    pl.when(pl.col("recency") <= 1).then(pl.lit("0-1d"))
     .when(pl.col("recency") <= 3).then(pl.lit("2-3d"))
     .when(pl.col("recency") <= 7).then(pl.lit("4-7d"))
     .when(pl.col("recency") <= 21).then(pl.lit("8-21d"))
     .otherwise(pl.lit("22d+")).alias("rbin"),
    pl.when(pl.col("gap_ratio") < 0.5).then(pl.lit("1 early (<0.5x)"))
     .when(pl.col("gap_ratio") < 1.0).then(pl.lit("2 due (0.5-1x)"))
     .when(pl.col("gap_ratio") < 2.0).then(pl.lit("3 overdue (1-2x)"))
     .otherwise(pl.lit("4 stale (>2x)")).alias("gbin"))

g = (p.group_by(["rbin", "gbin"])
     .agg(pl.len().alias("n"), (pl.col("frac") > 0).mean().alias("replay"),
          pl.col("frac").mean().alias("pts"))
     .sort(["rbin", "gbin"]))
print("\nreplay rate by (days since last play) x (gap vs this pair's own cadence)")
print("if the columns are flat within a row, gap_ratio adds nothing over recency\n")
with pl.Config(tbl_rows=40, tbl_width_chars=140):
    print(g.pivot(on="gbin", index="rbin", values="replay", aggregate_function="first"))
    print("\ncounts:")
    print(g.pivot(on="gbin", index="rbin", values="n", aggregate_function="first"))
