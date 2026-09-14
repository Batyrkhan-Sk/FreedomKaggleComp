"""Is TRACK LENGTH a lever on the metric itself?

score = clip(listened_duration / track_duration, 0, 1), quantised. A 30-second
track hits 1.0 from 30 seconds of listening; a 5-minute track hits 0.1 from the
same 30 seconds. So for equal play probability, SHORT tracks pay more.

track_duration is a feature, but the model regresses predicted y -- it is never
told that duration is a lever on the PAYOUT rather than just an item property.
If short tracks systematically earn a higher fraction when played, a deliberate
short-track bias raises expected score at no cost in play probability.

This is the shape of thing that produces a leaderboard CLIFF: a property of the
metric, not a better model.

Measures, on the holdout: mean listening fraction by duration bucket, and what
the model's chosen 50 look like versus what the value actually sits on.
"""
import numpy as np
import polars as pl

from evaluate import CUT, load, truth

train, hold, meta, users = load(CUT)
t = truth(hold, meta, users)
md = pl.read_csv("item_metadata.csv", columns=["item_id", "track_duration"])
t = t.join(md, on="item_id", how="left").filter(pl.col("track_duration") > 0)

B = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 10**9)]
print(f"holdout pairs {t.height:,}\n")
print(f"{'duration':>14} {'pairs':>9} {'mean frac':>10} {'sum pts':>9} {'pts/pair':>9}")
for lo, hi in B:
    b = t.filter((pl.col("track_duration") >= lo) & (pl.col("track_duration") < hi))
    if not b.height:
        continue
    lbl = f"{lo}-{hi}s" if hi < 10**9 else f"{lo}s+"
    print(f"{lbl:>14} {b.height:9,} {b['frac'].mean():10.4f} "
          f"{b['frac'].sum():9,.0f} {b['frac'].sum()/b.height:9.4f}")

# is the effect real, or just that short tracks are played more often?
played = t.filter(pl.col("frac") > 0)
print(f"\nAMONG PLAYED PAIRS ONLY (controls for play probability):")
print(f"{'duration':>14} {'pairs':>9} {'mean frac':>10}")
for lo, hi in B:
    b = played.filter((pl.col("track_duration") >= lo) & (pl.col("track_duration") < hi))
    if not b.height:
        continue
    lbl = f"{lo}-{hi}s" if hi < 10**9 else f"{lo}s+"
    print(f"{lbl:>14} {b.height:9,} {b['frac'].mean():10.4f}")

# what does the CANDIDATE POOL look like by duration vs where the value is?
cut_dt = pl.lit(CUT).str.to_date()
recent = (train.with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
          .filter(pl.col("age") <= 14).join(md, on="item_id", how="left")
          .filter(pl.col("track_duration") > 0)
          .with_columns((pl.col("listened_duration")/pl.col("track_duration")).clip(0,1).alias("f")))
pool = (recent.group_by("item_id").agg(pl.col("f").sum().alias("s"))
        .sort("s", descending=True).head(1500).join(md, on="item_id", how="left"))
print(f"\nmedian duration -- pool {pool['track_duration'].median():.0f}s   "
      f"all holdout value-weighted {t['track_duration'].median():.0f}s")
short = t.filter(pl.col("track_duration") < 120)
print(f"tracks <120s hold {short['frac'].sum():,.0f} of {t['frac'].sum():,.0f} pts "
      f"({short['frac'].sum()/t['frac'].sum():.1%}) from {short.height/t.height:.1%} of pairs")
