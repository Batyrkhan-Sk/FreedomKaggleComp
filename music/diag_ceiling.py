import polars as pl, numpy as np
from evaluate import load, truth, TOPK
hist, hold, meta, users = load()      # cut 2025-08-16
t = truth(hold, meta, users)
active = t["user_id"].unique()
NA = len(active)
ideal = (t.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
          .head(TOPK).group_by("user_id").agg(pl.col("frac").sum().alias("s")))
tot = ideal["s"].sum()
print(f"active users {NA}/1500 = {NA/15:.1f}%")
print(f"ORACLE  /1500  = {tot/1500/TOPK:.5f}   <- what evaluate.py's denominator implies")
print(f"ORACLE  /{NA}  = {tot/NA/TOPK:.5f}   <- corrected (host's population is all-active)")
CEIL = tot/NA/TOPK
print(f"\nleaders 0.54775 = {0.54775/CEIL*100:.1f}% of oracle")
print(f"3rd     0.40145 = {0.40145/CEIL*100:.1f}% of oracle")
print(f"ours    0.38351 = {0.38351/CEIL*100:.1f}% of oracle")
print(f"\nto reach 3rd: +{0.40145-0.38351:.5f} LB = +{(0.40145-0.38351)/1500*NA:.5f} on evaluate.py's /1500 scale")

# decomposition: history vs new, on the corrected basis
seen = hist.select("user_id","item_id").unique()
t2 = t.join(seen.with_columns(pl.lit(1).alias("h")), on=["user_id","item_id"], how="left").with_columns(pl.col("h").fill_null(0))
for lbl, sub in [("history", t2.filter(pl.col("h")==1)), ("new", t2.filter(pl.col("h")==0))]:
    o = (sub.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
           .head(TOPK).group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    print(f"{lbl:>8}-only oracle @50 slots = {o['s'].sum()/NA/TOPK:.5f}   total value {sub['frac'].sum():,.0f}")
print(f"   all value {t2['frac'].sum():,.0f}  (history {t2.filter(pl.col('h')==1)['frac'].sum()/t2['frac'].sum()*100:.1f}%)")

# how much test-window value is a recent repeat?
h = hist.with_columns((pl.lit("2025-08-16").str.to_date() - pl.col("d").str.to_date()).dt.total_days().alias("age"))
for w in (7,14,30,60,180):
    r = h.filter(pl.col("age")<=w).select("user_id","item_id").unique().with_columns(pl.lit(1).alias("k"))
    v = t.join(r,on=["user_id","item_id"],how="left").filter(pl.col("k")==1)["frac"].sum()
    print(f"  value from tracks played in last {w:>3}d: {v/t['frac'].sum()*100:5.1f}%")

# ceiling if users must have >= N distinct plays (stricter host selection)
print("\nceiling vs minimum holdout activity (how host might have sampled):")
nd = t.group_by("user_id").agg(pl.len().alias("n"))
for m in (1,3,5,10,20,50):
    keep = nd.filter(pl.col("n")>=m)["user_id"]
    o = ideal.filter(pl.col("user_id").is_in(keep.implode()))
    print(f"  >= {m:>2} distinct tracks: {len(keep):>4} users, oracle {o['s'].sum()/len(keep)/TOPK:.5f}")
