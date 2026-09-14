"""Is the holdout window representative of the TEST window?

Every config was tuned on Aug16-30 and applied to Sep1-15. If the repeat/new mix
or the per-user activity level drifts over time, the holdout-optimal balance is
not the test-optimal one -- and no amount of model tuning fixes that.

For each 15-day window: the C-weighted share of value that comes from tracks the
user had ALREADY played before that window, plus activity levels."""
import polars as pl, numpy as np, datetime
from evaluate import TOPK
inter = pl.read_csv("interactions.csv", columns=["user_id","item_id","listened_duration","listened_datetime"]) \
    .with_columns(pl.col("listened_datetime").str.slice(0,10).alias("d"))
meta = pl.read_csv("item_metadata.csv", columns=["item_id","track_duration"])
users = pl.read_csv("test.csv")["user_id"]
inter = inter.filter(pl.col("user_id").is_in(users.implode()))
print(f"{'window':<24}{'active':>8}{'n_play med':>12}{'HIST share':>12}{'new share':>11}{'oracle C':>10}")
start = datetime.date(2025,4,1)
while start <= datetime.date(2025,8,16):
    lo, hi = str(start), str(start + datetime.timedelta(days=15))
    fut = inter.filter((pl.col("d")>=lo)&(pl.col("d")<hi))
    if fut.height == 0: start += datetime.timedelta(days=15); continue
    t = (fut.group_by(["user_id","item_id"]).agg(pl.col("listened_duration").sum().alias("s"))
         .join(meta,on="item_id",how="left").filter(pl.col("track_duration")>0)
         .with_columns(((pl.col("s")/pl.col("track_duration")).clip(0,1)*4).round().truediv(4).alias("frac"))
         .select("user_id","item_id","frac"))
    NA = t["user_id"].n_unique()
    npl = t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
    seen = inter.filter(pl.col("d")<lo).select("user_id","item_id").unique().with_columns(pl.lit(1).alias("h"))
    tw = (t.join(npl,on="user_id").join(seen,on=["user_id","item_id"],how="left")
          .with_columns(pl.col("h").fill_null(0), (pl.col("frac")/pl.col("den")).alias("w")))
    tot = tw["w"].sum()
    hist_share = tw.filter(pl.col("h")==1)["w"].sum()/tot*100
    orc = (t.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
           .head(TOPK).group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    o = npl.join(orc,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    print(f"{lo+'..'+hi:<24}{NA:>8}{int(npl['n'].median()):>12}{hist_share:>11.1f}%{100-hist_share:>10.1f}%"
          f"{(o['s']/o['den']).sum()/NA:>10.4f}", flush=True)
    start += datetime.timedelta(days=15)
