"""Headroom is quoted at 50 slots, but we only give ~17 to new tracks.
What is each block's oracle at the slot count it ACTUALLY gets?"""
import polars as pl, numpy as np
from evaluate import TOPK, truth
from train4 import load_all
inter, meta, artists, genres, users = load_all()
APPLY="2025-08-16"
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
C = pl.read_parquet("cache_fix/apply.parquet", columns=["user_id","item_id","is_hist"]).with_columns(
    pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
del inter
tt = t.join(C, on=["user_id","item_id"], how="inner")
def orc(df,k):
    o=(df.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
        .head(k).group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA
h=tt.filter(pl.col("is_hist")==1); n=tt.filter(pl.col("is_hist")==0)
print(f"{'k':>4}{'hist oracle':>14}{'new oracle':>13}")
for k in (5,10,17,25,33,40,50):
    print(f"{k:>4}{orc(h,k):>14.5f}{orc(n,k):>13.5f}")
print(f"\nwe currently get:  history 0.313 (at ~33 slots)   new 0.064 (at ~17 slots)")
print(f"headroom at ACTUAL slots: history {orc(h,33)-0.313:+.3f}   new {orc(n,17)-0.064:+.3f}")
