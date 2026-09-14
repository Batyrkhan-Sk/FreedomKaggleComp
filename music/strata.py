"""Under metric C every user is worth the same. Where is the loss?"""
import polars as pl, numpy as np
from evaluate import TOPK, truth
from train4 import load_all
inter, meta, artists, genres, users = load_all()
APPLY="2025-08-16"
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
top = pl.read_parquet("holdout_top50.parquet")
npl = t.group_by("user_id").agg(pl.len().alias("n"))
ideal=(t.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
        .head(TOPK).group_by("user_id").agg(pl.col("frac").sum().alias("ideal")))
got=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
       .group_by("user_id").agg(pl.col("frac").sum().alias("s"),
                                (pl.col("frac")*(pl.col("is_hist")==1)).sum().alias("sh"),
                                (pl.col("frac")*(pl.col("is_hist")==0)).sum().alias("sn")))
p=(npl.join(got,on="user_id",how="left").with_columns(pl.col("s","sh","sn").fill_null(0.0)).join(ideal,on="user_id",how="left")
     .with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
     .with_columns((pl.col("s")/pl.col("den")).alias("c"),
                   (pl.col("ideal")/pl.col("den")).alias("cmax"),
                   (pl.col("sh")/pl.col("den")).alias("ch"),
                   (pl.col("sn")/pl.col("den")).alias("cn")))
NA=p.height
print(f"active {NA}   score C = {p['c'].mean():.5f}   oracle C = {p['cmax'].mean():.5f}")
print(f"\n{'n_played':<12}{'users':>7}{'ourC':>9}{'oracleC':>9}{'%orac':>8}{'fromHist':>10}{'fromNew':>9}{'LOSS/user':>11}{'tot loss':>10}")
bins=[(1,2),(3,5),(6,10),(11,20),(21,50),(51,120),(121,10**9)]
for lo,hi in bins:
    q=p.filter((pl.col("n")>=lo)&(pl.col("n")<=hi))
    loss=(q["cmax"]-q["c"]).mean(); tot=(q["cmax"]-q["c"]).sum()/NA
    lbl=f"{lo}-{hi if hi<10**9 else '+'}"
    print(f"{lbl:<12}{q.height:>7}{q['c'].mean():>9.4f}{q['cmax'].mean():>9.4f}"
          f"{q['c'].mean()/q['cmax'].mean()*100:>7.0f}%{q['ch'].mean():>10.4f}{q['cn'].mean():>9.4f}"
          f"{loss:>11.4f}{tot:>10.4f}")
print(f"\ntotal headroom to oracle = {(p['cmax']-p['c']).mean():.4f}")
# what if we just add the user's most-recent unheard-in-window tracks?  where do slots go?
print(f"\nslot usage: hist slots {top.filter(pl.col('is_hist')==1).height/NA:.1f}/50, "
      f"new {top.filter(pl.col('is_hist')==0).height/NA:.1f}/50")
