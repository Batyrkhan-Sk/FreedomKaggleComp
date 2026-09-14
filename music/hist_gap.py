"""The biggest block: history ranking. Where exactly does it lose?"""
import polars as pl, numpy as np
from evaluate import TOPK, truth
from train4 import load_all
inter, meta, artists, genres, users = load_all()
APPLY="2025-08-16"
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
D = pl.read_parquet("scored_apply.parquet").filter(pl.col("is_hist")==1)
print(f"history candidates: {D.height/1500:.0f}/user  (min {D.group_by('user_id').len()['len'].min()}, "
      f"max {D.group_by('user_id').len()['len'].max()})")

# does every played-history track even APPEAR as a candidate?
seen_pre = inter.filter(pl.col("d")<APPLY).select("user_id","item_id").unique()
th = t.join(seen_pre,on=["user_id","item_id"],how="semi")   # truth that IS history
inpool = th.join(D.select("user_id","item_id"),on=["user_id","item_id"],how="semi")
print(f"history truth pairs {th.height:,}; present as candidates {inpool.height:,} "
      f"({inpool.height/th.height*100:.1f}%)  value {inpool['frac'].sum()/th['frac'].sum()*100:.1f}%")

# rank of played tracks within the user's history candidates
R = D.with_columns(pl.col("p").rank("min",descending=True).over("user_id").alias("r"))
hit = th.join(R.select("user_id","item_id","r"),on=["user_id","item_id"],how="inner")
hv = hit.join(npl,on="user_id").with_columns((pl.col("frac")/pl.col("den")).alias("w"))
tot = th.join(npl,on="user_id").with_columns((pl.col("frac")/pl.col("den")).alias("w"))["w"].sum()
print(f"\nC-weighted history value by our rank within history candidates (total {tot:.1f}):")
cum=0
for k in (10,20,30,34,50,80,120,200,10**9):
    v=hv.filter(pl.col("r")<=k)["w"].sum()
    print(f"   top {str(k) if k<10**9 else 'all':>4}: {v/tot*100:5.1f}% cumulative")

# the oracle's view: how deep would you have to go with PERFECT ranking?
print("\nfor comparison, perfect ranking captures 100% of it in the top-n where n = the")
print("number of played history tracks:")
nh = th.group_by("user_id").len()["len"]
print(f"   played history tracks/user: median {nh.median():.0f}, p75 {nh.quantile(.75):.0f}, p90 {nh.quantile(.9):.0f}")

# is the loss "wrong track" or "right track wrong frac"?
top50h = R.filter(pl.col("r")<=TOPK)
picked = top50h.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
print(f"\nof our 50 history picks/user: {picked.filter(pl.col('frac')>0).height/NA:.1f} are played, "
      f"{picked.filter(pl.col('frac')==0).height/NA:.1f} are dead")
print(f"   mean frac of the played ones: {picked.filter(pl.col('frac')>0)['frac'].mean():.3f}")
