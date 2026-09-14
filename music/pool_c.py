"""Under metric C: how much of the ceiling does our CANDIDATE POOL allow?"""
import polars as pl
from evaluate import TOPK, truth
from train4 import load_all
from train12 import cached
inter, meta, artists, genres, users = load_all()
APPLY="2025-08-16"
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
C = cached("apply", None)
pool = C.select("user_id","item_id","is_hist").with_columns(pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
print(f"pool rows {pool.height:,}  = {pool.height/1500:.0f} cands/user "
      f"(hist {pool.filter(pl.col('is_hist')==1).height/1500:.0f}, new {pool.filter(pl.col('is_hist')==0).height/1500:.0f})")

def cscore(df):
    o=(df.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
        .head(TOPK).group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA

print(f"\nORACLE, unrestricted          {cscore(t):.5f}")
inp = t.join(pool,on=["user_id","item_id"],how="inner")
print(f"ORACLE, restricted to our pool {cscore(inp):.5f}   <- pool ceiling")
print(f"OUR MODEL                      0.37698")
print(f"\n  cost of the POOL     {cscore(t)-cscore(inp):.5f}")
print(f"  cost of the RANKING  {cscore(inp)-0.37698:.5f}")
h=inp.filter(pl.col("is_hist")==1); n=inp.filter(pl.col("is_hist")==0)
print(f"\n  in-pool oracle, history only {cscore(h):.5f}")
print(f"  in-pool oracle, new only     {cscore(n):.5f}")
# per-user recall of value, C-weighted
tv=t.join(npl,on="user_id").with_columns((pl.col("frac")/pl.col("den")).alias("w"))
iv=inp.join(npl,on="user_id").with_columns((pl.col("frac")/pl.col("den")).alias("w"))
print(f"\n  C-weighted value recall of the pool: {iv['w'].sum()/tv['w'].sum()*100:.1f}%")
hh=tv.join(pool.filter(pl.col("is_hist")==1),on=["user_id","item_id"],how="semi")
print(f"    of which reachable via HISTORY: {hh['w'].sum()/tv['w'].sum()*100:.1f}%")
