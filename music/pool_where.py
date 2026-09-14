"""Pool costs 0.127 of C. Where does the unreachable value live?"""
import polars as pl
from evaluate import TOPK, truth
from train4 import load_all
from train12 import cached
inter, meta, artists, genres, users = load_all()
APPLY="2025-08-16"
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
tw = t.join(npl,on="user_id").with_columns((pl.col("frac")/pl.col("den")/NA).alias("w"))
TOT = tw["w"].sum()
pool = cached("apply",None).select("user_id","item_id").with_columns(
    pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
miss = tw.join(pool,on=["user_id","item_id"],how="anti")
print(f"total C value {TOT:.4f}  (== oracle {0.81964:.4f})")
print(f"OUT of pool   {miss['w'].sum():.4f} = {miss['w'].sum()/TOT*100:.1f}%  "
      f"({miss.height:,} pairs)")

# is the missed value by an artist the user already played?
print(f"\nartists table: {artists.columns}  rows {artists.height:,}")
hist = inter.filter(pl.col("d")<APPLY).select("user_id","item_id").unique()
ua = (hist.join(artists,on="item_id",how="inner").select("user_id","artist_name").unique())
print(f"user-artist pairs from history: {ua.height:,} ({ua.height/1500:.0f}/user)")
m2 = miss.join(artists,on="item_id",how="left")
known = m2.join(ua,on=["user_id","artist_name"],how="semi")
print(f"\nof the OUT-of-pool value:")
print(f"   by an artist the user ALREADY played: {known['w'].sum()/miss['w'].sum()*100:.1f}%  "
      f"(= {known['w'].sum():.4f} of total C, {known['w'].sum()/TOT*100:.1f}%)")
print(f"   distinct such (user,item) pairs to add: {known.height:,} = {known.height/1500:.0f}/user")
# how big would the pool get if we added every track by every artist the user has played?
allc = ua.join(artists,on="artist_name",how="inner").select("user_id","item_id").unique()
print("done")
