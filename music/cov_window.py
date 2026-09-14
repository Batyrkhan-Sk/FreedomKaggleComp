"""Would a longer covis window actually reach the 313 uncovered active users?"""
import polars as pl
from evaluate import truth
from train4 import load_all
inter, meta, artists, genres, users = load_all()
APPLY="2025-08-16"
cut_dt = pl.lit(APPLY).str.to_date()
h = (inter.filter(pl.col("d")<APPLY)
     .filter(pl.col("user_id").is_in(users.implode()))
     .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age")))
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
act = set(t["user_id"].unique().to_list())
del inter
print(f"{'window':>8}{'users w/ qualifying plays':>28}{'of the 1153 active':>22}")
for w in (30, 60, 90, 180, 365):
    u = set(h.filter((pl.col("age")<=w) & (pl.col("listened_duration")>=30))["user_id"].unique().to_list())
    print(f"{w:>8}{len(u):>28}{len(u & act):>22}")
# and with a laxer min_secs, at 30d
for ms in (10, 5, 0):
    u = set(h.filter((pl.col("age")<=30) & (pl.col("listened_duration")>=ms))["user_id"].unique().to_list())
    print(f"  30d, min_secs={ms:>3}: {len(u):>5} users, {len(u & act):>5} active")
