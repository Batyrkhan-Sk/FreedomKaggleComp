"""How much is a BETTER POOL worth? The unrestricted new-track oracle.

Arithmetic that forces the question: the leader's 0.54775 = 24,479 pts. Perfect
history ranking is 17,583 (the oracle at 34 slots), leaving 6,896 that MUST come
from new tracks -- but the in-pool oracle at 16 slots is only 8,615. So the
leader needs ~80% of a perfect new-track ranker on top of a perfect history
ranker, and with realistic history the requirement EXCEEDS the in-pool oracle
entirely. The pool, not the ranker, is what caps us.

Every pool experiment on record varies the same GLOBAL list (wider: #8; reweighted:
power-user). Nobody has tried a PER-USER candidate set. This measures the prize:

  in-pool oracle @k   -- the ceiling with today's candidates (already known)
  unrestricted oracle @k -- the ceiling if retrieval were perfect for each user

The gap is what per-user retrieval could buy, and it bounds any two-tower / ALS-
retrieval / covisitation-neighbour work before a line of it is written.
"""
import numpy as np
import polars as pl

from evaluate import CUT, load, truth

train, hold, meta, users = load(CUT)
t = truth(hold, meta, users)
hist = (train.filter(pl.col("user_id").is_in(users.implode()))
        .select("user_id", "item_id").unique())
new = t.join(hist, on=["user_id", "item_id"], how="anti")

cut_dt = pl.lit(CUT).str.to_date()
recent = (train.with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
          .filter(pl.col("age") <= 14).join(meta, on="item_id", how="left")
          .filter(pl.col("track_duration") > 0)
          .with_columns((pl.col("listened_duration") / pl.col("track_duration")).clip(0, 1).alias("f")))

active = t["user_id"].n_unique()
SCALE = 1.29 / (active * 50)
TOTAL_NEW = new["frac"].sum()
print(f"active {active}   new-track value available {TOTAL_NEW:,.0f} pts")
print(f"leader 0.54775 = {0.54775/SCALE:,.0f} pts   you = {16886:,} pts\n")

def oracle(df, k):
    v = df.group_by("user_id").agg(pl.col("frac").sort(descending=True).head(k).sum())
    return v["frac"].sum()

for N in (1500, 5000, 20000, 100000):
    pool = set(recent.group_by("item_id").agg(pl.col("f").sum().alias("s"))
               .sort("s", descending=True).head(N)["item_id"].to_list())
    inp = new.filter(pl.col("item_id").is_in(list(pool)))
    print(f"global top-{N:<6d} holds {inp['frac'].sum():8,.0f} of {TOTAL_NEW:,.0f} "
          f"({inp['frac'].sum()/TOTAL_NEW:5.1%})   oracle@16 {oracle(inp,16):8,.0f}  "
          f"@20 {oracle(inp,20):8,.0f}")

print()
print(f"{'UNRESTRICTED (perfect per-user retrieval)':44s} "
      f"oracle@16 {oracle(new,16):8,.0f}  @20 {oracle(new,20):8,.0f}")
print()
h_or = 17583
for k, label in ((16, "34/16"), (20, "30/20")):
    inp = new.filter(pl.col("item_id").is_in(
        list(set(recent.group_by("item_id").agg(pl.col("f").sum().alias("s"))
                 .sort("s", descending=True).head(1500)["item_id"].to_list()))))
    cur, unr = oracle(inp, k), oracle(new, k)
    print(f"split {label}: perfect history + perfect ranking...")
    print(f"   within today's pool : {(h_or+cur)*SCALE:.4f}")
    print(f"   with perfect retrieval: {(h_or+unr)*SCALE:.4f}")
print(f"\n-> if the in-pool number is BELOW 0.548, the pool alone caps us short of the leader")
