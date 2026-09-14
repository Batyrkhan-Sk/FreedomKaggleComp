"""Absolute oracle: best possible 50 tracks per user, chosen with full knowledge
of the holdout. Nothing can beat this under the assumed metric."""
import polars as pl
from evaluate import load, truth, TOPK

hist, hold, meta, users = load()
t = truth(hold, meta, users)
print(f"truth rows {t.height:,}, users with any truth {t['user_id'].n_unique()}/{len(users)}")

# per-user: sum of top-50 fracs
per = (t.sort(["user_id", "frac"], descending=[False, True])
        .group_by("user_id", maintain_order=True).head(TOPK)
        .group_by("user_id").agg(pl.col("frac").sum().alias("s"),
                                 pl.len().alias("n")))
tot = per["s"].sum()
print(f"ABSOLUTE ORACLE = {tot/len(users)/TOPK:.5f}   (mean {tot/len(users):.2f}/50 over all {len(users)} users)")

# how many distinct tracks does each user actually play in the window?
nd = t.group_by("user_id").agg(pl.len().alias("n_distinct"),
                               pl.col("frac").sum().alias("all_frac"))
import numpy as np
a = nd["n_distinct"].to_numpy()
print("distinct tracks played in holdout, over ACTIVE users:")
for q in [10,25,50,75,90,95,99]:
    print(f"   p{q}: {np.percentile(a,q):.0f}")
print(f"   mean {a.mean():.1f}   share with >=50 distinct: {(a>=50).mean()*100:.1f}%")

# ceiling if metric normalised per user by min(50, n_distinct) instead of 50
per2 = per.with_columns((pl.col("s")/pl.min_horizontal(pl.lit(TOPK), pl.col("n"))).alias("r"))
print(f"if normalised by min(50,n_played): oracle = {per2['r'].sum()/len(users):.5f}")
print(f"   and over ACTIVE users only     = {per2['r'].mean():.5f}")

# what does the current shipped submission-equivalent get?  (score history-only oracle)
