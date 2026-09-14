"""Premise test: do max/top3/wsum rank new-track value better than the sum,
and are they decorrelated enough from it to add anything?"""
import numpy as np, polars as pl
from evaluate import TOPK, truth
from train4 import load_all
from rank4 import build_candidates
from train7 import APPLY, N_POOL
from covis2 import covis_multi

inter, meta, artists, genres, users = load_all()
C = build_candidates(inter, meta, artists, genres, users, APPLY, n_pool=N_POOL)
pool = C.filter(pl.col("is_hist")==0)["item_id"].unique().to_list()
cv = covis_multi(inter, users, pool, APPLY, prefix="cv")
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA = t["user_id"].n_unique()
npl = t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter

new = (C.filter(pl.col("is_hist")==0).select("user_id","item_id")
        .with_columns(pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
        .join(cv, on=["user_id","item_id"], how="left")
        .with_columns([pl.col(c).fill_null(0.0) for c in cv.columns if c not in ("user_id","item_id")])
        .join(t, on=["user_id","item_id"], how="left").with_columns(pl.col("frac").fill_null(0.0))
        .join(npl.select("user_id","den"), on="user_id", how="left")
        .with_columns(pl.col("den").fill_null(TOPK)))
print(f"new candidate rows {new.height:,}, with value {new.filter(pl.col('frac')>0).height:,}")

stats = ["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last"]
print("\ncorrelation with cv_sum:")
s0 = new["cv_sum"].to_numpy()
for s in stats[1:]:
    print(f"   {s:>9}: r = {np.corrcoef(s0, new[s].to_numpy())[0,1]:.4f}")

print("\nC-weighted new-track value captured in the top-17 per user by each statistic:")
for s in stats:
    top=(new.sort(["user_id",s],descending=[False,True]).group_by("user_id",maintain_order=True).head(17))
    v=(top.group_by("user_id").agg((pl.col("frac")/pl.col("den")).sum().alias("c"))["c"].sum()/NA)
    print(f"   {s:>9}: {v:.5f}")
# the model's own new-side capture at 17 slots is 0.064; oracle 0.195
print("\n   (model gets 0.064 at 17 new slots; in-pool oracle 0.195)")
