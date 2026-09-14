"""We tune on the 1,153 users active in Aug16-30. The TEST scores 1,500 -- all
active by construction -- so ~350 users who are invisible to our holdout are
scored for real. The closest proxy for them: users active in the holdout who
were INACTIVE in the preceding window ("returning" users).

If returning users score materially worse, that is a real segment we have never
optimised for, and it is over-represented in the test set relative to our holdout.
"""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
M9=["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
W2=["cv90","cv90_rank"]; STACK=FEATS+W2+M9
inter, meta, artists, genres, users = load_all()
def merge(n): return (pl.read_parquet(f"cache_cv90/{n}.parquet")
                      .hstack(pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9)))
tr=pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS]); C=merge("apply")
t=truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
# who was active in the PRECEDING window (Aug 1-16)?
prev=set(inter.filter((pl.col("d")>="2025-08-01")&(pl.col("d")<APPLY))["user_id"].unique().to_list())
del inter
m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,max_leaf_nodes=63,
  min_samples_leaf=200,l2_regularization=1.0,random_state=0,early_stopping=True,
  validation_fraction=0.1,n_iter_no_change=40).fit(
      tr.select(STACK).to_numpy().astype(np.float32), tr["y"].to_numpy())
D=(C.with_columns(pl.Series("p",m.predict(C.select(STACK).to_numpy().astype(np.float32))))
   .with_columns(pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)))
top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
   .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
ideal=(t.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
       .head(TOPK).group_by("user_id").agg(pl.col("frac").sum().alias("i")))
per=(npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
     .join(ideal,on="user_id",how="left")
     .with_columns((pl.col("s")/pl.col("den")).alias("c"), (pl.col("i")/pl.col("den")).alias("cmax"),
                   pl.col("user_id").is_in(pl.Series(list(prev)).implode()).alias("was_active")))
print(f"overall C = {per['c'].mean():.5f}\n")
print(f"{'segment':<26}{'users':>7}{'ourC':>9}{'oracleC':>9}{'%orac':>8}{'n_play med':>12}")
for lbl, q in (("active in prev window", per.filter(pl.col("was_active"))),
               ("RETURNING (was absent)", per.filter(~pl.col("was_active")))):
    print(f"{lbl:<26}{q.height:>7}{q['c'].mean():>9.4f}{q['cmax'].mean():>9.4f}"
          f"{q['c'].mean()/q['cmax'].mean()*100:>7.1f}%{int(q['n'].median()):>12}")
