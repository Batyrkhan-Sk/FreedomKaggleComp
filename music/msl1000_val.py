"""SHIP DECISION: final_val.py with min_samples_leaf=1000 instead of the inherited 200.
Control is final_val.log itself: members 0.37916 sd 0.00072, ENSEMBLE-5 0.38090.
At 200 iters x 3 seeds msl=1000 was +0.00114 members, +0.00107 ens3, and the curve
turned over by 3000, so the optimum is interior. This checks it survives at the
shipped capacity, where coarser leaves may want a different iteration count.
Reference points on this same holdout:
   shipped config draw            0.37698  (-> LB 0.38351)
   fixed-covis baseline, 3 seeds  0.37889
   stack + ensemble @200 iters    0.38066
"""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
M9=["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
W2=["cv90","cv90_rank"]; ALL=FEATS+W2+M9
inter, meta, artists, genres, users = load_all()
def merge(n):
    return pl.read_parquet(f"cache_cv90/{n}.parquet").hstack(pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9))
tr=pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS]); C=merge("apply")
t=truth(inter.filter(pl.col("d")>=APPLY), meta, users); NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
X=tr.select(ALL).to_numpy().astype(np.float32); y=tr["y"].to_numpy()
Xa=C.select(ALL).to_numpy().astype(np.float32); base=C.select("user_id","item_id")
def sc(p):
    D=base.with_columns(pl.Series("p",p))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA
ps=[]
for s in range(5):
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.03,max_leaf_nodes=63,
      min_samples_leaf=1000,l2_regularization=1.0,random_state=s,early_stopping=True,
      validation_fraction=0.1,n_iter_no_change=40).fit(X,y)
    ps.append(m.predict(Xa)); print(f"  seed {s}: {sc(ps[-1]):.5f}  ({m.n_iter_} iters)", flush=True)
v=np.array([sc(p) for p in ps])
R=np.vstack([pl.Series(p).rank().to_numpy() for p in ps])
e=sc(R.mean(axis=0))
print(f"\n  members mean {v.mean():.5f} sd {v.std(ddof=1):.5f}")
print(f"  ENSEMBLE-5   {e:.5f}")
print(f"  vs stack+ens @200 iters 0.38066 : {e-0.38066:+.5f}")
print(f"  vs fixed baseline       0.37889 : {e-0.37889:+.5f}")
print(f"  vs shipped draw         0.37698 : {e-0.37698:+.5f}")
