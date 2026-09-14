"""Final task-2 stack: FEATS + covis@90d coverage (2) + multi-statistic covis (9),
then a 5-seed rank-averaged ensemble on top.

Individually measured: cv90 +0.00066, multi-covis +0.00044, ensemble +0.00127.
Stacking correlated small positives is not additive by default -- hence the
control arm and the paired seeds."""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

M9 = ["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last",
      "cv_max_rank","cv_wsum_rank","cv_top3_rank"]
W2 = ["cv90","cv90_rank"]
inter, meta, artists, genres, users = load_all()
def merge(name):
    a = pl.read_parquet(f"cache_cv90/{name}.parquet")
    b = pl.read_parquet(f"cache_cv2/{name}.parquet").select(M9)
    return a.hstack(b)
tr = pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS])
C  = merge("apply")
t  = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA = t["user_id"].n_unique()
npl= t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
ytr = tr["y"].to_numpy(); base = C.select("user_id","item_id")
ALL = FEATS + W2 + M9
print(f"train {tr.height:,}   stacked features {len(ALL)}", flush=True)

def sc(p):
    D=base.with_columns(pl.Series("p",p))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA

def go(feats, lbl, seeds):
    X=tr.select(feats).to_numpy().astype(np.float32); Xa=C.select(feats).to_numpy().astype(np.float32)
    ps=[]
    for s in seeds:
        m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,max_leaf_nodes=63,
          min_samples_leaf=200,l2_regularization=1.0,random_state=s,early_stopping=True,
          validation_fraction=0.1,n_iter_no_change=40).fit(X,ytr)
        ps.append(m.predict(Xa))
    v=np.array([sc(p) for p in ps])
    print(f"  {lbl:<34} mean {v.mean():.5f}  sd {v.std(ddof=1):.5f}", flush=True)
    return v.mean(), ps

a,_  = go(FEATS, "A 35 feats [CTRL, expect .37889]", (0,1,2))
d,ps = go(ALL,   "D 46 stacked feats", (0,1,2,3,4))
R = np.vstack([pl.Series(p).rank().to_numpy() for p in ps])
e = sc(R.mean(axis=0))
print(f"  {'E D + ensemble-5 rank-average':<34} {e:.5f}")
print(f"\n  stack   D-A = {d-a:+.5f}")
print(f"  +ens    E-A = {e-a:+.5f}   <- total offline gain over the shipped config")
print(f"  (shipped submission scored LB 0.38351 from an offline draw of 0.37698)")
