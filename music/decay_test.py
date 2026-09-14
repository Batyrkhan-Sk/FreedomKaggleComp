"""Does multi-scale decay add over the shipped stack?
Screened at 200 iters (paired, so the delta transfers); winner confirmed at 400+ensemble.
cache_fix baseline at 200 iters = 0.37889 (3 seeds, sd 0.00103)."""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

M9=["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
W2=["cv90","cv90_rank"]
D4=["decay_plays_7","decay_full_7","decay_plays_60","decay_full_60"]
inter, meta, artists, genres, users = load_all()
def merge(n):
    a=pl.read_parquet(f"cache_cv90/{n}.parquet")
    a=a.hstack(pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9))
    return a.hstack(pl.read_parquet(f"cache_decay/{n}.parquet").select(D4))
tr=pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS]); C=merge("apply")
t=truth(inter.filter(pl.col("d")>=APPLY), meta, users); NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
y=tr["y"].to_numpy(); base=C.select("user_id","item_id")
print(f"train {tr.height:,}  cols {tr.width}", flush=True)
def sc(p):
    D=base.with_columns(pl.Series("p",p))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA
def go(feats,lbl,iters=200,seeds=(0,1,2)):
    X=tr.select(feats).to_numpy().astype(np.float32); Xa=C.select(feats).to_numpy().astype(np.float32)
    v=[]
    for s in seeds:
        m=HistGradientBoostingRegressor(max_iter=iters,learning_rate=0.03,max_leaf_nodes=63,
          min_samples_leaf=200,l2_regularization=1.0,random_state=s,early_stopping=True,
          validation_fraction=0.1,n_iter_no_change=40).fit(X,y)
        v.append(sc(m.predict(Xa)))
    v=np.array(v); print(f"  {lbl:<36} mean {v.mean():.5f}  sd {v.std(ddof=1):.5f}", flush=True)
    return v.mean()
a=go(FEATS,               "A FEATS [CTRL, expect .37889]")
b=go(FEATS+D4,            "B FEATS + 4 multi-scale decay")
c=go(FEATS+W2+M9,         "C shipped stack (46 feats)")
d=go(FEATS+W2+M9+D4,      "D shipped stack + decay (50)")
print(f"\n  decay alone   B-A = {b-a:+.5f}")
print(f"  decay on top  D-C = {d-c:+.5f}   (bar ~0.003)")
