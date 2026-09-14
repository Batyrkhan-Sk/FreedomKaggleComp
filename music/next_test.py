"""Two untried axes on the shipped 46-feature stack, paired against a control.

  1. RAW target -- the model currently trains on the quantised fraction
     {0,.25,.5,.75,1}. Quantising before training discards ranking information:
     a 13% listen and a 37% listen both become 0.25. Scoring is unchanged.
  2. ALS as an EXTRA feature. Recorded at 2.09x the SVD block's strength but
     only ever measured standalone -- never added to the model.
"""
import sys, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

M9=["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
W2=["cv90","cv90_rank"]; STACK=FEATS+W2+M9
import pathlib
HAVE_ALS = pathlib.Path("cache_als/apply.parquet").exists()
HAVE_RAW = pathlib.Path("cache_rawy/tr_2025-07-01.parquet").exists()
inter, meta, artists, genres, users = load_all()
def merge(n):
    a=pl.read_parquet(f"cache_cv90/{n}.parquet").hstack(
      pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9))
    if HAVE_ALS: a=a.hstack(pl.read_parquet(f"cache_als/{n}.parquet"))
    return a
tr=pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS]); C=merge("apply")
yq=tr["y"].to_numpy()
yr=None
if HAVE_RAW:
    yr=pl.concat([pl.read_parquet(f"cache_rawy/tr_{c}.parquet") for c,_ in WINDOWS])["y_raw"].to_numpy()
t=truth(inter.filter(pl.col("d")>=APPLY), meta, users); NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
base=C.select("user_id","item_id")
print(f"train {tr.height:,}  ALS={HAVE_ALS}  RAW={HAVE_RAW}", flush=True)
def sc(p):
    D=base.with_columns(pl.Series("p",p))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA
def go(feats,yy,lbl,seeds=(0,1,2)):
    X=tr.select(feats).to_numpy().astype(np.float32); Xa=C.select(feats).to_numpy().astype(np.float32)
    v=[]
    for s in seeds:
        m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,max_leaf_nodes=63,
          min_samples_leaf=200,l2_regularization=1.0,random_state=s,early_stopping=True,
          validation_fraction=0.1,n_iter_no_change=40).fit(X,yy)
        v.append(sc(m.predict(Xa)))
    v=np.array(v); print(f"  {lbl:<40} mean {v.mean():.5f} sd {v.std(ddof=1):.5f}", flush=True)
    return v.mean()
ctrl = go(STACK, yq, "CTRL stack + quantised y")
if HAVE_RAW: r = go(STACK, yr, "RAW target"); print(f"    raw-target delta {r-ctrl:+.5f}", flush=True)
if HAVE_ALS:
    a = go(STACK+["als","als_rank"], yq, "stack + ALS feature")
    print(f"    ALS delta {a-ctrl:+.5f}", flush=True)
    if HAVE_RAW:
        b = go(STACK+["als","als_rank"], yr, "stack + ALS + RAW target")
        print(f"    both delta {b-ctrl:+.5f}", flush=True)
