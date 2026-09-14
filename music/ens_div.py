"""Ensembling is the one thing that works here (+0.00127). But we only ever
varied the SEED -- five near-identical models. Ensembles gain from DIVERSITY:
models that err differently cancel. Vary capacity, learning rate and leaf count
instead and the members decorrelate.

Paired against a seed-only ensemble of the same size on the same features, so
the comparison isolates diversity from ensemble size."""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

M9=["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
W2=["cv90","cv90_rank"]; ALL=FEATS+W2+M9
inter, meta, artists, genres, users = load_all()
def merge(n):
    return pl.read_parquet(f"cache_cv90/{n}.parquet").hstack(
           pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9))
tr=pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS]); C=merge("apply")
t=truth(inter.filter(pl.col("d")>=APPLY), meta, users); NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
X=tr.select(ALL).to_numpy().astype(np.float32); y=tr["y"].to_numpy()
Xa=C.select(ALL).to_numpy().astype(np.float32); base=C.select("user_id","item_id")
print(f"train {tr.height:,}  feats {len(ALL)}", flush=True)

def sc(p):
    D=base.with_columns(pl.Series("p",p))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA
def fit(it,lr,lv,msl,seed):
    m=HistGradientBoostingRegressor(max_iter=it,learning_rate=lr,max_leaf_nodes=lv,
      min_samples_leaf=msl,l2_regularization=1.0,random_state=seed,early_stopping=True,
      validation_fraction=0.1,n_iter_no_change=40).fit(X,y)
    return m.predict(Xa)
def rankavg(ps): return np.vstack([pl.Series(p).rank().to_numpy() for p in ps]).mean(0)

# A: seed-only ensemble (the shipped recipe), 5 members
seedp=[fit(400,0.03,63,200,s) for s in range(5)]
for i,p in enumerate(seedp): print(f'  seed member {i}: {sc(p):.5f}', flush=True)
A=sc(rankavg(seedp)); print(f"A seed-only ensemble-5      {A:.5f}", flush=True)

# B: config-diverse ensemble, 5 members, one seed each
GRID=[(400,0.03,63,200),(200,0.03,63,200),(300,0.045,63,200),(400,0.03,31,200),(250,0.03,95,1000)]
divp=[fit(*g,seed=i) for i,g in enumerate(GRID)]
for g,p in zip(GRID,divp): print(f'  diverse {g}: {sc(p):.5f}', flush=True)
B=sc(rankavg(divp)); print(f"B config-diverse ensemble-5 {B:.5f}", flush=True)

# C: both pools together (10 members)
Cv=sc(rankavg(seedp+divp)); print(f"C all 10 members           {Cv:.5f}", flush=True)
print(f"\n  diversity vs seeds  B-A = {B-A:+.5f}")
print(f"  all-10    vs seeds  C-A = {Cv-A:+.5f}   (shipped offline = 0.38090)")
