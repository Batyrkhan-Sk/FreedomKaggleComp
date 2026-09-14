"""Step 2: re-adjudicate the covis parameter optimum on DETERMINISTIC features
under metric C. The recorded verdict (-0.0020) was measured with ~0.003 of
build noise present, so it never actually discriminated.

Baseline (defaults 30d/20 seeds/damp 0.5) = 0.37889 over 3 seeds, sd 0.00103.
Usage: covis_cfg.py <window_days> <n_seeds> <pop_damp>
"""
import sys, pathlib, functools, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
W, S, D = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
tag = f"w{W}_s{S}_d{D}"

import covis, train7
train7.covis_features = functools.partial(covis.covis_features,
                                          window_days=W, n_seeds=S, pop_damp=D)
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise
from rank3 import target

CACHE = pathlib.Path(f"cache_{tag}"); CACHE.mkdir(exist_ok=True)
inter, meta, artists, genres, users = load_all()
def build(name, fn):
    f = CACHE / f"{name}.parquet"
    if f.exists(): return pl.read_parquet(f)
    print(f"  building {name}", flush=True); df = fn(); df.write_parquet(f); return df

tr = pl.concat([build(f"tr_{c}", lambda c=c,h=h: (
    featurise(inter,meta,artists,genres,users,c)
    .join(target(inter,meta,users,c,h),on=["user_id","item_id"],how="left")
    .with_columns(pl.col("y").fill_null(0.0)).select(FEATS+["y"]))) for c,h in WINDOWS])
C = build("apply", lambda: featurise(inter,meta,artists,genres,users,APPLY))
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA = t["user_id"].n_unique()
npl= t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
Xtr=tr.select(FEATS).to_numpy().astype(np.float32); ytr=tr["y"].to_numpy()
Xap=C.select(FEATS).to_numpy().astype(np.float32)
base=C.select("user_id","item_id").with_columns(pl.col("user_id").cast(pl.Int64),
                                                pl.col("item_id").cast(pl.Int64))
nz = (C["covis"].to_numpy()!=0).mean()*100
print(f"\ncovis {tag}: nonzero {nz:.1f}%   train {tr.height:,}", flush=True)
vals=[]
for seed in (0,1,2):
    m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,max_leaf_nodes=63,
      min_samples_leaf=200,l2_regularization=1.0,random_state=seed,early_stopping=True,
      validation_fraction=0.1,n_iter_no_change=40).fit(Xtr,ytr)
    D_=base.with_columns(pl.Series("p",m.predict(Xap)))
    top=(D_.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    mm=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    v=(mm["s"]/mm["den"]).sum()/NA; vals.append(v)
    print(f"  seed {seed}: C = {v:.5f}", flush=True)
vals=np.array(vals)
print(f"\n{tag}  mean {vals.mean():.5f}  sd {vals.std(ddof=1):.5f}   "
      f"vs default mean 0.37889: {vals.mean()-0.37889:+.5f}")
