"""Slot allocation re-swept under metric C (the real one). All prior allocation
tuning used the point-weighted metric, which favours heavy users."""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise
from rank3 import target
from train12 import cached

inter, meta, artists, genres, users = load_all()
tr = pl.concat([cached(f"tr_{c}", lambda c=c,h=h: (
    featurise(inter,meta,artists,genres,users,c)
    .join(target(inter,meta,users,c,h),on=["user_id","item_id"],how="left")
    .with_columns(pl.col("y").fill_null(0.0)).select(FEATS+["y"]))) for c,h in WINDOWS])
C = cached("apply", None)
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))

m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
    min_samples_leaf=200, l2_regularization=1.0, random_state=2, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=40)
m.fit(tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy())
D = (C.with_columns(pl.Series("p", m.predict(C.select(FEATS).to_numpy().astype(np.float32))))
      .with_columns(pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)))
D.select("user_id","item_id","is_hist","p").write_parquet("scored_apply.parquet")

def cscore(recs, lbl):
    o=(recs.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    mm=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    v=(mm["s"]/mm["den"]).sum()/NA
    print(f"  {lbl:<28} C = {v:.5f}", flush=True); return v

rk = D.with_columns(pl.col("p").rank("min",descending=True).over(["user_id","is_hist"]).alias("r"))
print(f"\nBASELINE joint top-50:")
joint = D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK)
cscore(joint.select("user_id","item_id"), "joint (shipped)")
print(f"\nFIXED history/new splits:")
best=(None,0)
for k in (50,45,40,36,34,30,25,20,15,10):
    r = pl.concat([rk.filter((pl.col("is_hist")==1)&(pl.col("r")<=k)).select("user_id","item_id"),
                   rk.filter((pl.col("is_hist")==0)&(pl.col("r")<=TOPK-k)).select("user_id","item_id")])
    v=cscore(r, f"hist {k} / new {TOPK-k}")
    if v>best[1]: best=(k,v)
print(f"\nbest fixed split: hist {best[0]} -> {best[1]:.5f}")
