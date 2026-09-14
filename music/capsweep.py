"""Re-sweep CAPACITY under metric C on deterministic features.

This is the only lever that ever produced a real gain here (+0.0055 real), and
the sweep behind it was run on non-deterministic features (~0.003 build noise)
against the point-weighted metric. Metric C weights low-activity users -- who
have the least data to fit -- equally, so the optimum plausibly sits LOWER.

Baseline: 200 iters x lr 0.03, 63 leaves = 0.37889 (3 seeds, sd 0.00103).
Screening at 2 seeds; winner re-run at 5.
"""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

inter, meta, artists, genres, users = load_all()
tr = pl.concat([pl.read_parquet(f"cache_fix/tr_{c}.parquet") for c,_ in WINDOWS])
C  = pl.read_parquet("cache_fix/apply.parquet")
t  = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA = t["user_id"].n_unique()
npl= t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
X=tr.select(FEATS).to_numpy().astype(np.float32); y=tr["y"].to_numpy()
Xa=C.select(FEATS).to_numpy().astype(np.float32)
base=C.select("user_id","item_id")
print(f"train {tr.height:,}", flush=True)

def sc(p):
    D=base.with_columns(pl.Series("p",p))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA

def run(it, lr, leaves, msl, seeds, tag):
    v=[]
    for s in seeds:
        m=HistGradientBoostingRegressor(max_iter=it, learning_rate=lr, max_leaf_nodes=leaves,
          min_samples_leaf=msl, l2_regularization=1.0, random_state=s, early_stopping=True,
          validation_fraction=0.1, n_iter_no_change=40).fit(X,y)
        v.append(sc(m.predict(Xa)))
    v=np.array(v)
    print(f"  {tag:<34} mean {v.mean():.5f}  ({v.mean()-0.37889:+.5f})  "
          f"n={len(seeds)}  {' '.join(f'{x:.5f}' for x in v)}", flush=True)
    return v.mean()

print("\nCAPACITY SWEEP (2-seed screen), baseline 200x.03x63 = 0.37889:", flush=True)
res={}
grid = [(200,0.03,63,200,"200 x .03 x63  [BASELINE]"),
        (120,0.03,63,200,"120 x .03 x63"),
        ( 80,0.03,63,200," 80 x .03 x63"),
        ( 40,0.03,63,200," 40 x .03 x63"),
        ( 40,0.06,63,200," 40 x .06 x63"),
        ( 80,0.06,63,200," 80 x .06 x63"),
        (200,0.03,31,200,"200 x .03 x31 leaves"),
        (200,0.03,63,1000,"200 x .03 x63 msl1000"),
        (120,0.03,31,1000,"120 x .03 x31 msl1000")]
for it,lr,lv,ms,tag in grid:
    res[tag]=run(it,lr,lv,ms,(0,1),tag)
best=max(res,key=res.get)
print(f"\nbest screen: {best} = {res[best]:.5f}", flush=True)
