"""Low capacity is monotonically worse under metric C (200>120>80>40), reversing
the metric-B finding that put the optimum at 40-150. So test UPWARD -- the
direction nobody has tried against the correct objective."""
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
Xa=C.select(FEATS).to_numpy().astype(np.float32); base=C.select("user_id","item_id")
def sc(p):
    D=base.with_columns(pl.Series("p",p))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA
def run(it,lr,lv,tag,seeds=(0,1)):
    v=[]
    for s in seeds:
        m=HistGradientBoostingRegressor(max_iter=it,learning_rate=lr,max_leaf_nodes=lv,
          min_samples_leaf=200,l2_regularization=1.0,random_state=s,early_stopping=True,
          validation_fraction=0.1,n_iter_no_change=40).fit(X,y)
        v.append(sc(m.predict(Xa)))
        print(f"      {tag} seed {s}: {v[-1]:.5f} ({m.n_iter_} iters used)", flush=True)
    v=np.array(v); print(f"  {tag:<26} mean {v.mean():.5f}  ({v.mean()-0.37835:+.5f} vs 200x.03 2-seed)", flush=True)
    return v.mean()
print("UPWARD capacity, vs 200x.03x63 = 0.37835 (2-seed):", flush=True)
for it,lr,lv,tag in [(200,0.03,63,"200 x .03 x63 [CTRL]"),
                     (400,0.03,63,"400 x .03 x63"),
                     (800,0.03,63,"800 x .03 x63"),
                     (400,0.03,127,"400 x .03 x127")]:
    run(it,lr,lv,tag)
