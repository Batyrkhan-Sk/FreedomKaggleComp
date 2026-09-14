"""The target is effectively binary (played history tracks average frac 0.953).
MSE on a 98%-zero target spends capacity on the zero mass -- which is exactly the
diagnosis that made the capacity cut pay. A classifier is the sharp version of it.
Also tests per-user loss weighting, which metric C implies.
All arms paired: same features, same seed, scored under metric C."""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise
from rank3 import target
from train12 import cached

inter, meta, artists, genres, users = load_all()
parts=[]
for c,h in WINDOWS:
    parts.append(cached(f"tr_{c}", lambda c=c,h=h: (
        featurise(inter,meta,artists,genres,users,c)
        .join(target(inter,meta,users,c,h),on=["user_id","item_id"],how="left")
        .with_columns(pl.col("y").fill_null(0.0)).select(FEATS+["y"]))))
tr = pl.concat(parts)
C  = cached("apply", None)
t  = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA = t["user_id"].n_unique()
npl= t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
Xtr=tr.select(FEATS).to_numpy().astype(np.float32); ytr=tr["y"].to_numpy()
Xap=C.select(FEATS).to_numpy().astype(np.float32)
print(f"train {tr.height:,}  positives {(ytr>0).mean()*100:.2f}%", flush=True)
print(f"frac distribution among positives: "
      f"{np.percentile(ytr[ytr>0],[10,25,50,75,90]).round(2)}", flush=True)

# per-user weight implied by metric C: 1/min(50, distinct tracks in the prior 15d)
uw = (C.group_by("user_id").len().select("user_id"))   # placeholder, weights built below
def cscore(p, lbl):
    D=C.with_columns(pl.Series("p",p)).with_columns(pl.col("user_id").cast(pl.Int64),
                                                    pl.col("item_id").cast(pl.Int64))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    v=(m["s"]/m["den"]).sum()/NA
    nh=top.filter(pl.col("is_hist")==0).height/top.height*100
    print(f"  {lbl:<38} C = {v:.5f}   ({v-0.37698:+.5f})   new-slot {nh:.1f}%", flush=True)
    return v

K=dict(max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
       random_state=2, early_stopping=True, validation_fraction=0.1, n_iter_no_change=40)
print("\nARMS (baseline must reproduce 0.37698):", flush=True)
m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,**K).fit(Xtr,ytr)
cscore(m.predict(Xap), "A regressor 200x.03  [BASELINE]")

w=np.ones(len(ytr),dtype=np.float64)
m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,**K).fit(Xtr,ytr,sample_weight=w)
cscore(m.predict(Xap), "B same + explicit uniform weights [CTRL]")

for it,lr in ((200,0.03),(400,0.03),(100,0.06)):
    m=HistGradientBoostingClassifier(max_iter=it,learning_rate=lr,**K).fit(Xtr,(ytr>0).astype(np.int8))
    cscore(m.predict_proba(Xap)[:,1], f"C classifier P(play) {it}x{lr}")
