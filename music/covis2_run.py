"""Full test: shipped 35 features + 9 multi-statistic covis features.
Baseline on cache_fix = 0.37889 over 3 seeds (sd 0.00103)."""
import pathlib, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise
from rank3 import target
from covis2 import covis_multi

CACHE = pathlib.Path("cache_cv2"); CACHE.mkdir(exist_ok=True)
NEW = ["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last",
       "cv_max_rank","cv_wsum_rank","cv_top3_rank"]
inter, meta, artists, genres, users = load_all()

def feat2(cut):
    C = featurise(inter, meta, artists, genres, users, cut)
    pool = C.filter(pl.col("is_hist")==0)["item_id"].unique().to_list()
    cv2 = covis_multi(inter, users, pool, cut, prefix="cv")
    C = C.with_columns(pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
    if cv2 is None:
        return C.with_columns([pl.lit(0.0).alias(c) for c in NEW]).sort(["user_id","item_id"])
    C = C.join(cv2, on=["user_id","item_id"], how="left")
    fills = {c: (9999.0 if c.endswith("_rank") else 0.0) for c in NEW}
    return C.with_columns([pl.col(c).fill_null(v) for c, v in fills.items()]).sort(["user_id","item_id"])

def build(name, fn):
    f = CACHE / f"{name}.parquet"
    if f.exists(): print(f"  hit {name}", flush=True); return pl.read_parquet(f)
    print(f"  building {name}", flush=True); df = fn(); df.write_parquet(f); return df

ALL = FEATS + NEW
tr = pl.concat([build(f"tr_{c}", lambda c=c,h=h: (
    feat2(c).join(target(inter,meta,users,c,h),on=["user_id","item_id"],how="left")
    .with_columns(pl.col("y").fill_null(0.0)).select(ALL+["y"]))) for c,h in WINDOWS])
C = build("apply", lambda: feat2(APPLY))
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA = t["user_id"].n_unique()
npl= t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
Xtr=tr.select(ALL).to_numpy().astype(np.float32); ytr=tr["y"].to_numpy()
base=C.select("user_id","item_id")
print(f"\ntrain {tr.height:,}  features {len(ALL)}", flush=True)

def go(feats, lbl):
    X = tr.select(feats).to_numpy().astype(np.float32)
    Xa = C.select(feats).to_numpy().astype(np.float32)
    vals=[]
    for seed in (0,1,2):
        m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,max_leaf_nodes=63,
          min_samples_leaf=200,l2_regularization=1.0,random_state=seed,early_stopping=True,
          validation_fraction=0.1,n_iter_no_change=40).fit(X,ytr)
        D=base.with_columns(pl.Series("p",m.predict(Xa)))
        top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
        o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
            .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
        mm=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
        vals.append((mm["s"]/mm["den"]).sum()/NA)
    v=np.array(vals)
    print(f"  {lbl:<34} mean {v.mean():.5f}  sd {v.std(ddof=1):.5f}  "
          f"seeds {' '.join(f'{x:.5f}' for x in v)}", flush=True)
    return v.mean()

a = go(FEATS, "A 35 feats [CTRL, expect .37889]")
b = go(ALL,   "B 35 + 9 multi-covis")
print(f"\ndelta B-A = {b-a:+.5f}   (bar: ~0.003)")
