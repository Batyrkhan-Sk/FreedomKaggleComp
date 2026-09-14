"""Add a SECOND covis at a longer window + laxer min_secs, as EXTRA features.

Motivation is a measured coverage gap, not a tuning hunch: 313 of 1153 active
users have NO covis at all (window 30d, min_secs 30) and they hold 0.078 of the
0.195 in-pool new-track oracle at 17 slots -- 40% of it. Under the old
point-weighted metric these low-activity users were nearly worthless, which is
why this never surfaced.

Kept as ADDITIONAL columns so the 30-day values are untouched -- the recorded
window sweep replaced them and made covis more accurate but more redundant with
recency (the proxy trap).
"""
import pathlib, functools, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise
from rank3 import target
from covis import covis_features

W, MS = 90, 10
NEW = ["cv90", "cv90_rank"]
CACHE = pathlib.Path("cache_cv90"); CACHE.mkdir(exist_ok=True)
inter, meta, artists, genres, users = load_all()

def feat3(cut):
    C = featurise(inter, meta, artists, genres, users, cut).with_columns(
        pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
    pool = C.filter(pl.col("is_hist")==0)["item_id"].unique().to_list()
    cv = covis_features(inter, users, pool, cut, window_days=W, min_secs=MS)
    if cv.height == 0:
        return C.with_columns(pl.lit(0.0).alias("cv90"), pl.lit(9999.0).alias("cv90_rank")).sort(["user_id","item_id"])
    cv = cv.rename({"covis":"cv90"}).with_columns(
        pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
    print(f"    cv90 covers {cv['user_id'].n_unique()} users", flush=True)
    C = C.join(cv, on=["user_id","item_id"], how="left").with_columns(pl.col("cv90").fill_null(0.0))
    return (C.with_columns(pl.col("cv90").rank("min", descending=True).over("user_id").alias("cv90_rank"))
             .sort(["user_id","item_id"]))

def build(name, fn):
    f = CACHE / f"{name}.parquet"
    if f.exists(): print(f"  hit {name}", flush=True); return pl.read_parquet(f)
    print(f"  building {name}", flush=True); df = fn(); df.write_parquet(f); return df

ALL = FEATS + NEW
tr = pl.concat([build(f"tr_{c}", lambda c=c,h=h: (
    feat3(c).join(target(inter,meta,users,c,h),on=["user_id","item_id"],how="left")
    .with_columns(pl.col("y").fill_null(0.0)).select(ALL+["y"]))) for c,h in WINDOWS])
C = build("apply", lambda: feat3(APPLY))
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA = t["user_id"].n_unique()
npl= t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
ytr=tr["y"].to_numpy(); base=C.select("user_id","item_id")
print(f"\ntrain {tr.height:,}  features {len(ALL)}", flush=True)

def go(feats, lbl):
    X=tr.select(feats).to_numpy().astype(np.float32); Xa=C.select(feats).to_numpy().astype(np.float32)
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
    print(f"  {lbl:<36} mean {v.mean():.5f}  sd {v.std(ddof=1):.5f}  "
          f"seeds {' '.join(f'{x:.5f}' for x in v)}", flush=True)
    return v.mean()

a = go(FEATS, "A 35 feats [CTRL, expect .37889]")
b = go(ALL,   f"B 35 + covis@{W}d/min{MS}s")
print(f"\ndelta B-A = {b-a:+.5f}   (bar ~0.003)")
