"""Does the item-item embedding add anything the 35 shipped features lack?
Control arm must reproduce 0.37889 (cache_fix 3-seed mean, seed sd 0.00103)."""
import pathlib, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise
from rank3 import target
from item2vec import i2v_features

NEW = ["i2v", "i2v_rank"]
CACHE = pathlib.Path("cache_i2v"); CACHE.mkdir(exist_ok=True)
inter, meta, artists, genres, users = load_all()

def feat(cut):
    C = featurise(inter, meta, artists, genres, users, cut).with_columns(
        pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
    pool = C.filter(pl.col("is_hist") == 0)["item_id"].unique().to_list()
    e = i2v_features(inter, users, pool, cut)
    if e is None:
        return C.with_columns(pl.lit(0.0).alias("i2v"), pl.lit(9999.0).alias("i2v_rank")).sort(["user_id","item_id"])
    e = e.with_columns(pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
    C = C.join(e, on=["user_id","item_id"], how="left")
    return C.with_columns(pl.col("i2v").fill_null(0.0),
                          pl.col("i2v_rank").fill_null(9999.0)).sort(["user_id","item_id"])

def build(n, fn):
    f = CACHE / f"{n}.parquet"
    if f.exists(): print(f"  hit {n}", flush=True); return pl.read_parquet(f)
    print(f"  building {n}", flush=True); df = fn(); df.write_parquet(f); return df

ALL = FEATS + NEW
tr = pl.concat([build(f"tr_{c}", lambda c=c,h=h: (
    feat(c).join(target(inter,meta,users,c,h),on=["user_id","item_id"],how="left")
    .with_columns(pl.col("y").fill_null(0.0)).select(ALL+["y"]))) for c,h in WINDOWS])
C = build("apply", lambda: feat(APPLY))
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA = t["user_id"].n_unique()
npl = t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
ytr = tr["y"].to_numpy(); base = C.select("user_id","item_id")
print(f"\ntrain {tr.height:,}  features {len(ALL)}", flush=True)

def go(feats, lbl):
    X = tr.select(feats).to_numpy().astype(np.float32); Xa = C.select(feats).to_numpy().astype(np.float32)
    v=[]
    for s in (0,1,2):
        m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.03,max_leaf_nodes=63,
          min_samples_leaf=200,l2_regularization=1.0,random_state=s,early_stopping=True,
          validation_fraction=0.1,n_iter_no_change=40).fit(X,ytr)
        D=base.with_columns(pl.Series("p",m.predict(Xa)))
        top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
        o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
            .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
        mm=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
        v.append((mm["s"]/mm["den"]).sum()/NA)
    v=np.array(v); print(f"  {lbl:<34} mean {v.mean():.5f} sd {v.std(ddof=1):.5f}", flush=True)
    return v.mean()

a=go(FEATS,"A 35 feats [CTRL]"); b=go(ALL,"B 35 + item2vec (2)")
print(f"\ndelta = {b-a:+.5f}   (bar ~0.003; 400 iters is the metric-C optimum)")
