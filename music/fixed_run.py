"""Rebuild features with the DETERMINISTIC covis, then score under metric C.

Reference points, both of which are single draws from the broken covis's
run-to-run variance:
    cache_feats draw = 0.37698   (this is the one that maps to LB 0.38351)
    cache_all   draw = 0.37998
Three model seeds here give the remaining noise floor now that features are fixed.
"""
import pathlib, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise
from rank3 import target

CACHE = pathlib.Path("cache_fix"); CACHE.mkdir(exist_ok=True)
inter, meta, artists, genres, users = load_all()

def build(name, fn):
    f = CACHE / f"{name}.parquet"
    if f.exists():
        print(f"  hit {name}", flush=True); return pl.read_parquet(f)
    print(f"  building {name} ...", flush=True)
    df = fn(); df.write_parquet(f); return df

parts = []
for c, h in WINDOWS:
    parts.append(build(f"tr_{c}", lambda c=c, h=h: (
        featurise(inter, meta, artists, genres, users, c)
        .join(target(inter, meta, users, c, h), on=["user_id","item_id"], how="left")
        .with_columns(pl.col("y").fill_null(0.0)).select(FEATS + ["y"]))))
tr = pl.concat(parts)
C = build("apply", lambda: featurise(inter, meta, artists, genres, users, APPLY))
t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
NA = t["user_id"].n_unique()
npl = t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
Xtr = tr.select(FEATS).to_numpy().astype(np.float32); ytr = tr["y"].to_numpy()
Xap = C.select(FEATS).to_numpy().astype(np.float32)
print(f"\ntrain {tr.height:,}  apply {C.height:,}  active {NA}", flush=True)

vals = []
for seed in (0, 1, 2):
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, random_state=seed, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40).fit(Xtr, ytr)
    D = (C.with_columns(pl.Series("p", m.predict(Xap)))
          .with_columns(pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)))
    top = (D.sort(["user_id","p"], descending=[False,True])
             .group_by("user_id", maintain_order=True).head(TOPK))
    o = (top.join(t, on=["user_id","item_id"], how="left").with_columns(pl.col("frac").fill_null(0.0))
          .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    mm = npl.join(o, on="user_id", how="left").with_columns(pl.col("s").fill_null(0.0))
    v = (mm["s"]/mm["den"]).sum()/NA; vals.append(v)
    print(f"  FIXED covis, seed {seed}:  C = {v:.5f}   (vs 0.37698 shipped draw: {v-0.37698:+.5f})", flush=True)

vals = np.array(vals)
print(f"\nmean {vals.mean():.5f}   seed sd {vals.std(ddof=1):.5f}   range {vals.max()-vals.min():.5f}")
print(f"broken-covis draws were 0.37698 and 0.37998 (spread 0.00300)")
