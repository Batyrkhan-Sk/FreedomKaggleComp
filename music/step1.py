"""Step 1 on the deterministic cache: no feature rebuild needed.
Baseline = fixed-covis 3-seed mean 0.37889 (sd 0.00103)."""
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
Xtr = tr.select(FEATS).to_numpy().astype(np.float32); ytr = tr["y"].to_numpy()
Xap = C.select(FEATS).to_numpy().astype(np.float32)
hist_tr = tr["is_hist"].to_numpy() == 1
hist_ap = (C["is_hist"].to_numpy() == 1)
base = C.select("user_id","item_id").with_columns(
    pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
print(f"train {tr.height:,}  apply {C.height:,}", flush=True)

def score(p, lbl):
    D = base.with_columns(pl.Series("p", p))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    v=(m["s"]/m["den"]).sum()/NA
    print(f"  {lbl:<40} C = {v:.5f}  ({v-0.37889:+.5f} vs mean)", flush=True)
    return v

def fit(X, y, seed):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, random_state=seed, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40).fit(X, y)

# --- A: seed ensembling, 5 members, score-average and rank-average
preds = []
for s in range(5):
    preds.append(fit(Xtr, ytr, s).predict(Xap))
    print(f"    member seed {s}: ", end=""); score(preds[-1], f"member {s}")
P = np.vstack(preds)
score(P.mean(axis=0), "A1 ensemble-5, score-average")
R = np.vstack([pl.Series(p).rank().to_numpy() for p in P])
score(R.mean(axis=0), "A2 ensemble-5, rank-average")

# --- B: two specialists, adaptive joint top-50
ph = fit(Xtr[hist_tr], ytr[hist_tr], 0).predict(Xap)
pn = fit(Xtr[~hist_tr], ytr[~hist_tr], 0).predict(Xap)
spec = np.where(hist_ap, ph, pn)
score(spec, "B  two specialists, joint top-50")
