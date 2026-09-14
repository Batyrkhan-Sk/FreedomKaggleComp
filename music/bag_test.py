"""Ensemble DIVERSITY by bagging, not by seed.

Ensembling is one of only three things that ever paid on this task: a 5-seed
rank-average is worth +0.0017 over its mean member, and it is the last lever
still returning anything.  But every ensemble tried so far draws its diversity
from `random_state` alone (`ens_div.py` varied hyperparameters and added only
+0.0002 on top).  With early_stopping=True the seeds differ ONLY by which 10%
of rows go to the validation split -- a very weak source of diversity.

Random-subspace and row-bagged ensembles inject far more, and neither has been
tried here:

  B  feature bagging -- each member sees a random 70% of the 46 features
  C  row bagging     -- each member fits a random 70% of the 7.25M rows

Control is the KNOWN 5-seed rank-average on the same 46 feats at 200 iters:
0.38006 (`step1.log`, "A2 ensemble-5, rank-average"), whose member mean is
0.37889.  Members are expected to be individually WORSE here; the ensemble is
what matters, so both are printed.
"""
import sys, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

M9 = ["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
STACK = FEATS + ["cv90","cv90_rank"] + M9
ENS5 = 0.38006          # known 5-seed rank-average control, same feats/iters
NMEM = 5

inter, meta, artists, genres, users = load_all()
def merge(n):
    return pl.read_parquet(f"cache_cv90/{n}.parquet").hstack(
        pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9))
tr = pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS]); C = merge("apply")
y = tr["y"].to_numpy()
t = truth(inter.filter(pl.col("d") >= APPLY), meta, users); NA = t["user_id"].n_unique()
npl = (t.group_by("user_id").agg(pl.len().alias("n"))
        .with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den")))
del inter
base = C.select("user_id","item_id")
X  = tr.select(STACK).to_numpy().astype(np.float32)
Xa = C.select(STACK).to_numpy().astype(np.float32)
del tr, C
print(f"train {X.shape}  ens5 control {ENS5}", flush=True)

def sc(p):
    D = base.with_columns(pl.Series("p", p))
    top = (D.sort(["user_id","p"], descending=[False,True])
            .group_by("user_id", maintain_order=True).head(TOPK))
    o = (top.join(t, on=["user_id","item_id"], how="left")
            .with_columns(pl.col("frac").fill_null(0.0))
            .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m = npl.join(o, on="user_id", how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA

def fit(Xt, yt, s):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, random_state=s, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40).fit(Xt, yt)

def run(mode, frac=0.7):
    nf = X.shape[1]; k = int(round(frac*nf)); v=[]; ranks=[]
    for s in range(NMEM):
        rng = np.random.default_rng(1000+s)
        if mode == "feat":
            cols = np.sort(rng.choice(nf, k, replace=False))
            m = fit(X[:, cols], y, s); pr = m.predict(Xa[:, cols])
        else:
            idx = rng.choice(X.shape[0], int(frac*X.shape[0]), replace=False)
            m = fit(X[idx], y[idx], s); pr = m.predict(Xa)
        v.append(sc(pr)); ranks.append(pl.Series(pr).rank().to_numpy())
        print(f"    member {s}: {v[-1]:.5f}", flush=True)
    v = np.array(v); ens = sc(np.vstack(ranks).mean(axis=0))
    print(f"  {mode} bagging {frac}: members mean {v.mean():.5f} sd {v.std(ddof=1):.5f}   "
          f"ENS{NMEM} {ens:.5f}  delta {ens-ENS5:+.5f}", flush=True)

for mode in (sys.argv[1:] or ["feat", "row"]):
    run(mode)
