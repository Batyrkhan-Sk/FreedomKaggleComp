"""Regularisation knobs that were NEVER swept -- on either metric.

`capsweep`/`capsweep2` swept max_iter, learning_rate and max_leaf_nodes and found
400 x .03 x 63.  Three settings in the same call were inherited untouched:

  1. early_stopping=True + validation_fraction=0.1.  At a FIXED 400 iters the
     stopper never fires (every log line reads "400 iters used"), so all it does
     is withhold 10% of the training rows from fitting, every seed.  Turning it
     off is +11% training data for free.  Risk: the random validation split is
     part of what makes the 5 seeds differ, and the ensemble is worth +0.0017 --
     so this is checked as an ENSEMBLE too, not just per-member.
  2. min_samples_leaf=200 -- never swept.  The one real modelling win on this
     task was LESS fitting (the metric rewards top-50 ORDERING, not MSE on a
     98.4%-zero target), and min_samples_leaf is the regularisation axis that
     acts on leaf purity rather than on tree count.
  3. l2_regularization=1.0 -- never swept.

Control is the standing 46-feature harness: 200 x .03 x 63, seeds (0,1,2),
3 cached windows = 0.37966 (sd 0.00083), reproduced to 5 decimals twice.
Arms run at 200 iters for comparability; any winner gets re-confirmed at the
shipped 400 x 5-seed config before it goes near a submission.
"""
import sys, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

M9 = ["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
STACK = FEATS + ["cv90","cv90_rank"] + M9
CTRL = 0.37966          # the standing control, reproduced exactly on 08-29

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
print(f"train {X.shape}  CTRL {CTRL}", flush=True)

def sc(p):
    D = base.with_columns(pl.Series("p", p))
    top = (D.sort(["user_id","p"], descending=[False,True])
            .group_by("user_id", maintain_order=True).head(TOPK))
    o = (top.join(t, on=["user_id","item_id"], how="left")
            .with_columns(pl.col("frac").fill_null(0.0))
            .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m = npl.join(o, on="user_id", how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA

def go(lbl, seeds=(0,1,2), **kw):
    p = dict(max_iter=200, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=200,
             l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
             n_iter_no_change=40)
    p.update(kw)
    v, ranks = [], []
    for s in seeds:
        m = HistGradientBoostingRegressor(random_state=s, **p).fit(X, y)
        pr = m.predict(Xa); v.append(sc(pr))
        ranks.append(pl.Series(pr).rank().to_numpy())
    v = np.array(v); ens = sc(np.vstack(ranks).mean(axis=0))
    print(f"  {lbl:<32} mean {v.mean():.5f} sd {v.std(ddof=1):.5f}  "
          f"delta {v.mean()-CTRL:+.5f}   ens3 {ens:.5f}", flush=True)
    return v.mean()

ARMS = [
    ("CTRL [expect .37966]",       dict()),
    ("early_stopping=False",   dict(early_stopping=False)),
    ("min_samples_leaf=50",    dict(min_samples_leaf=50)),
    ("min_samples_leaf=1000",  dict(min_samples_leaf=1000)),
    ("l2_regularization=10",   dict(l2_regularization=10.0)),
]
want = sys.argv[1:]
for lbl, kw in ARMS:
    if want and not any(w in lbl for w in want): continue
    go(lbl, **kw)
