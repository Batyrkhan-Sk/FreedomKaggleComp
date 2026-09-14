"""Metric-aligned per-user SAMPLE WEIGHTS.

The corrected metric (2026-08-27) is  C = mean over ACTIVE users of
    sum(frac over the 50 slots) / min(50, n_played_u)
so one correct slot is worth  1/den_u  -- up to 50x more for a light user than
for a power user.  The regressor is trained with EVERY ROW WEIGHTED EQUALLY,
which is the opposite: power users have more candidate rows AND lower per-slot
value, so they dominate the loss twice over.

The recorded weighting failure (-0.0049) upweighted POSITIVES, and was measured
in the old point-weighted metric space.  Per-USER weighting under metric C is a
different axis and has never been measured.

  w_u = (50 / den_u) ** alpha        den_u = min(50, n_played_u in the target window)
  alpha = 0 is the control.

Inactive users in a training window (den = 0) keep w = 1.
"""
import sys, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
from rank3 import target as _t

M9 = ["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
W2 = ["cv90","cv90_rank"]
STACK = FEATS + W2 + M9
ALPHAS = [float(a) for a in (sys.argv[1:] or ["0", "0.5", "1.0"])]

inter, meta, artists, genres, users = load_all()

def merge(n):
    return pl.read_parquet(f"cache_cv90/{n}.parquet").hstack(
        pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9))

tr = pl.concat([merge(f"tr_{c}") for c, _ in WINDOWS])
C  = merge("apply")
y  = tr["y"].to_numpy()

# --- per-row den, from each training window's own target ---------------------
dens = []
for cut, hi in WINDOWS:
    uid = pl.read_parquet(f"cache_uid/tr_{cut}.parquet")
    tw = _t(inter, meta, users, cut, hi)
    d = (tw.group_by("user_id").agg(pl.len().alias("n"))
           .with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den")))
    m = uid.join(d.select("user_id", "den"), on="user_id", how="left")
    assert m.height == uid.height
    dens.append(m["den"].fill_null(TOPK).cast(pl.Float64).to_numpy())
    act = d.height
    q = np.percentile(d["den"].to_numpy(), [10, 25, 50, 75, 90])
    print(f"  {cut}: {act} active users, den pct10/25/50/75/90 = {q}", flush=True)
den = np.concatenate(dens)
assert len(den) == tr.height, (len(den), tr.height)
print(f"  rows with den<50: {100*(den<TOPK).mean():.1f}%", flush=True)

# --- scorer (metric C) -------------------------------------------------------
t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
NA = t["user_id"].n_unique()
npl = (t.group_by("user_id").agg(pl.len().alias("n"))
        .with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den")))
del inter
base = C.select("user_id", "item_id")
X  = tr.select(STACK).to_numpy().astype(np.float32)
Xa = C.select(STACK).to_numpy().astype(np.float32)
del tr, C
print(f"train {X.shape}", flush=True)

def sc(p):
    D = base.with_columns(pl.Series("p", p))
    top = (D.sort(["user_id","p"], descending=[False,True])
            .group_by("user_id", maintain_order=True).head(TOPK))
    o = (top.join(t, on=["user_id","item_id"], how="left")
            .with_columns(pl.col("frac").fill_null(0.0))
            .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m = npl.join(o, on="user_id", how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"] / m["den"]).sum() / NA

def go(w, lbl, seeds=(0,1,2)):
    v = []
    for s in seeds:
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
            min_samples_leaf=200, l2_regularization=1.0, random_state=s, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=40).fit(X, y, sample_weight=w)
        v.append(sc(m.predict(Xa)))
    v = np.array(v)
    print(f"  {lbl:<34} mean {v.mean():.5f} sd {v.std(ddof=1):.5f}", flush=True)
    return v.mean()

ctrl = None
for a in ALPHAS:
    w = None if a == 0 else (TOPK / den) ** a
    lbl = "CTRL (unweighted) [expect .37966]" if a == 0 else f"alpha = {a}"
    r = go(w, lbl)
    if a == 0: ctrl = r
    elif ctrl is not None: print(f"     delta {r-ctrl:+.5f}   (bar ~0.003)", flush=True)
