"""Is the residual 0.53% from OUR code or from sklearn's threading?

The notebook now agrees with submit15 on 99.47% of slots (1440/1500 users
bit-identical), up from 83.58% this morning. Two candidates for the remainder:

  (a) our code -- some join or selection still order-dependent  -> fixable
  (b) sklearn -- HistGradientBoosting builds histograms with OpenMP, and
      floating-point addition is not associative, so thread scheduling perturbs
      the last bits of the gradient sums  -> not fixable without single-threading

Decisive and cheap: fit the SAME data twice in ONE process with the same seed.
Features are byte-identical by construction here, so any difference in the
predictions is sklearn, not us.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from train7 import FEATS, WINDOWS
from train21 import CACHE

tr = pl.concat([pl.read_parquet(CACHE / f"tr_{c}.parquet") for c, _ in WINDOWS])
C = pl.read_parquet(CACHE / "apply.parquet")
X = tr.select(FEATS).to_numpy().astype(np.float32)
y = tr["y"].to_numpy()
Xap = C.select(FEATS).to_numpy().astype(np.float32)

def fit():
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, random_state=2, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40).fit(X, y).predict(Xap)

a, b = fit(), fit()
same = np.array_equal(a, b)
print(f"identical predictions from two fits on identical data: {same}")
if not same:
    d = np.abs(a - b)
    print(f"  differing rows {int((d>0).sum()):,}/{len(d):,}  max |diff| {d.max():.3e}")
    print("  -> sklearn/OpenMP float non-associativity, not our code")
else:
    print("  -> the fit is deterministic; the residual is in notebook-vs-script code")
