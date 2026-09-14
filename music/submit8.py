"""Real submission from the two-stage model, which measured best on the holdout.

train8.py compared three ways of turning the same features into a top-50 and
two-stage won: 0.29281 offline against 0.29036 for the regressor that produced
the current 0.37710. Small -- about 2.5x the ~0.001 run-to-run noise -- but it is
the first thing to beat the regressor on this harness.

The split exists because ~98.4% of the 7.25M candidate rows have y = 0 (115,277
positive). One regressor has to fit a spike at zero and a continuum above it at
once; separating P(listens) from E[fraction | listens] lets each model do one job.

Trained on the same four stacked windows as submit7.py, applied at 2025-08-31.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from evaluate import TOPK
from rank3 import target
from train4 import load_all
from train7 import FEATS, featurise

WINDOWS = [("2025-07-01", "2025-07-16"), ("2025-07-16", "2025-07-31"),
           ("2025-08-01", "2025-08-16"), ("2025-08-16", "2025-08-31")]
APPLY = "2025-08-31"

inter, meta, artists, genres, users = load_all()
parts = []
for cut, hi in WINDOWS:
    C = featurise(inter, meta, artists, genres, users, cut)
    parts.append(C.join(target(inter, meta, users, cut, hi), on=["user_id", "item_id"],
                        how="left").with_columns(pl.col("y").fill_null(0.0))
                 .select(FEATS + ["y"]))
    print(f"  window {cut}: {parts[-1].height:,}", flush=True)
tr = pl.concat(parts)
X = tr.select(FEATS).to_numpy().astype(np.float32)
y = tr["y"].to_numpy()
print(f"total {len(y):,} rows, {(y > 0).sum():,} positive ({(y > 0).mean():.2%})", flush=True)

clf = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
    random_state=0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
clf.fit(X, (y > 0).astype(int))
pos = y > 0
reg = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=100,
    random_state=0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
reg.fit(X[pos], y[pos])
print(f"fitted clf {clf.n_iter_} iters, reg {reg.n_iter_} iters", flush=True)

C = featurise(inter, meta, artists, genres, users, APPLY)
Xa = C.select(FEATS).to_numpy().astype(np.float32)
C = C.with_columns(pl.Series("p", clf.predict_proba(Xa)[:, 1] * reg.predict(Xa)))

top = (C.sort(["user_id", "p"], descending=[False, True])
         .group_by("user_id", maintain_order=True).head(TOPK)
         .with_columns(pl.int_range(pl.len()).over("user_id").add(1).alias("rank"))
         .select("user_id", "item_id", "rank"))

# the host requires exactly 50 rows per user, ranks 1..50, no duplicate items
n = top.group_by("user_id").len()
assert n["len"].min() == TOPK and n["len"].max() == TOPK, "not exactly 50 per user"
assert top.n_unique(["user_id", "item_id"]) == top.height, "duplicate item for a user"
assert top["user_id"].n_unique() == len(users), "missing users"

out = top.with_row_index("id").select("id", "user_id", "item_id", "rank")
out.write_csv("submission_twostage.csv")
print(f"wrote submission_twostage.csv: {out.height} rows, "
      f"{top['user_id'].n_unique()} users x {TOPK}")
