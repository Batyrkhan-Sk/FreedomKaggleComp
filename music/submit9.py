"""Real submission: two-stage model plus the metadata columns never loaded before.

Six of eight item_metadata columns and all of user_metadata went unused, so the
model judged every track by PLAY counts alone. That is most damaging for new
tracks, where there is no user-item history to fall back on and the measured
capture is 7.5% against 44% for repeats.

Adds 14 features: explicit like/dislike/download counts and ratios, the user's
own like/download propensity, demographics, and whether the track's genres
contain the user's stated top genre.

Trained on the same four windows as submit7/submit8, applied at 2025-08-31.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from evaluate import TOPK
from meta_feats import META_FEATS, add_meta, load_meta
from rank3 import target
from train4 import load_all
from train7 import FEATS as BASE, featurise

ALL = BASE + META_FEATS
WINDOWS = [("2025-07-01", "2025-07-16"), ("2025-07-16", "2025-07-31"),
           ("2025-08-01", "2025-08-16"), ("2025-08-16", "2025-08-31")]
APPLY = "2025-08-31"

inter, meta, artists, genres, users = load_all()
item_m, user_m = load_meta()

parts = []
for cut, hi in WINDOWS:
    C = add_meta(featurise(inter, meta, artists, genres, users, cut), item_m, user_m)
    p = C.join(target(inter, meta, users, cut, hi), on=["user_id", "item_id"],
               how="left").with_columns(pl.col("y").fill_null(0.0))
    parts.append(p.select(ALL + ["y"]))
    print(f"  window {cut}: {p.height:,}", flush=True)
tr = pl.concat(parts)
X = tr.select(ALL).to_numpy().astype(np.float32)
y = tr["y"].to_numpy()
print(f"total {len(y):,} rows, {(y > 0).sum():,} positive, {len(ALL)} features", flush=True)

clf = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
    random_state=0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
clf.fit(X, (y > 0).astype(int))
pos = y > 0
reg = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=100,
    random_state=0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
reg.fit(X[pos], y[pos])
print(f"fitted clf {clf.n_iter_}, reg {reg.n_iter_}", flush=True)

# which of the new features the model actually leans on
imp = None
try:
    from sklearn.inspection import permutation_importance
except Exception:
    pass

C = add_meta(featurise(inter, meta, artists, genres, users, APPLY), item_m, user_m)
Xa = C.select(ALL).to_numpy().astype(np.float32)
C = C.with_columns(pl.Series("p", clf.predict_proba(Xa)[:, 1] * reg.predict(Xa)))

top = (C.sort(["user_id", "p"], descending=[False, True])
         .group_by("user_id", maintain_order=True).head(TOPK)
         .with_columns(pl.int_range(pl.len()).over("user_id").add(1).alias("rank"))
         .select("user_id", "item_id", "rank", "is_hist"))

n = top.group_by("user_id").len()
assert n["len"].min() == TOPK and n["len"].max() == TOPK, "not exactly 50 per user"
assert top.n_unique(["user_id", "item_id"]) == top.height, "duplicate item for a user"
assert top["user_id"].n_unique() == len(users), "missing users"

share_new = (top["is_hist"] == 0).sum() / top.height
print(f"new-track slots: {share_new:.1%} (baseline model was 34.2%)")

out = top.drop("is_hist").with_row_index("id").select("id", "user_id", "item_id", "rank")
out.write_csv("submission_meta.csv")
print(f"wrote submission_meta.csv: {out.height} rows")
