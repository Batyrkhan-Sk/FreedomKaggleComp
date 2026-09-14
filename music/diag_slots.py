"""Where do the 50 slots go, and what does each kind of slot actually earn?

Pool recall says the right new track is available ~51% of the time, yet widening
the pool to ~69% recall moved nothing. That points at discrimination rather than
coverage -- but "points at" is not a measurement, and the cheap version is to
look at the slots the model already fills.

If new-track slots are numerous but earn almost nothing, the model cannot tell
which available new track a user will play, and a sequence model attacks exactly
that. If new-track slots are few, the model is declining to spend slots on them
and the fix is calibration between the two kinds instead.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, APPLY, featurise
from train8 import build


def main():
    inter, meta, artists, genres, users = load_all()
    tr = build(inter, meta, artists, genres, users)
    Xtr = tr.select(FEATS).to_numpy().astype(np.float32)
    ytr = tr["y"].to_numpy()

    C = featurise(inter, meta, artists, genres, users, APPLY)
    Xap = C.select(FEATS).to_numpy().astype(np.float32)
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()

    # the two-stage model, which measured best
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
        random_state=0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30).fit(Xtr, (ytr > 0).astype(int))
    pos = ytr > 0
    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=100,
        random_state=0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30).fit(Xtr[pos], ytr[pos])
    C = C.with_columns(pl.Series("p", clf.predict_proba(Xap)[:, 1] * reg.predict(Xap)))

    top = (C.sort(["user_id", "p"], descending=[False, True])
             .group_by("user_id", maintain_order=True).head(TOPK)
             .select("user_id", "item_id", "is_hist"))
    hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("frac").fill_null(0.0))

    print(f"\noffline score {hit['frac'].sum()/active/TOPK:.5f}   active users {active}")
    print(f"\n{'slot kind':12s} {'slots':>8} {'per user':>9} {'points':>9} "
          f"{'pts/slot':>9} {'hit rate':>9}")
    for name, flag in (("history", 1), ("new", 0)):
        s = hit.filter(pl.col("is_hist") == flag)
        n, pts = s.height, s["frac"].sum()
        print(f"{name:12s} {n:8d} {n/len(users):9.1f} {pts:9.0f} "
              f"{pts/max(n,1):9.4f} {(s['frac']>0).sum()/max(n,1):9.1%}")

    # what was ON THE TABLE for each kind, to compare earned against available
    hist_pairs = (inter.filter(pl.col("d") < APPLY)
                  .filter(pl.col("user_id").is_in(users.implode()))
                  .select("user_id", "item_id").unique()
                  .with_columns(pl.lit(1).alias("h")))
    tt = t.join(hist_pairs, on=["user_id", "item_id"], how="left")
    for name, expr in (("history", pl.col("h").is_not_null()),
                       ("new", pl.col("h").is_null())):
        avail = tt.filter(expr)["frac"].sum()
        print(f"  available {name:8s} {avail:9.0f} points "
              f"({avail/active:.1f} per active user)")


if __name__ == "__main__":
    main()
