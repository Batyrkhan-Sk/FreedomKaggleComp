"""Does the unused metadata lift new-track discrimination?

Same harness, same windows, same two-stage model as train8.py -- the only
change is 14 extra features from the item and user metadata columns the
pipeline never loaded. Reports A/C with and without them so the delta is
attributable, and breaks the score down by slot kind, because the whole
argument for these features is that they should help NEW tracks specifically.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from evaluate import TOPK, truth
from meta_feats import META_FEATS, add_meta, load_meta
from rank3 import target
from train4 import load_all
from train7 import FEATS as BASE, WINDOWS, APPLY, featurise

ALL = BASE + META_FEATS


def two_stage(Xtr, ytr, Xap):
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
        random_state=0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30).fit(Xtr, (ytr > 0).astype(int))
    pos = ytr > 0
    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=100,
        random_state=0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30).fit(Xtr[pos], ytr[pos])
    return clf.predict_proba(Xap)[:, 1] * reg.predict(Xap)


def main():
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
    ytr = tr["y"].to_numpy()

    C = add_meta(featurise(inter, meta, artists, genres, users, APPLY), item_m, user_m)
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()

    for name, feats in (("baseline (no metadata)", BASE), ("with metadata", ALL)):
        p = two_stage(tr.select(feats).to_numpy().astype(np.float32), ytr,
                      C.select(feats).to_numpy().astype(np.float32))
        D = C.with_columns(pl.Series("p", p))
        top = (D.sort(["user_id", "p"], descending=[False, True])
                 .group_by("user_id", maintain_order=True).head(TOPK)
                 .select("user_id", "item_id", "is_hist"))
        hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
            pl.col("frac").fill_null(0.0))
        s = hit["frac"].sum() / active / TOPK
        newp = hit.filter(pl.col("is_hist") == 0)["frac"].sum()
        histp = hit.filter(pl.col("is_hist") == 1)["frac"].sum()
        print(f"\n{name:24s} offline {s:.5f}  (proj LB {s*1.29:.3f})")
        print(f"    history {histp:7.0f} pts  capture {histp/30946:5.1%}")
        print(f"    new     {newp:7.0f} pts  capture {newp/42321:5.1%}", flush=True)


if __name__ == "__main__":
    main()
