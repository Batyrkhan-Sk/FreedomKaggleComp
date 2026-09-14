"""Three ways to turn features into a top-50 list, compared on the same holdout.

The existing model regresses the listening fraction and sorts by the prediction.
That optimises a proxy: squared error spends capacity on getting the exact
fraction right for tracks that will never reach the top 50, while the metric
only cares which 50 are chosen.

  A  regressor          - the current approach, as the baseline
  B  listwise ranker    - XGBoost rank:ndcg, optimising the ordering directly
  C  two-stage          - P(listens at all) x E[fraction | listens]

C exists because a single regressor has to model a spike at zero and a
continuous distribution at once, which trees handle poorly: ~98% of candidate
rows have y = 0.
"""

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from evaluate import TOPK, truth
from rank3 import target
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise


def build(inter, meta, artists, genres, users):
    parts = []
    for cut, hi in WINDOWS:
        C = featurise(inter, meta, artists, genres, users, cut)
        p = C.join(target(inter, meta, users, cut, hi), on=["user_id", "item_id"],
                   how="left").with_columns(pl.col("y").fill_null(0.0))
        parts.append(p.select(["user_id"] + FEATS + ["y"]))
        print(f"  window {cut}: {p.height:,}", flush=True)
    return pl.concat(parts)


def score_ranking(C, pred_col, t, active):
    top = (C.sort(["user_id", pred_col], descending=[False, True])
             .group_by("user_id", maintain_order=True).head(TOPK)
             .select("user_id", "item_id"))
    hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("frac").fill_null(0.0))
    return hit["frac"].sum() / active / TOPK


def main():
    inter, meta, artists, genres, users = load_all()
    tr = build(inter, meta, artists, genres, users)
    print(f"total {tr.height:,} rows, {tr.filter(pl.col('y')>0).height:,} positive")

    Xtr = tr.select(FEATS).to_numpy().astype(np.float32)
    ytr = tr["y"].to_numpy()

    C = featurise(inter, meta, artists, genres, users, APPLY)
    Xap = C.select(FEATS).to_numpy().astype(np.float32)
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()

    # ---- A: regressor (current) -----------------------------------------
    a = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, random_state=0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40)
    a.fit(Xtr, ytr)
    C = C.with_columns(pl.Series("pa", a.predict(Xap)))
    sa = score_ranking(C, "pa", t, active)
    print(f"\nA  regressor        {sa:.5f}   (proj LB {sa*1.29:.3f})", flush=True)

    # ---- B: listwise ranker ---------------------------------------------
    # rank:ndcg needs integer relevance grades and rows grouped by user
    srt = tr.sort("user_id")
    Xr = srt.select(FEATS).to_numpy().astype(np.float32)
    yr = (srt["y"].to_numpy() * 4).round().astype(np.int32)
    qid = srt["user_id"].to_numpy()
    b = xgb.XGBRanker(objective="rank:ndcg", eval_metric="ndcg@50",
                      n_estimators=600, learning_rate=0.06, max_depth=8,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                      random_state=0, verbosity=0, lambdarank_num_pair_per_sample=8)
    b.fit(Xr, yr, qid=qid)
    C = C.with_columns(pl.Series("pb", b.predict(Xap)))
    sb = score_ranking(C, "pb", t, active)
    print(f"B  listwise ranker  {sb:.5f}   (proj LB {sb*1.29:.3f})", flush=True)

    # ---- C: two-stage ----------------------------------------------------
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
        random_state=0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
    clf.fit(Xtr, (ytr > 0).astype(int))
    pos = ytr > 0
    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=100,
        random_state=0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)
    reg.fit(Xtr[pos], ytr[pos])
    pc = clf.predict_proba(Xap)[:, 1] * reg.predict(Xap)
    C = C.with_columns(pl.Series("pc", pc))
    sc = score_ranking(C, "pc", t, active)
    print(f"C  two-stage        {sc:.5f}   (proj LB {sc*1.29:.3f})", flush=True)

    # ---- blends ----------------------------------------------------------
    for wa, wb, wc, name in ((0.5, 0.5, 0.0, "A+B"), (0.34, 0.33, 0.33, "A+B+C")):
        def z(col):
            v = C[col].to_numpy()
            return (v - v.mean()) / (v.std() + 1e-9)
        C = C.with_columns(pl.Series("pz", wa*z("pa") + wb*z("pb") + wc*z("pc")))
        s = score_ranking(C, "pz", t, active)
        print(f"   blend {name:<6}    {s:.5f}   (proj LB {s*1.29:.3f})", flush=True)


if __name__ == "__main__":
    main()
