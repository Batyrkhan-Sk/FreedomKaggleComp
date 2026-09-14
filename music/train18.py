"""#4 A different GBM, and a blend -- variance reduction via a different bias.

Both lightgbm and xgboost are in this project's dependencies and neither has
ever been used; the whole task runs on sklearn's HistGradientBoosting. A
different implementation is a different inductive bias (leaf-wise vs depth-wise
growth, different binning, different regularisation), so its errors are only
partly correlated with HGB's -- which is what makes a blend worth more than
either alone.

Deliberately matched to the tuned HGB config rather than tuned separately: today
established that TOTAL FITTING is what this problem is sensitive to (capacity
down = better, loss reweighting = worse), so the fair comparison holds the
amount of fitting roughly constant and varies only the algorithm.

Reports each model alone and the rank-averaged blend. Rank-averaging rather than
score-averaging because the two libraries produce differently-scaled outputs and
only the ORDER is scored.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb

from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
from train12 import cached

BASE = 0.29834


def main():
    inter, meta, artists, genres, users = load_all()
    tr = pl.concat([cached(f"tr_{c}", None) for c, _ in WINDOWS])
    C = cached("apply", None)
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()
    Xtr = tr.select(FEATS).to_numpy().astype(np.float32)
    ytr = tr["y"].to_numpy()
    Xap = C.select(FEATS).to_numpy().astype(np.float32)
    print(f"baseline (shipped) {BASE:.5f}   bar {BASE+0.005:.5f}\n", flush=True)

    def score(p, label):
        D = C.with_columns(pl.Series("p", p))
        top = (D.sort(["user_id", "p"], descending=[False, True])
                 .group_by("user_id", maintain_order=True).head(TOPK)
                 .select("user_id", "item_id", "is_hist"))
        hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
            pl.col("frac").fill_null(0.0))
        off = hit["frac"].sum() / active / TOPK
        hs = hit.filter(pl.col("is_hist") == 1)["frac"].sum()
        ns = hit.filter(pl.col("is_hist") == 0)["frac"].sum()
        flag = "  <-- CLEARS BAR" if off >= BASE + 0.005 else ""
        print(f"{label:26s} offline {off:.5f}  LB~{off*1.29:.4f}  ({off-BASE:+.5f})  "
              f"hist {hs:,.0f} new {ns:,.0f}{flag}", flush=True)
        return off

    h = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, random_state=0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40).fit(Xtr, ytr)
    ph = h.predict(Xap)
    score(ph, "HGB (shipped config)")

    g = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.03, num_leaves=63, min_child_samples=200,
        reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=-1).fit(Xtr, ytr)
    pg = g.predict(Xap)
    score(pg, "LightGBM")

    # rank-average: the libraries produce different scales, only ORDER is scored
    rk = lambda p: pl.Series(p).rank("average").to_numpy().astype(np.float64)
    rh, rg = rk(ph), rk(pg)
    for w in (0.25, 0.5, 0.75):
        score(w * rg + (1 - w) * rh, f"blend rank {1-w:.2f}H/{w:.2f}L")


if __name__ == "__main__":
    main()
