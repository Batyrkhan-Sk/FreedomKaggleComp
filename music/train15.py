"""Extend the capacity sweep below 150 -- the optimum is at the search boundary.

Five measured points are monotonic in the same direction, and history capture
(the side carrying the +0.088 oracle gap) rises the whole way down:

  2500 -0.0082 | 1500x127 -0.0103 | 600 base | 300 +0.0025 | 150 +0.0046

150 ran 150 of 150 -- capped, still improving. A best value at the edge of the
search means the search is mis-centred, not that 150 is the answer.

Also sweeps learning_rate, because iterations and step size trade off directly:
a smaller lr at the same iteration count is a different point on the same
regularisation axis, and fixing lr=0.06 while only moving iters searches a line
through a plane.

Bar is +0.005 over the cached baseline 0.29035.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
from train12 import cached

BASE = 0.29035
GRID = [(40, 0.06), (70, 0.06), (100, 0.06), (130, 0.06),
        (100, 0.12), (200, 0.03)]


def main():
    inter, meta, artists, genres, users = load_all()
    tr = pl.concat([cached(f"tr_{cut}", None) for cut, _ in WINDOWS])
    C = cached("apply", None)
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()
    Xtr = tr.select(FEATS).to_numpy().astype(np.float32)
    ytr = tr["y"].to_numpy()
    Xap = C.select(FEATS).to_numpy().astype(np.float32)
    print(f"baseline {BASE:.5f}  bar {BASE+0.005:.5f}   "
          f"(150x0.06 measured at +0.00458)\n", flush=True)

    best = (0.0, None)
    for it, lr in GRID:
        m = HistGradientBoostingRegressor(
            max_iter=it, max_leaf_nodes=63, learning_rate=lr,
            min_samples_leaf=200, l2_regularization=1.0, random_state=0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=40)
        m.fit(Xtr, ytr)
        D = C.with_columns(pl.Series("p", m.predict(Xap)))
        top = (D.sort(["user_id", "p"], descending=[False, True])
                 .group_by("user_id", maintain_order=True).head(TOPK)
                 .select("user_id", "item_id", "is_hist"))
        hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
            pl.col("frac").fill_null(0.0))
        off = hit["frac"].sum() / active / TOPK
        hs = hit.filter(pl.col("is_hist") == 1)["frac"].sum()
        ns = hit.filter(pl.col("is_hist") == 0)["frac"].sum()
        best = max(best, (off, (it, lr)))
        flag = "  <-- CLEARS BAR" if off >= BASE + 0.005 else ""
        print(f"{it:4d} iters  lr {lr:.2f}  (ran {m.n_iter_:4d})  offline {off:.5f}  "
              f"LB~{off*1.29:.4f}  ({off-BASE:+.5f})  hist {hs:,.0f} new {ns:,.0f}{flag}",
              flush=True)

    print(f"\nbest {best[1]} at {best[0]:.5f} ({best[0]-BASE:+.5f})")


if __name__ == "__main__":
    main()
