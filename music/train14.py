"""Sweep capacity DOWNWARD -- the direction never tested.

C1 (2500 iters) scored -0.0082 and C2 (1500 x 127 leaves) -0.0103, both still
capped. Capacity is monotonically worse above 600, so 600 was acting as
accidental regularisation rather than as a limit. It was inherited, never tuned
against the ranking metric, and the sweep below it has never been run.

This is completing a sweep, not testing a new hypothesis about the data -- the
gradient is already measured and it points down.

Bar remains +0.005 over the cached baseline. Fits get faster as iters fall.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
from train12 import cached

BASE = 0.29035


def main():
    inter, meta, artists, genres, users = load_all()
    tr = pl.concat([cached(f"tr_{cut}", None) for cut, _ in WINDOWS])
    C = cached("apply", None)
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()
    Xtr = tr.select(FEATS).to_numpy().astype(np.float32)
    ytr = tr["y"].to_numpy()
    Xap = C.select(FEATS).to_numpy().astype(np.float32)
    print(f"baseline 600 iters x 63 leaves = {BASE:.5f}   bar {BASE+0.005:.5f}\n", flush=True)

    for it, leaves in ((150, 63), (300, 63), (450, 63), (300, 31)):
        m = HistGradientBoostingRegressor(
            max_iter=it, max_leaf_nodes=leaves, learning_rate=0.06,
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
        flag = "  <-- CLEARS BAR" if off >= BASE + 0.005 else ""
        print(f"{it:5d} iters x {leaves:3d} leaves  (ran {m.n_iter_:4d})  "
              f"offline {off:.5f}  LB~{off*1.29:.4f}  ({off-BASE:+.5f})  "
              f"hist {hs:,.0f} new {ns:,.0f}{flag}", flush=True)


if __name__ == "__main__":
    main()
