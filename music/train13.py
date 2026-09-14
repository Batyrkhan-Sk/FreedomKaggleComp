"""The baseline is capacity-limited -- every fit hit max_iter=600 without early stopping.

With early_stopping=True and n_iter_no_change=40, finishing at exactly 600/600
means the validation loss was STILL improving when the cap cut it off. That is
evidence from the fit logs, not another hypothesis about the data, and it is the
cheapest untested lever left on this task.

Two arms, both against the cached baseline 0.29035 (same rows, same seed, so the
reference is exact rather than a re-measurement):
  C1  max_iter 2500              -- just let it run to convergence
  C2  max_iter 1500, 127 leaves  -- more capacity per tree instead of more trees

Bar is +0.005. Reuses cache_feats/, so featurisation is free.
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
    print(f"train {tr.height:,}  apply {C.height:,}  baseline {BASE:.5f}\n", flush=True)

    for label, kw in (("C1 max_iter 2500", dict(max_iter=2500, max_leaf_nodes=63)),
                      ("C2 1500 x 127 leaves", dict(max_iter=1500, max_leaf_nodes=127))):
        m = HistGradientBoostingRegressor(
            learning_rate=0.06, min_samples_leaf=200, l2_regularization=1.0,
            random_state=0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=40, **kw)
        m.fit(Xtr, ytr)
        capped = " (STILL CAPPED)" if m.n_iter_ == kw["max_iter"] else " (converged)"
        D = C.with_columns(pl.Series("p", m.predict(Xap)))
        top = (D.sort(["user_id", "p"], descending=[False, True])
                 .group_by("user_id", maintain_order=True).head(TOPK)
                 .select("user_id", "item_id", "is_hist"))
        hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
            pl.col("frac").fill_null(0.0))
        off = hit["frac"].sum() / active / TOPK
        hs = hit.filter(pl.col("is_hist") == 1)["frac"].sum()
        ns = hit.filter(pl.col("is_hist") == 0)["frac"].sum()
        print(f"{label:22s} {m.n_iter_:4d} iters{capped}  offline {off:.5f}  "
              f"LB~{off*1.29:.4f}  ({off-BASE:+.5f})  hist {hs:,.0f} new {ns:,.0f}",
              flush=True)

    print(f"\nbar to ship: {BASE+0.005:.5f}")


if __name__ == "__main__":
    main()
