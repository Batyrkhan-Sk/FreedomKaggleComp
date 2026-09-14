"""#1 Sample weighting -- attack the loss/metric divergence directly.

Today's capacity sweep established the mechanism: the target is ~98% zeros, so
continued MSE fitting buys accuracy on the near-zero mass and pays for it in the
top-50 ordering that actually scores. Reducing capacity (600 -> 200 iters at
lr .03) recovered +0.0054 by simply doing LESS of that fitting.

Sample weighting attacks the same thing at its source rather than by starving
the model: upweight the rows that carry score so the loss cares about ranking
the valuable candidates rather than nailing the zeros. If the diagnosis is
right this should stack with the capacity fix, because it is a different
instrument on the same problem.

Baseline is the SHIPPED tuned config (200 x lr.03) on deterministic features,
offline 0.29834 = LB 0.38333. Bar is +0.005 as always.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

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
    pos = ytr > 0
    print(f"train {tr.height:,}  positives {pos.sum():,} ({pos.mean():.2%})")
    print(f"baseline (shipped 200 x lr.03) {BASE:.5f}   bar {BASE+0.005:.5f}\n", flush=True)

    for w in (1.0, 2.0, 5.0, 10.0, 25.0):
        sw = np.where(pos, w, 1.0).astype(np.float64)
        m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
            min_samples_leaf=200, l2_regularization=1.0, random_state=0,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=40).fit(Xtr, ytr, sample_weight=sw)
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
        print(f"pos weight {w:5.1f}  (ran {m.n_iter_:3d})  offline {off:.5f}  "
              f"LB~{off*1.29:.4f}  ({off-BASE:+.5f})  hist {hs:,.0f} new {ns:,.0f}{flag}",
              flush=True)


if __name__ == "__main__":
    main()
