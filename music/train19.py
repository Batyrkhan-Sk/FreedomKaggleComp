"""#5 MORE TRAINING WINDOWS -- the one axis nothing has touched.

Three approaches died this round and they rule out the model as the constraint:
  #1 reshaping the objective (sample weights)   -> monotonic -0.0049
  #2 reducing variance (6-seed ensemble)        -> cannot beat a lucky seed
  #4 a different algorithm (LightGBM + blends)  -> best blend +0.0004, under noise

What none of them changed is the DATA. The model trains on 3 stacked 15-day
windows (Jul 1, Jul 16, Aug 1) while interactions.csv reaches back to 2025-02-28
-- roughly four months of unused history. More windows is more supervision, and
it is the only remaining axis that is not "the same data, fitted differently".

It also interacts with today's central finding. Capacity had to be cut hard
(600 -> 200 iters) because the model overfits 7.25M rows. With 2-3x the training
data the optimal capacity should move back UP, so this sweeps capacity jointly
rather than assuming 200 still holds -- otherwise a real gain from more data
would be masked by a now-wrong hyperparameter.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from rank3 import target
from train4 import load_all
from train7 import FEATS, APPLY, featurise
from train12 import cached

BASE = 0.29834
EXTRA = [("2025-05-01", "2025-05-16"), ("2025-05-16", "2025-05-31"),
         ("2025-06-01", "2025-06-16"), ("2025-06-16", "2025-07-01")]
CORE = [("2025-07-01", "2025-07-16"), ("2025-07-16", "2025-07-31"),
        ("2025-08-01", "2025-08-16")]


def main():
    inter, meta, artists, genres, users = load_all()

    def win(cut, hi):
        def build():
            print(f"  featurising {cut}", flush=True)
            C = featurise(inter, meta, artists, genres, users, cut)
            return (C.join(target(inter, meta, users, cut, hi),
                           on=["user_id", "item_id"], how="left")
                    .with_columns(pl.col("y").fill_null(0.0)).select(FEATS + ["y"]))
        return cached(f"tr_{cut}", build)

    core = [win(c, h) for c, h in CORE]
    extra = [win(c, h) for c, h in EXTRA]
    C = cached("apply", None)
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()
    Xap = C.select(FEATS).to_numpy().astype(np.float32)
    print(f"\nbaseline (3 windows, 200 iters) {BASE:.5f}  bar {BASE+0.005:.5f}\n", flush=True)

    for nwin, parts in ((3, core), (5, extra[2:] + core), (7, extra + core)):
        tr = pl.concat(parts)
        Xtr = tr.select(FEATS).to_numpy().astype(np.float32)
        ytr = tr["y"].to_numpy()
        for iters in (200, 400):
            m = HistGradientBoostingRegressor(
                max_iter=iters, learning_rate=0.03, max_leaf_nodes=63,
                min_samples_leaf=200, l2_regularization=1.0, random_state=0,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=40).fit(Xtr, ytr)
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
            print(f"{nwin} windows ({tr.height/1e6:5.2f}M rows)  {iters:3d} iters  "
                  f"offline {off:.5f}  LB~{off*1.29:.4f}  ({off-BASE:+.5f})  "
                  f"hist {hs:,.0f} new {ns:,.0f}{flag}", flush=True)


if __name__ == "__main__":
    main()
