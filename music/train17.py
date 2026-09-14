"""#2 Seed ensembling -- average N models that differ only by random_state.

Never tried on this task. Averaging is the most reliable small gain in GBM work
and it is nearly free here: the variance it cancels is real, because
early_stopping carves a DIFFERENT positional validation split per seed, so each
model sees different data and stops at a different point.

That last part matters more than usual here. Today's sweep showed this model is
sharply sensitive to how much fitting it does -- the difference between 200 and
600 iterations is worth 0.005. So per-seed variation in where early stopping
lands is a real source of noise that averaging should suppress.

Reports each individual seed as well as the running ensemble, so we can see
whether the ensemble beats its own best member (a real gain) or merely beats the
average member (which would just be seed luck).
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
from train12 import cached

BASE = 0.29834
N_SEEDS = 6


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

    def score(p):
        D = C.with_columns(pl.Series("p", p))
        top = (D.sort(["user_id", "p"], descending=[False, True])
                 .group_by("user_id", maintain_order=True).head(TOPK)
                 .select("user_id", "item_id"))
        hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
            pl.col("frac").fill_null(0.0))
        return hit["frac"].sum() / active / TOPK

    acc = np.zeros(len(Xap), np.float64)
    singles = []
    for s in range(N_SEEDS):
        m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
            min_samples_leaf=200, l2_regularization=1.0, random_state=s,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=40).fit(Xtr, ytr)
        p = m.predict(Xap)
        singles.append(score(p))
        acc += p
        ens = score(acc / (s + 1))
        flag = "  <-- CLEARS BAR" if ens >= BASE + 0.005 else ""
        print(f"seed {s} (ran {m.n_iter_:3d})  single {singles[-1]:.5f}   "
              f"ensemble of {s+1}: {ens:.5f}  LB~{ens*1.29:.4f}  "
              f"({ens-BASE:+.5f}){flag}", flush=True)

    print(f"\nsingle-model spread: min {min(singles):.5f} max {max(singles):.5f} "
          f"(range {max(singles)-min(singles):.5f})")
    print(f"ensemble beats best single member: {score(acc/N_SEEDS) > max(singles)}")


if __name__ == "__main__":
    main()
