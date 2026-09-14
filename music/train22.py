"""Does a 10% better covis FEATURE make a better MODEL?

Swept covis generation params (never tuned; inherited 30/20/0.5) and found an
interior optimum on all three axes at 120 days / 40 seeds / damp 0.4, worth
+10.2% on covis's own new-track ranking (1,490 -> 1,642).

But covis is a FEATURE among 35, not the ranker. A 10% better feature does not
imply a 10% better model, and this task has repeatedly shown a signal can look
strong in isolation and add nothing once the GBM already has it. So: full
retrain, same everything else, against the shipped 0.29817 (LB 0.38351).
"""
import pathlib

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from rank3 import target
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise

SHIPPED = 0.29817
CACHE = pathlib.Path("cache_covis"); CACHE.mkdir(exist_ok=True)


def main():
    inter, meta, artists, genres, users = load_all()

    def cached(name, build):
        f = CACHE / f"{name}.parquet"
        if f.exists():
            print(f"  cache hit {f.name}", flush=True); return pl.read_parquet(f)
        d = build(); d.write_parquet(f); return d

    parts = []
    for cut, hi in WINDOWS:
        def build(cut=cut, hi=hi):
            print(f"  featurising {cut} (120d covis -- slower)", flush=True)
            return (featurise(inter, meta, artists, genres, users, cut)
                    .join(target(inter, meta, users, cut, hi),
                          on=["user_id", "item_id"], how="left")
                    .with_columns(pl.col("y").fill_null(0.0)).select(FEATS + ["y"]))
        parts.append(cached(f"tr_{cut}", build))
    tr = pl.concat(parts)
    C = cached("apply", lambda: featurise(inter, meta, artists, genres, users, APPLY))
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()
    Xtr = tr.select(FEATS).to_numpy().astype(np.float32)
    ytr = tr["y"].to_numpy()
    Xap = C.select(FEATS).to_numpy().astype(np.float32)
    print(f"\nshipped covis 30/20/0.5 = {SHIPPED:.5f} (LB 0.38351)")
    print(f"tuned   covis 120/40/0.4:\n", flush=True)

    for seed in (2, 0, 5):
        m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
            min_samples_leaf=200, l2_regularization=1.0, random_state=seed,
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
        flag = "  <-- CLEARS BAR" if off - SHIPPED >= 0.005 else ""
        print(f"  seed {seed}  offline {off:.5f}  LB~{off*1.286:.4f}  "
              f"({off-SHIPPED:+.5f})  hist {hs:,.0f} new {ns:,.0f}{flag}", flush=True)


if __name__ == "__main__":
    main()
