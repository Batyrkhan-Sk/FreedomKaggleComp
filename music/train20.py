"""#7 METADATA FEATURES + the tuned capacity -- a combination never run.

train10.log records metadata features at offline 0.29408 against a then-baseline
of 0.29099 (bias.log, bonus 0.00) = +0.0031. Below the +0.005 bar, so it was
correctly never shipped. But that measurement is stale in three ways that all
point the same direction:

  1. measured on NON-DETERMINISTIC features -- the noise floor was contaminated
     by the covis-seed bug fixed today
  2. measured with the untuned 600-iteration config, which we now know overfits
  3. features and capacity are INDEPENDENT axes, so +0.0031 and +0.0054 may stack

Three approaches died this round (loss shaping, variance reduction, a different
algorithm) and together they say the MODEL is not the constraint. That is
precisely the argument for trying more FEATURES, which is what this is -- and
these 14 columns are already written, already validated, and simply never made
it into the shipped path.

Sweeps capacity jointly: 14 extra features change how much fitting is optimal,
and holding iters at 200 could mask a real gain behind a stale hyperparameter.
"""
import pathlib

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from meta_feats import META_FEATS, add_meta, load_meta
from rank3 import target
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise

BASE = 0.29834
ALL = FEATS + META_FEATS
CACHE = pathlib.Path("cache_meta"); CACHE.mkdir(exist_ok=True)


def main():
    inter, meta, artists, genres, users = load_all()
    item_m, user_m = load_meta()

    def cached(name, build):
        f = CACHE / f"{name}.parquet"
        if f.exists():
            print(f"  cache hit {f.name}", flush=True)
            return pl.read_parquet(f)
        d = build(); d.write_parquet(f); return d

    parts = []
    for cut, hi in WINDOWS:
        def build(cut=cut, hi=hi):
            print(f"  featurising {cut}", flush=True)
            C = add_meta(featurise(inter, meta, artists, genres, users, cut), item_m, user_m)
            return (C.join(target(inter, meta, users, cut, hi), on=["user_id", "item_id"],
                           how="left").with_columns(pl.col("y").fill_null(0.0))
                    .select(ALL + ["y"]))
        parts.append(cached(f"tr_{cut}", build))
    tr = pl.concat(parts)
    C = cached("apply", lambda: add_meta(
        featurise(inter, meta, artists, genres, users, APPLY), item_m, user_m))
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()
    ytr = tr["y"].to_numpy()
    print(f"\ntrain {tr.height:,}  features {len(ALL)} (was {len(FEATS)})")
    print(f"baseline (shipped, no metadata) {BASE:.5f}   bar {BASE+0.005:.5f}\n", flush=True)

    for feats, tag in ((FEATS, "no meta "), (ALL, "+metadata")):
        Xtr = tr.select(feats).to_numpy().astype(np.float32)
        Xap = C.select(feats).to_numpy().astype(np.float32)
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
            print(f"{tag}  {iters:3d} iters  offline {off:.5f}  LB~{off*1.29:.4f}  "
                  f"({off-BASE:+.5f})  hist {hs:,.0f} new {ns:,.0f}{flag}", flush=True)


if __name__ == "__main__":
    main()
