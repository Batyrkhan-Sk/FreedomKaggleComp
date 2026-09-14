"""#8 STACK the small wins on a clean, row-order-pinned baseline.

Metadata measured +0.0017..+0.0021 -- real, reproducible at two capacities, but
under the +0.005 bar alone. Every remaining lever looks to be worth about that
much individually, so the question is no longer "does X clear the bar" but
"do the independent ones ADD UP past it".

Metadata is item/user quality. Play-shape is per-pair listening behaviour. They
describe different things, so they have a real chance of stacking.

Row order is now pinned, so every arm here shares one baseline and the controls
that plagued the last three experiments are unnecessary -- differences are the
features, not the ordering.
"""
import pathlib

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from meta_feats import META_FEATS, add_meta, load_meta
from playshape import SHAPE_FEATS, add_shape, shape_features
from rank3 import target
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise

CACHE = pathlib.Path("cache_all"); CACHE.mkdir(exist_ok=True)
ALL = FEATS + META_FEATS + SHAPE_FEATS


def main():
    inter, meta, artists, genres, users = load_all()
    item_m, user_m = load_meta()

    def cached(name, build):
        f = CACHE / f"{name}.parquet"
        if f.exists():
            print(f"  cache hit {f.name}", flush=True); return pl.read_parquet(f)
        d = build(); d.write_parquet(f); return d

    def make(cut):
        C = featurise(inter, meta, artists, genres, users, cut)
        C = add_meta(C, item_m, user_m)
        return add_shape(C, shape_features(inter, meta, users, cut))

    parts = []
    for cut, hi in WINDOWS:
        def build(cut=cut, hi=hi):
            print(f"  featurising {cut}", flush=True)
            return (make(cut).join(target(inter, meta, users, cut, hi),
                                   on=["user_id", "item_id"], how="left")
                    .with_columns(pl.col("y").fill_null(0.0)).select(ALL + ["y"]))
        parts.append(cached(f"tr_{cut}", build))
    tr = pl.concat(parts)
    C = cached("apply", lambda: make(APPLY))
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()
    ytr = tr["y"].to_numpy()
    print(f"\ntrain {tr.height:,}  row order pinned  features up to {len(ALL)}\n", flush=True)

    res = {}
    for feats, tag in ((FEATS, "base (35)          "),
                       (FEATS + META_FEATS, "base+meta (49)     "),
                       (FEATS + SHAPE_FEATS, "base+shape (41)    "),
                       (ALL, "base+meta+shape(55)")):
        Xtr = tr.select(feats).to_numpy().astype(np.float32)
        Xap = C.select(feats).to_numpy().astype(np.float32)
        m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
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
        res[tag] = off
        b = res.get("base (35)          ")
        d = f"  ({off-b:+.5f})" if b is not None else "  <- new baseline"
        hs = hit.filter(pl.col("is_hist") == 1)["frac"].sum()
        ns = hit.filter(pl.col("is_hist") == 0)["frac"].sum()
        flag = "  CLEARS BAR" if b is not None and off - b >= 0.005 else ""
        print(f"{tag} offline {off:.5f}  LB~{off*1.29:.4f}{d}  "
              f"hist {hs:,.0f} new {ns:,.0f}{flag}", flush=True)


if __name__ == "__main__":
    main()
