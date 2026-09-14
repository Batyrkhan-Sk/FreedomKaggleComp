"""Re-confirm the tuning gain on DETERMINISTIC features.

The +0.00508 was measured on features built before the covis seed fix and the
rank("min") change. Those fixes alter the features, so the result does not carry
over on trust -- it has to be re-measured. This rebuilds the cache with the
fixed pipeline and re-runs the three configs that matter.

If the gain survives, submission_tuned.csv gets regenerated from the fixed
pipeline and the notebook can be verified byte-for-byte against it, which is
what the deliverable rule actually wants.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from rank3 import target
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise
from train12 import cached

inter, meta, artists, genres, users = load_all()
parts = []
for cut, hi in WINDOWS:
    def build(cut=cut, hi=hi):
        print(f"  window {cut}", flush=True)
        C = featurise(inter, meta, artists, genres, users, cut)
        return (C.join(target(inter, meta, users, cut, hi), on=["user_id", "item_id"],
                       how="left").with_columns(pl.col("y").fill_null(0.0))
                .select(FEATS + ["y"]))
    parts.append(cached(f"tr_{cut}", build))
tr = pl.concat(parts)
C = cached("apply", lambda: featurise(inter, meta, artists, genres, users, APPLY))
t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
active = t["user_id"].n_unique()
Xtr = tr.select(FEATS).to_numpy().astype(np.float32)
ytr = tr["y"].to_numpy()
Xap = C.select(FEATS).to_numpy().astype(np.float32)
print(f"\ntrain {tr.height:,}  apply {C.height:,}  active {active}")
print("(features now deterministic: covis seeds pinned, rank=min)\n", flush=True)

res = {}
for label, it, lr in (("baseline 600 x lr.06", 600, 0.06),
                      ("tuned    200 x lr.03", 200, 0.03),
                      ("tuned     40 x lr.06", 40, 0.06)):
    m = HistGradientBoostingRegressor(
        max_iter=it, learning_rate=lr, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, random_state=0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40).fit(Xtr, ytr)
    D = C.with_columns(pl.Series("p", m.predict(Xap)))
    top = (D.sort(["user_id", "p"], descending=[False, True])
             .group_by("user_id", maintain_order=True).head(TOPK)
             .select("user_id", "item_id", "is_hist"))
    hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("frac").fill_null(0.0))
    off = hit["frac"].sum() / active / TOPK
    res[label] = off
    hs = hit.filter(pl.col("is_hist") == 1)["frac"].sum()
    ns = hit.filter(pl.col("is_hist") == 0)["frac"].sum()
    print(f"{label}  offline {off:.5f}  LB~{off*1.29:.4f}  hist {hs:,.0f} new {ns:,.0f}",
          flush=True)

b = res["baseline 600 x lr.06"]
print()
for k, v in res.items():
    if k.startswith("tuned"):
        ok = "CLEARS BAR" if v - b >= 0.005 else "below bar"
        print(f"{k}: {v-b:+.5f} vs baseline  -> {ok}")
print(f"\n(pre-fix measurement was +0.00508 for 200x.03, +0.00510 for 40x.06)")
