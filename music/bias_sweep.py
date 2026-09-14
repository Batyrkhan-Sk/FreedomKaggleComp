"""Where is the optimum allocation between repeat and discovery slots?

The original sweep only tested bonus >= 0 -- forcing MORE discovery -- because it
assumed the harness undervalued it, on the strength of co-visitation measuring
+0.002 offline and delivering +0.015 real. That inference was later identified as
noise over-interpreted into a rule (the standard error was +/-0.008), so the sweep
searched one side of zero for no good reason and every value it tried lost.

The slot diagnostic gives the opposite prior: history slots earn 0.269 points each
against 0.131 for new slots, so at the margin the allocation should move TOWARD
repeats, not away. Negative bonus tests that and has never been run.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from rank3 import target
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise

inter, meta, artists, genres, users = load_all()
parts = []
for cut, hi in WINDOWS:
    C = featurise(inter, meta, artists, genres, users, cut)
    parts.append(C.join(target(inter, meta, users, cut, hi), on=["user_id", "item_id"],
                        how="left").with_columns(pl.col("y").fill_null(0.0)).select(FEATS + ["y"]))
tr = pl.concat(parts)
m = HistGradientBoostingRegressor(
    max_iter=600, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
    l2_regularization=1.0, random_state=0, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=40)
m.fit(tr.select(FEATS).to_numpy(), tr["y"].to_numpy())

C = featurise(inter, meta, artists, genres, users, APPLY)
C = C.with_columns(pl.Series("p", m.predict(C.select(FEATS).to_numpy())))
t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
active = t["user_id"].n_unique()

print(f"{'bonus':>7} {'new-slot %':>11} {'offline':>9} {'proj LB':>9}")
for bonus in (-0.20, -0.12, -0.08, -0.05, -0.02, 0.0, 0.02):
    D = C.with_columns((pl.col("p") + bonus * (pl.col("is_hist") == 0)).alias("p2"))
    top = (D.sort(["user_id", "p2"], descending=[False, True])
             .group_by("user_id", maintain_order=True).head(TOPK)
             .select("user_id", "item_id", "is_hist"))
    frac_new = 100 * top.filter(pl.col("is_hist") == 0).height / top.height
    hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(pl.col("frac").fill_null(0.0))
    s = hit["frac"].sum() / active / TOPK
    print(f"{bonus:>7.2f} {frac_new:>10.1f}% {s:>9.5f} {s*1.29:>9.3f}")
