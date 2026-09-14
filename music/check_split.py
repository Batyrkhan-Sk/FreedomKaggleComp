"""Does the tuned config's slot split match what it did offline?

submission_tuned.csv emerges at 20.5% new slots; the baseline runs ~34%. Slot
allocation is a KNOWN-sensitive quantity here -- it was swept in both directions
and 34/16 was the measured peak -- so a config that drifts to ~40/10 on its own
deserves a look before it costs a submission.

The offline sweep scored the tuned config but never printed its split, so this
re-runs baseline and tuned on the cached offline window and prints both. If the
tuned config sits near 20% offline TOO, the shift is a property of the config
and the offline +0.005 already prices it in. If it sits near 34% offline and
only collapses on the real window, something window-specific is happening and
the offline number does not transfer.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
from train12 import cached

inter, meta, artists, genres, users = load_all()
tr = pl.concat([cached(f"tr_{cut}", None) for cut, _ in WINDOWS])
C = cached("apply", None)
t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
active = t["user_id"].n_unique()
Xtr = tr.select(FEATS).to_numpy().astype(np.float32)
ytr = tr["y"].to_numpy()
Xap = C.select(FEATS).to_numpy().astype(np.float32)

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
    newpct = 100 * top.filter(pl.col("is_hist") == 0).height / top.height
    print(f"{label}  offline {off:.5f}  new-slots {newpct:5.1f}%  "
          f"(= {50*newpct/100:.0f} of 50)", flush=True)

print("\nsubmission_tuned.csv (real window, 4 windows) emerged at 20.5% new slots")
