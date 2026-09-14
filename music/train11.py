"""Two specialist rankers instead of one generalist -- paired against the baseline.

WHY (measured, `oracle_split.py`): history capture is 44.1% against an oracle of
56.8% at the SAME 34-slot budget, so history is a RANKING failure, not a budget
ceiling -- 12.7pp = +0.088 of score sitting unclaimed. Nine of the ten closed
levers aimed at new tracks; only replay cadence ever aimed here.

WHY IT SHOULD WORK (structural, from rank4.build_candidates):
  * new candidates have all 15 PAIR features fill_null(0.0)   -> structurally zero
  * history candidates have covis/latent = 0 (computed over the pool only)
so a single tree must branch on `is_hist` before any of those features carries
information, and re-spend that split at every branch. Two specialists remove the
regime mixture, and each drops the features that are constant on its own side.

NOT the failed two-stage: that decomposed by OUTCOME (P(listen) x E(frac)) on the
same joint problem. This decomposes by CANDIDATE TYPE, and it is only legal
because the optimal split is known and flat (oracle sweep: 30/20 = 0.5883 vs
34/16 = 0.5862), so the two scores never need to be comparable.

Both arms are scored in ONE run on identical features and identical rows: the
harness has ~0.001 run-to-run noise and a +-0.008 SE, so a paired comparison is
the only kind worth making here. Bar for shipping is +0.005 offline.
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK, truth
from rank3 import target
from rank4 import PAIR
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise

COVIS = ["covis", "covis_rank", "latent", "latent_rank"]
# each side drops what is structurally constant on that side
HIST_FEATS = [f for f in FEATS if f not in COVIS + ["is_hist"]]
NEW_FEATS = [f for f in FEATS if f not in PAIR + ["is_hist"]]


def fit(X, y, seed=0):
    m = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, random_state=seed, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40)
    m.fit(X, y)
    return m


def evaluate(top, t, active, label):
    hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("frac").fill_null(0.0))
    off = hit["frac"].sum() / active / TOPK
    nh = 100 * top.filter(pl.col("is_hist") == 0).height / top.height
    hs = hit.filter(pl.col("is_hist") == 1)
    ns = hit.filter(pl.col("is_hist") == 0)
    print(f"{label:28s} offline {off:.5f}  LB~{off*1.29:.4f}  "
          f"new-slots {nh:4.1f}%  hist {hs['frac'].sum():7,.0f} "
          f"new {ns['frac'].sum():6,.0f} pts")
    return off


def main():
    inter, meta, artists, genres, users = load_all()

    parts = []
    for cut, hi in WINDOWS:
        print(f"window {cut}", flush=True)
        C = featurise(inter, meta, artists, genres, users, cut)
        p = C.join(target(inter, meta, users, cut, hi), on=["user_id", "item_id"],
                   how="left").with_columns(pl.col("y").fill_null(0.0))
        parts.append(p.select(FEATS + ["y"]))
    tr = pl.concat(parts)
    print(f"train rows {tr.height:,}  "
          f"(hist {tr.filter(pl.col('is_hist')==1).height:,} / "
          f"new {tr.filter(pl.col('is_hist')==0).height:,})", flush=True)

    C = featurise(inter, meta, artists, genres, users, APPLY)
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()
    print(f"apply rows {C.height:,}  active users {active}\n", flush=True)

    # ---------------- arm A: the shipping model, one regressor, joint top-50 ----
    mA = fit(tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy())
    print(f"A fitted {mA.n_iter_} iters", flush=True)
    CA = C.with_columns(pl.Series("p", mA.predict(C.select(FEATS).to_numpy().astype(np.float32))))
    topA = (CA.sort(["user_id", "p"], descending=[False, True])
              .group_by("user_id", maintain_order=True).head(TOPK)
              .select("user_id", "item_id", "is_hist"))
    base = evaluate(topA, t, active, "A single model (baseline)")

    # ---------------- arm B: two specialists, fixed split ----------------------
    trh = tr.filter(pl.col("is_hist") == 1)
    trn = tr.filter(pl.col("is_hist") == 0)
    mH = fit(trh.select(HIST_FEATS).to_numpy().astype(np.float32), trh["y"].to_numpy())
    print(f"B hist  fitted {mH.n_iter_} iters on {trh.height:,} rows, "
          f"{len(HIST_FEATS)} feats", flush=True)
    mN = fit(trn.select(NEW_FEATS).to_numpy().astype(np.float32), trn["y"].to_numpy())
    print(f"B new   fitted {mN.n_iter_} iters on {trn.height:,} rows, "
          f"{len(NEW_FEATS)} feats\n", flush=True)

    Ch = C.filter(pl.col("is_hist") == 1)
    Cn = C.filter(pl.col("is_hist") == 0)
    Ch = Ch.with_columns(pl.Series("p", mH.predict(Ch.select(HIST_FEATS).to_numpy().astype(np.float32))))
    Cn = Cn.with_columns(pl.Series("p", mN.predict(Cn.select(NEW_FEATS).to_numpy().astype(np.float32))))
    Ch = Ch.sort(["user_id", "p"], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("user_id").alias("r"))
    Cn = Cn.sort(["user_id", "p"], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("user_id").alias("r"))

    for H in (30, 34, 38):
        h = Ch.filter(pl.col("r") < H)
        # users with fewer than H history items get their slack back as new slots
        short = (h.group_by("user_id").agg(pl.len().alias("nh"))
                 .with_columns((TOPK - pl.col("nh")).alias("need")))
        allu = pl.DataFrame({"user_id": users.to_list()}).join(short, on="user_id", how="left")
        allu = allu.with_columns(pl.col("need").fill_null(TOPK))
        n = (Cn.join(allu.select("user_id", "need"), on="user_id", how="left")
               .filter(pl.col("r") < pl.col("need")))
        top = pl.concat([h.select("user_id", "item_id", "is_hist"),
                         n.select("user_id", "item_id", "is_hist")])
        evaluate(top, t, active, f"B specialists {H}/{TOPK-H}")

    print(f"\nbar to ship: baseline {base:.5f} + 0.005 = {base+0.005:.5f} offline")


if __name__ == "__main__":
    main()
