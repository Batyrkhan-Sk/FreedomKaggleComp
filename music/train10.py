"""Do cadence features lift the score above the metadata best?

Same harness, same windows, same model as train9.py -- the only change is four
features derived from columns already present. Reports with and without so the
delta is attributable, and breaks the score down by slot kind, because the whole
argument is that this should help HISTORY ranking (44.1% capture) rather than
new-track discovery.

Standing rule on this task: the harness is NOT decision-grade below ~0.005
offline. Run-to-run noise is ~0.001 and three separate sub-0.005 deltas have
misled -- co-visitation's +0.002 read as a 6x discovery rule, and two-stage's
+0.0025 which lost 0.0034 for real. Require +0.005 before spending a submission.
"""
import numpy as np
import polars as pl

from cadence_feats import CADENCE_FEATS, add_cadence
from evaluate import TOPK, truth
from meta_feats import META_FEATS, add_meta, load_meta
from rank3 import target
from train4 import load_all
from train7 import FEATS as BASE, WINDOWS, APPLY, featurise
from train9 import two_stage

WITH_META = BASE + META_FEATS
ALL = WITH_META + CADENCE_FEATS


def main():
    inter, meta, artists, genres, users = load_all()
    item_m, user_m = load_meta()

    parts = []
    for cut, hi in WINDOWS:
        C = add_cadence(add_meta(
            featurise(inter, meta, artists, genres, users, cut), item_m, user_m))
        p = C.join(target(inter, meta, users, cut, hi), on=["user_id", "item_id"],
                   how="left").with_columns(pl.col("y").fill_null(0.0))
        parts.append(p.select(ALL + ["y"]))
        print(f"  window {cut}: {p.height:,}", flush=True)
    tr = pl.concat(parts)
    ytr = tr["y"].to_numpy()

    C = add_cadence(add_meta(
        featurise(inter, meta, artists, genres, users, APPLY), item_m, user_m))
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()

    out = {}
    for name, feats in (("metadata best (train9)", WITH_META), ("+ cadence", ALL)):
        p = two_stage(tr.select(feats).to_numpy().astype(np.float32), ytr,
                      C.select(feats).to_numpy().astype(np.float32))
        D = C.with_columns(pl.Series("p", p))
        top = (D.sort(["user_id", "p"], descending=[False, True])
                 .group_by("user_id", maintain_order=True).head(TOPK)
                 .select("user_id", "item_id", "is_hist"))
        hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
            pl.col("frac").fill_null(0.0))
        s = hit["frac"].sum() / active / TOPK
        out[name] = s
        newp = hit.filter(pl.col("is_hist") == 0)["frac"].sum()
        histp = hit.filter(pl.col("is_hist") == 1)["frac"].sum()
        print(f"\n{name:24s} offline {s:.5f}  (proj LB {s*1.29:.3f})")
        print(f"    history {histp:7.0f} pts  capture {histp/30946:5.1%}")
        print(f"    new     {newp:7.0f} pts  capture {newp/42321:5.1%}", flush=True)

    d = out["+ cadence"] - out["metadata best (train9)"]
    print(f"\ndelta {d:+.5f} offline  (projected LB {d*1.29:+.4f})")
    print("SHIP IT" if d >= 0.005 else
          "BELOW THE +0.005 BAR -- do not spend a submission (noise ~0.001, SE ~0.008)")


if __name__ == "__main__":
    main()
