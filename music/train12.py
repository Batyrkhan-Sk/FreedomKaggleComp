"""Separate MODEL quality from SLOT ALLOCATION -- train11 confounded them.

train11 fixed the split at 34/16 and lost 0.025. But the two sides moved in
opposite directions: new tracks earned 3,764 pts in 16 slots (baseline: 3,180 in
17) while history fell to 11,772 in 38 slots (baseline: 13,552 in 33). That is
the signature of an ALLOCATION failure, not a model failure -- the baseline
spends slots per user, and forcing 34 everywhere starves users whose value is
concentrated in history and wastes slots on users whose is not.

My oracle sweep found 30/20 ~= 34/16 and I read that as licence to fix the
split. It was not: that sweep compared fixed splits to each other and never
measured the value of ADAPTIVITY.

Both specialists regress the same target (expected quantised y), so their
outputs are already on a common scale and no fixed split is needed.

Arms, all on identical rows and features:
  A   baseline single model, joint top-50            (what ships)
  B2  two specialists, JOINT top-50 -- adaptive      (the corrected idea)
  D1  history ranking only, fixed 34 slots: A vs H   (pure ranking, no allocation)
  D2  new ranking only,     fixed 16 slots: A vs N   (pure ranking, no allocation)

D1/D2 are the diagnostics that say whether each specialist is actually a better
RANKER, which train11 could not answer.

Caches the featurised frames to parquet so later iterations skip the ~5 min of
candidate building.
"""
import pathlib

import numpy as np
import polars as pl

from evaluate import TOPK, truth
from rank3 import target
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise
from train11 import HIST_FEATS, NEW_FEATS, fit

CACHE = pathlib.Path("cache_feats")
CACHE.mkdir(exist_ok=True)


def cached(name, build=None):
    f = CACHE / f"{name}.parquet"
    if f.exists():
        print(f"  cache hit {f.name}", flush=True)
        return pl.read_parquet(f)
    df = build()
    df.write_parquet(f)
    return df


def slots(df, k):
    return df.filter(pl.col("r") < k)


def report(top, t, active, label, base=None):
    hit = top.join(t, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("frac").fill_null(0.0))
    off = hit["frac"].sum() / active / TOPK
    d = f"  ({off-base:+.5f})" if base is not None else ""
    nh = 100 * top.filter(pl.col("is_hist") == 0).height / top.height
    print(f"{label:34s} offline {off:.5f}  LB~{off*1.29:.4f}  "
          f"new-slots {nh:4.1f}%{d}")
    return off


def main():
    inter, meta, artists, genres, users = load_all()

    parts = []
    for cut, hi in WINDOWS:
        def build(cut=cut, hi=hi):
            print(f"window {cut}", flush=True)
            C = featurise(inter, meta, artists, genres, users, cut)
            return (C.join(target(inter, meta, users, cut, hi),
                           on=["user_id", "item_id"], how="left")
                    .with_columns(pl.col("y").fill_null(0.0)).select(FEATS + ["y"]))
        parts.append(cached(f"tr_{cut}", build))
    tr = pl.concat(parts)

    C = cached("apply", lambda: featurise(inter, meta, artists, genres, users, APPLY))
    t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
    active = t["user_id"].n_unique()
    print(f"train {tr.height:,}  apply {C.height:,}  active {active}\n", flush=True)

    X = lambda d, f: d.select(f).to_numpy().astype(np.float32)
    mA = fit(X(tr, FEATS), tr["y"].to_numpy());          print("A fitted", flush=True)
    trh, trn = tr.filter(pl.col("is_hist") == 1), tr.filter(pl.col("is_hist") == 0)
    mH = fit(X(trh, HIST_FEATS), trh["y"].to_numpy());   print("H fitted", flush=True)
    mN = fit(X(trn, NEW_FEATS), trn["y"].to_numpy());    print("N fitted\n", flush=True)

    CA = C.with_columns(pl.Series("pA", mA.predict(X(C, FEATS))))
    Ch = CA.filter(pl.col("is_hist") == 1)
    Cn = CA.filter(pl.col("is_hist") == 0)
    Ch = Ch.with_columns(pl.Series("pS", mH.predict(X(Ch, HIST_FEATS))))
    Cn = Cn.with_columns(pl.Series("pS", mN.predict(X(Cn, NEW_FEATS))))

    rank = lambda d, col: d.sort(["user_id", col], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("user_id").alias("r"))

    # ---- A: baseline ----
    topA = (rank(CA, "pA").filter(pl.col("r") < TOPK)
            .select("user_id", "item_id", "is_hist"))
    base = report(topA, t, active, "A baseline (single, joint 50)")

    # ---- B2: two specialists, joint top-50, ADAPTIVE ----
    both = pl.concat([Ch.select("user_id", "item_id", "is_hist", "pS"),
                      Cn.select("user_id", "item_id", "is_hist", "pS")])
    topB2 = (rank(both, "pS").filter(pl.col("r") < TOPK)
             .select("user_id", "item_id", "is_hist"))
    report(topB2, t, active, "B2 specialists, joint 50 (adaptive)", base)

    # ---- D1 / D2: pure ranking quality, allocation held fixed ----
    print()
    for side, df, k, name in ((1, Ch, 34, "history"), (0, Cn, 16, "new")):
        tf = t.join(df.select("user_id", "item_id").with_columns(pl.lit(1).alias("k")),
                    on=["user_id", "item_id"], how="inner")
        for col, who in (("pA", "baseline model"), ("pS", "specialist   ")):
            sel = slots(rank(df, col), k).select("user_id", "item_id")
            pts = sel.join(t, on=["user_id", "item_id"], how="left")["frac"].sum()
            print(f"  D {name:8s} @{k:2d} slots  {who}  {pts:8,.0f} pts")

    print(f"\nbar to ship: {base:.5f} + 0.005 = {base+0.005:.5f}")


if __name__ == "__main__":
    main()
