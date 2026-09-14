"""Sweep the co-visitation hyperparameters -- never tuned, and it is the BEST feature.

n_seeds=20, window_days=30, pop_damp=0.5 are inherited defaults that have never
been swept. covis is the single best feature on this task and the main driver of
NEW-track ranking, which sits at 36% of oracle -- the weakest link, and the side
where +804 points (3rd place) would need 36% -> 46%.

This is the same move that paid today: max_iter=600 was an inherited default
nobody had swept, and tuning it was worth +0.0055 real. Tuning the best
FEATURE's generation parameters is that move on a more important component.

Cheap proxy first. A full retrain per config costs ~12 min (covis is recomputed
per window); instead this scores each config by how well covis ALONE ranks new
tracks in the 16 slots the model spends on them. That is exactly the job covis
does inside the model, it needs no retraining, and it ranks the configs so only
the winner pays for a full confirmation.

Absolute numbers here will be far below the model's, because covis alone is not
the model -- only the RANKING of configs matters.
"""
import itertools

import numpy as np
import polars as pl
import scipy.sparse as sp

from covis import _index
from evaluate import CUT, load, truth

SLOTS = 16

train, hold, meta, users = load(CUT)
t = truth(hold, meta, users)
hist = (train.filter(pl.col("user_id").is_in(users.implode()))
        .select("user_id", "item_id").unique())
new = t.join(hist, on=["user_id", "item_id"], how="anti")
TOTAL = new["frac"].sum()

cut_dt = pl.lit(CUT).str.to_date()
aged = train.with_columns(
    (cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))

recent14 = (aged.filter(pl.col("age") <= 14).join(meta, on="item_id", how="left")
            .filter(pl.col("track_duration") > 0)
            .with_columns((pl.col("listened_duration")/pl.col("track_duration"))
                          .clip(0, 1).alias("f")))
pool_ids = (recent14.group_by("item_id").agg(pl.col("f").sum().alias("s"))
            .sort("s", descending=True).head(1500)["item_id"].to_numpy())

truth_by_user = {}
for u, i, f in new.iter_rows():
    truth_by_user.setdefault(u, {})[i] = f
hist_by_user = {u: set(v) for u, v in
                hist.group_by("user_id").agg(pl.col("item_id")).iter_rows()}
print(f"new-track value {TOTAL:,.0f} pts   pool {len(pool_ids)}   slots {SLOTS}")
print(f"{'window':>7} {'seeds':>6} {'damp':>5}   captured   share of new value", flush=True)

for wd, ns, pd_ in [(120,40,0.3),(120,40,0.2),(120,40,0.4),(120,20,0.3),(169,40,0.3)]:
    r = aged.filter((pl.col("age") <= wd) & (pl.col("listened_duration") >= 30)) \
            .select("user_id", "item_id", "age")
    uu, umap = _index(r["user_id"].to_numpy())
    ii, imap = _index(r["item_id"].to_numpy())
    rows = np.array([umap[u] for u in r["user_id"].to_list()], np.int32)
    cols = np.array([imap[i] for i in r["item_id"].to_list()], np.int32)
    R = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(len(uu), len(ii)))
    R.sum_duplicates(); R.data[:] = 1.0
    pop = np.asarray(R.sum(0)).ravel()
    damp = 1.0 / np.power(np.maximum(pop, 1.0), pd_)
    Rt = R.T.tocsr()

    seeds = (r.filter(pl.col("user_id").is_in(users.implode()))
             .group_by(["user_id", "item_id"]).agg(pl.col("age").min())
             .sort(["user_id", "age", "item_id"])
             .group_by("user_id", maintain_order=True).head(ns))
    by_user = {}
    for u, i, _ in seeds.iter_rows():
        by_user.setdefault(u, []).append(imap[i])

    pcols = np.array([imap.get(int(p), -1) for p in pool_ids])
    ok = pcols >= 0
    got = 0.0
    test = [u for u in users.to_list() if u in by_user]
    for s in range(0, len(test), 250):
        block = test[s:s+250]
        rr, cc = [], []
        for k, u in enumerate(block):
            for i in by_user[u]:
                rr.append(i); cc.append(k)
        P = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)),
                          shape=(len(ii), len(block)))
        S = np.asarray((Rt @ (R @ P)).todense()) * damp[:, None]
        for k, u in enumerate(block):
            tb = truth_by_user.get(u)
            if not tb:
                continue
            sc = np.full(len(pool_ids), -np.inf)
            sc[ok] = S[pcols[ok], k]
            h = hist_by_user.get(u, ())
            for j, it in enumerate(pool_ids):
                if int(it) in h:
                    sc[j] = -np.inf
            for j in np.argpartition(-sc, SLOTS)[:SLOTS]:
                if np.isfinite(sc[j]):
                    got += tb.get(int(pool_ids[j]), 0.0)
    print(f"{wd:7d} {ns:6d} {pd_:5.1f}   {got:8,.0f}   {got/TOTAL:6.2%}", flush=True)
