"""Hybrid: user's own rotation for repeat slots, personalised discovery for the rest.

The blend baseline fills leftover slots with the same global trending list for
every user. Here those slots go to CF candidates the user has *not* heard,
falling back to trending only if CF cannot fill them (or the user was dropped
from the CF matrix by the skip filter).
"""

import argparse

import numpy as np
import polars as pl

import cf
from evaluate import TOPK, load, score, truth
from v2 import hist_recency
import v2


def trending(train, days, n):
    lo = (v2.CUT_DT - pl.duration(days=days)).cast(pl.Utf8)
    return (
        train.filter(pl.col("d") >= lo)
        .group_by("item_id")
        .agg(pl.col("user_id").n_unique().alias("n"))
        .sort("n", descending=True)
        .head(n)["item_id"]
        .to_list()
    )


def build(train, users, halflife=21.0, n_hist=25, item_alpha=0.5, topn_users=200,
          pop_days=14, topk=TOPK, cut=None, min_secs=30):
    """n_hist caps how many slots the user's own history may claim."""
    hist = hist_recency(train, users, halflife, topk).filter(pl.col("rank") <= n_hist)
    hist_by_user = {}
    for u, i, r in hist.iter_rows():
        hist_by_user.setdefault(u, []).append(i)

    cf_recs = cf.recommend(train, users, halflife, item_alpha, topn_users,
                           exclude_seen=True, topk=topk * 2, cut=cut, min_secs=min_secs)
    cf_by_user = {}
    for u, i, r in cf_recs.iter_rows():
        cf_by_user.setdefault(u, []).append(i)

    pop = trending(train, pop_days, topk * 4)

    rows = []
    for u in users.to_list():
        chosen, seen = [], set()
        for src in (hist_by_user.get(u, []), cf_by_user.get(u, []), pop):
            for it in src:
                if len(chosen) >= topk:
                    break
                if it not in seen:
                    seen.add(it)
                    chosen.append(it)
            if len(chosen) >= topk:
                break
        for r, it in enumerate(chosen[:topk], 1):
            rows.append((u, it, r))
    return pl.DataFrame(rows, schema=["user_id", "item_id", "rank"], orient="row")


if __name__ == "__main__":
    train, holdout, meta, users = load()
    t = truth(holdout, meta, users)
    print("baseline blend (history + trending pad) = 0.20812\n")

    for n_hist in (15, 25, 40):
        s = score(build(train, users, n_hist=n_hist), t, users, verbose=False)
        print(f"hist_slots={n_hist:>2}  CF discovery -> {s:.5f}")
