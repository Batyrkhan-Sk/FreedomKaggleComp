"""Recency-aware baselines.

Two hypotheses:
  1. Music consumption is trend-driven, so popularity measured over the last
     couple of weeks should beat all-time popularity.
  2. A user's *recent* rotation predicts next-fortnight listening far better
     than their all-time play counts, which are dominated by months-old habits.
"""

import polars as pl

from evaluate import CUT, TOPK, load, score, truth
from baselines import rank_from

CUT_DT = pl.lit(CUT).str.to_date()


def recent_popular(train, users, days=14, topk=TOPK):
    lo = (CUT_DT - pl.duration(days=days)).cast(pl.Utf8)
    top = (
        train.filter(pl.col("d") >= lo)
        .group_by("item_id")
        .agg(pl.col("user_id").n_unique().alias("n"))
        .sort("n", descending=True)
        .head(topk)["item_id"]
        .to_list()
    )
    return pl.DataFrame({
        "user_id": [u for u in users for _ in top],
        "item_id": top * len(users),
        "rank": list(range(1, topk + 1)) * len(users),
    })


def hist_recency(train, users, halflife=21.0, topk=TOPK):
    """Play counts decayed exponentially by age in days."""
    h = (
        train.filter(pl.col("user_id").is_in(users.implode()))
        .with_columns(
            (CUT_DT - pl.col("d").str.to_date()).dt.total_days().alias("age")
        )
        .with_columns((0.5 ** (pl.col("age") / halflife)).alias("w"))
        .group_by(["user_id", "item_id"])
        .agg(
            pl.col("w").sum().alias("score"),
            pl.col("listened_duration").sum().alias("secs"),
        )
    )
    return rank_from(h, ["score", "secs"], [True, True])


def blend(train, users, halflife=21.0, days=14, topk=TOPK):
    """User's decayed history first, then trending tracks they have not heard."""
    hist = hist_recency(train, users, halflife, topk)
    counts = hist.group_by("user_id").agg(pl.len().alias("n"))
    lo = (CUT_DT - pl.duration(days=days)).cast(pl.Utf8)
    pop = (
        train.filter(pl.col("d") >= lo)
        .group_by("item_id")
        .agg(pl.col("user_id").n_unique().alias("n"))
        .sort("n", descending=True)
        .head(topk * 3)["item_id"]
        .to_list()
    )
    seen = {}
    for u, i in hist.select("user_id", "item_id").iter_rows():
        seen.setdefault(u, set()).add(i)
    have = dict(counts.iter_rows())
    rows = []
    for u in users.to_list():
        r = have.get(u, 0) + 1
        s = seen.get(u, set())
        for it in pop:
            if r > topk:
                break
            if it not in s:
                rows.append((u, it, r))
                r += 1
    extra = pl.DataFrame(rows, schema=["user_id", "item_id", "rank"], orient="row")
    return pl.concat([hist, extra])


if __name__ == "__main__":
    train, holdout, meta, users = load()
    t = truth(holdout, meta, users)

    print("=== popularity window ===")
    for d in (7, 14, 30):
        print(f"recent popular {d}d", end=" ")
        print(f"-> {score(recent_popular(train, users, d), t, users, verbose=False):.5f}")

    print("\n=== history recency half-life ===")
    for hl in (7.0, 21.0, 60.0):
        print(f"hist halflife {hl:>4.0f}d", end=" ")
        print(f"-> {score(hist_recency(train, users, hl), t, users, verbose=False):.5f}")

    print("\n=== blended ===")
    for hl in (7.0, 21.0):
        for d in (7, 14):
            s = score(blend(train, users, hl, d), t, users, verbose=False)
            print(f"blend hl={hl:>4.0f}d pop={d:>2}d -> {s:.5f}")
