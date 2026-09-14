"""Baseline recommenders, scored on the held-out final 15 days."""

import polars as pl

from evaluate import TOPK, load, score, truth


def rank_from(df, sort_cols, descending):
    """Take the per-user top-50 by the given ordering and attach ranks 1..50."""
    return (
        df.sort(["user_id"] + sort_cols, descending=[False] + descending)
        .group_by("user_id", maintain_order=True)
        .head(TOPK)
        .with_columns(pl.int_range(pl.len()).over("user_id").add(1).alias("rank"))
        .select("user_id", "item_id", "rank")
    )


def popular(train, users, topk=TOPK):
    """Global top-50 by number of distinct listeners -- same list for everyone."""
    top = (
        train.group_by("item_id")
        .agg(pl.col("user_id").n_unique().alias("n"))
        .sort("n", descending=True)
        .head(topk)["item_id"]
        .to_list()
    )
    return pl.DataFrame(
        {
            "user_id": [u for u in users for _ in top],
            "item_id": top * len(users),
            "rank": list(range(1, topk + 1)) * len(users),
        }
    )


def user_history(train, users, topk=TOPK):
    """Each user's own most-played tracks, replayed back to them."""
    h = (
        train.filter(pl.col("user_id").is_in(users.implode()))
        .group_by(["user_id", "item_id"])
        .agg(
            pl.len().alias("plays"),
            pl.col("listened_duration").sum().alias("secs"),
            pl.col("d").max().alias("last"),
        )
    )
    return rank_from(h, ["plays", "secs"], [True, True])


def history_then_popular(train, users, topk=TOPK):
    """User history first, padded with global popularity to fill 50 slots."""
    hist = user_history(train, users, topk)
    counts = hist.group_by("user_id").agg(pl.len().alias("n"))
    short = counts.filter(pl.col("n") < topk)
    if short.height == 0:
        return hist
    pop = (
        train.group_by("item_id")
        .agg(pl.col("user_id").n_unique().alias("n"))
        .sort("n", descending=True)
        .head(topk * 2)["item_id"]
        .to_list()
    )
    seen = {(u, i) for u, i in hist.select("user_id", "item_id").iter_rows()}
    rows = []
    for u, n in short.iter_rows():
        r = n + 1
        for it in pop:
            if r > topk:
                break
            if (u, it) not in seen:
                rows.append((u, it, r))
                r += 1
    return pl.concat([hist, pl.DataFrame(rows, schema=["user_id", "item_id", "rank"],
                                         orient="row")])


if __name__ == "__main__":
    train, holdout, meta, users = load()
    t = truth(holdout, meta, users)
    print(f"holdout truth pairs: {t.height:,}\n")

    for name, fn in [
        ("global popularity", popular),
        ("user history (plays)", user_history),
        ("history + popularity pad", history_then_popular),
    ]:
        recs = fn(train, users)
        s = score(recs, t, users)
        print(f"{name:28s} -> {s:.5f}\n")
