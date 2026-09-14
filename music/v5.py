"""Artist-affinity discovery.

The oracle says history-only tops out at 0.284, but leaderboard scores exceed
that, so the winning signal must predict genuinely new tracks. CF (taste
neighbours) scored 0.04 and lost badly to global trending at 0.14.

The signal CF misses in a music domain is the *artist*. People follow artists:
a track you have never heard, by someone whose catalogue you play constantly,
is far likelier to get a full listen than a generic chart hit.
"""

import polars as pl

from evaluate import TOPK, load, score, truth


def load_artists():
    """Full item metadata -- evaluate.load() only reads track_duration."""
    return pl.read_csv("item_metadata.csv", columns=["item_id", "artist_name"])
from v3 import trend_list
from v4 import user_scores
import v2


def artist_candidates(train, meta, users, halflife=21.0, per_artist=25, n_artists=40):
    meta = load_artists()
    """For each user: unheard tracks by the artists they actually listen to."""
    w = (
        train.filter(pl.col("user_id").is_in(users.implode()))
        .join(meta.select("item_id", "artist_name"), on="item_id", how="left")
        .filter(pl.col("artist_name").is_not_null())
        .with_columns((v2.CUT_DT - pl.col("d").str.to_date()).dt.total_days().alias("age"))
        .with_columns((0.5 ** (pl.col("age") / halflife)).alias("decay"))
    )
    aff = (
        w.group_by(["user_id", "artist_name"])
        .agg(pl.col("decay").sum().alias("a"))
        .sort(["user_id", "a"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .head(n_artists)
    )
    # Popularity of each track within its artist, measured recently.
    lo = (v2.CUT_DT - pl.duration(days=60)).cast(pl.Utf8)
    pop = (
        train.filter(pl.col("d") >= lo)
        .join(meta.select("item_id", "artist_name"), on="item_id", how="left")
        .filter(pl.col("artist_name").is_not_null())
        .group_by(["artist_name", "item_id"])
        .agg(pl.col("user_id").n_unique().alias("p"))
        .sort(["artist_name", "p"], descending=[False, True])
        .group_by("artist_name", maintain_order=True)
        .head(per_artist)
    )
    cand = (
        aff.join(pop, on="artist_name", how="inner")
        .with_columns((pl.col("a") * pl.col("p").log1p()).alias("s"))
        .sort(["user_id", "s"], descending=[False, True])
        .select("user_id", "item_id", "s")
    )
    return cand


def build(train, meta, users, halflife=21.0, n_hist=50, mode="decay_full",
          days=14, topk=TOPK, use_artist=True):
    g = user_scores(train, meta, users, halflife, mode)
    hist = (
        g.sort(["user_id", "s", "secs"], descending=[False, True, True])
        .group_by("user_id", maintain_order=True)
        .head(n_hist)
    )
    hist_by_user = {}
    for u, i in hist.select("user_id", "item_id").iter_rows():
        hist_by_user.setdefault(u, []).append(i)

    art_by_user = {}
    if use_artist:
        for u, i, _ in artist_candidates(train, meta, users, halflife).iter_rows():
            art_by_user.setdefault(u, []).append(i)

    pop = trend_list(train, meta, days, "completion", topk * 4)

    rows = []
    for u in users.to_list():
        chosen, seen = [], set()
        for src in (hist_by_user.get(u, []), art_by_user.get(u, []), pop):
            for it in src:
                if len(chosen) >= topk:
                    break
                if it not in seen:
                    seen.add(it); chosen.append(it)
            if len(chosen) >= topk:
                break
        for r, it in enumerate(chosen[:topk], 1):
            rows.append((u, it, r))
    return pl.DataFrame(rows, schema=["user_id", "item_id", "rank"], orient="row")


if __name__ == "__main__":
    train, holdout, meta, users = load()
    t = truth(holdout, meta, users)
    print("current best = 0.21079 (decay_full history + trending pad)")
    print("history-only ORACLE ceiling = 0.28428\n")

    for n_hist in (50, 40, 30, 20):
        s = score(build(train, meta, users, n_hist=n_hist), t, users, verbose=False)
        print(f"  hist_slots={n_hist:>2} + artist discovery -> {s:.5f}")
