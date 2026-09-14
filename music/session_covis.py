"""Session-level co-visitation.

The existing co-visitation feature was the only change that moved this task
(+0.0147 real), but it counted co-occurrence at the *user* level: "the same
person played both tracks at some point across six months". That is a blunt
signal -- a heavy listener touches hundreds of tracks across many moods.

Tracks played in the same *sitting* are far more tightly related: same mood,
same playlist, same context. The interaction log has microsecond timestamps, and
the gap between consecutive plays by one user is a median of 47 seconds, so
sessions separate cleanly at a ten-minute threshold (~12 tracks per session).

score(user, candidate) = sum over the user's recent seed tracks of
                         (sessions containing both seed and candidate),
with popularity damping so that ubiquitous tracks do not sit in everyone's
neighbourhood.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

SESSION_GAP_SECONDS = 600


def build_sessions(inter: pl.DataFrame, cut: str, window_days: int = 60,
                   min_secs: int = 30) -> pl.DataFrame:
    """Label each interaction with a session id.

    Short plays are dropped first: a five-second sample is a skip, and letting
    skips define session boundaries (or contribute co-occurrence) adds noise.
    """
    cut_dt = pl.lit(cut).str.to_date()
    df = (
        inter.filter(pl.col("d") < cut)
        .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
        .filter((pl.col("age") <= window_days) & (pl.col("listened_duration") >= min_secs))
        .with_columns(pl.col("listened_datetime").str.to_datetime(strict=False).alias("ts"))
        .drop_nulls("ts")
        .sort(["user_id", "ts"])
    )
    gap = pl.col("ts").diff().over("user_id").dt.total_seconds()
    return df.with_columns(
        ((gap.is_null()) | (gap > SESSION_GAP_SECONDS)).cum_sum().alias("session")
    ).select("user_id", "item_id", "session")


def _index(values: np.ndarray):
    uniq = np.unique(values)
    return uniq, {v: k for k, v in enumerate(uniq.tolist())}


def session_covis_features(inter: pl.DataFrame, users: pl.Series, pool_items: list[int],
                           cut: str, window_days: int = 60, n_seeds: int = 30,
                           chunk: int = 200, pop_damp: float = 0.5) -> pl.DataFrame:
    sess = build_sessions(inter, cut, window_days)
    if sess.height == 0:
        return pl.DataFrame({"user_id": [], "item_id": [], "scovis": []})

    sids, smap = _index(sess["session"].to_numpy())
    iids, imap = _index(sess["item_id"].to_numpy())
    rows = np.fromiter((smap[s] for s in sess["session"].to_list()),
                       dtype=np.int32, count=sess.height)
    cols = np.fromiter((imap[i] for i in sess["item_id"].to_list()),
                       dtype=np.int32, count=sess.height)
    S = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(sids), len(iids)))
    S.data[:] = 1.0                       # a track counts once per session
    S.sum_duplicates()
    S.data[:] = np.minimum(S.data, 1.0)
    print(f"  sessions {S.shape[0]:,}  items {S.shape[1]:,}  nnz {S.nnz:,}")

    pop = np.asarray(S.sum(axis=0)).ravel() + 1.0
    Sd = (S @ sp.diags((1.0 / pop**pop_damp).astype(np.float32))).tocsr()

    keep = [i for i in pool_items if i in imap]
    if not keep:
        return pl.DataFrame({"user_id": [], "item_id": [], "scovis": []})
    pool_ids = np.array(keep)
    Spool = Sd[:, np.array([imap[i] for i in keep], dtype=np.int32)].tocsc()

    # seeds: the user's most-played recent tracks, which anchor the neighbourhood
    seeds = (
        sess.filter(pl.col("user_id").is_in(users.implode()))
        .group_by(["user_id", "item_id"])
        .agg(pl.len().alias("n"))
        .sort(["user_id", "n"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .head(n_seeds)
    )
    by_user: dict[int, list[int]] = {}
    for u, i, _ in seeds.iter_rows():
        by_user.setdefault(u, []).append(imap[i])

    test = [u for u in users.to_list() if u in by_user]
    out_u, out_i, out_v = [], [], []

    for start in range(0, len(test), chunk):
        block = test[start : start + chunk]
        r, c = [], []
        for bi, u in enumerate(block):
            for j in by_user[u]:
                r.append(bi); c.append(j)
        Q = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                          shape=(len(block), S.shape[1]))
        # kept sparse throughout: the session axis has ~10^6 entries and
        # densifying the intermediate would need tens of gigabytes
        scores = np.asarray(((Q @ Sd.T) @ Spool).todense())

        for bi, u in enumerate(block):
            nz = np.flatnonzero(scores[bi])
            if len(nz) == 0:
                continue
            out_u.append(np.full(len(nz), u, dtype=np.int64))
            out_i.append(pool_ids[nz])
            out_v.append(scores[bi][nz].astype(np.float32))

    if not out_u:
        return pl.DataFrame({"user_id": [], "item_id": [], "scovis": []})
    return pl.DataFrame({
        "user_id": np.concatenate(out_u),
        "item_id": np.concatenate(out_i),
        "scovis": np.concatenate(out_v),
    })
