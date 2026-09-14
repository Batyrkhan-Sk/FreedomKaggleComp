"""Co-visitation strength between a user's current rotation and each candidate.

The missing signal for new-track slots. Standalone user-based CF scored 0.04
and lost badly to trending, but that measured it as a *ranker*; here it becomes
one feature among thirty, letting the model use it only where it helps.

For candidate j and user u:  sum over u's recent seed tracks i of  co(i, j),
where co(i, j) counts users who played both recently, damped by item
popularity so blockbusters do not dominate every neighbourhood.

Computed only for pool (new) candidates -- history pairs already have far
stronger direct features.
"""

import numpy as np
import polars as pl
import scipy.sparse as sp


def _index(vals):
    uniq = np.unique(vals)
    return uniq, {v: k for k, v in enumerate(uniq.tolist())}


def covis_features(inter, users, pool_items, cut, window_days=30,
                   n_seeds=20, min_secs=30, chunk=250, pop_damp=0.5):
    cut_dt = pl.lit(cut).str.to_date()
    recent = (
        inter.filter(pl.col("d") < cut)
        .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
        .filter((pl.col("age") <= window_days) & (pl.col("listened_duration") >= min_secs))
        .select("user_id", "item_id", "age")
    )
    if recent.height == 0:
        return pl.DataFrame({"user_id": [], "item_id": [], "covis": []})

    uu, umap = _index(recent["user_id"].to_numpy())
    ii, imap = _index(recent["item_id"].to_numpy())
    rows = np.fromiter((umap[u] for u in recent["user_id"].to_list()),
                       dtype=np.int32, count=recent.height)
    cols = np.fromiter((imap[i] for i in recent["item_id"].to_list()),
                       dtype=np.int32, count=recent.height)
    R = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(uu), len(ii)))
    R.data[:] = 1.0                      # binary: played it recently or not

    # popularity damping, applied once to the item axis
    pop = np.asarray(R.sum(axis=0)).ravel() + 1.0
    Rd = (R @ sp.diags((1.0 / pop**pop_damp).astype(np.float32))).tocsr()

    pool_idx = np.array([imap[i] for i in pool_items if i in imap], dtype=np.int32)
    pool_ids = np.array([i for i in pool_items if i in imap])
    if len(pool_idx) == 0:
        return pl.DataFrame({"user_id": [], "item_id": [], "covis": []})
    Rpool = Rd[:, pool_idx].tocsc()

    # seeds: each test user's most recent distinct tracks
    seeds = (
        recent.filter(pl.col("user_id").is_in(users.implode()))
        .sort(["user_id", "age"])
        .unique(subset=["user_id", "item_id"], keep="first")
        .group_by("user_id", maintain_order=True)
        .head(n_seeds)
    )
    by_user = {}
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
        S = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                          shape=(len(block), R.shape[1]))
        A = (S @ Rd.T)                     # block x users -- shared listeners
        scores = np.asarray((A @ Rpool).todense())   # block x pool

        for bi, u in enumerate(block):
            row = scores[bi]
            nz = np.flatnonzero(row)
            if len(nz) == 0:
                continue
            out_u.append(np.full(len(nz), u, dtype=np.int64))
            out_i.append(pool_ids[nz])
            out_v.append(row[nz].astype(np.float32))

    if not out_u:
        return pl.DataFrame({"user_id": [], "item_id": [], "covis": []})
    return pl.DataFrame({
        "user_id": np.concatenate(out_u),
        "item_id": np.concatenate(out_i),
        "covis": np.concatenate(out_v),
    })
