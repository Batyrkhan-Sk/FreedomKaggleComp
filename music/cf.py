"""User-based collaborative filtering over a recency-weighted interaction matrix.

Rationale: only 38.9% of holdout listening is repeat, so the majority of the
score has to come from tracks the user has never played. The trending pad
recommends the same 50 items to everybody; CF personalises those slots by
finding users with overlapping taste and borrowing what they played.

Scored as: S = (R_test . R^T) . R  -- neighbour similarity, then their items.
Computed in chunks because the intermediate is 1500 x 314k.
"""

import numpy as np
import polars as pl
import scipy.sparse as sp

from evaluate import CUT, TOPK


def build_matrix(train, halflife=21.0, cut=None, min_secs=30):
    """Recency-weighted user x item sparse matrix.

    Interactions under `min_secs` are dropped: a 5-second play is a skip, and
    treating it as a positive signal pollutes the similarity structure.
    """
    cut_dt = pl.lit(cut or CUT).str.to_date()
    w = (
        train.filter(pl.col("listened_duration") >= min_secs)
        .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
        .with_columns((0.5 ** (pl.col("age") / halflife)).alias("w"))
        .group_by(["user_id", "item_id"])
        .agg(pl.col("w").sum().alias("w"))
    )
    uids = w["user_id"].unique().sort()
    iids = w["item_id"].unique().sort()
    umap = {u: k for k, u in enumerate(uids.to_list())}
    imap = {i: k for k, i in enumerate(iids.to_list())}
    rows = np.fromiter((umap[u] for u in w["user_id"].to_list()), dtype=np.int32, count=w.height)
    cols = np.fromiter((imap[i] for i in w["item_id"].to_list()), dtype=np.int32, count=w.height)
    vals = w["w"].to_numpy().astype(np.float32)
    R = sp.csr_matrix((vals, (rows, cols)), shape=(len(umap), len(imap)))
    return R, umap, imap, iids.to_numpy()


def normalise(R, item_alpha=0.5):
    """Down-weight blockbuster tracks, then L2-normalise each user row.

    Without the popularity discount, CF degenerates into a popularity ranker --
    everyone shares neighbours through the same handful of megahits.
    """
    pop = np.asarray(R.sum(axis=0)).ravel() + 1.0
    D = sp.diags((1.0 / pop**item_alpha).astype(np.float32))
    Rn = (R @ D).tocsr()
    norms = np.sqrt(Rn.multiply(Rn).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    Rn = sp.diags((1.0 / norms).astype(np.float32)) @ Rn
    return Rn.tocsr()


def recommend(train, users, halflife=21.0, item_alpha=0.5, topn_users=200,
              exclude_seen=False, chunk=200, topk=TOPK, cut=None, min_secs=30):
    R, umap, imap, iids = build_matrix(train, halflife, cut, min_secs)
    Rn = normalise(R, item_alpha)

    test = [u for u in users.to_list() if u in umap]
    rows_out = []
    seen_csr = R.tocsr()

    for start in range(0, len(test), chunk):
        block = test[start : start + chunk]
        idx = np.array([umap[u] for u in block])
        sims = (Rn[idx] @ Rn.T).toarray()          # block x n_users
        sims[np.arange(len(idx)), idx] = 0.0        # never be your own neighbour

        # keep only the strongest neighbours; the long tail is noise
        if topn_users and topn_users < sims.shape[1]:
            cut_at = np.partition(sims, -topn_users, axis=1)[:, -topn_users][:, None]
            sims[sims < cut_at] = 0.0

        scores = sp.csr_matrix(sims) @ R            # block x n_items

        for bi, u in enumerate(block):
            row = scores.getrow(bi).toarray().ravel()
            if exclude_seen:
                row[seen_csr.getrow(umap[u]).indices] = -np.inf
            top = np.argpartition(-row, topk)[:topk]
            top = top[np.argsort(-row[top])]
            for r, it in enumerate(top, 1):
                rows_out.append((u, int(iids[it]), r))

    return pl.DataFrame(rows_out, schema=["user_id", "item_id", "rank"], orient="row")
