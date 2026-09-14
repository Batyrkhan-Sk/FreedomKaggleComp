"""Latent-factor affinity between a user and each candidate track.

Co-visitation was the only signal that moved this task, but it can only relate
two tracks that were actually played together by someone. Latent factors relate
tracks that occupy the same region of taste space even when they never co-occur
-- which is exactly the long tail where new-track value lives (new-item value is
spread over ~15,000 tracks, and the top 1,000 hold only half of it).

Truncated SVD of the recency-weighted user x item matrix gives item factors; a
user is placed at the weighted centroid of the tracks they play, and affinity is
the dot product. This is added as a *feature*, not used as a ranker -- standalone
CF scored 0.04 here, while co-visitation as a feature was worth +0.015 real.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def _index(values: np.ndarray):
    uniq = np.unique(values)
    return uniq, {v: k for k, v in enumerate(uniq.tolist())}


def latent_features(inter: pl.DataFrame, users: pl.Series, pool_items: list[int],
                    cut: str, window_days: int = 90, n_factors: int = 96,
                    halflife: float = 30.0, min_secs: int = 30,
                    seed: int = 0) -> pl.DataFrame:
    cut_dt = pl.lit(cut).str.to_date()
    recent = (
        inter.filter(pl.col("d") < cut)
        .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
        .filter((pl.col("age") <= window_days) & (pl.col("listened_duration") >= min_secs))
        .with_columns((0.5 ** (pl.col("age") / halflife)).alias("w"))
        .group_by(["user_id", "item_id"])
        .agg(pl.col("w").sum().alias("w"))
    )
    if recent.height == 0:
        return pl.DataFrame({"user_id": [], "item_id": [], "latent": []})

    uu, umap = _index(recent["user_id"].to_numpy())
    ii, imap = _index(recent["item_id"].to_numpy())
    rows = np.fromiter((umap[u] for u in recent["user_id"].to_list()),
                       dtype=np.int32, count=recent.height)
    cols = np.fromiter((imap[i] for i in recent["item_id"].to_list()),
                       dtype=np.int32, count=recent.height)
    vals = np.log1p(recent["w"].to_numpy()).astype(np.float32)
    R = sp.csr_matrix((vals, (rows, cols)), shape=(len(uu), len(ii)))
    print(f"  latent matrix {R.shape}, nnz {R.nnz:,}")

    svd = TruncatedSVD(n_components=n_factors, random_state=seed)
    svd.fit(R)
    item_f = svd.components_.T.astype(np.float32)            # items x factors
    item_f /= np.linalg.norm(item_f, axis=1, keepdims=True) + 1e-9

    # place each user at the weighted centroid of the tracks they play
    user_f = np.asarray((R @ item_f))
    user_f /= np.linalg.norm(user_f, axis=1, keepdims=True) + 1e-9

    keep = [i for i in pool_items if i in imap]
    if not keep:
        return pl.DataFrame({"user_id": [], "item_id": [], "latent": []})
    pool_idx = np.array([imap[i] for i in keep], dtype=np.int32)
    pool_ids = np.array(keep)
    pool_f = item_f[pool_idx]

    test = [u for u in users.to_list() if u in umap]
    out_u, out_i, out_v = [], [], []
    for start in range(0, len(test), 500):
        block = test[start : start + 500]
        idx = np.array([umap[u] for u in block])
        scores = user_f[idx] @ pool_f.T                      # block x pool
        for bi, u in enumerate(block):
            out_u.append(np.full(len(pool_ids), u, dtype=np.int64))
            out_i.append(pool_ids)
            out_v.append(scores[bi].astype(np.float32))

    return pl.DataFrame({
        "user_id": np.concatenate(out_u),
        "item_id": np.concatenate(out_i),
        "latent": np.concatenate(out_v),
    })
