"""Item-item co-occurrence embeddings -- the one feature family the winners of
structurally identical competitions (OTTO, H&M) use and this pipeline lacks.

`latent` factorises the USER x ITEM matrix. This factorises the ITEM x ITEM
co-occurrence matrix under a PPMI weighting, which Levy & Goldberg showed is what
word2vec's skip-gram-with-negative-sampling implicitly computes. So this is
item2vec without gensim (which will not build on this Python).

Why it is not just covis again: covis is a raw damped count, 42% exactly zero.
A factorisation gives a DENSE similarity for every pair, including pairs that
never co-occur -- reached through the latent space. That is precisely the long
tail where new-track value lives.

Why the "order carries no signal" result does not kill it: that measured DIRECTED
transitions against their own symmetrised counts (-0.46%). This is symmetric by
construction, so that finding does not apply.
"""
import numpy as np, polars as pl, scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


def i2v_features(inter, users, pool_items, cut, window_days=60, n_factors=128,
                 min_secs=30, halflife=21.0, seed=0):
    cut_dt = pl.lit(cut).str.to_date()
    recent = (inter.filter(pl.col("d") < cut)
        .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
        .filter((pl.col("age") <= window_days) & (pl.col("listened_duration") >= min_secs))
        .select("user_id", "item_id", "age"))
    if recent.height == 0:
        return None
    uu = np.unique(recent["user_id"].to_numpy()); umap = {v: k for k, v in enumerate(uu.tolist())}
    ii = np.unique(recent["item_id"].to_numpy()); imap = {v: k for k, v in enumerate(ii.tolist())}
    r = np.fromiter((umap[u] for u in recent["user_id"].to_list()), np.int32, recent.height)
    c = np.fromiter((imap[i] for i in recent["item_id"].to_list()), np.int32, recent.height)
    R = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(uu), len(ii)))
    R.data[:] = 1.0
    R.sum_duplicates(); R.data[:] = 1.0

    C = (R.T @ R).tocoo()                       # item-item co-occurrence
    keep = C.row != C.col
    row, col, cnt = C.row[keep], C.col[keep], C.data[keep].astype(np.float64)
    tot = cnt.sum()
    pi = np.asarray(R.sum(0)).ravel().astype(np.float64) + 1.0
    # PPMI: log( p(i,j) / (p(i) p(j)) ), clipped at 0
    pmi = np.log((cnt / tot) / ((pi[row] / pi.sum()) * (pi[col] / pi.sum())) + 1e-12)
    v = np.clip(pmi, 0, None)
    M = sp.csr_matrix((v, (row, col)), shape=(len(ii), len(ii)))
    print(f"  i2v: {len(ii):,} items, ppmi nnz {M.nnz:,}", flush=True)

    U, S, _ = randomized_svd(M, n_components=n_factors, random_state=seed)
    E = (U * np.sqrt(S)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9

    # user vector = recency-weighted centroid of the items they played
    w = (0.5 ** (recent["age"].to_numpy() / halflife)).astype(np.float32)
    UV = np.zeros((len(uu), n_factors), np.float32)
    np.add.at(UV, r, E[c] * w[:, None])
    UV /= np.linalg.norm(UV, axis=1, keepdims=True) + 1e-9

    pool_idx = np.array([imap[i] for i in pool_items if i in imap], np.int32)
    pool_ids = np.array([i for i in pool_items if i in imap])
    if len(pool_idx) == 0:
        return None
    P = E[pool_idx]
    out_u, out_i, out_v = [], [], []
    ul = users.to_list()
    for s in range(0, len(ul), 256):
        blk = [u for u in ul[s:s + 256] if u in umap]
        if not blk: continue
        sc = UV[[umap[u] for u in blk]] @ P.T
        for j, u in enumerate(blk):
            out_u.append(np.full(len(pool_ids), u, np.int64))
            out_i.append(pool_ids); out_v.append(sc[j])
    if not out_u:
        return None
    return pl.DataFrame({"user_id": np.concatenate(out_u),
                         "item_id": np.concatenate(out_i),
                         "i2v": np.concatenate(out_v)}).with_columns(
        pl.col("i2v").rank("min", descending=True).over("user_id").alias("i2v_rank"))
