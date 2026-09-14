"""Covis decomposed into MULTIPLE statistics over the user's seed tracks.

The shipped feature is sum_i co(seed_i, cand) over 20 seeds -- one scalar. That
sum destroys the shape of the match: a candidate very close to the ONE track the
user is currently obsessed with scores the same as one weakly related to all 20.
Winning solutions on structurally identical tasks (OTTO, H&M) aggregate
similarity across the session into max / mean / top-k / position-weighted
statistics plus their ranks, not a single sum.

Also emits a SECOND, longer window as separate features rather than tuning one
window -- the recorded sweep made covis more accurate and more redundant with
recency (the "proxy trap"); keeping both scales as distinct features is the
standard fix and was never tried.
"""
import numpy as np, polars as pl, scipy.sparse as sp

def _index(vals):
    uniq = np.unique(vals)
    return uniq, {v: k for k, v in enumerate(uniq.tolist())}

def covis_multi(inter, users, pool_items, cut, window_days=30, n_seeds=20,
                min_secs=30, pop_damp=0.5, halflife=7.0, prefix="cv"):
    cut_dt = pl.lit(cut).str.to_date()
    recent = (inter.filter(pl.col("d") < cut)
        .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
        .filter((pl.col("age") <= window_days) & (pl.col("listened_duration") >= min_secs))
        .select("user_id", "item_id", "age"))
    if recent.height == 0: return None
    uu, umap = _index(recent["user_id"].to_numpy())
    ii, imap = _index(recent["item_id"].to_numpy())
    rows = np.fromiter((umap[u] for u in recent["user_id"].to_list()), dtype=np.int32, count=recent.height)
    cols = np.fromiter((imap[i] for i in recent["item_id"].to_list()), dtype=np.int32, count=recent.height)
    R = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(uu), len(ii)))
    R.data[:] = 1.0
    pop = np.asarray(R.sum(axis=0)).ravel() + 1.0
    Rd = (R @ sp.diags((1.0/pop**pop_damp).astype(np.float32))).tocsr()

    pool_idx = np.array([imap[i] for i in pool_items if i in imap], dtype=np.int32)
    pool_ids = np.array([i for i in pool_items if i in imap])
    if len(pool_idx) == 0: return None
    Rpool = Rd[:, pool_idx].tocsc()

    # deterministic seeds: dedup by aggregate, total sort key
    seeds = (recent.filter(pl.col("user_id").is_in(users.implode()))
             .group_by(["user_id","item_id"]).agg(pl.col("age").min().alias("age"))
             .sort(["user_id","age","item_id"])
             .group_by("user_id", maintain_order=True).head(n_seeds))
    by_user = {}
    for u, i, a in seeds.iter_rows():
        by_user.setdefault(u, ([], []))
        by_user[u][0].append(imap[i]); by_user[u][1].append(a)

    seed_items = np.unique(np.concatenate([np.array(v[0]) for v in by_user.values()]))
    smap = {int(v): k for k, v in enumerate(seed_items.tolist())}
    print(f"  {prefix}: {len(by_user)} users, {len(seed_items):,} distinct seed items", flush=True)
    # seed-item x pool similarity, computed ONCE
    SIM = np.asarray((Rd[:, seed_items].T @ Rpool).todense(), dtype=np.float32)

    out = {k: [] for k in ("user_id","item_id","sum","max","mean","top3","wsum","last")}
    for u in users.to_list():
        if u not in by_user: continue
        idx, ages = by_user[u]
        M = SIM[[smap[j] for j in idx]]                       # seeds x pool
        nz = np.flatnonzero(M.any(axis=0))
        if len(nz) == 0: continue
        M = M[:, nz]
        w = (0.5 ** (np.asarray(ages, dtype=np.float32)/halflife))[:, None]
        k = min(3, M.shape[0])
        out["user_id"].append(np.full(len(nz), u, dtype=np.int64))
        out["item_id"].append(pool_ids[nz])
        out["sum"].append(M.sum(0)); out["max"].append(M.max(0)); out["mean"].append(M.mean(0))
        out["top3"].append(np.sort(M, axis=0)[-k:].mean(0))
        out["wsum"].append((M*w).sum(0)); out["last"].append(M[0])
    if not out["user_id"]: return None
    df = pl.DataFrame({("user_id" if k=="user_id" else "item_id" if k=="item_id" else f"{prefix}_{k}"):
                       np.concatenate(v) for k, v in out.items()})
    return df.with_columns([
        pl.col(f"{prefix}_{s}").rank("min", descending=True).over("user_id").alias(f"{prefix}_{s}_rank")
        for s in ("max","wsum","top3")])
