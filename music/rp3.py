"""RP3-beta: a degree-normalised 3-step random walk over the co-listen graph.

The shipped `covis` feature is the signal the whole new-track side leans on, and
it has three structural defects that RP3-beta fixes at once:

  1. it uses only the 20 most recent seed items in a 30-day window, throwing
     away most of the user's profile;
  2. it damps popularity on the TARGET side only (1/pop**0.5) and applies NO
     user-degree normalisation, so co-occurrence is dominated by hyperactive
     users -- and the test users are power users (median 562 plays vs 5 for the
     population), so the graph is built mostly from broad, diffuse listening
     that links everything to everything;
  3. it is 1-hop.

RP3-beta is item -> user -> item with BOTH transitions row-normalised (a proper
random walk), then the target column divided by its degree**beta.  It is not
low-rank, which is why it is a different bet from ALS / SVD / item2vec (all
flat here): those compress the graph, this walks it.

  score(u, j) = sum_i Pui[u,i] * Piu[i,v] * Pui[v,j]   /  pop[j]**beta

Computed only for test users x pool items (history items score 0, and a pool
item is by construction unplayed by the user so there is no self-loop leak).
Chunked over users so the intermediate user-space matrix never materialises.

Emits beta=0.4 and beta=0 (plain P3-alpha) plus per-user ranks -- the column
scaling is post-hoc, so the second beta is nearly free.

Aligned to cache_fix via build_candidates' own decay_plays column, the same
check als_feat.py and build_uid.py use.
"""
import pathlib, numpy as np, polars as pl, scipy.sparse as sp
from train4 import load_all
from rank4 import build_candidates
from train7 import WINDOWS, APPLY, N_POOL

CACHE = pathlib.Path("cache_rp3"); CACHE.mkdir(exist_ok=True)
WIN, MIN_SECS, BETA, CHUNK = 180, 30, 0.4, 150
COLS = ["rp3", "rp3_rank", "rp3b0", "rp3b0_rank"]
inter, meta, artists, genres, users = load_all()


def _rownorm(M):
    d = np.asarray(M.sum(axis=1)).ravel()
    d[d == 0] = 1.0
    return sp.diags((1.0 / d).astype(np.float32)) @ M


def build(cut):
    C = (build_candidates(inter, meta, artists, genres, users, cut, n_pool=N_POOL)
         .sort(["user_id", "item_id"]))
    cut_dt = pl.lit(cut).str.to_date()
    r = (inter.filter(pl.col("d") < cut)
         .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
         .filter((pl.col("age") <= WIN) & (pl.col("listened_duration") >= MIN_SECS))
         .select("user_id", "item_id").unique())
    uu = np.unique(r["user_id"].to_numpy()); umap = {v: k for k, v in enumerate(uu.tolist())}
    ii = np.unique(r["item_id"].to_numpy()); imap = {v: k for k, v in enumerate(ii.tolist())}
    rows = np.fromiter((umap[u] for u in r["user_id"].to_list()), np.int32, r.height)
    cols = np.fromiter((imap[i] for i in r["item_id"].to_list()), np.int32, r.height)
    R = sp.csr_matrix((np.ones(r.height, np.float32), (rows, cols)), shape=(len(uu), len(ii)))
    print(f"    RP3 graph {R.shape} nnz {R.nnz:,}", flush=True)

    Pui = _rownorm(R).tocsr()                    # user -> item
    Piu = _rownorm(R.T.tocsr()).tocsr()          # item -> user
    pop = np.asarray(R.sum(axis=0)).ravel() + 1.0

    pool_ids = C.filter(pl.col("is_hist") == 0)["item_id"].unique().to_numpy()
    pool_idx = np.array([imap[i] for i in pool_ids.tolist() if i in imap], np.int32)
    pool_keep = np.array([i for i in pool_ids.tolist() if i in imap])
    Pui_pool = Pui[:, pool_idx].tocsc()
    damp = (1.0 / pop[pool_idx] ** BETA).astype(np.float32)

    tu = np.array([umap.get(u, -1) for u in users.to_list()], np.int64)
    S = np.zeros((len(users), len(pool_idx)), np.float32)
    for a in range(0, len(users), CHUNK):
        b = min(a + CHUNK, len(users))
        sel = tu[a:b]; ok = sel >= 0
        if not ok.any(): continue
        A = Pui[sel[ok]]                          # (chunk x items)
        S[np.arange(a, b)[ok]] = np.asarray((A @ Piu @ Pui_pool).todense(), np.float32)
    print(f"    walked; nonzero {100*(S>0).mean():.1f}% of user x pool cells", flush=True)

    long = pl.DataFrame({
        "user_id": np.repeat(users.to_numpy(), len(pool_keep)),
        "item_id": np.tile(pool_keep, len(users)),
        "rp3": (S * damp).ravel(),
        "rp3b0": S.ravel(),
    }).filter((pl.col("rp3") > 0) | (pl.col("rp3b0") > 0))

    out = (C.select("user_id", "item_id")
            .join(long, on=["user_id", "item_id"], how="left")
            .with_columns(pl.col("rp3").fill_null(0.0), pl.col("rp3b0").fill_null(0.0))
            .with_columns(
                pl.col("rp3").rank("min", descending=True).over("user_id").alias("rp3_rank"),
                pl.col("rp3b0").rank("min", descending=True).over("user_id").alias("rp3b0_rank")))
    return C, out.select(COLS)


for tag, cut in [(f"tr_{c}", c) for c, _ in WINDOWS] + [("apply", APPLY)]:
    f = CACHE / f"{tag}.parquet"
    if f.exists(): print(f"  hit {tag}", flush=True); continue
    print(f"  building {tag}", flush=True)
    C, E = build(cut)
    ref = pl.read_parquet(f"cache_fix/{tag}.parquet", columns=["decay_plays"])
    assert C.height == ref.height, f"{tag}: {C.height} vs {ref.height}"
    assert np.allclose(C["decay_plays"].to_numpy(), ref["decay_plays"].to_numpy(),
                       rtol=1e-9), f"{tag}: ROW ORDER MISMATCH"
    assert E.height == C.height
    nz = (E["rp3"].to_numpy() != 0).mean() * 100
    print(f"    aligned; rp3 nonzero on {nz:.1f}% of rows", flush=True)
    E.write_parquet(f)
print("done")
