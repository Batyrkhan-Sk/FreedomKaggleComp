"""Add ALS as an EXTRA feature alongside the existing SVD block.

Memory records ALS at 2.09x the shipped SVD block's strength, but it was only
ever measured as a standalone ranker / in comparison -- never added to the
model. SVD and ALS differ in exactly the way that matters for implicit feedback:
SVD treats every unobserved pair as a hard 0, ALS treats it as low confidence
and SOLVES for the user vector instead of projecting to a centroid.

Kept as an ADDITION, not a replacement, so the SVD block is untouched.
Aligned to cache_fix via build_candidates' own decay_plays column.
"""
import pathlib, numpy as np, polars as pl, scipy.sparse as sp
from train4 import load_all
from rank4 import build_candidates
from train7 import WINDOWS, APPLY, N_POOL
from als import als

CACHE = pathlib.Path("cache_als"); CACHE.mkdir(exist_ok=True)
WIN, F, REG, ALPHA, ITERS = 180, 64, 0.05, 40.0, 12
inter, meta, artists, genres, users = load_all()

def build(cut):
    C = (build_candidates(inter, meta, artists, genres, users, cut, n_pool=N_POOL)
         .sort(["user_id", "item_id"]))
    cut_dt = pl.lit(cut).str.to_date()
    r = (inter.filter(pl.col("d") < cut)
         .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
         .filter((pl.col("age") <= WIN) & (pl.col("listened_duration") >= 30))
         .group_by(["user_id","item_id"]).agg(pl.len().alias("w")))
    uu = np.unique(r["user_id"].to_numpy()); umap = {v:k for k,v in enumerate(uu.tolist())}
    ii = np.unique(r["item_id"].to_numpy()); imap = {v:k for k,v in enumerate(ii.tolist())}
    rows = np.fromiter((umap[u] for u in r["user_id"].to_list()), np.int32, r.height)
    cols = np.fromiter((imap[i] for i in r["item_id"].to_list()), np.int32, r.height)
    R = sp.csr_matrix((np.log1p(r["w"].to_numpy()).astype(np.float64), (rows, cols)),
                      shape=(len(uu), len(ii)))
    print(f"    ALS matrix {R.shape} nnz {R.nnz:,}", flush=True)
    U, V = als(R, factors=F, iters=ITERS, reg=REG, alpha=ALPHA, seed=0, verbose=False)
    ui = np.array([umap.get(u, -1) for u in C["user_id"].to_list()], np.int64)
    it = np.array([imap.get(i, -1) for i in C["item_id"].to_list()], np.int64)
    ok = (ui >= 0) & (it >= 0)
    vals = np.zeros(C.height, np.float32)
    vals[ok] = np.einsum('ij,ij->i', U[ui[ok]], V[it[ok]]).astype(np.float32)
    out = C.select("user_id").with_columns(pl.Series("als", vals))
    out = out.with_columns(pl.col("als").rank("min", descending=True).over("user_id").alias("als_rank"))
    return C, out.select("als", "als_rank")

for tag, cut in [(f"tr_{c}", c) for c, _ in WINDOWS] + [("apply", APPLY)]:
    f = CACHE / f"{tag}.parquet"
    if f.exists(): print(f"  hit {tag}", flush=True); continue
    print(f"  building {tag}", flush=True)
    C, E = build(cut)
    ref = pl.read_parquet(f"cache_fix/{tag}.parquet", columns=["decay_plays"])
    assert C.height == ref.height, f"{tag}: {C.height} vs {ref.height}"
    assert np.allclose(C["decay_plays"].to_numpy(), ref["decay_plays"].to_numpy(),
                       rtol=1e-9), f"{tag}: ROW ORDER MISMATCH"
    nz = (E["als"].to_numpy() != 0).mean() * 100
    print(f"    aligned; als nonzero on {nz:.1f}% of rows", flush=True)
    E.write_parquet(f)
print("done")
