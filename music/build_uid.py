"""Cache the user_id column aligned to the feature caches.

Ranking objectives (lambdarank / pairwise) need per-user GROUP boundaries, and
the cached training frames carry only FEATS+y. Rebuilding the keys via
build_candidates and verifying against cache_fix's own decay_plays column is the
same alignment check the decay and ALS features passed."""
import pathlib, numpy as np, polars as pl
from train4 import load_all
from rank4 import build_candidates
from train7 import WINDOWS, APPLY, N_POOL
CACHE = pathlib.Path("cache_uid"); CACHE.mkdir(exist_ok=True)
inter, meta, artists, genres, users = load_all()
for tag, cut in [(f"tr_{c}", c) for c,_ in WINDOWS] + [("apply", APPLY)]:
    f = CACHE / f"{tag}.parquet"
    if f.exists(): print(f"  hit {tag}", flush=True); continue
    C = (build_candidates(inter, meta, artists, genres, users, cut, n_pool=N_POOL)
         .sort(["user_id","item_id"]))
    ref = pl.read_parquet(f"cache_fix/{tag}.parquet", columns=["decay_plays"])
    assert C.height == ref.height, f"{tag}: {C.height} vs {ref.height}"
    assert np.allclose(C["decay_plays"].to_numpy(), ref["decay_plays"].to_numpy(),
                       rtol=1e-9), f"{tag}: ROW ORDER MISMATCH"
    # groups must be contiguous for lambdarank
    u = C["user_id"].to_numpy()
    assert (np.diff(u) >= 0).all(), f"{tag}: user_id not monotonic -- groups would be wrong"
    print(f"  {tag}: aligned, {len(np.unique(u))} contiguous user groups", flush=True)
    C.select("user_id").write_parquet(f)
print("done")
