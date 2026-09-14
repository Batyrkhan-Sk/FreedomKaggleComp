"""Which cache's covis does the CURRENT covis.py reproduce?
If cache_all is not reproducible, its +0.0030 is a ghost like the old
non-deterministic submission -- and the lead becomes 'sweep covis under C'."""
import polars as pl, numpy as np
from train4 import load_all
from rank4 import build_candidates
from covis import covis_features
from train7 import APPLY, N_POOL

inter, meta, artists, genres, users = load_all()
C = build_candidates(inter, meta, artists, genres, users, APPLY, n_pool=N_POOL)
pool = C.filter(pl.col("is_hist")==0)["item_id"].unique().to_list()
print(f"candidates {C.height:,}  pool {len(pool):,}", flush=True)
cv = covis_features(inter, users, pool, APPLY)
print(f"covis rows {cv.height:,}", flush=True)

key = C.select("user_id","item_id").join(cv, on=["user_id","item_id"], how="left").with_columns(
    pl.col("covis").fill_null(0.0)).sort(["user_id","item_id"])
fresh = key["covis"].to_numpy()
for d in ["cache_feats","cache_all"]:
    ref = pl.read_parquet(f"{d}/apply.parquet", columns=["user_id","item_id","covis"]).sort(["user_id","item_id"])
    assert ref["user_id"].to_numpy().tolist()==key["user_id"].to_numpy().tolist()
    v = ref["covis"].to_numpy()
    same = np.isclose(fresh, v, rtol=1e-9, atol=1e-12)
    print(f"  current covis.py vs {d:>12}: {same.mean()*100:6.2f}% identical  "
          f"(max abs diff {np.abs(fresh-v).max():.6f})")
