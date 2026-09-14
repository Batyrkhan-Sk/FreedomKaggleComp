"""halflife=21.0 has never been swept -- and it is the profile that paid before
(max_iter=600 was an inherited default worth +0.0055). It drives decay_plays and
decay_full, and the control experiment found decay_full is the best no-model
history ranker.

Rather than tune ONE value, add SHORT (7d) and LONG (60d) scales as extra
features and let the model choose per user -- the same shape winning solutions
use, and it avoids betting the run on a single swept value.

Cheap path: build_candidates regenerates the (user,item) keys without the
expensive covis/latent work, and its own decay_plays column verifies that the
row order matches the cache exactly before anything is spliced.
"""
import pathlib, numpy as np, polars as pl
from train4 import load_all
from rank4 import build_candidates
from train7 import WINDOWS, APPLY, N_POOL

HL = (7.0, 60.0)
NEW = [f"{s}_{int(h)}" for h in HL for s in ("decay_plays", "decay_full")]
CACHE = pathlib.Path("cache_decay"); CACHE.mkdir(exist_ok=True)
inter, meta, artists, genres, users = load_all()

def extra(cut):
    C = (build_candidates(inter, meta, artists, genres, users, cut, n_pool=N_POOL)
         .sort(["user_id", "item_id"]))
    cut_dt = pl.lit(cut).str.to_date()
    h = (inter.filter(pl.col("d") < cut)
         .filter(pl.col("user_id").is_in(users.implode()))
         .join(meta, on="item_id", how="left").filter(pl.col("track_duration") > 0)
         .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"),
                       (pl.col("listened_duration")/pl.col("track_duration")).clip(0,1).alias("f")))
    aggs = []
    for hl in HL:
        w = 0.5 ** (pl.col("age")/hl)
        aggs += [w.sum().alias(f"decay_plays_{int(hl)}"),
                 (pl.col("f")*w).sum().alias(f"decay_full_{int(hl)}")]
    g = h.group_by(["user_id","item_id"]).agg(aggs)
    C = C.join(g, on=["user_id","item_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in NEW])
    return C.select(["decay_plays"] + NEW)

for tag, cut in [(f"tr_{c}", c) for c,_ in WINDOWS] + [("apply", APPLY)]:
    f = CACHE / f"{tag}.parquet"
    if f.exists(): print(f"  hit {tag}", flush=True); continue
    print(f"  building {tag}", flush=True)
    E = extra(cut)
    ref = pl.read_parquet(f"cache_fix/{tag}.parquet", columns=["decay_plays"])
    assert E.height == ref.height, f"{tag}: {E.height} vs {ref.height} rows"
    same = np.allclose(E["decay_plays"].to_numpy(), ref["decay_plays"].to_numpy(), rtol=1e-9)
    assert same, f"{tag}: ROW ORDER MISMATCH -- cannot splice"
    print(f"    aligned with cache_fix ({E.height:,} rows)", flush=True)
    E.select(NEW).write_parquet(f)
print("all aligned and written")
