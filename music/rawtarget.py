"""The training target is quantised to {0,.25,.5,.75,1} before the model ever
sees it. The METRIC quantises, but for RANKING that throws away information: a
13% listen and a 37% listen both become 0.25, yet they say different things
about how likely the next play is.

Train on the raw continuous fraction instead; score unchanged (the evaluator
quantises the truth as always). Aligned to cache_fix via build_candidates'
own decay_plays column, exactly as the decay experiment verified."""
import pathlib, numpy as np, polars as pl
from train4 import load_all
from rank4 import build_candidates
from train7 import WINDOWS, N_POOL

CACHE = pathlib.Path("cache_rawy"); CACHE.mkdir(exist_ok=True)
inter, meta, artists, genres, users = load_all()

def raw_target(lo, hi):
    return (inter.filter((pl.col("d") >= pl.lit(lo)) & (pl.col("d") < pl.lit(hi)))
            .filter(pl.col("user_id").is_in(users.implode()))
            .group_by(["user_id","item_id"]).agg(pl.col("listened_duration").sum().alias("s2"))
            .join(meta, on="item_id", how="left").filter(pl.col("track_duration") > 0)
            .with_columns((pl.col("s2")/pl.col("track_duration")).clip(0,1).alias("y_raw"))
            .select("user_id","item_id","y_raw"))

for cut, hi in WINDOWS:
    f = CACHE / f"tr_{cut}.parquet"
    if f.exists(): print(f"  hit {cut}", flush=True); continue
    print(f"  building {cut}", flush=True)
    C = (build_candidates(inter, meta, artists, genres, users, cut, n_pool=N_POOL)
         .sort(["user_id","item_id"]))
    ref = pl.read_parquet(f"cache_fix/tr_{cut}.parquet", columns=["decay_plays","y"])
    assert C.height == ref.height, f"{cut}: {C.height} vs {ref.height}"
    assert np.allclose(C["decay_plays"].to_numpy(), ref["decay_plays"].to_numpy(),
                       rtol=1e-9), f"{cut}: ROW ORDER MISMATCH -- cannot splice"
    E = (C.select("user_id","item_id")
         .join(raw_target(cut,hi), on=["user_id","item_id"], how="left")
         .with_columns(pl.col("y_raw").fill_null(0.0)))
    # sanity: quantising y_raw must reproduce the cached y exactly
    q = ((E["y_raw"]*4).round()/4).to_numpy()
    assert np.allclose(q, ref["y"].to_numpy(), atol=1e-9), f"{cut}: y_raw does not quantise to y"
    print(f"    aligned, and quantise(y_raw)==y on all {E.height:,} rows", flush=True)
    E.select("y_raw").write_parquet(f)
print("done")
