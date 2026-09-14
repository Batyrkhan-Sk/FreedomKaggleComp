"""Re-anchor a user's history to THEIR OWN last activity, not the cut date.

Returning users (absent in the previous window) capture 36.5% of their oracle vs
50.2% for continuously-active users -- and they are 31% of the holdout and a
LARGER share of the test set, where all 1,500 users are active by construction.

Mechanism: plays_7, plays_30, momentum, recency and every decay weight are
measured backwards from the CUT. For someone away 30 days these are all zero or
crushed, so the model ranks them with a fraction of its features. Anchoring to
the user's last active day restores the signal; for continuously-active users
the anchor IS the cut, so nothing changes for them.

Aligned to cache_fix via build_candidates' own decay_plays column."""
import pathlib, numpy as np, polars as pl
from train4 import load_all
from rank4 import build_candidates
from train7 import WINDOWS, APPLY, N_POOL
NEW = ["decay_plays_rel","decay_full_rel","recency_rel","plays_7_rel","plays_30_rel","u_gap"]
CACHE = pathlib.Path("cache_rel"); CACHE.mkdir(exist_ok=True)
inter, meta, artists, genres, users = load_all()

def build(cut):
    C = (build_candidates(inter, meta, artists, genres, users, cut, n_pool=N_POOL)
         .sort(["user_id","item_id"]))
    cut_dt = pl.lit(cut).str.to_date()
    h = (inter.filter(pl.col("d") < cut)
         .filter(pl.col("user_id").is_in(users.implode()))
         .join(meta, on="item_id", how="left").filter(pl.col("track_duration") > 0)
         .with_columns(pl.col("d").str.to_date().alias("dt"),
                       (pl.col("listened_duration")/pl.col("track_duration")).clip(0,1).alias("f")))
    # per-user anchor: their most recent listening day
    anch = h.group_by("user_id").agg(pl.col("dt").max().alias("u_last"))
    h = h.join(anch, on="user_id", how="left").with_columns(
        (pl.col("u_last") - pl.col("dt")).dt.total_days().alias("age_rel"))
    g = h.group_by(["user_id","item_id"]).agg(
        (0.5 ** (pl.col("age_rel")/21.0)).sum().alias("decay_plays_rel"),
        ((pl.col("f")) * (0.5 ** (pl.col("age_rel")/21.0))).sum().alias("decay_full_rel"),
        pl.col("age_rel").min().alias("recency_rel"),
        (pl.col("age_rel") <= 7).sum().alias("plays_7_rel"),
        (pl.col("age_rel") <= 30).sum().alias("plays_30_rel"))
    gap = anch.with_columns((cut_dt - pl.col("u_last")).dt.total_days().alias("u_gap")).select("user_id","u_gap")
    C = (C.join(g, on=["user_id","item_id"], how="left")
           .join(gap, on="user_id", how="left")
           .with_columns([pl.col(c).fill_null(0.0) for c in NEW[:-1]] + [pl.col("u_gap").fill_null(999)]))
    return C

for tag, cut in [(f"tr_{c}", c) for c,_ in WINDOWS] + [("apply", APPLY)]:
    f = CACHE / f"{tag}.parquet"
    if f.exists(): print(f"  hit {tag}", flush=True); continue
    print(f"  building {tag}", flush=True)
    C = build(cut)
    ref = pl.read_parquet(f"cache_fix/{tag}.parquet", columns=["decay_plays"])
    assert C.height == ref.height, f"{tag}: {C.height} vs {ref.height}"
    assert np.allclose(C["decay_plays"].to_numpy(), ref["decay_plays"].to_numpy(),
                       rtol=1e-9), f"{tag}: ROW ORDER MISMATCH"
    gaps = C.group_by("user_id").agg(pl.col("u_gap").first())["u_gap"]
    print(f"    aligned; user gap-days: median {gaps.median():.0f}, "
          f"share with gap>7: {(gaps>7).mean()*100:.0f}%", flush=True)
    C.select(NEW).write_parquet(f)
print("done")
