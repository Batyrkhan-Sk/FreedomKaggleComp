"""Does rank("min") make featurisation deterministic?

submit7.py reproduces its own output at only 83.58% set overlap. Cause:
rank("ordinal") gives tied values distinct ranks by ROW POSITION, and 42.4% of
covis / 28.1% of latent are exactly 0.0 -- a median of 360 tied rows per user
whose rank is an artifact of polars row emission.

rank("min") gives tied values the SAME rank: deterministic, and semantically
right (360 items with zero co-visitation are not ranked 1..360).

Featurises the SAME window twice per mode and compares the rank columns. Two
passes with "ordinal" should differ; two with "min" should be identical.
"""
import numpy as np
import polars as pl

from train4 import load_all
from rank4 import build_candidates
from covis import covis_features
from latent import latent_features

CUT = "2025-08-16"
inter, meta, artists, genres, users = load_all()


def featurise(mode):
    C = build_candidates(inter, meta, artists, genres, users, CUT, n_pool=1500)
    pool = C.filter(pl.col("is_hist") == 0)["item_id"].unique().to_list()
    C = (C.join(covis_features(inter, users, pool, CUT), on=["user_id", "item_id"], how="left")
          .join(latent_features(inter, users, pool, CUT), on=["user_id", "item_id"], how="left")
          .with_columns(pl.col("covis").fill_null(0.0), pl.col("latent").fill_null(0.0)))
    C = C.with_columns(
        pl.col("covis").rank(mode, descending=True).over("user_id").alias("covis_rank"),
        pl.col("latent").rank(mode, descending=True).over("user_id").alias("latent_rank"))
    # compare on a stable key so row order itself cannot mask the result
    return C.sort(["user_id", "item_id"]).select(
        "user_id", "item_id", "covis", "latent", "covis_rank", "latent_rank")


for mode in ("ordinal", "min"):
    a = featurise(mode)
    b = featurise(mode)
    if a.height != b.height:
        print(f"{mode:8s}: DIFFERENT ROW COUNTS {a.height:,} vs {b.height:,}")
        continue
    same_rows = a.equals(b)
    dc = (a["covis_rank"] != b["covis_rank"]).sum()
    dl = (a["latent_rank"] != b["latent_rank"]).sum()
    dv = (a["covis"] != b["covis"]).sum() + (a["latent"] != b["latent"]).sum()
    print(f"{mode:8s}: identical={same_rows}  covis_rank diffs {dc:,}  "
          f"latent_rank diffs {dl:,}  raw-value diffs {dv:,}  (of {a.height:,})",
          flush=True)
