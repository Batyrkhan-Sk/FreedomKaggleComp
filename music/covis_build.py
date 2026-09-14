import sys, polars as pl
from train4 import load_all
from rank4 import build_candidates
from covis import covis_features
from train7 import APPLY, N_POOL
inter, meta, artists, genres, users = load_all()
C = build_candidates(inter, meta, artists, genres, users, APPLY, n_pool=N_POOL)
pool = C.filter(pl.col("is_hist")==0)["item_id"].unique().to_list()
cv = covis_features(inter, users, pool, APPLY)
out = (C.select("user_id","item_id").join(cv, on=["user_id","item_id"], how="left")
        .with_columns(pl.col("covis").fill_null(0.0)).sort(["user_id","item_id"]))
out.write_parquet(sys.argv[1])
print("wrote", sys.argv[1], out.height)
