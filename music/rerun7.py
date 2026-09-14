"""Real submissions from the latent+covis model: default and discovery-biased."""
import sys
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK
from rank3 import target
from train4 import load_all
import train7
from train7 import FEATS, featurise

WINDOWS = [("2025-07-01","2025-07-16"), ("2025-07-16","2025-07-31"),
           ("2025-08-01","2025-08-16"), ("2025-08-16","2025-08-31")]
APPLY = "2025-08-31"

inter, meta, artists, genres, users = load_all()
parts = []
for cut, hi in WINDOWS:
    C = featurise(inter, meta, artists, genres, users, cut)
    parts.append(C.join(target(inter, meta, users, cut, hi), on=["user_id","item_id"],
                        how="left").with_columns(pl.col("y").fill_null(0.0)).select(FEATS+["y"]))
    print(f"  window {cut}: {parts[-1].height:,}", flush=True)
tr = pl.concat(parts)
m = HistGradientBoostingRegressor(
    max_iter=600, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
    l2_regularization=1.0, random_state=0, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=40)
m.fit(tr.select(FEATS).to_numpy(), tr["y"].to_numpy())
print(f"fitted {m.n_iter_} iters", flush=True)

C = featurise(inter, meta, artists, genres, users, APPLY)
C = C.with_columns(pl.Series("p", m.predict(C.select(FEATS).to_numpy())))

for bonus, name in ((0.0, "rerun_latent.csv"), (0.05, "rerun_discovery.csv")):
    D = C.with_columns((pl.col("p") + bonus*(pl.col("is_hist")==0)).alias("p2"))
    top = (D.sort(["user_id","p2"], descending=[False,True])
             .group_by("user_id", maintain_order=True).head(TOPK)
             .with_columns(pl.int_range(pl.len()).over("user_id").add(1).alias("rank"))
             .select("user_id","item_id","rank","is_hist"))
    pct = 100*top.filter(pl.col("is_hist")==0).height/top.height
    recs = top.select("user_id","item_id","rank").sort(["user_id","rank"])
    n = recs.group_by("user_id").agg(pl.len().alias("n"))
    assert n["n"].min()==TOPK==n["n"].max(), f"bad counts {n['n'].min()}..{n['n'].max()}"
    assert recs.select("user_id","item_id").is_duplicated().sum()==0
    assert set(recs["user_id"].unique())==set(users)
    recs.with_row_index("id").select("id","user_id","item_id","rank").write_csv(name)
    print(f"wrote {name}: new-slot {pct:.1f}%")
