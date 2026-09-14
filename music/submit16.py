"""Submission from the REPRODUCIBLE pipeline: deterministic covis + 5-seed ensemble.

Two reasons this replaces submit15's artifact:
  1. submit15 ran on the non-deterministic covis, so its output cannot be
     reproduced by any notebook -- the deliverable-rule risk.
  2. rank-averaging 5 seeds measured +0.00127 over the member mean on the
     holdout, and removes the seed lottery (members spanned 0.37794..0.37998).
"""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK
from rank3 import target
from train4 import load_all
from train7 import FEATS, featurise

WINDOWS = [("2025-07-01","2025-07-16"), ("2025-07-16","2025-07-31"),
           ("2025-08-01","2025-08-16"), ("2025-08-16","2025-08-31")]
APPLY = "2025-08-31"
OUT = "submission_fixed_ens.csv"

inter, meta, artists, genres, users = load_all()
parts=[]
for cut,hi in WINDOWS:
    C = featurise(inter, meta, artists, genres, users, cut)
    parts.append(C.join(target(inter,meta,users,cut,hi),on=["user_id","item_id"],how="left")
                  .with_columns(pl.col("y").fill_null(0.0)).select(FEATS+["y"]))
    print(f"  window {cut}: {parts[-1].height:,}", flush=True)
tr = pl.concat(parts)
X = tr.select(FEATS).to_numpy().astype(np.float32); y = tr["y"].to_numpy()
C = featurise(inter, meta, artists, genres, users, APPLY)
Xap = C.select(FEATS).to_numpy().astype(np.float32)

ranks=[]
for s in range(5):
    m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,max_leaf_nodes=63,
      min_samples_leaf=200,l2_regularization=1.0,random_state=s,early_stopping=True,
      validation_fraction=0.1,n_iter_no_change=40).fit(X,y)
    ranks.append(pl.Series(m.predict(Xap)).rank().to_numpy())
    print(f"  fitted seed {s} ({m.n_iter_} iters)", flush=True)
C = C.with_columns(pl.Series("p", np.vstack(ranks).mean(axis=0)))

top=(C.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True)
      .head(TOPK).with_columns(pl.int_range(pl.len()).over("user_id").add(1).alias("rank")))
recs = top.select("user_id","item_id","rank").sort(["user_id","rank"])
cnt = recs.group_by("user_id").agg(pl.len().alias("n"))
assert cnt["n"].min()==TOPK==cnt["n"].max(), f"bad counts {cnt['n'].min()}..{cnt['n'].max()}"
assert recs.select("user_id","item_id").is_duplicated().sum()==0, "duplicate item per user"
assert set(recs["user_id"].unique())==set(users), "user set mismatch"
assert recs.height==len(users)*TOPK, f"expected {len(users)*TOPK}, got {recs.height}"
assert recs["rank"].min()==1 and recs["rank"].max()==TOPK
assert recs.null_count().sum_horizontal().item()==0, "nulls present"
recs.with_row_index("id").select("id","user_id","item_id","rank").write_csv(OUT)
pct = 100*top.filter(pl.col("is_hist")==0).height/top.height
print(f"wrote {OUT}: {recs.height:,} rows, new-slot {pct:.1f}%")
