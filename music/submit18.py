"""Task-2 submission 18 = submit17 + min_samples_leaf=1000.

min_samples_leaf was inherited at 200 and never swept on either metric. Sweep on the
46-feature holdout (200 iters, 3 seeds): 50 -> -0.00102, 200 -> 0, 1000 -> +0.00114,
3000 -> +0.00043, so 1000 is an INTERIOR optimum. Confirmed one-variable at the shipped
capacity (400 iters x 5 seeds) against final_val.log: ensemble 0.38090 -> 0.38173 (+0.00083).

Original submit17 header follows.

"""
"""

  deterministic covis  (reproducibility -- the shipped artifact was not)
+ covis@90d/min10s     (+0.00066: coverage 1010->1205 users)
+ 9 multi-statistic covis features (+0.00044)
+ 400 iters x lr.03 x 63 leaves    (metric-C capacity optimum; the recorded
                                    40-150 plateau was a point-weighted artifact)
+ 5-seed rank-averaged ensemble    (+0.00127, and removes the seed lottery)

Holdout: 0.38090 vs the shipped config's draw of 0.37698 (which scored LB 0.38351).
"""
import sys
import numpy as np, polars as pl
MSL = int(sys.argv[1]) if len(sys.argv)>1 else 1000
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK
from rank3 import target
from train4 import load_all
from train7 import FEATS, featurise
from covis import covis_features
from covis2 import covis_multi

WINDOWS = [("2025-07-01","2025-07-16"), ("2025-07-16","2025-07-31"),
           ("2025-08-01","2025-08-16"), ("2025-08-16","2025-08-31")]
APPLY = "2025-08-31"
OUT = "submission18.csv"
M9 = ["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
W2 = ["cv90","cv90_rank"]
ALL = FEATS + W2 + M9
inter, meta, artists, genres, users = load_all()

def feat(cut):
    C = featurise(inter, meta, artists, genres, users, cut).with_columns(
        pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
    pool = C.filter(pl.col("is_hist")==0)["item_id"].unique().to_list()
    cv = covis_features(inter, users, pool, cut, window_days=90, min_secs=10)
    if cv.height:
        cv = cv.rename({"covis":"cv90"}).with_columns(
            pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
        C = C.join(cv, on=["user_id","item_id"], how="left").with_columns(pl.col("cv90").fill_null(0.0))
    else:
        C = C.with_columns(pl.lit(0.0).alias("cv90"))
    C = C.with_columns(pl.col("cv90").rank("min", descending=True).over("user_id").alias("cv90_rank"))
    m = covis_multi(inter, users, pool, cut, prefix="cv")
    if m is not None:
        C = C.join(m, on=["user_id","item_id"], how="left")
        C = C.with_columns([pl.col(c).fill_null(9999.0 if c.endswith("_rank") else 0.0) for c in M9])
    else:
        C = C.with_columns([pl.lit(9999.0 if c.endswith("_rank") else 0.0).alias(c) for c in M9])
    return C.sort(["user_id","item_id"])

parts=[]
for cut,hi in WINDOWS:
    p = (feat(cut).join(target(inter,meta,users,cut,hi),on=["user_id","item_id"],how="left")
         .with_columns(pl.col("y").fill_null(0.0)).select(ALL+["y"]))
    parts.append(p); print(f"  window {cut}: {p.height:,}", flush=True)
tr = pl.concat(parts)
X = tr.select(ALL).to_numpy().astype(np.float32); y = tr["y"].to_numpy()
C = feat(APPLY); Xa = C.select(ALL).to_numpy().astype(np.float32)
print(f"  apply {C.height:,}", flush=True)

ranks=[]
for s in range(5):
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.03,max_leaf_nodes=63,
      min_samples_leaf=MSL,l2_regularization=1.0,random_state=s,early_stopping=True,
      validation_fraction=0.1,n_iter_no_change=40).fit(X,y)
    ranks.append(pl.Series(m.predict(Xa)).rank().to_numpy())
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
print(f"wrote {OUT}: {recs.height:,} rows, "
      f"new-slot {100*top.filter(pl.col('is_hist')==0).height/top.height:.1f}%")
