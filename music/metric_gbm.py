"""Score the SHIPPED config at the holdout under every candidate metric.
We know its real LB exactly: 0.38351 (submit15). Whichever variant lands there
with no free multiplier is the host's metric."""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY, featurise
from rank3 import target
from train12 import cached

inter, meta, artists, genres, users = load_all()
parts = []
for cut, hi in WINDOWS:
    parts.append(cached(f"tr_{cut}", lambda cut=cut, hi=hi: (
        featurise(inter, meta, artists, genres, users, cut)
        .join(target(inter, meta, users, cut, hi), on=["user_id","item_id"], how="left")
        .with_columns(pl.col("y").fill_null(0.0)).select(FEATS+["y"]))))
tr = pl.concat(parts)
C = cached("apply", lambda: featurise(inter, meta, artists, genres, users, APPLY))
t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
NA = t["user_id"].n_unique()
npl = t.group_by("user_id").agg(pl.len().alias("n"))
ideal = (t.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
          .head(TOPK).group_by("user_id").agg(pl.col("frac").sum().alias("ideal")))
print(f"train {tr.height:,}  apply {C.height:,}  active {NA}", flush=True)

m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
    min_samples_leaf=200, l2_regularization=1.0, random_state=2, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=40)
m.fit(tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy())
print(f"fitted {m.n_iter_}", flush=True)
D = C.with_columns(pl.Series("p", m.predict(C.select(FEATS).to_numpy().astype(np.float32))))
top = (D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True)
        .head(TOPK).select("user_id","item_id","is_hist"))
top.write_parquet("holdout_top50.parquet")

per = (top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
          .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
per = (npl.join(per,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
          .join(ideal,on="user_id",how="left"))
A_=per["s"].sum()/1500/TOPK; B_=per["s"].sum()/NA/TOPK
Cv=(per["s"]/per["n"].clip(upper_bound=TOPK)).sum()/NA
Dv=(per["s"]/per["ideal"]).sum()/NA
print(f"\n  KNOWN LB = 0.38351")
print(f"  A  sum/50 /1500      = {A_:.5f}")
print(f"  B  sum/50 /active    = {B_:.5f}   <- what remeasure.py prints")
print(f"  C  sum/min(50,n)/act = {Cv:.5f}")
print(f"  D  sum/ideal /active = {Dv:.5f}")
