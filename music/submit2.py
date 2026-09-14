"""Real submission from the stacked-window ranker: train through Aug 30, predict Sept 1-15."""
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import TOPK
from rank2 import FEATS, featurise, target
from train2 import load_all
from v3 import trend_list
import v2

# One more window is available now that we are not holding Aug 16-30 out.
WINDOWS = [("2025-06-15","2025-06-30"), ("2025-07-01","2025-07-16"),
           ("2025-07-16","2025-07-31"), ("2025-08-01","2025-08-16"),
           ("2025-08-16","2025-08-31")]
APPLY = "2025-08-31"

inter, meta, artists, users = load_all()
parts = []
for cut, hi in WINDOWS:
    X = featurise(inter, meta, artists, users, cut)
    y = target(inter, meta, users, cut, hi)
    parts.append(X.join(y, on=["user_id","item_id"], how="left")
                  .with_columns(pl.col("y").fill_null(0.0)).select(FEATS + ["y"]))
    print(f"  window {cut}: {parts[-1].height:,}")
tr = pl.concat(parts)
print(f"total training rows {tr.height:,}")

m = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200,
    l2_regularization=1.0, random_state=0, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=30)
m.fit(tr.select(FEATS).to_numpy(), tr["y"].to_numpy())
print(f"fitted {m.n_iter_} iters")

X = featurise(inter, meta, artists, users, APPLY)
X = X.with_columns(pl.Series("p", m.predict(X.select(FEATS).to_numpy())))
hist = X.sort(["user_id","p"], descending=[False,True]).group_by("user_id", maintain_order=True).head(TOPK)
by_user = {}
for u, i in hist.select("user_id","item_id").iter_rows():
    by_user.setdefault(u, []).append(i)

v2.CUT_DT = pl.lit(APPLY).str.to_date()
pop = trend_list(inter, meta, 14, "completion", TOPK*4)

rows = []
for u in users.to_list():
    chosen, seen = [], set()
    for src in (by_user.get(u, []), pop):
        for it in src:
            if len(chosen) >= TOPK: break
            if it not in seen:
                seen.add(it); chosen.append(it)
    for r, it in enumerate(chosen[:TOPK], 1):
        rows.append((u, it, r))
recs = pl.DataFrame(rows, schema=["user_id","item_id","rank"], orient="row").sort(["user_id","rank"])

n = recs.group_by("user_id").agg(pl.len().alias("n"))
assert n["n"].min() == 50 == n["n"].max(), f"counts {n['n'].min()}..{n['n'].max()}"
assert recs.select("user_id","item_id").is_duplicated().sum() == 0
assert set(recs["user_id"].unique()) == set(users)
out = recs.with_row_index("id").select("id","user_id","item_id","rank")
out.write_csv("submission_gbm.csv")
print(f"wrote submission_gbm.csv: {out.height:,} rows, {out['user_id'].n_unique()} users")
