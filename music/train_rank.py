"""Train the LightGBM ranker and score it on the Aug 16-30 holdout."""

from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import polars as pl

from evaluate import TOPK, load, score, truth
from rank_model import FEATS, featurise, target
from v3 import trend_list
import v2

# Train on an earlier window so the model never sees holdout outcomes.
TRAIN_FEAT_CUT = "2025-08-01"
TRAIN_TGT_HI = "2025-08-16"
APPLY_CUT = "2025-08-16"


def main():
    inter = pl.read_csv(
        "interactions.csv",
        columns=["user_id", "item_id", "listened_duration", "listened_datetime"],
    ).with_columns(pl.col("listened_datetime").str.slice(0, 10).alias("d"))
    meta = pl.read_csv("item_metadata.csv", columns=["item_id", "track_duration"])
    users = pl.read_csv("test.csv")["user_id"]

    print("building training features...")
    Xtr = featurise(inter, meta, users, TRAIN_FEAT_CUT)
    ytr = target(inter, meta, users, TRAIN_FEAT_CUT, TRAIN_TGT_HI)
    tr = Xtr.join(ytr, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("y").fill_null(0.0)
    )
    print(f"  train pairs {tr.height:,}   positives {tr.filter(pl.col('y')>0).height:,}")

    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=100, l2_regularization=1.0, random_state=0,
    )
    m.fit(tr.select(FEATS).to_numpy(), tr["y"].to_numpy())

    print("applying to holdout window...")
    Xap = featurise(inter, meta, users, APPLY_CUT)
    pred = m.predict(Xap.select(FEATS).to_numpy())
    Xap = Xap.with_columns(pl.Series("p", pred))

    hist = (
        Xap.sort(["user_id", "p"], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .head(TOPK)
    )
    by_user = {}
    for u, i in hist.select("user_id", "item_id").iter_rows():
        by_user.setdefault(u, []).append(i)

    v2.CUT_DT = pl.lit(APPLY_CUT).str.to_date()
    train_only = inter.filter(pl.col("d") < APPLY_CUT)
    pop = trend_list(train_only, meta, 14, "completion", TOPK * 4)

    rows = []
    for u in users.to_list():
        chosen, seen = [], set()
        for src in (by_user.get(u, []), pop):
            for it in src:
                if len(chosen) >= TOPK:
                    break
                if it not in seen:
                    seen.add(it); chosen.append(it)
        for r, it in enumerate(chosen[:TOPK], 1):
            rows.append((u, it, r))
    recs = pl.DataFrame(rows, schema=["user_id", "item_id", "rank"], orient="row")

    _, holdout, meta2, users2 = load()
    t = truth(holdout, meta2, users2)
    s = score(recs, t, users2, verbose=True)
    print(f"\nGBM ranker -> {s:.5f}")
    print("   heuristic best = 0.21079   history-only oracle = 0.28428")


if __name__ == "__main__":
    main()
