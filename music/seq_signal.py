"""Does listening ORDER predict anything that co-occurrence does not?

SASRec's entire marginal value over the co-visitation feature already in the
model is directionality: co-visitation says "users who play A also play B",
a sequence model says "after A, users play B". If the directed transition
A->B carries no more signal than the undirected pair {A,B}, then order is noise
here and a sequence model has nothing to add -- and that is worth knowing in
twenty minutes rather than after five hours of training.

Test: score each user's unseen candidates from their last few tracks using
(a) directed transitions, (b) the same counts symmetrised, (c) popularity, and
compare how much real new-track value each surfaces in the holdout.
"""
import numpy as np
import polars as pl
from evaluate import CUT, load, truth

SEEDS, GAP_MIN, TOPN = 10, 60, 200


def main():
    inter, holdout, meta, users = load()
    t = truth(holdout, meta, users)
    hist = (inter.filter(pl.col("user_id").is_in(users.implode()))
                 .select("user_id", "item_id").unique()
                 .with_columns(pl.lit(1).alias("h")))
    new = (t.join(hist, on=["user_id", "item_id"], how="left")
            .filter(pl.col("h").is_null()).drop("h"))
    total = new["frac"].sum()
    print(f"new-track value available in holdout: {total:.0f}")

    # consecutive plays within one listening session
    recent = (inter.filter(pl.col("d") >= "2025-06-16")
              .select("user_id", "item_id", "listened_datetime")
              .sort(["user_id", "listened_datetime"]))
    step = recent.with_columns([
        pl.col("item_id").shift(-1).over("user_id").alias("nxt"),
        pl.col("listened_datetime").shift(-1).over("user_id").alias("t2"),
        pl.col("user_id").shift(-1).over("user_id").alias("u2"),
    ]).drop_nulls()
    step = step.with_columns(
        ((pl.col("t2").str.to_datetime() - pl.col("listened_datetime").str.to_datetime())
         .dt.total_seconds() / 60).alias("gap")
    ).filter((pl.col("gap") >= 0) & (pl.col("gap") <= GAP_MIN)
             & (pl.col("item_id") != pl.col("nxt")))
    print(f"in-session consecutive pairs: {step.height:,}")

    trans = step.group_by(["item_id", "nxt"]).agg(pl.len().alias("w"))
    print(f"distinct directed transitions: {trans.height:,}")

    # a user's most recent distinct tracks act as the seeds
    seeds = (inter.filter(pl.col("user_id").is_in(users.implode()))
             .sort("listened_datetime", descending=True)
             .unique(["user_id", "item_id"], keep="first")
             .sort(["user_id", "listened_datetime"], descending=[False, True])
             .group_by("user_id", maintain_order=True).head(SEEDS)
             .select("user_id", "item_id"))

    pop = (inter.filter(pl.col("d") >= "2025-07-16")
           .group_by("item_id").agg(pl.col("user_id").n_unique().alias("w")))

    def recall_of(cand, label):
        top = (cand.sort("s", descending=True)
                   .group_by("user_id", maintain_order=True).head(TOPN)
                   .select("user_id", pl.col("cand").alias("item_id")))
        got = (new.join(top.with_columns(pl.lit(1).alias("k")),
                        on=["user_id", "item_id"], how="left")
                  .filter(pl.col("k").is_not_null())["frac"].sum())
        print(f"  {label:26s} new-track value in top-{TOPN}: {got/total:6.2%}")
        return got / total

    # (a) directed:  seed -> candidate
    fwd = (seeds.join(trans, on="item_id")
                .group_by(["user_id", "nxt"]).agg(pl.col("w").sum().alias("s"))
                .rename({"nxt": "cand"}))
    # (b) symmetrised: the same counts with direction discarded
    # swap the endpoints, then re-select so the column ORDER matches too --
    # rename alone leaves them transposed and vstack refuses
    flip = trans.select(pl.col("nxt").alias("item_id"),
                        pl.col("item_id").alias("nxt"), pl.col("w"))
    sym = (pl.concat([trans.select("item_id", "nxt", "w"), flip])
             .group_by(["item_id", "nxt"]).agg(pl.col("w").sum().alias("w")))
    und = (seeds.join(sym, on="item_id")
                .group_by(["user_id", "nxt"]).agg(pl.col("w").sum().alias("s"))
                .rename({"nxt": "cand"}))
    # (c) popularity, ignoring the user entirely
    popc = (seeds.select("user_id").unique().join(pop.head(20000), how="cross")
                 .rename({"item_id": "cand", "w": "s"}))

    print()
    a = recall_of(fwd, "directed  (order matters)")
    b = recall_of(und, "symmetrised (order lost)")
    recall_of(popc, "popularity only")
    print()
    print(f"  order premium: {a - b:+.2%}  "
          f"({'ORDER CARRIES SIGNAL -> SASRec worth building' if a - b > 0.01 else 'order adds little -> SASRec unlikely to pay'})")


if __name__ == "__main__":
    main()
