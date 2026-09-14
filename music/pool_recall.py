"""How much of the available new-track value does a candidate pool even contain?

Ranking cannot recover what was never a candidate, so pool recall is a hard
ceiling on the score. The global trending pool was measured at 44% of new-track
value at 1,000 items -- and simply making that list longer did nothing, which is
unsurprising: it adds the SAME tracks for every user.

This measures whether a *personalised* pool of the same size reaches further,
before any of it is built. If it does not, the idea is dead and no model on top
would have saved it.
"""
import numpy as np
import polars as pl
from evaluate import CUT, load, truth

SEED_DAYS, N_SEEDS = 30, 20


def main():
    inter, holdout, meta, users = load()
    t = truth(holdout, meta, users)                    # user_id, item_id, y
    hist = (inter.filter(pl.col("user_id").is_in(users.implode()))
                 .select("user_id", "item_id").unique())
    # only NEW tracks matter here; repeats are covered by history candidates
    new = t.join(hist.with_columns(pl.lit(1).alias("h")),
                 on=["user_id", "item_id"], how="left").filter(pl.col("h").is_null())
    total = new["frac"].sum()
    print(f'{len(new)} new (user,item) pairs in the holdout, total value {total:.0f}')

    cut_dt = pl.lit(CUT).str.to_date()
    recent = inter.with_columns(
        (cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))

    # ---- A: the current global trending pool -----------------------------
    for N in (1500, 5000, 15000):
        top = (recent.filter(pl.col("age") <= 30)
               .group_by("item_id").agg(pl.col("user_id").n_unique().alias("p"))
               .sort("p", descending=True).head(N)["item_id"])
        got = new.filter(pl.col("item_id").is_in(top.implode()))["frac"].sum()
        print(f'  global top-{N:<6} recall {got/total:6.1%}')

    # ---- B: personalised -- co-visitation neighbours of each user's seeds --
    seeds = (recent.filter((pl.col("age") <= SEED_DAYS)
                           & pl.col("user_id").is_in(users.implode()))
             .sort("age").group_by("user_id").head(N_SEEDS)
             .select("user_id", "item_id"))
    # users who played each seed item recently, then what else those users played
    co = (recent.filter(pl.col("age") <= 60)
          .select("user_id", "item_id").unique())
    nb = (seeds.join(co, on="item_id", suffix="_o")          # co-listeners
              .select(pl.col("user_id"), pl.col("user_id_o"))
              .unique()
              .join(co.rename({"user_id": "user_id_o", "item_id": "cand"}),
                    on="user_id_o")
              .group_by(["user_id", "cand"]).agg(pl.len().alias("w")))
    for K in (500, 1500, 5000):
        per = (nb.sort("w", descending=True).group_by("user_id").head(K)
                 .select("user_id", pl.col("cand").alias("item_id")))
        got = (new.join(per.with_columns(pl.lit(1).alias("in_pool")),
                        on=["user_id", "item_id"], how="left")
                  .filter(pl.col("in_pool").is_not_null())["frac"].sum())
        print(f'  personalised top-{K:<5} recall {got/total:6.1%}')


if __name__ == "__main__":
    main()
