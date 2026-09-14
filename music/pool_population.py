"""Is the trending pool built from the wrong population?

The pool is the top-N tracks by recent listening across ALL users. But the 1,500
test users are power users -- median 562 plays against 5 for the population --
so the global list is dominated by casual listeners whose taste may differ.

This is NOT experiment #8. That widened the GLOBAL pool 1500 -> 4000 (same
population, more items) and correctly showed recall is not the constraint. This
holds the size fixed and changes WHOSE listening defines the ranking. It also
matters beyond coverage: `cand_rank` is a model feature derived from this order,
so a mis-targeted ranking feeds the model a misleading signal on every new track.

Measures, for each pool definition, the share of holdout NEW-track value it
covers. If power-user trending is not clearly better, drop the idea.
"""
import polars as pl

from evaluate import CUT, load, truth

train, hold, meta, users = load(CUT)
t = truth(hold, meta, users)

hist = (train.filter(pl.col("user_id").is_in(users.implode()))
        .group_by(["user_id", "item_id"]).agg(pl.len()))
new = t.join(hist, on=["user_id", "item_id"], how="anti")
total = new["frac"].sum()
print(f"holdout NEW-track value: {total:,.0f} points over "
      f"{new['item_id'].n_unique():,} distinct tracks\n")

recent = (train.with_columns(
    (pl.lit(CUT).str.to_date() - pl.col("d").str.to_date()).dt.total_days().alias("age"))
    .filter(pl.col("age") <= 14)
    .join(meta, on="item_id", how="left")
    .filter(pl.col("track_duration") > 0)
    .with_columns((pl.col("listened_duration") / pl.col("track_duration"))
                  .clip(0, 1).alias("f")))

# who counts as a power user: match the test cohort's own activity profile
plays = train.group_by("user_id").agg(pl.len().alias("n"))
test_med = plays.filter(pl.col("user_id").is_in(users.implode()))["n"].median()
print(f"median plays -- test users {test_med:.0f}, population "
      f"{plays['n'].median():.0f}")

def pool(df, n):
    return set(df.group_by("item_id").agg(pl.col("f").sum().alias("s"))
               .sort("s", descending=True).head(n)["item_id"].to_list())

def cover(items):
    return new.filter(pl.col("item_id").is_in(list(items)))["frac"].sum() / total

for thresh in (None, 50, 200, 562):
    if thresh is None:
        src, label = recent, "global (all users)"
    else:
        keep = plays.filter(pl.col("n") >= thresh)["user_id"]
        src = recent.filter(pl.col("user_id").is_in(keep.implode()))
        label = f"users with >= {thresh} plays"
    print(f"\n{label}  ({src.height:,} recent plays)")
    for n in (1500, 5000):
        p = pool(src, n)
        print(f"   top-{n:<5d} covers {cover(p):6.1%} of new-track value")
