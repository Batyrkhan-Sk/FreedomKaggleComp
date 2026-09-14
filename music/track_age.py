"""Is TRACK AGE the new-track signal that nothing in the dead list covers?

Every closed lever on this task -- sequence order, pool width, power-user pools,
two-stage, slot re-allocation, replay cadence, co-visitation, discovery bias --
scores a track by how it CO-OCCURS with what the user already played. A track
that arrived on the service three weeks ago has almost no co-occurrence history
by construction, so it is close to invisible to every feature we have, while
being exactly the kind of thing a user plays in a fixed future fortnight.

That is the only "categorically different" shape I can find that the memory's
dead list does not already cover, and it is cheap to falsify.

Step 1 (this script) asks only the premise question:
  does holdout NEW-track value concentrate on tracks that recently APPEARED,
  and does the current trending pool already contain them?

If young tracks hold little value, or the pool already covers them at the same
rate as old ones, the idea is dead here for ~20 minutes and we stop.

Left-censoring note: the log starts 2025-02-28, so a track whose first play is
that day may be any age. That only blurs the OLD end; the young buckets are
unambiguous, and they are the ones the hypothesis is about.

Pool definition is copied from pool_population.py so the coverage numbers are
comparable to the 49.5% already on record.
"""
import polars as pl

from evaluate import CUT, load, truth

train, hold, meta, users = load(CUT)
t = truth(hold, meta, users)

hist = (train.filter(pl.col("user_id").is_in(users.implode()))
        .select("user_id", "item_id").unique())
new = t.join(hist, on=["user_id", "item_id"], how="anti")
total = new["frac"].sum()
print(f"holdout NEW-track value: {total:,.0f} points over "
      f"{new['item_id'].n_unique():,} distinct tracks")

# ---- age of every track at the cut, from its first appearance in the log ----
first = (train.group_by("item_id")
         .agg(pl.col("d").min().alias("first_d"))
         .with_columns((pl.lit(CUT).str.to_date() - pl.col("first_d").str.to_date())
                       .dt.total_days().alias("age")))
LOG_SPAN = first["age"].max()
print(f"log spans {LOG_SPAN} days before the cut "
      f"({first.height:,} tracks ever played in train)\n")

# ---- the current trending pool: last 14 days, top-1500 by listened fraction ----
recent = (train.with_columns(
    (pl.lit(CUT).str.to_date() - pl.col("d").str.to_date()).dt.total_days().alias("age_d"))
    .filter(pl.col("age_d") <= 14)
    .join(meta, on="item_id", how="left")
    .filter(pl.col("track_duration") > 0)
    .with_columns((pl.col("listened_duration") / pl.col("track_duration"))
                  .clip(0, 1).alias("f")))
pool = (recent.group_by("item_id").agg(pl.col("f").sum().alias("s"))
        .sort("s", descending=True).head(1500)["item_id"])
pool_set = pool.implode()

nv = (new.join(first, on="item_id", how="left")
        .with_columns(pl.col("age").fill_null(LOG_SPAN + 1),
                      pl.col("item_id").is_in(pool_set).alias("in_pool")))

BUCKETS = [(0, 7), (8, 14), (15, 30), (31, 60), (61, 90), (91, 140),
           (141, 10_000)]

print(f"{'age at cut':>12} {'points':>9} {'share':>7} {'tracks':>8} "
      f"{'pts/track':>10} {'in pool':>8} {'pool pts':>9}")
for lo, hi in BUCKETS:
    b = nv.filter((pl.col("age") >= lo) & (pl.col("age") <= hi))
    if not b.height:
        continue
    pts = b["frac"].sum()
    ntr = b["item_id"].n_unique()
    inp = b.filter(pl.col("in_pool"))
    label = f"{lo}-{hi}d" if hi < 10_000 else f"{lo}d+ / censored"
    print(f"{label:>12} {pts:9,.0f} {pts/total:6.1%} {ntr:8,} "
          f"{pts/max(ntr,1):10.3f} {inp['item_id'].n_unique():8,} "
          f"{inp['frac'].sum()/max(pts,1):8.1%}")

young = nv.filter(pl.col("age") <= 30)
print(f"\ntracks <=30d old hold {young['frac'].sum():,.0f} pts "
      f"({young['frac'].sum()/total:.1%} of new-track value); "
      f"the pool covers {young.filter(pl.col('in_pool'))['frac'].sum()/max(young['frac'].sum(),1):.1%} of it")
old = nv.filter(pl.col("age") > 30)
print(f"tracks  >30d old hold {old['frac'].sum():,.0f} pts "
      f"({old['frac'].sum()/total:.1%}); "
      f"the pool covers {old.filter(pl.col('in_pool'))['frac'].sum()/max(old['frac'].sum(),1):.1%} of it")

# ---- what the pool itself is made of, by age ----
pool_age = (pl.DataFrame({"item_id": pool}).join(first, on="item_id", how="left")
            .with_columns(pl.col("age").fill_null(LOG_SPAN + 1)))
print(f"\npool composition: {pool_age.filter(pl.col('age') <= 30).height}/1500 "
      f"of the trending pool is itself <=30d old")
