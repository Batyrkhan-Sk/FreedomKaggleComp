"""How much headroom is left in HISTORY ranking, which nothing has attacked?

Nine of the ten closed levers aim at new tracks or at the split. Only replay
cadence ever aimed at history, and it moved nothing. Yet history sits at 44.1%
capture, and "perfect history ranking caps near 0.46 real" -- above 3rd place.

A capture percentage on its own is not headroom: the 34-slot budget may simply
not FIT a heavy user's replay value, in which case 44.1% is close to the ceiling
and the whole side is closed. So this computes the ORACLE at the same budget --
rank each user's own candidates by their TRUE holdout value and take the top K.

  oracle_hist(34) ~= 44.1%  -> history is budget-bound, nothing to win, closed
  oracle_hist(34) >> 44.1%  -> it is a RANKING failure, and it is the cheapest
                               unexplored surface left on this task

Also sweeps the split, because 34/16 was tuned against the CURRENT ranker: if
history ranking has headroom, the optimum moves, and the two interact.
"""
import numpy as np
import polars as pl

from evaluate import CUT, load, truth

train, hold, meta, users = load(CUT)
t = truth(hold, meta, users)

hist_pairs = (train.filter(pl.col("user_id").is_in(users.implode()))
              .select("user_id", "item_id").unique())

# the trending pool, same definition as everywhere else
cut_dt = pl.lit(CUT).str.to_date()
recent = (train.with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
          .filter(pl.col("age") <= 14).join(meta, on="item_id", how="left")
          .filter(pl.col("track_duration") > 0)
          .with_columns((pl.col("listened_duration") / pl.col("track_duration"))
                        .clip(0, 1).alias("f")))
pool = set(recent.group_by("item_id").agg(pl.col("f").sum().alias("s"))
           .sort("s", descending=True).head(1500)["item_id"].to_list())

tagged = t.join(hist_pairs.with_columns(pl.lit(True).alias("h")),
                on=["user_id", "item_id"], how="left").with_columns(
                    pl.col("h").fill_null(False),
                    pl.col("item_id").is_in(list(pool)).alias("inpool"))

avail_h = tagged.filter(pl.col("h"))["frac"].sum()
avail_n = tagged.filter(~pl.col("h"))["frac"].sum()
avail_np = tagged.filter(~pl.col("h") & pl.col("inpool"))["frac"].sum()
active = t["user_id"].n_unique()
print(f"active users {active}/{len(users)}")
print(f"available  history {avail_h:9,.0f} pts   new {avail_n:9,.0f} pts "
      f"(of which in-pool {avail_np:,.0f})\n")

# per-user sorted truth values, for history and for in-pool new tracks
hv, nv = {}, {}
for u, g in tagged.filter(pl.col("h")).group_by("user_id"):
    hv[u[0]] = np.sort(g["frac"].to_numpy())[::-1]
for u, g in tagged.filter(~pl.col("h") & pl.col("inpool")).group_by("user_id"):
    nv[u[0]] = np.sort(g["frac"].to_numpy())[::-1]

def oracle(d, k):
    return sum(v[:k].sum() for v in d.values())

SCALE = 1.29 / (active * 50)          # captured points -> leaderboard score
print(f"{'slots':>6} {'oracle hist':>12} {'% of avail':>11} "
      f"{'oracle new':>11} {'% of avail':>11}")
for k in (8, 16, 24, 34, 42, 50):
    oh, on = oracle(hv, k), oracle(nv, k)
    print(f"{k:6d} {oh:12,.0f} {oh/avail_h:10.1%} {on:11,.0f} {on/avail_np:10.1%}")

print(f"\nACTUAL today: history 44.1% ({0.441*avail_h:,.0f} pts at 34 slots), "
      f"new 7.5% ({0.075*avail_n:,.0f} pts at 16 slots)")
print(f"oracle at the SAME budget: history {oracle(hv,34)/avail_h:.1%}, "
      f"new {oracle(nv,16)/avail_np:.1%} of in-pool\n")

print("split sweep, oracle on both sides (upper bound at each allocation):")
best = (0, None)
for h in (26, 30, 34, 38, 42, 46, 50):
    pts = oracle(hv, h) + oracle(nv, 50 - h)
    sc = pts * SCALE
    best = max(best, (sc, h))
    print(f"  {h:2d} history / {50-h:2d} new -> {pts:9,.0f} pts = {sc:.4f} leaderboard-equivalent")
print(f"\nbest oracle split: {best[1]} history, score {best[0]:.4f}")

hist_only = oracle(hv, 34) * SCALE
print(f"\nIf history ranking alone were perfect and new stayed at 7.5%: "
      f"{(oracle(hv,34) + 0.075*avail_n) * SCALE:.4f}")
print(f"Each +1pp of history capture is worth "
      f"{0.01*avail_h*SCALE:.4f} of leaderboard score.")
print(f"Each +1pp of new capture is worth "
      f"{0.01*avail_n*SCALE:.4f}.")
