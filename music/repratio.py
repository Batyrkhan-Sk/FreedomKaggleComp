"""Per-user repeat-ratio policy analysis (Li et al., TOIS 2023, section 5.5).

The reality-check paper finds that repeat-biased NBR methods FAIL TO BEAT plain
global popularity for users whose repetition ratio is in [0, 0.2] -- i.e. users
whose next basket is mostly novel items. Our model is repeat-biased (79% of its
slots are history) and our RETURNING segment sits at 36.5% of oracle, so the
same failure mode may be present and has never been measured per user.

This is a POLICY question, not a feature question: for an identifiable group of
users, would a different ranker score better? If yes, and the group is
predictable from data available BEFORE the test window, a per-user switch is a
real lever that no feature change can reach.

Fits one seed (the question is about large per-group differences, not about
+0.001 effects, which this harness has now demonstrably failed to resolve), then
decomposes the holdout by ground-truth repeat ratio.
"""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

M9 = ["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
STACK = FEATS + ["cv90","cv90_rank"] + M9
inter, meta, artists, genres, users = load_all()
def merge(n):
    return pl.read_parquet(f"cache_cv90/{n}.parquet").hstack(
        pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9))
tr = pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS]); C = merge("apply")
y = tr["y"].to_numpy()
t = truth(inter.filter(pl.col("d") >= APPLY), meta, users)
npl = (t.group_by("user_id").agg(pl.len().alias("n"))
        .with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den")))
del inter
X = tr.select(STACK).to_numpy().astype(np.float32)
Xa = C.select(STACK).to_numpy().astype(np.float32)
del tr
m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
    min_samples_leaf=200, l2_regularization=1.0, random_state=0, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=40).fit(X, y)
C = C.with_columns(pl.Series("p", m.predict(Xa)))
print("fitted", flush=True)

# ground-truth repeat ratio per user: share of their test value on items they already played
hist = C.filter(pl.col("is_hist")==1).select("user_id","item_id").with_columns(pl.lit(1).alias("h"))
tv = (t.join(hist, on=["user_id","item_id"], how="left").with_columns(pl.col("h").fill_null(0))
       .group_by("user_id").agg((pl.col("frac")*pl.col("h")).sum().alias("rep"),
                                pl.col("frac").sum().alias("tot")))
tv = tv.with_columns((pl.col("rep")/pl.col("tot")).alias("rr")).filter(pl.col("tot")>0)

def peruser(df, col, name):
    top = (df.sort(["user_id", col], descending=[False, True])
             .group_by("user_id", maintain_order=True).head(TOPK))
    o = (top.join(t, on=["user_id","item_id"], how="left")
           .with_columns(pl.col("frac").fill_null(0.0))
           .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    r = npl.join(o, on="user_id", how="left").with_columns(pl.col("s").fill_null(0.0))
    return r.with_columns((pl.col("s")/pl.col("den")).alias(name)).select("user_id", name)

D = C.with_columns((-pl.col("cand_rank").cast(pl.Float64)).alias("popscore"))
M = (tv.join(peruser(D, "p", "model"), on="user_id")
       .join(peruser(D, "decay_full", "decayfull"), on="user_id")
       .join(peruser(D, "popscore", "trending"), on="user_id"))

print(f"\nactive users {M.height}   overall model C {M['model'].mean():.5f}")
print("\n  rr bin      users   model  decayfull  trending   best-alt-minus-model")
edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
for a, b in zip(edges[:-1], edges[1:]):
    g = M.filter((pl.col("rr") >= a) & (pl.col("rr") < b))
    if not g.height: continue
    mo, de, tp = g["model"].mean(), g["decayfull"].mean(), g["trending"].mean()
    print(f"  [{a:.1f},{b:.1f})   {g.height:5d}  {mo:.4f}    {de:.4f}    {tp:.4f}   {max(de,tp)-mo:+.4f}")

# what would a PERFECT per-user oracle switch be worth overall?
sw = M.with_columns(pl.max_horizontal("model","decayfull","trending").alias("best"))
print(f"\n  oracle per-user switch over 3 rankers: {sw['best'].mean():.5f} "
      f"vs model {M['model'].mean():.5f}  (+{sw['best'].mean()-M['model'].mean():.5f})")
print(f"  users where an alternative wins: {sw.filter(pl.col('best')>pl.col('model')).height}")

# is the losing group predictable BEFORE the window? use historical repeat rate
uh = C.group_by("user_id").agg(pl.col("u_repeat_rate").first(), pl.col("u_items").first())
Q = M.join(uh, on="user_id")
print("\n  by HISTORICAL u_repeat_rate quartile (predictable in advance):")
Q = Q.with_columns(pl.col("u_repeat_rate").qcut(4, labels=["q1","q2","q3","q4"]).alias("q"))
for q in ["q1","q2","q3","q4"]:
    g = Q.filter(pl.col("q") == q)
    print(f"    {q}: n={g.height:4d}  rr={g['rr'].mean():.3f}  model={g['model'].mean():.4f}  "
          f"decayfull={g['decayfull'].mean():.4f}  trending={g['trending'].mean():.4f}")
