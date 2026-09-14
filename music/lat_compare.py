"""Is the latent block the weak link, and if so WHICH of its three flaws?

latent.py has three separable weaknesses. Testing "ALS vs SVD" head-to-head
would confound them, and then a win would not say what to build. So four arms,
all scored identically -- rank the top-1500 trending pool (minus the user's own
history) and take the 16 new-track slots the model actually spends, then measure
the captured share of the 42,321 points of holdout new-track value:

  A popularity      the floor: what pool order alone earns
  B svd-96 @ 90d    exactly what ships today
  C svd-96 @ 180d   B plus the rest of the log -> isolates the WINDOW
  D als-64 @ 180d   C plus confidence weighting and fitted user factors

Reference point: the full GBM, with every feature, captures 7.5% in these same
16 slots. These arms are rankers, and this task has already shown a signal can
fail as a ranker and win as a feature (co-visitation, 0.04 -> best feature), so
a LOW absolute number here is not disqualifying. The informative part is B vs C
vs D, which is apples-to-apples and says where the headroom is:

  C >> B  -> the fix is one line (window_days), not two hours of ALS
  D >> C  -> the implicit-feedback formulation is the real gap; build ALS
  D ~= C  -> the latent block is not the constraint; music is closed
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from als import als
from evaluate import CUT, load, truth

SLOTS = 16          # what the model currently spends on new tracks
POOL_N = 1500
MIN_USER_PLAYS = 5  # keep the factorisation tractable
MAX_ITEMS = 60_000

train, hold, meta, users = load(CUT)
t = truth(hold, meta, users)
hist = (train.filter(pl.col("user_id").is_in(users.implode()))
        .select("user_id", "item_id").unique())
new = t.join(hist, on=["user_id", "item_id"], how="anti")
TOTAL = new["frac"].sum()
print(f"holdout NEW-track value {TOTAL:,.0f} pts\n")

cut_dt = pl.lit(CUT).str.to_date()
aged = train.with_columns(
    (cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))

# ---- the trending pool, same definition as pool_population.py ----
recent = (aged.filter(pl.col("age") <= 14).join(meta, on="item_id", how="left")
          .filter(pl.col("track_duration") > 0)
          .with_columns((pl.col("listened_duration") / pl.col("track_duration"))
                        .clip(0, 1).alias("f")))
pool_df = (recent.group_by("item_id").agg(pl.col("f").sum().alias("s"))
           .sort("s", descending=True).head(POOL_N))
pool_ids = pool_df["item_id"].to_numpy()
pop_rank = {int(i): r for r, i in enumerate(pool_ids)}   # arm A

# history as a set per user, to exclude from the ranking
hist_by_user = {u: set(g) for u, g in
                zip(*hist.group_by("user_id").agg(pl.col("item_id"))
                    .to_dict(as_series=False).values())}
truth_map = {(u, i): f for u, i, f in zip(new["user_id"], new["item_id"], new["frac"])}
test_users = users.to_list()


def build_matrix(window):
    w = (aged.filter((pl.col("age") <= window) & (pl.col("listened_duration") >= 30))
         .with_columns((0.5 ** (pl.col("age") / 30.0)).alias("w"))
         .group_by(["user_id", "item_id"]).agg(pl.col("w").sum().alias("w")))
    keep_i = (w.group_by("item_id").agg(pl.len().alias("n"))
              .sort("n", descending=True).head(MAX_ITEMS)["item_id"])
    w = w.filter(pl.col("item_id").is_in(keep_i.implode()))
    keep_u = (w.group_by("user_id").agg(pl.len().alias("n"))
              .filter((pl.col("n") >= MIN_USER_PLAYS)
                      | pl.col("user_id").is_in(users.implode()))["user_id"])
    w = w.filter(pl.col("user_id").is_in(keep_u.implode()))
    uu = np.unique(w["user_id"].to_numpy()); umap = {v: k for k, v in enumerate(uu.tolist())}
    ii = np.unique(w["item_id"].to_numpy()); imap = {v: k for k, v in enumerate(ii.tolist())}
    rows = np.array([umap[u] for u in w["user_id"].to_list()], dtype=np.int32)
    cols = np.array([imap[i] for i in w["item_id"].to_list()], dtype=np.int32)
    vals = np.log1p(w["w"].to_numpy()).astype(np.float32)
    R = sp.csr_matrix((vals, (rows, cols)), shape=(len(uu), len(ii)))
    print(f"  matrix {R.shape} nnz {R.nnz:,} ({window}d window)")
    return R, umap, imap


def evaluate(name, score_fn):
    """score_fn(user) -> array of scores aligned with pool_ids, or None to skip."""
    got = 0.0
    for u in test_users:
        s = score_fn(u)
        if s is None:
            continue
        h = hist_by_user.get(u, ())
        s = s.copy()
        for j, it in enumerate(pool_ids):
            if int(it) in h:
                s[j] = -np.inf
        for j in np.argpartition(-s, SLOTS)[:SLOTS]:
            if np.isfinite(s[j]):
                got += truth_map.get((u, int(pool_ids[j])), 0.0)
    print(f"{name:22s} captures {got:8,.0f} pts = {got / TOTAL:6.2%} "
          f"of new-track value in {SLOTS} slots")
    return got / TOTAL


# ---- A: popularity ----
pop_scores = -np.arange(len(pool_ids), dtype=np.float32)
evaluate("A popularity", lambda u: pop_scores)


def factor_arm(name, R, umap, imap, item_f, user_f):
    cols, miss = [], []
    for it in pool_ids:
        j = imap.get(int(it))
        cols.append(j if j is not None else -1)
    cols = np.array(cols)
    ok = cols >= 0
    P = np.zeros((len(pool_ids), item_f.shape[1]), dtype=np.float32)
    P[ok] = item_f[cols[ok]]
    print(f"  {ok.sum()}/{len(pool_ids)} pool items present in the factorisation")

    def fn(u):
        r = umap.get(u)
        if r is None:
            return None
        s = P @ user_f[r]
        s[~ok] = -np.inf
        return s
    return evaluate(name, fn)


for window, label in ((90, "B svd-96 @ 90d"), (180, "C svd-96 @ 180d")):
    R, umap, imap = build_matrix(window)
    svd = TruncatedSVD(n_components=96, random_state=0).fit(R)
    itf = svd.components_.T.astype(np.float32)
    itf /= np.linalg.norm(itf, axis=1, keepdims=True) + 1e-9
    usf = np.asarray(R @ itf)                      # the centroid projection, as shipped
    usf /= np.linalg.norm(usf, axis=1, keepdims=True) + 1e-9
    factor_arm(label, R, umap, imap, itf, usf)
    if window == 180:
        print("  fitting als...", flush=True)
        X, Y = als(R, factors=64, iters=10)
        factor_arm("D als-64 @ 180d", R, umap, imap, Y, X)
