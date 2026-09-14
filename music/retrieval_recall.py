"""Does a PER-USER pool cover more new-track value than the global one?

The global top-1500 holds 49.5% of new-track value, and every pool experiment on
record varies that same GLOBAL list (wider: #8, reweighted: power-user). Nobody
has tried giving each user their OWN 1500 candidates.

That is a different intervention from #8: same pool SIZE, but low-value global
tracks swapped for high-value personal ones, raising the density of relevant
items instead of adding candidates the ranker cannot discriminate.

Retrieval here is co-visitation: score every item by how strongly it co-occurs
with the user's 20 most recent tracks, popularity-damped exactly as covis.py
does. Cheap, and it is the same signal already proven to be the best FEATURE on
this task -- just used for a different job.

Compares, on the holdout:
   global top-N        (today's pool)
   per-user top-N      (pure retrieval)
   hybrid N/2 + N/2    (personal + trending, the usual production shape)
If per-user recall is not clearly higher, the whole retrieval idea is dead for
~20 minutes and we stop.
"""
import numpy as np
import polars as pl
import scipy.sparse as sp

from covis import _index
from evaluate import CUT, load, truth

WINDOW, N_SEEDS, POP_DAMP, MIN_SECS = 30, 20, 0.5, 30

train, hold, meta, users = load(CUT)
t = truth(hold, meta, users)
hist = (train.filter(pl.col("user_id").is_in(users.implode()))
        .select("user_id", "item_id").unique())
new = t.join(hist, on=["user_id", "item_id"], how="anti")
TOTAL = new["frac"].sum()
active = t["user_id"].n_unique()
print(f"new-track value {TOTAL:,.0f} pts over {active} active users\n", flush=True)

cut_dt = pl.lit(CUT).str.to_date()
recent = (train.with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
          .filter((pl.col("age") <= WINDOW) & (pl.col("listened_duration") >= MIN_SECS))
          .select("user_id", "item_id", "age"))

uu, umap = _index(recent["user_id"].to_numpy())
ii, imap = _index(recent["item_id"].to_numpy())
rows = np.array([umap[u] for u in recent["user_id"].to_list()], dtype=np.int32)
cols = np.array([imap[i] for i in recent["item_id"].to_list()], dtype=np.int32)
R = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(uu), len(ii)))
R.sum_duplicates(); R.data[:] = 1.0
print(f"matrix {R.shape} nnz {R.nnz:,}", flush=True)

pop = np.asarray(R.sum(0)).ravel()
damp = 1.0 / np.power(np.maximum(pop, 1.0), POP_DAMP)

seeds = (recent.filter(pl.col("user_id").is_in(users.implode()))
         .group_by(["user_id", "item_id"]).agg(pl.col("age").min())
         .sort(["user_id", "age", "item_id"])
         .group_by("user_id", maintain_order=True).head(N_SEEDS))
by_user = {}
for u, i, _ in seeds.iter_rows():
    by_user.setdefault(u, []).append(imap[i])

# global trending pool, same definition as everywhere else
recent14 = (train.with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
            .filter(pl.col("age") <= 14).join(meta, on="item_id", how="left")
            .filter(pl.col("track_duration") > 0)
            .with_columns((pl.col("listened_duration")/pl.col("track_duration")).clip(0,1).alias("f")))
gl = (recent14.group_by("item_id").agg(pl.col("f").sum().alias("s"))
      .sort("s", descending=True))

truth_by_user = {}
for u, i, f in new.iter_rows():
    truth_by_user.setdefault(u, {})[i] = f

Rt = R.T.tocsr()
test = [u for u in users.to_list() if u in by_user]
CHUNK = 250


def retrieve(block):
    """Co-visitation scores over ALL items for a block of users -> items x block.

    P selects each user's seed items; R @ P counts, for every user in the
    corpus, how many of this test user's seeds they also played; Rt @ (that)
    then counts co-occurrence for every item. One sparse product per chunk
    rather than a Python loop over users.
    """
    r, c = [], []
    for k, u in enumerate(block):
        for i in by_user[u]:
            r.append(i); c.append(k)
    P = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)),
                      shape=(len(ii), len(block)))
    return np.asarray((Rt @ (R @ P)).todense()) * damp[:, None]


truth_by_user = {}
for u, i, f in new.iter_rows():
    truth_by_user.setdefault(u, {})[i] = f

for N in (1500, 5000):
    gpool = set(gl.head(N)["item_id"].to_list())
    ghalf = set(gl.head(N // 2)["item_id"].to_list())
    gcov = new.filter(pl.col("item_id").is_in(list(gpool)))["frac"].sum()
    pu_cov = hy_cov = 0.0
    for s in range(0, len(test), CHUNK):
        block = test[s:s + CHUNK]
        Sc = retrieve(block)
        for k, u in enumerate(block):
            tb = truth_by_user.get(u)
            if not tb:
                continue
            col = Sc[:, k]
            top = np.argpartition(-col, min(N, len(ii) - 1))[:N]
            pu = {int(ii[j]) for j in top}
            pu_cov += sum(v for key, v in tb.items() if key in pu)
            top2 = np.argpartition(-col, min(N // 2, len(ii) - 1))[:N // 2]
            hy = ghalf | {int(ii[j]) for j in top2}
            hy_cov += sum(v for key, v in tb.items() if key in hy)
        print(f"    ...{min(s+CHUNK, len(test))}/{len(test)} users", flush=True)
    print(f"\nN={N}")
    print(f"  global   top-{N:<5d} covers {gcov/TOTAL:6.1%} of new-track value")
    print(f"  PER-USER top-{N:<5d} covers {pu_cov/TOTAL:6.1%}")
    print(f"  hybrid   {N//2}+{N//2:<4d} covers {hy_cov/TOTAL:6.1%}", flush=True)
