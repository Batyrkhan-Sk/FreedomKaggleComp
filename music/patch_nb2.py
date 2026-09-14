"""Bring music-recommender-solution.ipynb up to submit17.py -- the pipeline that
actually produced the submitted 0.38391.

The notebook currently reproduces an OLDER, lower-scoring config: 200 iters, a
single model, no 90-day covis coverage and no multi-statistic covis. Under the
deliverable rule the organizers must be able to run the notebook and obtain the
SUBMITTED result, so a notebook reproducing a different pipeline is an exclusion
risk. Idempotent: re-running detects the patch and stops.
"""
import json, pathlib
p = pathlib.Path("music-recommender-solution.ipynb")
nb = json.loads(p.read_text())
src = lambda c: "".join(c["source"])
if any("covis_multi" in src(c) for c in nb["cells"]):
    raise SystemExit("already patched -- covis_multi present")

def cell(kind, text):
    return {"cell_type": kind, "metadata": {},
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            "source": text.splitlines(keepends=True)}

MD = """## Wider covis coverage, and covis as more than one number

Two additions over the single co-visitation score, each measured on the holdout:

* **A second 30->90 day window (min_secs 10).** The 30-day window leaves ~27% of
  active users with no co-visitation feature at all, and under the competition's
  per-user metric those low-activity users count exactly as much as anyone else.
  Coverage rises 1,010 -> 1,205 users.
* **The match SHAPE, not just its total.** `sum` over 20 seed tracks cannot tell
  a candidate that is very close to the one track a user is obsessed with from
  one weakly related to all twenty. Emitting max / mean / top-3 / recency-
  weighted / last-seed, plus ranks, keeps that distinction.
"""

CODE = '''def covis_multi(inter, users, pool_items, cut, window_days=30, n_seeds=20,
                min_secs=30, pop_damp=0.5, halflife=7.0, prefix="cv"):
    """Co-visitation summarised several ways instead of one sum."""
    cut_dt = pl.lit(cut).str.to_date()
    recent = (inter.filter(pl.col("d") < cut)
        .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
        .filter((pl.col("age") <= window_days) & (pl.col("listened_duration") >= min_secs))
        .select("user_id", "item_id", "age"))
    if recent.height == 0:
        return None
    uu, umap = _index(recent["user_id"].to_numpy())
    ii, imap = _index(recent["item_id"].to_numpy())
    rows = np.fromiter((umap[u] for u in recent["user_id"].to_list()), dtype=np.int32, count=recent.height)
    cols = np.fromiter((imap[i] for i in recent["item_id"].to_list()), dtype=np.int32, count=recent.height)
    R = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(uu), len(ii)))
    R.data[:] = 1.0
    pop = np.asarray(R.sum(axis=0)).ravel() + 1.0
    Rd = (R @ sp.diags((1.0 / pop ** pop_damp).astype(np.float32))).tocsr()
    pool_idx = np.array([imap[i] for i in pool_items if i in imap], dtype=np.int32)
    pool_ids = np.array([i for i in pool_items if i in imap])
    if len(pool_idx) == 0:
        return None
    Rpool = Rd[:, pool_idx].tocsc()
    seeds = (recent.filter(pl.col("user_id").is_in(users.implode()))
             .group_by(["user_id", "item_id"]).agg(pl.col("age").min().alias("age"))
             .sort(["user_id", "age", "item_id"])
             .group_by("user_id", maintain_order=True).head(n_seeds))
    by_user = {}
    for u, i, a in seeds.iter_rows():
        by_user.setdefault(u, ([], []))
        by_user[u][0].append(imap[i]); by_user[u][1].append(a)
    seed_items = np.unique(np.concatenate([np.array(v[0]) for v in by_user.values()]))
    smap = {int(v): k for k, v in enumerate(seed_items.tolist())}
    SIM = np.asarray((Rd[:, seed_items].T @ Rpool).todense(), dtype=np.float32)
    out = {k: [] for k in ("user_id", "item_id", "sum", "max", "mean", "top3", "wsum", "last")}
    for u in users.to_list():
        if u not in by_user:
            continue
        idx, ages = by_user[u]
        M = SIM[[smap[j] for j in idx]]
        nz = np.flatnonzero(M.any(axis=0))
        if len(nz) == 0:
            continue
        M = M[:, nz]
        w = (0.5 ** (np.asarray(ages, dtype=np.float32) / halflife))[:, None]
        k = min(3, M.shape[0])
        out["user_id"].append(np.full(len(nz), u, dtype=np.int64))
        out["item_id"].append(pool_ids[nz])
        out["sum"].append(M.sum(0)); out["max"].append(M.max(0)); out["mean"].append(M.mean(0))
        out["top3"].append(np.sort(M, axis=0)[-k:].mean(0))
        out["wsum"].append((M * w).sum(0)); out["last"].append(M[0])
    if not out["user_id"]:
        return None
    df = pl.DataFrame({("user_id" if k == "user_id" else "item_id" if k == "item_id" else f"{prefix}_{k}"):
                       np.concatenate(v) for k, v in out.items()})
    return df.with_columns([
        pl.col(f"{prefix}_{s}").rank("min", descending=True).over("user_id").alias(f"{prefix}_{s}_rank")
        for s in ("max", "wsum", "top3")])


M9 = ["cv_sum", "cv_max", "cv_mean", "cv_top3", "cv_wsum", "cv_last",
      "cv_max_rank", "cv_wsum_rank", "cv_top3_rank"]
W2 = ["cv90", "cv90_rank"]
'''

# insert after the covis cell (index 5)
i = next(k for k, c in enumerate(nb["cells"]) if "def _index" in src(c))
nb["cells"][i+1:i+1] = [cell("markdown", MD), cell("code", CODE)]

def sub(old, new, why):
    for c in nb["cells"]:
        s = src(c)
        if old in s:
            c["source"] = s.replace(old, new, 1).splitlines(keepends=True)
            print("patched:", why); return
    raise SystemExit(f"NOT FOUND: {why}")

sub('FEATS = FEATS + ["covis", "covis_rank", "latent", "latent_rank"]',
    'FEATS = FEATS + ["covis", "covis_rank", "latent", "latent_rank"] + W2 + M9',
    "FEATS list")

sub("""    cv = covis_features(inter, users, pool_items, cut)
    lf = latent_features(inter, users, pool_items, cut)
    C = (C.join(cv, on=["user_id", "item_id"], how="left")
          .join(lf, on=["user_id", "item_id"], how="left")
          .with_columns(pl.col("covis").fill_null(0.0), pl.col("latent").fill_null(0.0)))""",
    """    cv = covis_features(inter, users, pool_items, cut)
    lf = latent_features(inter, users, pool_items, cut)
    C = (C.join(cv, on=["user_id", "item_id"], how="left")
          .join(lf, on=["user_id", "item_id"], how="left")
          .with_columns(pl.col("covis").fill_null(0.0), pl.col("latent").fill_null(0.0)))
    # second, wider covis window -- coverage, not accuracy
    c90 = covis_features(inter, users, pool_items, cut, window_days=90, min_secs=10)
    if c90.height:
        C = C.join(c90.rename({"covis": "cv90"}), on=["user_id", "item_id"], how="left")
        C = C.with_columns(pl.col("cv90").fill_null(0.0))
    else:
        C = C.with_columns(pl.lit(0.0).alias("cv90"))
    C = C.with_columns(pl.col("cv90").rank("min", descending=True).over("user_id").alias("cv90_rank"))
    mm = covis_multi(inter, users, pool_items, cut, prefix="cv")
    if mm is not None:
        C = C.join(mm, on=["user_id", "item_id"], how="left")
        C = C.with_columns([pl.col(c).fill_null(9999.0 if c.endswith("_rank") else 0.0) for c in M9])
    else:
        C = C.with_columns([pl.lit(9999.0 if c.endswith("_rank") else 0.0).alias(c) for c in M9])""",
    "covis@90d + multi-statistic covis")

sub("""model = HistGradientBoostingRegressor(
    max_iter=200, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=200,
    l2_regularization=1.0, random_state=2, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=40)
model.fit(tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy())
print(f"fitted {model.n_iter_} iterations")""",
    """# 400 iterations, not 200: re-swept against the per-user metric, where the
# capacity curve peaks at 400 (40/80/120/200 are all monotonically worse).
# Five seeds rank-averaged: single seeds spanned 0.37794..0.37998 on the
# holdout, so averaging removes the lottery as well as adding ~+0.0013.
X = tr.select(FEATS).to_numpy().astype(np.float32)
Y = tr["y"].to_numpy()
models = []
for s in range(5):
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, random_state=s, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40)
    m.fit(X, Y)
    models.append(m)
    print(f"fitted seed {s}: {m.n_iter_} iterations")""",
    "400 iters + 5-seed ensemble")

sub("""C = C.with_columns(pl.Series("p", model.predict(C.select(FEATS).to_numpy().astype(np.float32))))""",
    """Xa = C.select(FEATS).to_numpy().astype(np.float32)
ranks = [pl.Series(m.predict(Xa)).rank().to_numpy() for m in models]
C = C.with_columns(pl.Series("p", np.vstack(ranks).mean(axis=0)))""",
    "ensemble prediction")

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
import ast
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        ast.parse("\n".join("" if l.lstrip().startswith(("!", "%")) else l
                            for l in src(c).split("\n")))
print(f"\nOK: {len(nb['cells'])} cells, all code cells parse")
