"""Bring music-recommender-solution.ipynb up to the pipeline that actually scored.

The notebook reproduces the covis model (its own title says LB 0.37553). The
SELECTED submission is 0.37779 = submit7.py, which adds the SVD latent block.
Under the competition's deliverable rule the organizers must be able to run the
notebook and obtain the submitted result, so a notebook reproducing a different
(worse) pipeline is an exclusion risk, not a rounding difference.

Idempotent: re-running detects the latent cell and stops.
"""
import json
import pathlib

p = pathlib.Path("music-recommender-solution.ipynb")
nb = json.loads(p.read_text())
src = lambda c: "".join(c["source"])

if any("latent_features" in src(c) for c in nb["cells"]):
    raise SystemExit("already patched -- latent cell present")

def cell(kind, text):
    return {"cell_type": kind, "metadata": {},
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            "source": text.splitlines(keepends=True)}

LAT_MD = """## Latent-factor affinity

Co-visitation can only relate two tracks somebody actually played together.
Latent factors relate tracks occupying the same region of taste space even when
they never co-occur -- which is where new-track value lives, spread over ~15,000
tracks. Truncated SVD of the recency-weighted user x item matrix gives item
factors; a user sits at the weighted centroid of what they play, and affinity is
the dot product. Added as a *feature*: standalone CF scored 0.04 here against a
trending list's 0.14, but as a feature this was worth +0.002 real (0.37553 ->
0.37779).
"""

LAT_CODE = '''from sklearn.decomposition import TruncatedSVD


def latent_features(inter, users, pool_items, cut, window_days=90, n_factors=96,
                    halflife=30.0, min_secs=30, seed=0):
    cut_dt = pl.lit(cut).str.to_date()
    recent = (
        inter.filter(pl.col("d") < cut)
        .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
        .filter((pl.col("age") <= window_days) & (pl.col("listened_duration") >= min_secs))
        .with_columns((0.5 ** (pl.col("age") / halflife)).alias("w"))
        .group_by(["user_id", "item_id"])
        .agg(pl.col("w").sum().alias("w"))
    )
    if recent.height == 0:
        return pl.DataFrame({"user_id": [], "item_id": [], "latent": []})

    uu, umap = _index(recent["user_id"].to_numpy())
    ii, imap = _index(recent["item_id"].to_numpy())
    rows = np.fromiter((umap[u] for u in recent["user_id"].to_list()),
                       dtype=np.int32, count=recent.height)
    cols = np.fromiter((imap[i] for i in recent["item_id"].to_list()),
                       dtype=np.int32, count=recent.height)
    vals = np.log1p(recent["w"].to_numpy()).astype(np.float32)
    R = sp.csr_matrix((vals, (rows, cols)), shape=(len(uu), len(ii)))
    print(f"  latent matrix {R.shape}, nnz {R.nnz:,}")

    svd = TruncatedSVD(n_components=n_factors, random_state=seed)
    svd.fit(R)
    item_f = svd.components_.T.astype(np.float32)
    item_f /= np.linalg.norm(item_f, axis=1, keepdims=True) + 1e-9
    user_f = np.asarray((R @ item_f))
    user_f /= np.linalg.norm(user_f, axis=1, keepdims=True) + 1e-9

    keep = [i for i in pool_items if i in imap]
    if not keep:
        return pl.DataFrame({"user_id": [], "item_id": [], "latent": []})
    pool_idx = np.array([imap[i] for i in keep], dtype=np.int32)
    pool_ids = np.array(keep)
    pool_f = item_f[pool_idx]

    test = [u for u in users.to_list() if u in umap]
    out_u, out_i, out_v = [], [], []
    for start in range(0, len(test), 500):
        block = test[start : start + 500]
        idx = np.array([umap[u] for u in block])
        scores = user_f[idx] @ pool_f.T
        for bi, u in enumerate(block):
            out_u.append(np.full(len(pool_ids), u, dtype=np.int64))
            out_i.append(pool_ids)
            out_v.append(scores[bi].astype(np.float32))

    return pl.DataFrame({
        "user_id": np.concatenate(out_u),
        "item_id": np.concatenate(out_i),
        "latent": np.concatenate(out_v),
    })
'''

# insert after the covis code cell (index 5)
nb["cells"][6:6] = [cell("markdown", LAT_MD), cell("code", LAT_CODE)]

# --- cell 7 (now 9): add latent to FEATS and to the featurisation --------------
i = next(k for k, c in enumerate(nb["cells"]) if 'FEATS = FEATS + ["covis"' in src(c))
s = src(nb["cells"][i])
s = s.replace('FEATS = FEATS + ["covis", "covis_rank"]',
              'FEATS = FEATS + ["covis", "covis_rank", "latent", "latent_rank"]')
s = s.replace(
    '''    cv = covis_features(inter, users, pool_items, cut)
    C = C.join(cv, on=["user_id", "item_id"], how="left").with_columns(pl.col("covis").fill_null(0.0))
    return C.with_columns(
        pl.col("covis").rank("ordinal", descending=True).over("user_id").alias("covis_rank")
    )''',
    '''    cv = covis_features(inter, users, pool_items, cut)
    lf = latent_features(inter, users, pool_items, cut)
    C = (C.join(cv, on=["user_id", "item_id"], how="left")
          .join(lf, on=["user_id", "item_id"], how="left")
          .with_columns(pl.col("covis").fill_null(0.0), pl.col("latent").fill_null(0.0)))
    return C.with_columns(
        pl.col("covis").rank("ordinal", descending=True).over("user_id").alias("covis_rank"),
        pl.col("latent").rank("ordinal", descending=True).over("user_id").alias("latent_rank"),
    )''')
assert "latent_rank" in s, "cell-7 patch did not apply"
nb["cells"][i]["source"] = s.splitlines(keepends=True)

# --- header ------------------------------------------------------------------
h = src(nb["cells"][0]).replace(
    "# Personalized Music Recommender - reproducible solution (LB 0.37553)",
    "# Personalized Music Recommender - reproducible solution (LB 0.37779)")
h = h.replace(
    "**CPU only**, roughly 25-40 minutes.",
    "**CPU only**, roughly 35-55 minutes.")
nb["cells"][0]["source"] = h.splitlines(keepends=True)

p.write_text(json.dumps(nb, indent=1) + "\n")
print(f"patched: {len(nb['cells'])} cells")
for k, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        first = next((l for l in src(c).splitlines() if l.strip()), "")
        print(f"  [{k}] {first[:72]}")
