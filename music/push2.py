"""Arms 4-6, reading cache_all (which carries the 20 metadata/play-shape columns).
Arm 4a is the control: same FEATS from cache_all must reproduce 0.37698."""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

EXTRA = ['track_like_count','track_dislike_count','track_download_count','like_ratio',
         'like_per_play','download_ratio','u_liked','u_disliked','u_downloaded',
         'u_like_rate','u_age_bin','u_children','u_gender','genre_match',
         'n_full','n_skip','n_part','full_rate','skip_rate','f_std']
inter, meta, artists, genres, users = load_all()
tr = pl.concat([pl.read_parquet(f"cache_all/tr_{c}.parquet") for c,_ in WINDOWS])
C  = pl.read_parquet("cache_all/apply.parquet")
t  = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA = t["user_id"].n_unique()
npl= t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter, artists, genres
y = tr["y"].to_numpy()
print(f"train {tr.height:,}  apply {C.height:,}", flush=True)

def run(feats, lbl, mask=None):
    X = tr.select(feats).to_numpy().astype(np.float32); yy = y
    if mask is not None: X = X[mask]; yy = y[mask]
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, random_state=2, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40).fit(X, yy)
    D = (C.with_columns(pl.Series("p", m.predict(C.select(feats).to_numpy().astype(np.float32))))
          .with_columns(pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    mm=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    v=(mm["s"]/mm["den"]).sum()/NA
    print(f"  {lbl:<44} C = {v:.5f}  ({v-0.37698:+.5f})", flush=True)

run(FEATS, "4a FEATS from cache_all             [CTRL]")
run(FEATS+EXTRA, "4b FEATS + 20 metadata/play-shape cols")
h = tr["is_hist"].to_numpy(); rng = np.random.default_rng(0)
keep = (h==1) | (y>0) | (rng.random(len(y))<0.35)
print(f"  rebalance keeps {keep.mean()*100:.0f}% of rows (hist share {h[keep].mean()*100:.0f}% vs {h.mean()*100:.0f}%)", flush=True)
run(FEATS, "5  downsample new-candidate zeros x0.35", mask=keep)
run(FEATS+EXTRA, "6  (4b) + downsample", mask=keep)
