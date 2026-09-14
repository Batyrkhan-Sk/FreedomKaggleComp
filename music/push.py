"""Untested levers, all paired, all scored under metric C. Target: +0.018."""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
from train12 import cached

EXTRA = ['track_like_count','track_dislike_count','track_download_count','like_ratio',
         'like_per_play','download_ratio','u_liked','u_disliked','u_downloaded',
         'u_like_rate','u_age_bin','u_children','u_gender','genre_match',
         'n_full','n_skip','n_part','full_rate','skip_rate','f_std']
inter, meta, artists, genres, users = load_all()
tr = pl.concat([cached(f"tr_{c}", None) for c,_ in WINDOWS])
C  = cached("apply", None)
t  = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA = t["user_id"].n_unique()
npl= t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
y = tr["y"].to_numpy()

# metric C's implied per-user weight: 1/min(50, distinct tracks played in the window).
# proxy available at apply time: plays in last 7d scaled to 15d, divided by repeat rate.
def est_n(df):
    p7 = np.nan_to_num(df["u_plays_7"].to_numpy().astype(float), nan=0.0)
    rr = np.nan_to_num(df["u_repeat_rate"].to_numpy().astype(float), nan=1.0)
    return (p7*15/7 / np.maximum(rr,1.0)).clip(1,50)
w_tr = 1.0/est_n(tr)
print(f"train {tr.height:,}   est n_played: median {np.median(est_n(tr)):.1f}", flush=True)

def run(feats, lbl, sw=None, mask=None):
    X = tr.select(feats).to_numpy().astype(np.float32); yy = y; ww = sw
    if mask is not None:
        X = X[mask]; yy = y[mask]; ww = None if sw is None else sw[mask]
    if ww is not None: ww = ww/ww.mean()
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, random_state=2, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=40).fit(X, yy, sample_weight=ww)
    D = (C.with_columns(pl.Series("p", m.predict(C.select(feats).to_numpy().astype(np.float32))))
          .with_columns(pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    mm=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    v=(mm["s"]/mm["den"]).sum()/NA
    print(f"  {lbl:<44} C = {v:.5f}  ({v-0.37698:+.5f})", flush=True)
    return v

print("\n(baseline without sample_weight = 0.37698; with uniform weights = 0.37686)")
run(FEATS, "1 baseline FEATS                    [CTRL]")
run(FEATS, "2 per-user weight 1/n_hat", sw=w_tr)
run(FEATS, "3 per-user weight 1/sqrt(n_hat)", sw=np.sqrt(w_tr))
run(FEATS+EXTRA, "4 FEATS + 20 cached metadata/play-shape")
run(FEATS+EXTRA, "5 (4) + per-user weight", sw=w_tr)
h = tr["is_hist"].to_numpy()
rng = np.random.default_rng(0)
keep = (h==1) | (y>0) | (rng.random(len(y))<0.35)
print(f"\n  rebalance keeps {keep.mean()*100:.0f}% of rows "
      f"(hist share {h[keep].mean()*100:.0f}% vs {h.mean()*100:.0f}%)")
run(FEATS, "6 downsample new-candidate zeros x0.35", mask=keep)
