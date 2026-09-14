"""Does RP3-beta add anything the 46-feature stack does not already have?

Control is the standing harness: 46 feats, 200 iters x lr.03 x 63 leaves,
seeds (0,1,2), 3 cached windows = 0.37966 (sd 0.00083), ens3 0.38023.
Reproduced to 5 decimals three times, so a paired delta here is trustworthy.

Arms add the 4 RP3 columns. Also reports NEW-SLOT capture, because that is the
quantity this is aimed at: new-track capture is 7.5% now and 10.1% IS third
place, so the diagnostic matters as much as the score.
"""
import sys, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

M9 = ["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
STACK = FEATS + ["cv90","cv90_rank"] + M9
RP3 = ["rp3","rp3_rank","rp3b0","rp3b0_rank"]
CTRL, ENS3 = 0.37966, 0.38023

inter, meta, artists, genres, users = load_all()
def merge(n):
    a = pl.read_parquet(f"cache_cv90/{n}.parquet").hstack(
        pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9))
    return a.hstack(pl.read_parquet(f"cache_rp3/{n}.parquet").select(RP3))
tr = pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS]); C = merge("apply")
y = tr["y"].to_numpy()
t = truth(inter.filter(pl.col("d") >= APPLY), meta, users); NA = t["user_id"].n_unique()
npl = (t.group_by("user_id").agg(pl.len().alias("n"))
        .with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den")))
del inter
base = C.select("user_id","item_id","is_hist")
print(f"train {tr.height:,}  rp3 nonzero {100*(tr['rp3'].to_numpy()!=0).mean():.1f}%", flush=True)

def sc(p, diag=False):
    D = base.with_columns(pl.Series("p", p))
    top = (D.sort(["user_id","p"], descending=[False,True])
            .group_by("user_id", maintain_order=True).head(TOPK))
    h = top.join(t, on=["user_id","item_id"], how="left").with_columns(pl.col("frac").fill_null(0.0))
    o = h.group_by("user_id").agg(pl.col("frac").sum().alias("s"))
    m = npl.join(o, on="user_id", how="left").with_columns(pl.col("s").fill_null(0.0))
    v = (m["s"]/m["den"]).sum()/NA
    if diag:
        newpts = h.filter(pl.col("is_hist")==0)["frac"].sum()
        return v, 100*top.filter(pl.col("is_hist")==0).height/top.height, newpts
    return v

def go(feats, lbl, seeds=(0,1,2)):
    X = tr.select(feats).to_numpy().astype(np.float32)
    Xa = C.select(feats).to_numpy().astype(np.float32)
    v=[]; ranks=[]; slot=[]; npts=[]
    for s in seeds:
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=63,
            min_samples_leaf=200, l2_regularization=1.0, random_state=s, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=40).fit(X, y)
        pr = m.predict(Xa); a,b,c2 = sc(pr, diag=True)
        v.append(a); slot.append(b); npts.append(c2); ranks.append(pl.Series(pr).rank().to_numpy())
    v=np.array(v); ens = sc(np.vstack(ranks).mean(axis=0))
    print(f"  {lbl:<26} mean {v.mean():.5f} sd {v.std(ddof=1):.5f}  delta {v.mean()-CTRL:+.5f}"
          f"  ens3 {ens:.5f} ({ens-ENS3:+.5f})  new-slot {np.mean(slot):.1f}%  new-pts {np.mean(npts):.0f}",
          flush=True)

go(STACK, "CTRL 46 feats")
go(STACK+RP3, "+ RP3-beta (4)")
