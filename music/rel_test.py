"""Do activity-anchored features help, and do they help the RETURNING segment?"""
import numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
M9=["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
W2=["cv90","cv90_rank"]; STACK=FEATS+W2+M9
REL=["decay_plays_rel","decay_full_rel","recency_rel","plays_7_rel","plays_30_rel","u_gap"]
inter, meta, artists, genres, users = load_all()
def merge(n):
    return (pl.read_parquet(f"cache_cv90/{n}.parquet")
            .hstack(pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9))
            .hstack(pl.read_parquet(f"cache_rel/{n}.parquet").select(REL)))
tr=pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS]); C=merge("apply")
t=truth(inter.filter(pl.col("d")>=APPLY), meta, users); NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
prev=set(inter.filter((pl.col("d")>="2025-08-01")&(pl.col("d")<APPLY))["user_id"].unique().to_list())
del inter
y=tr["y"].to_numpy(); base=C.select("user_id","item_id")
ideal=(t.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
       .head(TOPK).group_by("user_id").agg(pl.col("frac").sum().alias("i")))
def per_user(p):
    D=base.with_columns(pl.Series("p",p)).with_columns(
        pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
       .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    return (npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
            .join(ideal,on="user_id",how="left")
            .with_columns((pl.col("s")/pl.col("den")).alias("c"),
                          pl.col("user_id").is_in(pl.Series(list(prev)).implode()).alias("act")))
def go(feats,lbl):
    X=tr.select(feats).to_numpy().astype(np.float32); Xa=C.select(feats).to_numpy().astype(np.float32)
    tot=[]; act=[]; ret=[]
    for s in (0,1,2):
        m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,max_leaf_nodes=63,
          min_samples_leaf=200,l2_regularization=1.0,random_state=s,early_stopping=True,
          validation_fraction=0.1,n_iter_no_change=40).fit(X,y)
        pu=per_user(m.predict(Xa))
        tot.append(pu["c"].mean()); act.append(pu.filter(pl.col("act"))["c"].mean())
        ret.append(pu.filter(~pl.col("act"))["c"].mean())
    print(f"  {lbl:<34} overall {np.mean(tot):.5f} (sd {np.std(tot,ddof=1):.5f}) | "
          f"active {np.mean(act):.4f} | RETURNING {np.mean(ret):.4f}", flush=True)
    return np.mean(tot), np.mean(ret)
a,ar = go(STACK,      "CTRL stack [expect .37966]")
b,br = go(STACK+REL,  "stack + activity-anchored")
print(f"\n  overall   delta {b-a:+.5f}   (bar ~0.003)")
print(f"  returning delta {br-ar:+.5f}   <- the segment this targets")
