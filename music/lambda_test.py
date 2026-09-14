"""PAIRWISE ranking objective -- research-backed, and NOTES flagged the first
attempt as needing a retry with tuned pair sampling.

Why it should matter here: pointwise MSE spends model capacity making scores
globally comparable across all 1,500 users, but the metric only ever takes each
user's OWN top-50 -- within-user order is the only thing that scores. LambdaRank
optimises exactly that and ignores cross-user calibration. The RecSys-2018
winner used a pairwise GBM re-ranker over blended retrieval for the same reason.

Key knob the first attempt likely missed: lambdarank_truncation_level defaults
to 30, but our metric is top-50, so gradients for positions 31-50 were being
discarded. Labels are y*4 -> integers {0..4}, which is graded relevance rather
than binary.
"""
import numpy as np, polars as pl, lightgbm as lgb
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY
M9=["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
W2=["cv90","cv90_rank"]; STACK=FEATS+W2+M9
inter, meta, artists, genres, users = load_all()
def merge(n):
    return (pl.read_parquet(f"cache_cv90/{n}.parquet")
            .hstack(pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9)))
parts=[merge(f"tr_{c}") for c,_ in WINDOWS]
uids =[pl.read_parquet(f"cache_uid/tr_{c}.parquet")["user_id"].to_numpy() for c,_ in WINDOWS]
tr=pl.concat(parts); C=merge("apply")
t=truth(inter.filter(pl.col("d")>=APPLY), meta, users); NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
base=C.select("user_id","item_id")
y=tr["y"].to_numpy(); lab=np.rint(y*4).astype(np.int32)
# groups: contiguous runs of user_id WITHIN each window, concatenated in order
groups=np.concatenate([np.unique(u, return_counts=True)[1] for u in uids])
assert groups.sum()==tr.height, (groups.sum(), tr.height)
X=tr.select(STACK).to_numpy().astype(np.float32)
Xa=C.select(STACK).to_numpy().astype(np.float32)
print(f"train {tr.height:,}  groups {len(groups):,}  positives {(lab>0).mean()*100:.2f}%", flush=True)
def sc(p):
    D=base.with_columns(pl.Series("p",p))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA
for trunc, nl, ne in ((50,63,300),(50,63,600),(30,63,300)):
    v=[]
    for s in (0,1):
        m=lgb.LGBMRanker(objective="lambdarank", n_estimators=ne, learning_rate=0.05,
            num_leaves=nl, min_child_samples=200, lambda_l2=1.0, random_state=s,
            lambdarank_truncation_level=trunc, label_gain=[0,1,2,3,4],
            n_jobs=-1, verbose=-1).fit(X, lab, group=groups)
        v.append(sc(m.predict(Xa)))
    v=np.array(v)
    print(f"  lambdarank trunc={trunc} leaves={nl} n={ne}: mean {v.mean():.5f} "
          f"sd {v.std(ddof=1):.5f}  delta {v.mean()-0.37966:+.5f}", flush=True)
