"""Local calibration: does list-level RRF fusion keep improving with more members?

The leaderboard already showed a 4-way fusion of scored submissions beats the best
single (0.38415 -> 0.38480). That cannot say whether MORE members keep helping,
because building another member costs a full test-window pipeline run. This scores
the same operation on the holdout, where members are cheap (one fit each).

Two fusion styles, both compared against the single-model control and against
score-averaging, which is what the shipped 5-seed ensemble already does:
  RRF   : sum_m w / (K + rank_m(item))      -- rank-based, what was submitted
  MEAN  : mean of per-model predictions     -- the shipped ensemble
"""
import numpy as np, polars as pl, collections
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import TOPK, truth
from train4 import load_all
from train7 import FEATS, WINDOWS, APPLY

M9=["cv_sum","cv_max","cv_mean","cv_top3","cv_wsum","cv_last","cv_max_rank","cv_wsum_rank","cv_top3_rank"]
STACK=FEATS+["cv90","cv90_rank"]+M9
inter, meta, artists, genres, users = load_all()
def merge(n):
    return pl.read_parquet(f"cache_cv90/{n}.parquet").hstack(
           pl.read_parquet(f"cache_cv2/{n}.parquet").select(M9))
tr=pl.concat([merge(f"tr_{c}") for c,_ in WINDOWS]); C=merge("apply")
y=tr["y"].to_numpy()
t=truth(inter.filter(pl.col("d")>=APPLY), meta, users); NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
base=C.select("user_id","item_id")
uid=base["user_id"].to_numpy(); iid=base["item_id"].to_numpy()

def score_sets(sel):
    """sel: boolean mask over rows of `base` marking the chosen 50 per user."""
    top=base.filter(pl.Series(sel))
    o=(top.join(t,on=["user_id","item_id"],how="left")
         .with_columns(pl.col("frac").fill_null(0.0))
         .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA

def topk_mask(p):
    D=base.with_columns(pl.Series("p",p)).with_row_index("ri")
    keep=(D.sort(["user_id","p"],descending=[False,True])
            .group_by("user_id",maintain_order=True).head(TOPK)["ri"].to_numpy())
    m=np.zeros(len(p),dtype=bool); m[keep]=True; return m

def ranks_of(p):
    """per-user 1-based rank of every row under prediction p (dense, ties by item_id)."""
    D=(base.with_columns(pl.Series("p",p)).with_row_index("ri")
         .sort(["user_id","p","item_id"],descending=[False,True,False]))
    D=D.with_columns(pl.int_range(1,pl.len()+1).over("user_id").alias("rk"))
    r=np.empty(len(p),dtype=np.int32); r[D["ri"].to_numpy()]=D["rk"].to_numpy(); return r

SEEDS=[0,1,2,3,4]
print(f"train {tr.height:,}  apply {C.height:,}  members {len(SEEDS)}", flush=True)
X=tr.select(STACK).to_numpy().astype(np.float32); Xa=C.select(STACK).to_numpy().astype(np.float32)
preds=[]
for s in SEEDS:
    m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,max_leaf_nodes=63,
        min_samples_leaf=200,l2_regularization=1.0,random_state=s,early_stopping=True,
        validation_fraction=0.1,n_iter_no_change=40).fit(X,y)
    p=m.predict(Xa); preds.append(p)
    print(f"  member seed={s}  C={score_sets(topk_mask(p)):.5f}", flush=True)

K=60
print("\n n   RRF        MEAN       (metric C on the holdout)", flush=True)
for n in (1,2,3,4,5):
    if n>len(preds): break
    sub=preds[:n]
    rr=np.zeros(len(uid))
    for p in sub: rr += 1.0/(K+ranks_of(p))
    mean=np.mean(sub,axis=0)
    print(f"{n:>2}   {score_sets(topk_mask(rr)):.5f}    {score_sets(topk_mask(mean)):.5f}", flush=True)
