"""HYBRID CANDIDATE POOL -- the rejection that rested on a proxy.

Today every user gets the SAME 1,500 trending tracks (a cross join). Per-user
retrieval was rejected on RECALL (33.0% vs global 49.5%; hybrid 750+750 = 30.0%)
-- but recall was never the objective, and this project's own hardest lesson is
that co-visitation failed as a ranker (0.04) and became the best FEATURE. A pool
is judged by what the model scores with it, not by its recall.

So: keep the global top-1500 intact and ADD a per-user tail (artists the user
already plays, tracks they have not). Measured earlier: that reaches 20.5% of
the out-of-pool value. Union, never replace -- which is what the OTTO/H&M
winners do and what the earlier per-user test did not.

Scored end-to-end on base FEATS so the pool is the only thing that varies.
"""
import pathlib, sys, numpy as np, polars as pl
PER_USER = int(sys.argv[1]) if len(sys.argv) > 1 else 300
import rank4, train7
from train4 import load_all
from rank3 import target
from evaluate import TOPK, truth
from train7 import FEATS, WINDOWS, APPLY, N_POOL
from sklearn.ensemble import HistGradientBoostingRegressor

_orig = rank4.build_candidates
def hybrid(inter, meta, artists, genres, users, cut, halflife=21.0, n_pool=N_POOL):
    C = _orig(inter, meta, artists, genres, users, cut, halflife, n_pool)
    cut_dt = pl.lit(cut).str.to_date()
    h = (inter.filter(pl.col("d") < cut)
         .filter(pl.col("user_id").is_in(users.implode()))
         .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
         .filter(pl.col("age") <= 90)
         .join(artists, on="item_id", how="inner"))
    ua = (h.with_columns((0.5 ** (pl.col("age")/21.0)).alias("w"))
          .group_by(["user_id","artist_name"]).agg(pl.col("w").sum().alias("aff")))
    pop = (inter.filter(pl.col("d") < cut)
           .with_columns((cut_dt - pl.col("d").str.to_date()).dt.total_days().alias("age"))
           .filter(pl.col("age") <= 30).group_by("item_id").agg(pl.len().alias("ip")))
    cat = artists.join(pop, on="item_id", how="inner")
    extra = (ua.join(cat, on="artist_name", how="inner")
             .with_columns((pl.col("aff")*pl.col("ip").log1p()).alias("sc"))
             .join(C.select("user_id","item_id").with_columns(pl.lit(1).alias("hit")),
                   on=["user_id","item_id"], how="left")
             .filter(pl.col("hit").is_null())
             .sort(["user_id","sc"], descending=[False,True])
             .group_by("user_id", maintain_order=True).head(PER_USER)
             .with_columns(pl.int_range(pl.len()).over("user_id").add(N_POOL).cast(pl.UInt32).alias("cand_rank"),
                           pl.lit(0).alias("is_hist"))
             .select("user_id","item_id","is_hist","cand_rank"))
    print(f"    hybrid: +{extra.height:,} rows ({extra.height/len(users):.0f}/user)", flush=True)
    return pl.concat([C, extra], how="diagonal").unique(
        subset=["user_id","item_id"], keep="first").sort(["user_id","item_id"])
train7.build_candidates = hybrid

CACHE = pathlib.Path(f"cache_hyb{PER_USER}"); CACHE.mkdir(exist_ok=True)
inter, meta, artists, genres, users = load_all()
def build(name, fn):
    f = CACHE / f"{name}.parquet"
    if f.exists(): print(f"  hit {name}", flush=True); return pl.read_parquet(f)
    print(f"  building {name}", flush=True); d = fn(); d.write_parquet(f); return d
tr = pl.concat([build(f"tr_{c}", lambda c=c,h=h: (
    train7.featurise(inter,meta,artists,genres,users,c)
    .join(target(inter,meta,users,c,h),on=["user_id","item_id"],how="left")
    .with_columns(pl.col("y").fill_null(0.0)).select(FEATS+["y"]))) for c,h in WINDOWS])
C = build("apply", lambda: train7.featurise(inter,meta,artists,genres,users,APPLY))
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users); NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
del inter
X=tr.select(FEATS).to_numpy().astype(np.float32); y=tr["y"].to_numpy()
Xa=C.select(FEATS).to_numpy().astype(np.float32); base=C.select("user_id","item_id")
print(f"\ntrain {tr.height:,} (vs 7,250,683 standard)  apply {C.height:,} (vs 2,465,312)", flush=True)
# pool recall under metric C, for reference
tw=t.join(npl,on="user_id").with_columns((pl.col("frac")/pl.col("den")).alias("w"))
inp=tw.join(base.with_columns(pl.col("user_id").cast(pl.Int64),pl.col("item_id").cast(pl.Int64)),
            on=["user_id","item_id"],how="semi")
print(f"C-weighted pool recall {inp['w'].sum()/tw['w'].sum()*100:.1f}%  (standard pool = 71.9%)", flush=True)
v=[]
for s in (0,1,2):
    m=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.03,max_leaf_nodes=63,
      min_samples_leaf=200,l2_regularization=1.0,random_state=s,early_stopping=True,
      validation_fraction=0.1,n_iter_no_change=40).fit(X,y)
    D=base.with_columns(pl.Series("p",m.predict(Xa)))
    top=(D.sort(["user_id","p"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    mm=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    v.append((mm["s"]/mm["den"]).sum()/NA)
    print(f"  seed {s}: {v[-1]:.5f}", flush=True)
v=np.array(v)
print(f"\nhybrid+{PER_USER} FEATS mean {v.mean():.5f}  sd {v.std(ddof=1):.5f}")
print(f"standard pool FEATS    mean 0.37889  -> delta {v.mean()-0.37889:+.5f}")
