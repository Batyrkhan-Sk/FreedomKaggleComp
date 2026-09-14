"""How much C-value sits with users who have NO covis feature at all?
That bounds any coverage fix."""
import polars as pl
from evaluate import TOPK, truth
from train4 import load_all
inter, meta, artists, genres, users = load_all()
APPLY="2025-08-16"
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
C = pl.read_parquet("cache_cv2/apply.parquet", columns=["user_id","item_id","is_hist","cv_sum","covis"])
del inter
cov = (C.group_by("user_id").agg((pl.col("cv_sum")!=0).sum().alias("n_cv"),
                                 (pl.col("covis")!=0).sum().alias("n_cv_old")))
covered = cov.filter(pl.col("n_cv")>0)["user_id"]
print(f"users with ANY covis: {len(covered)}/1500")
act = t["user_id"].unique()
print(f"active users: {len(act)}")
nocov_active = set(act.to_list()) - set(covered.to_list())
print(f"ACTIVE users with NO covis: {len(nocov_active)}")

# oracle share held by those users (in-pool, new tracks only)
pool = C.select("user_id","item_id","is_hist").with_columns(
    pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64))
tt = t.join(pool, on=["user_id","item_id"], how="inner").filter(pl.col("is_hist")==0)
def orc(df, k=17):
    o=(df.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
        .head(k).group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    return (m["s"]/m["den"]).sum()/NA
nc = pl.Series(list(nocov_active))
print(f"\nin-pool NEW oracle @17 slots, all users        : {orc(tt):.5f}")
print(f"  ... held by ACTIVE users with NO covis       : {orc(tt.filter(pl.col('user_id').is_in(nc.implode()))):.5f}")
print(f"  ... held by users WITH covis                 : {orc(tt.filter(~pl.col('user_id').is_in(nc.implode()))):.5f}")
