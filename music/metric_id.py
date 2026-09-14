"""Which metric definition reproduces the known offline->LB pairs?
v2.blend is submission #4 (NOTES: offline 0.208 -> LB 0.339)."""
import polars as pl
from evaluate import load, truth, TOPK
import v2, baselines

hist, hold, meta, users = load()
t = truth(hold, meta, users)
NA = t["user_id"].n_unique()
npl = t.group_by("user_id").agg(pl.len().alias("n"))
ideal = (t.sort(["user_id","frac"],descending=[False,True]).group_by("user_id",maintain_order=True)
          .head(TOPK).group_by("user_id").agg(pl.col("frac").sum().alias("ideal")))

def variants(recs, label):
    per = (recs.filter(pl.col("rank")<=TOPK)
             .join(t, on=["user_id","item_id"], how="left")
             .with_columns(pl.col("frac").fill_null(0.0))
             .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    per = (npl.join(per, on="user_id", how="left").with_columns(pl.col("s").fill_null(0.0))
              .join(ideal, on="user_id", how="left"))
    A = per["s"].sum()/1500/TOPK
    B = per["s"].sum()/NA/TOPK
    C = (per["s"]/per["n"].clip(upper_bound=TOPK)).sum()/NA
    D = (per["s"]/per["ideal"]).sum()/NA
    E = (per["s"]/per["n"].clip(upper_bound=TOPK)).sum()/1500
    F = (per["s"]/per["ideal"]).sum()/1500
    print(f"{label:<24} A(/50,/1500)={A:.4f}  B(/50,/act)={B:.4f}  C(/minN,/act)={C:.4f} "
          f" D(/ideal,/act)={D:.4f}  E(/minN,/1500)={E:.4f}  F(/ideal,/1500)={F:.4f}")

variants(v2.blend(hist, users), "#4 decay+trend  LB.339")
variants(baselines.popular(hist, users), "top50 popular   LB  ?")
variants(v2.hist_recency(hist, users), "history only    LB  ?")
