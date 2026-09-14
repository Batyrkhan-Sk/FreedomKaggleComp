"""C  vs  B*1.29 : they only disagree for rankings with an unusual hist/new mix.
Submission #12 (discovery bias) had 38.3% new slots and scored LB 0.362."""
import polars as pl
from evaluate import TOPK, truth
from train4 import load_all
inter, meta, artists, genres, users = load_all()
APPLY="2025-08-16"
t = truth(inter.filter(pl.col("d")>=APPLY), meta, users)
NA=t["user_id"].n_unique()
npl=t.group_by("user_id").agg(pl.len().alias("n")).with_columns(pl.col("n").clip(upper_bound=TOPK).alias("den"))
D = pl.read_parquet("scored_apply.parquet")
lo,hi = D["p"].min(), D["p"].max()

def ev(bias):
    E = D.with_columns((pl.col("p") + bias*(pl.col("is_hist")==0)).alias("q"))
    top = (E.sort(["user_id","q"],descending=[False,True]).group_by("user_id",maintain_order=True).head(TOPK))
    mix = top.filter(pl.col("is_hist")==0).height/top.height
    o=(top.join(t,on=["user_id","item_id"],how="left").with_columns(pl.col("frac").fill_null(0.0))
        .group_by("user_id").agg(pl.col("frac").sum().alias("s")))
    m=npl.join(o,on="user_id",how="left").with_columns(pl.col("s").fill_null(0.0))
    B=m["s"].sum()/NA/TOPK; C=(m["s"]/m["den"]).sum()/NA
    return mix,B,C

print(f"{'new-slot%':>10}{'B':>9}{'B*1.29':>9}{'C':>9}{'C/B':>7}")
for b in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.15):
    mix,B,C = ev(b)
    print(f"{mix*100:>9.1f}%{B:>9.4f}{B*1.29:>9.4f}{C:>9.4f}{C/B:>7.3f}")
print("\nreference points (NOTES):")
print("  shipped  22.2% new slots -> LB 0.38351")
print("  #12      38.3% new slots -> LB 0.362")
