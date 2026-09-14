"""`age` is in whole DAYS everywhere. The timestamp has microsecond resolution.
Memory rule: don't ask "does this signal exist?", ask "can the model already see it?"
So test ONLY within integer-day ties -- information the current features cannot express."""
import polars as pl
from evaluate import truth, TOPK
CUT="2025-08-16"
users = pl.read_csv("test.csv")["user_id"]
d = (pl.scan_csv("interactions.csv")
       .select("user_id","item_id","listened_duration","listened_datetime")
       .filter(pl.col("user_id").is_in(users.implode()))
       .with_columns(pl.col("listened_datetime").str.slice(0,10).alias("d"))
       .collect())
meta = pl.read_csv("item_metadata.csv", columns=["item_id","track_duration"])
hist = d.filter(pl.col("d") < CUT)
t = truth(d.filter(pl.col("d") >= CUT), meta, users)

g = (hist.group_by(["user_id","item_id"])
        .agg(pl.col("listened_datetime").max().alias("ts"), pl.len().alias("plays"))
        .with_columns((pl.lit(CUT).str.to_date() - pl.col("ts").str.slice(0,10).str.to_date())
                      .dt.total_days().alias("age")))
g = g.join(t, on=["user_id","item_id"], how="left").with_columns(pl.col("frac").fill_null(0.0))
print(f"history pairs {g.height:,}", flush=True)

print("\nWITHIN each integer-day age bucket, split by exact time-of-day (early/late half):")
print(f"{'age(d)':>7}{'pairs':>10}{'EARLY replay%':>15}{'LATE replay%':>14}{'lift':>8}{'E frac':>9}{'L frac':>9}")
for a in range(0,8):
    b = g.filter(pl.col("age")==a)
    if b.height < 500: continue
    b = b.with_columns((pl.col("ts").str.slice(11,2).cast(pl.Int32)*3600
                    + pl.col("ts").str.slice(14,2).cast(pl.Int32)*60
                    + pl.col("ts").str.slice(17,2).cast(pl.Int32)).alias("tod")).drop_nulls("tod")
    med = b["tod"].median()
    e = b.filter(pl.col("tod") <  med); l = b.filter(pl.col("tod") >= med)
    er = (e["frac"]>0).mean()*100; lr = (l["frac"]>0).mean()*100
    print(f"{a:>7}{b.height:>10,}{er:>14.2f}%{lr:>13.2f}%{lr-er:>+8.2f}{e['frac'].mean():>9.3f}{l['frac'].mean():>9.3f}", flush=True)

print("\nWITHIN USER, among tracks tied at the SAME age, does exact-timestamp order predict replay?")
for a in (1,2,3):
    b = g.filter(pl.col("age")==a).with_columns(
        pl.col("ts").rank("ordinal",descending=True).over("user_id").alias("r"),
        pl.len().over("user_id").alias("n"))
    b = b.filter(pl.col("n")>=8).with_columns((pl.col("r")/pl.col("n")*4).ceil().clip(1,4).alias("q"))
    print(f"  age={a}  ({b.height:,} pairs, users with >=8 ties)")
    for q in (1,2,3,4):
        s = b.filter(pl.col("q")==q)
        if s.height==0: continue
        lbl = "most recent" if q==1 else ("oldest" if q==4 else f"q{q}")
        print(f"     {lbl:>12}: replay {(s['frac']>0).mean()*100:5.2f}%   mean frac {s['frac'].mean():.3f}", flush=True)
