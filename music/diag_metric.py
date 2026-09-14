import polars as pl
df = pl.read_csv("interactions.csv", columns=["user_id","item_id","listened_duration","listened_datetime"])
d = df.with_columns(pl.col("listened_datetime").str.slice(0,10).alias("d"))
print("interactions date range:", d["d"].min(), "->", d["d"].max(), f"  rows {d.height:,}")
print("\nrows per day, last 25 days present:")
per = d.group_by("d").agg(pl.len().alias("n")).sort("d")
print(per.tail(25))
print("\nfirst 5 days:"); print(per.head(5))

users = pl.read_csv("test.csv")["user_id"]
print(f"\ntest users {len(users)}, distinct {users.n_unique()}")
tu = d.filter(pl.col("user_id").is_in(users.implode()))
print("test-user date range:", tu["d"].min(), "->", tu["d"].max())

# active share of the 1500 in each trailing 15-day window
print("\nshare of the 1500 test users ACTIVE in each 15-day window:")
import datetime
start = datetime.date(2025,3,1)
while start < datetime.date(2025,8,31):
    hi = start + datetime.timedelta(days=15)
    n = tu.filter((pl.col("d")>=str(start)) & (pl.col("d")<str(hi)))["user_id"].n_unique()
    print(f"  {start} .. {hi}:  {n}/1500 = {n/15:.1f}%")
    start = hi

# how much truth is dropped by the meta join
meta = pl.read_csv("item_metadata.csv", columns=["item_id","track_duration"])
hold = tu.filter(pl.col("d")>="2025-08-16")
j = hold.join(meta, on="item_id", how="left")
print(f"\nholdout test-user rows {hold.height:,}")
print(f"  no meta row:        {j.filter(pl.col('track_duration').is_null()).height:,}")
print(f"  track_duration<=0:  {j.filter(pl.col('track_duration')<=0).height:,}")
print(f"  usable:             {j.filter(pl.col('track_duration')>0).height:,}")
print("\ntrack_duration percentiles:", [round(meta['track_duration'].quantile(q),1) for q in (.01,.25,.5,.75,.99)])
print("listened_duration percentiles:", [round(d['listened_duration'].quantile(q),1) for q in (.01,.25,.5,.75,.99)])
