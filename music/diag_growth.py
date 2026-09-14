import polars as pl, datetime
df = pl.read_csv("interactions.csv", columns=["user_id","listened_datetime"]).with_columns(
    pl.col("listened_datetime").str.slice(0,10).alias("d"))
users = pl.read_csv("test.csv")["user_id"]
tu = df.filter(pl.col("user_id").is_in(users.implode()))

# a matched control: 1500 users sampled from the SAME play-count stratum as test users
cnt_all = df.group_by("user_id").agg(pl.len().alias("n"))
cnt_test = tu.group_by("user_id").agg(pl.len().alias("n"))
print(f"test users play-count median {cnt_test['n'].median():.0f}, pop median {cnt_all['n'].median():.0f}")
lo, hi = cnt_test["n"].quantile(.05), cnt_test["n"].quantile(.95)
pool = cnt_all.filter((pl.col("n")>=lo)&(pl.col("n")<=hi)&(~pl.col("user_id").is_in(users.implode())))
ctrl = pool.sample(min(1500,pool.height), seed=0)["user_id"]
print(f"matched control: {len(ctrl)} users from the {lo:.0f}..{hi:.0f} play-count band")
cu = df.filter(pl.col("user_id").is_in(ctrl.implode()))

print(f"\n{'window':<26}{'TEST 1500':>12}{'MATCHED CTRL':>15}{'all users':>12}")
start = datetime.date(2025,3,1)
while start < datetime.date(2025,8,31):
    w = start + datetime.timedelta(days=15)
    f = lambda x: x.filter((pl.col("d")>=str(start))&(pl.col("d")<str(w)))["user_id"].n_unique()
    a,b,c = f(tu), f(cu), f(df)
    print(f"{str(start)+'..'+str(w):<26}{a/1500*100:>11.1f}%{b/len(ctrl)*100:>14.1f}%{c:>12,}")
    start = w
