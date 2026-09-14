"""Do item_id / user_id carry structure? Classic leak, never checked here.
Uses only the small metadata files so it cannot contend with a running build."""
import numpy as np, polars as pl
im = pl.read_csv("item_metadata.csv")
um = pl.read_csv("user_metadata.csv")
te = pl.read_csv("test.csv")["user_id"]
print("item_metadata cols:", im.columns)
iid = im["item_id"].to_numpy()
print(f"\nitem_id: n={len(iid):,} min={iid.min():,} max={iid.max():,} "
      f"unique={len(np.unique(iid)):,}")
# is item_id monotone with any metadata? (chronological id assignment leak)
for c in im.columns:
    if c == "item_id": continue
    s = im[c]
    if s.dtype.is_numeric():
        v = s.to_numpy().astype(float)
        ok = np.isfinite(v)
        if ok.sum() > 1000:
            r = np.corrcoef(iid[ok], v[ok])[0,1]
            if abs(r) > 0.03: print(f"  corr(item_id, {c}) = {r:+.4f}")
print("\nuser_id structure:")
uid = um["user_id"].to_numpy()
print(f"  all users: n={len(uid):,} min={uid.min():,} max={uid.max():,}")
t = te.to_numpy()
print(f"  test users: n={len(t):,} min={t.min():,} max={t.max():,}")
q_all = np.percentile(uid, [10,25,50,75,90])
q_te  = np.percentile(t,   [10,25,50,75,90])
print(f"  percentiles all : {[int(x) for x in q_all]}")
print(f"  percentiles test: {[int(x) for x in q_te]}")
# are test users drawn uniformly from the id range?
from scipy import stats
ks = stats.ks_2samp(t, uid)
print(f"  KS test (test vs all user_id): stat={ks.statistic:.4f} p={ks.pvalue:.3g}")
print("\ntest-user metadata vs population:")
j = um.join(te.to_frame().with_columns(pl.lit(1).alias("is_test")), on="user_id", how="left")
for c in um.columns:
    if c=="user_id": continue
    s=j[c]
    if s.dtype.is_numeric():
        a=j.filter(pl.col("is_test")==1)[c].drop_nulls().to_numpy()
        b=j.filter(pl.col("is_test").is_null())[c].drop_nulls().to_numpy()
        if len(a)>100 and len(b)>100:
            print(f"  {c:<28} test mean {a.mean():10.2f}   pop mean {b.mean():10.2f}   ratio {a.mean()/(b.mean()+1e-9):.2f}x")
