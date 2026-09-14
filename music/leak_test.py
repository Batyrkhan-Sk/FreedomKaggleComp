"""Do the item_metadata counts contain FUTURE information?

item_metadata.csv ships track_like_count / dislike / download, and nothing says
when that snapshot was taken. If it was exported AFTER the Sept 1-15 test window,
tracks that became popular DURING the test window carry inflated counts -- future
information sitting in a provided file. That would explain a 12-point cliff
between 2nd and 3rd, because it is a discontinuous advantage rather than
something you reach by better features.

The test: split the log at the holdout cut. If the metadata counts are a
pre-cut snapshot, they should correlate with PAST activity at least as well as
with FUTURE activity. If they were taken later, they will correlate BETTER with
the future -- which no legitimately-timed feature can do.

Compares, per item:
   corr(metadata count, plays BEFORE cut)
   corr(metadata count, plays AFTER  cut)   <- holdout window
and then whether metadata beats training-popularity at ranking holdout value.
"""
import numpy as np
import polars as pl

from evaluate import CUT, load, truth

train, hold, meta_dur, users = load(CUT)
m = pl.read_csv("item_metadata.csv",
                columns=["item_id", "track_like_count", "track_dislike_count",
                         "track_download_count"])

before = train.group_by("item_id").agg(pl.len().alias("n_before"))
after = hold.group_by("item_id").agg(pl.len().alias("n_after"))
d = (m.join(before, on="item_id", how="left").join(after, on="item_id", how="left")
     .with_columns(pl.col("n_before").fill_null(0), pl.col("n_after").fill_null(0))
     .filter((pl.col("n_before") + pl.col("n_after")) > 0))
print(f"{d.height:,} items with any activity\n")

def sp(a, b):
    ra = pl.Series(a).rank("average").to_numpy()
    rb = pl.Series(b).rank("average").to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])

nb = d["n_before"].to_numpy().astype(float)
na = d["n_after"].to_numpy().astype(float)
print(f"{'column':24s} {'corr w/ PAST':>13} {'corr w/ FUTURE':>15}   verdict")
for c in ("track_like_count", "track_dislike_count", "track_download_count"):
    v = d[c].to_numpy().astype(float)
    cb, ca = sp(v, nb), sp(v, na)
    flag = "  <-- FUTURE-LEANING" if ca > cb + 0.01 else ""
    print(f"{c:24s} {cb:13.4f} {ca:15.4f}{flag}")
print(f"{'(past plays themselves)':24s} {1.0:13.4f} {sp(nb, na):15.4f}   <- the honest ceiling")

# does metadata beat train-popularity at ranking holdout NEW-track value?
t = truth(hold, meta_dur, users)
hist = (train.filter(pl.col("user_id").is_in(users.implode()))
        .select("user_id", "item_id").unique())
new = t.join(hist, on=["user_id", "item_id"], how="anti")
TOTAL = new["frac"].sum()
val = new.group_by("item_id").agg(pl.col("frac").sum().alias("v"))
print(f"\nholdout new-track value {TOTAL:,.0f} pts")
print(f"{'ranking signal':24s}  top-1500 captures")
for name, col in (("train plays (legitimate)", "n_before"),
                  ("track_like_count", "track_like_count"),
                  ("track_download_count", "track_download_count")):
    top = set(d.sort(col, descending=True).head(1500)["item_id"].to_list())
    got = val.filter(pl.col("item_id").is_in(list(top)))["v"].sum()
    print(f"{name:24s}  {got:8,.0f} pts = {got/TOTAL:6.2%}")
