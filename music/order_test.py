"""Are the feature ROWS emitted in a stable order?

det_test.py compared features after an explicit .sort(), so it proved the VALUES
are deterministic -- but not the row ORDER. Order matters: HistGradientBoosting
with early_stopping=True carves its validation_fraction POSITIONALLY, so a
different row order gives a different validation split, different early
stopping, and a different model from identical features.

That is the remaining suspect for the notebook still diverging from submit15
(94.65% set overlap) after featurisation was pinned.
"""
import polars as pl

from train4 import load_all
from train7 import featurise

CUT = "2025-08-16"
inter, meta, artists, genres, users = load_all()

a = featurise(inter, meta, artists, genres, users, CUT)
b = featurise(inter, meta, artists, genres, users, CUT)

ka = a.select("user_id", "item_id")
kb = b.select("user_id", "item_id")
print(f"rows {a.height:,} vs {b.height:,}")
print(f"row ORDER identical: {ka.equals(kb)}")
if not ka.equals(kb):
    n = (ka["user_id"] != kb["user_id"]).sum() + (ka["item_id"] != kb["item_id"]).sum()
    print(f"  positions differing: {n:,}")
    print("  -> validation split is positional, so the MODEL differs run to run")
sa = a.sort(["user_id", "item_id"])
sb = b.sort(["user_id", "item_id"])
print(f"after sorting, values identical: {sa.equals(sb)}")
