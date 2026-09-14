"""Submission from the decay_full history ranking + completion-weighted trending pad."""
import polars as pl
import evaluate, v2, v4

FULL = "2025-08-31"
evaluate.CUT = FULL
v2.CUT_DT = pl.lit(FULL).str.to_date()

train, holdout, meta, users = evaluate.load(FULL)
recs = v4.build(train, meta, users, halflife=21.0, mode="decay_full", days=14).sort(["user_id", "rank"])

n = recs.group_by("user_id").agg(pl.len().alias("n"))
assert n["n"].min() == 50 == n["n"].max(), f"counts {n['n'].min()}..{n['n'].max()}"
assert recs.select("user_id", "item_id").is_duplicated().sum() == 0
assert set(recs["user_id"].unique()) == set(users)

out = recs.with_row_index("id").select("id", "user_id", "item_id", "rank")
out.write_csv("submission_v4.csv")
print(f"wrote submission_v4.csv: {out.height:,} rows, {out['user_id'].n_unique()} users")
