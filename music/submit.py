"""Build the real submission: train on ALL interactions, predict Sept 1-15.

Same recipe validated on the Aug 16-30 holdout, but with the cutoff moved past
the end of the data so nothing is thrown away.
"""

import argparse

import polars as pl

import evaluate
import v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halflife", type=float, default=21.0)
    ap.add_argument("--pop-days", type=int, default=14)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    # Cut past the last interaction so `train` is everything and `holdout` empty.
    FULL = "2025-08-31"
    evaluate.CUT = FULL
    v2.CUT_DT = pl.lit(FULL).str.to_date()

    train, holdout, meta, users = evaluate.load(FULL)
    print(f"train rows {train.height:,}  (holdout {holdout.height:,} -- should be ~0)")

    recs = v2.blend(train, users, args.halflife, args.pop_days)
    recs = recs.sort(["user_id", "rank"])

    n = recs.group_by("user_id").agg(pl.len().alias("n"))
    assert n["n"].min() == 50 and n["n"].max() == 50, f"bad counts: {n['n'].min()}..{n['n'].max()}"
    assert recs.height == 50 * len(users), f"{recs.height} != {50*len(users)}"
    assert recs.select("user_id", "item_id").is_duplicated().sum() == 0, "duplicate user-item"
    assert set(recs["user_id"].unique()) == set(users), "user set mismatch"

    out = recs.with_row_index("id").select("id", "user_id", "item_id", "rank")
    out.write_csv(args.out)
    print(f"wrote {args.out}: {out.height:,} rows, {out['user_id'].n_unique()} users")
    print(out.head(3))


if __name__ == "__main__":
    main()
