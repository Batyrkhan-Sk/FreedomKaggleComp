"""Verify the music notebook reproduces the SELECTED submission.

The deliverable rule: organizers must be able to run the notebook and obtain the
submitted result. A notebook that parses is not evidence of that -- only running
its own cell source and diffing against the shipped CSV is.

Executes the notebook's OWN source (not a copy that can drift), with two
substitutions so it runs here rather than on Kaggle:
  IN -> the local data directory
  the /kaggle/working output path -> a scratch file
Then diffs against submission_final.csv, the submitted 0.38391 artifact.

Row order is not semantically meaningful (the file is sorted by user then rank),
so the comparison is on the (user_id, item_id, rank) triples.
"""
import json
import pathlib
import sys

import polars as pl

NB = pathlib.Path("music-recommender-solution.ipynb")
REF = pathlib.Path("submission_final.csv")   # the SUBMITTED 0.38391 artifact
OUT = pathlib.Path("nb_submission17.csv")

code = "".join("".join(c["source"]) + "\n"
               for c in json.loads(NB.read_text())["cells"]
               if c["cell_type"] == "code")

code = code.replace('IN = "/kaggle/input/personalized-music-recommender"', 'IN = "."')
code = code.replace('"/kaggle/working/submission.csv"', f'"{OUT}"')
assert 'IN = "."' in code and str(OUT) in code, "path substitution failed"

print(f"executing {len(code.splitlines())} lines of notebook source...", flush=True)
g = {"__name__": "__nb__"}
exec(compile(code, str(NB), "exec"), g)

got = pl.read_csv(OUT).select("user_id", "item_id", "rank").sort(["user_id", "rank"])
ref = pl.read_csv(REF).select("user_id", "item_id", "rank").sort(["user_id", "rank"])
print(f"\nnotebook {got.height:,} rows   reference {ref.height:,} rows")

if got.equals(ref):
    print(f"IDENTICAL -- the notebook reproduces {REF} exactly")
    sys.exit(0)

j = ref.join(got, on=["user_id", "rank"], how="left", suffix="_nb")
same = j.filter(pl.col("item_id") == pl.col("item_id_nb")).height
print(f"MISMATCH: {same:,}/{ref.height:,} slots agree ({same/ref.height:.2%})")
per_user = (j.with_columns((pl.col("item_id") == pl.col("item_id_nb")).alias("ok"))
            .group_by("user_id").agg(pl.col("ok").sum().alias("n")))
print(f"users fully identical: {per_user.filter(pl.col('n') == 50).height:,}")
sys.exit(1)
