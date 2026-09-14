"""Mechanical delta prediction for a solver change, plus the calibration log.

Task 3 has no offline harness -- there is no held-out split to score against,
only the 5 free ground-truth answers and the leaderboard integer. So every
"this should gain N questions" number this session came from judgement, and
judgement has been wrong twice: the year bucket was sized at ~6 questions on
the 92-split and delivered ~1.2.

This replaces the judgement with arithmetic and a running record:

    predicted delta = (answers changed) x (92/220) x CONVERSION

CONVERSION is the fraction of changed answers that turn a wrong answer into a
right one, NET of the ones that go the other way. It starts as a prior and is
refitted from calibration.md as real leaderboard deltas come in. After two or
three submissions the prior stops mattering, which is the entire point.

Usage:  python3 predict.py audit18.csv audit_v3.csv "year filter + tie-break"
"""
import csv
import json
import sys

SPLIT = 92 / 220          # share of questions that land in the scored 92
CONVERSION = 0.55         # PRIOR ONLY -- refit from calibration.md after 2 runs


def answers(path):
    return {r["ID"]: r["answer"] for r in csv.DictReader(open(path))}


def margins(cand_path):
    try:
        rows = json.load(open(cand_path))
    except OSError:
        return {}
    out = {}
    for r in rows:
        c = r["candidates"]
        m = 99.0
        if len(c) >= 2:
            top = c[0]["value"]
            for x in c[1:]:
                if x["value"] != top:
                    m = c[0]["score"] - x["score"]
                    break
        out[r["ID"]] = m
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    base, new = answers(sys.argv[1]), answers(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else "(unnamed change)"
    cands = margins(sys.argv[4]) if len(sys.argv) > 4 else {}

    changed = [i for i in base if base[i] != new.get(i)]
    gated = [i for i in changed if cands.get(i, 99.0) < 2.0]

    scored = len(changed) * SPLIT
    lo, mid, hi = (scored * c for c in (0.30, CONVERSION, 0.80))
    print(f"change: {label}")
    print(f"  answers changed          {len(changed)}/220")
    if cands:
        print(f"    decided by Gemma       {len(gated)}")
        print(f"    decided by the solver  {len(changed) - len(gated)}")
    print(f"  expected to land in the scored 92: {scored:.1f}")
    print(f"  predicted delta   {lo:+.1f} .. {mid:+.1f} .. {hi:+.1f} questions"
          f"   (conversion 30% / {CONVERSION:.0%} / 80%)")
    print()
    print("  record the ACTUAL leaderboard delta in calibration.md, then refit")
    print("  CONVERSION = (sum of actual deltas) / (sum of scored-change counts)")


if __name__ == "__main__":
    main()
