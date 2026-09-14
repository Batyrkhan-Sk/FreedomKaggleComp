"""Dump UNGATED questions (solver was confident) -- never examined before."""
import json, sys
rows = json.load(open("candidates200.json"))
def margin(row):
    c = row["candidates"]
    if len(c) < 2: return 99.0
    top = c[0]["value"]
    for x in c[1:]:
        if x["value"] != top: return c[0]["score"] - x["score"]
    return 99.0
ung = [r for r in rows if r["candidates"] and margin(r) >= 2.0]
print(f"{len(ung)}/{len(rows)} ungated", file=sys.stderr)
lo, hi, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
for r in ung[lo:hi]:
    print(f"\n{'='*76}\nID {r['ID']}  [{r['source']}]  margin={margin(r):.1f}")
    print(f"Q: {r['question']}")
    for i, c in enumerate(r["candidates"][:K], 1):
        print(f"  {i:>2}. {c['text'][:150]}")
