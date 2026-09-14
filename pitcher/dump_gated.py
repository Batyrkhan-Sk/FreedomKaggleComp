"""Dump gated questions + their candidate lists in readable form, for labelling."""
import json, sys
rows = json.load(open("candidates200.json"))
MARGIN = 2.0
def margin(row):
    c = row["candidates"]
    if len(c) < 2: return 99.0
    top = c[0]["value"]
    for x in c[1:]:
        if x["value"] != top: return c[0]["score"] - x["score"]
    return 99.0
gated = [r for r in rows if r["candidates"] and margin(r) < MARGIN]
print(f"{len(gated)}/{len(rows)} gated at margin<{MARGIN}", file=sys.stderr)
lo  = int(sys.argv[1]) if len(sys.argv) > 1 else 0
hi  = int(sys.argv[2]) if len(sys.argv) > 2 else 10
K   = int(sys.argv[3]) if len(sys.argv) > 3 else 25
for r in gated[lo:hi]:
    print(f"\n{'='*78}\nID {r['ID']}  [{r['source']}]  margin={margin(r):.2f}")
    print(f"Q: {r['question']}")
    seen = {}
    for i, c in enumerate(r["candidates"][:K], 1):
        # collapse duplicate values, they are the same decision
        print(f"  {i:>3}. {c['text']}")
