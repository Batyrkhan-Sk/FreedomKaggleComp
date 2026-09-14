"""Local GENERATIVE answering via Ollama over the WIDE candidate list.

Different from local_rerank.py in two ways that matter:
  * candidates200.json (up to 200 leaves) instead of 8 -- the 8-list caps ANY
    selection method at 79/92, and the leaders sit above that;
  * the model emits a VALUE in words, not an index, then `snap()` maps it back
    to a STORED value so we always ship source bytes (exact match is the metric).

Mirrors kaggle-generative-answer.ipynb's render/snap so the notebook can
reproduce whatever this finds.
"""
import json, csv, sys, collections, urllib.request, re, time

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-coder:latest"
TOPK  = int(sys.argv[2]) if len(sys.argv) > 2 else 60
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
OUT   = sys.argv[4] if len(sys.argv) > 4 else "gen_picks.json"

SYSTEM = ("You answer questions about a baseball statistics database. You are shown "
          "the database fields that might contain the answer, each with its location "
          "and its exact stored value. Reply with ONLY the value that answers the "
          "question -- copy it exactly as shown, no units, no words, no explanation.")

rows = json.load(open("candidates200.json"))

def render(row, k):
    out = []
    for i, c in enumerate(row["candidates"][:k], 1):
        out.append(f"{i}. {c['text'].replace(' > ', ' / ')}")
    return "\n".join(out)

def ask(row, k):
    u = (f"Question: {row['question']}\n\nDatabase fields:\n{render(row, k)}\n\n"
         f"Which value answers the question? Reply with the value only.")
    body = json.dumps({"model": MODEL, "prompt": f"{SYSTEM}\n\n{u}\n\nValue:",
                       "stream": False,
                       "options": {"temperature": 0, "num_predict": 40, "seed": 0}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"].strip()

def snap(text, row):
    """Map model words back to a STORED value. Never emit model text."""
    vals = [c["value"] for c in row["candidates"]]
    t = text.strip()
    for v in sorted(set(vals), key=len, reverse=True):
        if t == str(v): return v, "exact"
    for v in sorted(set(vals), key=len, reverse=True):
        if str(v) and str(v) in t: return v, "contained"
    norm = re.sub(r"[^0-9a-z.]", "", t.lower())
    for v in sorted(set(vals), key=len, reverse=True):
        if norm and re.sub(r"[^0-9a-z.]", "", str(v).lower()) == norm: return v, "normalised"
    return (vals[0] if vals else "no answer"), "FALLBACK"

todo = list(range(len(rows)))
if LIMIT: todo = todo[:LIMIT]
print(f"{len(todo)} questions  model={MODEL} TOPK={TOPK}", flush=True)

picks, t0 = {}, time.time()
for n, i in enumerate(todo):
    row = rows[i]
    if not row["candidates"]:
        picks[row["ID"]] = ["no answer", "empty", ""]
        continue
    try:
        raw = ask(row, TOPK)
    except Exception as e:
        print(f"  ID {row['ID']} ERROR {e}", flush=True); raw = ""
    val, how = snap(raw, row)
    picks[row["ID"]] = [val, how, raw[:120]]
    if (n + 1) % 10 == 0:
        el = time.time() - t0
        print(f"  {n+1}/{len(todo)}  {el:.0f}s  ({el/(n+1):.1f}s/q)", flush=True)

json.dump(picks, open(OUT, "w"), indent=1)

base = {r["ID"]: (r["candidates"][0]["value"] if r["candidates"] else "no answer") for r in rows}
moves = [q for q, p in picks.items() if p[0] != base[q]]
print(f"\nmodel MOVED {len(moves)} of {len(picks)}")
print("snap modes:", dict(collections.Counter(p[1] for p in picks.values())))

lab = {r["ID"]: r for r in csv.DictReader(open("labels.csv"))}
byid = {r["ID"]: r for r in rows}
print("\n  ID  truth  truth_rank  solver_top1  model_pick  verdict")
tally = collections.Counter()
for ID, L in lab.items():
    if ID not in picks: continue
    truth, tr = L["truth"], L["truth_rank"]
    mp = picks[ID][0]
    if not truth: v = "n/a (no truth)"
    elif mp == truth and base[ID] == truth: v = "OK both right"
    elif mp == truth: v = "FIXED"
    elif base[ID] == truth: v = "REGRESSION"
    else: v = "both wrong"
    tally[v.split()[0] if v[0] != 'O' else v] += 1
    print(f"  {ID:>3}  {truth[:14]:>14}  {tr:>6}  {base[ID][:14]:>14}  {mp[:14]:>14}  {v}")
print("\n  ", dict(tally))
