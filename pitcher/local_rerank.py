"""Local re-rank dry run via Ollama, to see APPROXIMATELY what the Kaggle pass would do.

This is NOT the Kaggle run. Differences that matter:
  * a 4B model here vs Gemma-3-27B in 4-bit on Kaggle -- the whole point of the
    Kaggle run is that the chooser is stronger, so treat these numbers as a FLOOR;
  * generated text here vs digit LOGITS there, so no per-call confidence -- the
    vote share across permutations is the only confidence signal;
  * a different model family, so the prompt is not tuned for it.

What it CAN answer, which is what the memory says is unknown: once position bias
is handled by permutation voting, does the chooser move anything at all, and when
it moves, is it right? `labels.csv` gives 14 hand-labelled truth ranks -- 7 already
correct at rank 1 (regression risk) and 4 winnable -- so accuracy is measurable,
not just churn.

Mirrors the notebook exactly: same MARGIN, same margin(), same crc32 permutation
seed, same vote gate semantics, and the answer is always copied VERBATIM from the
stored value.
"""
import json, csv, zlib, sys, collections, urllib.request
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma3:4b"
PERMS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MARGIN = 2.0
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0

SYSTEM = ("You match baseball questions to the correct field of a statistics database. "
          "You are given a question and a numbered list of candidate fields, each shown "
          "with the group it belongs to and its value. Reply with ONLY the number of the "
          "candidate that answers the question. No words, no explanation.")

rows = json.load(open("candidates_v7.json"))

def margin(row):
    c = row["candidates"]
    if len(c) < 2: return 99.0
    top = c[0]["value"]
    for x in c[1:]:
        if x["value"] != top:
            return c[0]["score"] - x["score"]
    return 99.0

def ask(question, cands, order):
    lines = [f"{j+1}. {cands[o]['text']}" for j, o in enumerate(order)]
    prompt = f"{SYSTEM}\n\nQuestion: {question}\n\n" + "\n".join(lines) + "\n\nNumber:"
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 4, "seed": 0}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        txt = json.loads(r.read())["response"]
    for ch in txt:
        if ch.isdigit():
            v = int(ch)
            if 1 <= v <= len(cands): return v
    return None

eligible = [i for i, r in enumerate(rows) if r["candidates"] and margin(rows[i]) < MARGIN]
eligible.sort(key=lambda i: margin(rows[i]))
if LIMIT: eligible = eligible[:LIMIT]
print(f"{len(eligible)}/{len(rows)} gated (margin < {MARGIN})  model={MODEL} PERMS={PERMS}", flush=True)

raw = {}
for n, i in enumerate(eligible):
    row = rows[i]; c = row["candidates"]; k = len(c)
    rng = np.random.default_rng(zlib.crc32(str(row["ID"]).encode()))
    votes = np.zeros(k + 1, int)
    for r in range(PERMS):
        order = np.arange(k) if r == 0 else rng.permutation(k)
        v = ask(row["question"], c, order)
        if v is not None: votes[order[v - 1] + 1] += 1
    best = int(votes.argmax())
    raw[i] = (best, float(votes[best] / PERMS))
    if (n + 1) % 10 == 0: print(f"  {n+1}/{len(eligible)}", flush=True)

moved = {i: kp for i, kp in raw.items() if kp[0] >= 2}
print(f"\nmodel wanted to MOVE {len(moved)} of {len(raw)} gated questions")
if moved:
    ps = np.array([kp[1] for kp in moved.values()])
    for g in sorted({round(v/PERMS, 4) for v in range(1, PERMS+1)}):
        print(f"  gate {g:.2f}: {(ps >= g - 1e-9).sum()} moves survive")

# --- write a submission per gate, plus the raw picks -------------------------
json.dump({str(rows[i]["ID"]): [raw[i][0], raw[i][1]] for i in raw},
          open("local_raw_picks.json", "w"), indent=1)
base =[(r["candidates"][0]["value"] if r["candidates"] else "no answer") for r in rows]
for g in sorted({round(v/PERMS, 4) for v in range(1, PERMS+1)}):
    ans = list(base)
    for i, (k, pv) in raw.items():
        if k >= 1 and pv >= g - 1e-9:
            ans[i] = rows[i]["candidates"][k - 1]["value"]
    f =f"submission_local{int(round(g*100)):03d}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["ID", "ANSWER"])
        for r, a in zip(rows, ans): w.writerow([r["ID"], a])
    ndiff = sum(1 for a, b in zip(ans, base) if a != b)
    print(f"  gate {g:.2f} -> {f}  ({ndiff} answers differ from solver)")

# --- accuracy on the 14 hand-labelled questions -----------------------------
lab = {r["ID"]: r for r in csv.DictReader(open("labels.csv"))}
byid = {r["ID"]: i for i, r in enumerate(rows)}
print("\n  ID  truth_rank  gated  model_pick  verdict")
tally = collections.Counter()
for ID, L in lab.items():
    i = byid.get(ID)
    if i is None: continue
    tr = L["truth_rank"]
    if i not in raw:
        tally["not gated"] += 1
        print(f"  {ID:>3}  {tr:>10}  no     -           (solver keeps rank 1)")
        continue
    pick = raw[i][0] or 1
    if tr == "OUT": v = "n/a (truth not in list)"
    elif tr == "1": v = "OK kept rank 1" if pick <= 1 else f"REGRESSION -> {pick}"
    elif tr.isdigit(): v = "FIXED" if pick == int(tr) else f"missed (wanted {tr})"
    else: v = f"ambiguous ({tr})"
    tally[v.split()[0]] += 1
    print(f"  {ID:>3}  {tr:>10}  yes    {pick:<11} {v}")
print("\n  ", dict(tally))
