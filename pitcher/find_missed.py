"""Which questions name a player that exists in players.json but never resolved?

cache_players.py matches a slug by literal substring of the question, so any
name whose spelling carries punctuation the slug drops -- "<First> A. <Last>"
vs first_a_last, "<First> O'<Last>" vs first_olast -- silently fails to
resolve and the question is answered from the TEAM subtree instead.
"""
import json, csv, re, sys
from kb import stream_player_names

subset = set(json.load(open("players_subset.json")))
qs = list(csv.DictReader(open("test.csv")))
def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower())
qn = {r["ID"]: " " + re.sub(r"\s+", " ", norm(r["question"])) + " " for r in qs}

names = list(stream_player_names("players.json"))
print(f"players.json root keys: {len(names)}", flush=True)
new = {}
for n in names:
    if n in subset: continue
    spelled = norm(n.replace("_", " "))
    if len(spelled.split()) < 2: continue
    for ID, q in qn.items():
        if f" {spelled} " in q:
            new.setdefault(ID, []).append(n)
print(f"\nquestions that would NEWLY resolve to a player: {len(new)}")
byid = {r["ID"]: r["question"] for r in qs}
for ID, ns in sorted(new.items(), key=lambda x: int(x[0])):
    print(f"  ID {ID:>3} -> {','.join(ns):<28} {byid[ID][:62]}")
