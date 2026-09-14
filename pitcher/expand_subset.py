"""Add players the cache builder's name matching missed.

`cache_players.py` resolves a slug by literal substring of the question, so any
name whose spelling carries punctuation the slug drops never matches:
  "<First> A. <Last>" vs first_a_last
  "<First> O'<Last>"  vs first_olast       (also defeated by the possessive 's)
  "A.J. <Last>"       vs a_j_last
Those questions fall through to the TEAM subtree and are answered from team
season splits -- the wrong entity entirely.
"""
import json, csv, re
from kb import stream_players

subset = json.load(open("players_subset.json"))
extra  = json.load(open("/tmp/extra_players.json"))
missing = [k for k in extra if k not in subset]
print(f"subset has {len(subset)}; adding {len(missing)}: {missing}", flush=True)
add = stream_players("players.json", set(missing))
print(f"streamed {len(add)}: {sorted(add)}", flush=True)
subset.update(add)
json.dump(subset, open("players_subset2.json", "w"))
print(f"wrote players_subset2.json with {len(subset)} players", flush=True)

bad = []
def walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): walk(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o): walk(v, f"{p}[{i}]")
    elif isinstance(o, float): bad.append(p)
walk(add)
print(f"float leaves in the added players (must be 0): {len(bad)}")
