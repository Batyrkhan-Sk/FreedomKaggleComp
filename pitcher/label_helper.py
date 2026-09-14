"""Show everything needed to hand-label one question, including what the
shortlist MISSED.

The decision this exists to serve: is the correct value even among the 8
candidates the re-ranker chooses from? If it usually is, a stronger model has
headroom and the GPU window is worth spending. If it usually is not, the
ceiling is near where we already are and no re-ranker can help.

So for each question it prints the 8 candidates AND every other leaf in the
resolved subtree whose name matches the stat the question asks about -- the
ones the scorer ranked below 8th and therefore made unreachable.
"""
import json
import re
import sys

from kb import flatten, load_exact, team_aliases
from match import phrase_keys, tokens
from solve import find_player, find_team

CAND = "candidates_v4.json"


def stat_terms(q):
    """Leaf names the question plausibly asks for."""
    t = {k.split(".")[-1] for k in phrase_keys(q)}
    return t


def main():
    ids = sys.argv[1:]
    rows = {r["ID"]: r for r in json.load(open(CAND))}
    teams = load_exact("teams.json")
    aliases = team_aliases(teams)
    players = json.load(open("players_subset.json"))
    cache = {}

    for ID in ids:
        r = rows[ID]
        q = r["question"]
        print("=" * 100)
        print(f"[{ID}] {q}")
        print(f"      source={r['source']}")
        terms = stat_terms(q)
        print(f"      phrase-implied leaves: {sorted(terms) or '(none)'}")
        print("  --- the 8 candidates the re-ranker sees ---")
        for i, c in enumerate(r["candidates"], 1):
            print(f"   {i}. {c['value']!r:<11} {'.'.join(c['path'][4:])}")

        # what else exists in the subtree with a matching leaf name?
        player = find_player(q, list(players)) if players else None
        team, opp = find_team(q, aliases, subject_is_player=player is not None)
        if player:
            flat = cache.setdefault(player, flatten(players[player]))
            ent = player
        elif team:
            flat = cache.setdefault(team, flatten(teams[team]))
            ent = team
        else:
            print("  (entity unresolved)\n")
            continue
        shown = {".".join(c["path"]) for c in r["candidates"]}
        yrs = set(re.findall(r"\b(19\d\d|20\d\d)\b", q))
        alt = []
        for path, val in flat:
            if not path or ".".join(path) in shown:
                continue
            if terms and path[-1].lower() not in terms:
                continue
            if yrs and not (yrs & {p for p in path}):
                continue
            alt.append((path, val))
        print(f"  --- entity={ent}: {len(alt)} OTHER leaves with a matching name"
              f"{' in ' + '/'.join(sorted(yrs)) if yrs else ''} (not in the 8) ---")
        for path, val in alt[:6]:
            print(f"      {val!r:<12} {'.'.join(path[3:])}")
        if len(alt) > 6:
            print(f"      ... and {len(alt)-6} more")
        print()


if __name__ == "__main__":
    main()
