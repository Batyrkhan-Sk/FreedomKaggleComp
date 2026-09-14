"""Extract only the players the questions mention into a small local cache.

players.json is 1.7 GB and every solver run streams all of it to reach the 108
players that 220 questions actually name -- about eight minutes per iteration,
which is most of the cost of trying an idea. The subset is a few MB.

Values stay strings throughout (kb parses with parse_float=str), and json.dump
writes them as strings, so "123.4500" survives the round trip with its trailing
zeros -- which is the whole game under exact-match scoring.
"""
import csv, json, sys
from kb import match_player_slug, stream_player_names, stream_players

OUT = 'players_subset.json'

questions = [r['question'].lower() for r in csv.DictReader(open('test.csv'))]
names = stream_player_names('players.json')
# same resolver the solver uses, so the cache cannot omit a player the solver
# would have found -- a formal given name must pull in the short-name slug
wanted = {s for s in (match_player_slug(q, names) for q in questions) if s}
wanted |= {n for n in names if any(n.replace('_', ' ') in q for q in questions)}
print(f'{len(names)} players in file, {len(wanted)} referenced by questions')

subset = stream_players('players.json', wanted)
json.dump(subset, open(OUT, 'w'))
print(f'wrote {OUT}')

# a value that lost its trailing zeros would be silently unscoreable
bad = []
def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items(): walk(v, f'{p}.{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o): walk(v, f'{p}[{i}]')
    elif isinstance(o, float):
        bad.append(p)
walk(subset)
print(f'float leaves (must be 0): {len(bad)}' + (f'  e.g. {bad[:3]}' if bad else '  OK'))
