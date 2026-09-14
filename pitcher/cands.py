import csv, sys
sys.path.insert(0, '.')
from kb import flatten, load_exact, stream_player_names, stream_players, team_aliases
from match import best_leaf, explicit_date
from solve import find_player, find_team, game_key_for_date

qs = list(csv.DictReader(open('test.csv')))
teams = load_exact('teams.json'); aliases = team_aliases(teams)
names = stream_player_names('players.json')
want = set()
for n in names:
    if any(n.replace('_',' ').lower() in q['question'].lower() for q in qs): want.add(n)
players = stream_players('players.json', want)

targets = [int(x) for x in sys.argv[1].split(',')]
for row in qs:
    if int(row['ID']) not in targets: continue
    q = row['question']
    pl = find_player(q, list(players))
    tm, opp = find_team(q, aliases, subject_is_player=pl is not None)
    flat = flatten(players[pl]) if pl else (flatten(teams[tm]) if tm else [])
    ent = set()
    for e in (pl, tm, opp):
        if e: ent |= set(e.lower().split('_'))
    anchor = game_key_for_date(flat, explicit_date(q)) if explicit_date(q) else None
    scored = []
    for path, value in flat:
        b, sc = best_leaf([(path, value)], q, opponent=opp, want_player=pl is not None,
                          entity_tokens=ent, player_slug=pl, game_anchor=anchor)
        if b: scored.append((sc, path, value))
    scored.sort(key=lambda t: -t[0])
    print(f"[{row['ID']}] {q}")
    for sc, path, val in scored[:5]:
        print(f"   {sc:6.1f}  {'.'.join(path)[-92:]} = {val!r}")
    print()
