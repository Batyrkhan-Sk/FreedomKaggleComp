"""Answer the MLB questions and write submission.csv.

Pipeline per question:
  1. resolve the entity  -- a team (teams.json) or a player (players.json);
     failing both, fall back to the league-wide blob
  2. flatten that entity's subtree into (path, value) leaves
  3. score every leaf against the question wording and take the best
  4. emit the value verbatim -- never reformatted

Unanswerable questions emit the literal "no answer", matching the baseline file,
so a miss costs nothing beyond the point it was never going to win.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from kb import (flatten, load_exact, match_player_slug, slugify,
                stream_players, team_aliases)
from describe import describe, game_sides, sides_for
from match import best_leaf, explicit_date, rank_leaves, tokens


def game_key_for_date(flat, iso_date: str) -> str | None:
    """Which last_10_games entry was played on `iso_date`.

    The date is stored as the value of a `scheduled` leaf, not as a path
    segment, so it cannot be matched by path scoring alone -- the containing
    game key has to be resolved first and then used as an anchor.
    """
    for path, value in flat:
        if path and path[-1] == "scheduled" and isinstance(value, str) \
                and value.startswith(iso_date):
            for seg in path:
                if "matches ago" in seg or seg == "previous match":
                    return seg
    return None

NO_ANSWER = "no answer"

# Injuries and standings live only in league.json, but both are asked about
# using a player or team as the subject -- so entity resolution sends them to
# players.json/teams.json, where the answer does not exist and the scorer picks
# the least-bad leaf it can find (a birthdate for an injury date, a save
# opportunity for a games-back). Route them explicitly instead.
STANDINGS_WORDS = ("games back", "wild card", "elimination", "standings",
                   "streak", "winning percentage", "division rank", "last 10")


def league_route(question: str, league_flat, player: str | None,
                 team: str | None):
    """The league.json subtree this question really belongs to, or None."""
    q = question.lower()
    if "injur" in q and player:
        sub = [(p, v) for p, v in league_flat
               if "injuries" in [x.lower() for x in p]
               and player in [x.lower() for x in p]]
        if sub:
            return sub
    if any(w in q for w in STANDINGS_WORDS) and team:
        sub = [(p, v) for p, v in league_flat
               if "standings" in [x.lower() for x in p]
               and team in [x.lower() for x in p]]
        if sub:
            return sub
    return None


def locate_teams(question: str, aliases: dict[str, str]) -> list[tuple[int, str]]:
    """Every team mentioned, as (character position, root key), left to right.

    Longest alias wins at each position so that "Chicago White Sox" is not
    captured by the shorter "Chicago" belonging to the Cubs.
    """
    q = question.lower()
    found: dict[int, tuple[int, str]] = {}
    for alias, key in aliases.items():
        for m in re.finditer(rf"\b{re.escape(alias)}\b", q):
            start = m.start()
            prev = found.get(start)
            if prev is None or len(alias) > prev[0]:
                found[start] = (len(alias), key)
    # drop mentions nested inside a longer one ("Chicago" inside "Chicago Cubs")
    spans = sorted((pos, ln, key) for pos, (ln, key) in found.items())
    kept: list[tuple[int, str]] = []
    for pos, ln, key in spans:
        if kept and pos < kept[-1][0] + len(kept[-1][1]):
            continue
        kept.append((pos, key))
    return kept


def find_team(question: str, aliases: dict[str, str],
              subject_is_player: bool = False) -> tuple[str | None, str | None]:
    """Return (subject team, opponent team).

    "How many X did the Mets have against the White Sox" names two teams: the
    subject whose statistics are being asked about, and the opponent that
    selects a split. Taking the longest alias would pick the White Sox and
    silently answer a different question, so the word "against" decides roles.
    """
    mentions = locate_teams(question, aliases)
    if not mentions:
        return None, None
    q = question.lower()
    against = [m.end() for m in re.finditer(r"\bagainst\b", q)]
    if len(mentions) == 1:
        # "<player>'s OPS against <team>" names one team, and it is the
        # opponent, not the subject -- the subject is the player. Treating it as
        # the subject silently drops the opponent split and answers with a
        # season total instead.
        pos, key = mentions[0]
        if subject_is_player and against and pos > against[0]:
            return None, key
        return key, None
    if against:
        after = [(pos, key) for pos, key in mentions if pos > against[0]]
        before = [(pos, key) for pos, key in mentions if pos < against[0]]
        if after and before:
            return before[-1][1], after[0][1]
        if after:
            return None, after[0][1]
    return mentions[0][1], mentions[-1][1]


def find_player(question: str, names: list[str]) -> str | None:
    """Root keys are slugs ("miles_mikolas"); questions spell names out."""
    return match_player_slug(question, names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", default="teams.json")
    ap.add_argument("--league", default="league.json")
    ap.add_argument("--players", default="players.json")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--out", default="submission_pred.csv")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="a wrong guess scores the same as 'no answer', so guess by default")
    ap.add_argument("--audit", default="audit.csv",
                    help="write question -> chosen path -> answer, for error analysis")
    ap.add_argument("--debug", type=int, default=0, help="print the first N resolutions")
    ap.add_argument("--candidates", default=None,
                    help="write the top-K leaves per question, for LLM re-ranking")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--player-cache", default="players_subset.json",
                    help="pre-extracted player subtrees; default keeps the verified behaviour")
    args = ap.parse_args()

    questions = list(csv.DictReader(open(args.test)))
    print(f"{len(questions)} questions")

    teams = load_exact(args.teams)
    league = load_exact(args.league)
    aliases = team_aliases(teams)
    print(f"teams {len(teams)}  aliases {len(aliases)}")

    players: dict = {}
    cache = Path(args.player_cache)
    if cache.exists():
        # cache_players.py pulled the 108 referenced players out of the 1.7 GB
        # file once; re-streaming it every run costs eight minutes an iteration
        import json
        players = json.load(open(cache))
        print(f"players from {cache} ({len(players)})")
    elif Path(args.players).exists():
        from kb import stream_player_names
        all_names = stream_player_names(args.players)
        wanted = set()
        for n in all_names:
            spelled = n.replace("_", " ").lower()
            if any(spelled in q["question"].lower() for q in questions):
                wanted.add(n)
        # ALSO run the fuzzy resolver, exactly as cache_players.py does. Literal
        # substring alone misses every player whose question spells the name
        # differently from the slug (formal given name -> short-name slug), so a
        # run WITHOUT players_subset.json used to resolve one player fewer than a
        # run with it -- i.e. the shipped answers silently depended on a cache
        # file that no clean checkout could rebuild.
        wanted |= {s for s in (match_player_slug(q["question"].lower(), all_names)
                               for q in questions) if s}
        print(f"players in file {len(all_names)}, referenced by questions {len(wanted)}")
        players = stream_players(args.players, wanted)
    else:
        print("players.json missing -- team/league questions only")

    league_flat = flatten(league)
    team_flat: dict[str, list] = {}
    player_flat: dict[str, list] = {}

    answers, audit, cand_rows = [], [], []
    side_cache: dict = {}
    stats = {"team": 0, "player": 0, "league": 0, "none": 0}
    for row in questions:
        q = row["question"]
        player = find_player(q, list(players)) if players else None
        team, opponent = find_team(q, aliases, subject_is_player=player is not None)

        routed = league_route(q, league_flat, player, team)
        if routed is not None:
            flat, source = routed, "league"
        elif player:
            flat = player_flat.setdefault(player, flatten(players[player]))
            source = "player"
        elif team:
            flat = team_flat.setdefault(team, flatten(teams[team]))
            source = "team"
        else:
            flat, source = league_flat, "league"

        ent_tokens: set[str] = set()
        if player:
            ent_tokens |= set(player.lower().split("_"))
        if team:
            ent_tokens |= set(team.lower().split("_"))
        if opponent:
            ent_tokens |= set(opponent.lower().split("_"))
        anchor = None
        iso = explicit_date(q)
        if iso:
            anchor = game_key_for_date(flat, iso)
        kw = dict(opponent=opponent, game_anchor=anchor,
                  want_player=player is not None, entity_tokens=ent_tokens,
                  player_slug=player)
        if args.candidates:
            ranked = rank_leaves(flat, q, k=args.topk, **kw)
            sides_tbl = side_cache.setdefault(id(flat), game_sides(flat))
            # Collapse candidates that hold the SAME VALUE.
            #
            # The answer is the value, so two paths carrying "0" are the same
            # answer wearing different clothes -- picking between them is a
            # decision with no consequence. Measured over the 87 gated
            # questions the shortlist averages only 5.4 distinct values out of
            # 8 shown, and 10 questions are a binary choice presented as an
            # eight-way one. Every redundant option is a chance to pick wrong
            # for no possible gain.
            #
            # The highest-scoring path for each value is kept as its
            # representative, and the count of other paths agreeing on it is
            # surfaced -- corroboration the flat list was throwing away.
            seen: dict[str, dict] = {}
            for sc, p, v in ranked:
                hit = seen.get(v)
                if hit is None:
                    seen[v] = {"path": list(p), "value": v, "score": round(sc, 1),
                               "text": describe(p, v, sides_for(p, sides_tbl)),
                               "n_paths": 1}
                else:
                    hit["n_paths"] += 1
            cands = list(seen.values())
            for c in cands:
                if c["n_paths"] > 1:
                    c["text"] += f"   [{c['n_paths']} fields agree on this value]"
            cand_rows.append({
                "ID": row["ID"], "question": q, "source": source,
                "candidates": cands,
            })
            best, score = ((ranked[0][1], ranked[0][2]), ranked[0][0]) \
                if ranked and ranked[0][0] > 0 else (None, 0.0)
        else:
            best, score = best_leaf(flat, q, **kw)
        if best is None or score < args.min_score:
            answers.append(NO_ANSWER)
            stats["none"] += 1
            audit.append((source, "-", NO_ANSWER, f"{score:.1f}"))
        else:
            answers.append(best[1])
            stats[source] += 1
            audit.append((source, ".".join(best[0]), best[1], f"{score:.1f}"))

        if args.debug and len(answers) <= args.debug:
            path = ".".join(best[0]) if best else "-"
            print(f"  [{row['ID']}] {q[:70]}")
            print(f"        -> {source}:{path}  = {answers[-1]!r}  (score {score:.1f})")

    if args.candidates:
        import json as _json
        with open(args.candidates, "w") as f:
            _json.dump(cand_rows, f, indent=1)
        print(f"wrote {len(cand_rows)} candidate sets to {args.candidates}")

    if args.audit:
        with open(args.audit, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ID", "question", "source", "path", "answer", "score"])
            for row, rec in zip(questions, audit):
                w.writerow([row["ID"], row["question"], *rec])
        print(f"wrote audit trail to {args.audit}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "ANSWER"])
        for row, ans in zip(questions, answers):
            w.writerow([row["ID"], ans])

    print(f"\nresolved via {stats}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
