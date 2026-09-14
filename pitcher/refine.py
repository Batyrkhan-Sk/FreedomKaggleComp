"""Deterministic post-rank rules over the WIDE candidate list.

WHY A SEPARATE PASS. `match.py`'s scorer is 34 KB of interacting terms and every
past attempt to retune it scored 0.00 or worse on the leaderboard. These rules
instead run AFTER ranking and are structured as a FILTER CHAIN, not first-match-
wins: each step removes candidates that are structurally disqualified, then a
preference step picks among what survives. First-match-wins was tried and was
wrong -- on "division elimination number" the season-type rule fired first and
returned the plain `elimination_number`, pre-empting the more specific leaf.

Every filter carries the guard the year filter earned in `rank_leaves`:
ONLY DISQUALIFY A CANDIDATE WHEN AN EQUIVALENT REPLACEMENT ACTUALLY EXISTS.
A question matching no rule keeps the solver's answer byte-for-byte.
"""
import json, re, sys, collections

SIDE = re.compile(r'([A-Z][A-Za-z .]+?) \((home|away) team\)')
PREW = re.compile(r'preseason|spring|exhibition', re.I)

def side_of(c):
    m = SIDE.search(c["text"])
    return m.group(1).strip() if m else None

def pslug(c):
    p = [s.lower() for s in c["path"]]
    for key in ("players", "roster"):
        if key in p:
            i = p.index(key)
            if i + 1 < len(p): return p[i + 1]
    return None

def is_team_agg(c):
    p = [s.lower() for s in c["path"]]
    return "statistics" in p and "players" not in p and "roster" not in p

def leaf(c): return c["path"][-1].lower()

def named_player(r):
    """A players/<slug> in this question's candidates whose name is in the text."""
    ql, best = r["question"].lower(), None
    for c in r["candidates"]:
        s = pslug(c)
        if not s: continue
        hits = sum(1 for t in s.split("_") if len(t) > 2 and t in ql)
        if hits and (best is None or hits > best[1]): best = (s, hits)
    return best[0] if best else None


# ---- filters: each returns a (possibly shorter) candidate list ---------------

GAME_KEYS = ("last_10_games", "future_10_games")
YEAR = re.compile(r"^(19|20)\d\d$")
PSTW = re.compile(r"postseason|playoff|world series", re.I)
DATED = re.compile(r"\b(january|february|march|april|may|june|july|august|september|"
                   r"october|november|december)\s+\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b", re.I)

def is_game_dump(c): return any(g in c["path"] for g in GAME_KEYS)
def years_in(c):     return {p for p in c["path"] if YEAR.fullmatch(p)}


def filter_year_gamedump(r, cands):
    """A year-scoped question answered from a year-less box score.

    `rank_leaves` scopes a named year HARD but deliberately exempts game dumps,
    because they carry no year segment and dropping them unconditionally cost
    more than it earned. Exempting them entirely is the other extreme: 'in the
    2022 preseason' then gets answered from a game played last week. Guarded the
    same way the year filter is -- drop a game dump only when the SAME leaf
    exists inside the requested season.
    """
    qy = set(re.findall(r"\b(?:19|20)\d\d\b", r["question"]))
    if not qy: return cands
    # An explicit DATE names one specific game, so the game dumps are exactly
    # what the question is about -- dropping them threw away the right answer
    # ("in the game against the Orioles on August 23, 2025" fell back to a
    # season split against Baltimore).
    if DATED.search(r["question"]): return cands
    replaceable = {leaf(c) for c in cands if not is_game_dump(c) and years_in(c) & qy}
    out = [c for c in cands if not (is_game_dump(c) and leaf(c) in replaceable)]
    return out or cands


def want_season_type(q):
    if PSTW.search(q): return "PST"
    if PREW.search(q): return "PRE"
    return "REG"


def filter_season_type(r, cands):
    """Keep the season type the question asks for.

    Standings and season splits carry PRE, REG and PST copies of the same field
    and the scorer has no season-type preference, so it takes whichever ranks
    first -- answering 'games back in the wild card race' with 0.0 from spring
    training. Three-way, because a PRE-only rule sends a postseason question to
    the regular season, which is just as wrong. Guarded: a candidate is dropped
    only when the SAME leaf exists under the wanted type, so a genuinely
    spring-only question ('in March 2021', no regular-season March) is untouched.
    """
    want = want_season_type(r["question"])
    have = {leaf(c) for c in cands if want in c["path"]}
    out = [c for c in cands
           if not ({"PRE", "REG", "PST"} & set(c["path"])
                   and want not in c["path"] and leaf(c) in have)]
    return out or cands


PITCHW = re.compile(r"\bpitch(er|ers|ing)\b|\ballow(ed)?\b|\bbullpen\b", re.I)
BATW   = re.compile(r"\bbatt(er|ers|ing)\b|\bhitt(er|ers|ing)\b|\boffense\b", re.I)


def stat_branch(c):
    p = [s.lower() for s in c["path"]]
    if "pitching" in p: return "pitching"
    if "hitting" in p or "batting" in p: return "hitting"
    return None


def filter_stat_branch(r, cands):
    """The question says 'pitchers' and the answer came from the batting line.

    Both branches carry identically-named leaves (`iw`, `ktotal`), so the scorer
    has nothing to separate them and takes whichever ranks first. Only applied
    when the question names exactly ONE side, so 'walks allowed by batters'
    style phrasings that mention both are left alone.
    """
    wp, wb = bool(PITCHW.search(r["question"])), bool(BATW.search(r["question"]))
    if wp == wb: return cands
    want = "pitching" if wp else "hitting"
    have = {leaf(c) for c in cands if stat_branch(c) == want}
    out = [c for c in cands
           if not (stat_branch(c) not in (None, want) and leaf(c) in have)]
    return out or cands


PITCH_POS = {"P", "SP", "RP", "CP"}
_players = None

def _load_players():
    """players_subset.json, only for primary_position. Optional by design."""
    global _players
    if _players is None:
        try: _players = json.load(open("players_subset.json"))
        except Exception: _players = {}
    return _players


def filter_pitcher_role(r, cands):
    """A pitcher's BATTING line answering a question about him.

    'How many strikeouts did <pitcher> achieve in <month> <year>' returned the
    times he struck out AT THE PLATE -- when he is a starting pitcher and the
    answer is the strikeouts he threw. The question carries no pitching keyword, so
    `filter_stat_branch` cannot fire; the disambiguator is the player's own
    `primary_position`, which is sitting in the data.
    """
    ps = _load_players()
    if not ps: return cands
    # If the question names a branch itself, `filter_stat_branch` is the
    # authority and this rule must not second-guess it: a question that names
    # the HITTING statistics of a pitcher is asking for the batting line
    # precisely because it is unusual for a pitcher.
    if PITCHW.search(r["question"]) or BATW.search(r["question"]): return cands
    q = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", r["question"].lower())).strip() + " "
    slug = next((s for s in ps
                 if f" {re.sub(r'[^a-z0-9 ]', ' ', s.replace('_', ' ')).strip()} " in q), None)
    if not slug: return cands
    try:
        prof = ps[slug]["data"]["player_profile"]["player"]
        pos = prof.get("primary_position") or prof.get("position")
    except Exception:
        return cands
    if pos not in PITCH_POS: return cands
    have = {leaf(c) for c in cands if stat_branch(c) == "pitching"}
    out = [c for c in cands
           if not (stat_branch(c) == "hitting" and leaf(c) in have)]
    return out or cands


def filter_club_side(r, cands):
    """Drop the other dugout when the question names a club.

    `describe.py` computes the home/away club names but only to RENDER them for
    the LLM re-ranker -- which was a no-op in every shipped submission, so the
    scorer has never once seen which club a box-score leaf belongs to.
    Blocked whenever the question names a PLAYER: then the club mentioned is the
    opponent and the answer belongs on the player's OWN side.
    """
    if named_player(r): return cands
    ql = r["question"].lower()
    clubs = {side_of(c) for c in cands} - {None}
    named = {cl for cl in clubs if cl.split()[-1].lower() in ql}
    if not named: return cands
    out = [c for c in cands if side_of(c) is None or side_of(c) in named]
    return out if any(side_of(c) in named for c in out) else cands


def filter_player_line(r, cands):
    """Drop team totals when the question names a player with his own line."""
    sl = named_player(r)
    if not sl: return cands
    if not any(pslug(c) == sl for c in cands): return cands
    out = [c for c in cands if not (is_team_agg(c) and
                                    leaf(c) in {leaf(x) for x in cands if pslug(x) == sl})]
    return out or cands


# ---- preferences: reorder what survived --------------------------------------

def prefer_exact_leaf_phrase(r, cands):
    """The question spells a leaf name out in full and a shorter leaf won.

    'division elimination number' is a real field, `division_elimination_number`,
    sitting beside the plain `elimination_number` that outranks it.
    """
    ql = re.sub(r"\s+", " ", " " + re.sub(r"[^a-z0-9 ]", " ", r["question"].lower()) + " ")
    def key(c):
        name = leaf(c).replace("_", " ")
        return len(name) if (" " in name and len(name) >= 12 and f" {name} " in ql) else 0
    return sorted(cands, key=key, reverse=True) if any(key(c) for c in cands) else cands


def prefer_reference(r, cands):
    """'reference number' is the `reference` field, not the UUID `id`."""
    if "reference number" not in r["question"].lower(): return cands
    return sorted(cands, key=lambda c: leaf(c) == "reference", reverse=True)


FILTERS     = [filter_year_gamedump, filter_season_type, filter_stat_branch,
               filter_pitcher_role, filter_club_side, filter_player_line]
PREFERENCES = [prefer_exact_leaf_phrase, prefer_reference]


def keep_original_leaf(orig_leaf, cands):
    """Filters choose WHICH COPY of a field; they must not change WHICH FIELD.

    Dropping candidates lets an unrelated leaf win by score: removing the
    batting `slg` copies for a pitcher promoted the batting `ops` (0.0) rather
    than the pitching `slg` (0.25). So once the filters have run, any survivor
    carrying the solver's original leaf goes first. Changing the field is the
    job of the PREFERENCES, which run after this and are allowed to override it.
    """
    same = [c for c in cands if leaf(c) == orig_leaf]
    return same + [c for c in cands if leaf(c) != orig_leaf] if same else cands


def refine_row(r):
    cands = list(r["candidates"])
    orig_leaf = leaf(cands[0])
    applied = []
    for f in FILTERS:
        new = f(r, cands)
        if new is not cands and [id(x) for x in new] != [id(x) for x in cands]:
            applied.append(f.__name__)
        cands = new
    cands = keep_original_leaf(orig_leaf, cands)
    for p in PREFERENCES:
        new = p(r, cands)
        if new and new[0] is not cands[0]: applied.append(p.__name__)
        cands = new
    return cands[0]["value"], applied


def refine(rows):
    out, fired = {}, []
    for r in rows:
        if not r["candidates"]:
            out[r["ID"]] = "no answer"; continue
        base = r["candidates"][0]["value"]
        pick, applied = refine_row(r)
        out[r["ID"]] = pick
        if pick != base:
            fired.append((r["ID"], ",".join(applied) or "?", base, pick, r["question"]))
    return out, fired


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "candidates200.json"
    rows = json.load(open(src))
    ans, fired = refine(rows)
    print(f"{len(fired)} of {len(rows)} answers changed\n")
    for ID, why, b, p, q in fired:
        print(f"  ID {ID:>3} [{why[:38]:<38}] {b[:16]!r} -> {p[:16]!r}")
        print(f"          {q[:86]}")
    print("\nby rule:", dict(collections.Counter(f[1] for f in fired)))
