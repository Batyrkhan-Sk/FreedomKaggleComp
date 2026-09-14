"""Resolve a question to one leaf of the knowledge base.

There is no useful template structure to exploit: 220 questions produce 208
distinct phrasings. So instead of pattern-matching whole questions, we resolve
the *entity* (a team or a player), flatten that entity's JSON subtree, and score
every leaf path against the question's wording.

Baseball data is written in abbreviations -- `era`, `whip`, `obp`, `bb` -- while
questions use English -- "earned run average", "walks". `SYNONYMS` bridges that
gap and is the main lever on accuracy.
"""

from __future__ import annotations

import heapq
import re

# question wording -> path tokens that should be treated as a match
SYNONYMS: dict[str, tuple[str, ...]] = {
    "abbreviation": ("abbr",),
    "innings": ("ip_2",),
    "walks": ("bb",),
    "intentional": ("ibb",),
    "strikeouts": ("so", "ktotal", "k"),
    "saves": ("sv", "svo", "save"),
    "runs": ("runs", "r"),
    "hits": ("h", "hits"),
    "homers": ("hr",),
    "home run": ("hr",),
    "home runs": ("hr",),
    "home": ("home",),
    "average": ("avg", "oba", "obp"),
    "batting": ("avg", "hitting"),
    "slugging": ("slg",),
    "on-base": ("obp", "oba"),
    "percentage": ("pct", "obp", "slg", "fpct"),
    "era": ("era",),
    "whip": ("whip",),
    "ratio": ("gofo", "kbb", "bbk"),
    "at-bats": ("ab",),
    "batters": ("bf",),
    "faced": ("bf",),
    "pitch": ("pitch_count", "pitching"),
    "count": ("pitch_count", "count"),
    "starters": ("starters",),
    "bullpen": ("bullpen",),
    "overall": ("overall",),
    "salary": ("salary",),
    "position": ("position", "primary_position"),
    "owner": ("owner",),
    "founded": ("founded",),
    "market": ("market",),
    "mascot": ("mascot",),
    "sponsor": ("sponsor",),
    "song": ("fight_song",),
    "affiliates": ("minorleague_affiliate",),
    "affiliate": ("minorleague_affiliate",),
    "reference": ("reference", "id"),
    "injured": ("injuries",),
    "elimination": ("elimination_number", "elim"),
    "wild": ("wild_card",),
    "back": ("games_back", "gb"),
    "wins": ("win", "wins", "w"),
    "won": ("win", "wins", "w"),
    "losses": ("loss", "losses", "l"),
    "lose": ("loss", "losses", "l"),
    "lost": ("loss", "losses", "l"),
    "rank": ("rank",),
    "games": ("games", "g", "gp"),
    "played": ("gp", "games_played"),
    "started": ("gs", "games_start"),
    "preseason": ("pre",),
    "spring": ("pre",),
    "regular": ("reg",),
    "season": ("season", "reg"),
    "singles": ("single", "singles"),
    "grounded": ("go",),
    "fly": ("fo",),
    # handedness splits are keyed by a single letter in the data
    "left-handed": ("l",),
    "right-handed": ("r",),
    "lefties": ("l",),
    "righties": ("r",),
}

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

ORDINALS = {"previous": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "2": 2, "3": 3, "4": 4, "5": 5, "last": 1}

STOP = {"the", "a", "an", "of", "in", "for", "is", "was", "what", "how", "many",
        "did", "do", "does", "are", "were", "has", "have", "had", "with",
        "and", "to", "by", "at", "on", "s", "their", "his", "her", "number"}


def tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9_\-]+", text.lower()) if t and t not in STOP]


def expand(question_tokens: list[str]) -> set[str]:
    """Question tokens plus the path tokens they imply."""
    out = set(question_tokens)
    for t in question_tokens:
        out.update(SYNONYMS.get(t, ()))
    return out


def score_path(path: tuple[str, ...], q_expanded: set[str], q_raw: list[str],
               strict_time: bool = False) -> float:
    """How well one leaf path matches the question.

    Deeper agreement is worth more than shallow: matching the leaf name itself
    ("era") should outweigh matching a container ("statistics").
    """
    score = 0.0
    for depth, part in enumerate(path):
        weight = 1.0 + depth / max(len(path) - 1, 1)      # later segments matter more
        parts = re.split(r"[_\s]+", part.lower())
        for p in parts:
            if not p:
                continue
            if p in q_expanded:
                score += 2.0 * weight
            elif len(p) > 3 and any(p in q or q in p for q in q_raw):
                score += 0.8 * weight                      # partial: "abbr" in "abbreviation"
    # a year mentioned in the question must appear in the path if the path has one
    years_q = {t for t in q_raw if re.fullmatch(r"(19|20)\d{2}", t)}
    years_p = {p for p in path if re.fullmatch(r"(19|20)\d{2}", p)}
    if years_q:
        if years_p:
            score += 6.0 if (years_q & years_p) else -6.0
        elif strict_time:
            # a season total that carries no year at all is not the 2023 answer
            score -= 4.0
    # likewise months
    # A named month scopes the answer the same way a named year does, so it
    # gets the same symmetric treatment: a path carrying the wrong month, or no
    # month at all, is answering a different question.
    months_q = {t for t in q_raw if t in MONTHS}
    if months_q:
        parts = {p.lower() for p in path}
        if parts & months_q:
            score += 6.0
        elif parts & set(MONTHS):
            score -= 6.0
        elif "month" in parts:
            score -= 6.0
        elif strict_time:
            score -= 6.0
    return score


# multi-word phrases that map to a single stat key; checked before single words
# Multi-word phrases that name one specific leaf. Checked before single words so
# that "intentional walks" resolves to ibb rather than bb.
#
# Most of these exist because the statistics are not flat: they sit in nested
# groups (onbase, runs, outs, outcome, steal, in_play, games, pitches) whose
# leaf names are two-letter codes. Without this mapping a question about
# "singles" or "earned runs" scores no better than the generic obp or run total
# sitting a level above it.
PHRASES: dict[str, str] = {
    # onbase group
    "intentional walk": "ibb",
    "from singles": "s",
    "singles": "s",
    "doubles": "d",
    "triples": "t",
    "total bases": "tb",
    "hit by pitch": "hbp",
    "fielder's choice": "fc",
    "reached on error": "roe",
    # runs group
    "earned run total": ("runs.earned", "er"),
    "earned runs": ("runs.earned", "er"),
    "earned run": ("runs.earned", "er"),
    "unearned run": "unearned",
    "inherited runner": "ir",
    # strikeout flavours
    # `outs.ktotal` is batters struck out; `outcome.ktotal` is strikes thrown,
    # four times larger. Same story for klook/kswing. Verified on one game:
    # Skubal outs.ktotal=10 over 7 innings, outcome.ktotal=37.
    "strikeouts looking": "outs.klook",
    "strikeout looking": "outs.klook",
    "strikeouts swinging": "outs.kswing",
    "strikeout swinging": "outs.kswing",
    "total strikeout": "outs.ktotal",
    "strikeout count": "outs.ktotal",
    "strikeouts": "outs.ktotal",
    "strikeout": "outs.ktotal",
    "struck out": "outs.ktotal",
    # pitch-level outcomes, which are a different question
    "swinging strike": "outcome.kswing",
    "swing strike": "outcome.kswing",
    "called strike": "outcome.klook",
    "foul ball": "outcome.foul",
    "dirt ball": "outcome.dirtball",
    # outs / batted-ball
    "ground out to fly out": "gofo",
    "ground ball to fly ball": "gbfb",
    "ground out": "go",
    "grounded out": "go",
    "fly out": "fo",
    "flied out": "fo",
    "double play": "gidp",
    "sacrifice fly": "sacfly",
    "sacrifice hit": "sachit",
    "line drive": "linedrive",
    "ground ball": "groundball",
    "fly ball": "flyball",
    "pop up": "popup",
    "batted balls in play": "bip",
    # steal group
    "caught stealing": "caught",
    "stolen base": "stolen",
    "pickoff": "pickoff",
    # games group
    "complete game": "complete",
    "quality start": "qstart",
    "shutout": "shutout",
    "blown save": "blown_save",
    "games started": "start",
    "games played": "play",
    "did not start": "start",
    "save opportunit": "svo",
    "svo": "svo",
    "holds": "hold",
    # pitches
    "pitch count": "pitches.count",
    "pitches thrown": "pitches.count",
    "number of pitches": "pitches.count",
    "total pitches": "pitches.count",
    "balls thrown": "pitches.btotal",
    "number of balls": "pitches.btotal",
    "strikes thrown": "pitches.ktotal",
    "number of strikes": "pitches.ktotal",
    # longer phrases claim their span first, so these keep "number of pitches"
    # from capturing "number of pitches per batter faced"
    "pitches per batter faced": "per_bf",
    "pitches per plate appearance": "per_pa",
    "per batter faced": "per_bf",
    "pitches per inning": "per_ip",
    "play count": "play",
    # rate stats
    "earned run average": "era",
    "on-base percentage": "obp",
    "on base percentage": "obp",
    "on-base average": "oba",
    "slugging percentage": "slg",
    "batting average": "avg",
    "at-bat": "ab",
    "batters faced": "bf",
    "runs batted in": "rbi",
    "runs": ("runs", "r"),
    "run total": ("runs", "r"),
    "extra base hit": "xbh",
    "isolated power": "iso",
    "secondary average": "seca",
    "left on base": "lob",
    "range factor": "fpct",
    "putout": "outs.po",
    "put out": "outs.po",
    "wild pitch": "wp",
    "balk": "bk",
    "hit a batter": "onbase.hbp",
    "hit batsman": "onbase.hbp",
    "on base plus slugging": "ops",
    "ops": "ops",
    "times through the order": "times_through_order",
    "babip": "babip",
    "batting average on balls in play": "babip",
    "k 9": "k9",
    "strikeouts per nine": "k9",
    "h 9": "h9",
    "hr 9": "hr9",
    "walks per nine": "bb9",
    "walk to strikeout": "bbk",
    "strikeout to walk": "kbb",
    "scoring position": "hit_risp",
    "innings pitched": "ip_2",
    "innings": "ip_2",
    # team profile
    "games back": "games_back",
    "wild card": "wild_card_back",
    "elimination number": "elimination_number",
    "rank in the national league": "rank.league",
    "rank in the american league": "rank.league",
    "league rank": "rank.league",
    "division rank": "rank.division",
    "winning percentage": "win_p",
    "last 10": "last_10_won",
    "fight song": "fight_song",
    "minor league affiliate": "minorleague_affiliate",
    "home win": "win",
    "home loss": "loss",
    "team win": "team_win",
    "team loss": "team_loss",
    "home run": "hr",
}


def phrase_keys(question: str) -> set[str]:
    """Leaf names implied by phrases in the question, longest match wins.

    Overlapping phrases must not both fire: "ground out to fly out ratio"
    contains "fly out", and letting the shorter one through makes `fo` compete
    with the `gofo` the question actually asks for.
    """
    # "extra-base hits" and "K/9" must reach the same phrases as their spaced
    # spellings. Both substitutions are one character for one, so the positions
    # used by the overlap check below stay valid.
    q = re.sub(r"[-/]", " ", question.lower())
    hits: list[tuple[int, int, str]] = []
    for phrase, key in PHRASES.items():
        pos = q.find(phrase)
        if pos >= 0:
            hits.append((pos, len(phrase), key))
    hits.sort(key=lambda h: -h[1])          # longest first
    claimed: list[tuple[int, int]] = []
    keys: set[str] = set()
    for pos, length, key in hits:
        span = (pos, pos + length)
        if any(pos < c_end and span[1] > c_start for c_start, c_end in claimed):
            continue                        # inside a longer phrase already taken
        claimed.append(span)
        keys.update((key,) if isinstance(key, str) else key)
    return keys


RECENCY = {
    "previous match": "previous match",
    "last match": "previous match",
    "last appearance": "previous match",
    "two matches ago": "2 matches ago",
    "2 matches ago": "2 matches ago",
    "three matches ago": "3 matches ago",
    "3 matches ago": "3 matches ago",
    "four matches ago": "4 matches ago",
    "4 matches ago": "4 matches ago",
    "five matches ago": "5 matches ago",
    "5 matches ago": "5 matches ago",
}


def recency_key(question: str) -> str | None:
    """"the game five matches ago" names a specific last_10_games entry."""
    q = question.lower()
    for phrase, key in RECENCY.items():
        if phrase in q:
            return key
    return None


def wants_number(question: str) -> bool:
    q = question.lower()
    return (q.startswith("how many") or "number of" in q
            or any(w in q for w in ("ratio", "percentage", "average", "count",
                                    "salary", "total")))


def is_number(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", value.strip()))


# Biographical facts sit beside the statistics under player_profile, and a
# question naming one is asking for that exact leaf -- "the height of <player>"
# was being answered from his roster `status`, and "batting hand" from a
# pitcher_hand split's batting average.
PROFILE_ATTR: dict[str, str] = {
    "height": "height",
    "weight": "weight",
    "college": "college",
    "high school": "high_school",
    "birthdate": "birthdate",
    "birth date": "birthdate",
    "date of birth": "birthdate",
    "born": "birthdate",
    "pro debut": "pro_debut",
    "debut": "pro_debut",
    "rookie year": "rookie_year",
    "batting hand": "bat_hand",
    "bats from": "bat_hand",
    "throwing hand": "throw_hand",
    "throws with": "throw_hand",
    "birth city": "birthcity",
    "birth state": "birthstate",
    "birth country": "birthcountry",
}


def profile_attr(question: str) -> str | None:
    """The player_profile leaf this question names, longest phrase winning."""
    q = re.sub(r"[-/]", " ", question.lower())
    best, best_len = None, 0
    for phrase, leaf in PROFILE_ATTR.items():
        if re.search(rf"\b{re.escape(phrase)}\b", q) and len(phrase) > best_len:
            best, best_len = leaf, len(phrase)
    return best


# asking about these means the answer lives in team_profile, not in any split
PROFILE_WORDS = {"owner", "founded", "market", "mascot", "sponsor", "nickname",
                 "abbreviation", "song", "president", "manager", "championship",
                 "affiliate", "venue", "stadium", "league", "division"}

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-", re.I)


# Every player node carries hitting, pitching and fielding side by side, and a
# leaf name like `pitch_count`, `runs` or `po` exists in more than one of them.
# When the question names a stat that belongs to exactly one family, that
# settles which subtree the answer is in -- a pitcher's pitch count is never
# read off his hitting line.
FAMILY_WORDS = {
    "pitching": ("era", "whip", "innings pitched", "innings", "saves", "save",
                 "batters faced", "pitches thrown", "balls thrown",
                 "strikes thrown", "wild pitch", "balk", "holds", "blown save",
                 "quality start", "complete game", "shutout", "k/9", "k9",
                 "pitched", "bullpen", "starting pitcher", "starters",
                 "pitches per", "earned run"),
    "hitting":  ("batting average", "rbi", "runs batted in", "at-bat", "at-bats",
                 "home run", "home runs", "doubles", "triples", "slugging",
                 "on-base", "on base", "ops", "extra-base", "extra base",
                 "stolen base", "plate appearance", "isolated power",
                 "batted balls"),
    "fielding": ("error", "errors", "putout", "putouts", "assists",
                 "fielding percentage", "range factor", "double play turned"),
}


def stat_family(question: str) -> str | None:
    """The one statistics family this question can belong to, if unambiguous."""
    # word boundaries matter: "overall" contains "era", which would silently
    # mark every question mentioning it as a pitching question
    q = re.sub(r"[-/]", " ", question.lower())
    # The stat noun alone does not settle the family: a pitcher *allows* home
    # runs and a batter *hits* them, and both questions say "home runs". The
    # verb decides, so it is checked before the vocabulary vote.
    if re.search(r"\b(allow|allowed|allowing|gave up|give up|surrender\w*)\b", q) \
            or re.search(r"\bwhile pitching\b|\bas a pitcher\b", q):
        return "pitching"
    hit = {fam for fam, words in FAMILY_WORDS.items()
           if any(re.search(rf"\b{re.escape(re.sub(r'[-/]', ' ', w))}\b", q)
                  for w in words)}
    return hit.pop() if len(hit) == 1 else None


# The splits tree has eleven dimensions (opponent, venue, month, home_away,
# day_night, hitter_hand, pitcher_hand, surface, last_start(s), total). Only
# hand, month and day/night were being enforced, so "at Wrigley Field" or "on
# turf" scored no better than a season total that ignores the qualifier.
VENUE_RE = re.compile(
    r"\bat\s+((?:[A-Z][\w.'’-]*\s+){0,3}?[A-Z][\w.'’-]*\s+"
    r"(?:Field|Park|Stadium|Ballpark|Center|Centre|Coliseum|Dome))")
VENUE_GENERIC = {"field", "park", "stadium", "ballpark", "center", "centre",
                 "coliseum", "dome", "the", "at"}
SURFACE_RE = re.compile(r"\bon\s+(turf|grass)\b|\b(turf|grass)\b", re.I)


def venue_hint(question: str) -> set[str]:
    """Distinctive tokens of a named ballpark, or empty.

    Venue keys read `pittsburgh_pnc_park` and `texas_globe_life_field`, so the
    generic suffix carries no information and is dropped -- otherwise every
    venue matches every other one.
    """
    m = VENUE_RE.search(question)
    if not m:
        return set()
    # length-1 tokens are the leftovers of possessives ("Chicago's" -> chicago, s)
    toks = {t for t in re.split(r"[^a-z0-9]+", m.group(1).lower()) if len(t) > 1}
    return toks - VENUE_GENERIC


def surface_hint(question: str) -> str | None:
    m = SURFACE_RE.search(question)
    return (m.group(1) or m.group(2)).lower() if m else None


def homeaway_hint(question: str) -> str | None:
    q = question.lower()
    if "at home" in q:
        return "home"
    if "on the road" in q:
        return "away"
    m = re.search(r"\b(home|away|road)\s+(?!runs?\b)(\w+)", q)
    if m and m.group(2) not in ("team", "plate"):
        return "home" if m.group(1) == "home" else "away"
    return None


HAND_RE = re.compile(r"\b(left|right)-handed\b", re.I)
DAYNIGHT_RE = re.compile(r"\b(day|night)\s+games?\b", re.I)
DATE_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+(\d{1,2}),?\s*(\d{4})", re.I)


def handedness(question: str) -> str | None:
    m = HAND_RE.search(question)
    return None if not m else ("l" if m.group(1).lower() == "left" else "r")


def explicit_date(question: str) -> str | None:
    """"on August 23, 2025" -> "2025-08-23", to pick a game by date."""
    m = DATE_RE.search(question)
    if not m:
        return None
    month = MONTHS[m.group(1).lower()]
    return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"


def path_ends_with(path: tuple[str, ...], keys: set[str]) -> bool:
    """Does this path end with any of `keys`, which may be dotted suffixes?

    The same two-letter leaf name lives in several groups -- `outs.ktotal` is
    batters struck out while `outcome.ktotal` is strikes thrown, and they differ
    by a factor of four. Matching on the bare leaf picks between them at random,
    so a phrase that means one of them has to name its group as well.
    """
    lower = [p.lower() for p in path]
    for key in keys:
        segs = key.split(".")
        if lower[-len(segs):] == segs:
            return True
    return False


# "What position does X play" wants `position` ("C"); "primary position of X"
# wants `primary_position` ("3B"). Both are confirmed by the two position
# questions in the provided ground truth.
POSITION_RE = re.compile(r"\b(what|which)\s+position\b|\bposition\s+does\b", re.I)
PRIMARY_POS_RE = re.compile(r"\bprimary\s+position\b", re.I)


def _scored_leaves(subtree_flat, question: str, opponent: str | None = None,
              want_player: bool = False, entity_tokens: set[str] | None = None,
              player_slug: str | None = None, game_anchor: str | None = None):
    """Yield (score, path, value) for every leaf.

    `entity_tokens` are dropped from the question before scoring. The entity has
    already been resolved, so letting "Suarez" match the path segment
    `roster.ranger_suarez` just rewards every leaf that happens to sit under his
    name -- including his preferred_name -- over the one holding the statistic
    actually asked for.
    """
    q_raw = tokens(question)
    if entity_tokens:
        q_raw = [t for t in q_raw if t not in entity_tokens]
    q_exp = expand(q_raw) | phrase_keys(question)
    forced = phrase_keys(question)
    recency = recency_key(question)
    numeric = wants_number(question)
    asks_id = any(w in question.lower() for w in ("reference number", " id ", "identifier"))
    profile_q = bool(set(q_raw) & PROFILE_WORDS)
    asks_name = question.lower().lstrip().startswith("who") or "name of" in question.lower()
    hand = handedness(question)
    venue = venue_hint(question)
    surface = surface_hint(question)
    homeaway = homeaway_hint(question)
    ql = question.lower()
    want_pen = "bullpen" in ql or "relief" in ql
    want_start = bool(re.search(r"\bstart(ing|er)s?\b", ql)) and not want_pen
    family = stat_family(question)
    dn = DAYNIGHT_RE.search(question)
    daynight = dn.group(1).lower() if dn else None
    on_date = explicit_date(question)
    asks_salary = "salary" in question.lower()
    asks_jersey = "jersey" in question.lower()
    attr = profile_attr(question)
    asks_primary_pos = bool(PRIMARY_POS_RE.search(question))
    asks_position = asks_primary_pos or bool(POSITION_RE.search(question))
    for path, value in subtree_flat:
        s = score_path(path, q_exp, q_raw, strict_time=not on_date)
        leaf = path[-1].lower() if path else ""
        if forced:
            # A phrase names one specific stat. Matching its group as well is
            # worth most, but a bare leaf match still has to beat an unrelated
            # leaf -- otherwise forcing `outs.ktotal` throws away the April
            # split in "strikeouts ... in April 2025", whose path ends in a
            # plain `ktotal` and is the better answer.
            if path_ends_with(path, forced):
                s += 8.0
            elif leaf in {k.split(".")[-1] for k in forced}:
                s += 2.0
            else:
                s -= 3.0
        if opponent:
            in_path = opponent in "/".join(path).lower()
            if recency:
                # "in the previous match against X" -- the game already pins the
                # opponent, so an opponent split is not expected in the path
                s += 6.0 if in_path else 0.0
            else:
                # "against X in the 2023 season" must come from that opponent's
                # split; a season total or a handedness split answers a
                # different question entirely
                s += 12.0 if in_path else -12.0
        # A team-level question must not be answered from one player's box score
        # nested inside a game summary, and vice versa.
        inside_player = "players" in [p.lower() for p in path]
        if inside_player and not want_player:
            s -= 8.0
        if player_slug:
            # The named player's own statistics outrank the surrounding team's.
            # Name tokens are excluded from generic scoring (see docstring), so
            # this bonus has to be applied explicitly rather than falling out of
            # token overlap.
            in_his = player_slug in [p.lower() for p in path]
            s += 9.0 if in_his else 0.0
            # A box score holds both sides' team totals next to every player's
            # own line, and the team totals are larger and score just as well on
            # the stat name. "How many fly balls did <pitcher> record" is
            # never answered by his team's fly-ball count.
            if not in_his and "statistics" in [p.lower() for p in path] \
                    and "players" not in [p.lower() for p in path]:
                s -= 12.0
        if recency:
            # anchor to the right game; without this a "previous match" question
            # can drift into season-long or venue metadata
            s += 10.0 if recency in [p.lower() for p in path] else -4.0
        if numeric:
            # "how many" cannot be answered by a stadium surface or a team name
            s += 2.0 if is_number(value) else -6.0
        # Identifiers are never the answer unless explicitly requested, and a
        # leaf whose value merely repeats a path segment is a label, not data.
        if UUID_RE.match(value) and not asks_id:
            s -= 20.0
        if leaf in ("id", "sr_id", "reference") and not asks_id:
            s -= 12.0
        # `value` and `total` are container artefacts sitting beside the real
        # named stats; they win ties they should never be in
        if leaf in ("value", "total") and forced:
            s -= 10.0
        # Only a TEXT value repeating a path segment is a label ("market" =
        # "Minnesota" under .../minnesota_twins/). A numeric one is almost
        # always colliding with a flattened ARRAY INDEX instead: `runs` = "0"
        # inside `...overall.0.month.march.runs` was losing 15 points for
        # matching the index, which buried the correct leaf under `ab`, `obp`
        # and `slg` in an exact tie. Zero is the most common value in the
        # corpus, so this fired hardest on precisely the true answers.
        # A TEXT value repeating a NAMED segment is a label, not data
        # (`market`="Minnesota" under `.../minnesota_twins/`). But the tree is
        # flattened with numeric ARRAY INDICES too, so `hr`="0" inside
        # `...overall.0.venue.minnesota_target_field.hr` matches the index and
        # loses 15 points -- dropping the correct leaf below `runs`="1". Zero
        # is the most common value in this corpus, so the penalty fired hardest
        # on true answers.
        #
        # The exemption is deliberately narrow. Gating it on "the question names
        # the leaf" was too loose: a named leaf could then win from the WRONG
        # region (Baltimore's 2025 win total became `opponent.minnesota_twins.
        # win`, and Mayza's March split became a season total with the stat
        # family flipped). Gating on `forced` -- the leaf being the target of a
        # CURATED PHRASE in this file -- keeps every genuine fix (hr, runs,
        # team_loss, complete) and excludes all three of those regressions.
        in_path = value.lower() in {p.lower() for p in path}
        index_only = in_path and value.lower() not in {
            p.lower() for p in path if not p.isdigit()}
        if in_path and not (index_only and forced
                            and leaf in {k.split(".")[-1] for k in forced}):
            s -= 15.0
        # Roster metadata (preferred_name, jersey, position...) sits under the
        # same player node as the statistics, so it has to be ruled out
        # explicitly when the question asks for a number or a named stat.
        if leaf.endswith("_name") or leaf in ("name", "first_name", "last_name",
                                              "full_name", "preferred_name",
                                              "jersey_number"):
            if not asks_name:
                s -= 14.0
        if profile_q:
            s += 8.0 if "team_profile" in [p.lower() for p in path] else -2.0
        lower_path = [p.lower() for p in path]
        if hand:
            # the split is keyed 'l'/'r' under hitter_hand or pitcher_hand;
            # picking the wrong letter silently answers the opposite question
            if any(h in lower_path for h in ("hitter_hand", "pitcher_hand")):
                s += 8.0 if hand in lower_path else -10.0
        if family:
            fams = {"hitting", "pitching", "fielding"} & set(lower_path)
            if fams:
                s += 7.0 if family in fams else -9.0
        if venue:
            if "venue" in lower_path:
                i = lower_path.index("venue")
                seg = set(lower_path[i + 1].split("_")) if i + 1 < len(path) else set()
                s += 10.0 if venue & seg else -10.0
            else:
                s -= 8.0
        if surface:
            if "surface" in lower_path:
                s += 8.0 if surface in lower_path else -10.0
            else:
                s -= 5.0
        if homeaway:
            if "home_away" in lower_path:
                s += 8.0 if homeaway in lower_path else -10.0
            else:
                s -= 4.0
        if want_pen:
            s += 7.0 if "bullpen" in lower_path else -5.0
        if want_start:
            s += 7.0 if "starters" in lower_path else -5.0
        if daynight:
            # "in day games" selects the day_night split; a season total
            # answers a different question
            if "day_night" in lower_path:
                s += 8.0 if daynight in lower_path else -10.0
            else:
                s -= 4.0
        if game_anchor:
            # the question named a calendar date; only that game's subtree counts
            s += 12.0 if game_anchor in path else -8.0
        if asks_salary:
            s += 14.0 if leaf == "salary" else -4.0
        if asks_jersey:
            s += 14.0 if leaf in ("jersey_number", "jersey") else -4.0
        if attr:
            s += 16.0 if leaf == attr else -6.0
        if not homeaway and leaf.startswith(("home_", "away_")):
            # `away_loss` matches "losses" as well as `loss` does; without a
            # home/away qualifier in the question it is the wrong leaf
            s -= 8.0
        if asks_position:
            # `fielding.positions.p.games.play` is a games-played count that
            # scores well on "position" and "play" alike; it is never the answer
            want = "primary_position" if asks_primary_pos else "position"
            s += 16.0 if leaf == want else -6.0
        if asks_id:
            # the player's own profile id, not a per-game roster entry
            s += 15.0 if "player_profile" in lower_path else -8.0
        yield s, path, value


YEAR_RE = re.compile(r"(19|20)\d{2}")
# Game-level dumps carry no year in their path but only ever hold current-season
# games, so they can never answer a question about an earlier season.
GAME_KEYS = ("last_10_games", "future_10_games")


def year_scope(path, q_years: set[str]) -> str:
    """Whether `path` sits inside, outside, or apart from the asked-for season."""
    py = {p for p in path if YEAR_RE.fullmatch(p)}
    if not py:
        return "none"
    return "in" if (py & q_years) else "wrong"


def rank_leaves(subtree_flat, question: str, k: int = 8, **kw):
    """The k best-scoring leaves, highest first.

    The solver picks one leaf and throws the rest away, but the runners-up are
    exactly what a re-ranker needs: by this point the right *region* is nearly
    always in the shortlist, and what remains is choosing between neighbouring
    leaves in it.

    A named year is a HARD scope, not a score term. `score_path` already pays
    +/-6 for a year match, but path scores span 3-62, so a better stat-name
    match routinely outbids the right season -- 13 questions were answered from
    the wrong year, and for 9 of them the right-year leaf never even reached the
    shortlist, which puts it out of reach of the re-ranker too. Filtering beats
    reweighting here: no bonus can be large enough without distorting everything
    else, and the wrong season is not a worse answer, it is a different question.

    Only applied when the requested season actually exists in this subtree, so a
    question about a year the entity has no data for still falls back to
    whatever the scorer can find rather than emitting nothing.
    """
    scored = _scored_leaves(subtree_flat, question, **kw)
    q_years = {t for t in tokens(question) if YEAR_RE.fullmatch(t)}
    if q_years:
        scored = list(scored)
        # Only drop an out-of-season path when the SAME leaf exists in the
        # requested season. Dropping unconditionally was worse than doing
        # nothing: where the season genuinely lacks that stat the scorer just
        # fell back to a nearer-but-wrong leaf, turning "runs" into "at bats"
        # and "assists as a catcher" into an overall total. A replacement has
        # to be available before the original is worth removing.
        # Game dumps are deliberately NOT filtered here even though they carry
        # no year. Dropping them cost more than it earned: "did Thielbar start
        # any games in 2020" has a usable games.start in the game data and
        # nothing but team metadata inside the 2020 subtree, so removing it
        # turned a plausible "0" into "Minnesota".
        in_leaves = {p[-1].lower() for _, p, _ in scored
                     if p and year_scope(p, q_years) == "in"}
        if in_leaves:
            scored = [t for t in scored
                      if not (t[1] and t[1][-1].lower() in in_leaves
                              and year_scope(t[1], q_years) == "wrong")]
    # Break exact ties toward a leaf the question names. The array-index
    # collision above drags true answers down into a tie with their neighbours
    # -- `runs`="0" lands level with `ab`, `obp` and `slg` for "how many runs
    # did he score" -- and the winner among equals is otherwise just whichever
    # key the JSON happened to list first. This only reorders candidates that
    # already score identically, so it cannot move an answer to a different
    # season, split or stat family the way a score adjustment did.
    q_exp_leaves = expand(tokens(question)) | phrase_keys(question)
    return heapq.nlargest(k, scored,
                          key=lambda t: (t[0], bool(t[1]) and t[1][-1].lower() in q_exp_leaves))


def best_leaf(subtree_flat, question: str, **kw):
    """Highest-scoring (path, value) for this question, or None."""
    top = rank_leaves(subtree_flat, question, k=1, **kw)
    if not top or top[0][0] <= 0.0:
        return None, (top[0][0] if top else 0.0)
    s, path, value = top[0]
    return (path, value), s
