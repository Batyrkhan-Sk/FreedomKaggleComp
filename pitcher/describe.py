"""Render a JSON leaf path as a sentence a language model can actually read.

The previous Gemma attempt scored 0.478, below the plain heuristic, because it
was shown paths like `statistics.pitching.overall.outs.ktotal` and asked to
pick an index. Two-letter codes with no context is close to the hardest
possible framing for a small model.

The information needed to disambiguate them is not in the model, it is in this
file: `outs.ktotal` is batters struck out while `outcome.ktotal` is strikes
thrown, and saying so in words is the whole difference.
"""
from __future__ import annotations

import re

SEGMENT = {
    "reg": "regular season", "pre": "preseason", "pst": "postseason",
    "leagues": "league", "divisions": "division",
    "spring_leagues": "spring league", "game": "game",
    "totals": "season totals", "splits": "split by",
    "overall": "overall", "starters": "starting pitchers", "bullpen": "bullpen",
    "seasonal_statistics": "season statistics",
    "seasonal_splits": "season splits",
    "last_10_games": "recent game", "summary": "box score",
    "player_profile": "player profile", "team_profile": "team profile",
    "statistics": "statistics", "hitting": "batting", "pitching": "pitching",
    "fielding": "fielding", "home": "home team", "away": "away team",
    "players": "player", "roster": "roster entry", "lineup": "lineup slot",
    "opponent": "opponent", "venue": "ballpark", "month": "month",
    "home_away": "home/away", "day_night": "day/night", "surface": "field surface",
    "hitter_hand": "batter handedness", "pitcher_hand": "pitcher handedness",
    "onbase": "times on base", "outs": "outs recorded", "outcome": "pitch outcomes",
    "pitches": "pitch counts", "steal": "stolen bases", "in_play": "balls in play",
    "games": "game counts", "runs": "runs", "errors": "errors",
    "positions": "by fielding position", "standings": "standings",
    "injuries": "injuries", "scoring": "scoring by inning",
}

# Some codes mean different things depending on the group above them: `c` is
# the AL/NL Central under `divisions` but catcher under `positions`, and `l`/`r`
# is a handedness split. A flat map cannot express that, and rendering the bare
# letter tells the model nothing -- the same failure as the home/away sides.
CONTEXT = {
    ("divisions", "e"): "East division", ("divisions", "c"): "Central division",
    ("divisions", "w"): "West division",
    ("leagues", "al"): "American League", ("leagues", "nl"): "National League",
    ("spring_leagues", "acl"): "Arizona Complex League",
    ("spring_leagues", "fsl"): "Florida State League",
    ("pitcher_hand", "l"): "vs left-handed pitchers",
    ("pitcher_hand", "r"): "vs right-handed pitchers",
    ("hitter_hand", "l"): "vs left-handed batters",
    ("hitter_hand", "r"): "vs right-handed batters",
    ("positions", "c"): "catcher", ("positions", "p"): "pitcher",
    ("positions", "1b"): "first base", ("positions", "2b"): "second base",
    ("positions", "3b"): "third base", ("positions", "ss"): "shortstop",
    ("positions", "lf"): "left field", ("positions", "cf"): "center field",
    ("positions", "rf"): "right field", ("positions", "dh"): "designated hitter",
    ("surface", "turf"): "artificial turf", ("surface", "grass"): "natural grass",
    ("day_night", "day"): "day games", ("day_night", "night"): "night games",
}

LEAF = {
    "era": "earned run average", "whip": "walks+hits per inning",
    "ip_1": "outs recorded (NOT innings)", "ip_2": "innings pitched",
    "bf": "batters faced", "oba": "opponent batting average",
    "obp": "on-base percentage", "slg": "slugging percentage",
    "ops": "on-base plus slugging", "avg": "batting average",
    "babip": "batting average on balls in play", "k9": "strikeouts per 9 innings",
    "h9": "hits per 9 innings", "hr9": "home runs per 9 innings",
    "kbb": "strikeout-to-walk ratio", "bbk": "walk-to-strikeout ratio",
    "gofo": "ground-out to fly-out ratio", "gbfb": "ground-ball to fly-ball ratio",
    "ab": "at-bats", "ap": "plate appearances", "rbi": "runs batted in",
    "xbh": "extra-base hits", "tb": "total bases", "lob": "runners left on base",
    "iso": "isolated power", "seca": "secondary average", "bip": "balls in play",
    "s": "singles", "d": "doubles", "t": "triples", "hr": "home runs",
    "h": "hits", "bb": "walks", "ibb": "intentional walks",
    "hbp": "hit by pitch", "fc": "fielder's choice", "roe": "reached on error",
    "earned": "earned runs", "unearned": "unearned runs", "total": "total",
    "er": "earned runs", "a": "assists", "dp": "double plays",
    "abhr": "at-bats per home run", "runs": "runs", "hits": "hits",
    "order": "batting order slot", "rank": "rank",
    "ktotal": "strikeouts", "klook": "strikeouts looking",
    "kswing": "strikeouts swinging", "ball": "balls (pitch outcome)",
    "foul": "foul balls", "dirtball": "balls in the dirt",
    "btotal": "total balls thrown", "count": "total pitches thrown",
    "po": "putouts", "fo": "fly outs", "go": "ground outs",
    "gidp": "grounded into double play", "sacfly": "sacrifice flies",
    "sachit": "sacrifice hits", "linedrive": "line drives",
    "groundball": "ground balls", "flyball": "fly balls", "popup": "pop-ups",
    "caught": "caught stealing", "stolen": "stolen bases", "pickoff": "pickoffs",
    "win": "wins", "loss": "losses", "team_win": "team wins", "team_loss": "team losses",
    "save": "saves", "svo": "save opportunities", "hold": "holds",
    "blown_save": "blown saves", "start": "games started", "play": "games played",
    "complete": "complete games", "qstart": "quality starts", "shutout": "shutouts",
    "wp": "wild pitches", "bk": "balks", "fpct": "fielding percentage",
    "games_back": "games back", "wild_card_back": "games back in wild card",
    "elimination_number": "elimination number", "win_p": "winning percentage",
    "per_bf": "pitches per batter faced", "per_ip": "pitches per inning",
    "pitch_count": "pitches seen/thrown",
}

# leaf codes whose meaning depends on the group above them
AMBIGUOUS = {"ktotal", "klook", "kswing", "total", "hr", "h", "bb", "runs"}


def humanise(seg: str) -> str:
    s = seg.lower()
    if s in SEGMENT:
        return SEGMENT[s]
    if re.fullmatch(r"(19|20)\d\d", s):
        return s
    if re.fullmatch(r"\d+", s):
        return None                      # bare list index, carries no meaning
    return seg.replace("_", " ")


def describe(path, value: str, sides: dict | None = None) -> str:
    """One readable line for a candidate leaf.

    `sides` names which club is home and which is away for the game this leaf
    belongs to, e.g. {"home": "<home club>", "away": "<away club>"}.

    Without it a box-score leaf renders as "home team > batting > slugging",
    which is unanswerable: the question names a club, and nothing in the prompt
    says which side that club is on. Measured case -- a question about the away
    club's slugging percentage in its previous match picked `game.home`, which
    was the other club. `home_team` and `away_team` sit right beside the leaf in the
    source and were simply never shown to the model.
    """
    leaf = path[-1].lower()
    parts = []
    for i, p in enumerate(path[:-1]):
        parent = path[i - 1].lower() if i else ""
        h = CONTEXT.get((parent, p.lower())) or humanise(p)
        if sides and p.lower() in ("home", "away") and sides.get(p.lower()):
            h = f"{sides[p.lower()]} ({h})"
        if h:
            parts.append(h)
    trail = " > ".join(parts)
    name = LEAF.get(leaf, leaf.replace("_", " "))
    if leaf in AMBIGUOUS and len(path) > 1:
        group = path[-2].lower()
        name = f"{name} [{SEGMENT.get(group, group)}]"
    return f"{trail} > **{name}** = {value}"


def game_sides(flat):
    """Map a game's path prefix -> {"home": club, "away": club}.

    Keyed on the game node, so a candidate is matched to its own game by
    longest-prefix rather than by guessing the depth, which differs between the
    player and team files.
    Club names live at `<game>.home.market` + `<game>.home.name` ("Arizona" +
    "Diamondbacks"). The sibling `home_team` field is only a UUID -- rendering
    that instead is worse than rendering nothing, which is what a first pass at
    this did.
    """
    parts: dict[tuple, dict] = {}
    for path, value in flat:
        if len(path) < 2 or not isinstance(value, str):
            continue
        side, leaf = path[-2].lower(), path[-1].lower()
        if side in ("home", "away") and leaf in ("market", "name"):
            parts.setdefault(tuple(path[:-2]), {})[(side, leaf)] = value
    out: dict[tuple, dict] = {}
    for prefix, got in parts.items():
        named = {}
        for side in ("home", "away"):
            m, n = got.get((side, "market")), got.get((side, "name"))
            if m or n:
                named[side] = " ".join(x for x in (m, n) if x)
        if named:
            out[prefix] = named
    return out


def sides_for(path, table):
    """The home/away naming for whichever game `path` sits inside."""
    best = None
    for prefix, sides in table.items():
        if len(path) >= len(prefix) and tuple(path[:len(prefix)]) == prefix:
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, sides)
    return best[1] if best else None
