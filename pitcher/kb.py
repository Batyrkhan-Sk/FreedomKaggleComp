"""Knowledge base over the MLB JSON dumps, with exact value preservation.

The single most important detail in this task is *formatting*. Submissions are
scored by exact string match against the original JSON value, and the rules call
this out explicitly: 123.4500 must stay "123.4500", not become "123.45".
Python's default JSON parser turns that literal into a float and destroys the
trailing zeros, so every number is loaded with `parse_float=str` /
`parse_int=str` and never round-trips through a numeric type.

players.json is 1.7 GB, which would become many gigabytes as a Python dict, so
it is streamed and only the players actually named in the questions are kept.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

STR_PARSE = {"parse_float": str, "parse_int": str, "parse_constant": str}


def load_exact(path: str | Path) -> dict:
    """Load JSON preserving the literal text of every number."""
    with open(path) as f:
        return json.load(f, **STR_PARSE)


def _iter_top_level(path: str | Path):
    """Yield (key, raw_json_text) for each entry of a top-level JSON object.

    players.json is 1.7 GB. Parsing it whole costs many gigabytes of dicts, and
    the streaming libraries are not in Kaggle's image, so this walks the file in
    chunks tracking brace depth and hands back each value as raw text. Callers
    then json.loads only the entries they need -- with parse_float=str, so the
    literal digits survive.
    """
    with open(path, "r") as f:
        buf, depth, in_str, esc = "", 0, False, False
        key, val_start, seeking_key = None, None, True
        pos = 0
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            i = pos
            while i < len(buf):
                ch = buf[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                        if depth == 1 and seeking_key:
                            key = json.loads(buf[key_start : i + 1])
                            seeking_key = False
                elif ch == '"':
                    in_str = True
                    if depth == 1 and seeking_key:
                        key_start = i
                elif ch == "{":
                    depth += 1
                    if depth == 2 and key is not None and val_start is None:
                        val_start = i
                elif ch == "}":
                    depth -= 1
                    if depth == 1 and val_start is not None:
                        yield key, buf[val_start : i + 1]
                        buf = buf[i + 1 :]
                        i, key, val_start, seeking_key = -1, None, None, True
                    elif depth == 0:
                        return
                elif ch == "," and depth == 1:
                    seeking_key = True
                i += 1
            pos = max(len(buf) - 1, 0) if val_start is None else len(buf)
            if val_start is not None:
                pos = len(buf)


def stream_player_names(path: str | Path) -> list[str]:
    """Root keys of players.json without materialising the file."""
    return [key for key, _ in _iter_top_level(path)]


def stream_players(path: str | Path, wanted: set[str]) -> dict:
    """Load only the named players, preserving every number's literal text."""
    out: dict = {}
    for key, raw in _iter_top_level(path):
        if key in wanted:
            out[key] = json.loads(raw, **STR_PARSE)
            if len(out) == len(wanted):
                break
    return out


def flatten(obj, prefix: tuple = ()) -> list[tuple[tuple, str]]:
    """Depth-first (path, leaf-value) pairs.

    Leaves are returned as strings because that is what the submission needs;
    nothing is reformatted on the way out.
    """
    out: list[tuple[tuple, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "_comment":
                continue
            out.extend(flatten(v, prefix + (str(k),)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(flatten(v, prefix + (str(i),)))
    elif obj is not None:
        out.append((prefix, obj if isinstance(obj, str) else str(obj)))
    return out


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def team_aliases(teams: dict) -> dict[str, str]:
    """Map every way a team is written in a question to its root key.

    Questions use the full "Cincinnati Reds", but also bare "Reds", the market
    "Cincinnati", and abbreviations, so all of them index the same entry.
    """
    alias: dict[str, str] = {}
    for key, blob in teams.items():
        profile = blob.get("data", {}).get("team_profile", {})
        market, name, abbr = (profile.get("market", ""), profile.get("name", ""),
                              profile.get("abbr", ""))
        for form in (f"{market} {name}", name, abbr, key.replace("_", " ")):
            if form and form.strip():
                alias.setdefault(form.lower().strip(), key)
    return alias

def match_player_slug(question: str, slugs) -> str | None:
    """Resolve a player named in the question to a root key.

    Root keys use the name the player goes by, while questions sometimes use
    the formal one -- a formal given name has to reach the short-name slug. So an
    exact spelled-out match is tried first, and failing that a surname match
    where one given name is a prefix of the other.
    """
    q = question.lower()
    hit, hit_len = None, 0
    for slug in slugs:
        spelled = slug.replace("_", " ")
        if len(spelled) > hit_len and re.search(rf"\b{re.escape(spelled)}\b", q):
            hit, hit_len = slug, len(spelled)
    if hit:
        return hit

    for slug in slugs:
        parts = slug.split("_")
        if len(parts) < 2:
            continue
        first, last = parts[0], " ".join(parts[1:])
        if not re.search(rf"\b{re.escape(last)}\b", q):
            continue
        m = re.search(rf"\b([a-z\u00c0-\u024f.'-]+)\s+{re.escape(last)}\b", q)
        if not m:
            continue
        qfirst = m.group(1)
        if qfirst.startswith(first) or first.startswith(qfirst):
            return slug
    return None
