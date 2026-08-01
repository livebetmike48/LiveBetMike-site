"""
Park factors from Baseball Savant's official Statcast park-factor
leaderboard (MLB's own numbers -- on-brand, no licensing questions).
Hits factor per venue, 100 = neutral, fetched daily. If the fetch or
parse fails, every park returns neutral 1.0 and we log it loudly --
a wrong park factor is worse than none.
"""
import time
import logging
import unicodedata

import re as _re

import requests

log = logging.getLogger("parks")

URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"

# Savant serves fine to plain requests elsewhere in this codebase, but the
# leaderboard CSV routes have been picky before -- send a real UA so a
# bot-filter can be ruled out as a cause rather than suspected forever.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LiveBetMike/1.0)"}

# Param shapes tried in order. The first is what this file has always
# sent; the rest are narrower fallbacks. Whichever PARSES wins -- we never
# guess which is right, we try and check.
PARAM_SHAPES = [
    {"type": "year", "year": "{year}", "batSide": "", "stat": "{stat}",
     "condition": "All", "rolling": "3", "csv": "true"},
    {"type": "year", "year": "{year}", "batSide": "", "stat": "{stat}",
     "condition": "All", "rolling": "", "csv": "true"},
    {"type": "year", "year": "{year}", "stat": "{stat}", "csv": "true"},
]

_cache = {"ts": 0, "data": None}
# Last fetch detail, for /api/parks -- so a failure is diagnosable from
# the site instead of guessed at from a one-line warning.
_last = {}


def _fetch_raw(stat: str, year: int = 2026) -> tuple:
    """Try each param shape until one returns something CSV-shaped.
    Returns (text, detail) where detail records exactly what happened."""
    attempts = []
    for i, shape in enumerate(PARAM_SHAPES):
        params = {k: v.format(year=year, stat=stat) if isinstance(v, str) else v
                  for k, v in shape.items()}
        try:
            resp = requests.get(URL, params=params, headers=HEADERS, timeout=20)
            head = (resp.text or "")[:200].replace("\n", " ")
            # "usable" = a 200 with a body; the page itself carries the
            # data as embedded JSON now, so HTML is no longer a failure.
            looks_csv = resp.status_code == 200 and len(resp.text or "") > 500
            attempts.append({"shape": i, "status": resp.status_code,
                             "content_type": resp.headers.get("content-type", ""),
                             "bytes": len(resp.text or ""), "looks_csv": looks_csv,
                             "head": head})
            if looks_csv:
                return resp.text, {"ok": True, "shape_used": i, "attempts": attempts}
        except Exception as e:
            attempts.append({"shape": i, "error": str(e)})
    return "", {"ok": False, "attempts": attempts}


def _json_blocks(html: str) -> dict:
    """Savant renders its leaderboards from JSON embedded in the page --
    `var something = [ {...}, {...} ];`. The CSV download parameter no
    longer returns CSV (Aug 1: every shape came back as 122KB of HTML),
    so we read the same data the page itself uses.

    Scans for `var NAME = [` and walks the brackets respecting strings, so
    a comma or bracket inside a value can't end the block early. Returns
    {name: parsed_list} for every block that parses as a list of dicts."""
    import json as _json
    out = {}
    for m in _re.finditer(r"var\s+([A-Za-z_$][\w$]*)\s*=\s*\[", html):
        name = m.group(1)
        i = html.index("[", m.end() - 1)
        depth, in_str, esc, j = 0, False, False, i
        while j < len(html):
            c = html[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            continue
        try:
            parsed = _json.loads(html[i:j + 1])
        except Exception:
            continue
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            out[name] = parsed
    return out


def _pick_cols(row: dict, stat_keys: tuple) -> tuple:
    """(venue_key, stat_key) for a parsed row, or (None, None)."""
    keys = list(row.keys())
    lower = {k.lower(): k for k in keys}
    venue = next((lower[k] for k in lower
                  if "venue" in k or k in ("name_display_club", "team_name")
                  or ("name" in k and "player" not in k)), None)
    stat = None
    for want in stat_keys:
        if want in lower:
            stat = lower[want]
            break
    if not stat:
        stat = next((lower[k] for k in lower
                     if any(w.split("_")[-1] in k for w in stat_keys) and "index" in k), None)
    return venue, stat


def _from_html(html: str, stat_keys: tuple) -> tuple:
    """Park factors out of the embedded JSON. Returns (data, detail)."""
    blocks = _json_blocks(html)
    detail = {"blocks_found": {n: {"rows": len(v), "keys": list(v[0].keys())[:14]}
                               for n, v in blocks.items()}}
    for name, rows in blocks.items():
        venue, stat = _pick_cols(rows[0], stat_keys)
        if not venue or not stat:
            continue
        data = {}
        for r in rows:
            try:
                data[_fold(str(r[venue]))] = float(r[stat]) / 100.0
            except (TypeError, ValueError, KeyError):
                continue
        if len(data) >= 20:   # a real park list is ~30 venues
            detail.update({"block_used": name, "venue_key": venue, "stat_key": stat})
            return data, detail
    return {}, detail


def _fold(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "")
    return "".join(c for c in t if not unicodedata.combining(c)).lower().strip()


def _parse_csv(text: str) -> dict:
    import csv, io
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return {}
    cols = {c.lower(): c for c in reader.fieldnames}
    venue_col = next((cols[k] for k in cols if "venue" in k or "name" in k), None)
    hits_col = next((cols[k] for k in cols
                     if k in ("index_hits", "hits", "h") or ("hit" in k and "index" in k)), None)
    if not venue_col or not hits_col:
        return {}   # not CSV -- the HTML/JSON path handles it
    out = {}
    for row in reader:
        try:
            out[_fold(row[venue_col])] = float(row[hits_col]) / 100.0
        except (TypeError, ValueError, KeyError):
            continue
    return out


def factors() -> dict:
    now = time.time()
    if _cache["data"] is not None and now - _cache["ts"] < 86400:
        return _cache["data"]
    data = {}
    text, detail = _fetch_raw("index_hits")
    _last["hits"] = detail
    if text:
        data = _parse_csv(text)          # real CSV, if Savant ever serves it again
        if not data:
            data, hdet = _from_html(text, ("index_hits", "hits"))
            detail.update(hdet)
        detail["parsed_venues"] = len(data)
    if data:
        log.info("park factors loaded: %d venues (e.g. %s)",
                 len(data), list(data.items())[:2])
    else:
        # never silent: say WHY, with the status and first bytes, so this
        # is fixable from a log line instead of a guessing game
        log.warning("park factors unavailable -- running park-neutral. attempts=%s",
                    detail.get("attempts"))
    _cache.update({"ts": now, "data": data})
    return data


def factor_for(venue_name: str) -> float | None:
    """Hits factor for a venue (1.0 neutral). None when unknown so the
    model can distinguish 'neutral park' from 'no data'."""
    if not venue_name:
        return None
    data = factors()
    if not data:
        return None
    key = _fold(venue_name)
    if key in data:
        return data[key]
    for k, v in data.items():
        if key in k or k in key:
            return v
    return None


# ---------- strikeout park factors (additive; hits machinery untouched) ----------

_k_cache = {"ts": 0, "data": None}


def k_factors() -> dict:
    """Savant strikeout park factors, same endpoint/honesty rules as hits:
    neutral-on-failure, logged loudly."""
    now = time.time()
    if _k_cache["data"] is not None and now - _k_cache["ts"] < 86400:
        return _k_cache["data"]
    data = {}
    text, detail = _fetch_raw("index_strikeout")
    _last["strikeout"] = detail
    if text:
        data = _parse_k_csv(text)
        if not data:
            data, hdet = _from_html(text, ("index_strikeout", "index_so", "strikeout"))
            detail.update(hdet)
        detail["parsed_venues"] = len(data)
    if data:
        log.info("park K factors loaded: %d venues (e.g. %s)",
                 len(data), list(data.items())[:2])
    else:
        log.warning("park K factors unavailable -- running park-neutral. attempts=%s",
                    detail.get("attempts"))
    _k_cache.update({"ts": now, "data": data})
    return data


def _parse_k_csv(text: str) -> dict:
    import csv, io
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return {}
    cols = {c.lower(): c for c in reader.fieldnames}
    venue_col = next((cols[k] for k in cols if "venue" in k or "name" in k), None)
    k_col = next((cols[k] for k in cols
                  if k in ("index_strikeout", "index_so", "strikeout", "so")
                  or (("strikeout" in k or k.endswith("_so")) and "index" in k)), None)
    if not venue_col or not k_col:
        return {}   # not CSV -- the HTML/JSON path handles it
    out = {}
    for row in reader:
        try:
            out[_fold(row[venue_col])] = float(row[k_col]) / 100.0
        except (TypeError, ValueError, KeyError):
            continue
    return out


def k_factor_for(venue_name: str) -> float | None:
    """Strikeout factor for a venue (1.0 neutral). None when unknown."""
    if not venue_name:
        return None
    data = k_factors()
    if not data:
        return None
    key = _fold(venue_name)
    if key in data:
        return data[key]
    for k, v in data.items():
        if key in k or k in key:
            return v
    return None


def diagnose() -> dict:
    """What the last park-factor fetch actually did -- status codes,
    content types, first bytes of the response, and how many venues
    parsed. Read-only; forces a fresh fetch so it reflects right now."""
    _cache.update({"ts": 0, "data": None})
    _k_cache.update({"ts": 0, "data": None})
    h = factors()
    k = k_factors()
    return {
        "hits": {"venues": len(h), "sample": list(h.items())[:3],
                 "fetch": _last.get("hits")},
        "strikeout": {"venues": len(k), "sample": list(k.items())[:3],
                      "fetch": _last.get("strikeout")},
        "note": ("Both empty means the Savant CSV isn't coming back in the "
                 "shape this file expects. 'head' shows the first 200 bytes "
                 "of what it DID return -- that identifies the problem."),
    }
