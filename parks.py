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


def _fetch_raw(stat: str, year: int | None = None) -> tuple:
    year = int(year) if year else int(time.strftime("%Y"))
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


def _pick_cols(row: dict, stat_names: tuple) -> tuple:
    """(venue_key, stat_key) for a parsed row, or (None, None).

    stat_names are the BARE stat names to look for (e.g. ("hits","hit","h")).
    Savant's column is some index_* variant and the exact spelling has
    moved before, so match on the suffix after the index_ prefix rather
    than hardcoding a full column name."""
    lower = {k.lower(): k for k in row.keys()}
    # Exact names in priority order FIRST. Substring matching alone picks
    # up "grouping_venue_conditions" -- whose value is "All" on every row,
    # collapsing all 29 parks into one entry (the Aug 1 failure).
    venue = None
    for cand in ("venue_name", "venue", "name_display_club", "team_name"):
        if cand in lower:
            venue = lower[cand]
            break
    if not venue:
        venue = next((lower[k] for k in lower
                      if k.endswith("_name") and not k.startswith("grouping")), None)
    # exact index_<name> first, then a bare <name> column
    for want in stat_names:
        for cand in ("index_" + want, want):
            if cand in lower:
                return venue, lower[cand]
    # last resort: any index_* whose suffix starts with one of the names
    for k in lower:
        if k.startswith("index_"):
            suffix = k[len("index_"):]
            if any(suffix == w or suffix.startswith(w) for w in stat_names):
                return venue, lower[k]
    return venue, None


def _from_html(html: str, stat_keys: tuple) -> tuple:
    """Park factors out of the embedded JSON. Returns (data, detail)."""
    blocks = _json_blocks(html)
    # Report EVERY key, and the index_* columns separately -- a truncated
    # key list is how the first pass missed the real column name.
    detail = {"blocks_found": {
        n: {"rows": len(v), "keys": list(v[0].keys()),
            "index_cols": [k for k in v[0].keys() if k.lower().startswith("index_")]}
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
            data, hdet = _from_html(text, ("hits", "hit", "h"))
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


_hits_year_cache = {}   # season -> {"ts","data"} for historical backtests


def hits_factors_year(year: int) -> dict:
    """Savant hits park factors for a PAST season -- same endpoint and
    honesty rules as factors(): neutral-on-failure, logged loudly, cached
    per year (historical factors are final)."""
    year = int(year)
    now = time.time()
    hit = _hits_year_cache.get(year)
    if hit and now - hit["ts"] < 86400 * 30:
        return hit["data"]
    data = {}
    text, _detail = _fetch_raw("index_hits", year=year)
    if text:
        data = _parse_csv(text)
        if not data:
            data, _h = _from_html(text, ("hits", "hit", "h"))
    if data:
        log.info("park hits factors %s: %d venues", year, len(data))
    else:
        log.warning("park hits factors %s unavailable -- park-neutral", year)
    _hits_year_cache[year] = {"ts": now, "data": data}
    return data


def factor_for(venue_name: str, year: int | None = None) -> float | None:
    """Hits factor for a venue (1.0 neutral). None when unknown so the
    model can distinguish 'neutral park' from 'no data'. year=None = the
    current season (live path untouched); a past year uses that season's
    own factors."""
    if not venue_name:
        return None
    data = hits_factors_year(year) if year and int(year) != int(time.strftime("%Y")) \
        else factors()
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


_k_year_cache = {}  # season -> {"ts","data"} for historical backtests


def k_factors(year: int | None = None) -> dict:
    """Savant strikeout park factors, same endpoint/honesty rules as hits:
    neutral-on-failure, logged loudly. year=None = current season (live
    behavior, its own cache slot untouched); a past year gets that season's
    own factors, cached per year."""
    now = time.time()
    if year:
        year = int(year)
        hit = _k_year_cache.get(year)
        if hit and now - hit["ts"] < 86400 * 30:   # historical factors are final
            return hit["data"]
        text, _detail = _fetch_raw("index_strikeout", year=year)
        data = {}
        if text:
            data = _parse_k_csv(text)
            if not data:
                data, _h = _from_html(text, ("strikeout", "so", "k"))
        if data:
            log.info("park K factors %s: %d venues", year, len(data))
        else:
            log.warning("park K factors %s unavailable -- park-neutral", year)
        _k_year_cache[year] = {"ts": now, "data": data}
        return data
    if _k_cache["data"] is not None and now - _k_cache["ts"] < 86400:
        return _k_cache["data"]
    data = {}
    text, detail = _fetch_raw("index_strikeout")
    _last["strikeout"] = detail
    if text:
        data = _parse_k_csv(text)
        if not data:
            data, hdet = _from_html(text, ("strikeout", "so", "k"))
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


def k_factor_for(venue_name: str, year: int | None = None) -> float | None:
    """Strikeout factor for a venue (1.0 neutral). None when unknown.
    year routes to that season's own factors (backtests); None = live."""
    if not venue_name:
        return None
    data = k_factors(year)
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
