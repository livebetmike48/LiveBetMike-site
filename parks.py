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
            looks_csv = resp.status_code == 200 and "," in (resp.text or "")[:2000]
            attempts.append({"shape": i, "status": resp.status_code,
                             "content_type": resp.headers.get("content-type", ""),
                             "bytes": len(resp.text or ""), "looks_csv": looks_csv,
                             "head": head})
            if looks_csv:
                return resp.text, {"ok": True, "shape_used": i, "attempts": attempts}
        except Exception as e:
            attempts.append({"shape": i, "error": str(e)})
    return "", {"ok": False, "attempts": attempts}


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
        log.warning("park CSV columns unrecognized: %s", reader.fieldnames[:10])
        return {}
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
        data = _parse_csv(text)
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
        log.warning("park K CSV columns unrecognized: %s", reader.fieldnames[:10])
        return {}
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
