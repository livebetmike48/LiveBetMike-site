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

_cache = {"ts": 0, "data": None}


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
    try:
        resp = requests.get(URL, params={"type": "year", "year": 2026, "batSide": "",
                                          "stat": "index_hits", "condition": "All",
                                          "rolling": "3", "csv": "true"}, timeout=20)
        if resp.status_code == 200 and "," in resp.text[:2000]:
            data = _parse_csv(resp.text)
        if data:
            log.info("park factors loaded: %d venues (e.g. %s)",
                     len(data), list(data.items())[:2])
        else:
            log.warning("park factors unavailable -- running park-neutral")
    except Exception as e:
        log.warning("park factor fetch failed (%s) -- running park-neutral", e)
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
    try:
        resp = requests.get(URL, params={"type": "year", "year": 2026, "batSide": "",
                                          "stat": "index_strikeout", "condition": "All",
                                          "rolling": "3", "csv": "true"}, timeout=20)
        if resp.status_code == 200 and "," in resp.text[:2000]:
            data = _parse_k_csv(resp.text)
        if data:
            log.info("park K factors loaded: %d venues (e.g. %s)",
                     len(data), list(data.items())[:2])
        else:
            log.warning("park K factors unavailable -- running park-neutral")
    except Exception as e:
        log.warning("park K factor fetch failed (%s) -- running park-neutral", e)
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
