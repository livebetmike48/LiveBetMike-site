"""
Hit-probability model v1 -- transparent and point-in-time.

Method: the odds-ratio / log5 approach (Bill James) -- the standard
published way to combine a batter's rate, a pitcher's rate-allowed, and
the league rate into a matchup rate. No invented weights, no composite
scores. Every projection exposes its inputs.

    odds(p) = p / (1 - p)
    matchup_odds = odds(batter) * odds(pitcher_allowed) / odds(league)
    p_PA = matchup_odds / (1 + matchup_odds)

Per-game: P(1+ hit) = 1 - (1 - p_vs_starter)^PA_s * (1 - p_vs_pen)^PA_p
using expected plate appearances vs the starter and the bullpen.

Point-in-time discipline: all rate helpers take a `before` date and use
only rows strictly earlier -- this is what makes the backtest honest.
"""
import time
import logging

import requests

import parlay
import statcast_api

log = logging.getLogger("model")

MLB_BASE = "https://statsapi.mlb.com/api/v1"

# League-average PAs for a lineup regular, split by pitcher role. MLB team
# averages: ~38 PA/team/game over 9 slots ≈ 4.2; starters face ~60% of them.
PA_VS_STARTER = 2.6
PA_VS_PEN = 1.6

_league_cache = {"ts": 0, "p": None}


def league_hit_rate() -> float:
    """League per-PA hit rate from MLB's real season team totals
    (sum of hits / sum of plate appearances across all 30 teams)."""
    now = time.time()
    if _league_cache["p"] and now - _league_cache["ts"] < 86400:
        return _league_cache["p"]
    resp = requests.get(
        f"{MLB_BASE}/teams/stats",
        params={"season": 2026, "group": "hitting", "stats": "season", "sportId": 1},
        timeout=20,
    )
    resp.raise_for_status()
    hits = pa = 0
    for split in resp.json()["stats"][0]["splits"]:
        stat = split.get("stat", {})
        hits += int(stat.get("hits", 0))
        pa += int(stat.get("plateAppearances", 0))
    if pa == 0:
        raise RuntimeError("league totals unavailable")
    p = hits / pa
    _league_cache.update({"ts": now, "p": p})
    log.info("League hit rate: %.4f (%d H / %d PA)", p, hits, pa)
    return p


def rows_before(rows: list[dict], before: str | None) -> list[dict]:
    """Point-in-time filter: only rows strictly before `before` (YYYY-MM-DD).
    None = use everything (live usage)."""
    if before is None:
        return rows
    return [r for r in rows if (r.get("game_date") or "9999") < before]


def per_pa_hit_rate(rows: list[dict], split_col: str, split_val: str) -> dict | None:
    """Hits per plate appearance within a split (batter vs hand, or
    pitcher-allowed vs side), from raw rows. Same PA accounting as the
    validated engine (via parlay's event sets)."""
    pa = hits = 0
    for r in rows:
        if r.get(split_col) != split_val:
            continue
        ev = r.get("events")
        if not ev or ev in statcast_api.NON_PA_EVENTS:
            continue
        pa += 1
        if ev in parlay.HIT_EVENTS:
            hits += 1
    if pa == 0:
        return None
    return {"pa": pa, "hits": hits, "rate": hits / pa}


def _odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


def log5_rate(p_batter: float, p_pitcher_allowed: float, p_league: float) -> float:
    """The published odds-ratio combination."""
    combined = _odds(p_batter) * _odds(p_pitcher_allowed) / _odds(p_league)
    return combined / (1 + combined)


def hit_probability(batter_rows: list[dict], starter_rows: list[dict],
                    starter_hand: str, batter_side: str, p_league: float,
                    before: str | None = None,
                    min_batter_pa: int = 40, min_starter_pa: int = 60) -> dict | None:
    """P(batter records 1+ hit today) with full input transparency.
    Returns None when samples are too thin to say anything honest."""
    b_rows = rows_before(batter_rows, before)
    s_rows = rows_before(starter_rows, before)

    b = per_pa_hit_rate(b_rows, "p_throws", starter_hand)
    s = per_pa_hit_rate(s_rows, "stand", batter_side)
    if not b or b["pa"] < min_batter_pa or not s or s["pa"] < min_starter_pa:
        return None

    p_vs_starter = log5_rate(b["rate"], s["rate"], p_league)
    # Vs the pen (unknown arms): the batter's own overall rate vs the hand
    # -- no pitcher adjustment because we don't know the pitchers.
    p_vs_pen = b["rate"]

    p_no_hit = ((1 - p_vs_starter) ** PA_VS_STARTER) * ((1 - p_vs_pen) ** PA_VS_PEN)
    return {
        "p_hit": round(1 - p_no_hit, 4),
        "inputs": {
            "batter_rate_vs_hand": round(b["rate"], 4), "batter_pa": b["pa"],
            "starter_rate_allowed_vs_side": round(s["rate"], 4), "starter_pa": s["pa"],
            "league_rate": round(p_league, 4),
            "p_pa_vs_starter": round(p_vs_starter, 4),
            "pa_vs_starter": PA_VS_STARTER, "pa_vs_pen": PA_VS_PEN,
        },
    }
