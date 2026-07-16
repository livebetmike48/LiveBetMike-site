"""
Bullpen usage: pitches thrown per arm per day over the last week, from
MLB boxscores (real recorded pitch counts). Starter appearances are
excluded so the grid shows the PEN's workload -- who's gassed, who's
fresh. Quality composites (their 'xFIP docked' score) deliberately not
replicated: usage is fact, the composite is invention.
"""
import time
import logging
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger("bullpen")

MLB_BASE = "https://statsapi.mlb.com/api/v1"

_teams_cache = {"ts": 0, "data": None}
_usage_cache: dict = {}
CACHE_SECONDS = 1800


def all_teams() -> list[dict]:
    """All 30 MLB teams for the picker."""
    now = time.time()
    if _teams_cache["data"] and now - _teams_cache["ts"] < 86400:
        return _teams_cache["data"]
    resp = requests.get(f"{MLB_BASE}/teams", params={"sportId": 1}, timeout=15)
    resp.raise_for_status()
    teams = [
        {"id": t["id"], "abbrev": t.get("abbreviation", "?"), "name": t.get("name", "?")}
        for t in resp.json().get("teams", [])
    ]
    teams.sort(key=lambda t: t["abbrev"])
    _teams_cache.update({"ts": now, "data": teams})
    return teams


def _usage_from_boxscores(boxscores: list[tuple[str, dict]], team_id: int) -> dict:
    """Pure parser (testable): [(date, boxscore_json)] -> usage grid."""
    arms: dict = {}
    dates = []
    for date, box in boxscores:
        if date not in dates:
            dates.append(date)
        for side in ("home", "away"):
            side_data = (box.get("teams") or {}).get(side) or {}
            if ((side_data.get("team") or {}).get("id")) != team_id:
                continue
            for player in (side_data.get("players") or {}).values():
                pitching = ((player.get("stats") or {}).get("pitching")) or {}
                pitches = pitching.get("numberOfPitches") or pitching.get("pitchesThrown")
                if not pitches:
                    continue
                if pitching.get("gamesStarted"):
                    continue  # starter's workload isn't bullpen usage
                name = (player.get("person") or {}).get("fullName", "?")
                arm = arms.setdefault(name, {"name": name, "counts": {}, "total": 0})
                arm["counts"][date] = arm["counts"].get(date, 0) + pitches
                arm["total"] += pitches
    dates.sort(reverse=True)
    arm_list = sorted(arms.values(), key=lambda a: -a["total"])
    return {"dates": dates, "arms": arm_list}


def get_usage(team_id: int, days: int = 7) -> dict:
    now = time.time()
    cached = _usage_cache.get(team_id)
    if cached and now - cached[0] < CACHE_SECONDS:
        return cached[1]

    end = datetime.now(timezone.utc) - timedelta(hours=4)
    start = end - timedelta(days=days)
    resp = requests.get(
        f"{MLB_BASE}/schedule",
        params={"sportId": 1, "teamId": team_id,
                "startDate": start.strftime("%Y-%m-%d"), "endDate": end.strftime("%Y-%m-%d")},
        timeout=15,
    )
    resp.raise_for_status()

    boxscores = []
    for d in resp.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            try:
                box = requests.get(f"{MLB_BASE}/game/{g['gamePk']}/boxscore", timeout=15).json()
                boxscores.append((g.get("officialDate", ""), box))
            except Exception as e:
                log.warning("boxscore fetch failed for %s: %s", g.get("gamePk"), e)

    result = _usage_from_boxscores(boxscores, team_id)
    _usage_cache[team_id] = (now, result)
    return result
