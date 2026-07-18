"""
Pitcher Projections -- interactive rate x opportunity tool.

For each of today's starters: real per-TBF rates (K%, H/TBF, BB/TBF)
from his season rows, Season and L30 windows, plus his REAL average
batters-faced-per-start prefilled from his own game logs (not a blank
for the user to guess). The frontend recomputes K/H/BB/Outs live as the
TBF slider moves: projection = rate x TBF. Outs ~= TBF - H - BB.

This is also the strikeouts-model skeleton: per-TBF rates x a TBF
distribution is exactly how the K prop engine will price lines.
"""
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import parlay
import statcast_api

log = logging.getLogger("pitchers")

K_EVENTS = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk", "intent_walk"}

_cache = {"date": None, "status": "cold", "data": None}
_progress = {"done": 0, "total": 0}
_lock = threading.Lock()


def _rates(rows: list[dict]) -> dict | None:
    """Per-TBF rates from raw rows (PA accounting = the validated engine)."""
    tbf = k = h = bb = 0
    for r in rows:
        ev = r.get("events")
        if not ev or ev in statcast_api.NON_PA_EVENTS:
            continue
        tbf += 1
        if ev in K_EVENTS:
            k += 1
        elif ev in parlay.HIT_EVENTS:
            h += 1
        elif ev in BB_EVENTS:
            bb += 1
    if tbf < 30:
        return None
    return {"tbf": tbf, "k_rate": round(k / tbf, 4), "h_rate": round(h / tbf, 4),
            "bb_rate": round(bb / tbf, 4)}


def _pitch_counts(rows: list[dict]) -> tuple[float, float]:
    """(avg pitches per start, pitches per batter faced) -- every row is
    one pitch, so this is direct counting on the same games/PA accounting."""
    games = set()
    pitches = 0
    tbf = 0
    for r in rows:
        gpk, date = r.get("game_pk"), r.get("game_date")
        if gpk is None or not date:
            continue
        games.add((date, gpk))
        pitches += 1
        ev = r.get("events")
        if ev and ev not in statcast_api.NON_PA_EVENTS:
            tbf += 1
    if not games or not tbf:
        return 0.0, 0.0
    return pitches / len(games), pitches / tbf


def _tbf_per_start(rows: list[dict]) -> tuple[float, int]:
    games: dict = {}
    for r in rows:
        gpk, date = r.get("game_pk"), r.get("game_date")
        if gpk is None or not date:
            continue
        ev = r.get("events")
        if ev and ev not in statcast_api.NON_PA_EVENTS:
            games[(date, gpk)] = games.get((date, gpk), 0) + 1
    if not games:
        return 0.0, 0
    return sum(games.values()) / len(games), len(games)


def _rows_last_days(rows: list[dict], days: int) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=4) - timedelta(days=days)).strftime("%Y-%m-%d")
    return [r for r in rows if (r.get("game_date") or "") >= cutoff]


def _build() -> dict:
    slate = parlay.get_today_slate()
    tasks = []
    for g in slate:
        for side, opp in (("home", "away"), ("away", "home")):
            team = g["teams"][side]
            if team.get("starter_id"):
                tasks.append({
                    "starter_id": team["starter_id"],
                    "starter_name": team["starter_name"],
                    "team": team["abbrev"],
                    "opp": g["teams"][opp]["abbrev"],
                })
    _progress.update({"done": 0, "total": len(tasks)})

    def _one(t):
        try:
            hand = parlay.get_starter_hand(t["starter_id"])
            rows = parlay.get_player_season_rows(t["starter_id"], True)
        except Exception:
            return None
        finally:
            _progress["done"] += 1
        if not rows:
            return None
        season = _rates(rows)
        l30 = _rates(_rows_last_days(rows, 30))
        tbf_avg, starts = _tbf_per_start(rows)
        pitches_avg, ppb = _pitch_counts(rows)
        if not season or not starts or not ppb:
            return None
        return {
            "name": t["starter_name"], "hand": hand or "?",
            "team": t["team"], "opp": t["opp"],
            "tbf_avg": round(tbf_avg, 1), "starts": starts,
            "pitches_avg": round(pitches_avg), "ppb": round(ppb, 3),
            "season": season, "l30": l30,
        }

    out = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(_one, tasks):
            if r:
                out.append(r)
    out.sort(key=lambda p: -(p["season"]["k_rate"]))
    return {"date": parlay.et_date_str(0), "pitchers": out}


def get_pitchers() -> dict:
    today = parlay.et_date_str(0)
    with _lock:
        if _cache["date"] == today and _cache["status"] == "ready":
            return {"status": "ready", **_cache["data"]}
        if _cache["status"] == "warming":
            return {"status": "warming",
                    "progress": f"{_progress['done']}/{_progress['total']} starters" if _progress["total"] else "starting"}
        _cache.update({"date": today, "status": "warming"})

    def _work():
        try:
            data = _build()
            with _lock:
                _cache.update({"data": data, "status": "ready"})
            log.info("Pitcher projections ready: %d starters", len(data["pitchers"]))
        except Exception as e:
            log.error("pitcher build failed: %s", e)
            with _lock:
                _cache["status"] = "cold"

    threading.Thread(target=_work, daemon=True).start()
    return {"status": "warming", "progress": "starting"}
