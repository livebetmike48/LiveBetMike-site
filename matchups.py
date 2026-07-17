"""
Builds the day's hitter-vs-starter matchup board from the validated engine.
Heavy (one season fetch per hitter), so it's computed once per day in a
background thread and cached; the API serves instantly after warmup.
"""
import os
import json
import sqlite3
import threading
import logging
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import requests

import statcast_api
import parlay
import odds_api
import odds_log

log = logging.getLogger("matchups")

_cache = {"date": None, "status": "cold", "data": None}
_progress = {"done": 0, "total": 0}
_lock = threading.Lock()
DB_PATH = os.getenv("DB_PATH", "odds_history.db")


def _db_save_board(data: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("CREATE TABLE IF NOT EXISTS daily_board (date TEXT PRIMARY KEY, payload TEXT)")
        conn.execute("INSERT OR REPLACE INTO daily_board VALUES (?, ?)",
                     (data["date"], json.dumps(data)))
        conn.commit(); conn.close()
    except Exception as e:
        log.warning("board persist failed: %s", e)


def _db_load_board(date: str) -> dict | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("CREATE TABLE IF NOT EXISTS daily_board (date TEXT PRIMARY KEY, payload TEXT)")
        row = conn.execute("SELECT payload FROM daily_board WHERE date=?", (date,)).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception as e:
        log.warning("board load failed: %s", e)
        return None
_lineup_cache = {"ts": 0, "date": None, "data": {}}

HITTERS_PER_TEAM = 6
MLB_BASE = "https://statsapi.mlb.com/api/v1"


def _rows_last_days(rows: list[dict], days: int) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=4) - timedelta(days=days)).strftime("%Y-%m-%d")
    return [r for r in rows if (r.get("game_date") or "") >= cutoff]


def _window_stats(rows: list[dict], hand: str) -> dict | None:
    vs = statcast_api.vs_handedness_stats(rows, "p_throws", hand)
    if not vs:
        return None
    rate, games = parlay.hit_game_rate(rows)
    return {
        "pa": vs["pa"], "avg": vs.get("avg"), "xba": vs.get("xba"),
        "xwoba": vs.get("xwoba"), "k_pct": vs.get("k_pct"), "bb_pct": vs.get("bb_pct"),
        "whiff_pct": vs.get("whiff_pct"), "hit_game_pct": round(rate * 100, 1), "games": games,
    }


def _orders_from_boxscore(box: dict) -> dict:
    """Pure parser: boxscore -> {player_id: batting order 1-9} once the
    lineup is posted (empty dict before that)."""
    orders = {}
    for side in ("home", "away"):
        side_data = (box.get("teams") or {}).get(side) or {}
        for i, pid in enumerate(side_data.get("battingOrder") or [], start=1):
            orders[pid] = i
    return orders


def _get_lineups(game_pks: list[int]) -> dict:
    """{game_pk: {player_id: order}} -- cached 10 min, refreshed all day so
    orders appear as teams post lineups."""
    today = parlay.et_date_str(0)
    now = time.time()
    if (_lineup_cache["date"] == today and now - _lineup_cache["ts"] < 600):
        return _lineup_cache["data"]
    data = {}
    for pk in game_pks:
        try:
            box = requests.get(f"{MLB_BASE}/game/{pk}/boxscore", timeout=15).json()
            data[pk] = _orders_from_boxscore(box)
        except Exception:
            data[pk] = {}
    _lineup_cache.update({"ts": now, "date": today, "data": data})
    return data


def last10_strip(rows: list[dict]) -> list[bool]:
    """Ten most recent games as hit/no-hit booleans -- the row's signature
    visual, straight from real game logs."""
    games: dict = {}
    for r in rows:
        gpk, date = r.get("game_pk"), r.get("game_date")
        if gpk is None or not date:
            continue
        key = (date, gpk)
        games.setdefault(key, False)
        if r.get("events") in parlay.HIT_EVENTS:
            games[key] = True
    ordered = [games[k] for k in sorted(games.keys())]
    return ordered[-10:]


def _build_matchups() -> dict:
    slate = parlay.get_today_slate()

    # Odds: display today's moneylines AND archive them for future
    # backtesting (our own history, accumulating daily)
    odds_events = []
    try:
        odds_events = odds_api.get_mlb_odds("h2h")
        if odds_events:
            odds_log.init_db()
            odds_log.log_snapshot(parlay.et_date_str(0), odds_events, "h2h")
    except Exception as e:
        log.warning("Odds fetch/log skipped: %s", e)

    games_out = []
    for g in slate:
        game_entry = {
            "game_pk": g["game_pk"],
            "away": g["teams"]["away"]["abbrev"],
            "home": g["teams"]["home"]["abbrev"],
            "hitters": [],
        }
        ev = odds_api.find_event(
            odds_events, g["teams"]["home"]["name"], g["teams"]["away"]["name"]
        ) if odds_events else None
        if ev:
            ml = {}
            for side in ("home", "away"):
                prices = odds_api.all_prices(ev, "h2h", g["teams"][side]["name"])
                bp = odds_api.best_price(prices)
                if bp:
                    ml[side] = {"price": bp[1], "book": bp[0]}
            if ml:
                game_entry["moneyline"] = ml
        for side, opp_side in (("home", "away"), ("away", "home")):
            team = g["teams"][side]
            opp = g["teams"][opp_side]
            if not opp["starter_id"]:
                continue
            try:
                hand = parlay.get_starter_hand(opp["starter_id"])
            except Exception:
                continue
            if hand not in ("L", "R"):
                continue
            batters = parlay.shortlist_hitters([team["abbrev"]], "xba", HITTERS_PER_TEAM)
            _progress["total"] += len(batters)

            def _eval(batter):
                try:
                    rows = parlay.get_player_season_rows(batter["player_id"], False)
                except Exception:
                    return None
                finally:
                    _progress["done"] += 1
                if not rows:
                    return None
                season = _window_stats(rows, hand)
                if not season or season["pa"] < 40:
                    return None
                l30 = _window_stats(_rows_last_days(rows, 30), hand)
                return {
                    "player_id": batter["player_id"],
                    "name": batter["name"],
                    "team": team["abbrev"],
                    "starter": opp["starter_name"],
                    "starter_id": opp["starter_id"],
                    "hand": hand,
                    "windows": {"season": season, "l30": l30},
                    "streak": parlay.hitting_streak(rows),
                    "last10": last10_strip(rows),
                }

            with ThreadPoolExecutor(max_workers=6) as pool:
                for result in pool.map(_eval, batters):
                    if result:
                        game_entry["hitters"].append(result)
        game_entry["hitters"].sort(key=lambda h: -((h["windows"]["season"] or {}).get("xba") or 0))
        games_out.append(game_entry)
    return {"date": parlay.et_date_str(0), "games": games_out}


def get_matchups() -> dict:
    """Serve from cache; kick off a rebuild in the background when the
    day rolls over. Never blocks the request."""
    today = parlay.et_date_str(0)
    with _lock:
        if _cache["status"] == "cold" or _cache["date"] != today:
            persisted = _db_load_board(today)
            if persisted:
                _cache.update({"date": today, "status": "ready", "data": persisted})
                log.info("Matchup board restored from DB for %s", today)
        if _cache["date"] == today and _cache["status"] == "ready":
            data = _cache["data"]
            try:
                lineups = _get_lineups([g["game_pk"] for g in data["games"]])
                posted = 0
                for g in data["games"]:
                    orders = lineups.get(g["game_pk"], {})
                    g["lineup_posted"] = bool(orders)
                    posted += 1 if orders else 0
                    for h in g["hitters"]:
                        h["order"] = orders.get(h["player_id"])
                data["lineups_posted"] = posted
            except Exception as e:
                log.warning("lineup enrich skipped: %s", e)
            return {"status": "ready", **data}
        if _cache["status"] == "warming":
            return {"status": "warming",
                    "progress": f"{_progress['done']}/{_progress['total']} hitters" if _progress["total"] else "starting"}
        _cache["status"] = "warming"
        _cache["date"] = today
        _progress.update({"done": 0, "total": 0})

    def _warm():
        try:
            data = _build_matchups()
            with _lock:
                _cache["data"] = data
                _cache["status"] = "ready"
            _db_save_board(data)
            log.info("Matchup board ready: %d games (%d hitters evaluated)",
                     len(data["games"]), _progress["done"])
        except Exception as e:
            log.error("Matchup build failed: %s", e)
            with _lock:
                _cache["status"] = "cold"

    threading.Thread(target=_warm, daemon=True).start()
    return {"status": "warming"}


def get_detail(batter_id: int, starter_id: int, hand: str) -> dict:
    """The drill-down: batter vs each pitch (vs this hand) merged with the
    starter's mix to the batter's side -- same shape as /matchup."""
    batter_rows = parlay.get_player_season_rows(batter_id, False)
    starter_rows = parlay.get_player_season_rows(starter_id, True)

    batter_vs_hand = [r for r in batter_rows if r.get("p_throws") == hand]
    per_pitch = statcast_api.vs_each_pitch(batter_vs_hand, min_pitches=10)

    # Batter's side vs this starter (assume R unless rows say otherwise)
    sides = [r.get("stand") for r in batter_rows if r.get("stand")]
    batter_side = max(set(sides), key=sides.count) if sides else "R"
    mix = statcast_api.pitch_mix_breakdown(
        [r for r in starter_rows if r.get("stand") == batter_side]
    )
    return {
        "per_pitch": per_pitch,
        "starter_mix": mix,
        "batter_side": batter_side,
        "batter_line": statcast_api.vs_handedness_stats(batter_rows, "p_throws", hand),
        "starter_line": statcast_api.vs_handedness_stats(starter_rows, "stand", batter_side),
    }
