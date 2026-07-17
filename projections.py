"""
The model goes live -- carefully. Today's hit probabilities for the slate
from the frozen lab config, stored permanently, and AUTO-GRADED against
boxscores the next day. Every day of predictions becomes out-of-sample
evidence: a permanent result log nobody can cherry-pick.
"""
import time
import logging
import sqlite3
from contextlib import contextmanager

import requests

import parlay
import statcast_api
import model
import lab
import odds_api

log = logging.getLogger("projections")

MLB_BASE = "https://statsapi.mlb.com/api/v1"

_today_cache = {"date": None, "ts": 0, "data": None}
_build_state = {"running": False}
_grade_ts = {"ts": 0}
import threading as _threading


@contextmanager
def _conn():
    conn = sqlite3.connect(lab.DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions (
            date TEXT, game_pk INTEGER, player_id INTEGER, name TEXT,
            team TEXT, starter TEXT, p REAL, created INTEGER,
            graded INTEGER DEFAULT 0, hit INTEGER,
            PRIMARY KEY (date, game_pk, player_id))""")


def get_today() -> dict:
    """Instant answer: cached projections if fresh, else kick a BACKGROUND
    build and report status -- never blocks the web request (first build
    of a day takes minutes; that must not live inside an HTTP timeout)."""
    today = parlay.et_date_str(0)
    now = time.time()
    if _today_cache["date"] == today and _today_cache["data"] and now - _today_cache["ts"] < 1800:
        return {"status": "ready", **_today_cache["data"]}
    if _build_state["running"]:
        return {"status": "building"}
    _build_state["running"] = True

    def _work():
        try:
            build_today()
        except Exception as e:
            log.error("projections build failed: %s", e)
        finally:
            _build_state["running"] = False

    _threading.Thread(target=_work, daemon=True).start()
    return {"status": "building"}


def build_today() -> dict:
    """Predict every shortlisted hitter on today's slate with the frozen
    lab config. Cached 30 min; predictions stored permanently (first
    write wins per player/game/day -- no revisionism)."""
    today = parlay.et_date_str(0)
    now = time.time()
    if _today_cache["date"] == today and now - _today_cache["ts"] < 1800 and _today_cache["data"]:
        return _today_cache["data"]

    lab._apply_config()
    p_league = model.league_hit_rate()
    init_db()

    slate = parlay.get_today_slate()
    rows_out = []
    for g in slate:
        for side, opp_side in (("home", "away"), ("away", "home")):
            team = g["teams"][side]
            opp = g["teams"][opp_side]
            if not opp["starter_id"]:
                continue
            try:
                hand = parlay.get_starter_hand(opp["starter_id"])
                starter_rows = parlay.get_player_season_rows(opp["starter_id"], True)
            except Exception:
                continue
            if hand not in ("L", "R"):
                continue
            for batter in parlay.shortlist_hitters([team["abbrev"]], "xba", 9):
                try:
                    b_rows = parlay.get_player_season_rows(batter["player_id"], False)
                except Exception:
                    continue
                sides = [r.get("stand") for r in b_rows if r.get("stand")]
                if not sides:
                    continue
                b_side = max(set(sides), key=sides.count)
                pred = model.hit_probability(b_rows, starter_rows, hand, b_side,
                                              p_league, batter_name=batter["name"])
                if not pred:
                    continue
                rows_out.append({
                    "date": today, "game_pk": g["game_pk"],
                    "player_id": batter["player_id"], "name": batter["name"],
                    "team": team["abbrev"], "starter": opp["starter_name"],
                    "home_name": g["teams"]["home"]["name"],
                    "away_name": g["teams"]["away"]["name"],
                    "p": pred["p_hit"],
                })
    rows_out.sort(key=lambda r: -r["p"])
    with _conn() as c:
        for r in rows_out:
            c.execute("""INSERT OR IGNORE INTO predictions
                (date, game_pk, player_id, name, team, starter, p, created)
                VALUES (?,?,?,?,?,?,?,?)""",
                (r["date"], r["game_pk"], r["player_id"], r["name"],
                 r["team"], r["starter"], r["p"], int(now)))
    data = {"date": today, "projections": rows_out}
    _today_cache.update({"date": today, "ts": now, "data": data})
    return data


def grade_pending():
    """Grade past ungraded predictions vs real boxscores. Players with no
    recorded PA in the final box are left ungraded (didn't play -- there
    was no bet to win or lose)."""
    if time.time() - _grade_ts["ts"] < 3600:
        return
    _grade_ts["ts"] = time.time()
    init_db()
    today = parlay.et_date_str(0)
    with _conn() as c:
        pending = c.execute(
            "SELECT DISTINCT date, game_pk FROM predictions WHERE graded=0 AND date < ?",
            (today,)).fetchall()
    for date, game_pk in pending[:40]:
        try:
            box = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=15).json()
        except Exception as e:
            log.warning("grade fetch failed %s: %s", game_pk, e)
            continue
        hits_by_pid = {}
        for s in ("home", "away"):
            for player in ((box.get("teams") or {}).get(s, {}).get("players") or {}).values():
                pid = (player.get("person") or {}).get("id")
                batting = ((player.get("stats") or {}).get("batting")) or {}
                if pid and batting.get("plateAppearances"):
                    hits_by_pid[pid] = 1 if batting.get("hits", 0) > 0 else 0
        with _conn() as c:
            for pid, hit in hits_by_pid.items():
                c.execute("UPDATE predictions SET graded=1, hit=? WHERE date=? AND game_pk=? AND player_id=?",
                          (hit, date, game_pk, pid))
    log.info("grading pass done (%d games checked)", len(pending[:40]))


def result_log() -> dict:
    """The permanent record: every graded prediction ever, with rolling
    calibration and Brier vs the constant baseline."""
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT p, hit FROM predictions WHERE graded=1").fetchall()
        days = c.execute("SELECT COUNT(DISTINCT date) FROM predictions WHERE graded=1").fetchone()[0]
    if not rows:
        return {"n": 0, "days": 0}
    n = len(rows)
    base = sum(h for _, h in rows) / n
    brier = round(sum((p - h) ** 2 for p, h in rows) / n, 4)
    brier_const = round(sum((base - h) ** 2 for p, h in rows) / n, 4)
    buckets = {}
    for p, h in rows:
        lo = int(p * 10) * 10
        b = buckets.setdefault(lo, {"n": 0, "hits": 0, "p_sum": 0.0})
        b["n"] += 1; b["hits"] += h; b["p_sum"] += p
    calibration = [
        {"bucket": f"{lo}-{lo+10}%", "n": b["n"],
         "predicted": round(b["p_sum"] / b["n"] * 100, 1),
         "actual": round(b["hits"] / b["n"] * 100, 1)}
        for lo, b in sorted(buckets.items())
    ]
    return {"n": n, "days": days, "brier_model": brier, "brier_constant": brier_const,
            "overall_hit_rate": round(base * 100, 1), "calibration": calibration}


def fair_american(p: float) -> int:
    """The model's fair price for a probability."""
    p = min(max(p, 0.01), 0.99)
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def attach_odds(data: dict) -> dict:
    """Join live 'to record a hit' prices onto today's projections and
    compute EV at the best available price. EV needs no de-vig: it is
    the actual profit expectation at the bettable price -- the vig is
    already the hurdle inside it. Degrades to blanks without an odds key."""
    events = odds_api.get_events()
    if not events:
        data["odds_note"] = "no odds key on this service -- add ODDS_API_KEY to light up prices"
        return data
    ev_cache: dict = {}
    for r in data.get("projections", []):
        gpk = r["game_pk"]
        if gpk not in ev_cache:
            ev = odds_api.find_event(events, r.get("home_name", ""), r.get("away_name", ""))
            props = odds_api.get_event_props(ev.get("id"), "batter_hits") if ev else None
            ev_cache[gpk] = props
        props = ev_cache[gpk]
        r["fair"] = fair_american(r["p"])
        for side, prob in (("over", r["p"]), ("under", 1 - r["p"])):
            priced = odds_api.player_prop_prices(props, "batter_hits", r["name"], side=side) if props else None
            if not priced or priced.get("point") != 0.5:
                continue
            bp = odds_api.best_price(priced["prices"])
            if not bp:
                continue
            dec = odds_api.american_to_decimal(bp[1])
            r[f"price_{side}"] = bp[1]
            r[f"book_{side}"] = bp[0]
            r[f"ev_{side}"] = round((prob * dec - 1) * 100, 1)
    return data
