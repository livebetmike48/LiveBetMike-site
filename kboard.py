"""
K Board -- today's starters through the validated K model, priced against
live pitcher_strikeouts lines. The strikeouts twin of the Model Board.

House rules carried over:
  - Reads the same validated row data as the backtest (parlay layer); the
    live path is kbacktest's input assembly with before=None and lineups
    from today's boxscore instead of a final one. kmodel is UNTOUCHED.
  - Lineup not posted yet -> all nine slots priced at league and the row
    says so loudly. No invented lineups.
  - Whole-number lines can push; the model's P(over) has no push mass, so
    EV is only computed on half-point lines (same rule as the market test).
  - PERMANENT RESULT LOG: the FIRST priced read of each start is frozen
    (insert-or-ignore) before first pitch and graded next day against the
    real boxscore -- the forward, out-of-sample record. No cherry-picking,
    no revisions.

Fully separate from matchups.py / model.py / projections.py / pitchers.py.
"""
import os
import json
import math
import sqlite3
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

import parlay
import odds_api
import kmodel

try:
    import parks
except ImportError:
    parks = None

log = logging.getLogger("kboard")

MLB_BASE = "https://statsapi.mlb.com/api/v1"
DB_PATH = os.getenv("DB_PATH", "odds_history.db")
REFRESH_SECONDS = 900   # rebuild at most every 15 min, and only when viewed
EV_LOG_MIN = 2.0        # paper-track units simulate flat-betting edges >= this

_boards: dict = {}   # date -> {"status", "data", "built", "progress"}
_graded_on: set = set()
_lock = threading.Lock()


# ---------- storage: the frozen forward log ----------

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS k_board_log (
        date TEXT, game_pk INTEGER, starter_id INTEGER, name TEXT,
        line REAL, p_over REAL, p_over_raw REAL,
        price_over INTEGER, book_over TEXT, ev_over REAL,
        price_under INTEGER, book_under TEXT, ev_under REAL,
        lineup_posted INTEGER, logged_ts INTEGER,
        actual_k INTEGER, cleared INTEGER,
        PRIMARY KEY (date, starter_id))""")
    return conn


def _log_predictions(data: dict):
    """Freeze the first priced read of each start. INSERT OR IGNORE means
    later rebuilds (moving lines, posted lineups) never revise a logged
    prediction -- logged before, graded after."""
    rows = []
    for s in data.get("starters", []):
        if s.get("status") != "ok" or s.get("line") is None or s.get("ev_skipped"):
            continue
        if not s.get("over") and not s.get("under"):
            continue
        rows.append((
            data["date"], s["game_pk"], s["starter_id"], s["starter"],
            s["line"], s["p_over"], s["p_over_raw"],
            (s.get("over") or {}).get("price"), (s.get("over") or {}).get("book"),
            s.get("ev_over"),
            (s.get("under") or {}).get("price"), (s.get("under") or {}).get("book"),
            s.get("ev_under"),
            1 if s.get("lineup_posted") else 0, int(time.time()),
        ))
    if not rows:
        return
    with _conn() as c:
        c.executemany("""INSERT OR IGNORE INTO k_board_log
            (date, game_pk, starter_id, name, line, p_over, p_over_raw,
             price_over, book_over, ev_over, price_under, book_under, ev_under,
             lineup_posted, logged_ts, actual_k, cleared)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)""", rows)


def _grade_pending(today: str):
    """Grade every logged prediction from finished past days against the
    real boxscore. Only Final games grade; everything else waits."""
    with _conn() as c:
        pending = c.execute(
            "SELECT date, game_pk, starter_id, line FROM k_board_log "
            "WHERE actual_k IS NULL AND date < ?", (today,)).fetchall()
    if not pending:
        return
    finals: dict = {}
    for date in {p[0] for p in pending}:
        try:
            sched = requests.get(f"{MLB_BASE}/schedule",
                                 params={"sportId": 1, "date": date}, timeout=20).json()
            for d in sched.get("dates", []):
                for g in d.get("games", []):
                    if (g.get("status") or {}).get("codedGameState") == "F":
                        finals[g["gamePk"]] = True
        except Exception as e:
            log.warning("k grade: schedule failed for %s: %s", date, e)
    graded = 0
    for date, game_pk, starter_id, line in pending:
        if not finals.get(game_pk):
            continue
        try:
            box = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=20).json()
        except Exception as e:
            log.warning("k grade: boxscore %s failed: %s", game_pk, e)
            continue
        actual = None
        for side in ("home", "away"):
            sp = (((box.get("teams") or {}).get(side) or {}).get("players") or {}).get(f"ID{starter_id}")
            if sp:
                actual = (((sp.get("stats") or {}).get("pitching")) or {}).get("strikeOuts")
                break
        if actual is None:
            continue
        with _conn() as c:
            c.execute("UPDATE k_board_log SET actual_k=?, cleared=? "
                      "WHERE date=? AND starter_id=?",
                      (int(actual), 1 if actual > line else 0, date, starter_id))
        graded += 1
    if graded:
        log.info("k board: graded %d predictions", graded)


def _result_log_summary() -> dict:
    """The forward record: Brier of logged P(over) vs reality, plus flat
    1u paper units on every logged edge >= EV_LOG_MIN at the logged price."""
    with _conn() as c:
        rows = c.execute(
            "SELECT p_over, cleared, ev_over, price_over, ev_under, price_under "
            "FROM k_board_log WHERE cleared IS NOT NULL").fetchall()
        days = c.execute(
            "SELECT COUNT(DISTINCT date) FROM k_board_log WHERE cleared IS NOT NULL"
        ).fetchone()[0]
        pending = c.execute(
            "SELECT COUNT(*) FROM k_board_log WHERE cleared IS NULL").fetchone()[0]
    if not rows:
        return {"n": 0, "pending": pending}
    brier = round(sum((p - h) ** 2 for p, h, *_ in rows) / len(rows), 4)
    base = sum(h for _, h, *_ in rows) / len(rows)
    brier_constant = round(sum((base - h) ** 2 for _, h, *_ in rows) / len(rows), 4)
    units = bets = wins = 0
    for p_over, cleared, ev_o, pr_o, ev_u, pr_u in rows:
        for side_hit, ev, price in ((cleared, ev_o, pr_o), (1 - cleared, ev_u, pr_u)):
            if ev is None or price is None or ev < EV_LOG_MIN:
                continue
            bets += 1
            if side_hit:
                wins += 1
                units += odds_api.american_to_decimal(price) - 1
            else:
                units -= 1
    return {"n": len(rows), "days": days, "pending": pending,
            "brier_model": brier, "brier_constant": brier_constant,
            "bets": bets, "wins": wins, "units": round(units, 2)}


# ---------- live input assembly (kbacktest's, with before=None) ----------

def _slate(date: str) -> list[dict]:
    """The slate for ANY date straight from MLB's schedule (probables +
    teams + venue in one call) -- so tomorrow works exactly like today."""
    out = []
    try:
        sched = requests.get(f"{MLB_BASE}/schedule",
                             params={"sportId": 1, "date": date,
                                     "hydrate": "probablePitcher,team"},
                             timeout=20).json()
    except Exception as e:
        log.warning("k board: schedule failed for %s: %s", date, e)
        return out
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            teams = {}
            for side in ("home", "away"):
                t = ((g.get("teams") or {}).get(side)) or {}
                team = t.get("team") or {}
                pp = t.get("probablePitcher") or {}
                teams[side] = {
                    "abbrev": team.get("abbreviation") or team.get("teamName") or "?",
                    "name": team.get("name") or "",
                    "starter_id": pp.get("id"),
                    "starter_name": pp.get("fullName") or "TBD",
                }
            out.append({"game_pk": g.get("gamePk"),
                        "venue": ((g.get("venue") or {}).get("name")),
                        "teams": teams})
    return out


def _events_on(events: list[dict], date: str) -> list[dict]:
    """Only odds events whose first pitch falls on this ET date -- a
    series means the same team pair exists on BOTH days, and name-matching
    without a date filter would price the wrong game."""
    keep = []
    for ev in events or []:
        ct = ev.get("commence_time") or ""
        try:
            dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            et_date = (dt - timedelta(hours=4)).strftime("%Y-%m-%d")
        except Exception:
            continue
        if et_date == date:
            keep.append(ev)
    return keep


def _fair_line(kdist: dict) -> float | None:
    """The model's own line: the half-point where calibrated P(over) is
    closest to a coin flip. The number to compare openers against."""
    best, best_gap = None, None
    max_k = len(kdist["dist"])
    for half in range(max_k):
        line = half + 0.5
        p = kmodel.calibrate(kmodel.prob_over(kdist["dist"], line))
        gap = abs(p - 0.5)
        if best_gap is None or gap < best_gap:
            best, best_gap = line, gap
    return best


def _lineup_order(game_pk: int) -> list[int]:
    """{'home': [...], 'away': [...]} batting orders (player ids, slots
    1-9); empty lists before a lineup posts, {} if the boxscore fetch fails."""
    try:
        box = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=15).json()
    except Exception:
        return {}
    orders = {}
    for side in ("home", "away"):
        team = ((box.get("teams") or {}).get(side)) or {}
        orders[side] = (team.get("battingOrder") or [])[:9]
    return orders


def _majority_side(rows: list[dict]) -> str | None:
    sides = [r.get("stand") for r in rows if r.get("stand")]
    return max(set(sides), key=sides.count) if sides else None


def _park_k(venue: str | None) -> float | None:
    if not venue or parks is None:
        return None
    fn = getattr(parks, "k_factor_for", None)
    try:
        return fn(venue) if fn else None
    except Exception:
        return None


def _build_lineup(order: list[int]) -> tuple[list, int]:
    """kmodel lineup entries from a posted batting order; ([]None x9, 0)
    when the lineup isn't up yet."""
    if not order:
        return [None] * 9, 0
    lineup = []
    known = 0
    for pid in order[:9]:
        try:
            rows = parlay.get_player_season_rows(pid, False)
        except Exception:
            lineup.append(None)
            continue
        side = _majority_side(rows)
        if side:
            lineup.append({"rows": rows, "side": side, "name": pid})
            known += 1
        else:
            lineup.append(None)
    while len(lineup) < 9:
        lineup.append(None)
    return lineup, known


def _price_starter(events, home_name, away_name, starter_name, kdist):
    """Live pitcher_strikeouts read: consensus line + best price each side
    + model EV. Returns dict of price fields (possibly empty)."""
    out = {"line": None, "over": None, "under": None,
           "ev_over": None, "ev_under": None, "p_over": None,
           "p_over_raw": None, "ev_skipped": None, "n_books": 0}
    ev_match = odds_api.find_event(events, home_name, away_name) if events else None
    if not ev_match:
        return out
    props = odds_api.get_event_props(ev_match.get("id"), "pitcher_strikeouts")
    if not props:
        return out
    over = odds_api.player_prop_prices(props, "pitcher_strikeouts", starter_name, side="over")
    if not over or over.get("point") is None:
        return out
    line = over["point"]
    out["line"] = line
    under = odds_api.player_prop_prices(props, "pitcher_strikeouts", starter_name, side="under")
    if under and under.get("point") != line:
        under = None  # only pair sides at the same point
    priced = kmodel.price_line(kdist, line)
    out["p_over"], out["p_over_raw"] = priced["p_over"], priced["p_over_raw"]
    out["n_books"] = len(over.get("prices") or {})
    if line != math.floor(line) + 0.5:
        out["ev_skipped"] = "whole-number line — pushes possible, EV not computed"
    bp = odds_api.best_price(over.get("prices") or {})
    if bp:
        out["over"] = {"book": bp[0], "price": bp[1]}
        if not out["ev_skipped"]:
            out["ev_over"] = round((priced["p_over"] * odds_api.american_to_decimal(bp[1]) - 1) * 100, 1)
    bp = odds_api.best_price((under or {}).get("prices") or {})
    if bp:
        out["under"] = {"book": bp[0], "price": bp[1]}
        if not out["ev_skipped"]:
            out["ev_under"] = round((priced["p_under"] * odds_api.american_to_decimal(bp[1]) - 1) * 100, 1)
    return out


def _build_board(date: str, progress: dict) -> dict:
    p_league = kmodel.league_k_rate()
    slate = _slate(date)
    events = []
    try:
        events = _events_on(odds_api.get_events(), date)
    except Exception as e:
        log.warning("k board: odds events skipped: %s", e)

    progress["total"] = sum(
        1 for g in slate for side in ("home", "away")
        if g["teams"][side]["starter_id"])
    starters = []
    for g in slate:
        orders = _lineup_order(g["game_pk"]) or {}
        for side, opp_side in (("home", "away"), ("away", "home")):
            team = g["teams"][side]          # the pitching team
            opp = g["teams"][opp_side]       # the batting team
            if not team["starter_id"]:
                continue
            entry = {"game_pk": g["game_pk"], "starter_id": team["starter_id"],
                     "starter": team["starter_name"], "team": team["abbrev"],
                     "opp": opp["abbrev"]}
            try:
                try:
                    hand = parlay.get_starter_hand(team["starter_id"])
                except Exception:
                    hand = None
                if hand not in ("L", "R"):
                    entry.update({"status": "no read", "why": "handedness unavailable"})
                    continue
                entry["hand"] = hand
                try:
                    s_rows = parlay.get_player_season_rows(team["starter_id"], True)
                except Exception:
                    s_rows = []
                order = (orders.get(opp_side) or []) if isinstance(orders, dict) else []
                lineup, known = _build_lineup(order)
                entry["lineup_posted"] = bool(order)
                entry["lineup_known_slots"] = known
                kdist = kmodel.k_distribution(
                    lineup, s_rows, hand, p_league,
                    before=None, park_k_factor=_park_k(g.get("venue")))
                if kdist is None:
                    entry.update({"status": "no read",
                                  "why": "starter sample too thin (house minimums)"})
                    continue
                entry.update({
                    "status": "ok",
                    "mean_k": kdist["mean_k"],
                    "tbf_mean": kdist["tbf_mean"],
                    "fair_line": _fair_line(kdist),
                    "league_fallback_slots": kdist["inputs"]["league_fallback_slots"],
                })
                entry.update(_price_starter(
                    events, g["teams"]["home"]["name"], g["teams"]["away"]["name"],
                    team["starter_name"], kdist))
            except Exception as e:
                log.warning("k board: %s failed: %s", team["starter_name"], e)
                entry.update({"status": "no read", "why": "build error (see logs)"})
            finally:
                progress["done"] += 1
                starters.append(entry)

    def _best_ev(s):
        evs = [e for e in (s.get("ev_over"), s.get("ev_under")) if e is not None]
        return max(evs) if evs else -999
    starters.sort(key=lambda s: -_best_ev(s))
    return {"date": date, "starters": starters,
            "lineups_posted": sum(1 for s in starters if s.get("lineup_posted")),
            "built_at": int(time.time())}


def refresh(offset: int = 0) -> dict:
    """Synchronous build for background consumers (the K plays scanner):
    builds the board, freezes new log reads, and shares the result with
    the site's cache so a scan also refreshes the tab."""
    offset = 1 if offset == 1 else 0
    date = parlay.et_date_str(offset)
    data = _build_board(date, {"done": 0, "total": 0})
    _log_predictions(data)
    with _lock:
        entry = _boards.setdefault(date, {"status": "cold", "data": None,
                                          "built": 0, "progress": {"done": 0, "total": 0}})
        entry.update({"data": data, "status": "ready", "built": time.time()})
    return data


def get_board(offset: int = 0) -> dict:
    """Board for today (offset 0) or tomorrow (offset 1). Per-date cache,
    background rebuild when stale (15 min). Grades pending log entries the
    first time each real day is viewed. Never blocks; Odds API credits are
    only spent while someone is looking."""
    offset = 1 if offset == 1 else 0
    today = parlay.et_date_str(0)
    date = parlay.et_date_str(offset)
    with _lock:
        # drop cached boards for past dates
        for d in [d for d in _boards if d < today]:
            del _boards[d]
        entry = _boards.setdefault(date, {"status": "cold", "data": None,
                                          "built": 0, "progress": {"done": 0, "total": 0}})
        fresh = (entry["status"] == "ready"
                 and time.time() - entry["built"] < REFRESH_SECONDS)
        if fresh:
            return {"status": "ready", "offset": offset,
                    "result_log": _result_log_summary(), **entry["data"]}
        if entry["status"] == "warming":
            pr = entry["progress"]
            out = {"status": "warming", "offset": offset,
                   "progress": f"start {pr['done']}/{pr['total']}"
                   if pr["total"] else "starting"}
            if entry["data"]:
                out.update({"stale": True, "result_log": _result_log_summary(),
                            **entry["data"]})
                out["status"] = "ready"
            return out
        entry["status"] = "warming"
        entry["progress"] = {"done": 0, "total": 0}
        progress = entry["progress"]
        need_grading = today not in _graded_on
        if need_grading:
            _graded_on.add(today)

    def _warm():
        try:
            if need_grading:
                try:
                    _grade_pending(today)
                except Exception as e:
                    log.warning("k board grading failed: %s", e)
            data = _build_board(date, progress)
            _log_predictions(data)
            with _lock:
                _boards[date].update({"data": data, "status": "ready",
                                      "built": time.time()})
            log.info("K board ready for %s: %d starters (%d priced)",
                     date, len(data["starters"]),
                     sum(1 for s in data["starters"] if s.get("line") is not None))
        except Exception as e:
            log.error("K board build failed for %s: %s", date, e)
            with _lock:
                _boards[date]["status"] = "cold" if not _boards[date]["data"] else "ready"

    threading.Thread(target=_warm, daemon=True).start()
    with _lock:
        if _boards[date]["data"]:
            return {"status": "ready", "stale": True, "offset": offset,
                    "result_log": _result_log_summary(), **_boards[date]["data"]}
    return {"status": "warming", "offset": offset, "progress": "starting"}
