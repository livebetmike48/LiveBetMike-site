"""
Props backtest -- hits allowed + walks allowed, graded the way the K
harness grades strikeouts. Point-in-time everywhere: every rate is built
from rows BEFORE the game being priced, lineups come from the boxscore
(the K harness's "actual" mode -- fine for Brier work, where the question
is calibration, not market timing).

What this measures: does pprops' distribution match reality? Ladder of
over-probabilities per start, graded 0/1 from the real boxscore, Brier'd
against a per-line constant baseline, bucketed for calibration. Zero odds
credits -- no lines involved.

What it deliberately does NOT do yet: market tests (needs verified
market keys + a credit decision) and calibration curves (fit AFTER the
first real runs show the bucket shape -- fitting a curve before seeing
raw buckets is how you launder bias into "correction").

Storage: pp_backtest_runs in the same volume DB. One row per run,
report JSON, so runs survive redeploys and the tab renders history.
"""
import os
import json
import time
import sqlite3
import logging
import threading
from datetime import datetime, timedelta, timezone

import requests

import parlay
import kmodel
import pprops
import backtest

log = logging.getLogger("ppbacktest")

DB_PATH = os.getenv("DB_PATH", "matchups.db")
MLB_BASE = "https://statsapi.mlb.com/api/v1"

# Fixed ladders, one per market -- same shape as the K harness's 3.5-7.5.
LADDERS = {
    "hits": [2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
    "walks": [0.5, 1.5, 2.5, 3.5],
}
BOX_KEYS = {"hits": "hits", "walks": "baseOnBalls"}

_state = {"running": False, "progress": "", "started": 0}
_lock = threading.Lock()


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS pp_backtest_runs (
        ts INTEGER PRIMARY KEY, days INTEGER, report TEXT)""")
    return conn


def _majority_side(b_rows: list, before: str) -> str | None:
    """Which side the hitter has batted from, point-in-time."""
    l = r = 0
    for row in b_rows:
        d = row.get("game_date")
        if d and d >= before:
            continue
        if row.get("stand") == "L":
            l += 1
        elif row.get("stand") == "R":
            r += 1
    if not l and not r:
        return None
    return "L" if l >= r else "R"


def _game_prop_starts(game_pk: int, date_str: str, rates: dict,
                      hand_cache: dict, venue: str | None) -> list[dict]:
    """Both starters of one final game: point-in-time hits+walks
    distributions and the actual boxscore numbers to grade against."""
    box = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=20).json()
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        pitching = (box.get("teams") or {}).get(side) or {}
        batting = (box.get("teams") or {}).get(opp) or {}
        pitchers = pitching.get("pitchers") or []
        order = batting.get("battingOrder") or []
        if not pitchers or not order:
            continue
        starter_id = pitchers[0]
        sp = (pitching.get("players") or {}).get(f"ID{starter_id}") or {}
        pstats = ((sp.get("stats") or {}).get("pitching")) or {}
        actuals = {m: pstats.get(BOX_KEYS[m]) for m in LADDERS}
        if any(v is None for v in actuals.values()):
            continue
        name = ((sp.get("person") or {}).get("fullName")) or str(starter_id)
        try:
            s_rows = parlay.get_player_season_rows(starter_id, True)
        except Exception:
            continue
        if starter_id not in hand_cache:
            try:
                hand_cache[starter_id] = parlay.get_starter_hand(starter_id)
            except Exception:
                hand_cache[starter_id] = None
        hand = hand_cache[starter_id]
        if hand not in ("L", "R"):
            continue
        lineup = []
        for pid in list(order)[:9]:
            try:
                b_rows = parlay.get_player_season_rows(pid, False)
            except Exception:
                lineup.append(None)
                continue
            b_side = _majority_side(b_rows, date_str)
            lineup.append({"rows": b_rows, "side": b_side, "name": pid}
                          if b_side else None)
        while len(lineup) < 9:
            lineup.append(None)
        start_pks = kmodel.fetch_start_games(starter_id, before=date_str)
        entry = {"starter": name, "date": date_str, "markets": {}}
        for market in LADDERS:
            d = pprops.prop_distribution(
                market, lineup, s_rows, hand, rates[market],
                before=date_str, start_game_pks=start_pks,
                park_factor_value=pprops.park_factor(venue, market))
            if not d or d.get("error"):
                continue
            entry["markets"][market] = {
                "dist": d["dist"], "mean": d["mean"], "actual": actuals[market]}
        if entry["markets"]:
            out.append(entry)
    return out


def run_pp_backtest(days: int, progress=None) -> dict:
    """Walk the window, price every start point-in-time, grade the ladder.
    Returns per-market Brier + calibration buckets."""
    rates = pprops.league_rates()
    end = datetime.now(timezone.utc) - timedelta(hours=4)
    hand_cache: dict = {}
    preds = {m: [] for m in LADDERS}   # (p_over, outcome, line)
    starts_priced = 0
    for i in range(1, days + 1):
        date_str = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        if progress:
            progress(f"day {i}/{days} — {starts_priced} starts priced")
        try:
            games = backtest._final_games(date_str)
        except Exception as e:
            log.warning("pp backtest: schedule failed %s: %s", date_str, e)
            continue
        for g in games:
            try:
                starts = _game_prop_starts(g["gamePk"], date_str, rates,
                                           hand_cache, g.get("venue"))
            except Exception as e:
                log.warning("pp backtest: game %s failed: %s", g.get("gamePk"), e)
                continue
            for s in starts:
                counted = False
                for market, md in s["markets"].items():
                    for line in LADDERS[market]:
                        p = kmodel.prob_over(md["dist"], line)
                        preds[market].append((p, 1 if md["actual"] > line else 0, line))
                    counted = True
                if counted:
                    starts_priced += 1
    report = {"days": days, "starts": starts_priced, "markets": {}}
    for market, rows in preds.items():
        if not rows:
            report["markets"][market] = {"n": 0}
            continue
        n = len(rows)
        brier = sum((p - o) ** 2 for p, o, _ in rows) / n
        # constant baseline: per-line empirical rate over this same window
        by_line: dict = {}
        for p, o, line in rows:
            by_line.setdefault(line, []).append(o)
        base = {line: sum(v) / len(v) for line, v in by_line.items()}
        brier_const = sum((base[line] - o) ** 2 for _, o, line in rows) / n
        buckets = []
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            sel = [(p, o) for p, o, _ in rows if lo <= p < hi or (b == 9 and p == 1)]
            if sel:
                buckets.append({
                    "bucket": f"{b*10}-{b*10+10}%", "n": len(sel),
                    "pred": round(sum(p for p, _ in sel) / len(sel) * 100, 1),
                    "actual": round(sum(o for _, o in sel) / len(sel) * 100, 1)})
        report["markets"][market] = {
            "n": n, "brier": round(brier, 4),
            "brier_constant": round(brier_const, 4),
            "beats": brier < brier_const,
            "buckets": buckets,
            "mean_projected": round(sum(1 for _ in rows) and
                                    sum(p for p, _, _ in rows) / n, 4),
        }
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO pp_backtest_runs (ts, days, report) "
                  "VALUES (?,?,?)", (int(time.time()), days, json.dumps(report)))
    return report


def run_async(days: int) -> bool:
    with _lock:
        if _state["running"]:
            return False
        _state.update({"running": True, "progress": "starting…",
                       "started": int(time.time())})

    def _go():
        try:
            run_pp_backtest(days, progress=lambda s: _state.__setitem__("progress", s))
        except Exception as e:
            log.error("pp backtest failed: %s", e)
            _state["progress"] = f"failed: {e}"
        finally:
            _state["running"] = False

    threading.Thread(target=_go, daemon=True).start()
    return True


def history(limit: int = 6) -> dict:
    with _conn() as c:
        rows = c.execute("SELECT ts, days, report FROM pp_backtest_runs "
                         "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return {"running": _state["running"], "progress": _state["progress"],
            "runs": [{"ts": ts, "days": d, **json.loads(rep)}
                     for ts, d, rep in rows]}
