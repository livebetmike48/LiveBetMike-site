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
import kbacktest     # season windows + the shared odds archive (pay once ever)
import kseason
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
                      hand_cache: dict, venue: str | None,
                      rows_fn=None,
                      season: int | None = None,
                      rejects: dict | None = None,
                      box: dict | None = None,
                      league_pppa: float | None = None) -> list[dict]:
    """Both starters of one final game: point-in-time hits+walks
    distributions and the actual boxscore numbers to grade against.
    Every skipped starter is COUNTED by cause in `rejects` -- a zero-start
    run must name its reason instead of shrugging (the Aug-26 lesson:
    2025 priced nothing, silently)."""
    def _rej(key):
        if rejects is not None:
            rejects[key] = rejects.get(key, 0) + 1
    rows_fn = rows_fn or parlay.get_player_season_rows
    if box is None:
        box = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=20).json()
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        pitching = (box.get("teams") or {}).get(side) or {}
        batting = (box.get("teams") or {}).get(opp) or {}
        pitchers = pitching.get("pitchers") or []
        order = batting.get("battingOrder") or []
        if not pitchers or not order:
            _rej("no_lineup_or_pitchers")
            continue
        starter_id = pitchers[0]
        sp = (pitching.get("players") or {}).get(f"ID{starter_id}") or {}
        pstats = ((sp.get("stats") or {}).get("pitching")) or {}
        actuals = {m: pstats.get(BOX_KEYS[m]) for m in LADDERS}
        if any(v is None for v in actuals.values()):
            _rej("no_boxscore_actuals")
            continue
        name = ((sp.get("person") or {}).get("fullName")) or str(starter_id)
        try:
            s_rows = rows_fn(starter_id, True)
        except Exception:
            _rej("starter_rows_error")
            continue
        if starter_id not in hand_cache:
            try:
                hand_cache[starter_id] = parlay.get_starter_hand(starter_id)
            except Exception:
                hand_cache[starter_id] = None
        hand = hand_cache[starter_id]
        if hand not in ("L", "R"):
            _rej("no_hand")
            continue
        lineup = []
        for pid in list(order)[:9]:
            try:
                # before-filter at the SOURCE: every consumer of these rows
                # (slot rates, majority side, workload patience) sees only
                # the past -- the workload leak died here
                b_rows = kmodel.rows_before(rows_fn(pid, False), date_str)
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
                league_pppa=league_pppa,
                park_factor_value=pprops.park_factor(venue, market,
                                                     year=season))
            if not d or d.get("error"):
                _rej(f"dist_gated_{market}")
                continue
            entry["markets"][market] = {
                "dist": d["dist"], "mean": d["mean"], "actual": actuals[market]}
        entry["starter_id"] = starter_id
        if entry["markets"]:
            out.append(entry)
        else:
            _rej("no_market_priced")
    return out


def run_pp_backtest(days: int, progress=None,
                    end_date: str | None = None) -> dict:
    """Walk the window, price every start point-in-time, grade the ladder.
    Returns per-market Brier + calibration buckets."""
    end, season, rows_fn = kbacktest._season_window(days, end_date,
                                                    market=False)
    rates = pprops.league_rates(season)
    hand_cache: dict = {}
    preds = {m: [] for m in LADDERS}   # (p_over, outcome, line)
    starts_priced = 0
    rejects: dict = {}
    pppa_days = 0     # days the workload arm actually had league-P/PA fuel
    for i in range(1, days + 1):
        date_str = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        if progress:
            top = ""
            if not starts_priced and rejects:
                k = max(((k, v) for k, v in rejects.items()
                         if isinstance(v, int)), key=lambda kv: kv[1],
                        default=None)
                top = f" (top reject: {k[0]} x{k[1]})" if k else ""
            progress(f"day {i}/{days} — {starts_priced} starts priced{top}")
        try:
            games = backtest._final_games(date_str)
        except Exception as e:
            log.warning("pp backtest: schedule failed %s: %s", date_str, e)
            continue
        day = []
        for g in games:
            try:
                bx = requests.get(f"{MLB_BASE}/game/{g['gamePk']}/boxscore",
                                  timeout=20).json()
                day.append((g, bx))
            except Exception as e:
                rejects["game_error"] = rejects.get("game_error", 0) + 1
                rejects["last_game_error"] = f"{type(e).__name__}: {e}"
                log.warning("pp backtest: boxscore %s failed: %s",
                            g.get("gamePk"), e)
        pppa = None
        try:
            rows_lists = []
            for g, bx in day:
                for _side in ("home", "away"):
                    _ps = (((bx.get("teams") or {}).get(_side) or {})
                           .get("pitchers")) or []
                    if _ps:
                        try:
                            rows_lists.append(kmodel.rows_before(
                                rows_fn(_ps[0], True), date_str))
                        except Exception:
                            pass
            pppa = pprops.league_pitches_per_pa(rows_lists)
        except Exception:
            pppa = None
        if pppa is not None:
            pppa_days += 1
        for g, bx in day:
            try:
                starts = _game_prop_starts(g["gamePk"], date_str, rates,
                                           hand_cache,
                                           (g.get("venue") or {}).get("name"),
                                           rows_fn=rows_fn, season=season,
                                           rejects=rejects,
                                           box=bx, league_pppa=pppa)
            except Exception as e:
                rejects["game_error"] = rejects.get("game_error", 0) + 1
                rejects["last_game_error"] = f"{type(e).__name__}: {e}"
                log.warning("pp backtest: game %s failed: %s", g.get("gamePk"), e)
                continue
            for s in starts:
                counted = False
                for market, md in s["markets"].items():
                    for line in LADDERS[market]:
                        p = pprops.calibrate(market,
                                             kmodel.prob_over(md["dist"], line))
                        preds[market].append((p, 1 if md["actual"] > line else 0, line))
                    counted = True
                if counted:
                    starts_priced += 1
    report = {"days": days, "starts": starts_priced,
              "rejects": rejects, "pppa_days": pppa_days,
              "season": season, "end_date": end.strftime("%Y-%m-%d"),
              "rows_source": ("season-scoped" if season != kseason.current_season()
                              else "live"),
              "calibrated": {m: bool(pprops.P_CALIB.get(m)) for m in LADDERS},
              "markets": {}}
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


# ------------------------------------------------------------------ fit
def fit_pp_calibration(market: str, run_report: dict) -> dict:
    """Within-run curve for ONE market from its raw buckets (>=100 preds,
    the K Fit's floor). Points land in pprops.P_CALIB as FRACTIONS -- the
    scale pprops.calibrate consumes. Returns receipts."""
    mk = (run_report.get("markets") or {}).get(market) or {}
    pts, used, total = [], 0, 0
    for b in mk.get("buckets") or []:
        total += 1
        if b.get("n", 0) >= 100 and b.get("pred") is not None \
           and b.get("actual") is not None:
            pts.append((float(b["pred"]) / 100.0, float(b["actual"]) / 100.0))
            used += 1
    pts.sort()
    pprops.P_CALIB[market] = pts
    return {"market": market, "buckets_used": used, "buckets_total": total,
            "points": pts}


# ------------------------------------------------------------ market test
ODDS_MARKETS = {"hits": "pitcher_hits_allowed", "walks": "pitcher_walks"}
PP_EV_MIN, PP_EV_MAX = 2.0, 20.0     # the K harness's paper convention


def _amer_units(price: float) -> float:
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def _imp(price: float) -> float:
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def run_pp_market_backtest(days: int, market: str, progress=None,
                           end_date: str | None = None) -> dict:
    """Units vs REAL closing lines for one props market, archive-first
    through the SAME k_odds_archive (market key is part of the key, so
    hits/walks history shares the pay-once-ever store). Same grading
    discipline as the K market test: closing snapshot at commence time,
    per-book best price on the model's side, 2-20%% counted band, flat 1u."""
    if market not in ODDS_MARKETS:
        return {"error": f"market must be one of {sorted(ODDS_MARKETS)}"}
    end, season, rows_fn = kbacktest._season_window(days, end_date, market=True)
    rates = pprops.league_rates(season)
    odds_key = ODDS_MARKETS[market]
    kbacktest._fetch_stats["odds_api"] = 0
    kbacktest._fetch_stats["odds_hit"] = 0
    hand_cache: dict = {}
    bets, no_price = [], 0
    for i in range(1, days + 1):
        date_str = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        if progress:
            progress(f"{market} day {i}/{days} -- {len(bets)} bets")
        snapshot = f"{date_str}T23:00:00Z"
        try:
            events = kbacktest._hist_events(snapshot)
        except Exception as e:
            log.warning("pp market: events failed %s: %s", date_str, e)
            continue
        try:
            games = backtest._final_games(date_str)
        except Exception:
            continue
        gmap = {}
        for g in games:
            try:
                st = _game_prop_starts(g["gamePk"], date_str, rates,
                                       hand_cache,
                                       (g.get("venue") or {}).get("name"),
                                       rows_fn=rows_fn, season=season)
            except Exception:
                continue
            for s0 in st:
                gmap[s0["starter"].lower()] = s0
        if not gmap:
            continue
        for ev in events or []:
            eid = ev.get("id")
            commence = ev.get("commence_time") or f"{date_str}T23:00:00Z"
            try:
                data = kbacktest._hist_odds(eid, commence, odds_key)
            except Exception:
                continue
            books = ((data or {}).get("data") or {}).get("bookmakers") or []
            quotes: dict = {}
            for bk in books:
                for m0 in bk.get("markets") or []:
                    if m0.get("key") != odds_key:
                        continue
                    for o in m0.get("outcomes") or []:
                        pl = (o.get("description") or "").lower()
                        pt = o.get("point")
                        side = o.get("name")
                        pr = o.get("price")
                        if pl and pt is not None and side in ("Over", "Under") \
                           and pr is not None and float(pt) % 1 == 0.5:
                            quotes.setdefault((pl, float(pt)), {}).setdefault(
                                side, []).append(float(pr))
            for (pl, pt), sides in quotes.items():
                s0 = gmap.get(pl)
                if not s0 or market not in s0["markets"]:
                    continue
                if "Over" not in sides or "Under" not in sides:
                    continue
                md = s0["markets"][market]
                p_over = pprops.calibrate(market,
                                          kmodel.prob_over(md["dist"], pt))
                for side, p_side in (("Over", p_over), ("Under", 1 - p_over)):
                    best = max(sides[side])
                    ev_pct = (p_side * (1 + _amer_units(best)) - 1) * 100
                    if not (PP_EV_MIN <= ev_pct <= PP_EV_MAX):
                        continue
                    actual = md["actual"]
                    won = (actual > pt) if side == "Over" else (actual < pt)
                    bets.append({
                        "date": date_str, "starter": s0["starter"],
                        "market": market, "side": side, "line": pt,
                        "price": best, "ev": round(ev_pct, 1),
                        "model_p": round(p_side, 4),
                        "won": int(won),
                        "units": round(_amer_units(best), 3) if won else -1.0})
                    break     # one side per line, the model's side
    if not bets:
        return {"error": "no bets in window (no priced lines matched starts)",
                "no_price": no_price, "season": season,
                "odds_fetches": {"api": kbacktest._fetch_stats["odds_api"],
                                 "archive": kbacktest._fetch_stats["odds_hit"]}}
    units = round(sum(b["units"] for b in bets), 2)
    wins = sum(b["won"] for b in bets)
    _store_market(market, bets)
    bands = []
    for lo, hi in ((2, 5), (5, 10), (10, 15), (15, 20)):
        sel = [b for b in bets if lo <= b["ev"] < hi or (hi == 20 and b["ev"] == 20)]
        if sel:
            bands.append({"band": f"{lo}-{hi}%", "bets": len(sel),
                          "wins": sum(b["won"] for b in sel),
                          "units": round(sum(b["units"] for b in sel), 2)})
    return {"market": market, "season": season,
            "end_date": end.strftime("%Y-%m-%d"), "days": days,
            "bets": len(bets), "wins": wins, "units": units,
            "roi": round(units / len(bets) * 100, 1),
            "bands": bands,
            "calibrated": bool(pprops.P_CALIB.get(market)),
            "odds_fetches": {"api": kbacktest._fetch_stats["odds_api"],
                             "archive": kbacktest._fetch_stats["odds_hit"]}}


def _store_market(market: str, bets: list):
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS pp_market_bets (
            ts REAL, market TEXT, bet TEXT)""")
        now = time.time()
        for b in bets:
            c.execute("INSERT INTO pp_market_bets VALUES (?,?,?)",
                      (now, market, json.dumps(b)))


# ------------------------------------------------------------ season suite
def run_pp_season_suite(season: int, market_test: bool = False,
                        progress=None) -> dict:
    """The K season suite's exact policy, per props market: burn-in raw
    to May 15 -> within-year curve per market -> OOS to Sep 28 on it ->
    optional units vs that season's real closing lines."""
    this_year = int(time.strftime("%Y"))
    if season >= this_year:
        return {"error": f"{season} is the live season"}
    saved = {m: list(pprops.P_CALIB.get(m) or []) for m in LADDERS}
    out: dict = {"season": season}
    try:
        for m in LADDERS:
            pprops.P_CALIB[m] = []
        if progress: progress(f"{season}: props burn-in (raw)…")
        burn = run_pp_backtest(45, progress=progress,
                               end_date=f"{season}-05-15")
        out["burn_in"] = {m: {k: (burn["markets"].get(m) or {}).get(k)
                              for k in ("n", "brier", "brier_const")}
                          for m in LADDERS}
        out["burn_rejects"] = burn.get("rejects")
        out["burn_starts"] = burn.get("starts")
        out["fit"] = {m: fit_pp_calibration(m, burn) for m in LADDERS}
        if progress: progress(f"{season}: props OOS (calibrated)…")
        oos = run_pp_backtest(130, progress=progress,
                              end_date=f"{season}-09-28")
        out["oos"] = oos["markets"]
        out["oos_rejects"] = oos.get("rejects")
        out["oos_starts"] = oos.get("starts")
        out["oos_pppa_days"] = oos.get("pppa_days")
        if market_test:
            for m in LADDERS:
                if progress: progress(f"{season}: {m} market test…")
                out.setdefault("market", {})[m] = run_pp_market_backtest(
                    130, m, progress=progress, end_date=f"{season}-09-28")
    except ValueError as e:
        out["error"] = str(e)
    except Exception:
        log.exception("pp season suite %s failed", season)
        out["error"] = "pp season suite failed -- see server log"
    finally:
        for m in LADDERS:
            pprops.P_CALIB[m] = saved[m]
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS pp_season_runs (
            ts REAL, season INTEGER, report TEXT)""")
        c.execute("INSERT INTO pp_season_runs VALUES (?,?,?)",
                  (time.time(), season, json.dumps(out)))
    return out


from contextlib import contextmanager


@contextmanager
def pp_knobs(park_weight=None, workload_weight=None):
    """Temporarily set pprops' variant knobs; ALWAYS restores on exit
    (kseason.knobs' pattern)."""
    saved = (pprops.P_PARK_WEIGHT, pprops.P_WORKLOAD_WEIGHT)
    try:
        if park_weight is not None:
            pprops.P_PARK_WEIGHT = float(park_weight)
        if workload_weight is not None:
            pprops.P_WORKLOAD_WEIGHT = float(workload_weight)
        yield
    finally:
        pprops.P_PARK_WEIGHT, pprops.P_WORKLOAD_WEIGHT = saved


# Per-market variant arms -- hits and walks each race their OWN ideas.
PP_ARMS = {
    "base":     {},
    "park0":    {"park_weight": 0.0},       # the never-settled Coors receipt
    "workload": {"workload_weight": 1.0},   # patient lineups shorten outings
}


def run_pp_season_suite_arm(season: int, arm: str = "base",
                            market_test: bool = False, progress=None) -> dict:
    if arm not in PP_ARMS:
        return {"error": f"unknown props arm '{arm}'"}
    with pp_knobs(**PP_ARMS[arm]):
        out = run_pp_season_suite(season, market_test=market_test,
                                  progress=progress)
    out["arm"] = arm
    return out


_pp_state = {"status": "idle", "progress": "", "season": None}
_pp_last: dict = {}


def start_pp_season_suite(season: int, market_test: bool = False) -> dict:
    with _lock:
        if _pp_state["status"] == "running":
            return {"error": f"a props season suite is already running "
                             f"({_pp_state['season']}) -- {_pp_state['progress']}"}
        _pp_state.update({"status": "running", "season": season,
                          "progress": "starting…"})

    def _work():
        global _pp_last
        try:
            _pp_last = run_pp_season_suite(
                season, market_test=market_test,
                progress=lambda m: _pp_state.__setitem__("progress", m))
            _pp_state.update({"status": "idle", "progress": f"done -- {season}"})
        except Exception as e:
            log.exception("pp season thread")
            _pp_state.update({"status": "idle", "progress": f"failed: {e}"})

    threading.Thread(target=_work, daemon=True).start()
    return {"started": True, "season": season, "market": market_test}


def pp_season_state() -> dict:
    hist = []
    try:
        with _conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS pp_season_runs (
                ts REAL, season INTEGER, report TEXT)""")
            for ts, season, rep in c.execute(
                    "SELECT ts, season, report FROM pp_season_runs "
                    "ORDER BY ts DESC LIMIT 20"):
                try:
                    r = json.loads(rep)
                except Exception:
                    r = {}
                hist.append({"ts": ts, "season": season,
                             "error": r.get("error"),
                             "markets": list((r.get("oos") or {}).keys())})
    except Exception:
        pass
    return {"run": dict(_pp_state), "last_result": _pp_last or None,
            "history": hist}
