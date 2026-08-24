"""Season backtest suite — the burn-in-then-OOS runner for past MLB seasons.

Policy (proposed July/Aug, unvetoed): the first ~6 weeks of a season fit
that YEAR'S calibration curve (a 2026 curve on 2024 games would be regime
bleed); everything after grades OUT-OF-SAMPLE on that within-year curve.

Additive module: lab.py / kbacktest.py / kmodel.py untouched. Uses the same
DB (lab.DB_PATH), the same token gate pattern, and kbacktest's season
machinery (end_date windows, per-season rows/league/parks, odds archive).

Phases per run:
  A. burn-in  : RAW accuracy backtest (curve emptied) over the season's
                first ~6 weeks -> per-bucket calibration receipts
  B. fit      : within-year curve from phase-A buckets (>=100-pred floor,
                same rule as the live Fit)
  C. oos      : accuracy backtest over the remainder, graded on that curve
  D. market   : optional units test vs real closing lines over the same
                OOS span (archive-first; 2023 auto-clipped to props-history
                start 2023-05-03 by kbacktest's own refusal rules)

The live model's curve/knobs are saved and restored around every phase
(try/finally) -- a season run can never leave the live board mis-curved.
"""
from __future__ import annotations
import json, logging, sqlite3, threading, time
from datetime import date

import kmodel
import kbacktest
import lab

log = logging.getLogger("kseasonlab")

BURN_END_MD = "-05-15"     # season start -> mid-May ~= 6 weeks
BURN_DAYS = 45
OOS_END_MD = "-09-28"      # regular-season close
OOS_DAYS = 130             # May 21 -> Sep 28 stays inside one season
MIN_BUCKET_PREDS = 100     # same floor as the live Fit


def _init():
    with sqlite3.connect(lab.DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS k_season_runs (
            ts REAL, season INTEGER, phase TEXT, report TEXT)""")


def _store(season: int, phase: str, report: dict):
    _init()
    with sqlite3.connect(lab.DB_PATH) as c:
        c.execute("INSERT INTO k_season_runs VALUES (?,?,?,?)",
                  (time.time(), season, phase, json.dumps(report)))


def _fit_points(calibration: list) -> tuple[list, dict]:
    """Within-year curve from a raw run's calibration buckets.
    Same shape the live Fit produces: [(predicted_mid, actual)]."""
    pts, used, total = [], 0, 0
    for b in calibration or []:
        total += 1
        n = b.get("n", 0)
        if n >= MIN_BUCKET_PREDS and b.get("predicted") is not None \
           and b.get("actual") is not None:
            pts.append((float(b["predicted"]), float(b["actual"])))
            used += 1
    pts.sort()
    return pts, {"buckets_used": used, "buckets_total": total,
                 "min_bucket_preds": MIN_BUCKET_PREDS}


def run_season_suite(season: int, market: bool = False,
                     progress=None) -> dict:
    """Full burn-in -> fit -> OOS (-> market) suite for one past season."""
    this_year = date.today().year
    if season >= this_year:
        return {"error": f"{season} is the live season — use the normal Lab "
                         "runs; season suites are for completed years"}
    if season < 2015:
        return {"error": "Savant-era data only (2015+)"}

    saved_pts = list(kmodel.K_CALIB_POINTS)
    saved_w = kmodel.K_CALIB_WEIGHT
    out: dict = {"season": season, "policy": {
        "burn_in_end": f"{season}{BURN_END_MD}", "burn_days": BURN_DAYS,
        "oos_end": f"{season}{OOS_END_MD}", "oos_days": OOS_DAYS,
        "curve": "fit within-year on burn-in, OOS graded on it"}}
    try:
        # A. burn-in, RAW
        kmodel.K_CALIB_POINTS = []
        if progress: progress(f"{season}: burn-in raw backtest…")
        burn = kbacktest.run_k_backtest(BURN_DAYS,
                                        end_date=f"{season}{BURN_END_MD}")
        _store(season, "burn_raw", burn)
        out["burn_in"] = {"predictions": burn.get("n"),
                          "starts": burn.get("starts"),
                          "brier": burn.get("brier_model"),
                          "avg_projected_k": burn.get("avg_projected_k"),
                          "avg_actual_k": burn.get("avg_actual_k"),
                          "rows_source": burn.get("rows_source")}
        # B. within-year fit
        pts, fit_receipt = _fit_points(burn.get("calibration"))
        out["fit"] = fit_receipt | {"points": pts}
        if not pts:
            out["fit"]["note"] = ("no bucket met the floor — OOS runs RAW "
                                  "(honest, not silently curved)")
        # C. OOS on that curve
        kmodel.K_CALIB_POINTS = pts
        kmodel.K_CALIB_WEIGHT = saved_w if pts else 0.0
        if progress: progress(f"{season}: OOS calibrated backtest…")
        oos = kbacktest.run_k_backtest(OOS_DAYS,
                                       end_date=f"{season}{OOS_END_MD}")
        _store(season, "oos", oos)
        out["oos"] = {"predictions": oos.get("n"),
                      "starts": oos.get("starts"),
                      "brier": oos.get("brier_model"),
                      "brier_constant": oos.get("brier_constant"),
                      "brier_naive": oos.get("brier_naive"),
                      "calibration": oos.get("calibration"),
                      "rows_source": oos.get("rows_source")}
        # D. market (optional, credits — archive-first makes reruns free)
        if market:
            if progress: progress(f"{season}: market test vs closing lines…")
            mkt = kbacktest.run_k_market_backtest(OOS_DAYS,
                                                  end_date=f"{season}{OOS_END_MD}")
            _store(season, "market", mkt)
            mkt.pop("_candidates", None)
            out["market"] = {k: v for k, v in mkt.items()
                             if k not in ("sample_bets",)}
    except ValueError as e:
        out["error"] = str(e)          # season-window refusals arrive loudly
    except Exception:
        log.exception("season suite %s failed", season)
        out["error"] = "season suite failed — see server log"
    finally:
        kmodel.K_CALIB_POINTS = saved_pts
        kmodel.K_CALIB_WEIGHT = saved_w
    return out


_state = {"status": "idle", "progress": "", "season": None}
_lock = threading.Lock()
_last_result: dict = {}


def start_season_suite(season: int, market: bool = False) -> dict:
    """Background start (lab's run pattern) — returns immediately; poll
    season_state(); finished receipts land in _last_result + the DB."""
    with _lock:
        if _state["status"] == "running":
            return {"error": f"a season suite is already running "
                             f"({_state['season']}) — {_state['progress']}"}
        _state.update({"status": "running", "season": season,
                       "progress": "starting…"})

    def _prog(msg):
        _state["progress"] = msg

    def _work():
        global _last_result
        try:
            _last_result = run_season_suite(season, market=market,
                                            progress=_prog)
            _state.update({"status": "idle",
                           "progress": f"done — {season}"
                           + (f" ({_last_result['error']})"
                              if "error" in _last_result else "")})
        except Exception as e:
            log.exception("season suite thread")
            _state.update({"status": "idle", "progress": f"failed: {e}"})

    threading.Thread(target=_work, daemon=True).start()
    return {"started": True, "season": season, "market": market}


def season_state() -> dict:
    return {"run": dict(_state), "last_result": _last_result or None}


def season_history() -> list[dict]:
    """Stored season-suite receipts, newest first, for the Lab UI."""
    _init()
    with sqlite3.connect(lab.DB_PATH) as c:
        rows = c.execute("SELECT ts, season, phase, report FROM k_season_runs "
                         "ORDER BY ts DESC LIMIT 60").fetchall()
    out = []
    for ts, season, phase, rep in rows:
        try:
            r = json.loads(rep)
        except Exception:
            r = {}
        out.append({"ts": ts, "season": season, "phase": phase,
                    "brier": r.get("brier_model"), "predictions": r.get("n"),
                    "units": r.get("units"), "roi": r.get("roi")})
    return out
