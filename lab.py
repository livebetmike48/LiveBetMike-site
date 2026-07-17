"""
Model Lab -- the model's home. Stores every backtest run (so improvement
is visible over time), holds the tunable knobs Mike controls, and tracks
the prop-model roadmap. Persists to SQLite on the volume (DB_PATH); falls
back to a local file with a visible warning if no volume is mounted.
"""
import os
import json
import time
import sqlite3
import logging
import threading
from contextlib import contextmanager

import model
import backtest

log = logging.getLogger("lab")

DB_PATH = os.getenv("DB_PATH", "odds_history.db")
PERSISTENT = DB_PATH.startswith("/data")

# The knobs, with honest descriptions of what they do
CONFIG_DEFAULTS = {
    "pa_vs_starter": {"value": 2.6, "label": "Expected PAs vs the starter",
                      "note": "League avg for lineup regulars ~2.6"},
    "pa_vs_pen": {"value": 1.6, "label": "Expected PAs vs the bullpen",
                  "note": "Rest of a regular's ~4.2 PA/game"},
    "min_batter_pa": {"value": 40, "label": "Min batter PA vs hand",
                      "note": "Below this the model refuses to predict"},
    "min_starter_pa": {"value": 60, "label": "Min starter PA vs side",
                       "note": "Below this the model refuses to predict"},
    "shrink_pa": {"value": 150, "label": "Shrinkage (phantom league PAs)",
                  "note": "Regression to the mean — higher = more skeptical of hot/cold splits. 0 = raw rates (the overconfident v1)"},
    "personal_pa": {"value": 1, "label": "Personal playing time (1=on, 0=off)",
                    "note": "Scale PAs by the batter's own PA/game — captures lineup slot + pinch-hit risk from real logs"},
    "xba_weight": {"value": 0, "label": "xBA blend weight (0-1)",
                   "note": "0 = actual hit rates (v4 champion), 1 = fully luck-stripped expected rates, 0.5 = half and half"},
    "arsenal_weight": {"value": 0, "label": "Arsenal matchup weight (0-1)",
                       "note": "Tilt batter rates by the starter's real pitch usage vs his per-pitch results (heavily shrunk). 0 = off (champion)"},
    "prior_weight": {"value": 0, "label": "Player prior weight (0-1)",
                     "note": "Shrink toward each player's own projected talent (uploaded priors, e.g. preseason Steamer) instead of league avg. 0 = league (champion)"},
}

PROP_ROADMAP = [
    {"prop": "Hits O/U 0.5 (both sides)", "status": "live-beta", "note": "log5 model on the board with EV vs live prices; result log grading daily"},
    {"prop": "Batter walks 0.5", "status": "planned", "note": "log5 on BB rates — same machinery as hits, likely the easiest next binary"},
    {"prop": "Home runs 0.5", "status": "planned", "note": "HR-rate model + park factors required to be honest"},
    {"prop": "Pitcher strikeouts", "status": "planned", "note": "per-hitter K probs → sum-of-Bernoullis distribution; batters-faced leash profiles (~23 / ~17); price the line vs the shape — this engine then unlocks the whole count-prop tier"},
    {"prop": "Total bases / H+R+RBI", "status": "idea", "note": "count props — need the distribution engine from the K model"},
    {"prop": "Pitcher outs / ER / hits / walks allowed", "status": "idea", "note": "distribution + leash modeling; after the K template"},
    {"prop": "Stolen bases 0.5", "status": "idea", "note": "attempt rates are player/manager-specific but loggable"},
]

_run_state = {"status": "idle", "progress": "", "started": None}
_market_state = {"status": "idle", "progress": ""}
_lock = threading.Lock()


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS backtest_runs (
            ts INTEGER, days INTEGER, config TEXT, report TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS model_config (
            key TEXT PRIMARY KEY, value REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS market_runs (
            ts INTEGER, days INTEGER, report TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS player_priors (
            name_folded TEXT PRIMARY KEY, display_name TEXT, rate REAL, pa INTEGER)""")


def load_priors_csv(csv_text: str) -> dict:
    """Parse a projections CSV (FanGraphs export format: Name, PA, H
    columns among others) into per-PA hit-rate priors. Duplicate names are
    DROPPED (can't disambiguate two Will Smiths honestly)."""
    import csv as csv_mod
    import io
    reader = csv_mod.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        return {"error": "no header row found"}
    cols = {c.strip().strip('"').lower(): c for c in reader.fieldnames}
    name_c, pa_c, h_c = cols.get("name"), cols.get("pa"), cols.get("h")
    if not (name_c and pa_c and h_c):
        return {"error": f"need Name, PA, H columns — found {list(cols)[:12]}"}
    parsed, dupes = {}, set()
    for row in reader:
        try:
            name = (row[name_c] or "").strip()
            pa = float(row[pa_c]); h = float(row[h_c])
        except (TypeError, ValueError, KeyError):
            continue
        if not name or pa < 50:
            continue
        key = model.fold_name(name)
        if key in parsed:
            dupes.add(key)
            continue
        parsed[key] = {"display_name": name, "rate": h / pa, "pa": int(pa)}
    for key in dupes:
        parsed.pop(key, None)
    if not parsed:
        return {"error": "no usable rows parsed"}
    init_db()
    with _conn() as c:
        c.execute("DELETE FROM player_priors")
        for key, p in parsed.items():
            c.execute("INSERT INTO player_priors VALUES (?, ?, ?, ?)",
                      (key, p["display_name"], p["rate"], p["pa"]))
    return {"loaded": len(parsed), "dropped_duplicates": len(dupes)}


def get_priors() -> dict:
    init_db()
    with _conn() as c:
        return {name: rate for name, _, rate, _ in
                c.execute("SELECT name_folded, display_name, rate, pa FROM player_priors")}


def priors_count() -> int:
    init_db()
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM player_priors").fetchone()[0]


def get_config() -> dict:
    init_db()
    cfg = {k: dict(v) for k, v in CONFIG_DEFAULTS.items()}
    with _conn() as c:
        for key, value in c.execute("SELECT key, value FROM model_config"):
            if key in cfg:
                cfg[key]["value"] = value
    return cfg


def set_config(updates: dict) -> dict:
    init_db()
    with _conn() as c:
        for key, value in updates.items():
            if key in CONFIG_DEFAULTS:
                c.execute("INSERT OR REPLACE INTO model_config VALUES (?, ?)",
                          (key, float(value)))
    return get_config()


def _apply_config():
    cfg = get_config()
    model.PA_VS_STARTER = cfg["pa_vs_starter"]["value"]
    model.PA_VS_PEN = cfg["pa_vs_pen"]["value"]
    model.SHRINK_PA = cfg["shrink_pa"]["value"]
    model.PERSONAL_PA = int(cfg["personal_pa"]["value"])
    model.XBA_WEIGHT = max(0.0, min(1.0, cfg["xba_weight"]["value"]))
    model.ARSENAL_WEIGHT = max(0.0, min(1.0, cfg["arsenal_weight"]["value"]))
    model.PRIOR_WEIGHT = max(0.0, min(1.0, cfg["prior_weight"]["value"]))
    model.PRIORS = get_priors() if model.PRIOR_WEIGHT > 0 else {}
    return cfg


def run_backtest_async(days: int) -> bool:
    """Kick a backtest in a background thread. False if one is running."""
    with _lock:
        if _run_state["status"] == "running":
            return False
        _run_state.update({"status": "running", "progress": "starting…",
                           "started": time.time()})

    def _progress(done, total, n, detail=""):
        _run_state["progress"] = f"day {done}/{total} ({detail}) — {n} predictions graded"

    def _work():
        try:
            cfg = _apply_config()
            report = backtest.run_backtest(days, progress=_progress)
            with _conn() as c:
                c.execute("INSERT INTO backtest_runs VALUES (?, ?, ?, ?)",
                          (int(time.time()), days,
                           json.dumps({k: v["value"] for k, v in cfg.items()}),
                           json.dumps(report)))
            _run_state.update({"status": "idle", "progress": "done"})
        except Exception as e:
            log.error("backtest failed: %s", e)
            _run_state.update({"status": "idle", "progress": f"failed: {e}"})

    threading.Thread(target=_work, daemon=True).start()
    return True


def lab_state() -> dict:
    init_db()
    runs = []
    with _conn() as c:
        for ts, days, cfg, report in c.execute(
                "SELECT ts, days, config, report FROM backtest_runs ORDER BY ts DESC LIMIT 20"):
            runs.append({"ts": ts, "days": days, "config": json.loads(cfg),
                         "report": json.loads(report)})
    return {
        "run": dict(_run_state),
        "config": get_config(),
        "roadmap": PROP_ROADMAP,
        "history": runs,
        "persistent": PERSISTENT,
        "priors_loaded": priors_count(),
        "market": dict(_market_state),
        "market_history": market_history(),
    }


def export_csv() -> str:
    """Every run's calibration rows, flat -- opens straight in Sheets."""
    state = lab_state()
    lines = ["run_time,days,bucket,n,predicted_pct,actual_pct,brier_model,brier_constant,brier_naive"]
    for run in state["history"]:
        rep = run["report"]
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(run["ts"]))
        for c in rep.get("calibration", []):
            lines.append(f"{when},{run['days']},{c['bucket']},{c['n']},{c['predicted']},"
                         f"{c['actual']},{rep.get('brier_model')},{rep.get('brier_constant')},{rep.get('brier_naive')}")
    return "\n".join(lines)


def run_market_async(days: int) -> bool:
    with _lock:
        if _market_state["status"] == "running":
            return False
        _market_state.update({"status": "running", "progress": "starting…"})

    def _progress(day, total, games, cands):
        _market_state["progress"] = f"day {day}/{total} — {games} games priced, {cands} edge candidates"

    def _work():
        try:
            _apply_config()
            report = backtest.run_market_backtest(days, progress=_progress)
            init_db()
            with _conn() as c:
                c.execute("INSERT INTO market_runs VALUES (?, ?, ?)",
                          (int(time.time()), days, json.dumps(report)))
            _market_state.update({"status": "idle", "progress": "done"})
        except Exception as e:
            log.error("market backtest failed: %s", e)
            _market_state.update({"status": "idle", "progress": f"failed: {e}"})

    threading.Thread(target=_work, daemon=True).start()
    return True


def market_history() -> list[dict]:
    init_db()
    out = []
    with _conn() as c:
        for ts, days, report in c.execute(
                "SELECT ts, days, report FROM market_runs ORDER BY ts DESC LIMIT 10"):
            out.append({"ts": ts, "days": days, "report": json.loads(report)})
    return out
