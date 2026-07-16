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
}

PROP_ROADMAP = [
    {"prop": "Hits (1+)", "status": "backtesting", "note": "log5 model live in lab"},
    {"prop": "Home runs", "status": "planned", "note": "needs HR-rate model + park factors to be honest"},
    {"prop": "Pitcher strikeouts", "status": "planned", "note": "K-rate log5 vs lineup, needs PA-distribution work"},
    {"prop": "Total bases", "status": "idea", "note": "only after hits model proves calibrated"},
]

_run_state = {"status": "idle", "progress": "", "started": None}
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
    return cfg


def run_backtest_async(days: int) -> bool:
    """Kick a backtest in a background thread. False if one is running."""
    with _lock:
        if _run_state["status"] == "running":
            return False
        _run_state.update({"status": "running", "progress": "starting…",
                           "started": time.time()})

    def _progress(done, total, n):
        _run_state["progress"] = f"day {done}/{total} — {n} predictions graded"

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
