"""
K Plays -- posts the K model's plays to Discord when a live edge clears
the threshold. Lives inside the site service (the model, odds client, and
forward log are already here); posting goes through a channel WEBHOOK, so
no second bot token and no gateway connection.

House rules:
  - Flat 1U per play (KPLAYS_UNITS). That's the staking every validated
    market test and the forward log use -- no invented staking models.
  - Edges above KPLAYS_MAX_EV (default 20%) are SKIPPED, same suspect-line
    rule as the market test: that big vs a live line is model error or a
    stale/suspended price, not value.
  - Every posted play is also frozen in the K Board forward log (the scan
    runs the same build), so the public record grades the same reads.
  - Scans cost Odds API credits (~2/game each). Defaults are deliberately
    modest: hourly, 10:00-19:00 ET, today only. All env-tunable.

Railway (site service) variables:
  DISCORD_KPLAYS_WEBHOOK  -- required to enable; channel webhook URL
  KPLAYS_MIN_EV=5.0       -- post threshold (percent EV at best price)
  KPLAYS_MAX_EV=20.0      -- suspect ceiling (skip above)
  KPLAYS_UNITS=1.0        -- stake shown per play
  KPLAYS_POLL_MINUTES=60  -- scan cadence
  KPLAYS_START_HOUR_ET=10 / KPLAYS_END_HOUR_ET=19 -- scan window
  KPLAYS_SCAN_TOMORROW=0  -- 1 = also scan tomorrow's openers (doubles
                             credit spend; how you catch soft openers)
"""
import os
import time
import sqlite3
import logging
import threading
from datetime import datetime, timedelta, timezone

import requests

import kboard

log = logging.getLogger("kplays")

WEBHOOK = os.getenv("DISCORD_KPLAYS_WEBHOOK", "")
MIN_EV = float(os.getenv("KPLAYS_MIN_EV", "5.0"))
MAX_EV = float(os.getenv("KPLAYS_MAX_EV", "20.0"))
UNITS = float(os.getenv("KPLAYS_UNITS", "1.0"))
POLL_MINUTES = max(15, int(os.getenv("KPLAYS_POLL_MINUTES", "60")))
START_HOUR_ET = int(os.getenv("KPLAYS_START_HOUR_ET", "10"))
END_HOUR_ET = int(os.getenv("KPLAYS_END_HOUR_ET", "19"))
SCAN_TOMORROW = os.getenv("KPLAYS_SCAN_TOMORROW", "0") == "1"
DB_PATH = os.getenv("DB_PATH", "odds_history.db")

_started = False


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS kplays_posted (
        date TEXT, starter_id INTEGER, side TEXT, line REAL,
        ev REAL, price INTEGER, posted_ts INTEGER,
        PRIMARY KEY (date, starter_id, side, line))""")
    return conn


def _already_posted(date: str, starter_id: int, side: str, line: float) -> bool:
    with _conn() as c:
        return c.execute(
            "SELECT 1 FROM kplays_posted WHERE date=? AND starter_id=? AND side=? AND line=?",
            (date, starter_id, side, line)).fetchone() is not None


def _mark_posted(date, starter_id, side, line, ev, price):
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO kplays_posted VALUES (?,?,?,?,?,?,?)",
                  (date, starter_id, side, line, ev, price, int(time.time())))


def _find_plays(data: dict) -> list[dict]:
    """Every side clearing the band [MIN_EV, MAX_EV] on a priced half-point
    line, minus anything already posted."""
    plays = []
    for s in data.get("starters", []):
        if s.get("status") != "ok" or s.get("line") is None or s.get("ev_skipped"):
            continue
        for side in ("over", "under"):
            ev = s.get("ev_" + side)
            book = s.get(side)
            if ev is None or not book:
                continue
            if ev < MIN_EV:
                continue
            if ev > MAX_EV:
                log.info("kplays: skipping %s %s %.1f at +%.1f%% EV (above %.0f%% "
                         "suspect ceiling)", s["starter"], side, s["line"], ev, MAX_EV)
                continue
            if _already_posted(data["date"], s["starter_id"], side, s["line"]):
                continue
            plays.append({"date": data["date"], "starter": s["starter"],
                          "starter_id": s["starter_id"], "team": s["team"],
                          "opp": s["opp"], "side": side, "line": s["line"],
                          "price": book["price"], "book": book["book"],
                          "ev": ev, "p": s["p_over"] if side == "over" else round(1 - s["p_over"], 4),
                          "mean_k": s.get("mean_k"),
                          "lineup_posted": s.get("lineup_posted"),
                          "n_books": s.get("n_books", 0)})
    return plays


def _post_play(p: dict) -> bool:
    price = f"{p['price']:+d}"
    u = f"{UNITS:g}U"
    lineup = "lineup posted" if p["lineup_posted"] else "lineup NOT posted — priced at league, number may move"
    embed = {
        "title": f"K PLAY — {p['starter']} {p['side'].upper()} {p['line']} Ks",
        "description": (f"**{price}** at **{p['book']}** ({p['n_books']} books priced) — **{u}**\n"
                        f"Model: **{p['p']*100:.1f}%** to hit · EV **+{p['ev']}%** per $1 at best price\n"
                        f"Projected {p['mean_k']} K · {p['team']} vs {p['opp']} · {lineup}"),
        "footer": {"text": ("BETA — K model is in forward validation; this read is frozen in the "
                            "K Board result log and graded against the real boxscore. "
                            "Research, not advice.")},
        "color": 0x4caf7d,
    }
    try:
        r = requests.post(WEBHOOK, json={"embeds": [embed]}, timeout=15)
        if r.status_code in (200, 204):
            return True
        log.warning("kplays webhook %s: %s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("kplays webhook failed: %s", e)
    return False


def _in_window() -> bool:
    hour_et = (datetime.now(timezone.utc) - timedelta(hours=4)).hour
    return START_HOUR_ET <= hour_et < END_HOUR_ET


def _scan_once():
    offsets = [0] + ([1] if SCAN_TOMORROW else [])
    for off in offsets:
        try:
            data = kboard.refresh(off)
        except Exception as e:
            log.warning("kplays scan (d=%d) failed: %s", off, e)
            continue
        plays = _find_plays(data)
        if not plays:
            log.info("kplays scan (d=%d): %d starters, no plays over +%.1f%%",
                     off, len(data.get("starters", [])), MIN_EV)
            continue
        for p in plays:
            if _post_play(p):
                _mark_posted(p["date"], p["starter_id"], p["side"], p["line"],
                             p["ev"], p["price"])
                log.info("kplays: posted %s %s %.1f (+%.1f%%)",
                         p["starter"], p["side"], p["line"], p["ev"])


def _loop():
    log.info("kplays: scanner up — every %d min, %02d:00-%02d:00 ET, min EV +%.1f%%, "
             "tomorrow=%s", POLL_MINUTES, START_HOUR_ET, END_HOUR_ET, MIN_EV,
             "on" if SCAN_TOMORROW else "off")
    while True:
        try:
            if _in_window():
                _scan_once()
            else:
                log.info("kplays: outside %02d:00-%02d:00 ET window, sleeping",
                         START_HOUR_ET, END_HOUR_ET)
        except Exception as e:
            log.error("kplays loop error: %s", e)
        time.sleep(POLL_MINUTES * 60)


def start():
    """Called once from app.py at startup. No webhook configured = feature
    off, one log line, zero cost."""
    global _started
    if _started:
        return
    _started = True
    if not WEBHOOK:
        log.info("kplays: DISCORD_KPLAYS_WEBHOOK not set — play alerts disabled")
        return
    threading.Thread(target=_loop, daemon=True).start()
