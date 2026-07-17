"""
Logs a daily snapshot of every book's lines into SQLite -- building our OWN
odds archive from today forward (The Odds API's historical archive is a
paid tier; this costs nothing beyond requests we already make). This is the
dataset future projection models get backtested against.
"""
import os
import sqlite3
import logging
from contextlib import contextmanager

log = logging.getLogger("odds_log")

DB_PATH = os.getenv("DB_PATH", "odds_history.db")


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
        c.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                snap_date TEXT,
                home_team TEXT,
                away_team TEXT,
                commence_time TEXT,
                book TEXT,
                market TEXT,
                outcome TEXT,
                price INTEGER,
                point REAL,
                PRIMARY KEY (snap_date, home_team, away_team, book, market, outcome)
            )
        """)


def log_snapshot(snap_date: str, events: list[dict], market_key: str):
    """Stores every outcome from every book for the day. INSERT OR REPLACE
    keyed by date, so re-runs the same day just refresh the snapshot."""
    if not events:
        return 0
    rows = 0
    with _conn() as c:
        for ev in events:
            for book in ev.get("bookmakers", []) or []:
                for market in book.get("markets", []) or []:
                    if market.get("key") != market_key:
                        continue
                    for outcome in market.get("outcomes", []) or []:
                        c.execute(
                            "INSERT OR REPLACE INTO odds_snapshots VALUES (?,?,?,?,?,?,?,?,?)",
                            (
                                snap_date,
                                ev.get("home_team"), ev.get("away_team"),
                                ev.get("commence_time"),
                                book.get("title"), market_key,
                                outcome.get("name"), outcome.get("price"),
                                outcome.get("point"),
                            ),
                        )
                        rows += 1
    log.info("Odds archive: stored %d rows for %s (%s)", rows, snap_date, market_key)
    return rows


def archive_summary() -> dict:
    """How much history we've accumulated -- surfaced on the site."""
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(DISTINCT snap_date), COUNT(*) FROM odds_snapshots"
        ).fetchone()
        return {"days": row[0] or 0, "rows": row[1] or 0}
