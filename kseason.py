"""
Season scoping for the K model's multi-year backtests.

The live pipeline fetches CURRENT-season rows (parlay.get_player_season_rows
is hardwired to today's year). A 2024 backtest priced with 2026 rows would be
a clean-looking number built on the wrong players -- this module exists so
that can never happen silently.

Three jobs, all additive; nothing here touches the live board path:

1. season_rows(pid, is_pitcher, season) -- full-season Savant rows for ANY
   season via the validated statcast_api.fetch_statcast (which already takes
   date bounds), persisted in sqlite so each player-season is fetched from
   Savant exactly once, ever. Point-in-time discipline stays where it already
   lives: kbacktest filters with kmodel.rows_before(date) per predicted day.

2. rows_provider(season) -- the router. Current season -> the exact live
   parlay path (byte-identical behavior, its own caching). Past season ->
   the persistent cache. kbacktest calls through this, so "which year's
   players" is decided in one place.

3. knobs(...) -- context manager that flips kmodel's variant knobs
   (K_MATCHUP_WEIGHT / K_CSW_DELTA_WEIGHT) at runtime and ALWAYS restores
   them. The knobs are read from env at import, so multi-arm suites in one
   process need this; restoring in a finally means a crashed run can't leave
   an arm's knob bleeding into the next run or the live board.
"""
import gzip
import json
import logging
import os
import sqlite3
import time
from collections import OrderedDict
from contextlib import contextmanager

import parlay
import statcast_api
import kmodel

log = logging.getLogger("kseason")

# Same volume the odds archive lives on -- pay Savant once, replay forever.
ROWS_DB = os.getenv("KSEASON_DB", os.getenv("DB_PATH", "odds_history.db"))

# Wide bounds per season; statcast_api already filters to regular season.
SEASON_START = "{season}-03-15"
SEASON_END = "{season}-11-10"


def _conn():
    conn = sqlite3.connect(ROWS_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS k_season_rows (
        player_id INTEGER, is_pitcher INTEGER, season INTEGER,
        rows_gz BLOB, n_rows INTEGER, fetched_ts REAL,
        PRIMARY KEY (player_id, is_pitcher, season))""")
    return conn


# In-process row cache -- BOUNDED. The original dict here grew without
# limit: every player-season any suite touched stayed decoded in RAM
# (~1KB/pitch measured, ~2.8MB per player-season), so a full-season
# matrix run pinned 1.5-2GB and back-to-back seasons stacked on top --
# the climbing Railway memory graph. Now an LRU with a byte budget:
# hot players stay decoded, cold ones fall back to the sqlite disk
# cache (~130ms to reload one player-season -- seconds per suite, not
# minutes). KSEASON_MEM_MB sizes it; junk values fall back to the
# default instead of crashing import (house rule).
_EST_BYTES_PER_ROW = 1000          # measured: decoded Savant pitch row in RAM
try:
    _MEM_MB = max(64, int(os.getenv("KSEASON_MEM_MB", "512")))
except (TypeError, ValueError):
    _MEM_MB = 512
_MEM_MAX_ROWS = _MEM_MB * 1_000_000 // _EST_BYTES_PER_ROW

_mem: OrderedDict = OrderedDict()  # (pid, role, season) -> rows, LRU order
_mem_rows = 0                      # sum of len(rows) across cached entries
fetch_stats = {"savant": 0, "disk": 0, "mem": 0, "evict": 0}


def _remember(key, rows):
    """Insert into the LRU and evict oldest entries past the byte budget.
    Never evicts the entry just inserted (a single oversized player still
    caches -- the budget bounds the TOTAL, not one player)."""
    global _mem_rows
    if key in _mem:
        _mem.move_to_end(key)
        return
    _mem[key] = rows
    _mem_rows += max(1, len(rows))
    while _mem_rows > _MEM_MAX_ROWS and len(_mem) > 1:
        _, old = _mem.popitem(last=False)
        _mem_rows -= max(1, len(old))
        fetch_stats["evict"] += 1
        if fetch_stats["evict"] % 200 == 0:
            log.info("season cache: %d evictions (budget %dMB holding)",
                     fetch_stats["evict"], _MEM_MB)


def clear_cache() -> dict:
    """Drop every decoded player-season and return what was freed. Called
    when a suite finishes so the site process goes back to baseline --
    the disk cache is untouched (Savant is still paid exactly once)."""
    global _mem_rows
    freed = {"players": len(_mem), "est_mb": round(_mem_rows * _EST_BYTES_PER_ROW / 1e6, 1)}
    _mem.clear()
    _mem_rows = 0
    return freed


def cache_stats() -> dict:
    """Live cache receipts for the season-suite UI polling."""
    return {"players": len(_mem),
            "est_mb": round(_mem_rows * _EST_BYTES_PER_ROW / 1e6, 1),
            "budget_mb": _MEM_MB, **fetch_stats}


def season_rows(player_id: int, is_pitcher: bool, season: int) -> list[dict]:
    """Every pitch-level row for one player in one season. Fetched from
    Savant at most once ever; served from memory, then disk, then the API."""
    key = (int(player_id), bool(is_pitcher), int(season))
    if key in _mem:
        fetch_stats["mem"] += 1
        _mem.move_to_end(key)      # LRU: a hit is recency
        return _mem[key]
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT rows_gz FROM k_season_rows WHERE player_id=? AND is_pitcher=? AND season=?",
            (key[0], int(key[1]), key[2])).fetchone()
        if row is not None:
            rows = json.loads(gzip.decompress(row[0]).decode("utf-8"))
            fetch_stats["disk"] += 1
            _remember(key, rows)
            return rows
        rows = statcast_api.fetch_statcast(
            key[0], key[1],
            SEASON_START.format(season=season), SEASON_END.format(season=season))
        fetch_stats["savant"] += 1
        conn.execute(
            "INSERT OR REPLACE INTO k_season_rows VALUES (?,?,?,?,?,?)",
            (key[0], int(key[1]), key[2],
             gzip.compress(json.dumps(rows).encode("utf-8")), len(rows), time.time()))
        conn.commit()
        _remember(key, rows)
        log.info("season rows %s %s %s: %d rows fetched from Savant",
                 key[0], "P" if key[1] else "B", season, len(rows))
        return rows
    finally:
        conn.close()


def current_season() -> int:
    return int(time.strftime("%Y"))


def rows_provider(season: int):
    """The router kbacktest calls through. Current season = the exact live
    parlay path (its caching, its behavior, untouched). Past season = the
    persistent per-season cache. Never mixes years."""
    season = int(season)
    if season == current_season():
        return parlay.get_player_season_rows
    def _past(player_id: int, is_pitcher: bool) -> list[dict]:
        return season_rows(player_id, is_pitcher, season)
    return _past


@contextmanager
def knobs(matchup_weight: float | None = None,
          csw_delta_weight: float | None = None):
    """Temporarily set kmodel's variant knobs; ALWAYS restores on exit.
    None = leave that knob exactly as it is."""
    saved = (kmodel.K_MATCHUP_WEIGHT, kmodel.K_CSW_DELTA_WEIGHT)
    try:
        if matchup_weight is not None:
            kmodel.K_MATCHUP_WEIGHT = float(matchup_weight)
        if csw_delta_weight is not None:
            kmodel.K_CSW_DELTA_WEIGHT = float(csw_delta_weight)
        yield
    finally:
        kmodel.K_MATCHUP_WEIGHT, kmodel.K_CSW_DELTA_WEIGHT = saved
