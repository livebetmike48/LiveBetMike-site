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
import statcast_api
import odds_api
import kmodel
import kseason

try:
    import parks
except ImportError:
    parks = None

log = logging.getLogger("kboard")

MLB_BASE = "https://statsapi.mlb.com/api/v1"
DB_PATH = os.getenv("DB_PATH", "odds_history.db")
# Rebuild cadence: rebuilds happen ONLY when someone views the board, and
# at most this often. Same-day rebuilds mostly hit the daily caches
# (rows, lineups, team rates), so the marginal cost is the odds refetch
# (~1 credit per priced game) + arithmetic. On the 5M-credit plan a 5-min
# cadence is ~2-3K credits/day worst case -- so the floor is 120s, and
# the frozen log is first-read-wins, so faster rebuilds can never touch
# the record. KBOARD_REFRESH_SECONDS env; default 300 (was 900).
try:
    REFRESH_SECONDS = max(120, int(os.getenv("KBOARD_REFRESH_SECONDS", "300") or 300))
except ValueError:
    REFRESH_SECONDS = 300
EV_LOG_MIN = 2.0        # paper-track units simulate flat-betting edges >= this
# Paper bets follow the SAME 2-20% policy the market tests validated:
# edges above EV_LOG_MAX are logged and shown (20+ band in the breakdown)
# but NOT counted as paper bets -- they're overwhelmingly stale/thin
# lines, and the tests that earned the model's credentials excluded them.
EV_LOG_MAX = float(os.getenv("KBOARD_EV_LOG_MAX", "20.0"))
# Cumulative EV thresholds and exclusive EV bands for the forward-log
# breakdown. Thresholds match the Lab's market-test convention (>= X) so
# the forward record is directly comparable; bands answer the
# winner's-curse question (does ROI fall as EV rises?) and isolate the
# >20% zone the market tests exclude as suspect.
EV_BANDS = ((2.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, None))

# Model era stamped onto every logged read. Bump when the MODEL changes
# (mixture, filter, curve source -- anything that alters predictions), so
# the recap can split records by era instead of silently mixing them.
# v1 = launch model (through 2026-07-27). v2 = start-only workload
# mixture + curve refit from the 2,950-pred filtered run (2026-07-28).
K_MODEL_VER = os.getenv("K_MODEL_VERSION", "v2")

# ---------- the forward cage: paper variants priced at the SAME frozen
# moment, same line, same prices as the live read. v2.2 stays the only
# live/public model; these arms exist purely in the k_cage log so the
# rest-of-season answers "which model" with receipts. Arms are LOCKED
# here on purpose -- no mid-stream additions (KCAGE=0 disables).
KCAGE_ON = os.getenv("KCAGE", "1") != "0"
CAGE_ARMS = (
    ("matchup", {"matchup_weight": 1.0}),
    ("cswdelta", {"csw_delta_weight": 1.0}),
)
_cage_coefs = {"date": None, "coefs": None}


def _cage_csw_coefs(today: str):
    """CSW-delta needs the fitted called%/SwStr% -> K-rate mapping, which
    only the backtest knows how to fit. Fit it ONCE per day, point-in-time
    (rows strictly before today), population = starters the board has seen
    in the last 30 days plus today's slate. Zero odds credits. None (thin
    population / any failure) = the delta arm honestly logs nothing."""
    if _cage_coefs["date"] == today:
        return _cage_coefs["coefs"]
    coefs = None
    try:
        import kbacktest  # lazy: keep kboard's boot path independent
        with _conn() as c:
            sids = {r[0] for r in c.execute(
                "SELECT DISTINCT starter_id FROM k_board_log WHERE date >= ?",
                ((datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d"),))}
        coefs = kbacktest.fit_csw_mapping(sids, today)
    except Exception as e:
        log.warning("cage: csw mapping fit failed: %s", e)
    _cage_coefs.update({"date": today, "coefs": coefs})
    log.info("cage: csw coefs for %s: %s", today,
             "fitted" if coefs else "unavailable (delta arm idle today)")
    return coefs
K_MODEL_ERA_LABELS = {"v1": "model v1 — through 7/27",
                      "v2": "model v2 — since 7/28 (start-only workloads)"}
# The K model prices against the MAIN books only -- lines you can actually
# bet. Soft/regional books produced fantasy best-prices and fantasy EVs.
# Comma-separated Odds API book keys; also halves prop-credit cost
# (named books <=10 bill as 1 unit vs 2 for both regions).
KBOARD_BOOKS = os.getenv("KBOARD_BOOKS",
                         "fanduel,draftkings,betmgm,williamhill_us")

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
    conn.execute("""CREATE TABLE IF NOT EXISTS k_cage (
        date TEXT, game_pk INTEGER, starter_id INTEGER, name TEXT, arm TEXT,
        line REAL, p_over REAL,
        price_over INTEGER, book_over TEXT, ev_over REAL,
        price_under INTEGER, book_under TEXT, ev_under REAL,
        lineup_posted INTEGER, excluded INTEGER, logged_ts INTEGER,
        actual_k INTEGER, cleared INTEGER,
        PRIMARY KEY (date, starter_id, arm))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS k_lineup_snaps (
        date TEXT, game_pk INTEGER, starter_id INTEGER, name TEXT,
        posted_ts INTEGER, line REAL, price_over INTEGER, book_over TEXT,
        price_under INTEGER, book_under TEXT, p_over REAL,
        latest_line REAL, latest_price_over INTEGER, latest_price_under INTEGER,
        latest_ts INTEGER,
        PRIMARY KEY (date, starter_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS k_overrides (
        date TEXT, starter_id INTEGER, pitch_limit INTEGER, tbf_cap INTEGER,
        exclude INTEGER DEFAULT 0, note TEXT, set_ts INTEGER,
        PRIMARY KEY (date, starter_id))""")
    try:
        conn.execute("ALTER TABLE k_board_log ADD COLUMN excluded INTEGER")
        conn.execute("ALTER TABLE k_board_log ADD COLUMN excl_note TEXT")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE k_board_log ADD COLUMN model_ver TEXT")
        # one-time backfill: reads frozen on/after the v2 cutover date were
        # made by the v2 model before this column existed
        conn.execute("UPDATE k_board_log SET model_ver='v2' "
                     "WHERE date >= '2026-07-28' AND model_ver IS NULL")
    except Exception:
        pass  # column already exists
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
            K_MODEL_VER,
            1 if s.get("excluded") else None,
            s.get("override_note") if s.get("excluded") else None,
        ))
    cage_rows = []
    for s in data.get("starters", []):
        for arm, pv in (s.get("_cage") or {}).items():
            if pv.get("line") is None:
                continue
            cage_rows.append((
                data["date"], s["game_pk"], s["starter_id"], s["starter"], arm,
                pv["line"], pv["p_over"],
                (pv.get("over") or {}).get("price"), (pv.get("over") or {}).get("book"),
                pv.get("ev_over"),
                (pv.get("under") or {}).get("price"), (pv.get("under") or {}).get("book"),
                pv.get("ev_under"),
                1 if s.get("lineup_posted") else 0,
                1 if s.get("excluded") else None,
                int(time.time()),
            ))
    if cage_rows:
        with _conn() as c:
            c.executemany("""INSERT OR IGNORE INTO k_cage
                (date, game_pk, starter_id, name, arm, line, p_over,
                 price_over, book_over, ev_over, price_under, book_under, ev_under,
                 lineup_posted, excluded, logged_ts, actual_k, cleared)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)""", cage_rows)
    if not rows:
        return
    with _conn() as c:
        c.executemany("""INSERT OR IGNORE INTO k_board_log
            (date, game_pk, starter_id, name, line, p_over, p_over_raw,
             price_over, book_over, ev_over, price_under, book_under, ev_under,
             lineup_posted, logged_ts, model_ver, excluded, excl_note,
             actual_k, cleared)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)""", rows)


def _snap_lineup(date: str, game_pk, s: dict, snapped: set):
    """First posted sighting -> insert the moment. Later pre-start builds
    -> refresh latest_* (last write before first pitch ~= close)."""
    sid = s.get("starter_id")
    if not sid or s.get("status") != "ok":
        return
    over, under = s.get("over") or {}, s.get("under") or {}
    now = int(time.time())
    if s.get("lineup_posted") and sid not in snapped:
        with _conn() as c:
            c.execute("INSERT OR IGNORE INTO k_lineup_snaps "
                      "(date, game_pk, starter_id, name, posted_ts, line, "
                      " price_over, book_over, price_under, book_under, p_over, "
                      " latest_line, latest_price_over, latest_price_under, latest_ts) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (date, game_pk, sid, s.get("starter"), now, s.get("line"),
                       over.get("price"), over.get("book"),
                       under.get("price"), under.get("book"), s.get("p_over"),
                       s.get("line"), over.get("price"), under.get("price"), now))
        snapped.add(sid)
        log.info("lineup snap: %s posted, line %s (O %s / U %s), model %.0f%% over",
                 s.get("starter"), s.get("line"), over.get("price"),
                 under.get("price"), (s.get("p_over") or 0) * 100)
    elif sid in snapped:
        with _conn() as c:
            c.execute("UPDATE k_lineup_snaps SET latest_line=?, "
                      "latest_price_over=?, latest_price_under=?, latest_ts=? "
                      "WHERE date=? AND starter_id=?",
                      (s.get("line"), over.get("price"), under.get("price"),
                       now, date, sid))


def lineup_snaps(date: str) -> dict:
    """The tracker, readable: for each posted lineup -- when it posted,
    the line/prices at that moment, the model's read, and where the line
    sits now (or sat at last pre-start build ~= close). After some weeks
    this answers: is there edge in acting AT the post, before books move?"""
    with _conn() as c:
        rows = c.execute(
            "SELECT name, starter_id, posted_ts, line, price_over, book_over, "
            "price_under, book_under, p_over, latest_line, latest_price_over, "
            "latest_price_under, latest_ts FROM k_lineup_snaps WHERE date=? "
            "ORDER BY posted_ts", (date,)).fetchall()
    out, moved_with, moved_against = [], 0, 0
    for (name, sid, ts, line, po, bo, pu, bu, p, ll, lpo, lpu, lts) in rows:
        drift = None
        if line is not None and ll is not None and ll != line:
            model_over = (p or 0) >= 0.5
            drift = "toward model" if ((ll > line) == model_over) else "against model"
            if drift == "toward model":
                moved_with += 1
            else:
                moved_against += 1
        out.append({"starter": name, "starter_id": sid, "posted_ts": ts,
                    "at_post": {"line": line, "over": {"price": po, "book": bo},
                                "under": {"price": pu, "book": bu},
                                "model_p_over": p},
                    "latest": {"line": ll, "price_over": lpo, "price_under": lpu,
                               "ts": lts},
                    "line_drift": drift})
    return {"date": date, "n": len(out), "snaps": out,
            "line_moves": {"toward_model": moved_with, "against_model": moved_against},
            "note": ("posted_ts is the first board build that saw the lineup "
                     "(~15-30 min granularity); latest_* is the last pre-start "
                     "refresh, which stands in for the close")}


def pitches_per_pa(rows: list[dict]) -> float | None:
    """His OWN pitches per batter faced -- what turns a pitch limit into a
    batter cap. A 75-pitch limit is a different outing for a 3.5 P/PA
    strike-thrower than for a 4.3 P/PA grinder, so no league constant."""
    pa = sum(1 for r in rows
             if r.get("events") and r.get("events") not in statcast_api.NON_PA_EVENTS)
    return (len(rows) / pa) if pa and rows else None


def set_override(date: str, starter_id: int, pitch_limit: int | None = None,
                 tbf_cap: int | None = None, exclude: bool = False,
                 note: str = "") -> dict:
    """Record a pitch-count / exclusion override for one start.

    Two things it can do, and they're different:
      pitch_limit or tbf_cap -> CAP the workload. His real start lengths
        are kept and each one is capped at the limit: a limit truncates
        the top of the distribution, it does NOT make the outing
        certain. If he's getting hit he's still pulled at 14 batters, and
        that downside tail is exactly what decides unders.
      exclude -> the read is priced and shown but NOT counted in the
        record. For when the information (a hard limit, a piggyback, a
        bullpen game) makes the model's workload assumption simply wrong.

    TIMING RULE, and it is the whole reason this is trustworthy: an
    override may only be set BEFORE first pitch. After that it would be a
    revision to a frozen read, and the record's value is that nothing is
    ever revised. Excluded reads are never deleted -- they stay in the
    log, flagged, and the record reports them as their own line."""
    if not starter_id:
        return {"error": "starter_id required"}
    if pitch_limit and not tbf_cap:
        try:
            rows = parlay.get_player_season_rows(int(starter_id), True)
        except Exception:
            rows = []
        ppa = pitches_per_pa(rows)
        if not ppa:
            return {"error": "no pitch rows for this starter -- give tbf directly"}
        tbf_cap = max(3, int(round(pitch_limit / ppa)))
    if tbf_cap is not None:
        tbf_cap = max(3, min(45, int(tbf_cap)))
    with _conn() as c:
        row = c.execute("SELECT cleared, actual_k FROM k_board_log "
                        "WHERE date=? AND starter_id=?", (date, starter_id)).fetchone()
        if row and row[0] is not None:
            return {"error": "that start is already graded -- overrides are "
                             "pre-game only, never retroactive"}
        c.execute("INSERT OR REPLACE INTO k_overrides "
                  "(date, starter_id, pitch_limit, tbf_cap, exclude, note, set_ts) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (date, int(starter_id), pitch_limit, tbf_cap,
                   1 if exclude else 0, note or "", int(time.time())))
        if exclude:
            # a read already frozen today still gets flagged, not deleted
            c.execute("UPDATE k_board_log SET excluded=1, excl_note=? "
                      "WHERE date=? AND starter_id=? AND cleared IS NULL",
                      (note or "manual", date, starter_id))
    # The board is cached (REFRESH_SECONDS) and reads overrides at BUILD
    # time, so without this the cap sits there doing nothing until the
    # cache expires -- which looks exactly like a broken feature.
    binds = _invalidate_board(date)
    log.info("k override %s %s: limit=%s cap=%s exclude=%s (%s)",
             date, starter_id, pitch_limit, tbf_cap, exclude, note)
    out = {"ok": True, "date": date, "starter_id": int(starter_id),
           "pitch_limit": pitch_limit, "tbf_cap": tbf_cap,
           "exclude": bool(exclude), "note": note, "rebuilding": True}
    # Say plainly whether the cap actually changes anything. A 90-pitch
    # limit is close to a normal start for most arms; if his workload
    # already sits under the cap, the honest answer is "no effect".
    cur = _cached_tbf(date, int(starter_id))
    if tbf_cap and cur is not None:
        out["current_tbf_mean"] = cur
        if tbf_cap >= cur + 3:
            out["binds"] = "barely — his average outing is %.1f batters, " \
                           "so this only trims his longest starts" % cur
        elif tbf_cap >= cur:
            out["binds"] = "lightly — cap %d vs his %.1f average" % (tbf_cap, cur)
        else:
            out["binds"] = "yes — cap %d is below his %.1f average" % (tbf_cap, cur)
    return out


def _cached_tbf(date: str, starter_id: int):
    """His projected TBF from the last built board, for the does-this-cap-
    even-matter answer. None if the board hasn't been built yet."""
    with _lock:
        data = (_boards.get(date) or {}).get("data")
    for s in ((data or {}).get("starters") or []):
        if s.get("starter_id") == starter_id:
            return s.get("tbf_mean")
    return None


def _invalidate_board(date: str) -> bool:
    """Force the next view of this date to rebuild, so an override takes
    effect on the next load instead of whenever the 15-minute cache
    happens to expire."""
    with _lock:
        entry = _boards.get(date)
        if not entry:
            return False
        entry["built"] = 0
        return True


def clear_override(date: str, starter_id: int) -> dict:
    _invalidate_board(date)
    with _conn() as c:
        c.execute("DELETE FROM k_overrides WHERE date=? AND starter_id=?",
                  (date, int(starter_id)))
        c.execute("UPDATE k_board_log SET excluded=NULL, excl_note=NULL "
                  "WHERE date=? AND starter_id=? AND cleared IS NULL",
                  (date, int(starter_id)))
    return {"ok": True, "cleared": True}


def get_overrides(date: str) -> dict:
    with _conn() as c:
        rows = c.execute(
            "SELECT starter_id, pitch_limit, tbf_cap, exclude, note, set_ts "
            "FROM k_overrides WHERE date=?", (date,)).fetchall()
    return {sid: {"pitch_limit": pl, "tbf_cap": cap, "exclude": bool(ex),
                  "note": note, "set_ts": ts}
            for sid, pl, cap, ex, note, ts in rows}


def _apply_tbf_cap(kdist: dict, cap: int) -> dict:
    """Re-mix the SAME per-slot probabilities over capped workload samples.
    Built from kmodel's own exported pieces (slot_pa_counts +
    poisson_binomial), so a capped read can never drift from the
    validated math -- only the workload assumption changes."""
    samples = [min(t, cap) for t in (kdist.get("tbf_samples") or [])]
    if not samples:
        return kdist
    slot_probs = [s["p_k_per_pa"] for s in kdist["inputs"]["slots"]]
    weight = 1.0 / len(samples)
    dist = [0.0] * (max(samples) + 1)
    for tbf in samples:
        counts = kmodel.slot_pa_counts(tbf)
        seq = [slot_probs[i] for i in range(9) for _ in range(counts[i])]
        pb = kmodel.poisson_binomial(seq)
        for k, m in enumerate(pb):
            dist[k] += weight * m
    out = dict(kdist)
    out["dist"] = [round(m, 6) for m in dist]
    out["mean_k"] = round(sum(k * m for k, m in enumerate(dist)), 3)
    out["tbf_samples"] = samples
    out["tbf_mean"] = round(sum(samples) / len(samples), 1)
    return out


def _grade_pending(today: str):
    """Grade every logged prediction from finished past days against the
    real boxscore. Only Final games grade; everything else waits."""
    with _conn() as c:
        pending = c.execute(
            "SELECT date, game_pk, starter_id, line FROM k_board_log "
            "WHERE actual_k IS NULL AND date < ? "
            "UNION "
            "SELECT date, game_pk, starter_id, line FROM k_cage "
            "WHERE actual_k IS NULL AND date < ?", (today, today)).fetchall()
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
                      "WHERE date=? AND starter_id=? AND actual_k IS NULL",
                      (int(actual), 1 if actual > line else 0, date, starter_id))
            # cage rows grade against their OWN stored line (same rule)
            c.execute("UPDATE k_cage SET actual_k=?, "
                      "cleared=(CASE WHEN ? > line THEN 1 ELSE 0 END) "
                      "WHERE date=? AND starter_id=? AND actual_k IS NULL",
                      (int(actual), int(actual), date, starter_id))
        graded += 1
    if graded:
        log.info("k board: graded %d predictions", graded)


def cage_summary(days: int = 400) -> dict:
    """The cage scoreboard: every arm vs the LIVE model on the SAME graded
    starts (inner join on date+starter), counted-band paper convention
    (flat 1u, EV_LOG_MIN..EV_LOG_MAX, excluded reads out) plus Brier on
    every graded read. Apples-to-apples by construction: an arm is only
    ever compared on starts where both it and the live model froze a read."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=4)
              - timedelta(days=max(1, min(400, days)))).strftime("%Y-%m-%d")

    def _paper(p_over, ev_o, pr_o, ev_u, pr_u, cleared):
        """One read -> (brier_term, bet) under the house convention."""
        b = (p_over - cleared) ** 2
        best = None
        for side, ev, pr in (("over", ev_o, pr_o), ("under", ev_u, pr_u)):
            if ev is None or pr is None:
                continue
            if EV_LOG_MIN <= ev <= EV_LOG_MAX and (best is None or ev > best[1]):
                best = (side, ev, pr)
        if best is None:
            return b, None
        side, ev, pr = best
        won = (cleared == 1) if side == "over" else (cleared == 0)
        units = (pr / 100.0 if pr > 0 else 100.0 / abs(pr)) if won else -1.0
        return b, {"won": won, "units": units, "ev": ev}

    def _stats(reads):
        out = {"graded": len(reads), "brier": None, "bets": 0, "wins": 0,
               "units": 0.0, "roi": None}
        if not reads:
            return out
        bsum = 0.0
        for r in reads:
            b, bet = _paper(*r)
            bsum += b
            if bet:
                out["bets"] += 1
                out["wins"] += int(bet["won"])
                out["units"] += bet["units"]
        out["brier"] = round(bsum / len(reads), 4)
        out["units"] = round(out["units"], 2)
        if out["bets"]:
            out["roi"] = round(out["units"] / out["bets"] * 100, 1)
        return out

    with _conn() as c:
        arms = [r[0] for r in c.execute(
            "SELECT DISTINCT arm FROM k_cage WHERE date >= ?", (cutoff,))]
        out = {"days": days, "arms": {}, "note":
               "each arm vs live on the SAME graded starts; flat 1u, "
               f"{EV_LOG_MIN:g}-{EV_LOG_MAX:g}% counted band, excluded reads out"}
        for arm in arms:
            rows = c.execute(
                "SELECT k.p_over, k.ev_over, k.price_over, k.ev_under, k.price_under, k.cleared, "
                "       l.p_over, l.ev_over, l.price_over, l.ev_under, l.price_under, l.cleared "
                "FROM k_cage k JOIN k_board_log l "
                "  ON l.date = k.date AND l.starter_id = k.starter_id "
                "WHERE k.arm = ? AND k.date >= ? "
                "  AND k.cleared IS NOT NULL AND l.cleared IS NOT NULL "
                "  AND (k.excluded IS NULL OR k.excluded = 0) "
                "  AND (l.excluded IS NULL OR l.excluded = 0)",
                (arm, cutoff)).fetchall()
            arm_reads = [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]
            live_reads = [(r[6], r[7], r[8], r[9], r[10], r[11]) for r in rows]
            out["arms"][arm] = {"arm": _stats(arm_reads),
                                "live_on_same_starts": _stats(live_reads)}
        pend = c.execute("SELECT COUNT(*) FROM k_cage WHERE cleared IS NULL "
                         "AND date >= ?", (cutoff,)).fetchone()[0]
        out["pending"] = pend
    return out


def _result_log_summary() -> dict:
    """The forward record: Brier of logged P(over) vs reality, plus flat
    1u paper units on every logged edge >= EV_LOG_MIN at the logged price."""
    with _conn() as c:
        rows = c.execute(
            "SELECT p_over, cleared, ev_over, price_over, ev_under, price_under "
            "FROM k_board_log WHERE cleared IS NOT NULL "
            "AND (excluded IS NULL OR excluded = 0)").fetchall()
        n_excluded = c.execute(
            "SELECT COUNT(*) FROM k_board_log WHERE excluded = 1").fetchone()[0]
        days = c.execute(
            "SELECT COUNT(DISTINCT date) FROM k_board_log WHERE cleared IS NOT NULL"
        ).fetchone()[0]
        pending = c.execute(
            "SELECT COUNT(*) FROM k_board_log WHERE cleared IS NULL").fetchone()[0]
    if not rows:
        return {"n": 0, "pending": pending, "excluded": n_excluded}
    brier = round(sum((p - h) ** 2 for p, h, *_ in rows) / len(rows), 4)
    base = sum(h for _, h, *_ in rows) / len(rows)
    brier_constant = round(sum((base - h) ** 2 for _, h, *_ in rows) / len(rows), 4)
    units = bets = wins = 0
    for p_over, cleared, ev_o, pr_o, ev_u, pr_u in rows:
        for side_hit, ev, price in ((cleared, ev_o, pr_o), (1 - cleared, ev_u, pr_u)):
            if ev is None or price is None or ev < EV_LOG_MIN or ev > EV_LOG_MAX:
                continue
            bets += 1
            if side_hit:
                wins += 1
                units += odds_api.american_to_decimal(price) - 1
            else:
                units -= 1
    return {"n": len(rows), "days": days, "pending": pending,
            "excluded": n_excluded,
            "brier_model": brier, "brier_constant": brier_constant,
            "bets": bets, "wins": wins, "units": round(units, 2)}


# ---------- live model state (the July-24 boot fix, now actually deployed) ----------

_live_model_loaded = 0.0


def _ensure_live_model():
    """Load saved K knobs + refit the calibration curve into the live
    kmodel before pricing anything. Without this, every redeploy leaves
    the board on boot defaults with an EMPTY curve -- the raw K-shy model
    at fantasy EVs. Throttled to once per 5 minutes."""
    global _live_model_loaded
    if time.time() - _live_model_loaded < 300:
        return
    try:
        import lab
        lab._apply_k_config()
        _live_model_loaded = time.time()
    except Exception as e:
        log.warning("live model config load failed (board runs on current state): %s", e)


# ---------- opponent team K rate (real data for unknown lineup slots) ----------

_team_rate_cache: dict = {"date": None, "rates": {}}


def _team_k_rate(team_id: int, hand: str) -> float | None:
    """The opposing TEAM's season K/PA vs this starter hand, from MLB
    statSplits (vl/vr). Cached per day. None (-> league) when the split
    is thin (<500 PA) or the fetch fails -- never invented."""
    if not team_id or hand not in ("L", "R"):
        return None
    today = parlay.et_date_str(0)
    if _team_rate_cache["date"] != today:
        _team_rate_cache.update({"date": today, "rates": {}})
    key = (team_id, hand)
    if key in _team_rate_cache["rates"]:
        return _team_rate_cache["rates"][key]
    rate = None
    try:
        split = "vl" if hand == "L" else "vr"
        data = requests.get(
            f"{MLB_BASE}/teams/{team_id}/stats",
            params={"stats": "statSplits", "sitCodes": split,
                    "group": "hitting", "season": today[:4]},
            timeout=15).json()
        for s in (data.get("stats") or []):
            for sp in (s.get("splits") or []):
                st = sp.get("stat") or {}
                pa = st.get("plateAppearances")
                so = st.get("strikeOuts")
                if pa and so is not None and int(pa) >= 500:
                    rate = int(so) / int(pa)
    except Exception as e:
        log.warning("team K rate fetch failed (team %s vs %sHP): %s", team_id, hand, e)
    _team_rate_cache["rates"][key] = rate
    return rate


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
                    "id": team.get("id"),
                    "abbrev": team.get("abbreviation") or team.get("teamName") or "?",
                    "name": team.get("name") or "",
                    "starter_id": pp.get("id"),
                    "starter_name": pp.get("fullName") or "TBD",
                }
            out.append({"game_pk": g.get("gamePk"),
                        "venue": ((g.get("venue") or {}).get("name")),
                        "game_date_utc": g.get("gameDate"),
                        "state": ((g.get("status") or {})
                                  .get("abstractGameState")) or "",
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
    props = odds_api.get_event_props(ev_match.get("id"), "pitcher_strikeouts",
                                     bookmakers=KBOARD_BOOKS)
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
    _ensure_live_model()
    p_league = kmodel.league_k_rate()
    slate = _slate(date)
    events = []
    try:
        events = _events_on(odds_api.get_events(), date)
    except Exception as e:
        log.warning("k board: odds events skipped: %s", e)

    overrides = get_overrides(date)
    # Lineup-post tracker: the one edge the season test quantified (~4% vs
    # stale openers) is lineup INFORMATION. Retro-testing is impossible --
    # nobody archives when a lineup posted -- so we log it forward: the
    # first build that sees a lineup posted snapshots the moment (line,
    # best prices, model read). Every later pre-start build updates the
    # latest_* fields, so the final update ~= the close and the drift
    # after posting is measurable. Granularity = build cadence (~15-30
    # min), stated honestly in the API.
    try:
        with _conn() as _sc:
            _snapped = {r[0] for r in _sc.execute(
                "SELECT starter_id FROM k_lineup_snaps WHERE date=?", (date,))}
    except Exception:
        _snapped = set()
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
            started = (g.get("state") in ("Live", "Final"))
            if not started and g.get("game_date_utc"):
                try:
                    gd = datetime.fromisoformat(
                        g["game_date_utc"].replace("Z", "+00:00"))
                    started = datetime.now(timezone.utc) >= gd
                except (TypeError, ValueError):
                    pass
            if started:
                entry.update({"status": "no read",
                              "why": "game underway — live lines are remaining-K "
                                     "lines; pregame reads already frozen in the log"})
                starters.append(entry)
                continue
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
                entry["lineup_posted"] = bool(order)
                entry["lineup_source"] = "posted"
                if not order:
                    # Tier-2 projection: the opponent's most recent REAL
                    # posted lineup vs this hand (~7-8/9 repeat). Labeled,
                    # never passed off as today's; team-rate remains the
                    # per-slot fallback underneath.
                    proxy = kmodel.fetch_recent_lineup(opp.get("id"), hand)
                    if proxy:
                        order = proxy["batter_ids"]
                        entry["lineup_source"] = f"projected (last vs {hand}HP, {proxy['date']})"
                    else:
                        entry["lineup_source"] = "team avg (no recent lineup found)"
                lineup, known = _build_lineup(order)
                entry["lineup_known_slots"] = known
                kdist = kmodel.k_distribution(
                    lineup, s_rows, hand, p_league,
                    before=None, park_k_factor=_park_k(g.get("venue")),
                    unknown_slot_rate=_team_k_rate(opp.get("id"), hand),
                    start_game_pks=kmodel.fetch_start_games(team["starter_id"]))
                if kdist is None:
                    entry.update({"status": "no read",
                                  "why": "starter sample too thin (house minimums)"})
                    continue
                try:
                    _snap_lineup(date, g.get("game_pk"), entry, _snapped)
                except Exception as _e:
                    log.warning("lineup snap failed for %s: %s", entry.get("starter"), _e)
                ov = overrides.get(team["starter_id"])
                if ov:
                    if ov.get("tbf_cap"):
                        kdist = _apply_tbf_cap(kdist, ov["tbf_cap"])
                        entry["tbf_cap"] = ov["tbf_cap"]
                        entry["pitch_limit"] = ov.get("pitch_limit")
                    entry["excluded"] = bool(ov.get("exclude"))
                    entry["override_note"] = ov.get("note") or None
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
                # ---- forward cage: price each locked arm on the SAME
                # inputs (rows, lineup, park, slot rate, start pks, TBF
                # cap) against the SAME events payload -- identical line
                # and prices, different model. Never touches the entry
                # the public board renders.
                if KCAGE_ON and entry.get("line") is not None:
                    cage = {}
                    coefs = _cage_csw_coefs(date)
                    for arm, kn in CAGE_ARMS:
                        if arm == "cswdelta" and not coefs:
                            continue
                        try:
                            saved_coefs = kmodel.K_CSW_COEFS
                            try:
                                with kseason.knobs(**kn):
                                    if arm == "cswdelta":
                                        kmodel.K_CSW_COEFS = coefs
                                    kd_v = kmodel.k_distribution(
                                        lineup, s_rows, hand, p_league,
                                        before=None, park_k_factor=_park_k(g.get("venue")),
                                        unknown_slot_rate=_team_k_rate(opp.get("id"), hand),
                                        start_game_pks=kmodel.fetch_start_games(team["starter_id"]))
                            finally:
                                kmodel.K_CSW_COEFS = saved_coefs
                            if kd_v is None:
                                continue
                            if ov and ov.get("tbf_cap"):
                                kd_v = _apply_tbf_cap(kd_v, ov["tbf_cap"])
                            priced_v = _price_starter(
                                events, g["teams"]["home"]["name"],
                                g["teams"]["away"]["name"],
                                team["starter_name"], kd_v)
                            if priced_v.get("line") is not None:
                                cage[arm] = priced_v
                        except Exception as _ce:
                            log.warning("cage arm %s failed for %s: %s",
                                        arm, team["starter_name"], _ce)
                    if cage:
                        entry["_cage"] = cage
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


def _breakdown(graded_bets: list[dict]) -> dict:
    """Totals + exclusive-band breakdown of graded paper bets. Every bet
    lives in exactly ONE band and the totals line counts it exactly once
    (no cumulative >=X rows -- Mike's call: a public record should never
    look double-counted). risked = bets at flat 1u. 20+ band isolates the
    reads the market tests exclude as suspect; shown, never counted."""
    def _stats(bets):
        n = len(bets)
        wins = sum(1 for b in bets if b["won"])
        units = round(sum(b["units"] for b in bets), 2)
        roi = round(units / n * 100, 1) if n else None
        return {"bets": n, "wins": wins, "units": units, "roi": roi}

    counted = [b for b in graded_bets if b["ev"] <= EV_LOG_MAX]
    totals = _stats(counted)
    totals["risked"] = totals["bets"]  # flat 1u -- risked units = bet count
    bands = []
    for lo, hi in EV_BANDS:
        # Boundary rule matches the counting policy exactly: the band
        # ending at EV_LOG_MAX includes it (a 20.0% bet is counted), and
        # the open top band is strictly ABOVE the cap (never counted).
        if hi is None:
            members = [b for b in graded_bets if b["ev"] > lo]
        elif hi == EV_LOG_MAX:
            members = [b for b in graded_bets if lo <= b["ev"] <= hi]
        else:
            members = [b for b in graded_bets if lo <= b["ev"] < hi]
        s = _stats(members)
        s["lo"] = lo
        s["hi"] = hi
        s["counted"] = not (hi is None and lo >= EV_LOG_MAX)
        bands.append(s)
    return {"totals": totals, "bands": bands}


def log_details(days: int = 1) -> dict:
    """Graded forward-log detail + stats for the last N days (400 = season)
    -- the recap/record feed. Grades pending rows first (cheap, no odds
    credits), then returns each read with its paper-bet outcomes using the
    same >=EV_LOG_MIN flat-1u convention as the summary, plus window-level
    Brier / lean accuracy / ROI so every consumer shows identical numbers."""
    days = max(1, min(400, days))
    today = parlay.et_date_str(0)
    try:
        _grade_pending(today)
    except Exception as e:
        log.warning("klog grading pass failed: %s", e)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=4)
              - timedelta(days=days)).strftime("%Y-%m-%d")
    with _conn() as c:
        rows = c.execute(
            "SELECT date, name, line, p_over, ev_over, price_over, book_over, "
            "ev_under, price_under, book_under, actual_k, cleared, lineup_posted, "
            "model_ver, excluded, excl_note "
            "FROM k_board_log WHERE date >= ? AND date < ? ORDER BY date, name",
            (cutoff, today)).fetchall()
    out_rows = []
    graded_bets = []
    units = bets = wins = 0
    for (date, name, line, p_over, ev_o, pr_o, bk_o, ev_u, pr_u, bk_u,
         actual, cleared, lineup, mver, excluded, excl_note) in rows:
        mver = mver or "v1"  # rows logged before versioning = launch model
        row = {"date": date, "starter": name, "line": line,
               "p_over": p_over, "actual_k": actual, "cleared": cleared,
               "lineup_posted": bool(lineup), "model": mver,
               "excluded": bool(excluded), "excl_note": excl_note, "bets": []}
        if excluded:
            # priced, shown, permanently on the record -- but not counted.
            # Never deleted: a record you can quietly delete from is worth
            # nothing, so exclusions live in the open as their own line.
            out_rows.append(row)
            continue
        for side, ev, price, book, hit in (
                ("over", ev_o, pr_o, bk_o, cleared),
                ("under", ev_u, pr_u, bk_u,
                 (1 - cleared) if cleared is not None else None)):
            if ev is None or price is None or ev < EV_LOG_MIN:
                continue
            if ev > EV_LOG_MAX:
                # visible in rows + the 20+ breakdown band, never a counted bet
                u = (round(odds_api.american_to_decimal(price) - 1, 2)
                     if (cleared is not None and hit) else
                     (-1.0 if cleared is not None else None))
                row["bets"].append({"side": side, "price": price, "book": book,
                                    "ev": ev, "won": (bool(hit) if cleared is not None else None),
                                    "units": u, "counted": False})
                if cleared is not None:
                    graded_bets.append({"ev": ev, "won": bool(hit), "units": u,
                                        "model": mver})
                continue
            if cleared is None:
                # logged bet, game not graded yet -- keep its identity
                row["bets"].append({"side": side, "price": price, "book": book,
                                    "ev": ev, "won": None, "units": None,
                                    "counted": True})
                continue
            u = round(odds_api.american_to_decimal(price) - 1, 2) if hit else -1.0
            row["bets"].append({"side": side, "price": price, "book": book,
                                "ev": ev, "won": bool(hit), "units": u,
                                "counted": True})
            graded_bets.append({"ev": ev, "won": bool(hit), "units": u,
                                "model": mver})
            bets += 1
            wins += 1 if hit else 0
            units += u
        out_rows.append(row)
    excluded_rows = [r for r in out_rows if r.get("excluded")]
    graded_rows = [r for r in out_rows
                   if r["cleared"] is not None and not r.get("excluded")]
    brier = brier_constant = lean_hits = None
    if graded_rows:
        brier = round(sum((r["p_over"] - r["cleared"]) ** 2
                          for r in graded_rows) / len(graded_rows), 4)
        base = sum(r["cleared"] for r in graded_rows) / len(graded_rows)
        brier_constant = round(sum((base - r["cleared"]) ** 2
                                   for r in graded_rows) / len(graded_rows), 4)
        lean_hits = sum(1 for r in graded_rows
                        if (r["p_over"] >= 0.5) == bool(r["cleared"]))
    # Per-era recaps: each logged read carries the model version that
    # produced it, so records split cleanly instead of silently mixing a
    # retired model's bets with the live one's. History is never
    # recomputed -- just grouped.
    eras = []
    for ver in sorted({r["model"] for r in out_rows}, reverse=True):
        e_rows = [r for r in graded_rows if r["model"] == ver]
        e_bets = [b for b in graded_bets if b["model"] == ver]
        e_brier = (round(sum((r["p_over"] - r["cleared"]) ** 2
                             for r in e_rows) / len(e_rows), 4)
                   if e_rows else None)
        eras.append({"model": ver,
                     "label": K_MODEL_ERA_LABELS.get(ver, f"model {ver}"),
                     "live": ver == K_MODEL_VER,
                     "graded": len(e_rows), "brier": e_brier,
                     "breakdown": _breakdown(e_bets)})
    return {"days": days, "rows": out_rows,
            "excluded": len(excluded_rows),
            "graded": len(graded_rows),
            "pending": sum(1 for r in out_rows if r["cleared"] is None),
            "bets": bets, "wins": wins, "units": round(units, 2),
            "roi": round(units / bets * 100, 1) if bets else None,
            "brier": brier, "brier_constant": brier_constant,
            "lean_hits": lean_hits,
            "breakdown": _breakdown(graded_bets),
            "eras": eras,
            "overall": _result_log_summary()}


def validation_summary() -> dict:
    """The model's credentials: the newest stored K market test per window
    (backtested vs real historical closing lines) plus the latest CALIBRATED
    backtest -- read straight from the Lab's tables so Discord shows exactly
    what the site shows. Clearly historical; the forward log is the live test."""
    market = []
    backtest = None
    with _conn() as c:
        seen_days = set()
        for ts, days, report in c.execute(
                "SELECT ts, days, report FROM k_market_runs ORDER BY ts DESC"):
            if days in seen_days:
                continue
            seen_days.add(days)
            rep = json.loads(report)
            thr = (rep.get("by_threshold") or {}).get("2") or {}
            market.append({"days": days, "ts": ts,
                           "games": rep.get("games_priced"),
                           "bets": thr.get("bets"), "wins": thr.get("wins"),
                           "units": thr.get("units"), "roi_pct": thr.get("roi_pct")})
        for ts, days, config, report in c.execute(
                "SELECT ts, days, config, report FROM k_backtest_runs ORDER BY ts DESC"):
            cfg = json.loads(config) if config else {}
            if not cfg.get("k_calib_weight"):
                continue
            rep = json.loads(report)
            if not rep.get("n"):
                continue
            beats = (rep.get("brier_model") is not None
                     and rep["brier_model"] < (rep.get("brier_constant") or 1)
                     and (rep.get("brier_naive") is None
                          or rep["brier_model"] < rep["brier_naive"]))
            backtest = {"days": days, "ts": ts, "n": rep["n"],
                        "brier_model": rep.get("brier_model"),
                        "brier_constant": rep.get("brier_constant"),
                        "brier_naive": rep.get("brier_naive"), "beats": beats}
            break
    market.sort(key=lambda m: -(m["days"] or 0))
    return {"market_tests": market[:4], "backtest": backtest}


# ---------- K Sim: run any lineup through the exact validated model ----------

_players_cache = {"date": None, "players": []}


def players_list() -> list[dict]:
    """All active MLB players (one schedule-API call, cached per day) --
    feeds the sim's name autocomplete and id->name display."""
    date = parlay.et_date_str(0)
    if _players_cache["date"] == date and _players_cache["players"]:
        return _players_cache["players"]
    try:
        season = date[:4]
        data = requests.get(f"{MLB_BASE}/sports/1/players",
                            params={"season": season}, timeout=30).json()
        players = [{"id": p["id"], "name": p.get("fullName", "")}
                   for p in data.get("people", []) if p.get("id")]
        if players:
            _players_cache.update({"date": date, "players": players})
    except Exception as e:
        log.warning("k sim: players fetch failed: %s", e)
    return _players_cache["players"]


def projected_lineups(offset: int = 0) -> dict:
    """Every team on the slate with the nine hitters they'll send up.

    Nothing new is computed here -- the board already resolves both of
    these on every build; this just surfaces them in one place so they can
    be read at a glance and loaded into the sim in a click:
      posted    = today's REAL batting order from the boxscore
      projected = that team's most recent real order vs this starter hand
                  (kmodel.fetch_recent_lineup -- what the board prices
                  with until lineups drop), stamped with the date it came
                  from so it is never passed off as today's
      none      = no order posted and no recent match found

    Read-only. No odds credits, no model calls, no effect on pricing or
    on the frozen forward log."""
    offset = 1 if offset == 1 else 0
    date = parlay.et_date_str(offset)
    names = {p["id"]: p["name"] for p in players_list()}
    games = []
    for g in _slate(date):
        orders = _lineup_order(g["game_pk"])
        orders = orders if isinstance(orders, dict) else {}
        entry = {"game_pk": g.get("game_pk"), "venue": g.get("venue"),
                 "state": g.get("state"), "teams": []}
        for side, opp_side in (("home", "away"), ("away", "home")):
            bat = g["teams"][side]      # the hitting team
            opp = g["teams"][opp_side]  # whose starter they face
            hand = None
            if opp.get("starter_id"):
                try:
                    hand = parlay.get_starter_hand(opp["starter_id"])
                except Exception:
                    hand = None
            order = list((orders.get(side) or [])[:9])
            source, from_date = "posted", None
            if not order:
                source = "none"
                if hand in ("L", "R") and bat.get("id"):
                    proxy = kmodel.fetch_recent_lineup(bat["id"], hand)
                    if proxy:
                        order = list(proxy["batter_ids"])[:9]
                        source, from_date = "projected", proxy.get("date")
            batters = []
            for pid in order:
                try:
                    pid = int(pid)
                except (TypeError, ValueError):
                    continue
                batters.append({"id": pid, "name": names.get(pid, str(pid))})
            entry["teams"].append({
                "team": bat.get("abbrev"), "team_id": bat.get("id"),
                "vs_starter": opp.get("starter_name"),
                "vs_starter_id": opp.get("starter_id"),
                "vs_hand": hand,
                "source": source, "from_date": from_date,
                "batters": batters,
            })
        games.append(entry)
    posted = sum(1 for g in games for t in g["teams"] if t["source"] == "posted")
    return {"date": date, "offset": offset,
            "posted": posted, "teams_total": sum(len(g["teams"]) for g in games),
            "games": games,
            "note": ("posted = today's real order from the boxscore; "
                     "projected = that team's most recent real order vs this "
                     "hand (the same one the K Board prices with until "
                     "lineups drop), stamped with the date it came from")}


def sim_lineup(starter_id: int, batter_ids: list, offset: int = 0,
               tbf_override: int | None = None,
               pitch_limit: int | None = None) -> dict:
    """What-if: this starter vs an arbitrary 9-man order. Same
    k_distribution the backtests validated -- no separate sim math. Blank
    slots price at league exactly like an unposted lineup. Market compare
    uses the cached board's prices only (no odds credits); the EVs are
    recomputed for the SIMMED probabilities at those prices."""
    offset = 1 if offset == 1 else 0
    _ensure_live_model()
    date = parlay.et_date_str(offset)
    names = {p["id"]: p["name"] for p in players_list()}
    try:
        hand = parlay.get_starter_hand(starter_id)
    except Exception:
        hand = None
    if hand not in ("L", "R"):
        return {"error": "starter handedness unavailable"}
    try:
        s_rows = parlay.get_player_season_rows(starter_id, True)
    except Exception:
        s_rows = []
    ppa = None
    if pitch_limit and not tbf_override:
        # Same conversion the Admin override uses: HIS pitches-per-PA, not
        # a league constant -- 90 pitches is a different outing for a 3.5
        # P/PA strike-thrower than a 4.3 grinder.
        ppa = pitches_per_pa(s_rows)
        if ppa:
            tbf_override = max(3, min(45, int(round(pitch_limit / ppa))))
    slate_entry = None
    for g in _slate(date):
        for side in ("home", "away"):
            if g["teams"][side]["starter_id"] == starter_id:
                opp_side = "away" if side == "home" else "home"
                slate_entry = {"game_pk": g["game_pk"], "venue": g.get("venue"),
                               "team": g["teams"][side]["abbrev"],
                               "opp": g["teams"][opp_side]["abbrev"],
                               "opp_id": g["teams"][opp_side].get("id"),
                               "opp_side": opp_side}
    lineup_basis = None
    if not batter_ids and slate_entry:
        # No lineup given -> the board's own resolution order, so a bare
        # sim faces the REAL opponent instead of a league-average one:
        # posted order first, else the opponent's most recent real order
        # vs this hand (the same proxy the board prices with), else fall
        # through to league slots exactly as before.
        try:
            orders = _lineup_order(slate_entry["game_pk"])
            posted = list(((orders or {}).get(slate_entry["opp_side"]) or [])[:9])
        except Exception:
            posted = []
        if posted:
            batter_ids = posted
            lineup_basis = "posted lineup"
        else:
            try:
                proxy = kmodel.fetch_recent_lineup(slate_entry.get("opp_id"), hand)
            except Exception:
                proxy = None
            if proxy:
                batter_ids = proxy["batter_ids"]
                lineup_basis = f"projected (last vs {hand}HP, {proxy['date']})"
    lineup = []
    for pid in (batter_ids or [])[:9]:
        if not pid:
            lineup.append(None)
            continue
        try:
            rows = parlay.get_player_season_rows(int(pid), False)
        except Exception:
            lineup.append(None)
            continue
        side = _majority_side(rows)
        lineup.append({"rows": rows, "side": side,
                       "name": names.get(int(pid), str(pid))} if side else None)
    while len(lineup) < 9:
        lineup.append(None)
    kdist = kmodel.k_distribution(
        lineup, s_rows, hand, kmodel.league_k_rate(),
        before=None, park_k_factor=_park_k((slate_entry or {}).get("venue")),
        start_game_pks=kmodel.fetch_start_games(starter_id))
    if kdist is None:
        return {"error": "model refuses this start (starter sample under house minimums)"}
    tbf_mode = "workload mixture (his real start logs)"
    if tbf_override:
        # Fixed-TBF what-if (pitch limits, piggybacks, deep-leash days).
        # Rebuilt from kmodel's OWN exported pieces -- the per-slot K probs
        # the model just computed, its slot-PA arithmetic, its exact
        # Poisson-binomial -- so a fixed-TBF sim can never drift from the
        # validated math; only the workload assumption changes.
        tbf = max(9, min(45, int(tbf_override)))
        slot_probs = [s["p_k_per_pa"] for s in kdist["inputs"]["slots"]]
        counts = kmodel.slot_pa_counts(tbf)
        seq = [slot_probs[i] for i in range(9) for _ in range(counts[i])]
        pb = kmodel.poisson_binomial(seq)
        kdist = dict(kdist)
        kdist["dist"] = [round(m, 6) for m in pb]
        kdist["mean_k"] = round(sum(k * m for k, m in enumerate(pb)), 3)
        kdist["tbf_mean"] = float(tbf)
        tbf_mode = f"FIXED at {tbf} TBF (override)"
    ladder = []
    for half in range(3, 8):
        line = half + 0.5
        pr = kmodel.price_line(kdist, line)
        ladder.append({"line": line, "p_over": pr["p_over"]})
    market = None
    with _lock:
        cached = (_boards.get(date) or {}).get("data")
    if cached:
        for s in cached.get("starters", []):
            if s.get("starter_id") == starter_id and s.get("line") is not None:
                pr = kmodel.price_line(kdist, s["line"])
                market = {"line": s["line"], "p_over": pr["p_over"],
                          "board_p_over": s.get("p_over"),
                          "over": s.get("over"), "under": s.get("under"),
                          "ev_skipped": s.get("ev_skipped")}
                if not s.get("ev_skipped"):
                    for side, p in (("over", pr["p_over"]), ("under", pr["p_under"])):
                        bk = s.get(side)
                        if bk:
                            market["ev_" + side] = round(
                                (p * odds_api.american_to_decimal(bk["price"]) - 1) * 100, 1)
                break
    return {"date": date, "starter": names.get(starter_id, str(starter_id)),
            "hand": hand, "slate": slate_entry,
            "basis": lineup_basis or ("custom lineup" if batter_ids
                                      else "league-average slots"),
            "pitch_limit": pitch_limit,
            "ppa": round(ppa, 2) if ppa else None,
            "tbf_cap": tbf_override,
            "mean_k": kdist["mean_k"], "tbf_mean": kdist["tbf_mean"],
            "fair_line": _fair_line(kdist),
            "league_fallback_slots": kdist["inputs"]["league_fallback_slots"],
            "park_k_factor": kdist["inputs"]["park_k_factor"],
            "slots": kdist["inputs"]["slots"],
            "tbf_mode": tbf_mode,
            "ladder": ladder, "market": market}


def log_csv(days: int = 400) -> str:
    """The forward log as a spreadsheet: one row per logged paper bet
    (side, price, book, EV, result, units) plus no-bet reads (side
    'no-bet') so lean accuracy is analyzable too. Raw material for Mike's
    own threshold/side analysis -- same frozen data, zero new collection."""
    d = log_details(days)
    lines = ["date,starter,line,side,price,book,ev_pct,model_p_over,actual_k,result,units,lineup_posted"]
    def esc(x):
        s = "" if x is None else str(x)
        return f'"{s}"' if "," in s else s
    for r in d["rows"]:
        base = [r["date"], esc(r["starter"]), r["line"]]
        if r["bets"]:
            for b in r["bets"]:
                res = "" if b["won"] is None else ("win" if b["won"] else "loss")
                lines.append(",".join(str(x) for x in base + [
                    b["side"], b["price"], esc(b["book"]), b["ev"], r["p_over"],
                    r["actual_k"] if r["actual_k"] is not None else "",
                    res, b["units"] if b["units"] is not None else "",
                    1 if r["lineup_posted"] else 0]))
        else:
            res = "" if r["cleared"] is None else (
                "lean-hit" if (r["p_over"] >= 0.5) == bool(r["cleared"]) else "lean-miss")
            lines.append(",".join(str(x) for x in base + [
                "no-bet", "", "", "", r["p_over"],
                r["actual_k"] if r["actual_k"] is not None else "",
                res, "", 1 if r["lineup_posted"] else 0]))
    return "\n".join(lines) + "\n"


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
