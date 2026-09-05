"""
Outs Lab -- does "outs = Binomial(batters faced, r)" describe real starts?

Zero odds credits. Zero model. Every number here is a count from MLB
StatsAPI play-by-play, so the report can be checked line by line.

The question that started it: what is P(15 outs within the first n
batters faced)? Boxscore totals can't answer that -- when a starter
finishes with 21 BF and 15 outs you don't know WHEN the 15th out came,
and conditioning on final TBF is polluted by the hook (pitchers pulled at
17 BF are pulled because it went badly). So this reads the plays: for
every start, the running outs total after batter 1, 2, 3, ... and asks,
among starts that faced at least n batters, how often outs-after-n >= 15.
That is the closed form's exact claim, measured directly.

Alongside it, the two rates the argument turned on:
  retire rate   = 1 - reach rate   (batter did not reach base)
  outs per BF   = total outs / TBF (includes DP / CS / pickoffs)
and whether the gap between them depends on how many runners are on --
if extra outs come from traffic, a flat per-batter rate can't be right.

What it does NOT do: touch lines, price anything, or import from the
props engine. Standalone on purpose -- nothing in pprops can break it.

Storage: outs_lab_runs in the volume DB. Report JSON per run.
"""
import os
import json
import math
import time
import sqlite3
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger("outs_lab")

DB_PATH = os.getenv("DB_PATH", "matchups.db")
MLB_BASE = "https://statsapi.mlb.com/api/v1"
WORKERS = int(os.getenv("OUTS_LAB_WORKERS", "4"))
TARGET_OUTS = 15                    # the 14.5 line (headline row)
OUTS_LINES = (12.5, 14.5, 15.5, 17.5, 18.5, 20.5)   # every line books post
N_RANGE = tuple(range(12, 30))      # batters-faced checkpoints
FOCUS = (17, 18, 19, 20, 21, 22, 23)   # Mike's asked-for band
# Every stat the grid answers, with the line ladders books actually post.
STATS = {
    "outs":  {"lines": OUTS_LINES, "label": "Outs"},
    "hits":  {"lines": (2.5, 3.5, 4.5, 5.5, 6.5, 7.5), "label": "Hits allowed"},
    "walks": {"lines": (0.5, 1.5, 2.5, 3.5), "label": "Walks"},
    "ks":    {"lines": (3.5, 4.5, 5.5, 6.5, 7.5, 8.5), "label": "Strikeouts"},
}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
WALK_EVENTS = {"walk", "intent_walk"}
K_EVENTS = {"strikeout", "strikeout_double_play", "strikeout_triple_play"}

# Trim the play-by-play payload to what we read. StatsAPI's `fields`
# filter is a flat list of field names applied at any depth.
PBP_FIELDS = ",".join([
    "allPlays", "result", "eventType", "type", "about", "inning",
    "halfInning", "isComplete", "count", "outs", "matchup", "pitcher",
    "batter", "id", "postOnFirst", "postOnSecond", "postOnThird",
])

# Plate-appearance-ending events. Anything in RUNNER_EVENTS is a
# mid-PA runner play (the batter's PA continues or the inning ends on the
# bases). Anything in neither set is counted as a PA AND tallied in the
# report under unknown_events so a new StatsAPI code can't hide.
REACH_EVENTS = {
    "single", "double", "triple", "home_run", "walk", "intent_walk",
    "hit_by_pitch", "field_error", "catcher_interf", "fielders_choice",
}
OUT_EVENTS = {
    "strikeout", "strikeout_double_play", "strikeout_triple_play",
    "field_out", "force_out", "grounded_into_double_play",
    "grounded_into_triple_play", "double_play", "triple_play",
    "sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
    "fielders_choice_out", "batter_out",
}
RUNNER_EVENTS = {
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_1b", "pickoff_2b", "pickoff_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home", "pickoff_error_1b", "pickoff_error_2b",
    "pickoff_error_3b", "stolen_base_2b", "stolen_base_3b",
    "stolen_base_home", "wild_pitch", "passed_ball", "balk",
    "other_advance", "runner_double_play", "defensive_indiff",
    "runner_placed", "game_advisory", "batter_timeout", "mound_visit",
    "no_pitch", "pitching_substitution", "offensive_substitution",
    "defensive_substitution", "defensive_switch", "umpire_substitution",
    "injury", "ejection", "at_bat_start", "pitch_challenge",
}

_state = {"running": False, "progress": "", "started": 0.0, "error": ""}
_lock = threading.Lock()
_session = requests.Session()


# ------------------------------------------------------------------ store
def _conn():
    conn = sqlite3.connect(DB_PATH)
    # one row per starter per game -- the dataset. Re-running a season
    # skips games already here, so a crashed run resumes for free.
    conn.execute("""CREATE TABLE IF NOT EXISTS outs_lab_starts (
        game_pk INTEGER, half TEXT, season INTEGER, date TEXT,
        pitcher_id INTEGER, tbf INTEGER, outs INTEGER, extra_outs INTEGER,
        rec TEXT, PRIMARY KEY (game_pk, half))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS outs_lab_games (
        game_pk INTEGER PRIMARY KEY, season INTEGER, date TEXT, starts INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS outs_lab_runs (
        ts REAL PRIMARY KEY, label TEXT, report TEXT)""")
    return conn


def history(limit: int = 8) -> list[dict]:
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT ts, label, report FROM outs_lab_runs "
                "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "label": r[1], "report": json.loads(r[2])}
                for r in rows]
    except Exception as e:
        log.warning("outs lab history failed: %s", e)
        return []


def coverage() -> list[dict]:
    """What's in the dataset, per season -- the 'how many examples' answer."""
    try:
        with _conn() as c:
            rows = c.execute("""SELECT season, COUNT(*), SUM(starts),
                MIN(date), MAX(date) FROM outs_lab_games
                GROUP BY season ORDER BY season""").fetchall()
        return [{"season": r[0], "games": r[1], "starts": r[2] or 0,
                 "first": r[3], "last": r[4]} for r in rows]
    except Exception as e:
        log.warning("outs lab coverage failed: %s", e)
        return []


def _done_games(season: int) -> set:
    with _conn() as c:
        return {r[0] for r in c.execute(
            "SELECT game_pk FROM outs_lab_games WHERE season=?", (season,))}


def _save_game(season: int, g: dict, seqs: list[dict]):
    with _conn() as c:
        for rec in seqs:
            c.execute("INSERT OR REPLACE INTO outs_lab_starts VALUES (?,?,?,?,?,?,?,?,?)",
                      (g["gamePk"], rec["half"], season, g["date"],
                       rec["pitcher_id"], rec["tbf"], rec["outs"],
                       rec["extra_outs"], json.dumps({
                           "pa": rec["pa"], "cum": rec["cum"],
                           "cum_hits": rec["cum_hits"],
                           "cum_walks": rec["cum_walks"],
                           "cum_ks": rec["cum_ks"],
                           "unknown": rec["unknown"],
                           "runner_outs": rec["runner_outs"]})))
        c.execute("INSERT OR REPLACE INTO outs_lab_games VALUES (?,?,?,?)",
                  (g["gamePk"], season, g["date"], len(seqs)))


def load_starts(seasons: list[int]) -> list[dict]:
    if not seasons:
        return []
    q = ",".join("?" * len(seasons))
    with _conn() as c:
        rows = c.execute(f"""SELECT game_pk, half, season, date, pitcher_id,
            tbf, outs, extra_outs, rec FROM outs_lab_starts
            WHERE season IN ({q})""", seasons).fetchall()
    out = []
    for r in rows:
        rec = json.loads(r[8])
        rec.update({"game_pk": r[0], "half": r[1], "season": r[2],
                    "date": r[3], "pitcher_id": r[4], "tbf": r[5],
                    "outs": r[6], "extra_outs": r[7]})
        out.append(rec)
    return out


def state() -> dict:
    with _lock:
        s = dict(_state)
    if s["running"]:
        s["elapsed_s"] = round(time.time() - s["started"])
    return s


def _progress(msg: str):
    with _lock:
        _state["progress"] = msg


# ----------------------------------------------------------------- fetch
def final_games(start: str, end: str) -> list[dict]:
    """Every regular-season Final between two dates, one schedule call."""
    r = _session.get(f"{MLB_BASE}/schedule", params={
        "sportId": 1, "startDate": start, "endDate": end,
        "gameType": "R"}, timeout=30)
    r.raise_for_status()
    out = []
    for day in r.json().get("dates") or []:
        for g in day.get("games") or []:
            st = (g.get("status") or {})
            if st.get("codedGameState") == "F" or \
               (st.get("abstractGameState") == "Final"
                    and "Completed" in (st.get("detailedState") or "Final")):
                out.append({"gamePk": g["gamePk"], "date": day.get("date")})
    return out


def fetch_plays(game_pk: int) -> list[dict]:
    """allPlays for one game, trimmed. Falls back to the full payload if
    the fields filter ever returns an unexpected shape."""
    url = f"{MLB_BASE}/game/{game_pk}/playByPlay"
    r = _session.get(url, params={"fields": PBP_FIELDS}, timeout=30)
    r.raise_for_status()
    plays = r.json().get("allPlays")
    if plays is None:
        r = _session.get(url, timeout=30)
        r.raise_for_status()
        plays = r.json().get("allPlays") or []
    return plays


# ----------------------------------------------------------------- parse
def _runners_after(m: dict) -> int:
    return sum(1 for k in ("postOnFirst", "postOnSecond", "postOnThird")
               if m.get(k))


def starter_sequences(plays: list[dict]) -> list[dict]:
    """Both starters' batter-by-batter records from one game's allPlays.

    Each start -> {
      pitcher_id, half,
      pa: [ {reached, out_delta, runners_before, extra} ... ]  # per PA
      cum: [outs after batter 1, after 2, ...],
      tbf, outs, extra_outs, runner_outs, unknown: {eventType: n}
    }
    A play is one plate appearance unless its eventType is a runner
    event, in which case its outs are credited to the CURRENT batter's
    row as an extra out (no new BF). The starter is whoever pitched the
    first play of each half.
    """
    starts = {}
    for half in ("top", "bottom"):
        # first play of this half names the starter
        first = next((p for p in plays
                      if (p.get("about") or {}).get("halfInning") == half),
                     None)
        if not first:
            continue
        pid = ((first.get("matchup") or {}).get("pitcher") or {}).get("id")
        if not pid:
            continue
        rec = {"pitcher_id": pid, "half": half, "pa": [], "cum": [],
               "cum_hits": [], "cum_walks": [], "cum_ks": [],
               "unknown": {}, "runner_outs": 0}
        n_h = n_bb = n_k = 0
        cur_inning = None
        runners_before = 0
        prev_cum = 0
        pending_extra = 0        # runner outs before the next PA resolves
        for p in plays:
            ab = p.get("about") or {}
            if ab.get("halfInning") != half:
                continue
            m = p.get("matchup") or {}
            if (m.get("pitcher") or {}).get("id") != pid:
                break             # starter is gone; plays are chronological
            inning = ab.get("inning") or 1
            if inning != cur_inning:
                cur_inning = inning
                runners_before = 0
            outs_after_play = (p.get("count") or {}).get("outs")
            if outs_after_play is None:
                continue
            cum = 3 * (inning - 1) + int(outs_after_play)
            delta = max(0, cum - prev_cum)
            ev = ((p.get("result") or {}).get("eventType") or "").lower()
            if ev in RUNNER_EVENTS:
                # mid-PA runner play: outs (if any) are extra outs
                rec["runner_outs"] += delta
                pending_extra += delta
            else:
                if ev in REACH_EVENTS:
                    reached = True
                elif ev in OUT_EVENTS:
                    reached = False
                else:
                    # unknown code: infer from outs, but count it
                    rec["unknown"][ev or "(none)"] = \
                        rec["unknown"].get(ev or "(none)", 0) + 1
                    reached = delta == 0
                batter_out = 0 if reached else 1
                extra = pending_extra + max(0, delta - batter_out)
                rec["pa"].append({"reached": reached, "out_delta": delta,
                                  "runners_before": runners_before,
                                  "extra": extra})
                rec["cum"].append(cum)
                n_h += ev in HIT_EVENTS
                n_bb += ev in WALK_EVENTS
                n_k += ev in K_EVENTS
                rec["cum_hits"].append(n_h)
                rec["cum_walks"].append(n_bb)
                rec["cum_ks"].append(n_k)
                pending_extra = 0
            prev_cum = cum
            runners_before = _runners_after(m)
        if rec["pa"]:
            rec["tbf"] = len(rec["pa"])
            rec["outs"] = rec["cum"][-1] + pending_extra
            rec["extra_outs"] = sum(x["extra"] for x in rec["pa"])
            starts[half] = rec
    return list(starts.values())


# ------------------------------------------------------------- statistics
def binom_tail(n: int, k: int, r: float) -> float:
    """P(X >= k), X ~ Binomial(n, r)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) * r ** i * (1 - r) ** (n - i)
               for i in range(k, n + 1))


def american(p: float) -> str:
    if p <= 0 or p >= 1:
        return "n/a"
    return f"+{round(100 * (1 - p) / p)}" if p < .5 \
        else f"-{round(100 * p / (1 - p))}"


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (round(ctr - half, 4), round(ctr + half, 4))


def build_report(starts: list[dict], label: str, meta: dict) -> dict:
    """All the tables, from parsed starts only. Pure -- testable offline."""
    n_starts = len(starts)
    tbf_tot = sum(s["tbf"] for s in starts)
    outs_tot = sum(s["outs"] for s in starts)
    reached_tot = sum(1 for s in starts for x in s["pa"] if x["reached"])
    extra_tot = sum(s["extra_outs"] for s in starts)
    unknown: dict = {}
    for s in starts:
        for k, v in s["unknown"].items():
            unknown[k] = unknown.get(k, 0) + v

    r_outs = outs_tot / tbf_tot if tbf_tot else 0.0
    r_retire = 1 - reached_tot / tbf_tot if tbf_tot else 0.0

    # ---- the ladder: outs after first n batters, among starts with TBF>=n
    ladder = []
    for n in N_RANGE:
        elig = [s for s in starts if s["tbf"] >= n]
        if not elig:
            continue
        after = [s["cum"][n - 1] for s in elig]
        hits = sum(1 for a in after if a >= TARGET_OUTS)
        k = len(elig)
        emp = hits / k
        mean_after = sum(after) / k
        var_after = (sum((a - mean_after) ** 2 for a in after) / (k - 1)
                     if k > 1 else 0.0)
        binom_var = n * r_outs * (1 - r_outs)
        lo, hi = _wilson(hits, k)
        ladder.append({
            "n": n, "starts": k, "hit": hits,
            "empirical": round(emp, 4), "ci95": [lo, hi],
            "fair": american(emp),
            "pred_outs_rate": round(binom_tail(n, TARGET_OUTS, r_outs), 4),
            "pred_retire_rate": round(binom_tail(n, TARGET_OUTS, r_retire), 4),
            "mean_outs_after": round(mean_after, 3),
            "expected_outs_after": round(n * r_outs, 3),
            "dispersion": round(var_after / binom_var, 3) if binom_var else None,
            "focus": n in FOCUS,
        })

    # ---- conversion grids: P(stat > line | first n batters), per stat.
    # The lookup table an outs model reads: project TBF, read the price.
    # League-wide shape; a pitcher's own rate slots in later. Binomial
    # with the pooled per-BF rate sits under each cell as the iid check.
    cum_key = {"outs": "cum", "hits": "cum_hits", "walks": "cum_walks",
               "ks": "cum_ks"}
    rates = {}
    for st, key in cum_key.items():
        tot = sum(s[key][-1] for s in starts if s.get(key))
        rates[st] = round(tot / tbf_tot, 4) if tbf_tot else 0.0
    grids = {}
    for st, spec in STATS.items():
        key = cum_key[st]
        rows = []
        for n in N_RANGE:
            elig = [s for s in starts if s["tbf"] >= n and s.get(key)]
            if len(elig) < 20:
                continue
            after = [s[key][n - 1] for s in elig]
            row = {"n": n, "starts": len(elig)}
            for line in spec["lines"]:
                k = math.ceil(line)
                hits = sum(1 for a in after if a >= k)
                row[str(line)] = round(hits / len(elig), 4)
                row[f"binom_{line}"] = round(binom_tail(n, k, rates[st]), 4)
            rows.append(row)
        grids[st] = {"label": spec["label"], "lines": list(spec["lines"]),
                     "per_bf_rate": rates[st], "rows": rows}
    grid = grids["outs"]["rows"]

    # ---- selection view: condition on FINAL tbf == n (the hook-polluted one)
    final_view = []
    for n in N_RANGE:
        sel = [s for s in starts if s["tbf"] == n]
        if len(sel) < 10:
            continue
        hits = sum(1 for s in sel if s["outs"] >= TARGET_OUTS)
        final_view.append({"n": n, "starts": len(sel),
                           "p15": round(hits / len(sel), 4)})

    # ---- extra outs by base state
    by_state = {}
    for s in starts:
        for x in s["pa"]:
            b = by_state.setdefault(x["runners_before"],
                                    {"pa": 0, "extra": 0, "reached": 0})
            b["pa"] += 1
            b["extra"] += x["extra"]
            b["reached"] += int(x["reached"])
    base_state = [{"runners_on": k, "pa": v["pa"],
                   "extra_outs_per_pa": round(v["extra"] / v["pa"], 4),
                   "reach_rate": round(v["reached"] / v["pa"], 4)}
                  for k, v in sorted(by_state.items())]

    # ---- outs/BF by traffic in the start (baserunners per 9 BF buckets)
    traffic = {}
    for s in starts:
        if s["tbf"] < 12:
            continue
        rb = sum(1 for x in s["pa"] if x["reached"]) / s["tbf"]
        key = "<0.25" if rb < .25 else "0.25-0.30" if rb < .30 \
            else "0.30-0.35" if rb < .35 else "0.35-0.40" if rb < .40 else ">=0.40"
        t = traffic.setdefault(key, {"starts": 0, "tbf": 0, "outs": 0,
                                     "extra": 0, "reached": 0})
        t["starts"] += 1
        t["tbf"] += s["tbf"]
        t["outs"] += s["outs"]
        t["extra"] += s["extra_outs"]
        t["reached"] += sum(1 for x in s["pa"] if x["reached"])
    order = ["<0.25", "0.25-0.30", "0.30-0.35", "0.35-0.40", ">=0.40"]
    traffic_rows = [{"reach_rate_bucket": k, "starts": traffic[k]["starts"],
                     "outs_per_bf": round(traffic[k]["outs"] / traffic[k]["tbf"], 4),
                     "retire_rate": round(1 - traffic[k]["reached"] / traffic[k]["tbf"], 4),
                     "extra_outs_per_bf": round(traffic[k]["extra"] / traffic[k]["tbf"], 4)}
                    for k in order if k in traffic]

    # ---- times through the order
    tto = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    for s in starts:
        for i, x in enumerate(s["pa"]):
            t = min(3, i // 9 + 1)
            tto[t][0] += 1
            tto[t][1] += int(x["reached"])
    tto_rows = [{"tto": t, "pa": v[0],
                 "reach_rate": round(v[1] / v[0], 4) if v[0] else None}
                for t, v in tto.items()]

    focus17 = next((row for row in ladder if row["n"] == 17), None)
    return {
        "label": label, "meta": meta,
        "starts": n_starts, "tbf": tbf_tot, "outs": outs_tot,
        "reached": reached_tot, "extra_outs": extra_tot,
        "outs_per_bf": round(r_outs, 4),
        "retire_rate": round(r_retire, 4),
        "extra_outs_per_bf": round(extra_tot / tbf_tot, 4) if tbf_tot else None,
        "ladder": ladder,
        "grid": grid, "grid_lines": list(OUTS_LINES),
        "grids": grids, "per_bf_rates": rates,
        "seasons": sorted({s.get("season") for s in starts if s.get("season")}),
        "final_tbf_view": final_view,
        "base_state": base_state,
        "traffic": traffic_rows,
        "tto": tto_rows,
        "unknown_events": unknown,
        "headline": {
            "question": "P(15 outs within first 17 batters faced)",
            "empirical": focus17["empirical"] if focus17 else None,
            "fair": focus17["fair"] if focus17 else None,
            "starts": focus17["starts"] if focus17 else None,
            "closed_form_outs_rate": focus17["pred_outs_rate"] if focus17 else None,
            "closed_form_retire_rate": focus17["pred_retire_rate"] if focus17 else None,
        },
    }


# ------------------------------------------------------------------- run
def fetch_season(season: int, progress=None) -> dict:
    """Pull every final regular-season game of one season into the
    dataset. Skips games already stored -- resumable, idempotent."""
    progress = progress or _progress
    start, end = season_bounds(season)
    progress(f"{season}: schedule {start}..{end}")
    games = final_games(start, end)
    done = _done_games(season)
    todo = [g for g in games if g["gamePk"] not in done]
    total = len(todo)
    if not games:
        return {"error": f"no final regular-season games for {season}"}
    if not todo:
        return {"season": season, "games": len(games), "fetched": 0,
                "skipped": len(done), "note": "already complete"}
    n_done = errors = n_starts = 0
    t0 = time.time()

    def _one(g):
        return g, starter_sequences(fetch_plays(g["gamePk"]))

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_one, g) for g in todo]
        for f in as_completed(futs):
            n_done += 1
            try:
                g, seqs = f.result()
                _save_game(season, g, seqs)
                n_starts += len(seqs)
            except Exception as e:
                errors += 1
                log.warning("outs lab: game failed: %s", e)
            if n_done % 10 == 0 or n_done == total:
                el = time.time() - t0
                eta = (total - n_done) * el / n_done if n_done else 0
                progress(f"{season}: game {n_done}/{total} — {n_starts} starts, "
                         f"{errors} errors, ~{int(eta)}s left")
    return {"season": season, "games": len(games), "fetched": total,
            "skipped": len(done), "starts": n_starts, "errors": errors,
            "seconds": round(time.time() - t0)}


def report_for(seasons: list[int]) -> dict:
    """Build the full report from stored starts for any season set."""
    starts = load_starts(seasons)
    if not starts:
        return {"error": f"no stored starts for {seasons} -- fetch first"}
    label = ", ".join(str(x) for x in sorted(seasons))
    return build_report(starts, label,
                        {"seasons": sorted(seasons), "starts": len(starts)})


def season_bounds(season: int) -> tuple[str, str]:
    """Wide net -- the schedule call filters to gameType R + Final anyway."""
    this_year = datetime.now(timezone.utc).year
    end = f"{season}-10-05"
    if season >= this_year:
        end = (datetime.now(timezone.utc) - timedelta(hours=4)
               ).strftime("%Y-%m-%d")
    return f"{season}-03-15", end


def start_async(seasons: list[int]) -> dict:
    """Fetch each season in turn (resumable), then store one pooled report
    for the whole set so the page has something to show without a query."""
    seasons = sorted({int(x) for x in seasons if x})
    if not seasons:
        return {"started": False, "reason": "no seasons given"}
    with _lock:
        if _state["running"]:
            return {"started": False, "reason": "already running",
                    "progress": _state["progress"]}
        _state.update({"running": True, "progress": "starting…",
                       "started": time.time(), "error": ""})

    def _work():
        summary = []
        try:
            for yr in seasons:
                summary.append(fetch_season(yr))
            _progress("building report…")
            rep = report_for(seasons)
            rep["fetch"] = summary
            with _conn() as c:
                c.execute("INSERT INTO outs_lab_runs VALUES (?,?,?)",
                          (time.time(), rep.get("label", ""), json.dumps(rep)))
        except Exception as e:
            log.exception("outs lab run failed")
            with _lock:
                _state["error"] = f"{type(e).__name__}: {e}"
        finally:
            with _lock:
                _state["running"] = False
                _state["progress"] = "done"
    threading.Thread(target=_work, daemon=True).start()
    return {"started": True, "seasons": seasons}


# ------------------------------------------------------------------ routes
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Outs Lab</title>
<style>
body{background:#0f1115;color:#e6e6e6;font:14px/1.4 -apple-system,system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 4px}h2{font-size:14px;margin:18px 0 6px;color:#9aa}
.sub{color:#8a8f98;font-size:12px}
button{background:#2b6cb0;color:#fff;border:0;border-radius:6px;padding:8px 12px;font-size:14px;cursor:pointer}
button.tab{background:#1e2430}button.tab.on{background:#2b6cb0}
input{background:#181b22;color:#eee;border:1px solid #333;border-radius:6px;padding:7px;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:4px}
th,td{padding:5px 6px;text-align:right;border-bottom:1px solid #22262e;white-space:nowrap}
th{color:#9aa;font-weight:600}td:first-child,th:first-child{text-align:left}
tr.focus td{background:#161a24}
.bar{height:6px;background:#1e2430;border-radius:3px;margin:8px 0}
.bar>div{height:100%;background:#2b6cb0;border-radius:3px;width:0}
#prog{color:#cfd3da;font-family:ui-monospace,monospace;font-size:12px;min-height:16px}
.hl{background:#161a24;border:1px solid #2a3040;border-radius:8px;padding:10px 12px;margin:10px 0}
.hl b{font-size:20px}.warn{color:#f0b26b}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}
.scroll{overflow-x:auto}label{margin-right:8px;white-space:nowrap}
</style></head><body>
<h1>Outs Lab</h1>
<div class="sub">MLB play-by-play only — free, no lines, no model. Every cell is a count of real starts.</div>
<h2>1. Fetch seasons into the dataset</h2>
<div class="row"><input id="fetchyrs" style="width:220px" value="2021,2022,2023,2024,2025"> <button onclick="go()">Fetch</button>
<span class="sub">comma-separated; resumable, already-stored games are skipped</span></div>
<div id="prog">idle</div><div class="bar"><div id="fill"></div></div>
<div id="cov"></div>
<h2>2. Query any season set</h2>
<div class="row" id="yrs"></div>
<div class="row" id="stats"></div>
<div id="out"></div>
<script>
const $=s=>document.querySelector(s);
const tok=new URLSearchParams(location.search).get('token')||'';
let poll=null, cov=[], stat='outs', report=null;
async function go(){const yrs=$('#fetchyrs').value.split(',').map(x=>+x.trim()).filter(Boolean);
 const r=await fetch('/api/outs-lab/run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({token:tok,seasons:yrs})});
 const j=await r.json(); if(j.error||j.reason)$('#prog').textContent=j.error||j.reason; tick();}
function tick(){if(poll)clearInterval(poll);poll=setInterval(refresh,1500);refresh();}
async function refresh(){
 const s=await (await fetch('/api/outs-lab')).json();
 $('#prog').textContent=(s.state.running?'running: ':'')+(s.state.progress||'idle')+(s.state.error?' — '+s.state.error:'');
 const m=/game (\\d+)\\/(\\d+)/.exec(s.state.progress||'');
 $('#fill').style.width=m?Math.round(100*m[1]/m[2])+'%':(s.state.running?'2%':'0%');
 const wasRunning=!!poll; if(!s.state.running&&poll){clearInterval(poll);poll=null;}
 cov=s.coverage||[]; renderCov(); renderYrs();
 if(!report||(wasRunning&&!s.state.running))query();
}
function renderCov(){
 if(!cov.length){$('#cov').innerHTML='<p class="sub">dataset empty — fetch a season</p>';return;}
 let h='<table><tr><th>season</th><th>games</th><th>starts</th><th>first</th><th>last</th></tr>';
 for(const c of cov)h+=`<tr><td>${c.season}</td><td>${c.games}</td><td>${c.starts}</td><td>${c.first}</td><td>${c.last}</td></tr>`;
 $('#cov').innerHTML=h+'</table>';
}
function renderYrs(){
 const have=new Set([...document.querySelectorAll('#yrs input')].map(i=>+i.value));
 for(const c of cov)if(!have.has(c.season))$('#yrs').insertAdjacentHTML('beforeend',`<label><input type="checkbox" value="${c.season}" checked onchange="query()"> ${c.season}</label>`);
 if(cov.length&&!$('#yrs button'))$('#yrs').insertAdjacentHTML('beforeend','<button onclick="query()">Query</button>');
}
async function query(){
 const yrs=[...document.querySelectorAll('#yrs input:checked')].map(i=>i.value);
 if(!yrs.length){$('#out').innerHTML='<p class="sub">pick at least one season</p>';return;}
 report=await (await fetch('/api/outs-lab/report?seasons='+yrs.join(','))).json();
 renderStats(); render();
}
function renderStats(){
 if(!report||!report.grids)return;
 $('#stats').innerHTML=Object.entries(report.grids).map(([k,g])=>`<button class="tab ${k===stat?'on':''}" onclick="stat='${k}';renderStats();render()">${g.label}</button>`).join('');
}
const pct=x=>x==null?'—':(100*x).toFixed(1)+'%';
function render(){
 const r=report; if(!r||r.error){$('#out').innerHTML=`<p class="sub">${r?r.error:'no report'}</p>`;return;}
 const h=r.headline, g=r.grids[stat];
 let html=`<div class="hl"><div class="sub">seasons ${r.label} · ${r.starts} starts · ${r.tbf} BF</div>
 <div>${h.question}</div><b>${pct(h.empirical)}</b> <span class="sub">(${h.fair}, n=${h.starts} starts that faced ≥17)</span>
 <div class="sub">closed form: ${pct(h.closed_form_outs_rate)} using outs/BF · ${pct(h.closed_form_retire_rate)} using retire rate</div>
 <div class="sub">league outs/BF <b style="font-size:14px">${r.outs_per_bf}</b> · retire rate <b style="font-size:14px">${r.retire_rate}</b> · extra outs/BF ${r.extra_outs_per_bf}</div></div>`;
 if(Object.keys(r.unknown_events||{}).length)html+=`<div class="warn">unknown eventTypes counted as PA: ${JSON.stringify(r.unknown_events)}</div>`;
 html+=`<h2>${g.label}: P(over line | first n batters faced) — empirical, binomial(${g.per_bf_rate}/BF) underneath</h2><div class="scroll"><table><tr><th>n BF</th><th>starts</th>`;
 for(const l of g.lines)html+=`<th>${l}</th>`;
 html+=`</tr>`;
 for(const x of g.rows){html+=`<tr class="${x.n>=17&&x.n<=23?'focus':''}"><td>${x.n}</td><td>${x.starts}</td>`;
  for(const l of g.lines)html+=`<td><b>${pct(x[l])}</b><br><span class="sub">${pct(x['binom_'+l])}</span></td>`;
  html+=`</tr>`;}
 html+=`</table></div>`;
 if(stat==='outs'){
 html+=`<h2>14.5 detail — with 95% CI and dispersion vs binomial</h2><div class="scroll"><table><tr><th>n</th><th>starts</th><th>empirical</th><th>95% CI</th><th>fair</th><th>binom(outs/BF)</th><th>binom(retire)</th><th>mean outs</th><th>expected</th><th>dispersion</th></tr>`;
 for(const x of r.ladder)html+=`<tr class="${x.focus?'focus':''}"><td>${x.n}</td><td>${x.starts}</td><td><b>${pct(x.empirical)}</b></td><td>${pct(x.ci95[0])}–${pct(x.ci95[1])}</td><td>${x.fair}</td><td>${pct(x.pred_outs_rate)}</td><td>${pct(x.pred_retire_rate)}</td><td>${x.mean_outs_after}</td><td>${x.expected_outs_after}</td><td>${x.dispersion??'—'}</td></tr>`;
 html+=`</table></div>`;
 html+=`<h2>Extra outs by base state (DP / CS / pickoff per PA)</h2><table><tr><th>runners on</th><th>PA</th><th>extra outs / PA</th><th>reach rate</th></tr>`;
 for(const x of r.base_state)html+=`<tr><td>${x.runners_on}</td><td>${x.pa}</td><td>${x.extra_outs_per_pa}</td><td>${pct(x.reach_rate)}</td></tr>`;
 html+=`</table><h2>Out rates by how the start went (TBF ≥ 12)</h2><table><tr><th>reach-rate bucket</th><th>starts</th><th>outs/BF</th><th>retire</th><th>extra/BF</th></tr>`;
 for(const x of r.traffic)html+=`<tr><td>${x.reach_rate_bucket}</td><td>${x.starts}</td><td>${x.outs_per_bf}</td><td>${x.retire_rate}</td><td>${x.extra_outs_per_bf}</td></tr>`;
 html+=`</table><h2>Reach rate by time through the order</h2><table><tr><th>TTO</th><th>PA</th><th>reach rate</th></tr>`;
 for(const x of r.tto)html+=`<tr><td>${x.tto}</td><td>${x.pa}</td><td>${pct(x.reach_rate)}</td></tr>`;
 html+=`</table><h2>For contrast: P(15+ outs) conditioned on FINAL TBF = n (hook-polluted)</h2><table><tr><th>final TBF</th><th>starts</th><th>P(15+)</th></tr>`;
 for(const x of r.final_tbf_view)html+=`<tr><td>${x.n}</td><td>${x.starts}</td><td>${pct(x.p15)}</td></tr>`;
 html+=`</table>`;}
 $('#out').innerHTML=html;
}
refresh();
</script></body></html>"""


def register(app):
    """Wire /outs-lab (page) + /api/outs-lab (state+history) +
    POST /api/outs-lab/run (LAB_TOKEN-gated like the other lab runners)."""
    from fastapi.responses import HTMLResponse
    lab_token = os.getenv("LAB_TOKEN", "")

    @app.get("/outs-lab")
    def outs_lab_page():
        return HTMLResponse(PAGE)

    @app.get("/api/outs-lab")
    def outs_lab_state():
        return {"state": state(), "runs": history(), "coverage": coverage()}

    @app.get("/api/outs-lab/report")
    def outs_lab_report(seasons: str = ""):
        """Grids for any stored season set, e.g. ?seasons=2023,2024,2025."""
        try:
            yrs = [int(x) for x in seasons.split(",") if x.strip()]
            if not yrs:
                yrs = [c["season"] for c in coverage()]
            return report_for(yrs)
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/outs-lab/run")
    def outs_lab_run(payload: dict):
        if not lab_token or payload.get("token") != lab_token:
            return {"error": "bad token"}
        try:
            yrs = payload.get("seasons") or [payload.get("season")]
            return start_async([int(x) for x in yrs if x])
        except Exception as e:
            return {"error": str(e)}
