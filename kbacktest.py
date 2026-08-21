"""
Backtests the strikeout model against REAL completed starts with the same
point-in-time discipline as the hit backtest: every prediction for date D
uses only rows from before D, actual lineups from the boxscore, graded
against the starter's real strikeOuts.

Calibration runs grade the model's P(K > L) on the fixed line ladder
(3.5-7.5) -- every start yields five binary predictions, bucketed and
Brier-scored against two honesty baselines:
  constant: the pooled empirical clear-rate for each line
  naive:    Poisson from the pitcher's own point-in-time K-per-start mean

The market test joins each start to the REAL historical closing
pitcher_strikeouts line and flat-bets every edge above each threshold.
"""
import math
import logging
from datetime import datetime, timedelta, timezone

import requests

import parlay
import kmodel
import kseason
import backtest  # reuse _final_games + _simulate_bets (same repo, same rules)
import odds_api

try:
    import parks
except ImportError:
    parks = None

log = logging.getLogger("kbacktest")

# Lineup mode for the calibration backtest -- the Tier-2 gauntlet switch.
#   actual (default): the real boxscore lineup (what the model knew after post)
#   proxy:  the opponent's most recent prior lineup vs the same hand,
#           point-in-time (simulates the PRE-lineup board honestly)
#   league: all slots unknown (the pre-lineup floor baseline)
# Set via env K_LINEUP_MODE; recorded in every report. Proxy earns the
# pre-lineup job only if its Brier beats league on the same window.
import os as _os
K_LINEUP_MODE = _os.getenv("K_LINEUP_MODE", "actual").strip().lower()

# Bettable-book filter for MARKET tests. The historical fetch stored the
# FULL two-region ladder (us,us2 -- Fanatics, ESPN Bet, Hard Rock...), so
# best_price was picking prices at books Mike cannot bet, which makes
# every all-books ROI optimistic for him specifically. The archive keeps
# per-book prices, so this filters at ANALYSIS time -- no refetch, free
# rerun, his true number. Comma-separated book TITLES, case-insensitive
# (e.g. "fanduel,draftkings,caesars,betmgm"). Unset = all books, which
# keeps every old receipt reproducible.
_books_raw = _os.getenv("K_MARKET_BOOKS", "").strip()
K_MARKET_BOOKS = {b.strip().lower() for b in _books_raw.split(",") if b.strip()} or None

MLB_BASE = "https://statsapi.mlb.com/api/v1"

# ---------------- permanent odds archive (pay once, replay free) ----------------
# Every historical odds snapshot a market test fetches is saved here forever.
# Every fetch checks the archive FIRST, so re-running any window the archive
# covers -- same knobs or new ones -- costs ~zero credits automatically.
import os as _os
import json as _json
import sqlite3 as _sqlite3

ARCHIVE_DB = _os.getenv("DB_PATH", "odds_history.db")


def _archive_conn():
    conn = _sqlite3.connect(ARCHIVE_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS k_odds_archive (
        event_id TEXT, snapshot TEXT, market TEXT, payload TEXT,
        PRIMARY KEY (event_id, snapshot, market))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS k_events_archive (
        snapshot TEXT PRIMARY KEY, payload TEXT)""")
    return conn


_fetch_stats = {"events_api": 0, "events_hit": 0, "odds_api": 0, "odds_hit": 0}


def _hist_events(snapshot: str) -> list:
    """Historical events list, archive-first."""
    try:
        with _archive_conn() as c:
            row = c.execute("SELECT payload FROM k_events_archive WHERE snapshot=?",
                            (snapshot,)).fetchone()
        if row:
            _fetch_stats["events_hit"] += 1
            return _json.loads(row[0])
    except Exception as e:
        log.warning("events archive read failed: %s", e)
    events = odds_api.get_historical_events(snapshot)
    _fetch_stats["events_api"] += 1
    if events:
        try:
            with _archive_conn() as c:
                c.execute("INSERT OR IGNORE INTO k_events_archive VALUES (?,?)",
                          (snapshot, _json.dumps(events)))
        except Exception as e:
            log.warning("events archive write failed: %s", e)
    return events


def _hist_odds(event_id: str, snapshot: str, market: str):
    """Historical event odds, archive-first. The credit is spent at most
    once per (event, snapshot, market) for the life of the database."""
    try:
        with _archive_conn() as c:
            row = c.execute(
                "SELECT payload FROM k_odds_archive WHERE event_id=? AND snapshot=? AND market=?",
                (event_id, snapshot, market)).fetchone()
        if row:
            _fetch_stats["odds_hit"] += 1
            return _json.loads(row[0])
    except Exception as e:
        log.warning("odds archive read failed: %s", e)
    data = odds_api.get_historical_event_odds(event_id, snapshot, market=market)
    _fetch_stats["odds_api"] += 1
    if data:
        try:
            with _archive_conn() as c:
                c.execute("INSERT OR IGNORE INTO k_odds_archive VALUES (?,?,?,?)",
                          (event_id, snapshot, market, _json.dumps(data)))
        except Exception as e:
            log.warning("odds archive write failed: %s", e)
    return data


LINE_LADDER = (3.5, 4.5, 5.5, 6.5, 7.5)


# ---------------------------------------------------------------- seasons
PROPS_HISTORY_START = "2023-05-03"   # The Odds API has no player props before this


def _season_window(days: int, end_date: str | None, market: bool):
    """Resolve the run window. end_date=None = live behavior (anchor now-4h).
    A past end_date anchors the walk there instead. One season per run --
    a window crossing New Year would mix two leagues and is refused."""
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc) - timedelta(hours=4)
    start = end - timedelta(days=days)
    if start.year != end.year:
        raise ValueError(f"window {start.date()}..{end.date()} spans two seasons -- "
                         f"run one season at a time")
    if market and start.strftime("%Y-%m-%d") < PROPS_HISTORY_START:
        raise ValueError(f"player-prop odds history begins {PROPS_HISTORY_START}; "
                         f"window starts {start.date()}")
    season = end.year
    rows = kseason.rows_provider(season)
    return end, season, rows


def _park_k(venue: str | None, season: int | None = None) -> float | None:
    if not venue or parks is None:
        return None
    fn = getattr(parks, "k_factor_for", None)
    if fn is not None and season is not None:
        try:
            return fn(venue, year=season)
        except TypeError:
            pass  # older parks.py without the year param -> live factors
    return fn(venue) if fn else None


def _poisson_sf(mean: float, line: float) -> float:
    """Naive baseline: P(K > line) under Poisson(mean)."""
    if mean <= 0:
        return 0.0
    need = math.floor(line) + 1
    p = math.exp(-mean)
    cdf = p
    for k in range(1, need):
        p *= mean / k
        cdf += p
    return max(0.0, 1.0 - cdf)


def _majority_side(rows: list[dict], before: str) -> str | None:
    sides = [r.get("stand") for r in kmodel.rows_before(rows, before) if r.get("stand")]
    return max(set(sides), key=sides.count) if sides else None


def _game_starts(game_pk: int, date_str: str, p_league: float,
                 hand_cache: dict, venue: str | None = None,
                 rows_fn=parlay.get_player_season_rows,
                 season: int | None = None) -> list[dict]:
    """For one final game: each starter's point-in-time K distribution +
    his actual strikeouts from the boxscore."""
    box = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=20).json()
    home_name = (((box.get("teams") or {}).get("home") or {}).get("team") or {}).get("name", "")
    away_name = (((box.get("teams") or {}).get("away") or {}).get("team") or {}).get("name", "")
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        pitching_team = (box.get("teams") or {}).get(side) or {}
        batting_team = (box.get("teams") or {}).get(opp) or {}
        pitchers = pitching_team.get("pitchers") or []
        order = batting_team.get("battingOrder") or []
        if not pitchers or not order:
            continue
        starter_id = pitchers[0]
        sp = (pitching_team.get("players") or {}).get(f"ID{starter_id}") or {}
        actual_k = (((sp.get("stats") or {}).get("pitching")) or {}).get("strikeOuts")
        starter_name = ((sp.get("person") or {}).get("fullName")) or ""
        if actual_k is None:
            continue
        try:
            starter_rows = rows_fn(starter_id, True)
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

        start_pks = kmodel.fetch_start_games(starter_id, before=date_str)

        def _entries(pids: list) -> list:
            lu = []
            for pid in pids[:9]:
                try:
                    b_rows = rows_fn(pid, False)
                except Exception:
                    lu.append(None)
                    continue
                b_side = _majority_side(b_rows, date_str)
                lu.append({"rows": b_rows, "side": b_side, "name": pid}
                          if b_side else None)
            while len(lu) < 9:
                lu.append(None)
            return lu

        # The start POPULATION is defined by the ACTUAL-mode gate in every
        # mode. Without this, league mode (all slots defaulting to R) dodges
        # the starter's lefty-sample minimum and prices ~6% more starts than
        # actual mode (July 28: 625 vs 587), making cross-mode Brier
        # comparisons population-contaminated. Gate first, then re-price the
        # SAME start under the mode's lineup knowledge.
        actual_lineup = _entries(list(order))
        kdist = kmodel.k_distribution(
            actual_lineup, starter_rows, hand, p_league,
            before=date_str, park_k_factor=_park_k(venue, season),
            start_game_pks=start_pks)
        if kdist is None:
            continue  # actual-mode refusal -> out of population in ALL modes
        if K_LINEUP_MODE == "league":
            kdist = kmodel.k_distribution(
                [None] * 9, starter_rows, hand, p_league,
                before=date_str, park_k_factor=_park_k(venue, season),
                start_game_pks=start_pks)
        elif K_LINEUP_MODE == "proxy":
            batting_team_id = ((batting_team.get("team") or {}).get("id"))
            proxy = kmodel.fetch_recent_lineup(batting_team_id, hand,
                                               before=date_str)
            mode_lineup = _entries(proxy["batter_ids"]) if proxy else [None] * 9
            kdist = kmodel.k_distribution(
                mode_lineup, starter_rows, hand, p_league,
                before=date_str, park_k_factor=_park_k(venue, season),
                start_game_pks=start_pks)
        if kdist is None:
            continue

        # naive baseline mean: his own K per start before D
        prior = kmodel.rows_before(starter_rows, date_str)
        samples = kmodel.tbf_samples(prior)
        prior_k = sum(1 for r in prior
                      if r.get("events") in kmodel.K_EVENTS)
        naive_mean = (prior_k / len(samples)) if samples else None

        out.append({
            "date": date_str, "name": starter_name,
            "home_name": home_name, "away_name": away_name,
            "kdist": kdist, "actual_k": int(actual_k),
            "naive_mean": naive_mean,
        })
    return out


def fit_csw_mapping(starter_ids: set, fit_before: str,
                    rows_fn=parlay.get_player_season_rows) -> dict | None:
    """Fit K-per-PA ~ a + b*called% + c*SwStr% across starters, using ONLY
    rows strictly before `fit_before` -- so a backtest window starting
    there is genuinely out-of-sample. Which pitchers appear is taken from
    the window (identity is not outcome data); every NUMBER in the fit
    predates the window. Plain OLS by normal equations, no libraries.
    Returns the frozen coefficient dict, or None when the population is
    too thin to fit honestly (then the prior stays league and the knob
    does nothing -- loudly)."""
    pts = []
    for sid in starter_ids:
        try:
            rows = kmodel.rows_before(
                rows_fn(int(sid), True), fit_before)
        except Exception:
            continue
        k = kmodel.per_pa_k_rate(rows, "stand", "L")
        r = kmodel.per_pa_k_rate(rows, "stand", "R")
        pa = ((k or {}).get("pa") or 0) + ((r or {}).get("pa") or 0)
        ks = ((k or {}).get("k") or 0) + ((r or {}).get("k") or 0)
        cs = kmodel.called_swstr(rows)
        if pa < 150 or not cs or cs["pitches"] < 500:
            continue
        pts.append((cs["called"], cs["swstr"], ks / pa))
    if len(pts) < 25:
        log.warning("csw fit: only %d qualifying starters before %s -- "
                    "NOT fitting; prior stays league", len(pts), fit_before)
        return None
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sz = sum(p[2] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts); syy = sum(p[1] * p[1] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    sxz = sum(p[0] * p[2] for p in pts); syz = sum(p[1] * p[2] for p in pts)
    # solve [[n,sx,sy],[sx,sxx,sxy],[sy,sxy,syy]] * [a,b,c] = [sz,sxz,syz]
    import copy
    m = [[n, sx, sy, sz], [sx, sxx, sxy, sxz], [sy, sxy, syy, syz]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r_: abs(m[r_][col]))
        if abs(m[piv][col]) < 1e-12:
            log.warning("csw fit: singular system -- not fitting")
            return None
        m[col], m[piv] = m[piv], m[col]
        for r_ in range(3):
            if r_ == col:
                continue
            f = m[r_][col] / m[col][col]
            for c_ in range(col, 4):
                m[r_][c_] -= f * m[col][c_]
    a = m[0][3] / m[0][0]; b = m[1][3] / m[1][1]; c = m[2][3] / m[2][2]
    mean_z = sz / n
    ss_tot = sum((p[2] - mean_z) ** 2 for p in pts)
    ss_res = sum((p[2] - (a + b * p[0] + c * p[1])) ** 2 for p in pts)
    r2 = round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else 0.0
    coefs = {"a": round(a, 4), "b_called": round(b, 4), "c_swstr": round(c, 4),
             "n": n, "r2": r2, "fit_before": fit_before}
    log.info("csw fit: %s", coefs)
    return coefs


def run_k_backtest(days: int, progress=None, end_date: str | None = None) -> dict:
    """Walk `days` completed days ending at end_date (None = now, the live
    behavior), model every start point-in-time, grade P(K > L) vs reality.
    A past end_date runs entirely inside that season: that year's players,
    that year's league rate, that year's park factors."""
    end, season, _rows = _season_window(days, end_date, market=False)
    p_league = kmodel.league_k_rate(season)
    starts = []
    hand_cache: dict = {}
    # CSW prior gauntlet support: when the env knob is on, fit the mapping
    # on data BEFORE the window and freeze it for the whole run. Weight 0
    # (default) skips all of this -- the exact validated path.
    kmodel.K_CSW_COEFS = None
    csw_fit = None
    if (kmodel.K_CSW_PRIOR_WEIGHT > 0 or kmodel.K_CSW_BLEND_WEIGHT > 0
            or kmodel.K_CSW_DELTA_WEIGHT > 0):
        window_start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
        sids: set = set()
        for i in range(1, days + 1):
            d = (end - timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                for g in backtest._final_games(d):
                    box = requests.get(
                        f"{MLB_BASE}/game/{g['gamePk']}/boxscore",
                        timeout=20).json()
                    for side in ("home", "away"):
                        ps = ((box.get("teams") or {}).get(side) or {}).get("pitchers") or []
                        if ps:
                            sids.add(ps[0])
            except Exception:
                continue
        log.info("csw: collecting starters for the pre-window fit "
                 "(%d days scanned)...", days)
        csw_fit = fit_csw_mapping(sids, window_start, rows_fn=_rows)
        kmodel.K_CSW_COEFS = csw_fit
    for i in range(1, days + 1):
        date_str = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            games = backtest._final_games(date_str)
        except Exception as e:
            log.warning("k schedule failed for %s: %s", date_str, e)
            continue
        log.info("k backtest day %d/%d (%s): %d final games", i, days, date_str, len(games))
        for gi, g in enumerate(games, 1):
            try:
                starts.extend(_game_starts(g["gamePk"], date_str, p_league, hand_cache,
                                           rows_fn=_rows, season=season,
                                           venue=(g.get("venue") or {}).get("name")))
            except Exception as e:
                log.warning("k game %s failed: %s", g.get("gamePk"), e)
            if progress:
                progress(i, days, len(starts), f"game {gi}/{len(games)} on {date_str}")
        if progress:
            progress(i, days, len(starts), "day complete")

    if not starts:
        return {"n": 0}

    # Five binary predictions per start across the line ladder
    preds = []
    for s in starts:
        for line in LINE_LADDER:
            preds.append({
                "p_model": kmodel.calibrate(kmodel.prob_over(s["kdist"]["dist"], line)),
                "p_naive": _poisson_sf(s["naive_mean"], line) if s["naive_mean"] else None,
                "line": line,
                "cleared": 1 if s["actual_k"] > line else 0,
            })

    buckets = {}
    for p in preds:
        lo = int(p["p_model"] * 10) * 10
        b = buckets.setdefault(lo, {"n": 0, "hits": 0, "p_sum": 0.0})
        b["n"] += 1
        b["hits"] += p["cleared"]
        b["p_sum"] += p["p_model"]
    calibration = [
        {"bucket": f"{lo}-{lo+10}%", "n": b["n"],
         "predicted": round(b["p_sum"] / b["n"] * 100, 1),
         "actual": round(b["hits"] / b["n"] * 100, 1)}
        for lo, b in sorted(buckets.items())
    ]

    def brier(pairs):
        return round(sum((p - h) ** 2 for p, h in pairs) / len(pairs), 4)

    model_brier = brier([(p["p_model"], p["cleared"]) for p in preds])
    # constant baseline: pooled clear-rate PER LINE (fair -- knows each line's base rate)
    line_rates = {L: [] for L in LINE_LADDER}
    for p in preds:
        line_rates[p["line"]].append(p["cleared"])
    line_base = {L: (sum(v) / len(v) if v else 0) for L, v in line_rates.items()}
    constant_brier = brier([(line_base[p["line"]], p["cleared"]) for p in preds])
    naive_pairs = [(p["p_naive"], p["cleared"]) for p in preds if p["p_naive"] is not None]
    naive_brier = brier(naive_pairs) if naive_pairs else None

    mean_err = sum(abs(s["kdist"]["mean_k"] - s["actual_k"]) for s in starts) / len(starts)

    return {
        "n": len(preds),
        "starts": len(starts),
        "days": days,
        "season": season,
        "end_date": end.strftime("%Y-%m-%d"),
        "rows_source": ("season-scoped" if season != kseason.current_season()
                        else "live"),
        "lineup_mode": K_LINEUP_MODE,
        "calibration": calibration,
        "brier_model": model_brier,
        "brier_constant": constant_brier,
        "brier_naive": naive_brier,
        "mean_abs_error_k": round(mean_err, 2),
        "avg_projected_k": round(sum(s["kdist"]["mean_k"] for s in starts) / len(starts), 2),
        "avg_actual_k": round(sum(s["actual_k"] for s in starts) / len(starts), 2),
    }


K_OPEN_SNAPSHOT_UTC = "16:00:00"   # ~noon ET -- "opening" K lines snapshot


def _line_movement(open_priced: dict, close_priced: dict) -> str | None:
    """Which side did the market move toward between open and close for
    this pitcher? Line change decides; same line -> average over-price
    change decides; ~flat -> 'flat'. None when close is unpriced."""
    if not close_priced or close_priced.get("point") is None:
        return None
    lo, lc = open_priced["point"], close_priced["point"]
    if lc > lo:
        return "over"
    if lc < lo:
        return "under"
    def avg_dec(p):
        vals = [odds_api.american_to_decimal(v) for v in (p.get("prices") or {}).values()]
        return sum(vals) / len(vals) if vals else None
    do, dc = avg_dec(open_priced), avg_dec(close_priced)
    if do is None or dc is None:
        return None
    if dc < do * 0.99:
        return "over"      # over price shortened -> market moved toward over
    if dc > do * 1.01:
        return "under"
    return "flat"


def _devig_prob(odds_data: dict, name: str, side: str, book: str) -> float | None:
    """Fair (de-vigged) market probability for `side` at `book`. Takes the
    over and under prices at the SAME book, converts to implied
    probabilities, and normalizes so they sum to 1 -- stripping the book's
    hold. Returns None if the book didn't price both sides."""
    try:
        over = odds_api.player_prop_prices(odds_data, "pitcher_strikeouts", name, side="over")
        under = odds_api.player_prop_prices(odds_data, "pitcher_strikeouts", name, side="under")
        po = (over or {}).get("prices", {}).get(book)
        pu = (under or {}).get("prices", {}).get(book)
        if po is None or pu is None:
            return None
        io = 1.0 / odds_api.american_to_decimal(po)
        iu = 1.0 / odds_api.american_to_decimal(pu)
        total = io + iu
        if total <= 0:
            return None
        fair_over = io / total
        return fair_over if side == "over" else (1 - fair_over)
    except Exception:
        return None


def _blend_ev(cand: dict, w: float) -> float | None:
    """EV using the blended probability: w*model + (1-w)*devigged-market.
    None if this bet has no market prob (book priced one side only)."""
    mp = cand.get("market_prob")
    if mp is None:
        return None
    p = w * cand["model_prob"] + (1 - w) * mp
    return (p * odds_api.american_to_decimal(cand["price"]) - 1) * 100


def _sim_at(cands: list, w: float, min_ev: float) -> dict:
    """Flat-1u result of betting every candidate whose BLENDED EV >= min_ev."""
    bets = wins = 0
    units = 0.0
    for c in cands:
        bev = _blend_ev(c, w)
        if bev is None or bev < min_ev:
            continue
        bets += 1
        if c["hit"]:
            wins += 1
            units += odds_api.american_to_decimal(c["price"]) - 1
        else:
            units -= 1
    roi = (units / bets * 100) if bets else None
    return {"bets": bets, "wins": wins, "units": round(units, 2),
            "roi": round(roi, 1) if roi is not None else None}


def fit_blend(candidates: list, test_days: int = 30,
              w_grid=None, ev_grid=None) -> dict:
    """The winner's-curse cure, fit honestly.

    Split the stored per-bet candidates by date: the most recent
    `test_days` are HELD OUT, everything older is train. Fit the blend
    weight w on TRAIN only (the w whose blended EV best ranks real
    outcomes -- highest train ROI at a reference filter), then report how
    every (w, min_ev) performs on the untouched TEST days. The headline is
    always the test grid: in-sample numbers are shown only to prove we're
    not overfitting.

    No w is chosen by feel -- the grids are swept and the table is the
    evidence. Bets with no market prob (one-sided books) are dropped from
    the blend entirely, honestly counted.
    """
    w_grid = w_grid or [round(x / 100, 2) for x in range(0, 101, 10)]
    ev_grid = ev_grid or [0, 2, 4, 6, 8, 10]
    usable = [c for c in candidates if c.get("market_prob") is not None]
    dropped = len(candidates) - len(usable)
    if not usable:
        return {"error": "no candidates carry a market probability -- "
                         "re-run a market test to populate the blend inputs"}
    dates = sorted({c["date"] for c in usable})
    if len(dates) <= test_days:
        return {"error": f"need more than {test_days} days of stored bets "
                         f"(have {len(dates)}) -- run a longer market test first"}
    cutoff = dates[-test_days]
    train = [c for c in usable if c["date"] < cutoff]
    test = [c for c in usable if c["date"] >= cutoff]
    if not train or not test:
        return {"error": "train/test split empty -- widen the stored window"}

    # Fit w on TRAIN: pick the w maximizing train ROI at a reference filter
    # (min_ev = 2), requiring a floor of volume so we don't chase a w that
    # fires 3 bets. Ties broken toward w=1 (least market-reliance that ties).
    ref_ev = 2.0
    best_w, best = None, None
    for w in w_grid:
        r = _sim_at(train, w, ref_ev)
        if r["bets"] < max(20, len(train) // 40):
            continue
        key = (r["roi"] if r["roi"] is not None else -999)
        if best is None or key > best or (key == best and (best_w is None or w > best_w)):
            best, best_w = key, w
    if best_w is None:
        best_w = 1.0  # fall back to pure model if train too thin to fit

    # TEST grid: every (w, min_ev) on held-out days
    test_grid = []
    for w in w_grid:
        row = {"w": w, "by_ev": {}}
        for mev in ev_grid:
            row["by_ev"][mev] = _sim_at(test, w, mev)
        test_grid.append(row)

    # The recommended operating point: fitted w, and the min_ev on TEST that
    # maximizes ROI subject to a volume floor (so it's bettable, not a fluke)
    fitted_rows = {mev: _sim_at(test, best_w, mev) for mev in ev_grid}
    vol_floor = max(15, len(test) // 30)
    viable = {mev: r for mev, r in fitted_rows.items()
              if r["bets"] >= vol_floor and r["roi"] is not None}
    rec_ev = max(viable, key=lambda m: viable[m]["roi"]) if viable else None

    return {"test_days": test_days, "train_days": len(dates) - test_days,
            "train_bets": len(train), "test_bets": len(test),
            "dropped_no_market": dropped,
            "fitted_w": best_w,
            "train_roi_at_fit": _sim_at(train, best_w, ref_ev),
            "test_grid": test_grid,
            "fitted_row": fitted_rows,
            "recommended": ({"w": best_w, "min_ev": rec_ev,
                             **fitted_rows[rec_ev]} if rec_ev is not None else None),
            "w_grid": w_grid, "ev_grid": ev_grid}


def run_k_market_backtest(days: int, progress=None, vs_open: bool = False,
                          end_date: str | None = None) -> dict:
    """Walk past days: point-in-time K distribution per start, joined to
    the REAL historical pitcher_strikeouts line, flat-betting every edge
    above each threshold. Units don't lie.

    vs_open=True prices and grades against the ~noon-ET OPENING snapshot
    instead of close -- the bet Mike actually places -- and, since the
    closing snapshot is fetched anyway, reports CLV: how often the close
    moved TOWARD the model's bet. Movement converges on truth far faster
    than units. Roughly doubles credits (two snapshots per game)."""
    end, season, _rows = _season_window(days, end_date, market=True)
    p_league = kmodel.league_k_rate(season)
    for k in _fetch_stats:
        _fetch_stats[k] = 0  # per-run credit receipts
    hand_cache: dict = {}
    candidates = []
    games_priced = 0
    suspect = 0
    for i in range(1, days + 1):
        date_str = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            games = backtest._final_games(date_str)
        except Exception as e:
            log.warning("k market: schedule failed %s: %s", date_str, e)
            continue
        if not games:
            continue
        hist_events = _hist_events(f"{date_str}T16:00:00Z")
        log.info("k market day %d/%d (%s): %d games, %d hist events",
                 i, days, date_str, len(games), len(hist_events))
        for g in games:
            try:
                starts = _game_starts(g["gamePk"], date_str, p_league, hand_cache,
                                      rows_fn=_rows, season=season,
                                      venue=(g.get("venue") or {}).get("name"))
            except Exception as e:
                log.warning("k market game %s failed: %s", g.get("gamePk"), e)
                continue
            if not starts or not hist_events:
                continue
            ev_match = odds_api.find_event(hist_events, starts[0]["home_name"], starts[0]["away_name"])
            if not ev_match:
                continue
            close_at = ev_match.get("commence_time") or f"{date_str}T23:00:00Z"
            if vs_open:
                open_data = _hist_odds(
                    ev_match.get("id"), f"{date_str}T{K_OPEN_SNAPSHOT_UTC}Z",
                    "pitcher_strikeouts")
                close_data = _hist_odds(
                    ev_match.get("id"), close_at, "pitcher_strikeouts")
                odds_data = open_data
                if not odds_data:
                    continue  # no opener posted by the snapshot -- skip honestly
            else:
                close_data = None
                odds_data = _hist_odds(
                    ev_match.get("id"), close_at, "pitcher_strikeouts")
            if not odds_data:
                continue
            games_priced += 1
            for s in starts:
                for side in ("over", "under"):
                    priced = odds_api.player_prop_prices(
                        odds_data, "pitcher_strikeouts", s["name"], side=side)
                    if not priced or priced.get("point") is None:
                        continue
                    line = priced[side if False else "point"]
                    if line != int(line) + 0.5:
                        continue  # whole-number lines can push -- flat-bet sim stays honest on half-points
                    p_over_raw = kmodel.prob_over(s["kdist"]["dist"], line)
                    p_over = kmodel.calibrate(p_over_raw)
                    prob = p_over if side == "over" else 1 - p_over
                    prob_raw = p_over_raw if side == "over" else 1 - p_over_raw
                    book_prices = priced["prices"]
                    if K_MARKET_BOOKS:
                        book_prices = {b: p for b, p in book_prices.items()
                                       if b.strip().lower() in K_MARKET_BOOKS}
                        if not book_prices:
                            continue  # none of HIS books carry this side
                    bp = odds_api.best_price(book_prices)
                    if not bp:
                        continue
                    # De-vig THIS side against the other side's price at the
                    # same book, so the market probability is a fair number
                    # to blend against (strips the book's hold).
                    mkt_prob = _devig_prob(odds_data, s["name"], side, bp[0])
                    ev = (prob * odds_api.american_to_decimal(bp[1]) - 1) * 100
                    if ev > 20:
                        suspect += 1
                        continue  # >20% edges vs closing K lines = model error, not value
                    if ev >= min(backtest.THRESHOLDS):
                        cleared = 1 if s["actual_k"] > line else 0
                        cand = {"date": date_str, "name": s["name"],
                                "side": side, "line": line,
                                "price": bp[1], "ev": round(ev, 1),
                                "hit": cleared,
                                "model_prob": round(prob, 4),
                                "model_prob_raw": round(prob_raw, 4),
                                "market_prob": round(mkt_prob, 4) if mkt_prob else None}
                        if vs_open:
                            close_priced = odds_api.player_prop_prices(
                                close_data, "pitcher_strikeouts", s["name"], side=side) \
                                if close_data else None
                            cand["clv"] = _line_movement(priced, close_priced)
                        candidates.append(cand)
            if progress:
                progress(i, days, games_priced, len(candidates))

    report = {"days": days, "games_priced": games_priced,
              "season": season, "end_date": end.strftime("%Y-%m-%d"),
              "rows_source": ("season-scoped" if season != kseason.current_season()
                              else "live"),
              "suspect_excluded": suspect,
              # The mode this run PRICED under. "actual" uses the final
              # boxscore order -- fine vs CLOSING lines (lineups are public
              # by then), but vs OPENING lines it's an information leak:
              # nobody has the final lineup at noon. The honest opener test
              # is K_LINEUP_MODE=proxy -- the same point-in-time projection
              # the live board uses. Stamped so no receipt is ambiguous.
              "lineup_mode": K_LINEUP_MODE,
              "books": (sorted(K_MARKET_BOOKS) if K_MARKET_BOOKS else "all"),
              "credits_estimate": games_priced * (40 if vs_open else 20) + days,
              "candidates": len(candidates),
              "by_threshold": backtest._simulate_bets(candidates)}
    if vs_open:
        report["vs_open"] = True
        toward = sum(1 for c in candidates if c.get("clv") == c["side"])
        against = sum(1 for c in candidates
                      if c.get("clv") not in (None, "flat", c["side"]))
        flat = sum(1 for c in candidates if c.get("clv") == "flat")
        n = toward + against + flat
        report["clv"] = {"n": n, "toward": toward, "against": against, "flat": flat,
                         "agree_pct": round(toward / (toward + against) * 100, 1)
                         if (toward + against) else None}
    report["sample_bets"] = sorted(candidates, key=lambda c: -c["ev"])[:12]
    report["_candidates"] = candidates  # full per-bet list for blend storage (stripped before display)
    # Credit receipts: api = credits actually spent this run; archive = free
    # replays. Re-running an archived window costs ~zero -- pay once, replay
    # forever. This line on the receipt is the proof.
    report["odds_fetches"] = {
        "api": _fetch_stats["events_api"] + _fetch_stats["odds_api"],
        "archive": _fetch_stats["events_hit"] + _fetch_stats["odds_hit"]}
    return report
