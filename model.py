"""
Hit-probability model v1 -- transparent and point-in-time.

Method: the odds-ratio / log5 approach (Bill James) -- the standard
published way to combine a batter's rate, a pitcher's rate-allowed, and
the league rate into a matchup rate. No invented weights, no composite
scores. Every projection exposes its inputs.

    odds(p) = p / (1 - p)
    matchup_odds = odds(batter) * odds(pitcher_allowed) / odds(league)
    p_PA = matchup_odds / (1 + matchup_odds)

Per-game: P(1+ hit) = 1 - (1 - p_vs_starter)^PA_s * (1 - p_vs_pen)^PA_p
using expected plate appearances vs the starter and the bullpen.

Point-in-time discipline: all rate helpers take a `before` date and use
only rows strictly earlier -- this is what makes the backtest honest.
"""
import time
import logging

import requests

import parlay
import statcast_api

log = logging.getLogger("model")

MLB_BASE = "https://statsapi.mlb.com/api/v1"

# League-average PAs for a lineup regular, split by pitcher role. MLB team
# averages: ~38 PA/team/game over 9 slots ≈ 4.2; starters face ~60% of them.
PA_VS_STARTER = 2.6
PA_VS_PEN = 1.6

# Regression to the mean: observed rates are shrunk toward league average
# by SHRINK_PA "phantom PAs" of league performance. The published cure for
# small-sample overconfidence -- a 60 PA hot split gets pulled hard toward
# league; a 600 PA sample barely moves.
SHRINK_PA = 150

# Personal playing time: scale expected PAs by the batter's OWN PA/game
# from his real logs (captures lineup slot, pinch-hit risk, rest patterns
# automatically). 1 = on, 0 = off (flat league PAs). The knob anchors
# (PA_VS_STARTER + PA_VS_PEN) define the league-average total that a
# personal rate scales against.
PERSONAL_PA = 1
LEAGUE_PA_PER_GAME = 3.9  # only used to normalize the scaling factor

# xBA blend: 0 = actual hit rates (v4 champion), 1 = fully expected
# (luck-stripped) rates from Savant's per-ball estimates, using the same
# accounting we validated against Savant pages.
XBA_WEIGHT = 0.0

# Player-specific priors (e.g. preseason Steamer): shrink toward the
# player's own projected talent instead of league average. PRIORS maps
# folded player name -> projected per-PA hit rate. PRIOR_WEIGHT blends
# the shrink target: 0 = pure league (v4 champion), 1 = pure player prior.
PRIORS: dict = {}
PRIOR_WEIGHT = 0.0

# Arsenal layer (the KSplit-style matchup weighting, Mike's per-pitch idea):
# reweight the batter's rate by the STARTER'S actual pitch usage vs his
# side -- "his slider problem counts in proportion to the sliders he'll
# see." Per-pitch rates are heavily shrunk (ARSENAL_SHRINK pseudo-PAs
# toward the batter's own overall rate) because per-pitch samples are tiny.
ARSENAL_WEIGHT = 0.0
ARSENAL_SHRINK = 100

# Calibration layer: piecewise-linear correction fit from the model's OWN
# largest backtest receipt (predicted->actual per bucket). Fixes the
# documented cold-bias the market test punished. CALIB_POINTS is loaded by
# the lab from stored runs; weight 0 = raw model.
CALIB_POINTS: list = []   # [(predicted, actual), ...] both 0-1, sorted
CALIB_WEIGHT = 1.0

# Park factor: multiplicative hits-factor (1.0 neutral, Coors ~1.1).
PARK_WEIGHT = 1.0


def calibrate(p: float) -> float:
    """Map a raw probability through the fitted correction curve."""
    if not CALIB_POINTS or CALIB_WEIGHT <= 0:
        return p
    pts = CALIB_POINTS
    if p <= pts[0][0]:
        corrected = p + (pts[0][1] - pts[0][0])
    elif p >= pts[-1][0]:
        corrected = p + (pts[-1][1] - pts[-1][0])
    else:
        corrected = p
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            if x1 <= p <= x2:
                t = (p - x1) / (x2 - x1) if x2 > x1 else 0
                corrected = y1 + t * (y2 - y1)
                break
    corrected = min(max(corrected, 0.02), 0.98)
    return CALIB_WEIGHT * corrected + (1 - CALIB_WEIGHT) * p

import unicodedata as _ud

def fold_name(text: str) -> str:
    text = _ud.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not _ud.combining(ch)).lower().strip()

_league_cache = {"ts": 0, "p": None}


def league_hit_rate() -> float:
    """League per-PA hit rate from MLB's real season team totals
    (sum of hits / sum of plate appearances across all 30 teams)."""
    now = time.time()
    if _league_cache["p"] and now - _league_cache["ts"] < 86400:
        return _league_cache["p"]
    resp = requests.get(
        f"{MLB_BASE}/teams/stats",
        params={"season": 2026, "group": "hitting", "stats": "season", "sportId": 1},
        timeout=20,
    )
    resp.raise_for_status()
    hits = pa = 0
    for split in resp.json()["stats"][0]["splits"]:
        stat = split.get("stat", {})
        hits += int(stat.get("hits", 0))
        pa += int(stat.get("plateAppearances", 0))
    if pa == 0:
        raise RuntimeError("league totals unavailable")
    p = hits / pa
    _league_cache.update({"ts": now, "p": p})
    log.info("League hit rate: %.4f (%d H / %d PA)", p, hits, pa)
    return p


def rows_before(rows: list[dict], before: str | None) -> list[dict]:
    """Point-in-time filter: only rows strictly before `before` (YYYY-MM-DD).
    None = use everything (live usage)."""
    if before is None:
        return rows
    return [r for r in rows if (r.get("game_date") or "9999") < before]


def _sf(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def per_pa_hit_rate(rows: list[dict], split_col: str, split_val: str) -> dict | None:
    """Hits per PA within a split -- both ACTUAL and EXPECTED (xBA-based).
    Expected uses the Savant-validated accounting: per-ball estimates for
    tracked batted balls, 0 for strikeouts/walks, untracked batted balls
    excluded from both numerator and denominator."""
    pa = hits = 0
    x_num = 0.0
    untracked = 0
    for r in rows:
        if r.get(split_col) != split_val:
            continue
        ev = r.get("events")
        if not ev or ev in statcast_api.NON_PA_EVENTS:
            continue
        pa += 1
        if ev in parlay.HIT_EVENTS:
            hits += 1
        est = _sf(r.get("estimated_ba_using_speedangle"))
        if est is not None:
            x_num += est
        elif r.get("type") == "X" or ev in parlay.HIT_EVENTS:
            # a batted ball Savant didn't track -- unusable for expected
            untracked += 1
    if pa == 0:
        return None
    x_pa = pa - untracked
    x_rate = (x_num / x_pa) if x_pa > 0 else None
    return {"pa": pa, "hits": hits, "rate": hits / pa, "x_rate": x_rate}


def pa_per_game_vs_hand(rows: list[dict], hand: str) -> tuple[float, int]:
    """PA/game restricted to games the batter's FIRST plate appearance came
    against this hand -- i.e., games started by that hand. This is the
    platoon reality: a lefty-shy bat's PA/game vs LHP starters is a
    different, smaller number than his overall."""
    games: dict = {}
    for r in rows:
        gpk, date = r.get("game_pk"), r.get("game_date")
        if gpk is None or not date:
            continue
        key = (date, gpk)
        g = games.setdefault(key, {"pa": 0, "first_ab": None, "first_hand": None})
        ev = r.get("events")
        if ev and ev not in statcast_api.NON_PA_EVENTS:
            g["pa"] += 1
        try:
            ab = int(float(r.get("at_bat_number") or 9999))
        except (TypeError, ValueError):
            ab = 9999
        if g["first_ab"] is None or ab < g["first_ab"]:
            g["first_ab"] = ab
            g["first_hand"] = r.get("p_throws")
    matched = [g["pa"] for g in games.values() if g["first_hand"] == hand]
    if not matched:
        return 0.0, 0
    return sum(matched) / len(matched), len(matched)


def pa_per_game(rows: list[dict]) -> tuple[float, int]:
    """The batter's real plate appearances per game from his logs.
    Returns (pa_per_game, games)."""
    games = set()
    pa = 0
    for r in rows:
        gpk, date = r.get("game_pk"), r.get("game_date")
        ev = r.get("events")
        if gpk is None or not date:
            continue
        games.add((date, gpk))
        if ev and ev not in statcast_api.NON_PA_EVENTS:
            pa += 1
    n = len(games)
    if n == 0:
        return 0.0, 0
    return pa / n, n


def shrunk_rate(hits: int, pa: int, p_league: float) -> float:
    """Empirical-Bayes shrinkage toward the league rate."""
    return (hits + SHRINK_PA * p_league) / (pa + SHRINK_PA)


def arsenal_rate(batter_rows: list[dict], starter_rows: list[dict],
                 batter_overall: float) -> dict | None:
    """Usage-weighted expected hit rate: for each pitch the starter throws
    to this side, the batter's (shrunk) per-PA hit rate against that pitch
    type, weighted by the starter's real usage. Returns None if either
    side lacks pitch data."""
    usage: dict = {}
    for r in starter_rows:
        pt = r.get("pitch_type")
        if pt:
            usage[pt] = usage.get(pt, 0) + 1
    total = sum(usage.values())
    if total < 100:
        return None

    per_pitch: dict = {}
    for r in batter_rows:
        ev = r.get("events")
        if not ev or ev in statcast_api.NON_PA_EVENTS:
            continue
        pt = r.get("pitch_type")
        if not pt:
            continue
        d = per_pitch.setdefault(pt, {"pa": 0, "hits": 0})
        d["pa"] += 1
        if ev in parlay.HIT_EVENTS:
            d["hits"] += 1
    if not per_pitch:
        return None

    rate = 0.0
    detail = {}
    for pt, count in usage.items():
        w = count / total
        d = per_pitch.get(pt, {"pa": 0, "hits": 0})
        shrunk = (d["hits"] + ARSENAL_SHRINK * batter_overall) / (d["pa"] + ARSENAL_SHRINK)
        rate += w * shrunk
        if w >= 0.05:
            detail[pt] = {"usage": round(w, 3), "batter_pa": d["pa"], "rate": round(shrunk, 4)}
    return {"rate": rate, "detail": detail}


def _odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


def log5_rate(p_batter: float, p_pitcher_allowed: float, p_league: float) -> float:
    """The published odds-ratio combination."""
    combined = _odds(p_batter) * _odds(p_pitcher_allowed) / _odds(p_league)
    return combined / (1 + combined)


def hit_probability(batter_rows: list[dict], starter_rows: list[dict],
                    starter_hand: str, batter_side: str, p_league: float,
                    before: str | None = None,
                    min_batter_pa: int = 40, min_starter_pa: int = 60,
                    batter_name: str | None = None,
                    park_factor: float | None = None) -> dict | None:
    """P(batter records 1+ hit today) with full input transparency.
    Returns None when samples are too thin to say anything honest."""
    b_rows = rows_before(batter_rows, before)
    s_rows = rows_before(starter_rows, before)

    b = per_pa_hit_rate(b_rows, "p_throws", starter_hand)
    s = per_pa_hit_rate(s_rows, "stand", batter_side)
    if not b or b["pa"] < min_batter_pa or not s or s["pa"] < min_starter_pa:
        return None

    # Shrink target: league rate, optionally blended toward the player's
    # own projected talent (preseason prior) when one is on file
    b_target = p_league
    prior_rate = None
    if PRIOR_WEIGHT > 0 and batter_name:
        prior_rate = PRIORS.get(fold_name(batter_name))
        if prior_rate is not None:
            b_target = PRIOR_WEIGHT * prior_rate + (1 - PRIOR_WEIGHT) * p_league

    def blended(stats, target):
        actual = stats["hits"] / stats["pa"]
        if XBA_WEIGHT > 0 and stats.get("x_rate") is not None:
            raw = XBA_WEIGHT * stats["x_rate"] + (1 - XBA_WEIGHT) * actual
        else:
            raw = actual
        return (raw * stats["pa"] + SHRINK_PA * target) / (stats["pa"] + SHRINK_PA)

    b_rate = blended(b, b_target)
    s_rate = blended(s, p_league)

    # Arsenal fit: tilt the batter's rate toward what he does against the
    # pitches this specific starter actually throws to his side
    ars = None
    if ARSENAL_WEIGHT > 0:
        b_split = [r for r in b_rows if r.get("p_throws") == starter_hand]
        s_split = [r for r in s_rows if r.get("stand") == batter_side]
        ars = arsenal_rate(b_split, s_split, b["hits"] / b["pa"])
        if ars:
            # shrink the arsenal rate like everything else, then blend
            a_shrunk = (ars["rate"] * b["pa"] + SHRINK_PA * b_target) / (b["pa"] + SHRINK_PA)
            b_rate = ARSENAL_WEIGHT * a_shrunk + (1 - ARSENAL_WEIGHT) * b_rate
    p_vs_starter = log5_rate(b_rate, s_rate, p_league)
    # Vs the pen (unknown arms): the batter's own shrunk rate vs the hand
    p_vs_pen = b_rate

    pa_s, pa_p = PA_VS_STARTER, PA_VS_PEN
    personal = None
    if PERSONAL_PA:
        # HAND-CONDITIONAL playing time: PA/game in games started by
        # today's hand (the platoon/pinch-hit reality). Falls back to
        # overall when the conditional sample is thin.
        pg_h, n_h = pa_per_game_vs_hand(b_rows, starter_hand)
        pg_all, n_all = pa_per_game(b_rows)
        if n_h >= 8 and pg_h > 0:
            pg, n_games, basis = pg_h, n_h, f"vs {starter_hand}HP starters"
        elif n_all >= 15 and pg_all > 0:
            pg, n_games, basis = pg_all, n_all, "overall"
        else:
            pg, n_games, basis = 0, 0, None
        if n_games:
            factor = max(0.55, min(1.25, pg / LEAGUE_PA_PER_GAME))
            pa_s, pa_p = PA_VS_STARTER * factor, PA_VS_PEN * factor
            personal = {"pa_per_game": round(pg, 2), "games": n_games,
                        "basis": basis, "scale": round(factor, 3)}

    # Park: scale per-PA hit rates by the venue's hits factor
    park = None
    if park_factor and PARK_WEIGHT > 0:
        pf = max(0.85, min(1.15, park_factor)) ** PARK_WEIGHT
        p_vs_starter = min(p_vs_starter * pf, 0.95)
        p_vs_pen = min(p_vs_pen * pf, 0.95)
        park = round(park_factor, 3)

    p_no_hit = ((1 - p_vs_starter) ** pa_s) * ((1 - p_vs_pen) ** pa_p)
    p_raw = 1 - p_no_hit
    return {
        "p_hit": round(calibrate(p_raw), 4),
        "p_raw": round(p_raw, 4),
        "inputs": {
            "batter_rate_vs_hand": round(b["rate"], 4), "batter_rate_shrunk": round(b_rate, 4), "batter_pa": b["pa"],
            "starter_rate_allowed_vs_side": round(s["rate"], 4), "starter_rate_shrunk": round(s_rate, 4), "starter_pa": s["pa"],
            "shrink_pa": SHRINK_PA,
            "prior_rate": round(prior_rate, 4) if prior_rate is not None else None,
            "arsenal": (ars or {}).get("detail") if ARSENAL_WEIGHT > 0 else None,
            "park_factor": park,
            "league_rate": round(p_league, 4),
            "p_pa_vs_starter": round(p_vs_starter, 4),
            "pa_vs_starter": round(pa_s, 2), "pa_vs_pen": round(pa_p, 2),
            "personal_pa": personal,
        },
    }
