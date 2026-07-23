"""
Strikeout-probability model v1 -- transparent and point-in-time.

SEPARATE from the hit model and from Pitcher Projections: this module
never touches model.py or pitchers.py state. It reads the same validated
row data and follows the same house rules -- log5, empirical-Bayes
shrinkage, `before`-date discipline, every input exposed.

Method (the Lab roadmap spec, now real):
  1. Per-PA K probability for each lineup hitter: log5 of the STARTER's
     K rate vs that side x the BATTER's K rate vs that hand / league.
  2. Batters faced is a DISTRIBUTION, not a point: the starter's own real
     TBF-per-start samples (point-in-time). His actual workload variance
     supplies the "leash" -- no invented profiles.
  3. Lineup slots get PAs by batting order arithmetic (slot 1 bats more
     than slot 9), each PA a Bernoulli with its slot's probability.
  4. Total strikeouts = mixture over TBF of the exact Poisson-binomial.
     P(K >= line) read straight off the distribution -- no normal approx.
"""
import time
import logging

import requests

import parlay
import statcast_api
from pitchers import K_EVENTS  # single source of truth for K accounting

log = logging.getLogger("kmodel")

MLB_BASE = "https://statsapi.mlb.com/api/v1"

# ---- knobs (set by lab._apply_k_config; defaults = raw v1 baseline) ----
K_SHRINK_PA = 120          # phantom league PAs both sides
K_MIN_BATTER_PA = 40       # below: batter priced at league (flagged), not refused
K_MIN_STARTER_TBF = 60     # below: refuse to predict
K_MIN_STARTS = 3           # need real TBF samples for the workload mixture
K_ARSENAL_WEIGHT = 0.0     # per-pitch K layer; 0 until a backtest earns it
K_ARSENAL_SHRINK = 100
K_PARK_WEIGHT = 1.0        # Savant strikeout park factor
K_CALIB_WEIGHT = 1.0       # correction curve fit from stored K runs
K_CALIB_POINTS: list = []  # [(predicted, actual)] on P(over) probabilities


def calibrate(p: float) -> float:
    """Same piecewise-linear correction as the hit model, fit from the
    K model's OWN stored raw runs."""
    if not K_CALIB_POINTS or K_CALIB_WEIGHT <= 0:
        return p
    pts = K_CALIB_POINTS
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
    corrected = min(max(corrected, 0.01), 0.99)
    return K_CALIB_WEIGHT * corrected + (1 - K_CALIB_WEIGHT) * p


_league_cache = {"ts": 0, "p": None}


def league_k_rate() -> float:
    """League per-PA strikeout rate from MLB's real season team totals."""
    now = time.time()
    if _league_cache["p"] and now - _league_cache["ts"] < 86400:
        return _league_cache["p"]
    resp = requests.get(
        f"{MLB_BASE}/teams/stats",
        params={"season": 2026, "group": "hitting", "stats": "season", "sportId": 1},
        timeout=20,
    )
    resp.raise_for_status()
    ks = pa = 0
    for split in resp.json()["stats"][0]["splits"]:
        stat = split.get("stat", {})
        ks += int(stat.get("strikeOuts", 0))
        pa += int(stat.get("plateAppearances", 0))
    if pa == 0:
        raise RuntimeError("league totals unavailable")
    p = ks / pa
    _league_cache.update({"ts": now, "p": p})
    log.info("League K rate: %.4f (%d K / %d PA)", p, ks, pa)
    return p


def rows_before(rows: list[dict], before: str | None) -> list[dict]:
    if before is None:
        return rows
    return [r for r in rows if (r.get("game_date") or "9999") < before]


def per_pa_k_rate(rows: list[dict], split_col: str, split_val: str) -> dict | None:
    """K per PA within a split -- validated PA accounting."""
    pa = k = 0
    for r in rows:
        if r.get(split_col) != split_val:
            continue
        ev = r.get("events")
        if not ev or ev in statcast_api.NON_PA_EVENTS:
            continue
        pa += 1
        if ev in K_EVENTS:
            k += 1
    if pa == 0:
        return None
    return {"pa": pa, "k": k, "rate": k / pa}


def shrunk(k: int, pa: int, p_league: float, pseudo: float | None = None) -> float:
    ps = K_SHRINK_PA if pseudo is None else pseudo
    return (k + ps * p_league) / (pa + ps)


def _odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


def log5(p_batter_k: float, p_pitcher_k: float, p_league: float) -> float:
    combined = _odds(p_batter_k) * _odds(p_pitcher_k) / _odds(p_league)
    return combined / (1 + combined)


def k_arsenal_rate(batter_rows: list[dict], starter_rows: list[dict],
                   batter_overall_k: float) -> dict | None:
    """Usage-weighted per-PA K rate: the batter's (heavily shrunk) K rate
    against each pitch type, weighted by the starter's real usage vs his
    side -- 'his slider problem counts in proportion to the sliders he'll
    see', K edition."""
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
        d = per_pitch.setdefault(pt, {"pa": 0, "k": 0})
        d["pa"] += 1
        if ev in K_EVENTS:
            d["k"] += 1
    if not per_pitch:
        return None
    rate = 0.0
    detail = {}
    for pt, count in usage.items():
        w = count / total
        d = per_pitch.get(pt, {"pa": 0, "k": 0})
        sh = (d["k"] + K_ARSENAL_SHRINK * batter_overall_k) / (d["pa"] + K_ARSENAL_SHRINK)
        rate += w * sh
        if w >= 0.05:
            detail[pt] = {"usage": round(w, 3), "batter_pa": d["pa"], "k_rate": round(sh, 4)}
    return {"rate": rate, "detail": detail}


def tbf_samples(starter_rows: list[dict]) -> list[int]:
    """The starter's REAL batters-faced count for each of his starts --
    the workload distribution, straight from his logs."""
    games: dict = {}
    for r in starter_rows:
        gpk, date = r.get("game_pk"), r.get("game_date")
        if gpk is None or not date:
            continue
        ev = r.get("events")
        if ev and ev not in statcast_api.NON_PA_EVENTS:
            games[(date, gpk)] = games.get((date, gpk), 0) + 1
    return sorted(games.values())


def slot_pa_counts(tbf: int) -> list[int]:
    """PAs for batting-order slots 1-9 given total batters faced.
    Slot i bats on trips i, i+9, i+18, ... -- exact order arithmetic."""
    return [((tbf - i) // 9 + 1) if tbf >= i else 0 for i in range(1, 10)]


def poisson_binomial(probs: list[float]) -> list[float]:
    """Exact distribution of the sum of independent Bernoullis.
    Returns [P(K=0), P(K=1), ...]."""
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for k, m in enumerate(dist):
            nxt[k] += m * (1 - p)
            nxt[k + 1] += m * p
        dist = nxt
    return dist


def prob_over(dist: list[float], line: float) -> float:
    """P(K > line) for a half-point line (e.g. 5.5 -> P(K >= 6))."""
    import math
    need = math.floor(line) + 1
    return sum(dist[need:]) if need < len(dist) else 0.0


def k_distribution(lineup: list[dict | None], starter_rows: list[dict],
                   starter_hand: str, p_league: float,
                   before: str | None = None,
                   park_k_factor: float | None = None) -> dict | None:
    """The strikeout distribution for one start.

    lineup: 9 entries in batting order -- {'rows': [...], 'side': 'L'/'R',
    'name': str} or None when the slot is unknown (priced at league).
    Returns None only when the STARTER's sample is too thin to say
    anything honest.
    """
    s_rows = rows_before(starter_rows, before)
    samples = tbf_samples(s_rows)
    if len(samples) < K_MIN_STARTS:
        return None

    slot_probs = []
    slot_inputs = []
    league_fallbacks = 0
    for slot, entry in enumerate(lineup[:9], start=1):
        side = (entry or {}).get("side") or "R"
        s = per_pa_k_rate(s_rows, "stand", side)
        if not s or s["pa"] < K_MIN_STARTER_TBF:
            return None  # starter sample vs this side too thin -- refuse
        s_rate = shrunk(s["k"], s["pa"], p_league)

        b_rate = p_league
        b_info = {"slot": slot, "name": None, "basis": "league (slot unknown)"}
        if entry:
            b_rows = rows_before(entry["rows"], before)
            b = per_pa_k_rate(b_rows, "p_throws", starter_hand)
            if b and b["pa"] >= K_MIN_BATTER_PA:
                b_rate = shrunk(b["k"], b["pa"], p_league)
                basis = f"{b['pa']} PA vs {starter_hand}HP"
                if K_ARSENAL_WEIGHT > 0:
                    b_split = [r for r in b_rows if r.get("p_throws") == starter_hand]
                    s_split = [r for r in s_rows if r.get("stand") == side]
                    ars = k_arsenal_rate(b_split, s_split, b["k"] / b["pa"])
                    if ars:
                        a_shrunk = (ars["rate"] * b["pa"] + K_SHRINK_PA * p_league) / (b["pa"] + K_SHRINK_PA)
                        b_rate = K_ARSENAL_WEIGHT * a_shrunk + (1 - K_ARSENAL_WEIGHT) * b_rate
                        basis += " +arsenal"
                b_info = {"slot": slot, "name": entry.get("name"), "basis": basis}
            else:
                league_fallbacks += 1
                b_info = {"slot": slot, "name": entry.get("name"),
                          "basis": "league (thin sample)"}

        p = log5(b_rate, s_rate, p_league)
        if park_k_factor and K_PARK_WEIGHT > 0:
            p = min(p * (max(0.85, min(1.15, park_k_factor)) ** K_PARK_WEIGHT), 0.9)
        slot_probs.append(p)
        b_info["p_k_per_pa"] = round(p, 4)
        slot_inputs.append(b_info)

    # Mixture over the starter's real workload distribution
    weight = 1.0 / len(samples)
    max_tbf = max(samples)
    dist = [0.0] * (max_tbf + 1)
    for tbf in samples:
        counts = slot_pa_counts(tbf)
        seq = [slot_probs[i] for i in range(9) for _ in range(counts[i])]
        pb = poisson_binomial(seq)
        for k, m in enumerate(pb):
            dist[k] += weight * m
    mean = sum(k * m for k, m in enumerate(dist))

    return {
        "dist": [round(m, 6) for m in dist],
        "mean_k": round(mean, 3),
        "tbf_samples": samples,
        "tbf_mean": round(sum(samples) / len(samples), 1),
        "inputs": {
            "league_k_rate": round(p_league, 4),
            "starter_hand": starter_hand,
            "shrink_pa": K_SHRINK_PA,
            "park_k_factor": round(park_k_factor, 3) if park_k_factor else None,
            "league_fallback_slots": league_fallbacks,
            "slots": slot_inputs,
        },
    }


def price_line(kdist: dict, line: float) -> dict:
    """Model read on a posted line: calibrated P(over) / P(under) + fair
    probability straight off the distribution shape."""
    raw_over = prob_over(kdist["dist"], line)
    p_over = calibrate(raw_over)
    return {"line": line, "p_over": round(p_over, 4), "p_over_raw": round(raw_over, 4),
            "p_under": round(1 - p_over, 4)}
