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

July 27 additive changes (defaults reproduce the validated model exactly):
  - k_distribution gains unknown_slot_rate (default None): unknown lineup
    slots price at this rate when provided (the K Board passes the
    opposing TEAM's real K rate vs the starter's hand) instead of league.
    Backtests never pass it -- every validated path is untouched.
  - K_MIN_TBF_SAMPLE (default 0 = OFF): optional floor on TBF samples
    entering the workload mixture, dropping ultra-short true starts
    (opener games, injury exits). A Lab gauntlet candidate, not a live
    change; set via env K_MIN_TBF_SAMPLE.
"""
import os
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

# Mixture floor: TBF samples BELOW this are dropped from the workload
# mixture when > 0 (never filtering to nothing). Default 0 = OFF = the
# exact validated mixture. Short-real-start candidate ~8-12 TBF.
K_MIN_TBF_SAMPLE = int(os.getenv("K_MIN_TBF_SAMPLE", "0"))


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


def tbf_samples(starter_rows: list[dict],
                start_game_pks: set | None = None) -> list[int]:
    """The starter's REAL batters-faced count for each of his STARTS --
    the workload distribution, straight from his logs.

    July 27 fix: the spec always said start logs, but the grouping
    counted EVERY appearance -- so a converted reliever's mixture was a
    pile of 1-inning stints, projecting ~2-3 K and manufacturing fantasy
    under-edges (the Griffin Jax +59% read). A game only enters the
    mixture if he appeared in the FIRST inning (he started it). Rows
    without inning data fall back to the legacy all-appearances grouping
    -- never silently refusing on missing fields.

    K_MIN_TBF_SAMPLE > 0 additionally drops ultra-short starts (never
    filtering to nothing); at 0 (default) only the start-filter applies."""
    games: dict = {}
    first_inning: dict = {}
    has_inning = False
    for r in starter_rows:
        gpk, date = r.get("game_pk"), r.get("game_date")
        if gpk is None or not date:
            continue
        key = (date, gpk)
        inn = r.get("inning")
        if inn is not None:
            has_inning = True
            try:
                inn = int(inn)
                if key not in first_inning or inn < first_inning[key]:
                    first_inning[key] = inn
            except (TypeError, ValueError):
                pass
        ev = r.get("events")
        if ev and ev not in statcast_api.NON_PA_EVENTS:
            games[key] = games.get(key, 0) + 1
    if start_game_pks is not None and games:
        # authoritative: MLB game log says which games he STARTED.
        # game_pk types normalized both sides (Savant rows can carry
        # str/float pks; the game log gives ints).
        def _pk(v):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None
        norm_starts = {_pk(p) for p in start_game_pks}
        filtered = {k: v for k, v in games.items() if _pk(k[1]) in norm_starts}
        if filtered:
            games = filtered
        else:
            # filter matched NOTHING against real appearances -> data
            # mismatch, not a reliever. Legacy grouping, loudly.
            log.warning("start filter matched 0 of %d games -- game_pk "
                        "mismatch vs game log; using legacy grouping", len(games))
    elif has_inning:
        games = {k: v for k, v in games.items() if first_inning.get(k) == 1}
    samples = sorted(games.values())
    if K_MIN_TBF_SAMPLE > 0:
        kept = [t for t in samples if t >= K_MIN_TBF_SAMPLE]
        if kept:
            samples = kept
    return samples


_starts_cache = {"date": None, "pks": {}}


def fetch_start_games(starter_id: int, before: str | None = None) -> set | None:
    """game_pks of games this pitcher STARTED, from MLB's game log
    (gamesStarted per game -- authoritative, no inference). Cached per
    day. `before` filters to starts strictly before that date so the
    backtest's point-in-time discipline holds. None on ANY failure --
    the mixture then falls back to its legacy grouping, never refusing
    someone because a fetch hiccuped."""
    if not starter_id:
        return None
    today = time.strftime("%Y-%m-%d")
    if _starts_cache["date"] != today:
        _starts_cache.update({"date": today, "pks": {}})
    if starter_id in _starts_cache["pks"]:
        entries = _starts_cache["pks"][starter_id]
    else:
        try:
            season = today[:4]
            data = requests.get(
                f"{MLB_BASE}/people/{starter_id}/stats",
                params={"stats": "gameLog", "group": "pitching",
                        "season": season, "sportId": 1},
                timeout=20).json()
            entries = []
            splits_seen = 0
            for s in (data.get("stats") or []):
                for sp in (s.get("splits") or []):
                    splits_seen += 1
                    st = sp.get("stat") or {}
                    # gamePk location varies by stats-API shape -- check
                    # every place it's known to appear
                    gpk = ((sp.get("game") or {}).get("gamePk")
                           or sp.get("gamePk")
                           or (st.get("game") or {}).get("gamePk")
                           if isinstance(st.get("game"), dict) else
                           ((sp.get("game") or {}).get("gamePk") or sp.get("gamePk")))
                    date = sp.get("date") or ((sp.get("game") or {}).get("date"))
                    if gpk and int(st.get("gamesStarted") or 0) >= 1:
                        try:
                            gpk = int(gpk)
                        except (TypeError, ValueError):
                            continue
                        entries.append((gpk, date or ""))
            _starts_cache["pks"][starter_id] = entries
            log.info("start-games %s: %d starts from %d game-log rows",
                     starter_id, len(entries), splits_seen)
            if splits_seen and not entries:
                log.warning("start-games %s: %d log rows but 0 starts parsed "
                            "-- gamePk location mismatch? sample keys: %s",
                            starter_id, splits_seen,
                            sorted((data.get("stats") or [{}])[0].get("splits", [{}])[0].keys())
                            if (data.get("stats") or [{}])[0].get("splits") else "none")
        except Exception as e:
            log.warning("start-games fetch failed for %s: %s", starter_id, e)
            return None
    if not entries:
        return None  # no starts on record -> legacy fallback (logged above)
    if before:
        return {gpk for gpk, d in entries if d and d < before}
    return {gpk for gpk, _ in entries}


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
                   park_k_factor: float | None = None,
                   unknown_slot_rate: float | None = None,
                   start_game_pks: set | None = None) -> dict | None:
    """The strikeout distribution for one start.

    lineup: 9 entries in batting order -- {'rows': [...], 'side': 'L'/'R',
    'name': str} or None when the slot is unknown (priced at league, or at
    unknown_slot_rate when provided -- the K Board passes the opposing
    team's real K rate vs this hand; backtests never pass it).
    Returns None only when the STARTER's sample is too thin to say
    anything honest.
    """
    s_rows = rows_before(starter_rows, before)
    samples = tbf_samples(s_rows, start_game_pks=start_game_pks)
    if len(samples) < K_MIN_STARTS:
        return None

    fallback_rate = p_league if unknown_slot_rate is None else unknown_slot_rate
    fb_unknown = ("league (slot unknown)" if unknown_slot_rate is None
                  else f"team avg vs {starter_hand}HP (slot unknown)")
    fb_thin = ("league (thin sample)" if unknown_slot_rate is None
               else "team avg (thin sample)")

    slot_probs = []
    slot_inputs = []
    league_fallbacks = 0
    for slot, entry in enumerate(lineup[:9], start=1):
        side = (entry or {}).get("side") or "R"
        s = per_pa_k_rate(s_rows, "stand", side)
        if not s or s["pa"] < K_MIN_STARTER_TBF:
            return None  # starter sample vs this side too thin -- refuse
        s_rate = shrunk(s["k"], s["pa"], p_league)

        b_rate = fallback_rate
        b_info = {"slot": slot, "name": None, "basis": fb_unknown}
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
                          "basis": fb_thin}

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
