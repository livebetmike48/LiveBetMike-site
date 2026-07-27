"""
K model -- per-start strikeout distributions.

Per-hitter K probabilities (log5 vs the starter, shrunk, platoon-correct)
run through the starter's REAL workload distribution to an exact
Poisson-binomial strikeout distribution for the start. No invented
leashes, no composite scores -- every number traces to Savant/MLB rows.

July 27 changes (both additive; defaults reproduce the validated model
byte-for-byte):
1. k_distribution gains unknown_slot_rate (default None). When a lineup
   slot is unknown (not posted / unmatchable), the slot prices at this
   rate instead of league when provided. The K Board passes the OPPOSING
   TEAM's season K rate vs the starter's hand here -- real data, not an
   invented lineup. Backtests never pass it, so every validated path is
   untouched.
2. K_MIN_TBF_SAMPLE (default 0 = OFF): optional floor on the TBF samples
   feeding the workload mixture. The mixture is already built from
   per-START samples only, so relief outings never enter it -- but very
   short true starts (opener games, injury-shortened starts) do, and for
   a pitcher whose role changed they can drag the mixture. This is a
   MODEL change, so it ships as a knob at 0 and must earn weight in the
   Lab like everything else. Env-set for now (K_MIN_TBF_SAMPLE).
"""
import os
import math
import logging

log = logging.getLogger("kmodel")

# ---- knobs (lab._apply_k_config overwrites these from saved config) ----
K_SHRINK_PA = 120
K_MIN_BATTER_PA = 40
K_MIN_STARTER_TBF = 60
K_MIN_STARTS = 3
K_ARSENAL_WEIGHT = 0.0
K_PARK_WEIGHT = 1.0
K_CALIB_WEIGHT = 1.0
K_CALIB_POINTS: list = []

# Mixture floor: TBF samples BELOW this are excluded from the workload
# mixture when > 0. Default 0 = off = the exact validated mixture.
# A "short real start" floor candidate is ~8-12 TBF (~30 pitches).
K_MIN_TBF_SAMPLE = int(os.getenv("K_MIN_TBF_SAMPLE", "0"))


# ------------------------- rate extraction -------------------------

def _sum_rows(rows: list[dict], keys: tuple[str, ...]) -> float:
    total = 0.0
    for r in rows or []:
        for k in keys:
            v = r.get(k)
            if v is not None:
                try:
                    total += float(v)
                except (TypeError, ValueError):
                    pass
                break
    return total


def _k_rate_vs(rows: list[dict], vs_hand: str, shrink_to: float,
               min_pa: int) -> tuple[float | None, float]:
    """(shrunk K/PA vs that hand, raw PA). None when the split is absent.
    Empirical-Bayes shrink: (k + shrink_pa*league) / (pa + shrink_pa)."""
    side_rows = [r for r in rows or []
                 if (r.get("p_throws") or r.get("stand") or "").upper().startswith(vs_hand)]
    if not side_rows:
        side_rows = [r for r in rows or [] if r.get("split", "").upper() == f"VS {vs_hand}HP"
                     or r.get("split", "").upper() == f"VS {vs_hand}HB"]
    pa = _sum_rows(side_rows, ("pa", "abs", "ab"))
    ks = _sum_rows(side_rows, ("strikeout", "so", "k"))
    if pa <= 0:
        return None, 0.0
    rate = (ks + K_SHRINK_PA * shrink_to) / (pa + K_SHRINK_PA)
    return rate, pa


def league_k_rate() -> float:
    """Season league K/PA. The live layer (kboard) caches the real number
    from MLB stats; this fallback keeps the module importable standalone."""
    return _LEAGUE_CACHE.get("rate", 0.221)


_LEAGUE_CACHE: dict = {}


def set_league_k_rate(rate: float):
    _LEAGUE_CACHE["rate"] = float(rate)


# --------------------- workload mixture (per-start) ---------------------

def _tbf_samples(starter_rows: list[dict]) -> list[int]:
    """Per-START batters-faced samples from the starter's game rows.
    Rows are start-only by construction upstream (parlay's per-start log
    fetch) -- relief outings never appear here. K_MIN_TBF_SAMPLE > 0
    additionally drops ultra-short starts (opener games, injury exits)
    from the mixture; at 0 (default) the validated mixture is untouched."""
    samples = []
    for r in starter_rows or []:
        tbf = r.get("tbf") or r.get("batters_faced")
        if tbf is None:
            continue
        try:
            tbf = int(tbf)
        except (TypeError, ValueError):
            continue
        if tbf > 0:
            samples.append(tbf)
    if K_MIN_TBF_SAMPLE > 0:
        kept = [t for t in samples if t >= K_MIN_TBF_SAMPLE]
        if kept:  # never filter down to nothing
            samples = kept
    return samples


def slot_pa_counts(tbf: int) -> list[int]:
    """PA per lineup slot for a given TBF: slot i (0-8) bats
    (tbf - i + 8) // 9 times -- exact batting-order arithmetic."""
    return [max(0, (int(tbf) - i + 8) // 9) for i in range(9)]


def poisson_binomial(probs: list[float]) -> list[float]:
    """Exact distribution of #successes over independent non-identical
    Bernoullis. O(n^2), n <= ~45."""
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for k, m in enumerate(dist):
            nxt[k] += m * (1 - p)
            nxt[k + 1] += m * p
        dist = nxt
    return dist


def prob_over(dist: list[float], line: float) -> float:
    """P(K > line). Half-point semantics: over 5.5 = P(K >= 6)."""
    need = math.floor(line) + 1
    return sum(m for k, m in enumerate(dist) if k >= need)


def calibrate(p: float) -> float:
    """Piecewise-linear correction from the fitted curve, weighted by
    K_CALIB_WEIGHT. Empty curve = identity."""
    if K_CALIB_WEIGHT <= 0 or not K_CALIB_POINTS:
        return p
    pts = K_CALIB_POINTS
    if p <= pts[0][0]:
        corrected = pts[0][1]
    elif p >= pts[-1][0]:
        corrected = pts[-1][1]
    else:
        corrected = p
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= p <= x1:
                t = 0.0 if x1 == x0 else (p - x0) / (x1 - x0)
                corrected = y0 + t * (y1 - y0)
                break
    out = (1 - K_CALIB_WEIGHT) * p + K_CALIB_WEIGHT * corrected
    return min(1.0, max(0.0, out))


def price_line(kdist: dict, line: float) -> dict:
    raw = prob_over(kdist["dist"], line)
    cal = calibrate(raw)
    return {"p_over": round(cal, 4), "p_over_raw": round(raw, 4),
            "p_under": round(1 - cal, 4)}


# ----------------------------- the model -----------------------------

def k_distribution(lineup: list, starter_rows: list[dict], hand: str,
                   p_league: float, before=None, park_k_factor: float | None = None,
                   unknown_slot_rate: float | None = None) -> dict | None:
    """Exact K distribution for one start.

    lineup: 9 entries, each {"rows": [...], "side": "L"/"R", "name": ...}
    or None for an unknown slot. Unknown slots price at unknown_slot_rate
    when provided (the K Board passes the opposing team's K rate vs this
    hand), else league -- the exact validated default.
    """
    if hand not in ("L", "R"):
        return None

    # starter K rate vs each batter side, gated on real sample
    s_rate = {}
    for side in ("L", "R"):
        rate, pa = _k_rate_vs(starter_rows, side, p_league, 0)
        if rate is None or pa < K_MIN_STARTER_TBF:
            return None  # house minimum: refuse thin starters
        s_rate[side] = rate

    # workload mixture from real per-start samples
    samples = _tbf_samples(starter_rows)
    if len(samples) < K_MIN_STARTS:
        return None
    tbf_mean = sum(samples) / len(samples)

    slot_fallback_rate = unknown_slot_rate if unknown_slot_rate is not None else p_league
    slot_basis_unknown = (f"team avg vs {hand}HP (slot unknown)"
                          if unknown_slot_rate is not None else "league (slot unknown)")

    # per-slot K/PA via log5 vs the starter
    slots = []
    fallback = 0
    for i in range(9):
        entry = lineup[i] if i < len(lineup) else None
        if not entry:
            p = _log5(slot_fallback_rate, s_rate[_default_side(hand)], p_league)
            slots.append({"slot": i + 1, "name": None,
                          "p_k_per_pa": round(p, 4), "basis": slot_basis_unknown})
            fallback += 1
            continue
        b_rate, b_pa = _k_rate_vs(entry["rows"], hand, p_league, K_MIN_BATTER_PA)
        if b_rate is None or b_pa < K_MIN_BATTER_PA:
            p = _log5(slot_fallback_rate, s_rate[_default_side(hand)], p_league)
            slots.append({"slot": i + 1, "name": entry.get("name"),
                          "p_k_per_pa": round(p, 4),
                          "basis": ("team avg (thin sample)"
                                    if unknown_slot_rate is not None
                                    else "league (thin sample)")})
            fallback += 1
            continue
        side = entry.get("side") or _default_side(hand)
        p = _log5(b_rate, s_rate.get(side, s_rate[_default_side(hand)]), p_league)
        slots.append({"slot": i + 1, "name": entry.get("name"),
                      "p_k_per_pa": round(p, 4), "basis": f"log5 vs {hand}HP"})

    # park factor (Savant K index, weight-controlled, neutral on absence)
    park = 1.0
    if park_k_factor is not None and K_PARK_WEIGHT > 0:
        park = 1.0 + K_PARK_WEIGHT * (float(park_k_factor) - 1.0)
    slot_probs = [min(0.95, max(0.005, s["p_k_per_pa"] * park)) for s in slots]

    # mixture over the real TBF samples
    mixture_len = max(samples) + 1
    mixture = [0.0] * mixture_len
    for tbf in samples:
        counts = slot_pa_counts(tbf)
        seq = [slot_probs[i] for i in range(9) for _ in range(counts[i])]
        pb = poisson_binomial(seq)
        for k, m in enumerate(pb):
            mixture[k] += m / len(samples)

    mean_k = sum(k * m for k, m in enumerate(mixture))
    return {
        "dist": [round(m, 6) for m in mixture],
        "mean_k": round(mean_k, 3),
        "tbf_mean": round(tbf_mean, 1),
        "inputs": {
            "slots": slots,
            "league_fallback_slots": fallback,
            "park_k_factor": park_k_factor,
            "tbf_samples": len(samples),
        },
    }


def _default_side(hand: str) -> str:
    """Unknown batters modeled as the platoon-common side vs this hand."""
    return "L" if hand == "R" else "R"


def _log5(b_rate: float, p_rate: float, league: float) -> float:
    """Classic log5: batter K rate x pitcher K rate / league."""
    if league <= 0:
        return b_rate
    num = (b_rate * p_rate) / league
    den = num + ((1 - b_rate) * (1 - p_rate)) / (1 - league)
    return num / den if den > 0 else b_rate
