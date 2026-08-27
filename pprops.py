"""
Pitcher props -- hits allowed, walks allowed, and the shared engine under
them. SEPARATE from the K model by design: kmodel is fine-tuned, live, and
under a running forward-log tripwire, so this module IMPORTS its validated
primitives and never edits it.

What's reused verbatim from kmodel (all of it already backtested):
  log5           -- pitcher rate x batter rate / league
  shrunk         -- empirical-Bayes shrinkage (own pseudo-count passed in)
  tbf_samples    -- the starter's REAL start-log workload distribution
  slot_pa_counts -- batting-order PA arithmetic
  poisson_binomial / prob_over -- the exact distribution, no normal approx
  fetch_start_games / fetch_recent_lineup / rows_before

What's new here:
  per_pa_rate    -- kmodel.per_pa_k_rate generalized to ANY event set
  league_rates   -- league per-PA hit/walk rates from MLB's team totals
  prop_distribution -- k_distribution's shape for an arbitrary market
  pitches_per_pa + workload_adjustment -- OPPONENT-ADJUSTED WORKLOAD, the
                    piece the K model doesn't have (see below)

Honest scope, stated up front:
  hits / walks   -- same math as strikeouts. Per-PA binary outcomes.
  outs (IP)      -- derived from TBF and baserunners; buildable next.
  earned runs    -- NOT here and not coming from this engine. ER depends on
                    SEQUENCING (three singles in a row vs spread over three
                    innings are wildly different outcomes with identical
                    per-PA rates). Independent-PA math cannot price it
                    honestly; that needs a base-state simulation.

Nothing here touches kmodel, kboard, the live K board, or the frozen K
forward log.
"""
import os
import time
import logging

import requests

import parlay
import statcast_api
import kmodel
import kboard

try:
    import parks
except ImportError:
    parks = None

log = logging.getLogger("pprops")

MLB_BASE = "https://statsapi.mlb.com/api/v1"

# ---- knobs (own namespace -- the K model's knobs are NOT inherited) ----
P_SHRINK_PA = int(os.getenv("P_SHRINK_PA", "60"))
P_MIN_BATTER_PA = int(os.getenv("P_MIN_BATTER_PA", "40"))
P_MIN_STARTER_TBF = int(os.getenv("P_MIN_STARTER_TBF", "60"))
P_MIN_STARTS = int(os.getenv("P_MIN_STARTS", "3"))
# NOTE (July 31, retracted the same day it was added): a short workload
# distribution is NOT automatically a bug. An OPENER really does face
# three or four batters, and 3.3 TBF is the honest projection for him --
# refusing it deletes an accurate read. The Griffin Jax failure was a
# different thing entirely: a reliever being priced as a STARTER, so a
# relief-length workload met a line set for a starter's. The workload
# distribution is the prediction; when it says "opener", believe it.
# Opponent workload adjustment: 0 = OFF (exact validated workload mixture,
# opponent-blind, same as the K model). 1 = full. Earns its weight in the
# Lab like every other knob -- default 0 until then.
P_WORKLOAD_WEIGHT = float(os.getenv("P_WORKLOAD_WEIGHT", "0"))
WORKLOAD_CLAMP = (0.85, 1.15)
# Park factor for HITS, from parks.factor_for -- Savant's official
# index_hits, the same source and the same neutral-on-failure honesty the
# K model already uses for strikeouts. Applied per-slot exactly the way
# kmodel applies its K factor, with the same 0.85-1.15 clamp.
#
# Defaulted ON (1.0) rather than off, and here's the reasoning: a model
# that ignores Coors is KNOWABLY wrong, and this isn't a modeling
# invention -- it's MLB's own published adjustment, already trusted in the
# K model at weight 1. It stays env-tunable so the first hits backtest can
# sweep 0 vs 1 and settle it with a receipt rather than an argument.
P_PARK_WEIGHT = float(os.getenv("P_PARK_WEIGHT", "1"))
PARK_CLAMP = (0.85, 1.15)
# Walks are deliberately park-NEUTRAL: a walk is a pitcher-and-batter
# event, not a dimensions event, so there's no honest factor to apply.
PARK_MARKETS = {"hits"}


def park_factor(venue: str | None, market: str,
                year: int | None = None) -> float | None:
    """Official Savant park factor for a market, or None when unknown --
    so 'neutral park' and 'no data' never get confused. year=None = the
    current season; a past year uses that season's own factors."""
    if not venue or parks is None or market not in PARK_MARKETS:
        return None
    try:
        f = parks.factor_for(venue, year=year)
    except TypeError:
        f = parks.factor_for(venue)         # older parks.py, live path
    except Exception as e:
        log.warning("park factor lookup failed for %s: %s", venue, e)
        return None
    # SANITY CLAMP (the 2023 lesson): a real MLB park factor lives well
    # inside [0.7, 1.3]. Anything outside is broken upstream data -- the
    # 2023 Savant pull poisoned every park-ON hits read to a 0.35 Brier
    # while park0 sailed at 0.18. Bad data prices park-NEUTRAL and says
    # so in the log; it never silently multiplies into the model again.
    try:
        f = float(f)
    except (TypeError, ValueError):
        return None
    if not (0.7 <= f <= 1.3):
        log.warning("park factor REJECTED for %s (%s, year=%s): %.3f "
                    "outside [0.7, 1.3] — pricing park-neutral", venue,
                    market, year, f)
        return None
    return f

# Markets: an event set is the ONLY thing that distinguishes them.
MARKETS = {
    "hits": {"events": statcast_api.HIT_EVENTS, "label": "hits allowed",
             "stat_key": "hits"},
    "walks": {"events": statcast_api.BB_EVENTS, "label": "walks allowed",
              "stat_key": "baseOnBalls"},
}

# Strikeouts are NOT in MARKETS on purpose. The K model is further along
# than this engine -- calibration curve fitted from 2,935 graded
# predictions, park factor, team-rate slots for unknown lineups, a
# validated forward log -- so pricing K here with the generic engine
# would put a SECOND, worse strikeout number on the same site. Instead
# the props view CALLS kmodel directly (k_projection below), so the K
# column and the K Board are the same number by construction. One model,
# shown in two places -- which is the endpoint Mike wants anyway: every
# prop in one spot, with K keeping its own clean section until the rest
# catch up.


# ---------- league rates (same source + cadence as kmodel.league_k_rate) ----------

_league_cache = {"ts": 0, "rates": None}


_league_year_cache: dict = {}


def league_rates(season: int | None = None) -> dict:
    """League per-PA rate for every market, from MLB's real season team
    totals. season=None = current (live path, its own cache). A past
    season uses that YEAR'S totals -- 2024 walks are never priced against
    2026's league."""
    now = time.time()
    if season and int(season) != int(time.strftime("%Y")):
        season = int(season)
        hit = _league_year_cache.get(season)
        if hit and now - hit["ts"] < 86400 * 30:
            return hit["rates"]
        resp = requests.get(
            f"{MLB_BASE}/teams/stats",
            params={"season": season, "group": "hitting",
                    "stats": "season", "sportId": 1}, timeout=20)
        resp.raise_for_status()
        totals: dict = {}
        pa = 0
        for split in resp.json()["stats"][0]["splits"]:
            stat = split.get("stat", {})
            pa += int(stat.get("plateAppearances", 0))
            for market, cfg in MARKETS.items():
                totals[market] = totals.get(market, 0) + int(stat.get(cfg["stat_key"], 0))
        if pa == 0:
            raise RuntimeError(f"league totals unavailable for {season}")
        rates = {m: totals[m] / pa for m in MARKETS}
        _league_year_cache[season] = {"ts": now, "rates": rates}
        return rates
    if _league_cache["rates"] and now - _league_cache["ts"] < 86400:
        return _league_cache["rates"]
    resp = requests.get(
        f"{MLB_BASE}/teams/stats",
        params={"season": time.strftime("%Y"), "group": "hitting",
                "stats": "season", "sportId": 1},
        timeout=20)
    resp.raise_for_status()
    totals: dict = {}
    pa = 0
    for split in resp.json()["stats"][0]["splits"]:
        stat = split.get("stat", {})
        pa += int(stat.get("plateAppearances", 0))
        for market, cfg in MARKETS.items():
            totals[market] = totals.get(market, 0) + int(stat.get(cfg["stat_key"], 0))
    if pa == 0:
        raise RuntimeError("league totals unavailable")
    rates = {m: totals[m] / pa for m in MARKETS}
    _league_cache.update({"ts": now, "rates": rates})
    log.info("League per-PA rates over %d PA: %s", pa,
             {m: round(r, 4) for m, r in rates.items()})
    return rates


# ---------- generic per-PA rate ----------

def per_pa_rate(rows: list[dict], split_col: str, split_val: str,
                events: set) -> dict | None:
    """kmodel.per_pa_k_rate, generalized. Same validated PA accounting --
    NON_PA_EVENTS excluded so runner events never end a plate appearance."""
    pa = hits = 0
    for r in rows:
        if r.get(split_col) != split_val:
            continue
        ev = r.get("events")
        if not ev or ev in statcast_api.NON_PA_EVENTS:
            continue
        pa += 1
        if ev in events:
            hits += 1
    if pa == 0:
        return None
    return {"pa": pa, "n": hits, "rate": hits / pa}


# ---------- pitches per plate appearance ----------

def pitches_per_pa(rows: list[dict]) -> dict | None:
    """P/PA from raw rows -- pitches divided by plate appearances. Works
    for a pitcher (how many pitches he spends per batter) and for a hitter
    (how many he makes a pitcher throw). Same quantity, both sides."""
    pa = sum(1 for r in rows
             if r.get("events") and r.get("events") not in statcast_api.NON_PA_EVENTS)
    if pa == 0 or not rows:
        return None
    return {"pitches": len(rows), "pa": pa, "pppa": len(rows) / pa}


def league_pitches_per_pa(rows_lists: list) -> float | None:
    """League P/PA from any decent sample of PITCHER rows. Every pitch
    thrown is a pitch seen, so the pitcher side and the hitter side
    measure the same league constant -- computed, never assumed."""
    pitches = pa = 0
    used = 0
    for rows in rows_lists:
        s = pitches_per_pa(rows or [])
        if not s or s["pa"] < 100:
            continue
        pitches += s["pitches"]
        pa += s["pa"]
        used += 1
    if used < 5 or pa == 0:
        log.info("league P/PA: only %d usable samples -- not computing", used)
        return None
    return pitches / pa


def workload_adjustment(lineup: list, league_pppa: float | None,
                        weight: float = None) -> float:
    """How much this LINEUP stretches or shortens the starter's outing.

    The K model's workload mixture is opponent-blind: it's his own start
    lengths, averaged over whoever he happened to face. But a patient
    lineup makes him spend more pitches per batter, so he reaches his
    limit sooner and faces FEWER batters -- which moves every pitcher prop
    at once, since TBF multiplies all of them.

    factor = league P/PA / this lineup's P/PA, damped by the knob and
    clamped. Patient lineup (high P/PA) -> factor < 1 -> shorter outing.
    Returns 1.0 (no change) whenever the inputs aren't solid."""
    w = P_WORKLOAD_WEIGHT if weight is None else weight
    if w <= 0 or not league_pppa:
        return 1.0
    pitches = pa = 0
    for entry in (lineup or []):
        if not entry:
            continue
        s = pitches_per_pa(entry.get("rows") or [])
        if s and s["pa"] >= 40:
            pitches += s["pitches"]
            pa += s["pa"]
    if pa < 200 or not pitches:
        return 1.0
    lineup_pppa = pitches / pa
    if lineup_pppa <= 0:
        return 1.0
    raw = league_pppa / lineup_pppa
    factor = 1 + w * (raw - 1)
    return min(max(factor, WORKLOAD_CLAMP[0]), WORKLOAD_CLAMP[1])


# ---------- the distribution ----------

def prop_distribution(market: str, lineup: list, starter_rows: list[dict],
                      starter_hand: str, p_league: float,
                      before: str | None = None,
                      unknown_slot_rate: float | None = None,
                      start_game_pks: set | None = None,
                      league_pppa: float | None = None,
                      park_factor_value: float | None = None) -> dict | None:
    """k_distribution's exact shape for an arbitrary market.

    Per-slot probability = log5(batter rate vs hand, starter rate vs side,
    league), then the exact Poisson-binomial mixed over the starter's real
    workload distribution. Returns None only when the STARTER's sample is
    too thin to say anything honest."""
    cfg = MARKETS.get(market)
    if not cfg:
        return {"error": f"unknown market {market!r}"}
    events = cfg["events"]
    s_rows = kmodel.rows_before(starter_rows, before)
    samples = kmodel.tbf_samples(s_rows, start_game_pks=start_game_pks)
    if len(samples) < P_MIN_STARTS:
        return None

    factor = workload_adjustment(lineup, league_pppa)
    if factor != 1.0:
        samples = [max(1, int(round(t * factor))) for t in samples]

    fallback = p_league if unknown_slot_rate is None else unknown_slot_rate
    slot_probs, slot_inputs, fallbacks = [], [], 0
    for slot, entry in enumerate(lineup[:9], start=1):
        side = (entry or {}).get("side") or "R"
        s = per_pa_rate(s_rows, "stand", side, events)
        if not s or s["pa"] < P_MIN_STARTER_TBF:
            return None
        s_rate = kmodel.shrunk(s["n"], s["pa"], p_league, pseudo=P_SHRINK_PA)

        b_rate = fallback
        basis = "league (slot unknown)" if unknown_slot_rate is None else "team avg (slot unknown)"
        name = None
        if entry:
            name = entry.get("name")
            b_rows = kmodel.rows_before(entry["rows"], before)
            b = per_pa_rate(b_rows, "p_throws", starter_hand, events)
            if b and b["pa"] >= P_MIN_BATTER_PA:
                b_rate = kmodel.shrunk(b["n"], b["pa"], p_league, pseudo=P_SHRINK_PA)
                basis = f"{b['pa']} PA vs {starter_hand}HP"
            else:
                fallbacks += 1
                basis = "league (thin sample)"
        p = kmodel.log5(b_rate, s_rate, p_league)
        if park_factor_value and P_PARK_WEIGHT > 0:
            pf = max(PARK_CLAMP[0], min(PARK_CLAMP[1], park_factor_value))
            p = min(p * (pf ** P_PARK_WEIGHT), 0.9)
        slot_probs.append(p)
        slot_inputs.append({"slot": slot, "name": name, "basis": basis,
                            "p_per_pa": round(p, 4)})

    weight = 1.0 / len(samples)
    dist = [0.0] * (max(samples) + 1)
    for tbf in samples:
        counts = kmodel.slot_pa_counts(tbf)
        seq = [slot_probs[i] for i in range(9) for _ in range(counts[i])]
        pb = kmodel.poisson_binomial(seq)
        for k, m in enumerate(pb):
            dist[k] += weight * m
    mean = sum(k * m for k, m in enumerate(dist))
    return {
        "market": market, "label": cfg["label"],
        "dist": [round(m, 6) for m in dist],
        "mean": round(mean, 3),
        "tbf_samples": samples,
        "tbf_mean": round(sum(samples) / len(samples), 1),
        "inputs": {
            "league_rate": round(p_league, 4),
            "starter_hand": starter_hand,
            "shrink_pa": P_SHRINK_PA,
            "workload_factor": round(factor, 3),
            "park_factor": (round(park_factor_value, 3)
                            if park_factor_value else None),
            "park_weight": P_PARK_WEIGHT if park_factor_value else None,
            "league_fallback_slots": fallbacks,
            "slots": slot_inputs,
        },
    }


def k_projection(lineup: list, starter_rows: list[dict], starter_hand: str,
                 start_game_pks: set | None = None,
                 park_k_factor: float | None = None,
                 unknown_slot_rate: float | None = None) -> dict | None:
    """Strikeouts, straight from the live K model -- same call the K Board
    makes, same knobs, same fitted calibration curve. Never a reimplementation."""
    kd = kmodel.k_distribution(lineup, starter_rows, starter_hand,
                               kmodel.league_k_rate(),
                               before=None, park_k_factor=park_k_factor,
                               unknown_slot_rate=unknown_slot_rate,
                               start_game_pks=start_game_pks)
    if not kd:
        return None
    return {"mean": kd["mean_k"], "tbf_mean": kd["tbf_mean"],
            "calibrated": True, "source": "K model (live)",
            "over_line": {str(l + 0.5): kmodel.price_line(kd, l + 0.5)["p_over"]
                          for l in range(2, 10)}}


# Per-market calibration curves, fit by ppbacktest from REAL raw runs
# (>=100-pred buckets, within-year for season work). Empty = raw, honestly
# flagged. Same piecewise shape as the K model's curve.
P_CALIB: dict[str, list] = {"hits": [], "walks": []}


def calibrate(market: str, p: float) -> float:
    pts = P_CALIB.get(market) or []
    if not pts:
        return p
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    if p <= xs[0]:
        return max(0.0, min(1.0, ys[0]))
    if p >= xs[-1]:
        return max(0.0, min(1.0, ys[-1]))
    for i in range(1, len(xs)):
        if p <= xs[i]:
            t = (p - xs[i-1]) / (xs[i] - xs[i-1]) if xs[i] != xs[i-1] else 0
            return max(0.0, min(1.0, ys[i-1] + t * (ys[i] - ys[i-1])))
    return p


def price_line(dist: dict, line: float, market: str | None = None) -> dict:
    """Model read on a posted line. A market WITH a fitted curve gets it
    applied (calibrated: true); no curve = raw, honestly flagged."""
    raw = kmodel.prob_over(dist["dist"], line)
    p = calibrate(market, raw) if market else raw
    return {"line": line, "p_over": round(p, 4), "p_under": round(1 - p, 4),
            "raw_p_over": round(raw, 4),
            "calibrated": bool(market and P_CALIB.get(market))}


# ---------------------------------------------------------------------------
# Verification surface. Same discipline the CSW work used: look at the
# numbers on a real slate BEFORE anything is backtested or priced. Read
# only -- no odds credits, no logging, no effect on the K board.
# ---------------------------------------------------------------------------

def slate_projections(offset: int = 0, markets: tuple = ("hits", "walks")) -> dict:
    """Every modelable starter on the slate with his projected hits and
    walks allowed, built exactly like the K board builds strikeouts:
    posted lineup when it's up, the board's own projection when it isn't,
    each hitter priced on his real rate vs this hand.

    The league P/PA constant is computed from THIS slate's starters -- a
    real sample, never an assumed 3.90 -- and only used if enough of them
    qualify."""
    offset = 1 if offset == 1 else 0
    date = parlay.et_date_str(offset)
    try:
        rates = league_rates()
    except Exception as e:
        return {"error": f"league rates unavailable: {e}", "date": date}

    slate = kboard._slate(date)
    # pass 1: collect starter rows (also gives us the league P/PA sample)
    loaded = []
    for g in slate:
        orders = kboard._lineup_order(g["game_pk"])
        orders = orders if isinstance(orders, dict) else {}
        for side, opp_side in (("home", "away"), ("away", "home")):
            team, opp = g["teams"][side], g["teams"][opp_side]
            if not team.get("starter_id"):
                continue
            try:
                hand = parlay.get_starter_hand(team["starter_id"])
            except Exception:
                hand = None
            if hand not in ("L", "R"):
                continue
            try:
                s_rows = parlay.get_player_season_rows(team["starter_id"], True)
            except Exception:
                s_rows = []
            if not s_rows:
                continue
            loaded.append({"game": g, "team": team, "opp": opp, "hand": hand,
                           "rows": s_rows, "orders": orders, "side": side,
                           "opp_side": opp_side})
    league_pppa = league_pitches_per_pa([e["rows"] for e in loaded])

    out = []
    for e in loaded:
        team, opp, hand = e["team"], e["opp"], e["hand"]
        order = list((e["orders"].get(e["opp_side"]) or [])[:9])
        source = "posted"
        if not order:
            proxy = kmodel.fetch_recent_lineup(opp.get("id"), hand)
            if proxy:
                order, source = proxy["batter_ids"], f"projected ({proxy['date']})"
            else:
                source = "team avg"
        lineup, known = kboard._build_lineup(order)
        row = {"starter": team.get("starter_name"), "starter_id": team.get("starter_id"),
               "team": team.get("abbrev"), "opp": opp.get("abbrev"), "hand": hand,
               "lineup_source": source, "known_slots": known}
        ppa = pitches_per_pa(e["rows"])
        row["pppa"] = round(ppa["pppa"], 2) if ppa else None
        try:
            row["strikeouts"] = k_projection(
                lineup, e["rows"], hand,
                start_game_pks=kmodel.fetch_start_games(team["starter_id"]),
                park_k_factor=kboard._park_k((e["game"] or {}).get("venue")))
        except Exception as exc:
            log.warning("pprops: K projection failed for %s: %s",
                        team.get("starter_name"), exc)
            row["strikeouts"] = None
        row["workload_factor"] = round(
            workload_adjustment(lineup, league_pppa, weight=1.0), 3)
        venue = (e["game"] or {}).get("venue")
        row["venue"] = venue
        row["park_hits"] = park_factor(venue, "hits")
        for m in markets:
            d = prop_distribution(m, lineup, e["rows"], hand, rates[m],
                                  start_game_pks=kmodel.fetch_start_games(team["starter_id"]),
                                  league_pppa=league_pppa,
                                  park_factor_value=park_factor(venue, m))
            if not d or d.get("error"):
                row[m] = None
                continue
            row[m] = {"mean": d["mean"], "tbf_mean": d["tbf_mean"],
                      "over_line": {str(l + 0.5): round(kmodel.prob_over(d["dist"], l + 0.5), 3)
                                    for l in range(2, 9)}}
        out.append(row)
    out.sort(key=lambda r: -((r.get("hits") or {}).get("mean") or -1))
    return {"date": date, "league_pppa": round(league_pppa, 3) if league_pppa else None,
            "league_rates": {m: round(r, 4) for m, r in rates.items()},
            "n": len(out), "starters": out,
            "note": ("BETA — engine only, never backtested, no calibration curve, "
                     "nothing logged. Compare the means against your own read "
                     "before trusting any of it.")}


def slate_projections_html(offset: int = 0) -> str:
    d = slate_projections(offset)
    if d.get("error"):
        return f"<p>{d['error']}</p>"
    rows = []
    for r in d["starters"]:
        h, w = r.get("hits"), r.get("walks")
        if not h and not w:
            rows.append(f"<tr><td style='text-align:left'>{r['starter']}</td>"
                        f"<td>{r['team']}</td><td>{r['opp']}</td>"
                        f"<td colspan='6'><i>no read — house minimums</i></td></tr>")
            continue
        lu = ("posted" if r["lineup_source"] == "posted"
              else f"<span style='color:#e0a12f'>{r['lineup_source']}</span>")
        rows.append(
            f"<tr><td style='text-align:left'>{r['starter']}</td><td>{r['team']}</td>"
            f"<td>{r['opp']}</td><td>{r['hand']}</td>"
            f"<td>{(r.get('strikeouts') or {}).get('mean', '-')}</td>"
            f"<td><b>{h['mean'] if h else '-'}</b></td>"
            f"<td>{w['mean'] if w else '-'}</td>"
            f"<td>{h['tbf_mean'] if h else '-'}</td>"
            f"<td>{r['pppa'] or '-'}</td>"
            f"<td>{r.get('park_hits') or '-'}</td>"
            f"<td>{r['workload_factor']}</td><td>{lu}</td></tr>")
    return f"""<!doctype html><meta charset="utf-8">
<title>Pitcher props (BETA) - {d['date']}</title>
<style>
 body{{background:#0f1216;color:#e6e9ee;font:14px/1.55 -apple-system,Segoe UI,Inter,sans-serif;
      padding:18px;max-width:1050px;margin:auto}}
 h1{{font-size:17px;margin:0 0 2px}} p.note{{color:#8b95a1;font-size:12.5px;margin:2px 0 14px}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{padding:5px 8px;text-align:center;border-bottom:1px solid #222831}}
 th{{color:#8b95a1;font-weight:600;font-size:12px}}
</style>
<h1>Pitcher props — {d['date']} — {d['n']} starters</h1>
<p class="note">{d['note']}<br>
League per-PA: hits {d['league_rates'].get('hits')} · walks {d['league_rates'].get('walks')} ·
league P/PA {d['league_pppa'] or 'n/a'} (computed from this slate's starters, not assumed).
Workload factor is what the opponent-adjustment WOULD do at full weight — it is currently
OFF in the model.</p>
<table>
<tr><th style="text-align:left">Starter</th><th>Tm</th><th>vs</th><th>Hand</th>
<th>Ks</th><th>Hits allowed</th><th>Walks</th><th>TBF</th><th>His P/PA</th>
<th>Park (hits)</th><th>Workload x</th><th>Lineup</th></tr>
{''.join(rows)}</table>"""
