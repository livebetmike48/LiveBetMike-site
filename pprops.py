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

log = logging.getLogger("pprops")

MLB_BASE = "https://statsapi.mlb.com/api/v1"

# ---- knobs (own namespace -- the K model's knobs are NOT inherited) ----
P_SHRINK_PA = int(os.getenv("P_SHRINK_PA", "60"))
P_MIN_BATTER_PA = int(os.getenv("P_MIN_BATTER_PA", "40"))
P_MIN_STARTER_TBF = int(os.getenv("P_MIN_STARTER_TBF", "60"))
P_MIN_STARTS = int(os.getenv("P_MIN_STARTS", "3"))
# Opponent workload adjustment: 0 = OFF (exact validated workload mixture,
# opponent-blind, same as the K model). 1 = full. Earns its weight in the
# Lab like every other knob -- default 0 until then.
P_WORKLOAD_WEIGHT = float(os.getenv("P_WORKLOAD_WEIGHT", "0"))
WORKLOAD_CLAMP = (0.85, 1.15)

# Markets: an event set is the ONLY thing that distinguishes them.
MARKETS = {
    "hits": {"events": statcast_api.HIT_EVENTS, "label": "hits allowed",
             "stat_key": "hits"},
    "walks": {"events": statcast_api.BB_EVENTS, "label": "walks allowed",
              "stat_key": "baseOnBalls"},
}


# ---------- league rates (same source + cadence as kmodel.league_k_rate) ----------

_league_cache = {"ts": 0, "rates": None}


def league_rates() -> dict:
    """League per-PA rate for every market, from MLB's real season team
    totals. Cached a day, exactly like the K model's league rate."""
    now = time.time()
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
                      league_pppa: float | None = None) -> dict | None:
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
            "league_fallback_slots": fallbacks,
            "slots": slot_inputs,
        },
    }


def price_line(dist: dict, line: float) -> dict:
    """Model read on a posted line, straight off the distribution shape.
    No calibration curve yet -- each market earns its own in the Lab, and
    an unfitted curve would be a lie, not a default."""
    raw = kmodel.prob_over(dist["dist"], line)
    return {"line": line, "p_over": round(raw, 4), "p_under": round(1 - raw, 4),
            "calibrated": False}


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
        row["workload_factor"] = workload_adjustment(lineup, league_pppa, weight=1.0)
        for m in markets:
            d = prop_distribution(m, lineup, e["rows"], hand, rates[m],
                                  start_game_pks=kmodel.fetch_start_games(team["starter_id"]),
                                  league_pppa=league_pppa)
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
            f"<td><b>{h['mean'] if h else '-'}</b></td>"
            f"<td>{w['mean'] if w else '-'}</td>"
            f"<td>{h['tbf_mean'] if h else '-'}</td>"
            f"<td>{r['pppa'] or '-'}</td>"
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
<th>Hits allowed</th><th>Walks</th><th>TBF</th><th>His P/PA</th><th>Workload x</th>
<th>Lineup</th></tr>
{''.join(rows)}</table>"""
