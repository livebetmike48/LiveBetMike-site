"""
K-model arsenal matchup (kmatchup -- NOT matchups.py, which is the
Matchup Board's hitter-vs-starter module; separate file, separate job).

Arsenal matchup: how much this pitcher's ACTUAL pitch mix troubles THESE
hitters, built only from well-sampled pieces.

The composition, and why it's built this way:

  pitcher side  -- his usage% per pitch type TO this batter side. Hundreds
                   to ~1,000 pitches per offering. Solid.
  hitter side   -- his whiff-per-swing against each pitch type, season,
                   all hands. A regular sees ~150-400 swings at his most
                   common pitch types. Usable ONLY with shrinkage.
  combination   -- usage-weighted average of the hitter's (shrunk) whiff
                   rates across the pitches this guy actually throws.

What this deliberately does NOT do: hitter-vs-THIS-pitcher head-to-head.
Most batter-pitcher pairs have under 15 career PA -- the most famous noise
source in baseball analytics, and exactly what flashy matchup boards sell.
Everything here is composed from two independently well-sampled rates.

Hand is NOT split on the hitter's per-pitch rates on purpose: the K model
already prices batter-vs-hand separately, and splitting here would halve
an already thin sample to re-learn something the model has.

No composite scores: every number below is a real rate with its own
denominator shown, and the output is a weighted average of those rates --
traceable end to end, per the house rule.

NOT WIRED INTO THE MODEL. Verification surface only, until a gauntlet run
earns it a weight.
"""
import logging

import statcast_api
import parlay
import kmodel
import csw as csw_mod

log = logging.getLogger("kmatchup")

# Shrinkage strength, in swings. A pitch type the hitter has swung at
# SHRINK_SWINGS times gets weighted half his own rate, half his overall
# rate. 50 is a starting point -- it becomes a knob if this earns a weight.
SHRINK_SWINGS = 50

MIN_USAGE_PCT = 3.0     # ignore pitches he barely throws
MIN_BATTER_SWINGS = 200  # below this, the hitter's overall rate is itself thin


# The math now lives in kmodel (single source of truth) so the page and the
# model can never disagree. kmatchup keeps the lineup walk + rendering.
_swings_whiffs = kmodel._swings_whiffs
batter_pitch_whiffs = kmodel.batter_pitch_whiffs
arsenal_whiff = kmodel.arsenal_whiff
SHRINK_SWINGS = kmodel.MATCHUP_SHRINK_SWINGS
MIN_USAGE_PCT = kmodel.MATCHUP_MIN_USAGE_PCT
MIN_BATTER_SWINGS = kmodel.MATCHUP_MIN_SWINGS


def lineup_matchup(starter_id: int, opp_team_id: int, hand: str,
                   batter_ids: list[int] | None = None) -> dict:
    """The whole lineup against one starter's arsenal. batter_ids optional:
    without it, uses the same proxy lineup the K board prices with (the
    opponent's most recent real order vs this hand)."""
    try:
        s_rows = parlay.get_player_season_rows(int(starter_id), True)
    except Exception as e:
        return {"error": f"starter rows failed: {e}"}
    if not s_rows:
        return {"error": "no pitch rows for this starter"}

    source = "given"
    if not batter_ids:
        proxy = kmodel.fetch_recent_lineup(opp_team_id, hand)
        if not proxy:
            return {"error": "no recent lineup found for this team vs this hand"}
        batter_ids = proxy["batter_ids"]
        source = f"projected (last vs {hand}HP, {proxy['date']})"

    hitters, diffs = [], []
    for bid in batter_ids:
        try:
            b_rows = parlay.get_player_season_rows(int(bid), False)
        except Exception:
            b_rows = None
        if not b_rows:
            hitters.append({"batter_id": bid, "note": "no pitch rows"})
            continue
        side = statcast_api.effective_bat_side(
            (b_rows[0].get("stand") or "R"), hand)
        m = arsenal_whiff(s_rows, b_rows, side)
        if not m:
            hitters.append({"batter_id": bid, "note": "not enough data"})
            continue
        name = (b_rows[0].get("player_name") or "").strip()
        if "," in name:
            last, first = [p.strip() for p in name.split(",", 1)]
            name = f"{first} {last}"
        hitters.append({"batter_id": bid, "name": name or str(bid),
                        "side": side, **m})
        diffs.append(m["diff_pts"])

    team = None
    if diffs:
        team = {"hitters_priced": len(diffs),
                "avg_diff_pts": round(sum(diffs) / len(diffs), 2),
                "worse_vs_this_arsenal": sum(1 for d in diffs if d > 0)}
    return {"starter_id": starter_id, "hand": hand, "lineup_source": source,
            "team": team, "hitters": hitters,
            "note": ("expected whiff = the hitter's own whiff rates per pitch "
                     "type, shrunk toward his overall rate by sample, then "
                     "weighted by this pitcher's real usage to that side. "
                     "Positive diff = this arsenal misses his bat more than "
                     "average pitching does.")}


def slate_matchups(date: str | None = None) -> dict:
    """Every probable starter today vs the lineup he'll actually face."""
    slate = csw_mod.slate_csw(date)   # reuses the schedule walk + date logic
    if slate.get("error"):
        return slate
    import requests
    out = []
    try:
        sched = requests.get(f"{csw_mod.MLB_BASE}/schedule",
                             params={"sportId": 1, "date": slate["date"],
                                     "hydrate": "probablePitcher,team"},
                             timeout=20).json()
    except Exception as e:
        return {"error": f"schedule fetch failed: {e}"}
    for d in (sched.get("dates") or []):
        for g in (d.get("games") or []):
            teams = g.get("teams") or {}
            for side, opp_side in (("home", "away"), ("away", "home")):
                t = teams.get(side) or {}
                opp = teams.get(opp_side) or {}
                p = t.get("probablePitcher") or {}
                pid = p.get("id")
                opp_id = ((opp.get("team") or {}).get("id"))
                if not pid or not opp_id:
                    continue
                try:
                    hand = parlay.get_starter_hand(pid)
                except Exception:
                    hand = None
                if hand not in ("L", "R"):
                    continue
                m = lineup_matchup(pid, opp_id, hand)
                out.append({
                    "starter": p.get("fullName", "?"),
                    "team": ((t.get("team") or {}).get("abbreviation") or "?"),
                    "opp": ((opp.get("team") or {}).get("abbreviation") or "?"),
                    "hand": hand,
                    "error": m.get("error"),
                    "team_summary": m.get("team"),
                    "lineup_source": m.get("lineup_source"),
                })
    out.sort(key=lambda r: -((r.get("team_summary") or {}).get("avg_diff_pts") or -99))
    return {"date": slate["date"], "n": len(out), "starters": out}


def matchup_html(starter_name: str, date: str | None = None) -> str:
    """One starter's lineup matchup as a readable table."""
    import requests
    found = statcast_api.resolve_player(starter_name)
    if not found:
        return f"<p>no player found for {starter_name!r}</p>"
    slate_date = date
    try:
        import time as _t
        et = _t.gmtime(_t.time() - 4 * 3600)
        slate_date = slate_date or _t.strftime("%Y-%m-%d", et)
        sched = requests.get(f"{csw_mod.MLB_BASE}/schedule",
                             params={"sportId": 1, "date": slate_date,
                                     "hydrate": "probablePitcher,team"},
                             timeout=20).json()
    except Exception as e:
        return f"<p>schedule fetch failed: {e}</p>"
    opp_id, opp_abbr = None, "?"
    for d in (sched.get("dates") or []):
        for g in (d.get("games") or []):
            teams = g.get("teams") or {}
            for side, o in (("home", "away"), ("away", "home")):
                p = ((teams.get(side) or {}).get("probablePitcher") or {})
                if p.get("id") == found["id"]:
                    opp_id = (((teams.get(o) or {}).get("team") or {}).get("id"))
                    opp_abbr = (((teams.get(o) or {}).get("team") or {})
                                .get("abbreviation") or "?")
    if not opp_id:
        return (f"<p>{found['name']} isn't a probable starter on {slate_date}. "
                f"Try a starter from today's slate.</p>")
    hand = found.get("pitch_hand") or parlay.get_starter_hand(found["id"])
    m = lineup_matchup(found["id"], opp_id, hand)
    if m.get("error"):
        return f"<p>{m['error']}</p>"
    rows = []
    for h in m["hitters"]:
        if h.get("note"):
            rows.append(f"<tr><td style='text-align:left'>{h.get('name', h['batter_id'])}</td>"
                        f"<td colspan='5'><i>{h['note']}</i></td></tr>")
            continue
        d = h["diff_pts"]
        col = "#4caf7d" if d > 0 else ("#d7483a" if d < 0 else "#8b95a1")
        comps = " · ".join(
            f"{c['pitch']} {c['usage_pct']}%→{c['shrunk_whiff_pct']}%"
            + (f"<small>(raw {c['batter_whiff_pct']}, {c['batter_swings']} sw)</small>"
               if c['batter_swings'] else "<small>(no sample)</small>")
            for c in h["components"])
        rows.append(
            f"<tr><td style='text-align:left'>{h['name']}{' ⚠️' if h.get('thin') else ''}</td>"
            f"<td>{h['side']}</td>"
            f"<td><b>{h['expected_whiff_pct']}</b></td><td>{h['overall_whiff_pct']}</td>"
            f"<td style='color:{col};font-weight:700'>{d:+.1f}</td>"
            f"<td style='text-align:left'><small>{comps}</small></td></tr>")
    t = m["team"] or {}
    return f"""<!doctype html><meta charset="utf-8">
<title>Matchup - {found['name']}</title>
<style>
 body{{background:#0f1216;color:#e6e9ee;font:14px/1.55 -apple-system,Segoe UI,Inter,sans-serif;
      padding:18px;max-width:1100px;margin:auto}}
 h1{{font-size:17px;margin:0 0 2px}} p.note{{color:#8b95a1;font-size:12.5px;margin:2px 0 14px}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{padding:5px 8px;text-align:center;border-bottom:1px solid #222831;vertical-align:top}}
 th{{color:#8b95a1;font-weight:600;font-size:12px}} small{{color:#6d7681}}
</style>
<h1>{found['name']} ({hand}HP) vs {opp_abbr} — {slate_date}</h1>
<p class="note">Lineup: {m['lineup_source']} · team avg
 <b>{t.get('avg_diff_pts', '-')}</b> pts, {t.get('worse_vs_this_arsenal', '-')}/{t.get('hitters_priced', '-')}
 hitters whiff more vs this mix than vs average pitching.<br>{m['note']}
 ⚠️ = fewer than {MIN_BATTER_SWINGS} total swings, treat with suspicion.</p>
<table>
<tr><th style='text-align:left'>Hitter</th><th>Side</th><th>Exp whiff%</th>
<th>His overall</th><th>Diff</th><th style='text-align:left'>Per pitch (usage → shrunk whiff)</th></tr>
{''.join(rows)}</table>"""
