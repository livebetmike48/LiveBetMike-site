"""
CSW% (Called Strikes + Whiffs) from the pitch-level rows we already pull.

CSW = (called strikes + whiffs) / total pitches -- Pollack/Fast, Pitcher
List, 2018. It is NOT a Baseball Savant column: Savant serves the raw
pitch descriptions, and CSW is an aggregation of them. FanGraphs
publishes the leaderboard, which is what these numbers should be checked
against.

Definitional care (this is the whole ballgame for matching a public
number), settled empirically on 2026-07-29 against FanGraphs:

  CSW whiffs = swinging_strike + swinging_strike_blocked ONLY.
  FOUL TIPS ARE NOT COUNTED.

Verified on Burns / Misiorowski / Luzardo: our called-strike rate matched
FanGraphs exactly (13.8 / 17.0 / 17.9) while our whiff rate ran 1.0-1.6
points hot; removing foul tips reproduced all nine of their figures to
the decimal (implied 23 / 29 / 20 foul tips = 1.0-1.6% of pitches).

Savant's OWN Whiff% is a different metric and DOES count foul tips
(validated years ago against Soto: 17.3% with, 15.4% without). Both live
here, deliberately named apart, so the two can never be confused:
  swstr_pct           -- swinging strikes per PITCH   (FanGraphs SwStr%)
  whiff_pct_of_swings -- Savant whiffs per SWING      (Savant Whiff%)

Nothing here touches the model. It exists so the input can be verified
against FanGraphs BEFORE it earns a gauntlet run.
"""
import logging

import requests

import statcast_api
import parlay

MLB_BASE = "https://statsapi.mlb.com/api/v1"

log = logging.getLogger("csw")

# The other half of CSW. Savant's description field for a taken strike.
CALLED_DESCRIPTIONS = {"called_strike"}

# CSW's whiff half: swinging strikes only. Deliberately NOT
# statcast_api.WHIFF_DESCRIPTIONS, which additionally counts foul tips and
# missed bunts -- correct for Savant's Whiff%, wrong for CSW (proven
# against FanGraphs, see module docstring).
CSW_WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}


def csw_stats(rows: list[dict]) -> dict | None:
    """CSW over any subset of pitch rows. Denominator = ALL pitches in the
    subset (CSW's defining choice -- unlike whiff%, which is per swing)."""
    if not rows:
        return None
    pitches = len(rows)
    called = sum(1 for r in rows if r.get("description") in CALLED_DESCRIPTIONS)
    sw_str = sum(1 for r in rows if r.get("description") in CSW_WHIFF_DESCRIPTIONS)
    savant_whiffs = sum(1 for r in rows
                        if r.get("description") in statcast_api.WHIFF_DESCRIPTIONS)
    swings = sum(1 for r in rows if r.get("description") in statcast_api.SWING_DESCRIPTIONS)
    out = {
        "pitches": pitches,
        "called_strikes": called,
        "swinging_strikes": sw_str,
        "foul_tips": savant_whiffs - sw_str,   # the FanGraphs gap, made visible
        "csw_pct": round((called + sw_str) / pitches * 100, 1),
        "called_pct": round(called / pitches * 100, 1),
        "swstr_pct": round(sw_str / pitches * 100, 1),   # FanGraphs SwStr%
    }
    if swings:
        # Savant's "Whiff%" -- per SWING and foul tips INCLUDED. A different
        # metric from the two above, kept beside them so nobody mixes them up.
        out["whiff_pct_of_swings"] = round(savant_whiffs / swings * 100, 1)
    return out


def csw_by_pitch_type(rows: list[dict], min_pitches: int = 25) -> dict:
    """CSW per pitch type, sorted by usage. This is the arsenal-v2 fuel:
    which specific pitch is actually generating the free strikes."""
    by_type: dict[str, list] = {}
    for r in rows:
        pt = r.get("pitch_type")
        if pt:
            by_type.setdefault(pt, []).append(r)
    out = {}
    for pt, pt_rows in sorted(by_type.items(), key=lambda x: -len(x[1])):
        if len(pt_rows) < min_pitches:
            continue
        s = csw_stats(pt_rows)
        if s:
            s["usage_pct"] = round(len(pt_rows) / len(rows) * 100, 1)
            out[pt] = s
    return out


def csw_by_side(rows: list[dict]) -> dict:
    """CSW split by batter side -- the same vL/vR split the K model already
    prices on."""
    return {
        "vs_L": csw_stats([r for r in rows if r.get("stand") == "L"]),
        "vs_R": csw_stats([r for r in rows if r.get("stand") == "R"]),
    }


def pitcher_csw(pitcher_id: int | None = None, name: str | None = None) -> dict:
    """Full CSW report for one pitcher, season to date (regular season
    only -- fetch_statcast passes hfGT=R|, matching FanGraphs' window).
    Give a name or an id."""
    if not pitcher_id:
        if not name:
            return {"error": "give a pitcher name or id"}
        found = statcast_api.resolve_player(name)
        if not found:
            return {"error": f"no player found for {name!r}"}
        pitcher_id = found["id"]
        name = found["name"]
    try:
        rows = parlay.get_player_season_rows(int(pitcher_id), True)
    except Exception as e:
        log.warning("csw: row fetch failed for %s: %s", pitcher_id, e)
        return {"error": f"row fetch failed: {e}"}
    if not rows:
        return {"error": "no pitch rows for this pitcher this season"}
    overall = csw_stats(rows)
    dates = sorted(r.get("game_date") for r in rows if r.get("game_date"))
    return {
        "pitcher_id": int(pitcher_id),
        "name": name,
        "window": {"first_pitch": dates[0] if dates else None,
                   "last_pitch": dates[-1] if dates else None,
                   "note": "regular season only (hfGT=R|), season to date"},
        "overall": overall,
        "by_side": csw_by_side(rows),
        "by_pitch_type": csw_by_pitch_type(rows),
        "verify_against": ("FanGraphs CSW% leaderboard -- definition matched "
                           "exactly (called strikes + swinging strikes, "
                           "foul tips and foul balls EXCLUDED, over total "
                           "pitches). Reconciled to the decimal 2026-07-29 "
                           "on Burns/Misiorowski/Luzardo."),
    }


# ---------------------------------------------------------------------------
# Bulk verification view. Temporary by design: this whole section plus the two
# app.py routes can be deleted in one commit once CSW is verified and either
# adopted or dropped. Nothing else imports it.
# ---------------------------------------------------------------------------

def slate_csw(date: str | None = None) -> dict:
    """Every probable starter on a slate with his season CSW, sorted high to
    low -- the whole board in one pass instead of one name at a time.

    Costs zero odds credits (MLB schedule + Savant rows only). The first
    call of a day is slow: one Savant pull per starter, cached per player
    per day by parlay.get_player_season_rows, so reloads are instant."""
    import time as _time
    et = _time.gmtime(_time.time() - 4 * 3600)
    date = date or _time.strftime("%Y-%m-%d", et)
    try:
        sched = requests.get(f"{MLB_BASE}/schedule",
                             params={"sportId": 1, "date": date,
                                     "hydrate": "probablePitcher,team"},
                             timeout=20).json()
    except Exception as e:
        return {"error": f"schedule fetch failed: {e}", "date": date}

    seen: set = set()
    starters = []
    for d in (sched.get("dates") or []):
        for g in (d.get("games") or []):
            for side in ("home", "away"):
                t = (g.get("teams") or {}).get(side) or {}
                p = t.get("probablePitcher") or {}
                pid = p.get("id")
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                starters.append({
                    "pitcher_id": pid,
                    "name": p.get("fullName", "?"),
                    "team": ((t.get("team") or {}).get("abbreviation") or "?"),
                })

    rows = []
    for s in starters:
        try:
            pr = parlay.get_player_season_rows(int(s["pitcher_id"]), True)
        except Exception as e:
            log.warning("slate csw: rows failed for %s: %s", s["name"], e)
            pr = None
        if not pr:
            rows.append({**s, "csw_pct": None, "note": "no pitch rows"})
            continue
        st = csw_stats(pr) or {}
        by_pt = csw_by_pitch_type(pr)
        # HIGHEST-CSW pitch, not the most-used one (csw_by_pitch_type sorts by
        # volume). "Best pitch" must mean best, or the column lies.
        best = max(by_pt.items(), key=lambda kv: kv[1]["csw_pct"], default=None)
        most = next(iter(by_pt.items()), None)
        rows.append({**s, **st,
                     "best_pitch": (f"{best[0]} {best[1]['csw_pct']}% CSW "
                                    f"@ {best[1]['usage_pct']}% usage") if best else None,
                     "most_used": (f"{most[0]} @ {most[1]['usage_pct']}%") if most else None})
    rows.sort(key=lambda r: -(r.get("csw_pct") or -1))
    return {"date": date, "n": len(rows), "starters": rows,
            "note": ("CSW = (called strikes + swinging strikes) / total pitches. "
                     "Foul tips and foul balls excluded -- matches FanGraphs. "
                     "League average ~29%.")}


def slate_csw_html(date: str | None = None) -> str:
    """The same thing as a plain table you can eyeball against FanGraphs."""
    data = slate_csw(date)
    if data.get("error"):
        return f"<p>{data['error']}</p>"
    head = ("<tr><th>#</th><th style='text-align:left'>Pitcher</th><th>Tm</th>"
            "<th>CSW%</th><th>CStr%</th><th>SwStr%</th><th>Whiff%<br><small>per swing</small></th>"
            "<th>Pitches</th><th style='text-align:left'>Best pitch (highest CSW)</th>"
            "<th style='text-align:left'>Most used</th></tr>")
    body = []
    for i, r in enumerate(data["starters"], 1):
        if r.get("csw_pct") is None:
            body.append(f"<tr><td>{i}</td><td>{r['name']}</td><td>{r['team']}</td>"
                        f"<td colspan='7'><i>{r.get('note', 'no data')}</i></td></tr>")
            continue
        c = r["csw_pct"]
        color = "#4caf7d" if c >= 32 else ("#e0a12f" if c >= 28 else "#8b95a1")
        body.append(
            f"<tr><td>{i}</td><td style='text-align:left'>{r['name']}</td>"
            f"<td>{r['team']}</td>"
            f"<td style='color:{color};font-weight:700'>{c}</td>"
            f"<td>{r['called_pct']}</td><td>{r['swstr_pct']}</td>"
            f"<td>{r.get('whiff_pct_of_swings', '-')}</td>"
            f"<td>{r['pitches']}</td>"
            f"<td style='text-align:left'><small>{r.get('best_pitch') or '-'}</small></td>"
            f"<td style='text-align:left'><small>{r.get('most_used') or '-'}</small></td></tr>")
    return f"""<!doctype html><meta charset="utf-8">
<title>CSW check - {data['date']}</title>
<style>
 body{{background:#0f1216;color:#e6e9ee;font:14px/1.5 -apple-system,Segoe UI,Inter,sans-serif;
      padding:18px;max-width:1000px;margin:auto}}
 h1{{font-size:17px;margin:0 0 4px}} p.note{{color:#8b95a1;font-size:12.5px;margin:0 0 14px}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{padding:5px 8px;text-align:center;border-bottom:1px solid #222831}}
 th{{color:#8b95a1;font-weight:600;font-size:12px}} small{{color:#8b95a1}}
</style>
<h1>CSW check - {data['date']} - {data['n']} starters</h1>
<p class="note">{data['note']} Green 32+, amber 28+. Compare the CSW% column
against the FanGraphs leaderboard.</p>
<table>{head}{''.join(body)}</table>"""
