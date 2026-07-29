"""
CSW% (Called Strikes + Whiffs) from the pitch-level rows we already pull.

CSW = (called strikes + whiffs) / total pitches -- Pollack/Fast, Pitcher
List, 2018. It is NOT a Baseball Savant column: Savant serves the raw
pitch descriptions, and CSW is an aggregation of them. FanGraphs
publishes the leaderboard, which is what these numbers should be checked
against.

Definitional care (this is the whole ballgame for matching a public
number): the standard counts called strikes, swinging strikes, and foul
tips into the glove, and EXCLUDES foul balls. That is exactly
statcast_api.WHIFF_DESCRIPTIONS (validated line-by-line against Savant's
own whiff numbers) plus called_strike -- so this module imports that set
rather than redefining it. One source of truth, per the house rule.

Nothing here touches the model. It exists so the input can be verified
against FanGraphs BEFORE it earns a gauntlet run.
"""
import logging

import statcast_api
import parlay

log = logging.getLogger("csw")

# The other half of CSW. Savant's description field for a taken strike.
CALLED_DESCRIPTIONS = {"called_strike"}


def csw_stats(rows: list[dict]) -> dict | None:
    """CSW over any subset of pitch rows. Denominator = ALL pitches in the
    subset (CSW's defining choice -- unlike whiff%, which is per swing)."""
    if not rows:
        return None
    pitches = len(rows)
    called = sum(1 for r in rows if r.get("description") in CALLED_DESCRIPTIONS)
    whiffs = sum(1 for r in rows if r.get("description") in statcast_api.WHIFF_DESCRIPTIONS)
    swings = sum(1 for r in rows if r.get("description") in statcast_api.SWING_DESCRIPTIONS)
    out = {
        "pitches": pitches,
        "called_strikes": called,
        "whiffs": whiffs,
        "csw_pct": round((called + whiffs) / pitches * 100, 1),
        "called_pct": round(called / pitches * 100, 1),
        "swstr_pct": round(whiffs / pitches * 100, 1),   # whiffs per PITCH
    }
    if swings:
        # whiffs per SWING -- Savant's "Whiff%", kept beside CSW so the two
        # never get confused with each other
        out["whiff_pct_of_swings"] = round(whiffs / swings * 100, 1)
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
        "verify_against": ("FanGraphs CSW% leaderboard (same definition: "
                           "called strikes + whiffs incl. foul tips, "
                           "excl. foul balls, over total pitches). "
                           "League average ~29%."),
    }
