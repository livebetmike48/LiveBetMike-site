import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import os

import matchups
import bullpen
import lab
import kboard
import kseasonlab
import kplays
import csw
import kmatchup
import pprops
import ppbacktest
import outs_lab
import thread as thread_gen
import projections
import pitchers as pitchers_mod

LAB_TOKEN = os.getenv("LAB_TOKEN", "")
# 2026 regular season opened March 25 (Opening Night; traditional Opening
# Day March 26). Env-tunable so next season is a variable, not a code edit.
MLB_SEASON_START = os.getenv("MLB_SEASON_START", "2026-03-25")
# Day windows the Lab accepts. "Season" is computed, never hardcoded --
# a fixed number would be wrong tomorrow.
LAB_DAYS = (3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 150, 180, 200)
MARKET_DAYS = (7, 14, 30, 60, 120, 150, 180, 200)


def season_days() -> int:
    """Completed days since Opening Night, capped at the largest window
    the Lab accepts. This is what the 'Season' option resolves to."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    try:
        start = _dt.strptime(MLB_SEASON_START, "%Y-%m-%d")
    except Exception:
        return 120
    et_now = _dt.now(_tz.utc) - _td(hours=4)
    days = (et_now.replace(tzinfo=None) - start).days
    return max(7, min(days, max(LAB_DAYS)))


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app")

app = FastAPI(title="Matchup Board")

kplays.start()  # K play auto-poster: OFF unless DISCORD_KPLAYS_WEBHOOK is set
outs_lab.register(app)  # /outs-lab + /api/outs-lab (play-by-play outs ladder test)


def _days(payload: dict, default: int, allowed: tuple):
    """Shared day parsing: a number from the allowed set, or the string
    'season' which resolves to days-since-Opening-Night. Returns an error
    dict (not an int) when the value isn't allowed, so callers can pass it
    straight back."""
    raw = payload.get("days", default)
    if isinstance(raw, str) and raw.strip().lower() == "season":
        return season_days()
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return {"error": f"days must be a number or 'season'"}
    if days in allowed:
        return days
    if days == season_days():
        return days  # the computed season length is always legal
    return {"error": "days must be one of "
                     + "/".join(str(d) for d in allowed) + " or 'season'"}


@app.get("/api/parks")
def api_parks():
    """What the park-factor fetch actually returned — status codes, content
    types, first bytes, venues parsed. Diagnostic: the K model's park knob
    has been silently neutral because this fetch fails."""
    try:
        import parks as _parks
        return _parks.diagnose()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/season")
def api_season():
    """What 'Season' means today -- so the UI can label the option with a
    real number instead of guessing."""
    return {"start": MLB_SEASON_START, "days": season_days(),
            "lab_days": list(LAB_DAYS), "market_days": list(MARKET_DAYS)}

@app.get("/api/matchups")
def api_matchups():
    return matchups.get_matchups()


@app.get("/api/detail")
def api_detail(batter_id: int, starter_id: int, hand: str):
    if hand not in ("L", "R"):
        return {"error": "hand must be L or R"}
    try:
        return matchups.get_detail(batter_id, starter_id, hand)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/teams")
def api_teams():
    try:
        return {"teams": bullpen.all_teams()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/bullpen")
def api_bullpen(team_id: int):
    try:
        return bullpen.get_usage(team_id)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/lab")
def api_lab():
    try:
        return lab.lab_state()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/kboard")
def api_kboard(d: int = 0):
    try:
        return kboard.get_board(d)
    except Exception as e:
        return {"error": str(e)}


@app.get("/pprops")
def page_pprops(d: int = 0):
    """Pitcher props verification page (BETA, engine only — never
    backtested, nothing logged). Temporary like /csw: delete pprops.py and
    these two routes together if it doesn't earn its place."""
    try:
        return HTMLResponse(pprops.slate_projections_html(d))
    except Exception as e:
        return HTMLResponse(f"<p>error: {e}</p>")


@app.get("/api/pprops")
def api_pprops(d: int = 0):
    try:
        return pprops.slate_projections(d)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/koverride")
def api_koverride(payload: dict):
    """Pitch-count cap / exclusion for one start. Token in the BODY like
    every other write route. Pre-game only -- the route refuses once the
    start is graded."""
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    try:
        date = payload.get("date") or ""
        if not date:
            import parlay as _p
            date = _p.et_date_str(int(payload.get("d") or 0))
        if payload.get("clear"):
            return kboard.clear_override(date, int(payload.get("starter_id") or 0))
        return kboard.set_override(
            date, int(payload.get("starter_id") or 0),
            pitch_limit=int(payload["pitch_limit"]) if payload.get("pitch_limit") else None,
            tbf_cap=int(payload["tbf_cap"]) if payload.get("tbf_cap") else None,
            exclude=bool(payload.get("exclude")),
            note=str(payload.get("note") or ""))
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/koverrides")
def api_koverrides(d: int = 0):
    try:
        import parlay as _p
        return {"date": _p.et_date_str(d),
                "overrides": kboard.get_overrides(_p.et_date_str(d))}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/ppbacktest")
def api_ppbacktest_run(payload: dict):
    """Run the props Brier backtest (hits + walks). Free — no odds calls.
    Token-gated like every run route."""
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    days = _days(payload, 14, (7, 14, 30, 45, 60))
    if isinstance(days, dict):
        return days
    started = ppbacktest.run_async(days)
    return {"started": started} if started else {"error": "a props backtest is already running"}


@app.get("/api/ppbacktest")
def api_ppbacktest_history():
    try:
        return ppbacktest.history()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/klineupsnaps")
def api_klineupsnaps(d: int = 0):
    """The lineup-post tracker: for each lineup that posted today, the
    K line/prices + model read AT that moment, and where the line went
    after. Forward data for the one edge the season test quantified."""
    try:
        import parlay as _p
        return kboard.lineup_snaps(_p.et_date_str(d))
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/thread")
def api_thread(d: int = 1):
    """The nightly X-thread draft: every team, starter + last-start pitch
    count + fact-based bullpen report, in Mike's exact layout, chunked
    under Discord's 2000-char cap. d=1 (default) = tomorrow's slate."""
    try:
        return thread_gen.build_thread(max(0, min(2, d)))
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/lineups")
def api_lineups(d: int = 0):
    """Today's (d=0) or tomorrow's (d=1) lineups -- posted where they're up,
    projected where they aren't. Read-only, no odds credits."""
    try:
        return kboard.projected_lineups(d)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/kplayers")
def api_kplayers():
    try:
        return {"players": kboard.players_list()}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/ksim")
def api_ksim(payload: dict):
    try:
        return kboard.sim_lineup(int(payload.get("starter_id") or 0),
                                 payload.get("batter_ids") or [],
                                 int(payload.get("d") or 0),
                                 int(payload["tbf"]) if payload.get("tbf") else None,
                                 int(payload["pitch_limit"]) if payload.get("pitch_limit") else None)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/klog.csv")
def api_klog_csv(days: int = 400):
    try:
        return PlainTextResponse(kboard.log_csv(days), media_type="text/csv",
                                 headers={"Content-Disposition":
                                          "attachment; filename=k_forward_log.csv"})
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/kvalidation")
def api_kvalidation():
    try:
        return kboard.validation_summary()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/klog")
def api_klog(days: int = 1):
    try:
        return kboard.log_details(days)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/lab/run")
def api_lab_run(payload: dict):
    # token in the BODY, never the URL -- URLs get written to logs
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    days = _days(payload, 7, LAB_DAYS)
    if isinstance(days, dict):
        return days
    started = lab.run_backtest_async(days)
    return {"started": started}


@app.post("/api/lab/config")
def api_lab_config(payload: dict):
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    updates = {k: v for k, v in payload.items() if k != "token"}
    try:
        return {"config": lab.set_config(updates)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/lab/market")
def api_lab_market(payload: dict):
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    days = _days(payload, 14, MARKET_DAYS)
    if isinstance(days, dict):
        return days
    return {"started": lab.run_market_async(days)}


@app.post("/api/klab/run")
def api_klab_run(payload: dict):
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    days = _days(payload, 7, LAB_DAYS)
    if isinstance(days, dict):
        return days
    return {"started": lab.run_k_backtest_async(days)}


@app.post("/api/klab/market")
def api_klab_market(payload: dict):
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    days = _days(payload, 14, MARKET_DAYS)
    if isinstance(days, dict):
        return days
    return {"started": lab.run_k_market_async(days, bool(payload.get("vs_open")))}


@app.post("/api/klab/blend")
def api_klab_blend(payload: dict):
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    try:
        test_days = int(payload.get("test_days", 30))
        source = "open" if payload.get("source") == "open" else "close"
        return lab.fit_k_blend(test_days=test_days, source=source)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/klab/fit")
def api_klab_fit(payload: dict):
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    try:
        return lab.fit_k_calibration_now()
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/klab/season")
def api_klab_season(payload: dict):
    """Season backtest suite: burn-in fits that YEAR'S curve, remainder
    graded OOS; market=True adds the units test vs real closing lines
    (archive-first). Long-running — the UI warns."""
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    try:
        return kseasonlab.start_season_suite(int(payload.get("season", 0)),
                                             market=bool(payload.get("market")),
                                             arm=str(payload.get("arm", "v2")))
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/kseasons")
def api_kseasons():
    """Run state + last result + stored receipts (read-only, house pattern)."""
    try:
        return kseasonlab.season_state() | {"runs": kseasonlab.season_history()}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/pplab/season")
def api_pplab_season(payload: dict):
    """Props (hits/walks) season suite -- burn-in fits each market's own
    within-year curve, remainder graded OOS; market_test adds units vs
    real closing lines (archive-first, shared pay-once store)."""
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    try:
        return ppbacktest.start_pp_season_suite(
            int(payload.get("season", 0)),
            market_test=bool(payload.get("market")))
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ppseasons")
def api_ppseasons():
    try:
        return ppbacktest.pp_season_state()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/kcage")
def api_kcage(days: int = 400):
    """K forward-cage scoreboard: each locked arm vs the LIVE model on the
    SAME graded starts (read-only results, house pattern -- no token)."""
    try:
        return kboard.cage_summary(days)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ppcage")
def api_ppcage(days: int = 400):
    """Props forward-cage scoreboard: hits + walks, per market per arm,
    flat-1u counted band + Brier, calibrated-vs-raw era split."""
    try:
        return kboard.pp_cage_summary(days)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/klab/config")
def api_klab_config(payload: dict):
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    updates = {k: v for k, v in payload.items() if k != "token"}
    try:
        return {"config": lab.set_k_config(updates)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/lab/priors")
def api_lab_priors(payload: dict):
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    csv_text = payload.get("csv", "")
    if not csv_text or len(csv_text) > 3_000_000:
        return {"error": "missing or oversized csv"}
    try:
        return lab.load_priors_csv(csv_text)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/pitchers")
def api_pitchers():
    try:
        return pitchers_mod.get_pitchers()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/projections")
def api_projections():
    try:
        projections.grade_pending()
        data = projections.get_today()
        if data.get("status") == "ready":
            data = projections.attach_odds(dict(data))
        data["result_log"] = projections.result_log()
        return data
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/csw")
def api_csw(name: str = "", pitcher_id: int = 0):
    """CSW% verification: our numbers from Savant pitch rows, to be checked
    against the FanGraphs leaderboard before CSW is allowed near the model.
    Read-only, no odds credits."""
    try:
        return csw.pitcher_csw(pitcher_id or None, name or None)
    except Exception as e:
        return {"error": str(e)}


@app.get("/csw")
def page_csw(date: str = ""):
    """Bulk CSW verification page -- today's starters in one table. Temporary:
    delete this route, /api/csw, and csw.py together when done."""
    try:
        return HTMLResponse(csw.slate_csw_html(date or None))
    except Exception as e:
        return HTMLResponse(f"<p>error: {e}</p>")


@app.get("/api/csw/slate")
def api_csw_slate(date: str = ""):
    try:
        return csw.slate_csw(date or None)
    except Exception as e:
        return {"error": str(e)}


@app.get("/kmatchup")
def page_kmatchup(starter: str = "", date: str = ""):
    """Arsenal-vs-lineup verification page. Temporary, like /csw: delete
    kmatchup.py and these two routes together when done."""
    if not starter:
        return HTMLResponse("<p>add ?starter=Name (a probable starter today)</p>")
    try:
        return HTMLResponse(kmatchup.matchup_html(starter, date or None))
    except Exception as e:
        return HTMLResponse(f"<p>error: {e}</p>")


@app.get("/api/kmatchup")
def api_kmatchup_arsenal(starter_id: int = 0, opp_team_id: int = 0, hand: str = ""):
    try:
        if hand not in ("L", "R"):
            return {"error": "hand must be L or R"}
        return kmatchup.lineup_matchup(starter_id, opp_team_id, hand)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/kmarketbets.csv")
def api_kmarketbets_csv(run_ts: int | None = None):
    """Every per-bet row from the K market tests, flat CSV. run_ts limits
    to one run (from the Lab history); default = all stored runs."""
    try:
        return PlainTextResponse(lab.export_market_bets_csv(run_ts),
                                 media_type="text/csv",
                                 headers={"Content-Disposition":
                                          "attachment; filename=k_market_bets.csv"})
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/lab/export.csv")
def api_lab_export():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(lab.export_csv(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=model_backtests.csv"})


@app.get("/")
def index():
    """The Pitcher Props tab registers itself from static/pprops_tab.js.
    The script tag is injected HERE rather than edited into index.html so
    that file stays byte-for-byte as it is -- delete this function's body
    back to the FileResponse line and the tab disappears with it.
    Idempotent: if the tag is ever added to index.html directly, this
    won't double it. Falls back to serving the file untouched on any
    read error, so the site can never go dark over a cosmetic tab."""
    try:
        with open("static/index.html", encoding="utf-8") as f:
            html = f.read()
        if "pprops_tab.js" not in html and "</body>" in html:
            html = html.replace(
                "</body>",
                '<script src="/static/pprops_tab.js"></script>\n'
                '<script src="/static/kadmin_tab.js"></script>\n'
                '<script src="/static/labdays.js"></script>\n'
                '<script src="/static/outs_lab_tab.js"></script>\n</body>', 1)
        return HTMLResponse(html)
    except Exception as e:
        log.warning("index injection skipped (%s) -- serving file as-is", e)
        return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
