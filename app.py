import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os

import matchups
import bullpen
import lab

LAB_TOKEN = os.getenv("LAB_TOKEN", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Matchup Board")


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


@app.post("/api/lab/run")
def api_lab_run(payload: dict):
    # token in the BODY, never the URL -- URLs get written to logs
    if not LAB_TOKEN or payload.get("token") != LAB_TOKEN:
        return {"error": "bad token"}
    days = int(payload.get("days", 7))
    if days not in (3, 5, 7, 10, 14):
        return {"error": "days must be 3/5/7/10/14"}
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


@app.get("/api/lab/export.csv")
def api_lab_export():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(lab.export_csv(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=model_backtests.csv"})


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
