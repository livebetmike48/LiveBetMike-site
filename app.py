import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import matchups

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


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
