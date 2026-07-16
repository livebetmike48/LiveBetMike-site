"""
Backtests the hit-probability model against REAL completed games with
point-in-time discipline: every prediction for date D uses only data from
before D. Grades against actual boxscore outcomes (did the batter record
a hit). Reports calibration -- when we said 70%, did they hit ~70%? --
plus Brier scores against two honesty baselines. This decides whether the
model ships.
"""
import logging
from datetime import datetime, timedelta, timezone

import requests

import parlay
import model

log = logging.getLogger("backtest")

MLB_BASE = "https://statsapi.mlb.com/api/v1"


def _final_games(date_str: str) -> list[dict]:
    resp = requests.get(f"{MLB_BASE}/schedule",
                        params={"sportId": 1, "date": date_str}, timeout=20)
    resp.raise_for_status()
    games = []
    for d in resp.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final" \
               and g.get("gameType") == "R":
                games.append(g)
    return games


def _game_predictions(game_pk: int, date_str: str, p_league: float,
                      hand_cache: dict) -> list[dict]:
    """For one final game: each lineup batter's point-in-time prediction
    plus the actual outcome from the boxscore."""
    box = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=20).json()
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        team = (box.get("teams") or {}).get(side) or {}
        opp_team = (box.get("teams") or {}).get(opp) or {}
        order = team.get("battingOrder") or []
        opp_pitchers = opp_team.get("pitchers") or []
        if not order or not opp_pitchers:
            continue
        starter_id = opp_pitchers[0]  # first pitcher listed = the starter
        try:
            starter_rows = parlay.get_player_season_rows(starter_id, True)
        except Exception:
            continue
        if starter_id not in hand_cache:
            try:
                hand_cache[starter_id] = parlay.get_starter_hand(starter_id)
            except Exception:
                hand_cache[starter_id] = None
        hand = hand_cache[starter_id]
        if hand not in ("L", "R"):
            continue
        for pid in order[:9]:
            player = (team.get("players") or {}).get(f"ID{pid}") or {}
            batting = ((player.get("stats") or {}).get("batting")) or {}
            ab = batting.get("atBats", 0)
            pa = batting.get("plateAppearances", ab)
            if not pa:
                continue  # in lineup but no PA recorded (early exit) -- ungradeable
            try:
                batter_rows = parlay.get_player_season_rows(pid, False)
            except Exception:
                continue
            sides = [r.get("stand") for r in model.rows_before(batter_rows, date_str) if r.get("stand")]
            if not sides:
                continue
            batter_side = max(set(sides), key=sides.count)
            pred = model.hit_probability(batter_rows, starter_rows, hand,
                                          batter_side, p_league, before=date_str)
            if pred is None:
                continue
            # Baseline 2: batter's raw hit-game rate before D (no starter adj)
            prior_rows = model.rows_before(batter_rows, date_str)
            rate, games = parlay.hit_game_rate(prior_rows)
            out.append({
                "date": date_str,
                "p_model": pred["p_hit"],
                "p_naive": rate if games >= 20 else None,
                "hit": 1 if batting.get("hits", 0) > 0 else 0,
            })
    return out


def run_backtest(days: int, progress=None) -> dict:
    """Walk the last `days` completed days, predict every lineup batter
    point-in-time, grade vs reality, and report calibration + Brier."""
    p_league = model.league_hit_rate()
    end = datetime.now(timezone.utc) - timedelta(hours=4)
    preds = []
    hand_cache: dict = {}
    for i in range(1, days + 1):
        date_str = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            games = _final_games(date_str)
        except Exception as e:
            log.warning("schedule failed for %s: %s", date_str, e)
            continue
        for g in games:
            try:
                preds.extend(_game_predictions(g["gamePk"], date_str, p_league, hand_cache))
            except Exception as e:
                log.warning("game %s failed: %s", g.get("gamePk"), e)
        if progress:
            progress(i, days, len(preds))

    if not preds:
        return {"n": 0}

    # Calibration buckets
    buckets = {}
    for p in preds:
        lo = int(p["p_model"] * 10) * 10
        b = buckets.setdefault(lo, {"n": 0, "hits": 0, "p_sum": 0.0})
        b["n"] += 1
        b["hits"] += p["hit"]
        b["p_sum"] += p["p_model"]
    calibration = [
        {"bucket": f"{lo}-{lo+10}%", "n": b["n"],
         "predicted": round(b["p_sum"] / b["n"] * 100, 1),
         "actual": round(b["hits"] / b["n"] * 100, 1)}
        for lo, b in sorted(buckets.items())
    ]

    def brier(pairs):
        return round(sum((p - h) ** 2 for p, h in pairs) / len(pairs), 4)

    model_brier = brier([(p["p_model"], p["hit"]) for p in preds])
    league_game_rate = sum(p["hit"] for p in preds) / len(preds)
    constant_brier = brier([(league_game_rate, p["hit"]) for p in preds])
    naive_pairs = [(p["p_naive"], p["hit"]) for p in preds if p["p_naive"] is not None]
    naive_brier = brier(naive_pairs) if naive_pairs else None

    return {
        "n": len(preds),
        "days": days,
        "calibration": calibration,
        "brier_model": model_brier,
        "brier_constant": constant_brier,
        "brier_naive": naive_brier,
        "overall_hit_rate": round(league_game_rate * 100, 1),
    }
