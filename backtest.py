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
    home_name = (((box.get("teams") or {}).get("home") or {}).get("team") or {}).get("name", "")
    away_name = (((box.get("teams") or {}).get("away") or {}).get("team") or {}).get("name", "")
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
            batter_name = ((player.get("person") or {}).get("fullName"))
            pred = model.hit_probability(batter_rows, starter_rows, hand,
                                          batter_side, p_league, before=date_str,
                                          batter_name=batter_name)
            if pred is None:
                continue
            # Baseline 2: batter's raw hit-game rate before D (no starter adj)
            prior_rows = model.rows_before(batter_rows, date_str)
            rate, games = parlay.hit_game_rate(prior_rows)
            out.append({
                "date": date_str,
                "name": batter_name,
                "home_name": home_name, "away_name": away_name,
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
        log.info("backtest day %d/%d (%s): %d final games", i, days, date_str, len(games))
        for gi, g in enumerate(games, 1):
            try:
                preds.extend(_game_predictions(g["gamePk"], date_str, p_league, hand_cache))
            except Exception as e:
                log.warning("game %s failed: %s", g.get("gamePk"), e)
            if progress:
                progress(i, days, len(preds), f"game {gi}/{len(games)} on {date_str}")
        log.info("backtest day %d/%d done: %d total predictions", i, days, len(preds))
        if progress:
            progress(i, days, len(preds), "day complete")

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


# ---------- market backtest: model vs real closing lines ----------

import odds_api

THRESHOLDS = (2, 5, 10)


def _simulate_bets(candidates: list[dict], thresholds=THRESHOLDS) -> dict:
    """Flat 1u on every candidate above each EV threshold. Pure + testable.
    candidate: {ev, side, price, hit}"""
    out = {}
    for thr in thresholds:
        bets = [c for c in candidates if c["ev"] >= thr]
        units = 0.0
        side_units = {"over": 0.0, "under": 0.0}
        wins = 0
        for b in bets:
            dec = odds_api.american_to_decimal(b["price"])
            won = (b["hit"] == 1) if b["side"] == "over" else (b["hit"] == 0)
            profit = (dec - 1) if won else -1.0
            units += profit
            side_units[b["side"]] += profit
            wins += 1 if won else 0
        out[str(thr)] = {
            "bets": len(bets), "wins": wins,
            "units": round(units, 2),
            "roi_pct": round(units / len(bets) * 100, 1) if bets else None,
            "over_bets": sum(1 for b in bets if b["side"] == "over"),
            "under_bets": sum(1 for b in bets if b["side"] == "under"),
            "over_units": round(side_units["over"], 2),
            "under_units": round(side_units["under"], 2),
        }
    return out


def run_market_backtest(days: int, progress=None) -> dict:
    """Walk past days: point-in-time model prediction per lineup batter,
    joined to the REAL closing hit-prop lines from the historical odds
    archive, flat-betting every edge above each threshold. Units don't lie."""
    p_league = model.league_hit_rate()
    end = datetime.now(timezone.utc) - timedelta(hours=4)
    hand_cache: dict = {}
    candidates = []
    games_priced = 0
    for i in range(1, days + 1):
        date_str = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            games = _final_games(date_str)
        except Exception as e:
            log.warning("market: schedule failed %s: %s", date_str, e)
            continue
        if not games:
            continue
        hist_events = odds_api.get_historical_events(f"{date_str}T16:00:00Z")
        log.info("market day %d/%d (%s): %d games, %d hist events",
                 i, days, date_str, len(games), len(hist_events))
        for g in games:
            try:
                preds = _game_predictions(g["gamePk"], date_str, p_league, hand_cache)
            except Exception as e:
                log.warning("market game %s failed: %s", g.get("gamePk"), e)
                continue
            if not preds or not hist_events:
                continue
            ev_match = odds_api.find_event(hist_events, preds[0]["home_name"], preds[0]["away_name"])
            if not ev_match:
                continue
            snapshot_at = ev_match.get("commence_time") or f"{date_str}T23:00:00Z"
            odds_data = odds_api.get_historical_event_odds(ev_match.get("id"), snapshot_at)
            if not odds_data:
                continue
            games_priced += 1
            for p in preds:
                for side, prob in (("over", p["p_model"]), ("under", 1 - p["p_model"])):
                    priced = odds_api.player_prop_prices(odds_data, "batter_hits", p["name"], side=side)
                    if not priced or priced.get("point") != 0.5:
                        continue
                    bp = odds_api.best_price(priced["prices"])
                    if not bp:
                        continue
                    ev = (prob * odds_api.american_to_decimal(bp[1]) - 1) * 100
                    if ev >= min(THRESHOLDS):
                        candidates.append({"date": date_str, "name": p["name"], "side": side,
                                           "price": bp[1], "ev": round(ev, 1), "hit": p["hit"]})
            if progress:
                progress(i, days, games_priced, len(candidates))

    report = {"days": days, "games_priced": games_priced,
              "credits_estimate": games_priced * 10 + days,
              "candidates": len(candidates),
              "by_threshold": _simulate_bets(candidates)}
    top = sorted(candidates, key=lambda c: -c["ev"])[:12]
    report["sample_bets"] = top
    return report
