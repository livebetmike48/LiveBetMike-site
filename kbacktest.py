"""
Backtests the strikeout model against REAL completed starts with the same
point-in-time discipline as the hit backtest: every prediction for date D
uses only rows from before D, actual lineups from the boxscore, graded
against the starter's real strikeOuts.

Calibration runs grade the model's P(K > L) on the fixed line ladder
(3.5-7.5) -- every start yields five binary predictions, bucketed and
Brier-scored against two honesty baselines:
  constant: the pooled empirical clear-rate for each line
  naive:    Poisson from the pitcher's own point-in-time K-per-start mean

The market test joins each start to the REAL historical closing
pitcher_strikeouts line and flat-bets every edge above each threshold.
"""
import math
import logging
from datetime import datetime, timedelta, timezone

import requests

import parlay
import kmodel
import backtest  # reuse _final_games + _simulate_bets (same repo, same rules)
import odds_api

try:
    import parks
except ImportError:
    parks = None

log = logging.getLogger("kbacktest")

MLB_BASE = "https://statsapi.mlb.com/api/v1"

LINE_LADDER = (3.5, 4.5, 5.5, 6.5, 7.5)


def _park_k(venue: str | None) -> float | None:
    if not venue or parks is None:
        return None
    fn = getattr(parks, "k_factor_for", None)
    return fn(venue) if fn else None


def _poisson_sf(mean: float, line: float) -> float:
    """Naive baseline: P(K > line) under Poisson(mean)."""
    if mean <= 0:
        return 0.0
    need = math.floor(line) + 1
    p = math.exp(-mean)
    cdf = p
    for k in range(1, need):
        p *= mean / k
        cdf += p
    return max(0.0, 1.0 - cdf)


def _majority_side(rows: list[dict], before: str) -> str | None:
    sides = [r.get("stand") for r in kmodel.rows_before(rows, before) if r.get("stand")]
    return max(set(sides), key=sides.count) if sides else None


def _game_starts(game_pk: int, date_str: str, p_league: float,
                 hand_cache: dict, venue: str | None = None) -> list[dict]:
    """For one final game: each starter's point-in-time K distribution +
    his actual strikeouts from the boxscore."""
    box = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=20).json()
    home_name = (((box.get("teams") or {}).get("home") or {}).get("team") or {}).get("name", "")
    away_name = (((box.get("teams") or {}).get("away") or {}).get("team") or {}).get("name", "")
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        pitching_team = (box.get("teams") or {}).get(side) or {}
        batting_team = (box.get("teams") or {}).get(opp) or {}
        pitchers = pitching_team.get("pitchers") or []
        order = batting_team.get("battingOrder") or []
        if not pitchers or not order:
            continue
        starter_id = pitchers[0]
        sp = (pitching_team.get("players") or {}).get(f"ID{starter_id}") or {}
        actual_k = (((sp.get("stats") or {}).get("pitching")) or {}).get("strikeOuts")
        starter_name = ((sp.get("person") or {}).get("fullName")) or ""
        if actual_k is None:
            continue
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

        lineup = []
        for pid in order[:9]:
            try:
                b_rows = parlay.get_player_season_rows(pid, False)
            except Exception:
                lineup.append(None)
                continue
            b_side = _majority_side(b_rows, date_str)
            lineup.append({"rows": b_rows, "side": b_side, "name": pid} if b_side else None)

        kdist = kmodel.k_distribution(lineup, starter_rows, hand, p_league,
                                      before=date_str, park_k_factor=_park_k(venue))
        if kdist is None:
            continue

        # naive baseline mean: his own K per start before D
        prior = kmodel.rows_before(starter_rows, date_str)
        samples = kmodel.tbf_samples(prior)
        prior_k = sum(1 for r in prior
                      if r.get("events") in kmodel.K_EVENTS)
        naive_mean = (prior_k / len(samples)) if samples else None

        out.append({
            "date": date_str, "name": starter_name,
            "home_name": home_name, "away_name": away_name,
            "kdist": kdist, "actual_k": int(actual_k),
            "naive_mean": naive_mean,
        })
    return out


def run_k_backtest(days: int, progress=None) -> dict:
    """Walk the last `days` completed days, model every start point-in-time,
    grade P(K > L) across the line ladder vs reality."""
    p_league = kmodel.league_k_rate()
    end = datetime.now(timezone.utc) - timedelta(hours=4)
    starts = []
    hand_cache: dict = {}
    for i in range(1, days + 1):
        date_str = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            games = backtest._final_games(date_str)
        except Exception as e:
            log.warning("k schedule failed for %s: %s", date_str, e)
            continue
        log.info("k backtest day %d/%d (%s): %d final games", i, days, date_str, len(games))
        for gi, g in enumerate(games, 1):
            try:
                starts.extend(_game_starts(g["gamePk"], date_str, p_league, hand_cache,
                                           venue=(g.get("venue") or {}).get("name")))
            except Exception as e:
                log.warning("k game %s failed: %s", g.get("gamePk"), e)
            if progress:
                progress(i, days, len(starts), f"game {gi}/{len(games)} on {date_str}")
        if progress:
            progress(i, days, len(starts), "day complete")

    if not starts:
        return {"n": 0}

    # Five binary predictions per start across the line ladder
    preds = []
    for s in starts:
        for line in LINE_LADDER:
            preds.append({
                "p_model": kmodel.calibrate(kmodel.prob_over(s["kdist"]["dist"], line)),
                "p_naive": _poisson_sf(s["naive_mean"], line) if s["naive_mean"] else None,
                "line": line,
                "cleared": 1 if s["actual_k"] > line else 0,
            })

    buckets = {}
    for p in preds:
        lo = int(p["p_model"] * 10) * 10
        b = buckets.setdefault(lo, {"n": 0, "hits": 0, "p_sum": 0.0})
        b["n"] += 1
        b["hits"] += p["cleared"]
        b["p_sum"] += p["p_model"]
    calibration = [
        {"bucket": f"{lo}-{lo+10}%", "n": b["n"],
         "predicted": round(b["p_sum"] / b["n"] * 100, 1),
         "actual": round(b["hits"] / b["n"] * 100, 1)}
        for lo, b in sorted(buckets.items())
    ]

    def brier(pairs):
        return round(sum((p - h) ** 2 for p, h in pairs) / len(pairs), 4)

    model_brier = brier([(p["p_model"], p["cleared"]) for p in preds])
    # constant baseline: pooled clear-rate PER LINE (fair -- knows each line's base rate)
    line_rates = {L: [] for L in LINE_LADDER}
    for p in preds:
        line_rates[p["line"]].append(p["cleared"])
    line_base = {L: (sum(v) / len(v) if v else 0) for L, v in line_rates.items()}
    constant_brier = brier([(line_base[p["line"]], p["cleared"]) for p in preds])
    naive_pairs = [(p["p_naive"], p["cleared"]) for p in preds if p["p_naive"] is not None]
    naive_brier = brier(naive_pairs) if naive_pairs else None

    mean_err = sum(abs(s["kdist"]["mean_k"] - s["actual_k"]) for s in starts) / len(starts)

    return {
        "n": len(preds),
        "starts": len(starts),
        "days": days,
        "calibration": calibration,
        "brier_model": model_brier,
        "brier_constant": constant_brier,
        "brier_naive": naive_brier,
        "mean_abs_error_k": round(mean_err, 2),
        "avg_projected_k": round(sum(s["kdist"]["mean_k"] for s in starts) / len(starts), 2),
        "avg_actual_k": round(sum(s["actual_k"] for s in starts) / len(starts), 2),
    }


def run_k_market_backtest(days: int, progress=None) -> dict:
    """Walk past days: point-in-time K distribution per start, joined to
    the REAL historical closing pitcher_strikeouts line, flat-betting every
    edge above each threshold. Units don't lie."""
    p_league = kmodel.league_k_rate()
    end = datetime.now(timezone.utc) - timedelta(hours=4)
    hand_cache: dict = {}
    candidates = []
    games_priced = 0
    suspect = 0
    for i in range(1, days + 1):
        date_str = (end - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            games = backtest._final_games(date_str)
        except Exception as e:
            log.warning("k market: schedule failed %s: %s", date_str, e)
            continue
        if not games:
            continue
        hist_events = odds_api.get_historical_events(f"{date_str}T16:00:00Z")
        log.info("k market day %d/%d (%s): %d games, %d hist events",
                 i, days, date_str, len(games), len(hist_events))
        for g in games:
            try:
                starts = _game_starts(g["gamePk"], date_str, p_league, hand_cache,
                                      venue=(g.get("venue") or {}).get("name"))
            except Exception as e:
                log.warning("k market game %s failed: %s", g.get("gamePk"), e)
                continue
            if not starts or not hist_events:
                continue
            ev_match = odds_api.find_event(hist_events, starts[0]["home_name"], starts[0]["away_name"])
            if not ev_match:
                continue
            snapshot_at = ev_match.get("commence_time") or f"{date_str}T23:00:00Z"
            odds_data = odds_api.get_historical_event_odds(
                ev_match.get("id"), snapshot_at, market="pitcher_strikeouts")
            if not odds_data:
                continue
            games_priced += 1
            for s in starts:
                for side in ("over", "under"):
                    priced = odds_api.player_prop_prices(
                        odds_data, "pitcher_strikeouts", s["name"], side=side)
                    if not priced or priced.get("point") is None:
                        continue
                    line = priced["point"]
                    if line != int(line) + 0.5:
                        continue  # whole-number lines can push -- flat-bet sim stays honest on half-points
                    p_over = kmodel.calibrate(kmodel.prob_over(s["kdist"]["dist"], line))
                    prob = p_over if side == "over" else 1 - p_over
                    bp = odds_api.best_price(priced["prices"])
                    if not bp:
                        continue
                    ev = (prob * odds_api.american_to_decimal(bp[1]) - 1) * 100
                    if ev > 20:
                        suspect += 1
                        continue  # >20% edges vs closing K lines = model error, not value
                    if ev >= min(backtest.THRESHOLDS):
                        cleared = 1 if s["actual_k"] > line else 0
                        candidates.append({"date": date_str, "name": s["name"],
                                           "side": side, "line": line,
                                           "price": bp[1], "ev": round(ev, 1),
                                           "hit": cleared})
            if progress:
                progress(i, days, games_priced, len(candidates))

    report = {"days": days, "games_priced": games_priced,
              "suspect_excluded": suspect,
              "credits_estimate": games_priced * 20 + days,
              "candidates": len(candidates),
              "by_threshold": backtest._simulate_bets(candidates)}
    report["sample_bets"] = sorted(candidates, key=lambda c: -c["ev"])[:12]
    return report
