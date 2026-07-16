"""
Builds the day's hitter-vs-starter matchup board from the validated engine.
Heavy (one season fetch per hitter), so it's computed once per day in a
background thread and cached; the API serves instantly after warmup.
"""
import threading
import logging

import statcast_api
import parlay

log = logging.getLogger("matchups")

_cache = {"date": None, "status": "cold", "data": None}
_lock = threading.Lock()

HITTERS_PER_TEAM = 6


def last10_strip(rows: list[dict]) -> list[bool]:
    """Ten most recent games as hit/no-hit booleans -- the row's signature
    visual, straight from real game logs."""
    games: dict = {}
    for r in rows:
        gpk, date = r.get("game_pk"), r.get("game_date")
        if gpk is None or not date:
            continue
        key = (date, gpk)
        games.setdefault(key, False)
        if r.get("events") in parlay.HIT_EVENTS:
            games[key] = True
    ordered = [games[k] for k in sorted(games.keys())]
    return ordered[-10:]


def _build_matchups() -> dict:
    slate = parlay.get_today_slate()
    games_out = []
    for g in slate:
        game_entry = {
            "game_pk": g["game_pk"],
            "away": g["teams"]["away"]["abbrev"],
            "home": g["teams"]["home"]["abbrev"],
            "hitters": [],
        }
        for side, opp_side in (("home", "away"), ("away", "home")):
            team = g["teams"][side]
            opp = g["teams"][opp_side]
            if not opp["starter_id"]:
                continue
            try:
                hand = parlay.get_starter_hand(opp["starter_id"])
            except Exception:
                continue
            if hand not in ("L", "R"):
                continue
            for batter in parlay.shortlist_hitters([team["abbrev"]], "xba", HITTERS_PER_TEAM):
                try:
                    rows = parlay.get_player_season_rows(batter["player_id"], False)
                except Exception:
                    continue
                if not rows:
                    continue
                vs = statcast_api.vs_handedness_stats(rows, "p_throws", hand)
                if not vs or vs.get("pa", 0) < 40:
                    continue
                rate, games_played = parlay.hit_game_rate(rows)
                game_entry["hitters"].append({
                    "player_id": batter["player_id"],
                    "name": batter["name"],
                    "team": team["abbrev"],
                    "starter": opp["starter_name"],
                    "starter_id": opp["starter_id"],
                    "hand": hand,
                    "pa": vs["pa"],
                    "avg": vs.get("avg"),
                    "xba": vs.get("xba"),
                    "xwoba": vs.get("xwoba"),
                    "k_pct": vs.get("k_pct"),
                    "bb_pct": vs.get("bb_pct"),
                    "whiff_pct": vs.get("whiff_pct"),
                    "hit_game_pct": round(rate * 100, 1),
                    "games": games_played,
                    "streak": parlay.hitting_streak(rows),
                    "last10": last10_strip(rows),
                })
        game_entry["hitters"].sort(key=lambda h: -(h["xba"] or 0))
        games_out.append(game_entry)
    return {"date": parlay.et_date_str(0), "games": games_out}


def get_matchups() -> dict:
    """Serve from cache; kick off a rebuild in the background when the
    day rolls over. Never blocks the request."""
    today = parlay.et_date_str(0)
    with _lock:
        if _cache["date"] == today and _cache["status"] == "ready":
            return {"status": "ready", **_cache["data"]}
        if _cache["status"] == "warming":
            return {"status": "warming"}
        _cache["status"] = "warming"
        _cache["date"] = today

    def _warm():
        try:
            data = _build_matchups()
            with _lock:
                _cache["data"] = data
                _cache["status"] = "ready"
            log.info("Matchup board ready: %d games", len(data["games"]))
        except Exception as e:
            log.error("Matchup build failed: %s", e)
            with _lock:
                _cache["status"] = "cold"

    threading.Thread(target=_warm, daemon=True).start()
    return {"status": "warming"}


def get_detail(batter_id: int, starter_id: int, hand: str) -> dict:
    """The drill-down: batter vs each pitch (vs this hand) merged with the
    starter's mix to the batter's side -- same shape as /matchup."""
    batter_rows = parlay.get_player_season_rows(batter_id, False)
    starter_rows = parlay.get_player_season_rows(starter_id, True)

    batter_vs_hand = [r for r in batter_rows if r.get("p_throws") == hand]
    per_pitch = statcast_api.vs_each_pitch(batter_vs_hand, min_pitches=10)

    # Batter's side vs this starter (assume R unless rows say otherwise)
    sides = [r.get("stand") for r in batter_rows if r.get("stand")]
    batter_side = max(set(sides), key=sides.count) if sides else "R"
    mix = statcast_api.pitch_mix_breakdown(
        [r for r in starter_rows if r.get("stand") == batter_side]
    )
    return {
        "per_pitch": per_pitch,
        "starter_mix": mix,
        "batter_side": batter_side,
    }
