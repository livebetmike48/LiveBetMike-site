"""
Tests for the Outs Lab interactive query layer (line_cell / line_ladder /
routes / PAGE card). Additive -- test_outs_lab.py is untouched.
Run: python3 -m pytest test_outs_query.py -q
"""
import os
import tempfile

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "t.db"))

import outs_lab  # noqa: E402


def _mk_start(pk, half, cum, hits=None, walks=None, ks=None):
    n = len(cum)
    hits = hits or [0] * n
    walks = walks or [0] * n
    ks = ks or [0] * n
    return {
        "pitcher_id": 100 + pk, "half": half,
        "pa": [{"reached": False, "out_delta": 1, "runners_before": 0,
                "extra": 0} for _ in range(n)],
        "cum": cum, "cum_hits": hits, "cum_walks": walks, "cum_ks": ks,
        "unknown": {}, "runner_outs": 0,
        "tbf": n, "outs": cum[-1], "extra_outs": 0,
    }


def _seed(season=2024):
    """25 clean starts (outs after 17 = 17) + 15 rough (outs after 17 = 10),
    all 27 TBF, so P(15+ outs by 17 BF) = 25/40 exactly."""
    outs_lab._starts_cache.clear()
    with outs_lab._conn() as c:
        c.execute("DELETE FROM outs_lab_starts")
        c.execute("DELETE FROM outs_lab_games")
    for i in range(25):
        cum = [min(j + 1, 27) for j in range(27)]            # retires everyone
        hits = [0] * 27
        outs_lab._save_game(season, {"gamePk": 1000 + i, "date": "2024-06-01"},
                            [_mk_start(i, "top", cum, hits=hits)])
    for i in range(15):
        # 10 outs after 17 batters, finishes 27 TBF / 18 outs
        cum = [round((j + 1) * 10 / 17) for j in range(17)] + \
              [10 + round((j + 1) * 8 / 10) for j in range(10)]
        hits = [min(j // 3 + 1, 9) for j in range(27)]        # hitty start
        outs_lab._save_game(season, {"gamePk": 2000 + i, "date": "2024-06-02"},
                            [_mk_start(50 + i, "top", cum, hits=hits)])


def test_over_threshold():
    assert outs_lab._over_threshold(14.5) == 15
    assert outs_lab._over_threshold(18.5) == 19
    assert outs_lab._over_threshold(18) == 19       # integer line: strictly more
    assert outs_lab._over_threshold(0.5) == 1


def test_cell_hand_math():
    _seed()
    c = outs_lab.line_cell("outs", 14.5, 17, [2024])
    assert c["starts"] == 40 and c["hit"] == 25
    assert abs(c["empirical"] - 25 / 40) < 1e-9
    assert c["need"] == 15
    assert c["fair"] == "-167"                       # 62.5% -> -166.67
    assert c["ci95"][0] < 25 / 40 < c["ci95"][1]
    assert 0 < c["binomial"] < 1
    assert c["note"] == ""


def test_cell_integer_line_push_semantics():
    _seed()
    # over 17 needs 18+; the clean starts have exactly 17 after 17 BF
    c = outs_lab.line_cell("outs", 17, 17, [2024])
    assert c["need"] == 18 and c["hit"] == 0
    assert "strictly more" in c["note"]


def test_cell_other_stats_and_ranges():
    _seed()
    h = outs_lab.line_cell("hits", 4.5, 17, [2024])
    # only the 15 rough starts have hits; after 17 BF they have 6 hits
    assert h["starts"] == 40 and h["hit"] == 15
    assert outs_lab.line_cell("era", 2.5, 17, [2024])["error"]
    assert outs_lab.line_cell("outs", 14.5, 3, [2024])["error"]   # bf < 5
    assert outs_lab.line_cell("outs", 99, 17, [2024])["error"]    # line range
    assert outs_lab.line_cell("outs", "x", 17, [2024])["error"]
    assert outs_lab.line_cell("outs", 14.5, 17, [1999])["error"]  # no data


def test_cell_thin_sample_refuses():
    _seed()
    # bf=28: every start has tbf=27, none faced 28 -> 0 eligible
    c = outs_lab.line_cell("outs", 14.5, 28, [2024])
    assert "too thin" in c["error"]


def test_ladder_matches_cell():
    _seed()
    lad = outs_lab.line_ladder("outs", 14.5, [2024])
    row17 = next(r for r in lad["rows"] if r["n"] == 17)
    cell = outs_lab.line_cell("outs", 14.5, 17, [2024])
    assert row17["empirical"] == cell["empirical"]
    assert row17["starts"] == cell["starts"]
    assert row17["binomial"] == cell["binomial"]
    # monotone: clearing 15 outs by more batters can only get easier
    emps = [r["empirical"] for r in lad["rows"]]
    assert emps == sorted(emps)


def test_starts_cache_hits_and_evicts():
    _seed()
    a = outs_lab.cached_starts([2024])
    b = outs_lab.cached_starts([2024])
    assert a is b                                   # served from cache
    outs_lab._starts_cache[(2024,)] = (0.0, a)      # expire it
    c = outs_lab.cached_starts([2024])
    assert c is not a and len(c) == len(a)
    for i in range(15):                             # burst of entries
        outs_lab._starts_cache[(3000 + i,)] = (outs_lab.time.time(), [])
    outs_lab.cached_starts([1998])                  # next insert enforces cap
    assert len(outs_lab._starts_cache) <= 12


def test_build_report_still_works_after_cum_key_hoist():
    _seed()
    rep = outs_lab.build_report(outs_lab.load_starts([2024]), "t", {})
    assert rep["starts"] == 40
    assert set(rep["grids"]) == {"outs", "hits", "walks", "ks"}
    assert rep["headline"]["empirical"] is not None   # report shape unchanged


def test_routes():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    _seed()
    app = FastAPI()
    outs_lab.register(app)
    cl = TestClient(app)
    r = cl.get("/api/outs-lab/cell?stat=outs&line=14.5&bf=17&seasons=2024").json()
    assert r["hit"] == 25 and r["fair"] == "-167"
    r = cl.get("/api/outs-lab/cell?stat=outs&line=18.5&bf=26").json()
    assert r["seasons"] == [2024]                    # empty -> coverage fallback
    r = cl.get("/api/outs-lab/ladder?stat=hits&line=4.5&seasons=2024").json()
    assert r["rows"] and r["need"] == 5
    r = cl.get("/api/outs-lab/report?seasons=2024").json()
    assert r["starts"] == 40                          # old route untouched


def test_page_has_card_and_headline_gone():
    p = outs_lab.PAGE
    for needle in ("qstat", "qline", "qbf", "ask()", "/api/outs-lab/cell",
                   "/api/outs-lab/ladder"):
        assert needle in p, needle
    assert "P(15 outs within first 17 batters faced)" not in p
    # diagnostics kept
    for keep in ("Extra outs by base state", "hook-polluted", "reach-rate bucket"):
        assert keep in p, keep


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))


# ===================== Vegas-implied-TBF backtest tests =====================
import sys
import types


def _payload(market, name, point, over_price=-110, under_price=-110,
             book="DraftKings"):
    return {"bookmakers": [{"title": book, "markets": [{
        "key": market, "outcomes": [
            {"name": "Over", "description": name, "point": point,
             "price": over_price},
            {"name": "Under", "description": name, "point": point,
             "price": under_price}]}]}]}


def _seed_two_seasons():
    """Prior season 2021 (grid fuel) + test season 2024 (one game)."""
    outs_lab._starts_cache.clear()
    with outs_lab._conn() as c:
        for t in ("outs_lab_starts", "outs_lab_games", "vtbf_runs",
                  "vtbf_bets"):
            c.execute(f"DELETE FROM {t}")
    clean_cum = [min(j + 1, 27) for j in range(27)]
    rough_cum = [round((j + 1) * 10 / 17) for j in range(17)] + \
                [10 + round((j + 1) * 8 / 10) for j in range(10)]
    rough_hits = [min(j // 3 + 1, 9) for j in range(27)]
    for i in range(30):
        outs_lab._save_game(2021, {"gamePk": 100 + i, "date": "2021-06-01"},
                            [_mk_start(i, "top", list(clean_cum))])
    for i in range(10):
        outs_lab._save_game(2021, {"gamePk": 200 + i, "date": "2021-06-02"},
                            [_mk_start(40 + i, "top", list(rough_cum),
                                       hits=list(rough_hits))])
    # 2024: one game, two starters -- pitcher 999 (priced), 998 (no outs line)
    s1 = _mk_start(0, "top", list(clean_cum))
    s1["pitcher_id"] = 999
    s2 = _mk_start(0, "bottom", list(clean_cum))
    s2["pitcher_id"] = 998
    outs_lab._save_game(2024, {"gamePk": 9001, "date": "2024-06-01"}, [s1, s2])
    # a second 2024 game with NO outs market at all
    s3 = _mk_start(0, "top", list(clean_cum))
    s3["pitcher_id"] = 997
    outs_lab._save_game(2024, {"gamePk": 9002, "date": "2024-06-01"}, [s3])


def _fake_kbacktest(odds_by_market_event, calls):
    m = types.ModuleType("kbacktest")
    m._fetch_stats = {"events_api": 0, "events_hit": 0,
                      "odds_api": 0, "odds_hit": 0}
    m.K_MARKET_BOOKS = None

    def _hist_events(snapshot):
        m._fetch_stats["events_hit"] += 1
        return [{"id": "ev1", "commence_time": "2024-06-01T23:10:00Z",
                 "home_team": "St. Louis Cardinals",
                 "away_team": "Pittsburgh Pirates"},
                {"id": "ev2", "commence_time": "2024-06-01T23:10:00Z",
                 "home_team": "New York Mets",
                 "away_team": "Atlanta Braves"}]

    def _hist_odds(event_id, snapshot, market):
        calls.append((event_id, market))
        m._fetch_stats["odds_api"] += 1
        return odds_by_market_event.get((event_id, market))
    m._hist_events = _hist_events
    m._hist_odds = _hist_odds
    return m


def _vtbf_env(monkeypatch, odds_map, calls):
    monkeypatch.setitem(sys.modules, "kbacktest",
                        _fake_kbacktest(odds_map, calls))
    monkeypatch.setattr(outs_lab, "VTBF_GRID_MIN_STARTS", 10)
    monkeypatch.setattr(outs_lab, "_day_games", lambda d: {
        9001: {"home": "St. Louis Cardinals", "away": "Pittsburgh Pirates"},
        9002: {"home": "New York Mets", "away": "Atlanta Braves"}})
    monkeypatch.setattr(outs_lab, "_pitcher_names", lambda ids: outs_lab._names)
    outs_lab._names.clear()
    outs_lab._names.update({999: "Kyle Test", 998: "Rich Nolines",
                            997: "No Market Guy"})


def test_vtbf_grid_math():
    _seed_two_seasons()
    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        mp.setattr(outs_lab, "VTBF_GRID_MIN_STARTS", 10)
        g = outs_lab.VtbfGrid([2021])
        # league outs rate: (30*27 + 10*18) / (40*27) = 990/1080
        assert abs(g.rates["outs"] - 990 / 1080) < 1e-9
        # O+H+BB: clean 27, rough 18+9+0=27 -> ratio exactly 1.0
        assert g.tbf_ratio == 1.0
        # p at integer n and interpolated between equal neighbours
        assert g.p("outs", 14.5, 20) == 0.75
        assert g.p("outs", 14.5, 20.5) == 0.75
        b20 = outs_lab.binom_tail(20, 15, g.rates["outs"])
        b21 = outs_lab.binom_tail(21, 15, g.rates["outs"])
        assert abs(g.binom("outs", 14.5, 20.5, g.rates["outs"])
                   - (b20 + b21) / 2) < 1e-9


def test_pitcher_history_is_point_in_time():
    _seed_two_seasons()
    idx = outs_lab.pitcher_history([2021, 2024])
    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        mp.setattr(outs_lab, "VTBF_MIN_PITCHER_BF", 20)
        # pitcher 999's only start is ON 2024-06-01 -> excluded before it
        assert outs_lab.pitcher_rate(idx, 999, "outs", "2024-06-01") is None
        # after that date it counts: 27 outs / 27 bf
        assert outs_lab.pitcher_rate(idx, 999, "outs", "2024-06-02") == 1.0
    # below the real 200-BF floor -> None
    assert outs_lab.pitcher_rate(idx, 999, "outs", "2024-06-02") is None


def test_pitcher_arm_prob_slots_rate_into_grid_shape():
    _seed_two_seasons()
    import pytest as _pt
    with _pt.MonkeyPatch.context() as mp:
        mp.setattr(outs_lab, "VTBF_GRID_MIN_STARTS", 10)
        g = outs_lab.VtbfGrid([2021])
        base = g.p("outs", 14.5, 20.5)
        better = outs_lab._pitcher_arm_prob(g, "outs", 14.5, 20.5, base, 0.99)
        worse = outs_lab._pitcher_arm_prob(g, "outs", 14.5, 20.5, base, 0.60)
        assert better > base > worse
        assert outs_lab._pitcher_arm_prob(g, "outs", 14.5, 20.5, base, None) == base
        assert 0.005 <= worse and better <= 0.995


def test_vtbf_end_to_end(monkeypatch):
    _seed_two_seasons()
    calls = []
    odds_map = {
        ("ev1", "pitcher_outs"): _payload(
            "pitcher_outs", "Kyle Test", 14.5, over_price=-250, under_price=180),
        ("ev1", "pitcher_hits_allowed"): _payload(
            "pitcher_hits_allowed", "Kyle Test", 4.5, over_price=150, under_price=-250),
        ("ev1", "pitcher_walks"): _payload(
            "pitcher_walks", "Kyle Test", 1.5, over_price=-110, under_price=-110),
        # ev2 has NO outs market (returns None below)
    }
    _vtbf_env(monkeypatch, odds_map, calls)
    rep = outs_lab.run_vtbf_season(2024)
    assert not rep.get("error"), rep
    # gates: 998 skipped (no outs line), ev2 game skipped (no outs market)
    assert rep["skips"]["no_outs_line"] == 1
    assert rep["skips"]["no_outs_market"] == 1
    assert rep["starts_priced"] == 1
    # ev2's hits/walks were never fetched -- the gate saves the credits
    assert ("ev2", "pitcher_hits_allowed") not in calls
    assert ("ev2", "pitcher_outs") in calls
    # hand math -- implied TBF 14.5+4.5+1.5 = 20.5 * ratio 1.0
    league = rep["arms"]["league"]
    # outs: base p=0.75, over @ -250 (dec 1.4) -> EV +5.0, actual 27 > 14.5 -> win +0.4u
    assert league["outs"]["bets"] == 1 and league["outs"]["record"] == "1-0"
    assert abs(league["outs"]["units"] - 0.4) < 0.01
    # hits: p(over)=0.25 -> under 0.75 @ -250 -> EV +5.0, actual 0 hits -> win
    assert league["hits"]["bets"] == 1 and league["hits"]["record"] == "1-0"
    # walks: p(over)=0 -> under prob 1.0 @ -110 -> EV 90.9% = suspect, no bet
    assert league["walks"]["bets"] == 0
    assert rep["suspect_excluded"] >= 1
    assert league["total"]["bets"] == 2 and abs(league["total"]["units"] - 0.8) < 0.01
    # pitcher 999 has no PRIOR starts -> rate missing, pitcher arm == league
    assert rep["pitcher_rate_missing"] >= 1
    assert rep["arms"]["pitcher"]["total"]["bets"] == 2
    # receipts + grid provenance
    assert rep["odds_fetches"]["odds_api"] == len(calls)
    assert rep["grid_seasons"] == [2021] and rep["tbf_ratio"] == 1.0
    # persisted: run + bets + csv
    runs = outs_lab.vtbf_history()
    assert runs and runs[0]["season"] == 2024
    csv = outs_lab.vtbf_bets_csv()
    assert "Kyle Test" in csv and csv.count("\n") >= 4  # header + 4 arm-bets


def test_vtbf_errors(monkeypatch):
    _seed_two_seasons()
    _vtbf_env(monkeypatch, {}, [])
    assert "not in the dataset" in outs_lab.run_vtbf_season(2030)["error"]
    # earliest stored season has nothing before it -> grid refusal
    assert "before" in outs_lab.run_vtbf_season(2021)["error"]


def test_vtbf_routes(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    _seed_two_seasons()
    monkeypatch.setenv("LAB_TOKEN", "tk")
    app = FastAPI()
    outs_lab.register(app)
    cl = TestClient(app)
    r = cl.get("/api/outs-lab/vtbf").json()
    assert "state" in r and "runs" in r
    r = cl.post("/api/outs-lab/vtbf/run", json={"token": "wrong",
                                                "seasons": [2024]}).json()
    assert r["error"] == "bad token"
    r = cl.get("/api/outs-lab/vtbf.csv")
    assert r.status_code == 200 and r.text.startswith("run_ts,")


def test_page_and_report_have_vtbf():
    assert "/api/outs-lab/vtbf/run" in outs_lab.PAGE
    assert "Vegas-TBF market backtest" in outs_lab.PAGE
