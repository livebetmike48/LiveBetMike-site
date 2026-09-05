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
