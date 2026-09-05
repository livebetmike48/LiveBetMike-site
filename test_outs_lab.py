import outs_lab as L

def play(half, inning, outs_after, ev, pitcher=1, post=None):
    m = {"pitcher": {"id": pitcher}, "batter": {"id": 99}}
    for k in (post or []): m[k] = {"id": 5}
    return {"about": {"halfInning": half, "inning": inning},
            "count": {"outs": outs_after}, "result": {"eventType": ev}, "matchup": m}

# Top half: starter 1. Inning 1: K, single (runner on 1st), GIDP (2 outs) -> 3 outs on 3 PA
# Inning 2: walk, caught stealing (runner out, mid-PA), then batter singles (PA continues, inning 1 out),
#           flyout, flyout -> 3 outs, PA in inning = 3 (walk, single, out, out) = 4 PA
# Inning 3: unknown event w/ delta 1 (counted PA, inferred out), single, then new pitcher.
plays = [
 play("top",1,1,"strikeout"),
 play("top",1,1,"single",post=["postOnFirst"]),
 play("top",1,3,"grounded_into_double_play"),
 play("top",2,0,"walk",post=["postOnFirst"]),
 play("top",2,1,"caught_stealing_2b"),          # mid-PA runner out
 play("top",2,1,"single",post=["postOnFirst"]),
 play("top",2,2,"field_out",post=["postOnFirst"]),
 play("top",2,3,"field_out"),
 play("top",3,1,"weird_new_code"),              # unknown, delta 1 -> out
 play("top",3,1,"single",post=["postOnFirst"]),
 play("top",3,2,"strikeout",pitcher=2),         # reliever
 # bottom half: starter 7, one perfect inning
 play("bottom",1,1,"field_out",pitcher=7), play("bottom",1,2,"strikeout",pitcher=7), play("bottom",1,3,"sac_bunt",pitcher=7),
]
seqs = L.starter_sequences(plays)
top = next(s for s in seqs if s["half"]=="top"); bot = next(s for s in seqs if s["half"]=="bottom")
assert top["pitcher_id"]==1 and bot["pitcher_id"]==7
assert top["tbf"]==9, top["tbf"]          # 3 + 4 + 2
assert top["outs"]==7, top["outs"]        # 3 + 3 + 1
assert top["cum"]==[1,1,3,3,4,5,6,7,7], top["cum"]
# extra outs: GIDP +1, CS +1 (credited to the single that followed)
assert top["extra_outs"]==2, top["extra_outs"]
assert top["runner_outs"]==1
assert top["pa"][4]["extra"]==1 and top["pa"][4]["reached"] is True   # CS rolled onto the single
assert top["pa"][2]["runners_before"]==1 and top["pa"][2]["extra"]==1  # GIDP w/ runner on
assert top["pa"][3]["runners_before"]==0                               # new inning resets
assert top["unknown"]=={"weird_new_code":1}
assert bot["tbf"]==3 and bot["outs"]==3 and bot["extra_outs"]==0
print("parse OK")

# binomial check vs the hand numbers from earlier
assert abs(L.binom_tail(21,15,0.6986)-0.545)<0.005
assert abs(L.binom_tail(17,15,0.670)-0.047)<0.002
assert L.american(0.545)=="-120" and L.american(0.076)=="+1216"
print("binom OK")

# report on a synthetic population: 200 starts, each 24 BF with binomial outs
import random; random.seed(1)
starts=[]
for i in range(200):
    pa=[];cum=[];c=0
    for j in range(24):
        reached = random.random()<0.32
        d = 0 if reached else 1
        c+=d; pa.append({"reached":reached,"out_delta":d,"runners_before":0,"extra":0}); cum.append(c)
    starts.append({"pitcher_id":i,"half":"top","pa":pa,"cum":cum,"unknown":{},"runner_outs":0,"tbf":24,"outs":c,"extra_outs":0})
rep = L.build_report(starts,"synthetic",{"start":"x","end":"y","games":100,"game_errors":0,"seconds":1})
assert rep["starts"]==200 and rep["tbf"]==4800
assert abs(rep["outs_per_bf"]-0.68)<0.02 and abs(rep["retire_rate"]-rep["outs_per_bf"])<1e-9  # no extra outs -> equal
row21 = next(r for r in rep["ladder"] if r["n"]==21)
assert row21["starts"]==200 and abs(row21["empirical"]-row21["pred_outs_rate"])<0.10, row21
assert row21["focus"] and not next(r for r in rep["ladder"] if r["n"]==15)["focus"]
assert rep["headline"]["starts"]==200
assert rep["tto"][0]["pa"]==1800 and rep["tto"][2]["pa"]==1200
assert rep["final_tbf_view"][0]["n"]==24
print("report OK", "ladder n=21:", row21["empirical"], "pred", row21["pred_outs_rate"])

# FastAPI registration + Discord-irrelevant but route sanity
from fastapi import FastAPI
from fastapi.testclient import TestClient
app=FastAPI(); L.register(app); c=TestClient(app)
assert c.get("/outs-lab").status_code==200 and "Outs Lab" in c.get("/outs-lab").text
j=c.get("/api/outs-lab").json(); assert "state" in j and "runs" in j
assert c.post("/api/outs-lab/run",json={"token":"nope","season":2025}).json()=={"error":"bad token"}
print("routes OK")

# async run path with mocked network -> DB write + state transitions
import time, json
L.final_games = lambda s,e: [{"gamePk":k,"date":"2025-06-01"} for k in range(12)]
L.fetch_plays = lambda pk: plays
r = L.start_async(season=2025); assert r["started"]
assert L.start_async(season=2025)["reason"]=="already running"
for _ in range(100):
    if not L.state()["running"]: break
    time.sleep(0.05)
st = L.state(); assert not st["running"] and st["progress"]=="done" and not st["error"], st
h = L.history(); assert h and h[0]["report"]["starts"]==24 and h[0]["report"]["meta"]["games"]==12
assert h[0]["report"]["unknown_events"]=={"weird_new_code":12}
print("async+db OK")
