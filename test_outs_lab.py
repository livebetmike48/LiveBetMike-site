"""Offline tests for outs_lab -- parsing, math, report, storage, routes.
Live StatsAPI is mocked; the fields= payload shape is the one thing these
can't prove."""
import os, time, random
os.environ.setdefault("DB_PATH", "/tmp/outs_lab_test.db")
if os.path.exists(os.environ["DB_PATH"]): os.remove(os.environ["DB_PATH"])
import outs_lab as L

def play(half, inning, outs_after, ev, pitcher=1, post=None):
    m = {"pitcher": {"id": pitcher}, "batter": {"id": 99}}
    for k in (post or []): m[k] = {"id": 5}
    return {"about": {"halfInning": half, "inning": inning},
            "count": {"outs": outs_after}, "result": {"eventType": ev}, "matchup": m}

# Top: starter 1. Inn1: K, single, GIDP. Inn2: walk, CS (mid-PA), single, out, out.
# Inn3: unknown code (delta 1), single, then reliever. Bottom: starter 7, 1-2-3.
plays = [
 play("top",1,1,"strikeout"), play("top",1,1,"single",post=["postOnFirst"]),
 play("top",1,3,"grounded_into_double_play"),
 play("top",2,0,"walk",post=["postOnFirst"]), play("top",2,1,"caught_stealing_2b"),
 play("top",2,1,"single",post=["postOnFirst"]), play("top",2,2,"field_out",post=["postOnFirst"]),
 play("top",2,3,"field_out"),
 play("top",3,1,"weird_new_code"), play("top",3,1,"single",post=["postOnFirst"]),
 play("top",3,2,"strikeout",pitcher=2),
 play("bottom",1,1,"field_out",pitcher=7), play("bottom",1,2,"strikeout",pitcher=7),
 play("bottom",1,3,"sac_bunt",pitcher=7),
]
seqs = L.starter_sequences(plays)
top = next(s for s in seqs if s["half"]=="top"); bot = next(s for s in seqs if s["half"]=="bottom")
assert top["pitcher_id"]==1 and bot["pitcher_id"]==7
assert top["tbf"]==9 and top["outs"]==7
assert top["cum"]==[1,1,3,3,4,5,6,7,7]
assert top["cum_hits"]==[0,1,1,1,2,2,2,2,3] and top["cum_walks"]==[0,0,0,1,1,1,1,1,1]
assert top["cum_ks"]==[1,1,1,1,1,1,1,1,1]
assert top["extra_outs"]==2 and top["runner_outs"]==1
assert top["pa"][4]["extra"]==1 and top["pa"][4]["reached"] is True
assert top["pa"][2]["runners_before"]==1 and top["pa"][3]["runners_before"]==0
assert top["unknown"]=={"weird_new_code":1}
assert bot["tbf"]==3 and bot["outs"]==3 and bot["cum_ks"]==[0,1,1]
print("parse OK")

assert abs(L.binom_tail(21,15,0.6986)-0.545)<0.005
assert abs(L.binom_tail(17,15,0.670)-0.047)<0.002
assert L.american(0.545)=="-120" and L.american(0.076)=="+1216"
print("binom OK")

# synthetic population: 200 starts x 24 BF, iid batters
random.seed(1); starts=[]
for i in range(200):
    pa=[];cum=[];ch=[];cw=[];ck=[];o=h=w=k=0
    for j in range(24):
        u=random.random()
        if u<0.22: h+=1; reached=True
        elif u<0.32: w+=1; reached=True
        else:
            reached=False; o+=1
            if u<0.55: k+=1
        pa.append({"reached":reached,"out_delta":0 if reached else 1,"runners_before":0,"extra":0})
        cum.append(o); ch.append(h); cw.append(w); ck.append(k)
    starts.append({"pitcher_id":i,"half":"top","season":2025,"pa":pa,"cum":cum,"cum_hits":ch,
                   "cum_walks":cw,"cum_ks":ck,"unknown":{},"runner_outs":0,"tbf":24,"outs":o,"extra_outs":0})
rep = L.build_report(starts,"synthetic",{})
assert rep["starts"]==200 and rep["tbf"]==4800 and rep["seasons"]==[2025]
assert abs(rep["outs_per_bf"]-0.68)<0.02 and abs(rep["retire_rate"]-rep["outs_per_bf"])<1e-9
assert set(rep["grids"])=={"outs","hits","walks","ks"}
g=rep["grids"]; r24=lambda st: next(x for x in g[st]["rows"] if x["n"]==24)
assert r24("outs")["14.5"]==next(x for x in rep["ladder"] if x["n"]==24)["empirical"]
o=r24("outs"); assert o["12.5"]>=o["14.5"]>=o["15.5"]>=o["17.5"]>=o["18.5"]>=o["20.5"]
assert abs(o["17.5"]-o["binom_17.5"])<0.10 and abs(o["binom_20.5"]-0.027)<0.01
hh=r24("hits"); assert abs(g["hits"]["per_bf_rate"]-0.22)<0.02 and abs(hh["4.5"]-hh["binom_4.5"])<0.10
kk=r24("ks"); assert abs(g["ks"]["per_bf_rate"]-0.23)<0.03
ww=r24("walks"); assert ww["0.5"]>ww["3.5"]
assert g["outs"]["rows"][0]["n"]==12 and rep["headline"]["starts"]==200
print("report OK  outs 14.5@21:", next(x for x in g['outs']['rows'] if x['n']==21)["14.5"],
      " hits 4.5@18:", next(x for x in g['hits']['rows'] if x['n']==18)["4.5"])

# storage: fetch_season w/ mocked HTTP -> resumable -> report_for -> coverage
# mock honors the season in the date range, like the real schedule does
SCHED = {"2024": [{"gamePk":k,"date":"2024-06-01"} for k in range(12)],
         "2023": [{"gamePk":100+k,"date":"2023-06-01"} for k in range(5)]}
L.final_games = lambda s,e: SCHED.get(s[:4], [])
L.fetch_plays = lambda pk: plays
r1 = L.fetch_season(2024); assert r1["fetched"]==12 and r1["starts"]==24 and r1["errors"]==0
r2 = L.fetch_season(2024); assert r2["fetched"]==0 and r2["skipped"]==12 and r2.get("note")
cov = L.coverage(); assert cov==[{"season":2024,"games":12,"starts":24,"first":"2024-06-01","last":"2024-06-01"}]
st = L.load_starts([2024]); assert len(st)==24 and st[0]["cum_hits"] and st[0]["season"]==2024
rep2 = L.report_for([2024]); assert rep2["starts"]==24 and rep2["unknown_events"]=={"weird_new_code":12}
assert "error" in L.report_for([1999])
# second season -> pooled report spans both
L.fetch_season(2023); assert L.report_for([2023,2024])["starts"]==34
assert [c["season"] for c in L.coverage()]==[2023,2024]
print("storage OK")

# async runner: fetch list -> pooled report row in outs_lab_runs
r = L.start_async([2023,2024]); assert r["started"] and r["seasons"]==[2023,2024]
assert L.start_async([2023])["reason"]=="already running"
for _ in range(200):
    if not L.state()["running"]: break
    time.sleep(0.05)
s = L.state(); assert not s["running"] and not s["error"], s
h = L.history(); assert h and h[0]["report"]["starts"]==34 and h[0]["report"]["label"]=="2023, 2024"
assert len(h[0]["report"]["fetch"])==2 and h[0]["report"]["fetch"][0]["note"]
print("async OK")

# routes
from fastapi import FastAPI
from fastapi.testclient import TestClient
app=FastAPI(); L.register(app); c=TestClient(app)
assert "Outs Lab" in c.get("/outs-lab").text
j=c.get("/api/outs-lab").json(); assert {"state","runs","coverage"}<=set(j) and j["coverage"][0]["season"]==2023
assert c.get("/api/outs-lab/report?seasons=2024").json()["starts"]==24
assert c.get("/api/outs-lab/report").json()["starts"]==34          # default = everything stored
assert c.post("/api/outs-lab/run",json={"token":"nope","seasons":[2025]}).json()=={"error":"bad token"}
print("routes OK")
