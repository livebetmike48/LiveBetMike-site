"""
Nightly thread generator -- Mike's daily X thread, mechanically drafted.

Per team (alphabetical by team name, exactly his layout):
  - "<Starter> makes the start for the <Team>."
  - "<Last> threw <N> pitches in his last start."
  - "Bullpen Report: <facts>"

The bot posts it at 11:05pm ET in 4 chunks; Mike copies into Sheets and
adds the editorial layer (opener reads, leash calls, IL context). This
module produces FACTS only, from MLB's own feeds:
  back-to-back days, 3-of-last-4, heavy outings ("2/26" = IP/pitches).
Games still in progress at generation time get a partial-usage note --
the facts shown are real, the label says they're incomplete.
"""
import time
import logging
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger("thread")

MLB = "https://statsapi.mlb.com/api/v1"
_cache: dict = {}


def _get(url: str, ttl: int = 900) -> dict:
    now = time.time()
    hit = _cache.get(url)
    if hit and now - hit[0] < ttl:
        return hit[1]
    data = requests.get(url, timeout=20).json()
    _cache[url] = (now, data)
    return data


def _et_today() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=4)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _club(name: str) -> str:
    """'Boston Red Sox' -> 'Red Sox', 'Arizona Diamondbacks' -> 'Diamondbacks',
    'Athletics' -> 'Athletics' -- the short club name Mike's lines use."""
    words = name.split()
    if len(words) == 1:
        return name
    if words[-1] in ("Sox", "Jays"):
        return " ".join(words[-2:])
    return words[-1]


def _schedule(date_str: str) -> list[dict]:
    j = _get(f"{MLB}/schedule?sportId=1&date={date_str}&hydrate=probablePitcher")
    out = []
    for d in j.get("dates") or []:
        out.extend(d.get("games") or [])
    return out


def _last_start_pitches(pid: int, season: int) -> int | None:
    try:
        j = _get(f"{MLB}/people/{pid}/stats?stats=gameLog&group=pitching&season={season}",
                 ttl=3600)
    except Exception:
        return None
    splits = ((j.get("stats") or [{}])[0].get("splits")) or []
    # NEVER trust the API's ordering -- Mike's own bullpen-bot client sorts
    # gameLog by date for exactly this reason. Unsorted, "the last start"
    # can silently become some earlier start, scattering wrong pitch
    # counts across the whole thread.
    splits = sorted(splits, key=lambda sp: sp.get("date") or "")
    for sp in reversed(splits):
        st = sp.get("stat") or {}
        if st.get("gamesStarted"):
            n = st.get("numberOfPitches") or st.get("pitchesThrown")
            return int(n) if n else None
    return None


def _box_usage(game_pk: int) -> dict:
    """{team_id: [(name, ip_str, pitches, is_starter)]} for one game."""
    try:
        box = _get(f"{MLB}/game/{game_pk}/boxscore", ttl=1800)
    except Exception:
        return {}
    out = {}
    for side in ("home", "away"):
        t = (box.get("teams") or {}).get(side) or {}
        tid = ((t.get("team") or {}).get("id"))
        arms = []
        for i, pid in enumerate(t.get("pitchers") or []):
            p = (t.get("players") or {}).get(f"ID{pid}") or {}
            st = ((p.get("stats") or {}).get("pitching")) or {}
            n = st.get("numberOfPitches") or st.get("pitchesThrown")
            arms.append((((p.get("person") or {}).get("fullName")) or str(pid),
                         str(st.get("inningsPitched") or "?"),
                         int(n) if n else 0, i == 0))
        if tid:
            out[tid] = arms
    return out


def _usage_by_day(days: list[str]) -> tuple[dict, set]:
    """usage[team_id][date] = [reliever tuples]; plus team_ids whose game
    today isn't final (partial-usage note)."""
    usage: dict = {}
    partial: set = set()
    today = days[0]
    for date_str in days:
        for g in _schedule(date_str):
            state = ((g.get("status") or {}).get("abstractGameState")) or ""
            pk = g.get("gamePk")
            ids = [((g.get("teams") or {}).get(s) or {}).get("team", {}).get("id")
                   for s in ("home", "away")]
            if date_str == today and state not in ("Final",):
                for tid in ids:
                    if tid:
                        partial.add(tid)
                if state != "Live":
                    continue
            if state not in ("Final", "Live"):
                continue
            for tid, arms in _box_usage(pk).items():
                usage.setdefault(tid, {})[date_str] = [a for a in arms if not a[3]]
    return usage, partial


def _bullpen_facts(tid: int, usage: dict, days: list[str], day_name: str) -> str:
    """Fact lines only: back-to-backs, 3-of-4, heavy recent outings."""
    by_day = usage.get(tid) or {}
    seen: dict = {}
    for i, d in enumerate(days):
        for name, ip, pitches, _ in by_day.get(d, []):
            seen.setdefault(name, {"days": set(), "latest": None})["days"].add(i)
            if seen[name]["latest"] is None:
                seen[name]["latest"] = (ip, pitches)
    b2b, three, heavy = [], [], []
    for name, info in seen.items():
        di = info["days"]
        if 0 in di and 1 in di:
            b2b.append(name)
        elif len(di & {0, 1, 2, 3}) >= 3:
            three.append(name)
        elif 0 in di and info["latest"] and (
                info["latest"][1] >= 30 or info["latest"][0].startswith("2")):
            heavy.append((name, info["latest"]))
    bits = []
    if b2b:
        who = " and ".join(b2b) if len(b2b) <= 2 else ", ".join(b2b[:-1]) + " and " + b2b[-1]
        verb = "went" if len(b2b) == 1 else "all went"
        bits.append(f"{who} {verb} back to back and won't pitch on {day_name}.")
    if three:
        who = " and ".join(three)
        bits.append(f"{who} pitched 3 of the last 4 days and should be down.")
    for name, (ip, pitches) in heavy[:3]:
        bits.append(f"{name} went {ip}/{pitches}.")
    return " ".join(bits) if bits else "Would expect everybody to remain available here."


def build_thread(offset: int = 1) -> dict:
    """The thread for today+offset's slate (post at 11:05pm the night
    before with offset=1). Returns {"date", "chunks": [str x4], "text"}."""
    et = _et_today()
    slate_day = et + timedelta(days=offset)
    date_str = slate_day.strftime("%Y-%m-%d")
    day_name = slate_day.strftime("%A")
    nice = f"{slate_day.strftime('%B')} {_ordinal(slate_day.day)}"
    days = [(et - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]
    usage, partial = _usage_by_day(days)

    teams: dict = {}
    for g in _schedule(date_str):
        for side in ("home", "away"):
            t = (g.get("teams") or {}).get(side) or {}
            info = t.get("team") or {}
            if info.get("id"):
                teams[info["name"]] = {"id": info["id"],
                                       "prob": t.get("probablePitcher") or {}}
    season = slate_day.year
    sections = []
    for name in sorted(teams):
        t = teams[name]
        prob = t["prob"]
        lines = [name, ""]
        if prob.get("fullName"):
            full = prob["fullName"]
            last = full.split()[-1]
            lines.append(f"- {full} makes the start for the {_club(name)}.")
            n = _last_start_pitches(prob.get("id"), season)
            lines.append(f"- {last} threw {n} pitches in his last start." if n
                         else f"- {last}'s last-start pitch count unavailable.")
        else:
            lines.append(f"- Bullpen game for the {_club(name)}.")
            lines.append("-")
        note = ("(Tonight's game not final — bullpen usage below is partial.) "
                if t["id"] in partial else "")
        lines.append(f"- Bullpen Report: {note}"
                     f"{_bullpen_facts(t['id'], usage, days, day_name)}")
        sections.append("\n".join(lines))

    intro = (f"Here is EVERYTHING you NEED to know if you are BETTING on MLB "
             f"player props for {nice}:\n\n"
             "Every pitcher. Every bullpen. All 30 teams. I put in the work so "
             "you don't have to. Sign up to Kalshi with my referral link today. "
             "Trade 10, Get 10 with code LIVEBETMIKE. 🔗 kalshi.com/r/livebetmike")
    outro = ("If you enjoy these threads please:\n\n"
             "- Like the first tweet to help others find it.\n"
             "- Follow me @LiveBetMike\n"
             "- Keep showing support so I have reason to keep doing them. "
             "I will be back again tomorrow!")

    chunks, buf = [], intro
    per = max(1, (len(sections) + 3) // 4)
    for i, sec in enumerate(sections):
        cand = buf + "\n\n" + sec
        if (i and i % per == 0 and len(chunks) < 3) or len(cand) > 1900:
            chunks.append(buf)
            buf = sec
        else:
            buf = cand
    chunks.append(buf)
    if len(chunks[-1]) + len(outro) + 2 <= 1950:
        chunks[-1] += "\n\n" + outro
    else:
        chunks.append(outro)
    return {"date": date_str, "for": nice, "n_teams": len(sections),
            "chunks": chunks, "text": "\n\n".join([intro] + sections + [outro])}
