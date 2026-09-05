/* Outs Lab tab -- self-registering, index.html is never edited.
   REPLACES the earlier outs_lab_tab.js. Adds the interactive query card:
   pick a stat, type ANY line and ANY batters-faced, get the empirical
   clear rate + fair odds + CI with the binomial closed form alongside.
   The fixed "15 outs in 17 BF" headline is gone -- the card is the page.
   Pattern identical to pprops_tab.js: append a tab button, wrap
   window.showView, delegate every other view to the original. Plain
   strings only (no template literals). Delete this file + its script
   tag and the site is exactly as before. */
(function () {
  "use strict";

  var VIEW = "outslab";
  var S = {
    active: false, poll: null, cov: [], report: null, stat: "outs",
    qstat: "outs", qline: "18.5", qbf: "26",
    checked: null,            // Set of season numbers, null until coverage
    cell: null, ladder: null, err: ""
  };

  function tok() {
    var box = document.getElementById("olb-token");
    var v = box ? box.value.trim() : "";
    if (v) window.LAB_TOKEN_MEM = v;
    return v || window.LAB_TOKEN_MEM || "";
  }

  function pct(x) { return (x === null || x === undefined) ? "\u2014" : (100 * x).toFixed(1) + "%"; }
  function esc(x) { return String(x).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  function yrsPicked() {
    if (!S.checked) return [];
    var out = []; S.checked.forEach(function (y) { out.push(y); });
    return out.sort();
  }

  // ------------------------------------------------------------- fetches
  function refresh(then) {
    fetch("/api/outs-lab").then(function (r) { return r.json(); }).then(function (s) {
      if (!S.active) return;
      S.state = s.state || {};
      S.cov = s.coverage || [];
      if (S.checked === null && S.cov.length) {
        S.checked = new Set();
        S.cov.forEach(function (c) { S.checked.add(c.season); });
      }
      paintProgress();
      paintCov();
      paintYears();
      if (!S.state.running && S.poll) { clearInterval(S.poll); S.poll = null; loadAll(); }
      if (then) then();
    }).catch(function (e) { S.err = String(e); paintProgress(); });
  }

  function startFetch() {
    var box = document.getElementById("olb-fetchyrs");
    var yrs = (box ? box.value : "").split(",").map(function (x) { return +x.trim(); }).filter(Boolean);
    fetch("/api/outs-lab/run", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ token: tok(), seasons: yrs })
    }).then(function (r) { return r.json(); }).then(function (j) {
      if (j.error || j.reason) {
        var p = document.getElementById("olb-prog");
        if (p) p.textContent = j.error || j.reason;
        if (j.error === "bad token") return;   // don't poll a refused run
      }
      if (S.poll) clearInterval(S.poll);
      S.poll = setInterval(refresh, 1500);
      refresh();
    });
  }

  function loadAll() { loadReport(); loadCell(); loadVtbf(); }

  function loadReport() {
    var yrs = yrsPicked();
    if (!yrs.length) { S.report = null; paintGrids(); return; }
    fetch("/api/outs-lab/report?seasons=" + yrs.join(",")).then(function (r) { return r.json(); }).then(function (rep) {
      if (!S.active) return;
      S.report = rep; paintGrids();
    });
  }

  function loadCell() {
    var yrs = yrsPicked();
    if (!yrs.length) { S.cell = null; S.ladder = null; paintCard(); return; }
    var q = "stat=" + encodeURIComponent(S.qstat) + "&line=" + encodeURIComponent(S.qline) +
      "&bf=" + encodeURIComponent(S.qbf) + "&seasons=" + yrs.join(",");
    fetch("/api/outs-lab/cell?" + q).then(function (r) { return r.json(); }).then(function (c) {
      if (!S.active) return;
      S.cell = c; paintCard();
      if (c && !c.error) {
        fetch("/api/outs-lab/ladder?stat=" + encodeURIComponent(S.qstat) +
          "&line=" + encodeURIComponent(S.qline) + "&seasons=" + yrs.join(",")
        ).then(function (r) { return r.json(); }).then(function (l) {
          if (!S.active) return;
          S.ladder = l; paintLadder();
        });
      } else { S.ladder = null; paintLadder(); }
    });
  }

  function startVtbf() {
    var box = document.getElementById("olb-vyrs");
    var yrs = (box ? box.value : "").split(",").map(function (x) { return +x.trim(); }).filter(Boolean);
    fetch("/api/outs-lab/vtbf/run", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ token: tok(), seasons: yrs })
    }).then(function (r) { return r.json(); }).then(function (j) {
      if (j.error || j.reason) {
        var p = document.getElementById("olb-prog");
        if (p) p.textContent = j.error || j.reason;
        if (j.error === "bad token") return;
      }
      if (S.poll) clearInterval(S.poll);
      S.poll = setInterval(refresh, 1500);
      refresh();
    });
  }

  function loadVtbf() {
    fetch("/api/outs-lab/vtbf").then(function (r) { return r.json(); }).then(function (v) {
      if (!S.active) return;
      S.vtbf = v.runs || []; paintVtbf();
    });
  }

  function vtbfRow(name, s) {
    if (!s || !s.bets) return "<tr><td>" + name + '</td><td colspan="5" class="meta">no bets</td></tr>';
    return "<tr><td>" + name + "</td><td>" + s.bets + "</td><td>" + esc(s.record) +
      "</td><td><b>" + (s.units > 0 ? "+" : "") + s.units + "u</b></td><td>" + s.roi_pct +
      "%</td><td>" + s.brier + (s.market_brier ? ' <span class="meta">mkt ' + s.market_brier + "</span>" : "") + "</td></tr>";
  }

  function paintVtbf() {
    var el = document.getElementById("olb-vout");
    if (!el) return;
    var runs = S.vtbf || [];
    if (!runs.length) { el.innerHTML = '<div class="meta">no runs yet</div>'; return; }
    var h = "", seen = {};
    runs.forEach(function (run) {
      if (seen[run.season]) return;
      seen[run.season] = 1;
      var r = run.report;
      h += "<h3>" + r.season + " \u2014 " + r.starts_priced + " starts priced over " +
        r.days_walked + " days \u00b7 grid " + (r.grid_seasons || []).join("/") +
        " (" + r.grid_starts + " starts) \u00b7 TBF ratio " + r.tbf_ratio + "</h3>";
      ["league", "pitcher"].forEach(function (arm) {
        h += '<div class="meta" style="margin-top:6px"><b>' + arm + " arm</b>" +
          (arm === "pitcher" ? " (own trailing rate slotted in)" : "") + "</div>";
        h += '<div style="overflow-x:auto"><table><tr><th>market</th><th>bets</th><th>W-L</th><th>units</th><th>ROI</th><th>Brier</th></tr>';
        var a = r.arms[arm];
        h += vtbfRow("outs \u26a0\ufe0f circular", a.outs) + vtbfRow("hits", a.hits) +
          vtbfRow("walks", a.walks) + vtbfRow("TOTAL", a.total);
        h += "</table></div>";
      });
      var sk = [];
      Object.keys(r.skips || {}).forEach(function (k) { sk.push(k + " \u00d7" + r.skips[k]); });
      h += '<div class="meta">skips: ' + (sk.join(" \u00b7 ") || "none") + "</div>";
      h += '<div class="meta">suspect >20% excluded: ' + r.suspect_excluded +
        " \u00b7 pitcher-rate missing: " + r.pitcher_rate_missing +
        " \u00b7 credits: api " + ((r.odds_fetches || {}).odds_api || 0) +
        " / archive " + ((r.odds_fetches || {}).odds_hit || 0) + "</div>";
      h += '<div class="meta">' + esc(r.policy) + " \u00b7 " + esc(r.note) + "</div>";
    });
    el.innerHTML = h;
  }

  // -------------------------------------------------------------- paint
  function paintProgress() {
    var p = document.getElementById("olb-prog");
    if (!p) return;
    var st = S.state || {};
    p.textContent = (st.running ? "running: " : "") + (st.progress || "idle") +
      (st.error ? " \u2014 " + st.error : "") + (S.err ? " \u2014 " + S.err : "");
    var m = /game (\d+)\/(\d+)/.exec(st.progress || "");
    var fill = document.getElementById("olb-fill");
    if (fill) fill.style.width = m ? Math.round(100 * m[1] / m[2]) + "%" : (st.running ? "2%" : "0%");
  }

  function paintCov() {
    var el = document.getElementById("olb-cov");
    if (!el) return;
    if (!S.cov.length) { el.innerHTML = '<div class="meta">dataset empty \u2014 fetch a season</div>'; return; }
    var h = "<table><tr><th>season</th><th>games</th><th>starts</th><th>first</th><th>last</th></tr>";
    S.cov.forEach(function (c) {
      h += "<tr><td>" + c.season + "</td><td>" + c.games + "</td><td>" + c.starts +
        "</td><td>" + esc(c.first) + "</td><td>" + esc(c.last) + "</td></tr>";
    });
    el.innerHTML = h + "</table>";
  }

  function paintYears() {
    var el = document.getElementById("olb-yrs");
    if (!el) return;
    var h = "";
    S.cov.forEach(function (c) {
      var on = S.checked && S.checked.has(c.season);
      h += '<label style="margin-right:10px;white-space:nowrap"><input type="checkbox" data-yr="' +
        c.season + '"' + (on ? " checked" : "") + "> " + c.season + "</label>";
    });
    el.innerHTML = h;
    var boxes = el.querySelectorAll("input[data-yr]");
    for (var i = 0; i < boxes.length; i++) {
      boxes[i].onchange = function () {
        var y = +this.getAttribute("data-yr");
        if (this.checked) S.checked.add(y); else S.checked.delete(y);
        loadAll();
      };
    }
  }

  function paintCard() {
    var el = document.getElementById("olb-card");
    if (!el) return;
    var c = S.cell;
    if (!c) { el.innerHTML = '<div class="meta">pick at least one season</div>'; return; }
    if (c.error) { el.innerHTML = '<div class="olb-hl" style="color:#f28b82">' + esc(c.error) + "</div>"; return; }
    var gap = ((c.empirical - c.binomial) * 100).toFixed(1);
    var h = '<div class="olb-hl">' +
      "<div>P(over " + c.line + " " + esc(c.label).toLowerCase() + " within the first " + c.bf +
      " batters faced) \u2014 needs " + c.need + "+</div>" +
      '<span class="olb-big">' + pct(c.empirical) + "</span> " +
      '<span class="meta">fair ' + esc(c.fair) + " \u00b7 " + c.hit + " of " + c.starts +
      " starts that faced \u2265" + c.bf + " \u00b7 95% CI " + pct(c.ci95[0]) + "\u2013" + pct(c.ci95[1]) + "</span>" +
      '<div class="meta">closed form (binomial, pooled ' + c.per_bf_rate + "/BF): " + pct(c.binomial) +
      " \u2014 empirical runs <b>" + (gap > 0 ? "+" : "") + gap + " pts</b> vs iid math</div>" +
      (c.note ? '<div style="color:#f0b26b">' + esc(c.note) + "</div>" : "") +
      '<div class="meta">seasons ' + c.seasons.join(", ") + "</div></div>";
    el.innerHTML = h;
  }

  function paintLadder() {
    var el = document.getElementById("olb-ladder");
    if (!el) return;
    var l = S.ladder;
    if (!l || l.error || !l.rows) { el.innerHTML = ""; return; }
    var h = "<h3>Over " + l.line + " " + esc(l.label).toLowerCase() +
      " at every batters-faced checkpoint</h3>" +
      '<div style="overflow-x:auto"><table><tr><th>n BF</th><th>starts</th><th>empirical</th><th>95% CI</th><th>fair</th><th>binomial</th></tr>';
    l.rows.forEach(function (x) {
      var hot = x.n === +S.qbf ? ' style="background:#161a24"' : "";
      h += "<tr" + hot + "><td>" + x.n + "</td><td>" + x.starts + "</td><td><b>" + pct(x.empirical) +
        "</b></td><td>" + pct(x.ci95[0]) + "\u2013" + pct(x.ci95[1]) + "</td><td>" + esc(x.fair) +
        "</td><td>" + pct(x.binomial) + "</td></tr>";
    });
    el.innerHTML = h + "</table></div>";
  }

  function paintGrids() {
    var tabs = document.getElementById("olb-stats");
    var out = document.getElementById("olb-out");
    if (!tabs || !out) return;
    var r = S.report;
    if (!r || r.error || !r.grids) {
      tabs.innerHTML = "";
      out.innerHTML = '<div class="meta">' + esc(r && r.error ? r.error : "no report yet") + "</div>";
      return;
    }
    var th = "";
    Object.keys(r.grids).forEach(function (k) {
      th += '<button class="olb-tab' + (k === S.stat ? " on" : "") + '" data-st="' + k + '">' +
        esc(r.grids[k].label) + "</button> ";
    });
    tabs.innerHTML = th;
    var btns = tabs.querySelectorAll("button[data-st]");
    for (var i = 0; i < btns.length; i++) {
      btns[i].onclick = function () { S.stat = this.getAttribute("data-st"); paintGrids(); };
    }
    var g = r.grids[S.stat];
    var h = '<div class="meta">seasons ' + esc(r.label) + " \u00b7 " + r.starts + " starts \u00b7 " + r.tbf +
      " BF \u00b7 league outs/BF " + r.outs_per_bf + " \u00b7 retire rate " + r.retire_rate +
      " \u00b7 extra outs/BF " + r.extra_outs_per_bf + "</div>";
    if (r.unknown_events && Object.keys(r.unknown_events).length) {
      h += '<div style="color:#f0b26b">unknown eventTypes counted as PA: ' +
        esc(JSON.stringify(r.unknown_events)) + "</div>";
    }
    h += "<h3>" + esc(g.label) + ": P(over line | first n batters) \u2014 empirical, binomial(" +
      g.per_bf_rate + '/BF) underneath</h3><div style="overflow-x:auto"><table><tr><th>n BF</th><th>starts</th>';
    g.lines.forEach(function (l) { h += "<th>" + l + "</th>"; });
    h += "</tr>";
    g.rows.forEach(function (x) {
      h += "<tr" + (x.n >= 17 && x.n <= 23 ? ' style="background:#161a24"' : "") + "><td>" + x.n +
        "</td><td>" + x.starts + "</td>";
      g.lines.forEach(function (l) {
        h += "<td><b>" + pct(x[l]) + '</b><br><span class="meta">' + pct(x["binom_" + l]) + "</span></td>";
      });
      h += "</tr>";
    });
    h += "</table></div>";
    if (S.stat === "outs") {
      h += "<h3>Extra outs by base state (DP / CS / pickoff per PA)</h3><table><tr><th>runners on</th><th>PA</th><th>extra outs / PA</th><th>reach rate</th></tr>";
      (r.base_state || []).forEach(function (x) {
        h += "<tr><td>" + x.runners_on + "</td><td>" + x.pa + "</td><td>" + x.extra_outs_per_pa +
          "</td><td>" + pct(x.reach_rate) + "</td></tr>";
      });
      h += "</table><h3>Out rates by how the start went (TBF \u2265 12)</h3><table><tr><th>reach-rate bucket</th><th>starts</th><th>outs/BF</th><th>retire</th><th>extra/BF</th></tr>";
      (r.traffic || []).forEach(function (x) {
        h += "<tr><td>" + esc(x.reach_rate_bucket) + "</td><td>" + x.starts + "</td><td>" + x.outs_per_bf +
          "</td><td>" + x.retire_rate + "</td><td>" + x.extra_outs_per_bf + "</td></tr>";
      });
      h += "</table><h3>Reach rate by time through the order</h3><table><tr><th>TTO</th><th>PA</th><th>reach rate</th></tr>";
      (r.tto || []).forEach(function (x) {
        h += "<tr><td>" + x.tto + "</td><td>" + x.pa + "</td><td>" + pct(x.reach_rate) + "</td></tr>";
      });
      h += "</table><h3>For contrast: P(15+ outs) conditioned on FINAL TBF = n (hook-polluted)</h3><table><tr><th>final TBF</th><th>starts</th><th>P(15+)</th></tr>";
      (r.final_tbf_view || []).forEach(function (x) {
        h += "<tr><td>" + x.n + "</td><td>" + x.starts + "</td><td>" + pct(x.p15) + "</td></tr>";
      });
      h += "</table>";
    }
    out.innerHTML = h;
  }

  // -------------------------------------------------------------- shell
  function shell() {
    var mem = window.LAB_TOKEN_MEM || "";
    return '' +
      '<style>' +
      '#olb .olb-hl{background:#161a24;border:1px solid #2a3040;border-radius:8px;padding:10px 12px;margin:10px 0}' +
      '#olb .olb-big{font-size:22px;font-weight:700;margin-right:6px}' +
      '#olb .olb-bar{height:6px;background:#1e2430;border-radius:3px;margin:8px 0}' +
      '#olb .olb-bar>div{height:100%;background:#2b6cb0;border-radius:3px;width:0}' +
      '#olb #olb-prog{font-family:ui-monospace,monospace;font-size:12px;min-height:16px}' +
      '#olb .olb-tab{background:#1e2430;color:#fff;border:0;border-radius:6px;padding:7px 11px;cursor:pointer}' +
      '#olb .olb-tab.on{background:#2b6cb0}' +
      '#olb h3{font-size:14px;margin:16px 0 6px;color:#9aa}' +
      '#olb input,#olb select{background:#181b22;color:#eee;border:1px solid #333;border-radius:6px;padding:6px;font-size:14px}' +
      '#olb table{border-collapse:collapse;font-size:13px;margin-top:4px}' +
      '#olb th,#olb td{padding:5px 6px;text-align:right;border-bottom:1px solid #22262e;white-space:nowrap}' +
      '#olb th{color:#9aa;font-weight:600}#olb td:first-child,#olb th:first-child{text-align:left}' +
      '#olb .olb-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}' +
      '</style>' +
      '<div id="olb" class="board">' +
      '<h2>Outs Lab</h2>' +
      '<div class="meta">MLB play-by-play only \u2014 free, no lines, no model. Every number is a count of real starts.</div>' +
      '<h3>1. Fetch seasons into the dataset</h3>' +
      '<div class="olb-row"><input id="olb-token" type="password" placeholder="lab token" style="width:110px" value="' + esc(mem) + '">' +
      '<input id="olb-fetchyrs" style="width:200px" value="2021,2022,2023,2024,2025,2026">' +
      '<button class="olb-tab on" id="olb-fetchbtn">Fetch</button>' +
      '<span class="meta">resumable \u2014 stored games are skipped</span></div>' +
      '<div id="olb-prog">idle</div><div class="olb-bar"><div id="olb-fill"></div></div>' +
      '<div id="olb-cov"></div>' +
      '<h3>2. Ask the dataset anything</h3>' +
      '<div class="olb-row" id="olb-yrs"></div>' +
      '<div class="olb-row">' +
      '<select id="olb-qstat"><option value="outs">Outs</option><option value="hits">Hits allowed</option>' +
      '<option value="walks">Walks</option><option value="ks">Strikeouts</option></select>' +
      '<label>line <input id="olb-qline" style="width:64px" value="' + esc(S.qline) + '"></label>' +
      '<label>batters faced <input id="olb-qbf" style="width:56px" value="' + esc(S.qbf) + '"></label>' +
      '<button class="olb-tab on" id="olb-askbtn">Ask</button>' +
      '</div>' +
      '<div id="olb-card"></div><div id="olb-ladder"></div>' +
      '<h3>Full grids \u2014 every posted line at every BF</h3>' +
      '<div class="olb-row" id="olb-stats"></div>' +
      '<div id="olb-out"></div>' +
      '<h3>3. Vegas-TBF market backtest (uses Odds API credits)</h3>' +
      '<div class="meta">implied TBF = outs + hits + walks lines \u00d7 real TBF ratio \u2192 empirical grid \u2192 flat 1u vs closing prices. No outs line = start skipped (the gate). Grid from prior seasons only.</div>' +
      '<div class="olb-row"><input id="olb-vyrs" style="width:190px" value="2023,2024,2025,2026">' +
      '<button class="olb-tab on" id="olb-vrunbtn">Run</button>' +
      '<span class="meta">archive-first \u2014 a season re-run costs ~0 credits</span></div>' +
      '<div id="olb-vout"></div>' +
      '</div>';
  }

  function wire() {
    document.getElementById("olb-fetchbtn").onclick = startFetch;
    document.getElementById("olb-vrunbtn").onclick = startVtbf;
    var ask = function () {
      S.qstat = document.getElementById("olb-qstat").value;
      S.qline = document.getElementById("olb-qline").value;
      S.qbf = document.getElementById("olb-qbf").value;
      loadCell();
    };
    document.getElementById("olb-askbtn").onclick = ask;
    document.getElementById("olb-qstat").onchange = ask;
    document.getElementById("olb-qline").onchange = ask;
    document.getElementById("olb-qbf").onchange = ask;
    document.getElementById("olb-qstat").value = S.qstat;
  }

  function show() {
    var main = document.getElementById("main");
    if (!main) return;
    S.active = true;
    main.innerHTML = shell();
    wire();
    refresh(loadAll);
  }

  function hide() {
    S.active = false;
    if (S.poll) { clearInterval(S.poll); S.poll = null; }
  }

  function markButtons(on) {
    var mine = document.getElementById("olb-tab-btn");
    if (!mine) return;
    var sibs = mine.parentNode ? mine.parentNode.querySelectorAll("button") : [];
    for (var i = 0; i < sibs.length; i++) {
      if (sibs[i] === mine) continue;
      if (on) { sibs[i].classList.remove("active"); sibs[i].classList.remove("on"); }
    }
    if (on) { mine.classList.add("active"); mine.classList.add("on"); }
    else { mine.classList.remove("active"); mine.classList.remove("on"); }
  }

  function install() {
    var tabs = document.querySelector(".tabs");
    if (!tabs || document.getElementById("olb-tab-btn")) return;
    var btn = document.createElement("button");
    btn.id = "olb-tab-btn";
    btn.textContent = "Outs Lab";
    var ref = tabs.querySelector("button");
    if (ref) btn.className = ref.className.replace(/\bactive\b/, "").replace(/\bon\b/, "").trim();
    tabs.appendChild(btn);
    var orig = window.showView;
    btn.onclick = function () {
      if (typeof window.showView === "function") window.showView(VIEW);
    };
    window.showView = function (v) {
      if (v === VIEW) { markButtons(true); show(); return; }
      hide();
      markButtons(false);
      if (typeof orig === "function") return orig(v);
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install);
  } else {
    install();
  }
})();
