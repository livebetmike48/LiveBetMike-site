/* Outs Lab tab — the play-by-play dataset behind the outs model.
 *
 * Self-registering like pprops_tab.js: adds its own tab button and wraps
 * showView, so index.html is untouched (app.py injects the script tag).
 * Delete the script tag and this file together and the site is as it was.
 *
 * Talks to outs_lab.py:  GET /api/outs-lab            state + coverage
 *                        GET /api/outs-lab/report     grids for a season set
 *                        POST /api/outs-lab/run       fetch seasons (token)
 * Zero odds credits. Zero model. Every cell is a count of real starts.
 */
(function () {
  var TAB_ID = "tab-outslab";
  var STAT = "outs";
  var REPORT = null;
  var COV = [];
  var CHECKED = null;          // Set of seasons ticked; null = all
  var POLL = null;
  var TOKEN_MEM = "";

  function addTab() {
    var tabs = document.querySelector(".tabs");
    if (!tabs || document.getElementById(TAB_ID)) return;
    var btn = document.createElement("button");
    btn.className = "tab";
    btn.id = TAB_ID;
    btn.textContent = "Outs Lab";
    btn.onclick = function () { showView("outslab"); };
    tabs.appendChild(btn);
  }

  var _showView = window.showView;
  window.showView = function (v) {
    if (v === "outslab") {
      VIEW = "outslab";
      ["board", "bullpen", "model", "kboard", "pitchers", "lab", "pprops"].forEach(function (t) {
        var el = document.getElementById("tab-" + t);
        if (el) el.classList.remove("active");
      });
      var me = document.getElementById(TAB_ID);
      if (me) me.classList.add("active");
      renderOutsLab();
      return;
    }
    var me2 = document.getElementById(TAB_ID);
    if (me2) me2.classList.remove("active");
    if (POLL) { clearInterval(POLL); POLL = null; }
    return _showView.apply(this, arguments);
  };

  // Same token convention as the Model Lab: a password box, remembered
  // for the session. If the Lab tab already has one filled in, reuse it.
  function token() {
    var el = document.getElementById("outslab-token");
    if (el && el.value) TOKEN_MEM = el.value;
    if (!TOKEN_MEM && typeof LAB_TOKEN_MEM !== "undefined" && LAB_TOKEN_MEM) TOKEN_MEM = LAB_TOKEN_MEM;
    return TOKEN_MEM;
  }

  var pct = function (x) { return x == null ? "–" : (100 * x).toFixed(1) + "%"; };
  var esc = function (s) { return String(s).replace(/</g, "&lt;"); };

  window.renderOutsLab = async function renderOutsLab() {
    var main = document.getElementById("main");
    main.innerHTML = "<div class='state'>Loading the outs dataset…<div class='bar'><i></i></div></div>";
    var s;
    try { s = await (await fetch("/api/outs-lab")).json(); }
    catch (e) {
      main.innerHTML = "<div class='state'>Server busy — retrying…</div>";
      setTimeout(function () { if (VIEW === "outslab") renderOutsLab(); }, 8000);
      return;
    }
    if (VIEW !== "outslab") return;
    COV = s.coverage || [];
    if (CHECKED === null) CHECKED = new Set(COV.map(function (c) { return c.season; }));
    main.innerHTML = "";

    var hdr = document.createElement("div");
    hdr.className = "gamehdr";
    hdr.textContent = "Outs Lab — how often starters clear a line at each batters-faced level";
    main.appendChild(hdr);

    var note = document.createElement("div");
    note.className = "meta";
    note.style.cssText = "margin:2px 0 6px";
    note.innerHTML = "MLB play-by-play only. <b>No lines, no model, zero odds credits.</b> Every cell is a count of real starts: " +
      "among starts that faced at least <i>n</i> batters, how many had cleared the line after batter <i>n</i>. " +
      "Binomial with the pooled per-batter rate sits under each cell — where they match, the closed form holds; where they don't, it doesn't.";
    main.appendChild(note);

    // ---- 1. fetch controls
    var ctl = document.createElement("div");
    ctl.className = "winrow";
    ctl.style.flexWrap = "wrap";
    var defaultYrs = COV.length ? "" : "2021,2022,2023,2024,2025";
    ctl.innerHTML =
      '<input id="outslab-years" placeholder="seasons e.g. 2021,2022,2023" value="' + defaultYrs + '" ' +
        'style="background:var(--panel-2);color:var(--white);border:1.5px solid var(--line);border-radius:6px;padding:6px 10px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;width:230px">' +
      '<input id="outslab-token" type="password" placeholder="lab token" value="' + esc(TOKEN_MEM) + '" ' +
        'style="background:var(--panel-2);color:var(--white);border:1.5px solid var(--line);border-radius:6px;padding:6px 10px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;width:120px">' +
      '<button class="winbtn" onclick="OUTSLAB_FETCH()">Fetch seasons</button>' +
      '<span class="meta" id="outslab-status">' + statusText(s.state) + '</span>';
    main.appendChild(ctl);
    var help = document.createElement("div");
    help.className = "meta";
    help.style.cssText = "margin:4px 0 0";
    help.textContent = "Free and resumable — games already stored are skipped, so re-run a season any time. ~12–15 min per season.";
    main.appendChild(help);

    // ---- coverage
    var covBox = document.createElement("div");
    covBox.className = "board";
    covBox.style.cssText = "margin-top:10px;max-width:560px";
    if (!COV.length) {
      covBox.innerHTML = "<div class='meta' style='padding:12px 14px'>Dataset empty — fetch a season above.</div>";
    } else {
      var ch = "<table><thead><tr><th style='text-align:left;padding-left:14px'>Season</th><th>Games</th><th>Starts</th><th>First</th><th>Last</th></tr></thead><tbody>";
      COV.forEach(function (c) {
        ch += "<tr><td class='name' style='cursor:default'>" + c.season + "</td><td>" + c.games + "</td><td>" + c.starts +
          "</td><td>" + c.first + "</td><td>" + c.last + "</td></tr>";
      });
      covBox.innerHTML = ch + "</tbody></table>";
    }
    main.appendChild(covBox);

    if (!COV.length) { if (s.state && s.state.running) startPoll(); return; }

    // ---- 2. query controls
    var q = document.createElement("div");
    q.className = "winrow";
    q.style.flexWrap = "wrap";
    var qh = "";
    COV.forEach(function (c) {
      qh += '<label class="meta" style="margin-left:0;cursor:pointer"><input type="checkbox" value="' + c.season + '" ' +
        (CHECKED.has(c.season) ? "checked" : "") + ' onchange="OUTSLAB_TOGGLE(' + c.season + ',this.checked)"> ' + c.season + '</label>';
    });
    qh += '<button class="winbtn" onclick="OUTSLAB_QUERY()">Query</button>';
    q.innerHTML = qh;
    main.appendChild(q);

    var out = document.createElement("div");
    out.id = "outslab-out";
    main.appendChild(out);

    if (s.state && s.state.running) startPoll();
    if (!REPORT) await OUTSLAB_QUERY(); else renderReport();
  };

  function statusText(st) {
    if (!st) return "";
    if (st.running) return "⏳ " + (st.progress || "starting…");
    if (st.error) return "✖ " + st.error;
    return st.progress ? st.progress : "idle";
  }

  function startPoll() {
    if (POLL) clearInterval(POLL);
    POLL = setInterval(async function () {
      if (VIEW !== "outslab") { clearInterval(POLL); POLL = null; return; }
      var s;
      try { s = await (await fetch("/api/outs-lab")).json(); } catch (e) { return; }
      var el = document.getElementById("outslab-status");
      if (el) el.textContent = statusText(s.state);
      if (!s.state.running) {
        clearInterval(POLL); POLL = null;
        CHECKED = null; REPORT = null;
        renderOutsLab();              // refresh coverage + grids
      }
    }, 2000);
  }

  window.OUTSLAB_FETCH = async function () {
    var yrs = (document.getElementById("outslab-years").value || "")
      .split(",").map(function (x) { return parseInt(x.trim(), 10); }).filter(Boolean);
    var el = document.getElementById("outslab-status");
    if (!yrs.length) { el.textContent = "type at least one season"; return; }
    var r = await fetch("/api/outs-lab/run", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({token: token(), seasons: yrs})});
    var j = await r.json();
    el.textContent = j.error ? "✖ " + j.error : (j.reason ? j.reason : "⏳ starting " + yrs.join(", ") + "…");
    if (j.started) startPoll();
  };

  window.OUTSLAB_TOGGLE = function (yr, on) { if (on) CHECKED.add(yr); else CHECKED.delete(yr); };

  window.OUTSLAB_QUERY = async function () {
    var yrs = Array.from(CHECKED).sort();
    var out = document.getElementById("outslab-out");
    if (!out) return;
    if (!yrs.length) { out.innerHTML = "<div class='meta'>pick at least one season</div>"; return; }
    out.innerHTML = "<div class='state' style='padding:20px 0'>Counting…<div class='bar'><i></i></div></div>";
    try { REPORT = await (await fetch("/api/outs-lab/report?seasons=" + yrs.join(","))).json(); }
    catch (e) { out.innerHTML = "<div class='meta'>server busy</div>"; return; }
    renderReport();
  };

  window.OUTSLAB_STAT = function (k) { STAT = k; renderReport(); };

  function renderReport() {
    var out = document.getElementById("outslab-out");
    if (!out) return;
    var r = REPORT;
    if (!r || r.error) { out.innerHTML = "<div class='meta'>" + (r ? esc(r.error) : "no report") + "</div>"; return; }
    var h = r.headline, g = r.grids[STAT];
    var html = "";

    html += "<div class='board' style='margin-top:12px;max-width:760px'><div style='padding:12px 14px'>" +
      "<div class='meta' style='margin:0'>seasons " + esc(r.label) + " · " + r.starts + " starts · " + r.tbf + " batters faced</div>" +
      "<div style='margin-top:6px'>" + h.question + "</div>" +
      "<div style='font-size:26px;font-weight:700;color:var(--blue);font-family:\"IBM Plex Mono\",monospace'>" + pct(h.empirical) +
      " <span class='meta'>(" + h.fair + " · n=" + h.starts + " starts that faced ≥17)</span></div>" +
      "<div class='meta' style='margin:4px 0 0'>closed form: " + pct(h.closed_form_outs_rate) + " using outs/BF · " +
      pct(h.closed_form_retire_rate) + " using retire rate</div>" +
      "<div class='meta' style='margin:2px 0 0'>league outs/BF <b>" + r.outs_per_bf + "</b> · retire rate <b>" + r.retire_rate +
      "</b> · extra outs/BF (DP/CS/pickoff) <b>" + r.extra_outs_per_bf + "</b></div></div></div>";

    if (r.unknown_events && Object.keys(r.unknown_events).length)
      html += "<div class='meta' style='color:#e0a12f;margin-top:6px'>⚠️ unknown StatsAPI eventTypes counted as PA: " + esc(JSON.stringify(r.unknown_events)) + "</div>";

    // stat tabs
    html += "<div class='winrow'>";
    Object.keys(r.grids).forEach(function (k) {
      html += '<button class="winbtn ' + (k === STAT ? "active" : "") + '" onclick="OUTSLAB_STAT(\'' + k + '\')">' + esc(r.grids[k].label) + '</button>';
    });
    html += "<span class='meta'>per-BF rate " + g.per_bf_rate + " · rows 17–23 highlighted</span></div>";

    // the grid
    html += "<div class='board' style='margin-top:8px'><table><thead><tr><th style='text-align:left;padding-left:14px'>Batters faced</th><th>Starts</th>";
    g.lines.forEach(function (l) { html += "<th>" + l + "</th>"; });
    html += "</tr></thead><tbody>";
    g.rows.forEach(function (x) {
      var focus = x.n >= 17 && x.n <= 23;
      html += "<tr" + (focus ? " style='background:#141a22'" : "") + "><td class='name' style='cursor:default'>" + x.n + "</td><td>" + x.starts + "</td>";
      g.lines.forEach(function (l) {
        html += "<td><b>" + pct(x[l]) + "</b><br><span style='color:var(--gray);font-size:11px'>" + pct(x["binom_" + l]) + "</span></td>";
      });
      html += "</tr>";
    });
    html += "</tbody></table></div>";
    html += "<div class='meta' style='margin-top:6px'>top number = empirical share of starts that had cleared the line after batter <i>n</i>; grey = binomial(n, per-BF rate). " +
      "Starts count is the same across a row: everyone who faced at least <i>n</i>.</div>";

    if (STAT === "outs") {
      html += "<div class='board' style='margin-top:14px'><table><thead><tr><th style='text-align:left;padding-left:14px' colspan='10'>14.5 outs — detail with 95% CI and dispersion vs binomial</th></tr>" +
        "<tr><th style='text-align:left;padding-left:14px'>n</th><th>Starts</th><th>Empirical</th><th>95% CI</th><th>Fair</th><th>Binom (outs/BF)</th><th>Binom (retire)</th><th>Mean outs</th><th>Expected</th><th>Dispersion</th></tr></thead><tbody>";
      r.ladder.forEach(function (x) {
        html += "<tr" + (x.focus ? " style='background:#141a22'" : "") + "><td class='name' style='cursor:default'>" + x.n + "</td><td>" + x.starts + "</td><td><b>" + pct(x.empirical) +
          "</b></td><td>" + pct(x.ci95[0]) + "–" + pct(x.ci95[1]) + "</td><td>" + x.fair + "</td><td>" + pct(x.pred_outs_rate) + "</td><td>" + pct(x.pred_retire_rate) +
          "</td><td>" + x.mean_outs_after + "</td><td>" + x.expected_outs_after + "</td><td>" + (x.dispersion == null ? "–" : x.dispersion) + "</td></tr>";
      });
      html += "</tbody></table></div>";

      html += "<div style='display:flex;gap:14px;flex-wrap:wrap;margin-top:14px'>";
      html += "<div class='board' style='flex:1;min-width:300px'><table><thead><tr><th style='text-align:left;padding-left:14px' colspan='4'>Extra outs by base state</th></tr>" +
        "<tr><th style='text-align:left;padding-left:14px'>Runners on</th><th>PA</th><th>Extra outs / PA</th><th>Reach rate</th></tr></thead><tbody>";
      r.base_state.forEach(function (x) {
        html += "<tr><td class='name' style='cursor:default'>" + x.runners_on + "</td><td>" + x.pa + "</td><td>" + x.extra_outs_per_pa + "</td><td>" + pct(x.reach_rate) + "</td></tr>";
      });
      html += "</tbody></table></div>";
      html += "<div class='board' style='flex:1;min-width:300px'><table><thead><tr><th style='text-align:left;padding-left:14px' colspan='5'>Out rates by how the start went (TBF ≥ 12)</th></tr>" +
        "<tr><th style='text-align:left;padding-left:14px'>Reach-rate bucket</th><th>Starts</th><th>Outs/BF</th><th>Retire</th><th>Extra/BF</th></tr></thead><tbody>";
      r.traffic.forEach(function (x) {
        html += "<tr><td class='name' style='cursor:default'>" + x.reach_rate_bucket + "</td><td>" + x.starts + "</td><td>" + x.outs_per_bf + "</td><td>" + x.retire_rate + "</td><td>" + x.extra_outs_per_bf + "</td></tr>";
      });
      html += "</tbody></table></div>";
      html += "<div class='board' style='flex:1;min-width:220px'><table><thead><tr><th style='text-align:left;padding-left:14px' colspan='3'>Reach rate by time through order</th></tr>" +
        "<tr><th style='text-align:left;padding-left:14px'>TTO</th><th>PA</th><th>Reach</th></tr></thead><tbody>";
      r.tto.forEach(function (x) {
        html += "<tr><td class='name' style='cursor:default'>" + x.tto + "</td><td>" + x.pa + "</td><td>" + pct(x.reach_rate) + "</td></tr>";
      });
      html += "</tbody></table></div>";
      html += "<div class='board' style='flex:1;min-width:220px'><table><thead><tr><th style='text-align:left;padding-left:14px' colspan='3'>Contrast: conditioned on FINAL TBF (hook-polluted)</th></tr>" +
        "<tr><th style='text-align:left;padding-left:14px'>Final TBF</th><th>Starts</th><th>P(15+)</th></tr></thead><tbody>";
      r.final_tbf_view.forEach(function (x) {
        html += "<tr><td class='name' style='cursor:default'>" + x.n + "</td><td>" + x.starts + "</td><td>" + pct(x.p15) + "</td></tr>";
      });
      html += "</tbody></table></div></div>";
    }
    out.innerHTML = html;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addTab);
  } else {
    addTab();
  }
})();
