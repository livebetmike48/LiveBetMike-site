/* Pitcher Props tab — hits + walks allowed, from the pprops engine.
 *
 * Self-registering on purpose: it adds its own tab button and wraps
 * showView, so index.html needs ONE line (the script tag) instead of
 * three edits scattered through a 1,300-line file. Delete the script tag
 * and this file together and the site is exactly as it was.
 *
 * BETA: the engine has never been backtested and has no calibration
 * curve. The page says so, loudly, in the same place every time.
 */
(function () {
  var TAB_ID = "tab-pprops";
  var PPDAY = 0;

  function addTab() {
    var tabs = document.querySelector(".tabs");
    if (!tabs || document.getElementById(TAB_ID)) return;
    var btn = document.createElement("button");
    btn.className = "tab";
    btn.id = TAB_ID;
    btn.textContent = "Pitcher Props";
    btn.onclick = function () { showView("pprops"); };
    tabs.appendChild(btn);
  }

  // Wrap the site's own view switcher so the existing tabs keep working
  // untouched and ours joins the rotation.
  var _showView = window.showView;
  window.showView = function (v) {
    if (v === "pprops") {
      VIEW = "pprops";
      ["board", "bullpen", "model", "kboard", "pitchers", "lab"].forEach(function (t) {
        var el = document.getElementById("tab-" + t);
        if (el) el.classList.remove("active");
      });
      var me = document.getElementById(TAB_ID);
      if (me) me.classList.add("active");
      renderPProps();
      return;
    }
    var me2 = document.getElementById(TAB_ID);
    if (me2) me2.classList.remove("active");
    return _showView.apply(this, arguments);
  };

  window.renderPProps = async function renderPProps() {
    var main = document.getElementById("main");
    main.innerHTML = "<div class='state'>Projecting today's starters — first build of the day pulls every arm's season rows, so give it a minute…<div class='bar'><i></i></div></div>";
    var j;
    try {
      var r = await fetch("/api/pprops?d=" + PPDAY);
      j = await r.json();
    } catch (e) {
      main.innerHTML = "<div class='state'>Server busy — retrying…</div>";
      setTimeout(function () { if (VIEW === "pprops") renderPProps(); }, 8000);
      return;
    }
    if (VIEW !== "pprops") return;
    if (j.error) { main.innerHTML = "<div class='state'>✖ " + j.error + "</div>"; return; }
    main.innerHTML = "";

    var hdr = document.createElement("div");
    hdr.className = "gamehdr";
    hdr.textContent = (PPDAY === 1 ? "Tomorrow's" : "Today's") + " pitcher props — BETA (engine only)";
    main.appendChild(hdr);

    var warn = document.createElement("div");
    warn.className = "meta";
    warn.style.cssText = "margin:2px 0 6px;color:#e0a12f";
    warn.innerHTML = "⚠️ <b>Never backtested. No calibration curve. Nothing logged.</b> " +
      "The K model went through 2,935 graded predictions and a fitted curve before it was trusted — " +
      "this has been through neither. Numbers here are the raw engine talking. Eyeball them; don't bet them yet.";
    main.appendChild(warn);

    var dayRow = document.createElement("div");
    dayRow.className = "winrow";
    dayRow.innerHTML =
      '<button class="winbtn ' + (PPDAY === 0 ? "active" : "") + '" onclick="PPROPS_DAY(0)">Today</button>' +
      '<button class="winbtn ' + (PPDAY === 1 ? "active" : "") + '" onclick="PPROPS_DAY(1)">Tomorrow</button>' +
      '<span class="meta">league per-PA: hits ' + (j.league_rates && j.league_rates.hits) +
      ' · walks ' + (j.league_rates && j.league_rates.walks) +
      ' · league P/PA ' + (j.league_pppa || "n/a") +
      ' <span title="computed from this slate\'s starters, never an assumed constant">(computed)</span></span>';
    main.appendChild(dayRow);

    var rows = (j.starters || []).filter(function (s) { return s.hits || s.walks; });
    var noRead = (j.starters || []).filter(function (s) { return !s.hits && !s.walks; });
    if (!rows.length) {
      main.innerHTML += "<div class='state'>No modelable starters yet — probables may not be posted.</div>";
      return;
    }

    var box = document.createElement("div");
    box.className = "board";
    var h = "<table><thead><tr>" +
      "<th style='text-align:left;padding-left:14px'>Starter</th>" +
      "<th style='text-align:left'>vs</th>" +
      "<th title='from the live K model — same number as the K Board, calibration curve and all'>Ks</th>" +
      "<th>Hits</th><th>Walks</th><th>TBF</th>" +
      "<th title='his own pitches per plate appearance'>P/PA</th>" +
      "<th title=\"Savant's official index_hits park factor, applied to hits only — walks aren't a dimensions event\">Park</th>" +
      "<th title='what the opponent workload adjustment WOULD apply at full weight — currently OFF in the model'>Workload ×</th>" +
      "<th>Lineup</th></tr></thead><tbody>";
    rows.forEach(function (s) {
      var lu = s.lineup_source === "posted"
        ? "posted"
        : "<span style='color:#e0a12f' title='no lineup posted yet — the opponent\\'s most recent real order vs this hand'>" + s.lineup_source + "</span>";
      var wf = s.workload_factor;
      var wcol = wf > 1.02 ? "#4caf7d" : (wf < 0.98 ? "#d7483a" : "#7a8694");
      h += "<tr><td class='name' style='cursor:default'>" + s.starter +
        " <span class='vs'>" + s.team + " " + s.hand + "HP</span></td>" +
        "<td style='text-align:left;color:#7a8694;font-family:Inter'>" + s.opp + "</td>" +
        "<td style='color:var(--blue);font-weight:600'>" +
          (s.strikeouts ? s.strikeouts.mean : "–") + "</td>" +
        "<td>" + (s.hits ? s.hits.mean : "–") + "</td>" +
        "<td>" + (s.walks ? s.walks.mean : "–") + "</td>" +
        "<td>" + (s.hits ? s.hits.tbf_mean : "–") + "</td>" +
        "<td>" + (s.pppa || "–") + "</td>" +
        "<td>" + (s.park_hits || "–") + "</td>" +
        "<td style='color:" + wcol + "'>" + wf + "</td>" +
        "<td>" + lu + "</td></tr>";
    });
    h += "</tbody></table>";
    box.innerHTML = h;
    main.appendChild(box);

    var lad = document.createElement("div");
    lad.className = "meta";
    lad.style.cssText = "margin-top:8px";
    lad.innerHTML = "Hits/Walks are projected MEANS over the starter's real workload distribution — " +
      "same engine as the K board, different event set. Over-line probabilities per starter are in " +
      "<a href='/api/pprops?d=" + PPDAY + "' style='color:var(--blue)'>the JSON</a> until they earn a column. " +
      "Earned runs is deliberately absent: it depends on sequencing, which per-PA math cannot price honestly.";
    main.appendChild(lad);

    if (noRead.length) {
      var nr = document.createElement("div");
      nr.className = "meta";
      nr.style.cssText = "margin-top:6px";
      nr.textContent = "No read (house minimums): " + noRead.map(function (s) { return s.starter; }).join(" · ");
      main.appendChild(nr);
    }
  };

  window.PPROPS_DAY = function (d) { PPDAY = d; renderPProps(); };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addTab);
  } else {
    addTab();
  }
})();
