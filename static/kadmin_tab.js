/* Admin tab — pitch-count caps and exclusions for today's / tomorrow's K board.
 *
 * Self-registering like the props tab, so index.html is never edited.
 *
 * Why this is admin-only and pre-game only, stated where it can't be
 * lost: the forward log's whole value is that reads freeze before first
 * pitch and are never revised. A cap or an exclusion set BEFORE the game
 * is new information the model didn't have. The same action AFTER the
 * game is editing your own scorecard. The server enforces it (a graded
 * start refuses the override); this UI just says so out loud.
 */
(function () {
  var TAB_ID = "tab-kadmin";
  var KADAY = 0;
  var TOKEN = "";

  function addTab() {
    var tabs = document.querySelector(".tabs");
    if (!tabs || document.getElementById(TAB_ID)) return;
    var btn = document.createElement("button");
    btn.className = "tab";
    btn.id = TAB_ID;
    btn.textContent = "Admin";
    btn.onclick = function () { showView("kadmin"); };
    tabs.appendChild(btn);
  }

  var _prev = window.showView;
  window.showView = function (v) {
    if (v === "kadmin") {
      VIEW = "kadmin";
      ["board", "bullpen", "model", "kboard", "pitchers", "lab", "pprops"].forEach(function (t) {
        var el = document.getElementById("tab-" + t);
        if (el) el.classList.remove("active");
      });
      var me = document.getElementById(TAB_ID);
      if (me) me.classList.add("active");
      renderKAdmin();
      return;
    }
    var me2 = document.getElementById(TAB_ID);
    if (me2) me2.classList.remove("active");
    return _prev.apply(this, arguments);
  };

  window.KADMIN_DAY = function (d) { KADAY = d; renderKAdmin(); };

  window.renderKAdmin = async function renderKAdmin() {
    var main = document.getElementById("main");
    main.innerHTML = "<div class='state'>Loading board…<div class='bar'><i></i></div></div>";
    var board, ovr;
    try {
      var r1 = await fetch("/api/kboard?d=" + KADAY);
      board = await r1.json();
      var r2 = await fetch("/api/koverrides?d=" + KADAY);
      ovr = await r2.json();
    } catch (e) {
      main.innerHTML = "<div class='state'>Server busy — try again in a minute.</div>";
      return;
    }
    if (VIEW !== "kadmin") return;
    main.innerHTML = "";

    var hdr = document.createElement("div");
    hdr.className = "gamehdr";
    hdr.textContent = "Admin — pitch counts & exclusions";
    main.appendChild(hdr);

    var note = document.createElement("div");
    note.className = "meta";
    note.style.cssText = "margin:2px 0 8px";
    note.innerHTML =
      "<b style='color:var(--white)'>Pre-game only.</b> A cap or exclusion set before first pitch is " +
      "information the model didn't have. The same action after the game is editing your own scorecard — " +
      "the server refuses it, and excluded reads are flagged in the log, never deleted, and reported as " +
      "their own line in the record.<br>" +
      "<b>Pitch limit</b> → converted to a batter cap using <i>his own</i> pitches-per-PA, then his real " +
      "start lengths are capped at it (short outings stay — a limit truncates the top, it doesn't make " +
      "the outing certain). <b>Exclude</b> → still priced and shown, not counted in units or Brier.";
    main.appendChild(note);

    var ctl = document.createElement("div");
    ctl.className = "winrow";
    ctl.innerHTML =
      '<button class="winbtn ' + (KADAY === 0 ? "active" : "") + '" onclick="KADMIN_DAY(0)">Today</button>' +
      '<button class="winbtn ' + (KADAY === 1 ? "active" : "") + '" onclick="KADMIN_DAY(1)">Tomorrow</button>' +
      '<input id="kadmin-token" type="password" placeholder="lab token" ' +
      'style="background:var(--panel-2);color:var(--white);border:1.5px solid var(--line);' +
      'border-radius:6px;padding:6px 10px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;width:130px">' +
      '<span class="meta" id="kadmin-status">' + (board.date || "") + '</span>';
    main.appendChild(ctl);

    var starters = (board.starters || []).filter(function (s) { return s.status === "ok"; });
    if (!starters.length) {
      main.innerHTML += "<div class='state'>No modelable starters on this slate yet.</div>";
      return;
    }
    var over = (ovr && ovr.overrides) || {};

    var box = document.createElement("div");
    box.className = "board";
    var inp = "background:#0d1420;border:1px solid #26303e;color:var(--white);" +
      "padding:4px 6px;border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:12px";
    var h = "<table><thead><tr>" +
      "<th style='text-align:left;padding-left:14px'>Starter</th><th>Line</th><th>Mean K</th><th>TBF</th>" +
      "<th>Pitch limit</th><th>Exclude</th><th style='text-align:left'>Reason</th><th></th>" +
      "</tr></thead><tbody>";
    starters.forEach(function (s) {
      var o = over[s.starter_id] || over[String(s.starter_id)] || {};
      var active = o.tbf_cap || o.exclude;
      h += "<tr" + (active ? " style='background:rgba(224,161,47,.08)'" : "") + ">" +
        "<td class='name' style='cursor:default'>" + s.starter +
          " <span class='vs'>" + s.team + " v " + s.opp +
          (o.tbf_cap ? " · capped " + o.tbf_cap + " TBF" : "") +
          (o.exclude ? " · <b style='color:#e0a12f'>EXCLUDED</b>" : "") + "</span></td>" +
        "<td>" + (s.line != null ? s.line : "–") + "</td>" +
        "<td>" + (s.mean_k != null ? s.mean_k : "–") + "</td>" +
        "<td>" + (s.tbf_mean != null ? s.tbf_mean : "–") + "</td>" +
        "<td><input id='kal-" + s.starter_id + "' type='number' min='20' max='120' " +
          "placeholder='–' value='" + (o.pitch_limit || "") + "' style='" + inp + ";width:64px'></td>" +
        "<td><input id='kax-" + s.starter_id + "' type='checkbox'" + (o.exclude ? " checked" : "") + "></td>" +
        "<td style='text-align:left'><input id='kan-" + s.starter_id + "' " +
          "placeholder='why (shown in the record)' value='" + (o.note || "").replace(/"/g, "&quot;") +
          "' style='" + inp + ";width:190px'></td>" +
        "<td><button class='winbtn' onclick='kadminSave(" + s.starter_id + ")'>Save</button>" +
        (active ? " <button class='winbtn' onclick='kadminClear(" + s.starter_id + ")'>Clear</button>" : "") +
        "</td></tr>";
    });
    h += "</tbody></table>";
    box.innerHTML = h;
    main.appendChild(box);

    var foot = document.createElement("div");
    foot.className = "meta";
    foot.style.cssText = "margin-top:8px";
    foot.innerHTML = "Saving forces the board to rebuild, so a cap shows up in Mean K / TBF within a minute or " +
      "two (a full slate takes that long to re-price). <b>A cap may not change much</b> — 90 pitches is close to " +
      "a normal start, so if his average outing already sits under the cap it only trims his longest ones; the " +
      "save message tells you whether it binds. A read already frozen today keeps its logged price — the cap " +
      "changes what the board shows and, if you exclude it, whether it counts.";
    main.appendChild(foot);
  };

  window.kadminSave = async function (sid) {
    var st = document.getElementById("kadmin-status");
    var tok = document.getElementById("kadmin-token");
    if (tok && tok.value) TOKEN = tok.value;
    var limit = parseInt((document.getElementById("kal-" + sid) || {}).value) || null;
    var excl = !!(document.getElementById("kax-" + sid) || {}).checked;
    var note = ((document.getElementById("kan-" + sid) || {}).value || "").trim();
    if (!limit && !excl) { st.textContent = "✖ set a pitch limit or tick exclude"; return; }
    st.textContent = "saving…";
    var r = await fetch("/api/koverride", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({token: TOKEN, d: KADAY, starter_id: sid,
                            pitch_limit: limit, exclude: excl, note: note})});
    var j = await r.json();
    if (j.error) { st.textContent = "✖ " + j.error; return; }
    st.textContent = "✓ saved" + (j.tbf_cap ? " — capped at " + j.tbf_cap + " batters" : "") +
      (j.binds ? " · binds " + j.binds : "") +
      (j.exclude ? " · excluded from the record" : "") +
      " · rebuilding the board…";
    // the board rebuilds in the background (~1-2 min for a full slate);
    // re-render after a beat so the new Mean K / TBF show up on their own
    setTimeout(renderKAdmin, 1500);
    setTimeout(function () { if (VIEW === "kadmin") renderKAdmin(); }, 45000);
  };

  window.kadminClear = async function (sid) {
    var st = document.getElementById("kadmin-status");
    var tok = document.getElementById("kadmin-token");
    if (tok && tok.value) TOKEN = tok.value;
    var r = await fetch("/api/koverride", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({token: TOKEN, d: KADAY, starter_id: sid, clear: true})});
    var j = await r.json();
    st.textContent = j.error ? "✖ " + j.error : "✓ cleared";
    renderKAdmin();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addTab);
  } else {
    addTab();
  }
})();
