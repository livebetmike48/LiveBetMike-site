/* Longer backtest windows + a computed "Season" option.
 *
 * index.html hardcodes the day dropdowns (3..120 for backtests, 7..120 for
 * market tests). Rather than edit that file, this watches for those
 * selects appearing and extends them. Season is COMPUTED server-side from
 * Opening Night, so the label is right every day without a code change --
 * a hardcoded number would be wrong tomorrow.
 */
(function () {
  var SEASON = null;      // {start, days, lab_days, market_days}
  var EXTRA_LAB = [150, 180, 200];
  var EXTRA_MARKET = [150, 180, 200];

  async function loadSeason() {
    if (SEASON) return SEASON;
    try {
      var r = await fetch("/api/season");
      SEASON = await r.json();
    } catch (e) {
      SEASON = null;
    }
    return SEASON;
  }

  function extend(sel, extras) {
    if (!sel || sel.dataset.daysExtended) return;
    var have = {};
    for (var i = 0; i < sel.options.length; i++) have[sel.options[i].value] = 1;
    extras.forEach(function (d) {
      if (have[String(d)]) return;
      var o = document.createElement("option");
      o.value = String(d);
      o.textContent = String(d);
      sel.appendChild(o);
    });
    if (SEASON && SEASON.days) {
      var o = document.createElement("option");
      o.value = String(SEASON.days);
      o.textContent = "Season (" + SEASON.days + "d)";
      sel.appendChild(o);
    }
    sel.dataset.daysExtended = "1";
  }

  async function sweep() {
    await loadSeason();
    extend(document.getElementById("lab-days"), EXTRA_LAB);
    extend(document.getElementById("klab-days"), EXTRA_LAB);
    extend(document.getElementById("market-days"), EXTRA_MARKET);
    extend(document.getElementById("kmarket-days"), EXTRA_MARKET);
  }

  // The Lab renders on demand and re-renders while runs are in flight, so
  // watch #main rather than trying to hook the render function.
  function start() {
    sweep();
    var main = document.getElementById("main");
    if (!main || !window.MutationObserver) return;
    var pending = null;
    new MutationObserver(function () {
      clearTimeout(pending);
      pending = setTimeout(sweep, 120);
    }).observe(main, {childList: true, subtree: true});
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
