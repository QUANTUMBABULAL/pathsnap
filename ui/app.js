/* PathSnap evaluator UI (Laptop 2).
 * Talks to the FastAPI backend only: GET /health, GET /trips, POST /predict {trip_id}.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const fmtPct = (v) => (v == null ? "–" : (100 * v).toFixed(2) + "%");
  const fmtDur = (s) => (s >= 3600 ? `${(s / 3600).toFixed(1)} h` : `${Math.round(s / 60)} min`);

  let apiBase = "";
  try { apiBase = localStorage.getItem("pathsnap.api") || ""; } catch (_) { /* storage unavailable */ }
  apiBase = apiBase || (window.PATHSNAP_CONFIG && window.PATHSNAP_CONFIG.API_BASE) || "http://localhost:8000";
  $("apiBase").value = apiBase;

  const state = { trips: [], selected: null, scores: {}, running: false, last: null, replay: null };

  // ------------------------------------------------------------------ map
  const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([30, 10], 2);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, className: "basemap",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
  const renderer = L.canvas({ padding: 0.3 });
  const layers = {
    network: L.layerGroup().addTo(map),
    gps: L.layerGroup().addTo(map),
    matched: L.layerGroup(),
    route: L.layerGroup().addTo(map),
    markers: L.layerGroup().addTo(map),
  };
  const css = getComputedStyle(document.documentElement);
  const color = (v) => css.getPropertyValue(v).trim();

  document.querySelectorAll(".layers input").forEach((cb) => {
    cb.addEventListener("change", () => {
      const lg = layers[cb.dataset.layer];
      cb.checked ? lg.addTo(map) : map.removeLayer(lg);
    });
  });

  // ------------------------------------------------------------------ log
  function log(msg, cls = "") {
    const li = document.createElement("li");
    const t = new Date().toLocaleTimeString([], { hour12: false });
    li.textContent = `${t}  ${msg}`;
    li.title = li.textContent;
    if (cls) li.className = cls;
    $("log").prepend(li);
  }
  $("clearLog").onclick = () => ($("log").innerHTML = "");

  function setStatus(kind, text) {
    const p = $("statusPill");
    p.className = `pill pill-${kind}`;
    $("statusText").textContent = text;
  }

  async function api(path, opts = {}) {
    const t0 = performance.now();
    const res = await fetch(apiBase.replace(/\/$/, "") + path, {
      ...opts, headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    });
    const ms = Math.round(performance.now() - t0);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      log(`${opts.method || "GET"} ${path} → ${res.status} (${ms} ms) ${detail}`, "err");
      throw new Error(detail);
    }
    const body = await res.json();
    log(`${opts.method || "GET"} ${path} → ${res.status} (${ms} ms)`, "ok");
    return body;
  }

  // ------------------------------------------------------------------ connect
  async function connect() {
    setStatus("busy", "connecting…");
    $("modelInfo").innerHTML = "";
    try {
      const h = await api("/health");
      const t = await api("/trips");
      state.trips = t.trips;
      setStatus("ok", "connected");
      const rows = [
        ["Model", `${h.model}${h.tuned ? " (tuned)" : ""}`],
        ["HMM α / β", `${h.hmm.alpha} / ${h.hmm.beta}`],
        ["Candidates", `r=${h.candidate_config.radius} m, k=${h.candidate_config.k}`],
      ];
      $("modelInfo").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
      renderTrips();
    } catch (e) {
      setStatus("err", "unreachable");
      state.trips = [];
      renderTrips();
      log(`cannot reach ${apiBase}: ${e.message}`, "err");
    }
  }

  $("apiForm").addEventListener("submit", (e) => {
    e.preventDefault();
    apiBase = $("apiBase").value.trim();
    try { localStorage.setItem("pathsnap.api", apiBase); } catch (_) {}
    connect();
  });

  // ------------------------------------------------------------------ trips
  function renderTrips() {
    const ul = $("tripList");
    ul.innerHTML = "";
    $("tripCount").textContent = state.trips.length ? `${state.trips.length} trips` : "–";
    state.trips.forEach((t, i) => {
      const li = document.createElement("li");
      li.className = "trip";
      li.setAttribute("role", "option");
      li.tabIndex = 0;
      li.setAttribute("aria-selected", state.selected === t.trip_id ? "true" : "false");
      const sc = state.scores[t.trip_id];
      const scCls = sc == null ? "" : sc >= 0.9 ? "" : sc >= 0.75 ? "mid" : "low";
      li.innerHTML = `
        <span class="idx">${i + 1}</span>
        <span><div class="tid">${t.trip_id}</div>
          <div class="meta">${t.n_points.toLocaleString()} pts · ${t.length_km} km · ${fmtDur(t.duration_s)}</div></span>
        <span class="score ${scCls}">${sc == null ? "" : (100 * sc).toFixed(1) + "%"}</span>`;
      li.onclick = () => select(t.trip_id);
      li.ondblclick = () => { select(t.trip_id); run(); };
      li.onkeydown = (e) => { if (e.key === "Enter") { select(t.trip_id); run(); } };
      ul.appendChild(li);
    });
    updateRunBtn();
  }

  function select(id) {
    state.selected = id;
    document.querySelectorAll(".trip").forEach((li) => {
      li.setAttribute("aria-selected", li.querySelector(".tid").textContent === id ? "true" : "false");
    });
    const t = state.trips.find((x) => x.trip_id === id);
    if (t && !state.running) map.flyTo(t.center, 12, { duration: 0.8 });
    updateRunBtn();
  }

  function updateRunBtn() {
    const b = $("runBtn");
    b.disabled = !state.selected || state.running;
    b.querySelector(".btn-label").textContent = state.running
      ? "Running…" : state.selected ? `Run map-matching · ${state.selected}` : "Select a trip";
  }
  $("runBtn").onclick = () => run();

  // ------------------------------------------------------------------ predict
  async function run() {
    if (!state.selected || state.running) return;
    const id = state.selected;
    state.running = true;
    updateRunBtn();
    stopReplay();
    $("busy").classList.remove("hidden");
    $("busyText").textContent = `Running pipeline on API for trip ${id}…`;
    try {
      const r = await api("/predict", { method: "POST", body: JSON.stringify({ trip_id: id }) });
      state.last = r;
      state.scores[id] = r.metrics.top1_accuracy_hmm;
      draw(r);
      showResults(r);
      renderTrips();
      log(`trip ${id}: Top-1 ${fmtPct(r.metrics.top1_accuracy_hmm)}, route F1 ${fmtPct(r.metrics.route_f1)}`, "ok");
    } catch (e) {
      $("resSub").textContent = `Error: ${e.message}`;
    } finally {
      state.running = false;
      $("busy").classList.add("hidden");
      updateRunBtn();
    }
  }

  function draw(r) {
    Object.values(layers).forEach((l) => l.clearLayers());
    $("mapEmpty").classList.add("hidden");
    L.polyline(r.network, { color: color("--net"), weight: 1.6, opacity: 0.65, renderer, interactive: false }).addTo(layers.network);
    r.gps.forEach((p, i) =>
      L.circleMarker(p, { radius: 2.6, stroke: false, fillColor: color("--gps"), fillOpacity: 0.85, renderer })
        .bindTooltip(`GPS #${i}`).addTo(layers.gps));
    r.matched_points.forEach((p) =>
      L.circleMarker(p, { radius: 2.4, stroke: false, fillColor: color("--match"), fillOpacity: 0.95, renderer }).addTo(layers.matched));
    L.polyline(r.predicted_route, { color: "#fff", weight: 7, opacity: 0.7, renderer, interactive: false }).addTo(layers.route);
    L.polyline(r.predicted_route, { color: color("--route"), weight: 4, opacity: 0.95, renderer })
      .bindTooltip(`Predicted route · ${r.n_route_arcs} arcs`).addTo(layers.route);
    const start = r.gps[0], end = r.gps[r.gps.length - 1];
    L.circleMarker(start, { radius: 7, color: "#fff", weight: 2, fillColor: "#16a34a", fillOpacity: 1 }).bindTooltip("Start").addTo(layers.markers);
    L.circleMarker(end, { radius: 7, color: "#fff", weight: 2, fillColor: "#111827", fillOpacity: 1 }).bindTooltip("End").addTo(layers.markers);
    map.fitBounds(r.bounds, { padding: [40, 40] });
    $("replayBtn").disabled = false;
  }

  function showResults(r) {
    const m = r.metrics;
    $("resTitle").textContent = `Trip ${r.trip_id}`;
    $("resSub").textContent = `${r.split} · model ${r.model}`;
    $("mTop1").textContent = fmtPct(m.top1_accuracy_hmm);
    $("mTop1Bar").style.width = `${(100 * m.top1_accuracy_hmm).toFixed(1)}%`;
    const d = m.top1_accuracy_hmm - m.top1_accuracy_classifier;
    $("mDelta").textContent = `${d >= 0 ? "+" : ""}${(100 * d).toFixed(2)} pp vs classifier`;
    $("mDelta").style.color = d >= 0 ? color("--accent") : color("--bad");
    $("mCls").textContent = fmtPct(m.top1_accuracy_classifier);
    $("mF1").textContent = fmtPct(m.route_f1);
    $("mP").textContent = fmtPct(m.route_precision);
    $("mR").textContent = fmtPct(m.route_recall);
    $("mCont").textContent = fmtPct(m.route_continuity);
    $("mMiss").textContent = fmtPct(m.candidate_miss_rate);
    $("kPts").textContent = r.n_points.toLocaleString();
    $("kCand").textContent = r.n_candidates.toLocaleString();
    $("kArcs").textContent = r.n_route_arcs.toLocaleString();
    $("kGaps").textContent = `${r.gaps.filled} (${r.gaps.unfilled} open)`;

    const parts = [
      ["Preprocess", r.timings.preprocess_s, "#64748b"],
      ["Features", r.timings.features_s, "#60a5fa"],
      ["Classifier", r.timings.classifier_s, "#a78bfa"],
      ["HMM/Viterbi", r.timings.hmm_s, "#34d399"],
      ["Route fill", r.timings.route_s, "#fbbf24"],
      ["Scoring", r.timings.scoring_s, "#f43f5e"],
    ];
    const total = parts.reduce((s, p) => s + p[1], 0) || 1;
    $("kTotal").textContent = `${total.toFixed(2)} s`;
    $("latBar").innerHTML = parts.map(([n, v, c]) => `<span title="${n} ${v.toFixed(3)} s" style="width:${(100 * v) / total}%;background:${c}"></span>`).join("");
    $("latLegend").innerHTML = parts.map(([n, v, c]) => `<li><i style="background:${c}"></i>${n}<b>${(1000 * v).toFixed(0)} ms</b></li>`).join("");
  }

  // ------------------------------------------------------------------ replay
  function stopReplay() {
    if (state.replay) { cancelAnimationFrame(state.replay.raf); state.replay.marker.remove(); state.replay = null; }
    $("replayBtn").textContent = "▶ Replay";
  }
  $("replayBtn").onclick = () => {
    if (state.replay) return stopReplay();
    const pts = state.last && state.last.predicted_route;
    if (!pts || pts.length < 2) return;
    const icon = L.divIcon({ className: "", html: '<div class="car-icon"></div>', iconSize: [14, 14] });
    const marker = L.marker(pts[0], { icon, interactive: false }).addTo(map);
    const duration = Math.min(20000, Math.max(6000, pts.length * 12));
    const t0 = performance.now();
    $("replayBtn").textContent = "■ Stop";
    const step = (now) => {
      const f = Math.min(1, (now - t0) / duration);
      const x = f * (pts.length - 1), i = Math.floor(x), w = x - i;
      const a = pts[i], b = pts[Math.min(i + 1, pts.length - 1)];
      marker.setLatLng([a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w]);
      if (f < 1) state.replay.raf = requestAnimationFrame(step); else stopReplay();
    };
    state.replay = { marker, raf: requestAnimationFrame(step) };
  };

  connect();
})();
