// Trang dự án: Phần 1 xem dự báo (bản đồ lớn + khung 3 bước + biểu đồ rộng), Phần 2 kết quả nghiên cứu
// (xếp hạng, theo hạn, nhìn lại 2 năm, lũ, bốn mức, giải thích, "nếu... thì..."). Số liệu lấy từ API máy chủ.
import * as maplibregl from "/static/vendor/maplibre/maplibre-gl.mjs";
import { t, getLang, setLang, applyStatic } from "/static/js/i18n.js";
import { aboutHtml } from "/static/js/about.js";

const $ = id => document.getElementById(id);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const LEVEL_VARS = ["--lv-low", "--lv-medium", "--lv-high", "--lv-vhigh", "--lv-flood"];
const LEVELS = ["low", "medium", "high", "very_high"];
const CORES = ["lstm", "gru", "transformer", "s4d", "mamba"];
const CORE = {lstm: "LSTM", gru: "GRU", transformer: "Transformer", s4d: "S4D", mamba: "Mamba"};
const CORE_COLOR = {lstm: "#1d6fa5", gru: "#c2622d", transformer: "#5b8c3a", s4d: "#8a4f9e", mamba: "#0f8a84"};
const MODES = ["past", "hybrid", "gefs", "perfect"];
const MODE_COLOR = {past: "#b5651d", hybrid: "#7a5cc0", gefs: "#1d6fa5", perfect: "#2f7f6f", persistence: "#8b9397"};
const LINE_COLORS = ["#1d6fa5", "#c2622d", "#5b8c3a", "#8a4f9e"];
const ENS_COLOR = "#0f8a84";
const TASKS = ["c2", "h", "d"];
const MAX_COMPARE = 3;
const FONT = "Be Vietnam Pro, system-ui, sans-serif";
const BASE = ["SeqLSTM_Q", "Persistence_qmax"];

// Phóng tới mức này thì chấm đổi thành 2 cột (thực đo, dự báo); chạy thời gian ×1 = 400 ms/ngày
const BAR_ZOOM = 8.5;
const STEP_MS = 400;
const CACHE_MAX = 2500;

const state = {
  site: null, catalog: null, dates: [], ids: [], basin: "2", lastBasin: "2", day: 0,
  legendOpen: (() => { try { const v = localStorage.getItem("legendOpen"); if (v) return v === "1"; } catch { /* bỏ qua */ } return window.innerWidth > 700; })(),
  task: "c2", main: {c2: null, h: null, d: null}, lead: 1,
  compare: {c2: ["SeqLSTM_Q"], h: ["Persistence_qmax"], d: ["SeqLSTM_Q"]}, ensemble: false,
  colorBy: "observed", cmpTask: "c2", cmpLead: 1, cmpRun: null, nseRef: "",
  sec: {flood: {run: null, lead: 1}, levels: {run: null, lead: 1}},
  leaderboard: [], seeds: null, dayData: null, levelsNow: null, speed: 1, resultsReady: false, wiDone: false,
};
const cache = new Map();

// ---------- tiện ích ----------
async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(body?.error ?? `HTTP ${res.status}`);
  return body;
}
async function cached(path) {
  if (!cache.has(path)) {
    // Giới hạn bộ nhớ đệm: bỏ mục cũ nhất (Map giữ thứ tự thêm vào)
    if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value);
    cache.set(path, api(path).catch(e => { cache.delete(path); throw e; }));
  }
  return cache.get(path);
}
const qs = obj => "?" + new URLSearchParams(Object.entries(obj).flatMap(([k, v]) => [].concat(v).map(x => [k, x])));
const locale = () => (getLang() === "vi" ? "vi-VN" : "en-GB");
const fmtFixed = (v, d) => (v == null || !Number.isFinite(v) ? "–" : new Intl.NumberFormat(locale(), {minimumFractionDigits: d, maximumFractionDigits: d}).format(v));
const fmt = (v, d = 2) => (v == null || !Number.isFinite(v) ? "–" : new Intl.NumberFormat(locale(), {maximumFractionDigits: d, minimumFractionDigits: 0}).format(v));
const fmtQ = v => (v == null || !Number.isFinite(v) ? "–" : fmt(v, v < 10 ? 2 : v < 100 ? 1 : 0));
const fmtSigned = (v, d = 3) => (v == null || !Number.isFinite(v) ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "") + fmtFixed(Math.abs(v), d));
const fmtDate = (iso, long = false) => new Intl.DateTimeFormat(locale(), long ? {weekday: "short", day: "numeric", month: "short", year: "numeric"} : {day: "numeric", month: "short", year: "numeric"}).format(new Date(iso + "T00:00:00"));
const days = n => t(n === 1 ? "card.day1" : "card.days", {n});
const pct = (v, d = 0) => (v == null || !Number.isFinite(v) ? "–" : new Intl.NumberFormat(locale(), {style: "percent", maximumFractionDigits: d}).format(v));
const separators = () => (getLang() === "vi" ? ",." : ".,");
function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { box.hidden = true; }, 6000);
}
async function run(task) {
  try { await task(); } catch (error) { console.error(error); toast(t("err.load", {msg: error.message})); }
}
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  node.append(...children.filter(c => c != null && c !== false));
  return node;
}
const legendKey = (color, label, {style = "", value = null} = {}) =>
  el("span", {class: "key"}, el("i", {class: style, style: `color:${color}`}), el("span", {}, label), value == null ? null : el("em", {}, value));
const isRegion = () => state.basin === "all";
const testYears = () => `${state.dates[0].slice(0, 4)}–${state.dates.at(-1).slice(0, 4)}`;
// Cỡ chữ biểu đồ: nhãn trục, chú thích không dưới 12 px; màn hẹp giảm nhẹ
const chartFont = () => ({family: FONT, size: window.innerWidth < 640 ? 12 : 14, color: css("--ink")});
function baseLayout(extra = {}) {
  return {font: chartFont(), separators: separators(), paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    margin: {l: 64, r: 16, t: 12, b: 52}, showlegend: false, hoverlabel: {font: {family: FONT, size: 14}}, ...extra};
}
const plotConfig = {responsive: true, displaylogo: false, displayModeBar: false};

// ---------- tên mô hình dễ hiểu (kèm mã cấu hình) ----------
const models = () => state.catalog.models;
const model = runId => models().find(m => m.run === runId);
function parseConf(conf) {
  const m = /^([a-z0-9]+)_L(\d+)_x(\d+)_lr/.exec(conf ?? "");
  return m ? {core: m[1], depth: Number(m[2]), size: Number(m[3])} : null;
}
const coreOf = m => m?.core ?? parseConf(m?.configuration)?.core ?? null;
function variant(m) {
  // "2 lớp, cùng cỡ" - phần mô tả cấu hình, không có tên kiến trúc
  const c = parseConf(m.configuration);
  if (!c) return "";
  const unit = ["s4d", "mamba"].includes(c.core) ? "m.blocks" : "m.layers";
  return `${t(unit, {n: c.depth})}, ${t(c.size === 1 ? "m.same" : "m.big")}`;
}
function shortName(m) {
  if (!m) return "";
  if (m.run === "SeqLSTM_Q") return t("m.biascast");
  if (m.run === "Persistence_qmax") return t("m.persistence");
  const core = CORE[coreOf(m)];
  if (m.stage === "H") return `${core} · ${t("mode.short." + m.mode)}`;
  const text = `${core} ${variant(m)}`;
  return m.stage === "D" ? `${text} · ${t("m.lookback")}` : text;
}
const taskOf = m => (m.stage === "C2" ? "c2" : m.stage === "D" ? "d" : m.stage === "H" ? "h" : null);
const taskLabel = task => t("task." + task);
const valNse = m => m.validation_nse ?? m.validation_nse_lead_mean ?? -Infinity;
const stageModels = stage => models().filter(m => m.stage === stage).sort((a, b) => valNse(b) - valNse(a));
const STAGE = {c2: "C2", h: "H", d: "D"};
const activeRun = () => state.main[state.task];
const activeLead = () => (state.task === "h" ? state.lead : 1);
const hasLead = (runId, lead) => model(runId)?.leads.includes(lead);
function compareRuns() {
  const lead = activeLead();
  return state.compare[state.task].filter(r => r !== activeRun() && hasLead(r, lead));
}

// ---------- mức lưu lượng ----------
function level(q, basin) {
  if (q == null || !Number.isFinite(q)) return -1;
  const flood = state.site.flood_m3_s[basin]?.["1.5"];
  if (flood != null && q >= flood) return 4;
  const cuts = state.site.stations[basin].levels_m3_s;
  let k = 0;
  while (k < 3 && q >= cuts[k]) k += 1;
  return k;
}
const levelColor = k => (k < 0 ? css("--lv-none") : css(LEVEL_VARS[k]));
const levelText = k => (k < 0 ? t("lv.none") : t("lv." + k));

// ---------- bản đồ ----------
let map, nseMap, miniMap;
function baseStyle(withLabels = true) {
  const origin = location.origin;
  const layers = basemaps.layers("protomaps", basemaps.namedFlavor("light"), {lang: getLang()});
  return {
    version: 8,
    glyphs: `${origin}/static/vendor/basemap/fonts/{fontstack}/{range}.pbf`,
    sprite: `${origin}/static/vendor/basemap/sprites/v4/light`,
    sources: {protomaps: {type: "vector", url: `pmtiles://${origin}/tiles/basemap.pmtiles`,
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · <a href="https://protomaps.com">Protomaps</a>'}},
    layers: withLabels ? layers : layers.filter(l => l.type !== "symbol"),
  };
}
function studyBounds() {
  const lons = state.catalog.points.map(p => p.lon), lats = state.catalog.points.map(p => p.lat);
  return [[Math.min(...lons) - 0.3, Math.min(...lats) - 0.2], [Math.max(...lons) + 0.3, Math.max(...lats) + 0.2]];
}
// Ranh giới vùng nghiên cứu: 71 vòng ngoài của hợp các lưu vực => một MultiPolygon
const outlineGeo = () => ({type: "Feature", properties: {}, geometry: {type: "MultiPolygon", coordinates: state.catalog.outline.map(r => [r])}});
function gaugeGeoJSON(props) {
  return {type: "FeatureCollection", features: state.catalog.points.map(p => ({
    type: "Feature", geometry: {type: "Point", coordinates: [p.lon, p.lat]},
    properties: {id: p.id, ...props(p.id)}}))};
}
function newMap(container) {
  const m = new maplibregl.Map({container, style: baseStyle(), bounds: studyBounds(), fitBoundsOptions: {padding: 30},
    maxBounds: [[5.5, 44.5], [20, 52.5]], minZoom: 5, maxZoom: 14, attributionControl: {compact: true}, dragRotate: false, pitchWithRotate: false});
  m.addControl(new maplibregl.NavigationControl({showCompass: false}), "bottom-right");
  m.touchZoomRotate.disableRotation();
  return m;
}
const selFilter = () => ["==", ["get", "id"], isRegion() ? "__none__" : state.basin];
function addOutline(m, width = 2.2) {
  m.addSource("study", {type: "geojson", data: outlineGeo()});
  m.addLayer({id: "study-line", type: "line", source: "study", paint: {"line-color": css("--deep"), "line-width": width, "line-opacity": 0.85}});
}
// Màu chấm theo mức: đọc feature-state (đổi mỗi ngày không phải nạp lại dữ liệu điểm)
const levelMatch = key => ["match", ["coalesce", ["feature-state", key], -1],
  0, levelColor(0), 1, levelColor(1), 2, levelColor(2), 3, levelColor(3), 4, levelColor(4), levelColor(-1)];
const dotKey = () => (state.colorBy === "observed" ? "lo" : "lp");
function dotRadius(key) {
  return ["interpolate", ["linear"], ["zoom"], 6, ["match", ["coalesce", ["feature-state", key], -1], 4, 6.5, 3, 5.5, 4], 8.5, 7];
}
// Biểu tượng 2 cột: trái thực đo, phải dự báo; cao theo mức, màu theo mức; vẽ khi bản đồ cần đến
const BAR_H = [8, 13, 18, 23, 28];
function barImage(lo, lp) {
  const r = 2, w = 24, h = 34, g = document.createElement("canvas").getContext("2d", {willReadFrequently: true});
  g.canvas.width = w * r; g.canvas.height = h * r; g.scale(r, r);
  const bar = (x, k) => {
    const bh = k < 0 ? 4 : BAR_H[k], y = h - 3 - bh;
    g.fillStyle = "#ffffff"; g.fillRect(x - 1.5, y - 1.5, 10, bh + 1.5);
    g.fillStyle = levelColor(k); g.fillRect(x, y, 7, bh);
  };
  bar(3.5, lo); bar(13.5, lp);
  g.fillStyle = "#ffffff"; g.fillRect(0.5, h - 4, 23, 3);
  g.fillStyle = css("--ink"); g.fillRect(1.5, h - 3, 21, 1.5);
  return {width: w * r, height: h * r, data: g.getImageData(0, 0, w * r, h * r).data};
}
function addBarImages() {
  // Nạp sẵn 36 hình cột (6 mức thực đo × 6 mức dự báo, gồm "không có số liệu") => lớp cột không phải chờ ảnh khi vẽ
  for (let lo = -1; lo <= 4; lo += 1) for (let lp = -1; lp <= 4; lp += 1) {
    const id = `bar_${lo}_${lp}`;
    if (!map.hasImage(id)) map.addImage(id, barImage(lo, lp), {pixelRatio: 2});
  }
}
function addMainLayers() {
  addBarImages();
  map.addSource("basins", {type: "geojson", data: "/api/basins"});
  map.addLayer({id: "study-fill", type: "fill", source: "basins", paint: {"fill-color": css("--accent"), "fill-opacity": isRegion() ? 0.1 : 0.04}});
  map.addLayer({id: "basin-all", type: "line", source: "basins", minzoom: 7, paint: {"line-color": css("--accent-ink"), "line-width": 0.5, "line-opacity": 0.35}});
  map.addLayer({id: "basin-fill", type: "fill", source: "basins", filter: selFilter(), paint: {"fill-color": css("--sel"), "fill-opacity": 0.2}});
  map.addLayer({id: "basin-line", type: "line", source: "basins", filter: selFilter(), paint: {"line-color": css("--sel"), "line-width": 3}});
  addOutline(map);
  map.addSource("gauges", {type: "geojson", data: gaugeGeoJSON(() => ({})), promoteId: "id"});
  map.addSource("bars", {type: "geojson", data: {type: "FeatureCollection", features: []}, promoteId: "id", buffer: 256});
  map.addLayer({id: "gauges", type: "circle", source: "gauges", maxzoom: BAR_ZOOM, paint: {
    "circle-radius": dotRadius(dotKey()), "circle-color": levelMatch(dotKey()),
    "circle-stroke-color": "#ffffff", "circle-stroke-width": 1}});
  map.addLayer({id: "gauge-sel", type: "circle", source: "gauges", filter: selFilter(), paint: {
    "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 9, 8.5, 11, 12, 14], "circle-color": "rgba(0,0,0,0)",
    "circle-stroke-color": css("--sel"), "circle-stroke-width": 3}});
  map.addLayer({id: "bars", type: "symbol", source: "bars", minzoom: BAR_ZOOM, layout: {
    "icon-image": ["concat", "bar_", ["to-string", ["get", "lo"]], "_", ["to-string", ["get", "lp"]]],
    "icon-anchor": "bottom", "icon-overlap": "always", "icon-ignore-placement": true,
    "icon-size": ["interpolate", ["linear"], ["zoom"], BAR_ZOOM, 0.85, 12, 1.25]}});
}
const MAIN_LAYERS = ["study-fill", "basin-all", "basin-fill", "basin-line", "study-line", "gauges", "gauge-sel", "bars"];
const barsVisible = () => map && map.getZoom() >= BAR_ZOOM - 0.3;
function updateBars() {
  // Cột chỉ cần khi phóng to; xa thì để nguồn rỗng cho nhẹ
  const lv = state.levelsNow;
  if (!lv || !map?.getSource("bars")) return;
  if (!barsVisible()) { if (updateBars.filled) { map.getSource("bars").setData({type: "FeatureCollection", features: []}); updateBars.filled = false; } return; }
  map.getSource("bars").setData(gaugeGeoJSON(id => ({lo: lv.lo.get(id) ?? -1, lp: lv.lp.get(id) ?? -1})));
  updateBars.filled = true;
}
function initMainMap() {
  map = newMap("map");
  map.on("styleimagemissing", e => {
    const m = /^bar_(-?\d)_(-?\d)$/.exec(e.id);
    if (m && !map.hasImage(e.id)) map.addImage(e.id, barImage(Number(m[1]), Number(m[2])), {pixelRatio: 2});
  });
  map.on("load", () => {
    addMainLayers();
    run(refreshDay);
    updateMiniView();
    const popup = new maplibregl.Popup({closeButton: false, closeOnClick: false, offset: 12});
    const hover = e => {
      map.getCanvas().style.cursor = "pointer";
      const f = e.features[0], id = String(f.properties.id), s = state.site.stations[id];
      const d = state.dayData, i = state.ids.indexOf(id);
      const name = shortName(model(activeRun()));
      popup.setLngLat(f.geometry.coordinates).setDOMContent(el("div", {}, el("b", {}, s.name), `${t("card.river", {r: s.river})} · ID ${id}`,
        el("div", {class: "pair"}, el("span", {}, t("card.observed")), el("span", {}, `${fmtQ(d?.observed[i])} m³/s`),
          el("span", {}, `${t("card.forecast")} (${name})`), el("span", {}, `${fmtQ(d?.predicted[i])} m³/s`)))).addTo(map);
    };
    for (const layer of ["gauges", "bars"]) {
      map.on("mousemove", layer, hover);
      map.on("mouseleave", layer, () => { map.getCanvas().style.cursor = ""; popup.remove(); });
      map.on("click", layer, e => selectBasin(String(e.features[0].properties.id), false));
    }
    map.on("moveend", updateMiniView);
    let wasBars = false;
    map.on("zoomend", () => {
      const now = barsVisible();
      if (now !== wasBars) { wasBars = now; updateBars(); renderLegend(); }
    });
  });
}
function viewBox() {
  const b = map.getBounds();
  return {type: "Feature", properties: {}, geometry: {type: "Polygon", coordinates: [[
    [b.getWest(), b.getSouth()], [b.getEast(), b.getSouth()], [b.getEast(), b.getNorth()], [b.getWest(), b.getNorth()], [b.getWest(), b.getSouth()]]]}};
}
function initMiniMap() {
  // Bản đồ nhỏ chỉ để định vị: vùng nghiên cứu trong Trung Âu, khung cam là phần bản đồ lớn đang xem
  miniMap = new maplibregl.Map({container: "minimap", style: baseStyle(false), center: [12.5, 48.2], zoom: 3.1,
    interactive: false, attributionControl: false});
  miniMap.on("load", () => {
    miniMap.addSource("study", {type: "geojson", data: outlineGeo()});
    miniMap.addLayer({id: "study-fill", type: "fill", source: "study", paint: {"fill-color": css("--accent"), "fill-opacity": 0.45}});
    miniMap.addLayer({id: "study-line", type: "line", source: "study", paint: {"line-color": css("--deep"), "line-width": 1}});
    miniMap.addSource("view", {type: "geojson", data: {type: "FeatureCollection", features: []}});
    miniMap.addLayer({id: "view", type: "line", source: "view", paint: {"line-color": "#c2622d", "line-width": 1.5}});
    updateMiniView();
  });
}
function updateMiniView() { if (map?.loaded()) miniMap?.getSource("view")?.setData(viewBox()); }
function relabelMap(m, keepIds, withLabels = true) {
  // Đổi ngôn ngữ nhãn nền, giữ nguyên các lớp của trang
  m.setStyle(baseStyle(withLabels), {transformStyle: (prev, next) => ({...next,
    sources: {...next.sources, ...Object.fromEntries(Object.entries(prev.sources).filter(([k]) => k !== "protomaps"))},
    layers: [...next.layers, ...prev.layers.filter(l => keepIds.includes(l.id))]})});
}
function barsIcon() {
  // Hình nhỏ minh họa 2 cột trong chú giải (cùng cách vẽ với bản đồ)
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 30");
  for (const [x, hgt, k] of [[4, 18, 2], [13, 26, 3]]) {
    const r = document.createElementNS(ns, "rect");
    Object.entries({x, y: 27 - hgt, width: 7, height: hgt, fill: levelColor(k)}).forEach(([a, v]) => r.setAttribute(a, v));
    svg.append(r);
  }
  const base = document.createElementNS(ns, "rect");
  Object.entries({x: 1, y: 27, width: 22, height: 1.5, fill: css("--ink")}).forEach(([a, v]) => base.setAttribute(a, v));
  svg.append(base);
  return svg;
}
function renderLegend() {
  // Chú giải thu gọn được (nhớ lựa chọn trên máy người xem); màn hẹp mặc định thu gọn để không che bản đồ
  const box = $("map-legend"), bars = barsVisible(), opened = state.legendOpen;
  $("colorby").hidden = bars;
  const head = el("button", {type: "button", class: "legend-toggle", "aria-expanded": String(opened),
    title: t(opened ? "map.legendHide" : "map.legendShow"),
    onclick: () => { state.legendOpen = !state.legendOpen; try { localStorage.setItem("legendOpen", state.legendOpen ? "1" : "0"); } catch { /* bỏ qua */ } renderLegend(); }},
    el("b", {}, t(bars ? "map.legendLevel" : state.colorBy === "observed" ? "map.legendObs" : "map.legendPred")), el("i", {class: "chev", "aria-hidden": "true"}));
  box.classList.toggle("closed", !opened);
  box.replaceChildren(head, ...(opened ? [
    ...[4, 3, 2, 1, 0, -1].map(k => el("span", {class: "swatch"}, el("i", {style: `background:${levelColor(k)}`}), levelText(k))),
    el("span", {class: "swatch"}, el("i", {class: "sel"}), t("map.sel")),
    el("span", {class: "swatch"}, el("i", {class: "ring"}), t("map.outline")),
    el("span", {class: "bars-note"}, barsIcon(), t(bars ? "map.barsOn" : "map.bars"))] : []));
  box.title = opened ? t("lv.note") : "";
}
const dayUrl = (date, runId = activeRun(), lead = activeLead()) => `/api/day${qs({date, run: runId, lead})}`;
// Nạp trước theo khối 60 ngày (một lần gọi /api/days) => khi chạy thời gian bản đồ không phải chờ từng ngày
const CHUNK = 60;
const chunkPending = new Map();
function putCache(path, value) {
  if (cache.has(path)) return;
  if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value);
  cache.set(path, Promise.resolve(value));
}
function prefetchDays(ahead) {
  const runId = activeRun(), lead = activeLead();
  const from = state.day + 1, last = Math.min(state.dates.length - 1, state.day + ahead);
  let k = from;
  while (k <= last && cache.has(dayUrl(state.dates[k], runId, lead))) k += 1;
  if (k > last) return null;
  // Mỗi mô hình chỉ một khối đang nạp; gọi lại khi khối chưa về thì trả đúng lời hứa đó để chờ
  const key = `${runId}|${lead}`;
  if (chunkPending.has(key)) return chunkPending.get(key);
  const job = api(`/api/days${qs({run: runId, lead, start: state.dates[k], n: CHUNK})}`)
    .then(res => res.dates.forEach((date, j) => putCache(dayUrl(date, runId, lead),
      {date, run: runId, lead, ids: res.ids, observed: res.observed[j], predicted: res.predicted[j]})))
    .catch(e => console.error(e))
    .finally(() => chunkPending.delete(key));
  chunkPending.set(key, job);
  return job;
}
async function refreshDay() {
  const runId = activeRun(), lead = activeLead(), date = state.dates[state.day];
  const data = await cached(dayUrl(date, runId, lead));
  if (date !== state.dates[state.day] || runId !== activeRun()) return;
  state.dayData = data;
  const lo = new Map(), lp = new Map();
  data.ids.forEach((id, i) => { lo.set(id, level(data.observed[i], id)); lp.set(id, level(data.predicted[i], id)); });
  state.levelsNow = {lo, lp};
  if (map?.getSource("gauges")) {
    for (const id of data.ids) map.setFeatureState({source: "gauges", id}, {lo: lo.get(id), lp: lp.get(id)});
    updateBars();
  }
  renderStatus();
}

// ---------- bước 1: nơi xem ----------
function renderStation() {
  $("region-btn").setAttribute("aria-pressed", String(isRegion()));
  if (isRegion()) {
    $("station-meta").textContent = "";
    $("station-name").textContent = t("region.name");
    $("station-river").textContent = t("region.desc", {n: state.ids.length});
    return;
  }
  const s = state.site.stations[state.basin];
  $("station-name").textContent = s.name;
  $("station-river").textContent = `${t("card.river", {r: s.river})} · ${t("c." + s.country)}`;
  $("station-meta").textContent = `${t("card.station", {id: state.basin})} · ${t("card.basin", {r: s.region})} · ${t("card.area", {a: fmt(s.area_km2, 0)})} · ${t("card.elev", {e: fmt(s.elev_m, 0)})}`;
}

// ---------- số liệu ngày đang xem ----------
function statBox(label, q, k, unit = "m³/s") {
  const missing = q == null || !Number.isFinite(q);
  return el("div", {},
    el("div", {class: "k", title: label}, label),
    missing ? el("div", {class: "v na"}, t("card.noData")) : el("div", {class: "v"}, fmtQ(q), el("small", {}, ` ${unit}`)),
    k == null || missing ? null : el("div", {class: "level-chip"}, el("i", {style: `background:${levelColor(k)}`}), levelText(k)));
}
function renderStatus() {
  const d = state.dayData;
  $("status-title").textContent = t("step.status", {d: fmtDate(state.dates[state.day])});
  if (!d) return;
  const name = shortName(model(activeRun()));
  if (isRegion()) {
    // Trung bình cả vùng theo mm/ngày (đổi m³/s theo diện tích), chỉ trên trạm có cả thực đo lẫn dự báo
    let so = 0, sp = 0, n = 0;
    const high = {obs: 0, pred: 0};
    d.ids.forEach((id, i) => {
      const o = d.observed[i], p = d.predicted[i];
      if (level(o, id) >= 3) high.obs += 1;
      if (level(p, id) >= 3) high.pred += 1;
      if (o == null || p == null) return;
      const k = 86.4 / state.site.stations[id].area_km2;
      so += o * k; sp += p * k; n += 1;
    });
    $("status").replaceChildren(statBox(t("card.observedMean"), n ? so / n : null, null, t("unit.mmday")),
      statBox(t("card.forecastMean", {m: name}), n ? sp / n : null, null, t("unit.mmday")),
      el("div", {class: "wide"}, el("div", {class: "k"}, t("card.highCount")),
        el("div", {class: "v"}, `${high.obs}`, el("small", {}, ` ${t("card.highObs")}`), "  ", `${high.pred}`, el("small", {}, ` ${t("card.highPred")}`))));
    return;
  }
  const i = state.ids.indexOf(state.basin);
  const o = d.observed[i], p = d.predicted[i];
  $("status").replaceChildren(statBox(t("card.observed"), o, level(o, state.basin)),
    statBox(`${t("card.forecast")} · ${name}`, p, level(p, state.basin)));
}

// ---------- bước 3: bài toán, mô hình chính, so sánh ----------
function groupedOptions(list, labelOf, selected, groupOf) {
  const groups = new Map();
  for (const m of list) {
    const g = groupOf(m);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(m);
  }
  return [...groups].map(([label, items]) => {
    const og = el("optgroup", {label});
    og.append(...items.map(m => new Option(labelOf(m), m.run, false, m.run === selected)));
    return og;
  });
}
const byCore = list => [...list].sort((a, b) => CORES.indexOf(coreOf(a)) - CORES.indexOf(coreOf(b)) || valNse(b) - valNse(a));
const byMode = list => [...list].sort((a, b) => MODES.indexOf(a.mode) - MODES.indexOf(b.mode) || CORES.indexOf(a.core) - CORES.indexOf(b.core));
const withCode = (text, m) => (m.configuration ? `${text} (${m.configuration})` : text);
function mainOptions() {
  const task = state.task, list = stageModels(STAGE[task]);
  if (task === "h") return groupedOptions(byMode(list), m => `${CORE[m.core]} (${m.configuration})`, activeRun(), m => t("mode." + m.mode));
  return groupedOptions(byCore(list), m => withCode(`${CORE[coreOf(m)]} ${variant(m)}`, m), activeRun(), m => CORE[coreOf(m)]);
}
function compareCandidates() {
  // Ứng viên so sánh: mô hình cùng hạn (khác mô hình chính, chưa chọn) + mốc
  const lead = activeLead(), taken = new Set([activeRun(), ...compareRuns()]);
  const ok = m => !taken.has(m.run) && m.leads.includes(lead);
  const bases = models().filter(m => m.stage === "Mốc" && ok(m));
  if (state.task === "h") return [...byMode(stageModels("H")).filter(ok), ...bases];
  return [...byCore(stageModels("C2")).filter(ok), ...byCore(stageModels("D")).filter(ok), ...bases];
}
function compareGroup(m) {
  if (m.stage === "Mốc") return t("grp.base");
  if (m.stage === "H") return t("mode." + m.mode);
  return `${t("task." + taskOf(m))} · ${CORE[coreOf(m)]}`;
}
function compareLabel(m) {
  if (m.stage === "Mốc") return shortName(m);
  if (m.stage === "H") return `${CORE[m.core]} (${m.configuration})`;
  return withCode(`${CORE[coreOf(m)]} ${variant(m)}`, m);
}
function lineColor(runId, index) {
  if (runId === "SeqLSTM_Q") return css("--c-biascast");
  if (runId === "Persistence_qmax") return css("--c-persist");
  return LINE_COLORS[index % LINE_COLORS.length];
}
function renderModelControls() {
  $("task-tabs").replaceChildren(...TASKS.map(task => el("button", {type: "button", "aria-pressed": String(state.task === task),
    onclick: () => { if (state.task === task) return; state.task = task; syncCompareSection(); renderModelControls(); refreshAll(); }},
    t("task." + task), el("small", {}, t("task." + task + ".sub")))));
  $("task-note").textContent = t(`task.${state.task}.note`);
  $("model").replaceChildren(...mainOptions());
  $("lead-field").hidden = state.task !== "h";
  $("lead-value").textContent = days(state.lead);
  $("lead").value = state.lead;
  const cands = compareCandidates(), full = compareRuns().length >= MAX_COMPARE;
  const placeholder = new Option(full ? t("card.addFull") : t("card.addHint"), "", true, true);
  placeholder.disabled = true;
  $("add-model").replaceChildren(placeholder, ...groupedOptions(cands, compareLabel, null, compareGroup));
  $("add-model").disabled = full || !cands.length;
  const lines = [activeRun(), ...compareRuns()];
  $("main-swatch").style.background = lineColor(activeRun(), 0);
  const xIcon = () => {
    const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg"), p = document.createElementNS(ns, "path");
    svg.setAttribute("viewBox", "0 0 14 14"); svg.setAttribute("aria-hidden", "true");
    p.setAttribute("d", "M3 3l8 8M11 3l-8 8"); svg.append(p);
    return svg;
  };
  $("chips").replaceChildren(...compareRuns().map(r => el("li", {},
    el("i", {style: `background:${lineColor(r, lines.indexOf(r))}`}), el("span", {}, shortName(model(r))),
    el("button", {type: "button", title: t("card.remove", {m: shortName(model(r))}), "aria-label": t("card.remove", {m: shortName(model(r))}), onclick: () => {
      state.compare[state.task] = state.compare[state.task].filter(x => x !== r);
      renderModelControls(); run(refreshChart);
    }}, xIcon()))));
  const ensOk = lines.filter(r => !BASE.includes(r)).length >= 2;
  $("ensemble").disabled = !ensOk;
  $("ensemble").checked = state.ensemble && ensOk;
  $("ensemble-row").title = ensOk ? "" : t("card.ensembleNeed");
  $("ensemble-row").classList.toggle("off", !ensOk);
}

// ---------- biểu đồ lưu lượng ----------
let chartRange = null;
function windowAround(idx, half = 60) {
  const a = Math.max(0, idx - half), b = Math.min(state.dates.length - 1, idx + half);
  return [state.dates[a], state.dates[b]];
}
async function refreshChart({keepRange = false} = {}) {
  const main = activeRun(), lead = activeLead(), basin = state.basin, extra = compareRuns();
  const ensemble = $("ensemble").checked;
  const data = await cached(`/api/chart${qs({basin, lead, run: [main, ...extra], ...(ensemble ? {ensemble: 1} : {})})}`);
  const fan = state.task === "h" && !isRegion() ? await cached(`/api/fan${qs({basin, run: main, date: state.dates[state.day]})}`).catch(() => null) : null;
  if (basin !== state.basin || main !== activeRun()) return;
  state.chartData = data;
  const region = data.unit === "mm/day", unit = region ? t("unit.mmday") : "m³/s";
  const traces = [{x: data.dates, y: data.observed, name: t("chart.obs"), line: {color: css("--c-obs"), width: 2.4}}];
  const keys = [legendKey(css("--c-obs"), t("chart.obs"))];
  data.predictions.forEach((p, i) => {
    const color = lineColor(p.run, i), dotted = p.run === "Persistence_qmax";
    const label = shortName(model(p.run)) + (i === 0 ? ` (${t("chart.main")})` : "");
    traces.push({x: data.dates, y: p.values, name: label, line: {color, width: i === 0 ? 2.4 : 1.8, dash: dotted ? "dot" : "solid"}});
    keys.push(legendKey(color, label, {style: dotted ? "dot" : "", value: `NSE ${fmtFixed(p.nse, 3)}`}));
  });
  if (data.ensemble) {
    traces.push({x: data.dates, y: data.ensemble.values, name: t("chart.ensemble"), line: {color: ENS_COLOR, width: 2.6, dash: "dash"}});
    keys.push(legendKey(ENS_COLOR, t("chart.ensembleN", {n: data.ensemble.runs.length}), {style: "dash", value: `NSE ${fmtFixed(data.ensemble.nse, 3)}`}));
  }
  if (fan) {
    traces.push({x: fan.steps.map(s => s.target_date), y: fan.steps.map(s => s.qmax_m3_s), name: t("chart.fan", {d: fmtDate(fan.origin)}),
      mode: "lines+markers", line: {color: css("--lv-vhigh"), width: 2.2}, marker: {size: 7}});
    keys.push(legendKey(css("--lv-vhigh"), t("chart.fanKey")));
  }
  for (const tr of traces) {
    tr.type = "scatter"; tr.mode ??= "lines"; tr.connectgaps = false;
    tr.hovertemplate = `%{y:.2f} ${unit}<extra>%{fullData.name}</extra>`;
  }
  $("chart-legend").replaceChildren(...keys);
  const years = testYears();
  $("chart-title").textContent = region ? t("chart.titleRegion", {y: years})
    : t("chart.title", {s: state.site.stations[basin].name, y: years});
  $("chart-sub").textContent = `${t("task." + state.task + ".long")}${state.task === "h" ? ` · ${t("card.lead")} ${days(lead)}` : ""} · `
    + t("chart.nseDays", {n: data.nse_days}) + (region ? ` ${t("chart.regionNote")}` : "");
  const range = keepRange && chartRange ? chartRange : windowAround(state.day);
  chartRange = range;
  const box = $("hydrograph");
  const layout = baseLayout({
    margin: {l: 66, r: 16, t: 8, b: 8}, hovermode: "x unified", height: Math.max(260, box.clientHeight),
    xaxis: {type: "date", range, rangeslider: {visible: true, thickness: 0.08, bgcolor: css("--paper")}, gridcolor: css("--grid")},
    yaxis: {title: {text: region ? t("chart.yRegion") : t("chart.y")}, rangemode: "tozero", gridcolor: css("--grid")},
  });
  await Plotly.react(box, traces, layout, {responsive: true, displaylogo: false, locale: getLang(),
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines", "hoverCompareCartesian", "hoverClosestCartesian"],
    toImageButtonOptions: {filename: `discharge_${basin}`}});
  if (!box.dataset.bound) {
    box.dataset.bound = "1";
    box.on("plotly_afterplot", placeDayLine);
    box.on("plotly_relayout", ev => {
      if (ev["xaxis.range[0]"] || ev["xaxis.range"] || ev["xaxis.autorange"]) {
        const r = box.layout.xaxis.range;
        chartRange = [String(r[0]).slice(0, 10), String(r[1]).slice(0, 10)];
        fitY();
      }
    });
  }
  fitY();
}
function fitY() {
  // Trục Q co theo số liệu trong khung đang xem; ngưỡng lũ chỉ vẽ khi gần khung, còn lại ghi chữ
  const box = $("hydrograph");
  if (!box.data || !chartRange) return;
  const [a, b] = chartRange;
  let top = 0;
  for (const tr of box.data) {
    tr.x.forEach((x, i) => { const y = tr.y[i]; if (x >= a && x <= b && y != null && y > top) top = y; });
  }
  top = top > 0 ? top * 1.12 : 1;
  const floods = isRegion() ? [] : Object.entries(state.site.flood_m3_s[state.basin] ?? {}).map(([p, v]) => [Number(p), v]).sort((x, y) => x[0] - y[0]);
  const shown = floods.filter(([, v]) => v <= top * 1.25);
  if (shown.length) top = Math.max(top, shown.at(-1)[1] * 1.08);
  const hidden = floods.filter(([, v]) => v > top);
  const shapes = [];
  const annotations = [];
  // Nhãn ngưỡng xen kẽ phải, trái để hai ngưỡng gần nhau không đè chữ
  shown.forEach(([p, v], i) => {
    const right = i % 2 === (shown.length - 1) % 2;
    shapes.push({type: "line", xref: "paper", x0: 0, x1: 1, y0: v, y1: v, line: {color: css("--lv-flood"), width: 1, dash: "dot"}});
    annotations.push({xref: "paper", x: right ? 1 : 0, y: v, xanchor: right ? "right" : "left", yanchor: "bottom", showarrow: false,
      bgcolor: "rgba(255,255,255,0.8)", borderpad: 1,
      text: `${t("chart.flood", {p: fmt(p, 1)})} · ${fmtQ(v)} m³/s`, font: {size: 12.5, color: css("--lv-flood")}});
  });
  Plotly.relayout(box, {"yaxis.range": [0, top], "yaxis.autorange": false, shapes, annotations});
  $("chart-note").textContent = (hidden.length
    ? t("chart.thresholdsHidden", {list: hidden.map(([p, v]) => `${t("flood.years", {p: fmt(p, 1)})} ${fmtQ(v)} m³/s`).join(", ")}) + " "
    : "") + t("chart.help");
}

// ---------- thanh thời gian ----------
function renderDayLabel() {
  $("day-label").textContent = fmtDate(state.dates[state.day], true);
  $("day-slider").value = state.day;
  markHighlights();
}
let dayFrame = null;
function placeDayLine() {
  // Vạch ngày đang xem là một thẻ HTML đặt chồng lên vùng vẽ: dời vạch không bắt Plotly vẽ lại (khoảng 150 ms mỗi lần)
  const box = $("hydrograph"), fl = box._fullLayout, line = $("dayline");
  if (!fl?.xaxis?.range || !line) return;
  const [r0, r1] = fl.xaxis.range.map(v => Date.parse(String(v).slice(0, 10) + "T00:00:00Z"));
  const tDay = Date.parse(state.dates[state.day] + "T00:00:00Z");
  const inside = tDay >= r0 && tDay <= r1;
  line.hidden = !inside;
  if (!inside) return;
  const x = fl.xaxis._offset + (tDay - r0) / (r1 - r0) * fl.xaxis._length;
  line.style.cssText = `left:${x}px;top:${fl.yaxis._offset}px;height:${fl.yaxis._length}px`;
}
function moveDayLine() {
  // Ngày ra ngoài cửa sổ đang xem => dời cửa sổ (trục Q co lại qua sự kiện plotly_relayout); còn lại chỉ dời vạch
  const box = $("hydrograph");
  if (!box._fullLayout) return;
  const day = state.dates[state.day];
  if (chartRange && (day < chartRange[0] || day > chartRange[1])) {
    chartRange = windowAround(state.day, playTimer ? 90 : 60);
    Plotly.relayout(box, {"xaxis.range": chartRange});
  } else placeDayLine();
}
function setDay(idx, {fromSlider = false} = {}) {
  state.day = Math.max(0, Math.min(state.dates.length - 1, idx));
  renderDayLabel();
  if (!fromSlider) stopPlay();
  cancelAnimationFrame(dayFrame);
  dayFrame = requestAnimationFrame(() => {
    run(refreshDay);
    if (state.task === "h" && !isRegion()) run(() => refreshChart({keepRange: true}));
    else moveDayLine();
    resetWhatIf();
  });
}
let playTimer = null;
function stopPlay() {
  if (!playTimer) return;
  clearTimeout(playTimer); playTimer = null;
  $("play").classList.remove("on");
  // Bản dự báo 7 ngày (hình quạt) chỉ vẽ lại khi dừng, không vẽ ở mỗi bước chạy
  if (state.task === "h" && !isRegion()) run(() => refreshChart({keepRange: true}));
}
async function playStep() {
  if (state.day >= state.dates.length - 1) return stopPlay();
  const started = performance.now();
  prefetchDays(CHUNK / 2);
  // Ngày sau chưa nạp xong thì đợi (chạy chậm lại) thay vì nhảy cóc làm bản đồ giật
  const next = dayUrl(state.dates[state.day + 1]);
  if (!cache.has(next)) await prefetchDays(CHUNK);
  if (!playTimer) return;
  state.day += 1;
  renderDayLabel();
  await refreshDay().catch(e => console.error(e));
  moveDayLine();
  if (!playTimer) return;
  playTimer = setTimeout(playStep, Math.max(0, STEP_MS / state.speed - (performance.now() - started)));
}
function togglePlay() {
  if (playTimer) return stopPlay();
  $("play").classList.add("on");
  resetWhatIf();
  playTimer = setTimeout(playStep, 0);
}
function setSpeed(k) {
  state.speed = k;
  document.querySelectorAll("[data-speed]").forEach(b => b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === k)));
}

// ---------- chọn trạm hoặc cả vùng ----------
function applySelection() {
  for (const layer of ["basin-fill", "basin-line", "gauge-sel"]) map?.getLayer(layer) && map.setFilter(layer, selFilter());
  map?.getLayer("study-fill") && map.setPaintProperty("study-fill", "fill-opacity", isRegion() ? 0.1 : 0.04);
  renderStation();
  renderStatus();
  markHighlights();
  run(() => refreshChart());
  if (state.resultsReady) { run(refreshFloodBasin); run(refreshXai); }
  resetWhatIf(true);
}
function selectBasin(id, fly = true) {
  if (!state.site.stations[id]) return;
  state.basin = id;
  state.lastBasin = id;
  if (fly && map) {
    const p = state.catalog.points.find(x => x.id === id);
    map.easeTo({center: [p.lon, p.lat], zoom: Math.max(map.getZoom(), 9.5), duration: 700});
  }
  applySelection();
  try { history.replaceState(null, "", `#basin=${id}`); } catch { /* bỏ qua */ }
}
function toggleRegion() {
  if (isRegion()) return selectBasin(state.lastBasin);
  state.basin = "all";
  map?.fitBounds(studyBounds(), {padding: 30, duration: 700});
  applySelection();
  try { history.replaceState(null, "", "#basin=all"); } catch { /* bỏ qua */ }
}
// Ô tìm trạm tự viết thay cho datalist: datalist của trình duyệt bấm vào chưa bung, 451 dòng bung chậm, không lọc bỏ dấu
const fold = text => text.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/đ/g, "d").toLowerCase();
const COMBO_MAX = 60;
function bindSearch() {
  const input = $("station-search"), list = $("station-list");
  const index = state.ids.map(id => {
    const s = state.site.stations[id];
    return {id, name: s.name, sub: `${s.river} · ${id}`, key: fold(`${s.name} ${s.river} ${s.region ?? ""} ${id}`)};
  }).sort((x, y) => x.name.localeCompare(y.name));
  let shown = [], active = -1;
  const close = () => { list.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); active = -1; };
  const mark = k => {
    active = k;
    [...list.children].forEach((li, j) => li.setAttribute("aria-selected", String(j === k)));
    const li = list.children[k];
    if (li?.id) { input.setAttribute("aria-activedescendant", li.id); li.scrollIntoView({block: "nearest"}); }
  };
  const pick = item => { close(); input.value = ""; input.blur(); selectBasin(item.id); };
  const open = () => {
    const words = fold(input.value.trim()).split(/\s+/).filter(Boolean);
    const hits = words.length ? index.filter(it => words.every(w => it.key.includes(w))) : index;
    shown = hits.slice(0, COMBO_MAX);
    list.replaceChildren(...shown.map((it, j) => el("li", {id: `stn-${it.id}`, role: "option", "aria-selected": "false",
      onmousedown: e => { e.preventDefault(); pick(it); }}, el("b", {}, it.name), el("span", {}, it.sub))));
    if (!hits.length) list.append(el("li", {class: "note"}, t("card.searchNone")));
    else if (hits.length > COMBO_MAX) list.append(el("li", {class: "note"}, t("card.searchMore", {n: hits.length - COMBO_MAX})));
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
    mark(words.length && shown.length ? 0 : -1);
  };
  input.addEventListener("focus", open);
  input.addEventListener("input", open);
  input.addEventListener("blur", () => { close(); input.value = ""; });
  input.addEventListener("keydown", e => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (list.hidden) return open();
      if (shown.length) mark((active + (e.key === "ArrowDown" ? 1 : shown.length - 1 + (active < 0 ? 1 : 0))) % shown.length);
    } else if (e.key === "Enter" && active >= 0 && shown[active]) { e.preventDefault(); pick(shown[active]); }
    else if (e.key === "Escape") { close(); input.value = ""; input.blur(); }
  });
}

// ---------- đợt lũ lớn trong kỳ kiểm tra (nút nhảy nhanh dưới lời giới thiệu) ----------
async function initHighlights() {
  try { state.highlights = (await api("/api/highlights")).events.slice(0, 6); }
  catch (e) { console.error(e); state.highlights = []; }
  renderHighlights();
}
function renderHighlights() {
  const list = state.highlights ?? [];
  $("highlights").hidden = !list.length;
  $("hl-list").replaceChildren(...list.map(ev => el("li", {}, el("button", {type: "button", class: "hl", "data-basin": ev.basin, "data-date": ev.date,
    title: t("hl.title", {s: ev.name, r: ev.river, q: fmtQ(ev.qmax_m3_s), x: fmt(ev.ratio, 1), t: fmtQ(ev.t10_m3_s)}),
    onclick: () => goToEvent(ev)},
    el("b", {}, `${fmtDate(ev.date)} · ${ev.river}`), el("span", {}, t("hl.count", {n: ev.gauges}))))));
  markHighlights();
}
function markHighlights() {
  const day = state.dates[state.day];
  document.querySelectorAll("#hl-list .hl").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.basin === state.basin && b.dataset.date === day)));
}
function goToEvent(ev, {scroll = true} = {}) {
  // Đặt ngày trước rồi mới chọn trạm => biểu đồ dựng một lần, cửa sổ thời gian quanh ngày lũ
  const idx = state.dates.indexOf(ev.date);
  if (idx < 0) return;
  setDay(idx);
  showTab("flow");
  selectBasin(ev.basin);
  if (scroll) scrollToDash();
}

// ---------- tải số liệu của Hình 1.1 ----------
function downloadChartCsv() {
  const d = state.chartData;
  if (!d) return;
  const unit = d.unit === "mm/day" ? "mm_day" : "m3_s";
  const cols = [["observed", d.observed], ...d.predictions.map(p => [p.run, p.values]), ...(d.ensemble ? [["ensemble_mean", d.ensemble.values]] : [])];
  const rows = [["date", ...cols.map(([k]) => `${k}_${unit}`)].join(",")];
  d.dates.forEach((day, i) => rows.push([day, ...cols.map(([, v]) => v[i] ?? "")].join(",")));
  const name = `qmax_${d.basin === "all" ? "region" : `basin${d.basin}`}_${state.task}_lead${d.lead}.csv`;
  const url = URL.createObjectURL(new Blob(["﻿" + rows.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"}));
  const a = el("a", {href: url, download: name});
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  toast(t("csv.done", {f: name}));
}

// ---------- hướng dẫn xem nhanh: làm nổi từng vùng và giải thích ----------
const TOUR = [
  {sel: "#highlights", key: "hl", before: () => state.highlights?.length && goToEvent(state.highlights[0], {scroll: false})},
  {sel: ".map-pane", key: "map"},
  {sel: "#panel", key: "panel"},
  {sel: "#task-tabs", key: "task"},
  {sel: ".timeline", key: "time"},
  {sel: "#pane-flow", key: "chart", before: () => showTab("flow")},
  {sel: "#tab-whatif", key: "whatif"},
  {sel: "#how .arch-fig", key: "how"},
  {sel: ".results-head", key: "results"},
];
const tour = {steps: [], k: -1, target: null};
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const topbarH = () => document.querySelector(".topbar").offsetHeight;
const isShown = node => node && !node.closest("[hidden]") && node.getClientRects().length > 0;
function startTour() {
  stopPlay();
  tour.steps = TOUR.filter(st => st.key === "hl" ? (state.highlights?.length ?? 0) > 0 : true);
  $("tour-spot").hidden = false;
  $("tour-card").hidden = false;
  document.addEventListener("keydown", tourKeys, true);
  window.addEventListener("scroll", placeTour, {passive: true});
  window.addEventListener("resize", placeTour);
  tourGo(0, 1);
}
function endTour() {
  tour.k = -1;
  tour.target = null;
  $("tour-spot").hidden = true;
  $("tour-card").hidden = true;
  document.removeEventListener("keydown", tourKeys, true);
  window.removeEventListener("scroll", placeTour);
  window.removeEventListener("resize", placeTour);
  $("tour-btn").focus({preventScroll: true});
}
function tourGo(k, dir = 1) {
  if (k >= tour.steps.length) return endTour();
  if (k < 0) return;
  const step = tour.steps[k];
  step.before?.();
  const target = document.querySelector(step.sel);
  // Vùng đang ẩn (VD thẻ thử kịch bản khi mô hình không chạy được trực tiếp) => bỏ qua theo hướng đang đi
  if (!isShown(target)) return tourGo(k + dir, dir);
  tour.k = k;
  tour.target = target;
  renderTour();
  const r = target.getBoundingClientRect(), top0 = topbarH();
  const room = innerHeight - top0;
  const y = r.height + 220 < room ? scrollY + r.top - top0 - (room - r.height) / 2 + 60 : scrollY + r.top - top0 - 12;
  window.scrollTo({top: Math.max(0, y), behavior: reduceMotion() ? "auto" : "smooth"});
  placeTour();
  [250, 600, 1000].forEach(ms => setTimeout(placeTour, ms));
  $("tour-next").focus({preventScroll: true});
}
function renderTour() {
  if (tour.k < 0) return;
  const step = tour.steps[tour.k];
  $("tour-step").textContent = t("tour.step", {i: tour.k + 1, n: tour.steps.length});
  $("tour-title").textContent = t(`tour.${step.key}.title`);
  $("tour-text").textContent = t(`tour.${step.key}.text`);
  $("tour-prev").disabled = tour.k === 0;
  $("tour-next").textContent = t(tour.k === tour.steps.length - 1 ? "tour.done" : "tour.next");
}
function placeTour() {
  if (tour.k < 0 || !tour.target) return;
  const pad = 6, m = 12, top0 = topbarH();
  const r = tour.target.getBoundingClientRect(), spot = $("tour-spot"), card = $("tour-card");
  const top = Math.max(r.top - pad, top0), bottom = Math.min(r.bottom + pad, innerHeight);
  Object.assign(spot.style, {left: `${r.left - pad}px`, top: `${top}px`, width: `${r.width + 2 * pad}px`, height: `${Math.max(0, bottom - top)}px`});
  const cw = card.offsetWidth, ch = card.offsetHeight;
  let x = r.left, y;
  if (innerHeight - bottom >= ch + 2 * m) y = bottom + m;
  else if (top - top0 >= ch + 2 * m) y = top - ch - m;
  else if (innerWidth - r.right >= cw + 2 * m) { x = r.right + m; y = top + m; }
  else if (r.left >= cw + 2 * m) { x = r.left - cw - m; y = top + m; }
  else { x = innerWidth - cw - 2 * m; y = innerHeight - ch - 2 * m; }
  card.style.left = `${Math.max(m, Math.min(x, innerWidth - cw - m))}px`;
  card.style.top = `${Math.max(top0 + m, Math.min(y, innerHeight - ch - m))}px`;
}
function tourKeys(e) {
  if (e.key === "Escape") { e.preventDefault(); e.stopImmediatePropagation(); endTour(); }
  else if (e.key === "ArrowRight" && !e.target.closest("input, select, textarea")) { e.preventDefault(); e.stopImmediatePropagation(); tourGo(tour.k + 1, 1); }
  else if (e.key === "ArrowLeft" && !e.target.closest("input, select, textarea")) { e.preventDefault(); e.stopImmediatePropagation(); tourGo(tour.k - 1, -1); }
}

// ---------- 2.1 xếp hạng mô hình ----------
const cmpLeadNow = () => (state.cmpTask === "h" ? state.cmpLead : 1);
function rankRows() {
  const lead = cmpLeadNow(), stage = STAGE[state.cmpTask];
  return state.leaderboard.filter(r => r.lead === lead && (model(r.run)?.stage === stage || (model(r.run)?.stage === "Mốc" && hasLead(r.run, lead))));
}
function syncCompareSection() {
  // Đổi mô hình chính ở phía trên => các mục 2.1, 2.4, 2.5 chuyển theo; sau đó mỗi mục vẫn đổi riêng được
  state.cmpTask = state.task;
  state.cmpLead = activeLead();
  state.cmpRun = activeRun();
  state.sec.flood = {run: activeRun(), lead: activeLead()};
  state.sec.levels = {run: activeRun(), lead: activeLead()};
  renderCompareControls();
  renderSectionPickers();
  if (state.resultsReady) run(refreshResults);
}
function renderCompareControls() {
  $("cmp-task").replaceChildren(...TASKS.map(task => el("button", {type: "button", "aria-pressed": String(state.cmpTask === task),
    onclick: () => { state.cmpTask = task; state.cmpRun = null; renderCompareControls(); run(refreshCompare); }}, taskLabel(task))));
  $("cmp-lead-field").hidden = state.cmpTask !== "h";
  $("cmp-lead").replaceChildren(...[1, 2, 3, 4, 5, 6, 7].map(k => new Option(days(k), k, false, k === state.cmpLead)));
}
// Ô chọn riêng của mục 2.4, 2.5: mọi mô hình (ngày mai, 7 ngày, nhìn lại 2 năm, mốc), hạn hiện khi mô hình có nhiều hạn
function sectionGroup(m) {
  if (m.stage === "Mốc") return t("grp.base");
  if (m.stage === "H") return `${t("task.h")} · ${t("mode." + m.mode)}`;
  return `${t("task." + taskOf(m))} · ${CORE[coreOf(m)]}`;
}
function sectionList() {
  return [...byCore(stageModels("C2")), ...byMode(stageModels("H")), ...byCore(stageModels("D")), ...models().filter(m => m.stage === "Mốc")];
}
function renderSectionPickers() {
  for (const [key, box, refresh] of [["flood", "flood-pick", refreshFloodSection], ["levels", "lvl-pick", refreshLevels]]) {
    const sel = state.sec[key], m = model(sel.run);
    if (!m) continue;
    if (!m.leads.includes(sel.lead)) sel.lead = m.leads[0];
    const pick = el("select", {"aria-label": t("pick.model"), onchange: e => { sel.run = e.target.value; renderSectionPickers(); run(refresh); }},
      ...groupedOptions(sectionList(), compareLabel, sel.run, sectionGroup));
    const lead = m.leads.length > 1 ? el("label", {class: "field inline"}, el("span", {}, t("card.lead")),
      el("select", {onchange: e => { sel.lead = Number(e.target.value); run(refresh); }}, ...m.leads.map(k => new Option(days(k), k, false, k === sel.lead)))) : null;
    $(box).replaceChildren(...[el("label", {class: "field inline"}, el("span", {}, t("pick.model")), pick), lead].filter(Boolean));
  }
}
const secName = sec => shortName(model(sec.run)) + (model(sec.run)?.leads.length > 1 ? ` · ${t("card.lead").toLowerCase()} ${days(sec.lead)}` : "");
async function refreshFloodSection() { await Promise.all([refreshFloodAll(), refreshFloodBasin()]); }
async function refreshCompare() {
  if (!state.leaderboard.length) state.leaderboard = (await cached("/api/leaderboard")).rows;
  if (!state.seeds) state.seeds = new Map((await cached("/api/c2-seeds")).rows.map(r => [r.configuration, r]));
  const rows = rankRows().sort((a, b) => b.NSE - a.NSE);
  const lead = cmpLeadNow();
  if (!state.cmpRun || !rows.some(r => r.run === state.cmpRun)) state.cmpRun = rows.find(r => model(r.run).stage !== "Mốc")?.run ?? rows[0]?.run;
  const showSeeds = state.cmpTask === "c2";
  const head = el("tr", {}, el("th", {}, t("cmp.col.model")), el("th", {class: "n"}, t("cmp.col.nse")), el("th", {class: "bar-col"}, ""),
    showSeeds ? el("th", {class: "n", title: t("cmp.col.seedsHint")}, t("cmp.col.seeds")) : null,
    el("th", {class: "n"}, t("cmp.col.good")), el("th", {class: "n"}, t("cmp.col.kge")));
  const body = rows.map(r => {
    const m = model(r.run), base = m.stage === "Mốc", seeds = state.seeds.get(m.configuration);
    const code = base ? (m.run === "SeqLSTM_Q" ? "SeqLSTM_Q · Konold et al., HESS 2026" : "Persistence_qmax") : `${m.configuration} · seed ${m.seed}`;
    const tr = el("tr", {class: r.run === state.cmpRun ? "on" : "", tabindex: "0", onclick: () => { state.cmpRun = r.run; run(refreshCompare); }},
      el("td", {}, shortName(m), el("span", {class: "code"}, code)),
      el("td", {class: "n"}, fmtFixed(r.NSE, 3)),
      el("td", {class: "bar-cell"}, el("div", {class: "bar" + (base ? " base" : "")}, el("i", {style: `width:${Math.max(0, Math.min(1, r.NSE)) * 100}%`}))),
      showSeeds ? el("td", {class: "n sd"}, seeds && !base ? `${fmtFixed(seeds.mean, 3)} ± ${fmtFixed(seeds.sd, 3)}` : "–") : null,
      el("td", {class: "n"}, pct(r.share_good)),
      el("td", {class: "n"}, fmtFixed(r.KGE, 3)));
    tr.addEventListener("keydown", e => { if (e.key === "Enter") tr.click(); });
    return tr;
  });
  $("rank").replaceChildren(el("thead", {}, head), el("tbody", {}, ...body));
  $("rank-title").textContent = t("cmp.rankTitle", {task: t("task." + state.cmpTask + ".long") + (state.cmpTask === "h" ? ` · ${t("card.lead").toLowerCase()} ${days(lead)}` : "")});
  $("cmp-foot").textContent = t("cmp.foot") + (showSeeds ? " " + t("cmp.footSeeds") : "");
  const refs = [["", t("cmp.absolute")], ...(lead === 1 ? [["SeqLSTM_Q", shortName(model("SeqLSTM_Q"))]] : []), ["Persistence_qmax", shortName(model("Persistence_qmax"))]]
    .filter(([v]) => v !== state.cmpRun);
  if (!refs.some(([v]) => v === state.nseRef)) state.nseRef = "";
  $("nse-ref").replaceChildren(...refs.map(([v, label]) => new Option(label, v, false, v === state.nseRef)));
  await refreshNseMap(lead);
}
const NSE_STEPS = [[-Infinity, "#9b1c1f"], [0, "#e08a3a"], [0.5, "#b7d3c8"], [0.7, "#4f9b8f"], [0.85, "#145f6e"]];
async function refreshNseMap(lead) {
  const runId = state.cmpRun, ref = state.nseRef;
  const data = await cached(`/api/basin-scores${qs({run: runId, lead, ...(ref ? {reference: ref} : {})})}`);
  if (runId !== state.cmpRun) return;
  $("nse-title").textContent = ref ? t("cmp.mapDelta", {m: shortName(model(runId)), r: shortName(model(ref))}) : t("cmp.mapTitle", {m: shortName(model(runId))});
  const geo = gaugeGeoJSON(id => ({v: data.scores[id] ?? null}));
  geo.features = geo.features.filter(f => f.properties.v != null);
  const color = ref
    ? ["interpolate", ["linear"], ["get", "v"], -0.2, "#9b1c1f", -0.05, "#e9b59b", 0, "#e4e4df", 0.05, "#9fc6d6", 0.2, "#0f5f7c"]
    : ["step", ["get", "v"], NSE_STEPS[0][1], ...NSE_STEPS.slice(1).flat()];
  const apply = () => {
    if (nseMap.getSource("nse")) nseMap.getSource("nse").setData(geo);
    else {
      addOutline(nseMap, 1.6);
      nseMap.addSource("nse", {type: "geojson", data: geo});
      nseMap.addLayer({id: "nse", type: "circle", source: "nse", paint: {"circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 4.5, 10, 9], "circle-stroke-color": "#fff", "circle-stroke-width": 0.8}});
      const popup = new maplibregl.Popup({closeButton: false, closeOnClick: false, offset: 8});
      nseMap.on("mousemove", "nse", e => {
        const f = e.features[0], s = state.site.stations[f.properties.id];
        nseMap.getCanvas().style.cursor = "pointer";
        popup.setLngLat(f.geometry.coordinates).setDOMContent(el("div", {}, el("b", {}, s.name), `${state.nseRef ? "ΔNSE" : "NSE"} ${fmt(f.properties.v, 3)}`)).addTo(nseMap);
      });
      nseMap.on("mouseleave", "nse", () => { nseMap.getCanvas().style.cursor = ""; popup.remove(); });
      nseMap.on("click", "nse", e => { selectBasin(String(e.features[0].properties.id)); scrollToDash(); });
    }
    nseMap.setPaintProperty("nse", "circle-color", color);
  };
  nseMap.loaded() ? apply() : nseMap.once("load", apply);
  const legend = ref
    ? [["#9b1c1f", `≤ −0,2 (${t("cmp.worse")})`], ["#e4e4df", "0"], ["#0f5f7c", `≥ +0,2 (${t("cmp.better")})`]]
    : [["#9b1c1f", "< 0"], ["#e08a3a", "0 – 0,5"], ["#b7d3c8", "0,5 – 0,7"], ["#4f9b8f", "0,7 – 0,85"], ["#145f6e", "≥ 0,85"]];
  $("nse-legend").replaceChildren(...legend.map(([c, label]) => el("span", {class: "swatch"}, el("i", {style: `background:${c}`}), getLang() === "vi" ? label : label.replaceAll(",", "."))));
}

// ---------- 2.2 dự báo theo hạn ----------
async function refreshLeads() {
  const rows = (await cached("/api/by-lead")).rows;
  const keyOf = r => `${r.mode}|${r.core ?? ""}`;
  const groups = new Map();
  for (const r of rows) { if (!groups.has(keyOf(r))) groups.set(keyOf(r), []); groups.get(keyOf(r)).push(r); }
  const traces = [...groups.values()].map(list => {
    list.sort((a, b) => a.lead - b.lead);
    const {mode, core} = list[0];
    const name = mode === "persistence" ? t("m.persistence") : `${CORE[core]} · ${t("mode.short." + mode)}`;
    return {type: "scatter", mode: "lines+markers", x: list.map(r => r.lead), y: list.map(r => r.mean), name,
      error_y: {type: "data", array: list.map(r => r.sd ?? 0), visible: mode !== "persistence", thickness: 1.2, width: 3, color: MODE_COLOR[mode]},
      line: {color: MODE_COLOR[mode], width: 2.4, dash: mode === "persistence" ? "dot" : core === "mamba" ? "dash" : "solid"},
      marker: {size: 7, symbol: core === "mamba" ? "diamond" : "circle"},
      hovertemplate: `%{y:.3f}<extra>${name}</extra>`};
  });
  const low = Math.min(-0.2, ...rows.map(r => r.mean)) - 0.05;
  await Plotly.react($("leads-chart"), traces, baseLayout({hovermode: "x unified",
    xaxis: {title: {text: t("leads.x")}, dtick: 1, gridcolor: css("--grid")},
    yaxis: {title: {text: t("leads.y")}, gridcolor: css("--grid"), zeroline: true, zerolinecolor: css("--muted"), range: [low, 0.8]},
    shapes: [{type: "rect", xref: "paper", x0: 0, x1: 1, y0: low, y1: 0, fillcolor: "rgba(161,29,31,0.06)", line: {width: 0}, layer: "below"}]}), plotConfig);
  $("leads-legend").replaceChildren(...MODES.map(m => legendKey(MODE_COLOR[m], t("mode." + m), {style: "box"})),
    legendKey(MODE_COLOR.persistence, t("m.persistence"), {style: "dot"}),
    legendKey(css("--ink-2"), "LSTM"), legendKey(css("--ink-2"), "Mamba", {style: "dash"}));
  const get = (mode, core, lead) => rows.find(r => r.mode === mode && r.core === core && r.lead === lead)?.mean;
  const neg = rows.filter(r => r.mode === "persistence" && r.mean < 0).map(r => r.lead);
  $("leads-foot").textContent = t("leads.foot", {
    g1: fmtFixed(get("gefs", "lstm", 1), 2), g7: fmtFixed(get("gefs", "lstm", 7), 2),
    p1: fmtFixed(get("past", "lstm", 1), 2), p7: fmtFixed(get("past", "lstm", 7), 2),
    u7: fmtFixed(get("perfect", "lstm", 7), 2), neg: neg.length ? Math.min(...neg) : "–"});
}

// ---------- 2.3 nhìn lại 365 hay 730 ngày ----------
async function refreshMemory() {
  const rows = (await cached("/api/d-pairs")).rows.slice()
    .sort((a, b) => CORES.indexOf(parseConf(a.configuration).core) - CORES.indexOf(parseConf(b.configuration).core) || a.seed - b.seed);
  const label = r => `${CORE[parseConf(r.configuration).core]} · seed ${r.seed}`;
  const y = rows.map(label);
  const connectors = {type: "scatter", mode: "lines", hoverinfo: "skip", showlegend: false, line: {color: css("--line"), width: 3},
    x: rows.flatMap(r => [r.median_365, r.median_730, null]), y: rows.flatMap(r => [label(r), label(r), null])};
  await Plotly.react($("mem-chart"), [connectors,
    {type: "scatter", mode: "markers", x: rows.map(r => r.median_365), y, name: t("mem.d365"), marker: {size: 13, color: css("--c-persist")}, hovertemplate: `%{x:.4f}<extra>${t("mem.d365")}</extra>`},
    {type: "scatter", mode: "markers", x: rows.map(r => r.median_730), y, name: t("mem.d730"), marker: {size: 13, color: css("--accent"), symbol: "diamond"}, hovertemplate: `%{x:.4f}<extra>${t("mem.d730")}</extra>`}],
  baseLayout({margin: {l: 140, r: 16, t: 40, b: 52}, xaxis: {title: {text: t("mem.x")}, gridcolor: css("--grid")}, yaxis: {autorange: "reversed", ticksuffix: "  "},
    legend: {orientation: "h", x: 0, y: 1.1}, showlegend: true}), plotConfig);
  const head = el("tr", {}, ...[["mem.col.model", ""], ["mem.d365", "n"], ["mem.d730", "n"], ["mem.col.delta", "n"], ["mem.col.better", "n"], ["mem.col.p", "n"]]
    .map(([k, cls]) => el("th", {class: cls || null}, t(k))));
  $("mem-table").replaceChildren(el("thead", {}, head), el("tbody", {}, ...rows.map(r => {
    const d = r.median_730 - r.median_365;
    return el("tr", {}, el("td", {}, label(r)), el("td", {class: "n"}, fmtFixed(r.median_365, 3)), el("td", {class: "n"}, fmtFixed(r.median_730, 3)),
      el("td", {class: "n " + (d > 0 ? "up" : d < 0 ? "down" : "")}, fmtSigned(d, 4)),
      el("td", {class: "n"}, `${r.improved}/${r.n_pairs}`), el("td", {class: "n"}, r.p_holm < 0.01 ? `< ${fmtFixed(0.01, 2)}` : fmtFixed(r.p_holm, 2)));
  })));
  const deltas = rows.map(r => r.median_730 - r.median_365);
  // So với mức dao động NSE validation giữa 3 hạt giống của B0 (0,0079) để nói chênh lệch có đáng kể không
  const small = deltas.every(d => Math.abs(d) < 0.0079);
  $("mem-foot").textContent = t("mem.foot", {lo: fmtSigned(Math.min(...deltas), 4), hi: fmtSigned(Math.max(...deltas), 4), sig: rows.filter(r => r.p_holm < 0.05).length, n: rows.length})
    + (small ? " " + t("mem.small") : "");
}

// ---------- 2.4 cảnh báo lũ ----------
async function refreshFloodAll() {
  const sec = state.sec.flood, runId = sec.run, lead = sec.lead;
  const data = await cached(`/api/flood-summary${qs({run: runId, lead})}`);
  if (runId !== sec.run || lead !== sec.lead) return;
  $("flood-all-title").textContent = t("flood.all", {m: secName(sec)});
  const labels = data.rows.map(r => t("flood.period", {p: fmt(r.return_period_years, 1)}));
  const series = [["hit", "flood.hit", "--c-hit"], ["missed", "flood.missed", "--c-miss"], ["false_alarm", "flood.false", "--c-false"]];
  await Plotly.react($("flood-chart"), series.map(([key, label, color]) => ({type: "bar", orientation: "h", y: labels, x: data.rows.map(r => r[key]),
    name: t(label), marker: {color: css(color)}, hovertemplate: "%{x}<extra>%{fullData.name}</extra>"})),
  baseLayout({barmode: "stack", margin: {l: 130, r: 16, t: 10, b: 52}, yaxis: {autorange: "reversed", ticksuffix: "  "},
    xaxis: {title: {text: t("flood.x")}, gridcolor: css("--grid")}}), plotConfig);
  $("flood-legend").replaceChildren(...series.map(([, label, color]) => legendKey(css(color), t(label), {style: "box"})));
  const two = data.rows.find(r => r.return_period_years === 2);
  if (two) {
    const total = two.hit + two.missed;
    $("flood-sentence").textContent = t("flood.sentence", {hit: two.hit, total, pod: total ? pct(two.hit / total) : "–", fa: two.false_alarm});
  }
}
async function refreshFloodBasin() {
  const sec = state.sec.flood, runId = sec.run, lead = sec.lead, basin = state.basin;
  if (!runId) return;
  if (isRegion()) {
    $("flood-basin-title").textContent = t("flood.basinPick");
    $("flood-events").replaceChildren(el("tbody", {}, el("tr", {}, el("td", {class: "empty"}, t("flood.pickHint")))));
    return;
  }
  const data = await cached(`/api/flood${qs({run: runId, basin, lead})}`);
  if (runId !== sec.run || lead !== sec.lead || basin !== state.basin) return;
  $("flood-basin-title").textContent = t("flood.basin", {s: state.site.stations[basin].name, m: secName(sec)});
  const rows = (data.event_cases ?? []).slice().sort((a, b) => (a.obs_onset ?? a.sim_onset).localeCompare(b.obs_onset ?? b.sim_onset) || a.return_period_years - b.return_period_years);
  if (!rows.length) {
    $("flood-events").replaceChildren(el("tbody", {}, el("tr", {}, el("td", {class: "empty"}, t("flood.noEvents")))));
    return;
  }
  const head = el("tr", {}, ...["flood.col.start", "flood.col.period", "flood.col.status"].map(k => el("th", {}, t(k))));
  $("flood-events").replaceChildren(el("thead", {}, head), el("tbody", {}, ...rows.map(r => {
    const when = r.obs_onset ?? r.sim_onset;
    return el("tr", {title: when, onclick: () => { setDay(state.dates.indexOf(when)); scrollToDash(); }},
      el("td", {}, fmtDate(when)), el("td", {}, t("flood.years", {p: fmt(r.return_period_years, 1)})),
      el("td", {}, el("span", {class: `pill ${r.status}`}, t("flood.st." + r.status))));
  })));
}

// ---------- 2.5 bốn mức lưu lượng ----------
function heat(share) {
  // Nền nhạt => xanh đậm theo tỉ lệ trong hàng; chữ trắng khi nền đậm
  const a = Math.max(0, Math.min(1, share));
  const mix = (x, y) => Math.round(x + (y - x) * a);
  return {bg: `rgb(${mix(241, 12)},${mix(246, 70)},${mix(247, 92)})`, fg: a > 0.6 ? "#fff" : css("--ink")};
}
async function refreshLevels() {
  const sec = state.sec.levels, runId = sec.run, lead = sec.lead;
  const ref = lead === 1 && runId !== "SeqLSTM_Q" ? "SeqLSTM_Q" : runId !== "Persistence_qmax" ? "Persistence_qmax" : null;
  const [data, base] = await Promise.all([cached(`/api/confusion${qs({run: runId, lead})}`), ref ? cached(`/api/confusion${qs({run: ref, lead})}`) : null]);
  if (runId !== sec.run || lead !== sec.lead) return;
  const name = shortName(model(runId));
  $("lvl-title").textContent = t("lvl.matrix", {m: secName(sec)});
  $("lvl-recall-title").textContent = t("lvl.recallTitle") + ` · ${secName(sec)}` + (ref ? ` ${t("lvl.vs")} ${shortName(model(ref))}` : "");
  if (!data.matrix) { $("confusion").replaceChildren(el("tbody", {}, el("tr", {}, el("td", {class: "empty"}, t("lvl.none"))))); return; }
  const m = data.matrix, rowSum = m.map(r => r.reduce((a, b) => a + b, 0));
  const head = el("tr", {}, el("th", {class: "axis rowh"}, t("lvl.obsPred")), ...LEVELS.map((_, j) => el("th", {}, levelText(j))));
  const body = m.map((row, i) => el("tr", {}, el("th", {class: "rowh"}, levelText(i)), ...row.map((v, j) => {
    const share = rowSum[i] ? v / rowSum[i] : 0, c = heat(share);
    return el("td", {class: i === j ? "diag" : null, style: `background:${c.bg};color:${c.fg}`, title: `${fmt(v, 0)} ${t("lvl.unit")}`},
      pct(share), el("small", {}, fmt(v, 0)));
  })));
  $("confusion").replaceChildren(el("thead", {}, head), el("tbody", {}, ...body));
  const recall = mat => mat.map((row, i) => { const s = row.reduce((a, b) => a + b, 0); return s ? row[i] / s : null; });
  const x = LEVELS.map((_, i) => levelText(i));
  const traces = [{type: "bar", x, y: recall(m), name, marker: {color: css("--accent")},
    text: recall(m).map(v => pct(v)), textposition: "outside", hovertemplate: `%{y:.1%}<extra>${name}</extra>`}];
  if (base?.matrix) traces.push({type: "bar", x, y: recall(base.matrix), name: shortName(model(ref)),
    marker: {color: css(ref === "SeqLSTM_Q" ? "--c-biascast" : "--c-persist")}, text: recall(base.matrix).map(v => pct(v)), textposition: "outside",
    hovertemplate: `%{y:.1%}<extra>${shortName(model(ref))}</extra>`});
  await Plotly.react($("lvl-chart"), traces, baseLayout({barmode: "group", margin: {l: 64, r: 16, t: 40, b: 52},
    yaxis: {title: {text: t("lvl.recallY")}, range: [0, 1.15], tickformat: ".0%", gridcolor: css("--grid")},
    legend: {orientation: "h", x: 0, y: 1.12}, showlegend: true}), plotConfig);
  const vh = m[3], vhSum = rowSum[3];
  const colVh = m.reduce((a, r) => a + r[3], 0);
  const correct = m.reduce((a, r, i) => a + r[i], 0), total = rowSum.reduce((a, b) => a + b, 0);
  $("lvl-foot").textContent = t("lvl.foot", {acc: pct(correct / total), miss: pct((vhSum - vh[3]) / vhSum), fa: pct((colVh - vh[3]) / colVh), n: fmt(total, 0)});
}

// ---------- 2.6 giải thích (XAI) ----------
function xaiLayout(extra) {
  const layout = baseLayout({margin: {l: 64, r: 16, t: 12, b: 56}, ...extra});
  if (window.innerWidth < 640) layout.margin = {...layout.margin, l: Math.min(layout.margin.l, 170)};
  return layout;
}
async function refreshXai() {
  const basin = state.basin, task = state.task;
  if (isRegion()) {
    $("xai-status").textContent = t("xai.pick");
    $("xai-key").replaceChildren();
    Plotly.purge($("xai-1")); Plotly.purge($("xai-2"));
    $("xai-1-title").textContent = ""; $("xai-2-title").textContent = "";
    return;
  }
  const s = state.site.stations[basin], main = model(activeRun());
  const params = task === "h" ? {scope: "h", basin, mode: main.mode, lead: state.lead} : {scope: "c2", basin, core: coreOf(main) ?? "lstm"};
  const data = await cached(`/api/explanation${qs(params)}`).catch(() => null);
  if (basin !== state.basin || task !== state.task) return;
  if (!data || data.scope === "none") { $("xai-status").textContent = t("xai.none"); return; }
  const keyCores = data.scope === "h" ? ["lstm", "mamba"].filter(c => data.cores[c]) : data.cores_order.filter(c => data.group_share[c]);
  const keys = keyCores.map(c => legendKey(CORE_COLOR[c], CORE[c], {style: "box"}));
  if (data.scope === "h") keys.push(legendKey(css("--ink-2"), t("xai.fore")), legendKey(css("--ink-2"), t("xai.past"), {style: "dot"}));
  $("xai-key").replaceChildren(...keys);
  if (data.scope === "h") {
    const li = data.lead - 1, cores = ["lstm", "mamba"].filter(c => data.cores[c]);
    $("xai-status").textContent = t("xai.hstatus", {s: s.name, mode: t("mode." + data.mode), lead: data.lead, n: data.n_samples_by_lead[li]});
    $("xai-1-title").textContent = t("xai.phase");
    $("xai-2-title").textContent = t("xai.groupsH", {lead: days(data.lead)});
    const tr1 = cores.flatMap(c => [["fore", "solid"], ["past", "dot"]].map(([ph, dash]) => ({type: "scatter", mode: "lines+markers", x: data.leads,
      y: data.cores[c].phase_by_lead[ph].map(v => (v == null ? null : v * 100)), name: `${CORE[c]} · ${t("xai." + ph)}`,
      line: {color: CORE_COLOR[c], dash, width: 2.4}, marker: {size: 7}})));
    await Plotly.react($("xai-1"), tr1, xaiLayout({xaxis: {title: {text: t("xai.phaseX")}, dtick: 1},
      yaxis: {title: {text: t("xai.share")}, rangemode: "tozero", gridcolor: css("--grid")}}), plotConfig);
    const groups = data.groups.filter(g => cores.some(c => (data.cores[c].group_share_lead[g] ?? 0) > 1e-3));
    const gname = g => t("g." + g.split(":").at(-1));
    await Plotly.react($("xai-2"), cores.map(c => ({type: "bar", orientation: "h", y: groups.map(gname), x: groups.map(g => (data.cores[c].group_share_lead[g] ?? 0) * 100),
      name: CORE[c], marker: {color: CORE_COLOR[c]}})), xaiLayout({barmode: "group",
      margin: {l: 250, r: 16, t: 12, b: 56}, yaxis: {autorange: "reversed", ticksuffix: "  "}, xaxis: {title: {text: t("xai.share")}, gridcolor: css("--grid")}}), plotConfig);
  } else {
    const order = data.cores_order.filter(c => data.group_share[c]);
    const n = data.memory.find(r => r.core === params.core)?.n_samples ?? data.memory[0]?.n_samples;
    $("xai-status").textContent = t("xai.c2status", {s: s.name, n}) + (task === "d" ? " " + t("xai.dnote") : "");
    $("xai-1-title").textContent = t("xai.groups");
    $("xai-2-title").textContent = t("xai.lag");
    await Plotly.react($("xai-1"), order.map(c => ({type: "bar", orientation: "h", y: data.groups.map(g => t("g." + g)), x: data.groups.map(g => (data.group_share[c][g] ?? 0) * 100),
      name: CORE[c], marker: {color: CORE_COLOR[c]}, opacity: c === params.core ? 1 : 0.5})),
    xaiLayout({barmode: "group", margin: {l: 250, r: 16, t: 12, b: 56},
      yaxis: {autorange: "reversed", ticksuffix: "  "}, xaxis: {title: {text: t("xai.share")}, gridcolor: css("--grid")}}), plotConfig);
    const lags = Array.from({length: 61}, (_, i) => i);
    await Plotly.react($("xai-2"), order.filter(c => data.lag_cum_60[c]).map(c => ({type: "scatter", mode: "lines", x: lags,
      y: data.lag_cum_60[c].map(v => (v == null ? null : v * 100)), name: CORE[c], line: {color: CORE_COLOR[c], width: c === params.core ? 3.2 : 1.8}})),
    xaiLayout({xaxis: {title: {text: t("xai.lagX")}}, yaxis: {title: {text: t("xai.lagY")}, range: [0, 102], gridcolor: css("--grid")},
      shapes: [{type: "line", x0: 0, x1: 60, y0: 90, y1: 90, line: {color: css("--muted"), dash: "dash", width: 1}}]}), plotConfig);
  }
}

// ---------- thẻ "Thử kịch bản thời tiết" (chạy mô hình thật trên máy chủ) ----------
function wiDate(m) {
  let date = state.dates[state.day];
  if (m?.origin_start && date < m.origin_start) date = m.origin_start;
  if (m?.origin_end && date > m.origin_end) date = m.origin_end;
  return date;
}
function resetWhatIf(force = false) {
  // Đổi trạm hoặc ngày => kết quả cũ không còn ứng với lựa chọn: xóa và ghi lại tiêu đề
  if (!$("wi-model").options.length) return;
  if (state.wiDone || force) { Plotly.purge($("wi-chart")); state.wiDone = false; }
  const m = model($("wi-model").value);
  $("wi-title").textContent = t("wi.title", {s: isRegion() ? t("region.name") : state.site.stations[state.basin].name, d: fmtDate(wiDate(m))});
  $("wi-run").disabled = isRegion();
  $("wi-status").textContent = isRegion() ? t("wi.needBasin") : "";
}
function showTab(which) {
  const flow = which === "flow";
  $("tab-flow").setAttribute("aria-selected", String(flow));
  $("tab-whatif").setAttribute("aria-selected", String(!flow));
  $("pane-flow").hidden = !flow;
  $("pane-whatif").hidden = flow;
  if (flow) Plotly.Plots.resize($("hydrograph"));
  else resetWhatIf();
}
function initWhatIf() {
  const live = models().filter(m => m.what_if);
  $("tab-whatif").hidden = !live.length;
  if (!live.length) return;
  initWhatIf.fill = () => {
    const keep = $("wi-model").value, main = live.some(m => m.run === activeRun()) ? activeRun() : null;
    $("wi-model").replaceChildren(...groupedOptions(live, m => withCode(shortName(m), m), keep || main || live[0].run, m => t("task." + taskOf(m))));
  };
  initWhatIf.fill();
  $("wi-model").addEventListener("change", () => resetWhatIf(true));
  $("tab-flow").addEventListener("click", () => showTab("flow"));
  $("tab-whatif").addEventListener("click", () => showTab("whatif"));
  const showValues = () => {
    $("wi-rain-v").textContent = `× ${fmt(Number($("wi-rain").value), 1)}`;
    const dt = Number($("wi-temp").value);
    $("wi-temp-v").textContent = `${dt > 0 ? "+" : ""}${fmt(dt, 1)} °C`;
  };
  for (const id of ["wi-rain", "wi-temp"]) $(id).addEventListener("input", showValues);
  showValues();
  $("wi-form").addEventListener("submit", e => {
    e.preventDefault();
    if (isRegion()) return;
    const btn = $("wi-run");
    btn.disabled = true;
    btn.textContent = t("wi.running");
    const m = model($("wi-model").value);
    const date = wiDate(m);
    stopPlay();
    run(async () => {
      try {
        const res = await api("/api/what-if", {method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({run: m.run, basin: state.basin, date, rain_multiplier: Number($("wi-rain").value), temperature_delta: Number($("wi-temp").value)})});
        const x = res.baseline.map(r => fmtDate(r.target_date));
        const before = res.baseline.map(r => r.qmax_m3_s), after = res.changed.map(r => r.qmax_m3_s);
        const d = after[0] - before[0];
        $("wi-status").textContent = t("wi.status", {s: state.site.stations[state.basin].name, d: fmtDate(date)}) + " "
          + t("wi.delta", {d: x[0], a: fmtQ(before[0]), b: fmtQ(after[0]), c: `${d >= 0 ? "+" : "−"}${fmtQ(Math.abs(d))}`, p: pct(before[0] ? d / before[0] : null)});
        const bar = (y, name, color) => ({type: "bar", x, y, name, marker: {color}, text: y.map(fmtQ), textposition: "outside", cliponaxis: false,
          hovertemplate: "%{y:.2f} m³/s<extra>%{fullData.name}</extra>"});
        await Plotly.react($("wi-chart"), [bar(before, t("wi.base"), css("--c-persist")), bar(after, t("wi.changed"), css("--c-model"))],
          baseLayout({barmode: "group", bargap: x.length === 1 ? 0.55 : 0.25, showlegend: true, legend: {orientation: "h", x: 0, y: 1.14}, margin: {l: 66, r: 16, t: 36, b: 52},
            xaxis: {type: "category", title: {text: t("wi.x")}}, yaxis: {title: {text: t("chart.y")}, rangemode: "tozero", gridcolor: css("--grid")}}), plotConfig);
        state.wiDone = true;
        $("wi-title").textContent = t("wi.title", {s: state.site.stations[state.basin].name, d: fmtDate(date)});
      } finally { btn.disabled = isRegion(); btn.textContent = t("wi.run"); }
    });
  });
}

// ---------- làm mới theo lựa chọn ----------
function refreshAll() {
  run(refreshDay);
  run(() => refreshChart());
  if (state.resultsReady) run(refreshXai);
  initWhatIf.fill?.();
  resetWhatIf(true);
}
async function refreshResults() {
  await refreshCompare();
  run(refreshFloodSection);
  run(refreshLevels);
}
function scrollToDash() { document.querySelector(".dash").scrollIntoView({behavior: "smooth"}); }
function initResultsLazy() {
  // Phần 2 (bản đồ NSE, 6 hình) chỉ dựng khi người dùng cuộn gần tới => trang đầu nhẹ, bớt một bản đồ WebGL lúc tải
  const io = new IntersectionObserver(entries => {
    if (!entries.some(e => e.isIntersecting) || state.resultsReady) return;
    state.resultsReady = true;
    io.disconnect();
    nseMap = newMap("nse-map");
    run(refreshResults);
    run(refreshLeads);
    run(refreshMemory);
    run(refreshXai);
  }, {rootMargin: "600px 0px"});
  io.observe($("results"));
}
function initScrollSpy() {
  // Tô mục đang xem trên thanh mục lục Phần 2 và thanh trên cùng
  const links = [...document.querySelectorAll(".subnav a, .nav a")];
  const ids = ["explore", "how", "results", "about", "compare", "leads", "memory", "flood", "levels", "xai"];
  const seen = new Map();
  const io = new IntersectionObserver(entries => {
    for (const e of entries) seen.set(e.target.id, e.isIntersecting);
    const visible = id => seen.get(id);
    const sub = ["xai", "levels", "flood", "memory", "leads", "compare"].find(visible);
    const top = visible("about") ? "about" : visible("results") ? "results" : visible("how") ? "how" : "explore";
    for (const a of links) {
      const id = a.getAttribute("href").slice(1);
      a.classList.toggle("on", a.closest(".subnav") ? id === sub : id === top);
    }
  }, {rootMargin: "-45% 0px -50% 0px"});
  for (const id of ids) io.observe($(id));
}
function renderTexts() {
  applyStatic();
  $("lede").textContent = t("intro.lede", {basins: state.ids.length, years: testYears()});
  $("years").replaceChildren(...[...new Set(state.dates.map(d => d.slice(0, 4)))].map(y => el("span", {}, y)));
  document.querySelectorAll("[data-lang]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.lang === getLang())));
  document.querySelectorAll("[data-colorby]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.colorby === state.colorBy)));
  $("about-body").innerHTML = aboutHtml(getLang(), state);
  document.title = t("brand");
  renderLegend();
  renderStation();
  renderStatus();
  renderDayLabel();
  renderModelControls();
  renderCompareControls();
  renderSectionPickers();
  initWhatIf.fill?.();
  resetWhatIf();
  renderHighlights();
  renderTour();
}

async function start() {
  const [catalog, site] = await Promise.all([api("/api/catalog"), api("/api/site")]);
  Object.assign(state, {catalog, site, dates: catalog.dates, ids: catalog.ids});
  state.main.c2 = catalog.default_run;
  state.main.h = (models().find(m => m.stage === "H" && m.mode === "gefs" && m.core === "lstm") ?? stageModels("H")[0]).run;
  state.main.d = stageModels("D")[0].run;
  state.cmpRun = state.main.c2;
  state.sec.flood = {run: state.main.c2, lead: 1};
  state.sec.levels = {run: state.main.c2, lead: 1};
  const fromHash = /basin=(\w+)/.exec(location.hash)?.[1];
  state.basin = fromHash === "all" || site.stations[fromHash] ? fromHash : "2";
  state.lastBasin = state.basin === "all" ? "2" : state.basin;
  state.day = state.dates.indexOf("2016-06-01") >= 0 ? state.dates.indexOf("2016-06-01") : Math.floor(state.dates.length / 2);
  // Tên tháng, ngày tiếng Việt cho trục thời gian của Plotly
  Plotly.register({moduleType: "locale", name: "vi", dictionary: {}, format: {
    days: ["Chủ nhật", "Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy"],
    shortDays: ["CN", "T2", "T3", "T4", "T5", "T6", "T7"],
    months: Array.from({length: 12}, (_, i) => `Tháng ${i + 1}`),
    shortMonths: Array.from({length: 12}, (_, i) => `Th${i + 1}`),
    date: "%d/%m/%Y", decimal: ",", thousands: "."}});
  setLang(getLang());
  $("day-slider").max = state.dates.length - 1;
  initWhatIf();
  renderTexts();
  bindSearch();
  const protocol = new pmtiles.Protocol();
  maplibregl.addProtocol("pmtiles", protocol.tile);
  initMainMap();
  initMiniMap();
  initResultsLazy();
  initScrollSpy();
  initHighlights();
  $("tour-btn").addEventListener("click", startTour);
  $("tour-next").addEventListener("click", () => tourGo(tour.k + 1, 1));
  $("tour-prev").addEventListener("click", () => tourGo(tour.k - 1, -1));
  $("tour-close").addEventListener("click", endTour);
  $("chart-csv").addEventListener("click", downloadChartCsv);

  $("day-slider").addEventListener("input", e => setDay(Number(e.target.value), {fromSlider: true}));
  $("prev-day").addEventListener("click", () => setDay(state.day - 1));
  $("next-day").addEventListener("click", () => setDay(state.day + 1));
  $("play").addEventListener("click", togglePlay);
  document.querySelectorAll("[data-speed]").forEach(b => b.addEventListener("click", () => setSpeed(Number(b.dataset.speed))));
  $("region-btn").addEventListener("click", toggleRegion);
  document.addEventListener("keydown", e => {
    if (e.target.closest("input, select, textarea, button")) return;
    if (e.key === "ArrowLeft") setDay(state.day - 1);
    if (e.key === "ArrowRight") setDay(state.day + 1);
  });
  document.querySelectorAll("[data-colorby]").forEach(b => b.addEventListener("click", () => {
    state.colorBy = b.dataset.colorby;
    document.querySelectorAll("[data-colorby]").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    renderLegend();
    if (map?.getLayer("gauges")) {
      map.setPaintProperty("gauges", "circle-color", levelMatch(dotKey()));
      map.setPaintProperty("gauges", "circle-radius", dotRadius(dotKey()));
    }
  }));
  document.querySelectorAll("[data-lang]").forEach(b => b.addEventListener("click", () => {
    if (b.dataset.lang === getLang()) return;
    setLang(b.dataset.lang);
    renderTexts();
    relabelMap(map, MAIN_LAYERS);
    map.once("idle", () => run(refreshDay));
    run(() => refreshChart({keepRange: true}));
    if (state.resultsReady) {
      relabelMap(nseMap, ["study-line", "nse"]);
      run(refreshResults);
      run(refreshLeads);
      run(refreshMemory);
      run(refreshXai);
    }
  }));
  $("model").addEventListener("change", e => { state.main[state.task] = e.target.value; syncCompareSection(); renderModelControls(); refreshAll(); });
  $("lead").addEventListener("input", e => { state.lead = Number(e.target.value); syncCompareSection(); renderModelControls(); refreshAll(); });
  $("add-model").addEventListener("change", e => {
    if (!e.target.value) return;
    state.compare[state.task] = [...compareRuns(), e.target.value].slice(0, MAX_COMPARE);
    renderModelControls();
    run(refreshChart);
  });
  $("ensemble").addEventListener("change", e => { state.ensemble = e.target.checked; run(() => refreshChart({keepRange: true})); });
  $("cmp-lead").addEventListener("change", e => { state.cmpLead = Number(e.target.value); state.cmpRun = null; run(refreshCompare); });
  $("nse-ref").addEventListener("change", e => { state.nseRef = e.target.value; run(() => refreshNseMap(cmpLeadNow())); });

  // Khung biểu đồ đổi cỡ => chỉnh chiều cao hình theo khung
  const hydro = $("hydrograph");
  new ResizeObserver(() => {
    const h = Math.max(260, hydro.clientHeight);
    if (hydro._fullLayout && Math.abs(hydro._fullLayout.height - h) > 2) Plotly.relayout(hydro, {height: h});
  }).observe(hydro);
  let resizeTimer;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => run(() => refreshChart({keepRange: true})), 250); });

  applySelection();
}

window.addEventListener("DOMContentLoaded", () => run(start));
