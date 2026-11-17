"use strict";

const $ = id => document.getElementById(id);
const color = ["#157a68", "#31569c", "#bb5627", "#8062ad", "#857023", "#376d7e", "#96565d", "#527a3a", "#676c72"];
const number = new Intl.NumberFormat("vi", {maximumFractionDigits: 4});
const dateText = input => input.split("-").reverse().join("/");
let catalog, page = "forecast", revision = 0, comparisonTables = {};
const chosen = new Set();

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error ?? `Yêu cầu chưa thực hiện được (HTTP ${response.status}).`);
  }
  return response.json();
}
function options(select, values, selected) {
  select.replaceChildren(...values.map(([value, label]) => new Option(label, value, false, String(value) === String(selected))));
}
function query(values) {
  const result = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    for (const v of [].concat(value)) if (v !== "") result.append(key, v);
  }
  return "?" + result;
}
function table(container, rows, keys = Object.keys(rows[0] ?? {})) {
  container.replaceChildren();
  if (!rows.length) {
    const message = document.createElement("p");
    message.textContent = "Chưa có kết quả cho lựa chọn này.";
    container.append(message);
    return;
  }
  const wrap = document.createElement("div");
  wrap.className = "table-scroll";
  const grid = document.createElement("table");
  const head = grid.createTHead().insertRow();
  for (const key of keys) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = key;
    head.append(th);
  }
  const body = grid.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const key of keys) {
      const value = row[key];
      tr.insertCell().textContent = typeof value === "number" ? number.format(value) : (value ?? "—");
    }
  }
  wrap.append(grid);
  container.append(wrap);
}
const base = {paper_bgcolor: "#fff", plot_bgcolor: "#fff", font: {family: "DM Sans, Segoe UI, sans-serif", color: "#172e29", size: 12}};
const plotOptions = {responsive: true, displayModeBar: false};

function map(target, scores = null) {
  const byBasin = new Map(scores?.map(r => [r.basin_id, r.NSE]));
  const points = scores ? catalog.points.filter(p => Number.isFinite(byBasin.get(p.id))) : catalog.points;
  const graph = $(target);
  const selected = $("basin").value;
  const marker = scores
    ? {size: 8, color: points.map(p => byBasin.get(p.id)), colorscale: "RdBu", cmid: 0, showscale: true}
    : {size: points.map(p => p.id === selected ? 10 : 5), color: points.map(p => p.id === selected ? "#c55226" : "#137b67")};
  const regions = (catalog.outline ?? []).map(polygon => ({type: "scatter", mode: "lines", fill: "toself",
    x: polygon.map(p => p[0]), y: polygon.map(p => p[1]), line: {color: "#b7cdbf", width: 1},
    fillcolor: "#edf4ec", hoverinfo: "skip", showlegend: false}));
  Plotly.react(graph, [...regions, {type: "scatter", mode: "markers", x: points.map(p => p.lon), y: points.map(p => p.lat),
    customdata: points.map(p => p.id), marker,
    text: points.map(p => scores ? `ID ${p.id} · NSE ${number.format(byBasin.get(p.id))}` : `ID ${p.id}`),
    hovertemplate: "%{text}<extra></extra>"
  }], {...base, showlegend: false, margin: {l: 10, r: 10, t: 6, b: 6}, xaxis: {visible: false},
    yaxis: {visible: false, scaleanchor: "x", scaleratio: 1.5}, dragmode: "pan"}, plotOptions).then(() => {
    if (target === "map") {
      graph.removeAllListeners("plotly_click");
      graph.on("plotly_click", event => {
        $("basin").value = event.points[0].customdata;
        changedForecast();
      });
    }
  });
}

function updateToggles() {
  const lead = Number($("lead").value);
  $("models").replaceChildren();
  const groups = new Map();
  for (const model of catalog.models) {
    if (!groups.has(model.stage)) groups.set(model.stage, []);
    groups.get(model.stage).push(model);
  }
  for (const [stage, models] of groups) {
    const label = document.createElement("div");
    label.className = "group";
    label.textContent = stage;
    $("models").append(label);
    for (const m of models) {
      const row = document.createElement("label");
      row.className = "check";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = chosen.has(m.run);
      input.disabled = !m.leads.includes(lead);
      row.classList.toggle("inactive", input.disabled);
      input.addEventListener("change", () => {
        if (input.checked) chosen.add(m.run); else chosen.delete(m.run);
        run(refreshForecast);
      });
      row.append(input, document.createTextNode(m.name + (m.seed === undefined ? "" : ` · seed ${m.seed}`)));
      $("models").append(row);
    }
  }
}

async function refreshForecast() {
  const serial = ++revision;
  const basin = $("basin").value, date = $("day").value, lead = $("lead").value;
  const data = await api("/api/forecast" + query({basin, date, lead, run: [...chosen]}));
  if (page !== "forecast" || serial !== revision) return;
  $("plot-subtitle").textContent = `Lưu vực ${basin} · Hạn ${lead} ngày · Ngày đích ${dateText(date)}`;
  const series = [{x: data.dates, y: data.observed, name: "Quan trắc", line: {color: "#172e29", width: 2.5}}];
  for (const {run, values} of data.predictions) {
    const i = catalog.models.findIndex(m => m.run === run);
    series.push({x: data.dates, y: values, name: catalog.models[i].name, line: {color: color[i % color.length], width: 2}});
  }
  for (const [period, value] of Object.entries(data.thresholds).sort((a, b) => Number(a[0]) - Number(b[0]))) {
    series.push({x: [data.dates[0], data.dates.at(-1)], y: [value, value], name: `Ngưỡng ${number.format(Number(period))} năm`,
      line: {dash: "dot", color: "#b76e4c", width: 1}});
  }
  await Plotly.react($("hydrograph"), series.map(trace => ({...trace, type: "scatter", mode: "lines", connectgaps: false})),
    {...base, margin: {l: 65, r: 22, t: 16, b: 75}, hovermode: "x unified", yaxis: {title: {text: "qmax (m³/s)"}, rangemode: "tozero"},
      xaxis: {type: "date"}, legend: {orientation: "h", y: -.23},
      shapes: [{type: "line", x0: date, x1: date, yref: "paper", y0: 0, y1: 1, line: {color: "#809a8b", dash: "dash", width: 1}}]}, plotOptions);
}
function changedForecast() {
  $("what-if-results").replaceChildren();
  map("map");
  updateToggles();
  run(refreshForecast);
}
const tableNames = {C2_config_table: "C2 · Hiệu năng", C2_cost_table: "C2 · Chi phí", D2_CPU_benchmark: "D2 · CPU",
  c2_scenarios: "C2 · Kịch bản S0–S4", D: "D · 365/730 bước", D_cost: "D · Chi phí"};
function showComparisonTable() {
  table($("comparison-content"), comparisonTables[$("comparison-table").value] ?? []);
}
function updateModelLeads(modelPicker, leadPicker) {
  const active = catalog.models.find(m => m.run === $(modelPicker).value);
  options($(leadPicker), active.leads.map(lead => [lead, `${lead} ngày`]), $(leadPicker).value);
}
function updateReferences() {
  const active = catalog.models.find(m => m.run === $("comparison-run").value);
  const lead = Number($("comparison-lead").value);
  let choices = [];
  if (active.stage === "D") {
    choices = [["C2_same_seed", "C2 365 bước cùng cấu hình và seed"]];
  } else {
    const inH = active.stage === "H" || (active.run === "Persistence_qmax" && lead > 1);
    choices = catalog.models.filter(m => m.leads.includes(lead) && m.stage !== "D" &&
      (m.run === "Persistence_qmax" || inH === (m.stage === "H")))
      .map(m => [m.run, m.name]);
  }
  options($("comparison-ref"), [["", "NSE tuyệt đối"], ...choices], $("comparison-ref").value);
  if ($("comparison-ref").selectedIndex < 0) $("comparison-ref").selectedIndex = 0;
}
async function refreshComparison() {
  const serial = ++revision;
  const data = await api("/api/comparison" + query({run: $("comparison-run").value,
    reference: $("comparison-ref").value, lead: $("comparison-lead").value}));
  if (page !== "comparison" || serial !== revision) return;
  map("nse-map", data.metrics);
  $("nse-status").textContent = data.metrics.length ? "" : "Chưa có bảng NSE trên giao ngày chung cho lựa chọn này.";
  comparisonTables = data.tables;
  options($("comparison-table"), Object.keys(data.tables).map(k => [k, tableNames[k] ?? `H · Hạn ${k.split("lead").at(-1)}`]), $("comparison-table").value);
  showComparisonTable();
}
async function refreshFlood() {
  const serial = ++revision;
  const data = await api("/api/flood" + query({run: $("flood-model").value, basin: $("flood-basin").value, lead: $("flood-lead").value}));
  if (page !== "flood" || serial !== revision) return;
  const content = $("flood-content");
  content.replaceChildren();
  for (const [title, key, fields] of [
    ["Tóm tắt theo ngưỡng lũ", "events", ["return_period_years", "hit", "missed", "false_alarm", "observed_censored", "predicted_censored"]],
    ["Từng trận", "event_cases", ["return_period_years", "status", "obs_onset", "sim_onset", "obs_peak", "sim_peak"]],
    ["Bốn mức lưu lượng", "levels", undefined]
  ]) {
    const heading = document.createElement("h2"), detail = document.createElement("div");
    heading.textContent = title;
    content.append(heading, detail);
    table(detail, data[key] ?? [], fields);
  }
}
async function refreshExplanation() {
  const selected = $("explanation-case").value;
  const [basin, origin] = selected.split("/");
  const data = await api("/api/explanation" + (selected ? query({basin, origin}) : ""));
  if (page !== "explanation") return;
  const content = $("explanation-content");
  $("explanation-picker").hidden = !data.cases.length;
  if (!data.cases.length) {
    content.textContent = "Đang chờ kết quả Integrated Gradients của bước G.";
    return;
  }
  options($("explanation-case"), data.cases.map(row => [row.basin_id+"/"+row.origin_date,
    `ID ${row.basin_id} · ${dateText(row.origin_date)}`]), data.basin+"/"+data.origin);
  const rows = data.rows.filter(row => Number.isFinite(row.day_offset) && Number.isFinite(row.attribution));
  const traces = [];
  for (const core of [...new Set(rows.map(row => row.core))].sort()) {
    const own = rows.filter(row => row.core === core);
    const groups = [...new Set(own.map(row => row.group))].sort();
    const days = [...new Set(own.map(row => row.day_offset))].sort((a,b) => a-b);
    const lookup = new Map(own.map(row => [row.group+"/"+row.day_offset, row.attribution]));
    const z = groups.map(group => days.map(day => lookup.get(group+"/"+day) ?? null));
    traces.push({type: "heatmap", x: days, y: groups, z, name: core,
      xaxis: traces.length ? `x${traces.length+1}` : "x",
      yaxis: traces.length ? `y${traces.length+1}` : "y", coloraxis: "coloraxis"});
  }
  const layout = {...base, height: Math.max(230, traces.length*225),
    grid: {rows: traces.length, columns: 1, pattern: "independent"},
    margin: {l: 95, r: 65, t: 15, b: 65},
    coloraxis: {colorscale: "RdBu", cmid: 0, colorbar: {title: {text: "IG"}}}};
  for (const [index, trace] of traces.entries()) {
    layout[index ? `xaxis${index+1}` : "xaxis"] = {title: {text: `${trace.name} · ngày tương đối so với ngày phát hành`}};
  }
  if (traces.length) await Plotly.react($("explanation-plot"), traces, layout, plotOptions);
  else Plotly.purge($("explanation-plot"));
  table(content, rows.slice(0, 30), ["core", "group", "day_offset", "attribution"]);
}
function loadPage(next) {
  page = next;
  ++revision;
  for (const section of document.querySelectorAll(".page")) section.hidden = section.id !== page;
  for (const button of document.querySelectorAll("[data-page]")) {
    if (button.dataset.page === page) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  }
  const draw = {forecast: changedForecast, comparison: refreshComparison, flood: refreshFlood, explanation: refreshExplanation}[page];
  if (draw) run(draw);
  requestAnimationFrame(() => {
    for (const plot of $(page).querySelectorAll(".js-plotly-plot")) Plotly.Plots.resize(plot);
  });
}
async function run(action) {
  $("error").hidden = true;
  try { await action(); } catch (error) { $("error").textContent = error.message; $("error").hidden = false; }
}
async function start() {
  catalog = await api("/api/catalog");
  const leads = Array.from({length: Math.max(...catalog.models.flatMap(m => m.leads))}, (_, i) => [i+1, `${i+1} ngày`]);
  options($("lead"), leads, 1);
  for (const id of ["basin", "flood-basin"]) options($(id), catalog.ids.map(id => [id, `ID ${id}`]), catalog.ids[0]);
  const names = catalog.models.map(m => [m.run, m.name]);
  for (const id of ["comparison-run", "flood-model"]) options($(id), names, catalog.default_run);
  updateModelLeads("comparison-run", "comparison-lead");
  updateModelLeads("flood-model", "flood-lead");
  updateReferences();
  $("day").min = catalog.dates[0];
  $("day").max = catalog.dates.at(-1);
  $("day").value = catalog.dates[Math.floor(catalog.dates.length / 2)];
  $("live-day").min = $("day").min;
  $("live-day").max = $("day").max;
  $("live-day").value = $("day").value;
  chosen.add(catalog.default_run);
  const live = catalog.models.filter(m => m.what_if);
  options($("live-model"), live.map(m => [m.run, m.name]), catalog.default_run);
  $("run-live").disabled = !live.length;
  $("what-if-form").hidden = !live.length;
  $("what-if-status").textContent = live.length
    ? "Thay lượng mưa hoặc nhiệt độ ECMWF HRES tại ngày phát hành."
    : "Cần checkpoint và kết quả kiểm CPU để bật phần này.";
  const liveInputs = () => ({
    basin: $("basin").value, date: $("live-day").value, run: $("live-model").value,
    rain_multiplier: Number($("rain").value), temperature_delta: Number($("temp").value)
  });
  const clearLiveResult = () => $("what-if-results").replaceChildren();
  $("what-if-form").addEventListener("submit", event => {
    event.preventDefault();
    const button = $("run-live");
    button.disabled = true;
    clearLiveResult();
    const requested = liveInputs();
    const isCurrent = () => page === "forecast" && Object.entries(liveInputs())
      .every(([key, value]) => value === requested[key]);
    run(async () => {
      try {
        const result = await api("/api/what-if", {method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify(requested)});
        if (!isCurrent()) return;
        table($("what-if-results"), result.changed.map((row, i) => ({"Ngày đích": row.target_date, "Hạn": row.lead,
          "Gốc (m³/s)": result.baseline[i].qmax_m3_s, "Nếu… thì… (m³/s)": row.qmax_m3_s,
          "Chênh lệch (m³/s)": row.qmax_m3_s-result.baseline[i].qmax_m3_s})));
      } catch (error) {
        if (isCurrent()) throw error;
      } finally { button.disabled = false; }
    });
  });
  for (const id of ["basin", "day", "lead"]) $(id).addEventListener("change", changedForecast);
  $("day").addEventListener("change", () => { $("live-day").value = $("day").value; });
  for (const id of ["live-model", "live-day"]) $(id).addEventListener("change", clearLiveResult);
  for (const id of ["rain", "temp"]) $(id).addEventListener("input", clearLiveResult);
  $("comparison-run").addEventListener("change", () => {
    updateModelLeads("comparison-run", "comparison-lead");
    updateReferences();
    run(refreshComparison);
  });
  $("comparison-lead").addEventListener("change", () => { updateReferences(); run(refreshComparison); });
  $("comparison-ref").addEventListener("change", () => run(refreshComparison));
  $("flood-model").addEventListener("change", () => {
    updateModelLeads("flood-model", "flood-lead");
    run(refreshFlood);
  });
  for (const id of ["flood-basin", "flood-lead"]) $(id).addEventListener("change", () => run(refreshFlood));
  $("comparison-table").addEventListener("change", showComparisonTable);
  $("explanation-case").addEventListener("change", () => run(refreshExplanation));
  document.querySelectorAll("[data-page]").forEach(button => button.addEventListener("click", () => loadPage(button.dataset.page)));
  loadPage("forecast");
}
run(start);
