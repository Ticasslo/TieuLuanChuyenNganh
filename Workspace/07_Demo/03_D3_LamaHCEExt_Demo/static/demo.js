"use strict";

const $ = id => document.getElementById(id);
const color = ["#157a68", "#31569c", "#bb5627", "#8062ad", "#857023", "#376d7e", "#96565d", "#527a3a", "#676c72"];
const number = new Intl.NumberFormat("vi", {maximumFractionDigits: 4});
const dateText = input => input.split("-").reverse().join("/");
let catalog, page = "forecast", revision = 0, comparisonTables = {}, xaiTables = {};
const chosen = new Set();
const coreColor = {lstm: "#1f77b4", gru: "#ff7f0e", transformer: "#2ca02c", s4d: "#d62728", mamba: "#9467bd"};
const coreName = {lstm: "LSTM", gru: "GRU", transformer: "Transformer", s4d: "S4D", mamba: "Mamba"};
const hModeName = {gefs: "GEFS 7 ngày", hybrid: "ECMWF HRES ngày đầu (lai)", past: "Chỉ dữ liệu quá khứ", perfect: "Thời tiết tái phân tích (hoàn hảo)"};
const xaiTableNames = {
  GH_lead_summary: "H · Tỉ phần & L50/L90 theo hạn (TB 3 seed)",
  GH_strata_level: "H · Phân tầng theo 4 mức lưu lượng",
  GH_core_diffs: "H · Chênh lệch Mamba − LSTM (× nhiễu)",
  G_run_metrics: "C2 · Tỉ phần nhóm nguồn & L50/L90 (15 run)",
  G_strata: "C2 · Phân tầng theo mức lưu lượng và mùa",
  G_core_diffs: "C2 · Chênh lệch so với LSTM (× nhiễu)",
  G_probe_gain: "C2 · Dò trạng thái ẩn Y4 (phần vượt đối chứng)"
};

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

function map(target, scores = null, isDelta = false) {
  const byBasin = new Map(scores?.map(r => [r.basin_id, r.NSE]));
  const points = scores ? catalog.points.filter(p => Number.isFinite(byBasin.get(p.id))) : catalog.points;
  const graph = $(target);
  const selected = $("basin").value;
  let bound = 1;
  if (scores && isDelta && points.length) {
    const absVals = points.map(p => Math.abs(byBasin.get(p.id))).sort((a, b) => a - b);
    const p95 = absVals[Math.floor(0.95 * (absVals.length - 1))] || 0.2;
    bound = Math.min(1, Math.max(0.05, Number(p95.toPrecision(2))));
  }
  const labelMetric = isDelta ? "ΔNSE" : "NSE";
  const marker = scores
    ? {size: 8, color: points.map(p => byBasin.get(p.id)), colorscale: "RdBu", cmid: 0,
       cmin: -bound, cmax: bound, showscale: true}
    : {size: points.map(p => p.id === selected ? 10 : 5), color: points.map(p => p.id === selected ? "#c55226" : "#137b67")};
  const regions = (catalog.outline ?? []).map(polygon => ({type: "scatter", mode: "lines", fill: "toself",
    x: polygon.map(p => p[0]), y: polygon.map(p => p[1]), line: {color: "#b7cdbf", width: 1},
    fillcolor: "#edf4ec", hoverinfo: "skip", showlegend: false}));
  Plotly.react(graph, [...regions, {type: "scatter", mode: "markers", x: points.map(p => p.lon), y: points.map(p => p.lat),
    customdata: points.map(p => p.id), marker,
    text: points.map(p => scores ? `ID ${p.id} · ${labelMetric} ${number.format(byBasin.get(p.id))}` : `ID ${p.id}`),
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
  return bound;
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
  c2_scenarios: "C2 · Kịch bản S0–S4", D: "D · 365/730 bước", D_ensemble: "D · Tổ hợp 3 seed", D_cost: "D · Chi phí",
  D_length_pairs: "D · Kiểm định 730–365"};
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
  const reference = $("comparison-ref").value;
  const isDelta = Boolean(reference);
  const data = await api("/api/comparison" + query({run: $("comparison-run").value,
    reference, lead: $("comparison-lead").value}));
  if (page !== "comparison" || serial !== revision) return;
  const bound = map("nse-map", data.metrics, isDelta);
  $("nse-status").textContent = !data.metrics.length
    ? "Chưa có bảng NSE trên giao ngày chung cho lựa chọn này."
    : isDelta
      ? `Thang màu đối xứng ±${number.format(bound)} theo P95 |ΔNSE|; rê chuột xem giá trị thực từng lưu vực.`
      : "Thang màu cố định [-1; 1] cho NSE; rê chuột xem giá trị thực từng lưu vực (kể cả điểm âm lớn).";
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
function showXaiTable() {
  table($("explanation-content"), xaiTables[$("xai-table").value] ?? []);
}
function renderKpis(items) {
  const box = $("xai-kpis");
  box.replaceChildren(...items.map(([lbl, val]) => {
    const card = document.createElement("div");
    card.className = "kpi";
    const l = document.createElement("div");
    l.className = "kpi-label";
    l.textContent = lbl;
    const v = document.createElement("div");
    v.className = "kpi-val";
    v.textContent = val;
    card.append(l, v);
    return card;
  }));
  box.hidden = !items.length;
}
function pctText(v) {
  return Number.isFinite(v) ? `${number.format(v * 100)}%` : "—";
}
function updateXaiControls(scope, avail) {
  const scopes = [];
  if (avail.h_basin) scopes.push(["h", "H · Nhiều hạn 1–7 ngày (theo 451 lưu vực)"]);
  if (avail.c2_basin || avail.c2_summary) scopes.push(["c2", avail.c2_basin ? "C2 · 5 lõi kiến trúc (theo 451 lưu vực)" : "C2 · 5 lõi kiến trúc (tổng hợp + L90 lưu vực)"]);
  if (avail.legacy) scopes.push(["legacy", "Bảng IG chi tiết (CSV)"]);
  options($("xai-scope"), scopes, scope);
  $("xai-controls").hidden = !scopes.length || scope === "legacy";
  $("xai-mode-wrap").hidden = scope !== "h";
  $("xai-lead-wrap").hidden = scope !== "h";
  $("xai-core-wrap").hidden = !(scope === "c2" && avail.c2_basin);
}
async function refreshExplanation() {
  const serial = ++revision;
  const scopeVal = $("xai-scope").value;
  const basinVal = $("xai-basin").value || catalog.ids[0];
  const modeVal = $("xai-mode").value || "gefs";
  const leadVal = $("xai-lead").value || "7";
  const coreVal = $("xai-core").value || "lstm";
  const params = scopeVal === "legacy"
    ? (() => { const [b, o] = $("explanation-case").value.split("/"); return b && o ? {scope: "legacy", basin: b, origin: o} : {scope: "legacy"}; })()
    : {scope: scopeVal, basin: basinVal, mode: modeVal, lead: leadVal, core: coreVal};
  const data = await api("/api/explanation" + query(params));
  if (page !== "explanation" || serial !== revision) return;
  const avail = data.available ?? {};
  updateXaiControls(data.scope, avail);

  if (data.scope === "none") {
    $("xai-kpis").hidden = true;
    $("xai-charts").hidden = true;
    $("xai-table-wrap").hidden = true;
    $("explanation-picker").hidden = true;
    $("explanation-plot").hidden = true;
    $("xai-status").textContent = "Đang chờ kết quả Integrated Gradients của bước G.";
    $("explanation-content").replaceChildren();
    return;
  }

  if (data.scope === "legacy") {
    $("xai-kpis").hidden = true;
    $("xai-charts").hidden = true;
    $("xai-table-wrap").hidden = true;
    $("explanation-plot").hidden = false;
    $("explanation-picker").hidden = !data.cases.length;
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
    table($("explanation-content"), rows.slice(0, 30), ["core", "group", "day_offset", "attribution"]);
    return;
  }

  $("explanation-picker").hidden = true;
  $("explanation-plot").hidden = true;
  $("xai-charts").hidden = false;

  if (data.scope === "h") {
    const lstm = data.cores.lstm, mamba = data.cores.mamba;
    const li = data.lead - 1;
    $("xai-status").textContent = `Lưu vực ID ${data.basin} · ${hModeName[data.mode] ?? data.mode} · Hạn ${data.lead} ngày · ${data.n_samples_by_lead[li]} mẫu đạt completeness (tổng ${data.n_samples_all} lần phát dự báo test).`;
    renderKpis([
      [`Pha dự báo (hạn ${data.lead})`, `LSTM ${pctText(lstm?.phase_by_lead.fore[li])} · Mamba ${pctText(mamba?.phase_by_lead.fore[li])}`],
      [`Pha quá khứ (hạn ${data.lead})`, `LSTM ${pctText(lstm?.phase_by_lead.past[li])} · Mamba ${pctText(mamba?.phase_by_lead.past[li])}`],
      [`Độ dài nhớ L90 (hạn ${data.lead})`, `LSTM ${lstm?.memory_L90[li] == null ? "—" : number.format(lstm.memory_L90[li]) + " ngày"} · Mamba ${mamba?.memory_L90[li] == null ? "—" : number.format(mamba.memory_L90[li]) + " ngày"}`],
      ["Run đại diện (chọn theo validation)", `${lstm?.run ?? "—"} | ${mamba?.run ?? "—"}`]
    ]);

    // Chart 1: Phase share (future & past) across leads 1..7
    const traces1 = [];
    for (const [ck, obj] of [["lstm", lstm], ["mamba", mamba]]) {
      if (!obj) continue;
      traces1.push({type: "scatter", mode: "lines+markers", x: data.leads,
        y: obj.phase_by_lead.fore.map(v => v == null ? null : v * 100),
        name: `${coreName[ck]} · Dự báo tương lai`, line: {color: coreColor[ck], width: 2.4}});
      traces1.push({type: "scatter", mode: "lines+markers", x: data.leads,
        y: obj.phase_by_lead.past.map(v => v == null ? null : v * 100),
        name: `${coreName[ck]} · Quá khứ 364 ngày`, line: {color: coreColor[ck], width: 1.6, dash: "dot"}});
    }
    await Plotly.react($("xai-plot-1"), traces1, {...base,
      title: {text: `Tỉ phần |IG| theo hạn 1–7 ngày (ID ${data.basin})`, font: {size: 13}},
      margin: {l: 52, r: 16, t: 38, b: 68}, xaxis: {title: "Hạn dự báo (ngày)", dtick: 1},
      yaxis: {title: "Tỉ phần |IG| (%)", rangemode: "tozero"}, legend: {orientation: "h", y: -0.26}}, plotOptions);

    // Chart 2: Group share at selected lead
    const gNames = data.groups.filter(g => (lstm?.group_share_lead[g] ?? 0) > 1e-4 || (mamba?.group_share_lead[g] ?? 0) > 1e-4);
    const traces2 = [["lstm", lstm], ["mamba", mamba]].filter(([, o]) => o).map(([ck, obj]) => ({
      type: "bar", orientation: "h", y: gNames, x: gNames.map(g => (obj.group_share_lead[g] ?? 0) * 100),
      name: coreName[ck], marker: {color: coreColor[ck]}
    }));
    await Plotly.react($("xai-plot-2"), traces2, {...base, barmode: "group",
      title: {text: `Đóng góp theo nhóm nguồn tại hạn ${data.lead}`, font: {size: 13}},
      margin: {l: 122, r: 16, t: 38, b: 50}, xaxis: {title: "Tỉ phần |IG| (%)", rangemode: "tozero"},
      yaxis: {autorange: "reversed"}, legend: {orientation: "h", y: -0.22}}, plotOptions);

    // Chart 3: Forecast-day attribution heatmap (LSTM vs Mamba) OR L50/L90 memory by lead when mode === "past"
    if (data.mode !== "past" && lstm && mamba) {
      const daysX = data.leads.map(d => `Ngày +${d}`);
      const leadsY = data.leads.map(k => `Hạn ${k}`);
      const zL = lstm.fore_day_share.map(row => row.map(v => v == null ? null : v * 100));
      const zM = mamba.fore_day_share.map(row => row.map(v => v == null ? null : v * 100));
      await Plotly.react($("xai-plot-3"), [
        {type: "heatmap", x: daysX, y: leadsY, z: zL, xaxis: "x", yaxis: "y", coloraxis: "coloraxis", name: "LSTM"},
        {type: "heatmap", x: daysX, y: leadsY, z: zM, xaxis: "x2", yaxis: "y2", coloraxis: "coloraxis", name: "Mamba"}
      ], {...base,
        title: {text: `Tỉ phần |IG| từng ngày dự báo (1–7) ở mỗi hạn — trái: LSTM, phải: Mamba (ID ${data.basin})`, font: {size: 13}},
        grid: {rows: 1, columns: 2, pattern: "independent"}, margin: {l: 64, r: 72, t: 42, b: 48},
        xaxis: {title: "LSTM · ngày trong pha dự báo"}, xaxis2: {title: "Mamba · ngày trong pha dự báo"},
        yaxis: {autorange: "reversed"}, yaxis2: {autorange: "reversed"},
        coloraxis: {colorscale: "YlGnBu", colorbar: {title: {text: "% |IG|"}}}}, plotOptions);
    } else {
      const traces3 = [];
      for (const [ck, obj] of [["lstm", lstm], ["mamba", mamba]]) {
        if (!obj) continue;
        traces3.push({type: "scatter", mode: "lines+markers", x: data.leads, y: obj.memory_L90,
          name: `${coreName[ck]} · L90`, line: {color: coreColor[ck], width: 2.3}});
        traces3.push({type: "scatter", mode: "lines+markers", x: data.leads, y: obj.memory_L50,
          name: `${coreName[ck]} · L50`, line: {color: coreColor[ck], width: 1.5, dash: "dash"}});
      }
      await Plotly.react($("xai-plot-3"), traces3, {...base,
        title: {text: `Độ dài nhớ hiệu dụng L50 & L90 theo hạn (ID ${data.basin})`, font: {size: 13}},
        margin: {l: 58, r: 16, t: 38, b: 56}, xaxis: {title: "Hạn dự báo (ngày)", dtick: 1},
        yaxis: {title: "Số ngày quá khứ", rangemode: "tozero"}, legend: {orientation: "h", y: -0.24}}, plotOptions);
    }
  } else if (data.scope === "c2") {
    const byCore = new Map(data.memory.map(r => [r.core, r]));
    $("xai-status").textContent = data.c2_basin
      ? `C2 (hạn 1 ngày) tại lưu vực ID ${data.basin} · dữ liệu trực tiếp theo lưu vực từ G_demo_basin.npz (${byCore.get("lstm")?.n_samples ?? 0} mẫu đạt completeness).`
      : `C2 (hạn 1 ngày) · hiển thị tổng hợp toàn miền 10.511 mẫu và L90 riêng của lưu vực ID ${data.basin} (khi bổ sung G_demo_basin.npz sẽ tự mở thêm bản đồ nhiệt từng lưu vực).`;
    const l90Text = c => byCore.get(c)?.L90 == null ? "—" : number.format(byCore.get(c).L90);
    renderKpis([
      [`L90 lưu vực ID ${data.basin} (LSTM / Mamba)`, `${l90Text("lstm")} / ${l90Text("mamba")} ngày`],
      [`L90 lưu vực ID ${data.basin} (S4D / Transformer)`, `${l90Text("s4d")} / ${l90Text("transformer")} ngày`],
      ["Nhóm lớn nhất (LSTM)", `Q ${pctText(data.group_share.lstm?.Q)} · Tĩnh ${pctText(data.group_share.lstm?.static)}`],
      ["Dò trạng thái ẩn Y4", "Chỉ ô nhớ c LSTM vượt đối chứng mưa 30–180d"]
    ]);

    // Chart 1: 7 source groups across 5 cores
    const traces1 = data.cores_order.filter(c => data.group_share[c]).map(c => ({
      type: "bar", x: data.groups, y: data.groups.map(g => (data.group_share[c][g] ?? 0) * 100),
      name: coreName[c], marker: {color: coreColor[c]}
    }));
    await Plotly.react($("xai-plot-1"), traces1, {...base, barmode: "group",
      title: {text: data.c2_basin ? `Tỉ phần 7 nhóm nguồn tại ID ${data.basin}` : "Tỉ phần 7 nhóm nguồn (TB 3 seed, 10.511 mẫu)", font: {size: 13}},
      margin: {l: 52, r: 16, t: 38, b: 64}, yaxis: {title: "Tỉ phần |IG| (%)", rangemode: "tozero"},
      legend: {orientation: "h", y: -0.24}}, plotOptions);

    // Chart 2: Cumulative lag share 0..60 days across 5 cores
    const lags60 = Array.from({length: 61}, (_, i) => i);
    const traces2 = data.cores_order.filter(c => data.lag_cum_60[c]).map(c => ({
      type: "scatter", mode: "lines", x: lags60, y: data.lag_cum_60[c].map(v => v == null ? null : v * 100),
      name: coreName[c], line: {color: coreColor[c], width: 2.2}
    }));
    await Plotly.react($("xai-plot-2"), traces2, {...base,
      title: {text: "Tích lũy |IG| theo độ trễ 0–60 ngày (5 lõi C2)", font: {size: 13}},
      margin: {l: 54, r: 16, t: 38, b: 64}, xaxis: {title: "Ngày trễ (0 = ngày đích t)"},
      yaxis: {title: "Tỉ phần tích lũy (%)", range: [0, 102]},
      shapes: [{type: "line", x0: 0, x1: 60, y0: 90, y1: 90, line: {color: "#809a8b", dash: "dash", width: 1}}],
      legend: {orientation: "h", y: -0.24}}, plotOptions);

    // Chart 3: Per-basin heatmap (when G_C2_demo.npz exists) OR per-basin L90 comparison vs global
    if (data.heatmap) {
      await Plotly.react($("xai-plot-3"), [{
        type: "heatmap", x: data.heatmap.lags, y: data.heatmap.source_groups, z: data.heatmap.z,
        colorscale: "YlGnBu", colorbar: {title: {text: "|IG|"}}
      }], {...base,
        title: {text: `Bản đồ nhiệt |IG| (ngày trễ 0–30 × nhóm nguồn) · ${coreName[data.heatmap.core]} · ID ${data.basin}`, font: {size: 13}},
        margin: {l: 100, r: 64, t: 38, b: 48}, xaxis: {title: "Ngày trễ trước ngày đích t"}}, plotOptions);
    } else {
      const memCores = data.memory.map(r => coreName[r.core]);
      await Plotly.react($("xai-plot-3"), [
        {type: "bar", x: memCores, y: data.memory.map(r => r.L90), name: `L90 tại lưu vực ID ${data.basin}`,
          marker: {color: data.memory.map(r => coreColor[r.core])}},
        {type: "scatter", mode: "markers", x: memCores, y: data.memory.map(r => r.L90_global), name: "Trung vị toàn 451 lưu vực",
          marker: {color: "#172e29", symbol: "diamond", size: 10}}
      ], {...base,
        title: {text: `Độ dài nhớ hiệu dụng L90 tại lưu vực ID ${data.basin} so với toàn miền`, font: {size: 13}},
        margin: {l: 58, r: 16, t: 38, b: 56}, yaxis: {title: "Số ngày quá khứ giữ 90% |IG|", rangemode: "tozero"},
        legend: {orientation: "h", y: -0.24}}, plotOptions);
    }
  }

  xaiTables = data.tables ?? {};
  const tKeys = Object.keys(xaiTables);
  $("xai-table-wrap").hidden = !tKeys.length;
  if (tKeys.length) {
    options($("xai-table"), tKeys.map(k => [k, xaiTableNames[k] ?? k]), $("xai-table").value);
    showXaiTable();
  } else {
    $("explanation-content").replaceChildren();
  }
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
  for (const id of ["basin", "flood-basin", "xai-basin"]) options($(id), catalog.ids.map(id => [id, `ID ${id}`]), catalog.ids[0]);
  options($("xai-mode"), Object.entries(hModeName), "gefs");
  options($("xai-lead"), Array.from({length: 7}, (_, i) => [i + 1, `${i + 1} ngày`]), 7);
  options($("xai-core"), ["lstm", "gru", "mamba", "s4d", "transformer"].map(c => [c, coreName[c]]), "lstm");
  const names = catalog.models.map(m => [m.run, m.name]);
  for (const id of ["comparison-run", "flood-model"]) options($(id), names, catalog.default_run);
  updateModelLeads("comparison-run", "comparison-lead");
  updateModelLeads("flood-model", "flood-lead");
  updateReferences();
  $("day").min = catalog.dates[0];
  $("day").max = catalog.dates.at(-1);
  $("day").value = catalog.dates[Math.floor(catalog.dates.length / 2)];
  chosen.add(catalog.default_run);
  const live = catalog.models.filter(m => m.what_if);
  options($("live-model"), live.map(m => [m.run, m.name]), catalog.default_run);
  $("run-live").disabled = !live.length;
  $("what-if-form").hidden = !live.length;
  const syncLiveDate = (date = $("live-day").value) => {
    const selected = live.find(model => model.run === $("live-model").value);
    const day = $("live-day");
    day.min = selected?.origin_start ?? $("day").min;
    day.max = selected?.origin_end ?? $("day").max;
    day.value = date < day.min ? day.min : date > day.max ? day.max : date;
    $("what-if-status").textContent = selected?.stage === "H"
      ? "Thay ECMWF HRES tại ngày phát hành và xem lưu lượng của cả bảy hạn."
      : selected ? "Thay lượng mưa hoặc nhiệt độ ECMWF HRES tại ngày phát hành."
                 : "Cần checkpoint và kết quả kiểm CPU để bật phần này.";
  };
  syncLiveDate($("day").value);
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
  $("day").addEventListener("change", () => syncLiveDate($("day").value));
  $("live-model").addEventListener("change", () => { syncLiveDate(); clearLiveResult(); });
  $("live-day").addEventListener("change", clearLiveResult);
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
  $("xai-table").addEventListener("change", showXaiTable);
  for (const id of ["xai-scope", "xai-basin", "xai-mode", "xai-lead", "xai-core", "explanation-case"]) {
    $(id).addEventListener("change", () => run(refreshExplanation));
  }
  document.querySelectorAll("[data-page]").forEach(button => button.addEventListener("click", () => loadPage(button.dataset.page)));
  loadPage("forecast");
}
run(start);
