import argparse
import atexit
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request
from werkzeug.exceptions import HTTPException


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "DemoRuntime"))
from forecast import DemoForecaster, PRED_ATOL, PRED_RTOL, qmax_m3_s
from h_cpu import H_PRED_ATOL, HWorker, HWorkerError


def columns_for_view(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def create_app(bundle, raw=None, h_python=None):
    bundle = Path(bundle)
    catalog = json.loads((bundle/"catalog.json").read_text(encoding="utf-8"))
    if catalog.get("schema") != 1:
        raise ValueError("Phiên bản gói demo không được hỗ trợ.")
    models = {m["run"]: m for m in catalog["models"]}
    ids = {bid: i for i, bid in enumerate(catalog["ids"])}
    date_index = {day: i for i, day in enumerate(catalog["dates"])}
    if len(models) != len(catalog["models"]) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in models):
        raise ValueError("ID mô hình trùng hoặc ngoài định dạng.")
    app = Flask(__name__, static_folder=str(HERE/"static"))
    live = {run for run, model in models.items() if raw is not None and model.get("what_if")
            and (model["stage"] != "H" or h_python is not None)}
    h_models = [m for m in catalog["models"] if m["run"] in live and m["stage"] == "H"]
    h_worker = None
    if h_models:
        if not Path(h_python).is_file():
            raise ValueError(f"Không tìm thấy Python của H: {h_python}.")
        h_worker = HWorker(h_python, bundle, raw, ids, (catalog["areas"][i] for i in ids), h_models)
        atexit.register(h_worker.close)
    if any(models[run]["stage"] != "H" for run in live):
        from importlib.metadata import distribution
        import torch

        version = distribution("neuralhydrology").read_text("direct_url.json")
        installed = json.loads(version or "{}").get("vcs_info", {}).get("commit_id")
        if installed != catalog["library_commit"]:
            raise ValueError("Môi trường chạy trực tiếp cần đúng commit đã chấm qua D2.")
        threads = max(1, int(os.environ.get("DEMO_THREADS", min(4, os.cpu_count() or 1))))
        torch.set_num_threads(threads)

    @app.errorhandler(HTTPException)
    def api_error(exc):
        if request.path.startswith("/api/"):
            return jsonify(error=exc.description), exc.code
        return exc

    @lru_cache(maxsize=2)
    def predictor(run):
        return DemoForecaster(bundle/"checkpoints"/run, raw, tuple(ids),
                              np.array([catalog["areas"][i] for i in ids]), None, device="cpu")

    @lru_cache(maxsize=96)
    def matrix(path):
        result = np.load(bundle/path, mmap_mode="r", allow_pickle=False)
        if result.shape != (len(ids), len(date_index)):
            raise ValueError(f"Sai kích thước dữ liệu phát lại: {path}")
        return result

    @lru_cache(maxsize=64)
    def table(filename):
        path = bundle/f"{filename}.csv"
        return pd.read_csv(path, dtype={"basin_id": str, "origin_date": str, "basin": str}) if path.is_file() else None

    @lru_cache(maxsize=8)
    def npz_data(filename):
        path = bundle/filename
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as archive:
            return {k: np.array(archive[k]) for k in archive.files}

    def finite_or_none(value):
        v = float(value)
        return v if np.isfinite(v) else None

    def integer_arg(name, default, end):
        try:
            value = int(request.args.get(name, default))
        except ValueError:
            abort(400, f"Giá trị {name} phải là số nguyên.")
        if not 1 <= value <= end:
            abort(400, f"Giá trị {name} phải từ 1 đến {end}.")
        return value

    def points(arr):
        return [float(x) if np.isfinite(x) else None for x in arr]

    def uses_h_tables(run, lead):
        return models.get(run, {}).get("stage") == "H" or (run == "Persistence_qmax" and lead > 1)

    @app.get("/")
    def home():
        return app.send_static_file("index.html")

    @app.get("/plotly.js")
    def plotly_script():
        from plotly.offline import get_plotlyjs

        return app.response_class(get_plotlyjs(), mimetype="text/javascript")

    @app.get("/api/catalog")
    def get_catalog():
        data = dict(catalog)
        data.pop("flood_m3_s")
        data.pop("areas")
        data["models"] = [dict(m, what_if=m["run"] in live) for m in models.values()]
        return jsonify(data)

    @app.get("/api/forecast")
    def forecast():
        basin = request.args.get("basin")
        date = request.args.get("date")
        if basin not in ids or date not in date_index:
            abort(400, "Chọn lưu vực và ngày trong kỳ test.")
        lead = integer_arg("lead", 1, 7)
        run_names = request.args.getlist("run")
        if any(run not in models for run in run_names):
            abort(400, "Run không có trong catalog.")
        center = date_index[date]
        selection = slice(max(center-30,0), min(center+31,len(date_index)))
        obs = matrix("observed.npy")[ids[basin], selection]
        lines = [{"run": run, "values": points(matrix(f"series/{run}_lead{lead}.npy")[ids[basin], selection])}
                 for run in dict.fromkeys(run_names) if lead in models[run]["leads"]]
        return jsonify(dates=catalog["dates"][selection], observed=points(obs), predictions=lines,
                       thresholds=catalog["flood_m3_s"][basin])

    @app.post("/api/what-if")
    def what_if():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            abort(400, "Thiếu dữ liệu cho lần suy luận.")
        run = body.get("run")
        if not isinstance(run, str) or run not in live:
            abort(400, "Chưa có bản CPU đã đối chiếu cho cấu hình này.")
        basin, date = body.get("basin"), body.get("date")
        if not isinstance(basin, str) or basin not in ids or not isinstance(date, str) or date not in date_index:
            abort(400, "Chọn lưu vực và ngày phát hành trong kỳ test.")
        meta = models[run]
        try:
            rain = float(body.get("rain_multiplier", 1))
            temp = float(body.get("temperature_delta", 0))
            if not np.isfinite(rain) or rain < 0 or not np.isfinite(temp):
                raise ValueError("Hệ số mưa cần hữu hạn và không âm. Mức cộng nhiệt cần hữu hạn.")
            if meta["stage"] == "H":
                if not meta["origin_start"] <= date <= meta["origin_end"]:
                    raise ValueError(f"Chọn ngày phát hành H từ {meta['origin_start']} đến {meta['origin_end']}.")
                series = h_worker.predict(run, basin, date, rain, temp)
                baseline, changed = series["baseline"], series["changed"]
            else:
                model = predictor(run)
                baseline = model.predict(basin, date)
                changed = (baseline if (rain, temp) == (1, 0) else
                           model.predict(basin, date, rain_multiplier=rain, temperature_delta=temp))
            expected_steps = [(lead, catalog["dates"][date_index[date]+lead-1]) for lead in meta["leads"]]
            if any([(row["lead"], row["target_date"]) for row in values] != expected_steps
                   for values in (baseline, changed)):
                raise ValueError("Lịch suy luận CPU khác ngày phát hành và hạn đã chọn.")
            basin_index = ids[basin]
            tolerance = H_PRED_ATOL if meta["stage"] == "H" else PRED_ATOL
            atol = qmax_m3_s(tolerance, catalog["areas"][basin])
            for item in baseline:
                target_index = date_index[item["target_date"]]
                expected = matrix(f"series/{run}_lead{item['lead']}.npy")[basin_index, target_index]
                if not np.isclose(item["qmax_m3_s"], expected, atol=atol, rtol=PRED_RTOL):
                    raise ValueError("Dự báo gốc CPU khác tệp phát lại tại cửa sổ đã chọn.")
        except HWorkerError as exc:
            return jsonify(error=str(exc)), 503
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            return jsonify(error=str(exc)), 422
        return jsonify(baseline=baseline, changed=changed)

    @app.get("/api/comparison")
    def comparison():
        run, reference = request.args.get("run"), request.args.get("reference")
        lead = integer_arg("lead", 1, 7)
        scope = models.get(run, {}).get("stage")
        suffix = f"H_metrics_lead{lead}" if uses_h_tables(run, lead) else "metrics"
        frame = table(suffix)
        metrics = []
        if scope == "D" and lead == 1:
            frame = table("D_basin")
            if frame is not None:
                model = models[run]
                selected = frame[frame.configuration.eq(model["configuration"]) & frame.seed.eq(model["seed"])]
                if not reference:
                    metrics = columns_for_view(selected[["basin_id", "NSE_730"]].rename(columns={"NSE_730":"NSE"}))
                elif reference == "C2_same_seed":
                    metrics = columns_for_view(selected[["basin_id", "delta_NSE"]].rename(columns={"delta_NSE":"NSE"}))
        elif frame is not None and run in models:
            data = frame[(frame.scope.eq("common_source_all")) & frame.scenario.eq("S0") & frame.lead.eq(lead)]
            a = data[data.run.eq(run)].set_index("basin_id")["NSE"]
            if reference:
                b = data[data.run.eq(reference)].set_index("basin_id")["NSE"]
                a = a.subtract(b) if len(b) else a.iloc[0:0]
            metrics = columns_for_view(a.rename("NSE").reset_index())
        names = ("C2_config_table", "C2_cost_table", "D2_CPU_benchmark", "c2_scenarios", "D", "D_ensemble", "D_cost",
                 "D_length_pairs", *[f"H_summary_lead{k}" for k in range(1, 8)])
        tables = {}
        for key in names:
            values = table(key)
            if values is not None:
                tables[key] = columns_for_view(values)
        return jsonify(metrics=metrics, tables=tables)

    @app.get("/api/flood")
    def flood():
        run, basin = request.args.get("run"), request.args.get("basin")
        lead = integer_arg("lead", 1, 7)
        if run not in models or basin not in ids:
            abort(400, "Chọn run và lưu vực.")
        is_d = models[run].get("stage") == "D"
        if is_d and lead != 1:
            return jsonify(events=[], levels=[], event_cases=[])
        result = {}
        for name in ("events", "levels", "event_cases"):
            key = (f"H_{name}_lead{lead}" if uses_h_tables(run, lead)
                   else f"D_{name}" if is_d
                   else ("events_by_case" if name == "event_cases" else name))
            frame = table(key)
            if frame is None:
                continue
            rows = frame[frame.run.eq(run) & frame.basin_id.eq(basin) & frame.lead.eq(lead)
                         & frame.scenario.eq("S0") & frame.scope.eq("common_source_all")]
            if "grouping" in rows:
                rows = rows[rows.grouping.eq("USWRC")]
            result[name] = columns_for_view(rows)
        return jsonify(result)

    @app.get("/api/explanation")
    def explanation():
        gh_npz = npz_data("G_H_demo.npz")
        gc2_npz = npz_data("G_C2_demo.npz")
        g_metrics = table("G_run_metrics")
        g_legacy = table("G")
        available = dict(
            h_basin=gh_npz is not None,
            c2_basin=gc2_npz is not None,
            c2_summary=g_metrics is not None,
            legacy=g_legacy is not None,
        )
        req_scope = request.args.get("scope", "").strip().lower()
        if not req_scope:
            if available["h_basin"]:
                req_scope = "h"
            elif available["c2_basin"] or available["c2_summary"]:
                req_scope = "c2"
            elif available["legacy"]:
                req_scope = "legacy"
            else:
                return jsonify(scope="none", available=available, cases=[], rows=[])
        basin = request.args.get("basin") or catalog["ids"][0]
        if basin not in ids:
            abort(400, "Chọn lưu vực trong danh sách 451 lưu vực.")
        b_idx = ids[basin]

        if req_scope == "h":
            if gh_npz is None:
                return jsonify(scope="h", available=available, error="Gói demo chưa có G_H_demo.npz."), 404
            mode = request.args.get("mode", "gefs").strip().lower()
            valid_modes = ("gefs", "hybrid", "past", "perfect")
            if mode not in valid_modes:
                abort(400, "Cách cấp thời tiết phải thuộc gefs, hybrid, past hoặc perfect.")
            lead = integer_arg("lead", 7, 7)
            runs_list = [str(x) for x in gh_npz["runs"]]
            run_idx = {r: i for i, r in enumerate(runs_list)}
            demo_runs = [str(x) for x in gh_npz["demo_runs"]]
            phases = [str(x) for x in gh_npz["phases"]]
            groups = [str(x) for x in gh_npz["groups"]]
            cores_out = {}
            for core in ("lstm", "mamba"):
                match = next((r for r in demo_runs if r.startswith(f"{core}_{mode}_seed")), None)
                if match is None:
                    match = next((r for r in runs_list if r.startswith(f"{core}_{mode}_seed")), None)
                if match is None:
                    continue
                ri = run_idx[match]
                p_mat = gh_npz["phase_share"][ri, b_idx]
                g_vec = gh_npz["group_share"][ri, b_idx, lead - 1]
                f_mat = gh_npz["fore_day_share"][ri, b_idx]
                cores_out[core] = dict(
                    run=match,
                    phase_by_lead={ph: [finite_or_none(p_mat[k, pi]) for k in range(7)] for pi, ph in enumerate(phases)},
                    memory_L50=[finite_or_none(x) for x in gh_npz["memory_L50"][ri, b_idx]],
                    memory_L90=[finite_or_none(x) for x in gh_npz["memory_L90"][ri, b_idx]],
                    group_share_lead={g: finite_or_none(g_vec[gi]) for gi, g in enumerate(groups)},
                    fore_day_share=[[finite_or_none(f_mat[k, d]) for d in range(7)] for k in range(7)],
                )
            tables = {}
            for tname, cols in (
                ("GH_lead_summary", ["mode", "core", "lead", "share past mean", "share fore mean",
                                     "share static mean", "share fore:ECMWF mean", "share fore:GEFS mean",
                                     "L50 median mean", "L90 median mean"]),
                ("GH_strata_level", None),
                ("GH_core_diffs", None),
            ):
                df = table(tname)
                if df is not None:
                    keep = [c for c in cols if c in df] if cols else list(df.columns)
                    tables[tname] = columns_for_view(df[keep])
            return jsonify(
                scope="h", available=available, basin=basin, mode=mode, lead=lead,
                modes=list(valid_modes), leads=list(range(1, 8)), phases=phases, groups=groups,
                n_samples_by_lead=[int(x) for x in gh_npz["n_samples"][b_idx]],
                n_samples_all=int(gh_npz["n_samples_all"][b_idx]),
                cores=cores_out, tables=tables,
            )

        if req_scope == "c2":
            if gc2_npz is None and g_metrics is None:
                return jsonify(scope="c2", available=available, error="Gói demo chưa có dữ liệu giải thích C2."), 404
            cores_order = ["lstm", "gru", "mamba", "s4d", "transformer"]
            req_core = request.args.get("core", "lstm").strip().lower()
            if req_core not in cores_order:
                abort(400, "Lõi phải thuộc lstm, gru, mamba, s4d hoặc transformer.")
            groups = ["ERA5L", "EOBS", "MSWEP", "GLEAM", "Q", "ECMWF", "static"]
            group_share, lag_cum_60, memory, heatmap = {}, {}, [], None
            if gc2_npz is not None:
                runs_list = [str(x) for x in gc2_npz["runs"]]
                run_idx = {r: i for i, r in enumerate(runs_list)}
                demo_runs = [str(x) for x in gc2_npz["demo_runs"]]
                groups = [str(x) for x in gc2_npz["groups"]]
                src_groups = [str(x) for x in gc2_npz["source_groups"]]
                n_samp = int(gc2_npz["n_samples"][b_idx])
                for core in cores_order:
                    match = next((r for r in demo_runs if r.startswith(f"{core}_")), None)
                    if match is None:
                        continue
                    ri = run_idx[match]
                    g_vec = gc2_npz["group_share"][ri, b_idx]
                    group_share[core] = {g: finite_or_none(g_vec[gi]) for gi, g in enumerate(groups)}
                    l_vec = gc2_npz["lag_share"][ri, b_idx]
                    cum = np.cumsum(np.nan_to_num(l_vec, nan=0.0))[:61] if np.isfinite(l_vec).any() else np.full(61, np.nan)
                    lag_cum_60[core] = [finite_or_none(v) for v in cum]
                    memory.append(dict(core=core, run=match,
                                       L50=finite_or_none(gc2_npz["memory_L50"][ri, b_idx]),
                                       L90=finite_or_none(gc2_npz["memory_L90"][ri, b_idx]),
                                       n_samples=n_samp))
                    if core == req_core:
                        gl = gc2_npz["group_lag_0_60"][ri, b_idx, :31, :]
                        heatmap = dict(core=core, run=match, lags=list(range(31)), source_groups=src_groups,
                                       z=[[finite_or_none(gl[lg, si]) for lg in range(31)] for si in range(len(src_groups))])
            else:
                # Dự phòng từ bảng tổng hợp C2 XAI (trong repo) khi chưa tải tệp G_demo_basin.npz
                basin_l90_df = table("G_basin_L90")
                b_l90_row = None
                if basin_l90_df is not None:
                    hit = basin_l90_df[basin_l90_df["basin"].astype(str).eq(basin)]
                    b_l90_row = hit.iloc[0] if not hit.empty else None
                lag_npz = npz_data("G_lag_share.npz")
                for core in cores_order:
                    sub = g_metrics[g_metrics["core"].eq(core)]
                    if sub.empty:
                        continue
                    group_share[core] = {g: finite_or_none(sub[f"share {g}"].mean()) for g in groups if f"share {g}" in sub}
                    if lag_npz is not None:
                        arrs = [lag_npz[k] for k in lag_npz if k.startswith(f"{core}_")]
                        if arrs:
                            cum = np.cumsum(np.mean(arrs, axis=0))[:61]
                            lag_cum_60[core] = [finite_or_none(v) for v in cum]
                    memory.append(dict(
                        core=core, run=f"{core} (TB 3 seed)",
                        L50=finite_or_none(sub["L50 median"].mean()),
                        L90=finite_or_none(b_l90_row[core]) if b_l90_row is not None and core in b_l90_row else finite_or_none(sub["L90 median"].mean()),
                        L90_global=finite_or_none(sub["L90 median"].mean()),
                        n_samples=None,
                    ))
            tables = {}
            for tname, cols in (
                ("G_run_metrics", ["core", "seed", "L50 median", "L90 median", "share ERA5L", "share EOBS",
                                   "share MSWEP", "share GLEAM", "share Q", "share ECMWF", "share static"]),
                ("G_strata", None),
                ("G_core_diffs", None),
                ("G_probe_gain", None),
            ):
                df = table(tname)
                if df is not None:
                    keep = [c for c in cols if c in df] if cols else list(df.columns)
                    tables[tname] = columns_for_view(df[keep])
            return jsonify(
                scope="c2", available=available, basin=basin, core=req_core, cores_order=cores_order,
                c2_basin=gc2_npz is not None, groups=groups, group_share=group_share,
                lag_cum_60=lag_cum_60, memory=memory, heatmap=heatmap, tables=tables,
            )

        if g_legacy is None:
            return jsonify(scope="none", available=available, cases=[], rows=[])
        keys = ["basin_id", "origin_date"]
        required = {*keys, "core", "group", "day_offset", "attribution"}
        if not required <= set(g_legacy):
            return jsonify(error="Bảng Integrated Gradients chưa khớp các cột đã bàn giao."), 422
        cases = g_legacy[keys].drop_duplicates()
        origin = request.args.get("origin")
        if request.args.get("basin") is None or origin is None:
            first = cases.iloc[0] if not cases.empty else None
            basin, origin = (first.basin_id, first.origin_date) if first is not None else (None, None)
        selected = g_legacy[g_legacy.basin_id.eq(basin) & g_legacy.origin_date.eq(origin)]
        return jsonify(scope="legacy", available=available, cases=columns_for_view(cases),
                       rows=columns_for_view(selected), basin=basin, origin=origin)

    @app.get("/api/health")
    def health():
        return jsonify(ok=True, schema=catalog["schema"], models=len(models), live_what_if=len(live),
                       xai_h=npz_data("G_H_demo.npz") is not None,
                       xai_c2_basin=npz_data("G_C2_demo.npz") is not None,
                       xai_c2_summary=table("G_run_metrics") is not None)

    return app


def from_env():
    bundle = os.environ.get("DEMO_BUNDLE")
    if not bundle:
        raise RuntimeError("Đặt biến môi trường DEMO_BUNDLE trỏ tới thư mục gói dữ liệu demo.")
    raw = os.environ.get("DEMO_RAW")
    h_python = os.environ.get("DEMO_H_PYTHON")
    return create_app(Path(bundle), Path(raw) if raw else None, Path(h_python) if h_python else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy bản demo trên laptop hoặc máy chủ CPU.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="Thư mục LamaH_extended đã kiểm nguồn để bật nếu…thì…")
    parser.add_argument("--h-python", type=Path, help="Python riêng cài freeze-H để bật suy luận H cách lai.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    create_app(args.bundle, args.raw, args.h_python).run(host=args.host, port=args.port)
