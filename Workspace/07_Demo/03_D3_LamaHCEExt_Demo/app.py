import argparse
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


def columns_for_view(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def create_app(bundle, raw=None):
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
    if raw is not None and any(m.get("what_if") for m in models.values()):
        from importlib.metadata import distribution
        version = distribution("neuralhydrology").read_text("direct_url.json")
        installed = json.loads(version or "{}").get("vcs_info", {}).get("commit_id")
        if installed != catalog["library_commit"]:
            raise ValueError("Môi trường chạy trực tiếp cần đúng commit đã chấm qua D2.")

    @app.errorhandler(HTTPException)
    def api_error(exc):
        if request.path.startswith("/api/"):
            return jsonify(error=exc.description), exc.code
        return exc

    @lru_cache(maxsize=2)
    def predictor(run):
        import torch

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        return DemoForecaster(bundle/"checkpoints"/run, raw, tuple(ids),
                              np.array([catalog["areas"][i] for i in ids]), None, device="cpu")

    @lru_cache(maxsize=72)
    def matrix(path):
        result = np.load(bundle/path, mmap_mode="r", allow_pickle=False)
        if result.shape != (len(ids), len(date_index)):
            raise ValueError(f"Sai kích thước dữ liệu phát lại: {path}")
        return result

    @lru_cache(maxsize=24)
    def table(filename):
        path = bundle/f"{filename}.csv"
        return pd.read_csv(path, dtype={"basin_id": str}) if path.is_file() else None

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
        data["models"] = [dict(m, what_if=bool(raw is not None and m.get("what_if"))) for m in models.values()]
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
        if raw is None or not isinstance(run, str) or not models.get(run, {}).get("what_if"):
            abort(400, "Chưa có bản CPU đã đối chiếu cho cấu hình này.")
        try:
            rain = float(body.get("rain_multiplier", 1))
            temp = float(body.get("temperature_delta", 0))
            model = predictor(run)
            baseline = model.predict(body["basin"], body["date"])
            basin_index = ids[body["basin"]]
            atol = qmax_m3_s(PRED_ATOL, catalog["areas"][body["basin"]])
            for item in baseline:
                target_index = date_index[item["target_date"]]
                expected = matrix(f"series/{run}_lead{item['lead']}.npy")[basin_index, target_index]
                if not np.isclose(item["qmax_m3_s"], expected, atol=atol, rtol=PRED_RTOL):
                    raise ValueError("Dự báo gốc CPU khác tệp phát lại tại cửa sổ đã chọn.")
            changed = model.predict(body["basin"], body["date"], rain_multiplier=rain, temperature_delta=temp)
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            return jsonify(error=str(exc)), 422
        return jsonify(baseline=baseline, changed=changed)

    @app.get("/api/comparison")
    def comparison():
        run, reference = request.args.get("run"), request.args.get("reference")
        lead = integer_arg("lead", 1, 7)
        scope = models.get(run, {}).get("stage")
        suffix = f"H_metrics_lead{lead}" if scope == "H" else "metrics"
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
        names = ("C2_config_table", "C2_cost_table", "D2_CPU_benchmark", "c2_scenarios", "D", "D_cost",
                 *[f"H_summary_lead{k}" for k in range(1, 8)])
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
        result = {}
        for name in ("events", "levels", "event_cases"):
            key = f"H_{name}_lead{lead}" if models[run]["stage"] == "H" else ("events_by_case" if name=="event_cases" else name)
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
        found = table("G")
        if found is None:
            return jsonify(cases=[], rows=[])
        keys = ["basin_id", "origin_date"]
        required = {*keys, "core", "group", "day_offset", "attribution"}
        if not required <= set(found):
            return jsonify(error="Bảng Integrated Gradients chưa khớp các cột đã bàn giao."), 422
        cases = found[keys].drop_duplicates()
        basin, origin = request.args.get("basin"), request.args.get("origin")
        if basin is None or origin is None:
            first = cases.iloc[0] if not cases.empty else None
            basin, origin = (first.basin_id, first.origin_date) if first is not None else (None,None)
        selected = found[found.basin_id.eq(basin) & found.origin_date.eq(origin)]
        return jsonify(cases=columns_for_view(cases), rows=columns_for_view(selected),
                       basin=basin, origin=origin)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy bản demo trên laptop.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="Thư mục LamaH_extended đã kiểm nguồn để bật nếu…thì…")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    create_app(args.bundle, args.raw).run(host="127.0.0.1", port=args.port)
