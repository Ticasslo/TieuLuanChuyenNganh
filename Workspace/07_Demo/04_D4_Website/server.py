# Máy chủ website: dùng lại create_app của D3 (API và phép tính đã đối chiếu D1/D2, không sửa),
# thay giao diện bằng thư mục static/ của website, thêm API cho bản đồ, chuỗi cả kỳ test, bảng xếp hạng,
# tóm tắt cảnh báo lũ toàn miền, lớp bản đồ khí tượng/đất/thực vật (env.json, env/), phục vụ nền bản đồ và địa hình
# PMTiles tự lưu, chặn quá tải phần "nếu... thì...".
# Chạy thử: python server.py --bundle <gói website> --pmtiles <cee_z11.pmtiles> --terrain <terrain_z11.pmtiles>
# Chạy thật (VPS): Docker Compose trong deploy/ (gunicorn 1 tiến trình, 4 luồng, "server:from_env()"), hướng dẫn ở deploy/DEPLOY.md

import argparse
import json
import math
import mimetypes
import os
import sys
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore, Lock

import numpy as np
import pandas as pd
from flask import abort, jsonify, request, send_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "03_D3_LamaHCEExt_Demo"))
from app import create_app as create_d3_app  # noqa: E402

# Bảng MIME của Windows/Linux có thể thiếu .mjs hoặc ghi .js là text/plain => trình duyệt chặn module
for ext in (".js", ".mjs"):
    mimetypes.add_type("text/javascript", ext)
mimetypes.add_type("application/x-protobuf", ".pbf")

SCOPE, SCENARIO = "common_source_all", "S0"
RETURN_PERIODS = (1.5, 2.0, 5.0, 10.0)


def create_app(bundle, raw=None, h_python=None, pmtiles=None, what_if_slots=1, what_if_per_minute=12, client_ip_header=None,
               terrain=None):
    bundle = Path(bundle).resolve()
    pmtiles = Path(pmtiles).resolve() if pmtiles else None
    terrain = Path(terrain).resolve() if terrain else None
    app = create_d3_app(bundle, raw, h_python)
    app.static_folder = str(HERE / "static")
    app.view_functions["home"] = lambda: app.send_static_file("index.html")

    catalog = json.loads((bundle / "catalog.json").read_text(encoding="utf-8"))
    site = json.loads((bundle / "site.json").read_text(encoding="utf-8"))
    models = {m["run"]: m for m in catalog["models"]}
    ids = {bid: i for i, bid in enumerate(catalog["ids"])}
    date_index = {day: i for i, day in enumerate(catalog["dates"])}

    @lru_cache(maxsize=96)
    def matrix(name):
        result = np.load(bundle / name, mmap_mode="r", allow_pickle=False)
        if result.shape != (len(ids), len(date_index)):
            raise ValueError(f"Sai kích thước dữ liệu phát lại: {name}")
        return result

    @lru_cache(maxsize=64)
    def table(name):
        path = bundle / f"{name}.csv"
        return pd.read_csv(path, dtype={"basin_id": str}) if path.is_file() else None

    def values(arr):
        return [round(float(x), 4) if np.isfinite(x) else None for x in arr]

    def lead_arg():
        try:
            lead = int(request.args.get("lead", 1))
        except ValueError:
            abort(400, "Hạn phải là số nguyên.")
        if not 1 <= lead <= 7:
            abort(400, "Hạn từ 1 đến 7 ngày.")
        return lead

    def run_arg(lead, name="run"):
        run = request.args.get(name)
        if run not in models or lead not in models[run]["leads"]:
            abort(400, "Mô hình không có hạn này.")
        return run

    def uses_h(run, lead):
        return models[run]["stage"] == "H" or (run == "Persistence_qmax" and lead > 1)

    @app.get("/api/site")
    def get_site():
        return jsonify(dict(site, flood_m3_s=catalog["flood_m3_s"]))

    @app.get("/api/fan")
    def get_fan():
        # Một lần phát dự báo 7 ngày: ngày phát hành i cho ngày đích i + hạn - 1 (cùng lịch với "nếu... thì..." của D3)
        basin, date = request.args.get("basin"), request.args.get("date")
        if basin not in ids or date not in date_index:
            abort(400, "Chọn lưu vực và ngày phát hành.")
        run = request.args.get("run")
        if run not in models:
            abort(400, "Chọn mô hình.")
        start, row = date_index[date], ids[basin]
        steps = []
        for lead in models[run]["leads"]:
            col = start + lead - 1
            if col < len(date_index):
                value = matrix(f"series/{run}_lead{lead}.npy")[row, col]
                steps.append(dict(lead=lead, target_date=catalog["dates"][col], qmax_m3_s=values([value])[0],
                                  observed_m3_s=values([matrix("observed.npy")[row, col]])[0]))
        return jsonify(basin=basin, run=run, origin=date, steps=steps)

    @lru_cache(maxsize=1)
    def flood_highlights():
        # Đợt lũ lớn trong kỳ test theo quan trắc: ngày-trạm có Qmax vượt ngưỡng chu kỳ lặp lại 10 năm của trạm,
        # gom các ngày cách nhau không quá 3 ngày thành một đợt; mỗi đợt lấy trạm vượt ngưỡng nhiều lần nhất làm đại diện
        obs = np.asarray(matrix("observed.npy"), dtype=float)
        t10 = np.array([float((catalog["flood_m3_s"].get(bid) or {}).get("10.0", np.nan)) for bid in catalog["ids"]])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = obs / t10[:, None]
        over = np.isfinite(ratio) & (ratio >= 1.0)
        days_over = np.flatnonzero(over.any(axis=0))
        groups, current = [], []
        for col in days_over:
            if current and col - current[-1] > 3:
                groups.append(current)
                current = []
            current.append(int(col))
        if current:
            groups.append(current)
        out = []
        for cols in groups:
            block = np.where(over[:, cols], ratio[:, cols], np.nan)
            gauges = np.flatnonzero(np.isfinite(block).any(axis=1))
            row, j = np.unravel_index(np.nanargmax(block), block.shape)
            bid = catalog["ids"][row]
            stn = site["stations"][bid]
            out.append(dict(basin=bid, name=stn["name"], river=stn["river"], country=stn["country"],
                            date=catalog["dates"][cols[j]], start=catalog["dates"][cols[0]], end=catalog["dates"][cols[-1]],
                            gauges=int(len(gauges)), ratio=round(float(block[row, j]), 2),
                            qmax_m3_s=round(float(obs[row, cols[j]]), 1), t10_m3_s=round(float(t10[row]), 1)))
        out.sort(key=lambda e: (-e["gauges"], -e["ratio"]))
        return out

    @app.get("/api/highlights")
    def get_highlights():
        return jsonify(events=flood_highlights()[:8])

    # Tệp dữ liệu bản đồ (geojson, env.json) đổi khi dựng lại gói => trình duyệt hỏi lại mỗi lần (ETag, không đổi trả 304)
    # thay vì giữ bản cũ một ngày
    @app.get("/api/basins")
    def get_basins():
        return send_file(bundle / "basins.geojson", mimetype="application/geo+json", max_age=0)

    # Phần riêng của lưu vực (tô lớp bản đồ không chồng màu) và mạng sông HydroRIVERS (build_map_layers.py);
    # gói cũ chưa có thì trả tập rỗng để trang vẫn chạy
    def optional_geojson(name):
        path = bundle / name
        if not path.is_file():
            return jsonify(type="FeatureCollection", features=[])
        return send_file(path, mimetype="application/geo+json", max_age=0)

    @app.get("/api/basins-own")
    def get_basins_own():
        return optional_geojson("basins_own.geojson")

    @app.get("/api/rivers")
    def get_rivers():
        return optional_geojson("rivers.geojson")

    @app.get("/tiles/basemap.pmtiles")
    def get_tiles():
        if pmtiles is None:
            abort(404, "Máy chủ chưa có nền bản đồ.")
        # conditional=True => trả 206 theo Range, PMTiles đọc từng đoạn tệp
        return send_file(pmtiles, mimetype="application/octet-stream", conditional=True, max_age=86400)

    @app.get("/tiles/terrain.pmtiles")
    def get_terrain():
        # Mô hình độ cao Mapterhorn (terrarium, webp) cắt theo vùng nghiên cứu => bóng núi và địa hình 3D
        if terrain is None:
            abort(404, "Máy chủ chưa có dữ liệu địa hình.")
        return send_file(terrain, mimetype="application/octet-stream", conditional=True, max_age=86400)

    env_path = bundle / "env.json"
    env_meta = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else None
    if env_meta is not None and (env_meta["ids"] != catalog["ids"] or env_meta["start"] != catalog["dates"][0]
                                 or env_meta["days"] != len(catalog["dates"])):
        raise ValueError("env.json không cùng lưu vực, ngày với catalog.json.")
    env_daily = {layer["key"] for layer in env_meta["layers"] if layer["kind"] == "daily"} if env_meta else set()

    @app.get("/api/env")
    def get_env():
        # Danh sách lớp bản đồ (đơn vị, ngưỡng màu) và giá trị thuộc tính tĩnh của mọi lưu vực
        if env_meta is None:
            return jsonify(layers=[])
        return send_file(env_path, mimetype="application/json", max_age=0)

    @app.get("/api/env-days")
    def get_env_days():
        # Giá trị trung bình lưu vực của một lớp theo ngày (tối đa 90 ngày một lần gọi, như /api/days);
        # lớp gió kèm thành phần đông (u), bắc (v) để vẽ mũi tên hướng gió
        layer = request.args.get("layer")
        if layer not in env_daily:
            abort(400, "Chọn lớp bản đồ theo ngày.")
        start = request.args.get("start")
        if start not in date_index:
            abort(400, "Chọn ngày bắt đầu trong kỳ test 2014-2017.")
        try:
            n = int(request.args.get("n", 60))
        except ValueError:
            abort(400, "Số ngày phải là số nguyên.")
        a = date_index[start]
        b = min(len(date_index), a + max(1, min(90, n)))
        out = dict(layer=layer, ids=catalog["ids"], dates=catalog["dates"][a:b], values=block(matrix(f"env/{layer}.npy")[:, a:b]))
        if layer == "wind":
            out.update(u=block(matrix("env/wind_u.npy")[:, a:b]), v=block(matrix("env/wind_v.npy")[:, a:b]))
        return jsonify(out)

    @app.get("/api/day")
    def get_day():
        # Lưu lượng quan trắc và dự báo của mọi lưu vực ở một ngày đích (tô màu bản đồ)
        date = request.args.get("date")
        if date not in date_index:
            abort(400, "Chọn ngày trong kỳ test 2014-2017.")
        lead = lead_arg()
        run = run_arg(lead)
        col = date_index[date]
        return jsonify(date=date, run=run, lead=lead, ids=catalog["ids"],
                       observed=values(matrix("observed.npy")[:, col]),
                       predicted=values(matrix(f"series/{run}_lead{lead}.npy")[:, col]))

    @app.get("/api/days")
    def get_days():
        # Nhiều ngày liền nhau trong một lần gọi (tối đa 90): trang nạp trước khi chạy thời gian, khỏi gọi từng ngày
        start = request.args.get("start")
        if start not in date_index:
            abort(400, "Chọn ngày bắt đầu trong kỳ test 2014-2017.")
        try:
            n = int(request.args.get("n", 60))
        except ValueError:
            abort(400, "Số ngày phải là số nguyên.")
        n = max(1, min(90, n))
        lead = lead_arg()
        run = run_arg(lead)
        a = date_index[start]
        b = min(len(date_index), a + n)
        return jsonify(run=run, lead=lead, ids=catalog["ids"], dates=catalog["dates"][a:b],
                       observed=block(matrix("observed.npy")[:, a:b]), predicted=block(matrix(f"series/{run}_lead{lead}.npy")[:, a:b]))

    def block(arr):
        # Ma trận [lưu vực, ngày] => danh sách theo ngày, làm tròn 4 chữ số, NaN thành null (vector hóa, nhanh hơn từng phần tử)
        arr = np.round(np.asarray(arr, dtype=np.float64).T, 4)
        out = arr.astype(object)
        out[~np.isfinite(arr)] = None
        return out.tolist()

    @app.get("/api/series")
    def get_series():
        # Cả kỳ test của một lưu vực: quan trắc và các mô hình đã chọn ở cùng hạn
        basin = request.args.get("basin")
        if basin not in ids:
            abort(400, "Chọn lưu vực.")
        lead = lead_arg()
        runs = list(dict.fromkeys(request.args.getlist("run")))
        if not runs or len(runs) > 6 or any(r not in models or lead not in models[r]["leads"] for r in runs):
            abort(400, "Chọn 1-6 mô hình có hạn này.")
        row = ids[basin]
        return jsonify(basin=basin, lead=lead, dates=catalog["dates"],
                       observed=values(matrix("observed.npy")[row]),
                       predictions=[dict(run=r, values=values(matrix(f"series/{r}_lead{lead}.npy")[row])) for r in runs],
                       flood_m3_s=catalog["flood_m3_s"][basin])

    @lru_cache(maxsize=1)
    def leaderboard_rows():
        rows = []
        base = table("metrics")
        base = base[base.scope.eq(SCOPE) & base.scenario.eq(SCENARIO) & base.lead.eq(1)]
        d_basin = table("D_basin")
        for run, model in models.items():
            for lead in model["leads"]:
                if model["stage"] == "D":
                    frame = d_basin[d_basin.configuration.eq(model["configuration"]) & d_basin.seed.eq(model["seed"])]
                    nse, kge, n = frame.NSE_730, frame.KGE_730, frame.basin_id.nunique()
                else:
                    src = base if not uses_h(run, lead) else table(f"H_metrics_lead{lead}")
                    if uses_h(run, lead):
                        src = src[src.scope.eq(SCOPE) & src.scenario.eq(SCENARIO)]
                    frame = src[src.run.eq(run)]
                    nse, kge, n = frame.NSE, frame.KGE, frame.basin_id.nunique()
                if n:
                    rows.append(dict(run=run, lead=lead, NSE=float(nse.median()), KGE=float(kge.median()), basins=int(n),
                                     share_good=float((nse > 0.5).mean())))
        return rows

    @app.get("/api/leaderboard")
    def get_leaderboard():
        return jsonify(rows=leaderboard_rows())

    @app.get("/api/basin-scores")
    def basin_scores():
        # NSE từng lưu vực của một mô hình (bản đồ so sánh), có thể trừ mốc cùng hạn
        lead = lead_arg()
        run = run_arg(lead)
        reference = request.args.get("reference") or None

        def scores(name):
            model = models[name]
            if model["stage"] == "D":
                d = table("D_basin")
                d = d[d.configuration.eq(model["configuration"]) & d.seed.eq(model["seed"])]
                return d.set_index(d.basin_id.astype(str)).NSE_730
            src = table(f"H_metrics_lead{lead}") if uses_h(name, lead) else table("metrics")
            src = src[src.scope.eq(SCOPE) & src.scenario.eq(SCENARIO) & src.lead.eq(lead) & src.run.eq(name)]
            return src.set_index("basin_id").NSE

        a = scores(run)
        if reference is not None:
            if reference not in models or lead not in models[reference]["leads"]:
                abort(400, "Mốc không có hạn này.")
            b = scores(reference)
            a = (a - b).dropna()
        return jsonify(run=run, reference=reference, lead=lead,
                       scores={k: round(float(v), 4) for k, v in a.items() if np.isfinite(v)})

    @app.get("/api/flood-summary")
    def flood_summary():
        # Cộng trận lũ của 451 lưu vực theo ngưỡng chu kỳ lặp lại (gộp trận USWRC)
        lead = lead_arg()
        run = run_arg(lead)
        stage = models[run]["stage"]
        name = f"H_events_lead{lead}" if uses_h(run, lead) else "D_events" if stage == "D" else "events"
        frame = table(name)
        if frame is None:
            return jsonify(rows=[])
        frame = frame[frame.run.eq(run) & frame.lead.eq(lead) & frame.scope.eq(SCOPE)
                      & frame.scenario.eq(SCENARIO) & frame.grouping.eq("USWRC")]
        rows = []
        for period in RETURN_PERIODS:
            part = frame[np.isclose(frame.return_period_years, period)]
            hit, missed, false = (int(part[c].sum()) for c in ("hit", "missed", "false_alarm"))
            rows.append(dict(return_period_years=period, hit=hit, missed=missed, false_alarm=false,
                             basins_with_events=int((part.observed_total > 0).sum())))
        return jsonify(run=run, lead=lead, rows=rows)

    areas = np.array([float(catalog["areas"][bid]) for bid in catalog["ids"]])

    def nse(obs, sim):
        # NSE trên các ngày đã chọn sẵn (mọi đường đều có số); phương sai quan trắc 0 => không tính
        denom = float(np.sum((obs - obs.mean()) ** 2))
        return None if denom <= 0 or obs.size < 2 else round(1 - float(np.sum((sim - obs) ** 2)) / denom, 4)

    @app.get("/api/chart")
    def get_chart():
        # Biểu đồ lưu lượng: 1-4 mô hình cùng hạn, tùy chọn trung bình các mô hình (tổ hợp minh họa);
        # basin=all => trung bình theo diện tích (mm/ngày) của mọi lưu vực, mỗi ngày chỉ lấy lưu vực mọi đường đều có số
        basin = request.args.get("basin")
        if basin != "all" and basin not in ids:
            abort(400, "Chọn lưu vực.")
        lead = lead_arg()
        runs = list(dict.fromkeys(request.args.getlist("run")))
        if not runs or len(runs) > 6 or any(r not in models or lead not in models[r]["leads"] for r in runs):
            abort(400, "Chọn 1-6 mô hình có hạn này.")
        ensemble = request.args.get("ensemble") == "1" and len(runs) >= 2
        ens_runs = [r for r in runs if models[r]["stage"] != "Mốc"] if ensemble else []
        ensemble = len(ens_runs) >= 2
        if basin == "all":
            scale = (86.4 / areas)[:, None]
            obs = matrix("observed.npy") * scale
            sims = {r: matrix(f"series/{r}_lead{lead}.npy") * scale for r in runs}
            ok = np.isfinite(obs) & np.logical_and.reduce([np.isfinite(s) for s in sims.values()])
            count = ok.sum(axis=0)
            with np.errstate(invalid="ignore"):
                mean = lambda a: np.where(count > 0, np.where(ok, a, 0).sum(axis=0) / np.maximum(count, 1), np.nan)  # noqa: E731
                obs_line = mean(obs)
                lines = {r: mean(s) for r, s in sims.items()}
            unit = "mm/day"
        else:
            row = ids[basin]
            obs_line = np.asarray(matrix("observed.npy")[row], dtype=float)
            lines = {r: np.asarray(matrix(f"series/{r}_lead{lead}.npy")[row], dtype=float) for r in runs}
            unit = "m3/s"
        if ensemble:
            lines["ensemble"] = np.mean([lines[r] for r in ens_runs], axis=0)
        common = np.isfinite(obs_line) & np.logical_and.reduce([np.isfinite(v) for v in lines.values()])
        scores = {k: nse(obs_line[common], v[common]) for k, v in lines.items()}
        return jsonify(basin=basin, lead=lead, unit=unit, dates=catalog["dates"], observed=values(obs_line),
                       predictions=[dict(run=r, values=values(lines[r]), nse=scores[r]) for r in runs],
                       ensemble=dict(runs=ens_runs, values=values(lines["ensemble"]), nse=scores["ensemble"]) if ensemble else None,
                       nse_days=int(common.sum()),
                       flood_m3_s=catalog["flood_m3_s"].get(basin) if basin != "all" else None)

    LEVEL_ORDER = ["low", "medium", "high", "very_high"]

    @app.get("/api/confusion")
    def get_confusion():
        # Ma trận nhầm lẫn bốn mức gộp 451 lưu vực (Evaluation V13): hàng = mức thực đo, cột = mức dự báo
        lead = lead_arg()
        run = run_arg(lead)
        frame = table("confusion")
        if frame is None:
            return jsonify(run=run, lead=lead, matrix=None)
        part = "D" if models[run]["stage"] == "D" else f"H_lead{lead}" if uses_h(run, lead) else "C2_B"
        part_rows = frame[frame.part.eq(part) & frame.run.eq(run) & frame.lead.eq(lead)]
        grid = [[int(part_rows[part_rows.observed_level.eq(o) & part_rows.predicted_level.eq(p)].days.sum())
                 for p in LEVEL_ORDER] for o in LEVEL_ORDER]
        return jsonify(run=run, lead=lead, part=part, levels=LEVEL_ORDER, matrix=grid)

    @lru_cache(maxsize=1)
    def by_lead_rows():
        # NSE trung vị theo hạn của mọi run H và persistence, cùng phạm vi bảng xếp hạng
        rows = []
        for lead in range(1, 8):
            frame = table(f"H_metrics_lead{lead}")
            frame = frame[frame.scope.eq(SCOPE) & frame.scenario.eq(SCENARIO)]
            for run, part in frame.groupby("run"):
                rows.append(dict(run=run, lead=lead, NSE=float(part.NSE.median())))
        return rows

    @app.get("/api/by-lead")
    def get_by_lead():
        # Gom 3 hạt giống của từng cách x lõi: trung bình và độ lệch chuẩn của NSE trung vị
        groups = defaultdict(list)
        for row in by_lead_rows():
            run = row["run"]
            if run == "Persistence_qmax":
                key = ("persistence", None)
            else:
                stem = run.split("_H")[1]
                key = (stem.split("_seed")[0], run.split("_")[1])
            groups[(key, row["lead"])].append(row["NSE"])
        out = [dict(mode=k[0], core=k[1], lead=lead, mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, n=len(v))
               for (k, lead), v in groups.items()]
        return jsonify(rows=sorted(out, key=lambda r: (r["mode"], r["core"] or "", r["lead"])))

    @lru_cache(maxsize=1)
    def c2_seed_rows():
        base = table("metrics")
        base = base[base.scope.eq(SCOPE) & base.scenario.eq(SCENARIO) & base.lead.eq(1) & base.run.str.endswith(tuple(f"_C2_seed{s}" for s in (111, 222, 333)))]
        out = []
        for conf, part in base.groupby(base.run.str.removeprefix("SF_").str.split("_C2_").str[0]):
            medians = part.groupby("run").NSE.median()
            out.append(dict(configuration=conf, mean=float(medians.mean()), sd=float(medians.std(ddof=1)), n=int(medians.size)))
        return out

    @app.get("/api/c2-seeds")
    def get_c2_seeds():
        return jsonify(rows=c2_seed_rows())

    @app.get("/api/d-pairs")
    def get_d_pairs():
        # 365 so với 730 ngày, cùng cấu hình và hạt giống (D_length_pairs của 17_D_LamaHCEExt_Compare)
        frame = table("D_length_pairs")
        frame = frame[frame.metric.eq("NSE")]
        cols = ["configuration", "seed", "median_365", "median_730", "improved", "degraded", "n_pairs", "p_holm"]
        return jsonify(rows=[{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in frame[cols].to_dict("records")])

    # Chặn quá tải phần chạy mô hình trên CPU: tối đa what_if_slots lần cùng lúc, mỗi IP what_if_per_minute lần/phút
    # (sau Caddy, IP thật là mục cuối của X-Forwarded-For vì Caddy nối thêm vào cuối; sau Cloudflare Tunnel, Caddy
    # chỉ thấy IP của cloudflared => đọc header client_ip_header do Cloudflare đặt, VD CF-Connecting-IP)
    slots = BoundedSemaphore(what_if_slots)
    recent, recent_lock = defaultdict(deque), Lock()
    original_what_if = app.view_functions["what_if"]

    def guarded_what_if():
        client = (client_ip_header and request.headers.get(client_ip_header, "").strip()) or \
            request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[-1].strip()
        now = time.monotonic()
        with recent_lock:
            calls = recent[client]
            while calls and now - calls[0] > 60:
                calls.popleft()
            if len(calls) >= what_if_per_minute:
                wait = max(1, math.ceil(60 - (now - calls[0])))
                return jsonify(error="rate_limited", code="rate_limited", retry_after=wait, limit=what_if_per_minute), 429, {"Retry-After": str(wait)}
            calls.append(now)
        if not slots.acquire(timeout=20):
            return jsonify(error="busy", code="busy", retry_after=5), 503, {"Retry-After": "5"}
        try:
            return original_what_if()
        finally:
            slots.release()

    app.view_functions["what_if"] = guarded_what_if

    @app.after_request
    def headers(response):
        # Thư viện, phông chữ không đổi giữa các lần triển khai => lưu 7 ngày; mã trang (js, css) đổi mỗi lần
        # triển khai mà không đổi tên tệp => no-cache (trình duyệt, Cloudflare hỏi lại bằng ETag), tránh HTML mới chạy JS cũ
        if request.path.startswith("/static/vendor/") or request.path == "/plotly.js":
            response.cache_control.no_cache = None
            response.cache_control.max_age = 604800
            response.cache_control.public = True
        elif request.path.startswith("/static/"):
            response.cache_control.max_age = None
            response.cache_control.no_cache = True
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    return app


def from_env():
    env = os.environ
    if not env.get("DEMO_BUNDLE"):
        raise RuntimeError("Đặt DEMO_BUNDLE trỏ tới gói dữ liệu website.")
    path = lambda key: Path(env[key]) if env.get(key) else None  # noqa: E731
    return create_app(Path(env["DEMO_BUNDLE"]), path("DEMO_RAW"), path("DEMO_H_PYTHON"), path("DEMO_PMTILES"),
                      int(env.get("DEMO_WHAT_IF_SLOTS", 1)), int(env.get("DEMO_WHAT_IF_PER_MINUTE", 12)), env.get("DEMO_CLIENT_IP_HEADER") or None,
                      path("DEMO_TERRAIN"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy website trên laptop (máy chủ thử của Flask).")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pmtiles", type=Path)
    parser.add_argument("--terrain", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--h-python", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8060)
    args = parser.parse_args()
    create_app(args.bundle, args.raw, args.h_python, args.pmtiles, terrain=args.terrain).run(host=args.host, port=args.port, threaded=True)
