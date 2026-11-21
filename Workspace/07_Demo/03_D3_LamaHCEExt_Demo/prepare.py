import argparse
import runpy
import shutil
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from pyproj import Transformer
import shapefile
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import transform, unary_union


HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
sys.path.insert(0, str(DEMO / "DemoRuntime"))
from forecast import PRED_RTOL, compare_arrays, qmax_m3_s
from h_cpu import FREEZE_H, H_CHECKPOINT_FILES, H_PRED_ATOL, HWorker

H_REPORT_TABLES = ("summary", "metrics", "events", "levels", "event_cases")
H_AUDIT_COLUMNS = ["file", "bytes", "sha256", "passed", "run", "mode", "lead", "library_commit"]
D_REPORT_FILES = {
    "D_eval_summary.csv": "Evaluation_D_summary.csv",
    "D_length_pairs.csv": "Evaluation_D_length_pairs.csv",
    "D_metrics.csv": "Evaluation_D_metrics.csv",
    "D_events.csv": "Evaluation_D_events.csv",
    "D_levels.csv": "Evaluation_D_levels.csv",
    "D_event_cases.csv": "Evaluation_D_event_cases.csv",
}
D_AUDIT_COLUMNS = ["file", "bytes", "sha256", "passed"]
H_MODE_LABELS = {"past": "Chỉ dữ liệu quá khứ", "hybrid": "ECMWF HRES ngày đầu",
                 "gefs": "ECMWF HRES và GEFS", "perfect": "Thời tiết tái phân tích (tham chiếu lý tưởng)"}
G_C2_TABLES = ("G_demo_runs.csv", "G_run_metrics.csv", "G_core_diffs.csv", "G_strata.csv",
               "G_basin_L90.csv", "G_probe_gain.csv", "G_deletion.csv", "G_lag_share.npz", "g_summary.json")
G_H_TABLES = ("GH_demo_runs.csv", "GH_lead_summary.csv", "GH_core_diffs.csv", "GH_strata_level.csv",
              "GH_deletion.csv", "gh_summary.json")


def load_audit():
    code = runpy.run_path(str(DEMO / "01_D1_LamaHCEExt_Inference/01_D1_LamaHCEExt_Inference.py"))
    return code


def load_evaluator():
    path = DEMO.parent / "06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.py"
    return runpy.run_path(str(path))


def locate_a2(source):
    roots = {p.parent.parent for p in source.rglob("SeqLSTM_Q") if p.is_dir() and p.parent.name == "runs"}
    if len(roots) != 1:
        raise ValueError("Cần đúng một thư mục a2_output/runs/SeqLSTM_Q từ A2 Version 2.")
    return roots.pop()


def check_inputs(source, with_live=False, replay_only_cores=(), h_evaluation=None, h_python=None, d_evaluation=None):
    audit = load_audit()
    problems = []

    def check(label, load):
        try:
            value = load()
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"{label} | {exc}")
            return None
        return value

    def raw_files():
        raw = audit["locate_raw"](source)
        for suffix in (".shp", ".dbf"):
            path = raw / f"A_basins_total_upstrm/3_shapefiles/Basins_A{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
        return raw

    raw = check("Dataset lamah-ce-ext", raw_files)
    for label, filename in (
        ("A1 V3", "A1_P10_qmax_level_thresholds.csv"),
        ("A1 V3", "A1_P10_flood_return_levels.csv"),
        ("A3 V1", "A3_persistence_predictions.npz"),
        ("A3 V1", "A3_Q_source_window.npz"),
        ("C2 Compare V1", "c2_summary.json"),
    ):
        check(label, lambda filename=filename: audit["unique_file"](source, filename))
    check("A2 V2", lambda: locate_a2(source))
    if with_live:
        check("D2 V1", lambda: audit["unique_file"](source, "D2_summary.json"))
    if not problems:
        checked = check("C2 Compare V1 và A3", lambda: audit["compare_inputs"](source))
        if with_live and checked is not None:
            _, compare, records = checked
            check("Checkpoint chạy trực tiếp",
                  lambda: live_c2_sources(source, compare, audit["c2_winners"](records), audit, replay_only_cores))
    if h_evaluation is not None or h_python is not None:
        evaluator = load_evaluator()
        check("H Compare", lambda: require_h_compare(source, audit, evaluator))
    if h_evaluation is not None:
        check("Evaluation H", lambda: h_report_files(h_evaluation, audit, evaluator))
    if d_evaluation is not None:
        d_cmp = check("D Compare", lambda: audit["unique_file"](source, "d_summary.json").parent)
        if d_cmp is not None:
            check("Evaluation D", lambda: d_report_files(d_evaluation, d_cmp, audit))
    g_file = audit["unique_file"](source, "G_IntegratedGradients.csv", required=False)
    if g_file is not None:
        evaluator_g = load_evaluator()
        check("G Integrated Gradients", lambda: read_g_table(g_file, evaluator_g["basin_id"]))
    g_c2_npz = audit["unique_file"](source, "G_demo_basin.npz", required=False)
    g_h_npz = audit["unique_file"](source, "GH_demo_basin.npz", required=False)
    if (g_c2_npz is not None or g_h_npz is not None) and not problems:
        ctx_g, _, _ = audit["compare_inputs"](source)
        if g_c2_npz is not None:
            check("G C2 XAI (G_demo_basin.npz)", lambda: validate_g_c2_demo(g_c2_npz, ctx_g["ids"], audit))
        if g_h_npz is not None:
            check("G H XAI (GH_demo_basin.npz)", lambda: validate_g_h_demo(g_h_npz, ctx_g["ids"], audit))
    if h_python is not None and not h_python.is_file():
        problems.append(f"Python của H | Không tìm thấy {h_python}")
    if problems:
        print("Đầu vào D3 cần bổ sung hoặc chỉnh:\n" + "\n".join(f"- {line}" for line in problems))
    else:
        print(f"Đã tìm thấy các nguồn chính. Thư mục raw để chạy app: {raw}")
    return not problems


def gauge_points(raw, ids, basin_id):
    gauges = pd.read_csv(raw/"D_gauges/1_attributes/Gauge_attributes.csv", sep=";")
    gauges["basin_id"] = gauges.ID.map(basin_id)
    selected = gauges.set_index("basin_id", verify_integrity=True).loc[list(ids)]
    longitude, latitude = Transformer.from_crs(3035, 4326, always_xy=True).transform(
        selected.lon.to_numpy(float), selected.lat.to_numpy(float))
    if not np.isfinite([longitude, latitude]).all() or (
            not np.all(np.abs(latitude) <= 90) or not np.all(np.abs(longitude) <= 180)):
        raise ValueError("Tọa độ trạm không thể đổi từ EPSG:3035 sang WGS84.")
    return [dict(id=bid, lon=x, lat=y) for bid, x, y in zip(ids, longitude, latitude)]


def study_outline(raw, ids, basin_id):
    path = raw/"A_basins_total_upstrm/3_shapefiles/Basins_A.shp"
    wanted = set(ids)
    transform_coords = Transformer.from_crs(3035, 4326, always_xy=True).transform
    with shapefile.Reader(str(path), encoding="latin1") as reader:
        polygons = {basin_id(item.record["ID"]): shape(item.shape.__geo_interface__)
                    for item in reader.iterShapeRecords() if basin_id(item.record["ID"]) in wanted}
    if set(polygons) != wanted:
        raise ValueError("Đa giác vùng nghiên cứu thiếu lưu vực trong A3.")
    boundary = unary_union(list(polygons.values())).simplify(1200, preserve_topology=True)
    boundary = transform(transform_coords, boundary)
    if not isinstance(boundary, (Polygon, MultiPolygon)) or boundary.is_empty:
        raise ValueError("Đường bao vùng nghiên cứu không hợp lệ.")
    parts = list(boundary.geoms) if isinstance(boundary, MultiPolygon) else [boundary]
    return [list(map(list, polygon.exterior.coords)) for polygon in parts]


def add_model(output, name, stage, title, forecasts, context, **metadata):
    target = output / "series"
    target.mkdir(exist_ok=True)
    leads = []
    for lead, forecast in forecasts:
        ids = forecast.ids
        day_offset = context["dates"].get_indexer(forecast.dates)
        if ids != context["ids"] or (day_offset < 0).any() or not np.array_equal(forecast.area, context["area"]):
            raise ValueError(f"Ngữ cảnh khác A3: {name}, hạn {lead}")
        # Đệm lịch toàn test bằng NaN ở các ngày ngoài phạm vi của H.
        values = np.full_like(context["obs"], np.nan)
        values[day_offset] = qmax_m3_s(forecast.sim, forecast.area[None, :])
        np.save(target/f"{name}_lead{lead}.npy", values.T)
        leads.append(lead)
    return dict(run=name, stage=stage, name=title, leads=sorted(leads), **metadata)


def require_h_compare(source, audit, evaluator):
    summary = audit["unique_file"](source, "h_summary.json")
    root = summary.parent
    contracts = list(root.glob("contracts/SF_*_H*_test_S0_lead*.npz"))
    expected = len(evaluator["H_CONFIGS"]) * len(evaluator["H_MODES"]) * len(evaluator["C2_SEEDS"]) * len(evaluator["H_LEADS"])
    if not (root/"manifest.csv").is_file() or len(contracts) != expected:
        raise ValueError(f"Cần Output H Compare với manifest.csv và đủ {expected} NPZ. Tìm thấy {len(contracts)} NPZ.")
    return root


def h_report_files(root, audit, evaluator):
    summary = audit["read_json"](root/"Evaluation_H_summary.json")
    if (summary.get("complete") is not True
            or [row["lead"] for row in summary.get("coverage", [])] != list(evaluator["H_LEADS"])):
        raise ValueError("Evaluation H chưa chấm đủ các hạn.")
    paths = {f"H_{label}_lead{lead}.csv": root/f"Evaluation_H_{label}_lead{lead}.csv"
             for lead in evaluator["H_LEADS"] for label in H_REPORT_TABLES}
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    checked = pd.read_csv(root/"Evaluation_H_input_check.csv")
    if (not set(H_AUDIT_COLUMNS) <= set(checked) or checked["file"].duplicated().any()
            or not checked["passed"].eq(True).all()
            or checked["run"].notna().sum() != summary.get("contracts")):
        raise ValueError("Bảng kiểm nguồn Evaluation H thiếu tệp hoặc có tệp chưa đạt.")
    return paths, checked[H_AUDIT_COLUMNS].sort_values("file", ignore_index=True)


def d_report_files(root, d_compare_root, audit):
    summary = audit["read_json"](root / "Evaluation_D_summary.json")
    d_summary = audit["read_json"](d_compare_root / "d_summary.json")
    if (summary.get("complete") is not True
            or summary.get("contracts") != d_summary.get("contracts")
            or summary.get("configs") != d_summary.get("configs")):
        raise ValueError("Evaluation D chưa hoàn thành hoặc khác danh sách cấu hình/hợp đồng của D Compare.")
    paths = {dest: root / src for dest, src in D_REPORT_FILES.items()}
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    checked = pd.read_csv(root / "Evaluation_D_input_check.csv")
    _, source_checked = audit["verified_manifest"](d_compare_root)
    if (not set(D_AUDIT_COLUMNS) <= set(checked) or checked["file"].duplicated().any()
            or not checked["passed"].eq(True).all()
            or not checked[D_AUDIT_COLUMNS].sort_values("file", ignore_index=True).equals(
                source_checked[D_AUDIT_COLUMNS].sort_values("file", ignore_index=True))):
        raise ValueError("Bảng kiểm nguồn Evaluation D khác bộ đầu vào D Compare trên máy.")
    return paths


def verify_g_manifest_entry(path, audit):
    # Kiểm riêng tệp demo/G*_demo_basin.npz với manifest.csv của g_output/ (không bắt buộc giải nén 1,8 GB ig/*.npz).
    g_root = path.parent.parent
    manifest_path = g_root / "manifest.csv"
    if not manifest_path.is_file():
        return
    man = pd.read_csv(manifest_path, dtype={"sha256": str})
    col = "file" if "file" in man else "tệp"
    rel = path.relative_to(g_root).as_posix()
    hit = man[man[col].eq(rel)]
    if len(hit) != 1:
        raise ValueError(f"Manifest {manifest_path} thiếu mục {rel}.")
    row = hit.iloc[0]
    if audit["sha256"](path) != row["sha256"] or ("bytes" in row and int(row["bytes"]) != path.stat().st_size):
        raise ValueError(f"SHA-256 hoặc dung lượng không khớp manifest: {rel}.")


def validate_g_c2_demo(path, expected_ids, audit):
    verify_g_manifest_entry(path, audit)
    with np.load(path, allow_pickle=False) as d:
        needed = {"runs", "basin_ids", "groups", "source_groups", "lag_share", "group_share",
                  "memory_L50", "memory_L90", "group_lag_0_60", "n_samples", "demo_runs"}
        if not needed <= set(d.files):
            raise ValueError(f"G_demo_basin.npz thiếu mảng: {sorted(needed - set(d.files))}.")
        bids = tuple(str(int(str(x).strip())) for x in d["basin_ids"])
        if bids != tuple(expected_ids):
            raise ValueError("G_demo_basin.npz khác danh sách hoặc thứ tự 451 lưu vực A3.")
        runs = [str(x) for x in d["runs"]]
        demo_runs = [str(x) for x in d["demo_runs"]]
        if not demo_runs or not set(demo_runs) <= set(runs):
            raise ValueError("G_demo_basin.npz có demo_runs ngoài danh sách runs.")
        nr, nb, ng, ns = len(runs), len(bids), len(d["groups"]), len(d["source_groups"])
        if (d["lag_share"].shape != (nr, nb, 365) or d["group_share"].shape != (nr, nb, ng)
                or d["group_lag_0_60"].shape != (nr, nb, 61, ns)
                or d["memory_L50"].shape != (nr, nb) or d["memory_L90"].shape != (nr, nb)):
            raise ValueError("G_demo_basin.npz sai kích thước mảng.")
        has = d["n_samples"] > 0
        if not has.any():
            raise ValueError("G_demo_basin.npz không có lưu vực nào có mẫu.")
        gs, ls = d["group_share"][:, has], d["lag_share"][:, has]
        if (not np.isfinite(gs).all() or not np.isfinite(ls).all()
                or not np.isfinite(d["group_lag_0_60"]).all()
                or not np.allclose(gs.sum(-1), 1.0, atol=1e-3)
                or not np.allclose(ls.sum(-1), 1.0, atol=1e-3)):
            raise ValueError("G_demo_basin.npz có giá trị không hữu hạn hoặc tỉ phần không cộng bằng 1.")
    return dict(runs=runs, demo_runs=demo_runs, sha256=audit["sha256"](path))


def validate_g_h_demo(path, expected_ids, audit):
    verify_g_manifest_entry(path, audit)
    with np.load(path, allow_pickle=False) as d:
        needed = {"runs", "basin_ids", "leads", "phases", "groups", "phase_share", "group_share",
                  "fore_day_share", "memory_L50", "memory_L90", "n_samples", "n_samples_all", "demo_runs"}
        if not needed <= set(d.files):
            raise ValueError(f"GH_demo_basin.npz thiếu mảng: {sorted(needed - set(d.files))}.")
        bids = tuple(str(int(str(x).strip())) for x in d["basin_ids"])
        if bids != tuple(expected_ids):
            raise ValueError("GH_demo_basin.npz khác danh sách hoặc thứ tự 451 lưu vực A3.")
        leads = [int(x) for x in d["leads"]]
        phases = [str(x) for x in d["phases"]]
        if leads != list(range(1, 8)) or phases != ["past", "fore", "static"]:
            raise ValueError("GH_demo_basin.npz sai trục hạn 1–7 hoặc thứ tự 3 pha.")
        runs = [str(x) for x in d["runs"]]
        demo_runs = [str(x) for x in d["demo_runs"]]
        if not demo_runs or not set(demo_runs) <= set(runs):
            raise ValueError("GH_demo_basin.npz có demo_runs ngoài danh sách runs.")
        nr, nb, ng = len(runs), len(bids), len(d["groups"])
        if (d["phase_share"].shape != (nr, nb, 7, 3) or d["group_share"].shape != (nr, nb, 7, ng)
                or d["fore_day_share"].shape != (nr, nb, 7, 7)
                or d["memory_L50"].shape != (nr, nb, 7) or d["memory_L90"].shape != (nr, nb, 7)
                or d["n_samples"].shape != (nb, 7) or d["n_samples_all"].shape != (nb,)):
            raise ValueError("GH_demo_basin.npz sai kích thước mảng.")
        has = d["n_samples"] > 0
        if not has.any():
            raise ValueError("GH_demo_basin.npz không có ô (lưu vực, hạn) nào có mẫu.")
        ps, gs = d["phase_share"][:, has], d["group_share"][:, has]
        ds = d["fore_day_share"][:, has].sum(-1) - ps[..., 1]
        if (not np.isfinite(ps).all() or not np.isfinite(gs).all()
                or not np.isfinite(d["memory_L90"][:, has]).all()
                or not np.allclose(ps.sum(-1), 1.0, atol=1e-3)
                or not np.allclose(gs.sum(-1), 1.0, atol=1e-3)
                or float(np.max(np.abs(ds))) > 1e-3):
            raise ValueError("GH_demo_basin.npz có giá trị không hữu hạn hoặc sai tổng tỉ phần.")
    return dict(runs=runs, demo_runs=demo_runs, sha256=audit["sha256"](path))


def copy_optional_xai_files(names, search_roots, output):
    out_res = output.resolve()
    copied = []
    for filename in names:
        found = None
        for root in search_roots:
            if root is None or not root.exists():
                continue
            hits = [
                p for p in sorted(root.rglob(filename))
                if not p.resolve().is_relative_to(out_res)
                and not any(part.startswith("D3_") for part in p.relative_to(root).parts)
            ]
            if hits:
                found = hits[0]
                break
        if found is not None:
            shutil.copy2(found, output / filename)
            copied.append(filename)
    return copied


def package_xai(source, output, ids, evaluator, audit):
    repo_xai = DEMO.parent / "08_XAI"
    xai_meta = dict(c2_basin=False, c2_summary=False, h_basin=False, h_summary=False, legacy=False,
                    c2_demo_runs=[], h_demo_runs=[])
    g_path = audit["unique_file"](source, "G_IntegratedGradients.csv", required=False)
    if g_path is not None:
        read_g_table(g_path, evaluator["basin_id"], valid_ids=ids).to_csv(output / "G.csv", index=False, encoding="utf-8-sig")
        xai_meta["legacy"] = True
    g_c2 = audit["unique_file"](source, "G_demo_basin.npz", required=False)
    if g_c2 is not None:
        info = validate_g_c2_demo(g_c2, ids, audit)
        shutil.copy2(g_c2, output / "G_C2_demo.npz")
        xai_meta.update(c2_basin=True, c2_demo_runs=info["demo_runs"], c2_sha256=info["sha256"])
    g_h = audit["unique_file"](source, "GH_demo_basin.npz", required=False)
    if g_h is not None:
        info = validate_g_h_demo(g_h, ids, audit)
        shutil.copy2(g_h, output / "G_H_demo.npz")
        xai_meta.update(h_basin=True, h_demo_runs=info["demo_runs"], h_sha256=info["sha256"])
    c2_roots = [g_c2.parent.parent if g_c2 is not None else None, repo_xai / "01_G_C2_XAI" / "Tables", source]
    h_roots = [g_h.parent.parent if g_h is not None else None, repo_xai / "02_G_H_XAI" / "Tables", source]
    c2_copied = copy_optional_xai_files(G_C2_TABLES, c2_roots, output)
    h_copied = copy_optional_xai_files(G_H_TABLES, h_roots, output)
    if "G_run_metrics.csv" in c2_copied:
        xai_meta["c2_summary"] = True
    if "GH_lead_summary.csv" in h_copied:
        xai_meta["h_summary"] = True
    return xai_meta


def read_g_table(path, normalize_basin_id, valid_ids=None):
    frame = pd.read_csv(path, dtype={"basin_id": str, "origin_date": str})
    required = {"basin_id", "origin_date", "core", "group", "day_offset", "attribution"}
    if frame.empty or not required <= set(frame):
        raise ValueError("Bảng G_IntegratedGradients.csv rỗng hoặc thiếu cột bắt buộc.")
    frame["basin_id"] = frame["basin_id"].map(lambda v: normalize_basin_id(str(v).strip().removeprefix("ID_")))
    frame["origin_date"] = frame["origin_date"].astype(str).str.strip()
    if (valid_ids is not None and not set(frame["basin_id"]).issubset(set(valid_ids))
            or not np.isfinite(pd.to_numeric(frame["day_offset"], errors="coerce")).all()
            or not np.isfinite(pd.to_numeric(frame["attribution"], errors="coerce")).all()):
        raise ValueError("Bảng G_IntegratedGradients.csv có basin_id ngoài danh sách hoặc giá trị không hữu hạn.")
    return frame


def prepare(source, output, with_live=False, replay_only_cores=(), h_evaluation=None, h_python=None, d_evaluation=None):
    evaluator = load_evaluator()
    audit = load_audit()
    h_source, h_tables, h_checked, d_tables = source, {}, None, {}
    if h_evaluation is not None or h_python is not None:
        h_source = require_h_compare(source, audit, evaluator)
    if h_evaluation is not None:
        h_tables, h_checked = h_report_files(h_evaluation, audit, evaluator)
    d_summary_file = audit["unique_file"](source, "d_summary.json", required=False)
    if d_evaluation is None and d_summary_file is not None:
        auto_d = audit["unique_file"](source, "Evaluation_D_summary.json", required=False)
        if auto_d is not None:
            d_evaluation = auto_d.parent
    if d_evaluation is not None:
        if d_summary_file is None:
            raise FileNotFoundError("Cần Output D Compare trong --inputs khi ghép bảng Evaluation D.")
        d_tables = d_report_files(d_evaluation, d_summary_file.parent, audit)
    context, compare, records = audit["compare_inputs"](source)
    raw = audit["locate_raw"](source)
    output.mkdir(parents=True, exist_ok=True)
    summary_file = output/"catalog.json"
    if summary_file.exists():
        raise FileExistsError("Chọn thư mục output mới để mỗi gói có một catalog duy nhất.")
    ids = context["ids"]
    a1 = audit["unique_file"](source, "A1_P10_flood_return_levels.csv").parent
    _, flood, _ = evaluator["read_thresholds"](a1, ids)
    areas = dict(zip(ids, context["area"]))
    threshold = flood.reset_index()
    threshold["qmax_m3_s"] = qmax_m3_s(threshold.threshold_mm_day, threshold.basin_id.map(areas))
    levels = threshold.pivot(index="basin_id", columns="return_period_years", values="qmax_m3_s")
    np.save(output/"observed.npy", qmax_m3_s(context["obs"], context["area"][None, :]).T)
    models, winners = [], audit["c2_winners"](records)
    for record in winners:
        forecast, _ = audit["c2_forecast"](record, context)
        name = forecast.name
        models.append(add_model(output, name, "C2", record["conf"], [(1, forecast)], context,
                                core=record["cfg"]["core"], configuration=record["conf"], seed=record["seed"], epoch=record["epoch"],
                                validation_nse=record["seed_info"]["val"]))
    # Mốc A2 dùng đúng run SeqLSTM_Q của tác giả và chính sách chặn từ config A2.
    biascast = next(f for f in evaluator["read_a2"](locate_a2(source), context)[0] if f.name == "SeqLSTM_Q")
    models.append(add_model(output, "SeqLSTM_Q", "Mốc", "BiasCast tác giả", [(1, biascast)], context))
    persistence = evaluator["Forecast"](name="Persistence_qmax", sim=context["persistence"],
                                        raw=context["persistence"], **context)
    models.append(add_model(output, "Persistence_qmax", "Mốc", "Persistence qmax", [(1, persistence)], context))
    compare_files, _ = audit["verified_manifest"](compare)
    for filename in ("C2_config_table.csv", "C2_cost_table.csv"):
        shutil.copy2(compare_files[f"Tables/{filename}"], output/filename)

    append_extensions(source, output, context, models, evaluator, audit, h_source=h_source, h_checked=h_checked)
    if with_live:
        package_live_c2(source, output, compare, winners, models, audit, replay_only_cores)
    if h_python is not None:
        package_live_h(source, output, h_source, raw, h_python, models, context, audit)

    for label, pattern in {
        "metrics": "Evaluation_metrics.csv", "levels": "Evaluation_levels.csv",
        "events": "Evaluation_events.csv", "events_by_case": "Evaluation_event_cases.csv",
        "c2_scenarios": "C2_scenarios_stats.csv",
    }.items():
        path = audit["unique_file"](source, pattern, required=False)
        if path is not None:
            shutil.copy2(path, output/f"{label}.csv")
    d_search = d_summary_file.parent if d_summary_file is not None else source
    for label, pattern in {
        "D": "D_score_stats.csv", "D_ensemble": "D_ensemble_scores.csv",
        "D_cost": "D_cost.csv", "D_basin": "D_basin_scores.csv",
    }.items():
        path = audit["unique_file"](d_search, pattern, required=False)
        if path is not None:
            shutil.copy2(path, output/f"{label}.csv")
    xai_meta = package_xai(source, output, ids, evaluator, audit)
    if not h_tables:
        for lead in evaluator["H_LEADS"]:
            for label in H_REPORT_TABLES:
                name = f"H_{label}_lead{lead}.csv"
                path = audit["unique_file"](source, f"Evaluation_{name}", required=False)
                if path is not None:
                    h_tables[name] = path
    for name, path in h_tables.items():
        shutil.copy2(path, output/name)
    for name, path in d_tables.items():
        shutil.copy2(path, output/name)
    selected = next(m["run"] for m in models if m["run"] == audit["C2_DEFAULT"])
    audit["write_json"](summary_file, dict(schema=1, test_start=str(context["dates"][0].date()),
                                        dates=context["dates"].strftime("%Y-%m-%d").tolist(), ids=ids, areas=areas,
                                        points=gauge_points(raw, ids, evaluator["basin_id"]),
                                        outline=study_outline(raw, ids, evaluator["basin_id"]),
                                        flood_m3_s={i: {str(k):float(v) for k,v in levels.loc[i].items()} for i in ids},
                                        models=models, default_run=selected, library_commit=audit["FREEZE_C"],
                                        C2_manifest_sha256=audit["sha256"](compare/"manifest.csv"), xai=xai_meta))
    print(f"Gói demo: {output} – {len(ids)} lưu vực, {len(models)} đường dự báo theo cấu hình.")


def append_extensions(source, output, context, models, evaluator, audit, *, h_source=None, h_checked=None):
    d_root = audit["unique_file"](source, "d_summary.json", required=False)
    if d_root is not None:
        files, _ = audit["verified_manifest"](d_root.parent)
        summary = audit["read_json"](files["d_summary.json"])
        cfg = pd.read_csv(files["D_cost.csv"])
        candidates = cfg[cfg.seq_length.eq(730)]
        winners = (candidates.sort_values(["validation_NSE", "seed"], ascending=[False, True])
                   .drop_duplicates("configuration"))
        if (summary.get("complete") is not True or summary.get("library_commit") != audit["FREEZE_C"]
                or set(summary["configs"]) != set(winners["configuration"])
                or candidates.duplicated(["configuration", "seed"]).any()
                or any(set(f.seed) != {111,222,333} for _, f in candidates.groupby("configuration"))
                or not np.isfinite(candidates.validation_NSE).all()):
            raise ValueError("D Compare chưa hoàn thành hoặc không dùng freeze-C.")
        if {"D_demo_runs.csv", "D_handoff.csv"} <= set(files):
            demo_tbl = pd.read_csv(files["D_demo_runs.csv"]).set_index("configuration", verify_integrity=True)
            hand_tbl = pd.read_csv(files["D_handoff.csv"]).set_index(["configuration", "seed"], verify_integrity=True)
        else:
            demo_tbl = hand_tbl = None
        for r in winners.itertuples(index=False):
            run = f"SF_{r.configuration}_D_seed{r.seed}"
            f = evaluator["read_contract"](files[f"contracts/{run}_test_S0_lead1.npz"])
            if f.name != run or f.lead != 1 or f.metadata["library_commit"] != audit["FREEZE_C"]:
                raise ValueError(f"Hợp đồng D sai danh tính: {run}")
            if demo_tbl is not None:
                d_sel = demo_tbl.loc[r.configuration]
                h_sel = hand_tbl.loc[(r.configuration, int(r.seed))]
                if (int(d_sel["seed"]) != int(r.seed) or d_sel["run"] != run
                        or h_sel["sha256_config_yml"] != f.metadata.get("config_sha256")
                        or int(h_sel["selected_epoch"]) != f.metadata.get("selected_epoch")):
                    raise ValueError(f"Bảng chọn/bàn giao D khác hợp đồng hoặc thứ tự validation: {run}")
            models.append(add_model(output, run, "D", f"{r.configuration} / 730 bước", [(1, f)], context,
                                    configuration=r.configuration, seed=int(r.seed), validation_nse=r.validation_NSE))
    h_source = source if h_source is None else h_source
    if any(h_source.rglob("SF_*_H*_test_S0_lead*.npz")):
        forecasts, checked = evaluator["read_h_contracts"](h_source, context)
        if h_checked is not None:
            source_audit = checked[H_AUDIT_COLUMNS].sort_values("file", ignore_index=True)
            if not source_audit.equals(h_checked):
                raise ValueError("Tệp H Compare trên máy khác bộ đầu vào đã chấm trong Evaluation H.")
        by_run = {}
        for forecast in forecasts:
            by_run.setdefault(forecast.name, []).append(forecast)
        winners = h_winners(source, by_run, audit, evaluator, checked)
        for (configuration, mode), record in winners.items():
            run = record["run"]
            series = sorted(((f.lead, f) for f in by_run[run]), key=lambda pair: pair[0])
            core = configuration.split("_", 1)[0]
            label = f"{'LSTM' if core == 'lstm' else 'Mamba'} · {H_MODE_LABELS[mode]}"
            models.append(add_model(output, run, "H", label, series, context, core=core,
                                    configuration=configuration, mode=mode, seed=record["seed"],
                                    validation_nse_lead_mean=record["score"], **record.get("checkpoint", {})))
        persistence = next((model for model in models if model["run"] == "Persistence_qmax"), None)
        if persistence is not None:
            reference_run = next(iter(by_run.values()))
            references = [
                (f.lead, evaluator["Forecast"](name=persistence["run"], sim=f.persistence, raw=f.persistence,
                                              dates=f.dates, ids=f.ids, area=f.area, obs=f.obs,
                                              persistence=f.persistence))
                for f in reference_run if f.lead > 1
            ]
            added = add_model(output, persistence["run"], persistence["stage"], persistence["name"], references, context)
            persistence["leads"] = sorted(persistence["leads"] + added["leads"])


def h_winners_from_compare(files, all_runs, evaluator):
    handoff = pd.read_csv(files["Tables/H_handoff.csv"])
    selection = pd.read_csv(files["Tables/H_demo_runs.csv"])
    key_cols = ["lõi", "cách", "hạt giống"]
    checkpoint_hashes = {"sha256 best_model.pt": "checkpoint_sha256",
                         "sha256 config.yml": "config_sha256",
                         "sha256 train_data_scaler.yml": "scaler_sha256"}
    handoff_columns = {*key_cols, *checkpoint_hashes, "notebook", "thư mục run", "epoch chọn",
                       "library_commit", "forecast_input_mode", "số tham số", "demo", "NPZ"}
    selection_columns = {*key_cols, "epoch chọn", "NSE_lead_mean val", "notebook"}
    runs = {}
    for run in all_runs:
        match = evaluator["H_NAME"].fullmatch(f"{run}_test_S0_lead1.npz")
        key = (match["config"].split("_", 1)[0], match["mode"], int(match["seed"]))
        runs[key] = (run, match["config"])
    if (not handoff_columns <= set(handoff) or not selection_columns <= set(selection)
            or handoff.duplicated(key_cols).any()
            or set(handoff[key_cols].itertuples(index=False, name=None)) != set(runs)
            or not handoff["demo"].isin([True, False]).all()
            or selection.duplicated(key_cols[:2]).any()
            or set(selection[key_cols[:2]].itertuples(index=False, name=None)) != {key[:2] for key in runs}
            or set(selection[key_cols].itertuples(index=False, name=None))
            != set(handoff.loc[handoff["demo"].eq(True), key_cols].itertuples(index=False, name=None))):
        raise ValueError("Hai bảng H Compare thiếu run hoặc không cùng danh sách chọn demo.")

    rows = {}
    for row in handoff.to_dict("records"):
        key = tuple(row[col] for col in key_cols)
        run = runs[key][0]
        if (row["NPZ"] != f"contracts/{run}_test_S0_lead{{1..{evaluator['H_LEADS'][-1]}}}.npz"
                or any(not evaluator["digest_string"](row[field]) for field in checkpoint_hashes)):
            raise ValueError(f"Bàn giao H sai đường dẫn NPZ hoặc mã băm: {run}.")
        metadata = dict(config_sha256=row["sha256 config.yml"], library_commit=row["library_commit"],
                        selected_epoch=row["epoch chọn"], forecast_input_mode=row["forecast_input_mode"],
                        n_params=row["số tham số"], training_notebook=row["notebook"])
        if any(f.metadata.get(field) != value for f in all_runs[run] for field, value in metadata.items()):
            raise ValueError(f"Bàn giao H khác metadata NPZ: {run}.")
        rows[key] = (row, metadata)

    winners = {}
    for selected in selection.to_dict("records"):
        key = tuple(selected[col] for col in key_cols)
        run, configuration = runs[key]
        row, metadata = rows[key]
        score = float(selected["NSE_lead_mean val"])
        if (not np.isfinite(score) or selected["epoch chọn"] != row["epoch chọn"]
                or selected["notebook"] != row["notebook"]):
            raise ValueError(f"Bảng chọn H không khớp checkpoint bàn giao: {run}.")
        checkpoint = dict(metadata, run_directory=row["thư mục run"],
                          **{dest: row[field] for field, dest in checkpoint_hashes.items()
                             if dest != "config_sha256"})
        winners[(configuration, key[1])] = dict(run=run, seed=int(key[2]), score=score, checkpoint=checkpoint)
    return winners


def h_winners(source, all_runs, audit, evaluator, checked):
    files = dict(zip(checked["file"], map(Path, checked["path"])))
    handoff_names = {"Tables/H_handoff.csv", "Tables/H_demo_runs.csv"}
    if handoff_names & files.keys():
        if not handoff_names <= files.keys():
            raise ValueError("Manifest H Compare phải có cả Tables/H_handoff.csv và Tables/H_demo_runs.csv.")
        return h_winners_from_compare(files, all_runs, evaluator)

    # Giữ tương thích với nguồn chọn seed trước khi H Compare có hai bảng bàn giao.
    selection = audit["unique_file"](source, "H_demo_selection.csv", required=False)
    candidates = {}
    if selection is not None:
        files, _ = audit["verified_manifest"](selection.parent)
        table = pd.read_csv(files[selection.name])
        if table["run"].duplicated().any() or set(table["run"]) != set(all_runs):
            raise ValueError(f"Bảng chọn H phải có đủ {len(all_runs)} run.")
        for r in table.itertuples(index=False):
            f = all_runs[r.run][0]
            name = f"SF_{r.configuration}_H{r.mode}_seed{r.seed}"
            if (name != r.run or f.seed != r.seed or r.config_sha256 != f.metadata["config_sha256"]
                    or not np.isfinite(r.validation_NSE_lead_mean)):
                raise ValueError(f"Bảng chọn H khác hợp đồng: {r.run}")
            candidates[r.run] = dict(run=r.run, key=(r.configuration,r.mode), seed=r.seed,
                                     score=r.validation_NSE_lead_mean)
    else:
        candidates = h_candidates_from_training(source, all_runs, audit)
    if set(candidates) != set(all_runs):
        raise ValueError(f"Cần bảng bàn giao H Compare hoặc Output huấn luyện đủ {len(all_runs)} run.")
    winners = {}
    for candidate in sorted(candidates.values(), key=lambda row: (-row["score"], row["seed"])):
        winners.setdefault(candidate["key"], candidate)
    return winners


def h_candidates_from_training(source, all_runs, audit):
    candidates = {}
    expected_prefix = "H_"
    for path in source.rglob("environment.json"):
        if path.parent.name != "h_output":
            continue
        env = audit["read_json"](path)
        summary = env["summary"]
        exp = env["experiment_name"]
        seed, mode = summary["seed"], summary["mode"]
        suffix = f"_{mode}_seed{seed}"
        if not exp.startswith(expected_prefix) or not exp.endswith(suffix) or summary["status"] != "xong":
            continue
        conf = exp[len(expected_prefix):-len(suffix)]
        run = f"SF_{conf}_H{mode}_seed{seed}"
        if run not in all_runs:
            continue
        if (run in candidates or env["config_sha256"] != all_runs[run][0].metadata["config_sha256"]
                or env["neuralhydrology_commit"] != all_runs[run][0].metadata["library_commit"]
                or summary["select_metric"] != "NSE_lead_mean"):
            raise ValueError(f"Output chọn H khác hợp đồng {run}.")
        score = summary["val_nse_lead_mean_median_best_model"]
        if not np.isfinite(score):
            raise ValueError(f"Validation H không hữu hạn: {run}")
        candidates[run] = dict(run=run, key=(conf, mode), seed=seed, score=score)
    return candidates


def live_c2_sources(source, compare, winners, audit, replay_only_cores):
    # 5 lõi chạy trực tiếp, chọn cấu hình/seed tốt nhất trong từng lõi theo validation.
    live_records = {}
    for record in winners:
        live_records.setdefault(record["cfg"]["core"], record)
    d2 = audit["unique_file"](source, "D2_summary.json").parent
    d2_files, _ = audit["verified_manifest"](d2)
    summary = audit["read_json"](d2_files["D2_summary.json"])
    if (summary.get("complete") is not True or summary.get("library_commit") != audit["FREEZE_C"]
            or summary.get("C2_compare_manifest_sha256") != audit["sha256"](compare/"manifest.csv")):
        raise ValueError("D2 không khớp nguồn C2.")
    benchmark = pd.read_csv(d2_files["D2_CPU_benchmark.csv"]).set_index("run", verify_integrity=True)
    passed = [r for core,r in live_records.items() if core not in replay_only_cores
              and benchmark.loc[audit["record_run"](r), "status"] == "verified"]
    sources = audit["c2_train_sources"](source, passed)
    for run, files in sources.items():
        if benchmark.loc[run, "checkpoint_sha256"] != audit["sha256"](files["best_model.pt"]):
            raise ValueError(f"Trọng số đóng gói khác D2: {run}")
    return sources, d2_files["D2_CPU_benchmark.csv"]


def package_live_c2(source, output, compare, winners, models, audit, replay_only_cores):
    sources, benchmark_path = live_c2_sources(source, compare, winners, audit, replay_only_cores)
    shutil.copy2(benchmark_path, output/"D2_CPU_benchmark.csv")
    for item in models:
        run = item["run"]
        if run not in sources:
            continue
        dest = output/"checkpoints"/run
        for name, path in sources[run].items():
            (dest/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest/name)
        item["what_if"] = True


def package_live_h(source, output, h_source, raw, python, models, context, audit):
    selected = [m for m in models if m["stage"] == "H" and m["mode"] == "hybrid"]
    if len(selected) != 2 or {m["core"] for m in selected} != {"lstm", "mamba"}:
        raise ValueError("Cần hai checkpoint H cách lai được chọn trên validation.")
    inputs = audit["read_json"](h_source/"environment.json")["inputs"]["H"]
    env_by_hash = {}
    for path in source.rglob("h_output/environment.json"):
        env_by_hash.setdefault(audit["sha256"](path), []).append(path)
    for m in selected:
        expected = inputs[f"{m['core']}_hybrid_seed{m['seed']}"]
        paths = env_by_hash.get(expected["environment_sha256"], [])
        if len(paths) != 1:
            raise FileNotFoundError(f"Cần đúng một Output Version 1 của {m['training_notebook']}.")
        path = paths[0]
        env = audit["read_json"](path)
        summary = env["summary"]
        if (env["notebook"] != m["training_notebook"] or env["notebook"] != expected["notebook"]
                or env["run_dir"] != m["run_directory"] or env["neuralhydrology_commit"] != FREEZE_H
                or env["config_sha256"] != m["config_sha256"] or summary["status"] != "xong"
                or summary["saved_epoch"] != m["selected_epoch"] or summary["n_params"] != m["n_params"]
                or summary["seed"] != m["seed"] or summary["forecast_input_mode"] != "first_day"):
            raise ValueError(f"Output H Train khác bàn giao: {m['training_notebook']}.")
        manifest = pd.read_csv(path.parent/"manifest.csv").set_index("tệp", verify_integrity=True)["sha256"]
        for filename, key in H_CHECKPOINT_FILES.items():
            relpath = f"{m['run_directory']}/{filename}"
            checkpoint = audit["contained_file"](path.parent.parent, relpath)
            if manifest.get(relpath) != m[key] or audit["sha256"](checkpoint) != m[key]:
                raise ValueError(f"Checkpoint khác SHA-256 đã bàn giao: {relpath}.")
            dest = output/"checkpoints"/m["run"]/filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint, dest)

    worker = HWorker(python, output, raw, context["ids"], context["area"], selected)
    try:
        rows = [check_live_h(worker, h_source, m, context) for m in selected]
    finally:
        worker.close()
    pd.DataFrame(rows).to_csv(output/"H_CPU_check.csv", index=False)
    for m, row in zip(selected, rows):
        m.update(what_if=True, origin_start=row["origin_start"], origin_end=row["origin_end"])


def check_live_h(worker, h_source, model, context):
    run = model["run"]
    ledgers = {}
    for lead in model["leads"]:
        file = h_source/"contracts"/f"{run}_test_S0_lead{lead}.npz"
        with np.load(file, allow_pickle=False) as source:
            days = source["origin_dates"]
            ledgers[lead] = (days, source["sim_raw_mm_day"], source["sim_mm_day"])
    first = max(days[0] for days, _, _ in ledgers.values())
    last = min(days[-1] for days, _, _ in ledgers.values())
    dates = sorted({str(first), str(last), "2016-02-29", "2016-06-15",
                    str(first + (last - first)//4), str(last - (last - first)//4)})
    ids = list(context["ids"])
    raw_errors, clipped_errors, warm_ms = [], [], []
    for basin in ("2", "6", "130", "876"):
        index = ids.index(basin)
        for number, date in enumerate(dates):
            before = perf_counter()
            baseline = worker.predict(run, basin, date)["baseline"]
            elapsed = (perf_counter() - before)*1000
            if not raw_errors:
                first_ms = elapsed
            if number:
                warm_ms.append(elapsed)
            if [row["lead"] for row in baseline] != model["leads"]:
                raise ValueError(f"CPU thiếu hạn dự báo: {run}, {basin}, {date}.")
            for row in baseline:
                lead = row["lead"]
                day = np.datetime64(date, "D")
                days, raw, clipped = ledgers[lead]
                loc = np.searchsorted(days, day)
                target = str(day + np.timedelta64(lead - 1, "D"))
                if days[loc] != day or row["origin_date"] != date or row["target_date"] != target:
                    raise ValueError(f"Lịch CPU khác NPZ: {run}, hạn {lead}.")
                raw_errors.append(compare_arrays(np.array([row["qmax_raw_mm_day"]]), raw[loc:loc+1, index], H_PRED_ATOL, PRED_RTOL))
                clipped_errors.append(compare_arrays(np.array([row["qmax_mm_day"]]), clipped[loc:loc+1, index], H_PRED_ATOL, PRED_RTOL))
    scenario = worker.predict(run, "6", "2016-06-15", rain_multiplier=1.5, temperature_delta=2)
    if not any(a["qmax_raw_mm_day"] != b["qmax_raw_mm_day"] for a, b in zip(scenario["baseline"], scenario["changed"])):
        raise ValueError(f"CPU không phản hồi khi thay đổi ECMWF HRES: {run}.")
    row = dict(run=run, seed=model["seed"], library_commit=FREEZE_H, checkpoint_sha256=model["checkpoint_sha256"],
               origin_start=str(first), origin_end=str(last), windows=4*len(dates), lead_checks=len(raw_errors),
               atol_mm_day=H_PRED_ATOL, rtol=PRED_RTOL, max_raw_abs_mm_day=max(raw_errors), max_clipped_abs_mm_day=max(clipped_errors),
               first_ms=first_ms, warm_p50_ms=float(np.median(warm_ms)), warm_p95_ms=float(np.percentile(warm_ms, 95)))
    print(f"{run}: {row['lead_checks']} dự báo CPU khớp NPZ, p95 sau nạp lưu vực {row['warm_p95_ms']:.1f} ms.", flush=True)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đóng gói dự báo và bảng đã lưu cho demo laptop.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inputs", action="store_true", help="Liệt kê nguồn còn thiếu trước khi đóng gói.")
    parser.add_argument("--h-evaluation", type=Path,
                        help="Thư mục H/ của Evaluation, yêu cầu kèm H Compare trong --inputs.")
    parser.add_argument("--d-evaluation", type=Path,
                        help="Thư mục D/ của Evaluation, yêu cầu kèm D Compare trong --inputs.")
    parser.add_argument("--h-python", type=Path, help="Python cài freeze-H để kiểm và bật CPU cho hai run H cách lai.")
    parser.add_argument("--with-live", action="store_true")
    parser.add_argument("--replay-only", choices=("lstm", "gru", "transformer", "s4d", "mamba"),
                        action="append", default=[], help="Lõi đã chọn chỉ phát lại sau khi xem thời gian D2.")
    args = parser.parse_args()
    if args.check_inputs:
        sys.exit(0 if check_inputs(args.inputs, args.with_live, args.replay_only,
                                   args.h_evaluation, args.h_python, args.d_evaluation) else 2)
    if args.output is None:
        parser.error("Cần --output khi đóng gói.")
    prepare(args.inputs, args.output, args.with_live, args.replay_only,
            args.h_evaluation, args.h_python, args.d_evaluation)
