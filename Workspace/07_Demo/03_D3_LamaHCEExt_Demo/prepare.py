import argparse
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
import shapefile
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import transform, unary_union


HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
sys.path.insert(0, str(DEMO / "DemoRuntime"))
from forecast import qmax_m3_s

H_REPORT_TABLES = ("summary", "metrics", "events", "levels", "event_cases")
H_AUDIT_COLUMNS = ["file", "bytes", "sha256", "passed", "run", "mode", "lead", "library_commit"]
H_MODE_LABELS = {"past": "Chỉ dữ liệu quá khứ", "hybrid": "ECMWF HRES ngày đầu",
                 "gefs": "ECMWF HRES và GEFS", "perfect": "Thời tiết tái phân tích (tham chiếu lý tưởng)"}


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


def check_inputs(source, with_live=False, replay_only_cores=(), h_evaluation=None):
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
        check(label, lambda: audit["unique_file"](source, filename))
    check("A2 V2", lambda: locate_a2(source))
    if with_live:
        check("D2 V1", lambda: audit["unique_file"](source, "D2_summary.json"))
    if not problems:
        checked = check("C2 Compare V1 và A3", lambda: audit["compare_inputs"](source))
        if with_live and checked is not None:
            _, compare, records = checked
            check("Checkpoint chạy trực tiếp",
                  lambda: live_c2_sources(source, compare, audit["c2_winners"](records), audit, replay_only_cores))
    if h_evaluation is not None:
        evaluator = load_evaluator()
        check("H Compare", lambda: require_h_compare(source, audit, evaluator))
        check("Evaluation H", lambda: h_report_files(h_evaluation, audit, evaluator))
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


def prepare(source, output, with_live=False, replay_only_cores=(), h_evaluation=None):
    evaluator = load_evaluator()
    audit = load_audit()
    h_source, h_tables, h_checked = source, {}, None
    if h_evaluation is not None:
        h_source = require_h_compare(source, audit, evaluator)
        h_tables, h_checked = h_report_files(h_evaluation, audit, evaluator)
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

    for label, pattern in {
        "metrics": "Evaluation_metrics.csv", "levels": "Evaluation_levels.csv",
        "events": "Evaluation_events.csv", "events_by_case": "Evaluation_event_cases.csv",
        "c2_scenarios": "C2_scenarios_stats.csv",
        "D": "D_score_stats.csv", "D_cost": "D_cost.csv", "D_basin": "D_basin_scores.csv",
        "G": "G_IntegratedGradients.csv",
    }.items():
        path = audit["unique_file"](source, pattern, required=False)
        if path is not None:
            shutil.copy2(path, output/f"{label}.csv")
    if not h_tables:
        for lead in evaluator["H_LEADS"]:
            for label in H_REPORT_TABLES:
                name = f"H_{label}_lead{lead}.csv"
                path = audit["unique_file"](source, f"Evaluation_{name}", required=False)
                if path is not None:
                    h_tables[name] = path
    for name, path in h_tables.items():
        shutil.copy2(path, output/name)
    selected = next(m["run"] for m in models if m["run"] == audit["C2_DEFAULT"])
    audit["write_json"](summary_file, dict(schema=1, test_start=str(context["dates"][0].date()),
                                        dates=context["dates"].strftime("%Y-%m-%d").tolist(), ids=ids, areas=areas,
                                        points=gauge_points(raw, ids, evaluator["basin_id"]),
                                        outline=study_outline(raw, ids, evaluator["basin_id"]),
                                        flood_m3_s={i: {str(k):float(v) for k,v in levels.loc[i].items()} for i in ids},
                                        models=models, default_run=selected, library_commit=audit["FREEZE_C"],
                                        C2_manifest_sha256=audit["sha256"](compare/"manifest.csv")))
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
        for r in winners.itertuples(index=False):
            run = f"SF_{r.configuration}_D_seed{r.seed}"
            f = evaluator["read_contract"](files[f"contracts/{run}_test_S0_lead1.npz"])
            if f.name != run or f.lead != 1 or f.metadata["library_commit"] != audit["FREEZE_C"]:
                raise ValueError(f"Hợp đồng D sai danh tính: {run}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đóng gói dự báo và bảng đã lưu cho demo laptop.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inputs", action="store_true", help="Liệt kê nguồn còn thiếu trước khi đóng gói.")
    parser.add_argument("--h-evaluation", type=Path,
                        help="Thư mục H/ của Evaluation, yêu cầu kèm H Compare trong --inputs.")
    parser.add_argument("--with-live", action="store_true")
    parser.add_argument("--replay-only", choices=("lstm", "gru", "transformer", "s4d", "mamba"),
                        action="append", default=[], help="Lõi đã chọn chỉ phát lại sau khi xem thời gian D2.")
    args = parser.parse_args()
    if args.check_inputs:
        sys.exit(0 if check_inputs(args.inputs, args.with_live, args.replay_only, args.h_evaluation) else 2)
    if args.output is None:
        parser.error("Cần --output khi đóng gói.")
    prepare(args.inputs, args.output, args.with_live, args.replay_only, args.h_evaluation)
