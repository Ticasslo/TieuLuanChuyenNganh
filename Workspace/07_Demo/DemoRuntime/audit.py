from pathlib import Path, PurePosixPath
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata


MODEL_FILES = ("config.yml", "best_model.pt", "train_data/train_data_scaler.yml")
C2_DEFAULT = "SF_gru_L2_x1_lr1e-3_C2_seed111"


def install_library(commit, mamba_cpu=False):
    packages = ("ruamel.yaml", "tensorboard", "psutil") if mamba_cpu else ("ruamel.yaml", "tensorboard")
    dependencies = [p for p in packages if not package_installed(p)]
    if mamba_cpu:
        dependencies.append("mambapy==1.2.0")
    if dependencies:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *dependencies])
    if installed_commit() != commit:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "--quiet",
                               f"git+https://github.com/Ticasslo/neuralhydrology.git@{commit}"])
    if installed_commit() != commit:
        raise RuntimeError("Bản thư viện đang cài khác commit đã ghim.")


def package_installed(name):
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def installed_commit():
    try:
        source = metadata.distribution("neuralhydrology").read_text("direct_url.json")
        return json.loads(source or "{}").get("vcs_info", {}).get("commit_id")
    except metadata.PackageNotFoundError:
        return None


def install_mamba_cuda(input_root):
    import torch

    path = unique_file(input_root, "build_info.json")
    info = read_json(path)
    expected = {key: info[key] for key in ("python", "torch", "torch_cuda")}
    expected["gpu_cc"] = (7, 5)
    actual = dict(python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda,
                  gpu_cc=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
    mismatches = [f"{key}: đang là {value!r}, wheel cần {expected[key]!r}"
                  for key, value in actual.items() if value != expected[key]]
    if mismatches:
        raise ValueError(f"Môi trường D1 khác wheel từ {path}: " + " | ".join(mismatches)
                         + ". Chạy bằng ảnh T4 cu128 của notebook C2 đã lưu.")
    wheels = [contained_file(path.parent, name) for name in info["wheels"]]
    if any(sha256(p) != info["wheels"][p.name] for p in wheels):
        raise ValueError("SHA-256 wheel Mamba sai.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", *map(str, wheels)])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def locate_raw(root):
    candidates = {p.parents[2] for p in root.rglob("Gauge_attributes.csv")
                  if p.parent.name == "1_attributes" and p.parent.parent.name == "D_gauges"
                  and (p.parents[2] / "A_basins_total_upstrm/1_attributes/Catchment_attributes.csv").is_file()}
    if len(candidates) != 1:
        raise ValueError(f"Cần đúng một thư mục dữ liệu LamaH Extended, tìm thấy {len(candidates)}.")
    return candidates.pop()


def compare_inputs(input_root):
    a3 = unique_file(input_root, "A3_persistence_predictions.npz").parent
    context, paths = read_context(a3)
    if len(context["ids"]) != 451:
        raise ValueError("Cần đúng 451 lưu vực theo mốc A3.")
    compare = unique_file(input_root, "c2_summary.json").parent
    records, _, _, _ = c2_records(compare, paths)
    return context, compare, records


def record_run(record):
    return f"SF_{record['conf']}_C2_seed{record['seed']}"


def c2_train_sources(root, records):
    env_paths = {}
    for path in root.rglob("environment.json"):
        if path.parent.name == "c_output":
            env_paths.setdefault(sha256(path), []).append(path)
    sources = {}
    for record in records:
        handoff = record["handoff"]
        candidates = env_paths.get(record["source"]["environment_sha256"], [])
        if len(candidates) != 1:
            raise FileNotFoundError(f"Cần đúng một Output Version 1 của {record['train_notebook']}.")
        path = candidates[0]
        env = read_json(path)
        if (env["run_dir"] != handoff["thư mục run"] or env["notebook"] != record["train_notebook"]
                or env["neuralhydrology_commit"] != FREEZE_C or env["summary"]["status"] != "xong"
                or env["summary"]["saved_epoch"] != record["epoch"]
                or env["summary"]["n_params"] != record["cfg"]["n_params"]
                or env["config_sha256"] != handoff["sha256 config.yml"]):
            raise ValueError(f"Nguồn huấn luyện khác bàn giao: {record['train_notebook']}")
        run = env["run_dir"]
        files = {name: contained_file(path.parent.parent, f"{run}/{name}") for name in MODEL_FILES}
        if any(sha256(file) != handoff[f"sha256 {file.name}"] for file in files.values()):
            raise ValueError(f"Checkpoint, config hoặc scaler sai hash: {record['train_notebook']}")
        sources[record_run(record)] = files
    return sources


def export_manifest(root, paths):
    table = pd.DataFrame(dict(file=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha256(p)) for p in paths)
    table.to_csv(root / "manifest.csv", index=False)


def c2_winners(records):
    # Mỗi cấu hình lấy một seed theo validation. Tie-break seed nhỏ hơn.
    winners = {}
    for record in sorted(records, key=lambda r: (-r["seed_info"]["val"], r["seed"], r["conf"])):
        winners.setdefault(record["conf"], record)
    return list(winners.values())


def d_demo_sources(input_root, context, c2_compare_dir):
    summary_path = unique_file(input_root, "d_summary.json", required=False)
    if summary_path is None:
        return None, [], {}
    d_root = summary_path.parent
    files, _ = verified_manifest(d_root)
    required = {"d_summary.json", "D_handoff.csv", "D_demo_runs.csv"}
    if not required.issubset(files):
        raise ValueError(f"{D_COMPARE_NOTEBOOK}: thiếu tệp bàn giao {sorted(required - set(files))}.")
    summary = read_json(files["d_summary.json"])
    s4d_inc = summary.get("S4D_included")
    if not isinstance(s4d_inc, bool):
        raise ValueError("d_summary.json thiếu cờ boolean S4D_included.")
    expected_confs = list(D_CONFIGS if s4d_inc else D_CONFIGS[:-1])
    inputs = summary.get("inputs") or {}
    if (summary.get("complete") is not True or summary.get("notebook") != D_COMPARE_NOTEBOOK
            or summary.get("step") != "D" or summary.get("library_commit") != FREEZE_C
            or summary.get("configs") != expected_confs
            or inputs.get("C2_manifest_sha256") != sha256(Path(c2_compare_dir) / "manifest.csv")):
        raise ValueError("Bàn giao D Compare chưa hoàn tất hoặc khác C2 Compare / freeze-C.")
    handoff = pd.read_csv(files["D_handoff.csv"]).set_index(["configuration", "seed"], verify_integrity=True)
    demo_runs = pd.read_csv(files["D_demo_runs.csv"])
    if len(demo_runs) != len(expected_confs) or set(demo_runs["configuration"]) != set(expected_confs):
        raise ValueError("D_demo_runs.csv không khớp danh sách cấu hình D.")
    env_paths = {}
    for path in Path(input_root).rglob("environment.json"):
        if path.parent.name == "c_output":
            env_paths.setdefault(sha256(path), []).append(path)
    ref_base = Forecast(name="A3", sim=context["persistence"], raw=context["persistence"], **context).validate()
    records, sources = [], {}
    for _, demo_row in demo_runs.iterrows():
        conf, seed = str(demo_row["configuration"]), int(demo_row["seed"])
        h_row = handoff.loc[(conf, seed)]
        run = str(h_row["run"])
        if run != str(demo_row["run"]) or str(h_row["demo"]).lower() != "true":
            raise ValueError(f"D_demo_runs.csv lệch D_handoff.csv tại {run}.")
        rel = str(h_row["contract_npz"])
        if rel not in files or sha256(files[rel]) != str(h_row["sha256_contract_npz"]):
            raise ValueError(f"NPZ D thiếu hoặc sai SHA-256: {rel}")
        ref = read_contract(files[rel])
        check_same_context(ref_base, ref)
        ref.obs, ref.persistence, ref.area = context["obs"], context["persistence"], context["area"]
        candidates = env_paths.get(str(h_row["sha256_environment_json"]), [])
        if len(candidates) != 1:
            raise FileNotFoundError(f"Cần gắn đúng một Output huấn luyện của {h_row['notebook']} cho {run}.")
        env_path = candidates[0]
        env = read_json(env_path)
        run_dir = str(h_row["run_dir"])
        if (env.get("run_dir") != run_dir or env.get("notebook") != str(h_row["notebook"])
                or env.get("neuralhydrology_commit") != FREEZE_C
                or env.get("summary", {}).get("status") != "xong"
                or env.get("summary", {}).get("seq_length") != 730
                or env.get("summary", {}).get("saved_epoch") != int(h_row["selected_epoch"])
                or env.get("summary", {}).get("n_params") != int(h_row["n_params"])
                or env.get("config_sha256") != str(h_row["sha256_config_yml"])):
            raise ValueError(f"Nguồn huấn luyện D khác D_handoff.csv: {h_row['notebook']}")
        sha_keys = {"config.yml": "sha256_config_yml", "best_model.pt": "sha256_best_model_pt",
                    "train_data/train_data_scaler.yml": "sha256_train_data_scaler_yml"}
        model_files = {name: contained_file(env_path.parent.parent, f"{run_dir}/{name}") for name in MODEL_FILES}
        if any(sha256(f) != str(h_row[sha_keys[name]]) for name, f in model_files.items()):
            raise ValueError(f"Checkpoint, config hoặc scaler D sai SHA-256: {h_row['notebook']}")
        sources[run] = model_files
        records.append(dict(
            run=run, conf=conf, core=str(h_row["core"]), seed=seed, epoch=int(h_row["selected_epoch"]),
            n_params=int(h_row["n_params"]), seq_length=730, contract=files[rel], forecast=ref,
            notebook=str(h_row["notebook"]),
        ))
    return d_root, records, sources


def audit_prediction(series, ref, index):
    prediction, observation = align_basin_lead(*series, ref.dates, ref.lead)
    result = dict(days=len(ref.dates),
                max_obs_abs_error=compare_arrays(observation, ref.obs[:, index], 1e-5, 1e-5),
                max_raw_abs_error=compare_arrays(prediction, ref.raw[:, index], PRED_ATOL, PRED_RTOL),
                max_clip_abs_error=compare_arrays(np.maximum(prediction, 0), ref.sim[:, index], PRED_ATOL, PRED_RTOL))
    return prediction, result
