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


def audit_prediction(series, ref, index):
    prediction, observation = align_basin_lead(*series, ref.dates, ref.lead)
    result = dict(days=len(ref.dates),
                max_obs_abs_error=compare_arrays(observation, ref.obs[:, index], 1e-5, 1e-5),
                max_raw_abs_error=compare_arrays(prediction, ref.raw[:, index], PRED_ATOL, PRED_RTOL),
                max_clip_abs_error=compare_arrays(np.maximum(prediction, 0), ref.sim[:, index], PRED_ATOL, PRED_RTOL))
    return prediction, result
