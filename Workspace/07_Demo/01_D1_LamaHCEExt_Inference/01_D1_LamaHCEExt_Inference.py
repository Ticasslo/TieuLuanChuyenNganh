# %% [markdown]
# # 01_D1_LamaHCEExt_Inference
#
# T4 cu128, Internet On. 24 checkpoint C2, A3, C2 Compare và wheel Mamba.
# Hướng dẫn Input và nghiệm thu: 01_D1_LamaHCEExt_Inference.md.

# %% Phần 1 - Nguồn, manifest và hợp đồng C2
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

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
import itertools
import math
import re

METRICS = ("NSE", "PNSE", "KGE", "r", "alpha", "beta")

FREEZE_C = "db6dc1ffcd8c14994cd80be3ed8ff417002c01a1"

C2_NOTEBOOK = "13_C2_LamaHCEExt_Compare"

C2_CONFIGS = ("lstm_L2_x1_lr1e-3", "gru_L2_x1_lr1e-3", "transformer_L2_x1_lr1e-3",
              "transformer_L2_x3_lr5e-4", "s4d_L4_x1_lr1e-3", "s4d_L4_x3_lr1e-3",
              "mamba_L2_x1_lr1e-3", "mamba_L4_x3_lr1e-3")

C2_SEEDS = (111, 222, 333)

LOG_FLOOR_MM_DAY = 1e-6

TEST_START, TEST_END = "2014-01-01", "2017-12-31"

SUPPLEMENTARY_EVENT_TOLERANCE_DAYS = 1

def primary_event_tolerance_days(lead, forecast_horizon=None):
    """Dung sai của chính hạn đang chấm; tổng số hạn chỉ kiểm hợp đồng."""
    if not isinstance(lead, int) or isinstance(lead, bool) or not 1 <= lead <= 7:
        raise ValueError("lead phải là số nguyên 1–7")
    horizon = lead if forecast_horizon is None else forecast_horizon
    if (not isinstance(horizon, int) or isinstance(horizon, bool)
            or not lead <= horizon <= 7):
        raise ValueError("forecast_horizon phải là số nguyên từ lead đến 7")
    return lead // 3

def basin_id(value):
    value = str(value).strip()
    if not value.isdecimal():
        raise ValueError(f"ID không phải số nguyên không âm: {value!r}")
    return str(int(value))

@dataclass
class Forecast:
    name: str
    dates: pd.DatetimeIndex
    ids: tuple
    area: np.ndarray
    obs: np.ndarray
    sim: np.ndarray
    persistence: np.ndarray
    raw: np.ndarray
    period: str = "test"
    scenario: str = "S0"
    lead: int = 1
    seed: int | None = None
    native_obs: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def validate(self):
        self.dates = pd.DatetimeIndex(self.dates).as_unit("ns")
        if (len(self.dates) == 0 or self.dates.hasnans or self.dates.has_duplicates
                or self.dates.tz is not None or not self.dates.equals(self.dates.normalize())
                or not self.dates.equals(pd.date_range(self.dates[0], self.dates[-1], freq="D"))):
            raise ValueError("Cần lịch ngày liên tục, duy nhất, tăng dần, không múi giờ")
        normalized = tuple(basin_id(x) for x in self.ids)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("ID rỗng hoặc trùng sau chuẩn hóa")
        self.ids = normalized
        if "forecast_horizon" in self.metadata and self.metadata["forecast_horizon"] is None:
            raise ValueError("forecast_horizon khai báo trong metadata phải là số nguyên")
        primary_event_tolerance_days(self.lead, self.metadata.get("forecast_horizon"))
        if not self.name or not self.period or not self.scenario:
            raise ValueError("Thiếu tên, kỳ hoặc kịch bản")
        if any(re.fullmatch(r"[A-Za-z0-9_-]+", value) is None for value in (self.period, self.scenario)):
            raise ValueError("Tên kỳ/kịch bản chỉ gồm chữ ASCII, số, gạch dưới hoặc gạch nối")
        self.area = np.asarray(self.area, dtype=np.float64)
        if self.area.shape != (len(self.ids),) or not np.isfinite(self.area).all() or (self.area <= 0).any():
            raise ValueError("area_gov phải hữu hạn, dương và khớp ID")
        shape = (len(self.dates), len(self.ids))
        for name in ("obs", "sim", "persistence", "raw", "native_obs"):
            value = getattr(self, name)
            if value is None and name == "native_obs":
                continue
            value = np.asarray(value, dtype=np.float64)
            if value.shape != shape or np.isinf(value).any():
                raise ValueError(f"{name}: sai kích thước hoặc có vô cực")
            if name in {"obs", "persistence", "native_obs"} and (value[np.isfinite(value)] < 0).any():
                raise ValueError(f"{name}: có quan trắc âm sau hợp đồng loader")
            setattr(self, name, value)
        return self

    @property
    def group(self):
        return self.period, self.scenario, self.lead

def check_same_context(left, right):
    if (not left.dates.equals(right.dates) or left.ids != right.ids
            or not np.array_equal(left.area, right.area)):
        raise ValueError("Các dự báo so sánh khác lịch, ID hoặc diện tích")
    for key in ("obs", "persistence"):
        if not np.array_equal(getattr(left, key), getattr(right, key), equal_nan=True):
            raise ValueError(f"Khác {key} chuẩn; không tự thay chuỗi hoặc nới mặt nạ")

def metrics(obs, sim, persistence=None, mask=None):
    arrays = [np.asarray(x, dtype=np.float64) for x in (obs, sim)]
    if persistence is not None:
        arrays.append(np.asarray(persistence, dtype=np.float64))
    if any(x.ndim != 1 or x.shape != arrays[0].shape for x in arrays):
        raise ValueError("Chuỗi phải 1D và cùng kích thước")
    if any(np.isinf(x).any() for x in arrays):
        raise ValueError("Có vô cực trong chuỗi")
    if mask is None:
        mask = np.logical_and.reduce([np.isfinite(x) for x in arrays])
    else:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != arrays[0].shape:
            raise ValueError("mask phải boolean cùng kích thước")
        if any(not np.isfinite(x[mask]).all() for x in arrays):
            raise ValueError("mask chọn ngày không hữu hạn")
    result = {key: np.nan for key in METRICS}
    result.update({key + "_status": "no_valid_pairs" for key in METRICS})
    result.update(n=int(mask.sum()), sse_model=np.nan, sse_persistence=np.nan, obs_tss=np.nan)
    if not result["n"]:
        return result
    observed, predicted = [x[mask] for x in arrays[:2]]
    obs_constant = bool(np.all(observed == observed[0]))
    sim_constant = bool(np.all(predicted == predicted[0]))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mo = float(observed[0]) if obs_constant else float(observed.mean())
        ms = float(predicted[0]) if sim_constant else float(predicted.mean())
        co, cs = observed - mo, predicted - ms
        vo = 0.0 if obs_constant else float(np.sum(co ** 2))
        vs = 0.0 if sim_constant else float(np.sum(cs ** 2))
        sse = float(np.sum((predicted - observed) ** 2))
        result.update(sse_model=sse, obs_tss=vo)
        if vo > 0:
            result.update(NSE=1 - sse / vo, NSE_status="ok", alpha=math.sqrt(vs / vo), alpha_status="ok")
        else:
            result.update(NSE_status="obs_zero_variance", alpha_status="obs_zero_variance")
        if mo != 0:
            result.update(beta=ms / mo, beta_status="ok")
        else:
            result["beta_status"] = "obs_mean_zero"
        if result["n"] < 2:
            result["r_status"] = "insufficient_pairs"
        elif vo == 0:
            result["r_status"] = "obs_zero_variance"
        elif vs == 0:
            result["r_status"] = "sim_constant"
        else:
            r = float(np.dot(co / math.sqrt(vo), cs / math.sqrt(vs)))
            result.update(r=float(np.clip(r, -1, 1)), r_status="ok")
        missing = [key for key in ("r", "alpha", "beta") if result[key + "_status"] != "ok"]
        if missing:
            result["KGE_status"] = "undefined_components:" + ",".join(missing)
        else:
            result.update(KGE=1 - math.sqrt(sum((result[key] - 1) ** 2 for key in ("r", "alpha", "beta"))), KGE_status="ok")
        if persistence is None:
            result["PNSE_status"] = "persistence_not_provided"
        else:
            sp = float(np.sum((arrays[2][mask] - observed) ** 2))
            result["sse_persistence"] = sp
            if sp > 0:
                result.update(PNSE=1 - sse / sp, PNSE_status="ok")
            else:
                result["PNSE_status"] = "persistence_sse_zero"
    if any(not np.isfinite(result[key]) for key in METRICS if result[key + "_status"] == "ok"):
        raise FloatingPointError("Chỉ số khai báo ok nhưng không hữu hạn")
    return result

def unique_file(root, pattern, required=True):
    found = sorted(Path(root).rglob(pattern))
    if len(found) > 1:
        raise ValueError(f"Có {len(found)} nguồn {pattern}; chỉ định thư mục riêng: {found}")
    if not found and required:
        raise FileNotFoundError(f"Thiếu {pattern} dưới {root}; Add Input Output đã Save Version")
    return found[0] if found else None

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_context(a3_root):
    path = unique_file(a3_root, "A3_persistence_predictions.npz")
    with np.load(path, allow_pickle=False) as data:
        dates = pd.DatetimeIndex(data["dates"]).as_unit("ns")
        if not dates.equals(pd.date_range(TEST_START, TEST_END)):
            raise ValueError("Ngữ cảnh A3 không đúng toàn kỳ test 2014–2017")
        if not np.array_equal(data["source_dates"], data["dates"] - np.timedelta64(1, "D")):
            raise ValueError("Sai ngày nguồn persistence")
        context = dict(dates=dates, ids=tuple(basin_id(x) for x in data["basin_ids"]),
                       obs=np.array(data["obs_source_mm_day"], dtype=float),
                       persistence=np.array(data["persistence_qmax_mm_day"], dtype=float))
    raw_path = unique_file(a3_root, "A3_Q_source_window.npz")
    with np.load(raw_path, allow_pickle=False) as data:
        if tuple(basin_id(x) for x in data["basin_ids"]) != context["ids"]:
            raise ValueError("ID nguồn Q khác persistence")
        source_dates = pd.DatetimeIndex(data["dates"]).as_unit("ns")
        if not source_dates.equals(pd.date_range("2013-12-31", TEST_END)):
            raise ValueError("Nguồn Q cần đủ lịch test cộng một ngày trước test")
        context["area"] = np.array(data["area_gov_km2"], dtype=float)
        q = np.array(data["qmax_numeric_m3_s"], dtype=float)
        q[q < 0] = np.nan
        physical = q * 86.4 / context["area"][None, :]
        for stored, derived in ((context["obs"], physical[1:]), (context["persistence"], physical[:-1])):
            if not np.allclose(stored, derived, rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError("Persistence/obs không khớp Q gốc hoặc diện tích")
    return context, [path, raw_path]

def read_contract(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"].item()))
        required = ("run", "period", "scenario", "lead", "forecast_horizon", "seed", "unit", "config_sha256", "library_commit", "clip_to_zero")
        if any(key not in meta for key in required) or meta["unit"] != "mm/day_area_gov":
            raise ValueError("Thiếu metadata hoặc sai đơn vị hợp đồng")
        if not isinstance(meta["clip_to_zero"], bool):
            raise ValueError("clip_to_zero phải boolean")
        if meta["forecast_horizon"] is None:
            raise ValueError("forecast_horizon trong hợp đồng phải là số nguyên")
        primary_event_tolerance_days(meta["lead"], meta["forecast_horizon"])
        dates = np.array(data["dates"], dtype="datetime64[D]")
        if (not np.array_equal(data["origin_dates"], dates - np.timedelta64(int(meta["lead"]) - 1, "D"))
                or not np.array_equal(data["persistence_source_dates"], dates - np.timedelta64(int(meta["lead"]), "D"))):
            raise ValueError("Sai origin hoặc persistence nhiều hạn")
        raw = np.array(data["sim_raw_mm_day"], dtype=float)
        sim = np.array(data["sim_mm_day"], dtype=float)
        expected = np.maximum(raw, 0) if meta["clip_to_zero"] else raw
        if not np.array_equal(sim, expected, equal_nan=True):
            raise ValueError("sim khác chính sách clip")
        result = Forecast(name=meta["run"], period=meta["period"], scenario=meta["scenario"], lead=meta["lead"],
                          seed=meta["seed"], dates=pd.DatetimeIndex(dates), ids=tuple(data["basin_ids"]),
                          area=np.array(data["area_gov_km2"]), obs=np.array(data["obs_mm_day"]),
                          persistence=np.array(data["persistence_mm_day"]), sim=sim, raw=raw, metadata=meta)
    return result.validate()

def digest_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

def contained_file(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError(f"Đường dẫn manifest sai: {name!r}")
    rel = PurePosixPath(name)
    root = Path(root).resolve()
    path = root.joinpath(*rel.parts).resolve()
    if (rel.is_absolute() or ".." in rel.parts or rel.as_posix() != name
            or not path.is_relative_to(root) or not path.is_file()):
        raise ValueError(f"Tệp không hợp lệ trong bàn giao: {name}")
    return path

def verified_manifest(root, manifest=None):
    root = Path(root)
    manifest = Path(manifest) if manifest is not None else root / "manifest.csv"
    table = pd.read_csv(manifest, dtype={"sha256": str})
    column = "file" if "file" in table else "tệp"
    if not {column, "sha256"} <= set(table) or table.empty or table[column].duplicated().any():
        raise ValueError(f"Manifest thiếu cột, rỗng hoặc trùng: {manifest}")
    files, checks = {}, []
    for row in table.to_dict("records"):
        name = row[column]
        path = contained_file(root, name)
        digest = sha256(path)
        size = path.stat().st_size
        if row["sha256"] != digest or ("bytes" in row and row["bytes"] != size):
            raise ValueError(f"Manifest sai SHA-256 hoặc bytes: {name}")
        files[name] = path
        checks.append(dict(file=name, path=str(path), bytes=size, sha256=digest, passed=True))
    return files, pd.DataFrame(checks)

def c2_records(root, context_paths=(), b3_root=None, b0_noise=None):
    root = Path(root)
    files, checks = verified_manifest(root)
    summary = json.loads(files["c2_summary.json"].read_text(encoding="utf-8"))
    env = json.loads(files["environment.json"].read_text(encoding="utf-8"))
    if (summary.get("complete") is not True or summary.get("step") != "C2" or summary.get("epochs") != 30
            or summary.get("seeds") != list(C2_SEEDS) or summary.get("library_commit") != FREEZE_C
            or env.get("notebook") != C2_NOTEBOOK or env.get("complete") is not True
            or env.get("library_commit_of_runs") != FREEZE_C):
        raise ValueError("C2 Compare sai giao thức hoặc commit")
    inputs = env["inputs"]
    if inputs["A3"]["version"] != 1 or any(inputs["A3"]["files"].get(p.name) != sha256(p) for p in context_paths):
        raise ValueError("C2 Compare dùng A3 khác Input đánh giá")
    if b3_root is not None and inputs["B3_Compare"] != {
        "version": 1, "b3_summary_sha256": sha256(Path(b3_root) / "b3_summary.json"),
        "manifest_sha256": sha256(Path(b3_root) / "manifest.csv")}:
        raise ValueError("C2 Compare dùng B3 khác Input đánh giá")
    if b0_noise is not None and not np.isclose(summary["bstar"]["noise_sd"], b0_noise, atol=1e-12, rtol=0):
        raise ValueError("C2 Compare dùng ngưỡng nhiễu B0 khác")
    if set(summary["configs"]) != set(C2_CONFIGS):
        raise ValueError("C2 Compare cần đúng tám cấu hình")
    handoff = pd.read_csv(files["Tables/C2_handoff.csv"], encoding="utf-8-sig")
    audits = pd.read_csv(files["Tables/C2_contract_check.csv"], encoding="utf-8-sig")
    keys = ["cấu hình", "hạt giống"]
    expected = set(itertools.product(C2_CONFIGS, C2_SEEDS))
    for table in (handoff, audits):
        if len(table) != 24 or set(table[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError("Bảng C2 không đủ đúng 24 cấu hình/seed")
    handoff, audits = handoff.set_index(keys), audits.set_index(keys)
    expected_names = {f"{c}_seed{s}": f"contracts/SF_{c}_C2_seed{s}_test_S0_lead1.npz" for c,s in expected}
    if (summary["contracts"] != expected_names or set(inputs["C2"]) != set(expected_names)
            or {n for n in files if n.endswith(".npz")} != set(expected_names.values())):
        raise ValueError("Danh sách NPZ C2 sai hoặc không đủ")
    records = []
    for conf in C2_CONFIGS:
        cfg = summary["configs"][conf]
        core, depth, level, lr = conf.split("_")
        if (cfg["core"] != core or cfg["depth"] != int(depth[1:]) or cfg["level"] != int(level[1:])
                or cfg["lr"] != float(lr[2:]) or cfg["n_seeds"] != 3
                or type(cfg["n_params"]) is not int or abs(cfg["n_params"] / (84865 * cfg["level"]) - 1) > .01
                or set(cfg["seeds"]) != {str(s) for s in C2_SEEDS}):
            raise ValueError(f"C2 sai định danh/tham số: {conf}")
        for seed in C2_SEEDS:
            key = f"{conf}_seed{seed}"
            seed_info = cfg["seeds"][str(seed)]
            row = handoff.loc[(conf, seed)]
            audit = audits.loc[(conf, seed)]
            source = inputs["C2"][key]
            name = expected_names[key]
            epoch = seed_info["selected_epoch"]
            notebook = f"12_C2_Train_{conf}_Seed{seed}" + ("_Cont" if cfg["level"] == 3 else "")
            hashes = [row[f"sha256 {n}"] for n in ("best_model.pt", "config.yml", "train_data_scaler.yml")]
            if (type(epoch) is not int or not 1 <= epoch <= 30 or row["epoch chọn"] != epoch
                    or audit["epoch chọn"] != epoch or row["notebook"] != notebook or source["notebook"] != notebook
                    or not digest_string(source["environment_sha256"]) or not all(map(digest_string, hashes))
                    or row["NPZ"] != name or audit["tệp"] != PurePosixPath(name).name
                    or pd.isna(audit["đạt"]) or str(audit["đạt"]).lower() != "true"
                    or not np.isclose(row["NSE val"], seed_info["val"], atol=1e-12, rtol=0)
                    or not np.isclose(row["NSE test"], seed_info["test"], atol=1e-12, rtol=0)
                    or not np.isclose(audit["NSE trung vị (thư viện)"], seed_info["test"], atol=1e-12, rtol=0)):
                raise ValueError(f"C2 bảng bàn giao khác nguồn: {key}")
            records.append(dict(conf=conf, seed=seed, cfg=cfg, epoch=epoch, train_notebook=notebook,
                                seed_info=seed_info, source=source, handoff=row, audit=audit,
                                contract=files[name], path=name))
    return records, files, checks, summary

def c2_forecast(record, context):
    r = record
    f = read_contract(r["contract"])
    meta = f.metadata
    cfg, a, handoff = r["cfg"], r["audit"], r["handoff"]
    name = f"SF_{r['conf']}_C2_seed{r['seed']}"
    expected = dict(run=name, period="test", scenario="S0", lead=1, forecast_horizon=1, seed=r["seed"],
                    clip_to_zero=True, step="C2", model="sequential_forecast", core=cfg["core"], depth=cfg["depth"],
                    level=cfg["level"], learning_rate=cfg["lr"], config=r["conf"], n_params=cfg["n_params"],
                    selected_epoch=r["epoch"], grad_guard_skipped_steps=r["seed_info"]["grad_guard_skipped_steps"],
                    training_notebook=r["train_notebook"], library_commit=FREEZE_C,
                    config_sha256=handoff["sha256 config.yml"])
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f"C2 metadata sai: {name}")
    check_same_context(Forecast(name="A3", sim=context["persistence"], raw=context["persistence"], **context), f)
    nse = np.array([metrics(f.obs[:, j], f.sim[:, j])["NSE"] for j in range(len(f.ids))])
    median = float(np.nanmedian(nse))
    if (str(a["clip_to_zero"]).lower() != "true" or a["ngày-lưu vực có sim"] != np.isfinite(f.sim).sum()
            or a["dự báo âm trước chặn"] != (f.raw < 0).sum()
            or not np.isclose(a["NSE trung vị từ NPZ (obs A3)"], median, atol=1e-8, rtol=0)
            or not np.isclose(r["seed_info"]["test"], median, atol=1e-5, rtol=0)):
        raise ValueError(f"C2 audit clip/mặt nạ/NSE sai: {name}")
    f.obs, f.persistence, f.area = context["obs"], context["persistence"], context["area"]
    f.metadata = dict(meta, compare_notebook=C2_NOTEBOOK, compare_version_expected=1, training_notebook_version_expected=1,
                      source_file_sha256=sha256(r["contract"]), source=str(r["contract"]),
                      checkpoint_sha256=handoff["sha256 best_model.pt"], checkpoint_bytes_verified=False,
                      scaler_sha256=handoff["sha256 train_data_scaler.yml"])
    return f, dict(run=name, configuration_id=r["conf"], seed=r["seed"], selected_epoch=r["epoch"],
                   n_params=cfg["n_params"], IDs=len(f.ids), target_days=len(f.dates),
                   finite_prediction_day_IDs=int(np.isfinite(f.sim).sum()), canonical_context_matches=True,
                   native_test_NSE=r["seed_info"]["test"], NPZ_test_NSE=median,
                   contract_sha256=f.metadata["source_file_sha256"], checkpoint_sha256=f.metadata["checkpoint_sha256"])


# %% Phần 2 - Hàm dự báo dùng config, checkpoint, scaler đã lưu
from collections import OrderedDict
from bisect import bisect_left
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import threading

import numpy as np
import pandas as pd


PRED_ATOL, PRED_RTOL = 2e-5, 1e-6
HRES = ("ECMWF_t2m", "ECMWF_d2m", "ECMWF_ssrd", "ECMWF_tp", "ECMWF_e")


def qmax_m3_s(mm_day, area_gov):
    return mm_day * area_gov / 86.4


def compare_arrays(actual, reference, atol, rtol):
    if actual.shape != reference.shape or not np.array_equal(np.isfinite(actual), np.isfinite(reference)):
        raise ValueError("Kích thước hoặc mặt nạ hữu hạn khác dự báo nguồn.")
    if not np.allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=True) or np.isinf(actual).any():
        raise ValueError(f"Dự báo khác nguồn, max abs={np.nanmax(np.abs(actual-reference)):.8g}")
    delta = np.abs(actual[np.isfinite(actual)] - reference[np.isfinite(reference)])
    return float(delta.max(initial=0))


def on_cpu_mamba(cfg):
    from mambapy.mamba import MambaBlock, MambaConfig
    from neuralhydrology.modelzoo.sequential_forecast import SequentialForecast
    from neuralhydrology.modelzoo.sf_cores import ResidualCore, mamba_expand

    # Dựng phần nhúng và head qua cùng lớp của fork; thay duy nhất lõi Mamba CUDA.
    shell = copy.deepcopy(cfg)
    shell.update_config({"sf_core": "s4d", "sf_layers": 0})
    model = SequentialForecast(shell)

    def block(width):
        config = MambaConfig(d_model=width, n_layers=1, d_state=cfg.sf_mamba_d_state,
                             d_conv=cfg.sf_mamba_d_conv, use_cuda=False)
        config.d_inner = int(width * mamba_expand(cfg, width))
        return MambaBlock(config)

    model.core = ResidualCore(cfg, model.hindcast_embedding_net.output_size, block)
    model.cfg = cfg
    return model


class DemoForecaster:
    def __init__(self, run, raw_dir, ids, areas, scratch, device="cuda:0", *, n_params=None):
        import torch
        from neuralhydrology.datautils.utils import load_scaler
        from neuralhydrology.datasetzoo.lamah import (
            LamaH, load_lamah_attributes, load_lamah_discharge, load_lamah_forcing, _normalize_discharge,
        )
        from neuralhydrology.modelzoo import get_model
        from neuralhydrology.utils.config import Config

        class DemoLamaH(LamaH):
            def _load_basin_data(self, basin):
                frame = super()._load_basin_data(basin)
                for name in ("qmean", "qmax"):
                    flow = load_lamah_discharge(self.cfg.data_dir, basin, temporal_resolution="1D",
                                                normalize_discharge=False, target_name=name)[name]
                    flow = _normalize_discharge(ser=flow, area=self._basin_area.loc[basin],
                                                temporal_resolution="1D").reindex(frame.index)
                    # C2/H/D đọc lại CSV với định dạng %.17g khi dựng q*_mmd.
                    frame[f"{name}_mmd"] = pd.read_csv(StringIO(flow.to_csv(index=False, header=["q"],
                                                                         float_format="%.17g")))["q"].to_numpy()
                return frame

        self.torch, self.dataset_type = torch, DemoLamaH
        self.load_forcing = load_lamah_forcing
        self.device = torch.device(device)
        ids = tuple(map(str, ids))
        areas = np.asarray(areas)
        if (len(ids) != len(areas) or len(set(ids)) != len(ids)
                or not np.isfinite(areas).all() or np.any(areas <= 0)):
            raise ValueError("ID hoặc diện tích lưu vực không hợp lệ.")
        self.areas = dict(zip(ids, areas))
        attrs = load_lamah_attributes(Path(raw_dir), sub_dataset="lamah_a").loc[list(ids), "area_gov"]
        if not np.array_equal(attrs.to_numpy(), areas):
            raise ValueError("Diện tích lưu vực của dữ liệu demo khác A3.")
        run = Path(run)
        cfg = Config(run / "config.yml")
        self.length, self.horizon = cfg.seq_length, cfg.forecast_seq_length
        self.mode = cfg.forecast_input_mode
        checks = (
            ("model", cfg.model, cfg.model == "sequential_forecast"),
            ("dataset", cfg.dataset, cfg.dataset == "lamah_a"),
            ("seq_length/forecast_seq_length", (self.length, self.horizon),
             (self.length, self.horizon) in ((365, 1), (371, 7), (730, 1))),
            ("predict_last_n", cfg.predict_last_n, cfg.predict_last_n == self.horizon),
            ("forecast_overlap", cfg.forecast_overlap, cfg.forecast_overlap in (None, 0)),
            ("target_variables", cfg.target_variables, cfg.target_variables == ["qmax"]),
            ("clip_targets_to_zero", cfg.clip_targets_to_zero, "qmax" in cfg.clip_targets_to_zero),
            ("additional_feature_files", cfg.additional_feature_files, not cfg.additional_feature_files),
            ("use_basin_id_encoding", cfg.use_basin_id_encoding, not cfg.use_basin_id_encoding),
        )
        invalid = [f"{key}={value!r}" for key, value, valid in checks if not valid]
        if invalid:
            raise ValueError(f"{run.name}: config ngoài giao thức C2/H/D ({', '.join(invalid)}).")
        self.scratch = TemporaryDirectory(dir=scratch)
        basins = Path(self.scratch.name) / "basins.txt"
        basins.write_text("\n".join(ids) + "\n", encoding="utf-8")
        cfg.update_config(dict(data_dir=Path(raw_dir), train_basin_file=basins,
                               validation_basin_file=basins, test_basin_file=basins, device=device))
        self.cfg = cfg
        self.dates = pd.date_range(self.cfg.test_start_date, self.cfg.test_end_date)
        self.forecast_features = tuple(self.cfg.forecast_inputs_flattened)
        self.accepts_hres = set(self.forecast_features) == set(HRES) and self.mode in ("all", "first_day")
        self.scaler = load_scaler(run)
        self.centers = self.scaler["xarray_feature_center"]
        self.scales = self.scaler["xarray_feature_scale"]
        if self.device.type == "cpu" and self.cfg.sf_core == "mamba":
            self.model = on_cpu_mamba(self.cfg)
        else:
            self.model = get_model(self.cfg)
        self.model.to(self.device)
        weights = torch.load(run / "best_model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        if n_params is not None and sum(p.numel() for p in self.model.parameters() if p.requires_grad) != n_params:
            raise ValueError("Số tham số khác bản bàn giao.")
        self.model.eval()
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def close(self):
        self.scratch.cleanup()

    def _dataset(self, basin):
        if basin not in self.areas:
            raise ValueError(f"Lưu vực ngoài tập đánh giá: {basin}")
        if basin not in self.cache:
            dataset = self.dataset_type(self.cfg, is_train=False, period="test", basin=basin,
                                        scaler=copy.deepcopy(self.scaler))
            if dataset.is_train or dataset._compute_scaler:
                raise ValueError("Demo cần dùng scaler train đã lưu.")
            self.cache[basin] = dataset
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(basin)
        return self.cache[basin]

    def _forward(self, batch):
        dates = np.asarray(batch["date"]).astype("datetime64[D]")
        if (dates.ndim != 2 or dates.shape[1] != self.length or not np.all(np.diff(dates, axis=1) == np.timedelta64(1, "D"))
                or any(v.shape[1] != self.length-self.horizon for v in batch["x_d_hindcast"].values())
                or any(v.shape[1] != self.horizon for v in batch["x_d_forecast"].values())):
            raise ValueError("Lịch hoặc độ dài các pha khác config đã lưu.")
        with self.torch.inference_mode():
            predictions = self.model(self.model.pre_model_hook(batch, is_train=False))["y_hat"]
        values = predictions[:, -self.horizon:, 0].cpu().numpy()
        obs = batch["y"][:, -self.horizon:, 0].cpu().numpy()
        # Giữ dtype của phép hoàn nguyên trong Tester (scaler có thể lưu float64).
        scale = self.scales[["qmax"]].to_array().values
        center = self.centers[["qmax"]].to_array().values
        return dates[:, -self.horizon:], values * scale + center, obs * scale + center

    def predict_basin(self, basin, batch_size=None):
        from neuralhydrology.datautils.gpu_loader import GpuBatchLoader

        with self.lock:
            loader = GpuBatchLoader(self._dataset(str(basin)), batch_size=batch_size or self.cfg.batch_size,
                                    device=self.device, shuffle=False, with_dates=True)
            return tuple(np.concatenate(chunks) for chunks in zip(*(self._forward(batch) for batch in loader)))

    def _window(self, basin, origin_date):
        day = pd.Timestamp(origin_date)
        last = day + pd.Timedelta(days=self.horizon - 1)
        if day not in self.dates or last not in self.dates:
            raise ValueError("Cửa sổ dự báo phải nằm trong kỳ test.")
        dataset = self._dataset(basin)
        daily = dataset._dates[basin]["1D"]
        last_date = last.to_datetime64()
        # H bắt đầu mẫu khi đủ bảy ngày đích trong kỳ test. Chỉ mục lookup khác chỉ số lịch.
        def end_date(index):
            _, positions = dataset.lookup_table[index]
            return daily[positions[0]]

        index = bisect_left(range(len(dataset)), last_date, key=end_date)
        if index == len(dataset) or end_date(index) != last_date:
            raise ValueError("Ngày này không có cửa sổ hoàn chỉnh trong lookup.")
        batch = dataset.collate_fn([dataset[index]])
        for key, value in batch.items():
            if self.torch.is_tensor(value):
                batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                batch[key] = {k: v.to(self.device) for k, v in value.items()}
        if np.asarray(batch["date"])[0, -self.horizon] != day.to_datetime64():
            raise ValueError("Ngày đầu dự báo khác lịch lookup.")
        return batch

    def _raw_hres(self, basin, day):
        if not self.accepts_hres:
            raise ValueError("Run này không nhận ECMWF HRES ngày đầu.")
        frame = self.load_forcing(self.cfg.data_dir, basin, self.cfg.dataset, temporal_resolution="1D")
        return frame.loc[pd.Timestamp(day), list(HRES)].to_dict()

    def _apply_hres(self, inputs, raw, hres, rain_multiplier, temperature_delta):
        values = dict(hres or {})
        if set(values) - set(HRES) or not all(np.isfinite(v) for v in values.values()):
            raise ValueError("ECMWF HRES thay thế cần đúng tên biến và giá trị hữu hạn.")
        if not np.isfinite(rain_multiplier) or rain_multiplier < 0 or not np.isfinite(temperature_delta):
            raise ValueError("Hệ số mưa cần hữu hạn và không âm. Mức cộng nhiệt cần hữu hạn.")
        for name, apply, change in (("ECMWF_tp", rain_multiplier != 1, lambda v: v * rain_multiplier),
                                    ("ECMWF_t2m", temperature_delta != 0, lambda v: v + temperature_delta)):
            if apply:
                old = values.get(name, raw[name])
                if not np.isfinite(old):
                    raise ValueError(f"Thiếu {name} ngày đầu.")
                values[name] = change(old)
        for name, value in values.items():
            scale = self.scales[name].item()
            if not np.isfinite(scale) or scale <= 0 or (name == "ECMWF_tp" and value < 0):
                raise ValueError(f"Giá trị ECMWF HRES ngoài miền: {name}")
            # LamaH chuyển giá trị về float32 trước khi dùng scaler.
            value = np.float32(value)
            normalized = (value - self.centers[name].to_numpy()) / self.scales[name].to_numpy()
            if not np.isfinite(normalized):
                raise ValueError(f"Giá trị ECMWF HRES ngoài miền: {name}")
            inputs[name] = inputs[name].clone()
            inputs[name][:, 0, :] = float(normalized)

    def predict(self, basin, origin_date, hres=None, *, rain_multiplier=1.0, temperature_delta=0.0):
        # hres là các giá trị ngày đầu trong đơn vị của dữ liệu phát hành.
        basin = str(basin)
        with self.lock:
            batch = self._window(basin, origin_date)
            if hres is not None or rain_multiplier != 1.0 or temperature_delta != 0.0:
                original = self._raw_hres(basin, origin_date)
                self._apply_hres(batch["x_d_forecast"], original, hres, rain_multiplier, temperature_delta)
            days, raw, _ = self._forward(batch)
        if not np.isfinite(raw).all():
            raise ValueError("Dự báo có giá trị không hữu hạn.")
        clipped = np.maximum(raw[0], 0)
        return [dict(basin_id=basin, origin_date=str(days[0, 0]), target_date=str(day), lead=k,
                     qmax_raw_mm_day=float(raw[0, k-1]), qmax_mm_day=float(clipped[k-1]),
                     qmax_m3_s=float(qmax_m3_s(clipped[k-1], self.areas[basin])))
                for k, day in enumerate(days[0], 1)]


def align_basin_lead(dates, raw, obs, reference_dates, lead):
    days = dates[:, lead-1]
    reference_dates = np.asarray(reference_dates, dtype="datetime64[D]")
    index = np.searchsorted(days, reference_dates)
    if np.any(index >= len(days)) or not np.array_equal(days[index], reference_dates):
        raise ValueError(f"Lịch dự báo hạn {lead} khác hợp đồng.")
    return raw[index, lead-1], obs[index, lead-1]

# %% Phần 3 - Đối chiếu và lưu kết quả
def run_d1(input_root, output):
    import torch
    from time import perf_counter

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "D1_summary.json"
    write_json(summary_path, {"complete": False})
    context, compare, records = compare_inputs(input_root)
    sources = c2_train_sources(input_root, records)
    raw_dir = locate_raw(input_root)
    basin_rows, run_rows, examples = [], [], []
    for record in records:
        run = record_run(record)
        ref, _ = c2_forecast(record, context)
        started = perf_counter()
        predictor = DemoForecaster(sources[run]["best_model.pt"].parent, raw_dir, ref.ids, ref.area, output,
                                   n_params=record["cfg"]["n_params"])
        records_start = len(basin_rows)
        try:
            for j, basin in enumerate(ref.ids):
                series = predictor.predict_basin(basin)
                prediction, audit = audit_prediction(series, ref, j)
                basin_rows.append(dict(run=run, basin_id=basin, passed=True, **audit))
                if j in (0, len(ref.ids)//2, len(ref.ids)-1):
                    i = len(ref.dates)//2
                    day = str(ref.dates[i].date())
                    actual = predictor.predict(basin, day)[0]
                    compare_arrays(np.array([actual["qmax_raw_mm_day"]]), prediction[i:i+1], PRED_ATOL, PRED_RTOL)
                    examples.append(dict(run=run, **actual))
        finally:
            predictor.close()
        elapsed = perf_counter() - started
        run_rows.append(dict(run=run, configuration=record["conf"], seed=record["seed"], epoch=record["epoch"],
                             core=record["cfg"]["core"], n_params=record["cfg"]["n_params"], horizon=1, days=len(ref.dates),
                             IDs=len(basin_rows)-records_start, seconds=elapsed, passed=True,
                             source_contract_sha256=sha256(record["contract"]),
                             **{f"{Path(name).name}_sha256": sha256(file) for name, file in sources[run].items()}))
        del predictor
        torch.cuda.empty_cache()
        pd.DataFrame(basin_rows).to_csv(output / "D1_reproduction.csv", index=False)
        pd.DataFrame(run_rows).to_csv(output / "D1_run_check.csv", index=False)
        print(f"{len(run_rows)}/24 {run}: đạt đối chiếu {len(ref.ids)} lưu vực ({elapsed:.1f}s).", flush=True)
    pd.DataFrame(examples).to_csv(output / "D1_examples.csv", index=False)
    summary = dict(complete=True, notebook="01_D1_LamaHCEExt_Inference", library_commit=FREEZE_C,
                   C2_compare_manifest_sha256=sha256(compare/"manifest.csv"), C2_compare_version_expected=1,
                   runs=len(run_rows), checks=len(basin_rows), IDs=len(context["ids"]), days=len(context["dates"]),
                   atol=PRED_ATOL, rtol=PRED_RTOL, optimizer_steps=0,
                   max_raw_abs_error=max(row["max_raw_abs_error"] for row in basin_rows),
                   max_clip_abs_error=max(row["max_clip_abs_error"] for row in basin_rows),
                   default_demo_run=C2_DEFAULT)
    write_json(summary_path, summary)
    env = output / "environment.json"
    write_json(env, dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         torch=torch.__version__, torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                         neuralhydrology_commit=FREEZE_C, C2_train_version_expected=1, A3_version_expected=1))
    names = ("D1_summary.json", "D1_run_check.csv", "D1_reproduction.csv", "D1_examples.csv", "environment.json")
    export_manifest(output, [output/name for name in names])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Gửi D1_summary.json, D1_run_check.csv, D1_reproduction.csv, D1_examples.csv, environment.json, manifest.csv.")


if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Chọn Kaggle T4 cho D1.")
    root = Path("/kaggle/input")
    install_mamba_cuda(root)
    install_library(FREEZE_C)
    run_d1(root, Path("/kaggle/working/d1_output"))
