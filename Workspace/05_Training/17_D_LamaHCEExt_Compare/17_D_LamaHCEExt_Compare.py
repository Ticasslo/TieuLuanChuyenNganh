# %% [markdown]
# # D — So sánh cửa sổ 365 và 730 bước
#
# Kaggle CPU, Internet Off. Input là C2 Compare Version 1 và A3 Version 1,
# thêm các Output huấn luyện C2 và D cùng cấu hình/seed. Mã nguồn nhận
# Output hoàn thành có cấu trúc summary/environment/manifest như khuôn C.
# Chuỗi 365/730 gồm 364/729 bước lịch sử và một ngày dự báo ECMWF.
#
# 1. Hợp đồng và chỉ số dùng chung. 2. Kiểm nguồn và cấu hình.
# 3. Xuất NPZ. 4. Chấm giao chung, chi phí, hình và manifest.

# %% Phần 1 - Hợp đồng và chỉ số dùng chung với Evaluation
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import hashlib
import itertools
import json
import math
import pickle
import re
import sys

import numpy as np
import pandas as pd
import yaml

NOTEBOOK = "17_D_LamaHCEExt_Compare"
METRICS = ("NSE", "PNSE", "KGE", "r", "alpha", "beta")

# Các hàm sau lấy từ 01_Evaluation_LamaHCEExt.py để notebook CPU tự đủ.

FREEZE_C = "db6dc1ffcd8c14994cd80be3ed8ff417002c01a1"

C2_NOTEBOOK = "13_C2_LamaHCEExt_Compare"

C2_CONFIGS = ("lstm_L2_x1_lr1e-3", "gru_L2_x1_lr1e-3", "transformer_L2_x1_lr1e-3",
              "transformer_L2_x3_lr5e-4", "s4d_L4_x1_lr1e-3", "s4d_L4_x3_lr1e-3",
              "mamba_L2_x1_lr1e-3", "mamba_L4_x3_lr1e-3")

C2_SEEDS = (111, 222, 333)

TEST_START, TEST_END = "2014-01-01", "2017-12-31"

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

def common_mask(forecasts):
    if not forecasts:
        raise ValueError("Nhóm dự báo rỗng")
    base = forecasts[0]
    mask = np.isfinite(base.obs) & np.isfinite(base.persistence)
    for forecast in forecasts:
        check_same_context(base, forecast)
        mask &= np.isfinite(forecast.sim)
    return mask

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
    f.metadata = dict(meta, compare_notebook=C2_NOTEBOOK, compare_version_expected=1, training_notebook_version_expected=1,
                      source_file_sha256=sha256(r["contract"]), source=str(r["contract"]),
                      checkpoint_sha256=handoff["sha256 best_model.pt"], checkpoint_bytes_verified=False,
                      scaler_sha256=handoff["sha256 train_data_scaler.yml"])
    return f, dict(run=name, configuration_id=r["conf"], seed=r["seed"], selected_epoch=r["epoch"],
                   n_params=cfg["n_params"], IDs=len(f.ids), target_days=len(f.dates),
                   finite_prediction_day_IDs=int(np.isfinite(f.sim).sum()), canonical_context_matches=True,
                   native_test_NSE=r["seed_info"]["test"], NPZ_test_NSE=median,
                   contract_sha256=f.metadata["source_file_sha256"], checkpoint_sha256=f.metadata["checkpoint_sha256"])
# %% Phần 2 - Khóa nguồn huấn luyện, cùng cặp seed và kiểm cấu hình
D_CONFIGS = ("lstm_L2_x1_lr1e-3", "mamba_L2_x1_lr1e-3", "s4d_L4_x1_lr1e-3")
RUNTIME_KEYS = {
    "data_dir", "run_dir", "train_dir", "img_log_dir", "train_basin_file", "validation_basin_file",
    "test_basin_file", "device", "num_workers", "verbose", "experiment_name", "commit_hash",
    "package_version", "number_of_basins", "max_training_hours", "resume_training_state",
}
COST_KEYS = ("train_minutes_per_epoch_median", "val_minutes_per_epoch_median", "test_eval_minutes_median",
             "train_hours", "gpu_mib_max")


def show_table(title, table, digits=4):
    if "ipykernel" in sys.modules:
        from IPython.display import HTML, display
        display(HTML(f"<h4 style='margin:12px 0 4px'>{title}</h4>"))
        display(table.style.hide(axis="index").format(precision=digits, na_rep="-"))
    else:
        print(f"=== {title} ({len(table)} dòng) ===")


def run_costs(summary):
    # Gộp cả phiên 1 (previous_session) khi run chạy qua hai phiên (_Cont), gồm cả trường hợp phiên 2 SKIP_TRAIN.
    prev = summary.get("previous_session") or {}
    sessions = 2 if prev else 1
    gpu_vals = [v for v in (summary.get("gpu_mib_max"), prev.get("gpu_mib_max")) if v is not None]
    out = {"sessions": sessions}
    for key in COST_KEYS:
        if key == "train_hours":
            s_h, p_h = summary.get("train_hours"), prev.get("train_hours")
            out[key] = None if (s_h is None and p_h is None) else float((s_h or 0.0) + (p_h or 0.0))
        elif key == "gpu_mib_max":
            out[key] = int(max(gpu_vals)) if gpu_vals else None
        else:
            val = summary.get(key)
            out[key] = val if val is not None else prev.get(key)
    return out


def completed_run(env_file, step, conf, seed, expected=None):
    out = env_file.parent
    mount = out.parent
    env = json.loads(env_file.read_text(encoding="utf-8-sig"))
    summary_path = out / "summary.json"
    s = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    core, depth, _, lr = conf.split("_")
    epoch = s["saved_epoch"]
    exp = f"{step}_{conf}_seed{seed}"
    if (s != env.get("summary") or (s["step"], s["core"], s["depth"], s["level"], s["seed"], s["learning_rate"])
            != (step, core, int(depth[1:]), 1, seed, float(lr[2:]))
            or s["status"] != "xong" or s["epochs_trained"] != 30 or s["epochs_tested"] != 30
            or s["train_done"] is not True or s["test_done"] is not True or s["test_every_epoch"] is not True
            or type(epoch) is not int or not 1 <= epoch <= 30 or epoch != s["true_best_epoch"]
            or s["library_commit"] != FREEZE_C or env["neuralhydrology_commit"] != FREEZE_C
            or env["experiment_name"] != exp or s["experiment_name"] != exp
            or s["notebook"] != env["notebook"] or PurePosixPath(env["run_dir"]).name != s["run_dir"]):
        raise ValueError(f"{exp}: sai giao thức hoặc chưa đủ 30 epoch; dùng Output cuối nếu có _Cont")
    if step == "D":
        session = s.get("session", 1)
        expected_nb = f"16_D_Train_{conf}_Seed{seed}" + ("_Cont" if session == 2 else "")
        has_prev = isinstance(s.get("previous_session"), dict)
        if (session not in (1, 2) or env.get("session", 1) != session
                or env["notebook"] != expected_nb or has_prev != (session == 2)):
            raise ValueError(f"{exp}: tên notebook hoặc trạng thái phiên _Cont không khớp")
    if expected and (sha256(env_file) != expected["source"]["environment_sha256"]
                     or env["notebook"] != expected["train_notebook"] or env["run_dir"] != expected["handoff"]["thư mục run"]
                     or epoch != expected["epoch"] or s["n_params"] != expected["cfg"]["n_params"]):
        raise ValueError(f"{exp}: Output C2 không khớp C2 Compare")
    files, _ = verified_manifest(mount, out / "manifest.csv")
    needed = ["config.yml", "best_model.pt", "train_data/train_data_scaler.yml"]
    if step == "D":
        needed += ["test/best_model/test_results.p", "test/best_model/test_metrics.csv"]
    chosen = {name: files[f"{env['run_dir']}/{name}"] for name in needed}
    if expected:
        for name in needed:
            if sha256(chosen[name]) != expected["handoff"][f"sha256 {PurePosixPath(name).name}"]:
                raise ValueError(f"{exp}: hash tệp train khác bàn giao C2: {name}")
    curve_path = out / f"curve_{exp}.csv"
    if curve_path.is_file():
        curve = pd.read_csv(curve_path)
        val_col, test_col = "NSE trung vị validation (tệp)", "NSE trung vị test"
        if (curve["epoch"].tolist() != list(range(1, 31))
                or val_col not in curve or test_col not in curve
                or not np.isfinite(curve[val_col]).all() or not curve[test_col].notna().all()
                or int(curve.loc[curve[val_col].idxmax(), "epoch"]) != epoch
                or not np.isclose(curve.set_index("epoch").loc[epoch, val_col],
                                  s["val_nse_median_best_model"], rtol=0, atol=1e-12)
                or not np.isclose(curve.set_index("epoch").loc[epoch, test_col],
                                  s["test_nse_median_best_model"], rtol=0, atol=1e-6)):
            raise ValueError(f"{exp}: curve_{exp}.csv không khớp summary/best_model")
    cfg = yaml.safe_load(chosen["config.yml"].read_text(encoding="utf-8-sig"))
    if (sha256(chosen["config.yml"]) != env["config_sha256"]
            or (cfg["model"], cfg["sf_core"], cfg["sf_layers"], cfg["seed"], cfg["learning_rate"])
               != ("sequential_forecast", core, int(depth[1:]), seed, float(lr[2:]))
            or cfg["seq_length"] != (365 if step == "C2" else 730)
            or cfg["forecast_seq_length"] != 1 or cfg["predict_last_n"] != 1
            or cfg.get("forecast_overlap") not in (0, None) or cfg["target_variables"] != ["qmax"]
            or cfg.get("epochs") != 30 or cfg.get("data_on_gpu") is not True
            or cfg.get("nan_handling_method") != "masked_mean"
            or cfg.get("nan_step_probability") != .1 or cfg.get("nan_sequence_probability") != .12
            or "qmax" not in cfg["clip_targets_to_zero"]):
        raise ValueError(f"{exp}: cấu hình không giữ đúng nền freeze-C")
    proto = env.get("protocol") or {}
    gg_skips = proto.get("grad_guard_skipped_steps", 0)
    if type(gg_skips) is not int or gg_skips < 0:
        raise ValueError(f"{exp}: grad_guard_skipped_steps không hợp lệ: {gg_skips!r}")
    gg_enabled = bool(proto.get("grad_guard_enabled", "grad_guard_skipped_steps" in proto))
    expected_skips = expected["seed_info"].get("grad_guard_skipped_steps") if expected else None
    if expected and gg_skips != (0 if expected_skips is None else expected_skips):
        raise ValueError(f"{exp}: grad_guard_skipped_steps khác C2 Compare")
    return dict(files=chosen, summary=s, cfg=cfg, env=env, env_path=env_file,
                grad_guard_enabled=gg_enabled, grad_guard_skipped_steps=gg_skips)


def compare_configs(c2, d):
    changes = {k for k in set(c2) | set(d) if c2.get(k) != d.get(k)} - RUNTIME_KEYS
    if changes != {"seq_length"} or (c2["seq_length"], d["seq_length"]) != (365, 730):
        raise ValueError(f"Khác biệt khoa học phải đúng seq_length 365→730, thấy {sorted(changes)}")


def find_runs(input_root, selected):
    lookup = {}
    names = {f"{step}_{c}_seed{s}": (step, c, s)
             for c, s in selected for step in ("C2", "D")}
    for path in Path(input_root).rglob("environment.json"):
        if path.parent.name not in ("c_output", "d_output"):
            continue
        env = json.loads(path.read_text(encoding="utf-8-sig"))
        key = names.get(env.get("experiment_name"))
        if key is None or env.get("summary", {}).get("status") != "xong":
            continue
        if key in lookup:
            raise ValueError(f"Trùng Output hoàn thành: {key}")
        lookup[key] = path
    return lookup


# %% Phần 3 - Đọc dự báo D, kiểm quan trắc và xuất NPZ
def export_d(run, ref, record, dest):
    # Chỉ đọc pickle do notebook huấn luyện D lưu sau khi kiểm manifest gốc.
    with run["files"]["test/best_model/test_results.p"].open("rb") as stream:
        data = pickle.load(stream)
    keys = {basin_id(k): k for k in data}
    if len(keys) != len(data) or set(keys) != set(ref.ids):
        raise ValueError("test_results.p D sai danh sách lưu vực")
    raw, obs = np.full_like(ref.obs, np.nan), np.full_like(ref.obs, np.nan)
    for j, basin in enumerate(ref.ids):
        ds = data[keys[basin]]["1D"]["xr"]
        if (not pd.DatetimeIndex(ds["date"].values).as_unit("ns").equals(ref.dates)
                or ds["time_step"].values.tolist() != [0]):
            raise ValueError(f"D: lịch hoặc hạn không khớp C2, ID {basin}")
        raw[:, j] = ds["qmax_sim"].transpose("date", "time_step").values[:, 0]
        obs[:, j] = ds["qmax_obs"].transpose("date", "time_step").values[:, 0]
    if (not np.allclose(obs, ref.obs, atol=1e-5, rtol=1e-5, equal_nan=True)
            or not np.isfinite(raw).all()):
        raise ValueError("D: quan trắc khác A3 hoặc dự báo thiếu/vô cực")
    sim = np.maximum(raw, 0)
    native_nse = np.array([metrics(obs[:, j], sim[:, j])["NSE"] for j in range(len(ref.ids))])
    actual = pd.read_csv(run["files"]["test/best_model/test_metrics.csv"], index_col=0)
    actual.index = [basin_id(v) for v in actual.index]
    if not actual.index.is_unique or set(actual.index) != set(ref.ids):
        raise ValueError("D: test_metrics sai ID")
    source = actual.reindex(ref.ids)["NSE"].to_numpy()
    if (not np.allclose(native_nse, source, atol=1e-5, rtol=0, equal_nan=True)
            or not np.isclose(np.nanmedian(source), run["summary"]["test_nse_median_best_model"], atol=1e-8, rtol=0)):
        raise ValueError("D: NSE tính lại khác test metrics hoặc summary")
    conf, seed = record["conf"], record["seed"]
    meta = dict(run=f"SF_{conf}_D_seed{seed}", step="D", config=conf, core=record["cfg"]["core"],
                depth=record["cfg"]["depth"], level=1, n_params=run["summary"]["n_params"],
                learning_rate=record["cfg"]["lr"], seed=seed, selected_epoch=run["summary"]["saved_epoch"],
                period="test", scenario="S0", lead=1, forecast_horizon=1, unit="mm/day_area_gov",
                seq_length=730, hindcast_steps=729, forecast_steps=1, clip_to_zero=True, library_commit=FREEZE_C,
                grad_guard_enabled=run["grad_guard_enabled"],
                grad_guard_skipped_steps=run["grad_guard_skipped_steps"],
                config_sha256=sha256(run["files"]["config.yml"]), checkpoint_sha256=sha256(run["files"]["best_model.pt"]),
                scaler_sha256=sha256(run["files"]["train_data/train_data_scaler.yml"]),
                training_notebook=run["env"]["notebook"], training_environment_sha256=sha256(run["env_path"]),
                baseline_run=ref.name, baseline_contract_sha256=sha256(record["contract"]))
    days = ref.dates.to_numpy(dtype="datetime64[D]")
    path = dest / f"{meta['run']}_test_S0_lead1.npz"
    np.savez_compressed(path, dates=days, origin_dates=days, persistence_source_dates=days-np.timedelta64(1,"D"),
                        basin_ids=np.array(ref.ids), area_gov_km2=ref.area, obs_mm_day=ref.obs,
                        persistence_mm_day=ref.persistence, sim_raw_mm_day=raw, sim_mm_day=sim,
                        metadata_json=json.dumps(meta, ensure_ascii=False))
    pred = read_contract(path)
    check_same_context(ref, pred)
    audit = dict(run=pred.name, baseline=ref.name, seed=seed, selected_epoch=meta["selected_epoch"], IDs=len(ref.ids),
                 days=len(days), native_nse_median=float(np.nanmedian(native_nse)),
                 finite_predictions=int(np.isfinite(raw).sum()),
                 grad_guard_enabled=run["grad_guard_enabled"],
                 grad_guard_skipped_steps=run["grad_guard_skipped_steps"],
                 SHA256=sha256(path), passed=True)
    return pred, path, audit


# %% Phần 4 - Giao chung 365/730, điểm từng lưu vực, tổ hợp 3 hạt giống, bàn giao và chi phí
def score_d(pairs, val_pairs=None, b0_noise=None):
    shared = common_mask([forecast for c, d in pairs for forecast in (c, d)])
    if not shared.any():
        raise ValueError("D và C2 không có giao ngày hợp lệ")
    columns = ("NSE", "PNSE", "KGE")
    basin_rows, seed_rows = [], []
    for c, d in pairs:
        original = [metrics(c.obs[:, j], c.sim[:, j], c.persistence[:, j], shared[:, j]) for j in range(len(c.ids))]
        extended = [metrics(d.obs[:, j], d.sim[:, j], d.persistence[:, j], shared[:, j]) for j in range(len(d.ids))]
        conf, seed = d.metadata["config"], d.seed
        group = dict(configuration=conf, seed=seed)
        seed_row = dict(**group)
        if val_pairs is not None and (conf, seed) in val_pairs:
            v365, v730 = val_pairs[(conf, seed)]
            dv = v730 - v365
            seed_row.update(val_NSE_365=v365, val_NSE_730=v730, delta_val_NSE=dv)
            if b0_noise is not None and b0_noise > 0:
                seed_row["delta_val_x_noise"] = dv / b0_noise
        for j, basin in enumerate(c.ids):
            row = dict(**group, basin_id=basin, common_days=int(shared[:, j].sum()))
            for m in columns:
                row[f"{m}_365"] = original[j][m]
                row[f"{m}_730"] = extended[j][m]
                row[f"delta_{m}"] = extended[j][m] - original[j][m]
            basin_rows.append(row)
        for m in columns:
            a, b = (np.array([r[m] for r in scores]) for scores in (original, extended))
            valid = np.isfinite(a) & np.isfinite(b)
            med_a, med_b = (float(np.median(v[valid])) if valid.any() else np.nan for v in (a,b))
            effect = float(np.median(b[valid]-a[valid])) if valid.any() else np.nan
            seed_row.update({f"{m}_365": med_a, f"{m}_730": med_b, f"delta_{m}": med_b-med_a,
                             f"paired_median_delta_{m}": effect, f"paired_IDs_{m}": int(valid.sum())})
            if m == "NSE" and b0_noise is not None and b0_noise > 0:
                seed_row["delta_NSE_x_noise"] = (med_b - med_a) / b0_noise
        seed_rows.append(seed_row)
    return pd.DataFrame(basin_rows), pd.DataFrame(seed_rows), shared


def ensemble_d(pairs, configs, shared, seed_df, b0_noise=None):
    # Tổ hợp 3 hạt giống (trung bình dự báo sim_mm_day trên giao chung; chỉ tính trong notebook, không xuất NPZ).
    rows = []
    for conf in configs:
        sub = [(c, d) for c, d in pairs if d.metadata["config"] == conf]
        sim_365 = np.mean([c.sim for c, _ in sub], axis=0)
        sim_730 = np.mean([d.sim for _, d in sub], axis=0)
        base = sub[0][0]
        sc_365 = [metrics(base.obs[:, j], sim_365[:, j], base.persistence[:, j], shared[:, j])
                  for j in range(len(base.ids))]
        sc_730 = [metrics(base.obs[:, j], sim_730[:, j], base.persistence[:, j], shared[:, j])
                  for j in range(len(base.ids))]
        part = seed_df.loc[seed_df.configuration.eq(conf)]
        row = dict(configuration=conf, seeds=len(sub))
        for m in ("NSE", "PNSE", "KGE"):
            a = np.array([r[m] for r in sc_365], dtype=float)
            b = np.array([r[m] for r in sc_730], dtype=float)
            valid = np.isfinite(a) & np.isfinite(b)
            ens_365 = float(np.median(a[valid])) if valid.any() else np.nan
            ens_730 = float(np.median(b[valid])) if valid.any() else np.nan
            mean_365 = float(part[f"{m}_365"].mean())
            mean_730 = float(part[f"{m}_730"].mean())
            row.update({
                f"{m}_seed_mean_365": mean_365,
                f"{m}_ensemble_365": ens_365,
                f"{m}_ensemble_gain_365": ens_365 - mean_365,
                f"{m}_seed_mean_730": mean_730,
                f"{m}_ensemble_730": ens_730,
                f"{m}_ensemble_gain_730": ens_730 - mean_730,
                f"delta_{m}_ensemble": ens_730 - ens_365,
            })
            if m == "NSE" and b0_noise is not None and b0_noise > 0:
                row["delta_NSE_ensemble_x_noise"] = (ens_730 - ens_365) / b0_noise
        rows.append(row)
    return pd.DataFrame(rows)


def scalar_json(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def compare(input_root, output_root, *, expected_ids=451):
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "d_summary.json"
    summary_path.write_text('{"complete": false}', encoding="utf-8")
    input_root = Path(input_root)
    a3 = unique_file(input_root, "A3_persistence_predictions.npz").parent
    context, a3_paths = read_context(a3)
    if len(context["ids"]) != expected_ids:
        raise ValueError(f"D cần đúng {expected_ids} lưu vực")
    c2_root = unique_file(input_root, "c2_summary.json").parent
    records, _, _, c2_summary = c2_records(c2_root, a3_paths)
    b0_noise = float(c2_summary["bstar"]["noise_sd"])
    candidates = {(c, s) for c in D_CONFIGS for s in C2_SEEDS}
    runs = find_runs(input_root, candidates)
    # Bỏ S4D khi cắt lịch chỉ bằng cách gắn sáu Output LSTM/Mamba và không gắn Output D của S4D.
    s4 = any(step == "D" and conf == D_CONFIGS[-1] for step, conf, _ in runs)
    if not s4:
        for p in input_root.rglob("environment.json"):
            if p.parent.name in ("c_output", "d_output"):
                candidate = json.loads(p.read_text(encoding="utf-8-sig")).get("experiment_name", "")
                if candidate.startswith(f"D_{D_CONFIGS[-1]}_"):
                    raise ValueError("Đã gắn S4D nhưng chưa có đủ Output hoàn thành; không tự bỏ cấu hình")
    configs = D_CONFIGS if s4 else D_CONFIGS[:-1]
    selected = {(c, s) for c in configs for s in C2_SEEDS}
    missing = sorted((step,c,s) for c,s in selected for step in ("C2","D") if (step,c,s) not in runs)
    if missing:
        raise FileNotFoundError(f"Thiếu Output C2/D hoàn thành: {missing}")
    contracts = output / "contracts"
    contracts.mkdir(exist_ok=True)
    checked, pairs, costs, exports, config_rows, handoff_rows = [], [], [], [], [], []
    val_pairs = {}
    for record in records:
        c, s = record["conf"], record["seed"]
        if (c, s) not in selected:
            continue
        c2 = completed_run(runs[("C2",c,s)], "C2", c, s, expected=record)
        d = completed_run(runs[("D",c,s)], "D", c, s)
        compare_configs(c2["cfg"], d["cfg"])
        if c2["summary"]["n_params"] != d["summary"]["n_params"]:
            raise ValueError(f"Số tham số D khác C2: {c} seed {s}")
        n_c2, n_d = c2["summary"].get("train_samples"), d["summary"].get("train_samples")
        if n_c2 is not None and n_d is not None and n_c2 != n_d:
            raise ValueError(f"Số mẫu train khác giữa hai cửa sổ: {c} seed {s}")
        if any(n is not None and n != 1_112_307 for n in (n_c2, n_d)):
            raise ValueError("Số mẫu train khác mốc B2 đã đối chiếu cho hai cửa sổ")
        config_rows.append(dict(configuration=c, seed=s, seq_length_365=365, seq_length_730=730,
                                n_params=d["summary"]["n_params"], configuration_difference="seq_length",
                                train_samples_365=n_c2, train_samples_730=n_d,
                                sample_counts_verified=n_c2 is not None and n_d is not None,
                                grad_guard_enabled_365=c2["grad_guard_enabled"],
                                grad_guard_skipped_steps_365=c2["grad_guard_skipped_steps"],
                                grad_guard_enabled_730=d["grad_guard_enabled"],
                                grad_guard_skipped_steps_730=d["grad_guard_skipped_steps"],
                                strict_pair_equivalent=(c2["grad_guard_skipped_steps"] == 0
                                                        and d["grad_guard_skipped_steps"] == 0),
                                scaler_365_sha256=sha256(c2["files"]["train_data/train_data_scaler.yml"]),
                                scaler_730_sha256=sha256(d["files"]["train_data/train_data_scaler.yml"])))
        ref, _ = c2_forecast(record, context)
        pred, path, audit = export_d(d, ref, record, contracts)
        exports.append(path)
        checked.append(audit)
        pairs.append((ref,pred))
        val_pairs[(c, s)] = (float(c2["summary"]["val_nse_median_best_model"]),
                             float(d["summary"]["val_nse_median_best_model"]))
        d_cost_info = run_costs(d["summary"])
        handoff_rows.append(dict(
            configuration=c, seed=s, run=pred.name, notebook=d["env"]["notebook"],
            sessions=d_cost_info["sessions"], run_dir=d["env"]["run_dir"],
            selected_epoch=int(d["summary"]["saved_epoch"]), n_params=int(d["summary"]["n_params"]),
            library_commit=FREEZE_C, seq_length=730,
            grad_guard_enabled=d["grad_guard_enabled"],
            grad_guard_skipped_steps=d["grad_guard_skipped_steps"],
            validation_NSE=float(d["summary"]["val_nse_median_best_model"]),
            test_NSE_native=float(audit["native_nse_median"]),
            demo=False,
            sha256_best_model_pt=sha256(d["files"]["best_model.pt"]),
            sha256_config_yml=sha256(d["files"]["config.yml"]),
            sha256_train_data_scaler_yml=sha256(d["files"]["train_data/train_data_scaler.yml"]),
            contract_npz=path.relative_to(output).as_posix(),
            sha256_contract_npz=audit["SHA256"],
        ))
        for length, source in ((365,c2),(730,d)):
            summary = source["summary"]
            costs.append(dict(configuration=c, seed=s, seq_length=length, notebook=source["env"]["notebook"],
                              GPU=source["env"].get("gpu"), validation_NSE=summary["val_nse_median_best_model"],
                              **run_costs(summary)))
    if {p.resolve() for p in contracts.glob("*.npz")} != {p.resolve() for p in exports}:
        raise ValueError("Thư mục contracts chứa kết quả lần trước. Chạy Save & Run All từ phiên sạch.")
    basins, seeds, mask = score_d(pairs, val_pairs=val_pairs, b0_noise=b0_noise)
    ensemble = ensemble_d(pairs, configs, mask, seeds, b0_noise=b0_noise)
    handoff_df = pd.DataFrame(handoff_rows)
    demo_rows = []
    for conf in configs:
        ordered = (handoff_df.loc[handoff_df.configuration.eq(conf)]
                   .sort_values(["validation_NSE", "seed"], ascending=[False, True]))
        best = ordered.iloc[0]
        gap = float(best["validation_NSE"] - ordered.iloc[1]["validation_NSE"]) if len(ordered) > 1 else np.nan
        tie = bool((ordered["validation_NSE"] == best["validation_NSE"]).sum() > 1)
        handoff_df.loc[(handoff_df.configuration == conf) & (handoff_df.seed == int(best["seed"])), "demo"] = True
        demo_rows.append(dict(
            configuration=conf, seed=int(best["seed"]), run=best["run"],
            selected_epoch=int(best["selected_epoch"]),
            validation_NSE=float(best["validation_NSE"]),
            gap_to_second_seed=gap, tie=tie,
            test_NSE_native=float(best["test_NSE_native"]),
            notebook=best["notebook"], sessions=int(best["sessions"]),
            run_dir=best["run_dir"], contract_npz=best["contract_npz"],
        ))
    demo_df = pd.DataFrame(demo_rows)
    cost = pd.DataFrame(costs)
    wide = cost.pivot(index=["configuration","seed"], columns="seq_length", values=list(COST_KEYS))
    ratio = pd.DataFrame(index=wide.index)
    for k in COST_KEYS:
        ratio[f"{k}_730_over_365"] = wide[k][730].div(wide[k][365].where(wide[k][365] > 0))
    tables = dict(D_config_check=pd.DataFrame(config_rows), D_contract_check=pd.DataFrame(checked),
                  D_basin_scores=basins, D_seed_scores=seeds,
                  D_score_stats=seeds.drop(columns="seed").groupby("configuration").agg(["mean","std"]),
                  D_ensemble_scores=ensemble, D_cost=cost, D_cost_ratios=ratio.reset_index(),
                  D_demo_runs=demo_df, D_handoff=handoff_df)
    tables["D_score_stats"].columns = ["_".join(col) for col in tables["D_score_stats"].columns]
    tables["D_score_stats"] = tables["D_score_stats"].reset_index()
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    show_table("Kiểm cấu hình cặp 365/730 (D_config_check.csv)", tables["D_config_check"])
    show_table("Kiểm hợp đồng NPZ D (D_contract_check.csv)", tables["D_contract_check"])
    show_table("Điểm từng hạt giống trên giao chung (D_seed_scores.csv)", tables["D_seed_scores"])
    show_table("Tổ hợp 3 hạt giống 365/730 (D_ensemble_scores.csv)", tables["D_ensemble_scores"])
    show_table("Tỷ số chi phí 730 chia 365 (D_cost_ratios.csv)", tables["D_cost_ratios"])
    show_table("Run D chọn cho demo theo validation (D_demo_runs.csv)", tables["D_demo_runs"])
    np.savez_compressed(output / "D_common_mask.npz", dates=context["dates"].to_numpy(dtype="datetime64[D]"),
                        basin_ids=np.array(context["ids"]), mask=mask)
    common_ids = int(mask.any(axis=0).sum())
    draw_d(seeds, ratio.reset_index(), configs, output, dates=context["dates"], common_basins=common_ids)
    inputs = dict(C2_compare_version_expected=1, C2_manifest_sha256=sha256(c2_root/"manifest.csv"),
                  A3={p.name:sha256(p) for p in a3_paths})
    skipped_runs = [f"D_{row['configuration']}_seed{row['seed']}"
                    for row in config_rows if row["grad_guard_skipped_steps_730"] > 0]
    summary = dict(complete=True, notebook=NOTEBOOK, step="D", library_commit=FREEZE_C, configs=list(configs),
                   seeds=list(C2_SEEDS), S4D_included=s4, contracts=len(exports), IDs=len(context["ids"]),
                   forecast_horizon=1, b0_noise_sd=b0_noise, common_day_IDs=int(mask.sum()), IDs_with_days=common_ids,
                   train_sample_counts_verified=all(row["sample_counts_verified"] for row in config_rows),
                   grad_guard_runs_with_skips=skipped_runs,
                   all_pairs_strictly_equivalent=all(row["strict_pair_equivalent"] for row in config_rows),
                   demo_runs={r["configuration"]: {"seed": r["seed"], "run": r["run"],
                                                   "validation_NSE": r["validation_NSE"],
                                                   "notebook": r["notebook"]} for r in demo_rows},
                   scope="test_S0_common_all_C2_and_D", inputs=inputs,
                   output_contracts=[p.relative_to(output).as_posix() for p in exports])
    summary_path.write_text(json.dumps(summary, default=scalar_json, ensure_ascii=False, indent=2), encoding="utf-8")
    source_rows = [dict(stage=stage, configuration=c, seed=seed, file=str(path), sha256=sha256(path))
                   for (stage,c,seed), path in runs.items() if (c,seed) in selected]
    pd.DataFrame(source_rows).to_csv(output/"D_input_sources.csv", index=False, encoding="utf-8-sig")
    environment = dict(notebook=NOTEBOOK, source_neuralhydrology_commit=FREEZE_C, python=sys.version.split()[0],
                       numpy=np.__version__, pandas=pd.__version__, A3_version_expected=1, **inputs)
    env_path = output/"environment.json"
    env_path.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    paths = [*exports, output/"D_input_sources.csv", *(output/f"{name}.csv" for name in tables), output/"D_common_mask.npz",
             output/"D_Length_Comparison.png", summary_path, env_path]
    pd.DataFrame([dict(file=p.relative_to(output).as_posix(), bytes=p.stat().st_size, sha256=sha256(p))
                  for p in paths]).to_csv(output / "manifest.csv", index=False)
    if skipped_runs:
        print(f"CẢNH BÁO: có run bỏ bước cập nhật do tràn gradient {skipped_runs}; "
              "rà lại tính tương đương cặp 365/730 trước khi đưa vào so sánh chính.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("GỬI LẠI: d_summary.json, D_config_check.csv, D_contract_check.csv, D_seed_scores.csv, "
          "D_score_stats.csv, D_ensemble_scores.csv, D_cost.csv, D_cost_ratios.csv, D_demo_runs.csv, "
          "D_handoff.csv, D_Length_Comparison.png, manifest.csv.")
    return summary


CORE_LABELS = {"lstm": "LSTM", "mamba": "Mamba", "s4d": "S4D"}


def caption(text):
    print(f"Hình: {text}")


def vn_num(value, digits=4, signed=False):
    if value is None or not np.isfinite(value):
        return "không xác định"
    fmt = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return fmt.replace(".", ",")


def draw_d(scores, costs, configs, output, dates=None, common_basins=None):
    import matplotlib.pyplot as plt

    period_txt = (f"{dates[0].year}-{dates[-1].year}" if dates is not None and len(dates)
                  else f"{TEST_START[:4]}-{TEST_END[:4]}")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panels = ((scores, "delta_NSE", "(a) Test median NSE difference", "730 minus 365 days", 0.0),
              (costs, "train_minutes_per_epoch_median_730_over_365", "(b) Training time per epoch ratio",
               "730 / 365 days", 1.0))
    labels = [CORE_LABELS.get(c.split("_")[0], c.split("_")[0].upper()) for c in configs]
    for ax, (data, key, title, ylabel, reference) in zip(axes, panels):
        for i, conf in enumerate(configs):
            vals = data.loc[data.configuration.eq(conf)].sort_values("seed")[key].to_numpy(float)
            offset = np.linspace(-0.09, 0.09, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(i + offset, vals, color="tab:blue", zorder=3,
                       label="Individual seeds" if i == 0 else None)
            if len(vals) >= 2 and np.isfinite(vals).all():
                ax.errorbar(i + 0.22, vals.mean(), yerr=vals.std(ddof=1), fmt="D", color="tab:orange",
                            capsize=3, zorder=4, label="Mean +/- sample SD" if i == 0 else None)
        ax.axhline(reference, color="gray", linestyle="--", linewidth=0.9)
        ax.set_xticks(range(len(configs)), labels)
        ax.set_xlim(-0.45, len(configs) - 0.35)
        ax.set(title=title, ylabel=ylabel, xlabel="Core architecture (x1 parameter budget)")
        ax.grid(axis="y", alpha=0.25)
    handles, names = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, names, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=9)
    fig.suptitle(f"D Sequence-Length Comparison (730 vs. 365 days, test {period_txt}, 1-day lead)")
    fig.tight_layout()
    fig.savefig(output / "D_Length_Comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    n_seeds = int(scores.groupby("configuration")["seed"].nunique().min()) if len(scores) else 0
    basins_vn = f"{common_basins:,}".replace(",", ".") if common_basins is not None else "các"
    nse_parts, cost_parts = [], []
    for conf, lab in zip(configs, labels):
        d_vals = scores.loc[scores.configuration.eq(conf), "delta_NSE"].to_numpy(float)
        c_vals = costs.loc[costs.configuration.eq(conf), "train_minutes_per_epoch_median_730_over_365"].to_numpy(float)
        nse_sd = d_vals.std(ddof=1) if len(d_vals) > 1 else np.nan
        nse_parts.append(f"{lab} {vn_num(d_vals.mean(), 4, signed=True)} +/- {vn_num(nse_sd, 4)}")
        cost_parts.append(f"{lab} {vn_num(np.nanmean(c_vals), 2)} lần")
    caption(f"so sánh độ dài chuỗi đầu vào 730 ngày so với 365 ngày trên kỳ test {period_txt} "
            f"({basins_vn} lưu vực có ngày chung, hạn dự báo 1 ngày, {n_seeds} hạt giống mỗi lõi). "
            f"(a) Hiệu NSE trung vị giữa cửa sổ 730 và 365 ngày (trung bình +/- độ lệch chuẩn mẫu: "
            f"{'; '.join(nse_parts)}; vạch đứt mức 0 là ngưỡng không đổi). "
            f"(b) Tỷ số thời gian huấn luyện trung vị mỗi epoch của 730 chia cho 365 ngày "
            f"(trung bình: {'; '.join(cost_parts)}; vạch đứt mức 1 là chi phí tương đương).")


if __name__ == "__main__":
    compare(Path("/kaggle/input"), Path("/kaggle/working/d_compare"))
