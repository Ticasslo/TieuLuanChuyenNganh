# %% [markdown]
# # D — So sánh cửa sổ 365 và 730 bước
#
# Kaggle CPU, Internet Off. Input là C2 Compare Version 1 và A3 Version 1,
# thêm các Output huấn luyện C2 và D cùng cấu hình/seed. Mã nguồn nhận
# Output hoàn thành có cấu trúc summary/environment/manifest như khuôn C.
# Chuỗi 365/730 gồm 364/729 bước lịch sử và một ngày dự báo ECMWF.
#
# 1. Hợp đồng và chỉ số dùng chung. 2. Kiểm nguồn và cấu hình.
# 3. Xuất NPZ. 4. Chấm giao chung, tổ hợp, chi phí và bàn giao.
# 5. Trực quan hóa đa tiêu chí (9 hình từ tổng quan đến chẩn đoán nâng cao).

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
    curve = None
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
    return dict(files=chosen, summary=s, cfg=cfg, env=env, env_path=env_file, curve=curve,
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


def flow_signature_biases(obs, sim, mask):
    # %BiasFHV (top 2% đường duy trì dòng chảy, Yilmaz và cs. 2008), sai số ngày lưu lượng rất cao (>= Q98)
    # và sai số ngày lưu lượng thấp (<= Q30) trên giao chung của từng lưu vực.
    y, yhat = np.asarray(obs, dtype=float)[mask], np.asarray(sim, dtype=float)[mask]
    if len(y) == 0:
        return dict(FHV=np.nan, hi_bias=np.nan, lo_bias=np.nan)
    k2 = max(1, int(math.ceil(0.02 * len(y))))
    so = np.sort(y)[-k2:]
    ss = np.sort(yhat)[-k2:]
    fhv = 100.0 * float(ss.sum() - so.sum()) / float(so.sum()) if so.sum() > 0 else np.nan
    hi = y >= float(np.quantile(y, 0.98))
    hi_den = float(y[hi].sum())
    hi_bias = 100.0 * float(yhat[hi].sum() - hi_den) / hi_den if hi.any() and hi_den > 0 else np.nan
    lo = y <= float(np.quantile(y, 0.30))
    lo_den = float(y[lo].sum())
    lo_bias = 100.0 * float(yhat[lo].sum() - lo_den) / lo_den if lo.any() and lo_den > 0 else np.nan
    return dict(FHV=fhv, hi_bias=hi_bias, lo_bias=lo_bias)


def ensemble_basin_diagnostics(pairs, configs, shared):
    # Điểm từng lưu vực của tổ hợp 3 hạt giống (NSE, PNSE, KGE, r, alpha, beta, %BiasFHV, hi_bias, lo_bias)
    # kèm diện tích area_gov_km2 để phân tầng và chẩn đoán chế độ dòng chảy.
    rows = []
    for conf in configs:
        sub = [(c, d) for c, d in pairs if d.metadata["config"] == conf]
        sim_365 = np.mean([c.sim for c, _ in sub], axis=0)
        sim_730 = np.mean([d.sim for _, d in sub], axis=0)
        base = sub[0][0]
        for j, bid in enumerate(base.ids):
            m_col = shared[:, j]
            m3 = metrics(base.obs[:, j], sim_365[:, j], base.persistence[:, j], m_col)
            m7 = metrics(base.obs[:, j], sim_730[:, j], base.persistence[:, j], m_col)
            b3 = flow_signature_biases(base.obs[:, j], sim_365[:, j], m_col)
            b7 = flow_signature_biases(base.obs[:, j], sim_730[:, j], m_col)
            row = dict(configuration=conf, basin_id=bid, area_gov_km2=float(base.area[j]),
                       common_days=int(m_col.sum()))
            for k in ("NSE", "PNSE", "KGE", "r", "alpha", "beta"):
                row[f"{k}_365"] = m3[k]
                row[f"{k}_730"] = m7[k]
                row[f"delta_{k}"] = m7[k] - m3[k]
            for k in ("FHV", "hi_bias", "lo_bias"):
                row[f"{k}_365"] = b3[k]
                row[f"{k}_730"] = b7[k]
                row[f"delta_{k}"] = b7[k] - b3[k]
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
    checked, pairs, costs, exports, config_rows, handoff_rows, curve_rows = [], [], [], [], [], [], []
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
            configuration=c, core=str(d["summary"]["core"]), seed=s, run=pred.name, notebook=d["env"]["notebook"],
            sessions=d_cost_info["sessions"], run_dir=d["env"]["run_dir"],
            selected_epoch=int(d["summary"]["saved_epoch"]), n_params=int(d["summary"]["n_params"]),
            library_commit=FREEZE_C, seq_length=730,
            grad_guard_enabled=d["grad_guard_enabled"],
            grad_guard_skipped_steps=d["grad_guard_skipped_steps"],
            validation_NSE=float(d["summary"]["val_nse_median_best_model"]),
            test_NSE_native=float(audit["native_nse_median"]),
            demo=False,
            sha256_environment_json=sha256(d["env_path"]),
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
            if source["curve"] is not None:
                cdf = source["curve"]
                sel_ep = int(summary["saved_epoch"])
                for _, crow in cdf.iterrows():
                    curve_rows.append(dict(
                        configuration=c, seed=s, seq_length=length, epoch=int(crow["epoch"]),
                        selected_epoch=sel_ep,
                        val_NSE=float(crow["NSE trung vị validation (tệp)"]),
                        test_NSE=float(crow["NSE trung vị test"]),
                        train_loss=float(crow["loss huấn luyện"]) if "loss huấn luyện" in cdf else np.nan,
                        val_loss=float(crow["loss validation"]) if "loss validation" in cdf else np.nan,
                    ))
    if {p.resolve() for p in contracts.glob("*.npz")} != {p.resolve() for p in exports}:
        raise ValueError("Thư mục contracts chứa kết quả lần trước. Chạy Save & Run All từ phiên sạch.")
    basins, seeds, mask = score_d(pairs, val_pairs=val_pairs, b0_noise=b0_noise)
    ensemble = ensemble_d(pairs, configs, mask, seeds, b0_noise=b0_noise)
    ens_basins = ensemble_basin_diagnostics(pairs, configs, mask)
    curves_df = pd.DataFrame(curve_rows)
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
                  D_basin_scores=basins, D_ensemble_basin_scores=ens_basins, D_seed_scores=seeds,
                  D_score_stats=seeds.drop(columns="seed").groupby("configuration").agg(["mean","std"]),
                  D_ensemble_scores=ensemble, D_cost=cost, D_cost_ratios=ratio.reset_index(),
                  D_demo_runs=demo_df, D_handoff=handoff_df)
    if not curves_df.empty:
        tables["D_curves"] = curves_df
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
    figure_paths = draw_all_d(
        seeds=seeds, ensemble=ensemble, ens_basins=ens_basins, cost=cost, ratios=ratio.reset_index(),
        configs=configs, output=output, bstar=c2_summary.get("bstar"), b0_noise=b0_noise,
        curves=curves_df if not curves_df.empty else None, dates=context["dates"], common_basins=common_ids,
    )
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
                   figures=[p.relative_to(output).as_posix() for p in figure_paths],
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
             *figure_paths, summary_path, env_path]
    pd.DataFrame([dict(file=p.relative_to(output).as_posix(), bytes=p.stat().st_size, sha256=sha256(p))
                  for p in paths]).to_csv(output / "manifest.csv", index=False)
    if skipped_runs:
        print(f"CẢNH BÁO: có run bỏ bước cập nhật do tràn gradient {skipped_runs}; "
              "rà lại tính tương đương cặp 365/730 trước khi đưa vào so sánh chính.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fig_list = ", ".join(p.name for p in figure_paths)
    print("GỬI LẠI: d_summary.json, D_config_check.csv, D_contract_check.csv, D_seed_scores.csv, "
          "D_score_stats.csv, D_ensemble_scores.csv, D_ensemble_basin_scores.csv, D_cost.csv, "
          f"D_cost_ratios.csv, D_demo_runs.csv, D_handoff.csv, {fig_list}, manifest.csv.")
    return summary


# %% Phần 5 - Trực quan hóa đa tiêu chí (9 hình từ tổng quan đến phân tích nâng cao)
# Bảng màu lõi thống nhất với C2 Compare, H Compare và XAI: LSTM = C0, Mamba = C4, S4D = C3.
# Toàn bộ chữ trong hình tiếng Anh (thập phân dấu chấm), chú giải ngoài vùng vẽ, 200 dpi;
# mỗi hình gọi plt.show() và in một chú thích tiếng Việt tính trực tiếp từ số liệu thực.
CORE_LABELS = {"lstm": "LSTM", "mamba": "Mamba", "s4d": "S4D"}
CORE_COLOR = {"lstm": "C0", "mamba": "C4", "s4d": "C3"}


def core_key(conf):
    return str(conf).split("_")[0]


def core_label(conf):
    k = core_key(conf)
    return CORE_LABELS.get(k, k.upper())


def core_color(conf):
    return CORE_COLOR.get(core_key(conf), "C7")


def caption(text):
    print(f"Hình: {text}\n")


def vn_num(value, digits=4, signed=False):
    if value is None or not np.isfinite(value):
        return "không xác định"
    fmt = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return fmt.replace(".", ",")


def vn_int(value):
    return f"{int(round(value)):,}".replace(",", ".")


def kde_1d(values, grid):
    # Ước lượng mật độ Gaussian 1D (quy tắc Silverman) thuần NumPy, không phụ thuộc SciPy.
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.zeros_like(grid, dtype=float)
    sd = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sigma = min(sd, iqr / 1.34) if iqr > 0 else sd
    bw = max(0.9 * sigma * (len(x) ** -0.2), 1e-4)
    diff = (grid[:, None] - x[None, :]) / bw
    return np.mean(np.exp(-0.5 * diff * diff), axis=1) / (bw * math.sqrt(2.0 * math.pi))


def bootstrap_median_ci(values, n_boot=2000, seed=42):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(x))
    if len(x) == 1:
        return med, med, med
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return med, float(lo), float(hi)


def style_ax(ax, grid_axis="y", despine=True):
    # Phong cách tối giản chuẩn tạp chí HESS/Nature: bỏ viền trên/phải, lưới mảnh dịu mắt.
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("0.30")
        ax.spines[sp].set_linewidth(0.85)
    ax.tick_params(colors="0.20", length=3.5, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color="0.85", linewidth=0.65, alpha=0.8)
        ax.set_axisbelow(True)


def finish_fig(fig, path, handles=None, ncol=3, legend_y=-0.01, rect=None):
    import matplotlib.pyplot as plt

    if rect is not None:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, legend_y),
                   ncol=ncol, fontsize=8.8, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def draw_overview_accuracy(seeds, ensemble, configs, output, bstar=None, b0_noise=None,
                           period_txt="2014-2017", common_basins=None):
    # Hình 1 (Đơn giản / Tổng quan — đồ thị điểm Cleveland chuẩn C2_Bstar):
    # NSE trung vị validation và test của 365 vs 730 ngày, không chèn chữ bên trong vùng dữ liệu.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), sharey=True)
    ens_idx = ensemble.set_index("configuration")
    rows = [(conf, L) for conf in configs for L in (365, 730)]
    y_pos = []
    cur = 0.0
    for _ in range(len(configs)):
        y_pos.extend([cur, cur + 0.72])
        cur += 1.95
    y_labels = [f"{core_label(c)} ({L}d)" for c, L in rows]

    b_val = float(bstar["val_mean"]) if bstar and bstar.get("val_mean") is not None else None
    b_test = float(bstar["test_mean"]) if bstar and bstar.get("test_mean") is not None else None
    b_ens = float(bstar["ensemble_test_nse"]) if bstar and bstar.get("ensemble_test_nse") is not None else None

    panels = [
        (axes[0], "val", "(a) Validation median NSE (2010-2013)", b_val, None),
        (axes[1], "test", f"(b) Test median NSE ({period_txt})", b_test, b_ens),
    ]
    for ax, kind, title, ref_line, ref_ens in panels:
        style_ax(ax, grid_axis="x", despine=True)
        if ref_line is not None:
            if kind == "val" and b0_noise is not None:
                ax.axvspan(ref_line - b0_noise, ref_line + b0_noise, color="0.91", zorder=0)
            ax.axvline(ref_line, color="0.20", lw=1.0, zorder=1)
        if ref_ens is not None:
            ax.axvline(ref_ens, color="0.20", lw=1.0, ls="--", zorder=1)
        vals_all = [v for v in (ref_line, ref_ens) if v is not None]
        for y, (conf, length) in zip(y_pos, rows):
            col_c = core_color(conf)
            sub = seeds.loc[seeds.configuration.eq(conf)].sort_values("seed")
            col_name = f"val_NSE_{length}" if kind == "val" else f"NSE_{length}"
            s_vals = sub[col_name].to_numpy(float)
            m_val = float(np.mean(s_vals))
            sd_val = float(np.std(s_vals, ddof=1)) if len(s_vals) > 1 else np.nan
            vals_all.extend(s_vals.tolist())
            is_730 = (length == 730)
            mfc = col_c if is_730 else "white"
            mk = "D" if is_730 else "o"
            ax.scatter(s_vals, np.full_like(s_vals, y), facecolor=mfc, edgecolor=col_c,
                       alpha=0.55, s=24, linewidth=1.0, zorder=2)
            ax.errorbar(m_val, y, xerr=sd_val if np.isfinite(sd_val) else None, fmt=mk,
                        color=col_c, mfc=mfc, mec=col_c, mew=1.4, capsize=3.2, ms=7.0, zorder=4)
            if kind == "test":
                e_val = float(ens_idx.loc[conf, f"NSE_ensemble_{length}"])
                vals_all.append(e_val)
                ax.plot(e_val, y, marker="*", color=col_c, mfc=mfc, mec=col_c, mew=1.0, ms=11.5, ls="", zorder=5)
                txt = f"{m_val:.4f} | {e_val:.4f}"
            else:
                txt = f"{m_val:.4f} ± {sd_val:.4f}" if np.isfinite(sd_val) else f"{m_val:.4f}"
            ax.annotate(txt, (1.0, y), xytext=(8, 0), xycoords=("axes fraction", "data"),
                        textcoords="offset points", va="center", ha="left", fontsize=8.5,
                        color=col_c, fontweight="bold" if is_730 else "normal")
        finite_vals = [v for v in vals_all if v is not None and np.isfinite(v)]
        span = max(finite_vals) - min(finite_vals)
        ax.set_xlim(min(finite_vals) - 0.12 * max(span, 0.01), max(finite_vals) + 0.12 * max(span, 0.01))
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_xlabel("Median NSE across 451 basins")
        for idx_c in range(len(configs) - 1):
            sep = 0.5 * (y_pos[2 * idx_c + 1] + y_pos[2 * idx_c + 2])
            ax.axhline(sep, color="0.86", lw=0.8, ls=":")
        hdr = "mean ± SD" if kind == "val" else "mean | ens"
        ax.annotate(hdr, (1.0, y_pos[0] - 0.62), xytext=(8, 0), xycoords=("axes fraction", "data"),
                    textcoords="offset points", va="center", ha="left", fontsize=8.3,
                    fontweight="bold", color="0.30")
    axes[0].set_yticks(y_pos, y_labels)
    axes[0].set_ylim(y_pos[-1] + 0.65, y_pos[0] - 0.95)
    fig.suptitle("Sequence Length 365 vs. 730 Days: Validation and Test Median NSE", fontsize=11.8, y=0.99)
    handles = [
        Line2D([], [], color="0.25", marker="o", mfc="white", mec="0.25", mew=1.4, ls="", ms=7,
               label="365d mean ± SD (dots = seeds)"),
        Line2D([], [], color="0.25", marker="D", mfc="0.25", mec="0.25", mew=1.4, ls="", ms=7,
               label="730d mean ± SD (dots = seeds)"),
        Line2D([], [], color="0.25", marker="*", mfc="0.25", mec="0.25", ls="", ms=11,
               label="3-seed ensemble (hollow = 365d, filled = 730d)"),
    ]
    if b_val is not None:
        handles.append(Line2D([], [], color="0.20", lw=1.0, label=f"B* mean (val {b_val:.4f} / test {b_test:.4f})"))
    if b_ens is not None:
        handles.append(Line2D([], [], color="0.20", lw=1.0, ls="--", label=f"B* ensemble (test {b_ens:.4f})"))
    if b_val is not None and b0_noise is not None:
        handles.append(Rectangle((0, 0), 1, 1, color="0.91", label=f"B* ± B0 noise (±{b0_noise:.4f})"))
    fig.subplots_adjust(wspace=0.45)
    path = output / "D_Overview_Accuracy.png"
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8.6, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    val_parts, test_parts = [], []
    for conf in configs:
        sub = seeds.loc[seeds.configuration.eq(conf)]
        lab = core_label(conf)
        v3, v7 = sub["val_NSE_365"].mean(), sub["val_NSE_730"].mean()
        t3, t7 = sub["NSE_365"].mean(), sub["NSE_730"].mean()
        e3, e7 = ens_idx.loc[conf, "NSE_ensemble_365"], ens_idx.loc[conf, "NSE_ensemble_730"]
        val_parts.append(f"{lab} {vn_num(v3)} -> {vn_num(v7)} ({vn_num(v7 - v3, signed=True)})")
        test_parts.append(f"{lab} TB {vn_num(t3)} -> {vn_num(t7)}, tổ hợp {vn_num(e3)} -> {vn_num(e7)}")
    basins_vn = vn_int(common_basins) if common_basins is not None else "451"
    caption(f"tổng quan NSE trung vị trên kỳ validation 2010-2013 (a, dùng để chọn) và kỳ test {period_txt} (b, giao chung "
            f"{basins_vn} lưu vực) cho cửa sổ 365 ngày (ký hiệu rỗng) so với 730 ngày (ký hiệu đặc). "
            f"Validation: {'; '.join(val_parts)}. Test: {'; '.join(test_parts)}. "
            + (f"Mốc đối chứng B* có trung bình validation {vn_num(b_val)}, test {vn_num(b_test)}, tổ hợp test {vn_num(b_ens)} "
               f"và ngưỡng nhiễu hạt giống B0 {vn_num(b0_noise)}." if b_val is not None and b0_noise is not None else ""))
    return path


def draw_d(scores, costs, configs, output, dates=None, common_basins=None, b0_noise=None, ensemble=None):
    # Hình 2 (Cốt lõi RQ5 — giữ nguyên tên tệp D_Length_Comparison.png bắt buộc cho Evaluation):
    # hiệu NSE test (730 - 365) kèm dải nhiễu B0 và tỷ số thời gian huấn luyện mỗi epoch (730 / 365).
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 10})
    period_txt = (f"{dates[0].year}-{dates[-1].year}" if dates is not None and len(dates)
                  else f"{TEST_START[:4]}-{TEST_END[:4]}")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    labels = [core_label(c) for c in configs]
    ens_idx = ensemble.set_index("configuration") if ensemble is not None else None

    ax = axes[0]
    style_ax(ax, grid_axis="y", despine=True)
    if b0_noise is not None and np.isfinite(b0_noise):
        ax.axhspan(-b0_noise, b0_noise, color="0.92", zorder=0)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.95, zorder=1)
    y_all_a = [-b0_noise, b0_noise, 0.0] if b0_noise is not None else [0.0]
    for i, conf in enumerate(configs):
        col = core_color(conf)
        vals = scores.loc[scores.configuration.eq(conf)].sort_values("seed")["delta_NSE"].to_numpy(float)
        y_all_a.extend(vals.tolist())
        offset = np.linspace(-0.09, 0.09, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(i + offset, vals, color=col, alpha=0.58, s=34, zorder=3)
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        ax.errorbar(i + 0.21, mu, yerr=sd if np.isfinite(sd) else None, fmt="D", color=col,
                    mec="black", mew=0.55, capsize=3.2, ms=6.8, zorder=4)
        y_all_a.extend([mu + (sd if np.isfinite(sd) else 0.0), mu - (sd if np.isfinite(sd) else 0.0)])
        if ens_idx is not None and conf in ens_idx.index:
            de = float(ens_idx.loc[conf, "delta_NSE_ensemble"])
            y_all_a.append(de)
            ax.plot(i - 0.21, de, marker="*", color=col, mec="black", mew=0.5, ms=11.5, ls="", zorder=5)
    span_a = max(y_all_a) - min(y_all_a)
    ax.set_ylim(min(y_all_a) - 0.14 * span_a, max(y_all_a) + 0.14 * span_a)
    ax.set_xticks(range(len(configs)), labels)
    ax.set_xlim(-0.48, len(configs) - 0.45)
    ax.set(title="(a) Test median NSE change (730 - 365 days)", ylabel="ΔNSE (730 minus 365 days)",
           xlabel="Core architecture (x1 budget)")

    ax = axes[1]
    style_ax(ax, grid_axis="y", despine=True)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.95, zorder=1)
    ax.axhline(2.0, color="0.55", linestyle=":", linewidth=0.95, zorder=1)
    y_all_b = [1.0, 2.0]
    for i, conf in enumerate(configs):
        col = core_color(conf)
        vals = costs.loc[costs.configuration.eq(conf)].sort_values("seed")[
            "train_minutes_per_epoch_median_730_over_365"].to_numpy(float)
        y_all_b.extend(vals.tolist())
        offset = np.linspace(-0.09, 0.09, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(i + offset, vals, color=col, alpha=0.58, s=34, zorder=3)
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan
        ax.errorbar(i + 0.21, mu, yerr=sd if np.isfinite(sd) else None, fmt="D", color=col,
                    mec="black", mew=0.55, capsize=3.2, ms=6.8, zorder=4)
        y_all_b.extend([mu + (sd if np.isfinite(sd) else 0.0), mu - (sd if np.isfinite(sd) else 0.0)])
    ax.set_ylim(0.92, max(y_all_b) + 0.14)
    ax.set_xticks(range(len(configs)), labels)
    ax.set_xlim(-0.48, len(configs) - 0.45)
    ax.set(title="(b) Training time ratio per epoch (730 / 365 days)", ylabel="Ratio (730 / 365 days)",
           xlabel="Core architecture (x1 budget)")

    handles = [
        Line2D([], [], color="0.3", marker="o", alpha=0.58, ls="", ms=6.0, label="Individual seeds (111, 222, 333)"),
        Line2D([], [], color="0.3", marker="D", mec="black", mew=0.55, ls="", ms=6.8, label="Seed mean ± SD"),
    ]
    if ens_idx is not None:
        handles.append(Line2D([], [], color="0.3", marker="*", mec="black", mew=0.5, ls="", ms=11,
                              label="3-seed ensemble ΔNSE"))
    if b0_noise is not None and np.isfinite(b0_noise):
        handles.append(Rectangle((0, 0), 1, 1, color="0.92", label=f"± B0 seed noise (±{b0_noise:.4f})"))
    handles.append(Line2D([], [], color="0.55", ls=":", lw=1.0, label="2.0× linear scaling (panel b)"))
    fig.suptitle(f"Accuracy Gain vs. Training Time Overhead (730 vs. 365 days, test {period_txt})", fontsize=11.5)
    path = finish_fig(fig, output / "D_Length_Comparison.png", handles=handles, ncol=3, legend_y=-0.01)

    n_seeds = int(scores.groupby("configuration")["seed"].nunique().min()) if len(scores) else 0
    basins_vn = vn_int(common_basins) if common_basins is not None else "các"
    nse_parts, cost_parts = [], []
    for conf, lab in zip(configs, labels):
        d_vals = scores.loc[scores.configuration.eq(conf), "delta_NSE"].to_numpy(float)
        c_vals = costs.loc[costs.configuration.eq(conf), "train_minutes_per_epoch_median_730_over_365"].to_numpy(float)
        nse_sd = d_vals.std(ddof=1) if len(d_vals) > 1 else np.nan
        ens_txt = ""
        if ens_idx is not None and conf in ens_idx.index:
            ens_txt = f", tổ hợp {vn_num(ens_idx.loc[conf, 'delta_NSE_ensemble'], 4, signed=True)}"
        nse_parts.append(f"{lab} {vn_num(d_vals.mean(), 4, signed=True)} ± {vn_num(nse_sd, 4)}{ens_txt}")
        cost_parts.append(f"{lab} {vn_num(np.nanmean(c_vals), 2)} lần")
    caption(f"so sánh cốt lõi độ dài chuỗi đầu vào 730 ngày so với 365 ngày trên kỳ test {period_txt} "
            f"({basins_vn} lưu vực có ngày chung, hạn dự báo 1 ngày, {n_seeds} hạt giống mỗi lõi). "
            f"(a) Hiệu NSE trung vị giữa cửa sổ 730 và 365 ngày ({'; '.join(nse_parts)}; "
            + (f"dải xám là ± ngưỡng nhiễu hạt giống B0 {vn_num(b0_noise)}, " if b0_noise is not None else "")
            + f"vạch đứt mức 0 là không đổi). "
            f"(b) Tỷ số thời gian huấn luyện trung vị mỗi epoch của 730 chia cho 365 ngày "
            f"(trung bình: {'; '.join(cost_parts)}; vạch chấm mức 2,0 lần là tỉ lệ tuyến tính theo độ dài chuỗi).")
    return path


def draw_multimetric_ensemble(seeds, ensemble, configs, output, period_txt="2014-2017", common_basins=None):
    # Hình 3 (Đa tiêu chí — đồ thị quả tạ + điểm hạt giống + sao tổ hợp, không dùng cột cắt trục 0):
    # So sánh đồng thời 3 chỉ số chuẩn thủy văn NSE, PNSE và KGE giữa 365 và 730 ngày.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.4))
    ens_idx = ensemble.set_index("configuration")
    labels = [core_label(c) for c in configs]
    x = np.arange(len(configs))
    dx = 0.16
    metric_specs = [
        ("NSE", "(a) Test median NSE", "Median NSE"),
        ("PNSE", "(b) Test median PNSE", "Median PNSE"),
        ("KGE", "(c) Test median KGE", "Median KGE"),
    ]
    for ax, (m, title, ylabel) in zip(axes, metric_specs):
        style_ax(ax, grid_axis="y", despine=True)
        all_v = []
        for i, conf in enumerate(configs):
            col = core_color(conf)
            sub = seeds.loc[seeds.configuration.eq(conf)]
            s3, s7 = sub[f"{m}_365"].to_numpy(float), sub[f"{m}_730"].to_numpy(float)
            m3, m7 = float(np.mean(s3)), float(np.mean(s7))
            sd3 = float(np.std(s3, ddof=1)) if len(s3) > 1 else np.nan
            sd7 = float(np.std(s7, ddof=1)) if len(s7) > 1 else np.nan
            e3, e7 = float(ens_idx.loc[conf, f"{m}_ensemble_365"]), float(ens_idx.loc[conf, f"{m}_ensemble_730"])
            all_v.extend([*s3.tolist(), *s7.tolist(), e3, e7])
            # Đường nối độ dốc trung bình hạt giống và tổ hợp 3 hạt giống giữa 365d và 730d.
            ax.plot([i - dx, i + dx], [m3, m7], color=col, lw=1.6, alpha=0.65, zorder=2)
            ax.plot([i - dx, i + dx], [e3, e7], color=col, lw=1.6, ls="--", alpha=0.80, zorder=2)
            ax.scatter(np.full_like(s3, i - dx), s3, facecolor="white", edgecolor=col, alpha=0.65, s=22, lw=1.0, zorder=3)
            ax.scatter(np.full_like(s7, i + dx), s7, facecolor=col, edgecolor="white", alpha=0.65, s=22, lw=0.5, zorder=3)
            ax.errorbar(i - dx, m3, yerr=sd3 if np.isfinite(sd3) else None, fmt="o", color=col,
                        mfc="white", mec=col, mew=1.5, capsize=2.8, ms=6.8, zorder=4)
            ax.errorbar(i + dx, m7, yerr=sd7 if np.isfinite(sd7) else None, fmt="D", color=col,
                        mfc=col, mec="black", mew=0.6, capsize=2.8, ms=6.6, zorder=4)
            ax.plot(i - dx, e3, marker="*", color=col, mfc="white", mec=col, mew=1.1, ms=12.0, ls="", zorder=5)
            ax.plot(i + dx, e7, marker="*", color=col, mfc=col, mec="black", mew=0.55, ms=12.0, ls="", zorder=5)
        lo, hi = min(all_v), max(all_v)
        span = max(hi - lo, 0.025)
        ax.set_ylim(lo - 0.16 * span, hi + 0.16 * span)
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.55, len(configs) - 0.45)
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Core (left: 365d, right: 730d)")
    handles = [
        Line2D([], [], color="0.30", marker="o", mfc="white", mec="0.30", mew=1.5, ls="-", lw=1.5, ms=6.8,
               label="365d seed mean ± SD"),
        Line2D([], [], color="0.30", marker="D", mfc="0.30", mec="black", mew=0.6, ls="-", lw=1.5, ms=6.6,
               label="730d seed mean ± SD"),
        Line2D([], [], color="0.30", marker="*", mfc="white", mec="0.30", ls="--", lw=1.5, ms=11,
               label="365d 3-seed ensemble"),
        Line2D([], [], color="0.30", marker="*", mfc="0.30", mec="black", ls="--", lw=1.5, ms=11,
               label="730d 3-seed ensemble"),
    ]
    fig.suptitle(f"Multi-Metric Skill Comparison: NSE, PNSE and KGE (365 vs. 730 days, test {period_txt})", fontsize=11.8)
    path = finish_fig(fig, output / "D_MultiMetric_Ensemble.png", handles=handles, ncol=4, legend_y=-0.01)

    parts = []
    for conf in configs:
        lab = core_label(conf)
        r = ens_idx.loc[conf]
        parts.append(f"{lab}: ΔNSE tổ hợp {vn_num(r['delta_NSE_ensemble'], 4, signed=True)}, "
                     f"ΔPNSE {vn_num(r['delta_PNSE_ensemble'], 4, signed=True)}, "
                     f"ΔKGE {vn_num(r['delta_KGE_ensemble'], 4, signed=True)}")
    basins_vn = vn_int(common_basins) if common_basins is not None else "451"
    caption(f"so sánh đa tiêu chí trên kỳ test {period_txt} ({basins_vn} lưu vực giao chung) bằng cặp điểm 365 ngày "
            f"(trái, ký hiệu rỗng) nối sang 730 ngày (phải, ký hiệu đặc) cho (a) NSE, (b) PNSE và (c) KGE; nét liền nối "
            f"trung bình 3 hạt giống, nét đứt nối tổ hợp 3 hạt giống (dấu sao). Chênh lệch tổ hợp (730 trừ 365): "
            f"{'; '.join(parts)}. Tổ hợp 3 hạt giống luôn vượt trung bình hạt giống đơn trên NSE và PNSE.")
    return path


def draw_cost_tradeoff(ensemble, cost, ratios, configs, output, period_txt="2014-2017"):
    # Hình 4 (Hiệu năng - Chi phí): (a) Quỹ đạo Pareto 365 -> 730 tối giản (chỉ ghi tên lõi gọn),
    # (b) Tổng giờ huấn luyện 30 epoch (gốc 0), (c) Đỉnh VRAM Tesla T4 (gốc 0, chỉ ghi hệ số nhân trên cột 730d).
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), gridspec_kw={"width_ratios": [1.4, 1.0, 1.0]})
    ens_idx = ensemble.set_index("configuration")
    c_agg = cost.groupby(["configuration", "seq_length"]).agg(
        train_min=("train_minutes_per_epoch_median", "mean"),
        train_hrs=("train_hours", "mean"),
        gpu_mib=("gpu_mib_max", "max"),
    )

    ax = axes[0]
    style_ax(ax, grid_axis="both", despine=True)
    xs, ys = [], []
    for conf in configs:
        col = core_color(conf)
        lab = core_label(conf)
        x3, x7 = float(c_agg.loc[(conf, 365), "train_min"]), float(c_agg.loc[(conf, 730), "train_min"])
        y3, y7 = float(ens_idx.loc[conf, "NSE_ensemble_365"]), float(ens_idx.loc[conf, "NSE_ensemble_730"])
        xs.extend([x3, x7])
        ys.extend([y3, y7])
        ax.plot(x3, y3, marker="o", ms=8.8, mfc="white", mec=col, mew=2.0, ls="", zorder=4)
        ax.plot(x7, y7, marker="D", ms=8.0, mfc=col, mec="black", mew=0.65, ls="", zorder=5)
        ax.annotate("", xy=(x7, y7), xytext=(x3, y3),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, shrinkA=7, shrinkB=7, alpha=0.9), zorder=3)
        mid_x, mid_y = math.sqrt(x3 * x7), 0.5 * (y3 + y7)
        y_off, va = (-11, "top") if lab == "Mamba" else (9, "bottom")
        ax.annotate(lab, (mid_x, mid_y), xytext=(0, y_off), textcoords="offset points",
                    ha="center", va=va, fontsize=9.2, fontweight="bold", color=col, zorder=6)
    ax.set_xscale("log")
    xmin, xmax = min(xs) / 1.35, max(xs) * 1.38
    ax.set_xlim(xmin, xmax)
    ticks = [t for t in (2, 3, 5, 8, 10, 15, 20, 30) if xmin <= t <= xmax]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    yspan = max(ys) - min(ys)
    ax.set_ylim(min(ys) - 0.28 * max(yspan, 0.015), max(ys) + 0.28 * max(yspan, 0.015))
    ax.set(title="(a) Ensemble NSE vs. epoch time", xlabel="Training min / epoch (log scale, T4)",
           ylabel="Test 3-seed ensemble median NSE")

    x_pos = np.arange(len(configs))
    w = 0.35
    ax_h = axes[1]
    style_ax(ax_h, grid_axis="y", despine=True)
    max_h = 0.0
    for i, conf in enumerate(configs):
        col = core_color(conf)
        h3, h7 = float(c_agg.loc[(conf, 365), "train_hrs"]), float(c_agg.loc[(conf, 730), "train_hrs"])
        max_h = max(max_h, h3, h7)
        ax_h.bar(i - w / 2, h3, width=w * 0.86, color=col, alpha=0.28, edgecolor=col, lw=1.2, zorder=2)
        ax_h.bar(i + w / 2, h7, width=w * 0.86, color=col, alpha=0.84, edgecolor=col, lw=1.2, zorder=2)
        ax_h.text(i + w / 2, h7 + 0.25, f"{h7 / h3:.2f}×", ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold", color=col)
    ax_h.set_xticks(x_pos, [core_label(c) for c in configs])
    ax_h.set_ylim(0, max_h * 1.18)
    ax_h.set(title="(b) 30-epoch training hours", ylabel="Total hours (mean of 3 seeds)", xlabel="Core architecture")

    ax_v = axes[2]
    style_ax(ax_v, grid_axis="y", despine=True)
    max_g = 0.0
    for i, conf in enumerate(configs):
        col = core_color(conf)
        g3, g7 = float(c_agg.loc[(conf, 365), "gpu_mib"]) / 1024.0, float(c_agg.loc[(conf, 730), "gpu_mib"]) / 1024.0
        max_g = max(max_g, g3, g7)
        ax_v.bar(i - w / 2, g3, width=w * 0.86, color=col, alpha=0.28, edgecolor=col, lw=1.2, zorder=2)
        ax_v.bar(i + w / 2, g7, width=w * 0.86, color=col, alpha=0.84, edgecolor=col, lw=1.2, zorder=2)
        ax_v.text(i + w / 2, g7 + 0.08, f"{g7 / g3:.2f}×", ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold", color=col)
    ax_v.set_xticks(x_pos, [core_label(c) for c in configs])
    ax_v.set_ylim(0, max_g * 1.18)
    ax_v.set(title="(c) Peak GPU memory", ylabel="Peak VRAM (GiB, Tesla T4)", xlabel="Core architecture")

    handles = [
        Line2D([], [], color="0.25", marker="o", mfc="white", mec="0.25", mew=1.8, ls="", ms=7.8, label="365 days (panel a)"),
        Line2D([], [], color="0.25", marker="D", mfc="0.25", mec="black", ls="", ms=7.2, label="730 days (arrow = 365 -> 730)"),
        Patch(facecolor="0.45", alpha=0.28, edgecolor="0.35", lw=1.1, label="365 days (panels b, c)"),
        Patch(facecolor="0.45", alpha=0.84, edgecolor="0.35", lw=1.1, label="730 days (panels b, c)"),
    ]
    fig.suptitle(f"Compute–Accuracy Tradeoff: 365 vs. 730 Days (Tesla T4, test {period_txt})", fontsize=11.8)
    path = finish_fig(fig, output / "D_Cost_Tradeoff.png", handles=handles, ncol=4, legend_y=-0.01)

    parts = []
    for conf in configs:
        lab = core_label(conf)
        x3, x7 = c_agg.loc[(conf, 365), "train_min"], c_agg.loc[(conf, 730), "train_min"]
        h3, h7 = c_agg.loc[(conf, 365), "train_hrs"], c_agg.loc[(conf, 730), "train_hrs"]
        m3, m7 = c_agg.loc[(conf, 365), "gpu_mib"], c_agg.loc[(conf, 730), "gpu_mib"]
        parts.append(f"{lab}: {vn_num(x3, 2)} -> {vn_num(x7, 2)} phút/epoch ({vn_num(x7 / x3, 2)}×), "
                     f"{vn_num(h3, 2)} -> {vn_num(h7, 2)} giờ ({vn_num(h7 / h3, 2)}×), "
                     f"VRAM {vn_int(m3)} -> {vn_int(m7)} MiB ({vn_num(m7 / m3, 2)}×)")
    caption(f"đánh đổi giữa độ chính xác và chi phí tính toán trên Tesla T4 khi tăng độ dài chuỗi từ 365 lên 730 ngày. "
            f"(a) Quỹ đạo 365 -> 730 ngày theo phút huấn luyện mỗi epoch (thang log) và NSE tổ hợp test. "
            f"(b) Tổng giờ huấn luyện 30 epoch và (c) đỉnh bộ nhớ GPU VRAM (nhãn trên cột 730 ngày là bội số so với 365 ngày): "
            f"{'; '.join(parts)}.")
    return path


def draw_curves_365_vs_730(curves, configs, output, period_txt="2014-2017"):
    # Hình 5 (Động học huấn luyện 30 epoch): Validation và Test NSE qua 30 epoch cho 365 vs 730 ngày.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10})
    ncols = len(configs)
    fig, axes = plt.subplots(2, ncols, figsize=(3.9 * ncols, 6.2), sharex=True, sharey=True)
    if ncols == 1:
        axes = axes[:, None]
    y_from = 3
    late = curves.loc[curves["epoch"] >= y_from, ["val_NSE", "test_NSE"]].to_numpy(float).ravel()
    all_v = curves[["val_NSE", "test_NSE"]].to_numpy(float).ravel()
    y_lo = float(np.nanmin(late)) - 0.012
    y_hi = float(np.nanmax(all_v)) + 0.012
    n_cut = int((all_v < y_lo).sum())
    min_cut = float(np.nanmin(all_v[all_v < y_lo])) if n_cut else np.nan

    panel_letters = "abcdef"
    for j, conf in enumerate(configs):
        col = core_color(conf)
        lab = core_label(conf)
        sub_c = curves.loc[curves.configuration.eq(conf)]
        for i, (metric_col, split_name) in enumerate((("val_NSE", "Validation 2010-2013"),
                                                      ("test_NSE", f"Test {period_txt}"))):
            ax = axes[i, j]
            style_ax(ax, grid_axis="both", despine=True)
            for length, ls, lw, alpha_band, mk, mfc in (
                (365, "--", 1.6, 0.13, "o", "white"),
                (730, "-", 2.0, 0.20, "D", col),
            ):
                part = sub_c.loc[sub_c.seq_length.eq(length)]
                if part.empty:
                    continue
                piv = part.pivot(index="epoch", columns="seed", values=metric_col).sort_index()
                ep = piv.index.to_numpy(int)
                mu = piv.mean(axis=1).to_numpy(float)
                lo = piv.min(axis=1).to_numpy(float)
                hi = piv.max(axis=1).to_numpy(float)
                line_col = "0.45" if length == 365 else col
                ax.fill_between(ep, lo, hi, color=line_col, alpha=alpha_band, lw=0)
                ax.plot(ep, mu, color=line_col, ls=ls, lw=lw, zorder=3 if length == 365 else 4)
                if i == 0:
                    for _, grp in part.groupby("seed"):
                        sel = int(grp["selected_epoch"].iloc[0])
                        yv = float(grp.loc[grp["epoch"].eq(sel), metric_col].iloc[0])
                        ax.plot(sel, yv, marker=mk, ms=5.5, mfc=mfc, mec=line_col if length == 365 else "black",
                                mew=0.85, ls="", zorder=5 if length == 365 else 6)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlim(1, 30)
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
            ax.set_title(f"({panel_letters[i * ncols + j]}) {lab} — {split_name}", fontsize=10, pad=6)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median NSE (451 basins)")

    handles = [
        Line2D([], [], color="0.45", ls="--", lw=1.6, label="365 days (3-seed mean)"),
        Line2D([], [], color="C0", ls="-", lw=2.0, label="730 days (3-seed mean; color = core)"),
        Patch(facecolor="0.5", alpha=0.20, label="Min–max range across seeds"),
        Line2D([], [], color="0.45", marker="o", mfc="white", mec="0.45", ls="", ms=5.8, label="best_model.pt (365d)"),
        Line2D([], [], color="0.25", marker="D", mfc="0.25", mec="black", ls="", ms=5.8, label="best_model.pt (730d)"),
    ]
    fig.suptitle("Training Dynamics: 30-Epoch Validation and Test NSE (365 vs. 730 days)", fontsize=11.8)
    path = finish_fig(fig, output / "D_Curves_365_vs_730.png", handles=handles, ncol=3, legend_y=-0.01)

    ep_parts = []
    for conf in configs:
        lab = core_label(conf)
        sub_c = curves.loc[curves.configuration.eq(conf)]
        e3 = sorted(sub_c.loc[sub_c.seq_length.eq(365)].groupby("seed")["selected_epoch"].first().tolist())
        e7 = sorted(sub_c.loc[sub_c.seq_length.eq(730)].groupby("seed")["selected_epoch"].first().tolist())
        ep_parts.append(f"{lab}: 365d epoch {'/'.join(map(str, e3))} vs 730d epoch {'/'.join(map(str, e7))}")
    cut_txt = (f" Trục dọc bắt đầu từ {vn_num(y_lo, 2)} (chung các ô): có {n_cut} điểm ở epoch 1-{y_from - 1} nằm dưới "
               f"trục (thấp nhất {vn_num(min_cut, 3)})." if n_cut else "")
    caption(f"đường học NSE trung vị qua 30 epoch trên kỳ validation (hàng trên) và test {period_txt} (hàng dưới) "
            f"cho 365 ngày (nét đứt xám) và 730 ngày (nét liền theo màu lõi), dải mờ là khoảng 3 hạt giống, "
            f"ký hiệu tròn/thoi đánh dấu epoch chọn best_model.pt theo validation. Epoch chọn: {'; '.join(ep_parts)}."
            + cut_txt)
    return path


def draw_basin_cdf_and_thresholds(ens_basins, configs, output, period_txt="2014-2017"):
    # Hình 6 (Phân phối 451 lưu vực — chuẩn HESS/Kratzert):
    # (a) Đường phân phối tích lũy thực nghiệm ECDF tối giản, (b) Tỷ lệ/số lưu vực theo bậc chất lượng Moriasi (2007).
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

    ax = axes[0]
    style_ax(ax, grid_axis="both", despine=True)
    ax.axvline(0.5, color="0.45", ls="--", lw=0.95, zorder=1)
    ax.axvline(0.7, color="0.45", ls=":", lw=0.95, zorder=1)
    ax.axhline(0.5, color="0.65", ls="-.", lw=0.8, zorder=1)

    cdf_handles = []
    for conf in configs:
        col = core_color(conf)
        lab = core_label(conf)
        sub = ens_basins.loc[ens_basins.configuration.eq(conf)]
        for length, ls, lw, alpha in ((365, "--", 1.5, 0.78), (730, "-", 2.1, 0.98)):
            vals = np.sort(sub[f"NSE_{length}"].dropna().to_numpy(float))
            ecdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, ecdf, color=col, ls=ls, lw=lw, alpha=alpha, zorder=3 if length == 365 else 4)
            cdf_handles.append(Line2D([], [], color=col, ls=ls, lw=lw, label=f"{lab} {length}d"))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set(title=f"(a) Empirical CDF of test NSE ({period_txt})",
           xlabel="Basin test NSE (view [0, 1])", ylabel="Cumulative fraction of basins")

    ax = axes[1]
    style_ax(ax, grid_axis="x", despine=True)
    tier_defs = [
        ("NSE ≥ 0.70", lambda v: v >= 0.70, "#2b83ba"),
        ("0.50–0.70", lambda v: (v >= 0.50) & (v < 0.70), "#abdda4"),
        ("0.00–0.50", lambda v: (v >= 0.00) & (v < 0.50), "#fdae61"),
        ("NSE < 0.00", lambda v: v < 0.00, "#d7191c"),
    ]
    y_pos, y_labs = [], []
    cur = 0.0
    tier_summary = {}
    for conf in configs:
        sub = ens_basins.loc[ens_basins.configuration.eq(conf)]
        for offset, length in ((0.0, 365), (0.72, 730)):
            y = cur + offset
            y_pos.append(y)
            y_labs.append(f"{core_label(conf)} ({length}d)")
            vals = sub[f"NSE_{length}"].dropna().to_numpy(float)
            left = 0
            counts = []
            for _, cond, t_col in tier_defs:
                cnt = int(cond(vals).sum())
                counts.append(cnt)
                ax.barh(y, cnt, left=left, height=0.56, color=t_col, edgecolor="white", lw=0.7, zorder=2)
                if cnt >= 30:
                    txt_col = "white" if t_col in ("#2b83ba", "#d7191c") else "0.18"
                    ax.text(left + cnt / 2.0, y, str(cnt), ha="center", va="center",
                            fontsize=8.4, fontweight="bold", color=txt_col, zorder=3)
                left += cnt
            n_good = counts[0] + counts[1]
            tier_summary[(conf, length)] = (counts[0], n_good, counts[3], len(vals))
            ax.annotate(f"{100.0 * n_good / len(vals):.1f}%", (left, y), xytext=(6, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=8.4, fontweight="bold" if length == 730 else "normal", color="0.25")
        cur += 1.95
    ax.set_yticks(y_pos, y_labs)
    ax.set_ylim(y_pos[-1] + 0.65, y_pos[0] - 0.90)
    n_max = int(ens_basins.groupby("configuration")["basin_id"].nunique().max())
    ax.annotate("≥ 0.5", (n_max, y_pos[0] - 0.60), xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=8.2, fontweight="bold", color="0.30")
    ax.set_xlim(0, n_max * 1.18)
    ax.set(title="(b) Basin counts by skill tier (Moriasi 2007)", xlabel="Number of basins")

    handles = cdf_handles + [Patch(facecolor=c, edgecolor="white", label=n) for n, _, c in tier_defs]
    fig.suptitle(f"Spatial Skill Distribution Across {n_max} Basins (3-Seed Ensemble, Test {period_txt})", fontsize=11.8)
    path = finish_fig(fig, output / "D_Basin_CDF_And_Thresholds.png", handles=handles, ncol=5, legend_y=-0.01)

    parts = []
    for conf in configs:
        lab = core_label(conf)
        g70_3, g50_3, neg_3, n_tot = tier_summary[(conf, 365)]
        g70_7, g50_7, neg_7, _ = tier_summary[(conf, 730)]
        parts.append(f"{lab}: NSE >= 0,5 từ {g50_3} sang {g50_7}/{n_tot} (NSE >= 0,7: {g70_3} -> {g70_7}; âm: {neg_3} -> {neg_7})")
    caption(f"phân phối kỹ năng theo không gian trên {vn_int(n_max)} lưu vực (tổ hợp 3 hạt giống, kỳ test {period_txt}). "
            f"(a) Đường phân phối tích lũy thực nghiệm (ECDF) của NSE trong vùng [0, 1] (nét đứt 365 ngày, nét liền 730 ngày; "
            f"vạch dọc đứt/chấm là mốc NSE = 0,5 và 0,7; vạch ngang là trung vị 50% lưu vực). "
            f"(b) Số lưu vực theo 4 bậc chất lượng Moriasi và cs. (2007) kèm tỷ lệ đạt NSE >= 0,5 ở cột phải: {'; '.join(parts)}.")
    return path


def draw_basin_paired_delta(ens_basins, configs, output, b0_noise=None, period_txt="2014-2017"):
    # Hình 7 (Nâng cao — Ghép cặp từng lưu vực, phong cách Raincloud/HESS không dùng hộp chữ lớn):
    # Hàng trên: scatter 1:1 gọn nhẹ (tỷ lệ cải thiện ghi trên tiêu đề ô);
    # Hàng dưới: mật độ phân phối ghép cặp ΔNSE kèm thanh hộp IQR + râu P5–P95 ngay bên dưới.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10})
    ncols = len(configs)
    fig, axes = plt.subplots(2, ncols, figsize=(4.0 * ncols, 6.9))
    if ncols == 1:
        axes = axes[:, None]
    gain_col, loss_col = "#1b7837", "#c51b7d"
    d_clip = 0.08
    grid = np.linspace(-d_clip, d_clip, 300)
    bins = np.linspace(-d_clip, d_clip, 33)
    letters = "abcdef"
    summary_parts = []
    # Tính trước mật độ cho mọi lõi để dùng chung trục tung ở hàng dưới, dễ so độ nhọn giữa các lõi.
    cached = []
    max_peak = 1.0
    for conf in configs:
        sub = ens_basins.loc[ens_basins.configuration.eq(conf)].dropna(subset=["NSE_365", "NSE_730"])
        n3 = sub["NSE_365"].to_numpy(float)
        n7 = sub["NSE_730"].to_numpy(float)
        dn = n7 - n3
        dn_in = dn[(dn >= -d_clip) & (dn <= d_clip)]
        hist_vals, hist_edges = np.histogram(dn_in, bins=bins, density=True)
        kde_y = kde_1d(dn_in, grid)
        max_peak = max(max_peak, float(hist_vals.max()), float(kde_y.max()))
        cached.append((conf, n3, n7, dn, hist_vals, hist_edges, kde_y))

    for j, (conf, n3, n7, dn, hist_vals, hist_edges, kde_y) in enumerate(cached):
        col = core_color(conf)
        lab = core_label(conf)
        n_tot = len(dn)
        n_imp = int((dn > 0).sum())
        pct_imp = 100.0 * n_imp / n_tot
        n_g2 = int((dn > 0.02).sum())
        n_l2 = int((dn < -0.02).sum())
        med_dn = float(np.median(dn))
        q05, q25, q75, q95 = [float(v) for v in np.percentile(dn, [5, 25, 75, 95])]
        summary_parts.append(f"{lab}: {n_imp}/{n_tot} ({vn_num(pct_imp, 1)}%) lưu vực tăng, "
                             f"trung vị ΔNSE {vn_num(med_dn, 4, signed=True)} (IQR [{vn_num(q25, 4, signed=True)}, "
                             f"{vn_num(q75, 4, signed=True)}]), |ΔNSE| > 0,02 có {n_g2} tăng / {n_l2} giảm")

        # Hàng trên: 1:1 scatter sạch trên [0, 1]
        ax_top = axes[0, j]
        style_ax(ax_top, grid_axis="both", despine=False)
        in_box = (n3 >= 0.0) & (n7 >= 0.0)
        ax_top.fill_between([0.0, 1.0], [-0.02, 0.98], [0.02, 1.02], color="0.93", zorder=0)
        ax_top.plot([0.0, 1.0], [0.0, 1.0], color="0.25", ls="--", lw=0.95, zorder=1)
        pos = in_box & (dn > 0)
        neg = in_box & (dn <= 0)
        ax_top.scatter(n3[neg], n7[neg], s=13, alpha=0.55, color=loss_col, edgecolors="none", zorder=2)
        ax_top.scatter(n3[pos], n7[pos], s=13, alpha=0.60, color=gain_col, edgecolors="none", zorder=3)
        ax_top.set_xlim(0.0, 1.0)
        ax_top.set_ylim(0.0, 1.0)
        ax_top.set_aspect("equal", adjustable="box")
        ax_top.set_title(f"({letters[j]}) {lab} ({pct_imp:.1f}% improved)", fontsize=10.0, pad=6)
        ax_top.set_xlabel("365-day ensemble NSE")
        if j == 0:
            ax_top.set_ylabel("730-day ensemble NSE")

        # Hàng dưới: Raincloud-style mật độ + hộp IQR/P5-P95 gọn ngay phía dưới đường 0 (chung trục tung)
        ax_bot = axes[1, j]
        style_ax(ax_bot, grid_axis="x", despine=True)
        if b0_noise is not None and np.isfinite(b0_noise):
            ax_bot.axvspan(-b0_noise, b0_noise, color="0.92", zorder=0)
        ax_bot.axvline(0.0, color="0.30", ls="--", lw=0.95, zorder=1)
        bin_w = np.diff(hist_edges)
        bin_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        bar_colors = [gain_col if c > 0 else loss_col for c in bin_centers]
        ax_bot.bar(hist_edges[:-1], hist_vals, width=bin_w, align="edge", color=bar_colors,
                   alpha=0.38, edgecolor="white", lw=0.4, zorder=2)
        ax_bot.plot(grid, kde_y, color=col, lw=2.0, zorder=4)
        ax_bot.axvline(med_dn, color=col, ls="-", lw=1.6, zorder=5)
        box_y = -0.14 * max_peak
        box_h = 0.08 * max_peak
        ax_bot.axhline(0.0, color="0.45", lw=0.7, zorder=2)
        ax_bot.plot([max(q05, -d_clip), min(q95, d_clip)], [box_y, box_y], color=col, lw=1.6, zorder=4)
        ax_bot.add_patch(plt.Rectangle((max(q25, -d_clip), box_y - box_h / 2),
                                       min(q75, d_clip) - max(q25, -d_clip), box_h,
                                       facecolor=col, alpha=0.35, edgecolor=col, lw=1.2, zorder=5))
        ax_bot.plot(med_dn, box_y, marker="D", ms=5.0, mfc=col, mec="black", mew=0.5, zorder=6)
        ax_bot.set_ylim(-0.26 * max_peak, 1.12 * max_peak)
        ax_bot.set_yticks([t for t in ax_bot.get_yticks() if t >= 0])
        ax_bot.set_xlim(-d_clip, d_clip)
        ax_bot.set_title(f"({letters[ncols + j]}) {lab} (med Δ = {med_dn:+.4f})", fontsize=10.0, pad=6)
        ax_bot.set_xlabel("ΔNSE = NSE(730d) − NSE(365d)")
        if j == 0:
            ax_bot.set_ylabel("Density  |  IQR & P5–P95")

    handles = [
        Line2D([], [], color=gain_col, marker="o", ls="", ms=5.8, label="730d improves (ΔNSE > 0)"),
        Line2D([], [], color=loss_col, marker="o", ls="", ms=5.8, label="730d degrades (ΔNSE ≤ 0)"),
        Patch(facecolor="0.92", label="±0.02 band (top) / ±B0 noise (bottom)" if b0_noise else "±0.02 band (top)"),
        Line2D([], [], color="0.25", lw=1.8, marker="D", ms=5, label="KDE, median line & IQR/P5–P95 strip"),
    ]
    fig.suptitle(f"Paired Basin-Level Diagnostics Across 451 Basins (3-Seed Ensemble, Test {period_txt})", fontsize=11.8)
    path = finish_fig(fig, output / "D_Basin_Paired_Delta.png", handles=handles, ncol=2, legend_y=-0.01)

    caption(f"chẩn đoán ghép cặp từng lưu vực giữa 730 và 365 ngày trên tổ hợp 3 hạt giống (kỳ test {period_txt}). "
            f"Hàng trên (a-c): đồ thị tán xạ 1:1 trong vùng [0, 1] (xanh là 730 cải thiện, hồng đậm là giảm, dải xám là ±0,02). "
            f"Hàng dưới (d-f): phân phối hiệu ghép cặp ΔNSE trong [-0,08, +0,08] kèm mật độ KDE, vạch trung vị và thanh hộp "
            f"IQR (Q25-Q75) cùng râu P5-P95 phía dưới"
            + (f" (dải xám dọc là ± nhiễu B0 {vn_num(b0_noise)})" if b0_noise is not None else "")
            + f". Thống kê 451 lưu vực: {'; '.join(summary_parts)}.")
    return path


def draw_basin_stratification(ens_basins, configs, output, b0_noise=None, period_txt="2014-2017"):
    # Hình 8 (Nâng cao — Phân tầng theo đặc trưng lưu vực, chuẩn sạch không rải số trong vùng vẽ):
    # Trung vị ΔNSE kèm khoảng tin cậy bootstrap 95% theo tứ phân vị diện tích và kỹ năng nền 365 ngày.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    area_ref = ens_basins.drop_duplicates("basin_id").set_index("basin_id")["area_gov_km2"]
    area_edges = np.quantile(area_ref.to_numpy(float), [0.0, 0.25, 0.50, 0.75, 1.0])
    area_labels = [
        f"Q1\n(<{area_edges[1]:.0f} km²)",
        f"Q2\n({area_edges[1]:.0f}–{area_edges[2]:.0f})",
        f"Q3\n({area_edges[2]:.0f}–{area_edges[3]:.0f})",
        f"Q4\n(>{area_edges[3]:.0f} km²)",
    ]
    skill_labels = ["Q1\n(lowest 25%)", "Q2\n(25–50%)", "Q3\n(50–75%)", "Q4\n(highest 25%)"]
    offsets = np.linspace(-0.20, 0.20, len(configs)) if len(configs) > 1 else np.array([0.0])

    cap_area, cap_skill = [], []
    y_all = [0.0]
    if b0_noise is not None and np.isfinite(b0_noise):
        y_all.extend([-b0_noise, b0_noise])

    for ax, mode, xlabels, title in [
        (axes[0], "area", area_labels, "(a) By catchment area quartile (area_gov)"),
        (axes[1], "skill", skill_labels, "(b) By 365-day baseline NSE quartile"),
    ]:
        style_ax(ax, grid_axis="y", despine=True)
        if b0_noise is not None and np.isfinite(b0_noise):
            ax.axhspan(-b0_noise, b0_noise, color="0.92", zorder=0)
        ax.axhline(0.0, color="0.30", ls="--", lw=0.95, zorder=1)
        for ci, conf in enumerate(configs):
            col = core_color(conf)
            lab = core_label(conf)
            sub = ens_basins.loc[ens_basins.configuration.eq(conf)].dropna(subset=["NSE_365", "NSE_730"]).copy()
            key_col = "area_gov_km2" if mode == "area" else "NSE_365"
            sub["qbin"] = pd.qcut(sub[key_col], 4, labels=False, duplicates="drop")
            meds, lo_err, hi_err, xs = [], [], [], []
            for q in range(4):
                vals = (sub.loc[sub["qbin"].eq(q), "delta_NSE"]).to_numpy(float)
                med, lo, hi = bootstrap_median_ci(vals, n_boot=2000, seed=42 + 17 * ci + q)
                xs.append(q + offsets[ci])
                meds.append(med)
                lo_err.append(med - lo)
                hi_err.append(hi - med)
                y_all.extend([lo, hi])
            ax.plot(xs, meds, color=col, lw=1.3, alpha=0.60, zorder=3)
            ax.errorbar(xs, meds, yerr=[lo_err, hi_err], fmt="o", color=col, mec="black", mew=0.5,
                        capsize=3.0, ms=6.6, lw=1.5, zorder=4)
            if mode == "area":
                cap_area.append(f"{lab} Q1 {vn_num(meds[0], 4, signed=True)} -> Q4 {vn_num(meds[3], 4, signed=True)}")
            else:
                cap_skill.append(f"{lab} Q1 {vn_num(meds[0], 4, signed=True)} -> Q4 {vn_num(meds[3], 4, signed=True)}")
        ax.set_xticks(range(4), xlabels)
        ax.set_xlim(-0.55, 3.55)
        ax.set_title(title, fontsize=10.5, pad=8)
    span = max(y_all) - min(y_all)
    axes[0].set_ylim(min(y_all) - 0.12 * span, max(y_all) + 0.14 * span)
    axes[0].set_ylabel("Median paired ΔNSE (95% bootstrap CI)")

    handles = [Line2D([], [], color=core_color(c), marker="o", mec="black", mew=0.5, lw=1.5, ms=6.8,
                      label=f"{core_label(c)} (median ± 95% CI)") for c in configs]
    if b0_noise is not None and np.isfinite(b0_noise):
        handles.append(Rectangle((0, 0), 1, 1, color="0.92", label=f"± B0 seed noise (±{b0_noise:.4f})"))
    fig.suptitle(f"Stratified Paired ΔNSE by Catchment Area and Baseline Skill ({period_txt})", fontsize=11.6)
    path = finish_fig(fig, output / "D_Basin_Stratification.png", handles=handles, ncol=4, legend_y=-0.01)

    caption(f"phân tầng hiệu ghép cặp ΔNSE (730 trừ 365 ngày, tổ hợp 3 hạt giống, trung vị và khoảng tin cậy bootstrap 95% "
            f"với 2.000 lượt) trên kỳ test {period_txt}. "
            f"(a) Theo 4 tứ phân vị diện tích lưu vực area_gov ({'; '.join(cap_area)}). "
            f"(b) Theo 4 tứ phân vị kỹ năng nền NSE 365 ngày ({'; '.join(cap_skill)}): chênh lệch giữa hai độ dài chuỗi "
            f"tập trung ở nhóm lưu vực khó nhất (Q1), còn nhóm vốn dự báo tốt (Q3-Q4) gần như không đổi.")
    return path


def draw_hydrograph_diagnostics(ens_basins, configs, output, period_txt="2014-2017"):
    # Hình 9 (Nâng cao — Đồ thị quả tạ có hướng tối giản cho thành phần KGE và thiên lệch theo chế độ dòng chảy):
    # Không rải số dày trên từng mũi tên; để hướng mũi tên và mốc lý tưởng truyền tải ngay cơ chế vật lý.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    offsets = np.linspace(-0.22, 0.22, len(configs)) if len(configs) > 1 else np.array([0.0])

    ax = axes[0]
    style_ax(ax, grid_axis="y", despine=True)
    kge_specs = [
        ("r", "Correlation r\n(timing)"),
        ("alpha", "Variability α\n(σ_sim / σ_obs)"),
        ("beta", "Volume bias β\n(μ_sim / μ_obs)"),
    ]
    ax.axhline(1.0, color="0.25", ls="--", lw=0.95, zorder=1)
    vals_a = [1.0]
    kge_cap = []
    for ci, conf in enumerate(configs):
        col = core_color(conf)
        lab = core_label(conf)
        sub = ens_basins.loc[ens_basins.configuration.eq(conf)]
        comp_txt = []
        for gi, (key, _) in enumerate(kge_specs):
            v3 = float(np.nanmedian(sub[f"{key}_365"]))
            v7 = float(np.nanmedian(sub[f"{key}_730"]))
            vals_a.extend([v3, v7])
            xp = gi + offsets[ci]
            if abs(v7 - v3) >= 0.008:
                ax.annotate("", xy=(xp, v7), xytext=(xp, v3),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, shrinkA=4, shrinkB=4), zorder=3)
            else:
                ax.plot([xp, xp], [v3, v7], color=col, lw=1.8, zorder=3)
            ax.plot(xp, v3, marker="o", ms=7.4, mfc="white", mec=col, mew=1.8, ls="", zorder=4)
            ax.plot(xp, v7, marker="D", ms=7.0, mfc=col, mec="black", mew=0.55, ls="", zorder=5)
            comp_txt.append(f"{key} {vn_num(v3, 3)}->{vn_num(v7, 3)}")
        kge_cap.append(f"{lab} ({', '.join(comp_txt)})")
    span_a = max(vals_a) - min(vals_a)
    ax.set_ylim(min(vals_a) - 0.14 * span_a, max(vals_a) + 0.14 * span_a)
    ax.set_xticks(range(len(kge_specs)), [t for _, t in kge_specs])
    ax.set_xlim(-0.55, len(kge_specs) - 0.45)
    ax.set(title="(a) KGE components (ideal = 1.0)", ylabel="Median across 451 basins")

    ax = axes[1]
    style_ax(ax, grid_axis="y", despine=True)
    bias_specs = [
        ("FHV", "Top 2% FDC\n(%BiasFHV)"),
        ("hi_bias", "Peak days\n(obs ≥ Q98)"),
        ("lo_bias", "Low-flow days\n(obs ≤ Q30)"),
    ]
    ax.axhline(0.0, color="0.25", ls="--", lw=0.95, zorder=1)
    vals_b = [0.0]
    bias_cap = []
    for ci, conf in enumerate(configs):
        col = core_color(conf)
        lab = core_label(conf)
        sub = ens_basins.loc[ens_basins.configuration.eq(conf)]
        b_txt = []
        for gi, (key, _) in enumerate(bias_specs):
            v3 = float(np.nanmedian(sub[f"{key}_365"]))
            v7 = float(np.nanmedian(sub[f"{key}_730"]))
            vals_b.extend([v3, v7])
            xp = gi + offsets[ci]
            if abs(v7 - v3) >= 1.5:
                ax.annotate("", xy=(xp, v7), xytext=(xp, v3),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, shrinkA=4, shrinkB=4), zorder=3)
            else:
                ax.plot([xp, xp], [v3, v7], color=col, lw=1.8, zorder=3)
            ax.plot(xp, v3, marker="o", ms=7.4, mfc="white", mec=col, mew=1.8, ls="", zorder=4)
            ax.plot(xp, v7, marker="D", ms=7.0, mfc=col, mec="black", mew=0.55, ls="", zorder=5)
            b_txt.append(f"{key} {vn_num(v3, 1)}%->{vn_num(v7, 1)}%")
        bias_cap.append(f"{lab} ({', '.join(b_txt)})")
    span_b = max(vals_b) - min(vals_b)
    ax.set_ylim(min(vals_b) - 0.14 * span_b, max(vals_b) + 0.14 * span_b)
    ax.set_xticks(range(len(bias_specs)), [t for _, t in bias_specs])
    ax.set_xlim(-0.55, len(bias_specs) - 0.45)
    ax.set(title="(b) Flow-regime percent bias (ideal = 0%)", ylabel="Median percent bias (%)")

    handles = [
        Line2D([], [], color="0.25", marker="o", mfc="white", mec="0.25", mew=1.8, ls="", ms=7.2, label="365 days"),
        Line2D([], [], color="0.25", marker="D", mfc="0.25", mec="black", mew=0.55, ls="", ms=6.8,
               label="730 days (arrow = 365 -> 730)"),
    ] + [Line2D([], [], color=core_color(c), lw=2.2, label=core_label(c)) for c in configs]
    fig.suptitle(f"Hydrograph Diagnostics: KGE Components and Flow-Regime Biases ({period_txt})", fontsize=11.6)
    path = finish_fig(fig, output / "D_Hydrograph_Diagnostics.png", handles=handles, ncol=5, legend_y=-0.01)

    caption(f"chẩn đoán vật lý đường quá trình dòng chảy (tổ hợp 3 hạt giống, trung vị qua 451 lưu vực, test {period_txt}); "
            f"mũi tên chỉ dịch chuyển từ 365 ngày (tròn rỗng) sang 730 ngày (thoi đặc). "
            f"(a) Ba thành phần KGE (tương quan r, tỷ số độ biến thiên α và tỷ số lượng β, đường đứt là mốc lý tưởng 1,0): "
            f"{'; '.join(kge_cap)}. "
            f"(b) Sai số thiên lệch theo chế độ dòng chảy (%BiasFHV top 2%, ngày đỉnh >= Q98 và ngày kiệt <= Q30, đường đứt là 0%): "
            f"{'; '.join(bias_cap)}. Biểu đồ làm rõ cơ chế biến đổi KGE và thiên lệch đỉnh/kiệt của từng kiến trúc lõi khi nhân đôi độ dài chuỗi.")
    return path


def draw_all_d(seeds, ensemble, ens_basins, cost, ratios, configs, output,
               bstar=None, b0_noise=None, curves=None, dates=None, common_basins=None):
    period_txt = (f"{dates[0].year}-{dates[-1].year}" if dates is not None and len(dates)
                  else f"{TEST_START[:4]}-{TEST_END[:4]}")
    saved = [
        draw_overview_accuracy(seeds, ensemble, configs, output, bstar=bstar, b0_noise=b0_noise,
                               period_txt=period_txt, common_basins=common_basins),
        draw_d(seeds, ratios, configs, output, dates=dates, common_basins=common_basins,
               b0_noise=b0_noise, ensemble=ensemble),
        draw_multimetric_ensemble(seeds, ensemble, configs, output, period_txt=period_txt,
                                  common_basins=common_basins),
        draw_cost_tradeoff(ensemble, cost, ratios, configs, output, period_txt=period_txt),
    ]
    if curves is not None and not curves.empty:
        saved.append(draw_curves_365_vs_730(curves, configs, output, period_txt=period_txt))
    saved.extend([
        draw_basin_cdf_and_thresholds(ens_basins, configs, output, period_txt=period_txt),
        draw_basin_paired_delta(ens_basins, configs, output, b0_noise=b0_noise, period_txt=period_txt),
        draw_basin_stratification(ens_basins, configs, output, b0_noise=b0_noise, period_txt=period_txt),
        draw_hydrograph_diagnostics(ens_basins, configs, output, period_txt=period_txt),
    ])
    return saved


if __name__ == "__main__":
    compare(Path("/kaggle/input"), Path("/kaggle/working/d_compare"))
