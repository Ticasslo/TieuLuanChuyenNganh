# %% [markdown]
# # C2 — Suy luận các kịch bản vận hành
#
# Kaggle T4, Internet On. Gắn lamah-ce-ext, 02_A3_LamaHCEExt_Benchmarks V1,
# 13_C2_LamaHCEExt_Compare V1, 01_0_MambaSSM_Build V1 và 24 Output C2 train V1
# theo cột notebook trong Tables/C2_handoff.csv. Với mức x3 dùng Output _Cont.
# 1. Kiểm manifest/checkpoint và lịch. 2. Suy luận S0, S1_Q1/Q3/Q7, S2/S3/S4.
# 3. Chấm giao chung và bàn giao NPZ. Giữ checkpoint đã chọn theo validation.

# %% Phần 1 - Khóa nguồn C2 và kiểm thư viện
import copy
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from importlib import metadata
from io import StringIO
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

NOTEBOOK = "04_C2_LamaHCEExt_Scenarios"
ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working/c2_scenarios")
SCENARIOS = ("S0", "S1_Q1", "S1_Q3", "S1_Q7", "S2", "S3", "S4")
PRED_RTOL, PRED_ATOL = 1e-6, 2e-5
TEST_START, TEST_END = "2014-01-01", "2017-12-31"

# Các hàm kiểm dùng cùng mã nguồn với Evaluation và Demo D1.


METRICS = ("NSE", "PNSE", "KGE", "r", "alpha", "beta")

FREEZE_C = "db6dc1ffcd8c14994cd80be3ed8ff417002c01a1"

C2_NOTEBOOK = "13_C2_LamaHCEExt_Compare"

C2_CONFIGS = ("lstm_L2_x1_lr1e-3", "gru_L2_x1_lr1e-3", "transformer_L2_x1_lr1e-3",
              "transformer_L2_x3_lr5e-4", "s4d_L4_x1_lr1e-3", "s4d_L4_x3_lr1e-3",
              "mamba_L2_x1_lr1e-3", "mamba_L4_x3_lr1e-3")

C2_SEEDS = (111, 222, 333)

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

COMMIT = "db6dc1ffcd8c14994cd80be3ed8ff417002c01a1"

CONF, SEED, EPOCH = "gru_L2_x1_lr1e-3", 111, 20

DATES = pd.date_range("2014-01-01", "2017-12-31")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def checked_path(root, name):
    rel = PurePosixPath(name)
    root = root.resolve()
    path = root.joinpath(*rel.parts).resolve()
    if (rel.is_absolute() or ".." in rel.parts or rel.as_posix() != name
            or "\\" in name or ":" in name or not path.is_relative_to(root) or not path.is_file()):
        raise ValueError(f"Đường dẫn Input sai: {name}")
    return path

def checked_manifest(root, manifest, names):
    table = pd.read_csv(manifest)
    column = "file" if "file" in table else "tệp"
    if table[column].duplicated().any():
        raise ValueError(f"Manifest trùng tên: {manifest}")
    table = table.set_index(column)
    checked = {}
    for name in names:
        path = checked_path(root, name)
        row = table.loc[name]
        if row["sha256"] != sha256(path) or ("bytes" in row and row["bytes"] != path.stat().st_size):
            raise ValueError(f"SHA-256 hoặc dung lượng không khớp: {name}")
        checked[name] = path
    return checked

def unique(paths, label):
    paths = sorted(set(paths))
    if len(paths) != 1:
        raise ValueError(f"Cần đúng một nguồn {label}, thấy {len(paths)}.")
    return paths[0]

def install_library():
    missing = []
    for package in ("ruamel.yaml", "tensorboard"):
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])

    def installed():
        try:
            return json.loads(metadata.distribution("neuralhydrology").read_text("direct_url.json"))["vcs_info"]["commit_id"]
        except (metadata.PackageNotFoundError, KeyError, TypeError):
            return None

    if installed() != COMMIT:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
                               f"git+https://github.com/Ticasslo/neuralhydrology.git@{COMMIT}"])
    if installed() != COMMIT:
        raise RuntimeError("Thư viện đang cài khác freeze-C.")

def physical_prediction(values, scale, center):
    return values * scale + center

def qmax_m3_s(mm_day, area_gov):
    return mm_day * area_gov / 86.4

class DemoForecaster:
    def __init__(self, run, raw_dir, ids, areas, scratch, device="cuda:0", *,
                 core="gru", depth=2, seed=SEED, n_params=84503):
        import torch
        from neuralhydrology.datautils.utils import load_scaler
        from neuralhydrology.datasetzoo.lamah import LamaH, load_lamah_attributes, load_lamah_discharge, _normalize_discharge
        from neuralhydrology.modelzoo import get_model
        from neuralhydrology.utils.config import Config

        class DemoLamaH(LamaH):
            def _load_basin_data(self, basin):
                frame = super()._load_basin_data(basin)
                # C2 ghi Q mm/ngày qua CSV 17 chữ số. Căn ngày trước khi tái tạo đường đọc.
                for base in ("qmean", "qmax"):
                    values = load_lamah_discharge(self.cfg.data_dir, basin, temporal_resolution="1D",
                                                  normalize_discharge=False, target_name=base)[base]
                    values = _normalize_discharge(ser=values, area=self._basin_area.loc[basin],
                                                  temporal_resolution="1D").reindex(frame.index)
                    frame[f"{base}_mmd"] = pd.read_csv(StringIO(values.to_csv(index=False, header=["q"],
                                                                              float_format="%.17g")))["q"].to_numpy()
                return frame

        self.torch, self.dataset_type = torch, DemoLamaH
        self.device = torch.device(device)
        self.areas = dict(zip(ids, areas))
        areas_raw = load_lamah_attributes(raw_dir, sub_dataset="lamah_a").loc[list(ids), "area_gov"]
        if not np.array_equal(areas_raw.to_numpy(), areas):
            raise ValueError("Diện tích lưu vực của dữ liệu demo khác A3.")
        cfg = Config(run / "config.yml")
        if (cfg.model != "sequential_forecast" or cfg.sf_core != core or cfg.sf_layers != depth
                or cfg.seed != seed or cfg.seq_length != 365 or cfg.forecast_seq_length != 1
                or cfg.predict_last_n != 1 or cfg.forecast_overlap not in (0, None)
                or cfg.target_variables != ["qmax"] or "qmax" not in cfg.clip_targets_to_zero
                or cfg.additional_feature_files or cfg.use_basin_id_encoding):
            raise ValueError(f"{run.name}: config suy luận khác mô hình C2 đã chọn.")
        basins = scratch / "D1_basins.txt"
        basins.write_text("\n".join(ids) + "\n", encoding="utf-8")
        cfg.update_config(dict(data_dir=Path(raw_dir), train_basin_file=basins,
                               validation_basin_file=basins, test_basin_file=basins, device=device))
        self.cfg = cfg
        self.scaler = load_scaler(run)
        self.scale = self.scaler["xarray_feature_scale"][["qmax"]].to_array().values
        self.center = self.scaler["xarray_feature_center"][["qmax"]].to_array().values
        self.model = get_model(self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(run / "best_model.pt", map_location=self.device, weights_only=True), strict=True)
        if sum(p.numel() for p in self.model.parameters() if p.requires_grad) != n_params:
            raise ValueError("Số tham số không khớp C2.")
        self.model.eval()
        self.cache = OrderedDict()

    def _dataset(self, basin):
        basin = str(basin)
        if basin not in self.areas:
            raise ValueError(f"Lưu vực ngoài 451 ID: {basin}")
        if basin not in self.cache:
            dataset = self.dataset_type(self.cfg, is_train=False, period="test", basin=basin,
                                        scaler=copy.deepcopy(self.scaler))
            if dataset.is_train or dataset._compute_scaler:
                raise ValueError("Demo phải dùng scaler train đã lưu.")
            self.cache[basin] = dataset
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(basin)
        return self.cache[basin]

    def _convert(self, output):
        return physical_prediction(output, self.scale, self.center).reshape(-1)

    def _forward(self, batch, scenario="S0"):
        dates = np.asarray(batch["date"]).astype("datetime64[D]")
        if (dates.ndim != 2 or dates.shape[1] != 365 or not np.all(np.diff(dates, axis=1) == np.timedelta64(1, "D"))
                or any(v.shape[1] != 364 for v in batch["x_d_hindcast"].values())
                or any(v.shape[1] != 1 for v in batch["x_d_forecast"].values())):
            raise ValueError("Cửa sổ demo sai, cần quá khứ kết thúc t-1 và một ngày ECMWF.")
        batch = mask_inputs(batch, scenario)
        with self.torch.inference_mode():
            pred = self.model(self.model.pre_model_hook(batch, is_train=False))["y_hat"][:, -1:, :].cpu().numpy()
        return dates[:, -1], self._convert(pred), self._convert(batch["y"][:, -1:, :].cpu().numpy())

    def predict_basin(self, basin, scenario="S0"):
        from neuralhydrology.datautils.gpu_loader import GpuBatchLoader

        loader = GpuBatchLoader(self._dataset(basin), batch_size=self.cfg.batch_size,
                                device=self.device, shuffle=False, with_dates=True)
        return tuple(np.concatenate(values) for values in zip(*(self._forward(batch, scenario) for batch in loader)))

    def predict(self, basin, target_date):
        day = pd.Timestamp(target_date)
        if day not in DATES:
            raise ValueError("Ngày đích phải thuộc kỳ test 2014–2017, không kèm giờ.")
        basin = str(basin)
        dataset = self._dataset(basin)
        index = int(DATES.get_loc(day))
        batch = dataset.collate_fn([dataset[index]])
        for name, value in batch.items():
            if self.torch.is_tensor(value):
                batch[name] = value.to(self.device)
            elif isinstance(value, dict):
                batch[name] = {k: v.to(self.device) for k, v in value.items()}
        dates, raw, _ = self._forward(batch)
        if dates[0] != day.to_datetime64() or not np.isfinite(raw[0]):
            raise ValueError("Không nhận đúng ngày đích hoặc dự báo không hữu hạn.")
        clipped = float(max(raw[0], 0))
        return dict(basin_id=basin, target_date=str(day.date()), origin_date=str(day.date()), lead=1,
                    qmax_raw_mm_day=float(raw[0]), qmax_mm_day=clipped,
                    qmax_m3_s=float(qmax_m3_s(clipped, self.areas[basin])))

def compare_arrays(actual, reference, atol, rtol):
    if not np.array_equal(np.isfinite(actual), np.isfinite(reference)):
        raise ValueError("Mặt nạ hữu hạn khác dự báo đã lưu.")
    if np.isinf(actual).any() or not np.allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=True):
        raise ValueError(f"Dự báo khác nguồn, max abs={np.nanmax(np.abs(actual-reference)):.8g}")
    diff = np.abs(actual - reference)
    return float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
def install_mamba(input_root):
    path = unique_file(input_root, "build_info.json")
    info = json.loads(path.read_text(encoding="utf-8-sig"))
    import torch

    actual = dict(python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda)
    if any(info[key] != version for key,version in actual.items()) or torch.cuda.get_device_capability(0) != (7,5):
        raise ValueError("Wheel Mamba phải khớp Python, torch và CUDA của T4.")
    wheels = []
    for name, digest in info["wheels"].items():
        wheel = contained_file(path.parent, name)
        if sha256(wheel) != digest:
            raise ValueError(f"Wheel Mamba sai hash {name}")
        wheels.append(str(wheel))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", *wheels])


def source_run(root, record):
    env_path = unique([p for p in root.rglob("environment.json")
                       if p.parent.name == "c_output" and sha256(p) == record["source"]["environment_sha256"]],
                      record["train_notebook"])
    env = read_json(env_path)
    handoff = record["handoff"]
    if (env["run_dir"] != handoff["thư mục run"] or env["notebook"] != record["train_notebook"]
            or env["neuralhydrology_commit"] != FREEZE_C or env["config_sha256"] != handoff["sha256 config.yml"]
            or env["summary"]["status"] != "xong" or env["summary"]["saved_epoch"] != record["epoch"]
            or env["summary"]["n_params"] != record["cfg"]["n_params"]):
        raise ValueError(f"Output train khác bảng bàn giao: {record['train_notebook']}")
    names = ("config.yml", "best_model.pt", "train_data/train_data_scaler.yml")
    paths = {name:f"{env['run_dir']}/{name}" for name in names}
    checked = checked_manifest(env_path.parent.parent, env_path.parent/"manifest.csv", paths.values())
    if any(sha256(checked[rel]) != handoff[f"sha256 {PurePosixPath(name).name}"] for name,rel in paths.items()):
        raise ValueError(f"Hash checkpoint/config/scaler sai: {record['train_notebook']}")
    return checked[paths["best_model.pt"]].parent, env_path


# %% Phần 2 - Che đầu vào sau chuẩn hóa và suy luận checkpoint đã lưu
def mask_inputs(batch, scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f"Sai kịch bản {scenario}.")
    if scenario == "S0":
        return batch
    hidden = dict(batch["x_d_hindcast"])
    for name, values in hidden.items():
        n = 0
        if name in ("qmean_mmd", "qmax_mmd"):
            if scenario.startswith("S1_Q"):
                n = int(scenario[-1])
            elif scenario == "S2":
                n = values.shape[1]
            elif scenario == "S4":
                n = 1
        if scenario in ("S3", "S4"):
            if name.startswith("ERA5L_"):
                n = 5
            elif name.startswith("GLEAM_"):
                n = values.shape[1]
        if n:
            masked = values.clone()
            masked[:, -n:, :] = float("nan")
            hidden[name] = masked
    return dict(batch, x_d_hindcast=hidden)


def infer_basin(predictor, basin):
    from neuralhydrology.datautils.gpu_loader import GpuBatchLoader

    dataset = predictor._dataset(basin)
    loader = GpuBatchLoader(dataset, batch_size=predictor.cfg.batch_size, device=predictor.device,
                            shuffle=False, with_dates=True)
    chunks = {s:[] for s in SCENARIOS}
    dates, observations = [], []
    for batch in loader:
        for scenario in SCENARIOS:
            d, pred, obs = predictor._forward(batch, scenario)
            chunks[scenario].append(pred)
            if scenario == "S0":
                dates.append(d)
                observations.append(obs)
    return np.concatenate(dates), np.concatenate(observations), {s:np.concatenate(v) for s,v in chunks.items()}


def write_scenario(ref, scenario, raw, output):
    days = ref.dates.to_numpy(dtype="datetime64[D]")
    meta = dict(ref.metadata, scenario=scenario, inference_library_commit=FREEZE_C,
                S0_source_contract_sha256=ref.metadata["source_file_sha256"],
                checkpoint_bytes_verified=True, training_rerun=False)
    path = output / f"{ref.name}_test_{scenario}_lead1.npz"
    np.savez_compressed(path, dates=days, origin_dates=days, persistence_source_dates=days-np.timedelta64(1,"D"),
                        basin_ids=np.array(ref.ids), area_gov_km2=ref.area, obs_mm_day=ref.obs,
                        persistence_mm_day=ref.persistence, sim_raw_mm_day=raw, sim_mm_day=np.maximum(raw,0),
                        metadata_json=json.dumps(meta,ensure_ascii=False))
    forecast = read_contract(path)
    check_same_context(ref, forecast)
    forecast.obs, forecast.persistence, forecast.area = ref.obs, ref.persistence, ref.area
    return path, forecast


# %% Phần 3 - Chấm giao chung, mức suy giảm và bàn giao
def run_scenarios():
    from time import perf_counter

    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    summary_file = OUT/"c2_scenarios_summary.json"
    summary_file.write_text('{"complete": false}',encoding="utf-8")
    a3_root = unique_file(ROOT,"A3_persistence_predictions.npz").parent
    context, a3 = read_context(a3_root)
    if len(context["ids"]) != 451:
        raise ValueError("C2 cần đúng 451 ID.")
    compare_root = unique_file(ROOT,"c2_summary.json").parent
    records, _, _, _ = c2_records(compare_root,a3)
    train_sources = [source_run(ROOT,r) for r in records]
    raw = unique([p.parents[2] for p in ROOT.rglob("Catchment_attributes.csv")
                  if p.parent.name=="1_attributes" and p.parent.parent.name=="A_basins_total_upstrm"
                  and (p.parents[2]/"D_gauges/1_attributes/Gauge_attributes.csv").is_file()], "lamah-ce-ext")
    out_contracts = OUT/"contracts"
    out_contracts.mkdir(exist_ok=True)
    forecasts, audit_rows, source_rows, exports = [], [], [], []
    shape = context["obs"].shape
    for index, (record, (run, env_path)) in enumerate(zip(records, train_sources), start=1):
        started = perf_counter()
        conf, seed, cfg = record["conf"],record["seed"],record["cfg"]
        ref,_ = c2_forecast(record,context)
        predictor = DemoForecaster(run,raw,ref.ids,ref.area,OUT,core=cfg["core"],depth=cfg["depth"],
                                   seed=seed,n_params=cfg["n_params"])
        raw_predictions = {s:np.empty(shape,dtype=np.float32) for s in SCENARIOS}
        for j,basin in enumerate(ref.ids):
            days, obs, predictions = infer_basin(predictor,basin)
            if not np.array_equal(days,ref.dates.to_numpy(dtype="datetime64[D]")):
                raise ValueError(f"Sai lịch {ref.name}/{basin}")
            obs_error = compare_arrays(obs,ref.obs[:,j],1e-5,1e-5)
            abs_error = compare_arrays(predictions["S0"],ref.raw[:,j],PRED_ATOL,PRED_RTOL)
            audit_rows.append(dict(run=ref.name,basin_id=basin,max_raw_abs_error=abs_error,obs_abs_error=obs_error,passed=True))
            for scenario in SCENARIOS:
                values=predictions[scenario]
                if not np.isfinite(values).all():
                    raise ValueError(f"Dự báo không hữu hạn {ref.name}/{basin}/{scenario}")
                raw_predictions[scenario][:,j]=values
        # Giữ đúng dự báo S0 nguồn sau khi đối chiếu mô hình. Chỉ xuất 6 NPZ mới.
        forecasts.append(ref)
        for scenario in SCENARIOS[1:]:
            path,forecast=write_scenario(ref,scenario,raw_predictions[scenario],out_contracts)
            forecasts.append(forecast)
            exports.append(path)
        source_rows.append(dict(run=ref.name,configuration=conf,seed=seed,epoch=record["epoch"],
                                notebook=record["train_notebook"],environment_sha256=sha256(env_path),
                                checkpoint_sha256=record["handoff"]["sha256 best_model.pt"],
                                config_sha256=record["handoff"]["sha256 config.yml"],
                                scaler_sha256=record["handoff"]["sha256 train_data_scaler.yml"],
                                S0_contract_sha256=sha256(record["contract"])))
        del predictor,raw_predictions
        gc.collect()
        torch.cuda.empty_cache()
        elapsed = perf_counter() - started
        print(f"{index}/{len(records)} {ref.name}: S0 khớp, sáu kịch bản đã xuất ({elapsed:.1f}s).", flush=True)
    if {p.resolve() for p in out_contracts.glob("*.npz")} != {p.resolve() for p in exports}:
        raise ValueError("Có NPZ cũ trong Output. Chạy từ phiên sạch.")
    mask = common_mask(forecasts)
    if not mask.any():
        raise ValueError("Không có ngày–lưu vực chung qua mọi kịch bản.")
    basin_rows, summaries = [], []
    for f in forecasts:
        rows = [dict(run=f.name,configuration=f.metadata["config"],seed=f.seed,scenario=f.scenario,basin_id=bid,
                     **{k:score[k] for k in ("NSE","PNSE","KGE","n")})
                for j,bid in enumerate(f.ids)
                for score in [metrics(f.obs[:,j],f.sim[:,j],f.persistence[:,j],mask[:,j])]]
        basin_rows.extend(rows)
        fit=pd.DataFrame(rows)
        summaries.append(dict(run=f.name,configuration=f.metadata["config"],seed=f.seed,
                              scenario=f.scenario,**{k:float(fit[k].median()) for k in ("NSE","PNSE","KGE")}))
    table=pd.DataFrame(summaries)
    anchor=table[table.scenario=="S0"].set_index("run")
    for key in ("NSE","PNSE","KGE"):
        table[f"delta_{key}"]=table[key]-table.run.map(anchor[key])
    stats=table.groupby(["configuration","scenario"])[["NSE","PNSE","KGE","delta_NSE"]].agg(["mean","std"])
    stats.columns=["_".join(pair) for pair in stats.columns]
    outputs = dict(C2_scenarios_S0_check=pd.DataFrame(audit_rows),C2_scenarios_sources=pd.DataFrame(source_rows),
                   C2_scenarios_basin_scores=pd.DataFrame(basin_rows),C2_scenarios_seed_scores=table,
                   C2_scenarios_stats=stats.reset_index())
    for name,df in outputs.items():
        df.to_csv(OUT/f"{name}.csv",index=False,encoding="utf-8-sig")
    np.savez_compressed(OUT/"C2_scenarios_common_mask.npz", dates=context["dates"].to_numpy(dtype="datetime64[D]"),
                        basin_ids=np.array(context["ids"]),mask=mask)
    summary=dict(complete=True,notebook=NOTEBOOK,library_commit=FREEZE_C,period="test",lead=1,
                 configurations=list(C2_CONFIGS),seeds=list(C2_SEEDS),scenarios=list(SCENARIOS),
                 S0_reproduced=len(source_rows),S0_basin_checks=len(audit_rows),new_contracts=len(exports),
                 shared_days_IDs=int(mask.sum()),IDs_with_days=int(mask.any(axis=0).sum()),
                 comparison_scope="same_days_all_C2_runs_and_scenarios",C2_compare_version_expected=1,
                 C2_manifest_sha256=sha256(compare_root/"manifest.csv"),A3={p.name:sha256(p) for p in a3},optimizer_steps=0)
    summary_file.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    environment = dict(notebook=NOTEBOOK, neuralhydrology_commit=FREEZE_C, python=platform.python_version(),
                       torch=torch.__version__, torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                       C2_compare_version_expected=1, C2_train_version_expected=1, A3_version_expected=1,
                       mamba_wheel_version_expected=1)
    env_path = OUT/"environment.json"
    env_path.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    files=[*exports,*(OUT/f"{name}.csv" for name in outputs),OUT/"C2_scenarios_common_mask.npz",summary_file,env_path]
    pd.DataFrame([dict(file=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in files]
                 ).to_csv(OUT/"manifest.csv",index=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("GỬI LẠI: c2_scenarios_summary.json, C2_scenarios_S0_check.csv, C2_scenarios_sources.csv, "
          "C2_scenarios_seed_scores.csv, C2_scenarios_stats.csv, manifest.csv.")


if __name__ == "__main__":
    import torch
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Chọn Kaggle T4 cho suy luận C2.")
    install_library()
    install_mamba(ROOT)
    run_scenarios()
