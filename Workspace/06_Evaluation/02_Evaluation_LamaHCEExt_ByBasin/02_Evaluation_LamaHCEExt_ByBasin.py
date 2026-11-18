# %% [markdown]
# # Phân tích kết quả theo đặc điểm lưu vực — Extended LamaH-CE
#
# Đọc A1/A3 và Output bộ đánh giá 56 run (A2–B3 + năm lõi C2; vẫn tương thích 11/14/32 run),
# đồng thời tự động phân tích phân vùng D (730 trừ 365 bước trên cùng cấu hình và seed) nếu có `D/`.
# Không huấn luyện hoặc chọn lại mô hình; mọi phép so bước B, C2 theo lõi và D đều ghép cùng seed.
# S0 là không che thêm đầu vào; hạn một ngày dự báo ngày kế tiếp lịch sử quan trắc.
# Một ngày–lưu vực là một ngày tại một ID. Mặt nạ chọn các cặp được chấm;
# giao chung giữ các cặp có quan trắc, persistence và mọi dự báo cùng hữu hạn.
#
# Mục lục
# 1. Phạm vi, nguồn và cấu hình
# 2. Đọc bảng đánh giá, thuộc tính và kiểm khóa ID
# 3. Chẩn đoán độ phủ và biến thiên quan trắc từ A3
# 4. Hiệu kỹ năng và biến thiên ba seed
# 5. Spearman, hoán vị ID và Holm
# 6. Phân nhóm thuộc tính, góc hướng và lớp phủ
# 7. Hình kết quả, chú thích và dữ liệu vẽ
# 8. Kiểm thử độc lập
# 9. Chạy phân tích và lưu kết quả

# %% [markdown]
# ## Phần 1 — Phạm vi, nguồn và cấu hình
# ### 1.1. Các lựa chọn cố định trước khi xem liên hệ
# Dùng test/S0/hạn một ngày và common_source_all; giữ mọi điểm âm, đủ 451 ID.
# Bốn thuộc tính trình bày chính: area_calc, elev_mean, frac_snow, arid_1.
# Thuộc tính tĩnh là dữ liệu tổng hợp sẵn, không gọi chúng là thống kê fit riêng từ train.

# %% Phần 1.1 — Thư viện và hợp đồng phân tích
# Không cài gói hoặc sửa đầu vào; mã nguồn này tự đủ cho notebook Kaggle.
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

STATIC_FIELDS = (
    "area_calc", "elev_mean", "elev_med", "elev_std", "elev_ran", "slope_mean",
    "mvert_dist", "mvert_ang", "elon_ratio", "strm_dens", "p_mean", "et0_mean",
    "eta_mean", "arid_1", "arid_2", "p_season", "frac_snow", "hi_prec_fr",
    "hi_prec_du", "lo_prec_fr", "lo_prec_du", "lc_dom", "agr_fra", "bare_fra",
    "forest_fra", "lake_fra", "urban_fra", "lai_max", "lai_diff", "ndvi_max",
    "ndvi_min", "gvf_max", "gvf_diff",
)
NUMERIC_FIELDS = tuple(x for x in STATIC_FIELDS if x not in {"lc_dom", "mvert_ang"})
PRIORITY_FIELDS = ("area_calc", "elev_mean", "frac_snow", "arid_1")
FEATURE_LABELS = {"area_calc": "Catchment area (km²)", "elev_mean": "Mean elevation (m a.s.l.)",
                  "frac_snow": "Snow fraction of precipitation", "arid_1": "Aridity (ET0 / precipitation)"}
DIAGNOSTICS = ("train_missing_fraction", "train_qmax_std_mm_day", "train_sample_n",
               "test_common_qmax_std_mm_day", "test_common_days")
MODELS = ("SeqLSTM_Q", "SeqLSTM_noQ", "EncDec_Q", "EncDec_noQ", "D_FC", "D_RA",
          "DLinear_seed_111", "DLinear_seed_222", "DLinear_seed_333")
CONTRASTS = tuple((name, "Persistence_qmax") for name in MODELS) + (
    ("SeqLSTM_Q", "SeqLSTM_noQ"), ("EncDec_Q", "EncDec_noQ"), ("D_RA", "D_FC"),
)
SEEDS = (111, 222, 333)
A4_MODELS = tuple(f"SeqLSTM_Q_A4_seed{seed}" for seed in SEEDS)
A4_CONTRASTS = tuple((name, "Persistence_qmax") for name in A4_MODELS) + tuple(
    (name, "SeqLSTM_Q") for name in A4_MODELS)
B_STEPS = ("B0", "B1a", "B1b", "B2", "B3_P10S12", "B3_P05S05")
B_MODELS = {step: tuple(f"SeqLSTM_Q_{step}_seed{seed}" for seed in SEEDS) for step in B_STEPS}
B_PREVIOUS = dict(B0="A4", B1a="B0", B1b="B1a", B2="B1b", B3_P10S12="B2", B3_P05S05="B2")
B_CONTRASTS = tuple((f"SeqLSTM_Q_{step}_seed{seed}", f"SeqLSTM_Q_{B_PREVIOUS[step]}_seed{seed}")
                    for step in B_STEPS for seed in SEEDS)
FREEZE_C = "db6dc1ffcd8c14994cd80be3ed8ff417002c01a1"
C2_CONFIG_SPECS = {
    "lstm_L2_x1_lr1e-3": ("lstm", 2, 1, 0.001, 84106),
    "gru_L2_x1_lr1e-3": ("gru", 2, 1, 0.001, 84503),
    "transformer_L2_x1_lr1e-3": ("transformer", 2, 1, 0.001, 84823),
    "transformer_L2_x3_lr5e-4": ("transformer", 2, 3, 0.0005, 254425),
    "s4d_L4_x1_lr1e-3": ("s4d", 4, 1, 0.001, 84449),
    "s4d_L4_x3_lr1e-3": ("s4d", 4, 3, 0.001, 255833),
    "mamba_L2_x1_lr1e-3": ("mamba", 2, 1, 0.001, 85039),
    "mamba_L4_x3_lr1e-3": ("mamba", 4, 3, 0.001, 254677),
}
C2_CONFIGS = tuple(C2_CONFIG_SPECS)
C2_NON_LSTM_CONFIGS = tuple(conf for conf in C2_CONFIGS if conf != "lstm_L2_x1_lr1e-3")
C2_MODELS = {conf: tuple(f"SF_{conf}_C2_seed{seed}" for seed in SEEDS) for conf in C2_CONFIGS}
C2_ALL_RUNS = tuple(name for names in C2_MODELS.values() for name in names)
C2_PERSISTENCE_CONTRASTS = tuple((name, "Persistence_qmax") for name in C2_ALL_RUNS)
C2_BSTAR_CONTRASTS = tuple((f"SF_{conf}_C2_seed{seed}", f"SeqLSTM_Q_B3_P10S12_seed{seed}")
                           for conf in C2_CONFIGS for seed in SEEDS)
C2_CORE_CONTRASTS = tuple((f"SF_{conf}_C2_seed{seed}", f"SF_lstm_L2_x1_lr1e-3_C2_seed{seed}")
                          for conf in C2_NON_LSTM_CONFIGS for seed in SEEDS)
C2_CONTRASTS = C2_PERSISTENCE_CONTRASTS + C2_BSTAR_CONTRASTS + C2_CORE_CONTRASTS
D_CONFIGS = ("lstm_L2_x1_lr1e-3", "mamba_L2_x1_lr1e-3", "s4d_L4_x1_lr1e-3")

SEED_FAMILIES = {
    "DLinear": tuple(f"DLinear_seed_{seed}" for seed in SEEDS),
    "LSTM_A4": A4_MODELS,
    **{f"LSTM_{step}": names for step, names in B_MODELS.items()},
    **{f"C2_{conf}": names for conf, names in C2_MODELS.items()},
    **{f"C2_{conf}_vs_Bstar": names for conf, names in C2_MODELS.items()},
    **{f"C2_{conf}_vs_LSTM": C2_MODELS[conf] for conf in C2_NON_LSTM_CONFIGS},
}
FAMILY_REFERENCES = {family: ("Persistence_qmax",)*3 for family in ("DLinear", "LSTM_A4")}
FAMILY_REFERENCES.update({f"LSTM_{step}": tuple(f"SeqLSTM_Q_{B_PREVIOUS[step]}_seed{seed}" for seed in SEEDS)
                          for step in B_STEPS})
FAMILY_REFERENCES.update({f"C2_{conf}": ("Persistence_qmax",)*3 for conf in C2_CONFIGS})
FAMILY_REFERENCES.update({f"C2_{conf}_vs_Bstar": B_MODELS["B3_P10S12"] for conf in C2_CONFIGS})
FAMILY_REFERENCES.update({f"C2_{conf}_vs_LSTM": C2_MODELS["lstm_L2_x1_lr1e-3"] for conf in C2_NON_LSTM_CONFIGS})

LIBRARY_COMMITS = dict(A4="dda45fcf19ee7ffd481b9d192778539df46fb074",
    B0="f229e02bf98cbd979835a4b6cd503d0e3adb8bc3", B1a="f229e02bf98cbd979835a4b6cd503d0e3adb8bc3",
    B1b="f229e02bf98cbd979835a4b6cd503d0e3adb8bc3", B2="2ccaa147a6ec87d0f56fa3e2f11b1bf38079253f",
    B3="2ccaa147a6ec87d0f56fa3e2f11b1bf38079253f", C2=FREEZE_C, D=FREEZE_C)
ENDPOINTS = ("delta_NSE", "delta_PNSE", "delta_POD_T2")
PERMUTATIONS, PERMUTATION_SEED, MIN_ASSOCIATION_N = 9999, 20261023, 10
TRAIN_START, TRAIN_END = "2003-01-01", "2009-12-31"
TEST_START, TEST_END = "2014-01-01", "2017-12-31"
METRIC_KEYS = ("NSE", "PNSE", "KGE", "r", "alpha", "beta")


def analysis_contrasts(runs):
    runs = set(runs)
    baseline = set(MODELS) | {"Persistence_qmax", "Persistence_qmean_aux"}
    b_all = {name for names in B_MODELS.values() for name in names}
    if runs == baseline:
        return CONTRASTS
    if runs == baseline | set(A4_MODELS):
        return CONTRASTS + A4_CONTRASTS
    if runs == baseline | set(A4_MODELS) | b_all:
        return CONTRASTS + A4_CONTRASTS + B_CONTRASTS
    if runs == baseline | set(A4_MODELS) | b_all | set(C2_ALL_RUNS):
        return CONTRASTS + A4_CONTRASTS + B_CONTRASTS + C2_CONTRASTS
    raise ValueError(
        "Cần đúng 11 run gốc, 14 run có A4, 32 run đủ A4/B0–B3 hoặc 56 run đủ A4/B0–B3/C2; "
        "không nhận thiếu seed hoặc run lạ"
    )


def family_reference_rule(family):
    if family.startswith("LSTM_B"):
        return "same_seed_previous_ablation_step"
    if family.endswith("_vs_Bstar"):
        return "same_seed_Bstar_P10S12"
    if family.endswith("_vs_LSTM"):
        return "same_seed_C2_lstm_L2_x1"
    if family.startswith("D_"):
        return "same_seed_C2_365_counterpart"
    return "Persistence_qmax"


def family_contrast_names(family, families=SEED_FAMILIES, references=FAMILY_REFERENCES):
    return tuple(f"{left}__minus__{right}" for left, right in zip(families[family], references[family]))


def active_seed_families(runs, contrasts=None):
    runs = set(runs)
    available = set(contrasts) if contrasts is not None else None
    result = {}
    for family, names in SEED_FAMILIES.items():
        if set(names).issubset(runs) and set(FAMILY_REFERENCES[family]).issubset(runs):
            if available is None or set(family_contrast_names(family)).issubset(available):
                result[family] = names
    return result


def stable_seed(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


# %% [markdown]
# ## Phần 2 — Đọc bảng đánh giá, thuộc tính và kiểm khóa ID
# ### 2.1. Nguồn duy nhất và ghép theo ID
# Gắn bốn Input: dataset lamah-ce-ext, Output A1, Output A3, Output bộ đánh giá (và tự nhận `D/` nếu có).
# Không cần gắn lại A2/A4/B/C2/D Compare: Output đánh giá đã chứa chỉ số, run index và bảng kiểm nguồn/hợp đồng.
# Có nhiều tệp cùng tên thì chỉ định đường dẫn, không tự chọn phiên bản mới nhất.

# %% Phần 2.1 — Tìm tệp và kiểm định dạng khóa
# Mọi phép ghép đều kiểm một–một; 0 của quan trắc/thuộc tính là giá trị hợp lệ.
def unique_file(root, name, suffix=None, optional=False):
    paths = sorted(p for p in Path(root).rglob(name) if p.is_file()
                   and (suffix is None or p.as_posix().endswith(suffix)))
    if optional and not paths:
        return None
    if len(paths) != 1:
        raise FileNotFoundError(f"Cần đúng một {name} dưới {root}; thấy {len(paths)}: {paths[:5]}")
    return paths[0]


def normalize_ids(values):
    text = pd.Series(values, dtype="string").str.strip()
    if text.isna().any() or not text.str.fullmatch(r"\d+").all():
        raise ValueError("ID phải là số nguyên không âm, không rỗng")
    result = text.map(lambda x: str(int(x)))
    if result.duplicated().any():
        raise ValueError("ID trùng sau chuẩn hóa; không tự bỏ dòng")
    return pd.Index(result, name="basin_id")


def read_indexed(path, id_column="basin_id", sep=","):
    frame = pd.read_csv(path, sep=sep, dtype={id_column: str}, keep_default_na=False)
    if id_column not in frame:
        raise ValueError(f"{path}: thiếu cột {id_column}")
    frame.index = normalize_ids(frame.pop(id_column))
    return frame


def numeric_column(values, name):
    text = values.astype("string").str.strip()
    missing = text.isin(["", "NA", "NaN", "nan", "-999"])
    numbers = pd.to_numeric(text.where(~missing), errors="coerce").astype(float)
    if ((~missing) & numbers.isna()).any() or np.isinf(numbers).any():
        raise ValueError(f"{name}: có token không khai báo hoặc vô cực")
    return numbers


def validate_metrics_and_events(metrics, events, expected_ids):
    required_runs = set(metrics.run)
    if metrics.duplicated(["run", "basin_id"]).any():
        raise ValueError("Trùng run–ID trong bảng chỉ số")
    for frame in (metrics, events):
        frame["basin_id"] = frame.basin_id.map(lambda x: str(int(x)) if re.fullmatch(r"\d+", str(x)) else None)
        if frame.basin_id.isna().any():
            raise ValueError("ID đánh giá không hợp lệ")
    if metrics.duplicated(["run", "basin_id"]).any():
        raise ValueError("Trùng run–ID sau chuẩn hóa khóa")
    values = metrics[list(METRIC_KEYS)+["n", "obs_tss"]].to_numpy(dtype=float)
    if np.isinf(values).any() or not (metrics.n.ge(0) & metrics.n.eq(np.floor(metrics.n))).all():
        raise ValueError("Chỉ số có vô cực hoặc số ngày chấm không phải số nguyên không âm")
    ids = normalize_ids(sorted(metrics.basin_id.unique(), key=int))
    if len(ids) != expected_ids or any(set(g.basin_id) != set(ids) for _, g in metrics.groupby("run")):
        raise ValueError("Danh sách ID không đầy đủ ở từng run")
    counts = metrics.pivot(index="basin_id", columns="run", values="n").reindex(ids)
    if counts.isna().any().any() or not counts.eq(counts.iloc[:, 0], axis=0).all().all():
        raise ValueError("Các run không có cùng số ngày chấm ở từng ID")
    denominators = metrics.pivot(index="basin_id", columns="run", values="obs_tss").reindex(ids)
    if not np.allclose(denominators, denominators.iloc[:, [0]].to_numpy(),
                       equal_nan=True, rtol=1e-10, atol=1e-12):
        raise ValueError("Mẫu số NSE khác nhau giữa các run; cần quan trắc và phạm vi chấm chung")
    events = events.loc[events.grouping.eq("USWRC") & events.return_period_years.eq(2)
                        & events.matching_tolerance_days.eq(0)].copy()
    if (set(events.run) != required_runs or events.duplicated(["run", "basin_id"]).any()
            or len(events) != len(metrics)):
        raise ValueError("Thiếu bảng USWRC/T=2/tolerance=0 theo từng run–ID")
    event_counts = events[["hit", "missed", "false_alarm"]].to_numpy(dtype=float)
    if (not np.isfinite(event_counts).all() or (event_counts < 0).any()
            or not np.equal(event_counts, np.floor(event_counts)).all()):
        raise ValueError("Số trận phải là số nguyên hữu hạn không âm")
    for _, group in events.groupby("run"):
        if set(group.basin_id) != set(ids):
            raise ValueError("Sự kiện không giữ đủ danh sách ID")
    obs_counts = events.assign(observed_n=events.hit + events.missed).pivot(
        index="basin_id", columns="run", values="observed_n")
    if not obs_counts.eq(obs_counts.iloc[:, 0], axis=0).all().all():
        raise ValueError("Mẫu số trận thật khác nhau giữa các run; không được trừ POD")
    expected_pod = np.divide(events.hit, events.hit + events.missed,
                             out=np.full(len(events), np.nan), where=(events.hit + events.missed).gt(0))
    if not np.allclose(events.POD, expected_pod, equal_nan=True, rtol=1e-10, atol=1e-12):
        raise ValueError("POD không khớp số đếm hoặc đã gán 0 cho lưu vực không có trận")
    return metrics, events, ids


def load_evaluation(root, expected_ids=451):
    paths = [Path(root) / f"Evaluation_{name}" for name in (
        "summary.json", "protocol.json", "metrics.csv", "events.csv")]
    summary, protocol = [json.loads(p.read_text(encoding="utf-8-sig")) for p in paths[:2]]
    if summary.get("complete") is not True or summary.get("A3_reference_matched") is not True:
        raise ValueError("Bộ đánh giá chưa hoàn thành hoặc chưa khớp A3")
    version = protocol.get("protocol_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 2:
        raise ValueError("Cần Output bộ đánh giá từ giao thức 2 trở lên")
    tolerance = protocol.get("event_tolerance_days_by_lead", {}).get("1", protocol.get("event_tolerance_days"))
    if tolerance != 0:
        raise ValueError("Cần bảng sự kiện chính cùng ngày bắt đầu ở hạn một ngày")
    metrics, events = [pd.read_csv(p, dtype={"basin_id": str}) for p in paths[2:]]
    selected = lambda frame: frame.loc[frame.period.eq("test") & frame.scenario.eq("S0")
                                     & frame.lead.eq(1) & frame.scope.eq("common_source_all")].copy()
    metrics, events = selected(metrics), selected(events)
    analysis_contrasts(metrics.run)
    required_runs = set(metrics.run)
    if required_runs & set(A4_MODELS):
        contract_path = Path(root) / "Evaluation_A4_contract_check.csv"
        input_path = Path(root) / "Evaluation_A4_input_check.csv"
        if (version < 3 or summary.get("A4_contracts_matched") is not True
                or set(summary.get("A4_runs", [])) != set(A4_MODELS)
                or summary.get("A4_compare_version_expected") != 2):
            raise ValueError("Nguồn có A4 cần giao thức 3 và xác nhận đủ ba hợp đồng A4 Compare Version 2")
        contracts, inputs = [pd.read_csv(p) for p in (contract_path, input_path)]
        if (len(contracts) != 3 or set(contracts.run) != set(A4_MODELS)
                or contracts.run.duplicated().any() or len(inputs) == 0
                or not inputs.passed.astype(str).str.lower().eq("true").all()):
            raise ValueError("Bảng kiểm nguồn hoặc danh sách hợp đồng A4 không đạt")
        for flag in ("canonical_context_matches", "manifest_sha256_matches"):
            if not contracts[flag].astype(str).str.lower().eq("true").all():
                raise ValueError(f"Hợp đồng A4 không đạt {flag}")
        for seed, name in zip(SEEDS, A4_MODELS):
            contract = contracts.set_index("run").loc[name]
            if (contract.seed != seed or contract.IDs != expected_ids
                    or contract.period != "test" or contract.scenario != "S0" or contract.lead != 1
                    or contract.compare_version_expected != 2):
                raise ValueError(f"{name}: metadata trong bảng kiểm A4 không hợp lệ")
            for frame in (metrics, events):
                selected_seed = frame.loc[frame.run.eq(name), "seed"]
                if len(selected_seed) == 0 or not selected_seed.eq(seed).all():
                    raise ValueError(f"{name}: seed chỉ số/sự kiện không khớp hợp đồng")
        paths.extend([contract_path, input_path])
    elif summary.get("A4_runs"):
        raise ValueError("Summary có A4 nhưng bảng chỉ số không giữ đủ ba seed")
    if required_runs & set(B_MODELS["B0"]):
        paths.extend(audit_B_evaluation(root, metrics, events, summary, version, expected_ids))
    elif summary.get("B0_runs"):
        raise ValueError("Summary có bước B nhưng bảng chỉ số không giữ đủ các run B")
    if required_runs & set(C2_ALL_RUNS):
        paths.extend(audit_C2_evaluation(root, metrics, events, summary, version, expected_ids))
    elif summary.get("C2_runs"):
        raise ValueError("Summary có C2 nhưng bảng chỉ số không giữ đủ 24 run C2")
    metrics, events, ids = validate_metrics_and_events(metrics, events, expected_ids)
    return metrics, events, ids, summary, protocol, paths


def audit_B_evaluation(root, metrics, events, summary, version, expected_ids):
    # Kiểm bảng bàn giao Evaluation V8 (32 run) hoặc V9–V12 (56 run); không nhận là đã đọc lại NPZ gốc.
    root = Path(root)
    runs = set(metrics.run)
    if len(runs) not in (32, 56) or version != 3:
        raise ValueError("Hồ sơ có bước B cần đủ 32 run (V8) hoặc 56 run (V9+) của giao thức Evaluation 3")
    coverage = [row for row in summary.get("coverage", []) if isinstance(row, dict)
                and (row.get("period"), row.get("scenario"), row.get("lead")) == ("test", "S0", 1)]
    if (len(coverage) != 1 or coverage[0].get("runs") != len(runs)
            or coverage[0].get("common_day_IDs") != int(metrics.groupby("run").n.sum().iloc[0])
            or coverage[0].get("IDs_with_days") != metrics.loc[metrics.n.gt(0), "basin_id"].nunique()):
        raise ValueError("Độ phủ trong summary khác bảng test/S0/hạn 1")
    specifications = {
        "A4": (A4_MODELS, "A4_contracts_matched", "A4_runs", "A4_compare_version_expected", 2),
        "B0": (B_MODELS["B0"], "B0_contracts_matched", "B0_runs", "B0_compare_version_expected", 1),
        "B1a": (B_MODELS["B1a"], "B1_contracts_matched", "B1a_runs", "B1_compare_version_expected", 1),
        "B1b": (B_MODELS["B1b"], "B1_contracts_matched", "B1b_runs", "B1_compare_version_expected", 1),
        "B2": (B_MODELS["B2"], "B2_contracts_matched", "B2_runs", "B2_compare_version_expected", 1),
        "B3": (B_MODELS["B3_P10S12"]+B_MODELS["B3_P05S05"], "B3_contracts_matched", "B3_runs", "B3_compare_version_expected", 1),
    }
    if summary.get("B3_runs_by_mask") != {mask: list(B_MODELS[f"B3_{mask}"]) for mask in ("P10S12", "P05S05")}:
        raise ValueError("Summary B3 không giữ đúng hai mức che và ba seed mỗi mức")
    path = root / "Evaluation_run_index.json"
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(records, list) or len(records) != len(runs)
            or any(not isinstance(row, dict) for row in records)
            or len({row.get("run") for row in records}) != len(runs)
            or {row.get("run") for row in records} != runs):
        raise ValueError("Run index không giữ đúng một metadata cho từng run")
    indexed = {row["run"]: row for row in records}
    paths, a3_hashes = [path], {}
    days = len(pd.date_range(TEST_START, TEST_END))
    for step, (names, flag, summary_key, version_key, compare_version) in specifications.items():
        if (summary.get(flag) is not True or set(summary.get(summary_key, [])) != set(names)
                or summary.get(version_key) != compare_version):
            raise ValueError(f"{step}: summary chưa xác nhận đủ hợp đồng/Version")
        contract_path, input_path = [root / f"Evaluation_{step}_{kind}_check.csv" for kind in ("contract", "input")]
        contracts, inputs = [pd.read_csv(p) for p in (contract_path, input_path)]
        required = {"run", "seed", "selected_epoch", "training_notebook_version", "compare_version_expected",
                    "IDs", "target_days", "library_commit", "config_sha256", "canonical_context_matches",
                    "manifest_sha256_matches"}
        if step in ("B2", "B3"):
            required.update({"NSE_audit_mask", "NSE_audit_median", "NSE_audit_reported_median",
                             "NSE_audit_abs_difference", "training_notebook"})
        if step == "B3":
            required.update({"mask_level", "nan_step_probability", "nan_sequence_probability"})
        input_required = {"file", "expected_sha256", "observed_sha256", "passed"}
        if (not required.issubset(contracts.columns) or not input_required.issubset(inputs.columns)
                or len(contracts) != len(names) or contracts.run.duplicated().any()
                or set(contracts.run) != set(names) or inputs.empty):
            raise ValueError(f"{step}: thiếu cột, seed hoặc bảng kiểm nguồn")
        for values in (contracts.canonical_context_matches, contracts.manifest_sha256_matches, inputs.passed):
            if not values.astype(str).str.lower().eq("true").all():
                raise ValueError(f"{step}: có cờ kiểm nguồn/hợp đồng không đạt")
        for field in ("expected_sha256", "observed_sha256"):
            if not inputs[field].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
                raise ValueError(f"{step}: SHA-256 nguồn không hợp lệ")
        if not inputs.expected_sha256.eq(inputs.observed_sha256).all():
            raise ValueError(f"{step}: SHA-256 nguồn không khớp")
        for name in ("A3_persistence_predictions.npz", "A3_Q_source_window.npz"):
            source = inputs.loc[inputs.file.eq(name)]
            if len(source) != 1:
                raise ValueError(f"{step}: cần đúng một fingerprint {name}")
            digest = source.iloc[0].observed_sha256
            if name in a3_hashes and a3_hashes[name] != digest:
                raise ValueError("Các bước không dùng cùng ngữ cảnh A3")
            a3_hashes[name] = digest
        for _, contract in contracts.iterrows():
            name = contract.run
            seed = next(seed for seed in SEEDS if name.endswith(f"_seed{seed}"))
            row, meta = indexed[name], indexed[name].get("source_metadata", {})
            training_version = 2 if step == "A4" and seed != 111 else 1
            if (not isinstance(meta, dict) or contract.seed != seed or contract.IDs != expected_ids
                    or contract.target_days != days or contract.compare_version_expected != compare_version
                    or contract.training_notebook_version != training_version
                    or not 1 <= contract.selected_epoch <= 30 or contract.selected_epoch != int(contract.selected_epoch)
                    or contract.library_commit != LIBRARY_COMMITS[step]
                    or not re.fullmatch(r"[0-9a-f]{64}", str(contract.config_sha256))
                    or any(row.get(key) != value for key, value in dict(seed=seed, IDs=expected_ids, days=days,
                        period="test", scenario="S0", lead=1).items())
                    or any(meta.get(key) != value for key, value in dict(run=name, seed=seed, period="test", scenario="S0",
                        lead=1, forecast_horizon=1, unit="mm/day_area_gov", clip_to_zero=True,
                        selected_epoch=int(contract.selected_epoch), config_sha256=contract.config_sha256,
                        library_commit=LIBRARY_COMMITS[step], compare_version_expected=compare_version,
                        training_notebook_version=training_version).items())):
                raise ValueError(f"{name}: metadata ngày/seed/epoch/commit/Version khác bảng kiểm")
            integer_fields = ("seed", "lead", "forecast_horizon", "selected_epoch", "compare_version_expected",
                              "training_notebook_version")
            if any(not isinstance(meta.get(key), int) or isinstance(meta.get(key), bool) for key in integer_fields):
                raise ValueError(f"{name}: metadata số nguyên không được là boolean hoặc số thực")
            if meta.get("clip_to_zero") is not True:
                raise ValueError(f"{name}: clip_to_zero không phải boolean True")
            for frame in (metrics, events):
                part = frame.loc[frame.run.eq(name)]
                if (part.empty or not part.seed.eq(seed).all() or not part.period.eq("test").all()
                        or not part.scenario.eq("S0").all() or not part.lead.eq(1).all()):
                    raise ValueError(f"{name}: nhãn kỳ/kịch bản/hạn/seed trong kết quả không khớp")
            for key, expected in (("period", "test"), ("scenario", "S0"), ("lead", 1)):
                if key in contracts and contract[key] != expected:
                    raise ValueError(f"{name}: nhãn {key} trong hợp đồng không đúng")
            relative = f"contracts/{name}_test_S0_lead1.npz"
            source = inputs.loc[inputs.file.eq(relative)]
            if (len(source) != 1 or meta.get("source_file_sha256") != source.iloc[0].observed_sha256
                    or Path(str(meta.get("source", ""))).name != Path(relative).name):
                raise ValueError(f"{name}: fingerprint NPZ khác run index")
            if step in ("B2", "B3"):
                if (contract.NSE_audit_mask != "finite_obs_and_sim"
                        or not np.isfinite(contract.NSE_audit_abs_difference)
                        or not 0 <= contract.NSE_audit_abs_difference <= 1e-5
                        or not np.isfinite([contract.NSE_audit_median, contract.NSE_audit_reported_median]).all()
                        or not np.isclose(abs(contract.NSE_audit_median-contract.NSE_audit_reported_median),
                                          contract.NSE_audit_abs_difference, rtol=0, atol=1e-12)):
                    raise ValueError(f"{name}: audit NSE từ NPZ không đạt")
                notebook = (f"06_B2_LamaHCEExt_Train_Seed{seed}" if step == "B2" else
                            f"08_B3_LamaHCEExt_Train_{contract.mask_level}_Seed{seed}")
                if contract.training_notebook != notebook or meta.get("training_notebook") != notebook:
                    raise ValueError(f"{name}: tên nguồn train khác hợp đồng")
            if step == "B3":
                mask = "P10S12" if "_P10S12_" in name else "P05S05"
                probabilities = (.1, .12) if mask == "P10S12" else (.05, .05)
                if (contract.mask_level != mask or meta.get("mask_level") != mask
                        or meta.get("step") != "B3"
                        or any(contract[key] != probability or meta.get(key) != probability
                               for key, probability in zip(("nan_step_probability", "nan_sequence_probability"), probabilities))):
                    raise ValueError(f"{name}: mức che/xác suất B3 không khớp")
        paths.extend([contract_path, input_path])
    return paths


def audit_C2_evaluation(root, metrics, events, summary, version, expected_ids):
    root = Path(root)
    runs = set(metrics.run)
    if (len(runs) != 56 or version != 3 or summary.get("C2_contracts_matched") is not True
            or summary.get("C2_compare_version_expected") != 1
            or set(summary.get("C2_runs", [])) != set(C2_ALL_RUNS)):
        raise ValueError("Hồ sơ C2 cần đủ 56 run (24 run C2), giao thức 3 và C2 Compare Version 1 đạt hợp đồng")
    contract_path = root / "Evaluation_C2_contract_check.csv"
    input_path = root / "Evaluation_C2_input_check.csv"
    index_path = root / "Evaluation_run_index.json"
    contracts, inputs = [pd.read_csv(p) for p in (contract_path, input_path)]
    records = json.loads(index_path.read_text(encoding="utf-8-sig"))
    indexed = {row["run"]: row for row in records if isinstance(row, dict) and "run" in row}
    required_cols = {"run", "configuration_id", "seed", "selected_epoch", "n_params", "IDs", "target_days",
                     "finite_prediction_day_IDs", "canonical_context_matches", "native_test_NSE", "NPZ_test_NSE",
                     "contract_sha256", "checkpoint_sha256"}
    input_cols = {"file", "path", "bytes", "sha256", "passed"}
    if (not required_cols.issubset(contracts.columns) or not input_cols.issubset(inputs.columns)
            or len(contracts) != 24 or contracts.run.duplicated().any()
            or set(contracts.run) != set(C2_ALL_RUNS) or inputs.empty):
        raise ValueError("C2: thiếu cột, thiếu run hoặc bảng kiểm nguồn rỗng")
    if (not contracts.canonical_context_matches.astype(str).str.lower().eq("true").all()
            or not inputs.passed.astype(str).str.lower().eq("true").all()
            or not inputs.sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}").all()):
        raise ValueError("C2: cờ hợp đồng/manifest hoặc SHA-256 nguồn không đạt")
    days = len(pd.date_range(TEST_START, TEST_END))
    for _, contract in contracts.iterrows():
        name = contract.run
        conf = contract.configuration_id
        if conf not in C2_CONFIG_SPECS:
            raise ValueError(f"{name}: cấu hình C2 lạ {conf}")
        core, depth, level, lr, n_params = C2_CONFIG_SPECS[conf]
        seed = int(contract.seed)
        expected_name = f"SF_{conf}_C2_seed{seed}"
        row = indexed.get(name, {})
        meta = row.get("source_metadata", {})
        allowed_notebooks = {f"12_C2_Train_{conf}_Seed{seed}", f"12_C2_Train_{conf}_Seed{seed}_Cont"}
        if (name != expected_name or seed not in SEEDS or contract.IDs != expected_ids
                or contract.target_days != days or int(contract.n_params) != n_params
                or not 1 <= int(contract.selected_epoch) <= 30
                or not np.isfinite([contract.native_test_NSE, contract.NPZ_test_NSE]).all()
                or abs(float(contract.native_test_NSE) - float(contract.NPZ_test_NSE)) > 1e-5
                or not re.fullmatch(r"[0-9a-f]{64}", str(contract.contract_sha256))
                or not re.fullmatch(r"[0-9a-f]{64}", str(contract.checkpoint_sha256))
                or not isinstance(meta, dict)
                or any(row.get(k) != v for k, v in dict(seed=seed, IDs=expected_ids, days=days,
                    period="test", scenario="S0", lead=1).items())
                or any(meta.get(k) != v for k, v in dict(run=name, step="C2", model="sequential_forecast",
                    config=conf, core=core, depth=depth, level=level, n_params=n_params,
                    selected_epoch=int(contract.selected_epoch), period="test", scenario="S0",
                    lead=1, forecast_horizon=1, unit="mm/day_area_gov", clip_to_zero=True,
                    library_commit=FREEZE_C, compare_notebook="13_C2_LamaHCEExt_Compare",
                    compare_version_expected=1, training_notebook_version_expected=1,
                    source_file_sha256=contract.contract_sha256,
                    checkpoint_sha256=contract.checkpoint_sha256).items())
                or not np.isclose(float(meta.get("learning_rate", np.nan)), lr, rtol=0, atol=1e-12)
                or meta.get("training_notebook") not in allowed_notebooks):
            raise ValueError(f"{name}: metadata C2 hoặc SHA-256 khác bảng kiểm hợp đồng")
        gg = meta.get("grad_guard_skipped_steps")
        if gg is not None and (not isinstance(gg, int) or isinstance(gg, bool) or gg < 0):
            raise ValueError(f"{name}: grad_guard_skipped_steps không hợp lệ")
        rel = f"contracts/{name}_test_S0_lead1.npz"
        src = inputs.loc[inputs.file.eq(rel)]
        if len(src) != 1 or src.iloc[0].sha256 != contract.contract_sha256:
            raise ValueError(f"{name}: SHA-256 NPZ không khớp bảng C2_input_check")
        for frame in (metrics, events):
            part = frame.loc[frame.run.eq(name)]
            if (part.empty or not part.seed.eq(seed).all() or not part.period.eq("test").all()
                    or not part.scenario.eq("S0").all() or not part.lead.eq(1).all()):
                raise ValueError(f"{name}: nhãn kỳ/kịch bản/hạn/seed trong kết quả C2 không khớp")
    return [contract_path, input_path]


def load_d_evaluation(d_root, ids):
    d_root = Path(d_root)
    req_names = ("summary.json", "coverage.csv", "contract_check.csv", "input_check.csv",
                 "metrics.csv", "events.csv", "common_mask.npz")
    paths = [d_root / f"Evaluation_D_{name}" for name in req_names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Phân vùng D thiếu tệp bắt buộc: {path}")
    summary = json.loads(paths[0].read_text(encoding="utf-8-sig"))
    coverage = pd.read_csv(paths[1])
    contracts = pd.read_csv(paths[2])
    inputs = pd.read_csv(paths[3])
    s4d_inc = summary.get("S4D_included")
    if not isinstance(s4d_inc, bool):
        raise ValueError("Evaluation_D_summary.json thiếu cờ boolean S4D_included")
    expected_configs = list(D_CONFIGS if s4d_inc else D_CONFIGS[:-1])
    expected_d_runs = [f"SF_{conf}_D_seed{seed}" for conf in expected_configs for seed in SEEDS]
    expected_c2_runs = [f"SF_{conf}_C2_seed{seed}" for conf in expected_configs for seed in SEEDS]
    expected_all = set(expected_d_runs) | set(expected_c2_runs) | {"Persistence_qmax"}
    if (summary.get("complete") is not True or summary.get("protocol_version") != 3
            or summary.get("compare_notebook") != "17_D_LamaHCEExt_Compare"
            or summary.get("library_commit") != FREEZE_C
            or summary.get("configs") != expected_configs
            or summary.get("seeds") != list(SEEDS)
            or summary.get("contracts") != len(expected_d_runs)
            or summary.get("paired_c2_runs") != len(expected_c2_runs)
            or summary.get("total_scored_runs") != len(expected_all)
            or summary.get("event_tolerance_days") != 0):
        raise ValueError("Evaluation_D_summary.json không khớp hợp đồng bước D")
    if (len(contracts) != len(expected_d_runs) or contracts.run.duplicated().any()
            or set(contracts.run) != set(expected_d_runs)
            or not contracts.canonical_context_matches.astype(str).str.lower().eq("true").all()
            or inputs.empty or not inputs.passed.astype(str).str.lower().eq("true").all()
            or not inputs.sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}").all()):
        raise ValueError("Bảng hợp đồng hoặc bảng kiểm nguồn của Evaluation_D không đạt")
    days = len(pd.date_range(TEST_START, TEST_END))
    for _, row in contracts.iterrows():
        conf = row.configuration_id
        if conf not in expected_configs:
            raise ValueError(f"Cấu hình D ngoài danh sách khai báo: {conf}")
        _, _, _, _, n_params = C2_CONFIG_SPECS[conf]
        seed = int(row.seed)
        if (row.run != f"SF_{conf}_D_seed{seed}" or row.baseline_run != f"SF_{conf}_C2_seed{seed}"
                or seed not in SEEDS or int(row.seq_length) != 730 or int(row.n_params) != n_params
                or int(row.IDs) != len(ids) or int(row.target_days) != days
                or not 1 <= int(row.selected_epoch) <= 30
                or abs(float(row.native_test_NSE) - float(row.NPZ_test_NSE)) > 1e-5
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.contract_sha256))
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.checkpoint_sha256))):
            raise ValueError(f"{row.run}: metadata hợp đồng D không hợp lệ")
        rel = f"contracts/{row.run}_test_S0_lead1.npz"
        src = inputs.loc[inputs.file.eq(rel)]
        if len(src) != 1 or src.iloc[0].sha256 != row.contract_sha256:
            raise ValueError(f"{row.run}: SHA-256 NPZ D không khớp Evaluation_D_input_check.csv")
    metrics, events = [pd.read_csv(p, dtype={"basin_id": str}) for p in paths[4:6]]
    selected = lambda frame: frame.loc[frame.period.eq("test") & frame.scenario.eq("S0")
                                     & frame.lead.eq(1) & frame.scope.eq("common_source_all")].copy()
    metrics, events = selected(metrics), selected(events)
    if set(metrics.run) != expected_all:
        raise ValueError("Evaluation_D_metrics.csv thiếu hoặc thừa run so với cặp 365/730 + Persistence_qmax")
    metrics, events, d_ids = validate_metrics_and_events(metrics, events, len(ids))
    if not d_ids.equals(ids):
        raise ValueError("Danh sách ID của phân vùng D khác danh sách chuẩn 451 ID")
    with np.load(paths[6], allow_pickle=False) as mask_npz:
        mask_ids = normalize_ids(mask_npz["basin_ids"])
        mask_dates = normalize_calendar(mask_npz["dates"])
        order = mask_ids.get_indexer(ids)
        mask = np.asarray(mask_npz["mask"])
        if (min(order) < 0 or not mask_dates.equals(pd.date_range(TEST_START, TEST_END))
                or mask.dtype != np.bool_ or mask.shape != (days, len(ids))):
            raise ValueError("Evaluation_D_common_mask.npz sai lịch, ID hoặc kiểu dữ liệu")
        mask = mask[:, order]
    ref_counts = metrics.loc[metrics.run.eq("Persistence_qmax")].set_index("basin_id").reindex(ids)["n"].to_numpy(dtype=int)
    if (not np.array_equal(mask.sum(axis=0), ref_counts) or len(coverage) != 1
            or int(coverage.iloc[0].common_day_IDs) != int(mask.sum())
            or int(coverage.iloc[0].IDs_with_days) != int(mask.any(axis=0).sum())):
        raise ValueError("Số ngày chấm của Evaluation_D khác mặt nạ chung hoặc bảng coverage")
    contrast_pairs = tuple((f"SF_{conf}_D_seed{seed}", f"SF_{conf}_C2_seed{seed}")
                           for conf in expected_configs for seed in SEEDS)
    d_families = {f"D_{conf}_730_minus_365": tuple(f"SF_{conf}_D_seed{seed}" for seed in SEEDS)
                  for conf in expected_configs}
    d_refs = {f"D_{conf}_730_minus_365": tuple(f"SF_{conf}_C2_seed{seed}" for seed in SEEDS)
              for conf in expected_configs}
    return metrics, events, contrast_pairs, d_families, d_refs, summary, contracts, paths


# %% [markdown]
# ### 2.2. 33 thuộc tính và hai kiểu dữ liệu cần tách
# lc_dom là mã CORINE, mvert_ang là góc hướng trục dài so với Bắc; không xếp hạng
# hai trường này như thuộc tính số liên tục. Giữ số âm khác −999 và GVF ngoài [0,1].
# area_gov của trạm dùng đổi Q; area_calc của lưu vực dùng phân tích thuộc tính.

# %% Phần 2.2 — Đọc thuộc tính thực và ánh xạ A1
# Kiểm 33 trường theo bảng A1; xuất chất lượng từng trường, không điền thiếu.
def load_attributes(raw_root, a1_root, ids):
    mapping_path = unique_file(a1_root, "A1_P3_static_attribute_sources.csv")
    mapping = pd.read_csv(mapping_path)
    if (set(mapping.attribute) != set(STATIC_FIELDS) or mapping.attribute.duplicated().any()
            or not mapping.source.eq("A:Catchment_attributes").all()):
        raise ValueError("Ánh xạ thuộc tính A1 khác hợp đồng 33 trường đã kiểm")
    catchment_path = unique_file(raw_root, "Catchment_attributes.csv",
                                "A_basins_total_upstrm/1_attributes/Catchment_attributes.csv")
    gauge_path = unique_file(raw_root, "Gauge_attributes.csv", "D_gauges/1_attributes/Gauge_attributes.csv")
    catchment, gauge = [read_indexed(p, "ID", ";") for p in (catchment_path, gauge_path)]
    if not set(ids).issubset(catchment.index) or not set(ids).issubset(gauge.index):
        raise ValueError("Thuộc tính thiếu ID trong danh sách đánh giá")
    features = pd.DataFrame(index=ids)
    quality = []
    for name in STATIC_FIELDS:
        values = catchment.loc[ids, name]
        if name == "lc_dom":
            features[name] = values.astype("string").str.strip().replace({"": pd.NA, "-999": pd.NA})
        else:
            features[name] = numeric_column(values, name)
        quality.append(dict(feature=name, type="categorical" if name == "lc_dom" else "circular_angle"
                            if name == "mvert_ang" else "numeric", IDs=len(ids),
                            missing=int(features[name].isna().sum()), unique=int(features[name].nunique()),
                            source="Catchment_attributes.csv", used_in_Spearman=name in NUMERIC_FIELDS))
    features["area_gov"] = numeric_column(gauge.loc[ids, "area_gov"], "area_gov")
    if not np.isfinite(features.area_gov).all() or not features.area_gov.gt(0).all():
        raise ValueError("area_gov phải dương, hữu hạn ở mọi ID")
    for name in ("lon", "lat", "country"):
        if name in gauge:
            features[name] = gauge.loc[ids, name]
    return features, pd.DataFrame(quality), [mapping_path, catchment_path, gauge_path]


# %% [markdown]
# ## Phần 3 — Độ phủ và biến thiên quan trắc từ A3
# ### 3.1. Phân biệt thống kê train và chẩn đoán sau khi đọc test
# Train 2003–2009 chỉ mô tả nhãn/cửa sổ đã có. Test dùng đúng mặt nạ A3 đã chấm.
# Std test liên hệ toán học với mẫu số NSE; không gọi liên hệ này là phát hiện nhân quả.

# %% Phần 3.1 — Đọc NPZ, đổi đơn vị và kiểm lại số ngày/obs_tss
# Tái sử dụng dữ liệu số thô và mặt nạ; không đọc lại 451 CSV hoặc fit scaler mới.
def normalize_calendar(values):
    dates = pd.DatetimeIndex(values).as_unit("ns")
    if (not len(dates) or dates.has_duplicates or dates.hasnans or dates.tz is not None
            or not dates.equals(dates.normalize())
            or not dates.equals(pd.date_range(dates[0], dates[-1], freq="D"))):
        raise ValueError("Cần lịch ngày liên tục, duy nhất, không múi giờ")
    return dates


def read_calendar_archive(path, ids, fields):
    with np.load(path, allow_pickle=False) as data:
        source_ids = normalize_ids(data["basin_ids"])
        order = source_ids.get_indexer(ids)
        dates = normalize_calendar(data["dates"])
        if min(order) < 0 or set(source_ids) != set(ids):
            raise ValueError(f"{path.name}: ID/lịch ngày không hợp lệ")
        arrays = {}
        for field in fields:
            values = np.array(data[field])
            if values.ndim == 1:
                if values.shape != (len(source_ids),):
                    raise ValueError(f"{field}: sai chiều lưu vực")
                arrays[field] = values[order]
            else:
                if values.shape != (len(dates), len(source_ids)):
                    raise ValueError(f"{field}: sai chiều ngày–lưu vực")
                arrays[field] = values[:, order]
    return dates, arrays


def load_diagnostics(a3_root, ids, features, metrics):
    audit_path = unique_file(a3_root, "A3_DLinear_data_audit.npz")
    mask_path = unique_file(a3_root, "A3_final_evaluation_masks.npz")
    loss_path = unique_file(a3_root, "A3_DLinear_loss_stats.csv")
    dates, data = read_calendar_archive(audit_path, ids,
        ["qmax_numeric_m3_s", "area_gov_km2", "complete_past_input"])
    test_dates, saved_masks = read_calendar_archive(mask_path, ids, ["common_source_all"])
    mask, history = saved_masks["common_source_all"], data["complete_past_input"]
    if mask.dtype != np.bool_ or history.dtype != np.bool_:
        raise ValueError("Mặt nạ phải có dtype bool")
    if not test_dates.equals(pd.date_range(TEST_START, TEST_END)):
        raise ValueError("Mặt nạ A3 không phải lịch test 2014–2017")
    if not np.allclose(data["area_gov_km2"], features.area_gov, rtol=1e-12, atol=1e-12):
        raise ValueError("Diện tích NPZ A3 khác Gauge_attributes")
    q = data["qmax_numeric_m3_s"].astype(float)
    if np.isinf(q).any():
        raise ValueError("Q gốc có vô cực")
    negative_n = int((q < 0).sum())
    q[q < 0] = np.nan
    q = q * 86.4 / features.area_gov.to_numpy()[None, :]
    train = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    if int(train.sum()) != len(pd.date_range(TRAIN_START, TRAIN_END)):
        raise ValueError("NPZ không bao phủ đủ lịch train")
    positions = dates.get_indexer(test_dates)
    if min(positions) < 0 or not np.isfinite(q[positions][mask]).all():
        raise ValueError("Mặt nạ test chọn Q thiếu hoặc ngày ngoài dữ liệu")
    saved = read_indexed(loss_path)
    saved = saved.reindex(ids)
    saved["sigma_qmax_train_mm_day"] = numeric_column(saved["sigma_qmax_train_mm_day"], "sigma_qmax_train_mm_day")
    common = metrics.loc[metrics.run.eq("Persistence_qmax")].set_index("basin_id").reindex(ids)
    rows = []
    for j, identifier in enumerate(ids):
        train_q = q[train, j]
        finite = train_q[np.isfinite(train_q)]
        observed = q[positions, j][mask[:, j]]
        n = len(observed)
        sigma = float(finite.std(ddof=0)) if len(finite) >= 2 else np.nan
        test_std = float(observed.std(ddof=0)) if n else np.nan
        sample_n = int((history[train, j] & np.isfinite(train_q)).sum())
        if (len(finite) != int(saved.at[identifier, "train_label_n"])
                or sample_n != int(saved.at[identifier, "train_sample_n"])
                or not np.isclose(sigma, float(saved.at[identifier, "sigma_qmax_train_mm_day"]),
                                  equal_nan=True, rtol=1e-10, atol=1e-12)
                or n != int(common.at[identifier, "n"])
                or not np.isclose(n * test_std ** 2, common.at[identifier, "obs_tss"],
                                  equal_nan=True, rtol=1e-10, atol=1e-12)):
            raise ValueError(f"ID {identifier}: thống kê train hoặc mặt nạ/obs_tss khác A3/Evaluation")
        left, right = train_q[:-1], train_q[1:]
        paired = np.isfinite(left) & np.isfinite(right)
        lag_r = (float(stats.pearsonr(left[paired], right[paired]).statistic)
                 if paired.sum() >= 2 and np.std(left[paired]) > 0 and np.std(right[paired]) > 0 else np.nan)
        mean = float(finite.mean()) if len(finite) else np.nan
        test_mean = float(observed.mean()) if n else np.nan
        rows.append(dict(basin_id=identifier, train_days=int(train.sum()), train_label_n=len(finite),
                         train_sample_n=sample_n, train_missing_fraction=1-len(finite)/int(train.sum()),
                         train_history_complete_days=int(history[train, j].sum()),
                         train_qmax_mean_mm_day=mean, train_qmax_std_mm_day=sigma,
                         train_qmax_CV=ratio(sigma, mean), train_zero_fraction=ratio(int((finite == 0).sum()), len(finite)),
                         train_lag_one_r=lag_r, train_lag_pairs=int(paired.sum()),
                         test_common_days=n, test_common_qmax_mean_mm_day=test_mean,
                         test_common_qmax_std_mm_day=test_std,
                         train_to_test_mean_difference_mm_day=test_mean-mean,
                         train_to_test_std_ratio=ratio(test_std, sigma)))
    return pd.DataFrame(rows).set_index("basin_id"), [audit_path, mask_path, loss_path], negative_n


# %% [markdown]
# ## Phần 4 — Hiệu kỹ năng và biến thiên ba seed
# ### 4.1. Các cặp so sánh cố định
# 12 cặp gốc + 6 cặp A4 + 18 cặp ablation B = 36 cặp (V8); thêm 24 C2−persistence,
# 24 C2−B* cùng seed và 21 C2 lõi khác−LSTM C2 cùng seed = 105 cặp (V9+).
# Khi có phân vùng D, tính riêng 6 hoặc 9 cặp 730−365 cùng cấu hình/seed trên giao chung D.
# Chênh PNSE luôn là PNSE trái trừ PNSE phải; POD T=2 giữ NaN ở lưu vực không có trận thật.

# %% Phần 4.1 — Dựng bảng hiệu trên cùng ID/ngày và giữ đủ lưu vực
# Chênh lệch là mô tả các run đã có; so có/không Q một seed không phải thí nghiệm nhân quả.
def build_contrasts(metrics, events, ids, pairs=None):
    m = metrics.set_index(["run", "basin_id"])
    e = events.set_index(["run", "basin_id"])
    pairs = pairs if pairs is not None else analysis_contrasts(metrics.run)
    rows = []
    for left, right in pairs:
        a, b = [m.loc[name].reindex(ids) for name in (left, right)]
        ea, eb = [e.loc[name].reindex(ids) for name in (left, right)]
        part = pd.DataFrame(index=ids)
        for metric in METRIC_KEYS:
            part[f"left_{metric}"] = a[metric]
            part[f"right_{metric}"] = b[metric]
            part[f"delta_{metric}"] = a[metric] - b[metric]
        for prefix, source in (("left", ea), ("right", eb)):
            for name in ("hit", "missed", "false_alarm", "POD", "FAR"):
                part[f"{prefix}_{name}_T2"] = source[name]
        part["delta_POD_T2"] = ea.POD - eb.POD
        part["contrast"], part["left_run"], part["right_run"] = f"{left}__minus__{right}", left, right
        part["n_days"], part["period"], part["scenario"], part["lead"] = a.n, "test", "S0", 1
        part["scope"] = "common_source_all"
        rows.append(part.reset_index())
    return pd.concat(rows, ignore_index=True)


def seed_spread(contrasts, family="DLinear", families=SEED_FAMILIES, references=FAMILY_REFERENCES):
    names = families[family]
    ref_map = dict(zip(names, references[family]))
    data = contrasts.loc[contrasts.left_run.isin(names)
                         & contrasts.right_run.eq(contrasts.left_run.map(ref_map))]
    if (set(data.left_run) != set(names) or data.duplicated(["basin_id", "left_run"]).any()
            or not data.groupby("basin_id").left_run.nunique().eq(len(SEEDS)).all()):
        raise ValueError(f"{family}: cần đúng ba seed ở từng lưu vực")
    rule = family_reference_rule(family)
    role = ("three_scores_per_basin_not_prediction_ensemble" if rule == "Persistence_qmax"
            else "three_paired_effects_not_prediction_ensemble")
    rows = []
    for identifier, part in data.groupby("basin_id", sort=False):
        for outcome in ENDPOINTS:
            values = part[outcome].to_numpy(dtype=float)
            complete = len(values) == 3 and np.isfinite(values).all()
            rows.append(dict(basin_id=identifier, outcome=outcome, finite_seeds=int(np.isfinite(values).sum()),
                             mean=float(values.mean()) if complete else np.nan,
                             sd_sample=float(values.std(ddof=1)) if complete else np.nan,
                             minimum=float(values.min()) if complete else np.nan,
                             maximum=float(values.max()) if complete else np.nan,
                             reference_rule=rule, role=role))
    return pd.DataFrame(rows)


# %% [markdown]
# ### 4.2. Hướng liên hệ thuộc tính qua ba seed
# Tóm tắt rho của cùng thuộc tính/outcome trên cùng giao chung. DLinear/A4/C2 cơ bản so persistence;
# các họ B, C2_vs_Bstar, C2_vs_LSTM và D ghép cùng seed với mốc tham chiếu tương ứng.
# Ba rho không phải ba lưu vực mới; không gộp p hoặc gọi min–max là khoảng tin cậy.

# %% Phần 4.2 — Tổng hợp độ ổn định, giữ rõ seed thiếu/hệ số không xác định
# Hàm được gọi sau khi tính tương quan tại Phần 5; không chọn seed theo test.
def seed_association_stability(associations, families=SEED_FAMILIES, references=FAMILY_REFERENCES):
    rows = []
    for family in families:
        wanted = list(family_contrast_names(family, families, references))
        data = associations.loc[associations.contrast.isin(wanted)]
        if data.empty:
            continue
        if set(data.contrast) != set(wanted) or data.duplicated(["contrast", "feature", "outcome"]).any():
            raise ValueError(f"{family}: thiếu hoặc trùng kết quả liên hệ theo seed")
        rule = family_reference_rule(family)
        for (feature, outcome), part in data.groupby(["feature", "outcome"], sort=False):
            if set(part.contrast) != set(wanted):
                raise ValueError(f"{family}/{feature}/{outcome}: thiếu hàng của seed")
            part = part.set_index("contrast").reindex(wanted)
            values = part.rho.to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError("rho có vô cực")
            complete = np.isfinite(values).all()
            direction = ("undefined_seed" if not complete else "positive" if (values > 0).all()
                         else "negative" if (values < 0).all() else "all_zero" if (values == 0).all()
                         else "mixed_or_zero")
            rows.append(dict(model_family=family, feature=feature, outcome=outcome,
                reference_rule=rule, finite_seeds=int(np.isfinite(values).sum()),
                n_min=int(part.n.min()), n_max=int(part.n.max()),
                same_ID_count=bool(part.n.nunique() == 1), direction=direction,
                same_nonzero_direction=bool(direction in {"positive", "negative"}) if complete else None,
                rho_mean=float(values.mean()) if complete else np.nan,
                rho_sd_sample=float(values.std(ddof=1)) if complete else np.nan,
                rho_min=float(values.min()) if complete else np.nan,
                rho_max=float(values.max()) if complete else np.nan,
                **{f"rho_seed_{seed}": value for seed, value in zip(SEEDS, values)},
                role="descriptive_seed_stability_no_p_pooling_no_confidence_interval"))
    return pd.DataFrame(rows)


# %% [markdown]
# ## Phần 5 — Spearman, hoán vị ID và Holm
# ### 5.1. Hệ số, số cặp và p có điều kiện trao đổi ID
# Spearman đo quan hệ đơn điệu, dùng thứ hạng trung bình cho các giá trị bằng nhau.
# Với dưới 500 ID, dùng 9.999 hoán vị ID của outcome, giữ thuộc tính cố định.
# P hai phía = (1 + số |rho hoán vị| ≥ |rho thật|)/(B+1), không trả p=0.
# Hoán vị này vẫn giả định ID trao đổi được; lưu vực lồng nhau/gần nhau làm giả định yếu.
# Kết luận chính dựa rho/n; p hỗ trợ thăm dò, không xác định nguyên nhân hoặc feature importance.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

# %% Phần 5.1 — Tính theo nhóm mặt nạ và batch, giữ các ca không xác định
# Mỗi họ Holm gồm 31 thuộc tính của đúng một contrast × outcome; chỉ đếm p hữu hạn.
def holm_finite(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.isinf(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("p phải là vector [0,1] hoặc NaN")
    output = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite], kind="stable")]
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, values[index] * (len(finite)-rank))
        output[index] = min(previous, 1.)
    return output


def rank_associations(features, outcome, permutations, seed, minimum_n=MIN_ASSOCIATION_N, batch_size=128):
    if any(not isinstance(value, int) or isinstance(value, bool) or value < lower
           for value, lower in ((permutations, 0), (batch_size, 1), (minimum_n, 2))):
        raise ValueError("Số hoán vị/batch không hợp lệ")
    y = np.asarray(outcome, dtype=float)
    x = features.to_numpy(dtype=float)
    if y.ndim != 1 or x.shape[0] != len(y) or np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("Sai chiều hoặc có vô cực")
    rows, groups = [], {}
    for j, name in enumerate(features.columns):
        mask = np.isfinite(y) & np.isfinite(x[:, j])
        groups.setdefault(mask.tobytes(), (mask, []))[1].append((j, name))
    for key, (mask, columns) in groups.items():
        n = int(mask.sum())
        names = [name for _, name in columns]
        if n < 2:
            rows.extend(dict(feature=name, n=n, rho=np.nan, p_permutation_iid=np.nan,
                             status="insufficient_pairs") for name in names)
            continue
        xr = stats.rankdata(x[mask][:, [j for j, _ in columns]], axis=0, method="average")
        yr = stats.rankdata(y[mask], method="average")
        xr, yr = xr-xr.mean(axis=0), yr-yr.mean()
        xn, yn = np.sqrt((xr*xr).sum(axis=0)), np.sqrt((yr*yr).sum())
        good = (xn > 0) & (yn > 0)
        correlations = np.full(len(columns), np.nan)
        if good.any():
            normalized = xr[:, good] / xn[good]
            ranked_y = yr / yn
            correlations[good] = np.clip(ranked_y @ normalized, -1., 1.)
        counts = np.zeros(int(good.sum()), dtype=int)
        if good.any() and n >= minimum_n and permutations:
            rng = np.random.default_rng(stable_seed(seed, key.hex()))
            for start in range(0, permutations, batch_size):
                size = min(batch_size, permutations-start)
                order = np.array([rng.permutation(n) for _ in range(size)])
                null = ranked_y[order] @ normalized
                counts += (np.abs(null) >= np.abs(correlations[good])[None, :]-64*np.finfo(float).eps).sum(axis=0)
        p = np.full(len(columns), np.nan)
        if n >= minimum_n and permutations:
            p[good] = (counts+1)/(permutations+1)
        for j, name in enumerate(names):
            status = ("constant_attribute_or_outcome" if not good[j] else "insufficient_pairs_for_test"
                      if n < minimum_n else "rho_only" if not permutations else "computed_iid_permutation")
            rows.append(dict(feature=name, n=n, rho=correlations[j], p_permutation_iid=p[j], status=status))
    return pd.DataFrame(rows).set_index("feature").reindex(features.columns).rename_axis("feature").reset_index()


def association_tables(contrasts, features, permutations=PERMUTATIONS, diagnostic_cols=DIAGNOSTICS, label="Liên hệ thuộc tính"):
    main, diagnostics = [], []
    total = contrasts.contrast.nunique()
    for number, (name, part) in enumerate(contrasts.groupby("contrast", sort=False), 1):
        part = part.set_index("basin_id").reindex(features.index)
        for outcome in ENDPOINTS:
            table = rank_associations(features[list(NUMERIC_FIELDS)], part[outcome], permutations,
                                      stable_seed(PERMUTATION_SEED, name, outcome))
            table["p_holm_iid"] = holm_finite(table.p_permutation_iid)
            table["family_valid_tests"] = int(table.p_permutation_iid.notna().sum())
            table["family_planned_tests"] = len(NUMERIC_FIELDS)
            table["contrast"], table["outcome"] = name, outcome
            table["family_id"] = name+"|"+outcome+"|31_static"
            table["cohort"] = "all_selected_IDs_with_finite_pairs"
            main.append(table)
            diagnostic = rank_associations(features[list(diagnostic_cols)], part[outcome], 0, 0)
            diagnostic["contrast"], diagnostic["outcome"] = name, outcome
            diagnostics.append(diagnostic)
        if number in (1, total) or number % 15 == 0:
            print(f"{label} {number}/{total}: {name}", flush=True)
    return pd.concat(main, ignore_index=True), pd.concat(diagnostics, ignore_index=True)


# %% [markdown]
# ## Phần 6 — Phân nhóm thuộc tính, góc hướng và lớp phủ
# ### 6.1. Biên nhóm chỉ dựa thuộc tính của 451 ID
# Tính các tứ phân vị trước khi ghép outcome; bỏ biên trùng, không chọn điểm cắt theo NSE.
# Hệ số theo lưu vực và POD cộng dồn số trận có trọng số khác nhau; xuất cả mẫu số.

# %% Phần 6.1 — Phân nhóm, mô tả và kiểm độ nhạy theo topology A1
# Cả ID thiếu ngày chấm vẫn có mặt trong ID count; chỉ số không xác định giữ NaN.
def quartile_groups(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    result = np.full(len(array), "missing", dtype=object)
    if not len(finite):
        return result, []
    edges = np.unique(np.quantile(finite, [0, .25, .5, .75, 1]))
    labels = [f"Q{i+1}" for i in range(max(1, len(edges)-1))]
    result[np.isfinite(array)] = np.array(labels)[np.searchsorted(edges[1:-1], finite, side="right")]
    definitions = [dict(group=label, lower=float(edges[min(i, len(edges)-1)]),
                        upper=float(edges[min(i+1, len(edges)-1)]), rule="lower_inclusive_upper_exclusive_last_inclusive")
                   for i, label in enumerate(labels)]
    return result, definitions


def group_summary(contrasts, features):
    rows, bins = [], []
    joined = contrasts.merge(features.reset_index(), on="basin_id", how="left", validate="many_to_one")
    for feature in PRIORITY_FIELDS:
        assigned, definitions = quartile_groups(features[feature])
        labels = pd.Series(assigned, index=features.index)
        bins.extend(dict(feature=feature, **item) for item in definitions)
        joined["attribute_group"] = joined.basin_id.map(labels)
        for contrast, part in joined.groupby("contrast", sort=False):
            for group in [d["group"] for d in definitions]+["missing"]:
                selected = part.loc[part.attribute_group.eq(group)]
                row = dict(feature=feature, group=group, contrast=contrast, IDs=len(selected))
                for outcome in ENDPOINTS:
                    finite = selected[outcome].dropna()
                    row.update({f"{outcome}_n": len(finite), f"{outcome}_median": finite_median(finite),
                                f"{outcome}_P25": float(finite.quantile(.25)) if len(finite) else np.nan,
                                f"{outcome}_P75": float(finite.quantile(.75)) if len(finite) else np.nan})
                for prefix in ("left", "right"):
                    h, m = [int(selected[f"{prefix}_{key}_T2"].sum()) for key in ("hit", "missed")]
                    row.update({f"{prefix}_hit_T2": h, f"{prefix}_observed_T2": h+m,
                                f"{prefix}_POD_T2_pooled": ratio(h, h+m)})
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(bins)


def category_summary(contrasts, features):
    rows = []
    categories = {"lc_dom": features.lc_dom.fillna("missing").astype(str)}
    angle = features.mvert_ang.to_numpy(dtype=float)
    valid = np.isfinite(angle) & (angle >= 0) & (angle <= 360)
    sectors = np.full(len(angle), "missing_or_outside_0_360", dtype=object)
    sectors[valid] = np.array(["0_90_deg", "90_180_deg", "180_270_deg", "270_360_deg"])[
        (np.mod(angle[valid], 360)//90).astype(int)]
    categories["mvert_ang_sector"] = pd.Series(sectors, index=features.index)
    for name, labels in categories.items():
        for contrast, part in contrasts.groupby("contrast", sort=False):
            part = part.set_index("basin_id").reindex(features.index)
            for group in sorted(labels.unique()):
                selected = part.loc[labels.eq(group)]
                for outcome in ENDPOINTS:
                    rows.append(dict(feature=name, group=group, contrast=contrast, outcome=outcome,
                                     IDs=len(selected), finite_IDs=int(selected[outcome].notna().sum()),
                                     median=finite_median(selected[outcome]), role="descriptive_only_no_ordered_correlation"))
    return pd.DataFrame(rows)


def topology_sensitivity(a1_root, contrasts, features):
    path = unique_file(a1_root, "A1_P3_final_selected_topology.csv", optional=True)
    columns = ["feature", "n", "rho", "p_permutation_iid", "status", "contrast", "outcome", "cohort"]
    if path is None:
        return pd.DataFrame(columns=columns), dict(status="not_run_missing_topology_table"), []
    topology = read_indexed(path).reindex(features.index)
    if (topology.path_status.isna().any() or not topology.path_status.eq("reaches_outlet").all()
            or not topology.has_known_selected_upstream.astype(str).str.lower().isin(["true", "false"]).all()):
        raise ValueError("Topology A1 chưa đủ đường tới outlet hoặc cờ upstream chưa hợp lệ")
    headwater = topology.has_known_selected_upstream.astype(str).str.lower().eq("false")
    selected = features.loc[headwater, list(PRIORITY_FIELDS)]
    rows = []
    for name, part in contrasts.groupby("contrast", sort=False):
        part = part.set_index("basin_id").reindex(selected.index)
        for outcome in ENDPOINTS:
            table = rank_associations(selected, part[outcome], 0, 0)
            table["contrast"], table["outcome"], table["cohort"] = name, outcome, "no_known_selected_upstream"
            rows.append(table)
    return pd.concat(rows, ignore_index=True), dict(status="rho_only_not_spatial_independence_proof",
        headwater_IDs=int(headwater.sum()), known_nested_IDs=int((~headwater).sum())), [path]


# %% [markdown]
# ## Phần 7 — Hình kết quả, chú thích và dữ liệu vẽ
# ### 7.1. Giữ các điểm âm, hiện hình trong notebook và in chú thích tiếng Việt
# Hình dùng tiếng Anh; trục symlog giữ cả phần đuôi âm. Không chọn ca theo p hoặc test tốt nhất.

# %% Phần 7.1 — Vẽ hệ số, hiệu kỹ năng và POD cộng dồn
# Chọn trước các cặp hiển thị; mọi cặp và ba seed vẫn có trong CSV đầy đủ.
def save_and_show(fig, path, caption_text, plt):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)
    print(f"Hình: {caption_text}", flush=True)
    return path


def short_label(name):
    text = (name.replace("SeqLSTM_Q_A4_seed", "LSTM A4 ")
                .replace("SeqLSTM_Q_B3_P10S12_seed", "B* ")
                .replace("SeqLSTM_Q_", "LSTM ")
                .replace("DLinear_seed_", "DLinear ")
                .replace("SF_", "")
                .replace("_lr1e-3", "")
                .replace("_lr5e-4", "")
                .replace("_C2_seed", " C2 ")
                .replace("_D_seed", " D ")
                .replace("_seed", " ")
                .replace("Persistence_qmax", "Persistence")
                .replace("__minus__", " minus "))
    return text


def draw_figures(associations, contrasts, grouped, features, output, spreads=None):
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = output / "Figures"
    figure_dir.mkdir(exist_ok=True)
    files = []
    heat = associations.loc[associations.outcome.eq("delta_NSE")].pivot(index="feature", columns="contrast", values="rho")
    order = list(contrasts.contrast.drop_duplicates())
    heat = heat.reindex(index=NUMERIC_FIELDS, columns=order)
    b_order = [f"{a}__minus__{b}" for a, b in B_CONTRASTS if f"{a}__minus__{b}" in order]
    c2_pers_order = [f"{a}__minus__{b}" for a, b in C2_PERSISTENCE_CONTRASTS if f"{a}__minus__{b}" in order]
    c2_bstar_order = [f"{a}__minus__{b}" for a, b in C2_BSTAR_CONTRASTS if f"{a}__minus__{b}" in order]
    c2_core_order = [f"{a}__minus__{b}" for a, b in C2_CORE_CONTRASTS if f"{a}__minus__{b}" in order]
    special = set(b_order) | set(c2_pers_order) | set(c2_bstar_order) | set(c2_core_order)
    heat_groups = [("", [name for name in order if name not in special],
                    "Spearman rho: static attributes vs paired delta NSE",
                    "Ma trận tương quan Spearman giữa 31 thuộc tính tĩnh và hiệu NSE ghép cặp của các mốc nền/A4.")]
    if b_order:
        heat_groups.append(("B_", b_order,
                            "B ablations: static attributes vs paired delta NSE",
                            "Ma trận tương quan Spearman của 18 phép so ablation bước B ghép cùng seed trên toàn bộ lưu vực hữu hạn."))
    if c2_bstar_order:
        heat_groups.append(("C2_vs_Bstar_", c2_bstar_order,
                            "C2 five cores minus B* (same seed): static attributes vs delta NSE",
                            "Ma trận tương quan Spearman giữa 31 thuộc tính tĩnh và hiệu NSE của 24 run C2 trừ B* cùng seed."))
    if c2_core_order:
        heat_groups.append(("C2_vs_LSTM_", c2_core_order,
                            "C2 non-LSTM cores minus C2 LSTM L2 x1 (same seed): static attributes vs delta NSE",
                            "Ma trận tương quan Spearman giữa 31 thuộc tính tĩnh và hiệu NSE của bảy cấu hình C2 lõi khác trừ LSTM L2 x1 cùng seed."))
    for prefix, columns, title, vi_caption in heat_groups:
        fig, ax = plt.subplots(figsize=(max(16, len(columns)*0.95), 12))
        im = ax.imshow(heat[columns], cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(columns)), [short_label(name) for name in columns], rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(len(NUMERIC_FIELDS)), NUMERIC_FIELDS, fontsize=9)
        fig.suptitle(title+" | all finite basins")
        fig.colorbar(im, ax=ax, label="Spearman rho", shrink=.7)
        fig.tight_layout()
        files.append(save_and_show(fig, figure_dir / f"BasinAnalysis_{prefix}static_delta_NSE_rho.png", vi_caption, plt))
    fixed = [("SeqLSTM_Q", "Persistence_qmax"), ("SeqLSTM_Q", "SeqLSTM_noQ"),
             ("DLinear_seed_111", "Persistence_qmax")]
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), squeeze=False)
    fig.suptitle("Paired delta NSE vs catchment attributes | fixed illustrative contrasts")
    for row, (left, right) in enumerate(fixed):
        part = contrasts.loc[contrasts.left_run.eq(left) & contrasts.right_run.eq(right)].set_index("basin_id").reindex(features.index)
        for col, feature in enumerate(PRIORITY_FIELDS):
            x, y = features[feature], part.delta_NSE
            valid = np.isfinite(x) & np.isfinite(y)
            ax = axes[row, col]
            ax.scatter(x[valid], y[valid], s=12, alpha=.55)
            ax.axhline(0, color="gray", lw=.8)
            ax.set_yscale("symlog", linthresh=.1)
            if feature == "area_calc" and (x[valid] > 0).all():
                ax.set_xscale("log")
            panel = chr(ord("a") + row*4 + col)
            ax.set(xlabel=FEATURE_LABELS[feature], ylabel="Paired delta NSE",
                   title=f"({panel}) {short_label(left)} minus {short_label(right)}\nn={int(valid.sum())}")
            ax.grid(alpha=.15)
    fig.tight_layout()
    files.append(save_and_show(
        fig, figure_dir / "BasinAnalysis_priority_scatter.png",
        "Phân tán hiệu NSE theo bốn thuộc tính ưu tiên (diện tích, độ cao, tỷ phần tuyết, khô hạn); giữ mọi điểm âm bằng trục symlog.",
        plt,
    ))
    panels = [("Author and DLinear", ("SeqLSTM_Q", "EncDec_Q", *SEED_FAMILIES["DLinear"]))]
    if set(A4_MODELS).issubset(set(contrasts.left_run)):
        panels.append(("Author and LSTM A4", ("SeqLSTM_Q", *A4_MODELS)))
    styles = {"SeqLSTM_Q": ("#0072B2", "-"), "EncDec_Q": ("#009E73", "-")}
    for family, color in (("DLinear", "#D55E00"), ("LSTM_A4", "#0072B2")):
        styles.update({name: (color, style) for name, style in zip(SEED_FAMILIES[family], ("--", "-.", ":"))})
    for endpoint, ylabel, filename, vi_desc in (
        ("delta_NSE_median", "Median paired delta NSE", "quartile_delta_NSE",
         "Trung vị hiệu NSE so với persistence theo bốn nhóm tứ phân vị cố định của từng thuộc tính ưu tiên."),
        ("left_POD_T2_pooled", "Pooled POD at T=2 years", "quartile_flood_POD",
         "POD cộng dồn ở ngưỡng lũ hai năm (USWRC, cùng ngày bắt đầu) theo cùng biên tứ phân vị thuộc tính."),
    ):
        fig, axes = plt.subplots(len(panels), 4, figsize=(17, 4.6*len(panels)), squeeze=False, sharey=True)
        fig.suptitle(f"{ylabel} by catchment attribute | same quartile boundaries and daily mask")
        for row, (family_label, display_runs) in enumerate(panels):
            for col, (ax, feature) in enumerate(zip(axes[row], PRIORITY_FIELDS)):
                for run in display_runs:
                    name = f"{run}__minus__Persistence_qmax"
                    part = grouped.loc[grouped.feature.eq(feature) & grouped.contrast.eq(name) & grouped.group.ne("missing")]
                    color, style = styles[run]
                    ax.plot(part.group, part[endpoint], marker="o", ms=4, label=short_label(run), color=color, linestyle=style)
                panel = chr(ord("a") + row*4 + col)
                ax.set(title=f"({panel}) {FEATURE_LABELS[feature]}", xlabel="Attribute quartile (ties may reduce bins)",
                       ylabel=f"{family_label}\n{ylabel}")
                if endpoint == "delta_NSE_median":
                    ax.axhline(0, color="gray", lw=.8)
                ax.grid(alpha=.2)
            axes[row, -1].legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
        fig.tight_layout()
        files.append(save_and_show(fig, figure_dir / f"BasinAnalysis_{filename}.png", vi_desc, plt))
    b_families = [family for family in SEED_FAMILIES if family.startswith("LSTM_B")
                  and set(family_contrast_names(family)).issubset(set(contrasts.contrast))]
    for block, families in enumerate((b_families[:3], b_families[3:]), 1):
        if not families:
            continue
        for endpoint, ylabel, filename in (("delta_NSE_median", "Median paired delta NSE", "delta_NSE"),
                                           ("left_POD_T2_pooled", "Pooled POD at T=2 years", "flood_POD")):
            fig, axes = plt.subplots(len(families), 4, figsize=(17, 4.3*len(families)), squeeze=False, sharey=True)
            fig.suptitle(f"B ablations by catchment attribute | {ylabel} | fixed same-seed comparisons")
            for row, family in enumerate(families):
                for col, (ax, feature) in enumerate(zip(axes[row], PRIORITY_FIELDS)):
                    for seed, name, style in zip(SEEDS, family_contrast_names(family), ("--", "-.", ":")):
                        part = grouped.loc[grouped.feature.eq(feature) & grouped.contrast.eq(name) & grouped.group.ne("missing")]
                        ax.plot(part.group, part[endpoint], marker="o", ms=4, label=f"Seed {seed}", linestyle=style)
                    panel = chr(ord("a")+row*4+col)
                    left = family.removeprefix("LSTM_")
                    reference = B_PREVIOUS[left]
                    contrast_label = f"{left} minus {reference}" if endpoint == "delta_NSE_median" else f"{left} (reference {reference})"
                    ax.set(title=f"({panel}) {FEATURE_LABELS[feature]}", xlabel="Attribute quartile",
                           ylabel=f"{contrast_label}\n{ylabel}")
                    if endpoint == "delta_NSE_median":
                        ax.axhline(0, color="gray", lw=.8)
                    ax.grid(alpha=.2)
                axes[row, -1].legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
            fig.tight_layout()
            files.append(save_and_show(
                fig, figure_dir / f"BasinAnalysis_B_block{block}_quartile_{filename}.png",
                f"Nhóm {block} gồm {len(families)} bước B × {len(PRIORITY_FIELDS)} thuộc tính; mỗi đường một seed, cùng biên tứ phân vị.",
                plt,
            ))
    if c2_core_order:
        core_families = [f"C2_{conf}_vs_LSTM" for conf in C2_NON_LSTM_CONFIGS]
        core_colors = {"gru": "#009E73", "transformer": "#E69F00", "s4d": "#CC79A7", "mamba": "#D55E00"}
        for block, families in enumerate((core_families[:4], core_families[4:]), 1):
            for endpoint, ylabel, filename in (("delta_NSE_median", "Median paired delta NSE", "delta_NSE"),
                                               ("left_POD_T2_pooled", "Pooled POD at T=2 years", "flood_POD")):
                fig, axes = plt.subplots(len(families), 4, figsize=(17, 4.1*len(families)), squeeze=False, sharey=True)
                fig.suptitle(f"C2 cores vs C2 LSTM L2 x1 by catchment attribute | {ylabel} | same-seed pairs")
                for row, family in enumerate(families):
                    conf = family.removeprefix("C2_").removesuffix("_vs_LSTM")
                    core = C2_CONFIG_SPECS[conf][0]
                    for col, (ax, feature) in enumerate(zip(axes[row], PRIORITY_FIELDS)):
                        for seed, name, style in zip(SEEDS, family_contrast_names(family), ("-", "--", ":")):
                            part = grouped.loc[grouped.feature.eq(feature) & grouped.contrast.eq(name) & grouped.group.ne("missing")]
                            ax.plot(part.group, part[endpoint], marker="o", ms=4, label=f"Seed {seed}",
                                    color=core_colors[core], linestyle=style)
                        panel = chr(ord("a")+row*4+col)
                        label_left = conf.replace("_lr1e-3", "").replace("_lr5e-4", "")
                        contrast_label = (f"{label_left} minus lstm_L2_x1" if endpoint == "delta_NSE_median"
                                          else f"{label_left} (ref lstm_L2_x1)")
                        ax.set(title=f"({panel}) {FEATURE_LABELS[feature]}", xlabel="Attribute quartile",
                               ylabel=f"{contrast_label}\n{ylabel}")
                        if endpoint == "delta_NSE_median":
                            ax.axhline(0, color="gray", lw=.8)
                        ax.grid(alpha=.2)
                    axes[row, -1].legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
                fig.tight_layout()
                files.append(save_and_show(
                    fig, figure_dir / f"BasinAnalysis_C2_block{block}_quartile_{filename}.png",
                    f"Nhóm {block} các cấu hình C2 so với LSTM L2 x1 cùng seed theo bốn tứ phân vị thuộc tính ưu tiên ({ylabel}).",
                    plt,
                ))
    if spreads:
        primary_spreads = {k: v for k, v in spreads.items()
                           if not k.endswith("_vs_Bstar") and not k.endswith("_vs_LSTM")}
        fig, axes = plt.subplots(1, 3, figsize=(max(18, len(primary_spreads)*1.35), 5.8), squeeze=False)
        fig.suptitle("Variation across three seeds per basin | family-specific paired effects")
        def compact_family_label(fam):
            text = fam.removeprefix("LSTM_").replace("_lr1e-3", "").replace("_lr5e-4", "")
            text = (text.replace("C2_transformer_", "C2 Trans ")
                        .replace("C2_mamba_", "C2 Mamba ")
                        .replace("C2_s4d_", "C2 S4D ")
                        .replace("C2_lstm_", "C2 LSTM ")
                        .replace("C2_gru_", "C2 GRU ")
                        .replace("_L", " L").replace("_x", "x"))
            return text
        for col, (ax, outcome) in enumerate(zip(axes[0], ENDPOINTS)):
            values, labels, counts = [], [], []
            for family, table in primary_spreads.items():
                data = table.loc[table.outcome.eq(outcome), "sd_sample"].dropna().to_numpy()
                if len(data):
                    values.append(data)
                    counts.append(len(data))
                    labels.append(compact_family_label(family))
            uniform_n = len(set(counts)) == 1
            if not uniform_n:
                labels = [f"{lbl} (n={cnt})" for lbl, cnt in zip(labels, counts)]
            if values:
                ax.boxplot(values, showfliers=True)
                ax.set_xticks(range(1, len(labels)+1), labels, rotation=55, ha="right", fontsize=8)
            ax.set_yscale("symlog", linthresh=.001)
            ax.set_ylim(bottom=0)
            n_note = f" (n={counts[0]})" if uniform_n and counts else ""
            ax.set(title=f"({chr(ord('a')+col)}) {outcome}{n_note}", ylabel="Sample SD across three paired effects")
            ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        files.append(save_and_show(
            fig, figure_dir / "BasinAnalysis_seed_spread.png",
            f"SD mẫu qua ba seed tại từng lưu vực cho {len(primary_spreads)} họ mô hình (DLinear/A4/C2 so persistence; bước B so bước trước cùng seed).",
            plt,
        ))
    return files


def draw_d_figures(associations, contrasts, grouped, spreads, d_families, d_refs, output):
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = output / "Figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    files = []
    order = list(contrasts.contrast.drop_duplicates())
    heat = (associations.loc[associations.outcome.eq("delta_NSE")]
            .pivot(index="feature", columns="contrast", values="rho")
            .reindex(index=NUMERIC_FIELDS, columns=order))
    fig, ax = plt.subplots(figsize=(max(12, len(order)*1.25), 11))
    im = ax.imshow(heat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)), [short_label(name) for name in order], rotation=55, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(NUMERIC_FIELDS)), NUMERIC_FIELDS, fontsize=9)
    fig.suptitle("Step D (730 minus 365 steps, same seed): Spearman rho with static attributes | all finite basins")
    fig.colorbar(im, ax=ax, label="Spearman rho", shrink=.7)
    fig.tight_layout()
    files.append(save_and_show(
        fig, figure_dir / "BasinAnalysis_D_static_delta_NSE_rho.png",
        f"Ma trận tương quan Spearman giữa 31 thuộc tính tĩnh và hiệu NSE (730 trừ 365 bước) trên {len(order)} cặp cấu hình–seed của bước D.",
        plt,
    ))
    core_colors = {"lstm": "#0072B2", "mamba": "#D55E00", "s4d": "#CC79A7"}
    families = list(d_families)
    for endpoint, ylabel, filename, vi_desc in (
        ("delta_NSE_median", "Median paired delta NSE (730 minus 365)", "quartile_delta_NSE",
         "Trung vị hiệu NSE (730 trừ 365 bước cùng seed) theo bốn nhóm tứ phân vị cố định của diện tích, độ cao, tuyết và khô hạn."),
        ("left_POD_T2_pooled", "Pooled POD at T=2 years (730 steps)", "quartile_flood_POD",
         "POD cộng dồn ở ngưỡng lũ hai năm của cấu hình 730 bước theo bốn nhóm tứ phân vị thuộc tính ưu tiên."),
    ):
        fig, axes = plt.subplots(len(families), 4, figsize=(17, 4.2*len(families)), squeeze=False, sharey=True)
        fig.suptitle(f"Step D length effect by catchment attribute | {ylabel}")
        for row, family in enumerate(families):
            conf = family.removeprefix("D_").removesuffix("_730_minus_365")
            core = C2_CONFIG_SPECS[conf][0]
            names = family_contrast_names(family, d_families, d_refs)
            for col, (ax, feature) in enumerate(zip(axes[row], PRIORITY_FIELDS)):
                for seed, cname, style in zip(SEEDS, names, ("-", "--", ":")):
                    part = grouped.loc[grouped.feature.eq(feature) & grouped.contrast.eq(cname) & grouped.group.ne("missing")]
                    ax.plot(part.group, part[endpoint], marker="o", ms=4, label=f"Seed {seed}",
                            color=core_colors[core], linestyle=style)
                panel = chr(ord("a")+row*4+col)
                short_conf = conf.replace("_lr1e-3", "")
                ax.set(title=f"({panel}) {FEATURE_LABELS[feature]}", xlabel="Attribute quartile",
                       ylabel=f"{short_conf} (730 vs 365)\n{ylabel}")
                if endpoint == "delta_NSE_median":
                    ax.axhline(0, color="gray", lw=.8)
                ax.grid(alpha=.2)
            axes[row, -1].legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
        fig.tight_layout()
        files.append(save_and_show(fig, figure_dir / f"BasinAnalysis_D_{filename}.png", vi_desc, plt))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), squeeze=False)
    fig.suptitle("Step D: variation of paired (730 minus 365) effects across three seeds per basin")
    for col, (ax, outcome) in enumerate(zip(axes[0], ENDPOINTS)):
        values, labels = [], []
        for family, table in spreads.items():
            data = table.loc[table.outcome.eq(outcome), "sd_sample"].dropna().to_numpy()
            if len(data):
                values.append(data)
                short_fam = family.removeprefix("D_").removesuffix("_730_minus_365").replace("_lr1e-3", "")
                labels.append(f"{short_fam}\nn={len(data)}")
        if values:
            ax.boxplot(values, showfliers=True)
            ax.set_xticks(range(1, len(labels)+1), labels)
        ax.set_yscale("symlog", linthresh=.001)
        ax.set_ylim(bottom=0)
        ax.set(title=f"({chr(ord('a')+col)}) {outcome}", ylabel="Sample SD across three paired length effects")
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    files.append(save_and_show(
        fig, figure_dir / "BasinAnalysis_D_seed_spread.png",
        f"SD mẫu qua ba seed của hiệu độ dài chuỗi (730 trừ 365 bước) tại từng lưu vực cho {len(spreads)} cấu hình bước D.",
        plt,
    ))
    return files


def run_d_subanalysis(d_root, output, ids, features, a1_root, permutations=PERMUTATIONS):
    d_out = output / "D"
    d_out.mkdir(parents=True, exist_ok=True)
    (d_out / "BasinAnalysis_D_summary.json").write_text(json.dumps(dict(
        complete=False, status="started_not_verified", output=str(d_out)), indent=2), encoding="utf-8")
    d_metrics, d_events, d_pairs, d_families, d_refs, d_eval_summary, d_contracts, d_paths = load_d_evaluation(d_root, ids)
    d_features = features.copy()
    d_common = d_metrics.loc[d_metrics.run.eq("Persistence_qmax")].set_index("basin_id").reindex(ids)
    d_features["d_test_common_days"] = d_common["n"].astype(int)
    d_features["d_test_common_qmax_std_mm_day"] = np.sqrt(
        np.divide(d_common["obs_tss"].to_numpy(dtype=float), d_common["n"].to_numpy(dtype=float),
                  out=np.full(len(ids), np.nan), where=d_common["n"].to_numpy(dtype=float) > 0)
    )
    d_diag_cols = ("train_missing_fraction", "train_qmax_std_mm_day", "train_sample_n",
                   "d_test_common_qmax_std_mm_day", "d_test_common_days")
    contrasts = build_contrasts(d_metrics, d_events, ids, pairs=d_pairs)
    associations, diag_assoc = association_tables(contrasts, d_features, permutations,
                                                  diagnostic_cols=d_diag_cols, label="Liên hệ thuộc tính D")
    stability = seed_association_stability(associations, d_families, d_refs)
    spreads = {fam: seed_spread(contrasts, fam, d_families, d_refs) for fam in d_families}
    grouped, bins = group_summary(contrasts, d_features)
    categories = category_summary(contrasts, d_features)
    topology, topology_status, used = topology_sensitivity(a1_root, contrasts, d_features)
    d_paths.extend(used)
    tables = dict(
        paired_contrasts=contrasts, static_associations=associations,
        diagnostic_associations=diag_assoc, quartile_definitions=bins,
        grouped_skill=grouped, categorical_summary=categories,
        topology_sensitivity=topology, seed_association_stability=stability,
        contract_check=d_contracts,
        **{f"{fam}_seed_spread": tbl for fam, tbl in spreads.items()},
    )
    written = []
    for name, table in tables.items():
        path = d_out / f"BasinAnalysis_D_{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    figures = draw_d_figures(associations, contrasts, grouped, spreads, d_families, d_refs, d_out)
    written.extend(figures)
    protocol = dict(
        analysis_protocol_version=4, step="D", scope="test/S0/lead1/common_source_all_D_partition",
        expected_IDs=len(ids), configs=list(d_eval_summary["configs"]), S4D_included=bool(d_eval_summary["S4D_included"]),
        all_pairs_strictly_equivalent=bool(d_eval_summary.get("all_pairs_strictly_equivalent", True)),
        grad_guard_runs_with_skips=list(d_eval_summary.get("grad_guard_runs_with_skips") or []),
        contrasts=[dict(left=a, right=b) for a, b in d_pairs],
        seed_families={k: list(v) for k, v in d_families.items()},
        family_references={k: list(v) for k, v in d_refs.items()},
        static_predictors=list(NUMERIC_FIELDS), priority_attributes=list(PRIORITY_FIELDS),
        outcomes=list(ENDPOINTS), permutations=permutations, seed=PERMUTATION_SEED,
        holm_family="31_static_features_per_D_contrast_and_outcome_finite_p_only",
        test_used_for_selection=False,
    )
    (d_out / "BasinAnalysis_D_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(
        complete=True, analysis_protocol_version=4, step="D", IDs=len(ids),
        IDs_with_common_days=int((d_features.d_test_common_days > 0).sum()),
        common_day_IDs=int(d_features.d_test_common_days.sum()),
        configs=list(d_eval_summary["configs"]), S4D_included=bool(d_eval_summary["S4D_included"]),
        all_pairs_strictly_equivalent=bool(d_eval_summary.get("all_pairs_strictly_equivalent", True)),
        grad_guard_runs_with_skips=list(d_eval_summary.get("grad_guard_runs_with_skips") or []),
        contrasts=len(d_pairs), association_rows=len(associations),
        finite_rho=int(associations.rho.notna().sum()),
        seed_stability_rows=len(stability), topology=topology_status,
        figures=len(figures), output=str(d_out),
    )
    if not all(p.is_file() and p.stat().st_size > 0 for p in written):
        raise RuntimeError("Thiếu output phân vùng D sau khi ghi")
    (d_out / "BasinAnalysis_D_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    written.extend([d_out / "BasinAnalysis_D_protocol.json", d_out / "BasinAnalysis_D_summary.json"])
    return summary, written, d_paths


# %% [markdown]
# ## Phần 8 — Kiểm thử độc lập
# ### 8.1. Ví dụ số và ca suy biến, không phải kết quả LamaH-CE
# Kiểm thứ hạng, phép ghép ID, p hoán vị, NaN, bin có ties, nhóm không có trận và đủ bốn hồ sơ 11/14/32/56 run + D.

# %% Phần 8.1 — Kiểm trước khi đọc dữ liệu thực nghiệm
# Chạy --self-test không cần Input; mọi số liệu nhỏ ở đây chỉ là fixture.
def self_test():
    rows = []
    def check(name, value):
        rows.append(dict(case=name, passed=bool(value)))
        if not value:
            raise AssertionError(name)
    def rejects(name, action):
        try:
            action()
        except ValueError:
            check(name, True)
        else:
            check(name, False)
    check("33_fields_31_rank_predictors", len(STATIC_FIELDS) == 33 and len(NUMERIC_FIELDS) == 31
          and not {"lc_dom", "mvert_ang"} & set(NUMERIC_FIELDS))
    check("12_contrasts_three_endpoints", len(CONTRASTS) == 12 and len(ENDPOINTS) == 3)
    baseline_runs = (*MODELS, "Persistence_qmax", "Persistence_qmean_aux")
    check("legacy_11_run_plan", analysis_contrasts(baseline_runs) == CONTRASTS)
    check("A4_14_run_18_contrasts", len(analysis_contrasts((*baseline_runs, *A4_MODELS))) == 18)
    rejects("partial_A4_seeds_rejected", lambda: analysis_contrasts((*baseline_runs, A4_MODELS[0])))
    rejects("unknown_model_rejected", lambda: analysis_contrasts((*baseline_runs, "new_model")))
    check("normalize_leading_zero", list(normalize_ids(["002", "03"])) == ["2", "3"])
    check("NPZ_calendar_day_resolution_normalized", normalize_calendar(np.array(
        ["2014-01-01", "2014-01-02"], dtype="datetime64[D]")).equals(pd.date_range("2014-01-01", periods=2)))
    rejects("gapped_calendar_rejected", lambda: normalize_calendar(["2014-01-01", "2014-01-03"]))
    rejects("duplicate_IDs_rejected", lambda: normalize_ids(["2", "002"]))
    rejects("decimal_IDs_rejected", lambda: normalize_ids(["2.0"]))
    values = numeric_column(pd.Series(["0", "-999", "", "-2", "1.2"]), "fixture")
    check("keep_zero_and_non_sentinel_negative", values.iloc[0] == 0 and values.iloc[3] == -2 and values.isna().sum() == 2)
    rejects("unexpected_numeric_token_rejected", lambda: numeric_column(pd.Series(["oops"]), "fixture"))
    rejects("infinite_attribute_rejected", lambda: numeric_column(pd.Series(["inf"]), "fixture"))
    x = pd.DataFrame({"up": np.arange(12.), "down": -np.arange(12.), "ties": np.repeat(np.arange(6.), 2), "constant": 1.})
    y = np.arange(12.)
    rejects("boolean_permutation_count_rejected", lambda: rank_associations(x, y, True, 1))
    rejects("fractional_batch_rejected", lambda: rank_associations(x, y, 9, 1, batch_size=1.5))
    rejects("invalid_minimum_n_rejected", lambda: rank_associations(x, y, 9, 1, minimum_n=1))
    actual = rank_associations(x, y, 99, 123)
    for name in ("up", "down", "ties"):
        row = actual.set_index("feature").loc[name]
        check("Spearman_reference_"+name, np.isclose(row.rho, stats.spearmanr(x[name], y).statistic))
    check("permutation_p_never_zero", actual.p_permutation_iid.dropna().between(1/100, 1).all())
    check("permutation_deterministic", actual.equals(rank_associations(x, y, 99, 123)))
    check("constant_predictor_undefined", np.isnan(actual.set_index("feature").at["constant", "rho"]))
    check("constant_outcome_undefined", rank_associations(x, np.ones(12), 99, 1).rho.isna().all())
    missing = x.copy()
    missing.loc[0, "up"] = np.nan
    check("finite_pair_mask_recorded", rank_associations(missing, y, 0, 1).set_index("feature").at["up", "n"] == 11)
    check("short_sample_no_p", rank_associations(x.iloc[:4], y[:4], 99, 1).p_permutation_iid.isna().all())
    check("rho_only_diagnostics_no_p", rank_associations(x, y, 0, 1).p_permutation_iid.isna().all())
    check("Holm_finite_only", np.allclose(holm_finite([.01, np.nan, .03]), [.02, np.nan, .03], equal_nan=True))
    check("Holm_empty_all_missing", len(holm_finite([])) == 0 and np.isnan(holm_finite([np.nan])).all())
    labels, definitions = quartile_groups([0., 0., 0., 0., 1., 1., 1., np.nan])
    check("quartile_ties_no_outcome_cut", len(definitions) <= 4 and labels[-1] == "missing" and labels[0] == labels[1])
    labels, definitions = quartile_groups([3., 3., np.nan])
    check("constant_attribute_single_bin", len(definitions) == 1 and list(labels) == ["Q1", "Q1", "missing"])
    check("no_events_POD_remains_undefined", np.isnan(ratio(0, 0)))
    check("negative_NSE_delta_retained", finite_median([-100., -2., 1.]) == -2)
    check("seed_stability_not_ensemble", seed_spread(pd.DataFrame([
        dict(basin_id="1", left_run=f"DLinear_seed_{seed}", right_run="Persistence_qmax",
             delta_NSE=i, delta_PNSE=i, delta_POD_T2=i) for i, seed in enumerate((111, 222, 333))
    ])).sd_sample.eq(1).all())
    a4_fixture = pd.DataFrame([
        dict(basin_id="1", left_run=name, right_run="Persistence_qmax",
             delta_NSE=value, delta_PNSE=value, delta_POD_T2=value)
        for name, value in zip(A4_MODELS, (0., 1., 2.))])
    check("A4_seed_spread_sample_SD", seed_spread(a4_fixture, "LSTM_A4").sd_sample.eq(1).all())
    a4_fixture.loc[0, "delta_POD_T2"] = np.nan
    check("undefined_seed_not_averaged", np.isnan(seed_spread(a4_fixture, "LSTM_A4").set_index(
        "outcome").at["delta_POD_T2", "sd_sample"]))
    rejects("missing_seed_score_row_rejected", lambda: seed_spread(a4_fixture.iloc[:2], "LSTM_A4"))
    association_fixture = pd.DataFrame([
        dict(contrast=f"{name}__minus__Persistence_qmax", feature="fixture", outcome="delta_NSE", n=12, rho=value)
        for name, value in zip(A4_MODELS, (.1, .3, .5))])
    stability = seed_association_stability(association_fixture).iloc[0]
    check("rho_seed_range_not_CI", stability.direction == "positive" and np.isclose(stability.rho_min, .1)
          and np.isclose(stability.rho_max, .5) and np.isclose(stability.rho_sd_sample, .2))
    association_fixture.loc[0, "rho"] = -.1
    check("mixed_rho_direction_recorded", seed_association_stability(association_fixture).iloc[0].direction == "mixed_or_zero")
    association_fixture.loc[0, "rho"] = np.nan
    stability = seed_association_stability(association_fixture).iloc[0]
    check("undefined_rho_seed_recorded", stability.direction == "undefined_seed" and np.isnan(stability.rho_mean))
    rejects("duplicate_seed_association_rejected", lambda: seed_association_stability(pd.concat(
        [association_fixture, association_fixture.iloc[:1]], ignore_index=True)))
    full_b_runs = (*baseline_runs, *A4_MODELS, *(name for names in B_MODELS.values() for name in names))
    full_c2_runs = (*full_b_runs, *C2_ALL_RUNS)
    check("B_32_run_36_contrasts", len(analysis_contrasts(full_b_runs)) == 36)
    check("C2_56_run_105_contrasts", len(analysis_contrasts(full_c2_runs)) == 105)
    check("B_18_sequential_contrasts_same_seed", len(B_CONTRASTS) == 18 and all(
        left.rsplit("_seed", 1)[1] == right.rsplit("_seed", 1)[1] for left, right in B_CONTRASTS))
    check("C2_same_seed_Bstar_and_LSTM_contrasts", len(C2_BSTAR_CONTRASTS) == 24 and len(C2_CORE_CONTRASTS) == 21
          and all(l.rsplit("_seed", 1)[1] == r.rsplit("_seed", 1)[1] for l, r in C2_BSTAR_CONTRASTS + C2_CORE_CONTRASTS))
    check("B_contrasts_predeclared_steps", set(B_CONTRASTS) == {
        (f"SeqLSTM_Q_{step}_seed{seed}", f"SeqLSTM_Q_{previous}_seed{seed}")
        for step, previous in B_PREVIOUS.items() for seed in SEEDS})
    rejects("partial_B_group_rejected", lambda: analysis_contrasts(full_b_runs[:-1]))
    rejects("partial_C2_group_rejected", lambda: analysis_contrasts(full_c2_runs[:-1]))
    rejects("intermediate_B_only_profile_rejected", lambda: analysis_contrasts((*baseline_runs, *A4_MODELS, *B_MODELS["B0"])))
    check("B_8_complete_seed_families", len(active_seed_families(full_b_runs)) == 8)
    check("C2_31_complete_seed_families", len(active_seed_families(full_c2_runs)) == 31)
    paired = pd.DataFrame([dict(basin_id="1", left_run=left, right_run=right,
        delta_NSE=i, delta_PNSE=i, delta_POD_T2=i)
        for i, (left, right) in enumerate(zip(SEED_FAMILIES["LSTM_B3_P10S12"], FAMILY_REFERENCES["LSTM_B3_P10S12"]))])
    spread = seed_spread(paired, "LSTM_B3_P10S12")
    check("B_seed_SD_of_paired_effects_labeled", spread.sd_sample.eq(1).all()
          and spread.reference_rule.eq("same_seed_previous_ablation_step").all()
          and spread.role.eq("three_paired_effects_not_prediction_ensemble").all())
    rejects("B_seed_cross_reference_rejected", lambda: seed_spread(
        paired.assign(right_run="Persistence_qmax"), "LSTM_B3_P10S12"))
    association = pd.DataFrame([dict(contrast=name, feature="fixture", outcome="delta_NSE", n=12, rho=value)
        for name, value in zip(family_contrast_names("LSTM_B3_P10S12"), (.1, .2, .3))])
    stability = seed_association_stability(association).iloc[0]
    check("B_rho_stability_uses_same_seed_step_reference", stability.model_family == "LSTM_B3_P10S12"
          and stability.reference_rule == "same_seed_previous_ablation_step" and np.isclose(stability.rho_sd_sample, .1))
    import tempfile
    with tempfile.TemporaryDirectory(prefix="bybasin_contract_fixture_") as directory:
        root = Path(directory)
        def populate_eval_fixture(target_dir, include_c2=False):
            target_dir.mkdir(parents=True, exist_ok=True)
            run_list = full_c2_runs if include_c2 else full_b_runs
            records, metric_rows, event_rows = [], [], []
            for name in run_list:
                seed = next((seed for seed in SEEDS if name.endswith(f"_seed{seed}")), None)
                records.append(dict(run=name, seed=seed, IDs=2, days=1461, period="test", scenario="S0", lead=1,
                                    source_metadata={}))
                for identifier in ("1", "2"):
                    valid = identifier == "1"
                    metric_rows.append(dict(run=name, basin_id=identifier, seed=seed, period="test", scenario="S0", lead=1,
                        scope="common_source_all", n=2 if valid else 0, obs_tss=1. if valid else np.nan,
                        **{metric: .5 if valid else np.nan for metric in METRIC_KEYS}))
                    event_rows.append(dict(run=name, basin_id=identifier, seed=seed, period="test", scenario="S0", lead=1,
                        scope="common_source_all", grouping="USWRC", return_period_years=2, matching_tolerance_days=0,
                        hit=int(valid), missed=int(valid), false_alarm=int(valid), POD=.5 if valid else np.nan,
                        FAR=.5 if valid else np.nan))
            pd.DataFrame(metric_rows).to_csv(target_dir/"Evaluation_metrics.csv", index=False)
            pd.DataFrame(event_rows).to_csv(target_dir/"Evaluation_events.csv", index=False)
            summary = dict(complete=True, A3_reference_matched=True, coverage=[dict(
                period="test", scenario="S0", lead=1, runs=len(run_list), common_day_IDs=2, IDs_with_days=1)],
                B3_runs_by_mask={mask:list(B_MODELS[f"B3_{mask}"]) for mask in ("P10S12", "P05S05")})
            protocol = dict(protocol_version=3, event_tolerance_days_by_lead={"1": 0})
            for step, names in (("A4", A4_MODELS), ("B0", B_MODELS["B0"]), ("B1a", B_MODELS["B1a"]),
                                ("B1b", B_MODELS["B1b"]), ("B2", B_MODELS["B2"]),
                                ("B3", B_MODELS["B3_P10S12"]+B_MODELS["B3_P05S05"])):
                common_step = "B1" if step in ("B1a", "B1b") else step
                summary[common_step+"_contracts_matched"] = True
                summary[step+"_runs"] = list(names)
                summary[common_step+"_compare_version_expected"] = 2 if step == "A4" else 1
                contracts, inputs = [], [dict(file=name, expected_sha256=digit*64, observed_sha256=digit*64, passed=True)
                                        for name, digit in (("A3_persistence_predictions.npz", "3"), ("A3_Q_source_window.npz", "4"))]
                for name in names:
                    record = next(row for row in records if row["run"] == name)
                    seed = record["seed"]
                    training_version = 2 if step == "A4" and seed != 111 else 1
                    meta = dict(run=name, seed=seed, period="test", scenario="S0", lead=1, forecast_horizon=1,
                        unit="mm/day_area_gov", clip_to_zero=True, selected_epoch=2, config_sha256="2"*64,
                        library_commit=LIBRARY_COMMITS[step], compare_version_expected=2 if step == "A4" else 1,
                        training_notebook_version=training_version, source_file_sha256="1"*64,
                        source=f"/fixture/contracts/{name}_test_S0_lead1.npz")
                    record["source_metadata"] = meta
                    contract = {key:value for key, value in meta.items() if key not in ("source", "source_file_sha256")}
                    contract.update(IDs=2, target_days=1461, canonical_context_matches=True, manifest_sha256_matches=True)
                    if step in ("B2", "B3"):
                        contract.update(NSE_audit_mask="finite_obs_and_sim", NSE_audit_median=.5,
                                        NSE_audit_reported_median=.5, NSE_audit_abs_difference=0.)
                        notebook = f"06_B2_LamaHCEExt_Train_Seed{seed}" if step == "B2" else (
                            f"08_B3_LamaHCEExt_Train_{'P10S12' if '_P10S12_' in name else 'P05S05'}_Seed{seed}")
                        contract["training_notebook"] = meta["training_notebook"] = notebook
                    if step == "B2":
                        contract["embedded_training_notebook"] = meta["embedded_training_notebook"] = f"07_B2_LamaHCEExt_Train_Seed{seed}"
                    if step == "B3":
                        mask = "P10S12" if "_P10S12_" in name else "P05S05"
                        contract["mask_level"] = meta["mask_level"] = mask
                        meta["step"] = "B3"
                        for key, value in zip(("nan_step_probability", "nan_sequence_probability"),
                                              (.1, .12) if mask == "P10S12" else (.05, .05)):
                            contract[key] = meta[key] = value
                    contracts.append(contract)
                    inputs.append(dict(file=f"contracts/{name}_test_S0_lead1.npz", expected_sha256="1"*64,
                                       observed_sha256="1"*64, passed=True))
                pd.DataFrame(contracts).to_csv(target_dir/f"Evaluation_{step}_contract_check.csv", index=False)
                pd.DataFrame(inputs).to_csv(target_dir/f"Evaluation_{step}_input_check.csv", index=False)
            if include_c2:
                summary.update(C2_contracts_matched=True, C2_compare_version_expected=1, C2_runs=list(C2_ALL_RUNS))
                c2_contracts, c2_inputs = [], []
                for conf, (core, depth, level, lr, n_params) in C2_CONFIG_SPECS.items():
                    for seed in SEEDS:
                        name = f"SF_{conf}_C2_seed{seed}"
                        record = next(row for row in records if row["run"] == name)
                        nb = f"12_C2_Train_{conf}_Seed{seed}" + ("_Cont" if level == 3 else "")
                        c_sha = hashlib.sha256(name.encode()).hexdigest()
                        k_sha = hashlib.sha256((name+":pt").encode()).hexdigest()
                        record["source_metadata"] = dict(
                            run=name, period="test", scenario="S0", lead=1, forecast_horizon=1, seed=seed,
                            unit="mm/day_area_gov", config_sha256="6"*64, library_commit=FREEZE_C, clip_to_zero=True,
                            step="C2", model="sequential_forecast", core=core, depth=depth, level=level,
                            learning_rate=lr, config=conf, n_params=n_params, selected_epoch=20,
                            grad_guard_skipped_steps=1 if name == "SF_gru_L2_x1_lr1e-3_C2_seed222" else None,
                            training_notebook=nb, compare_notebook="13_C2_LamaHCEExt_Compare",
                            compare_version_expected=1, training_notebook_version_expected=1,
                            source_file_sha256=c_sha, checkpoint_sha256=k_sha, scaler_sha256="7"*64,
                        )
                        c2_contracts.append(dict(
                            run=name, configuration_id=conf, seed=seed, selected_epoch=20, n_params=n_params,
                            IDs=2, target_days=1461, finite_prediction_day_IDs=2922, canonical_context_matches=True,
                            native_test_NSE=.71, NPZ_test_NSE=.71, contract_sha256=c_sha, checkpoint_sha256=k_sha,
                        ))
                        c2_inputs.append(dict(file=f"contracts/{name}_test_S0_lead1.npz",
                                              path=f"/fixture/{name}.npz", bytes=100, sha256=c_sha, passed=True))
                pd.DataFrame(c2_contracts).to_csv(target_dir/"Evaluation_C2_contract_check.csv", index=False)
                pd.DataFrame(c2_inputs).to_csv(target_dir/"Evaluation_C2_input_check.csv", index=False)
            for filename, value in (("summary", summary), ("protocol", protocol), ("run_index", records)):
                (target_dir/f"Evaluation_{filename}.json").write_text(json.dumps(value), encoding="utf-8")
            return records

        b_dir = root / "v8"
        records = populate_eval_fixture(b_dir, include_c2=False)
        loaded = load_evaluation(b_dir, expected_ids=2)
        check("B_32_fixture_loads_all_contracts", loaded[0].run.nunique() == 32 and len(set(loaded[-1])) == 17)
        effects = build_contrasts(loaded[0], loaded[1], loaded[2])
        check("B_36_fixture_contrasts_retain_all_IDs", len(effects) == 72 and effects.contrast.nunique() == 36)
        check("B_fixture_input_metadata_not_rewritten", json.loads((b_dir/"Evaluation_run_index.json").read_text()) == records)
        saved = {path.name:path.read_bytes() for path in b_dir.iterdir()}
        def bad_json(target_dir, backup, name, filename, edit):
            path = target_dir/f"Evaluation_{filename}.json"
            value = json.loads(backup[path.name])
            edit(value)
            path.write_text(json.dumps(value), encoding="utf-8")
            try:
                rejects(name, lambda: load_evaluation(target_dir, expected_ids=2))
            finally:
                path.write_bytes(backup[path.name])
        def bad_csv(target_dir, backup, name, filename, edit):
            path = target_dir/f"Evaluation_{filename}.csv"
            value = pd.read_csv(path)
            value = edit(value)
            value.to_csv(path, index=False)
            try:
                rejects(name, lambda: load_evaluation(target_dir, expected_ids=2))
            finally:
                path.write_bytes(backup[path.name])
        bad_json(b_dir, saved, "B_false_summary_contract_rejected", "summary", lambda value:value.update(B2_contracts_matched=False))
        bad_json(b_dir, saved, "B_wrong_summary_Version_rejected", "summary", lambda value:value.update(B3_compare_version_expected=2))
        bad_json(b_dir, saved, "B_wrong_summary_coverage_rejected", "summary", lambda value:value["coverage"][0].update(common_day_IDs=3))
        b2_index = next(i for i, row in enumerate(records) if row["run"] == B_MODELS["B2"][0])
        bad_json(b_dir, saved, "B_wrong_metadata_period_rejected", "run_index", lambda value:value[b2_index]["source_metadata"].update(period="validation"))
        bad_json(b_dir, saved, "B_wrong_metadata_commit_rejected", "run_index", lambda value:value[b2_index]["source_metadata"].update(library_commit=LIBRARY_COMMITS["B0"]))
        bad_json(b_dir, saved, "B_boolean_metadata_lead_rejected", "run_index", lambda value:value[b2_index]["source_metadata"].update(lead=True))
        bad_json(b_dir, saved, "B_wrong_metadata_epoch_rejected", "run_index", lambda value:value[b2_index]["source_metadata"].update(selected_epoch=3))
        bad_json(b_dir, saved, "B_duplicate_run_index_rejected", "run_index", lambda value:value.append(value[-1]))
        bad_csv(b_dir, saved, "B_missing_contract_seed_rejected", "B3_contract_check", lambda value:value.iloc[:-1])
        bad_csv(b_dir, saved, "B_false_canonical_check_rejected", "B2_contract_check", lambda value:value.assign(canonical_context_matches=False))
        bad_csv(b_dir, saved, "B_wrong_contract_mask_rejected", "B3_contract_check", lambda value:value.assign(mask_level="P10S12"))
        bad_csv(b_dir, saved, "B_wrong_contract_probability_rejected", "B3_contract_check", lambda value:value.assign(nan_step_probability=.2))
        bad_csv(b_dir, saved, "B_wrong_source_hash_rejected", "B2_input_check", lambda value:value.assign(observed_sha256="5"*64))
        bad_csv(b_dir, saved, "B_false_source_flag_rejected", "B1b_input_check", lambda value:value.assign(passed=False))
        bad_csv(b_dir, saved, "B_wrong_metric_seed_rejected", "metrics", lambda value:value.assign(seed=np.where(
            value.run.eq(B_MODELS["B1b"][0]), 333, value.seed)))
        bad_csv(b_dir, saved, "B_wrong_event_seed_rejected", "events", lambda value:value.assign(seed=np.where(
            value.run.eq(B_MODELS["B2"][0]), 333, value.seed)))
        bad_csv(b_dir, saved, "B_different_common_day_count_rejected", "metrics", lambda value:value.assign(n=np.where(
            value.run.eq(B_MODELS["B2"][0]) & value.basin_id.eq(1), 3, value.n)))
        bad_csv(b_dir, saved, "B_different_obs_tss_rejected", "metrics", lambda value:value.assign(obs_tss=np.where(
            value.run.eq(B_MODELS["B2"][0]) & value.basin_id.eq(1), 2., value.obs_tss)))

        c2_dir = root / "v12"
        c2_records = populate_eval_fixture(c2_dir, include_c2=True)
        loaded_c2 = load_evaluation(c2_dir, expected_ids=2)
        check("C2_56_fixture_loads_all_contracts", loaded_c2[0].run.nunique() == 56 and len(set(loaded_c2[-1])) == 19)
        c2_effects = build_contrasts(loaded_c2[0], loaded_c2[1], loaded_c2[2])
        check("C2_105_fixture_contrasts_retain_all_IDs", len(c2_effects) == 210 and c2_effects.contrast.nunique() == 105)
        c2_saved = {path.name:path.read_bytes() for path in c2_dir.iterdir() if path.is_file()}
        c2_idx = next(i for i, row in enumerate(c2_records) if row["run"] == "SF_mamba_L2_x1_lr1e-3_C2_seed111")
        bad_json(c2_dir, c2_saved, "C2_wrong_commit_rejected", "run_index",
                 lambda value:value[c2_idx]["source_metadata"].update(library_commit=LIBRARY_COMMITS["B3"]))
        bad_csv(c2_dir, c2_saved, "C2_wrong_n_params_rejected", "C2_contract_check",
                lambda value:value.assign(n_params=np.where(value.run.eq("SF_mamba_L2_x1_lr1e-3_C2_seed111"), 99999, value.n_params)))
        bad_csv(c2_dir, c2_saved, "C2_wrong_input_hash_rejected", "C2_input_check",
                lambda value:value.assign(sha256="9"*64))

        for include_s4d, expected_pairs in ((False, 6), (True, 9)):
            d_dir = c2_dir / f"D_{expected_pairs}"
            d_dir.mkdir(parents=True, exist_ok=True)
            confs = list(D_CONFIGS if include_s4d else D_CONFIGS[:-1])
            d_runs = [f"SF_{c}_D_seed{s}" for c in confs for s in SEEDS]
            c2_sub = [f"SF_{c}_C2_seed{s}" for c in confs for s in SEEDS]
            all_d_runs = [*c2_sub, *d_runs, "Persistence_qmax"]
            m_rows, e_rows, con_rows, in_rows = [], [], [], []
            for name in all_d_runs:
                seed = next((s for s in SEEDS if name.endswith(f"_seed{s}")), None)
                bonus = .05 if "_D_seed" in name else 0.
                for identifier in ("1", "2"):
                    m_rows.append(dict(run=name, basin_id=identifier, seed=seed, period="test", scenario="S0", lead=1,
                        scope="common_source_all", n=1461, obs_tss=10.,
                        **{metric: .6 + bonus for metric in METRIC_KEYS}))
                    e_rows.append(dict(run=name, basin_id=identifier, seed=seed, period="test", scenario="S0", lead=1,
                        scope="common_source_all", grouping="USWRC", return_period_years=2, matching_tolerance_days=0,
                        hit=2, missed=2, false_alarm=1, POD=.5, FAR=1/3))
            for conf in confs:
                _, _, _, _, n_p = C2_CONFIG_SPECS[conf]
                for seed in SEEDS:
                    rname = f"SF_{conf}_D_seed{seed}"
                    c_sha = hashlib.sha256(rname.encode()).hexdigest()
                    con_rows.append(dict(
                        run=rname, baseline_run=f"SF_{conf}_C2_seed{seed}", configuration_id=conf, seed=seed,
                        selected_epoch=22, n_params=n_p, seq_length=730, grad_guard_enabled=False,
                        grad_guard_skipped_steps=0, strict_pair_equivalent=True, IDs=2, target_days=1461,
                        finite_prediction_day_IDs=2922, canonical_context_matches=True,
                        native_test_NSE=.65, NPZ_test_NSE=.65, contract_sha256=c_sha, checkpoint_sha256="a"*64,
                    ))
                    in_rows.append(dict(file=f"contracts/{rname}_test_S0_lead1.npz",
                                        path=f"/fixture/{rname}.npz", bytes=100, sha256=c_sha, passed=True))
            pd.DataFrame(m_rows).to_csv(d_dir/"Evaluation_D_metrics.csv", index=False)
            pd.DataFrame(e_rows).to_csv(d_dir/"Evaluation_D_events.csv", index=False)
            pd.DataFrame(con_rows).to_csv(d_dir/"Evaluation_D_contract_check.csv", index=False)
            pd.DataFrame(in_rows).to_csv(d_dir/"Evaluation_D_input_check.csv", index=False)
            cov = [dict(period="test", scenario="S0", lead=1, c2_runs=expected_pairs, d_runs=expected_pairs,
                        runs=len(all_d_runs), days=1461, start=TEST_START, end=TEST_END,
                        common_day_IDs=2922, IDs_with_days=2)]
            pd.DataFrame(cov).to_csv(d_dir/"Evaluation_D_coverage.csv", index=False)
            np.savez_compressed(d_dir/"Evaluation_D_common_mask.npz",
                                dates=pd.date_range(TEST_START, TEST_END).to_numpy(dtype="datetime64[D]"),
                                basin_ids=np.array(["1", "2"]), mask=np.ones((1461, 2), dtype=bool))
            d_sum = dict(complete=True, protocol_version=3, compare_notebook="17_D_LamaHCEExt_Compare",
                         library_commit=FREEZE_C, configs=confs, seeds=list(SEEDS), S4D_included=include_s4d,
                         contracts=expected_pairs, paired_c2_runs=expected_pairs, total_scored_runs=len(all_d_runs),
                         grad_guard_runs_with_skips=[], all_pairs_strictly_equivalent=True,
                         event_tolerance_days=0, supplementary_event_tolerance_days=1, coverage=cov)
            (d_dir/"Evaluation_D_summary.json").write_text(json.dumps(d_sum), encoding="utf-8")
            d_loaded = load_d_evaluation(d_dir, loaded_c2[2])
            check(f"D_{expected_pairs}_pair_fixture_loads", len(d_loaded[2]) == expected_pairs)
            bad_d = pd.read_csv(d_dir/"Evaluation_D_contract_check.csv").assign(seq_length=365)
            bad_d.to_csv(d_dir/"Evaluation_D_contract_check.csv", index=False)
            rejects(f"D_{expected_pairs}_wrong_seq_length_rejected", lambda d_dir=d_dir: load_d_evaluation(d_dir, loaded_c2[2]))
    return pd.DataFrame(rows)


# %% [markdown]
# ## Phần 9 — Chạy phân tích và lưu kết quả
# ### 9.1. Kiểm đầu vào → tính bảng → vẽ → phân vùng D (nếu có) → ghi nguồn
# Xuất trực tiếp /kaggle/working/basin_analysis_output; không tạo ZIP.
# Tệp .py là nguồn chỉnh sửa; .ipynb ở đây là bản nhập, bản chính thức tải từ Save & Run All.

# %% Phần 9.1 — Điều phối, manifest và kết luận đúng phạm vi
# Không gọi mô hình; chỉ đặt complete sau khi mọi bảng/hình được ghi thành công.
def run(input_root="/kaggle/input", output_root="/kaggle/working/basin_analysis_output",
        evaluation_root=None, raw_root=None, a1_root=None, a3_root=None, d_evaluation_root=None,
        permutations=PERMUTATIONS, expected_ids=451):
    started = time.perf_counter()
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "BasinAnalysis_summary.json").write_text(json.dumps(dict(
        complete=False, status="started_not_verified", output=str(output)), indent=2), encoding="utf-8")
    checks = self_test()
    evaluation_root = Path(evaluation_root) if evaluation_root else unique_file(input_root, "Evaluation_metrics.csv").parent
    a1_root = Path(a1_root) if a1_root else unique_file(input_root, "A1_P3_static_attribute_sources.csv").parent
    a3_root = Path(a3_root) if a3_root else unique_file(input_root, "A3_DLinear_data_audit.npz").parent
    raw_root = Path(raw_root) if raw_root else Path(input_root)
    if d_evaluation_root is None:
        auto_d = evaluation_root / "D"
        if (auto_d / "Evaluation_D_summary.json").is_file():
            d_evaluation_root = auto_d
        else:
            found_d = unique_file(input_root, "Evaluation_D_summary.json", optional=True)
            if found_d is not None:
                d_evaluation_root = found_d.parent
    print("Phần 1–2: nguồn Evaluation, A1, A3:", evaluation_root, a1_root, a3_root, sep="\n")
    metrics, events, ids, evaluation_summary, evaluation_protocol, paths = load_evaluation(evaluation_root, expected_ids)
    contrast_plan = analysis_contrasts(metrics.run)
    contrast_names = [f"{a}__minus__{b}" for a, b in contrast_plan]
    a4_included = set(A4_MODELS).issubset(set(metrics.run))
    b_included = set(B_MODELS["B0"]).issubset(set(metrics.run))
    c2_included = set(C2_ALL_RUNS).issubset(set(metrics.run))
    families = active_seed_families(metrics.run, contrast_names)
    print(f"Phần 2: {metrics.run.nunique()} run; {len(contrast_plan)} phép so; A4={a4_included}; B0–B3={b_included}; C2={c2_included}")
    features, quality, used = load_attributes(raw_root, a1_root, ids)
    paths.extend(used)
    diagnostics, used, negatives = load_diagnostics(a3_root, ids, features, metrics)
    paths.extend(used)
    features = features.join(diagnostics, validate="one_to_one")
    print(f"Phần 3: {len(ids)} ID; {int((features.test_common_days > 0).sum())} có ngày chấm; kiểm train/mask/obs_tss đạt")
    contrasts = build_contrasts(metrics, events, ids)
    associations, diagnostic_associations = association_tables(contrasts, features, permutations)
    stability = seed_association_stability(associations)
    spreads = {family: seed_spread(contrasts, family) for family in families}
    grouped, bins = group_summary(contrasts, features)
    categories = category_summary(contrasts, features)
    topology, topology_status, used = topology_sensitivity(a1_root, contrasts, features)
    paths.extend(used)
    tables = dict(checks=checks, attributes_and_diagnostics=features.reset_index(), attribute_quality=quality,
                  paired_contrasts=contrasts, static_associations=associations,
                  diagnostic_associations=diagnostic_associations, quartile_definitions=bins,
                  grouped_skill=grouped, categorical_summary=categories,
                  DLinear_seed_spread=spreads["DLinear"], topology_sensitivity=topology,
                  seed_association_stability=stability)
    tables.update({f"{family}_seed_spread": table for family, table in spreads.items() if family != "DLinear"})
    written = []
    for name, table in tables.items():
        path = output / f"BasinAnalysis_{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    figures = draw_figures(associations, contrasts, grouped, features, output, spreads)
    written.extend(figures)
    d_summary_info, d_written = None, []
    if d_evaluation_root is not None:
        print(f"Phần 9.2: phát hiện phân vùng D tại {d_evaluation_root}; phân tích hiệu 730 trừ 365 bước...", flush=True)
        d_summary_info, d_written, d_paths = run_d_subanalysis(
            d_evaluation_root, output, ids, features, a1_root, permutations=permutations)
        written.extend(d_written)
        paths.extend(d_paths)
    manifest = pd.DataFrame([dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                             for p in sorted(set(paths))])
    manifest.to_csv(output / "BasinAnalysis_input_manifest.csv", index=False, encoding="utf-8-sig")
    protocol = dict(analysis_protocol_version=4, scope="test/S0/lead1/common_source_all", expected_IDs=expected_ids,
        main_event_grouping="USWRC", return_period_years=2, event_tolerance_days=0,
        static_predictors=list(NUMERIC_FIELDS), excluded_rank_predictors=dict(lc_dom="categorical_CORINE", mvert_ang="circular_angle"),
        priority_attributes=list(PRIORITY_FIELDS), contrasts=[dict(left=a, right=b) for a, b in contrast_plan],
        runs=sorted(set(metrics.run)), A4_included=a4_included, A4_runs=list(A4_MODELS) if a4_included else [],
        B_included=b_included, B_runs={step:list(names) for step, names in B_MODELS.items()} if b_included else {},
        C2_included=c2_included, C2_runs=list(C2_ALL_RUNS) if c2_included else [],
        D_included=d_summary_info is not None,
        D_configs=d_summary_info["configs"] if d_summary_info else [],
        evaluation_version_expected=(9 if c2_included else 8 if b_included else None),
        source_audit="Evaluation_summary_run_index_and_contract_input_tables_not_raw_NPZ_reexecution",
        seed_families={family: list(SEED_FAMILIES[family]) for family in spreads},
        family_references={family: list(FAMILY_REFERENCES[family]) for family in spreads},
        seed_stability="rho_sign_and_range_across_three_family_specific_paired_effects_no_p_pooling",
        reference_mask="A3_common_source_all_counts_and_obs_tss_must_match_Evaluation",
        outcomes=list(ENDPOINTS), permutations=permutations, seed=PERMUTATION_SEED,
        permutation_type="permute_outcome_ID_pairings_keep_attributes_fixed",
        two_sided_p="(1+count_abs_permuted_rho_ge_abs_observed_rho)/(B+1)",
        minimum_n_for_p=MIN_ASSOCIATION_N, holm_family="31_static_features_per_contrast_and_outcome_finite_p_only",
        cross_family_error_control=False, spatial_exchangeability_verified=False,
        diagnostics_role="rho_only_descriptive_separate_from_static_inference",
        static_periods="precomputed_LamaH_source_periods_not_train_only",
        train_period=[TRAIN_START, TRAIN_END], test_diagnostics="post_hoc_not_for_model_selection",
        negative_Q_conversion="negative_to_NaN_matching_A3_contract", negative_Q_cells=negatives,
        test_used_for_selection=False, changes_to_predictions_or_thresholds=False,
        sources=["https://essd.copernicus.org/articles/13/4529/2021/",
                 "https://hess.copernicus.org/articles/30/5067/2026/",
                 "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html"])
    (output / "BasinAnalysis_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(complete=True, fixture_checks=len(checks), IDs=len(ids),
        IDs_with_common_days=int((features.test_common_days > 0).sum()),
        common_day_IDs=int(features.test_common_days.sum()), runs=int(metrics.run.nunique()),
        contrasts=len(contrast_plan), A4_included=a4_included, A4_runs=list(A4_MODELS) if a4_included else [],
        B_included=b_included, C2_included=c2_included, D_included=d_summary_info is not None,
        D_summary=d_summary_info, seed_families=list(spreads),
        evaluation_version_minimum_expected=(9 if c2_included else 8 if b_included else None),
        static_predictors=len(NUMERIC_FIELDS), association_rows=len(associations),
        finite_rho=int(associations.rho.notna().sum()), permutations=permutations,
        topology=topology_status, figures=len(figures), evaluation_protocol_version=evaluation_protocol.get("protocol_version"),
        seed_stability_rows=len(stability), analysis_protocol_version=4,
        input_contract_checked=True, models_retrained=False, causal_feature_importance_claim=False,
        seconds=time.perf_counter()-started, output=str(output),
        python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__)
    if not all(p.is_file() and p.stat().st_size > 0 for p in written):
        raise RuntimeError("Thiếu output sau khi ghi")
    (output / "BasinAnalysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Phần 9: hoàn thành; rho/n là kết quả mô tả chính, p có giả định trao đổi ID.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    requested = [output / "BasinAnalysis_summary.json", output / "BasinAnalysis_protocol.json",
                 output / "BasinAnalysis_input_manifest.csv", output / "BasinAnalysis_checks.csv",
                 output / "BasinAnalysis_static_associations.csv", output / "BasinAnalysis_grouped_skill.csv",
                 output / "BasinAnalysis_quartile_definitions.csv", output / "BasinAnalysis_seed_association_stability.csv",
                 output / "BasinAnalysis_topology_sensitivity.csv", *figures]
    if d_written:
        requested.extend(d_written)
    print(f"GỬI LẠI {len(requested)} tệp chính đã tạo trong {output}; kèm mọi cảnh báo/traceback:")
    for path in requested:
        print(path.relative_to(output).as_posix())
    return tables


def main():
    parser = argparse.ArgumentParser(description="Liên hệ đặc điểm lưu vực với kết quả đã lưu; không huấn luyện")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working/basin_analysis_output")
    parser.add_argument("--evaluation")
    parser.add_argument("--d-evaluation")
    parser.add_argument("--raw")
    parser.add_argument("--a1")
    parser.add_argument("--a3")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args([] if "ipykernel" in sys.modules else None)
    if args.self_test:
        print(self_test().to_string(index=False))
    else:
        run(args.input, args.output, args.evaluation, args.raw, args.a1, args.a3,
            args.d_evaluation, args.permutations)


if __name__ == "__main__":
    main()
