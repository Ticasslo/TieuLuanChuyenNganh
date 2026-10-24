# %% [markdown]
# # Phân tích kết quả theo đặc điểm lưu vực — Extended LamaH-CE
#
# Đọc A1/A3 và bộ đánh giá 14 run gồm ba LSTM A4; không huấn luyện hoặc chọn lại mô hình.
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
# 7. Hình kết quả và dữ liệu vẽ
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
SEED_FAMILIES = {
    "DLinear": tuple(f"DLinear_seed_{seed}" for seed in SEEDS),
    "LSTM_A4": A4_MODELS,
}
ENDPOINTS = ("delta_NSE", "delta_PNSE", "delta_POD_T2")
PERMUTATIONS, PERMUTATION_SEED, MIN_ASSOCIATION_N = 9999, 20261023, 10
TRAIN_START, TRAIN_END = "2003-01-01", "2009-12-31"
TEST_START, TEST_END = "2014-01-01", "2017-12-31"
METRIC_KEYS = ("NSE", "PNSE", "KGE", "r", "alpha", "beta")


def analysis_contrasts(runs):
    runs = set(runs)
    baseline = set(MODELS) | {"Persistence_qmax", "Persistence_qmean_aux"}
    if runs == baseline:
        return CONTRASTS
    if runs == baseline | set(A4_MODELS):
        return CONTRASTS + A4_CONTRASTS
    raise ValueError("Cần đúng 11 run gốc hoặc đủ 14 run có ba seed A4; không nhận thiếu seed hoặc run lạ")


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
# Gắn bốn Input: dataset lamah-ce-ext, Output A1, Output A3, Output bộ đánh giá.
# Không cần gắn lại A2/A4 Compare vì Output đánh giá đã chứa chỉ số và kiểm hợp đồng A4.
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


def load_evaluation(root, expected_ids=451):
    paths = [Path(root) / f"Evaluation_{name}" for name in (
        "summary.json", "protocol.json", "metrics.csv", "events.csv")]
    summary, protocol = [json.loads(p.read_text(encoding="utf-8-sig")) for p in paths[:2]]
    if not summary.get("complete") or not summary.get("A3_reference_matched"):
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
    if metrics.duplicated(["run", "basin_id"]).any():
        raise ValueError("Trùng run–ID trong bảng chỉ số")
    if required_runs & set(A4_MODELS):
        contract_path = Path(root) / "Evaluation_A4_contract_check.csv"
        input_path = Path(root) / "Evaluation_A4_input_check.csv"
        if (version < 3 or summary.get("A4_contracts_matched") is not True
                or set(summary.get("A4_runs", [])) != set(A4_MODELS)
                or summary.get("A4_compare_version_expected") != 2):
            raise ValueError("Nguồn 14 run cần giao thức 3 và xác nhận đủ ba hợp đồng A4 Compare Version 2")
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
    for frame in (metrics, events):
        frame["basin_id"] = frame.basin_id.map(lambda x: str(int(x)) if re.fullmatch(r"\d+", x) else None)
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
    return metrics, events, ids, summary, protocol, paths


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
# 12 cặp gốc; thêm ba A4 trừ persistence và ba A4 trừ trọng số tác giả thành 18 cặp.
# Chênh PNSE luôn là PNSE trái trừ PNSE phải, không giả định mốc phải luôn bằng 0.
# POD T=2 chỉ xác định ở lưu vực có trận thật; không gán 0 cho mẫu số bằng 0.

# %% Phần 4.1 — Dựng bảng hiệu trên cùng ID/ngày và giữ đủ lưu vực
# Chênh lệch là mô tả các run đã có; so có/không Q một seed không phải thí nghiệm nhân quả.
def build_contrasts(metrics, events, ids):
    m = metrics.set_index(["run", "basin_id"])
    e = events.set_index(["run", "basin_id"])
    rows = []
    for left, right in analysis_contrasts(metrics.run):
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


def seed_spread(contrasts, family="DLinear"):
    names = SEED_FAMILIES[family]
    data = contrasts.loc[contrasts.left_run.isin(names) & contrasts.right_run.eq("Persistence_qmax")]
    if (set(data.left_run) != set(names) or data.duplicated(["basin_id", "left_run"]).any()
            or not data.groupby("basin_id").left_run.nunique().eq(len(SEEDS)).all()):
        raise ValueError(f"{family}: cần đúng ba seed ở từng lưu vực")
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
                             role="three_scores_per_basin_not_prediction_ensemble"))
    return pd.DataFrame(rows)


# %% [markdown]
# ### 4.2. Hướng liên hệ thuộc tính qua ba seed
# Tóm tắt rho của cùng thuộc tính/outcome, mỗi seed so persistence trên cùng giao chung.
# Ba rho không phải ba lưu vực mới; không gộp p hoặc gọi min–max là khoảng tin cậy.

# %% Phần 4.2 — Tổng hợp độ ổn định, giữ rõ seed thiếu/hệ số không xác định
# Hàm được gọi sau khi tính tương quan tại Phần 5; không chọn seed theo test.
def seed_association_stability(associations):
    rows = []
    for family, names in SEED_FAMILIES.items():
        wanted = [f"{name}__minus__Persistence_qmax" for name in names]
        data = associations.loc[associations.contrast.isin(wanted)]
        if data.empty:
            continue
        if set(data.contrast) != set(wanted) or data.duplicated(["contrast", "feature", "outcome"]).any():
            raise ValueError(f"{family}: thiếu hoặc trùng kết quả liên hệ theo seed")
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
                finite_seeds=int(np.isfinite(values).sum()), n_min=int(part.n.min()), n_max=int(part.n.max()),
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


def association_tables(contrasts, features, permutations=PERMUTATIONS):
    main, diagnostics = [], []
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
            diagnostic = rank_associations(features[list(DIAGNOSTICS)], part[outcome], 0, 0)
            diagnostic["contrast"], diagnostic["outcome"] = name, outcome
            diagnostics.append(diagnostic)
        print(f"Liên hệ thuộc tính {number}/{contrasts.contrast.nunique()}: {name}", flush=True)
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
# ## Phần 7 — Hình kết quả và dữ liệu vẽ
# ### 7.1. Giữ các điểm âm và dùng cùng quy tắc phân nhóm
# Hình dùng tiếng Anh; trục symlog giữ cả phần đuôi âm. Không chọn ca theo p hoặc test tốt nhất.

# %% Phần 7.1 — Vẽ hệ số, hiệu kỹ năng và POD cộng dồn
# Chọn trước các cặp hiển thị; mọi cặp và ba seed vẫn có trong CSV đầy đủ.
def draw_figures(associations, contrasts, grouped, features, output, spreads=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = output / "Figures"
    figure_dir.mkdir(exist_ok=True)
    files = []
    heat = associations.loc[associations.outcome.eq("delta_NSE")].pivot(index="feature", columns="contrast", values="rho")
    order = list(contrasts.contrast.drop_duplicates())
    heat = heat.reindex(index=NUMERIC_FIELDS, columns=order)
    short = lambda name: name.replace("SeqLSTM_Q_A4_seed", "LSTM A4 ").replace("DLinear_seed_", "DLinear ").replace(
        "Persistence_qmax", "Persistence").replace("__minus__", " minus ")
    fig, ax = plt.subplots(figsize=(max(16, len(order)*1.15), 12))
    im = ax.imshow(heat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)), [short(name) for name in order], rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(NUMERIC_FIELDS)), NUMERIC_FIELDS, fontsize=9)
    fig.suptitle("Spearman rho: static attributes vs paired delta NSE | all finite basins")
    fig.colorbar(im, ax=ax, label="Spearman rho", shrink=.7)
    fig.tight_layout()
    path = figure_dir / "BasinAnalysis_static_delta_NSE_rho.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(path)
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
            ax.set(xlabel=FEATURE_LABELS[feature], ylabel="Paired delta NSE", title=f"({panel}) {short(left)} minus {short(right)}\nn={int(valid.sum())}")
            ax.grid(alpha=.15)
    fig.tight_layout()
    path = figure_dir / "BasinAnalysis_priority_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(path)
    panels = [("Author and DLinear", ("SeqLSTM_Q", "EncDec_Q", *SEED_FAMILIES["DLinear"]))]
    if set(A4_MODELS).issubset(set(contrasts.left_run)):
        panels.append(("Author and LSTM A4", ("SeqLSTM_Q", *A4_MODELS)))
    styles = {"SeqLSTM_Q": ("#0072B2", "-"), "EncDec_Q": ("#009E73", "-")}
    for family, color in (("DLinear", "#D55E00"), ("LSTM_A4", "#0072B2")):
        styles.update({name: (color, style) for name, style in zip(SEED_FAMILIES[family], ("--", "-.", ":"))})
    for endpoint, ylabel, filename in (("delta_NSE_median", "Median paired delta NSE", "quartile_delta_NSE"),
                                       ("left_POD_T2_pooled", "Pooled POD at T=2 years", "quartile_flood_POD")):
        fig, axes = plt.subplots(len(panels), 4, figsize=(17, 4.6*len(panels)), squeeze=False, sharey=True)
        fig.suptitle(f"{ylabel} by catchment attribute | same quartile boundaries and daily mask")
        for row, (family_label, display_runs) in enumerate(panels):
            for col, (ax, feature) in enumerate(zip(axes[row], PRIORITY_FIELDS)):
                for run in display_runs:
                    name = f"{run}__minus__Persistence_qmax"
                    part = grouped.loc[grouped.feature.eq(feature) & grouped.contrast.eq(name) & grouped.group.ne("missing")]
                    color, style = styles[run]
                    ax.plot(part.group, part[endpoint], marker="o", ms=4, label=short(run), color=color, linestyle=style)
                panel = chr(ord("a") + row*4 + col)
                ax.set(title=f"({panel}) {FEATURE_LABELS[feature]}", xlabel="Attribute quartile (ties may reduce bins)",
                       ylabel=f"{family_label}\n{ylabel}")
                if endpoint == "delta_NSE_median":
                    ax.axhline(0, color="gray", lw=.8)
                ax.grid(alpha=.2)
            axes[row, -1].legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
        fig.tight_layout()
        path = figure_dir / f"BasinAnalysis_{filename}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(path)
    if spreads:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), squeeze=False)
        fig.suptitle("Variation across three seeds per basin | scores relative to persistence")
        for col, (ax, outcome) in enumerate(zip(axes[0], ENDPOINTS)):
            values, labels = [], []
            for family, table in spreads.items():
                data = table.loc[table.outcome.eq(outcome), "sd_sample"].dropna().to_numpy()
                if len(data):
                    values.append(data)
                    labels.append(f"{family}\nn={len(data)}")
            if values:
                ax.boxplot(values, showfliers=True)
                ax.set_xticks(range(1, len(labels)+1), labels)
            ax.set_yscale("symlog", linthresh=.001)
            ax.set_ylim(bottom=0)
            ax.set(title=f"({chr(ord('a')+col)}) {outcome}", ylabel="Sample SD across three scores")
            ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        path = figure_dir / "BasinAnalysis_seed_spread.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(path)
    return files


# %% [markdown]
# ## Phần 8 — Kiểm thử độc lập
# ### 8.1. Ví dụ số và ca suy biến, không phải kết quả LamaH-CE
# Kiểm thứ hạng, phép ghép ID, p hoán vị, NaN, bin có ties và nhóm không có trận.

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
    return pd.DataFrame(rows)


# %% [markdown]
# ## Phần 9 — Chạy phân tích và lưu kết quả
# ### 9.1. Kiểm đầu vào → tính bảng → vẽ → ghi nguồn
# Xuất trực tiếp /kaggle/working/basin_analysis_output; không tạo ZIP.
# Tệp .py là nguồn chỉnh sửa; .ipynb ở đây là bản nhập, bản chính thức tải từ Save & Run All.

# %% Phần 9.1 — Điều phối, manifest và kết luận đúng phạm vi
# Không gọi mô hình; chỉ đặt complete sau khi mọi bảng/hình được ghi thành công.
def run(input_root="/kaggle/input", output_root="/kaggle/working/basin_analysis_output",
        evaluation_root=None, raw_root=None, a1_root=None, a3_root=None, permutations=PERMUTATIONS,
        expected_ids=451):
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
    print("Phần 1–2: nguồn Evaluation, A1, A3:", evaluation_root, a1_root, a3_root, sep="\n")
    metrics, events, ids, evaluation_summary, evaluation_protocol, paths = load_evaluation(evaluation_root, expected_ids)
    contrast_plan = analysis_contrasts(metrics.run)
    a4_included = set(A4_MODELS).issubset(set(metrics.run))
    print(f"Phần 2: {metrics.run.nunique()} run; {len(contrast_plan)} phép so; A4={a4_included}")
    features, quality, used = load_attributes(raw_root, a1_root, ids)
    paths.extend(used)
    diagnostics, used, negatives = load_diagnostics(a3_root, ids, features, metrics)
    paths.extend(used)
    features = features.join(diagnostics, validate="one_to_one")
    print(f"Phần 3: {len(ids)} ID; {int((features.test_common_days > 0).sum())} có ngày chấm; kiểm train/mask/obs_tss đạt")
    contrasts = build_contrasts(metrics, events, ids)
    associations, diagnostic_associations = association_tables(contrasts, features, permutations)
    stability = seed_association_stability(associations)
    spreads = {family: seed_spread(contrasts, family) for family in SEED_FAMILIES
               if family == "DLinear" or a4_included}
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
    if a4_included:
        tables["LSTM_A4_seed_spread"] = spreads["LSTM_A4"]
    written = []
    for name, table in tables.items():
        path = output / f"BasinAnalysis_{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    figures = draw_figures(associations, contrasts, grouped, features, output, spreads)
    written.extend(figures)
    manifest = pd.DataFrame([dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                             for p in sorted(set(paths))])
    manifest.to_csv(output / "BasinAnalysis_input_manifest.csv", index=False, encoding="utf-8-sig")
    protocol = dict(analysis_protocol_version=2, scope="test/S0/lead1/common_source_all", expected_IDs=expected_ids,
        main_event_grouping="USWRC", return_period_years=2, event_tolerance_days=0,
        static_predictors=list(NUMERIC_FIELDS), excluded_rank_predictors=dict(lc_dom="categorical_CORINE", mvert_ang="circular_angle"),
        priority_attributes=list(PRIORITY_FIELDS), contrasts=[dict(left=a, right=b) for a, b in contrast_plan],
        runs=sorted(set(metrics.run)), A4_included=a4_included, A4_runs=list(A4_MODELS) if a4_included else [],
        seed_families={family: list(SEED_FAMILIES[family]) for family in spreads},
        seed_stability="rho_sign_and_range_across_three_runs_vs_persistence_no_p_pooling",
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
                 "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html"])
    (output / "BasinAnalysis_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(complete=True, fixture_checks=len(checks), IDs=len(ids),
        IDs_with_common_days=int((features.test_common_days > 0).sum()),
        common_day_IDs=int(features.test_common_days.sum()), runs=int(metrics.run.nunique()),
        contrasts=len(contrast_plan), A4_included=a4_included, A4_runs=list(A4_MODELS) if a4_included else [],
        static_predictors=len(NUMERIC_FIELDS), association_rows=len(associations),
        finite_rho=int(associations.rho.notna().sum()), permutations=permutations,
        topology=topology_status, figures=len(figures), evaluation_protocol_version=evaluation_protocol.get("protocol_version"),
        seed_stability_rows=len(stability), analysis_protocol_version=2,
        input_contract_checked=True, models_retrained=False, causal_feature_importance_claim=False,
        seconds=time.perf_counter()-started, output=str(output),
        python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__)
    if not all(p.is_file() and p.stat().st_size > 0 for p in written):
        raise RuntimeError("Thiếu output sau khi ghi")
    (output / "BasinAnalysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Phần 9: hoàn thành; rho/n là kết quả mô tả chính, p có giả định trao đổi ID.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("GỬI LẠI: summary/protocol/checks, static_associations, grouped_skill, quartile_definitions,")
    print("DLinear_seed_spread, LSTM_A4_seed_spread (nếu có), seed_association_stability,")
    print("attribute_quality, topology_sensitivity và năm PNG; kèm mọi cảnh báo/traceback.")
    return tables


def main():
    parser = argparse.ArgumentParser(description="Liên hệ đặc điểm lưu vực với kết quả đã lưu; không huấn luyện")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working/basin_analysis_output")
    parser.add_argument("--evaluation")
    parser.add_argument("--raw")
    parser.add_argument("--a1")
    parser.add_argument("--a3")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args([] if "ipykernel" in sys.modules else None)
    if args.self_test:
        print(self_test().to_string(index=False))
    else:
        run(args.input, args.output, args.evaluation, args.raw, args.a1, args.a3, args.permutations)


if __name__ == "__main__":
    main()
