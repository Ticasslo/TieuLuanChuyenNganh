# %% [markdown]
# # Bộ đánh giá dự báo lưu lượng — Extended LamaH-CE
#
# Mã nguồn duy nhất của notebook. Mỗi đề mục Markdown đi cùng cell theo chức năng.
# Đọc Output A1/A2/A3 và A4 Compare Version 2; xuất CSV/PNG/JSON trực tiếp.
# Chạy kiểm độc lập: python 01_Evaluation_LamaHCEExt.py --self-test
#
# Mục lục
# 1. Môi trường và quy ước
# 2. Hợp đồng dự báo và kiểm đầu vào
# 3. NSE, PNSE, KGE và chữ ký đường duy trì
# 4. Bốn mức lưu lượng
# 5. Đoạn vượt ngưỡng và cụm USWRC
# 6. Ghép sự kiện một–một
# 7. So sánh ghép cặp, Holm và kịch bản
# 8. Đọc Output A1/A2/A3 và ba seed LSTM A4
# 9. Kiểm thử bằng ví dụ số
# 10. Chấm, đối chiếu A3 và xuất bảng/hình
# 11. Chạy notebook và kết luận

# %% [markdown]
# ## Phần 1 — Môi trường và quy ước
# ### 1.1. Thư viện và hằng số
# Đơn vị chấm là mm/ngày theo area_gov. Các phép tính chỉ số dùng float64.
# S0 là kịch bản không che thêm đầu vào; dữ liệu gốc vẫn có thể thiếu.
# Hạn k là bước ngày thứ k tính từ ngày đích đầu tiên; hạn 1 dự báo ngày kế tiếp lịch sử.
# Một ngày–lưu vực là một ngày tại một ID. Mặt nạ là bảng True/False chọn các cặp này.
# Giao chung chỉ giữ ngày–lưu vực có obs, persistence và dự báo của mọi run cùng hữu hạn.

# %% Phần 1.1 — Thư viện và hằng số
# Không cài hay thay phiên bản môi trường khi chạy đánh giá.
from dataclasses import dataclass, field, replace
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import pickle
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ("NSE", "PNSE", "KGE", "r", "alpha", "beta")
SIGNATURES = ("BiasFHV", "BiasFMS", "BiasFLV")
LEVEL_NAMES = ("low", "medium", "high", "very_high")
A2_RUNS = ("SeqLSTM_Q", "SeqLSTM_noQ", "EncDec_Q", "EncDec_noQ", "D_FC", "D_RA")
A4_SEEDS = (111, 222, 333)
A4_SEED_VERSIONS = {111: 1, 222: 2, 333: 2}
A4_COMPARE_VERSION = 2
A4_LIBRARY_COMMIT = "dda45fcf19ee7ffd481b9d192778539df46fb074"
RETURN_PERIODS = (1.5, 2.0, 5.0, 10.0)
LOG_FLOOR_MM_DAY = 1e-6
TEST_START, TEST_END = "2014-01-01", "2017-12-31"
PRIMARY_EVENT_TOLERANCE_POLICY = "plus_minus_floor_actual_lead_divided_by_3_calendar_days"
SUPPLEMENTARY_EVENT_TOLERANCE_DAYS = 1
PROTOCOL_VERSION = 3


def primary_event_tolerance_days(lead, forecast_horizon=None):
    """Dung sai của chính hạn đang chấm; tổng số hạn chỉ kiểm hợp đồng."""
    if not isinstance(lead, int) or isinstance(lead, bool) or not 1 <= lead <= 7:
        raise ValueError("lead phải là số nguyên 1–7")
    horizon = lead if forecast_horizon is None else forecast_horizon
    if (not isinstance(horizon, int) or isinstance(horizon, bool)
            or not lead <= horizon <= 7):
        raise ValueError("forecast_horizon phải là số nguyên từ lead đến 7")
    return lead // 3


# %% [markdown]
# ## Phần 2 — Hợp đồng dự báo và kiểm đầu vào
# ### 2.1. Một tệp tương ứng một cấu hình, seed, kỳ, kịch bản và hạn
# origin_date là ngày đích đầu của đợt dự báo, theo quy ước notebook A3/B2.
# Hạn k dự báo target = origin + k−1; persistence luôn lấy Q(origin−1).
# Ngày phát hành thực tế phải được bổ sung từ bên xuất, không suy từ lịch ngày.

# %% Phần 2.1 — Cấu trúc dự báo và kiểm lịch, ID, đơn vị
# Các ma trận có chiều [ngày đích, lưu vực]; cùng nhóm phải có cùng quan trắc.
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


# %% [markdown]
# ## Phần 3 — Các chỉ số liên tục
# ### 3.1. NSE, PNSE, KGE và r, alpha, beta
# Một mặt nạ cho sáu chỉ số; không thêm epsilon vào mẫu số suy biến.

# %% Phần 3.1 — Hàm chỉ số cho một lưu vực
# Kiểm cả các ngày được yêu cầu; không tự bỏ thêm NaN từ mask đã truyền.
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


# %% [markdown]
# ### 3.2. Sai lệch đường duy trì lưu lượng
# FHV dùng 2% ngày cao nhất; FLV dùng 30% thấp nhất; FMS dùng phân vị 0,8 và 0,3.
# Chỉ phép log của FMS/FLV dùng sàn 10⁻⁶ mm/ngày đã khai báo; mẫu số 0 giữ NaN.

# %% Phần 3.2 — FHV, FMS, FLV và trạng thái
# Xếp obs/sim riêng trên cùng tập ngày, không thay đầu vào của NSE/KGE.
def signatures(obs, sim, mask):
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != np.asarray(obs).shape:
        raise ValueError("Sai mặt nạ FDC")
    observed, predicted = [np.asarray(x, dtype=float)[mask] for x in (obs, sim)]
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("FDC chọn ngày không hữu hạn")
    if (observed < 0).any() or (predicted < 0).any():
        raise ValueError("FDC cần bản dự báo không âm theo chính sách đã khai báo")
    out = {key: np.nan for key in SIGNATURES}
    out.update({key + "_status": "no_valid_pairs" for key in SIGNATURES})
    out.update(fdc_log_floor_mm_day=LOG_FLOOR_MM_DAY,
               obs_below_log_floor=int((observed < LOG_FLOOR_MM_DAY).sum()),
               sim_below_log_floor=int((predicted < LOG_FLOOR_MM_DAY).sum()))
    if not len(observed):
        return out
    so, ss = np.sort(observed), np.sort(predicted)
    high = max(1, math.ceil(.02 * len(so)))
    denominator = float(so[-high:].sum())
    out["BiasFHV_status"] = "obs_high_sum_zero"
    if denominator > 0:
        out.update(BiasFHV=100 * (ss[-high:].sum() - denominator) / denominator, BiasFHV_status="ok")
    log_o, log_s = np.log(np.maximum(so, LOG_FLOOR_MM_DAY)), np.log(np.maximum(ss, LOG_FLOOR_MM_DAY))
    om = np.quantile(log_o, .8, method="linear") - np.quantile(log_o, .3, method="linear")
    sm = np.quantile(log_s, .8, method="linear") - np.quantile(log_s, .3, method="linear")
    out["BiasFMS_status"] = "obs_log_slope_zero"
    if om > 0:
        out.update(BiasFMS=100 * (sm - om) / om, BiasFMS_status="ok")
    low = max(1, math.ceil(.3 * len(so)))
    ol = float(np.sum(log_o[:low] - log_o[0]))
    sl = float(np.sum(log_s[:low] - log_s[0]))
    out["BiasFLV_status"] = "obs_low_log_spread_zero"
    if ol > 0:
        out.update(BiasFLV=-100 * (sl - ol) / ol, BiasFLV_status="ok")
    return out


# %% [markdown]
# ## Phần 4 — Đánh giá bốn mức
# ### 4.1. Ma trận nhầm lẫn và chỉ số từng mức
# Dùng ngưỡng A1 cố định; đúng ngưỡng được xếp vào mức cao hơn.

# %% Phần 4.1 — Phân lớp và chỉ số
# Mẫu số 0 trả NaN, không gán recall/precision bằng 1 cho mức không có mẫu.
def ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def confusion(obs, sim, bounds, mask):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (3,) or not np.isfinite(bounds).all() or not (np.diff(bounds) > 0).all():
        raise ValueError("Cần ba ngưỡng q30 < q80 < q98 hữu hạn")
    obs, sim = np.asarray(obs), np.asarray(sim)
    if np.asarray(mask).dtype != np.bool_ or np.asarray(mask).shape != obs.shape or sim.shape != obs.shape:
        raise ValueError("Sai mặt nạ phân mức")
    if not np.isfinite(obs[mask]).all() or not np.isfinite(sim[mask]).all():
        raise ValueError("Phân mức chọn giá trị thiếu")
    left, right = [np.searchsorted(bounds, x[mask], side="right") for x in (obs, sim)]
    return np.bincount(4 * left + right, minlength=16).reshape(4, 4)


def class_scores(matrix):
    rows = []
    for k, level in enumerate(LEVEL_NAMES):
        true, called, hit = int(matrix[k].sum()), int(matrix[:, k].sum()), int(matrix[k, k])
        rows.append(dict(level=level, observed_days=true, predicted_days=called, correct=hit,
                         recall=ratio(hit, true), precision=ratio(hit, called),
                         F1=ratio(2 * hit, true + called), accuracy=ratio(int(np.trace(matrix)), int(matrix.sum()))))
    return rows


# %% [markdown]
# ## Phần 5 — Trích đoạn vượt ngưỡng và gộp USWRC
# ### 5.1. Đoạn liên tiếp và cờ biên/thiếu
# Vượt ngưỡng dùng dấu >, khác quy tắc bằng ngưỡng của phân mức ngày.

# %% Phần 5.1 — Đoạn vượt và gộp cụm
# Giữ chỉ mục trên lịch gốc; không nối qua NaN, không nội suy giữa các ngày.
def exceedance_runs(values, threshold):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.isinf(values).any() or not np.isfinite(threshold):
        raise ValueError("Chuỗi/ngưỡng sự kiện không hợp lệ")
    above = np.isfinite(values) & (values > threshold)
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    ends = np.flatnonzero(above & ~np.r_[above[1:], False])
    runs = []
    for number, (start, end) in enumerate(zip(starts, ends), 1):
        part = values[start:end + 1]
        peaks = np.flatnonzero(part == part.max()) + start
        runs.append(dict(start=int(start), end=int(end), peak=float(part.max()),
                         first_peak=int(peaks[0]), last_peak=int(peaks[-1]), above_days=int(end - start + 1),
                         onset_observed=bool(start > 0 and np.isfinite(values[start - 1])),
                         ending_observed=bool(end + 1 < len(values) and np.isfinite(values[end + 1])),
                         members=[number]))
    return runs


def uswrc(values, runs, area_km2):
    if not np.isfinite(area_km2) or area_km2 <= 0:
        raise ValueError("Diện tích USWRC phải dương")
    minimum = math.ceil(5 + math.log(area_km2 / 1.609344 ** 2))
    minimum = max(1, minimum)
    values = np.asarray(values, dtype=float)
    stack, audit = [], []
    for event in runs:
        stack.append(dict(event, members=list(event["members"])))
        while len(stack) >= 2:
            left, right = stack[-2:]
            a, b = left["last_peak"], right["first_peak"]
            between = values[a + 1:b]
            missing = bool((~np.isfinite(between)).any())
            valley = float(between.min()) if len(between) and not missing else np.nan
            distance_ok = b - a >= minimum
            recession_ok = bool(np.isfinite(valley) and valley < .75 * min(left["peak"], right["peak"]))
            decision = "unresolved_missing" if missing else "separate" if distance_ok and recession_ok else "merge"
            audit.append(dict(left_members="|".join(map(str, left["members"])),
                              right_members="|".join(map(str, right["members"])), decision=decision,
                              peak_distance_days=b - a, minimum_days=minimum, valley=valley))
            if decision != "merge":
                break
            peak = max(left["peak"], right["peak"])
            highest = [x for x in (left, right) if x["peak"] == peak]
            merged = dict(start=left["start"], end=right["end"], peak=peak,
                          first_peak=min(x["first_peak"] for x in highest),
                          last_peak=max(x["last_peak"] for x in highest),
                          above_days=left["above_days"] + right["above_days"],
                          onset_observed=left["onset_observed"], ending_observed=right["ending_observed"],
                          members=left["members"] + right["members"])
            stack[-2:] = [merged]
    return stack, audit


# %% [markdown]
# ## Phần 6 — Ghép sự kiện một–một
# ### 6.1. Số cặp tối đa, sau đó tổng lệch onset nhỏ nhất
# Bảng chính dùng ±floor(hạn đang chấm/3) ngày: 0, 0, 1, 1, 1, 2, 2 cho hạn 1–7.
# Hạn 1 vẫn yêu cầu cùng ngày bắt đầu; bảng phụ luôn dùng ±1 ngày.
# Tổng số hạn của mô hình không thay dung sai của từng hạn.
# NOAA CR TA 98-03 dùng ±1/3 thời gian báo trước: 24 giờ tương ứng ±8 giờ.
# Với lịch ngày, cùng ngày là lựa chọn của đề tài; không xác nhận độ chính xác dưới ngày.
# https://repository.library.noaa.gov/view/noaa/30340/noaa_30340_DS1.pdf (Mục III.B)
# Nearing 2024 dùng ±2 ngày; bảng phụ ±1 của đề tài không tái lập đúng cửa sổ đó.
# https://www.nature.com/articles/s41586-024-07145-1 (Methods, Metrics)
# Loại sự kiện có onset/ending không quan sát được và cửa sổ ghép có ngày thiếu.
# Số sự kiện bị kiểm duyệt được báo riêng, giữ ngày thiếu trong lịch đầy đủ.

# %% Phần 6.1 — Ghép tối ưu theo lịch và chấm POD/FAR
# Quy hoạch động không tái sử dụng một sự kiện cho nhiều lần báo.
def match_events(observed, predicted, tolerance):
    if not isinstance(tolerance, int) or isinstance(tolerance, bool) or tolerance < 0:
        raise ValueError("tolerance phải là số nguyên ngày không âm")
    n, m = len(observed), len(predicted)
    score = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    action = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [(score[i - 1][j], 1), (score[i][j - 1], 2)]
            distance = abs(observed[i - 1]["start"] - predicted[j - 1]["start"])
            if distance <= tolerance:
                prev = score[i - 1][j - 1]
                choices.append(((prev[0] + 1, prev[1] - distance), 3))
            score[i][j], action[i, j] = max(choices, key=lambda x: (x[0], x[1]))
    pairs = []
    i, j = n, m
    while i and j:
        step = action[i, j]
        if step == 3:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def event_rates(hit, missed, false_alarm):
    return dict(POD=ratio(hit, hit + missed), FAR=ratio(false_alarm, hit + false_alarm),
                precision=ratio(hit, hit + false_alarm), F1=ratio(2 * hit, 2 * hit + missed + false_alarm),
                CSI=ratio(hit, hit + missed + false_alarm))


def score_events(obs, sim, threshold, area, mask, lead=1, grouped=True, forecast_horizon=None,
                 tolerance=None):
    primary_tolerance = primary_event_tolerance_days(lead, forecast_horizon)
    if tolerance is None:
        tolerance = primary_tolerance
    if not isinstance(tolerance, int) or isinstance(tolerance, bool) or tolerance < 0:
        raise ValueError("tolerance phải là số nguyên ngày không âm")
    observed, predicted = [np.asarray(x, dtype=float).copy() for x in (obs, sim)]
    if np.asarray(mask).dtype != np.bool_ or np.asarray(mask).shape != observed.shape or predicted.shape != observed.shape:
        raise ValueError("Sai mặt nạ sự kiện")
    if not np.isfinite(observed[mask]).all() or not np.isfinite(predicted[mask]).all():
        raise ValueError("Sự kiện chọn ngày không hữu hạn")
    observed[~mask], predicted[~mask] = np.nan, np.nan
    all_events, eligible, audits = [], [], []
    for values in (observed, predicted):
        events = exceedance_runs(values, threshold)
        audit = []
        if grouped:
            events, audit = uswrc(values, events, area)
        accepted = []
        for event in events:
            start = event["start"]
            full_radius = (start - tolerance >= 0 and start + tolerance < len(mask)
                           and bool(np.all(mask[start - tolerance:start + tolerance + 1])))
            if event["onset_observed"] and event["ending_observed"] and full_radius:
                accepted.append(event)
        all_events.append(events)
        eligible.append(accepted)
        audits.append(audit)
    obs_events, sim_events = eligible
    pairs = match_events(obs_events, sim_events, tolerance)
    hit = len(pairs)
    missed, false_alarm = len(obs_events) - hit, len(sim_events) - hit
    timing = [sim_events[j]["first_peak"] - obs_events[i]["first_peak"] for i, j in pairs]
    result = dict(hit=hit, missed=missed, false_alarm=false_alarm, matching_tolerance_days=tolerance,
                  observed_total=len(all_events[0]), predicted_total=len(all_events[1]),
                  observed_censored=len(all_events[0]) - len(obs_events),
                  predicted_censored=len(all_events[1]) - len(sim_events),
                  unresolved_obs=sum(x["decision"] == "unresolved_missing" for x in audits[0]),
                  unresolved_sim=sum(x["decision"] == "unresolved_missing" for x in audits[1]),
                  timing_signed_sum=sum(timing), timing_abs_sum=sum(abs(x) for x in timing),
                  mean_peak_timing_days=float(np.mean(timing)) if timing else np.nan,
                  mean_abs_peak_timing_days=float(np.mean(np.abs(timing))) if timing else np.nan,
                  valid_days=int(mask.sum()), **event_rates(hit, missed, false_alarm))
    details = [dict(obs_onset=obs_events[i]["start"], sim_onset=sim_events[j]["start"],
                    obs_peak=obs_events[i]["first_peak"], sim_peak=sim_events[j]["first_peak"],
                    peak_timing_days=sim_events[j]["first_peak"] - obs_events[i]["first_peak"])
               for i, j in pairs]
    return result, details


# %% [markdown]
# ## Phần 7 — So sánh mô hình và kịch bản
# ### 7.1. Wilcoxon hai phía, Cohen's dz và Holm
# Mỗi lưu vực là một cặp; giữ điểm âm. Holm chỉ đếm các p-value hữu hạn trong họ.

# %% Phần 7.1 — Kiểm định ghép cặp và hiệu chỉnh
# d không xác định khi std hiệu bằng 0; tất cả hiệu bằng 0 có p=1 theo quy ước.
def paired_test(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("Hai bảng ghép cặp không cùng hình dạng")
    paired = np.isfinite(left) & np.isfinite(right)
    difference = left[paired] - right[paired]
    result = dict(n_pairs=len(difference), improved=int((difference > 0).sum()),
                  degraded=int((difference < 0).sum()), ties=int((difference == 0).sum()),
                  mean_difference=np.nan, median_difference=np.nan, cohen_dz=np.nan,
                  cohen_status="insufficient_pairs", p_value=np.nan, test_status="insufficient_pairs")
    if not len(difference):
        return result
    result.update(mean_difference=float(difference.mean()), median_difference=float(np.median(difference)))
    if len(difference) < 2:
        return result
    sd = float(np.std(difference, ddof=1))
    result["cohen_status"] = "zero_std_difference" if sd == 0 else "ok"
    if sd > 0:
        result["cohen_dz"] = float(difference.mean() / sd)
    if np.all(difference == 0):
        result.update(p_value=1.0, test_status="all_zero_differences")
    else:
        test = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided", correction=False, method="auto")
        result.update(p_value=float(test.pvalue), test_status="ok_auto_method")
    return result


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or np.isinf(pvalues).any() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("p phải thuộc [0,1] hoặc NaN")
    adjusted = np.full(len(pvalues), np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    order = valid[np.argsort(pvalues[valid], kind="stable")]
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(valid) - rank) * pvalues[index])
        adjusted[index] = min(1.0, previous)
    return adjusted


def scenario_changes(forecasts):
    rows = []
    index = {(f.name, f.period, f.lead, f.scenario): f for f in forecasts}
    for forecast in forecasts:
        if forecast.scenario == "S0":
            continue
        base = index.get((forecast.name, forecast.period, forecast.lead, "S0"))
        if base is None:
            raise ValueError(f"Thiếu S0 để đối chiếu {forecast.name}/{forecast.scenario}")
        check_same_context(base, forecast)
        mask = common_mask([base, forecast])
        for j, identifier in enumerate(forecast.ids):
            before = metrics(base.obs[:, j], base.sim[:, j], base.persistence[:, j], mask[:, j])
            after = metrics(forecast.obs[:, j], forecast.sim[:, j], forecast.persistence[:, j], mask[:, j])
            rows.append(dict(run=forecast.name, period=forecast.period, lead=forecast.lead,
                             scenario=forecast.scenario, basin_id=identifier, n=int(mask[:, j].sum()),
                             **{f"delta_{key}_vs_S0": after[key] - before[key] for key in METRICS}))
    columns = ["run", "period", "lead", "scenario", "basin_id", "n"]
    columns.extend(f"delta_{key}_vs_S0" for key in METRICS)
    return pd.DataFrame(rows, columns=columns)


# %% [markdown]
# ## Phần 8 — Đọc Output A1/A2/A3/A4
# ### 8.1. Chọn nguồn duy nhất và ghi SHA-256
# Nếu Add Input có nhiều phiên bản cùng tên, chỉ định đường dẫn thay vì tự chọn.

# %% Phần 8.1 — Tìm tệp và kiểm ngữ cảnh canonical A3
# NPZ đọc với allow_pickle=False. Pickle chỉ dùng cho export A2 của nhóm.
def unique_file(root, pattern, required=True):
    found = sorted(Path(root).rglob(pattern))
    if len(found) > 1:
        raise ValueError(f"Có {len(found)} nguồn {pattern}; chỉ định thư mục riêng: {found}")
    if not found and required:
        raise FileNotFoundError(f"Thiếu {pattern} dưới {root}; Add Input Output đã Save Version")
    return found[0] if found else None


def find_a2_root(root):
    # Nhận diện cấu trúc sáu run, vì A4 cũng có environment.json.
    found = sorted(path.parent for path in Path(root).rglob("environment.json")
                   if all((path.parent / "runs" / name).is_dir() for name in A2_RUNS))
    if len(found) != 1:
        raise ValueError(f"Cần đúng một export A2 có runs và đủ sáu run, thấy {len(found)}: {found}")
    return found[0]


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


# %% [markdown]
# ### 8.2. Sáu run A2 và ba seed DLinear
# A2 giữ obs native để đối chiếu; phép so chính dùng obs canonical A3.

# %% Phần 8.2 — Bộ đọc dự báo đã lưu
# Không gọi mô hình hoặc chạy huấn luyện trong bộ đánh giá.
def read_a2(a2_root, context):
    import yaml
    forecasts, paths = [], []
    for name in A2_RUNS:
        root = Path(a2_root) / "runs" / name
        pred_path = unique_file(root, "test_results.p")
        config_path = unique_file(root, "config.yml")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
        clip = config.get("clip_targets_to_zero", [])
        clip = [clip] if isinstance(clip, str) else (clip or [])
        with pred_path.open("rb") as stream:
            saved = pickle.load(stream)
        keys = {basin_id(key): key for key in saved}
        if len(keys) != len(saved) or set(keys) != set(context["ids"]):
            raise ValueError(f"{name}: ID thiếu, thừa hoặc trùng")
        native, raw = np.full_like(context["obs"], np.nan), np.full_like(context["obs"], np.nan)
        for j, identifier in enumerate(context["ids"]):
            dataset = saved[keys[identifier]]["1D"]["xr"]
            if not np.array_equal(dataset["time_step"].values, [0]):
                raise ValueError("A2 phải có đúng time_step=[0]")
            dates = pd.DatetimeIndex(dataset["date"].values).as_unit("ns")
            if not dates.equals(context["dates"]):
                raise ValueError("Lịch A2 khác toàn kỳ test")
            for field_name, output in (("qmax_obs", native), ("qmax_sim", raw)):
                array = dataset[field_name]
                if set(array.dims) != {"date", "time_step"}:
                    raise ValueError("Sai chiều xarray A2")
                output[:, j] = array.transpose("date", "time_step").values[:, 0]
        if not np.array_equal(np.isfinite(native), np.isfinite(context["obs"])):
            raise ValueError("Mặt nạ obs native A2 khác canonical")
        sim = np.maximum(raw, 0) if "qmax" in clip else raw.copy()
        forecasts.append(Forecast(name=name, sim=sim, raw=raw, native_obs=native, **context,
                                  metadata=dict(config_sha256=sha256(config_path), source=str(pred_path),
                                                clip_to_zero="qmax" in clip)).validate())
        paths.extend([pred_path, config_path])
    return forecasts, paths


def read_dlinear(a3_root, context):
    forecasts, paths = [], []
    for seed in (111, 222, 333):
        path = unique_file(Path(a3_root) / "dlinear" / f"seed_{seed}", "test_predictions.npz")
        with np.load(path, allow_pickle=False) as data:
            if (not pd.DatetimeIndex(data["dates"]).as_unit("ns").equals(context["dates"])
                    or tuple(basin_id(x) for x in data["basin_ids"]) != context["ids"]
                    or not np.array_equal(data["area_gov_km2"], context["area"])
                    or not np.array_equal(data["obs_mm_day"], context["obs"], equal_nan=True)
                    or not np.array_equal(data["source_last_dates"], data["dates"] - np.timedelta64(1, "D"))
                    or int(data["seed"]) != seed or str(data["period"].item()) != "test"):
                raise ValueError(f"DLinear {seed}: khác nguồn, lịch, diện tích hoặc seed")
            raw, sim = np.array(data["sim_raw_mm_day"], dtype=float), np.array(data["sim_mm_day"], dtype=float)
            if not np.array_equal(sim, np.maximum(raw, 0), equal_nan=True):
                raise ValueError("DLinear không khớp chính sách clip A3")
            metadata = dict(source=str(path), config_sha256=str(data["config_sha256"].item()),
                            selected_epoch=int(data["selected_epoch"]))
        forecasts.append(Forecast(name=f"DLinear_seed_{seed}", seed=seed, raw=raw, sim=sim,
                                  metadata=metadata, **context).validate())
        paths.append(path)
    return forecasts, paths


# %% [markdown]
# ### 8.3. Ngưỡng A1 và dự báo định dạng chung cho các bước sau
# Ngưỡng là đầu vào cố định; bộ đánh giá không fit lại theo test hoặc theo sim.

# %% Phần 8.3 — Đọc ngưỡng và hợp đồng NPZ mới
# Định dạng mở rộng được mô tả trong tệp thuyết minh cùng tên.
def read_thresholds(a1_root, ids):
    level_path = unique_file(a1_root, "A1_P10_qmax_level_thresholds.csv")
    flood_path = unique_file(a1_root, "A1_P10_flood_return_levels.csv")
    levels = pd.read_csv(level_path, dtype={"basin_id": str})
    levels["basin_id"] = levels.basin_id.map(basin_id)
    if levels.basin_id.duplicated().any() or set(levels.basin_id) != set(ids):
        raise ValueError("Ngưỡng mức không đúng danh sách ID")
    if not (levels.reference_start.eq("1981-01-01").all() and levels.reference_end.eq("2013-12-31").all()):
        raise ValueError("Ngưỡng mức không đúng tham chiếu 1981–2013")
    levels = levels.set_index("basin_id").reindex(ids)
    bounds = levels[["q30_mm_day", "q80_mm_day", "q98_mm_day"]].to_numpy(dtype=float)
    if not np.isfinite(bounds).all() or not (np.diff(bounds, axis=1) > 0).all():
        raise ValueError("Ngưỡng mức thiếu, suy biến hoặc sai thứ tự")
    floods = pd.read_csv(flood_path, dtype={"basin_id": str})
    floods["basin_id"] = floods.basin_id.map(basin_id)
    floods = floods.loc[floods.policy.eq("half_calendar_primary")].copy()
    if floods.duplicated(["basin_id", "return_period_years"]).any():
        raise ValueError("Ngưỡng lũ bị trùng")
    expected = pd.MultiIndex.from_product([ids, RETURN_PERIODS], names=["basin_id", "return_period_years"])
    floods = floods.set_index(["basin_id", "return_period_years"])
    if set(floods.index) != set(expected):
        raise ValueError("Ngưỡng lũ không đủ ID × bốn T")
    floods = floods.reindex(expected)
    if (not np.isfinite(floods.threshold_mm_day).all() or (floods.threshold_mm_day < 0).any()
            or not floods.fit_status.eq("fit_defined").all() or (floods.n_years < 3).any()):
        raise ValueError("Ngưỡng Gumbel hoặc số năm fit không hợp lệ")
    return levels, floods, [level_path, flood_path]


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


# %% [markdown]
# ### 8.4. Ba seed LSTM A4 từ Output notebook Compare
# Gắn Output 04_A4_LamaHCEExt_Compare Version 2, không cần ba Output huấn luyện riêng.
# Kiểm manifest của a4_compare, hash nguồn A3, ba seed và ngữ cảnh trước khi chấm.
# Số Version Compare là thông tin bàn giao; danh tính tệp được ghi bằng SHA-256.

# %% Phần 8.4 - Kiểm và đọc ba NPZ A4
# Thiếu tệp, trùng nguồn, sai hash hoặc khác ngữ cảnh => dừng; không tự bỏ seed.
def read_a4(a4_root, context, context_paths):
    root = unique_file(a4_root, "a4_summary.json").parent
    summary_path = root / "a4_summary.json"
    environment_path = root / "environment.json"
    manifest_path = root / "manifest.csv"
    for path in (summary_path, environment_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"A4 thiếu {path.name}; gắn Output Compare Version {A4_COMPARE_VERSION}")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if (not {"tệp", "sha256"}.issubset(manifest.columns) or manifest.empty
            or manifest["tệp"].duplicated().any()):
        raise ValueError("Manifest A4 cần cột tệp/sha256, không rỗng và không trùng đường dẫn")
    files, input_rows = {}, []
    for row in manifest.to_dict("records"):
        relative = row["tệp"]
        path = root / relative
        if (not relative or "\\" in relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError(f"Đường dẫn manifest A4 phải nằm trong a4_compare: {relative}")
        expected = row["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or not path.is_file():
            raise ValueError(f"A4 thiếu tệp hoặc digest không hợp lệ: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"SHA-256 A4 không khớp manifest: {relative}")
        files[relative] = path
        input_rows.append(dict(kind="A4_export", file=relative, path=str(path),
                               expected_sha256=expected, observed_sha256=observed, passed=True))
    required = {"a4_summary.json", "environment.json"} | {
        f"contracts/SeqLSTM_Q_A4_seed{seed}_test_S0_lead1.npz" for seed in A4_SEEDS}
    if not required.issubset(files):
        raise ValueError(f"Manifest A4 thiếu các tệp bắt buộc: {sorted(required - set(files))}")
    contract_files = {p.relative_to(root).as_posix() for p in (root / "contracts").glob("*.npz")}
    if contract_files != {name for name in required if name.endswith(".npz")}:
        raise ValueError("A4 Compare phải có đúng ba NPZ seed 111/222/333 ở contracts")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    environment = json.loads(environment_path.read_text(encoding="utf-8-sig"))
    if (environment.get("notebook") != "04_A4_LamaHCEExt_Compare"
            or environment.get("neuralhydrology_commit_of_runs") != A4_LIBRARY_COMMIT):
        raise ValueError("Sai notebook nguồn hoặc commit thư viện A4")
    seed_keys = {str(seed) for seed in A4_SEEDS}
    for key in ("contracts", "saved_epoch", "seed_test_nse_median"):
        if not isinstance(summary.get(key), dict) or set(summary[key]) != seed_keys:
            raise ValueError(f"A4 summary thiếu hoặc thừa seed ở {key}")
    source_a3 = environment.get("inputs", {}).get("a3", {})
    if source_a3.get("version") != 1:
        raise ValueError("A4 cần khai báo nguồn A3 Version 1 như bản bàn giao")
    by_name = {path.name: path for path in context_paths}
    for name in ("A3_persistence_predictions.npz", "A3_Q_source_window.npz"):
        expected = source_a3.get("files", {}).get(name, {}).get("sha256")
        observed = sha256(by_name[name])
        if expected != observed:
            raise ValueError(f"A4 dùng nguồn A3 khác Input hiện tại: {name}")
        input_rows.append(dict(kind="A3_context_used_by_A4", file=name, path=str(by_name[name]),
                               expected_sha256=expected, observed_sha256=observed, passed=True))
    reference = Forecast("A3_context", sim=context["persistence"], raw=context["persistence"], **context).validate()
    forecasts, contract_rows = [], []
    for seed in A4_SEEDS:
        key = str(seed)
        relative = f"contracts/SeqLSTM_Q_A4_seed{seed}_test_S0_lead1.npz"
        if not str(summary["contracts"][key]).endswith("/" + relative):
            raise ValueError(f"A4 summary không trỏ đúng tệp seed {seed}")
        source_version = environment.get("inputs", {}).get("seeds", {}).get(key, {}).get("version")
        if type(source_version) is not int or source_version != A4_SEED_VERSIONS[seed]:
            raise ValueError(f"A4 seed {seed}: Version nguồn không khớp bản bàn giao")
        forecast = read_contract(files[relative])
        meta = forecast.metadata
        selected_epoch = meta.get("selected_epoch")
        if (forecast.name != f"SeqLSTM_Q_A4_seed{seed}" or type(forecast.seed) is not int
                or forecast.seed != seed or forecast.group != ("test", "S0", 1)
                or meta.get("forecast_horizon") != 1 or meta.get("clip_to_zero") is not True
                or meta.get("library_commit") != A4_LIBRARY_COMMIT
                or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("config_sha256")))
                or type(selected_epoch) is not int or selected_epoch <= 0
                or summary["saved_epoch"][key] != selected_epoch):
            raise ValueError(f"A4 seed {seed}: sai run/kỳ/hạn/seed/epoch/commit/config hoặc chính sách clip")
        check_same_context(reference, forecast)
        forecast.metadata = dict(meta, source=str(files[relative]), source_file_sha256=sha256(files[relative]),
                                 compare_version_expected=A4_COMPARE_VERSION, training_notebook_version=source_version)
        forecasts.append(forecast)
        contract_rows.append(dict(run=forecast.name, seed=seed, selected_epoch=selected_epoch,
                                  training_notebook_version=source_version,
                                  compare_version_expected=A4_COMPARE_VERSION,
                                  period=forecast.period, scenario=forecast.scenario, lead=forecast.lead,
                                  IDs=len(forecast.ids), target_days=len(forecast.dates),
                                  finite_prediction_day_IDs=int(np.isfinite(forecast.sim).sum()),
                                  paired_obs_prediction_day_IDs=int((np.isfinite(forecast.obs) & np.isfinite(forecast.sim)).sum()),
                                  sim_raw_negative_day_IDs=int((forecast.raw < 0).sum()),
                                  config_sha256=meta["config_sha256"], library_commit=meta["library_commit"],
                                  canonical_context_matches=True, manifest_sha256_matches=True))
    paths = [manifest_path, *files.values()]
    return forecasts, paths, pd.DataFrame(input_rows), pd.DataFrame(contract_rows)


def show_table(title, table):
    # Notebook hiển thị bảng; script vẫn đọc được qua stdout.
    if "ipykernel" in sys.modules:
        from IPython.display import HTML, display
        display(HTML(f"<h4>{title}</h4>"))
        display(table.style.hide(axis="index").format(precision=6, na_rep="-"))
    else:
        print(title)
        print(table.to_string(index=False))


# %% [markdown]
# ## Phần 9 — Kiểm thử độc lập
# ### 9.1. Ví dụ số, ca suy biến và bộ lọc
# Kết quả ở phần này là fixture, tách khỏi số liệu LamaH-CE.

# %% Phần 9.1 — Kiểm số học và giao thức trước khi đọc dataset
# Chạy cả khi chưa Add Input; lỗi kiểm thử dừng trước mọi bảng kết quả thực nghiệm.
def self_test():
    checks = []

    def check(name, condition):
        checks.append(dict(case=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    def rejects(name, operation):
        try:
            operation()
        except (ValueError, FloatingPointError):
            check(name, True)
        else:
            check(name, False)

    obs, sim, persistence = np.array([1., 2., 3., 4.]), np.array([1., 2., 2., 4.]), np.array([1., 1., 2., 3.])
    result = metrics(obs, sim, persistence)
    check("worked_NSE_PNSE", np.allclose([result["NSE"], result["PNSE"]], [.8, 2 / 3]))
    check("Pearson_independent_reference", np.isclose(result["r"], stats.pearsonr(obs, sim).statistic))
    check("KGE_components", np.allclose([result["alpha"], result["beta"], result["KGE"]], [.9746794344808964, .9, .8715022325785454]))
    check("scale_invariance", all(np.isclose(result[k], metrics(obs * 20, sim * 20, persistence * 20)[k]) for k in METRICS))
    check("perfect_prediction", all(np.isclose(metrics(obs, obs, persistence)[k], 1) for k in METRICS))
    check("constant_sim_KGE_undefined", np.isnan(metrics(obs, np.ones(4), persistence)["KGE"]))
    check("constant_obs_NSE_undefined", metrics(np.ones(4), sim)["NSE_status"] == "obs_zero_variance")
    check("zero_persistence_SSE_undefined", metrics(obs, sim, obs)["PNSE_status"] == "persistence_sse_zero")
    check("tiny_positive_variance_retained", metrics(np.array([1., 1.00000001, 1.00000002]), np.ones(3) * 2)["NSE"] < -1e15)
    check("zero_pairs_retained", metrics([np.nan], [1.])["n"] == 0)
    rejects("invalid_explicit_mask", lambda: metrics(obs, sim, mask=np.array([1, 1, 1, 1])))
    rejects("mask_does_not_drop_NaN", lambda: metrics([1., np.nan], [1., 2.], mask=np.ones(2, dtype=bool)))
    rejects("reject_infinity", lambda: metrics([1., np.inf], [1., 2.]))
    check("FDC_identity", all(np.isclose(signatures(np.arange(1., 101.), np.arange(1., 101.), np.ones(100, dtype=bool))[k], 0) for k in SIGNATURES))
    check("FDC_zero_denominators", np.isnan(signatures(np.zeros(10), np.zeros(10), np.ones(10, dtype=bool))["BiasFLV"]))
    c = confusion(np.array([0., 1., 2., 3.]), np.array([0., 1., 2., 3.]), [1, 2, 3], np.ones(4, dtype=bool))
    check("level_equal_boundary_moves_up", np.array_equal(c, np.eye(4, dtype=int)))
    check("empty_class_undefined", np.isnan(class_scores(np.zeros((4, 4), dtype=int))[0]["recall"]))
    values = np.array([0., 5., 0., 6., 0., np.nan, 7., 0.])
    runs = exceedance_runs(values, 4)
    groups, audit = uswrc(values, runs, 101)
    check("USWRC_merges_close_peaks", len(groups) == 2 and groups[0]["above_days"] == 2)
    check("USWRC_missing_blocks_merge", any(x["decision"] == "unresolved_missing" for x in audit))
    check("missing_onset_flag", not runs[-1]["onset_observed"])
    check("strict_flood_exceedance", not exceedance_runs([0., 4., 0.], 4))
    separated = np.zeros(45)
    separated[[3, 30]] = [5, 6]
    check("USWRC_independent_peaks", len(uswrc(separated, exceedance_runs(separated, 4), 101)[0]) == 2)
    recession = separated.copy()
    recession[4:30] = 3.75
    check("USWRC_recession_equality_merges", len(uswrc(recession, exceedance_runs(recession, 4), 101)[0]) == 1)
    pairs = match_events([dict(start=2), dict(start=4)], [dict(start=1), dict(start=3)], 1)
    check("matching_max_cardinality", pairs == [(0, 0), (1, 1)])
    check("matching_no_reuse", len(match_events([dict(start=2), dict(start=3)], [dict(start=2)], 1)) == 1)
    eobs, esim = np.zeros(16), np.zeros(16)
    eobs[[4, 11]], esim[[4]] = 5, 5
    rates, _ = score_events(eobs, esim, 4, 101, np.ones(16, dtype=bool), grouped=False)
    check("event_hit_missed", (rates["hit"], rates["missed"], rates["false_alarm"]) == (1, 1, 0))
    check("event_POD_FAR", rates["POD"] == .5 and rates["FAR"] == 0)
    # Persistence trễ đúng một ngày: chính không bắt được; phụ ±1 bắt cả hai trận.
    delayed = np.r_[0., eobs[:-1]]
    strict, strict_pairs = score_events(eobs, delayed, 4, 101, np.ones(16, dtype=bool), grouped=False)
    relaxed, relaxed_pairs = score_events(eobs, delayed, 4, 101, np.ones(16, dtype=bool),
                                           grouped=False, tolerance=1)
    check("delayed_persistence_not_a_same_day_hit", (strict["hit"], strict["missed"], strict["false_alarm"]) == (0, 2, 2)
          and strict["POD"] == 0 and strict["FAR"] == 1 and not strict_pairs)
    check("delayed_persistence_supplementary_POD_one", relaxed["POD"] == 1 and relaxed["FAR"] == 0
          and len(relaxed_pairs) == 2 and relaxed["mean_peak_timing_days"] == 1)
    early = np.r_[eobs[1:], 0.]
    strict_early, _ = score_events(eobs, early, 4, 101, np.ones(16, dtype=bool), grouped=False)
    check("early_prediction_not_a_same_day_hit", strict_early["hit"] == 0 and strict_early["missed"] == 2)
    missing = np.ones(16, dtype=bool)
    missing[3] = False
    censored, _ = score_events(eobs, eobs, 4, 101, missing, grouped=False)
    check("zero_tolerance_still_censors_missing_onset", censored["observed_censored"] == 1 and censored["hit"] == 1)
    rejects("negative_tolerance_rejected", lambda: score_events(eobs, delayed, 4, 101, missing, tolerance=-1))
    rejects("fractional_tolerance_rejected", lambda: match_events([], [], .5))
    horizon_obs, horizon_sim = np.zeros(12), np.zeros(12)
    horizon_obs[4], horizon_sim[6] = 5, 5
    horizon_score, _ = score_events(horizon_obs, horizon_sim, 4, 101, np.ones(12, dtype=bool),
                                    lead=1, forecast_horizon=7, grouped=False)
    check("H_lead1_keeps_same_day_primary", horizon_score["hit"] == 0 and horizon_score["matching_tolerance_days"] == 0)
    # Biên dung sai, hai hướng và bán kính kiểm duyệt cho từng hạn thực tế.
    lead_obs = np.zeros(48)
    lead_obs[20] = 5
    lead_mask = np.ones(48, dtype=bool)
    for lead, expected in enumerate((0, 0, 1, 1, 1, 2, 2), start=1):
        check(f"H_lead{lead}_declared_tolerance", primary_event_tolerance_days(lead, 7) == expected)
        for direction in (-1, 1):
            within, outside = np.zeros(48), np.zeros(48)
            within[20 + direction * expected] = 5
            outside[20 + direction * (expected + 1)] = 5
            accepted, _ = score_events(lead_obs, within, 4, 101, lead_mask, lead=lead,
                                       forecast_horizon=7, grouped=False)
            rejected, _ = score_events(lead_obs, outside, 4, 101, lead_mask, lead=lead,
                                       forecast_horizon=7, grouped=False)
            check(f"H_lead{lead}_boundary_direction{direction}", accepted["hit"] == 1
                  and accepted["matching_tolerance_days"] == expected
                  and (rejected["hit"], rejected["missed"], rejected["false_alarm"]) == (0, 1, 1))
        radius_mask = lead_mask.copy()
        radius_mask[18] = False
        radius_score, _ = score_events(lead_obs, lead_obs, 4, 101, radius_mask,
                                       lead=lead, forecast_horizon=7, grouped=False)
        check(f"H_lead{lead}_missing_radius", radius_score["hit"] == int(expected < 2)
              and radius_score["observed_censored"] == int(expected == 2))
        supplementary, _ = score_events(lead_obs, np.roll(lead_obs, 1), 4, 101, lead_mask,
                                        lead=lead, forecast_horizon=7, grouped=False,
                                        tolerance=SUPPLEMENTARY_EVENT_TOLERANCE_DAYS)
        check(f"H_lead{lead}_supplementary_fixed_one_day", supplementary["hit"] == 1
              and supplementary["matching_tolerance_days"] == 1)
    explicit, _ = score_events(lead_obs, np.roll(lead_obs, 1), 4, 101, lead_mask,
                               lead=7, forecast_horizon=7, grouped=False, tolerance=0)
    check("explicit_zero_not_replaced_by_default", explicit["hit"] == 0 and explicit["matching_tolerance_days"] == 0)
    edge_obs = np.zeros(48)
    edge_obs[1] = 5
    edge_short, _ = score_events(edge_obs, edge_obs, 4, 101, lead_mask, lead=1, forecast_horizon=7, grouped=False)
    edge_long, _ = score_events(edge_obs, edge_obs, 4, 101, lead_mask, lead=7, forecast_horizon=7, grouped=False)
    check("long_lead_radius_censors_calendar_edge", edge_short["hit"] == 1
          and edge_long["hit"] == 0 and edge_long["observed_censored"] == 1)
    for label, value in (("bool", True), ("fractional", 1.5), ("float", 1.),
                         ("zero", 0), ("above_seven", 8), ("string", "3")):
        rejects(f"invalid_lead_{label}_rejected", lambda value=value: score_events(
            lead_obs, lead_obs, 4, 101, lead_mask, lead=value, forecast_horizon=7))
    for label, value in (("bool", True), ("fractional", 3.5), ("float", 7.),
                         ("below_lead", 2), ("above_seven", 8), ("string", "7")):
        rejects(f"invalid_horizon_{label}_rejected", lambda value=value: score_events(
            lead_obs, lead_obs, 4, 101, lead_mask, lead=3, forecast_horizon=value))
    for label, value in (("bool", True), ("fractional", .5), ("string", "1")):
        rejects(f"invalid_score_tolerance_{label}_rejected", lambda value=value: score_events(
            lead_obs, lead_obs, 4, 101, lead_mask, tolerance=value))
    check("no_events_not_perfect", np.isnan(event_rates(0, 0, 0)["POD"]))
    check("Holm_known_values", np.allclose(holm([.01, .04, .03]), [.03, .06, .06]))
    check("Holm_excludes_NaN_from_family", np.allclose(holm([.01, np.nan, .03]), [.02, np.nan, .03], equal_nan=True))
    check("Holm_all_NaN_or_empty", np.isnan(holm([np.nan, np.nan])).all() and len(holm([])) == 0)
    check("Holm_single_finite_p", np.allclose(holm([np.nan, .04]), [np.nan, .04], equal_nan=True))
    check("paired_test_identity", paired_test(obs, obs)["p_value"] == 1)
    check("paired_dz_known", np.isclose(paired_test([2., 4., 6.], [1., 2., 3.])["cohen_dz"], 2))
    check("paired_undefined_IDs_counted", paired_test([1., np.nan, 2.], [1., 0., 3.])["n_pairs"] == 2)
    dates = pd.date_range("2014-01-01", periods=4)
    fixture = Forecast("fixture", dates, ("01",), np.array([101.]), obs[:, None], sim[:, None], persistence[:, None], sim[:, None]).validate()
    check("IDs_normalized", fixture.ids == ("1",))
    rejects("duplicate_IDs_rejected", lambda: replace(fixture, ids=("1", "01")).validate())
    changed = replace(fixture, obs=(obs + .1)[:, None])
    rejects("different_canonical_obs_rejected", lambda: common_mask([fixture, changed]))
    changed = replace(fixture, scenario="S2", sim=np.array([[1.], [2.], [np.nan], [4.]]))
    delta = scenario_changes([fixture, changed])
    check("scenario_same_day_mask", int(delta.iloc[0]["n"]) == 3
          and list(scenario_changes([fixture]).columns) == list(delta.columns))
    # Fixture I/O: toàn bộ lịch test nhưng chỉ hai ID, không phải số liệu thực nghiệm.
    import xarray as xr
    import yaml
    with tempfile.TemporaryDirectory(prefix="evaluation_fixture_") as folder:
        root = Path(folder)
        a1, a2, a3 = root / "a1", root / "a2", root / "a3"
        for directory in (a1, a2, a3):
            directory.mkdir()
        full_dates = pd.date_range("2013-12-31", TEST_END)
        dates = full_dates[1:]
        ids = ("1", "2")
        area = np.array([101., 50.])
        t = np.arange(len(full_dates), dtype=float)
        q = np.column_stack([1 + .4 * np.sin(t / 15), 2 + .8 * np.cos(t / 30)])
        q[50, 0] = np.nan
        m3_s = q * area[None, :] / 86.4
        physical = m3_s * 86.4 / area[None, :]
        np.savez_compressed(a3 / "A3_Q_source_window.npz", dates=full_dates.to_numpy(dtype="datetime64[D]"),
                            basin_ids=np.array(ids), area_gov_km2=area, qmax_numeric_m3_s=m3_s)
        np.savez_compressed(a3 / "A3_persistence_predictions.npz", dates=dates.to_numpy(dtype="datetime64[D]"),
                            source_dates=full_dates[:-1].to_numpy(dtype="datetime64[D]"), basin_ids=np.array(ids),
                            obs_source_mm_day=physical[1:], persistence_qmax_mm_day=physical[:-1],
                            persistence_qmean_aux_mm_day=physical[:-1] * .8)
        context, _ = read_context(a3)
        check("context_reconstructs_daily_persistence", np.array_equal(context["persistence"], physical[:-1], equal_nan=True))
        level_table = pd.DataFrame(dict(basin_id=ids, reference_start="1981-01-01", reference_end="2013-12-31",
                                       q30_mm_day=[.8, 1.5], q80_mm_day=[1.2, 2.3], q98_mm_day=[1.35, 2.7]))
        level_table.to_csv(a1 / "A1_P10_qmax_level_thresholds.csv", index=False)
        flood_table = pd.DataFrame([dict(basin_id=i, policy="half_calendar_primary", return_period_years=p,
                                         threshold_mm_day=1 + .1 * p, n_years=8 if i == "1" else 30, fit_status="fit_defined")
                                    for i, p in itertools.product(ids, RETURN_PERIODS)])
        flood_table.to_csv(a1 / "A1_P10_flood_return_levels.csv", index=False)
        levels, floods, _ = read_thresholds(a1, ids)
        check("threshold_primary_policy_and_order", len(floods) == 8 and tuple(levels.index) == ids)
        for name in A2_RUNS:
            directory = a2 / "runs" / name
            directory.mkdir(parents=True)
            (directory / "config.yml").write_text(yaml.safe_dump({"clip_targets_to_zero": ["qmax"]}), encoding="utf-8")
            saved = {i: {"1D": {"xr": xr.Dataset(
                {"qmax_obs": (("date", "time_step"), physical[1:, j:j + 1]),
                 "qmax_sim": (("date", "time_step"), physical[1:, j:j + 1])},
                coords={"date": dates, "time_step": [0]})}} for j, i in enumerate(ids)}
            with (directory / "test_results.p").open("wb") as stream:
                pickle.dump(saved, stream)
        models, _ = read_a2(a2, context)
        check("six_A2_pickles_read", len(models) == 6 and all(np.array_equal(f.sim, context["obs"], equal_nan=True) for f in models))
        for seed in (111, 222, 333):
            directory = a3 / "dlinear" / f"seed_{seed}"
            directory.mkdir(parents=True)
            np.savez_compressed(directory / "test_predictions.npz", dates=dates.to_numpy(dtype="datetime64[D]"),
                                source_last_dates=full_dates[:-1].to_numpy(dtype="datetime64[D]"), basin_ids=np.array(ids),
                                area_gov_km2=area, obs_mm_day=physical[1:], sim_mm_day=physical[1:], sim_raw_mm_day=physical[1:],
                                seed=seed, selected_epoch=1, period="test", config_sha256="fixture_only")
        linear, _ = read_dlinear(a3, context)
        models.extend(linear)
        models.append(Forecast("Persistence_qmax", sim=context["persistence"], raw=context["persistence"], **context).validate())
        check("three_DLinear_exports_read", [f.seed for f in linear] == [111, 222, 333])
        tables, shared = evaluate_group(models, levels, floods)
        check("I_O_to_scoring_keeps_calendar", shared.shape == (1461, 2) and int(shared.sum()) == 2920)
        perfect = tables["metrics"].loc[tables["metrics"].scope.eq("common_source_all") & tables["metrics"].run.eq("SeqLSTM_Q")]
        check("I_O_perfect_scores", np.allclose(perfect[list(METRICS)].to_numpy(), 1))
        check("pooled_levels_from_counts", np.allclose(pool_levels(tables["confusion"]).loc[lambda x: x.run.eq("SeqLSTM_Q"), "accuracy"], 1))
        check("events_short_fit_cohort_kept", set(pool_events(tables["events"]).cohort) == {"all", "fit_ge10_years", "fit_lt10_years"})
        check("primary_and_supplementary_event_tables_separate",
              set(tables["events"].matching_tolerance_days) == {0}
              and set(tables["events_tolerance1"].matching_tolerance_days) == {1}
              and list(tables["event_matches"].columns) == list(tables["event_matches_tolerance1"].columns)
              and (tables["event_matches"].obs_onset == tables["event_matches"].sim_onset).all())
        for lead in range(1, 8):
            lead_model = replace(models[0], lead=lead, metadata={"forecast_horizon": 7}).validate()
            lead_tables, _ = evaluate_group([lead_model], levels, floods)
            check(f"H_lead{lead}_separate_table_dispatch",
                  len(lead_tables["events"]) == len(ids) * len(RETURN_PERIODS) * 2
                  and len(lead_tables["events_tolerance1"]) == len(lead_tables["events"])
                  and set(lead_tables["events"].matching_tolerance_days) == {lead // 3}
                  and set(lead_tables["events_tolerance1"].matching_tolerance_days) == {1}
                  and (lead // 3 != 1 or lead_tables["events"].equals(lead_tables["events_tolerance1"])))
        test_table = pairwise_tests(tables["metrics"])
        check("Holm_all_pairs_family", set(test_table.family_size) == {45} and set(test_table.planned_family_size) == {45})
        partial = tables["metrics"].copy()
        partial.loc[partial.run.eq("SeqLSTM_Q"), ["NSE", "PNSE", "KGE"]] = np.nan
        partial_tests = pairwise_tests(partial)
        check("Holm_table_family_counts_only_finite_tests", set(partial_tests.family_size) == {36}
              and set(partial_tests.planned_family_size) == {45}
              and partial_tests.loc[partial_tests.p_value.isna(), "p_holm"].isna().all())
        tables["metrics"].to_csv(a3 / "A3_all_metrics_per_basin.csv", index=False)
        checked, _ = check_a3_reference(tables["metrics"], a3)
        check("A3_reference_read_and_join", bool(checked.passed.all()))
        for lead in range(1, 8):
            contract_path = root / f"lead{lead}.npz"
            target = dates[lead - 1:].to_numpy(dtype="datetime64[D]")
            meta = dict(run="future_seed_111", period="test", scenario="S0", lead=lead, forecast_horizon=7, seed=111,
                        unit="mm/day_area_gov", config_sha256="fixture_only", library_commit="fixture_only", clip_to_zero=True)
            payload = dict(dates=target, origin_dates=target - np.timedelta64(lead - 1, "D"),
                           persistence_source_dates=target - np.timedelta64(lead, "D"), basin_ids=np.array(ids),
                           area_gov_km2=area, obs_mm_day=physical[lead:], sim_raw_mm_day=physical[lead:],
                           sim_mm_day=physical[lead:], persistence_mm_day=physical[:-lead])
            np.savez_compressed(contract_path, **payload, metadata_json=json.dumps(meta))
            future = read_contract(contract_path)
            check(f"H_lead{lead}_persistence_at_origin_minus_one", future.lead == lead
                  and np.array_equal(future.persistence, physical[:-lead], equal_nan=True))
        for key, value in (("lead", True), ("forecast_horizon", True), ("lead", 7.0),
                           ("forecast_horizon", 6), ("forecast_horizon", None)):
            invalid_meta = dict(meta, **{key: value})
            np.savez_compressed(contract_path, **payload, metadata_json=json.dumps(invalid_meta))
            rejects(f"contract_{key}_{value!r}_rejected", lambda: read_contract(contract_path))
    return pd.DataFrame(checks)


# %% [markdown]
# ## Phần 10 — Chấm thực nghiệm và xuất kết quả
# ### 10.1. Phạm vi riêng và giao chung
# Giữ đủ ID và trạng thái. Bảng chính common_source_all dùng cùng ngày cho mọi run.

# %% Phần 10.1 — Chấm theo lưu vực
# native không cấp persistence; paired_source và common_source_all dùng sáu chỉ số cùng mask.
def evaluate_group(forecasts, levels=None, floods=None):
    shared = common_mask(forecasts)
    metric_rows, level_rows, matrix_rows = [], [], []
    # Vai trò bảng độc lập với số dung sai: hạn 3–5 có cả hai bảng cùng ±1.
    event_rows = {role: [] for role in ("primary", "tolerance1")}
    match_rows = {role: [] for role in event_rows}
    for forecast in forecasts:
        horizon = forecast.metadata.get("forecast_horizon", forecast.lead)
        primary_tolerance = primary_event_tolerance_days(forecast.lead, horizon)
        base_meta = dict(run=forecast.name, period=forecast.period, scenario=forecast.scenario,
                         lead=forecast.lead, seed=forecast.seed)
        native_obs = forecast.native_obs if forecast.native_obs is not None else forecast.obs
        scopes = {"native": np.isfinite(native_obs) & np.isfinite(forecast.sim),
                  "paired_source": np.isfinite(forecast.obs) & np.isfinite(forecast.sim) & np.isfinite(forecast.persistence),
                  "common_source_all": shared}
        for scope, mask in scopes.items():
            for j, identifier in enumerate(forecast.ids):
                observed = native_obs[:, j] if scope == "native" else forecast.obs[:, j]
                pers = None if scope == "native" else forecast.persistence[:, j]
                row = dict(**base_meta, scope=scope, basin_id=identifier,
                           raw_negative_days=int((forecast.raw[:, j][mask[:, j]] < 0).sum()),
                           **metrics(observed, forecast.sim[:, j], pers, mask[:, j]))
                row.update(signatures(observed, forecast.sim[:, j], mask[:, j]))
                metric_rows.append(row)
        for j, identifier in enumerate(forecast.ids):
            obs, sim, mask = forecast.obs[:, j], forecast.sim[:, j], shared[:, j]
            meta = dict(**base_meta, scope="common_source_all", basin_id=identifier)
            if levels is not None:
                bounds = levels.loc[identifier, ["q30_mm_day", "q80_mm_day", "q98_mm_day"]].to_numpy(dtype=float)
                matrix = confusion(obs, sim, bounds, mask)
                level_rows.extend(dict(**meta, **row) for row in class_scores(matrix))
                matrix_rows.extend(dict(**meta, observed_level=LEVEL_NAMES[a], predicted_level=LEVEL_NAMES[b], days=int(matrix[a, b]))
                                   for a, b in itertools.product(range(4), repeat=2))
            if floods is not None:
                for period in RETURN_PERIODS:
                    threshold = floods.loc[(identifier, period)]
                    for grouped in (True, False):
                        event_meta = dict(**meta, return_period_years=period, grouping="USWRC" if grouped else "consecutive",
                                          n_fit_years=int(threshold.n_years), threshold_mm_day=float(threshold.threshold_mm_day))
                        for role, tolerance in (("primary", primary_tolerance),
                                                ("tolerance1", SUPPLEMENTARY_EVENT_TOLERANCE_DAYS)):
                            score, pairs = score_events(obs, sim, threshold.threshold_mm_day, forecast.area[j], mask,
                                                        lead=forecast.lead, grouped=grouped, tolerance=tolerance,
                                                        forecast_horizon=horizon)
                            event_rows[role].append(dict(**event_meta, **score))
                            for pair in pairs:
                                match_rows[role].append(dict(**event_meta, matching_tolerance_days=tolerance,
                                    **{key: str(forecast.dates[value].date()) if key != "peak_timing_days" else value
                                       for key, value in pair.items()}))
    # Giữ schema CSV khi không có cặp ghép đúng ngày.
    match_columns = ["run", "period", "scenario", "lead", "seed", "scope", "basin_id", "return_period_years",
                     "grouping", "n_fit_years", "threshold_mm_day", "matching_tolerance_days",
                     "obs_onset", "sim_onset", "obs_peak", "sim_peak", "peak_timing_days"]
    return dict(metrics=pd.DataFrame(metric_rows), levels=pd.DataFrame(level_rows), confusion=pd.DataFrame(matrix_rows),
                events=pd.DataFrame(event_rows["primary"]),
                event_matches=pd.DataFrame(match_rows["primary"], columns=match_columns),
                events_tolerance1=pd.DataFrame(event_rows["tolerance1"]),
                event_matches_tolerance1=pd.DataFrame(match_rows["tolerance1"], columns=match_columns)), shared


def summarize(table):
    keys = ["run", "period", "scenario", "lead", "scope"]
    rows = []
    for values, part in table.groupby(keys, sort=False, dropna=False):
        for metric in METRICS + SIGNATURES:
            valid = part.loc[np.isfinite(part[metric]), metric]
            rows.append(dict(zip(keys, values), metric=metric, total_IDs=len(part), finite_IDs=len(valid),
                             mean=float(valid.mean()) if len(valid) else np.nan,
                             median=float(valid.median()) if len(valid) else np.nan,
                             median_absolute=float(valid.abs().median()) if len(valid) else np.nan,
                             **{f"P{p}": float(valid.quantile(p / 100)) if len(valid) else np.nan for p in (10, 25, 75, 90)}))
    return pd.DataFrame(rows)


def pairwise_tests(table):
    rows = []
    common = table.loc[table.scope.eq("common_source_all")]
    for group, part in common.groupby(["period", "scenario", "lead"], sort=False):
        names = sorted(set(part.run) - {"Persistence_qmean_aux"})
        for metric in ("NSE", "PNSE", "KGE"):
            wide = part.pivot(index="basin_id", columns="run", values=metric)
            family = []
            for left, right in itertools.combinations(names, 2):
                family.append(dict(period=group[0], scenario=group[1], lead=group[2], metric=metric,
                                   left=left, right=right, **paired_test(wide[left], wide[right])))
            adjusted = holm([row["p_value"] for row in family])
            valid_family_size = sum(np.isfinite(row["p_value"]) for row in family)
            for row, p in zip(family, adjusted):
                row.update(p_holm=p, family_size=int(valid_family_size), planned_family_size=len(family),
                           significant_005=bool(p <= .05) if np.isfinite(p) else None)
            rows.extend(family)
    return pd.DataFrame(rows)


def pool_levels(confusions):
    if confusions.empty:
        return pd.DataFrame()
    keys = ["run", "period", "scenario", "lead", "scope"]
    rows = []
    for group, part in confusions.groupby(keys, sort=False):
        matrix = (part.groupby(["observed_level", "predicted_level"]).days.sum()
                  .unstack().reindex(index=LEVEL_NAMES, columns=LEVEL_NAMES).to_numpy(dtype=int))
        rows.extend(dict(zip(keys, group), **item) for item in class_scores(matrix))
    return pd.DataFrame(rows)


def pool_events(events):
    if events.empty:
        return pd.DataFrame()
    keys = ["run", "period", "scenario", "lead", "return_period_years", "grouping", "matching_tolerance_days"]
    rows = []
    for group, part in events.groupby(keys, sort=False):
        cohorts = {"all": part, "fit_ge10_years": part.loc[part.n_fit_years.ge(10)],
                   "fit_lt10_years": part.loc[part.n_fit_years.lt(10)]}
        for cohort, selected in cohorts.items():
            counts = {key: int(selected[key].sum()) for key in ("hit", "missed", "false_alarm", "observed_censored", "predicted_censored")}
            rows.append(dict(zip(keys, group), cohort=cohort, IDs=len(selected),
                             IDs_with_observed_events=int((selected.hit + selected.missed > 0).sum()),
                             IDs_with_predicted_events=int((selected.hit + selected.false_alarm > 0).sum()),
                             **counts, **event_rates(counts["hit"], counts["missed"], counts["false_alarm"]),
                             mean_peak_timing_days=ratio(selected.timing_signed_sum.sum(), counts["hit"]),
                             mean_abs_peak_timing_days=ratio(selected.timing_abs_sum.sum(), counts["hit"])))
    return pd.DataFrame(rows)


# %% [markdown]
# ### 10.2. Đối chiếu chỉ số liên tục với A3
# Kiểm từng ID trên giao chung, không dùng bảng kết quả chép sẵn để thay phép tính.

# %% Phần 10.2 — Kiểm A3 và vẽ phân phối
# Hình tiếng Anh; dữ liệu vẽ xuất CSV cùng bảng chính, giữ cả điểm âm.
def check_a3_reference(table, a3_root):
    path = unique_file(a3_root, "A3_all_metrics_per_basin.csv")
    reference = pd.read_csv(path, dtype={"basin_id": str})
    reference["basin_id"] = reference.basin_id.map(basin_id)
    reference = reference.loc[reference.scope.eq("common_source_all")]
    actual = table.loc[table.scope.eq("common_source_all") & table.period.eq("test") & table.scenario.eq("S0") & table.lead.eq(1)]
    keys = ["run", "basin_id"]
    ref, computed = reference.set_index(keys, verify_integrity=True), actual.set_index(keys, verify_integrity=True)
    if set(ref.index) != set(computed.index):
        raise ValueError("Tập run–ID đối chiếu A3 khác nhau")
    ref = ref.reindex(computed.index)
    rows = []
    for metric in METRICS:
        left, right = computed[metric].to_numpy(), ref[metric].to_numpy()
        same = np.isclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=True)
        rows.append(dict(metric=metric, checked_rows=len(left), matched_rows=int(same.sum()), passed=bool(same.all())))
    rows.append(dict(metric="n", checked_rows=len(actual), matched_rows=int((computed.n == ref.n).sum()),
                     passed=bool((computed.n == ref.n).all())))
    result = pd.DataFrame(rows)
    if not result.passed.all():
        raise AssertionError("Sai khác A3; kiểm scope/clip/đơn vị trước khi dùng bảng đánh giá")
    return result, path


# %% [markdown]
# ### 10.3. Hình CDF trên giao ngày chung
# Ba seed A4 cùng màu với LSTM nền và khác kiểu nét; chú giải ngoài vùng vẽ.

# %% Phần 10.3 - Vẽ hình và lưu các điểm CDF
# Giữ toàn bộ điểm hữu hạn, kể cả đuôi âm; số n lấy riêng cho từng run/chỉ số.
def draw_figures(table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for group, part in table.loc[table.scope.eq("common_source_all")].groupby(["period", "scenario", "lead"], sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        styles = {"SeqLSTM_Q": ("#0072B2", "-"), "SeqLSTM_noQ": ("#56B4E9", "--"),
                  "EncDec_Q": ("#009E73", "-"), "EncDec_noQ": ("#E69F00", "--"),
                  "D_FC": ("#9467BD", "-"), "D_RA": ("#CC79A7", "-"),
                  "Persistence_qmax": ("#333333", "--"), "Persistence_qmean_aux": ("#888888", ":")}
        for seed, linestyle in zip(A4_SEEDS, ("--", "-.", ":")):
            styles[f"SeqLSTM_Q_A4_seed{seed}"] = ("#0072B2", linestyle)
            styles[f"DLinear_seed_{seed}"] = ("#D55E00", linestyle)
        for panel, (ax, metric) in enumerate(zip(axes, ("NSE", "PNSE", "KGE"))):
            for run, data in part.groupby("run", sort=False):
                finite = np.sort(data.loc[np.isfinite(data[metric]), metric].to_numpy())
                if len(finite):
                    y = np.arange(1, len(finite) + 1) / len(finite)
                    color, linestyle = styles.get(run, (None, "-"))
                    label = run.replace("SeqLSTM_Q_A4_seed", "SeqLSTM_Q A4 ")
                    ax.step(finite, y, where="post", color=color, linestyle=linestyle,
                            label=f"{label} (n={len(finite)})")
                    rows.extend(dict(period=group[0], scenario=group[1], lead=group[2], run=run, metric=metric,
                                     value=float(x), cumulative_fraction=float(p)) for x, p in zip(finite, y))
            ax.set_xscale("symlog", linthresh=1)
            ax.set(xlabel=metric, ylabel="Fraction of finite basin scores", ylim=(0, 1.02))
            ax.set_title(f"({chr(97+panel)}) {metric}")
            ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, .5))
        fig.suptitle(f"Common daily mask | {group[0]} | {group[1]} | lead {group[2]} | all finite scores retained")
        fig.tight_layout()
        fig.savefig(output / f"Evaluation_CDF_{group[0]}_{group[1]}_lead{group[2]}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Hình: phân phối NSE, PNSE, KGE theo lưu vực trên giao ngày chung; kỳ {group[0]}, "
              f"kịch bản {group[1]}, hạn {group[2]}, {part.run.nunique()} run. "
              "Đường càng về phải có điểm càng cao; n là số lưu vực có chỉ số hữu hạn; giữ toàn bộ điểm âm.")
    return pd.DataFrame(rows)


# %% [markdown]
# ## Phần 11 — Chạy toàn bộ và kết luận
# ### 11.1. Input, chạy kiểm thử và đánh giá
# Add Input Output A1, A2, A3 và A4 Compare Version 2; dataset gốc không bắt buộc.
# Notebook tự đọc đủ ba seed A4; script có thể chỉ rõ --a1, --a2, --a3, --a4.

# %% Phần 11.1 — Điều phối và lưu kết quả trực tiếp
# Mọi dữ liệu giả chỉ ở self_test; trạng thái thực nghiệm chỉ đặt sau khi đọc/chấm thành công.
def run(input_root, output_root, a1_root=None, a2_root=None, a3_root=None, extra_contracts=(), *,
        a4_root=None, expected_ids=451):
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "Evaluation_summary.json").write_text(
        json.dumps(dict(complete=False, status="incomplete", expected_A4_seeds=list(A4_SEEDS)), indent=2),
        encoding="utf-8")
    checks = self_test()
    checks.to_csv(output / "Evaluation_checks.csv", index=False, encoding="utf-8-sig")
    a3_root = Path(a3_root) if a3_root else unique_file(input_root, "A3_persistence_predictions.npz").parent
    a2_root = Path(a2_root) if a2_root else find_a2_root(input_root)
    a1_root = Path(a1_root) if a1_root else unique_file(input_root, "A1_P10_qmax_level_thresholds.csv").parent
    a4_root = unique_file(a4_root or input_root, "a4_summary.json").parent
    context, paths = read_context(a3_root)
    context_paths = list(paths)
    if len(context["ids"]) != expected_ids:
        raise ValueError(f"Danh sách có {len(context['ids'])} ID, kỳ vọng {expected_ids}")
    print("Phần 1–2: kiểm thử đạt; nguồn A1/A2/A3/A4 Compare:", a1_root, a2_root, a3_root, a4_root, sep="\n")
    levels, floods, threshold_paths = read_thresholds(a1_root, context["ids"])
    paths.extend(threshold_paths)
    forecasts, read_paths = read_a2(a2_root, context)
    paths.extend(read_paths)
    dlinear, read_paths = read_dlinear(a3_root, context)
    forecasts.extend(dlinear)
    paths.extend(read_paths)
    forecasts.append(Forecast(name="Persistence_qmax", sim=context["persistence"], raw=context["persistence"], **context).validate())
    pers_path = unique_file(a3_root, "A3_persistence_predictions.npz")
    with np.load(pers_path, allow_pickle=False) as data:
        auxiliary = np.array(data["persistence_qmean_aux_mm_day"], dtype=float)
    forecasts.append(Forecast(name="Persistence_qmean_aux", sim=auxiliary, raw=auxiliary, **context).validate())
    a3_only = list(forecasts)
    a4_forecasts, read_paths, a4_input_check, a4_contract_check = read_a4(a4_root, context, context_paths)
    forecasts.extend(a4_forecasts)
    paths.extend(read_paths)
    show_table("Phần 8.4 - Manifest A4 và nguồn A3", a4_input_check[["kind", "file", "passed"]])
    show_table("Phần 8.4 - Ba hợp đồng LSTM A4", a4_contract_check[[
        "run", "seed", "selected_epoch", "IDs", "target_days", "finite_prediction_day_IDs",
        "paired_obs_prediction_day_IDs", "sim_raw_negative_day_IDs", "canonical_context_matches"]])
    for path in extra_contracts:
        forecast = read_contract(path)
        if forecast.ids != context["ids"]:
            raise ValueError("Hợp đồng mở rộng cần cùng danh sách/thứ tự 451 ID")
        forecasts.append(forecast)
        paths.append(Path(path))
    identities = [(f.name, f.period, f.scenario, f.lead) for f in forecasts]
    if len(set(identities)) != len(identities):
        raise ValueError("Trùng run/kỳ/kịch bản/hạn; tên run cần phân biệt seed")
    grouped = {}
    for forecast in forecasts:
        grouped.setdefault(forecast.group, []).append(forecast)
    for members in grouped.values():
        if not any(f.name == "Persistence_qmax" for f in members):
            base = members[0]
            baseline = replace(base, name="Persistence_qmax", seed=None, native_obs=None,
                               sim=base.persistence, raw=base.persistence,
                               metadata={"role": "observed_Q_reference_for_scoring"}).validate()
            members.append(baseline)
            forecasts.append(baseline)
    results = {key: [] for key in ("metrics", "levels", "confusion", "events", "event_matches",
                                  "events_tolerance1", "event_matches_tolerance1")}
    coverages = []
    for group, members in grouped.items():
        print(f"Phần 3–6: chấm {group}; {len(members)} run, giữ {len(members[0].ids)} ID", flush=True)
        tables, shared = evaluate_group(members, levels, floods)
        for key in results:
            results[key].append(tables[key])
        coverages.append(dict(period=group[0], scenario=group[1], lead=group[2], runs=len(members),
                              common_day_IDs=int(shared.sum()), IDs_with_days=int(shared.any(axis=0).sum())))
    for key, tables in results.items():
        results[key] = pd.concat(tables, ignore_index=True)
    print("Phần 7: so sánh theo cặp và tổng hợp sự kiện", flush=True)
    results["summary"] = summarize(results["metrics"])
    results["paired_tests"] = pairwise_tests(results["metrics"])
    results["levels_pooled"] = pool_levels(results["confusion"])
    results["events_pooled"] = pool_events(results["events"])
    results["events_pooled_tolerance1"] = pool_events(results["events_tolerance1"])
    results["scenario_changes"] = scenario_changes(forecasts)
    results["coverage"] = pd.DataFrame(coverages)
    results["A4_input_check"] = a4_input_check
    results["A4_contract_check"] = a4_contract_check
    reference_table = results["metrics"]
    if a4_forecasts or extra_contracts:
        shared = common_mask(a3_only)
        reference_table = pd.DataFrame([
            dict(run=f.name, period="test", scenario="S0", lead=1, scope="common_source_all", basin_id=identifier,
                 **metrics(f.obs[:, j], f.sim[:, j], f.persistence[:, j], shared[:, j]))
            for f in a3_only for j, identifier in enumerate(f.ids)])
    reference, reference_path = check_a3_reference(reference_table, a3_root)
    paths.append(reference_path)
    results["A3_reference_check"] = reference
    results["CDF_points"] = draw_figures(results["metrics"], output)
    for name, table in results.items():
        table.to_csv(output / f"Evaluation_{name}.csv", index=False, encoding="utf-8-sig")
    input_manifest = pd.DataFrame([dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path)) for path in sorted(set(paths))])
    input_manifest.to_csv(output / "Evaluation_input_manifest.csv", index=False, encoding="utf-8-sig")
    actual_leads = sorted({f.lead for f in forecasts})
    applied_tolerances = {str(lead): primary_event_tolerance_days(lead) for lead in actual_leads}
    # Giữ trường scalar thuận tiện cho lần chấm một hạn; nhiều hạn dùng bản đồ.
    scalar_tolerance = next(iter(applied_tolerances.values())) if len(actual_leads) == 1 else None
    protocol = dict(protocol_version=PROTOCOL_VERSION, metric_ddof=0, same_mask_for_six_metrics=True, metric_epsilon=None,
                    level_equality="higher_class", flood_exceedance="strictly_greater",
                    FDC_log_floor_mm_day=LOG_FLOOR_MM_DAY, FDC_quantile_method="linear",
                    FDC_tail_count="ceil_fraction_times_n", event_grouping="USWRC and consecutive",
                    event_matching="maximum_cardinality_then_minimum_total_absolute_onset_error",
                    event_tolerance=PRIMARY_EVENT_TOLERANCE_POLICY,
                    event_tolerance_days=scalar_tolerance,
                    event_tolerance_days_by_lead=applied_tolerances,
                    supported_event_tolerance_days_by_lead={str(lead): primary_event_tolerance_days(lead)
                                                           for lead in range(1, 8)},
                    supplementary_event_tolerance_days=SUPPLEMENTARY_EVENT_TOLERANCE_DAYS,
                    event_timing_reference=dict(
                        NOAA="https://repository.library.noaa.gov/view/noaa/30340/noaa_30340_DS1.pdf",
                        NOAA_window="plus_minus_one_third_forecast_lead_time_24h_means_8h",
                        daily_adaptation="floor_actual_lead_divided_by_3_whole_days_not_subdaily_verification",
                        forecast_horizon_role="contract_validation_only_does_not_set_matching_tolerance",
                        Nearing2024="https://www.nature.com/articles/s41586-024-07145-1",
                        Nearing2024_window_days=2,
                        supplementary_note="one_day_sensitivity_is_not_exact_Nearing_replication"),
                    censored_events="excluded_from_scores_and_counted_separately",
                    holm_family="all_unordered_run_pairs_excluding_qmean_aux_per_metric_period_scenario_lead",
                    holm_family_size="finite_p_values_only_NaN_excluded_planned_size_reported_separately",
                    A4_compare_version_expected=A4_COMPARE_VERSION,
                    A4_input_verification="manifest_SHA256_and_same_A3_context",
                    A3_reference_scope="original_11_runs_common_mask_recomputed_separately",
                    statistics="two_sided_Wilcoxon_auto_and_paired_Cohen_dz",
                    test_used_for_selection=False)
    (output / "Evaluation_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    run_index = [dict(run=f.name, period=f.period, scenario=f.scenario, lead=f.lead, seed=f.seed,
                      IDs=len(f.ids), days=len(f.dates), **{"source_metadata": f.metadata}) for f in forecasts]
    (output / "Evaluation_run_index.json").write_text(json.dumps(run_index, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(complete=True, protocol_version=PROTOCOL_VERSION,
                   event_tolerance=PRIMARY_EVENT_TOLERANCE_POLICY,
                   event_tolerance_days=scalar_tolerance,
                   event_tolerance_days_by_lead=applied_tolerances,
                   supplementary_event_tolerance_days=SUPPLEMENTARY_EVENT_TOLERANCE_DAYS,
                   fixture_checks=int(len(checks)), A3_reference_matched=True,
                   A4_contracts_matched=bool(a4_contract_check.canonical_context_matches.all()),
                   A4_runs=[f.name for f in a4_forecasts],
                   A4_compare_version_expected=A4_COMPARE_VERSION,
                   coverage=coverages, scenarios=sorted({f.scenario for f in forecasts}),
                   leads=sorted({f.lead for f in forecasts}),
                   operational_scenarios_run=any(f.scenario != "S0" for f in forecasts),
                   multi_lead_run=any(f.lead > 1 for f in forecasts),
                   python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                   scipy=__import__("scipy").__version__, output=str(output))
    (output / "Evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Phần 11: hoàn thành; lưu trực tiếp cho Kaggle Save Version.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("GỬI LẠI: hai bảng Phần 8.4, log số run/giao ngày và kết luận Phần 11; "
          "giữ các CSV Evaluation_A4_input_check, Evaluation_A4_contract_check, coverage, summary và paired_tests.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Chấm dự báo đã lưu; không huấn luyện và không đóng gói ZIP")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working/evaluation_output")
    parser.add_argument("--a1")
    parser.add_argument("--a2")
    parser.add_argument("--a3")
    parser.add_argument("--a4")
    parser.add_argument("--extra-contract", action="append", default=[])
    args = parser.parse_args([] if "ipykernel" in sys.modules else None)
    if args.self_test:
        print(self_test().to_string(index=False))
    else:
        run(args.input, args.output, args.a1, args.a2, args.a3, args.extra_contract, a4_root=args.a4)


if __name__ == "__main__":
    main()
