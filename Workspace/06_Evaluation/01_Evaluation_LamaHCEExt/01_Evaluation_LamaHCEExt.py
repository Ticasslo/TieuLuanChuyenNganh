# %% [markdown]
# # Bộ đánh giá dự báo lưu lượng - Extended LamaH-CE
#
# Notebook đọc dự báo đã lưu, kiểm đầu vào và tính cùng một giao thức cho các mốc A2, DLinear, persistence, LSTM A4, B0, B1a, B1b, B2 và hai mức B3. Bộ đánh giá không huấn luyện mô hình.
#
# **Phạm vi:** test 2014–2017, giữ 451 lưu vực. Bảng chính chấm 56 run S0/hạn 1. Thư mục C2_scenarios chấm bảy kịch bản C2. Khi gắn H Compare hoặc D Compare, thư mục `H/` và `D/` chấm riêng các phân vùng mở rộng mà không đổi bốn bảng S0 56 run.
#
# ## Cài đặt và đầu vào
#
# Dùng CPU, Internet Off. Giữ chín Notebook Output của Version 9 và thêm toàn bộ Output `04_C2_LamaHCEExt_Scenarios` Version 1 đã hoàn thành.
#
# | Output notebook | Version | Vai trò |
# |---|---:|---|
# | `01_A1_LamaHCEExt_Explore` | 3 | Ngưỡng bốn mức và ngưỡng sự kiện |
# | `01_A2_LamaHCEExt_Reproduce` | 2 | Dự báo sáu run từ trọng số tác giả |
# | `02_A3_LamaHCEExt_Benchmarks` | 1 | Quan trắc chuẩn, persistence và ba seed DLinear |
# | `04_A4_LamaHCEExt_Compare` | 2 | Ba seed A4 và manifest |
# | `02_B0_LamaHCEExt_Compare` | 1 | Ba seed B0 và manifest |
# | `05_B1_LamaHCEExt_Compare` | 1 | Ba seed B1a, ba seed B1b và manifest |
# | `07_B2_LamaHCEExt_Compare` | 1 | Ba seed B2, audit nguồn, manifest và NPZ |
# | `09_B3_LamaHCEExt_Compare` | 1 | Hai mức che B3, mỗi mức ba seed, sáu NPZ |
# | `13_C2_LamaHCEExt_Compare` | 1 | 24 NPZ C2, manifest và bảng bàn giao |
# | `04_C2_LamaHCEExt_Scenarios` | 1 | 144 NPZ kịch bản, mặt nạ chung, audit S0, summary và manifest |
# | `15_H_LamaHCEExt_Compare` | 1 | Gắn khi chấm H. Thư mục `h_compare/` có 168 NPZ, manifest và bảng bàn giao |
# | `17_D_LamaHCEExt_Compare` | 1 | Gắn khi chấm D (`evaluation_output/D/`). Thư mục `d_compare/` có 6 hoặc 9 NPZ 730 bước, mặt nạ chung và manifest |
#
# Mã tự tìm tệp trong `/kaggle/input`, kiểm manifest/sha256 và dừng khi thiếu, trùng nguồn hoặc sai hợp đồng. Các bảng CSV, metadata JSON và hình được lưu trực tiếp vào `/kaggle/working/evaluation_output`.
#
# ## Mục lục
#
# 1. [Môi trường và quy ước](#evaluation-part-1)
# 2. [Hợp đồng dự báo và kiểm đầu vào](#evaluation-part-2)
# 3. [Các chỉ số liên tục](#evaluation-part-3)
# 4. [Đánh giá bốn mức](#evaluation-part-4)
# 5. [Trích đoạn vượt ngưỡng và gộp USWRC](#evaluation-part-5)
# 6. [Ghép sự kiện một–một](#evaluation-part-6)
# 7. [So sánh mô hình và kịch bản](#evaluation-part-7)
# 8. [Đọc và kiểm Output A1/A2/A3/A4/B0/B1/B2/B3](#evaluation-part-8)
# 9. [Kiểm thử độc lập](#evaluation-part-9)
# 10. [Chấm thực nghiệm và xuất kết quả](#evaluation-part-10)
# 11. [Chạy toàn bộ và kết luận](#evaluation-part-11)

# %% [markdown]
# <a id="evaluation-part-1"></a>
#
# ## Phần 1 — Môi trường và quy ước

# %% [markdown]
# ### 1.1. Thư viện và hằng số
#
# Đơn vị chấm là mm/ngày theo area_gov. Các phép tính chỉ số dùng float64.
# S0 là kịch bản không che thêm đầu vào; dữ liệu gốc vẫn có thể thiếu.
# Hạn k là bước ngày thứ k tính từ ngày đích đầu tiên; hạn 1 dự báo ngày kế tiếp lịch sử.
# Một ngày–lưu vực là một ngày tại một ID. Mặt nạ là bảng True/False chọn các cặp này.
# Giao chung chỉ giữ ngày–lưu vực có obs, persistence và dự báo của mọi run cùng hữu hạn.

# %% Phần 1.1 - Thư viện và hằng số
# Không cài hay thay phiên bản môi trường khi chạy đánh giá.
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
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
B0_SEEDS = (111, 222, 333)
B0_SEED_VERSIONS = {111: 1, 222: 1, 333: 1}
B0_COMPARE_VERSION = 1
B0_LIBRARY_COMMIT = "f229e02bf98cbd979835a4b6cd503d0e3adb8bc3"
B1_STEPS = ("B1a", "B1b")
B1_SEEDS = (111, 222, 333)
B1_SEED_VERSIONS = {111: 1, 222: 1, 333: 1}
B1_COMPARE_VERSION = 1
B1_LIBRARY_COMMIT = "f229e02bf98cbd979835a4b6cd503d0e3adb8bc3"
B2_SEEDS = (111, 222, 333)
B2_COMPARE_VERSION = 1
B2_LIBRARY_COMMIT = "2ccaa147a6ec87d0f56fa3e2f11b1bf38079253f"
B3_SEEDS = (111, 222, 333)
B3_COMPARE_VERSION = 1
B3_LIBRARY_COMMIT = B2_LIBRARY_COMMIT
FREEZE_C = "db6dc1ffcd8c14994cd80be3ed8ff417002c01a1"
FREEZE_H = "6f68f471bec255f67d446acd6c6556358ff9bfea"
C2_NOTEBOOK = "13_C2_LamaHCEExt_Compare"
C2_CONFIGS = ("lstm_L2_x1_lr1e-3", "gru_L2_x1_lr1e-3", "transformer_L2_x1_lr1e-3",
              "transformer_L2_x3_lr5e-4", "s4d_L4_x1_lr1e-3", "s4d_L4_x3_lr1e-3",
              "mamba_L2_x1_lr1e-3", "mamba_L4_x3_lr1e-3")
C2_SEEDS = (111, 222, 333)
H_CONFIGS = ("lstm_L2_x1_lr1e-3", "mamba_L2_x1_lr1e-3")
H_LIBRARY_COMMITS = {
    "past": FREEZE_H,
    "hybrid": FREEZE_H,
    "gefs": "840450f85958538903b16f281ca432d40e0a88e1",
    "perfect": FREEZE_H,
}
H_MODES = tuple(H_LIBRARY_COMMITS)
H_LEADS = tuple(range(1, 8))
C2_SCENARIOS = ("S0", "S1_Q1", "S1_Q3", "S1_Q7", "S2", "S3", "S4")
C2_SCENARIOS_NOTEBOOK = "04_C2_LamaHCEExt_Scenarios"
D_CONFIGS = ("lstm_L2_x1_lr1e-3", "mamba_L2_x1_lr1e-3", "s4d_L4_x1_lr1e-3")
D_COMPARE_NOTEBOOK = "17_D_LamaHCEExt_Compare"
D_NAME = re.compile(
    rf"(?P<run>SF_(?P<config>{'|'.join(map(re.escape, D_CONFIGS))})"
    rf"_D_seed(?P<seed>{'|'.join(map(str, C2_SEEDS))}))_test_S0_lead1\.npz"
)
H_NAME = re.compile(
    rf"(?P<run>SF_(?P<config>{'|'.join(map(re.escape, H_CONFIGS))})"
    rf"_H(?P<mode>{'|'.join(H_MODES)})_seed(?P<seed>{'|'.join(map(str, C2_SEEDS))}))"
    r"_test_S0_lead(?P<lead>[1-7])\.npz"
)
B3_MASKS = {"P10S12": (.1, .12), "P05S05": (.05, .05)}
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
# <a id="evaluation-part-2"></a>
#
# ## Phần 2 — Hợp đồng dự báo và kiểm đầu vào

# %% [markdown]
# ### 2.1. Một tệp tương ứng một cấu hình, seed, kỳ, kịch bản và hạn
#
# origin_date là ngày đích đầu của đợt dự báo, theo quy ước notebook A3/B2.
# Hạn k dự báo target = origin + k−1; persistence luôn lấy Q(origin−1).
# Ngày phát hành thực tế phải được bổ sung từ bên xuất, không suy từ lịch ngày.

# %% Phần 2.1 - Cấu trúc dự báo và kiểm lịch, ID, đơn vị
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
# <a id="evaluation-part-3"></a>
#
# ## Phần 3 — Các chỉ số liên tục

# %% [markdown]
# ### 3.1. NSE, PNSE, KGE và r, alpha, beta
#
# Một mặt nạ cho sáu chỉ số; không thêm epsilon vào mẫu số suy biến.

# %% Phần 3.1 - Hàm chỉ số cho một lưu vực
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
#
# FHV dùng 2% ngày cao nhất; FLV dùng 30% thấp nhất; FMS dùng phân vị 0,8 và 0,3.
# Chỉ phép log của FMS/FLV dùng sàn 10⁻⁶ mm/ngày đã khai báo; mẫu số 0 giữ NaN.

# %% Phần 3.2 - FHV, FMS, FLV và trạng thái
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
# <a id="evaluation-part-4"></a>
#
# ## Phần 4 — Đánh giá bốn mức

# %% [markdown]
# ### 4.1. Ma trận nhầm lẫn và chỉ số từng mức
#
# Dùng ngưỡng A1 cố định; đúng ngưỡng được xếp vào mức cao hơn.

# %% Phần 4.1 - Phân lớp và chỉ số
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
# <a id="evaluation-part-5"></a>
#
# ## Phần 5 — Trích đoạn vượt ngưỡng và gộp USWRC

# %% [markdown]
# ### 5.1. Đoạn liên tiếp và cờ biên/thiếu
#
# Vượt ngưỡng dùng dấu >, khác quy tắc bằng ngưỡng của phân mức ngày.

# %% Phần 5.1 - Đoạn vượt và gộp cụm
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
# <a id="evaluation-part-6"></a>
#
# ## Phần 6 — Ghép sự kiện một–một

# %% [markdown]
# ### 6.1. Số cặp tối đa, sau đó tổng lệch onset nhỏ nhất
#
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

# %% Phần 6.1 - Ghép tối ưu theo lịch và chấm POD/FAR
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
                 tolerance=None, case_rows=None):
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
    if case_rows is not None:
        case_rows.extend(dict(status="hit", **row) for row in details)
        matched_obs = {i for i, _ in pairs}
        matched_sim = {j for _, j in pairs}
        for events, matched, prefix, status in ((obs_events, matched_obs, "obs", "missed"),
                                                (sim_events, matched_sim, "sim", "false_alarm")):
            case_rows.extend({"status": status, f"{prefix}_onset": event["start"],
                              f"{prefix}_peak": event["first_peak"]}
                             for i, event in enumerate(events) if i not in matched)
    return result, details

# %% [markdown]
# <a id="evaluation-part-7"></a>
#
# ## Phần 7 — So sánh mô hình và kịch bản

# %% [markdown]
# ### 7.1. Wilcoxon hai phía, Cohen's dz và Holm
#
# Mỗi lưu vực là một cặp; giữ điểm âm. Holm chỉ đếm các p-value hữu hạn trong họ.

# %% Phần 7.1 - Kiểm định ghép cặp và hiệu chỉnh
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


def scenario_changes(forecasts, *, shared_mask=None):
    rows = []
    index = {(f.name, f.period, f.lead, f.scenario): f for f in forecasts}
    for forecast in forecasts:
        if forecast.scenario == "S0":
            continue
        base = index.get((forecast.name, forecast.period, forecast.lead, "S0"))
        if base is None:
            raise ValueError(f"Thiếu S0 để đối chiếu {forecast.name}/{forecast.scenario}")
        check_same_context(base, forecast)
        own_mask = common_mask([base, forecast])
        if shared_mask is not None and (shared_mask.shape != own_mask.shape or (shared_mask & ~own_mask).any()):
            raise ValueError("Mặt nạ chênh lệch kịch bản nằm ngoài cặp hợp lệ.")
        mask = own_mask if shared_mask is None else shared_mask
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
# <a id="evaluation-part-8"></a>
#
# ## Phần 8 — Đọc Output A1/A2/A3/A4/B0/B1

# %% [markdown]
# ### 8.1. Chọn nguồn duy nhất và ghi SHA-256
#
# Nếu Add Input có nhiều phiên bản cùng tên, chỉ định đường dẫn thay vì tự chọn.

# %% Phần 8.1 - Tìm tệp và kiểm ngữ cảnh canonical A3
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
#
# A2 giữ obs native để đối chiếu; phép so chính dùng obs canonical A3.

# %% Phần 8.2 - Bộ đọc dự báo đã lưu
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
#
# Ngưỡng là đầu vào cố định; bộ đánh giá không fit lại theo test hoặc theo sim.

# %% Phần 8.3 - Đọc ngưỡng và hợp đồng NPZ mới
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
# ### 8.3.2. Output C2 Compare Version 1 và dự báo H
#
# C2 dùng manifest, summary, environment, bảng checkpoint/hợp đồng và 24 NPZ do Compare xuất.
# H dùng một manifest với 168 NPZ trên cùng lịch đích. Chấm riêng H vì lịch và persistence khác.

# %% Phần 8.3.2 - Kiểm bàn giao C2 và H
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


def read_c2_compare(root, context, context_paths, b3_root=None, b0_noise=None):
    records, files, checks, _ = c2_records(root, context_paths, b3_root, b0_noise)
    members, audits = zip(*(c2_forecast(r, context) for r in records))
    return list(members), [Path(root) / "manifest.csv", *files.values()], checks, pd.DataFrame(audits)


def read_h_contracts(root, context):
    root = Path(root)
    candidates = sorted(root.rglob("SF_*_H*_test_S0_lead*.npz"))
    expected = {f"SF_{c}_H{m}_seed{s}_test_S0_lead{k}.npz"
                for c, m, s, k in itertools.product(H_CONFIGS, H_MODES, C2_SEEDS, H_LEADS)}
    names = {p.name for p in candidates}
    if len(candidates) != len(expected) or names != expected:
        raise ValueError(f"H cần đủ {len(expected)} NPZ (24 run × 7 hạn) cho {', '.join(H_MODES)}. "
                         f"Tìm thấy {len(candidates)} tệp, thiếu {len(expected - names)} tên, "
                         f"thừa {len(names - expected)} tên, trùng {len(candidates) - len(names)} tệp.")
    manifest_roots = [p for p in candidates[0].parents if p.is_relative_to(root) and (p / "manifest.csv").is_file()]
    if not manifest_roots:
        raise ValueError("Thiếu manifest.csv của H Compare")
    export = manifest_roots[0]
    files, checks = verified_manifest(export)
    if {p.resolve() for p in candidates} != {p for name, p in files.items() if name.endswith(".npz")}:
        raise ValueError("NPZ H không khớp tập tệp manifest")
    horizon = H_LEADS[-1]
    trim = horizon - 1
    days = context["dates"][trim:-trim]
    obs = context["obs"][trim:-trim]
    source_q = np.vstack((context["persistence"][:1], context["obs"]))
    members, hashes = [], {}
    contract_checks = []
    for p in candidates:
        match = H_NAME.fullmatch(p.name)
        run, mode, config = match["run"], match["mode"], match["config"]
        seed, lead = int(match["seed"]), int(match["lead"])
        f = read_contract(p)
        pers = source_q[horizon - lead:len(source_q) - trim - lead]
        if ((f.name, f.seed, f.period, f.scenario, f.lead) != (run, seed, "test", "S0", lead)
                or f.metadata["forecast_horizon"] != horizon
                or f.metadata.get("mode", mode) != mode or f.metadata.get("config", config) != config
                or not f.metadata["clip_to_zero"] or not digest_string(f.metadata["config_sha256"])):
            raise ValueError(f"H metadata khác hợp đồng: {p.name}")
        commit = f.metadata["library_commit"]
        expected_commit = H_LIBRARY_COMMITS[mode]
        if commit != expected_commit:
            raise ValueError(f"H library_commit sai cho {mode} ({p.name}): {commit!r}, kỳ vọng {expected_commit}.")
        digest = f.metadata["config_sha256"]
        if hashes.setdefault(run, digest) != digest:
            raise ValueError(f"Các hạn H không cùng cấu hình: {run}")
        reference = Forecast(name="A3", dates=days, ids=context["ids"], area=context["area"],
                             obs=obs, persistence=pers, sim=pers, raw=pers)
        check_same_context(reference, f)
        f.obs, f.persistence, f.area = obs, pers, context["area"]
        members.append(f)
        contract_checks.append(dict(path=str(p.resolve()), run=run, mode=mode, lead=lead, library_commit=commit))
    return members, checks.merge(pd.DataFrame(contract_checks), on="path", how="left", validate="one_to_one")


def read_c2_scenarios(root, c2, c2_root, context_paths):
    if len(c2) != 24:
        raise ValueError("Gắn C2 Compare Version 1 để đối chiếu kịch bản C2.")
    files, checks = verified_manifest(root)
    summary = json.loads(files["c2_scenarios_summary.json"].read_text(encoding="utf-8"))
    refs = {f.name: f for f in c2}
    names = {f"contracts/{name}_test_{scenario}_lead1.npz": (ref, scenario)
             for name, ref in refs.items() for scenario in C2_SCENARIOS[1:]}
    mask_name = "C2_scenarios_common_mask.npz"
    npz_names = {name for name in files if name.endswith(".npz")}
    if npz_names != {*names, mask_name}:
        raise ValueError(f"{C2_SCENARIOS_NOTEBOOK}: cần đúng {len(names)} NPZ dự báo và {mask_name}.")
    if (summary["complete"] is not True or summary["library_commit"] != FREEZE_C
            or summary.get("notebook") != C2_SCENARIOS_NOTEBOOK
            or summary.get("period") != "test" or summary.get("lead") != 1
            or summary.get("configurations") != list(C2_CONFIGS) or summary.get("seeds") != list(C2_SEEDS)
            or summary.get("C2_compare_version_expected") != 1
            or summary.get("comparison_scope") != "same_days_all_C2_runs_and_scenarios"
            or summary["scenarios"] != list(C2_SCENARIOS) or summary["S0_reproduced"] != len(c2)
            or summary["S0_basin_checks"] != len(c2) * len(c2[0].ids)
            or summary["new_contracts"] != len(names) or summary["optimizer_steps"] != 0
            or summary["C2_manifest_sha256"] != sha256(Path(c2_root) / "manifest.csv")
            or summary["A3"] != {p.name: sha256(p) for p in context_paths}):
        raise ValueError("Bàn giao các kịch bản C2 không khớp C2 Compare/A3.")
    audit = pd.read_csv(files["C2_scenarios_S0_check.csv"], dtype={"basin_id": str})
    keys = ["run", "basin_id"]
    if (len(audit) != len(c2) * len(c2[0].ids)
            or set(audit[keys].itertuples(index=False, name=None)) != set(itertools.product(refs, c2[0].ids))
            or not audit["passed"].eq(True).all()):
        raise ValueError("Audit tái lập S0 không đủ theo checkpoint và ID.")
    forecasts = list(c2)
    for name, (ref, scenario) in names.items():
        f = read_contract(files[name])
        if ((f.name, f.seed, f.period, f.scenario, f.lead) != (ref.name, ref.seed, "test", scenario, 1)
                or f.metadata.get("checkpoint_bytes_verified") is not True
                or f.metadata.get("training_rerun") is not False
                or f.metadata.get("inference_library_commit") != FREEZE_C
                or f.metadata.get("S0_source_contract_sha256") != ref.metadata["source_file_sha256"]
                or any(f.metadata.get(k) != ref.metadata[k] for k in (
                    "config_sha256", "checkpoint_sha256", "scaler_sha256", "selected_epoch", "library_commit",
                    "clip_to_zero", "forecast_horizon", "training_notebook", "step", "config"))):
            raise ValueError(f"Kịch bản C2 sai nguồn: {name}")
        check_same_context(ref, f)
        f.obs, f.persistence, f.area = ref.obs, ref.persistence, ref.area
        forecasts.append(f)
    shared = common_mask(forecasts)
    mask_ref = c2[0]
    with np.load(files[mask_name], allow_pickle=False) as stored:
        if (not {"dates", "basin_ids", "mask"} <= set(stored.files)
                or not np.array_equal(stored["dates"], mask_ref.dates.to_numpy(dtype="datetime64[D]"))
                or not np.array_equal(stored["basin_ids"], np.array(mask_ref.ids))
                or stored["mask"].dtype != np.dtype(bool) or not np.array_equal(stored["mask"], shared)
                or summary.get("shared_days_IDs") != int(shared.sum())
                or summary.get("IDs_with_days") != int(shared.any(axis=0).sum())):
            raise ValueError("Mặt nạ chung của 04_C2 khác giao ngày–lưu vực tính lại từ dự báo.")
    return forecasts, checks


def score_partitions(members, levels, floods, output, prefix, key_kind):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_file = output / f"{prefix}_summary.json"
    summary_file.write_text('{"complete": false}', encoding="utf-8")
    shared = np.logical_and.reduce([common_mask(group) for group in members.values()])
    if not shared.any():
        raise ValueError(f"{prefix} không có ngày–lưu vực chung")
    first = next(iter(members.values()))[0]
    days = first.dates.to_numpy(dtype="datetime64[D]")
    np.savez_compressed(output / f"{prefix}_common_mask.npz", dates=days, basin_ids=np.array(first.ids), mask=shared)
    coverage = []
    for key, group in members.items():
        base = group[0]
        print(f"{prefix}: chấm {key}; {len(group)} run và persistence trên giao chung", flush=True)
        if not base.dates.equals(first.dates) or base.ids != first.ids:
            raise ValueError(f"{prefix} không cùng lịch hoặc ID.")
        baseline = replace(base, name="Persistence_qmax", seed=None, sim=base.persistence, raw=base.persistence,
                           metadata={"forecast_horizon": base.metadata["forecast_horizon"]}).validate()
        tables, _ = evaluate_group([*group, baseline], levels, floods, shared_mask=shared)
        tables["summary"] = summarize(tables["metrics"])
        tables["paired_tests"] = pairwise_tests(tables["metrics"])
        tables["levels_pooled"] = pool_levels(tables["confusion"])
        tables["events_pooled"] = pool_events(tables["events"])
        tables["events_pooled_tolerance1"] = pool_events(tables["events_tolerance1"])
        label = f"lead{key}" if key_kind == "lead" else key
        for name, table in tables.items():
            table.to_csv(output / f"{prefix}_{name}_{label}.csv", index=False, encoding="utf-8-sig")
        coverage.append(dict(**{key_kind: key}, runs=len(group), days=len(days), start=str(days[0]), end=str(days[-1]),
                             common_day_IDs=int(shared.sum()), IDs_with_days=int(shared.any(axis=0).sum())))
    pd.DataFrame(coverage).to_csv(output / f"{prefix}_coverage.csv", index=False)
    summary_file.write_text(json.dumps(dict(complete=True, contracts=sum(len(g) for g in members.values()),
                                 coverage=coverage, common_mask_across=key_kind,
                                 tolerance_days={g[0].lead: primary_event_tolerance_days(
                                     g[0].lead, g[0].metadata["forecast_horizon"]) for g in members.values()}),
                                 indent=2), encoding="utf-8")
    return shared


def score_h(forecasts, levels, floods, output):
    members = {k: [f for f in forecasts if f.lead == k] for k in H_LEADS}
    score_partitions(members, levels, floods, output, "Evaluation_H", "lead")


def read_d_compare(root, c2_forecasts, c2_root, context, context_paths):
    if len(c2_forecasts) != 24 or c2_root is None:
        raise ValueError("Gắn C2 Compare Version 1 để đối chiếu D Compare.")
    export = unique_file(root, "d_summary.json").parent
    files, checks = verified_manifest(export)
    required_meta = {"d_summary.json", "environment.json", "D_config_check.csv", "D_contract_check.csv",
                     "D_seed_scores.csv", "D_score_stats.csv", "D_cost.csv", "D_cost_ratios.csv",
                     "D_basin_scores.csv", "D_input_sources.csv", "D_Length_Comparison.png", "D_common_mask.npz"}
    if not required_meta.issubset(files):
        raise ValueError(f"{D_COMPARE_NOTEBOOK}: thiếu tệp bắt buộc {sorted(required_meta - set(files))}.")
    summary = json.loads(files["d_summary.json"].read_text(encoding="utf-8-sig"))
    environment = json.loads(files["environment.json"].read_text(encoding="utf-8-sig"))
    s4d_included = summary.get("S4D_included")
    if not isinstance(s4d_included, bool):
        raise ValueError("D Compare thiếu cờ boolean S4D_included.")
    expected_configs = list(D_CONFIGS if s4d_included else D_CONFIGS[:-1])
    inputs = summary.get("inputs") or {}
    expected_a3 = {p.name: sha256(p) for p in context_paths}
    expected_c2_sha = sha256(Path(c2_root) / "manifest.csv")
    if (summary.get("complete") is not True or summary.get("notebook") != D_COMPARE_NOTEBOOK
            or environment.get("notebook") != D_COMPARE_NOTEBOOK
            or summary.get("step") != "D" or summary.get("library_commit") != FREEZE_C
            or environment.get("source_neuralhydrology_commit") != FREEZE_C
            or summary.get("configs") != expected_configs or summary.get("seeds") != list(C2_SEEDS)
            or summary.get("forecast_horizon") != 1 or summary.get("IDs") != len(context["ids"])
            or summary.get("scope") != "test_S0_common_all_C2_and_D"
            or inputs.get("C2_compare_version_expected") != 1
            or inputs.get("C2_manifest_sha256") != expected_c2_sha
            or environment.get("C2_manifest_sha256") != expected_c2_sha
            or inputs.get("A3") != expected_a3 or environment.get("A3") != expected_a3):
        raise ValueError("Bàn giao D Compare không khớp C2 Compare Version 1, A3 hoặc freeze-C.")
    expected_rels = [f"contracts/SF_{conf}_D_seed{seed}_test_S0_lead1.npz"
                     for conf in expected_configs for seed in C2_SEEDS]
    npz_names = {name for name in files if name.endswith(".npz")}
    actual_contracts = {p.relative_to(export).as_posix() for p in (export / "contracts").glob("*.npz")}
    if (npz_names != {*expected_rels, "D_common_mask.npz"} or actual_contracts != set(expected_rels)
            or summary.get("contracts") != len(expected_rels)
            or set(summary.get("output_contracts") or ()) != set(expected_rels)):
        raise ValueError(f"{D_COMPARE_NOTEBOOK}: cần đúng {len(expected_rels)} NPZ dự báo và D_common_mask.npz.")
    cfg_check = pd.read_csv(files["D_config_check.csv"])
    con_check = pd.read_csv(files["D_contract_check.csv"])
    expected_pairs = {(c, s) for c in expected_configs for s in C2_SEEDS}
    if (len(cfg_check) != len(expected_pairs)
            or set(cfg_check[["configuration", "seed"]].itertuples(index=False, name=None)) != expected_pairs
            or not cfg_check["seq_length_365"].eq(365).all()
            or not cfg_check["seq_length_730"].eq(730).all()
            or not cfg_check["configuration_difference"].eq("seq_length").all()):
        raise ValueError("Bảng D_config_check.csv không khớp cặp cấu hình/seed hoặc seq_length 365/730.")
    if len(con_check) != len(expected_pairs) or not con_check["passed"].astype(str).str.lower().eq("true").all():
        raise ValueError("Bảng D_contract_check.csv thiếu run hoặc có run chưa đạt.")
    con_by_run = con_check.set_index("run", verify_integrity=True)
    cfg_by_pair = cfg_check.set_index(["configuration", "seed"], verify_integrity=True)
    handoff_by_pair = None
    handoff_names = {"D_handoff.csv", "D_demo_runs.csv"}
    if handoff_names & set(files):
        if not handoff_names <= set(files):
            raise ValueError(f"{D_COMPARE_NOTEBOOK}: manifest phải có đủ D_handoff.csv và D_demo_runs.csv.")
        handoff = pd.read_csv(files["D_handoff.csv"])
        demo_runs = pd.read_csv(files["D_demo_runs.csv"])
        if (len(handoff) != len(expected_pairs)
                or set(handoff[["configuration", "seed"]].itertuples(index=False, name=None)) != expected_pairs
                or len(demo_runs) != len(expected_configs)
                or set(demo_runs["configuration"]) != set(expected_configs)
                or set(demo_runs[["configuration", "seed"]].itertuples(index=False, name=None))
                   != set(handoff.loc[handoff["demo"].astype(str).str.lower().eq("true"),
                                      ["configuration", "seed"]].itertuples(index=False, name=None))):
            raise ValueError("Bảng D_handoff.csv hoặc D_demo_runs.csv thiếu run hoặc lệch danh sách demo.")
        handoff_by_pair = handoff.set_index(["configuration", "seed"], verify_integrity=True)
    c2_by_name = {f.name: f for f in c2_forecasts}
    selected_c2, d_forecasts, contract_rows = [], [], []
    for conf in expected_configs:
        for seed in C2_SEEDS:
            ref = c2_by_name[f"SF_{conf}_C2_seed{seed}"]
            rel = f"contracts/SF_{conf}_D_seed{seed}_test_S0_lead1.npz"
            f = read_contract(files[rel])
            meta = f.metadata
            epoch = meta.get("selected_epoch")
            gg_skips = meta.get("grad_guard_skipped_steps", 0)
            gg_enabled = bool(meta.get("grad_guard_enabled", False))
            audit_row = con_by_run.loc[f.name]
            cfg_row = cfg_by_pair.loc[(conf, seed)]
            contract_digest = sha256(files[rel])
            if ((f.name, f.seed, f.period, f.scenario, f.lead) != (f"SF_{conf}_D_seed{seed}", seed, "test", "S0", 1)
                    or meta.get("step") != "D" or meta.get("config") != conf
                    or meta.get("core") != ref.metadata["core"] or meta.get("depth") != ref.metadata["depth"]
                    or meta.get("level") != 1 or meta.get("n_params") != ref.metadata["n_params"]
                    or meta.get("learning_rate") != ref.metadata["learning_rate"]
                    or meta.get("seq_length") != 730 or meta.get("hindcast_steps") != 729
                    or meta.get("forecast_steps") != 1 or meta.get("forecast_horizon") != 1
                    or meta.get("clip_to_zero") is not True or meta.get("library_commit") != FREEZE_C
                    or meta.get("baseline_run") != ref.name
                    or meta.get("baseline_contract_sha256") != ref.metadata["source_file_sha256"]
                    or type(epoch) is not int or not 1 <= epoch <= 30 or int(audit_row["selected_epoch"]) != epoch
                    or type(gg_skips) is not int or gg_skips < 0 or isinstance(gg_skips, bool)
                    or not all(digest_string(meta.get(k)) for k in (
                        "config_sha256", "checkpoint_sha256", "scaler_sha256", "training_environment_sha256"))
                    or str(audit_row["SHA256"]) != contract_digest
                    or int(cfg_row["n_params"]) != ref.metadata["n_params"]):
                raise ValueError(f"Hợp đồng D không khớp C2 đối chứng hoặc audit: {rel}")
            if handoff_by_pair is not None:
                h_row = handoff_by_pair.loc[(conf, seed)]
                if (h_row["run"] != f.name or int(h_row["selected_epoch"]) != epoch
                        or int(h_row["n_params"]) != meta["n_params"] or h_row["library_commit"] != FREEZE_C
                        or int(h_row["seq_length"]) != 730
                        or str(h_row["grad_guard_enabled"]).lower() != str(gg_enabled).lower()
                        or int(h_row["grad_guard_skipped_steps"]) != gg_skips
                        or h_row["notebook"] != meta.get("training_notebook")
                        or h_row["sha256_best_model_pt"] != meta["checkpoint_sha256"]
                        or h_row["sha256_config_yml"] != meta["config_sha256"]
                        or h_row["sha256_train_data_scaler_yml"] != meta["scaler_sha256"]
                        or h_row["contract_npz"] != rel or h_row["sha256_contract_npz"] != contract_digest):
                    raise ValueError(f"Bảng D_handoff.csv khác metadata NPZ D: {f.name}")
            check_same_context(ref, f)
            nse_vals = np.array([metrics(f.obs[:, j], f.sim[:, j])["NSE"] for j in range(len(f.ids))])
            npz_med = float(np.nanmedian(nse_vals))
            if not np.isclose(float(audit_row["native_nse_median"]), npz_med, atol=1e-5, rtol=0):
                raise ValueError(f"NSE trung vị từ NPZ D khác bảng D_contract_check.csv: {f.name}")
            f.obs, f.persistence, f.area = ref.obs, ref.persistence, ref.area
            f.metadata = dict(meta, compare_notebook=D_COMPARE_NOTEBOOK,
                              source_file_sha256=sha256(files[rel]), source=str(files[rel]))
            selected_c2.append(ref)
            d_forecasts.append(f)
            contract_rows.append(dict(
                run=f.name, baseline_run=ref.name, configuration_id=conf, seed=seed,
                selected_epoch=epoch, n_params=meta["n_params"], seq_length=730,
                grad_guard_enabled=gg_enabled, grad_guard_skipped_steps=gg_skips,
                strict_pair_equivalent=bool(cfg_row.get("strict_pair_equivalent", gg_skips == 0)),
                IDs=len(f.ids), target_days=len(f.dates),
                finite_prediction_day_IDs=int(np.isfinite(f.sim).sum()),
                canonical_context_matches=True,
                native_test_NSE=float(audit_row["native_nse_median"]), NPZ_test_NSE=npz_med,
                contract_sha256=f.metadata["source_file_sha256"], checkpoint_sha256=meta["checkpoint_sha256"],
            ))
    shared_pair = common_mask([*selected_c2, *d_forecasts])
    with np.load(files["D_common_mask.npz"], allow_pickle=False) as stored:
        if (not {"dates", "basin_ids", "mask"} <= set(stored.files)
                or not np.array_equal(stored["dates"], selected_c2[0].dates.to_numpy(dtype="datetime64[D]"))
                or not np.array_equal(stored["basin_ids"], np.array(selected_c2[0].ids))
                or stored["mask"].dtype != np.dtype(bool) or not np.array_equal(stored["mask"], shared_pair)
                or summary.get("common_day_IDs") != int(shared_pair.sum())
                or summary.get("IDs_with_days") != int(shared_pair.any(axis=0).sum())):
            raise ValueError("Mặt nạ D_common_mask.npz khác giao ngày–lưu vực tính lại từ dự báo C2/D.")
    return selected_c2, d_forecasts, checks, pd.DataFrame(contract_rows), summary


def score_d_length_pairs(metrics_table, selected_c2, d_forecasts):
    common = metrics_table.loc[metrics_table.scope.eq("common_source_all")]
    rows = []
    for metric in METRICS + SIGNATURES:
        wide = common.pivot(index="basin_id", columns="run", values=metric)
        family = []
        for c_run, d_run in zip(selected_c2, d_forecasts):
            a_vals = wide[c_run.name].to_numpy(dtype=float)
            b_vals = wide[d_run.name].to_numpy(dtype=float)
            valid = np.isfinite(a_vals) & np.isfinite(b_vals)
            med_365 = float(np.median(a_vals[valid])) if valid.any() else np.nan
            med_730 = float(np.median(b_vals[valid])) if valid.any() else np.nan
            family.append(dict(
                configuration=d_run.metadata["config"], seed=d_run.seed,
                run_365=c_run.name, run_730=d_run.name, metric=metric,
                median_365=med_365, median_730=med_730,
                difference_of_medians=med_730 - med_365 if valid.any() else np.nan,
                **paired_test(b_vals, a_vals),
            ))
        adjusted = holm([r["p_value"] for r in family])
        valid_size = sum(np.isfinite(r["p_value"]) for r in family)
        for row, p_adj in zip(family, adjusted):
            row.update(p_holm=p_adj, family_size=int(valid_size), planned_family_size=len(family),
                       significant_005=bool(p_adj <= .05) if np.isfinite(p_adj) else None)
        rows.extend(family)
    return pd.DataFrame(rows)


def score_d(selected_c2, d_forecasts, levels, floods, output, d_checks, d_contract_check, d_source_summary):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_file = output / "Evaluation_D_summary.json"
    summary_file.write_text('{"complete": false}', encoding="utf-8")
    base = selected_c2[0]
    baseline = replace(base, name="Persistence_qmax", seed=None, native_obs=None,
                       sim=base.persistence, raw=base.persistence,
                       metadata={"forecast_horizon": 1, "role": "observed_Q_reference_for_scoring"}).validate()
    group = [*selected_c2, *d_forecasts, baseline]
    shared = common_mask(group)
    if not shared.any():
        raise ValueError("Evaluation_D không có ngày–lưu vực chung")
    days = base.dates.to_numpy(dtype="datetime64[D]")
    np.savez_compressed(output / "Evaluation_D_common_mask.npz", dates=days, basin_ids=np.array(base.ids), mask=shared)
    print(f"Evaluation_D: chấm {len(d_forecasts)} cặp 365/730 và persistence trên giao chung", flush=True)
    tables, _ = evaluate_group(group, levels, floods, shared_mask=shared)
    tables["summary"] = summarize(tables["metrics"])
    tables["paired_tests"] = pairwise_tests(tables["metrics"])
    tables["length_pairs"] = score_d_length_pairs(tables["metrics"], selected_c2, d_forecasts)
    tables["levels_pooled"] = pool_levels(tables["confusion"])
    tables["events_pooled"] = pool_events(tables["events"])
    tables["events_pooled_tolerance1"] = pool_events(tables["events_tolerance1"])
    for name, table in tables.items():
        table.to_csv(output / f"Evaluation_D_{name}.csv", index=False, encoding="utf-8-sig")
    coverage = [dict(period="test", scenario="S0", lead=1, c2_runs=len(selected_c2), d_runs=len(d_forecasts),
                     runs=len(group), days=len(days), start=str(days[0]), end=str(days[-1]),
                     common_day_IDs=int(shared.sum()), IDs_with_days=int(shared.any(axis=0).sum()))]
    pd.DataFrame(coverage).to_csv(output / "Evaluation_D_coverage.csv", index=False, encoding="utf-8-sig")
    d_checks.to_csv(output / "Evaluation_D_input_check.csv", index=False, encoding="utf-8-sig")
    d_contract_check.to_csv(output / "Evaluation_D_contract_check.csv", index=False, encoding="utf-8-sig")
    summary_payload = dict(
        complete=True, protocol_version=PROTOCOL_VERSION, compare_notebook=D_COMPARE_NOTEBOOK,
        library_commit=FREEZE_C, configs=list(d_source_summary["configs"]),
        seeds=list(C2_SEEDS), S4D_included=bool(d_source_summary["S4D_included"]),
        contracts=len(d_forecasts), paired_c2_runs=len(selected_c2), total_scored_runs=len(group),
        grad_guard_runs_with_skips=list(d_source_summary.get("grad_guard_runs_with_skips") or []),
        all_pairs_strictly_equivalent=bool(d_source_summary.get("all_pairs_strictly_equivalent", True)),
        event_tolerance_days=primary_event_tolerance_days(1, 1),
        supplementary_event_tolerance_days=SUPPLEMENTARY_EVENT_TOLERANCE_DAYS,
        coverage=coverage,
    )
    summary_file.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return shared


# %% [markdown]
# ### 8.4. Hai mươi mốt NPZ LSTM A4/B0/B1a/B1b/B2/B3 từ Output Compare
#
# A4/B0 có schema riêng; B1 Compare chứa chung sáu NPZ và summary nested B0/B1a/B1b. Hai nguồn nền: A4 dùng Compare Version 2 và commit `dda45fc`; B0 dùng Compare Version 1 và commit `f229e02`. Cả ba notebook huấn luyện B0 đều Version 1.
#
# Mỗi bước kiểm đủ ba seed, manifest, sha256, epoch được chọn và ngữ cảnh A3 trước khi chấm. Ngưỡng B* phải bằng độ lệch chuẩn mẫu của ba NSE trung vị validation B0; không tính ngưỡng từ test.
# B3 Compare Version 1 gồm P10S12 và P05S05, mỗi mức ba seed; thư viện cùng commit B2, nguồn B2 Compare Version 1 và A3 Version 1 phải khớp hash.

# %% Phần 8.4 - Kiểm hai mươi mốt NPZ A4/B0/B1a/B1b/B2/B3
# Thiếu tệp, trùng nguồn, sai hash hoặc khác ngữ cảnh => dừng; không tự bỏ seed.
def read_compare(compare_root, context, context_paths, step):
    if step not in ("A4", "B0", *B1_STEPS):
        raise ValueError("Compare chỉ hỗ trợ A4, B0, B1a hoặc B1b")
    is_b1 = step in B1_STEPS
    seeds = B1_SEEDS if is_b1 else (A4_SEEDS if step == "A4" else B0_SEEDS)
    seed_versions = B1_SEED_VERSIONS if is_b1 else (A4_SEED_VERSIONS if step == "A4" else B0_SEED_VERSIONS)
    compare_version = B1_COMPARE_VERSION if is_b1 else (A4_COMPARE_VERSION if step == "A4" else B0_COMPARE_VERSION)
    commit = B1_LIBRARY_COMMIT if is_b1 else (A4_LIBRARY_COMMIT if step == "A4" else B0_LIBRARY_COMMIT)
    notebook = "05_B1_LamaHCEExt_Compare" if is_b1 else (
        "04_A4_LamaHCEExt_Compare" if step == "A4" else "02_B0_LamaHCEExt_Compare")
    summary_name = "b1_summary.json" if is_b1 else f"{step.lower()}_summary.json"
    root = unique_file(compare_root, summary_name).parent
    summary_path = root / summary_name
    environment_path = root / "environment.json"
    manifest_path = root / "manifest.csv"
    for path in (summary_path, environment_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"{step} thiếu {path.name}; gắn Output Compare Version {compare_version}")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if (not {"tệp", "sha256"}.issubset(manifest.columns) or manifest.empty
            or manifest["tệp"].duplicated().any()):
        raise ValueError(f"Manifest {step} cần cột tệp/sha256, không rỗng và không trùng đường dẫn")
    files, input_rows = {}, []
    for row in manifest.to_dict("records"):
        relative = row["tệp"]
        path = root / relative
        if (not relative or "\\" in relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError(f"Đường dẫn manifest {step} phải nằm trong thư mục Compare: {relative}")
        expected = row["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or not path.is_file():
            raise ValueError(f"{step} thiếu tệp hoặc digest không hợp lệ: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"SHA-256 {step} không khớp manifest: {relative}")
        files[relative] = path
        input_rows.append(dict(kind=f"{step}_export", file=relative, path=str(path),
                               expected_sha256=expected, observed_sha256=observed, passed=True))
    required = {summary_name, "environment.json"} | {
        f"contracts/SeqLSTM_Q_{step}_seed{seed}_test_S0_lead1.npz" for seed in seeds}
    if not required.issubset(files):
        raise ValueError(f"Manifest {step} thiếu các tệp bắt buộc: {sorted(required - set(files))}")
    contract_files = {p.relative_to(root).as_posix() for p in (root / "contracts").glob("*.npz")}
    expected_contracts = ({f"contracts/SeqLSTM_Q_{st}_seed{seed}_test_S0_lead1.npz"
                           for st in B1_STEPS for seed in B1_SEEDS} if is_b1 else
                          {name for name in required if name.endswith(".npz")})
    if contract_files != expected_contracts or not expected_contracts.issubset(files):
        raise ValueError(f"{step} Compare phải có đúng {'sáu B1a/B1b' if is_b1 else 'ba'} NPZ seed 111/222/333 ở contracts/manifest")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    environment = json.loads(environment_path.read_text(encoding="utf-8-sig"))
    source_commit = environment.get("neuralhydrology_commit_of_runs")
    if step == "B0":
        source_commit = source_commit.get("B0") if isinstance(source_commit, dict) else None
    if environment.get("notebook") != notebook or source_commit != commit:
        raise ValueError(f"Sai notebook nguồn hoặc commit thư viện {step}")
    seed_keys = {str(seed) for seed in seeds}
    if is_b1:
        expected_keys = {f"{st}_{seed}" for st in B1_STEPS for seed in B1_SEEDS}
        if not isinstance(summary.get("contracts"), dict) or set(summary["contracts"]) != expected_keys:
            raise ValueError("B1 summary cần đúng sáu hợp đồng B1a/B1b")
        for key in ("saved_epoch", "seed_test_nse_median", "seed_val_nse_median"):
            if (not isinstance(summary.get(key), dict) or set(summary[key]) != {"B0", *B1_STEPS}
                    or any(not isinstance(values, dict) or set(values) != seed_keys for values in summary[key].values())):
                raise ValueError(f"B1 summary sai cấu trúc B0/B1a/B1b tại {key}")
        original_summary = summary
        summary = dict(contracts={str(seed): original_summary["contracts"][f"{step}_{seed}"] for seed in seeds},
                       saved_epoch=original_summary["saved_epoch"][step],
                       seed_test_nse_median=original_summary["seed_test_nse_median"][step],
                       seed_val_nse_median=original_summary["seed_val_nse_median"]["B0"],
                       bstar_threshold_val_nse_sd=original_summary.get("bstar_threshold_val_nse_sd"))
    for key in ("contracts", "saved_epoch", "seed_test_nse_median"):
        if not isinstance(summary.get(key), dict) or set(summary[key]) != seed_keys:
            raise ValueError(f"{step} summary thiếu hoặc thừa seed ở {key}")
    if step == "B0" or is_b1:
        values = summary.get("seed_val_nse_median", {})
        if not isinstance(values, dict) or set(values) != seed_keys:
            raise ValueError("B0 summary thiếu hoặc thừa seed validation")
        val = np.asarray([values[str(seed)] for seed in seeds], dtype=float)
        threshold = summary.get("bstar_threshold_val_nse_sd")
        if (not np.isfinite(val).all() or not isinstance(threshold, (int, float))
                or not np.isfinite(threshold) or threshold < 0
                or not np.isclose(threshold, np.std(val, ddof=1), atol=1e-12, rtol=1e-9)):
            raise ValueError("Ngưỡng B* phải là std mẫu của ba NSE trung vị validation B0")
    source_a3 = environment.get("inputs", {}).get("a3", {})
    if source_a3.get("version") != 1:
        raise ValueError(f"{step} cần khai báo nguồn A3 Version 1 như bản bàn giao")
    by_name = {path.name: path for path in context_paths}
    for name in ("A3_persistence_predictions.npz", "A3_Q_source_window.npz"):
        expected = source_a3.get("files", {}).get(name, {}).get("sha256")
        observed = sha256(by_name[name])
        if expected != observed:
            raise ValueError(f"{step} dùng nguồn A3 khác Input hiện tại: {name}")
        input_rows.append(dict(kind=f"A3_context_used_by_{step}", file=name, path=str(by_name[name]),
                               expected_sha256=expected, observed_sha256=observed, passed=True))
    reference = Forecast("A3_context", sim=context["persistence"], raw=context["persistence"], **context).validate()
    forecasts, contract_rows = [], []
    for seed in seeds:
        key = str(seed)
        relative = f"contracts/SeqLSTM_Q_{step}_seed{seed}_test_S0_lead1.npz"
        if not str(summary["contracts"][key]).endswith("/" + relative):
            raise ValueError(f"{step} summary không trỏ đúng tệp seed {seed}")
        source = environment.get("inputs", {}).get(step if is_b1 else "seeds", {}).get(key, {})
        source_version = source.get("version")
        if is_b1 and source.get("notebook") != f"{'03' if step == 'B1a' else '04'}_{step}_LamaHCEExt_Train_Seed{seed}":
            raise ValueError(f"{step} seed {seed}: tên notebook huấn luyện không khớp bàn giao")
        if type(source_version) is not int or source_version != seed_versions[seed]:
            raise ValueError(f"{step} seed {seed}: Version nguồn không khớp bản bàn giao")
        forecast = read_contract(files[relative])
        meta = forecast.metadata
        selected_epoch = meta.get("selected_epoch")
        if (forecast.name != f"SeqLSTM_Q_{step}_seed{seed}" or type(forecast.seed) is not int
                or forecast.seed != seed or forecast.group != ("test", "S0", 1)
                or meta.get("forecast_horizon") != 1 or meta.get("clip_to_zero") is not True
                or meta.get("library_commit") != commit
                or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("config_sha256")))
                or type(selected_epoch) is not int or selected_epoch <= 0
                or summary["saved_epoch"][key] != selected_epoch):
            raise ValueError(f"{step} seed {seed}: sai run/kỳ/hạn/seed/epoch/commit/config hoặc chính sách clip")
        check_same_context(reference, forecast)
        forecast.metadata = dict(meta, source=str(files[relative]), source_file_sha256=sha256(files[relative]),
                                 compare_version_expected=compare_version, training_notebook_version=source_version)
        if step == "B0" or is_b1:
            forecast.metadata["bstar_threshold_val_nse_sd"] = summary["bstar_threshold_val_nse_sd"]
            forecast.metadata["reference_B0_seed_val_nse_median"] = summary["seed_val_nse_median"]
            forecast.metadata["reference_B0_saved_epoch"] = (
                original_summary["saved_epoch"]["B0"] if is_b1 else summary["saved_epoch"])
        forecasts.append(forecast)
        contract_rows.append(dict(run=forecast.name, seed=seed, selected_epoch=selected_epoch,
                                  training_notebook_version=source_version,
                                  compare_version_expected=compare_version,
                                  period=forecast.period, scenario=forecast.scenario, lead=forecast.lead,
                                  IDs=len(forecast.ids), target_days=len(forecast.dates),
                                  finite_prediction_day_IDs=int(np.isfinite(forecast.sim).sum()),
                                  paired_obs_prediction_day_IDs=int((np.isfinite(forecast.obs) & np.isfinite(forecast.sim)).sum()),
                                  sim_raw_negative_day_IDs=int((forecast.raw < 0).sum()),
                                  config_sha256=meta["config_sha256"], library_commit=meta["library_commit"],
                                  canonical_context_matches=True, manifest_sha256_matches=True))
    paths = [manifest_path, *files.values()]
    return forecasts, paths, pd.DataFrame(input_rows), pd.DataFrame(contract_rows)


def read_a4(a4_root, context, context_paths):
    return read_compare(a4_root, context, context_paths, "A4")


def read_b0(b0_root, context, context_paths):
    return read_compare(b0_root, context, context_paths, "B0")


def read_b1(b1_root, context, context_paths, step):
    if step not in B1_STEPS:
        raise ValueError("B1 chỉ gồm B1a và B1b")
    return read_compare(b1_root, context, context_paths, step)


def read_b2(b2_root, context, context_paths, b0_forecasts, b1_root):
    # B2 có manifest file/bytes/sha256, nguồn A3 viết hoa và đường dẫn NPZ tương đối.
    root = unique_file(b2_root, "b2_summary.json").parent
    manifest_path = root / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if (not {"file", "bytes", "sha256"}.issubset(manifest.columns) or manifest.empty
            or manifest["file"].duplicated().any()):
        raise ValueError("Manifest B2 thiếu cột, rỗng hoặc trùng đường dẫn")
    files, input_rows = {}, []
    for row in manifest.to_dict("records"):
        relative = row["file"]
        path = root / relative
        if (not relative or "\\" in relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError(f"Đường dẫn manifest B2 ra ngoài nguồn: {relative}")
        if (not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or not path.is_file()
                or str(path.stat().st_size) != row["bytes"] or sha256(path) != row["sha256"]):
            raise ValueError(f"B2 không khớp kích thước/SHA-256: {relative}")
        files[relative] = path
        input_rows.append(dict(kind="B2_export", file=relative, path=str(path),
                               expected_sha256=row["sha256"], observed_sha256=row["sha256"], passed=True))
    contracts = {f"contracts/SeqLSTM_Q_B2_seed{s}_test_S0_lead1.npz" for s in B2_SEEDS}
    required = contracts | {"b2_summary.json", "environment.json", "B2_seed_table.csv", "B2_seed_stats.csv",
                            "B2_contract_check.csv", "B2_input_sources.csv", "B2_B1b_reported_summary.csv", "B2_Seeds.png"}
    actual_contracts = {p.relative_to(root).as_posix() for p in (root / "contracts").glob("*.npz")}
    if not required.issubset(files) or actual_contracts != contracts:
        raise ValueError("B2 Compare thiếu tệp bắt buộc hoặc không có đúng ba NPZ seed 111/222/333")
    summary = json.loads(files["b2_summary.json"].read_text(encoding="utf-8-sig"))
    environment = json.loads(files["environment.json"].read_text(encoding="utf-8-sig"))
    if (summary.get("complete") is not True or summary.get("step") != "B2"
            or summary.get("scope") != "test_S0_lead1_three_seeds"
            or summary.get("library_commit") != B2_LIBRARY_COMMIT
            or environment.get("notebook") != "07_B2_LamaHCEExt_Compare"
            or environment.get("library_commit_of_runs") != B2_LIBRARY_COMMIT
            or summary.get("Bstar_selected") is not False or summary.get("operational_scenarios_run") is not False):
        raise ValueError("B2 summary/environment sai danh tính, commit hoặc phạm vi test/S0/hạn1")
    seed_keys = {str(s) for s in B2_SEEDS}
    for key in ("contracts", "saved_epoch", "seed_val_nse_median", "seed_test_nse_median", "training_versions_expected"):
        if not isinstance(summary.get(key), dict) or set(summary[key]) != seed_keys:
            raise ValueError(f"B2 summary thiếu hoặc thừa seed: {key}")
    source_a3 = environment.get("inputs", {}).get("A3", {})
    by_name = {p.name: p for p in context_paths}
    if source_a3.get("version") != 1:
        raise ValueError("B2 cần A3 Version 1")
    for name in ("A3_persistence_predictions.npz", "A3_Q_source_window.npz"):
        digest = sha256(by_name[name])
        # Compare B2 Version 1 ghi trực tiếp {tên_tệp: chuỗi_sha256}.
        expected = source_a3.get("files", {}).get(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or expected != digest:
            raise ValueError(f"B2 dùng nguồn A3 khác Input hiện tại: {name}")
        input_rows.append(dict(kind="A3_context_used_by_B2", file=name, path=str(by_name[name]),
                               expected_sha256=digest, observed_sha256=digest, passed=True))
    b1_summary_path = unique_file(b1_root, "b1_summary.json")
    b1_environment_path = b1_summary_path.parent / "environment.json"
    expected_b1 = dict(notebook="05_B1_LamaHCEExt_Compare", version=B1_COMPARE_VERSION,
                       summary_sha256=sha256(b1_summary_path), environment_sha256=sha256(b1_environment_path))
    if summary.get("B1b_source") != expected_b1 or environment.get("inputs", {}).get("B1_Compare") != expected_b1:
        raise ValueError("B2 dùng B1 Compare khác Input hiện tại")
    for path in (b1_summary_path, b1_environment_path):
        digest = sha256(path)
        input_rows.append(dict(kind="B1_context_used_by_B2", file=path.name, path=str(path),
                               expected_sha256=digest, observed_sha256=digest, passed=True))
    b1_summary = json.loads(b1_summary_path.read_text(encoding="utf-8-sig"))
    val = np.array([summary["seed_val_nse_median"][str(s)] for s in B2_SEEDS], dtype=float)
    native_test = np.array([summary["seed_test_nse_median"][str(s)] for s in B2_SEEDS], dtype=float)
    reference_val = np.mean(list(b1_summary["seed_val_nse_median"]["B1b"].values()))
    threshold = b0_forecasts[0].metadata["bstar_threshold_val_nse_sd"]
    delta = summary.get("B1b_validation_comparison", {})
    expected_delta = dict(B1b_validation_NSE_mean=reference_val, B2_validation_NSE_mean=float(val.mean()),
                          validation_NSE_mean_difference=float(val.mean() - reference_val), B0_Bstar_threshold=threshold)
    if (not np.isfinite(val).all() or not np.isfinite(native_test).all()
            or any(not np.isclose(delta.get(k, np.nan), v, atol=1e-12, rtol=1e-9) for k, v in expected_delta.items())
            or delta.get("Bstar_selected") is not False or delta.get("common_mask_evaluation_completed") is not False
            or not np.isclose(summary.get("seed_test_nse_mean", np.nan), native_test.mean(), atol=1e-12)
            or not np.isclose(summary.get("seed_test_nse_sd", np.nan), native_test.std(ddof=1), atol=1e-12)):
        raise ValueError("B2 tham chiếu validation B1b, ngưỡng B0 hoặc thống kê seed không khớp")
    tables = {}
    for name, seed_column in (("B2_seed_table.csv", "seed"), ("B2_input_sources.csv", "seed"),
                               ("B2_contract_check.csv", "hạt giống")):
        table = pd.read_csv(files[name], keep_default_na=False).set_index(seed_column)
        if table.index.has_duplicates or set(table.index) != set(B2_SEEDS):
            raise ValueError(f"B2 thiếu/thừa/trùng seed trong {name}")
        tables[name] = table
    reference = Forecast("A3_context", sim=context["persistence"], raw=context["persistence"], **context).validate()
    forecasts, contract_rows = [], []
    for seed in B2_SEEDS:
        key = str(seed)
        relative = f"contracts/SeqLSTM_Q_B2_seed{seed}_test_S0_lead1.npz"
        source = environment.get("inputs", {}).get("B2", {}).get(key, {})
        actual_notebook = f"06_B2_LamaHCEExt_Train_Seed{seed}"
        embedded = source.get("notebook_in_environment")
        if (summary["contracts"][key] != relative or type(summary["training_versions_expected"][key]) is not int
                or summary["training_versions_expected"][key] != 1 or type(source.get("version_expected")) is not int
                or source.get("version_expected") != 1
                or source.get("notebook") != actual_notebook
                or embedded not in (actual_notebook, f"07_B2_LamaHCEExt_Train_Seed{seed}")
                or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("environment_sha256")))):
            raise ValueError(f"B2 seed {seed}: sai đường dẫn NPZ hoặc nguồn Train Version 1")
        forecast = read_contract(files[relative])
        meta = forecast.metadata
        epoch = meta.get("selected_epoch")
        if (forecast.name != f"SeqLSTM_Q_B2_seed{seed}" or type(forecast.seed) is not int or forecast.seed != seed
                or forecast.group != ("test", "S0", 1) or meta.get("forecast_horizon") != 1
                or meta.get("clip_to_zero") is not True or meta.get("library_commit") != B2_LIBRARY_COMMIT
                or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("config_sha256")))
                or type(epoch) is not int or not 1 <= epoch <= 30 or epoch != summary["saved_epoch"][key]
                or meta.get("training_notebook_metadata") != embedded):
            raise ValueError(f"B2 seed {seed}: sai run/epoch/commit/config/clip hoặc tên nhúng")
        check_same_context(reference, forecast)
        seed_row = tables["B2_seed_table.csv"].loc[seed]
        source_row = tables["B2_input_sources.csv"].loc[seed]
        audit = tables["B2_contract_check.csv"].loc[seed]
        flags = ("clip_to_zero", "NaN NSE trùng thư viện", "mặt nạ obs trùng A3",
                 "giá trị obs khớp A3 trong sai số float32", "đạt")
        # Audit của Compare dùng giao obs/sim, độc lập với sự có mặt của persistence.
        # Giao chung cho sáu chỉ số vẫn được áp riêng trong evaluate_group.
        nse = np.array([metrics(forecast.obs[:, j], forecast.sim[:, j])["NSE"]
                        for j in range(len(forecast.ids))])
        audit_nse = float(np.nanmedian(nse))
        reported_nse = float(audit["NSE trung vị từ NPZ (obs A3)"])
        checks = [
            ("seed_table.selected_epoch", seed_row.selected_epoch, epoch, int(seed_row.selected_epoch)==epoch),
            ("seed_table.validation_NSE", seed_row.validation_NSE_median, val[B2_SEEDS.index(seed)],
             np.isclose(seed_row.validation_NSE_median, val[B2_SEEDS.index(seed)], atol=1e-12)),
            ("seed_table.test_NSE", seed_row.test_NSE_median, native_test[B2_SEEDS.index(seed)],
             np.isclose(seed_row.test_NSE_median, native_test[B2_SEEDS.index(seed)], atol=1e-12)),
            ("source.notebook", source_row.notebook, actual_notebook, source_row.notebook==actual_notebook),
            ("source.embedded_notebook", source_row.notebook_in_environment, embedded, source_row.notebook_in_environment==embedded),
            ("source.version", source_row.version, 1, source_row.version==1),
            ("source.environment_sha256", source_row.environment_sha256, source["environment_sha256"],
             source_row.environment_sha256==source["environment_sha256"]),
            ("audit.contract_errors", audit["lỗi hợp đồng"], "-", audit["lỗi hợp đồng"]=="-"),
            ("audit.selected_epoch", audit.selected_epoch, epoch, int(audit.selected_epoch)==epoch),
            ("audit.filename", audit["tệp"], Path(relative).name, audit["tệp"]==Path(relative).name),
            ("audit.finite_predictions", audit["ngày-lưu vực có sim"], int(np.isfinite(forecast.sim).sum()),
             int(audit["ngày-lưu vực có sim"])==int(np.isfinite(forecast.sim).sum())),
            ("audit.raw_negative_predictions", audit["dự báo âm trước chặn"], int((forecast.raw<0).sum()),
             int(audit["dự báo âm trước chặn"])==int((forecast.raw<0).sum())),
            ("audit.max_library_NSE_difference", audit["max |ΔNSE| so với thư viện"], "0..1e-5",
             0<=float(audit["max |ΔNSE| so với thư viện"])<=1e-5),
            ("audit.NSE_median_obs_sim", audit_nse, reported_nse,
             np.isclose(audit_nse, reported_nse, atol=1e-9, rtol=1e-9))]
        checks.extend((f"audit.{k}", audit[k], True, str(audit[k])=="True") for k in flags)
        failed = [f"{name}: nhận {actual!s}, cần {expected!s}" for name,actual,expected,passed in checks if not passed]
        if failed:
            raise ValueError(f"B2 seed {seed}: bảng seed/nguồn/audit khác NPZ hoặc summary:\n"+"\n".join(failed))
        forecast.metadata = dict(meta, source=str(files[relative]), source_file_sha256=sha256(files[relative]),
            compare_version_expected=B2_COMPARE_VERSION, training_notebook_version=1,
            training_notebook=actual_notebook, embedded_training_notebook=embedded,
            bstar_threshold_val_nse_sd=threshold)
        forecasts.append(forecast)
        contract_rows.append(dict(run=forecast.name, seed=seed, selected_epoch=epoch, training_notebook_version=1,
            training_notebook=actual_notebook, embedded_training_notebook=embedded,
            compare_version_expected=B2_COMPARE_VERSION, IDs=len(forecast.ids), target_days=len(forecast.dates),
            finite_prediction_day_IDs=int(np.isfinite(forecast.sim).sum()),
            paired_obs_prediction_day_IDs=int((np.isfinite(forecast.obs) & np.isfinite(forecast.sim)).sum()),
            sim_raw_negative_day_IDs=int((forecast.raw < 0).sum()), library_commit=B2_LIBRARY_COMMIT,
            config_sha256=meta["config_sha256"], canonical_context_matches=True, manifest_sha256_matches=True))
        contract_rows[-1].update(NSE_audit_mask="finite_obs_and_sim", NSE_audit_median=audit_nse,
            NSE_audit_reported_median=reported_nse, NSE_audit_abs_difference=abs(audit_nse-reported_nse))
    return forecasts, [manifest_path, *files.values(), b1_summary_path, b1_environment_path], pd.DataFrame(input_rows), pd.DataFrame(contract_rows)


def read_b3(b3_root, context, context_paths, b0_forecasts, b2_root):
    # Đọc đúng artifact Compare B3 Version 1; audit NSE dùng obs/sim, không phụ thuộc persistence.
    root = unique_file(b3_root, "b3_summary.json").parent
    manifest_path = root / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if (not {"file", "bytes", "sha256"}.issubset(manifest.columns) or manifest.empty
            or manifest["file"].duplicated().any()):
        raise ValueError("Manifest B3 thiếu cột, rỗng hoặc trùng đường dẫn")
    files, input_rows = {}, []
    for row in manifest.to_dict("records"):
        relative = row["file"]
        path = root / relative
        if (not relative or "\\" in relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError(f"Đường dẫn manifest B3 ra ngoài nguồn: {relative}")
        if (not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or not path.is_file()
                or str(path.stat().st_size) != row["bytes"] or sha256(path) != row["sha256"]):
            raise ValueError(f"B3 không khớp kích thước/SHA-256: {relative}")
        files[relative] = path
        input_rows.append(dict(kind="B3_export", file=relative, path=str(path),
                               expected_sha256=row["sha256"], observed_sha256=row["sha256"], passed=True))
    contracts = {f"contracts/SeqLSTM_Q_B3_{level}_seed{s}_test_S0_lead1.npz"
                 for level in B3_MASKS for s in B3_SEEDS}
    required = contracts | {"b3_summary.json", "environment.json", "B3_seed_table.csv", "B3_seed_stats.csv",
                            "B3_contract_check.csv", "B3_input_sources.csv", "B3_B2_reported_summary.csv", "B3_Seeds.png"}
    actual_contracts = {p.relative_to(root).as_posix() for p in root.rglob("*.npz")}
    if not required.issubset(files) or actual_contracts != contracts:
        raise ValueError("B3 Compare cần đúng sáu NPZ: hai mức P10S12/P05S05, seed 111/222/333")
    summary = json.loads(files["b3_summary.json"].read_text(encoding="utf-8-sig"))
    environment = json.loads(files["environment.json"].read_text(encoding="utf-8-sig"))
    if (summary.get("complete") is not True or summary.get("step") != "B3"
            or summary.get("scope") != "test_S0_lead1_two_mask_levels_three_seeds_each"
            or summary.get("library_commit") != B3_LIBRARY_COMMIT
            or environment.get("notebook") != "09_B3_LamaHCEExt_Compare"
            or environment.get("library_commit_of_runs") != B3_LIBRARY_COMMIT
            or summary.get("Bstar_selected") is not False or environment.get("Bstar_selected") is not False
            or summary.get("operational_scenarios_run") is not False or summary.get("multi_lead_run") is not False):
        raise ValueError("B3 summary/environment sai danh tính, commit hoặc phạm vi test/S0/hạn1")
    if set(summary.get("mask_levels", {})) != set(B3_MASKS):
        raise ValueError("B3 thiếu hoặc thừa mức che trong summary")
    source_a3 = environment.get("inputs", {}).get("A3", {})
    by_name = {p.name: p for p in context_paths}
    if type(source_a3.get("version")) is not int or source_a3["version"] != 1:
        raise ValueError("B3 cần A3 Version 1")
    for name in ("A3_persistence_predictions.npz", "A3_Q_source_window.npz"):
        digest = sha256(by_name[name])
        expected = source_a3.get("files", {}).get(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or expected != digest:
            raise ValueError(f"B3 dùng nguồn A3 khác Input hiện tại: {name}")
        input_rows.append(dict(kind="A3_context_used_by_B3", file=name, path=str(by_name[name]),
                               expected_sha256=digest, observed_sha256=digest, passed=True))
    b2_summary_path = unique_file(b2_root, "b2_summary.json")
    b2_environment_path = b2_summary_path.parent / "environment.json"
    identity = dict(notebook="07_B2_LamaHCEExt_Compare", version=B2_COMPARE_VERSION,
                    summary_sha256=sha256(b2_summary_path), environment_sha256=sha256(b2_environment_path))
    source_b2 = summary.get("B2_source", {})
    if (type(source_b2.get("version")) is not int or source_b2 != identity
            or environment.get("inputs", {}).get("B2_Compare") != identity):
        raise ValueError("B3 dùng B2 Compare khác Input hiện tại")
    for path in (b2_summary_path, b2_environment_path):
        digest = sha256(path)
        input_rows.append(dict(kind="B2_context_used_by_B3", file=path.name, path=str(path),
                               expected_sha256=digest, observed_sha256=digest, passed=True))
    b2_summary = json.loads(b2_summary_path.read_text(encoding="utf-8-sig"))
    b2_environment = json.loads(b2_environment_path.read_text(encoding="utf-8-sig"))
    reference_val = float(np.mean(list(b2_summary["seed_val_nse_median"].values())))
    threshold = b0_forecasts[0].metadata["bstar_threshold_val_nse_sd"]
    if not np.isclose(summary.get("bstar_threshold_val_nse_sd", np.nan), threshold, atol=1e-12, rtol=1e-9):
        raise ValueError("B3 ngưỡng B0 khác Input hiện tại")
    tables = {}
    expected_keys = set(itertools.product(B3_MASKS, B3_SEEDS))
    for name, seed_column in (("B3_seed_table.csv", "seed"), ("B3_input_sources.csv", "seed"),
                               ("B3_contract_check.csv", "hạt giống")):
        table = pd.read_csv(files[name], keep_default_na=False).set_index(["mask_level", seed_column])
        if table.index.has_duplicates or set(table.index) != expected_keys:
            raise ValueError(f"B3 thiếu/thừa/trùng mức che và seed trong {name}")
        tables[name] = table
    stats = pd.read_csv(files["B3_seed_stats.csv"]).set_index(["mask_level", "metric"])
    columns = ("validation_NSE_median", "test_NSE_median", "test_KGE_median", "train_hours")
    if stats.index.has_duplicates or set(stats.index) != set(itertools.product(B3_MASKS, columns)):
        raise ValueError("B3 seed stats thiếu/thừa/trùng mức che và chỉ số")
    deltas = pd.read_csv(files["B3_B2_reported_summary.csv"], keep_default_na=False).set_index("comparison")
    if deltas.index.has_duplicates or set(deltas.index) != {f"B3 {level} - B2" for level in B3_MASKS}:
        raise ValueError("B3 bảng validation delta thiếu/thừa/trùng mức che")
    reference = Forecast("A3_context", sim=context["persistence"], raw=context["persistence"], **context).validate()
    forecasts, contract_rows = [], []
    for level, (p_step, p_sequence) in B3_MASKS.items():
        member = summary["mask_levels"][level]
        if (member.get("nan_step_probability") != p_step or member.get("nan_sequence_probability") != p_sequence):
            raise ValueError(f"B3 {level}: sai hai xác suất che")
        for key in ("contracts", "saved_epoch", "seed_val_nse_median", "seed_test_nse_median", "training_versions_expected"):
            if not isinstance(member.get(key), dict) or set(member[key]) != {str(s) for s in B3_SEEDS}:
                raise ValueError(f"B3 {level}: summary thiếu/thừa seed ở {key}")
        val = np.array([member["seed_val_nse_median"][str(s)] for s in B3_SEEDS], dtype=float)
        native_test = np.array([member["seed_test_nse_median"][str(s)] for s in B3_SEEDS], dtype=float)
        if not np.isfinite(val).all() or not np.isfinite(native_test).all():
            raise ValueError(f"B3 {level}: NSE summary không hữu hạn")
        expected_delta = dict(B2_validation_NSE_mean=reference_val, B3_validation_NSE_mean=float(val.mean()),
                              validation_NSE_mean_difference=float(val.mean()-reference_val), B0_Bstar_threshold=threshold)
        delta = deltas.loc[f"B3 {level} - B2"]
        if (any(not np.isclose(delta.get(k, np.nan), v, atol=1e-12, rtol=1e-9) for k,v in expected_delta.items())
                or str(delta.Bstar_selected) != "False" or str(delta.common_mask_evaluation_completed) != "False"):
            raise ValueError(f"B3 {level}: validation delta hoặc cờ lựa chọn không khớp")
        for column in columns:
            values = tables["B3_seed_table.csv"].loc[level, column].to_numpy(dtype=float)
            row = stats.loc[(level, column)]
            if (not np.isfinite(values).all() or not np.isclose(row["mean"], values.mean(), atol=1e-12, rtol=1e-9)
                    or not np.isclose(row["std"], values.std(ddof=1), atol=1e-12, rtol=1e-9)):
                raise ValueError(f"B3 {level}: mean/SD không khớp {column}")
        for i, seed in enumerate(B3_SEEDS):
            key = str(seed)
            relative = f"contracts/SeqLSTM_Q_B3_{level}_seed{seed}_test_S0_lead1.npz"
            source = environment.get("inputs", {}).get("B3", {}).get(level, {}).get(key, {})
            notebook = f"08_B3_LamaHCEExt_Train_{level}_Seed{seed}"
            if (member["contracts"][key] != relative or type(member["training_versions_expected"][key]) is not int
                    or member["training_versions_expected"][key] != 1 or type(source.get("version_expected")) is not int
                    or source.get("version_expected") != 1 or source.get("notebook") != notebook
                    or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("environment_sha256")))):
                raise ValueError(f"B3 {level} seed {seed}: sai nguồn Train Version 1 hoặc đường dẫn NPZ")
            forecast = read_contract(files[relative])
            meta = forecast.metadata
            epoch = meta.get("selected_epoch")
            if (forecast.name != f"SeqLSTM_Q_B3_{level}_seed{seed}" or type(forecast.seed) is not int or forecast.seed != seed
                    or forecast.group != ("test", "S0", 1) or meta.get("forecast_horizon") != 1
                    or meta.get("step") != "B3" or meta.get("mask_level") != level
                    or meta.get("nan_step_probability") != p_step or meta.get("nan_sequence_probability") != p_sequence
                    or meta.get("clip_to_zero") is not True or meta.get("library_commit") != B3_LIBRARY_COMMIT
                    or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("config_sha256")))
                    or type(epoch) is not int or not 1 <= epoch <= 30
                    or type(member["saved_epoch"][key]) is not int or epoch != member["saved_epoch"][key]
                    or meta.get("training_notebook_metadata") != notebook):
                raise ValueError(f"B3 {level} seed {seed}: sai metadata run/mức che/epoch/commit/config/clip")
            check_same_context(reference, forecast)
            seed_row = tables["B3_seed_table.csv"].loc[(level, seed)]
            source_row = tables["B3_input_sources.csv"].loc[(level, seed)]
            audit = tables["B3_contract_check.csv"].loc[(level, seed)]
            b2_source_hash = b2_environment["inputs"]["B2"][key]["environment_sha256"]
            nse = np.array([metrics(forecast.obs[:, j], forecast.sim[:, j])["NSE"] for j in range(len(forecast.ids))])
            audit_nse = float(np.nanmedian(nse))
            reported_nse = float(audit["NSE trung vị từ NPZ (obs A3)"])
            checks = [
                ("seed.epoch", int(seed_row.selected_epoch)==epoch),
                ("seed.validation_NSE", np.isclose(seed_row.validation_NSE_median,val[i],atol=1e-12,rtol=1e-9)),
                ("seed.test_NSE", np.isclose(seed_row.test_NSE_median,native_test[i],atol=1e-12,rtol=1e-9)),
                ("seed.mask", seed_row.nan_step_probability==p_step and seed_row.nan_sequence_probability==p_sequence),
                ("source.notebook", source_row.notebook==notebook),
                ("source.version", source_row.version_expected==1),
                ("source.environment_SHA256", source_row.environment_sha256==source["environment_sha256"]),
                ("source.B2_environment_SHA256", source_row.B2_environment_sha256==b2_source_hash),
                ("audit.epoch", int(audit.selected_epoch)==epoch),
                ("audit.filename", audit["tệp"]==Path(relative).name),
                ("audit.contract_errors", audit["lỗi hợp đồng"]=="-"),
                ("audit.finite_predictions", int(audit["ngày-lưu vực có sim"])==int(np.isfinite(forecast.sim).sum())),
                ("audit.raw_negative_predictions", int(audit["dự báo âm trước chặn"])==int((forecast.raw<0).sum())),
                ("audit.library_NSE", 0<=float(audit["max |ΔNSE| so với thư viện"])<=1e-5),
                ("audit.NSE_median_obs_sim", np.isclose(audit_nse,reported_nse,atol=1e-9,rtol=1e-9))]
            flags = ("clip_to_zero", "NaN NSE trùng thư viện", "mặt nạ obs trùng A3",
                     "giá trị obs khớp A3 trong sai số float32", "đạt")
            checks.extend((f"audit.{k}",str(audit[k])=="True") for k in flags)
            failed = [name for name,passed in checks if not passed]
            if failed:
                raise ValueError(f"B3 {level} seed {seed}: bảng khác summary/NPZ: {failed}; "
                                 f"NSE obs/sim tính lại={audit_nse}, bảng={reported_nse}")
            forecast.metadata = dict(meta, source=str(files[relative]), source_file_sha256=sha256(files[relative]),
                compare_version_expected=B3_COMPARE_VERSION, training_notebook_version=1,
                training_notebook=notebook, bstar_threshold_val_nse_sd=threshold)
            forecasts.append(forecast)
            contract_rows.append(dict(run=forecast.name, mask_level=level, seed=seed, selected_epoch=epoch,
                nan_step_probability=p_step, nan_sequence_probability=p_sequence, training_notebook_version=1,
                training_notebook=notebook, compare_version_expected=B3_COMPARE_VERSION,
                IDs=len(forecast.ids), target_days=len(forecast.dates),
                finite_prediction_day_IDs=int(np.isfinite(forecast.sim).sum()),
                paired_obs_prediction_day_IDs=int((np.isfinite(forecast.obs)&np.isfinite(forecast.sim)).sum()),
                sim_raw_negative_day_IDs=int((forecast.raw<0).sum()), library_commit=B3_LIBRARY_COMMIT,
                config_sha256=meta["config_sha256"], canonical_context_matches=True, manifest_sha256_matches=True,
                NSE_audit_mask="finite_obs_and_sim", NSE_audit_median=audit_nse,
                NSE_audit_reported_median=reported_nse, NSE_audit_abs_difference=abs(audit_nse-reported_nse)))
    return forecasts, [manifest_path,*files.values(),b2_summary_path,b2_environment_path], pd.DataFrame(input_rows), pd.DataFrame(contract_rows)


def check_b1_b0_reference(b0_forecasts, b1_forecasts):
    reference, received = b0_forecasts[0].metadata, b1_forecasts[0].metadata
    keys = [str(seed) for seed in B0_SEEDS]
    if (not np.isclose(received["bstar_threshold_val_nse_sd"], reference["bstar_threshold_val_nse_sd"],
                       atol=1e-12, rtol=1e-9)
            or received["reference_B0_saved_epoch"] != reference["reference_B0_saved_epoch"]
            or not np.allclose([received["reference_B0_seed_val_nse_median"][key] for key in keys],
                               [reference["reference_B0_seed_val_nse_median"][key] for key in keys],
                               atol=1e-12, rtol=1e-9)):
        raise ValueError("B1 Compare dùng ngưỡng/seed validation/epoch B0 khác Output B0 Compare hiện tại")


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
# <a id="evaluation-part-9"></a>
#
# ## Phần 9 — Kiểm thử độc lập

# %% [markdown]
# ### 9.1. Ví dụ số, ca suy biến và bộ lọc
#
# Kết quả ở phần này là fixture, tách khỏi số liệu LamaH-CE.

# %% Phần 9.1 - Kiểm số học và giao thức trước khi đọc dataset
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
        # Fixture Compare dùng đúng hai schema bàn giao; không phải kết quả thực nghiệm.
        context_paths = [a3 / "A3_persistence_predictions.npz", a3 / "A3_Q_source_window.npz"]
        compare_models = []
        for step in ("A4", "B0"):
            compare = root / step.lower()
            contracts = compare / "contracts"
            contracts.mkdir(parents=True)
            commit = A4_LIBRARY_COMMIT if step == "A4" else B0_LIBRARY_COMMIT
            versions = A4_SEED_VERSIONS if step == "A4" else B0_SEED_VERSIONS
            summary_path = compare / f"{step.lower()}_summary.json"
            env_path = compare / "environment.json"
            val = {"111": .74, "222": .75, "333": .76}
            summary = dict(contracts={}, saved_epoch={}, seed_test_nse_median={})
            for seed in (111, 222, 333):
                name = f"SeqLSTM_Q_{step}_seed{seed}"
                relative = f"contracts/{name}_test_S0_lead1.npz"
                meta = dict(run=name, period="test", scenario="S0", lead=1, forecast_horizon=1,
                            seed=seed, unit="mm/day_area_gov", config_sha256="a" * 64,
                            library_commit=commit, clip_to_zero=True, selected_epoch=25)
                payload = dict(dates=dates.to_numpy(dtype="datetime64[D]"),
                               origin_dates=dates.to_numpy(dtype="datetime64[D]"),
                               persistence_source_dates=full_dates[:-1].to_numpy(dtype="datetime64[D]"),
                               basin_ids=np.array(ids), area_gov_km2=area, obs_mm_day=context["obs"],
                               persistence_mm_day=context["persistence"], sim_raw_mm_day=context["obs"],
                               sim_mm_day=context["obs"])
                np.savez_compressed(compare / relative, **payload, metadata_json=json.dumps(meta))
                summary["contracts"][str(seed)] = f"{step.lower()}_compare/{relative}"
                summary["saved_epoch"][str(seed)] = 25
                summary["seed_test_nse_median"][str(seed)] = 1.
            if step == "B0":
                summary.update(seed_val_nse_median=val, bstar_threshold_val_nse_sd=float(np.std(list(val.values()), ddof=1)))
            env = dict(notebook="04_A4_LamaHCEExt_Compare" if step == "A4" else "02_B0_LamaHCEExt_Compare",
                       neuralhydrology_commit_of_runs=commit if step == "A4" else {"B0": commit, "A4": A4_LIBRARY_COMMIT},
                       inputs=dict(seeds={str(k): {"version": v} for k, v in versions.items()},
                                   a3=dict(version=1, files={p.name: {"sha256": sha256(p)} for p in context_paths})))

            def seal_compare():
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                env_path.write_text(json.dumps(env), encoding="utf-8")
                pd.DataFrame([{"tệp": p.relative_to(compare).as_posix(), "sha256": sha256(p)}
                              for p in sorted(compare.rglob("*")) if p.is_file() and p.name != "manifest.csv"]).to_csv(
                                  compare / "manifest.csv", index=False)

            seal_compare()
            found, _, checked_inputs, checked_contracts = read_compare(compare, context, context_paths, step)
            compare_models.extend(found)
            check(f"{step}_three_seeds_manifest_and_context", len(found) == 3 and checked_inputs.passed.all()
                  and checked_contracts.canonical_context_matches.all())
            env["inputs"]["seeds"]["222"]["version"] = 9
            seal_compare()
            rejects(f"{step}_wrong_training_version_rejected", lambda: read_compare(compare, context, context_paths, step))
            env["inputs"]["seeds"]["222"]["version"] = versions[222]
            env["neuralhydrology_commit_of_runs"] = "wrong" if step == "A4" else {"B0": A4_LIBRARY_COMMIT}
            seal_compare()
            rejects(f"{step}_wrong_library_commit_rejected", lambda: read_compare(compare, context, context_paths, step))
            env["neuralhydrology_commit_of_runs"] = commit if step == "A4" else {"B0": commit}
            env["inputs"]["a3"]["files"][context_paths[0].name]["sha256"] = "b" * 64
            seal_compare()
            rejects(f"{step}_different_A3_context_rejected", lambda: read_compare(compare, context, context_paths, step))
            env["inputs"]["a3"]["files"][context_paths[0].name]["sha256"] = sha256(context_paths[0])
            if step == "B0":
                summary["bstar_threshold_val_nse_sd"] = .5
                seal_compare()
                rejects("B0_wrong_validation_std_rejected", lambda: read_b0(compare, context, context_paths))
                summary["bstar_threshold_val_nse_sd"] = float(np.std(list(val.values()), ddof=1))
            seal_compare()
            summary_path.write_text("{}", encoding="utf-8")
            rejects(f"{step}_manifest_tampering_rejected", lambda: read_compare(compare, context, context_paths, step))
            seal_compare()
            missing = summary["contracts"].pop("333")
            seal_compare()
            rejects(f"{step}_missing_seed_rejected", lambda: read_compare(compare, context, context_paths, step))
            summary["contracts"]["333"] = missing
            seal_compare()
        # B1 Compare thực chứa sáu NPZ, metadata/env và summary nested theo bước.
        compare = root / "b1"
        (compare / "contracts").mkdir(parents=True)
        summary = dict(contracts={}, saved_epoch={}, seed_test_nse_median={}, seed_val_nse_median={})
        for st in ("B0", *B1_STEPS):
            summary["saved_epoch"][st] = {str(seed): 25 for seed in B1_SEEDS}
            summary["seed_test_nse_median"][st] = {str(seed): 1. for seed in B1_SEEDS}
            summary["seed_val_nse_median"][st] = {"111": .74, "222": .75, "333": .76}
        summary["bstar_threshold_val_nse_sd"] = float(np.std([.74, .75, .76], ddof=1))
        env = dict(notebook="05_B1_LamaHCEExt_Compare", neuralhydrology_commit_of_runs=B1_LIBRARY_COMMIT,
                   inputs={st: {str(seed): dict(notebook=f"{'03' if st == 'B1a' else '04'}_{st}_LamaHCEExt_Train_Seed{seed}",
                                                version=1) for seed in B1_SEEDS} for st in B1_STEPS})
        env["inputs"]["a3"] = dict(version=1, files={p.name: {"sha256": sha256(p)} for p in context_paths})
        for st, seed in itertools.product(B1_STEPS, B1_SEEDS):
            name = f"SeqLSTM_Q_{st}_seed{seed}"
            relative = f"contracts/{name}_test_S0_lead1.npz"
            meta = dict(run=name, period="test", scenario="S0", lead=1, forecast_horizon=1, seed=seed,
                        unit="mm/day_area_gov", config_sha256="a" * 64, library_commit=B1_LIBRARY_COMMIT,
                        clip_to_zero=True, selected_epoch=25)
            np.savez_compressed(compare / relative, **payload, metadata_json=json.dumps(meta))
            summary["contracts"][f"{st}_{seed}"] = f"b1_compare/{relative}"

        def seal_b1():
            (compare / "b1_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (compare / "environment.json").write_text(json.dumps(env), encoding="utf-8")
            pd.DataFrame([{"tệp": p.relative_to(compare).as_posix(), "sha256": sha256(p)}
                          for p in sorted(compare.rglob("*")) if p.is_file() and p.name != "manifest.csv"]).to_csv(
                              compare / "manifest.csv", index=False)

        seal_b1()
        for st in B1_STEPS:
            found, _, checked_inputs, checked_contracts = read_b1(compare, context, context_paths, st)
            compare_models.extend(found)
            check(f"{st}_shared_Compare_three_seeds_manifest_and_context", len(found) == 3
                  and checked_inputs.passed.all() and checked_contracts.canonical_context_matches.all())
        env["inputs"]["B1a"]["222"]["version"] = 2
        seal_b1()
        rejects("B1_wrong_training_version_rejected", lambda: read_b1(compare, context, context_paths, "B1a"))
        env["inputs"]["B1a"]["222"]["version"] = 1
        env["inputs"]["B1a"]["222"]["notebook"] = "04_B1b_LamaHCEExt_Train_Seed222"
        seal_b1()
        rejects("B1_wrong_training_notebook_rejected", lambda: read_b1(compare, context, context_paths, "B1a"))
        env["inputs"]["B1a"]["222"]["notebook"] = "03_B1a_LamaHCEExt_Train_Seed222"
        env["neuralhydrology_commit_of_runs"] = A4_LIBRARY_COMMIT
        seal_b1()
        rejects("B1_wrong_library_commit_rejected", lambda: read_b1(compare, context, context_paths, "B1b"))
        env["neuralhydrology_commit_of_runs"] = B1_LIBRARY_COMMIT
        env["inputs"]["a3"]["files"][context_paths[0].name]["sha256"] = "b" * 64
        seal_b1()
        rejects("B1_different_A3_context_rejected", lambda: read_b1(compare, context, context_paths, "B1b"))
        env["inputs"]["a3"]["files"][context_paths[0].name]["sha256"] = sha256(context_paths[0])
        summary["saved_epoch"]["B1b"]["333"] = 24
        seal_b1()
        rejects("B1_selected_epoch_mismatch_rejected", lambda: read_b1(compare, context, context_paths, "B1b"))
        summary["saved_epoch"]["B1b"]["333"] = 25
        summary["bstar_threshold_val_nse_sd"] = .5
        seal_b1()
        rejects("B1_wrong_B0_validation_std_rejected", lambda: read_b1(compare, context, context_paths, "B1a"))
        summary["bstar_threshold_val_nse_sd"] = float(np.std([.74, .75, .76], ddof=1))
        value = summary["contracts"].pop("B1b_333")
        seal_b1()
        rejects("B1_missing_shared_summary_contract_rejected", lambda: read_b1(compare, context, context_paths, "B1a"))
        summary["contracts"]["B1b_333"] = value
        extra = compare / "contracts/extra.npz"
        extra.write_bytes(b"not_a_contract")
        seal_b1()
        rejects("B1_extra_contract_rejected", lambda: read_b1(compare, context, context_paths, "B1b"))
        extra.unlink()
        seal_b1()
        manifest = pd.read_csv(compare / "manifest.csv")
        manifest.loc[~manifest["tệp"].str.contains("B1b_seed333")].to_csv(compare / "manifest.csv", index=False)
        rejects("B1_missing_other_step_manifest_contract_rejected", lambda: read_b1(compare, context, context_paths, "B1a"))
        seal_b1()
        (compare / "b1_summary.json").write_text("{}", encoding="utf-8")
        rejects("B1_manifest_tampering_rejected", lambda: read_b1(compare, context, context_paths, "B1b"))
        seal_b1()
        b0_reference = [f for f in compare_models if "_B0_" in f.name]
        b1_reference = [f for f in compare_models if "_B1b_" in f.name]
        check_b1_b0_reference(b0_reference, b1_reference)
        check("B1_B0_reference_consistent_across_Compare", True)
        for field, wrong in (
                ("bstar_threshold_val_nse_sd", .5),
                ("reference_B0_seed_val_nse_median", {"111": .76, "222": .75, "333": .74}),
                ("reference_B0_saved_epoch", {"111": 24, "222": 25, "333": 25})):
            changed = replace(b1_reference[0], metadata=dict(b1_reference[0].metadata, **{field: wrong}))
            rejects(f"B1_B0_cross_Compare_{field}_mismatch_rejected",
                    lambda: check_b1_b0_reference(b0_reference, [changed]))
        # B2 dùng đúng schema Compare Version 1, cả tên nhúng trước/sau đổi số.
        b2 = root / "b2"
        (b2 / "contracts").mkdir(parents=True)
        b1_identity = dict(notebook="05_B1_LamaHCEExt_Compare", version=1,
            summary_sha256=sha256(compare / "b1_summary.json"), environment_sha256=sha256(compare / "environment.json"))
        b2_val = {"111": .742, "222": .752, "333": .762}
        b2_delta = dict(comparison="B2 - B1b", B1b_validation_NSE_mean=.75, B2_validation_NSE_mean=.752,
            validation_NSE_mean_difference=.002, B0_Bstar_threshold=summary["bstar_threshold_val_nse_sd"],
            common_mask_evaluation_completed=False, Bstar_selected=False)
        b2_summary = dict(complete=True, step="B2", scope="test_S0_lead1_three_seeds", library_commit=B2_LIBRARY_COMMIT,
            training_versions_expected={str(s): 1 for s in B2_SEEDS}, contracts={},
            saved_epoch={str(s): 25 for s in B2_SEEDS}, seed_val_nse_median=b2_val,
            seed_test_nse_median={str(s): 1. for s in B2_SEEDS}, seed_test_nse_mean=1., seed_test_nse_sd=0.,
            B1b_source=b1_identity, B1b_validation_comparison=b2_delta, Bstar_selected=False, operational_scenarios_run=False)
        b2_env = dict(notebook="07_B2_LamaHCEExt_Compare", library_commit_of_runs=B2_LIBRARY_COMMIT,
            inputs={"B2": {}, "B1_Compare": b1_identity,
                    "A3": dict(version=1, files={p.name: sha256(p) for p in context_paths})}, Bstar_selected=False)
        b2_seeds, b2_sources, b2_audits = [], [], []
        for seed in B2_SEEDS:
            notebook = f"06_B2_LamaHCEExt_Train_Seed{seed}"
            embedded = notebook if seed == 111 else f"07_B2_LamaHCEExt_Train_Seed{seed}"
            source_hash = str(seed)[0] * 64
            relative = f"contracts/SeqLSTM_Q_B2_seed{seed}_test_S0_lead1.npz"
            b2_summary["contracts"][str(seed)] = relative
            b2_env["inputs"]["B2"][str(seed)] = dict(notebook=notebook, version_expected=1,
                notebook_in_environment=embedded, environment_sha256=source_hash)
            b2_meta = dict(run=f"SeqLSTM_Q_B2_seed{seed}", period="test", scenario="S0", lead=1,
                forecast_horizon=1, seed=seed, unit="mm/day_area_gov", config_sha256="a"*64,
                library_commit=B2_LIBRARY_COMMIT, clip_to_zero=True, selected_epoch=25,
                training_notebook_metadata=embedded)
            b2_payload = dict(payload)
            if seed == 222:
                # B2 vẫn dự báo ngày có obs nhưng persistence thiếu (Q hôm trước thiếu).
                changed_sim = context["obs"].copy()
                gap = np.isfinite(context["obs"]) & ~np.isfinite(context["persistence"])
                changed_sim[gap] += .2
                b2_payload.update(sim_raw_mm_day=changed_sim, sim_mm_day=changed_sim)
            fixture_nse = []
            for j in range(len(ids)):
                valid = np.isfinite(context["obs"][:, j]) & np.isfinite(b2_payload["sim_mm_day"][:, j])
                observed = context["obs"][valid, j]
                predicted = b2_payload["sim_mm_day"][valid, j]
                fixture_nse.append(1 - np.sum((predicted - observed)**2) / np.sum((observed - observed.mean())**2))
            native_median = float(np.nanmedian(fixture_nse))
            b2_summary["seed_test_nse_median"][str(seed)] = native_median
            np.savez_compressed(b2 / relative, **b2_payload, metadata_json=json.dumps(b2_meta))
            b2_seeds.append(dict(seed=seed, selected_epoch=25, validation_NSE_median=b2_val[str(seed)],
                test_NSE_median=native_median, test_KGE_median=1., train_hours=.1))
            b2_sources.append(dict(seed=seed, notebook=notebook, version=1, notebook_in_environment=embedded,
                metadata_name_matches_source=notebook==embedded, environment_sha256=source_hash))
            b2_audits.append({"hạt giống": seed, "tệp": Path(relative).name, "selected_epoch": 25,
                "clip_to_zero": True, "dự báo âm trước chặn": 0,
                "ngày-lưu vực có sim": int(np.isfinite(context["obs"]).sum()), "lỗi hợp đồng": "-",
                "max |ΔNSE| so với thư viện": 0., "NaN NSE trùng thư viện": True, "mặt nạ obs trùng A3": True,
                "giá trị obs khớp A3 trong sai số float32": True, "đạt": True,
                "NSE trung vị từ NPZ (obs A3)": native_median})
        test_values = list(b2_summary["seed_test_nse_median"].values())
        b2_summary["seed_test_nse_mean"] = float(np.mean(test_values))
        b2_summary["seed_test_nse_sd"] = float(np.std(test_values, ddof=1))

        def seal_b2():
            (b2 / "b2_summary.json").write_text(json.dumps(b2_summary), encoding="utf-8")
            (b2 / "environment.json").write_text(json.dumps(b2_env), encoding="utf-8")
            table = pd.DataFrame(b2_seeds)
            table.to_csv(b2 / "B2_seed_table.csv", index=False)
            table.drop(columns=["seed", "selected_epoch"]).agg(["mean", "std"]).T.reset_index(names="metric").to_csv(
                b2 / "B2_seed_stats.csv", index=False)
            pd.DataFrame(b2_sources).to_csv(b2 / "B2_input_sources.csv", index=False)
            pd.DataFrame(b2_audits).to_csv(b2 / "B2_contract_check.csv", index=False)
            pd.DataFrame([b2_delta]).to_csv(b2 / "B2_B1b_reported_summary.csv", index=False)
            (b2 / "B2_Seeds.png").write_bytes(b"synthetic_image_placeholder_not_experimental_output")
            pd.DataFrame([dict(file=p.relative_to(b2).as_posix(), bytes=p.stat().st_size, sha256=sha256(p))
                          for p in sorted(b2.rglob("*")) if p.is_file() and p.name != "manifest.csv"]).to_csv(
                              b2 / "manifest.csv", index=False)

        seal_b2()
        b2_models, _, b2_inputs, b2_contracts = read_b2(b2, context, context_paths, b0_reference, compare)
        check("B2_three_seeds_manifest_and_context", len(b2_models)==3 and b2_inputs.passed.all()
              and b2_contracts.canonical_context_matches.all())
        check("B2_preserves_actual_and_embedded_notebook_names", b2_contracts.iloc[1].training_notebook.startswith("06_")
              and b2_contracts.iloc[1].embedded_training_notebook.startswith("07_"))
        masked_member = next(f for f in b2_models if f.seed==222)
        with_persistence = np.nanmedian([metrics(masked_member.obs[:, j], masked_member.sim[:, j],
            masked_member.persistence[:, j])["NSE"] for j in range(len(ids))])
        checked_222 = b2_contracts.set_index("seed").loc[222]
        check("B2_NSE_audit_keeps_obs_sim_days_with_missing_persistence",
              np.isclose(checked_222.NSE_audit_median, b2_summary["seed_test_nse_median"]["222"], atol=1e-12)
              and not np.isclose(checked_222.NSE_audit_median, with_persistence, atol=1e-9, rtol=1e-9))
        for label, owner, key, wrong in (
            ("training_version", b2_env["inputs"]["B2"]["222"], "version_expected", 2),
            ("boolean_training_version", b2_env["inputs"]["B2"]["222"], "version_expected", True),
            ("library_commit", b2_env, "library_commit_of_runs", B1_LIBRARY_COMMIT),
            ("A3_context", b2_env["inputs"]["A3"]["files"], context_paths[0].name, "b"*64),
            ("A3_hash_type", b2_env["inputs"]["A3"]["files"], context_paths[0].name, {"sha256": sha256(context_paths[0])}),
            ("B1_source_hash", b2_summary["B1b_source"], "summary_sha256", "b"*64),
            ("B0_threshold", b2_delta, "B0_Bstar_threshold", .5),
            ("selected_epoch", b2_summary["saved_epoch"], "333", 24),
            ("embedded_name", b2_env["inputs"]["B2"]["333"], "notebook_in_environment", "unknown"),
            ("audit_NSE", b2_audits[0], "NSE trung vị từ NPZ (obs A3)", .5)):
            previous = owner[key]
            owner[key] = wrong
            seal_b2()
            rejects(f"B2_wrong_{label}_rejected", lambda: read_b2(b2, context, context_paths, b0_reference, compare))
            if label == "audit_NSE":
                try:
                    read_b2(b2, context, context_paths, b0_reference, compare)
                except ValueError as error:
                    check("B2_audit_failure_reports_field_and_values", "audit.NSE_median_obs_sim" in str(error)
                          and "nhận" in str(error) and "cần" in str(error))
            owner[key] = previous
        value = b2_summary["contracts"].pop("333")
        seal_b2()
        rejects("B2_missing_seed_rejected", lambda: read_b2(b2, context, context_paths, b0_reference, compare))
        b2_summary["contracts"]["333"] = value
        extra = b2 / "contracts/extra.npz"
        extra.write_bytes(b"not_a_contract")
        seal_b2()
        rejects("B2_extra_NPZ_rejected", lambda: read_b2(b2, context, context_paths, b0_reference, compare))
        extra.unlink()
        seal_b2()
        (b2 / "b2_summary.json").write_text("{}", encoding="utf-8")
        rejects("B2_manifest_tampering_rejected", lambda: read_b2(b2, context, context_paths, b0_reference, compare))
        seal_b2()
        # B3: hai mức, ba seed; cùng schema của 09_B3 Compare Version 1.
        b3 = root / "b3"
        (b3 / "contracts").mkdir(parents=True)
        b2_identity = dict(notebook="07_B2_LamaHCEExt_Compare", version=1,
            summary_sha256=sha256(b2 / "b2_summary.json"), environment_sha256=sha256(b2 / "environment.json"))
        b3_summary = dict(complete=True, step="B3", scope="test_S0_lead1_two_mask_levels_three_seeds_each",
            library_commit=B3_LIBRARY_COMMIT, B2_source=b2_identity, mask_levels={},
            bstar_threshold_val_nse_sd=summary["bstar_threshold_val_nse_sd"],
            Bstar_selected=False, operational_scenarios_run=False, multi_lead_run=False)
        b3_env = dict(notebook="09_B3_LamaHCEExt_Compare", library_commit_of_runs=B3_LIBRARY_COMMIT,
            inputs={"B3": {}, "B2_Compare": b2_identity, "A3": {"version": 1,
                    "files": {p.name: sha256(p) for p in context_paths}}}, Bstar_selected=False)
        b3_seed_rows, b3_sources, b3_audits, b3_deltas = [], [], [], []
        for level, (p_step, p_sequence) in B3_MASKS.items():
            block = dict(nan_step_probability=p_step, nan_sequence_probability=p_sequence,
                training_versions_expected={}, saved_epoch={}, seed_val_nse_median={}, seed_test_nse_median={}, contracts={})
            b3_summary["mask_levels"][level] = block
            b3_env["inputs"]["B3"][level] = {}
            for j, seed in enumerate(B3_SEEDS):
                key = str(seed)
                notebook = f"08_B3_LamaHCEExt_Train_{level}_Seed{seed}"
                relative = f"contracts/SeqLSTM_Q_B3_{level}_seed{seed}_test_S0_lead1.npz"
                with np.load(b2 / f"contracts/SeqLSTM_Q_B2_seed{seed}_test_S0_lead1.npz", allow_pickle=False) as data:
                    b3_payload = {k: np.array(data[k]) for k in data.files if k != "metadata_json"}
                raw = b3_payload["sim_raw_mm_day"].copy() + (j+1) * .003
                sim = np.maximum(raw, 0)
                b3_payload.update(sim_raw_mm_day=raw, sim_mm_day=sim)
                native_nse = float(np.nanmedian([metrics(context["obs"][:, k], sim[:, k])["NSE"] for k in range(len(ids))]))
                epoch = (23,19,14)[j] if level == "P10S12" else (14,11,9)[j]
                meta = dict(run=f"SeqLSTM_Q_B3_{level}_seed{seed}", period="test", scenario="S0", lead=1,
                    forecast_horizon=1, seed=seed, unit="mm/day_area_gov", config_sha256="c"*64,
                    library_commit=B3_LIBRARY_COMMIT, clip_to_zero=True, selected_epoch=epoch, step="B3", mask_level=level,
                    nan_step_probability=p_step, nan_sequence_probability=p_sequence, training_notebook_metadata=notebook)
                np.savez_compressed(b3 / relative, **b3_payload, metadata_json=json.dumps(meta))
                own_hash = ("d" if level=="P10S12" else "e") * 64
                b3_env["inputs"]["B3"][level][key] = dict(notebook=notebook, version_expected=1, environment_sha256=own_hash)
                block["training_versions_expected"][key] = 1
                block["saved_epoch"][key] = epoch
                block["seed_val_nse_median"][key] = .745+j*.005
                block["seed_test_nse_median"][key] = native_nse
                block["contracts"][key] = relative
                b3_seed_rows.append(dict(mask_level=level, seed=seed, selected_epoch=epoch, nan_step_probability=p_step,
                    nan_sequence_probability=p_sequence, validation_NSE_median=block["seed_val_nse_median"][key],
                    test_NSE_median=native_nse, test_KGE_median=.8+j*.005, train_hours=.1+j*.01))
                b3_sources.append(dict(mask_level=level, seed=seed, notebook=notebook, version_expected=1,
                    environment_sha256=own_hash, B2_environment_sha256=b2_env["inputs"]["B2"][key]["environment_sha256"]))
                b3_audits.append({"mask_level": level, "hạt giống": seed, "tệp": Path(relative).name,
                    "selected_epoch": epoch, "clip_to_zero": True, "dự báo âm trước chặn": int((raw<0).sum()),
                    "ngày-lưu vực có sim": int(np.isfinite(sim).sum()), "lỗi hợp đồng": "-",
                    "max |ΔNSE| so với thư viện": 0., "NaN NSE trùng thư viện": True, "mặt nạ obs trùng A3": True,
                    "giá trị obs khớp A3 trong sai số float32": True, "đạt": True,
                    "NSE trung vị obs run (đối chiếu thư viện)": native_nse,
                    "NSE trung vị từ NPZ (obs A3)": native_nse})
            value = float(np.mean(list(block["seed_val_nse_median"].values())))
            reference_value = float(np.mean(list(b2_summary["seed_val_nse_median"].values())))
            b3_deltas.append(dict(comparison=f"B3 {level} - B2", B2_validation_NSE_mean=reference_value,
                B3_validation_NSE_mean=value, validation_NSE_mean_difference=value-reference_value,
                B0_Bstar_threshold=summary["bstar_threshold_val_nse_sd"],
                common_mask_evaluation_completed=False, Bstar_selected=False))

        def seal_b3():
            (b3 / "b3_summary.json").write_text(json.dumps(b3_summary), encoding="utf-8")
            (b3 / "environment.json").write_text(json.dumps(b3_env), encoding="utf-8")
            seed_table = pd.DataFrame(b3_seed_rows)
            seed_table.to_csv(b3 / "B3_seed_table.csv", index=False)
            stats_table = seed_table.melt(id_vars=["mask_level","seed"],
                value_vars=["validation_NSE_median","test_NSE_median","test_KGE_median","train_hours"], var_name="metric")
            stats_table.groupby(["mask_level","metric"],sort=False)["value"].agg(["mean","std"]).reset_index().to_csv(
                b3 / "B3_seed_stats.csv", index=False)
            pd.DataFrame(b3_sources).to_csv(b3 / "B3_input_sources.csv", index=False)
            pd.DataFrame(b3_audits).to_csv(b3 / "B3_contract_check.csv", index=False)
            pd.DataFrame(b3_deltas).to_csv(b3 / "B3_B2_reported_summary.csv", index=False)
            (b3 / "B3_Seeds.png").write_bytes(b"synthetic_plot_placeholder_not_experimental_output")
            pd.DataFrame([dict(file=p.relative_to(b3).as_posix(),bytes=p.stat().st_size,sha256=sha256(p))
                          for p in sorted(b3.rglob("*")) if p.is_file() and p.name!="manifest.csv"]).to_csv(b3 / "manifest.csv",index=False)

        seal_b3()
        b3_models, _, b3_inputs, b3_contracts = read_b3(b3, context, context_paths, b0_reference, b2)
        check("B3_six_contracts_two_levels_three_seeds", len(b3_models)==6 and b3_inputs.passed.all()
              and b3_contracts.canonical_context_matches.all())
        check("B3_audit_obs_sim_without_persistence", b3_contracts.NSE_audit_abs_difference.le(1e-12).all()
              and b3_contracts.loc[b3_contracts.seed.eq(222),"NSE_audit_median"].lt(1.).all())
        for label, target, key, wrong in (
            ("wrong_mask", b3_summary["mask_levels"]["P10S12"], "nan_step_probability", .2),
            ("wrong_version", b3_summary["mask_levels"]["P05S05"]["training_versions_expected"], "111", 2),
            ("boolean_version", b3_env["inputs"]["B3"]["P10S12"]["111"], "version_expected", True),
            ("wrong_A3_hash", b3_env["inputs"]["A3"]["files"], "A3_persistence_predictions.npz", "0"*64),
            ("wrong_B2_hash", b3_summary["B2_source"], "summary_sha256", "0"*64),
            ("wrong_B0_threshold", b3_summary, "bstar_threshold_val_nse_sd", .5),
            ("wrong_source_B2_environment", b3_sources[0], "B2_environment_sha256", "0"*64),
            ("wrong_source_notebook", b3_sources[0], "notebook", "unknown"),
            ("wrong_seed_mask", b3_seed_rows[0], "nan_sequence_probability", .5),
            ("wrong_negative_count", b3_audits[0], "dự báo âm trước chặn", 123),
            ("wrong_audit_NSE", b3_audits[1], "NSE trung vị từ NPZ (obs A3)", .5),
            ("failed_audit", b3_audits[0], "đạt", False),
            ("wrong_delta", b3_deltas[0], "validation_NSE_mean_difference", .1),
            ("premature_Bstar", b3_summary, "Bstar_selected", True)):
            previous = target[key]
            target[key] = wrong
            seal_b3()
            rejects(f"B3_{label}_rejected", lambda: read_b3(b3, context, context_paths, b0_reference, b2))
            target[key] = previous
        seal_b3()
        special = b3 / b3_summary["mask_levels"]["P10S12"]["contracts"]["111"]
        original = special.read_bytes()
        with np.load(special,allow_pickle=False) as data:
            b3_payload = {k:np.array(data[k]) for k in data.files if k!="metadata_json"}
            original_meta = json.loads(str(data["metadata_json"].item()))
        for key, wrong in (("mask_level","P05S05"),("nan_sequence_probability",.05),("selected_epoch",True),
                           ("library_commit","f229e02"),("training_notebook_metadata","unknown")):
            changed = dict(original_meta,**{key:wrong})
            np.savez_compressed(special,**b3_payload,metadata_json=json.dumps(changed))
            seal_b3()
            rejects(f"B3_contract_{key}_rejected",lambda:read_b3(b3,context,context_paths,b0_reference,b2))
        special.write_bytes(original)
        extra = b3 / "contracts/extra.npz"
        extra.write_bytes(b"invalid")
        seal_b3()
        rejects("B3_extra_contract_rejected",lambda:read_b3(b3,context,context_paths,b0_reference,b2))
        extra.unlink()
        seal_b3()
        special.unlink()
        seal_b3()
        rejects("B3_missing_contract_rejected",lambda:read_b3(b3,context,context_paths,b0_reference,b2))
        special.write_bytes(original)
        seal_b3()
        (b3 / "b3_summary.json").write_text("{}",encoding="utf-8")
        rejects("B3_manifest_tampering_rejected",lambda:read_b3(b3,context,context_paths,b0_reference,b2))
        seal_b3()

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
        expanded = models + compare_models + [Forecast("Persistence_qmean_aux", sim=context["persistence"] * .8,
                    raw=context["persistence"] * .8, **context).validate()]
        expanded_tables, expanded_mask = evaluate_group(expanded, levels, floods)
        expanded_pairs = pairwise_tests(expanded_tables["metrics"])
        check("A4_B0_B1_scored_together_23_runs", len(expanded) == 23
              and expanded_tables["metrics"].run.nunique() == 23
              and np.array_equal(expanded_mask, shared))
        check("A4_B0_B1_Holm_231_pairs_per_metric", set(expanded_pairs.planned_family_size) == {231}
              and len(expanded_pairs) == 693)
        b2_tables, b2_mask = evaluate_group(expanded + b2_models, levels, floods)
        b2_pairs = pairwise_tests(b2_tables["metrics"])
        check("A4_B0_B1_B2_scored_together_26_runs", b2_tables["metrics"].run.nunique()==26
              and np.array_equal(b2_mask, shared))
        check("A4_B0_B1_B2_Holm_300_pairs_per_metric", set(b2_pairs.planned_family_size)=={300} and len(b2_pairs)==900)

        b3_tables, b3_mask = evaluate_group(expanded + b2_models + b3_models, levels, floods)
        b3_pairs = pairwise_tests(b3_tables["metrics"])
        check("A4_B0_B1_B2_B3_scored_together_32_runs", b3_tables["metrics"].run.nunique()==32
              and np.array_equal(b3_mask,b2_mask))
        check("A4_B0_B1_B2_B3_Holm_465_pairs_per_metric", set(b3_pairs.planned_family_size)=={465}
              and len(b3_pairs)==1395)

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

        # Kiểm thử phân vùng D (Phương án A: chấm riêng evaluation_output/D/ cho 6 hoặc 9 cặp 365/730).
        c2_fix_dir = root / "c2_fix"
        c2_fix_dir.mkdir()
        (c2_fix_dir / "manifest.csv").write_text("file,bytes,sha256\n", encoding="utf-8")
        c2_fix_forecasts = []
        for conf in C2_CONFIGS:
            core, depth_s, level_s, lr_s = conf.split("_")
            n_p = 84106 if int(level_s[1:]) == 1 else 254425
            for seed in C2_SEEDS:
                c2_name = f"SF_{conf}_C2_seed{seed}"
                c2_npz = c2_fix_dir / f"{c2_name}_test_S0_lead1.npz"
                c2_meta = dict(
                    run=c2_name, period="test", scenario="S0", lead=1, forecast_horizon=1, seed=seed,
                    unit="mm/day_area_gov", config_sha256="c" * 64, library_commit=FREEZE_C, clip_to_zero=True,
                    step="C2", model="sequential_forecast", core=core, depth=int(depth_s[1:]),
                    level=int(level_s[1:]), learning_rate=float(lr_s[2:]), config=conf, n_params=n_p,
                    selected_epoch=20,
                )
                np.savez_compressed(
                    c2_npz, dates=dates.to_numpy(dtype="datetime64[D]"),
                    origin_dates=dates.to_numpy(dtype="datetime64[D]"),
                    persistence_source_dates=full_dates[:-1].to_numpy(dtype="datetime64[D]"),
                    basin_ids=np.array(ids), area_gov_km2=area, obs_mm_day=context["obs"],
                    persistence_mm_day=context["persistence"], sim_raw_mm_day=context["obs"] * .95,
                    sim_mm_day=np.maximum(context["obs"] * .95, 0), metadata_json=json.dumps(c2_meta),
                )
                f_c2 = read_contract(c2_npz)
                f_c2.metadata["source_file_sha256"] = sha256(c2_npz)
                c2_fix_forecasts.append(f_c2)
        c2_by_name = {f.name: f for f in c2_fix_forecasts}

        def build_d_fixture(d_dir, include_s4d):
            if d_dir.exists():
                for item in sorted(d_dir.rglob("*"), reverse=True):
                    if item.is_file():
                        item.unlink()
                    else:
                        item.rmdir()
            (d_dir / "contracts").mkdir(parents=True, exist_ok=True)
            confs = list(D_CONFIGS if include_s4d else D_CONFIGS[:-1])
            cfg_rows, con_rows, hand_rows, demo_rows, out_rels = [], [], [], [], []
            d_members = []
            for conf in confs:
                for seed in C2_SEEDS:
                    ref = c2_by_name[f"SF_{conf}_C2_seed{seed}"]
                    d_name = f"SF_{conf}_D_seed{seed}"
                    rel = f"contracts/{d_name}_test_S0_lead1.npz"
                    d_raw = context["obs"] * .98
                    d_sim = np.maximum(d_raw, 0)
                    d_meta = dict(
                        run=d_name, step="D", config=conf, core=ref.metadata["core"],
                        depth=ref.metadata["depth"], level=1, n_params=ref.metadata["n_params"],
                        learning_rate=ref.metadata["learning_rate"], seed=seed, selected_epoch=22,
                        period="test", scenario="S0", lead=1, forecast_horizon=1, unit="mm/day_area_gov",
                        seq_length=730, hindcast_steps=729, forecast_steps=1, clip_to_zero=True,
                        grad_guard_enabled=False, grad_guard_skipped_steps=0, library_commit=FREEZE_C,
                        config_sha256="d" * 64, checkpoint_sha256="e" * 64, scaler_sha256="f" * 64,
                        training_notebook=f"16_D_Train_{conf}_Seed{seed}", training_environment_sha256="1" * 64,
                        baseline_run=ref.name, baseline_contract_sha256=ref.metadata["source_file_sha256"],
                    )
                    np.savez_compressed(
                        d_dir / rel, dates=dates.to_numpy(dtype="datetime64[D]"),
                        origin_dates=dates.to_numpy(dtype="datetime64[D]"),
                        persistence_source_dates=full_dates[:-1].to_numpy(dtype="datetime64[D]"),
                        basin_ids=np.array(ids), area_gov_km2=area, obs_mm_day=context["obs"],
                        persistence_mm_day=context["persistence"], sim_raw_mm_day=d_raw, sim_mm_day=d_sim,
                        metadata_json=json.dumps(d_meta),
                    )
                    f_d = read_contract(d_dir / rel)
                    d_members.append((ref, f_d))
                    nse_med = float(np.nanmedian([metrics(f_d.obs[:, j], f_d.sim[:, j])["NSE"] for j in range(len(ids))]))
                    npz_sha = sha256(d_dir / rel)
                    is_demo = seed == 111
                    cfg_rows.append(dict(
                        configuration=conf, seed=seed, seq_length_365=365, seq_length_730=730,
                        n_params=ref.metadata["n_params"], configuration_difference="seq_length",
                        grad_guard_enabled_365=False, grad_guard_skipped_steps_365=0,
                        grad_guard_enabled_730=False, grad_guard_skipped_steps_730=0,
                        strict_pair_equivalent=True,
                    ))
                    con_rows.append(dict(
                        run=d_name, baseline=ref.name, seed=seed, selected_epoch=22, IDs=len(ids),
                        days=len(dates), native_nse_median=nse_med, finite_predictions=int(np.isfinite(d_raw).sum()),
                        grad_guard_enabled=False, grad_guard_skipped_steps=0, SHA256=npz_sha, passed=True,
                    ))
                    hand_rows.append(dict(
                        configuration=conf, seed=seed, run=d_name, notebook=d_meta["training_notebook"],
                        sessions=1, run_dir=f"c_runs/D_{conf}_seed{seed}_0101_000000",
                        selected_epoch=22, n_params=ref.metadata["n_params"], library_commit=FREEZE_C,
                        seq_length=730, grad_guard_enabled=False, grad_guard_skipped_steps=0,
                        validation_NSE=.78 - seed * 1e-5, test_NSE_native=nse_med, demo=is_demo,
                        sha256_best_model_pt="e" * 64, sha256_config_yml="d" * 64,
                        sha256_train_data_scaler_yml="f" * 64, contract_npz=rel, sha256_contract_npz=npz_sha,
                    ))
                    if is_demo:
                        demo_rows.append(dict(
                            configuration=conf, seed=seed, run=d_name, selected_epoch=22,
                            validation_NSE=.78 - seed * 1e-5, gap_to_second_seed=1.11e-3, tie=False,
                            test_NSE_native=nse_med, notebook=d_meta["training_notebook"], sessions=1,
                            run_dir=f"c_runs/D_{conf}_seed{seed}_0101_000000", contract_npz=rel,
                        ))
                    out_rels.append(rel)
            mask_d = common_mask([m for pair in d_members for m in pair])
            np.savez_compressed(d_dir / "D_common_mask.npz", dates=dates.to_numpy(dtype="datetime64[D]"),
                                basin_ids=np.array(ids), mask=mask_d)
            pd.DataFrame(cfg_rows).to_csv(d_dir / "D_config_check.csv", index=False)
            pd.DataFrame(con_rows).to_csv(d_dir / "D_contract_check.csv", index=False)
            pd.DataFrame(hand_rows).to_csv(d_dir / "D_handoff.csv", index=False)
            pd.DataFrame(demo_rows).to_csv(d_dir / "D_demo_runs.csv", index=False)
            for empty_csv in ("D_seed_scores.csv", "D_score_stats.csv", "D_ensemble_scores.csv", "D_cost.csv",
                              "D_cost_ratios.csv", "D_basin_scores.csv", "D_input_sources.csv"):
                (d_dir / empty_csv).write_text("col\n1\n", encoding="utf-8")
            (d_dir / "D_Length_Comparison.png").write_bytes(b"png")
            d_inputs = dict(C2_compare_version_expected=1, C2_manifest_sha256=sha256(c2_fix_dir / "manifest.csv"),
                            A3={p.name: sha256(p) for p in context_paths})
            d_sum = dict(
                complete=True, notebook=D_COMPARE_NOTEBOOK, step="D", library_commit=FREEZE_C,
                configs=confs, seeds=list(C2_SEEDS), S4D_included=include_s4d, contracts=len(out_rels),
                IDs=len(ids), forecast_horizon=1, common_day_IDs=int(mask_d.sum()),
                IDs_with_days=int(mask_d.any(axis=0).sum()), grad_guard_runs_with_skips=[],
                all_pairs_strictly_equivalent=True, scope="test_S0_common_all_C2_and_D",
                inputs=d_inputs, output_contracts=out_rels,
            )
            d_env = dict(notebook=D_COMPARE_NOTEBOOK, source_neuralhydrology_commit=FREEZE_C, **d_inputs)
            (d_dir / "d_summary.json").write_text(json.dumps(d_sum), encoding="utf-8")
            (d_dir / "environment.json").write_text(json.dumps(d_env), encoding="utf-8")

            def seal_d():
                pd.DataFrame([
                    dict(file=p.relative_to(d_dir).as_posix(), bytes=p.stat().st_size, sha256=sha256(p))
                    for p in sorted(d_dir.rglob("*")) if p.is_file() and p.name != "manifest.csv"
                ]).to_csv(d_dir / "manifest.csv", index=False)

            seal_d()
            return seal_d

        for include_s4d, expected_n in ((False, 6), (True, 9)):
            d_fix = root / f"d_fix_{expected_n}"
            seal_d = build_d_fixture(d_fix, include_s4d)
            sel_c2, sel_d, d_in_chk, d_con_chk, d_src_sum = read_d_compare(
                d_fix, c2_fix_forecasts, c2_fix_dir, context, context_paths)
            check(f"D_compare_{expected_n}_contracts_verified",
                  len(sel_c2) == expected_n and len(sel_d) == expected_n and bool(d_in_chk.passed.all())
                  and bool(d_con_chk.canonical_context_matches.all()))
            d_out = root / f"eval_d_{expected_n}"
            score_d(sel_c2, sel_d, levels, floods, d_out, d_in_chk, d_con_chk, d_src_sum)
            d_eval_sum = json.loads((d_out / "Evaluation_D_summary.json").read_text(encoding="utf-8"))
            d_lp = pd.read_csv(d_out / "Evaluation_D_length_pairs.csv")
            d_ev = pd.read_csv(d_out / "Evaluation_D_events.csv")
            d_lv = pd.read_csv(d_out / "Evaluation_D_levels.csv")
            check(f"D_partition_{expected_n}_scored_with_flood_tables",
                  d_eval_sum["complete"] is True and d_eval_sum["contracts"] == expected_n
                  and len(d_lp) == expected_n * len(METRICS + SIGNATURES)
                  and set(sel_d[0].name for _ in (0,)).issubset(set(d_ev["run"]))
                  and set(sel_d[0].name for _ in (0,)).issubset(set(d_lv["run"])))
        # Kiểm từ chối khi sửa manifest, lệch C2_manifest_sha256 hoặc lệch hash D_handoff.csv.
        d_tamper = root / "d_fix_6"
        (d_tamper / "d_summary.json").write_text("{}", encoding="utf-8")
        rejects("D_manifest_tampering_rejected",
                lambda: read_d_compare(d_tamper, c2_fix_forecasts, c2_fix_dir, context, context_paths))
        seal_6 = build_d_fixture(d_tamper, False)
        bad_sum = json.loads((d_tamper / "d_summary.json").read_text(encoding="utf-8"))
        bad_sum["inputs"]["C2_manifest_sha256"] = "0" * 64
        (d_tamper / "d_summary.json").write_text(json.dumps(bad_sum), encoding="utf-8")
        seal_6()
        rejects("D_wrong_C2_manifest_rejected",
                lambda: read_d_compare(d_tamper, c2_fix_forecasts, c2_fix_dir, context, context_paths))
        seal_6 = build_d_fixture(d_tamper, False)
        bad_h = pd.read_csv(d_tamper / "D_handoff.csv")
        bad_h.loc[0, "sha256_best_model_pt"] = "0" * 64
        bad_h.to_csv(d_tamper / "D_handoff.csv", index=False)
        seal_6()
        rejects("D_handoff_hash_mismatch_rejected",
                lambda: read_d_compare(d_tamper, c2_fix_forecasts, c2_fix_dir, context, context_paths))
    return pd.DataFrame(checks)

# %% [markdown]
# <a id="evaluation-part-10"></a>
#
# ## Phần 10 — Chấm thực nghiệm và xuất kết quả

# %% [markdown]
# ### 10.1. Phạm vi riêng và giao chung
#
# Giữ đủ ID và trạng thái. Bảng chính common_source_all dùng cùng ngày cho mọi run.

# %% Phần 10.1 - Chấm theo lưu vực
# native không cấp persistence; paired_source và common_source_all dùng sáu chỉ số cùng mask.
def evaluate_group(forecasts, levels=None, floods=None, *, shared_mask=None):
    shared = common_mask(forecasts)
    if shared_mask is not None:
        shared_mask = np.asarray(shared_mask)
        if shared_mask.dtype != bool or shared_mask.shape != shared.shape or (shared_mask & ~shared).any():
            raise ValueError("Mặt nạ chấm phải nằm trong giao hợp lệ của hạn")
        shared = shared_mask
    metric_rows, level_rows, matrix_rows = [], [], []
    # Vai trò bảng độc lập với số dung sai: hạn 3–5 có cả hai bảng cùng ±1.
    event_rows = {role: [] for role in ("primary", "tolerance1")}
    match_rows = {role: [] for role in event_rows}
    event_cases = []
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
                            cases = [] if role == "primary" and grouped else None
                            score, pairs = score_events(obs, sim, threshold.threshold_mm_day, forecast.area[j], mask,
                                                        lead=forecast.lead, grouped=grouped, tolerance=tolerance,
                                                        forecast_horizon=horizon, case_rows=cases)
                            event_rows[role].append(dict(**event_meta, **score))
                            for case in cases or ():
                                event_cases.append(dict(**event_meta, matching_tolerance_days=tolerance,
                                    **{key: str(forecast.dates[value].date()) if key.endswith(("_onset", "_peak")) else value
                                       for key, value in case.items()}))
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
                event_cases=pd.DataFrame(event_cases, columns=[*match_columns, "status"]),
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
#
# Kiểm từng ID trên giao chung, không dùng bảng kết quả chép sẵn để thay phép tính.

# %% Phần 10.2 - Kiểm A3 và vẽ phân phối
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
#
# Ba ô thể hiện NSE, PNSE và KGE. Mỗi đường dùng toàn bộ điểm hữu hạn của run tương ứng, giữ cả đuôi âm; số `n` được tính riêng cho từng chỉ số.
#
# Các seed trong cùng họ dùng chung màu và khác kiểu nét. Chú giải nằm ngoài vùng vẽ; điểm CDF được xuất thành CSV để đối chiếu với bảng chỉ số.

# %% Phần 10.3 - Vẽ hình và lưu các điểm CDF
# Giữ toàn bộ điểm hữu hạn, kể cả đuôi âm; số n lấy riêng cho từng run/chỉ số.
def draw_figures(table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for group, part in table.loc[table.scope.eq("common_source_all")].groupby(["period", "scenario", "lead"], sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(16, max(5, .16*part.run.nunique()+.5)))
        styles = {"SeqLSTM_Q": ("#0072B2", "-"), "SeqLSTM_noQ": ("#56B4E9", "--"),
                  "EncDec_Q": ("#009E73", "-"), "EncDec_noQ": ("#E69F00", "--"),
                  "D_FC": ("#9467BD", "-"), "D_RA": ("#CC79A7", "-"),
                  "Persistence_qmax": ("#333333", "--"), "Persistence_qmean_aux": ("#888888", ":")}
        for seed, linestyle in zip(A4_SEEDS, ("--", "-.", ":")):
            styles[f"SeqLSTM_Q_A4_seed{seed}"] = ("#0072B2", linestyle)
            styles[f"SeqLSTM_Q_B0_seed{seed}"] = ("#8C564B", linestyle)
            styles[f"SeqLSTM_Q_B1a_seed{seed}"] = ("#009E73", linestyle)
            styles[f"SeqLSTM_Q_B1b_seed{seed}"] = ("#CC79A7", linestyle)
            styles[f"SeqLSTM_Q_B2_seed{seed}"] = ("#6F4E7C", linestyle)
            styles[f"SeqLSTM_Q_B3_P10S12_seed{seed}"] = ("#A50F15", linestyle)
            styles[f"SeqLSTM_Q_B3_P05S05_seed{seed}"] = ("#008B8B", linestyle)
            styles[f"DLinear_seed_{seed}"] = ("#D55E00", linestyle)
        c2_colors = ("#7E2F8E", "#1F77B4", "#E69F00", "#CC79A7", "#009E73",
                     "#17BECF", "#D55E00", "#A50F15")
        for conf, color in zip(C2_CONFIGS, c2_colors):
            for seed, linestyle in zip(C2_SEEDS, ("-", "--", ":")):
                styles[f"SF_{conf}_C2_seed{seed}"] = color, linestyle
        metrics_panels = tuple(zip(axes, ("NSE", "PNSE", "KGE")))
        for panel, (ax, metric) in enumerate(metrics_panels):
            for run, data in part.groupby("run", sort=False):
                finite = np.sort(data.loc[np.isfinite(data[metric]), metric].to_numpy())
                if len(finite):
                    y = np.arange(1, len(finite) + 1) / len(finite)
                    color, linestyle = styles.get(run, (None, "-"))
                    label = run.replace("SeqLSTM_Q_A4_seed", "SeqLSTM_Q A4 ").replace("SeqLSTM_Q_B0_seed", "SeqLSTM_Q B0 ").replace("SeqLSTM_Q_B1a_seed", "SeqLSTM_Q B1a ").replace("SeqLSTM_Q_B1b_seed", "SeqLSTM_Q B1b ").replace("SeqLSTM_Q_B2_seed", "SeqLSTM_Q B2 ")
                    if run.startswith("SeqLSTM_Q_B3_"):
                        label = run.replace("SeqLSTM_Q_B3_", "SeqLSTM_Q B3 ").replace("_seed", " ")
                    if run.startswith("SF_") and "_C2_seed" in run:
                        conf, seed = run.removeprefix("SF_").split("_C2_seed")
                        core, depth, level, lr = conf.split("_")
                        label = f"C2 {core.upper()} {depth} {level} {lr} {seed}"
                    ax.step(finite, y, where="post", color=color, linestyle=linestyle,
                            label=f"{label} (n={len(finite)})")
                    rows.extend(dict(period=group[0], scenario=group[1], lead=group[2], run=run, metric=metric,
                                     value=float(x), cumulative_fraction=float(p)) for x, p in zip(finite, y))
            ax.set_xscale("symlog", linthresh=1)
            ax.set(xlabel=metric, ylabel="Fraction of finite basin scores", ylim=(0, 1.02))
            ax.set_title(f"({chr(97+panel)}) {metric}")
            ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, .5),
                        ncol=2 if part.run.nunique() > 40 else 1)
        fig.suptitle(f"Common daily mask | {group[0]} | {group[1]} | lead {group[2]} | all finite scores retained")
        fig.tight_layout()
        fig.savefig(output / f"Evaluation_CDF_{group[0]}_{group[1]}_lead{group[2]}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Hình: phân phối NSE, PNSE, KGE theo lưu vực trên giao ngày chung; kỳ {group[0]}, "
              f"kịch bản {group[1]}, hạn {group[2]}, {part.run.nunique()} run. "
              "Đường càng về phải có điểm càng cao; n là số lưu vực có chỉ số hữu hạn; giữ toàn bộ điểm âm.")
    return pd.DataFrame(rows)

# %% [markdown]
# <a id="evaluation-part-11"></a>
#
# ## Phần 11 — Chạy toàn bộ và kết luận

# %% [markdown]
# ### 11.1. Kiểm đầu vào, chấm và lưu kết quả
#
# Chạy toàn bộ notebook từ đầu trong phiên sạch. Version 10 cần đủ 10 Notebook Output. Kết quả kịch bản C2 nằm trong `C2_scenarios/`.
#
# Kết luận cuối ghi trạng thái hoàn thành, số run, giao ngày chấm, kết quả kiểm A4/B0/B1/B2/B3 và phạm vi kịch bản/hạn đã chạy. `complete=true` chỉ được ghi khi toàn bộ bước kiểm và chấm đã thành công. Các bảng `scenario_changes` có thể rỗng khi chỉ có S0.
#
# Sau khi chạy thành công, lưu bằng Save & Run All trong chính notebook này. Kết quả test dùng báo cáo; lựa chọn checkpoint và ngưỡng B* dựa trên validation.

# %% Phần 11.1 - Điều phối và lưu kết quả trực tiếp
# Mọi dữ liệu giả chỉ ở self_test; trạng thái thực nghiệm chỉ đặt sau khi đọc/chấm thành công.
def run(input_root, output_root, a1_root=None, a2_root=None, a3_root=None, extra_contracts=(), *,
        a4_root=None, b0_root=None, b1_root=None, b2_root=None, b3_root=None, c2_root=None, h_root=None, d_root=None,
        expected_ids=451, require_c2_scenarios=False, require_h=False, require_d=False):
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "Evaluation_summary.json").write_text(
        json.dumps(dict(complete=False, status="incomplete", expected_A4_seeds=list(A4_SEEDS), expected_B0_seeds=list(B0_SEEDS), expected_B1_steps=list(B1_STEPS), expected_B2_seeds=list(B2_SEEDS), expected_B3_masks=list(B3_MASKS)), indent=2),
        encoding="utf-8")
    scenario_source = unique_file(input_root, "c2_scenarios_summary.json", required=False)
    if scenario_source is None and require_c2_scenarios:
        raise FileNotFoundError(f"Gắn toàn bộ Notebook Output {C2_SCENARIOS_NOTEBOOK} Version 1 đã hoàn thành.")
    if h_root is None:
        h_summary = unique_file(input_root, "h_summary.json", required=False)
        if h_summary:
            h_root = h_summary.parent
        elif any(Path(input_root).rglob("SF_*_H*_test_S0_lead*.npz")):
            h_root = Path(input_root)
        elif require_h:
            raise FileNotFoundError("Gắn toàn bộ Notebook Output 15_H_LamaHCEExt_Compare Version 1.")
    if d_root is None:
        d_summary = unique_file(input_root, "d_summary.json", required=False)
        if d_summary:
            d_root = d_summary.parent
        elif require_d:
            raise FileNotFoundError(f"Gắn toàn bộ Notebook Output {D_COMPARE_NOTEBOOK} đã hoàn thành.")
    checks = self_test()
    checks.to_csv(output / "Evaluation_checks.csv", index=False, encoding="utf-8-sig")
    a3_root = Path(a3_root) if a3_root else unique_file(input_root, "A3_persistence_predictions.npz").parent
    a2_root = Path(a2_root) if a2_root else find_a2_root(input_root)
    a1_root = Path(a1_root) if a1_root else unique_file(input_root, "A1_P10_qmax_level_thresholds.csv").parent
    a4_root = unique_file(a4_root or input_root, "a4_summary.json").parent
    b0_root = unique_file(b0_root or input_root, "b0_summary.json").parent
    b1_root = unique_file(b1_root or input_root, "b1_summary.json").parent
    b2_root = unique_file(b2_root or input_root, "b2_summary.json").parent
    b3_root = unique_file(b3_root or input_root, "b3_summary.json").parent
    context, paths = read_context(a3_root)
    context_paths = list(paths)
    if len(context["ids"]) != expected_ids:
        raise ValueError(f"Danh sách có {len(context['ids'])} ID, kỳ vọng {expected_ids}")
    print("Phần 1–2: kiểm thử đạt; nguồn A1/A2/A3/A4/B0/B1/B2/B3 Compare:", a1_root, a2_root, a3_root, a4_root, b0_root, b1_root, b2_root, b3_root, sep="\n")
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
    b0_forecasts, read_paths, b0_input_check, b0_contract_check = read_b0(b0_root, context, context_paths)
    forecasts.extend(b0_forecasts)
    paths.extend(read_paths)
    show_table("Phần 8.4 - Manifest B0 và nguồn A3", b0_input_check[["kind", "file", "passed"]])
    show_table("Phần 8.4 - Ba hợp đồng LSTM B0", b0_contract_check[[
        "run", "seed", "selected_epoch", "training_notebook_version", "IDs", "target_days",
        "finite_prediction_day_IDs", "canonical_context_matches"]])
    b1_forecasts, b1_checks = {}, {}
    for step in B1_STEPS:
        members, read_paths, input_check, contract_check = read_b1(b1_root, context, context_paths, step)
        check_b1_b0_reference(b0_forecasts, members)
        b1_forecasts[step], b1_checks[step] = members, (input_check, contract_check)
        forecasts.extend(members)
        paths.extend(read_paths)
        show_table(f"Phần 8.4 - Manifest {step} và nguồn A3", input_check[["kind", "file", "passed"]])
        show_table(f"Phần 8.4 - Ba hợp đồng LSTM {step}", contract_check[[
            "run", "seed", "selected_epoch", "training_notebook_version", "IDs", "target_days",
            "finite_prediction_day_IDs", "canonical_context_matches"]])
    b2_forecasts, read_paths, b2_input_check, b2_contract_check = read_b2(
        b2_root, context, context_paths, b0_forecasts, b1_root)
    forecasts.extend(b2_forecasts)
    paths.extend(read_paths)
    show_table("Phần 8.4 - Manifest B2 và nguồn A3/B1", b2_input_check[["kind", "file", "passed"]])
    show_table("Phần 8.4 - Ba hợp đồng LSTM B2", b2_contract_check[[
        "run", "seed", "selected_epoch", "training_notebook_version", "IDs", "target_days",
        "finite_prediction_day_IDs", "paired_obs_prediction_day_IDs", "NSE_audit_abs_difference", "canonical_context_matches"]])
    b3_forecasts, read_paths, b3_input_check, b3_contract_check = read_b3(
        b3_root, context, context_paths, b0_forecasts, b2_root)
    forecasts.extend(b3_forecasts)
    paths.extend(read_paths)
    show_table("Phần 8.4 - Manifest B3 và nguồn A3/B2", b3_input_check[["kind", "file", "passed"]])
    show_table("Phần 8.4 - Sáu hợp đồng LSTM B3", b3_contract_check[[
        "run", "mask_level", "seed", "selected_epoch", "training_notebook_version", "IDs", "target_days",
        "finite_prediction_day_IDs", "paired_obs_prediction_day_IDs", "NSE_audit_abs_difference", "canonical_context_matches"]])
    if c2_root is None:
        summary_path = unique_file(input_root, "c2_summary.json", required=False)
        c2_root = summary_path.parent if summary_path else None
    c2_paths, c2_forecasts, h_members, d_members, d_source_summary, c2_scenarios_count = [], [], [], [], None, 0
    if c2_root is not None:
        c2_forecasts, c2_paths, c2_sources, c2_check = read_c2_compare(
            c2_root, context, context_paths, b3_root, b0_forecasts[0].metadata["bstar_threshold_val_nse_sd"])
        forecasts.extend(c2_forecasts)
        paths.extend(c2_paths)
        show_table("Phần 8.3.2 - 24 hợp đồng C2 Compare Version 1", c2_check)
    verified_c2 = {p.resolve() for p in c2_paths if p.suffix == ".npz"}
    if scenario_source:
        scenarios_c2, checks_c2 = read_c2_scenarios(scenario_source.parent, c2_forecasts, c2_root, context_paths)
        scenario_manifest = scenario_source.parent / "manifest.csv"
        paths.extend([scenario_manifest, *map(Path, checks_c2["path"])])
        c2_scenarios_count = len(scenarios_c2) - len(c2_forecasts)
        by_scenario = {s: [f for f in scenarios_c2 if f.scenario == s] for s in C2_SCENARIOS}
        scenario_output = output / "C2_scenarios"
        prefix = "Evaluation_C2_scenarios"
        scenario_mask = score_partitions(by_scenario, levels, floods, scenario_output, prefix, "scenario")
        checks_c2.to_csv(scenario_output / f"{prefix}_input_check.csv", index=False, encoding="utf-8-sig")
        scenario_changes(scenarios_c2, shared_mask=scenario_mask).to_csv(
            scenario_output / f"{prefix}_changes.csv", index=False, encoding="utf-8-sig")
    if h_root is not None:
        h_members, h_checks = read_h_contracts(h_root, context)
        h_output = output / "H"
        score_h(h_members, levels, floods, h_output)
        h_checks.to_csv(h_output / "Evaluation_H_input_check.csv", index=False, encoding="utf-8-sig")
    if d_root is not None:
        d_c2_refs, d_members, d_checks, d_contract_check, d_source_summary = read_d_compare(
            d_root, c2_forecasts, c2_root, context, context_paths)
        d_output = output / "D"
        score_d(d_c2_refs, d_members, levels, floods, d_output, d_checks, d_contract_check, d_source_summary)
        show_table("Phần 8.3.2 - Hợp đồng D Compare (730 bước)", d_contract_check)
    for path in extra_contracts:
        if Path(path).resolve() in verified_c2:
            continue
        forecast = read_contract(path)
        if (forecast.metadata.get("step") in ("C2", "D")
                or H_NAME.fullmatch(Path(path).name) or D_NAME.fullmatch(Path(path).name)):
            raise ValueError("Gắn nguyên Output Compare C2/H/D để kiểm manifest và phạm vi trước khi chấm")
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
    results = {key: [] for key in ("metrics", "levels", "confusion", "events", "event_matches", "event_cases",
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
    results["B0_input_check"] = b0_input_check
    results["B0_contract_check"] = b0_contract_check
    results["B2_input_check"] = b2_input_check
    results["B2_contract_check"] = b2_contract_check
    results["B3_input_check"] = b3_input_check
    results["B3_contract_check"] = b3_contract_check
    for step, (input_check, contract_check) in b1_checks.items():
        results[f"{step}_input_check"] = input_check
        results[f"{step}_contract_check"] = contract_check
    if c2_forecasts:
        results["C2_input_check"] = c2_sources
        results["C2_contract_check"] = c2_check
    reference_table = results["metrics"]
    if a4_forecasts or b0_forecasts or b1_forecasts or b2_forecasts or b3_forecasts or extra_contracts:
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
                    B1_compare_version_expected=B1_COMPARE_VERSION,
                    B1_input_verification="manifest_SHA256_same_A3_context_and_B0_validation_seed_std",
                    B3_compare_version_expected=B3_COMPARE_VERSION,
                    B3_input_verification="manifest_bytes_SHA256_same_A3_B2_Compare_B0_threshold_masks_and_obs_sim_NSE_audit",
                    B2_compare_version_expected=B2_COMPARE_VERSION,
                    B2_input_verification="manifest_bytes_SHA256_same_A3_B1_Compare_B0_threshold_and_contract_audit",
                    B0_compare_version_expected=B0_COMPARE_VERSION,
                    B0_bstar_threshold_val_nse_sd=b0_forecasts[0].metadata["bstar_threshold_val_nse_sd"],
                    B0_input_verification="manifest_SHA256_same_A3_context_and_validation_seed_std",
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
                   B3_contracts_matched=bool(b3_contract_check.canonical_context_matches.all()),
                   B3_runs=[f.name for f in b3_forecasts],
                   B3_runs_by_mask={level:[f.name for f in b3_forecasts if f.metadata["mask_level"]==level] for level in B3_MASKS},
                   B3_compare_version_expected=B3_COMPARE_VERSION,
                   B2_contracts_matched=bool(b2_contract_check.canonical_context_matches.all()),
                   B2_runs=[f.name for f in b2_forecasts],
                   B2_compare_version_expected=B2_COMPARE_VERSION,
                   B1_contracts_matched=all(bool(c.canonical_context_matches.all()) for _, c in b1_checks.values()),
                   B1a_runs=[f.name for f in b1_forecasts["B1a"]],
                   B1b_runs=[f.name for f in b1_forecasts["B1b"]],
                   B1_compare_version_expected=B1_COMPARE_VERSION,
                   B0_contracts_matched=bool(b0_contract_check.canonical_context_matches.all()),
                   B0_runs=[f.name for f in b0_forecasts],
                   B0_compare_version_expected=B0_COMPARE_VERSION,
                   B0_bstar_threshold_val_nse_sd=b0_forecasts[0].metadata["bstar_threshold_val_nse_sd"],
                   A4_compare_version_expected=A4_COMPARE_VERSION,
                   C2_compare_version_expected=1 if c2_forecasts else None,
                   C2_runs=[f.name for f in c2_forecasts],
                   C2_contracts_matched=True if c2_forecasts else None,
                   C2_new_scenario_contracts=c2_scenarios_count,
                   C2_scenarios_version_expected=1 if c2_scenarios_count else None,
                   C2_scenarios_manifest_sha256=sha256(scenario_manifest) if c2_scenarios_count else None,
                   H_evaluated=bool(h_members), H_contracts=len(h_members),
                   H_modes=list(H_MODES) if h_members else [],
                   D_evaluated=bool(d_members), D_contracts=len(d_members),
                   D_configs=list(d_source_summary["configs"]) if d_source_summary else [],
                   D_S4D_included=bool(d_source_summary["S4D_included"]) if d_source_summary else None,
                   D_all_pairs_strictly_equivalent=(bool(d_source_summary.get("all_pairs_strictly_equivalent", True))
                                                    if d_source_summary else None),
                   D_grad_guard_runs_with_skips=(list(d_source_summary.get("grad_guard_runs_with_skips") or [])
                                                 if d_source_summary else []),
                   coverage=coverages, scenarios=sorted({f.scenario for f in forecasts}),
                   leads=actual_leads,
                   C2_scenarios=list(C2_SCENARIOS) if c2_scenarios_count else [],
                   H_leads=list(H_LEADS) if h_members else [],
                   operational_scenarios_run=bool(c2_scenarios_count) or any(f.scenario != "S0" for f in forecasts),
                   multi_lead_run=bool(h_members) or any(lead > 1 for lead in actual_leads),
                   python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                   scipy=__import__("scipy").__version__, output=str(output))
    (output / "Evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Phần 11: hoàn thành; lưu trực tiếp cho Kaggle Save Version.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if c2_forecasts:
        print("GỬI LẠI: Evaluation_summary.json, Evaluation_protocol.json, Evaluation_run_index.json, "
              "Evaluation_coverage.csv, Evaluation_summary.csv, Evaluation_metrics.csv, Evaluation_paired_tests.csv, "
              "Evaluation_C2_input_check.csv, Evaluation_C2_contract_check.csv, Evaluation_events_pooled.csv, "
              "Evaluation_event_cases.csv, "
              "Evaluation_CDF_test_S0_lead1.png.")
    if c2_scenarios_count:
        print("GỬI THÊM TỪ C2_scenarios/: Evaluation_C2_scenarios_summary.json, "
              "Evaluation_C2_scenarios_input_check.csv, Evaluation_C2_scenarios_coverage.csv, "
              "Evaluation_C2_scenarios_changes.csv, cùng bảng levels_pooled, events_pooled và event_cases "
              "của bảy kịch bản.")
    if h_members:
        print("GỬI THÊM TỪ H/: Evaluation_H_summary.json, Evaluation_H_coverage.csv, "
              "Evaluation_H_input_check.csv và các bảng theo hạn.")
    if d_members:
        print("GỬI THÊM TỪ D/: Evaluation_D_summary.json, Evaluation_D_coverage.csv, "
              "Evaluation_D_input_check.csv, Evaluation_D_contract_check.csv, Evaluation_D_length_pairs.csv, "
              "Evaluation_D_summary.csv, Evaluation_D_levels_pooled.csv, Evaluation_D_events_pooled.csv, "
              "Evaluation_D_levels.csv, Evaluation_D_events.csv, Evaluation_D_event_cases.csv.")
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
    parser.add_argument("--b0")
    parser.add_argument("--b1")
    parser.add_argument("--b2")
    parser.add_argument("--b3")
    parser.add_argument("--c2")
    parser.add_argument("--h")
    parser.add_argument("--d")
    parser.add_argument("--require-c2-scenarios", action="store_true")
    parser.add_argument("--require-h", action="store_true")
    parser.add_argument("--require-d", action="store_true")
    parser.add_argument("--extra-contract", action="append", default=[])
    args = parser.parse_args([] if "ipykernel" in sys.modules else None)
    if args.self_test:
        print(self_test().to_string(index=False))
    else:
        if "ipykernel" in sys.modules and not list(Path(args.input).rglob("c2_summary.json")):
            raise FileNotFoundError("Gắn Output 13_C2_LamaHCEExt_Compare Version 1.")
        run(args.input, args.output, args.a1, args.a2, args.a3, args.extra_contract, a4_root=args.a4,
            b0_root=args.b0, b1_root=args.b1, b2_root=args.b2, b3_root=args.b3, c2_root=args.c2, h_root=args.h,
            d_root=args.d, require_c2_scenarios=args.require_c2_scenarios or "ipykernel" in sys.modules,
            require_h=args.require_h, require_d=args.require_d)


if __name__ == "__main__":
    main()
