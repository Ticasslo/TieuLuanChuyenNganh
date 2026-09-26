# =============================================================================
# Extended LamaH-CE (BiasCast, HESS 2026) — tải dữ liệu + kết quả tác giả, kiểm tra chất lượng, persistence
# Bài: Konold và cs., "BiasCast: learning and adjusting real time biases from meteorological forecasts
#      to enhance runoff predictions", HESS 30:5067–5096 (2026). Dữ liệu Zenodo 17119635 (CC BY-NC 4.0),
#      kết quả thí nghiệm Zenodo 17292895.
# Notebook Kaggle: Accelerator = None, bật Internet, dán Phần 1–3 vào 3 cell (hoặc cả file vào 1 cell),
# rồi Run All. Xong: tab Output → New Dataset (Private) "lamah-ce-ext" (giữ cả 2 file ZIP + CSV + PNG).
# Chạy được cả trên Colab (lưu vào /content).
#
# MỤC LỤC
#   Phần 1 — Thư viện, hằng số, hàm tải theo luồng dùng chung
#   Phần 2 — Tải Extended LamaH-CE (0,95 GB) và kết quả thí nghiệm của tác giả (0,23 GB) → 2 file ZIP
#   Phần 3 — Chất lượng dữ liệu 451 lưu vực + baseline persistence so với mô hình của tác giả
# =============================================================================


# %% Phần 1 — Thư viện, hằng số, hàm tải theo luồng dùng chung
# Đọc file .tar.gz từ Zenodo theo luồng (không lưu file nén gốc) và ghi thẳng mọi file vào MỘT file ZIP
# (tránh giới hạn số file của output Kaggle; pandas đọc thẳng CSV trong ZIP được).
# Log mỗi 60 giây. Thiếu dòng "Kiểm tra ZIP" ở cuối log = tải lỗi, chạy lại cell.
import io
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {  # tên ZIP lưu ra: (URL Zenodo, dung lượng nén dự phòng khi Zenodo không trả Content-Length, GB)
    "lamah_ce_ext.zip": ("https://zenodo.org/records/17119635/files/Extended_LamaH-CE_daily.tar.gz?download=1", 0.95),
    "biascast_experiments.zip": ("https://zenodo.org/records/17292895/files/Experiments.tar.gz?download=1", 0.23),
}
ZIP_LEVEL = 6
LOG_EVERY_SECONDS = 60
READ_BUFFER_BYTES = 8 * 1024 * 1024
NETWORK_TIMEOUT_S = 300             # mạng treo quá 5 phút thì báo lỗi thay vì đứng im

IS_KAGGLE = Path("/kaggle/working").exists()
IS_COLAB = "google.colab" in sys.modules
BASE_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("/content") if IS_COLAB else Path.cwd()


class CountingReader(io.RawIOBase):
    """Bọc luồng HTTP để đếm số byte nén đã đọc (theo dõi tiến độ)."""

    def __init__(self, raw):
        self.raw, self.bytes_read = raw, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        chunk = self.raw.read(len(buffer))
        n = len(chunk)
        buffer[:n] = chunk
        self.bytes_read += n
        return n


def stream_tar_to_zip(url: str, fallback_gb: float, zip_path: Path) -> None:
    """Tải .tar.gz theo luồng, ghi mọi file vào zip_path (giữ nguyên đường dẫn gốc), in tiến độ và kiểm tra CRC."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {zip_path.name} ===")
    print(f"Nền tảng: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Local'} | "
          f"ổ đĩa trống: {shutil.disk_usage(zip_path.parent).free / 1e9:.1f} GB")
    response = urllib.request.urlopen(url, timeout=NETWORK_TIMEOUT_S)
    total_bytes = int(response.headers.get("Content-Length") or fallback_gb * 1e9)
    counter = CountingReader(response)
    stream = io.BufferedReader(counter, buffer_size=READ_BUFFER_BYTES)

    t0 = last_log = time.time()
    kept = raw_bytes = 0
    groups = {}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=ZIP_LEVEL,
                         allowZip64=True) as zf, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_file = archive.extractfile(member)
            zf.writestr(member.name, member_file.read() if member_file else b"")
            kept += 1
            raw_bytes += member.size
            key = "/".join(member.name.split("/")[:4][:-1]) or member.name
            groups[key] = groups.get(key, 0) + 1
            if time.time() - last_log > LOG_EVERY_SECONDS:
                last_log = time.time()
                print(f"[{(last_log - t0) / 60:5.1f} phút] {counter.bytes_read / total_bytes:6.1%} "
                      f"| {kept} file | đang ở: {'/'.join(member.name.split('/')[:3])}", flush=True)

    print(f"XONG sau {(time.time() - t0) / 60:.1f} phút | đọc {counter.bytes_read / 1e9:.2f} GB nén | "
          f"{kept} file | gốc {raw_bytes / 1e9:.2f} GB → ZIP {zip_path.stat().st_size / 1e9:.2f} GB")
    for key in sorted(groups):
        print(f"  {key}: {groups[key]} file")
    with zipfile.ZipFile(zip_path) as zf:   # kiểm tra ZIP đọc lại được, không hỏng
        print(f"Kiểm tra ZIP: {len(zf.namelist())} mục, lỗi CRC: {zf.testzip()}")


# %% Phần 2 — Tải Extended LamaH-CE (0,95 GB) và kết quả thí nghiệm của tác giả (0,23 GB) → 2 file ZIP
# lamah_ce_ext.zip: toàn bộ dữ liệu ngày (A_basins_total_upstrm, D_gauges có qmin/qmean/qmax).
# biascast_experiments.zip: 24 cấu hình thí nghiệm (config.yml, best_model.pt, scaler, test_metrics.csv).
for zip_name, (url, fallback_gb) in SOURCES.items():
    stream_tar_to_zip(url, fallback_gb, BASE_DIR / zip_name)


# %% Phần 3 — Chất lượng dữ liệu 451 lưu vực + baseline persistence so với mô hình của tác giả
# (a) % thiếu lưu lượng (nhãn qmax, đầu vào qmean) theo kỳ train / validation / test của bài;
# (b) % thiếu từng biến khí tượng 2003–2017; (c) persistence trên test 2014–2017 — ý phản biện 1 đề nghị,
# bài không làm: qmax(t) ≈ qmax(t−1) và qmax(t) ≈ qmean(t−1) (thông tin mô hình có Q thực sự nhận).
# NSE theo lưu vực không đổi khi đổi đơn vị, nên so trực tiếp được với NSE (mm/ngày) trong test_metrics.csv.
# Chạy lại riêng phần này (không tải lại): gắn Kaggle Dataset "lamah-ce-ext" bằng Add Input, chạy Phần 1 rồi Phần 3.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_zip(name: str) -> Path:
    """Tìm ZIP trong thư mục làm việc (vừa tải ở Phần 2) hoặc trong dataset đã gắn (/kaggle/input, Drive)."""
    for root in [BASE_DIR, Path("/kaggle/input"), Path("/content/drive/MyDrive")]:
        hits = [root / name] if (root / name).exists() else sorted(root.rglob(name)) if root.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Không thấy {name}: chạy Phần 2 hoặc gắn dataset lamah-ce-ext")


EXT_ZIP = find_zip("lamah_ce_ext.zip")
EXP_ZIP = find_zip("biascast_experiments.zip")
MISSING_VALUE = -999
PERIODS = {"train": ("2003-01-01", "2009-12-31"),
           "validation": ("2010-01-01", "2013-12-31"),
           "test": ("2014-01-01", "2017-12-31")}
MODEL_RUNS = {  # tên hiển thị: đoạn đường dẫn duy nhất của thí nghiệm trong biascast_experiments.zip
    "Sequential LSTM có Q (Simple)": "Sequential_Forecast_LSTM_RA_with_q_Simple_Embedding/",
    "Sequential LSTM có Q (Complex)": "Sequential_Forecast_LSTM_RA_with_q_Complex_Embedding/",
    "Sequential LSTM không Q": "Sequential_Forecast_LSTM_RA_without_q_Simple_Embedding/",
    "Encoder–Decoder có Q (Simple)": "Encoder_Decoder_LSTM_with_q_Simple_Embedding/",
    "Baseline chỉ dự báo": "Baseline/Forecast/",
}
PERSISTENCE = {"Persistence qmax(t−1)": "qmax", "Persistence qmean(t−1)": "qmean"}


def find_member(names, *parts):
    """Tìm đúng 1 file trong ZIP chứa mọi đoạn trong parts; báo lỗi rõ nếu không có hoặc có nhiều."""
    hits = [n for n in names if all(p in n for p in parts)]
    if len(hits) != 1:
        raise ValueError(f"Tìm {parts}: {len(hits)} kết quả {hits[:5]}")
    return hits[0]


def nse(obs: pd.Series, sim: pd.Series) -> float:
    mask = obs.notna() & sim.notna()
    o, s = obs[mask], sim[mask]
    if len(o) < 30 or o.var() == 0:
        return np.nan
    return 1 - ((o - s) ** 2).sum() / ((o - o.mean()) ** 2).sum()


def with_date_index(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df[["YYYY", "MM", "DD"]].rename(columns={"YYYY": "year", "MM": "month", "DD": "day"}))
    return df.drop(columns=["YYYY", "MM", "DD"])


# --- NSE theo lưu vực của các mô hình tác giả (tập test 2014–2017) ---
with zipfile.ZipFile(EXP_ZIP) as zf:
    exp_names = zf.namelist()
    model_nse = pd.DataFrame({
        label: pd.read_csv(zf.open(find_member(exp_names, part, "test_metrics.csv")),
                           dtype={"basin": str}).set_index("basin")["NSE"]
        for label, part in MODEL_RUNS.items()})
basins = sorted(model_nse.index)
print(f"Số lưu vực trong kết quả của tác giả: {len(basins)}")

# --- Lưu lượng trạm (D_gauges) và khí tượng lưu vực (A_basins_total_upstrm) của 451 lưu vực ---
GAUGE_DIR = "D_gauges/2_timeseries/daily/"
MET_DIR = "A_basins_total_upstrm/2_timeseries/daily/"
rows, forcing_missing, forcing_total, ecmwf_first, met_first = [], pd.Series(dtype=float), 0, [], []
forcing_ok = True   # tắt phần (b) nếu cấu trúc tệp khí tượng khác dự kiến, không làm hỏng (a) và (c)
with zipfile.ZipFile(EXT_ZIP) as zf:
    ext_names = zf.namelist()
    try:
        sample = find_member(ext_names, MET_DIR, f"/ID_{basins[0]}.csv")
        print("Cột tệp khí tượng mẫu:", list(pd.read_csv(zf.open(sample), sep=";", nrows=2).columns))
    except (ValueError, KeyError) as err:
        forcing_ok = False
        print("BỎ QUA phần (b):", err, "| ví dụ tên file A:", [n for n in ext_names if "A_basins" in n][:5])
    for bid in basins:
        raw = pd.read_csv(zf.open(find_member(ext_names, GAUGE_DIR, f"/ID_{bid}.csv")), sep=";")
        q = with_date_index(raw)[["qmin", "qmean", "qmax"]]
        n_missing_code = int((q == MISSING_VALUE).sum().sum())
        q = q.where(q >= 0)                                   # -999 và giá trị âm → thiếu
        row = {"basin": bid, "n_missing_code": n_missing_code,
               "first_valid": q["qmax"].first_valid_index(), "last_valid": q["qmax"].last_valid_index()}
        for period, (start, end) in PERIODS.items():
            # reindex đủ mọi ngày của kỳ: ngày không có dòng trong tệp (trạm bắt đầu đo muộn) cũng tính là thiếu
            part = q.reindex(pd.date_range(start, end, freq="D"))
            row[f"{period}_days"] = len(part)
            row[f"{period}_qmax_missing"] = int(part["qmax"].isna().sum())
            row[f"{period}_qmean_missing"] = int(part["qmean"].isna().sum())
        test_start, test_end = PERIODS["test"]
        obs = q["qmax"].loc[test_start:test_end]
        for label, col in PERSISTENCE.items():
            row[label] = nse(obs, q[col].shift(1).loc[test_start:test_end])
        rows.append(row)

        if not forcing_ok:
            continue
        met = with_date_index(pd.read_csv(zf.open(find_member(ext_names, MET_DIR, f"/ID_{bid}.csv")), sep=";"))
        met_first.append(met.index.min())   # hindcast 730 ngày cần khí tượng từ 2001 cho mẫu train đầu tiên
        met = met.loc[PERIODS["train"][0]:PERIODS["test"][1]]
        forcing_missing = forcing_missing.add((met.isna() | (met == MISSING_VALUE)).sum(), fill_value=0)
        forcing_total += len(met)
        ecmwf_cols = [c for c in met.columns if c.startswith("ECMWF")]
        if ecmwf_cols:
            ecmwf_first.append(met[ecmwf_cols[0]].where(met[ecmwf_cols[0]] != MISSING_VALUE).first_valid_index())

quality = pd.DataFrame(rows).set_index("basin")

# --- (a) Thiếu lưu lượng theo kỳ ---
print("\n(a) LƯU LƯỢNG THIẾU THEO KỲ (451 lưu vực)")
print(f"Ô mang mã thiếu {MISSING_VALUE} trong toàn bộ tệp D_gauges: {quality['n_missing_code'].sum()}")
for period in PERIODS:
    days = quality[f"{period}_days"]
    for col in ["qmax", "qmean"]:
        miss = quality[f"{period}_{col}_missing"]
        print(f"  {period:10s} {col:5s}: {days.iloc[0]} ngày/lưu vực | mẫu hợp lệ {int((days - miss).sum()):,} "
              f"/ {int(days.sum()):,} | thiếu {miss.sum() / days.sum():6.2%} | lưu vực có thiếu {(miss > 0).sum()} "
              f"| thiếu >10% {(miss > 0.1 * days).sum()} | thiếu toàn bộ {(miss == days).sum()}")
print(f"Ngày có qmax đầu tiên: sớm nhất {quality['first_valid'].min().date()}, muộn nhất {quality['first_valid'].max().date()}")
print(f"Ngày có qmax cuối cùng: sớm nhất {quality['last_valid'].min().date()}, muộn nhất {quality['last_valid'].max().date()}")
late = quality[quality["first_valid"] > pd.Timestamp(PERIODS["train"][0])]
print(f"Lưu vực bắt đầu có qmax sau {PERIODS['train'][0]}: {len(late)} — "
      + ", ".join(f"{b} ({d.date()})" for b, d in late["first_valid"].sort_values().items()))

# --- (b) Thiếu khí tượng 2003–2017 ---
if forcing_total:
    print(f"\n(b) KHÍ TƯỢNG THIẾU 2003–2017 (tỷ lệ trên {forcing_total:,} ngày-lưu vực), 15 cột thiếu nhiều nhất:")
    print((forcing_missing / forcing_total).sort_values(ascending=False).head(15).map("{:.2%}".format).to_string())
if ecmwf_first:
    firsts = pd.Series(ecmwf_first).dropna()
    print(f"Ngày có ECMWF đầu tiên: sớm nhất {firsts.min().date()}, muộn nhất {firsts.max().date()}")
if met_first:
    firsts = pd.Series(met_first)
    print(f"Ngày đầu tiên của tệp khí tượng: sớm nhất {firsts.min().date()}, muộn nhất {firsts.max().date()}")

# --- (c) Persistence so với mô hình ---
nse_all = model_nse.join(quality[list(PERSISTENCE)])
summary = nse_all.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T[["count", "10%", "25%", "50%", "75%", "90%"]]
print("\n(c) NSE TẬP TEST 2014–2017 THEO LƯU VỰC (50% = trung vị):")
print(summary.round(3).to_string())
for label in MODEL_RUNS:
    for p_label in PERSISTENCE:
        diff = (nse_all[label] - nse_all[p_label]).dropna()
        print(f"  {label} − {p_label}: trung vị ΔNSE {diff.median():+.3f} | mô hình hơn ở {(diff > 0).mean():.1%} "
              f"trong {len(diff)} lưu vực")

# --- Lưu kết quả + hình CDF giống cách bài trình bày ---
quality.join(model_nse).to_csv(BASE_DIR / "biascast_data_quality.csv")
fig, ax = plt.subplots(figsize=(8, 5))
for col in nse_all.columns:
    values = np.sort(nse_all[col].dropna().clip(lower=-1))
    ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=col,
            linestyle="--" if col in PERSISTENCE else "-")
ax.set_xlim(-1, 1)
ax.set_xlabel("NSE tập test 2014–2017 (cắt dưới tại −1)")
ax.set_ylabel("Tỷ lệ lưu vực (CDF)")
ax.set_title("Mô hình BiasCast so với persistence — 451 lưu vực")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(BASE_DIR / "biascast_persistence_cdf.png", dpi=150)
plt.show()
print(f"\nĐã lưu: biascast_data_quality.csv, biascast_persistence_cdf.png trong {BASE_DIR}")
