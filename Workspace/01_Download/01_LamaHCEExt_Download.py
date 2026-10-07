# =============================================================================
# Extended LamaH-CE (BiasCast, HESS 2026) — tải dữ liệu và kết quả thí nghiệm của tác giả
# Bài: Konold và cs., "BiasCast: learning and adjusting real time biases from meteorological forecasts
#      to enhance runoff predictions", HESS 30:5067–5096 (2026). Dữ liệu Zenodo 17119635 (CC BY-NC 4.0),
#      kết quả thí nghiệm Zenodo 17292895.
# Notebook Kaggle: Accelerator = None, bật Internet, dán Phần 1–2 vào 2 cell (hoặc cả file vào 1 cell),
# rồi Run All. Xong: tab Output → New Dataset (Private) "lamah-ce-ext" (giữ cả 2 file ZIP;
# Kaggle tự giải nén thành thư mục khi tạo dataset). Khám phá dữ liệu: Workspace/02_Exploration/.
# Chạy được cả trên Colab (lưu vào /content).
#
# MỤC LỤC
#   Phần 1 — Thư viện, hằng số, hàm tải theo luồng dùng chung
#   Phần 2 — Tải Extended LamaH-CE (0,95 GB) và kết quả thí nghiệm của tác giả (0,23 GB) → 2 file ZIP
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
