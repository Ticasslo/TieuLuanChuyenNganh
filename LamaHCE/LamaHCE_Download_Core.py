# =============================================================================
# LamaH-CE — Notebook 1: tải phần lõi bài cơ sở cần
# Bài cơ sở: Kirschstein & Sun (ICML 2024), "The Merit of River Network Topology for Neural Flood Forecasting"
#            — LamaH-CE theo giờ, 358 trạm; mã github.com/nkirschi/neural-flood-forecasting
# Notebook Kaggle: Accelerator = None, bật Internet, dán Phần 1 vào cell 1, Phần 2 vào cell 2,
# rồi Save Version → Save & Run All (chạy nền). Xong: tab Output → New Dataset (Private) "lamah-ce-core".
# Phần 1 giống hệt Phần 1 của LamaHCE_Download_Extra.py (mỗi notebook phải tự đủ code).
#
# MỤC LỤC
#   Phần 1 — Thư viện, hằng số, hàm tải theo luồng dùng chung
#   Phần 2 — Tải phần lõi bài cơ sở cần (B + D theo giờ) → lamah_ce_core.zip
# =============================================================================


# %% Phần 1 — Thư viện, hằng số, hàm tải theo luồng dùng chung
# Đọc file 14,8 GB từ Zenodo theo luồng (không lưu file nén gốc), giữ file theo điều kiện truyền vào và
# ghi thẳng vào MỘT file ZIP trong /kaggle/working (tránh giới hạn ~500 file của output notebook Kaggle;
# ZIP giữ nguyên tên file gốc .csv nên đọc bằng zipfile/pandas hoặc giải nén là dùng được code của Kirschstein & Sun).
# Vượt ngưỡng dung lượng thì bỏ qua phần còn lại thay vì dừng. Nếu lỗi mạng giữa chừng: ZIP không có dòng
# "Kiểm tra ZIP" ở cuối log là chưa đủ — chạy lại cả notebook (không tải tiếp được từ chỗ đứt).
# Log mỗi 60 giây: % đã tải, tốc độ, thời gian còn lại dự kiến (ETA), thư mục đang duyệt.
# % tính theo số byte nén đã đọc nên vẫn tăng đều kể cả khi đang duyệt qua các file bị bỏ qua.
import io
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

LAMAH_URL = "https://zenodo.org/records/5153305/files/1_LamaH-CE_daily_hourly.tar.gz?download=1"
LAMAH_COMPRESSED_GB = 14.8             # dự phòng khi Zenodo không trả Content-Length
DAILY_KEY = "/2_timeseries/daily/"
CORE_DIRS = ("B_basins_intermediate_all/", "D_gauges/")   # 2 thư mục bài cơ sở đọc (gói lõi)
ZIP_LEVEL = 6
MAX_OUTPUT_GB = 18.0                   # /kaggle/working chỉ lưu tối đa 20 GB
LOG_EVERY_SECONDS = 60
READ_BUFFER_BYTES = 8 * 1024 * 1024
NETWORK_TIMEOUT_S = 300             # mạng treo quá 5 phút thì báo lỗi thay vì đứng im tới hết 12 giờ

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


def progress_line(read_bytes, total_bytes, elapsed_s, seen, kept, raw_bytes, saved_bytes, name):
    """Một dòng tiến độ: % đã tải, tốc độ trung bình, thời gian còn lại dự kiến, thư mục đang duyệt."""
    frac = read_bytes / total_bytes if total_bytes else 0.0
    speed_mb_s = read_bytes / 1e6 / elapsed_s if elapsed_s > 0 else 0.0
    eta_min = elapsed_s * (1 - frac) / frac / 60 if frac > 0 else float("nan")
    folder = "/".join(name.split("/")[:3])
    return (f"[{elapsed_s / 60:6.1f} phút | {frac:6.1%} | {speed_mb_s:5.1f} MB/s | còn ~{eta_min:5.0f} phút] "
            f"đọc {read_bytes / 1e9:5.2f}/{total_bytes / 1e9:.2f} GB nén | duyệt {seen} file, giữ {kept} "
            f"({raw_bytes / 1e9:.2f} GB gốc → {saved_bytes / 1e9:.2f} GB lưu) | đang ở: {folder}")


def stream_lamah(zip_path: Path, keep, max_output_gb: float = MAX_OUTPUT_GB) -> None:
    """Tải LamaH-CE theo luồng, ghi các file có keep(name) == True vào zip_path (giữ nguyên đường dẫn gốc)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Nền tảng: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Local'} | Lưu vào: {zip_path}")
    print(f"Ổ đĩa trống: {shutil.disk_usage(zip_path.parent).free / 1e9:.1f} GB")

    response = urllib.request.urlopen(LAMAH_URL, timeout=NETWORK_TIMEOUT_S)
    total_bytes = int(response.headers.get("Content-Length") or LAMAH_COMPRESSED_GB * 1e9)
    print(f"Kích thước file nén trên Zenodo: {total_bytes / 1e9:.2f} GB")
    counter = CountingReader(response)
    stream = io.BufferedReader(counter, buffer_size=READ_BUFFER_BYTES)

    t0 = last_log = time.time()
    seen = kept = raw_bytes = 0
    manifest, groups, skipped_over_cap = ["name;raw_bytes"], {}, []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=ZIP_LEVEL,
                         allowZip64=True) as zf, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            seen += 1
            if keep(member.name):
                if zip_path.stat().st_size / 1e9 > max_output_gb:
                    skipped_over_cap.append(member.name)
                else:
                    member_file = archive.extractfile(member)
                    zf.writestr(member.name, member_file.read() if member_file else b"")
                    kept += 1
                    raw_bytes += member.size
                    manifest.append(f"{member.name};{member.size}")
                    parts = member.name.split("/")
                    key = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:-1]) or parts[0]
                    groups[key] = groups.get(key, 0) + 1

            if time.time() - last_log > LOG_EVERY_SECONDS:
                last_log = time.time()
                print(progress_line(counter.bytes_read, total_bytes, last_log - t0, seen, kept,
                                    raw_bytes, zip_path.stat().st_size, member.name), flush=True)
        zf.writestr("manifest.csv", "\n".join(manifest))

    print(f"\nXONG sau {(time.time() - t0) / 60:.1f} phút | đọc {counter.bytes_read / 1e9:.2f} GB nén | "
          f"duyệt {seen} file | giữ {kept} file | gốc {raw_bytes / 1e9:.2f} GB → ZIP {zip_path.stat().st_size / 1e9:.2f} GB")
    for key in sorted(groups):
        print(f"  {key}: {groups[key]} file")
    if skipped_over_cap:
        print(f"\nCẢNH BÁO: bỏ qua {len(skipped_over_cap)} file vì vượt {max_output_gb} GB, ví dụ: "
              f"{skipped_over_cap[:5]}")
    with zipfile.ZipFile(zip_path) as zf:   # kiểm tra ZIP đọc lại được, không hỏng
        print(f"Kiểm tra ZIP: {len(zf.namelist())} mục, lỗi CRC: {zf.testzip()}")


# %% Phần 2 — Tải phần lõi bài cơ sở cần (B + D theo giờ) → Kaggle Dataset "lamah-ce-core"
# Giữ đúng 3 thứ code của Kirschstein & Sun đọc (mạng sông Stream_dist, khí tượng giờ của B, lưu lượng giờ của D),
# đủ mọi năm và mọi cột, cộng thuộc tính của B, D và thư mục Info (ghi nguồn theo CC BY-SA 4.0).


def keep_core(name: str) -> bool:
    if name.lower().startswith("info"):
        return True
    return name.startswith(CORE_DIRS) and DAILY_KEY not in name


stream_lamah(BASE_DIR / "lamah_ce_core.zip", keep_core)
