# =============================================================================
# Extended LamaH-CE (BiasCast, HESS 2026) - khảo sát dữ liệu (bước A1)
# Bài: Konold và cs., HESS 30:5067-5096 (2026). Dữ liệu Zenodo 17119635 (CC BY-NC 4.0),
#      kết quả thí nghiệm Zenodo 17292895 - đã lưu thành Kaggle Dataset "lamah-ce-ext"
#      (Workspace/01_Download/01_LamaHCEExt_Download/01_LamaHCEExt_Download.py).
# Notebook Kaggle "01_A1_LamaHCEExt_Explore": Accelerator = None (CPU), Internet bật khi 2.5 cần tải Q giờ, Persistence = Files only,
# Add Input => Your Datasets => lamah-ce-ext; 2.5 dùng core sẵn có hoặc tự tải riêng Q giờ khi thiếu.
# Mỗi khối # %% là một cell; chạy từ trên xuống trong cùng phiên.
# Chạy được cả trên Colab/máy khác: đặt biến môi trường LAMAH_ROOT trỏ tới thư mục chứa dữ liệu đã giải nén.
# Chữ trong hình viết tiếng Anh; ghi chú, chú thích kết quả viết ở Document/03_Data/01_LamaHCE.md.
# Thuyết minh A1 kèm dẫn chứng và ví dụ: Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/01_A1_LamaHCEExt_Explore.md.
#
# MỤC LỤC
#   Phần 1 - Môi trường, đường dẫn, hằng số dùng chung
#   Phần 2 - Cấu trúc, cột, đơn vị, phủ ngày, mã thiếu và quan hệ ID giữa A và D
#   Phần 3 - Thuộc tính tĩnh, diện tích và bản đồ 859/451 lưu vực cùng trạm
#   Phần 4 - Thiếu theo kỳ, biến, năm, độ dài đoạn thiếu và trạm đo muộn
#   Phần 5 - Phân phối, mùa, đoạn hằng số, ngoại lai và đối chiếu qmax từ chuỗi giờ
#   Phần 6 - Ngữ nghĩa và lệch miền của 5 cặp ECMWF HRES / nguồn đối chiếu theo năm
#   Phần 7 - Cửa sổ CSV: có/không Q, 365/730 ngày, toàn khung/ô dùng theo pha
#   Phần 8 - So phân phối giữa các kỳ, ngày rất cao và NSE theo nguồn tác giả/A2 có thể kiểm
#   Phần 9 - Hồ sơ các lưu vực 55, 742, 758, 800
#   Phần 10 - Phân mức, Gumbel, ngày/đoạn vượt và cụm theo quy tắc USWRC
# Tổng hợp và bàn giao chỉ nằm trong file thuyết minh MD.
# =============================================================================

# %% [markdown]
# # A1 — Khảo sát Extended LamaH-CE
#
# Notebook thực thi khảo sát dữ liệu trên Kaggle CPU. Chạy các cell từ trên xuống trong cùng phiên; thêm Input `lamah-ce-ext`. Cell 2.5 dùng Q giờ sẵn có và tự tải khi thiếu, nên lần tải cần Internet. Có thể đặt `LAMAH_ROOT` khi chạy ở môi trường khác.
#
# Các phần 1–10 giữ phép tính, hình và bảng xuất vào `WORK_DIR/Figures`, `WORK_DIR/Tables`. Kết quả đã đo, diễn giải, giới hạn và tổng hợp bàn giao được trình bày trong [thuyết minh A1](01_A1_LamaHCEExt_Explore.md). Dữ liệu đầu vào và tập 451 ID được giữ nguyên; cảnh báo chất lượng và các kiểm tra nhất quán vẫn được thực thi.
#
# ## Mục lục
#
# - Phần 1. Môi trường và đường dẫn
# - Phần 2. Cấu trúc, tập ID và độ phủ dữ liệu
# - Phần 3. Thuộc tính tĩnh, diện tích và bản đồ
# - Phần 4. Dữ liệu thiếu theo kỳ, năm và đoạn liên tiếp
# - Phần 5. Phân phối Q, khí tượng và đối chiếu nguồn giờ
# - Phần 6. Năm cặp khí tượng và khác biệt giữa nguồn
# - Phần 7. Hợp đồng cửa sổ và số mẫu trên CSV
# - Phần 8. Phân phối nhãn giữa các kỳ
# - Phần 9. Hồ sơ bốn lưu vực chẩn đoán
# - Phần 10. Phân mức qmax, ngưỡng Gumbel và gộp đoạn vượt

# %% [markdown]
# ## Phần 1. Môi trường và đường dẫn
#
# Thiết lập thư mục, thư viện và các kỳ dữ liệu dùng chung.

# %% Phần 1 - Môi trường, đường dẫn, hằng số dùng chung
# Tìm thư mục dữ liệu (không ghi cứng đường dẫn Kaggle), in phiên bản thư viện, liệt kê cây thư mục cấp đầu
# kèm số tệp và dung lượng để Phần 2 biết đọc tệp nào.
import importlib
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IS_KAGGLE = Path("/kaggle/working").exists()
SEARCH_ROOT = Path(os.environ.get("LAMAH_ROOT", "/kaggle/input" if IS_KAGGLE else "."))
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("explore_output")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Kỳ chia của bài (02_BasePaper.md Mục 11.1); khởi động 364 ngày trước mỗi kỳ
PERIODS = {
    "train": ("2003-01-01", "2009-12-31"),
    "validation": ("2010-01-01", "2013-12-31"),
    "test": ("2014-01-01", "2017-12-31"),
}
WARMUP_DAYS = 364


def find_dir(root: Path, name: str) -> Path:
    # Tìm thư mục đầu tiên có tên `name` dưới `root`; báo lỗi rõ nếu không có.
    hits = sorted(p for p in root.rglob(name) if p.is_dir())
    if not hits:
        raise FileNotFoundError(f"Không thấy thư mục '{name}' dưới {root} - kiểm tra đã Add Input lamah-ce-ext chưa")
    if len(hits) > 1:
        print(f"Lưu ý: có {len(hits)} thư mục '{name}', dùng cái đầu: {hits}")
    return hits[0]


def dir_summary(path: Path) -> tuple[int, float]:
    # Số tệp và tổng dung lượng (GB) của một thư mục.
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files) / 1e9


LAMAH_DIR = find_dir(SEARCH_ROOT, "LamaH_extended")
EXP_DIR = find_dir(SEARCH_ROOT, "Experiments")

print("Python", platform.python_version(), "|", sys.platform)
for lib in ["numpy", "pandas", "matplotlib", "geopandas", "shapely", "pyogrio", "scipy", "torch", "xarray"]:
    try:
        print(f"  {lib:<11}", importlib.import_module(lib).__version__)
    except ImportError:
        print(f"  {lib:<11} KHÔNG CÓ")

print("\nLAMAH_DIR:", LAMAH_DIR)
print("EXP_DIR  :", EXP_DIR)
for base in [LAMAH_DIR, EXP_DIR]:
    print(f"\n{base.name}/")
    for sub in sorted(base.iterdir()):
        if sub.is_dir():
            n, gb = dir_summary(sub)
            print(f"  {sub.name + '/':<40} {n:>6} tệp  {gb:6.2f} GB")
        else:
            print(f"  {sub.name:<40} {sub.stat().st_size / 1e6:8.2f} MB")

# %% [markdown]
# ## Phần 2. Cấu trúc, tập ID và độ phủ dữ liệu
#
# Kiểm kê nguồn A/D, mã thiếu, lịch quan trắc và quan hệ giữa các tập ID.

# %% [markdown]
# ### 2.1. Cấu trúc thư mục và số lượng tệp
#
# Kiểm kê riêng chuỗi ngày, thuộc tính, gaps và tệp bản đồ; giữ cả tệp phụ nếu có.

# %% Phần 2.1 - Cấu trúc thư mục và số lượng tệp
# Kiểm kê riêng chuỗi ngày, thuộc tính, gaps và tệp bản đồ; giữ cả tệp phụ nếu có.
# Lưu bảng vào WORK_DIR/Tables; không thay đổi dữ liệu đầu vào.
import csv
import re

from IPython.display import display

if not all(name in globals() for name in ("LAMAH_DIR", "EXP_DIR", "WORK_DIR", "SEARCH_ROOT")):
    raise RuntimeError("Chạy Phần 1 trước trong cùng phiên Kaggle.")

TABLE_DIR = WORK_DIR / "Tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
SUB_DATASETS = {"A": "A_basins_total_upstrm", "D": "D_gauges"}
DATE_COLS = ["YYYY", "MM", "DD"]
SOURCE_PREFIXES = ["ERA5L", "ECMWF", "EOBS", "MSWEP", "GLEAM"]
ID_PATTERN = re.compile(r"ID_(\d+)\.csv", re.IGNORECASE)
PREVIEW_ROWS = 5


def a1_show(title, frame, filename):
    print(f"\n{title}")
    display(frame)
    frame.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")


def a1_is_metadata(path):
    return path.name.startswith("._") or "__MACOSX" in path.parts or path.name == ".DS_Store"


def a1_read_header(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream, delimiter=";"), [])


# Kiểm kê cả tệp phụ để giải thích vì sao số tệp có thể khác số trạm.
file_rows = []
for group, folder in SUB_DATASETS.items():
    base = LAMAH_DIR / folder
    if not base.is_dir():
        raise FileNotFoundError(f"Không thấy {base}; kiểm tra cấu trúc Input và đường dẫn ở Phần 1.")
    for path in sorted(base.rglob("*")):
        if path.is_file():
            file_rows.append({
                "group": group, "folder": path.parent.relative_to(LAMAH_DIR).as_posix(),
                "filename": path.name, "extension": path.suffix.lower() or "[none]",
                "metadata": a1_is_metadata(path), "bytes": path.stat().st_size,
            })
A1_FILE_INVENTORY = pd.DataFrame(file_rows)
A1_FILE_INVENTORY.to_csv(TABLE_DIR / "A1_P2_file_inventory.csv", index=False, encoding="utf-8-sig")
folder_summary = A1_FILE_INVENTORY.groupby(
    ["group", "folder", "extension", "metadata"], as_index=False
).agg(files=("filename", "size"), bytes=("bytes", "sum"))
a1_show("(1) Tất cả tệp theo thư mục / đuôi / tệp phụ", folder_summary,
        "A1_P2_folder_summary.csv")

# %% [markdown]
# ### 2.2. Cấu trúc cột của toàn bộ tệp chuỗi ngày
#
# Đọc header từng CSV ngày của A và D để phát hiện cấu trúc khác nhau.

# %% Phần 2.2 - Cấu trúc cột của toàn bộ tệp chuỗi ngày
# Đọc header từng CSV ngày của A và D để phát hiện cấu trúc khác nhau.
# Tách các tệp phụ, không xem chúng như chuỗi của một lưu vực/trạm.
daily_maps, schema_rows, excluded_rows = {}, [], []
for group, folder in SUB_DATASETS.items():
    daily_dir = LAMAH_DIR / folder / "2_timeseries" / "daily"
    if not daily_dir.is_dir():
        raise FileNotFoundError(f"Không thấy {daily_dir}; dừng để đối chiếu bảng (1).")
    daily_maps[group] = {}
    for path in sorted(daily_dir.iterdir()):
        if not path.is_file():
            continue
        match = ID_PATTERN.fullmatch(path.name)
        if a1_is_metadata(path) or match is None:
            excluded_rows.append({"group": group, "filename": path.name,
                                  "reason": "metadata" if a1_is_metadata(path) else "not ID_<digits>.csv"})
            continue
        basin_id = str(int(match.group(1)))
        if basin_id in daily_maps[group]:
            raise ValueError(f"ID ngày trùng {group}/{basin_id}: {daily_maps[group][basin_id]} và {path}")
        daily_maps[group][basin_id] = path
        header = a1_read_header(path)
        schema_rows.append({"group": group, "basin_id": basin_id, "columns": len(header),
                            "header": ";".join(header), "has_date_columns": set(DATE_COLS).issubset(header),
                            "path": path.relative_to(LAMAH_DIR).as_posix()})

A1_DAILY_FILES = daily_maps
A1_SCHEMAS = pd.DataFrame(schema_rows)
if A1_SCHEMAS.empty or any(not paths for paths in daily_maps.values()):
    raise ValueError("Thiếu CSV ngày hợp lệ ở A hoặc D; dừng để kiểm bảng (1).")
A1_SCHEMAS.to_csv(TABLE_DIR / "A1_P2_daily_headers.csv", index=False, encoding="utf-8-sig")
schema_summary = A1_SCHEMAS.groupby(["group", "columns", "header", "has_date_columns"], as_index=False).agg(
    files=("basin_id", "size"), example=("path", "first")
)
with pd.option_context("display.max_colwidth", None):
    a1_show("(2) Mỗi dòng là một kiểu header thật, đã đọc mọi CSV ngày", schema_summary,
            "A1_P2_schema_summary.csv")
excluded = pd.DataFrame(excluded_rows, columns=["group", "filename", "reason"])
a1_show("(3) Tệp ở thư mục daily không được coi là chuỗi ID (tối đa 20 dòng hiển thị)",
        excluded.head(20), "A1_P2_excluded_daily_preview.csv")
excluded.to_csv(TABLE_DIR / "A1_P2_excluded_daily_all.csv", index=False, encoding="utf-8-sig")

# %% [markdown]
# ### 2.3. Đối chiếu ID lưu vực, trạm và danh sách được chọn
#
# Nối các tập ID theo tên tệp và kiểm mọi ID được chọn có đủ khí tượng lẫn lưu lượng.

# %% Phần 2.3 - Đối chiếu ID lưu vực, trạm và danh sách được chọn
# Nối các tập ID theo tên tệp và kiểm mọi ID được chọn có đủ khí tượng lẫn lưu lượng.
# So số đo với ghi chú cũ; chưa suy ra tiêu chí tác giả lọc 451 lưu vực.
basin_file = EXP_DIR / "basins_filtered.txt"
basin_tokens = basin_file.read_text(encoding="utf-8-sig").split()
if not basin_tokens or not all(token.isdecimal() for token in basin_tokens):
    raise ValueError(f"Danh sách ID rỗng hoặc sai định dạng: {basin_file}")
BASINS = [str(int(token)) for token in basin_tokens]
if len(BASINS) != len(set(BASINS)):
    raise ValueError("basins_filtered.txt có ID trùng; cần kiểm trước khi nối dữ liệu.")
selected, a_ids, d_ids = set(BASINS), set(daily_maps["A"]), set(daily_maps["D"])
A1_ID_LINKS = pd.DataFrame([
    {"basin_id": basin, "in_A_daily": basin in a_ids, "in_D_daily": basin in d_ids,
     "selected": basin in selected}
    for basin in sorted(a_ids | d_ids | selected, key=int)
])
A1_ID_LINKS.to_csv(TABLE_DIR / "A1_P2_id_links.csv", index=False, encoding="utf-8-sig")
checks = pd.DataFrame([
    {"item": "A daily unique IDs", "measured": len(a_ids), "previous_note": 859},
    {"item": "D daily unique IDs", "measured": len(d_ids), "previous_note": 882},
    {"item": "Selected unique IDs", "measured": len(selected), "previous_note": 451},
])
checks["matches_previous_note"] = checks["measured"] == checks["previous_note"]
a1_show("(4) Số đo lần này so với ghi chú cũ (không ép cho khớp)", checks, "A1_P2_count_checks.csv")
if not checks["matches_previous_note"].all():
    print("CẢNH BÁO: số đo khác Mục 6.1–6.3; cần đối chiếu phạm vi dữ liệu và bảng kiểm kê.")
for title, ids in [("D có chuỗi ngày nhưng A không có", d_ids - a_ids),
                   ("A có chuỗi ngày nhưng D không có", a_ids - d_ids),
                   ("ID được chọn thiếu A hoặc D", selected - (a_ids & d_ids))]:
    print(f"{title}: {len(ids)} ID; danh sách = {sorted(ids, key=int)}")
print("Tổng tệp dưới D/2_timeseries (gồm tệp phụ và mọi thư mục con):",
      int(A1_FILE_INVENTORY["folder"].str.startswith("D_gauges/2_timeseries").sum()),
      "| ghi chú cũ: 1.112; xem bảng (1) để giải thích thành phần.")

# %% [markdown]
# ### 2.4. Đọc mẫu dữ liệu thô của một lưu vực
#
# Xem 5 dòng đầu và kiểu dữ liệu suy ra từ mẫu nhỏ; không đổi ô trống hay giá trị âm.

# %% Phần 2.4 - Đọc mẫu dữ liệu thô của một lưu vực
# Xem 5 dòng đầu và kiểu dữ liệu suy ra từ mẫu nhỏ; không đổi ô trống hay giá trị âm.
# Tên cột và độ lớn của số chưa đủ để xác nhận đơn vị hoặc độ phủ thời gian.
common = sorted(selected & a_ids & d_ids, key=int)
if not common:
    raise ValueError("Không có ID chung giữa danh sách chọn, A và D.")
EXAMPLE_BASIN = common[0]
column_rows = []
for group in SUB_DATASETS:
    path = daily_maps[group][EXAMPLE_BASIN]
    raw = pd.read_csv(path, sep=";", nrows=PREVIEW_ROWS, dtype=str,
                      keep_default_na=False, encoding="utf-8-sig")
    inferred = pd.read_csv(path, sep=";", nrows=PREVIEW_ROWS, encoding="utf-8-sig")
    print(f"\n(5{group}) ID {EXAMPLE_BASIN}: {path.relative_to(LAMAH_DIR)} — chỉ {PREVIEW_ROWS} dòng đầu")
    display(raw)
    for column in raw.columns:
        source = next((prefix for prefix in SOURCE_PREFIXES if column.startswith(prefix + "_")),
                      "date" if column in DATE_COLS else "Q" if column in ("qmin", "qmean", "qmax") else "other")
        column_rows.append({"group": group, "column": column, "source_prefix": source,
                            "dtype_first_5_rows": str(inferred[column].dtype),
                            "raw_examples": " | ".join(raw[column].unique()[:3]),
                            "unit_status": "chưa xác nhận từ metadata/tài liệu nguồn"})
a1_show("(6) Cột và kiểu suy ra từ mẫu nhỏ; chưa phải kiểu của toàn chuỗi", pd.DataFrame(column_rows),
        "A1_P2_column_preview.csv")

# %% [markdown]
# ### 2.5. Chuẩn bị Q giờ: dùng nguồn sẵn có, giải nén ZIP hoặc tải khi thiếu
#
# Q giờ phục vụ đối chiếu qmax ở Phần 5; chỉ lưu 451 ID và ca 410, không tải khí tượng giờ vào ổ đĩa.

# %% Phần 2.5 - Chuẩn bị Q giờ: dùng nguồn sẵn có, giải nén ZIP hoặc tải khi thiếu
# Q giờ phục vụ đối chiếu qmax ở Phần 5; chỉ lưu 451 ID và ca 410, không tải khí tượng giờ vào ổ đĩa.
# Bật Internet cho lần tải đầu; lần sau dùng lại Input hoặc cache đã kiểm toàn vẹn.
import hashlib
import io
import json
import shutil
import tarfile
import time
import urllib.request
import zipfile

A1_HOURLY_URL = "https://zenodo.org/records/5153305/files/1_LamaH-CE_daily_hourly.tar.gz?download=1"
A1_HOURLY_ARCHIVE_MD5 = "121b2292b288ef1c2ef4d96250be77ac"
A1_HOURLY_CACHE = WORK_DIR / "Sources" / "LamaH_CE_v1"


def a1_prepare_hourly():
    wanted = set(BASINS) | {"410"}
    cache = A1_HOURLY_CACHE.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "hourly_manifest.json"
    roots = list(dict.fromkeys([SEARCH_ROOT.resolve(), WORK_DIR.resolve()]))
    files, records, source_rows = {}, {}, []

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def hourly_id(name):
        parts = name.replace("\\", "/").strip("/").split("/")
        if len(parts) < 4 or [p.lower() for p in parts[-4:-1]] != ["d_gauges", "2_timeseries", "hourly"]:
            return None
        match = ID_PATTERN.fullmatch(parts[-1])
        return str(int(match.group(1))) if match else None

    def destination(name):
        basin = hourly_id(name)
        if basin in wanted:
            return cache / "D_gauges" / "2_timeseries" / "hourly" / f"ID_{basin}.csv"
        # Giữ tài liệu Info kèm nguồn và quy ước thời gian; không lấy đường dẫn tùy ý trong archive.
        parts = name.replace("\\", "/").split("/")
        for pos, part in enumerate(parts):
            if part.lower().startswith("info"):
                tail = parts[pos:]
                if all(p not in ("", ".", "..") and ":" not in p for p in tail):
                    return cache / "SourceInfo" / Path(*tail)
        return None

    def check_header(path):
        header = a1_read_header(path)
        if not {"YYYY", "MM", "DD", "hh"}.issubset(header):
            raise ValueError(f"Header Q giờ cần kiểm: {path}: {header}")

    def store(stream, target, expected_size, source):
        if shutil.disk_usage(cache).free < expected_size + 1024 ** 3:
            raise OSError("Không đủ chỗ lưu tệp giờ và 1 GiB dự phòng; chưa đánh dấu chuẩn bị thành công.")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        size = 0
        checksum = hashlib.sha256()
        with temporary.open("wb") as output:
            while block := stream.read(1024 ** 2):
                output.write(block)
                checksum.update(block)
                size += len(block)
        if size != expected_size:
            raise IOError(f"Tệp chưa đủ byte: {target.name}: {size}/{expected_size}")
        if hourly_id(target.as_posix()) is not None:
            check_header(temporary)
        temporary.replace(target)
        basin = hourly_id(target.as_posix())
        if basin is not None:
            records[basin] = {"relative_path": target.relative_to(cache).as_posix(), "bytes": size,
                              "sha256": checksum.hexdigest(), "source": source}
            files[basin] = target

    # Nguồn đã giải nén trong Input/working; cache riêng chỉ nhận qua manifest kiểm SHA-256 bên dưới.
    directories = set()
    archives = set()
    for root in roots:
        if not root.is_dir():
            continue
        directories.update(p.resolve() for p in root.rglob("hourly") if p.is_dir())
        archives.update(p.resolve() for p in root.rglob("*.zip") if p.is_file())
    for directory in sorted(directories):
        if directory.is_relative_to(cache):
            continue
        added = 0
        for path in sorted(directory.glob("*.csv")):
            basin = hourly_id(path.as_posix())
            if basin in wanted and basin not in files:
                check_header(path)
                files[basin] = path
                added += 1
        if added:
            source_rows.append({"source": str(directory), "method": "existing_directory", "added_ids": added})
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for basin, record in manifest.get("files", {}).items():
            if basin not in wanted or basin in files:
                continue
            path = (cache / record["relative_path"]).resolve()
            if (path.is_relative_to(cache) and path.is_file() and path.stat().st_size == record["bytes"]
                    and digest(path) == record["sha256"]):
                check_header(path)
                files[basin], records[basin] = path, record
        source_rows.append({"source": str(cache), "method": "verified_cache", "added_ids": len(records)})
    for archive_path in sorted(archives):
        if wanted.issubset(files):
            break
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            candidates = [m for m in members if not m.is_dir() and hourly_id(m.filename) in wanted - files.keys()]
            if not candidates:
                continue
            print(f"Đọc Q giờ từ ZIP sẵn có: {archive_path}", flush=True)
            for member in members:
                target = destination(member.filename)
                basin = hourly_id(member.filename)
                if member.is_dir() or target is None or (basin is not None and basin in files):
                    continue
                with archive.open(member) as stream:
                    store(stream, target, member.file_size, str(archive_path))
            source_rows.append({"source": str(archive_path), "method": "existing_zip", "added_ids": len(candidates)})
    if not wanted.issubset(files):
        missing = wanted - files.keys()
        print(f"Còn thiếu {len(missing)} ID Q giờ. Tải theo luồng từ Zenodo (gói nguồn khoảng 14,8 GB nén).", flush=True)
        print("Chỉ lưu Q giờ cần dùng và Info; không lưu cả TAR.GZ hoặc khí tượng giờ. Nếu lỗi mạng, chạy lại 2.5.", flush=True)

        class Meter(io.RawIOBase):
            def __init__(self, response):
                self.response, self.count = response, 0
                self.md5 = hashlib.md5()
                self.started = self.last_log = time.monotonic()

            def readable(self):
                return True

            def readinto(self, buffer):
                block = self.response.read(len(buffer))
                buffer[:len(block)] = block
                self.count += len(block)
                self.md5.update(block)
                if time.monotonic() - self.last_log >= 30:
                    self.last_log = time.monotonic()
                    print(f"Đã đọc {self.count / 1e9:.2f} GB nén | {(self.last_log - self.started) / 60:.1f} phút | "
                          f"có {len(files)}/{len(wanted)} ID", flush=True)
                return len(block)

        request = urllib.request.Request(A1_HOURLY_URL, headers={"User-Agent": "LamaHCE-A1-hourly-audit/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                meter = Meter(response)
                with io.BufferedReader(meter, buffer_size=4 * 1024 ** 2) as stream:
                    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                        for member in archive:
                            target = destination(member.name)
                            basin = hourly_id(member.name)
                            if not member.isfile() or target is None or (basin is not None and basin in files):
                                continue
                            with archive.extractfile(member) as content:
                                store(content, target, member.size, A1_HOURLY_URL)
                    # Đọc hết phần còn lại để MD5 áp dụng toàn bộ gói nén, không chỉ các thành viên đã giữ.
                    while stream.read(4 * 1024 ** 2):
                        pass
                if meter.md5.hexdigest() != A1_HOURLY_ARCHIVE_MD5:
                    raise IOError("MD5 toàn gói tải khác Zenodo; chưa công nhận cache mới.")
            source_rows.append({"source": A1_HOURLY_URL, "method": "download_verified_md5", "added_ids": len(missing)})
        except Exception as error:
            raise RuntimeError("Chuẩn bị Q giờ chưa xong. Kiểm Internet Kaggle, mạng và dung lượng rồi chạy lại 2.5. "
                               "Tệp mới chưa có manifest hợp lệ sẽ không được dùng làm cache hoàn chỉnh.") from error
    remaining = wanted - files.keys()
    if remaining:
        raise FileNotFoundError(f"Nguồn giờ chưa đủ ID: {sorted(remaining, key=int)}")
    if records:
        temporary = manifest_path.with_suffix(".json.part")
        temporary.write_text(json.dumps({"source_record": "https://zenodo.org/records/5153305", "files": records},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
    return files, pd.DataFrame(source_rows, columns=["source", "method", "added_ids"])


A1_HOURLY_FILES, A1_HOURLY_SOURCES = a1_prepare_hourly()
A1_HOURLY_DIRS = sorted({path.parent for path in A1_HOURLY_FILES.values()})
a1_show("(7) Nguồn Q giờ đã chuẩn bị; chưa kiểm tổng hợp ngày/múi giờ", A1_HOURLY_SOURCES,
        "A1_P2_hourly_sources.csv")
print("ID mô hình có Q giờ:", len(set(BASINS) & A1_HOURLY_FILES.keys()), "/", len(BASINS))
print("Ca 410 có Q giờ:", "410" in A1_HOURLY_FILES)
for basin in ["10", "55", "94", "410", "742", "758", "800"]:
    if basin in A1_HOURLY_FILES:
        print(f"ID {basin}: {A1_HOURLY_FILES[basin]} | header: {a1_read_header(A1_HOURLY_FILES[basin])}")

# %% [markdown]
# ### 2.6. Đọc cấu trúc tệp gaps
#
# Xem tên tệp, header và 8 dòng thô của một tệp gaps để hiểu 230 tệp bổ sung chứa gì.

# %% Phần 2.6 - Đọc cấu trúc tệp gaps
# Xem tên tệp, header và 8 dòng thô của một tệp gaps để hiểu 230 tệp bổ sung chứa gì.
# Chưa dùng gaps làm cờ cho chuỗi ngày; phải xác nhận ý nghĩa và cách gắn thời gian trước.
GAP_PREVIEW_ROWS = 8
GAP_DIR = LAMAH_DIR / "D_gauges" / "2_timeseries" / "gaps"
gap_files = sorted(path for path in GAP_DIR.glob("*.csv") if not a1_is_metadata(path))
if not gap_files:
    raise FileNotFoundError(f"Không có CSV gaps ở {GAP_DIR}; kiểm tra thư mục dữ liệu gaps.")

print("Thư mục:", GAP_DIR)
print("Số tệp CSV:", len(gap_files))
print("Ví dụ tên tệp:", [path.name for path in gap_files[:6]])

gap_example = next((path for path in gap_files
                    if (match := ID_PATTERN.fullmatch(path.name))
                    and str(int(match.group(1))) in selected), gap_files[0])
with gap_example.open(encoding="utf-8-sig", newline="") as stream:
    gap_lines = [line.rstrip("\r\n") for _, line in zip(range(GAP_PREVIEW_ROWS + 1), stream)]
print("\nTệp mẫu:", gap_example.name)
print("Header và tối đa 8 dòng tiếp theo, giữ nguyên dấu phân cách:")
for line in gap_lines:
    print(line)

gap_sample = pd.read_csv(gap_example, sep=";", nrows=GAP_PREVIEW_ROWS, dtype=str,
                         keep_default_na=False, encoding="utf-8-sig")
print("\nBảng mẫu (chuỗi ký tự, giữ nguyên ô trống):")
with pd.option_context("display.max_columns", None, "display.max_colwidth", 100):
    display(gap_sample)
gap_sample.to_csv(TABLE_DIR / "A1_P2_gap_preview.csv", index=False, encoding="utf-8-sig")
print("\nĐã lưu:", TABLE_DIR / "A1_P2_gap_preview.csv")

# %% [markdown]
# ### 2.7. Đối chiếu một mốc gaps với lưu lượng ngày
#
# Đọc Q ngày của cùng ID và xem ngày chứa mốc gaps đầu tiên cùng hai ngày lân cận.

# %% Phần 2.7 - Đối chiếu một mốc gaps với lưu lượng ngày
# Đọc Q ngày của cùng ID và xem ngày chứa mốc gaps đầu tiên cùng hai ngày lân cận.
# Giữ nguyên ô trống, số âm; đây chỉ là đối chiếu lịch ghi trong tệp, chưa xác nhận múi giờ.
GAP_TIME_COLUMNS = {"YYYY": "year", "MM": "month", "DD": "day", "hh": "hour", "mm": "minute"}
Q_COLUMNS = ["qmin", "qmean", "qmax"]
GAP_CONTEXT_DAYS = 1
if gap_sample.empty or not set(GAP_TIME_COLUMNS).issubset(gap_sample.columns):
    raise ValueError("Mẫu gaps rỗng hoặc khác header đã quan sát; kiểm tra cấu trúc ở Phần 2.6.")
gap_times = pd.to_datetime(gap_sample[list(GAP_TIME_COLUMNS)].rename(columns=GAP_TIME_COLUMNS))
gap_time = gap_times.iloc[0]
gap_id_match = ID_PATTERN.fullmatch(gap_example.name)
if gap_id_match is None:
    raise ValueError(f"Không xác định được ID từ {gap_example.name}.")
gap_basin = str(int(gap_id_match.group(1)))
q_path = A1_DAILY_FILES["D"][gap_basin]
q_raw = pd.read_csv(q_path, sep=";", usecols=DATE_COLS + Q_COLUMNS,
                    dtype=str, keep_default_na=False, encoding="utf-8-sig")
q_dates = pd.to_datetime(q_raw[DATE_COLS].rename(columns={"YYYY": "year", "MM": "month", "DD": "day"}))
if q_dates.duplicated().any():
    raise ValueError(f"Tệp Q của ID {gap_basin} có ngày trùng; cần kiểm trước khi đối chiếu.")
q_indexed = q_raw[Q_COLUMNS].set_axis(pd.DatetimeIndex(q_dates, name="date"))
context_dates = pd.date_range(gap_time.normalize() - pd.Timedelta(days=GAP_CONTEXT_DAYS),
                             gap_time.normalize() + pd.Timedelta(days=GAP_CONTEXT_DAYS), name="date")
A1_GAP_Q_PROBE = q_indexed.reindex(context_dates)
A1_GAP_Q_PROBE.insert(0, "row_present", context_dates.isin(q_indexed.index))
print(f"ID {gap_basin} | Mốc gaps: {gap_time} (múi giờ chưa xác nhận)")
print("Tệp Q:", q_path)
print("row_present=False: không có dòng; <empty>: ô rỗng trong dòng có sẵn; số âm giữ nguyên.")
display(A1_GAP_Q_PROBE.replace("", "<empty>"))
probe_path = TABLE_DIR / f"A1_P2_gap_Q_probe_ID_{gap_basin}.csv"
A1_GAP_Q_PROBE.to_csv(probe_path, encoding="utf-8-sig")
print("Đã lưu:", probe_path)

# %% [markdown]
# ### 2.8. Độ phủ thời gian của từng biến khí tượng trên toàn bộ lưu vực A
#
# Quét mọi tệp A, lưu ngày đầu/cuối có số hữu hạn và đếm riêng -999 cho từng biến, từng ID.

# %% Phần 2.8 - Độ phủ thời gian của từng biến khí tượng trên toàn bộ lưu vực A
# Quét mọi tệp A, lưu ngày đầu/cuối có số hữu hạn và đếm riêng -999 cho từng biến, từng ID.
# Chỉ dùng mặt nạ để khảo sát, không sửa số liệu; khoảng đầu-cuối không chứng minh đủ ngày ở giữa.
import time

DATE_COLUMN_NAMES = {"YYYY": "year", "MM": "month", "DD": "day"}
forcing_coverage_rows, forcing_file_rows = [], []
scan_started = time.perf_counter()
for file_no, (basin, path) in enumerate(A1_DAILY_FILES["A"].items(), start=1):
    forcing = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    dates = pd.to_datetime(forcing[DATE_COLS].rename(columns=DATE_COLUMN_NAMES))
    if dates.empty or dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"ID {basin}: ngày rỗng, không hợp lệ hoặc trùng; dừng để kiểm.")
    variables = [column for column in forcing.columns
                 if any(column.startswith(prefix + "_") for prefix in SOURCE_PREFIXES)]
    numeric = forcing[variables].apply(pd.to_numeric, errors="raise")
    valid = np.isfinite(numeric) & numeric.ne(-999)
    forcing_file_rows.append({"basin_id": basin, "rows": len(dates),
                              "file_start": dates.min(), "file_end": dates.max(),
                              "missing_date_rows": (dates.max() - dates.min()).days + 1 - len(dates)})
    for column in variables:
        observed_dates = dates[valid[column]]
        forcing_coverage_rows.append({
            "basin_id": basin, "source": column.split("_", 1)[0], "variable": column,
            "first_value_date": observed_dates.min(), "last_value_date": observed_dates.max(),
            "valid_days": int(valid[column].sum()), "minus999_cells": int(numeric[column].eq(-999).sum()),
            "nonfinite_cells": int((~np.isfinite(numeric[column])).sum()),
        })
    if file_no % 100 == 0 or file_no == len(A1_DAILY_FILES["A"]):
        print(f"Đã quét {file_no}/{len(A1_DAILY_FILES['A'])} tệp A", flush=True)

A1_FORCING_COVERAGE = pd.DataFrame(forcing_coverage_rows)
A1_FORCING_FILE_SPANS = pd.DataFrame(forcing_file_rows)
A1_SOURCE_COVERAGE = A1_FORCING_COVERAGE.groupby("source", as_index=False).agg(
    basins=("basin_id", "nunique"), variables=("variable", "nunique"),
    earliest_first=("first_value_date", "min"), latest_first=("first_value_date", "max"),
    earliest_last=("last_value_date", "min"), latest_last=("last_value_date", "max"),
    empty_series=("valid_days", lambda values: int(values.eq(0).sum())),
    minus999_cells=("minus999_cells", "sum"),
)
print("\n(1) Khoảng ngày có dòng trong toàn bộ tệp A:")
display(A1_FORCING_FILE_SPANS.agg({"file_start": ["min", "max"], "file_end": ["min", "max"]}))
print("Số tệp thiếu dòng ngày bên trong khoảng đầu-cuối:",
      int(A1_FORCING_FILE_SPANS["missing_date_rows"].gt(0).sum()))
print("\n(2) Mốc có giá trị theo nguồn, gộp mọi biến và mọi ID; -999 đếm riêng, tạm không coi là số đo:")
display(A1_SOURCE_COVERAGE)
for table_name, frame in [("forcing_coverage", A1_FORCING_COVERAGE),
                          ("forcing_file_spans", A1_FORCING_FILE_SPANS), ("source_coverage", A1_SOURCE_COVERAGE)]:
    frame.to_csv(TABLE_DIR / f"A1_P2_{table_name}.csv", index=False, encoding="utf-8-sig")
print(f"Thời gian quét: {time.perf_counter() - scan_started:.1f} giây. Bảng chi tiết lưu ở {TABLE_DIR}.")
print("earliest/latest_first: ngày bắt đầu sớm/muộn nhất; earliest/latest_last: ngày kết thúc sớm/muộn nhất.")
print("empty_series đếm cặp (ID, biến) không có số đo theo quy ước quét; chưa đo Q hoặc xác nhận đơn vị.")

# %% [markdown]
# ### 2.9. Ngày kết thúc ECMWF theo biến và lưu vực
#
# Dùng bảng đã quét ở 2.8, không đọc lại 859 tệp; tách toàn bộ A và 451 ID được chọn.

# %% Phần 2.9 - Ngày kết thúc ECMWF theo biến và lưu vực
# Dùng bảng đã quét ở 2.8, không đọc lại 859 tệp; tách toàn bộ A và 451 ID được chọn.
# Kiểm riêng năm biến forecast trong mã BiasCast; mốc cuối chung chưa chứng minh đủ dữ liệu ở giữa.
A1_FORECAST_VARIABLES = ["ECMWF_t2m", "ECMWF_d2m", "ECMWF_ssrd", "ECMWF_tp", "ECMWF_e"]
coverage = (A1_FORCING_COVERAGE.copy() if "A1_FORCING_COVERAGE" in globals()
            else pd.read_csv(TABLE_DIR / "A1_P2_forcing_coverage.csv", dtype={"basin_id": str}))
coverage["basin_id"] = coverage["basin_id"].astype(str).map(lambda value: str(int(value)))
coverage["last_value_date"] = pd.to_datetime(coverage["last_value_date"])
ecmwf = coverage.loc[coverage["source"].eq("ECMWF")].copy()
selected_ids = {str(int(value)) for value in (EXP_DIR / "basins_filtered.txt").read_text().split()}
ecmwf["used_forecast"] = ecmwf["variable"].isin(A1_FORECAST_VARIABLES)
scope_tables = []
for scope, subset in [("all_A", ecmwf), ("selected", ecmwf[ecmwf["basin_id"].isin(selected_ids)])]:
    counts = subset.groupby(["variable", "used_forecast", "last_value_date"], dropna=False).agg(
        basins=("basin_id", "nunique")).reset_index()
    counts.insert(0, "scope", scope)
    scope_tables.append(counts)
A1_ECMWF_END_COUNTS = pd.concat(scope_tables, ignore_index=True)
print("(1) Số lưu vực theo biến và ngày cuối có giá trị; used_forecast đánh dấu 5 biến mô hình dùng:")
display(A1_ECMWF_END_COUNTS)

A1_ECMWF_END_BY_BASIN = ecmwf.pivot(index="basin_id", columns="variable", values="last_value_date")
missing_variables = set(A1_FORECAST_VARIABLES) - set(A1_ECMWF_END_BY_BASIN.columns)
if missing_variables:
    raise ValueError(f"Thiếu biến forecast trong bảng độ phủ: {sorted(missing_variables)}")
five_ends = A1_ECMWF_END_BY_BASIN[A1_FORECAST_VARIABLES]
end_profile = pd.DataFrame({"selected": five_ends.index.isin(selected_ids),
                            "series_with_values": five_ends.notna().sum(axis=1),
                            "earliest_end": five_ends.min(axis=1), "latest_end": five_ends.max(axis=1)})
print("\n(2) Các kiểu ngày cuối của 5 biến forecast trong 451 ID (không kiểm ô trống ở giữa):")
display(end_profile[end_profile["selected"]].reset_index().groupby(
    ["series_with_values", "earliest_end", "latest_end"], dropna=False
).agg(basins=("basin_id", "size"),
      example_ids=("basin_id", lambda ids: ", ".join(sorted(ids, key=int)[:10]))).reset_index())
print("\n(3) Ví dụ tối đa 10 ID được chọn có ngày cuối khác nhau giữa 5 biến:")
different_ends = end_profile["selected"] & five_ends.nunique(axis=1, dropna=False).gt(1)
display(A1_ECMWF_END_BY_BASIN.loc[different_ends].head(10))
A1_ECMWF_END_COUNTS.to_csv(TABLE_DIR / "A1_P2_ecmwf_end_counts.csv", index=False, encoding="utf-8-sig")
A1_ECMWF_END_BY_BASIN.to_csv(TABLE_DIR / "A1_P2_ecmwf_end_by_basin.csv", encoding="utf-8-sig")
end_profile.to_csv(TABLE_DIR / "A1_P2_ecmwf_forecast_end_profile.csv", encoding="utf-8-sig")
print("Đã lưu ba bảng ECMWF vào:", TABLE_DIR)

# %% [markdown]
# ### 2.10. Quét độ phủ và mã thiếu của toàn bộ chuỗi Q ngày
#
# Đọc 882 tệp D, phân biệt thiếu dòng ngày, ô rỗng, số âm và giá trị không chuyển được thành số.

# %% Phần 2.10 - Quét độ phủ và mã thiếu của toàn bộ chuỗi Q ngày
# Đọc 882 tệp D, phân biệt thiếu dòng ngày, ô rỗng, số âm và giá trị không chuyển được thành số.
# Đo mốc có Q hữu hạn không âm; giữ nguyên dữ liệu thô, chưa kiểm quan hệ qmin/qmean/qmax.
q_coverage_rows, q_file_rows = [], []
q_scan_started = time.perf_counter()
for file_no, (basin, path) in enumerate(A1_DAILY_FILES["D"].items(), start=1):
    raw_q = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    dates = pd.to_datetime(raw_q[DATE_COLS].rename(columns=DATE_COLUMN_NAMES))
    if dates.empty or dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"ID {basin}: ngày rỗng, không hợp lệ hoặc trùng; dừng để kiểm.")
    q_file_rows.append({"basin_id": basin, "selected": basin in selected_ids, "rows": len(dates),
                        "file_start": dates.min(), "file_end": dates.max(),
                        "missing_date_rows": (dates.max() - dates.min()).days + 1 - len(dates)})
    for column in Q_COLUMNS:
        text_values = raw_q[column].str.strip()
        values = pd.to_numeric(text_values, errors="coerce")
        valid = np.isfinite(values) & values.ge(0)
        observed_dates = dates[valid]
        q_coverage_rows.append({
            "basin_id": basin, "selected": basin in selected_ids, "variable": column,
            "first_value_date": observed_dates.min(), "last_value_date": observed_dates.max(),
            "valid_days": int(valid.sum()), "blank_cells": int(text_values.eq("").sum()),
            "negative_cells": int(values.lt(0).sum()), "minus999_cells": int(values.eq(-999).sum()),
            "non_numeric_nonblank": int((values.isna() & text_values.ne("")).sum()),
            "infinite_cells": int(np.isinf(values).sum()),
        })
    if file_no % 200 == 0 or file_no == len(A1_DAILY_FILES["D"]):
        print(f"Đã quét {file_no}/{len(A1_DAILY_FILES['D'])} tệp D", flush=True)

A1_Q_COVERAGE = pd.DataFrame(q_coverage_rows)
A1_Q_FILE_SPANS = pd.DataFrame(q_file_rows)
q_report = pd.concat([A1_Q_COVERAGE.assign(scope="all_D"),
                      A1_Q_COVERAGE[A1_Q_COVERAGE["selected"]].assign(scope="selected")], ignore_index=True)
q_count_columns = ["valid_days", "blank_cells", "negative_cells", "minus999_cells",
                   "non_numeric_nonblank", "infinite_cells"]
A1_Q_COVERAGE_SUMMARY = q_report.groupby(["scope", "variable"], as_index=False).agg(
    basins=("basin_id", "nunique"), earliest_first=("first_value_date", "min"),
    latest_first=("first_value_date", "max"), earliest_last=("last_value_date", "min"),
    latest_last=("last_value_date", "max"), empty_series=("valid_days", lambda counts: int(counts.eq(0).sum())),
    **{column: (column, "sum") for column in q_count_columns},
)
print("\n(1) Ngày có dòng trong 882 tệp D:")
display(A1_Q_FILE_SPANS.agg({"file_start": ["min", "max"], "file_end": ["min", "max"]}))
print("Số tệp thiếu dòng ngày bên trong khoảng đầu-cuối:", int(A1_Q_FILE_SPANS["missing_date_rows"].gt(0).sum()))
print("\n(2) Mốc có Q hữu hạn không âm; empty_series là số cặp (ID, biến) không có giá trị như vậy:")
display(A1_Q_COVERAGE_SUMMARY.drop(columns=q_count_columns))
print("\n(3) Đếm trên toàn bộ dòng có trong tệp; minus999_cells là tập con của negative_cells, không cộng hai cột:")
display(A1_Q_COVERAGE_SUMMARY[["scope", "variable"] + q_count_columns])
for table_name, frame in [("Q_coverage", A1_Q_COVERAGE), ("Q_file_spans", A1_Q_FILE_SPANS),
                          ("Q_coverage_summary", A1_Q_COVERAGE_SUMMARY)]:
    frame.to_csv(TABLE_DIR / f"A1_P2_{table_name}.csv", index=False, encoding="utf-8-sig")
print(f"Thời gian quét: {time.perf_counter() - q_scan_started:.1f} giây. Bảng chi tiết lưu ở {TABLE_DIR}.")
print("non_numeric_nonblank gồm cả ký hiệu chữ như NaN nếu có; ô rỗng đếm riêng. Chưa tính tỷ lệ thiếu theo kỳ.")

# %% [markdown]
# ### 2.11. Khoanh vùng các giá trị Q âm
#
# Chỉ đọc lại các ID có số âm theo bảng 2.10; lưu giá trị thô, ngày và kỳ thí nghiệm.

# %% Phần 2.11 - Khoanh vùng các giá trị Q âm
# Chỉ đọc lại các ID có số âm theo bảng 2.10; lưu giá trị thô, ngày và kỳ thí nghiệm.
# Đếm theo ô và theo ngày riêng biệt; chưa đổi số âm thành NaN hoặc kết luận nguyên nhân.
negative_ids = A1_Q_COVERAGE.loc[A1_Q_COVERAGE["negative_cells"].gt(0), "basin_id"].astype(str).unique()
negative_parts = []
for basin in sorted(negative_ids, key=int):
    raw_q = pd.read_csv(A1_DAILY_FILES["D"][basin], sep=";", dtype=str,
                        keep_default_na=False, encoding="utf-8-sig")
    values = raw_q[Q_COLUMNS].apply(pd.to_numeric, errors="coerce")
    negative = values.lt(0)
    hit = raw_q.loc[negative.any(axis=1), Q_COLUMNS].copy()
    hit.insert(0, "date", pd.to_datetime(raw_q.loc[hit.index, DATE_COLS].rename(columns=DATE_COLUMN_NAMES)))
    hit.insert(0, "selected", basin in selected_ids)
    hit.insert(0, "basin_id", basin)
    hit["negative_variables"] = negative.loc[hit.index].apply(
        lambda row: ", ".join(row.index[row]), axis=1)
    negative_parts.append(hit)

if not negative_parts:
    raise ValueError("Bảng 2.10 không ghi nhận số âm; gửi lại output để kiểm trước khi tiếp tục.")
A1_NEGATIVE_Q_ROWS = pd.concat(negative_parts, ignore_index=True)
A1_NEGATIVE_Q_ROWS["period"] = "outside_2003_2017"
for period, (start, end) in PERIODS.items():
    A1_NEGATIVE_Q_ROWS.loc[A1_NEGATIVE_Q_ROWS["date"].between(start, end), "period"] = period
print(f"Đã đọc lại {len(negative_ids)} tệp có ghi nhận số âm.")
measured = A1_NEGATIVE_Q_ROWS[Q_COLUMNS].apply(pd.to_numeric, errors="coerce").lt(0).sum()
previous = A1_Q_COVERAGE.groupby("variable")["negative_cells"].sum()
count_check = pd.DataFrame({"measured_cells": measured, "cells_from_2_10": previous}).reindex(Q_COLUMNS)
count_check["matches"] = count_check["measured_cells"].eq(count_check["cells_from_2_10"])
print("\n(1) Kiểm lại số ô âm, phải khớp 2.10:")
display(count_check)
print("\n(2) Các ID và kỳ có ngày chứa Q âm; days đếm ngày, không phải số ô:")
display(A1_NEGATIVE_Q_ROWS.groupby(["basin_id", "selected", "period"], as_index=False).agg(
    days=("date", "size"), first_date=("date", "min"), last_date=("date", "max")))
print("\n(3) Tối đa 20 dòng thô, ưu tiên 451 ID được chọn; toàn bộ dòng lưu trong CSV:")
display(A1_NEGATIVE_Q_ROWS.sort_values(["selected", "date"], ascending=[False, True]).head(20))
negative_path = TABLE_DIR / "A1_P2_negative_Q_rows.csv"
A1_NEGATIVE_Q_ROWS.to_csv(negative_path, index=False, encoding="utf-8-sig")
print("Đã lưu:", negative_path)

# %% [markdown]
# ### 2.12. Nhất quán qmin/qmean/qmax, ngày bằng 0 và mặt nạ thiếu
#
# Quét Q ngày: so từng cặp khi cả hai số hữu hạn; đếm riêng vi phạm trên ba Q không âm.

# %% Phần 2.12 - Nhất quán qmin/qmean/qmax, ngày bằng 0 và mặt nạ thiếu
# Quét Q ngày: so từng cặp khi cả hai số hữu hạn; đếm riêng vi phạm trên ba Q không âm.
# Kiểm ngày bằng 0 và thiếu lệch qmean/qmax; chỉ lưu chẩn đoán, không sửa hay loại ngày.
Q_ORDER_PAIRS = [("qmin", "qmean"), ("qmean", "qmax"), ("qmin", "qmax")]
quality_counts, quality_examples = [], []
quality_started = time.perf_counter()
for file_no, (basin, path) in enumerate(A1_DAILY_FILES["D"].items(), start=1):
    raw_q = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    q = raw_q[Q_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(q)
    usable = finite & q.ge(0)
    flags = pd.DataFrame({f"{low}_gt_{high}": finite[low] & finite[high] & q[low].gt(q[high])
                          for low, high in Q_ORDER_PAIRS})
    order_bad = flags.any(axis=1)
    flags["order_bad_nonnegative"] = order_bad & usable.all(axis=1)
    flags["qmean_qmax_missing_diff"] = usable["qmean"].ne(usable["qmax"])
    for column in Q_COLUMNS:
        flags[f"{column}_zero"] = q[column].eq(0)
    quality_counts.append({"basin_id": basin, "selected": basin in selected_ids,
                           "days_in_file": len(q), "nonnegative_triplet_days": int(usable.all(axis=1).sum()),
                           **flags.sum().astype(int).to_dict()})
    # Giữ mọi vi phạm và ngày thiếu lệch; ngày qmax=0 chỉ lấy tối đa 3 ví dụ mỗi ID.
    keep = order_bad | flags["qmean_qmax_missing_diff"] | (flags["qmax_zero"] & flags["qmax_zero"].cumsum().le(3))
    if keep.any():
        example = raw_q.loc[keep, Q_COLUMNS].copy()
        example.insert(0, "date", pd.to_datetime(raw_q.loc[keep, DATE_COLS].rename(columns=DATE_COLUMN_NAMES)))
        example.insert(0, "selected", basin in selected_ids)
        example.insert(0, "basin_id", basin)
        example["issues"] = flags.loc[keep].apply(lambda row: ", ".join(row.index[row]), axis=1)
        example["order_bad"] = order_bad.loc[keep]
        quality_examples.append(example)
    if file_no % 200 == 0 or file_no == len(A1_DAILY_FILES["D"]):
        print(f"Đã kiểm {file_no}/{len(A1_DAILY_FILES['D'])} tệp D", flush=True)

A1_Q_QUALITY_COUNTS = pd.DataFrame(quality_counts)
A1_Q_QUALITY_EXAMPLES = (pd.concat(quality_examples, ignore_index=True) if quality_examples
                         else pd.DataFrame(columns=["basin_id", "selected", "date"] + Q_COLUMNS + ["issues", "order_bad"]))
metrics = [column for column in A1_Q_QUALITY_COUNTS if column not in ("basin_id", "selected")]
print("\n(1) Số ngày theo tiêu chí, đếm toàn chuỗi; các hàng có thể chồng lấp, không cộng thành tổng:")
display(pd.DataFrame({"all_D": A1_Q_QUALITY_COUNTS[metrics].sum(),
                      "selected": A1_Q_QUALITY_COUNTS.loc[A1_Q_QUALITY_COUNTS["selected"], metrics].sum()}).rename_axis("metric"))
print("\n(2) Tối đa 15 ví dụ vi phạm thứ tự; gt nghĩa là lớn hơn, so trực tiếp số trong tệp không áp dung sai:")
display(A1_Q_QUALITY_EXAMPLES[A1_Q_QUALITY_EXAMPLES["order_bad"].astype(bool)].head(15))
print("\n(3) Tối đa 15 ví dụ qmax=0 hoặc thiếu lệch qmean/qmax, ưu tiên ID được chọn:")
other_examples = A1_Q_QUALITY_EXAMPLES["issues"].str.contains("qmax_zero|qmean_qmax_missing_diff", regex=True, na=False).astype(bool)
display(A1_Q_QUALITY_EXAMPLES[other_examples].sort_values("selected", ascending=False).head(15))
A1_Q_QUALITY_COUNTS.to_csv(TABLE_DIR / "A1_P2_Q_quality_per_basin.csv", index=False, encoding="utf-8-sig")
A1_Q_QUALITY_EXAMPLES.to_csv(TABLE_DIR / "A1_P2_Q_quality_diagnostics.csv", index=False, encoding="utf-8-sig")
print(f"Thời gian kiểm: {time.perf_counter() - quality_started:.1f} giây. Đã lưu hai bảng vào {TABLE_DIR}.")
print("missing_diff coi Q âm/không hữu hạn là thiếu; vi phạm thứ tự xét cả số âm khi hai số hữu hạn.")

# %% [markdown]
# ### 2.13. Phân bố qmax bằng 0 trong 451 lưu vực
#
# Đếm theo lưu vực/kỳ, chia cho ngày qmax hợp lệ; đo chuỗi 0 liên tiếp và tháng xảy ra.

# %% Phần 2.13 - Phân bố qmax bằng 0 trong 451 lưu vực
# Đếm theo lưu vực/kỳ, chia cho ngày qmax hợp lệ; đo chuỗi 0 liên tiếp và tháng xảy ra.
# NaN ngắt chuỗi 0; độ dài đoạn được cắt tại ranh giới kỳ, không suy ra nguyên nhân số 0.
q_calendar = pd.date_range(A1_Q_FILE_SPANS["file_start"].min(), A1_Q_FILE_SPANS["file_end"].max())
ZERO_PERIODS = {"full_record": (q_calendar.min(), q_calendar.max()),
                "pre_2003": (q_calendar.min(), pd.Timestamp(PERIODS["train"][0]) - pd.Timedelta(days=1)),
                **PERIODS}
zero_rows = []
zero_started = time.perf_counter()
for file_no, basin in enumerate(sorted(selected_ids, key=int), start=1):
    frame = pd.read_csv(A1_DAILY_FILES["D"][basin], sep=";", usecols=DATE_COLS + ["qmax"], encoding="utf-8-sig")
    dates = pd.to_datetime(frame[DATE_COLS].rename(columns=DATE_COLUMN_NAMES))
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"ID {basin}: ngày sai hoặc trùng; dừng để kiểm.")
    values = pd.to_numeric(frame["qmax"], errors="raise")
    qmax = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates)).reindex(q_calendar)
    qmax = qmax.where(np.isfinite(qmax) & qmax.ge(0))
    for period, (start, end) in ZERO_PERIODS.items():
        window = qmax.loc[start:end]
        zero = window.eq(0)
        run_lengths = zero.groupby((~zero).cumsum()).sum()
        valid_days, zero_days = int(window.notna().sum()), int(zero.sum())
        zero_dates = window.index[zero]
        month_counts = zero_dates.month.value_counts().reindex(range(1, 13), fill_value=0)
        zero_rows.append({
            "basin_id": basin, "period": period, "valid_days": valid_days, "zero_days": zero_days,
            "zero_pct": 100 * zero_days / valid_days if valid_days else np.nan,
            "longest_zero_run_days": int(run_lengths.max()) if len(run_lengths) else 0,
            "first_zero_date": zero_dates.min(), "last_zero_date": zero_dates.max(),
            **{f"zero_m{month:02d}": int(count) for month, count in month_counts.items()},
        })
    if file_no % 100 == 0 or file_no == len(selected_ids):
        print(f"Đã đếm {file_no}/{len(selected_ids)} lưu vực", flush=True)

A1_QMAX_ZERO_PROFILE = pd.DataFrame(zero_rows)
zero_summary = A1_QMAX_ZERO_PROFILE.groupby("period").agg(
    valid_days=("valid_days", "sum"), zero_days=("zero_days", "sum"),
    basins_with_zero=("zero_days", lambda counts: int(counts.gt(0).sum())),
    longest_run_days=("longest_zero_run_days", "max"))
zero_summary["zero_pct"] = 100 * zero_summary["zero_days"] / zero_summary["valid_days"].replace(0, np.nan)
zero_summary = zero_summary.reindex(ZERO_PERIODS)
print("\n(1) Theo kỳ; full_record đã gồm các kỳ bên dưới, không cộng lặp:")
display(zero_summary)
full_zero = A1_QMAX_ZERO_PROFILE[A1_QMAX_ZERO_PROFILE["period"].eq("full_record")]
print("\n(2) Tối đa 15 lưu vực nhiều ngày qmax=0 nhất trên toàn chuỗi:")
display(full_zero.loc[full_zero["zero_days"].gt(0),
                      ["basin_id", "valid_days", "zero_days", "zero_pct", "longest_zero_run_days",
                       "first_zero_date", "last_zero_date"]].sort_values("zero_days", ascending=False).head(15))
print("\n(3) Tổng số ngày qmax=0 theo tháng trong 451 lưu vực, toàn chuỗi:")
display(full_zero[[f"zero_m{month:02d}" for month in range(1, 13)]].sum().rename("zero_days").to_frame().rename_axis("month"))
A1_QMAX_ZERO_PROFILE.to_csv(TABLE_DIR / "A1_P2_qmax_zero_by_basin_period.csv", index=False, encoding="utf-8-sig")
zero_summary.to_csv(TABLE_DIR / "A1_P2_qmax_zero_period_summary.csv", encoding="utf-8-sig")
print(f"Thời gian đếm: {time.perf_counter() - zero_started:.1f} giây. Đã lưu hai bảng vào {TABLE_DIR}.")
print("Các tháng là số đếm, chưa hiệu chỉnh theo số ngày quan trắc từng tháng. Chưa phân biệt cạn thật với lỗi.")

# %% [markdown]
# ### 2.14. Thiếu qmax theo kỳ và các lưu vực bắt đầu đo muộn
#
# Dùng bảng 2.10 và 2.13: tách thiếu dòng khỏi ô trống, so số ghi ở Mục 6.1, không đọc lại Q.

# %% Phần 2.14 - Thiếu qmax theo kỳ và các lưu vực bắt đầu đo muộn
# Dùng bảng 2.10 và 2.13: tách thiếu dòng khỏi ô trống, so số ghi ở Mục 6.1, không đọc lại Q.
# Q bằng 0 vẫn là có dữ liệu; danh sách bắt đầu muộn dựa vào ngày đầu có qmax hợp lệ.
REFERENCE_Q_MISSING = {"train": (40900, 3040), "validation": (2630, 2300), "test": (1545, 1545)}
q_files = A1_Q_FILE_SPANS[A1_Q_FILE_SPANS["selected"]].set_index("basin_id")
q_selected = A1_Q_COVERAGE[A1_Q_COVERAGE["selected"] & A1_Q_COVERAGE["variable"].eq("qmax")]
if q_files["missing_date_rows"].ne(0).any():
    raise ValueError("Có thiếu dòng bên trong tệp; không suy số dòng từ hai mốc ngày.")
if q_selected[["negative_cells", "non_numeric_nonblank", "infinite_cells"]].to_numpy().sum() != 0:
    raise ValueError("qmax có giá trị bất thường; chưa thể gọi mọi ô không hợp lệ là ô trống.")
missing_parts = []
for period, (start, end) in PERIODS.items():
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    valid = A1_QMAX_ZERO_PROFILE[A1_QMAX_ZERO_PROFILE["period"].eq(period)].set_index("basin_id")["valid_days"]
    part = q_files[["file_start", "file_end"]].join(valid, validate="one_to_one")
    if part["valid_days"].isna().any() or set(part.index) != selected_ids:
        raise ValueError(f"{period}: bảng đếm không đủ các ID được chọn.")
    first, last = part["file_start"].clip(lower=start), part["file_end"].clip(upper=end)
    part["expected_days"] = (end - start).days + 1
    part["rows_present"] = ((last - first).dt.days + 1).clip(lower=0)
    part["missing_rows"] = part["expected_days"] - part["rows_present"]
    part["blank_cells"] = part["rows_present"] - part["valid_days"]
    part["total_missing"] = part["expected_days"] - part["valid_days"]
    part["period"] = period
    missing_parts.append(part.reset_index())
A1_Q_MISSING_BY_BASIN = pd.concat(missing_parts, ignore_index=True)
count_columns = ["expected_days", "rows_present", "valid_days", "missing_rows", "blank_cells", "total_missing"]
A1_Q_MISSING_PERIODS = A1_Q_MISSING_BY_BASIN.groupby("period")[count_columns].sum().reindex(PERIODS)
A1_Q_MISSING_PERIODS["missing_pct"] = 100 * A1_Q_MISSING_PERIODS["total_missing"] / A1_Q_MISSING_PERIODS["expected_days"]
A1_Q_MISSING_PERIODS["matches_note"] = [
    (row["total_missing"], row["blank_cells"]) == REFERENCE_Q_MISSING[period]
    for period, row in A1_Q_MISSING_PERIODS.iterrows()]
print("(1) Thiếu qmax trong 451 lưu vực; missing_rows và blank_cells không chồng lấp:")
display(A1_Q_MISSING_PERIODS)
if not A1_Q_MISSING_PERIODS["matches_note"].all():
    print("CẢNH BÁO: khác Mục 6.1; giữ nguyên số đo này để kiểm, không ép khớp.")
train_counts = A1_QMAX_ZERO_PROFILE[A1_QMAX_ZERO_PROFILE["period"].eq("train")].set_index("basin_id")["valid_days"]
A1_LATE_GAUGES = q_selected.loc[q_selected["first_value_date"].gt(pd.Timestamp(PERIODS["train"][0])),
                               ["basin_id", "first_value_date"]].copy()
A1_LATE_GAUGES["train_valid_days"] = A1_LATE_GAUGES["basin_id"].map(train_counts)
print(f"\n(2) Có {len(A1_LATE_GAUGES)} lưu vực có qmax đầu tiên sau 01/01/2003; ghi chú cũ: 27:")
display(A1_LATE_GAUGES.sort_values(["first_value_date", "basin_id"]))
print("ID không có qmax hợp lệ trong train:", sorted(train_counts[train_counts.eq(0)].index, key=int))
if len(A1_LATE_GAUGES) != 27:
    print("CẢNH BÁO: số lưu vực bắt đầu muộn khác ghi chú; cần kiểm trước khi cập nhật tài liệu.")
for table_name, frame in [("Q_missing_by_basin", A1_Q_MISSING_BY_BASIN),
                          ("late_gauges", A1_LATE_GAUGES)]:
    frame.to_csv(TABLE_DIR / f"A1_P2_{table_name}.csv", index=False, encoding="utf-8-sig")
A1_Q_MISSING_PERIODS.to_csv(TABLE_DIR / "A1_P2_Q_missing_by_period.csv", encoding="utf-8-sig")
print("Đã lưu ba bảng vào:", TABLE_DIR)

# %% [markdown]
# ### 2.15. Hình dòng thời gian phủ dữ liệu theo nguồn
#
# Vẽ mốc có giá trị từ bảng đã đo; tách hai nhóm HRES và phân biệt khoảng ngoài với giao các mốc.

# %% Phần 2.15 - Hình dòng thời gian phủ dữ liệu theo nguồn
# Vẽ mốc có giá trị từ bảng đã đo; tách hai nhóm HRES và phân biệt khoảng ngoài với giao các mốc.
# Thanh không biểu diễn ô thiếu bên trong; lưu PNG 150 dpi và bảng ngày dùng vẽ vào WORK_DIR.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SOURCE_LABELS = {"ERA5L": "ERA5-Land", "ECMWF": "ECMWF HRES", "EOBS": "E-OBS",
                 "MSWEP": "MSWEP", "GLEAM": "GLEAM"}
source_order = {source: rank for rank, source in enumerate(SOURCE_PREFIXES)}
span_rows = []
for (source, first, last), group in A1_FORCING_COVERAGE.groupby(
        ["source", "first_value_date", "last_value_date"], dropna=False):
    variables = sorted(group["variable"].unique())
    suffix = ": " + ", ".join(variable.removeprefix("ECMWF_") + ("*" if variable in A1_FORECAST_VARIABLES else "")
                               for variable in variables) if source == "ECMWF" else ""
    span_rows.append({"label": SOURCE_LABELS[source] + suffix, "order": source_order[source],
                      "series_count": len(group), "ids": group["basin_id"].nunique(),
                      "outer_start": first, "outer_end": last, "inner_start": first, "inner_end": last})
q_span = A1_Q_COVERAGE[A1_Q_COVERAGE["selected"] & A1_Q_COVERAGE["variable"].eq("qmax")]
span_rows.append({"label": "Discharge qmax (selected gauges)", "order": len(source_order),
                  "series_count": len(q_span), "ids": q_span["basin_id"].nunique(),
                  "outer_start": q_span["first_value_date"].min(), "outer_end": q_span["last_value_date"].max(),
                  "inner_start": q_span["first_value_date"].max(), "inner_end": q_span["last_value_date"].min()})
A1_DATE_SPANS = pd.DataFrame(span_rows).sort_values(["order", "outer_end"]).reset_index(drop=True)
FIGURE_DIR = WORK_DIR / "Figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(13, 6.4))
for label, (start, end), color in zip(["Train", "Validation", "Test"], PERIODS.values(),
                                    ["#dcedc8", "#ffe0b2", "#ffcdd2"]):
    left, right = mdates.date2num(pd.Timestamp(start)), mdates.date2num(pd.Timestamp(end) + pd.Timedelta(days=1))
    ax.axvspan(left, right, color=color, alpha=0.55, zorder=0)
    ax.text((left + right) / 2, -0.85, label, ha="center", va="center", fontsize=8)
for y, row in A1_DATE_SPANS.iterrows():
    outer = mdates.date2num([row["outer_start"], row["outer_end"] + pd.Timedelta(days=1)])
    ax.hlines(y, *outer, color="#b9cee3", linewidth=13, zorder=2)
    if pd.notna(row["inner_start"]) and row["inner_start"] <= row["inner_end"]:
        inner = mdates.date2num([row["inner_start"], row["inner_end"] + pd.Timedelta(days=1)])
        ax.hlines(y, *inner, color="#245b8c", linewidth=8, zorder=3)
        if row["inner_start"] != row["outer_start"] or row["inner_end"] != row["outer_end"]:
            ax.vlines(inner, y - 0.12, y + 0.12, color="#173e61", linewidth=1, zorder=4)
            ax.text(inner[0], y + 0.28, row["inner_start"].strftime("%Y-%m-%d"),
                    fontsize=8, color="#173e61", ha="left", va="top")
            ax.text(inner[1], y + 0.28, row["inner_end"].strftime("%Y-%m-%d"),
                    fontsize=8, color="#173e61", ha="right", va="top")
    ax.text(outer[0], y - 0.2, row["outer_start"].strftime("%Y-%m-%d"), fontsize=7, ha="left")
    ax.text(outer[1], y - 0.2, row["outer_end"].strftime("%Y-%m-%d"), fontsize=7, ha="right")
ax.set_yticks(range(len(A1_DATE_SPANS)), A1_DATE_SPANS["label"])
ax.set_ylim(len(A1_DATE_SPANS) - 0.15, -1.2)
ax.set_xlim(mdates.date2num(A1_DATE_SPANS["outer_start"].min()) - 200,
            mdates.date2num(A1_DATE_SPANS["outer_end"].max()) + 200)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlabel("Calendar year")
ax.set_title("Date spans of meteorological inputs and daily maximum discharge", pad=12)
ax.grid(axis="x", alpha=0.2)
fig.legend(handles=[Line2D([0], [0], color="#b9cee3", lw=9, label="Outer span across series"),
                    Line2D([0], [0], color="#245b8c", lw=6,
                           label="Latest first date → earliest last date across series")],
           loc="lower center", bbox_to_anchor=(0.64, 0.105), ncol=2, frameon=False, fontsize=8)
fig.text(0.64, 0.025, "* Model forecast input (2 of 5 extend through 2019).\nMeteorology: 859 basins; discharge: 451 selected gauges.\nInternal gaps are not shown; bars do not imply continuous valid values.",
         ha="center", fontsize=8)
fig.subplots_adjust(left=0.31, right=0.98, top=0.88, bottom=0.27)
timeline_path = FIGURE_DIR / "A1_P2_source_date_spans.png"
fig.savefig(timeline_path, dpi=150, bbox_inches="tight")
plt.show()
A1_DATE_SPANS.drop(columns="order").to_csv(TABLE_DIR / "A1_P2_source_date_spans.csv", index=False, encoding="utf-8-sig")
print("Hình: màu nhạt là khoảng ngoài cùng; màu đậm là giao các mốc đầu-cuối. Không biểu diễn ô thiếu ở giữa.")
print("Riêng Q: thanh đậm bắt đầu muộn vì có trạm chỉ đo từ 2010; không có nghĩa mọi trạm thiếu trước 2010.")
print("Dấu * đánh dấu biến forecast mô hình dùng: chỉ t2m và d2m có số tới 2019; tp, ssrd và e dừng ở 2017.")
print("Đã lưu hình:", timeline_path)
print("Bảng mốc ngày dùng vẽ:")
display(A1_DATE_SPANS.drop(columns="order"))

# %% [markdown]
# ### 2.16. Cấu trúc các tệp thuộc tính trạm D
#
# Đọc tên cột và kiểm khóa ID của mọi CSV thuộc tính D; giữ toàn bộ giá trị ở dạng chữ.

# %% Phần 2.16 - Cấu trúc các tệp thuộc tính trạm D
# Đọc tên cột và kiểm khóa ID của mọi CSV thuộc tính D; giữ toàn bộ giá trị ở dạng chữ.
# Chưa nối hoặc diễn giải mã phân loại; dùng output để chọn trường kiểm 23 trạm không có A.
D_ATTRIBUTE_DIR = LAMAH_DIR / "D_gauges" / "1_attributes"
attribute_files = sorted(path for path in D_ATTRIBUTE_DIR.glob("*.csv") if not a1_is_metadata(path))
if not attribute_files:
    raise FileNotFoundError(f"Không thấy CSV thuộc tính trong {D_ATTRIBUTE_DIR}.")
A1_D_ATTRIBUTES = {}
attribute_inventory, attribute_schema = [], []
for path in attribute_files:
    frame = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    A1_D_ATTRIBUTES[path.name] = frame
    ids = frame["ID"].str.strip() if "ID" in frame else None
    attribute_inventory.append({
        "file": path.name, "rows": len(frame), "columns": len(frame.columns), "has_ID": ids is not None,
        "unique_IDs": ids.nunique() if ids is not None else None,
        "blank_ID_rows": int(ids.eq("").sum()) if ids is not None else None,
        "rows_in_duplicate_IDs": int(ids.duplicated(keep=False).sum()) if ids is not None else None,
    })
    for column in frame.columns:
        nonempty = frame.loc[frame[column].str.strip().ne(""), column]
        attribute_schema.append({"file": path.name, "column": column,
                                 "first_nonempty_raw": nonempty.iloc[0] if len(nonempty) else ""})
print("(1) Kiểm kê thuộc tính trạm; chỉ kiểm tên khóa ID và trùng/rỗng, chưa ghép dữ liệu:")
display(pd.DataFrame(attribute_inventory))
print("\n(2) Toàn bộ tên cột thật của từng tệp:")
for filename, frame in A1_D_ATTRIBUTES.items():
    print(f"\n{filename} ({len(frame.columns)} cột):")
    print(", ".join(frame.columns))
pd.DataFrame(attribute_inventory).to_csv(TABLE_DIR / "A1_P2_D_attribute_inventory.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(attribute_schema).to_csv(TABLE_DIR / "A1_P2_D_attribute_schema.csv", index=False, encoding="utf-8-sig")
print("\nĐã lưu bảng kiểm kê và schema vào:", TABLE_DIR)

# %% [markdown]
# ### 2.17. Đối chiếu tập ID, hồ sơ 23 trạm và tiêu chí u/l
#
# Nối bằng ID đã kiểm, đối chiếu hai bảng Hydro với A và xem thuộc tính thô của D không có A.

# %% Phần 2.17 - Đối chiếu tập ID, hồ sơ 23 trạm và tiêu chí u/l
# Nối bằng ID đã kiểm, đối chiếu hai bảng Hydro với A và xem thuộc tính thô của D không có A.
# Kiểm điều kiện u/l trong bài với danh sách 451; không tự gán nguyên nhân hay thay danh sách chọn.
A1_D_ATTRIBUTES_INDEXED, id_set_rows = {}, []
a_ids, d_ids = set(A1_DAILY_FILES["A"]), set(A1_DAILY_FILES["D"])
for filename, original in A1_D_ATTRIBUTES.items():
    frame = original.copy()
    ids = frame["ID"].str.strip()
    if not ids.str.fullmatch(r"\d+").all():
        raise ValueError(f"{filename}: ID không phải số nguyên; cần kiểm trước khi nối.")
    frame["ID"] = ids.map(lambda value: str(int(value)))
    if frame["ID"].duplicated().any():
        raise ValueError(f"{filename}: ID trùng sau chuẩn hóa; dừng, không tự gộp.")
    frame = frame.set_index("ID").rename_axis("basin_id")
    A1_D_ATTRIBUTES_INDEXED[filename] = frame
    actual_ids = set(frame.index)
    id_set_rows.append({"file": filename, "IDs": len(actual_ids),
                        "A_IDs_missing": len(a_ids - actual_ids), "extra_to_A": len(actual_ids - a_ids),
                        "D_IDs_missing": len(d_ids - actual_ids), "extra_to_D": len(actual_ids - d_ids),
                        "selected_IDs_missing": len(selected_ids - actual_ids)})
print("(1) So tập ID thực tế, không chỉ so số lượng:")
display(pd.DataFrame(id_set_rows))

gauge = A1_D_ATTRIBUTES_INDEXED["Gauge_attributes.csv"]
d_without_a = d_ids - a_ids
missing_metadata = d_without_a - set(gauge.index)
if missing_metadata:
    print("CẢNH BÁO: ID D không có A nhưng cũng thiếu thuộc tính trạm:", sorted(missing_metadata, key=int))
PROFILE_FIELDS = ["name", "river", "country", "area_gov", "area_ratio", "typimpact", "degimpact", "gaps_pre", "gaps_post"]
A1_UNDELINEATED_GAUGES = gauge.reindex(sorted(d_without_a, key=int))[PROFILE_FIELDS].copy()
A1_UNDELINEATED_GAUGES.insert(0, "selected", A1_UNDELINEATED_GAUGES.index.isin(selected_ids))
print(f"\n(2) {len(d_without_a)} trạm D không có A; area_gov: km²; gaps_pre/post: ‰, không phải %:")
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 45):
    display(A1_UNDELINEATED_GAUGES)
print("Mã degimpact trong nhóm này:", A1_UNDELINEATED_GAUGES["degimpact"].value_counts(dropna=False).to_dict())

degree = gauge["degimpact"].str.strip().str.lower()
low_impact_candidates = set(gauge.index[degree.isin(["u", "l"])]) & a_ids
cohort_check = pd.DataFrame([{
    "u_l_in_A": len(low_impact_candidates), "selected": len(selected_ids),
    "selected_not_u_l": len(selected_ids - low_impact_candidates),
    "u_l_not_selected": len(low_impact_candidates - selected_ids),
    "same_ID_set": low_impact_candidates == selected_ids,
}])
print("\n(3) Điều kiện không/ít tác động (u/l) trong bài so với basins_filtered.txt:")
display(cohort_check)
print("ID được chọn nhưng không thuộc u/l của A:", sorted(selected_ids - low_impact_candidates, key=int))
print("ID thuộc u/l của A nhưng không được chọn:", sorted(low_impact_candidates - selected_ids, key=int))
pd.DataFrame(id_set_rows).to_csv(TABLE_DIR / "A1_P2_attribute_ID_sets.csv", index=False, encoding="utf-8-sig")
A1_UNDELINEATED_GAUGES.to_csv(TABLE_DIR / "A1_P2_23_gauge_profiles.csv", encoding="utf-8-sig")
cohort_check.to_csv(TABLE_DIR / "A1_P2_u_l_cohort_check.csv", index=False, encoding="utf-8-sig")
print("Đã lưu ba bảng vào:", TABLE_DIR)

# %% [markdown]
# ### 2.18. Hồ sơ Q của các ID u/l không nằm trong danh sách BiasCast
#
# Chọn ID từ hiệu hai tập ở 2.17; xem thuộc tính, mốc Q và đếm qmax hợp lệ theo từng kỳ.

# %% Phần 2.18 - Hồ sơ Q của các ID u/l không nằm trong danh sách BiasCast
# Chọn ID từ hiệu hai tập ở 2.17; xem thuộc tính, mốc Q và đếm qmax hợp lệ theo từng kỳ.
# Chỉ đọc lại tệp của các ID chênh lệch; không đổi danh sách 451 hoặc suy nguyên nhân loại trạm.
A1_EXTRA_LOW_IMPACT_IDS = sorted(low_impact_candidates - selected_ids, key=int)
if not A1_EXTRA_LOW_IMPACT_IDS:
    raise ValueError("Hai tập ID không còn chênh lệch; kiểm output 2.17 trước khi chạy cell này.")
detail_fields = ["name", "river", "country", "degimpact", "typimpact", "obsbeg_day", "obsend",
                 "gaps_pre", "gaps_post"]
A1_EXTRA_LOW_IMPACT_GAUGES = gauge.loc[A1_EXTRA_LOW_IMPACT_IDS, detail_fields].copy()
print("ID u/l của A không nằm trong danh sách BiasCast:", A1_EXTRA_LOW_IMPACT_IDS)
print("(1) Thuộc tính thô; obsbeg_day/obsend là năm, obsend=0 là cờ của bộ dữ liệu;")
print("gaps_pre/post là ‰ của chuỗi giờ, không phải tỷ lệ thiếu qmax ngày theo kỳ:")
with pd.option_context("display.max_columns", None, "display.max_colwidth", 45):
    display(A1_EXTRA_LOW_IMPACT_GAUGES)

A1_EXTRA_LOW_IMPACT_Q = A1_Q_COVERAGE.loc[
    A1_Q_COVERAGE["basin_id"].isin(A1_EXTRA_LOW_IMPACT_IDS),
    ["basin_id", "variable", "first_value_date", "last_value_date", "valid_days", "blank_cells",
     "negative_cells", "non_numeric_nonblank", "infinite_cells"]].copy()
print("\n(2) Mốc Q và số ô trên toàn chuỗi từ lần quét 2.10; hợp lệ = hữu hạn và không âm:")
display(A1_EXTRA_LOW_IMPACT_Q)

extra_period_rows = []
for basin_id in A1_EXTRA_LOW_IMPACT_IDS:
    raw = pd.read_csv(A1_DAILY_FILES["D"][basin_id], sep=";", usecols=DATE_COLS + ["qmax"],
                      dtype=str, keep_default_na=False)
    dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES)))
    if dates.hasnans or dates.duplicated().any():
        raise ValueError(f"ID {basin_id}: ngày rỗng/trùng; dừng, không tự gộp.")
    raw = raw[["qmax"]].set_axis(dates)
    for period, (start, end) in PERIODS.items():
        calendar = pd.date_range(start, end)
        values_raw = raw.reindex(calendar)["qmax"]
        present = pd.Series(calendar.isin(dates), index=calendar)
        blank = present & values_raw.fillna("").str.strip().eq("")
        values = pd.to_numeric(values_raw, errors="coerce")
        valid = present & np.isfinite(values) & values.ge(0)
        invalid_nonblank = present & ~blank & ~valid
        extra_period_rows.append({
            "basin_id": basin_id, "period": period, "expected_days": len(calendar),
            "rows_present": int(present.sum()), "valid_days": int(valid.sum()),
            "missing_rows": int((~present).sum()), "blank_cells": int(blank.sum()),
            "invalid_nonblank": int(invalid_nonblank.sum()), "zero_days": int((valid & values.eq(0)).sum()),
            "total_missing": int((~valid).sum()), "missing_pct": 100 * (~valid).mean(),
            "first_valid_date": calendar[valid.to_numpy()].min(),
            "last_valid_date": calendar[valid.to_numpy()].max(),
        })
A1_EXTRA_LOW_IMPACT_PERIODS = pd.DataFrame(extra_period_rows)
if not (A1_EXTRA_LOW_IMPACT_PERIODS["missing_rows"] + A1_EXTRA_LOW_IMPACT_PERIODS["blank_cells"]
        + A1_EXTRA_LOW_IMPACT_PERIODS["invalid_nonblank"]).eq(A1_EXTRA_LOW_IMPACT_PERIODS["total_missing"]).all():
    raise ValueError("Các loại thiếu không khớp tổng; kiểm trước khi diễn giải.")
print("\n(3) qmax theo kỳ; ba loại thiếu không chồng lấp, số 0 vẫn hợp lệ:")
with pd.option_context("display.max_columns", None):
    display(A1_EXTRA_LOW_IMPACT_PERIODS)
for suffix, table in [("gauges", A1_EXTRA_LOW_IMPACT_GAUGES.reset_index()),
                      ("Q_coverage", A1_EXTRA_LOW_IMPACT_Q), ("Q_periods", A1_EXTRA_LOW_IMPACT_PERIODS)]:
    table.to_csv(TABLE_DIR / f"A1_P2_extra_low_impact_{suffix}.csv", index=False, encoding="utf-8-sig")
print("Đã lưu ba bảng vào:", TABLE_DIR)

# %% [markdown]
# ### 2.19. Năm bắt đầu quan trắc của toàn bộ nhóm u/l
#
# So obsbeg_day và ngày qmax hợp lệ đầu tiên giữa nhóm được chọn và nhóm u/l còn lại.

# %% Phần 2.19 - Năm bắt đầu quan trắc của toàn bộ nhóm u/l
# So obsbeg_day và ngày qmax hợp lệ đầu tiên giữa nhóm được chọn và nhóm u/l còn lại.
# Kiểm một ranh giới lấy từ nhóm được chọn; khớp tập ID cũng chỉ là mô tả hậu nghiệm.
A1_LOW_IMPACT_STARTS = gauge.loc[sorted(low_impact_candidates, key=int),
                               ["name", "degimpact", "obsbeg_day"]].copy()
A1_LOW_IMPACT_STARTS["selected"] = A1_LOW_IMPACT_STARTS.index.isin(selected_ids)
years = pd.to_numeric(A1_LOW_IMPACT_STARTS["obsbeg_day"].str.strip(), errors="coerce")
A1_LOW_IMPACT_STARTS["obsbeg_day_year"] = years.where(years.gt(0) & years.mod(1).eq(0)).astype("Int64")
first_q = A1_Q_COVERAGE.loc[A1_Q_COVERAGE["variable"].eq("qmax"),
                           ["basin_id", "first_value_date"]].set_index("basin_id")
A1_LOW_IMPACT_STARTS = A1_LOW_IMPACT_STARTS.join(first_q, validate="one_to_one")
A1_LOW_IMPACT_STARTS["qmax_first_year"] = A1_LOW_IMPACT_STARTS["first_value_date"].dt.year.astype("Int64")
A1_LOW_IMPACT_STARTS["start_year_agrees"] = (
    A1_LOW_IMPACT_STARTS["obsbeg_day_year"].eq(A1_LOW_IMPACT_STARTS["qmax_first_year"])
    .where(A1_LOW_IMPACT_STARTS["first_value_date"].notna()))
A1_LOW_IMPACT_START_SUMMARY = A1_LOW_IMPACT_STARTS.groupby("selected").agg(
    basins=("name", "size"), unknown_obsbeg_year=("obsbeg_day_year", lambda x: int(x.isna().sum())),
    earliest_obsbeg_year=("obsbeg_day_year", "min"), latest_obsbeg_year=("obsbeg_day_year", "max"),
    unknown_qmax_start=("first_value_date", lambda x: int(x.isna().sum())),
    earliest_qmax_start=("first_value_date", "min"), latest_qmax_start=("first_value_date", "max"))
print("(1) Toàn bộ nhóm u/l của A; selected=True là danh sách BiasCast:")
with pd.option_context("display.max_columns", None):
    display(A1_LOW_IMPACT_START_SUMMARY)

latest_selected_year = A1_LOW_IMPACT_STARTS.loc[A1_LOW_IMPACT_STARTS["selected"], "obsbeg_day_year"].max()
if pd.isna(latest_selected_year):
    raise ValueError("Không có năm obsbeg_day hợp lệ trong nhóm được chọn; chưa kiểm được ranh giới.")
latest_selected_year = int(latest_selected_year)
boundary = A1_LOW_IMPACT_STARTS["obsbeg_day_year"].eq(latest_selected_year).fillna(False)
boundary = ~A1_LOW_IMPACT_STARTS["selected"] | (A1_LOW_IMPACT_STARTS["selected"] & boundary)
print("\n(2) Các ID được chọn có obsbeg_day muộn nhất và mọi ID u/l không được chọn:")
with pd.option_context("display.max_columns", None, "display.max_rows", None):
    display(A1_LOW_IMPACT_STARTS.loc[boundary].sort_values(["obsbeg_day_year", "first_value_date"]))

passes = A1_LOW_IMPACT_STARTS["obsbeg_day_year"].le(latest_selected_year).fillna(False)
described_ids = set(A1_LOW_IMPACT_STARTS.index[passes])
unexplained_selected = selected_ids - described_ids
extra_described = described_ids - selected_ids
A1_LOW_IMPACT_START_CHECK = pd.DataFrame([{
    "observed_cutoff_year": latest_selected_year, "IDs_at_or_before_cutoff": len(described_ids),
    "selected_IDs_not_reproduced": len(unexplained_selected), "extra_IDs": len(extra_described),
    "same_ID_set": described_ids == selected_ids,
}])
print("\n(3) Kiểm mô tả: u/l của A và obsbeg_day <= năm muộn nhất trong nhóm được chọn:")
display(A1_LOW_IMPACT_START_CHECK)
print("ID được chọn nhưng mô tả không khớp:", sorted(unexplained_selected, key=int))
print("ID ngoài danh sách nhưng mô tả vẫn nhận:", sorted(extra_described, key=int))
print("ID thiếu/không đọc được obsbeg_day:", A1_LOW_IMPACT_STARTS.index[
    A1_LOW_IMPACT_STARTS["obsbeg_day_year"].isna()].tolist())
start_year_differences = A1_LOW_IMPACT_STARTS.index[
    A1_LOW_IMPACT_STARTS["start_year_agrees"].eq(False).fillna(False)].tolist()
print("Số ID có năm obsbeg_day khác năm qmax đầu tiên:", len(start_year_differences),
      "| tối đa 10 ID:", start_year_differences[:10])
print("Khác năm chưa phải lỗi: chuỗi Q đang xét bị giới hạn từ 1981 và có thể có ô thiếu đầu chuỗi.")
for suffix, table in [("all_IDs", A1_LOW_IMPACT_STARTS.reset_index()),
                      ("summary", A1_LOW_IMPACT_START_SUMMARY.reset_index()), ("check", A1_LOW_IMPACT_START_CHECK)]:
    table.to_csv(TABLE_DIR / f"A1_P2_low_impact_start_{suffix}.csv", index=False, encoding="utf-8-sig")
print("Đã lưu ba bảng vào:", TABLE_DIR)

# %% [markdown]
# ### 2.20. Đối chiếu mốc ngày/giờ của các trạm lệch năm bắt đầu Q
#
# Tự lấy các ID lệch năm ở 2.19, in quốc gia và obsbeg_hr, rồi so mốc đầu của cả ba biến Q.

# %% Phần 2.20 - Đối chiếu mốc ngày/giờ của các trạm lệch năm bắt đầu Q
# Tự lấy các ID lệch năm ở 2.19, in quốc gia và obsbeg_hr, rồi so mốc đầu của cả ba biến Q.
# Dùng bảng đã quét; khớp năm chỉ hỗ trợ giả thuyết, chưa thay thế đối chiếu chuỗi giờ gốc.
probe_mask = A1_LOW_IMPACT_STARTS["start_year_agrees"].eq(False).fillna(False)
A1_START_YEAR_PROBE_IDS = sorted(A1_LOW_IMPACT_STARTS.index[probe_mask], key=int)
if not A1_START_YEAR_PROBE_IDS:
    raise ValueError("Không có ID lệch năm trong nhóm u/l; kiểm output 2.19 trước khi chạy cell này.")
A1_START_YEAR_PROBE_GAUGES = gauge.loc[A1_START_YEAR_PROBE_IDS,
    ["name", "river", "country", "obsbeg_day", "obsbeg_hr", "obsend"]].copy()
A1_START_YEAR_PROBE_GAUGES.insert(0, "selected", A1_START_YEAR_PROBE_GAUGES.index.isin(selected_ids))
print("ID lệch năm trong nhóm u/l của A:", A1_START_YEAR_PROBE_IDS)
print("(1) Thuộc tính thô; obsbeg_day và obsbeg_hr là năm, obsend=0 là cờ tại thời điểm lập dữ liệu:")
with pd.option_context("display.max_columns", None, "display.max_colwidth", 40):
    display(A1_START_YEAR_PROBE_GAUGES)

first_dates = (A1_Q_COVERAGE[A1_Q_COVERAGE["basin_id"].isin(A1_START_YEAR_PROBE_IDS)]
               .pivot(index="basin_id", columns="variable", values="first_value_date")
               .reindex(index=A1_START_YEAR_PROBE_IDS, columns=Q_COLUMNS))
first_dates.columns.name = None
file_starts = A1_Q_FILE_SPANS.set_index("basin_id")[["file_start"]]
A1_START_YEAR_PROBE_DATES = file_starts.loc[A1_START_YEAR_PROBE_IDS].join(
    first_dates.add_prefix("first_valid_"), validate="one_to_one")
print("\n(2) Ngày đầu có dòng và ngày Q hợp lệ đầu tiên (hữu hạn, không âm) trong Extended LamaH-CE:")
with pd.option_context("display.max_columns", None):
    display(A1_START_YEAR_PROBE_DATES)

A1_START_YEAR_PROBE_CHECK = pd.DataFrame(index=first_dates.index)
for field in ["obsbeg_day", "obsbeg_hr"]:
    years = pd.to_numeric(A1_START_YEAR_PROBE_GAUGES[field].str.strip(), errors="coerce")
    A1_START_YEAR_PROBE_CHECK[field] = years.where(years.gt(0) & years.mod(1).eq(0)).astype("Int64")
A1_START_YEAR_PROBE_CHECK["qmax_first_year"] = first_dates["qmax"].dt.year.astype("Int64")
for field in ["obsbeg_day", "obsbeg_hr"]:
    A1_START_YEAR_PROBE_CHECK[f"qmax_matches_{field}"] = (
        A1_START_YEAR_PROBE_CHECK["qmax_first_year"].eq(A1_START_YEAR_PROBE_CHECK[field]))
A1_START_YEAR_PROBE_CHECK["Q3_same_first"] = (
    first_dates.notna().all(axis=1) & first_dates.eq(first_dates["qmax"], axis=0).all(axis=1))
print("\n(3) So theo năm; <NA> là không đủ mốc để so, Q3_same_first so chính xác ngày của ba biến:")
with pd.option_context("display.max_columns", None):
    display(A1_START_YEAR_PROBE_CHECK)
print("ID khớp năm obsbeg_hr:", A1_START_YEAR_PROBE_CHECK.index[
    A1_START_YEAR_PROBE_CHECK["qmax_matches_obsbeg_hr"].fillna(False)].tolist())
print("Chỉ các trạm được in mới được kiểm ở cell này; chưa suy kết quả cho toàn bộ D.")
for suffix, table in [("gauges", A1_START_YEAR_PROBE_GAUGES), ("dates", A1_START_YEAR_PROBE_DATES),
                      ("check", A1_START_YEAR_PROBE_CHECK)]:
    table.to_csv(TABLE_DIR / f"A1_P2_start_year_probe_{suffix}.csv", encoding="utf-8-sig")
print("Đã lưu ba bảng vào:", TABLE_DIR)

# %% [markdown]
# ### 2.21. Kiểm trực tiếp các ô Q năm 1981 của trạm 94
#
# Đọc tệp thô của 94, phân loại từng ô trong năm 1981 và in mẫu quanh ngày Q bắt đầu có giá trị.

# %% Phần 2.21 - Kiểm trực tiếp các ô Q năm 1981 của trạm 94
# Đọc tệp thô của 94, phân loại từng ô trong năm 1981 và in mẫu quanh ngày Q bắt đầu có giá trị.
# Phân biệt dòng thiếu, ô rỗng, số âm và ký hiệu khác; không điền hoặc sửa dữ liệu.
Q_START_PROBE_ID, Q_START_PROBE_YEAR = "94", 1981
probe_raw = pd.read_csv(A1_DAILY_FILES["D"][Q_START_PROBE_ID], sep=";", usecols=DATE_COLS + Q_COLUMNS,
                        dtype=str, keep_default_na=False, encoding="utf-8-sig")
probe_dates = pd.DatetimeIndex(pd.to_datetime(probe_raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES)))
if probe_dates.hasnans or probe_dates.duplicated().any():
    raise ValueError("Trạm 94 có ngày rỗng/trùng; cần kiểm trước khi đối chiếu.")
probe_q = probe_raw[Q_COLUMNS].set_axis(probe_dates.rename("date")).sort_index()
probe_calendar = pd.date_range(f"{Q_START_PROBE_YEAR}-01-01", f"{Q_START_PROBE_YEAR}-12-31", name="date")
A1_Q94_YEAR_RAW = probe_q.reindex(probe_calendar)
row_present = pd.Series(probe_calendar.isin(probe_dates), index=probe_calendar)
probe_count_rows = []
for column in Q_COLUMNS:
    text = A1_Q94_YEAR_RAW[column].fillna("").str.strip()
    values = pd.to_numeric(text, errors="coerce")
    blank = row_present & text.eq("")
    valid = row_present & np.isfinite(values) & values.ge(0)
    negative = row_present & np.isfinite(values) & values.lt(0)
    nonnumeric = row_present & text.ne("") & values.isna()
    infinite = row_present & np.isinf(values)
    if int(blank.sum() + valid.sum() + negative.sum() + nonnumeric.sum() + infinite.sum()) != int(row_present.sum()):
        raise ValueError(f"{column}: các loại ô không khớp số dòng có mặt.")
    probe_count_rows.append({
        "variable": column, "expected_days": len(probe_calendar), "rows_present": int(row_present.sum()),
        "missing_rows": int((~row_present).sum()), "blank_cells": int(blank.sum()),
        "valid_days": int(valid.sum()), "negative_cells": int(negative.sum()),
        "minus999_cells": int(values.eq(-999).sum()), "non_numeric_nonblank": int(nonnumeric.sum()),
        "infinite_cells": int(infinite.sum()),
    })
A1_Q94_YEAR_COUNTS = pd.DataFrame(probe_count_rows)
print(f"ID {Q_START_PROBE_ID} | năm {Q_START_PROBE_YEAR} | tệp:", A1_DAILY_FILES["D"][Q_START_PROBE_ID])
print("(1) Đếm từng loại ô; minus999_cells nằm trong negative_cells, không cộng hai cột:")
with pd.option_context("display.max_columns", None):
    display(A1_Q94_YEAR_COUNTS)

first_qmax = A1_Q_COVERAGE.loc[A1_Q_COVERAGE["basin_id"].eq(Q_START_PROBE_ID)
                              & A1_Q_COVERAGE["variable"].eq("qmax"), "first_value_date"].iloc[0]
if pd.isna(first_qmax):
    raise ValueError("Không có qmax hợp lệ để chọn mẫu ranh giới.")
sample_dates = (probe_calendar[:3].append(probe_calendar[-3:])
                .append(pd.date_range(first_qmax, periods=3)).unique().sort_values())
A1_Q94_START_SAMPLE = probe_q.reindex(sample_dates).rename_axis("date")
A1_Q94_START_SAMPLE.insert(0, "row_present", sample_dates.isin(probe_dates))
sample_display = A1_Q94_START_SAMPLE.copy()
for column in Q_COLUMNS:
    sample_display[column] = sample_display[column].map(
        lambda value: "<no row>" if pd.isna(value) else "<empty>" if value == ""
        else "<whitespace>" if not value.strip() else value)
print("\n(2) Mẫu thô ở đầu/cuối năm kiểm và ba ngày từ mốc qmax đầu tiên:")
print("<empty>: ô rỗng; <whitespace>: chỉ có khoảng trắng; <no row>: không có dòng.")
display(sample_display)
A1_Q94_YEAR_COUNTS.to_csv(TABLE_DIR / "A1_P2_Q94_1981_counts.csv", index=False, encoding="utf-8-sig")
A1_Q94_YEAR_RAW.assign(row_present=row_present).to_csv(TABLE_DIR / "A1_P2_Q94_1981_raw.csv", encoding="utf-8-sig")
A1_Q94_START_SAMPLE.to_csv(TABLE_DIR / "A1_P2_Q94_start_sample_raw.csv", encoding="utf-8-sig")
print("Đã lưu ba bảng; CSV raw giữ nguyên chuỗi ký tự, không dùng ký hiệu hiển thị:", TABLE_DIR)

# %% [markdown]
# ## Phần 3. Thuộc tính tĩnh, diện tích và bản đồ
#
# Kiểm khóa ghép, thuộc tính cấu hình yêu cầu, diện tích và hình học.

# %% [markdown]
# ### 3.1. Kiểm kê thuộc tính lưu vực A và khóa ID
#
# Đọc mọi CSV thuộc tính A dạng chuỗi, kiểm khóa ID và in tên cột thật trước khi chọn thuộc tính mô hình.

# %% Phần 3.1 - Kiểm kê thuộc tính lưu vực A và khóa ID
# Đọc mọi CSV thuộc tính A dạng chuỗi, kiểm khóa ID và in tên cột thật trước khi chọn thuộc tính mô hình.
# Lưu schema cùng ô rỗng/-999 để phân tích tiếp; chưa ghép bảng hoặc tự coi -999 là thiếu.
A_ATTRIBUTE_DIR = LAMAH_DIR / "A_basins_total_upstrm" / "1_attributes"
a_attribute_files = sorted(path for path in A_ATTRIBUTE_DIR.glob("*.csv") if not a1_is_metadata(path))
if not a_attribute_files:
    raise FileNotFoundError(f"Không thấy CSV thuộc tính lưu vực trong {A_ATTRIBUTE_DIR}.")
A1_A_ATTRIBUTES_RAW, A1_A_ATTRIBUTES_INDEXED = {}, {}
a_attribute_inventory, a_attribute_schema = [], []
for path in a_attribute_files:
    frame = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    A1_A_ATTRIBUTES_RAW[path.name] = frame
    row = {"file": path.name, "rows": len(frame), "columns": len(frame.columns), "has_ID": "ID" in frame}
    if "ID" in frame:
        ids = frame["ID"].str.strip()
        numeric_id = ids.str.fullmatch(r"\d+")
        normalized = ids.map(lambda value: str(int(value)) if re.fullmatch(r"\d+", value) else pd.NA)
        valid_ids = set(normalized.dropna())
        duplicate_rows = int(normalized.dropna().duplicated(keep=False).sum())
        row.update({"unique_IDs": len(valid_ids), "blank_ID_rows": int(ids.eq("").sum()),
                    "noninteger_ID_rows": int((ids.ne("") & ~numeric_id).sum()),
                    "rows_in_duplicate_IDs": duplicate_rows, "A_IDs_missing": len(a_ids - valid_ids),
                    "extra_to_A": len(valid_ids - a_ids), "selected_IDs_missing": len(selected_ids - valid_ids)})
        if numeric_id.all() and duplicate_rows == 0:
            A1_A_ATTRIBUTES_INDEXED[path.name] = frame.assign(ID=normalized).set_index("ID").rename_axis("basin_id")
        else:
            print(f"CẢNH BÁO {path.name}: khóa ID chưa hợp lệ; giữ bảng thô, không tạo bảng đánh chỉ mục.")
    else:
        print(f"CẢNH BÁO {path.name}: không có cột ID; không đoán cột khóa và không ghép.")
    a_attribute_inventory.append(row)
    for column in frame.columns:
        stripped = frame[column].str.strip()
        nonblank = stripped.ne("")
        numeric = pd.to_numeric(stripped, errors="coerce")
        a_attribute_schema.append({
            "file": path.name, "column": column, "blank_cells": int((~nonblank).sum()),
            "nonblank_cells": int(nonblank.sum()), "numeric_nonblank": int((nonblank & numeric.notna()).sum()),
            "minus999_cells": int(numeric.eq(-999).sum()),
            "raw_examples": " | ".join(frame.loc[nonblank, column].drop_duplicates().head(3)),
        })
A1_A_ATTRIBUTE_INVENTORY = pd.DataFrame(a_attribute_inventory)
A1_A_ATTRIBUTE_SCHEMA = pd.DataFrame(a_attribute_schema)
print("(1) Kiểm kê mọi CSV thuộc tính A; ID được chuẩn hóa về chuỗi số nguyên trước khi so tập:")
with pd.option_context("display.max_columns", None):
    display(A1_A_ATTRIBUTE_INVENTORY)
print("\n(2) Toàn bộ tên cột thật, không mặc định mọi CSV là đầu vào mô hình:")
for filename, frame in A1_A_ATTRIBUTES_RAW.items():
    print(f"\n{filename} ({len(frame.columns)} cột, gồm cả ID nếu có):")
    print(", ".join(frame.columns))
A1_A_ATTRIBUTE_INVENTORY.to_csv(TABLE_DIR / "A1_P3_A_attribute_inventory.csv", index=False, encoding="utf-8-sig")
A1_A_ATTRIBUTE_SCHEMA.to_csv(TABLE_DIR / "A1_P3_A_attribute_schema.csv", index=False, encoding="utf-8-sig")
print("\nĐã lưu kiểm kê và schema chi tiết, gồm ví dụ thô và số ô rỗng/-999:", TABLE_DIR)
print("Ô '-' hoặc -999 chưa tự đổi thành thiếu; chưa suy đơn vị và chưa chọn danh sách 33 thuộc tính.")

# %% [markdown]
# ### 3.2. Đọc thuộc tính tĩnh từ cấu hình thí nghiệm và đối chiếu nguồn
#
# Đọc static_attributes của mọi config.yml, so cả tập tên lẫn thứ tự, rồi xác định cột nằm ở bảng A hay D.

# %% Phần 3.2 - Đọc thuộc tính tĩnh từ cấu hình thí nghiệm và đối chiếu nguồn
# Đọc static_attributes của mọi config.yml, so cả tập tên lẫn thứ tự, rồi xác định cột nằm ở bảng A hay D.
# Không ghi cứng 33 tên từ bài; chỉ ánh xạ và xem ví dụ thô, chưa ghép hay chuẩn hóa đầu vào.
try:
    import yaml
    a1_load_yaml = yaml.safe_load
except ImportError:
    from ruamel.yaml import YAML
    a1_load_yaml = YAML(typ="safe").load

config_paths = sorted(path for path in EXP_DIR.rglob("config.yml") if not a1_is_metadata(path))
if not config_paths:
    raise FileNotFoundError(f"Không tìm thấy config.yml dưới {EXP_DIR}.")
if "Catchment_attributes.csv" not in A1_A_ATTRIBUTES_INDEXED:
    raise ValueError("Chưa có bảng Catchment_attributes.csv với ID hợp lệ; kiểm output 3.1.")
catchment = A1_A_ATTRIBUTES_INDEXED["Catchment_attributes.csv"]
gauge = A1_D_ATTRIBUTES_INDEXED["Gauge_attributes.csv"]
ordered_patterns, field_sets, config_rows, configured_sets = {}, {}, [], []
for path in config_paths:
    cfg = a1_load_yaml(path.read_text(encoding="utf-8-sig"))
    attrs = cfg.get("static_attributes") if isinstance(cfg, dict) else None
    if not isinstance(attrs, list) or not attrs or not all(isinstance(name, str) and name.strip() for name in attrs):
        raise ValueError(f"{path.relative_to(EXP_DIR)}: static_attributes không phải danh sách tên hợp lệ.")
    if any(name != name.strip() for name in attrs) or len(attrs) != len(set(attrs)):
        raise ValueError(f"{path.relative_to(EXP_DIR)}: tên thuộc tính có khoảng trắng thừa/trùng; không tự sửa.")
    ordered, field_set = tuple(attrs), tuple(sorted(attrs))
    if ordered not in ordered_patterns:
        ordered_patterns[ordered] = f"O{len(ordered_patterns) + 1}"
    if field_set not in field_sets:
        field_sets[field_set] = f"S{len(field_sets) + 1}"
    configured_sets.append(set(attrs))
    config_rows.append({"config": path.relative_to(EXP_DIR).as_posix(), "attributes": len(attrs),
                        "ordered_pattern": ordered_patterns[ordered], "field_set": field_sets[field_set],
                        "ordered_names": " | ".join(attrs)})
A1_STATIC_CONFIGS = pd.DataFrame(config_rows)
A1_STATIC_CONFIG_SUMMARY = A1_STATIC_CONFIGS.groupby(
    ["field_set", "ordered_pattern", "attributes"], as_index=False).agg(
    configs=("config", "size"), example_config=("config", "first"))
print("(1) Nhóm cấu hình theo tập tên (S) và thứ tự (O) của static_attributes:")
with pd.option_context("display.max_colwidth", 110, "display.max_columns", None):
    display(A1_STATIC_CONFIG_SUMMARY)

attribute_union = list(dict.fromkeys(name for ordered in ordered_patterns for name in ordered))
attribute_map_rows = []
for name in attribute_union:
    in_a, in_d = name in catchment.columns, name in gauge.columns
    source = "A:Catchment_attributes" if in_a and not in_d else "D:Gauge_attributes" if in_d and not in_a else (
        "AMBIGUOUS:A+D" if in_a and in_d else "MISSING")
    examples = ""
    if in_a != in_d:
        values = (catchment if in_a else gauge).loc[sorted(selected_ids, key=int), name]
        examples = " | ".join(values[values.str.strip().ne("")].drop_duplicates().head(3))
    attribute_map_rows.append({"attribute": name, "source": source,
                               "configs_using": sum(name in fields for fields in configured_sets),
                               "raw_examples_selected": examples})
A1_MODEL_STATIC_MAP = pd.DataFrame(attribute_map_rows)
print("\n(2) Ánh xạ các tên được cấu hình yêu cầu; thứ tự hiển thị là thứ tự gặp đầu tiên trong các cấu hình:")
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 60):
    display(A1_MODEL_STATIC_MAP)
unused_catchment = [name for name in catchment.columns if name not in attribute_union]
print("\n(3) Số thuộc tính Catchment thực tế (không gồm ID):", len(catchment.columns))
print("Số tên khác nhau được mọi cấu hình cộng lại yêu cầu:", len(attribute_union))
print("Số thuộc tính Catchment không được cấu hình nào yêu cầu:", len(unused_catchment))
print("Danh sách Catchment không được yêu cầu:", unused_catchment)
print("Số tập tên khác nhau:", len(field_sets), "| số thứ tự khác nhau:", len(ordered_patterns))
if A1_MODEL_STATIC_MAP["source"].isin(["MISSING", "AMBIGUOUS:A+D"]).any():
    print("CẢNH BÁO: có tên thiếu hoặc trùng nguồn; chưa được phép tự chọn cột để ghép.")
for suffix, table in [("configs", A1_STATIC_CONFIGS), ("config_summary", A1_STATIC_CONFIG_SUMMARY),
                      ("attribute_sources", A1_MODEL_STATIC_MAP)]:
    table.to_csv(TABLE_DIR / f"A1_P3_static_{suffix}.csv", index=False, encoding="utf-8-sig")
print("Đã lưu ba bảng vào:", TABLE_DIR)

# %% [markdown]
# ### 3.3. Chất lượng và khoảng giá trị của thuộc tính tĩnh
#
# Kiểm toàn bộ 62 trường và riêng các trường mô hình dùng ở 859/451; tách mã phân loại khỏi đại lượng số.

# %% Phần 3.3 - Chất lượng và khoảng giá trị của thuộc tính tĩnh
# Kiểm toàn bộ 62 trường và riêng các trường mô hình dùng ở 859/451; tách mã phân loại khỏi đại lượng số.
# Giữ nguyên bảng gốc; -999 tạm không dùng tính phân vị, các số âm khác và GVF ngoài 0–1 vẫn giữ.
if len(ordered_patterns) != 1:
    raise ValueError("Có nhiều thứ tự thuộc tính; cần chọn phạm vi cấu hình trước khi kiểm chung.")
A1_MODEL_STATIC_FIELDS = list(next(iter(ordered_patterns)))
if not set(A1_MODEL_STATIC_FIELDS).issubset(catchment.columns):
    raise ValueError("Có thuộc tính mô hình ngoài Catchment; cần xử lý nguồn trước khi kiểm cell này.")
TEXT_CATEGORIES = {"gc_dom", "hi_prec_ti", "lo_prec_ti"}
CATEGORY_FIELDS = TEXT_CATEGORIES | {"lc_dom"}
attribute_quality_rows = []
for scope, ids in [("all_A", sorted(a_ids, key=int)), ("selected", sorted(selected_ids, key=int))]:
    scoped = catchment.loc[ids]
    for column in scoped.columns:
        text_values = scoped[column].str.strip()
        values = pd.to_numeric(text_values, errors="coerce")
        blank = text_values.eq("")
        nonnumeric = text_values.ne("") & values.isna()
        sentinel = values.eq(-999)
        finite = values[np.isfinite(values) & ~sentinel]
        quantiles = finite.quantile([0.01, 0.5, 0.99])
        attribute_quality_rows.append({
            "scope": scope, "basins": len(scoped), "attribute": column,
            "used_model": column in A1_MODEL_STATIC_FIELDS, "categorical": column in CATEGORY_FIELDS,
            "blank_cells": int(blank.sum()), "minus999_cells": int(sentinel.sum()),
            "infinite_cells": int(np.isinf(values).sum()), "non_numeric_nonblank": int(nonnumeric.sum()),
            "unexpected_nonnumeric": int(nonnumeric.sum()) if column not in TEXT_CATEGORIES else 0,
            "unique_nonblank_raw": int(text_values[~blank].nunique()), "numeric_cells_for_stats": len(finite),
            "min": finite.min(), "p01": quantiles.loc[0.01], "median": quantiles.loc[0.5],
            "p99": quantiles.loc[0.99], "max": finite.max(), "std_sample": finite.std(ddof=1),
        })
A1_STATIC_ATTRIBUTE_QUALITY = pd.DataFrame(attribute_quality_rows)
quality_counts = ["blank_cells", "minus999_cells", "infinite_cells", "unexpected_nonnumeric"]
summary_rows = []
for scope, scoped in A1_STATIC_ATTRIBUTE_QUALITY.groupby("scope", sort=False):
    for label, part in [("all_catchment_fields", scoped), ("model_fields", scoped[scoped["used_model"]])]:
        summary_rows.append({"scope": scope, "basins": int(part["basins"].iloc[0]), "fields": label,
                             "attributes": len(part), **part[quality_counts].sum().astype(int).to_dict(),
                             "numeric_std_zero": int(part["std_sample"].eq(0).sum())})
A1_STATIC_QUALITY_SUMMARY = pd.DataFrame(summary_rows)
print("(1) Ô cần xem xét và trường có độ lệch chuẩn số bằng 0; các loại đếm không chồng lấp:")
print("gc_dom, hi_prec_ti, lo_prec_ti là trường phân loại chữ; không gọi mã chữ của chúng là lỗi chuyển số.")
with pd.option_context("display.max_columns", None):
    display(A1_STATIC_QUALITY_SUMMARY)

A1_STATIC_MODEL_RANGES = A1_STATIC_ATTRIBUTE_QUALITY.loc[
    A1_STATIC_ATTRIBUTE_QUALITY["scope"].eq("selected") & A1_STATIC_ATTRIBUTE_QUALITY["used_model"]
    & ~A1_STATIC_ATTRIBUTE_QUALITY["categorical"],
    ["attribute", "numeric_cells_for_stats", "unique_nonblank_raw", "min", "p01", "median", "p99", "max", "std_sample"]]
print("\n(2) Khoảng giá trị các thuộc tính số mô hình dùng trong 451; lc_dom in riêng ở (4):")
print("Phân vị tạm bỏ -999/vô hạn/ô không phải số; không bỏ mọi số âm và không ép GVF về 0–1.")
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    display(A1_STATIC_MODEL_RANGES)
A1_STATIC_QUALITY_ISSUES = A1_STATIC_ATTRIBUTE_QUALITY.loc[
    A1_STATIC_ATTRIBUTE_QUALITY[quality_counts].sum(axis=1).gt(0),
    ["scope", "attribute", "used_model"] + quality_counts]
print("\n(3) Thuộc tính có ô cần xem xét, tối đa 30 dòng; toàn bộ lưu trong CSV:")
display(A1_STATIC_QUALITY_ISSUES.head(30))
A1_LC_DOM_COUNTS = pd.DataFrame({"all_A": catchment["lc_dom"].str.strip().value_counts(),
    "selected": catchment.loc[sorted(selected_ids, key=int), "lc_dom"].str.strip().value_counts()})
A1_LC_DOM_COUNTS = A1_LC_DOM_COUNTS.fillna(0).astype(int).sort_index().rename_axis("lc_dom_raw")
print("\n(4) Tần số mã lc_dom; mã là nhãn CORINE, không phải đại lượng có thứ tự độ lớn vật lý:")
display(A1_LC_DOM_COUNTS.rename(index={"": "<empty>"}))
for suffix, table in [("attribute_quality", A1_STATIC_ATTRIBUTE_QUALITY), ("quality_summary", A1_STATIC_QUALITY_SUMMARY),
                      ("model_ranges", A1_STATIC_MODEL_RANGES), ("quality_issues", A1_STATIC_QUALITY_ISSUES)]:
    table.to_csv(TABLE_DIR / f"A1_P3_static_{suffix}.csv", index=False, encoding="utf-8-sig")
A1_LC_DOM_COUNTS.to_csv(TABLE_DIR / "A1_P3_lc_dom_counts.csv", encoding="utf-8-sig")
print("Đã lưu năm bảng vào:", TABLE_DIR)
print("Không điền thiếu, đổi đơn vị, cắt ngưỡng hoặc mã hóa lại thuộc tính ở cell này.")

# %% [markdown]
# ### 3.4. Ghép theo ID và đối chiếu diện tích, hình dạng lưu vực
#
# Ghép diện tích A/trạm bằng ID; kiểm mẫu số chuẩn hóa Q và đối chiếu area_ratio, elon_ratio theo metadata.

# %% Phần 3.4 - Ghép theo ID và đối chiếu diện tích, hình dạng lưu vực
# Ghép diện tích A/trạm bằng ID; kiểm mẫu số chuẩn hóa Q và đối chiếu area_ratio, elon_ratio theo metadata.
# Chỉ chẩn đoán bằng các số đã lưu; không chọn diện tích thay thế, sửa hình học hoặc lọc theo ngưỡng mới.
area_fields = ["area_calc", "mvert_dist", "elon_ratio"]
gauge_fields = ["name", "country", "area_gov", "area_ratio"]
A1_AREA_SHAPE_AUDIT = catchment[area_fields].rename(columns=lambda name: name + "_raw").join(
    gauge[gauge_fields].rename(columns={"area_gov": "area_gov_raw", "area_ratio": "area_ratio_raw"}),
    how="left", validate="one_to_one")
A1_AREA_SHAPE_AUDIT["selected"] = A1_AREA_SHAPE_AUDIT.index.isin(selected_ids)
for column in area_fields + ["area_gov", "area_ratio"]:
    A1_AREA_SHAPE_AUDIT[column] = pd.to_numeric(A1_AREA_SHAPE_AUDIT[column + "_raw"], errors="coerce")
audit = A1_AREA_SHAPE_AUDIT
calc_ok = np.isfinite(audit["area_calc"]) & audit["area_calc"].gt(0)
gov_ok = np.isfinite(audit["area_gov"]) & audit["area_gov"].gt(0)
paired = calc_ok & gov_ok
audit["area_calc_over_gov"] = (audit["area_calc"] / audit["area_gov"].where(gov_ok)).where(paired)
audit["area_diff_pct_of_gov"] = 100 * (audit["area_calc_over_gov"] - 1)
audit["area_ratio_delta"] = (audit["area_calc_over_gov"] - audit["area_ratio"]).where(
    np.isfinite(audit["area_ratio"]))
audit["Q_to_mm_day_factor_gov"] = 86.4 / audit["area_gov"].where(gov_ok)
length_ok = np.isfinite(audit["mvert_dist"]) & audit["mvert_dist"].gt(0)
audit["elon_from_saved_A_L"] = (2 * np.sqrt(audit["area_calc"].where(calc_ok) / np.pi)
                                  / audit["mvert_dist"].where(length_ok))
audit["elon_abs_delta"] = (audit["elon_from_saved_A_L"] - audit["elon_ratio"]).abs().where(
    np.isfinite(audit["elon_ratio"]))
area_summary_rows, shape_summary_rows = [], []
for scope, mask in [("all_A", pd.Series(True, index=audit.index)), ("selected", audit["selected"])]:
    part = audit.loc[mask]
    abs_pct = part["area_diff_pct_of_gov"].abs()
    ratio_delta = part["area_ratio_delta"].abs()
    area_summary_rows.append({
        "scope": scope, "basins": len(part), "calc_nonpositive_or_nonfinite": int((mask & ~calc_ok).sum()),
        "gov_nonpositive_or_nonfinite": int((mask & ~gov_ok).sum()),
        "paired_positive_areas": int(part["area_calc_over_gov"].notna().sum()),
        "metadata_ratio_nonfinite": int((~np.isfinite(part["area_ratio"])).sum()),
        "abs_area_diff_pct_median": abs_pct.median(), "abs_area_diff_pct_p99": abs_pct.quantile(0.99),
        "abs_area_diff_pct_max": abs_pct.max(), "abs_ratio_delta_max": ratio_delta.max(),
    })
    shape_summary_rows.append({
        "scope": scope, "basins": len(part), "length_nonpositive_or_nonfinite": int((mask & ~length_ok).sum()),
        "stored_elon_gt_1": int(part["elon_ratio"].gt(1).sum()),
        "elon_abs_delta_median": part["elon_abs_delta"].median(),
        "elon_abs_delta_max": part["elon_abs_delta"].max(),
    })
A1_AREA_AUDIT_SUMMARY = pd.DataFrame(area_summary_rows)
A1_SHAPE_AUDIT_SUMMARY = pd.DataFrame(shape_summary_rows)
print("(1) Kiểm diện tích; % chênh dùng area_gov làm mẫu số, delta so với area_ratio metadata:")
with pd.option_context("display.max_columns", None):
    display(A1_AREA_AUDIT_SUMMARY)
print("\n(2) Tối đa 10 ID được chọn có |% chênh diện tích| lớn nhất; diện tích dùng km²:")
area_examples = audit[audit["selected"]].sort_values(
    "area_diff_pct_of_gov", key=lambda values: values.abs(), ascending=False).head(10)
with pd.option_context("display.max_columns", None, "display.max_colwidth", 35):
    display(area_examples[["name", "country", "area_calc", "area_gov", "area_ratio",
                           "area_calc_over_gov", "area_diff_pct_of_gov", "Q_to_mm_day_factor_gov"]])
print("\n(3) Kiểm elon_ratio theo Re=2*sqrt(area_calc/pi)/mvert_dist, cả A và L dùng km²/km:")
display(A1_SHAPE_AUDIT_SUMMARY)
print("Số lưu có thể đã làm tròn; sai khác công thức chưa tự chứng minh lỗi hay đơn vị sai.")
print("\n(4) Tối đa 10 ID được chọn có elon_ratio>1; chỉ liệt kê để kiểm, không gán nhãn lỗi:")
shape_examples = audit[audit["selected"] & audit["elon_ratio"].gt(1)].sort_values("elon_ratio", ascending=False).head(10)
display(shape_examples[["name", "area_calc", "mvert_dist", "elon_ratio", "elon_from_saved_A_L", "elon_abs_delta"]])
missing_geol_ids = catchment.index[catchment["geol_perme"].str.strip().eq("")].tolist()
print("ID có geol_perme rỗng (không thuộc đầu vào 33 trường):", missing_geol_ids)
if (~gov_ok & audit["selected"]).any():
    print("CẢNH BÁO: diện tích area_gov không dương/hữu hạn ở ID được chọn:",
          audit.index[~gov_ok & audit["selected"]].tolist())
for suffix, table in [("area_shape_by_basin", audit.reset_index()), ("area_audit_summary", A1_AREA_AUDIT_SUMMARY),
                      ("shape_audit_summary", A1_SHAPE_AUDIT_SUMMARY)]:
    table.to_csv(TABLE_DIR / f"A1_P3_{suffix}.csv", index=False, encoding="utf-8-sig")
print("Đã lưu ba bảng vào:", TABLE_DIR)

# %% [markdown]
# ### 3.5. Đọc lớp bản đồ và kiểm hệ tọa độ, hình học, khóa ID
#
# Đọc các shapefile của A/D, kiểm CRS và hình học, đối chiếu tập ID với CSV trước khi vẽ.

# %% Phần 3.5 - Đọc lớp bản đồ và kiểm hệ tọa độ, hình học, khóa ID
# Đọc các shapefile của A/D, kiểm CRS và hình học, đối chiếu tập ID với CSV trước khi vẽ.
# Giữ nguyên hình học; không tự sửa CRS, đổi tọa độ hoặc sửa đa giác không hợp lệ.
import geopandas as gpd

A1_GEO_LAYERS, A1_GEO_LAYERS_INDEXED = {}, {}
geo_inventory, geo_id_checks, geo_schema = [], [], []
for group, folder in SUB_DATASETS.items():
    shape_dir = LAMAH_DIR / folder / "3_shapefiles"
    shape_files = sorted(path for path in shape_dir.iterdir()
                         if path.is_file() and path.suffix.lower() == ".shp" and not a1_is_metadata(path))
    if not shape_files:
        raise FileNotFoundError(f"Không thấy shapefile trong {shape_dir}.")
    for path in shape_files:
        layer_key = f"{group}/{path.name}"
        geo = gpd.read_file(path)
        A1_GEO_LAYERS[layer_key] = geo
        geometry = geo.geometry
        present = geometry.notna()
        nonempty = present & ~geometry.is_empty
        invalid = nonempty & ~geometry.is_valid
        bounds = geo.total_bounds
        crs = geo.crs
        units = ", ".join(sorted({axis.unit_name or "UNKNOWN" for axis in crs.axis_info})) if crs is not None else "UNKNOWN"
        geo_inventory.append({
            "layer": layer_key, "features": len(geo), "columns": len(geo.columns),
            "crs": crs.to_string() if crs is not None else "MISSING",
            "epsg": crs.to_epsg() if crs is not None else None,
            "axis_units": units, "geometry_types": str(geometry.geom_type.value_counts().to_dict()),
            "missing_geometry": int((~present).sum()), "empty_geometry": int((present & ~nonempty).sum()),
            "invalid_nonempty_geometry": int(invalid.sum()), "finite_total_bounds": bool(np.isfinite(bounds).all()),
        })
        expected_ids = set(A1_DAILY_FILES[group])
        id_row = {"layer": layer_key, "has_ID": "ID" in geo}
        if "ID" in geo:
            id_values = pd.to_numeric(geo["ID"], errors="coerce")
            id_ok = (np.isfinite(id_values) & id_values.gt(0) & id_values.mod(1).eq(0)).fillna(False)
            canonical = pd.Series(pd.NA, index=geo.index, dtype="object")
            canonical.loc[id_ok] = id_values.loc[id_ok].map(lambda value: str(int(value)))
            actual_ids = set(canonical.dropna())
            duplicate_rows = int(canonical.dropna().duplicated(keep=False).sum())
            id_row.update({"unique_IDs": len(actual_ids), "invalid_ID_rows": int((~id_ok).sum()),
                           "rows_in_duplicate_IDs": duplicate_rows,
                           "expected_IDs_missing": len(expected_ids - actual_ids),
                           "extra_to_expected": len(actual_ids - expected_ids),
                           "selected_IDs_missing": len(selected_ids - actual_ids)})
            if id_ok.all() and duplicate_rows == 0:
                A1_GEO_LAYERS_INDEXED[layer_key] = geo.set_axis(pd.Index(canonical, name="basin_id"))
            else:
                print(f"CẢNH BÁO {layer_key}: ID chưa hợp lệ/duy nhất, không tạo bảng đánh chỉ mục.")
        else:
            print(f"CẢNH BÁO {layer_key}: chưa thấy cột ID; xem schema, không đoán cột khóa.")
        geo_id_checks.append(id_row)
        for column in geo.columns:
            if column != geo.geometry.name:
                geo_schema.append({"layer": layer_key, "column": column, "dtype": str(geo[column].dtype)})
        if crs is None or invalid.any() or not nonempty.all():
            print(f"CẢNH BÁO {layer_key}: cần xem CRS/hình học trước khi vẽ, chưa tự sửa.")
A1_GEO_INVENTORY = pd.DataFrame(geo_inventory)
A1_GEO_ID_CHECKS = pd.DataFrame(geo_id_checks)
A1_GEO_SCHEMA = pd.DataFrame(geo_schema)
print("(1) Lớp bản đồ, CRS và hình học; các loại thiếu/rỗng/không hợp lệ được tách riêng:")
with pd.option_context("display.max_columns", None, "display.max_colwidth", 65):
    display(A1_GEO_INVENTORY)
print("\n(2) Khóa ID; A đối chiếu với ID khí tượng, D đối chiếu với ID Q ngày:")
with pd.option_context("display.max_columns", None):
    display(A1_GEO_ID_CHECKS)
print("\n(3) Tên cột, kiểu đọc từ shapefile và phạm vi tọa độ trong CRS gốc:")
for layer_key, geo in A1_GEO_LAYERS.items():
    print(f"\n{layer_key}:")
    print(", ".join(f"{column}: {geo[column].dtype}" for column in geo.columns))
    print("Bounds [xmin, ymin, xmax, ymax]:", geo.total_bounds.tolist())
for suffix, table in [("geo_inventory", A1_GEO_INVENTORY), ("geo_ID_checks", A1_GEO_ID_CHECKS),
                      ("geo_schema", A1_GEO_SCHEMA)]:
    table.to_csv(TABLE_DIR / f"A1_P3_{suffix}.csv", index=False, encoding="utf-8-sig")
print("Đã lưu ba bảng vào:", TABLE_DIR)
print("Chưa coi lon/lat là độ chỉ từ tên; đơn vị phải đọc cùng CRS. Chưa kiểm vị trí trạm so với lưu vực.")

# %% [markdown]
# ### 3.6. Bản đồ tập 859/451, diện tích, độ cao và quốc gia đặt trạm
#
# Ghép hình học với thuộc tính CSV bằng ID; vẽ bốn bản đồ cùng phạm vi, thang log chỉ áp cho màu diện tích.

# %% Phần 3.6 - Bản đồ tập 859/451, diện tích, độ cao và quốc gia đặt trạm
# Ghép hình học với thuộc tính CSV bằng ID; vẽ bốn bản đồ cùng phạm vi, thang log chỉ áp cho màu diện tích.
# Đa giác lớn vẽ trước vì mức A có thể lồng nhau; màu quốc gia chỉ quốc gia của trạm, không phải lãnh thổ.
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

basin_geo = A1_GEO_LAYERS_INDEXED["A/Basins_A.shp"]
gauge_geo = A1_GEO_LAYERS_INDEXED["D/Gauges.shp"]
if set(basin_geo.index) != a_ids or set(gauge_geo.index) != d_ids:
    raise ValueError("Tập ID bản đồ khác CSV; kiểm kết quả 3.5 trước khi vẽ.")
if basin_geo.crs != gauge_geo.crs or basin_geo.crs.to_epsg() != 3035:
    raise ValueError("CRS khác kết quả EPSG:3035 đã kiểm; chưa tự đổi CRS ở cell này.")
map_attributes = catchment[["area_calc", "elev_mean"]].apply(pd.to_numeric, errors="raise")
map_attributes = map_attributes.join(gauge[["name", "country"]], validate="one_to_one")
map_attributes["country"] = map_attributes["country"].str.strip().replace("", "UNKNOWN")
map_attributes["selected"] = map_attributes.index.isin(selected_ids)
A1_MAP_BASINS = basin_geo[[basin_geo.geometry.name]].join(map_attributes, validate="one_to_one")
A1_MAP_GAUGES = gauge_geo[[gauge_geo.geometry.name]].loc[A1_MAP_BASINS.index].join(
    map_attributes, validate="one_to_one")
if A1_MAP_BASINS[["area_calc", "elev_mean", "country"]].isna().any().any():
    raise ValueError("Thiếu thuộc tính sau ghép bản đồ; không vẽ màu ngầm cho ô thiếu.")
if not np.isfinite(A1_MAP_BASINS[["area_calc", "elev_mean"]].to_numpy()).all() or A1_MAP_BASINS["area_calc"].le(0).any():
    raise ValueError("Diện tích/độ cao chưa phù hợp để vẽ; kiểm trước khi dùng thang log.")
draw_basins = A1_MAP_BASINS.sort_values("area_calc", ascending=False)
selected_basins = draw_basins[draw_basins["selected"]]
selected_gauges = A1_MAP_GAUGES.loc[selected_basins.index]
A1_MAP_COUNTRY_COUNTS = pd.DataFrame({
    "all_D_gauges": gauge["country"].str.strip().replace("", "UNKNOWN").value_counts(),
    "A_basins": A1_MAP_BASINS["country"].value_counts(),
    "selected": selected_basins["country"].value_counts(),
}).fillna(0).astype(int).sort_index().rename_axis("gauge_country")
country_codes = A1_MAP_COUNTRY_COUNTS.index.tolist()
palette = plt.get_cmap("tab10").colors
country_colors = {code: palette[i % len(palette)] for i, code in enumerate(country_codes)}
country_colors["UNKNOWN"] = "#888888"
country_names = {"AUT": "Austria", "DEU": "Germany", "CZE": "Czechia", "CHE": "Switzerland",
                 "LIE": "Liechtenstein", "ITA": "Italy", "SVK": "Slovakia", "SVN": "Slovenia", "HUN": "Hungary"}
fig, axes = plt.subplots(2, 2, figsize=(17, 12))
fig.subplots_adjust(left=0.065, right=0.96, top=0.91, bottom=0.16, hspace=0.42, wspace=0.23)
fig.suptitle("Extended LamaH-CE: study basins and static attributes", fontsize=16, y=0.97)


def a1_map_background(ax):
    draw_basins.plot(ax=ax, facecolor="#edf0f2", edgecolor="#c9cfd5", linewidth=0.12, zorder=1)


def a1_map_numeric(ax, column, title, norm, cmap, label):
    a1_map_background(ax)
    selected_basins.plot(ax=ax, column=column, norm=norm, cmap=cmap,
                         edgecolor="white", linewidth=0.18, zorder=2)
    ax.scatter(selected_gauges.geometry.x, selected_gauges.geometry.y,
               c=selected_gauges[column].to_numpy(), norm=norm, cmap=cmap,
               s=10, edgecolors="#20252b", linewidths=0.25, zorder=4)
    colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                           fraction=0.038, pad=0.025, shrink=0.8)
    colorbar.set_label(label, fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, pad=10)


ax = axes[0, 0]
a1_map_background(ax)
selected_basins.plot(ax=ax, facecolor="#58a5c7", edgecolor="#155c7a", linewidth=0.25, zorder=2)
ax.scatter(selected_gauges.geometry.x, selected_gauges.geometry.y, c="#163e55", s=7, zorder=4)
ax.set_title(f"(a) Selected subset: {len(selected_basins)} of {len(draw_basins)} A basins", fontsize=11, pad=10)
ax.legend(handles=[Patch(facecolor="#edf0f2", edgecolor="#aeb7bf", label=f"All A basins ({len(draw_basins)})"),
                   Patch(facecolor="#58a5c7", edgecolor="#155c7a", label=f"Selected ({len(selected_basins)})"),
                   Line2D([0], [0], marker="o", color="none", markerfacecolor="#163e55", markersize=4,
                          label=f"Selected gauges ({len(selected_gauges)})")],
          loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=8, frameon=False)
a1_map_numeric(axes[0, 1], "area_calc", "(b) Catchment area: selected basins and gauges",
               colors.LogNorm(vmin=selected_basins["area_calc"].min(), vmax=selected_basins["area_calc"].max()),
               "viridis", "Calculated catchment area (km²; log scale)")
a1_map_numeric(axes[1, 0], "elev_mean", "(c) Mean basin elevation: selected basins and gauges",
               colors.Normalize(vmin=selected_basins["elev_mean"].min(), vmax=selected_basins["elev_mean"].max()),
               "terrain", "Mean catchment elevation (m a.s.l.)")
ax = axes[1, 1]
a1_map_background(ax)
selected_basins.plot(ax=ax, color=selected_basins["country"].map(country_colors),
                     edgecolor="white", linewidth=0.18, zorder=2)
ax.scatter(selected_gauges.geometry.x, selected_gauges.geometry.y,
           c=selected_gauges["country"].map(country_colors).tolist(),
           s=10, edgecolors="#20252b", linewidths=0.25, zorder=4)
ax.set_title("(d) Outlet-gauge country: selected basins and gauges", fontsize=11, pad=10)
country_handles = [Patch(facecolor=country_colors[code], edgecolor="#444444",
    label=f"{country_names.get(code, code)} ({code}): {A1_MAP_COUNTRY_COUNTS.loc[code, 'selected']}")
    for code in country_codes if A1_MAP_COUNTRY_COUNTS.loc[code, "selected"] > 0]
ax.legend(handles=country_handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
          ncol=2 if len(country_handles) <= 4 else 3, fontsize=8, frameon=False)
xmin, ymin, xmax, ymax = A1_MAP_BASINS.total_bounds
dx, dy = xmax - xmin, ymax - ymin
for ax in axes.flat:
    ax.set_xlim(xmin - 0.025 * dx, xmax + 0.025 * dx)
    ax.set_ylim(ymin - 0.035 * dy, ymax + 0.035 * dy)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value / 1000:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value / 1000:.0f}"))
    ax.set_xlabel("Easting (km)", fontsize=9)
    ax.set_ylabel("Northing (km)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.15, linewidth=0.4)
fig.text(0.5, 0.035, "Nested A basins overlap; large polygons drawn first. Points are outlet gauges.\n"
         "Country colours identify the outlet gauge's country; basins may cross national borders. CRS: EPSG:3035, shown in km.",
         ha="center", va="center", fontsize=9)
FIGURE_DIR = WORK_DIR / "Figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
map_path = FIGURE_DIR / "A1_P3_basin_maps.png"
fig.savefig(map_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
A1_MAP_COUNTRY_COUNTS.to_csv(TABLE_DIR / "A1_P3_map_country_counts.csv", encoding="utf-8-sig")
A1_MAP_ATTRIBUTES = map_attributes.assign(gauge_x_m=A1_MAP_GAUGES.geometry.x, gauge_y_m=A1_MAP_GAUGES.geometry.y)
A1_MAP_ATTRIBUTES.to_csv(TABLE_DIR / "A1_P3_map_attributes.csv", encoding="utf-8-sig")
print(f"Đã ghép theo ID: {len(A1_MAP_BASINS)} đa giác, {len(A1_MAP_GAUGES)} trạm tương ứng A; chọn {len(selected_basins)}.")
print("Quốc gia theo vị trí trạm, không phải số quốc gia mà lưu vực đi qua:")
display(A1_MAP_COUNTRY_COUNTS)
print("Đã lưu hình:", map_path)
print("Màu diện tích dùng area_calc trên thang log; độ cao là elev_mean lưu vực, không phải elev trạm.")
print("Đa giác mức A có thể chồng nhau; chưa kiểm định lượng vị trí trạm trong đa giác hoặc diện tích hình học.")

# %% [markdown]
# ### 3.7. Phân phối diện tích và đối chiếu area_calc với area_gov
#
# Vẽ diện tích trên trục tuyến tính/log, scatter hai diện tích và histogram tỷ số tính lại.

# %% Phần 3.7 - Phân phối diện tích và đối chiếu area_calc với area_gov
# Vẽ diện tích trên trục tuyến tính/log, scatter hai diện tích và histogram tỷ số tính lại.
# Histogram dùng cùng biên lớp và tỷ lệ trong từng tập; tập 451 nằm trong 859, không phải hai mẫu độc lập.
area_plot = A1_AREA_SHAPE_AUDIT.copy()
if not area_plot.index.is_unique or set(area_plot.index) != a_ids:
    raise ValueError("Khóa diện tích chưa khớp tập A; kiểm cell 3.4.")
area_values = area_plot[["area_calc", "area_gov"]].to_numpy(dtype=float)
if not np.isfinite(area_values).all() or (area_values <= 0).any():
    raise ValueError("Diện tích có ô thiếu/không dương; không tự bỏ điểm trước khi vẽ log.")
if set(area_plot.index[area_plot["selected"]]) != selected_ids:
    raise ValueError("Tập chọn của bảng diện tích khác basins_filtered.txt.")
area_plot["ratio_calc_gov"] = area_plot["area_calc"] / area_plot["area_gov"]
area_groups = [("all_A", area_plot, "#737d87", "All A"),
               ("selected", area_plot.loc[area_plot["selected"]], "#1976a2", "Selected")]
linear_bins = np.linspace(area_plot["area_calc"].min(), area_plot["area_calc"].max(), 31)
log_bins = np.geomspace(area_plot["area_calc"].min(), area_plot["area_calc"].max(), 31)
ratio_values = area_plot["ratio_calc_gov"]
ratio_bins = np.linspace(min(ratio_values.min(), 1.0), max(ratio_values.max(), 1.0), 31)
area_hist_rows, area_stats_rows = [], []
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
fig.subplots_adjust(left=0.085, right=0.97, top=0.90, bottom=0.12, hspace=0.38, wspace=0.30)
fig.suptitle("Catchment areas: distributions and agreement between sources", fontsize=15, y=0.97)
hist_specs = [(axes[0, 0], "linear_area", "area_calc", linear_bins,
               "(a) Calculated area: linear axis", "Calculated catchment area (km²)"),
              (axes[0, 1], "log_area", "area_calc", log_bins,
               "(b) Calculated area: logarithmic axis", "Calculated catchment area (km²; log axis)"),
              (axes[1, 1], "area_ratio", "ratio_calc_gov", ratio_bins,
               "(d) Ratio of calculated to reported area", "Calculated area / reported area")]
for ax, view, column, bins, title, xlabel in hist_specs:
    for scope, part, color, label in area_groups:
        values = part[column].to_numpy(dtype=float)
        counts, edges = np.histogram(values, bins=bins)
        if int(counts.sum()) != len(part):
            raise ValueError(f"Biên histogram {view}/{scope} chưa bao hết dữ liệu.")
        percentages = counts * 100.0 / len(part)
        ax.stairs(percentages, edges, color=color, linewidth=1.8, label=f"{label} (n={len(part)})")
        for left, right, count, pct in zip(edges[:-1], edges[1:], counts, percentages):
            area_hist_rows.append({"view": view, "scope": scope, "bin_left": left,
                                   "bin_right": right, "count": int(count), "pct_of_scope": pct})
    ax.set(title=title, xlabel=xlabel, ylabel="Basins per bin (% of each set)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.18)
axes[0, 1].set_xscale("log")
axes[1, 1].axvline(1, color="#b84727", linestyle="--", linewidth=1, label="Equal areas")
axes[1, 1].legend(fontsize=8, frameon=False)
ax = axes[1, 0]
for scope, part, color, label in area_groups:
    ax.scatter(part["area_gov"], part["area_calc"], color=color,
               s=14 if scope == "all_A" else 19, alpha=0.5 if scope == "all_A" else 0.8,
               edgecolors="none", label=f"{label} (n={len(part)})", zorder=2 if scope == "all_A" else 3)
area_min, area_max = area_values.min() / 1.15, area_values.max() * 1.15
ax.plot([area_min, area_max], [area_min, area_max], "--", color="#b84727", linewidth=1, label="1:1 line", zorder=1)
ax.set(xscale="log", yscale="log", xlim=(area_min, area_max), ylim=(area_min, area_max),
       title="(c) Matched areas by basin ID", xlabel="Reported area: area_gov (km²)",
       ylabel="Calculated area: area_calc (km²)")
ax.set_aspect("equal", adjustable="box")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.18)
if "205" in area_plot.index:
    point = area_plot.loc["205"]
    ax.annotate("ID 205", xy=(point["area_gov"], point["area_calc"]), xytext=(35, -26),
                textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 0.8}, zorder=5)
for scope, part, color, label in area_groups:
    row = {"scope": scope, "basins": len(part)}
    for column in ["area_calc", "area_gov", "ratio_calc_gov"]:
        values = part[column]
        row.update({f"{column}_{key}": value for key, value in
                    [("min", values.min()), ("median", values.median()),
                     ("p99", values.quantile(0.99)), ("max", values.max())]})
    area_stats_rows.append(row)
fig.text(0.5, 0.035, "Selected basins are included in All A. Each histogram sums to 100% within its own set.\n"
         "Common bins within each panel; panel (b) uses equal log-width bins. No areas are clipped or transformed for training.",
         ha="center", va="center", fontsize=9)
area_figure_path = FIGURE_DIR / "A1_P3_area_distributions.png"
fig.savefig(area_figure_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
A1_AREA_PLOT_SUMMARY = pd.DataFrame(area_stats_rows)
A1_AREA_HISTOGRAMS = pd.DataFrame(area_hist_rows)
A1_AREA_PLOT_SUMMARY.to_csv(TABLE_DIR / "A1_P3_area_plot_summary.csv", index=False, encoding="utf-8-sig")
A1_AREA_HISTOGRAMS.to_csv(TABLE_DIR / "A1_P3_area_histogram_bins.csv", index=False, encoding="utf-8-sig")
print("Thống kê diện tích và tỷ số tính lại, không dùng tỷ số metadata đã làm tròn:")
with pd.option_context("display.max_columns", None):
    display(A1_AREA_PLOT_SUMMARY)
print("Đã lưu hình:", area_figure_path)
print("Đã lưu bảng thống kê và số đếm từng bin vào:", TABLE_DIR)
print("Histogram: tỷ lệ trong từng tập, không phải mật độ/km²; bin cuối gồm cả cạnh phải.")
print("Hai tập lồng nhau; hình chưa là kiểm định tương đương phân phối hoặc tiêu chí loại lưu vực.")

# %% [markdown]
# ### 3.8. Phân phối 33 thuộc tính mô hình dùng, chia theo bốn nhóm
#
# Vẽ histogram có chung biên lớp cho 859/451; lc_dom dùng tần số mã, area_calc dùng trục log.

# %% Phần 3.8 - Phân phối 33 thuộc tính mô hình dùng, chia theo bốn nhóm
# Vẽ histogram có chung biên lớp cho 859/451; lc_dom dùng tần số mã, area_calc dùng trục log.
# Giữ mọi giá trị hợp lệ, gồm NDVI/GVF âm hoặc GVF >1; không đổi biểu diễn đầu vào huấn luyện.
static_plot_groups = {
    "topography": ("Topography and basin shape", [
        ("area_calc", "km²"), ("elev_mean", "m"), ("elev_med", "m"), ("elev_std", "m"),
        ("elev_ran", "m"), ("slope_mean", "m/km"), ("mvert_dist", "km"),
        ("mvert_ang", "degrees; circular"), ("elon_ratio", "dimensionless"), ("strm_dens", "km/km²")]),
    "climate": ("Long-term climate attributes", [
        ("p_mean", "mm/day"), ("et0_mean", "mm/day"), ("eta_mean", "mm/day"),
        ("arid_1", "dimensionless"), ("arid_2", "dimensionless"), ("p_season", "dimensionless"),
        ("frac_snow", "fraction"), ("hi_prec_fr", "days/year"), ("hi_prec_du", "days"),
        ("lo_prec_fr", "days/year"), ("lo_prec_du", "days")]),
    "land_cover": ("Land-cover attributes", [
        ("lc_dom", "CORINE code"), ("agr_fra", "fraction"), ("bare_fra", "fraction"),
        ("forest_fra", "fraction"), ("lake_fra", "fraction"), ("urban_fra", "fraction")]),
    "vegetation": ("Vegetation attributes", [
        ("lai_max", "dimensionless"), ("lai_diff", "dimensionless"),
        ("ndvi_max", "dimensionless"), ("ndvi_min", "dimensionless"),
        ("gvf_max", "dimensionless"), ("gvf_diff", "dimensionless")]),
}
plot_fields = [field for title, fields in static_plot_groups.values() for field, unit in fields]
if len(plot_fields) != len(set(plot_fields)) or set(plot_fields) != set(attribute_union):
    raise ValueError("Nhóm biểu đồ chưa khớp các tên static_attributes đã đo ở 3.2.")
static_values = catchment[plot_fields].apply(pd.to_numeric, errors="raise")
if not static_values.index.is_unique or set(static_values.index) != a_ids:
    raise ValueError("Khóa thuộc tính chưa khớp tập A.")
if not np.isfinite(static_values.to_numpy()).all() or static_values.eq(-999).any().any():
    raise ValueError("Phát hiện giá trị thiếu/không hữu hạn/-999; không tự bỏ trước khi vẽ.")
if static_values["area_calc"].le(0).any() or static_values["lc_dom"].mod(1).ne(0).any():
    raise ValueError("Diện tích không dương hoặc mã CORINE không nguyên; cần xem lại.")
static_scopes = [("all_A", static_values, "#737d87", "All A"),
                 ("selected", static_values.loc[static_values.index.isin(selected_ids)], "#1976a2", "Selected")]
static_stats, static_bins, static_categories, static_paths = [], [], [], []
for group, (group_title, fields) in static_plot_groups.items():
    nrows = (len(fields) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.1 * nrows + 1.8), squeeze=False)
    fig.subplots_adjust(left=0.065, right=0.98, top=0.87, bottom=0.11, hspace=0.62, wspace=0.30)
    fig.suptitle(group_title, fontsize=16, y=0.97)
    fig.legend(handles=[Line2D([0], [0], color=color, linewidth=2, label=f"{label} (n={len(part)})")
                        for scope, part, color, label in static_scopes],
               loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False, fontsize=10)
    for ax, (field, unit) in zip(axes.flat, fields):
        values_all = static_values[field]
        categorical = field == "lc_dom"
        if categorical:
            categories = sorted(values_all.astype(int).unique())
        elif field == "area_calc":
            edges = np.geomspace(values_all.min(), values_all.max(), 21)
        elif field == "mvert_ang":
            if not values_all.between(0, 360).all():
                raise ValueError("Góc nằm ngoài 0–360 độ; chưa tự quy về vòng.")
            edges = np.linspace(0, 360, 19)
        else:
            edges = np.histogram_bin_edges(values_all.to_numpy(), bins=20)
        for position, (scope, part, color, label) in enumerate(static_scopes):
            values = part[field]
            row = {"group": group, "attribute": field, "unit": unit, "scope": scope,
                   "kind": "categorical" if categorical else "numeric", "basins": len(part),
                   "unique_values": values.nunique()}
            if categorical:
                counts = values.astype(int).value_counts().reindex(categories, fill_value=0)
                percentages = counts.to_numpy() * 100.0 / len(part)
                ax.bar(np.arange(len(categories)) + (position - 0.5) * 0.38, percentages,
                       width=0.38, color=color, alpha=0.85)
                for code, count, pct in zip(categories, counts, percentages):
                    static_categories.append({"attribute": field, "scope": scope, "code": int(code),
                                              "count": int(count), "pct_of_scope": pct})
            else:
                counts, edges = np.histogram(values.to_numpy(), bins=edges)
                if int(counts.sum()) != len(part):
                    raise ValueError(f"Biên lớp chưa bao hết {field}/{scope}.")
                percentages = counts * 100.0 / len(part)
                ax.stairs(percentages, edges, color=color, linewidth=1.7)
                for left, right, count, pct in zip(edges[:-1], edges[1:], counts, percentages):
                    static_bins.append({"group": group, "attribute": field, "scope": scope,
                                        "bin_left": left, "bin_right": right, "count": int(count), "pct_of_scope": pct})
                row.update({"min": values.min(), "p05": values.quantile(0.05), "median": values.median(),
                            "p95": values.quantile(0.95), "max": values.max()})
            static_stats.append(row)
        if categorical:
            ax.set_xticks(np.arange(len(categories)), [str(code) for code in categories], rotation=60)
            ax.set_ylabel("Basins per class (%)", fontsize=9)
        else:
            ax.set_ylabel("Basins per bin (%)", fontsize=9)
        if field == "area_calc":
            ax.set_xscale("log")
        ax.set_title(field, fontsize=11)
        ax.set_xlabel(unit + ("; log axis" if field == "area_calc" else ""), fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
    for ax in axes.flat[len(fields):]:
        ax.set_visible(False)
    special_note = {"topography": "Area uses a log axis. Direction is circular: 0 and 360 degrees meet.",
                    "climate": "Attributes describe long-term climate, not individual daily forcing values.",
                    "land_cover": "CORINE codes are categorical labels; equal spacing does not imply numerical order.",
                    "vegetation": "Negative NDVI/GVF and GVF above 1 are retained; no clipping is applied."}[group]
    fig.text(0.5, 0.035, "Selected is a subset of All A. Percentages use each set's own basin count; common bins within each panel.\n"
             + special_note, ha="center", va="center", fontsize=9)
    path = FIGURE_DIR / f"A1_P3_static_{group}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    static_paths.append(path)
A1_STATIC_PLOT_SUMMARY = pd.DataFrame(static_stats)
A1_STATIC_HISTOGRAMS = pd.DataFrame(static_bins)
A1_STATIC_CLASS_COUNTS = pd.DataFrame(static_categories)
for suffix, table in [("static_plot_summary", A1_STATIC_PLOT_SUMMARY),
                      ("static_histogram_bins", A1_STATIC_HISTOGRAMS), ("static_class_counts", A1_STATIC_CLASS_COUNTS)]:
    table.to_csv(TABLE_DIR / f"A1_P3_{suffix}.csv", index=False, encoding="utf-8-sig")
print("(1) Phân vị của 32 thuộc tính số trong 451; lc_dom là mã lớp nên không tính phân vị:")
display(A1_STATIC_PLOT_SUMMARY.query("scope == 'selected' and kind == 'numeric'")[
    ["group", "attribute", "unit", "basins", "p05", "median", "p95"]])
print("(2) Tần số mã lc_dom, phần trăm tính riêng cho từng tập:")
display(A1_STATIC_CLASS_COUNTS)
print("Đã lưu bốn hình:")
for path in static_paths:
    print(path)
print("Đã lưu ba bảng vào:", TABLE_DIR)

# %% [markdown]
# ### 3.9. Kiểm chốt thuộc tính: cách vẽ, quan hệ, tương quan, scaler và hình học
#
# Dùng các bảng nhỏ đã nạp; không đọc lại chuỗi ngày, không chọn/bỏ biến hoặc huấn luyện mô hình.

# %% Phần 3.9 - Kiểm chốt thuộc tính: cách vẽ, quan hệ, tương quan, scaler và hình học
# Dùng các bảng nhỏ đã nạp; không đọc lại chuỗi ngày, không chọn/bỏ biến hoặc huấn luyện mô hình.
# Các mục A–F độc lập về ý nghĩa; bảng CSV lưu chi tiết, màn hình chỉ in kết quả cần chốt Phần 3.
from collections.abc import Mapping

final_static = static_values.loc[sorted(selected_ids, key=int)].copy()
if set(final_static.index) != selected_ids or not np.isfinite(final_static.to_numpy()).all():
    raise ValueError("Thuộc tính tập chọn chưa đầy đủ; kiểm 3.3/3.8 trước.")
final_tables = {}

# A. Đếm số 0 thật và đo lưới thập phân quan sát được; không đoán nguyên nhân làm tròn.
grid_rows, zero_rows = [], []
for field in plot_fields:
    values = static_values[field].to_numpy()
    grid_step = next((10.0 ** (-d) for d in range(7)
                      if np.allclose(values * 10.0 ** d, np.rint(values * 10.0 ** d), rtol=0, atol=1e-6)), np.nan)
    grid_counts = np.array([], dtype=int)
    if field not in {"lc_dom", "mvert_ang", "area_calc"} and np.isfinite(grid_step):
        nlevels = int(round((values.max() - values.min()) / grid_step)) + 1
        if nlevels <= 10000:
            grid = values.min() + np.arange(nlevels) * grid_step
            grid[-1] = values.max()
            grid_counts = np.histogram(grid, bins=np.histogram_bin_edges(values, bins=20))[0]
    grid_rows.append({"attribute": field, "unique_all_A": len(np.unique(values)),
                      "unique_selected": final_static[field].nunique(), "observed_decimal_grid": grid_step,
                      "old_bin_width": (values.max() - values.min()) / 20,
                      "grid_levels_per_old_bin_min": grid_counts.min() if grid_counts.size else np.nan,
                      "grid_levels_per_old_bin_max": grid_counts.max() if grid_counts.size else np.nan})
    if field != "lc_dom":
        for scope, part in [("all_A", static_values), ("selected", final_static)]:
            zero_rows.append({"scope": scope, "attribute": field, "basins": len(part),
                              "zero_count": int(part[field].eq(0).sum()), "zero_pct": part[field].eq(0).mean() * 100})
final_tables["static_grid_diagnostics"] = pd.DataFrame(grid_rows)
final_tables["static_exact_zeros"] = pd.DataFrame(zero_rows)
focus_fields = ["mvert_dist", "hi_prec_du", "p_season", "frac_snow", "lai_max", "ndvi_min", "lake_fra", "urban_fra"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.17, top=0.84, hspace=0.48, wspace=0.30)
fig.suptitle("Empirical cumulative distributions: no histogram bins", fontsize=15, y=0.97)
unit_map = {field: unit for title, fields in static_plot_groups.values() for field, unit in fields}
for ax, field in zip(axes.flat, focus_fields):
    for scope, part, color, label in static_scopes:
        unique, counts = np.unique(part[field], return_counts=True)
        ax.step(unique, 100 * counts.cumsum() / len(part), where="post", color=color,
                linewidth=1.6, label=f"{label} (n={len(part)})")
    if field == "mvert_dist":
        ax.set_xscale("log")
    ax.set(title=field, xlabel=unit_map[field] + ("; log axis" if field == "mvert_dist" else ""),
           ylabel="Basins at or below x (%)", ylim=(0, 102))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.18)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
fig.text(0.5, 0.035, "All stored values are retained. Steps are empirical counts; no smoothing or binning is applied.\n"
         "Length uses a log axis. Exact zero counts are reported separately; Selected is contained in All A.",
         ha="center", va="center", fontsize=9)
fig.savefig(FIGURE_DIR / "A1_P3_static_ecdf.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

# B. Quan hệ trong cùng một dòng: không ép GVF vào 0–1, không đòi các tỷ phần được chọn cộng thành 1.
fractions = ["agr_fra", "bare_fra", "forest_fra", "lake_fra", "urban_fra", "frac_snow"]
row_checks = {
    "fraction_outside_0_1": (~static_values[fractions].ge(0) | ~static_values[fractions].le(1)).any(axis=1),
    "ndvi_min_gt_max": static_values["ndvi_min"].gt(static_values["ndvi_max"]),
    "lai_diff_negative_or_gt_max": static_values["lai_diff"].lt(0) | static_values["lai_diff"].gt(static_values["lai_max"]),
    "gvf_diff_negative": static_values["gvf_diff"].lt(0),
    "angle_outside_0_360": ~static_values["mvert_ang"].between(0, 360),
    "wet_or_dry_duration_lt_1": static_values[["hi_prec_du", "lo_prec_du"]].lt(1).any(axis=1),
}
check_rows = []
for check, mask in row_checks.items():
    for scope, ids in [("all_A", static_values.index), ("selected", final_static.index)]:
        bad = mask.loc[ids]
        check_rows.append({"check": check, "scope": scope, "basins": len(ids), "flagged": int(bad.sum()),
                           "example_IDs": ", ".join(bad.index[bad][:10])})
final_tables["static_row_checks"] = pd.DataFrame(check_rows)

# C. Pearson tuyến tính và Spearman thứ hạng; loại mã lớp và góc thô, không tính p-value.
corr_fields = [field for field in plot_fields if field not in {"lc_dom", "mvert_ang"}]
correlations, pair_rows = {}, []
for scope, part in [("all_A", static_values), ("selected", final_static)]:
    pearson = part[corr_fields].corr(method="pearson")
    spearman = part[corr_fields].corr(method="spearman")
    correlations[scope] = (pearson, spearman)
    final_tables[f"static_pearson_{scope}"] = pearson.rename_axis("attribute").reset_index()
    final_tables[f"static_spearman_{scope}"] = spearman.rename_axis("attribute").reset_index()
    for i, left in enumerate(corr_fields):
        for right in corr_fields[i + 1:]:
            pair_rows.append({"scope": scope, "left": left, "right": right, "basins": len(part),
                              "pearson": pearson.loc[left, right], "spearman": spearman.loc[left, right]})
pair_table = pd.DataFrame(pair_rows)
final_tables["static_correlation_pairs"] = pair_table
fig, axes = plt.subplots(1, 2, figsize=(21, 11))
fig.subplots_adjust(left=0.075, right=0.91, bottom=0.23, top=0.89, wspace=0.32)
for ax, matrix, title in zip(axes, correlations["selected"], ["Pearson: linear association", "Spearman: rank association"]):
    picture = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_fields)), corr_fields, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(corr_fields)), corr_fields, fontsize=8)
    ax.set_title(title, fontsize=12, pad=12)
colorbar = fig.colorbar(picture, cax=fig.add_axes([0.935, 0.27, 0.012, 0.53]))
colorbar.set_label("Correlation coefficient")
fig.suptitle(f"Static attributes: {len(final_static)} selected basins, {len(corr_fields)} numeric fields", fontsize=16)
fig.text(0.5, 0.035, "CORINE codes and raw circular angles excluded. No feature selection or causal inference.\n"
         "Nested basins may be spatially dependent; coefficients are descriptive, with no significance tests.",
         ha="center", fontsize=10)
fig.savefig(FIGURE_DIR / "A1_P3_static_correlations.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

# D. Đọc thống kê thuộc tính trong từng scaler YAML của tác giả; so tham chiếu 451, không suy sai tập huấn luyện.
scaler_runs, scaler_rows = [], []
for config_path in config_paths:
    run_name = config_path.parent.relative_to(EXP_DIR).as_posix()
    scaler_path = config_path.parent / "train_data" / "train_data_scaler.yml"
    run_row = {"run": run_name, "status": "missing_yaml", "attributes": 0}
    if not scaler_path.is_file():
        scaler_runs.append(run_row)
        continue
    try:
        saved = a1_load_yaml(scaler_path.read_text(encoding="utf-8-sig"))
        mean_key = "attribute_means" if "attribute_means" in saved else "camels_attr_means"
        std_key = "attribute_stds" if "attribute_stds" in saved else "camels_attr_stds"
        if not isinstance(saved.get(mean_key), Mapping) or not isinstance(saved.get(std_key), Mapping):
            raise ValueError("Không có hai mapping mean/std thuộc tính được hỗ trợ.")
        means = pd.to_numeric(pd.Series(saved[mean_key]), errors="raise").reindex(plot_fields)
        stds = pd.to_numeric(pd.Series(saved[std_key]), errors="raise").reindex(plot_fields)
        if not np.isfinite(means).all() or not np.isfinite(stds).all() or stds.le(0).any():
            raise ValueError("Mean/std thiếu, không hữu hạn hoặc std không dương.")
        z = (final_static - means) / stds
        run_row.update(status="read_ok", attributes=len(means),
                       means_match_selected=bool(np.allclose(means, final_static.mean(), rtol=1e-6, atol=1e-8)),
                       stds_match_selected_ddof1=bool(np.allclose(stds, final_static.std(ddof=1), rtol=1e-6, atol=1e-8)))
        for field in plot_fields:
            scaler_rows.append({"run": run_name, "attribute": field, "saved_mean": means[field],
                                "saved_std": stds[field], "selected_z_min": z[field].min(),
                                "selected_z_max": z[field].max(), "max_abs_z": z[field].abs().max(),
                                "max_abs_z_ID": z[field].abs().idxmax(),
                                "kind": "coded_category" if field == "lc_dom" else "raw_angle" if field == "mvert_ang" else "numeric"})
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        run_row.update(status="unreadable_or_unsupported", detail=str(error))
    scaler_runs.append(run_row)
final_tables["static_scaler_runs"] = pd.DataFrame(scaler_runs)
final_tables["static_scaler_values"] = pd.DataFrame(scaler_rows)

# E. Quan hệ trạm–đa giác và diện tích hình học trong CRS bằng mét đã kiểm ở 3.5.
basin_shape = A1_MAP_BASINS.geometry
gauge_shape = A1_MAP_GAUGES.geometry.reindex(basin_shape.index)
spatial = pd.DataFrame({"selected": basin_shape.index.isin(selected_ids),
                        "gauge_covered": basin_shape.covers(gauge_shape, align=True),
                        "gauge_distance_to_polygon_m": basin_shape.distance(gauge_shape, align=True),
                        "geometry_area_km2": basin_shape.area / 1e6}, index=basin_shape.index)
spatial["geometry_minus_attribute_pct"] = 100 * (spatial["geometry_area_km2"] / static_values["area_calc"] - 1)
spatial_rows = []
for scope, part in [("all_A", spatial), ("selected", spatial.loc[final_static.index])]:
    spatial_rows.append({"scope": scope, "basins": len(part), "gauges_not_covered": int((~part["gauge_covered"]).sum()),
                         "max_distance_m": part["gauge_distance_to_polygon_m"].max(),
                         "abs_area_delta_pct_median": part["geometry_minus_attribute_pct"].abs().median(),
                         "abs_area_delta_pct_max": part["geometry_minus_attribute_pct"].abs().max()})
final_tables["static_geometry_by_basin"] = spatial.rename_axis("basin_id").reset_index()
final_tables["static_geometry_summary"] = pd.DataFrame(spatial_rows)

# F. Lưu đủ bằng chứng; chỉ kết luận sau khi đọc output thực tế, không tự sửa dữ liệu/feature/scaler.
for suffix, table in final_tables.items():
    table.to_csv(TABLE_DIR / f"A1_P3_{suffix}.csv", index=False, encoding="utf-8-sig")
print("(1) Số 0 thật trong các tỷ phần được lưu; bin đầu histogram không đồng nghĩa giá trị 0:")
display(final_tables["static_exact_zeros"].query("scope == 'selected' and attribute in @fractions"))
print("(2) Lưới thập phân quan sát và số mức lưới trong bin cũ; chưa chứng minh nguồn làm tròn:")
display(final_tables["static_grid_diagnostics"].query("attribute in @focus_fields"))
print("(3) Quan hệ trong dòng; flagged chỉ là số cần đọc lại, chưa tự sửa:")
display(final_tables["static_row_checks"])
print("(4) Các cặp tương quan mạnh nhất theo |Spearman| trong tập chọn; toàn bộ cặp ở CSV:")
selected_pairs = pair_table.query("scope == 'selected'")
display(selected_pairs.loc[selected_pairs["spearman"].abs().sort_values(ascending=False).index].head(12))
requested_pairs = [{"elev_mean", "elev_med"}, {"lai_max", "lai_diff"}, {"gvf_max", "gvf_diff"}, {"arid_1", "arid_2"}]
display(selected_pairs.loc[[{row.left, row.right} in requested_pairs for row in selected_pairs.itertuples()]])
print("(5) Đọc scaler từng run; so với 451 hiện tại, False chưa chứng minh scaler sai:")
display(final_tables["static_scaler_runs"])
if scaler_rows:
    z_table = final_tables["static_scaler_values"]
    z_numeric = z_table.query("kind == 'numeric'").sort_values("max_abs_z", ascending=False)
    display(z_numeric.drop_duplicates("attribute").head(10))
print("(6) Quan hệ hình học; khoảng cách từ trạm tới phần đa giác, 0 nếu trong/trên biên:")
display(final_tables["static_geometry_summary"])
outside = spatial.loc[~spatial["gauge_covered"]].sort_values("gauge_distance_to_polygon_m", ascending=False)
if not outside.empty:
    display(outside.head(10))
print("Đã lưu hai hình static_ecdf/static_correlations và", len(final_tables), "bảng A1_P3_* vào", WORK_DIR)
print("Gửi hai hình và bảng (1)–(6). Đây là lượt kiểm chốt Phần 3; chưa coi các số chưa chạy là kết quả.")

# %% [markdown]
# ### 3.10. Đối chiếu cuối: tọa độ, tập đầu nguồn, tổng tỷ phần và scaler thiếu
#
# Chỉ dùng bảng đã nạp và cấu hình; đầu nguồn trong 451 phải lần qua cả các trạm không được chọn.

# %% Phần 3.10 - Đối chiếu cuối: tọa độ, tập đầu nguồn, tổng tỷ phần và scaler thiếu
# Chỉ dùng bảng đã nạp và cấu hình; đầu nguồn trong 451 phải lần qua cả các trạm không được chọn.
# Không đổi CRS, sửa tổng tỷ phần, nạp pickle hoặc chép scaler của run khác vào run thiếu.
remaining_tables = {}
geo_d = A1_GEO_LAYERS_INDEXED["D/Gauges.shp"]

# A. Metadata nguồn ghi lon/lat là tọa độ EPSG:3035 theo mét: so trực tiếp với geometry x/y.
coordinate_rows, coordinate_details = [], []
for source_name, table in [("Gauge_attributes.csv", gauge), ("Gauges.shp fields", geo_d)]:
    xy = table[["lon", "lat"]].apply(pd.to_numeric, errors="coerce").reindex(geo_d.index)
    detail = pd.DataFrame({"source": source_name, "lon_minus_x_m": xy["lon"] - geo_d.geometry.x,
                           "lat_minus_y_m": xy["lat"] - geo_d.geometry.y}, index=geo_d.index)
    detail["basin_id"] = detail.index
    coordinate_details.append(detail.reset_index(drop=True))
    for scope, ids in [("all_D", geo_d.index), ("selected", final_static.index)]:
        part = detail.loc[ids, ["lon_minus_x_m", "lat_minus_y_m"]]
        coordinate_rows.append({"source": source_name, "scope": scope, "gauges": len(part),
                                "nonfinite_pairs": int((~np.isfinite(part).all(axis=1)).sum()),
                                "max_abs_lon_minus_x_m": part.iloc[:, 0].abs().max(),
                                "max_abs_lat_minus_y_m": part.iloc[:, 1].abs().max()})
remaining_tables["final_coordinate_checks"] = pd.DataFrame(coordinate_rows)
remaining_tables["final_coordinate_by_gauge"] = pd.concat(coordinate_details, ignore_index=True)

# B. So HIERARCHY gốc với đầu nguồn trong tập chọn; NEXTDOWNID=0 là không có trạm hạ lưu.
next_down = pd.to_numeric(geo_d["NEXTDOWNID"], errors="coerce")
hierarchy = pd.to_numeric(geo_d["HIERARCHY"], errors="coerce")
path_rows = []
for start_id in sorted(selected_ids, key=int):
    cursor, seen, nearest_selected, hops, status = start_id, {start_id}, None, 0, ""
    while True:
        value = next_down.loc[cursor]
        if not np.isfinite(value) or value < 0 or value % 1 != 0:
            status = "missing_or_invalid_nextdown"
            break
        if value == 0:
            status = "reaches_outlet"
            break
        target = str(int(value))
        if target in seen:
            status = "cycle"
            break
        if target not in geo_d.index:
            status = "target_absent_from_D"
            break
        hops += 1
        if nearest_selected is None and target in selected_ids:
            nearest_selected = target
        seen.add(target)
        cursor = target
    path_rows.append({"basin_id": start_id, "original_HIERARCHY": hierarchy.loc[start_id],
                      "nearest_downstream_selected": nearest_selected, "path_status": status, "total_hops": hops})
paths = pd.DataFrame(path_rows)
known_nested = set(paths["nearest_downstream_selected"].dropna())
paths["has_known_selected_upstream"] = paths["basin_id"].isin(known_nested)
complete_paths = paths["path_status"].eq("reaches_outlet").all()
headwater_count = int((~paths["has_known_selected_upstream"]).sum())
headwater_pct = 100 * headwater_count / len(paths) if complete_paths else np.nan
remaining_tables["final_selected_topology"] = paths
remaining_tables["final_headwater_comparison"] = pd.DataFrame([{
    "selected_basins": len(paths), "original_HIERARCHY_1": int(hierarchy.loc[final_static.index].eq(1).sum()),
    "original_HIERARCHY_missing": int(hierarchy.loc[final_static.index].isna().sum()),
    "all_selected_paths_resolved": bool(complete_paths),
    "no_known_selected_upstream": headwater_count, "known_nested_in_selection": len(known_nested),
    "headwater_pct_if_paths_resolved": headwater_pct,
    "matches_paper_72_5_to_one_decimal": bool(round(headwater_pct, 1) == 72.5) if complete_paths else None}])

# C. Năm tỷ phần là một phần các lớp phủ; kiểm tổng <=1, không đòi bằng 1.
land_fields = ["agr_fra", "bare_fra", "forest_fra", "lake_fra", "urban_fra"]
land_sum = static_values[land_fields].sum(axis=1)
land_rows = []
for scope, ids in [("all_A", static_values.index), ("selected", final_static.index)]:
    values = land_sum.loc[ids]
    land_rows.append({"scope": scope, "basins": len(values), "sum_min": values.min(),
                      "sum_median": values.median(), "sum_max": values.max(),
                      "sum_gt_1_exact": int(values.gt(1).sum()),
                      "sum_gt_1_tol1e_12": int(values.gt(1 + 1e-12).sum()),
                      "example_IDs": ", ".join(values.index[values.gt(1 + 1e-12)][:10])})
remaining_tables["final_land_fraction_sum"] = pd.DataFrame(land_rows)

# D. Kiểm đường dẫn, định dạng thay thế và base_run_dir của những run thiếu scaler YAML.
run_dirs = [path.parent for path in config_paths]
missing_rows = []
for config_path in config_paths:
    run_dir = config_path.parent
    if (run_dir / "train_data" / "train_data_scaler.yml").is_file():
        continue
    cfg = a1_load_yaml(config_path.read_text(encoding="utf-8-sig"))
    reference = str(cfg.get("base_run_dir", "")).replace("\\", "/").rstrip("/")
    matches = [path for path in run_dirs if reference.endswith("/" + path.relative_to(EXP_DIR).as_posix())
               or reference == path.relative_to(EXP_DIR).as_posix()]
    if not matches and reference:
        matches = [path for path in run_dirs if path.name == reference.split("/")[-1]]
    alternatives = [path.relative_to(run_dir).as_posix() for path in run_dir.rglob("train_data_scaler*")
                    if path.is_file() and not a1_is_metadata(path)]
    row = {"run": run_dir.relative_to(EXP_DIR).as_posix(), "local_candidates": " | ".join(alternatives),
           "base_run_dir_declared": reference, "base_matches_in_input": len(matches),
           "base_run_resolved": matches[0].relative_to(EXP_DIR).as_posix() if len(matches) == 1 else "",
           "base_yaml_exists": False, "base_static_matches_selected": None}
    if len(matches) == 1:
        base_scaler = matches[0] / "train_data" / "train_data_scaler.yml"
        row["base_yaml_exists"] = base_scaler.is_file()
        if base_scaler.is_file():
            saved = a1_load_yaml(base_scaler.read_text(encoding="utf-8-sig"))
            mean_values = saved.get("attribute_means", saved.get("camels_attr_means", {}))
            std_values = saved.get("attribute_stds", saved.get("camels_attr_stds", {}))
            means = pd.to_numeric(pd.Series(mean_values), errors="coerce").reindex(plot_fields)
            stds = pd.to_numeric(pd.Series(std_values), errors="coerce").reindex(plot_fields)
            row["base_static_matches_selected"] = bool(
                np.allclose(means, final_static.mean(), rtol=1e-6, atol=1e-8)
                and np.allclose(stds, final_static.std(ddof=1), rtol=1e-6, atol=1e-8))
    missing_rows.append(row)
remaining_tables["final_missing_scaler_probe"] = pd.DataFrame(missing_rows)
for suffix, table in remaining_tables.items():
    table.to_csv(TABLE_DIR / f"A1_P3_{suffix}.csv", index=False, encoding="utf-8-sig")
for label, suffix in [("(1) Đối chiếu lon/lat với x/y, đơn vị mét:", "final_coordinate_checks"),
                      ("(2) HIERARCHY gốc và đầu nguồn trong đúng tập 451:", "final_headwater_comparison"),
                      ("(3) Tổng năm tỷ phần lớp phủ:", "final_land_fraction_sum"),
                      ("(4) Run thiếu scaler: chỉ kiểm và đọc nguồn tham chiếu, không thay tệp:", "final_missing_scaler_probe")]:
    print(label)
    with pd.option_context("display.max_columns", None, "display.max_colwidth", 160):
        display(remaining_tables[suffix])
if not complete_paths:
    print("CẢNH BÁO: đường dẫn mạng còn thiếu/lỗi; chưa báo tỷ lệ đầu nguồn như kết quả đã xác nhận.")
    display(paths.loc[paths["path_status"].ne("reaches_outlet")])
print("Đã lưu", len(remaining_tables), "bảng vào:", TABLE_DIR)
print("Gửi bảng (1)–(4) và cảnh báo nếu có. Không cần thêm hình; đọc kết quả này để chốt Phần 3.")

# %% [markdown]
# ## Phần 4. Dữ liệu thiếu theo kỳ, năm và đoạn liên tiếp
#
# Phân biệt thiếu dòng, ô rỗng, sentinel và khoảng chưa có quan trắc.

# %% [markdown]
# ### 4.1. Thiếu khí tượng theo biến và kỳ trong 451 lưu vực
#
# Đọc chuỗi thật, đối chiếu lịch đầy đủ của từng kỳ; tách thiếu dòng, ô rỗng, -999, chữ và vô hạn.

# %% Phần 4.1 - Thiếu khí tượng theo biến và kỳ trong 451 lưu vực
# Đọc chuỗi thật, đối chiếu lịch đầy đủ của từng kỳ; tách thiếu dòng, ô rỗng, -999, chữ và vô hạn.
# Giữ nguyên số âm hợp lệ; lưu các ngày thiếu để phân tích theo năm/đoạn mà không đọc lại chuỗi gốc.
forcing_vars = A1_FORCING_COVERAGE["variable"].drop_duplicates().tolist()
forcing_vars = sorted(forcing_vars, key=lambda field: (SOURCE_PREFIXES.index(field.split("_", 1)[0]), field))
period_calendars = {period: pd.date_range(start, end, freq="D") for period, (start, end) in PERIODS.items()}
missing_counts = ["expected_cells", "valid_cells", "missing_rows", "blank_cells", "minus999_cells",
                  "non_numeric_nonblank", "infinite_cells", "total_missing"]
forcing_missing_parts, forcing_missing_cells = [], []
started_p4 = time.perf_counter()
for file_no, basin_id in enumerate(sorted(selected_ids, key=int), start=1):
    path = A1_DAILY_FILES["A"][basin_id]
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str,
                      usecols=DATE_COLS + forcing_vars, keep_default_na=False, na_filter=False)
    dates = pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"ID {basin_id}: ngày không hợp lệ/trùng/không tăng; kiểm trước khi đếm thiếu.")
    indexed = raw[forcing_vars].set_axis(pd.DatetimeIndex(dates, name="date"))
    for period, calendar in period_calendars.items():
        aligned = indexed.reindex(calendar)
        present = pd.Series(calendar.isin(indexed.index), index=calendar)
        cleaned = aligned.apply(lambda column: column.str.strip())
        numeric = cleaned.apply(pd.to_numeric, errors="coerce")
        blank = cleaned.eq("")
        nonnumeric = cleaned.notna() & cleaned.ne("") & numeric.isna()
        minus999 = numeric.eq(-999)
        infinite = np.isinf(numeric)
        valid = np.isfinite(numeric) & ~minus999
        part = pd.DataFrame({"variable": forcing_vars, "basin_id": basin_id, "period": period}).set_index("variable")
        part["source"] = part.index.str.split("_").str[0]
        part["expected_cells"] = len(calendar)
        part["valid_cells"] = valid.sum()
        part["missing_rows"] = int((~present).sum())
        part["blank_cells"] = blank.sum()
        part["minus999_cells"] = minus999.sum()
        part["non_numeric_nonblank"] = nonnumeric.sum()
        part["infinite_cells"] = infinite.sum()
        part["total_missing"] = part["expected_cells"] - part["valid_cells"]
        if not part[["missing_rows", "blank_cells", "minus999_cells", "non_numeric_nonblank", "infinite_cells"]].sum(axis=1).eq(part["total_missing"]).all():
            raise ValueError(f"ID {basin_id}/{period}: các loại thiếu chưa cộng đúng tổng.")
        forcing_missing_parts.append(part.reset_index())
        row_positions, column_positions = np.where(~valid.to_numpy())
        for i, j in zip(row_positions, column_positions):
            variable = forcing_vars[j]
            reason = ("missing_row" if not present.iloc[i] else "blank" if blank.iat[i, j]
                      else "minus999" if minus999.iat[i, j] else "infinite" if infinite.iat[i, j]
                      else "non_numeric_nonblank")
            forcing_missing_cells.append({"basin_id": basin_id, "period": period,
                                          "date": calendar[i].strftime("%Y-%m-%d"), "variable": variable,
                                          "source": variable.split("_", 1)[0], "reason": reason,
                                          "raw_token": aligned.iat[i, j] if present.iloc[i] else None})
    if file_no % 100 == 0 or file_no == len(selected_ids):
        print(f"Đã kiểm thiếu khí tượng {file_no}/{len(selected_ids)} lưu vực", flush=True)
A1_FORCING_MISSING_BY_BASIN = pd.concat(forcing_missing_parts, ignore_index=True)
A1_FORCING_MISSING_BY_BASIN["missing_pct"] = 100 * A1_FORCING_MISSING_BY_BASIN["total_missing"] / A1_FORCING_MISSING_BY_BASIN["expected_cells"]
A1_FORCING_MISSING_CELLS = pd.DataFrame(forcing_missing_cells, columns=[
    "basin_id", "period", "date", "variable", "source", "reason", "raw_token"])
by_variable = A1_FORCING_MISSING_BY_BASIN.groupby(["period", "source", "variable"], as_index=False).agg(
    **{name: (name, "sum") for name in missing_counts}, basins=("basin_id", "nunique"))
by_variable["missing_pct"] = 100 * by_variable["total_missing"] / by_variable["expected_cells"]
A1_FORCING_MISSING_BY_VARIABLE = by_variable
by_source = by_variable.groupby(["period", "source"], as_index=False).agg(
    **{name: (name, "sum") for name in missing_counts}, variables=("variable", "nunique"))
by_source["missing_pct"] = 100 * by_source["total_missing"] / by_source["expected_cells"]
A1_FORCING_MISSING_BY_SOURCE = by_source
if int(by_variable["total_missing"].sum()) != len(A1_FORCING_MISSING_CELLS):
    raise ValueError("Danh sách ngày–biến thiếu chưa khớp bảng đếm theo biến.")
missing_pivot = by_variable.pivot(index="variable", columns="period", values="missing_pct").reindex(
    index=forcing_vars, columns=list(PERIODS))
positive_rates = missing_pivot.to_numpy()[missing_pivot.to_numpy() > 0]
xmin_log = 10 ** (np.floor(np.log10(positive_rates.min())) - 1) if positive_rates.size else 0.001
xmax_log = min(100.0, max(xmin_log * 10, positive_rates.max() * 2)) if positive_rates.size else 1.0
fig, axes = plt.subplots(1, len(PERIODS), figsize=(16, 14), sharey=True, squeeze=False)
fig.subplots_adjust(left=0.23, right=0.98, top=0.91, bottom=0.11, wspace=0.16)
fig.suptitle(f"Missing meteorological values: {len(selected_ids)} selected basins", fontsize=16, y=0.97)
for ax, period in zip(axes.flat, PERIODS):
    rates = missing_pivot[period].to_numpy()
    positive = rates > 0
    ax.barh(np.arange(len(forcing_vars))[positive], rates[positive], color="#217da6", height=0.7)
    for position in np.flatnonzero(~positive):
        ax.text(0.02, position, "0", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color="#555555")
    ax.set_xscale("log")
    ax.set_xlim(xmin_log, xmax_log)
    ax.set_title(period.capitalize(), fontsize=12, pad=12)
    ax.set_xlabel("Missing day-variable cells (%; log axis)", fontsize=9)
    ax.set_yticks(np.arange(len(forcing_vars)), forcing_vars, fontsize=8)
    ax.grid(axis="x", alpha=0.18)
axes[0, 0].invert_yaxis()
fig.text(0.5, 0.035, "Each variable uses the full calendar of its period across the selected basins.\n"
         "Only positive rates have bars; 0 is shown as text, with no artificial epsilon. Negative finite measurements are retained.",
         ha="center", va="center", fontsize=10)
forcing_missing_figure = FIGURE_DIR / "A1_P4_forcing_missing_by_period.png"
fig.savefig(forcing_missing_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
for suffix, table in [("forcing_missing_by_basin", A1_FORCING_MISSING_BY_BASIN),
                      ("forcing_missing_by_variable", A1_FORCING_MISSING_BY_VARIABLE),
                      ("forcing_missing_by_source", A1_FORCING_MISSING_BY_SOURCE),
                      ("forcing_missing_cells", A1_FORCING_MISSING_CELLS)]:
    table.to_csv(TABLE_DIR / f"A1_P4_{suffix}.csv", index=False, encoding="utf-8-sig")
print("(1) Tỷ lệ thiếu (%) từng biến trong 451 lưu vực, mẫu số là lịch đầy đủ của từng kỳ:")
display(missing_pivot)
print("(2) Đếm theo nguồn; các loại thiếu không chồng lấp, số âm hữu hạn khác -999 vẫn hợp lệ:")
with pd.option_context("display.max_columns", None):
    display(A1_FORCING_MISSING_BY_SOURCE)
print("(3) Tối đa 15 (ID, biến, kỳ) có tỷ lệ thiếu cao nhất; toàn bộ lưu ở CSV:")
display(A1_FORCING_MISSING_BY_BASIN.loc[A1_FORCING_MISSING_BY_BASIN["total_missing"].gt(0)].sort_values(
    ["missing_pct", "basin_id", "variable"], ascending=[False, True, True]).head(15))
print("Số ô ngày–biến thiếu đã lưu:", len(A1_FORCING_MISSING_CELLS))
print("Đã lưu hình:", forcing_missing_figure)
print("Đã lưu bốn bảng vào:", TABLE_DIR)
print(f"Thời gian quét: {time.perf_counter() - started_p4:.1f} giây.")

# %% [markdown]
# ### 4.2. Bản đồ nhiệt thiếu qmax theo năm, tỷ lệ theo kỳ và số lưu vực có dữ liệu
#
# Dùng lịch năm đầy đủ, gồm ngày trước/sau bản ghi; Q=0 hợp lệ, Q âm/không hữu hạn là không hợp lệ.

# %% Phần 4.2 - Bản đồ nhiệt thiếu qmax theo năm, tỷ lệ theo kỳ và số lưu vực có dữ liệu
# Dùng lịch năm đầy đủ, gồm ngày trước/sau bản ghi; Q=0 hợp lệ, Q âm/không hữu hạn là không hợp lệ.
# Lưu ma trận mã thiếu và bảng theo năm để các cell độ dài đoạn thiếu dùng lại, không đọc lại CSV.
q_ids = sorted(selected_ids, key=int)
q_spans = A1_Q_FILE_SPANS.set_index("basin_id").loc[q_ids]
q_start = pd.to_datetime(q_spans["file_start"]).min().replace(month=1, day=1)
q_end = pd.to_datetime(q_spans["file_end"]).max().replace(month=12, day=31)
A1_QMAX_AUDIT_CALENDAR = pd.date_range(q_start, q_end, freq="D")
q_years = sorted(A1_QMAX_AUDIT_CALENDAR.year.unique())
year_positions = {year: np.flatnonzero(A1_QMAX_AUDIT_CALENDAR.year == year) for year in q_years}
period_positions = {period: np.flatnonzero((A1_QMAX_AUDIT_CALENDAR >= start) & (A1_QMAX_AUDIT_CALENDAR <= end))
                    for period, (start, end) in PERIODS.items()}
if any(len(period_positions[period]) != len(pd.date_range(start, end)) for period, (start, end) in PERIODS.items()):
    raise ValueError("Lịch Q chưa bao trọn các kỳ; kiểm lại phạm vi tệp ở 2.10.")
A1_QMAX_CODEBOOK = pd.DataFrame({"code": range(7), "meaning": [
    "valid_nonnegative", "missing_row", "blank_cell", "finite_negative_except_minus999",
    "minus999", "non_numeric_nonblank", "infinite"]})


def a1_qmax_code_counts(codes):
    counts = np.bincount(codes, minlength=7)
    return {"expected_days": len(codes), "rows_present": len(codes) - int(counts[1]),
            "valid_days": int(counts[0]), "missing_rows": int(counts[1]), "blank_cells": int(counts[2]),
            "negative_except_minus999": int(counts[3]), "minus999_cells": int(counts[4]),
            "non_numeric_nonblank": int(counts[5]), "infinite_cells": int(counts[6]),
            "invalid_nonblank": int(counts[3:].sum()), "total_missing": len(codes) - int(counts[0])}


codes_matrix = np.zeros((len(q_ids), len(A1_QMAX_AUDIT_CALENDAR)), dtype=np.uint8)
year_rows, period_rows, profile_rows = [], [], []
started_q_p4 = time.perf_counter()
for row_index, basin_id in enumerate(q_ids):
    raw = pd.read_csv(A1_DAILY_FILES["D"][basin_id], sep=";", encoding="utf-8-sig", dtype=str,
                      usecols=DATE_COLS + ["qmax"], keep_default_na=False, na_filter=False)
    dates = pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce")
    if dates.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"ID {basin_id}: ngày Q rỗng/không hợp lệ/trùng/không tăng.")
    if dates.min() < q_start or dates.max() > q_end:
        raise ValueError(f"ID {basin_id}: phạm vi tệp đã đổi; chạy lại kiểm kê 2.10 trước.")
    indexed = raw["qmax"].set_axis(pd.DatetimeIndex(dates))
    text_values = indexed.reindex(A1_QMAX_AUDIT_CALENDAR).str.strip()
    values = pd.to_numeric(text_values, errors="coerce")
    present = A1_QMAX_AUDIT_CALENDAR.isin(indexed.index)
    conditions = [~present, text_values.eq(""), values.eq(-999), np.isfinite(values) & values.lt(0),
                  text_values.notna() & text_values.ne("") & values.isna(), np.isinf(values)]
    codes = np.select(conditions, [1, 2, 4, 3, 5, 6], default=0).astype(np.uint8)
    valid = np.isfinite(values) & values.ge(0)
    if not np.array_equal(codes == 0, valid.to_numpy()):
        raise ValueError(f"ID {basin_id}: mã thiếu chưa khớp định nghĩa Q hợp lệ.")
    codes_matrix[row_index] = codes
    valid_dates = A1_QMAX_AUDIT_CALENDAR[codes == 0]
    profile_rows.append({"basin_id": basin_id, "id_numeric": int(basin_id),
                         "first_valid_date": valid_dates.min(), "last_valid_date": valid_dates.max()})
    for year, positions in year_positions.items():
        year_rows.append({"basin_id": basin_id, "year": year, **a1_qmax_code_counts(codes[positions])})
    for period, positions in period_positions.items():
        period_rows.append({"basin_id": basin_id, "period": period, **a1_qmax_code_counts(codes[positions])})
    if (row_index + 1) % 100 == 0 or row_index + 1 == len(q_ids):
        print(f"Đã lập bảng thiếu qmax {row_index + 1}/{len(q_ids)} lưu vực", flush=True)
A1_QMAX_MISSING_BY_YEAR = pd.DataFrame(year_rows)
A1_QMAX_MISSING_BY_YEAR["missing_pct"] = 100 * A1_QMAX_MISSING_BY_YEAR["total_missing"] / A1_QMAX_MISSING_BY_YEAR["expected_days"]
A1_QMAX_MISSING_BY_PERIOD_BASIN = pd.DataFrame(period_rows)
count_fields = list(a1_qmax_code_counts(np.array([], dtype=np.uint8)))
A1_QMAX_MISSING_BY_PERIOD = A1_QMAX_MISSING_BY_PERIOD_BASIN.groupby("period")[count_fields].sum().reindex(PERIODS)
A1_QMAX_MISSING_BY_PERIOD["missing_pct"] = 100 * A1_QMAX_MISSING_BY_PERIOD["total_missing"] / A1_QMAX_MISSING_BY_PERIOD["expected_days"]
check_fields = ["expected_days", "rows_present", "valid_days", "missing_rows", "blank_cells", "total_missing"]
old_period_counts = A1_Q_MISSING_PERIODS.reindex(PERIODS)[check_fields]
new_period_counts = A1_QMAX_MISSING_BY_PERIOD[check_fields]
A1_QMAX_MISSING_BY_PERIOD["matches_P2_counts"] = new_period_counts.eq(old_period_counts).all(axis=1)
year_summary = A1_QMAX_MISSING_BY_YEAR.groupby("year")[count_fields].sum()
year_summary["basins_with_any_valid_Q"] = A1_QMAX_MISSING_BY_YEAR.groupby("year")["valid_days"].apply(lambda days: int(days.gt(0).sum()))
fully_valid = A1_QMAX_MISSING_BY_YEAR["valid_days"].eq(A1_QMAX_MISSING_BY_YEAR["expected_days"])
year_summary["basins_with_full_year_valid_Q"] = fully_valid.groupby(A1_QMAX_MISSING_BY_YEAR["year"]).sum()
year_summary["missing_pct"] = 100 * year_summary["total_missing"] / year_summary["expected_days"]
A1_QMAX_YEAR_SUMMARY = year_summary
profile = pd.DataFrame(profile_rows).sort_values(["first_valid_date", "id_numeric"], na_position="last").reset_index(drop=True)
profile.insert(0, "row_rank", np.arange(1, len(profile) + 1))
profile["train_valid_days"] = profile["basin_id"].map(
    A1_QMAX_MISSING_BY_PERIOD_BASIN.query("period == 'train'").set_index("basin_id")["valid_days"])
A1_QMAX_HEATMAP_ORDER = profile.drop(columns="id_numeric")
heatmap = A1_QMAX_MISSING_BY_YEAR.pivot(index="basin_id", columns="year", values="missing_pct").reindex(
    index=profile["basin_id"], columns=q_years)
fig = plt.figure(figsize=(17, 15))
grid = fig.add_gridspec(2, 2, height_ratios=[4.4, 1.2], left=0.16, right=0.95,
                        top=0.90, bottom=0.11, hspace=0.29, wspace=0.27)
ax_heat = fig.add_subplot(grid[0, :])
picture = ax_heat.imshow(heatmap.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=100,
                         interpolation="nearest")
tick_rows = np.unique(np.r_[np.arange(0, len(profile), 100), len(profile) - 1])
ax_heat.set_yticks(tick_rows, [f"{i + 1} (ID {profile.iloc[i]['basin_id']})" for i in tick_rows], fontsize=9)
tick_years = np.unique(np.r_[np.arange(0, len(q_years), 3), len(q_years) - 1])
ax_heat.set_xticks(tick_years, [str(q_years[i]) for i in tick_years], fontsize=9)
ax_heat.set(xlabel="Calendar year", ylabel="Row rank (basin ID); sorted by first valid qmax date")
ax_heat.set_title("(a) Missing qmax by basin and calendar year", pad=28)
for period, (start, end) in PERIODS.items():
    position = q_years.index(pd.Timestamp(start).year) - 0.5
    ax_heat.axvline(position, color="#355c7d", linestyle="--", linewidth=0.8)
    ax_heat.text(position + 0.15, 1.01, f"{period.capitalize()} {pd.Timestamp(start).year}",
                 transform=ax_heat.get_xaxis_transform(), fontsize=9)
colorbar = fig.colorbar(picture, ax=ax_heat, fraction=0.018, pad=0.02)
colorbar.set_label("Missing days (%)")
ax_period = fig.add_subplot(grid[1, 0])
bottom = np.zeros(len(PERIODS))
for column, label, color in [("missing_rows", "No row", "#b35435"),
                             ("blank_cells", "Blank cell", "#e2ac56"),
                             ("invalid_nonblank", "Other invalid", "#785b9b")]:
    percentages = 100 * A1_QMAX_MISSING_BY_PERIOD[column].to_numpy() / A1_QMAX_MISSING_BY_PERIOD["expected_days"].to_numpy()
    ax_period.bar(np.arange(len(PERIODS)), percentages, bottom=bottom, color=color, label=label, width=0.6)
    bottom += percentages
for i, value in enumerate(bottom):
    ax_period.annotate(f"{value:.3f}%", (i, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
ax_period.set_xticks(np.arange(len(PERIODS)), [period.capitalize() for period in PERIODS])
ax_period.set(title="(b) Missing qmax by experiment period", ylabel="Expected days missing (%)")
ax_period.set_ylim(0, max(1, bottom.max() * 1.35))
ax_period.legend(fontsize=8, frameon=False)
ax_period.grid(axis="y", alpha=0.18)
ax_year = fig.add_subplot(grid[1, 1])
ax_year.plot(year_summary.index, year_summary["basins_with_any_valid_Q"], color="#1976a2", label="At least one valid day")
ax_year.plot(year_summary.index, year_summary["basins_with_full_year_valid_Q"], color="#8b6633", label="Every calendar day valid")
ax_year.set(title="(c) Basins with valid qmax", xlabel="Calendar year", ylabel="Number of basins",
            ylim=(0, len(q_ids) + 10))
ax_year.legend(fontsize=8, frameon=False, loc="lower right")
ax_year.grid(alpha=0.18)
fig.suptitle(f"Daily maximum discharge: missingness in {len(q_ids)} selected basins", fontsize=16, y=0.97)
fig.text(0.5, 0.035, "Full calendar years include days before and after each station record. Zero discharge remains valid.\n"
         "Heatmap scale: 0–100%. Annual availability counts basins; period rates count expected basin-days.",
         ha="center", va="center", fontsize=10)
q_missing_figure = FIGURE_DIR / "A1_P4_qmax_missing_overview.png"
fig.savefig(q_missing_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
code_order = [q_ids.index(basin_id) for basin_id in profile["basin_id"]]
A1_QMAX_MISSING_CODES = codes_matrix[code_order]
np.savez_compressed(TABLE_DIR / "A1_P4_qmax_missing_codes.npz",
                    basin_ids=profile["basin_id"].to_numpy(dtype="U"),
                    dates=A1_QMAX_AUDIT_CALENDAR.to_numpy(dtype="datetime64[D]"), codes=A1_QMAX_MISSING_CODES)
for suffix, table in [("qmax_missing_by_year", A1_QMAX_MISSING_BY_YEAR),
                      ("qmax_missing_by_period_basin", A1_QMAX_MISSING_BY_PERIOD_BASIN),
                      ("qmax_missing_by_period", A1_QMAX_MISSING_BY_PERIOD.reset_index()),
                      ("qmax_year_summary", A1_QMAX_YEAR_SUMMARY.reset_index()),
                      ("qmax_heatmap_order", A1_QMAX_HEATMAP_ORDER), ("qmax_codebook", A1_QMAX_CODEBOOK)]:
    table.to_csv(TABLE_DIR / f"A1_P4_{suffix}.csv", index=False, encoding="utf-8-sig")
print("(1) Thiếu qmax theo kỳ và đối chiếu số đếm ở Phần 2:")
with pd.option_context("display.max_columns", None):
    display(A1_QMAX_MISSING_BY_PERIOD)
if not A1_QMAX_MISSING_BY_PERIOD["matches_P2_counts"].all():
    print("CẢNH BÁO: số đếm khác Phần 2; giữ số đo mới để kiểm, chưa ép khớp.")
print("(2) Theo năm; có ít nhất một ngày khác với đủ cả năm:")
display(A1_QMAX_YEAR_SUMMARY[["expected_days", "valid_days", "total_missing", "missing_pct",
                              "basins_with_any_valid_Q", "basins_with_full_year_valid_Q"]])
print("(3) Mười hai hàng cuối của thứ tự vẽ; toàn bộ 451 hàng lưu ở CSV:")
display(A1_QMAX_HEATMAP_ORDER.tail(12))
print("Số ID có qmax đầu tiên sau đầu train:", int(profile["first_valid_date"].gt(pd.Timestamp(PERIODS["train"][0])).sum()))
print("ID không có qmax hợp lệ trong train:", sorted(profile.loc[profile["train_valid_days"].eq(0), "basin_id"], key=int))
print("Đã lưu hình:", q_missing_figure)
print("Đã lưu sáu CSV và cache mã thiếu NPZ vào:", TABLE_DIR)
print(f"Thời gian quét: {time.perf_counter() - started_q_p4:.1f} giây.")

# %% [markdown]
# ### 4.3. Độ dài đoạn thiếu qmax và dòng thời gian 27 lưu vực bắt đầu muộn
#
# Dùng ma trận 4.2; một đoạn gồm các ngày thiếu liền nhau, kể cả khi mã nguyên nhân đổi.

# %% Phần 4.3 - Độ dài đoạn thiếu qmax và dòng thời gian 27 lưu vực bắt đầu muộn
# Dùng ma trận 4.2; một đoạn gồm các ngày thiếu liền nhau, kể cả khi mã nguyên nhân đổi.
# Tách vị trí trước/giữa/sau các ngày Q hợp lệ; vị trí không tự xác nhận nguyên nhân thiếu.
import matplotlib.dates as mdates

run_calendar = pd.DatetimeIndex(A1_QMAX_AUDIT_CALENDAR)
run_profile = A1_QMAX_HEATMAP_ORDER.copy().reset_index(drop=True)
run_ids = run_profile["basin_id"].astype(str).to_numpy()
run_codes = np.asarray(A1_QMAX_MISSING_CODES)
if (run_codes.shape != (len(run_ids), len(run_calendar)) or len(set(run_ids)) != len(run_ids)
        or set(run_ids) != set(selected_ids) or not run_calendar.equals(pd.date_range(run_calendar[0], run_calendar[-1]))
        or not np.isin(run_codes, np.arange(7)).all()):
    raise ValueError("Cache Q chưa khớp ID/lịch/mã của 4.2; chạy lại 4.2 trước.")


def a1_missing_runs(codes):
    """Trả [start, stop), vị trí so với Q hợp lệ; không cắt ở ranh giới năm/kỳ."""
    codes = np.asarray(codes)
    valid_positions = np.flatnonzero(codes == 0)
    changes = np.diff(np.r_[False, codes != 0, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        position = ("no_valid_Q" if not valid_positions.size else
                    "before_first_valid" if stop <= valid_positions[0] else
                    "after_last_valid" if start > valid_positions[-1] else "between_valid_dates")
        yield int(start), int(stop), position


# 4.3a. Đếm đoạn trên toàn lịch; kiểm cả tổng ngày lẫn mốc hợp lệ của từng hàng cache.
started_runs = time.perf_counter()
run_rows, overlap_rows = [], []
for row_index, basin_id in enumerate(run_ids):
    codes = run_codes[row_index]
    valid_dates = run_calendar[codes == 0]
    for column, measured in [("first_valid_date", valid_dates.min()), ("last_valid_date", valid_dates.max())]:
        stored = pd.to_datetime(run_profile.at[row_index, column])
        if not ((pd.isna(stored) and pd.isna(measured)) or stored == measured):
            raise ValueError(f"ID {basin_id}: cache và mốc {column} khác nhau; chạy lại 4.2.")
    for start, stop, position in a1_missing_runs(codes):
        run_id = len(run_rows) + 1
        counts = a1_qmax_code_counts(codes[start:stop])
        run_rows.append({"run_id": run_id, "basin_id": basin_id, "position": position,
                         "start_date": run_calendar[start], "end_date": run_calendar[stop - 1],
                         "run_days": stop - start, "touches_calendar_start": start == 0,
                         "touches_calendar_end": stop == len(run_calendar),
                         **{name: counts[name] for name in ["missing_rows", "blank_cells", "invalid_nonblank"]}})
        for period, (period_start, period_end) in PERIODS.items():
            overlap_start = max(run_calendar[start], pd.Timestamp(period_start))
            overlap_end = min(run_calendar[stop - 1], pd.Timestamp(period_end))
            if overlap_start <= overlap_end:
                overlap_rows.append({"run_id": run_id, "basin_id": basin_id, "period": period,
                                     "position": position, "full_run_days": stop - start,
                                     "overlap_start": overlap_start, "overlap_end": overlap_end,
                                     "overlap_days": (overlap_end - overlap_start).days + 1,
                                     "continues_before_period": run_calendar[start] < pd.Timestamp(period_start),
                                     "continues_after_period": run_calendar[stop - 1] > pd.Timestamp(period_end)})
A1_QMAX_MISSING_RUNS = pd.DataFrame(run_rows, columns=[
    "run_id", "basin_id", "position", "start_date", "end_date", "run_days", "touches_calendar_start",
    "touches_calendar_end", "missing_rows", "blank_cells", "invalid_nonblank"])
A1_QMAX_RUN_PERIOD_OVERLAPS = pd.DataFrame(overlap_rows, columns=[
    "run_id", "basin_id", "period", "position", "full_run_days", "overlap_start", "overlap_end",
    "overlap_days", "continues_before_period", "continues_after_period"])
if int(A1_QMAX_MISSING_RUNS["run_days"].sum()) != int((run_codes != 0).sum()):
    raise ValueError("Tổng ngày của các đoạn chưa khớp ma trận thiếu.")
run_summary = A1_QMAX_MISSING_RUNS.groupby("position").agg(
    runs=("run_id", "size"), basins=("basin_id", "nunique"), missing_days=("run_days", "sum"),
    median_run_days=("run_days", "median"), p95_run_days=("run_days", lambda x: x.quantile(0.95)),
    max_run_days=("run_days", "max"), missing_rows=("missing_rows", "sum"), blank_cells=("blank_cells", "sum"),
    invalid_nonblank=("invalid_nonblank", "sum"))
A1_QMAX_RUN_SUMMARY = run_summary
period_check = A1_QMAX_RUN_PERIOD_OVERLAPS.groupby("period")["overlap_days"].sum().reindex(PERIODS, fill_value=0).to_frame()
period_check["missing_days_from_4_2"] = A1_QMAX_MISSING_BY_PERIOD["total_missing"]
period_check["matches"] = period_check["overlap_days"].eq(period_check["missing_days_from_4_2"])
if not period_check["matches"].all():
    raise ValueError("Ngày giao các đoạn với kỳ chưa khớp 4.2.")
A1_QMAX_RUN_PERIOD_CHECK = period_check
internal_runs = A1_QMAX_MISSING_RUNS.query("position == 'between_valid_dates'").sort_values(
    ["run_days", "start_date"], ascending=[False, True])
no_q_years = A1_QMAX_MISSING_BY_YEAR.loc[A1_QMAX_MISSING_BY_YEAR["valid_days"].eq(0)].merge(
    run_profile[["basin_id", "first_valid_date", "last_valid_date"]], on="basin_id", validate="many_to_one")
no_q_years["whole_year_between_valid_dates"] = (
    pd.to_datetime(no_q_years["year"].astype(str) + "-01-01").ge(no_q_years["first_valid_date"])
    & pd.to_datetime(no_q_years["year"].astype(str) + "-12-31").le(no_q_years["last_valid_date"]))
A1_QMAX_NO_VALID_YEAR = no_q_years

# 4.3b. Histogram đếm số đoạn, phân vị trí; từng ô dùng bin nguyên trên thang log, không cắt đuôi.
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
positions_and_labels = [("before_first_valid", "Before first valid qmax"),
                        ("between_valid_dates", "Between valid qmax dates"),
                        ("after_last_valid", "After last valid qmax")]
for ax, (position, label) in zip(axes, positions_and_labels):
    lengths = A1_QMAX_MISSING_RUNS.loc[A1_QMAX_MISSING_RUNS["position"].eq(position), "run_days"].to_numpy(dtype=int)
    if lengths.size > 1:
        edges = np.unique(np.ceil(np.geomspace(1, lengths.max() + 1, 26))) - 0.5
        counts, edges = np.histogram(lengths, bins=edges)
        ax.stairs(counts, edges, color="#1976a2", linewidth=1.6)
        ax.set_xscale("log")
        ax.text(0.5, -0.34, f"{len(lengths)} runs; {lengths.sum()} missing basin-days; max {lengths.max()} days",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
    elif lengths.size == 1:
        ax.text(0.5, 0.57, f"One missing run\n{lengths[0]} days", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        one_run = A1_QMAX_MISSING_RUNS.loc[A1_QMAX_MISSING_RUNS["position"].eq(position)].iloc[0]
        ax.text(0.5, 0.32, f"ID {one_run.basin_id}\n{one_run.start_date:%Y-%m-%d} to {one_run.end_date:%Y-%m-%d}",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, "No runs", transform=ax.transAxes, ha="center")
    ax.set(title=label, xlabel="Consecutive missing days (log axis)", ylabel="Number of runs")
    if lengths.size <= 1:
        ax.set(xlabel="", ylabel="")
    ax.grid(alpha=0.18)
fig.suptitle("Missing qmax runs: positions relative to each basin's valid dates", fontsize=14)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.37, top=0.80, wspace=0.28)
fig.text(0.5, 0.025, "Daily runs are not split at year, period or missing-code boundaries. Panel scales and bins may differ.\n"
         "Runs touching the audit-calendar edges may extend beyond it. Position does not establish a cause.", ha="center", fontsize=9)
runs_figure = FIGURE_DIR / "A1_P4_qmax_missing_run_lengths.png"
fig.savefig(runs_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

# 4.3c. Dòng thời gian: khoảng đầu/cuối có Q là nền xanh; ngày thiếu ở giữa tô đỏ theo cache.
late_profile = run_profile.loc[run_profile["first_valid_date"].gt(pd.Timestamp(PERIODS["train"][0]))].copy()
A1_QMAX_LATE_TIMELINE = late_profile
if len(late_profile) != 27:
    print(f"CẢNH BÁO: lần này có {len(late_profile)} lưu vực bắt đầu muộn, khác số 27 trước đó.")
fig, ax = plt.subplots(figsize=(14, max(5, len(late_profile) * 0.30 + 2.5)))
period_colors = ["#e7f1db", "#fff0d8", "#fce1e5"]
for (period, (start, end)), color in zip(PERIODS.items(), period_colors):
    ax.axvspan(mdates.date2num(pd.Timestamp(start)), mdates.date2num(pd.Timestamp(end) + pd.Timedelta(days=1)),
               color=color, alpha=0.7, zorder=0)
    midpoint = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ax.text(mdates.date2num(midpoint), 1.01, period.capitalize(), transform=ax.get_xaxis_transform(), ha="center", fontsize=10)
for y, row in enumerate(late_profile.itertuples(index=False)):
    left = mdates.date2num(row.first_valid_date)
    right = mdates.date2num(row.last_valid_date + pd.Timedelta(days=1))
    ax.barh(y, right - left, left=left, height=0.60, color="#438bb0", zorder=2)
    for gap in internal_runs.loc[internal_runs["basin_id"].eq(row.basin_id)].itertuples(index=False):
        gap_left = mdates.date2num(gap.start_date)
        gap_right = mdates.date2num(gap.end_date + pd.Timedelta(days=1))
        ax.barh(y, gap_right - gap_left, left=gap_left, height=0.60, color="#b83d38", zorder=3)
        if gap.run_days < 7:
            ax.plot((gap_left + gap_right) / 2, y, marker="|", color="#b83d38", markersize=11,
                    markeredgewidth=1.6, linestyle="none", zorder=4)
    ax.annotate(row.first_valid_date.strftime("%Y-%m-%d"), (left, y), xytext=(-5, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8)
ax.set_yticks(np.arange(len(late_profile)), [f"ID {row.basin_id} | train Q: {row.train_valid_days} days"
                                              for row in late_profile.itertuples(index=False)], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(mdates.date2num(pd.Timestamp(PERIODS["train"][0]) - pd.Timedelta(days=550)),
            mdates.date2num(pd.Timestamp(PERIODS["test"][1]) + pd.Timedelta(days=100)))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlabel("Calendar year")
ax.grid(axis="x", alpha=0.2)
ax.legend(handles=[Patch(facecolor="#438bb0", label="First-to-last valid qmax span"),
                   Patch(facecolor="#b83d38", label="Missing days inside the span"),
                   Line2D([], [], marker="|", color="#b83d38", linestyle="none", markersize=10,
                          label="Gap under 7 days (marker width is symbolic)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=9)
fig.suptitle(f"{len(late_profile)} basins with first valid qmax after the training-period start", y=0.98, fontsize=14)
fig.subplots_adjust(left=0.23, right=0.99, top=0.91, bottom=0.18)
fig.text(0.5, 0.025, "Dates label the first valid qmax. Red bars show actual missing-day spans; ticks make gaps under 7 days visible.\n"
         "Train-Q counts are valid labels, not counts of usable training windows. Zero qmax remains valid.", ha="center", fontsize=9)
late_figure = FIGURE_DIR / "A1_P4_late_gauges_timeline.png"
fig.savefig(late_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
for suffix, table in [("qmax_missing_runs", A1_QMAX_MISSING_RUNS), ("qmax_run_summary", run_summary.reset_index()),
                      ("qmax_run_period_overlaps", A1_QMAX_RUN_PERIOD_OVERLAPS),
                      ("qmax_run_period_check", period_check.reset_index()),
                      ("qmax_no_valid_year", no_q_years), ("late_gauges_timeline", late_profile)]:
    table.to_csv(TABLE_DIR / f"A1_P4_{suffix}.csv", index=False, encoding="utf-8-sig")
print("(1) Đoạn thiếu trên toàn lịch: vị trí là trước/giữa/sau Q hợp lệ, không phải nguyên nhân:")
display(run_summary)
print("(2) Ngày giao các đoạn với từng kỳ; một đoạn có thể đi qua ranh giới kỳ:")
display(period_check)
print("(3) Tối đa 15 đoạn thiếu bên trong dài nhất; mã thiếu khác nhau vẫn thuộc cùng đoạn:")
display(internal_runs.head(15))
print("(4) Các ID không có Q hợp lệ trong cả năm 2010 hoặc 2014, kèm mốc đầu/cuối toàn chuỗi:")
display(no_q_years.loc[no_q_years["year"].isin([2010, 2014]), [
    "basin_id", "year", "expected_days", "missing_rows", "blank_cells", "invalid_nonblank",
    "first_valid_date", "last_valid_date", "whole_year_between_valid_dates"]])
print("Số lưu vực trong timeline:", len(late_profile))
print("Số đoạn thuộc chuỗi không có Q hợp lệ:", int(A1_QMAX_MISSING_RUNS["position"].eq("no_valid_Q").sum()))
print("Đã lưu hai hình:", runs_figure, "|", late_figure)
print("Đã lưu sáu CSV vào:", TABLE_DIR)
print(f"Thời gian phân tích cache: {time.perf_counter() - started_runs:.1f} giây.")

# %% [markdown]
# ### 4.4. EOBS_hu: thiếu theo năm, tháng và độ dài đoạn liên tiếp
#
# Tái sử dụng danh sách ô thiếu 4.1; mẫu số năm/tháng là số ngày lịch thật nhân 451 lưu vực.

# %% Phần 4.4 - EOBS_hu: thiếu theo năm, tháng và độ dài đoạn liên tiếp
# Tái sử dụng danh sách ô thiếu 4.1; mẫu số năm/tháng là số ngày lịch thật nhân 451 lưu vực.
# Không làm trơn, điền khuyết hoặc đổi số ô thiếu thành số mẫu huấn luyện bị loại.
started_hu = time.perf_counter()
hu_variable = "EOBS_hu"
hu_ids = sorted(selected_ids, key=int)
hu_start = min(pd.Timestamp(start) for start, end in PERIODS.values())
hu_end = max(pd.Timestamp(end) for start, end in PERIODS.values())
hu_calendar = pd.date_range(hu_start, hu_end, freq="D")
period_days = [pd.date_range(start, end) for start, end in PERIODS.values()]
if not pd.DatetimeIndex(np.concatenate([days.to_numpy() for days in period_days])).equals(hu_calendar):
    raise ValueError("Các kỳ chưa tạo thành một lịch liên tiếp không trùng; không được nối qua khoảng không kiểm.")
hu_cells = A1_FORCING_MISSING_CELLS.loc[A1_FORCING_MISSING_CELLS["variable"].eq(hu_variable)].copy()
hu_cells["basin_id"] = hu_cells["basin_id"].astype(str)
hu_cells["date"] = pd.to_datetime(hu_cells["date"], errors="coerce")
if (hu_cells["date"].isna().any() or hu_cells.duplicated(["basin_id", "date"]).any()
        or not set(hu_cells["basin_id"]).issubset(selected_ids)
        or not hu_cells["date"].isin(hu_calendar).all()):
    raise ValueError("Danh sách EOBS_hu có ngày/ID không hợp lệ, trùng hoặc ngoài lịch 4.1.")
for period, (start, end) in PERIODS.items():
    dated = hu_cells["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    if not hu_cells["period"].eq(period).equals(dated):
        raise ValueError(f"Nhãn kỳ {period} không khớp ngày trong danh sách thiếu 4.1.")
hu_reference = A1_FORCING_MISSING_BY_VARIABLE.loc[
    A1_FORCING_MISSING_BY_VARIABLE["variable"].eq(hu_variable)].set_index("period").reindex(PERIODS)
if (hu_reference["total_missing"].isna().any() or
        not hu_reference["total_missing"].eq(hu_cells.groupby("period").size().reindex(PERIODS, fill_value=0)).all()):
    raise ValueError("Danh sách EOBS_hu thiếu chưa khớp các tổng của 4.1.")


def a1_true_runs(mask):
    """Các khoảng [start, stop) có mask=True liên tục trên một lịch đều."""
    changes = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


# 4.4a. Đếm theo năm và tháng; tỷ lệ tháng đã hiệu chỉnh số ngày 28/29/30/31.
hu_years = sorted(hu_calendar.year.unique())
hu_year = pd.DataFrame(index=pd.Index(hu_years, name="year"))
hu_year["expected_cells"] = [int((hu_calendar.year == year).sum()) * len(hu_ids) for year in hu_years]
hu_year["missing_cells"] = hu_cells.groupby(hu_cells["date"].dt.year).size().reindex(hu_years, fill_value=0)
hu_year["affected_basins"] = hu_cells.groupby(hu_cells["date"].dt.year)["basin_id"].nunique().reindex(hu_years, fill_value=0)
hu_year["missing_pct"] = 100 * hu_year["missing_cells"] / hu_year["expected_cells"]
hu_month = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
hu_month["expected_cells"] = [int((hu_calendar.month == month).sum()) * len(hu_ids) for month in range(1, 13)]
hu_month["missing_cells"] = hu_cells.groupby(hu_cells["date"].dt.month).size().reindex(range(1, 13), fill_value=0)
hu_month["affected_basins"] = hu_cells.groupby(hu_cells["date"].dt.month)["basin_id"].nunique().reindex(range(1, 13), fill_value=0)
hu_month["missing_pct"] = 100 * hu_month["missing_cells"] / hu_month["expected_cells"]
A1_HU_MISSING_BY_YEAR = hu_year
A1_HU_MISSING_BY_MONTH = hu_month

# 4.4b. Không cắt đoạn ở ranh giới kỳ; lưu riêng số ngày giao để so lại 4.1.
hu_runs, hu_profiles, hu_intersections = [], [], []
hu_dates_by_id = {basin_id: group["date"] for basin_id, group in hu_cells.groupby("basin_id")}
for basin_id in hu_ids:
    missing = hu_calendar.isin(hu_dates_by_id.get(basin_id, []))
    intervals = a1_true_runs(missing)
    hu_profiles.append({"basin_id": basin_id, "expected_cells": len(hu_calendar),
                        "missing_cells": int(missing.sum()), "runs": len(intervals),
                        "longest_run_days": max((stop - start for start, stop in intervals), default=0)})
    for start, stop in intervals:
        run_id = len(hu_runs) + 1
        hu_runs.append({"run_id": run_id, "basin_id": basin_id, "variable": hu_variable,
                        "start_date": hu_calendar[start], "end_date": hu_calendar[stop - 1],
                        "run_days": int(stop - start), "touches_audit_start": start == 0,
                        "touches_audit_end": stop == len(hu_calendar)})
        for period, (p_start, p_end) in PERIODS.items():
            first = max(hu_calendar[start], pd.Timestamp(p_start))
            last = min(hu_calendar[stop - 1], pd.Timestamp(p_end))
            if first <= last:
                hu_intersections.append({"run_id": run_id, "basin_id": basin_id, "period": period,
                                         "overlap_days": (last - first).days + 1})
A1_HU_MISSING_RUNS = pd.DataFrame(hu_runs, columns=["run_id", "basin_id", "variable", "start_date", "end_date",
                                                "run_days", "touches_audit_start", "touches_audit_end"])
A1_HU_MISSING_PROFILE = pd.DataFrame(hu_profiles)
A1_HU_MISSING_PROFILE["missing_pct"] = 100 * A1_HU_MISSING_PROFILE["missing_cells"] / len(hu_calendar)
A1_HU_RUN_PERIOD_OVERLAPS = pd.DataFrame(hu_intersections, columns=["run_id", "basin_id", "period", "overlap_days"])
hu_check = A1_HU_RUN_PERIOD_OVERLAPS.groupby("period")["overlap_days"].sum().reindex(PERIODS, fill_value=0).to_frame()
hu_check["missing_cells_from_4_1"] = hu_reference["total_missing"]
hu_check["matches"] = hu_check["overlap_days"].eq(hu_check["missing_cells_from_4_1"])
A1_HU_RUN_PERIOD_CHECK = hu_check
if (not hu_check["matches"].all() or int(A1_HU_MISSING_RUNS["run_days"].sum()) != len(hu_cells)
        or int(hu_year["missing_cells"].sum()) != len(hu_cells) or int(hu_month["missing_cells"].sum()) != len(hu_cells)):
    raise ValueError("Tổng đoạn/năm/tháng EOBS_hu chưa khớp danh sách và tổng theo kỳ 4.1.")
hu_lengths = A1_HU_MISSING_RUNS["run_days"].to_numpy(dtype=int)
hu_summary = pd.DataFrame([{"variable": hu_variable, "basins": len(hu_ids),
    "affected_basins": int(A1_HU_MISSING_PROFILE["missing_cells"].gt(0).sum()), "missing_cells": len(hu_cells),
    "runs": len(hu_lengths), "median_run_days": float(np.median(hu_lengths)) if hu_lengths.size else np.nan,
    "p95_run_days": float(np.quantile(hu_lengths, 0.95)) if hu_lengths.size else np.nan,
    "max_run_days": int(hu_lengths.max()) if hu_lengths.size else 0,
    "runs_touching_audit_edges": int((A1_HU_MISSING_RUNS["touches_audit_start"]
                                     | A1_HU_MISSING_RUNS["touches_audit_end"]).sum())}])
A1_HU_RUN_SUMMARY = hu_summary

# 4.4c. Ba ô bổ sung: diễn biến từng năm, tháng đã hiệu chỉnh mẫu số, phân phối độ dài đoạn.
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
axes[0].bar(hu_year.index, hu_year["missing_pct"], color="#217da6", width=0.8)
axes[0].set(title="(a) Missing EOBS_hu by calendar year", xlabel="Calendar year", ylabel="Expected cells missing (%)")
axes[0].set_xticks(hu_years[::2])
axes[1].bar(hu_month.index, hu_month["missing_pct"], color="#217da6", width=0.8)
axes[1].set(title="(b) Missing EOBS_hu by calendar month", xlabel="Calendar month", ylabel="Expected cells missing (%)")
axes[1].set_xticks(range(1, 13))
if hu_lengths.size:
    edges = np.unique(np.ceil(np.geomspace(1, hu_lengths.max() + 1, 25))) - 0.5
    counts, edges = np.histogram(hu_lengths, bins=edges)
    axes[2].stairs(counts, edges, color="#217da6", linewidth=1.6)
    axes[2].set_xscale("log")
axes[2].set(title="(c) Consecutive missing EOBS_hu days", xlabel="Run length (days; log axis)", ylabel="Number of runs")
for ax in axes:
    ax.grid(axis="y", alpha=0.18)
fig.suptitle(f"EOBS_hu missingness: {len(hu_ids)} selected basins, {hu_start.year}–{hu_end.year}", fontsize=15, y=0.98)
fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.24, wspace=0.30)
fig.text(0.5, 0.045, "Rates use actual calendar days times the number of selected basins; months are pooled across the audit years.\n"
         "Daily runs continue across year and period boundaries. Missing cells are not counts of lost training samples.",
         ha="center", fontsize=9)
hu_figure = FIGURE_DIR / "A1_P4_EOBS_hu_missing_patterns.png"
fig.savefig(hu_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
for suffix, table in [("EOBS_hu_missing_by_year", hu_year.reset_index()), ("EOBS_hu_missing_by_month", hu_month.reset_index()),
                      ("EOBS_hu_missing_runs", A1_HU_MISSING_RUNS), ("EOBS_hu_missing_profile", A1_HU_MISSING_PROFILE),
                      ("EOBS_hu_run_period_overlaps", A1_HU_RUN_PERIOD_OVERLAPS),
                      ("EOBS_hu_run_period_check", hu_check.reset_index()), ("EOBS_hu_run_summary", hu_summary)]:
    table.to_csv(TABLE_DIR / f"A1_P4_{suffix}.csv", index=False, encoding="utf-8-sig")
print("(1) Tổng hợp đoạn thiếu EOBS_hu trên lịch các kỳ ghép liền; chưa xét ngày khởi động trước 2003:")
display(hu_summary)
print("(2) Tỷ lệ thiếu theo năm; affected_basins đếm ID có ít nhất một ô thiếu:")
display(hu_year)
print("(3) Tỷ lệ theo tháng, mẫu số dùng đúng số ngày lịch của tháng trong 2003–2017:")
display(hu_month)
print("(4) Tối đa 15 đoạn liên tiếp dài nhất:")
display(A1_HU_MISSING_RUNS.sort_values(["run_days", "start_date"], ascending=[False, True]).head(15))
print("(5) Đối chiếu số ngày giao từng kỳ với 4.1:")
display(hu_check)
print("Đã lưu hình:", hu_figure)
print("Đã lưu bảy CSV vào:", TABLE_DIR)
print(f"Thời gian phân tích danh sách thiếu: {time.perf_counter() - started_hu:.1f} giây.")

# %% [markdown]
# ### 4.5. Toàn bộ gaps: đoạn giờ trước xử lý, Q ngày của 451 ID và ca ID 410
#
# Ngưỡng <=6/>6 giờ theo bài LamaH-CE; phân loại từ danh sách trước xử lý, không xác nhận đã nội suy.

# %% Phần 4.5 - Toàn bộ gaps: đoạn giờ trước xử lý, Q ngày của 451 ID và ca ID 410
# Ngưỡng <=6/>6 giờ theo bài LamaH-CE; phân loại từ danh sách trước xử lý, không xác nhận đã nội suy.
# Ghép theo ngày ghi trong tệp, không dịch múi giờ; chưa đo ảnh hưởng lên giá trị/đỉnh qmax.
started_gaps = time.perf_counter()
gap_paths = sorted(path for path in GAP_DIR.glob("*.csv") if not a1_is_metadata(path))
if not gap_paths:
    raise FileNotFoundError(f"Không có CSV trong {GAP_DIR}.")
gap_file_rows, gap_run_rows, gap_day_parts, seen_gap_ids = [], [], [], set()
for file_number, path in enumerate(gap_paths, 1):
    match = ID_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Tên tệp gaps không có ID theo quy ước: {path.name}")
    basin_id = str(int(match.group(1)))
    if basin_id in seen_gap_ids or basin_id not in A1_DAILY_FILES["D"]:
        raise ValueError(f"ID gaps trùng hoặc không có tệp Q ngày: {basin_id}")
    seen_gap_ids.add(basin_id)
    raw = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", keep_default_na=False, na_filter=False)
    if not set(GAP_TIME_COLUMNS).issubset(raw.columns):
        raise ValueError(f"ID {basin_id}: header gaps khác YYYY/MM/DD/hh/mm.")
    numeric = raw[list(GAP_TIME_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(numeric.to_numpy()).all() or not numeric.eq(np.floor(numeric)).all().all()
            or not numeric["hh"].between(0, 23).all() or not numeric["mm"].eq(0).all()):
        raise ValueError(f"ID {basin_id}: thời gian rỗng/không nguyên/không đúng giờ; chưa phân đoạn trên lưới giờ.")
    parsed = pd.DatetimeIndex(pd.to_datetime(numeric.rename(columns=GAP_TIME_COLUMNS), errors="coerce"))
    if parsed.isna().any():
        raise ValueError(f"ID {basin_id}: có ngày gaps không hợp lệ.")
    duplicates = int(parsed.duplicated().sum())
    times = parsed.drop_duplicates().sort_values()
    starts = np.r_[0, np.flatnonzero(np.diff(times.asi8) != pd.Timedelta(hours=1).value) + 1] if len(times) else np.array([], dtype=int)
    stops = np.r_[starts[1:], len(times)] if len(times) else np.array([], dtype=int)
    lengths = stops - starts
    long_flags = np.repeat(lengths > 6, lengths)
    for first, stop in zip(starts, stops):
        if (times[stop - 1] - times[first]) / pd.Timedelta(hours=1) + 1 != stop - first:
            raise ValueError(f"ID {basin_id}: số mốc không khớp độ dài đoạn giờ.")
        gap_run_rows.append({"basin_id": basin_id, "selected": basin_id in selected_ids,
                             "start_time": times[first], "end_time": times[stop - 1],
                             "run_hours": int(stop - first), "length_group": "up_to_6h" if stop - first <= 6 else "over_6h"})
    daily = pd.DataFrame({"date": times.normalize(), "gap_hours": np.ones(len(times), dtype=int),
                          "short_run_hours": (~long_flags).astype(int), "long_run_hours": long_flags.astype(int)})
    daily = daily.groupby("date", as_index=False)[["gap_hours", "short_run_hours", "long_run_hours"]].sum()
    daily.insert(0, "basin_id", basin_id)
    daily["selected"] = basin_id in selected_ids
    gap_day_parts.append(daily)
    gap_file_rows.append({"basin_id": basin_id, "selected": basin_id in selected_ids, "raw_rows": len(raw),
                          "unique_hours": len(times), "duplicate_rows": duplicates,
                          "raw_sorted": parsed.is_monotonic_increasing, "first_time": times.min(), "last_time": times.max(),
                          "runs": len(lengths), "short_runs": int((lengths <= 6).sum()), "long_runs": int((lengths > 6).sum()),
                          "short_run_hours": int((~long_flags).sum()), "long_run_hours": int(long_flags.sum()),
                          "max_run_hours": int(lengths.max()) if lengths.size else 0})
    if file_number % 50 == 0 or file_number == len(gap_paths):
        print(f"Đã đọc gaps {file_number}/{len(gap_paths)} tệp", flush=True)
A1_GAPS_FILE_PROFILE = pd.DataFrame(gap_file_rows)
A1_GAPS_HOURLY_RUNS = pd.DataFrame(gap_run_rows, columns=[
    "basin_id", "selected", "start_time", "end_time", "run_hours", "length_group"])
A1_GAPS_DAILY = pd.concat(gap_day_parts, ignore_index=True)
if (A1_GAPS_DAILY["gap_hours"].gt(24).any()
        or int(A1_GAPS_FILE_PROFILE["unique_hours"].sum()) != int(A1_GAPS_HOURLY_RUNS["run_hours"].sum())
        or int(A1_GAPS_DAILY["gap_hours"].sum()) != int(A1_GAPS_FILE_PROFILE["unique_hours"].sum())):
    raise ValueError("Tổng giờ của tệp/đoạn/ngày không khớp hoặc có ngày hơn 24 mốc giờ duy nhất.")
if len(gap_paths) != 230 or A1_GAPS_FILE_PROFILE["duplicate_rows"].sum():
    print("CẢNH BÁO: số tệp khác 230 hoặc có dòng trùng; xem bảng kiểm kê. Đoạn chỉ tính trên mốc duy nhất.")

# 4.5a. Đối chiếu Q ngày trong cache 4.2; các ngày không có mốc gaps ghi 'no_listed_gap'.
gap_calendar = pd.DatetimeIndex(A1_QMAX_AUDIT_CALENDAR)
gap_cache_ids = A1_QMAX_HEATMAP_ORDER["basin_id"].astype(str).to_numpy()
if A1_QMAX_MISSING_CODES.shape != (len(gap_cache_ids), len(gap_calendar)) or set(gap_cache_ids) != set(selected_ids):
    raise ValueError("Cache Q/ID 4.2 không khớp; chạy lại 4.2.")
A1_GAPS_DAILY["qmax_code"] = -2  # -2: chưa kiểm Q; -1: ngoài lịch Q được đối chiếu.
for row_index, basin_id in enumerate(gap_cache_ids):
    mask = A1_GAPS_DAILY["basin_id"].eq(basin_id)
    positions = gap_calendar.get_indexer(A1_GAPS_DAILY.loc[mask, "date"])
    values = np.full(len(positions), -1, dtype=int)
    inside = positions >= 0
    values[inside] = A1_QMAX_MISSING_CODES[row_index, positions[inside]]
    A1_GAPS_DAILY.loc[mask, "qmax_code"] = values
gap_classes = ["no_listed_gap", "short_runs_only", "includes_over_6h_run"]
cross_rows = []
for period, (start, end) in PERIODS.items():
    days = A1_GAPS_DAILY.loc[A1_GAPS_DAILY["selected"] & A1_GAPS_DAILY["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if days["qmax_code"].lt(0).any():
        raise ValueError("Ngày gaps trong kỳ nằm ngoài cache Q; không ghép thiếu thành Q thiếu.")
    days["gap_class"] = np.where(days["long_run_hours"].gt(0), gap_classes[2], gap_classes[1])
    counts = []
    for category in gap_classes[1:]:
        group = days.loc[days["gap_class"].eq(category)]
        counts.append({"period": period, "gap_class": category, "valid_qmax_days": int(group["qmax_code"].eq(0).sum()),
                       "missing_qmax_days": int(group["qmax_code"].gt(0).sum()),
                       "no_row_days": int(group["qmax_code"].eq(1).sum()),
                       "blank_qmax_days": int(group["qmax_code"].eq(2).sum()),
                       "other_invalid_qmax_days": int(group["qmax_code"].ge(3).sum())})
    reference = A1_QMAX_MISSING_BY_PERIOD.loc[period]
    counts.insert(0, {"period": period, "gap_class": gap_classes[0],
                      "valid_qmax_days": int(reference["valid_days"]) - sum(x["valid_qmax_days"] for x in counts),
                      "missing_qmax_days": int(reference["total_missing"]) - sum(x["missing_qmax_days"] for x in counts),
                      "no_row_days": int(reference["missing_rows"]) - sum(x["no_row_days"] for x in counts),
                      "blank_qmax_days": int(reference["blank_cells"]) - sum(x["blank_qmax_days"] for x in counts),
                      "other_invalid_qmax_days": int(reference["invalid_nonblank"]) - sum(x["other_invalid_qmax_days"] for x in counts)})
    cross_rows.extend(counts)
A1_GAPS_QMAX_PERIOD_CROSS_DETAIL = pd.DataFrame(cross_rows)
A1_GAPS_QMAX_PERIOD_CROSS_DETAIL["basin_days"] = A1_GAPS_QMAX_PERIOD_CROSS_DETAIL["valid_qmax_days"] + A1_GAPS_QMAX_PERIOD_CROSS_DETAIL["missing_qmax_days"]
if (A1_GAPS_QMAX_PERIOD_CROSS_DETAIL[["no_row_days", "blank_qmax_days", "other_invalid_qmax_days"]].lt(0).any().any()
        or not A1_GAPS_QMAX_PERIOD_CROSS_DETAIL[["no_row_days", "blank_qmax_days", "other_invalid_qmax_days"]].sum(axis=1).eq(
            A1_GAPS_QMAX_PERIOD_CROSS_DETAIL["missing_qmax_days"]).all()):
    raise ValueError("Các trạng thái Q thiếu chưa cộng đúng tổng ở bảng giao chi tiết.")
A1_GAPS_QMAX_PERIOD_CROSS = A1_GAPS_QMAX_PERIOD_CROSS_DETAIL[[
    "period", "gap_class", "valid_qmax_days", "missing_qmax_days", "basin_days"]].copy()
if (A1_GAPS_QMAX_PERIOD_CROSS[["valid_qmax_days", "missing_qmax_days"]].lt(0).any().any()
        or not A1_GAPS_QMAX_PERIOD_CROSS.groupby("period")["basin_days"].sum().reindex(PERIODS).eq(
            A1_QMAX_MISSING_BY_PERIOD["expected_days"]).all()):
    raise ValueError("Bảng giao gaps/Q không khớp tổng lịch 451 lưu vực của 4.2.")

# 4.5b. ID 410: đo số mốc thực tế và ngày Q; mẫu số metadata chỉ là giả thuyết lịch giờ đầy đủ.
probe_id = "410"
if probe_id not in A1_DAILY_FILES["D"] or probe_id not in seen_gap_ids:
    raise ValueError("Không tìm thấy Q hoặc gaps ID 410 để kiểm giả thuyết đã ghi ở Phần 2.")
probe_raw = pd.read_csv(A1_DAILY_FILES["D"][probe_id], sep=";", dtype=str, encoding="utf-8-sig",
                        usecols=DATE_COLS + ["qmax"], keep_default_na=False, na_filter=False)
probe_dates = pd.DatetimeIndex(pd.to_datetime(probe_raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
if probe_dates.empty or probe_dates.isna().any() or probe_dates.duplicated().any() or not probe_dates.is_monotonic_increasing:
    raise ValueError("ID 410: ngày Q rỗng/không hợp lệ/trùng/không tăng.")
probe_calendar = pd.date_range(probe_dates.min(), probe_dates.max(), freq="D")
probe_text = probe_raw["qmax"].set_axis(probe_dates).reindex(probe_calendar).str.strip()
probe_numeric = pd.to_numeric(probe_text, errors="coerce")
probe_present = probe_calendar.isin(probe_dates)
probe_valid = np.isfinite(probe_numeric) & probe_numeric.ge(0)
probe_daily = A1_GAPS_DAILY.loc[A1_GAPS_DAILY["basin_id"].eq(probe_id)].set_index("date")
probe = probe_daily[["gap_hours", "short_run_hours", "long_run_hours"]].reindex(probe_calendar, fill_value=0)
probe.index.name = "date"
probe["row_present"] = probe_present
probe["qmax_raw"] = probe_text.to_numpy()
probe["valid_qmax"] = probe_valid.to_numpy()
probe["missing_qmax"] = ~probe_valid.to_numpy()
probe["blank_qmax"] = (probe_present & probe_text.eq("")).to_numpy()
probe["gap_class"] = np.select([probe["long_run_hours"].gt(0), probe["gap_hours"].gt(0)], gap_classes[:0:-1], default=gap_classes[0])
A1_GAPS_ID410_DAILY = probe.reset_index()
probe_profile = A1_GAPS_FILE_PROFILE.set_index("basin_id").loc[probe_id]
pre_permil = pd.to_numeric(gauge.loc[probe_id, "gaps_pre"], errors="coerce")
post_permil = pd.to_numeric(gauge.loc[probe_id, "gaps_post"], errors="coerce")
assumed_hours = len(probe_calendar) * 24
A1_GAPS_ID410_SUMMARY = pd.DataFrame([{
    "basin_id": probe_id, "file_start": probe_calendar[0], "file_end": probe_calendar[-1],
    "unique_gap_hours": int(probe_profile["unique_hours"]), "short_run_hours": int(probe_profile["short_run_hours"]),
    "long_run_hours": int(probe_profile["long_run_hours"]), "gap_hours_outside_Q_file_span": int(probe_daily.loc[
        ~probe_daily.index.isin(probe_calendar), "gap_hours"].sum()),
    "missing_qmax_days": int((~probe_valid).sum()), "blank_qmax_days": int(probe["blank_qmax"].sum()),
    "long_run_hours_on_valid_qmax_days": int(probe.loc[probe["valid_qmax"], "long_run_hours"].sum()),
    "long_run_hours_on_missing_qmax_days": int(probe.loc[~probe["valid_qmax"], "long_run_hours"].sum()),
    "assumed_full_calendar_hours": assumed_hours, "metadata_gaps_pre_permil": pre_permil,
    "metadata_gaps_post_permil": post_permil, "pre_times_assumed_hours": pre_permil * assumed_hours / 1000,
    "post_times_assumed_hours": post_permil * assumed_hours / 1000}])
A1_GAPS_ID410_CROSS = probe.groupby("gap_class").agg(days=("valid_qmax", "size"),
    valid_qmax_days=("valid_qmax", "sum"), missing_qmax_days=("missing_qmax", "sum"),
    blank_qmax_days=("blank_qmax", "sum"), gap_hours=("gap_hours", "sum"),
    short_run_hours=("short_run_hours", "sum"), long_run_hours=("long_run_hours", "sum")).reset_index()
summary_rows = []
for scope, profile_part in [("all_gap_files", A1_GAPS_FILE_PROFILE),
                            ("selected_gap_files", A1_GAPS_FILE_PROFILE.loc[A1_GAPS_FILE_PROFILE["selected"]])]:
    summary_rows.append({"scope": scope, "files": len(profile_part),
                         **{name: int(profile_part[name].sum()) for name in ["raw_rows", "unique_hours", "duplicate_rows",
                            "runs", "short_runs", "long_runs", "short_run_hours", "long_run_hours"]},
                         "max_run_hours": int(profile_part["max_run_hours"].max()) if len(profile_part) else 0})
A1_GAPS_SUMMARY = pd.DataFrame(summary_rows)

# 4.5c. Tách không có dòng/ô rỗng khỏi Q hợp lệ; ghi số đếm vì các lát nhỏ khó thấy.
fig, axes = plt.subplots(1, 2, figsize=(18, 6.8))
for selected_only, color, label in [(False, "#7d8790", "All gap files"), (True, "#1976a2", "Selected basins")]:
    lengths = A1_GAPS_HOURLY_RUNS.loc[A1_GAPS_HOURLY_RUNS["selected"] | (not selected_only), "run_hours"].to_numpy(dtype=int)
    if lengths.size:
        max_hours = int(A1_GAPS_HOURLY_RUNS["run_hours"].max())
        edges = np.unique(np.ceil(np.geomspace(1, max_hours + 1, 30))) - 0.5
        axes[0].stairs(np.histogram(lengths, bins=edges)[0], edges, color=color, label=label, linewidth=1.5)
axes[0].axvline(6.5, color="#b83d38", linestyle="--", linewidth=1, label="6-hour cutoff (source rule)")
axes[0].set_xscale("log")
axes[0].set(title="(a) Consecutive pre-processing gap timestamps", xlabel="Run length (hours; log axis)", ylabel="Number of runs")
axes[0].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2)
labels, denominator, group_values = [], [], []
for period in PERIODS:
    part = A1_GAPS_QMAX_PERIOD_CROSS_DETAIL.loc[A1_GAPS_QMAX_PERIOD_CROSS_DETAIL["period"].eq(period)].set_index("gap_class").reindex(gap_classes)
    q_groups = [("valid_qmax_days", "Valid Q"), ("no_row_days", "No row"), ("blank_qmax_days", "Blank cell")]
    if A1_GAPS_QMAX_PERIOD_CROSS_DETAIL["other_invalid_qmax_days"].sum():
        q_groups.append(("other_invalid_qmax_days", "Other invalid"))
    for column, label in q_groups:
        values = part[column].to_numpy(dtype=float)
        labels.append(f"{period.capitalize()}\n{label}")
        denominator.append(values.sum())
        group_values.append(values)
bottom = np.zeros(len(labels))
for j, (label, color) in enumerate(zip(gap_classes, ["#d9e0e5", "#48a1ae", "#b83d38"])):
    percentages = np.array([100 * values[j] / total if total else 0 for values, total in zip(group_values, denominator)])
    axes[1].bar(np.arange(len(labels)), percentages, bottom=bottom, color=color, label=label.replace("_", " "))
    bottom += percentages
for i, (values, total) in enumerate(zip(group_values, denominator)):
    axes[1].text(i, 104, f"n={int(total):,}\nshort={int(values[1]):,}\nlong={int(values[2]):,}",
                 ha="center", va="bottom", fontsize=7)
    if total == 0:
        axes[1].text(i, 45, "No days", ha="center", va="center", fontsize=8, rotation=90, color="#555555")
axes[1].set_xticks(np.arange(len(labels)), labels, fontsize=7)
axes[1].set_yticks(range(0, 101, 20))
axes[1].set(title="(b) Listed-gap days by daily qmax state", ylabel="Basin-days within each Q state (%)", ylim=(0, 135))
axes[1].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3)
for ax in axes:
    ax.grid(axis="y", alpha=0.18)
fig.suptitle("Hourly gap lists before processing and daily qmax availability", fontsize=15, y=0.98)
fig.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=0.30, wspace=0.23)
fig.text(0.5, 0.02, "Calendar dates are matched as written, without a timezone shift. No listed gap does not certify complete hourly observations.\n"
         "n counts all basin-days in the Q state; short/long count days in the two listed-gap classes. Interpolation and peak effects are not verified.",
         ha="center", fontsize=9)
gap_figure = FIGURE_DIR / "A1_P4_gap_hours_vs_daily_qmax.png"
fig.savefig(gap_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
for suffix, table in [("gaps_file_profile", A1_GAPS_FILE_PROFILE), ("gaps_hourly_runs", A1_GAPS_HOURLY_RUNS),
                      ("gaps_daily", A1_GAPS_DAILY), ("gaps_summary", A1_GAPS_SUMMARY),
                      ("gaps_qmax_period_cross", A1_GAPS_QMAX_PERIOD_CROSS), ("gaps_ID410_daily", A1_GAPS_ID410_DAILY),
                      ("gaps_ID410_summary", A1_GAPS_ID410_SUMMARY), ("gaps_ID410_cross", A1_GAPS_ID410_CROSS),
                      ("gaps_qmax_period_cross_detail", A1_GAPS_QMAX_PERIOD_CROSS_DETAIL)]:
    table.to_csv(TABLE_DIR / f"A1_P4_{suffix}.csv", index=False, encoding="utf-8-sig")
print("(1) Kiểm kê toàn bộ gaps; dòng trùng tách riêng, đoạn dùng mốc duy nhất:")
display(A1_GAPS_SUMMARY)
print("(2) Mười ID có nhiều mốc gaps nhất; bảng đầy đủ lưu ở CSV:")
display(A1_GAPS_FILE_PROFILE.sort_values("unique_hours", ascending=False).head(10))
print("(3) Ngày–lưu vực theo kỳ: Q hợp lệ/thiếu và không có mốc gaps/chỉ đoạn <=6h/có đoạn >6h:")
display(A1_GAPS_QMAX_PERIOD_CROSS)
print("Chi tiết tách không có dòng, ô rỗng và không hợp lệ khác:")
display(A1_GAPS_QMAX_PERIOD_CROSS_DETAIL)
print("(4) ID 410; mẫu số giờ lịch là giả thuyết, không phải độ phủ chuỗi giờ đã kiểm:")
with pd.option_context("display.max_columns", None):
    display(A1_GAPS_ID410_SUMMARY)
    display(A1_GAPS_ID410_CROSS)
    display(A1_GAPS_ID410_DAILY.loc[A1_GAPS_ID410_DAILY["valid_qmax"] & A1_GAPS_ID410_DAILY["gap_hours"].gt(0)])
print("Mốc giờ gaps nằm ngoài lịch Q kiểm cho 451 ID:", int(A1_GAPS_DAILY.loc[
    A1_GAPS_DAILY["selected"] & A1_GAPS_DAILY["qmax_code"].eq(-1), "gap_hours"].sum()))
print("Đã lưu hình:", gap_figure)
print("Đã lưu chín CSV (gồm bảng giao chi tiết) vào:", TABLE_DIR)
print(f"Thời gian quét gaps: {time.perf_counter() - started_gaps:.1f} giây.")

# %% [markdown]
# ## Phần 5. Phân phối Q, khí tượng và đối chiếu nguồn giờ
#
# Mô tả phân phối, các ca bất thường và phép tổng hợp giờ–ngày; giữ riêng bằng chứng và giả thuyết.

# %% [markdown]
# ### 5.1. Phân phối qmax: m³/s và mm/ngày, giữ số 0 và đối chiếu mặt nạ hợp lệ
#
# Đọc qmax thật của 451 ID; đổi đơn vị bằng area_gov như bộ nạp, không dùng area_calc thay thế.

# %% Phần 5.1 - Phân phối qmax: m³/s và mm/ngày, giữ số 0 và đối chiếu mặt nạ hợp lệ
# Đọc qmax thật của 451 ID; đổi đơn vị bằng area_gov như bộ nạp, không dùng area_calc thay thế.
# Thống kê giữ số 0; histogram log chỉ dùng giá trị dương, mẫu số được ghi riêng và không cắt đuôi.
# mm/ngày của qmax là cường độ tương đương, không phải tổng lượng dòng chảy thực tế trong ngày.
started_distribution = time.perf_counter()
A1_QMAX_VALUE_IDS = np.array(sorted(selected_ids, key=int), dtype="U")
A1_QMAX_VALUE_CALENDAR = pd.DatetimeIndex(A1_QMAX_AUDIT_CALENDAR)
area_gov = pd.to_numeric(gauge.loc[A1_QMAX_VALUE_IDS, "area_gov"], errors="coerce")
area_calc = pd.to_numeric(catchment.loc[A1_QMAX_VALUE_IDS, "area_calc"], errors="coerce")
if not (np.isfinite(area_gov) & area_gov.gt(0)).all():
    raise ValueError("area_gov không hữu hạn/dương; chưa thể đổi Q sang mm/ngày.")
A1_QMAX_AREA_CONVERSION = pd.DataFrame({"basin_id": A1_QMAX_VALUE_IDS, "area_gov_km2": area_gov.to_numpy(),
    "area_calc_km2": area_calc.to_numpy(), "m3_s_to_mm_day_factor": 86.4 / area_gov.to_numpy()})
A1_QMAX_VALUES_M3S = np.full((len(A1_QMAX_VALUE_IDS), len(A1_QMAX_VALUE_CALENDAR)), np.nan)
cache_rows = {str(basin_id): i for i, basin_id in enumerate(A1_QMAX_HEATMAP_ORDER["basin_id"])}
if set(cache_rows) != set(selected_ids) or A1_QMAX_MISSING_CODES.shape != A1_QMAX_VALUES_M3S.shape:
    raise ValueError("Cache 4.2 chưa khớp tập ID/kích thước của 5.1.")


def a1_q_distribution_stats(values):
    """Tính trên các giá trị Q hữu hạn không âm; số 0 vẫn tham gia mọi phân vị."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    stats = {"valid_days": len(values), "zero_days": int((values == 0).sum()),
             "positive_days": int((values > 0).sum()), "zero_pct": 100 * (values == 0).mean() if len(values) else np.nan}
    for label, probability in [("min", 0), ("p01", 0.01), ("p05", 0.05), ("median", 0.5),
                               ("p95", 0.95), ("p99", 0.99), ("max", 1)]:
        stats[label] = float(np.quantile(values, probability)) if len(values) else np.nan
    stats["mean"] = float(values.mean()) if len(values) else np.nan
    stats["std_sample"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    stats["skew_sample"] = float(pd.Series(values).skew()) if len(values) > 2 else np.nan
    return stats


q_distribution_rows = []
for row_index, basin_id in enumerate(A1_QMAX_VALUE_IDS):
    raw = pd.read_csv(A1_DAILY_FILES["D"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                      usecols=DATE_COLS + ["qmax"], keep_default_na=False, na_filter=False)
    dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
    if (dates.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing
            or not dates.isin(A1_QMAX_VALUE_CALENDAR).all()):
        raise ValueError(f"ID {basin_id}: ngày Q khác lịch/không hợp lệ/trùng/không tăng.")
    values = pd.to_numeric(raw["qmax"].str.strip(), errors="coerce").set_axis(dates).reindex(A1_QMAX_VALUE_CALENDAR).to_numpy()
    valid = np.isfinite(values) & (values >= 0)
    if not np.array_equal(valid, A1_QMAX_MISSING_CODES[cache_rows[basin_id]] == 0):
        raise ValueError(f"ID {basin_id}: mặt nạ Q hợp lệ khác 4.2; chưa gộp phân phối.")
    A1_QMAX_VALUES_M3S[row_index, valid] = values[valid]
    factor = A1_QMAX_AREA_CONVERSION.iloc[row_index]["m3_s_to_mm_day_factor"]
    first_max_date = A1_QMAX_VALUE_CALENDAR[np.flatnonzero(valid & (values == values[valid].max()))[0]] if valid.any() else pd.NaT
    for unit, scaled in [("m3/s", values[valid]), ("mm/day", values[valid] * factor)]:
        q_distribution_rows.append({"basin_id": basin_id, "unit": unit,
            "area_gov_km2": area_gov.iloc[row_index], "first_date_of_max": first_max_date,
            **a1_q_distribution_stats(scaled)})
    if (row_index + 1) % 100 == 0 or row_index + 1 == len(A1_QMAX_VALUE_IDS):
        print(f"Đã đọc phân phối qmax {row_index + 1}/{len(A1_QMAX_VALUE_IDS)} lưu vực", flush=True)
A1_QMAX_VALUES_MM_DAY = A1_QMAX_VALUES_M3S * A1_QMAX_AREA_CONVERSION["m3_s_to_mm_day_factor"].to_numpy()[:, None]
if not np.array_equal(A1_QMAX_VALUES_M3S == 0, A1_QMAX_VALUES_MM_DAY == 0):
    raise ValueError("Đổi đơn vị làm đổi mặt nạ số 0; kiểm lại hệ số diện tích.")
A1_QMAX_DISTRIBUTION_BY_BASIN = pd.DataFrame(q_distribution_rows)
pooled_rows, bin_rows = [], []
fig, axes = plt.subplots(1, 2, figsize=(15, 5.3))
for ax, unit, matrix, xlabel in zip(axes, ["m3/s", "mm/day"],
        [A1_QMAX_VALUES_M3S, A1_QMAX_VALUES_MM_DAY], ["qmax (m³/s; log axis)", "qmax (mm/day; log axis)"]):
    pooled = matrix[np.isfinite(matrix)]
    stats = a1_q_distribution_stats(pooled)
    pooled_rows.append({"unit": unit, "basins": len(A1_QMAX_VALUE_IDS), **stats})
    positive = pooled[pooled > 0]
    if len(positive):
        left, right = positive.min(), positive.max()
        if left == right:
            left, right = left / 1.1, right * 1.1
        edges = np.geomspace(left, np.nextafter(right, np.inf), 61)
        counts, edges = np.histogram(positive, bins=edges)
        if counts.sum() != len(positive):
            raise ValueError(f"Histogram {unit} chưa bao trọn mọi Q dương.")
        percentages = 100 * counts / len(positive)
        ax.stairs(percentages, edges, color="#1976a2", linewidth=1.6)
        ax.set_xscale("log")
        for i, count in enumerate(counts):
            bin_rows.append({"unit": unit, "left_edge": edges[i], "right_edge": edges[i + 1],
                             "right_edge_included": i == len(counts) - 1, "positive_days": int(count),
                             "pct_of_positive_days": percentages[i]})
    else:
        ax.text(0.5, 0.5, "No positive valid qmax", transform=ax.transAxes, ha="center")
    ax.set(xlabel=xlabel, ylabel="Positive basin-days per bin (%)", title=f"Daily maximum discharge: {unit}")
    ax.grid(alpha=0.18)
    ax.text(0.5, -0.24, f"Valid days: {stats['valid_days']:,}; zero days: {stats['zero_days']:,} ({stats['zero_pct']:.6f}%)",
            transform=ax.transAxes, ha="center", fontsize=9)
A1_QMAX_DISTRIBUTION_POOLED = pd.DataFrame(pooled_rows)
A1_QMAX_DISTRIBUTION_BINS = pd.DataFrame(bin_rows)
fig.suptitle(f"qmax distributions in {len(A1_QMAX_VALUE_IDS)} selected basins: volume and area-normalized units", fontsize=14, y=0.98)
fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.31, wspace=0.23)
fig.text(0.5, 0.025, "Conversion: qmax (m³/s) × 86.4 / area_gov (km²). Statistics retain zeros; log histograms use positive values only.\n"
         "All available record days are pooled: longer records contribute more days. No clipping, imputation or outlier removal is applied.",
         ha="center", fontsize=9)
q_distribution_figure = FIGURE_DIR / "A1_P5_qmax_distributions_units.png"
fig.savefig(q_distribution_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
for suffix, table in [("qmax_distribution_by_basin", A1_QMAX_DISTRIBUTION_BY_BASIN),
                      ("qmax_distribution_pooled", A1_QMAX_DISTRIBUTION_POOLED),
                      ("qmax_distribution_bins", A1_QMAX_DISTRIBUTION_BINS), ("qmax_area_conversion", A1_QMAX_AREA_CONVERSION)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
np.savez_compressed(TABLE_DIR / "A1_P5_qmax_valid_values.npz", basin_ids=A1_QMAX_VALUE_IDS,
                    dates=A1_QMAX_VALUE_CALENDAR.to_numpy(dtype="datetime64[D]"), qmax_m3s=A1_QMAX_VALUES_M3S,
                    area_gov_km2=area_gov.to_numpy(), mm_day_factors=A1_QMAX_AREA_CONVERSION["m3_s_to_mm_day_factor"].to_numpy())
print("(1) Phân phối gộp các ngày Q hợp lệ trên toàn chuỗi; số 0 tham gia phân vị, histogram chỉ dùng Q dương:")
with pd.option_context("display.max_columns", None):
    display(A1_QMAX_DISTRIBUTION_POOLED)
print("(2) Mười ID có qmax mm/ngày lớn nhất; ngày là lần đầu đạt cực đại, chưa gán nhãn lỗi:")
display(A1_QMAX_DISTRIBUTION_BY_BASIN.loc[A1_QMAX_DISTRIBUTION_BY_BASIN["unit"].eq("mm/day")].sort_values(
    "max", ascending=False)[["basin_id", "area_gov_km2", "valid_days", "zero_days", "median", "p95", "p99", "max", "first_date_of_max"]].head(10))
print("(3) Ví dụ hệ số đổi đơn vị tại ba diện tích nhỏ nhất và ba lớn nhất; đủ 451 ID lưu CSV:")
area_sorted = A1_QMAX_AREA_CONVERSION.sort_values("area_gov_km2")
display(pd.concat([area_sorted.head(3), area_sorted.tail(3)]).drop_duplicates("basin_id"))
print("Đã lưu hình:", q_distribution_figure)
print("Đã lưu bốn CSV và cache Q hợp lệ NPZ vào:", TABLE_DIR)
print(f"Thời gian phân tích phân phối: {time.perf_counter() - started_distribution:.1f} giây.")

# %% [markdown]
# ### 5.2. Kiểm đơn vị nhãn qua cấu hình và scaler của từng run
#
# Tái dùng Q 5.1, so m³/s và mm/ngày theo area_gov/area_calc trên đúng nhãn train của từng cấu hình.

# %% Phần 5.2 - Kiểm đơn vị nhãn qua cấu hình và scaler của từng run
# Tái dùng Q 5.1, so m³/s và mm/ngày theo area_gov/area_calc trên đúng nhãn train của từng cấu hình.
# Mô phỏng bước float32, NaN nhãn trước train và std ddof=0; chưa lọc cửa sổ hoặc thay scaler.
import xarray as xr

TARGET_SCALER_RTOL, TARGET_SCALER_ATOL = 1e-5, 1e-7
target_ids = [str(value) for value in A1_QMAX_VALUE_IDS]
target_calendar = pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
target_values = np.asarray(A1_QMAX_VALUES_M3S)
if (len(set(target_ids)) != len(target_ids) or set(target_ids) != set(selected_ids)
        or target_values.shape != (len(target_ids), len(target_calendar))):
    raise ValueError("Cache 5.1 chưa khớp ID/lịch/kích thước; chạy lại 5.1.")
target_row_index = {basin_id: i for i, basin_id in enumerate(target_ids)}
target_areas = A1_QMAX_AREA_CONVERSION.set_index("basin_id").reindex(target_ids)
target_gov_max = float(np.nanmax(A1_QMAX_VALUES_MM_DAY))
for field in ["area_gov_km2", "area_calc_km2"]:
    if not (np.isfinite(target_areas[field]) & target_areas[field].gt(0)).all():
        raise ValueError(f"{field} không hữu hạn/dương; chưa so hai cách đổi diện tích.")
target_reference_cache, target_comparison_rows, target_run_rows = {}, [], []
started_target_scaler = time.perf_counter()


def a1_target_date(value):
    if isinstance(value, str):
        return pd.to_datetime(value, format="%d/%m/%Y" if "/" in value else "%Y-%m-%d", errors="raise").normalize()
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Ngày train rỗng.")
    return result.normalize()


def a1_target_scalar(saved, key):
    dataset = xr.Dataset.from_dict(saved[key])
    if "qmax" not in dataset or dataset["qmax"].size != 1:
        raise ValueError(f"{key} không chứa một scalar qmax.")
    value = float(dataset["qmax"].to_numpy().item())
    if not np.isfinite(value):
        raise ValueError(f"{key}/qmax không hữu hạn.")
    return value


def a1_target_scope(cfg):
    # Không đọc pickle của train_data_file hoặc per-basin periods để đoán phạm vi scaler.
    if cfg.get("train_data_file") not in [None, "None"] or cfg.get("per_basin_train_periods_file") not in [None, "None"]:
        raise ValueError("Phạm vi dựa trên train_data_file/per-basin periods chưa được hỗ trợ trong phép kiểm này.")
    if cfg.get("dataset") not in ["lamah_a", "lamah_b", "lamah_c"]:
        raise ValueError("Cấu hình không chỉ tới lớp LamaH đã đọc mã.")
    frequencies = cfg.get("use_frequencies") or []
    frequencies = [frequencies] if isinstance(frequencies, str) else frequencies
    if frequencies not in [[], ["1D"], ["1d"], ["D"]]:
        raise ValueError("Phạm vi khác ngày đơn tần chưa được hỗ trợ.")
    if "qmax" not in cfg.get("target_variables", []):
        raise ValueError("qmax không được khai báo là nhãn.")
    custom = (cfg.get("custom_normalization") or {}).get("qmax", {})
    if (not isinstance(custom, Mapping) or any(key not in ["centering", "scaling"] for key in custom)
            or ("centering" in custom and custom["centering"] != "mean")
            or ("scaling" in custom and custom["scaling"] != "std")):
        raise ValueError("qmax có custom_normalization khác mean/std; không so như z-score mặc định.")
    if "qmax" in (cfg.get("random_holdout_from_dynamic_features") or {}):
        raise ValueError("qmax được random holdout; chưa tái dựng mặt nạ ngẫu nhiên.")
    declared = cfg.get("train_basin_file")
    if not isinstance(declared, str) or not declared:
        raise ValueError("Thiếu train_basin_file.")
    direct = Path(declared)
    basename = declared.replace("\\", "/").rsplit("/", 1)[-1]
    candidates = [direct] if direct.is_file() else [p for p in EXP_DIR.rglob(basename) if p.is_file() and not a1_is_metadata(p)]
    if len(candidates) != 1:
        raise ValueError(f"Không giải duy nhất train_basin_file theo tên {basename}: {len(candidates)} tệp.")
    tokens = candidates[0].read_text(encoding="utf-8-sig").split()
    if not tokens or not all(re.fullmatch(r"\d+", token) for token in tokens):
        raise ValueError("Tệp ID train rỗng hoặc có ID không nguyên.")
    basin_ids = sorted(str(int(token)) for token in tokens)  # Thứ tự chuỗi như load_basin_file.
    if len(set(basin_ids)) != len(basin_ids) or not set(basin_ids).issubset(target_row_index):
        raise ValueError("ID train trùng hoặc nằm ngoài cache 451 ID; chưa suy thống kê từ tập con khác.")
    starts, ends = cfg.get("train_start_date"), cfg.get("train_end_date")
    starts = starts if isinstance(starts, list) else [starts]
    ends = ends if isinstance(ends, list) else [ends]
    if len(starts) != len(ends) or not starts:
        raise ValueError("Danh sách đầu/cuối train không cùng độ dài.")
    intervals = [(a1_target_date(a), a1_target_date(b)) for a, b in zip(starts, ends)]
    if any(a > b for a, b in intervals):
        raise ValueError("Ngày bắt đầu train sau ngày kết thúc.")
    seq, predict = cfg.get("seq_length"), cfg.get("predict_last_n")
    if (not isinstance(seq, int) or isinstance(seq, bool) or not isinstance(predict, int)
            or isinstance(predict, bool) or seq < predict or predict < 1):
        raise ValueError("seq_length/predict_last_n không phải số ngày hợp lệ của cấu hình đơn tần.")
    return basin_ids, intervals, seq - predict, candidates[0]


def a1_target_references(basin_ids, intervals, warmup):
    signature = (tuple(basin_ids), tuple((a.isoformat(), b.isoformat()) for a, b in intervals), warmup)
    if signature in target_reference_cache:
        return target_reference_cache[signature]
    calendar = pd.date_range(min(a for a, b in intervals) - pd.Timedelta(days=warmup), max(b for a, b in intervals))
    positions = target_calendar.get_indexer(calendar)
    values = np.full((len(basin_ids), len(calendar)), np.nan)
    inside = positions >= 0
    source_rows = [target_row_index[basin_id] for basin_id in basin_ids]
    values[:, inside] = target_values[np.ix_(source_rows, positions[inside])]
    label_dates = np.zeros(len(calendar), dtype=bool)
    for start, end in intervals:
        label_dates |= (calendar >= start) & (calendar <= end)
    values[:, ~label_dates] = np.nan  # Nhãn bị xóa trong khởi động; chưa lọc cửa sổ.
    n_valid = int(np.isfinite(values).sum())
    if not n_valid:
        raise ValueError("Phạm vi train không có nhãn Q hợp lệ.")
    factors = {"m3_s": np.ones(len(basin_ids)),
               "mm_day_area_gov": 86.4 / target_areas.loc[basin_ids, "area_gov_km2"].to_numpy(),
               "mm_day_area_calc": 86.4 / target_areas.loc[basin_ids, "area_calc_km2"].to_numpy()}
    records = []
    for candidate, factor in factors.items():
        converted = values * factor[:, None]
        for precision in ["float32", "float64_control"]:
            array = xr.DataArray(converted.astype(np.float32 if precision == "float32" else np.float64), dims=["basin", "date"])
            records.append({"candidate": candidate, "precision": precision, "n_basins": len(basin_ids), "n_labels": n_valid,
                            "train_start": min(a for a, b in intervals), "train_end": max(b for a, b in intervals),
                            "train_intervals": " | ".join(f"{a:%Y-%m-%d}:{b:%Y-%m-%d}" for a, b in intervals),
                            "warmup_days": warmup, "mean": float(array.mean(skipna=True)),
                            "std_ddof0": float(array.std(skipna=True, ddof=0))})
    target_reference_cache[signature] = records
    return records


for config_path in config_paths:
    run_name = config_path.parent.relative_to(EXP_DIR).as_posix()
    row = {"run": run_name, "status": "not_checked", "saved_mean": np.nan, "saved_std": np.nan,
           "matched_candidates_float32": "", "whole_record_max_z_if_gov_matches": np.nan}
    try:
        cfg = a1_load_yaml(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(cfg, Mapping):
            raise ValueError("Cấu hình không phải mapping.")
        row.update(dataset=cfg.get("dataset"), target_variables=str(cfg.get("target_variables")),
                   normalize_discharge_key_present="normalize_discharge" in cfg,
                   normalize_discharge_value=str(cfg.get("normalize_discharge", "<absent>")),
                   custom_normalization_qmax=str((cfg.get("custom_normalization") or {}).get("qmax", {})),
                   declared_train_basin_file=str(cfg.get("train_basin_file")),
                   declared_train_start=str(cfg.get("train_start_date")), declared_train_end=str(cfg.get("train_end_date")))
        scaler_path = config_path.parent / "train_data" / "train_data_scaler.yml"
        if not scaler_path.is_file():
            row["status"] = "missing_yaml"
        else:
            saved = a1_load_yaml(scaler_path.read_text(encoding="utf-8-sig"))
            row["saved_mean"] = a1_target_scalar(saved, "xarray_feature_center")
            row["saved_std"] = a1_target_scalar(saved, "xarray_feature_scale")
            if row["saved_std"] <= 0:
                raise ValueError("Độ chia qmax không dương.")
            basin_ids, intervals, warmup, resolved_basin_file = a1_target_scope(cfg)
            references = a1_target_references(basin_ids, intervals, warmup)
            row.update(n_train_basins=len(basin_ids), n_train_labels=references[0]["n_labels"],
                       resolved_train_basin_file=resolved_basin_file.relative_to(EXP_DIR).as_posix()
                       if resolved_basin_file.is_relative_to(EXP_DIR) else str(resolved_basin_file),
                       train_start=references[0]["train_start"], train_end=references[0]["train_end"], warmup_days=warmup)
            matching = []
            for reference in references:
                mean_match = bool(np.isclose(reference["mean"], row["saved_mean"], rtol=TARGET_SCALER_RTOL, atol=TARGET_SCALER_ATOL))
                std_match = bool(np.isclose(reference["std_ddof0"], row["saved_std"], rtol=TARGET_SCALER_RTOL, atol=TARGET_SCALER_ATOL))
                target_comparison_rows.append({"run": run_name, **reference, "saved_mean": row["saved_mean"],
                    "saved_std": row["saved_std"], "mean_abs_delta": abs(reference["mean"] - row["saved_mean"]),
                    "std_abs_delta": abs(reference["std_ddof0"] - row["saved_std"]),
                    "mean_matches": mean_match, "std_matches": std_match, "both_match": mean_match and std_match})
                if reference["precision"] == "float32" and mean_match and std_match:
                    matching.append(reference["candidate"])
            row["matched_candidates_float32"] = " | ".join(matching)
            row["status"] = ("matched_gov" if matching == ["mm_day_area_gov"] else
                             "ambiguous_match" if len(matching) > 1 else "matched_other" if matching else "no_candidate_match")
            if "mm_day_area_gov" in matching:
                row["whole_record_max_z_if_gov_matches"] = (
                    target_gov_max - row["saved_mean"]) / row["saved_std"]
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        row.update(status="unsupported_or_unreadable", detail=str(error))
    target_run_rows.append(row)
A1_TARGET_SCALER_RUNS = pd.DataFrame(target_run_rows)
A1_TARGET_SCALER_COMPARISONS = pd.DataFrame(target_comparison_rows)
A1_TARGET_CONFIG_AUDIT = A1_TARGET_SCALER_RUNS.reindex(columns=["run", "dataset", "target_variables", "normalize_discharge_key_present",
                                                             "normalize_discharge_value", "custom_normalization_qmax"]).copy()
config_summary = A1_TARGET_CONFIG_AUDIT.groupby(["dataset", "target_variables", "normalize_discharge_key_present",
    "normalize_discharge_value", "custom_normalization_qmax"], dropna=False).size().rename("configs").reset_index()
if target_comparison_rows:
    reference_summary = A1_TARGET_SCALER_COMPARISONS.loc[A1_TARGET_SCALER_COMPARISONS["precision"].eq("float32")].groupby(
        ["candidate", "train_start", "train_end", "warmup_days", "n_basins", "n_labels", "mean", "std_ddof0"],
        dropna=False, as_index=False).agg(runs=("run", "nunique"), matching_runs=("both_match", "sum"))
else:
    reference_summary = pd.DataFrame()
A1_TARGET_SCALER_REFERENCE_SUMMARY = reference_summary
for suffix, table in [("target_scaler_runs", A1_TARGET_SCALER_RUNS), ("target_scaler_comparisons", A1_TARGET_SCALER_COMPARISONS),
                      ("target_config_audit", A1_TARGET_CONFIG_AUDIT), ("target_scaler_reference_summary", reference_summary)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
print("(1) Cấu hình thực tế; cờ normalize_discharge ở đây không thay lời gọi đổi đơn vị của lớp LamaH:")
display(config_summary)
print("(2) Scaler từng run; matched_gov cần cả mean/std khớp chỉ ứng viên mm/ngày theo area_gov:")
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    display(A1_TARGET_SCALER_RUNS.drop(columns=["declared_train_basin_file", "resolved_train_basin_file"], errors="ignore"))
print("(3) Thống kê nhãn train tính lại (float32, ddof=0), gộp các trường hợp cùng kết quả:")
display(reference_summary)
print("(4) Trạng thái còn mở; thiếu YAML hoặc không khớp chưa chứng minh scaler tác giả sai:")
display(A1_TARGET_SCALER_RUNS.loc[A1_TARGET_SCALER_RUNS["status"].ne("matched_gov")])
print("Số cấu hình:", len(config_paths), "| Số phạm vi train đã tính:", len(target_reference_cache))
print(f"Dung sai so scaler: rtol={TARGET_SCALER_RTOL:g}, atol={TARGET_SCALER_ATOL:g}.")
print("Đã lưu bốn CSV vào:", TABLE_DIR)
print(f"Thời gian kiểm scaler nhãn: {time.perf_counter() - started_target_scaler:.1f} giây.")

# %% [markdown]
# ### 5.3. Chẩn đoán sai khác scaler: độ chính xác tích lũy và backend, giữ dung sai 5.2
#
# Cùng Q/ID/ngày, đổi riêng dtype lưu trữ, dtype tích lũy và backend; không nới ngưỡng để ép khớp.

# %% Phần 5.3 - Chẩn đoán sai khác scaler: độ chính xác tích lũy và backend, giữ dung sai 5.2
# Cùng Q/ID/ngày, đổi riêng dtype lưu trữ, dtype tích lũy và backend; không nới ngưỡng để ép khớp.
# Tái dùng cache 5.1 và bảng 5.2, không đọc pickle hoặc ghi đè các CSV 5.2.
import importlib.metadata as metadata
import platform

started_numeric_scaler = time.perf_counter()
if A1_TARGET_SCALER_RUNS.empty or A1_TARGET_SCALER_COMPARISONS.empty:
    raise ValueError("Cần chạy 5.2 và có ít nhất một phạm vi đã tính để chẩn đoán.")
numeric_environment = [{"item": "python", "value": platform.python_version()}]
for package in ["numpy", "pandas", "xarray", "Bottleneck", "numbagg"]:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "not_installed"
    numeric_environment.append({"item": package, "value": version})
numeric_options_before = dict(xr.get_options())
numeric_disable_options = {key: False for key in ["use_bottleneck", "use_numbagg"] if key in numeric_options_before}
for key in ["use_bottleneck", "use_numbagg"]:
    numeric_environment.append({"item": f"xarray_option/{key}", "value": str(numeric_options_before.get(key, "unavailable"))})
A1_TARGET_NUMERIC_ENVIRONMENT = pd.DataFrame(numeric_environment)
numeric_scope_cache, numeric_comparisons, numeric_run_rows, numeric_open_configs = {}, [], [], []
numeric_previous = A1_TARGET_SCALER_RUNS.set_index("run")


def a1_numeric_reductions(basin_ids, intervals, warmup):
    signature = (tuple(basin_ids), tuple((a.isoformat(), b.isoformat()) for a, b in intervals), warmup)
    if signature in numeric_scope_cache:
        return numeric_scope_cache[signature]
    calendar = pd.date_range(min(a for a, b in intervals) - pd.Timedelta(days=warmup), max(b for a, b in intervals))
    positions = target_calendar.get_indexer(calendar)
    inside = positions >= 0
    source_rows = [target_row_index[basin_id] for basin_id in basin_ids]
    raw = np.full((len(basin_ids), len(calendar)), np.nan, dtype=np.float64)
    raw[:, inside] = target_values[np.ix_(source_rows, positions[inside])]
    mask = np.zeros(len(calendar), dtype=bool)
    for start, end in intervals:
        mask |= (calendar >= start) & (calendar <= end)
    raw[:, ~mask] = np.nan
    n_labels = int(np.isfinite(raw).sum())
    if not n_labels:
        raise ValueError("Không có nhãn train hữu hạn trong phạm vi này.")
    results = []
    for candidate, area_field in [("m3_s", None), ("mm_day_area_gov", "area_gov_km2"),
                                   ("mm_day_area_calc", "area_calc_km2")]:
        # Giữ thứ tự phép đổi đơn vị của _normalize_discharge để tách khỏi backend thống kê.
        values64 = raw if area_field is None else raw / (
            target_areas.loc[basin_ids, area_field].to_numpy()[:, None] * 1e6) * 1000 * 86400
        values32 = values64.astype(np.float32)
        counts = (int(np.isfinite(values32).sum()), int(np.isfinite(values64).sum()))
        if counts != (n_labels, n_labels):
            raise ValueError("Đổi đơn vị/float32 làm đổi mặt nạ hữu hạn; chưa so scaler.")
        array32 = xr.DataArray(values32, dims=["basin", "date"])
        methods = [
            ("numpy64_storage64_acc64", float(np.nanmean(values64, dtype=np.float64)),
             float(np.nanstd(values64, dtype=np.float64, ddof=0))),
            ("numpy64_storage32_acc64", float(np.nanmean(values32, dtype=np.float64)),
             float(np.nanstd(values32, dtype=np.float64, ddof=0))),
            ("numpy32_storage32_acc32", float(np.nanmean(values32)), float(np.nanstd(values32, ddof=0))),
            ("xarray32_default", float(array32.mean(skipna=True)), float(array32.std(skipna=True, ddof=0))),
        ]
        with xr.set_options(**numeric_disable_options):
            methods.append(("xarray32_acceleration_disabled", float(array32.mean(skipna=True)),
                            float(array32.std(skipna=True, ddof=0))))
        for method, mean, std in methods:
            results.append({"candidate": candidate, "method": method, "n_basins": len(basin_ids),
                            "n_labels": n_labels, "warmup_days": warmup,
                            "train_intervals": " | ".join(f"{a:%Y-%m-%d}:{b:%Y-%m-%d}" for a, b in intervals),
                            "mean": mean, "std_ddof0": std})
    numeric_scope_cache[signature] = results
    return results


for config_path in config_paths:
    run_name = config_path.parent.relative_to(EXP_DIR).as_posix()
    previous = numeric_previous.loc[run_name]
    row = {"run": run_name, "status_5_2": previous["status"], "status_5_3": "not_evaluated",
           "saved_mean": previous["saved_mean"], "saved_std": previous["saved_std"],
           "matching_candidates_numpy64": "", "matching_candidates_storage32_acc64": "",
           "whole_record_max_z_if_gov_supported": np.nan}
    try:
        cfg = a1_load_yaml(config_path.read_text(encoding="utf-8-sig"))
        if previous["status"] in ["missing_yaml", "unsupported_or_unreadable"]:
            # Ghi nguyên khai báo để đọc lại, không nạp train_data_file hoặc gán scaler từ run khác.
            numeric_open_configs.append({"run": run_name, "status_5_2": previous["status"],
                "train_data_file": str(cfg.get("train_data_file", "<absent>")),
                "per_basin_train_periods_file": str(cfg.get("per_basin_train_periods_file", "<absent>")),
                "train_basin_file": str(cfg.get("train_basin_file", "<absent>")),
                "detail_5_2": str(previous.get("detail", ""))})
        else:
            basin_ids, intervals, warmup, _ = a1_target_scope(cfg)
            references = a1_numeric_reductions(basin_ids, intervals, warmup)
            if references[0]["n_labels"] != int(previous["n_train_labels"]):
                raise ValueError("Số nhãn khác 5.2; chưa so sai số số học.")
            matching64, matching32acc64 = [], []
            for reference in references:
                mean_match = bool(np.isclose(reference["mean"], row["saved_mean"],
                                             rtol=TARGET_SCALER_RTOL, atol=TARGET_SCALER_ATOL))
                std_match = bool(np.isclose(reference["std_ddof0"], row["saved_std"],
                                            rtol=TARGET_SCALER_RTOL, atol=TARGET_SCALER_ATOL))
                numeric_comparisons.append({"run": run_name, **reference, "saved_mean": row["saved_mean"],
                    "saved_std": row["saved_std"], "mean_abs_delta": abs(reference["mean"] - row["saved_mean"]),
                    "std_abs_delta": abs(reference["std_ddof0"] - row["saved_std"]),
                    "mean_relative_delta_pct": 100 * abs(reference["mean"] - row["saved_mean"]) / abs(row["saved_mean"])
                    if row["saved_mean"] != 0 else np.nan,
                    "std_relative_delta_pct": 100 * abs(reference["std_ddof0"] - row["saved_std"]) / row["saved_std"],
                    "both_match": mean_match and std_match})
                if mean_match and std_match:
                    if reference["method"] == "numpy64_storage64_acc64":
                        matching64.append(reference["candidate"])
                    elif reference["method"] == "numpy64_storage32_acc64":
                        matching32acc64.append(reference["candidate"])
            row["matching_candidates_numpy64"] = " | ".join(matching64)
            row["matching_candidates_storage32_acc64"] = " | ".join(matching32acc64)
            row["status_5_3"] = ("gov_stats_match_numpy64" if matching64 == ["mm_day_area_gov"] else
                "ambiguous_numpy64" if len(matching64) > 1 else "other_stats_match_numpy64" if matching64 else "no_numpy64_match")
            if matching64 == ["mm_day_area_gov"]:
                row["whole_record_max_z_if_gov_supported"] = (target_gov_max - row["saved_mean"]) / row["saved_std"]
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        row.update(status_5_3="diagnostic_unreadable", detail=str(error))
    numeric_run_rows.append(row)
if dict(xr.get_options()) != numeric_options_before:
    raise RuntimeError("Tùy chọn xarray chưa trở lại trạng thái trước cell.")
A1_TARGET_NUMERIC_COMPARISONS = pd.DataFrame(numeric_comparisons)
A1_TARGET_NUMERIC_RUNS = pd.DataFrame(numeric_run_rows)
A1_TARGET_NUMERIC_OPEN_CONFIGS = pd.DataFrame(numeric_open_configs)
precision_columns = ["candidate", "precision", "n_basins", "n_labels", "mean", "std_ddof0", "saved_mean", "saved_std", "both_match"]
A1_TARGET_INITIAL_PRECISION = A1_TARGET_SCALER_COMPARISONS[precision_columns].drop_duplicates().reset_index(drop=True)
numeric_summary = (A1_TARGET_NUMERIC_COMPARISONS.groupby(
    ["candidate", "method", "n_basins", "n_labels", "train_intervals", "warmup_days", "mean", "std_ddof0",
     "saved_mean", "saved_std", "mean_relative_delta_pct", "std_relative_delta_pct"], dropna=False, as_index=False)
    .agg(runs=("run", "nunique"), matching_runs=("both_match", "sum"))) if numeric_comparisons else pd.DataFrame()
for suffix, table in [("target_numeric_environment", A1_TARGET_NUMERIC_ENVIRONMENT),
                      ("target_initial_precision", A1_TARGET_INITIAL_PRECISION),
                      ("target_numeric_comparisons", A1_TARGET_NUMERIC_COMPARISONS),
                      ("target_numeric_runs", A1_TARGET_NUMERIC_RUNS),
                      ("target_numeric_open_configs", A1_TARGET_NUMERIC_OPEN_CONFIGS)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None,
                       "display.precision", 12):
    print("(1) Môi trường và tùy chọn tăng tốc; cài gói không đồng nghĩa đã được dùng cho phép giảm này:")
    display(A1_TARGET_NUMERIC_ENVIRONMENT)
    print("(2) Cả float32 và float64_control đã tính trong 5.2; các dòng giống nhau được gộp:")
    display(A1_TARGET_INITIAL_PRECISION)
    print("(3) Cùng dữ liệu/ID/ngày, đổi riêng cách tính; ddof=0 và dung sai giữ nguyên 5.2:")
    display(numeric_summary)
    print("(4) Đối chiếu từng run bằng tham chiếu NumPy float64; chưa suy backend khi tác giả huấn luyện:")
    display(A1_TARGET_NUMERIC_RUNS)
    print("(5) Khai báo của run chưa đánh giá; giữ nguyên trạng thái thiếu YAML/phạm vi chưa hỗ trợ:")
    display(A1_TARGET_NUMERIC_OPEN_CONFIGS)
saved_pairs = A1_TARGET_NUMERIC_RUNS[["saved_mean", "saved_std"]].dropna()
print("Số cặp mean/std qmax khác nhau trong các scaler đã đọc:", len(saved_pairs.drop_duplicates()),
      "| Scaler đã đọc:", len(saved_pairs), "| Phạm vi chẩn đoán:", len(numeric_scope_cache))
print(f"Dung sai giữ nguyên: rtol={TARGET_SCALER_RTOL:g}, atol={TARGET_SCALER_ATOL:g}; tùy chọn xarray đã khôi phục.")
print("Đã lưu năm CSV mới vào:", TABLE_DIR)
print(f"Thời gian chẩn đoán scaler: {time.perf_counter() - started_numeric_scaler:.1f} giây.")

# %% [markdown]
# ### 5.4. Q theo lưu vực xếp diện tích và đóng góp vào đuôi p99
#
# Toàn bản ghi, cùng cache 5.1; phân vị giữ số 0. Đuôi là Q > p99 gộp theo ngày–lưu vực của từng đơn vị.

# %% Phần 5.4 - Q theo lưu vực xếp diện tích và đóng góp vào đuôi p99
# Toàn bản ghi, cùng cache 5.1; phân vị giữ số 0. Đuôi là Q > p99 gộp theo ngày–lưu vực của từng đơn vị.
# Chia nhóm theo tứ phân vị area_gov, giữ các diện tích bằng nhau cùng nhóm; chưa lọc hoặc biến đổi nhãn.
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

started_area_tail = time.perf_counter()
area_tail_ids = [str(value) for value in A1_QMAX_VALUE_IDS]
area_tail_calendar = pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
area_tail_raw = np.asarray(A1_QMAX_VALUES_M3S, dtype=np.float64)
if (len(set(area_tail_ids)) != len(area_tail_ids) or set(area_tail_ids) != set(selected_ids)
        or area_tail_raw.shape != (len(area_tail_ids), len(area_tail_calendar))):
    raise ValueError("Cache Q/ID/lịch 5.1 chưa khớp; chưa phân tích theo diện tích.")
area_rank = A1_QMAX_AREA_CONVERSION.copy()
area_rank["basin_id"] = area_rank["basin_id"].astype(str)
if area_rank["basin_id"].duplicated().any() or set(area_rank["basin_id"]) != set(area_tail_ids):
    raise ValueError("Bảng hệ số diện tích không khớp ID cache Q.")
if not (np.isfinite(area_rank["area_gov_km2"]) & area_rank["area_gov_km2"].gt(0)).all():
    raise ValueError("area_gov không hữu hạn/dương.")
area_rank["basin_number"] = area_rank["basin_id"].astype(int)
area_rank = area_rank.sort_values(["area_gov_km2", "basin_number"], kind="mergesort").reset_index(drop=True)
area_rank["area_rank"] = np.arange(1, len(area_rank) + 1)
group_codes, area_edges = pd.qcut(area_rank["area_gov_km2"], q=4, labels=False, retbins=True, duplicates="drop")
if group_codes.isna().any() or len(area_edges) < 3:
    raise ValueError("Không chia được ít nhất hai nhóm diện tích; chưa ép chia các diện tích bằng nhau.")
area_rank["area_group"] = group_codes.astype(int).map(lambda value: f"Q{value + 1}")
area_rank["name"] = area_rank["basin_id"].map(gauge["name"])
area_rank["country"] = area_rank["basin_id"].map(gauge["country"])
row_lookup = {basin_id: i for i, basin_id in enumerate(area_tail_ids)}
ranked_rows = [row_lookup[basin_id] for basin_id in area_rank["basin_id"]]
unit_matrices = {"m3/s": area_tail_raw[ranked_rows],
                 "mm/day": np.asarray(A1_QMAX_VALUES_MM_DAY, dtype=np.float64)[ranked_rows]}
if unit_matrices["mm/day"].shape != unit_matrices["m3/s"].shape:
    raise ValueError("Hai ma trận đơn vị khác kích thước.")
expected_mm = unit_matrices["m3/s"] * area_rank["m3_s_to_mm_day_factor"].to_numpy()[:, None]
if not np.allclose(expected_mm, unit_matrices["mm/day"], rtol=1e-12, atol=1e-12, equal_nan=True):
    raise ValueError("Cache mm/ngày chưa khớp hệ số area_gov theo ID.")
quantile_rows, threshold_rows, group_rows, box_stats = [], [], [], {}
quantile_probabilities = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
quantile_names = ["min", "p01", "p05", "q25", "median", "q75", "p95", "p99", "max"]
for unit, matrix in unit_matrices.items():
    valid_mask = np.isfinite(matrix)
    if np.any(matrix[valid_mask] < 0) or not valid_mask.any(axis=1).all():
        raise ValueError(f"{unit}: Q âm hoặc có lưu vực không có giá trị hợp lệ.")
    pooled = matrix[valid_mask]
    threshold = float(np.quantile(pooled, 0.99))  # Giữ số 0 và cùng trọng số ngày như 5.1.
    tail_mask = valid_mask & (matrix > threshold)
    tail_days = int(tail_mask.sum())
    if tail_days == 0:
        raise ValueError(f"{unit}: không có ngày vượt p99; chưa chia tỷ phần đuôi cho 0.")
    threshold_rows.append({"unit": unit, "basins": len(area_rank), "valid_days": len(pooled),
        "zero_days": int((pooled == 0).sum()), "pooled_p99": threshold,
        "days_equal_to_p99": int((pooled == threshold).sum()), "days_strictly_above_p99": tail_days,
        "tail_pct_of_valid_days": 100 * tail_days / len(pooled),
        "record_start": area_tail_calendar.min(), "record_end": area_tail_calendar.max()})
    boxes = []
    for position, basin in area_rank.iterrows():
        values = matrix[position, valid_mask[position]]
        qs = dict(zip(quantile_names, np.quantile(values, quantile_probabilities)))
        n_tail = int(tail_mask[position].sum())
        quantile_rows.append({"basin_id": basin["basin_id"], "unit": unit, "area_rank": basin["area_rank"],
            "name": basin["name"], "country": basin["country"], "area_gov_km2": basin["area_gov_km2"],
            "area_group": basin["area_group"], "valid_days": len(values), "zero_days": int((values == 0).sum()),
            **qs, "pooled_p99_threshold": threshold, "days_strictly_above_pooled_p99": n_tail,
            "tail_pct_of_basin_valid_days": 100 * n_tail / len(values),
            "share_of_all_tail_days_pct": 100 * n_tail / tail_days})
        boxes.append({"q1": qs["q25"], "med": qs["median"], "q3": qs["q75"],
                      "whislo": qs["p05"], "whishi": qs["p95"]})
    box_stats[unit] = boxes
    for group_number, (group, basins) in enumerate(area_rank.groupby("area_group", sort=True)):
        positions = basins.index.to_numpy()
        n_valid = int(valid_mask[positions].sum())
        n_tail = int(tail_mask[positions].sum())
        exposure_share = 100 * n_valid / len(pooled)
        tail_share = 100 * n_tail / tail_days
        group_rows.append({"unit": unit, "area_group": group, "basins": len(basins),
            "area_min_km2": basins["area_gov_km2"].min(), "area_max_km2": basins["area_gov_km2"].max(),
            "bin_lower_edge_km2": area_edges[group_number], "bin_upper_edge_km2": area_edges[group_number + 1],
            "bin_lower_inclusive": group_number == 0, "bin_upper_inclusive": True,
            "valid_days": n_valid, "share_of_all_valid_days_pct": exposure_share,
            "days_strictly_above_pooled_p99": n_tail,
            "basins_with_tail_days": int(tail_mask[positions].any(axis=1).sum()),
            "tail_pct_of_group_valid_days": 100 * n_tail / n_valid,
            "share_of_all_tail_days_pct": tail_share,
            "tail_share_over_valid_day_share": tail_share / exposure_share})
A1_QMAX_AREA_RANK = area_rank.drop(columns="basin_number")
A1_QMAX_AREA_QUANTILES = pd.DataFrame(quantile_rows)
A1_QMAX_AREA_THRESHOLDS = pd.DataFrame(threshold_rows)
A1_QMAX_AREA_TAIL_GROUPS = pd.DataFrame(group_rows)
reference_pooled = A1_QMAX_DISTRIBUTION_POOLED.set_index("unit")
for unit in unit_matrices:
    measured = A1_QMAX_AREA_THRESHOLDS.set_index("unit").loc[unit]
    if (int(measured["valid_days"]) != int(reference_pooled.loc[unit, "valid_days"])
            or int(measured["zero_days"]) != int(reference_pooled.loc[unit, "zero_days"])
            or not np.isclose(measured["pooled_p99"], reference_pooled.loc[unit, "p99"], rtol=1e-12, atol=1e-12)):
        raise ValueError(f"{unit}: thống kê gộp không khớp 5.1.")
    grouped = A1_QMAX_AREA_TAIL_GROUPS.query("unit == @unit")
    if (grouped["valid_days"].sum() != measured["valid_days"]
            or grouped["days_strictly_above_pooled_p99"].sum() != measured["days_strictly_above_p99"]
            or not np.isclose(grouped["share_of_all_tail_days_pct"].sum(), 100)):
        raise ValueError(f"{unit}: số đếm nhóm chưa cộng khớp.")

fig, axes = plt.subplots(2, 1, figsize=(17, 10))
tick_indices = np.unique(np.linspace(0, len(area_rank) - 1, min(7, len(area_rank))).round().astype(int))
tick_labels = [f"{i + 1} | ID {area_rank.iloc[i]['basin_id']}\n{area_rank.iloc[i]['area_gov_km2']:,.1f} km²" for i in tick_indices]
for ax, (unit, boxes) in zip(axes, box_stats.items()):
    ax.bxp(boxes, positions=area_rank["area_rank"], widths=0.65, showfliers=False, showcaps=False,
           patch_artist=True, manage_ticks=False,
           boxprops={"facecolor": "#77b7d5", "edgecolor": "#287da1", "linewidth": 0.35},
           medianprops={"color": "#15435b", "linewidth": 0.7}, whiskerprops={"color": "#8b969c", "linewidth": 0.4})
    maxima = A1_QMAX_AREA_QUANTILES.loc[A1_QMAX_AREA_QUANTILES["unit"].eq(unit), "max"].to_numpy()
    ax.scatter(area_rank["area_rank"], maxima, s=5, color="#ae4b31", alpha=0.7, zorder=3)
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.6)
    ax.set(xlim=(0, len(area_rank) + 1), ylim=(0, maxima.max() * 1.25),
           ylabel=f"qmax ({unit}; symlog)", title=f"Per-basin distributions: {unit}")
    ax.set_xticks(tick_indices + 1, tick_labels, fontsize=8)
    ax.set_xlabel("Basin rank by reported area, smallest to largest (rank | ID, area_gov)")
    ax.grid(axis="y", alpha=0.18)
fig.suptitle(f"Daily maximum discharge by basin: {len(area_rank)} basins sorted by area_gov", fontsize=15, y=0.98)
fig.legend(handles=[Patch(facecolor="#77b7d5", edgecolor="#287da1", label="Box: 25th–75th percentiles"),
                    Line2D([], [], color="#15435b", label="Median"),
                    Line2D([], [], color="#8b969c", label="Whiskers: 5th–95th percentiles"),
                    Line2D([], [], marker="o", linestyle="none", color="#ae4b31", markersize=4, label="Record maximum")],
           loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=4, frameon=False, fontsize=9)
fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.22, hspace=0.65)
fig.text(0.5, 0.02, "All available record days are used per basin; zeros are retained. Summary boxes do not display every observation.\n"
         "Symlog is linear from 0 to 0.1 and logarithmic above 0.1; it changes only the display. No clipping or training transform is applied.",
         ha="center", fontsize=9)
area_box_figure = FIGURE_DIR / "A1_P5_qmax_by_area_rank.png"
fig.savefig(area_box_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

group_definition = A1_QMAX_AREA_TAIL_GROUPS.loc[A1_QMAX_AREA_TAIL_GROUPS["unit"].eq("m3/s")].reset_index(drop=True)
x = np.arange(len(group_definition))
labels = [f"{row.area_group}\n{row.area_min_km2:,.1f}–{row.area_max_km2:,.1f} km²\nn={row.basins}" for row in group_definition.itertuples()]
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = {"m3/s": "#537b91", "mm/day": "#d47b38"}
for i, unit in enumerate(unit_matrices):
    grouped = A1_QMAX_AREA_TAIL_GROUPS.loc[A1_QMAX_AREA_TAIL_GROUPS["unit"].eq(unit)].reset_index(drop=True)
    axes[0].bar(x + (i - 0.5) * 0.32, grouped["share_of_all_tail_days_pct"], width=0.32,
                color=colors[unit], label=unit)
    axes[1].plot(x, grouped["tail_pct_of_group_valid_days"], marker="o", color=colors[unit], label=unit)
axes[0].plot(x, group_definition["share_of_all_valid_days_pct"], color="#31383c", marker="D", linestyle="--",
             markersize=4, label="Share of valid record days")
axes[0].set(title="(a) Share of all tail basin-days", ylabel="Percent of tail basin-days within each unit", ylim=(0, 105))
axes[1].set(title="(b) Exceedance rate within each area group", ylabel="Tail basin-days / valid group basin-days (%)", ylim=(0, None))
for ax in axes:
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_xlabel(f"Area groups: Q1 smallest → {area_rank['area_group'].iloc[-1]} largest")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle("Which basin sizes contribute to the pooled qmax tail?", fontsize=15, y=0.98)
fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.31, wspace=0.25)
fig.text(0.5, 0.035, "Tail = qmax strictly above the pooled 99th percentile of valid basin-days, separately for each unit; thresholds retain zeros.\n"
         "Groups use quartiles of area_gov; equal areas stay together. Record lengths differ, so valid-day shares are shown as exposure.\n"
         "Full records are pooled. These are descriptive counts, not independent samples or a basin filtering rule.", ha="center", fontsize=9)
area_tail_figure = FIGURE_DIR / "A1_P5_qmax_tail_by_area_group.png"
fig.savefig(area_tail_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
for suffix, table in [("qmax_area_rank", A1_QMAX_AREA_RANK), ("qmax_area_quantiles", A1_QMAX_AREA_QUANTILES),
                      ("qmax_area_thresholds", A1_QMAX_AREA_THRESHOLDS), ("qmax_area_tail_groups", A1_QMAX_AREA_TAIL_GROUPS)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    print("(1) Ngưỡng p99 gộp và số ngày vượt; bằng ngưỡng được đếm riêng, chưa gọi mọi giá trị này là lỗi:")
    display(A1_QMAX_AREA_THRESHOLDS)
    print("(2) Các nhóm diện tích; tỷ phần đuôi và mẫu số ngày hợp lệ tính riêng cho từng đơn vị:")
    display(A1_QMAX_AREA_TAIL_GROUPS)
    print("(3) Mười hai ID đóng góp nhiều ngày vượt p99 mm/ngày nhất; đây không phải xếp theo cực đại:")
    display(A1_QMAX_AREA_QUANTILES.loc[A1_QMAX_AREA_QUANTILES["unit"].eq("mm/day")].sort_values(
        ["days_strictly_above_pooled_p99", "area_rank"], ascending=[False, True])[[
        "basin_id", "name", "area_gov_km2", "area_group", "valid_days", "zero_days", "median", "p99", "max",
        "days_strictly_above_pooled_p99", "tail_pct_of_basin_valid_days", "share_of_all_tail_days_pct"]].head(12))
print("Hai đơn vị khớp số ngày hợp lệ, số 0 và p99 ở 5.1; số đếm nhóm cộng khớp.")
print("Đã lưu hai hình:", area_box_figure, "|", area_tail_figure)
print("Đã lưu bốn CSV vào:", TABLE_DIR)
print(f"Thời gian phân tích theo diện tích: {time.perf_counter() - started_area_tail:.1f} giây.")

# %% [markdown]
# ### 5.5. Mùa của qmax theo độ cao, năm thủy văn và độ phủ quan trắc
#
# Chọn năm thủy văn Oct–Sep trọn vẹn trong khoảng chung của first/last Q; vẫn kiểm lỗ bên trong.

# %% Phần 5.5 - Mùa của qmax theo độ cao, năm thủy văn và độ phủ quan trắc
# Chọn năm thủy văn Oct–Sep trọn vẹn trong khoảng chung của first/last Q; vẫn kiểm lỗ bên trong.
# Tính mean qmax ngày theo tháng từng lưu vực, rồi tóm tắt ngang lưu vực với trọng số bằng nhau.
# Tập đủ dữ liệu cả khoảng chỉ dùng đối chiếu mô tả, không thay 451 ID hoặc dữ liệu huấn luyện.
started_seasonality = time.perf_counter()
season_ids = [str(value) for value in A1_QMAX_VALUE_IDS]
season_calendar_all = pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
season_values_all = np.asarray(A1_QMAX_VALUES_MM_DAY, dtype=np.float64)
if (len(set(season_ids)) != len(season_ids) or set(season_ids) != set(selected_ids)
        or season_values_all.shape != (len(season_ids), len(season_calendar_all))
        or not season_calendar_all.equals(pd.date_range(season_calendar_all.min(), season_calendar_all.max()))):
    raise ValueError("Cache Q/ID/lịch 5.1 chưa hợp lệ cho khảo sát mùa.")
season_valid_all = np.isfinite(season_values_all)
if not season_valid_all.any(axis=1).all() or np.any(season_values_all[season_valid_all] < 0):
    raise ValueError("Có chuỗi không có Q hợp lệ hoặc có Q âm trong cache.")
season_first = season_calendar_all[season_valid_all.argmax(axis=1)]
season_last = season_calendar_all[len(season_calendar_all) - 1 - season_valid_all[:, ::-1].argmax(axis=1)]
season_latest_first, season_earliest_last = season_first.max(), season_last.min()
season_start = pd.Timestamp(year=season_latest_first.year, month=10, day=1)
if season_start < season_latest_first:
    season_start += pd.DateOffset(years=1)
season_end = pd.Timestamp(year=season_earliest_last.year, month=9, day=30)
if season_end > season_earliest_last:
    season_end -= pd.DateOffset(years=1)
if season_end < season_start:
    raise ValueError("Khoảng first/last chung không chứa một năm thủy văn Oct–Sep trọn vẹn.")
season_calendar = pd.date_range(season_start, season_end)
season_positions = season_calendar_all.get_indexer(season_calendar)
if np.any(season_positions < 0):
    raise ValueError("Khoảng mùa nằm ngoài cache Q.")
season_values = season_values_all[:, season_positions]
season_valid = np.isfinite(season_values)
season_complete = season_valid.all(axis=1)
season_hydro_years = season_end.year - season_start.year
season_months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
season_month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
season_info = pd.DataFrame({"basin_id": season_ids,
    "elev_mean_m": pd.to_numeric(catchment.loc[season_ids, "elev_mean"], errors="coerce").to_numpy(),
    "name": gauge.loc[season_ids, "name"].to_numpy(), "first_valid_date": season_first,
    "last_valid_date": season_last, "expected_days": len(season_calendar),
    "valid_days": season_valid.sum(axis=1), "missing_days": (~season_valid).sum(axis=1),
    "zero_days": (season_values == 0).sum(axis=1), "complete_window": season_complete})
if not np.isfinite(season_info["elev_mean_m"]).all():
    raise ValueError("Thiếu elev_mean lưu vực; không dùng elev trạm thay vào.")
elevation_codes, elevation_edges = pd.qcut(season_info["elev_mean_m"], q=4, labels=False,
                                         retbins=True, duplicates="drop")
if elevation_codes.isna().any() or len(elevation_edges) < 3:
    raise ValueError("Không chia được ít nhất hai nhóm độ cao, chưa ép chia các độ cao bằng nhau.")
season_info["elevation_group"] = elevation_codes.astype(int).map(lambda value: f"E{value + 1}")
season_info["missing_pct"] = 100 * season_info["missing_days"] / season_info["expected_days"]
season_group_rows, season_month_frames, season_summary_rows = [], [], []
for i, (group, basins) in enumerate(season_info.groupby("elevation_group", sort=True)):
    season_group_rows.append({"elevation_group": group, "basins": len(basins),
        "elev_min_m": basins["elev_mean_m"].min(), "elev_max_m": basins["elev_mean_m"].max(),
        "bin_lower_edge_m": elevation_edges[i], "bin_upper_edge_m": elevation_edges[i + 1],
        "bin_lower_inclusive": i == 0, "bin_upper_inclusive": True,
        "complete_window_basins": int(basins["complete_window"].sum()),
        "expected_days": int(basins["expected_days"].sum()), "valid_days": int(basins["valid_days"].sum()),
        "missing_pct": 100 * basins["missing_days"].sum() / basins["expected_days"].sum()})
for hydro_month, (month, label) in enumerate(zip(season_months, season_month_labels), start=1):
    monthly_values = season_values[:, season_calendar.month == month]
    expected = monthly_values.shape[1]  # Tính đúng cả 29/02 trong từng năm có mặt.
    counts = np.isfinite(monthly_values).sum(axis=1)
    means = np.divide(np.nansum(monthly_values, axis=1, dtype=np.float64), counts,
                      out=np.full(len(season_ids), np.nan), where=counts > 0)
    frame = season_info[["basin_id", "elev_mean_m", "elevation_group", "complete_window"]].copy()
    frame["hydro_month"], frame["calendar_month"], frame["month_name"] = hydro_month, month, label
    frame["expected_days"], frame["valid_days"] = expected, counts
    frame["missing_days"] = expected - counts
    frame["zero_days"] = (monthly_values == 0).sum(axis=1)
    frame["missing_pct"] = 100 * frame["missing_days"] / expected
    frame["mean_daily_qmax_mm_day"] = means  # Tháng không có Q giữ NaN, không biến thành 0.
    season_month_frames.append(frame)
    for group, rows in frame.groupby("elevation_group", sort=True):
        available = rows["mean_daily_qmax_mm_day"].dropna().to_numpy()
        complete = rows.loc[rows["complete_window"], "mean_daily_qmax_mm_day"].dropna().to_numpy()
        p25, med, p75 = np.quantile(available, [0.25, 0.5, 0.75]) if len(available) else [np.nan] * 3
        season_summary_rows.append({"elevation_group": group, "hydro_month": hydro_month,
            "calendar_month": month, "month_name": label, "basins": len(rows),
            "basins_with_any_valid_Q": len(available), "complete_window_basins": len(complete),
            "expected_days": int(rows["expected_days"].sum()), "valid_days": int(rows["valid_days"].sum()),
            "missing_pct": 100 * rows["missing_days"].sum() / rows["expected_days"].sum(),
            "p25_available_basin_month_means": p25, "median_available_basin_month_means": med,
            "p75_available_basin_month_means": p75,
            "median_complete_basin_month_means": float(np.median(complete)) if len(complete) else np.nan})
A1_QMAX_SEASON_BASIN_COVERAGE = season_info
A1_QMAX_SEASON_ELEVATION_GROUPS = pd.DataFrame(season_group_rows)
A1_QMAX_SEASON_BASIN_MONTHLY = pd.concat(season_month_frames, ignore_index=True)
A1_QMAX_SEASON_MONTHLY_SUMMARY = pd.DataFrame(season_summary_rows)
A1_QMAX_SEASON_SCOPE = pd.DataFrame([{"latest_first_valid_date": season_latest_first,
    "earliest_last_valid_date": season_earliest_last, "window_start": season_start, "window_end": season_end,
    "first_water_year_end_label": season_start.year + 1, "last_water_year_end_label": season_end.year,
    "complete_water_years": season_hydro_years, "calendar_days": len(season_calendar), "basins": len(season_ids),
    "expected_basin_days": season_valid.size, "valid_basin_days": int(season_valid.sum()),
    "missing_basin_days": int((~season_valid).sum()), "complete_window_basins": int(season_complete.sum())}])
if (A1_QMAX_SEASON_BASIN_MONTHLY["expected_days"].sum() != season_valid.size
        or A1_QMAX_SEASON_BASIN_MONTHLY["valid_days"].sum() != season_valid.sum()
        or A1_QMAX_SEASON_BASIN_MONTHLY["zero_days"].sum() != (season_values == 0).sum()
        or A1_QMAX_SEASON_MONTHLY_SUMMARY["valid_days"].sum() != season_valid.sum()):
    raise ValueError("Đếm theo tháng/nhóm chưa cộng khớp lịch chung.")

groups = A1_QMAX_SEASON_ELEVATION_GROUPS
season_x = np.arange(12)
fig, axes = plt.subplots(int(np.ceil(len(groups) / 2)), 2, figsize=(15, 9), sharex=True, sharey=True, squeeze=False)
for ax, group in zip(axes.flat, groups.itertuples()):
    profile = A1_QMAX_SEASON_MONTHLY_SUMMARY.loc[
        A1_QMAX_SEASON_MONTHLY_SUMMARY["elevation_group"].eq(group.elevation_group)].sort_values("hydro_month")
    ax.fill_between(season_x, profile["p25_available_basin_month_means"].to_numpy(),
                    profile["p75_available_basin_month_means"].to_numpy(), color="#7fbad3", alpha=0.35)
    ax.plot(season_x, profile["median_available_basin_month_means"], color="#1d789d", marker="o", markersize=3)
    ax.plot(season_x, profile["median_complete_basin_month_means"], color="#b45f28", linestyle="--", marker="s", markersize=3)
    ax.set_title(f"{group.elevation_group}: {group.elev_min_m:,.0f}–{group.elev_max_m:,.0f} m\n"
                 f"All basins: {group.basins}; complete-window subset: {group.complete_window_basins}", fontsize=11)
    ax.set_xticks(season_x, season_month_labels, fontsize=9)
    ax.set(xlim=(0, 11), ylabel="Monthly mean of daily qmax (mm/day)")
    ax.grid(alpha=0.18)
for ax in list(axes.flat)[len(groups):]:
    ax.set_visible(False)
season_display_values = A1_QMAX_SEASON_MONTHLY_SUMMARY[["p75_available_basin_month_means",
    "median_available_basin_month_means", "median_complete_basin_month_means"]].to_numpy()
season_display_max = float(np.nanmax(season_display_values))
axes.flat[0].set_ylim(0, season_display_max * 1.12 if season_display_max > 0 else 1)
for ax in axes[-1]:
    if ax.get_visible():
        ax.set_xlabel("Month in the October–September water year")
water_year_label = f"{season_start.year + 1}–{season_end.year}"
fig.suptitle(f"Seasonal qmax by mean basin elevation: complete water years {water_year_label}", fontsize=15, y=0.98)
fig.legend(handles=[Line2D([], [], color="#1d789d", marker="o", markersize=3, label="Median across available basin monthly means"),
                    Patch(facecolor="#7fbad3", alpha=0.35, label="25th–75th percentiles across basins"),
                    Line2D([], [], color="#b45f28", linestyle="--", marker="s", markersize=3, label="Median across complete-window basins")],
           loc="lower center", bbox_to_anchor=(0.5, 0.085), frameon=False, ncol=1, fontsize=9)
fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.27, hspace=0.46, wspace=0.2)
fig.text(0.5, 0.02, f"Fixed window: {season_start:%Y-%m-%d} to {season_end:%Y-%m-%d}. Each basin contributes one available monthly mean; zeros are retained.\n"
         "The complete-window subset has valid qmax on every day of this window; its membership is fixed across months.\n"
         "Missing months remain NaN. This is a descriptive sensitivity comparison, not a model basin filter or monthly runoff volume.", ha="center", fontsize=9)
season_figure = FIGURE_DIR / "A1_P5_qmax_seasonality_by_elevation.png"
fig.savefig(season_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

season_ranked = season_info.assign(basin_number=season_info["basin_id"].astype(int)).sort_values(["elev_mean_m", "basin_number"])
missing_grid = A1_QMAX_SEASON_BASIN_MONTHLY.pivot(index="basin_id", columns="hydro_month", values="missing_pct").reindex(
    index=season_ranked["basin_id"], columns=range(1, 13))
fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.15, 1]})
mesh = axes[0].imshow(missing_grid.to_numpy(), aspect="auto", interpolation="nearest", vmin=0, vmax=100, cmap="YlOrRd")
axes[0].set_xticks(season_x, season_month_labels, fontsize=9)
rank_ticks = np.unique(np.linspace(0, len(season_ranked) - 1, min(6, len(season_ranked))).round().astype(int))
axes[0].set_yticks(rank_ticks, [f"{i + 1} (ID {season_ranked.iloc[i]['basin_id']})" for i in rank_ticks], fontsize=8)
axes[0].set(title="(a) Missing Q by basin and month", ylabel="Basin rank by elev_mean, lowest to highest")
fig.colorbar(mesh, ax=axes[0], shrink=0.8, pad=0.015, label="Missing expected basin-days (%)")
season_colors = ["#267c9d", "#e58b32", "#4e995b", "#a0579b"]
for color, group in zip(season_colors, groups.itertuples()):
    profile = A1_QMAX_SEASON_MONTHLY_SUMMARY.loc[
        A1_QMAX_SEASON_MONTHLY_SUMMARY["elevation_group"].eq(group.elevation_group)].sort_values("hydro_month")
    axes[1].plot(season_x, profile["missing_pct"], color=color, marker="o", markersize=3,
                 label=f"{group.elevation_group}: {group.elev_min_m:,.0f}–{group.elev_max_m:,.0f} m")
axes[1].set_xticks(season_x, season_month_labels, fontsize=9)
axes[1].set(title="(b) Missing Q within each elevation group", ylabel="Missing / expected basin-days (%)", ylim=(0, None))
axes[1].grid(alpha=0.18)
for ax in axes:
    ax.set_xlabel("Month in the October–September water year")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=4, frameon=False, fontsize=9)
fig.suptitle(f"Observation coverage for seasonal qmax: water years {water_year_label}", fontsize=15, y=0.98)
fig.subplots_adjust(left=0.085, right=0.985, top=0.85, bottom=0.27, wspace=0.35)
fig.text(0.5, 0.025, "Expected counts use every calendar day of the fixed window, including leap days. No missing qmax is filled or set to zero.\n"
         "Shared first/last dates do not imply complete observations. Elevation groups use elev_mean of the basin, not gauge elevation.",
         ha="center", fontsize=9)
season_coverage_figure = FIGURE_DIR / "A1_P5_qmax_seasonality_coverage.png"
fig.savefig(season_coverage_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
for suffix, table in [("qmax_season_scope", A1_QMAX_SEASON_SCOPE), ("qmax_season_elevation_groups", groups),
                      ("qmax_season_basin_monthly", A1_QMAX_SEASON_BASIN_MONTHLY),
                      ("qmax_season_monthly_summary", A1_QMAX_SEASON_MONTHLY_SUMMARY),
                      ("qmax_season_basin_coverage", A1_QMAX_SEASON_BASIN_COVERAGE)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    print("(1) Khoảng first/last chung và năm thủy văn Oct–Sep trọn vẹn dùng cho khảo sát mùa:")
    display(A1_QMAX_SEASON_SCOPE)
    print("(2) Nhóm elev_mean lưu vực; complete_window chỉ là tập đối chiếu, không thay danh sách mô hình:")
    display(groups)
    print("(3) Trung vị ngang lưu vực của mean qmax ngày theo tháng; hai tập giữ trọng số mỗi lưu vực bằng nhau:")
    display(A1_QMAX_SEASON_MONTHLY_SUMMARY.pivot(index=["hydro_month", "month_name"], columns="elevation_group",
        values=["median_available_basin_month_means", "median_complete_basin_month_means"]))
    print("(4) Thiếu Q (%) theo nhóm và tháng, mẫu số là lịch đầy đủ của khoảng khảo sát:")
    display(A1_QMAX_SEASON_MONTHLY_SUMMARY.pivot(index=["hydro_month", "month_name"], columns="elevation_group", values="missing_pct"))
    print("(5) Mười hai lưu vực thiếu nhiều nhất trong khoảng này; không suy nguyên nhân từ bảng:")
    display(season_info.sort_values(["missing_pct", "basin_id"], ascending=[False, True])[[
        "basin_id", "name", "elev_mean_m", "elevation_group", "expected_days", "valid_days", "missing_days", "missing_pct", "complete_window"]].head(12))
print("Số đếm theo tháng/nhóm cộng khớp lịch khảo sát; các tháng không có quan trắc giữ NaN.")
print("Đã lưu hai hình:", season_figure, "|", season_coverage_figure)
print("Đã lưu năm CSV vào:", TABLE_DIR)
print(f"Thời gian khảo sát mùa: {time.perf_counter() - started_seasonality:.1f} giây.")

# %% [markdown]
# ### 5.6. Đường duy trì qmax và các đoạn giá trị hằng ít nhất 10 ngày
#
# Đường phân phối dùng cùng khoảng 5.5; đoạn hằng quét toàn lịch 5.1 bằng qmax m³/s gốc.

# %% Phần 5.6 - Đường duy trì qmax và các đoạn giá trị hằng ít nhất 10 ngày
# Đường phân phối dùng cùng khoảng 5.5; đoạn hằng quét toàn lịch 5.1 bằng qmax m³/s gốc.
# Giá trị bằng nhau chính xác ở độ chính xác lưu trong dữ liệu; NaN ngắt đoạn; số 0 được giữ riêng.
started_fdc_constant = time.perf_counter()
fdc_ids = [str(value) for value in A1_QMAX_VALUE_IDS]
fdc_calendar = pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
fdc_raw = np.asarray(A1_QMAX_VALUES_M3S, dtype=np.float64)
if (len(set(fdc_ids)) != len(fdc_ids) or set(fdc_ids) != set(selected_ids)
        or fdc_raw.shape != (len(fdc_ids), len(fdc_calendar))
        or not fdc_calendar.equals(pd.date_range(fdc_calendar.min(), fdc_calendar.max()))):
    raise ValueError("Cache Q/ID/lịch 5.1 không khớp để đếm đoạn liên tục.")
fdc_areas = A1_QMAX_AREA_CONVERSION.set_index("basin_id").reindex(fdc_ids)
fdc_metadata = A1_QMAX_SEASON_BASIN_COVERAGE.set_index("basin_id").reindex(fdc_ids)
if fdc_metadata["elevation_group"].isna().any() or not (
        np.isfinite(fdc_areas["m3_s_to_mm_day_factor"]) & fdc_areas["m3_s_to_mm_day_factor"].gt(0)).all():
    raise ValueError("Metadata 5.1/5.5 chưa khớp ID hoặc hệ số đổi đơn vị không dương.")
fdc_start = pd.Timestamp(A1_QMAX_SEASON_SCOPE.iloc[0]["window_start"])
fdc_end = pd.Timestamp(A1_QMAX_SEASON_SCOPE.iloc[0]["window_end"])
fdc_positions = fdc_calendar.get_indexer(pd.date_range(fdc_start, fdc_end))
if np.any(fdc_positions < 0):
    raise ValueError("Khoảng 5.5 nằm ngoài lịch cache Q.")
fdc_period_positions = {period: fdc_calendar.get_indexer(pd.date_range(start, end))
                        for period, (start, end) in PERIODS.items()}
if any(np.any(positions < 0) for positions in fdc_period_positions.values()):
    raise ValueError("Một kỳ train/validation/test nằm ngoài lịch quét; chưa đếm thiếu ngoài cache.")
FDC_EXCEEDANCE_PCT = np.unique(np.r_[np.linspace(0, 100, 101), [0.1, 0.5, 99.5, 99.9]])
CONSTANT_MIN_DAYS = 10
fdc_curves = np.full((len(fdc_ids), len(FDC_EXCEEDANCE_PCT)), np.nan)
fdc_grid_rows, fdc_basin_rows, constant_rows, constant_basin_rows, variability_rows = [], [], [], [], []


def a1_equal_q_runs(values):
    """Trả chỉ số đầu/cuối đóng của các đoạn Q hữu hạn bằng nhau, kể cả đoạn 1 ngày."""
    values = np.asarray(values)
    if not len(values):
        return np.array([], dtype=int), np.array([], dtype=int)
    valid = np.isfinite(values)
    same_adjacent = valid[1:] & valid[:-1] & (values[1:] == values[:-1])
    starts = np.flatnonzero(valid & ~np.r_[False, same_adjacent])
    ends = np.flatnonzero(valid & ~np.r_[same_adjacent, False])
    if len(starts) != len(ends):
        raise ValueError("Đầu/cuối đoạn hằng chưa ghép được.")
    return starts, ends


for i, basin_id in enumerate(fdc_ids):
    raw = fdc_raw[i]
    valid = np.isfinite(raw)
    if not valid.any() or np.any(raw[valid] < 0):
        raise ValueError(f"ID {basin_id}: không có Q hợp lệ hoặc cache có Q âm.")
    factor = float(fdc_areas.loc[basin_id, "m3_s_to_mm_day_factor"])
    values = raw[fdc_positions] * factor
    observed = values[np.isfinite(values)]
    complete_window = bool(np.isfinite(values).all())
    if complete_window != bool(fdc_metadata.loc[basin_id, "complete_window"]):
        raise ValueError(f"ID {basin_id}: mặt nạ đủ khoảng khác 5.5.")
    curve = np.quantile(observed, 1 - FDC_EXCEEDANCE_PCT / 100, method="linear") if len(observed) else np.full(len(FDC_EXCEEDANCE_PCT), np.nan)
    if len(observed) and np.any(np.diff(curve) > 1e-12):
        raise ValueError(f"ID {basin_id}: đường phân vị không giảm theo mức vượt.")
    fdc_curves[i] = curve
    fdc_basin_rows.append({"basin_id": basin_id, "name": gauge.loc[basin_id, "name"],
        "area_gov_km2": fdc_areas.loc[basin_id, "area_gov_km2"],
        "elevation_group": fdc_metadata.loc[basin_id, "elevation_group"], "complete_window": complete_window,
        "window_valid_days": len(observed), "window_zero_days": int((observed == 0).sum()),
        "at_1pct_exceed_mm_day": float(np.quantile(observed, 0.99)) if len(observed) else np.nan,
        "at_50pct_exceed_mm_day": float(np.quantile(observed, 0.5)) if len(observed) else np.nan,
        "at_99pct_exceed_mm_day": float(np.quantile(observed, 0.01)) if len(observed) else np.nan})
    for p, value in zip(FDC_EXCEEDANCE_PCT, curve):
        fdc_grid_rows.append({"basin_id": basin_id, "exceedance_level_pct": p, "qmax_mm_day": value,
            "elevation_group": fdc_metadata.loc[basin_id, "elevation_group"],
            "window_valid_days": len(observed), "complete_window": complete_window})
    starts, ends = a1_equal_q_runs(raw)
    lengths = ends - starts + 1
    zero_runs = raw[starts] == 0
    long_runs = lengths >= CONSTANT_MIN_DAYS
    for start, end, length, is_zero in zip(starts[long_runs], ends[long_runs], lengths[long_runs], zero_runs[long_runs]):
        overlaps = {}
        for period, positions in fdc_period_positions.items():
            overlaps[f"{period}_overlap_days"] = int(max(0, min(end, positions[-1]) - max(start, positions[0]) + 1))
        constant_rows.append({"run_id": len(constant_rows) + 1, "basin_id": basin_id,
            "start_date": fdc_calendar[start], "end_date": fdc_calendar[end], "run_days": int(length),
            "qmax_m3s": float(raw[start]), "qmax_mm_day": float(raw[start] * factor), "is_zero": bool(is_zero),
            "touches_audit_start": start == 0, "touches_audit_end": end == len(raw) - 1, **overlaps})
    n_zero = int(lengths[long_runs & zero_runs].sum())
    n_positive = int(lengths[long_runs & ~zero_runs].sum())
    if n_zero + n_positive > valid.sum():
        raise ValueError(f"ID {basin_id}: các đoạn hằng bị đếm chồng.")
    constant_basin_rows.append({"basin_id": basin_id, "name": gauge.loc[basin_id, "name"],
        "valid_days": int(valid.sum()), "qualifying_runs": int(long_runs.sum()),
        "zero_constant_days": n_zero, "positive_constant_days": n_positive,
        "constant_days": n_zero + n_positive, "constant_pct_of_valid_days": 100 * (n_zero + n_positive) / valid.sum(),
        "zero_constant_pct_of_valid_days": 100 * n_zero / valid.sum(),
        "positive_constant_pct_of_valid_days": 100 * n_positive / valid.sum(),
        "longest_equal_run_days": int(lengths.max())})
    for period, positions in fdc_period_positions.items():
        period_values = raw[positions] * factor
        finite = period_values[np.isfinite(period_values)]
        clipped_lengths = np.maximum(0, np.minimum(ends, positions[-1]) - np.maximum(starts, positions[0]) + 1)
        mean = float(np.mean(finite, dtype=np.float64)) if len(finite) else np.nan
        std = float(np.std(finite, dtype=np.float64, ddof=0)) if len(finite) else np.nan
        variability_rows.append({"basin_id": basin_id, "period": period, "expected_days": len(positions),
            "valid_days": len(finite), "zero_days": int((finite == 0).sum()), "distinct_values": len(np.unique(finite)),
            "mean_mm_day": mean, "std_ddof0_mm_day": std,
            "variance_ddof0_mm_day2": float(np.var(finite, dtype=np.float64, ddof=0)) if len(finite) else np.nan,
            "cv_std_over_mean": std / mean if mean > 0 else np.nan,
            "min_mm_day": float(finite.min()) if len(finite) else np.nan,
            "max_mm_day": float(finite.max()) if len(finite) else np.nan,
            "longest_equal_run_days_within_period": int(clipped_lengths.max()) if len(clipped_lengths) else 0,
            "longest_zero_run_days_within_period": int(clipped_lengths[zero_runs].max()) if zero_runs.any() else 0})
A1_QMAX_FDC_BASIN_GRID = pd.DataFrame(fdc_grid_rows)
fdc_basin_table = pd.DataFrame(fdc_basin_rows)
fdc_complete = fdc_metadata["complete_window"].to_numpy(dtype=bool)
group_curve_rows = []
for group in A1_QMAX_SEASON_ELEVATION_GROUPS["elevation_group"]:
    member_mask = fdc_metadata["elevation_group"].eq(group).to_numpy() & fdc_complete
    group_curves = fdc_curves[member_mask]
    q25, med, q75 = np.quantile(group_curves, [0.25, 0.5, 0.75], axis=0) if len(group_curves) else np.full((3, len(FDC_EXCEEDANCE_PCT)), np.nan)
    for j, p in enumerate(FDC_EXCEEDANCE_PCT):
        group_curve_rows.append({"elevation_group": group, "exceedance_level_pct": p,
            "complete_window_basins": len(group_curves), "p25_basin_quantiles": q25[j],
            "median_basin_quantiles": med[j], "p75_basin_quantiles": q75[j]})
A1_QMAX_FDC_GROUP_SUMMARY = pd.DataFrame(group_curve_rows)
ranked_areas = fdc_areas.assign(basin_number=[int(b) for b in fdc_ids]).sort_values(["area_gov_km2", "basin_number"])
example_roles = {}
for index, role in [(0, "smallest_area"), ((len(ranked_areas) - 1) // 2, "median_area_rank"), (len(ranked_areas) - 1, "largest_area")]:
    example_roles.setdefault(str(ranked_areas.index[index]), []).append(role)
for basin_id in ["55", "742", "758"]:
    if basin_id in fdc_ids:
        example_roles.setdefault(basin_id, []).append("audit_case")
A1_QMAX_FDC_EXAMPLES = fdc_basin_table.set_index("basin_id").loc[list(example_roles)].reset_index()
A1_QMAX_FDC_EXAMPLES["selection_role"] = A1_QMAX_FDC_EXAMPLES["basin_id"].map(lambda b: " | ".join(example_roles[b]))
constant_columns = ["run_id", "basin_id", "start_date", "end_date", "run_days", "qmax_m3s", "qmax_mm_day", "is_zero",
    "touches_audit_start", "touches_audit_end"] + [f"{period}_overlap_days" for period in PERIODS]
A1_QMAX_CONSTANT_RUNS = pd.DataFrame(constant_rows, columns=constant_columns)
A1_QMAX_CONSTANT_BY_BASIN = pd.DataFrame(constant_basin_rows)
A1_QMAX_VARIABILITY_BY_PERIOD = pd.DataFrame(variability_rows)
constant_summary_rows = []
for is_zero, label in [(True, "zero_value"), (False, "positive_value")]:
    rows = A1_QMAX_CONSTANT_RUNS.loc[A1_QMAX_CONSTANT_RUNS["is_zero"].eq(is_zero)]
    constant_summary_rows.append({"kind": label, "minimum_run_days": CONSTANT_MIN_DAYS, "runs": len(rows),
        "basins": rows["basin_id"].nunique(), "constant_days": int(rows["run_days"].sum()),
        "median_run_days": float(rows["run_days"].median()) if len(rows) else np.nan,
        "max_run_days": int(rows["run_days"].max()) if len(rows) else 0})
A1_QMAX_CONSTANT_SUMMARY = pd.DataFrame(constant_summary_rows)
A1_QMAX_FDC_SCOPE = pd.DataFrame([{"window_start": fdc_start, "window_end": fdc_end, "calendar_days": len(fdc_positions),
    "basins": len(fdc_ids), "complete_window_basins_for_group_curves": int(fdc_complete.sum()),
    "quantile_grid_points": len(FDC_EXCEEDANCE_PCT), "constant_audit_start": fdc_calendar.min(),
    "constant_audit_end": fdc_calendar.max(), "minimum_constant_run_days": CONSTANT_MIN_DAYS}])
if A1_QMAX_CONSTANT_BY_BASIN["constant_days"].sum() != A1_QMAX_CONSTANT_RUNS["run_days"].sum():
    raise ValueError("Số ngày đoạn hằng giữa bảng trạm và bảng đoạn chưa khớp.")

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
fdc_colors = ["#267c9d", "#e58b32", "#4e995b", "#a0579b"]
for color, group in zip(fdc_colors, A1_QMAX_SEASON_ELEVATION_GROUPS["elevation_group"]):
    curve = A1_QMAX_FDC_GROUP_SUMMARY.loc[A1_QMAX_FDC_GROUP_SUMMARY["elevation_group"].eq(group)]
    axes[0].fill_between(FDC_EXCEEDANCE_PCT, curve["p25_basin_quantiles"].to_numpy(),
                         curve["p75_basin_quantiles"].to_numpy(), color=color, alpha=0.12)
    axes[0].plot(FDC_EXCEEDANCE_PCT, curve["median_basin_quantiles"], color=color,
                 label=f"{group} (n={int(curve['complete_window_basins'].iloc[0])})")
for row in A1_QMAX_FDC_EXAMPLES.itertuples():
    axes[1].plot(FDC_EXCEEDANCE_PCT, fdc_curves[fdc_ids.index(row.basin_id)],
                 label=f"ID {row.basin_id} (n={row.window_valid_days})", linewidth=1.3)
for ax in axes:
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.6)
    ax.set(xlim=(0, 100), xlabel="Exceedance level p (%)", ylabel="qmax (mm/day; symlog)")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=2, fontsize=8, frameon=False)
displayed_values = np.r_[A1_QMAX_FDC_GROUP_SUMMARY["p75_basin_quantiles"].to_numpy(),
    fdc_curves[[fdc_ids.index(b) for b in example_roles]].ravel()]
displayed_values = displayed_values[np.isfinite(displayed_values)]
display_max = float(displayed_values.max()) if len(displayed_values) else 0
axes[0].set_ylim(0, display_max * 1.2 if display_max > 0 else 1)
axes[0].set_title("(a) Complete-window basins: median and IQR by elevation")
axes[1].set_title("(b) Selected diagnostic cases: available observations")
fig.suptitle(f"Duration curves of daily qmax: {fdc_start:%Y-%m-%d} to {fdc_end:%Y-%m-%d}", fontsize=15, y=0.98)
fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.40, wspace=0.2)
fig.text(0.5, 0.02, "Q(p) = linearly interpolated sample quantile at 1−p/100; p=0/100 are sample max/min. Zeros are retained.\n"
         "Left: complete-window subset, one curve per basin. Right: area-rank min/median/max and available audit IDs 55/742/758.\n"
         "n is the number of basins on the left, and observed days on the right. Missing days are omitted, never filled.\n"
         "Symlog is linear from 0 to 0.01. These are daily-maximum curves, with no time-order or runoff-volume information.",
         ha="center", fontsize=9)
fdc_figure = FIGURE_DIR / "A1_P5_qmax_duration_curves.png"
fig.savefig(fdc_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for is_zero, color, label in [(True, "#d99032", "Zero-value runs"), (False, "#227fa4", "Positive-value runs")]:
    lengths = A1_QMAX_CONSTANT_RUNS.loc[A1_QMAX_CONSTANT_RUNS["is_zero"].eq(is_zero), "run_days"].to_numpy()
    if len(lengths):
        unique, counts = np.unique(lengths, return_counts=True)
        axes[0].step(np.r_[unique[0] / 1.1, unique], np.r_[0, 100 * np.cumsum(counts) / len(lengths)],
                     where="post", color=color, label=f"{label} (n={len(lengths)})")
if len(A1_QMAX_CONSTANT_RUNS):
    axes[0].set_xscale("log")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, fontsize=8)
else:
    axes[0].text(0.5, 0.5, "No qualifying runs", transform=axes[0].transAxes, ha="center")
axes[0].set(title="(a) Lengths of qualifying constant-qmax runs", xlabel="Consecutive identical valid days (log axis when runs exist)",
             ylabel="Runs at or below length (%)", ylim=(0, 105))
axes[0].grid(alpha=0.18)
top_constant = A1_QMAX_CONSTANT_BY_BASIN.loc[A1_QMAX_CONSTANT_BY_BASIN["constant_days"].gt(0)].sort_values(
    ["constant_pct_of_valid_days", "constant_days"], ascending=False).head(10)
if len(top_constant):
    ys = np.arange(len(top_constant))
    axes[1].barh(ys, top_constant["zero_constant_pct_of_valid_days"], color="#d99032", label="Zero-value runs")
    axes[1].barh(ys, top_constant["positive_constant_pct_of_valid_days"],
                 left=top_constant["zero_constant_pct_of_valid_days"], color="#227fa4", label="Positive-value runs")
    axes[1].set_yticks(ys, [f"ID {b}" for b in top_constant["basin_id"]])
    axes[1].invert_yaxis()
    for y, value in zip(ys, top_constant["constant_pct_of_valid_days"]):
        axes[1].text(value + 0.6, y, f"{value:.1f}%", va="center", fontsize=8)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, fontsize=8)
else:
    axes[1].text(0.5, 0.5, "No qualifying basins", transform=axes[1].transAxes, ha="center")
axes[1].set(title="(b) Largest fractions of observed days in long constant runs", xlabel="Qualifying constant days / full-record valid days (%)", xlim=(0, 108))
axes[1].set_xticks([0, 25, 50, 75, 100])
axes[1].grid(axis="x", alpha=0.18)
fig.suptitle(f"Identical qmax for at least {CONSTANT_MIN_DAYS} consecutive calendar days", fontsize=15, y=0.98)
fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.29, wspace=0.3)
fig.text(0.5, 0.025, f"Audit: {fdc_calendar.min():%Y-%m-%d} to {fdc_calendar.max():%Y-%m-%d}. Equality uses stored raw qmax in m³/s, with no rounding or tolerance.\n"
         "Missing days break a run; zeros are valid and reported separately. Runs are not split at year or experiment boundaries.\n"
         "The 10-day rule is a diagnostic flag. It does not establish an error, justify deleting observations, or identify a physical cause.", ha="center", fontsize=9)
constant_figure = FIGURE_DIR / "A1_P5_qmax_constant_runs.png"
fig.savefig(constant_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
for suffix, table in [("qmax_fdc_scope", A1_QMAX_FDC_SCOPE), ("qmax_fdc_basin_grid", A1_QMAX_FDC_BASIN_GRID),
                      ("qmax_fdc_group_summary", A1_QMAX_FDC_GROUP_SUMMARY), ("qmax_fdc_examples", A1_QMAX_FDC_EXAMPLES),
                      ("qmax_constant_runs", A1_QMAX_CONSTANT_RUNS), ("qmax_constant_by_basin", A1_QMAX_CONSTANT_BY_BASIN),
                      ("qmax_variability_by_period", A1_QMAX_VARIABILITY_BY_PERIOD)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    print("(1) Khoảng của đường phân vị và của phép quét đoạn hằng; hai phạm vi được phân biệt rõ:")
    display(A1_QMAX_FDC_SCOPE)
    print("(2) Các ID minh họa được chọn theo diện tích và hồ sơ kiểm; không phải mẫu đại diện ngẫu nhiên:")
    display(A1_QMAX_FDC_EXAMPLES)
    print("(3) Đoạn qmax bằng nhau chính xác ít nhất 10 ngày lịch, số 0 vẫn hợp lệ:")
    display(A1_QMAX_CONSTANT_SUMMARY)
    print("(4) Mười lăm đoạn hằng dài nhất; overlap là số ngày giao, không tách đoạn gốc:")
    display(A1_QMAX_CONSTANT_RUNS.sort_values(["run_days", "basin_id"], ascending=[False, True]).head(15))
    print("(5) Biến thiên qmax mm/ngày theo kỳ của các ID hồ sơ 55/742/758/800; std và variance dùng ddof=0:")
    display(A1_QMAX_VARIABILITY_BY_PERIOD.loc[A1_QMAX_VARIABILITY_BY_PERIOD["basin_id"].isin(["55", "742", "758", "800"])])
print("ID hồ sơ không có trong cache:", sorted(set(["55", "742", "758", "800"]) - set(fdc_ids), key=int))
print("Số ngày đoạn hằng giữa bảng trạm và bảng đoạn cộng khớp; đoạn thiếu không được nối qua.")
print("Đã lưu hai hình:", fdc_figure, "|", constant_figure)
print("Đã lưu bảy CSV vào:", TABLE_DIR)
print(f"Thời gian khảo sát đường phân vị/đoạn hằng: {time.perf_counter() - started_fdc_constant:.1f} giây.")

# %% [markdown]
# ### 5.7. Quan hệ Q cùng ngày và đối chiếu các đỉnh lớn với qmean, qmin
#
# Quét CSV Q gốc của 451 ID; giữ riêng số âm/sentinel/rỗng. Q âm không làm mất các Q khác cùng dòng.

# %% Phần 5.7 - Quan hệ Q cùng ngày và đối chiếu các đỉnh lớn với qmean, qmin
# Quét CSV Q gốc của 451 ID; giữ riêng số âm/sentinel/rỗng. Q âm không làm mất các Q khác cùng dòng.
# Không đặt ngưỡng ngoại lai để sửa/xóa số. Tỷ số chỉ tính khi qmean>0 và qmax hợp lệ không âm.
import matplotlib.dates as mdates

started_peak_check = time.perf_counter()
peak_ids = [str(value) for value in A1_QMAX_VALUE_IDS]
peak_calendar = pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
peak_qmax = np.asarray(A1_QMAX_VALUES_M3S, dtype=np.float64)
if (set(peak_ids) != set(selected_ids) or len(set(peak_ids)) != len(peak_ids)
        or peak_qmax.shape != (len(peak_ids), len(peak_calendar))):
    raise ValueError("Cache Q/ID/lịch 5.1 chưa khớp.")
peak_areas = A1_QMAX_AREA_CONVERSION.set_index("basin_id").reindex(peak_ids)
peak_factors = peak_areas["m3_s_to_mm_day_factor"].to_numpy()
if not (np.isfinite(peak_factors) & (peak_factors > 0)).all():
    raise ValueError("Hệ số area_gov không hữu hạn/dương.")
A1_QMIN_PARSED_M3S = np.full_like(peak_qmax, np.nan)
A1_QMEAN_PARSED_M3S = np.full_like(peak_qmax, np.nan)
A1_QMAX_QMEAN_RATIO = np.full_like(peak_qmax, np.nan)
q_profile_rows, q_order_rows, q_flag_frames = [], [], []
order_pairs = {"qmin_gt_qmean": ("qmin", "qmean"), "qmean_gt_qmax": ("qmean", "qmax"),
               "qmin_gt_qmax": ("qmin", "qmax")}


def a1_q_ratio(maximum, mean):
    maximum, mean = np.broadcast_arrays(np.asarray(maximum, dtype=float), np.asarray(mean, dtype=float))
    result = np.full(maximum.shape, np.nan)
    valid = np.isfinite(maximum) & (maximum >= 0) & np.isfinite(mean) & (mean > 0)
    np.divide(maximum, mean, out=result, where=valid)
    return result


for i, basin_id in enumerate(peak_ids):
    raw = pd.read_csv(A1_DAILY_FILES["D"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                      usecols=DATE_COLS + ["qmin", "qmean", "qmax"], keep_default_na=False, na_filter=False)
    dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
    if (dates.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing
            or not dates.isin(peak_calendar).all()):
        raise ValueError(f"ID {basin_id}: lịch Q không hợp lệ hoặc khác cache 5.1.")
    positions = peak_calendar.get_indexer(dates)
    parsed, flags = {}, {}
    row_summary = {"basin_id": basin_id}
    for variable in ["qmin", "qmean", "qmax"]:
        text = raw[variable].str.strip()
        number = pd.to_numeric(text, errors="coerce").to_numpy(dtype=float)
        blank = text.eq("").to_numpy()
        finite = np.isfinite(number)
        sentinel = number == -999
        negative = finite & (number < 0) & ~sentinel
        q_profile_rows.append({"basin_id": basin_id, "variable": variable, "expected_days": len(peak_calendar),
            "rows_present": len(raw), "missing_rows": len(peak_calendar) - len(raw), "blank_cells": int(blank.sum()),
            "minus999_cells": int(sentinel.sum()), "negative_except_minus999": int(negative.sum()),
            "non_numeric_nonblank": int((~blank & np.isnan(number)).sum()), "infinite_cells": int(np.isinf(number).sum()),
            "finite_nonnegative_cells": int((finite & (number >= 0)).sum())})
        parsed[variable] = np.where(finite & ~sentinel, number, np.nan)  # Giữ số âm để gắn cờ/kiểm quan hệ.
        flags[f"negative_{variable}"] = negative
    A1_QMIN_PARSED_M3S[i, positions] = parsed["qmin"]
    A1_QMEAN_PARSED_M3S[i, positions] = parsed["qmean"]
    qmax_read = np.full(len(peak_calendar), np.nan)
    qmax_read[positions] = np.where(parsed["qmax"] >= 0, parsed["qmax"], np.nan)
    if not np.allclose(qmax_read, peak_qmax[i], rtol=0, atol=0, equal_nan=True):
        raise ValueError(f"ID {basin_id}: qmax đọc lại khác giá trị cache 5.1.")
    for check, (left, right) in order_pairs.items():
        comparable = np.isfinite(parsed[left]) & np.isfinite(parsed[right])
        flags[check] = comparable & (parsed[left] > parsed[right])
        row_summary[f"{check}_compared"] = int(comparable.sum())
    zero_mean = np.isfinite(parsed["qmax"]) & (parsed["qmax"] >= 0) & (parsed["qmean"] == 0)
    flags["zero_qmean_positive_qmax"] = zero_mean & (parsed["qmax"] > 0)
    row_summary["zero_qmean_positive_qmax_compared"] = int(zero_mean.sum())
    row_summary.update({check: int(mask.sum()) for check, mask in flags.items()})
    ratios = a1_q_ratio(qmax_read, A1_QMEAN_PARSED_M3S[i])
    A1_QMAX_QMEAN_RATIO[i] = ratios
    observed_ratios = ratios[np.isfinite(ratios)]
    row_summary.update(ratio_days=len(observed_ratios),
        ratio_median=float(np.median(observed_ratios)) if len(observed_ratios) else np.nan,
        ratio_p99=float(np.quantile(observed_ratios, .99)) if len(observed_ratios) else np.nan,
        ratio_max=float(observed_ratios.max()) if len(observed_ratios) else np.nan)
    q_order_rows.append(row_summary)
    any_flag = np.logical_or.reduce(list(flags.values()))
    if any_flag.any():
        frame = pd.DataFrame({"basin_id": basin_id, "date": dates[any_flag]})
        for variable in ["qmin", "qmean", "qmax"]:
            frame[f"{variable}_raw"] = raw.loc[any_flag, variable].to_numpy()
            frame[f"{variable}_m3s"] = parsed[variable][any_flag]
        for check, mask in flags.items():
            frame[check] = mask[any_flag]
        q_flag_frames.append(frame)
    if (i + 1) % 100 == 0 or i + 1 == len(peak_ids):
        print(f"Đã kiểm quan hệ Q {i + 1}/{len(peak_ids)} lưu vực", flush=True)
A1_Q_VARIABLE_PROFILE = pd.DataFrame(q_profile_rows)
profile_parts = ["missing_rows", "blank_cells", "minus999_cells", "negative_except_minus999",
                 "non_numeric_nonblank", "infinite_cells", "finite_nonnegative_cells"]
if not A1_Q_VARIABLE_PROFILE[profile_parts].sum(axis=1).eq(A1_Q_VARIABLE_PROFILE["expected_days"]).all():
    raise ValueError("Các loại ô Q không cộng khớp lịch đầy đủ.")
A1_Q_ORDER_BY_BASIN = pd.DataFrame(q_order_rows)
flag_columns = ["basin_id", "date"] + [f"{v}_{suffix}" for v in ["qmin", "qmean", "qmax"] for suffix in ["raw", "m3s"]] + list(flags)
A1_Q_ORDER_FLAGS = pd.concat(q_flag_frames, ignore_index=True) if q_flag_frames else pd.DataFrame(columns=flag_columns)
quality_summary = A1_Q_VARIABLE_PROFILE.groupby("variable").sum(numeric_only=True)
quality_summary["basins"] = A1_Q_VARIABLE_PROFILE.groupby("variable")["basin_id"].nunique()
order_summary = pd.DataFrame([{"check": check, "compared_days": A1_Q_ORDER_BY_BASIN[f"{check}_compared"].sum(),
    "flagged_days": A1_Q_ORDER_BY_BASIN[check].sum(), "affected_basins": int(A1_Q_ORDER_BY_BASIN[check].gt(0).sum())}
    for check in list(order_pairs) + ["zero_qmean_positive_qmax"]])
peak_rows, peak_day_rows = [], []
for i, basin_id in enumerate(peak_ids):
    values = peak_qmax[i]
    maximum = float(np.nanmax(values))
    max_positions = np.flatnonzero(values == maximum)
    first = max_positions[0]
    row = {"basin_id": basin_id, "name": gauge.loc[basin_id, "name"],
        "area_gov_km2": peak_areas.loc[basin_id, "area_gov_km2"], "max_qmax_m3s": maximum,
        "max_qmax_mm_day": maximum * peak_factors[i], "first_max_date": peak_calendar[first],
        "last_max_date": peak_calendar[max_positions[-1]], "days_at_global_max": len(max_positions),
        "qmin_m3s_at_first_max": A1_QMIN_PARSED_M3S[i, first], "qmean_m3s_at_first_max": A1_QMEAN_PARSED_M3S[i, first],
        "ratio_at_first_max": A1_QMAX_QMEAN_RATIO[i, first]}
    counted = 0
    for period, (start, end) in PERIODS.items():
        in_period = (peak_calendar[max_positions] >= pd.Timestamp(start)) & (peak_calendar[max_positions] <= pd.Timestamp(end))
        row[f"max_days_in_{period}"] = int(in_period.sum())
        counted += int(in_period.sum())
    row["max_days_outside_periods"] = len(max_positions) - counted
    if row["max_days_outside_periods"] < 0:
        raise ValueError("Các kỳ thực nghiệm chồng nhau; chưa cộng số ngày đồng cực đại.")
    peak_rows.append(row)
    for position in max_positions:
        label = next((period for period, (start, end) in PERIODS.items()
                      if pd.Timestamp(start) <= peak_calendar[position] <= pd.Timestamp(end)), "outside_periods")
        peak_day_rows.append({"basin_id": basin_id, "date": peak_calendar[position], "period": label,
            "is_first_max_date": position == first, "qmax_m3s": maximum,
            "qmean_m3s": A1_QMEAN_PARSED_M3S[i, position], "qmin_m3s": A1_QMIN_PARSED_M3S[i, position],
            "qmax_mm_day": maximum * peak_factors[i], "qmax_qmean_ratio": A1_QMAX_QMEAN_RATIO[i, position]})
A1_Q_PEAK_COMPARISON = pd.DataFrame(peak_rows).sort_values(["max_qmax_mm_day", "basin_id"], ascending=[False, True]).reset_index(drop=True)
A1_Q_PEAK_DAYS = pd.DataFrame(peak_day_rows)
context_indices, context_years = [], set()
for index, row in A1_Q_PEAK_COMPARISON.iterrows():
    year = row["first_max_date"].year
    if year not in context_years:
        context_indices.append(index)
        context_years.add(year)
    if len(context_indices) == 6:
        break
context_selection = A1_Q_PEAK_COMPARISON.loc[context_indices].copy()
A1_Q_PEAK_COMPARISON["selected_for_context"] = A1_Q_PEAK_COMPARISON.index.isin(context_indices)
context_rows = []
for row in context_selection.itertuples():
    i = peak_ids.index(row.basin_id)
    mask = (peak_calendar >= row.first_max_date - pd.Timedelta(days=15)) & (peak_calendar <= row.first_max_date + pd.Timedelta(days=15))
    for j in np.flatnonzero(mask):
        context_rows.append({"basin_id": row.basin_id, "date": peak_calendar[j], "first_max_date": row.first_max_date,
            "qmin_m3s": A1_QMIN_PARSED_M3S[i, j], "qmean_m3s": A1_QMEAN_PARSED_M3S[i, j], "qmax_m3s": peak_qmax[i, j],
            "qmin_mm_day": A1_QMIN_PARSED_M3S[i, j] * peak_factors[i],
            "qmean_mm_day": A1_QMEAN_PARSED_M3S[i, j] * peak_factors[i], "qmax_mm_day": peak_qmax[i, j] * peak_factors[i]})
A1_Q_PEAK_CONTEXT = pd.DataFrame(context_rows)
ratio_rows = []
for scope, values in [("all_valid_daily_pairs", A1_QMAX_QMEAN_RATIO.ravel()),
                      ("one_first_max_pair_per_basin", A1_Q_PEAK_COMPARISON["ratio_at_first_max"].to_numpy())]:
    values = values[np.isfinite(values)]
    ratio_rows.append({"scope": scope, "pairs": len(values), **{
        name: float(np.quantile(values, probability)) if len(values) else np.nan
        for name, probability in [("min", 0), ("p01", .01), ("median", .5), ("p95", .95), ("p99", .99), ("max", 1)]}})
A1_Q_RATIO_SUMMARY = pd.DataFrame(ratio_rows)
# Liệt kê các tỷ số lớn nhất từ cache; giữ qmean thực và ghi ngày/ID để đọc lại mẫu số nhỏ.
ratio_flat = A1_QMAX_QMEAN_RATIO.ravel()
ratio_positions = np.flatnonzero(np.isfinite(ratio_flat))
ratio_largest = ratio_positions[np.argsort(-ratio_flat[ratio_positions], kind="stable")[:15]]
ratio_extreme_rows = []
for flat_position in ratio_largest:
    i, j = np.unravel_index(flat_position, A1_QMAX_QMEAN_RATIO.shape)
    ratio_extreme_rows.append({"basin_id": peak_ids[i], "date": peak_calendar[j],
        "qmin_m3s": A1_QMIN_PARSED_M3S[i, j], "qmean_m3s": A1_QMEAN_PARSED_M3S[i, j],
        "qmax_m3s": peak_qmax[i, j], "qmax_qmean_ratio": A1_QMAX_QMEAN_RATIO[i, j]})
A1_Q_RATIO_EXTREMES = pd.DataFrame(ratio_extreme_rows, columns=[
    "basin_id", "date", "qmin_m3s", "qmean_m3s", "qmax_m3s", "qmax_qmean_ratio"])

fig, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
for ax, row in zip(axes.flat, context_selection.itertuples()):
    context = A1_Q_PEAK_CONTEXT.loc[A1_Q_PEAK_CONTEXT["basin_id"].eq(row.basin_id)]
    for field, color, width in [("qmin_mm_day", "#67a7b6", 1), ("qmean_mm_day", "#23749c", 1.4), ("qmax_mm_day", "#c76a2d", 1.6)]:
        ax.plot(context["date"], context[field], color=color, linewidth=width)
    ax.axvline(row.first_max_date, color="#7d4540", linestyle="--", linewidth=.8)
    ax.scatter([row.first_max_date], [row.max_qmax_mm_day], color="#9c3931", s=16, zorder=4)
    shown = context[["qmin_mm_day", "qmean_mm_day", "qmax_mm_day"]].to_numpy()
    finite = shown[np.isfinite(shown)]
    ax.set_ylim(bottom=min(0, float(finite.min()) * 1.1) if len(finite) else 0)
    ax.set_title(f"ID {row.basin_id}: {str(row.name)[:28]}\nFirst record maximum: {row.first_max_date:%Y-%m-%d}", fontsize=10)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.set_ylabel("Discharge equivalent (mm/day)", fontsize=9)
    ax.grid(alpha=.18)
for ax in list(axes.flat)[len(context_selection):]:
    ax.set_visible(False)
fig.suptitle("Daily Q around large record maxima: qmin, qmean and qmax", fontsize=15, y=.98)
fig.legend(handles=[Line2D([], [], color="#67a7b6", label="qmin"), Line2D([], [], color="#23749c", label="qmean"),
                    Line2D([], [], color="#c76a2d", label="qmax")], loc="lower center", bbox_to_anchor=(.5, .10), ncol=3, frameon=False)
fig.subplots_adjust(left=.065, right=.985, top=.86, bottom=.25, hspace=.60, wspace=.25)
fig.text(.5, .025, "Up to six basins: descending record-max qmax in mm/day, taking the first basin from each distinct calendar year.\n"
         "Windows are ±15 days around the first date of a basin's record maximum; ties are audited separately. Missing Q breaks lines.\n"
         "All three variables use the same area_gov factor. These are diagnostic contexts, not independent floods or verified hourly peaks.", ha="center", fontsize=9)
peak_context_figure = FIGURE_DIR / "A1_P5_q_peak_contexts.png"
fig.savefig(peak_context_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
peak_pairs = A1_Q_PEAK_COMPARISON.copy()
peak_pairs["qmean_mm_day_at_first_max"] = peak_pairs["qmean_m3s_at_first_max"] * (86.4 / peak_pairs["area_gov_km2"])
positive = np.isfinite(peak_pairs["qmean_mm_day_at_first_max"]) & peak_pairs["qmean_mm_day_at_first_max"].gt(0) & peak_pairs["max_qmax_mm_day"].gt(0)
plotted = peak_pairs.loc[positive]
axes[0].scatter(plotted["qmean_mm_day_at_first_max"], plotted["max_qmax_mm_day"], s=12, alpha=.5, color="#23749c", label=f"Positive pairs (n={len(plotted)})")
top_ten = peak_pairs.head(10).loc[positive.reindex(peak_pairs.head(10).index)]
axes[0].scatter(top_ten["qmean_mm_day_at_first_max"], top_ten["max_qmax_mm_day"], s=38, facecolors="none", edgecolors="#c76a2d", label="Top ten record maxima")
if len(plotted):
    low = min(plotted["qmean_mm_day_at_first_max"].min(), plotted["max_qmax_mm_day"].min())
    high = max(plotted["qmean_mm_day_at_first_max"].max(), plotted["max_qmax_mm_day"].max())
    axes[0].plot([low, high], [low, high], "--", color="#777777", linewidth=1, label="Equal values")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
axes[0].set(title="(a) First record-max date: paired Q", xlabel="qmean (mm/day; log axis for positive pairs)", ylabel="qmax (mm/day; log axis for positive pairs)")
axes[0].legend(frameon=False, fontsize=8)
peak_ratios = peak_pairs["ratio_at_first_max"].dropna().to_numpy()
if len(peak_ratios):
    unique, counts = np.unique(peak_ratios, return_counts=True)
    start = unique[0] / 1.02 if unique[0] > 0 else 0
    axes[1].step(np.r_[start, unique], np.r_[0, 100 * np.cumsum(counts) / len(peak_ratios)], where="post", color="#23749c")
    if unique[0] > 0:
        axes[1].set_xscale("log")
        axes[1].set_xlim(start, unique[-1] * 1.03)
    else:
        axes[1].set_xscale("symlog", linthresh=1)
ratio_axis_label = "log axis" if len(peak_ratios) and peak_ratios.min() > 0 else "linear to 1, log above 1"
axes[1].set(title="(b) Ratios at the first record-max date", xlabel=f"qmax / qmean ({ratio_axis_label})", ylabel="Paired basins at or below ratio (%)", ylim=(0, 105))
for ax in axes:
    ax.grid(alpha=.18)
fig.suptitle("Record maxima compared with the same-day mean discharge", fontsize=15, y=.98)
fig.subplots_adjust(left=.08, right=.985, top=.82, bottom=.28, wspace=.28)
fig.text(.5, .025, f"One first-max date per basin; all-record lengths differ. Log scatter omits {len(peak_pairs) - len(plotted)} nonpositive/missing pairs.\n"
         "Ratios require valid qmax ≥0 and qmean >0; zero denominators remain undefined. The common area factor cancels in the ratio.\n"
         "A large ratio or a large maximum is a diagnostic observation, not a rule for deleting or replacing Q.", ha="center", fontsize=9)
peak_pair_figure = FIGURE_DIR / "A1_P5_q_peak_mean_pairs.png"
fig.savefig(peak_pair_figure, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)
for suffix, table in [("q_variable_profile", A1_Q_VARIABLE_PROFILE), ("q_order_by_basin", A1_Q_ORDER_BY_BASIN),
                      ("q_order_flags", A1_Q_ORDER_FLAGS), ("q_ratio_summary", A1_Q_RATIO_SUMMARY),
                      ("q_peak_comparison", A1_Q_PEAK_COMPARISON), ("q_peak_days", A1_Q_PEAK_DAYS),
                      ("q_peak_context", A1_Q_PEAK_CONTEXT), ("q_ratio_extremes", A1_Q_RATIO_EXTREMES)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    print("(1) Kiểm ô Q đọc lại trên toàn lịch; các loại rỗng/âm/sentinel/không hữu hạn tách riêng:")
    display(quality_summary)
    print("(2) Quan hệ cùng ngày trên cặp hữu hạn không phải -999; số âm gắn cờ riêng ở (1)/(5):")
    display(order_summary)
    print("(3) Tỷ số qmax/qmean, giữ NaN khi mẫu số không dương; toàn bộ ngày và một ngày cực đại/ID là hai mẫu khác nhau:")
    display(A1_Q_RATIO_SUMMARY)
    print("(4) Mười cực đại mm/ngày lớn nhất; kiểm mọi ngày đồng cực đại, không chỉ ngày đầu:")
    display(A1_Q_PEAK_COMPARISON.head(10))
    print("(5) Tối đa 20 dòng cần đọc lại; không tự sửa hoặc loại các Q khác cùng dòng:")
    display(A1_Q_ORDER_FLAGS.head(20))
    print("(6) Các ID chọn vẽ cửa sổ; khác năm lịch chỉ để đa dạng hình, chưa xác nhận các sự kiện độc lập:")
    display(context_selection[["basin_id", "name", "first_max_date", "max_qmax_mm_day", "ratio_at_first_max"]])
    print("(7) Tối đa 15 tỷ số qmax/qmean lớn nhất trên mọi ngày; mẫu số dương nhỏ vẫn được ghi nguyên:")
    display(A1_Q_RATIO_EXTREMES)
print("qmax đọc lại khớp cache 5.1; số ngày đồng cực đại lưu ở bảng riêng.")
print("Số dòng ngày–ID có ít nhất một cờ:", len(A1_Q_ORDER_FLAGS))
print("Số top-10 ID có ít nhất một ngày đồng cực đại trong test:", int(A1_Q_PEAK_COMPARISON.head(10)["max_days_in_test"].gt(0).sum()))
print("Đã lưu hai hình:", peak_context_figure, "|", peak_pair_figure)
print("Đã lưu tám CSV vào:", TABLE_DIR)
print(f"Thời gian đối chiếu Q/đỉnh: {time.perf_counter() - started_peak_check:.1f} giây.")

# %% [markdown]
# ### 5.8. Phân phối 40 biến khí tượng, dấu ET, cặp GLEAM và tháng LAI/độ ẩm đất
#
# Quét toàn lịch tệp A của 451 ID; phân vị chính xác từ số float64, không lấy mẫu hoặc đổi dấu/đơn vị.

# %% Phần 5.8 - Phân phối 40 biến khí tượng, dấu ET, cặp GLEAM và tháng LAI/độ ẩm đất
# Quét toàn lịch tệp A của 451 ID; phân vị chính xác từ số float64, không lấy mẫu hoặc đổi dấu/đơn vị.
# Tệp nhị phân tạm được đóng/xóa sau phép đo; hình giữ cả số 0, số âm hữu hạn khác -999 và toàn bộ đuôi.
import tempfile
import contextlib
import shutil

started_forcing_values = time.perf_counter()
meteo_ids = sorted({str(value) for value in selected_ids}, key=int)
meteo_vars = sorted(A1_FORCING_COVERAGE["variable"].unique(),
                    key=lambda field: (SOURCE_PREFIXES.index(field.split("_", 1)[0]), field))
if len(meteo_vars) != 40:
    raise ValueError("Schema không có đúng 40 biến khí tượng; kiểm lại trước khi đo.")
meteo_spans = A1_FORCING_FILE_SPANS.loc[A1_FORCING_FILE_SPANS["basin_id"].astype(str).isin(meteo_ids)]
if set(meteo_spans["basin_id"].astype(str)) != set(meteo_ids):
    raise ValueError("Phạm vi tệp A ở 2.8 không đủ ID đang chọn.")
meteo_calendar = pd.date_range(pd.to_datetime(meteo_spans["file_start"]).min(),
                              pd.to_datetime(meteo_spans["file_end"]).max(), freq="D")
month_keys = meteo_calendar.to_period("M")
month_lengths = pd.Series(month_keys.unique().days_in_month, index=month_keys.unique())
monthly_fields = ["ERA5L_lai_high_veg", "ERA5L_lai_low_veg", "ERA5L_volsw_123", "ERA5L_volsw_4"]
required_fields = set(monthly_fields + ["GLEAM_ETA", "GLEAM_ETP"])
if not required_fields.issubset(meteo_vars):
    raise ValueError("Thiếu cột GLEAM/LAI/độ ẩm đất.")

# Đơn vị tham chiếu, không phải xác nhận chuỗi CSV đã qua đúng phép đổi đơn vị.
unit_ref = {f"ERA5L_{name}_{stat}": "°C" for name in ["2m_temp", "2m_dp_temp"] for stat in ["max", "mean", "min"]}
unit_ref.update({"ERA5L_10m_wind_u": "m/s", "ERA5L_10m_wind_v": "m/s", "ERA5L_fcst_alb": "dimensionless",
    "ERA5L_lai_high_veg": "m²/m²", "ERA5L_lai_low_veg": "m²/m²", "ERA5L_swe": "mm",
    "ERA5L_surf_press": "Pa", "ERA5L_prec": "mm", "ERA5L_total_et": "mm",
    "ERA5L_volsw_123": "m³/m³", "ERA5L_volsw_4": "m³/m³", "ECMWF_t2m": "°C", "ECMWF_d2m": "°C",
    "ECMWF_ssrd": "W/m²", "ECMWF_tp": "mm", "ECMWF_e": "mm", "EOBS_fg": "m/s", "EOBS_pp": "hPa",
    "EOBS_qq": "W/m²", "EOBS_rr": "mm", "EOBS_tg": "°C", "EOBS_tn": "°C", "EOBS_tx": "°C",
    "MSWEP_RR": "mm", "GLEAM_ETA": "mm", "GLEAM_ETP": "mm"})
unit_ref.update({f"ERA5L_surf_net_{kind}_rad_{stat}": "W/m²" for kind in ["solar", "therm"] for stat in ["max", "mean"]})
unit_rows = []
for field in meteo_vars:
    status = "reference_only; CSV conversion not verified" if field in unit_ref else "not listed in BiasCast A1; CSV unit unresolved"
    if field == "ECMWF_ssrd":
        status = "unresolved: BiasCast A1 W/m²; author analysis label J/m²"
    unit_rows.append({"variable": field, "source": field.split("_", 1)[0],
        "reference_unit": unit_ref.get(field, "unresolved"), "unit_status": status,
        "reference": "BiasCast Table A1, pp. 5086–5087" if field in unit_ref else "needs dataset conversion metadata"})
A1_FORCING_UNIT_REFERENCE = pd.DataFrame(unit_rows)

meteo_profile_rows, monthly_rows, const_month_rows, gleam_rows = [], [], [], []
meteo_stats_rows, meteo_bin_rows, meteo_histograms = [], [], {}
temporary_bytes_upper = len(meteo_ids) * len(meteo_calendar) * (len(meteo_vars) + 1) * 8
if shutil.disk_usage(TABLE_DIR).free < temporary_bytes_upper + 200_000_000:
    raise OSError("Không đủ dung lượng cho thống kê chính xác float64; chưa chuyển sang lấy mẫu.")
print(f"Lịch khí tượng: {meteo_calendar[0]:%Y-%m-%d}–{meteo_calendar[-1]:%Y-%m-%d}; "
      f"{len(meteo_ids)} ID, {len(meteo_vars)} biến. Tệp tạm tối đa khoảng {temporary_bytes_upper / 1e9:.2f} GB.", flush=True)
with contextlib.ExitStack() as stack:
    streams = {field: stack.enter_context(tempfile.TemporaryFile(dir=TABLE_DIR, prefix="_a1_p5_forcing_")) for field in meteo_vars}
    gleam_delta_stream = stack.enter_context(tempfile.TemporaryFile(dir=TABLE_DIR, prefix="_a1_p5_gleam_"))
    for i, basin_id in enumerate(meteo_ids):
        raw = pd.read_csv(A1_DAILY_FILES["A"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                          usecols=DATE_COLS + meteo_vars, keep_default_na=False, na_filter=False)
        dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
        if (dates.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing
                or not dates.isin(meteo_calendar).all()):
            raise ValueError(f"ID {basin_id}: lịch khí tượng không hợp lệ/khác 2.8.")
        positions = meteo_calendar.get_indexer(dates)
        parsed_fields = {}
        for field in meteo_vars:
            text = raw[field].str.strip()
            values = pd.to_numeric(text, errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            valid = finite & (values != -999)
            observed = values[valid]
            observed.tofile(streams[field])
            meteo_profile_rows.append({"basin_id": basin_id, "variable": field,
                "expected_days": len(meteo_calendar), "missing_rows": len(meteo_calendar) - len(dates),
                "blank_cells": int(text.eq("").sum()), "minus999_cells": int((values == -999).sum()),
                "non_numeric_nonblank": int((~text.eq("").to_numpy() & np.isnan(values)).sum()),
                "infinite_cells": int(np.isinf(values).sum()), "valid_days": len(observed),
                "negative_days": int((observed < 0).sum()), "zero_days": int((observed == 0).sum()),
                "positive_days": int((observed > 0).sum()),
                "first_value_date": dates[valid].min() if valid.any() else pd.NaT,
                "last_value_date": dates[valid].max() if valid.any() else pd.NaT})
            if field in required_fields:
                aligned = np.full(len(meteo_calendar), np.nan)
                aligned[positions[valid]] = observed
                parsed_fields[field] = aligned
            if field in monthly_fields:
                series = pd.Series(parsed_fields[field], index=meteo_calendar)
                climatology = series.groupby(meteo_calendar.month).agg(["count", "mean"]).reindex(range(1, 13))
                for month, row in climatology.iterrows():
                    monthly_rows.append({"basin_id": basin_id, "variable": field, "month": month,
                        "valid_days": int(row["count"]) if pd.notna(row["count"]) else 0,
                        "mean": row["mean"], "expected_days": int((meteo_calendar.month == month).sum())})
                by_month = series.groupby(month_keys).agg(["count", "nunique"])
                full = by_month["count"].eq(month_lengths.reindex(by_month.index))
                const_month_rows.append({"basin_id": basin_id, "variable": field, "calendar_months": len(by_month),
                    "months_with_any_value": int(by_month["count"].gt(0).sum()), "fully_observed_months": int(full.sum()),
                    "fully_observed_constant_months": int((full & by_month["nunique"].eq(1)).sum())})
        eta, etp = parsed_fields["GLEAM_ETA"], parsed_fields["GLEAM_ETP"]
        paired = np.isfinite(eta) & np.isfinite(etp)
        delta = eta[paired] - etp[paired]
        delta.tofile(gleam_delta_stream)
        max_position = np.flatnonzero(paired)[np.argmax(np.abs(delta))] if paired.any() else None
        gleam_rows.append({"basin_id": basin_id, "paired_days": len(delta), "equal_exact_days": int((delta == 0).sum()),
            "eta_gt_etp_days": int((delta > 0).sum()), "eta_lt_etp_days": int((delta < 0).sum()),
            "within_1e_6_days": int((np.abs(delta) <= 1e-6).sum()),
            "max_abs_delta": float(np.abs(delta).max()) if len(delta) else np.nan,
            "first_max_abs_delta_date": meteo_calendar[max_position] if max_position is not None else pd.NaT,
            "eta_at_max_abs_delta": eta[max_position] if max_position is not None else np.nan,
            "etp_at_max_abs_delta": etp[max_position] if max_position is not None else np.nan})
        if (i + 1) % 50 == 0 or i + 1 == len(meteo_ids):
            print(f"Đã đọc giá trị khí tượng {i + 1}/{len(meteo_ids)} ID", flush=True)
    A1_FORCING_VALUE_PROFILE = pd.DataFrame(meteo_profile_rows)
    categories = ["missing_rows", "blank_cells", "minus999_cells", "non_numeric_nonblank", "infinite_cells", "valid_days"]
    if not A1_FORCING_VALUE_PROFILE[categories].sum(axis=1).eq(A1_FORCING_VALUE_PROFILE["expected_days"]).all():
        raise ValueError("Các loại ô khí tượng không cộng khớp lịch.")
    counts = A1_FORCING_VALUE_PROFILE.groupby("variable")[categories + ["negative_days", "zero_days", "positive_days"]].sum()
    if not counts[["negative_days", "zero_days", "positive_days"]].sum(axis=1).eq(counts["valid_days"]).all():
        raise ValueError("Tổng dấu âm/0/dương không khớp.")
    coverage_reference = A1_FORCING_COVERAGE.copy()
    coverage_reference["basin_id"] = coverage_reference["basin_id"].astype(str)
    coverage_reference = coverage_reference.set_index(["basin_id", "variable"])["valid_days"]
    measured_counts = A1_FORCING_VALUE_PROFILE.set_index(["basin_id", "variable"])["valid_days"]
    if not measured_counts.eq(coverage_reference.reindex(measured_counts.index)).all():
        raise ValueError("Số ngày hợp lệ từng ID/biến không khớp phép quét toàn chuỗi ở 2.8.")
    log_fields = {"ERA5L_swe": .1, "ERA5L_prec": .1, "ECMWF_tp": .1, "EOBS_rr": .1, "MSWEP_RR": .1,
                  "ERA5L_total_et": .01, "ECMWF_e": .01, "GLEAM_ETA": .01, "GLEAM_ETP": .01}
    for field in meteo_vars:
        streams[field].flush()
        streams[field].seek(0)
        values = np.fromfile(streams[field], dtype=np.float64)
        if len(values) != counts.loc[field, "valid_days"]:
            raise ValueError(f"{field}: số quan trắc trong tệp tạm không khớp.")
        quantiles = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1]) if len(values) else np.full(9, np.nan)
        meteo_stats_rows.append({"variable": field, **counts.loc[field].astype(int).to_dict(),
            **dict(zip(["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"], quantiles)),
            "mean": float(values.mean()) if len(values) else np.nan,
            "std_ddof0": float(values.std(ddof=0)) if len(values) else np.nan})
        if len(values):
            low, high = quantiles[[0, -1]]
            if low == high:
                radius = max(abs(low) * .05, .5)
                edges = np.array([low - radius, high + radius])
            elif field in log_fields:
                from matplotlib.scale import SymmetricalLogTransform
                transform = SymmetricalLogTransform(base=10, linthresh=log_fields[field], linscale=1)
                edges = transform.inverted().transform(np.linspace(*transform.transform([low, high]), 41))
                edges[0], edges[-1] = low, high
            else:
                edges = np.linspace(low, high, 41)
            hist, edges = np.histogram(values, bins=edges)
            if hist.sum() != len(values):
                raise ValueError(f"{field}: histogram không bao trọn đuôi/0/số âm.")
            meteo_histograms[field] = (edges, 100 * hist / len(values))
            for left, right, number in zip(edges[:-1], edges[1:], hist):
                meteo_bin_rows.append({"variable": field, "left_edge": left, "right_edge": right,
                    "count": int(number), "pct_of_valid": 100 * number / len(values),
                    "rightmost_bin": right == edges[-1], "symlog_linthresh": log_fields.get(field, np.nan)})
        print(f"Đã tính phân vị: {field}", flush=True)
    gleam_delta_stream.flush()
    gleam_delta_stream.seek(0)
    delta = np.fromfile(gleam_delta_stream, dtype=np.float64)
    A1_GLEAM_PAIR_BY_BASIN = pd.DataFrame(gleam_rows)
    gleam_totals = A1_GLEAM_PAIR_BY_BASIN[["paired_days", "equal_exact_days", "eta_gt_etp_days", "eta_lt_etp_days", "within_1e_6_days"]].sum()
    if (gleam_totals["paired_days"] != len(delta)
            or gleam_totals[["equal_exact_days", "eta_gt_etp_days", "eta_lt_etp_days"]].sum() != len(delta)):
        raise ValueError("Số cặp GLEAM không cộng khớp.")
    A1_GLEAM_PAIR_SUMMARY = pd.DataFrame([{"basins": len(meteo_ids), **gleam_totals.astype(int).to_dict(),
        "equal_exact_pct": 100 * gleam_totals["equal_exact_days"] / len(delta) if len(delta) else np.nan,
        "delta_min": float(delta.min()) if len(delta) else np.nan,
        "delta_median": float(np.median(delta)) if len(delta) else np.nan,
        "delta_max": float(delta.max()) if len(delta) else np.nan,
        "max_abs_delta": float(np.abs(delta).max()) if len(delta) else np.nan}])
# Streams đã đóng; các tệp tạm không trở thành dữ liệu đầu vào hoặc cache mô hình.
A1_FORCING_DISTRIBUTIONS = A1_FORCING_UNIT_REFERENCE.merge(pd.DataFrame(meteo_stats_rows), on="variable", validate="one_to_one")
boundaries = A1_FORCING_VALUE_PROFILE.groupby("variable").agg(first_value_date=("first_value_date", "min"), last_value_date=("last_value_date", "max"))
A1_FORCING_DISTRIBUTIONS = A1_FORCING_DISTRIBUTIONS.join(boundaries, on="variable")
for sign in ["negative", "zero", "positive"]:
    A1_FORCING_DISTRIBUTIONS[f"{sign}_pct"] = 100 * A1_FORCING_DISTRIBUTIONS[f"{sign}_days"] / A1_FORCING_DISTRIBUTIONS["valid_days"].replace(0, np.nan)
A1_FORCING_HISTOGRAM_BINS = pd.DataFrame(meteo_bin_rows)
A1_FORCING_MONTHLY_BY_BASIN = pd.DataFrame(monthly_rows)
monthly_valid = A1_FORCING_MONTHLY_BY_BASIN.groupby(["basin_id", "variable"])["valid_days"].sum()
monthly_reference = A1_FORCING_VALUE_PROFILE.set_index(["basin_id", "variable"])["valid_days"].reindex(monthly_valid.index)
if not monthly_valid.eq(monthly_reference).all():
    raise ValueError("Tổng ngày trong các tháng chưa khớp hồ sơ từng ID/biến.")
A1_FORCING_MONTHLY_SUMMARY = A1_FORCING_MONTHLY_BY_BASIN.groupby(["variable", "month"]).agg(
    basins_with_values=("mean", "count"), median=("mean", "median"),
    p25=("mean", lambda values: values.quantile(.25)), p75=("mean", lambda values: values.quantile(.75)),
    valid_days=("valid_days", "sum"), expected_days=("expected_days", "sum")).reset_index()
A1_FORCING_CONSTANT_MONTHS = pd.DataFrame(const_month_rows)
constant_month_summary = A1_FORCING_CONSTANT_MONTHS.groupby("variable").sum(numeric_only=True)
constant_month_summary["constant_pct_of_full_months"] = 100 * constant_month_summary["fully_observed_constant_months"] / constant_month_summary["fully_observed_months"].replace(0, np.nan)
meteo_figure_paths = []
for source in SOURCE_PREFIXES:
    fields = [field for field in meteo_vars if field.startswith(source + "_")]
    columns = min(3, len(fields))
    rows = (len(fields) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.4 * rows + 1.8), squeeze=False)
    for ax, field in zip(axes.flat, fields):
        stats = A1_FORCING_DISTRIBUTIONS.set_index("variable").loc[field]
        if field in meteo_histograms:
            edges, pct = meteo_histograms[field]
            ax.stairs(pct, edges, color="#237f9f", linewidth=1.25)
        if field in log_fields:
            ax.set_xscale("symlog", linthresh=log_fields[field])
        unit = "raw; unit unresolved" if field not in unit_ref or field == "ECMWF_ssrd" else f"{unit_ref[field]} (reference)"
        count_label = (f"n={int(stats['valid_days']):,}; negative={stats['negative_pct']:.2f}%; zero={stats['zero_pct']:.2f}%"
                       if stats["valid_days"] else "n=0; no valid values")
        ax.set(title=f"{field}\n{count_label}",
               xlabel=unit + (f"; symlog L={log_fields[field]:g}" if field in log_fields else ""), ylabel="Valid basin-days per bin (%)")
        ax.title.set_fontsize(9)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.16)
    for ax in list(axes.flat)[len(fields):]:
        ax.set_visible(False)
    fig.suptitle(f"{source}: stored daily meteorological values, {len(meteo_ids)} basins, {meteo_calendar[0].year}–{meteo_calendar[-1].year}", fontsize=14, y=.985)
    fig.tight_layout(rect=(0, .10 if rows > 1 else .23, 1, .96 if rows > 1 else .91), h_pad=2)
    fig.text(.5, .012, "Each variable uses its own available days and units; valid means finite and different from -999.\n"
             "Full observed ranges, zeros and signed values are retained. Symlog changes display only; bin edges/counts are saved.\n"
             "Unit labels are documented references, not verified CSV conversions. Binning of discrete values may create apparent modes.", ha="center", fontsize=8)
    path = FIGURE_DIR / f"A1_P5_forcing_distributions_{source}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    meteo_figure_paths.append(path)
    plt.show()
    plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, field in zip(axes.flat, monthly_fields):
    monthly = A1_FORCING_MONTHLY_SUMMARY.loc[A1_FORCING_MONTHLY_SUMMARY["variable"].eq(field)]
    ax.plot(monthly["month"], monthly["median"], color="#237f9f", marker="o", markersize=3)
    ax.fill_between(monthly["month"], monthly["p25"], monthly["p75"], color="#82bfd2", alpha=.35)
    ax.set(title=field, xlabel="Calendar month", ylabel=f"Basin monthly mean ({unit_ref[field]})", xticks=range(1, 13))
    ax.grid(alpha=.18)
fig.suptitle("LAI and soil water: calendar-month means across basins", fontsize=14, y=.98)
fig.subplots_adjust(top=.86, bottom=.22, left=.09, right=.98, hspace=.42, wspace=.30)
fig.text(.5, .025, f"Window: {meteo_calendar[0]:%Y-%m-%d} to {meteo_calendar[-1]:%Y-%m-%d}; each basin contributes one mean per calendar month.\n"
         "Line: median across available basin means; band: 25th–75th percentiles across basins. Missing values remain missing.\n"
         "Exact within-year-month constancy is counted separately; a seasonal curve does not establish the native time resolution.", ha="center", fontsize=9)
path = FIGURE_DIR / "A1_P5_forcing_monthly_lai_soil.png"
fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
meteo_figure_paths.append(path)
plt.show()
plt.close(fig)
for suffix, table in [("forcing_unit_reference", A1_FORCING_UNIT_REFERENCE), ("forcing_distributions", A1_FORCING_DISTRIBUTIONS),
    ("forcing_value_profile", A1_FORCING_VALUE_PROFILE), ("forcing_histogram_bins", A1_FORCING_HISTOGRAM_BINS),
    ("GLEAM_pair_by_basin", A1_GLEAM_PAIR_BY_BASIN), ("GLEAM_pair_summary", A1_GLEAM_PAIR_SUMMARY),
    ("forcing_monthly_by_basin", A1_FORCING_MONTHLY_BY_BASIN), ("forcing_monthly_summary", A1_FORCING_MONTHLY_SUMMARY),
    ("forcing_constant_months", A1_FORCING_CONSTANT_MONTHS)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    print("(1) Đơn vị tham chiếu và phạm vi có giá trị; không tự gán đơn vị gốc ECMWF cho CSV:")
    display(A1_FORCING_DISTRIBUTIONS[["variable", "reference_unit", "unit_status", "first_value_date", "last_value_date", "valid_days"]])
    print("(2) Phân phối gộp mọi ngày hợp lệ; số 0 và số âm hữu hạn khác -999 được giữ:")
    display(A1_FORCING_DISTRIBUTIONS[["variable", "valid_days", "negative_pct", "zero_pct", "min", "p01", "p05", "median", "p95", "p99", "max", "mean", "std_ddof0"]])
    print("(3) GLEAM_ETA − GLEAM_ETP trên cặp cùng ID/ngày; bằng nhau chính xác khác dung sai 1e-6:")
    display(A1_GLEAM_PAIR_SUMMARY)
    display(A1_GLEAM_PAIR_BY_BASIN.sort_values("max_abs_delta", ascending=False).head(10))
    print("(4) Trung vị ngang các mean tháng của lưu vực; không dùng số 0 thay tháng thiếu:")
    display(A1_FORCING_MONTHLY_SUMMARY.pivot(index="month", columns="variable", values="median"))
    print("(5) Tháng năm có đủ mọi ngày và chỉ một giá trị lưu trữ; chưa xác định độ phân giải nguồn:")
    display(constant_month_summary)
    print("(6) Kiểm kê thiếu toàn lịch khí tượng; khác phạm vi 2003–2017 của 4.1:")
    display(A1_FORCING_DISTRIBUTIONS[["variable"] + categories])
print("CẢNH BÁO: ECMWF_ssrd chưa thống nhất W/m² ở Bảng A1 với J/m² trong nhãn mã phân tích tác giả; chưa đổi đơn vị.")
print("Các cột chưa có đơn vị trong Bảng A1:", [field for field in meteo_vars if field not in unit_ref])
print("Đã lưu sáu hình:", " | ".join(map(str, meteo_figure_paths)))
print("Đã lưu chín CSV vào:", TABLE_DIR, "; tệp nhị phân tạm đã đóng/xóa.")
print(f"Thời gian phân tích khí tượng: {time.perf_counter() - started_forcing_values:.1f} giây.")

# %% [markdown]
# ### 5.9. Định vị thiếu E-OBS, kiểm thang mưa ECMWF, cực trị SWE và nguồn Q giờ
#
# Đọc sáu cột A để giải các cờ từ 5.8; thử hệ số chỉ trong phép so, không đổi CSV hoặc khẳng định đơn vị.

# %% Phần 5.9 - Định vị thiếu E-OBS, kiểm thang mưa ECMWF, cực trị SWE và nguồn Q giờ
# Đọc sáu cột A để giải các cờ từ 5.8; thử hệ số chỉ trong phép so, không đổi CSV hoặc khẳng định đơn vị.
# Kiểm header/mẫu nguồn giờ nếu Input có sẵn; chưa tổng hợp Q giờ khi quy ước ngày chưa được đối chiếu.
started_forcing_followup = time.perf_counter()
follow_ids = sorted({str(value) for value in selected_ids}, key=int)
follow_calendar = pd.DatetimeIndex(meteo_calendar)
follow_fields = ["EOBS_fg", "EOBS_hu", "ECMWF_tp", "ECMWF_lsp", "ECMWF_cp", "ERA5L_swe"]
train_start = pd.Timestamp(PERIODS["train"][0])
warmup_start = train_start - pd.Timedelta(days=364)
period_masks = {period: (follow_calendar >= pd.Timestamp(start)) & (follow_calendar <= pd.Timestamp(end))
                for period, (start, end) in PERIODS.items()}
assigned = np.sum(list(period_masks.values()), axis=0)
if (assigned > 1).any():
    raise ValueError("Các kỳ chồng nhau; chưa gán nhãn ngày thiếu.")
period_labels = np.full(len(follow_calendar), "outside_periods", dtype=object)
period_labels[follow_calendar < min(pd.Timestamp(start) for start, end in PERIODS.values())] = "before_periods"
period_labels[follow_calendar > max(pd.Timestamp(end) for start, end in PERIODS.values())] = "after_periods"
for period, mask in period_masks.items():
    period_labels[mask] = period
warmup_mask = (follow_calendar >= warmup_start) & (follow_calendar < train_start)
value_reference = A1_FORCING_VALUE_PROFILE.copy()
value_reference["basin_id"] = value_reference["basin_id"].astype(str)
value_reference = value_reference.set_index(["basin_id", "variable"])
missing_rows_follow, swe_rows, cp_chunks, residual_chunks, rain_example_rows = [], [], [], [], []
grid_counts = {field: {"valid_days": 0, "on_0p01_grid": 0} for field in ["ECMWF_tp", "ECMWF_lsp", "ECMWF_cp"]}
rain_total_pairs, rain_negative_residual, cp_zero_positive_residual = 0, 0, 0
for file_no, basin_id in enumerate(follow_ids, start=1):
    raw = pd.read_csv(A1_DAILY_FILES["A"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                      usecols=DATE_COLS + follow_fields, keep_default_na=False, na_filter=False)
    dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
    if (dates.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing
            or not dates.isin(follow_calendar).all()):
        raise ValueError(f"ID {basin_id}: lịch không khớp phép quét 5.8.")
    positions = follow_calendar.get_indexer(dates)
    present = follow_calendar.isin(dates)
    parsed = {}
    for field in follow_fields:
        tokens = raw[field].str.strip()
        values = pd.to_numeric(tokens, errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(values) & (values != -999)
        aligned = np.full(len(follow_calendar), np.nan)
        aligned[positions[valid]] = values[valid]
        parsed[field] = aligned
        if int(valid.sum()) != value_reference.loc[(basin_id, field), "valid_days"]:
            raise ValueError(f"{basin_id}/{field}: số ngày hợp lệ khác 5.8.")
        if field.startswith("EOBS_"):
            reasons = np.full(len(follow_calendar), "missing_row", dtype=object)
            raw_tokens = np.full(len(follow_calendar), None, dtype=object)
            raw_tokens[positions] = tokens.to_numpy()
            source_reasons = np.where(tokens.eq(""), "blank", np.where(values == -999, "minus999",
                               np.where(np.isinf(values), "infinite", np.where(np.isnan(values), "non_numeric", "valid"))))
            reasons[positions] = source_reasons
            for j in np.flatnonzero(~np.isfinite(aligned)):
                missing_rows_follow.append({"basin_id": basin_id, "variable": field, "date": follow_calendar[j],
                    "period": period_labels[j], "in_364d_before_train": bool(warmup_mask[j]),
                    "reason": reasons[j], "raw_token": raw_tokens[j]})
        if field in grid_counts:
            grid_counts[field]["valid_days"] += len(values[valid])
            grid_counts[field]["on_0p01_grid"] += int(np.isclose(values[valid] * 100, np.rint(values[valid] * 100), atol=1e-8, rtol=0).sum())
    swe = parsed["ERA5L_swe"]
    finite_swe = np.flatnonzero(np.isfinite(swe))
    if len(finite_swe):
        first_max = finite_swe[np.argmax(swe[finite_swe])]
        swe_rows.append({"basin_id": basin_id, "valid_days": len(finite_swe), "max_stored_swe": swe[first_max],
            "first_date_of_max": follow_calendar[first_max], "days_at_max": int((swe == swe[first_max]).sum()),
            "days_ge_1000_stored": int((swe >= 1000).sum()), "reference_unit": "mm; CSV conversion not independently verified"})
    tp, lsp, cp = (parsed[field] for field in ["ECMWF_tp", "ECMWF_lsp", "ECMWF_cp"])
    paired = np.isfinite(tp) & np.isfinite(lsp) & np.isfinite(cp)
    residual = tp[paired] - lsp[paired]
    cp_values = cp[paired]
    cp_chunks.append(cp_values)
    residual_chunks.append(residual)
    rain_total_pairs += len(residual)
    rain_negative_residual += int((residual < -0.02 - 1e-9).sum())
    cp_zero_positive_residual += int(((cp_values == 0) & (residual > .02 + 1e-9)).sum())
    # Ví dụ có phần tp-lsp lớn nhất trong từng ID; không gọi đó là sai số hay nội suy.
    if paired.any():
        index = np.flatnonzero(paired)[np.argmax(residual)]
        rain_example_rows.append({"basin_id": basin_id, "date": follow_calendar[index],
            "tp_stored": tp[index], "lsp_stored": lsp[index], "cp_stored": cp[index],
            "tp_minus_lsp": tp[index] - lsp[index], "residual_cp_x1": tp[index] - lsp[index] - cp[index],
            "residual_cp_x1000": tp[index] - lsp[index] - 1000 * cp[index]})
    if file_no % 50 == 0 or file_no == len(follow_ids):
        print(f"Đã kiểm sáu cột khí tượng {file_no}/{len(follow_ids)} ID", flush=True)
A1_FORCING_FOLLOWUP_MISSING = pd.DataFrame(missing_rows_follow, columns=[
    "basin_id", "variable", "date", "period", "in_364d_before_train", "reason", "raw_token"])
for field in ["EOBS_fg", "EOBS_hu"]:
    measured = int(A1_FORCING_FOLLOWUP_MISSING["variable"].eq(field).sum())
    reference = int(value_reference.loc[(follow_ids, field), "expected_days"].sum() - value_reference.loc[(follow_ids, field), "valid_days"].sum())
    if measured != reference:
        raise ValueError(f"{field}: tổng thiếu toàn lịch không khớp 5.8.")
missing_period_rows, warmup_rows = [], []
for field in ["EOBS_fg", "EOBS_hu"]:
    part = A1_FORCING_FOLLOWUP_MISSING.loc[A1_FORCING_FOLLOWUP_MISSING["variable"].eq(field)]
    for period in ["before_periods"] + list(PERIODS) + ["after_periods", "outside_periods"]:
        segment = part.loc[part["period"].eq(period)]
        expected = int((period_labels == period).sum()) * len(follow_ids)
        missing_period_rows.append({"variable": field, "period": period, "expected_cells": expected,
            "missing_cells": len(segment), "affected_basins": segment["basin_id"].nunique(),
            "first_missing": segment["date"].min(), "last_missing": segment["date"].max()})
    segment = part.loc[part["in_364d_before_train"].eq(True)]
    warmup_rows.append({"variable": field, "window_start": warmup_start, "window_end": train_start - pd.Timedelta(days=1),
        "expected_cells": int(warmup_mask.sum()) * len(follow_ids), "missing_cells": len(segment),
        "affected_basins": segment["basin_id"].nunique(), "first_missing": segment["date"].min(), "last_missing": segment["date"].max()})
A1_FORCING_FOLLOWUP_PERIODS = pd.DataFrame(missing_period_rows)
A1_FORCING_FOLLOWUP_WARMUP = pd.DataFrame(warmup_rows)
# Kiểm cùng tập ngày–ID với 4.1, không chỉ so tổng số.
follow_p4_rows = []
for field in ["EOBS_fg", "EOBS_hu"]:
    for period in PERIODS:
        current = A1_FORCING_FOLLOWUP_MISSING.loc[A1_FORCING_FOLLOWUP_MISSING["variable"].eq(field) & A1_FORCING_FOLLOWUP_MISSING["period"].eq(period)]
        reference = A1_FORCING_MISSING_CELLS.loc[A1_FORCING_MISSING_CELLS["variable"].eq(field) & A1_FORCING_MISSING_CELLS["period"].eq(period)]
        current_keys = set(zip(current["basin_id"].astype(str), pd.to_datetime(current["date"])))
        reference_keys = set(zip(reference["basin_id"].astype(str), pd.to_datetime(reference["date"])))
        follow_p4_rows.append({"variable": field, "period": period, "current_missing": len(current_keys),
            "reference_missing_4_1": len(reference_keys), "same_id_date_set": current_keys == reference_keys})
A1_FORCING_FOLLOWUP_P4_CHECK = pd.DataFrame(follow_p4_rows)
if not A1_FORCING_FOLLOWUP_P4_CHECK["same_id_date_set"].all():
    raise ValueError("Thiếu E-OBS ở các kỳ không khớp 4.1 theo ID/ngày.")
cp_values = np.concatenate(cp_chunks)
residual = np.concatenate(residual_chunks)
del cp_chunks, residual_chunks
A1_ECMWF_RAIN_GRID = pd.DataFrame([{"variable": field, **stats,
    "all_values_on_0p01_grid": stats["valid_days"] > 0 and stats["valid_days"] == stats["on_0p01_grid"]}
    for field, stats in grid_counts.items()])
all_on_grid = bool(A1_ECMWF_RAIN_GRID["all_values_on_0p01_grid"].all())
rain_candidates = []
for factor in [1, 1000]:
    difference = residual - factor * cp_values
    quantiles = np.quantile(difference, [0, .05, .5, .95, 1]) if len(difference) else [np.nan] * 5
    # Chỉ là khoảng tương thích GIẢ ĐỊNH cả ba trường đều làm tròn gần nhất đến 0,01.
    allowance = .005 + .005 + factor * .005
    rain_candidates.append({"cp_multiplier": factor, "triplets": len(difference),
        **dict(zip(["residual_min", "residual_p05", "residual_median", "residual_p95", "residual_max"], quantiles)),
        "residual_mean": float(difference.mean()) if len(difference) else np.nan,
        "max_abs_residual": float(np.abs(difference).max()) if len(difference) else np.nan,
        "within_same_0p02": int((np.abs(difference) <= .02 + 1e-9).sum()), "same_tolerance": .02,
        "assumed_rounding_allowance": allowance,
        "within_assumed_rounding": int((np.abs(difference) <= allowance + 1e-9).sum()) if all_on_grid else np.nan,
        "all_three_on_0p01_grid": all_on_grid})
A1_ECMWF_RAIN_CANDIDATES = pd.DataFrame(rain_candidates)
A1_ECMWF_RAIN_EXAMPLES = pd.DataFrame(rain_example_rows, columns=[
    "basin_id", "date", "tp_stored", "lsp_stored", "cp_stored", "tp_minus_lsp", "residual_cp_x1", "residual_cp_x1000"])
A1_SWE_RECORD_MAXIMA = pd.DataFrame(swe_rows).sort_values("max_stored_swe", ascending=False, kind="stable")
reference_swe_max = A1_FORCING_DISTRIBUTIONS.set_index("variable").loc["ERA5L_swe", "max"]
if len(A1_SWE_RECORD_MAXIMA) and A1_SWE_RECORD_MAXIMA["max_stored_swe"].max() != reference_swe_max:
    raise ValueError("Cực đại SWE đọc lại không khớp 5.8.")
# Chỉ kiểm kê/đọc ba dòng mẫu của nguồn Q giờ; không giả định cột Q hoặc múi giờ.
# Bao gồm nguồn 2.5 chuẩn bị trong working, không chỉ cây Input ban đầu.
hourly_dirs = sorted(set(globals().get("A1_HOURLY_DIRS", [])) | {
    path.resolve() for root in {SEARCH_ROOT.resolve(), WORK_DIR.resolve()} if root.is_dir()
    for path in root.rglob("hourly")
    if path.is_dir() and path.parent.name == "2_timeseries" and path.parent.parent.name == "D_gauges"})
hourly_rows, hourly_previews = [], []
for directory in hourly_dirs:
    files = {str(int(match.group(1))): path for path in sorted(directory.glob("*.csv"))
             if (match := ID_PATTERN.fullmatch(path.name))}
    candidates = [basin for basin in ["10", "55", "742", "758", "800"] if basin in files]
    preview_id = candidates[0] if candidates else (sorted(files, key=int)[0] if files else None)
    hourly_rows.append({"directory": str(directory), "status": "csv_found" if files else "directory_without_ID_csv",
        "csv_files": len(files), "selected_ids_with_file": len(set(follow_ids) & set(files)), "preview_id": preview_id})
    if preview_id is not None:
        preview = pd.read_csv(files[preview_id], sep=";", dtype=str, encoding="utf-8-sig", nrows=3, keep_default_na=False)
        for column in preview.columns:
            hourly_previews.append({"directory": str(directory), "basin_id": preview_id, "column": column,
                "first_three_raw_values": " | ".join(preview[column].tolist())})
if not hourly_rows:
    hourly_rows.append({"directory": None, "status": "not_found_under_SEARCH_ROOT", "csv_files": 0,
                        "selected_ids_with_file": 0, "preview_id": None})
A1_HOURLY_FOLLOWUP_INVENTORY = pd.DataFrame(hourly_rows)
A1_HOURLY_FOLLOWUP_PREVIEW = pd.DataFrame(hourly_previews, columns=["directory", "basin_id", "column", "first_three_raw_values"])
for suffix, table in [("forcing_followup_eobs_missing", A1_FORCING_FOLLOWUP_MISSING), ("forcing_followup_eobs_periods", A1_FORCING_FOLLOWUP_PERIODS),
    ("forcing_followup_eobs_warmup", A1_FORCING_FOLLOWUP_WARMUP), ("forcing_followup_p4_check", A1_FORCING_FOLLOWUP_P4_CHECK),
    ("ECMWF_rain_grid", A1_ECMWF_RAIN_GRID), ("ECMWF_rain_candidates", A1_ECMWF_RAIN_CANDIDATES),
    ("ECMWF_rain_examples", A1_ECMWF_RAIN_EXAMPLES), ("swe_record_maxima", A1_SWE_RECORD_MAXIMA),
    ("hourly_followup_inventory", A1_HOURLY_FOLLOWUP_INVENTORY), ("hourly_followup_preview", A1_HOURLY_FOLLOWUP_PREVIEW)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    print("(1) Thiếu E-OBS toàn lịch, tách ngoài các kỳ; số đếm không phải số mẫu mất:")
    display(A1_FORCING_FOLLOWUP_PERIODS)
    print("(2) Cận quét 364 ngày trước đầu train; thuộc nhóm before_periods ở (1), không cộng lặp:")
    display(A1_FORCING_FOLLOWUP_WARMUP)
    print("(3) Tối đa 50 ngày EOBS_fg thiếu; toàn bộ E-OBS thiếu lưu CSV:")
    display(A1_FORCING_FOLLOWUP_MISSING.loc[A1_FORCING_FOLLOWUP_MISSING["variable"].eq("EOBS_fg")].head(50))
    print("(4) So tp − lsp − k×cp trên cùng bộ ba; k là giả thuyết thang đo, không sửa dữ liệu:")
    display(A1_ECMWF_RAIN_GRID)
    display(A1_ECMWF_RAIN_CANDIDATES)
    display(A1_ECMWF_RAIN_EXAMPLES.sort_values("tp_minus_lsp", ascending=False, kind="stable").head(10))
    print("(5) Mười ID có max SWE lớn nhất; ngưỡng 1000 là cờ mô tả theo giá trị lưu trữ:")
    display(A1_SWE_RECORD_MAXIMA.head(10))
    print("(6) Nguồn Q giờ hiện có và header/ba dòng mẫu, chưa tổng hợp:")
    display(A1_HOURLY_FOLLOWUP_INVENTORY)
    display(A1_HOURLY_FOLLOWUP_PREVIEW)
    print("(7) Đối chiếu đúng tập ngày–ID E-OBS thiếu với 4.1:")
    display(A1_FORCING_FOLLOWUP_P4_CHECK)
print("Số bộ ba mưa:", rain_total_pairs, "; tp-lsp < -0,02:", rain_negative_residual,
      "; cp=0 nhưng tp-lsp >0,02:", cp_zero_positive_residual)
print("CẢNH BÁO: khoảng làm tròn 0,01 là giả định; dung sai khác nhau giữa k=1/1000 không dùng để chọn đơn vị.")
print("CẢNH BÁO: bảng giờ chỉ xác nhận khả năng đọc file; chưa xác nhận cột Q, ngày/múi giờ, nội suy hoặc đỉnh.")
if not hourly_dirs:
    print("Chưa thấy Q giờ trong Input/working; chạy cell 2.5 để chuẩn bị nguồn trước khi đối chiếu giờ.")
print("Đã lưu mười CSV vào:", TABLE_DIR)
print(f"Thời gian kiểm các cờ khí tượng/nguồn giờ: {time.perf_counter() - started_forcing_followup:.1f} giây.")

# %% [markdown]
# ### 5.10. Kiểm mức SWE thấp nhất và diễn biến theo năm
#
# Đọc lại riêng SWE; ngưỡng mô tả không phải quy tắc sửa/lọc số hoặc xác nhận đơn vị CSV.

# %% Phần 5.10 - Kiểm mức SWE thấp nhất và diễn biến theo năm
# Đọc lại riêng SWE; ngưỡng mô tả không phải quy tắc sửa/lọc số hoặc xác nhận đơn vị CSV.
started_swe_audit = time.perf_counter()
swe_calendar = pd.DatetimeIndex(meteo_calendar)
swe_ids = sorted({str(value) for value in selected_ids}, key=int)
swe_reference = A1_SWE_RECORD_MAXIMA.copy()
swe_reference["basin_id"] = swe_reference["basin_id"].astype(str)
if swe_reference["basin_id"].duplicated().any():
    raise ValueError("Bảng SWE 5.9 trùng ID.")
swe_reference = swe_reference.set_index("basin_id")
swe_stats, swe_years, swe_checks = [], [], []
for file_no, basin_id in enumerate(swe_ids, start=1):
    raw = pd.read_csv(A1_DAILY_FILES["A"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                      usecols=DATE_COLS + ["ERA5L_swe"], keep_default_na=False, na_filter=False)
    dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
    if (dates.empty or dates.isna().any() or dates.duplicated().any()
            or not dates.is_monotonic_increasing or not dates.isin(swe_calendar).all()):
        raise ValueError(f"ID {basin_id}: lịch SWE không hợp lệ.")
    source = pd.to_numeric(raw["ERA5L_swe"].str.strip(), errors="coerce").to_numpy(dtype=np.float64)
    usable = np.isfinite(source) & (source != -999)
    values = np.full(len(swe_calendar), np.nan)
    values[swe_calendar.get_indexer(dates[usable])] = source[usable]
    valid = np.isfinite(values)
    observed = values[valid]
    n = len(observed)
    # Đối chiếu thêm min>=1000, độc lập với phép cộng số ngày vượt ở 5.9.
    ge_count = int(np.count_nonzero(observed >= 1000))
    quantiles = np.quantile(observed, [0, .01, .05, .5, .95, .99, 1]) if n else np.full(7, np.nan)
    minimum, maximum = quantiles[0], quantiles[-1]
    category = ("no_valid_value" if not n else "all_ge_1000" if minimum >= 1000
                else "some_ge_1000" if maximum >= 1000 else "none_ge_1000")
    row = {"basin_id": basin_id, "expected_days": len(swe_calendar), "valid_days": n,
           "missing_days": int((~valid).sum()), "negative_days": int((observed < 0).sum()),
           "zero_days": int((observed == 0).sum()), "days_lt_1000_stored": n - ge_count,
           "days_ge_1000_stored": ge_count, "threshold_class": category,
           **dict(zip(["min", "p01", "p05", "median", "p95", "p99", "max"], quantiles))}
    row.update({"first_valid_date": swe_calendar[valid][0] if n else pd.NaT,
                "first_valid_value": observed[0] if n else np.nan,
                "last_valid_date": swe_calendar[valid][-1] if n else pd.NaT,
                "last_valid_value": observed[-1] if n else np.nan,
                "first_date_of_min": swe_calendar[np.flatnonzero(valid)[np.argmin(observed)]] if n else pd.NaT,
                "first_date_of_max": swe_calendar[np.flatnonzero(valid)[np.argmax(observed)]] if n else pd.NaT})
    swe_stats.append(row)
    reference_present = basin_id in swe_reference.index
    old = swe_reference.loc[basin_id] if reference_present else None
    check = {"basin_id": basin_id, "valid_days_match_5_9": n == int(old["valid_days"]) if reference_present else n == 0,
             "max_match_5_9": maximum == old["max_stored_swe"] if reference_present else n == 0,
             "threshold_count_match_5_9": ge_count == int(old["days_ge_1000_stored"]) if reference_present else n == 0,
             "min_and_threshold_count_agree": bool((minimum >= 1000) == (ge_count == n)) if n else True}
    swe_checks.append(check)
    if not all(value for key, value in check.items() if key != "basin_id"):
        raise ValueError(f"ID {basin_id}: SWE đọc lại không khớp 5.9 hoặc min/mốc đếm mâu thuẫn.")
    for year in np.unique(swe_calendar.year):
        annual = values[swe_calendar.year == year]
        finite = annual[np.isfinite(annual)]
        swe_years.append({"basin_id": basin_id, "year": int(year), "expected_days": len(annual),
                          "valid_days": len(finite), "days_ge_1000_stored": int((finite >= 1000).sum()),
                          "min": finite.min() if len(finite) else np.nan,
                          "median": np.median(finite) if len(finite) else np.nan,
                          "max": finite.max() if len(finite) else np.nan})
    if file_no % 50 == 0 or file_no == len(swe_ids):
        print(f"Đã kiểm mức thấp nhất SWE {file_no}/{len(swe_ids)} ID", flush=True)
A1_SWE_VALUE_AUDIT = pd.DataFrame(swe_stats)
A1_SWE_ANNUAL_AUDIT = pd.DataFrame(swe_years)
A1_SWE_RECHECK = pd.DataFrame(swe_checks)
if A1_SWE_ANNUAL_AUDIT["valid_days"].sum() != A1_SWE_VALUE_AUDIT["valid_days"].sum():
    raise ValueError("Số ngày SWE theo năm không cộng khớp.")
A1_SWE_THRESHOLD_GROUPS = A1_SWE_VALUE_AUDIT.groupby("threshold_class", sort=False).agg(
    basins=("basin_id", "size"), expected_days=("expected_days", "sum"), valid_days=("valid_days", "sum"),
    missing_days=("missing_days", "sum"), days_lt_1000_stored=("days_lt_1000_stored", "sum"),
    days_ge_1000_stored=("days_ge_1000_stored", "sum")).reset_index()
# Tối đa sáu ca mô tả: ba max cao nhất, hồ sơ 55, và đối chiếu theo thứ tự max trung vị nhóm.
ranked_swe = A1_SWE_VALUE_AUDIT.assign(basin_number=lambda frame: frame["basin_id"].astype(int)).sort_values(
    ["max", "basin_number"], ascending=[False, True], kind="stable", na_position="last")
chosen_swe = []
def add_swe_case(basin_id, role):
    if basin_id not in [item["basin_id"] for item in chosen_swe]:
        chosen_swe.append({"basin_id": basin_id, "selection_role": role})
for basin_id in ranked_swe.loc[ranked_swe["valid_days"].gt(0), "basin_id"].head(3):
    add_swe_case(basin_id, "top_three_record_max")
if "55" in swe_ids and A1_SWE_VALUE_AUDIT.set_index("basin_id").loc["55", "valid_days"] > 0:
    add_swe_case("55", "existing_audit_case")
for category in ["some_ge_1000", "none_ge_1000"]:
    candidates = ranked_swe.loc[ranked_swe["threshold_class"].eq(category)
                               & ~ranked_swe["basin_id"].isin([item["basin_id"] for item in chosen_swe])]
    if len(candidates):
        add_swe_case(candidates.iloc[len(candidates) // 2]["basin_id"], f"middle_max_rank_{category}")
A1_SWE_PLOT_SELECTION = pd.DataFrame(chosen_swe, columns=["basin_id", "selection_role"]).merge(
    A1_SWE_VALUE_AUDIT, on="basin_id", how="left", validate="one_to_one")
swe_figure_path = None
if len(A1_SWE_PLOT_SELECTION):
    plot_rows = (len(A1_SWE_PLOT_SELECTION) + 1) // 2
    fig, axes = plt.subplots(plot_rows, 2, figsize=(15, 3.4 * plot_rows + 1.9), squeeze=False)
    for ax, item in zip(axes.flat, A1_SWE_PLOT_SELECTION.itertuples()):
        annual = A1_SWE_ANNUAL_AUDIT.loc[A1_SWE_ANNUAL_AUDIT["basin_id"].eq(item.basin_id)]
        ax.fill_between(annual["year"].to_numpy(), annual["min"].to_numpy(), annual["max"].to_numpy(),
                        color="#2583a4", alpha=.2, label="Annual minimum–maximum")
        ax.plot(annual["year"], annual["median"], color="#176783", label="Annual median")
        if item.max >= 1000:
            ax.axhline(1000, color="#b66522", ls="--", lw=1, label="1000 (panels reaching threshold)")
        ax.set(title=f"ID {item.basin_id} | full-record min: {item.min:,.2f}",
               xlabel="Calendar year", ylabel="Stored SWE (reference: mm)")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(A1_SWE_PLOT_SELECTION):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("ERA5-Land SWE: annual ranges and medians in selected diagnostic basins", fontsize=16)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065), ncol=3)
    fig.text(.5, .012, "Each panel has its own vertical scale. Annual summaries use available stored daily values; missing years remain missing.\n"
             "Cases are selected by recorded maxima and threshold classes, not randomly. The 1000 threshold is descriptive.\n"
             "CSV unit conversion and physical causes are not verified. No baseline subtraction, clipping or imputation is applied.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .95))
    swe_figure_path = FIGURE_DIR / "A1_P5_swe_annual_ranges.png"
    fig.savefig(swe_figure_path, dpi=160, bbox_inches="tight")
    plt.show()
for suffix, table in [("swe_value_audit", A1_SWE_VALUE_AUDIT), ("swe_annual_audit", A1_SWE_ANNUAL_AUDIT),
                      ("swe_threshold_groups", A1_SWE_THRESHOLD_GROUPS), ("swe_recheck", A1_SWE_RECHECK),
                      ("swe_plot_selection", A1_SWE_PLOT_SELECTION)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None):
    print("(1) Nhóm theo min/max SWE toàn chuỗi; all_ge_1000 chỉ nói về ngày có giá trị:")
    display(A1_SWE_THRESHOLD_GROUPS)
    print("(2) Mười ID có max SWE lớn nhất; kiểm thêm min/phân vị và giá trị đầu/cuối:")
    display(ranked_swe.drop(columns="basin_number").head(10))
    print("(3) Các ID chọn vẽ; không phải mẫu ngẫu nhiên hoặc danh sách lọc mô hình:")
    display(A1_SWE_PLOT_SELECTION)
    print("(4) Đối chiếu 5.9; bảng chi tiết từng ID lưu CSV:")
    display(A1_SWE_RECHECK.drop(columns="basin_id").all().rename("all_ids_pass").to_frame())
print("Đã lưu hình:", swe_figure_path, "; năm có dữ liệu/lịch dùng đúng mẫu số từng ID trong CSV.")
print("Đã lưu năm CSV vào:", TABLE_DIR)
print(f"Thời gian kiểm SWE: {time.perf_counter() - started_swe_audit:.1f} giây.")

# %% [markdown]
# ### 5.11. Đối chiếu Q giờ với qmin/qmean/qmax ngày và các ca cần kiểm
#
# Ghép ngày như timestamp được ghi, không dịch giờ; Q hữu hạn khác -999 tham gia tổng hợp, số âm gắn cờ riêng.

# %% Phần 5.11 - Đối chiếu Q giờ với qmin/qmean/qmax ngày và các ca cần kiểm
# Ghép ngày như timestamp được ghi, không dịch giờ; Q hữu hạn khác -999 tham gia tổng hợp, số âm gắn cờ riêng.
# Đọc từng ID, giữ thống kê đầy đủ và các dòng chênh/ca chọn; không điền thiếu hoặc sửa giá trị nguồn.
started_hourly_compare = time.perf_counter()
audit_calendar = pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
audit_ids = sorted(set(map(str, BASINS)) | {"410"}, key=int)
if not set(audit_ids).issubset(A1_HOURLY_FILES):
    raise RuntimeError("Chạy 2.5 để chuẩn bị đủ Q giờ trong cùng phiên.")
hourly_atol = .0005  # Dung sai so cố định m³/s, không phải xác nhận số chữ số làm tròn của nguồn.
hourly_eps = 1e-9
constant_probe = A1_QMAX_CONSTANT_RUNS.sort_values(["run_days", "run_id"], ascending=[False, True]).head(15).copy()
top_peak_ids = set(A1_Q_PEAK_COMPARISON.head(10)["basin_id"].astype(str))
peak_probe = A1_Q_PEAK_DAYS.loc[A1_Q_PEAK_DAYS["basin_id"].astype(str).isin(top_peak_ids)].copy()
profiles_h, flags_h, comparisons_h, status_h, examples_h, cases_h, peaks_h, constants_h, gaps_h = ([] for _ in range(9))


def a1_hourly_numbers(tokens):
    text = tokens.str.strip()
    number = pd.to_numeric(text, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(number)
    return number, {"blank": text.eq("").to_numpy(), "minus999": number == -999,
                    "negative_except_minus999": finite & (number < 0) & (number != -999),
                    "non_numeric_nonblank": ~text.eq("").to_numpy() & np.isnan(number),
                    "infinite": np.isinf(number), "finite_non_sentinel": finite & (number != -999)}


for file_no, basin_id in enumerate(audit_ids, 1):
    raw_h = pd.read_csv(A1_HOURLY_FILES[basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                        keep_default_na=False, na_filter=False)
    required = ["YYYY", "MM", "DD", "hh", "mm", "qobs"]
    if not set(required).issubset(raw_h):
        raise ValueError(f"ID {basin_id}: thiếu cột giờ đã thấy ở 5.9: {raw_h.columns.tolist()}")
    timestamps = pd.DatetimeIndex(pd.to_datetime(raw_h[required[:5]].rename(
        columns={**DATE_COLUMN_NAMES, "hh": "hour", "mm": "minute"}), errors="coerce"))
    if (timestamps.empty or timestamps.isna().any() or timestamps.duplicated().any()
            or not timestamps.is_monotonic_increasing or (timestamps.minute != 0).any()):
        raise ValueError(f"ID {basin_id}: timestamp không duy nhất/tăng dần/đúng lưới giờ; chưa tổng hợp.")
    qobs, kinds_h = a1_hourly_numbers(raw_h["qobs"])
    source_q = pd.Series(np.where(kinds_h["finite_non_sentinel"], qobs, np.nan), index=timestamps)
    inside = timestamps.normalize().isin(audit_calendar)
    profile = {"basin_id": basin_id, "raw_rows": len(raw_h), "first_time": timestamps.min(), "last_time": timestamps.max(),
               "expected_calendar_hours": len(audit_calendar) * 24, "rows_in_audit_calendar": int(inside.sum()),
               "rows_outside_audit_calendar": int((~inside).sum()),
               "missing_hour_rows_in_calendar": len(audit_calendar) * 24 - int(inside.sum())}
    profile.update({f"{key}_hours": int(value.sum()) for key, value in kinds_h.items()})
    profiles_h.append(profile)
    for flag in ["ckhs", "qceq", "qcol"]:
        if flag in raw_h:
            for token, count in raw_h[flag].value_counts(dropna=False).items():
                flags_h.append({"basin_id": basin_id, "flag": flag, "raw_token": token, "hours": int(count)})
    h = pd.DataFrame({"date": timestamps[inside].normalize(), "present": 1,
                      "qobs": source_q.to_numpy()[inside], "usable": kinds_h["finite_non_sentinel"][inside].astype(int)})
    daily_h = h.groupby("date").agg(hours_present=("present", "sum"), finite_non_sentinel_hours=("usable", "sum"),
                                   reconstructed_qmin=("qobs", "min"), reconstructed_qmean=("qobs", "mean"),
                                   reconstructed_qmax=("qobs", "max")).reindex(audit_calendar)
    daily_h[["hours_present", "finite_non_sentinel_hours"]] = daily_h[["hours_present", "finite_non_sentinel_hours"]].fillna(0).astype(int)
    if daily_h["hours_present"].gt(24).any():
        raise ValueError(f"ID {basin_id}: hơn 24 mốc giờ trong một ngày.")
    daily_h["coverage"] = np.select([daily_h["finite_non_sentinel_hours"].eq(24),
                                    daily_h["finite_non_sentinel_hours"].gt(0)], ["24_finite_hours", "1_to_23_finite_hours"],
                                    default="no_finite_hour")
    raw_d = pd.read_csv(A1_DAILY_FILES["D"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                       usecols=DATE_COLS + ["qmin", "qmean", "qmax"], keep_default_na=False, na_filter=False)
    dates_d = pd.DatetimeIndex(pd.to_datetime(raw_d[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
    if dates_d.isna().any() or dates_d.duplicated().any() or not dates_d.is_monotonic_increasing or not dates_d.isin(audit_calendar).all():
        raise ValueError(f"ID {basin_id}: lịch Q ngày không hợp lệ.")
    daily_h["daily_row_present"] = audit_calendar.isin(dates_d)
    positions = audit_calendar.get_indexer(dates_d)
    max_state = np.full(len(audit_calendar), "missing_daily_row", dtype=object)
    for variable in ["qmin", "qmean", "qmax"]:
        values_d, kinds_d = a1_hourly_numbers(raw_d[variable])
        stored = np.full(len(audit_calendar), np.nan)
        stored[positions] = np.where(kinds_d["finite_non_sentinel"], values_d, np.nan)
        daily_h[f"stored_{variable}"] = stored
        if variable == "qmax":
            labels = np.full(len(values_d), "finite_nonnegative", dtype=object)
            for key in ["blank", "minus999", "non_numeric_nonblank", "infinite", "negative_except_minus999"]:
                labels[kinds_d[key]] = key
            max_state[positions] = labels
        rebuilt = daily_h[f"reconstructed_{variable}"].to_numpy()
        for coverage in ["24_finite_hours", "1_to_23_finite_hours", "no_finite_hour"]:
            selected_days = daily_h["coverage"].eq(coverage).to_numpy()
            paired = selected_days & np.isfinite(stored) & np.isfinite(rebuilt)
            delta = np.abs(stored[paired] - rebuilt[paired])
            comparisons_h.append({"basin_id": basin_id, "variable": variable, "coverage": coverage,
                "calendar_days": int(selected_days.sum()), "paired_days": int(paired.sum()),
                "both_missing_days": int((selected_days & ~np.isfinite(stored) & ~np.isfinite(rebuilt)).sum()),
                "stored_only_days": int((selected_days & np.isfinite(stored) & ~np.isfinite(rebuilt)).sum()),
                "reconstructed_only_days": int((selected_days & ~np.isfinite(stored) & np.isfinite(rebuilt)).sum()),
                "within_1e_9_days": int((delta <= hourly_eps).sum()),
                "within_0p0005_days": int((delta <= hourly_atol + hourly_eps).sum()),
                "max_abs_difference_m3s": float(delta.max()) if len(delta) else np.nan})
        mismatch = np.flatnonzero(np.isfinite(stored) & np.isfinite(rebuilt) & (np.abs(stored - rebuilt) > hourly_atol + hourly_eps))
        if len(mismatch):
            largest = mismatch[np.argsort(-np.abs(stored[mismatch] - rebuilt[mismatch]), kind="stable")[:5]]
            for j in largest:
                examples_h.append({"basin_id": basin_id, "date": audit_calendar[j], "variable": variable,
                    "coverage": daily_h["coverage"].iloc[j], "stored_m3s": stored[j], "reconstructed_m3s": rebuilt[j],
                    "abs_difference_m3s": abs(stored[j] - rebuilt[j])})
    daily_h["qmax_state"] = max_state
    for (coverage, state), count in daily_h.groupby(["coverage", "qmax_state"]).size().items():
        status_h.append({"basin_id": basin_id, "coverage": coverage, "qmax_state": state, "basin_days": int(count)})
    case_mask = np.zeros(len(audit_calendar), dtype=bool)
    if basin_id == "410":
        case_mask |= daily_h["daily_row_present"].to_numpy()
    if basin_id == "94":
        case_mask |= audit_calendar.year.isin([1981, 1991])
    for peak in peak_probe.loc[peak_probe["basin_id"].astype(str).eq(basin_id)].itertuples():
        day = pd.Timestamp(peak.date)
        case_mask |= (audit_calendar >= day - pd.Timedelta(days=1)) & (audit_calendar <= day + pd.Timedelta(days=1))
        record = daily_h.loc[day]
        hourly_day = source_q.loc[(source_q.index >= day) & (source_q.index < day + pd.Timedelta(days=1))].dropna()
        peaks_h.append({"basin_id": basin_id, "date": day, "finite_hours": len(hourly_day),
                        "stored_qmax_m3s": record["stored_qmax"], "hourly_max_m3s": record["reconstructed_qmax"],
                        "abs_difference_m3s": abs(record["stored_qmax"] - record["reconstructed_qmax"]),
                        "first_hour_at_max": hourly_day.idxmax() if len(hourly_day) else pd.NaT})
    for run in constant_probe.loc[constant_probe["basin_id"].astype(str).eq(basin_id)].itertuples():
        mask = (audit_calendar >= pd.Timestamp(run.start_date)) & (audit_calendar <= pd.Timestamp(run.end_date))
        case_mask |= mask
        part = daily_h.loc[mask]
        hourly_part = source_q.loc[(source_q.index >= pd.Timestamp(run.start_date)) &
                                  (source_q.index < pd.Timestamp(run.end_date) + pd.Timedelta(days=1))].dropna()
        compared = part["stored_qmax"].notna() & part["reconstructed_qmax"].notna()
        constants_h.append({"run_id": run.run_id, "basin_id": basin_id, "start_date": run.start_date,
            "end_date": run.end_date, "run_days": run.run_days, "stored_constant_qmax_m3s": run.qmax_m3s,
            "days_24_finite_hours": int(part["finite_non_sentinel_hours"].eq(24).sum()),
            "paired_days": int(compared.sum()), "max_matches_within_0p0005_days": int((compared &
                part["stored_qmax"].sub(part["reconstructed_qmax"]).abs().le(hourly_atol + hourly_eps)).sum()),
            "distinct_hourly_values": hourly_part.nunique(), "distinct_daily_qmean": part["stored_qmean"].nunique(),
            "distinct_daily_qmin": part["stored_qmin"].nunique()})
    if case_mask.any():
        case = daily_h.loc[case_mask].copy().rename_axis("date").reset_index()
        case.insert(0, "basin_id", basin_id)
        cases_h.append(case)
    for run in A1_GAPS_HOURLY_RUNS.loc[A1_GAPS_HOURLY_RUNS["basin_id"].astype(str).eq(basin_id)].itertuples():
        listed_times = pd.date_range(run.start_time, run.end_time, freq="h")
        if len(listed_times) != run.run_hours:
            raise ValueError("Độ dài đoạn gaps không khớp lịch giờ.")
        present = listed_times.isin(timestamps)
        indices = timestamps.get_indexer(listed_times)
        aligned = np.full(len(listed_times), np.nan)
        aligned[present] = qobs[indices[present]]
        finite = np.isfinite(aligned) & (aligned != -999)
        gap_record = {"basin_id": basin_id, "start_time": run.start_time, "end_time": run.end_time,
            "length_group": run.length_group, "listed_hours": len(listed_times), "rows_present": int(present.sum()),
            "finite_non_sentinel_hours": int(finite.sum()), "minus999_hours": int((aligned == -999).sum()),
            "other_missing_qobs_hours": int((present & ~finite & (aligned != -999)).sum()),
            "missing_hour_rows": int((~present).sum()), "short_linear_max_abs_residual": np.nan}
        if run.run_hours <= 6:
            neighbors = source_q.reindex([listed_times[0] - pd.Timedelta(hours=1), listed_times[-1] + pd.Timedelta(hours=1)])
            if finite.all() and neighbors.notna().all():
                linear = np.linspace(neighbors.iloc[0], neighbors.iloc[1], len(listed_times) + 2)[1:-1]
                gap_record["short_linear_max_abs_residual"] = float(np.abs(aligned - linear).max())
        gaps_h.append(gap_record)
    if file_no % 25 == 0 or file_no == len(audit_ids):
        print(f"Đã đối chiếu Q giờ–ngày {file_no}/{len(audit_ids)} ID", flush=True)

A1_HOURLY_QOBS_PROFILE = pd.DataFrame(profiles_h)
A1_HOURLY_FLAG_COUNTS = pd.DataFrame(flags_h, columns=["basin_id", "flag", "raw_token", "hours"])
A1_HOURLY_DAILY_COMPARISON = pd.DataFrame(comparisons_h)
A1_HOURLY_DAILY_STATUS = pd.DataFrame(status_h)
A1_HOURLY_MISMATCH_EXAMPLES = pd.DataFrame(examples_h, columns=["basin_id", "date", "variable", "coverage", "stored_m3s", "reconstructed_m3s", "abs_difference_m3s"])
A1_HOURLY_CASE_DAILY = pd.concat(cases_h, ignore_index=True) if cases_h else pd.DataFrame()
A1_HOURLY_PEAK_CHECK = pd.DataFrame(peaks_h)
A1_HOURLY_CONSTANT_CHECK = pd.DataFrame(constants_h)
A1_HOURLY_GAPS_CHECK = pd.DataFrame(gaps_h)
comparison_counts = ["calendar_days", "paired_days", "both_missing_days", "stored_only_days", "reconstructed_only_days",
                     "within_1e_9_days", "within_0p0005_days"]
comparison_summary = A1_HOURLY_DAILY_COMPARISON.groupby(["variable", "coverage"])[comparison_counts].sum()
comparison_summary["max_abs_difference_m3s"] = A1_HOURLY_DAILY_COMPARISON.groupby(["variable", "coverage"])["max_abs_difference_m3s"].max()
if (A1_HOURLY_DAILY_STATUS["basin_days"].sum() != len(audit_ids) * len(audit_calendar)
        or not (A1_HOURLY_DAILY_COMPARISON[["paired_days", "both_missing_days", "stored_only_days", "reconstructed_only_days"]].sum(axis=1)
                == A1_HOURLY_DAILY_COMPARISON["calendar_days"]).all()):
    raise ValueError("Số đếm đối chiếu giờ–ngày không cộng khớp.")
if (not A1_HOURLY_GAPS_CHECK[["finite_non_sentinel_hours", "minus999_hours", "other_missing_qobs_hours", "missing_hour_rows"]]
        .sum(axis=1).eq(A1_HOURLY_GAPS_CHECK["listed_hours"]).all()):
    raise ValueError("Số loại mốc gaps không cộng khớp nguồn giờ.")
for suffix, table in [("hourly_qobs_profile", A1_HOURLY_QOBS_PROFILE), ("hourly_flag_counts", A1_HOURLY_FLAG_COUNTS),
    ("hourly_daily_comparison", A1_HOURLY_DAILY_COMPARISON), ("hourly_daily_status", A1_HOURLY_DAILY_STATUS),
    ("hourly_mismatch_examples", A1_HOURLY_MISMATCH_EXAMPLES), ("hourly_case_daily", A1_HOURLY_CASE_DAILY),
    ("hourly_peak_check", A1_HOURLY_PEAK_CHECK), ("hourly_constant_check", A1_HOURLY_CONSTANT_CHECK),
    ("hourly_gaps_check", A1_HOURLY_GAPS_CHECK)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None):
    print("(1) Tổng kiểm kê qobs giờ; số âm khác -999 giữ trong phép so và gắn cờ:")
    display(A1_HOURLY_QOBS_PROFILE.drop(columns=["basin_id", "first_time", "last_time"]).sum().to_frame("total"))
    display(A1_HOURLY_FLAG_COUNTS.groupby(["flag", "raw_token"])["hours"].sum().to_frame())
    print("(2) So Q ngày với min/mean/max giờ có số; hai mức dung sai m³/s cố định:")
    display(comparison_summary)
    display(A1_HOURLY_DAILY_STATUS.groupby(["coverage", "qmax_state"])["basin_days"].sum().to_frame())
    print("(3) Mọi ngày đồng cực đại của top 10 ID mm/ngày:")
    display(A1_HOURLY_PEAK_CHECK)
    print("(4) Đối chiếu 15 đoạn hằng dài nhất; số mức giờ khác số mức qmax ngày:")
    display(A1_HOURLY_CONSTANT_CHECK)
    print("(5) Mốc gaps trong nguồn giờ sau xử lý; gần tuyến tính chưa xác nhận toàn bộ quy trình:")
    display(A1_HOURLY_GAPS_CHECK.groupby("length_group")[["listed_hours", "rows_present", "finite_non_sentinel_hours",
        "minus999_hours", "other_missing_qobs_hours", "missing_hour_rows"]].sum())
    short_residual = A1_HOURLY_GAPS_CHECK["short_linear_max_abs_residual"].dropna()
    print("Đoạn ngắn có đủ số và hai biên để so tuyến tính:", len(short_residual),
          "; số đoạn sai khác ≤0,0005:", int(short_residual.le(hourly_atol + hourly_eps).sum()),
          "; max sai khác:", short_residual.max())
    print("(6) ID 410: trạng thái ngày trong khoảng tệp ngày và ngày có 1–23 giờ hữu hạn:")
    case410 = A1_HOURLY_CASE_DAILY.loc[A1_HOURLY_CASE_DAILY["basin_id"].eq("410") & A1_HOURLY_CASE_DAILY["daily_row_present"]]
    display(case410.groupby(["coverage", "qmax_state"]).size().rename("days").to_frame())
    display(case410.loc[case410["coverage"].eq("1_to_23_finite_hours")])
    print("(7) Tối đa 15 chênh lớn nhất; CSV giữ tối đa 5 ví dụ/ID/biến:")
    display(A1_HOURLY_MISMATCH_EXAMPLES.sort_values("abs_difference_m3s", ascending=False, kind="stable").head(15))
print("Phạm vi:", audit_calendar.min().date(), "–", audit_calendar.max().date(), "; ID:", len(audit_ids))
print("Đã lưu chín CSV vào:", TABLE_DIR)
print(f"Thời gian đối chiếu Q giờ–ngày: {time.perf_counter() - started_hourly_compare:.1f} giây.")

# %% [markdown]
# ### 5.12. Cờ tại đỉnh/đoạn hằng, giờ âm và hai ca nội suy cần đọc kỹ
#
# Chỉ đọc các ID liên quan từ 5.11; không quét lại 452 tệp, tải thêm hoặc sửa giá trị.

# %% Phần 5.12 - Cờ tại đỉnh/đoạn hằng, giờ âm và hai ca nội suy cần đọc kỹ
# Chỉ đọc các ID liên quan từ 5.11; không quét lại 452 tệp, tải thêm hoặc sửa giá trị.
# Biên 0,001 chỉ xét GIẢ THIẾT làm tròn gần nhất 0,001 ở cả hai biên và giá trị nội suy.
started_hourly_flags = time.perf_counter()
flag_eps = 1e-9
short_exceptions = A1_HOURLY_GAPS_CHECK.loc[
    A1_HOURLY_GAPS_CHECK["short_linear_max_abs_residual"].gt(.0005 + flag_eps)].copy()
negative_ids = set(A1_HOURLY_QOBS_PROFILE.loc[
    A1_HOURLY_QOBS_PROFILE["negative_except_minus999_hours"].gt(0), "basin_id"].astype(str))
flag_ids = sorted(negative_ids | set(short_exceptions["basin_id"].astype(str)) |
                  set(A1_HOURLY_PEAK_CHECK["basin_id"].astype(str)) |
                  set(A1_HOURLY_CONSTANT_CHECK["basin_id"].astype(str)), key=int)
exception_rows, exception_hour_rows, peak_flag_rows, constant_flag_rows, negative_frames = [], [], [], [], []


def a1_hourly_flag_counts(frame):
    return {"hour_rows": len(frame), "ckhs_0_hours": int(frame["ckhs"].eq("0").sum()),
            "ckhs_1_hours": int(frame["ckhs"].eq("1").sum()),
            "ckhs_blank_hours": int(frame["ckhs"].eq("").sum()),
            "qceq_1_hours": int(frame["qceq"].eq("1").sum()),
            "qcol_1_hours": int(frame["qcol"].eq("1").sum()),
            "qcol_blank_hours": int(frame["qcol"].eq("").sum())}


for number, basin_id in enumerate(flag_ids, 1):
    frame = pd.read_csv(A1_HOURLY_FILES[basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                        usecols=["YYYY", "MM", "DD", "hh", "mm", "qobs", "ckhs", "qceq", "qcol"],
                        keep_default_na=False, na_filter=False)
    dates = pd.DatetimeIndex(pd.to_datetime(frame[["YYYY", "MM", "DD", "hh", "mm"]].rename(
        columns={**DATE_COLUMN_NAMES, "hh": "hour", "mm": "minute"}), errors="coerce"))
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing or (dates.minute != 0).any():
        raise ValueError(f"ID {basin_id}: timestamp khác điều kiện 5.11.")
    frame = frame[["qobs", "ckhs", "qceq", "qcol"]].apply(lambda col: col.str.strip())
    frame.index = dates
    qobs = pd.to_numeric(frame["qobs"], errors="coerce").to_numpy(dtype=float)
    frame["qobs_m3s"] = np.where(np.isfinite(qobs) & (qobs != -999), qobs, np.nan)
    for peak in A1_HOURLY_PEAK_CHECK.loc[A1_HOURLY_PEAK_CHECK["basin_id"].astype(str).eq(basin_id)].itertuples():
        day = pd.Timestamp(peak.date)
        part = frame.loc[(frame.index >= day) & (frame.index < day + pd.Timedelta(days=1))]
        at_max = part.loc[np.isclose(part["qobs_m3s"], peak.stored_qmax_m3s, atol=flag_eps, rtol=0)]
        if at_max.empty:
            raise ValueError(f"ID {basin_id}: đỉnh đọc lại không còn khớp 5.11.")
        peak_flag_rows.append({"basin_id": basin_id, "date": day, "qmax_m3s": peak.stored_qmax_m3s,
            "finite_hours_in_day": int(part["qobs_m3s"].notna().sum()), "max_hours": len(at_max),
            "first_max_hour": at_max.index.min(), **a1_hourly_flag_counts(at_max)})
    for run in A1_HOURLY_CONSTANT_CHECK.loc[A1_HOURLY_CONSTANT_CHECK["basin_id"].astype(str).eq(basin_id)].itertuples():
        part = frame.loc[(frame.index >= pd.Timestamp(run.start_date)) &
                         (frame.index < pd.Timestamp(run.end_date) + pd.Timedelta(days=1))]
        constant_flag_rows.append({"run_id": run.run_id, "basin_id": basin_id, "start_date": run.start_date,
            "end_date": run.end_date, "run_days": run.run_days,
            "distinct_hourly_values": part["qobs_m3s"].nunique(), **a1_hourly_flag_counts(part)})
    if basin_id in negative_ids:
        neg = frame.loc[frame["qobs_m3s"].lt(0)].copy()
        negative = neg.rename_axis("timestamp").reset_index()
        negative.insert(0, "basin_id", basin_id)
        raw_day = pd.read_csv(A1_DAILY_FILES["D"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                              usecols=DATE_COLS + ["qmin"], keep_default_na=False, na_filter=False)
        day_dates = pd.DatetimeIndex(pd.to_datetime(raw_day[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
        if day_dates.isna().any() or day_dates.duplicated().any():
            raise ValueError(f"ID {basin_id}: lịch qmin ngày không hợp lệ.")
        negative["date"] = negative["timestamp"].dt.normalize()
        negative["stored_qmin_raw"] = pd.Series(raw_day["qmin"].to_numpy(), index=day_dates).reindex(negative["date"]).to_numpy()
        negative_frames.append(negative)
    for run in short_exceptions.loc[short_exceptions["basin_id"].astype(str).eq(basin_id)].itertuples():
        gap_times = pd.date_range(run.start_time, run.end_time, freq="h")
        window_times = pd.date_range(gap_times[0] - pd.Timedelta(hours=1), gap_times[-1] + pd.Timedelta(hours=1), freq="h")
        window = frame.reindex(window_times)
        values = window["qobs_m3s"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Ca nội suy ngoại lệ không còn đủ hai biên/giá trị như 5.11.")
        weights = np.arange(1, len(gap_times) + 1) / (len(gap_times) + 1)
        linear = (1 - weights) * values[0] + weights * values[-1]
        residual = values[1:-1] - linear
        maximum = float(np.abs(residual).max())
        if not np.isclose(maximum, run.short_linear_max_abs_residual, atol=flag_eps, rtol=0):
            raise ValueError("Số dư đoạn ngắn đọc lại khác 5.11.")
        on_grid = bool(np.isclose(values * 1000, np.rint(values * 1000), atol=1e-8, rtol=0).all())
        exception_rows.append({"basin_id": basin_id, "start_time": run.start_time, "end_time": run.end_time,
            "gap_hours": len(gap_times), "left_qobs_m3s": values[0], "right_qobs_m3s": values[-1],
            "all_window_values_on_0p001_grid": on_grid, "max_abs_residual_m3s": maximum,
            "within_common_0p0005": maximum <= .0005 + flag_eps,
            "within_assumed_0p001_bound": maximum <= .001 + flag_eps})
        for index, stamp in enumerate(window_times):
            inside = 0 < index < len(window_times) - 1
            item = window.loc[stamp]
            exception_hour_rows.append({"basin_id": basin_id, "gap_start": run.start_time, "timestamp": stamp,
                "role": "gap" if inside else "left_neighbor" if index == 0 else "right_neighbor",
                "qobs_raw": item["qobs"], "qobs_m3s": item["qobs_m3s"],
                "linear_from_stored_neighbors": linear[index - 1] if inside else np.nan,
                "residual_m3s": residual[index - 1] if inside else np.nan,
                "ckhs_raw": item["ckhs"], "qceq_raw": item["qceq"], "qcol_raw": item["qcol"]})
    if number % 10 == 0 or number == len(flag_ids):
        print(f"Đã đọc cờ/ca chi tiết {number}/{len(flag_ids)} ID liên quan", flush=True)
A1_HOURLY_SHORT_EXCEPTIONS = pd.DataFrame(exception_rows, columns=["basin_id", "start_time", "end_time", "gap_hours",
    "left_qobs_m3s", "right_qobs_m3s", "all_window_values_on_0p001_grid", "max_abs_residual_m3s",
    "within_common_0p0005", "within_assumed_0p001_bound"])
A1_HOURLY_SHORT_EXCEPTION_HOURS = pd.DataFrame(exception_hour_rows, columns=["basin_id", "gap_start", "timestamp", "role",
    "qobs_raw", "qobs_m3s", "linear_from_stored_neighbors", "residual_m3s", "ckhs_raw", "qceq_raw", "qcol_raw"])
A1_HOURLY_PEAK_FLAGS = pd.DataFrame(peak_flag_rows)
A1_HOURLY_CONSTANT_FLAGS = pd.DataFrame(constant_flag_rows)
A1_HOURLY_NEGATIVE_HOURS = pd.concat(negative_frames, ignore_index=True) if negative_frames else pd.DataFrame(
    columns=["basin_id", "timestamp", "qobs", "ckhs", "qceq", "qcol", "qobs_m3s", "date", "stored_qmin_raw"])
if len(A1_HOURLY_NEGATIVE_HOURS) != A1_HOURLY_QOBS_PROFILE["negative_except_minus999_hours"].sum():
    raise ValueError("Số giờ âm đọc lại không khớp 5.11.")
model_comparison = A1_HOURLY_DAILY_COMPARISON.loc[A1_HOURLY_DAILY_COMPARISON["basin_id"].astype(str).isin(set(map(str, BASINS)))]
count_fields = ["calendar_days", "paired_days", "both_missing_days", "stored_only_days", "reconstructed_only_days",
                "within_1e_9_days", "within_0p0005_days"]
A1_HOURLY_MODEL_COMPARISON = model_comparison.groupby(["variable", "coverage"])[count_fields].sum().reset_index()
for suffix, table in [("hourly_short_exceptions", A1_HOURLY_SHORT_EXCEPTIONS),
    ("hourly_short_exception_hours", A1_HOURLY_SHORT_EXCEPTION_HOURS), ("hourly_peak_flags", A1_HOURLY_PEAK_FLAGS),
    ("hourly_constant_flags", A1_HOURLY_CONSTANT_FLAGS), ("hourly_negative_hours", A1_HOURLY_NEGATIVE_HOURS),
    ("hourly_model_comparison", A1_HOURLY_MODEL_COMPARISON)]:
    table.to_csv(TABLE_DIR / f"A1_P5_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None):
    print("(1) Phép so riêng 451 ID mô hình, tách ca 410 khỏi mẫu số:")
    display(A1_HOURLY_MODEL_COMPARISON)
    print("(2) Các đoạn ngắn vượt 0,0005; biên 0,001 chỉ tương thích GIẢ ĐỊNH làm tròn cả hai biên và điểm giữa:")
    display(A1_HOURLY_SHORT_EXCEPTIONS)
    display(A1_HOURLY_SHORT_EXCEPTION_HOURS)
    print("(3) Cờ tại mọi giờ đạt đỉnh top 10; ckhs/qceq/qcol không tự là quy tắc loại Q:")
    display(A1_HOURLY_PEAK_FLAGS)
    print("(4) Cờ trong 15 đoạn qmax ngày hằng đã kiểm:")
    display(A1_HOURLY_CONSTANT_FLAGS)
    print("(5) Giờ Q âm khác -999; số ngày và tối đa 20 dòng, toàn bộ lưu CSV:")
    display(A1_HOURLY_NEGATIVE_HOURS.groupby("basin_id").agg(negative_hours=("timestamp", "size"),
        affected_days=("date", "nunique"), min_qobs_m3s=("qobs_m3s", "min")))
    display(A1_HOURLY_NEGATIVE_HOURS.head(20))
print("Tổng giờ âm:", len(A1_HOURLY_NEGATIVE_HOURS), "; ngày–ID có giờ âm:",
      len(A1_HOURLY_NEGATIVE_HOURS[["basin_id", "date"]].drop_duplicates()))
print("Đã lưu sáu CSV vào:", TABLE_DIR)
print(f"Thời gian đọc cờ/ca chi tiết: {time.perf_counter() - started_hourly_flags:.1f} giây.")

# %% [markdown]
# ## Phần 6. Năm cặp khí tượng và khác biệt giữa nguồn
#
# Ghép đúng ngày–ID để mô tả độ phủ, phân phối, mùa và tương quan; giữ giới hạn ngữ nghĩa/đơn vị CSV.

# %% [markdown]
# ### 6.1. Năm cặp khí tượng: ngữ nghĩa, ngày chung và mức lưu trữ theo năm
#
# Giữ đúng năm cặp trong mã phân tích tác giả; net/downward và đơn vị CSV chưa được coi tương đương.

# %% Phần 6.1 - Năm cặp khí tượng: ngữ nghĩa, ngày chung và mức lưu trữ theo năm
# Giữ đúng năm cặp trong mã phân tích tác giả; net/downward và đơn vị CSV chưa được coi tương đương.
# Mean của hai nguồn dùng cùng ngày–ID; không tính bias vật lý, đổi đơn vị/dấu, chuẩn hóa hay dịch ngày.
started_pair_scope = time.perf_counter()
from matplotlib.ticker import MaxNLocator
pair_rows = [
    ("solar", "ERA5L_surf_net_solar_rad_mean", "ECMWF_ssrd", "Solar radiation", "W/m²", "W/m²", "J/m²",
     "Net shortwave (reference name)", "Downward shortwave (native ssrd)", "quantity mismatch + CSV unit unresolved"),
    ("temperature", "ERA5L_2m_temp_mean", "ECMWF_t2m", "2 m temperature", "°C", "°C", "°C",
     "Daily mean 2 m temperature", "Daily mean forecast 2 m temperature", "CSV conversion and forecast-day convention to trace"),
    ("dewpoint", "ERA5L_2m_dp_temp_mean", "ECMWF_d2m", "2 m dew point", "°C", "°C", "°C",
     "Daily mean 2 m dew point", "Daily mean forecast 2 m dew point", "CSV conversion and forecast-day convention to trace"),
    ("precipitation", "MSWEP_RR", "ECMWF_tp", "Precipitation", "mm", "mm", "mm",
     "Daily precipitation from MSWEP", "Forecast total precipitation", "accumulation interval, CSV conversion and product version to trace"),
    ("evaporation", "GLEAM_ETA", "ECMWF_e", "Evaporation", "mm", "mm", "mm",
     "Actual evaporation from GLEAM", "Forecast evaporation", "sign, components, CSV conversion and product version to trace"),
]
A1_METEO_PAIR_CONTRACT = pd.DataFrame(pair_rows, columns=["pair_id", "reference_variable", "forecast_variable", "label",
    "reference_unit_paper", "forecast_unit_paper", "author_analysis_unit_label", "reference_quantity", "forecast_quantity", "open_question"])
pair_ids = sorted(set(map(str, selected_ids)), key=int)
pair_calendar = pd.DatetimeIndex(meteo_calendar)
pair_fields = sorted(set(A1_METEO_PAIR_CONTRACT["reference_variable"]) | set(A1_METEO_PAIR_CONTRACT["forecast_variable"]))
profile_reference = A1_FORCING_VALUE_PROFILE.copy()
profile_reference["basin_id"] = profile_reference["basin_id"].astype(str)
profile_reference = profile_reference.set_index(["basin_id", "variable"])
scope_masks = [("year", str(year), pair_calendar.year == year) for year in sorted(set(pair_calendar.year))]
scope_masks += [("period", period, (pair_calendar >= pd.Timestamp(start)) & (pair_calendar <= pd.Timestamp(end)))
                for period, (start, end) in PERIODS.items()]
profile_pairs, scoped_pairs = [], []
for file_no, basin_id in enumerate(pair_ids, 1):
    raw = pd.read_csv(A1_DAILY_FILES["A"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                      usecols=DATE_COLS + pair_fields, keep_default_na=False, na_filter=False)
    dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
    if dates.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing or not dates.isin(pair_calendar).all():
        raise ValueError(f"ID {basin_id}: lịch A khác điều kiện 5.8.")
    positions = pair_calendar.get_indexer(dates)
    parsed = {}
    for variable in pair_fields:
        values = pd.to_numeric(raw[variable].str.strip(), errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(values) & (values != -999)
        aligned = np.full(len(pair_calendar), np.nan)
        aligned[positions[usable]] = values[usable]
        if int(usable.sum()) != int(profile_reference.loc[(basin_id, variable), "valid_days"]):
            raise ValueError(f"{basin_id}/{variable}: số ngày có số khác 5.8.")
        parsed[variable] = aligned
    for pair in A1_METEO_PAIR_CONTRACT.itertuples():
        reference, forecast = parsed[pair.reference_variable], parsed[pair.forecast_variable]
        reference_ok, forecast_ok = np.isfinite(reference), np.isfinite(forecast)
        paired = reference_ok & forecast_ok
        profile_pairs.append({"basin_id": basin_id, "pair_id": pair.pair_id,
            "expected_cells": len(pair_calendar), "reference_cells": int(reference_ok.sum()),
            "forecast_cells": int(forecast_ok.sum()), "paired_cells": int(paired.sum()),
            "reference_only_cells": int((reference_ok & ~forecast_ok).sum()),
            "forecast_only_cells": int((~reference_ok & forecast_ok).sum()),
            "both_missing_cells": int((~reference_ok & ~forecast_ok).sum()),
            "first_paired_date": pair_calendar[paired].min(), "last_paired_date": pair_calendar[paired].max()})
        for scope_kind, scope, mask in scope_masks:
            selected_pairs = paired & mask
            x, y = reference[selected_pairs], forecast[selected_pairs]
            scoped_pairs.append({"basin_id": basin_id, "pair_id": pair.pair_id, "scope_kind": scope_kind, "scope": scope,
                "expected_cells": int(mask.sum()), "paired_cells": len(x),
                "reference_sum_raw": float(x.sum()), "forecast_sum_raw": float(y.sum()),
                "reference_zero_cells": int((x == 0).sum()), "forecast_zero_cells": int((y == 0).sum()),
                "reference_negative_cells": int((x < 0).sum()), "forecast_negative_cells": int((y < 0).sum()),
                "opposite_nonzero_sign_cells": int((((x < 0) & (y > 0)) | ((x > 0) & (y < 0))).sum())})
    if file_no % 50 == 0 or file_no == len(pair_ids):
        print(f"Đã ghép năm cặp khí tượng {file_no}/{len(pair_ids)} ID", flush=True)
A1_METEO_PAIR_PROFILE = pd.DataFrame(profile_pairs)
A1_METEO_PAIR_BY_BASIN_SCOPE = pd.DataFrame(scoped_pairs)
profile_count_fields = ["expected_cells", "reference_cells", "forecast_cells", "paired_cells", "reference_only_cells",
                        "forecast_only_cells", "both_missing_cells"]
pair_profile = A1_METEO_PAIR_PROFILE.groupby("pair_id")[profile_count_fields].sum()
pair_profile["basins"] = A1_METEO_PAIR_PROFILE.groupby("pair_id")["basin_id"].nunique()
pair_profile["basins_with_pairs"] = A1_METEO_PAIR_PROFILE.loc[A1_METEO_PAIR_PROFILE["paired_cells"].gt(0)].groupby("pair_id")["basin_id"].nunique().reindex(pair_profile.index, fill_value=0)
pair_profile["first_paired_date"] = A1_METEO_PAIR_PROFILE.groupby("pair_id")["first_paired_date"].min()
pair_profile["last_paired_date"] = A1_METEO_PAIR_PROFILE.groupby("pair_id")["last_paired_date"].max()
A1_METEO_PAIR_SCOPE = pair_profile.reset_index()
sum_fields = ["expected_cells", "paired_cells", "reference_sum_raw", "forecast_sum_raw", "reference_zero_cells",
              "forecast_zero_cells", "reference_negative_cells", "forecast_negative_cells", "opposite_nonzero_sign_cells"]
pair_scopes = A1_METEO_PAIR_BY_BASIN_SCOPE.groupby(["pair_id", "scope_kind", "scope"])[sum_fields].sum().reset_index()
denominator = pair_scopes["paired_cells"].replace(0, np.nan)
pair_scopes["reference_mean_raw"] = pair_scopes["reference_sum_raw"] / denominator
pair_scopes["forecast_mean_raw"] = pair_scopes["forecast_sum_raw"] / denominator
pair_scopes["paired_pct_expected"] = 100 * pair_scopes["paired_cells"] / pair_scopes["expected_cells"].replace(0, np.nan)
for field in ["reference_zero", "forecast_zero", "reference_negative", "forecast_negative", "opposite_nonzero_sign"]:
    pair_scopes[f"{field}_pct_paired"] = 100 * pair_scopes[f"{field}_cells"] / denominator
A1_METEO_PAIR_SCOPED_SUMMARY = pair_scopes
annual = pair_scopes.loc[pair_scopes["scope_kind"].eq("year")].copy()
annual["year"] = annual["scope"].astype(int)
year_counts = annual.pivot(index="year", columns="pair_id", values="paired_cells")
if (not A1_METEO_PAIR_PROFILE[["paired_cells", "reference_only_cells", "forecast_only_cells", "both_missing_cells"]]
        .sum(axis=1).eq(A1_METEO_PAIR_PROFILE["expected_cells"]).all()
        or not annual.groupby("pair_id")["paired_cells"].sum().reindex(pair_profile.index).eq(pair_profile["paired_cells"]).all()):
    raise ValueError("Số cặp theo năm/trạng thái không cộng khớp toàn lịch.")
figure_years = year_counts.index[year_counts.sum(axis=1).gt(0)]
pair_scope_figure = None
if len(figure_years):
    fig, axes = plt.subplots(3, 2, figsize=(16, 13.5))
    for ax, pair in zip(axes.flat, A1_METEO_PAIR_CONTRACT.itertuples()):
        part = annual.loc[annual["pair_id"].eq(pair.pair_id)].set_index("year").reindex(figure_years)
        ax.plot(figure_years, part["reference_mean_raw"], color="#247a99", marker="o", ms=3, label="Reference column")
        ax.plot(figure_years, part["forecast_mean_raw"], color="#cd7634", marker="s", ms=3, label="ECMWF column")
        note = " | net vs downward; CSV mapping unresolved" if pair.pair_id == "solar" else ""
        ax.set(title=pair.label + note, xlabel="Calendar year",
               ylabel="Stored values (CSV unit unresolved)" if pair.pair_id == "solar" else f"Stored values (paper reference: {pair.reference_unit_paper})")
        ax.grid(alpha=.2)
    ax = axes.flat[-1]
    for pair in A1_METEO_PAIR_CONTRACT.itertuples():
        part = annual.loc[annual["pair_id"].eq(pair.pair_id)].set_index("year").reindex(figure_years)
        ax.plot(figure_years, part["paired_pct_expected"], marker="o", ms=3, label=pair.label)
    ax.set(title="Same-date pair coverage", xlabel="Calendar year", ylabel="Expected basin-days paired (%)", ylim=(-3, 103))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="lower left")
    for ax in axes.flat:
        ax.set_xlim(figure_years.min() - .5, figure_years.max() + .5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    fig.suptitle("Five author-selected meteorological pairs: coverage and stored annual means", fontsize=16)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .064), ncol=2)
    fig.text(.5, .013, "Both means use exactly the same available basin-days in each year; zeros and finite signed values are retained.\n"
             "Reference columns are ERA5-Land, MSWEP and GLEAM; they are not independent ground truth.\n"
             "No unit conversion, sign reversal, normalization or date shift is applied. Solar quantities may differ by definition.\n"
             "Raw means describe stored values and do not establish physical forecast bias; overlapping coverage curves are retained.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .96))
    pair_scope_figure = FIGURE_DIR / "A1_P6_pair_scope_annual_means.png"
    fig.savefig(pair_scope_figure, dpi=160, bbox_inches="tight")
    plt.show()
for suffix, table in [("pair_contract", A1_METEO_PAIR_CONTRACT), ("pair_profile", A1_METEO_PAIR_PROFILE),
    ("pair_scope", A1_METEO_PAIR_SCOPE), ("pair_by_basin_scope", A1_METEO_PAIR_BY_BASIN_SCOPE),
    ("pair_scoped_summary", A1_METEO_PAIR_SCOPED_SUMMARY)]:
    table.to_csv(TABLE_DIR / f"A1_P6_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None, "display.max_colwidth", 90):
    print("(1) Năm cặp từ mã phân tích tác giả; đơn vị tham chiếu chưa xác nhận chuyển đổi CSV:")
    display(A1_METEO_PAIR_CONTRACT)
    print("(2) Phủ cặp đúng cùng ngày–ID trên toàn lịch; mọi loại ngoài cặp được đếm riêng:")
    display(A1_METEO_PAIR_SCOPE)
    print("(3) Số cặp theo năm có ít nhất một cặp; các năm không có cặp vẫn lưu trong CSV:")
    display(year_counts.loc[figure_years])
    print("(4) Mean và dấu trên đúng cùng cặp trong từng kỳ; chưa gọi là bias vật lý:")
    display(pair_scopes.loc[pair_scopes["scope_kind"].eq("period"), ["pair_id", "scope", "expected_cells", "paired_cells",
        "reference_mean_raw", "forecast_mean_raw", "reference_zero_pct_paired", "forecast_zero_pct_paired",
        "reference_negative_pct_paired", "forecast_negative_pct_paired", "opposite_nonzero_sign_pct_paired"]])
print("Đã lưu hình:", pair_scope_figure, "; năm không có cặp giữ mean NaN.")
print("Đã lưu năm CSV vào:", TABLE_DIR)
print(f"Thời gian kiểm phạm vi năm cặp: {time.perf_counter() - started_pair_scope:.1f} giây.")

# %% [markdown]
# ### 6.2. Phân phối và sai khác cùng ngày của năm cặp theo lưu vực/kỳ
#
# Ghép đúng ngày như 6.1; đo W1, sai khác cùng ngày và tương quan riêng mỗi ID/kỳ.

# %% Phần 6.2 - Phân phối và sai khác cùng ngày của năm cặp theo lưu vực/kỳ
# Ghép đúng ngày như 6.1; đo W1, sai khác cùng ngày và tương quan riêng mỗi ID/kỳ.
# Đây là mô tả giá trị lưu trữ; không chuẩn hóa, đổi dấu/đơn vị hoặc xác nhận sai số dự báo vật lý.
started_pair_dist = time.perf_counter()
from scipy.stats import wasserstein_distance


def pair_raw_metrics(reference, forecast):
    # Cùng n và trọng số đều: W1 là mean khoảng cách giữa hai dãy đã sắp xếp.
    # So thêm MAE cùng ngày để phân biệt phân phối giống nhau với đúng diễn biến ngày.
    n = len(reference)
    if n != len(forecast) or not (np.isfinite(reference).all() and np.isfinite(forecast).all()):
        raise ValueError("Hai dãy phải cùng chiều dài, hữu hạn và đã ghép ngày.")
    fields = ["reference_mean_raw", "forecast_mean_raw", "mean_delta_raw", "mae_raw", "rmse_raw", "w1_raw", "pearson_r"]
    result = {field: np.nan for field in fields}
    result.update(paired_cells=n, correlation_status="no_pairs" if n == 0 else "insufficient_pairs")
    if not n:
        return result
    delta = forecast - reference
    result.update(reference_mean_raw=float(reference.mean()), forecast_mean_raw=float(forecast.mean()),
                  mean_delta_raw=float(delta.mean()), mae_raw=float(np.abs(delta).mean()),
                  rmse_raw=float(np.sqrt(np.mean(delta * delta))),
                  w1_raw=float(np.abs(np.sort(forecast) - np.sort(reference)).mean()))
    x, y = reference - reference.mean(), forecast - forecast.mean()
    constant_x, constant_y = bool(np.all(reference == reference[0])), bool(np.all(forecast == forecast[0]))
    if constant_x or constant_y:
        result["correlation_status"] = "both_constant" if constant_x and constant_y else ("reference_constant" if constant_x else "forecast_constant")
    elif n >= 2:
        result["pearson_r"] = float(np.clip(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)), -1, 1))
        result["correlation_status"] = "defined"
    tolerance = 1e-10 * max(1., result["rmse_raw"])
    if not (abs(result["mean_delta_raw"]) <= result["w1_raw"] + tolerance
            and result["w1_raw"] <= result["mae_raw"] + tolerance
            and result["mae_raw"] <= result["rmse_raw"] + tolerance):
        raise ValueError("Bất đẳng thức |mean delta| <= W1 <= MAE <= RMSE không khớp.")
    return result


metric_rows, quantile_rows = [], []
dist_contract = A1_METEO_PAIR_CONTRACT.copy()
dist_ids = sorted(set(map(str, selected_ids)), key=int)
dist_calendar = pd.DatetimeIndex(meteo_calendar)
dist_fields = sorted(set(dist_contract.reference_variable) | set(dist_contract.forecast_variable))
dist_scope_reference = A1_METEO_PAIR_BY_BASIN_SCOPE.loc[A1_METEO_PAIR_BY_BASIN_SCOPE.scope_kind.eq("period")].copy()
dist_scope_reference["basin_id"] = dist_scope_reference.basin_id.astype(str)
dist_scope_reference = dist_scope_reference.set_index(["basin_id", "pair_id", "scope"])
quantile_levels = np.array([0, .01, .05, .5, .95, .99, 1.])
quantile_names = ["min", "p01", "p05", "median", "p95", "p99", "max"]
dist_masks = {period: (dist_calendar >= pd.Timestamp(start)) & (dist_calendar <= pd.Timestamp(end))
              for period, (start, end) in PERIODS.items()}
for file_no, basin_id in enumerate(dist_ids, 1):
    raw = pd.read_csv(A1_DAILY_FILES["A"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                      usecols=DATE_COLS + dist_fields, keep_default_na=False, na_filter=False)
    dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
    if dates.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing or not dates.isin(dist_calendar).all():
        raise ValueError(f"ID {basin_id}: lịch không khớp điều kiện 6.1.")
    positions = dist_calendar.get_indexer(dates)
    aligned_fields = {}
    for variable in dist_fields:
        values = pd.to_numeric(raw[variable].str.strip(), errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(values) & (values != -999)
        aligned = np.full(len(dist_calendar), np.nan)
        aligned[positions[usable]] = values[usable]
        aligned_fields[variable] = aligned
    for pair in dist_contract.itertuples():
        reference, forecast = aligned_fields[pair.reference_variable], aligned_fields[pair.forecast_variable]
        for period, scope_mask in dist_masks.items():
            paired = scope_mask & np.isfinite(reference) & np.isfinite(forecast)
            x, y = reference[paired], forecast[paired]
            prior = dist_scope_reference.loc[(basin_id, pair.pair_id, period)]
            if (len(x) != int(prior.paired_cells) or int(scope_mask.sum()) != int(prior.expected_cells)
                    or not np.allclose([x.sum(), y.sum()], [prior.reference_sum_raw, prior.forecast_sum_raw], rtol=1e-12, atol=1e-8)):
                raise ValueError(f"{basin_id}/{pair.pair_id}/{period}: cặp/giá trị không khớp 6.1.")
            metrics = pair_raw_metrics(x, y)
            # Kiểm độc lập W1 theo tích phân CDF của SciPy, không dùng phép đổi min-max tác giả.
            scipy_w1 = float(wasserstein_distance(x, y)) if len(x) else np.nan
            if not np.isclose(metrics["w1_raw"], scipy_w1, rtol=1e-10, atol=1e-10, equal_nan=True):
                raise ValueError(f"{basin_id}/{pair.pair_id}/{period}: W1 hai cách không khớp.")
            metric_rows.append(dict(basin_id=basin_id, pair_id=pair.pair_id, period=period,
                                    expected_cells=int(scope_mask.sum()), **metrics))
            for side, values in [("reference", x), ("forecast", y), ("forecast_minus_reference", y - x)]:
                qs = np.quantile(values, quantile_levels) if len(values) else np.full(len(quantile_levels), np.nan)
                quantile_rows.append(dict(basin_id=basin_id, pair_id=pair.pair_id, period=period, side=side,
                                          paired_cells=len(values), **dict(zip(quantile_names, map(float, qs)))))
    if file_no % 50 == 0 or file_no == len(dist_ids):
        print(f"Đã đo phân phối ghép {file_no}/{len(dist_ids)} ID", flush=True)
A1_METEO_PAIR_METRICS = pd.DataFrame(metric_rows)
A1_METEO_PAIR_QUANTILES = pd.DataFrame(quantile_rows)
summary_rows, check_rows = [], []
metric_fields = ["mean_delta_raw", "w1_raw", "mae_raw", "rmse_raw", "pearson_r"]
for (pair_id, period), part in A1_METEO_PAIR_METRICS.groupby(["pair_id", "period"], sort=False):
    n = int(part.paired_cells.sum())
    previous = A1_METEO_PAIR_SCOPED_SUMMARY.loc[(A1_METEO_PAIR_SCOPED_SUMMARY.pair_id.eq(pair_id))
               & A1_METEO_PAIR_SCOPED_SUMMARY.scope_kind.eq("period") & A1_METEO_PAIR_SCOPED_SUMMARY.scope.eq(period)].iloc[0]
    if n != int(previous.paired_cells):
        raise ValueError("Tổng ngày ghép khác 6.1.")
    row = dict(pair_id=pair_id, period=period, basins=len(part), basins_with_pairs=int(part.paired_cells.gt(0).sum()),
               complete_basins=int((part.paired_cells.eq(part.expected_cells) & part.expected_cells.gt(0)).sum()), paired_cells=n)
    for field in metric_fields:
        values = part[field].dropna()
        row[field + "_n_basins"] = len(values)
        for suffix, q in [("p05", .05), ("median", .5), ("p95", .95)]:
            row[field + "_" + suffix] = float(values.quantile(q)) if len(values) else np.nan
    summary_rows.append(row)
    check_rows.append(dict(pair_id=pair_id, period=period, paired_cells=n,
        no_pair_basins=int(part.paired_cells.eq(0).sum()), undefined_r_basins=int(part.pearson_r.isna().sum()),
        w1_dual_implementation_passed=True, inequality_checks_passed=True))
A1_METEO_PAIR_METRIC_SUMMARY = pd.DataFrame(summary_rows)
A1_METEO_PAIR_METRIC_CHECKS = pd.DataFrame(check_rows)
# Mỗi ID có trọng số bằng nhau; đây không phải phân vị của toàn bộ ngày gộp.
A1_METEO_PAIR_QUANTILE_SUMMARY = A1_METEO_PAIR_QUANTILES.groupby(["pair_id", "period", "side"])[quantile_names].median().reset_index()
fig, axes = plt.subplots(len(dist_contract), 2, figsize=(15, 18), squeeze=False)
period_colors = dict(zip(PERIODS, ["#247a99", "#cd7634", "#7a5299"]))
for row_no, pair in enumerate(dist_contract.itertuples()):
    for column, metric in enumerate(["w1_raw", "pearson_r"]):
        ax = axes[row_no, column]
        all_values = A1_METEO_PAIR_METRICS.loc[A1_METEO_PAIR_METRICS.pair_id.eq(pair.pair_id), metric].dropna()
        right_edge = max(float(all_values.max()) * 1.04, 1e-6) if column == 0 and len(all_values) else 1.
        if column == 0 and (not len(all_values) or all_values.max() == 0):
            right_edge = 1.
        for period, color in period_colors.items():
            vals = A1_METEO_PAIR_METRICS.loc[A1_METEO_PAIR_METRICS.pair_id.eq(pair.pair_id)
                    & A1_METEO_PAIR_METRICS.period.eq(period), metric].dropna().sort_values().to_numpy()
            if len(vals):
                # Ties gộp tại cùng x để ECDF nhảy đúng số ID, không nối giả giữa các mức.
                unique, counts = np.unique(vals, return_counts=True)
                cdf = 100 * np.cumsum(counts) / len(vals)
                ax.step(np.r_[0. if column == 0 else -1., unique, right_edge], np.r_[0., cdf, 100.], where="post", color=color,
                        label=f"{period.title()} (n={len(vals)})")
        ax.set_title(pair.label + (" | stored W1" if column == 0 else " | same-date Pearson r"), fontsize=11)
        ax.set_ylabel("Basins at or below value (%)")
        ax.set_ylim(0, 102)
        ax.set_xlabel("W1 (stored units; solar quantity/unit unresolved)" if pair.pair_id == "solar" and column == 0
                      else (f"W1 (stored units; paper reference: {pair.reference_unit_paper})" if column == 0 else "Pearson correlation"))
        ax.set_xlim(0, right_edge) if column == 0 else ax.set_xlim(-1.02, 1.02)
        ax.grid(alpha=.2)
        if len(all_values):
            ax.legend(fontsize=8, loc="lower right")
        else:
            ax.text(.5, .5, "No defined values", transform=ax.transAxes, ha="center")
fig.suptitle("Five stored meteorological pairs: distribution distance and same-date association", fontsize=15)
fig.text(.5, .012, "Each curve gives equal weight to basins with a defined metric; n is shown separately for each period.\n"
         "W1 ignores time order; Pearson r uses the same paired calendar dates. Constant series have undefined r.\n"
         "Finite signed values and zeros are retained. No scaling, unit/sign conversion, date shift or quality filter is applied.\n"
         "Distances describe stored columns, not verified physical forecast errors. Solar quantities/units remain unresolved.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .075, 1, .975))
pair_dist_figure = FIGURE_DIR / "A1_P6_pair_distributions_by_period.png"
fig.savefig(pair_dist_figure, dpi=160, bbox_inches="tight")
plt.show()
for suffix, table in [("pair_metrics_by_basin_period", A1_METEO_PAIR_METRICS), ("pair_quantiles_by_basin_period", A1_METEO_PAIR_QUANTILES),
        ("pair_metric_summary", A1_METEO_PAIR_METRIC_SUMMARY), ("pair_quantile_summary", A1_METEO_PAIR_QUANTILE_SUMMARY),
        ("pair_metric_checks", A1_METEO_PAIR_METRIC_CHECKS)]:
    table.to_csv(TABLE_DIR / f"A1_P6_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None):
    print("(1) Phân vị ngang các chỉ số từng lưu vực; mỗi ID trọng số bằng nhau, đơn vị là raw:")
    display(A1_METEO_PAIR_METRIC_SUMMARY[["pair_id", "period", "basins_with_pairs", "complete_basins", "paired_cells",
        "mean_delta_raw_median", "w1_raw_p05", "w1_raw_median", "w1_raw_p95", "mae_raw_median", "rmse_raw_median",
        "pearson_r_n_basins", "pearson_r_p05", "pearson_r_median", "pearson_r_p95"]])
    print("(2) Median ngang phân vị từng ID; đây không phải phân vị của ngày gộp:")
    display(A1_METEO_PAIR_QUANTILE_SUMMARY)
    print("(3) Kiểm W1 hai cách, bất đẳng thức và tương quan không xác định:")
    display(A1_METEO_PAIR_METRIC_CHECKS)
    print("(4) Tối đa ba ID có W1 raw lớn nhất mỗi cặp trong test; không xếp chéo đơn vị:")
    display(A1_METEO_PAIR_METRICS.loc[A1_METEO_PAIR_METRICS.period.eq("test") & A1_METEO_PAIR_METRICS.w1_raw.notna()]
            .sort_values(["pair_id", "w1_raw", "basin_id"], ascending=[True, False, True]).groupby("pair_id", sort=False).head(3))
print("Đã lưu hình:", pair_dist_figure, "; đã lưu năm CSV vào:", TABLE_DIR)
print(f"Thời gian đo phân phối ghép: {time.perf_counter() - started_pair_dist:.1f} giây.")
print("CẢNH BÁO: net/downward và đơn vị CSV bức xạ chưa xác nhận; không diễn giải khoảng cách raw là bias vật lý hoặc xếp hạng chéo cặp.")

# %% [markdown]
# ### 6.3. Mùa của năm cặp và dấu chênh tại các ca W1 lớn
#
# Tách tháng lịch trong từng kỳ, giữ đúng cặp ngày–ID và kiểm cộng lại khớp 6.2.

# %% Phần 6.3 - Mùa của năm cặp và dấu chênh tại các ca W1 lớn
# Tách tháng lịch trong từng kỳ, giữ đúng cặp ngày–ID và kiểm cộng lại khớp 6.2.
# W1 tháng không được cộng/trung bình thành W1 cả kỳ; thống kê chỉ mô tả hai cột raw.
started_pair_month = time.perf_counter()
import calendar as calendar_labels

month_contract = A1_METEO_PAIR_CONTRACT.copy()
month_calendar = pd.DatetimeIndex(meteo_calendar)
month_ids = sorted(set(map(str, selected_ids)), key=int)
month_fields = sorted(set(month_contract.reference_variable) | set(month_contract.forecast_variable))
month_period_masks = {period: (month_calendar >= pd.Timestamp(start)) & (month_calendar <= pd.Timestamp(end))
                      for period, (start, end) in PERIODS.items()}
month_previous = A1_METEO_PAIR_METRICS.copy()
month_previous["basin_id"] = month_previous.basin_id.astype(str)
month_previous = month_previous.set_index(["basin_id", "pair_id", "period"])
month_rows, month_checks = [], []
for file_no, basin_id in enumerate(month_ids, 1):
    raw = pd.read_csv(A1_DAILY_FILES["A"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                      usecols=DATE_COLS + month_fields, keep_default_na=False, na_filter=False)
    dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
    if dates.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing or not dates.isin(month_calendar).all():
        raise ValueError(f"ID {basin_id}: lịch không khớp 6.1/6.2.")
    positions = month_calendar.get_indexer(dates)
    aligned_fields = {}
    for variable in month_fields:
        values = pd.to_numeric(raw[variable].str.strip(), errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(values) & (values != -999)
        aligned = np.full(len(month_calendar), np.nan)
        aligned[positions[usable]] = values[usable]
        aligned_fields[variable] = aligned
    for pair in month_contract.itertuples():
        reference, forecast = aligned_fields[pair.reference_variable], aligned_fields[pair.forecast_variable]
        for period, period_mask in month_period_masks.items():
            paired = period_mask & np.isfinite(reference) & np.isfinite(forecast)
            full = pair_raw_metrics(reference[paired], forecast[paired])
            prior = month_previous.loc[(basin_id, pair.pair_id, period)]
            compare_fields = ["reference_mean_raw", "forecast_mean_raw", "mean_delta_raw", "mae_raw", "rmse_raw", "w1_raw", "pearson_r"]
            if (full["paired_cells"] != int(prior.paired_cells) or int(period_mask.sum()) != int(prior.expected_cells)
                    or not np.allclose([full[f] for f in compare_fields], [prior[f] for f in compare_fields], rtol=1e-10, atol=1e-10, equal_nan=True)):
                raise ValueError(f"{basin_id}/{pair.pair_id}/{period}: dữ liệu/chỉ số khác 6.2.")
            local_rows = []
            for month in range(1, 13):
                mask = period_mask & (month_calendar.month == month)
                keep = paired & mask
                x, y = reference[keep], forecast[keep]
                delta = y - x
                row = dict(basin_id=basin_id, pair_id=pair.pair_id, period=period, month=month,
                           expected_cells=int(mask.sum()), **pair_raw_metrics(x, y),
                           delta_positive_cells=int((delta > 0).sum()), delta_negative_cells=int((delta < 0).sum()),
                           delta_zero_cells=int((delta == 0).sum()))
                if row["delta_positive_cells"] + row["delta_negative_cells"] + row["delta_zero_cells"] != row["paired_cells"]:
                    raise ValueError("Số ngày theo dấu không cộng khớp.")
                local_rows.append(row)
            part = pd.DataFrame(local_rows)
            n = int(part.paired_cells.sum())
            if n != full["paired_cells"] or int(part.expected_cells.sum()) != int(prior.expected_cells):
                raise ValueError("Tổng tháng không khớp lịch/cặp cả kỳ.")
            # Chỉ mean, MAE và RMSE có phép gộp này; không gộp W1 hoặc Pearson r.
            rolled = {}
            for field in ["reference_mean_raw", "forecast_mean_raw", "mean_delta_raw", "mae_raw"]:
                rolled[field] = float((part[field].fillna(0) * part.paired_cells).sum() / n) if n else np.nan
            rolled["rmse_raw"] = float(np.sqrt((part.rmse_raw.fillna(0).pow(2) * part.paired_cells).sum() / n)) if n else np.nan
            if not np.allclose(list(rolled.values()), [full[f] for f in rolled], rtol=1e-10, atol=1e-10, equal_nan=True):
                raise ValueError("Mean/MAE/RMSE gộp đúng trọng số tháng không khớp 6.2.")
            month_checks.append(dict(basin_id=basin_id, pair_id=pair.pair_id, period=period, paired_cells=n,
                expected_cells=int(prior.expected_cells), empty_months=int(part.paired_cells.eq(0).sum()),
                undefined_r_months=int(part.pearson_r.isna().sum()),
                max_abs_rollup_difference_raw=max(abs(rolled[f] - full[f]) for f in rolled) if n else np.nan,
                matches_6_2=True, monthly_rollup_passed=True, delta_sign_partition_passed=True))
            month_rows.extend(local_rows)
    if file_no % 50 == 0 or file_no == len(month_ids):
        print(f"Đã đo mùa năm cặp {file_no}/{len(month_ids)} ID", flush=True)
A1_METEO_PAIR_MONTHLY = pd.DataFrame(month_rows)
A1_METEO_PAIR_MONTHLY_CHECKS = pd.DataFrame(month_checks)
month_summary_rows = []
for (pair_id, period, month), part in A1_METEO_PAIR_MONTHLY.groupby(["pair_id", "period", "month"], sort=False):
    row = dict(pair_id=pair_id, period=period, month=month, basins=len(part), paired_cells=int(part.paired_cells.sum()),
               expected_cells=int(part.expected_cells.sum()), basins_with_pairs=int(part.paired_cells.gt(0).sum()),
               complete_basins=int((part.paired_cells.eq(part.expected_cells) & part.expected_cells.gt(0)).sum()))
    for field in ["mean_delta_raw", "w1_raw", "mae_raw", "rmse_raw", "pearson_r"]:
        values = part[field].dropna()
        row[field + "_n_basins"] = len(values)
        for suffix, q in [("p05", .05), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95)]:
            row[field + "_" + suffix] = float(values.quantile(q)) if len(values) else np.nan
    for field in ["delta_positive_cells", "delta_negative_cells", "delta_zero_cells"]:
        row[field] = int(part[field].sum())
    month_summary_rows.append(row)
A1_METEO_PAIR_MONTHLY_SUMMARY = pd.DataFrame(month_summary_rows)
# Giữ đúng các ca lớn ở 6.2; lựa chọn test chỉ để mô tả, không chọn quy tắc hay mô hình.
month_cases = A1_METEO_PAIR_METRICS.loc[A1_METEO_PAIR_METRICS.period.eq("test") & A1_METEO_PAIR_METRICS.w1_raw.notna()].copy()
month_cases["basin_id"] = month_cases.basin_id.astype(str)
month_cases = month_cases.sort_values(["pair_id", "w1_raw", "basin_id"], ascending=[True, False, True]).groupby("pair_id", sort=False).head(3)
month_case_signs = A1_METEO_PAIR_MONTHLY.groupby(["basin_id", "pair_id", "period"])[
    ["paired_cells", "delta_positive_cells", "delta_negative_cells", "delta_zero_cells"]].sum().reset_index()
A1_METEO_PAIR_SIGN_CASES = month_cases[["basin_id", "pair_id", "period", "mean_delta_raw", "mae_raw", "w1_raw"]].merge(
    month_case_signs, on=["basin_id", "pair_id", "period"], how="left", validate="one_to_one")
month_check_summary = A1_METEO_PAIR_MONTHLY_CHECKS.groupby(["pair_id", "period"]).agg(
    basins=("basin_id", "nunique"), expected_cells=("expected_cells", "sum"), paired_cells=("paired_cells", "sum"),
    empty_months=("empty_months", "sum"), undefined_r_months=("undefined_r_months", "sum"),
    max_abs_rollup_difference_raw=("max_abs_rollup_difference_raw", "max"), matches_6_2=("matches_6_2", "all"),
    monthly_rollup_passed=("monthly_rollup_passed", "all"), delta_sign_partition_passed=("delta_sign_partition_passed", "all")).reset_index()
fig, axes = plt.subplots(len(month_contract), 2, figsize=(15, 17), squeeze=False)
month_colors = dict(zip(PERIODS, ["#247a99", "#cd7634", "#7a5299"]))
for row_no, pair in enumerate(month_contract.itertuples()):
    for column, field in enumerate(["mean_delta_raw", "w1_raw"]):
        ax = axes[row_no, column]
        for period, color in month_colors.items():
            part = A1_METEO_PAIR_MONTHLY_SUMMARY.loc[A1_METEO_PAIR_MONTHLY_SUMMARY.pair_id.eq(pair.pair_id)
                    & A1_METEO_PAIR_MONTHLY_SUMMARY.period.eq(period)].set_index("month").reindex(range(1, 13))
            ax.plot(range(1, 13), part[field + "_median"], color=color, marker="o", ms=3, label=period.title())
            if period == "test":
                ax.fill_between(range(1, 13), part[field + "_p25"].to_numpy(float), part[field + "_p75"].to_numpy(float),
                                color=color, alpha=.15)
        ax.axhline(0, color=".4", lw=.7, ls="--") if column == 0 else ax.set_ylim(bottom=0)
        ax.set_title(pair.label + (" | median same-date mean difference" if column == 0 else " | median monthly W1"), fontsize=11)
        units = "stored units; solar quantity/unit unresolved" if pair.pair_id == "solar" else f"stored units; paper reference: {pair.reference_unit_paper}"
        ax.set_ylabel(("Forecast minus reference\n" if column == 0 else "W1\n") + "(" + units + ")", fontsize=9)
        ax.set_xticks(range(1, 13), [calendar_labels.month_abbr[m] for m in range(1, 13)], rotation=0)
        ax.set_xlabel("Calendar month (years pooled within each period)")
        ax.set_xlim(.8, 12.2)
        ax.grid(alpha=.2)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(PERIODS), bbox_to_anchor=(.5, .048))
fig.suptitle("Seasonal differences of five stored meteorological pairs", fontsize=16)
fig.text(.5, .009, "Lines: median across basin-specific monthly metrics; each basin has equal weight. Shading: test interquartile range.\n"
         "Each month pools only paired dates from the indicated period. Monthly W1 does not average to full-period W1.\n"
         "Zeros and finite signed values remain; no unit/sign conversion, normalization, date shift or quality filtering.\n"
         "Raw differences do not establish physical forecast bias. Sample counts and undefined correlations are saved separately.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .09, 1, .975))
pair_month_figure = FIGURE_DIR / "A1_P6_pair_monthly_differences.png"
fig.savefig(pair_month_figure, dpi=160, bbox_inches="tight")
plt.show()
for suffix, table in [("pair_monthly_by_basin", A1_METEO_PAIR_MONTHLY), ("pair_monthly_summary", A1_METEO_PAIR_MONTHLY_SUMMARY),
        ("pair_monthly_checks", A1_METEO_PAIR_MONTHLY_CHECKS), ("pair_monthly_check_summary", month_check_summary),
        ("pair_sign_cases", A1_METEO_PAIR_SIGN_CASES)]:
    table.to_csv(TABLE_DIR / f"A1_P6_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None, "display.max_rows", 70):
    print("(1) Cộng tháng theo lịch/cặp và tái tạo mean/MAE/RMSE cả kỳ; không gộp W1 hoặc r:")
    display(month_check_summary)
    print("(2) Median ngang mean chênh của từng ID theo tháng/kỳ, forecast trừ đối chiếu raw:")
    display(A1_METEO_PAIR_MONTHLY_SUMMARY.pivot(index=["pair_id", "month"], columns="period", values="mean_delta_raw_median").reindex(columns=list(PERIODS)))
    print("(3) Test: W1/MAE/r theo tháng; các median không phải chỉ số trên ngày gộp:")
    display(A1_METEO_PAIR_MONTHLY_SUMMARY.loc[A1_METEO_PAIR_MONTHLY_SUMMARY.period.eq("test"),
        ["pair_id", "month", "paired_cells", "basins_with_pairs", "w1_raw_median", "mae_raw_median", "pearson_r_n_basins", "pearson_r_median"]])
    print("(4) Dấu chênh đúng từng ngày ở các ID test có W1 lớn đã chọn tại 6.2:")
    display(A1_METEO_PAIR_SIGN_CASES)
print("Đã lưu hình:", pair_month_figure, "; đã lưu năm CSV vào:", TABLE_DIR)
print(f"Thời gian khảo sát mùa năm cặp: {time.perf_counter() - started_pair_month:.1f} giây.")

# %% [markdown]
# ### 6.4. Tương quan sau trừ mean tháng train, giữ cùng ngày và cùng ID
#
# Mean tháng của mỗi nguồn chỉ lấy các cặp train đã kiểm ở 6.3; không fit lại trên validation/test.

# %% Phần 6.4 - Tương quan sau trừ mean tháng train, giữ cùng ngày và cùng ID
# Mean tháng của mỗi nguồn chỉ lấy các cặp train đã kiểm ở 6.3; không fit lại trên validation/test.
# Phép trừ chỉ phục vụ chẩn đoán tương quan, không đổi CSV, input mô hình hoặc scaler.
started_pair_center = time.perf_counter()


def pair_pearson_only(x, y):
    # Trả cả trạng thái; không biến tương quan không xác định thành 0.
    if len(x) != len(y) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Pearson cần cùng số cặp hữu hạn.")
    if len(x) < 2:
        return np.nan, "no_pairs" if not len(x) else "insufficient_pairs"
    cx, cy = bool(np.all(x == x[0])), bool(np.all(y == y[0]))
    if cx or cy:
        return np.nan, "both_constant" if cx and cy else ("reference_constant" if cx else "forecast_constant")
    a, b = x - x.mean(), y - y.mean()
    return float(np.clip(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)), -1, 1)), "defined"


center_contract = A1_METEO_PAIR_CONTRACT.copy()
center_calendar = pd.DatetimeIndex(meteo_calendar)
center_ids = sorted(set(map(str, selected_ids)), key=int)
center_fields = sorted(set(center_contract.reference_variable) | set(center_contract.forecast_variable))
center_train = A1_METEO_PAIR_MONTHLY.loc[A1_METEO_PAIR_MONTHLY.period.eq("train"),
    ["basin_id", "pair_id", "month", "expected_cells", "paired_cells", "reference_mean_raw", "forecast_mean_raw"]].copy()
center_train["basin_id"] = center_train.basin_id.astype(str)
if center_train.duplicated(["basin_id", "pair_id", "month"]).any():
    raise ValueError("Hồ sơ tháng train bị trùng.")
center_train = center_train.rename(columns={"paired_cells": "train_paired_cells", "expected_cells": "train_expected_cells",
    "reference_mean_raw": "reference_train_month_mean", "forecast_mean_raw": "forecast_train_month_mean"})
A1_METEO_PAIR_TRAIN_CLIMATOLOGY = center_train.copy()
center_train = center_train.set_index(["basin_id", "pair_id", "month"])
center_previous = A1_METEO_PAIR_METRICS.copy()
center_previous["basin_id"] = center_previous.basin_id.astype(str)
center_previous = center_previous.set_index(["basin_id", "pair_id", "period"])
center_period_masks = {period: (center_calendar >= pd.Timestamp(start)) & (center_calendar <= pd.Timestamp(end))
                       for period, (start, end) in PERIODS.items()}
center_month_number = center_calendar.month.to_numpy()
center_rows = []
for file_no, basin_id in enumerate(center_ids, 1):
    raw = pd.read_csv(A1_DAILY_FILES["A"][basin_id], sep=";", dtype=str, encoding="utf-8-sig",
                      usecols=DATE_COLS + center_fields, keep_default_na=False, na_filter=False)
    dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="coerce"))
    if dates.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing or not dates.isin(center_calendar).all():
        raise ValueError(f"ID {basin_id}: lịch không khớp 6.1–6.3.")
    positions = center_calendar.get_indexer(dates)
    aligned_fields = {}
    for variable in center_fields:
        values = pd.to_numeric(raw[variable].str.strip(), errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(values) & (values != -999)
        aligned = np.full(len(center_calendar), np.nan)
        aligned[positions[usable]] = values[usable]
        aligned_fields[variable] = aligned
    for pair in center_contract.itertuples():
        reference, forecast = aligned_fields[pair.reference_variable], aligned_fields[pair.forecast_variable]
        paired_train = center_period_masks["train"] & np.isfinite(reference) & np.isfinite(forecast)
        ref_clim, fc_clim = np.full(13, np.nan), np.full(13, np.nan)
        for month in range(1, 13):
            prior_month = center_train.loc[(basin_id, pair.pair_id, month)]
            keep = paired_train & (center_month_number == month)
            count = int(keep.sum())
            means = [float(reference[keep].mean()), float(forecast[keep].mean())] if count else [np.nan, np.nan]
            if (count != int(prior_month.train_paired_cells)
                    or int((center_period_masks["train"] & (center_month_number == month)).sum()) != int(prior_month.train_expected_cells)
                    or not np.allclose(means, [prior_month.reference_train_month_mean, prior_month.forecast_train_month_mean],
                                       rtol=1e-12, atol=1e-10, equal_nan=True)):
                raise ValueError(f"{basin_id}/{pair.pair_id}/{month}: mean tháng train khác 6.3.")
            if count:
                ref_clim[month], fc_clim[month] = means
        reference_anomaly = reference - ref_clim[center_month_number]
        forecast_anomaly = forecast - fc_clim[center_month_number]
        for period, period_mask in center_period_masks.items():
            paired = period_mask & np.isfinite(reference) & np.isfinite(forecast)
            previous = center_previous.loc[(basin_id, pair.pair_id, period)]
            full = pair_raw_metrics(reference[paired], forecast[paired])
            fields = ["reference_mean_raw", "forecast_mean_raw", "mean_delta_raw", "mae_raw", "rmse_raw", "w1_raw", "pearson_r"]
            if (int(paired.sum()) != int(previous.paired_cells) or int(period_mask.sum()) != int(previous.expected_cells)
                    or not np.allclose([full[f] for f in fields], [previous[f] for f in fields], rtol=1e-10, atol=1e-10, equal_nan=True)):
                raise ValueError(f"{basin_id}/{pair.pair_id}/{period}: dữ liệu/chỉ số khác 6.2.")
            eligible = paired & np.isfinite(reference_anomaly) & np.isfinite(forecast_anomaly)
            raw_r, raw_status = pair_pearson_only(reference[eligible], forecast[eligible])
            anomaly_r, anomaly_status = pair_pearson_only(reference_anomaly[eligible], forecast_anomaly[eligible])
            center_rows.append(dict(basin_id=basin_id, pair_id=pair.pair_id, period=period,
                paired_cells=int(paired.sum()), centered_cells=int(eligible.sum()), excluded_missing_train_month=int((paired & ~eligible).sum()),
                train_months_with_pairs=int(np.isfinite(ref_clim[1:]).sum()), raw_r_all_pairs=full["pearson_r"], raw_r_same_days=raw_r,
                centered_r=anomaly_r, delta_r=anomaly_r - raw_r if np.isfinite(raw_r) and np.isfinite(anomaly_r) else np.nan,
                raw_r_status=raw_status, centered_r_status=anomaly_status, matches_6_2=True, train_means_match_6_3=True))
    if file_no % 50 == 0 or file_no == len(center_ids):
        print(f"Đã kiểm tương quan trừ mùa {file_no}/{len(center_ids)} ID", flush=True)
A1_METEO_PAIR_SEASON_REMOVED = pd.DataFrame(center_rows)
center_summary_rows = []
for (pair_id, period), part in A1_METEO_PAIR_SEASON_REMOVED.groupby(["pair_id", "period"], sort=False):
    both = part.raw_r_same_days.notna() & part.centered_r.notna()
    common = part.loc[both]
    row = dict(pair_id=pair_id, period=period, basins=len(part), paired_cells=int(part.paired_cells.sum()),
        centered_cells=int(part.centered_cells.sum()), excluded_missing_train_month=int(part.excluded_missing_train_month.sum()),
        raw_r_defined_basins=int(part.raw_r_same_days.notna().sum()), centered_r_defined_basins=int(part.centered_r.notna().sum()),
        comparison_basins=int(both.sum()), centered_r_lower_basins=int(common.delta_r.lt(0).sum()),
        centered_r_higher_basins=int(common.delta_r.gt(0).sum()), centered_r_equal_basins=int(common.delta_r.eq(0).sum()))
    for field in ["raw_r_same_days", "centered_r", "delta_r"]:
        for suffix, quantile in [("p05", .05), ("median", .5), ("p95", .95)]:
            row[field + "_" + suffix] = float(common[field].quantile(quantile)) if len(common) else np.nan
    center_summary_rows.append(row)
A1_METEO_PAIR_SEASON_REMOVED_SUMMARY = pd.DataFrame(center_summary_rows)
center_clim_summary = A1_METEO_PAIR_TRAIN_CLIMATOLOGY.groupby("pair_id").agg(
    basins=("basin_id", "nunique"), basin_months=("month", "size"), min_train_pairs_per_month=("train_paired_cells", "min"),
    max_train_pairs_per_month=("train_paired_cells", "max"), missing_train_months=("train_paired_cells", lambda s: int(s.eq(0).sum()))).reset_index()
center_checks = A1_METEO_PAIR_SEASON_REMOVED.groupby(["pair_id", "period"]).agg(
    basins=("basin_id", "nunique"), paired_cells=("paired_cells", "sum"), centered_cells=("centered_cells", "sum"),
    excluded_missing_train_month=("excluded_missing_train_month", "sum"),
    matches_6_2=("matches_6_2", "all"), train_means_match_6_3=("train_means_match_6_3", "all")).reset_index()
center_cases = A1_METEO_PAIR_SEASON_REMOVED.loc[A1_METEO_PAIR_SEASON_REMOVED.period.eq("test") & A1_METEO_PAIR_SEASON_REMOVED.delta_r.notna()].copy()
center_cases["abs_delta_r"] = center_cases.delta_r.abs()
A1_METEO_PAIR_SEASON_REMOVED_CASES = center_cases.sort_values(["pair_id", "abs_delta_r", "basin_id"], ascending=[True, False, True]).groupby("pair_id", sort=False).head(3)
fig, axes = plt.subplots(3, 2, figsize=(15, 12.5))
for ax, pair in zip(axes.flat, center_contract.itertuples()):
    part = A1_METEO_PAIR_SEASON_REMOVED.loc[A1_METEO_PAIR_SEASON_REMOVED.pair_id.eq(pair.pair_id)
           & A1_METEO_PAIR_SEASON_REMOVED.period.eq("test")].dropna(subset=["raw_r_same_days", "centered_r"])
    for field, label, color, linestyle in [("raw_r_same_days", "Stored values", "#247a99", "-"),
            ("centered_r", "After train-month centering", "#7a5299", "--")]:
        values, counts = np.unique(part[field].to_numpy(), return_counts=True)
        if len(values):
            ax.step(np.r_[-1., values, 1.], np.r_[0., 100 * np.cumsum(counts) / counts.sum(), 100.],
                    where="post", color=color, ls=linestyle, label=f"{label} (n={len(part)})")
    all_r = part[["raw_r_same_days", "centered_r"]].to_numpy().ravel()
    ax.set_xlim(max(-1.02, float(all_r.min()) - .03), min(1.02, float(all_r.max()) + .03)) if len(all_r) else ax.set_xlim(-1, 1)
    ax.set(title=pair.label + " | test", xlabel="Same-date Pearson correlation", ylabel="Basins at or below r (%)", ylim=(0, 102))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="upper left") if len(part) else ax.text(.5, .5, "No comparable basins", transform=ax.transAxes, ha="center")
ax = axes.flat[-1]
positions = np.arange(len(center_contract))
for offset, (period, color) in zip([-.24, 0, .24], zip(PERIODS, ["#247a99", "#cd7634", "#7a5299"])):
    medians = A1_METEO_PAIR_SEASON_REMOVED_SUMMARY.loc[A1_METEO_PAIR_SEASON_REMOVED_SUMMARY.period.eq(period)].set_index("pair_id").reindex(center_contract.pair_id)
    ax.barh(positions + offset, medians.delta_r_median, height=.22, color=color, label=period.title())
ax.axvline(0, color=".4", lw=.8)
ax.set_yticks(positions, center_contract.label)
ax.invert_yaxis()
ax.set(title="Median within-basin change in r", xlabel="Centered r minus raw r", ylabel="Meteorological pair")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .073), ncol=len(PERIODS), fontsize=8)
ax.grid(axis="x", alpha=.2)
fig.suptitle("Same-date association after removing train calendar-month means", fontsize=16)
fig.text(.5, .012, "Each source uses its own basin-specific train monthly mean, estimated on paired train dates only.\n"
         "Raw and centered r use the same eligible dates; both curves use exactly the same basins. Constant series keep undefined r.\n"
         "Monthly centering only partly removes seasonality; it does not remove daily climatology, trends or all shared signals.\n"
         "This diagnostic does not change model inputs/scalers or establish independent forecast skill; solar quantity/units remain unresolved.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .12, 1, .97))
center_figure = FIGURE_DIR / "A1_P6_pair_train_month_centered_correlations.png"
fig.savefig(center_figure, dpi=160, bbox_inches="tight")
plt.show()
for suffix, table in [("pair_train_monthly_climatology", A1_METEO_PAIR_TRAIN_CLIMATOLOGY),
        ("pair_season_removed_by_basin_period", A1_METEO_PAIR_SEASON_REMOVED),
        ("pair_season_removed_summary", A1_METEO_PAIR_SEASON_REMOVED_SUMMARY),
        ("pair_season_removed_checks", center_checks), ("pair_season_removed_cases", A1_METEO_PAIR_SEASON_REMOVED_CASES)]:
    table.to_csv(TABLE_DIR / f"A1_P6_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None):
    print("(1) Mean tháng train: số cặp thực dùng, không fit validation/test hoặc điền tháng thiếu:")
    display(center_clim_summary)
    print("(2) r trước/sau trên cùng ngày và cùng ID có cả hai r xác định; delta là hiệu từng ID trước khi lấy median:")
    display(A1_METEO_PAIR_SEASON_REMOVED_SUMMARY)
    print("(3) Khớp nguồn 6.2, mean tháng 6.3 và số ngày bị loại do thiếu mean train:")
    display(center_checks)
    print("(4) Tối đa ba ID mỗi cặp có |delta r| lớn nhất ở test; chỉ đọc hồ sơ, không chọn xử lý/mô hình:")
    display(A1_METEO_PAIR_SEASON_REMOVED_CASES)
print("Đã lưu hình:", center_figure, "; đã lưu năm CSV vào:", TABLE_DIR)
print(f"Thời gian kiểm tương quan trừ mean tháng train: {time.perf_counter() - started_pair_center:.1f} giây.")

# %% [markdown]
# ### 6.5. Ba mốc nâng lưới HRES: hiệu và W1 theo năm, tháng, trước/sau
#
# Cell độc lập: chỉ cần Input `lamah-ce-ext`, chạy được trong phiên CPU mới mà không chạy lại Phần 1–6.4.
# Giữ năm cặp tác giả, thêm mưa/bốc hơi đối chiếu ERA5-Land; chỉ mô tả thay đổi, không hiệu chỉnh dữ liệu.
# Các vạch ngày nâng lưới là mốc đối chiếu, không phải kết quả kiểm định nguyên nhân.

# %% Phần 6.5 - Ba mốc nâng lưới HRES, cùng ngày và cùng lưu vực
# Đọc khí tượng 451 ID một lần; mean/W1 theo năm, tháng và cửa sổ trước/sau, giữ đơn vị lưu trữ.
# Không dùng kết quả test để chọn mô hình/scaler; bức xạ net/downward và dấu ET được ghi riêng.
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from IPython.display import display
from scipy.stats import wasserstein_distance

hres_started = time.perf_counter()
hres_search = Path(os.environ.get("LAMAH_ROOT", "/kaggle/input" if Path("/kaggle/input").exists() else "."))
hres_candidates = [p for p in hres_search.rglob("LamaH_extended") if p.is_dir()]
if hres_search.name == "LamaH_extended":
    hres_candidates.append(hres_search)
if "LAMAH_DIR" in globals() and Path(LAMAH_DIR).is_dir():
    hres_candidates = [Path(LAMAH_DIR)]
hres_candidates = sorted(set(p.resolve() for p in hres_candidates))
if len(hres_candidates) != 1:
    raise ValueError(f"Cần đúng một nguồn LamaH_extended; thấy {hres_candidates}. Kiểm Input hoặc LAMAH_ROOT.")
hres_root = hres_candidates[0]
hres_id_files = sorted(hres_search.rglob("basins_filtered.txt"))
if "EXP_DIR" in globals() and (Path(EXP_DIR) / "basins_filtered.txt").is_file():
    hres_id_files = [Path(EXP_DIR) / "basins_filtered.txt"]
if not hres_id_files:
    raise FileNotFoundError("Thiếu basins_filtered.txt của thí nghiệm BiasCast trong Input.")
hres_id_lists = [tuple(str(int(x)) for x in p.read_text(encoding="utf-8-sig").split()) for p in hres_id_files]
if any(len(x) != 451 or len(set(x)) != 451 for x in hres_id_lists):
    raise ValueError("Danh sách phải có đúng 451 ID không trùng.")
if any(set(x) != set(hres_id_lists[0]) for x in hres_id_lists):
    raise ValueError("Có nhiều danh sách 451 ID khác nhau; không tự chọn nguồn.")
hres_ids = sorted(hres_id_lists[0], key=int)
hres_daily = hres_root / "A_basins_total_upstrm" / "2_timeseries" / "daily"
hres_paths = {}
for hres_path in sorted(hres_daily.glob("ID_*.csv")):
    hres_match = re.fullmatch(r"ID_(\d+)\.csv", hres_path.name)
    if hres_match:
        hres_id = str(int(hres_match.group(1)))
        if hres_id in hres_paths:
            raise ValueError(f"Trùng tệp khí tượng ID {hres_id}.")
        hres_paths[hres_id] = hres_path
if set(hres_ids) - set(hres_paths):
    raise FileNotFoundError(f"Thiếu tệp khí tượng: {sorted(set(hres_ids) - set(hres_paths), key=int)}")
hres_work = Path(globals().get("WORK_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "explore_output"))
hres_tables = hres_work / "Tables"
hres_figures = hres_work / "Figures"
hres_tables.mkdir(parents=True, exist_ok=True)
hres_figures.mkdir(parents=True, exist_ok=True)
hres_contract = pd.DataFrame([
    ("temperature", "ERA5L_2m_temp_mean", "ECMWF_t2m", "Temperature", "degC", "ERA5-Land", "comparable_stored_columns"),
    ("dewpoint", "ERA5L_2m_dp_temp_mean", "ECMWF_d2m", "Dew point", "degC", "ERA5-Land", "comparable_stored_columns"),
    ("precipitation_era5l", "ERA5L_prec", "ECMWF_tp", "Precipitation", "mm/day", "ERA5-Land", "accumulation_convention_to_trace"),
    ("evaporation_era5l", "ERA5L_total_et", "ECMWF_e", "Evaporation", "stored mm/day", "ERA5-Land", "sign_components_to_trace_no_flip"),
    ("solar", "ERA5L_surf_net_solar_rad_mean", "ECMWF_ssrd", "Solar: net vs downward", "stored raw units", "ERA5-Land", "different_quantities_not_physical_bias"),
    ("precipitation", "MSWEP_RR", "ECMWF_tp", "Precipitation vs MSWEP", "mm/day", "MSWEP", "author_pair"),
    ("evaporation", "GLEAM_ETA", "ECMWF_e", "Evaporation vs GLEAM", "stored mm/day", "GLEAM", "author_pair_sign_components_to_trace"),
], columns=["pair_id", "reference_variable", "forecast_variable", "label", "unit", "reference_source", "interpretation"])
hres_events = pd.DataFrame([
    ("2006", "2006-02-01", "40_to_25_km"),
    ("2010", "2010-01-26", "25_to_16_km"),
    ("2016", "2016-03-08", "16_to_9_km"),
], columns=["event", "cut_date", "grid_change"])
hres_years = list(range(2002, 2018))
hres_months = sorted(set(m for cut in hres_events.cut_date for m in pd.period_range(
    pd.Timestamp(cut).to_period("M") - 12, pd.Timestamp(cut).to_period("M") + 11, freq="M")))


def a1_hres_metrics(reference, forecast, expected):
    valid = np.isfinite(reference) & np.isfinite(forecast)
    x = reference[valid]
    y = forecast[valid]
    n = len(x)
    result = {"expected_days": int(expected), "paired_days": n,
              "reference_mean": np.nan, "forecast_mean": np.nan, "mean_delta": np.nan,
              "w1": np.nan, "mae": np.nan, "reference_negative_days": int((x < 0).sum()),
              "forecast_negative_days": int((y < 0).sum())}
    if n:
        result.update(reference_mean=float(x.mean()), forecast_mean=float(y.mean()),
                      mean_delta=float((y - x).mean()), w1=float(np.abs(np.sort(y) - np.sort(x)).mean()),
                      mae=float(np.abs(y - x).mean()))
        tol = 1e-10 * (1 + max(abs(result["mean_delta"]), result["w1"], result["mae"]))
        if abs(result["mean_delta"]) > result["w1"] + tol or result["w1"] > result["mae"] + tol:
            raise AssertionError("Không thỏa |mean delta| <= W1 <= MAE.")
    return result


# Kiểm công thức W1 khi hai dãy có cùng số cặp; không dùng kiểm giả lập thay kết quả dataset.
for hres_x, hres_y in [([0., 1., 4.], [1., 2., 7.]), ([-2., 0., 3.], [3., -2., 0.])]:
    hres_probe = a1_hres_metrics(np.array(hres_x), np.array(hres_y), 3)
    if not np.isclose(hres_probe["w1"], wasserstein_distance(hres_x, hres_y), rtol=1e-12, atol=1e-12):
        raise AssertionError("W1 không khớp SciPy.")

hres_annual_rows = []
hres_monthly_rows = []
hres_window_rows = []
hres_inputs = []
hres_fields = sorted(set(hres_contract.reference_variable) | set(hres_contract.forecast_variable))
print("Phần 6.5 - HRES: chỉ đọc khí tượng, không huấn luyện; nguồn:", hres_root)
for hres_no, hres_id in enumerate(hres_ids, 1):
    hres_path = hres_paths[hres_id]
    hres_raw = pd.read_csv(hres_path, sep=";", dtype=str, encoding="utf-8-sig",
                           usecols=["YYYY", "MM", "DD"] + hres_fields, keep_default_na=False, na_filter=False)
    hres_dates = pd.DatetimeIndex(pd.to_datetime(hres_raw[["YYYY", "MM", "DD"]].rename(
        columns={"YYYY": "year", "MM": "month", "DD": "day"}), errors="raise"))
    if hres_dates.duplicated().any() or not hres_dates.is_monotonic_increasing:
        raise ValueError(f"ID {hres_id}: lịch trùng hoặc không tăng.")
    hres_keep = (hres_dates >= "2002-01-01") & (hres_dates <= "2017-12-31")
    hres_dates = hres_dates[hres_keep]
    hres_values = {}
    for field in hres_fields:
        tokens = hres_raw.loc[hres_keep, field].str.strip()
        numbers = pd.to_numeric(tokens, errors="coerce").to_numpy(dtype=float)
        if (tokens.ne("").to_numpy() & np.isnan(numbers)).any() or np.isinf(numbers).any():
            raise ValueError(f"ID {hres_id}/{field}: token lạ hoặc vô cực; kiểm trước khi tính.")
        hres_values[field] = np.where(numbers == -999, np.nan, numbers)
    hres_year_idx = {year: np.flatnonzero(hres_dates.year == year) for year in hres_years}
    hres_month_idx = {str(month): np.flatnonzero((hres_dates.year == month.year) & (hres_dates.month == month.month))
                      for month in hres_months}
    for pair in hres_contract.itertuples(index=False):
        reference = hres_values[pair.reference_variable]
        forecast = hres_values[pair.forecast_variable]
        for year, idx in hres_year_idx.items():
            expected = len(pd.date_range(f"{year}-01-01", f"{year}-12-31"))
            hres_annual_rows.append({"basin_id": hres_id, "pair_id": pair.pair_id, "year": year,
                                    **a1_hres_metrics(reference[idx], forecast[idx], expected)})
        for month in hres_months:
            idx = hres_month_idx[str(month)]
            hres_monthly_rows.append({"basin_id": hres_id, "pair_id": pair.pair_id, "month": str(month),
                                     **a1_hres_metrics(reference[idx], forecast[idx], month.days_in_month)})
        # Ngày chuyển đổi không được xem là kỳ mới chắc chắn của CSV: đối chiếu thêm dịch mốc +/-1 ngày.
        for event in hres_events.itertuples(index=False):
            for days, shift, year_shift in [(90, 0, -1), (90, 0, 0), (90, 0, 1), (365, 0, 0),
                                            (90, -1, 0), (90, 1, 0)]:
                cut = pd.Timestamp(event.cut_date) + pd.DateOffset(years=year_shift) + pd.Timedelta(days=shift)
                for side, left, right in [("before", cut - pd.Timedelta(days=days), cut),
                                           ("after", cut, cut + pd.Timedelta(days=days))]:
                    idx = np.flatnonzero((hres_dates >= left) & (hres_dates < right))
                    hres_window_rows.append({"basin_id": hres_id, "pair_id": pair.pair_id,
                        "event": event.event, "window_days": days, "cut_shift_days": shift,
                        "control_year_shift": year_shift, "cut_date": str(cut.date()), "side": side,
                        "start_date": str(left.date()), "end_date": str((right - pd.Timedelta(days=1)).date()),
                        **a1_hres_metrics(reference[idx], forecast[idx], days)})
    hres_inputs.append({"basin_id": hres_id, "path": str(hres_path), "bytes": hres_path.stat().st_size,
                        "sha256": hashlib.sha256(hres_path.read_bytes()).hexdigest()})
    if hres_no % 50 == 0 or hres_no == len(hres_ids):
        print(f"Đã đọc {hres_no}/{len(hres_ids)} ID; {time.perf_counter() - hres_started:.1f}s", flush=True)
A1_HRES_ANNUAL = pd.DataFrame(hres_annual_rows)
A1_HRES_MONTHLY = pd.DataFrame(hres_monthly_rows)
A1_HRES_WINDOWS = pd.DataFrame(hres_window_rows)


def a1_hres_summarize(frame, keys):
    rows = []
    for key, part in frame.groupby(keys, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        n = part.paired_days.to_numpy(dtype=float)
        row = dict(zip(keys, key))
        row.update(IDs=len(part), expected_day_IDs=int(part.expected_days.sum()), paired_day_IDs=int(n.sum()),
                   IDs_with_pairs=int((n > 0).sum()))
        for metric in ["reference_mean", "forecast_mean", "mean_delta", "w1"]:
            vals = part[metric].to_numpy(dtype=float)
            good = np.isfinite(vals) & (n > 0)
            row[metric + "_median"] = float(np.median(vals[good])) if good.any() else np.nan
            if metric != "w1":
                row[metric + "_pooled"] = float(np.average(vals[good], weights=n[good])) if good.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


A1_HRES_ANNUAL_SUMMARY = a1_hres_summarize(A1_HRES_ANNUAL, ["pair_id", "year"])
A1_HRES_MONTHLY_SUMMARY = a1_hres_summarize(A1_HRES_MONTHLY, ["pair_id", "month"])
hres_window_keys = ["basin_id", "pair_id", "event", "window_days", "cut_shift_days", "control_year_shift"]
hres_before = A1_HRES_WINDOWS.loc[A1_HRES_WINDOWS.side.eq("before")]
hres_after = A1_HRES_WINDOWS.loc[A1_HRES_WINDOWS.side.eq("after")]
hres_changes = hres_before.merge(hres_after, on=hres_window_keys, suffixes=("_before", "_after"), validate="one_to_one")
hres_changes["complete_both_windows"] = hres_changes.paired_days_before.eq(hres_changes.expected_days_before) & hres_changes.paired_days_after.eq(hres_changes.expected_days_after)
for metric in ["mean_delta", "w1"]:
    hres_changes[metric + "_change"] = (hres_changes[metric + "_after"] - hres_changes[metric + "_before"]).where(hres_changes.complete_both_windows)
hres_change_rows = []
for key, part in hres_changes.groupby(hres_window_keys[1:], sort=True):
    row = dict(zip(hres_window_keys[1:], key))
    row.update(IDs=len(part), complete_IDs=int(part.complete_both_windows.sum()))
    for metric in ["mean_delta", "w1"]:
        for suffix in ["before", "after", "change"]:
            vals = part.loc[part.complete_both_windows, metric + "_" + suffix].dropna()
            row[metric + "_" + suffix + "_median"] = float(vals.median()) if len(vals) else np.nan
        vals = part[metric + "_change"].dropna()
        row[metric + "_change_p05"] = float(vals.quantile(.05)) if len(vals) else np.nan
        row[metric + "_change_p95"] = float(vals.quantile(.95)) if len(vals) else np.nan
        row[metric + "_positive_change_IDs"] = int(vals.gt(0).sum())
    hres_change_rows.append(row)
A1_HRES_CHANGE_SUMMARY = pd.DataFrame(hres_change_rows)
hres_checks = {"IDs": len(hres_ids), "annual_rows": len(A1_HRES_ANNUAL),
    "monthly_rows": len(A1_HRES_MONTHLY), "window_rows": len(A1_HRES_WINDOWS),
    "annual_rows_expected": 451 * 7 * 16, "W1_scipy_fixture_matched": True,
    "annual_groups_all_451": bool(A1_HRES_ANNUAL_SUMMARY.IDs.eq(451).all()),
    "annual_complete_all_pairs": bool(A1_HRES_ANNUAL.paired_days.eq(A1_HRES_ANNUAL.expected_days).all()),
    "computed_on_dataset": True, "causal_upgrade_effect_established": False}
if len(A1_HRES_ANNUAL) != hres_checks["annual_rows_expected"] or not hres_checks["annual_groups_all_451"]:
    raise AssertionError("Sai số hàng/ID của bảng năm.")
# Nếu phiên đã có 6.1, kiểm năm cặp cũ so cùng nguồn; phiên mới không cần chạy lại 6.1.
if "A1_METEO_PAIR_SCOPED_SUMMARY" in globals():
    old = A1_METEO_PAIR_SCOPED_SUMMARY.loc[A1_METEO_PAIR_SCOPED_SUMMARY.scope_kind.eq("year")].copy()
    old["year"] = old.scope.astype(int)
    old = old.loc[old.year.between(2002, 2017)]
    same = old.merge(A1_HRES_ANNUAL_SUMMARY, on=["pair_id", "year"], validate="one_to_one")
    if len(same) != 80 or not same.paired_cells.eq(same.paired_day_IDs).all():
        raise AssertionError("Nguồn năm cặp cũ không khớp 6.1.")
    for old_col, new_col in [("reference_mean_raw", "reference_mean_pooled"), ("forecast_mean_raw", "forecast_mean_pooled")]:
        if not np.allclose(same[old_col], same[new_col], rtol=1e-10, atol=1e-10, equal_nan=True):
            raise AssertionError("Mean năm khác 6.1.")
    hres_checks["old_part6_1"] = "matched"
else:
    hres_checks["old_part6_1"] = "not_run_in_this_session"
for stem, frame in [("contract", hres_contract), ("events", hres_events), ("annual_by_basin", A1_HRES_ANNUAL),
    ("annual_summary", A1_HRES_ANNUAL_SUMMARY), ("monthly_by_basin", A1_HRES_MONTHLY),
    ("monthly_summary", A1_HRES_MONTHLY_SUMMARY), ("windows_by_basin", A1_HRES_WINDOWS),
    ("changes_by_basin", hres_changes), ("change_summary", A1_HRES_CHANGE_SUMMARY),
    ("input_manifest", pd.DataFrame(hres_inputs))]:
    frame.to_csv(hres_tables / f"A1_P6_HRES_{stem}.csv", index=False, encoding="utf-8-sig")
hres_plot_pairs = hres_contract.iloc[:5]
for resolution, data, time_key in [("annual", A1_HRES_ANNUAL_SUMMARY, "year"), ("monthly", A1_HRES_MONTHLY_SUMMARY, "month")]:
    fig, axes = plt.subplots(5, 2, figsize=(15, 18))
    for index, pair in enumerate(hres_plot_pairs.itertuples(index=False)):
        part = data.loc[data.pair_id.eq(pair.pair_id)].copy()
        if resolution == "monthly":
            # Các khoảng tháng quanh ba mốc không được nối qua đoạn chưa tính.
            full = pd.period_range(min(hres_months), max(hres_months), freq="M").astype(str)
            part = part.set_index("month").reindex(full).rename_axis("month").reset_index()
        x = pd.to_datetime(part[time_key].astype(str) + ("-07-01" if resolution == "annual" else "-01"))
        for col, metric, title in [(0, "mean_delta_pooled", "Mean HRES minus reference"), (1, "w1_median", "Median per-basin W1")]:
            ax = axes[index, col]
            ax.plot(x, part[metric], marker="o", ms=2, color="#247a99")
            if col == 0:
                ax.axhline(0, color="grey", lw=.8)
            for event in hres_events.itertuples(index=False):
                ax.axvline(pd.Timestamp(event.cut_date), color="#cd7634", ls="--", lw=.8)
            ax.set(title=f"({chr(97 + 2 * index + col)}) {pair.label} | {title}", ylabel=pair.unit, xlabel="Calendar date")
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            if col == 1:
                ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
    fig.suptitle(f"HRES vs ERA5-Land | {resolution} | common dates per basin", fontsize=15)
    fig.text(.5, .015, "Dashed lines: documented operational upgrades, not detected causal breaks.\n"
             "Solar net vs downward: stored-value diagnostic only. Evaporation: signs/components not harmonized.\n"
             "Mean: pooled paired basin-days. W1: median of per-basin distances; never averaged from monthly W1.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .97))
    fig.savefig(hres_figures / f"A1_P6_HRES_{resolution}_changes.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Hình: hiệu HRES trừ ERA5-Land và W1 theo {resolution}, {len(hres_ids)} lưu vực. "
          "Vạch đứt là ngày nâng lưới đã công bố; W1 là trung vị chỉ số từng lưu vực, không phải W1 gộp. "
          "Bức xạ chỉ so giá trị lưu trữ; dấu/thành phần ET chưa đồng nhất.")
hres_checks["basin_list_sha256"] = hashlib.sha256(hres_id_files[0].read_bytes()).hexdigest()
hres_checks["python"] = platform.python_version()
hres_checks["numpy"] = np.__version__
hres_checks["pandas"] = pd.__version__
hres_checks["scipy"] = scipy.__version__
hres_checks["seconds"] = time.perf_counter() - hres_started
(hres_tables / "A1_P6_HRES_checks.json").write_text(json.dumps(hres_checks, ensure_ascii=False, indent=2), encoding="utf-8")
print("6.5.1. Các cặp đối chiếu và ba mốc:")
display(hres_contract)
display(hres_events)
print("6.5.2. Năm liền kề mỗi mốc; W1 median ngang 451 ID, mean gộp ngày-ID:")
display(A1_HRES_ANNUAL_SUMMARY.loc[A1_HRES_ANNUAL_SUMMARY.year.isin([2005, 2006, 2007, 2009, 2010, 2011, 2015, 2016, 2017])])
print("6.5.3. Hiệu sau-trước theo từng ID rồi lấy median; chỉ ID đủ cả hai cửa sổ:")
display(A1_HRES_CHANGE_SUMMARY)
print("6.5.4. Kiểm dữ liệu:", json.dumps(hres_checks, ensure_ascii=False))
print("Không có nhãn 'có/không bước nhảy' tự động hoặc kết luận do nâng lưới từ một hiệu trước-sau.")
print("GỬI LẠI: annual_summary, monthly_summary, change_summary, checks.json và hai PNG; kèm warning/traceback.")
print("Output trực tiếp:", hres_tables, "và", hres_figures, "- dùng Kaggle Save Version, không đóng ZIP.")


# %% [markdown]
# ## Phần 7. Hợp đồng cửa sổ và số mẫu trên CSV
#
# Đếm theo cấu hình và ô từng pha sử dụng; phân biệt CSV hiện tại với cache và số mẫu huấn luyện lịch sử.

# %% [markdown]
# ### 7.1. Hợp đồng cửa sổ, biến từng pha và lịch khởi động trước khi đếm mẫu
#
# Đọc cấu hình thật, phân biệt cột bộ lọc gốc kiểm cả khung với ô mô hình dùng từng pha.

# %% Phần 7.1 - Hợp đồng cửa sổ, biến từng pha và lịch khởi động trước khi đếm mẫu
# Đọc cấu hình thật, phân biệt cột bộ lọc gốc kiểm cả khung với ô mô hình dùng từng pha.
# Đọc giá trị path như Config._parse_config: chỉ YAML null/chuỗi "None" thành None; path thật vẫn phải kiểm.
# 365/730 ở đây là tổng seq_length, gồm cả pha forecast; chưa đếm hay thay bộ lọc mẫu.
import json
import hashlib


SAMPLE_EXTERNAL_PATH_KEYS = ["additional_feature_files", "train_data_file", "per_basin_train_periods_file",
                             "per_basin_validation_periods_file", "per_basin_test_periods_file"]


def sample_normalize_paths(raw_cfg):
    # Cùng nhánh path của NeuralHydrology Config._parse_config; không đổi các chuỗi khác.
    parsed = dict(raw_cfg)
    for key, value in raw_cfg.items():
        if key.endswith(("_dir", "_path", "_file", "_files")):
            if value is None or value == "None":
                parsed[key] = None
            elif isinstance(value, list):
                parsed[key] = [Path(item) for item in value]
            else:
                parsed[key] = Path(value)
    return parsed


def sample_input_names(value, field):
    # Giữ thứ tự tên; cấu trúc dict theo tần suất được lưu nhưng chưa đưa vào mô phỏng ngày.
    if value is None:
        return []
    if isinstance(value, dict):
        return [name for values in value.values() for name in sample_input_names(values, field)]
    if isinstance(value, str):
        if not value or value != value.strip():
            raise ValueError(f"{field}: tên rỗng hoặc có khoảng trắng thừa.")
        return [value]
    if isinstance(value, list):
        return [name for item in value for name in sample_input_names(item, field)]
    raise ValueError(f"{field}: cấu trúc không hỗ trợ {type(value).__name__}.")


def sample_date(value):
    if isinstance(value, (list, dict)) or value is None:
        raise ValueError("Cần một mốc ngày, không phải nhiều lát kỳ.")
    if isinstance(value, str) and "/" in value:
        return pd.to_datetime(value, format="%d/%m/%Y")
    return pd.Timestamp(value)


def sample_contract(cfg):
    cfg = sample_normalize_paths(cfg)
    names = {key: sample_input_names(cfg.get(key, []), key) for key in ["dynamic_inputs", "hindcast_inputs", "forecast_inputs",
        "mass_inputs", "dynamic_conceptual_inputs", "autoregressive_inputs", "target_variables", "static_attributes"]}
    evolving = sample_input_names(cfg.get("evolving_attributes", cfg.get("static_inputs", [])), "evolving_attributes")
    dynamic = list(dict.fromkeys(names["mass_inputs"] + names["dynamic_inputs"] + names["dynamic_conceptual_inputs"]))
    issues = []
    if not isinstance(cfg.get("dataset"), str) or not cfg["dataset"].startswith("lamah"):
        issues.append("dataset_to_review")
    for field in ["dynamic_inputs", "hindcast_inputs", "forecast_inputs"]:
        spec = cfg.get(field, [])
        if not isinstance(spec, (list, dict)):
            issues.append(field + "_structure_to_review")
        elif isinstance(spec, list) and any(not isinstance(item, (str, list)) or (isinstance(item, list) and any(not isinstance(name, str) for name in item)) for item in spec):
            issues.append(field + "_nested_structure_to_review")
    if set(dynamic) & set(names["target_variables"]):
        issues.append("target_also_dynamic_to_review")
    for key in ["seq_length", "predict_last_n"]:
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            issues.append(key + "_not_positive_scalar")
    if cfg.get("predict_last_n") != 1:
        issues.append("predict_last_n_not_1")
    if not dynamic or not names["target_variables"]:
        issues.append("missing_dynamic_or_target")
    if isinstance(cfg.get("dynamic_inputs"), dict) or cfg.get("use_frequencies"):
        issues.append("frequency_mapping_to_review")
    for key in ["lagged_features"] + SAMPLE_EXTERNAL_PATH_KEYS:
        if cfg.get(key):
            issues.append(key + "_to_review")
    if evolving or names["autoregressive_inputs"] or names["dynamic_conceptual_inputs"]:
        issues.append("evolving_AR_or_conceptual_inputs_to_review")
    if any(isinstance(cfg.get(key), dict) for key in ["hindcast_inputs", "forecast_inputs"]):
        issues.append("phase_frequency_mapping_to_review")
    two_phase = bool(names["hindcast_inputs"])
    if names["forecast_inputs"] and not two_phase:
        issues.append("forecast_without_hindcast")
    forecast_length = cfg.get("forecast_seq_length")
    overlap = cfg.get("forecast_overlap")
    overlap = 0 if overlap is None else overlap
    if two_phase:
        if type(forecast_length) is not int or forecast_length < 1:
            issues.append("forecast_length_not_positive_scalar")
        elif type(cfg.get("seq_length")) is int and forecast_length >= cfg["seq_length"]:
            issues.append("forecast_not_shorter_than_sequence")
        if type(overlap) is not int or overlap < 0 or (type(forecast_length) is int and overlap > forecast_length):
            issues.append("forecast_overlap_to_review")
        if (set(names["hindcast_inputs"]) | set(names["forecast_inputs"])) - set(dynamic):
            issues.append("phase_fields_outside_dynamic")
    periods = {}
    for period in PERIODS:
        try:
            start, end = sample_date(cfg.get(period + "_start_date")), sample_date(cfg.get(period + "_end_date"))
            if pd.isna(start) or pd.isna(end) or start > end:
                raise ValueError("Kỳ không hợp lệ.")
            periods[period] = (start, end)
            if (start, end) != tuple(map(pd.Timestamp, PERIODS[period])):
                issues.append(period + "_differs_from_A1_period")
        except (ValueError, TypeError):
            issues.append(period + "_dates_to_review")
    issues = list(dict.fromkeys(issues))
    source_issues = [key + "_to_review" for key in SAMPLE_EXTERNAL_PATH_KEYS if cfg.get(key)]
    geometry_issues = [reason for reason in issues if reason not in source_issues]
    return dict(names=names, dynamic=dynamic, evolving=evolving, two_phase=two_phase,
                forecast_length=forecast_length, overlap=overlap, periods=periods, issues=issues,
                source_issues=source_issues, geometry_issues=geometry_issues)


sample_configs = sorted(path for path in EXP_DIR.rglob("config.yml") if not a1_is_metadata(path))
if not sample_configs:
    raise FileNotFoundError("Không thấy config.yml để kiểm hợp đồng mẫu.")
sample_run_rows, sample_groups, sample_offset_rows, sample_calendar_rows, sample_path_rows = [], {}, [], [], []
for path in sample_configs:
    raw_cfg = a1_load_yaml(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"{path}: YAML không phải mapping.")
    cfg = sample_normalize_paths(raw_cfg)
    contract = sample_contract(cfg)
    contract_keys = ["dataset", "seq_length", "predict_last_n", "forecast_seq_length", "forecast_overlap", "dynamic_inputs",
        "hindcast_inputs", "forecast_inputs", "mass_inputs", "dynamic_conceptual_inputs", "evolving_attributes", "static_inputs",
        "autoregressive_inputs", "lagged_features", "target_variables", "static_attributes", "use_frequencies", "nan_handling_method",
        "additional_feature_files", "train_data_file", "per_basin_train_periods_file", "per_basin_validation_periods_file", "per_basin_test_periods_file"]
    contract_keys += [period + suffix for period in PERIODS for suffix in ["_start_date", "_end_date"]]
    signature = json.dumps({key: cfg.get(key) for key in contract_keys}, sort_keys=True, ensure_ascii=False, default=str)
    contract_id = "W" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    raw_signature = json.dumps({key: raw_cfg.get(key) for key in contract_keys}, sort_keys=True, ensure_ascii=False, default=str)
    raw_contract_id = "W" + hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()[:12]
    run = path.relative_to(EXP_DIR).as_posix()
    status = "ready_for_daily_count" if not contract["issues"] else "review_required"
    geometry_status = "geometry_ready" if not contract["geometry_issues"] else "geometry_review_required"
    for key in SAMPLE_EXTERNAL_PATH_KEYS:
        sample_path_rows.append(dict(run=run, field=key, raw_present=key in raw_cfg,
            raw_type=type(raw_cfg[key]).__name__ if key in raw_cfg else "missing",
            raw_repr=repr(raw_cfg[key]) if key in raw_cfg else "<not_set>",
            parsed_type=type(cfg.get(key)).__name__, parsed_repr=repr(cfg.get(key)), active_after_parse=bool(cfg.get(key))))
    sample_run_rows.append(dict(run=run, raw_contract_id=raw_contract_id, contract_id=contract_id, model=cfg.get("model"), dataset=cfg.get("dataset"),
        seq_length=cfg.get("seq_length"), predict_last_n=cfg.get("predict_last_n"), forecast_seq_length=contract["forecast_length"],
        forecast_overlap=contract["overlap"], dynamic_columns=len(contract["dynamic"]),
        hindcast_columns=len(set(contract["names"]["hindcast_inputs"])), forecast_columns=len(set(contract["names"]["forecast_inputs"])),
        qmean_in_dynamic="qmean" in contract["dynamic"], nan_handling_method=cfg.get("nan_handling_method"),
        status=status, geometry_status=geometry_status, source_review_reasons=" | ".join(contract["source_issues"]),
        review_reasons=" | ".join(contract["issues"]), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if contract_id in sample_groups:
        if sample_groups[contract_id]["contract_json"] != signature:
            raise ValueError("Trùng mã hợp đồng nhưng khác nội dung JSON.")
        sample_groups[contract_id]["configs"] += 1
        continue
    sample_groups[contract_id] = dict(contract_id=contract_id, configs=1, example_config=run, status=status, geometry_status=geometry_status,
        source_review_reasons=" | ".join(contract["source_issues"]),
        review_reasons=" | ".join(contract["issues"]), contract_json=signature,
        dynamic_names=" | ".join(contract["dynamic"]), hindcast_names=" | ".join(contract["names"]["hindcast_inputs"]),
        forecast_names=" | ".join(contract["names"]["forecast_inputs"]), target_names=" | ".join(contract["names"]["target_variables"]))
    if contract["geometry_issues"]:
        continue
    for length in sorted({cfg["seq_length"], 365, 730}):
        forecast_length = contract["forecast_length"] if contract["two_phase"] else 0
        if length <= forecast_length:
            continue
        for variable in contract["dynamic"]:
            hindcast = contract["two_phase"] and variable in contract["names"]["hindcast_inputs"]
            forecast = contract["two_phase"] and variable in contract["names"]["forecast_inputs"]
            used_offsets = set(range(1 - length, 1)) if not contract["two_phase"] else set()
            if hindcast:
                used_offsets.update(range(1 - length, 1 - forecast_length + contract["overlap"]))
            if forecast:
                used_offsets.update(range(1 - forecast_length, 1))
            role = "full_sequence" if not contract["two_phase"] else "both_phases" if hindcast and forecast else "hindcast_only" if hindcast else "forecast_only" if forecast else "unused_dynamic"
            sample_offset_rows.append(dict(contract_id=contract_id, source_review_required=bool(contract["source_issues"]), sequence_total_days=length, variable=variable, role=role,
                original_filter_steps=length, model_consumed_steps=len(used_offsets), extra_original_filter_steps=length-len(used_offsets),
                original_offset_first=1-length, original_offset_last=0,
                hindcast_offset_first=1-length if hindcast else None,
                hindcast_offset_last=-forecast_length+contract["overlap"] if hindcast else None,
                forecast_offset_first=1-forecast_length if forecast else None, forecast_offset_last=0 if forecast else None))
        for period, (start, end) in contract["periods"].items():
            sample_calendar_rows.append(dict(contract_id=contract_id, source_review_required=bool(contract["source_issues"]), period=period, sequence_total_days=length,
                forecast_steps=forecast_length, hindcast_steps=length-forecast_length+contract["overlap"] if contract["two_phase"] else None,
                predict_last_n=cfg["predict_last_n"], warmup_calendar_days=length-cfg["predict_last_n"],
                warmup_start=start-pd.Timedelta(days=length-cfg["predict_last_n"]), target_start=start, target_end=end,
                expected_end_dates_per_basin=len(pd.date_range(start,end)),
                expected_end_dates_selected=len(pd.date_range(start,end))*len(set(map(str,selected_ids)))))
A1_SAMPLE_PATH_VALUES = pd.DataFrame(sample_path_rows)
A1_SAMPLE_CONTRACT_RUNS = pd.DataFrame(sample_run_rows)
A1_SAMPLE_CONTRACT_GROUPS = pd.DataFrame(list(sample_groups.values()))
offset_fields = ["contract_id","source_review_required","sequence_total_days","variable","role","original_filter_steps","model_consumed_steps","extra_original_filter_steps",
                 "original_offset_first","original_offset_last","hindcast_offset_first","hindcast_offset_last","forecast_offset_first","forecast_offset_last"]
calendar_fields = ["contract_id","source_review_required","period","sequence_total_days","forecast_steps","hindcast_steps","predict_last_n","warmup_calendar_days",
                   "warmup_start","target_start","target_end","expected_end_dates_per_basin","expected_end_dates_selected"]
A1_SAMPLE_CONTRACT_OFFSETS = pd.DataFrame(sample_offset_rows, columns=offset_fields)
A1_SAMPLE_CALENDAR_SCENARIOS = pd.DataFrame(sample_calendar_rows, columns=calendar_fields)
A1_SAMPLE_CONTRACT_ROLES = A1_SAMPLE_CONTRACT_OFFSETS.groupby(["contract_id","source_review_required","sequence_total_days","role","original_filter_steps",
    "model_consumed_steps","extra_original_filter_steps"], dropna=False).agg(variables=("variable","size"), names=("variable", lambda s:" | ".join(s))).reset_index()
for suffix, table in [("sample_contract_runs", A1_SAMPLE_CONTRACT_RUNS), ("sample_contract_groups", A1_SAMPLE_CONTRACT_GROUPS),
        ("sample_contract_offsets", A1_SAMPLE_CONTRACT_OFFSETS), ("sample_contract_roles", A1_SAMPLE_CONTRACT_ROLES),
        ("sample_calendar_scenarios", A1_SAMPLE_CALENDAR_SCENARIOS), ("sample_path_values", A1_SAMPLE_PATH_VALUES)]:
    table.to_csv(TABLE_DIR / f"A1_P7_{suffix}.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None,"display.max_colwidth",None):
    print("(1) Hợp đồng sau đọc path như Config; status/geometry_status chưa là số mẫu hợp lệ:")
    display(A1_SAMPLE_CONTRACT_RUNS.drop(columns=["config_sha256"]))
    print("(2) Nhóm theo cấu hình đã diễn giải path; raw_contract_id ở (1) giữ dấu vết nhóm thô:")
    display(A1_SAMPLE_CONTRACT_GROUPS.drop(columns=["contract_json"]))
    print("(3) Hình học ô kiểm/dùng (t=0 là ngày cuối); source_review_required vẫn giữ nếu có đường dẫn thật:")
    display(A1_SAMPLE_CONTRACT_ROLES)
    print("(4) Mẫu lịch theo kỳ toàn cục trong config; chưa áp tệp kỳ riêng/cache hoặc đếm mẫu:")
    display(A1_SAMPLE_CALENDAR_SCENARIOS)
    print('(5) Năm trường path: giá trị thô và sau diễn giải; chỉ đúng chuỗi "None"/YAML null thành None:')
    display(A1_SAMPLE_PATH_VALUES)
print("Đã lưu sáu CSV vào:",TABLE_DIR)
print("LƯU Ý: seq_length là tổng khung; nếu forecast=1, overlap=0 thì 365/730 gồm 364/729 ngày hindcast và 1 ngày forecast.")
print("Hợp đồng đối chiếu mã basedataset.py trong repo; chưa chạy loader hoặc xác nhận mọi bản cài có cùng bộ lọc.")

# %% [markdown]
# ### 7.2. Đếm cửa sổ CSV: bộ lọc toàn khung và ô dùng theo pha
#
# Đếm trên đúng 451 ID; chỉ nhận hợp đồng ready, không thay cache/kỳ riêng bằng CSV ngầm.

# %% Phần 7.2 - Đếm cửa sổ CSV: bộ lọc toàn khung và ô dùng theo pha
# Đếm trên đúng 451 ID; chỉ nhận hợp đồng ready, không thay cache/kỳ riêng bằng CSV ngầm.
# Train kiểm đầu vào + nhãn; evaluation giữ lookup lịch và báo riêng ô dùng đầy đủ/có nhãn.
# 365/730 là tổng khung; chưa chạy mô hình, scaler, điền thiếu hoặc sửa bộ lọc thư viện.
started_sample_count = time.perf_counter()
count_groups = A1_SAMPLE_CONTRACT_GROUPS.query("status == 'ready_for_daily_count'").copy()
A1_SAMPLE_COUNT_EXCLUDED = A1_SAMPLE_CONTRACT_GROUPS.query("status != 'ready_for_daily_count'").drop(columns="contract_json").copy()
if count_groups.empty:
    raise ValueError("Không có hợp đồng sẵn sàng; đọc kết quả 7.1 trước.")
count_ids = sorted(set(map(str, selected_ids)), key=int)
count_specs = []
for group in count_groups.itertuples(index=False):
    # contract_json là chữ ký nhóm, không phải config: cfg.get đã ghi cả trường vắng thành null.
    # Đọc lại YAML nguồn để giữ phân biệt trường không khai báo và trường khai báo null.
    config_path = EXP_DIR / group.example_config
    cfg = a1_load_yaml(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path}: YAML không phải mapping.")
    parsed_cfg = sample_normalize_paths(cfg)
    signature_keys = json.loads(group.contract_json).keys()
    current_signature = json.dumps({key: parsed_cfg.get(key) for key in signature_keys},
                                   sort_keys=True, ensure_ascii=False, default=str)
    if current_signature != group.contract_json:
        raise ValueError(f"{group.contract_id}: config nguồn khác hợp đồng đã lưu; chạy lại 7.1 trước.")
    contract = sample_contract(cfg)
    if contract["issues"] or contract["names"]["target_variables"] != ["qmax"]:
        raise ValueError(f"{group.contract_id}: issues={contract['issues']}; "
                         f"target_variables={contract['names']['target_variables']!r}; cần daily/qmax cuối khung.")
    # Tên/offset lấy từ 7.1, không tự suy hai pha chỉ theo tên mô hình.
    offsets = A1_SAMPLE_CONTRACT_OFFSETS.query("contract_id == @group.contract_id")
    for length, part in offsets.groupby("sequence_total_days", sort=True):
        if len(part) != len(contract["dynamic"]) or set(part.variable) != set(contract["dynamic"]):
            raise ValueError("Bảng offset không khớp danh sách dynamic.")
        ranges = {}
        for row in part.itertuples(index=False):
            if row.role in {"full_sequence", "both_phases"}:
                intervals = [(1-int(length), 0)]
            else:
                intervals = []
                for phase in ["hindcast", "forecast"]:
                    first, last = getattr(row, phase+"_offset_first"), getattr(row, phase+"_offset_last")
                    if pd.notna(first):
                        intervals.append((int(first), int(last)))
            for interval in intervals:
                ranges.setdefault(interval, []).append(row.variable)
        count_specs.append(dict(contract_id=group.contract_id, configs=group.configs, length=int(length),
            dynamic=contract["dynamic"], ranges=ranges, static=contract["names"]["static_attributes"],
            has_qmean="qmean" in contract["dynamic"], two_phase=contract["two_phase"]))
count_meteo_names = sorted({v for spec in count_specs for v in spec["dynamic"] if v != "qmean"})
count_static_names = sorted({v for spec in count_specs for v in spec["static"]})
if not set(count_static_names).issubset(catchment.columns):
    raise ValueError("Thuộc tính tĩnh ngoài nguồn Catchment đã kiểm; chưa tự ghép nguồn khác.")
count_static = catchment.loc[count_ids, count_static_names].apply(pd.to_numeric, errors="raise")
if (not np.isfinite(count_static.to_numpy()).all()) or count_static.eq(-999).any().any():
    raise ValueError("Thuộc tính tĩnh thiếu/không hữu hạn/sentinel; cần kiểm trước khi đếm.")
# area_gov thuộc D/Gauge_attributes.csv; Catchment chỉ có diện tích tính từ hình học area_calc.
count_area = pd.to_numeric(gauge.loc[count_ids, "area_gov"], errors="raise")
if not np.isfinite(count_area).all() or count_area.le(0).any():
    raise ValueError("Diện tích chuẩn hóa nhãn không hợp lệ.")
count_calendar = pd.date_range(min(pd.Timestamp(a) for a,b in PERIODS.values()) -
    pd.Timedelta(days=max(spec["length"] for spec in count_specs)-1),
    max(pd.Timestamp(b) for a,b in PERIODS.values()), freq="D")
count_positions = {p: np.flatnonzero((count_calendar >= a) & (count_calendar <= b)) for p,(a,b) in PERIODS.items()}


def sample_read_count_values(path, columns, calendar):
    # Đọc số như bộ nạp; không tự biến số âm khí tượng hoặc sentinel thành NaN.
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=DATE_COLS+columns,
                      dtype={name: str for name in DATE_COLS})
    dates = pd.DatetimeIndex(pd.to_datetime(raw[DATE_COLS].rename(columns=DATE_COLUMN_NAMES), errors="raise"))
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f"{path}: lịch không hợp lệ/trùng/không tăng.")
    numeric = raw[columns].apply(pd.to_numeric, errors="raise").set_axis(dates)
    if np.isinf(numeric.to_numpy()).any() or numeric.eq(-999).any().any():
        raise ValueError(f"{path}: có infinity/-999; không thể đồng nhất kiểm NaN với kiểm hữu hạn.")
    return numeric.reindex(calendar)


def sample_range_complete(missing, endpoints, first, last):
    # Cận đóng; tổng tiền tố tránh dựng ma trận trượt L x ngày x biến.
    endpoints = np.asarray(endpoints, dtype=int)
    left, right = endpoints + int(first), endpoints + int(last)
    covered = (left >= 0) & (right < len(missing)) & (left <= right)
    result = np.zeros(len(endpoints), dtype=bool)
    prefix = np.r_[0, np.cumsum(np.asarray(missing, dtype=np.int64))]
    result[covered] = (prefix[right[covered]+1] - prefix[left[covered]]) == 0
    return result


def sample_count_masks(values, names, ranges, endpoints, length):
    missing = values[names].isna().any(axis=1).to_numpy()
    original = sample_range_complete(missing, endpoints, 1-length, 0)
    phase = np.ones(len(endpoints), dtype=bool)
    for (first,last), used_names in ranges.items():
        used_missing = values[used_names].isna().any(axis=1).to_numpy()
        phase &= sample_range_complete(used_missing, endpoints, first, last)
    if np.any(original & ~phase):
        raise AssertionError("Ô dùng theo pha phải là tập con các ô kiểm toàn khung.")
    return original, phase


sample_count_rows, sample_year_rows, sample_source_rows = [], [], []
for file_no, basin_id in enumerate(count_ids, start=1):
    met = sample_read_count_values(A1_DAILY_FILES["A"][basin_id], count_meteo_names, count_calendar)
    discharge = sample_read_count_values(A1_DAILY_FILES["D"][basin_id], ["qmean","qmax"], count_calendar)
    if discharge["qmean"].lt(0).any():
        raise ValueError("Có qmean âm: cần đối chiếu quy tắc dựng Q input trước khi đếm.")
    # lamah.load_lamah_discharge bỏ Q target âm; giữ nguyên số 0 và không ghi tệp nguồn.
    label = discharge["qmax"].where(discharge["qmax"].ge(0))
    if not np.isfinite((label.dropna()*86.4/count_area.loc[basin_id]).to_numpy()).all():
        raise ValueError("Chuẩn hóa nhãn theo diện tích tạo số không hữu hạn.")
    values = met.assign(qmean=discharge["qmean"])
    for period, positions in count_positions.items():
        sample_source_rows.append(dict(basin_id=basin_id, period=period, calendar_days=len(positions),
            label_present_days=int(label.iloc[positions].notna().sum()),
            qmean_present_days=int(discharge["qmean"].iloc[positions].notna().sum()),
            qmean_qmax_missing_masks_equal=bool(np.array_equal(discharge["qmean"].iloc[positions].isna(),label.iloc[positions].isna()))))
    for spec in count_specs:
        for period, positions in count_positions.items():
            original, phase = sample_count_masks(values, spec["dynamic"], spec["ranges"], positions, spec["length"])
            target = label.iloc[positions].notna().to_numpy()
            original_pair, phase_pair = original & target, phase & target
            # Evaluation lookup không lọc NaN; complete-case chỉ là chẩn đoán, không là dự báo thực.
            lookup = original_pair if period == "train" else np.ones(len(positions), dtype=bool)
            flags = dict(label_present_days=target, original_input_complete_days=original,
                phase_input_complete_days=phase, original_complete_with_label_days=original_pair,
                phase_complete_with_label_days=phase_pair, recovered_with_label_days=phase_pair & ~original_pair,
                label_missing_days=~target, label_present_original_input_missing_days=target & ~original,
                original_lookup_rule_days=lookup)
            common = dict(basin_id=basin_id,contract_id=spec["contract_id"],period=period,
                          sequence_total_days=spec["length"],has_qmean=spec["has_qmean"])
            sample_count_rows.append(dict(**common,calendar_days=len(positions),
                **{key:int(mask.sum()) for key,mask in flags.items()}))
            for year in sorted(set(count_calendar[positions].year)):
                take = count_calendar[positions].year == year
                sample_year_rows.append(dict(**common,year=int(year),calendar_days=int(take.sum()),
                    **{key:int(mask[take].sum()) for key,mask in flags.items()}))
    if file_no % 50 == 0 or file_no == len(count_ids):
        print(f"Đã đếm cửa sổ CSV {file_no}/{len(count_ids)} ID")
A1_SAMPLE_COUNTS_BY_BASIN = pd.DataFrame(sample_count_rows)
A1_SAMPLE_COUNTS_BY_YEAR_BASIN = pd.DataFrame(sample_year_rows)
A1_SAMPLE_COUNT_SOURCES = pd.DataFrame(sample_source_rows)
sample_count_fields = ["calendar_days"] + list(flags)
sample_group_keys = ["contract_id","sequence_total_days","period","has_qmean"]
A1_SAMPLE_COUNTS = A1_SAMPLE_COUNTS_BY_BASIN.groupby(sample_group_keys,sort=False)[sample_count_fields].sum().reset_index()
A1_SAMPLE_COUNTS_BY_YEAR = A1_SAMPLE_COUNTS_BY_YEAR_BASIN.groupby(sample_group_keys+["year"],sort=False)[sample_count_fields].sum().reset_index()
sample_zero = A1_SAMPLE_COUNTS_BY_BASIN.groupby(sample_group_keys,sort=False).agg(
    basins=("basin_id","size"), zero_original_complete_with_label_basins=("original_complete_with_label_days",lambda x:int(x.eq(0).sum())),
    zero_phase_complete_with_label_basins=("phase_complete_with_label_days",lambda x:int(x.eq(0).sum())),
    basins_with_recovery=("recovered_with_label_days",lambda x:int(x.gt(0).sum()))).reset_index()
A1_SAMPLE_COUNTS = A1_SAMPLE_COUNTS.merge(sample_zero,on=sample_group_keys,validate="one_to_one")

# Đối chiếu qmax từng ID/kỳ với kiểm kê thật ở 4.2, không hard-code kết quả dự đoán.
sample_label_check = A1_SAMPLE_COUNT_SOURCES.merge(
    A1_QMAX_MISSING_BY_PERIOD_BASIN[["basin_id","period","valid_days"]],on=["basin_id","period"],validate="one_to_one")
if (len(sample_label_check) != len(count_ids)*len(PERIODS) or
        not sample_label_check.label_present_days.eq(sample_label_check.valid_days).all()):
    raise AssertionError("Nhãn đọc lại khác số ngày hợp lệ ở 4.2.")
for table in [A1_SAMPLE_COUNTS_BY_BASIN,A1_SAMPLE_COUNTS_BY_YEAR_BASIN,A1_SAMPLE_COUNTS,A1_SAMPLE_COUNTS_BY_YEAR]:
    if not table.calendar_days.eq(table.label_missing_days + table.label_present_original_input_missing_days +
                                 table.original_complete_with_label_days).all():
        raise AssertionError("Các lý do loại không cộng khớp lịch.")
    if not table.phase_complete_with_label_days.eq(table.original_complete_with_label_days+table.recovered_with_label_days).all():
        raise AssertionError("Số phục hồi không cộng khớp.")
sample_rebuilt = A1_SAMPLE_COUNTS_BY_YEAR_BASIN.groupby(["basin_id"]+sample_group_keys)[sample_count_fields].sum().sort_index()
sample_direct = A1_SAMPLE_COUNTS_BY_BASIN.set_index(["basin_id"]+sample_group_keys)[sample_count_fields].sort_index()
if not sample_rebuilt.equals(sample_direct):
    raise AssertionError("Cộng theo năm khác đếm cả kỳ.")
A1_SAMPLE_COUNT_CHECKS = pd.DataFrame([dict(label_counts_match_4_2=True,year_sums_match=True,
    rejection_partition_matches=True,phase_superset_check=True,ready_groups=len(count_groups),
    ready_configs=int(count_groups.configs.sum()),excluded_groups=len(A1_SAMPLE_COUNT_EXCLUDED),
    selected_basins=len(count_ids),static_fields_checked=len(count_static_names),
    static_nonfinite_or_sentinel_cells=0,count_start=count_calendar[0].date(),count_end=count_calendar[-1].date())])

fig, axes = plt.subplots(1,2,figsize=(15,6),sharey=True)
plot_ids = count_groups.contract_id.tolist()
for ax,length in zip(axes,[365,730]):
    part = A1_SAMPLE_COUNTS.query("period == 'train' and sequence_total_days == @length").set_index("contract_id").reindex(plot_ids)
    y = np.arange(len(plot_ids))
    for shift,field,color,label_text in [(-0.18,"original_complete_with_label_days","#2283a5","Full-window rule"),
                                         (0.18,"phase_complete_with_label_days","#c87531","Consumed-cell complete case")]:
        ax.barh(y+shift,100*part[field]/part.calendar_days,height=.34,color=color,label=label_text)
    ax.set_title(f"Training endpoints: total sequence {length} days")
    ax.set_xlabel("Complete inputs and available target / calendar endpoints (%)")
    ax.set_xlim(0,100);ax.set_yticks(y);ax.set_yticklabels(plot_ids,fontsize=9)
    ax.grid(axis="x",alpha=.18)
axes[0].invert_yaxis()
fig.suptitle("CSV window counts on the fixed selected basin set",fontsize=15)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.095),ncol=2,frameon=False)
fig.text(.5,.035,"Raw CSV availability only; qmean is joined by date from gauge files. Cached-source contract excluded.\n"
         "Phase-based counts are a hypothetical complete-case diagnostic; no loader, scaler or model was run.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.17,1,.92))
sample_count_figure = FIGURE_DIR / "A1_P7_sample_window_counts.png"
fig.savefig(sample_count_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("sample_counts_by_basin",A1_SAMPLE_COUNTS_BY_BASIN),("sample_counts_by_year_basin",A1_SAMPLE_COUNTS_BY_YEAR_BASIN),
    ("sample_counts",A1_SAMPLE_COUNTS),("sample_counts_by_year",A1_SAMPLE_COUNTS_BY_YEAR),
    ("sample_count_sources",A1_SAMPLE_COUNT_SOURCES),("sample_count_checks",A1_SAMPLE_COUNT_CHECKS),
    ("sample_count_excluded_contracts",A1_SAMPLE_COUNT_EXCLUDED)]:
    table.to_csv(TABLE_DIR/f"A1_P7_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None,"display.max_colwidth",None):
    print("(1) Phạm vi CSV và kiểm đếm; chưa xác nhận cache/scaler/cohort lịch sử của từng run:")
    display(A1_SAMPLE_COUNT_CHECKS)
    print("(2) Train: hai quy tắc complete-case cùng nhãn; lý do loại tách rời cộng đúng lịch:")
    display(A1_SAMPLE_COUNTS.query("period == 'train'"))
    print("(3) Validation/test: lookup gốc giữ lịch; complete-case có nhãn là chẩn đoán riêng:")
    display(A1_SAMPLE_COUNTS.query("period != 'train'").drop(columns=["label_missing_days","label_present_original_input_missing_days"]))
    print("(4) Năm 2003, tổng khung 730: kiểm riêng tác động lịch trước 2002:")
    display(A1_SAMPLE_COUNTS_BY_YEAR.query("period == 'train' and year == 2003 and sequence_total_days == 730"))
    print("(5) Các ID hồ sơ/không nhãn train; toàn bộ ID lưu CSV:")
    display(A1_SAMPLE_COUNTS_BY_BASIN.query("basin_id in ['55','324','600','623','800','742','758'] and period == 'train'")
            [["basin_id","contract_id","sequence_total_days","label_present_days","original_complete_with_label_days","phase_complete_with_label_days"]])
    print("(6) Hợp đồng chưa đếm vì nguồn ngoài; không gán số mẫu bằng 0:")
    display(A1_SAMPLE_COUNT_EXCLUDED[["contract_id","configs","example_config","source_review_reasons"]])
print("Đã lưu hình:",sample_count_figure,"; bảy CSV vào:",TABLE_DIR)
print(f"Thời gian đếm cửa sổ: {time.perf_counter()-started_sample_count:.1f} giây.")
print("CẢNH BÁO: phép đếm dùng selected_ids hiện tại, CSV gốc và thuộc tính tĩnh hữu hạn; chưa xác nhận basin files/cache/scaler lịch sử.")
print("Evaluation lookup đủ lịch không bảo đảm dự báo hữu hạn. Complete-case theo pha không mô phỏng masked mean hoặc khả năng chịu NaN của mô hình.")

# %% [markdown]
# ### 7.3. Danh sách ID từng config, nguồn cache và hồ sơ không có cửa sổ
#
# Đọc basin files theo từng kỳ, giữ rõ cách tìm path đã di chuyển và không tự dùng tập 451 thay tệp thiếu.

# %% Phần 7.3 - Danh sách ID từng config, nguồn cache và hồ sơ không có cửa sổ
# Đọc basin files theo từng kỳ, giữ rõ cách tìm path đã di chuyển và không tự dùng tập 451 thay tệp thiếu.
# Tái cộng kết quả 7.2 khi tập ID nằm trong selected_ids; chỉ kiểm sự hiện diện cache, chưa mở nội dung cache.
from pathlib import PurePosixPath

started_sample_provenance = time.perf_counter()
provenance_ids = set(map(str, selected_ids))
provenance_runs = A1_SAMPLE_CONTRACT_RUNS.set_index("run", verify_integrity=True)
provenance_files = {}
for path in EXP_DIR.rglob("*"):
    if path.is_file() and not a1_is_metadata(path):
        provenance_files.setdefault(path.name, []).append(path)


def sample_locate_declared_file(value):
    # Chỉ path tồn tại như khai báo là đối chiếu trực tiếp. Hai cách còn lại là ứng viên đã di chuyển.
    if value is None:
        return None, "not_declared", []
    raw_path = str(value)
    direct = Path(raw_path)
    if direct.is_file():
        return direct, "exists_as_declared", [direct]
    parts = PurePosixPath(raw_path.replace("\\", "/")).parts
    relocated = []
    for i, part in enumerate(parts):
        if part == EXP_DIR.name:
            candidate = EXP_DIR.joinpath(*parts[i+1:])
            if candidate.is_file():
                relocated.append(candidate)
    relocated = sorted(set(relocated))
    if len(relocated) == 1:
        return relocated[0], "candidate_by_Experiments_suffix", relocated
    if len(relocated) > 1:
        return None, "ambiguous_Experiments_suffix", relocated
    candidates = sorted(provenance_files.get(parts[-1], [])) if parts else []
    if len(candidates) == 1:
        return candidates[0], "candidate_by_unique_basename", candidates
    return None, "ambiguous_basename" if candidates else "not_found_in_current_Experiments", candidates


def sample_read_cohort(path):
    # Cùng đơn vị một ID mỗi dòng như load_basin_file; không tự đổi 001 thành 1 hoặc tách nhiều ID trong một dòng.
    ids = sorted(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    if not ids or any(not token.isdigit() or int(token) < 1 or token != str(int(token)) for token in ids):
        raise ValueError(f"{path}: cần danh sách ID nguyên dương dạng hiện hành, mỗi dòng một ID.")
    return ids


provenance_cohort_rows, provenance_source_rows, provenance_count_rows = [], [], []
for row in A1_SAMPLE_CONTRACT_RUNS.itertuples(index=False):
    config_path = EXP_DIR / row.run
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != row.config_sha256:
        raise ValueError(f"{row.run}: config khác kết quả 7.1; chưa được ghép số đếm cũ.")
    cfg = sample_normalize_paths(a1_load_yaml(config_path.read_text(encoding="utf-8-sig")))
    cache_path, cache_status, cache_candidates = sample_locate_declared_file(cfg.get("train_data_file"))
    data_dir = cfg.get("data_dir")
    data_leaf = PurePosixPath(str(data_dir).replace("\\", "/")).name if data_dir is not None else ""
    provenance_source_rows.append(dict(run=row.run,contract_id=row.contract_id,status_7_1=row.status,
        qmean_in_dynamic=row.qmean_in_dynamic,data_dir_declared=str(data_dir),data_dir_leaf=data_leaf,
        data_dir_exists_as_declared=bool(data_dir is not None and Path(data_dir).is_dir()),
        same_leaf_as_A1_LAMAH_DIR=data_leaf==LAMAH_DIR.name,
        train_cache_active=cfg.get("train_data_file") is not None,train_cache_declared=str(cfg.get("train_data_file")),
        cache_path_status=cache_status,cache_candidate=str(cache_path) if cache_path else "",
        cache_candidate_count=len(cache_candidates),cache_candidate_bytes=cache_path.stat().st_size if cache_path else np.nan,
        cache_content_checked=False,stored_count_source="current_A_D_CSV; qmean joined from D"))
    for period in PERIODS:
        field = period+"_basin_file"
        path, path_status, candidates = sample_locate_declared_file(cfg.get(field))
        ids = sample_read_cohort(path) if path is not None else None
        cohort_set = set(ids) if ids is not None else set()
        duplicates = len(ids)-len(cohort_set) if ids is not None else np.nan
        subset = bool(ids is not None and not duplicates and cohort_set.issubset(provenance_ids))
        cohort_key = hashlib.sha256("\n".join(sorted(cohort_set)).encode()).hexdigest()[:12] if ids is not None else ""
        provenance_cohort_rows.append(dict(run=row.run,contract_id=row.contract_id,period=period,
            declared_file=str(cfg.get(field)),path_status=path_status,candidate=str(path) if path else "",
            candidates=len(candidates),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path else "",
            cohort_key=cohort_key,listed_rows=len(ids) if ids is not None else np.nan,
            distinct_ids=len(cohort_set) if ids is not None else np.nan,duplicate_rows=duplicates,
            equals_selected_set=bool(ids is not None and not duplicates and cohort_set==provenance_ids),
            subset_of_selected_for_count=subset,
            selected_ids_not_in_file=" | ".join(sorted(provenance_ids-cohort_set,key=int)) if ids is not None else "",
            file_ids_outside_selected=" | ".join(sorted(cohort_set-provenance_ids,key=int)) if ids is not None else "",
            historical_file_identity_verified=False))
        for length in sorted(A1_SAMPLE_CALENDAR_SCENARIOS.sequence_total_days.unique()):
            counts = A1_SAMPLE_COUNTS_BY_BASIN.loc[
                A1_SAMPLE_COUNTS_BY_BASIN.contract_id.eq(row.contract_id) &
                A1_SAMPLE_COUNTS_BY_BASIN.period.eq(period) &
                A1_SAMPLE_COUNTS_BY_BASIN.sequence_total_days.eq(length)]
            status = "current_CSV_subset_count" if subset and not counts.empty else (
                "external_source_contract_not_counted" if counts.empty else "cohort_not_resolved_or_outside_selected")
            totals = {key:np.nan for key in ["calendar_days","label_present_days","original_complete_with_label_days", "phase_complete_with_label_days"]}
            if status == "current_CSV_subset_count":
                if set(counts.basin_id.astype(str)) != provenance_ids:
                    raise AssertionError("Số đếm 7.2 không đủ tập ID trước khi tái cộng theo cohort.")
                part = counts[counts.basin_id.astype(str).isin(cohort_set)]
                totals = {key:int(part[key].sum()) for key in totals}
            provenance_count_rows.append(dict(run=row.run,contract_id=row.contract_id,period=period,
                sequence_total_days=int(length),cohort_key=cohort_key,equals_selected_set=bool(ids is not None and not duplicates and cohort_set==provenance_ids),
                cohort_path_status=path_status,count_status=status,**totals))
A1_SAMPLE_CONFIG_COHORTS = pd.DataFrame(provenance_cohort_rows)
A1_SAMPLE_CONFIG_SOURCES = pd.DataFrame(provenance_source_rows)
A1_SAMPLE_CONFIG_COHORT_COUNTS = pd.DataFrame(provenance_count_rows)

A1_SAMPLE_ZERO_PROFILES = A1_SAMPLE_COUNTS_BY_BASIN.loc[
    A1_SAMPLE_COUNTS_BY_BASIN.original_complete_with_label_days.eq(0) |
    A1_SAMPLE_COUNTS_BY_BASIN.phase_complete_with_label_days.eq(0)].copy()
A1_SAMPLE_ZERO_PROFILES["reason"] = np.where(A1_SAMPLE_ZERO_PROFILES.label_present_days.eq(0),
    "no_target_in_period", "target_present_but_no_complete_window")
zero_rows = []
zero_keys = ["contract_id","sequence_total_days","period","has_qmean"]
for key,part in A1_SAMPLE_ZERO_PROFILES.groupby(zero_keys,sort=False):
    zero_rows.append(dict(zip(zero_keys,key),
        no_target_ids=" | ".join(sorted(part.loc[part.reason.eq("no_target_in_period"),"basin_id"].astype(str),key=int)),
        target_present_no_original_window_ids=" | ".join(sorted(part.loc[part.label_present_days.gt(0)&part.original_complete_with_label_days.eq(0),"basin_id"].astype(str),key=int)),
        target_present_no_phase_window_ids=" | ".join(sorted(part.loc[part.label_present_days.gt(0)&part.phase_complete_with_label_days.eq(0),"basin_id"].astype(str),key=int))))
A1_SAMPLE_ZERO_SUMMARY = pd.DataFrame(zero_rows,columns=zero_keys+[
    "no_target_ids","target_present_no_original_window_ids","target_present_no_phase_window_ids"])
if len(A1_SAMPLE_CONFIG_COHORTS) != len(provenance_runs)*len(PERIODS):
    raise AssertionError("Thiếu kiểm kê basin file theo config/kỳ.")
for suffix,table in [("sample_config_cohorts",A1_SAMPLE_CONFIG_COHORTS),("sample_config_sources",A1_SAMPLE_CONFIG_SOURCES),
    ("sample_config_cohort_counts",A1_SAMPLE_CONFIG_COHORT_COUNTS),("sample_zero_profiles",A1_SAMPLE_ZERO_PROFILES),
    ("sample_zero_summary",A1_SAMPLE_ZERO_SUMMARY)]:
    table.to_csv(TABLE_DIR/f"A1_P7_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None,"display.max_colwidth",None):
    print("(1) Danh sách ID theo từng kỳ; trạng thái candidate không xác nhận tệp lịch sử:")
    display(A1_SAMPLE_CONFIG_COHORTS.groupby(["period","path_status","cohort_key","distinct_ids","duplicate_rows","equals_selected_set"],dropna=False).size().rename("configs").reset_index())
    print("(2) Config/kỳ có tập khác 451 hoặc chưa đọc được; không thay bằng tập mặc định:")
    display(A1_SAMPLE_CONFIG_COHORTS.loc[~A1_SAMPLE_CONFIG_COHORTS.equals_selected_set,
        ["run","period","declared_file","path_status","distinct_ids","duplicate_rows","selected_ids_not_in_file","file_ids_outside_selected"]])
    print("(3) data_dir khai báo; cùng tên thư mục chưa xác nhận cùng nội dung:")
    display(A1_SAMPLE_CONFIG_SOURCES.groupby(["data_dir_declared","qmean_in_dynamic","data_dir_exists_as_declared","same_leaf_as_A1_LAMAH_DIR"],dropna=False).size().rename("configs").reset_index())
    print("(4) Cache thật được khai báo: chỉ kiểm sự hiện diện, không mở nội dung:")
    display(A1_SAMPLE_CONFIG_SOURCES.loc[A1_SAMPLE_CONFIG_SOURCES.train_cache_active,
        ["run","train_cache_declared","cache_path_status","cache_candidate","cache_candidate_count","cache_candidate_bytes","cache_content_checked"]])
    print("(5) ID không có cửa sổ kèm nhãn: tách không nhãn khỏi có nhãn nhưng thiếu đầu vào:")
    display(A1_SAMPLE_ZERO_SUMMARY)
    print("(6) Trạng thái tái cộng theo tập ID khai báo; chưa là số mẫu huấn luyện lịch sử:")
    display(A1_SAMPLE_CONFIG_COHORT_COUNTS.groupby(["period","sequence_total_days","count_status","equals_selected_set"],dropna=False).size().rename("configs").reset_index())
    print("Các số tái cộng khác tập 451 hoặc chưa đếm:")
    display(A1_SAMPLE_CONFIG_COHORT_COUNTS.loc[~A1_SAMPLE_CONFIG_COHORT_COUNTS.equals_selected_set |
        A1_SAMPLE_CONFIG_COHORT_COUNTS.count_status.ne("current_CSV_subset_count")])
print("Đã lưu năm CSV vào:",TABLE_DIR)
print(f"Thời gian kiểm cohort/cache/hồ sơ cửa sổ: {time.perf_counter()-started_sample_provenance:.1f} giây.")
print("CẢNH BÁO: candidate là cách tìm tệp trong Input hiện tại; cùng tên/path hoặc cùng tập ID không xác nhận bản dữ liệu/cache/scaler lúc train.")

# %% [markdown]
# ### 7.4. Chốt phạm vi CSV từng kỳ, tách cache train khỏi evaluation
#
# Cache chỉ thay nguồn ở train theo basedataset.py; evaluation chỉ tái dùng số CSV khi hợp đồng mặt nạ trùng đầy đủ.

# %% Phần 7.4 - Chốt phạm vi CSV từng kỳ, tách cache train khỏi evaluation
# Cache chỉ thay nguồn ở train theo basedataset.py; evaluation chỉ tái dùng số CSV khi hợp đồng mặt nạ trùng đầy đủ.
# Kiểm SHA config/cohort, biến động/tĩnh, target và offset/lịch; không mở cache hoặc sửa kết quả 7.2–7.3.
started_sample_finalize = time.perf_counter()
final_metrics = ["calendar_days","label_present_days","original_input_complete_days","phase_input_complete_days",
    "original_complete_with_label_days","phase_complete_with_label_days","recovered_with_label_days","original_lookup_rule_days"]
final_offset_fields = ["variable","original_offset_first","original_offset_last","hindcast_offset_first",
    "hindcast_offset_last","forecast_offset_first","forecast_offset_last"]
final_contracts = {}
for group in A1_SAMPLE_CONTRACT_GROUPS.itertuples(index=False):
    cfg = a1_load_yaml((EXP_DIR/group.example_config).read_text(encoding="utf-8-sig"))
    final_contracts[group.contract_id] = sample_contract(cfg)
for row in A1_SAMPLE_CONTRACT_RUNS.itertuples(index=False):
    if hashlib.sha256((EXP_DIR/row.run).read_bytes()).hexdigest() != row.config_sha256:
        raise ValueError(f"{row.run}: config đã đổi sau 7.1; chưa được dùng số đếm cũ.")


def sample_mask_signature(contract_id, length, period):
    # Đây là khóa mặt nạ CSV, không là khóa kiến trúc, scaler hoặc dữ liệu huấn luyện lịch sử.
    contract = final_contracts[contract_id]
    if contract["geometry_issues"] or contract["names"]["target_variables"] != ["qmax"]:
        return None
    calendar = A1_SAMPLE_CALENDAR_SCENARIOS.loc[
        A1_SAMPLE_CALENDAR_SCENARIOS.contract_id.eq(contract_id) &
        A1_SAMPLE_CALENDAR_SCENARIOS.sequence_total_days.eq(length) &
        A1_SAMPLE_CALENDAR_SCENARIOS.period.eq(period)]
    offsets = A1_SAMPLE_CONTRACT_OFFSETS.loc[
        A1_SAMPLE_CONTRACT_OFFSETS.contract_id.eq(contract_id) &
        A1_SAMPLE_CONTRACT_OFFSETS.sequence_total_days.eq(length)]
    if len(calendar)!=1 or offsets.variable.duplicated().any() or set(offsets.variable)!=set(contract["dynamic"]):
        raise ValueError(f"{contract_id}/{length}/{period}: lịch/offset không xác định duy nhất.")
    normalized_offsets = [tuple(None if pd.isna(value) else int(value) if field!="variable" else value
        for field,value in zip(final_offset_fields,values))
        for values in offsets.sort_values("variable")[final_offset_fields].itertuples(index=False,name=None)]
    row = calendar.iloc[0]
    payload = dict(dynamic=sorted(contract["dynamic"]),static=sorted(contract["names"]["static_attributes"]),
        target=contract["names"]["target_variables"],offsets=normalized_offsets,sequence_total_days=int(length),
        predict_last_n=int(row.predict_last_n),warmup_start=pd.Timestamp(row.warmup_start).isoformat(),
        target_start=pd.Timestamp(row.target_start).isoformat(),target_end=pd.Timestamp(row.target_end).isoformat())
    return json.dumps(payload,sort_keys=True,ensure_ascii=False)


final_donors = {}
for group in A1_SAMPLE_CONTRACT_GROUPS.query("status == 'ready_for_daily_count'").itertuples(index=False):
    for length in sorted(A1_SAMPLE_CALENDAR_SCENARIOS.sequence_total_days.unique()):
        for period in PERIODS:
            signature = sample_mask_signature(group.contract_id,length,period)
            if signature is not None:
                final_donors.setdefault(signature,[]).append(group.contract_id)
final_count_rows, final_map_rows = [], []
final_cohorts = A1_SAMPLE_CONFIG_COHORTS.set_index(["run","period"],verify_integrity=True)
for run in A1_SAMPLE_CONTRACT_RUNS.itertuples(index=False):
    contract = final_contracts[run.contract_id]
    for period in PERIODS:
        cohort = final_cohorts.loc[(run.run,period)]
        cohort_ids = None
        if cohort.subset_of_selected_for_count:
            path = Path(cohort.candidate)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=cohort.file_sha256:
                raise ValueError(f"{run.run}/{period}: tệp cohort đã đổi sau 7.3.")
            cohort_ids = set(sample_read_cohort(path))
            if not cohort_ids.issubset(set(map(str,selected_ids))):
                raise ValueError("Cohort có ID ngoài phạm vi đã đếm.")
        for length in sorted(A1_SAMPLE_CALENDAR_SCENARIOS.sequence_total_days.unique()):
            signature = sample_mask_signature(run.contract_id,length,period)
            matches = sorted(final_donors.get(signature,[])) if signature is not None else []
            source_issues = contract["source_issues"]
            # Chỉ ngoại lệ cache train được bỏ qua ở evaluation; các nguồn ngoài khác vẫn chưa hỗ trợ.
            if cohort_ids is None:
                status = "cohort_not_supported"
            elif contract["geometry_issues"]:
                status = "geometry_not_supported"
            elif source_issues and source_issues != ["train_data_file_to_review"]:
                status = "other_external_source_not_supported"
            elif source_issues and period=="train":
                status = "train_cache_content_unknown"
            elif not matches:
                status = "no_equivalent_CSV_mask_count"
            else:
                status = "own_current_CSV_count" if run.contract_id in matches else "equivalent_mask_current_CSV_count"
            provider = run.contract_id if run.contract_id in matches else matches[0] if matches else ""
            totals = {field:np.nan for field in final_metrics}
            if status in {"own_current_CSV_count","equivalent_mask_current_CSV_count"}:
                references = []
                for donor in matches:
                    part = A1_SAMPLE_COUNTS_BY_BASIN.loc[
                        A1_SAMPLE_COUNTS_BY_BASIN.contract_id.eq(donor) &
                        A1_SAMPLE_COUNTS_BY_BASIN.sequence_total_days.eq(length) &
                        A1_SAMPLE_COUNTS_BY_BASIN.period.eq(period)].set_index("basin_id",verify_integrity=True)
                    if not cohort_ids.issubset(set(part.index.astype(str))):
                        raise AssertionError("Nguồn số đếm không bao đủ cohort.")
                    references.append(part.loc[sorted(cohort_ids,key=int),final_metrics].sort_index())
                if not all(references[0].equals(reference) for reference in references[1:]):
                    raise AssertionError("Các hợp đồng cùng mặt nạ cho số đếm khác nhau.")
                totals = {field:int(references[0][field].sum()) for field in final_metrics}
            final_map_rows.append(dict(run=run.run,contract_id=run.contract_id,period=period,sequence_total_days=int(length),
                mask_signature_sha256=hashlib.sha256(signature.encode()).hexdigest() if signature else "",
                equivalent_ready_contracts=" | ".join(matches),provider_contract=provider if status in {
                    "own_current_CSV_count","equivalent_mask_current_CSV_count"} else "",count_status=status))
            final_count_rows.append(dict(run=run.run,contract_id=run.contract_id,period=period,sequence_total_days=int(length),
                cohort_key=cohort.cohort_key,cohort_path_status=cohort.path_status,count_status=status,
                historical_training_count_verified=False,**totals))
A1_SAMPLE_FINAL_MASK_MAP = pd.DataFrame(final_map_rows)
A1_SAMPLE_FINAL_COUNTS = pd.DataFrame(final_count_rows)
A1_SAMPLE_FINAL_SCOPE = A1_SAMPLE_FINAL_COUNTS.groupby(
    ["period","sequence_total_days","count_status"],dropna=False).size().rename("configs").reset_index()
expected_rows = len(A1_SAMPLE_CONTRACT_RUNS)*len(PERIODS)*A1_SAMPLE_CALENDAR_SCENARIOS.sequence_total_days.nunique()
if len(A1_SAMPLE_FINAL_COUNTS)!=expected_rows:
    raise AssertionError("Thiếu config/kỳ/L trong bảng phạm vi cuối.")
available = A1_SAMPLE_FINAL_COUNTS.count_status.isin(["own_current_CSV_count","equivalent_mask_current_CSV_count"])
if not A1_SAMPLE_FINAL_COUNTS.loc[~available,final_metrics].isna().all().all():
    raise AssertionError("Nguồn chưa biết phải giữ NaN.")
for suffix,table in [("sample_final_counts",A1_SAMPLE_FINAL_COUNTS),("sample_final_mask_map",A1_SAMPLE_FINAL_MASK_MAP),
                     ("sample_final_scope",A1_SAMPLE_FINAL_SCOPE)]:
    table.to_csv(TABLE_DIR/f"A1_P7_{suffix}.csv",index=False,encoding="utf-8-sig")
external_ids = A1_SAMPLE_CONTRACT_GROUPS.loc[A1_SAMPLE_CONTRACT_GROUPS.status.ne("ready_for_daily_count"),"contract_id"]
with pd.option_context("display.max_rows",None,"display.max_columns",None,"display.max_colwidth",None):
    print("(1) Đối chiếu mặt nạ của nhóm nguồn ngoài; khớp không xác nhận cache hoặc scaler:")
    display(A1_SAMPLE_FINAL_MASK_MAP.loc[A1_SAMPLE_FINAL_MASK_MAP.contract_id.isin(external_ids)])
    print("(2) Phạm vi cuối theo kỳ/L; train từ cache và phép đếm CSV được phân biệt:")
    display(A1_SAMPLE_FINAL_SCOPE)
    print("(3) Số CSV evaluation của nhóm cache khi có mặt nạ tương đương; train giữ chưa biết:")
    display(A1_SAMPLE_FINAL_COUNTS.loc[A1_SAMPLE_FINAL_COUNTS.contract_id.isin(external_ids)])
    print("(4) Các dòng còn chưa đếm, không gán 0:")
    display(A1_SAMPLE_FINAL_COUNTS.loc[~available])
print("Đã lưu ba CSV vào:",TABLE_DIR)
print(f"Thời gian chốt phạm vi số cửa sổ: {time.perf_counter()-started_sample_finalize:.1f} giây.")
print("CẢNH BÁO: số CSV mô tả Input hiện tại và cohort ứng viên đã đọc; không xác nhận dữ liệu/scaler lịch sử hoặc dự báo hữu hạn.")

# %% [markdown]
# ## Phần 8. Phân phối nhãn giữa các kỳ
#
# Mô tả nhãn, độ phủ train và đóng góp đuôi. Đối chiếu NSE tác giả chỉ là chẩn đoán theo nguồn, không dùng chọn mô hình.

# %% [markdown]
# ### 8.1. Phân phối qmax giữa các kỳ và tỷ lệ vượt ngưỡng chỉ tính từ train
#
# Dùng cache Q 5.1, giữ số 0, đối chiếu số nhãn từng ID/kỳ với 7.2; không đổi tập ID theo mặt nạ cửa sổ.

# %% Phần 8.1 - Phân phối qmax giữa các kỳ và tỷ lệ vượt ngưỡng chỉ tính từ train
# Dùng cache Q 5.1, giữ số 0, đối chiếu số nhãn từng ID/kỳ với 7.2; không đổi tập ID theo mặt nạ cửa sổ.
# Mean/std/W1 là mô tả phân phối; p95/p99 fit riêng từng ID trên train, không dùng để chọn mô hình bằng test.
from scipy.stats import wasserstein_distance

started_q_period = time.perf_counter()
q_period_ids = list(map(str,A1_QMAX_VALUE_IDS))
q_period_calendar = pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
q_period_values = np.asarray(A1_QMAX_VALUES_MM_DAY,dtype=np.float64)
if (len(set(q_period_ids))!=len(q_period_ids) or set(q_period_ids)!=set(map(str,selected_ids)) or
    q_period_values.shape!=(len(q_period_ids),len(q_period_calendar)) or q_period_calendar.has_duplicates or
    not q_period_calendar.equals(pd.date_range(q_period_calendar.min(),q_period_calendar.max(),freq="D")) or
    np.isinf(q_period_values).any() or (q_period_values<0).any()):
    raise ValueError("Cache Q 5.1 khác ID/lịch/giá trị hữu hạn không âm đã kiểm.")
q_period_positions = {}
for period,(start,end) in PERIODS.items():
    positions = np.flatnonzero((q_period_calendar>=start)&(q_period_calendar<=end))
    if len(positions)!=len(pd.date_range(start,end,freq="D")):
        raise ValueError(f"Cache không bao trọn {period}.")
    q_period_positions[period] = positions


def q_period_statistics(values):
    # ddof=0 như hồ sơ nhãn ở 5.6; phân vị tuyến tính trên ngày có số, không gộp ID.
    values = np.asarray(values,dtype=float)
    if not len(values):
        return {field:np.nan for field in ["min","p01","p05","p50","p95","p99","max","mean","std_ddof0"]}
    quantiles = np.quantile(values,[0,.01,.05,.5,.95,.99,1],method="linear")
    return dict(zip(["min","p01","p05","p50","p95","p99","max"],quantiles),
                mean=float(values.mean()),std_ddof0=0.0 if quantiles[0]==quantiles[-1] else float(values.std(ddof=0)))


q_period_rows,q_train_rows,q_tail_rows = [],[],[]
for basin_id,series in zip(q_period_ids,q_period_values):
    train_raw = series[q_period_positions["train"]]
    train = train_raw[np.isfinite(train_raw)]
    reference = q_period_statistics(train)
    q_train_rows.append(dict(basin_id=basin_id,train_valid_days=len(train),train_start=PERIODS["train"][0],
        train_end=PERIODS["train"][1],train_mean_mm_day=reference["mean"],train_std_ddof0_mm_day=reference["std_ddof0"],
        train_p95_mm_day=reference["p95"],train_p99_mm_day=reference["p99"],
        threshold_status="defined" if len(train) else "no_train_observations"))
    for period,positions in q_period_positions.items():
        raw = series[positions];values=raw[np.isfinite(raw)]
        stats=q_period_statistics(values)
        mean_ratio=stats["mean"]/reference["mean"] if len(values) and len(train) and reference["mean"]>0 else np.nan
        std_ratio=stats["std_ddof0"]/reference["std_ddof0"] if len(values) and len(train) and reference["std_ddof0"]>0 else np.nan
        row=dict(basin_id=basin_id,period=period,expected_days=len(raw),valid_days=len(values),
            missing_days=len(raw)-len(values),zero_days=int((values==0).sum()),train_valid_days=len(train),**stats,
            mean_ratio_to_train=mean_ratio,std_ratio_to_train=std_ratio,
            w1_to_train_mm_day=float(wasserstein_distance(train,values)) if len(train) and len(values) else np.nan)
        q_period_rows.append(row)
        for label in ["p95","p99"]:
            threshold=reference[label]
            if np.isfinite(threshold):
                above=int((values>threshold).sum());equal=int((values==threshold).sum());below=int((values<threshold).sum())
                if above+equal+below!=len(values):
                    raise AssertionError("Ba trạng thái so ngưỡng không cộng khớp ngày có số.")
            else:
                above=equal=below=np.nan
            q_tail_rows.append(dict(basin_id=basin_id,period=period,train_quantile=label,threshold_mm_day=threshold,
                train_valid_days=len(train),valid_days=len(values),expected_days=len(raw),missing_days=len(raw)-len(values),
                above_days=above,equal_days=equal,below_days=below,
                above_pct_valid=100*above/len(values) if np.isfinite(threshold) and len(values) else np.nan,
                status="no_train_observations" if not len(train) else "no_period_observations" if not len(values) else "defined"))
A1_Q_PERIOD_PROFILE=pd.DataFrame(q_period_rows)
A1_Q_TRAIN_THRESHOLDS=pd.DataFrame(q_train_rows)
A1_Q_TRAIN_THRESHOLD_COUNTS=pd.DataFrame(q_tail_rows)
q_label_check=A1_Q_PERIOD_PROFILE.merge(A1_SAMPLE_COUNT_SOURCES[["basin_id","period","label_present_days"]],
    on=["basin_id","period"],validate="one_to_one")
if len(q_label_check)!=len(q_period_ids)*len(PERIODS) or not q_label_check.valid_days.eq(q_label_check.label_present_days).all():
    raise AssertionError("Số nhãn cache khác 7.2; chưa so phân phối.")
q_period_summary_rows=[]
for period,part in A1_Q_PERIOD_PROFILE.groupby("period",sort=False):
    for metric in ["mean","std_ddof0","p99","mean_ratio_to_train","std_ratio_to_train","w1_to_train_mm_day"]:
        values=part[metric].to_numpy();values=values[np.isfinite(values)]
        quantiles=np.quantile(values,[0,.05,.5,.95,1],method="linear") if len(values) else [np.nan]*5
        q_period_summary_rows.append(dict(period=period,metric=metric,defined_basins=len(values),
            **dict(zip(["min","p05","median","p95","max"],quantiles))))
A1_Q_PERIOD_SUMMARY=pd.DataFrame(q_period_summary_rows)
q_tail_summary_rows=[]
for (period,level),part in A1_Q_TRAIN_THRESHOLD_COUNTS.groupby(["period","train_quantile"],sort=False):
    eligible=part.loc[part.threshold_mm_day.notna()]
    valid=int(eligible.valid_days.sum());above=int(eligible.above_days.sum())
    q_tail_summary_rows.append(dict(period=period,train_quantile=level,basins=len(part),
        basins_with_train_threshold=len(eligible),basins_with_defined_rate=int(part.above_pct_valid.notna().sum()),
        eligible_valid_days=valid,above_days=above,equal_days=int(eligible.equal_days.sum()),
        pooled_above_pct_valid=100*above/valid if valid else np.nan,
        median_basin_above_pct_valid=part.above_pct_valid.median()))
A1_Q_TRAIN_THRESHOLD_SUMMARY=pd.DataFrame(q_tail_summary_rows)

fig,axes=plt.subplots(2,2,figsize=(14,9))
q_period_styles={"train":("#2283a5","Train"),"validation":("#c87531","Validation"),"test":("#8655ad","Test")}


def q_period_ecdf(ax,values,label,color):
    values=np.asarray(values,dtype=float);values=np.sort(values[np.isfinite(values)])
    if len(values):ax.step(values,100*np.arange(1,len(values)+1)/len(values),where="post",color=color,label=f"{label} (n={len(values)})")


for ax,metric,title,xlabel in [(axes[0,0],"mean_ratio_to_train","Mean relative to the same basin's train mean","Period mean / train mean"),
    (axes[0,1],"std_ratio_to_train","Daily variability relative to train","Period population standard deviation / train standard deviation"),
    (axes[1,0],"w1_to_train_mm_day","Distribution distance from train","W1 of daily qmax distributions (mm/day)")]:
    for period in ["validation","test"]:
        color,label=q_period_styles[period]
        q_period_ecdf(ax,A1_Q_PERIOD_PROFILE.loc[A1_Q_PERIOD_PROFILE.period.eq(period),metric],label,color)
    ax.set_title(title,fontsize=11);ax.set_xlabel(xlabel,fontsize=9);ax.set_xscale("symlog",linthresh=.1 if metric=="w1_to_train_mm_day" else 1)
    ax.set_ylabel("Basins at or below value (%)");ax.set_ylim(0,102);ax.grid(alpha=.2)
    if metric!="w1_to_train_mm_day":ax.axvline(1,color="gray",linestyle=":",linewidth=1)
ax=axes[1,1]
for period,(color,label) in q_period_styles.items():
    part=A1_Q_TRAIN_THRESHOLD_COUNTS.loc[A1_Q_TRAIN_THRESHOLD_COUNTS.period.eq(period)&A1_Q_TRAIN_THRESHOLD_COUNTS.train_quantile.eq("p95")]
    q_period_ecdf(ax,part.above_pct_valid,label,color)
ax.set(title="Observed days above each basin's train p95",xlabel="Strict exceedances / available days (%)",ylabel="Basins at or below rate (%)",ylim=(0,102));ax.grid(alpha=.2)
for ax in axes.flat:
    if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8,loc="best")
fig.suptitle("Daily qmax across periods: basin distributions and train-defined thresholds",fontsize=14)
fig.text(.5,.04,"Equal basin weight in ECDFs; missing observations remain missing and zeros are retained.\n"
    "Train p95/p99 use linear sample quantiles; ties are reported separately. No train data or zero ratio denominator gives an undefined result.\n"
    "All axes retain the observed range; symlog only changes the display. This describes labels, not forecast performance or runoff volume.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.12,1,.94))
q_period_figure=FIGURE_DIR/"A1_P8_qmax_period_distributions.png"
fig.savefig(q_period_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("qmax_period_profile",A1_Q_PERIOD_PROFILE),("qmax_train_thresholds",A1_Q_TRAIN_THRESHOLDS),
    ("qmax_train_threshold_counts",A1_Q_TRAIN_THRESHOLD_COUNTS),("qmax_period_summary",A1_Q_PERIOD_SUMMARY),
    ("qmax_train_threshold_summary",A1_Q_TRAIN_THRESHOLD_SUMMARY)]:
    table.to_csv(TABLE_DIR/f"A1_P8_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Phủ nhãn từng kỳ, khớp 7.2; chưa lọc theo cửa sổ hoặc mô hình:")
    display(A1_Q_PERIOD_PROFILE.groupby("period",sort=False).agg(basins=("basin_id","size"),expected_days=("expected_days","sum"),
        valid_days=("valid_days","sum"),missing_days=("missing_days","sum"),zero_days=("zero_days","sum"),
        basins_with_labels=("valid_days",lambda s:int(s.gt(0).sum()))).reset_index())
    print("(2) Phân vị ngang chỉ số từng ID; không phải phân vị của các ngày gộp:")
    display(A1_Q_PERIOD_SUMMARY)
    print("(3) Vượt p95/p99 train: phép so >, ngày bằng ngưỡng đếm riêng; ID không có train không được gán ngưỡng:")
    display(A1_Q_TRAIN_THRESHOLD_SUMMARY)
    print("(4) Các ID hồ sơ và không nhãn train; tỷ số/W1 chưa xác định giữ NaN:")
    display(A1_Q_PERIOD_PROFILE.loc[A1_Q_PERIOD_PROFILE.basin_id.isin(["55","324","600","623","742","758","800"]),
        ["basin_id","period","valid_days","train_valid_days","mean","std_ddof0","p99","mean_ratio_to_train","std_ratio_to_train","w1_to_train_mm_day"]])
print("Đã lưu hình:",q_period_figure,"; năm CSV vào:",TABLE_DIR)
print(f"Thời gian so phân phối nhãn theo kỳ: {time.perf_counter()-started_q_period:.1f} giây.")
print("CẢNH BÁO: phần trăm vượt chỉ dùng ngày có nhãn và ngưỡng train xác định; thiếu quan trắc có thể đổi phân phối được thấy.")

# %% [markdown]
# ### 8.2. Đóng góp ngày vượt ngưỡng train theo ID/năm và đối chiếu riêng ID 800
#
# Giữ nguyên ngưỡng 8.1; tách ngày bằng ngưỡng, cộng năm khớp từng ID/kỳ và báo độ phủ tháng train.

# %% Phần 8.2 - Đóng góp ngày vượt ngưỡng train theo ID/năm và đối chiếu riêng ID 800
# Giữ nguyên ngưỡng 8.1; tách ngày bằng ngưỡng, cộng năm khớp từng ID/kỳ và báo độ phủ tháng train.
# Phép bỏ riêng ID 800 chỉ là đối chiếu mô tả cho hồ sơ đã xác định; không đổi tập mô hình hoặc fit ngưỡng bằng test.
started_q_tail = time.perf_counter()
tail_thresholds = A1_Q_TRAIN_THRESHOLDS.set_index("basin_id",verify_integrity=True)
if set(tail_thresholds.index.astype(str))!=set(q_period_ids):
    raise ValueError("Bảng ngưỡng 8.1 khác tập ID cache.")
for level in ["p95","p99"]:
    previous=A1_Q_TRAIN_THRESHOLD_COUNTS.loc[A1_Q_TRAIN_THRESHOLD_COUNTS.train_quantile.eq(level)]
    expected=previous.basin_id.map(tail_thresholds["train_"+level+"_mm_day"])
    if not np.allclose(previous.threshold_mm_day,expected,rtol=0,atol=0,equal_nan=True):
        raise AssertionError("Ngưỡng khác kết quả 8.1; chưa được phân rã số vượt.")
tail_year_positions = {(period,int(year)):positions[q_period_calendar[positions].year==year]
    for period,positions in q_period_positions.items() for year in sorted(set(q_period_calendar[positions].year))}
tail_train_month_rows,tail_train_coverage_rows,tail_year_rows = [],[],[]
for basin_id,series in zip(q_period_ids,q_period_values):
    train_positions=q_period_positions["train"]
    train_valid=np.isfinite(series[train_positions]);train_dates=q_period_calendar[train_positions[train_valid]]
    if len(train_dates)!=int(tail_thresholds.loc[basin_id,"train_valid_days"]):
        raise AssertionError("Độ phủ train khác bảng ngưỡng 8.1.")
    tail_train_coverage_rows.append(dict(basin_id=basin_id,train_valid_days=len(train_dates),
        train_expected_days=len(train_positions),train_month_types_observed=len(set(train_dates.month)),
        train_years_observed=len(set(train_dates.year)),first_train_observation=train_dates.min() if len(train_dates) else pd.NaT,
        last_train_observation=train_dates.max() if len(train_dates) else pd.NaT,
        train_p95_mm_day=tail_thresholds.loc[basin_id,"train_p95_mm_day"],
        train_p99_mm_day=tail_thresholds.loc[basin_id,"train_p99_mm_day"]))
    for month in range(1,13):
        expected=int((q_period_calendar[train_positions].month==month).sum());valid=int((train_dates.month==month).sum())
        tail_train_month_rows.append(dict(basin_id=basin_id,month=month,expected_days=expected,valid_days=valid,missing_days=expected-valid))
    for (period,year),positions in tail_year_positions.items():
        raw=series[positions];values=raw[np.isfinite(raw)]
        for level in ["p95","p99"]:
            threshold=tail_thresholds.loc[basin_id,"train_"+level+"_mm_day"]
            above=int((values>threshold).sum()) if np.isfinite(threshold) else np.nan
            equal=int((values==threshold).sum()) if np.isfinite(threshold) else np.nan
            below=int((values<threshold).sum()) if np.isfinite(threshold) else np.nan
            tail_year_rows.append(dict(basin_id=basin_id,period=period,year=year,train_quantile=level,
                threshold_mm_day=threshold,expected_days=len(raw),valid_days=len(values),missing_days=len(raw)-len(values),
                above_days=above,equal_days=equal,below_days=below))
A1_Q_TRAIN_MONTH_COUNTS=pd.DataFrame(tail_train_month_rows)
A1_Q_TRAIN_COVERAGE=pd.DataFrame(tail_train_coverage_rows)
A1_Q_THRESHOLD_YEAR_BASIN=pd.DataFrame(tail_year_rows)

# Cộng theo năm phải khớp 8.1, kể cả NaN cho ba ID không có train.
tail_keys=["basin_id","period","train_quantile"]
tail_integer_fields=["expected_days","valid_days","missing_days","above_days","equal_days","below_days"]
tail_rebuilt=A1_Q_THRESHOLD_YEAR_BASIN.groupby(tail_keys)[tail_integer_fields].sum(min_count=1).sort_index()
tail_reference=A1_Q_TRAIN_THRESHOLD_COUNTS.set_index(tail_keys,verify_integrity=True)[tail_integer_fields].sort_index()
if not tail_rebuilt.index.equals(tail_reference.index) or not np.allclose(
    tail_rebuilt.to_numpy(),tail_reference.to_numpy(),rtol=0,atol=0,equal_nan=True):
    raise AssertionError("Cộng số vượt theo năm khác từng ID/kỳ ở 8.1.")
month_check=A1_Q_TRAIN_MONTH_COUNTS.groupby("basin_id")[["expected_days","valid_days"]].sum().sort_index()
coverage_check=A1_Q_TRAIN_COVERAGE.set_index("basin_id").sort_index()
if not (month_check.expected_days.eq(coverage_check.train_expected_days).all() and month_check.valid_days.eq(coverage_check.train_valid_days).all()):
    raise AssertionError("Độ phủ tháng không cộng khớp train.")

tail_annual_rows=[]
for (period,year,level),part in A1_Q_THRESHOLD_YEAR_BASIN.groupby(["period","year","train_quantile"],sort=False):
    eligible=part[part.threshold_mm_day.notna()]
    valid=int(eligible.valid_days.sum());above=int(eligible.above_days.sum())
    tail_annual_rows.append(dict(period=period,year=year,train_quantile=level,basins_with_train_threshold=len(eligible),
        eligible_valid_days=valid,above_days=above,equal_days=int(eligible.equal_days.sum()),
        pooled_above_pct_valid=100*above/valid if valid else np.nan))
A1_Q_THRESHOLD_YEAR_SUMMARY=pd.DataFrame(tail_annual_rows)

tail_detail=A1_Q_TRAIN_THRESHOLD_COUNTS.copy()
train_rates=tail_detail.loc[tail_detail.period.eq("train"),["basin_id","train_quantile","above_pct_valid"]].rename(
    columns={"above_pct_valid":"empirical_train_above_pct"})
tail_detail=tail_detail.merge(train_rates,on=["basin_id","train_quantile"],validate="many_to_one")
tail_totals=tail_detail.groupby(["period","train_quantile"]).above_days.transform(lambda s:s.sum(min_count=1))
tail_detail["share_of_observed_exceedances_pct"]=100*tail_detail.above_days/tail_totals.where(tail_totals.gt(0))
# Đây là tham chiếu nếu tần suất train giữ nguyên, không phải mô hình kỳ vọng hoặc kiểm định nguyên nhân.
tail_detail["reference_count_at_empirical_train_rate"]=tail_detail.valid_days*tail_detail.empirical_train_above_pct/100
tail_detail["excess_count_vs_empirical_train_rate"]=tail_detail.above_days-tail_detail.reference_count_at_empirical_train_rate
A1_Q_TAIL_CONTRIBUTIONS=tail_detail.merge(A1_Q_TRAIN_COVERAGE[["basin_id","train_month_types_observed","train_years_observed"]],
    on="basin_id",validate="many_to_one")
tail_sensitivity_rows=[]
for scope,part in [("all_IDs_with_train_threshold",A1_Q_TAIL_CONTRIBUTIONS),
                   ("without_ID800_diagnostic",A1_Q_TAIL_CONTRIBUTIONS.loc[A1_Q_TAIL_CONTRIBUTIONS.basin_id.ne("800")])]:
    for (period,level),subset in part.groupby(["period","train_quantile"],sort=False):
        eligible=subset[subset.threshold_mm_day.notna()]
        valid=int(eligible.valid_days.sum());above=int(eligible.above_days.sum())
        tail_sensitivity_rows.append(dict(scope=scope,period=period,train_quantile=level,
            basins_with_train_threshold=len(eligible),eligible_valid_days=valid,above_days=above,
            pooled_above_pct_valid=100*above/valid if valid else np.nan,
            median_basin_above_pct_valid=eligible.above_pct_valid.median(),
            reference_count_at_empirical_train_rate=eligible.reference_count_at_empirical_train_rate.sum(min_count=1),
            excess_count_vs_empirical_train_rate=eligible.excess_count_vs_empirical_train_rate.sum(min_count=1)))
A1_Q_TAIL_DIAGNOSTIC_SCOPES=pd.DataFrame(tail_sensitivity_rows)
all_scope=A1_Q_TAIL_DIAGNOSTIC_SCOPES.query("scope=='all_IDs_with_train_threshold'")
scope_check=all_scope.merge(A1_Q_TRAIN_THRESHOLD_SUMMARY,on=["period","train_quantile"],validate="one_to_one",suffixes=("_new","_8_1"))
for field in ["basins_with_train_threshold","eligible_valid_days","above_days","pooled_above_pct_valid","median_basin_above_pct_valid"]:
    if not np.allclose(scope_check[field+"_new"],scope_check[field+"_8_1"],rtol=1e-12,atol=1e-12,equal_nan=True):
        raise AssertionError("Phân rã đóng góp làm đổi tổng 8.1.")

fig,axes=plt.subplots(2,2,figsize=(15,10))
for column,level in enumerate(["p95","p99"]):
    ax=axes[0,column]
    for period,(color,label) in q_period_styles.items():
        part=A1_Q_THRESHOLD_YEAR_SUMMARY.loc[A1_Q_THRESHOLD_YEAR_SUMMARY.period.eq(period)&A1_Q_THRESHOLD_YEAR_SUMMARY.train_quantile.eq(level)]
        ax.plot(part.year,part.pooled_above_pct_valid,marker="o",color=color,label=label)
    ax.axhline(5 if level=="p95" else 1,color="gray",linestyle=":",label="Nominal quantile reference")
    ax.set(title=f"Observed exceedance of train {level} by calendar year",xlabel="Calendar year",ylabel="Strict exceedances / eligible observed days (%)")
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=axes[1,column]
    part=A1_Q_TAIL_CONTRIBUTIONS.loc[A1_Q_TAIL_CONTRIBUTIONS.period.eq("test")&A1_Q_TAIL_CONTRIBUTIONS.train_quantile.eq(level)&A1_Q_TAIL_CONTRIBUTIONS.share_of_observed_exceedances_pct.notna()]
    part=part.sort_values(["above_days","basin_id"],ascending=[False,True],kind="stable").head(10).iloc[::-1]
    labels=[f"ID {r.basin_id} (train n={r.train_valid_days})" for r in part.itertuples()]
    ax.barh(labels,part.share_of_observed_exceedances_pct,color="#8655ad")
    ax.set(title=f"Largest shares of test days above train {level}",xlabel="Share of all observed test exceedances (%)")
    ax.grid(axis="x",alpha=.2)
fig.suptitle("Train-defined qmax tails: annual coverage and basin contributions",fontsize=15)
fig.text(.5,.04,"Thresholds are fixed from train; ties are counted separately. Annual denominators use observed days with a defined train threshold.\n"
    "Bars rank descriptive counts only; basins remain in the model set. Train n is the number of observed target days, not valid model windows.\n"
    "ID 800 exclusion is a separate diagnostic table; no thresholds or model filters are changed. Days are not independent flood events.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.12,1,.94))
q_tail_figure=FIGURE_DIR/"A1_P8_qmax_tail_contributions.png"
fig.savefig(q_tail_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("qmax_train_month_counts",A1_Q_TRAIN_MONTH_COUNTS),("qmax_train_coverage",A1_Q_TRAIN_COVERAGE),
    ("qmax_threshold_year_basin",A1_Q_THRESHOLD_YEAR_BASIN),("qmax_threshold_year_summary",A1_Q_THRESHOLD_YEAR_SUMMARY),
    ("qmax_tail_contributions",A1_Q_TAIL_CONTRIBUTIONS),("qmax_tail_diagnostic_scopes",A1_Q_TAIL_DIAGNOSTIC_SCOPES)]:
    table.to_csv(TABLE_DIR/f"A1_P8_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) 15 ID có ít ngày train nhất, gồm ID không có ngưỡng; toàn bộ độ phủ tháng lưu CSV:")
    display(A1_Q_TRAIN_COVERAGE.sort_values(["train_valid_days","basin_id"],kind="stable").head(15))
    print("(2) Năm ID góp nhiều ngày vượt nhất mỗi kỳ/nguỡng; chưa là xếp hạng lỗi mô hình:")
    top_parts=[]
    for period in ["validation","test"]:
        for level in ["p95","p99"]:
            part=A1_Q_TAIL_CONTRIBUTIONS.loc[A1_Q_TAIL_CONTRIBUTIONS.period.eq(period)&A1_Q_TAIL_CONTRIBUTIONS.train_quantile.eq(level)&A1_Q_TAIL_CONTRIBUTIONS.above_days.notna()]
            top_parts.append(part.sort_values(["above_days","basin_id"],ascending=[False,True],kind="stable").head(5))
    display(pd.concat(top_parts)[["basin_id","period","train_quantile","train_valid_days","train_month_types_observed", "threshold_mm_day", "valid_days","above_days","above_pct_valid","share_of_observed_exceedances_pct","empirical_train_above_pct","excess_count_vs_empirical_train_rate"]])
    print("(3) Cộng theo năm khớp đúng từng ID/kỳ ở 8.1; mẫu số không gồm các ID thiếu ngưỡng train:")
    display(A1_Q_THRESHOLD_YEAR_SUMMARY)
    print("(4) Hai phạm vi mô tả; bỏ ID 800 chỉ là phép đối chiếu, không đổi tập 451 hoặc fit lại ngưỡng:")
    display(A1_Q_TAIL_DIAGNOSTIC_SCOPES)
print("Đã lưu hình:",q_tail_figure,"; sáu CSV vào:",TABLE_DIR)
print(f"Thời gian phân rã đuôi qmax: {time.perf_counter()-started_q_tail:.1f} giây.")
print("CẢNH BÁO: phần trăm phản ánh ngày quan trắc/ngưỡng train hiện có; train ngắn hoặc thiếu theo mùa hạn chế tính đại diện của ngưỡng.")

# %% [markdown]
# ### 8.3. Kiểm kê CSV NSE: tách tệp tác giả khỏi output chạy lại A2
#
# Sáu run cố định lấy từ A2 Phần 2/5; chỉ nhận đúng best_model hoặc bố cục export A2 có environment.json.

# %% Phần 8.3 - Kiểm kê CSV NSE: tách tệp tác giả khỏi output chạy lại A2
# Sáu run cố định lấy từ A2 Phần 2/5; chỉ nhận đúng best_model hoặc bố cục export A2 có environment.json.
# Không tự chọn epoch/tệp theo NSE, không thay nguồn A2 bằng nguồn tác giả và không mở pickle chuỗi dự báo.
started_nse_inventory=time.perf_counter()
A1_A2_RUN_MAP={
    "SeqLSTM_Q":"Sequential_Forecast_LSTM/Sequential_Forecast_LSTM_with_q/Sequential_Forecast_LSTM_RA_with_q_Simple_Embedding",
    "SeqLSTM_noQ":"Sequential_Forecast_LSTM/Sequential_Forecast_LSTM_without_q/Sequential_Forecast_LSTM_RA_without_q_Simple_Embedding",
    "EncDec_Q":"Encoder_Decoder_LSTM/Encoder_Decoder_LSTM_with_q/Encoder_Decoder_LSTM_with_q_Complex_Embedding",
    "EncDec_noQ":"Encoder_Decoder_LSTM/Encoder_Decoder_LSTM_without_q/Encoder_Decoder_LSTM_RA_without_q_Complex_Embedding",
    "D_FC":"Baseline/Forecast",
    "D_RA":"Baseline/Reanalysis/Reanalysis_Simple_Embedding"}
nse_selected=set(map(str,selected_ids))
nse_file_rows,nse_value_rows,nse_manifest_rows=[],[],[]


def sample_nse_file(path,source,run,author_run,config_path,artifact_id,selection_basis,reported_repro_ok=None):
    row=dict(source=source,run=run,author_run=author_run,artifact_id=artifact_id,selection_basis=selection_basis,
        metric_file=str(path) if path else "",config_file=str(config_path),reported_repro_ok=reported_repro_ok,
        status="not_found",scope_status="not_checked",sha256="",columns="",nse_column="",
        rows=np.nan,finite_nse=np.nan,negative_finite_nse=np.nan,blank_or_nan_nse=np.nan,
        non_numeric_nse=np.nan,infinite_nse=np.nan,duplicate_id_rows=np.nan,
        selected_ids_without_row="",ids_outside_selected="",min_nse=np.nan,median_nse=np.nan,max_nse=np.nan)
    if path is None or not path.is_file():
        nse_file_rows.append(row);return
    row["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not config_path.is_file():
        row["status"]="config_not_found";nse_file_rows.append(row);return
    try:
        cfg=a1_load_yaml(config_path.read_text(encoding="utf-8-sig"))
        targets=sample_input_names(cfg.get("target_variables",[]),"target_variables")
        dates=(sample_date(cfg.get("test_start_date")),sample_date(cfg.get("test_end_date")))
        row["scope_status"]="declared_qmax_test_matches_A1" if targets==["qmax"] and dates==tuple(map(pd.Timestamp,PERIODS["test"])) else "declared_scope_mismatch"
        raw=pd.read_csv(path,dtype=str,keep_default_na=False,encoding="utf-8-sig")
    except Exception as exc:
        row["status"]="read_error: "+type(exc).__name__+": "+str(exc)[:160];nse_file_rows.append(row);return
    row.update(rows=len(raw),columns=" | ".join(raw.columns))
    candidates=[column for column in ["NSE","qmax_NSE"] if column in raw.columns]
    if "basin" not in raw.columns or len(candidates)!=1:
        row["status"]="unsupported_or_ambiguous_schema";nse_file_rows.append(row);return
    column=candidates[0];row["nse_column"]=column
    ids=raw.basin.str.strip()
    if not len(ids) or any(not value.isdigit() or int(value)<1 or value!=str(int(value)) for value in ids):
        row["status"]="invalid_basin_ids";nse_file_rows.append(row);return
    row["duplicate_id_rows"]=int(ids.duplicated().sum())
    row["selected_ids_without_row"]=" | ".join(sorted(nse_selected-set(ids),key=int))
    row["ids_outside_selected"]=" | ".join(sorted(set(ids)-nse_selected,key=int))
    tokens=raw[column].str.strip();values=pd.to_numeric(tokens,errors="coerce")
    missing=tokens.eq("")|tokens.str.lower().eq("nan")
    nonnumeric=~missing&values.isna();finite=np.isfinite(values)
    row.update(finite_nse=int(finite.sum()),negative_finite_nse=int((finite&values.lt(0)).sum()),
        blank_or_nan_nse=int(missing.sum()),non_numeric_nse=int(nonnumeric.sum()),infinite_nse=int(np.isinf(values).sum()),
        min_nse=values[finite].min(),median_nse=values[finite].median(),max_nse=values[finite].max())
    row["status"]="CSV_read" if not row["duplicate_id_rows"] and not nonnumeric.any() else "duplicate_ids_or_non_numeric_NSE"
    nse_file_rows.append(row)
    # Giữ giá trị âm/NaN/inf riêng; chưa loại hoặc dùng chúng để tính tương quan.
    if row["status"]=="CSV_read":
        for basin_id,value,token in zip(ids,values,tokens):
            nse_value_rows.append(dict(source=source,run=run,author_run=author_run,artifact_id=artifact_id,
                basin_id=basin_id,nse=value,raw_nse=token,finite=bool(np.isfinite(value)),
                in_selected=basin_id in nse_selected,scope_status=row["scope_status"],metric_file=str(path),file_sha256=row["sha256"]))


# Nguồn tác giả: chỉ dùng đúng test/best_model; các epoch khác chỉ được liệt kê, không tự chọn tệp đầu tiên.
nse_author_candidates=[]
for run,relative in A1_A2_RUN_MAP.items():
    directory=EXP_DIR/relative;canonical=directory/"test"/"best_model"/"test_metrics.csv"
    alternatives=sorted(path for path in directory.rglob("test_metrics.csv") if not a1_is_metadata(path)) if directory.is_dir() else []
    nse_author_candidates.append(dict(run=run,canonical_exists=canonical.is_file(),
        candidate_files=len(alternatives),candidates=" | ".join(map(str,alternatives))))
    sample_nse_file(canonical,"author_packaged",run,relative,directory/"config.yml","author_Experiments","exact_test/best_model")
A1_NSE_AUTHOR_CANDIDATES=pd.DataFrame(nse_author_candidates)

# Chỉ metadata được A2 xuất mới nhận diện nguồn rerun. Tìm trên Input và WORK_DIR hiện tại, không tải/chạy A2.
nse_search_roots=sorted(set(path.resolve() for path in [SEARCH_ROOT,WORK_DIR] if path.exists()))
manifest_paths=set()
for root_path in nse_search_roots:
    manifest_paths.update(path.resolve() for path in root_path.rglob("environment.json") if path.is_file() and not a1_is_metadata(path))
nse_exports=[]
for path in sorted(manifest_paths):
    try:
        metadata=json.loads(path.read_text(encoding="utf-8-sig"))
        is_a2=isinstance(metadata,dict) and isinstance(metadata.get("runs"),dict) and "repro_ok" in metadata
        status="A2_export_metadata" if is_a2 else "not_A2_export_schema"
    except Exception as exc:
        metadata={};is_a2=False;status="read_error: "+type(exc).__name__
    nse_manifest_rows.append(dict(file=str(path),status=status,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reported_repro_ok=metadata.get("repro_ok") if isinstance(metadata,dict) else None,
        reported_commit=metadata.get("neuralhydrology_commit","") if isinstance(metadata,dict) else ""))
    if is_a2:nse_exports.append((path,metadata))
for manifest,metadata in nse_exports:
    artifact_id=manifest.parent.as_posix()
    for run,relative in A1_A2_RUN_MAP.items():
        mapped=metadata["runs"].get(run)
        expected=manifest.parent/"runs"/run/"test_metrics.csv"
        if mapped!=relative:
            nse_file_rows.append(dict(source="A2_rerun_export",run=run,author_run=relative,artifact_id=artifact_id,
                selection_basis="A2_export_layout",metric_file=str(expected),config_file=str(expected.parent/"config.yml"),
                status="manifest_run_mapping_mismatch",scope_status="not_checked",reported_repro_ok=metadata.get("repro_ok")))
            continue
        sample_nse_file(expected,"A2_rerun_export",run,relative,expected.parent/"config.yml",artifact_id,
            "A2_export_layout",metadata.get("repro_ok"))
if not nse_exports:
    for run,relative in A1_A2_RUN_MAP.items():
        nse_file_rows.append(dict(source="A2_rerun_export",run=run,author_run=relative,artifact_id="",
            selection_basis="A2_environment.json_required",metric_file="",config_file="",status="A2_export_not_found_in_search_roots",scope_status="not_checked"))
A1_NSE_MANIFEST_AUDIT=pd.DataFrame(nse_manifest_rows,columns=["file","status","sha256","reported_repro_ok","reported_commit"])
A1_NSE_FILE_AUDIT=pd.DataFrame(nse_file_rows)
A1_NSE_VALUES=pd.DataFrame(nse_value_rows,columns=["source","run","author_run","artifact_id","basin_id","nse","raw_nse","finite","in_selected","scope_status","metric_file","file_sha256"])
for suffix,table in [("nse_author_candidates",A1_NSE_AUTHOR_CANDIDATES),("nse_manifest_audit",A1_NSE_MANIFEST_AUDIT),
    ("nse_file_audit",A1_NSE_FILE_AUDIT),("nse_values",A1_NSE_VALUES)]:
    table.to_csv(TABLE_DIR/f"A1_P8_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None,"display.max_colwidth",None):
    print("(1) Nguồn A2 trong phạm vi tìm kiếm hiện tại; reported_repro_ok là metadata khai báo, chưa kiểm lại:")
    print("Search roots:",[str(path) for path in nse_search_roots]);display(A1_NSE_MANIFEST_AUDIT)
    print("(2) CSV NSE theo run/nguồn: số âm giữ nguyên; chưa có phép ghép hoặc tương quan:")
    display(A1_NSE_FILE_AUDIT[["source","run","artifact_id","status","scope_status","rows","finite_nse","negative_finite_nse","blank_or_nan_nse","non_numeric_nse","infinite_nse","duplicate_id_rows","selected_ids_without_row","ids_outside_selected","min_nse","median_nse","max_nse"]])
    print("(3) Tệp metrics tác giả: chỉ canonical best_model được đọc; các ứng viên khác chưa được tự chọn:")
    display(A1_NSE_AUTHOR_CANDIDATES)
    print("(4) Các nguồn chưa đọc/schema khác và schema của các tệp hiện có:")
    display(A1_NSE_FILE_AUDIT[["source","run","artifact_id","metric_file","nse_column","columns","status"]])
    print("(5) NSE các ID hồ sơ nếu có trong CSV đã đọc; không thay NaN bằng 0 hoặc cắt NSE âm:")
    display(A1_NSE_VALUES.loc[A1_NSE_VALUES.basin_id.isin(["55","742","758","800"]),
        ["source","run","artifact_id","basin_id","nse","finite","scope_status"]])
print("Đã lưu bốn CSV vào:",TABLE_DIR)
print(f"Thời gian kiểm nguồn NSE: {time.perf_counter()-started_nse_inventory:.1f} giây.")
if not nse_exports:print("CẢNH BÁO: chưa thấy export A2 có environment.json trong các Search roots trên; CSV tác giả không được gọi là kết quả chạy lại A2.")
print("CẢNH BÁO: ngày đánh giá mới kiểm qua khai báo config; CSV NSE chưa xác nhận tập ngày/quan trắc thực sự được dùng để chấm.")

# %% [markdown]
# ### 8.4. Ghép đặc điểm nhãn hiện tại với NSE tác giả theo ID
#
# Chỉ dùng nguồn author_packaged đã kiểm 8.3; không thay thành NSE chạy lại A2 hoặc suy tập ngày chấm trùng CSV hiện tại.

# %% Phần 8.4 - Ghép đặc điểm nhãn hiện tại với NSE tác giả theo ID
# Chỉ dùng nguồn author_packaged đã kiểm 8.3; không thay thành NSE chạy lại A2 hoặc suy tập ngày chấm trùng CSV hiện tại.
# Pearson/Spearman là mô tả ngang lưu vực; giữ NSE âm và báo độ nhạy với các hồ sơ 742/758/800 đã xác định trước.
started_nse_associations=time.perf_counter()
assoc_ids=set(map(str,selected_ids))
assoc_runs=list(A1_A2_RUN_MAP)
assoc_audit=A1_NSE_FILE_AUDIT.loc[A1_NSE_FILE_AUDIT.source.eq("author_packaged")].copy()
if (len(assoc_audit)!=len(assoc_runs) or set(assoc_audit.run)!=set(assoc_runs) or assoc_audit.run.duplicated().any()
    or not assoc_audit.status.eq("CSV_read").all() or not assoc_audit.scope_status.eq("declared_qmax_test_matches_A1").all()):
    raise ValueError("Cần đủ sáu CSV tác giả đúng scope khai báo từ 8.3; chưa ghép NSE.")
for row in assoc_audit.itertuples():
    if hashlib.sha256(Path(row.metric_file).read_bytes()).hexdigest()!=row.sha256:
        raise ValueError(f"CSV tác giả {row.run} đã đổi sau 8.3; chạy lại kiểm kê trước.")
assoc_nse=A1_NSE_VALUES.loc[A1_NSE_VALUES.source.eq("author_packaged")].copy()
if (assoc_nse.duplicated(["run","basin_id"]).any() or set(assoc_nse.run)!=set(assoc_runs)
    or not assoc_nse.scope_status.eq("declared_qmax_test_matches_A1").all()):
    raise ValueError("Bảng giá trị NSE không còn duy nhất theo run/ID hoặc khác scope 8.3.")
for row in assoc_audit.itertuples():
    part=assoc_nse.loc[assoc_nse.run.eq(row.run)]
    if (len(part)!=row.rows or set(part.basin_id)!=assoc_ids or not part.file_sha256.eq(row.sha256).all()
        or not part.artifact_id.eq(row.artifact_id).all() or not part.metric_file.eq(row.metric_file).all()
        or int(np.isfinite(part.nse).sum())!=row.finite_nse):
        raise ValueError(f"{row.run}: giá trị/ID khác kiểm kê 8.3; chưa dùng cho tương quan.")

# Một hàng mỗi ID; giữ NaN ở tỷ số/W1/ngưỡng của các ID không có train.
assoc_train=A1_Q_TRAIN_COVERAGE.set_index("basin_id",verify_integrity=True)
assoc_thresholds=A1_Q_TRAIN_THRESHOLDS.set_index("basin_id",verify_integrity=True)
assoc_test=A1_Q_PERIOD_PROFILE.loc[A1_Q_PERIOD_PROFILE.period.eq("test")].set_index("basin_id",verify_integrity=True)
assoc_tail=A1_Q_TRAIN_THRESHOLD_COUNTS.loc[A1_Q_TRAIN_THRESHOLD_COUNTS.period.eq("test")&
    A1_Q_TRAIN_THRESHOLD_COUNTS.train_quantile.eq("p99")].set_index("basin_id",verify_integrity=True)
if any(set(table.index)!=assoc_ids for table in [assoc_train,assoc_thresholds,assoc_test,assoc_tail]):
    raise ValueError("Các bảng đặc điểm nhãn không cùng tập ID đã chọn.")
if not assoc_train.train_valid_days.eq(assoc_thresholds.train_valid_days).all():
    raise ValueError("Số nhãn train khác giữa 8.1 và 8.2.")
assoc_features=pd.DataFrame(index=pd.Index(sorted(assoc_ids,key=int),name="basin_id"))
assoc_features["train_valid_days"]=assoc_train.train_valid_days
assoc_features["train_month_types_observed"]=assoc_train.train_month_types_observed
assoc_features["test_valid_days"]=assoc_test.valid_days
assoc_features["train_coverage_pct"]=100*assoc_train.train_valid_days/assoc_train.train_expected_days
assoc_features["test_coverage_pct"]=100*assoc_test.valid_days/assoc_test.expected_days
assoc_features["train_std_mm_day"]=assoc_thresholds.train_std_ddof0_mm_day
assoc_features["test_std_mm_day"]=assoc_test.std_ddof0
assoc_features["test_mean_ratio_to_train"]=assoc_test.mean_ratio_to_train
assoc_features["test_std_ratio_to_train"]=assoc_test.std_ratio_to_train
assoc_features["test_w1_to_train_mm_day"]=assoc_test.w1_to_train_mm_day
assoc_features["test_above_p99_pct"]=assoc_tail.above_pct_valid
assoc_metrics={"train_coverage_pct":"Train coverage\n(%)","test_coverage_pct":"Test coverage\n(%)",
    "train_std_mm_day":"Train std\n(mm/day)","test_std_mm_day":"Test std\n(mm/day)",
    "test_mean_ratio_to_train":"Test / train\nmean","test_std_ratio_to_train":"Test / train\nstd",
    "test_w1_to_train_mm_day":"W1 to train\n(mm/day)","test_above_p99_pct":"Test > train p99\n(%)"}
A1_NSE_LABEL_JOIN=assoc_nse.merge(assoc_features.reset_index(),on="basin_id",how="left",validate="many_to_one")


def nse_label_correlations(x,y):
    # Spearman là Pearson của thứ hạng trung bình khi có giá trị trùng; không xuất p-value giả định ID độc lập.
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)<3:return np.nan,np.nan,"fewer_than_3_pairs"
    if np.ptp(x)==0 or np.ptp(y)==0:return np.nan,np.nan,"constant_feature_or_NSE"
    pearson=float(np.corrcoef(x,y)[0,1])
    spearman=float(np.corrcoef(pd.Series(x).rank(method="average"),pd.Series(y).rank(method="average"))[0,1])
    return pearson,spearman,"defined"


assoc_scopes={"all_selected":set(),"without_742_758_diagnostic":{"742","758"},"without_800_diagnostic":{"800"}}
assoc_corr_rows,assoc_coverage_rows=[],[]
for run in assoc_runs:
    whole=A1_NSE_LABEL_JOIN.loc[A1_NSE_LABEL_JOIN.run.eq(run)]
    assoc_coverage_rows.append(dict(source="author_packaged",run=run,selected_basins=len(assoc_ids),rows=len(whole),
        finite_nse=int(np.isfinite(whole.nse).sum()),negative_finite_nse=int((np.isfinite(whole.nse)&whole.nse.lt(0)).sum()),
        nonfinite_nse_ids=" | ".join(sorted(whole.loc[~np.isfinite(whole.nse),"basin_id"],key=int))))
    for scope,excluded in assoc_scopes.items():
        part=whole.loc[~whole.basin_id.isin(excluded)]
        finite_nse=np.isfinite(part.nse)
        for metric in assoc_metrics:
            finite_x=np.isfinite(part[metric]);eligible=finite_x&finite_nse
            pearson,spearman,status=nse_label_correlations(part.loc[eligible,metric],part.loc[eligible,"nse"])
            assoc_corr_rows.append(dict(source="author_packaged",run=run,scope=scope,metric=metric,
                scope_basins=len(part),paired_basins=int(eligible.sum()),
                nonfinite_nse_basins=int((~finite_nse).sum()),nonfinite_feature_with_finite_nse=int((finite_nse&~finite_x).sum()),
                excluded_nse_ids=" | ".join(sorted(part.loc[~finite_nse,"basin_id"],key=int)),
                excluded_feature_with_finite_nse_ids=" | ".join(sorted(part.loc[finite_nse&~finite_x,"basin_id"],key=int)),
                pearson_r=pearson,spearman_rho=spearman,status=status))
A1_NSE_LABEL_COVERAGE=pd.DataFrame(assoc_coverage_rows)
A1_NSE_LABEL_ASSOCIATIONS=pd.DataFrame(assoc_corr_rows)
assoc_full=A1_NSE_LABEL_ASSOCIATIONS.loc[A1_NSE_LABEL_ASSOCIATIONS.scope.eq("all_selected")]
A1_NSE_LABEL_SENSITIVITY=A1_NSE_LABEL_ASSOCIATIONS.loc[~A1_NSE_LABEL_ASSOCIATIONS.scope.eq("all_selected")].merge(
    assoc_full[["run","metric","paired_basins","pearson_r","spearman_rho"]].rename(columns={
        "paired_basins":"all_paired_basins","pearson_r":"all_pearson_r","spearman_rho":"all_spearman_rho"}),
    on=["run","metric"],validate="many_to_one")
A1_NSE_LABEL_SENSITIVITY["delta_pearson_r"]=A1_NSE_LABEL_SENSITIVITY.pearson_r-A1_NSE_LABEL_SENSITIVITY.all_pearson_r
A1_NSE_LABEL_SENSITIVITY["delta_spearman_rho"]=A1_NSE_LABEL_SENSITIVITY.spearman_rho-A1_NSE_LABEL_SENSITIVITY.all_spearman_rho

fig,axes=plt.subplots(1,2,figsize=(18,7))
for ax,column,title in zip(axes,["pearson_r","spearman_rho"],["Pearson r","Spearman rho"]):
    values=assoc_full.pivot(index="run",columns="metric",values=column).reindex(index=assoc_runs,columns=assoc_metrics)
    counts=assoc_full.pivot(index="run",columns="metric",values="paired_basins").reindex(index=assoc_runs,columns=assoc_metrics)
    im=ax.imshow(np.ma.masked_invalid(values.to_numpy()),vmin=-1,vmax=1,cmap="RdBu_r",aspect="auto")
    for i in range(len(assoc_runs)):
        for j in range(len(assoc_metrics)):
            value=values.iloc[i,j];label=f"{value:.2f}\nn={int(counts.iloc[i,j])}" if np.isfinite(value) else f"undefined\nn={int(counts.iloc[i,j])}"
            ax.text(j,i,label,ha="center",va="center",fontsize=8,color="white" if np.isfinite(value) and abs(value)>.6 else "black")
    ax.set_xticks(range(len(assoc_metrics)),list(assoc_metrics.values()),rotation=45,ha="right",fontsize=9)
    ax.set_yticks(range(len(assoc_runs)),assoc_runs,fontsize=10);ax.set_title(title)
fig.subplots_adjust(left=.08,right=.92,top=.86,bottom=.33,wspace=.24)
color_ax=fig.add_axes([.94,.34,.012,.5]);fig.colorbar(im,cax=color_ax,label="Correlation")
fig.suptitle("Author-packaged test NSE versus current CSV label characteristics",fontsize=15,y=.97)
fig.text(.5,.055,"Each cell uses finite feature/NSE pairs from the selected basin set; negative NSE is retained. No clipping or model selection.\n"
    "Author CSV metrics are not A2 rerun exports; config dates match, but actual scoring dates/observations are not verified here.\n"
    "Associations are descriptive; NSE includes variance normalization. Nested basins are not assumed independent. Sensitivity scopes are saved separately.",ha="center",fontsize=10)
assoc_figure=FIGURE_DIR/"A1_P8_author_nse_label_associations.png"
fig.savefig(assoc_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("author_nse_label_join",A1_NSE_LABEL_JOIN),("author_nse_label_coverage",A1_NSE_LABEL_COVERAGE),
    ("author_nse_label_associations",A1_NSE_LABEL_ASSOCIATIONS),("author_nse_label_sensitivity",A1_NSE_LABEL_SENSITIVITY)]:
    table.to_csv(TABLE_DIR/f"A1_P8_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Ghép đúng sáu CSV tác giả; NSE không hữu hạn giữ riêng, không gọi là kết quả chạy lại A2:");display(A1_NSE_LABEL_COVERAGE)
    print("(2) Tương quan ngang ID trên từng cặp hữu hạn; số thiếu NSE/đặc điểm không đếm lặp:")
    display(assoc_full[["run","metric","scope_basins","paired_basins","nonfinite_nse_basins","nonfinite_feature_with_finite_nse","pearson_r","spearman_rho","status"]])
    print("(3) Độ nhạy tại các hồ sơ đã biết; chênh hệ số = phạm vi đối chiếu trừ toàn bộ, không đổi danh sách mô hình:")
    display(A1_NSE_LABEL_SENSITIVITY.loc[A1_NSE_LABEL_SENSITIVITY.metric.isin(["test_std_mm_day","test_std_ratio_to_train","test_w1_to_train_mm_day"]),
        ["run","scope","metric","all_paired_basins","paired_basins","pearson_r","spearman_rho","delta_pearson_r","delta_spearman_rho","status"]])
    print("(4) Hồ sơ nhãn/NSE tác giả theo ID; không suy nguyên nhân hoặc thay NaN:")
    display(A1_NSE_LABEL_JOIN.loc[A1_NSE_LABEL_JOIN.basin_id.isin(["55","324","600","623","742","758","800"]),
        ["run","basin_id","nse","train_valid_days","train_month_types_observed","test_valid_days","test_std_mm_day","test_std_ratio_to_train","test_w1_to_train_mm_day","test_above_p99_pct"]])
print("Đã lưu hình:",assoc_figure,"; bốn CSV vào:",TABLE_DIR)
print(f"Thời gian ghép mô tả NSE tác giả: {time.perf_counter()-started_nse_associations:.1f} giây.")
print("CẢNH BÁO: đây là NSE tác giả, không phải export chạy lại A2; tương quan dùng đặc điểm CSV hiện tại, chưa xác nhận cùng tập ngày chấm hoặc quan hệ nhân quả.")

# %% [markdown]
# ## Phần 9. Hồ sơ bốn lưu vực chẩn đoán
#
# Đọc chuỗi tác giả của 55/742/758/800 để kiểm mặt nạ, đầu ra hằng/NaN và thành phần sai số; không chạy hoặc hiệu chỉnh mô hình.

# %% [markdown]
# ### 9.1. Đối chiếu chuỗi tác giả ở bốn hồ sơ 55/742/758/800
#
# Đọc lần lượt sáu test_results.p đúng cạnh CSV best_model đã kiểm; mỗi lần chỉ giữ chuỗi bốn ID để hạn chế RAM.

# %% Phần 9.1 - Đối chiếu chuỗi tác giả ở bốn hồ sơ 55/742/758/800
# Đọc lần lượt sáu test_results.p đúng cạnh CSV best_model đã kiểm; mỗi lần chỉ giữ chuỗi bốn ID để hạn chế RAM.
# So ngày/obs với cache CSV, giữ sim gốc và tính thêm bản chấm theo clip_targets_to_zero khai báo; không chạy mô hình hoặc sửa dữ liệu.
import pickle
import gc

started_profiles=time.perf_counter()
profile_ids=["55","742","758","800"]
profile_dates=pd.date_range(*PERIODS["test"],freq="D")
profile_cache=pd.DataFrame(np.asarray(A1_QMAX_VALUES_MM_DAY,dtype=float).T,
    index=pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR),columns=list(map(str,A1_QMAX_VALUE_IDS))).reindex(profile_dates)[profile_ids]
profile_audit=A1_NSE_FILE_AUDIT.loc[A1_NSE_FILE_AUDIT.source.eq("author_packaged")].set_index("run",verify_integrity=True)
if set(profile_audit.index)!=set(A1_A2_RUN_MAP) or not profile_audit.status.eq("CSV_read").all():
    raise ValueError("Cần đủ sáu run tác giả đã đọc ở 8.3.")
profile_metrics=A1_NSE_VALUES.loc[A1_NSE_VALUES.source.eq("author_packaged")].set_index(["run","basin_id"],verify_integrity=True)
profile_file_rows,profile_score_rows,profile_daily_parts=[],[],[]


def profile_hash_file(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def profile_extract_daily(item):
    dataset=item["1D"]["xr"]
    obs=dataset["qmax_obs"];sim=dataset["qmax_sim"]
    if set(obs.dims)!={"date","time_step"} or set(sim.dims)!={"date","time_step"}:
        raise ValueError("Chỉ nhận chuỗi điểm qmax theo date/time_step, chưa tự gộp samples hoặc chiều khác.")
    steps=np.asarray(dataset.time_step.values)
    if not len(steps) or len(np.unique(steps))!=len(steps) or steps[-1]!=0:
        raise ValueError("time_step cuối chưa xác nhận là 0 như bộ chấm ngày; không tự dịch ngày.")
    dates=pd.DatetimeIndex(dataset.date.values)
    if (dates.tz is not None or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
        or not dates.equals(dates.normalize())):
        raise ValueError("Ngày không duy nhất/tăng dần/đúng mốc ngày hoặc có múi giờ; chưa tự chuẩn hóa.")
    frame=pd.DataFrame({"obs_author":np.asarray(obs.isel(time_step=-1).values,dtype=float),
        "sim_raw":np.asarray(sim.isel(time_step=-1).values,dtype=float)},index=dates)
    if np.isinf(frame.to_numpy()).any():raise ValueError("Chuỗi có vô hạn; cần kiểm riêng trước khi tính sai số.")
    return frame,len(steps)


for run,relative in A1_A2_RUN_MAP.items():
    row=profile_audit.loc[run];metric_path=Path(row.metric_file)
    expected=EXP_DIR/relative/"test"/"best_model"/"test_metrics.csv"
    if (metric_path.resolve()!=expected.resolve() or profile_hash_file(metric_path)!=row.sha256
        or row.scope_status!="declared_qmax_test_matches_A1"):
        raise ValueError(f"{run}: nguồn metrics/scope đổi sau 8.3.")
    result_path=metric_path.with_name("test_results.p")
    file_row=dict(source="author_packaged",run=run,file=str(result_path),status="not_found",bytes=np.nan,sha256="")
    if not result_path.is_file():
        profile_file_rows.append(file_row);continue
    cfg=a1_load_yaml(Path(row.config_file).read_text(encoding="utf-8-sig"))
    if (sample_input_names(cfg.get("target_variables",[]),"target_variables")!=["qmax"]
        or (sample_date(cfg.get("test_start_date")),sample_date(cfg.get("test_end_date")))!=(profile_dates[0],profile_dates[-1])):
        raise ValueError(f"{run}: config target/ngày test đổi sau 8.3.")
    clip_qmax="qmax" in sample_input_names(cfg.get("clip_targets_to_zero",[]),"clip_targets_to_zero")
    file_row.update(bytes=result_path.stat().st_size,sha256=profile_hash_file(result_path),clip_qmax_declared=clip_qmax)
    results=None
    try:
        with result_path.open("rb") as stream:results=pickle.load(stream)
        if not isinstance(results,dict):raise ValueError("Kết quả không phải dict basin.")
        keys={str(key):key for key in results}
        if len(keys)!=len(results):raise ValueError("Khóa basin không duy nhất sau chuyển thành chuỗi.")
        file_row.update(status="read",basin_keys=len(keys))
        for basin_id in profile_ids:
            score=dict(source="author_packaged",run=run,basin_id=basin_id,status="basin_not_found",clip_qmax_declared=clip_qmax)
            if basin_id not in keys:profile_score_rows.append(score);continue
            try:
                raw,steps=profile_extract_daily(results[keys[basin_id]])
                outside=int((~raw.index.isin(profile_dates)).sum());daily=raw.reindex(profile_dates).copy()
                daily["result_row_present"]=profile_dates.isin(raw.index)
                daily["qmax_csv"]=profile_cache[basin_id]
                daily["sim_evaluated"]=daily.sim_raw.clip(lower=0) if clip_qmax else daily.sim_raw
                csv_ok=np.isfinite(daily.qmax_csv);obs_ok=np.isfinite(daily.obs_author);sim_ok=np.isfinite(daily.sim_evaluated)
                both=csv_ok&obs_ok;paired=obs_ok&sim_ok
                delta=(daily.obs_author-daily.qmax_csv).abs()
                o=daily.loc[paired,"obs_author"].to_numpy();p=daily.loc[paired,"sim_evaluated"].to_numpy()
                e=p-o;variance=float(np.var(o,ddof=0)) if len(o) else np.nan
                mse=float(np.mean(e*e)) if len(o) else np.nan
                bias=float(np.mean(e)) if len(o) else np.nan
                nse=1-mse/variance if len(o)>=2 and variance>0 else np.nan
                stored=float(profile_metrics.loc[(run,basin_id),"nse"])
                raw_mse=float(np.mean((daily.loc[paired,"sim_raw"].to_numpy()-o)**2)) if len(o) else np.nan
                score.update(status="compared",time_steps=steps,result_days=len(raw),rows_outside_test=outside,
                    test_days=len(profile_dates),missing_result_rows=int((~daily.result_row_present).sum()),
                    finite_obs_days=int(obs_ok.sum()),finite_csv_days=int(csv_ok.sum()),finite_sim_days=int(sim_ok.sum()),
                    paired_days=int(paired.sum()),obs_csv_common_days=int(both.sum()),
                    obs_csv_missing_mask_difference_days=int((obs_ok!=csv_ok).sum()),
                    obs_csv_max_abs_difference_mm_day=float(delta[both].max()) if both.any() else np.nan,
                    obs_csv_within_1e_7_days=int((delta[both]<=1e-7).sum()),obs_csv_within_1e_5_days=int((delta[both]<=1e-5).sum()),
                    negative_sim_raw_days=int((daily.sim_raw<0).sum()),
                    obs_std_mm_day=float(np.sqrt(variance)),mean_error_mm_day=bias,rmse_mm_day=float(np.sqrt(mse)),
                    error_variance_mm_day2=float(np.var(e,ddof=0)) if len(e) else np.nan,
                    mean_error_squared_over_obs_variance=bias*bias/variance if variance>0 else np.nan,
                    error_variance_over_obs_variance=float(np.var(e,ddof=0))/variance if variance>0 else np.nan,
                    nse_from_raw_sim=1-raw_mse/variance if variance>0 else np.nan,nse_recomputed_float64=nse,nse_csv=stored,
                    abs_nse_difference=abs(nse-stored) if np.isfinite(nse) and np.isfinite(stored) else np.nan,
                    nse_finite_state_matches=bool(np.isfinite(nse)==np.isfinite(stored)))
                if len(o) and not np.isclose(mse,bias*bias+np.var(e,ddof=0),rtol=1e-10,atol=1e-12):
                    raise AssertionError("Phân rã MSE không cộng khớp bình phương mean error và phương sai sai số.")
                daily["run"]=run;daily["basin_id"]=basin_id;daily.index.name="date"
                profile_daily_parts.append(daily.reset_index())
            except (KeyError,ValueError,TypeError) as exc:score["status"]="schema_error: "+str(exc)[:180]
            profile_score_rows.append(score)
    except (ValueError,TypeError,KeyError,pickle.UnpicklingError,EOFError) as exc:
        file_row["status"]="read_error: "+str(exc)[:180]
    finally:
        del results;gc.collect()
    profile_file_rows.append(file_row)
    print(f"Đã đọc hồ sơ chuỗi tác giả: {run}",flush=True)
A1_AUTHOR_PROFILE_FILES=pd.DataFrame(profile_file_rows)
A1_AUTHOR_PROFILE_SCORES=pd.DataFrame(profile_score_rows)
A1_AUTHOR_PROFILE_DAILY=pd.concat(profile_daily_parts,ignore_index=True) if profile_daily_parts else pd.DataFrame()
if len(A1_AUTHOR_PROFILE_DAILY):
    fig,axes=plt.subplots(4,1,figsize=(15,12),sharex=True)
    for ax,basin_id in zip(axes,profile_ids):
        ax.plot(profile_dates,profile_cache[basin_id],color="black",linewidth=.8,label="Current CSV qmax")
        for run,color in [("SeqLSTM_Q","#c87531"),("SeqLSTM_noQ","#2283a5")]:
            part=A1_AUTHOR_PROFILE_DAILY.loc[A1_AUTHOR_PROFILE_DAILY.run.eq(run)&A1_AUTHOR_PROFILE_DAILY.basin_id.eq(basin_id)]
            if len(part):
                ax.plot(part.date,part.sim_evaluated,color=color,linewidth=.7,alpha=.8,label=run+" (config evaluation)")
                if not np.isfinite(part.sim_evaluated).any():ax.text(.01,.9,run+": no finite simulation",transform=ax.transAxes,fontsize=9)
        ax.set_title("Basin "+basin_id,fontsize=11);ax.set_ylabel("qmax (mm/day)");ax.grid(alpha=.2);ax.legend(loc="upper right",fontsize=8)
    axes[-1].set_xlabel("Stored daily date (no time shift)")
    fig.suptitle("Four diagnostic basins: current CSV labels and author-packaged predictions",fontsize=14)
    fig.text(.5,.025,"Plots show two fixed runs; tables audit all six runs. Each panel has its own vertical scale.\n"
        "Evaluation simulations follow only the declared clip_targets_to_zero policy; raw simulations are saved separately.\n"
        "Source: author best_model archives, not A2 rerun exports. Observation agreement and scoring coverage are reported per run/basin.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.095,1,.95))
    profile_figure=FIGURE_DIR/"A1_P9_author_profile_series.png"
    fig.savefig(profile_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
    print("Đã lưu hình:",profile_figure)
for suffix,table in [("author_profile_files",A1_AUTHOR_PROFILE_FILES),("author_profile_scores",A1_AUTHOR_PROFILE_SCORES),
    ("author_profile_daily",A1_AUTHOR_PROFILE_DAILY)]:table.to_csv(TABLE_DIR/f"A1_P9_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Đúng tệp chuỗi tác giả cạnh best_model metrics; trạng thái schema/path:");display(A1_AUTHOR_PROFILE_FILES)
    print("(2) Đối chiếu ngày, obs và mask với CSV hiện tại; dung sai chỉ mô tả sai khác số học:")
    fields=["run","basin_id","status","time_steps","rows_outside_test","missing_result_rows","finite_obs_days","finite_csv_days","finite_sim_days","paired_days",
        "obs_csv_missing_mask_difference_days","obs_csv_max_abs_difference_mm_day","obs_csv_within_1e_7_days","obs_csv_within_1e_5_days"]
    display(A1_AUTHOR_PROFILE_SCORES.reindex(columns=fields))
    print("(3) Sai số trên đúng obs/sim hữu hạn; mean error = sim − obs; không dùng CSV obs thay vào NSE:")
    fields=["run","basin_id","status","paired_days","clip_qmax_declared","negative_sim_raw_days","obs_std_mm_day","mean_error_mm_day","rmse_mm_day",
        "mean_error_squared_over_obs_variance","error_variance_over_obs_variance","nse_from_raw_sim","nse_recomputed_float64","nse_csv","abs_nse_difference","nse_finite_state_matches"]
    display(A1_AUTHOR_PROFILE_SCORES.reindex(columns=fields))
print("Đã lưu ba CSV vào:",TABLE_DIR)
print(f"Thời gian đối chiếu bốn hồ sơ chuỗi tác giả: {time.perf_counter()-started_profiles:.1f} giây.")
print("CẢNH BÁO: float64 có thể lệch số học so NSE lưu; không thay quan trắc, tự đổi đơn vị/ngày hoặc cắt sim ngoài chính sách config. Khớp cùng nguồn không xác nhận độc lập quan trắc vật lý.")

# %% [markdown]
# ### 9.2. Độ biến thiên dự báo, ảnh hưởng chặn âm và sai khác obs ở bốn hồ sơ
#
# Dùng chuỗi/cache đã đọc ở 9.1, kiểm số cặp/sai số khớp lại; không đọc lại archive hoặc chạy mô hình.

# %% Phần 9.2 - Độ biến thiên dự báo, ảnh hưởng chặn âm và sai khác obs ở bốn hồ sơ
# Dùng chuỗi/cache đã đọc ở 9.1, kiểm số cặp/sai số khớp lại; không đọc lại archive hoặc chạy mô hình.
# Đếm mức sim chính xác và tách ngày có/không obs; giữ sai khác vượt dung sai, không tự nâng dung sai hoặc sửa obs.
started_profile_variation=time.perf_counter()
variation_daily=A1_AUTHOR_PROFILE_DAILY.copy(deep=True)
variation_scores=A1_AUTHOR_PROFILE_SCORES.set_index(["run","basin_id"],verify_integrity=True)
variation_keys={(run,basin_id) for run in A1_A2_RUN_MAP for basin_id in profile_ids}
if (variation_daily.duplicated(["run","basin_id","date"]).any() or
    set(zip(variation_daily.run,variation_daily.basin_id))!=variation_keys or
    set(variation_scores.index)!=variation_keys or not variation_scores.status.eq("compared").all()):
    raise ValueError("Cần đủ 24 hồ sơ đã đối chiếu 9.1; chưa kiểm độ biến thiên.")
variation_rows,variation_obs_rows=[],[]
for (run,basin_id),part in variation_daily.groupby(["run","basin_id"],sort=False):
    score=variation_scores.loc[(run,basin_id)]
    dates=pd.DatetimeIndex(part.date)
    if not dates.equals(profile_dates):raise ValueError(f"{run}/{basin_id}: lịch khác 9.1.")
    obs=part.obs_author.to_numpy(dtype=float);raw=part.sim_raw.to_numpy(dtype=float)
    evaluated=part.sim_evaluated.to_numpy(dtype=float);csv=part.qmax_csv.to_numpy(dtype=float)
    if np.isinf(np.column_stack([obs,raw,evaluated,csv])).any():raise ValueError("Chuỗi có vô hạn sau 9.1.")
    expected=np.maximum(raw,0) if bool(score.clip_qmax_declared) else raw
    if not np.array_equal(expected,evaluated,equal_nan=True):raise ValueError("sim_evaluated khác chính sách config đã kiểm.")
    sim_ok=np.isfinite(raw);obs_ok=np.isfinite(obs);paired=sim_ok&obs_ok
    if int(paired.sum())!=score.paired_days:raise ValueError("Số cặp khác 9.1.")
    if paired.any():
        o=obs[paired];e=evaluated[paired]-o
        if not np.isclose(np.mean(e*e),score.rmse_mm_day**2,rtol=1e-10,atol=1e-12):raise ValueError("MSE khác 9.1.")
    row=dict(source="author_packaged",run=run,basin_id=basin_id,test_days=len(part),
        finite_sim_days=int(sim_ok.sum()),paired_days=int(paired.sum()),
        negative_raw_all_days=int((raw<0).sum()),negative_raw_paired_days=int(((raw<0)&paired).sum()),
        negative_raw_without_obs_days=int(((raw<0)&~obs_ok).sum()),
        zero_evaluated_paired_days=int(((evaluated==0)&paired).sum()),
        mse_fraction_from_mean_error=score.mean_error_squared_over_obs_variance/(1-score.nse_recomputed_float64)
            if np.isfinite(score.nse_recomputed_float64) and score.nse_recomputed_float64<1 else np.nan,
        nse_gain_from_declared_clipping=score.nse_recomputed_float64-score.nse_from_raw_sim)
    for side,values in [("raw",raw[sim_ok]),("evaluated",evaluated[np.isfinite(evaluated)])]:
        row.update({side+"_distinct_values":len(np.unique(values)),side+"_constant":bool(len(values)>0 and np.ptp(values)==0),
            side+"_min":float(values.min()) if len(values) else np.nan,side+"_max":float(values.max()) if len(values) else np.nan,
            side+"_std_ddof0":(0.0 if np.ptp(values)==0 else float(np.std(values,ddof=0))) if len(values) else np.nan})
    variation_rows.append(row)
    both=obs_ok&np.isfinite(csv)
    difference=np.abs(obs-csv)
    # Lưu ba sai khác lớn nhất mỗi hồ sơ, không chọn theo NSE. Đếm vượt ngưỡng giữ đầy đủ ở bảng tóm tắt.
    positions=np.flatnonzero(both);positions=positions[np.argsort(-difference[positions],kind="stable")[:3]]
    for pos in positions:
        variation_obs_rows.append(dict(source="author_packaged",run=run,basin_id=basin_id,date=dates[pos],
            obs_author_mm_day=obs[pos],qmax_csv_mm_day=csv[pos],abs_difference_mm_day=difference[pos],
            exceeds_1e_5=bool(difference[pos]>1e-5),
            abs_difference_from_direct_csv_float32=float(abs(obs[pos]-float(np.float32(csv[pos]))))))
A1_AUTHOR_PROFILE_VARIATION=pd.DataFrame(variation_rows)
A1_AUTHOR_PROFILE_OBS_EXAMPLES=pd.DataFrame(variation_obs_rows)
variation_pair_rows=[]
for run in A1_A2_RUN_MAP:
    left=variation_daily.loc[variation_daily.run.eq(run)&variation_daily.basin_id.eq("742")].set_index("date",verify_integrity=True)
    right=variation_daily.loc[variation_daily.run.eq(run)&variation_daily.basin_id.eq("758")].set_index("date",verify_integrity=True)
    for side in ["sim_raw","sim_evaluated"]:
        x=left[side].to_numpy();y=right[side].to_numpy();ok=np.isfinite(x)&np.isfinite(y);delta=x[ok]-y[ok]
        varying=len(delta)>=3 and np.ptp(x[ok])>0 and np.ptp(y[ok])>0
        variation_pair_rows.append(dict(source="author_packaged",run=run,side=side,common_finite_days=int(ok.sum()),
            identical_value_days=int((delta==0).sum()),mean_abs_difference_mm_day=float(np.abs(delta).mean()) if len(delta) else np.nan,
            max_abs_difference_mm_day=float(np.abs(delta).max()) if len(delta) else np.nan,
            pearson_r=float(np.corrcoef(x[ok],y[ok])[0,1]) if varying else np.nan,
            correlation_status="defined" if varying else "constant_or_fewer_than_3_pairs"))
A1_AUTHOR_PROFILE_PAIR_SIMILARITY=pd.DataFrame(variation_pair_rows)

fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,basin_id in zip(axes,["742","758"]):
    part=variation_scores.xs(basin_id,level="basin_id").reindex(list(A1_A2_RUN_MAP))
    mean_part=part.mean_error_squared_over_obs_variance;var_part=part.error_variance_over_obs_variance
    x=np.arange(len(part))
    ax.bar(x,mean_part,color="#c87531",label="Squared mean error / obs variance")
    ax.bar(x,var_part,bottom=mean_part,color="#2283a5",label="Error variance / obs variance")
    ax.set_xticks(x,part.index,rotation=30,ha="right");ax.set_title("Basin "+basin_id)
    ax.set_ylabel("MSE / observation variance = 1 - NSE");ax.grid(axis="y",alpha=.2)
fig.suptitle("Author-prediction error components for two low-variance basins",fontsize=14)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.14),ncol=2,fontsize=9)
fig.text(.5,.035,"Components use each run's finite author-observation / evaluated-simulation pairs. No correction is fitted or applied.\n"
    "Evaluation follows only the original config clipping policy. Components explain the metric algebra, not the physical cause of prediction errors.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.22,1,.94));variation_figure=FIGURE_DIR/"A1_P9_author_profile_error_components.png"
fig.savefig(variation_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("author_profile_variation",A1_AUTHOR_PROFILE_VARIATION),("author_profile_obs_examples",A1_AUTHOR_PROFILE_OBS_EXAMPLES),
    ("author_profile_pair_similarity",A1_AUTHOR_PROFILE_PAIR_SIMILARITY)]:table.to_csv(TABLE_DIR/f"A1_P9_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Số mức và biên dự báo trên ngày sim hữu hạn; constant chỉ đúng khi max=min chính xác:")
    display(A1_AUTHOR_PROFILE_VARIATION[["run","basin_id","finite_sim_days","paired_days","raw_distinct_values","raw_constant","raw_min","raw_max","raw_std_ddof0",
        "evaluated_distinct_values","evaluated_constant","evaluated_min","evaluated_max","evaluated_std_ddof0"]])
    print("(2) Dự báo âm trong/ngoài tập ngày có obs; phần mean error là tỷ phần của MSE, không phải tỷ phần nguyên nhân:")
    display(A1_AUTHOR_PROFILE_VARIATION[["run","basin_id","negative_raw_all_days","negative_raw_paired_days","negative_raw_without_obs_days",
        "zero_evaluated_paired_days","mse_fraction_from_mean_error","nse_gain_from_declared_clipping"]])
    print("(3) Hai chuỗi dự báo 742/758 trên cùng ngày; hằng số giữ tương quan NaN:");display(A1_AUTHOR_PROFILE_PAIR_SIMILARITY)
    print("(4) Ví dụ obs/CSV có sai khác >1e-5 mm/ngày; không bỏ hoặc nâng dung sai. Ba sai khác lớn nhất mỗi hồ sơ lưu CSV:")
    display(A1_AUTHOR_PROFILE_OBS_EXAMPLES.loc[A1_AUTHOR_PROFILE_OBS_EXAMPLES.exceeds_1e_5])
print("Đã lưu hình:",variation_figure,"; ba CSV vào:",TABLE_DIR)
print(f"Thời gian kiểm độ biến thiên hồ sơ: {time.perf_counter()-started_profile_variation:.1f} giây.")
print("CẢNH BÁO: khớp mức dự báo hoặc chuỗi hằng chưa xác nhận nguyên nhân mô hình. Ép CSV sang float32 trực tiếp chỉ là đối chiếu số học, không tái tạo đầy đủ chuẩn hóa/khôi phục nhãn tác giả.")

# %% [markdown]
# ## Phần 10. Phân mức qmax, ngưỡng Gumbel và gộp đoạn vượt
#
# Dùng tham chiếu 1981–2013, kiểm độ phủ/độ nhạy rồi đếm ngày, đoạn và cụm theo quy tắc cố định trên quan trắc.

# %% [markdown]
# ### 10.1. Ngưỡng bốn mức qmax từ quan trắc 1981–2013
#
# Theo kế hoạch Mục 6.2: ngưỡng phân vị 30/80/98 riêng từng ID, chỉ từ giai đoạn trước test; khác p95/p99 train chẩn đoán ở Phần 8.

# %% Phần 10.1 - Ngưỡng bốn mức qmax từ quan trắc 1981–2013
# Theo kế hoạch Mục 6.2: ngưỡng phân vị 30/80/98 riêng từng ID, chỉ từ giai đoạn trước test; khác p95/p99 train chẩn đoán ở Phần 8.
# Bằng ngưỡng vào mức cao hơn theo quy ước tài liệu; giữ số 0, báo ngưỡng trùng và thiếu, chưa chấm dự báo hoặc fit Gumbel.
started_levels=time.perf_counter()
A1_LEVEL_REFERENCE_PERIOD=("1981-01-01","2013-12-31")
A1_LEVEL_NAMES=["low","medium","high","very_high"]
level_quantiles=[.30,.80,.98]
level_threshold_fields=["q30_mm_day","q80_mm_day","q98_mm_day"]
level_ids=list(map(str,A1_QMAX_VALUE_IDS))
level_calendar=pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
level_values=np.asarray(A1_QMAX_VALUES_MM_DAY,dtype=float)
if (len(level_ids)!=len(set(level_ids)) or set(level_ids)!=set(map(str,selected_ids))
    or level_values.shape!=(len(level_ids),len(level_calendar)) or level_calendar.has_duplicates
    or not level_calendar.equals(pd.date_range(level_calendar.min(),level_calendar.max(),freq="D"))
    or np.isinf(level_values).any() or (level_values<0).any()):
    raise ValueError("Cache qmax không còn đúng ID/lịch hoặc có giá trị ngoài phạm vi đã kiểm.")
if pd.Timestamp(A1_LEVEL_REFERENCE_PERIOD[1])>=pd.Timestamp(PERIODS["test"][0]):
    raise ValueError("Giai đoạn fit ngưỡng phải kết thúc trước test.")
level_periods={"reference_1981_2013":A1_LEVEL_REFERENCE_PERIOD,**PERIODS}
level_positions={}
for period,(start,end) in level_periods.items():
    pos=np.flatnonzero((level_calendar>=start)&(level_calendar<=end))
    if len(pos)!=len(pd.date_range(start,end,freq="D")):raise ValueError(f"Cache chưa bao trọn {period}.")
    level_positions[period]=pos


def four_level_classify(values,bounds):
    values=np.asarray(values,dtype=float);bounds=np.asarray(bounds,dtype=float)
    if bounds.shape!=(3,) or not np.isfinite(bounds).all() or (np.diff(bounds)<0).any() or not np.isfinite(values).all():
        raise ValueError("Phân mức cần giá trị hữu hạn và ba ngưỡng hữu hạn không giảm.")
    # low: q<q30; medium: q30≤q<q80; high: q80≤q<q98; very_high: q≥q98.
    return np.searchsorted(bounds,values,side="right")


level_threshold_rows,level_count_rows=[],[]
for basin_id,series in zip(level_ids,level_values):
    ref_pos=level_positions["reference_1981_2013"];ref_raw=series[ref_pos]
    ref=ref_raw[np.isfinite(ref_raw)];ref_dates=level_calendar[ref_pos[np.isfinite(ref_raw)]]
    bounds=np.quantile(ref,level_quantiles,method="linear") if len(ref) else np.full(3,np.nan)
    threshold_row=dict(basin_id=basin_id,reference_start=A1_LEVEL_REFERENCE_PERIOD[0],reference_end=A1_LEVEL_REFERENCE_PERIOD[1],
        expected_reference_days=len(ref_raw),valid_reference_days=len(ref),missing_reference_days=int(np.isnan(ref_raw).sum()),
        zero_reference_days=int((ref==0).sum()),reference_month_types=len(set(ref_dates.month)),reference_years=len(set(ref_dates.year)),
        first_reference_observation=ref_dates.min() if len(ref_dates) else pd.NaT,last_reference_observation=ref_dates.max() if len(ref_dates) else pd.NaT,
        status="no_reference_observations" if not len(ref) else "collapsed_thresholds" if len(np.unique(bounds))<3 else "defined_distinct_thresholds",
        distinct_thresholds=len(np.unique(bounds)) if len(ref) else 0,**dict(zip(level_threshold_fields,bounds)))
    level_threshold_rows.append(threshold_row)
    for period,pos in level_positions.items():
        raw=series[pos];valid=raw[np.isfinite(raw)];classified=four_level_classify(valid,bounds) if len(ref) else np.empty(0,dtype=int)
        counts=np.bincount(classified,minlength=4)
        unclassified=len(valid) if not len(ref) else 0
        if int(counts.sum())+unclassified+int(np.isnan(raw).sum())!=len(raw):raise AssertionError("Phân mức không cộng khớp lịch.")
        row=dict(basin_id=basin_id,period=period,expected_days=len(raw),valid_days=len(valid),missing_days=int(np.isnan(raw).sum()),
            classified_days=int(counts.sum()),valid_days_without_threshold=unclassified,
            threshold_status=threshold_row["status"],valid_reference_days=len(ref))
        for name,count in zip(A1_LEVEL_NAMES,counts):
            row[name+"_days"]=int(count);row[name+"_pct_classified"]=100*count/counts.sum() if counts.sum() else np.nan
        for field,bound in zip(level_threshold_fields,bounds):
            row["equal_"+field+"_days"]=int((valid==bound).sum()) if len(ref) else np.nan
        level_count_rows.append(row)
A1_LEVEL_THRESHOLDS=pd.DataFrame(level_threshold_rows)
A1_LEVEL_COUNTS_BY_BASIN=pd.DataFrame(level_count_rows)
# Kiểm số ngày có nhãn bằng 8.1 trên từng ID/kỳ; reference chứa cả train/validation nên không cộng lặp các phạm vi.
level_check=A1_LEVEL_COUNTS_BY_BASIN.loc[~A1_LEVEL_COUNTS_BY_BASIN.period.eq("reference_1981_2013")].merge(
    A1_Q_PERIOD_PROFILE[["basin_id","period","expected_days","valid_days","missing_days"]],on=["basin_id","period"],validate="one_to_one",suffixes=("","_8_1"))
if len(level_check)!=len(level_ids)*len(PERIODS) or any(not level_check[field].eq(level_check[field+"_8_1"]).all() for field in ["expected_days","valid_days","missing_days"]):
    raise AssertionError("Số nhãn phân mức khác từng ID/kỳ ở 8.1.")
level_summary_rows=[]
for period,part in A1_LEVEL_COUNTS_BY_BASIN.groupby("period",sort=False):
    classified=int(part.classified_days.sum())
    for name in A1_LEVEL_NAMES:
        days=int(part[name+"_days"].sum())
        level_summary_rows.append(dict(period=period,level=name,basins=len(part),basins_with_classified_days=int(part.classified_days.gt(0).sum()),
            basins_with_level_days=int(part[name+"_days"].gt(0).sum()),expected_days=int(part.expected_days.sum()),valid_days=int(part.valid_days.sum()),
            missing_days=int(part.missing_days.sum()),valid_days_without_threshold=int(part.valid_days_without_threshold.sum()),
            classified_days=classified,level_days=days,pooled_pct_classified=100*days/classified if classified else np.nan,
            median_basin_pct_classified=part[name+"_pct_classified"].median()))
A1_LEVEL_COUNTS_SUMMARY=pd.DataFrame(level_summary_rows)

fig,axes=plt.subplots(1,2,figsize=(15,6))
for field,color,label in zip(level_threshold_fields,["#2283a5","#c87531","#8655ad"],["q30 (low / medium)","q80 (medium / high)","q98 (high / very high)"]):
    values=A1_LEVEL_THRESHOLDS[field].dropna().sort_values().to_numpy()
    if len(values):axes[0].step(values,100*np.arange(1,len(values)+1)/len(values),where="post",color=color,label=f"{label}, n={len(values)}")
axes[0].set(xlabel="qmax threshold (mm/day)",ylabel="Basins at or below threshold (%)",title="Basin-specific pre-test thresholds",ylim=(0,102))
axes[0].set_xscale("symlog",linthresh=.1);axes[0].set_xlim(left=0);axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
level_colors=["#94c5df","#2283a5","#c87531","#8655ad"]
bottom=np.zeros(len(level_periods))
for name,color in zip(A1_LEVEL_NAMES,level_colors):
    part=A1_LEVEL_COUNTS_SUMMARY.loc[A1_LEVEL_COUNTS_SUMMARY.level.eq(name)].set_index("period").reindex(level_periods)
    values=part.pooled_pct_classified.to_numpy()
    axes[1].bar(np.arange(len(level_periods)),values,bottom=bottom,color=color,label=name.replace("_"," ").title());bottom+=values
axes[1].set_xticks(np.arange(len(level_periods)),["Reference\n1981–2013","Train","Validation","Test"])
axes[1].set(ylabel="Share of classified observed days (%)",title="Observed levels using the same thresholds",ylim=(0,105));axes[1].grid(axis="y",alpha=.2)
axes[1].legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.12),ncol=4)
fig.suptitle("Four daily-qmax levels: thresholds fitted only on observations from 1981–2013",fontsize=14)
fig.text(.5,.035,"Quantiles use linear interpolation; equality is assigned to the higher level. Zeros and tied values remain unchanged.\n"
    "Reference overlaps train/validation; these scopes must not be added together. Percentages use classified observed days only.\n"
    "Collapsed or absent thresholds are reported separately. These are flow levels, not independent floods or Gumbel return levels.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.2,1,.93));level_figure=FIGURE_DIR/"A1_P10_qmax_four_levels.png"
fig.savefig(level_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("qmax_level_thresholds",A1_LEVEL_THRESHOLDS),("qmax_level_counts_by_basin",A1_LEVEL_COUNTS_BY_BASIN),
    ("qmax_level_counts_summary",A1_LEVEL_COUNTS_SUMMARY)]:table.to_csv(TABLE_DIR/f"A1_P10_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Giai đoạn fit ngưỡng bốn mức, khác ngưỡng train ở Phần 8; số mức ngưỡng và độ phủ:")
    print("Reference:",A1_LEVEL_REFERENCE_PERIOD,"; bằng ngưỡng vào mức cao hơn; quantile method=linear")
    display(A1_LEVEL_THRESHOLDS.groupby("status",sort=False).agg(basins=("basin_id","size"),
        expected_days=("expected_reference_days","sum"),valid_days=("valid_reference_days","sum"),missing_days=("missing_reference_days","sum"),
        zero_days=("zero_reference_days","sum"),min_valid_days=("valid_reference_days","min"),max_valid_days=("valid_reference_days","max")).reset_index())
    print("(2) Số ngày bốn mức theo kỳ; phần trăm gộp ngày khác median từng ID:");display(A1_LEVEL_COUNTS_SUMMARY)
    print("(3) Độ phủ/ngưỡng ở các hồ sơ; không tự loại ID ít quan trắc:")
    display(A1_LEVEL_THRESHOLDS.loc[A1_LEVEL_THRESHOLDS.basin_id.isin(["55","324","600","623","742","758","800"])])
    print("(4) Ngưỡng trùng/không có; số ngày bằng từng ngưỡng được lưu CSV và có thể chồng nhau khi ngưỡng trùng:")
    display(A1_LEVEL_THRESHOLDS.loc[~A1_LEVEL_THRESHOLDS.status.eq("defined_distinct_thresholds")])
print("Đã lưu hình:",level_figure,"; ba CSV vào:",TABLE_DIR)
print(f"Thời gian dựng ngưỡng bốn mức: {time.perf_counter()-started_levels:.1f} giây.")
print("CẢNH BÁO: 1981–2013 gồm cả validation nhưng không dùng test; đây là giao thức ngưỡng đánh giá đã định trước, không phải scaler hay ngưỡng huấn luyện. Ties/độ phủ thiếu có thể làm tỷ lệ khác 30/50/18/2%.")

# %% [markdown]
# ### 10.2. Độ nhạy ngưỡng bốn mức giữa 1981–1997 và 1998–2013
#
# Hai khoảng đã định trong danh sách khảo sát; chỉ dùng ngày trước test, không chọn lại ngưỡng chính theo kết quả.

# %% Phần 10.2 - Độ nhạy ngưỡng bốn mức giữa 1981–1997 và 1998–2013
# Hai khoảng đã định trong danh sách khảo sát; chỉ dùng ngày trước test, không chọn lại ngưỡng chính theo kết quả.
# So mức trên cùng ngày tham chiếu 1981–2013; thiếu một nửa giữ NaN, không coi là ổn định hoặc thay bằng ngưỡng toàn kỳ.
started_level_stability=time.perf_counter()
A1_LEVEL_STABILITY_PERIODS={"early_1981_1997":("1981-01-01","1997-12-31"),"late_1998_2013":("1998-01-01","2013-12-31")}
stability_calendar=pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
stability_ids=list(map(str,A1_QMAX_VALUE_IDS));stability_values=np.asarray(A1_QMAX_VALUES_MM_DAY,dtype=float)
stability_official=A1_LEVEL_THRESHOLDS.set_index("basin_id",verify_integrity=True)
if (set(stability_ids)!=set(stability_official.index) or len(stability_ids)!=len(set(stability_ids))
    or set(stability_ids)!=set(map(str,selected_ids)) or stability_values.shape!=(len(stability_ids),len(stability_calendar))
    or stability_calendar.has_duplicates or not stability_calendar.equals(pd.date_range(stability_calendar.min(),stability_calendar.max(),freq="D"))
    or np.isinf(stability_values).any() or (stability_values<0).any()
    or tuple(A1_LEVEL_REFERENCE_PERIOD)!=("1981-01-01","2013-12-31")
    or not stability_official.reference_start.eq(A1_LEVEL_REFERENCE_PERIOD[0]).all()
    or not stability_official.reference_end.eq(A1_LEVEL_REFERENCE_PERIOD[1]).all()):
    raise ValueError("ID/cache/phạm vi ngưỡng chính khác 10.1.")
stability_positions={}
for name,(start,end) in {"reference":A1_LEVEL_REFERENCE_PERIOD,**A1_LEVEL_STABILITY_PERIODS}.items():
    pos=np.flatnonzero((stability_calendar>=start)&(stability_calendar<=end))
    if len(pos)!=len(pd.date_range(start,end,freq="D")):raise ValueError(f"Cache chưa bao trọn {name}.")
    if pd.Timestamp(end)>=pd.Timestamp(PERIODS["test"][0]):raise ValueError("Phép kiểm không được dùng ngày test.")
    stability_positions[name]=pos
stability_support_rows,stability_threshold_rows,stability_assignment_rows,stability_confusion_rows=[],[],[],[]
for basin_id,series in zip(stability_ids,stability_values):
    ref_raw=series[stability_positions["reference"]];reference=ref_raw[np.isfinite(ref_raw)]
    official=stability_official.loc[basin_id]
    full_bounds=np.quantile(reference,level_quantiles,method="linear") if len(reference) else np.full(3,np.nan)
    if len(reference)!=official.valid_reference_days or not np.array_equal(full_bounds,official[level_threshold_fields].to_numpy(dtype=float),equal_nan=True):
        raise ValueError(f"{basin_id}: cache/ngưỡng chính đổi sau 10.1.")
    support=dict(basin_id=basin_id);fits={}
    for label,name in [("early","early_1981_1997"),("late","late_1998_2013")]:
        pos=stability_positions[name];raw=series[pos];valid=raw[np.isfinite(raw)];dates=stability_calendar[pos[np.isfinite(raw)]]
        bounds=np.quantile(valid,level_quantiles,method="linear") if len(valid) else np.full(3,np.nan)
        fits[label]=bounds
        support.update({label+"_expected_days":len(raw),label+"_valid_days":len(valid),label+"_missing_days":int(np.isnan(raw).sum()),
            label+"_years_observed":len(set(dates.year)),label+"_month_types":len(set(dates.month)),
            label+"_distinct_thresholds":len(np.unique(bounds)) if len(valid) else 0})
    both=bool(support["early_valid_days"] and support["late_valid_days"])
    support["status"]="both_halves" if both else "early_only" if support["early_valid_days"] else "late_only" if support["late_valid_days"] else "no_reference_observations"
    stability_support_rows.append(support)
    for field,early,late,full in zip(level_threshold_fields,fits["early"],fits["late"],full_bounds):
        stability_threshold_rows.append(dict(basin_id=basin_id,threshold=field,status=support["status"],
            early_valid_days=support["early_valid_days"],late_valid_days=support["late_valid_days"],
            early_mm_day=early,late_mm_day=late,official_mm_day=full,
            delta_late_minus_early_mm_day=late-early if both else np.nan,
            ratio_late_to_early=late/early if both and early>0 else np.nan,
            ratio_status="defined" if both and early>0 else "early_threshold_zero" if both else "missing_half"))
    assignment=dict(basin_id=basin_id,status=support["status"],reference_valid_days=len(reference),compared_days=0,
        changed_days=np.nan,changed_pct=np.nan,very_high_switch_days=np.nan,early_very_high_days=np.nan,late_very_high_days=np.nan)
    if both:
        early_class=four_level_classify(reference,fits["early"]);late_class=four_level_classify(reference,fits["late"])
        confusion=np.bincount(4*early_class+late_class,minlength=16).reshape(4,4)
        changed=int((early_class!=late_class).sum());switched=int(((early_class==3)!=(late_class==3)).sum())
        if int(confusion.sum())!=len(reference) or len(reference)-int(np.trace(confusion))!=changed or switched>changed:
            raise AssertionError("Phân hoạch thay mức không cộng khớp.")
        assignment.update(compared_days=len(reference),changed_days=changed,changed_pct=100*changed/len(reference),
            very_high_switch_days=switched,early_very_high_days=int((early_class==3).sum()),late_very_high_days=int((late_class==3).sum()))
        for i,early_name in enumerate(A1_LEVEL_NAMES):
            for j,late_name in enumerate(A1_LEVEL_NAMES):
                stability_confusion_rows.append(dict(basin_id=basin_id,early_level=early_name,late_level=late_name,days=int(confusion[i,j])))
    stability_assignment_rows.append(assignment)
A1_LEVEL_STABILITY_SUPPORT=pd.DataFrame(stability_support_rows)
A1_LEVEL_STABILITY_THRESHOLDS=pd.DataFrame(stability_threshold_rows)
A1_LEVEL_STABILITY_ASSIGNMENTS=pd.DataFrame(stability_assignment_rows)
A1_LEVEL_STABILITY_CONFUSION=pd.DataFrame(stability_confusion_rows,columns=["basin_id","early_level","late_level","days"])
stability_summary_rows=[]
for field,part in A1_LEVEL_STABILITY_THRESHOLDS.groupby("threshold",sort=False):
    summary=dict(threshold=field,both_halves_basins=int(part.status.eq("both_halves").sum()),
        early_zero_threshold_basins=int(part.ratio_status.eq("early_threshold_zero").sum()),missing_half_basins=int(part.ratio_status.eq("missing_half").sum()))
    for label,column in [("delta","delta_late_minus_early_mm_day"),("ratio","ratio_late_to_early")]:
        values=part[column].to_numpy();values=values[np.isfinite(values)]
        summary[label+"_defined_basins"]=len(values)
        summary.update({label+"_"+name:value for name,value in zip(["p05","median","p95"],np.quantile(values,[.05,.5,.95],method="linear") if len(values) else [np.nan]*3)})
    stability_summary_rows.append(summary)
A1_LEVEL_STABILITY_THRESHOLD_SUMMARY=pd.DataFrame(stability_summary_rows)
stability_eligible=A1_LEVEL_STABILITY_ASSIGNMENTS.loc[A1_LEVEL_STABILITY_ASSIGNMENTS.status.eq("both_halves")]
stability_compared=int(stability_eligible.compared_days.sum());stability_changed=int(stability_eligible.changed_days.sum())
A1_LEVEL_STABILITY_ASSIGNMENT_SUMMARY=pd.DataFrame([dict(comparable_basins=len(stability_eligible),not_comparable_basins=len(stability_ids)-len(stability_eligible),
    compared_reference_days=stability_compared,changed_days=stability_changed,
    pooled_changed_pct=100*stability_changed/stability_compared if stability_compared else np.nan,
    median_basin_changed_pct=stability_eligible.changed_pct.median(),
    very_high_switch_days=stability_eligible.very_high_switch_days.sum(min_count=1))])

fig,axes=plt.subplots(1,2,figsize=(15,6))
for field,color,label in zip(level_threshold_fields,["#2283a5","#c87531","#8655ad"],["q30","q80","q98"]):
    values=A1_LEVEL_STABILITY_THRESHOLDS.loc[A1_LEVEL_STABILITY_THRESHOLDS.threshold.eq(field),"ratio_late_to_early"].dropna().sort_values().to_numpy()
    if len(values):axes[0].step(values,100*np.arange(1,len(values)+1)/len(values),where="post",color=color,label=f"{label}, n={len(values)}")
axes[0].set_xscale("symlog",linthresh=1);axes[0].set_xlim(left=0);axes[0].axvline(1,color="gray",linestyle=":")
axes[0].set(title="Threshold ratio: 1998–2013 / 1981–1997",xlabel="Late / early threshold (early > 0)",ylabel="Basins at or below ratio (%)",ylim=(0,102))
if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=9)
values=stability_eligible.changed_pct.sort_values().to_numpy()
if len(values):axes[1].step(values,100*np.arange(1,len(values)+1)/len(values),where="post",color="#2283a5",label=f"Both halves available, n={len(values)}");axes[1].legend(fontsize=9)
axes[1].set(title="Reference days assigned different levels",xlabel="Changed level on the same 1981–2013 observations (%)",ylabel="Basins at or below change (%)",xlim=(0,100),ylim=(0,102))
for ax in axes:ax.grid(alpha=.2)
fig.suptitle("Pre-test sensitivity of four qmax-level thresholds to the reference period",fontsize=14)
fig.text(.5,.035,"Official 1981–2013 thresholds remain unchanged. Each curve gives equal weight to basins with a defined comparison.\n"
    "Reclassification uses the same available reference observations, not independent validation. Missing halves and zero denominators remain undefined.\n"
    "Period differences combine observed variability and data coverage; they do not establish a physical trend or justify choosing thresholds using test data.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.18,1,.93));stability_figure=FIGURE_DIR/"A1_P10_qmax_level_stability.png"
fig.savefig(stability_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("level_stability_support",A1_LEVEL_STABILITY_SUPPORT),("level_stability_thresholds",A1_LEVEL_STABILITY_THRESHOLDS),
    ("level_stability_assignments",A1_LEVEL_STABILITY_ASSIGNMENTS),("level_stability_confusion",A1_LEVEL_STABILITY_CONFUSION),
    ("level_stability_threshold_summary",A1_LEVEL_STABILITY_THRESHOLD_SUMMARY),("level_stability_assignment_summary",A1_LEVEL_STABILITY_ASSIGNMENT_SUMMARY)]:
    table.to_csv(TABLE_DIR/f"A1_P10_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Hai khoảng tham chiếu cố định; có số ở một nửa chưa đủ để so độ nhạy:")
    print(A1_LEVEL_STABILITY_PERIODS)
    display(A1_LEVEL_STABILITY_SUPPORT.groupby("status",sort=False).agg(basins=("basin_id","size"),early_valid_days=("early_valid_days","sum"),
        late_valid_days=("late_valid_days","sum"),early_missing_days=("early_missing_days","sum"),late_missing_days=("late_missing_days","sum")).reset_index())
    print("(2) Chênh ngưỡng late−early và tỷ số late/early ngang ID; mẫu số 0 giữ NaN:");display(A1_LEVEL_STABILITY_THRESHOLD_SUMMARY)
    print("(3) Đổi mức trên cùng quan trắc reference; không phải ma trận nhầm lẫn dự báo hoặc kiểm định độc lập:");display(A1_LEVEL_STABILITY_ASSIGNMENT_SUMMARY)
    print("(4) Các hồ sơ có tham chiếu ngắn/đã kiểm; thiếu một nửa không thay bằng ngưỡng chính:")
    display(A1_LEVEL_STABILITY_SUPPORT.loc[A1_LEVEL_STABILITY_SUPPORT.basin_id.isin(["55","324","600","623","742","758","800"])])
    display(A1_LEVEL_STABILITY_THRESHOLDS.loc[A1_LEVEL_STABILITY_THRESHOLDS.basin_id.isin(["55","324","600","623","742","758","800"])])
print("Đã lưu hình:",stability_figure,"; sáu CSV vào:",TABLE_DIR)
print(f"Thời gian kiểm độ nhạy ngưỡng: {time.perf_counter()-started_level_stability:.1f} giây.")
print("CẢNH BÁO: phép so chỉ xác định khi mỗi nửa có quan trắc; chưa có điều kiện đủ năm/đủ mùa bảo đảm ngưỡng ổn định. Không dùng test, không đổi ngưỡng chính 10.1 hoặc chọn giai đoạn theo độ khớp.")

# %% [markdown]
# ### 10.3. Độ phủ và đỉnh năm thủy văn trước khi fit Gumbel
#
# Năm thủy văn kết thúc ngày 30/9, nhãn là năm kết thúc; chỉ đọc qmax trong reference 1981–2013, không lấy ngày test.

# %% Phần 10.3 - Độ phủ và đỉnh năm thủy văn trước khi fit Gumbel
# Năm thủy văn kết thúc ngày 30/9, nhãn là năm kết thúc; chỉ đọc qmax trong reference 1981–2013, không lấy ngày test.
# Báo cả quy tắc đủ ít nhất nửa lịch năm và biến thể yêu cầu năm nằm trọn reference; năm biên không được đổi mẫu số thành lát ngắn.
started_hydroyears=time.perf_counter()
hydro_reference=tuple(A1_LEVEL_REFERENCE_PERIOD)
if hydro_reference!=("1981-01-01","2013-12-31") or pd.Timestamp(hydro_reference[1])>=pd.Timestamp(PERIODS["test"][0]):
    raise ValueError("Phạm vi đỉnh năm phải đúng reference trước test.")
hydro_calendar=pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
hydro_ids=list(map(str,A1_QMAX_VALUE_IDS));hydro_values=np.asarray(A1_QMAX_VALUES_MM_DAY,dtype=float)
if (len(hydro_ids)!=len(set(hydro_ids)) or set(hydro_ids)!=set(map(str,selected_ids))
    or hydro_values.shape!=(len(hydro_ids),len(hydro_calendar)) or hydro_calendar.has_duplicates
    or not hydro_calendar.equals(pd.date_range(hydro_calendar.min(),hydro_calendar.max(),freq="D"))
    or np.isinf(hydro_values).any() or (hydro_values<0).any()):raise ValueError("Cache Q khác phạm vi đã kiểm.")
hydro_ref_positions=np.flatnonzero((hydro_calendar>=hydro_reference[0])&(hydro_calendar<=hydro_reference[1]))
hydro_ref_dates=hydro_calendar[hydro_ref_positions]
if not hydro_ref_dates.equals(pd.date_range(*hydro_reference,freq="D")):raise ValueError("Cache thiếu lịch reference.")
hydro_labels=hydro_ref_dates.year+(hydro_ref_dates.month>=10).astype(int)
hydro_year_specs={}
for year in sorted(set(hydro_labels)):
    full_dates=pd.date_range(f"{year-1}-10-01",f"{year}-09-30",freq="D")
    local=np.flatnonzero(hydro_labels==year)
    hydro_year_specs[int(year)]=(full_dates,local)
hydro_official=A1_LEVEL_THRESHOLDS.set_index("basin_id",verify_integrity=True)
hydro_rows=[]
for basin_id,series in zip(hydro_ids,hydro_values):
    ref=series[hydro_ref_positions]
    if int(np.isfinite(ref).sum())!=hydro_official.loc[basin_id,"valid_reference_days"]:
        raise ValueError(f"{basin_id}: số ngày reference khác 10.1.")
    for year,(full_dates,local) in hydro_year_specs.items():
        dates=hydro_ref_dates[local];raw=ref[local];finite=np.isfinite(raw);values=raw[finite];valid_dates=dates[finite]
        full_days=len(full_dates);reference_days=len(raw);outside=full_days-reference_days;missing=int((~finite).sum())
        meets_half=bool(2*len(values)>=full_days);full_window=outside==0
        month_expected=np.bincount(full_dates.month,minlength=13)[1:]
        month_valid=np.bincount(valid_dates.month,minlength=13)[1:]
        maximum=float(values.max()) if len(values) else np.nan
        max_dates=valid_dates[values==maximum] if len(values) else pd.DatetimeIndex([])
        row=dict(basin_id=basin_id,hydro_year_end=year,hydro_year_start=full_dates[0],hydro_year_end_date=full_dates[-1],
            full_year_days=full_days,reference_days=reference_days,days_outside_reference=outside,
            valid_reference_days=len(values),missing_inside_reference_days=missing,
            unobserved_full_year_days=outside+missing,observed_full_year_pct=100*len(values)/full_days,
            full_year_inside_reference=full_window,meets_half_calendar_rule=meets_half,
            full_window_meets_half=bool(full_window and meets_half),complete_observed_year=bool(len(values)==full_days),
            months_with_observations=int((month_valid>0).sum()),min_month_observed_pct=float(100*np.min(month_valid/month_expected)),
            observed_max_mm_day=maximum,first_observed_max_date=max_dates.min() if len(max_dates) else pd.NaT,
            last_observed_max_date=max_dates.max() if len(max_dates) else pd.NaT,days_at_observed_max=len(max_dates),
            coverage_status="no_observations" if not len(values) else "at_least_half_calendar" if meets_half else "below_half_calendar")
        if len(values)+missing!=reference_days or len(values)+row["unobserved_full_year_days"]!=full_days:
            raise AssertionError("Các trạng thái ngày năm không cộng khớp.")
        hydro_rows.append(row)
A1_HYDRO_YEAR_COVERAGE=pd.DataFrame(hydro_rows)
hydro_rebuilt=A1_HYDRO_YEAR_COVERAGE.groupby("basin_id").agg(reference_days=("reference_days","sum"),valid_days=("valid_reference_days","sum"))
hydro_expected=hydro_official.reindex(hydro_rebuilt.index)
if (not hydro_rebuilt.reference_days.eq(hydro_expected.expected_reference_days).all()
    or not hydro_rebuilt.valid_days.eq(hydro_expected.valid_reference_days).all()):raise AssertionError("Cộng năm khác reference 10.1.")
hydro_basin_rows=[]
for basin_id,part in A1_HYDRO_YEAR_COVERAGE.groupby("basin_id",sort=False):
    half=part.loc[part.meets_half_calendar_rule];full=part.loc[part.full_window_meets_half]
    hydro_basin_rows.append(dict(basin_id=basin_id,hydro_years_intersecting_reference=len(part),
        full_window_calendar_years=int(part.full_year_inside_reference.sum()),years_with_observations=int(part.valid_reference_days.gt(0).sum()),
        years_meeting_half_calendar=len(half),years_full_window_meeting_half=len(full),complete_observed_years=int(part.complete_observed_year.sum()),
        half_calendar_years_with_missing_months=int(half.months_with_observations.lt(12).sum()),
        boundary_years_meeting_half=int((part.meets_half_calendar_rule&~part.full_year_inside_reference).sum()),
        distinct_maxima_half_calendar=half.observed_max_mm_day.nunique(),distinct_maxima_full_window=full.observed_max_mm_day.nunique(),
        below_10_half_calendar=bool(len(half)<10),below_10_full_window=bool(len(full)<10)))
A1_HYDRO_YEAR_BASIN_SUPPORT=pd.DataFrame(hydro_basin_rows)
A1_HYDRO_YEAR_CALENDAR_SUMMARY=A1_HYDRO_YEAR_COVERAGE.groupby("hydro_year_end",sort=True).agg(
    hydro_year_start=("hydro_year_start","first"),hydro_year_end_date=("hydro_year_end_date","first"),
    full_year_days=("full_year_days","first"),reference_days=("reference_days","first"),days_outside_reference=("days_outside_reference","first"),
    basins=("basin_id","size"),basins_with_observations=("valid_reference_days",lambda x:int(x.gt(0).sum())),
    basins_meeting_half_calendar=("meets_half_calendar_rule","sum"),basins_full_window_meeting_half=("full_window_meets_half","sum"),
    median_observed_full_year_pct=("observed_full_year_pct","median")).reset_index()
hydro_rule_rows=[]
for rule,column in [("at_least_half_of_full_calendar","years_meeting_half_calendar"),("full_window_and_at_least_half","years_full_window_meeting_half")]:
    years=A1_HYDRO_YEAR_BASIN_SUPPORT[column]
    hydro_rule_rows.append(dict(rule=rule,basins=len(years),basins_zero_years=int(years.eq(0).sum()),basins_1_to_9_years=int(years.between(1,9).sum()),
        basins_ge_10_years=int(years.ge(10).sum()),min_years=int(years.min()),median_years=float(years.median()),max_years=int(years.max())))
A1_HYDRO_YEAR_RULE_SUMMARY=pd.DataFrame(hydro_rule_rows)

hydro_order=A1_HYDRO_YEAR_BASIN_SUPPORT.assign(id_numeric=lambda x:x.basin_id.astype(int)).sort_values(["years_full_window_meeting_half","id_numeric"]).basin_id
hydro_heat=A1_HYDRO_YEAR_COVERAGE.pivot(index="basin_id",columns="hydro_year_end",values="observed_full_year_pct").reindex(hydro_order)
fig,axes=plt.subplots(1,2,figsize=(15,7),gridspec_kw={"width_ratios":[1.35,1]})
im=axes[0].imshow(hydro_heat.to_numpy(),vmin=0,vmax=100,cmap="Blues",aspect="auto",interpolation="nearest")
xticks=list(range(0,len(hydro_heat.columns),5));xticks=sorted(set(xticks+[len(hydro_heat.columns)-1]))
yticks=np.unique(np.linspace(0,len(hydro_heat)-1,min(7,len(hydro_heat))).astype(int))
axes[0].set_xticks(xticks,[str(hydro_heat.columns[i]) for i in xticks]);axes[0].set_yticks(yticks,[hydro_heat.index[i] for i in yticks])
axes[0].set(title="Observed days / full hydrological-year calendar",xlabel="Hydrological year (ending year)",ylabel="Basin ID (ordered by full-window year count)")
fig.colorbar(im,ax=axes[0],label="Observed full-year days (%)",fraction=.04,pad=.02)
for column,color,label in [("years_meeting_half_calendar","#2283a5","At least half of full calendar"),
    ("years_full_window_meeting_half","#c87531","Also wholly inside reference")]:
    axes[1].hist(A1_HYDRO_YEAR_BASIN_SUPPORT[column],bins=np.arange(-.5,len(hydro_year_specs)+1.5,1),histtype="step",linewidth=1.7,color=color,label=label)
axes[1].axvline(10,color="gray",linestyle=":",label="10-year reporting flag")
axes[1].set(title="Available observed annual-maxima candidates",xlabel="Years satisfying the stated rule",ylabel="Basins");axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
fig.suptitle("Pre-test hydrological-year coverage before fitting flood return levels",fontsize=14)
fig.text(.5,.035,"Hydrological years run from 1 October to 30 September; only observations dated 1981–2013 are read.\n"
    "Boundary years retain their full 365/366-day denominator. Both availability rules are reported; no Gumbel fit or return level is produced here.\n"
    "Passing the half-year rule does not guarantee that an unobserved flood peak was captured. Fewer than 10 years is reported, not used to remove basins.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.16,1,.93));hydro_figure=FIGURE_DIR/"A1_P10_hydrological_year_coverage.png"
fig.savefig(hydro_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("hydrological_year_coverage",A1_HYDRO_YEAR_COVERAGE),("hydrological_year_basin_support",A1_HYDRO_YEAR_BASIN_SUPPORT),
    ("hydrological_year_calendar_summary",A1_HYDRO_YEAR_CALENDAR_SUMMARY),("hydrological_year_rule_summary",A1_HYDRO_YEAR_RULE_SUMMARY)]:
    table.to_csv(TABLE_DIR/f"A1_P10_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Phạm vi và quy tắc đếm; năm biên không rút mẫu số xuống số ngày giao reference:")
    print("Reference:",hydro_reference,"; năm thủy văn 01/10–30/09, nhãn năm kết thúc; đủ nửa khi 2×valid_days ≥ full_year_days.")
    display(A1_HYDRO_YEAR_COVERAGE.groupby(["full_year_inside_reference","coverage_status"],sort=False).agg(
        basin_years=("basin_id","size"),valid_days=("valid_reference_days","sum"),missing_inside_reference=("missing_inside_reference_days","sum"),
        days_outside_reference=("days_outside_reference","sum")).reset_index())
    print("(2) Số đỉnh năm quan trắc có thể dùng theo hai quy tắc; dưới 10 năm chỉ là cờ báo cáo:");display(A1_HYDRO_YEAR_RULE_SUMMARY)
    print("(3) Lịch năm/độ phủ; hai năm cắt biên và số ID đủ điều kiện được giữ riêng:");display(A1_HYDRO_YEAR_CALENDAR_SUMMARY)
    print("(4) Các hồ sơ; số năm có quan trắc khác số năm qua kiểm độ phủ:")
    display(A1_HYDRO_YEAR_BASIN_SUPPORT.loc[A1_HYDRO_YEAR_BASIN_SUPPORT.basin_id.isin(["55","324","600","623","742","758","800"])])
print("Đã lưu hình:",hydro_figure,"; bốn CSV vào:",TABLE_DIR)
print(f"Thời gian kiểm độ phủ năm thủy văn: {time.perf_counter()-started_hydroyears:.1f} giây.")
print("CẢNH BÁO: observed_max là max trên phần có số, chưa bảo đảm là đỉnh thật của năm. Năm đủ ≥50% ngày vẫn có thể mất đỉnh/mùa; chưa chọn một quy tắc biên để fit hoặc thay ngưỡng bốn mức.")

# %% [markdown]
# ### 10.4. Gumbel L-moments trên đỉnh năm quan trắc trước test
#
# Mẫu chính theo giao thức đã ghi: có ≥50% lịch năm trong reference, kể cả năm biên 1981 đủ điều kiện.

# %% Phần 10.4 - Gumbel L-moments trên đỉnh năm quan trắc trước test
# Mẫu chính theo giao thức đã ghi: có ≥50% lịch năm trong reference, kể cả năm biên 1981 đủ điều kiện.
# Mẫu chỉ gồm năm nằm trọn reference là độ nhạy riêng; không dùng test hoàn thành năm 2014.
# Giữ ID dưới 10 năm với cờ, không thay ngưỡng bốn mức, không chấm dự báo hoặc đếm trận trong cell này.
from scipy.stats import gumbel_r

started_gumbel=time.perf_counter()
A1_FLOOD_PRIMARY_POLICY="half_calendar_primary"
A1_FLOOD_RETURN_PERIODS=np.array([1.5,2.,5.,10.])
gumbel_policies={A1_FLOOD_PRIMARY_POLICY:"meets_half_calendar_rule","full_window_sensitivity":"full_window_meets_half"}
gumbel_years=A1_HYDRO_YEAR_COVERAGE.copy(deep=True)
gumbel_ids=set(map(str,selected_ids));gumbel_ref=tuple(A1_LEVEL_REFERENCE_PERIOD)
if (gumbel_ref!=("1981-01-01","2013-12-31") or pd.Timestamp(gumbel_ref[1])>=pd.Timestamp(PERIODS["test"][0])
    or set(gumbel_years.basin_id)!=gumbel_ids or gumbel_years.duplicated(["basin_id","hydro_year_end"]).any()
    or not gumbel_years.meets_half_calendar_rule.eq(2*gumbel_years.valid_reference_days>=gumbel_years.full_year_days).all()
    or not gumbel_years.full_window_meets_half.eq(gumbel_years.meets_half_calendar_rule&gumbel_years.full_year_inside_reference).all()
    or not gumbel_years.full_year_inside_reference.eq((gumbel_years.hydro_year_start>=pd.Timestamp(gumbel_ref[0]))&
        (gumbel_years.hydro_year_end_date<=pd.Timestamp(gumbel_ref[1]))).all()
    or not np.isfinite(gumbel_years.observed_max_mm_day).eq(gumbel_years.valid_reference_days.gt(0)).all()
    or gumbel_years.observed_max_mm_day.lt(0).any()):
    raise ValueError("Bảng đỉnh năm/điều kiện chọn năm khác 10.3.")
for field in ["first_observed_max_date","last_observed_max_date"]:
    dates=gumbel_years[field].dropna()
    if not dates.between(pd.Timestamp(gumbel_ref[0]),pd.Timestamp(gumbel_ref[1])).all():raise ValueError("Đỉnh được lưu có ngày ngoài reference.")


def fit_gumbel_lmoments(values):
    values=np.sort(np.asarray(values,dtype=float));n=len(values)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError("Đỉnh năm phải hữu hạn không âm.")
    result=dict(n_years=n,distinct_maxima=len(np.unique(values)),l1_mm_day=float(values.mean()) if n else np.nan,
        l2_mm_day=np.nan,location_mm_day=np.nan,scale_mm_day=np.nan,status="fewer_than_2_maxima")
    if n<2:return result
    # l2 = nửa trung bình độ chênh tuyệt đối của các cặp; kiểm độc lập bằng công thức trọng số thứ tự.
    l2=float(np.triu(np.abs(values[:,None]-values[None,:]),k=1).sum()/(n*(n-1)))
    weights=2*np.arange(1,n+1)-n-1
    alternative=float(np.dot(weights,values)/(n*(n-1)))
    if not np.isclose(l2,alternative,rtol=1e-10,atol=1e-10):raise AssertionError("Hai cách tính L2 không khớp.")
    result["l2_mm_day"]=l2
    if l2<=0:
        result["status"]="degenerate_zero_l2";return result
    scale=l2/np.log(2);location=result["l1_mm_day"]-np.euler_gamma*scale
    if not np.isfinite([location,scale]).all() or scale<=0:raise ValueError("Tham số Gumbel không hữu hạn hoặc scale không dương.")
    result.update(location_mm_day=float(location),scale_mm_day=float(scale),status="fit_defined")
    return result


def gumbel_return_levels(location,scale,return_periods):
    periods=np.asarray(return_periods,dtype=float)
    if not np.isfinite(periods).all() or (periods<=1).any() or not np.isfinite([location,scale]).all() or scale<=0:
        raise ValueError("Gumbel cần scale>0 và chu kỳ lặp lại hữu hạn >1.")
    levels=location-scale*np.log(-np.log1p(-1/periods))
    expected=gumbel_r.ppf(1-1/periods,loc=location,scale=scale)
    if not np.allclose(levels,expected,rtol=1e-12,atol=1e-10) or not np.isfinite(levels).all():
        raise AssertionError("Ngưỡng không khớp hàm phân vị Gumbel của SciPy.")
    return levels


gumbel_fit_rows,gumbel_level_rows,gumbel_sample_parts=[],[],[]
for policy,mask_column in gumbel_policies.items():
    for basin_id in sorted(gumbel_ids,key=int):
        sample=gumbel_years.loc[gumbel_years.basin_id.eq(basin_id)&gumbel_years[mask_column]].sort_values("hydro_year_end")
        values=sample.observed_max_mm_day.to_numpy(dtype=float)
        fitted=fit_gumbel_lmoments(values)
        row=dict(policy=policy,basin_id=basin_id,reference_start=gumbel_ref[0],reference_end=gumbel_ref[1],
            below_10_years=bool(len(sample)<10),first_hydro_year=int(sample.hydro_year_end.min()) if len(sample) else np.nan,
            last_hydro_year=int(sample.hydro_year_end.max()) if len(sample) else np.nan,
            used_years_with_missing_months=int(sample.months_with_observations.lt(12).sum()),
            used_complete_observed_years=int(sample.complete_observed_year.sum()),
            used_boundary_years=int((~sample.full_year_inside_reference).sum()),
            sample_max_mm_day=float(values.max()) if len(values) else np.nan,**fitted)
        gumbel_fit_rows.append(row)
        if len(sample):gumbel_sample_parts.append(sample.assign(policy=policy))
        levels=gumbel_return_levels(row["location_mm_day"],row["scale_mm_day"],A1_FLOOD_RETURN_PERIODS) if row["status"]=="fit_defined" else np.full(4,np.nan)
        if row["status"]=="fit_defined" and not (np.diff(levels)>0).all():raise AssertionError("Ngưỡng lũ phải tăng theo T.")
        for period,level in zip(A1_FLOOD_RETURN_PERIODS,levels):
            gumbel_level_rows.append(dict(policy=policy,basin_id=basin_id,return_period_years=float(period),threshold_mm_day=float(level),
                n_years=len(sample),below_10_years=row["below_10_years"],fit_status=row["status"],
                return_period_exceeds_sample_years=bool(period>len(sample)),
                threshold_above_observed_sample_max=bool(level>row["sample_max_mm_day"]) if np.isfinite(level) else None,
                negative_return_level=bool(level<0) if np.isfinite(level) else None))
A1_FLOOD_GUMBEL_FITS=pd.DataFrame(gumbel_fit_rows)
A1_FLOOD_RETURN_LEVELS=pd.DataFrame(gumbel_level_rows)
A1_FLOOD_ANNUAL_SAMPLE=pd.concat(gumbel_sample_parts,ignore_index=True) if gumbel_sample_parts else pd.DataFrame()
gumbel_main=A1_FLOOD_RETURN_LEVELS.loc[A1_FLOOD_RETURN_LEVELS.policy.eq(A1_FLOOD_PRIMARY_POLICY)]
gumbel_boundary=A1_FLOOD_RETURN_LEVELS.loc[~A1_FLOOD_RETURN_LEVELS.policy.eq(A1_FLOOD_PRIMARY_POLICY)]
A1_FLOOD_BOUNDARY_SENSITIVITY=gumbel_main[["basin_id","return_period_years","threshold_mm_day","n_years","fit_status"]].merge(
    gumbel_boundary[["basin_id","return_period_years","threshold_mm_day","n_years","fit_status"]],
    on=["basin_id","return_period_years"],validate="one_to_one",suffixes=("_primary","_full_window"))
gumbel_s=A1_FLOOD_BOUNDARY_SENSITIVITY
gumbel_s["sample_changed"]=gumbel_s.n_years_primary.ne(gumbel_s.n_years_full_window)
gumbel_s["delta_primary_minus_full_window_mm_day"]=gumbel_s.threshold_mm_day_primary-gumbel_s.threshold_mm_day_full_window
gumbel_s["ratio_primary_to_full_window"]=gumbel_s.threshold_mm_day_primary/gumbel_s.threshold_mm_day_full_window.where(gumbel_s.threshold_mm_day_full_window.gt(0))
A1_FLOOD_LEVEL_COMPARISON=gumbel_main.loc[gumbel_main.return_period_years.eq(1.5),["basin_id","threshold_mm_day","n_years","below_10_years","fit_status"]].rename(
    columns={"threshold_mm_day":"q_1p5_year_mm_day"}).merge(A1_LEVEL_THRESHOLDS[["basin_id","q98_mm_day"]],on="basin_id",validate="one_to_one")
A1_FLOOD_LEVEL_COMPARISON["ratio_1p5_year_to_daily_q98"]=A1_FLOOD_LEVEL_COMPARISON.q_1p5_year_mm_day/A1_FLOOD_LEVEL_COMPARISON.q98_mm_day.where(A1_FLOOD_LEVEL_COMPARISON.q98_mm_day.gt(0))
A1_FLOOD_LEVEL_COMPARISON["q_1p5_below_daily_q98"]=A1_FLOOD_LEVEL_COMPARISON.q_1p5_year_mm_day.lt(A1_FLOOD_LEVEL_COMPARISON.q98_mm_day).where(A1_FLOOD_LEVEL_COMPARISON.q_1p5_year_mm_day.notna())
gumbel_summary_rows=[]
for period,part in gumbel_main.groupby("return_period_years",sort=False):
    values=part.threshold_mm_day.dropna().to_numpy()
    gumbel_summary_rows.append(dict(return_period_years=period,defined_basins=len(values),below_10_years=int(part.below_10_years.sum()),
        negative_return_level_basins=int(part.negative_return_level.eq(True).sum()),
        T_exceeds_sample_years_basins=int(part.return_period_exceeds_sample_years.sum()),
        **dict(zip(["min_mm_day","p05_mm_day","median_mm_day","p95_mm_day","max_mm_day"],np.quantile(values,[0,.05,.5,.95,1],method="linear") if len(values) else [np.nan]*5))))
A1_FLOOD_RETURN_LEVEL_SUMMARY=pd.DataFrame(gumbel_summary_rows)
gumbel_sensitivity_summary=[]
for period,part in gumbel_s.loc[gumbel_s.sample_changed].groupby("return_period_years",sort=False):
    delta=part.delta_primary_minus_full_window_mm_day.dropna();ratio=part.ratio_primary_to_full_window.dropna()
    gumbel_sensitivity_summary.append(dict(return_period_years=period,basins_with_changed_sample=len(part),defined_delta_basins=len(delta),
        median_delta_mm_day=delta.median(),max_abs_delta_mm_day=delta.abs().max(),
        ratio_p05=ratio.quantile(.05),ratio_median=ratio.median(),ratio_p95=ratio.quantile(.95)))
A1_FLOOD_BOUNDARY_SUMMARY=pd.DataFrame(gumbel_sensitivity_summary,columns=["return_period_years","basins_with_changed_sample","defined_delta_basins",
    "median_delta_mm_day","max_abs_delta_mm_day","ratio_p05","ratio_median","ratio_p95"])

gumbel_main_fits=A1_FLOOD_GUMBEL_FITS.loc[A1_FLOOD_GUMBEL_FITS.policy.eq(A1_FLOOD_PRIMARY_POLICY)].set_index("basin_id",verify_integrity=True)
gumbel_plot_ids=["55","324","742","758","800","604"]
fig,axes=plt.subplots(2,3,figsize=(16,10))
for ax,basin_id in zip(axes.flat,gumbel_plot_ids):
    ax.set_title("Basin "+basin_id)
    if basin_id not in gumbel_main_fits.index:
        ax.text(.5,.5,"Not in selected basin set",ha="center",transform=ax.transAxes);continue
    fit=gumbel_main_fits.loc[basin_id];sample=A1_FLOOD_ANNUAL_SAMPLE.loc[A1_FLOOD_ANNUAL_SAMPLE.policy.eq(A1_FLOOD_PRIMARY_POLICY)&A1_FLOOD_ANNUAL_SAMPLE.basin_id.eq(basin_id)] if len(A1_FLOOD_ANNUAL_SAMPLE) else pd.DataFrame()
    values=np.sort(sample.observed_max_mm_day.to_numpy()) if len(sample) else np.array([])
    if len(values):
        ranks=np.arange(1,len(values)+1);empirical_t=(len(values)+1)/(len(values)+1-ranks)
        ax.scatter(empirical_t,values,s=20,color="#2283a5",label="Observed annual maxima")
    if fit.status=="fit_defined":
        grid=np.geomspace(1.5,max(12,len(values)+1),200)
        ax.plot(grid,gumbel_return_levels(fit.location_mm_day,fit.scale_mm_day,grid),color="#c87531",label="Gumbel L-moments")
        levels=gumbel_return_levels(fit.location_mm_day,fit.scale_mm_day,A1_FLOOD_RETURN_PERIODS)
        ax.scatter(A1_FLOOD_RETURN_PERIODS,levels,color="#c87531",marker="x",s=35,label="T = 1.5, 2, 5, 10")
    else:ax.text(.02,.86,fit.status,transform=ax.transAxes,fontsize=9)
    q98=A1_LEVEL_THRESHOLDS.set_index("basin_id").loc[basin_id,"q98_mm_day"]
    if np.isfinite(q98):ax.axhline(q98,color="gray",linestyle=":",label="Daily q98 (1981–2013)")
    ax.set_title(f"Basin {basin_id}: n={int(fit.n_years)}"+(" (<10 years)" if fit.below_10_years else ""),fontsize=11)
    ax.set_xscale("log");ax.set(xlabel="Return period (years)",ylabel="qmax (mm/day)");ax.grid(alpha=.2)
handles,labels=next((ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0]),([],[]))
fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.10),ncol=4,fontsize=9)
fig.suptitle("Observed annual maxima and Gumbel return levels: half-calendar primary sample",fontsize=14)
fig.text(.5,.025,"Primary sample: at least half of full-calendar days observed within 1981–2013; eligible boundary year 1981 retained.\n"
    "Points use Weibull plotting positions; curves start at T=1.5. Full-window fits are a separate sensitivity analysis.\n"
    "Short samples and missing seasons are flagged. Lines are fitted descriptions, not goodness-of-fit proof, independent floods or operational warnings.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.18,1,.94));gumbel_figure=FIGURE_DIR/"A1_P10_gumbel_return_levels.png"
fig.savefig(gumbel_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("flood_gumbel_fits",A1_FLOOD_GUMBEL_FITS),("flood_return_levels",A1_FLOOD_RETURN_LEVELS),
    ("flood_annual_sample",A1_FLOOD_ANNUAL_SAMPLE),("flood_boundary_sensitivity",A1_FLOOD_BOUNDARY_SENSITIVITY),
    ("flood_level_comparison",A1_FLOOD_LEVEL_COMPARISON),("flood_return_level_summary",A1_FLOOD_RETURN_LEVEL_SUMMARY),("flood_boundary_summary",A1_FLOOD_BOUNDARY_SUMMARY)]:
    table.to_csv(TABLE_DIR/f"A1_P10_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Mẫu chính đủ nửa lịch năm (kể cả năm biên đủ điều kiện), full-window là độ nhạy; fit không xác định giữ NaN, dưới 10 năm chỉ gắn cờ:")
    display(A1_FLOOD_GUMBEL_FITS.groupby(["policy","status","below_10_years"],sort=False).agg(basins=("basin_id","size"),
        min_years=("n_years","min"),median_years=("n_years","median"),max_years=("n_years","max"),
        used_years_with_missing_months=("used_years_with_missing_months","sum"),used_boundary_years=("used_boundary_years","sum")).reset_index())
    print("(2) Ngưỡng mẫu chính ngang ID, mm/ngày; đây không phải khoảng tin cậy:");display(A1_FLOOD_RETURN_LEVEL_SUMMARY)
    print("(3) Hồ sơ và toàn bộ ID ít hơn 10 năm trong mẫu chính:")
    display(gumbel_main_fits.loc[gumbel_main_fits.index.isin(["55","324","600","623","742","758","800"])|gumbel_main_fits.below_10_years,
        ["n_years","distinct_maxima","below_10_years","used_complete_observed_years","used_years_with_missing_months","location_mm_day","scale_mm_day","status"]])
    print("(4) Mẫu chính trừ full-window, chỉ tổng hợp ID có mẫu đổi; tỷ số chính/full-window khi mẫu số >0:");display(A1_FLOOD_BOUNDARY_SUMMARY)
    print("(5) Ngưỡng 1,5 năm so phân vị q98 ngày: hai thước đo khác nhau:")
    ratios=A1_FLOOD_LEVEL_COMPARISON.ratio_1p5_year_to_daily_q98.dropna()
    print("Defined ratios:",len(ratios),"; p05/median/p95:",ratios.quantile([.05,.5,.95]).to_dict(),
        "; Q1.5 < q98:",int(A1_FLOOD_LEVEL_COMPARISON.q_1p5_below_daily_q98.fillna(False).sum()))
    display(A1_FLOOD_LEVEL_COMPARISON.loc[A1_FLOOD_LEVEL_COMPARISON.basin_id.isin(["55","324","600","623","742","758","800"])])
print("Đã lưu hình:",gumbel_figure,"; bảy CSV vào:",TABLE_DIR)
print(f"Thời gian fit Gumbel L-moments: {time.perf_counter()-started_gumbel:.1f} giây.")
print("CẢNH BÁO: fit được không bảo đảm Gumbel phù hợp hoặc năm đủ quan trắc đỉnh. Mẫu <10 năm vẫn là ước lượng mẫu ngắn; chưa có khoảng bất định/kiểm định độ khớp. Không tự sửa/chặn ngưỡng hoặc thay q30/q80/q98.")

# %% [markdown]
# ### 10.5. Độ nhạy ngưỡng Gumbel khi bỏ lần lượt một đỉnh năm
#
# Fit lại cùng công thức trên n mẫu n−1 năm của mẫu chính; giữ năm trùng max và trạng thái suy biến riêng.

# %% Phần 10.5 - Độ nhạy ngưỡng Gumbel khi bỏ lần lượt một đỉnh năm
# Fit lại cùng công thức trên n mẫu n−1 năm của mẫu chính; giữ năm trùng max và trạng thái suy biến riêng.
# Miền min–max chỉ là độ nhạy bỏ một năm, không phải khoảng tin cậy; không đọc test hoặc thay ngưỡng 10.4.
started_loo=time.perf_counter()
loo_ids=list(map(str,selected_ids));loo_periods=np.asarray(A1_FLOOD_RETURN_PERIODS,dtype=float)
if (A1_FLOOD_PRIMARY_POLICY!="half_calendar_primary" or tuple(A1_LEVEL_REFERENCE_PERIOD)!=("1981-01-01","2013-12-31")
    or not np.array_equal(loo_periods,[1.5,2.,5.,10.])):raise ValueError("Giao thức khác 10.4.")
loo_fits=A1_FLOOD_GUMBEL_FITS.loc[A1_FLOOD_GUMBEL_FITS.policy.eq(A1_FLOOD_PRIMARY_POLICY)].copy(deep=True).set_index("basin_id",verify_integrity=True)
loo_sample=A1_FLOOD_ANNUAL_SAMPLE.loc[A1_FLOOD_ANNUAL_SAMPLE.policy.eq(A1_FLOOD_PRIMARY_POLICY)].copy(deep=True)
loo_levels=A1_FLOOD_RETURN_LEVELS.loc[A1_FLOOD_RETURN_LEVELS.policy.eq(A1_FLOOD_PRIMARY_POLICY)].copy(deep=True)
if (set(loo_fits.index)!=set(loo_ids) or loo_sample.duplicated(["basin_id","hydro_year_end"]).any()
    or loo_levels.duplicated(["basin_id","return_period_years"]).any()
    or not set(loo_sample.basin_id).issubset(set(loo_ids))
    or not loo_sample.meets_half_calendar_rule.all()
    or not np.isfinite(loo_sample.observed_max_mm_day).all()
    or loo_sample.observed_max_mm_day.lt(0).any()
    or not loo_sample.first_observed_max_date.between(pd.Timestamp(A1_LEVEL_REFERENCE_PERIOD[0]),pd.Timestamp(A1_LEVEL_REFERENCE_PERIOD[1])).all()
    or not loo_sample.last_observed_max_date.between(pd.Timestamp(A1_LEVEL_REFERENCE_PERIOD[0]),pd.Timestamp(A1_LEVEL_REFERENCE_PERIOD[1])).all()):
    raise ValueError("Đỉnh năm/fit mẫu chính khác phạm vi đã kiểm.")

loo_details=[];loo_basin_rows=[]
for index,basin_id in enumerate(sorted(loo_ids,key=int),start=1):
    sample=loo_sample.loc[loo_sample.basin_id.eq(basin_id)].sort_values("hydro_year_end")
    values=sample.observed_max_mm_day.to_numpy(dtype=float);n=len(values)
    fitted=fit_gumbel_lmoments(values);stored=loo_fits.loc[basin_id]
    if n!=stored.n_years or fitted["status"]!=stored.status:raise ValueError(f"{basin_id}: mẫu fit không khớp 10.4.")
    for field in ["l1_mm_day","l2_mm_day","location_mm_day","scale_mm_day"]:
        if not np.isclose(fitted[field],stored[field],rtol=1e-10,atol=1e-10,equal_nan=True):raise ValueError(f"{basin_id}: {field} khác 10.4.")
    baseline=gumbel_return_levels(fitted["location_mm_day"],fitted["scale_mm_day"],loo_periods) if fitted["status"]=="fit_defined" else np.full(len(loo_periods),np.nan)
    saved=loo_levels.loc[loo_levels.basin_id.eq(basin_id)].set_index("return_period_years").reindex(loo_periods)
    if len(saved)!=len(loo_periods) or saved.threshold_mm_day.isna().any() and fitted["status"]=="fit_defined":
        raise ValueError(f"{basin_id}: thiếu ngưỡng 10.4.")
    if not np.allclose(baseline,saved.threshold_mm_day.to_numpy(),rtol=1e-10,atol=1e-10,equal_nan=True):raise ValueError(f"{basin_id}: ngưỡng khác 10.4.")
    replicate_levels=np.full((n,len(loo_periods)),np.nan)
    for j,omitted in enumerate(sample.itertuples(index=False)):
        refit=fit_gumbel_lmoments(np.delete(values,j))
        if refit["status"]=="fit_defined":replicate_levels[j]=gumbel_return_levels(refit["location_mm_day"],refit["scale_mm_day"],loo_periods)
        for k,period in enumerate(loo_periods):
            level=replicate_levels[j,k];base_level=baseline[k]
            loo_details.append(dict(basin_id=basin_id,n_years=n,below_10_years=bool(n<10),return_period_years=float(period),
                omitted_hydro_year=int(omitted.hydro_year_end),omitted_max_mm_day=omitted.observed_max_mm_day,
                omitted_is_sample_max=bool(omitted.observed_max_mm_day==values.max()),
                omitted_complete_observed_year=bool(omitted.complete_observed_year),omitted_months_with_observations=omitted.months_with_observations,
                fit_status=refit["status"],baseline_mm_day=base_level,loo_mm_day=level,delta_mm_day=level-base_level,
                delta_pct_baseline=100*(level-base_level)/base_level if np.isfinite(base_level) and base_level>0 else np.nan))
    for k,period in enumerate(loo_periods):
        levels=replicate_levels[:,k];finite=np.isfinite(levels);base_level=baseline[k]
        delta=levels-base_level;defined_delta=np.isfinite(delta)
        worst=int(np.nanargmax(np.abs(delta))) if defined_delta.any() else None
        minimum=float(levels[finite].min()) if finite.any() else np.nan
        maximum=float(levels[finite].max()) if finite.any() else np.nan
        max_change=float(np.abs(delta[defined_delta]).max()) if defined_delta.any() else np.nan
        positive_base=bool(np.isfinite(base_level) and base_level>0)
        loo_basin_rows.append(dict(basin_id=basin_id,n_years=n,below_10_years=bool(n<10),return_period_years=float(period),
            baseline_fit_status=fitted["status"],baseline_mm_day=base_level,omissions=n,defined_omissions=int(finite.sum()),
            all_omissions_defined=bool(n>0 and finite.all()),min_loo_mm_day=minimum,max_loo_mm_day=maximum,
            range_loo_mm_day=maximum-minimum,max_abs_change_mm_day=max_change,
            range_pct_baseline=100*(maximum-minimum)/base_level if positive_base else np.nan,
            max_abs_change_pct_baseline=100*max_change/base_level if positive_base else np.nan,
            min_loo_pct_baseline=100*minimum/base_level if positive_base else np.nan,
            max_loo_pct_baseline=100*maximum/base_level if positive_base else np.nan,
            worst_omitted_hydro_year=int(sample.iloc[worst].hydro_year_end) if worst is not None else np.nan,
            worst_omitted_is_sample_max=bool(values[worst]==values.max()) if worst is not None else None,
            worst_omitted_complete_year=bool(sample.iloc[worst].complete_observed_year) if worst is not None else None,
            negative_loo_levels=int((levels[finite]<0).sum()),T_exceeds_n_minus_1=bool(period>n-1)))
    if index%50==0 or index==len(loo_ids):print(f"Đã kiểm bỏ một đỉnh năm {index}/{len(loo_ids)} ID")
A1_FLOOD_LOO_DETAILS=pd.DataFrame(loo_details)
A1_FLOOD_LOO_BY_BASIN=pd.DataFrame(loo_basin_rows)
if len(A1_FLOOD_LOO_DETAILS)!=int(loo_fits.n_years.sum())*len(loo_periods):raise AssertionError("Số phép bỏ năm không cộng khớp.")
loo_summary=[]
for (period,short),part in A1_FLOOD_LOO_BY_BASIN.groupby(["return_period_years","below_10_years"],sort=True):
    # Chỉ tổng hợp miền đủ n lần fit; mọi miền có fit suy biến vẫn giữ trong bảng chi tiết và đếm riêng.
    comparable=part.loc[part.all_omissions_defined&part.baseline_mm_day.gt(0)]
    changes=comparable.max_abs_change_pct_baseline.dropna();ranges=comparable.range_pct_baseline.dropna()
    loo_summary.append(dict(return_period_years=period,below_10_years=short,basins=len(part),
        all_omissions_defined_basins=int(part.all_omissions_defined.sum()),relative_summary_basins=len(changes),
        undefined_omissions=int((part.omissions-part.defined_omissions).sum()),
        nonpositive_or_undefined_baseline_basins=int((~part.baseline_mm_day.gt(0)).sum()),
        max_abs_change_pct_p05=changes.quantile(.05),max_abs_change_pct_median=changes.median(),max_abs_change_pct_p95=changes.quantile(.95),
        range_pct_median=ranges.median(),range_pct_p95=ranges.quantile(.95),negative_loo_levels=int(part.negative_loo_levels.sum())))
A1_FLOOD_LOO_SUMMARY=pd.DataFrame(loo_summary)
loo_profile_ids=["55","324","600","623","742","758","800","604"]
A1_FLOOD_LOO_PROFILES=A1_FLOOD_LOO_BY_BASIN.loc[A1_FLOOD_LOO_BY_BASIN.basin_id.isin(loo_profile_ids)].copy()

fig,axes=plt.subplots(1,2,figsize=(15,6))
loo_t10=A1_FLOOD_LOO_BY_BASIN.loc[A1_FLOOD_LOO_BY_BASIN.return_period_years.eq(10)]
for short,color,label in [(False,"#2283a5","At least 10 years"),(True,"#c87531","Fewer than 10 years")]:
    part=loo_t10.loc[loo_t10.below_10_years.eq(short)&loo_t10.all_omissions_defined&loo_t10.baseline_mm_day.gt(0)]
    values=np.sort(part.max_abs_change_pct_baseline.dropna().to_numpy())
    if len(values):axes[0].step(np.r_[0.,values],np.r_[0.,100*np.arange(1,len(values)+1)/len(values)],where="post",color=color,label=f"{label} (n={len(values)})")
axes[0].set(xlabel="Largest absolute change / original threshold (%)",ylabel="Basins at or below value (%)",
    title="10-year level: leave-one-year-out sensitivity",ylim=(0,101));axes[0].set_xscale("symlog",linthresh=1);axes[0].set_xlim(left=0)
if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=9)
profiles=loo_t10.set_index("basin_id").reindex([i for i in loo_profile_ids if i in set(loo_t10.basin_id)])
for y,(basin_id,row) in enumerate(profiles.iterrows()):
    if np.isfinite([row.min_loo_pct_baseline,row.max_loo_pct_baseline]).all():
        axes[1].plot([row.min_loo_pct_baseline,row.max_loo_pct_baseline],[y,y],color="#2283a5",marker="|",markersize=10)
    else:axes[1].text(.02,y,"Undefined relative range",transform=axes[1].get_yaxis_transform(),fontsize=8)
    if row.baseline_mm_day>0:axes[1].scatter([100],[y],color="#c87531",s=22,zorder=3)
axes[1].set_yticks(np.arange(len(profiles)),[f"ID {i} (n={int(r.n_years)}, {int(r.defined_omissions)}/{int(r.omissions)} defined)" for i,r in profiles.iterrows()])
axes[1].invert_yaxis();axes[1].axvline(100,color="gray",linestyle=":")
axes[1].set(xlabel="Leave-one-out threshold / original threshold (%)",title="Fixed profiles: min–max across omitted years")
for ax in axes:ax.grid(alpha=.2)
fig.suptitle("Gumbel return-level sensitivity to individual observed annual maxima",fontsize=14)
fig.text(.5,.025,"Same primary annual-maxima sample and L-moments formula as 10.4; no test observations or threshold changes.\n"
    "Ranges describe removing one recorded year, not confidence intervals. Undefined refits remain separate; ECDFs require every omission defined.\n"
    "This diagnostic does not recover missing peaks or establish Gumbel fit, stationarity or independence.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.17,1,.93));loo_figure=FIGURE_DIR/"A1_P10_gumbel_leave_one_year_out.png"
fig.savefig(loo_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("flood_loo_details",A1_FLOOD_LOO_DETAILS),("flood_loo_by_basin",A1_FLOOD_LOO_BY_BASIN),
    ("flood_loo_summary",A1_FLOOD_LOO_SUMMARY),("flood_loo_profiles",A1_FLOOD_LOO_PROFILES)]:
    table.to_csv(TABLE_DIR/f"A1_P10_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Độ nhạy bỏ một năm, mỗi ID trọng số bằng nhau; chỉ tổng hợp tương đối khi mọi lần bỏ đều fit được và ngưỡng gốc >0:");display(A1_FLOOD_LOO_SUMMARY)
    print("(2) Hồ sơ cố định ở T=10 năm; miền min–max không phải khoảng tin cậy:");display(profiles.reset_index())
    print("(3) Mười ID có thay đổi tương đối lớn nhất tại T=10 trong tập mọi lần bỏ đều xác định; chỉ xếp mô tả:")
    display(loo_t10.loc[loo_t10.all_omissions_defined&loo_t10.baseline_mm_day.gt(0)].sort_values(["max_abs_change_pct_baseline","basin_id"],ascending=[False,True]).head(10))
    print("(4) Trạng thái fit sau bỏ năm; mỗi năm giữ bốn T, không gộp các năm có cùng đỉnh:")
    display(A1_FLOOD_LOO_DETAILS.groupby(["return_period_years","fit_status"],sort=False).size().rename("refits").reset_index())
    display(A1_FLOOD_LOO_BY_BASIN.loc[~A1_FLOOD_LOO_BY_BASIN.all_omissions_defined])
print("Đã lưu hình:",loo_figure,"; bốn CSV vào:",TABLE_DIR)
print(f"Thời gian kiểm bỏ một đỉnh năm: {time.perf_counter()-started_loo:.1f} giây.")
print("CẢNH BÁO: miền bỏ một năm không phải khoảng tin cậy hoặc kiểm định độ khớp; mẫu ngắn, thiếu mùa/đỉnh, phụ thuộc và biến đổi theo thời gian chưa được giải quyết. Không đổi ngưỡng chính hoặc lọc ID theo độ nhạy.")

# %% [markdown]
# ### 10.6. Độ khớp mô tả giữa đỉnh năm quan trắc và Gumbel đã fit
#
# Giữ nguyên mẫu chính/tham số 10.4; đo khoảng cách CDF, sai khác phân vị và khối xác suất dưới 0.

# %% Phần 10.6 - Độ khớp mô tả giữa đỉnh năm quan trắc và Gumbel đã fit
# Giữ nguyên mẫu chính/tham số 10.4; đo khoảng cách CDF, sai khác phân vị và khối xác suất dưới 0.
# Không tính p-value/tiêu chí chấp nhận, không chọn phân phối hoặc sửa ngưỡng theo phép so trong mẫu.
started_fitdiag=time.perf_counter()
diag_ids=list(map(str,selected_ids))
if A1_FLOOD_PRIMARY_POLICY!="half_calendar_primary" or tuple(A1_LEVEL_REFERENCE_PERIOD)!=("1981-01-01","2013-12-31"):
    raise ValueError("Phạm vi mẫu chính khác 10.4.")
diag_fits=A1_FLOOD_GUMBEL_FITS.loc[A1_FLOOD_GUMBEL_FITS.policy.eq(A1_FLOOD_PRIMARY_POLICY)].copy(deep=True).set_index("basin_id",verify_integrity=True)
diag_sample=A1_FLOOD_ANNUAL_SAMPLE.loc[A1_FLOOD_ANNUAL_SAMPLE.policy.eq(A1_FLOOD_PRIMARY_POLICY)].copy(deep=True)
if (set(diag_fits.index)!=set(diag_ids) or diag_sample.duplicated(["basin_id","hydro_year_end"]).any()
    or not set(diag_sample.basin_id).issubset(set(diag_ids)) or not diag_sample.meets_half_calendar_rule.all()
    or not np.isfinite(diag_sample.observed_max_mm_day).all() or diag_sample.observed_max_mm_day.lt(0).any()
    or not diag_sample.first_observed_max_date.between(pd.Timestamp(A1_LEVEL_REFERENCE_PERIOD[0]),pd.Timestamp(A1_LEVEL_REFERENCE_PERIOD[1])).all()
    or not diag_sample.last_observed_max_date.between(pd.Timestamp(A1_LEVEL_REFERENCE_PERIOD[0]),pd.Timestamp(A1_LEVEL_REFERENCE_PERIOD[1])).all()):
    raise ValueError("Mẫu đỉnh năm khác phạm vi trước test.")


def describe_gumbel_sample(values,location,scale):
    values=np.sort(np.asarray(values,dtype=float));n=len(values)
    if n<2 or not np.isfinite(values).all() or (values<0).any() or not np.isfinite([location,scale]).all() or scale<=0:
        raise ValueError("CDF chẩn đoán cần mẫu/fit Gumbel xác định.")
    cdf=gumbel_r.cdf(values,loc=location,scale=scale)
    if not np.isfinite(cdf).all() or ((cdf<0)|(cdf>1)).any():raise AssertionError("CDF ngoài [0,1].")
    rank=np.arange(1,n+1);positions=rank/(n+1)
    theoretical=gumbel_r.ppf(positions,loc=location,scale=scale)
    if not np.isfinite(theoretical).all():raise AssertionError("Phân vị chẩn đoán không hữu hạn.")
    # D+ và D− dùng hai phía của bước nhảy ECDF; ties giữ nguyên, không thêm jitter.
    ecdf_left=np.searchsorted(values,values,side="left")/n;ecdf_right=np.searchsorted(values,values,side="right")/n
    dplus=float(np.max(ecdf_right-cdf));dminus=float(np.max(cdf-ecdf_left))
    delta=values-theoretical
    metrics=dict(cdf_dplus=dplus,cdf_dminus=dminus,cdf_max_abs_distance=max(dplus,dminus),
        quantile_rmse_mm_day=float(np.sqrt(np.mean(delta**2))),quantile_rmse_over_scale=float(np.sqrt(np.mean(delta**2))/scale),
        largest_order_stat_delta_mm_day=float(delta[-1]),largest_order_stat_delta_over_scale=float(delta[-1]/scale),
        fitted_cdf_at_zero=float(gumbel_r.cdf(0,loc=location,scale=scale)))
    return metrics,rank,positions,cdf,theoretical,delta


diag_rows=[];diag_point_parts=[]
diag_metric_names=["cdf_dplus","cdf_dminus","cdf_max_abs_distance","quantile_rmse_mm_day","quantile_rmse_over_scale",
    "largest_order_stat_delta_mm_day","largest_order_stat_delta_over_scale","fitted_cdf_at_zero"]
for index,basin_id in enumerate(sorted(diag_ids,key=int),start=1):
    sample=diag_sample.loc[diag_sample.basin_id.eq(basin_id)].sort_values(["observed_max_mm_day","hydro_year_end"])
    values=sample.observed_max_mm_day.to_numpy(dtype=float);n=len(values);stored=diag_fits.loc[basin_id]
    fitted=fit_gumbel_lmoments(values)
    if n!=stored.n_years or fitted["status"]!=stored.status:raise ValueError(f"{basin_id}: mẫu fit khác 10.4.")
    for field in ["l1_mm_day","l2_mm_day","location_mm_day","scale_mm_day"]:
        if not np.isclose(fitted[field],stored[field],rtol=1e-10,atol=1e-10,equal_nan=True):raise ValueError(f"{basin_id}: {field} khác 10.4.")
    row=dict(basin_id=basin_id,n_years=n,below_10_years=bool(n<10),distinct_maxima=fitted["distinct_maxima"],fit_status=fitted["status"],
        used_complete_observed_years=int(sample.complete_observed_year.sum()),
        used_years_with_missing_months=int(sample.months_with_observations.lt(12).sum()),
        sample_min_mm_day=float(values.min()) if n else np.nan,sample_max_mm_day=float(values.max()) if n else np.nan,
        **dict.fromkeys(diag_metric_names,np.nan))
    if fitted["status"]=="fit_defined":
        metrics,ranks,positions,cdf,theoretical,delta=describe_gumbel_sample(values,fitted["location_mm_day"],fitted["scale_mm_day"])
        row.update(metrics)
        diag_point_parts.append(pd.DataFrame(dict(basin_id=basin_id,hydro_year_end=sample.hydro_year_end.to_numpy(),
            rank=ranks,n_years=n,observed_max_mm_day=values,weibull_position=positions,fitted_cdf=cdf,
            ecdf_before_value=np.searchsorted(values,values,side="left")/n,ecdf_at_value=np.searchsorted(values,values,side="right")/n,
            fitted_quantile_mm_day=theoretical,order_stat_delta_mm_day=delta)))
    diag_rows.append(row)
    if index%50==0 or index==len(diag_ids):print(f"Đã đo độ khớp CDF mô tả {index}/{len(diag_ids)} ID")
A1_FLOOD_FIT_DIAGNOSTICS=pd.DataFrame(diag_rows)
diag_point_columns=["basin_id","hydro_year_end","rank","n_years","observed_max_mm_day","weibull_position","fitted_cdf",
    "ecdf_before_value","ecdf_at_value","fitted_quantile_mm_day","order_stat_delta_mm_day"]
A1_FLOOD_FIT_POINTS=pd.concat(diag_point_parts,ignore_index=True) if diag_point_parts else pd.DataFrame(columns=diag_point_columns)
diag_summary=[]
for short,part in A1_FLOOD_FIT_DIAGNOSTICS.groupby("below_10_years",sort=True):
    for metric in ["cdf_max_abs_distance","quantile_rmse_over_scale","largest_order_stat_delta_over_scale","fitted_cdf_at_zero"]:
        values=part[metric].dropna()
        diag_summary.append(dict(below_10_years=short,metric=metric,basins=len(part),defined_basins=len(values),
            min=values.min(),p05=values.quantile(.05),median=values.median(),p95=values.quantile(.95),max=values.max()))
A1_FLOOD_FIT_DIAGNOSTIC_SUMMARY=pd.DataFrame(diag_summary)
diag_profile_ids=["55","324","600","623","742","758","800","604"]
A1_FLOOD_FIT_DIAGNOSTIC_PROFILES=A1_FLOOD_FIT_DIAGNOSTICS.loc[A1_FLOOD_FIT_DIAGNOSTICS.basin_id.isin(diag_profile_ids)].copy()

fig,axes=plt.subplots(2,3,figsize=(15,9))
diag_plot_ids=["55","324","742","758","800","604"]
for ax,basin_id in zip(axes.flat,diag_plot_ids):
    ax.plot([0,1],[0,1],color="gray",linestyle=":",label="Equal CDFs")
    part=A1_FLOOD_FIT_POINTS.loc[A1_FLOOD_FIT_POINTS.basin_id.eq(basin_id)].sort_values("rank")
    row=A1_FLOOD_FIT_DIAGNOSTICS.loc[A1_FLOOD_FIT_DIAGNOSTICS.basin_id.eq(basin_id)]
    if len(part):
        n=len(part);ax.step(np.r_[0.,part.fitted_cdf.to_numpy(),1.],np.r_[0.,np.arange(1,n+1)/n,1.],where="post",color="#2283a5",label="Empirical CDF")
        ax.set_title(f"ID {basin_id}: n={n}, D={row.iloc[0].cdf_max_abs_distance:.3f}",fontsize=11)
    else:
        ax.set_title("ID "+basin_id);ax.text(.05,.85,"No defined fit/sample",transform=ax.transAxes,fontsize=9)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Fitted Gumbel CDF",ylabel="Empirical CDF of annual maxima");ax.grid(alpha=.2)
handles,labels=next((ax.get_legend_handles_labels() for ax in axes.flat if len(ax.get_legend_handles_labels()[0])==2),axes.flat[0].get_legend_handles_labels())
fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.09),ncol=2,fontsize=9)
fig.suptitle("Primary annual-maxima sample versus fitted Gumbel: descriptive P–P plots",fontsize=14)
fig.text(.5,.02,"Same pre-test annual maxima and L-moments parameters as 10.4; ties retained and missing peaks not reconstructed.\n"
    "D is the maximum CDF discrepancy on the fitted sample, not a p-value or an acceptance criterion.\n"
    "Fit/sample-size effects, incomplete seasons, temporal changes and dependence remain unresolved; no distribution or threshold is selected here.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.17,1,.94));diag_figure=FIGURE_DIR/"A1_P10_gumbel_cdf_diagnostics.png"
fig.savefig(diag_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("flood_fit_diagnostics",A1_FLOOD_FIT_DIAGNOSTICS),("flood_fit_points",A1_FLOOD_FIT_POINTS),
    ("flood_fit_diagnostic_summary",A1_FLOOD_FIT_DIAGNOSTIC_SUMMARY),("flood_fit_diagnostic_profiles",A1_FLOOD_FIT_DIAGNOSTIC_PROFILES)]:
    table.to_csv(TABLE_DIR/f"A1_P10_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Khoảng cách CDF và sai khác phân vị trong mẫu; từng ID trọng số bằng nhau, không phải kiểm định:");display(A1_FLOOD_FIT_DIAGNOSTIC_SUMMARY)
    print("(2) Hồ sơ cố định; F(0) là khối xác suất dưới 0 của Gumbel liên tục đã fit, không phải số năm Q âm quan trắc:");display(A1_FLOOD_FIT_DIAGNOSTIC_PROFILES)
    print("(3) Mười D lớn nhất trong nhóm n≥10; chỉ xếp mô tả, không áp ngưỡng chấp nhận hoặc lọc ID:")
    display(A1_FLOOD_FIT_DIAGNOSTICS.loc[~A1_FLOOD_FIT_DIAGNOSTICS.below_10_years&A1_FLOOD_FIT_DIAGNOSTICS.cdf_max_abs_distance.notna()].sort_values(
        ["cdf_max_abs_distance","basin_id"],ascending=[False,True]).head(10))
    print("(4) Trạng thái và số đỉnh dùng; fit không xác định giữ riêng:")
    display(A1_FLOOD_FIT_DIAGNOSTICS.groupby(["fit_status","below_10_years"],sort=False).agg(basins=("basin_id","size"),annual_maxima=("n_years","sum")).reset_index())
    display(A1_FLOOD_FIT_DIAGNOSTICS.loc[A1_FLOOD_FIT_DIAGNOSTICS.fit_status.ne("fit_defined")])
print("Đã lưu hình:",diag_figure,"; bốn CSV vào:",TABLE_DIR)
print(f"Thời gian đo độ khớp mô tả Gumbel: {time.perf_counter()-started_fitdiag:.1f} giây.")
print("CẢNH BÁO: D dùng chính mẫu đã fit tham số; không có p-value/kiểm định chấp nhận, không xếp chất lượng trạm theo D. Sai khác ở điểm lớn nhất dùng vị trí Weibull hữu hạn, không phải lỗi dự báo. F(0) chỉ mô tả miền hỗ trợ của mô hình, không tự chặn hoặc đổi phân phối.")

# %% [markdown]
# ### 10.7. Số ngày quan trắc vượt ngưỡng Gumbel cố định theo kỳ/năm
#
# Dùng >, ngày bằng ngưỡng đếm riêng; giữ ngày thiếu và ID không có ngày đủ điều kiện ngoài mẫu số tỷ lệ.

# %% Phần 10.7 - Số ngày quan trắc vượt ngưỡng Gumbel cố định theo kỳ/năm
# Dùng >, ngày bằng ngưỡng đếm riêng; giữ ngày thiếu và ID không có ngày đủ điều kiện ngoài mẫu số tỷ lệ.
# Không fit lại bằng test, không lọc theo cửa sổ mô hình; ngày vượt không phải số trận lũ độc lập.
started_flood_days=time.perf_counter()
flood_day_ids=list(map(str,A1_QMAX_VALUE_IDS));flood_day_calendar=pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
flood_day_values=np.asarray(A1_QMAX_VALUES_MM_DAY,dtype=float)
flood_day_periods=np.asarray(A1_FLOOD_RETURN_PERIODS,dtype=float)
if (A1_FLOOD_PRIMARY_POLICY!="half_calendar_primary" or tuple(A1_LEVEL_REFERENCE_PERIOD)!=("1981-01-01","2013-12-31")
    or not np.array_equal(flood_day_periods,[1.5,2.,5.,10.]) or set(PERIODS)!={"train","validation","test"}
    or len(flood_day_ids)!=len(set(flood_day_ids)) or set(flood_day_ids)!=set(map(str,selected_ids))
    or flood_day_values.shape!=(len(flood_day_ids),len(flood_day_calendar)) or flood_day_calendar.has_duplicates
    or not flood_day_calendar.equals(pd.date_range(flood_day_calendar.min(),flood_day_calendar.max(),freq="D"))
    or np.isinf(flood_day_values).any() or (flood_day_values<0).any()):raise ValueError("Giao thức/cache khác các phần đã kiểm.")
flood_day_scopes={"reference_1981_2013":A1_LEVEL_REFERENCE_PERIOD,**PERIODS};flood_day_positions={}
for scope,(start,end) in flood_day_scopes.items():
    pos=np.flatnonzero((flood_day_calendar>=start)&(flood_day_calendar<=end))
    if not flood_day_calendar[pos].equals(pd.date_range(start,end,freq="D")):raise ValueError(f"Thiếu lịch {scope}.")
    flood_day_positions[scope]=pos

# Tái tạo số ngày/max từng năm trong reference để không dùng ngưỡng của một bản cache khác.
flood_day_years=A1_HYDRO_YEAR_COVERAGE.copy(deep=True)
if (set(flood_day_years.basin_id)!=set(flood_day_ids) or flood_day_years.duplicated(["basin_id","hydro_year_end"]).any()
    or not flood_day_years.meets_half_calendar_rule.eq(2*flood_day_years.valid_reference_days>=flood_day_years.full_year_days).all()):
    raise ValueError("Bảng năm thủy văn khác 10.3.")
ref_pos=flood_day_positions["reference_1981_2013"];ref_dates=flood_day_calendar[ref_pos]
ref_years=ref_dates.year+(ref_dates.month>=10).astype(int)
if set(ref_years)!=set(flood_day_years.hydro_year_end):raise ValueError("Nhãn năm thủy văn không khớp reference.")
for year in sorted(set(ref_years)):
    raw=flood_day_values[:,ref_pos[np.flatnonzero(ref_years==year)]];finite=np.isfinite(raw);counts=finite.sum(axis=1)
    maxima=np.max(np.where(finite,raw,-np.inf),axis=1);maxima=np.where(counts>0,maxima,np.nan)
    stored=flood_day_years.loc[flood_day_years.hydro_year_end.eq(year)].set_index("basin_id").reindex(flood_day_ids)
    if not np.array_equal(counts,stored.valid_reference_days.to_numpy()) or not np.allclose(maxima,stored.observed_max_mm_day,rtol=1e-12,atol=1e-12,equal_nan=True):
        raise ValueError(f"Reference cache khác đỉnh/độ phủ 10.3 ở năm {year}.")
flood_day_thresholds=A1_FLOOD_RETURN_LEVELS.loc[A1_FLOOD_RETURN_LEVELS.policy.eq(A1_FLOOD_PRIMARY_POLICY)].copy(deep=True)
if (flood_day_thresholds.duplicated(["basin_id","return_period_years"]).any()
    or set(flood_day_thresholds.basin_id)!=set(flood_day_ids)
    or len(flood_day_thresholds)!=len(flood_day_ids)*len(flood_day_periods)):
    raise ValueError("Thiếu/trùng bảng ngưỡng mẫu chính.")


def count_fixed_flood_days(raw,threshold):
    raw=np.asarray(raw,dtype=float);finite=np.isfinite(raw);valid=int(finite.sum());defined=bool(np.isfinite(threshold))
    values=raw[finite];eligible=valid if defined else 0
    below=int((values<threshold).sum()) if defined else 0
    equal=int((values==threshold).sum()) if defined else 0
    above=int((values>threshold).sum()) if defined else 0
    if below+equal+above!=eligible:raise AssertionError("Các trạng thái không cộng khớp.")
    return dict(expected_days=len(raw),valid_days=valid,missing_days=len(raw)-valid,eligible_valid_days=eligible,
        valid_days_without_threshold=valid-eligible,below_days=below,equal_days=equal,above_days=above,
        above_pct_eligible=100*above/eligible if eligible else np.nan)


flood_day_rows=[];flood_day_annual=[]
for index,(basin_id,series) in enumerate(zip(flood_day_ids,flood_day_values),start=1):
    sample=flood_day_years.loc[flood_day_years.basin_id.eq(basin_id)&flood_day_years.meets_half_calendar_rule]
    fitted=fit_gumbel_lmoments(sample.observed_max_mm_day.to_numpy());n=fitted["n_years"]
    expected=gumbel_return_levels(fitted["location_mm_day"],fitted["scale_mm_day"],flood_day_periods) if fitted["status"]=="fit_defined" else np.full(len(flood_day_periods),np.nan)
    thresholds=flood_day_thresholds.loc[flood_day_thresholds.basin_id.eq(basin_id)].set_index("return_period_years").reindex(flood_day_periods)
    if (not thresholds.n_years.eq(n).all() or not thresholds.fit_status.eq(fitted["status"]).all()
        or not np.allclose(expected,thresholds.threshold_mm_day.to_numpy(),rtol=1e-10,atol=1e-10,equal_nan=True)):
        raise ValueError(f"{basin_id}: ngưỡng/n/trạng thái không khớp mẫu 10.4.")
    if np.isfinite(expected).all() and not (np.diff(expected)>0).all():raise AssertionError("Ngưỡng không tăng theo T.")
    for scope,pos in flood_day_positions.items():
        raw=series[pos];years=flood_day_calendar[pos].year
        annual_positions={year:np.flatnonzero(years==year) for year in sorted(set(years))}
        previous_above=None
        for period,threshold in zip(flood_day_periods,expected):
            counts=count_fixed_flood_days(raw,threshold)
            if np.isfinite(threshold) and previous_above is not None and counts["above_days"]>previous_above:raise AssertionError("Số ngày vượt tăng khi T tăng.")
            previous_above=counts["above_days"] if np.isfinite(threshold) else None
            common=dict(basin_id=basin_id,scope=scope,return_period_years=float(period),threshold_mm_day=float(threshold),n_fit_years=n,
                below_10_fit_years=bool(n<10),fit_status=fitted["status"],threshold_defined=bool(np.isfinite(threshold)))
            flood_day_rows.append(dict(**common,**counts))
            for year,local in annual_positions.items():flood_day_annual.append(dict(**common,year=int(year),**count_fixed_flood_days(raw[local],threshold)))
    if index%50==0 or index==len(flood_day_ids):print(f"Đã đếm ngày vượt ngưỡng Gumbel {index}/{len(flood_day_ids)} ID")
A1_FLOOD_EXCEEDANCE_BY_BASIN=pd.DataFrame(flood_day_rows)
A1_FLOOD_EXCEEDANCE_ANNUAL_BY_BASIN=pd.DataFrame(flood_day_annual)
flood_count_fields=["expected_days","valid_days","missing_days","eligible_valid_days","valid_days_without_threshold","below_days","equal_days","above_days"]
rebuilt=A1_FLOOD_EXCEEDANCE_ANNUAL_BY_BASIN.groupby(["basin_id","scope","return_period_years"])[flood_count_fields].sum().sort_index()
original=A1_FLOOD_EXCEEDANCE_BY_BASIN.set_index(["basin_id","scope","return_period_years"])[flood_count_fields].sort_index()
pd.testing.assert_frame_equal(rebuilt,original)
coverage=A1_FLOOD_EXCEEDANCE_BY_BASIN.loc[~A1_FLOOD_EXCEEDANCE_BY_BASIN.scope.eq("reference_1981_2013")&A1_FLOOD_EXCEEDANCE_BY_BASIN.return_period_years.eq(1.5)].merge(
    A1_Q_PERIOD_PROFILE[["basin_id","period","expected_days","valid_days","missing_days"]],left_on=["basin_id","scope"],right_on=["basin_id","period"],validate="one_to_one",suffixes=("","_8_1"),how="outer",indicator=True)
if not coverage._merge.eq("both").all() or any(not coverage[field].eq(coverage[field+"_8_1"]).all() for field in ["expected_days","valid_days","missing_days"]):
    raise AssertionError("Phủ nhãn khác 8.1.")


def summarize_fixed_flood_days(table,keys):
    rows=[]
    for group,part in table.groupby(keys,sort=False):
        group=group if isinstance(group,tuple) else (group,)
        counts={field:int(part[field].sum()) for field in flood_count_fields};eligible=counts["eligible_valid_days"]
        rows.append(dict(zip(keys,group),basins=len(part),basins_with_labels=int(part.valid_days.gt(0).sum()),
            basins_with_eligible_days=int(part.eligible_valid_days.gt(0).sum()),basins_with_above_days=int(part.above_days.gt(0).sum()),
            basins_with_eligible_but_no_above=int((part.eligible_valid_days.gt(0)&part.above_days.eq(0)).sum()),
            **counts,pooled_above_pct_eligible=100*counts["above_days"]/eligible if eligible else np.nan,
            median_basin_above_pct_eligible=part.above_pct_eligible.median()))
    return pd.DataFrame(rows)


A1_FLOOD_EXCEEDANCE_SUMMARY=summarize_fixed_flood_days(A1_FLOOD_EXCEEDANCE_BY_BASIN,["scope","return_period_years"])
A1_FLOOD_EXCEEDANCE_ANNUAL_SUMMARY=summarize_fixed_flood_days(A1_FLOOD_EXCEEDANCE_ANNUAL_BY_BASIN,["scope","year","return_period_years"])
A1_FLOOD_EXCEEDANCE_SHORT_SAMPLE_SUMMARY=summarize_fixed_flood_days(A1_FLOOD_EXCEEDANCE_BY_BASIN,["scope","return_period_years","below_10_fit_years"])
fig,axes=plt.subplots(2,2,figsize=(14,9))
for ax,period in zip(axes.flat,flood_day_periods):
    for scope,color in [("train","#2283a5"),("validation","#c87531"),("test","#8655ad")]:
        part=A1_FLOOD_EXCEEDANCE_BY_BASIN.loc[A1_FLOOD_EXCEEDANCE_BY_BASIN.scope.eq(scope)&A1_FLOOD_EXCEEDANCE_BY_BASIN.return_period_years.eq(period)]
        values=np.sort(part.above_pct_eligible.dropna().to_numpy())
        if len(values):ax.step(np.r_[0.,values],np.r_[0.,100*np.arange(1,len(values)+1)/len(values)],where="post",color=color,label=f"{scope.title()} (n={len(values)})")
    ax.set(title=f"Fixed {period:g}-year threshold",xlabel="Observed days above threshold / eligible days (%)",ylabel="Basins at or below rate (%)",ylim=(0,101))
    ax.set_xscale("symlog",linthresh=.1);ax.set_xlim(left=0);ax.grid(alpha=.2)
    if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
fig.suptitle("Observed daily qmax above fixed pre-test Gumbel return levels",fontsize=14)
fig.text(.5,.025,"Strict > comparison; ties counted separately. Rates use observed days with defined thresholds, with equal basin weight in ECDFs.\n"
    "No window/model filtering or threshold refitting; basins without eligible observations have undefined rates.\n"
    "Exceeding days are not independent flood events. Reference overlaps train/validation; those scopes must not be added together.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.15,1,.94));flood_day_figure=FIGURE_DIR/"A1_P10_gumbel_exceedance_days.png"
fig.savefig(flood_day_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("flood_exceedance_by_basin",A1_FLOOD_EXCEEDANCE_BY_BASIN),("flood_exceedance_summary",A1_FLOOD_EXCEEDANCE_SUMMARY),
    ("flood_exceedance_annual_by_basin",A1_FLOOD_EXCEEDANCE_ANNUAL_BY_BASIN),("flood_exceedance_annual_summary",A1_FLOOD_EXCEEDANCE_ANNUAL_SUMMARY),
    ("flood_exceedance_short_sample_summary",A1_FLOOD_EXCEEDANCE_SHORT_SAMPLE_SUMMARY)]:
    table.to_csv(TABLE_DIR/f"A1_P10_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Ngày vượt ngưỡng > theo kỳ; ngày bằng ngưỡng đếm riêng, tỷ lệ chỉ dùng quan trắc có ngưỡng xác định:");display(A1_FLOOD_EXCEEDANCE_SUMMARY)
    print("(2) Ngày vượt theo năm lịch trong ba kỳ; đây không phải số đỉnh năm thủy văn hay số trận:")
    display(A1_FLOOD_EXCEEDANCE_ANNUAL_SUMMARY.loc[~A1_FLOOD_EXCEEDANCE_ANNUAL_SUMMARY.scope.eq("reference_1981_2013")].pivot(index=["scope","year"],columns="return_period_years",values="above_days"))
    print("(3) Hồ sơ cố định trong test, gồm cả các mẫu fit ngắn:")
    display(A1_FLOOD_EXCEEDANCE_BY_BASIN.loc[A1_FLOOD_EXCEEDANCE_BY_BASIN.scope.eq("test")&A1_FLOOD_EXCEEDANCE_BY_BASIN.basin_id.isin(["55","324","600","623","742","758","800","604"]),
        ["basin_id","return_period_years","threshold_mm_day","n_fit_years","expected_days","valid_days","missing_days","eligible_valid_days","equal_days","above_days","above_pct_eligible"]])
    print("(4) Test tách n fit gốc <10/≥10; mọi nhóm giữ nguyên ngưỡng và tập ID:")
    display(A1_FLOOD_EXCEEDANCE_SHORT_SAMPLE_SUMMARY.loc[A1_FLOOD_EXCEEDANCE_SHORT_SAMPLE_SUMMARY.scope.eq("test")])
    print("ID/kỳ không có ngày đủ điều kiện (không coi là tỷ lệ vượt bằng 0):")
    display(A1_FLOOD_EXCEEDANCE_BY_BASIN.loc[A1_FLOOD_EXCEEDANCE_BY_BASIN.eligible_valid_days.eq(0),["basin_id","scope","fit_status","valid_days","valid_days_without_threshold"]].drop_duplicates())
print("Đã lưu hình:",flood_day_figure,"; năm CSV vào:",TABLE_DIR)
print(f"Thời gian đếm ngày vượt ngưỡng Gumbel: {time.perf_counter()-started_flood_days:.1f} giây.")
print("CẢNH BÁO: thiếu quan trắc có thể đổi tỷ lệ được thấy; ngưỡng fit từ mẫu ngắn/thiếu mùa và giới hạn Gumbel ở 10.3–10.6 vẫn giữ. Không diễn giải 1/T là tỷ lệ ngày vượt, không cộng các mức T lồng nhau hoặc cộng reference với train/validation.")

# %% [markdown]
# ### 10.8. Đoạn ngày vượt liên tiếp và ngữ cảnh đầu/cuối đoạn
#
# Ngày thiếu tách đoạn, không nối qua khoảng trống; ranh giới kỳ không được coi là khởi đầu/kết thúc đã biết.

# %% Phần 10.8 - Đoạn ngày vượt liên tiếp và ngữ cảnh đầu/cuối đoạn
# Ngày thiếu tách đoạn, không nối qua khoảng trống; ranh giới kỳ không được coi là khởi đầu/kết thúc đã biết.
# Chỉ kiểm các đoạn quan trắc theo > ngưỡng cố định, chưa gộp USWRC hoặc gọi là trận độc lập.
started_flood_runs=time.perf_counter()
run_ids=list(map(str,A1_QMAX_VALUE_IDS));run_calendar=pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
run_values=np.asarray(A1_QMAX_VALUES_MM_DAY,dtype=float);run_periods=np.asarray(A1_FLOOD_RETURN_PERIODS,dtype=float)
run_scopes={"reference_1981_2013":A1_LEVEL_REFERENCE_PERIOD,**PERIODS};run_positions={}
if (A1_FLOOD_PRIMARY_POLICY!="half_calendar_primary" or not np.array_equal(run_periods,[1.5,2.,5.,10.])
    or len(run_ids)!=len(set(run_ids)) or set(run_ids)!=set(map(str,selected_ids))
    or run_values.shape!=(len(run_ids),len(run_calendar)) or run_calendar.has_duplicates
    or not run_calendar.equals(pd.date_range(run_calendar.min(),run_calendar.max(),freq="D"))
    or np.isinf(run_values).any() or (run_values<0).any()):raise ValueError("Cache/giao thức khác 10.7.")
for scope,(start,end) in run_scopes.items():
    pos=np.flatnonzero((run_calendar>=start)&(run_calendar<=end))
    if not run_calendar[pos].equals(pd.date_range(start,end,freq="D")):raise ValueError(f"Thiếu lịch {scope}.")
    run_positions[scope]=pos
run_source=A1_FLOOD_EXCEEDANCE_BY_BASIN.copy(deep=True).set_index(["basin_id","scope","return_period_years"],verify_integrity=True)
run_expected_index=pd.MultiIndex.from_product([run_ids,list(run_scopes),run_periods],names=run_source.index.names)
if len(run_source)!=len(run_expected_index) or len(run_source.index.symmetric_difference(run_expected_index)):
    raise ValueError("Bảng ngày vượt không đủ đúng các ID/kỳ/T của 10.7.")
run_thresholds=A1_FLOOD_RETURN_LEVELS.loc[A1_FLOOD_RETURN_LEVELS.policy.eq(A1_FLOOD_PRIMARY_POLICY)].copy(deep=True).set_index(["basin_id","return_period_years"],verify_integrity=True)


def contiguous_above_runs(raw,dates,threshold):
    raw=np.asarray(raw,dtype=float);dates=pd.DatetimeIndex(dates)
    if (raw.ndim!=1 or len(raw)!=len(dates) or dates.has_duplicates
        or (len(dates) and not dates.equals(pd.date_range(dates.min(),dates.max(),freq="D")))
        or np.isinf(raw).any() or (raw<0).any()):raise ValueError("Đoạn cần lịch ngày đầy đủ và nhãn hợp lệ/NaN.")
    if not np.isfinite(threshold):return []
    above=np.isfinite(raw)&(raw>threshold)
    starts=np.flatnonzero(above&~np.r_[False,above[:-1]])
    ends=np.flatnonzero(above&~np.r_[above[1:],False])
    if len(starts)!=len(ends):raise AssertionError("Đầu/cuối đoạn không khớp.")
    rows=[]
    for number,(left,right) in enumerate(zip(starts,ends),start=1):
        values=raw[left:right+1];peak=float(values.max());peaks=np.flatnonzero(values==peak)
        before="scope_boundary" if left==0 else "missing_previous_day" if not np.isfinite(raw[left-1]) else "observed_not_above"
        after="scope_boundary" if right==len(raw)-1 else "missing_next_day" if not np.isfinite(raw[right+1]) else "observed_not_above"
        onset=before=="observed_not_above";ending=after=="observed_not_above"
        if not above[left:right+1].all() or (dates[right]-dates[left]).days+1!=right-left+1:raise AssertionError("Đoạn không liên tiếp.")
        rows.append(dict(run_number=number,start_date=dates[left],end_date=dates[right],run_days=int(right-left+1),
            observed_peak_mm_day=peak,first_peak_date=dates[left+peaks[0]],last_peak_date=dates[left+peaks[-1]],days_at_peak=len(peaks),
            left_context=before,right_context=after,onset_observed=onset,ending_observed=ending,both_contexts_observed=onset and ending))
    if sum(row["run_days"] for row in rows)!=int(above.sum()):raise AssertionError("Độ dài đoạn không tái tạo số ngày vượt.")
    return rows


run_details=[];run_basin_rows=[]
for index,(basin_id,series) in enumerate(zip(run_ids,run_values),start=1):
    for scope,pos in run_positions.items():
        raw=series[pos];dates=run_calendar[pos]
        for period in run_periods:
            baseline=run_source.loc[(basin_id,scope,period)];stored=run_thresholds.loc[(basin_id,period)]
            threshold=float(baseline.threshold_mm_day)
            if (not np.isclose(threshold,stored.threshold_mm_day,rtol=1e-12,atol=1e-12,equal_nan=True)
                or baseline.n_fit_years!=stored.n_years or baseline.fit_status!=stored.fit_status):raise ValueError("Ngưỡng/n/trạng thái khác 10.4/10.7.")
            counts=count_fixed_flood_days(raw,threshold)
            for field in ["expected_days","valid_days","missing_days","eligible_valid_days","valid_days_without_threshold","below_days","equal_days","above_days"]:
                if counts[field]!=baseline[field]:raise ValueError(f"{basin_id}/{scope}/{period}: {field} khác 10.7.")
            runs=contiguous_above_runs(raw,dates,threshold)
            common=dict(basin_id=basin_id,scope=scope,return_period_years=float(period),threshold_mm_day=threshold,
                n_fit_years=int(baseline.n_fit_years),below_10_fit_years=bool(baseline.below_10_fit_years),fit_status=baseline.fit_status)
            run_details.extend(dict(**common,**row) for row in runs)
            durations=np.array([row["run_days"] for row in runs],dtype=int)
            both=sum(row["onset_observed"] and row["ending_observed"] for row in runs)
            onset_only=sum(row["onset_observed"] and not row["ending_observed"] for row in runs)
            ending_only=sum(not row["onset_observed"] and row["ending_observed"] for row in runs)
            neither=sum(not row["onset_observed"] and not row["ending_observed"] for row in runs)
            run_basin_rows.append(dict(**common,expected_days=counts["expected_days"],valid_days=counts["valid_days"],
                missing_days=counts["missing_days"],eligible_valid_days=counts["eligible_valid_days"],above_days=counts["above_days"],
                run_count=len(runs),run_days_sum=int(durations.sum()),both_contexts_observed_runs=both,
                only_onset_observed_runs=onset_only,only_ending_observed_runs=ending_only,neither_context_observed_runs=neither,
                scope_boundary_touching_runs=sum(row["left_context"]=="scope_boundary" or row["right_context"]=="scope_boundary" for row in runs),
                missing_neighbor_runs=sum(row["left_context"]=="missing_previous_day" or row["right_context"]=="missing_next_day" for row in runs),
                median_run_days=float(np.median(durations)) if len(durations) else np.nan,max_run_days=int(durations.max()) if len(durations) else np.nan))
    if index%50==0 or index==len(run_ids):print(f"Đã kiểm đoạn vượt liên tiếp {index}/{len(run_ids)} ID")
run_detail_columns=list(common)+["run_number","start_date","end_date","run_days","observed_peak_mm_day","first_peak_date","last_peak_date","days_at_peak",
    "left_context","right_context","onset_observed","ending_observed","both_contexts_observed"]
A1_FLOOD_RUN_DETAILS=pd.DataFrame(run_details,columns=run_detail_columns)
A1_FLOOD_RUN_BY_BASIN=pd.DataFrame(run_basin_rows)
if not A1_FLOOD_RUN_BY_BASIN.run_days_sum.eq(A1_FLOOD_RUN_BY_BASIN.above_days).all():raise AssertionError("Tổng độ dài khác ngày vượt 10.7.")
context_fields=["both_contexts_observed_runs","only_onset_observed_runs","only_ending_observed_runs","neither_context_observed_runs"]
if not A1_FLOOD_RUN_BY_BASIN[context_fields].sum(axis=1).eq(A1_FLOOD_RUN_BY_BASIN.run_count).all():raise AssertionError("Nhóm ngữ cảnh không cộng khớp.")
run_summary=[]
for (scope,period),part in A1_FLOOD_RUN_BY_BASIN.groupby(["scope","return_period_years"],sort=False):
    details=A1_FLOOD_RUN_DETAILS.loc[A1_FLOOD_RUN_DETAILS.scope.eq(scope)&A1_FLOOD_RUN_DETAILS.return_period_years.eq(period)]
    duration=details.run_days
    run_summary.append(dict(scope=scope,return_period_years=period,basins=len(part),basins_with_eligible_days=int(part.eligible_valid_days.gt(0).sum()),
        basins_with_runs=int(part.run_count.gt(0).sum()),basins_eligible_no_runs=int((part.eligible_valid_days.gt(0)&part.run_count.eq(0)).sum()),
        basins_no_eligible_days=int(part.eligible_valid_days.eq(0).sum()),above_days=int(part.above_days.sum()),run_count=int(part.run_count.sum()),
        **{field:int(part[field].sum()) for field in context_fields+["scope_boundary_touching_runs","missing_neighbor_runs"]},
        pooled_run_days_median=duration.median() if len(duration) else np.nan,
        pooled_run_days_p95=duration.quantile(.95) if len(duration) else np.nan,max_run_days=duration.max() if len(duration) else np.nan))
A1_FLOOD_RUN_SUMMARY=pd.DataFrame(run_summary)
A1_FLOOD_RUN_CONTEXT_SUMMARY=A1_FLOOD_RUN_DETAILS.groupby(["scope","return_period_years","left_context","right_context"],sort=False).agg(
    runs=("run_number","size"),above_days=("run_days","sum")).reset_index()

fig,axes=plt.subplots(1,2,figsize=(15,6))
run_test=A1_FLOOD_RUN_SUMMARY.loc[A1_FLOOD_RUN_SUMMARY.scope.eq("test")].set_index("return_period_years").reindex(run_periods)
bottom=np.zeros(len(run_periods))
for field,color,label in zip(context_fields,["#2283a5","#c87531","#8655ad","#999999"],
    ["Both sides observed","Only onset observed","Only ending observed","Neither side observed"]):
    values=run_test[field].to_numpy(dtype=float);axes[0].bar(np.arange(len(run_periods)),values,bottom=bottom,color=color,label=label);bottom+=values
axes[0].set_xticks(np.arange(len(run_periods)),[f"{period:g}" for period in run_periods])
axes[0].set(xlabel="Fixed return-period threshold (years)",ylabel="Observed contiguous runs",title="Test: run counts by observed boundary context")
axes[0].legend(fontsize=8);axes[0].grid(axis="y",alpha=.2)
for period,color in zip(run_periods,["#2283a5","#c87531","#8655ad","#555555"]):
    values=np.sort(A1_FLOOD_RUN_DETAILS.loc[A1_FLOOD_RUN_DETAILS.scope.eq("test")&A1_FLOOD_RUN_DETAILS.return_period_years.eq(period),"run_days"].to_numpy())
    if len(values):axes[1].step(np.r_[0.,values],np.r_[0.,100*np.arange(1,len(values)+1)/len(values)],where="post",color=color,label=f"{period:g} years (n={len(values)})")
axes[1].set(xlabel="Observed contiguous days per run",ylabel="Runs at or below length (%)",title="Test: observed run lengths (including cut runs)",ylim=(0,101))
axes[1].set_xscale("symlog",linthresh=1);axes[1].set_xlim(left=0);axes[1].grid(alpha=.2)
if axes[1].get_legend_handles_labels()[0]:axes[1].legend(fontsize=8)
fig.suptitle("Observed above-threshold runs: gaps and scope boundaries retained",fontsize=14)
fig.text(.5,.025,"Strict > thresholds from 10.4, same observed days as 10.7. Missing days split runs; no bridging across gaps or scope boundaries.\n"
    "Onset/ending is known only with an observed non-above day immediately inside the scope. Run lengths may be truncated.\n"
    "These are contiguous observed blocks, not independent flood events. Duration ECDFs weight runs, not basins; nested thresholds overlap.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.18,1,.94));run_figure=FIGURE_DIR/"A1_P10_gumbel_exceedance_runs.png"
fig.savefig(run_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("flood_run_details",A1_FLOOD_RUN_DETAILS),("flood_run_by_basin",A1_FLOOD_RUN_BY_BASIN),
    ("flood_run_summary",A1_FLOOD_RUN_SUMMARY),("flood_run_context_summary",A1_FLOOD_RUN_CONTEXT_SUMMARY)]:
    table.to_csv(TABLE_DIR/f"A1_P10_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Đoạn vượt liên tiếp, tổng độ dài khớp ngày vượt 10.7; chưa là trận độc lập:");display(A1_FLOOD_RUN_SUMMARY)
    print("(2) Ngữ cảnh đầu/cuối các đoạn test; ranh giới kỳ và ngày thiếu không coi là ngày không vượt:")
    display(A1_FLOOD_RUN_CONTEXT_SUMMARY.loc[A1_FLOOD_RUN_CONTEXT_SUMMARY.scope.eq("test")])
    print("(3) Hồ sơ test cố định; không có ngày đủ điều kiện khác có quan trắc nhưng không có đoạn:")
    display(A1_FLOOD_RUN_BY_BASIN.loc[A1_FLOOD_RUN_BY_BASIN.scope.eq("test")&A1_FLOOD_RUN_BY_BASIN.basin_id.isin(["55","324","600","623","742","758","800","604"])])
    print("(4) Tối đa 20 đoạn test dài nhất; cùng chuỗi có thể xuất hiện ở nhiều T lồng nhau:")
    display(A1_FLOOD_RUN_DETAILS.loc[A1_FLOOD_RUN_DETAILS.scope.eq("test")].sort_values(
        ["run_days","basin_id","start_date","return_period_years"],ascending=[False,True,True,True]).head(20))
print("Đã lưu hình:",run_figure,"; bốn CSV vào:",TABLE_DIR)
print(f"Thời gian kiểm đoạn vượt liên tiếp: {time.perf_counter()-started_flood_runs:.1f} giây.")
print("CẢNH BÁO: hai đoạn cách ngày thiếu có thể thuộc cùng một trận chưa biết; đoạn có đủ hai phía quan trắc cũng chưa bảo đảm độc lập với đoạn khác. Ngưỡng cao hơn có thể tách một đoạn thành nhiều đoạn, nên số đoạn không buộc giảm theo T.")

# %% [markdown]
# ### 10.9. Gộp đoạn vượt theo khoảng cách đỉnh và mức nước rút đã định
#
# Dùng diện tích area_gov, ceil[5+ln(A mile²)] và đáy <0,75×đỉnh nhỏ; giữ ngưỡng, ngày thiếu và ranh giới kỳ.

# %% Phần 10.9 - Gộp đoạn vượt theo khoảng cách đỉnh và mức nước rút đã định
# Dùng diện tích area_gov, ceil[5+ln(A mile²)] và đáy <0,75×đỉnh nhỏ; giữ ngưỡng, ngày thiếu và ranh giới kỳ.
# Mỗi cụm giữ ngày bắt đầu đầu tiên, đỉnh lớn nhất; kiểm lại cặp cụm kề sau gộp, chưa ghép dự báo/chấm sự kiện.
started_uswrc=time.perf_counter()
us_ids=list(map(str,A1_QMAX_VALUE_IDS));us_dates=pd.DatetimeIndex(A1_QMAX_VALUE_CALENDAR)
us_values=np.asarray(A1_QMAX_VALUES_MM_DAY,dtype=float);us_periods=np.asarray(A1_FLOOD_RETURN_PERIODS,dtype=float)
us_scopes={"reference_1981_2013":A1_LEVEL_REFERENCE_PERIOD,**PERIODS}
if (A1_FLOOD_PRIMARY_POLICY!="half_calendar_primary" or not np.array_equal(us_periods,[1.5,2.,5.,10.])
    or len(us_ids)!=len(set(us_ids)) or set(us_ids)!=set(map(str,selected_ids))
    or us_values.shape!=(len(us_ids),len(us_dates)) or us_dates.has_duplicates
    or not us_dates.equals(pd.date_range(us_dates.min(),us_dates.max(),freq="D"))
    or np.isinf(us_values).any() or (us_values<0).any()):raise ValueError("Cache/ngưỡng khác 10.8.")
us_pos={scope:np.flatnonzero((us_dates>=start)&(us_dates<=end)) for scope,(start,end) in us_scopes.items()}
for scope,pos in us_pos.items():
    if not us_dates[pos].equals(pd.date_range(*us_scopes[scope],freq="D")):raise ValueError("Thiếu lịch kỳ.")
us_area=A1_QMAX_AREA_CONVERSION.copy(deep=True).set_index("basin_id",verify_integrity=True).reindex(us_ids)
us_area_values=pd.to_numeric(us_area.area_gov_km2,errors="raise").to_numpy(dtype=float)
if not (np.isfinite(us_area_values)&(us_area_values>0)).all():raise ValueError("area_gov cần hữu hạn, dương.")
if not np.allclose(us_area.m3_s_to_mm_day_factor,86.4/us_area_values,rtol=1e-12,atol=1e-12):raise ValueError("Diện tích khác hệ số đổi Q đã kiểm.")
A1_USWRC_PARAMETERS=pd.DataFrame(dict(basin_id=us_ids,area_gov_km2=us_area_values,area_square_miles=us_area_values/(1.609344**2)))
A1_USWRC_PARAMETERS["separation_days_real"]=5+np.log(A1_USWRC_PARAMETERS.area_square_miles)
if not (np.isfinite(A1_USWRC_PARAMETERS.separation_days_real)&A1_USWRC_PARAMETERS.separation_days_real.gt(0)).all():raise ValueError("Khoảng cách ngày không hợp lệ.")
A1_USWRC_PARAMETERS["minimum_peak_distance_days"]=np.ceil(A1_USWRC_PARAMETERS.separation_days_real).astype(int)
A1_USWRC_PARAMETERS["recession_fraction"]=.75
us_parameters=A1_USWRC_PARAMETERS.set_index("basin_id")
us_baseline=A1_FLOOD_RUN_BY_BASIN.copy(deep=True).set_index(["basin_id","scope","return_period_years"],verify_integrity=True)
us_expected=pd.MultiIndex.from_product([us_ids,list(us_scopes),us_periods],names=us_baseline.index.names)
if len(us_baseline)!=len(us_expected) or len(us_baseline.index.symmetric_difference(us_expected)):raise ValueError("Mẫu ID/kỳ/T khác 10.8.")


def group_uswrc_runs(raw,dates,runs,minimum_days,recession_fraction=.75):
    raw=np.asarray(raw,dtype=float);dates=pd.DatetimeIndex(dates)
    if minimum_days<1 or not 0<recession_fraction<1:raise ValueError("Tham số gộp không hợp lệ.")
    def single(row):return dict(members=[int(row["run_number"])],start_date=row["start_date"],end_date=row["end_date"],
        above_days=int(row["run_days"]),observed_peak_mm_day=float(row["observed_peak_mm_day"]),
        first_peak_date=row["first_peak_date"],last_peak_date=row["last_peak_date"],days_at_peak=int(row["days_at_peak"]),
        onset_observed=bool(row["onset_observed"]),ending_observed=bool(row["ending_observed"]))
    def assess(left,right):
        # Đồng đỉnh: khoảng cách gần nhất từ ngày max cuối bên trái đến ngày max đầu bên phải.
        a=int(dates.get_loc(left["last_peak_date"]));b=int(dates.get_loc(right["first_peak_date"]))
        if b<=a:raise AssertionError("Thứ tự đỉnh không hợp lệ.")
        middle=raw[a+1:b];missing=int((~np.isfinite(middle)).sum())
        valley=float(np.nanmin(middle)) if np.isfinite(middle).any() else np.nan
        cutoff=recession_fraction*min(left["observed_peak_mm_day"],right["observed_peak_mm_day"])
        time_ok=b-a>=minimum_days;recession_ok=bool(np.isfinite(valley) and valley<cutoff)
        decision="unresolved_missing" if missing else "separate" if time_ok and recession_ok else "merge_both" if not time_ok and not recession_ok else "merge_time" if not time_ok else "merge_recession"
        return dict(left_run_numbers="|".join(map(str,left["members"])),right_run_numbers="|".join(map(str,right["members"])),
            left_last_peak_date=left["last_peak_date"],right_first_peak_date=right["first_peak_date"],peak_distance_days=b-a,
            minimum_peak_distance_days=minimum_days,valley_mm_day=valley,recession_cutoff_mm_day=cutoff,
            time_condition_met=time_ok,recession_condition_met=recession_ok,missing_days_between_peaks=missing,decision=decision)
    def merge(left,right):
        peak=max(left["observed_peak_mm_day"],right["observed_peak_mm_day"])
        highest=[x for x in [left,right] if x["observed_peak_mm_day"]==peak]
        return dict(members=left["members"]+right["members"],start_date=left["start_date"],end_date=right["end_date"],
            above_days=left["above_days"]+right["above_days"],observed_peak_mm_day=peak,
            first_peak_date=min(x["first_peak_date"] for x in highest),last_peak_date=max(x["last_peak_date"] for x in highest),
            days_at_peak=sum(x["days_at_peak"] for x in highest),onset_observed=left["onset_observed"],ending_observed=right["ending_observed"])
    stack=[];audit=[]
    for row in runs:
        stack.append(single(row))
        while len(stack)>1:
            check=assess(stack[-2],stack[-1]);audit.append(dict(comparison_number=len(audit)+1,**check))
            if not check["decision"].startswith("merge_"):break
            right=stack.pop();left=stack.pop();stack.append(merge(left,right))
    boundaries=[]
    for j,(left,right) in enumerate(zip(stack[:-1],stack[1:]),start=1):
        check=assess(left,right)
        if check["decision"] not in {"separate","unresolved_missing"}:raise AssertionError("Còn cặp cụm phải gộp sau khi kiểm lại.")
        boundaries.append(dict(left_cluster=j,right_cluster=j+1,**check))
    return stack,audit,boundaries


us_clusters=[];us_membership=[];us_audit=[];us_boundaries=[];us_counts=[]
us_run_source=A1_FLOOD_RUN_DETAILS.copy(deep=True)
us_fixed_levels=A1_FLOOD_RETURN_LEVELS.loc[A1_FLOOD_RETURN_LEVELS.policy.eq(A1_FLOOD_PRIMARY_POLICY)].copy(deep=True).set_index(["basin_id","return_period_years"],verify_integrity=True)
us_metadata=["run_number","start_date","end_date","run_days","observed_peak_mm_day","first_peak_date","last_peak_date","days_at_peak",
    "left_context","right_context","onset_observed","ending_observed","both_contexts_observed"]
for index,(basin_id,series) in enumerate(zip(us_ids,us_values),start=1):
    minimum=int(us_parameters.loc[basin_id,"minimum_peak_distance_days"])
    for scope,pos in us_pos.items():
        raw=series[pos];dates=us_dates[pos]
        for period in us_periods:
            old=us_baseline.loc[(basin_id,scope,period)];threshold=float(old.threshold_mm_day);fixed=us_fixed_levels.loc[(basin_id,period)]
            if (not np.isclose(threshold,fixed.threshold_mm_day,rtol=1e-12,atol=1e-12,equal_nan=True)
                or old.n_fit_years!=fixed.n_years or old.fit_status!=fixed.fit_status):raise ValueError("Ngưỡng/n/trạng thái khác 10.4.")
            fresh=contiguous_above_runs(raw,dates,threshold)
            saved=us_run_source.loc[us_run_source.basin_id.eq(basin_id)&us_run_source.scope.eq(scope)&us_run_source.return_period_years.eq(period)].sort_values("run_number")
            if len(fresh)!=len(saved) or len(fresh)!=old.run_count or sum(x["run_days"] for x in fresh)!=old.above_days:raise ValueError("Số đoạn/ngày khác 10.8.")
            if fresh:pd.testing.assert_frame_equal(pd.DataFrame(fresh)[us_metadata].reset_index(drop=True),saved[us_metadata].reset_index(drop=True),check_dtype=False)
            common=dict(basin_id=basin_id,scope=scope,return_period_years=float(period),threshold_mm_day=threshold,
                n_fit_years=int(old.n_fit_years),below_10_fit_years=bool(old.below_10_fit_years),minimum_peak_distance_days=minimum)
            clusters,audit,boundaries=group_uswrc_runs(raw,dates,fresh,minimum)
            if sum(x["decision"].startswith("merge_") for x in audit)!=len(fresh)-len(clusters):raise AssertionError("Số thao tác gộp không khớp mức giảm.")
            assigned=[]
            for number,cluster in enumerate(clusters,start=1):
                members=cluster["members"];assigned+=members
                span=(cluster["end_date"]-cluster["start_date"]).days+1
                us_clusters.append(dict(**common,cluster_number=number,raw_run_numbers="|".join(map(str,members)),raw_run_count=len(members),
                    **{k:v for k,v in cluster.items() if k!="members"},span_days=span,non_above_days_inside_span=span-cluster["above_days"],
                    both_contexts_observed=cluster["onset_observed"] and cluster["ending_observed"]))
                for member in members:us_membership.append(dict(**common,run_number=member,cluster_number=number))
            if sorted(assigned)!=[x["run_number"] for x in fresh] or sum(x["above_days"] for x in clusters)!=old.above_days:raise AssertionError("Gộp mất/trùng đoạn hoặc ngày vượt.")
            us_audit.extend(dict(**{k:v for k,v in common.items() if k!="minimum_peak_distance_days"},**row) for row in audit)
            us_boundaries.extend(dict(**{k:v for k,v in common.items() if k!="minimum_peak_distance_days"},**row) for row in boundaries)
            us_counts.append(dict(**common,eligible_valid_days=int(old.eligible_valid_days),above_days=int(old.above_days),raw_runs=len(fresh),
                clusters=len(clusters),runs_removed=len(fresh)-len(clusters),reduction_pct=100*(len(fresh)-len(clusters))/len(fresh) if fresh else np.nan,
                clusters_with_observed_onset=sum(x["onset_observed"] for x in clusters),
                clusters_with_both_contexts=sum(x["onset_observed"] and x["ending_observed"] for x in clusters),
                unresolved_missing_boundaries=sum(x["decision"]=="unresolved_missing" for x in boundaries)))
    if index%50==0 or index==len(us_ids):print(f"Đã kiểm gộp theo tiêu chí USWRC {index}/{len(us_ids)} ID")
us_common_columns=list(common)
us_check_columns=["left_run_numbers","right_run_numbers","left_last_peak_date","right_first_peak_date","peak_distance_days",
    "minimum_peak_distance_days","valley_mm_day","recession_cutoff_mm_day","time_condition_met","recession_condition_met","missing_days_between_peaks","decision"]
us_audit_common=[k for k in us_common_columns if k!="minimum_peak_distance_days"]
A1_USWRC_CLUSTERS=pd.DataFrame(us_clusters,columns=us_common_columns+["cluster_number","raw_run_numbers","raw_run_count","start_date","end_date",
    "above_days","observed_peak_mm_day","first_peak_date","last_peak_date","days_at_peak","onset_observed","ending_observed","span_days","non_above_days_inside_span","both_contexts_observed"])
A1_USWRC_MEMBERSHIP=pd.DataFrame(us_membership,columns=us_common_columns+["run_number","cluster_number"])
A1_USWRC_COMPARISONS=pd.DataFrame(us_audit,columns=us_audit_common+["comparison_number"]+us_check_columns)
A1_USWRC_BOUNDARIES=pd.DataFrame(us_boundaries,columns=us_audit_common+["left_cluster","right_cluster"]+us_check_columns)
A1_USWRC_COUNTS=pd.DataFrame(us_counts)
us_summary=[]
for (scope,period),part in A1_USWRC_COUNTS.groupby(["scope","return_period_years"],sort=False):
    raw_count=int(part.raw_runs.sum());groups=int(part.clusters.sum())
    us_summary.append(dict(scope=scope,return_period_years=period,basins=len(part),basins_with_clusters=int(part.clusters.gt(0).sum()),
        eligible_valid_days=int(part.eligible_valid_days.sum()),above_days=int(part.above_days.sum()),raw_runs=raw_count,clusters=groups,
        runs_removed=raw_count-groups,pooled_reduction_pct=100*(raw_count-groups)/raw_count if raw_count else np.nan,
        median_basin_reduction_pct=part.reduction_pct.median(),clusters_with_observed_onset=int(part.clusters_with_observed_onset.sum()),
        clusters_with_both_contexts=int(part.clusters_with_both_contexts.sum()),unresolved_missing_boundaries=int(part.unresolved_missing_boundaries.sum())))
A1_USWRC_SUMMARY=pd.DataFrame(us_summary)

fig,axes=plt.subplots(1,2,figsize=(15,6));us_test=A1_USWRC_SUMMARY.loc[A1_USWRC_SUMMARY.scope.eq("test")].set_index("return_period_years").reindex(us_periods)
axis_x=np.arange(len(us_periods));axes[0].bar(axis_x-.18,us_test.raw_runs,.36,label="Contiguous runs",color="#2283a5")
axes[0].bar(axis_x+.18,us_test.clusters,.36,label="USWRC-rule clusters",color="#c87531")
axes[0].set_xticks(axis_x,[f"{t:g}" for t in us_periods]);axes[0].set(xlabel="Fixed return-period threshold (years)",ylabel="Observed runs / clusters",title="Test: before and after grouping");axes[0].legend(fontsize=9);axes[0].grid(axis="y",alpha=.2)
for period,color in zip(us_periods,["#2283a5","#c87531","#8655ad","#555555"]):
    values=np.sort(A1_USWRC_COUNTS.loc[A1_USWRC_COUNTS.scope.eq("test")&A1_USWRC_COUNTS.return_period_years.eq(period),"reduction_pct"].dropna().to_numpy())
    if len(values):axes[1].step(np.r_[0.,values],np.r_[0.,100*np.arange(1,len(values)+1)/len(values)],where="post",color=color,label=f"{period:g} years (n={len(values)})")
axes[1].set(xlabel="Reduction in contiguous-run count (%)",ylabel="Basins at or below reduction (%)",title="Test: basins with at least one run",xlim=(0,100),ylim=(0,101));axes[1].grid(alpha=.2)
if axes[1].get_legend_handles_labels()[0]:axes[1].legend(fontsize=8)
fig.suptitle("Fixed-threshold grouping by peak distance and recession criteria",fontsize=14)
fig.text(.5,.025,"Minimum distance = ceil[5 + ln(area in square miles)]; intervening minimum must be strictly below 0.75 of the smaller peak.\n"
    "Merged groups retain first onset and largest peak; adjacent groups are rechecked. Missing intervals block grouping and remain unresolved.\n"
    "Counts describe the stated rule on available daily qmax, not proof of physical/statistical independence or forecast skill.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.19,1,.94));us_figure=FIGURE_DIR/"A1_P10_uswrc_grouping.png"
fig.savefig(us_figure,dpi=150,bbox_inches="tight",facecolor="white");plt.show();plt.close(fig)
for suffix,table in [("uswrc_parameters",A1_USWRC_PARAMETERS),("uswrc_clusters",A1_USWRC_CLUSTERS),("uswrc_membership",A1_USWRC_MEMBERSHIP),
    ("uswrc_comparisons",A1_USWRC_COMPARISONS),("uswrc_boundaries",A1_USWRC_BOUNDARIES),("uswrc_counts",A1_USWRC_COUNTS),("uswrc_summary",A1_USWRC_SUMMARY)]:
    table.to_csv(TABLE_DIR/f"A1_P10_{suffix}.csv",index=False,encoding="utf-8-sig")
with pd.option_context("display.max_columns",None,"display.max_rows",None):
    print("(1) Trước/sau gộp theo quy tắc; bảo toàn ngày vượt, không gộp qua ngày thiếu:");display(A1_USWRC_SUMMARY)
    print("(2) Diện tích và khoảng cách ngày, natural log, 1 mile = 1.609344 km:");display(A1_USWRC_PARAMETERS.describe())
    print("(3) Hồ sơ test cố định:");display(A1_USWRC_COUNTS.loc[A1_USWRC_COUNTS.scope.eq("test")&A1_USWRC_COUNTS.basin_id.isin(["55","324","600","623","742","758","800","604"])])
    print("(4) Audit test: mọi phép so trung gian khác các ranh giới cuối còn lại:")
    if len(A1_USWRC_COMPARISONS):display(A1_USWRC_COMPARISONS.loc[A1_USWRC_COMPARISONS.scope.eq("test")].groupby(["return_period_years","decision"]).size().rename("comparisons").reset_index())
    if len(A1_USWRC_BOUNDARIES):display(A1_USWRC_BOUNDARIES.loc[A1_USWRC_BOUNDARIES.scope.eq("test")].groupby(["return_period_years","decision"]).size().rename("final_boundaries").reset_index())
    print("(5) Tối đa 15 cụm test gộp nhiều đoạn nhất; span có thể chứa ngày không vượt:")
    if len(A1_USWRC_CLUSTERS):display(A1_USWRC_CLUSTERS.loc[A1_USWRC_CLUSTERS.scope.eq("test")&A1_USWRC_CLUSTERS.raw_run_count.gt(1)].sort_values(["raw_run_count","basin_id","start_date"],ascending=[False,True,True]).head(15))
print("Đã lưu hình:",us_figure,"; bảy CSV vào:",TABLE_DIR)
print(f"Thời gian gộp theo tiêu chí USWRC: {time.perf_counter()-started_uswrc:.1f} giây.")
print("CẢNH BÁO: dùng qmax ngày, không phải thủy đồ tức thời; cụm theo quy tắc chưa chứng minh độc lập vật lý/thống kê. Khoảng thiếu và biên kỳ vẫn làm số quan trắc chưa đầy đủ. Không đổi ngưỡng, bỏ ID hoặc chọn quy tắc bằng test.")
