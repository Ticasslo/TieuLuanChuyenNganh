# =============================================================================
# Extended LamaH-CE (BiasCast, HESS 2026) - khảo sát dữ liệu (bước A1)
# Bài: Konold và cs., HESS 30:5067-5096 (2026). Dữ liệu Zenodo 17119635 (CC BY-NC 4.0),
#      kết quả thí nghiệm Zenodo 17292895 - đã lưu thành Kaggle Dataset "lamah-ce-ext"
#      (Workspace/01_Download/01_LamaHCEExt_Download.py).
# Notebook Kaggle "01_A1_LamaHCEExt_Explore": Accelerator = None (CPU), Internet tắt, Persistence = Files only,
# Add Input => Your Datasets => lamah-ce-ext. Mỗi Phần dán vào 1 cell, chạy lần lượt.
# Chạy được cả trên Colab/máy khác: đặt biến môi trường LAMAH_ROOT trỏ tới thư mục chứa dữ liệu đã giải nén.
# Chữ trong hình viết tiếng Anh; ghi chú, chú thích kết quả viết ở Document/03_Data/01_LamaHCE.md.
#
# MỤC LỤC
#   Phần 1 - Môi trường, đường dẫn, hằng số dùng chung
#   Phần 2 - Cấu trúc tệp và cột thật so với tên trong bài
#   Phần 3 - Thuộc tính và bản đồ 451 lưu vực
#   Phần 4 - Tỷ lệ thiếu theo kỳ và theo biến
#   Phần 5 - Phân phối từng biến
#   Phần 6 - So ECMWF HRES với tái phân tích cho 5 cặp biến (lệch miền)
#   Phần 7 - Số mẫu hợp lệ theo kỳ
#   Phần 8 - Tóm tắt và lưu kết quả
# =============================================================================


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
