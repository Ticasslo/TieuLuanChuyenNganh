# =============================================================================
# Bước 0 - Kiểm mô hình chung Sequential Forecast nhận 5 lõi (LSTM, GRU, Transformer, S4D, Mamba) trên Kaggle T4
# Mã thư viện: fork Ticasslo/neuralhydrology nhánh core-models (modelzoo/sf_cores.py, modelzoo/sequential_forecast.py,
# test/test_sequential_forecast.py); kiểm thử CPU đã đạt ở máy. Notebook này kiểm phần chỉ chạy được trên GPU và dữ liệu
# thật: kiểm thử của fork trên GPU, lõi LSTM so với SequentialForecastLSTM bằng trọng số tác giả BiasCast, Mamba
# (không nhìn tương lai, đúng trục, mambapy), số tham số 5 lõi, tốc độ.
# Kế hoạch: Document/01_Plan/02_Pipeline.md Mục 4.1, 4.2 (5 lõi, cùng tham số), 9.1 (mốc fork).
# Cài đặt: Accelerator = GPU T4 x2 (dùng GPU 0), Internet = On, Persistence = Files only,
# Environment = Pin to original environment, Add Input => lamah-ce-ext và Output notebook 01_0_MambaSSM_Build
# (Version 1, wheel mamba-ssm, causal-conv1d). Mỗi Phần dán vào 1 cell, chạy lần lượt.
#
# MỤC LỤC
#   Phần 1 - Cài fork nhánh core-models và mamba-ssm, kiểm môi trường
#   Phần 2 - Chạy toàn bộ kiểm thử của fork (pytest)
#   Phần 3 - Lõi LSTM của mô hình chung so với SequentialForecastLSTM bằng trọng số tác giả
#   Phần 4 - Lõi Mamba: không nhìn tương lai, đúng trục, công thức số tham số, mambapy
#   Phần 5 - Bảng số chiều cùng tham số của 5 lõi
#   Phần 6 - Huấn luyện ngắn 5 lõi trên dữ liệu thật: tốc độ, bộ nhớ, lặp lại từng bit
#   Phần 7 - Tóm tắt, môi trường, danh mục tệp, dọn thư mục tạm
# =============================================================================

# %% Phần 1 - Cài fork nhánh core-models và mamba-ssm, kiểm môi trường
# Clone fork đúng commit (Phần 2 cần thư mục test/, pip install git+ không cài test/) rồi pip install từ bản clone; cài
# wheel mamba-ssm theo cell Phần 9 của 01_0_MambaSSM_Build. In 3 bảng (thư viện, môi trường, điều kiện); chưa đạt => raise.
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display

NH_REPO = "https://github.com/Ticasslo/neuralhydrology.git"
NH_COMMIT = "fed53dca9815df963a5fa362558c448c769a6499"   # nhánh core-models = f229e02 (B0, B1) + mô hình chung 5 lõi
CHECK_LIBS = ["numpy", "pandas", "xarray", "torch", "scipy", "numba", "ruamel.yaml", "matplotlib", "tensorboard", "tqdm",
              "pytest"]
NH_MODULES = ["neuralhydrology.nh_run", "neuralhydrology.evaluation.tester", "neuralhydrology.datautils.gpu_loader",
              "neuralhydrology.modelzoo.sequential_forecast_lstm", "neuralhydrology.modelzoo.sequential_forecast",
              "neuralhydrology.modelzoo.sf_cores"]
TORCH_B0 = "2.11.0+cu128"                # torch của A4, B0 (ảnh Kaggle 10/2026); khác thì chỉ cảnh báo

IS_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("cores_work")
SRC_DIR = WORK_DIR / "nh_src"            # bản clone fork (Phần 2 chạy pytest trong thư mục này)
OUT_DIR = WORK_DIR / "cores_output"      # bảng, hình, tóm tắt, môi trường
WHEEL_ROOTS = [Path("/kaggle/input"), WORK_DIR / "wheels"]
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def installed_versions(libs):
    # Phiên bản đang cài trên đĩa (không phụ thuộc thứ đã nạp vào bộ nhớ).
    out = {}
    for lib in libs:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            out[lib] = None
    return out


def mem_gb(key):
    # Đọc /proc/meminfo (kB) => GB.
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1024 ** 2
    return None


def show(title, df):
    display(HTML(f"<h4 style='margin:12px 0 4px'>{title}</h4>"))
    display(df.style.hide(axis="index"))


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def mamba_install():
    # Chép từ 01_0_MambaSSM_Build Phần 9: tìm build_info.json, kiểm Python/torch/CUDA khớp lúc dựng, sha256, cài --no-deps,
    # chạy thử một khối Mamba xuôi + ngược.
    import torch
    infos = [p for root in WHEEL_ROOTS if root.exists() for p in root.rglob("build_info.json")]
    if not infos:
        raise FileNotFoundError("Không thấy build_info.json: gắn Output của notebook 01_0_MambaSSM_Build làm Input")
    if len(infos) > 1:
        print("Thấy nhiều build_info.json, dùng tệp đầu tiên:", *infos, sep="\n  ")
    info_path = infos[0]
    info = json.loads(info_path.read_text())
    now = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    diff = {k: (info[k], v) for k, v in now.items() if info[k] != v}
    if diff:
        raise RuntimeError(f"Môi trường khác lúc dựng wheel (lúc dựng, bây giờ): {diff}. Phải dựng lại ở 01_0_MambaSSM_Build")
    gpu_cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    if gpu_cc != (7, 5):
        raise RuntimeError(f"Wheel chỉ có mã cho {info['arch']} (T4), GPU hiện tại: {gpu_cc}")
    wheels = []
    for name, digest in info["wheels"].items():
        whl = info_path.parent / name
        if hashlib.sha256(whl.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"sha256 của {name} không khớp build_info.json")
        wheels.append(str(whl))
    torch_dist = md.version("torch")  # bản đã cài trên đĩa (torch.__version__ không đổi khi pip cài lại)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *wheels])
    if md.version("torch") != torch_dist:
        raise RuntimeError("torch bị đổi sau khi cài wheel, khởi động lại phiên và kiểm lại lệnh cài")
    from mamba_ssm import Mamba
    m = Mamba(d_model=16).cuda()
    y = m(torch.randn(2, 8, 16, device="cuda", requires_grad=True))
    y.sum().backward()
    if not torch.isfinite(y).all():
        raise RuntimeError("Khối Mamba chạy thử ra NaN/inf")
    print(f"mamba-ssm đã cài từ {info_path.parent} (dựng {info['built_at']}, torch {info['torch']}, {info['arch']}); "
          f"chạy thử xuôi + ngược đạt")
    return info, info_path


# Clone đúng commit rồi cài từ bản clone (không editable: thư viện nằm trong site-packages như các notebook trước)
before = installed_versions(CHECK_LIBS)
if SRC_DIR.exists():
    shutil.rmtree(SRC_DIR)
run(["git", "clone", "-q", "--branch", "core-models", NH_REPO, str(SRC_DIR)])
run(["git", "checkout", "-q", NH_COMMIT], cwd=SRC_DIR)
src_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SRC_DIR, capture_output=True, text=True,
                            check=True).stdout.strip()
run([sys.executable, "-m", "pip", "install", "-q", str(SRC_DIR)])
if before["pytest"] is None:
    run([sys.executable, "-m", "pip", "install", "-q", "pytest"])
after = installed_versions(CHECK_LIBS)
importlib.invalidate_caches()  # để tiến trình đang chạy thấy gói vừa cài
changed = [k for k in CHECK_LIBS if before[k] != after[k] and k != "pytest"]

MAMBA_BUILD_INFO, MAMBA_INFO_PATH = mamba_install()
mamba_versions = installed_versions(["mamba_ssm", "causal_conv1d"])

show("Thư viện (sau khi cài fork và mamba-ssm)", pd.DataFrame({
    "Thư viện": ["neuralhydrology", "mamba_ssm", "causal_conv1d"] + CHECK_LIBS,
    "Phiên bản": [f"{md.version('neuralhydrology')} (commit {src_commit[:7]})", mamba_versions["mamba_ssm"] or "THIẾU",
                  mamba_versions["causal_conv1d"] or "THIẾU"] + [after[k] or "THIẾU" for k in CHECK_LIBS],
    "pip đổi?": ["", "", ""] + [f"từ {before[k]}" if before[k] != after[k] else "" for k in CHECK_LIBS]}))

import torch  # nạp sau khi cài để in đúng thiết bị

n_gpu = torch.cuda.device_count()
gpus = [f"{i}: {torch.cuda.get_device_name(i)} (cc {'.'.join(map(str, torch.cuda.get_device_capability(i)))}, "
        f"{torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB)" for i in range(n_gpu)]
build_date = Path("/etc/build_date").read_text().strip() if Path("/etc/build_date").exists() else "không có"
ENV = {
    "Python": platform.python_version(),
    "torch / CUDA của torch": f"{torch.__version__} / {torch.version.cuda}",
    "cuDNN": torch.backends.cudnn.version(),
    "Số GPU": n_gpu,
    "GPU": "; ".join(gpus) or "không có GPU",
    "Số CPU": os.cpu_count(),
    "RAM tổng / còn trống (GB)": f"{mem_gb('MemTotal'):.1f} / {mem_gb('MemAvailable'):.1f}",
    "Đĩa trống ở WORK_DIR (GB)": f"{shutil.disk_usage(WORK_DIR).free / 1024 ** 3:.1f}",
    "Ngày dựng ảnh Kaggle": build_date,
    "Wheel mamba-ssm": str(MAMBA_INFO_PATH.parent),
}
show("Môi trường", pd.DataFrame({"Mục": list(ENV), "Giá trị": [str(v) for v in ENV.values()]}))

# Thử nạp các module sẽ dùng; sf_cores phải thấy mamba_ssm (MambaSSM khác None) thì lõi mamba mới dựng được
import_ok = {}
for mod in NH_MODULES:
    try:
        importlib.import_module(mod)
        import_ok[mod] = "ĐẠT"
    except Exception as e:  # in lỗi thật để biết thiếu gì
        import_ok[mod] = f"LỖI: {type(e).__name__}: {e}"

from neuralhydrology.modelzoo import sf_cores
from neuralhydrology.utils.config import Config

# Thư viện được nạp từ site-packages (bản vừa cài), không phải thư mục clone
nh_path = Path(importlib.import_module("neuralhydrology").__file__).resolve()
checks = {
    "Clone đúng commit fed53dc": src_commit == NH_COMMIT,
    "Nạp neuralhydrology từ site-packages": "site-packages" in str(nh_path) or "dist-packages" in str(nh_path),
    "pip không đổi thư viện có sẵn": not changed,
    "Có GPU T4 (dùng GPU 0)": n_gpu >= 1 and "T4" in gpus[0],
    "Có tensorboard, pytest": after["tensorboard"] is not None and after["pytest"] is not None,
    **{f"Nạp {m.split('.')[-1]}": v == "ĐẠT" for m, v in import_ok.items()},
    "sf_cores thấy mamba_ssm": sf_cores.MambaSSM is not None,
    "Config có khóa sf_core, data_on_gpu": all(isinstance(getattr(Config, k, None), property)
                                               for k in ["sf_core", "data_on_gpu"]),
}
show("Điều kiện", pd.DataFrame({"Điều kiện": list(checks), "Đạt": ["ĐẠT" if v else "KHÔNG" for v in checks.values()]}))
for mod, v in import_ok.items():
    if v != "ĐẠT":
        print(f"{mod}: {v}")
print(f"neuralhydrology nạp từ: {nh_path.parent}")

if torch.__version__ != TORCH_B0:
    print(f"Cảnh báo: torch {torch.__version__} khác bản của A4, B0 ({TORCH_B0}) - ảnh Kaggle đã đổi; ghi lại khi so sánh.")
if changed:
    print("Cảnh báo: pip đã đổi thư viện có sẵn:", ", ".join(f"{k} {before[k]} => {after[k]}" for k in changed))
if not all(v for k, v in checks.items() if k != "pip không đổi thư viện có sẵn"):
    raise RuntimeError("Phần 1 CHƯA đạt (dòng KHÔNG ở bảng Điều kiện) - dừng notebook")
print("=> Đạt. Chạy Phần 2.")


# %% Phần 2 - Chạy toàn bộ kiểm thử của fork (pytest)
# Chạy test/ của bản clone (cwd = SRC_DIR vì cấu hình thử dùng đường dẫn ./test/...). Ở máy (pandas 3.1) 159/162 kiểm thử
# cũ đạt, 3 hỏng do pandas 3 bỏ tần suất 'A' (hỏng cả ở f229e02); Kaggle dùng pandas 2.3.3 nên phải đạt hết. Cấu hình thử
# của thư viện chạy trên CPU; phần GPU kiểm ở Phần 3-4. Khoảng 20-40 phút. Có kiểm thử hỏng => raise.
import time
import xml.etree.ElementTree as ET

N_SF_TESTS = 43                          # số kiểm thử của test/test_sequential_forecast.py ở fed53dc (pytest --collect-only)

JUNIT = OUT_DIR / "pytest_junit.xml"
t0 = time.time()
proc = subprocess.run([sys.executable, "-m", "pytest", "test", "-q", "-p", "no:cacheprovider", "-W", "ignore",
                       f"--junitxml={JUNIT}"], cwd=SRC_DIR, capture_output=True, text=True)
pytest_minutes = (time.time() - t0) / 60

# Gom kết quả theo tệp kiểm thử từ junit xml
rows, failed = {}, []
for case in ET.parse(JUNIT).getroot().iter("testcase"):
    test_file = case.get("classname", "").split(".")[-1] + ".py"
    row = rows.setdefault(test_file, {"Tệp kiểm thử": test_file, "Đạt": 0, "Hỏng": 0, "Bỏ qua": 0, "Giây": 0.0})
    row["Giây"] += float(case.get("time", 0))
    if case.find("failure") is not None or case.find("error") is not None:
        row["Hỏng"] += 1
        failed.append(f"{test_file}::{case.get('name')}")
    elif case.find("skipped") is not None:
        row["Bỏ qua"] += 1
    else:
        row["Đạt"] += 1
table = pd.DataFrame(rows.values()).sort_values("Tệp kiểm thử")
table["Giây"] = table["Giây"].round(1)
show(f"Kiểm thử của fork (commit {src_commit[:7]}), {pytest_minutes:.1f} phút", table)
print("Dòng tổng kết của pytest:", proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(trống)")
for name in failed:
    print("HỎNG:", name)

n_new = rows.get("test_sequential_forecast.py", {}).get("Đạt", 0)
checks = {
    "pytest kết thúc bình thường (mã thoát 0)": proc.returncode == 0,
    "Không kiểm thử nào hỏng": not failed and int(table["Hỏng"].sum()) == 0,
    f"Đủ {N_SF_TESTS} kiểm thử của mô hình chung đạt": n_new == N_SF_TESTS,
}
show("Điều kiện", pd.DataFrame({"Điều kiện": list(checks), "Đạt": ["ĐẠT" if v else "KHÔNG" for v in checks.values()]}))
if not all(checks.values()):
    print(proc.stdout[-5000:])
    print(proc.stderr[-2000:])
    raise RuntimeError("Phần 2 CHƯA đạt - xem các dòng HỎNG và log pytest ở trên")
print(f"=> Đạt: {int(table['Đạt'].sum())} kiểm thử đạt. Chạy Phần 3.")


# %% Phần 3 - Lõi LSTM của mô hình chung so với SequentialForecastLSTM bằng trọng số tác giả
# (1) Dựng thư mục Q có cột qmean m³/s như B0 Phần 3 (run SeqLSTM có Q của tác giả cần cột này). (2) Chép run tác giả
# thành 2 run: "ref" giữ mô hình sequential_forecast_lstm, "sf_lstm" đổi sang sequential_forecast (sf_core: lstm) và đổi
# tên tham số lstm. => core.rnn. trong best_model.pt; cả hai data_on_gpu: true. (3) Chấm kỳ test 451 lưu vực bằng eval_run
# (best_model.pt, GPU 0). Đạt khi dự báo hai run trùng từng bit và NSE trung vị lệch NSE trung vị của tác giả <= 0,005
# (như A2). Cần Phần 1 (WORK_DIR, show).
import pickle
import time

import numpy as np
import torch
from ruamel.yaml import YAML

from neuralhydrology.datasetzoo.lamah import load_lamah_forcing
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.modelzoo import get_model
from neuralhydrology.nh_run import eval_run
from neuralhydrology.utils.config import Config

SEARCH_ROOT = Path(os.environ.get("LAMAH_ROOT", "/kaggle/input" if IS_KAGGLE else "."))
AUTHOR_RUN_REL = ("Sequential_Forecast_LSTM/Sequential_Forecast_LSTM_with_q/"
                  "Sequential_Forecast_LSTM_RA_with_q_Simple_Embedding")
N_BASINS = 451
RUNS_DIR = WORK_DIR / "cores_runs"
Q_DIR = WORK_DIR / "LamaH_expanded_q_input"   # cùng tên thư mục tác giả dùng
SUB_A = "A_basins_total_upstrm"
TS_DAILY = Path("2_timeseries") / "daily"
DATE_COLS = ["YYYY", "MM", "DD"]
STAT_TOL = 1e-3                                # lệch tương đối tối đa của thống kê qmean so với tệp chuẩn hóa tác giả
PAPER_TOL = 0.005                              # như A2: NSE trung vị lệch số của tác giả không quá 0,005
COPY_IGNORE = shutil.ignore_patterns("test", "validation", "model_epoch*.pt", "optimizer_state*")
PREFIX_OLD, PREFIX_NEW = "lstm.", "core.rnn."  # tên tham số SequentialForecastLSTM => SequentialForecast
RUNS = {"ref": {}, "sf_lstm": {"model": "sequential_forecast", "sf_core": "lstm"}}


def find_dir(root, name):
    # Thư mục duy nhất tên `name` dưới `root`; báo lỗi rõ nếu không có hoặc có nhiều.
    hits = sorted(p for p in root.rglob(name) if p.is_dir())
    if len(hits) != 1:
        raise FileNotFoundError(f"Cần đúng 1 thư mục '{name}' dưới {root}, thấy {len(hits)}: {hits} - kiểm Input lamah-ce-ext")
    return hits[0]


def read_text_csv(path, usecols=None):
    # Như B0 Phần 3: đọc CSV ';' dạng chữ, ô trống giữ chuỗi rỗng; thêm khóa ngày dạng số để ghép.
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, usecols=usecols)
    df["_key"] = pd.to_datetime(df[DATE_COLS].astype(int).rename(columns={"YYYY": "year", "MM": "month", "DD": "day"}))
    return df


def clean_qmean(ser):
    # Số âm (-999 = thiếu) => ô trống, giống load_lamah_discharge (lamah.py dòng 228-229).
    return ser.where(~(pd.to_numeric(ser, errors="coerce") < 0), "")


def build_basin_file(basin):
    # Như B0 Phần 3: ghép qmean của tệp trạm vào tệp khí tượng theo ngày, ghi qua tệp tạm rồi đổi tên.
    forcing = read_text_csv(LAMAH_DIR / SUB_A / TS_DAILY / f"ID_{basin}.csv")
    gauge = read_text_csv(LAMAH_DIR / "D_gauges" / TS_DAILY / f"ID_{basin}.csv", usecols=DATE_COLS + ["qmean"])
    merged = forcing.merge(gauge[["_key", "qmean"]], on="_key", how="left").drop(columns="_key")
    merged["qmean"] = clean_qmean(merged["qmean"].fillna(""))
    dst = Q_DIR / SUB_A / TS_DAILY / f"ID_{basin}.csv"
    tmp = dst.with_suffix(".tmp")
    merged.to_csv(tmp, sep=";", index=False)
    tmp.rename(dst)


def make_run(name, model_update):
    # Chép run tác giả sang RUNS_DIR/name, sửa đường dẫn (như A2 Phần 5), bật data_on_gpu, cập nhật khóa mô hình
    # (rỗng => giữ mô hình tác giả); có đổi mô hình thì đổi tên tham số lõi trong best_model.pt.
    dst = RUNS_DIR / name
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(AUTHOR_RUN, dst, ignore=COPY_IGNORE)
    yaml_rt = YAML()
    yaml_rt.width = 4096
    cfg_raw = yaml_rt.load((AUTHOR_RUN / "config.yml").read_text(encoding="utf-8"))
    workers = cfg_raw.get("num_workers") or 0
    cfg_raw.update({"data_dir": str(Q_DIR), "run_dir": str(dst), "train_dir": str(dst / "train_data"),
                    "train_basin_file": str(BASIN_FILE), "validation_basin_file": str(BASIN_FILE),
                    "test_basin_file": str(BASIN_FILE), "device": DEVICE,
                    "num_workers": min(workers, os.cpu_count() or 1), "data_on_gpu": True, **model_update})
    if "img_log_dir" in cfg_raw:
        cfg_raw["img_log_dir"] = str(dst / "img_log")
    with open(dst / "config.yml", "w", encoding="utf-8") as f:
        yaml_rt.dump(cfg_raw, f)
    if model_update:
        state = torch.load(AUTHOR_RUN / "best_model.pt", map_location="cpu", weights_only=True)
        renamed = {(PREFIX_NEW + k[len(PREFIX_OLD):] if k.startswith(PREFIX_OLD) else k): v for k, v in state.items()}
        torch.save(renamed, dst / "best_model.pt")
    return dst


def load_results(run_dir):
    # test_results.p do chính eval_run ở trên ghi trong phiên này (tester.py), không phải tệp từ nguồn ngoài
    with open(run_dir / "test" / "best_model" / "test_results.p", "rb") as f:
        return pickle.load(f)


# (1) Dữ liệu, run tác giả, thư mục Q qmean m³/s (như B0 Phần 3)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LAMAH_DIR = find_dir(SEARCH_ROOT, "LamaH_extended")
EXP_DIR = find_dir(SEARCH_ROOT, "Experiments")
BASIN_FILE = EXP_DIR / "basins_filtered.txt"
basins = BASIN_FILE.read_text().split()
AUTHOR_RUN = EXP_DIR / AUTHOR_RUN_REL
author_metrics = pd.read_csv(AUTHOR_RUN / "test" / "best_model" / "test_metrics.csv", dtype={"basin": str})
AUTHOR_NSE = float(author_metrics["NSE"].median())
if len(set(basins)) != N_BASINS or not (AUTHOR_RUN / "best_model.pt").exists():
    raise RuntimeError("Thiếu best_model.pt của run tác giả hoặc danh sách lưu vực không đủ 451 mã")

t0 = time.time()
(Q_DIR / SUB_A / TS_DAILY).mkdir(parents=True, exist_ok=True)
for link, target in [(Q_DIR / SUB_A / "1_attributes", LAMAH_DIR / SUB_A / "1_attributes"),
                     (Q_DIR / "D_gauges", LAMAH_DIR / "D_gauges")]:
    if not (link.exists() or link.is_symlink()):
        link.symlink_to(target, target_is_directory=True)
todo = [b for b in basins if not (Q_DIR / SUB_A / TS_DAILY / f"ID_{b}.csv").exists()]
for b in todo:
    build_basin_file(b)
n_files = len(list((Q_DIR / SUB_A / TS_DAILY).glob("ID_*.csv")))
# Kiểm đầu cuối như B0: thống kê qmean đọc bằng hàm của thư viện khớp tệp chuẩn hóa của tác giả
cfg_author = Config(AUTHOR_RUN / "config.yml")
scaler = load_scaler(AUTHOR_RUN)
start = cfg_author.train_start_date - pd.Timedelta(days=cfg_author.seq_length - 1)
arr = np.concatenate([load_lamah_forcing(Q_DIR, b, sub_dataset="lamah_a")["qmean"].loc[start:cfg_author.train_end_date]
                      .to_numpy() for b in basins])
ref_mean, ref_std = float(scaler["xarray_feature_center"]["qmean"]), float(scaler["xarray_feature_scale"]["qmean"])
stat_diff = max(abs(np.nanmean(arr) - ref_mean) / ref_mean, abs(np.nanstd(arr) - ref_std) / ref_std)
build_minutes = (time.time() - t0) / 60

# (2) Hai run: mô hình tác giả và mô hình chung lõi LSTM, nạp best_model.pt strict (thiếu hay thừa tham số => lỗi)
run_rows = []
for name, update in RUNS.items():
    rd = make_run(name, update)
    cfg = Config(rd / "config.yml")
    model = get_model(cfg)
    state = torch.load(rd / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    core_keys = [k for k in state if k.startswith((PREFIX_OLD, PREFIX_NEW))]
    run_rows.append({"run": name, "model": cfg.model, "sf_core": cfg.as_dict().get("sf_core", "-"),
                     "lớp mô hình": type(model).__name__, "số tham số": sum(p.numel() for p in model.parameters()),
                     "tham số lõi": f"{len(core_keys)}: {core_keys[0]}, ..."})
show("Hai run dùng best_model.pt của tác giả", pd.DataFrame(run_rows))

# (3) Chấm kỳ test bằng eval_run (best_model.pt), đo thời gian
eval_rows, results = [], {}
for name in RUNS:
    t1 = time.time()
    eval_run(run_dir=RUNS_DIR / name, period="test", epoch="best", gpu=0 if DEVICE != "cpu" else -1)
    metrics = pd.read_csv(RUNS_DIR / name / "test" / "best_model" / "test_metrics.csv", dtype={"basin": str})
    results[name] = load_results(RUNS_DIR / name)
    eval_rows.append({"run": name, "phút": round((time.time() - t1) / 60, 1),
                      "lưu vực có NSE": int(metrics["NSE"].notna().sum()), "NSE trung vị": float(metrics["NSE"].median()),
                      "NSE trung vị tác giả": AUTHOR_NSE, "chênh": float(metrics["NSE"].median()) - AUTHOR_NSE})
eval_df = pd.DataFrame(eval_rows)
show(f"Chấm kỳ test {N_BASINS} lưu vực (best_model.pt, {DEVICE}, data_on_gpu)",
     eval_df.round({"NSE trung vị": 6, "NSE trung vị tác giả": 6, "chênh": 6}))

# So dự báo từng ngày, từng lưu vực giữa hai run (NaN coi như bằng NaN)
same_basins = set(results["ref"]) == set(results["sf_lstm"])
n_equal, max_diff = 0, 0.0
for b in results["ref"]:
    a = results["ref"][b]["1D"]["xr"]["qmax_sim"].values
    c = results["sf_lstm"][b]["1D"]["xr"]["qmax_sim"].values
    n_equal += int(np.array_equal(a, c, equal_nan=True))
    if a.shape == c.shape and np.isfinite(a - c).any():
        max_diff = max(max_diff, float(np.nanmax(np.abs(a - c))))
show("So dự báo qmax_sim của hai run", pd.DataFrame([{
    "lưu vực": len(results["ref"]), "lưu vực trùng từng bit": n_equal, "|Δ| lớn nhất (mm/ngày)": max_diff}]))

checks = {
    f"Thư mục Q đủ {N_BASINS} tệp ({build_minutes:.1f} phút)": n_files == N_BASINS,
    f"Thống kê qmean khớp tệp chuẩn hóa tác giả (lệch {stat_diff:.1e})": stat_diff < STAT_TOL,
    "Cùng số tham số": run_rows[0]["số tham số"] == run_rows[1]["số tham số"],
    "Cùng danh sách lưu vực": same_basins,
    "Dự báo trùng từng bit ở mọi lưu vực": same_basins and n_equal == len(results["ref"]),
    f"NSE trung vị lệch tác giả <= {PAPER_TOL}": bool((eval_df["chênh"].abs() <= PAPER_TOL).all()),
}
show("Điều kiện", pd.DataFrame({"Điều kiện": list(checks), "Đạt": ["ĐẠT" if v else "KHÔNG" for v in checks.values()]}))
if not all(checks.values()):
    raise RuntimeError("Phần 3 CHƯA đạt (dòng KHÔNG ở bảng Điều kiện) - dừng notebook")
print("=> Đạt. Chạy Phần 4.")


# %% Phần 4 - Lõi Mamba: không nhìn tương lai, đúng trục, công thức số tham số, mambapy
# Lõi mamba dựng bằng get_core trên config của run sf_lstm (Phần 3) đổi sf_core: mamba, 2 lớp, hidden_size 64,
# sf_mamba_d_inner 130 (expand khoảng 2.03, kiểm luôn núm chỉnh tinh số tham số), chế độ eval.
# (1) Không nhìn tương lai: đổi mạnh đầu vào ở bước k => đầu ra trước k không đổi, đầu ra tại k phải đổi (kiểm thử 2 ở máy
# chỉ chạy được 4 lõi CPU). (2) Đúng trục: hoán vị mẫu trong lô => đầu ra hoán vị theo, và mỗi mẫu chạy riêng ra như chạy
# cả lô (bản Mamba có sẵn của thư viện quét theo trục lô nên sẽ hỏng phép này). (3) mamba_block_parameters (đếm số tham số
# trên CPU cho match_parameters) khớp khối mamba_ssm.Mamba thật. (4) mambapy 1.2.0 (PyTorch thuần, cho demo CPU) nạp
# trọng số từng khối strict, cùng đầu ra trên GPU và CPU. Cần Phần 1, 3 (RUNS_DIR, show, torch).
import copy

from neuralhydrology.modelzoo import sf_cores
from neuralhydrology.modelzoo.sequential_forecast import SequentialForecast

MAMBAPY_VERSION = "1.2.0"
MAMBA_UPDATE = {"sf_core": "mamba", "sf_layers": 2, "hidden_size": 64, "sf_mamba_d_inner": 130, "sf_dropout": 0.0}
N_BATCH, SEQ, INPUT_SIZE = 8, 365, 32          # như B1b: 365 bước, nhúng 16 + 16 = 32
LEAK_STEPS = [0, 100, 363, 364]                # bước bị đổi; 363 = ngày cuối quá khứ, 364 = ngày t
LEAK_TOL, CHANGE_MIN = 1e-6, 1e-3
AXIS_TOL = 1e-5
MAMBAPY_TOL = 1e-4                             # sai số tương đối lớn nhất
N_CPU_MAMBAPY = 4                              # số mẫu so trên CPU (pscan của mambapy tốn bộ nhớ)
# (d_model, d_state, d_conv, d_inner); d_inner khác 2 * d_model => expand lẻ qua sf_mamba_d_inner
PARAM_CHECK = [(16, 16, 4, 32), (64, 16, 4, 128), (96, 32, 4, 192), (128, 16, 3, 128), (49, 16, 4, 96), (38, 16, 4, 77)]


def rel_err(a, b):
    return float((a - b).abs().max() / b.abs().max())


cfg_mamba = Config(RUNS_DIR / "sf_lstm" / "config.yml")
cfg_mamba.update_config(MAMBA_UPDATE)
torch.manual_seed(0)
core = sf_cores.get_core(cfg_mamba, input_size=INPUT_SIZE).to(DEVICE).eval()
g = torch.Generator(device="cpu").manual_seed(1)
x = torch.randn(N_BATCH, SEQ, INPUT_SIZE, generator=g).to(DEVICE)

# (1) Không nhìn tương lai
leak_rows = []
with torch.no_grad():
    out = core(x)
    for k in LEAK_STEPS:
        x_k = x.clone()
        x_k[:, k] += 10 * torch.randn(N_BATCH, INPUT_SIZE, generator=g).to(DEVICE)
        out_k = core(x_k)
        before = (out[:, :k] - out_k[:, :k]).abs().max().item() if k > 0 else 0.0
        at_k = (out[:, k] - out_k[:, k]).abs().max().item()
        leak_rows.append({"bước đổi k": k, "|Δ| lớn nhất trước k": before, "|Δ| lớn nhất tại k": at_k,
                          "Đạt": "ĐẠT" if before <= LEAK_TOL and at_k > CHANGE_MIN else "KHÔNG"})
show(f"(1) Không nhìn tương lai ({N_BATCH} mẫu x {SEQ} bước; đạt khi trước k <= {LEAK_TOL:g}, tại k > {CHANGE_MIN:g})",
     pd.DataFrame(leak_rows))

# (2) Đúng trục: hoán vị mẫu, chạy từng mẫu riêng
with torch.no_grad():
    perm = torch.randperm(N_BATCH, generator=g).to(DEVICE)
    perm_diff = (core(x[perm]) - out[perm]).abs().max().item()
    single_diff = max((core(x[i:i + 1]) - out[i:i + 1]).abs().max().item() for i in range(N_BATCH))
axis_rows = [{"phép thử": "hoán vị mẫu trong lô => đầu ra hoán vị theo", "|Δ| lớn nhất": perm_diff},
             {"phép thử": "từng mẫu chạy riêng = chạy cả lô", "|Δ| lớn nhất": single_diff}]
for r in axis_rows:
    r["Đạt"] = "ĐẠT" if r["|Δ| lớn nhất"] <= AXIS_TOL else "KHÔNG"
show(f"(2) Quét đúng trục thời gian (đạt khi <= {AXIS_TOL:g})", pd.DataFrame(axis_rows))

# (3) Công thức số tham số so với khối thật
param_rows = []
for d, n, c, d_inner in PARAM_CHECK:
    cfg_p = copy.deepcopy(cfg_mamba)
    cfg_p.update_config({"sf_mamba_d_inner": d_inner})
    e = sf_cores.mamba_expand(cfg_p, d)
    block = sf_cores.MambaSSM(d_model=d, d_state=n, d_conv=c, expand=e)
    real, formula = sf_cores.count_parameters(block), sf_cores.mamba_block_parameters(d, n, c, e)
    param_rows.append({"d_model": d, "d_state": n, "d_conv": c, "d_inner": f"{block.d_inner} (expand {e:.4f})",
                       "khối thật": real, "công thức": formula,
                       "Đạt": "ĐẠT" if real == formula and block.d_inner == d_inner else "KHÔNG"})
# Cả mô hình: đếm thật (có mamba_ssm) so với cách đếm khi không có mamba_ssm (công thức)
model_real = sf_cores.count_parameters(SequentialForecast(cfg_mamba))
saved, sf_cores.MambaSSM = sf_cores.MambaSSM, None
try:
    model_formula = sf_cores.count_model_parameters(cfg_mamba)
finally:
    sf_cores.MambaSSM = saved
param_rows.append({"d_model": f"mô hình {MAMBA_UPDATE['hidden_size']}", "d_state": cfg_mamba.sf_mamba_d_state,
                   "d_conv": cfg_mamba.sf_mamba_d_conv, "d_inner": cfg_mamba.sf_mamba_d_inner, "khối thật": model_real,
                   "công thức": model_formula, "Đạt": "ĐẠT" if model_real == model_formula else "KHÔNG"})
show("(3) Số tham số: mamba_ssm.Mamba thật so với mamba_block_parameters", pd.DataFrame(param_rows))

# (4) mambapy: thay từng khối mamba_ssm bằng MambaBlock của mambapy, nạp strict
if installed_versions(["mambapy"])["mambapy"] != MAMBAPY_VERSION:
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", f"mambapy=={MAMBAPY_VERSION}"])
from mambapy.mamba import MambaBlock, MambaConfig


def to_mambapy(core_ssm, device):
    # Bản sao của lõi với mỗi khối mamba_ssm.Mamba thay bằng mambapy MambaBlock cùng trọng số (strict); d_inner đặt
    # thẳng vì mambapy tính d_inner = expand_factor * d_model không làm tròn (sf_cores.mamba_expand).
    core_py = copy.deepcopy(core_ssm).to(device)
    for i, block in enumerate(core_ssm.blocks):
        mp_cfg = MambaConfig(d_model=block.d_model, n_layers=1, d_state=block.d_state, d_conv=block.d_conv)
        mp_cfg.d_inner = block.d_inner
        mp_block = MambaBlock(mp_cfg).to(device)
        mp_block.load_state_dict({k: v.to(device) for k, v in block.state_dict().items()}, strict=True)
        core_py.blocks[i] = mp_block
    return core_py.eval()


with torch.no_grad():
    t1 = time.time()
    out_py_gpu = to_mambapy(core, DEVICE)(x)
    core_cpu = to_mambapy(core, "cpu")
    t2 = time.time()
    out_py_cpu = core_cpu(x[:N_CPU_MAMBAPY].cpu())
    cpu_seconds = time.time() - t2
mp_rows = [{"so sánh": f"mambapy GPU vs mamba_ssm GPU ({N_BATCH} mẫu)", "sai số tương đối": rel_err(out_py_gpu, out)},
           {"so sánh": f"mambapy CPU vs mamba_ssm GPU ({N_CPU_MAMBAPY} mẫu)",
            "sai số tương đối": rel_err(out_py_cpu, out[:N_CPU_MAMBAPY].cpu())}]
for r in mp_rows:
    r["Đạt"] = "ĐẠT" if r["sai số tương đối"] < MAMBAPY_TOL else "KHÔNG"
show(f"(4) mambapy {MAMBAPY_VERSION} dùng trọng số lõi mamba (đạt khi < {MAMBAPY_TOL:g})", pd.DataFrame(mp_rows))
print(f"mambapy trên CPU ({os.cpu_count()} lõi): {N_CPU_MAMBAPY} mẫu x {SEQ} bước x {MAMBA_UPDATE['sf_layers']} lớp, "
      f"chỉ xuôi: {cpu_seconds:.2f} giây")

all_rows = leak_rows + axis_rows + param_rows + mp_rows
if any(r["Đạt"] != "ĐẠT" for r in all_rows):
    raise RuntimeError("Phần 4 CHƯA đạt (dòng KHÔNG ở các bảng trên) - dừng notebook")
print("=> Đạt. Chạy Phần 5.")


# %% Phần 5 - Bảng số chiều cùng tham số của 5 lõi
# Mốc = số tham số của LSTM tác giả trên cấu hình này (84.785; mốc chính thức ở bước C là LSTM của B*). Với mỗi lõi và số
# lớp, match_parameters dò hidden_size (rồi kích thước trong: sf_transformer_dim_feedforward của Transformer,
# sf_mamba_d_inner của Mamba) cho lệch nhỏ nhất, đếm bằng mô hình
# thật (Mamba dựng khối mamba_ssm thật). Cấu hình không khớp <= 1% ghi "không khớp" (C1 không dùng), chỉ dừng khi cấu hình
# chính của một lõi không khớp. Lưu bảng cho Phần 6. Cần Phần 1, 3 (RUNS_DIR, OUT_DIR, show).
PARAM_TOL = 0.01
LAYERS = {"lstm": [1, 2], "gru": [1, 2], "transformer": [1, 2, 4], "s4d": [2, 4, 6], "mamba": [2, 4, 6]}
MAIN_LAYERS = {"lstm": 1, "gru": 1, "transformer": 2, "s4d": 2, "mamba": 2}   # cấu hình chính dùng ở Phần 6

cfg_base = Config(RUNS_DIR / "sf_lstm" / "config.yml")
TARGET = sf_cores.count_model_parameters(cfg_base)   # lõi LSTM 1 lớp, hidden_size tác giả
match_rows = []
t0 = time.time()
for core_name, layer_list in LAYERS.items():
    for n_layers in layer_list:
        cfg_c = copy.deepcopy(cfg_base)
        cfg_c.update_config({"sf_core": core_name, "sf_layers": n_layers})
        row = {"lõi": core_name, "số lớp": n_layers, "chính": "x" if MAIN_LAYERS[core_name] == n_layers else ""}
        try:
            update = sf_cores.match_parameters(cfg_c, TARGET, tolerance=PARAM_TOL)
            n_params = sf_cores.count_model_parameters(cfg_c, update)
            inner = update.get("sf_transformer_dim_feedforward", update.get("sf_mamba_d_inner", "-"))
            row.update({"hidden_size": update["hidden_size"], "kích thước trong": inner,
                        "số tham số": n_params, "lệch %": 100 * (n_params - TARGET) / TARGET, "khớp": "ĐẠT"})
        except ValueError as e:
            row.update({"hidden_size": "-", "kích thước trong": "-", "số tham số": "-", "lệch %": float("nan"),
                        "khớp": "không khớp: " + str(e).split("(closest: ")[-1].rstrip(").")})
        match_rows.append(row)
match_df = pd.DataFrame(match_rows)
show(f"Cùng tham số với LSTM tác giả ({TARGET:,} tham số, lệch <= {PARAM_TOL:.0%}), dò trong {time.time() - t0:.0f} giây",
     match_df.round({"lệch %": 3}))
match_df.to_csv(OUT_DIR / "core_param_match.csv", index=False)
print("Thuộc tính khác của lõi giữ mặc định:", {k: getattr(cfg_base, k) for k in
      ["sf_transformer_nheads", "sf_s4d_d_state", "sf_mamba_d_state", "sf_mamba_d_conv", "sf_mamba_expand"]})

main_ok = all(r["khớp"] == "ĐẠT" for r in match_rows if r["chính"] == "x")
if not main_ok:
    raise RuntimeError("Phần 5 CHƯA đạt: cấu hình chính của một lõi không khớp số tham số <= 1%")
print("=> Đạt (cấu hình chính của cả 5 lõi khớp). Chạy Phần 6.")


# %% Phần 6 - Huấn luyện ngắn 5 lõi trên dữ liệu thật: tốc độ, bộ nhớ, lặp lại từng bit
# Dựng dataset train 451 lưu vực một lần (config run sf_lstm của Phần 3, data_on_gpu), rồi với mỗi lõi ở cấu hình chính
# cùng tham số (Phần 5) chạy N_STEPS bước huấn luyện đúng như vòng _train_epoch của thư viện (basetrainer.py: đặt hạt
# giống như _set_random_seeds, dựng mô hình, get_optimizer, loss NSE, cắt gradient, cùng thứ tự lô) - không chạy
# validation. Mỗi lõi chạy 2 lần cùng hạt giống để biết có lặp lại từng bit trên GPU không (Mamba: đạo hàm không tất định,
# library.md). Thêm SequentialForecastLSTM của thư viện: phải trùng từng bit lõi LSTM (Phần 3 mới so dự báo, đây so
# huấn luyện trên GPU). Đo ms/bước huấn luyện, ms/lô dự báo, bộ nhớ GPU đỉnh. Cần Phần 1, 3, 5.
import random

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.gpu_loader import GpuBatchLoader
from neuralhydrology.training import get_loss_obj, get_optimizer, get_regularization_obj

N_STEPS, N_WARMUP, N_INFER = 200, 20, 50     # bước huấn luyện, bước bỏ khi đo thời gian (khởi động nhân), lô dự báo
SEED = 111
B0_STEPS_PER_EPOCH = 4275                    # số bước một epoch ở B0 (B1a Phần 4) - quy ra phút/epoch

cfg_train = copy.deepcopy(cfg_base)
cfg_train.update_config({"train_dir": str(WORK_DIR / "cores_train_data")})
cfg_train.train_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()
ds_train = get_dataset(cfg=cfg_train, is_train=True, period="train", scaler={})
loader = GpuBatchLoader(ds_train, batch_size=cfg_train.batch_size, device=DEVICE, shuffle=True)
dataset_minutes = (time.time() - t0) / 60
print(f"Dataset train: {len(ds_train):,} mẫu, {len(loader):,} lô {cfg_train.batch_size}, dựng {dataset_minutes:.1f} phút")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_short(update):
    # Như BaseTrainer: hạt giống => mô hình => optimizer, loss => lặp loader; trả trọng số, thời gian, bộ nhớ, loss.
    cfg = copy.deepcopy(cfg_train)
    cfg.update_config(update)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    model = get_model(cfg).to(DEVICE)
    optimizer = get_optimizer(model, cfg)
    loss_obj = get_loss_obj(cfg).to(DEVICE)
    loss_obj.set_regularization_terms(get_regularization_obj(cfg))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train()
    losses, t_start = [], None
    for i, data in enumerate(loader):
        if i >= N_STEPS:
            break
        if i == N_WARMUP:
            sync()
            t_start = time.time()
        data = model.pre_model_hook(data, is_train=True)
        loss, _ = loss_obj(model(data), data)
        optimizer.zero_grad()
        loss.backward()
        if cfg.clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_gradient_norm)
        optimizer.step()
        losses.append(loss.item())
    sync()
    if len(losses) < N_STEPS:
        raise RuntimeError(f"Loader chỉ có {len(losses)} lô, cần {N_STEPS}")
    ms_step = (time.time() - t_start) / (N_STEPS - N_WARMUP) * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else float("nan")
    # Dự báo: chế độ eval, không đạo hàm
    model.eval()
    with torch.no_grad():
        batches = iter(loader)
        for _ in range(5):
            model(next(batches))
        sync()
        t1 = time.time()
        for _ in range(N_INFER):
            model(next(batches))
        sync()
    ms_infer = (time.time() - t1) / N_INFER * 1000
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"state": state, "ms_step": ms_step, "ms_infer": ms_infer, "peak_gb": peak_gb, "losses": losses,
            "n_params": sf_cores.count_parameters(model)}


def max_state_diff(a, b, rename=lambda k: k):
    return max(float((a[k].float() - b[rename(k)].float()).abs().max()) for k in a)


# Cấu hình chính cùng tham số của từng lõi (như Phần 5); S4D dùng lr riêng = learning_rate / 10 cho tham số SSM (S4D-FT)
runs = {"SeqLSTM thư viện": {"model": "sequential_forecast_lstm"}}
for core_name, n_layers in MAIN_LAYERS.items():
    cfg_c = copy.deepcopy(cfg_base)
    update = {"model": "sequential_forecast", "sf_core": core_name, "sf_layers": n_layers}
    if core_name == "s4d":
        update["sf_s4d_lr"] = cfg_base.learning_rate / 10
    cfg_c.update_config(update)
    runs[core_name] = {**update, **sf_cores.match_parameters(cfg_c, TARGET, tolerance=PARAM_TOL)}

results6, rows6 = {}, []
for name, update in runs.items():
    first = train_short(update)
    second = train_short(update) if name != "SeqLSTM thư viện" else None
    results6[name] = first
    loss = np.array(first["losses"])
    rows6.append({
        "lõi": name, "số lớp": update.get("sf_layers", 1), "hidden_size": update.get("hidden_size", cfg_base.hidden_size),
        "kích thước trong": update.get("sf_transformer_dim_feedforward", update.get("sf_mamba_d_inner", "-")),
        "số tham số": first["n_params"], "ms/bước huấn luyện": first["ms_step"],
        "phút/epoch (ước tính)": first["ms_step"] * B0_STEPS_PER_EPOCH / 60000, "ms/lô dự báo": first["ms_infer"],
        "GPU đỉnh (GB)": first["peak_gb"], "loss 20 bước đầu": loss[:20].mean(), "loss 20 bước cuối": loss[-20:].mean(),
        "lặp lại trùng bit": "-" if second is None else ("có" if max_state_diff(first["state"], second["state"]) == 0
                                                          else f"không (lệch {max_state_diff(first['state'], second['state']):.1e})")})
table6 = pd.DataFrame(rows6)
show(f"Huấn luyện ngắn {N_STEPS} bước (lô {cfg_base.batch_size}, hạt giống {SEED}, {DEVICE}); thời gian đo từ bước {N_WARMUP}",
     table6.round({"ms/bước huấn luyện": 1, "phút/epoch (ước tính)": 2, "ms/lô dự báo": 1, "GPU đỉnh (GB)": 2,
                   "loss 20 bước đầu": 4, "loss 20 bước cuối": 4}))
table6.to_csv(OUT_DIR / "core_short_training.csv", index=False)

# SequentialForecastLSTM của thư viện so với lõi LSTM sau N_STEPS bước trên GPU
ref_state, sf_state = results6["SeqLSTM thư viện"]["state"], results6["lstm"]["state"]
ref_vs_sf = max_state_diff(ref_state, sf_state, rename=lambda k: k.replace(PREFIX_OLD, PREFIX_NEW, 1))
checks = {
    f"Dataset train đủ mẫu như B0 ({len(ds_train):,})": len(loader) == B0_STEPS_PER_EPOCH,
    "Loss hữu hạn ở mọi bước, mọi lõi": all(np.isfinite(r["losses"]).all() for r in results6.values()),
    "Loss giảm (20 bước cuối < 20 bước đầu) ở mọi lõi": all(r["loss 20 bước cuối"] < r["loss 20 bước đầu"] for r in rows6),
    f"Lõi LSTM trùng từng bit SequentialForecastLSTM sau {N_STEPS} bước (lệch {ref_vs_sf:.1e})": ref_vs_sf == 0,
    "Số tham số lệch LSTM tác giả <= 1%": all(abs(r["số tham số"] - TARGET) <= PARAM_TOL * TARGET for r in rows6),
}
show("Điều kiện", pd.DataFrame({"Điều kiện": list(checks), "Đạt": ["ĐẠT" if v else "KHÔNG" for v in checks.values()]}))
if not all(checks.values()):
    raise RuntimeError("Phần 6 CHƯA đạt (dòng KHÔNG ở bảng Điều kiện) - dừng notebook")
print("=> Đạt. Chạy Phần 7.")


# %% Phần 7 - Tóm tắt, môi trường, danh mục tệp, dọn thư mục tạm
# Output chính thức (cores_output/): pytest_junit.xml (Phần 2), chỉ số test hai run của Phần 3, bảng Phần 4-6 (CSV),
# cores_summary.json (số liệu chính của mọi Phần), environment.json (Input, thư viện, commit), manifest.csv (sha256).
# Xóa thư mục dựng lại được: bản clone fork, thư mục Q (1,35 GB), dataset train lưu tạm, hai run của Phần 3 (đã chép
# test_metrics.csv). Cần Phần 1-6 trong phiên.
import hashlib

ENV_LIBS = ["neuralhydrology", "mamba_ssm", "causal_conv1d", "mambapy", "torch", "numpy", "pandas", "xarray", "scipy",
            "numba", "ruamel.yaml", "pytest"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def du_gb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file() and not f.is_symlink()) / 1e9


# (1) Bảng và tóm tắt
for name in RUNS:
    shutil.copy2(RUNS_DIR / name / "test" / "best_model" / "test_metrics.csv", OUT_DIR / f"phan3_{name}_test_metrics.csv")
pd.DataFrame(leak_rows).to_csv(OUT_DIR / "phan4_mamba_causal.csv", index=False)
pd.DataFrame(axis_rows + mp_rows).to_csv(OUT_DIR / "phan4_mamba_axis_mambapy.csv", index=False)
pd.DataFrame(param_rows).to_csv(OUT_DIR / "phan4_mamba_param_formula.csv", index=False)
SUMMARY = {
    "phan2_pytest": {"minutes": round(pytest_minutes, 1), "returncode": proc.returncode,
                     "passed": int(table["Đạt"].sum()), "failed": int(table["Hỏng"].sum()),
                     "sequential_forecast_tests": int(rows.get("test_sequential_forecast.py", {}).get("Đạt", 0))},
    "phan3_lstm_vs_author": {"test_nse_median": dict(zip(eval_df["run"], eval_df["NSE trung vị"])),
                             "author_nse_median": AUTHOR_NSE, "basins_bit_equal": n_equal,
                             "basins": len(results["ref"]), "max_abs_diff": max_diff,
                             "eval_minutes": dict(zip(eval_df["run"], eval_df["phút"])),
                             "q_dir_minutes": round(build_minutes, 1)},
    "phan4_mamba": {"causal_max_before_k": max(r["|Δ| lớn nhất trước k"] for r in leak_rows),
                    "axis_max_diff": max(r["|Δ| lớn nhất"] for r in axis_rows),
                    "param_formula_ok": all(r["Đạt"] == "ĐẠT" for r in param_rows),
                    "mambapy_rel_err": [r["sai số tương đối"] for r in mp_rows], "mambapy_cpu_seconds": cpu_seconds},
    "phan5_target_parameters": TARGET,
    "phan5_match": match_df.to_dict("records"),
    "phan6_dataset_minutes": round(dataset_minutes, 1),
    "phan6_short_training": table6.to_dict("records"),
    "phan6_lstm_vs_library_max_diff": ref_vs_sf,
}
(OUT_DIR / "cores_summary.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")

# (2) Môi trường
env = {"python": sys.version.split()[0], "platform": platform.platform(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "neuralhydrology_commit": src_commit, "neuralhydrology_commit_expected": NH_COMMIT,
       "neuralhydrology_branch": "core-models",
       "mamba_build": {k: MAMBA_BUILD_INFO.get(k) for k in ["built_at", "torch", "torch_cuda", "python", "arch"]},
       "inputs": {"lamah_dir": str(LAMAH_DIR), "author_run": str(AUTHOR_RUN), "mamba_wheels": str(MAMBA_INFO_PATH.parent)},
       "libs": installed_versions(ENV_LIBS), "kaggle_image": build_date,
       "q_dir_note": "LamaH_expanded_q_input dựng ở Phần 3 (qmean m³/s từ D_gauges, số âm => trống), đã xóa khỏi Output"}
(OUT_DIR / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

# (3) Dọn thư mục tạm (rmtree không đi theo liên kết tới dữ liệu gốc), danh mục tệp output
for d in [SRC_DIR, Q_DIR, cfg_train.train_dir, RUNS_DIR]:
    if Path(d).exists():
        shutil.rmtree(d)
man = [{"tệp": p.relative_to(WORK_DIR).as_posix(), "KB": round(p.stat().st_size / 1e3, 1), "sha256": sha256(p)}
       for p in sorted(OUT_DIR.rglob("*")) if p.is_file()]
MANIFEST = pd.DataFrame(man)
MANIFEST.to_csv(OUT_DIR / "manifest.csv", index=False)
show("Tệp Output (manifest.csv)", MANIFEST.assign(sha256=MANIFEST["sha256"].str[:12] + "..."))
sizes = [{"thư mục": d.name, "GB": round(du_gb(d), 3)} for d in sorted(WORK_DIR.iterdir()) if d.is_dir()]
show("Dung lượng /kaggle/working (Output của bản lưu)", pd.DataFrame(sizes))

checks = {
    f"Commit thư viện đúng {NH_COMMIT[:7]}": src_commit == NH_COMMIT,
    "Có cores_summary.json, environment.json": all((OUT_DIR / f).exists() for f in ["cores_summary.json",
                                                                                    "environment.json"]),
    "Đã xóa thư mục tạm": not any(Path(d).exists() for d in [SRC_DIR, Q_DIR, cfg_train.train_dir, RUNS_DIR]),
    "Tổng Output <= 1 GB": sum(r["GB"] for r in sizes) <= 1,
}
show("Điều kiện", pd.DataFrame({"Điều kiện": list(checks), "Đạt": ["ĐẠT" if v else "KHÔNG" for v in checks.values()]}))
if not all(checks.values()):
    raise RuntimeError("Phần 7 CHƯA đạt (dòng KHÔNG ở bảng Điều kiện)")
print("=> Đạt. Xong notebook.")
