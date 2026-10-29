# =============================================================================
# Bước B1b - B1a + qmax quá khứ: Sequential Forecast LSTM có Q của BiasCast với qmean_mmd (mm/ngày theo area_gov, như B1a)
# và thêm qmax_mmd (đỉnh ngày quá khứ, cùng đơn vị nhãn qmax), chạy đủ 30 epoch, hạt giống 111 / 222 / 333, Kaggle T4
# Bài: Konold và cs., HESS 30:5067-5096 (2026). Dữ liệu Zenodo 17119635 (CC BY-NC 4.0), config tác giả Zenodo 17292895
#      - đã lưu thành Kaggle Dataset "lamah-ce-ext" (Workspace/01_Download/01_LamaHCEExt_Download/01_LamaHCEExt_Download.py).
# Kế hoạch: Document/01_Plan/02_Pipeline.md Mục 5.1 (giao thức 30 epoch), 7 (dòng B1b), 9.5 (dự báo xuất ra).
# Khác B1a (Workspace/05_Training/03_B1a_LamaHCEExt_Train/) đúng một biến đầu vào: thêm qmax_mmd ngay sau qmean_mmd ở
# cuối pha quá khứ của hindcast_inputs, dynamic_inputs (Phần 4). Phần 3 dựng thư mục Q giống hệt B1a. Thư viện, siêu tham
# số, giao thức 30 epoch, hạt giống giữ như B0, B1a.
# Mỗi hạt giống một tệp và một notebook Kaggle cùng tên: "05_B1b_LamaHCEExt_Train_Seed111", "..._Seed222", "..._Seed333"
# - ba tệp giống hệt nhau, chỉ khác dòng RUN_SEED ở Phần 1. So sánh với B1a, xuất NPZ cho bộ đánh giá ở notebook CPU
# "06_B1b_LamaHCEExt_Compare".
# Cài đặt mỗi notebook: Accelerator = GPU T4 x2 (dùng GPU 0), Internet = On, Persistence = Files only,
# Environment = Pin to original environment, Add Input => lamah-ce-ext. Mỗi Phần dán vào 1 cell, chạy lần lượt.
# Dán nguyên tệp vào notebook rồi Save & Run All, không sửa dòng nào; Output chính thức (b1b_runs/, b1b_output/) lấy
# từ bản lưu đó.
#
# MỤC LỤC
#   Phần 1 - Cài thư viện NeuralHydrology từ fork của nhóm, kiểm môi trường
#   Phần 2 - Tìm dữ liệu, danh sách lưu vực, config tác giả của run SeqLSTM có Q
#   Phần 3 - Dựng thư mục dữ liệu có cột qmean_mmd, qmax_mmd
#   Phần 4 - Tạo config cho hạt giống RUN_SEED, đếm mẫu huấn luyện
#   Phần 5 - Huấn luyện hạt giống RUN_SEED (30 epoch, validation mỗi epoch)
#   Phần 6 - Chấm test mọi epoch và best_model.pt, kiểm best_model.pt
#   Phần 7 - Hình đường học theo epoch và loss theo bước
#   Phần 8 - Ghi môi trường, danh mục tệp, dọn thư mục dữ liệu tạm
# =============================================================================

# %% Phần 1 - Cài thư viện NeuralHydrology từ fork của nhóm, kiểm môi trường
# Cài đúng commit f229e02 của nhánh research (dda45fc + sửa lưu best_model.pt e794d7c + khóa data_on_gpu;
# 02_Pipeline.md Mục 9.1). In 3 bảng (thư viện, môi trường, điều kiện). tensorboard bắt buộc vì training/logger.py dòng 8
# nạp SummaryWriter khi import. Phần 1-5 raise khi chưa đạt để Save & Run All dừng ngay, không tốn giờ GPU cho các Phần sau.
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
NH_COMMIT = "f229e02bf98cbd979835a4b6cd503d0e3adb8bc3"   # như B0
CHECK_LIBS = ["numpy", "pandas", "xarray", "torch", "scipy", "numba", "ruamel.yaml", "matplotlib", "tensorboard", "tqdm"]
NH_MODULES = ["neuralhydrology.nh_run", "neuralhydrology.training.logger", "neuralhydrology.datautils.gpu_loader",
              "neuralhydrology.evaluation.tester", "neuralhydrology.modelzoo.sequential_forecast_lstm"]
TORCH_A4 = "2.11.0+cu128"                # torch của A4, B0 (ảnh Kaggle 10/2026); khác thì chỉ cảnh báo

RUN_SEED = 333                           # 111 là hạt giống của tác giả (config.yml); 222, 333 như A4, B0, B1a
assert RUN_SEED in (111, 222, 333), "RUN_SEED phải là 111, 222 hoặc 333"

IS_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("b1b_work")
OUT_DIR = WORK_DIR / "b1b_output"        # bảng, hình, tóm tắt, môi trường (Phần 5-8); thư mục run ở b1b_runs/
WORK_DIR.mkdir(parents=True, exist_ok=True)


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


before = installed_versions(CHECK_LIBS)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"git+{NH_REPO}@{NH_COMMIT}"], check=True)
after = installed_versions(CHECK_LIBS)
importlib.invalidate_caches()  # để tiến trình đang chạy thấy gói vừa cài

# Xác nhận pip cài đúng commit (pip ghi nguồn cài vào direct_url.json)
direct = json.loads(md.distribution("neuralhydrology").read_text("direct_url.json") or "{}")
installed_commit = direct.get("vcs_info", {}).get("commit_id")
changed = [k for k in CHECK_LIBS if before[k] != after[k]]
show("Thư viện (sau khi cài fork)", pd.DataFrame({
    "Thư viện": ["neuralhydrology"] + CHECK_LIBS,
    "Phiên bản": [f"{md.version('neuralhydrology')} (commit {installed_commit[:7] if installed_commit else '?'})"]
                 + [after[k] or "THIẾU" for k in CHECK_LIBS],
    "pip đổi?": [""] + [f"từ {before[k]}" if k in changed else "" for k in CHECK_LIBS]}))

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
}
show("Môi trường", pd.DataFrame({"Mục": list(ENV), "Giá trị": [str(v) for v in ENV.values()]}))

# Thử nạp các module sẽ dùng: nh_run (huấn luyện), logger (cần tensorboard), bộ nạp trên GPU, tester (chấm test
# 30 epoch ở Phần 6), mô hình SeqLSTM
import_ok = {}
for mod in NH_MODULES:
    try:
        importlib.import_module(mod)
        import_ok[mod] = "ĐẠT"
    except Exception as e:  # in lỗi thật để biết thiếu gì
        import_ok[mod] = f"LỖI: {type(e).__name__}: {e}"

from neuralhydrology.utils.config import Config

checks = {
    "Cài đúng commit f229e02": installed_commit == NH_COMMIT,
    "pip không đổi thư viện có sẵn": not changed,
    "Có GPU T4 (dùng GPU 0)": n_gpu >= 1 and "T4" in gpus[0],
    "Có tensorboard": after["tensorboard"] is not None,
    **{f"Nạp {m.split('.')[-1]}": v == "ĐẠT" for m, v in import_ok.items()},
    "Config có khóa data_on_gpu": isinstance(getattr(Config, "data_on_gpu", None), property),
}
show("Điều kiện", pd.DataFrame({"Điều kiện": list(checks), "Đạt": ["ĐẠT" if v else "KHÔNG" for v in checks.values()]}))
for mod, v in import_ok.items():
    if v != "ĐẠT":
        print(f"{mod}: {v}")

if torch.__version__ != TORCH_A4:
    print(f"Cảnh báo: torch {torch.__version__} khác bản của A4, B0 ({TORCH_A4}) - ảnh Kaggle đã đổi; ghi lại khi so sánh.")
if changed:
    print("Cảnh báo: pip đã đổi thư viện có sẵn - huấn luyện chạy ở tiến trình mới nên dùng bản mới; ghi lại khi so sánh.")
# Save & Run All không dừng ở dòng in => lỗi phải raise để không tốn giờ GPU cho các Phần sau
if not all(v for k, v in checks.items() if k != "pip không đổi thư viện có sẵn"):
    raise RuntimeError("Phần 1 CHƯA đạt (dòng KHÔNG ở bảng Điều kiện) - dừng notebook")
print(f"=> Đạt. Hạt giống {RUN_SEED}. Chạy Phần 2.")


# %% Phần 2 - Tìm dữ liệu, danh sách lưu vực, config tác giả của run SeqLSTM có Q
# Chỉ đọc, như B0 Phần 2: tìm dữ liệu và kết quả tác giả trong Input, đọc config.yml của run SeqLSTM có Q nhúng đơn giản,
# kiểm đúng run bằng các siêu tham số, rồi xếp mọi khóa config vào 3 nhóm: giữ nguyên / đổi ở Phần 4 / bỏ.
# Khác B0: hindcast_inputs và dynamic_inputs cùng đổi ở Phần 4 (qmean => qmean_mmd, qmax_mmd tại cùng vị trí). dynamic_inputs là
# danh sách cột bộ nạp đọc vào bộ nhớ (basedataset.py dòng 350-353, 575-586); hindcast_inputs, forecast_inputs chỉ lấy lại
# từ các cột đó (dòng 172-176), thư viện không tự dựng dynamic_inputs (config.py dòng 360-362 ở f229e02) => phải đổi cả hai danh sách.
# Kiểm dynamic_inputs của tác giả = hindcast_inputs + forecast_inputs. Tên cột mới phải nằm ngoài _VALID_TARGETS của
# bộ nạp (lamah.py dòng 18, 122-135): tên trong danh sách đó bị đọc lại từ D_gauges thay vì tệp khí tượng.
# Tự đủ: chạy lại được sau Restart mà không cần chạy lại Phần 1.
import os
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from ruamel.yaml import YAML

from neuralhydrology.datasetzoo.lamah import _VALID_TARGETS

IS_KAGGLE = Path("/kaggle/working").exists()
SEARCH_ROOT = Path(os.environ.get("LAMAH_ROOT", "/kaggle/input" if IS_KAGGLE else "."))
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("b1b_work")
AUTHOR_RUN_REL = ("Sequential_Forecast_LSTM/Sequential_Forecast_LSTM_with_q/"
                  "Sequential_Forecast_LSTM_RA_with_q_Simple_Embedding")
N_BASINS = 451
Q_OLD = "qmean"                          # cột Q quá khứ của tác giả (m³/s)
Q_NEW = ["qmean_mmd", "qmax_mmd"]        # cột Q mm/ngày theo area_gov dựng ở Phần 3 (như B1a)
Q_USE = ["qmean_mmd", "qmax_mmd"]        # cột B1b đưa vào config thay Q_OLD (qmax_mmd ngay sau qmean_mmd)
# Giá trị phải có trong config tác giả (như B0 Phần 2) - lệch là đọc nhầm run
EXPECTED = {"model": "sequential_forecast_lstm", "seed": 111, "hidden_size": 128, "batch_size": 256, "epochs": 30,
            "learning_rate": 0.001, "output_dropout": 0.3, "seq_length": 365, "forecast_seq_length": 1,
            "early_stopping_patience": 5, "early_stopping_min_delta": 0.005, "early_stopping_metric": "NSE",
            "loss": "NSE", "clip_gradient_norm": 1, "target_variables": ["qmax"]}
# Khóa đổi ở Phần 4: 12 khóa của B0 (đường dẫn, thiết bị, số tiến trình đọc, thanh tiến độ, hạt giống, tên run, giao thức
# 30 epoch, dữ liệu sẵn trên GPU) + 2 danh sách biến; khóa bỏ như B0
KEYS_CHANGE = ["data_dir", "run_dir", "train_basin_file", "validation_basin_file", "test_basin_file",
               "device", "num_workers", "verbose", "seed", "experiment_name", "early_stopping_patience", "data_on_gpu",
               "hindcast_inputs", "dynamic_inputs"]
KEYS_DROP = ["commit_hash", "package_version", "number_of_basins", "train_dir", "img_log_dir"]


def show(title, df):
    display(HTML(f"<h4 style='margin:12px 0 4px'>{title}</h4>"))
    display(df.style.hide(axis="index"))


def find_dir(root, name):
    # Thư mục duy nhất tên `name` dưới `root`; báo lỗi rõ nếu không có hoặc có nhiều.
    hits = sorted(p for p in root.rglob(name) if p.is_dir())
    if len(hits) != 1:
        raise FileNotFoundError(f"Cần đúng 1 thư mục '{name}' dưới {root}, thấy {len(hits)}: {hits} - kiểm Input lamah-ce-ext")
    return hits[0]


def brief(value, width=90):
    # Rút gọn giá trị config để bảng dễ đọc (danh sách dài => số phần tử + vài phần tử đầu).
    if isinstance(value, list) and len(value) > 4:
        text = f"[{len(value)}] " + ", ".join(map(str, value[:3])) + ", ..."
    else:
        text = str(value)
    return text if len(text) <= width else text[:width - 3] + "..."


# (1) Đường dẫn, danh sách lưu vực, tệp của run tác giả
LAMAH_DIR = find_dir(SEARCH_ROOT, "LamaH_extended")
EXP_DIR = find_dir(SEARCH_ROOT, "Experiments")
BASIN_FILE = EXP_DIR / "basins_filtered.txt"
basins = BASIN_FILE.read_text().split()
AUTHOR_RUN = EXP_DIR / AUTHOR_RUN_REL
AUTHOR_FILES = {"config": AUTHOR_RUN / "config.yml", "best_model.pt": AUTHOR_RUN / "best_model.pt",
                "scaler": AUTHOR_RUN / "train_data" / "train_data_scaler.yml",
                "test_metrics": AUTHOR_RUN / "test" / "best_model" / "test_metrics.csv"}
show("Đường dẫn và tệp", pd.DataFrame(
    [("Dữ liệu", str(LAMAH_DIR), ""), ("Kết quả tác giả", str(EXP_DIR), ""),
     ("Danh sách lưu vực", str(BASIN_FILE), f"{len(basins)} lưu vực, {len(set(basins))} mã khác nhau")]
    + [(f"Run tác giả: {k}", str(p.relative_to(EXP_DIR)), "có" if p.exists() else "THIẾU") for k, p in AUTHOR_FILES.items()],
    columns=["mục", "đường dẫn", "ghi chú"]))
if len(basins) != N_BASINS or len(set(basins)) != N_BASINS or not all(p.exists() for p in AUTHOR_FILES.values()):
    raise RuntimeError("Thiếu tệp của run tác giả hoặc danh sách lưu vực không đủ 451 mã khác nhau")

# (2) Config tác giả: kiểm đúng run, vị trí qmean, tên cột mới
AUTHOR_CFG = YAML(typ="safe").load(AUTHOR_FILES["config"].read_text(encoding="utf-8"))
exp_rows = [{"khóa": k, "mong đợi": brief(v), "trong config": brief(AUTHOR_CFG.get(k)),
             "khớp": "ĐẠT" if AUTHOR_CFG.get(k) == v else "KHÔNG"} for k, v in EXPECTED.items()]
groups = {"hindcast_inputs": AUTHOR_CFG["hindcast_inputs"], "forecast_inputs": AUTHOR_CFG["forecast_inputs"],
          "static_attributes": AUTHOR_CFG["static_attributes"]}
hind = groups["hindcast_inputs"]
dyn = AUTHOR_CFG["dynamic_inputs"]
dyn_ok = dyn == hind + groups["forecast_inputs"]
all_inputs = dyn + hind + groups["forecast_inputs"] + groups["static_attributes"]
new_in_cfg = [c for c in Q_NEW if c in all_inputs]
new_in_targets = [c for c in Q_NEW if c.lower() in _VALID_TARGETS]
exp_rows += [{"khóa": "số biến quá khứ / ngày t / thuộc tính", "mong đợi": "32 / 5 / 33",
              "trong config": " / ".join(str(len(v)) for v in groups.values()),
              "khớp": "ĐẠT" if [len(v) for v in groups.values()] == [32, 5, 33] else "KHÔNG"},
             {"khóa": f"{Q_OLD} ở pha quá khứ, đúng 1 lần", "mong đợi": "1 lần", "trong config": f"{hind.count(Q_OLD)} lần",
              "khớp": "ĐẠT" if hind.count(Q_OLD) == 1 else "KHÔNG"},
             {"khóa": "dynamic_inputs = hindcast_inputs + forecast_inputs (đúng thứ tự)", "mong đợi": "trùng",
              "trong config": f"{len(dyn)} biến, " + ("trùng" if dyn_ok else "KHÁC"), "khớp": "ĐẠT" if dyn_ok else "KHÔNG"},
             {"khóa": f"{Q_OLD} không ở pha ngày t", "mong đợi": "không",
              "trong config": "có" if Q_OLD in groups["forecast_inputs"] else "không",
              "khớp": "ĐẠT" if Q_OLD not in groups["forecast_inputs"] else "KHÔNG"},
             {"khóa": "tên cột mới chưa có trong config", "mong đợi": "chưa có",
              "trong config": ", ".join(new_in_cfg) or "chưa có", "khớp": "ĐẠT" if not new_in_cfg else "KHÔNG"},
             {"khóa": "tên cột mới ngoài _VALID_TARGETS của bộ nạp", "mong đợi": f"ngoài {_VALID_TARGETS}",
              "trong config": ", ".join(new_in_targets) or "ngoài", "khớp": "ĐẠT" if not new_in_targets else "KHÔNG"}]
show("Kiểm đúng run SeqLSTM có Q nhúng đơn giản", pd.DataFrame(exp_rows))
print(f"Biến quá khứ của tác giả ({len(hind)}), {Q_OLD} ở vị trí {hind.index(Q_OLD) + 1 if Q_OLD in hind else '-'}:")
print(", ".join(hind))

# (3) Mọi khóa config: giữ / đổi ở Phần 4 / bỏ
key_rows = []
for k in sorted(set(AUTHOR_CFG) | set(KEYS_CHANGE)):
    action = "đổi ở Phần 4" if k in KEYS_CHANGE else ("bỏ (thư viện tự ghi)" if k in KEYS_DROP else "giữ")
    key_rows.append({"khóa": k, "giá trị tác giả": brief(AUTHOR_CFG.get(k, "(không có, mặc định)")), "xử lý": action})
key_df = pd.DataFrame(key_rows).sort_values(["xử lý", "khóa"], key=lambda s: s.map(
    {"đổi ở Phần 4": 0, "bỏ (thư viện tự ghi)": 1, "giữ": 2}) if s.name == "xử lý" else s)
show(f"Khóa config ({len(AUTHOR_CFG)} khóa của tác giả): {sum(key_df['xử lý'] == 'giữ')} giữ, "
     f"{sum(key_df['xử lý'] == 'đổi ở Phần 4')} đổi, {sum(key_df['xử lý'].str.startswith('bỏ'))} bỏ", key_df)

# (4) NSE test của tác giả (mốc so ở Phần 6)
author_metrics = pd.read_csv(AUTHOR_FILES["test_metrics"], dtype={"basin": str}).set_index("basin")
AUTHOR_TEST_NSE_MEDIAN = float(author_metrics["NSE"].median())  # pandas bỏ NaN khi tính trung vị
show("NSE test của tác giả (test_metrics.csv)", pd.DataFrame([{
    "số lưu vực": len(author_metrics), "có NSE": int(author_metrics["NSE"].notna().sum()),
    "NaN": ", ".join(author_metrics.index[author_metrics["NSE"].isna()]) or "-",
    "NSE trung vị": round(AUTHOR_TEST_NSE_MEDIAN, 6), "KGE trung vị": round(float(author_metrics["KGE"].median()), 6),
    "khớp danh sách lưu vực": "ĐẠT" if set(author_metrics.index) == set(basins) else "KHÔNG"}]))

ok = all(r["khớp"] == "ĐẠT" for r in exp_rows) and set(author_metrics.index) == set(basins)
if not ok:
    raise RuntimeError("Phần 2 CHƯA đạt (dòng KHÔNG ở trên) - dừng notebook")
print("=> Đạt. Chạy Phần 3.")


# %% Phần 3 - Dựng thư mục dữ liệu có cột qmean_mmd, qmax_mmd
# Bộ nạp chỉ đọc đầu vào từ tệp khí tượng (lamah.py dòng 116-119), Q nằm ở tệp trạm D_gauges => thêm 2 cột vào tệp khí
# tượng của 451 lưu vực, như B0 Phần 3 nhưng theo mm/ngày (mã B1 cũ, commit a06a224, đã chạy đạt; lamah.py, basedataset.py
# không đổi giữa dda45fc và f229e02). Đổi đơn vị bằng chính hàm của thư viện: load_lamah_discharge (số âm -999 => NaN,
# lamah.py dòng 228-229) rồi _normalize_discharge với area_gov (dòng 303-308, cùng phép tính bộ nạp dùng cho nhãn qmax).
# Ghi 17 chữ số có nghĩa (đủ để float64 đọc lại đúng), ô thiếu để trống. Cột cũ chép nguyên từng ký tự; không thêm cột
# qmean m³/s. Thuộc tính và D_gauges trỏ liên kết về bản gốc. Tệp đã có đủ 2 cột (Persistence) thì bỏ qua.
# Giống hệt B1a Phần 3 (đạt ở bản chạy tương tác B1a: 451 tệp, lệch 7.8e-14, ô trống trùng qmean m³/s). B1b đưa cả
# hai cột vào config (Phần 4).
# Kiểm: (1) đủ 451 tệp, area_gov dương; (2) 3 lưu vực ngẫu nhiên giữ nguyên cột cũ; (3) cả 451 lưu vực, đọc bằng
# load_lamah_forcing (đúng đường bộ nạp đọc đầu vào): qmax_mmd trùng nhãn qmax bộ nạp dựng, qmean_mmd trùng qmean D_gauges
# đổi đơn vị cùng cách (lệch tương đối <= 1e-12: pandas đọc lại số 17 chữ số có thể lệch bit cuối); ô trống qmean_mmd trùng ô trống qmean m³/s mà B0 dùng
# (=> cùng tập mẫu với B0); qmean_mmd, qmax_mmd cùng mặt nạ; không âm; qmax_mmd >= qmean_mmd; (4) thống kê trong khoảng
# tính chuẩn hóa (chỉ in, thư viện tự tính ở Phần 5).
# Cần chạy Phần 2 trước trong phiên (dùng LAMAH_DIR, WORK_DIR, basins, AUTHOR_FILES, Q_NEW, show).
import random
import time

from neuralhydrology.datasetzoo.lamah import (_normalize_discharge, load_lamah_attributes, load_lamah_discharge,
                                              load_lamah_forcing)
from neuralhydrology.utils.config import Config

Q_DIR = WORK_DIR / "LamaH_q_mmd_input"
SUB_A = "A_basins_total_upstrm"
TS_DAILY = Path("2_timeseries") / "daily"
DATE_COLS = ["YYYY", "MM", "DD"]
Q_SRC = {"qmean_mmd": "qmean", "qmax_mmd": "qmax"}  # cột mới => cột của D_gauges
assert list(Q_SRC) == Q_NEW
N_CHECK = 3
LABEL_TOL = 1e-12  # lệch tương đối tối đa giữa cột ghi ra và giá trị tính lại (bản chạy tương tác đo được 7.8e-14)
random.seed(0)


def read_text_csv(path, usecols=None):
    # Đọc CSV ';' dạng chữ (không đổi sang số), ô trống giữ là chuỗi rỗng; thêm khóa ngày dạng số để ghép.
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, usecols=usecols)
    df["_key"] = pd.to_datetime(df[DATE_COLS].astype(int).rename(columns={"YYYY": "year", "MM": "month", "DD": "day"}))
    return df


def q_raw(data_dir, basin, src):
    # Q (m³/s) của D_gauges, -999 => NaN, đúng hàm bộ nạp gọi cho nhãn.
    return load_lamah_discharge(data_dir, basin, temporal_resolution="1D", normalize_discharge=False, target_name=src)[src]


def q_mmd(data_dir, basin, src):
    # Q (m³/s) => mm/ngày theo area_gov, đúng phép tính bộ nạp dùng cho nhãn.
    return _normalize_discharge(ser=q_raw(data_dir, basin, src), area=AREA.loc[basin], temporal_resolution="1D")


def fmt(ser):
    # float64 => chữ 17 chữ số có nghĩa, NaN => ô trống
    return ser.map(lambda v: "" if pd.isna(v) else format(v, ".17g"))


def build_basin_file(basin):
    # Ghép 2 cột Q mm/ngày vào tệp khí tượng theo ngày; ghi qua tệp tạm rồi đổi tên để không để lại tệp dở.
    forcing = read_text_csv(LAMAH_DIR / SUB_A / TS_DAILY / f"ID_{basin}.csv")
    for new, src in Q_SRC.items():
        forcing[new] = fmt(forcing["_key"].map(q_mmd(LAMAH_DIR, basin, src)))
    dst = Q_DIR / SUB_A / TS_DAILY / f"ID_{basin}.csv"
    tmp = dst.with_suffix(".tmp")
    forcing.drop(columns="_key").to_csv(tmp, sep=";", index=False)
    tmp.rename(dst)


def has_new_cols(path):
    return path.exists() and all(c in path.open().readline().strip().split(";") for c in Q_NEW)


def rel_diff(a, b):
    # Lệch tương đối lớn nhất trên các ô cả hai có số, và số ô khác trạng thái trống.
    both = ~np.isnan(a) & ~np.isnan(b)
    rel = np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1e-300)
    return (float(rel.max()) if rel.size else 0.0), int((np.isnan(a) != np.isnan(b)).sum())


# (1) Dựng thư mục
t0 = time.time()
AREA = load_lamah_attributes(LAMAH_DIR, sub_dataset="lamah_a")["area_gov"]
area_ok = bool(np.isfinite(AREA.loc[basins]).all() and (AREA.loc[basins] > 0).all())
(Q_DIR / SUB_A / TS_DAILY).mkdir(parents=True, exist_ok=True)
for link, target in [(Q_DIR / SUB_A / "1_attributes", LAMAH_DIR / SUB_A / "1_attributes"),
                     (Q_DIR / "D_gauges", LAMAH_DIR / "D_gauges")]:
    if not (link.exists() or link.is_symlink()):
        link.symlink_to(target, target_is_directory=True)
todo = [b for b in basins if not has_new_cols(Q_DIR / SUB_A / TS_DAILY / f"ID_{b}.csv")]
for b in todo:
    build_basin_file(b)
files = [Q_DIR / SUB_A / TS_DAILY / f"ID_{b}.csv" for b in basins]
n_ok = sum(has_new_cols(f) for f in files)
show("Thư mục Q đã dựng", pd.DataFrame([
    ("Thư mục", str(Q_DIR)),
    ("Tệp khí tượng có qmean_mmd, qmax_mmd", f"{n_ok} / {len(basins)} (dựng mới {len(todo)}, có sẵn {len(basins) - len(todo)})"),
    ("area_gov của 451 lưu vực", f"{AREA.loc[basins].min():.1f} - {AREA.loc[basins].max():,.1f} km², "
                                 + ("đều dương" if area_ok else "CÓ GIÁ TRỊ KHÔNG HỢP LỆ")),
    ("Dung lượng", f"{sum(f.stat().st_size for f in files if f.exists()) / 1e9:.2f} GB"),
    ("Liên kết", "A_basins_total_upstrm/1_attributes, D_gauges => bản gốc"),
    ("Thời gian dựng", f"{time.time() - t0:.0f} giây"),
], columns=["mục", "giá trị"]))

# (2) 3 lưu vực ngẫu nhiên: cột cũ giữ nguyên từng ký tự, chỉ thêm đúng 2 cột
rows = []
for b in random.sample(basins, N_CHECK):
    src = read_text_csv(LAMAH_DIR / SUB_A / TS_DAILY / f"ID_{b}.csv")
    new = read_text_csv(Q_DIR / SUB_A / TS_DAILY / f"ID_{b}.csv")
    rows.append({"lưu vực": b, "số cột cũ / mới": f"{src.shape[1] - 1} / {new.shape[1] - 1}",
                 "cột thêm": ", ".join(c for c in new.columns if c not in src.columns),
                 "cột cũ giữ nguyên": "ĐẠT" if new.drop(columns=Q_NEW).equals(src) else "KHÔNG"})
show(f"Kiểm {N_CHECK} lưu vực ngẫu nhiên (số cột không tính cột khóa ngày)", pd.DataFrame(rows))

# (3) Cả 451 lưu vực, đọc lại bằng đường của bộ nạp, căn theo ngày của tệp khí tượng như _load_basin_data
#     (lamah.py dòng 107-135), từ ngày đầu khởi động kỳ train tới hết kỳ test
cfg_author = Config(AUTHOR_FILES["config"])
win_start = cfg_author.train_start_date - pd.Timedelta(days=cfg_author.seq_length - 1)
win_end = cfg_author.test_end_date
t1 = time.time()
area_q = load_lamah_attributes(Q_DIR, sub_dataset="lamah_a")["area_gov"]
assert area_q.loc[basins].equals(AREA.loc[basins]), "area_gov đọc từ thư mục Q khác bản gốc"
n_days = n_neg = n_order = n_mask_mm = n_nan_b0 = n_nan_label = n_nan_mean = 0
max_rel_label = max_rel_mean = 0.0
bad = {"qmax_mmd khác nhãn": [], "qmean_mmd khác qmean D_gauges": [], "ô trống qmean_mmd khác qmean m³/s": [],
       "mặt nạ qmean_mmd khác qmax_mmd": []}
norm_vals = {c: [] for c in Q_NEW}
for b in basins:
    df = load_lamah_forcing(Q_DIR, b, sub_dataset="lamah_a")
    w = df.loc[win_start:win_end]
    qm, qx = w["qmean_mmd"].to_numpy(), w["qmax_mmd"].to_numpy()
    label = q_mmd(Q_DIR, b, "qmax").reindex(df.index).loc[win_start:win_end].to_numpy()
    mean_ref = q_mmd(Q_DIR, b, "qmean").reindex(df.index).loc[win_start:win_end].to_numpy()
    mean_m3s = q_raw(Q_DIR, b, "qmean").reindex(df.index).loc[win_start:win_end].to_numpy()
    r_label, nan_label = rel_diff(qx, label)
    r_mean, nan_mean = rel_diff(qm, mean_ref)
    nan_b0 = int((np.isnan(qm) != np.isnan(mean_m3s)).sum())
    mask_mm = int((np.isnan(qm) != np.isnan(qx)).sum())
    n_days += len(w)
    max_rel_label, max_rel_mean = max(max_rel_label, r_label), max(max_rel_mean, r_mean)
    n_nan_label, n_nan_mean, n_nan_b0, n_mask_mm = n_nan_label + nan_label, n_nan_mean + nan_mean, n_nan_b0 + nan_b0, n_mask_mm + mask_mm
    n_neg += int((qm < 0).sum() + (qx < 0).sum())
    n_order += int((qx < qm).sum())
    for key, bad_b in [("qmax_mmd khác nhãn", r_label > LABEL_TOL or nan_label),
                       ("qmean_mmd khác qmean D_gauges", r_mean > LABEL_TOL or nan_mean),
                       ("ô trống qmean_mmd khác qmean m³/s", nan_b0), ("mặt nạ qmean_mmd khác qmax_mmd", mask_mm)]:
        if bad_b:
            bad[key].append(b)
    nw = df.loc[win_start:cfg_author.train_end_date]
    for c in Q_NEW:
        norm_vals[c].append(nw[c].to_numpy())
check_rows = [
    {"kiểm": "ngày-lưu vực đã kiểm", "kết quả": f"{n_days:,}", "đạt": ""},
    {"kiểm": "qmax_mmd so với nhãn qmax (lệch tương đối lớn nhất; ô trống khác)",
     "kết quả": f"{max_rel_label:.3g}; {n_nan_label:,}", "đạt": max_rel_label <= LABEL_TOL and n_nan_label == 0},
    {"kiểm": "qmean_mmd so với qmean D_gauges đổi đơn vị (lệch tương đối lớn nhất; ô trống khác)",
     "kết quả": f"{max_rel_mean:.3g}; {n_nan_mean:,}", "đạt": max_rel_mean <= LABEL_TOL and n_nan_mean == 0},
    {"kiểm": "ô trống qmean_mmd khác ô trống qmean m³/s của B0", "kết quả": f"{n_nan_b0:,}", "đạt": n_nan_b0 == 0},
    {"kiểm": "mặt nạ qmean_mmd khác qmax_mmd", "kết quả": f"{n_mask_mm:,}", "đạt": n_mask_mm == 0},
    {"kiểm": "giá trị âm", "kết quả": f"{n_neg:,}", "đạt": n_neg == 0},
    {"kiểm": "qmax_mmd < qmean_mmd", "kết quả": f"{n_order:,}", "đạt": n_order == 0}]
show(f"Kiểm cả {len(basins)} lưu vực, {win_start.date()} - {win_end.date()} ({time.time() - t1:.0f} giây)",
     pd.DataFrame(check_rows).assign(đạt=lambda d: d["đạt"].map({True: "ĐẠT", False: "KHÔNG", "": ""})))
for name, ids in bad.items():
    if ids:
        print(f"Lưu vực {name}:", ", ".join(ids[:20]), "..." if len(ids) > 20 else "")

# (4) Thống kê trong khoảng tính chuẩn hóa (364 ngày khởi động + kỳ train, basedataset.py dòng 437-449); chỉ in
stat_rows = []
for c in Q_NEW:
    arr = np.concatenate(norm_vals[c])
    stat_rows.append({"cột": c, "trung bình (mm/ngày)": f"{np.nanmean(arr):.4f}", "độ lệch chuẩn": f"{np.nanstd(arr):.4f}",
                      "ô trống": f"{np.isnan(arr).sum():,} / {arr.size:,} ({np.isnan(arr).mean():.2%})"})
show(f"Thống kê {win_start.date()} - {cfg_author.train_end_date.date()}, {len(basins)} lưu vực", pd.DataFrame(stat_rows))

ok = (n_ok == len(basins) and area_ok and all(r["cột cũ giữ nguyên"] == "ĐẠT" for r in rows)
      and all(r["đạt"] is True for r in check_rows[1:]))
if not ok:
    raise RuntimeError("Phần 3 CHƯA đạt (dòng KHÔNG ở trên) - dừng notebook")
print("=> Đạt. Chạy Phần 4.")


# %% Phần 4 - Tạo config cho hạt giống RUN_SEED, đếm mẫu huấn luyện
# Dựng từ config tác giả như B1a Phần 4. Đổi 14 khóa KEYS_CHANGE (Phần 2), bỏ 5 khóa KEYS_DROP; mọi siêu tham số khác
# giữ nguyên. Khác B1a đúng ở hai danh sách biến: thêm qmax_mmd ngay sau qmean_mmd (cuối pha quá khứ, trước 5 biến ECMWF
# HRES trong dynamic_inputs). Thêm một biến quá khứ => lớp nhúng fc [16] thêm 16 trọng số (84.785 => 84.801, như B1 cũ).
# Kiểm: (1) nạp được bằng Config; chỉ khác tác giả ở khóa đã liệt kê; 33 / 5 / 38 biến, không còn qmean, qmax_mmd ngay sau qmean_mmd;
# (2) dựng dataset train bằng thư viện (khoảng 2 phút; data_on_gpu chỉ đổi cách tạo lô ở basetrainer.py, tester.py, không
# đổi dataset) => số mẫu sau lọc, số bước/epoch, số mẫu, số bước bằng B0 (1.094.272 mẫu, 4.275 bước - A4
# Plans/02_A4_Timing_Test.ipynb, B1 cũ Phần 4) vì qmean_mmd, qmax_mmd cùng ô trống với qmean (Phần 3); tham số 84.801; mảng Q mà thư viện
# dùng tính chuẩn hóa (bắt tại _setup_normalization, basedataset.py dòng 755, 762) phải trùng dữ liệu Phần 3 từng ô: cùng
# ô trống, lệch giá trị không quá làm tròn float32 (thư viện ép float32, dòng 495). Thống kê chuẩn hóa chỉ in (thư viện tính
# trên float32 bằng xarray nên lệch số float64 của Phần 3 cỡ 1e-4 dù dữ liệu trùng - B1 cũ).
# Dataset dựng ở thư mục tạm, xóa sau khi đếm.
# Cần chạy Phần 1-3 trước trong phiên (dùng RUN_SEED, AUTHOR_FILES, AUTHOR_CFG, KEYS_CHANGE, KEYS_DROP, BASIN_FILE, Q_DIR,
# Q_OLD, Q_NEW, Q_USE, norm_vals, WORK_DIR, brief, show).
import gc
import shutil

import neuralhydrology.datasetzoo.basedataset as nh_basedataset
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.modelzoo import get_model

RUNS_DIR = WORK_DIR / "b1b_runs"        # thư mục cha của thư mục run
CFG_DIR = WORK_DIR / "b1b_configs"
COUNT_DIR = WORK_DIR / "b1b_count_tmp"  # train_dir tạm cho bước đếm mẫu (thư viện ghi tệp chuẩn hóa vào đây)
N_WORKERS = 2
PATIENCE = 30                           # bằng epochs của tác giả => không dừng sớm
B0_SAMPLES, B0_STEPS, B0_PARAMS = 1_094_272, 4_275, 84_785
B1B_PARAMS = B0_PARAMS + 16           # thêm một biến quá khứ: 16 trọng số của lớp nhúng fc [16]
F32_TOL = 1e-6                          # lệch tương đối tối đa từng ô giữa mảng thư viện (float32) và Phần 3 (float64)


def experiment_name(seed):
    return f"B1b_SeqLSTM_Q_seed{seed}"


def swap_q(inputs):
    # qmean => Q_USE tại cùng vị trí.
    i = inputs.index(Q_OLD)
    return inputs[:i] + Q_USE + inputs[i + 1:]


def write_config(seed):
    # Ghi tệp config cho một hạt giống, luôn dựng lại từ config gốc của tác giả để không cộng dồn thay đổi.
    yaml_rt = YAML()
    yaml_rt.width = 4096
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml_rt.load(AUTHOR_FILES["config"].read_text(encoding="utf-8"))
    new_vals = {"data_dir": str(Q_DIR), "run_dir": str(RUNS_DIR), "train_basin_file": str(BASIN_FILE),
                "validation_basin_file": str(BASIN_FILE), "test_basin_file": str(BASIN_FILE), "device": "cuda:0",
                "num_workers": N_WORKERS, "verbose": 0, "seed": seed, "experiment_name": experiment_name(seed),
                "early_stopping_patience": PATIENCE, "data_on_gpu": True,
                "hindcast_inputs": swap_q(list(cfg["hindcast_inputs"])),
                "dynamic_inputs": swap_q(list(cfg["dynamic_inputs"]))}
    assert set(new_vals) == set(KEYS_CHANGE), "new_vals phải đúng các khóa KEYS_CHANGE của Phần 2"
    for k in KEYS_DROP:
        cfg.pop(k, None)
    for k, v in new_vals.items():
        cfg[k] = v
    path = CFG_DIR / f"{experiment_name(seed)}.yml"
    with open(path, "w", encoding="utf-8") as f:
        yaml_rt.dump(cfg, f)
    return path


CFG_PATH = write_config(RUN_SEED)

# (1) Nạp bằng Config của thư viện, so từng khóa với config tác giả, kiểm danh sách biến
new = YAML(typ="safe").load(CFG_PATH.read_text(encoding="utf-8"))
cfg_nh = Config(CFG_PATH)  # báo lỗi nếu có khóa lạ hoặc giá trị sai kiểu
load_ok = (cfg_nh.epochs == PATIENCE and cfg_nh.early_stopping_patience == PATIENCE and cfg_nh.data_on_gpu is True
           and cfg_nh.seed == RUN_SEED)
show("Nạp config bằng Config của thư viện", pd.DataFrame([
    ("tệp", CFG_PATH.name), ("seed", cfg_nh.seed), ("epochs", cfg_nh.epochs),
    ("early_stopping_patience", cfg_nh.early_stopping_patience), ("data_on_gpu", cfg_nh.data_on_gpu),
    ("num_workers", cfg_nh.num_workers), ("device", cfg_nh.device), ("data_dir", str(cfg_nh.data_dir)),
    ("kết quả", "ĐẠT" if load_ok else "KHÔNG")], columns=["khóa", "giá trị"]).astype(str))
diff_rows = []
for k in sorted(set(AUTHOR_CFG) | set(new)):
    a, b = AUTHOR_CFG.get(k, "(không có)"), new.get(k, "(bỏ)")
    if a != b:
        diff_rows.append({"khóa": k, "tác giả": brief(a, 60), "B1b": brief(b, 60),
                          "được phép": "ĐẠT" if k in KEYS_CHANGE + KEYS_DROP else "KHÔNG"})
show(f"Khóa khác config tác giả ({len(diff_rows)} khóa; seed tác giả là 111 nên hạt giống 111 không có dòng seed)",
     pd.DataFrame(diff_rows))
hind_new, dyn_new, fc = cfg_nh.hindcast_inputs, cfg_nh.dynamic_inputs, cfg_nh.forecast_inputs
var_rows = [
    {"kiểm": "số biến quá khứ / ngày t / động", "B1b": f"{len(hind_new)} / {len(fc)} / {len(dyn_new)}",
     "mong đợi": "33 / 5 / 38", "đạt": [len(hind_new), len(fc), len(dyn_new)] == [33, 5, 38]},
    {"kiểm": "dynamic_inputs = hindcast_inputs + forecast_inputs", "B1b": "trùng" if dyn_new == hind_new + fc else "KHÁC",
     "mong đợi": "trùng", "đạt": dyn_new == hind_new + fc},
    {"kiểm": "qmean => qmean_mmd, qmax_mmd tại cùng vị trí, các biến khác giữ nguyên thứ tự",
     "B1b": ", ".join(f"{c} ở vị trí {hind_new.index(c) + 1}" for c in Q_USE if c in hind_new),
     "mong đợi": ", ".join(f"{c} ở vị trí {AUTHOR_CFG['hindcast_inputs'].index(Q_OLD) + 1 + i}" for i, c in enumerate(Q_USE)),
     "đạt": hind_new == swap_q(AUTHOR_CFG["hindcast_inputs"])},
    {"kiểm": f"không còn {Q_OLD}", "B1b": f"{dyn_new.count(Q_OLD)} lần", "mong đợi": "0 lần", "đạt": Q_OLD not in dyn_new}]
show("Biến đầu vào (đọc lại bằng Config)",
     pd.DataFrame(var_rows).assign(đạt=lambda d: d["đạt"].map({True: "ĐẠT", False: "KHÔNG"})))

# (2) Đếm mẫu bằng dataset train của thư viện (cùng hàm BaseTrainer._get_dataset gọi), số tham số mô hình
t0 = time.time()
shutil.rmtree(COUNT_DIR, ignore_errors=True)
cfg_count = Config(CFG_PATH)
cfg_count.train_dir = COUNT_DIR
cfg_count.device = "cpu"
_setup_norm = nh_basedataset.BaseDataset._setup_normalization
grabbed = {}


def _grab_norm_input(self, xr):
    # giữ lại đúng mảng thư viện dùng để tính chuẩn hóa rồi gọi hàm gốc
    grabbed["xr"] = xr[Q_USE].load()
    _setup_norm(self, xr)


nh_basedataset.BaseDataset._setup_normalization = _grab_norm_input
try:
    ds = get_dataset(cfg=cfg_count, is_train=True, period="train", scaler={})
finally:
    nh_basedataset.BaseDataset._setup_normalization = _setup_norm
n_samples = len(ds)
n_steps = -(-n_samples // cfg_count.batch_size)  # chia làm tròn lên: lô cuối không đủ 256 vẫn chạy
scaler = ds.scaler
n_params = sum(p.numel() for p in get_model(cfg_count).parameters())
t_ds = time.time() - t0
stat_rows = []
lib_xr = grabbed["xr"]
for c in Q_USE:
    lib = lib_xr[c].transpose("basin", "date").sel(basin=basins).to_numpy().astype(np.float64)
    mine = np.stack(norm_vals[c])
    same_shape = lib.shape == mine.shape
    same_nan = same_shape and bool((np.isnan(lib) == np.isnan(mine)).all())
    both = ~np.isnan(mine) & ~np.isnan(lib) if same_shape else np.zeros(1, bool)
    rel = float((np.abs(lib - mine)[both] / np.maximum(np.abs(mine[both]), 1e-6)).max()) if same_shape and both.any() else np.nan
    lib_m, lib_s = float(scaler["xarray_feature_center"][c]), float(scaler["xarray_feature_scale"][c])
    p3_m, p3_s = float(np.nanmean(mine)), float(np.nanstd(mine))
    stat_rows.append({"cột": c, "mảng thư viện / Phần 3": f"{lib.shape} / {mine.shape}",
                      "ô có số": f"{int((~np.isnan(lib)).sum()):,} / {int(both.sum()):,} chung",
                      "cùng ô trống": "ĐẠT" if same_nan else "KHÔNG", "lệch tương đối lớn nhất từng ô": f"{rel:.2g}",
                      "trung bình thư viện / Phần 3": f"{lib_m:.6f} / {p3_m:.6f}",
                      "độ lệch chuẩn thư viện / Phần 3": f"{lib_s:.6f} / {p3_s:.6f}",
                      "đạt": "ĐẠT" if same_nan and rel <= F32_TOL else "KHÔNG"})
del ds, grabbed, lib_xr
gc.collect()
shutil.rmtree(COUNT_DIR, ignore_errors=True)
count_rows = [{"mục": "mẫu huấn luyện sau lọc", "B1b": f"{n_samples:,}", "B0": f"{B0_SAMPLES:,}",
               "đạt": "ĐẠT" if n_samples == B0_SAMPLES else "KHÔNG"},
              {"mục": f"bước / epoch (batch {cfg_count.batch_size})", "B1b": f"{n_steps:,}", "B0": f"{B0_STEPS:,}",
               "đạt": "ĐẠT" if n_steps == B0_STEPS else "KHÔNG"},
              {"mục": "tham số mô hình", "B1b": f"{n_params:,}", "B0": f"{B0_PARAMS:,} (B1b mong đợi {B1B_PARAMS:,})",
               "đạt": "ĐẠT" if n_params == B1B_PARAMS else "KHÔNG"}]
show(f"Dataset train của thư viện ({t_ds / 60:.1f} phút)", pd.DataFrame(count_rows))
show("Mảng thư viện dùng để chuẩn hóa so với dữ liệu Phần 3 (thống kê chỉ in)", pd.DataFrame(stat_rows))

ok = (load_ok and all(r["được phép"] == "ĐẠT" for r in diff_rows) and all(r["đạt"] for r in var_rows)
      and all(r["đạt"] == "ĐẠT" for r in count_rows + stat_rows))
print(f"Config: {CFG_PATH}; thư mục run sẽ tạo dưới {RUNS_DIR}")
if not ok:
    raise RuntimeError("Phần 4 CHƯA đạt (dòng KHÔNG ở trên) - dừng notebook")
print("=> Đạt. Chạy Phần 5.")



# %% Phần 5 - Huấn luyện hạt giống RUN_SEED (30 epoch, validation mỗi epoch)
# Như A4 Phần 5: chạy "python -m neuralhydrology.nh_run train" thành tiến trình riêng trên GPU 0 (lỗi của thư viện không
# làm treo kernel, log đầy đủ ghi ra tệp). Thư viện tự tạo b1b_runs/B1b_SeqLSTM_Q_seed<N>_<ddmm_hhmmss>/, lưu
# model_epochNNN.pt mỗi epoch, validation/model_epochNNN/validation_metrics.csv, best_model.pt, output.log, tệp tensorboard.
# Thêm so với A4: mỗi 5 phút in kèm bộ nhớ và mức dùng GPU 0 (nvidia-smi); thời gian huấn luyện và validation từng epoch
# tính từ mốc giờ trong log ("%(asctime)s: %(message)s", logging_utils.py dòng 21):
#   huấn luyện epoch e = "Epoch e average loss" trừ dòng log cuối của epoch e-1 (epoch 1: dòng "Early stopping enabled",
#     dòng cuối của bước dựng, basetrainer.py dòng 224 - sau khi dựng dataset, bộ nạp, mô hình);
#   validation epoch e = "Epoch e average validation loss" trừ "Epoch e average loss" (gồm cả lưu trọng số epoch, vài giây);
#   validation epoch 1 gồm cả dựng dataset validation lần đầu (cache_validation_data).
# Kiểm cuối: mã thoát 0, đúng một thư mục run, đủ số epoch của config (không dừng sớm), có best_model.pt => thiếu là
# dừng notebook. Epoch ghi best_model.pt sau cùng là epoch NSE trung vị validation cao nhất (bản sửa e794d7c) và thời gian
# chỉ cảnh báo, không dừng: run đã xong, Phần 6-8 vẫn phải chấm test và lưu kết quả (Phần 6 kiểm lại bằng tensor).
# Số epoch, tên run đọc từ CFG_PATH nên cell TEST tạm dùng lại được mã này với config ngắn.
# Cần chạy Phần 1-4 trước trong phiên (dùng CFG_PATH, RUNS_DIR, OUT_DIR, WORK_DIR, Config, show).
import re
import time

POLL_S = 300                                               # in tiến trình mỗi 5 phút
LOG_KEYS = ("average loss", "average validation loss", "New best model", "stopped early", "Error", "Traceback")
cfg_run = Config(CFG_PATH)
N_EPOCHS, EXP_NAME = cfg_run.epochs, cfg_run.experiment_name
STDOUT_LOG = OUT_DIR / f"train_{EXP_NAME}_stdout.log"      # stdout + stderr của tiến trình (giữ cả traceback nếu lỗi)


def gpu_now():
    # Bộ nhớ đang dùng (MiB) và mức dùng (%) của GPU 0 theo nvidia-smi; None nếu không đọc được.
    try:
        out = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        mem, util = (int(x) for x in out.strip().split(","))
        return mem, util
    except Exception:
        return None


old_runs = sorted(RUNS_DIR.glob(f"{EXP_NAME}_*"))
if old_runs:
    raise RuntimeError(f"Đã có thư mục run {EXP_NAME}: {[p.name for p in old_runs]} - kiểm trước khi chạy lại.")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
cmd = [sys.executable, "-m", "neuralhydrology.nh_run", "train", "--config-file", str(CFG_PATH)]
gpu_samples = []                                           # (giờ từ lúc bắt đầu, MiB, %)
t_start = time.time()
with STDOUT_LOG.open("w") as fout:
    proc = subprocess.Popen(cmd, stdout=fout, stderr=subprocess.STDOUT, env=env, cwd=str(WORK_DIR))
    print(f"Bắt đầu {time.strftime('%H:%M:%S')}, {EXP_NAME}, {N_EPOCHS} epoch, PID {proc.pid}")
    n_seen = 0
    while True:
        done = proc.poll() is not None
        hours = (time.time() - t_start) / 3600
        g = None if done else gpu_now()
        if g:
            gpu_samples.append((hours, *g))
        lines = [l.rstrip() for l in STDOUT_LOG.read_text(errors="replace").splitlines() if any(k in l for k in LOG_KEYS)]
        for l in lines[n_seen:]:
            print(f"[{hours:.2f} giờ] {l}")
        n_seen = len(lines)
        if g:
            print(f"[{hours:.2f} giờ] GPU 0: {g[0]:,} MiB, {g[1]}%")
        if done:
            break
        try:
            proc.wait(timeout=POLL_S)                      # chờ tối đa 5 phút, thoát ngay khi tiến trình xong
        except subprocess.TimeoutExpired:
            pass
TRAIN_HOURS = (time.time() - t_start) / 3600

# Bảng theo epoch: loss huấn luyện, loss và chỉ số trung vị validation, thời gian, epoch ghi best_model.pt
runs = sorted(RUNS_DIR.glob(f"{EXP_NAME}_*"))
RUN_DIR = runs[0] if len(runs) == 1 else None
epoch_files = sorted(RUN_DIR.glob("model_epoch*.pt")) if RUN_DIR else []
val_dirs = sorted((RUN_DIR / "validation").glob("model_epoch*")) if RUN_DIR else []
text = STDOUT_LOG.read_text(errors="replace")

line_re = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}): (.*)$")
loss_re = re.compile(r"^Epoch (\d+) average loss: .*?avg_total_loss: ([-\w.]+)")
val_re = re.compile(r"^Epoch (\d+) average validation loss: ([-\w.]+)")
metric_re = re.compile(r"\b(NSE|KGE): ([-\w.]+)")
best_re = re.compile(r"^New best model saved at epoch (\d+)")
rows, last_mark, best_epochs = {}, None, []
for raw in text.splitlines():
    m = line_re.match(raw)
    if not m:
        continue
    ts, msg = pd.Timestamp(m.group(1).replace(",", ".")), m.group(2)
    if msg.startswith("Early stopping enabled"):
        last_mark = ts
    elif (mm := loss_re.match(msg)):
        e = int(mm.group(1))
        rows[e] = {"epoch": e, "loss huấn luyện": float(mm.group(2)), "_t_loss": ts,
                   "phút huấn luyện": (ts - last_mark).total_seconds() / 60 if last_mark is not None else np.nan}
        last_mark = ts
    elif (mm := val_re.match(msg)):
        e = int(mm.group(1))
        rows[e].update({"loss validation": float(mm.group(2)),
                        **{f"{k} trung vị validation": float(v) for k, v in metric_re.findall(msg)},
                        "phút validation": (ts - rows[e]["_t_loss"]).total_seconds() / 60})
        last_mark = ts
    elif (mm := best_re.match(msg)):
        best_epochs.append(int(mm.group(1)))
        last_mark = ts
TRAIN_LOG = pd.DataFrame(sorted(rows.values(), key=lambda r: r["epoch"])).drop(columns="_t_loss", errors="ignore")
nse_col = "NSE trung vị validation"
if len(TRAIN_LOG):
    TRAIN_LOG["ghi best_model.pt"] = TRAIN_LOG["epoch"].isin(best_epochs).map({True: "có", False: ""})
    TRAIN_LOG.to_csv(OUT_DIR / f"train_{EXP_NAME}_epochs.csv", index=False)
    show(f"Theo epoch - {EXP_NAME}", TRAIN_LOG.round(5))
GPU_LOG = pd.DataFrame(gpu_samples, columns=["giờ", "MiB", "%"])
GPU_LOG.to_csv(OUT_DIR / f"train_{EXP_NAME}_gpu.csv", index=False)

has_val = nse_col in TRAIN_LOG and TRAIN_LOG[nse_col].notna().any()
BEST_EPOCH = best_epochs[-1] if best_epochs else None
argmax_epoch = int(TRAIN_LOG.loc[TRAIN_LOG[nse_col].idxmax(), "epoch"]) if has_val else None
if has_val:
    show("Thời gian và bộ nhớ", pd.DataFrame([
        ("Huấn luyện (phút/epoch, trung vị)", f"{TRAIN_LOG['phút huấn luyện'].median():.2f}"),
        ("Validation (phút/lượt, trung vị từ epoch 2)", f"{TRAIN_LOG['phút validation'].iloc[1:].median():.2f}"),
        ("Validation epoch 1 (gồm dựng dataset)", f"{TRAIN_LOG['phút validation'].iloc[0]:.2f}"),
        ("Tổng thời gian tiến trình", f"{TRAIN_HOURS:.2f} giờ"),
        ("Bộ nhớ GPU 0 lớn nhất trong các lần đọc (nvidia-smi)", f"{GPU_LOG['MiB'].max():,} MiB" if len(GPU_LOG) else "chưa đọc"),
        ("Mức dùng GPU 0 trung vị trong các lần đọc", f"{GPU_LOG['%'].median():.0f}%" if len(GPU_LOG) else "chưa đọc"),
    ], columns=["mục", "giá trị"]))

n_val = int(TRAIN_LOG[nse_col].notna().sum()) if has_val else 0
checks = [("Mã thoát của tiến trình", proc.returncode, "0", proc.returncode == 0),
          ("Số thư mục run", len(runs), "1", len(runs) == 1),
          ("Số model_epoch*.pt / thư mục validation", f"{len(epoch_files)} / {len(val_dirs)}", f"{N_EPOCHS} / {N_EPOCHS}",
           len(epoch_files) == len(val_dirs) == N_EPOCHS),
          ("Số epoch có NSE validation trong log", n_val, N_EPOCHS, n_val == N_EPOCHS),
          ("Dừng sớm", "có" if "stopped early" in text else "không", "không", "stopped early" not in text),
          ("Có best_model.pt", bool(RUN_DIR and (RUN_DIR / "best_model.pt").exists()), "True",
           bool(RUN_DIR and (RUN_DIR / "best_model.pt").exists())),
          ("Epoch ghi best_model.pt sau cùng / epoch NSE validation cao nhất", f"{BEST_EPOCH} / {argmax_epoch}",
           "bằng nhau", BEST_EPOCH is not None and BEST_EPOCH == argmax_epoch),
          ("Thời gian huấn luyện", f"{TRAIN_HOURS:.2f} giờ", "<= 10 giờ", TRAIN_HOURS <= 10)]
N_HARD = 6                                                 # 6 dòng đầu thiếu là không chấm test được => dừng notebook
show("Kiểm", pd.DataFrame([{"mục": a, "giá trị": str(b), "cần": str(c), "đạt": "ĐẠT" if d else "KHÔNG",
                            "thiếu thì": "dừng" if i < N_HARD else "cảnh báo"} for i, (a, b, c, d) in enumerate(checks)]))
print(f"Thư mục run: {RUN_DIR}")
if not all(c[3] for c in checks[:N_HARD]):
    print("\n".join(text.splitlines()[-40:]))
    raise RuntimeError(f"Phần 5 CHƯA đạt - 40 dòng cuối của {STDOUT_LOG.name} in ở trên; dừng notebook")
print("=> Đạt. Chạy Phần 6." if all(c[3] for c in checks) else
      "=> Cảnh báo (dòng KHÔNG ở trên) - vẫn chạy Phần 6-8 để chấm test và lưu kết quả, xem lại sau.")


# %% Phần 6 - Chấm test mọi epoch và best_model.pt, kiểm best_model.pt
# Theo dõi test mỗi epoch (họp GVHD 25/10/2026); chọn mô hình vẫn chỉ theo NSE trung vị validation (02_Pipeline.md Mục 6.1).
# (1) best_model.pt: so từng tensor với mọi model_epochNNN.pt, phải trùng đúng một epoch và là epoch Phần 5 ghi sau cùng.
# (2) Validation: NSE, KGE trung vị 451 lưu vực từng epoch từ validation/model_epochNNN/validation_metrics.csv
#     (tester.py _save_results), đối chiếu số in trong log; epoch NSE cao nhất phải là epoch của best_model.pt.
# (3) Test 2014-2017 cho mọi epoch: một Tester kỳ test (get_tester, init_model=True), dựng dataset 451 lưu vực một lần vào
#     cached_datasets rồi gọi evaluate(epoch=e) cho từng epoch và "best". Tester đọc cache ở mọi kỳ khi
#     cache_validation_data bật, chỉ tự ghi cache ở kỳ validation (tester.py dòng 232-241) => tự điền cache trước; với
#     data_on_gpu, bản trên GPU gắn vào dataset (tester.py dòng 165-172) nên dựng một lần. Kết quả ghi như eval_run:
#     test/model_epochNNN/test_metrics.csv, test/best_model/. Chỉ giữ test_results.p của best_model.pt (notebook so sánh
#     dựng NPZ từ tệp này), xóa của các epoch khác để gọn Output.
# (4) Kiểm cache không đổi kết quả: chấm lại best_model.pt bằng Tester mới không cache (như eval_run), so NSE từng lưu vực.
# Cần chạy Phần 1-5 trước trong phiên (dùng RUN_DIR, TRAIN_LOG, BEST_EPOCH, EXP_NAME, TRAIN_HOURS, RUN_SEED, OUT_DIR,
# AUTHOR_TEST_NSE_MEDIAN, author_metrics, basins, show).
import torch
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.errors import NoEvaluationDataError

LOG_TOL = 1e-4                                             # log in chỉ số 5 chữ số thập phân
SAME_TOL = 1e-6                                            # cùng trọng số => cùng chỉ số


def epoch_of(path):
    return int(path.stem[-3:])


def same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def period_label(cfg, period):
    # "validation 2010-2013" từ ngày bắt đầu, kết thúc trong config (không ghi cứng kỳ dữ liệu).
    start, end = getattr(cfg, f"{period}_start_date"), getattr(cfg, f"{period}_end_date")
    start, end = (s[0] if isinstance(s, list) else s for s in (start, end))
    return f"{period} {start.year}-{end.year}"


def metric_medians(results):
    # evaluate trả {lưu vực: {tần suất: {chỉ số: giá trị, "xr": ...}}}; trả Series NSE, KGE theo lưu vực.
    df = pd.DataFrame({b: {k: v for k, v in next(iter(r.values())).items() if k in ("NSE", "KGE")}
                       for b, r in results.items()}).T
    return df.reindex(columns=["NSE", "KGE"]).astype(float)


cfg_saved = Config(RUN_DIR / "config.yml")                 # config thư viện ghi trong thư mục run
epoch_files = sorted(RUN_DIR.glob("model_epoch*.pt"))
EPOCHS = [epoch_of(p) for p in epoch_files]

# (1) epoch của best_model.pt
best_sd = torch.load(RUN_DIR / "best_model.pt", map_location="cpu", weights_only=True)
match = [epoch_of(p) for p in epoch_files if same_weights(best_sd, torch.load(p, map_location="cpu", weights_only=True))]
SAVED_EPOCH = match[0] if len(match) == 1 else None

# (2) validation theo epoch từ tệp
val_rows = []
for e in EPOCHS:
    m = pd.read_csv(RUN_DIR / "validation" / f"model_epoch{e:03d}" / "validation_metrics.csv", index_col=0)
    val_rows.append({"epoch": e, "NSE trung vị validation (tệp)": float(m["NSE"].median()),
                     "KGE trung vị validation (tệp)": float(m["KGE"].median()), "số lưu vực validation": len(m)})
CURVE = TRAIN_LOG.merge(pd.DataFrame(val_rows), on="epoch", how="outer")
CURVE["khớp log"] = ((CURVE["NSE trung vị validation (tệp)"] - CURVE["NSE trung vị validation"]).abs() <= LOG_TOL) & \
                    ((CURVE["KGE trung vị validation (tệp)"] - CURVE["KGE trung vị validation"]).abs() <= LOG_TOL)
TRUE_BEST = int(CURVE.loc[CURVE["NSE trung vị validation (tệp)"].idxmax(), "epoch"])

# (3) test mọi epoch bằng một Tester có cache
t0 = time.time()
tester = get_tester(cfg=cfg_saved, run_dir=RUN_DIR, period="test", init_model=True)
skipped = []
for b in tester.basins:
    try:
        tester.cached_datasets[b] = tester._get_dataset(b)
    except NoEvaluationDataError:
        skipped.append(b)
BUILD_MIN = (time.time() - t0) / 60
print(f"Dựng dataset test {len(tester.cached_datasets)} lưu vực: {BUILD_MIN:.1f} phút; bỏ qua {len(skipped)} {skipped}")

test_rows, test_min, per_basin_best = [], [], None
for e in EPOCHS + ["best"]:
    t1 = time.time()
    res = tester.evaluate(epoch=e, save_results=True, metrics=cfg_saved.metrics)
    test_min.append((time.time() - t1) / 60)
    pb = metric_medians(res)
    sub = "best_model" if e == "best" else f"model_epoch{e:03d}"
    f_metrics = RUN_DIR / "test" / sub / "test_metrics.csv"
    nse_file = float(pd.read_csv(f_metrics, index_col=0)["NSE"].median())
    if e == "best":
        per_basin_best = pb
    else:
        (RUN_DIR / "test" / sub / "test_results.p").unlink(missing_ok=True)
    test_rows.append({"epoch": e, "NSE trung vị test": float(pb["NSE"].median()), "KGE trung vị test": float(pb["KGE"].median()),
                      "lưu vực có NSE test": int(pb["NSE"].notna().sum()), "_nse_file": nse_file})
    if e != "best" and (e % 5 == 0 or e == EPOCHS[-1]):
        print(f"  test epoch {e}: NSE trung vị {test_rows[-1]['NSE trung vị test']:.4f}, {test_min[-1]:.2f} phút")
TEST_ALL = pd.DataFrame(test_rows)
best_row = TEST_ALL[TEST_ALL["epoch"] == "best"].iloc[0]
TEST_EPOCHS = TEST_ALL[TEST_ALL["epoch"] != "best"].astype({"epoch": int}).drop(columns="_nse_file")
CURVE = CURVE.merge(TEST_EPOCHS, on="epoch", how="left")
CURVE["ghi chú"] = [("best_model.pt" if e == SAVED_EPOCH else "") + (" + NSE val cao nhất" if e == TRUE_BEST else "")
                    for e in CURVE["epoch"]]
CURVE.to_csv(OUT_DIR / f"curve_{EXP_NAME}.csv", index=False)
show(f"Theo epoch - {EXP_NAME} ({period_label(cfg_saved, 'validation')}, {period_label(cfg_saved, 'test')}, "
     f"trung vị {len(basins)} lưu vực)",
     CURVE[["epoch", "loss huấn luyện", "loss validation", "NSE trung vị validation (tệp)", "KGE trung vị validation (tệp)",
            "NSE trung vị test", "KGE trung vị test", "ghi chú"]].round(5))
TEST_EVAL_MIN = float(np.median(test_min))
print(f"Chấm test: trung vị {TEST_EVAL_MIN:.2f} phút/epoch, tổng {sum(test_min) + BUILD_MIN:.1f} phút kể cả dựng dataset")

# (4) chấm lại best_model.pt bằng Tester mới không cache, không ghi tệp (như eval_run)
t1 = time.time()
fresh = metric_medians(get_tester(cfg=cfg_saved, run_dir=RUN_DIR, period="test", init_model=True)
                       .evaluate(epoch="best", save_results=False, metrics=cfg_saved.metrics))
FRESH_MIN = (time.time() - t1) / 60
both = per_basin_best.join(fresh, rsuffix="_fresh", how="outer")
diff_fresh = float((both["NSE"] - both["NSE_fresh"]).abs().max())
nan_same = bool((both["NSE"].isna() == both["NSE_fresh"].isna()).all())

ep_saved = TEST_EPOCHS.set_index("epoch").loc[SAVED_EPOCH] if SAVED_EPOCH in EPOCHS else None
show(f"{period_label(cfg_saved, 'test')} - {EXP_NAME}", pd.DataFrame([
    {"mô hình": f"best_model.pt (epoch {SAVED_EPOCH})", "NSE trung vị test": best_row["NSE trung vị test"],
     "KGE trung vị test": best_row["KGE trung vị test"], "lưu vực có NSE test": best_row["lưu vực có NSE test"]},
    {"mô hình": "tác giả (best_model.pt, Zenodo)", "NSE trung vị test": AUTHOR_TEST_NSE_MEDIAN,
     "KGE trung vị test": float(author_metrics["KGE"].median()),
     "lưu vực có NSE test": int(author_metrics["NSE"].notna().sum())}]).round(5))

SUMMARY = {"seed": RUN_SEED, "experiment_name": EXP_NAME, "run_dir": RUN_DIR.name, "n_epochs": len(EPOCHS),
           "saved_epoch": SAVED_EPOCH, "true_best_epoch": TRUE_BEST,
           "val_nse_median_best_model": float(CURVE.set_index("epoch").loc[SAVED_EPOCH, "NSE trung vị validation (tệp)"])
           if SAVED_EPOCH else None,
           "test_nse_median_best_model": float(best_row["NSE trung vị test"]),
           "test_kge_median_best_model": float(best_row["KGE trung vị test"]),
           "author_test_nse_median": AUTHOR_TEST_NSE_MEDIAN, "train_hours": TRAIN_HOURS,
           "test_eval_minutes_median": TEST_EVAL_MIN, "test_build_minutes": BUILD_MIN,
           "library_commit": NH_COMMIT}
(OUT_DIR / f"summary_{EXP_NAME}.json").write_text(json.dumps(SUMMARY, indent=2))

checks = [("best_model.pt trùng tensor với đúng 1 epoch", match, "1 epoch", len(match) == 1),
          ("Epoch best_model.pt = epoch Phần 5 ghi sau cùng = epoch NSE validation cao nhất (tệp)",
           f"{SAVED_EPOCH} / {BEST_EPOCH} / {TRUE_BEST}", "bằng nhau", SAVED_EPOCH == BEST_EPOCH == TRUE_BEST),
          ("NSE, KGE validation tệp khớp log (từng epoch)", f"{int(CURVE['khớp log'].sum())} / {len(CURVE)}",
           f"sai lệch <= {LOG_TOL}", bool(CURVE["khớp log"].all())),
          ("Validation đủ 451 lưu vực mỗi epoch", sorted(CURVE["số lưu vực validation"].unique().tolist()), "[451]",
           CURVE["số lưu vực validation"].eq(451).all()),
          ("Có test_metrics.csv cho mọi epoch + best", f"{TEST_EPOCHS['NSE trung vị test'].notna().sum()} + 1",
           f"{len(EPOCHS)} + 1", TEST_EPOCHS["NSE trung vị test"].notna().sum() == len(EPOCHS)),
          ("NSE trung vị test: kết quả trả về khớp tệp ghi", f"{(TEST_ALL['NSE trung vị test'] - TEST_ALL['_nse_file']).abs().max():.1e}",
           f"<= {SAME_TOL}", (TEST_ALL["NSE trung vị test"] - TEST_ALL["_nse_file"]).abs().max() <= SAME_TOL),
          ("Test best_model.pt = test epoch cùng trọng số (NSE trung vị)",
           f"{abs(best_row['NSE trung vị test'] - ep_saved['NSE trung vị test']):.1e}" if ep_saved is not None else "-",
           f"<= {SAME_TOL}", ep_saved is not None and abs(best_row["NSE trung vị test"] - ep_saved["NSE trung vị test"]) <= SAME_TOL),
          ("Có cache / không cache: lệch NSE lớn nhất từng lưu vực (best_model.pt)", f"{diff_fresh:.1e}, NaN trùng: {nan_same}",
           f"<= {SAME_TOL}, NaN trùng", diff_fresh <= SAME_TOL and nan_same),
          ("Lưu vực có NSE test (best_model.pt)", int(best_row["lưu vực có NSE test"]), "450 (lưu vực 55 không có qmean)",
           best_row["lưu vực có NSE test"] == 450),
          ("Chỉ còn test_results.p của best_model.pt", len(list((RUN_DIR / "test").glob("*/test_results.p"))), "1",
           len(list((RUN_DIR / "test").glob("*/test_results.p"))) == 1)]
show("Kiểm", pd.DataFrame([{"mục": a, "giá trị": str(b), "cần": c, "đạt": "ĐẠT" if d else "KHÔNG"} for a, b, c, d in checks]))
print(f"Tester không cache chấm best_model.pt: {FRESH_MIN:.1f} phút")
# Không raise: huấn luyện đã xong, Phần 7-8 vẫn phải vẽ, ghi môi trường và dọn thư mục; dòng KHÔNG xem lại ở notebook so sánh
print("=> Đạt. Chạy Phần 7." if all(c[3] for c in checks) else
      "=> CHƯA đạt (dòng KHÔNG ở trên) - vẫn chạy Phần 7-8 để lưu kết quả, xem lại sau.")


# %% Phần 7 - Hình đường học theo epoch và loss theo bước
# Một hình 4 ô (chữ trong hình tiếng Anh, chú thích tiếng Việt in ngay dưới bằng caption):
#   (a) loss NSE* trung bình theo epoch của kỳ huấn luyện và validation (log Phần 5);
#   (b) NSE trung vị validation và test theo epoch (Phần 6), vạch epoch best_model.pt;
#   (c) KGE trung vị validation và test theo epoch (thư viện tính cùng NSE, metrics: [NSE, KGE]);
#   (d) loss huấn luyện theo bước đọc từ tệp sự kiện tensorboard trong thư mục run (thẻ train/total_loss, ghi mỗi
#       log_interval bước - logger.py dòng 105-127) kèm trung bình trượt, trục ngang đổi ra epoch.
# Kỳ validation, test lấy từ config; lưu b1b_output/Figures/B1b_Curve_<tên run>.png, tải về đặt ở
# Workspace/05_Training/05_B1b_LamaHCEExt_Train/Figures/.
# Cần chạy Phần 1-6 trước trong phiên (dùng CURVE, SAVED_EPOCH, RUN_DIR, cfg_saved, period_label, EXP_NAME, RUN_SEED,
# OUT_DIR, basins).
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

STEP_TAG = "train/total_loss"


def caption(text):
    # Chú thích tiếng Việt in ngay dưới hình (chữ trong hình tiếng Anh).
    print(f"Hình: {text}")


def vn(x, digits=3):
    # Số thập phân trong chú thích tiếng Việt dùng dấu phẩy.
    return f"{x:.{digits}f}".replace(".", ",")


# Loss theo bước từ tensorboard
ea = EventAccumulator(str(RUN_DIR), size_guidance={"scalars": 0})
ea.Reload()
steps = pd.DataFrame([(ev.step, ev.value) for ev in ea.Scalars(STEP_TAG)], columns=["step", "loss"]) \
    if STEP_TAG in ea.Tags()["scalars"] else pd.DataFrame(columns=["step", "loss"])
n_epochs = int(CURVE["epoch"].max())
steps_per_epoch = steps["step"].max() / n_epochs if len(steps) else np.nan
steps["epoch"] = steps["step"] / steps_per_epoch
window = int(min(200, max(5, len(steps) // 20)))           # điểm ghi (mỗi log_interval bước)
steps["trung bình trượt"] = steps["loss"].rolling(window, center=True, min_periods=1).mean()
steps.to_csv(OUT_DIR / f"steps_{EXP_NAME}.csv", index=False)

lab_val, lab_test = period_label(cfg_saved, "validation"), period_label(cfg_saved, "test")
FIG_DIR = OUT_DIR / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0, 0]
ax.plot(CURVE["epoch"], CURVE["loss huấn luyện"], "o-", ms=3, label="train")
ax.plot(CURVE["epoch"], CURVE["loss validation"], "s-", ms=3, label=lab_val)
ax.set(xlabel="epoch", ylabel="mean NSE* loss", title="(a) Loss per epoch")
for ax, metric, title in [(axes[0, 1], "NSE", "(b) Median NSE per epoch"), (axes[1, 0], "KGE", "(c) Median KGE per epoch")]:
    ax.plot(CURVE["epoch"], CURVE[f"{metric} trung vị validation (tệp)"], "o-", ms=3, color="C2", label=lab_val)
    ax.plot(CURVE["epoch"], CURVE[f"{metric} trung vị test"], "s-", ms=3, color="C3", label=lab_test)
    ax.set(xlabel="epoch", ylabel=f"median {metric} of {len(basins)} basins", title=title)
ax = axes[1, 1]
ax.plot(steps["epoch"], steps["loss"], lw=0.4, alpha=0.35, color="C0", label=f"every {cfg_saved.log_interval} steps")
ax.plot(steps["epoch"], steps["trung bình trượt"], lw=1.5, color="C1", label=f"moving average ({window} points)")
ax.set(xlabel="epoch", ylabel="NSE* loss per batch", title="(d) Training loss per step")
for ax in axes.flat:
    if SAVED_EPOCH is not None and ax is not axes[1, 1]:
        ax.axvline(SAVED_EPOCH, color="grey", ls="--", lw=1, label=f"best_model.pt (epoch {SAVED_EPOCH})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
fig.suptitle(f"B1b Sequential Forecast LSTM, qmean + qmax in mm/day, seed {RUN_SEED}: learning curves ({n_epochs} epochs)")
fig.tight_layout()
fig_path = FIG_DIR / f"B1b_Curve_{EXP_NAME}.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
best = CURVE.set_index("epoch").loc[SAVED_EPOCH] if SAVED_EPOCH is not None else None
caption(f"đường học của hạt giống {RUN_SEED} qua {n_epochs} epoch. (a) loss NSE* trung bình mỗi epoch của kỳ huấn luyện "
        f"và kỳ validation; loss huấn luyện giảm trong khi loss validation tăng là dấu hiệu quá khớp. (b), (c) NSE và KGE "
        f"trung vị {len(basins)} lưu vực của kỳ validation (dùng để chọn mô hình) và kỳ test (chỉ để theo dõi); vạch xám là "
        f"epoch lưu thành best_model.pt"
        + (f" (epoch {SAVED_EPOCH}: NSE validation {vn(best['NSE trung vị validation (tệp)'])}, "
           f"NSE test {vn(best['NSE trung vị test'])})" if best is not None else "")
        + f". (d) loss của từng lô {cfg_saved.batch_size} mẫu, ghi mỗi {cfg_saved.log_interval} bước ({len(steps)} điểm), đường cam là "
        f"trung bình trượt {window} điểm; trục ngang đổi ra epoch ({steps_per_epoch:.0f} bước/epoch).")
print(f"Đã lưu {fig_path}")
print("=> Đạt. Chạy Phần 8." if len(steps) else "=> CHƯA có loss theo bước (không đọc được tệp tensorboard) - xem lại.")


# %% Phần 8 - Ghi môi trường, danh mục tệp, dọn thư mục dữ liệu tạm
# Output chính thức của notebook (Kaggle giữ cả /kaggle/working khi Save Version):
#   b1b_runs/<run>/ - config.yml, train_data/train_data_scaler.yml, model_epochNNN.pt và optimizer_state_epochNNN.pt mọi
#                    epoch, best_model.pt, validation/model_epochNNN/ (chỉ số, dự báo), test/model_epochNNN/test_metrics.csv
#                    mọi epoch, test/best_model/ (test_metrics.csv + test_results.p), tệp sự kiện tensorboard, output.log;
#   b1b_output/     - log, bảng epoch, đường học, loss theo bước, bộ nhớ GPU, tóm tắt, hình (Phần 5-7), environment.json,
#                    manifest.csv.
# Như A4: không xuất NPZ ở đây (hợp đồng NPZ cần ngữ cảnh A3 trùng từng bit), notebook so sánh dựng NPZ của 3 hạt giống
# từ test/best_model/test_results.p. Xóa thư mục dữ liệu dựng ở Phần 3 (1,53 GB, dựng lại được).
# Cần chạy Phần 1-7 trước trong phiên (dùng RUN_DIR, RUN_SEED, OUT_DIR, WORK_DIR, Q_DIR, SUMMARY, cfg_saved, NH_COMMIT, show).
import hashlib
from importlib import metadata

ENV_LIBS = ["neuralhydrology", "torch", "numpy", "pandas", "xarray", "scipy", "numba", "ruamel.yaml", "tensorboard"]
KEY_FILES = ["config.yml", "train_data/train_data_scaler.yml", "best_model.pt", "test/best_model/test_metrics.csv",
             "test/best_model/test_results.p", "output.log"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# (1) Môi trường
try:
    nh_commit = json.loads(metadata.distribution("neuralhydrology").read_text("direct_url.json"))["vcs_info"]["commit_id"]
except Exception:  # cài không qua git => không có direct_url.json
    nh_commit = "không rõ"
env = {"seed": RUN_SEED, "experiment_name": EXP_NAME, "run_dir": RUN_DIR.relative_to(WORK_DIR).as_posix(),
       "python": sys.version.split()[0], "platform": platform.platform(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "neuralhydrology_commit": nh_commit, "neuralhydrology_commit_expected": NH_COMMIT,
       "config_sha256": sha256(RUN_DIR / "config.yml"),
       "protocol": {"epochs": cfg_saved.epochs, "early_stopping_patience": cfg_saved.early_stopping_patience,
                    "data_on_gpu": cfg_saved.data_on_gpu, "num_workers": cfg_saved.num_workers,
                    "test_every_epoch": True, "deterministic": False},
       "libs": {lib: metadata.version(lib) for lib in ENV_LIBS}, "summary": SUMMARY,
       "q_inputs": Q_USE,
       "q_dir_note": "LamaH_q_mmd_input dựng bằng Phần 3 (qmean_mmd, qmax_mmd mm/ngày theo area_gov từ D_gauges, số âm => trống), đã xóa khỏi Output"}
(OUT_DIR / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

# (2) Danh mục tệp chính kèm sha256 (notebook so sánh kiểm lại khi đọc)
key_paths = [RUN_DIR / rel for rel in KEY_FILES] + sorted(RUN_DIR.glob("events.out.tfevents.*"))
man = [{"tệp": p.relative_to(WORK_DIR).as_posix(), "KB": round(p.stat().st_size / 1e3, 1), "sha256": sha256(p)}
       for p in key_paths if p.exists()]
missing = [p.relative_to(RUN_DIR).as_posix() for p in key_paths if not p.exists()]
n_events = len(key_paths) - len(KEY_FILES)
MANIFEST = pd.DataFrame(man)
MANIFEST.to_csv(OUT_DIR / "manifest.csv", index=False)
show("Tệp chính của run (manifest.csv)", MANIFEST.assign(sha256=MANIFEST["sha256"].str[:12] + "..."))

# (3) Dọn thư mục dữ liệu tạm, đo dung lượng Output
if Q_DIR.exists():
    shutil.rmtree(Q_DIR)  # chỉ xóa tệp đã dựng và liên kết; rmtree không đi theo liên kết tới dữ liệu gốc


def du_gb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file() and not f.is_symlink()) / 1e9


sizes = [{"thư mục": d.name, "GB": round(du_gb(d), 3)} for d in sorted(WORK_DIR.iterdir()) if d.is_dir()]
show("Dung lượng /kaggle/working (Output của bản lưu)", pd.DataFrame(sizes))

n_test = len(list((RUN_DIR / "test").glob("model_epoch*/test_metrics.csv")))
checks = [("Đủ tệp chính trong thư mục run", f"{len(man)} / {len(key_paths)}", "đủ", not missing),
          ("Có tệp sự kiện tensorboard", n_events, ">= 1", n_events >= 1),
          ("test_metrics.csv theo epoch", n_test, cfg_saved.epochs, n_test == cfg_saved.epochs),
          (f"Commit thư viện đúng {NH_COMMIT[:7]}", nh_commit[:7], NH_COMMIT[:7], nh_commit == NH_COMMIT),
          ("Đã xóa thư mục dữ liệu tạm", not Q_DIR.exists(), "True", not Q_DIR.exists()),
          ("Tổng Output", f"{sum(r['GB'] for r in sizes):.2f} GB", "<= 20 GB", sum(r["GB"] for r in sizes) <= 20)]
show("Kiểm", pd.DataFrame([{"mục": a, "giá trị": str(b), "cần": str(c), "đạt": "ĐẠT" if d else "KHÔNG"}
                           for a, b, c, d in checks]))
if missing:
    print("Thiếu:", missing)
print("=> Đạt. Xong notebook." if all(c[3] for c in checks) else "=> CHƯA đạt (dòng KHÔNG ở trên): gửi output này.")
