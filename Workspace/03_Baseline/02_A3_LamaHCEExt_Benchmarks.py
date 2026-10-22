# =============================================================================
# A3 — Các mốc so sánh trên Extended LamaH-CE
# Notebook Kaggle và thuyết minh cùng tên; chạy các mục từ trên xuống.
# MỤC LỤC
#   Phần 1.1 — Thư viện, hằng số và môi trường
#   Phần 1.2 — Tìm các nguồn đã gắn vào notebook
#   Phần 1.3 — Kiểm kê export A2
#   Phần 1.4 — Kiểm kê các bảng A1
#   Phần 1.5 — Kiểm kê dữ liệu gốc và 451 ID
#   Phần 1.6 — Kết luận kiểm kê Input
#   Phần 2.1 — Công cụ đọc cấu hình và kiểm nguồn
#   Phần 2.2 — Cố định đường dẫn và metadata A2
#   Phần 2.3 — Hash các tệp đầu vào
#   Phần 2.4 — Kiểm cấu hình sáu run A2
#   Phần 2.5 — Lưu kiểm nguồn và danh mục A1
#   Phần 3.1 — Lịch test và hàm đọc dự báo ngày
#   Phần 3.2 — Đọc và kiểm sáu pickle A2
#   Phần 3.3 — Độ phủ, dự báo hằng và dự báo thiếu
#   Phần 3.4 — Đối chiếu obs giữa các run
#   Phần 3.5 — Hồ sơ A1 và lưu kiểm schema
#   Phần 3.6 — Diện tích trạm và quy đổi đơn vị
#   Phần 3.7 — Đọc cửa sổ Q nguồn và kiểm giá trị thiếu
#   Phần 3.8 — Đối chiếu qmax nguồn với obs A2
#   Phần 3.9 — Lưu kết quả đối chiếu Q
#   Phần 4.1 — Hàm lag theo ngày và tóm tắt mặt nạ
#   Phần 4.2 — Ví dụ kiểm lag qua ngày thiếu
#   Phần 4.3 — Dựng hai chuỗi persistence
#   Phần 4.4 — Dựng các mặt nạ chấm điểm
#   Phần 4.5 — Kiểm ngày đầu test và các hồ sơ thiếu
#   Phần 4.6 — Lưu persistence và quy ước đánh giá
#   Phần 5.1 — Hàm NSE, PNSE, KGE và các thành phần
#   Phần 5.2 — Công cụ kiểm ví dụ số
#   Phần 5.3 — Ví dụ đúng trước và các ca suy biến
#   Phần 5.4 — Kiểm đầu vào và mặt nạ không hợp lệ
#   Phần 5.5 — Đối chiếu hàm tham chiếu
#   Phần 5.6 — Bảng kiểm và trạng thái hàm chỉ số
#   Phần 6.1 — Hợp đồng chấm điểm và dung sai CSV
#   Phần 6.2 — Chấm sáu A2 và hai persistence
#   Phần 6.3 — Đối chiếu NSE và KGE native với CSV
#   Phần 6.4 — Tổng hợp điểm, kỹ năng và NaN
#   Phần 6.5 — Lưu bảng điểm A2 và metadata đánh giá
#   Phần 6.6 — Hàm tham chiếu từ commit A2
#   Phần 6.7 — Môi trường replay và các ca đối chứng
#   Phần 6.8 — Replay KGE với kiểu gốc và float64
#   Phần 6.9 — Kết luận kiểm sai khác số học
#   Phần 7.1 — Lịch DLinear và hàm kiểm cửa sổ
#   Phần 7.2 — Đọc và kiểm Q train, validation, test
#   Phần 7.3 — Đếm lịch sử đầy đủ và mẫu từng kỳ
#   Phần 7.4 — Thống kê train và các hồ sơ A1
#   Phần 7.5 — Lưu dữ liệu và kiểm độ phủ DLinear
#   Phần 7.6 — Hàm scaler, đổi ngược và loss tham chiếu
#   Phần 7.7 — Fit scaler chung và chuẩn hóa qmean/qmax
#   Phần 7.8 — Std train từng ID và trọng số loss
#   Phần 7.9 — Chỉ mục mẫu và ngày suy luận
#   Phần 7.10 — Kiểm tiền xử lý và loss
#   Phần 7.11 — Cấu hình DLinear cố định
#   Phần 7.12 — Lưu scaler, dữ liệu chuẩn bị và cấu hình
#   Phần 8.1 — Thiết bị và chế độ tính xác định
#   Phần 8.2 — Kiến trúc DLinear và loss PyTorch
#   Phần 8.3 — Batch, huấn luyện và dự báo
#   Phần 8.4 — Kiểm kiến trúc và gradient loss
#   Phần 8.5 — Kiểm chỉ mục và gradient trên thiết bị
#   Phần 8.6 — Chạy thử một epoch và checkpoint
#   Phần 8.7 — Báo cáo kiểm triển khai và thời gian
#   Phần 9.1 — Lưu checkpoint và quy tắc dừng sớm
#   Phần 9.2 — Hàm chấm, hợp đồng train và xuất dự báo
#   Phần 9.3 — Hàm chạy một seed DLinear
#   Phần 9.4 — Kiểm chọn checkpoint và patience
#   Phần 9.5 — Huấn luyện và báo cáo seed 111
#   Phần 9.6 — Huấn luyện và báo cáo seed 222
#   Phần 9.7 — Huấn luyện seed 333 và kiểm đủ ba seed
#   Phần 10.1 — Hàm đọc export và chấm cùng phạm vi
#   Phần 10.2 — Kiểm export ba seed DLinear
#   Phần 10.3 — Lập giao ngày thật của mọi mô hình
#   Phần 10.4 — Tổng hợp sáu chỉ số trên từng phạm vi
#   Phần 10.5 — Thống kê giữa ba seed
#   Phần 10.6 — Kiểm nhất quán và lưu bảng so sánh
#   Phần 10.7 — Bảng kết quả và kỹ năng hơn persistence
#   Phần 11.1 — Đọc lại hồ sơ A1 đã cố định nguồn
#   Phần 11.2 — Độ phủ train/test của các hồ sơ cố định
#   Phần 11.3 — Hàm phân rã sai số và ảnh hưởng clip
#   Phần 11.4 — NSE âm và đối chiếu dự báo tác giả
#   Phần 11.5 — Độ nhạy và lý do chỉ số không xác định
#   Phần 11.6 — Lưu và đọc các bảng chẩn đoán
#   Phần 11.7 — Chuẩn bị dữ liệu và cách hiển thị hình
#   Phần 11.8 — ECDF của sáu chỉ số
#   Phần 11.9 — NSE mô hình ghép cặp với persistence
#   Phần 11.10 — Chuỗi dự báo ở 742, 758, 363 và 800
#   Phần 11.11 — Phân bố sai số ngày theo hồ sơ
#   Phần 11.12 — Thành phần sai số chuẩn hóa
#   Phần 11.13 — Độ phủ native/common và số mẫu train
#   Phần 11.14 — Danh mục hình và tệp kết quả
# Output nằm trong a3_output; lưu bằng Save & Commit của Kaggle.
# =============================================================================

# %% Phần 1.1 — Thư viện, hằng số và môi trường
# Khai báo nguồn Input, sáu run A2 và các hàm hiển thị dùng chung.
import json
import os
import platform
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

INPUT_ROOT = Path(os.environ.get("A3_INPUT_ROOT", os.environ.get("LAMAH_ROOT", "/kaggle/input")))
EXPECTED_RUNS = ("SeqLSTM_Q", "SeqLSTM_noQ", "EncDec_Q", "EncDec_noQ", "D_FC", "D_RA")
EXPECTED_COMMIT = "dda45fcf19ee7ffd481b9d192778539df46fb074"
A1_GROUPS = {
    "Phân phối/độ phủ Q và scaler": ("A1_P2_Q_", "A1_P5_q", "A1_P5_target_"),
    "Cửa sổ và số mẫu": ("A1_P7_sample_",),
    "Nguồn NSE và liên hệ với nhãn": ("A1_P8_nse_", "A1_P8_author_nse_"),
    "Hồ sơ bất thường theo ngày": ("A1_P9_author_profile_",),
}


def show_table(title, rows):
    display(Markdown(f"**{title}**"))
    if rows:
        with pd.option_context("display.max_colwidth", None, "display.max_rows", 40):
            display(pd.DataFrame(rows))
    else:
        print("Không tìm thấy trong Input đang gắn.")


def file_ok(path):
    return path.is_file() and path.stat().st_size > 0



print(f"Python: {platform.python_version()} | pandas: {pd.__version__} | Input: {INPUT_ROOT}")
if not INPUT_ROOT.is_dir():
    raise FileNotFoundError("Chưa thấy Input. Trên Kaggle hãy Add Input hai notebook A1/A2 và dataset lamah-ce-ext.")
show_table('Các mục ngay dưới Input', [
    {"tên": p.name, "loại": "thư mục" if p.is_dir() else "tệp"}
    for p in sorted(INPUT_ROOT.iterdir())
])

# %% Phần 1.2 — Tìm các nguồn đã gắn vào notebook
# Quét Input để nhận diện output A1/A2, dữ liệu LamaH và danh sách ID.
a2_exports, a2_copies, a1_csvs, raw_roots, basin_files, walk_errors = [], [], [], [], [], []
for directory, subdirs, filenames in os.walk(INPUT_ROOT, onerror=lambda exc: walk_errors.append(str(exc))):
    base = Path(directory)
    for name in list(subdirs):
        path = base / name
        if name == "a2_output":
            a2_exports.append(path)
            subdirs.remove(name)
        elif name == "a2_runs":
            a2_copies.append(path)
            subdirs.remove(name)
        elif name == "LamaH_extended":
            raw_roots.append(path)
            subdirs.remove(name)
        elif name == "Experiments":
            if (path / "basins_filtered.txt").is_file():
                basin_files.append(path / "basins_filtered.txt")
            subdirs.remove(name)
        elif name in {"LamaH_expanded_q_input", "Figures", "__pycache__", ".git"}:
            subdirs.remove(name)
    a1_csvs.extend(base / name for name in filenames if name.startswith("A1_") and name.endswith(".csv"))
    if "basins_filtered.txt" in filenames:
        basin_files.append(base / "basins_filtered.txt")

# %% Phần 1.3 — Kiểm kê export A2
# Kiểm metadata, các tệp chính và CSV metrics đã lưu của sáu run.
show_table('Các export A2 tìm được', [{"a2_output": str(p)} for p in sorted(a2_exports)])
if a2_copies:
    print("Có thêm bản sao a2_runs (không tự dùng thay export):", *map(str, sorted(a2_copies)), sep="\n  ")
a2_checks = []
for export in sorted(a2_exports):
    env_path = export / "environment.json"
    try:
        environment = json.loads(env_path.read_text(encoding="utf-8-sig"))
        if not isinstance(environment, dict):
            raise ValueError("environment.json không phải object")
        env_error = ""
    except (OSError, ValueError) as exc:
        environment, env_error = {}, str(exc)
    commit = environment.get("neuralhydrology_commit")
    print("\nExport:", export)
    print("repro_ok:", environment.get("repro_ok"), "| NH commit:", commit)
    print("Lỗi metadata:", env_error or "không")
    common = []
    for name in ("environment.json", "eval_summary.csv", "compare_per_basin.csv"):
        path = export / name
        common.append({"tệp": name, "có và khác rỗng": file_ok(path)})
    show_table('Tệp chung A2', common)
    run_rows = []
    for run in EXPECTED_RUNS:
        folder = export / "runs" / run
        results_path = folder / "test_results.p"
        metrics_path = folder / "test_metrics.csv"
        row = {"run": run, "test_results.p": file_ok(results_path),
               "MB pickle": round(results_path.stat().st_size / 1e6, 2) if results_path.is_file() else None,
               "test_metrics.csv": file_ok(metrics_path), "config.yml": file_ok(folder / "config.yml")}
        row["số dòng metrics"], row["NSE hữu hạn"], row["NSE trung vị"] = None, None, None
        row["lỗi đọc CSV"] = ""
        if row["test_metrics.csv"]:
            try:
                metrics = pd.read_csv(metrics_path, dtype=str)
                row["số dòng metrics"] = len(metrics)
                candidates = [c for c in metrics.columns if c.upper() == "NSE" or c.upper().endswith("_NSE")]
                if len(candidates) == 1:
                    scores = pd.to_numeric(metrics[candidates[0]], errors="coerce")
                    scores = scores[scores.notna() & ~scores.isin([float("inf"), -float("inf")])]
                    row["NSE hữu hạn"], row["NSE trung vị"] = len(scores), scores.median() if len(scores) else None
                else:
                    row["lỗi đọc CSV"] = f"Cần kiểm tên cột NSE: {list(metrics.columns)}"
            except Exception as exc:
                row["lỗi đọc CSV"] = f"{type(exc).__name__}: {exc}"
        run_rows.append(row)
    show_table('Sáu run A2 — chỉ kiểm tệp và đọc CSV đã lưu', run_rows)
    required_files_ok = all(r["test_results.p"] and r["test_metrics.csv"] and r["config.yml"]
                            and not r["lỗi đọc CSV"] and (r["số dòng metrics"] or 0) > 0 for r in run_rows)
    a2_checks.append(required_files_ok and environment.get("repro_ok") is True and commit == EXPECTED_COMMIT)

# %% Phần 1.4 — Kiểm kê các bảng A1
# Liệt kê các nhóm bảng có thể tái sử dụng; chưa thay thế việc kiểm nội dung.
a1_rows = []
for group, prefixes in A1_GROUPS.items():
    matches = sorted(p for p in a1_csvs if p.name.startswith(prefixes) and file_ok(p))
    a1_rows.append({"nhóm": group, "số CSV khác rỗng": len(matches),
                    "thư mục": "\n".join(sorted({str(p.parent) for p in matches})),
                    "ví dụ (tối đa 4)": "\n".join(p.name for p in matches[:4])})
show_table('Bảng A1 có thể dùng lại', a1_rows)
print("Tổng CSV tên A1_ tìm được:", len(a1_csvs), "| Có đủ từng nhóm không đồng nghĩa đã kiểm nội dung bảng.")

# %% Phần 1.5 — Kiểm kê dữ liệu gốc và 451 ID
# Kiểm tệp Q, khí tượng và tính duy nhất của danh sách lưu vực.
raw_rows = []
for root in sorted(raw_roots):
    gauges = root / "D_gauges" / "2_timeseries" / "daily"
    forcing = root / "A_basins_total_upstrm" / "2_timeseries" / "daily"
    raw_rows.append({"nguồn gốc": str(root), "Q ngày (ID_*.csv)": len(list(gauges.glob("ID_*.csv"))),
                     "khí tượng ngày (ID_*.csv)": len(list(forcing.glob("ID_*.csv")))})
show_table('Dữ liệu gốc đã giải nén', raw_rows)
id_rows = []
for path in sorted(set(basin_files)):
    try:
        ids = path.read_text(encoding="utf-8-sig").split()
        canonical_ids = [str(int(value)) for value in ids]
        id_rows.append({"danh sách lưu vực": str(path), "số dòng/ID": len(ids),
                        "ID phân biệt": len(set(canonical_ids)), "đúng 451 không trùng": len(ids) == len(set(canonical_ids)) == 451})
    except (OSError, ValueError) as exc:
        id_rows.append({"danh sách lưu vực": str(path), "lỗi": str(exc)})
show_table('Danh sách ID của bài', id_rows)

# %% Phần 1.6 — Kết luận kiểm kê Input
# Tóm tắt nguồn hiện có và các lỗi quét trước khi cố định đường dẫn.
display(Markdown('**Kết luận kiểm kê và output cần gửi**'))
if len(a2_exports) == 1 and a2_checks[0]:
    print("A2: ĐỦ TỆP CHÍNH để chuyển sang kiểm nội dung pickle/ngày/đơn vị ở cell sau.")
elif len(a2_exports) > 1:
    print("A2: Có nhiều export; cần chọn rõ phiên bản trước khi đọc dự báo.")
else:
    print("A2: CẦN KIỂM các tệp/metadata thiếu hoặc sai ở mục 1.2; chưa kết luận đủ đầu vào.")
print("A1:", "đã thấy CSV xuất từ A1." if a1_csvs else "chưa thấy CSV A1; kiểm lại output notebook đã thêm.")
print("Dữ liệu gốc:", "đã thấy Q ngày, khí tượng và danh sách 451 ID." if
      any(r["Q ngày (ID_*.csv)"] > 0 and r["khí tượng ngày (ID_*.csv)"] > 0 for r in raw_rows)
      and any(r.get("đúng 451 không trùng") for r in id_rows) else "cần kiểm lại mục 1.4.")
if walk_errors:
    print("Lỗi khi quét Input (không coi vùng chưa đọc được là không có tệp):", *walk_errors, sep="\n")
print("Chưa mở pickle hoặc xác nhận obs/sim; NSE ở đây chỉ đọc từ CSV A2, chưa phải kết quả A3.")

# %% Phần 2.1 — Công cụ đọc cấu hình và kiểm nguồn
# Định nghĩa kỳ dữ liệu, ánh xạ run và các hàm ngày/ID/hash.
import hashlib
from datetime import datetime, timezone

try:
    from ruamel.yaml import YAML
    a3_yaml_load = YAML(typ="safe").load
except ImportError:
    try:
        import yaml
        a3_yaml_load = yaml.safe_load
    except ImportError as exc:
        raise ImportError("Chưa có bộ đọc YAML an toàn. Gửi traceback để xác định gói cần cài; chưa đọc config.") from exc

if not all(name in globals() for name in ("a2_exports", "a1_csvs", "raw_roots", "basin_files", "walk_errors")):
    raise RuntimeError("Cần chạy Phần 1 trong cùng phiên trước Phần 2.")
if walk_errors:
    raise RuntimeError("Phần 1 có vùng Input chưa đọc được; cần kiểm lỗi quét trước khi cố định nguồn.")

A3_AUTHOR_RUNS = {
    "SeqLSTM_Q": "Sequential_Forecast_LSTM/Sequential_Forecast_LSTM_with_q/Sequential_Forecast_LSTM_RA_with_q_Simple_Embedding",
    "SeqLSTM_noQ": "Sequential_Forecast_LSTM/Sequential_Forecast_LSTM_without_q/Sequential_Forecast_LSTM_RA_without_q_Simple_Embedding",
    "EncDec_Q": "Encoder_Decoder_LSTM/Encoder_Decoder_LSTM_with_q/Encoder_Decoder_LSTM_with_q_Complex_Embedding",
    "EncDec_noQ": "Encoder_Decoder_LSTM/Encoder_Decoder_LSTM_without_q/Encoder_Decoder_LSTM_RA_without_q_Complex_Embedding",
    "D_FC": "Baseline/Forecast",
    "D_RA": "Baseline/Reanalysis/Reanalysis_Simple_Embedding",
}
A3_PERIODS = {"train": ("2003-01-01", "2009-12-31"),
              "validation": ("2010-01-01", "2013-12-31"), "test": ("2014-01-01", "2017-12-31")}
A3_PRIORITY_A1 = {
    "A1_P2_Q_coverage.csv", "A1_P2_Q_missing_by_basin.csv", "A1_P7_sample_final_counts.csv",
    "A1_P7_sample_final_scope.csv", "A1_P8_nse_values.csv", "A1_P8_nse_file_audit.csv",
    "A1_P9_author_profile_files.csv", "A1_P9_author_profile_scores.csv",
    "A1_P9_author_profile_daily.csv", "A1_P9_author_profile_variation.csv",
}


def a3_unique_path(paths, label):
    choices = sorted(set(paths))
    if len(choices) != 1:
        raise ValueError(f"{label}: cần đúng một nguồn, đang có {len(choices)}: {choices}")
    return choices[0]


def a3_hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def a3_names(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [name for item in value for name in a3_names(item)]
    if isinstance(value, dict):
        return [name for item in value.values() for name in a3_names(item)]
    raise ValueError(f"Cấu trúc tên biến chưa hỗ trợ: {type(value).__name__}")


def a3_date(value):
    if value is None or isinstance(value, (list, dict)):
        raise ValueError(f"Cần một mốc ngày rõ ràng, nhận được {value!r}")
    date = pd.to_datetime(value, format="%d/%m/%Y") if isinstance(value, str) and "/" in value else pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("Mốc ngày là NaT")
    return date.strftime("%Y-%m-%d")

# %% Phần 2.2 — Cố định đường dẫn và metadata A2
# Chọn duy nhất mỗi nguồn và tạo thư mục a3_output trên Working.
A3_A2_DIR = a3_unique_path(a2_exports, "Export chạy lại A2")
A3_LAMAH_DIR = a3_unique_path(raw_roots, "Dữ liệu LamaH_extended")
A3_BASIN_FILE = a3_unique_path(basin_files, "Danh sách 451 ID")
A3_A1_DIR = a3_unique_path([p.parent for p in a1_csvs], "Thư mục CSV A1")
A3_WORK_DIR = Path(os.environ.get("A3_WORK_DIR", "/kaggle/working" if Path("/kaggle/working").is_dir() else "."))
A3_OUT_DIR = A3_WORK_DIR / "a3_output"
A3_OUT_DIR.mkdir(parents=True, exist_ok=True)


show_table('Các nguồn đã chọn', [{"nguồn": label, "đường dẫn": str(path)} for label, path in (
    ("A2 chạy lại", A3_A2_DIR), ("A1 Tables", A3_A1_DIR),
    ("LamaH_extended", A3_LAMAH_DIR), ("Danh sách ID", A3_BASIN_FILE), ("Output A3", A3_OUT_DIR))])

a3_env = json.loads((A3_A2_DIR / "environment.json").read_text(encoding="utf-8-sig"))
a3_mapping = a3_env.get("runs", {})
a3_metadata_ok = (a3_env.get("repro_ok") is True and a3_env.get("neuralhydrology_commit") == EXPECTED_COMMIT
                  and isinstance(a3_mapping, dict) and a3_mapping == A3_AUTHOR_RUNS)
print("Metadata A2 và ánh xạ đúng sáu biến thể:", a3_metadata_ok)

# %% Phần 2.3 — Hash các tệp đầu vào
# Lập manifest cho export A2, danh sách ID và các CSV A1 ưu tiên.
a3_manifest_rows, a3_config_rows, A3_CONFIGS = [], [], {}
a3_sources = [("A2_rerun_export", "", A3_A2_DIR / name) for name in (
    "environment.json", "eval_summary.csv", "compare_per_basin.csv")]
a3_sources += [("A2_rerun_export", run, A3_A2_DIR / "runs" / run / name)
               for run in EXPECTED_RUNS for name in ("test_results.p", "test_metrics.csv", "config.yml")]
a3_sources += [("dataset_basin_list", "", A3_BASIN_FILE)]
a3_sources += [("A1_saved_output", "", p) for p in sorted(a1_csvs) if p.name in A3_PRIORITY_A1]
for source, run, path in a3_sources:
    row = {"source": source, "run": run, "file": path.name, "path": str(path), "status": "missing_or_empty",
           "bytes": 0, "sha256": ""}
    try:
        if file_ok(path):
            row.update(status="hashed", bytes=path.stat().st_size, sha256=a3_hash_file(path))
    except OSError as exc:
        row["status"] = f"read_error: {exc}"
    a3_manifest_rows.append(row)

# %% Phần 2.4 — Kiểm cấu hình sáu run A2
# Đối chiếu biến mục tiêu, đầu vào qmean, cửa sổ, kỳ dữ liệu và chính sách clip.
for run in EXPECTED_RUNS:
    row = {"run": run, "model": "", "seed": None, "target": "", "seq_length": None,
           "predict_last_n": None, "has_qmean": None, "clip_qmax_declared": None, "issues": ""}
    issues = []
    try:
        cfg = a3_yaml_load((A3_A2_DIR / "runs" / run / "config.yml").read_text(encoding="utf-8-sig"))
        if not isinstance(cfg, dict):
            raise ValueError("Config không phải mapping")
        A3_CONFIGS[run] = cfg
        targets = a3_names(cfg.get("target_variables", cfg.get("target_variable")))
        inputs = a3_names([cfg.get(key) for key in ("dynamic_inputs", "hindcast_inputs", "forecast_inputs")])
        row.update(model=cfg.get("model"), seed=cfg.get("seed"), target=", ".join(targets),
                   seq_length=cfg.get("seq_length"), predict_last_n=cfg.get("predict_last_n"),
                   has_qmean="qmean" in inputs, clip_qmax_declared="qmax" in a3_names(cfg.get("clip_targets_to_zero")))
        if targets != ["qmax"]:
            issues.append("target_not_single_qmax")
        if row["has_qmean"] != run.endswith("_Q"):
            issues.append("qmean_variant_mismatch")
        if cfg.get("predict_last_n") != 1:
            issues.append("predict_last_n_not_1")
        for period, expected_dates in A3_PERIODS.items():
            dates = tuple(a3_date(cfg.get(f"{period}_{boundary}_date")) for boundary in ("start", "end"))
            row[period] = " → ".join(dates)
            if dates != expected_dates:
                issues.append(f"{period}_dates_mismatch")
        row["saved_data_dir"] = str(cfg.get("data_dir", ""))
        row["saved_test_basin_file"] = str(cfg.get("test_basin_file", ""))
    except Exception as exc:
        issues.append(f"{type(exc).__name__}: {exc}")
    row["issues"] = " | ".join(issues)
    a3_config_rows.append(row)

A3_INPUT_MANIFEST = pd.DataFrame(a3_manifest_rows)
A3_CONFIG_AUDIT = pd.DataFrame(a3_config_rows)
show_table('Cấu hình sáu run — chỉ kiểm khai báo', [
    {key: value for key, value in row.items() if not key.startswith("saved_")} for row in a3_config_rows])
print("data_dir/test_basin_file trong config là đường dẫn của phiên A2 cũ; không yêu cầu còn tồn tại trong phiên mới.")
print("clip_qmax_declared chỉ là chính sách khai báo; chưa kiểm dự báo đã chặn hay chưa và chưa áp chặn lại.")
show_table('Manifest — hash toàn bộ tệp A2, danh sách ID và các CSV A1 ưu tiên', [
    {key: row[key] for key in ("source", "run", "file", "bytes", "status", "sha256")} for row in a3_manifest_rows])

# %% Phần 2.5 — Lưu kiểm nguồn và danh mục A1
# Lưu các bảng nhỏ và xác nhận điều kiện đọc dự báo.
A3_A1_INVENTORY = pd.DataFrame([{"file": p.name, "path": str(p), "bytes": p.stat().st_size,
                                "priority_for_A3": p.name in A3_PRIORITY_A1} for p in sorted(a1_csvs)])
show_table('CSV A1 ưu tiên và phần còn thiếu', [
    {"file": name, "có trong Input": any(p.name == name and file_ok(p) for p in a1_csvs)}
    for name in sorted(A3_PRIORITY_A1)])

A3_INPUT_MANIFEST.to_csv(A3_OUT_DIR / "A3_input_manifest.csv", index=False, encoding="utf-8-sig")
A3_CONFIG_AUDIT.to_csv(A3_OUT_DIR / "A3_config_audit.csv", index=False, encoding="utf-8-sig")
A3_A1_INVENTORY.to_csv(A3_OUT_DIR / "A3_A1_inventory.csv", index=False, encoding="utf-8-sig")
(A3_OUT_DIR / "A3_source_metadata.json").write_text(json.dumps({
    "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_A2_environment": a3_env,
    "expected_periods": A3_PERIODS, "configs": A3_CONFIGS,
    "note": "Config/checkpoint origin is declared by A2; prediction pickle contents and checkpoint weights have not been inspected."
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
a3_required_hashes_ok = all(row["status"] == "hashed" for row in a3_manifest_rows
                             if row["source"] in {"A2_rerun_export", "dataset_basin_list"})
A3_SOURCE_AUDIT_OK = a3_metadata_ok and a3_required_hashes_ok and A3_CONFIG_AUDIT["issues"].eq("").all()

print("A3_SOURCE_AUDIT_OK =", bool(A3_SOURCE_AUDIT_OK))
print("Đã lưu 4 tệp audit/manifest vào:", A3_OUT_DIR)
print("Hash định danh các tệp đang dùng, không tự chứng minh chúng đúng với một bản chuẩn độc lập.")

if not A3_SOURCE_AUDIT_OK:
    raise RuntimeError("Audit nguồn/config chưa đạt; giữ các tệp audit và gửi output trước khi mở pickle.")

# %% Phần 3.1 — Lịch test và hàm đọc dự báo ngày
# Cố định 451 ID, lịch test và schema obs/sim trong pickle A2.
import gc
import pickle

import numpy as np
import xarray as xr

if not globals().get("A3_SOURCE_AUDIT_OK", False):
    raise RuntimeError("Cần chạy và đạt Phần 2 trước khi đọc pickle.")

A3_SELECTED_IDS = [str(int(value)) for value in A3_BASIN_FILE.read_text(encoding="utf-8-sig").split()]
if len(A3_SELECTED_IDS) != 451 or len(set(A3_SELECTED_IDS)) != 451:
    raise ValueError("Danh sách mô hình không còn đúng 451 ID phân biệt.")
A3_TEST_DATES = pd.date_range(*A3_PERIODS["test"], freq="D", name="date")
A3_DAILY, a3_pickle_rows, a3_schema_rows = {}, [], []
A3_SCHEMA_AUDIT_OK = False

print("Kỳ đích:", A3_TEST_DATES[0].date(), "→", A3_TEST_DATES[-1].date(), "|", len(A3_TEST_DATES), "ngày")
print("Hiện tại: NumPy", np.__version__, "| xarray", xr.__version__)
print("Môi trường sinh A2 (khai báo):", a3_env.get("libs", {}))


def a3_extract_daily(item):
    if not isinstance(item, dict) or "1D" not in item or not isinstance(item["1D"], dict):
        raise ValueError("Thiếu cấu trúc basin['1D']")
    dataset = item["1D"].get("xr")
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("basin['1D']['xr'] không phải xarray.Dataset")
    for variable in ("qmax_obs", "qmax_sim"):
        if variable not in dataset or set(dataset[variable].dims) != {"date", "time_step"}:
            raise ValueError(f"{variable}: cần đúng hai chiều date/time_step, không tự gộp chiều khác")
        if not np.issubdtype(dataset[variable].dtype, np.number):
            raise TypeError(f"{variable}: dtype không phải số")
    steps = np.asarray(dataset["time_step"].values)
    if steps.shape != (1,) or steps[0] != 0:
        raise ValueError(f"Cần đúng time_step=[0], nhận {steps!r}; chưa tự chọn hạn hoặc dịch ngày")
    dates = pd.DatetimeIndex(dataset["date"].values)
    if (dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError("Ngày phải duy nhất, tăng dần, không NaT/múi giờ và ở đầu ngày")
    frame = pd.DataFrame({
        "obs": np.asarray(dataset["qmax_obs"].transpose("date", "time_step").isel(time_step=0).values, dtype=np.float64),
        "sim_raw": np.asarray(dataset["qmax_sim"].transpose("date", "time_step").isel(time_step=0).values, dtype=np.float64),
    }, index=dates)
    schema = {"obs_dims": str(dataset["qmax_obs"].dims), "sim_dims": str(dataset["qmax_sim"].dims),
              "obs_dtype": str(dataset["qmax_obs"].dtype), "sim_dtype": str(dataset["qmax_sim"].dtype),
              "time_steps": str(steps.tolist())}
    return frame, schema

# %% Phần 3.2 — Đọc và kiểm sáu pickle A2
# Kiểm hash, khóa ID, trục ngày và biến dự báo trên toàn bộ lưu vực.
for run in EXPECTED_RUNS:
    result_path = A3_A2_DIR / "runs" / run / "test_results.p"
    file_row = {"run": run, "status": "not_read", "hash_matches_part2": False, "basin_keys": 0,
                "missing_ids": "", "extra_ids": "", "schema_ok_ids": 0, "issues": ""}
    results = None
    try:
        recorded = A3_INPUT_MANIFEST.loc[
            (A3_INPUT_MANIFEST.source == "A2_rerun_export") & (A3_INPUT_MANIFEST.run == run)
            & (A3_INPUT_MANIFEST.file == "test_results.p"), "sha256"]
        if len(recorded) != 1:
            raise ValueError("Manifest Phần 2 không định danh duy nhất pickle")
        file_row["hash_matches_part2"] = a3_hash_file(result_path) == recorded.iloc[0]
        if not file_row["hash_matches_part2"]:
            raise ValueError("Hash pickle khác Phần 2; không mở tệp đã đổi")
        with result_path.open("rb") as stream:
            results = pickle.load(stream)
        if not isinstance(results, dict):
            raise TypeError("Pickle không phải dict theo basin")
        key_map = {}
        for key in results:
            if isinstance(key, (bool, np.bool_)) or not isinstance(key, (str, int, np.integer)):
                raise ValueError(f"Kiểu ID chưa hỗ trợ: {type(key).__name__}")
            canonical = str(int(key))
            if canonical in key_map:
                raise ValueError(f"ID trùng sau chuẩn hóa: {canonical}")
            key_map[canonical] = key
        missing_ids = set(A3_SELECTED_IDS) - set(key_map)
        extra_ids = set(key_map) - set(A3_SELECTED_IDS)
        file_row.update(basin_keys=len(key_map), missing_ids=", ".join(sorted(missing_ids, key=int)),
                        extra_ids=", ".join(sorted(extra_ids, key=int)))
        obs_matrix = pd.DataFrame(np.nan, index=A3_TEST_DATES, columns=A3_SELECTED_IDS)
        sim_matrix = pd.DataFrame(np.nan, index=A3_TEST_DATES, columns=A3_SELECTED_IDS)
        for basin_id in A3_SELECTED_IDS:
            row = {"run": run, "basin_id": basin_id, "status": "not_checked", "rows": 0,
                   "missing_test_dates": len(A3_TEST_DATES), "dates_outside_test": 0,
                   "obs_finite": 0, "sim_finite": 0, "paired_finite": 0, "obs_infinite": 0, "sim_infinite": 0,
                   "obs_negative": 0, "sim_negative": 0, "obs_unique": 0, "sim_unique": 0, "issues": ""}
            try:
                if basin_id not in key_map:
                    raise KeyError("ID không có trong pickle")
                frame, schema = a3_extract_daily(results[key_map[basin_id]])
                row.update(schema, rows=len(frame), missing_test_dates=len(A3_TEST_DATES.difference(frame.index)),
                           dates_outside_test=len(frame.index.difference(A3_TEST_DATES)),
                           first_date=str(frame.index.min()), last_date=str(frame.index.max()))
                aligned = frame.reindex(A3_TEST_DATES)
                obs, sim = aligned.obs.to_numpy(), aligned.sim_raw.to_numpy()
                obs_valid, sim_valid = np.isfinite(obs), np.isfinite(sim)
                row.update(obs_finite=int(obs_valid.sum()), sim_finite=int(sim_valid.sum()),
                           paired_finite=int((obs_valid & sim_valid).sum()),
                           obs_infinite=int(np.isinf(frame.obs.to_numpy()).sum()),
                           sim_infinite=int(np.isinf(frame.sim_raw.to_numpy()).sum()),
                           obs_negative=int((obs[obs_valid] < 0).sum()), sim_negative=int((sim[sim_valid] < 0).sum()),
                           obs_unique=len(np.unique(obs[obs_valid])), sim_unique=len(np.unique(sim[sim_valid])))
                problems = []
                if row["missing_test_dates"] or row["dates_outside_test"]:
                    problems.append("calendar_mismatch")
                if row["obs_infinite"] or row["sim_infinite"]:
                    problems.append("infinite_values")
                row["issues"] = " | ".join(problems)
                row["status"] = "ok" if not problems else "needs_review"
                obs_matrix.loc[:, basin_id] = obs
                sim_matrix.loc[:, basin_id] = sim
            except Exception as exc:
                row.update(status="read_error", issues=f"{type(exc).__name__}: {exc}")
            a3_schema_rows.append(row)
        run_rows = a3_schema_rows[-len(A3_SELECTED_IDS):]
        file_row["schema_ok_ids"] = sum(row["status"] == "ok" for row in run_rows)
        file_row["status"] = "ok" if not missing_ids and not extra_ids and file_row["schema_ok_ids"] == 451 else "needs_review"
        A3_DAILY[run] = {"obs": obs_matrix, "sim_raw": sim_matrix}
    except Exception as exc:
        file_row.update(status="read_error", issues=f"{type(exc).__name__}: {exc}")
    finally:
        del results
        gc.collect()
    a3_pickle_rows.append(file_row)

A3_PICKLE_AUDIT = pd.DataFrame(a3_pickle_rows)
A3_SCHEMA_AUDIT = pd.DataFrame(a3_schema_rows)
show_table('Đọc pickle, ID và kiểm lịch/schema toàn bộ lưu vực', a3_pickle_rows)

# %% Phần 3.3 — Độ phủ, dự báo hằng và dự báo thiếu
# Đếm ngày hữu hạn, số cặp và các chuỗi cần chẩn đoán.
a3_coverage_rows = []
if not A3_SCHEMA_AUDIT.empty:
    for run, group in A3_SCHEMA_AUDIT.groupby("run", sort=False):
        a3_coverage_rows.append({"run": run, "schema_ok_ids": int(group.status.eq("ok").sum()),
            "obs_finite_days": int(group.obs_finite.sum()), "sim_finite_days": int(group.sim_finite.sum()),
            "paired_days": int(group.paired_finite.sum()),
            "sim_all_missing_IDs": ", ".join(group.loc[group.status.eq("ok") & group.sim_finite.eq(0), "basin_id"]),
            "sim_constant_IDs": ", ".join(group.loc[group.status.eq("ok") & group.sim_finite.ge(2) & group.sim_unique.eq(1), "basin_id"]),
            "obs_negative_days": int(group.obs_negative.sum()), "sim_negative_days": int(group.sim_negative.sum())})
show_table('Độ phủ và chuỗi dự báo hằng/toàn thiếu trên kỳ test', a3_coverage_rows)

# %% Phần 3.4 — Đối chiếu obs giữa các run
# So quan trắc A2 trên cùng ngày–ID và giữ riêng sai khác số học.
a3_obs_comparison_rows = []
reference_run = "SeqLSTM_Q"
if reference_run in A3_DAILY:
    reference = A3_DAILY[reference_run]["obs"].to_numpy()
    for run in EXPECTED_RUNS:
        if run not in A3_DAILY:
            continue
        values = A3_DAILY[run]["obs"].to_numpy()
        valid = np.isfinite(values) & np.isfinite(reference)
        mask_difference = np.isfinite(values) != np.isfinite(reference)
        differences = np.abs(values[valid] - reference[valid])
        a3_obs_comparison_rows.append({"run": run, "reference": reference_run, "paired_obs_days": int(valid.sum()),
            "mask_difference_days": int(mask_difference.sum()), "mask_difference_IDs": int(mask_difference.any(axis=0).sum()),
            "abs_diff_median": float(np.median(differences)) if differences.size else np.nan,
            "abs_diff_max": float(differences.max()) if differences.size else np.nan,
            "days_diff_gt_1e_7": int((differences > 1e-7).sum()), "days_diff_gt_1e_5": int((differences > 1e-5).sum())})
show_table('Obs giữa các run trên cùng ngày–ID (chưa đối chiếu Q gốc)', a3_obs_comparison_rows)
print("Các mức 1e-7/1e-5 chỉ mô tả sai khác, chưa là dung sai chấp nhận hoặc lý do sửa obs.")

# %% Phần 3.5 — Hồ sơ A1 và lưu kiểm schema
# Xem 55/742/758/800, lưu bảng kiểm và trạng thái schema.
if not A3_SCHEMA_AUDIT.empty:
    profiles = A3_SCHEMA_AUDIT.loc[A3_SCHEMA_AUDIT.basin_id.isin(["55", "742", "758", "800"]),
        ["run", "basin_id", "status", "obs_finite", "sim_finite", "paired_finite", "obs_unique", "sim_unique", "sim_negative", "issues"]]
    show_table('Bốn hồ sơ A1 cần đối chiếu', profiles.to_dict("records"))
    errors = A3_SCHEMA_AUDIT.loc[A3_SCHEMA_AUDIT.status.ne("ok")]
    if not errors.empty:
        show_table('Các vấn đề schema/ngày — tối đa 20 dòng; đầy đủ trong CSV', errors.head(20).to_dict("records"))

A3_PICKLE_AUDIT.to_csv(A3_OUT_DIR / "A3_pickle_audit.csv", index=False, encoding="utf-8-sig")
A3_SCHEMA_AUDIT.to_csv(A3_OUT_DIR / "A3_prediction_schema.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(a3_coverage_rows).to_csv(A3_OUT_DIR / "A3_prediction_coverage.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(a3_obs_comparison_rows).to_csv(A3_OUT_DIR / "A3_interrun_obs_audit.csv", index=False, encoding="utf-8-sig")
A3_SCHEMA_AUDIT_OK = len(A3_DAILY) == len(EXPECTED_RUNS) and A3_PICKLE_AUDIT.status.eq("ok").all()

print("A3_SCHEMA_AUDIT_OK =", bool(A3_SCHEMA_AUDIT_OK))
print("Đã lưu 4 CSV audit vào:", A3_OUT_DIR)
print("Chưa kiểm đơn vị/quan trắc với Q gốc; chưa tính NSE/KGE/PNSE và chưa chặn sim_raw âm.")

if not A3_SCHEMA_AUDIT_OK:
    raise RuntimeError("Nội dung pickle/schema/ngày chưa đạt; gửi output trước khi kiểm Q gốc.")

# %% Phần 3.6 — Diện tích trạm và quy đổi đơn vị
# Đọc area_gov theo ID; dùng q(mm/ngày)=Q(m³/s)×86,4/area_gov(km²).
import re
import time

if not globals().get("A3_SCHEMA_AUDIT_OK", False):
    raise RuntimeError("Cần chạy và đạt Phần 3.1 trước khi đối chiếu nguồn Q.")

a3_q_started = time.perf_counter()
A3_Q_CALENDAR = pd.date_range(A3_TEST_DATES[0] - pd.Timedelta(days=1), A3_TEST_DATES[-1], freq="D", name="date")
A3_Q_SOURCE_STRUCTURE_OK = False
A3_Q_OBS_MASK_OK = False
A3_Q_VARIABLES = ("qmax", "qmean")
A3_Q_NUMERIC = {name: pd.DataFrame(np.nan, index=A3_Q_CALENDAR, columns=A3_SELECTED_IDS) for name in A3_Q_VARIABLES}
a3_q_file_rows, a3_q_value_rows, a3_raw_manifest_rows = [], [], []


print("Cửa sổ nguồn:", A3_Q_CALENDAR[0].date(), "→", A3_Q_CALENDAR[-1].date(), "|", len(A3_Q_CALENDAR), "ngày")
print("Chỉ lưu cửa sổ test cộng một ngày trước test; đây chưa phải dữ liệu train DLinear.")
a3_gauge_path = A3_LAMAH_DIR / "D_gauges" / "1_attributes" / "Gauge_attributes.csv"
a3_gauges = pd.read_csv(a3_gauge_path, sep=";", dtype=str, encoding="utf-8-sig", keep_default_na=False)
if not {"ID", "area_gov"}.issubset(a3_gauges.columns):
    raise ValueError("Gauge_attributes.csv thiếu ID/area_gov; không tự thay bằng area_calc.")
if not a3_gauges.ID.str.strip().str.fullmatch(r"\d+").all():
    raise ValueError("Thuộc tính trạm có ID không nguyên/rỗng.")
a3_gauges.index = a3_gauges.ID.map(lambda value: str(int(value.strip())))
if a3_gauges.index.has_duplicates or not set(A3_SELECTED_IDS).issubset(a3_gauges.index):
    raise ValueError("ID thuộc tính trùng hoặc thiếu một ID mô hình.")
A3_AREAS = pd.to_numeric(a3_gauges.loc[A3_SELECTED_IDS, "area_gov"], errors="coerce")
A3_AREAS.index.name = "basin_id"
if not (np.isfinite(A3_AREAS) & A3_AREAS.gt(0)).all():
    show_table('Diện tích không hợp lệ', A3_AREAS.loc[~(np.isfinite(A3_AREAS) & A3_AREAS.gt(0))].reset_index().to_dict("records"))
    raise ValueError("area_gov phải hữu hạn và dương ở đủ 451 ID.")
A3_AREA_CONVERSION = pd.DataFrame({"basin_id": A3_SELECTED_IDS, "area_gov_km2": A3_AREAS.to_numpy(),
                                   "m3_s_to_mm_day_factor": 86.4 / A3_AREAS.to_numpy()})
a3_raw_manifest_rows.append({"source": "raw_gauge_attributes", "run": "", "file": a3_gauge_path.name,
    "path": str(a3_gauge_path), "status": "hashed", "bytes": a3_gauge_path.stat().st_size,
    "sha256": a3_hash_file(a3_gauge_path)})
show_table('Diện tích dùng đổi đơn vị', [{"selected_IDs": len(A3_AREAS), "area_gov_unit": "km²",
    "area_min": float(A3_AREAS.min()), "area_median": float(A3_AREAS.median()), "area_max": float(A3_AREAS.max()),
    "conversion": "Q(m³/s) × 86.4 / area_gov(km²) = q(mm/day)"}])

# %% Phần 3.7 — Đọc cửa sổ Q nguồn và kiểm giá trị thiếu
# Giữ test cộng một ngày trước test; phân biệt giá trị thô và bản theo hợp đồng loader.
a3_daily_paths = {}
for path in (A3_LAMAH_DIR / "D_gauges" / "2_timeseries" / "daily").glob("ID_*.csv"):
    match = re.fullmatch(r"ID_(\d+)\.csv", path.name)
    if match:
        basin_id = str(int(match.group(1)))
        if basin_id in a3_daily_paths:
            raise ValueError(f"Hai tệp Q có cùng ID sau chuẩn hóa: {basin_id}")
        a3_daily_paths[basin_id] = path

for number, basin_id in enumerate(A3_SELECTED_IDS, 1):
    row = {"basin_id": basin_id, "status": "not_read", "source_rows": 0, "window_present_rows": 0,
           "window_absent_rows": len(A3_Q_CALENDAR), "issues": ""}
    path = a3_daily_paths.get(basin_id)
    try:
        if path is None:
            raise FileNotFoundError(f"Thiếu tệp Q ngày ID {basin_id}")
        raw = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", keep_default_na=False, na_filter=False,
                          usecols=["YYYY", "MM", "DD", *A3_Q_VARIABLES])
        date_parts = raw[["YYYY", "MM", "DD"]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(date_parts.to_numpy()).all() or not date_parts.eq(np.floor(date_parts)).all().all():
            raise ValueError("Ngày nguồn có thành phần không hữu hạn/không nguyên")
        dates = pd.DatetimeIndex(pd.to_datetime(date_parts.rename(columns={"YYYY": "year", "MM": "month", "DD": "day"}), errors="raise"))
        if dates.empty or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Ngày nguồn rỗng/NaT/trùng/không tăng; chưa tự gộp hoặc sắp xếp")
        row.update(source_rows=len(raw), window_present_rows=int(A3_Q_CALENDAR.isin(dates).sum()),
                   window_absent_rows=int((~A3_Q_CALENDAR.isin(dates)).sum()))
        issues = []
        for variable in A3_Q_VARIABLES:
            text = raw[variable].str.strip()
            values = pd.to_numeric(text, errors="coerce")
            known_missing = text.str.lower().isin(["", "nan", "na", "n/a", "null", "none"])
            unexpected = values.isna() & ~known_missing
            series = pd.Series(values.to_numpy(dtype=float), index=dates).reindex(A3_Q_CALENDAR)
            window_text = pd.Series(text.to_numpy(), index=dates).reindex(A3_Q_CALENDAR)
            window_unknown = pd.Series(unexpected.to_numpy(), index=dates).reindex(A3_Q_CALENDAR, fill_value=False)
            numbers = series.to_numpy()
            if window_unknown.any():
                issues.append(f"{variable}_unexpected_tokens_in_window")
            if np.isinf(numbers).any():
                issues.append(f"{variable}_infinite_in_window")
            A3_Q_NUMERIC[variable].loc[:, basin_id] = numbers
            a3_q_value_rows.append({"basin_id": basin_id, "variable": variable,
                "window_days": len(A3_Q_CALENDAR), "absent_rows": row["window_absent_rows"],
                "blank_cells": int(window_text.eq("").sum()), "explicit_na_tokens": int((window_text.notna() & window_text.str.lower().isin(["nan", "na", "n/a", "null", "none"])).sum()),
                "minus999_cells": int(series.eq(-999).sum()), "other_negative_cells": int((series.lt(0) & series.ne(-999)).sum()),
                "unexpected_tokens": int(window_unknown.sum()), "infinite_cells": int(np.isinf(numbers).sum()),
                "finite_nonnegative": int((np.isfinite(numbers) & (numbers >= 0)).sum()),
                "full_file_unexpected_tokens": int(unexpected.sum()),
                "full_file_other_negative": int((values.lt(0) & values.ne(-999)).sum()),
                "unexpected_examples": " | ".join(text.loc[unexpected].drop_duplicates().head(5))})
        row.update(status="ok" if not issues else "needs_review", issues=" | ".join(issues))
        a3_raw_manifest_rows.append({"source": "raw_daily_Q", "run": "", "file": path.name,
            "path": str(path), "status": "hashed", "bytes": path.stat().st_size, "sha256": a3_hash_file(path)})
    except Exception as exc:
        row.update(status="read_error", issues=f"{type(exc).__name__}: {exc}")
    a3_q_file_rows.append(row)
    if number % 100 == 0 or number == len(A3_SELECTED_IDS):
        print(f"Đã kiểm Q gốc {number}/{len(A3_SELECTED_IDS)} ID", flush=True)

A3_Q_FILE_AUDIT = pd.DataFrame(a3_q_file_rows)
A3_Q_VALUE_AUDIT = pd.DataFrame(a3_q_value_rows)
# Tái hiện quy tắc loader NH đã đối chiếu: mọi Q âm thành NaN; bản numeric vẫn giữ nguyên để audit.
A3_Q_LOADER_M3S = {name: frame.mask(frame.lt(0)) for name, frame in A3_Q_NUMERIC.items()}
A3_Q_MM_DAY = {name: frame.mul(86.4 / A3_AREAS, axis="columns") for name, frame in A3_Q_LOADER_M3S.items()}
show_table('Mã thiếu/giá trị bất thường trong cửa sổ nguồn', [
    {"variable": variable, "IDs_read": len(group), **{column: int(group[column].sum()) for column in (
        "window_days", "absent_rows", "blank_cells", "explicit_na_tokens", "minus999_cells", "other_negative_cells",
        "unexpected_tokens", "infinite_cells", "finite_nonnegative")}}
    for variable, group in A3_Q_VALUE_AUDIT.groupby("variable", sort=False)
] if not A3_Q_VALUE_AUDIT.empty else [])
print("Đổi Q âm thành NaN ở bản loader là tái hiện hợp đồng A2, không phải kết luận mọi quan trắc âm đều sai.")
print("Chưa chốt cách xử lý qmean cho DLinear; bản A3_Q_NUMERIC vẫn giữ giá trị số thô.")
q_errors = A3_Q_FILE_AUDIT.loc[A3_Q_FILE_AUDIT.status.ne("ok")]
if not q_errors.empty:
    show_table('Lỗi đọc nguồn — tối đa 20 dòng, đầy đủ trong CSV', q_errors.head(20).to_dict("records"))

# %% Phần 3.8 — Đối chiếu qmax nguồn với obs A2
# Kiểm giả thuyết đơn vị, mặt nạ và các hồ sơ lệch lớn nhất.
a3_q_obs_rows, a3_q_unit_rows = [], []
reference_mm = A3_Q_MM_DAY["qmax"].loc[A3_TEST_DATES, A3_SELECTED_IDS]
reference_m3s = A3_Q_LOADER_M3S["qmax"].loc[A3_TEST_DATES, A3_SELECTED_IDS]
for run in EXPECTED_RUNS:
    observed = A3_DAILY[run]["obs"].loc[A3_TEST_DATES, A3_SELECTED_IDS]
    for basin_id in A3_SELECTED_IDS:
        obs = observed[basin_id].to_numpy()
        source_mm = reference_mm[basin_id].to_numpy()
        common = np.isfinite(obs) & np.isfinite(source_mm)
        difference = np.abs(obs[common] - source_mm[common])
        a3_q_obs_rows.append({"run": run, "basin_id": basin_id, "paired_days": int(common.sum()),
            "mask_difference_days": int((np.isfinite(obs) != np.isfinite(source_mm)).sum()),
            "abs_diff_max_mm_day": float(difference.max()) if difference.size else np.nan,
            "days_diff_gt_1e_7": int((difference > 1e-7).sum()), "days_diff_gt_1e_5": int((difference > 1e-5).sum())})
    for candidate_unit, candidate in (("mm/day_area_gov", reference_mm), ("m3/s_raw", reference_m3s)):
        obs_values, source_values = observed.to_numpy(), candidate.to_numpy()
        common = np.isfinite(obs_values) & np.isfinite(source_values)
        differences = np.abs(obs_values[common] - source_values[common])
        a3_q_unit_rows.append({"run": run, "candidate_unit": candidate_unit, "paired_days": int(common.sum()),
            "mask_difference_days": int((np.isfinite(obs_values) != np.isfinite(source_values)).sum()),
            "abs_diff_median": float(np.median(differences)) if differences.size else np.nan,
            "abs_diff_p99": float(np.quantile(differences, 0.99)) if differences.size else np.nan,
            "abs_diff_max": float(differences.max()) if differences.size else np.nan,
            "days_diff_gt_1e_7": int((differences > 1e-7).sum()), "days_diff_gt_1e_5": int((differences > 1e-5).sum())})
A3_Q_OBS_AUDIT = pd.DataFrame(a3_q_obs_rows)
A3_Q_UNIT_AUDIT = pd.DataFrame(a3_q_unit_rows)
show_table('Obs A2 so với hai giả thuyết đơn vị của qmax nguồn', a3_q_unit_rows)
print("Sai khác với hai giả thuyết là chẩn đoán đơn vị, không phải chỉ số kỹ năng dự báo.")
print("Không tự coi 1e-7/1e-5 là ngưỡng đạt hoặc thay obs A2 bằng qmax nguồn.")
profiles = A3_Q_OBS_AUDIT.loc[A3_Q_OBS_AUDIT.basin_id.isin(["55", "742", "758", "800"])]
show_table('Đối chiếu Q nguồn ở bốn hồ sơ A1', profiles.to_dict("records"))
top_differences = A3_Q_OBS_AUDIT.sort_values("abs_diff_max_mm_day", ascending=False).head(10)
show_table('10 cặp run–ID lệch obs nguồn lớn nhất', top_differences.to_dict("records"))

# %% Phần 3.9 — Lưu kết quả đối chiếu Q
# Lưu diện tích, kiểm Q, cửa sổ nguồn và các cờ cấu trúc/mặt nạ.
A3_AREA_CONVERSION.to_csv(A3_OUT_DIR / "A3_area_conversion.csv", index=False, encoding="utf-8-sig")
A3_Q_FILE_AUDIT.to_csv(A3_OUT_DIR / "A3_Q_file_audit.csv", index=False, encoding="utf-8-sig")
A3_Q_VALUE_AUDIT.to_csv(A3_OUT_DIR / "A3_Q_value_audit.csv", index=False, encoding="utf-8-sig")
A3_Q_OBS_AUDIT.to_csv(A3_OUT_DIR / "A3_Q_observation_audit.csv", index=False, encoding="utf-8-sig")
A3_Q_UNIT_AUDIT.to_csv(A3_OUT_DIR / "A3_Q_unit_audit.csv", index=False, encoding="utf-8-sig")
A3_RAW_INPUT_MANIFEST = pd.DataFrame(a3_raw_manifest_rows)
A3_RAW_INPUT_MANIFEST.to_csv(A3_OUT_DIR / "A3_raw_Q_manifest.csv", index=False, encoding="utf-8-sig")
pd.concat([A3_INPUT_MANIFEST, A3_RAW_INPUT_MANIFEST], ignore_index=True).to_csv(
    A3_OUT_DIR / "A3_input_manifest.csv", index=False, encoding="utf-8-sig")
np.savez_compressed(A3_OUT_DIR / "A3_Q_source_window.npz", dates=A3_Q_CALENDAR.to_numpy(dtype="datetime64[D]"),
    basin_ids=np.asarray(A3_SELECTED_IDS, dtype="U"), area_gov_km2=A3_AREAS.to_numpy(),
    qmax_numeric_m3_s=A3_Q_NUMERIC["qmax"].to_numpy(), qmean_numeric_m3_s=A3_Q_NUMERIC["qmean"].to_numpy())
A3_Q_SOURCE_STRUCTURE_OK = len(A3_Q_FILE_AUDIT) == 451 and A3_Q_FILE_AUDIT.status.eq("ok").all()
A3_Q_OBS_MASK_OK = A3_Q_SOURCE_STRUCTURE_OK and A3_Q_OBS_AUDIT.mask_difference_days.eq(0).all()

print("A3_Q_SOURCE_STRUCTURE_OK =", bool(A3_Q_SOURCE_STRUCTURE_OK), "| A3_Q_OBS_MASK_OK =", bool(A3_Q_OBS_MASK_OK))
print("Đã lưu 6 CSV riêng, cập nhật manifest chung và lưu cửa sổ Q số thô dạng NPZ tại:", A3_OUT_DIR)
print("Hai cờ trên kiểm cấu trúc/mặt nạ, chưa tự xác nhận dung sai giá trị hoặc đơn vị.")
print(f"Thời gian đọc/đối chiếu: {time.perf_counter() - a3_q_started:.1f} giây")

if not A3_Q_SOURCE_STRUCTURE_OK or not A3_Q_OBS_MASK_OK:
    raise RuntimeError("Nguồn Q hoặc mặt nạ quan trắc chưa khớp; gửi output trước khi dựng persistence.")

# %% Phần 4.1 — Hàm lag theo ngày và tóm tắt mặt nạ
# Lấy đúng ngày lịch t−1; không lấy số đo hợp lệ gần nhất qua khoảng thiếu.
A3_PERSISTENCE_MASKS_OK = False
if not globals().get("A3_Q_SOURCE_STRUCTURE_OK", False) or not globals().get("A3_Q_OBS_MASK_OK", False):
    raise RuntimeError("Cần chạy và đạt Phần 3.2 trước khi dựng persistence.")


def a3_calendar_lag(frame, target_dates):
    # Chỉ dịch trên lịch ngày đầy đủ, không dịch chuỗi đã bỏ ô/ngày thiếu.
    dates = pd.DatetimeIndex(frame.index)
    target_dates = pd.DatetimeIndex(target_dates)
    required = pd.date_range(target_dates.min() - pd.Timedelta(days=1), target_dates.max(), freq="D")
    if (dates.empty or dates.hasnans or dates.has_duplicates or dates.tz is not None
            or not dates.equals(pd.date_range(dates.min(), dates.max(), freq="D"))
            or not dates.equals(dates.normalize()) or not required.isin(dates).all()):
        raise ValueError("Lag cần lịch ngày đầy đủ và đủ ngày t−1; không tự nối qua ngày thiếu.")
    return frame.shift(1).reindex(target_dates)


def a3_mask_summary(run, scope, mask, native=None):
    counts = mask.sum(axis=0)
    return {"run": run, "scope": scope, "valid_day_IDs": int(mask.to_numpy().sum()),
            "IDs_0_days": int(counts.eq(0).sum()), "IDs_1_day": int(counts.eq(1).sum()),
            "IDs_ge2_days": int(counts.ge(2).sum()), "valid_on_first_test_day": int(mask.iloc[0].sum()),
            "days_lost_vs_native": int(native.to_numpy().sum() - mask.to_numpy().sum()) if native is not None else 0}

# %% Phần 4.2 — Ví dụ kiểm lag qua ngày thiếu
# Ví dụ minh họa độc lập để kiểm ngày nguồn và giá trị persistence.
# Ví dụ minh họa độc lập: sau ngày Q thiếu, không lấy số đo gần nhất thay cho hôm qua.
a3_toy_dates = pd.date_range("2013-12-31", "2014-01-04", freq="D")
a3_toy = pd.DataFrame({"qmax": [10.0, 20.0, np.nan, 40.0, 50.0]}, index=a3_toy_dates)
a3_toy_lag = a3_calendar_lag(a3_toy, a3_toy_dates[1:])
assert np.array_equal(a3_toy_lag.qmax.to_numpy(), np.asarray([10.0, 20.0, np.nan, 40.0]), equal_nan=True)
show_table('Minh họa kiểm lag (không phải kết quả thực nghiệm)', [
    {"target_date": str(date.date()), "source_date": str((date - pd.Timedelta(days=1)).date()),
     "observed_today": a3_toy.at[date, "qmax"], "persistence": a3_toy_lag.at[date, "qmax"]}
    for date in a3_toy_dates[1:]])

# %% Phần 4.3 — Dựng hai chuỗi persistence
# Tạo qmax(t−1) làm mốc chính và qmean(t−1) làm mốc phụ.
A3_SOURCE_OBS = A3_Q_MM_DAY["qmax"].loc[A3_TEST_DATES, A3_SELECTED_IDS].copy()
A3_PERSISTENCE_QMAX = a3_calendar_lag(A3_Q_MM_DAY["qmax"], A3_TEST_DATES).loc[:, A3_SELECTED_IDS]
A3_PERSISTENCE_QMEAN_AUX = a3_calendar_lag(A3_Q_MM_DAY["qmean"], A3_TEST_DATES).loc[:, A3_SELECTED_IDS]
for variable, prediction in (("qmax", A3_PERSISTENCE_QMAX), ("qmean", A3_PERSISTENCE_QMEAN_AUX)):
    previous_days = A3_Q_MM_DAY[variable].reindex(A3_TEST_DATES - pd.Timedelta(days=1))
    if not np.array_equal(prediction.to_numpy(), previous_days.to_numpy(), equal_nan=True):
        raise ValueError(f"{variable}: lag không khớp phép tra trực tiếp ngày t−1")
    if np.isinf(prediction.to_numpy()).any():
        raise ValueError(f"{variable}: persistence có vô cực")

# %% Phần 4.4 — Dựng các mặt nạ chấm điểm
# Tách native, pnse_native, paired_source và giao A2/persistence.
a3_source_valid = np.isfinite(A3_SOURCE_OBS)
a3_persistence_valid = np.isfinite(A3_PERSISTENCE_QMAX)
A3_BASELINE_MASKS = {
    "Persistence_qmax": a3_source_valid & a3_persistence_valid,
    "Persistence_qmean_aux": a3_source_valid & np.isfinite(A3_PERSISTENCE_QMEAN_AUX),
}
A3_RUN_MASKS, a3_mask_rows, a3_mask_basin_rows = {}, [], []
for name, mask in A3_BASELINE_MASKS.items():
    a3_mask_rows.append(a3_mask_summary(name, "native_source", mask))
A3_COMMON_MASK = A3_BASELINE_MASKS["Persistence_qmax"].copy()
for run in EXPECTED_RUNS:
    observed = A3_DAILY[run]["obs"]
    sim = A3_DAILY[run]["sim_raw"]
    if not observed.index.equals(A3_TEST_DATES) or list(observed.columns) != A3_SELECTED_IDS:
        raise ValueError(f"{run}: chỉ mục obs khác hợp đồng")
    if not sim.index.equals(A3_TEST_DATES) or list(sim.columns) != A3_SELECTED_IDS:
        raise ValueError(f"{run}: chỉ mục sim khác hợp đồng")
    if not np.array_equal(np.isfinite(observed.to_numpy()), a3_source_valid.to_numpy()):
        raise ValueError(f"{run}: mặt nạ obs đổi sau Phần 3.2")
    native = np.isfinite(observed) & np.isfinite(sim)
    A3_RUN_MASKS[run] = {
        "native": native,
        "pnse_native": native & a3_persistence_valid,
        "paired_source": a3_source_valid & np.isfinite(sim) & a3_persistence_valid,
    }
    A3_COMMON_MASK &= A3_RUN_MASKS[run]["paired_source"]
    for scope, mask in A3_RUN_MASKS[run].items():
        a3_mask_rows.append(a3_mask_summary(run, scope, mask, native))

a3_mask_rows.append(a3_mask_summary("A2_six_runs_and_Persistence_qmax", "common_source", A3_COMMON_MASK,
                                     A3_BASELINE_MASKS["Persistence_qmax"]))
for run in EXPECTED_RUNS:
    for scope, mask in A3_RUN_MASKS[run].items():
        if scope != "native" and (mask & ~A3_RUN_MASKS[run]["native"]).to_numpy().any():
            raise ValueError(f"{run}/{scope}: mặt nạ không là tập con của native")
    if (A3_COMMON_MASK & ~A3_RUN_MASKS[run]["paired_source"]).to_numpy().any():
        raise ValueError(f"{run}: common_source chứa ngày mô hình không có dự báo")
    for basin_id in A3_SELECTED_IDS:
        counts = {scope + "_days": int(mask[basin_id].sum()) for scope, mask in A3_RUN_MASKS[run].items()}
        a3_mask_basin_rows.append({"run": run, "basin_id": basin_id, **counts,
            "common_source_days": int(A3_COMMON_MASK[basin_id].sum()),
            "lost_for_persistence": counts["native_days"] - counts["pnse_native_days"]})
A3_MASK_SUMMARY = pd.DataFrame(a3_mask_rows)
A3_MASK_COUNTS = pd.DataFrame(a3_mask_basin_rows)
show_table('Số ngày–ID của từng phạm vi', a3_mask_rows)
print("native/pnse_native: obs A2 của chính run; paired_source/common_source: obs qmax từ Q gốc.")
print("PNSE luôn dùng qmax(t−1); qmean(t−1) chỉ là mốc phụ để đối chiếu khảo sát trước.")
print("Giao common_source hiện gồm sáu run A2 và persistence qmax; sau DLinear sẽ lập lại giao có DLinear.")

# %% Phần 4.5 — Kiểm ngày đầu test và các hồ sơ thiếu
# Đối chiếu ngày nguồn thật, những ngày mất lag và bốn hồ sơ A1.
first_day = A3_TEST_DATES[0]
show_table('Ngày đầu test và ngày nguồn thật', [{"target_date": str(first_day.date()),
    "source_date": str((first_day - pd.Timedelta(days=1)).date()),
    "qmax_lag_finite_IDs": int(np.isfinite(A3_PERSISTENCE_QMAX.loc[first_day]).sum()),
    "qmean_lag_finite_IDs": int(np.isfinite(A3_PERSISTENCE_QMEAN_AUX.loc[first_day]).sum()),
    "obs_finite_IDs": int(a3_source_valid.loc[first_day].sum()),
    "qmax_valid_pairs": int(A3_BASELINE_MASKS["Persistence_qmax"].loc[first_day].sum())}])
reappearing = a3_source_valid & ~a3_persistence_valid
a3_gap_examples = []
for day_number, column_number in np.argwhere(reappearing.to_numpy())[:10]:
    date, basin_id = A3_TEST_DATES[day_number], A3_SELECTED_IDS[column_number]
    a3_gap_examples.append({"basin_id": basin_id, "target_date": str(date.date()),
        "source_date": str((date - pd.Timedelta(days=1)).date()), "obs_today": A3_SOURCE_OBS.at[date, basin_id],
        "persistence": A3_PERSISTENCE_QMAX.at[date, basin_id], "included_in_baseline": False})
show_table('Ngày có obs nhưng hôm trước thiếu — tối đa 10 ví dụ', a3_gap_examples)
print("Tổng ngày–ID có obs nhưng thiếu qmax hôm trước:", int(reappearing.to_numpy().sum()))
profiles = A3_MASK_COUNTS.loc[A3_MASK_COUNTS.basin_id.isin(["55", "742", "758", "800"])]
show_table('Số ngày chấm ở bốn hồ sơ A1', profiles.to_dict("records"))
print("ID không có ngày trong giao chung (vẫn giữ trong danh sách 451):",
      ", ".join(A3_COMMON_MASK.columns[A3_COMMON_MASK.sum(axis=0).eq(0)]))

# %% Phần 4.6 — Lưu persistence và quy ước đánh giá
# Lưu dự báo, mặt nạ, số ngày và hợp đồng đánh giá vào a3_output.
A3_EVALUATION_PROTOCOL = {
    "target_unit": "mm/day", "test_period": A3_PERIODS["test"], "main_baseline": "qmax_observed(t-1)",
    "auxiliary_baseline": "qmean_observed(t-1); not the PNSE reference",
    "native": "A2 run-specific obs and finite sim; reproduce saved metrics with config clipping later",
    "pnse_native": "native intersect finite qmax persistence; retain run-specific obs for both SSE terms",
    "paired_source": "finite source qmax obs, model sim and persistence; use same source obs for both forecasts",
    "common_source": "paired_source intersection of all six A2 runs; same source obs for every forecast",
    "common_runs": list(EXPECTED_RUNS), "missing_policy": "keep missing; no interpolation or last-valid-observation carry",
    "clip_policy": "raw simulations unchanged here; apply only config-declared policy in metric stage",
    "note": "Comparison scope will be rebuilt after DLinear; source dates are a daily index convention, not publication times.",
}
a3_saved_masks = {"dates": A3_TEST_DATES.to_numpy(dtype="datetime64[D]"), "basin_ids": np.asarray(A3_SELECTED_IDS, dtype="U"),
                  "common_source": A3_COMMON_MASK.to_numpy(dtype=bool)}
for run, scopes in A3_RUN_MASKS.items():
    for scope, mask in scopes.items():
        a3_saved_masks[f"{run}__{scope}"] = mask.to_numpy(dtype=bool)
for name, mask in A3_BASELINE_MASKS.items():
    a3_saved_masks[f"{name}__native_source"] = mask.to_numpy(dtype=bool)
np.savez_compressed(A3_OUT_DIR / "A3_evaluation_masks.npz", **a3_saved_masks)
np.savez_compressed(A3_OUT_DIR / "A3_persistence_predictions.npz",
    dates=A3_TEST_DATES.to_numpy(dtype="datetime64[D]"), basin_ids=np.asarray(A3_SELECTED_IDS, dtype="U"),
    source_dates=(A3_TEST_DATES - pd.Timedelta(days=1)).to_numpy(dtype="datetime64[D]"),
    obs_source_mm_day=A3_SOURCE_OBS.to_numpy(), persistence_qmax_mm_day=A3_PERSISTENCE_QMAX.to_numpy(),
    persistence_qmean_aux_mm_day=A3_PERSISTENCE_QMEAN_AUX.to_numpy())
A3_MASK_SUMMARY.to_csv(A3_OUT_DIR / "A3_mask_summary.csv", index=False, encoding="utf-8-sig")
A3_MASK_COUNTS.to_csv(A3_OUT_DIR / "A3_mask_counts_per_basin.csv", index=False, encoding="utf-8-sig")
(A3_OUT_DIR / "A3_evaluation_protocol.json").write_text(
    json.dumps(A3_EVALUATION_PROTOCOL, ensure_ascii=False, indent=2), encoding="utf-8")
A3_PERSISTENCE_MASKS_OK = True

print("A3_PERSISTENCE_MASKS_OK =", A3_PERSISTENCE_MASKS_OK)
print("Đã lưu dự báo persistence, mặt nạ, bảng độ phủ và quy ước đánh giá vào:", A3_OUT_DIR)
print("Chưa tính NSE/KGE/PNSE; Phần 5 sẽ định nghĩa và kiểm các hàm chỉ số.")

# %% Phần 5.1 — Hàm NSE, PNSE, KGE và các thành phần
# Tính float64 trên một mặt nạ; giữ NaN/lý do khi mẫu số hoặc tương quan không xác định.
import importlib.util
import math
import statistics
from importlib import metadata as a3_metadata

A3_METRIC_TESTS_OK = False
A3_NH_REFERENCE_OK = None
if not globals().get("A3_PERSISTENCE_MASKS_OK", False):
    raise RuntimeError("Cần chạy và đạt Phần 4 trước khi định nghĩa bộ chỉ số.")
A3_METRIC_NAMES = ("NSE", "PNSE", "KGE", "r", "alpha", "beta")


def a3_metrics(obs, sim, persistence=None, mask=None):
    """Một chuỗi, một mặt nạ cho cả sáu chỉ số; không clip sim hoặc đổi NaN thành 0."""
    arrays = [np.asarray(values, dtype=np.float64) for values in (obs, sim)]
    if persistence is not None:
        arrays.append(np.asarray(persistence, dtype=np.float64))
    if any(array.ndim != 1 or array.shape != arrays[0].shape for array in arrays):
        raise ValueError("obs/sim/persistence phải là các mảng 1D cùng kích thước")
    if mask is None:
        if any(np.isinf(array).any() for array in arrays):
            raise ValueError("Đầu vào có vô cực; không tự xem vô cực là thiếu")
        selected = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    else:
        selected = np.asarray(mask)
        if selected.dtype != np.bool_ or selected.ndim != 1 or selected.shape != arrays[0].shape:
            raise ValueError("mask phải là mảng boolean 1D cùng kích thước chuỗi")
        if any(not np.isfinite(array[selected]).all() for array in arrays):
            raise ValueError("mask chọn obs/sim/persistence không hữu hạn; không tự thu hẹp mặt nạ")
    result = {name: np.nan for name in A3_METRIC_NAMES}
    result.update({name + "_status": "no_valid_pairs" for name in A3_METRIC_NAMES})
    result.update(n_input=len(arrays[0]), n=int(selected.sum()), sse_model=np.nan, sse_persistence=np.nan,
                  obs_tss=np.nan, mean_obs=np.nan, mean_sim=np.nan, std_obs=np.nan, std_sim=np.nan)
    if not result["n"]:
        return result
    observed, predicted = arrays[0][selected], arrays[1][selected]
    n = result["n"]
    obs_constant = bool(np.all(observed == observed[0]))
    sim_constant = bool(np.all(predicted == predicted[0]))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean_obs = float(observed[0]) if obs_constant else float(observed.mean())
        mean_sim = float(predicted[0]) if sim_constant else float(predicted.mean())
        obs_centered, sim_centered = observed - mean_obs, predicted - mean_sim
        obs_tss = 0.0 if obs_constant else float(np.sum(obs_centered ** 2))
        sim_tss = 0.0 if sim_constant else float(np.sum(sim_centered ** 2))
        sse_model = float(np.sum((predicted - observed) ** 2))
        if not np.isfinite([mean_obs, mean_sim, obs_tss, sim_tss, sse_model]).all():
            raise FloatingPointError("Số học chỉ số không hữu hạn; không thêm epsilon hoặc cắt điểm")
        std_obs, std_sim = math.sqrt(obs_tss / n), math.sqrt(sim_tss / n)
        result.update(sse_model=sse_model, obs_tss=obs_tss, mean_obs=mean_obs, mean_sim=mean_sim,
                      std_obs=std_obs, std_sim=std_sim)
        if obs_tss > 0:
            result.update(NSE=1.0 - sse_model / obs_tss, NSE_status="ok",
                          alpha=std_sim / std_obs, alpha_status="ok")
        else:
            result.update(NSE_status="obs_zero_variance", alpha_status="obs_zero_variance")
        if mean_obs != 0:
            result.update(beta=mean_sim / mean_obs, beta_status="ok")
        else:
            result["beta_status"] = "obs_mean_zero"
        if n < 2:
            result["r_status"] = "insufficient_pairs"
        elif obs_tss == 0:
            result["r_status"] = "obs_zero_variance"
        elif sim_tss == 0:
            result["r_status"] = "sim_constant"
        else:
            correlation = float(np.dot(obs_centered / math.sqrt(obs_tss), sim_centered / math.sqrt(sim_tss)))
            # Pearson có miền [-1, 1]; giới hạn sai số làm tròn, không thay đổi dự báo/điểm NSE.
            result.update(r=float(np.clip(correlation, -1.0, 1.0)), r_status="ok")
        missing_components = [name for name in ("r", "alpha", "beta") if result[name + "_status"] != "ok"]
        if missing_components:
            result["KGE_status"] = "undefined_components:" + ",".join(missing_components)
        else:
            result.update(KGE=1.0 - math.sqrt(sum((result[name] - 1.0) ** 2 for name in ("r", "alpha", "beta"))),
                          KGE_status="ok")
        if persistence is None:
            result["PNSE_status"] = "persistence_not_provided"
        else:
            sse_persistence = float(np.sum((arrays[2][selected] - observed) ** 2))
            result["sse_persistence"] = sse_persistence
            if not np.isfinite(sse_persistence):
                raise FloatingPointError("SSE persistence không hữu hạn")
            if sse_persistence > 0:
                result.update(PNSE=1.0 - sse_model / sse_persistence, PNSE_status="ok")
            else:
                result["PNSE_status"] = "persistence_sse_zero"
    if any(not np.isfinite(result[name]) for name in A3_METRIC_NAMES if result[name + "_status"] == "ok"):
        raise FloatingPointError("Chỉ số khai báo ok nhưng không hữu hạn")
    return result


def a3_sim_for_scoring(sim_raw, clip_to_zero=False):
    """Tạo bản để chấm theo config; không ghi đè sim_raw."""
    values = np.asarray(sim_raw, dtype=np.float64).copy()
    return np.maximum(values, 0.0) if clip_to_zero else values



print("NSE/PNSE/KGE/r/alpha/beta cùng dùng một mặt nạ; std ddof=0; không epsilon ở mẫu số.")
print("Nếu không truyền mask và có persistence, cả sáu chỉ số dùng giao hữu hạn obs/sim/persistence.")
print("Nếu truyền mask, mọi giá trị được chọn phải hữu hạn; không âm thầm bỏ thêm ngày.")

# %% Phần 5.2 — Công cụ kiểm ví dụ số
# Định nghĩa cách so giá trị, số cặp và trạng thái kỳ vọng.
a3_metric_test_rows, a3_input_test_rows, a3_reference_rows = [], [], []


def a3_check_case(name, obs, sim, persistence=None, mask=None, expected=None, expected_n=None, statuses=None):
    measured = a3_metrics(obs, sim, persistence, mask)
    failures = []
    if expected_n is not None and measured["n"] != expected_n:
        failures.append(f"n={measured['n']}, expected={expected_n}")
    for metric, expected_value in (expected or {}).items():
        valid = np.isnan(measured[metric]) if np.isnan(expected_value) else np.isclose(
            measured[metric], expected_value, rtol=1e-12, atol=1e-12)
        if not valid:
            failures.append(f"{metric}={measured[metric]}, expected={expected_value}")
    for key, expected_status in (statuses or {}).items():
        if measured[key + "_status"] != expected_status:
            failures.append(f"{key}_status={measured[key + '_status']}")
    a3_metric_test_rows.append({"case": name, "passed": not failures, "n": measured["n"],
        **{key: measured[key] for key in A3_METRIC_NAMES},
        "undefined": " | ".join(f"{key}:{measured[key + '_status']}" for key in A3_METRIC_NAMES if measured[key + "_status"] != "ok"),
        "issues": " | ".join(failures)})
    return measured

# %% Phần 5.3 — Ví dụ đúng trước và các ca suy biến
# Kiểm dự báo hoàn hảo, persistence, chuỗi hằng, thiếu mẫu và đổi đơn vị.
a3_demo_obs, a3_demo_persistence = [1, 2, 3, 4], [1, 1, 2, 3]
a3_check_case("perfect", a3_demo_obs, a3_demo_obs, a3_demo_persistence,
              expected={key: 1.0 for key in A3_METRIC_NAMES}, expected_n=4)
a3_r_expected, a3_alpha_expected = 7 / math.sqrt(55), math.sqrt(11 / 20)
a3_kge_expected = 1 - math.sqrt((a3_r_expected - 1) ** 2 + (a3_alpha_expected - 1) ** 2 + (0.7 - 1) ** 2)
a3_persistence_demo = a3_check_case("persistence_equals_model", a3_demo_obs, a3_demo_persistence, a3_demo_persistence,
    expected={"NSE": 0.4, "PNSE": 0.0, "r": a3_r_expected, "alpha": a3_alpha_expected, "beta": 0.7, "KGE": a3_kge_expected})
a3_check_case("double_amplitude", a3_demo_obs, [2, 4, 6, 8], a3_demo_persistence,
    expected={"NSE": -5, "PNSE": -9, "r": 1, "alpha": 2, "beta": 2, "KGE": 1 - math.sqrt(2)})
a3_check_case("reversed_series", a3_demo_obs, [4, 3, 2, 1], a3_demo_persistence,
    expected={"NSE": -3, "r": -1, "alpha": 1, "beta": 1, "KGE": -1})
a3_check_case("constant_sim_at_obs_mean", a3_demo_obs, [2.5] * 4, a3_demo_persistence,
    expected={"NSE": 0, "PNSE": -2 / 3, "alpha": 0, "beta": 1, "r": np.nan, "KGE": np.nan},
    statuses={"r": "sim_constant", "KGE": "undefined_components:r"})
a3_check_case("constant_obs", [2, 2, 2], [1, 2, 3], [1, 1, 1],
    expected={"NSE": np.nan, "PNSE": 1 / 3, "r": np.nan, "alpha": np.nan, "beta": 1, "KGE": np.nan},
    statuses={"NSE": "obs_zero_variance"})
a3_check_case("zero_obs_mean_algebra_only", [-1, 0, 1], [-1, 0, 1],
    expected={"NSE": 1, "r": 1, "alpha": 1, "beta": np.nan, "KGE": np.nan}, statuses={"beta": "obs_mean_zero"})
a3_check_case("zero_persistence_SSE", [1, 2, 3], [2, 2, 2], [1, 2, 3],
    expected={"NSE": 0, "PNSE": np.nan}, statuses={"PNSE": "persistence_sse_zero"})
a3_check_case("both_SSE_zero", [1, 2, 3], [1, 2, 3], [1, 2, 3],
    expected={"NSE": 1, "PNSE": np.nan}, statuses={"PNSE": "persistence_sse_zero"})
a3_check_case("no_pairs", [np.nan], [1], [1], expected={key: np.nan for key in A3_METRIC_NAMES}, expected_n=0)
a3_check_case("one_pair", [2], [3], [1], expected={"NSE": np.nan, "r": np.nan, "KGE": np.nan, "PNSE": 0, "beta": 1.5},
              expected_n=1, statuses={"r": "insufficient_pairs"})
a3_check_case("explicit_mask", [1, 2, np.nan, 4], [1, 2, 999, 4], [0, 1, 0, 3],
    mask=np.asarray([True, True, False, True]), expected={key: 1 for key in A3_METRIC_NAMES}, expected_n=3)
a3_check_case("one_shared_auto_mask", [1, 2, 3, 4], [1, np.nan, 3, 4], [0, 1, np.nan, 3],
    expected={key: 1 for key in A3_METRIC_NAMES}, expected_n=2)
a3_near_constant = a3_metrics([1, 1 + 1e-8, 1 + 2e-8], [2, 2, 2], [1, 1, 1])
a3_metric_test_rows.append({"case": "small_positive_variance_not_regularized",
    "passed": bool(np.isfinite(a3_near_constant["NSE"]) and a3_near_constant["NSE"] < -1e12 and a3_near_constant["obs_tss"] > 0),
    "n": a3_near_constant["n"], **{key: a3_near_constant[key] for key in A3_METRIC_NAMES},
    "undefined": a3_near_constant["KGE_status"], "issues": ""})
a3_scaled_demo = a3_metrics(np.asarray(a3_demo_obs) * 2.5, np.asarray(a3_demo_persistence) * 2.5,
                            np.asarray(a3_demo_persistence) * 2.5)
a3_metric_test_rows.append({"case": "positive_unit_scale_invariance",
    "passed": all(np.isclose(a3_scaled_demo[key], a3_persistence_demo[key], atol=1e-12, rtol=1e-12) for key in A3_METRIC_NAMES),
    "n": a3_scaled_demo["n"], **{key: a3_scaled_demo[key] for key in A3_METRIC_NAMES}, "undefined": "", "issues": ""})
a3_raw_clip_demo = np.asarray([-1.0, 2.0, np.nan])
a3_clipped_demo = a3_sim_for_scoring(a3_raw_clip_demo, True)
a3_metric_test_rows.append({"case": "config_clip_copy_preserves_raw_and_nan", "passed": bool(
    np.array_equal(a3_clipped_demo, [0, 2, np.nan], equal_nan=True)
    and np.array_equal(a3_raw_clip_demo, [-1, 2, np.nan], equal_nan=True)), "issues": ""})

# %% Phần 5.4 — Kiểm đầu vào và mặt nạ không hợp lệ
# Từ chối sai shape, mask không boolean, vô cực và mask chọn giá trị thiếu.
for name, arguments in (
    ("different_lengths", {"obs": [1, 2], "sim": [1]}),
    ("two_dimensions", {"obs": [[1, 2]], "sim": [[1, 2]]}),
    ("integer_mask", {"obs": [1, 2], "sim": [1, 2], "mask": [1, 0]}),
    ("mask_selects_nan", {"obs": [1, np.nan], "sim": [1, 2], "mask": [True, True]}),
    ("infinite_input", {"obs": [1, 2], "sim": [1, np.inf]}),
    ("persistence_wrong_shape", {"obs": [1, 2], "sim": [1, 2], "persistence": [1]}),
):
    try:
        a3_metrics(**arguments)
        a3_input_test_rows.append({"case": name, "passed": False, "result": "invalid_input_accepted"})
    except ValueError as exc:
        a3_input_test_rows.append({"case": name, "passed": True, "result": str(exc)})

# %% Phần 5.5 — Đối chiếu hàm tham chiếu
# So Pearson và NeuralHydrology khi gói có sẵn; không tự cài thêm.
for name, predicted in (("perfect", a3_demo_obs), ("persistence", a3_demo_persistence),
                        ("double", [2, 4, 6, 8]), ("reversed", [4, 3, 2, 1])):
    computed = a3_metrics(a3_demo_obs, predicted)
    reference_r = statistics.correlation(a3_demo_obs, predicted)
    a3_reference_rows.append({"reference": "stdlib.statistics.correlation", "case": name,
        "metric": "r", "computed": computed["r"], "reference_value": reference_r,
        "passed": bool(np.isclose(computed["r"], reference_r, atol=1e-12, rtol=1e-12)), "status": "checked"})

if importlib.util.find_spec("neuralhydrology") is None:
    a3_reference_rows.append({"reference": "NeuralHydrology", "status": "not_run_package_not_installed",
                              "passed": None, "note": "Không tự cài gói; đối chiếu CSV A2 bắt buộc ở Phần 6."})
else:
    try:
        from neuralhydrology.evaluation import metrics as a3_nh_metrics
        import xarray as a3_reference_xr
        a3_nh_version = a3_metadata.version("neuralhydrology")
        a3_nh_direct = json.loads(a3_metadata.distribution("neuralhydrology").read_text("direct_url.json") or "{}")
        a3_nh_commit = a3_nh_direct.get("vcs_info", {}).get("commit_id", "unknown")
        print("NeuralHydrology tham chiếu:", a3_nh_version, "| commit:", a3_nh_commit)
        a3_nh_checks = []
        for name, predicted in (("perfect", a3_demo_obs), ("persistence", a3_demo_persistence),
                                ("double", [2, 4, 6, 8]), ("reversed", [4, 3, 2, 1])):
            computed = a3_metrics(a3_demo_obs, predicted)
            obs_xr = a3_reference_xr.DataArray(np.asarray(a3_demo_obs, dtype=np.float64), dims=["date"])
            sim_xr = a3_reference_xr.DataArray(np.asarray(predicted, dtype=np.float64), dims=["date"])
            for metric, function in (("NSE", a3_nh_metrics.nse), ("KGE", a3_nh_metrics.kge)):
                reference_value = function(obs_xr, sim_xr)
                passed = bool(np.isclose(computed[metric], reference_value, rtol=1e-12, atol=1e-12))
                a3_nh_checks.append(passed)
                a3_reference_rows.append({"reference": "NeuralHydrology", "case": name, "metric": metric,
                    "computed": computed[metric], "reference_value": reference_value, "passed": passed, "status": "checked",
                    "library_version": a3_nh_version, "library_commit": a3_nh_commit,
                    "commit_matches_A2": a3_nh_commit == EXPECTED_COMMIT})
        A3_NH_REFERENCE_OK = all(a3_nh_checks)
    except Exception as exc:
        A3_NH_REFERENCE_OK = False
        a3_reference_rows.append({"reference": "NeuralHydrology", "status": "reference_error", "passed": False,
                                  "note": f"{type(exc).__name__}: {exc}"})

# %% Phần 5.6 — Bảng kiểm và trạng thái hàm chỉ số
# Hiển thị/lưu các kiểm tra và xác nhận điều kiện chấm dữ liệu thật.
A3_METRIC_TESTS = pd.DataFrame(a3_metric_test_rows)
A3_METRIC_INPUT_TESTS = pd.DataFrame(a3_input_test_rows)
A3_METRIC_REFERENCES = pd.DataFrame(a3_reference_rows)
show_table('Ví dụ số và ca suy biến — không phải kết quả dataset', a3_metric_test_rows)
show_table('Từ chối đầu vào/mặt nạ không hợp lệ', a3_input_test_rows)
show_table('Đối chiếu tham chiếu; not_run không phải pass', a3_reference_rows)
A3_METRIC_TESTS.to_csv(A3_OUT_DIR / "A3_metric_unit_checks.csv", index=False, encoding="utf-8-sig")
A3_METRIC_INPUT_TESTS.to_csv(A3_OUT_DIR / "A3_metric_input_checks.csv", index=False, encoding="utf-8-sig")
A3_METRIC_REFERENCES.to_csv(A3_OUT_DIR / "A3_metric_reference_checks.csv", index=False, encoding="utf-8-sig")
A3_METRIC_TESTS_OK = bool(A3_METRIC_TESTS.passed.all() and A3_METRIC_INPUT_TESTS.passed.all()
    and A3_METRIC_REFERENCES.loc[A3_METRIC_REFERENCES.status.eq("checked"), "passed"].all()
    and A3_NH_REFERENCE_OK is not False)

print("A3_METRIC_TESTS_OK =", A3_METRIC_TESTS_OK, "| A3_NH_REFERENCE_OK =", A3_NH_REFERENCE_OK)
print("None ở NH nghĩa là chưa chạy đối chiếu gói; không được diễn giải thành khớp NeuralHydrology.")
print("Hàm chỉ số chưa chấm A2/persistence thực nghiệm; native A2 sẽ đối chiếu NSE/KGE CSV ở Phần 6.")
print("Đã lưu 3 CSV kiểm thử vào:", A3_OUT_DIR)

if not A3_METRIC_TESTS_OK:
    raise RuntimeError("Kiểm chỉ số chưa đạt; chưa chấm dữ liệu thực nghiệm, gửi output để kiểm.")

# %% Phần 6.1 — Hợp đồng chấm điểm và dung sai CSV
# Giữ quy tắc clip theo config và hàm chấm từng run–scope.
A3_BASELINE_EVAL_OK = False
A3_NATIVE_CSV_REVIEW_OK = False
if not globals().get("A3_METRIC_TESTS_OK", False) or not globals().get("A3_PERSISTENCE_MASKS_OK", False):
    raise RuntimeError("Cần đạt Phần 4 và Phần 5 trước khi chấm dữ liệu thật.")

# Ngưỡng sàng lọc chênh số học float64 so với CSV đã tính từ chuỗi float32; không phải ngưỡng A2 của bài.
A3_CSV_ATOL, A3_CSV_RTOL = 5e-6, 5e-6
a3_score_started = time.perf_counter()
a3_score_rows, a3_clip_rows, a3_comparison_rows = [], [], []
A3_SCORED_SIM = {}

print("So CSV: |computed − stored| ≤", A3_CSV_ATOL, "+", A3_CSV_RTOL, "× |stored|, đồng thời phải khớp trạng thái hữu hạn/NaN.")
print("Ngưỡng chỉ sàng lọc sai khác số học; không tự nâng ngưỡng khi gặp ca lệch và không thay cho tái lập tác giả.")


def a3_score_scope(run, scope, observation_source, observed, predicted, mask, persistence, clip_applied):
    expected_shape = (len(A3_TEST_DATES), len(A3_SELECTED_IDS))
    frames = (observed, predicted, mask) + ((persistence,) if persistence is not None else ())
    for frame in frames:
        if (frame.shape != expected_shape or not frame.index.equals(A3_TEST_DATES)
                or list(frame.columns) != A3_SELECTED_IDS):
            raise ValueError(f"{run}/{scope}: shape/ngày/thứ tự ID khác hợp đồng")
    obs_values, sim_values, mask_values = observed.to_numpy(), predicted.to_numpy(), mask.to_numpy()
    reference_values = persistence.to_numpy() if persistence is not None else None
    for column, basin_id in enumerate(A3_SELECTED_IDS):
        result = a3_metrics(obs_values[:, column], sim_values[:, column],
            reference_values[:, column] if reference_values is not None else None, mask_values[:, column])
        if result["n"] != int(mask_values[:, column].sum()):
            raise AssertionError(f"{run}/{basin_id}/{scope}: hàm tự đổi số ngày chấm")
        persistence_nse = (1 - result["sse_persistence"] / result["obs_tss"]
            if np.isfinite(result["sse_persistence"]) and result["obs_tss"] > 0 else np.nan)
        a3_score_rows.append({"run": run, "scope": scope, "observation_source": observation_source,
            "basin_id": basin_id, "clip_to_zero": clip_applied, **result,
            "NSE_persistence_same_mask": persistence_nse})

# %% Phần 6.2 — Chấm sáu A2 và hai persistence
# Tính các chỉ số trên từng phạm vi đã định; không thay obs native bằng Q nguồn.
for run in EXPECTED_RUNS:
    for filename in ("test_metrics.csv", "config.yml"):
        path = A3_A2_DIR / "runs" / run / filename
        recorded = A3_INPUT_MANIFEST.loc[(A3_INPUT_MANIFEST.source == "A2_rerun_export")
            & (A3_INPUT_MANIFEST.run == run) & (A3_INPUT_MANIFEST.file == filename), "sha256"]
        if len(recorded) != 1 or a3_hash_file(path) != recorded.iloc[0]:
            raise ValueError(f"{run}/{filename}: nguồn khác manifest Phần 2")
    raw = A3_DAILY[run]["sim_raw"]
    clip_declared = "qmax" in a3_names(A3_CONFIGS[run].get("clip_targets_to_zero"))
    scored = pd.DataFrame(a3_sim_for_scoring(raw.to_numpy(), clip_declared), index=raw.index, columns=raw.columns)
    A3_SCORED_SIM[run] = scored
    if not np.array_equal(np.isfinite(raw.to_numpy()), np.isfinite(scored.to_numpy())):
        raise AssertionError(f"{run}: clip làm đổi mặt nạ hữu hạn")
    negative = raw.lt(0)
    a3_clip_rows.append({"run": run, "clip_qmax_config": clip_declared,
        "raw_negative_all_days": int(negative.to_numpy().sum()),
        "raw_negative_native_days": int((negative & A3_RUN_MASKS[run]["native"]).to_numpy().sum()),
        "values_clipped": int(negative.to_numpy().sum()) if clip_declared else 0,
        "negative_after_policy": int(scored.lt(0).to_numpy().sum())})
    for scope in ("native", "pnse_native", "paired_source", "common_source"):
        observed = A3_DAILY[run]["obs"] if scope in {"native", "pnse_native"} else A3_SOURCE_OBS
        mask = A3_COMMON_MASK if scope == "common_source" else A3_RUN_MASKS[run][scope]
        persistence = None if scope == "native" else A3_PERSISTENCE_QMAX
        a3_score_scope(run, scope, "A2_run_obs" if scope in {"native", "pnse_native"} else "source_Q_area_gov",
                       observed, scored, mask, persistence, clip_declared)
    print("Đã chấm:", run, "(4 scope)", flush=True)

a3_score_scope("Persistence_qmax", "native_source", "source_Q_area_gov", A3_SOURCE_OBS,
    A3_PERSISTENCE_QMAX, A3_BASELINE_MASKS["Persistence_qmax"], A3_PERSISTENCE_QMAX, False)
a3_score_scope("Persistence_qmax", "common_source", "source_Q_area_gov", A3_SOURCE_OBS,
    A3_PERSISTENCE_QMAX, A3_COMMON_MASK, A3_PERSISTENCE_QMAX, False)
a3_score_scope("Persistence_qmean_aux", "native_source", "source_Q_area_gov", A3_SOURCE_OBS,
    A3_PERSISTENCE_QMEAN_AUX, A3_BASELINE_MASKS["Persistence_qmean_aux"], None, False)
a3_aux_paired_mask = A3_BASELINE_MASKS["Persistence_qmean_aux"] & np.isfinite(A3_PERSISTENCE_QMAX)
a3_score_scope("Persistence_qmean_aux", "paired_source", "source_Q_area_gov", A3_SOURCE_OBS,
    A3_PERSISTENCE_QMEAN_AUX, a3_aux_paired_mask, A3_PERSISTENCE_QMAX, False)
a3_aux_common_mask = A3_COMMON_MASK & np.isfinite(A3_PERSISTENCE_QMEAN_AUX)
A3_AUX_FULL_COMMON = bool(np.array_equal(a3_aux_common_mask.to_numpy(), A3_COMMON_MASK.to_numpy()))
a3_score_scope("Persistence_qmean_aux", "common_source" if A3_AUX_FULL_COMMON else "common_source_aux_available",
    "source_Q_area_gov", A3_SOURCE_OBS, A3_PERSISTENCE_QMEAN_AUX, a3_aux_common_mask, A3_PERSISTENCE_QMAX, False)
A3_METRICS_PER_BASIN = pd.DataFrame(a3_score_rows)
A3_CLIP_AUDIT = pd.DataFrame(a3_clip_rows)

# %% Phần 6.3 — Đối chiếu NSE và KGE native với CSV
# So cả giá trị, số cặp và trạng thái hữu hạn/NaN trên cùng nguồn A2.
for run in EXPECTED_RUNS:
    stored = pd.read_csv(A3_A2_DIR / "runs" / run / "test_metrics.csv", dtype=str,
                         keep_default_na=False, na_filter=False)
    id_columns = [name for name in ("basin", "basin_id", "gauge_id") if name in stored]
    if len(id_columns) != 1 or not stored[id_columns[0]].str.strip().str.fullmatch(r"\d+").all():
        raise ValueError(f"{run}: không nhận diện duy nhất cột ID nguyên trong CSV")
    stored.index = stored[id_columns[0]].map(lambda value: str(int(value.strip())))
    if stored.index.has_duplicates or set(stored.index) != set(A3_SELECTED_IDS):
        raise ValueError(f"{run}: CSV không đúng tập ID độc nhất của mô hình")
    native = A3_METRICS_PER_BASIN.loc[(A3_METRICS_PER_BASIN.run == run)
        & (A3_METRICS_PER_BASIN.scope == "native")].set_index("basin_id", verify_integrity=True)
    for metric in ("NSE", "KGE"):
        columns = [name for name in stored if name.lower() in {metric.lower(), "qmax_" + metric.lower()}]
        if len(columns) != 1:
            raise ValueError(f"{run}: chưa nhận diện duy nhất {metric}, các cột: {list(stored.columns)}")
        text = stored[columns[0]].str.strip()
        values = pd.to_numeric(text, errors="coerce")
        if (values.isna() & ~text.str.lower().isin(["", "nan", "na", "n/a", "none", "null"])).any() or np.isinf(values.to_numpy()).any():
            raise ValueError(f"{run}/{metric}: CSV chứa token lạ hoặc vô cực")
        for basin_id in A3_SELECTED_IDS:
            saved, calculated = float(values.loc[basin_id]), float(native.at[basin_id, metric])
            both_finite = np.isfinite(saved) and np.isfinite(calculated)
            both_nan = np.isnan(saved) and np.isnan(calculated)
            state_matches = bool(np.isfinite(saved) == np.isfinite(calculated))
            close = bool(np.isclose(calculated, saved, atol=A3_CSV_ATOL, rtol=A3_CSV_RTOL)) if both_finite else both_nan
            a3_comparison_rows.append({"run": run, "basin_id": basin_id, "metric": metric,
                "csv_value": saved, "recomputed_float64": calculated, "finite_state_matches": state_matches,
                "abs_difference": abs(calculated - saved) if both_finite else np.nan,
                "within_tolerance": close, "atol": A3_CSV_ATOL, "rtol": A3_CSV_RTOL,
                "status": "matched" if close else "value_mismatch" if state_matches else "finite_state_mismatch"})
A3_NATIVE_METRIC_COMPARISON = pd.DataFrame(a3_comparison_rows)
a3_comparison_summary = []
for (run, metric), group in A3_NATIVE_METRIC_COMPARISON.groupby(["run", "metric"], sort=False):
    delta = group.abs_difference.dropna()
    a3_comparison_summary.append({"run": run, "metric": metric,
        "csv_finite_IDs": int(np.isfinite(group.csv_value).sum()),
        "recomputed_finite_IDs": int(np.isfinite(group.recomputed_float64).sum()),
        "state_mismatches": int((~group.finite_state_matches).sum()),
        "outside_tolerance": int((~group.within_tolerance & group.finite_state_matches).sum()),
        "max_abs_difference": float(delta.max()) if len(delta) else np.nan,
        "median_abs_difference": float(delta.median()) if len(delta) else np.nan,
        "csv_median": float(group.csv_value.median()), "recomputed_median": float(group.recomputed_float64.median()),
        "all_matched": bool(group.within_tolerance.all())})
A3_NATIVE_COMPARISON_SUMMARY = pd.DataFrame(a3_comparison_summary)
A3_NATIVE_CSV_REVIEW_OK = bool(A3_NATIVE_METRIC_COMPARISON.within_tolerance.all())

# %% Phần 6.4 — Tổng hợp điểm, kỹ năng và NaN
# Báo median/mean/phân vị, số ID hơn persistence và lý do không xác định.
a3_summary_rows, a3_skill_rows, a3_status_rows, a3_median_rows = [], [], [], []
for (run, scope, obs_source), group in A3_METRICS_PER_BASIN.groupby(["run", "scope", "observation_source"], sort=False):
    median_row = {"run": run, "scope": scope, "observation_source": obs_source,
                  "valid_day_IDs": int(group.n.sum()), "IDs_with_days": int(group.n.gt(0).sum())}
    for metric in A3_METRIC_NAMES:
        valid = group.loc[np.isfinite(group[metric]), metric]
        median = float(valid.median()) if len(valid) else np.nan
        row = {"run": run, "scope": scope, "observation_source": obs_source, "metric": metric,
               "finite_IDs": len(valid), "undefined_IDs": len(group) - len(valid), "median": median,
               "mean": float(valid.mean()) if len(valid) else np.nan}
        row.update({label: float(valid.quantile(probability)) if len(valid) else np.nan
                    for label, probability in (("p10", 0.1), ("p25", 0.25), ("p75", 0.75), ("p90", 0.9))})
        a3_summary_rows.append(row)
        median_row[metric + "_median"] = median
        median_row[metric + "_finite_IDs"] = len(valid)
        if metric == "NSE":
            median_row["NSE_mean"] = row["mean"]
        for status, count in group[metric + "_status"].value_counts().items():
            a3_status_rows.append({"run": run, "scope": scope, "metric": metric, "status": status, "IDs": int(count)})
    a3_median_rows.append(median_row)
    pnse = group.loc[np.isfinite(group.PNSE), "PNSE"]
    if len(pnse) or scope in {"paired_source", "common_source", "common_source_aux_available"}:
        a3_skill_rows.append({"run": run, "scope": scope, "finite_PNSE_IDs": len(pnse),
            "better_than_qmax_persistence": int(pnse.gt(0).sum()), "equal": int(pnse.eq(0).sum()),
            "worse": int(pnse.lt(0).sum()), "undefined": len(group) - len(pnse)})
A3_SUMMARY = pd.DataFrame(a3_summary_rows)
A3_MEDIAN_TABLE = pd.DataFrame(a3_median_rows)
A3_PAIRWISE_SKILL_SUMMARY = pd.DataFrame(a3_skill_rows)
A3_METRIC_STATUS = pd.DataFrame(a3_status_rows)

show_table('Chính sách clip và số giá trị bị tác động', a3_clip_rows)
show_table('Đối chiếu NSE/KGE native với CSV A2', a3_comparison_summary)
a3_failures = A3_NATIVE_METRIC_COMPARISON.loc[~A3_NATIVE_METRIC_COMPARISON.within_tolerance]
if not a3_failures.empty:
    show_table('Các ca chưa khớp — tối đa 15; đầy đủ trong CSV', a3_failures.sort_values("abs_difference", ascending=False).head(15).to_dict("records"))
median_columns = ["run", "scope", "IDs_with_days", "valid_day_IDs", "NSE_median", "NSE_mean",
                  "PNSE_median", "PNSE_finite_IDs", "KGE_median", "KGE_finite_IDs", "r_median", "alpha_median", "beta_median"]
show_table('Phạm vi riêng — có khác obs/ngày, không đọc như bảng ghép cặp', A3_MEDIAN_TABLE.loc[
    A3_MEDIAN_TABLE.scope.isin(["native", "native_source"]), median_columns].to_dict("records"))
show_table('So sánh trên cùng obs Q nguồn và giao ngày chung', A3_MEDIAN_TABLE.loc[
    A3_MEDIAN_TABLE.scope.eq("common_source"), median_columns].to_dict("records"))
print("Qmean phụ có đủ toàn bộ giao chung:", A3_AUX_FULL_COMMON)
print("Các median r/alpha/beta tính riêng từng thành phần; không dùng các median này để dựng lại median KGE.")
show_table('Số ID hơn/kém qmax persistence trên mặt nạ tương ứng', A3_PAIRWISE_SKILL_SUMMARY.loc[
    A3_PAIRWISE_SKILL_SUMMARY.scope.isin(["paired_source", "common_source", "common_source_aux_available"])].to_dict("records"))
show_table('Chỉ số không xác định trên phạm vi riêng/chung', A3_METRIC_STATUS.loc[
    A3_METRIC_STATUS.scope.isin(["native", "native_source", "common_source"])
    & A3_METRIC_STATUS.metric.isin(["NSE", "PNSE", "KGE"])
    & ~A3_METRIC_STATUS.status.isin(["ok", "persistence_not_provided"])].to_dict("records"))

# %% Phần 6.5 — Lưu bảng điểm A2 và metadata đánh giá
# Giữ kết quả float64 và trạng thái đối chiếu; không nới dung sai.
for filename, table in (
    ("A3_metrics_per_basin.csv", A3_METRICS_PER_BASIN), ("A3_summary.csv", A3_SUMMARY),
    ("A3_median_table.csv", A3_MEDIAN_TABLE),
    ("A3_pairwise_skill_summary.csv", A3_PAIRWISE_SKILL_SUMMARY), ("A3_clip_audit.csv", A3_CLIP_AUDIT),
    ("A3_native_metric_comparison.csv", A3_NATIVE_METRIC_COMPARISON),
    ("A3_native_metric_comparison_summary.csv", A3_NATIVE_COMPARISON_SUMMARY), ("A3_metric_status.csv", A3_METRIC_STATUS)):
    table.to_csv(A3_OUT_DIR / filename, index=False, encoding="utf-8-sig")
a3_evaluation_libs = {}
for package in ("numpy", "pandas", "xarray", "scipy", "neuralhydrology"):
    try:
        a3_evaluation_libs[package] = a3_metadata.version(package)
    except a3_metadata.PackageNotFoundError:
        a3_evaluation_libs[package] = None
(A3_OUT_DIR / "A3_evaluation_environment.json").write_text(json.dumps({
    "libs": a3_evaluation_libs, "metric_dtype": "float64", "source_A2_commit": EXPECTED_COMMIT,
    "csv_comparison_atol": A3_CSV_ATOL, "csv_comparison_rtol": A3_CSV_RTOL,
    "NH_reference_ran_and_matched": A3_NH_REFERENCE_OK, "evaluation_protocol": A3_EVALUATION_PROTOCOL,
    "native_CSV_review_ok": A3_NATIVE_CSV_REVIEW_OK, "DLinear_included": False,
}, ensure_ascii=False, indent=2), encoding="utf-8")
A3_BASELINE_EVAL_OK = A3_NATIVE_CSV_REVIEW_OK

print("A3_NATIVE_CSV_REVIEW_OK =", A3_NATIVE_CSV_REVIEW_OK, "| A3_BASELINE_EVAL_OK =", A3_BASELINE_EVAL_OK)
print("Đã lưu 8 CSV và metadata môi trường chấm vào:", A3_OUT_DIR)
print(f"Thời gian chấm/tổng hợp: {time.perf_counter() - a3_score_started:.1f} giây")
print("Kết quả hiện gồm persistence và sáu run A2; chưa có DLinear, chưa làm kiểm định hoặc kết luận nguyên nhân NSE âm.")

if not A3_BASELINE_EVAL_OK:
    print("CHỜ KIỂM: native chưa khớp CSV; chỉ chuyển sang chẩn đoán Phần 6.2, chưa chuyển huấn luyện.")

# %% Phần 6.6 — Hàm tham chiếu từ commit A2
# Giữ bốn hàm nguyên bản và thông tin nguồn cho phép replay kiểu số.
import warnings
from typing import List, Tuple

import xarray as a3_replay_xr
from scipy import stats as a3_replay_stats

A3_PRECISION_REPLAY_OK = False
A3_BASELINE_EVAL_OK = False
if not globals().get("A3_METRIC_TESTS_OK", False) or "A3_NATIVE_METRIC_COMPARISON" not in globals():
    raise RuntimeError("Cần output Phần 6.1 để chọn đúng các ca cần chẩn đoán.")

A3_PINNED_METRICS_COMMIT = "dda45fcf19ee7ffd481b9d192778539df46fb074"
A3_PINNED_METRICS_MODULE_SHA256 = "363c64c80464b6f6fed03310b451eddb9e4e4e4de06c19d7ba8ba917349ef4af"
A3_PINNED_FUNCTIONS_SHA256 = "f1639d3d647af42df66f45a2e0a98bb4be80c717306d607d04db8e47de2ee06c"
# Mã tham chiếu: giữ nguyên bốn hàm tại commit A2, đặt trong phạm vi riêng.
# SHA-256 phần trích ở trên được đối chiếu với git blob khi tạo notebook.
# Dùng hàm Python trực tiếp để trình soạn thảo hiển thị đúng code và thụt lề.
def _a3_build_pinned_reference():
    DataArray = a3_replay_xr.DataArray
    stats = a3_replay_stats

    def _validate_inputs(obs: DataArray, sim: DataArray):
        if obs.shape != sim.shape:
            raise RuntimeError("Shapes of observations and simulations must match")
    
        if (len(obs.shape) > 1) and (obs.shape[1] > 1):
            raise RuntimeError("Metrics only defined for time series (1d or 2d with second dimension 1)")
    
    def _mask_valid(obs: DataArray, sim: DataArray) -> Tuple[DataArray, DataArray]:
        # mask of invalid entries. NaNs in simulations can happen during validation/testing
        idx = (~sim.isnull()) & (~obs.isnull())
    
        obs = obs[idx]
        sim = sim[idx]
    
        return obs, sim
    
    def nse(obs: DataArray, sim: DataArray) -> float:
        r"""Calculate Nash-Sutcliffe Efficiency [#]_
        
        Nash-Sutcliffe Efficiency is the R-square between observed and simulated discharge.
        
        .. math:: \text{NSE} = 1 - \frac{\sum_{t=1}^{T}(Q_m^t - Q_o^t)^2}{\sum_{t=1}^T(Q_o^t - \overline{Q}_o)^2},
        
        where :math:`Q_m` are the simulations (here, `sim`) and :math:`Q_o` are observations (here, `obs`).
        
        Parameters
        ----------
        obs : DataArray
            Observed time series.
        sim : DataArray
            Simulated time series.
    
        Returns
        -------
        float
            Nash-Sutcliffe Efficiency 
            
        References
        ----------
        .. [#] Nash, J. E.; Sutcliffe, J. V. (1970). "River flow forecasting through conceptual models part I - A 
            discussion of principles". Journal of Hydrology. 10 (3): 282-290. doi:10.1016/0022-1694(70)90255-6.
    
        """
    
        # verify inputs
        _validate_inputs(obs, sim)
    
        # get time series with only valid observations
        obs, sim = _mask_valid(obs, sim)
    
        denominator = ((obs - obs.mean())**2).sum()
        numerator = ((sim - obs)**2).sum()
    
        value = 1 - numerator / denominator
    
        return float(value)
    
    def kge(obs: DataArray, sim: DataArray, weights: List[float] = [1., 1., 1.]) -> float:
        r"""Calculate the Kling-Gupta Efficieny [#]_
        
        .. math:: 
            \text{KGE} = 1 - \sqrt{[ s_r (r - 1)]^2 + [s_\alpha ( \alpha - 1)]^2 + 
                [s_\beta(\beta_{\text{KGE}} - 1)]^2},
                
        where :math:`r` is the correlation coefficient, :math:`\alpha` the :math:`\alpha`-NSE decomposition, 
        :math:`\beta_{\text{KGE}}` the fraction of the means and :math:`s_r, s_\alpha, s_\beta` the corresponding weights
        (here the three float values in the `weights` parameter).
        
        Parameters
        ----------
        obs : DataArray
            Observed time series.
        sim : DataArray
            Simulated time series.
        weights : List[float]
            Weighting factors of the 3 KGE parts, by default each part has a weight of 1.
    
        Returns
        -------
        float
            Kling-Gupta Efficiency
        
        References
        ----------
        .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error 
            and NSE performance criteria: Implications for improving hydrological modelling. Journal of hydrology, 377(1-2),
            80-91.
    
        """
        if len(weights) != 3:
            raise ValueError("Weights of the KGE must be a list of three values")
    
        # verify inputs
        _validate_inputs(obs, sim)
    
        # get time series with only valid observations
        obs, sim = _mask_valid(obs, sim)
    
        if len(obs) < 2:
            return np.nan
    
        r, _ = stats.pearsonr(obs.values, sim.values)
    
        alpha = sim.std() / obs.std()
        beta = sim.mean() / obs.mean()
    
        value = (weights[0] * (r - 1)**2 + weights[1] * (alpha - 1)**2 + weights[2] * (beta - 1)**2)
    
        return 1 - np.sqrt(float(value))

    return {
        "_validate_inputs": _validate_inputs,
        "_mask_valid": _mask_valid,
        "nse": nse,
        "kge": kge,
    }


a3_pinned_namespace = _a3_build_pinned_reference()


def a3_precision_case_ok(row):
    return bool(
        row.get("status") == "checked"
        and row.get("roundtrip_exact")
        and row.get("original_float32")
        and row.get("original_matches_csv")
        and row.get("float64_reference_matches_A3")
    )

# %% Phần 6.7 — Môi trường replay và các ca đối chứng
# Chọn các sai khác KGE đã phát hiện cùng ca đối chứng của từng run.
print("Commit tham chiếu:", A3_PINNED_METRICS_COMMIT)
print("SHA-256 phần trích đã kiểm với git blob khi tạo notebook:", A3_PINNED_FUNCTIONS_SHA256)
print("Giữ nguyên dung sai CSV:", A3_CSV_ATOL, A3_CSV_RTOL, "| dung sai ref64 so A3: atol=rtol=1e-10")
a3_backend_options = {name: a3_replay_xr.get_options().get(name) for name in ("use_bottleneck", "use_numbagg", "use_flox")}
print("Xarray options hiện tại (không đổi để tìm điểm khớp):", a3_backend_options)
a3_replay_versions = {}
for package in ("numpy", "xarray", "scipy", "bottleneck", "numbagg", "flox"):
    try:
        a3_replay_versions[package] = a3_metadata.version(package)
    except a3_metadata.PackageNotFoundError:
        a3_replay_versions[package] = None
print("Phiên bản môi trường replay:", a3_replay_versions)

a3_unmatched = A3_NATIVE_METRIC_COMPARISON.loc[~A3_NATIVE_METRIC_COMPARISON.within_tolerance].copy()
a3_replay_eligible = bool(A3_NATIVE_METRIC_COMPARISON.finite_state_matches.all()
    and a3_unmatched.metric.eq("KGE").all())
a3_replay_cases = [{"run": row.run, "basin_id": row.basin_id, "role": "flagged"}
                   for row in a3_unmatched.itertuples()]
# Mỗi run thêm một ca đã khớp có chênh KGE lớn nhất làm đối chứng, không chọn nguồn/cách tính theo điểm tốt.
for run in EXPECTED_RUNS:
    controls = A3_NATIVE_METRIC_COMPARISON.loc[A3_NATIVE_METRIC_COMPARISON.run.eq(run)
        & A3_NATIVE_METRIC_COMPARISON.metric.eq("KGE") & A3_NATIVE_METRIC_COMPARISON.within_tolerance
        & np.isfinite(A3_NATIVE_METRIC_COMPARISON.csv_value)]
    if not controls.empty:
        control = controls.sort_values("abs_difference", ascending=False).iloc[0]
        a3_replay_cases.append({"run": run, "basin_id": control.basin_id, "role": "control"})
show_table('Các ca cần replay và đối chứng', a3_replay_cases)
print("Chỉ có sai khác KGE hữu hạn, không có lỗi trạng thái/NSE:", a3_replay_eligible)

# %% Phần 6.8 — Replay KGE với kiểu gốc và float64
# Giữ cùng ngày/obs/sim/clip để phân biệt tác động số học.
a3_replay_rows, a3_component_rows = [], []
for case in a3_replay_cases:
    run, basin_id = case["run"], case["basin_id"]
    row = {**case, "status": "not_checked", "roundtrip_exact": False, "original_float32": False,
           "original_matches_csv": False, "float64_reference_matches_A3": False, "issues": ""}
    try:
        schema = A3_SCHEMA_AUDIT.loc[A3_SCHEMA_AUDIT.run.eq(run) & A3_SCHEMA_AUDIT.basin_id.eq(basin_id)]
        if len(schema) != 1 or schema.iloc[0].status != "ok":
            raise ValueError("Không có schema gốc duy nhất đã đạt ở Phần 3.1")
        obs_dtype, sim_dtype = np.dtype(schema.iloc[0].obs_dtype), np.dtype(schema.iloc[0].sim_dtype)
        if obs_dtype.kind != "f" or sim_dtype.kind != "f":
            raise TypeError("Kiểu nguồn không phải số thực dấu phẩy động")
        obs_raw = A3_DAILY[run]["obs"][basin_id].to_numpy()
        sim_raw = A3_DAILY[run]["sim_raw"][basin_id].to_numpy()
        row.update(obs_original_dtype=str(obs_dtype), sim_original_dtype=str(sim_dtype),
            original_float32=obs_dtype == np.dtype("float32") and sim_dtype == np.dtype("float32"),
            roundtrip_exact=bool(np.array_equal(obs_raw, obs_raw.astype(obs_dtype).astype(np.float64), equal_nan=True)
                and np.array_equal(sim_raw, sim_raw.astype(sim_dtype).astype(np.float64), equal_nan=True)))
        # Đọc lại config/CSV hash ở đây để không giải thích một bảng đã đổi nguồn.
        for filename in ("config.yml", "test_metrics.csv"):
            path = A3_A2_DIR / "runs" / run / filename
            expected = A3_INPUT_MANIFEST.loc[A3_INPUT_MANIFEST.run.eq(run)
                & A3_INPUT_MANIFEST.source.eq("A2_rerun_export") & A3_INPUT_MANIFEST.file.eq(filename), "sha256"]
            if len(expected) != 1 or a3_hash_file(path) != expected.iloc[0]:
                raise ValueError(f"Nguồn {filename} khác manifest")
        original_record = A3_NATIVE_METRIC_COMPARISON.loc[A3_NATIVE_METRIC_COMPARISON.run.eq(run)
            & A3_NATIVE_METRIC_COMPARISON.basin_id.eq(basin_id) & A3_NATIVE_METRIC_COMPARISON.metric.eq("KGE")]
        if len(original_record) != 1:
            raise ValueError("Không tìm thấy duy nhất KGE native đã so CSV")
        stored = float(original_record.iloc[0].csv_value)
        computed64 = float(original_record.iloc[0].recomputed_float64)
        row.update(csv_KGE=stored, A3_float64_KGE=computed64)
        reference_values = {}
        for mode, odtype, sdtype in (("original", obs_dtype, sim_dtype), ("float64", np.dtype("float64"), np.dtype("float64"))):
            observed = a3_replay_xr.DataArray(obs_raw.astype(odtype), dims=["date"], coords={"date": A3_TEST_DATES})
            predicted = a3_replay_xr.DataArray(sim_raw.astype(sdtype), dims=["date"], coords={"date": A3_TEST_DATES})
            if "qmax" in a3_names(A3_CONFIGS[run].get("clip_targets_to_zero")):
                predicted = a3_replay_xr.where(predicted < 0, 0, predicted)
            mask = np.isfinite(observed.values) & np.isfinite(predicted.values)
            if not np.array_equal(mask, A3_RUN_MASKS[run]["native"][basin_id].to_numpy()):
                raise ValueError("Replay làm đổi mặt nạ native")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reference_values[mode] = a3_pinned_namespace["kge"](observed, predicted)
                masked_obs, masked_sim = a3_pinned_namespace["_mask_valid"](observed, predicted)
                correlation = a3_replay_stats.pearsonr(masked_obs.values, masked_sim.values).statistic
                alpha = masked_sim.std() / masked_obs.std()
                beta = masked_sim.mean() / masked_obs.mean()
            a3_component_rows.append({**case, "mode": mode, "n": int(mask.sum()),
                "obs_dtype": str(observed.dtype), "sim_dtype_after_clip": str(predicted.dtype),
                "r": float(correlation), "alpha": float(alpha), "beta": float(beta),
                "obs_mean": float(masked_obs.mean()), "sim_mean": float(masked_sim.mean()),
                "obs_std": float(masked_obs.std()), "sim_std": float(masked_sim.std()),
                "r_dtype": str(np.asarray(correlation).dtype), "alpha_dtype": str(alpha.dtype), "beta_dtype": str(beta.dtype),
                "KGE_pinned": reference_values[mode], "warnings": " | ".join(str(item.message) for item in caught)})
        row.update(status="checked", pinned_original_KGE=reference_values["original"],
            pinned_float64_KGE=reference_values["float64"],
            abs_original_vs_csv=abs(reference_values["original"] - stored),
            abs_float64ref_vs_A3=abs(reference_values["float64"] - computed64),
            original_matches_csv=bool(np.isclose(reference_values["original"], stored, atol=A3_CSV_ATOL, rtol=A3_CSV_RTOL)),
            float64_reference_matches_A3=bool(np.isclose(reference_values["float64"], computed64, atol=1e-10, rtol=1e-10)))
    except Exception as exc:
        row.update(status="replay_error", issues=f"{type(exc).__name__}: {exc}")
    row["precision_explained"] = a3_precision_case_ok(row)
    a3_replay_rows.append(row)

A3_KGE_PRECISION_REPLAY = pd.DataFrame(a3_replay_rows)
A3_KGE_COMPONENT_REPLAY = pd.DataFrame(a3_component_rows)
show_table('Replay kiểu gốc so CSV và replay float64 so hàm A3', a3_replay_rows)
show_table('Thành phần KGE ở các ca lệch (không phải toàn bộ lưu vực)',
    A3_KGE_COMPONENT_REPLAY.loc[A3_KGE_COMPONENT_REPLAY.role.eq("flagged")].to_dict("records") if not A3_KGE_COMPONENT_REPLAY.empty else [])

# %% Phần 6.9 — Kết luận kiểm sai khác số học
# Lưu replay và giữ riêng cờ so CSV nghiêm ngặt với cờ giải thích kiểu số.
A3_KGE_PRECISION_REPLAY.to_csv(A3_OUT_DIR / "A3_KGE_precision_replay.csv", index=False, encoding="utf-8-sig")
A3_KGE_COMPONENT_REPLAY.to_csv(A3_OUT_DIR / "A3_KGE_component_replay.csv", index=False, encoding="utf-8-sig")
A3_PRECISION_REPLAY_OK = bool(a3_replay_eligible and len(a3_replay_rows)
    and A3_KGE_PRECISION_REPLAY.precision_explained.all())
A3_BASELINE_EVAL_OK = bool(A3_NATIVE_CSV_REVIEW_OK or A3_PRECISION_REPLAY_OK)
a3_replay_report = {"pinned_commit": A3_PINNED_METRICS_COMMIT, "module_sha256": A3_PINNED_METRICS_MODULE_SHA256,
    "extracted_functions_sha256": A3_PINNED_FUNCTIONS_SHA256, "unmatched_count": len(a3_unmatched),
    "case_count": len(a3_replay_rows), "xarray_options": a3_backend_options,
    "replay_versions": a3_replay_versions,
    "numpy_version": np.__version__, "xarray_version": a3_replay_xr.__version__,
    "scipy_version": a3_metadata.version("scipy"), "source_A2_libs": a3_env.get("libs", {}),
    "CSV_atol_unchanged": A3_CSV_ATOL, "CSV_rtol_unchanged": A3_CSV_RTOL,
    "strict_float64_CSV_review_ok": bool(A3_NATIVE_CSV_REVIEW_OK),
    "precision_replay_ok": A3_PRECISION_REPLAY_OK, "baseline_evaluation_ok": A3_BASELINE_EVAL_OK,
    "main_tables_modified": False, "note": "Replay only explains arithmetic on selected native mismatches and controls; main scoring remains float64. No original A2 backend setting is inferred solely from replay."}
(A3_OUT_DIR / "A3_KGE_precision_report.json").write_text(json.dumps(a3_replay_report, ensure_ascii=False, indent=2), encoding="utf-8")

print("Strict float64 so CSV vẫn giữ:", A3_NATIVE_CSV_REVIEW_OK, "| A3_PRECISION_REPLAY_OK =", A3_PRECISION_REPLAY_OK)
print("A3_BASELINE_EVAL_OK =", A3_BASELINE_EVAL_OK)
print("Đã lưu 2 CSV và 1 JSON replay; giữ nguyên bảng điểm float64 và dung sai.")

if not A3_BASELINE_EVAL_OK:
    raise RuntimeError("Replay chưa giải thích đủ sai khác; chưa chuyển DLinear. Gửi output để kiểm tiếp, không nới dung sai.")

# %% Phần 7.1 — Lịch DLinear và hàm kiểm cửa sổ
# Dùng 365 ngày qmean trước ngày đích; kiểm ví dụ không lấy qmean(t).
from io import BytesIO

if not globals().get("A3_BASELINE_EVAL_OK", False):
    raise RuntimeError("Cần chấm mốc và giải thích sai khác native ở Phần 6 trước khi khảo sát DLinear.")

A3_DLINEAR_DATA_AUDIT_OK = False
A3_DLINEAR_LOOKBACK = 365
A3_DLINEAR_PROFILE_IDS = ["55", "324", "600", "623", "742", "758", "800"]
A3_DLINEAR_CALENDAR = pd.date_range(
    pd.Timestamp(A3_PERIODS["train"][0]) - pd.Timedelta(days=A3_DLINEAR_LOOKBACK),
    A3_PERIODS["test"][1], freq="D", name="date",
)
A3_DLINEAR_Q_NUMERIC = {
    name: pd.DataFrame(np.nan, index=A3_DLINEAR_CALENDAR, columns=A3_SELECTED_IDS)
    for name in A3_Q_VARIABLES
}


def a3_complete_past_mask(values, lookback):
    """Ngày i chỉ dùng values[i-lookback:i]; không đọc giá trị ngày đích."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("Cần chuỗi 1D và lookback nguyên dương")
    valid = np.isfinite(values) & (values >= 0)
    prefix = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    complete = np.zeros(len(values), dtype=bool)
    targets = np.arange(lookback, len(values))
    complete[targets] = (prefix[targets] - prefix[targets - lookback]) == lookback
    return complete


def a3_consumed_train_days(target_mask, lookback, train_calendar_mask):
    """Hợp ngày đầu vào của các mẫu train; đếm mỗi ngày một lần khi khảo sát fit."""
    targets = np.flatnonzero(target_mask)
    if targets.size and targets.min() < lookback:
        raise ValueError("Mẫu train chưa có đủ lịch ngữ cảnh")
    changes = np.zeros(len(target_mask) + 1, dtype=np.int64)
    np.add.at(changes, targets - lookback, 1)
    np.add.at(changes, targets, -1)
    consumed = np.cumsum(changes[:-1]) > 0
    return consumed & train_calendar_mask, int((consumed & ~train_calendar_mask).sum())


def a3_train_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0)]
    result = {"n": int(values.size), "zeros": int((values == 0).sum())}
    names = ("min", "p01", "p05", "median", "p95", "p99", "max")
    quantiles = np.quantile(values, [0, .01, .05, .5, .95, .99, 1]) if values.size else [np.nan] * 7
    result.update(zip(names, map(float, quantiles)))
    result["mean"] = float(values.mean()) if values.size else np.nan
    result["std_ddof0"] = float(values.std(ddof=0)) if values.size else np.nan
    return result


# Kiểm phép đánh chỉ mục bằng cách duyệt từng cửa sổ độc lập, gồm lỗ thiếu và ngày đích thiếu.
a3_window_example = np.array([1., 2., 3., np.nan, 5., 6., 7., 8.])
a3_window_expected = np.array([
    i >= 3 and np.isfinite(a3_window_example[i - 3:i]).all()
    for i in range(len(a3_window_example))
])
if not np.array_equal(a3_complete_past_mask(a3_window_example, 3), a3_window_expected):
    raise AssertionError("Sai mặt nạ quá khứ trong ví dụ có khoảng thiếu")
if not a3_complete_past_mask(a3_window_example, 3)[3]:
    raise AssertionError("Giá trị ngày đích đã bị dùng nhầm làm đầu vào")


print("Lịch nguồn:", A3_DLINEAR_CALENDAR[0].date(), "→", A3_DLINEAR_CALENDAR[-1].date())
print("Đầu vào ứng viên: qmean(t−365…t−1); nhãn: qmax(t). Không dùng qmean(t).")
print("Ví dụ kiểm chỉ mục đạt. Chỉ khảo sát cửa sổ đủ dữ liệu, chưa chốt scaler hoặc quy tắc điền thiếu.")

# %% Phần 7.2 — Đọc và kiểm Q train, validation, test
# Đối chiếu hash nguồn và mã thiếu trong lịch 2002–2017.
a3_dlinear_started = time.perf_counter()
a3_dl_file_rows, a3_dl_value_rows = [], []

# Kiểm area_gov chưa đổi so manifest đã dùng cho persistence và các obs nguồn.
a3_area_manifest = A3_RAW_INPUT_MANIFEST.loc[A3_RAW_INPUT_MANIFEST.source.eq("raw_gauge_attributes")]
if len(a3_area_manifest) != 1 or a3_hash_file(Path(a3_area_manifest.iloc[0].path)) != a3_area_manifest.iloc[0].sha256:
    raise ValueError("Nguồn diện tích không còn khớp manifest Phần 3.2")

for number, basin_id in enumerate(A3_SELECTED_IDS, 1):
    row = {"basin_id": basin_id, "status": "not_read", "hash_matches_part3": False, "issues": ""}
    try:
        path = a3_daily_paths[basin_id]
        expected = A3_RAW_INPUT_MANIFEST.loc[
            A3_RAW_INPUT_MANIFEST.source.eq("raw_daily_Q")
            & A3_RAW_INPUT_MANIFEST.path.eq(str(path)), "sha256"
        ]
        # Đọc byte một lần: cùng byte dùng tính hash và parse, tránh mở lại tệp Q để hash.
        content = path.read_bytes()
        row["hash_matches_part3"] = bool(
            len(expected) == 1 and hashlib.sha256(content).hexdigest() == expected.iloc[0]
        )
        if not row["hash_matches_part3"]:
            raise ValueError("Tệp Q khác manifest Phần 3.2")
        raw = pd.read_csv(
            BytesIO(content), sep=";", dtype=str, encoding="utf-8-sig",
            keep_default_na=False, na_filter=False, usecols=["YYYY", "MM", "DD", *A3_Q_VARIABLES],
        )
        parts = raw[["YYYY", "MM", "DD"]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(parts.to_numpy()).all() or not parts.eq(np.floor(parts)).all().all():
            raise ValueError("Thành phần ngày không hữu hạn hoặc không nguyên")
        dates = pd.DatetimeIndex(pd.to_datetime(
            parts.rename(columns={"YYYY": "year", "MM": "month", "DD": "day"}), errors="raise",
        ))
        if dates.empty or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Lịch nguồn rỗng/NaT/trùng/không tăng")
        issues = []
        for variable in A3_Q_VARIABLES:
            text = raw[variable].str.strip()
            numeric = pd.to_numeric(text, errors="coerce")
            known_na = text.str.lower().isin(["", "nan", "na", "n/a", "null", "none"])
            values = pd.Series(numeric.to_numpy(dtype=float), index=dates).reindex(A3_DLINEAR_CALENDAR)
            tokens = pd.Series(text.to_numpy(), index=dates).reindex(A3_DLINEAR_CALENDAR)
            unknown = pd.Series((numeric.isna() & ~known_na).to_numpy(), index=dates).reindex(
                A3_DLINEAR_CALENDAR, fill_value=False,
            )
            A3_DLINEAR_Q_NUMERIC[variable].loc[:, basin_id] = values.to_numpy()
            if unknown.any() or np.isinf(values.to_numpy()).any() or values.lt(0).any():
                issues.append(f"{variable}_tokens_infinity_or_negative_need_review")
            for period, bounds in {"context_before_train": (
                str(A3_DLINEAR_CALENDAR[0].date()),
                str((pd.Timestamp(A3_PERIODS["train"][0]) - pd.Timedelta(days=1)).date()),
            ), **A3_PERIODS}.items():
                calendar = pd.date_range(*bounds, freq="D")
                sub, token = values.loc[calendar], tokens.loc[calendar]
                a3_dl_value_rows.append({
                    "basin_id": basin_id, "variable": variable, "period": period,
                    "calendar_days": len(calendar), "absent_rows": int((~calendar.isin(dates)).sum()),
                    "blank_cells": int(token.eq("").sum()),
                    "explicit_na_tokens": int((token.notna() & token.str.lower().isin(
                        ["nan", "na", "n/a", "null", "none"])).sum()),
                    "minus999_cells": int(sub.eq(-999).sum()),
                    "other_negative_cells": int((sub.lt(0) & sub.ne(-999)).sum()),
                    "unexpected_tokens": int(unknown.loc[calendar].sum()),
                    "infinite_cells": int(np.isinf(sub.to_numpy()).sum()),
                    "finite_nonnegative": int((np.isfinite(sub) & sub.ge(0)).sum()),
                })
        row.update(status="ok" if not issues else "needs_review", issues=" | ".join(issues))
    except Exception as exc:
        row.update(status="read_error", issues=f"{type(exc).__name__}: {exc}")
    a3_dl_file_rows.append(row)
    if number % 100 == 0 or number == len(A3_SELECTED_IDS):
        print(f"Đã đọc Q cho DLinear {number}/{len(A3_SELECTED_IDS)} ID", flush=True)

A3_DLINEAR_FILE_AUDIT = pd.DataFrame(a3_dl_file_rows)
A3_DLINEAR_VALUE_AUDIT = pd.DataFrame(a3_dl_value_rows)
A3_DLINEAR_FILE_AUDIT.to_csv(A3_OUT_DIR / "A3_DLinear_file_audit.csv", index=False, encoding="utf-8-sig")
A3_DLINEAR_VALUE_AUDIT.to_csv(A3_OUT_DIR / "A3_DLinear_value_audit.csv", index=False, encoding="utf-8-sig")
show_table('Kiểm nguồn và các mã thiếu theo kỳ', [
    {"period": period, "variable": variable, "IDs_read": len(group), **{
        key: int(group[key].sum()) for key in group.columns if key not in ("basin_id", "variable", "period")
    }}
    for (period, variable), group in A3_DLINEAR_VALUE_AUDIT.groupby(["period", "variable"], sort=False)
] if not A3_DLINEAR_VALUE_AUDIT.empty else [])
bad_files = A3_DLINEAR_FILE_AUDIT.loc[A3_DLINEAR_FILE_AUDIT.status.ne("ok")]
if not bad_files.empty:
    show_table('Nguồn cần kiểm — tối đa 20 dòng, đầy đủ trong CSV', bad_files.head(20).to_dict("records"))
    raise RuntimeError("Giữ audit và gửi output: chưa chọn cách xử lý token/giá trị bất thường cho DLinear.")

# %% Phần 7.3 — Đếm lịch sử đầy đủ và mẫu từng kỳ
# Tách ngày có thể suy luận khỏi ngày đủ nhãn để train/chấm.
for variable in A3_Q_VARIABLES:
    if not np.array_equal(
        A3_DLINEAR_Q_NUMERIC[variable].loc[A3_Q_CALENDAR].to_numpy(),
        A3_Q_NUMERIC[variable].to_numpy(), equal_nan=True,
    ):
        raise ValueError(f"{variable}: đọc lại cửa sổ test khác dữ liệu Phần 3.2")

A3_DLINEAR_Q_MM_DAY = {
    name: frame.mul(86.4 / A3_AREAS, axis="columns")
    for name, frame in A3_DLINEAR_Q_NUMERIC.items()
}
A3_DLINEAR_HISTORY_VALID = pd.DataFrame({
    basin_id: a3_complete_past_mask(A3_DLINEAR_Q_NUMERIC["qmean"][basin_id], A3_DLINEAR_LOOKBACK)
    for basin_id in A3_SELECTED_IDS
}, index=A3_DLINEAR_CALENDAR)
A3_DLINEAR_SAMPLE_MASKS = {}
a3_dl_count_rows = []
for period, bounds in A3_PERIODS.items():
    calendar = pd.date_range(*bounds, freq="D")
    label = A3_DLINEAR_Q_NUMERIC["qmax"].loc[calendar]
    has_label = np.isfinite(label) & label.ge(0)
    history = A3_DLINEAR_HISTORY_VALID.loc[calendar]
    samples = history & has_label
    A3_DLINEAR_SAMPLE_MASKS[period] = samples
    for basin_id in A3_SELECTED_IDS:
        valid_dates = calendar[samples[basin_id].to_numpy()]
        a3_dl_count_rows.append({
            "period": period, "basin_id": basin_id, "calendar_days": len(calendar),
            "label_days": int(has_label[basin_id].sum()),
            "inference_input_days": int(history[basin_id].sum()),
            "candidate_sample_days": len(valid_dates),
            "labels_lost_for_incomplete_history": int((has_label[basin_id] & ~history[basin_id]).sum()),
            "first_candidate_target": str(valid_dates[0].date()) if len(valid_dates) else "",
            "last_candidate_target": str(valid_dates[-1].date()) if len(valid_dates) else "",
        })
A3_DLINEAR_WINDOW_COUNTS = pd.DataFrame(a3_dl_count_rows)
show_table('Cửa sổ đủ qmean(t−365…t−1) và nhãn qmax(t)', [
    {"period": period, "calendar_day_IDs": int(group.calendar_days.sum()),
     "label_day_IDs": int(group.label_days.sum()),
     "inference_input_day_IDs": int(group.inference_input_days.sum()),
     "candidate_sample_day_IDs": int(group.candidate_sample_days.sum()),
     "labels_lost_for_incomplete_history": int(group.labels_lost_for_incomplete_history.sum()),
     "IDs_without_labels": int(group.label_days.eq(0).sum()),
     "IDs_without_candidate_samples": int(group.candidate_sample_days.eq(0).sum()),
     "IDs_without_candidate_samples_list": ", ".join(group.loc[group.candidate_sample_days.eq(0), "basin_id"])}
    for period, group in A3_DLINEAR_WINDOW_COUNTS.groupby("period", sort=False)
])
print("Ngày có đủ đầu vào có thể suy luận dù thiếu nhãn; chỉ candidate_sample_days dùng chấm hoặc train.")

# %% Phần 7.4 — Thống kê train và các hồ sơ A1
# Mô tả phạm vi fit train và những ID ít/không có nhãn hoặc mẫu train.
a3_train_calendar_mask = (
    (A3_DLINEAR_CALENDAR >= pd.Timestamp(A3_PERIODS["train"][0]))
    & (A3_DLINEAR_CALENDAR <= pd.Timestamp(A3_PERIODS["train"][1]))
)
A3_DLINEAR_TRAIN_USAGE_MASKS = {
    "qmean": pd.DataFrame(False, index=A3_DLINEAR_CALENDAR, columns=A3_SELECTED_IDS),
    "qmax": A3_DLINEAR_SAMPLE_MASKS["train"].reindex(A3_DLINEAR_CALENDAR, fill_value=False),
}
a3_context_consumed = 0
for basin_id in A3_SELECTED_IDS:
    consumed, context_count = a3_consumed_train_days(
        A3_DLINEAR_TRAIN_USAGE_MASKS["qmax"][basin_id].to_numpy(),
        A3_DLINEAR_LOOKBACK, a3_train_calendar_mask,
    )
    A3_DLINEAR_TRAIN_USAGE_MASKS["qmean"].loc[:, basin_id] = consumed
    a3_context_consumed += context_count

a3_dl_stat_rows, a3_dl_profile_rows = [], []
for unit, frames in (("m3/s", A3_DLINEAR_Q_NUMERIC), ("mm/day_area_gov", A3_DLINEAR_Q_MM_DAY)):
    for variable in A3_Q_VARIABLES:
        values = frames[variable].to_numpy()
        for scope, mask in (
            ("all_finite_train_days", np.broadcast_to(a3_train_calendar_mask[:, None], values.shape)),
            ("unique_train_days_used_by_candidate_samples", A3_DLINEAR_TRAIN_USAGE_MASKS[variable].to_numpy()),
        ):
            a3_dl_stat_rows.append({"unit": unit, "variable": variable, "fit_scope_candidate": scope,
                                    **a3_train_distribution(values[mask])})
        for basin_id in A3_SELECTED_IDS:
            series = frames[variable].loc[a3_train_calendar_mask, basin_id].to_numpy()
            a3_dl_profile_rows.append({"basin_id": basin_id, "unit": unit, "variable": variable,
                                       **a3_train_distribution(series)})
A3_DLINEAR_TRAIN_DISTRIBUTION = pd.DataFrame(a3_dl_stat_rows)
A3_DLINEAR_TRAIN_BY_BASIN = pd.DataFrame(a3_dl_profile_rows)
show_table('Phân phối chỉ trên train — hai phạm vi fit ứng viên', a3_dl_stat_rows)
print("Thống kê gộp theo từng ngày–ID, float64, std ddof=0; không lấy trung bình mean/std của lưu vực.")
print("Ngày ngữ cảnh trước train được cửa sổ train sử dụng nhưng loại khỏi cả hai phạm vi fit:", a3_context_consumed)
print("Không so điểm validation/test để chọn đơn vị, biến đổi hoặc cách xử lý thiếu trong cell này.")

show_table('Bảy hồ sơ A1 — số nhãn và cửa sổ từng kỳ',
           A3_DLINEAR_WINDOW_COUNTS.loc[A3_DLINEAR_WINDOW_COUNTS.basin_id.isin(A3_DLINEAR_PROFILE_IDS)].to_dict("records"))
show_table('Bảy hồ sơ A1 — phân phối train mm/ngày',
           A3_DLINEAR_TRAIN_BY_BASIN.loc[
               A3_DLINEAR_TRAIN_BY_BASIN.basin_id.isin(A3_DLINEAR_PROFILE_IDS)
               & A3_DLINEAR_TRAIN_BY_BASIN.unit.eq("mm/day_area_gov")
           ].to_dict("records"))

# %% Phần 7.5 — Lưu dữ liệu và kiểm độ phủ DLinear
# Lưu bảng/cửa sổ và kiểm tác động dự kiến lên giao test.
A3_DLINEAR_CANDIDATE_COMMON_MASK = A3_COMMON_MASK & A3_DLINEAR_SAMPLE_MASKS["test"]
show_table('Tác động dự kiến lên giao test nếu chỉ dùng cửa sổ đủ dữ liệu', [{
    "current_A2_persistence_common_day_IDs": int(A3_COMMON_MASK.to_numpy().sum()),
    "candidate_with_DLinear_common_day_IDs": int(A3_DLINEAR_CANDIDATE_COMMON_MASK.to_numpy().sum()),
    "days_lost": int((A3_COMMON_MASK & ~A3_DLINEAR_CANDIDATE_COMMON_MASK).to_numpy().sum()),
    "candidate_IDs_with_pairs": int(A3_DLINEAR_CANDIDATE_COMMON_MASK.any(axis=0).sum()),
}])
print("Đây là mặt nạ dữ liệu ứng viên, chưa có dự báo DLinear; không sửa A3_COMMON_MASK hiện tại.")

for name, frame in (
    ("window_counts", A3_DLINEAR_WINDOW_COUNTS),
    ("train_distribution", A3_DLINEAR_TRAIN_DISTRIBUTION),
    ("train_by_basin", A3_DLINEAR_TRAIN_BY_BASIN),
):
    frame.to_csv(A3_OUT_DIR / f"A3_DLinear_{name}.csv", index=False, encoding="utf-8-sig")
np.savez_compressed(
    A3_OUT_DIR / "A3_DLinear_data_audit.npz",
    dates=A3_DLINEAR_CALENDAR.to_numpy(dtype="datetime64[D]"),
    basin_ids=np.asarray(A3_SELECTED_IDS, dtype="U"), area_gov_km2=A3_AREAS.to_numpy(),
    qmean_numeric_m3_s=A3_DLINEAR_Q_NUMERIC["qmean"].to_numpy(),
    qmax_numeric_m3_s=A3_DLINEAR_Q_NUMERIC["qmax"].to_numpy(),
    complete_past_input=A3_DLINEAR_HISTORY_VALID.to_numpy(),
    candidate_used_train_input=A3_DLINEAR_TRAIN_USAGE_MASKS["qmean"].to_numpy(),
    candidate_used_train_label=A3_DLINEAR_TRAIN_USAGE_MASKS["qmax"].to_numpy(),
)
A3_DLINEAR_DATA_AUDIT_OK = bool(
    len(A3_DLINEAR_FILE_AUDIT) == len(A3_SELECTED_IDS)
    and A3_DLINEAR_FILE_AUDIT.status.eq("ok").all()
)
(A3_OUT_DIR / "A3_DLinear_data_protocol.json").write_text(json.dumps({
    "lookback": A3_DLINEAR_LOOKBACK, "periods": A3_PERIODS,
    "source_start": str(A3_DLINEAR_CALENDAR[0].date()),
    "source_end": str(A3_DLINEAR_CALENDAR[-1].date()),
    "input": "qmean[t-365:t] excluding target day", "target": "qmax[t]",
    "audit_candidate": "complete finite nonnegative input and finite nonnegative label",
    "original_numeric_preserved": True, "scaler_fitted": False,
    "context_days_excluded_from_fit": True, "std_ddof_for_audit": 0,
    "data_audit_ok": A3_DLINEAR_DATA_AUDIT_OK,
    "source_manifest": "A3_raw_Q_manifest.csv from Part 3.2; hashes rechecked",
}, ensure_ascii=False, indent=2), encoding="utf-8")


print("A3_DLINEAR_DATA_AUDIT_OK =", A3_DLINEAR_DATA_AUDIT_OK)
print("Đã lưu 5 CSV, 1 NPZ và 1 JSON. Chưa fit scaler, điền thiếu hoặc huấn luyện DLinear.")
print(f"Thời gian đọc/kiểm: {time.perf_counter() - a3_dlinear_started:.1f} giây | Output: {A3_OUT_DIR}")

# %% Phần 7.6 — Hàm scaler, đổi ngược và loss tham chiếu
# Fit từ train; inverse scaling trước clip và giữ quy ước NSE* của NH.
if not globals().get("A3_DLINEAR_DATA_AUDIT_OK", False):
    raise RuntimeError("Cần kiểm và đạt Phần 7.1 trước khi cố định tiền xử lý DLinear.")

A3_DLINEAR_PREPROCESSING_OK = False
A3_DLINEAR_LOSS_EPS = 0.1  # Cùng quy ước NH: cộng vào std qmax train theo mm/ngày.
A3_DLINEAR_FIT_SCOPE = "all_finite_train_days"
a3_preparation_started = time.perf_counter()


def a3_fit_global_scaler(values, train_days):
    values = np.asarray(values, dtype=np.float64)
    train_days = np.asarray(train_days)
    if values.ndim != 2 or train_days.dtype != bool or train_days.shape != (len(values),):
        raise ValueError("Cần ma trận ngày–ID và mask ngày train boolean đúng kích thước")
    selected = values[train_days]
    if np.isinf(selected).any() or np.any(selected[np.isfinite(selected)] < 0):
        raise ValueError("Dữ liệu fit có vô cực hoặc giá trị âm chưa được xử lý")
    finite = selected[np.isfinite(selected)]
    if finite.size < 2:
        raise ValueError("Không đủ giá trị train để fit scaler chung")
    center, scale = float(finite.mean()), float(finite.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scaler chung có std không hữu hạn hoặc không dương; không tự thêm epsilon")
    return {"n": int(finite.size), "center": center, "scale": scale, "ddof": 0}


def a3_dlinear_decode(prediction_z, scaler, clip=False):
    """Đổi z về mm/ngày trước khi clip; giữ NaN, không sửa prediction_z."""
    values = np.asarray(prediction_z, dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Dự báo có vô cực")
    physical = values * scaler["scale"] + scaler["center"]
    return np.maximum(physical, 0) if clip else physical


def a3_nse_star_reference(prediction_z, target_z, sigma_train_mm_day, eps=0.1):
    """Tham chiếu NumPy cho loss NH: residual z, std qmax trước chuẩn hóa."""
    prediction_z, target_z, sigma = (
        np.asarray(array, dtype=np.float64)
        for array in (prediction_z, target_z, sigma_train_mm_day)
    )
    if prediction_z.ndim != 1 or prediction_z.shape != target_z.shape or sigma.shape != target_z.shape:
        raise ValueError("Loss cần ba vector 1D cùng kích thước")
    if not len(target_z) or not all(np.isfinite(array).all() for array in (prediction_z, target_z, sigma)):
        raise ValueError("Loss train không nhận mẫu thiếu hoặc vô cực")
    if (sigma < 0).any() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Std phải không âm, epsilon phải hữu hạn và dương")
    return float(np.mean(((prediction_z - target_z) / (sigma + eps)) ** 2))



print("Đơn vị qmean/qmax: mm/ngày theo area_gov; hai scaler chung, chỉ fit train, ddof=0.")
print("Phạm vi fit: mọi giá trị train hữu hạn, mỗi ngày–ID một lần; không gồm ngữ cảnh trước train.")
print("Thiếu lịch sử: giữ NaN dự báo, không nội suy/điền; giữ đủ danh sách ID.")

# %% Phần 7.7 — Fit scaler chung và chuẩn hóa qmean/qmax
# Mỗi ngày–ID train hữu hạn góp một lần; không fit ngữ cảnh trước train hoặc evaluation.
a3_dl_train_days = (
    (A3_DLINEAR_CALENDAR >= pd.Timestamp(A3_PERIODS["train"][0]))
    & (A3_DLINEAR_CALENDAR <= pd.Timestamp(A3_PERIODS["train"][1]))
)
A3_DLINEAR_SCALERS, A3_DLINEAR_FIT_MASKS = {}, {}
a3_dl_scaler_rows, a3_dl_check_rows = [], []
for variable in A3_Q_VARIABLES:
    frame = A3_DLINEAR_Q_MM_DAY[variable]
    if not frame.index.equals(A3_DLINEAR_CALENDAR) or list(frame.columns) != A3_SELECTED_IDS:
        raise ValueError(f"{variable}: trục ngày/ID khác audit")
    values = frame.to_numpy()
    reconstructed = A3_DLINEAR_Q_NUMERIC[variable].mul(86.4 / A3_AREAS, axis="columns").to_numpy()
    if not np.array_equal(values, reconstructed, equal_nan=True):
        raise ValueError(f"{variable}: bản mm/ngày khác phép đổi từ numeric gốc")
    scaler = a3_fit_global_scaler(values, a3_dl_train_days)
    A3_DLINEAR_SCALERS[variable] = scaler
    A3_DLINEAR_FIT_MASKS[variable] = np.isfinite(values) & a3_dl_train_days[:, None]
    reference = A3_DLINEAR_TRAIN_DISTRIBUTION.loc[
        A3_DLINEAR_TRAIN_DISTRIBUTION.unit.eq("mm/day_area_gov")
        & A3_DLINEAR_TRAIN_DISTRIBUTION.variable.eq(variable)
        & A3_DLINEAR_TRAIN_DISTRIBUTION.fit_scope_candidate.eq(A3_DLINEAR_FIT_SCOPE)
    ]
    matched = bool(
        len(reference) == 1 and scaler["n"] == int(reference.iloc[0]["n"])
        and np.isclose(scaler["center"], reference.iloc[0]["mean"], rtol=1e-12, atol=1e-12)
        and np.isclose(scaler["scale"], reference.iloc[0].std_ddof0, rtol=1e-12, atol=1e-12)
    )
    if not matched:
        raise ValueError(f"{variable}: scaler không khớp thống kê train đã kiểm ở 7.1")
    a3_dl_scaler_rows.append({"variable": variable, "unit": "mm/day_area_gov", **scaler,
                             "fit_scope": A3_DLINEAR_FIT_SCOPE, "matches_part7_1": matched})

A3_DLINEAR_X = np.ascontiguousarray((
    (A3_DLINEAR_Q_MM_DAY["qmean"].to_numpy() - A3_DLINEAR_SCALERS["qmean"]["center"])
    / A3_DLINEAR_SCALERS["qmean"]["scale"]
).astype(np.float32))
A3_DLINEAR_Y = np.ascontiguousarray((
    (A3_DLINEAR_Q_MM_DAY["qmax"].to_numpy() - A3_DLINEAR_SCALERS["qmax"]["center"])
    / A3_DLINEAR_SCALERS["qmax"]["scale"]
).astype(np.float32))
if np.isinf(A3_DLINEAR_X).any() or np.isinf(A3_DLINEAR_Y).any():
    raise ValueError("Chuẩn hóa/ép float32 tạo vô cực")
show_table('Scaler chung cố định từ train', a3_dl_scaler_rows)

# %% Phần 7.8 — Std train từng ID và trọng số loss
# Giữ NaN ở ID không đủ nhãn; không lấy std từ validation/test thay thế.
a3_dl_loss_rows = []
train_counts = A3_DLINEAR_WINDOW_COUNTS.loc[A3_DLINEAR_WINDOW_COUNTS.period.eq("train")].set_index("basin_id")
for basin_id in A3_SELECTED_IDS:
    series = A3_DLINEAR_Q_MM_DAY["qmax"].loc[a3_dl_train_days, basin_id].to_numpy()
    finite = series[np.isfinite(series)]
    sigma = float(finite.std(ddof=0)) if finite.size >= 2 else np.nan
    n_samples = int(train_counts.at[basin_id, "candidate_sample_days"])
    active = n_samples > 0
    if active and (not np.isfinite(sigma) or sigma < 0):
        raise ValueError(f"ID {basin_id}: có mẫu train nhưng std train không hợp lệ; không lấy std evaluation thay")
    weight = float(1 / (sigma + A3_DLINEAR_LOSS_EPS) ** 2) if np.isfinite(sigma) else np.nan
    a3_dl_loss_rows.append({
        "basin_id": basin_id, "train_label_n": int(finite.size), "train_sample_n": n_samples,
        "sigma_qmax_train_mm_day": sigma, "loss_eps_mm_day": A3_DLINEAR_LOSS_EPS,
        "loss_weight": weight, "active_in_training": active,
        "std_status": "defined" if np.isfinite(sigma) else "insufficient_train_labels",
    })
A3_DLINEAR_LOSS_STATS = pd.DataFrame(a3_dl_loss_rows)
A3_DLINEAR_SIGMA_TRAIN = A3_DLINEAR_LOSS_STATS.set_index("basin_id").sigma_qmax_train_mm_day
print("NSE* = mean((prediction_z − target_z)² / (sigma_qmax_train_mm_day + 0.1)²).")
print("Bám quy ước NH: residual đã chuẩn hóa, sigma tính trước chuẩn hóa; không dùng loss này làm NSE đánh giá.")
print("Số ID tham gia train:", int(A3_DLINEAR_LOSS_STATS.active_in_training.sum()))
show_table('Std/loss của bảy hồ sơ A1 và mọi ID không có mẫu train',
           A3_DLINEAR_LOSS_STATS.loc[
               A3_DLINEAR_LOSS_STATS.basin_id.isin(A3_DLINEAR_PROFILE_IDS)
               | ~A3_DLINEAR_LOSS_STATS.active_in_training
           ].to_dict("records"))

# %% Phần 7.9 — Chỉ mục mẫu và ngày suy luận
# Lưu ngày đích/cột ID theo lịch, tránh sao chép toàn bộ cửa sổ.
A3_DLINEAR_SAMPLE_INDEX, A3_DLINEAR_INFERENCE_INDEX = {}, {}
a3_dl_index_rows = []
for period, bounds in A3_PERIODS.items():
    calendar = pd.date_range(*bounds, freq="D")
    history = A3_DLINEAR_HISTORY_VALID.loc[calendar].to_numpy()
    labels = A3_DLINEAR_Q_MM_DAY["qmax"].loc[calendar].to_numpy()
    sample_mask = A3_DLINEAR_SAMPLE_MASKS[period].to_numpy()
    if not np.array_equal(sample_mask, history & np.isfinite(labels)):
        raise ValueError(f"{period}: mặt nạ mẫu khác giao input/nhãn")
    for kind, mask, index_store in (
        ("samples", sample_mask, A3_DLINEAR_SAMPLE_INDEX),
        ("inference", history, A3_DLINEAR_INFERENCE_INDEX),
    ):
        local_day, basin_col = np.nonzero(mask)
        target_day = A3_DLINEAR_CALENDAR.get_indexer(calendar)[local_day]
        if (target_day < A3_DLINEAR_LOOKBACK).any():
            raise ValueError("Chỉ mục ngày đích thiếu ngữ cảnh")
        index_store[period] = np.column_stack((target_day, basin_col)).astype(np.int32)
    expected_n = int(A3_DLINEAR_WINDOW_COUNTS.loc[
        A3_DLINEAR_WINDOW_COUNTS.period.eq(period), "candidate_sample_days"
    ].sum())
    if len(A3_DLINEAR_SAMPLE_INDEX[period]) != expected_n or expected_n == 0:
        raise ValueError(f"{period}: số mẫu khác audit hoặc không có mẫu")
    a3_dl_index_rows.append({"period": period, "samples": expected_n,
                             "inference_targets": len(A3_DLINEAR_INFERENCE_INDEX[period]),
                             "IDs_with_samples": int(sample_mask.any(axis=0).sum())})
show_table('Chỉ mục mẫu theo lịch gốc — không sao chép mọi cửa sổ', a3_dl_index_rows)

# %% Phần 7.10 — Kiểm tiền xử lý và loss
# Kiểm phạm vi fit, độ chính xác float32, inverse scaling và clip.
toy_train = np.array([True, True, False, False])
toy_values = np.array([[1.], [3.], [1e8], [np.nan]])
toy_scaler = a3_fit_global_scaler(toy_values, toy_train)
a3_dl_check_rows.append({"case": "fit_excludes_evaluation", "passed": toy_scaler["center"] == 2 and toy_scaler["scale"] == 1})
toy_scaler_out = {"center": 2., "scale": 4.}
toy_z = np.array([-1., 0., np.nan])
toy_raw = a3_dlinear_decode(toy_z, toy_scaler_out)
toy_clipped = a3_dlinear_decode(toy_z, toy_scaler_out, clip=True)
a3_dl_check_rows.append({"case": "clip_in_physical_units_preserves_raw_nan", "passed": bool(
    np.array_equal(toy_raw, [-2., 2., np.nan], equal_nan=True)
    and np.array_equal(toy_clipped, [0., 2., np.nan], equal_nan=True)
    and np.array_equal(toy_z, [-1., 0., np.nan], equal_nan=True)
)})
toy_loss = a3_nse_star_reference([1., 2.], [0., 0.], [0., .9], A3_DLINEAR_LOSS_EPS)
a3_dl_check_rows.append({"case": "NSE_star_zero_sigma_uses_declared_eps", "passed": bool(np.isclose(toy_loss, 52.))})
for variable, prepared in (("qmean", A3_DLINEAR_X), ("qmax", A3_DLINEAR_Y)):
    source = A3_DLINEAR_Q_MM_DAY[variable].to_numpy()
    decoded = a3_dlinear_decode(prepared, A3_DLINEAR_SCALERS[variable])
    finite = np.isfinite(source)
    max_error = float(np.max(np.abs(decoded[finite] - source[finite])))
    precision_bound = 8 * np.finfo(np.float32).eps * max(
        1., float(np.abs(source[finite]).max()), A3_DLINEAR_SCALERS[variable]["scale"],
    )
    a3_dl_check_rows.append({"case": f"{variable}_float32_roundtrip", "passed": bool(
        np.array_equal(np.isnan(source), np.isnan(prepared)) and max_error <= precision_bound
    ), "max_abs_error_mm_day": max_error, "float32_precision_bound": precision_bound})
    a3_dl_check_rows.append({"case": f"{variable}_fit_only_train", "passed": bool(
        not A3_DLINEAR_FIT_MASKS[variable][~a3_dl_train_days].any()
    )})
A3_DLINEAR_PREPROCESSING_CHECKS = pd.DataFrame(a3_dl_check_rows)
show_table('Kiểm phạm vi fit, đổi ngược, clip và loss', a3_dl_check_rows)
if not A3_DLINEAR_PREPROCESSING_CHECKS.passed.all():
    raise RuntimeError("Kiểm tiền xử lý chưa đạt; chưa cố định bộ cấu hình để huấn luyện")

# %% Phần 7.11 — Cấu hình DLinear cố định
# Giữ kiến trúc, optimizer, scheduler và tiêu chí chọn checkpoint cho ba seed.
A3_DLINEAR_CONFIG = {
    "schema_version": 1, "model": "DLinear", "seeds": [111, 222, 333],
    "architecture": {"seq_len": A3_DLINEAR_LOOKBACK, "pred_len": 1, "enc_in": 1,
        "individual": False, "moving_average_kernel": 25, "padding": "repeat_endpoints",
        "dropout": 0., "initialization": "PyTorch nn.Linear default",
        "source_commit": "7f7a25b6d32fa0df77ce4c20d4ffeb849b5a7e47",
        "source_url": "https://github.com/cure-lab/LTSF-Linear/blob/7f7a25b6d32fa0df77ce4c20d4ffeb849b5a7e47/models/DLinear.py"},
    "data": {"input": "qmean[t-365:t] excludes t", "target": "qmax[t]", "periods": A3_PERIODS,
        "working_unit": "mm/day_area_gov", "area_field": "area_gov", "basin_ids": A3_SELECTED_IDS,
        "normalization": "global_featurewise_z_score", "fit_scope": A3_DLINEAR_FIT_SCOPE,
        "fit_context_before_train": False, "missing_policy": "complete_365_day_input_else_prediction_NaN",
        "inference_requires_label": False, "training_requires_label": True, "dtype": "float32"},
    "loss": {"name": "NH_NSE_star", "residual_unit": "qmax_z",
        "sigma_unit": "mm/day_area_gov", "sigma_scope": "all_finite_qmax_train_days_per_basin",
        "sigma_ddof": 0, "eps": A3_DLINEAR_LOSS_EPS, "clip_during_training": False,
        "formula": "mean((prediction_z-target_z)^2/(sigma_qmax_train_mm_day+0.1)^2)",
        "source_commit": EXPECTED_COMMIT,
        "loss_module_sha256": "faa1b5cba0fa25885aa11a7c56439b3990399da83990d59ba61e71849fb469ae",
        "dataset_module_sha256": "9da162ecc1cff00783fbc66e85ace386193c3c9f0f7d272429f8c5521720a494"},
    "training": {"optimizer": "Adam", "learning_rate": .001, "weight_decay": 0.,
        "batch_size": 256, "shuffle": True, "drop_last": False, "sampling": "uniform_over_valid_samples",
        "max_epochs": 30, "scheduler": "CosineAnnealingLR", "scheduler_T_max": 30,
        "scheduler_eta_min": .00001, "scheduler_step": "after_each_epoch",
        "gradient_clip_norm": 1., "amp": False,
        "checkpoint_metric": "validation_median_NSE", "checkpoint_rule": "save_each_strict_finite_improvement",
        "early_stopping_patience": 5, "early_stopping_min_delta": .005,
        "early_stopping_rule": "reset_patience_only_if_best_significant_score_improves_by_more_than_min_delta",
        "report_test_each_epoch": True, "test_used_for_selection": False},
    "evaluation": {"clip_to_zero_after_inverse_scaling": True, "save_raw_and_clipped": True,
        "metrics": list(A3_METRIC_NAMES), "observation_source": "canonical_raw_qmax_area_gov",
        "unseen_train_basins": "infer_with_global_scaler_when_history_complete_no_loss_std_fallback",
        "validation_checkpoint_uses": "per_basin_NSE_on_available_pairs_then_median",
        "validation_NSE_star_not_required": True},
}

# %% Phần 7.12 — Lưu scaler, dữ liệu chuẩn bị và cấu hình
# Lưu các tệp cần dùng cho mô hình và xác nhận hash config.
a3_dl_files = []
for filename, payload in (
    ("A3_DLinear_scaler.json", {"unit": "mm/day_area_gov", "fit_scope": A3_DLINEAR_FIT_SCOPE,
                               "fit_period": A3_PERIODS["train"], "scalers": A3_DLINEAR_SCALERS}),
    ("A3_DLinear_config.json", A3_DLINEAR_CONFIG),
):
    path = A3_OUT_DIR / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    a3_dl_files.append(path)
for filename, frame in (
    ("A3_DLinear_loss_stats.csv", A3_DLINEAR_LOSS_STATS),
    ("A3_DLinear_preprocessing_checks.csv", A3_DLINEAR_PREPROCESSING_CHECKS),
):
    path = A3_OUT_DIR / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    a3_dl_files.append(path)
fit_path = A3_OUT_DIR / "A3_DLinear_fit_membership.npz"
np.savez_compressed(fit_path, dates=A3_DLINEAR_CALENDAR.to_numpy(dtype="datetime64[D]"),
                    basin_ids=np.asarray(A3_SELECTED_IDS, dtype="U"),
                    qmean_fit=A3_DLINEAR_FIT_MASKS["qmean"], qmax_fit=A3_DLINEAR_FIT_MASKS["qmax"])
indices_path = A3_OUT_DIR / "A3_DLinear_indices.npz"
np.savez_compressed(indices_path, **{
    f"{period}_{kind}": store[period]
    for kind, store in (("samples", A3_DLINEAR_SAMPLE_INDEX), ("inference", A3_DLINEAR_INFERENCE_INDEX))
    for period in A3_PERIODS
})
prepared_path = A3_OUT_DIR / "A3_DLinear_prepared_data.npz"
np.savez_compressed(prepared_path, dates=A3_DLINEAR_CALENDAR.to_numpy(dtype="datetime64[D]"),
                    basin_ids=np.asarray(A3_SELECTED_IDS, dtype="U"), X_z=A3_DLINEAR_X, Y_z=A3_DLINEAR_Y,
                    sigma_train_mm_day=A3_DLINEAR_SIGMA_TRAIN.reindex(A3_SELECTED_IDS).to_numpy())
a3_dl_files.extend([fit_path, indices_path, prepared_path])
A3_DLINEAR_PREPROCESSING_MANIFEST = pd.DataFrame([
    {"file": path.name, "bytes": path.stat().st_size, "sha256": a3_hash_file(path)}
    for path in a3_dl_files
])
A3_DLINEAR_PREPROCESSING_MANIFEST.to_csv(
    A3_OUT_DIR / "A3_DLinear_preprocessing_manifest.csv", index=False, encoding="utf-8-sig",
)
A3_DLINEAR_CONFIG_SHA256 = a3_hash_file(A3_OUT_DIR / "A3_DLinear_config.json")
A3_DLINEAR_PREPROCESSING_OK = True

show_table('Cấu hình đã cố định', [
    {"section": key, "value": json.dumps(value, ensure_ascii=False)}
    for key, value in A3_DLINEAR_CONFIG.items() if key not in ("data",)
])

print("A3_DLINEAR_PREPROCESSING_OK =", A3_DLINEAR_PREPROCESSING_OK)
print("Config SHA-256:", A3_DLINEAR_CONFIG_SHA256)
print("Đã lưu 2 JSON, 2 CSV, 3 NPZ và 1 CSV manifest; chưa tạo mô hình hoặc chạy seed.")
print(f"Thời gian chuẩn bị: {time.perf_counter() - a3_preparation_started:.1f} giây | Output: {A3_OUT_DIR}")

# %% Phần 8.1 — Thiết bị và chế độ tính xác định
# Thiết lập PyTorch, CUDA/cuBLAS, tắt TF32 và kiểm config đã cố định.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn

if not globals().get("A3_DLINEAR_PREPROCESSING_OK", False):
    raise RuntimeError("Cần đạt Phần 7.2 trong cùng phiên trước khi kiểm mô hình.")
A3_DLINEAR_MODEL_CHECKS_OK = False
if a3_hash_file(A3_OUT_DIR / "A3_DLinear_config.json") != A3_DLINEAR_CONFIG_SHA256:
    raise ValueError("Config đã đổi sau Phần 7.2")
# JSON đổi tuple kỳ thành list; chuẩn hóa biểu diễn trước khi so nội dung.
if json.loads(json.dumps(A3_DLINEAR_CONFIG)) != json.loads(
    (A3_OUT_DIR / "A3_DLinear_config.json").read_text(encoding="utf-8")
):
    raise ValueError("Config trong RAM khác tệp đã cố định")
for record in A3_DLINEAR_PREPROCESSING_MANIFEST.itertuples():
    if a3_hash_file(A3_OUT_DIR / record.file) != record.sha256:
        raise ValueError(f"Tệp tiền xử lý đã đổi: {record.file}")

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
A3_DLINEAR_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
A3_DLINEAR_SMOKE_SEED = 20261022
A3_DLINEAR_UPSTREAM_SHA256 = "0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc"
A3_DLINEAR_SMOKE_DIR = A3_OUT_DIR / "dlinear" / "smoke"
A3_DLINEAR_SMOKE_DIR.mkdir(parents=True, exist_ok=True)

# %% Phần 8.2 — Kiến trúc DLinear và loss PyTorch
# Hai nhánh tuyến tính theo mùa/xu thế, moving average và loss đúng quy ước.
class A3MovingAverage(nn.Module):
    """Trung bình trượt với lặp giá trị hai đầu, cùng hình học mã DLinear gốc."""

    def __init__(self, kernel_size):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("Kernel phải nguyên lẻ dương")
        self.radius = kernel_size // 2
        self.avg = nn.AvgPool1d(kernel_size, stride=1)

    def forward(self, x):
        left = x[:, :1, :].expand(-1, self.radius, -1)
        right = x[:, -1:, :].expand(-1, self.radius, -1)
        padded = torch.cat((left, x, right), dim=1)
        return self.avg(padded.transpose(1, 2)).transpose(1, 2)


class A3DLinear(nn.Module):
    """Một kênh qmean: nhánh residual và trend, hai phép chiếu 365→1."""

    def __init__(self, architecture):
        super().__init__()
        if architecture["enc_in"] != 1 or architecture["individual"] or architecture["pred_len"] != 1:
            raise ValueError("Cell này kiểm hợp đồng một kênh, individual=False, pred_len=1")
        self.seq_len = architecture["seq_len"]
        self.decomposition = A3MovingAverage(architecture["moving_average_kernel"])
        self.Linear_Seasonal = nn.Linear(self.seq_len, 1)
        self.Linear_Trend = nn.Linear(self.seq_len, 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, 1):
            raise ValueError("Input DLinear phải là [batch, 365, 1]")
        trend = self.decomposition(x)
        residual = (x - trend).transpose(1, 2)
        return (self.Linear_Seasonal(residual) + self.Linear_Trend(trend.transpose(1, 2))).transpose(1, 2)


def a3_dlinear_loss(prediction_z, target_z, sigma_train_mm_day):
    if prediction_z.shape != target_z.shape or prediction_z.shape != sigma_train_mm_day.shape:
        raise ValueError("Loss cần prediction/target/std cùng shape, không broadcast âm thầm")
    if prediction_z.numel() == 0:
        raise ValueError("Batch loss rỗng")
    if not all(torch.isfinite(array).all().item() for array in (prediction_z, target_z, sigma_train_mm_day)):
        raise ValueError("Batch loss có giá trị thiếu/vô cực")
    if (sigma_train_mm_day < 0).any().item():
        raise ValueError("Std train âm")
    return ((prediction_z - target_z).square() / (sigma_train_mm_day + A3_DLINEAR_LOSS_EPS).square()).mean()

# %% Phần 8.3 — Batch, huấn luyện và dự báo
# Đọc cửa sổ qua chỉ mục; giữ NaN khi không đủ lịch sử và đổi ngược trước chấm.
def a3_dlinear_batch(index_rows, device, with_targets=True):
    rows = np.asarray(index_rows)
    if rows.ndim != 2 or rows.shape[1] != 2 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Chỉ mục phải là ma trận nguyên [ngày đích, cột ID]")
    days, columns = rows[:, 0], rows[:, 1]
    if not len(rows) or (days < A3_DLINEAR_LOOKBACK).any() or (days >= len(A3_DLINEAR_X)).any():
        raise ValueError("Ngày đích rỗng/thiếu ngữ cảnh/ngoài dữ liệu")
    if (columns < 0).any() or (columns >= len(A3_SELECTED_IDS)).any():
        raise ValueError("Cột ID ngoài dữ liệu")
    offsets = np.arange(-A3_DLINEAR_LOOKBACK, 0)
    window = np.ascontiguousarray(A3_DLINEAR_X[days[:, None] + offsets, columns[:, None]])
    if not np.isfinite(window).all():
        raise ValueError("Chỉ mục chọn cửa sổ qmean không đầy đủ")
    x = torch.from_numpy(window).unsqueeze(-1).to(device)
    if not with_targets:
        return x, None, None
    target = A3_DLINEAR_Y[days, columns]
    sigma = A3_DLINEAR_SIGMA_TRAIN.reindex(A3_SELECTED_IDS).to_numpy()[columns].astype(np.float32)
    if not np.isfinite(target).all() or not np.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError("Mẫu train thiếu nhãn/std train; không tự dùng std evaluation")
    return x, torch.from_numpy(target.copy()).to(device).view(-1, 1, 1), torch.from_numpy(sigma).to(device).view(-1, 1, 1)


def a3_dlinear_train_epoch(model, optimizer, device, generator, progress_every=1000):
    rows = A3_DLINEAR_SAMPLE_INDEX["train"]
    order = generator.permutation(len(rows))
    size = A3_DLINEAR_CONFIG["training"]["batch_size"]
    total_loss, seen, batches, last_batch = 0., 0, 0, 0
    model.train()
    for start in range(0, len(rows), size):
        x, target, sigma = a3_dlinear_batch(rows[order[start:start + size]], device)
        optimizer.zero_grad(set_to_none=True)
        loss = a3_dlinear_loss(model(x), target, sigma)
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), A3_DLINEAR_CONFIG["training"]["gradient_clip_norm"],
            error_if_nonfinite=True,
        )
        optimizer.step()
        last_batch = len(x)
        total_loss += float(loss.detach().cpu()) * last_batch
        seen += last_batch
        batches += 1
        if progress_every and batches % progress_every == 0:
            print(f"Chạy thử train: {seen}/{len(rows)} mẫu", flush=True)
    if seen != len(rows) or not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("Epoch thiếu mẫu hoặc tạo trọng số không hữu hạn")
    return {"train_loss": total_loss / seen, "samples": seen, "batches": batches, "last_batch": last_batch}


def a3_dlinear_predict(model, period, device):
    calendar = pd.date_range(*A3_PERIODS[period], freq="D")
    raw = np.full((len(calendar), len(A3_SELECTED_IDS)), np.nan, dtype=np.float64)
    rows = A3_DLINEAR_INFERENCE_INDEX[period]
    size = A3_DLINEAR_CONFIG["training"]["batch_size"]
    offset = A3_DLINEAR_CALENDAR.get_indexer(calendar)[0]
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), size):
            batch = rows[start:start + size]
            x, _, _ = a3_dlinear_batch(batch, device, with_targets=False)
            prediction = model(x).reshape(-1).cpu().numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError("DLinear có dự báo không hữu hạn ở cửa sổ đầy đủ")
            raw[batch[:, 0] - offset, batch[:, 1]] = a3_dlinear_decode(prediction, A3_DLINEAR_SCALERS["qmax"])
    model.train(was_training)
    if not np.array_equal(np.isfinite(raw), A3_DLINEAR_HISTORY_VALID.loc[calendar].to_numpy()):
        raise ValueError("Mặt nạ dự báo khác mặt nạ đầu vào đủ lịch sử")
    return (
        pd.DataFrame(raw, index=calendar, columns=A3_SELECTED_IDS),
        pd.DataFrame(np.maximum(raw, 0), index=calendar, columns=A3_SELECTED_IDS),
    )


def a3_dlinear_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

# %% Phần 8.4 — Kiểm kiến trúc và gradient loss
# Đối chiếu NumPy, số tham số và đạo hàm giải tích trước chạy dữ liệu.
print("PyTorch:", torch.__version__, "| thiết bị:", A3_DLINEAR_DEVICE)
print("Thiết bị:", torch.cuda.get_device_name(A3_DLINEAR_DEVICE) if A3_DLINEAR_DEVICE.type == "cuda" else "CPU")
print("Deterministic:", torch.are_deterministic_algorithms_enabled(), "| CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
print("Chạy một epoch train tạm và validation; không chấm test, không ghi vào seed_111/222/333.")
a3_model_check_rows = []
architecture = A3_DLINEAR_CONFIG["architecture"]

# Kiểm cấu trúc bằng tính toán NumPy độc lập trên CPU float64.
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(A3_DLINEAR_SMOKE_SEED)
    a3_check_model = A3DLinear(architecture).double()
    values = np.linspace(-2, 3, 2 * A3_DLINEAR_LOOKBACK).reshape(2, A3_DLINEAR_LOOKBACK, 1)
    x64 = torch.from_numpy(values)
    radius = architecture["moving_average_kernel"] // 2
    padded = np.pad(values, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    trend = np.stack([padded[:, i:i + 2 * radius + 1].mean(axis=1) for i in range(A3_DLINEAR_LOOKBACK)], axis=1)
    torch_trend = a3_check_model.decomposition(x64).detach().numpy()
    manual = sum(
        component[:, :, 0] @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()
        for component, layer in ((values - trend, a3_check_model.Linear_Seasonal), (trend, a3_check_model.Linear_Trend))
    )
    output = a3_check_model(x64)
    a3_model_check_rows.extend([
        {"case": "moving_average_repeat_endpoints_numpy", "passed": bool(np.allclose(torch_trend, trend, atol=1e-12, rtol=1e-12))},
        {"case": "two_linear_branches_numpy", "passed": bool(np.allclose(output.detach().numpy()[:, :, 0], manual, atol=1e-12, rtol=1e-12))},
        {"case": "parameter_count_and_shape", "passed": bool(sum(p.numel() for p in a3_check_model.parameters()) == 732 and output.shape == (2, 1, 1))},
    ])
    predicted = torch.tensor([1., 2., -.5], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0., .25, .5], dtype=torch.float64)
    sigma = torch.tensor([0., .9, 2.], dtype=torch.float64)
    checked_loss = a3_dlinear_loss(predicted, target, sigma)
    checked_loss.backward()
    ref_loss = a3_nse_star_reference(predicted.detach().numpy(), target.numpy(), sigma.numpy(), A3_DLINEAR_LOSS_EPS)
    ref_grad = 2 * (predicted.detach().numpy() - target.numpy()) / (sigma.numpy() + A3_DLINEAR_LOSS_EPS) ** 2 / len(target)
    a3_model_check_rows.append({"case": "NSE_star_value_and_analytic_gradient", "passed": bool(
        np.isclose(float(checked_loss.detach()), ref_loss, atol=1e-12, rtol=1e-12)
        and np.allclose(predicted.grad.numpy(), ref_grad, atol=1e-12, rtol=1e-12)
    )})

# %% Phần 8.5 — Kiểm chỉ mục và gradient trên thiết bị
# Xác nhận ngày đầu/cuối input và gradient hữu hạn trên batch thật.
rows = A3_DLINEAR_SAMPLE_INDEX["train"]
probe = rows[np.unique([0, min(1, len(rows) - 1), len(rows) - 1])]
a3_probe_x, a3_probe_y, a3_probe_sigma = a3_dlinear_batch(probe, A3_DLINEAR_DEVICE)
a3_index_rows = []
for i, (day, col) in enumerate(probe):
    expected = A3_DLINEAR_X[day - A3_DLINEAR_LOOKBACK:day, col]
    correct = np.array_equal(a3_probe_x[i, :, 0].cpu().numpy(), expected)
    a3_model_check_rows.append({"case": f"exact_calendar_window_probe_{i}", "passed": correct})
    a3_index_rows.append({"basin_id": A3_SELECTED_IDS[col], "input_start": str(A3_DLINEAR_CALENDAR[day - A3_DLINEAR_LOOKBACK].date()),
                         "input_end": str(A3_DLINEAR_CALENDAR[day - 1].date()), "target": str(A3_DLINEAR_CALENDAR[day].date()), "n_input_days": len(expected)})
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(A3_DLINEAR_SMOKE_SEED)
    a3_probe_model = A3DLinear(architecture).to(A3_DLINEAR_DEVICE)
    a3_probe_loss = a3_dlinear_loss(a3_probe_model(a3_probe_x), a3_probe_y, a3_probe_sigma)
    a3_probe_loss.backward()
    a3_model_check_rows.append({"case": "device_forward_backward_all_gradients_finite", "passed": bool(
        all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in a3_probe_model.parameters())
        and any(p.grad.abs().sum().item() > 0 for p in a3_probe_model.parameters())
    )})
del a3_check_model, a3_probe_model
show_table('Kiểm kiến trúc, loss, gradient và chỉ mục', a3_model_check_rows)
show_table('Ngày đầu vào và ngày đích của các batch kiểm', a3_index_rows)
if not all(row["passed"] for row in a3_model_check_rows):
    pd.DataFrame(a3_model_check_rows).to_csv(A3_DLINEAR_SMOKE_DIR / "checks.csv", index=False)
    raise RuntimeError("Kiểm cấu trúc/loss chưa đạt; chưa chạy epoch tạm")

# %% Phần 8.6 — Chạy thử một epoch và checkpoint
# Dùng seed tạm; kiểm khôi phục dự báo, optimizer, scheduler và trạng thái shuffle.
if A3_DLINEAR_DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats(A3_DLINEAR_DEVICE)
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(A3_DLINEAR_SMOKE_SEED)
    a3_smoke_model = A3DLinear(architecture).to(A3_DLINEAR_DEVICE)
    a3_initial_weights = {key: value.detach().cpu().clone() for key, value in a3_smoke_model.state_dict().items()}
    training = A3_DLINEAR_CONFIG["training"]
    a3_smoke_optimizer = torch.optim.Adam(a3_smoke_model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    a3_smoke_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        a3_smoke_optimizer, T_max=training["scheduler_T_max"], eta_min=training["scheduler_eta_min"],
    )
    a3_smoke_generator = np.random.default_rng(A3_DLINEAR_SMOKE_SEED)
    a3_dlinear_sync(A3_DLINEAR_DEVICE)
    started = time.perf_counter()
    a3_smoke_training = a3_dlinear_train_epoch(a3_smoke_model, a3_smoke_optimizer, A3_DLINEAR_DEVICE, a3_smoke_generator)
    a3_smoke_scheduler.step()
    a3_dlinear_sync(A3_DLINEAR_DEVICE)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    a3_smoke_raw, a3_smoke_clipped = a3_dlinear_predict(a3_smoke_model, "validation", A3_DLINEAR_DEVICE)
    a3_validation_rows = []
    for basin_id in A3_SELECTED_IDS:
        observed = A3_DLINEAR_Q_MM_DAY["qmax"].loc[a3_smoke_raw.index, basin_id].to_numpy()
        simulated = a3_smoke_clipped[basin_id].to_numpy()
        result = a3_metrics(observed, simulated)
        a3_validation_rows.append({"basin_id": basin_id, "n": result["n"], "NSE": result["NSE"], "NSE_status": result["NSE_status"]})
    a3_dlinear_sync(A3_DLINEAR_DEVICE)
    validation_seconds = time.perf_counter() - started
    A3_DLINEAR_SMOKE_VALIDATION = pd.DataFrame(a3_validation_rows)
    a3_checkpoint_path = A3_DLINEAR_SMOKE_DIR / "one_epoch.pt"
    torch.save({
        "role": "diagnostic_only_not_official_seed", "smoke_seed": A3_DLINEAR_SMOKE_SEED, "epoch": 1,
        "config_sha256": A3_DLINEAR_CONFIG_SHA256,
        "model_state": a3_smoke_model.state_dict(), "optimizer_state": a3_smoke_optimizer.state_dict(),
        "scheduler_state": a3_smoke_scheduler.state_dict(), "numpy_generator_state": a3_smoke_generator.bit_generator.state,
    }, a3_checkpoint_path)
    loaded = torch.load(a3_checkpoint_path, map_location=A3_DLINEAR_DEVICE, weights_only=True)
    a3_reloaded_model = A3DLinear(architecture).to(A3_DLINEAR_DEVICE)
    a3_reloaded_model.load_state_dict(loaded["model_state"], strict=True)
    a3_reloaded_optimizer = torch.optim.Adam(a3_reloaded_model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    a3_reloaded_optimizer.load_state_dict(loaded["optimizer_state"])
    a3_reloaded_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        a3_reloaded_optimizer, T_max=training["scheduler_T_max"], eta_min=training["scheduler_eta_min"],
    )
    a3_reloaded_scheduler.load_state_dict(loaded["scheduler_state"])
    a3_reloaded_generator = np.random.default_rng()
    a3_reloaded_generator.bit_generator.state = loaded["numpy_generator_state"]
    a3_smoke_model.eval()
    a3_reloaded_model.eval()
    with torch.no_grad():
        reload_prediction_ok = torch.equal(a3_smoke_model(a3_probe_x), a3_reloaded_model(a3_probe_x))
    # Một bước tiếp theo phải cho cùng trọng số khi optimizer cũng được phục hồi.
    for model, optimizer in ((a3_smoke_model, a3_smoke_optimizer), (a3_reloaded_model, a3_reloaded_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        loss = a3_dlinear_loss(model(a3_probe_x), a3_probe_y, a3_probe_sigma)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
    a3_model_check_rows.extend([
        {"case": "checkpoint_prediction_exact", "passed": reload_prediction_ok},
        {"case": "checkpoint_optimizer_next_step_exact", "passed": all(
            torch.equal(a3_smoke_model.state_dict()[key], a3_reloaded_model.state_dict()[key]) for key in a3_initial_weights
        )},
        {"case": "checkpoint_scheduler_and_shuffle_state", "passed": bool(
            a3_smoke_scheduler.state_dict() == a3_reloaded_scheduler.state_dict()
            and np.array_equal(a3_smoke_generator.permutation(100), a3_reloaded_generator.permutation(100))
        )},
        {"case": "one_epoch_keeps_final_batch", "passed": a3_smoke_training["last_batch"] == (len(rows) - 1) % training["batch_size"] + 1},
        {"case": "weights_updated", "passed": any(
            not torch.equal(value, a3_smoke_model.state_dict()[key].detach().cpu()) for key, value in a3_initial_weights.items()
        )},
    ])

# %% Phần 8.7 — Báo cáo kiểm triển khai và thời gian
# Lưu kiểm, validation thử và bộ nhớ; đây không phải seed chính thức thứ tư.
A3_DLINEAR_MODEL_CHECKS = pd.DataFrame(a3_model_check_rows)
finite_validation = A3_DLINEAR_SMOKE_VALIDATION.NSE[np.isfinite(A3_DLINEAR_SMOKE_VALIDATION.NSE)]
if finite_validation.empty:
    raise RuntimeError("Chạy thử không có NSE validation hữu hạn")
A3_DLINEAR_SMOKE_REPORT = {
    "role": "diagnostic_only_not_official_seed", "smoke_seed": A3_DLINEAR_SMOKE_SEED,
    "torch": torch.__version__, "device": str(A3_DLINEAR_DEVICE),
    "device_name": torch.cuda.get_device_name(A3_DLINEAR_DEVICE) if A3_DLINEAR_DEVICE.type == "cuda" else "CPU",
    "cpu_threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "amp": False, "tf32": False,
    "upstream_commit": architecture["source_commit"], "upstream_source_sha256": A3_DLINEAR_UPSTREAM_SHA256,
    "config_sha256": A3_DLINEAR_CONFIG_SHA256, "parameters": 732,
    **a3_smoke_training, "train_seconds": train_seconds, "validation_seconds": validation_seconds,
    "validation_median_NSE": float(finite_validation.median()), "validation_finite_NSE_IDs": int(len(finite_validation)),
    "validation_paired_day_IDs": int(A3_DLINEAR_SMOKE_VALIDATION.n.sum()),
    "validation_prediction_day_IDs": int(np.isfinite(a3_smoke_raw.to_numpy()).sum()),
    "peak_cuda_allocated_MiB": torch.cuda.max_memory_allocated(A3_DLINEAR_DEVICE) / 2**20 if A3_DLINEAR_DEVICE.type == "cuda" else None,
    "prepared_X_Y_MiB": (A3_DLINEAR_X.nbytes + A3_DLINEAR_Y.nbytes) / 2**20,
    "projected_30_epoch_3_seed_train_plus_validation_hours": 30 * 3 * (train_seconds + validation_seconds) / 3600,
    "timing_excludes": "test inference per epoch, final exports, setup and checkpoint I/O; not a total runtime guarantee",
    "official_seeds_started": False,
}
A3_DLINEAR_MODEL_CHECKS_OK = bool(A3_DLINEAR_MODEL_CHECKS.passed.all())
A3_DLINEAR_MODEL_CHECKS.to_csv(A3_DLINEAR_SMOKE_DIR / "checks.csv", index=False, encoding="utf-8-sig")
A3_DLINEAR_SMOKE_VALIDATION.to_csv(A3_DLINEAR_SMOKE_DIR / "validation_per_basin.csv", index=False, encoding="utf-8-sig")
(A3_DLINEAR_SMOKE_DIR / "report.json").write_text(json.dumps(
    {**A3_DLINEAR_SMOKE_REPORT, "model_checks_ok": A3_DLINEAR_MODEL_CHECKS_OK}, ensure_ascii=False, indent=2, allow_nan=False,
), encoding="utf-8")
show_table('Một epoch train tạm và validation — không phải seed chính thức', [A3_DLINEAR_SMOKE_REPORT])
show_table('Kiểm checkpoint và batch cuối', a3_model_check_rows)

print("A3_DLINEAR_MODEL_CHECKS_OK =", A3_DLINEAR_MODEL_CHECKS_OK)
print("Đã lưu checks.csv, validation_per_basin.csv, report.json và one_epoch.pt tại:", A3_DLINEAR_SMOKE_DIR)
print("Ước lượng thời gian chỉ gồm train+validation; còn thiếu test mỗi epoch và I/O. Chưa chạy 111/222/333.")

del a3_smoke_model, a3_reloaded_model, a3_smoke_optimizer, a3_reloaded_optimizer
del a3_smoke_scheduler, a3_reloaded_scheduler, loaded
if not A3_DLINEAR_MODEL_CHECKS_OK:
    raise RuntimeError("Kiểm mô hình/checkpoint chưa đạt; chưa chạy seed chính thức")

# %% Phần 9.1 — Lưu checkpoint và quy tắc dừng sớm
# Tách việc lưu mọi cải thiện validation khỏi mốc reset patience.
import random

if not globals().get("A3_DLINEAR_MODEL_CHECKS_OK", False):
    raise RuntimeError("Cần đạt Phần 8 trước khi chạy seed chính thức.")


def a3_atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def a3_atomic_torch_save(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def a3_atomic_csv(path, frame):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def a3_update_dlinear_selection(selection, score, epoch, min_delta, patience):
    """Chỉ nhận validation: lưu mọi best, bộ đếm dừng sớm dùng mốc riêng."""
    if score is None or not np.isfinite(score):
        raise ValueError("NSE validation không hữu hạn; không chọn checkpoint theo test")
    current = dict(selection)
    best_improved = current["best_score"] is None or score > current["best_score"]
    if best_improved:
        current.update(best_score=float(score), best_epoch=int(epoch))
    significant = current["significant_anchor"] is None or score > current["significant_anchor"] + min_delta
    if significant:
        current.update(significant_anchor=float(score), bad_epochs=0)
    else:
        current["bad_epochs"] += 1
    return current, best_improved, current["bad_epochs"] >= patience

# %% Phần 9.2 — Hàm chấm, hợp đồng train và xuất dự báo
# Giữ đơn vị, nguồn và cấu hình khi đọc/lưu kết quả từng seed.
def a3_dlinear_nse_table(period, raw, clipped):
    rows = []
    source = A3_DLINEAR_Q_MM_DAY["qmax"].loc[raw.index]
    for basin_id in A3_SELECTED_IDS:
        values = a3_metrics(source[basin_id].to_numpy(), clipped[basin_id].to_numpy())
        rows.append({"period": period, "basin_id": basin_id, "n": values["n"],
                     "NSE": values["NSE"], "NSE_status": values["NSE_status"],
                     "sim_raw_negative_days": int(raw[basin_id].lt(0).sum())})
    table = pd.DataFrame(rows)
    finite = table.NSE[np.isfinite(table.NSE)]
    paired = int(table.n.sum())
    predictions = int(np.isfinite(raw.to_numpy()).sum())
    if paired != len(A3_DLINEAR_SAMPLE_INDEX[period]) or predictions != len(A3_DLINEAR_INFERENCE_INDEX[period]):
        raise ValueError(f"{period}: độ phủ dự báo/cặp khác chỉ mục đã cố định")
    return table, {"median_NSE": float(finite.median()) if len(finite) else None,
                   "finite_NSE_IDs": int(len(finite)), "paired_day_IDs": paired,
                   "prediction_day_IDs": predictions}


def a3_dlinear_training_contract(seed):
    config_path = A3_OUT_DIR / "A3_DLinear_config.json"
    if a3_hash_file(config_path) != A3_DLINEAR_CONFIG_SHA256:
        raise ValueError("Config trên đĩa khác config đã kiểm")
    if json.loads(config_path.read_text(encoding="utf-8")) != json.loads(json.dumps(A3_DLINEAR_CONFIG)):
        raise ValueError("Config trong RAM đã đổi")
    for item in A3_DLINEAR_PREPROCESSING_MANIFEST.itertuples():
        if a3_hash_file(A3_OUT_DIR / item.file) != item.sha256:
            raise ValueError(f"Tệp tiền xử lý khác manifest: {item.file}")
    with np.load(A3_OUT_DIR / "A3_DLinear_prepared_data.npz", allow_pickle=False) as stored:
        for key, values in (("X_z", A3_DLINEAR_X), ("Y_z", A3_DLINEAR_Y),
                            ("sigma_train_mm_day", A3_DLINEAR_SIGMA_TRAIN.reindex(A3_SELECTED_IDS).to_numpy())):
            if not np.array_equal(stored[key], values, equal_nan=True):
                raise ValueError(f"{key}: RAM khác dữ liệu tiền xử lý đã lưu")
        if list(stored["basin_ids"]) != A3_SELECTED_IDS:
            raise ValueError("Thứ tự ID khác dữ liệu đã lưu")
    with np.load(A3_OUT_DIR / "A3_DLinear_indices.npz", allow_pickle=False) as stored:
        for period in A3_PERIODS:
            for kind, mapping in (("samples", A3_DLINEAR_SAMPLE_INDEX), ("inference", A3_DLINEAR_INFERENCE_INDEX)):
                if not np.array_equal(stored[f"{period}_{kind}"], mapping[period]):
                    raise ValueError("Chỉ mục trong RAM khác chỉ mục đã lưu")
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Tính xác định đã bị tắt sau Phần 8")
    if (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.backends.cudnn.benchmark or torch.get_float32_matmul_precision() != "highest"):
        raise ValueError("Thiết lập số học khác Phần 8/config; chưa tự đổi cấu hình để tiếp tục")
    return {
        "seed": seed, "config_sha256": A3_DLINEAR_CONFIG_SHA256,
        "preprocessing_manifest_sha256": a3_hash_file(A3_OUT_DIR / "A3_DLinear_preprocessing_manifest.csv"),
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                    "device": str(A3_DLINEAR_DEVICE), "cpu_threads": torch.get_num_threads(),
                    "device_name": torch.cuda.get_device_name(A3_DLINEAR_DEVICE) if A3_DLINEAR_DEVICE.type == "cuda" else "CPU",
                    "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "deterministic": True, "matmul_precision": torch.get_float32_matmul_precision(),
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
    }


def a3_export_dlinear_prediction(path, period, raw, clipped, seed, epoch):
    dates = raw.index.to_numpy(dtype="datetime64[D]")
    source = A3_DLINEAR_Q_MM_DAY["qmax"].loc[raw.index, A3_SELECTED_IDS].to_numpy()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, dates=dates, source_last_dates=dates - np.timedelta64(1, "D"),
            basin_ids=np.asarray(A3_SELECTED_IDS, dtype="U"), area_gov_km2=A3_AREAS.to_numpy(),
            obs_mm_day=source, sim_raw_mm_day=raw.to_numpy(), sim_mm_day=clipped.to_numpy(),
            seed=np.int32(seed), selected_epoch=np.int32(epoch), period=np.asarray(period),
            config_sha256=np.asarray(A3_DLINEAR_CONFIG_SHA256),
        )
    with np.load(temporary, allow_pickle=False) as saved:
        if not np.array_equal(saved["dates"], dates) or list(saved["basin_ids"]) != A3_SELECTED_IDS:
            raise ValueError("Export sai lịch hoặc thứ tự ID")
        for key, values in (("obs_mm_day", source), ("sim_raw_mm_day", raw.to_numpy()), ("sim_mm_day", clipped.to_numpy())):
            if not np.array_equal(saved[key], values, equal_nan=True):
                raise ValueError(f"Đọc lại export khác dữ liệu: {key}")
    temporary.replace(path)

# %% Phần 9.3 — Hàm chạy một seed DLinear
# Khởi tạo riêng từng seed; chọn best bằng validation và ghi test chỉ để theo dõi.
def a3_run_dlinear_seed(seed):
    if seed not in A3_DLINEAR_CONFIG["seeds"]:
        raise ValueError("Seed không nằm trong config đã cố định")
    contract = a3_dlinear_training_contract(seed)
    root = A3_OUT_DIR / "dlinear" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    info_path, last_path = root / "run_info.json", root / "last.pt"
    if info_path.exists():
        if json.loads(info_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Seed cũ khác config/nguồn/môi trường; không trộn hai lần chạy")
    else:
        if last_path.exists():
            raise ValueError("Có checkpoint nhưng thiếu nguồn run_info; chưa tự khôi phục")
        a3_atomic_json(info_path, contract)
    summary_path, manifest_path = root / "summary.json", root / "manifest.csv"
    if summary_path.exists() and manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype={"file": str, "sha256": str})
        if all((root / name).is_file() for name in manifest.file):
            if not all(a3_hash_file(root / item.file) == item.sha256 for item in manifest.itertuples()):
                raise ValueError("Gói seed đã hoàn thành có tệp đổi hash; gửi output trước khi dùng lại")
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            if saved.get("complete") is True and saved.get("config_sha256") == contract["config_sha256"]:
                print(f"Seed {seed}: đã hoàn thành, mọi tệp khớp hash; đọc lại kết quả, không train thêm.")
                return saved

    training = A3_DLINEAR_CONFIG["training"]
    device = A3_DLINEAR_DEVICE
    cuda_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.default_generators[device.index].manual_seed(seed)
        generator, python_rng = np.random.default_rng(seed), random.Random(seed)
        model = A3DLinear(A3_DLINEAR_CONFIG["architecture"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training["scheduler_T_max"], eta_min=training["scheduler_eta_min"],
        )
        selection = {"best_score": None, "best_epoch": None, "significant_anchor": None, "bad_epochs": 0}
        history, best_state, completed_epoch = [], None, 0
        if last_path.exists():
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
            if checkpoint.get("contract") != contract or checkpoint.get("role") != "official_seed_epoch_boundary":
                raise ValueError("Checkpoint khác hợp đồng hoặc không phải trạng thái cuối epoch chính thức")
            history = checkpoint["history"]
            completed_epoch = int(checkpoint["epoch"])
            if completed_epoch < 1 or completed_epoch > training["max_epochs"]:
                raise ValueError("Epoch trong checkpoint ngoài cấu hình")
            if [item["epoch"] for item in history] != list(range(1, completed_epoch + 1)):
                raise ValueError("Lịch sử epoch không liên tục")
            replay = dict(selection)
            for item in history:
                replay, _, _ = a3_update_dlinear_selection(
                    replay, item["validation_median_NSE"], item["epoch"],
                    training["early_stopping_min_delta"], training["early_stopping_patience"],
                )
            if replay != checkpoint["selection"]:
                raise ValueError("Bộ chọn checkpoint không khớp lịch sử validation")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            generator.bit_generator.state = checkpoint["numpy_generator_state"]
            python_rng.setstate(checkpoint["python_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device)
            selection, best_state = checkpoint["selection"], checkpoint["best_model_state"]
            print(f"Seed {seed}: tiếp tục sau epoch {completed_epoch}; best epoch {selection['best_epoch']}.")
            del checkpoint
        else:
            print(f"Seed {seed}: khởi tạo mới; không nạp trọng số chạy thử.")

        for epoch in range(completed_epoch + 1, training["max_epochs"] + 1):
            if selection["bad_epochs"] >= training["early_stopping_patience"]:
                break
            print(f"Seed {seed} — epoch {epoch}/{training['max_epochs']}: train", flush=True)
            lr_used = float(optimizer.param_groups[0]["lr"])
            a3_dlinear_sync(device)
            started = time.perf_counter()
            train_result = a3_dlinear_train_epoch(model, optimizer, device, generator, progress_every=0)
            scheduler.step()
            a3_dlinear_sync(device)
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            validation_raw, validation_clipped = a3_dlinear_predict(model, "validation", device)
            _, validation = a3_dlinear_nse_table("validation", validation_raw, validation_clipped)
            a3_dlinear_sync(device)
            validation_seconds = time.perf_counter() - started
            selection, best_improved, should_stop = a3_update_dlinear_selection(
                selection, validation["median_NSE"], epoch,
                training["early_stopping_min_delta"], training["early_stopping_patience"],
            )
            if best_improved:
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            # Test chỉ ghi nhận; không truyền giá trị test vào bộ chọn hoặc bộ đếm dừng.
            started = time.perf_counter()
            test_raw, test_clipped = a3_dlinear_predict(model, "test", device)
            _, test = a3_dlinear_nse_table("test", test_raw, test_clipped)
            a3_dlinear_sync(device)
            test_seconds = time.perf_counter() - started
            history.append({
                "epoch": epoch, "learning_rate": lr_used, "train_loss": train_result["train_loss"],
                "train_samples": train_result["samples"], "train_batches": train_result["batches"],
                "last_batch": train_result["last_batch"],
                "validation_median_NSE": validation["median_NSE"], "test_median_NSE": test["median_NSE"],
                "best_epoch": selection["best_epoch"], "best_validation_NSE": selection["best_score"],
                "best_improved": best_improved, "bad_epochs": selection["bad_epochs"],
                "train_seconds": train_seconds, "validation_seconds": validation_seconds, "test_seconds": test_seconds,
            })
            a3_atomic_torch_save(last_path, {
                "role": "official_seed_epoch_boundary", "contract": contract, "epoch": epoch,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "selection": selection,
                "best_model_state": best_state, "history": history,
                "numpy_generator_state": generator.bit_generator.state, "python_rng_state": python_rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            })
            a3_atomic_csv(root / "history.csv", pd.DataFrame(history))
            completed_epoch = epoch
            test_text = f"{test['median_NSE']:.6f}" if test["median_NSE"] is not None else "undefined"
            print(f"epoch {epoch}: loss={train_result['train_loss']:.6f} | val={validation['median_NSE']:.6f} | "
                  f"test={test_text} | best={selection['best_epoch']} | patience={selection['bad_epochs']}/"
                  f"{training['early_stopping_patience']}", flush=True)
            if should_stop:
                break

        if best_state is None or not history:
            raise RuntimeError("Chưa có checkpoint validation hợp lệ")
        best_epoch = int(selection["best_epoch"])
        model.load_state_dict(best_state, strict=True)
        a3_atomic_torch_save(root / "best.pt", {
            "role": "official_best_validation_for_inference", "contract": contract,
            "selected_epoch": best_epoch, "validation_median_NSE": selection["best_score"],
            "model_state": best_state,
        })
        selected = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(selected["model_state"], strict=True)
        exports, profile_tables = {}, []
        for period in ("validation", "test"):
            raw, clipped = a3_dlinear_predict(model, period, device)
            table, stats = a3_dlinear_nse_table(period, raw, clipped)
            exports[period] = stats
            a3_atomic_csv(root / f"{period}_nse.csv", table)
            a3_export_dlinear_prediction(root / f"{period}_predictions.npz", period, raw, clipped, seed, best_epoch)
            profile_tables.append(table.loc[table.basin_id.isin([*A3_DLINEAR_PROFILE_IDS, "295"])])
        if not np.isclose(exports["validation"]["median_NSE"], selection["best_score"], atol=1e-12, rtol=1e-12):
            raise ValueError("Dự báo từ best.pt không tái hiện NSE validation đã chọn")
        a3_atomic_csv(root / "profile_nse.csv", pd.concat(profile_tables, ignore_index=True))
        a3_atomic_csv(root / "history.csv", pd.DataFrame(history))
        summary = {
            "complete": True, "seed": seed, "config_sha256": contract["config_sha256"],
            "selected_by": "validation_median_NSE_only", "selected_epoch": best_epoch,
            "best_validation_median_NSE": selection["best_score"], "epochs_completed": completed_epoch,
            "stop_reason": "patience" if selection["bad_epochs"] >= training["early_stopping_patience"] else "max_epochs",
            "training_samples_per_epoch": len(A3_DLINEAR_SAMPLE_INDEX["train"]),
            "train_validation_test_epoch_seconds": float(sum(
                item["train_seconds"] + item["validation_seconds"] + item["test_seconds"] for item in history
            )),
            "epoch_timing_excludes": "checkpoint I/O, setup and final export",
            "validation": exports["validation"], "test": exports["test"],
            "runtime": contract["runtime"], "output_dir": str(root),
        }
        a3_atomic_json(summary_path, summary)
        filenames = ["run_info.json", "last.pt", "best.pt", "history.csv", "summary.json", "profile_nse.csv",
                     "validation_nse.csv", "test_nse.csv", "validation_predictions.npz", "test_predictions.npz"]
        a3_atomic_csv(manifest_path, pd.DataFrame([
            {"file": name, "bytes": (root / name).stat().st_size, "sha256": a3_hash_file(root / name)} for name in filenames
        ]))
        return summary

# %% Phần 9.4 — Kiểm chọn checkpoint và patience
# Kiểm bằng chuỗi điểm nhỏ, gồm cải thiện nhỏ và trường hợp bằng điểm.
a3_selection_checks = []
initial_selection = {"best_score": None, "best_epoch": None, "significant_anchor": None, "bad_epochs": 0}
state = dict(initial_selection)
for epoch, score in enumerate([.5, .502, .504, .506], 1):
    state, _, stopped = a3_update_dlinear_selection(state, score, epoch, .005, 5)
a3_selection_checks.append({"case": "small_improvements_saved_significant_anchor_resets_later",
                            "passed": state["best_epoch"] == 4 and state["bad_epochs"] == 0 and not stopped})
state = dict(initial_selection)
for epoch, score in enumerate([.5, .501, .502, .503, .504, .505], 1):
    state, _, stopped = a3_update_dlinear_selection(state, score, epoch, .005, 5)
a3_selection_checks.append({"case": "save_small_best_on_patience_stop", "passed": state["best_epoch"] == 6 and stopped})
same, improved, _ = a3_update_dlinear_selection(state, .505, 7, .005, 5)
a3_selection_checks.append({"case": "ties_keep_earlier_best", "passed": not improved and same["best_epoch"] == 6})
if not all(item["passed"] for item in a3_selection_checks):
    raise AssertionError("Bộ chọn checkpoint/dừng sớm không đạt kiểm")


show_table('Kiểm riêng checkpoint tốt nhất và dừng sớm', a3_selection_checks)

# %% Phần 9.5 — Huấn luyện và báo cáo seed 111
# Chạy seed 111 theo config đã cố định; xuất best/last và dự báo từng kỳ.
show_table('Nguồn và cấu hình cố định', [
    {"item": "seed", "value": 111}, {"item": "config_sha256", "value": A3_DLINEAR_CONFIG_SHA256},
    {"item": "checkpoint_selection", "value": "validation_median_NSE only; test logged each epoch"},
    {"item": "early_stopping", "value": "patience=5; min_delta=0.005; save every strict validation improvement"},
    {"item": "output", "value": str(A3_OUT_DIR / "dlinear" / "seed_111")},
])
A3_DLINEAR_SEED_111_OK = False
A3_DLINEAR_SEED_111_REPORT = a3_run_dlinear_seed(111)
A3_DLINEAR_SEED_111_OK = bool(A3_DLINEAR_SEED_111_REPORT["complete"])
if "A3_DLINEAR_SEED_REPORTS" not in globals():
    A3_DLINEAR_SEED_REPORTS = {}
A3_DLINEAR_SEED_REPORTS[111] = A3_DLINEAR_SEED_111_REPORT
a3_seed111_root = A3_OUT_DIR / "dlinear" / "seed_111"
with pd.option_context("display.max_columns", None):
    history = pd.read_csv(a3_seed111_root / "history.csv")
    show_table('Lịch sử seed 111 — test chỉ ghi nhận', history[[
        "epoch", "learning_rate", "train_loss", "validation_median_NSE", "test_median_NSE",
        "best_epoch", "best_improved", "bad_epochs", "train_seconds", "validation_seconds", "test_seconds",
    ]].to_dict("records"))
    show_table('Kết quả từ checkpoint tốt nhất', [
        {"item": key, "value": json.dumps(value, ensure_ascii=False)}
        for key, value in A3_DLINEAR_SEED_111_REPORT.items()
    ])
    show_table('Độ phủ/NSE ở các hồ sơ A1', pd.read_csv(
        a3_seed111_root / "profile_nse.csv", dtype={"basin_id": str},
    ).to_dict("records"))

print("A3_DLINEAR_SEED_111_OK =", A3_DLINEAR_SEED_111_OK)
print("Đã xuất best/last checkpoint, lịch sử, dự báo validation/test, bảng NSE và manifest tại:", a3_seed111_root)
print("Chỉ chạy seed 111 trong cell này; chưa chạy 222/333, chưa có bảng so sánh chung ba seed.")

# %% Phần 9.6 — Huấn luyện và báo cáo seed 222
# Kiểm kết quả seed trước, khởi tạo trọng số riêng và giữ cùng config.
A3_DLINEAR_SEED_222_OK = False
if not globals().get("A3_DLINEAR_SEED_111_OK", False) or "a3_run_dlinear_seed" not in globals():
    raise RuntimeError("Cần seed 111 hoàn tất và hàm chạy seed từ Phần 9.1 trong cùng phiên.")

a3_seed111_root = A3_OUT_DIR / "dlinear" / "seed_111"
with (a3_seed111_root / "summary.json").open(encoding="utf-8") as stream:
    a3_seed111_saved = json.load(stream)
if (a3_seed111_saved.get("complete") is not True or a3_seed111_saved.get("seed") != 111
        or a3_seed111_saved.get("config_sha256") != A3_DLINEAR_CONFIG_SHA256):
    raise ValueError("Gói seed 111 chưa hoàn tất hoặc khác cấu hình hiện tại.")
a3_seed111_manifest = pd.read_csv(a3_seed111_root / "manifest.csv", dtype={"file": str, "sha256": str})
for record in a3_seed111_manifest.itertuples():
    path = a3_seed111_root / record.file
    if not path.is_file() or a3_hash_file(path) != record.sha256:
        raise ValueError(f"Gói seed 111 thiếu tệp hoặc khác manifest: {record.file}")


show_table('Cấu hình giữ nguyên và tách nguồn trọng số', [
    {"item": "seed", "value": 222},
    {"item": "config_sha256", "value": A3_DLINEAR_CONFIG_SHA256},
    {"item": "seed_111_artifacts_verified", "value": True},
    {"item": "initialization", "value": "new nn.Linear weights from seed 222; never load seed 111 or smoke"},
    {"item": "selection", "value": "same validation-only best/patience rules; test logged only"},
    {"item": "output", "value": str(A3_OUT_DIR / "dlinear" / "seed_222")},
])
A3_DLINEAR_SEED_222_REPORT = a3_run_dlinear_seed(222)
A3_DLINEAR_SEED_222_OK = bool(A3_DLINEAR_SEED_222_REPORT["complete"])
A3_DLINEAR_SEED_REPORTS[222] = A3_DLINEAR_SEED_222_REPORT
a3_seed222_root = A3_OUT_DIR / "dlinear" / "seed_222"
with pd.option_context("display.max_columns", None):
    history_222 = pd.read_csv(a3_seed222_root / "history.csv")
    show_table('Lịch sử seed 222 — test chỉ ghi nhận', history_222[[
        "epoch", "learning_rate", "train_loss", "validation_median_NSE", "test_median_NSE",
        "best_epoch", "best_improved", "bad_epochs", "train_seconds", "validation_seconds", "test_seconds",
    ]].to_dict("records"))
    show_table('Kết quả seed 222 từ checkpoint tốt nhất', [
        {"item": key, "value": json.dumps(value, ensure_ascii=False)}
        for key, value in A3_DLINEAR_SEED_222_REPORT.items()
    ])
    show_table('Độ phủ/NSE ở các hồ sơ A1', pd.read_csv(
        a3_seed222_root / "profile_nse.csv", dtype={"basin_id": str},
    ).to_dict("records"))

print("A3_DLINEAR_SEED_222_OK =", A3_DLINEAR_SEED_222_OK)
print("Đã xuất gói seed 222 tại:", a3_seed222_root)
print("Seed 111 được giữ nguyên; chưa chạy 333, chưa tổng hợp hoặc chọn seed theo test.")

# %% Phần 9.7 — Huấn luyện seed 333 và kiểm đủ ba seed
# Kiểm 111/222, chạy 333 và xác nhận ba kết quả hoàn tất với cùng config.
A3_DLINEAR_SEED_333_OK = False
A3_DLINEAR_THREE_SEEDS_OK = False
if not (globals().get("A3_DLINEAR_SEED_111_OK", False)
        and globals().get("A3_DLINEAR_SEED_222_OK", False)
        and "a3_run_dlinear_seed" in globals()):
    raise RuntimeError("Cần 111/222 hoàn tất và runner 9.1 trong cùng phiên trước seed 333.")

a3_previous_seed_checks = []
for previous_seed in (111, 222):
    previous_root = A3_OUT_DIR / "dlinear" / f"seed_{previous_seed}"
    saved_summary = json.loads((previous_root / "summary.json").read_text(encoding="utf-8"))
    if (saved_summary.get("complete") is not True or saved_summary.get("seed") != previous_seed
            or saved_summary.get("config_sha256") != A3_DLINEAR_CONFIG_SHA256
            or saved_summary.get("selected_by") != "validation_median_NSE_only"):
        raise ValueError(f"Gói seed {previous_seed} chưa đạt hoặc khác hợp đồng hiện tại")
    manifest = pd.read_csv(previous_root / "manifest.csv", dtype={"file": str, "sha256": str})
    for item in manifest.itertuples():
        path = previous_root / item.file
        if not path.is_file() or a3_hash_file(path) != item.sha256:
            raise ValueError(f"Seed {previous_seed}: thiếu tệp hoặc khác hash {item.file}")
    A3_DLINEAR_SEED_REPORTS[previous_seed] = saved_summary
    a3_previous_seed_checks.append({"seed": previous_seed, "complete": True,
                                    "same_config": True, "manifest_files_verified": len(manifest)})


show_table('Kiểm hai gói seed trước', a3_previous_seed_checks)
show_table('Cấu hình giữ nguyên, khởi tạo trọng số riêng', [
    {"item": "seed", "value": 333},
    {"item": "config_sha256", "value": A3_DLINEAR_CONFIG_SHA256},
    {"item": "initialization", "value": "new nn.Linear weights from seed 333; never load 111/222 or smoke"},
    {"item": "selection", "value": "same validation-only best/patience rules; test logged only"},
    {"item": "output", "value": str(A3_OUT_DIR / "dlinear" / "seed_333")},
])
A3_DLINEAR_SEED_333_REPORT = a3_run_dlinear_seed(333)
A3_DLINEAR_SEED_333_OK = bool(A3_DLINEAR_SEED_333_REPORT["complete"])
A3_DLINEAR_SEED_REPORTS[333] = A3_DLINEAR_SEED_333_REPORT
A3_DLINEAR_THREE_SEEDS_OK = all(
    A3_DLINEAR_SEED_REPORTS.get(seed, {}).get("complete") is True
    and A3_DLINEAR_SEED_REPORTS[seed].get("seed") == seed
    and A3_DLINEAR_SEED_REPORTS[seed].get("config_sha256") == A3_DLINEAR_CONFIG_SHA256
    for seed in A3_DLINEAR_CONFIG["seeds"]
)
a3_seed333_root = A3_OUT_DIR / "dlinear" / "seed_333"
with pd.option_context("display.max_columns", None):
    history_333 = pd.read_csv(a3_seed333_root / "history.csv")
    show_table('Lịch sử seed 333 — test chỉ ghi nhận', history_333[[
        "epoch", "learning_rate", "train_loss", "validation_median_NSE", "test_median_NSE",
        "best_epoch", "best_improved", "bad_epochs", "train_seconds", "validation_seconds", "test_seconds",
    ]].to_dict("records"))
    show_table('Kết quả seed 333 từ checkpoint tốt nhất', [
        {"item": key, "value": json.dumps(value, ensure_ascii=False)}
        for key, value in A3_DLINEAR_SEED_333_REPORT.items()
    ])
    show_table('Độ phủ/NSE ở các hồ sơ A1', pd.read_csv(
        a3_seed333_root / "profile_nse.csv", dtype={"basin_id": str},
    ).to_dict("records"))

print("A3_DLINEAR_SEED_333_OK =", A3_DLINEAR_SEED_333_OK)
print("A3_DLINEAR_THREE_SEEDS_OK =", A3_DLINEAR_THREE_SEEDS_OK)
print("Đã xuất gói seed 333 tại:", a3_seed333_root)
print("Hai gói 111/222 giữ nguyên; chưa tính lại NSE/KGE/PNSE trên giao chung hoặc tổng hợp ba seed.")

# %% Phần 10.1 — Hàm đọc export và chấm cùng phạm vi
# Kiểm ngày/ID/obs/clip và tính chỉ số từ các tệp DLinear đã lưu.
A3_FINAL_COMPARISON_OK = False
if not (globals().get("A3_DLINEAR_THREE_SEEDS_OK", False)
        and globals().get("A3_BASELINE_EVAL_OK", False)
        and globals().get("A3_METRIC_TESTS_OK", False)):
    raise RuntimeError("Cần đủ ba seed, đối chiếu mốc Phần 6 và hàm chỉ số đã đạt trước Phần 10")

a3_final_started = time.perf_counter()
A3_FINAL_COMMON_SCOPE = "common_source_all"
A3_DLINEAR_DAILY, A3_DLINEAR_SCORE_MASKS = {}, {}
a3_final_audit_rows, a3_new_score_rows = [], []


def a3_final_score(run, scope, observed, predicted, mask, persistence=None, seed=None, selected_epoch=None, clip=True):
    for frame in (observed, predicted, mask) + ((persistence,) if persistence is not None else ()):
        if not frame.index.equals(observed.index) or list(frame.columns) != A3_SELECTED_IDS:
            raise ValueError(f"{run}/{scope}: trục ngày/ID không đồng nhất")
    rows = []
    for basin_id in A3_SELECTED_IDS:
        result = a3_metrics(
            observed[basin_id].to_numpy(), predicted[basin_id].to_numpy(),
            persistence[basin_id].to_numpy() if persistence is not None else None,
            mask[basin_id].to_numpy(),
        )
        if result["n"] != int(mask[basin_id].sum()):
            raise AssertionError("Hàm chỉ số tự đổi số ngày chấm")
        persistence_nse = (
            1 - result["sse_persistence"] / result["obs_tss"]
            if np.isfinite(result["sse_persistence"]) and result["obs_tss"] > 0 else np.nan
        )
        rows.append({"run": run, "scope": scope, "observation_source": "source_Q_area_gov",
                     "basin_id": basin_id, "seed": seed, "selected_epoch": selected_epoch,
                     "clip_to_zero": clip, **result, "NSE_persistence_same_mask": persistence_nse})
    return rows


def a3_read_dlinear_export(seed, period, summary):
    root = A3_OUT_DIR / "dlinear" / f"seed_{seed}"
    path = root / f"{period}_predictions.npz"
    calendar = pd.date_range(*A3_PERIODS[period], freq="D")
    source = A3_DLINEAR_Q_MM_DAY["qmax"].loc[calendar, A3_SELECTED_IDS]
    with np.load(path, allow_pickle=False) as saved:
        if (not np.array_equal(saved["dates"], calendar.to_numpy(dtype="datetime64[D]"))
                or not np.array_equal(saved["source_last_dates"], calendar.to_numpy(dtype="datetime64[D]") - np.timedelta64(1, "D"))
                or list(saved["basin_ids"]) != A3_SELECTED_IDS):
            raise ValueError("Export sai lịch ngày/thứ tự ID/ngày cuối input")
        if (int(saved["seed"]) != seed or str(saved["period"].item()) != period
                or int(saved["selected_epoch"]) != summary["selected_epoch"]
                or str(saved["config_sha256"].item()) != A3_DLINEAR_CONFIG_SHA256):
            raise ValueError("Metadata seed/epoch/kỳ/config trong NPZ không khớp summary")
        if not np.array_equal(saved["area_gov_km2"], A3_AREAS.to_numpy()):
            raise ValueError("Diện tích NPZ khác area_gov đã dùng")
        arrays = {key: saved[key].copy() for key in ("obs_mm_day", "sim_raw_mm_day", "sim_mm_day")}
    if any(array.shape != source.shape or np.isinf(array).any() for array in arrays.values()):
        raise ValueError("Mảng export sai shape hoặc có vô cực")
    if not np.array_equal(arrays["obs_mm_day"], source.to_numpy(), equal_nan=True):
        raise ValueError("Obs export không trùng qmax nguồn mm/ngày")
    raw, sim = arrays["sim_raw_mm_day"], arrays["sim_mm_day"]
    if not np.array_equal(sim, np.maximum(raw, 0), equal_nan=True):
        raise ValueError("Sim scored không phải clip raw tại 0 mm/ngày")
    if not np.array_equal(np.isfinite(raw), A3_DLINEAR_HISTORY_VALID.loc[calendar].to_numpy()):
        raise ValueError("Mặt nạ dự báo không khớp đầu vào 365 ngày đã cố định")
    panels = {"obs": source, "sim_raw": pd.DataFrame(raw, index=calendar, columns=A3_SELECTED_IDS),
              "sim": pd.DataFrame(sim, index=calendar, columns=A3_SELECTED_IDS)}
    mask = np.isfinite(source) & np.isfinite(panels["sim"])
    native = pd.DataFrame(a3_final_score(f"DLinear_seed_{seed}", "native_source", source, panels["sim"], mask, seed=seed))
    csv = pd.read_csv(root / f"{period}_nse.csv", dtype={"basin_id": str})
    if csv.basin_id.duplicated().any() or set(csv.basin_id) != set(A3_SELECTED_IDS) or not csv.period.eq(period).all():
        raise ValueError("CSV NSE không có đúng tập ID/kỳ")
    csv, native = csv.set_index("basin_id").loc[A3_SELECTED_IDS], native.set_index("basin_id").loc[A3_SELECTED_IDS]
    if (not np.array_equal(csv.n.to_numpy(), native.n.to_numpy())
            or not np.array_equal(csv.NSE_status.to_numpy(), native.NSE_status.to_numpy())
            or not np.array_equal(np.isfinite(csv.NSE), np.isfinite(native.NSE))
            or not np.allclose(csv.NSE, native.NSE, equal_nan=True, atol=1e-12, rtol=1e-12)):
        raise ValueError("Tính lại NSE/số cặp/trạng thái từ NPZ không khớp CSV seed")
    finite = native.NSE[np.isfinite(native.NSE)]
    median = float(finite.median()) if len(finite) else None
    stats = summary[period]
    if (median is None or not np.isclose(median, stats["median_NSE"], atol=1e-12, rtol=1e-12)
            or len(finite) != stats["finite_NSE_IDs"] or int(native.n.sum()) != stats["paired_day_IDs"]
            or int(np.isfinite(sim).sum()) != stats["prediction_day_IDs"]):
        raise ValueError("NSE/độ phủ tính lại không khớp summary")
    differences = np.abs(csv.NSE.to_numpy() - native.NSE.to_numpy())
    return panels, {"paired_day_IDs": int(native.n.sum()), "finite_NSE_IDs": len(finite),
                    "median_NSE": median, "max_abs_NSE_csv_difference": float(np.nanmax(differences)),
                    "raw_negative_days": int((raw < 0).sum())}

# %% Phần 10.2 — Kiểm export ba seed DLinear
# So manifest, checkpoint, NPZ và CSV của validation/test.
print("Đọc NPZ/CSV/checkpoint trên CPU; không huấn luyện hoặc suy luận GPU lại.")
expected_seed_files = {"run_info.json", "last.pt", "best.pt", "history.csv", "summary.json", "profile_nse.csv",
                       "validation_nse.csv", "test_nse.csv", "validation_predictions.npz", "test_predictions.npz"}
for seed in A3_DLINEAR_CONFIG["seeds"]:
    run = f"DLinear_seed_{seed}"
    root = A3_OUT_DIR / "dlinear" / f"seed_{seed}"
    try:
        manifest = pd.read_csv(root / "manifest.csv", dtype={"file": str, "sha256": str})
        if manifest.file.duplicated().any() or set(manifest.file) != expected_seed_files:
            raise ValueError("Manifest không có đúng mười tệp của seed")
        for item in manifest.itertuples():
            if a3_hash_file(root / item.file) != item.sha256:
                raise ValueError(f"Tệp seed khác manifest: {item.file}")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if (summary.get("complete") is not True or summary.get("seed") != seed
                or summary.get("config_sha256") != A3_DLINEAR_CONFIG_SHA256
                or summary.get("selected_by") != "validation_median_NSE_only"):
            raise ValueError("Summary chưa hoàn tất/khác seed/config/cách chọn")
        history = pd.read_csv(root / "history.csv")
        best_row = history.loc[history.validation_median_NSE.idxmax()]
        if int(best_row.epoch) != summary["selected_epoch"]:
            raise ValueError("Epoch chọn không phải best validation trong lịch sử")
        best = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
        if (best.get("role") != "official_best_validation_for_inference"
                or best["contract"]["seed"] != seed
                or best["contract"]["config_sha256"] != A3_DLINEAR_CONFIG_SHA256
                or best["selected_epoch"] != summary["selected_epoch"]):
            raise ValueError("Metadata best.pt khác hợp đồng seed")
        del best
        for period in ("validation", "test"):
            panels, audit = a3_read_dlinear_export(seed, period, summary)
            a3_final_audit_rows.append({"seed": seed, "period": period, "status": "ok",
                                        "selected_epoch": summary["selected_epoch"], **audit, "issues": ""})
            if period == "test":
                A3_DLINEAR_DAILY[run] = {**panels, "seed": seed, "selected_epoch": summary["selected_epoch"]}
    except Exception as exc:
        a3_final_audit_rows.append({"seed": seed, "period": "source_or_export", "status": "failed",
                                    "issues": f"{type(exc).__name__}: {exc}"})
A3_DLINEAR_EXPORT_AUDIT = pd.DataFrame(a3_final_audit_rows)
A3_DLINEAR_EXPORT_AUDIT.to_csv(A3_OUT_DIR / "A3_DLinear_export_audit.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.max_columns", None):
    show_table('Kiểm nguồn/export và đối chiếu lại NSE của ba seed', a3_final_audit_rows)
if len(a3_final_audit_rows) != 2 * len(A3_DLINEAR_CONFIG["seeds"]) or not A3_DLINEAR_EXPORT_AUDIT.status.eq("ok").all():
    raise RuntimeError("Export chưa đạt; giữ audit và gửi output, chưa tổng hợp điểm")

# %% Phần 10.3 — Lập giao ngày thật của mọi mô hình
# Thêm ba DLinear vào giao A2/persistence bằng giá trị dự báo thực.
A3_FINAL_SCORED_SIM = {}
for run in EXPECTED_RUNS:
    for filename in ("config.yml", "test_metrics.csv"):
        path = A3_A2_DIR / "runs" / run / filename
        recorded = A3_INPUT_MANIFEST.loc[A3_INPUT_MANIFEST.run.eq(run)
            & A3_INPUT_MANIFEST.source.eq("A2_rerun_export") & A3_INPUT_MANIFEST.file.eq(filename), "sha256"]
        if len(recorded) != 1 or a3_hash_file(path) != recorded.iloc[0]:
            raise ValueError(f"Nguồn A2 đã đổi: {run}/{filename}")
    raw = A3_DAILY[run]["sim_raw"]
    clip = "qmax" in a3_names(A3_CONFIGS[run].get("clip_targets_to_zero"))
    scored = pd.DataFrame(a3_sim_for_scoring(raw.to_numpy(), clip), index=raw.index, columns=raw.columns)
    if not np.array_equal(scored.to_numpy(), A3_SCORED_SIM[run].to_numpy(), equal_nan=True):
        raise ValueError("Bản sim A2 trong RAM khác chính sách đã kiểm ở Phần 6")
    A3_FINAL_SCORED_SIM[run] = scored
old_common = np.isfinite(A3_SOURCE_OBS) & np.isfinite(A3_PERSISTENCE_QMAX)
for run in EXPECTED_RUNS:
    old_common &= np.isfinite(A3_FINAL_SCORED_SIM[run])
if not np.array_equal(old_common.to_numpy(), A3_COMMON_MASK.to_numpy()):
    raise ValueError("Giao A2/persistence đọc lại khác Phần 4")
A3_ALL_COMMON_MASK = old_common.copy()
for run, data in A3_DLINEAR_DAILY.items():
    if not data["obs"].index.equals(A3_TEST_DATES) or not np.array_equal(
        data["obs"].to_numpy(), A3_SOURCE_OBS.to_numpy(), equal_nan=True,
    ):
        raise ValueError("Obs test DLinear khác obs nguồn dùng so sánh")
    native = np.isfinite(A3_SOURCE_OBS) & np.isfinite(data["sim"])
    paired = native & np.isfinite(A3_PERSISTENCE_QMAX)
    A3_DLINEAR_SCORE_MASKS[run] = {"native_source": native, "paired_source": paired}
    A3_FINAL_SCORED_SIM[run] = data["sim"]
    A3_ALL_COMMON_MASK &= paired
    for scope, mask, persistence in (("native_source", native, None), ("paired_source", paired, A3_PERSISTENCE_QMAX)):
        a3_new_score_rows.extend(a3_final_score(run, scope, A3_SOURCE_OBS, data["sim"], mask,
            persistence, seed=data["seed"], selected_epoch=data["selected_epoch"]))
A3_FINAL_MAIN_RUNS = [*EXPECTED_RUNS, *A3_DLINEAR_DAILY, "Persistence_qmax"]
A3_FINAL_SCORED_SIM["Persistence_qmax"] = A3_PERSISTENCE_QMAX
A3_FINAL_SCORED_SIM["Persistence_qmean_aux"] = A3_PERSISTENCE_QMEAN_AUX
A3_AUX_FULL_FINAL_COMMON = bool(np.isfinite(A3_PERSISTENCE_QMEAN_AUX.to_numpy()[A3_ALL_COMMON_MASK.to_numpy()]).all())
for run in [*A3_FINAL_MAIN_RUNS, "Persistence_qmean_aux"]:
    auxiliary_partial = run == "Persistence_qmean_aux" and not A3_AUX_FULL_FINAL_COMMON
    scope = A3_FINAL_COMMON_SCOPE + "_aux_available" if auxiliary_partial else A3_FINAL_COMMON_SCOPE
    mask = A3_ALL_COMMON_MASK & np.isfinite(A3_FINAL_SCORED_SIM[run]) if auxiliary_partial else A3_ALL_COMMON_MASK
    metadata = A3_DLINEAR_DAILY.get(run, {})
    clip = "qmax" in a3_names(A3_CONFIGS[run].get("clip_targets_to_zero")) if run in EXPECTED_RUNS else run in A3_DLINEAR_DAILY
    a3_new_score_rows.extend(a3_final_score(run, scope, A3_SOURCE_OBS, A3_FINAL_SCORED_SIM[run], mask,
        A3_PERSISTENCE_QMAX, seed=metadata.get("seed"), selected_epoch=metadata.get("selected_epoch"), clip=clip))

# %% Phần 10.4 — Tổng hợp sáu chỉ số trên từng phạm vi
# Giữ bảng A2 gốc và báo số ID hữu hạn cho từng chỉ số.
historical = A3_METRICS_PER_BASIN.copy(deep=True)
historical["scope"] = historical.scope.replace({
    "common_source": "common_source_A2_persistence", "common_source_aux_available": "common_source_A2_persistence_aux_available",
})
A3_ALL_METRICS_PER_BASIN = pd.concat([historical, pd.DataFrame(a3_new_score_rows)], ignore_index=True)
if A3_ALL_METRICS_PER_BASIN.duplicated(["run", "scope", "basin_id"]).any():
    raise ValueError("Trùng hàng run/scope/ID trong bảng gộp")
a3_all_summary_rows, a3_all_median_rows, a3_all_status_rows, a3_all_skill_rows, a3_all_coverage_rows = [], [], [], [], []
for (run, scope), group in A3_ALL_METRICS_PER_BASIN.groupby(["run", "scope"], sort=False):
    coverage = {"run": run, "scope": scope, "valid_day_IDs": int(group.n.sum()),
                "IDs_with_days": int(group.n.gt(0).sum()), "IDs_without_days": int(group.n.eq(0).sum())}
    a3_all_coverage_rows.append(coverage)
    medians = dict(coverage)
    for metric in A3_METRIC_NAMES:
        finite = group.loc[np.isfinite(group[metric]), metric]
        median = float(finite.median()) if len(finite) else np.nan
        mean = float(finite.mean()) if len(finite) else np.nan
        a3_all_summary_rows.append({"run": run, "scope": scope, "metric": metric,
            "finite_IDs": len(finite), "undefined_IDs": len(group) - len(finite), "median": median, "mean": mean,
            "std_across_basins_ddof1": float(finite.std(ddof=1)) if len(finite) >= 2 else np.nan,
            **{name: float(finite.quantile(probability)) if len(finite) else np.nan
               for name, probability in (("p10", .1), ("p25", .25), ("p75", .75), ("p90", .9))}})
        medians[metric + "_median"], medians[metric + "_finite_IDs"] = median, len(finite)
        if metric == "NSE":
            medians["NSE_mean"] = mean
        for status, count in group[metric + "_status"].value_counts().items():
            a3_all_status_rows.append({"run": run, "scope": scope, "metric": metric, "status": status, "IDs": int(count)})
    a3_all_median_rows.append(medians)
    pnse = group.loc[np.isfinite(group.PNSE), "PNSE"]
    if len(pnse):
        a3_all_skill_rows.append({"run": run, "scope": scope, "finite_PNSE_IDs": len(pnse),
            "better": int(pnse.gt(0).sum()), "equal": int(pnse.eq(0).sum()), "worse": int(pnse.lt(0).sum()),
            "undefined": len(group) - len(pnse), "fraction_better": float(pnse.gt(0).mean())})
A3_ALL_SUMMARY, A3_ALL_MEDIAN_TABLE = pd.DataFrame(a3_all_summary_rows), pd.DataFrame(a3_all_median_rows)
A3_ALL_METRIC_STATUS, A3_ALL_PERSISTENCE_SKILL = pd.DataFrame(a3_all_status_rows), pd.DataFrame(a3_all_skill_rows)
A3_FINAL_COVERAGE = pd.DataFrame(a3_all_coverage_rows)

# %% Phần 10.5 — Thống kê giữa ba seed
# Tính mean và std mẫu giữa thống kê từng seed, tách khỏi ensemble dự báo.
a3_seed_stat_rows, a3_seed_spread_rows = [], []
seed_runs = {seed: f"DLinear_seed_{seed}" for seed in A3_DLINEAR_CONFIG["seeds"]}
for scope in ("native_source", "paired_source", A3_FINAL_COMMON_SCOPE):
    for metric in A3_METRIC_NAMES:
        values = pd.DataFrame({seed: A3_ALL_METRICS_PER_BASIN.loc[
            A3_ALL_METRICS_PER_BASIN.run.eq(run) & A3_ALL_METRICS_PER_BASIN.scope.eq(scope)
        ].set_index("basin_id")[metric].reindex(A3_SELECTED_IDS) for seed, run in seed_runs.items()})
        shared_ids = np.isfinite(values).all(axis=1)
        for cohort in ("as_defined_per_seed", "matched_finite_IDs_across_seeds"):
            cohort_values = values if cohort == "as_defined_per_seed" else values.loc[shared_ids]
            for statistic in ("median", "mean"):
                per_seed = []
                row = {"scope": scope, "metric": metric, "cohort": cohort, "statistic": statistic,
                       "matched_finite_IDs": int(shared_ids.sum())}
                for seed in seed_runs:
                    finite = cohort_values.loc[np.isfinite(cohort_values[seed]), seed]
                    value = float(getattr(finite, statistic)()) if len(finite) else np.nan
                    row[f"seed_{seed}"] = value
                    row[f"finite_IDs_{seed}"] = len(finite)
                    per_seed.append(value)
                all_three = bool(np.isfinite(per_seed).all())
                row.update(finite_seed_statistics=int(np.isfinite(per_seed).sum()),
                    seed_mean=float(np.mean(per_seed)) if all_three else np.nan,
                    seed_std_ddof1=float(np.std(per_seed, ddof=1)) if all_three else np.nan)
                a3_seed_stat_rows.append(row)
        if scope == A3_FINAL_COMMON_SCOPE:
            for basin_id, series in values.iterrows():
                finite = np.isfinite(series.to_numpy())
                a3_seed_spread_rows.append({"basin_id": basin_id, "metric": metric,
                    **{f"seed_{seed}": series[seed] for seed in seed_runs}, "finite_seeds": int(finite.sum()),
                    "mean_three_seeds": float(series.mean()) if finite.all() else np.nan,
                    "std_three_seeds_ddof1": float(series.std(ddof=1)) if finite.all() else np.nan})
A3_DLINEAR_SEED_STATISTICS = pd.DataFrame(a3_seed_stat_rows)
A3_DLINEAR_SEED_SPREAD_PER_BASIN = pd.DataFrame(a3_seed_spread_rows)

# %% Phần 10.6 — Kiểm nhất quán và lưu bảng so sánh
# Kiểm số ngày từng ID, PNSE persistence bằng 0 và lưu kết quả chung.
common_rows = A3_ALL_METRICS_PER_BASIN.loc[A3_ALL_METRICS_PER_BASIN.scope.eq(A3_FINAL_COMMON_SCOPE)]
for run in A3_FINAL_MAIN_RUNS:
    counts = common_rows.loc[common_rows.run.eq(run)].set_index("basin_id").n.reindex(A3_SELECTED_IDS)
    if not np.array_equal(counts.to_numpy(), A3_ALL_COMMON_MASK.sum(axis=0).to_numpy()):
        raise AssertionError("Run trong giao chung không dùng cùng số ngày từng ID")
baseline_pnse = common_rows.loc[common_rows.run.eq("Persistence_qmax"), "PNSE"].dropna()
if not baseline_pnse.eq(0).all():
    raise AssertionError("PNSE của qmax persistence không bằng 0 khi xác định")

a3_final_paths = []
for filename, frame in (
    ("A3_DLinear_export_audit.csv", A3_DLINEAR_EXPORT_AUDIT), ("A3_final_coverage.csv", A3_FINAL_COVERAGE),
    ("A3_all_metrics_per_basin.csv", A3_ALL_METRICS_PER_BASIN), ("A3_all_summary.csv", A3_ALL_SUMMARY),
    ("A3_all_median_table.csv", A3_ALL_MEDIAN_TABLE), ("A3_all_metric_status.csv", A3_ALL_METRIC_STATUS),
    ("A3_all_persistence_skill.csv", A3_ALL_PERSISTENCE_SKILL),
    ("A3_DLinear_seed_statistics.csv", A3_DLINEAR_SEED_STATISTICS),
    ("A3_DLinear_seed_spread_per_basin.csv", A3_DLINEAR_SEED_SPREAD_PER_BASIN),
):
    path = A3_OUT_DIR / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    a3_final_paths.append(path)
mask_path = A3_OUT_DIR / "A3_final_evaluation_masks.npz"
np.savez_compressed(mask_path, dates=A3_TEST_DATES.to_numpy(dtype="datetime64[D]"),
    basin_ids=np.asarray(A3_SELECTED_IDS, dtype="U"), common_source_all=A3_ALL_COMMON_MASK.to_numpy(),
    common_source_A2_persistence=A3_COMMON_MASK.to_numpy(), **{
        f"{run}_{scope}": mask.to_numpy() for run, masks in A3_DLINEAR_SCORE_MASKS.items() for scope, mask in masks.items()
    })
a3_final_paths.append(mask_path)
protocol_path = A3_OUT_DIR / "A3_final_evaluation_protocol.json"
protocol_path.write_text(json.dumps({
    "config_sha256": A3_DLINEAR_CONFIG_SHA256, "metric_dtype": "float64", "test_period": A3_PERIODS["test"],
    "common_scope": A3_FINAL_COMMON_SCOPE, "main_runs": A3_FINAL_MAIN_RUNS,
    "obs_common": "canonical qmax mm/day from raw source and area_gov",
    "persistence_for_PNSE": "qmax observed at calendar t-1 on exactly the same mask",
    "aux_qmean_full_common": A3_AUX_FULL_FINAL_COMMON,
    "old_common_scope_preserved_as": "common_source_A2_persistence",
    "Part6_files_modified": False, "seed_std_ddof": 1, "seed_summary_is_ensemble": False,
    "seed_cohorts": ["as_defined_per_seed", "matched_finite_IDs_across_seeds"],
}, ensure_ascii=False, indent=2), encoding="utf-8")
a3_final_paths.append(protocol_path)
pd.DataFrame([{"file": path.name, "bytes": path.stat().st_size, "sha256": a3_hash_file(path)} for path in a3_final_paths]).to_csv(
    A3_OUT_DIR / "A3_final_evaluation_manifest.csv", index=False, encoding="utf-8-sig",
)
A3_FINAL_COMPARISON_OK = True

# %% Phần 10.7 — Bảng kết quả và kỹ năng hơn persistence
# Trình bày native/paired/common, biến động giữa seed và lý do NaN.
with pd.option_context("display.max_columns", None):
    show_table('Giao ngày chung thực tế', [{
        "old_A2_persistence_day_IDs": int(A3_COMMON_MASK.to_numpy().sum()),
        "all_models_day_IDs": int(A3_ALL_COMMON_MASK.to_numpy().sum()),
        "days_lost": int((A3_COMMON_MASK & ~A3_ALL_COMMON_MASK).to_numpy().sum()),
        "IDs_with_days": int(A3_ALL_COMMON_MASK.any(axis=0).sum()),
        "IDs_without_days": ", ".join(A3_ALL_COMMON_MASK.columns[~A3_ALL_COMMON_MASK.any(axis=0)]),
        "aux_qmean_full_common": A3_AUX_FULL_FINAL_COMMON,
    }])
    show_table('DLinear theo phạm vi riêng — chưa ghép ngày với A2', A3_ALL_MEDIAN_TABLE.loc[
        A3_ALL_MEDIAN_TABLE.run.isin(seed_runs.values()) & A3_ALL_MEDIAN_TABLE.scope.isin(["native_source", "paired_source"]),
        ["run", "scope", "valid_day_IDs", "IDs_with_days", "NSE_median", "NSE_mean", "PNSE_median", "KGE_median", "KGE_finite_IDs"],
    ].to_dict("records"))
    show_table('Cùng obs và giao ngày của A2, DLinear, persistence', A3_ALL_MEDIAN_TABLE.loc[
        A3_ALL_MEDIAN_TABLE.scope.eq(A3_FINAL_COMMON_SCOPE),
        ["run", "valid_day_IDs", "IDs_with_days", "NSE_median", "NSE_mean", "PNSE_median", "KGE_median", "KGE_finite_IDs", "r_median", "alpha_median", "beta_median"],
    ].to_dict("records"))
    show_table('Mean ± std giữa ba median seed — cùng ID hữu hạn cho từng chỉ số', A3_DLINEAR_SEED_STATISTICS.loc[
        A3_DLINEAR_SEED_STATISTICS.scope.eq(A3_FINAL_COMMON_SCOPE)
        & A3_DLINEAR_SEED_STATISTICS.cohort.eq("matched_finite_IDs_across_seeds")
        & A3_DLINEAR_SEED_STATISTICS.statistic.eq("median"),
        ["metric", "matched_finite_IDs", "seed_111", "seed_222", "seed_333", "seed_mean", "seed_std_ddof1"],
    ].to_dict("records"))
    show_table('Số ID hơn/ngang/kém qmax persistence trên giao chung', A3_ALL_PERSISTENCE_SKILL.loc[
        A3_ALL_PERSISTENCE_SKILL.scope.eq(A3_FINAL_COMMON_SCOPE)
    ].to_dict("records"))
    show_table('Chỉ số không xác định trên giao chung', A3_ALL_METRIC_STATUS.loc[
        A3_ALL_METRIC_STATUS.scope.eq(A3_FINAL_COMMON_SCOPE) & ~A3_ALL_METRIC_STATUS.status.eq("ok")
    ].to_dict("records"))

print("A3_FINAL_COMPARISON_OK =", A3_FINAL_COMPARISON_OK)
print("Đã lưu 9 CSV, 1 NPZ, 1 JSON và 1 CSV manifest; giữ nguyên bảng/CSV Phần 6.")
print("Mean/std giữa seed không phải ensemble dự báo; median các thành phần không dựng lại median KGE.")
print(f"Thời gian kiểm/chấm: {time.perf_counter() - a3_final_started:.1f} giây | Output: {A3_OUT_DIR}")

# %% Phần 11.1 — Đọc lại hồ sơ A1 đã cố định nguồn
# Đối chiếu hash và 24 hồ sơ tác giả; giữ riêng nguồn A1 với A2 chạy lại.
A3_NEGATIVE_DIAGNOSTICS_OK = False
if not globals().get("A3_FINAL_COMPARISON_OK", False):
    raise RuntimeError("Cần Phần 10 đạt trước khi chẩn đoán NSE âm")
a3_negative_started = time.perf_counter()

print("NSE<0 và NSE<−1 chỉ là cờ chẩn đoán; kết quả chính vẫn giữ đủ danh sách ID.")
print("A1_P9 là chuỗi tác giả đóng gói, khác nguồn với A2 chạy lại và DLinear của A3.")

# Đọc đúng CSV ưu tiên trong manifest Phần 2, không tìm bản thay thế nếu hash/schema sai.
a3_a1_profile_tables, a3_a1_profile_audit = {}, []
for filename, required in {
    "A1_P9_author_profile_scores.csv": ["source", "run", "basin_id", "status", "paired_days",
        "obs_std_mm_day", "nse_recomputed_float64"],
    "A1_P9_author_profile_variation.csv": ["source", "run", "basin_id", "raw_constant",
        "evaluated_constant", "raw_distinct_values", "evaluated_distinct_values"],
}.items():
    records = A3_INPUT_MANIFEST.loc[A3_INPUT_MANIFEST.source.eq("A1_saved_output")
        & A3_INPUT_MANIFEST.file.eq(filename)]
    if len(records) != 1:
        raise ValueError(f"{filename}: không có đúng một bản đã cố định ở Phần 2")
    path = Path(records.iloc[0]["path"])
    if a3_hash_file(path) != records.iloc[0].sha256:
        raise ValueError(f"{filename}: hash thay đổi so với Phần 2")
    frame = pd.read_csv(path, dtype={"source": str, "run": str, "basin_id": str})
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{filename}: thiếu cột {sorted(set(required) - set(frame.columns))}")
    if not frame.source.eq("author_packaged").all() or not frame.basin_id.str.fullmatch(r"\d+").all():
        raise ValueError(f"{filename}: nguồn/ID không đúng hợp đồng A1")
    frame["basin_id"] = frame.basin_id.map(lambda value: str(int(value)))
    expected_keys = {(run, basin_id) for run in EXPECTED_RUNS for basin_id in ("55", "742", "758", "800")}
    if frame.duplicated(["run", "basin_id"]).any() or set(zip(frame.run, frame.basin_id)) != expected_keys:
        raise ValueError(f"{filename}: cần đúng 24 hồ sơ A1, không tự bỏ hoặc nhân bản dòng")
    for column in ("raw_constant", "evaluated_constant"):
        if column in frame:
            parsed = frame[column].astype(str).map({"True": True, "False": False})
            if parsed.isna().any():
                raise ValueError(f"{filename}: cờ {column} không phải True/False")
            frame[column] = parsed.astype(bool)
    a3_a1_profile_tables[filename] = frame[required].copy()
    a3_a1_profile_audit.append({"file": filename, "rows": len(frame), "hash_matches_part2": True,
                              "source": "author_packaged", "sha256": records.iloc[0].sha256})
a1_scores = a3_a1_profile_tables["A1_P9_author_profile_scores.csv"]
if not a1_scores.status.eq("compared").all():
    raise ValueError("Hồ sơ A1 chưa được đối chiếu đầy đủ; không dùng dòng lỗi làm bằng chứng")
for column in ("paired_days", "obs_std_mm_day", "nse_recomputed_float64"):
    a1_scores[column] = pd.to_numeric(a1_scores[column], errors="raise")
a1_context = a1_scores.merge(a3_a1_profile_tables["A1_P9_author_profile_variation.csv"],
    on=["source", "run", "basin_id"], validate="one_to_one")
a1_context = a1_context.rename(columns={name: "a1_author_" + name
    for name in a1_context.columns if name not in ("run", "basin_id")})
A3_A1_PROFILE_SOURCE_AUDIT = pd.DataFrame(a3_a1_profile_audit)
show_table('Nguồn A1 đọc lại — không thay bằng kết quả A2', a3_a1_profile_audit)

# %% Phần 11.2 — Độ phủ train/test của các hồ sơ cố định
# Phân biệt số nhãn train, mẫu DLinear và số ngày trên giao chung.
A3_A1_PROFILE_NOTES = {
    "55": "A1: missing Q history; author Q-input runs without predictions",
    "295": "A3: no complete training samples; retained for coverage audit",
    "324": "A1: no training labels; evaluation labels available",
    "600": "A1: no training labels; evaluation labels available",
    "623": "A1: no training labels; evaluation labels available",
    "742": "A1: small test observation variance; author EncDec_noQ constant",
    "758": "A1: small test observation variance; author EncDec_noQ constant",
    "800": "A1: only 38 training labels",
}
A3_DIAGNOSTIC_PROFILE_IDS = [basin_id for basin_id in A3_A1_PROFILE_NOTES if basin_id in A3_SELECTED_IDS]
train_calendar = pd.date_range(*A3_PERIODS["train"], freq="D")
train_qmax = A3_DLINEAR_Q_MM_DAY["qmax"].loc[train_calendar, A3_SELECTED_IDS]
train_label_counts = np.isfinite(train_qmax).sum(axis=0)
train_sample_counts = A3_DLINEAR_SAMPLE_MASKS["train"].sum(axis=0)
A3_PROFILE_TRAIN_COVERAGE = pd.DataFrame([{
    "basin_id": basin_id, "profile_note": A3_A1_PROFILE_NOTES[basin_id],
    "train_label_days": int(train_label_counts[basin_id]),
    "DLinear_train_samples": int(train_sample_counts[basin_id]),
    "train_qmax_std_mm_day": A3_DLINEAR_SIGMA_TRAIN[basin_id],
    "test_obs_days": int(np.isfinite(A3_SOURCE_OBS[basin_id]).sum()),
    "test_obs_std_mm_day": float(A3_SOURCE_OBS[basin_id].std(ddof=0)),
    "common_days": int(A3_ALL_COMMON_MASK[basin_id].sum()),
} for basin_id in A3_DIAGNOSTIC_PROFILE_IDS])

# %% Phần 11.3 — Hàm phân rã sai số và ảnh hưởng clip
# Giữ đúng mask; kiểm MSE=bias²+var(error), kể cả sim hằng.
def a3_diagnostic_row(run, scope, basin_id, obs, raw, sim, mask, persistence=None):
    """Phân rã trên đúng mask; MSE=bias²+var(error) dùng được cả với sim hằng."""
    observed, raw_values, scored = [np.asarray(values, dtype=np.float64) for values in (obs, raw, sim)]
    chosen = np.asarray(mask)
    if chosen.dtype != np.bool_ or chosen.shape != observed.shape or raw_values.shape != observed.shape:
        raise ValueError("Trục/mặt nạ hồ sơ không hợp lệ")
    if not np.isfinite(raw_values[chosen]).all():
        raise ValueError("Raw không hữu hạn trên ngày đã chọn; không đổi mask")
    result = a3_metrics(observed, scored, persistence, chosen)
    raw_result = a3_metrics(observed, raw_values, mask=chosen)
    o, p = observed[chosen], scored[chosen]
    row = {"run": run, "scope": scope, "basin_id": basin_id, **result,
        "prediction_source": "A2_rerun_export" if run in EXPECTED_RUNS else
            ("A3_DLinear" if run in A3_DLINEAR_DAILY else "calendar_persistence"),
        "observation_source": "A2_run_obs" if scope == "native" else "source_Q_area_gov",
        "profile_note": A3_A1_PROFILE_NOTES.get(basin_id, ""),
        "train_label_days": int(train_label_counts[basin_id]),
        "DLinear_train_samples": int(train_sample_counts[basin_id]),
        "NSE_raw_same_mask": raw_result["NSE"], "NSE_gain_from_declared_clip": result["NSE"] - raw_result["NSE"],
        "raw_negative_all_days": int((raw_values < 0).sum()),
        "raw_negative_scored_days": int((raw_values[chosen] < 0).sum()),
        "scored_zero_days": int((p == 0).sum()), "scored_distinct_values": len(np.unique(p)),
        "scored_constant": bool(len(p) and np.ptp(p) == 0),
        "obs_min": float(o.min()) if len(o) else np.nan, "obs_max": float(o.max()) if len(o) else np.nan,
        "sim_min": float(p.min()) if len(p) else np.nan, "sim_max": float(p.max()) if len(p) else np.nan,
        "mean_error": np.nan, "MSE": np.nan, "bias2_over_obs_variance": np.nan,
        "error_variance_over_obs_variance": np.nan, "correlation_decomposition_sum": np.nan,
        "MSE_decomposition_ok": None}
    if len(o):
        error = p - o
        bias, error_variance = float(error.mean()), float(np.var(error, ddof=0))
        mse = float(np.mean(error ** 2))
        row.update(mean_error=bias, MSE=mse,
            MSE_decomposition_ok=bool(np.isclose(mse, bias ** 2 + error_variance, rtol=1e-10, atol=1e-12)))
        if not row["MSE_decomposition_ok"]:
            raise AssertionError("MSE khác bias² + var(error)")
        variance = result["obs_tss"] / len(o)
        if variance > 0:
            row.update(bias2_over_obs_variance=bias ** 2 / variance,
                       error_variance_over_obs_variance=error_variance / variance)
            total = row["bias2_over_obs_variance"] + row["error_variance_over_obs_variance"]
            if not np.isclose(total, 1 - result["NSE"], rtol=1e-10, atol=1e-10):
                raise AssertionError("Hai thành phần chuẩn hóa không cộng thành 1−NSE")
            if result["r_status"] == "ok":
                row["correlation_decomposition_sum"] = (result["alpha"] - 1) ** 2 + 2 * result["alpha"] * (1 - result["r"]) + bias ** 2 / variance
                if not np.isclose(row["correlation_decomposition_sum"], total, rtol=1e-10, atol=1e-10):
                    raise AssertionError("Phân rã theo r/alpha không khớp")
    return row

# %% Phần 11.4 — NSE âm và đối chiếu dự báo tác giả
# Giữ toàn bộ ID âm; kiểm lại điểm và các trường hợp sim hằng.
diagnostic_rows = []
common_reference = A3_ALL_METRICS_PER_BASIN.loc[
    A3_ALL_METRICS_PER_BASIN.scope.isin([A3_FINAL_COMMON_SCOPE, A3_FINAL_COMMON_SCOPE + "_aux_available"])
].set_index(["run", "scope", "basin_id"], verify_integrity=True)
for run, sim in A3_FINAL_SCORED_SIM.items():
    raw = A3_DAILY[run]["sim_raw"] if run in EXPECTED_RUNS else A3_DLINEAR_DAILY[run]["sim_raw"] if run in A3_DLINEAR_DAILY else sim
    partial_aux = run == "Persistence_qmean_aux" and not A3_AUX_FULL_FINAL_COMMON
    common_scope = A3_FINAL_COMMON_SCOPE + "_aux_available" if partial_aux else A3_FINAL_COMMON_SCOPE
    common_mask = A3_ALL_COMMON_MASK & np.isfinite(sim) if partial_aux else A3_ALL_COMMON_MASK
    for basin_id in A3_SELECTED_IDS:
        row = a3_diagnostic_row(run, common_scope, basin_id, A3_SOURCE_OBS[basin_id], raw[basin_id],
            sim[basin_id], common_mask[basin_id], A3_PERSISTENCE_QMAX[basin_id])
        reference = common_reference.loc[(run, common_scope, basin_id)]
        if row["n"] != reference.n or any(row[name + "_status"] != reference[name + "_status"] or
            not np.isclose(row[name], reference[name], rtol=1e-12, atol=1e-12, equal_nan=True)
            for name in A3_METRIC_NAMES):
            raise AssertionError(f"{run}/{basin_id}: chẩn đoán không tái lập kết quả Phần 10")
        diagnostic_rows.append(row)
    # Native ở các hồ sơ cố định giúp thấy 55 còn dữ liệu ở run không Q; không cấp PNSE nếu lag thiếu.
    native_obs = A3_DAILY[run]["obs"] if run in EXPECTED_RUNS else A3_SOURCE_OBS
    native_mask = np.isfinite(native_obs) & np.isfinite(sim)
    native_scope = "native" if run in EXPECTED_RUNS else "native_source"
    for basin_id in A3_DIAGNOSTIC_PROFILE_IDS:
        diagnostic_rows.append(a3_diagnostic_row(run, native_scope, basin_id, native_obs[basin_id],
            raw[basin_id], sim[basin_id], native_mask[basin_id]))
A3_ERROR_DIAGNOSTICS = pd.DataFrame(diagnostic_rows).merge(a1_context,
    on=["run", "basin_id"], how="left", validate="many_to_one")
diagnostic_common = A3_ERROR_DIAGNOSTICS.loc[A3_ERROR_DIAGNOSTICS.scope.str.startswith(A3_FINAL_COMMON_SCOPE)].copy()
A3_NEGATIVE_NSE_ALL = diagnostic_common.loc[diagnostic_common.NSE < 0].copy()
A3_NEGATIVE_NSE_ALL["NSE_below_minus_one"] = A3_NEGATIVE_NSE_ALL.NSE < -1
A3_NEGATIVE_NSE_TOP10 = (A3_NEGATIVE_NSE_ALL.sort_values(["run", "NSE", "basin_id"], kind="stable")
    .groupby("run", sort=False).head(10).reset_index(drop=True))
A3_NEGATIVE_NSE_SUMMARY = pd.DataFrame([{
    "run": run, "scope": scope, "finite_NSE_IDs": int(np.isfinite(part.NSE).sum()),
    "undefined_NSE_IDs": int(part.NSE.isna().sum()), "NSE_lt_0_IDs": int((part.NSE < 0).sum()),
    "NSE_lt_minus1_IDs": int((part.NSE < -1).sum()), "minimum_NSE": part.NSE.min(),
    "negative_fixed_profile_IDs": ", ".join(sorted(set(part.loc[part.NSE.lt(0), "basin_id"]) & set(A3_DIAGNOSTIC_PROFILE_IDS), key=int)),
} for (run, scope), part in diagnostic_common.groupby(["run", "scope"], sort=False)])
A3_A1_PROFILE_COMPARISON = A3_ERROR_DIAGNOSTICS.loc[
    A3_ERROR_DIAGNOSTICS.run.isin(EXPECTED_RUNS) & A3_ERROR_DIAGNOSTICS.scope.eq("native")
    & A3_ERROR_DIAGNOSTICS.basin_id.isin(["55", "742", "758", "800"])
].copy()

# %% Phần 11.5 — Độ nhạy và lý do chỉ số không xác định
# Bảng bỏ hồ sơ chỉ là chẩn đoán bổ sung; kết quả chính giữ đủ danh sách ID.
sensitivity_rows = []
for (run, scope), part in diagnostic_common.groupby(["run", "scope"], sort=False):
    for cohort, excluded in (("all_selected", set()), ("without_742_758_diagnostic", {"742", "758"}),
                             ("without_800_diagnostic", {"800"})):
        retained = part.loc[~part.basin_id.isin(excluded)]
        for metric in A3_METRIC_NAMES:
            values = retained.loc[np.isfinite(retained[metric]), metric]
            sensitivity_rows.append({"run": run, "scope": scope, "cohort": cohort, "metric": metric,
                "selected_IDs": len(retained), "finite_IDs": len(values), "undefined_IDs": len(retained) - len(values),
                "mean": float(values.mean()) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan})
A3_DIAGNOSTIC_SENSITIVITY = pd.DataFrame(sensitivity_rows)
status_parts = []
for metric in A3_METRIC_NAMES:
    undefined = diagnostic_common.loc[~diagnostic_common[metric + "_status"].eq("ok")]
    for status, part in undefined.groupby(metric + "_status", sort=False):
        status_parts.append({"metric": metric, "status": status, "run_ID_pairs": len(part),
            "runs": ", ".join(dict.fromkeys(part.run)), "basin_IDs": ", ".join(sorted(set(part.basin_id), key=int))})
A3_DIAGNOSTIC_UNDEFINED = pd.DataFrame(status_parts)

# %% Phần 11.6 — Lưu và đọc các bảng chẩn đoán
# Lưu kết quả, hồ sơ A1 và trạng thái; không thay bảng so sánh chính.
diagnostic_paths = []
for filename, frame in (
    ("A3_A1_profile_source_audit.csv", A3_A1_PROFILE_SOURCE_AUDIT),
    ("A3_profile_train_coverage.csv", A3_PROFILE_TRAIN_COVERAGE), ("A3_error_diagnostics.csv", A3_ERROR_DIAGNOSTICS),
    ("A3_negative_NSE_all.csv", A3_NEGATIVE_NSE_ALL), ("A3_negative_NSE_top10.csv", A3_NEGATIVE_NSE_TOP10),
    ("A3_negative_NSE_summary.csv", A3_NEGATIVE_NSE_SUMMARY), ("A3_A1_profile_comparison.csv", A3_A1_PROFILE_COMPARISON),
    ("A3_diagnostic_sensitivity.csv", A3_DIAGNOSTIC_SENSITIVITY), ("A3_diagnostic_undefined.csv", A3_DIAGNOSTIC_UNDEFINED),
):
    path = A3_OUT_DIR / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    diagnostic_paths.append(path)
pd.DataFrame([{"file": path.name, "bytes": path.stat().st_size, "sha256": a3_hash_file(path)} for path in diagnostic_paths]).to_csv(
    A3_OUT_DIR / "A3_diagnostic_manifest.csv", index=False, encoding="utf-8-sig")
A3_NEGATIVE_DIAGNOSTICS_OK = True

# Hiển thị trực tiếp để cả các bảng dài không bị show_table giới hạn 40 dòng.
with pd.option_context("display.max_columns", None, "display.max_rows", None, "display.max_colwidth", None):
    for title, frame in (
        ('Cờ độ phủ và biến thiên ở các hồ sơ cố định', A3_PROFILE_TRAIN_COVERAGE),
        ('Mọi NSE âm vẫn được giữ trong kết quả chính', A3_NEGATIVE_NSE_SUMMARY),
        ('Tối đa ba NSE âm thấp nhất mỗi run — CSV giữ tối đa mười và danh sách âm đầy đủ',
            A3_NEGATIVE_NSE_TOP10.groupby("run", sort=False).head(3)[["run", "basin_id", "n", "NSE", "PNSE", "KGE", "std_obs", "mean_error", "profile_note"]]),
        ('Hai hồ sơ 742/758 trên giao chung — sim hằng không thay r bằng 0',
            diagnostic_common.loc[diagnostic_common.basin_id.isin(["742", "758"]), ["run", "basin_id", "n", "NSE", "KGE", "r", "alpha", "beta", "std_obs", "std_sim", "bias2_over_obs_variance", "error_variance_over_obs_variance", "scored_distinct_values", "NSE_gain_from_declared_clip"]]),
        ('A2 native cạnh hồ sơ tác giả A1 — hai nguồn dự báo riêng',
            A3_A1_PROFILE_COMPARISON[["run", "basin_id", "n", "NSE", "scored_constant", "a1_author_paired_days", "a1_author_nse_recomputed_float64", "a1_author_evaluated_constant"]]),
        ('Độ nhạy NSE khi bỏ hồ sơ cố định — chỉ bổ sung',
            A3_DIAGNOSTIC_SENSITIVITY.loc[A3_DIAGNOSTIC_SENSITIVITY.metric.eq("NSE"), ["run", "scope", "cohort", "selected_IDs", "finite_IDs", "mean", "median"]]),
        ('Toàn bộ nguyên nhân không xác định trên giao chung, gộp gọn từ 10.6', A3_DIAGNOSTIC_UNDEFINED),
    ):
        display(Markdown("**" + title + "**"))
        display(frame.reset_index(drop=True))

print("A3_NEGATIVE_DIAGNOSTICS_OK =", A3_NEGATIVE_DIAGNOSTICS_OK)
print("Đã lưu 9 CSV và 1 manifest; không ghi đè bảng Phần 6/10 hoặc gói seed.")
print("Phân rã sai số là đẳng thức mô tả, không xác nhận cơ chế vật lý hoặc lỗi quan trắc.")
print(f"Thời gian chẩn đoán: {time.perf_counter() - a3_negative_started:.1f} giây | Output: {A3_OUT_DIR}")

# %% Phần 11.7 — Chuẩn bị dữ liệu và cách hiển thị hình
# Kiểm nguồn bảng, cố định màu/nhãn và các hàm lưu hình/dữ liệu dùng vẽ.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from IPython.display import Image as DisplayImage

A3_FIGURES_OK = False
if not globals().get("A3_NEGATIVE_DIAGNOSTICS_OK", False) or not globals().get("A3_FINAL_COMPARISON_OK", False):
    raise RuntimeError("Cần Phần 10 và Phần 11.1 đạt trước khi vẽ")
a3_figures_started = time.perf_counter()

for filename in ("A3_final_evaluation_manifest.csv", "A3_diagnostic_manifest.csv"):
    for record in pd.read_csv(A3_OUT_DIR / filename).itertuples(index=False):
        if a3_hash_file(A3_OUT_DIR / record.file) != record.sha256:
            raise ValueError(f"Nguồn vẽ đã thay đổi: {record.file}")
A3_FIGURE_DIR = A3_OUT_DIR / "Figures"
A3_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
plot_scores = A3_ALL_METRICS_PER_BASIN.loc[A3_ALL_METRICS_PER_BASIN.scope.eq(A3_FINAL_COMMON_SCOPE)].copy()
plot_runs = [*A3_FINAL_MAIN_RUNS]
if A3_AUX_FULL_FINAL_COMMON:
    plot_runs.append("Persistence_qmean_aux")
plot_scores = plot_scores.loc[plot_scores.run.isin(plot_runs)]
score_index = plot_scores.set_index(["run", "basin_id"], verify_integrity=True)
for run in plot_runs:
    if set(plot_scores.loc[plot_scores.run.eq(run), "basin_id"]) != set(A3_SELECTED_IDS):
        raise ValueError(f"{run}: bảng vẽ không đủ danh sách ID")
    counts = score_index.loc[run].n.reindex(A3_SELECTED_IDS)
    if not np.array_equal(counts.to_numpy(), A3_ALL_COMMON_MASK.sum(axis=0).to_numpy()):
        raise ValueError(f"{run}: số ngày không khớp giao chung")

plot_style = {
    "SeqLSTM_Q": ("#0072B2", "-"), "SeqLSTM_noQ": ("#56B4E9", "--"),
    "EncDec_Q": ("#009E73", "-"), "EncDec_noQ": ("#E69F00", "--"),
    "D_FC": ("#9467BD", "-"), "D_RA": ("#CC79A7", "-"),
    "DLinear_seed_111": ("#D55E00", "-"), "DLinear_seed_222": ("#D55E00", "--"),
    "DLinear_seed_333": ("#D55E00", ":"), "Persistence_qmax": ("#202020", "--"),
    "Persistence_qmean_aux": ("#777777", ":"),
}
plot_labels = {run: run.replace("DLinear_seed_", "DLinear ").replace("Persistence_qmax", "Persistence qmax")
    .replace("Persistence_qmean_aux", "Persistence qmean (aux)") for run in plot_runs}
legend_handles = [Line2D([], [], color=plot_style[run][0], linestyle=plot_style[run][1],
    label=plot_labels[run], linewidth=1.7) for run in plot_runs]
figure_records, figure_data_paths = [], []


def a3_save_figure(fig, filename, purpose):
    """Lưu trước, đóng figure rồi hiển thị chính PNG đã lưu; không để nhiều canvas chiếm RAM."""
    path = A3_FIGURE_DIR / filename
    try:
        fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    finally:
        plt.close(fig)
    figure_records.append({"file": str(path.relative_to(A3_OUT_DIR)), "purpose": purpose,
        "bytes": path.stat().st_size, "sha256": a3_hash_file(path)})
    display(Markdown("**" + filename + "**"))
    display(DisplayImage(filename=str(path)))


def a3_save_figure_data(frame, filename):
    path = A3_OUT_DIR / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    figure_data_paths.append(path)

# %% Phần 11.8 — ECDF của sáu chỉ số
# Giữ toàn bộ điểm âm; mẫu số là số ID hữu hạn riêng cho mỗi run/chỉ số.
with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.formatter.use_locale": False}):
    ecdf_rows, count_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, metric in zip(axes.flat, A3_METRIC_NAMES):
        for run in plot_runs:
            part = plot_scores.loc[plot_scores.run.eq(run), ["basin_id", metric]].dropna(subset=[metric])
            part = part.sort_values(metric, kind="stable")
            values = part[metric].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("Giá trị chỉ số dùng vẽ có vô cực")
            n = len(values)
            count_rows.append({"run": run, "scope": A3_FINAL_COMMON_SCOPE, "metric": metric,
                "finite_IDs": n, "undefined_IDs": len(A3_SELECTED_IDS) - n,
                "minimum": float(values.min()) if n else np.nan, "maximum": float(values.max()) if n else np.nan})
            if not n:
                continue
            probabilities = np.arange(1, n + 1) / n
            ax.step(np.r_[values[0], values], np.r_[0., probabilities], where="post",
                color=plot_style[run][0], linestyle=plot_style[run][1], linewidth=1.3)
            ecdf_rows.extend({"run": run, "metric": metric, "basin_id": basin_id,
                "value": value, "rank": rank, "finite_IDs": n, "ecdf": probability}
                for rank, (basin_id, value, probability) in enumerate(zip(part.basin_id, values, probabilities), 1))
        if metric != "r":
            ax.set_xscale("symlog", linthresh=1.)
        else:
            ax.set_xlim(-1.02, 1.02)
        ax.axvline(0 if metric in ("NSE", "PNSE") else 1, color="#888888", linewidth=.8, alpha=.7)
        ax.set(title=metric, xlabel=metric + (" (symlog; linear within [-1, 1])" if metric != "r" else ""),
               ylabel="Fraction of finite basin scores", ylim=(0, 1.02))
        ax.grid(alpha=.2)
    fig.suptitle("Per-basin score distributions | same daily mask | all finite scores retained", y=.99)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, .96))
    display(Markdown('**Phân phối sáu chỉ số — trục symlog giữ toàn bộ đuôi âm**'))
    a3_save_figure(fig, "A3_score_ecdf.png", "Six metrics; finite cohort recorded separately per run/metric")
    A3_FIGURE_METRIC_COUNTS = pd.DataFrame(count_rows)
    a3_save_figure_data(pd.DataFrame(ecdf_rows), "A3_figure_ecdf_points.csv")
    a3_save_figure_data(A3_FIGURE_METRIC_COUNTS, "A3_figure_metric_counts.csv")

# %% Phần 11.9 — NSE mô hình ghép cặp với persistence
# So cùng ID/ngày, giữ đuôi âm và đánh dấu 742/758.
with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.formatter.use_locale": False}):
    paired_rows = []
    model_runs = [run for run in plot_runs if not run.startswith("Persistence_")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, run in zip(axes.flat, model_runs):
        model = score_index.loc[run].reindex(A3_SELECTED_IDS)
        baseline = score_index.loc["Persistence_qmax"].reindex(A3_SELECTED_IDS)
        valid = np.isfinite(model.NSE) & np.isfinite(baseline.NSE)
        x, y = baseline.loc[valid, "NSE"], model.loc[valid, "NSE"]
        paired = model.loc[valid, "PNSE"]
        finite_pnse = np.isfinite(paired)
        if not np.allclose((y - x)[finite_pnse], ((1 - x) * paired)[finite_pnse], rtol=1e-10, atol=1e-10):
            raise AssertionError("NSE model/persistence và PNSE không cùng mặt nạ")
        ax.scatter(x, y, s=13, color=plot_style[run][0], alpha=.55, edgecolors="none")
        lo = min(float(x.min()), float(y.min()), 0.) if len(x) else 0.
        hi = max(float(x.max()), float(y.max()), 1.) if len(x) else 1.
        margin = .03 * max(hi - lo, 1.)
        limits = (lo - margin, hi + margin)
        ax.set_xscale("symlog", linthresh=1.); ax.set_yscale("symlog", linthresh=1.)
        ax.plot(limits, limits, color="#777777", linewidth=.8)
        ax.set(xlim=limits, ylim=limits, title=plot_labels[run], xlabel="Persistence qmax NSE", ylabel="Model NSE")
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2)
        ax.text(.03, .97, f"finite pairs={len(x)}; undefined={len(A3_SELECTED_IDS)-len(x)}\n"
            f"PNSE > 0: {int((paired > 0).sum())}; < 0: {int((paired < 0).sum())}",
            transform=ax.transAxes, va="top", fontsize=8)
        for basin_id, offset in (("742", (5, 8)), ("758", (5, -12))):
            if basin_id in x.index:
                ax.annotate(basin_id, (x[basin_id], y[basin_id]), xytext=offset, textcoords="offset points", fontsize=8)
        paired_rows.extend({"run": run, "basin_id": basin_id, "persistence_NSE": x[basin_id],
            "model_NSE": y[basin_id], "delta_NSE": y[basin_id] - x[basin_id], "PNSE": paired[basin_id]}
            for basin_id in x.index)
    for ax in axes.flat[len(model_runs):]:
        ax.set_visible(False)
    fig.suptitle("Model vs persistence | paired basins | symlog axes, linear within [-1, 1]", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .97))
    display(Markdown('**NSE ghép cặp với persistence — giữ mọi điểm ngoài khoảng [0, 1]**'))
    a3_save_figure(fig, "A3_NSE_vs_persistence.png", "Paired NSE; full range, no excluded negative scores")
    a3_save_figure_data(pd.DataFrame(paired_rows), "A3_figure_paired_NSE.csv")

# %% Phần 11.10 — Chuỗi dự báo ở 742, 758, 363 và 800
# Hai cột A2/DLinear cùng obs, ngày và thang Q của mỗi hàng; giữ khoảng thiếu.
with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.formatter.use_locale": False}):
    A3_FIGURE_PROFILE_IDS = [basin_id for basin_id in ("742", "758", "363", "800") if basin_id in A3_SELECTED_IDS]
    if not A3_FIGURE_PROFILE_IDS:
        raise ValueError("Không có ID hồ sơ để vẽ")
    profile_rows = []
    for basin_id in A3_FIGURE_PROFILE_IDS:
        chosen = A3_ALL_COMMON_MASK[basin_id].to_numpy()
        for run in plot_runs:
            predicted = A3_FINAL_SCORED_SIM[run][basin_id].to_numpy()[chosen]
            observed = A3_SOURCE_OBS[basin_id].to_numpy()[chosen]
            if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
                raise ValueError("Chuỗi hồ sơ không hữu hạn trên giao chung")
            profile_rows.extend({"basin_id": basin_id, "run": run, "date": date, "obs_mm_day": obs,
                "sim_mm_day": sim, "error_mm_day": sim - obs}
                for date, obs, sim in zip(A3_SOURCE_OBS.index[chosen], observed, predicted))
    A3_FIGURE_PROFILE_DAILY = pd.DataFrame(profile_rows, columns=["basin_id", "run", "date", "obs_mm_day", "sim_mm_day", "error_mm_day"])
    a3_save_figure_data(A3_FIGURE_PROFILE_DAILY, "A3_figure_profile_daily.csv")
    groups = [("A2 reruns", list(EXPECTED_RUNS)), ("DLinear seeds", list(A3_DLINEAR_DAILY))]
    fig, axes = plt.subplots(len(A3_FIGURE_PROFILE_IDS), 2, figsize=(15, 3.2 * len(A3_FIGURE_PROFILE_IDS)),
                             sharex=True, sharey="row", squeeze=False)
    for row_number, basin_id in enumerate(A3_FIGURE_PROFILE_IDS):
        chosen = A3_ALL_COMMON_MASK[basin_id]
        for ax, (group_label, group_runs) in zip(axes[row_number], groups):
            ax.plot(A3_SOURCE_OBS.index, A3_SOURCE_OBS[basin_id].where(chosen), color="black", linewidth=1.2, label="Observed qmax", zorder=4)
            for run in ["Persistence_qmax", *group_runs]:
                ax.plot(A3_SOURCE_OBS.index, A3_FINAL_SCORED_SIM[run][basin_id].where(chosen),
                    color=plot_style[run][0], linestyle=plot_style[run][1], linewidth=.8, alpha=.8, label=plot_labels[run])
            ax.set(title=f"Basin {basin_id} | {group_label} | common days={int(chosen.sum())}", ylabel="qmax (mm/day)")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.grid(alpha=.2)
            if not chosen.any():
                ax.text(.5, .5, "No common days", transform=ax.transAxes, ha="center")
    for column in range(2):
        handles, labels = axes[0, column].get_legend_handles_labels()
        axes[-1, column].legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.24), ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Diagnostic hydrographs | canonical observations | same dates per basin | gaps retained", y=.995)
    fig.tight_layout(rect=(0, .08, 1, .98))
    display(Markdown('**Chuỗi 742/758/363/800 — mỗi hàng cùng thang Q và cùng ngày giữa hai nhóm**'))
    a3_save_figure(fig, "A3_profile_hydrographs.png", "A1 fixed cases and post-hoc basin 363; no interpolation across missing days")

# %% Phần 11.11 — Phân bố sai số ngày theo hồ sơ
# Boxplot sim−obs trên giao chung; giữ các điểm ngoài whisker.
with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.formatter.use_locale": False}):
    fig, axes = plt.subplots(len(A3_FIGURE_PROFILE_IDS), 1, figsize=(14, 3.2 * len(A3_FIGURE_PROFILE_IDS)), squeeze=False)
    for ax, basin_id in zip(axes.flat, A3_FIGURE_PROFILE_IDS):
        errors = [A3_FIGURE_PROFILE_DAILY.loc[A3_FIGURE_PROFILE_DAILY.basin_id.eq(basin_id)
            & A3_FIGURE_PROFILE_DAILY.run.eq(run), "error_mm_day"].to_numpy() for run in plot_runs]
        nonempty = [(position, values) for position, values in enumerate(errors, 1) if len(values)]
        if nonempty:
            boxes = ax.boxplot([values for _, values in nonempty], positions=[position for position, _ in nonempty],
                widths=.6, patch_artist=True, showfliers=True, flierprops={"markersize": 1.5, "alpha": .35})
            for box, (position, _) in zip(boxes["boxes"], nonempty):
                box.set_facecolor(plot_style[plot_runs[position-1]][0]); box.set_alpha(.55)
        ax.set_xticks(range(1, len(plot_runs)+1), [plot_labels[run] for run in plot_runs], rotation=25, ha="right")
        ax.set_yscale("symlog", linthresh=.01); ax.axhline(0, color="#777777", linewidth=.8)
        ax.set(title=f"Basin {basin_id} | days per run={len(errors[0])}", ylabel="sim - obs (mm/day)")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Paired daily errors | all outliers shown | symlog: linear within +/-0.01 mm/day", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .98))
    display(Markdown('**Phân bố sai số theo ngày — boxplot giữ điểm ngoại lệ**'))
    a3_save_figure(fig, "A3_profile_errors.png", "Daily errors on common mask; outliers retained")

# %% Phần 11.12 — Thành phần sai số chuẩn hóa
# Vẽ bias²/var(obs) và var(error)/var(obs), tổng bằng 1−NSE.
with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.formatter.use_locale": False}):
    component_data = A3_ERROR_DIAGNOSTICS.loc[A3_ERROR_DIAGNOSTICS.scope.eq(A3_FINAL_COMMON_SCOPE)
        & A3_ERROR_DIAGNOSTICS.basin_id.isin(A3_FIGURE_PROFILE_IDS) & A3_ERROR_DIAGNOSTICS.run.isin(plot_runs)].copy()
    component_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, basin_id in zip(axes.flat, A3_FIGURE_PROFILE_IDS):
        part = component_data.loc[component_data.basin_id.eq(basin_id)].set_index("run").reindex(plot_runs)
        bias = part.bias2_over_obs_variance.to_numpy(dtype=float)
        variability = part.error_variance_over_obs_variance.to_numpy(dtype=float)
        valid = np.isfinite(bias) & np.isfinite(variability)
        x = np.arange(len(plot_runs))
        ax.bar(x[valid], bias[valid], color="#D55E00", label="Squared mean error / obs variance")
        ax.bar(x[valid], variability[valid], bottom=bias[valid], color="#0072B2", label="Error variance / obs variance")
        for position in x[~valid]:
            ax.text(position, .04, "NA", ha="center", transform=ax.get_xaxis_transform(), fontsize=8)
        ax.axhline(1, color="#777777", linewidth=.8, linestyle="--")
        ax.set_xticks(x, [plot_labels[run] for run in plot_runs], rotation=45, ha="right", fontsize=8)
        ax.set_yscale("symlog", linthresh=1.)
        ax.set(title=f"Basin {basin_id} | total = 1 - NSE", ylabel="Normalized MSE (symlog)")
        ax.grid(axis="y", alpha=.2)
        component_rows.extend({"run": run, "basin_id": basin_id, "bias2_over_obs_variance": bias[i],
            "error_variance_over_obs_variance": variability[i], "NSE": part.iloc[i].NSE} for i, run in enumerate(plot_runs))
    for ax in axes.flat[len(A3_FIGURE_PROFILE_IDS):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Error decomposition | ddof=0 | descriptive identity, not causal attribution", y=.995)
    fig.tight_layout(rect=(0, .05, 1, .97))
    display(Markdown('**Phân rã 1−NSE — không cần thay r ở chuỗi hằng**'))
    a3_save_figure(fig, "A3_profile_error_components.png", "Squared mean error plus error variance, divided by observation variance")
    a3_save_figure_data(pd.DataFrame(component_rows), "A3_figure_error_components.csv")

# %% Phần 11.13 — Độ phủ native/common và số mẫu train
# Giữ hàng ID 55 và phân biệt nhãn train với cửa sổ train hoàn chỉnh.
with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.formatter.use_locale": False}):
    coverage_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    coverage_index = A3_ERROR_DIAGNOSTICS.set_index(["run", "scope", "basin_id"], verify_integrity=True)
    coverage_labels = []
    profile_info = A3_PROFILE_TRAIN_COVERAGE.set_index("basin_id")
    for basin_id in A3_DIAGNOSTIC_PROFILE_IDS:
        info = profile_info.loc[basin_id]
        coverage_labels.append(f"{basin_id} | train labels={int(info.train_label_days)}; DL samples={int(info.DLinear_train_samples)}")
    for ax, cohort in zip(axes, ("native", "common")):
        values = np.zeros((len(A3_DIAGNOSTIC_PROFILE_IDS), len(plot_runs)), dtype=int)
        for i, basin_id in enumerate(A3_DIAGNOSTIC_PROFILE_IDS):
            for j, run in enumerate(plot_runs):
                scope = A3_FINAL_COMMON_SCOPE if cohort == "common" else "native" if run in EXPECTED_RUNS else "native_source"
                values[i, j] = int(coverage_index.loc[(run, scope, basin_id), "n"])
                coverage_rows.append({"run": run, "basin_id": basin_id, "cohort": cohort, "scope": scope, "paired_days": int(values[i, j])})
        image_plot = ax.imshow(values, cmap="Blues", vmin=0, vmax=len(A3_SOURCE_OBS), aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color="white" if value > .6*len(A3_SOURCE_OBS) else "black")
        ax.set_xticks(range(len(plot_runs)), [plot_labels[run] for run in plot_runs], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(coverage_labels)), coverage_labels, fontsize=8)
        ax.set_title("Native paired days (run-specific observations)" if cohort == "native" else "Common paired days (canonical observations)")
    fig.suptitle("Coverage audit | no basin removed | training counts refer to DLinear inputs/labels", y=.98)
    fig.subplots_adjust(left=.28, right=.87, bottom=.30, top=.89, wspace=.12)
    colorbar_axis = fig.add_axes((.90, .30, .015, .59))
    fig.colorbar(image_plot, cax=colorbar_axis, label="Paired days")
    display(Markdown('**Độ phủ native/giao chung và số nhãn/mẫu train — giữ ID 55**'))
    a3_save_figure(fig, "A3_profile_coverage.png", "Native and common coverage; training labels are not complete training windows")
    a3_save_figure_data(pd.DataFrame(coverage_rows), "A3_figure_profile_coverage.csv")

# %% Phần 11.14 — Danh mục hình và tệp kết quả
# Lưu sáu PNG, dữ liệu dùng vẽ và danh mục vào a3_output để Kaggle lưu output.
A3_FIGURE_INDEX = pd.DataFrame(figure_records)
a3_save_figure_data(A3_FIGURE_INDEX, "A3_figure_index.csv")
pd.DataFrame([*figure_records, *[{"file": path.name, "purpose": "Figure data/index", "bytes": path.stat().st_size,
    "sha256": a3_hash_file(path)} for path in figure_data_paths]]).to_csv(
    A3_OUT_DIR / "A3_figure_manifest.csv", index=False, encoding="utf-8-sig")
A3_FIGURES_OK = len(figure_records) == 6
if not A3_FIGURES_OK:
    raise AssertionError("Chưa xuất đủ sáu hình")
display(Markdown('**Tệp hình và kết luận**'))
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    display(A3_FIGURE_INDEX[["file", "bytes", "purpose"]])
print("A3_FIGURES_OK =", A3_FIGURES_OK)
print("Đã lưu 6 PNG, 6 CSV dữ liệu, 1 CSV index và 1 CSV manifest trong:", A3_OUT_DIR)
print("ID 363 là ca chẩn đoán bổ sung sau khi đọc test; không dùng để chọn mô hình hoặc đổi tập ID.")
print(f"Thời gian vẽ/xuất: {time.perf_counter() - a3_figures_started:.1f} giây")
