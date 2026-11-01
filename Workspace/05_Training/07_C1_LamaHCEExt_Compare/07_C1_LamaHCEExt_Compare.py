# =============================================================================
# Bước C1 (phần so sánh) - tinh chỉnh 5 lõi của mô hình chung Sequential Forecast (LSTM, GRU, Transformer, S4D, Mamba),
# mỗi lõi 8 cấu hình (2 độ sâu x 2 mức dung lượng x 2 learning rate), 1 hạt giống (111), 30 epoch, chỉ validation.
# Bảng 40 cấu hình (39 lần C1 + LSTM 1 lớp, cùng tham số, learning rate 10^-3 = B* hạt giống 111), chọn hai cấu hình mỗi
# lõi cho C2 theo NSE trung vị validation, đánh đổi độ chính xác - chi phí, lệnh sinh notebook C2.
# Bài: Konold và cs., HESS 30:5067-5096 (2026). Kế hoạch: Document/01_Plan/02_Pipeline.md Mục 5.2 (lưới, cách chọn C2),
# 5.3 (nhiễu hạt giống), 6.1 (chỉ NSE trung vị validation dùng để chọn), 7 (dòng C1, C2).
# Notebook Kaggle "07_C1_LamaHCEExt_Compare": Accelerator = None (CPU), Internet = Off, Persistence = Files only,
# Environment = Pin to original environment. Add Input:
#   - Output 39 notebook 06_C1_LamaHCEExt_Tune_<cấu hình>_<lr>; với 8 cấu hình chạy hai phiên chỉ gắn Output notebook
#     phiên 2 (_Cont), thư mục run trong đó đủ 30 epoch;
#   - Output notebook huấn luyện B* hạt giống 111.
# Không cần GPU, không cần thư viện NeuralHydrology: chỉ đọc config.yml, validation_metrics.csv, output.log của thư mục run.
# Dán nguyên tệp vào notebook rồi Save & Run All, không sửa dòng nào.
#
# MỤC LỤC
#   Phần 1 - Tìm Output 40 run, kiểm sha256, giao thức, cấu hình
#   Phần 2 - Validation theo epoch, thời gian từng epoch, bảng 40 cấu hình
#   Phần 3 - Chọn cấu hình C2 mỗi lõi, so với nhiễu hạt giống
#   Phần 4 - Hình
#   Phần 5 - Lưu kết quả, lệnh sinh notebook C2, môi trường
# =============================================================================


# %% Phần 1 - Tìm Output 40 run, kiểm sha256, giao thức, cấu hình
# Mỗi notebook huấn luyện có <x>_output/environment.json (run_dir tính từ /kaggle/working của notebook đó, commit thư
# viện) và manifest.csv (tệp chính kèm sha256). Nhận Output theo environment.json, không theo tên đường dẫn. Notebook C1
# (khuôn Tools/C_Notebook_Template.py) có thêm c_output/summary.json (trạng thái, số tham số, mốc). Gắn trùng cả phiên 1
# và _Cont của một cấu hình thì giữ bản nhiều epoch hơn. B* = run hạt giống 111 duy nhất không phải C1. Kiểm: đủ 40 cấu
# hình lưới, C1 trạng thái "xong", 39 C1 cùng commit thư viện (in commit của B*), sha256 khớp, config.yml đúng cấu hình
# và giao thức 30 epoch.
import hashlib
import json
import os
import platform
import re
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from IPython.display import HTML, display

IS_KAGGLE = Path("/kaggle/working").exists()
INPUT_ROOT = Path(os.environ.get("C1_INPUT_ROOT", "/kaggle/input" if IS_KAGGLE else "c1_compare_input"))
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("c1_compare_work")
OUT_DIR = WORK_DIR / "c1_compare"
CORE_DEPTHS = {"lstm": (1, 2), "gru": (1, 2), "transformer": (1, 2), "s4d": (2, 4), "mamba": (2, 4)}
LEVELS = (1, 3)
LRS = {"lr1e-3": 1e-3, "lr5e-4": 5e-4}
B_STAR_CONF = "lstm_L1_x1_lr1e-3"                 # cấu hình C1 lấy từ B* hạt giống 111 (Pipeline Mục 5.2)
SEED = 111
EPOCHS = 30
N_BASINS = 451
NOISE_SD = 0.0079     # độ lệch chuẩn NSE trung vị validation giữa 3 hạt giống B0 (ngưỡng B*, Pipeline Mục 5.3)
C1_RE = re.compile(r"^C1_(lstm|gru|transformer|s4d|mamba)_L(\d)_x([13])_(lr1e-3|lr5e-4)_seed(\d+)$")
VERSION_DEFAULT = 1                               # Version bản lưu chính thức của mỗi notebook huấn luyện
VERSIONS = {}                                     # tên cấu hình => Version, khi khác VERSION_DEFAULT
ENV_LIBS = ["numpy", "pandas", "matplotlib", "PyYAML"]
OUT_DIR.mkdir(parents=True, exist_ok=True)


def show(title, df, digits=4):
    # Bảng HTML: số thực làm tròn `digits` chữ số, ô trống hiện "-".
    display(HTML(f"<h4 style='margin:12px 0 4px'>{title}</h4>"))
    display(df.style.hide(axis="index").format(precision=digits, na_rep="-"))


def show_checks(checks):
    # checks: [(mục, giá trị, cần, đạt)]; trả True nếu mọi dòng đạt.
    show("Kiểm", pd.DataFrame([{"mục": a, "giá trị": str(b), "cần": str(c), "đạt": "ĐẠT" if d else "KHÔNG"}
                               for a, b, c, d in checks]))
    return all(c[3] for c in checks)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_metrics(path):
    # validation_metrics.csv của NeuralHydrology (tester.py _save_results): cột đầu là mã lưu vực => str(int).
    m = pd.read_csv(path, index_col=0)
    m.index = [str(int(b)) for b in m.index]
    return m


def grid_configs():
    # 40 cấu hình lưới theo thứ tự lõi, độ sâu, mức, learning rate.
    return [f"{c}_L{d}_x{lv}_{lr}" for c, ds in CORE_DEPTHS.items() for d in ds for lv in LEVELS for lr in LRS]


# (1) Output các notebook huấn luyện: nhận theo environment.json
found = []
for env_path in sorted(INPUT_ROOT.rglob("environment.json")):
    env = json.loads(env_path.read_text(encoding="utf-8"))
    if "run_dir" not in env:
        continue                                  # environment.json của notebook khác (không có thư mục run)
    root = env_path.parent.parent                 # thư mục tương ứng /kaggle/working của notebook đó
    run = root / env["run_dir"]
    cfg = yaml.safe_load((run / "config.yml").read_text(encoding="utf-8-sig"))
    summary_path = env_path.parent / "summary.json"
    found.append({"root": root, "out": env_path.parent, "run": run, "env": env, "cfg": cfg,
                  "summary": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None,
                  "n_epochs": len(list(run.glob("model_epoch*.pt")))})

runs, dropped = {}, []
for f in found:
    m = C1_RE.match(f["cfg"]["experiment_name"])
    if m:
        conf = f"{m.group(1)}_L{m.group(2)}_x{m.group(3)}_{m.group(4)}"
    elif f["cfg"]["seed"] == SEED and f["cfg"]["experiment_name"].split("_")[0] not in ("C2", "TEST"):
        conf = B_STAR_CONF
    else:
        continue
    if conf in runs and conf != B_STAR_CONF:      # phiên 1 và _Cont cùng gắn => giữ bản nhiều epoch hơn
        keep, drop = sorted([runs[conf], f], key=lambda r: r["n_epochs"], reverse=True)
        runs[conf] = keep
        dropped.append(f"{conf}: bỏ {drop['out'].parent.name} ({drop['n_epochs']} epoch)")
    elif conf in runs:
        raise RuntimeError(f"Có 2 run được nhận là B* (hạt giống {SEED}, không phải C1): "
                           f"{runs[conf]['run'].name}, {f['run'].name} - chỉ gắn Output notebook B* hạt giống {SEED}")
    else:
        runs[conf] = f
GRID = grid_configs()
missing = [c for c in GRID if c not in runs]
if missing:
    raise RuntimeError(f"Thiếu {len(missing)} cấu hình: {missing} - kiểm Add Input")
for d in dropped:
    print("Gắn trùng:", d)

# (2) sha256 theo manifest.csv của từng Output
sha_rows = []
for conf in GRID:
    r = runs[conf]
    man = pd.read_csv(r["out"] / "manifest.csv")
    bad = [rel for rel, dg in zip(man["tệp"], man["sha256"]) if not (r["root"] / rel).exists() or sha256(r["root"] / rel) != dg]
    sha_rows.append({"cấu hình": conf, "tệp": len(man), "lệch": len(bad), "_bad": bad})

# (3) Bảng Output và kiểm config từng run
info_rows = []
for conf in GRID:
    r, cfg = runs[conf], runs[conf]["cfg"]
    core, depth, level, lr_tag = re.match(r"^(\w+?)_L(\d)_x(\d)_(lr.+)$", conf).groups()
    if conf == B_STAR_CONF:
        cfg_ok = cfg["seed"] == SEED and float(cfg["learning_rate"]) == LRS[lr_tag]
    else:
        cfg_ok = (cfg["model"] == "sequential_forecast" and cfg["sf_core"] == core and int(cfg["sf_layers"]) == int(depth)
                  and float(cfg["learning_rate"]) == LRS[lr_tag] and cfg["seed"] == SEED)
    s = r["summary"] or {}
    info_rows.append({"cấu hình": conf, "notebook": r["out"].parent.name if conf != B_STAR_CONF else "B*",
                      "Version": VERSIONS.get(conf, VERSION_DEFAULT), "thư mục run": r["run"].name,
                      "model": cfg["model"], "hidden_size": cfg["hidden_size"], "epochs": cfg["epochs"],
                      "patience": cfg["early_stopping_patience"], "data_on_gpu": cfg.get("data_on_gpu"),
                      "số model_epoch": r["n_epochs"], "trạng thái": s.get("status", "-"), "phiên": s.get("session", "-"),
                      "số tham số": s.get("n_params"), "mốc": s.get("target_params"),
                      "commit": r["env"].get("neuralhydrology_commit", "?")[:7], "config đúng": cfg_ok})
INFO = pd.DataFrame(info_rows)
c1_info = INFO[INFO["cấu hình"] != B_STAR_CONF]
TARGET = int(c1_info["mốc"].iloc[0])
INFO.loc[INFO["cấu hình"] == B_STAR_CONF, "số tham số"] = TARGET   # B* = lõi LSTM 1 lớp cùng tham số = mốc
show(f"Output {len(INFO)} run (B* = {runs[B_STAR_CONF]['cfg']['experiment_name']})",
     INFO.assign(**{"config đúng": INFO["config đúng"].map({True: "ĐẠT", False: "KHÔNG"})}))
commits = sorted(set(c1_info["commit"]))
B_STAR_COMMIT = INFO.set_index("cấu hình").loc[B_STAR_CONF, "commit"]
bad_sha = [r for r in sha_rows if r["lệch"]]
checks = [("Số cấu hình", len(INFO), "40 (39 C1 + B*)", len(INFO) == 40),
          ("sha256 khớp manifest.csv", f"{sum(r['tệp'] for r in sha_rows) - sum(r['lệch'] for r in sha_rows)} / "
           f"{sum(r['tệp'] for r in sha_rows)} tệp", "đủ", not bad_sha),
          ("Trạng thái C1", sorted(set(c1_info["trạng thái"])), "['xong']", set(c1_info["trạng thái"]) == {"xong"}),
          ("Config đúng cấu hình (lõi, độ sâu, lr, hạt giống)", int(INFO["config đúng"].sum()), 40,
           bool(INFO["config đúng"].all())),
          ("epochs / early_stopping_patience / data_on_gpu", sorted(set(zip(INFO["epochs"], INFO["patience"],
                                                                             INFO["data_on_gpu"]))),
           f"[({EPOCHS}, {EPOCHS}, True)]", set(zip(INFO["epochs"], INFO["patience"], INFO["data_on_gpu"]))
           == {(EPOCHS, EPOCHS, True)}),
          ("Số model_epoch*.pt mỗi run", sorted(set(INFO["số model_epoch"])), f"[{EPOCHS}]",
           set(INFO["số model_epoch"]) == {EPOCHS}),
          ("Mốc tham số chung của 39 C1", sorted(set(c1_info["mốc"])), "một giá trị", c1_info["mốc"].nunique() == 1),
          ("Lệch tham số so với mức x mốc (C1)",
           f"{(c1_info['số tham số'] / (c1_info['cấu hình'].str.extract(r'_x(\d)_')[0].astype(int) * TARGET) - 1).abs().max():.3%}",
           "<= 1%", bool(((c1_info["số tham số"] / (c1_info["cấu hình"].str.extract(r"_x(\d)_")[0].astype(int) * TARGET)
                          - 1).abs() <= 0.01).all())),
          ("Commit thư viện 39 C1", commits, "một commit (freeze-C)", len(commits) == 1)]
print(f"Commit thư viện của B*: {B_STAR_COMMIT} - C1 dùng lại B* hạt giống {SEED} khi B* chạy cùng mã bộ nạp với "
      f"freeze-C (Pipeline Mục 5.2); khác commit thì đối chiếu ở báo cáo.")
if bad_sha:
    print("Lệch sha256:", {r["cấu hình"]: r["_bad"] for r in bad_sha})
if not show_checks(checks):
    raise RuntimeError("Phần 1 CHƯA đạt (dòng KHÔNG ở trên) - dừng notebook")
print("=> Đạt. Chạy Phần 2.")


# %% Phần 2 - Validation theo epoch, thời gian từng epoch, bảng 40 cấu hình
# NSE, KGE trung vị 451 lưu vực từng epoch từ validation/model_epochNNN/validation_metrics.csv (tester.py _save_results).
# Epoch chọn = epoch NSE trung vị validation cao nhất (= best_model.pt từ bản sửa e794d7c; C1 đối chiếu saved_epoch trong
# summary.json). Thời gian mỗi epoch từ output.log của thư mục run (định dạng "%(asctime)s: %(message)s", logging_utils.py
# dòng 21; chạy hai phiên thì log nối tiếp): huấn luyện = "Epoch e average loss" trừ dòng log trước đó, validation =
# "Epoch e average validation loss" trừ "Epoch e average loss". Giờ GPU của cấu hình = tổng huấn luyện + validation.
LINE_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}): (.*)$")
LOSS_RE = re.compile(r"^Epoch (\d+) average loss:")
VAL_RE = re.compile(r"^Epoch (\d+) average validation loss:")
MARK_RE = re.compile(r"^(Early stopping enabled|New best model saved)")


def epoch_minutes(log_text):
    # {epoch: (phút huấn luyện, phút validation)} từ output.log; mốc đầu mỗi phiên là dòng "Early stopping enabled".
    out, mark, t_loss = {}, None, {}
    for raw in log_text.splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue
        ts, msg = pd.Timestamp(m.group(1).replace(",", ".")), m.group(2)
        if (mm := LOSS_RE.match(msg)):
            e = int(mm.group(1))
            t_loss[e] = ts
            out[e] = [(ts - mark).total_seconds() / 60 if mark is not None else np.nan, np.nan]
            mark = ts
        elif (mm := VAL_RE.match(msg)):
            e = int(mm.group(1))
            out[e][1] = (ts - t_loss[e]).total_seconds() / 60
            mark = ts
        elif MARK_RE.match(msg):
            mark = ts
    return out


CURVES, PER_BASIN, rows = {}, {}, []
for conf in GRID:
    r = runs[conf]
    ep_rows = []
    for e in range(1, EPOCHS + 1):
        m = read_metrics(r["run"] / "validation" / f"model_epoch{e:03d}" / "validation_metrics.csv")
        ep_rows.append({"epoch": e, "NSE": float(m["NSE"].median()), "KGE": float(m["KGE"].median()), "n": len(m)})
        PER_BASIN.setdefault(conf, {})[e] = m["NSE"]
    cur = pd.DataFrame(ep_rows)
    mins = epoch_minutes((r["run"] / "output.log").read_text(errors="replace"))
    cur["phút huấn luyện"] = cur["epoch"].map(lambda e: mins.get(e, [np.nan, np.nan])[0])
    cur["phút validation"] = cur["epoch"].map(lambda e: mins.get(e, [np.nan, np.nan])[1])
    CURVES[conf] = cur
    best = cur.loc[cur["NSE"].idxmax()]
    core, depth, level, lr_tag = re.match(r"^(\w+?)_L(\d)_x(\d)_(lr.+)$", conf).groups()
    s = runs[conf]["summary"] or {}
    rows.append({"cấu hình": conf, "lõi": core, "độ sâu": int(depth), "mức": f"x{level}", "lr": lr_tag,
                 "số tham số": int(INFO.set_index("cấu hình").loc[conf, "số tham số"]),
                 "epoch chọn": int(best["epoch"]), "NSE val": best["NSE"], "KGE val": best["KGE"],
                 "NSE val epoch 30": float(cur["NSE"].iloc[-1]),
                 "phút huấn luyện/epoch": float(cur["phút huấn luyện"].median()),
                 "phút validation/epoch": float(cur["phút validation"].iloc[1:].median()),
                 "giờ GPU": float((cur["phút huấn luyện"].fillna(cur["phút huấn luyện"].median())
                                   + cur["phút validation"]).sum() / 60),
                 "GPU MiB": s.get("gpu_mib_max"), "_saved": s.get("saved_epoch"), "_n": sorted(set(cur["n"]))})
TABLE = pd.DataFrame(rows)
TABLE["hạng trong lõi"] = TABLE.groupby("lõi")["NSE val"].rank(ascending=False, method="min").astype(int)
show(f"40 cấu hình C1 - NSE, KGE trung vị {N_BASINS} lưu vực ở epoch NSE validation cao nhất",
     TABLE.drop(columns=["_saved", "_n"]))
c1_tab = TABLE[TABLE["cấu hình"] != B_STAR_CONF]
checks = [("Validation đủ lưu vực mỗi epoch", sorted({n for v in TABLE["_n"] for n in v}), f"[{N_BASINS}]",
           all(v == [N_BASINS] for v in TABLE["_n"])),
          ("Epoch chọn = saved_epoch trong summary.json (39 C1)",
           f"{int((c1_tab['epoch chọn'] == c1_tab['_saved']).sum())} / {len(c1_tab)}", "bằng nhau",
           bool((c1_tab["epoch chọn"] == c1_tab["_saved"]).all())),
          ("Đọc được thời gian mọi epoch từ output.log",
           int(sum(CURVES[c]["phút validation"].notna().sum() for c in GRID)), 40 * EPOCHS,
           all(CURVES[c]["phút validation"].notna().all() for c in GRID))]
if not show_checks(checks):
    raise RuntimeError("Phần 2 CHƯA đạt (dòng KHÔNG ở trên) - dừng notebook")
print("=> Đạt. Chạy Phần 3.")


# %% Phần 3 - Chọn cấu hình C2 mỗi lõi, so với nhiễu hạt giống
# Pipeline Mục 5.2: mỗi lõi chạy C2 hai cấu hình - NSE trung vị validation cao nhất trong 4 cấu hình mức cùng tham số (x1)
# và cao nhất trong cả 8 (trùng thì một). Chọn theo giá trị cao nhất, không ngưỡng; ghi rõ khoảng cách tới cấu hình đứng
# sau so với nhiễu hạt giống (NOISE_SD) và khi epoch chọn là epoch cuối (đường học có thể chưa đạt đỉnh). Kèm hiệu NSE
# từng lưu vực so với B* (trung vị của hiệu, số lưu vực hơn) - chỉ để mô tả, so sánh chính thức ở C2 với 3 hạt giống.
# Ước giờ một lần C2 (30 epoch, thêm chấm test mỗi epoch): giờ GPU C1 + 30 lượt test, mỗi lượt lấy bằng một lượt
# validation (kỳ test 2014-2017 và validation 2010-2013 cùng 4 năm, 451 lưu vực; lượt test đo ở TEST khuôn ngắn hơn
# lượt validation) + 0,3 giờ cài đặt, dựng dữ liệu; vượt C2_SESSION_H thì cần notebook phiên 2 (_Cont).
C2_SESSION_H = 10.5     # 11 giờ giới hạn của khuôn trừ 0,25 giờ chừa Phần 7-8 và khoảng 0,25 giờ chấm best_model.pt


def b_star_diff(conf):
    # Hiệu NSE từng lưu vực (cấu hình - B*) ở epoch chọn của mỗi bên.
    a = PER_BASIN[conf][int(TABLE.set_index("cấu hình").loc[conf, "epoch chọn"])]
    b = PER_BASIN[B_STAR_CONF][int(TABLE.set_index("cấu hình").loc[B_STAR_CONF, "epoch chọn"])]
    d = (a - b).dropna()
    return float(d.median()), int((d > 0).sum()), len(d)


sel_rows, C2_CONFS = [], []
for core in CORE_DEPTHS:
    t = TABLE[TABLE["lõi"] == core].sort_values("NSE val", ascending=False)
    picks = {"tốt nhất mức x1": t[t["mức"] == "x1"].iloc[0], "tốt nhất cả 8": t.iloc[0]}
    for rule, row in picks.items():
        if row["cấu hình"] in C2_CONFS:
            sel_rows[-1]["quy tắc"] += " + " + rule
            continue
        pool = t[t["mức"] == "x1"] if rule == "tốt nhất mức x1" else t
        gap = float(row["NSE val"] - pool.iloc[1]["NSE val"])
        med, n_win, n = b_star_diff(row["cấu hình"])
        c2_hours = row["giờ GPU"] + EPOCHS * row["phút validation/epoch"] / 60 + 0.3
        sel_rows.append({"lõi": core, "quy tắc": rule, "cấu hình": row["cấu hình"], "NSE val": row["NSE val"],
                         "cấu hình đứng sau": pool.iloc[1]["cấu hình"], "cách đứng sau": gap,
                         "cách / nhiễu hạt giống": gap / NOISE_SD, "epoch chọn": int(row["epoch chọn"]),
                         "epoch chọn = 30": "có" if int(row["epoch chọn"]) == EPOCHS else "",
                         "hiệu NSE với B* (trung vị lưu vực)": med if row["cấu hình"] != B_STAR_CONF else np.nan,
                         "lưu vực hơn B*": f"{n_win} / {n}" if row["cấu hình"] != B_STAR_CONF else "-",
                         "giờ C2 ước": c2_hours, "cần _Cont": "có" if c2_hours > C2_SESSION_H else ""})
        C2_CONFS.append(row["cấu hình"])
SEL = pd.DataFrame(sel_rows)
show(f"Cấu hình chạy C2 ({len(SEL)} cấu hình x 3 hạt giống = {3 * len(SEL)} lần); nhiễu hạt giống {NOISE_SD}", SEL)
C2_CONT = SEL.loc[SEL["cần _Cont"] == "có", "cấu hình"].tolist()
n_close = int((SEL["cách / nhiễu hạt giống"] < 1).sum())
print(f"{n_close} / {len(SEL)} lựa chọn cách cấu hình đứng sau dưới 1 lần nhiễu hạt giống: thứ hạng giữa chúng chưa chắc "
      f"với 1 hạt giống, ghi rõ khi kết luận (Pipeline Mục 5.2).")
print(f"Tổng giờ C2 ước: {3 * SEL['giờ C2 ước'].sum():.0f} giờ; cần phiên 2: {C2_CONT or 'không'}")
print("=> Đạt. Chạy Phần 4.")


# %% Phần 4 - Hình
# Hình 1 (2 hàng x 5 lõi): (a) NSE trung vị validation của 8 cấu hình mỗi lõi (trục ngang độ sâu và mức, màu learning
# rate), đánh dấu cấu hình chọn cho C2; (b) đường học NSE trung vị validation theo epoch của 8 cấu hình.
# Hình 2: (a) NSE validation theo giờ GPU của 40 cấu hình (màu lõi, hình mức); (b) phân bố NSE từng lưu vực của các
# cấu hình chọn và B*. Chữ trong hình tiếng Anh, chú thích tiếng Việt in ngay dưới bằng caption.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG_DIR = OUT_DIR / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CORE_LABEL = {"lstm": "LSTM", "gru": "GRU", "transformer": "Transformer", "s4d": "S4D", "mamba": "Mamba"}
CORE_COLOR = {"lstm": "C0", "gru": "C1", "transformer": "C2", "s4d": "C3", "mamba": "C4"}
LR_COLOR = {"lr1e-3": "C0", "lr5e-4": "C3"}
UNIT = {"lstm": "L", "gru": "L", "transformer": "L", "s4d": "B", "mamba": "B"}


def caption(text):
    # Chú thích tiếng Việt in ngay dưới hình (chữ trong hình tiếng Anh).
    print(f"Hình: {text}")


def vn(x, digits=3):
    # Số thập phân trong chú thích tiếng Việt dùng dấu phẩy.
    return f"{x:.{digits}f}".replace(".", ",")


def conf_label(conf):
    core, depth, level, lr_tag = re.match(r"^(\w+?)_L(\d)_x(\d)_(lr.+)$", conf).groups()
    return f"{CORE_LABEL[core]} {depth}{UNIT[core]} x{level} {lr_tag.removeprefix('lr')}"


# Hình 1
fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey="row")
for j, core in enumerate(CORE_DEPTHS):
    t = TABLE[TABLE["lõi"] == core]
    ax = axes[0, j]
    slots = [(d, lv) for d in CORE_DEPTHS[core] for lv in LEVELS]
    for k, (d, lv) in enumerate(slots):
        for i, lr_tag in enumerate(LRS):
            row = t[(t["độ sâu"] == d) & (t["mức"] == f"x{lv}") & (t["lr"] == lr_tag)].iloc[0]
            chosen = row["cấu hình"] in C2_CONFS
            ax.scatter(k + (i - 0.5) * 0.25, row["NSE val"], s=90 if chosen else 40, color=LR_COLOR[lr_tag],
                       edgecolor="black" if chosen else "none", linewidth=1.5, zorder=3)
    ax.set_xticks(range(len(slots)), [f"{d}{UNIT[core]} x{lv}" for d, lv in slots])
    ax.set_title(f"(a{j + 1}) {CORE_LABEL[core]}: best validation NSE")
    ax.grid(alpha=0.3)
    ax = axes[1, j]
    for conf in t["cấu hình"]:
        cur = CURVES[conf]
        ax.plot(cur["epoch"], cur["NSE"], lw=2 if conf in C2_CONFS else 0.9, alpha=1 if conf in C2_CONFS else 0.6,
                label=conf_label(conf).split(" ", 1)[1])
    ax.set(xlabel="epoch", title=f"(b{j + 1}) {CORE_LABEL[core]}: validation NSE per epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
axes[0, 0].set_ylabel(f"median NSE of {N_BASINS} basins")
axes[1, 0].set_ylabel(f"median NSE of {N_BASINS} basins")
handles = [Line2D([], [], marker="o", ls="", color=LR_COLOR[k], label=f"lr {k.removeprefix('lr')}") for k in LRS] + [Line2D([], [], marker="o", ls="", color="white", markeredgecolor="black",
                                             markersize=9, label="selected for C2")]
axes[0, 4].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.suptitle(f"C1 tuning: 8 configurations per core (L = layers, B = blocks, x1 = {TARGET:,} parameters, x3 = about "
             f"three times), seed {SEED}, {EPOCHS} epochs")
fig.tight_layout()
fig.savefig(FIG_DIR / "C1_Configs.png", dpi=150, bbox_inches="tight")
plt.show()
best_all = TABLE.loc[TABLE["NSE val"].idxmax()]
caption(f"tinh chỉnh C1, mỗi cột một lõi. Hàng trên: NSE trung vị {N_BASINS} lưu vực của kỳ validation ở epoch tốt nhất "
        f"cho 8 cấu hình (trục ngang: số lớp L hoặc số khối B và mức dung lượng; màu: learning rate); chấm viền đen là "
        f"cấu hình chạy chính thức C2. Hàng dưới: NSE trung vị validation theo epoch, đường đậm là cấu hình chọn. Cao "
        f"nhất cả lưới là {conf_label(best_all['cấu hình'])} (NSE {vn(best_all['NSE val'])}); nhiễu giữa 3 hạt giống "
        f"của LSTM là {vn(NOISE_SD, 4)}, nên các chấm cách nhau ít hơn mức này chưa phân biệt được với 1 hạt giống.")

# Hình 2
fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
for core in CORE_DEPTHS:
    for lv, mk in [("x1", "o"), ("x3", "^")]:
        t = TABLE[(TABLE["lõi"] == core) & (TABLE["mức"] == lv)]
        ax.scatter(t["giờ GPU"], t["NSE val"], marker=mk, color=CORE_COLOR[core], s=45,
                   label=f"{CORE_LABEL[core]} {lv}")
ax.set(xlabel="GPU hours per C1 run (T4, 30 epochs, training + validation)", ylabel="median validation NSE",
       title="(a) Accuracy versus cost, 40 configurations")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
ax = axes[1]
box_confs = [B_STAR_CONF] + [c for c in C2_CONFS if c != B_STAR_CONF]
data = [PER_BASIN[c][int(TABLE.set_index("cấu hình").loc[c, "epoch chọn"])].dropna().clip(lower=-1) for c in box_confs]
bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True)
for patch, c in zip(bp["boxes"], box_confs):
    patch.set_facecolor(CORE_COLOR[c.split("_")[0]])
    patch.set_alpha(0.5)
ax.set_yticks(range(1, len(box_confs) + 1), [("B* " if c == B_STAR_CONF else "") + conf_label(c) for c in box_confs],
              fontsize=8)
ax.set(xlabel="validation NSE per basin (clipped at -1, outliers hidden)",
       title="(b) Basin NSE of B* and the configurations selected for C2")
ax.grid(alpha=0.3, axis="x")
fig.suptitle(f"C1 tuning: cost and per-basin spread (seed {SEED})")
fig.tight_layout()
fig.savefig(FIG_DIR / "C1_Cost_Basins.png", dpi=150, bbox_inches="tight")
plt.show()
cheap = TABLE.loc[TABLE["giờ GPU"].idxmin()]
dear = TABLE.loc[TABLE["giờ GPU"].idxmax()]
caption(f"(a) NSE trung vị validation theo số giờ GPU một lần C1 (huấn luyện + validation 30 epoch trên T4), màu theo "
        f"lõi, chấm tròn là mức cùng tham số, tam giác là mức gấp 3; rẻ nhất {conf_label(cheap['cấu hình'])} "
        f"({vn(cheap['giờ GPU'], 1)} giờ), đắt nhất {conf_label(dear['cấu hình'])} ({vn(dear['giờ GPU'], 1)} giờ). (b) phân "
        f"bố NSE validation từng lưu vực (hộp: tứ phân vị, vạch: trung vị; cắt ở -1, ẩn điểm ngoại lai) của B* và "
        f"{len(box_confs) - 1} cấu hình chọn cho C2.")
print("=> Đạt. Chạy Phần 5.")


# %% Phần 5 - Lưu kết quả, lệnh sinh notebook C2, môi trường
# c1_table.csv (40 cấu hình), c1_curves.csv (validation theo epoch), c1_selection.json (cấu hình C2, cần _Cont, lệnh
# chạy Tools/C_Notebooks_Build.py ở máy để sinh notebook C2), environment.json (Input, thư viện), manifest.csv (sha256).
TABLE.drop(columns=["_saved", "_n"]).to_csv(OUT_DIR / "c1_table.csv", index=False)
pd.concat([CURVES[c].assign(**{"cấu hình": c}) for c in GRID]).to_csv(OUT_DIR / "c1_curves.csv", index=False)
SEL.to_csv(OUT_DIR / "c1_selection.csv", index=False)
# Cấu hình của B* được chọn thì C2 dùng 3 hạt giống B* sẵn có (cùng mô hình, đã chấm test mọi epoch), không sinh lại
C2_BUILD = [c for c in C2_CONFS if c != B_STAR_CONF]
build_cmd = ("python Tools/C_Notebooks_Build.py c2 " + " ".join(C2_BUILD)
             + (" --cont " + " ".join(c for c in C2_CONT if c != B_STAR_CONF) if set(C2_CONT) - {B_STAR_CONF} else ""))
selection = {"c2_configs": C2_CONFS, "c2_build": C2_BUILD, "c2_reuse_b_star": B_STAR_CONF in C2_CONFS,
             "c2_cont": C2_CONT, "build_command": build_cmd, "noise_sd": NOISE_SD,
             "target_params": TARGET, "b_star_run": runs[B_STAR_CONF]["cfg"]["experiment_name"],
             "rows": json.loads(SEL.to_json(orient="records", force_ascii=False))}
(OUT_DIR / "c1_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")


def lib_version(lib):
    try:
        return metadata.version(lib)
    except metadata.PackageNotFoundError:
        return "không cài"


env = {"notebook": "07_C1_LamaHCEExt_Compare", "python": platform.python_version(), "platform": platform.platform(),
       "inputs": {c: {"notebook": INFO.set_index("cấu hình").loc[c, "notebook"],
                      "version": int(INFO.set_index("cấu hình").loc[c, "Version"]),
                      "run_dir": runs[c]["run"].name} for c in GRID},
       "neuralhydrology_commit": commits[0], "b_star_commit": B_STAR_COMMIT, "libs": {lib: lib_version(lib) for lib in ENV_LIBS}}
(OUT_DIR / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
man = [{"tệp": p.relative_to(WORK_DIR).as_posix(), "KB": round(p.stat().st_size / 1e3, 1), "sha256": sha256(p)}
       for p in sorted(OUT_DIR.rglob("*")) if p.is_file() and p.name != "manifest.csv"]
pd.DataFrame(man).to_csv(OUT_DIR / "manifest.csv", index=False)
show("Tệp kết quả (manifest.csv)", pd.DataFrame(man).assign(sha256=lambda d: d["sha256"].str[:12] + "..."))
print("Lệnh sinh notebook C2 (chạy ở máy, từ gốc repo):")
print("  " + build_cmd)
print("=> Đạt. Xong notebook.")
