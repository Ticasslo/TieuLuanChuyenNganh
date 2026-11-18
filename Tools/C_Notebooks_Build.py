# Sinh các notebook huấn luyện bước C và bước D từ khuôn Tools/C_Notebook_Template.py: mỗi tệp C1/C2 là khuôn chép nguyên,
# chỉ thay khối CẤU HÌNH ở Phần 1 (NOTEBOOK, STEP, CORE, DEPTH, LEVEL, LR, RUN_SEED); tệp D giữ nguyên khuôn C trên
# freeze-C (db6dc1f, 30 epoch, hạn 1 ngày) và áp D_PATCH để đổi đúng seq_length 365 -> 730 (Pipeline Mục 4.3, 8.3).
# Chạy (từ gốc repo):
#   python Tools/C_Notebooks_Build.py c1 [--out THƯ_MỤC]
#       39 notebook C1 (5 lõi x 2 độ sâu x 2 mức x 2 learning rate, bỏ lstm_L1_x1_lr1e-3 = B* hạt giống 111; vòng 1
#       tới epoch 15) và 39 notebook phiên 2 (_Cont, vòng 2 tới epoch 30, chỉ chạy cho cấu hình được giữ ở
#       11_C1_LamaHCEExt_Compare), mặc định vào Workspace/05_Training/10_C1_Tune/.
#   python Tools/C_Notebooks_Build.py c2 mamba_L4_x1_lr1e-3 [...] [--cont mamba_L4_x3_lr1e-3 ...] [--out THƯ_MỤC]
#       3 hạt giống (111, 222, 333) cho mỗi cấu hình C2, _Cont cho cấu hình ghi ở --cont, mặc định vào
#       Workspace/05_Training/12_C2_Train/. Tên C2 bỏ "LamaHCEExt" để vừa 50 ký tự (giới hạn tên notebook Kaggle).
#   python Tools/C_Notebooks_Build.py c2 gru_L2_x1_lr1e-3 --grad-guard gru_L2_x1_lr1e-3:222
#       Chỉ sinh lại tệp của cặp cấu hình:hạt giống ghi ở --grad-guard, thêm chốt chặn gradient: tiến trình huấn luyện bỏ
#       hẳn bước có gradient NaN/Inf (không cập nhật trọng số, trạng thái Adam), đếm số bước bỏ. Run không có gradient
#       NaN/Inf thì chốt chặn không đổi gì (bước như vậy ở thư viện gốc làm trọng số thành NaN, run dừng vì loss NaN).
#   python Tools/C_Notebooks_Build.py d [cấu_hình ...] [--grad-guard mamba_L2_x1_lr1e-3:222 ...] [--out THƯ_MỤC]
#       Mặc định sinh đủ 9 notebook D (lstm_L2_x1_lr1e-3, mamba_L2_x1_lr1e-3, s4d_L4_x1_lr1e-3 x hạt giống 111, 222, 333)
#       và 9 notebook _Cont vào Workspace/05_Training/16_D_Train/, tắt chốt chặn gradient mặc định (như 9 run C2 đối chứng).
#       Nếu một run gặp tràn gradient NaN/Inf, chạy lại với --grad-guard <cấu_hình>:<hạt_giống> để sinh lại cặp phiên 1 và
#       _Cont của run đó kèm ghi nhận số bước bỏ trong summary.json / environment.json.
# Chạy lại với cùng khuôn ra tệp giống hệt (in sha256 từng tệp để so).
#
# Mục lục:
#   Phần 1 - Lưới cấu hình, tên tệp
#   Phần 2 - Thay khối CẤU HÌNH, kiểm tệp sinh ra
#   Phần 3 - Dòng lệnh

import argparse
import hashlib
import re
import sys
from pathlib import Path

# %% Phần 1 - Lưới cấu hình, tên tệp
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "Tools" / "C_Notebook_Template.py"
OUT_DIRS = {"C1": ROOT / "Workspace" / "05_Training" / "10_C1_Tune",
            "C2": ROOT / "Workspace" / "05_Training" / "12_C2_Train",
            "D": ROOT / "Workspace" / "05_Training" / "16_D_Train"}
PREFIX = {"C1": "10_C1_Tune", "C2": "12_C2_Train", "D": "16_D_Train"}
MAX_NAME = 50                                # tên notebook Kaggle tối đa 50 ký tự
CORE_DEPTHS = {"lstm": (1, 2), "gru": (1, 2), "transformer": (1, 2), "s4d": (2, 4), "mamba": (2, 4)}
LEVELS = (1, 3)
LRS = {"lr1e-3": 1e-3, "lr5e-4": 5e-4}
C2_SEEDS = (111, 222, 333)
D_CONFIGS = ("lstm_L2_x1_lr1e-3", "mamba_L2_x1_lr1e-3", "s4d_L4_x1_lr1e-3")   # Pipeline Mục 4.3, 8.3
B_STAR = "lstm_L1_x1_lr1e-3"                 # cấu hình LSTM chuẩn của C1 = B* hạt giống 111, không chạy lại
CONF_RE = re.compile(r"^(lstm|gru|transformer|s4d|mamba)_L(\d)_x([13])_(lr1e-3|lr5e-4)$")
BLOCK_START = "# ---- CẤU HÌNH (Tools/C_Notebooks_Build.py ghi lại khối này) ----\n"
BLOCK_END = "# ---- HẾT CẤU HÌNH ----\n"
# Chốt chặn gradient (--grad-guard): ba khối chèn vào khuôn, mỗi khối là (dòng neo có đúng 1 lần trong khuôn, chèn "after"
# hoặc "before" dòng neo, khối); gỡ ba khối khỏi tệp sinh ra phải trùng tệp thường từng byte
GUARD_MODULE = r'''
import importlib.abc
import importlib.util
import os
import sys

TARGET = "neuralhydrology.training.basetrainer"


def _install():
    import torch
    state = {"step": 0, "skipped": 0}
    original_clip = torch.nn.utils.clip_grad_norm_

    def clip(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        state["step"] += 1
        grads = [p.grad for p in params if p.grad is not None]
        if grads and not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
            state["skipped"] += 1
            for p in params:
                p.grad = None
            print(f"[GRAD_GUARD] skipped optimizer step {state['step']}: non-finite gradient; "
                  f"total {state['skipped']}", flush=True)
            return grads[0].new_tensor(float("nan"))
        return original_clip(params, max_norm, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = clip
    print("[GRAD_GUARD] installed", flush=True)


class _GradGuardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            _install()
        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("C_GRAD_GUARD") == "1" and not any(type(f).__name__ == "_GradGuardFinder" for f in sys.meta_path):
    sys.meta_path.insert(0, _GradGuardFinder())
'''
GUARD_PATCH = [
    ('env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")\n', "after",
     "# Chốt chặn gradient (chỉ tệp sinh với --grad-guard, Tools/C_Notebooks_Build.py): tiến trình huấn luyện con nạp\n"
     "# module grad_guard qua tệp .pth (bật khi C_GRAD_GUARD=1), bỏ hẳn bước có gradient NaN/Inf - không cập nhật trọng\n"
     "# số và trạng thái Adam - và ghi \"[GRAD_GUARD]\" vào log; run không có gradient NaN/Inf thì không đổi gì.\n"
     "import site\n"
     "GUARD_SRC = r'''" + GUARD_MODULE + "'''\n"
     "compile(GUARD_SRC, \"grad_guard.py\", \"exec\")\n"
     "GUARD_SITE = Path(site.getsitepackages()[0])\n"
     "(GUARD_SITE / \"grad_guard.py\").write_text(GUARD_SRC, encoding=\"utf-8\")\n"
     "(GUARD_SITE / \"zz_grad_guard.pth\").write_text(\"import grad_guard\\n\", encoding=\"utf-8\")\n"
     "for pyc in (GUARD_SITE / \"__pycache__\").glob(\"grad_guard.*.pyc\"):\n"
     "    pyc.unlink()\n"
     "env[\"C_GRAD_GUARD\"] = \"1\"\n"
     "guard_chk = subprocess.run([sys.executable, \"-c\", \"import sys; print(any(type(f).__name__ == '_GradGuardFinder' \"\n"
     "                            \"for f in sys.meta_path))\"], capture_output=True, text=True, env=env)\n"
     "if guard_chk.stdout.strip() != \"True\":\n"
     "    raise RuntimeError(f\"Chốt chặn gradient chưa nạp ở tiến trình con: {guard_chk.stdout} {guard_chk.stderr[-500:]}\")\n"
     "print(f\"Chốt chặn gradient: đã gắn ({GUARD_SITE / 'zz_grad_guard.pth'})\")\n"),
    ('text = STDOUT_LOG.read_text(errors="replace")\n', "after",
     "GRAD_GUARD_SKIPS = text.count(\"[GRAD_GUARD] skipped\")         # chốt chặn gradient (--grad-guard)\n"
     "if \"[GRAD_GUARD] installed\" not in text:\n"
     "    raise RuntimeError(\"Log huấn luyện không có dòng [GRAD_GUARD] installed - chốt chặn gradient chưa chạy\")\n"
     "print(f\"Chốt chặn gradient: bỏ {GRAD_GUARD_SKIPS} bước có gradient NaN/Inf\")\n"),
    ('(OUT_DIR / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")\n',
     "before",
     "env[\"protocol\"][\"grad_guard_skipped_steps\"] = GRAD_GUARD_SKIPS  # chốt chặn gradient (--grad-guard)\n")]
D_GUARD_PATCH = [
    GUARD_PATCH[0],
    ('text = STDOUT_LOG.read_text(errors="replace")\n', "after",
     "GRAD_GUARD_SKIPS = (PREV_TEXT + text).count(\"[GRAD_GUARD] skipped\")  # chốt chặn gradient (--grad-guard, cả hai phiên)\n"
     "if not SKIP_TRAIN and \"[GRAD_GUARD] installed\" not in text:\n"
     "    raise RuntimeError(\"Log huấn luyện không có dòng [GRAD_GUARD] installed - chốt chặn gradient chưa chạy\")\n"
     "print(f\"Chốt chặn gradient: bỏ {GRAD_GUARD_SKIPS} bước có gradient NaN/Inf (tổng cộng)\")\n"),
    ('(OUT_DIR / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")\n',
     "before",
     "env[\"protocol\"][\"grad_guard_enabled\"] = True\n"
     "env[\"protocol\"][\"grad_guard_skipped_steps\"] = GRAD_GUARD_SKIPS  # chốt chặn gradient (--grad-guard)\n"
     "if GRAD_GUARD_SKIPS > 0:\n"
     "    print(f\"CẢNH BÁO: chốt chặn gradient đã bỏ {GRAD_GUARD_SKIPS} bước cập nhật; \"\n"
     "          \"rà lại tính tương đương cặp 365/730 trước khi đưa vào so sánh chính.\")\n")]
# Bản vá giao thức bước D (02_Pipeline.md Mục 4.3, 8.3): giữ nguyên khuôn freeze-C, 30 epoch, hạn 1 ngày, chỉ đổi
# seq_length 365 -> 730 (729 ngày quá khứ + 1 ngày dự báo); mỗi cặp (cũ, mới) xuất hiện đúng 1 lần và gỡ ngược từng byte
D_PATCH = [
    ('    (STEP in ("C1", "C2"), f"STEP {STEP} phải là C1 hoặc C2"),\n',
     '    (STEP in ("C1", "C2", "D"), f"STEP {STEP} phải là C1, C2 hoặc D"),\n'),
    ('win_start = cfg_author.train_start_date - pd.Timedelta(days=cfg_author.seq_length - 1)\n',
     'D_SEQ_LENGTH = 730                       # bước D (Pipeline Mục 4.3, 8.3): chuỗi 730 ngày (729 quá khứ + 1 ngày dự báo)\n'
     'win_start = cfg_author.train_start_date - pd.Timedelta(days=(D_SEQ_LENGTH if STEP == "D" else cfg_author.seq_length) - 1)\n'),
    ('cfg_c.update_config({**size, "experiment_name": EXP_NAME, "epochs": CFG_EPOCHS})\n',
     'cfg_c.update_config({**size, "experiment_name": EXP_NAME, "epochs": CFG_EPOCHS,\n'
     '                     **({"seq_length": D_SEQ_LENGTH} if STEP == "D" else {})})\n'),
    ('run_diff = config_diff(cfg_bstar_saved, cfg_run, KEYS_RUNTIME + KEYS_MODEL + (["epochs"] if STEP == "C1" else []))\n',
     'run_diff = config_diff(cfg_bstar_saved, cfg_run,\n'
     '                       KEYS_RUNTIME + KEYS_MODEL + (["epochs"] if STEP == "C1" else [])\n'
     '                       + (["seq_length"] if STEP == "D" else []))\n'),
    ('    {"kiểm": "lịch learning rate (T_max phải là số epoch của giao thức, không theo khóa epochs)",\n',
     '    {"kiểm": "seq_length / forecast_seq_length / predict_last_n (bước D: 730 / 1 / 1)",\n'
     '     "giá trị": f"{cfg_run.seq_length} / {cfg_run.forecast_seq_length} / {cfg_run.predict_last_n}",\n'
     '     "đạt": [cfg_run.seq_length, cfg_run.forecast_seq_length, cfg_run.predict_last_n] == [D_SEQ_LENGTH, 1, 1]},\n'
     '    {"kiểm": "lịch learning rate (T_max phải là số epoch của giao thức, không theo khóa epochs)",\n'),
    ('SAME_KEYS = ["model", "sf_core", "sf_layers", "sf_dropout", "sf_s4d_lr", "hidden_size", "learning_rate", "seed",\n'
     '             "epochs", "experiment_name", "batch_size", "hindcast_inputs", "forecast_inputs", "dynamic_inputs",\n',
     'SAME_KEYS = ["model", "sf_core", "sf_layers", "sf_dropout", "sf_s4d_lr", "hidden_size", "learning_rate", "seed",\n'
     '             "epochs", "experiment_name", "batch_size", "seq_length", "forecast_seq_length", "predict_last_n",\n'
     '             "hindcast_inputs", "forecast_inputs", "dynamic_inputs",\n'),
    ('DO_TEST = STEP == "C2"\n',
     'DO_TEST = STEP in ("C2", "D")\n'),
    ('                   ("Lưu vực có NSE test (best_model.pt)", int(best_row["lưu vực có NSE test"]),\n'
     '                    "450 (lưu vực 55 không có qmean)", best_row["lưu vực có NSE test"] == 450),\n',
     '                   ("Lưu vực có NSE test (best_model.pt)", int(best_row["lưu vực có NSE test"]),\n'
     '                    "451 (từ B2 lưu vực 55 có dự báo)", best_row["lưu vực có NSE test"] == 451),\n'),
    ('fig.suptitle(f"{STEP} Sequential Forecast, {CORE_LABEL[CORE]} core, {DEPTH} {DEPTH_UNIT[CORE]}s, {N_PARAMS:,} parameters "\n'
     '             f"(x{LEVEL}), lr {LR:g}, seed {RUN_SEED}: learning curves ({n_epochs} of {EPOCHS_FULL} epochs)")\n',
     'fig.suptitle(f"{STEP} Sequential Forecast (seq_length {cfg_saved.seq_length} days), {CORE_LABEL[CORE]} core, "\n'
     '             f"{DEPTH} {DEPTH_UNIT[CORE]}s, {N_PARAMS:,} parameters (x{LEVEL}), lr {LR:g}, seed {RUN_SEED}: "\n'
     '             f"learning curves ({n_epochs} of {EPOCHS_FULL} epochs)")\n'),
    ('caption(f"đường học của lõi {CORE_LABEL[CORE]} {DEPTH} {\'lớp\' if DEPTH_UNIT[CORE] == \'layer\' else \'khối\'}, "\n',
     'caption(f"đường học bước D (chuỗi {cfg_saved.seq_length} ngày) của lõi {CORE_LABEL[CORE]} {DEPTH} "\n'
     '        f"{\'lớp\' if DEPTH_UNIT[CORE] == \'layer\' else \'khối\'}, "\n'),
    ('           "hidden_size": cfg_saved.hidden_size, "inner_size": inner, "n_params": N_PARAMS, "target_params": TARGET,\n',
     '           "hidden_size": cfg_saved.hidden_size, "inner_size": inner, "n_params": N_PARAMS, "target_params": TARGET,\n'
     '           "seq_length": cfg_saved.seq_length,\n'),
    ('                    "max_training_hours": MAX_TRAIN_H, "test_every_epoch": DO_TEST, "deterministic": False},\n',
     '                    "max_training_hours": MAX_TRAIN_H, "test_every_epoch": DO_TEST, "deterministic": False,\n'
     '                    "seq_length": cfg_saved.seq_length, "grad_guard_enabled": False, "grad_guard_skipped_steps": 0},\n')]


def c1_configs():
    # 40 cấu hình lưới trừ B* => 39, theo thứ tự lõi, độ sâu, mức, learning rate.
    return [f"{core}_L{d}_x{lv}_{lr}" for core, depths in CORE_DEPTHS.items() for d in depths for lv in LEVELS
            for lr in LRS if f"{core}_L{d}_x{lv}_{lr}" != B_STAR]


def parse(conf):
    # "mamba_L4_x3_lr1e-3" => ("mamba", 4, 3, "lr1e-3"); sai lưới => lỗi.
    m = CONF_RE.match(conf)
    if not m or int(m.group(2)) not in CORE_DEPTHS[m.group(1)]:
        raise ValueError(f"Cấu hình {conf} không có trong lưới {CORE_DEPTHS}, mức {LEVELS}, {list(LRS)}")
    return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)


# %% Phần 2 - Thay khối CẤU HÌNH, kiểm tệp sinh ra
def render(template, step, conf, seed, cont):
    # Khuôn với khối CẤU HÌNH mới; giữ nguyên chú thích cuối mỗi dòng của khối.
    core, depth, level, lr_tag = parse(conf)
    name = f"{PREFIX[step]}_{conf}" + (f"_Seed{seed}" if step in ("C2", "D") else "") + ("_Cont" if cont else "")
    values = {"NOTEBOOK": f'"{name}"', "STEP": f'"{step}"', "CORE": f'"{core}"', "DEPTH": str(depth),
              "LEVEL": str(level), "LR": lr_tag.removeprefix("lr"), "RUN_SEED": str(seed)}
    a = template.index(BLOCK_START) + len(BLOCK_START)
    b = template.index(BLOCK_END)
    lines = []
    for line in template[a:b].splitlines():
        m = re.match(r"^([A-Z_]+) = (\S+)( +#.*)?$", line)
        if not m or m.group(1) not in values:
            raise ValueError(f"Dòng lạ trong khối CẤU HÌNH của khuôn: {line!r}")
        new = f"{m.group(1)} = {values.pop(m.group(1))}"
        if m.group(3):                                               # giữ cột bắt đầu chú thích như khuôn
            col = line.index("#", len(m.group(1)) + 3 + len(m.group(2)))
            new = new.ljust(col - 1) + " " + m.group(3).lstrip()
        lines.append(new + "\n")
    if values:
        raise ValueError(f"Khối CẤU HÌNH của khuôn thiếu {list(values)}")
    return name, template[:a] + "".join(lines) + template[b:]


def apply_d_patch(text, name):
    # Chèn các thay đổi giao thức bước D quanh dòng neo; neo không có đúng 1 lần => lỗi.
    for old, new in D_PATCH:
        if text.count(old) != 1:
            raise ValueError(f"{name}: khuôn không có đúng 1 khối {old!r}")
        text = text.replace(old, new)
    return text


def strip_d_patch(text):
    # Ngược của apply_d_patch: kiểm tệp D chỉ khác tệp khuôn ở đúng D_PATCH.
    for old, new in D_PATCH:
        text = text.replace(new, old)
    return text


def apply_grad_guard(text, name, patch=GUARD_PATCH):
    # Chèn ba khối GUARD_PATCH quanh dòng neo; neo không có đúng 1 lần => lỗi thay vì bỏ qua.
    for anchor, where, block in patch:
        if text.count(anchor) != 1:
            raise ValueError(f"{name}: khuôn không có đúng 1 dòng {anchor!r}")
        text = text.replace(anchor, anchor + block if where == "after" else block + anchor)
    return text


def strip_grad_guard(text, patch=GUARD_PATCH):
    # Ngược của apply_grad_guard: kiểm tệp có chốt chặn chỉ khác tệp thường ở ba khối chèn.
    for anchor, where, block in patch:
        text = text.replace(anchor + block if where == "after" else block + anchor, anchor)
    return text


def build(step, confs, seeds, cont_confs, out_dir, grad_guard=None):
    template = TEMPLATE.read_text(encoding="utf-8")
    if "\r" in template:
        raise ValueError("Khuôn có ký tự CR - lưu khuôn với xuống dòng LF")
    jobs = [(conf, seed, False) for conf in confs for seed in seeds]
    jobs += [(conf, seed, True) for conf in cont_confs for seed in seeds]
    if grad_guard is not None:
        jobs = [(conf, seed, False) for conf, seed in grad_guard]
        if step == "D":
            jobs += [(conf, seed, True) for conf, seed in grad_guard if conf in cont_confs]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    guard_patch = D_GUARD_PATCH if step == "D" else GUARD_PATCH
    for conf, seed, cont in jobs:
        name, plain = render(template, step, conf, seed, cont)
        diff = [i for i, (x, y) in enumerate(zip(template.splitlines(), plain.splitlines())) if x != y]
        if len(plain.splitlines()) != len(template.splitlines()) or len(diff) > 7:
            raise ValueError(f"{name}: khác khuôn ngoài khối CẤU HÌNH ({len(diff)} dòng)")
        text, added = plain, 0
        if step == "D":
            text = apply_d_patch(text, name)
            if strip_d_patch(text) != plain:
                raise ValueError(f"{name}: gỡ D_PATCH không trả về tệp khuôn thường")
            added += len(text.splitlines()) - len(plain.splitlines())
        if grad_guard is not None:
            before_guard = text
            text = apply_grad_guard(text, name, guard_patch)
            if strip_grad_guard(text, guard_patch) != before_guard:
                raise ValueError(f"{name}: gỡ chốt chặn không trả về tệp thường")
            added += len(text.splitlines()) - len(before_guard.splitlines())
        if len(name) > MAX_NAME:
            raise ValueError(f"{name}: {len(name)} ký tự, tên notebook Kaggle tối đa {MAX_NAME}")
        compile(text, name, "exec")                                  # lỗi cú pháp => dừng
        path = out_dir / f"{name}.py"
        path.write_bytes(text.encode("utf-8"))
        rows.append((name, len(name), len(diff) + added, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]))
    for name, n, nd, sha in rows:
        print(f"{name:<62} {n:>3} ký tự  {nd} dòng khác khuôn  sha256 {sha}")
    print(f"=> {len(rows)} tệp vào {out_dir} (tên dài nhất {max(r[1] for r in rows)} ký tự); "
          f"khuôn sha256 {hashlib.sha256(template.encode('utf-8')).hexdigest()[:12]}")


# %% Phần 3 - Dòng lệnh
def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")                         # console Windows mặc định cp1252 => lỗi khi in tiếng Việt
    ap = argparse.ArgumentParser(description="Sinh notebook huấn luyện bước C hoặc bước D từ khuôn")
    ap.add_argument("step", choices=["c1", "c2", "d"])
    ap.add_argument("configs", nargs="*", help="c2/d: cấu hình dạng mamba_L2_x1_lr1e-3 (d mặc định đủ 3 cấu hình)")
    ap.add_argument("--cont", nargs="*", default=None, help="c2/d: cấu hình cần notebook phiên 2 (_Cont)")
    ap.add_argument("--grad-guard", nargs="*", default=None,
                    help="c2/d: chỉ sinh cặp cấu_hình:hạt_giống, có chốt chặn gradient")
    ap.add_argument("--out", type=Path, default=None, help="thư mục ghi (mặc định thư mục bước trong Workspace)")
    args = ap.parse_args(argv)
    step = args.step.upper()
    if step == "C1":
        if args.configs or args.cont is not None:
            ap.error("c1 không nhận danh sách cấu hình (sinh đủ lưới)")
        confs = c1_configs()
        cont = confs                                             # C1 vòng 2: cấu hình nào cũng có thể được giữ
        seeds = (111,)
    elif step == "D":
        confs = args.configs or list(D_CONFIGS)
        if not set(confs) <= set(D_CONFIGS):
            ap.error(f"d chỉ nhận các cấu hình trong {D_CONFIGS}")
        cont = confs if args.cont is None else args.cont         # D mặc định sinh sẵn _Cont cho mọi cấu hình được chọn
        if not set(cont) <= set(confs):
            ap.error("--cont chỉ nhận cấu hình có trong danh sách d")
        seeds = C2_SEEDS
    else:
        if not args.configs:
            ap.error("c2 cần ít nhất một cấu hình")
        confs, cont, seeds = args.configs, args.cont or [], C2_SEEDS
        if not set(cont) <= set(confs):
            ap.error("--cont chỉ nhận cấu hình có trong danh sách c2")
    for c in confs + cont:
        parse(c)
    grad_guard = None
    if args.grad_guard:
        if step not in ("C2", "D"):
            ap.error("--grad-guard chỉ dùng cho c2 hoặc d")
        grad_guard = [(c, int(s)) for c, s in (x.split(":") for x in args.grad_guard)]
        if not all(c in confs and s in seeds for c, s in grad_guard):
            ap.error(f"--grad-guard chỉ nhận cấu hình trong danh sách {step.lower()} và hạt giống {seeds}")
    build(step, confs, seeds, cont, args.out or OUT_DIRS[step], grad_guard)


if __name__ == "__main__":
    sys.exit(main())
