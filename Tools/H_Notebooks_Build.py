# Sinh các notebook huấn luyện bước H từ khuôn Tools/H_Notebook_Template.py (ba cách past, hybrid, perfect) hoặc
# Tools/H_GEFS_Notebook_Template.py (cách gefs): mỗi tệp là khuôn chép nguyên, chỉ thay khối CẤU HÌNH ở Phần 1 (NOTEBOOK,
# CORE, MODE, RUN_SEED). Bước H: Document/01_Plan/02_Pipeline.md Mục 4.4 (2 lõi x 4 cách x 3 hạt giống = 24 run).
# Chạy (từ gốc repo):
#   python Tools/H_Notebooks_Build.py [--gefs] [--cont lstm_hybrid_Seed111 ...] [--out THƯ_MỤC]
#       18 notebook 14_H_Train_<cấu hình>_<cách>_Seed<N> (lstm_L2_x1_lr1e-3, mamba_L2_x1_lr1e-3 x past, hybrid, perfect
#       x 111, 222, 333), mặc định vào Workspace/05_Training/14_H_Train/; --gefs thay bằng 6 notebook cách gefs từ khuôn
#       GEFS. --cont <lõi>_<cách>_Seed<N> sinh thêm notebook phiên 2 (_Cont) cho run không xong trong một phiên (cách
#       gefs cần --gefs).
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
TEMPLATE = ROOT / "Tools" / "H_Notebook_Template.py"
TEMPLATE_GEFS = ROOT / "Tools" / "H_GEFS_Notebook_Template.py"
OUT_DIR = ROOT / "Workspace" / "05_Training" / "14_H_Train"
PREFIX = "14_H_Train"                        # NN tiếp sau 13_C2_LamaHCEExt_Compare
MAX_NAME = 50                                # tên notebook Kaggle tối đa 50 ký tự
CONFS = {"lstm": "lstm_L2_x1_lr1e-3", "mamba": "mamba_L2_x1_lr1e-3"}   # Pipeline Mục 4.4
MODES = ("hybrid", "past", "perfect")        # thứ tự chạy: (b) => (a) => (c)
MODES_GEFS = ("gefs",)                       # cách (d), khuôn GEFS
SEEDS = (111, 222, 333)
JOB_RE = re.compile(r"^(lstm|mamba)_(past|hybrid|perfect|gefs)_Seed(111|222|333)$")
BLOCK_START = "# ---- CẤU HÌNH (Tools/H_Notebooks_Build.py ghi lại khối này) ----\n"
BLOCK_END = "# ---- HẾT CẤU HÌNH ----\n"


def notebook_name(core, mode, seed, cont=False):
    return f"{PREFIX}_{CONFS[core]}_{mode}_Seed{seed}" + ("_Cont" if cont else "")


# %% Phần 2 - Thay khối CẤU HÌNH, kiểm tệp sinh ra
def render(template, core, mode, seed, cont):
    # Khuôn với khối CẤU HÌNH mới; giữ nguyên chú thích cuối mỗi dòng của khối.
    name = notebook_name(core, mode, seed, cont)
    values = {"NOTEBOOK": f'"{name}"', "CORE": f'"{core}"', "MODE": f'"{mode}"', "RUN_SEED": str(seed)}
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


def build(jobs, out_dir, template_path=TEMPLATE):
    template = template_path.read_text(encoding="utf-8")
    if "\r" in template:
        raise ValueError("Khuôn có ký tự CR - lưu khuôn với xuống dòng LF")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for core, mode, seed, cont in jobs:
        name, text = render(template, core, mode, seed, cont)
        if len(name) > MAX_NAME:
            raise ValueError(f"{name}: {len(name)} ký tự, tên notebook Kaggle tối đa {MAX_NAME}")
        compile(text, name, "exec")                                  # lỗi cú pháp => dừng
        diff = [i for i, (x, y) in enumerate(zip(template.splitlines(), text.splitlines())) if x != y]
        if len(text.splitlines()) != len(template.splitlines()) or len(diff) > 4:
            raise ValueError(f"{name}: khác khuôn ngoài khối CẤU HÌNH ({len(diff)} dòng)")
        (out_dir / f"{name}.py").write_bytes(text.encode("utf-8"))
        rows.append((name, len(name), len(diff), hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]))
    for name, n, nd, sha in rows:
        print(f"{name:<52} {n:>3} ký tự  {nd} dòng khác khuôn  sha256 {sha}")
    print(f"=> {len(rows)} tệp vào {out_dir} (tên dài nhất {max(r[1] for r in rows)} ký tự); "
          f"khuôn sha256 {hashlib.sha256(template.encode('utf-8')).hexdigest()[:12]}")


# %% Phần 3 - Dòng lệnh
def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")                         # console Windows mặc định cp1252 => lỗi khi in tiếng Việt
    ap = argparse.ArgumentParser(description="Sinh 18 notebook huấn luyện bước H (hoặc 6 notebook cách gefs) từ khuôn")
    ap.add_argument("--gefs", action="store_true", help="6 notebook cách gefs từ Tools/H_GEFS_Notebook_Template.py")
    ap.add_argument("--cont", nargs="*", default=[], help="run cần notebook phiên 2, dạng lstm_hybrid_Seed111")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="thư mục ghi (mặc định Workspace/05_Training/14_H_Train)")
    args = ap.parse_args(argv)
    modes = MODES_GEFS if args.gefs else MODES
    jobs = [(core, mode, seed, False) for mode in modes for core in CONFS for seed in SEEDS]
    for job in args.cont:
        m = JOB_RE.match(job)
        if not m or m.group(2) not in modes:
            ap.error(f"--cont {job}: cần dạng <lstm|mamba>_<{'|'.join(modes)}>_Seed<111|222|333>"
                     + ("" if args.gefs else " (cách gefs cần --gefs)"))
        jobs.append((m.group(1), m.group(2), int(m.group(3)), True))
    build(jobs, args.out, TEMPLATE_GEFS if args.gefs else TEMPLATE)


if __name__ == "__main__":
    sys.exit(main())
