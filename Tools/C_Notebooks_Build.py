# Sinh các notebook huấn luyện bước C từ khuôn Tools/C_Notebook_Template.py: mỗi tệp là khuôn chép nguyên, chỉ thay khối
# CẤU HÌNH ở Phần 1 (NOTEBOOK, STEP, CORE, DEPTH, LEVEL, LR, RUN_SEED). Lưới C1: Document/01_Plan/02_Pipeline.md Mục 5.2.
# Chạy (từ gốc repo):
#   python Tools/C_Notebooks_Build.py c1 [--out THƯ_MỤC]
#       39 notebook C1 (5 lõi x 2 độ sâu x 2 mức x 2 learning rate, bỏ lstm_L1_x1_lr1e-3 = B* hạt giống 111) và notebook
#       phiên 2 (_Cont) cho các cấu hình vượt 12 giờ, mặc định vào Workspace/05_Training/06_C1_LamaHCEExt_Tune/.
#   python Tools/C_Notebooks_Build.py c2 mamba_L4_x1_lr1e-3 [...] [--cont mamba_L4_x3_lr1e-3 ...] [--out THƯ_MỤC]
#       3 hạt giống (111, 222, 333) cho mỗi cấu hình C2, _Cont cho cấu hình ghi ở --cont, mặc định vào
#       Workspace/05_Training/08_C2_Train/. Tên C2 bỏ "LamaHCEExt" để vừa 50 ký tự (giới hạn tên notebook Kaggle).
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
OUT_DIRS = {"C1": ROOT / "Workspace" / "05_Training" / "06_C1_LamaHCEExt_Tune",
            "C2": ROOT / "Workspace" / "05_Training" / "08_C2_Train"}
PREFIX = {"C1": "06_C1_LamaHCEExt_Tune", "C2": "08_C2_Train"}  # số NN đặt lại sau B2, B3 (thứ tự chạy trong 05_Training)
MAX_NAME = 50                                # tên notebook Kaggle tối đa 50 ký tự
CORE_DEPTHS = {"lstm": (1, 2), "gru": (1, 2), "transformer": (1, 2), "s4d": (2, 4), "mamba": (2, 4)}
LEVELS = (1, 3)
LRS = {"lr1e-3": 1e-3, "lr5e-4": 5e-4}
C2_SEEDS = (111, 222, 333)
B_STAR = "lstm_L1_x1_lr1e-3"                 # cấu hình LSTM chuẩn của C1 = B* hạt giống 111, không chạy lại
# Cấu hình một lần C1 vượt 12 giờ một phiên Kaggle (05_0_C_Configs_Check Version 1, Mục 5 báo cáo) => thêm _Cont
C1_LONG = ("lstm_L2_x3", "gru_L2_x3", "s4d_L4_x3", "mamba_L4_x3")
CONF_RE = re.compile(r"^(lstm|gru|transformer|s4d|mamba)_L(\d)_x([13])_(lr1e-3|lr5e-4)$")
BLOCK_START = "# ---- CẤU HÌNH (Tools/C_Notebooks_Build.py ghi lại khối này) ----\n"
BLOCK_END = "# ---- HẾT CẤU HÌNH ----\n"


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
    name = f"{PREFIX[step]}_{conf}" + (f"_Seed{seed}" if step == "C2" else "") + ("_Cont" if cont else "")
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


def build(step, confs, seeds, cont_confs, out_dir):
    template = TEMPLATE.read_text(encoding="utf-8")
    if "\r" in template:
        raise ValueError("Khuôn có ký tự CR - lưu khuôn với xuống dòng LF")
    jobs = [(conf, seed, False) for conf in confs for seed in seeds]
    jobs += [(conf, seed, True) for conf in cont_confs for seed in seeds]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for conf, seed, cont in jobs:
        name, text = render(template, step, conf, seed, cont)
        if len(name) > MAX_NAME:
            raise ValueError(f"{name}: {len(name)} ký tự, tên notebook Kaggle tối đa {MAX_NAME}")
        compile(text, name, "exec")                                  # lỗi cú pháp => dừng
        diff = [i for i, (x, y) in enumerate(zip(template.splitlines(), text.splitlines())) if x != y]
        if len(text.splitlines()) != len(template.splitlines()) or len(diff) > 7:
            raise ValueError(f"{name}: khác khuôn ngoài khối CẤU HÌNH ({len(diff)} dòng)")
        path = out_dir / f"{name}.py"
        path.write_bytes(text.encode("utf-8"))
        rows.append((name, len(name), len(diff), hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]))
    for name, n, nd, sha in rows:
        print(f"{name:<62} {n:>3} ký tự  {nd} dòng khác khuôn  sha256 {sha}")
    print(f"=> {len(rows)} tệp vào {out_dir} (tên dài nhất {max(r[1] for r in rows)} ký tự); "
          f"khuôn sha256 {hashlib.sha256(template.encode('utf-8')).hexdigest()[:12]}")


# %% Phần 3 - Dòng lệnh
def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")                         # console Windows mặc định cp1252 => lỗi khi in tiếng Việt
    ap = argparse.ArgumentParser(description="Sinh notebook huấn luyện bước C từ khuôn")
    ap.add_argument("step", choices=["c1", "c2"])
    ap.add_argument("configs", nargs="*", help="c2: cấu hình dạng mamba_L4_x1_lr1e-3")
    ap.add_argument("--cont", nargs="*", default=None, help="c2: cấu hình cần notebook phiên 2 (_Cont)")
    ap.add_argument("--out", type=Path, default=None, help="thư mục ghi (mặc định thư mục bước trong Workspace)")
    args = ap.parse_args(argv)
    step = args.step.upper()
    if step == "C1":
        if args.configs or args.cont is not None:
            ap.error("c1 không nhận danh sách cấu hình (sinh đủ lưới)")
        confs = c1_configs()
        cont = [c for c in confs if c.rsplit("_", 1)[0] in C1_LONG]
        seeds = (111,)
    else:
        if not args.configs:
            ap.error("c2 cần ít nhất một cấu hình")
        confs, cont, seeds = args.configs, args.cont or [], C2_SEEDS
        if not set(cont) <= set(confs):
            ap.error("--cont chỉ nhận cấu hình có trong danh sách c2")
    for c in confs + cont:
        parse(c)
    build(step, confs, seeds, cont, args.out or OUT_DIRS[step])


if __name__ == "__main__":
    sys.exit(main())
