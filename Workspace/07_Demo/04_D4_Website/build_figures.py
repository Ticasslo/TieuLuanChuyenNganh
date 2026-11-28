# Chép hình quá trình huấn luyện (PNG trong Workspace/05_Training/*/Figures) vào gói website dưới dạng WebP thu nhỏ
# cho mục "Quá trình huấn luyện" của trang Kết quả. Danh sách hình trùng FIGURES trong static/js/app.js.
# Chạy: python build_figures.py --bundle <gói website>

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1] / "05_Training"
MAX_WIDTH = 1800
FIGURES = {
    "B1_Steps": "05_B1_LamaHCEExt_Compare",
    "B1_Curves": "05_B1_LamaHCEExt_Compare",
    "B3_Seeds": "09_B3_LamaHCEExt_Compare",
    "C2_Bstar_Validation": "13_C2_LamaHCEExt_Compare",
    "C2_Curves_LSTM_GRU_Transformer": "13_C2_LamaHCEExt_Compare",
    "C2_Curves_S4D_Mamba": "13_C2_LamaHCEExt_Compare",
    "H_Curves": "15_H_LamaHCEExt_Compare",
    "H_NSE_by_lead_test": "15_H_LamaHCEExt_Compare",
    "D_Curves_365_vs_730": "17_D_LamaHCEExt_Compare",
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    if not (args.bundle / "catalog.json").is_file():
        raise FileNotFoundError(f"{args.bundle} không phải gói website (thiếu catalog.json).")
    out = args.bundle / "figures"
    out.mkdir(exist_ok=True)
    meta = {}
    for name, folder in FIGURES.items():
        src = TRAINING / folder / "Figures" / f"{name}.png"
        if not src.is_file():
            raise FileNotFoundError(f"Thiếu hình {src}")
        with Image.open(src) as img:
            img = img.convert("RGB")
            if img.width > MAX_WIDTH:
                img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)), Image.Resampling.LANCZOS)
            dst = out / f"{name}.webp"
            img.save(dst, "WEBP", quality=86, method=6)
            meta[name] = {"width": img.width, "height": img.height, "source": f"{folder}/Figures/{name}.png",
                          "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
        print(f"{name}: {meta[name]['width']}x{meta[name]['height']}, {dst.stat().st_size // 1024} KB")
    (out / "figures.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(meta)} hình => {out}")


if __name__ == "__main__":
    main()
