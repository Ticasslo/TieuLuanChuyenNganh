# Dựng gói dữ liệu cho website: chép gói D3 đã kiểm (D3_CHD_CPU_V13) rồi thêm kết quả XAI bước G
# (G_demo_basin.npz, GH_demo_basin.npz và bảng tóm tắt) bằng đúng hàm package_xai của prepare.py D3,
# kiểm sha256 theo manifest.csv của Output G. Không tính lại dự báo nào.
# Chạy: python build_bundle.py --d3 <gói D3> --g-c2 <g_c2/g_output> --g-h <g_h/g_output> --output <thư mục gói mới>

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "03_D3_LamaHCEExt_Demo"))
import prepare  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finder(roots):
    # Tìm tệp theo tên trong các thư mục Output G; trùng tên ở nhiều chỗ => dừng
    def unique_file(_source, name, required=True):
        hits = [p for root in roots for p in sorted(root.rglob(name))]
        if len(hits) > 1:
            raise ValueError(f"Tìm thấy {len(hits)} tệp {name}: {hits}")
        if not hits and required:
            raise FileNotFoundError(f"Thiếu {name} trong {roots}")
        return hits[0] if hits else None
    return unique_file


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3", required=True, type=Path)
    parser.add_argument("--g-c2", required=True, type=Path)
    parser.add_argument("--g-h", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    d3, output = args.d3.resolve(), args.output.resolve()
    if output == d3 or output.is_relative_to(d3):
        raise ValueError("Thư mục gói mới phải khác gói D3.")
    if output.exists():
        raise FileExistsError(f"{output} đã có, xóa trước khi dựng lại.")

    shutil.copytree(d3, output)
    catalog_path = output / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    ids = list(catalog["ids"])

    audit = {"sha256": sha256, "unique_file": finder([args.g_c2.resolve(), args.g_h.resolve()])}
    meta = prepare.package_xai(args.g_c2.resolve(), output, ids, {"basin_id": str}, audit)
    if not (meta["c2_basin"] and meta["h_basin"]):
        raise ValueError(f"Thiếu NPZ XAI theo lưu vực: {meta}")

    catalog["xai"] = meta
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    added = sorted(p.name for p in output.iterdir() if not (d3 / p.name).exists())
    print(f"Gói website: {output}")
    print(f"Thêm {len(added)} tệp XAI: {', '.join(added)}")
    print(f"Run demo C2: {meta['c2_demo_runs']}")
    print(f"Run demo H: {meta['h_demo_runs']}")


if __name__ == "__main__":
    main()
