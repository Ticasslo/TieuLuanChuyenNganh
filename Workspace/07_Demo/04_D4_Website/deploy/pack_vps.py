# Gom dữ liệu chạy website lên VPS vào một thư mục: gói website, nền bản đồ, địa hình (Mapterhorn), dữ liệu thô LamaH chỉ 451 lưu vực của gói
# (thuộc tính lưu vực và trạm giữ đủ, chuỗi ngày chỉ lấy ID có trong catalog.json) và tệp SHA256SUMS
# để kiểm trên VPS bằng "sha256sum -c SHA256SUMS".
# Chạy: python pack_vps.py --bundle <gói website> --raw <LamaH_extended> --pmtiles <cee_z11.pmtiles>
#       --terrain <terrain_z11.pmtiles> --out <thư mục đích>

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# Thư mục con của LamaH_extended mà bộ dự báo đọc: thuộc tính đủ, chuỗi ngày theo ID
ATTRIBUTE_DIRS = ("A_basins_total_upstrm/1_attributes", "D_gauges/1_attributes")
DAILY_DIRS = ("A_basins_total_upstrm/2_timeseries/daily", "D_gauges/2_timeseries/daily")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Bỏ qua tệp đã chép trước đó cùng kích thước => chạy lại nhanh khi chỉ đổi một phần
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Đóng gói dữ liệu website để chép lên VPS.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--pmtiles", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    catalog = json.loads((args.bundle / "catalog.json").read_text(encoding="utf-8"))
    ids = [str(i) for i in catalog["ids"]]
    if len(ids) != 451 or len(set(ids)) != 451:
        raise SystemExit(f"catalog.json phải có 451 ID khác nhau, đang có {len(ids)}.")
    for sub in ATTRIBUTE_DIRS + DAILY_DIRS:
        if not (args.raw / sub).is_dir():
            raise SystemExit(f"Thiếu thư mục dữ liệu thô: {args.raw / sub}")

    out = args.out.resolve()
    if out == args.bundle.resolve() or out == args.raw.resolve():
        raise SystemExit("Thư mục đích phải khác thư mục nguồn.")
    plan = [(p, out / "bundle" / p.relative_to(args.bundle)) for p in sorted(args.bundle.rglob("*")) if p.is_file()]
    for sub in ATTRIBUTE_DIRS:
        plan += [(p, out / "raw/LamaH_extended" / sub / p.relative_to(args.raw / sub))
                 for p in sorted((args.raw / sub).rglob("*")) if p.is_file()]
    for sub in DAILY_DIRS:
        for bid in ids:
            src = args.raw / sub / f"ID_{bid}.csv"
            if not src.is_file():
                raise SystemExit(f"Thiếu chuỗi ngày của lưu vực {bid}: {src}")
            plan.append((src, out / "raw/LamaH_extended" / sub / src.name))
    plan.append((args.pmtiles, out / "basemap.pmtiles"))
    plan.append((args.terrain, out / "terrain.pmtiles"))

    for i, (src, dst) in enumerate(plan, 1):
        copy(src, dst)
        if i % 500 == 0 or i == len(plan):
            print(f"Đã chép {i}/{len(plan)} tệp")

    # Đúng định dạng của coreutils sha256sum: "<hash>  <đường dẫn tương đối>", dấu "/"
    lines = [f"{sha256(dst)}  {dst.relative_to(out).as_posix()}" for _, dst in plan]
    # newline="\n" để Windows không ghi CRLF (sha256sum trên Linux đọc "\r" là một phần tên tệp)
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    total = sum(dst.stat().st_size for _, dst in plan)
    print(f"Xong: {len(plan)} tệp, {total / 2**30:.2f} GiB tại {out}")


if __name__ == "__main__":
    main()
