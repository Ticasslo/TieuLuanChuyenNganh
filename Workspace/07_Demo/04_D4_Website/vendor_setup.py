# Chép thư viện giao diện đã tải (phiên bản cố định) từ downloads/vendor_src vào static/vendor
# và ghi static/vendor/VENDOR.json (phiên bản, giấy phép, sha256 từng tệp) để website chạy không cần CDN.
# Chạy: python -I vendor_setup.py [--source downloads/vendor_src]

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Gói npm đã tải, mỗi gói: tên, phiên bản, giấy phép, tệp nguồn => đích trong static/vendor
PACKAGES = [
    {"name": "maplibre-gl", "version": "6.17.0", "license": "BSD-3-Clause",
     "files": {"maplibre-gl-6.17.0/package/dist/maplibre-gl.mjs": "maplibre/maplibre-gl.mjs",
               "maplibre-gl-6.17.0/package/dist/maplibre-gl-shared.mjs": "maplibre/maplibre-gl-shared.mjs",
               "maplibre-gl-6.17.0/package/dist/maplibre-gl-worker.mjs": "maplibre/maplibre-gl-worker.mjs",
               "maplibre-gl-6.17.0/package/dist/maplibre-gl.css": "maplibre/maplibre-gl.css",
               "maplibre-gl-6.17.0/package/LICENSE.txt": "maplibre/LICENSE.txt"}},
    {"name": "pmtiles", "version": "4.5.0", "license": "BSD-3-Clause",
     "files": {"pmtiles-4.5.0/package/dist/pmtiles.js": "pmtiles/pmtiles.js"}},
    {"name": "@protomaps/basemaps", "version": "5.7.2", "license": "BSD-3-Clause",
     "files": {"basemaps-5.7.2/package/dist/basemaps.js": "basemaps/basemaps.js"}},
    {"name": "@fontsource/be-vietnam-pro", "version": "5", "license": "OFL-1.1",
     "files": {"bvp/package/LICENSE": "fonts/LICENSE"}},
]
FONT_SUBSETS = ("vietnamese", "latin-ext", "latin")
FONT_WEIGHTS = (400, 500, 600, 700)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target, record):
    if not source.is_file():
        raise FileNotFoundError(f"Thiếu tệp nguồn: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    record[target.relative_to(OUT).as_posix()] = sha256(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(HERE / "downloads" / "vendor_src"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    source = Path(args.source).resolve()
    if OUT.exists():
        shutil.rmtree(OUT)
    files = {}

    for package in PACKAGES:
        for src, dst in package["files"].items():
            copy(source / src, OUT / dst, files)

    # Phông Be Vietnam Pro: 3 bộ ký tự x 4 độ đậm, tự ghi fonts.css có unicode-range để trình duyệt chỉ tải phần cần
    bvp = source / "bvp" / "package"
    ranges = json.loads((bvp / "unicode.json").read_text(encoding="utf-8"))
    rules = []
    for subset in FONT_SUBSETS:
        for weight in FONT_WEIGHTS:
            name = f"be-vietnam-pro-{subset}-{weight}-normal.woff2"
            copy(bvp / "files" / name, OUT / "fonts" / name, files)
            rules.append("@font-face {\n  font-family: 'Be Vietnam Pro';\n  font-style: normal;\n"
                         f"  font-display: swap;\n  font-weight: {weight};\n"
                         f"  src: url(./{name}) format('woff2');\n  unicode-range: {ranges[subset]};\n}}\n")
    css = OUT / "fonts" / "fonts.css"
    css.write_text("".join(rules), encoding="utf-8")
    files[css.relative_to(OUT).as_posix()] = sha256(css)

    # Glyph và sprite của nền bản đồ Protomaps (giấy phép OFL cho phông Noto Sans)
    assets = source / "basemaps-assets"
    for path in sorted(assets.rglob("*")):
        if path.is_file():
            copy(path, OUT / "basemap" / path.relative_to(assets), files)

    record = {"packages": [{k: p[k] for k in ("name", "version", "license")} for p in PACKAGES],
              "basemap_assets": "protomaps/basemaps-assets (fonts Noto Sans OFL-1.1, sprites v4)",
              "files": files}
    (OUT / "VENDOR.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum((OUT / f).stat().st_size for f in files)
    print(f"Đã chép {len(files)} tệp, {total / 1e6:.1f} MB vào {OUT}")


OUT = HERE / "static" / "vendor"

if __name__ == "__main__":
    main()
