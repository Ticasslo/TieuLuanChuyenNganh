# Thêm vào gói website dữ liệu cho các lớp bản đồ khí tượng, đất, thực vật và đặc điểm lưu vực, đọc từ
# Extended LamaH-CE thô (Zenodo 17119635); mọi biến đều là đầu vào của mô hình C2 (config.yml của run C2):
# - env/<khóa>.npy: ma trận float32 [451 lưu vực, ngày kỳ test] cùng thứ tự với catalog.json, giá trị trung bình
#   cả lưu vực từ A_basins_total_upstrm/2_timeseries/daily/ID_<id>.csv (ô trống => NaN);
# - env.json: danh sách lớp, đơn vị (bảng A1, A2 của BiasCast), ngưỡng màu theo phân vị, giá trị thuộc tính tĩnh
#   từ Catchment_attributes.csv; đường tuyết ước lượng theo ngày (snowline) cho lớp tuyết trên địa hình 3D;
# - basins_own.geojson: phần riêng của từng lưu vực (lưu vực toàn phần trừ các lưu vực trạm thượng nguồn nằm trong nó,
#   tính ở EPSG:3035 trước khi làm gọn) => các mảnh không chồng nhau, mỗi chỗ tô một màu của lưu vực nhỏ nhất chứa nó.
# - rivers.geojson (khi có --rivers): mạng sông HydroRIVERS v1.0 (Lehner & Grill 2013, giấy phép HydroSHEDS v1) trong
#   vùng nghiên cứu, mỗi đoạn gắn trạm đo hạ lưu gần nhất (mảnh phần riêng chứa điểm giữa đoạn) để tô theo mức lưu lượng
#   của trạm đó; đoạn vẽ theo chiều nước chảy (điểm cuối nối điểm đầu đoạn NEXT_DOWN, kiểm khi dựng).
# Kiểm: đọc lại 40 ô ngẫu nhiên bằng mô-đun csv (không qua pandas), lệch thì dừng; các mảnh phủ đúng hợp các lưu vực
# và không chồng nhau (sai lệch diện tích quá 0,5% thì dừng).
# Chạy: python build_map_layers.py --raw <LamaH_extended> --bundle <gói website> [--rivers <HydroRIVERS_v10_eu.shp>]

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from pyproj import Transformer
from shapely import make_valid, union_all
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform
from shapely.strtree import STRtree

DAILY = "A_basins_total_upstrm/2_timeseries/daily"
ATTRS = "A_basins_total_upstrm/1_attributes/Catchment_attributes.csv"
BASINS_SHP = "A_basins_total_upstrm/3_shapefiles/Basins_A.shp"
SIMPLIFY_M = 100
# Mảnh nhỏ hơn mức này (m²) sau khi trừ là vệt do ranh giới số hóa lệch nhau, bỏ đi
SLIVER_M2 = 5e4
# Lưu vực có lượng nước trong tuyết từ mức này (mm) tính là "có tuyết" khi ước lượng đường tuyết
SNOW_MM = 5.0
# Sông: chỉ giữ đoạn có diện tích thượng nguồn từ mức này (km²); làm gọn đường theo độ (khoảng 30 m)
RIVER_UPLAND_KM2 = 20
# Sông lớn ngoài lưu vực có trạm (Inn, Salzach, Danube, Rhine...) vẫn vẽ cho thấy mạng sông, không gắn trạm (g rỗng)
RIVER_FREE_KM2 = 300
RIVER_SIMPLIFY_DEG = 0.0003
STUDY_BBOX = (7.5, 45.8, 17.8, 50.9)
# Bảng màu một sắc độ (OKLCH, độ sáng giảm đều) và bảng hai phía quanh 0 °C (xám ở giữa)
RAMPS = {
    "blue": ["#e1f0ff", "#b4cbea", "#88a8d1", "#5d85b8", "#32639f", "#004085"],
    "teal": ["#d8f5f6", "#a9d2d4", "#79b1b4", "#489094", "#007075", "#005157"],
    "orange": ["#ffe8da", "#eec4ac", "#d89f80", "#c27b53", "#ab5821", "#923100"],
    "green": ["#dff6de", "#b4d5b4", "#8ab58a", "#619562", "#37763b", "#005810"],
    "brown": ["#f8ece1", "#d6c4b4", "#b49d88", "#92785e", "#725436", "#53320b"],
    "violet": ["#eeecff", "#c8c4e8", "#a49ecf", "#8279b5", "#61559b", "#443081"],
    "slate": ["#eef1f4", "#c9d0d8", "#a4afbc", "#808ea0", "#5e6e84", "#3e4f68"],
    "diverging": ["#154d8c", "#7092bd", "#cbd9ec", "#f0efec", "#ebd0ce", "#c37f79", "#932b2a"],
}
# Lớp theo ngày: khóa => cột, đơn vị, bảng màu, cách đặt ngưỡng
DAILY_LAYERS = [
    dict(key="snow", column="ERA5L_swe", unit="mm", ramp="violet", scale="positive", source="ERA5-Land"),
    dict(key="rain", column="EOBS_rr", unit="mm/day", ramp="blue", scale="positive", source="E-OBS"),
    dict(key="temp", column="EOBS_tg", unit="°C", ramp="diverging", scale="diverging", source="E-OBS"),
    dict(key="soil", column="ERA5L_volsw_123", unit="m³/m³", ramp="teal", scale="range", source="ERA5-Land"),
    dict(key="eta", column="GLEAM_ETA", unit="mm/day", ramp="orange", scale="range", source="GLEAM"),
    dict(key="etp", column="GLEAM_ETP", unit="mm/day", ramp="orange", scale="range", source="GLEAM"),
    dict(key="lai", column="ERA5L_lai_high_veg", unit="m²/m²", ramp="green", scale="range", source="ERA5-Land"),
    dict(key="wind", column="EOBS_fg", unit="m/s", ramp="slate", scale="range", source="E-OBS"),
]
# Hướng gió vẽ mũi tên: thành phần đông, bắc của gió 10 m (ERA5-Land)
WIND_UV = ("ERA5L_10m_wind_u", "ERA5L_10m_wind_v")
STATIC_LAYERS = [
    dict(key="elev", column="elev_mean", unit="m", ramp="brown", scale="range"),
    dict(key="slope", column="slope_mean", unit="m/km", ramp="brown", scale="range"),
    dict(key="forest", column="forest_fra", unit="%", ramp="green", scale="fraction"),
    dict(key="snowfrac", column="frac_snow", unit="%", ramp="violet", scale="fraction"),
    dict(key="pmean", column="p_mean", unit="mm/day", ramp="blue", scale="range"),
]


def nice(v):
    # Làm tròn ngưỡng cho chú giải dễ đọc (2 chữ số có nghĩa)
    if not np.isfinite(v) or v == 0:
        return 0.0
    return float(f"{v:.2g}")


def stops(values, scale, n):
    # Ngưỡng màu: phân vị trên mọi ô (lưu vực x ngày) => màu trải đều theo tần suất, ít ô cực trị không làm nhạt cả bản đồ
    v = values[np.isfinite(values)]
    if scale == "positive":
        # Mưa, tuyết: 0 là "không có" (màu nhạt nhất), các bậc sau theo phân vị của ô dương
        pos = v[v > 0.1]
        q = np.quantile(pos, np.linspace(0.25, 0.99, n - 1))
        out = [0.0] + [nice(x) for x in q]
    elif scale == "diverging":
        # Nhiệt độ: 0 °C ở giữa, hai nhánh tới phân vị 1% và 99%
        lo, hi = np.quantile(v, [0.01, 0.99])
        half = n // 2
        out = [nice(lo * (1 - k / half)) for k in range(half)] + [0.0] + [nice(hi * k / half) for k in range(1, half + 1)]
    elif scale == "fraction":
        lo, hi = np.quantile(v, [0.02, 0.98])
        out = [round(float(x), 2) for x in np.linspace(lo, hi, n)]
    else:
        lo, hi = np.quantile(v, [0.02, 0.98])
        out = [nice(x) for x in np.linspace(lo, hi, n)]
    # Ngưỡng phải tăng ngặt (MapLibre interpolate)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            raise ValueError(f"Ngưỡng màu không tăng: {out}")
    return out


def own_parts(raw, ids):
    # Phần riêng của mỗi lưu vực: trừ các lưu vực nhỏ hơn nằm phần lớn (trên nửa diện tích) trong nó
    geoms = {}
    with shapefile.Reader(str(raw / BASINS_SHP), encoding="latin1") as reader:
        for item in reader.iterShapeRecords():
            bid = str(int(item.record["ID"]))
            if bid in ids:
                geoms[bid] = make_valid(shape(item.shape.__geo_interface__))
    if set(geoms) != set(ids):
        raise ValueError("Basins_A thiếu ranh giới của một số lưu vực.")
    order = list(ids)
    shapes = [geoms[b] for b in order]
    areas = np.array([g.area for g in shapes])
    tree = STRtree(shapes)
    parts, children = {}, {}
    for i, g in enumerate(shapes):
        inner = [j for j in tree.query(g) if j != i and areas[j] < areas[i]
                 and g.intersection(shapes[j]).area > 0.5 * areas[j]]
        own = g.difference(union_all([shapes[j] for j in inner])) if inner else g
        polys = [q for q in getattr(own, "geoms", [own]) if isinstance(q, Polygon) and q.area >= SLIVER_M2]
        parts[order[i]] = MultiPolygon(polys) if len(polys) > 1 else (polys[0] if polys else None)
        children[order[i]] = len(inner)
    # Kiểm: các mảnh phủ hợp các lưu vực và không chồng nhau
    kept = [g for g in parts.values() if g is not None]
    whole, mosaic, summed = union_all(shapes).area, union_all(kept).area, sum(g.area for g in kept)
    gap, overlap = (whole - mosaic) / whole, (summed - mosaic) / whole
    if gap > 0.005 or overlap > 0.005:
        raise ValueError(f"Mảnh lưu vực không khớp: hở {gap:.3%}, chồng {overlap:.3%}")
    to_wgs84 = Transformer.from_crs(3035, 4326, always_xy=True).transform
    features = []
    for bid in order:
        g = parts[bid]
        if g is None:
            continue
        g = transform(to_wgs84, g.simplify(SIMPLIFY_M, preserve_topology=True))
        features.append(dict(type="Feature", properties=dict(id=bid, own_km2=round(parts[bid].area / 1e6, 1)),
                             geometry=json.loads(json.dumps(mapping(g)), parse_float=lambda v: round(float(v), 5))))
    nested = sum(1 for n in children.values() if n)
    print(f"basins_own: {len(features)} mảnh ({len(order) - len(features)} lưu vực không còn phần riêng), "
          f"{nested} lưu vực có lưu vực con; hở {gap:.3%}, chồng {overlap:.3%}")
    return features, parts


def rivers(path, parts):
    # Đoạn sông trong vùng nghiên cứu, gắn trạm của mảnh phần riêng chứa điểm giữa đoạn
    from shapely.geometry import LineString, Point
    ids = [b for b, g in parts.items() if g is not None]
    tree = STRtree([parts[b] for b in ids])
    to_3035 = Transformer.from_crs(4326, 3035, always_xy=True).transform
    reaches = {}
    with shapefile.Reader(str(path)) as reader:
        for item in reader.iterShapeRecords(bbox=STUDY_BBOX):
            rec = item.record.as_dict()
            reaches[rec["HYRIV_ID"]] = (rec, item.shape.points)
    # Kiểm chiều vẽ: điểm cuối đoạn trùng điểm đầu đoạn hạ lưu
    linked = [(pts, reaches[r["NEXT_DOWN"]][1]) for r, pts in reaches.values() if r["NEXT_DOWN"] in reaches]
    forward = sum(1 for a, b in linked if abs(a[-1][0] - b[0][0]) + abs(a[-1][1] - b[0][1]) < 1e-6)
    if forward < 0.95 * len(linked):
        raise ValueError(f"Đoạn sông không vẽ theo chiều nước chảy: {forward}/{len(linked)}")
    features, per_gauge = [], {}
    for rec, pts in reaches.values():
        if rec["UPLAND_SKM"] < RIVER_UPLAND_KM2 or len(pts) < 2:
            continue
        line = LineString(pts)
        mid = Point(to_3035(*line.interpolate(0.5, normalized=True).coords[0]))
        hit = [ids[j] for j in tree.query(mid, predicate="within")]
        if not hit:
            if rec["UPLAND_SKM"] < RIVER_FREE_KM2:
                continue
            hit = [""]
        g = line.simplify(RIVER_SIMPLIFY_DEG)
        features.append(dict(type="Feature", properties=dict(g=hit[0], q=round(float(rec["DIS_AV_CMS"]), 2), u=round(float(rec["UPLAND_SKM"]))),
                             geometry=dict(type="LineString", coordinates=[[round(x, 5), round(y, 5)] for x, y in g.coords])))
        per_gauge[hit[0]] = per_gauge.get(hit[0], 0) + 1
    free = per_gauge.pop("", 0)
    print(f"rivers: {len(features)} đoạn (trong lưu vực có trạm từ {RIVER_UPLAND_KM2} km², ngoài lưu vực từ {RIVER_FREE_KM2} km²: "
          f"{free} đoạn), chiều vẽ xuôi dòng {forward}/{len(linked)}, {len(per_gauge)}/{len(ids)} trạm có đoạn sông")
    return features


def snowline(swe, elev):
    # Đường tuyết mỗi ngày: độ cao trung bình lưu vực tại đó xác suất "có tuyết" (>= SNOW_MM) là 50%,
    # hồi quy logistic theo độ cao trên 451 lưu vực (IRLS). Không lưu vực nào có tuyết => None;
    # đường tuyết nằm ngoài khoảng độ cao có dữ liệu => chặn ở biên (không ngoại suy)
    lo, hi = float(np.nanmin(elev)), float(np.nanmax(elev))
    z = (elev - elev.mean()) / elev.std()
    out = []
    for d in range(swe.shape[1]):
        ok = np.isfinite(swe[:, d])
        y, x = (swe[ok, d] >= SNOW_MM).astype(float), z[ok]
        if y.sum() == 0:
            out.append(None)
            continue
        if y.sum() == len(y):
            out.append(round(lo))
            continue
        beta = np.zeros(2)
        X = np.column_stack([np.ones_like(x), x])
        for _ in range(50):
            p = 1 / (1 + np.exp(-(X @ beta)))
            w = np.clip(p * (1 - p), 1e-6, None)
            step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-6 * np.eye(2), X.T @ (y - p))
            beta += step
            if np.abs(step).max() < 1e-8:
                break
        if beta[1] <= 0:
            out.append(None)
            continue
        h = float(elev.mean() + elev.std() * (-beta[0] / beta[1]))
        out.append(round(min(max(h, lo), hi)))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--rivers", type=Path, help="HydroRIVERS_v10_eu.shp (tùy chọn)")
    args = parser.parse_args()
    raw, bundle = args.raw.resolve(), args.bundle.resolve()
    catalog = json.loads((bundle / "catalog.json").read_text(encoding="utf-8"))
    ids, dates = list(catalog["ids"]), pd.to_datetime(catalog["dates"])
    columns = [layer["column"] for layer in DAILY_LAYERS] + list(WIND_UV)

    cube = {c: np.full((len(ids), len(dates)), np.nan, dtype=np.float32) for c in columns}
    for row, bid in enumerate(ids):
        frame = pd.read_csv(raw / DAILY / f"ID_{bid}.csv", sep=";", usecols=["YYYY", "MM", "DD", *columns])
        frame.index = pd.to_datetime(dict(year=frame.YYYY, month=frame.MM, day=frame.DD))
        if frame.index.duplicated().any():
            raise ValueError(f"Lưu vực {bid} có ngày lặp.")
        frame = frame.reindex(dates)
        for c in columns:
            cube[c][row] = frame[c].to_numpy(np.float32)

    # Kiểm độc lập: 40 ô ngẫu nhiên đọc lại bằng csv
    rng = np.random.default_rng(20261128)
    for _ in range(40):
        row, col, c = int(rng.integers(len(ids))), int(rng.integers(len(dates))), columns[int(rng.integers(len(columns)))]
        day = dates[col]
        with open(raw / DAILY / f"ID_{ids[row]}.csv", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            hit = next(r for r in reader if (int(r["YYYY"]), int(r["MM"]), int(r["DD"])) == (day.year, day.month, day.day))
        want = float(hit[c]) if hit[c] not in ("", "NaN", "nan") else np.nan
        got = float(cube[c][row, col])
        if not (np.isnan(want) and np.isnan(got)) and abs(want - got) > 1e-4 * max(1.0, abs(want)):
            raise ValueError(f"Lệch ô {ids[row]} {day.date()} {c}: CSV {want}, gói {got}")
    print("Kiểm 40 ô ngẫu nhiên: khớp CSV gốc")

    out_dir = bundle / "env"
    out_dir.mkdir(exist_ok=True)
    layers = []
    for layer in DAILY_LAYERS:
        values = cube[layer["column"]]
        np.save(out_dir / f"{layer['key']}.npy", values, allow_pickle=False)
        ramp = RAMPS[layer["ramp"]]
        layers.append(dict(layer, kind="daily", stops=stops(values, layer["scale"], len(ramp)), colors=ramp,
                           missing=int(np.isnan(values).sum())))
        print(f"{layer['key']:>8}: {layer['column']}, thiếu {int(np.isnan(values).sum())} ô, ngưỡng {layers[-1]['stops']}")
    for name, column in zip(("wind_u", "wind_v"), WIND_UV):
        np.save(out_dir / f"{name}.npy", cube[column], allow_pickle=False)

    attrs = pd.read_csv(raw / ATTRS, sep=";")
    attrs["ID"] = attrs.ID.astype(int).astype(str)
    attrs = attrs.set_index("ID", verify_integrity=True).loc[ids]
    for layer in STATIC_LAYERS:
        values = attrs[layer["column"]].to_numpy(float)
        if layer["scale"] == "fraction":
            values = values * 100
        ramp = RAMPS[layer["ramp"]]
        layers.append(dict(layer, kind="static", stops=stops(values, "range" if layer["scale"] == "fraction" else layer["scale"], len(ramp)),
                           colors=ramp, values=[round(float(v), 3) if np.isfinite(v) else None for v in values]))
        print(f"{layer['key']:>8}: {layer['column']}, ngưỡng {layers[-1]['stops']}")

    elev = attrs["elev_mean"].to_numpy(float)
    line = snowline(cube["ERA5L_swe"], elev)
    month = dates.month
    for m in (1, 4, 7, 10):
        vals = [v for v, mm in zip(line, month) if mm == m and v is not None]
        print(f"Đường tuyết tháng {m}: trung vị {np.median(vals) if vals else float('nan'):.0f} m, {len(vals)} ngày có tuyết")
    snow = dict(values=line, threshold_mm=SNOW_MM, elev_range=[round(float(elev.min())), round(float(elev.max()))],
                method="logistic P(SWE >= threshold) vs basin mean elevation, 451 basins, per day")

    own, parts = own_parts(raw, ids)
    if args.rivers:
        lines = rivers(args.rivers.resolve(), parts)
        (bundle / "rivers.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=lines), separators=(",", ":")),
                                               encoding="utf-8")
    (bundle / "basins_own.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=own), separators=(",", ":")),
                                               encoding="utf-8")

    meta = dict(schema=1, source="Extended LamaH-CE daily (Zenodo 17119635)", units_source="BiasCast (HESS 30:5067-5096) Table A1, A2",
                ids=ids, start=catalog["dates"][0], days=len(dates), layers=layers, snowline=snow)
    (bundle / "env.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    size = sum(p.stat().st_size for p in out_dir.glob("*.npy")) / 1e6
    print(f"env.json: {len(layers)} lớp; env/: {len(list(out_dir.glob('*.npy')))} tệp, {size:.1f} MB")


if __name__ == "__main__":
    main()
