# Thêm vào gói website dữ liệu cho bản đồ và thẻ lưu vực, đọc từ Extended LamaH-CE thô (Zenodo 17119635):
# - site.json: tên trạm, sông, diện tích, độ cao, quốc gia, vùng và ba ngưỡng bốn mức q30/q80/q98 (m³/s)
#   tính như A1 Phần 10.1 (phân vị mẫu qmax 1981-2013, nội suy tuyến tính, bằng ngưỡng vào mức cao hơn);
#   kiểm lại bằng số ngày quan trắc từng mức của levels.csv (Evaluation, run SeqLSTM_Q) => lệch thì dừng.
# - basins.geojson: ranh giới lưu vực (Basins_A, EPSG:3035 => WGS84), làm gọn 100 m.
# Chạy: python build_site_data.py --raw <LamaH_extended> --bundle <gói website>

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform

LEVELS = ("low", "medium", "high", "very_high")
QUANTILES = (0.30, 0.80, 0.98)
REFERENCE = (1981, 2013)
TEST_YEARS = (2014, 2017)
SIMPLIFY_M = 100


def thresholds(raw, bid, area):
    # Ngưỡng tính trên mm/ngày như A1 (qmax x 86,4 / area_gov); qmax test đổi đơn vị như thư viện
    # (_normalize_discharge, lamah.py) để kiểm số ngày từng mức đúng như bộ đánh giá
    frame = pd.read_csv(raw / f"D_gauges/2_timeseries/daily/ID_{bid}.csv", sep=";")
    q = frame.loc[frame.YYYY.between(*REFERENCE), "qmax"].dropna().to_numpy(float) * (86.4 / area)
    if q.size == 0:
        raise ValueError(f"Lưu vực {bid} không có qmax {REFERENCE}.")
    test = frame.loc[frame.YYYY.between(*TEST_YEARS), "qmax"].dropna().to_numpy(float) / (area * 1e6) * 1000 * 86400
    return [float(v) for v in np.quantile(q, QUANTILES, method="linear")], test


def classify(values, cuts):
    # 0 thấp, 1 trung bình, 2 cao, 3 rất cao; bằng ngưỡng vào mức cao hơn
    return np.searchsorted(np.asarray(cuts), values, side="right")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--a1-levels", type=Path, help="A1_P10_qmax_level_thresholds.csv (Output A1 Version 3)")
    parser.add_argument("--confusion", type=Path, help="Classification_confusion_pooled.csv (Output Evaluation Version 13)")
    args = parser.parse_args()
    raw, bundle = args.raw.resolve(), args.bundle.resolve()
    catalog = json.loads((bundle / "catalog.json").read_text(encoding="utf-8"))
    ids = list(catalog["ids"])

    if args.confusion:
        # Ma trận nhầm lẫn bốn mức gộp 451 lưu vực: kiểm cột, mức, đủ mọi run của web rồi chép vào gói
        cm = pd.read_csv(args.confusion)
        need = {"part", "run", "lead", "observed_level", "predicted_level", "days"}
        if not need <= set(cm.columns):
            raise ValueError(f"Bảng ma trận nhầm lẫn thiếu cột {need - set(cm.columns)}")
        if set(cm.observed_level) | set(cm.predicted_level) != set(LEVELS):
            raise ValueError("Bảng ma trận nhầm lẫn có mức lạ.")
        missing = {m["run"] for m in catalog["models"]} - set(cm.run)
        if missing:
            raise ValueError(f"Bảng ma trận nhầm lẫn thiếu run {missing}")
        cm.to_csv(bundle / "confusion.csv", index=False)
        print(f"confusion.csv: {len(cm)} dòng, {cm.run.nunique()} run")

    gauges = pd.read_csv(raw / "D_gauges/1_attributes/Gauge_attributes.csv", sep=";")
    gauges["basin_id"] = gauges.ID.astype(int).astype(str)
    gauges = gauges.set_index("basin_id", verify_integrity=True).loc[ids]
    found = {bid: thresholds(raw, bid, float(gauges.loc[bid, "area_gov"])) for bid in ids}
    cuts = {bid: found[bid][0] for bid in ids}
    source = "computed"
    if args.a1_levels:
        a1 = pd.read_csv(args.a1_levels, dtype={"basin_id": str})
        a1["basin_id"] = a1.basin_id.str.removeprefix("ID_").astype(int).astype(str)
        a1 = a1.set_index("basin_id", verify_integrity=True).loc[ids]
        cuts = {bid: [float(a1.loc[bid, f"q{k}_mm_day"]) for k in (30, 80, 98)] for bid in ids}
        source = "A1"

    # Kiểm ngưỡng: đếm ngày quan trắc test từng mức, so với levels.csv của bộ đánh giá
    levels = pd.read_csv(bundle / "levels.csv", dtype={"basin_id": str})
    ref = levels[levels.run.eq("SeqLSTM_Q")].pivot(index="basin_id", columns="level", values="observed_days")
    mismatch = []
    for bid in ids:
        if bid not in ref.index:
            continue
        values = found[bid][1]
        counts = [int(c) for c in np.bincount(classify(values, cuts[bid]), minlength=4)]
        if counts != [int(ref.loc[bid, lv]) for lv in LEVELS]:
            mismatch.append((bid, counts, [int(ref.loc[bid, lv]) for lv in LEVELS]))
    print(f"Kiểm ngưỡng bốn mức: {len(ref) - len(mismatch)}/{len(ref)} lưu vực khớp levels.csv")
    # levels.csv chỉ đếm ngày run SeqLSTM_Q có dự báo và dùng Q mm/ngày của bộ đánh giá, nên số ngày
    # tính lại từ Q thô lệch ở ngày sát ngưỡng; ngưỡng A1 là nguồn gốc nên chỉ báo, không dừng
    if mismatch:
        worst = max(max(abs(a - b) for a, b in zip(c, r)) for _, c, r in mismatch)
        print(f"Lệch số ngày ở {len(mismatch)} lưu vực, lệch lớn nhất {worst} ngày, ví dụ {mismatch[:2]}")

    stations = {}
    for bid in ids:
        g = gauges.loc[bid]
        stations[bid] = dict(name=str(g["name"]), river=str(g.river), area_km2=float(g.area_gov),
                             elev_m=float(g.elev), country=str(g.country), region=str(g.region),
                             levels_mm_day=cuts[bid],
                             levels_m3_s=[v * float(g.area_gov) / 86.4 for v in cuts[bid]])
    site = dict(schema=1, source="Extended LamaH-CE daily (Zenodo 17119635)", level_source=source,
                level_check=dict(matched=len(ref) - len(mismatch), total=len(ref)), level_quantiles=list(QUANTILES),
                level_reference_years=list(REFERENCE), stations=stations)
    (bundle / "site.json").write_text(json.dumps(site, ensure_ascii=False), encoding="utf-8")

    to_wgs84 = Transformer.from_crs(3035, 4326, always_xy=True).transform
    features = []
    with shapefile.Reader(str(raw / "A_basins_total_upstrm/3_shapefiles/Basins_A.shp"), encoding="latin1") as reader:
        for item in reader.iterShapeRecords():
            bid = str(int(item.record["ID"]))
            if bid not in stations:
                continue
            geom = transform(to_wgs84, shape(item.shape.__geo_interface__).simplify(SIMPLIFY_M, preserve_topology=True))
            features.append(dict(type="Feature", properties=dict(id=bid, area=stations[bid]["area_km2"]),
                                 geometry=json.loads(json.dumps(mapping(geom)), parse_float=lambda v: round(float(v), 5))))
    if {f["properties"]["id"] for f in features} != set(ids):
        raise ValueError("Basins_A thiếu ranh giới của một số lưu vực.")
    # Lưu vực lớn vẽ trước để lưu vực nhỏ nằm trên, bấm được
    features.sort(key=lambda f: -f["properties"]["area"])
    path = bundle / "basins.geojson"
    path.write_text(json.dumps(dict(type="FeatureCollection", features=features), separators=(",", ":")), encoding="utf-8")
    print(f"site.json: {len(stations)} trạm; basins.geojson: {len(features)} lưu vực, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
