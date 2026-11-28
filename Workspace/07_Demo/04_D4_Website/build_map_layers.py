# Thêm vào gói website dữ liệu cho các lớp bản đồ khí tượng, đất, thực vật và đặc điểm lưu vực, đọc từ
# Extended LamaH-CE thô (Zenodo 17119635); mọi biến đều là đầu vào của mô hình C2 (config.yml của run C2):
# - env/<khóa>.npy: ma trận float32 [451 lưu vực, ngày kỳ test] cùng thứ tự với catalog.json, giá trị trung bình
#   cả lưu vực từ A_basins_total_upstrm/2_timeseries/daily/ID_<id>.csv (ô trống => NaN);
# - env.json: danh sách lớp, đơn vị (bảng A1, A2 của BiasCast), ngưỡng màu theo phân vị, giá trị thuộc tính tĩnh
#   từ Catchment_attributes.csv.
# Kiểm: đọc lại 40 ô ngẫu nhiên bằng mô-đun csv (không qua pandas), lệch thì dừng.
# Chạy: python build_map_layers.py --raw <LamaH_extended> --bundle <gói website>

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DAILY = "A_basins_total_upstrm/2_timeseries/daily"
ATTRS = "A_basins_total_upstrm/1_attributes/Catchment_attributes.csv"
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
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

    meta = dict(schema=1, source="Extended LamaH-CE daily (Zenodo 17119635)", units_source="BiasCast (HESS 30:5067-5096) Table A1, A2",
                ids=ids, start=catalog["dates"][0], days=len(dates), layers=layers)
    (bundle / "env.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    size = sum(p.stat().st_size for p in out_dir.glob("*.npy")) / 1e6
    print(f"env.json: {len(layers)} lớp; env/: {len(list(out_dir.glob('*.npy')))} tệp, {size:.1f} MB")


if __name__ == "__main__":
    main()
