# =============================================================================
# Giới thiệu bộ dữ liệu Extended LamaH-CE (bài cơ sở BiasCast, HESS 2026) — bản ngắn cho GVHD
# Chỉ trình bày các dạng dữ liệu cơ bản; phân tích sâu (thiếu dữ liệu, phân phối, dự báo so với tái phân tích)
# ở LamaHCE/LamaHCEExt_Analysis.py.
# Notebook Kaggle: Accelerator = None; Add Input → dataset "lamah-ce-ext" (Kaggle đã giải nén sẵn thành thư mục).
# Chạy được trên Colab nếu chép thư mục dataset vào /content. Kế hoạch: Document/3-DuLieu/LamaHCE.md Mục 6.2.
#
# MỤC LỤC
#   Phần 1 — Tổng quan bộ dữ liệu
#   Phần 2 — Khu vực nghiên cứu
#   Phần 3 — Ví dụ dữ liệu
#   Phần 4 — Chia tập theo thời gian
# =============================================================================

# %% Phần 1 — Tổng quan bộ dữ liệu
# Thẻ tổng quan, sơ đồ cấu trúc dữ liệu, heatmap độ phủ nguồn × năm (đánh dấu ba giai đoạn chia tập). Đọc mọi tệp chuỗi ngày một lần (khoảng 1–3 phút trên Kaggle).
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.patches import FancyBboxPatch

IS_KAGGLE = Path("/kaggle/input").exists()
IS_COLAB = "google.colab" in sys.modules
SEARCH_ROOTS = [Path("/kaggle/input"), Path("/content"), Path.cwd()]
OUT_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("/content") if IS_COLAB else Path.cwd()
SEP, MISSING = ";", -999                      # tệp LamaH-CE: dấu chấm phẩy, mã thiếu -999
YEARS = range(1981, 2018)
PERIODS = {"Huấn luyện": ("2003", "2009"), "Kiểm định": ("2010", "2013"), "Kiểm tra": ("2014", "2017")}
SOURCES = {"ERA5L_": "ERA5-Land (tái phân tích)", "EOBS_": "E-OBS (quan trắc lưới)",
           "MSWEP_": "MSWEP (mưa tổng hợp)", "GLEAM_": "GLEAM (bốc hơi)", "ECMWF_": "ECMWF HRES (dự báo)"}
COLORS = {"data": "#dae8fc", "met": "#fff2cc", "fc": "#ffe6cc", "q": "#f8cecc", "attr": "#d5e8d4", "out": "#e1d5e7"}


def find_dir(name: str) -> Path:
    """Tìm thư mục theo tên trong các gốc tìm kiếm; báo lỗi rõ nếu chưa gắn dataset."""
    for root in SEARCH_ROOTS:
        if root.exists():
            hits = sorted(p for p in root.rglob(name) if p.is_dir())
            if hits:
                return hits[0]
    raise FileNotFoundError(f"Không thấy thư mục {name}: gắn dataset lamah-ce-ext bằng Add Input")


def read_series(path: Path) -> pd.DataFrame:
    """Đọc tệp chuỗi ngày, đặt chỉ mục ngày, đổi mã thiếu -999 thành NaN."""
    df = pd.read_csv(path, sep=SEP)
    df.index = pd.to_datetime(df[["YYYY", "MM", "DD"]].rename(columns={"YYYY": "year", "MM": "month", "DD": "day"}))
    return df.drop(columns=["YYYY", "MM", "DD"]).replace(MISSING, np.nan)


def source_of(col: str) -> str:
    return next((name for prefix, name in SOURCES.items() if col.startswith(prefix)), "Khác")


_CACHE = {}
def load_matrix(cols, kind: str = "met") -> dict:
    """Trả về {biến: DataFrame(ngày × lưu vực)} float32 cho mọi lưu vực; mỗi biến chỉ đọc một lần (cache).
    kind='met': tệp khí tượng lưu vực; kind='q': tệp lưu lượng trạm. Biến không có trong dữ liệu thì bỏ qua."""
    need = [c for c in cols if (kind, c) not in _CACHE]
    if need:
        files = MET_FILES if kind == "met" else [GAUGE_FILES[f.stem] for f in MET_FILES if f.stem in GAUGE_FILES]
        data = {c: {} for c in need}
        for k, f in enumerate(files):
            header = pd.read_csv(f, sep=SEP, nrows=0).columns
            use = [c for c in need if c in header]
            if use:
                df = pd.read_csv(f, sep=SEP, usecols=["YYYY", "MM", "DD"] + use)
                idx = pd.to_datetime(dict(year=df["YYYY"], month=df["MM"], day=df["DD"]))
                for c in use:
                    data[c][f.stem.removeprefix("ID_")] = pd.Series(
                        df[c].replace(MISSING, np.nan).astype("float32").values, index=idx)
            if (k + 1) % 300 == 0:
                print(f"  đọc {kind}: {k + 1}/{len(files)}", flush=True)
        for c in need:
            if data[c]:
                _CACHE[(kind, c)] = pd.DataFrame(data[c]).reindex(pd.date_range("1981-01-01", "2017-12-31"))
            else:
                print(f"  CẢNH BÁO: không có biến {c} trong dữ liệu, bỏ qua")
    return {c: _CACHE[(kind, c)] for c in cols if (kind, c) in _CACHE}


EXT = find_dir("LamaH_extended")
MET_FILES = sorted((EXT / "A_basins_total_upstrm" / "2_timeseries" / "daily").glob("ID_*.csv"))
GAUGE_FILES = {f.stem: f for f in (EXT / "D_gauges" / "2_timeseries" / "daily").glob("ID_*.csv")}
attr_files = sorted((EXT / "A_basins_total_upstrm" / "1_attributes").glob("*.csv"))
attrs = max((pd.read_csv(f, sep=SEP) for f in attr_files), key=lambda d: d.shape[1])   # bảng thuộc tính lớn nhất

# Đọc mọi tệp: độ phủ (% ô có dữ liệu) theo năm cho từng nguồn và cho lưu lượng
coverage, counts, first_date = {}, {}, {}
for k, f in enumerate(MET_FILES):
    met = read_series(f).drop(columns=["DOY"], errors="ignore")     # DOY là số thứ tự ngày, không phải biến đo
    parts = [("met", met)] + ([("q", read_series(GAUGE_FILES[f.stem]))] if f.stem in GAUGE_FILES else [])
    for kind, df in parts:
        for col in df.columns:
            src = source_of(col) if kind == "met" else "Lưu lượng trạm"
            ok = df[col].notna().groupby(df.index.year).mean().reindex(YEARS, fill_value=0)
            coverage[src] = coverage.get(src, 0) + ok
            counts[src] = counts.get(src, 0) + 1
            fv = df[col].first_valid_index()
            if fv is not None and (src not in first_date or fv < first_date[src]):
                first_date[src] = fv
    if (k + 1) % 200 == 0:
        print(f"  đã đọc {k + 1}/{len(MET_FILES)} lưu vực", flush=True)
cov = pd.DataFrame({src: coverage[src] / counts[src] * 100 for src in coverage}).T
print("Ngày đầu tiên có dữ liệu theo nguồn:", {s: str(d.date()) for s, d in first_date.items()})
print("Độ phủ năm đầu có dữ liệu (%):",
      {s: round(float(cov.loc[s, first_date[s].year]), 1) for s in cov.index if s in first_date})

sample_id = MET_FILES[0].stem
sample_met = read_series(MET_FILES[0]).drop(columns=["DOY"], errors="ignore")
sample_q = read_series(GAUGE_FILES[sample_id])
groups = {}
for col in sample_met.columns:
    groups.setdefault(source_of(col), []).append(col)
first_year = {src: d.year for src, d in first_date.items()}

# Bảng 1.1 — thẻ tổng quan
overview = pd.DataFrame([
    ("Khu vực", "Trung Âu: lưu vực sông Danube thượng nguồn và toàn bộ Áo (LamaH-CE)"),
    ("Số lưu vực có chuỗi khí tượng", len(MET_FILES)),
    ("Số trạm có chuỗi lưu lượng ngày", len(GAUGE_FILES)),
    ("Giai đoạn", f"{min(YEARS)}–{max(YEARS)}, theo ngày"),
    ("Biến khí tượng", "; ".join(f"{src}: {len(c)}" for src, c in groups.items() if src != "Khác")),
    ("Thuộc tính tĩnh", f"{attrs.shape[1] - 1} thuộc tính cho {attrs.shape[0]} lưu vực"),
    ("Lưu lượng", ", ".join(sample_q.columns) + " (m³/s); qmax là đại lượng cần dự báo"),
    ("Nguồn công bố", "Zenodo 17119635, CC BY-NC 4.0, 0,95 GB nén"),
], columns=["Mục", "Nội dung"])
print("Bảng 1.1. Tổng quan bộ dữ liệu Extended LamaH-CE")
display(overview)

# Hình 1.1 — sơ đồ cấu trúc dữ liệu
fig, ax = plt.subplots(figsize=(16, 6.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6.5)
ax.axis("off")


def box(x, y, w, h, text, color, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#555555", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, fontweight="bold" if bold else None)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4))


src_list = [s for s in groups if s != "Khác"]
h, gap = 0.62, 0.18
top = 6.0
for i, src in enumerate(src_list):
    y = top - (i + 1) * (h + gap)
    color = COLORS["fc"] if src.startswith("ECMWF") else COLORS["met"]
    box(0.2, y, 4.2, h, f"{src}\n{len(groups[src])} biến · từ {first_year.get(src, '?')}", color)
    arrow(4.45, y + h / 2, 5.6, 3.6)
box(5.6, 2.9, 4.2, 1.5, f"Chuỗi ngày theo lưu vực\n{len(MET_FILES)} lưu vực × {len(YEARS)} năm\n"
    f"{sum(len(c) for s, c in groups.items() if s != 'Khác')} biến khí tượng", COLORS["data"], bold=True)
box(5.6, 1.5, 4.2, 1.0, f"Lưu lượng ngày tại trạm cửa ra\nqmin · qmean · qmax (m³/s) · {len(GAUGE_FILES)} trạm",
    COLORS["q"], bold=True)
box(5.6, 4.8, 4.2, 1.0, f"Thuộc tính tĩnh lưu vực\n{attrs.shape[1] - 1} thuộc tính (địa hình, khí hậu, thảm phủ…)",
    COLORS["attr"], bold=True)
arrow(9.85, 3.65, 11.0, 3.65)
arrow(9.85, 2.0, 11.0, 3.2)
arrow(9.85, 5.3, 11.0, 4.1)
box(11.0, 2.6, 4.8, 2.1, "Một mẫu dự báo (lưu vực, ngày t)\n\nQuá khứ: 364 ngày khí tượng + lưu lượng\n"
    "Tương lai: dự báo thời tiết ngày t\nThuộc tính tĩnh của lưu vực\n→ Nhãn: qmax ngày t", COLORS["out"], bold=True)
ax.set_title("Hình 1.1. Cấu trúc bộ dữ liệu Extended LamaH-CE và cách tạo một mẫu dự báo", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "gt_1_1_cau_truc.png", dpi=150)
plt.show()

# Hình 1.2 — độ phủ dữ liệu nguồn × năm, đánh dấu ba giai đoạn chia tập
fig, ax = plt.subplots(figsize=(16, 4.2))
im = ax.imshow(cov.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
ax.set_yticks(range(len(cov.index)), cov.index)
ax.set_xticks(range(len(cov.columns)), cov.columns, rotation=90)
for name, (a, b) in PERIODS.items():
    x0, x1 = list(YEARS).index(int(a)) - 0.5, list(YEARS).index(int(b)) + 0.5
    ax.axvspan(x0, x1, ymin=1.0, ymax=1.07, color={"Huấn luyện": "#82b366", "Kiểm định": "#d6b656",
                                                    "Kiểm tra": "#b85450"}[name], clip_on=False)
    ax.text((x0 + x1) / 2, 1.035, name, ha="center", va="center", fontsize=9, color="white", fontweight="bold",
            transform=ax.get_xaxis_transform())
ax.set_title("Hình 1.2. Độ phủ dữ liệu theo năm (% ô có giá trị, trung bình mọi biến và mọi lưu vực)", pad=22)
fig.colorbar(im, ax=ax, label="%")
fig.tight_layout()
fig.savefig(OUT_DIR / "gt_1_2_do_phu.png", dpi=150)
plt.show()


# %% Phần 2 — Khu vực nghiên cứu
# Bản đồ vị trí trạm và ranh giới lưu vực, tô màu theo độ cao trung bình của lưu vực.
KEY_ATTRS = {"area_calc": "Diện tích (km²)", "elev_mean": "Độ cao trung bình (m)", "slope_mean": "Độ dốc trung bình",
             "p_mean": "Lượng mưa trung bình", "arid_1": "Chỉ số khô hạn (arid_1)",
             "frac_snow": "Tỷ lệ mưa dạng tuyết", "forest_fra": "Tỷ lệ rừng", "agr_fra": "Tỷ lệ đất nông nghiệp",
             "urban_fra": "Tỷ lệ đô thị", "lake_fra": "Tỷ lệ hồ"}
id_col = next(c for c in attrs.columns if c.lower() == "id")
A = attrs.assign(ID=attrs[id_col].astype(str)).set_index("ID")
key = {c: lab for c, lab in KEY_ATTRS.items() if c in A.columns}
print("Thuộc tính không tìm thấy:", [c for c in KEY_ATTRS if c not in A.columns] or "không có")

# Toạ độ trạm và ranh giới lưu vực từ shapefile (cần geopandas; thiếu thì dùng cột lon/lat của bảng trạm)
XY, CATCH = None, None
try:
    import geopandas as gpd
    g_shp = sorted((EXT / "D_gauges" / "3_shapefiles").glob("*.shp"))
    c_shp = sorted((EXT / "A_basins_total_upstrm" / "3_shapefiles").glob("*.shp"))
    if g_shp:
        G = gpd.read_file(g_shp[0])
        gid = next(c for c in G.columns if c.lower() == "id")
        XY = pd.DataFrame({"x": G.geometry.x.values, "y": G.geometry.y.values},
                          index=G[gid].astype(int).astype(str))
        print("Trạm lấy từ", g_shp[0].name, "| hệ toạ độ", G.crs)
    if c_shp:
        CATCH = gpd.read_file(c_shp[0])
        if XY is not None and CATCH.crs is not None and G.crs is not None and CATCH.crs != G.crs:
            CATCH = CATCH.to_crs(G.crs)
        print("Ranh giới lưu vực lấy từ", c_shp[0].name, f"({len(CATCH)} đa giác)")
except Exception as err:                       # thiếu geopandas hoặc shapefile khác cấu trúc
    print("Không đọc được shapefile:", err)
if XY is None:
    gauge_attr = max((pd.read_csv(f, sep=SEP) for f in sorted((EXT / "D_gauges" / "1_attributes").glob("*.csv"))),
                     key=lambda d: d.shape[1])
    lon = next((c for c in gauge_attr.columns if c.lower() in ("lon", "long", "longitude", "x")), None)
    lat = next((c for c in gauge_attr.columns if c.lower() in ("lat", "latitude", "y")), None)
    gid = next(c for c in gauge_attr.columns if c.lower() == "id")
    XY = gauge_attr.set_index(gauge_attr[gid].astype(str))[[lon, lat]].set_axis(["x", "y"], axis=1)
    print("Trạm lấy từ bảng thuộc tính trạm, cột", lon, lat)
XY = XY.loc[XY.index.intersection(A.index)]


def plot_map(ax, values: pd.Series, title: str, cmap: str = "viridis", log: bool = False):
    """Vẽ nền ranh giới lưu vực (nếu có) và điểm trạm tô màu theo values."""
    if CATCH is not None:
        CATCH.boundary.plot(ax=ax, color="#bbbbbb", linewidth=0.3)
    v = values.reindex(XY.index)
    sc = ax.scatter(XY["x"], XY["y"], c=np.log10(v) if log else v, s=14, cmap=cmap, edgecolor="k", linewidth=0.2)
    plt.colorbar(sc, ax=ax, shrink=0.75, label=("log10 " if log else "") + title)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


fig, ax = plt.subplots(figsize=(12, 8))
plot_map(ax, A["elev_mean"] if "elev_mean" in A else A[list(key)[0]],
         key.get("elev_mean", list(key.values())[0]), cmap="terrain")
ax.set_title(f"Hình 2.1. Vị trí {len(XY)} trạm và ranh giới lưu vực (màu: độ cao trung bình của lưu vực)")
fig.tight_layout()
fig.savefig(OUT_DIR / "gt_2_1_ban_do.png", dpi=150)
plt.show()


# %% Phần 3 — Ví dụ dữ liệu
# Chuỗi mưa và lưu lượng một lưu vực năm 2016; chu kỳ mùa của các nhóm biến khí tượng.
# Hình 3.1 — chuỗi mẫu một lưu vực năm 2016: mưa (cột, trục đảo) và lưu lượng
year = "2016"
prec_col = "ERA5L_prec" if "ERA5L_prec" in sample_met.columns else groups[src_list[0]][0]
fig, ax = plt.subplots(figsize=(16, 4))
ax.bar(sample_met.loc[year].index, sample_met.loc[year, prec_col], color="#9ecae1", label=f"Mưa ({prec_col})")
ax.invert_yaxis()
ax.set_ylabel("Lượng mưa")
ax2 = ax.twinx()
for col, color in [("qmax", "#b85450"), ("qmean", "#555555")]:
    if col in sample_q.columns:
        ax2.plot(sample_q.loc[year].index, sample_q.loc[year, col], color=color, label=f"{col} (m³/s)")
ax2.set_ylabel("Lưu lượng (m³/s)")
handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
ax.legend(handles, [h.get_label() for h in handles], loc="lower left")
ax.set_title(f"Hình 3.1. Lưu vực {sample_id}, năm {year}: mưa và lưu lượng ngày")
fig.tight_layout()
fig.savefig(OUT_DIR / "gt_3_1_chuoi_mau.png", dpi=150)
plt.show()

# Hình 3.2 — chu kỳ mùa theo nhóm biến (trung vị qua lưu vực, dải P25–P75)
MET_GROUPS = {
    "Nhiệt độ (°C)": ["ERA5L_2m_temp_mean", "EOBS_tg"],
    "Lượng mưa (mm/ngày)": ["ERA5L_prec", "EOBS_rr", "MSWEP_RR"],
    "Tuyết (nước tương đương)": ["ERA5L_swe"],
    "Bốc hơi thực tế": ["ERA5L_total_et", "GLEAM_ETA"],
    "Bức xạ mặt trời thuần": ["ERA5L_surf_net_solar_rad_mean"],
    "Ẩm đất tầng mặt": ["ERA5L_volsw_123"],
}
M = load_matrix([c for cs in MET_GROUPS.values() for c in cs], "met")
months = np.arange(1, 13)
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for ax, (title, cs) in zip(axes.flat, MET_GROUPS.items()):
    for c in cs:
        if c not in M:
            continue
        clim = M[c].groupby(M[c].index.month).mean()           # tháng × lưu vực
        ax.plot(months, clim.median(axis=1), marker="o", label=c)
        ax.fill_between(months, clim.quantile(0.25, axis=1), clim.quantile(0.75, axis=1), alpha=0.2)
    ax.set_title(title)
    ax.set_xticks(months)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("Hình 3.2. Chu kỳ mùa của các biến khí tượng 1981–2017 (đường: trung vị qua lưu vực; dải: P25–P75)",
             fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "gt_3_2_chu_ky_mua.png", dpi=150)
plt.show()


# %% Phần 4 — Chia tập theo thời gian
# Trục thời gian ba giai đoạn huấn luyện, kiểm định, kiểm tra (phần nhạt: 364 ngày quá khứ làm đầu vào).
WARMUP = 364
fig, ax = plt.subplots(figsize=(16, 2.6))
colors = {"Huấn luyện": "#82b366", "Kiểm định": "#d6b656", "Kiểm tra": "#b85450"}
for i, (name, (a, b)) in enumerate(PERIODS.items()):
    start, end = pd.Timestamp(f"{a}-01-01"), pd.Timestamp(f"{b}-12-31")
    ax.barh(0, (end - start).days, left=start, color=colors[name], height=0.5)
    ax.barh(0, WARMUP, left=start - pd.Timedelta(days=WARMUP), color=colors[name], alpha=0.3, height=0.5)
    ax.text(start + (end - start) / 2, 0, f"{name}\n{a}–{b}", ha="center", va="center", color="white",
            fontweight="bold")
ax.set_yticks([])
ax.set_xlim(pd.Timestamp("2001-06-01"), pd.Timestamp("2018-03-01"))
ax.set_title("Hình 4.1. Ba giai đoạn chia tập theo thời gian (phần nhạt: 364 ngày quá khứ dùng làm đầu vào)")
fig.tight_layout()
fig.savefig(OUT_DIR / "gt_4_1_chia_tap.png", dpi=150)
plt.show()
