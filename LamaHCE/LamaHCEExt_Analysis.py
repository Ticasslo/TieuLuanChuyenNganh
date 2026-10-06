# =============================================================================
# Phân tích sâu bộ dữ liệu Extended LamaH-CE (bài cơ sở BiasCast, HESS 2026)
# Phục vụ chương dữ liệu chi tiết và các quyết định khi dựng bộ nạp dữ liệu; bản giới thiệu ngắn cho GVHD ở
# LamaHCE/LamaHCEExt_Explore.py. Chỉ phân tích dữ liệu, không khảo sát kết quả của nhóm tác giả.
# Notebook Kaggle: Accelerator = None; Add Input → dataset "lamah-ce-ext".
#
# MỤC LỤC
#   Phần 1 — Thiết lập, thuộc tính lưu vực
#   Phần 2 — Khí tượng: so sánh nguồn mưa, tương quan giữa biến
#   Phần 3 — Lưu lượng: thiếu dữ liệu, phân phối, mùa lũ
#   Phần 4 — Dự báo thời tiết so với tái phân tích
#   Phần 5 — Khác biệt giữa các giai đoạn
#   Phần 6 — Tổng kết
# =============================================================================

# %% Phần 1 — Thiết lập, thuộc tính lưu vực
# Hàm dùng chung, nạp tệp và thuộc tính; bảng thống kê, bản đồ 4 thuộc tính, histogram, tương quan.
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

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
SUMMARY = []                                  # các điểm chính, gom lại ở Phần 7


def note(part: str, item: str, value) -> None:
    """Ghi một điểm chính vào bảng tổng kết."""
    SUMMARY.append({"Phần": part, "Nội dung": item, "Giá trị": value})


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


PERIOD_COLORS = {"Huấn luyện": "#82b366", "Kiểm định": "#d6b656", "Kiểm tra": "#b85450"}
rng = np.random.default_rng(0)

KEY_ATTRS = {"area_calc": "Diện tích (km²)", "elev_mean": "Độ cao trung bình (m)", "slope_mean": "Độ dốc trung bình",
             "p_mean": "Lượng mưa trung bình", "arid_1": "Chỉ số khô hạn (arid_1)",
             "frac_snow": "Tỷ lệ mưa dạng tuyết", "forest_fra": "Tỷ lệ rừng", "agr_fra": "Tỷ lệ đất nông nghiệp",
             "urban_fra": "Tỷ lệ đô thị", "lake_fra": "Tỷ lệ hồ"}
id_col = next(c for c in attrs.columns if c.lower() == "id")
A = attrs.assign(ID=attrs[id_col].astype(str)).set_index("ID")
key = {c: lab for c, lab in KEY_ATTRS.items() if c in A.columns}
print("Thuộc tính không tìm thấy:", [c for c in KEY_ATTRS if c not in A.columns] or "không có")

stats = A[list(key)].describe(percentiles=[0.25, 0.5, 0.75]).T[["min", "25%", "50%", "75%", "max"]]
stats.index = [key[c] for c in stats.index]
print("Bảng 1.1. Thuộc tính chính của các lưu vực")
display(stats.round(3))
if "area_calc" in A:
    note("1", "Diện tích lưu vực (km²): nhỏ nhất / trung vị / lớn nhất",
         f"{A['area_calc'].min():.0f} / {A['area_calc'].median():.0f} / {A['area_calc'].max():.0f}")
if "elev_mean" in A:
    note("1", "Độ cao trung bình (m): trung vị (khoảng)",
         f"{A['elev_mean'].median():.0f} ({A['elev_mean'].min():.0f}–{A['elev_mean'].max():.0f})")

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


map_attrs = [c for c in ["elev_mean", "area_calc", "arid_1", "frac_snow"] if c in A.columns]
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
for ax, c in zip(axes.flat, map_attrs):
    plot_map(ax, A[c], key[c], cmap={"elev_mean": "terrain", "arid_1": "YlOrBr", "frac_snow": "Blues"}.get(c, "viridis"),
             log=(c == "area_calc"))
for ax in list(axes.flat)[len(map_attrs):]:
    ax.axis("off")
fig.suptitle(f"Hình 1.1. Vị trí {len(XY)} trạm và ranh giới lưu vực, tô màu theo thuộc tính", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "pt_1_1_ban_do.png", dpi=150)
plt.show()

cols = list(key)
nrow = int(np.ceil(len(cols) / 4))
fig, axes = plt.subplots(nrow, 4, figsize=(18, 3.6 * nrow))
for ax, c in zip(axes.flat, cols):
    data = A[c].dropna()
    ax.hist(np.log10(data[data > 0]) if c == "area_calc" else data, bins=30, color="#6c8ebf", edgecolor="white")
    ax.axvline((np.log10(data.median()) if c == "area_calc" else data.median()), color="#b85450", ls="--")
    ax.set_title(("log10 " if c == "area_calc" else "") + key[c], fontsize=10)
for ax in list(axes.flat)[len(cols):]:
    ax.axis("off")
fig.suptitle("Hình 1.2. Phân phối thuộc tính lưu vực (đường đứt: trung vị)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "pt_1_2_thuoc_tinh.png", dpi=150)
plt.show()

corr = A[cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), [key[c] for c in cols], rotation=60, ha="right")
ax.set_yticks(range(len(cols)), [key[c] for c in cols])
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="Spearman")
ax.set_title("Hình 1.3. Tương quan giữa các thuộc tính lưu vực")
fig.tight_layout()
fig.savefig(OUT_DIR / "pt_1_3_tuong_quan.png", dpi=150)
plt.show()


# %% Phần 2 — Khí tượng: so sánh nguồn mưa, tương quan giữa biến
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
prec_cols = [c for c in MET_GROUPS["Lượng mưa (mm/ngày)"] if c in M]
annual = {c: M[c].loc["2003":"2017"].resample("YS").sum(min_count=300).mean() for c in prec_cols}   # mm/năm
ann_df = pd.DataFrame(annual)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].violinplot([ann_df[c].dropna() for c in prec_cols], showmedians=True)
axes[0].set_xticks(range(1, len(prec_cols) + 1), prec_cols)
axes[0].set_ylabel("Lượng mưa năm trung bình 2003–2017")
axes[0].set_title("Hình 2.1a. Lượng mưa năm của mỗi lưu vực theo nguồn")
if len(prec_cols) >= 2:
    base = prec_cols[0]
    for c in prec_cols[1:]:
        axes[1].scatter(ann_df[base], ann_df[c], s=10, alpha=0.6, label=f"{c} so với {base}")
    lim = [ann_df.min().min(), ann_df.max().max()]
    axes[1].plot(lim, lim, "k--", lw=1, label="đường 1:1")
    axes[1].set_xlabel(base)
    axes[1].set_ylabel("Nguồn khác")
    axes[1].legend()
axes[1].set_title("Hình 2.1b. So sánh lượng mưa năm giữa các nguồn")
fig.tight_layout()
fig.savefig(OUT_DIR / "pt_2_1_mua_theo_nguon.png", dpi=150)
plt.show()
med = ann_df.median()
print("Lượng mưa năm trung vị theo nguồn:", med.round(0).to_dict())
note("2", "Lượng mưa năm trung vị theo nguồn (2003–2017)", ", ".join(f"{c}: {v:.0f}" for c, v in med.items()))

met_cols = [c for cs in MET_GROUPS.values() for c in cs if c in M]
rng = np.random.default_rng(0)
pick = rng.choice(M[met_cols[0]].columns, size=min(80, M[met_cols[0]].shape[1]), replace=False)
pooled = pd.DataFrame({c: M[c][pick].loc["2003":"2017"].to_numpy().ravel() for c in met_cols}).dropna()
mcorr = pooled.corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(mcorr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(met_cols)), met_cols, rotation=60, ha="right")
ax.set_yticks(range(len(met_cols)), met_cols)
for i in range(len(met_cols)):
    for j in range(len(met_cols)):
        ax.text(j, i, f"{mcorr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="Spearman")
ax.set_title(f"Hình 2.2. Tương quan giữa các biến khí tượng theo ngày ({len(pick)} lưu vực ngẫu nhiên, 2003–2017)")
fig.tight_layout()
fig.savefig(OUT_DIR / "pt_2_2_tuong_quan_bien.png", dpi=150)
plt.show()


# %% Phần 3 — Lưu lượng: đại lượng cần dự báo
# Bảng tóm tắt; heatmap dữ liệu thiếu trạm × năm; phân phối qmax và lưu lượng riêng (mm/ngày theo diện tích);
# mùa lũ (boxplot theo tháng); tỷ số qmax/qmean.
Q = load_matrix(["qmax", "qmean"], "q")
qmax, qmean = Q["qmax"].where(Q["qmax"] >= 0), Q["qmean"].where(Q["qmean"] >= 0)
area = A["area_calc"].reindex(qmax.columns) if "area_calc" in A else None
first_valid = pd.to_datetime(qmax.apply(pd.Series.first_valid_index))
miss_total = qmax.isna().mean()

summary4 = pd.DataFrame([
    ("Số trạm", qmax.shape[1]),
    ("Năm bắt đầu đo: sớm nhất / muộn nhất", f"{first_valid.min().year} / {first_valid.max().year}"),
    ("% ngày thiếu qmax 1981–2017 (toàn bộ)", f"{qmax.isna().to_numpy().mean() * 100:.2f}%"),
    ("% ngày thiếu qmax 2003–2017", f"{qmax.loc['2003':].isna().to_numpy().mean() * 100:.2f}%"),
    ("Số trạm thiếu trên 10% trong 2003–2017", int((qmax.loc['2003':].isna().mean() > 0.1).sum())),
    ("qmax trung vị qua các trạm (m³/s)", f"{qmax.median().median():.2f}"),
    ("qmax lớn nhất toàn bộ (m³/s)", f"{np.nanmax(qmax.to_numpy()):.0f}"),
], columns=["Mục", "Giá trị"])
print("Bảng 3.1. Tóm tắt lưu lượng")
display(summary4)
note("3", "% ngày thiếu qmax 2003–2017", summary4.iloc[3, 1])
note("3", "Số trạm thiếu trên 10% (2003–2017)", summary4.iloc[4, 1])

miss_year = qmax.isna().groupby(qmax.index.year).mean().T                # trạm × năm
miss_year = miss_year.loc[first_valid.sort_values().index]
fig, ax = plt.subplots(figsize=(16, 7))
im = ax.imshow(miss_year.values * 100, aspect="auto", cmap="Reds", vmin=0, vmax=100, interpolation="nearest")
ax.set_xticks(range(0, len(miss_year.columns), 2), miss_year.columns[::2], rotation=90)
ax.set_ylabel(f"{len(miss_year)} trạm (xếp theo năm bắt đầu đo)")
ax.set_yticks([])
for name, (a, b) in PERIODS.items():
    ax.axvline(list(miss_year.columns).index(int(a)) - 0.5, color="#333333", lw=1, ls="--")
    ax.text(list(miss_year.columns).index(int(a)), -8, name, fontsize=9)
fig.colorbar(im, ax=ax, label="% ngày thiếu trong năm")
ax.set_title("Hình 3.1. Dữ liệu lưu lượng thiếu theo trạm và năm (đỏ = thiếu)")
fig.tight_layout()
fig.savefig(OUT_DIR / "pt_3_1_thieu_luu_luong.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
vals = qmax.to_numpy().ravel()
vals = vals[np.isfinite(vals) & (vals > 0)]
axes[0].hist(np.log10(vals), bins=80, color="#b85450")
axes[0].set_xlabel("log10 qmax (m³/s)")
axes[0].set_title("Hình 3.2a. Phân phối qmax (mọi trạm, mọi ngày)")
if area is not None:
    spec = qmax * 86.4 / area                                             # m³/s → mm/ngày theo diện tích
    sv = spec.to_numpy().ravel()
    sv = sv[np.isfinite(sv) & (sv > 0)]
    axes[1].hist(np.log10(sv), bins=80, color="#6c8ebf")
    axes[1].set_xlabel("log10 qmax riêng (mm/ngày) = qmax × 86,4 / diện tích")
    axes[1].set_title("Hình 3.2b. Lưu lượng riêng (so sánh được giữa các lưu vực)")
    note("3", "qmax riêng trung vị (mm/ngày)", f"{np.median(sv):.2f}")
ratio = (qmax / qmean).median()
axes[2].hist(ratio.dropna(), bins=40, color="#d79b00", edgecolor="white")
axes[2].set_xlabel("Trung vị qmax / qmean của mỗi trạm")
axes[2].set_title("Hình 3.2c. Đỉnh trong ngày cao hơn trung bình ngày bao nhiêu")
fig.tight_layout()
fig.savefig(OUT_DIR / "pt_3_2_phan_phoi.png", dpi=150)
plt.show()

norm = qmax / qmax.mean()                                                 # chuẩn hóa theo trung bình từng trạm
monthly = norm.groupby(norm.index.month).median()                          # tháng × trạm
peak_month = qmax.resample("YS").apply(lambda s: s.idxmax().month if s.notna().any() else np.nan)
fig, axes = plt.subplots(1, 2, figsize=(17, 5))
axes[0].boxplot([monthly.loc[m].dropna() for m in months], showfliers=False)
axes[0].set_xticks(months, [f"T{m}" for m in months])
axes[0].set_ylabel("qmax / qmax trung bình của trạm")
axes[0].set_title("Hình 3.3a. Chu kỳ mùa của lưu lượng (mỗi hộp: phân bố qua các trạm)")
pm = peak_month.to_numpy().ravel()
pm = pm[np.isfinite(pm)].astype(int)
axes[1].bar(months, np.bincount(pm, minlength=13)[1:] / len(pm) * 100, color="#b85450")
axes[1].set_xticks(months, [f"T{m}" for m in months])
axes[1].set_ylabel("% số đỉnh lũ năm")
axes[1].set_title("Hình 3.3b. Tháng xảy ra đỉnh lưu lượng lớn nhất năm (mọi trạm, mọi năm)")
fig.tight_layout()
fig.savefig(OUT_DIR / "pt_3_3_mua_lu.png", dpi=150)
plt.show()
note("3", "Tháng hay xảy ra đỉnh lũ năm nhất", f"T{np.bincount(pm, minlength=13)[1:].argmax() + 1}")


# %% Phần 4 — Dự báo thời tiết so với tái phân tích
# Cặp biến có mặt ở cả dự báo ECMWF và tái phân tích (như Mục 3.1 của bài cơ sở). Biểu đồ mật độ điểm (hexbin)
# với đường 1:1, sai lệch trung bình theo tháng, khoảng cách Wasserstein theo lưu vực và bản đồ.
from scipy.stats import wasserstein_distance

PAIRS = {"Nhiệt độ 2 m": ("ERA5L_2m_temp_mean", "ECMWF_t2m"),
         "Điểm sương 2 m": ("ERA5L_2m_dp_temp_mean", "ECMWF_d2m"),
         "Bức xạ mặt trời": ("ERA5L_surf_net_solar_rad_mean", "ECMWF_ssrd"),
         "Lượng mưa": ("MSWEP_RR", "ECMWF_tp"),
         "Bốc hơi": ("GLEAM_ETA", "ECMWF_e")}
P = load_matrix([c for pair in PAIRS.values() for c in pair], "met")
pairs = {k: v for k, v in PAIRS.items() if v[0] in P and v[1] in P}
if not pairs:
    print("Không có cặp biến dự báo – tái phân tích nào trong dữ liệu, bỏ qua Phần 5")
else:
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.3))
    rows5 = []
    for ax, (name, (ra, fc)) in zip(np.atleast_1d(axes), pairs.items()):
        x = P[ra].loc["2003":"2017"].to_numpy().ravel()
        y = P[fc].loc["2003":"2017"].to_numpy().ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        sub = rng.choice(len(x), size=min(400_000, len(x)), replace=False)
        ax.hexbin(x[sub], y[sub], gridsize=60, bins="log", cmap="viridis", mincnt=1)
        lim = [np.percentile(np.r_[x, y], 0.5), np.percentile(np.r_[x, y], 99.5)]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel(f"Tái phân tích: {ra}", fontsize=8)
        ax.set_ylabel(f"Dự báo: {fc}", fontsize=8)
        ax.set_title(name)
        rows5.append({"Biến": name, "Tái phân tích": ra, "Dự báo": fc, "Trung bình tái phân tích": x.mean(),
                      "Trung bình dự báo": y.mean(), "Sai lệch TB (dự báo − tái phân tích)": y.mean() - x.mean(),
                      "Tương quan": np.corrcoef(x, y)[0, 1]})
    fig.suptitle("Hình 4.1. Dự báo ECMWF so với tái phân tích theo ngày, 2003–2017 (đường đỏ: 1:1; màu: mật độ điểm)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pt_4_1_du_bao_vs_tai_phan_tich.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.8))
    for ax, (name, (ra, fc)) in zip(np.atleast_1d(axes), pairs.items()):
        diff = (P[fc] - P[ra]).loc["2003":"2017"]
        mb = diff.groupby(diff.index.month).mean()                             # tháng × lưu vực
        ax.plot(months, mb.median(axis=1), marker="o", color="#b85450")
        ax.fill_between(months, mb.quantile(0.1, axis=1), mb.quantile(0.9, axis=1), alpha=0.25, color="#b85450")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(months)
        ax.set_title(name)
    fig.suptitle("Hình 4.2. Sai lệch trung bình theo tháng: dự báo − tái phân tích (đường: trung vị; dải: P10–P90 lưu vực)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pt_4_2_sai_lech_theo_thang.png", dpi=150)
    plt.show()

    wdist = pd.DataFrame({name: {b: wasserstein_distance(P[ra][b].loc["2003":"2017"].dropna(),
                                                         P[fc][b].loc["2003":"2017"].dropna())
                                 for b in P[ra].columns
                                 if P[ra][b].notna().any() and P[fc][b].notna().any()}
                          for name, (ra, fc) in pairs.items()})
    for r in rows5:
        r["Wasserstein trung vị qua lưu vực"] = wdist[r["Biến"]].median()
    print("Bảng 4.1. Dự báo so với tái phân tích (đơn vị gốc của từng biến; nếu hai trung bình khác hẳn bậc thì cần "
          "kiểm tra đơn vị)")
    display(pd.DataFrame(rows5).round(3))
    map_pairs = [n for n in ["Lượng mưa", "Nhiệt độ 2 m"] if n in wdist]
    fig, axes = plt.subplots(1, len(map_pairs), figsize=(8.5 * len(map_pairs), 6.5))
    for ax, n in zip(np.atleast_1d(axes), map_pairs):
        plot_map(ax, wdist[n], f"Wasserstein – {n}", cmap="magma_r")
    fig.suptitle("Hình 4.3. Mức khác biệt giữa dự báo và tái phân tích theo lưu vực (càng đậm càng khác)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pt_4_3_ban_do_sai_lech.png", dpi=150)
    plt.show()
    if "Lượng mưa" in wdist and "elev_mean" in A:
        rho = wdist["Lượng mưa"].corr(A["elev_mean"].reindex(wdist.index), method="spearman")
        note("4", "Tương quan Spearman: khác biệt mưa dự báo – tái phân tích với độ cao", f"{rho:.2f}")


# %% Phần 5 — Khác biệt giữa các giai đoạn
# Trục thời gian ba giai đoạn (kèm 364 ngày khởi động); bảng số mẫu mỗi giai đoạn; CDF lưu lượng chuẩn hóa và
# lượng mưa năm theo giai đoạn; xu hướng theo năm.
WARMUP = 364
fig, ax = plt.subplots(figsize=(16, 2.6))
colors = {"Huấn luyện": "#82b366", "Kiểm định": "#d6b656", "Kiểm tra": "#b85450"}
rows6 = []
for i, (name, (a, b)) in enumerate(PERIODS.items()):
    start, end = pd.Timestamp(f"{a}-01-01"), pd.Timestamp(f"{b}-12-31")
    ax.barh(0, (end - start).days, left=start, color=PERIOD_COLORS[name], height=0.5)
    ax.barh(0, WARMUP, left=start - pd.Timedelta(days=WARMUP), color=PERIOD_COLORS[name], alpha=0.3, height=0.5)
    ax.text(start + (end - start) / 2, 0, f"{name}\n{a}–{b}", ha="center", va="center", color="white",
            fontweight="bold")
    part = qmax.loc[start:end]
    rows6.append({"Giai đoạn": name, "Năm": f"{a}–{b}", "Số ngày": len(part),
                  "Ngày-trạm có qmax": int(part.notna().to_numpy().sum()),
                  "% thiếu": f"{part.isna().to_numpy().mean() * 100:.2f}%",
                  "qmax chuẩn hóa P99": float(np.nanpercentile((part / qmax.mean()).to_numpy(), 99))})
ax.set_yticks([])
ax.set_xlim(pd.Timestamp("2001-06-01"), pd.Timestamp("2018-03-01"))
ax.set_title("Hình 5.1. Ba giai đoạn chia tập theo thời gian (phần nhạt: 364 ngày quá khứ dùng làm đầu vào)")
fig.tight_layout()
fig.savefig(OUT_DIR / "pt_5_1_chia_tap.png", dpi=150)
plt.show()
print("Bảng 5.1. Số mẫu và độ thiếu theo giai đoạn")
display(pd.DataFrame(rows6).round(2))

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
for name, (a, b) in PERIODS.items():
    v = (qmax / qmax.mean()).loc[a:b].to_numpy().ravel()
    v = np.sort(v[np.isfinite(v) & (v > 0)])
    axes[0].plot(v, np.linspace(0, 1, len(v)), color=PERIOD_COLORS[name], label=name)
    if "MSWEP_RR" in M or "ERA5L_prec" in M:
        pc = M.get("MSWEP_RR", M.get("ERA5L_prec"))
        ann = pc.loc[a:b].resample("YS").sum(min_count=300).mean()
        s = np.sort(ann.dropna().values)
        axes[1].plot(s, np.linspace(0, 1, len(s)), color=PERIOD_COLORS[name], label=name)
axes[0].set_xscale("log")
axes[0].set_xlabel("qmax / qmax trung bình của trạm (log)")
axes[0].set_ylabel("Tỷ lệ tích lũy")
axes[0].set_title("Hình 5.2a. CDF lưu lượng chuẩn hóa theo giai đoạn")
axes[1].set_xlabel("Lượng mưa năm trung bình của lưu vực")
axes[1].set_title("Hình 5.2b. CDF lượng mưa năm theo giai đoạn")
yearly = (qmax / qmax.mean()).resample("YS").mean().median(axis=1)
axes[2].plot(yearly.index.year, yearly.values, marker="o", color="#333333")
for name, (a, b) in PERIODS.items():
    axes[2].axvspan(int(a) - 0.5, int(b) + 0.5, color=PERIOD_COLORS[name], alpha=0.2)
axes[2].set_xlabel("Năm")
axes[2].set_ylabel("Trung vị qua trạm của qmax năm chuẩn hóa")
axes[2].set_title("Hình 5.2c. Biến động lưu lượng giữa các năm")
for ax in axes[:2]:
    ax.legend()
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "pt_5_2_phan_phoi_giai_doan.png", dpi=150)
plt.show()
note("5", "Số ngày-trạm có qmax: huấn luyện / kiểm định / kiểm tra",
     " / ".join(f"{r['Ngày-trạm có qmax']:,}" for r in rows6))


# %% Phần 6 — Tổng kết
# Bảng các điểm chính gom từ Phần 1–5; lưu CSV và liệt kê các hình đã lưu để đưa vào báo cáo.
summary = pd.DataFrame(SUMMARY)
print("Bảng 6.1. Các điểm chính về bộ dữ liệu Extended LamaH-CE")
display(summary)
summary.to_csv(OUT_DIR / "pt_tong_ket.csv", index=False, encoding="utf-8-sig")
print("\nĐã lưu:", "pt_tong_ket.csv,", ", ".join(sorted(p.name for p in OUT_DIR.glob("pt_*.png"))))
