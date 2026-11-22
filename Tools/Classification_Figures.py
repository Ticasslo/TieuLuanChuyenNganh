# Sinh hình đánh giá theo bốn mức lưu lượng và sự kiện lũ (Chương 6, báo cáo tiến độ).
# Chạy từ gốc repo:
#   python Tools/Classification_Figures.py --eval-output <thư mục evaluation_output của Evaluation Version 13>
#       => gộp ma trận nhầm lẫn 4x4 qua 451 lưu vực thành bảng nhỏ trong Tables/, kiểm khớp bảng pooled đã lưu, rồi vẽ hình
#   python Tools/Classification_Figures.py
#       => chỉ vẽ hình từ bảng đã lưu trong repo
# Nguồn trong repo: Evaluation Version 9 (56 run A, B, C2), Version 12 (H, hạn 1-7), Version 13 (D).
# Mọi chữ trong hình tiếng Anh (dấu chấm thập phân); chú thích hình trong .md viết tiếng Việt.
#
# Mục lục:
#   Phần 1 - Hằng số, đọc bảng
#   Phần 2 - Gộp ma trận nhầm lẫn từ Output Kaggle (tùy chọn)
#   Phần 3 - Hình 1: ma trận nhầm lẫn 4x4 của năm lõi C2 và persistence
#   Phần 4 - Hình 2: mức rất cao, recall và precision
#   Phần 5 - Hình 3: bắt lũ chu kỳ hai năm, POD và FAR
#   Phần 6 - Hình 4: bước H theo hạn
#   Phần 7 - Hình 5: bước D, cửa sổ 365 so với 730 ngày

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Phần 1 - Hằng số, đọc bảng
ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "Workspace/06_Evaluation/01_Evaluation_LamaHCEExt"
ARC = EVAL / "Archive"
FIG = EVAL / "Figures"
TAB = EVAL / "Tables"
CONF = TAB / "Classification_confusion_pooled.csv"
LEVELS = ["low", "medium", "high", "very_high"]
LEVEL_LABELS = ["Low\n(< q30)", "Medium\n(q30-q80)", "High\n(q80-q98)", "Very high\n(> q98)"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12, "axes.titleweight": "bold"})

# Bảng màu Okabe-Ito, mỗi lõi một màu cố định ở mọi hình
COLOR = {"LSTM": "#0072B2", "GRU": "#009E73", "Transformer": "#CC79A7", "S4D": "#E69F00", "Mamba": "#D55E00",
         "B*": "#56B4E9", "Persistence": "#7F7F7F"}
C2 = [("SF_lstm_L2_x1_lr1e-3_C2", "LSTM"), ("SF_gru_L2_x1_lr1e-3_C2", "GRU"),
      ("SF_transformer_L2_x1_lr1e-3_C2", "Transformer"), ("SF_s4d_L4_x1_lr1e-3_C2", "S4D"),
      ("SF_mamba_L2_x1_lr1e-3_C2", "Mamba")]
BSTAR = ("SeqLSTM_Q_B3_P10S12", "B*")
PERS = ("Persistence_qmax", "Persistence")


def group(run):
    # Bỏ đuôi hạt giống: ba run cùng cấu hình về một nhóm.
    return re.sub(r"_seed_?\d+$", "", run)


def read(path):
    d = pd.read_csv(path)
    d["g"] = d.run.map(group)
    return d


def flood(d, T=2.0):
    # Cụm USWRC, mọi lưu vực, chu kỳ lặp lại T năm.
    return d[(d.return_period_years == T) & (d.grouping == "USWRC") & (d.cohort == "all")]


def stat(d, g, col):
    # Trung bình và độ lệch chuẩn mẫu qua ba hạt giống, đơn vị %.
    v = d.loc[d.g == g, col] * 100
    if v.empty:
        raise ValueError(f"Không có {g} trong bảng")
    return v.mean(), (v.std() if len(v) > 1 else 0.0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Đã lưu", (FIG / name).relative_to(ROOT))


def label_bars(ax, bars, err, fmt="{:.1f}"):
    # Ghi số ngay sau đầu thanh sai số để chữ không đè lên thanh sai số.
    for b, e in zip(bars, err):
        w = b.get_width()
        ax.text(w + e + 0.8, b.get_y() + b.get_height() / 2, fmt.format(w), va="center", ha="left", fontsize=9.5)


# %% Phần 2 - Gộp ma trận nhầm lẫn từ Output Kaggle (tùy chọn)
def build_confusion(eval_output):
    sources = [("C2_B", eval_output / "Evaluation_confusion.csv", ARC / "Version9/Tables/Evaluation_levels_pooled.csv")]
    sources += [(f"H_lead{k}", eval_output / f"H/Evaluation_H_confusion_lead{k}.csv",
                 ARC / f"Version12/H/Evaluation_H_levels_pooled_lead{k}.csv") for k in range(1, 8)]
    sources += [("D", eval_output / "D/Evaluation_D_confusion.csv", ARC / "Version13/D/Evaluation_D_levels_pooled.csv")]
    parts = []
    for part, path, pooled in sources:
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu {path}")
        c = pd.read_csv(path, dtype={"basin_id": str})
        if c.basin_id.nunique() != 451:
            raise ValueError(f"{path.name}: {c.basin_id.nunique()} lưu vực, cần 451")
        s = c.groupby(["run", "lead", "observed_level", "predicted_level"], sort=False).days.sum().reset_index()
        # Kiểm: tổng hàng, tổng cột, đường chéo khớp bảng pooled Evaluation đã lưu trong repo
        ref = pd.read_csv(pooled)
        for run, m in s.groupby("run"):
            mat = m.pivot(index="observed_level", columns="predicted_level", values="days").reindex(index=LEVELS, columns=LEVELS)
            r = ref[ref.run == run].set_index("level").reindex(LEVELS)
            if not (np.array_equal(mat.sum(axis=1).to_numpy(), r.observed_days.to_numpy())
                    and np.array_equal(mat.sum(axis=0).to_numpy(), r.predicted_days.to_numpy())
                    and np.array_equal(np.diag(mat.to_numpy()), r.correct.to_numpy())):
                raise ValueError(f"{path.name}: {run} không khớp {pooled.name}")
        s.insert(0, "part", part)
        parts.append(s)
    out = pd.concat(parts, ignore_index=True)
    TAB.mkdir(parents=True, exist_ok=True)
    out.to_csv(CONF, index=False)
    print(f"Đã lưu {CONF.relative_to(ROOT)}: {len(out)} dòng, khớp bảng pooled ở {out.groupby('part').run.nunique().sum()} run")


# %% Phần 3 - Hình 1: ma trận nhầm lẫn 4x4
def fig_confusion():
    conf = pd.read_csv(CONF)
    conf = conf[conf.part == "C2_B"].copy()
    conf["g"] = conf.run.map(group)
    panels = [*C2, PERS]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for ax, (g, name) in zip(axes.flat, panels):
        m = (conf[conf.g == g].groupby(["observed_level", "predicted_level"]).days.sum()
             .unstack().reindex(index=LEVELS, columns=LEVELS).to_numpy(dtype=float))
        pct = m / m.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{pct[i, j]:.0f}%", ha="center", va="center", fontsize=11,
                        color="white" if pct[i, j] > 55 else "black", fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(4), [s.split("\n")[0] for s in LEVEL_LABELS], fontsize=9.5)
        ax.set_yticks(range(4), [s.split("\n")[0] for s in LEVEL_LABELS], fontsize=9.5)
        ax.set_xlabel("Predicted level")
        ax.set_ylabel("Observed level")
        seeds = "3 seeds" if g != PERS[0] else "copy of yesterday"
        ax.set_title(f"{name} ({seeds})", color=COLOR[name])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Where do observed days end up? Rows sum to 100% (test 2014-2017, lead 1, 451 basins)",
                 x=0.02, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, "Classification_Confusion_C2.png")


# %% Phần 4 - Hình 2: mức rất cao
def fig_very_high():
    L = read(ARC / "Version9/Tables/Evaluation_levels_pooled.csv")
    vh = L[L.level == "very_high"]
    models = [*C2, BSTAR, PERS]
    y = np.arange(len(models))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, title in ((axes[0], "recall", "(a) Recall: very-high days caught"),
                           (axes[1], "precision", "(b) Precision: very-high alerts that were right")):
        m, s = zip(*(stat(vh, g, col) for g, _ in models))
        bars = ax.barh(y, m, xerr=s, color=[COLOR[n] for _, n in models], capsize=3, height=0.65)
        label_bars(ax, bars, s)
        ax.set_xlim(0, 85)
        ax.set_xlabel("%")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [n for _, n in models])
    fig.suptitle("Very-high flow days (> q98), test 2014-2017, lead 1 - mean of 3 seeds, bars = seed SD",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "Classification_VeryHigh_Level.png")


# %% Phần 5 - Hình 3: bắt lũ chu kỳ hai năm
def fig_events():
    E = flood(read(ARC / "Version9/Tables/Evaluation_events_pooled.csv"))
    T = flood(read(ARC / "Version9/Tables/Evaluation_events_pooled_tolerance1.csv"))
    models = [*C2, BSTAR]
    y = np.arange(len(models))[::-1]
    h = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, title in ((axes[0], "POD", "(a) POD: floods caught (higher = better)"),
                           (axes[1], "FAR", "(b) FAR: false alarms (lower = better)")):
        for tab, off, alpha, lab in ((E, h / 2, 1.0, "same onset day"), (T, -h / 2, 0.45, "onset within ±1 day")):
            m, s = zip(*(stat(tab, g, col) for g, _ in models))
            bars = ax.barh(y + off, m, h, xerr=s, color=[COLOR[n] for _, n in models], alpha=alpha, capsize=2, label=lab)
            label_bars(ax, bars, s, "{:.0f}")
        ax.set_xlim(0, 85)
        ax.set_xlabel("%")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [n for _, n in models])
    leg = axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    for hdl in leg.legend_handles:
        hdl.set_color("#555555")
    n = int((E.loc[E.g == PERS[0], ["hit", "missed"]].sum(axis=1)).iloc[0])
    fig.suptitle(f"Floods above the 2-year return level: {n:,} observed floods, test 2014-2017 - mean of 3 seeds",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "Classification_Flood_Events.png")


# %% Phần 6 - Hình 4: bước H theo hạn
def fig_h():
    leads = list(range(1, 8))
    Lh = {k: read(ARC / f"Version12/H/Evaluation_H_levels_pooled_lead{k}.csv") for k in leads}
    Eh = {k: flood(read(ARC / f"Version12/H/Evaluation_H_events_pooled_lead{k}.csv")) for k in leads}
    modes = [("perfect", "perfect weather", "#009E73"), ("gefs", "GEFS forecast", "#0072B2"),
             ("hybrid", "ECMWF HRES day 1", "#E69F00"), ("past", "past data only", "#7F7F7F")]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey="row")
    rows = (("recall", Lh, "Very-high recall (%)"), ("POD", Eh, "Flood POD, 2-year (%)"))
    for r, (col, tabs, ylab) in enumerate(rows):
        for c, (core, cname) in enumerate((("lstm", "LSTM"), ("mamba", "Mamba"))):
            ax = axes[r, c]
            for mode, lab, color in modes:
                g = f"SF_{core}_L2_x1_lr1e-3_H{mode}"
                vals = [stat(tabs[k][tabs[k].level == "very_high"] if col == "recall" else tabs[k], g, col) for k in leads]
                m = np.array([v[0] for v in vals]); s = np.array([v[1] for v in vals])
                ax.plot(leads, m, color=color, marker="o", lw=2, ms=4, label=lab)
                ax.fill_between(leads, m - s, m + s, color=color, alpha=0.15, lw=0)
            if col == "recall":
                p = [stat(tabs[k][tabs[k].level == "very_high"], PERS[0], col)[0] for k in leads]
                ax.plot(leads, p, color="black", ls=":", lw=1.3, label="persistence (copy of yesterday)")
            ax.set_title(f"({'abcd'[2 * r + c]}) {cname}: {ylab}", loc="left")
            ax.grid(alpha=0.3)
            ax.set_xlim(0.8, 7.2)
            ax.set_xticks(leads)
            if r == 1:
                ax.set_xlabel("Lead time (days)")
    axes[0, 1].legend(title="Weather input", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle("Multi-day forecasts (step H): how flood skill decays with lead time - mean of 3 seeds, band = seed SD",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "Classification_H_by_Lead.png")


# %% Phần 7 - Hình 5: bước D
def fig_d():
    Ld = read(ARC / "Version13/D/Evaluation_D_levels_pooled.csv")
    Ed = flood(read(ARC / "Version13/D/Evaluation_D_events_pooled.csv"))
    cores = [("lstm_L2", "LSTM"), ("mamba_L2", "Mamba"), ("s4d_L4", "S4D")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(cores))
    for ax, d, col, title in ((axes[0], Ld[Ld.level == "very_high"], "recall", "(a) Very-high recall (%)"),
                              (axes[1], Ed, "POD", "(b) Flood POD, 2-year, same day (%)")):
        for k, (tag, lab, alpha) in enumerate((("C2", "365-day window", 0.45), ("D", "730-day window", 1.0))):
            m, s = zip(*(stat(d, f"SF_{c}_x1_lr1e-3_{tag}", col) for c, _ in cores))
            bars = ax.bar(x + (k - 0.5) * 0.38, m, 0.38, yerr=s, capsize=3, alpha=alpha,
                          color=[COLOR[n] for _, n in cores], label=lab)
            for b, e in zip(bars, s):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e + 0.3, f"{b.get_height():.1f}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, [n for _, n in cores])
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 60)
    axes[1].set_ylim(0, 22)
    leg = axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    for hdl in leg.legend_handles:
        hdl.set_color("#555555")
    fig.suptitle("Longer input window (step D): 365 vs 730 days, test 2014-2017 - mean of 3 seeds",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "Classification_D_Window.png")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-output", type=Path, help="Thư mục evaluation_output tải từ Evaluation Version 13")
    args = ap.parse_args()
    if args.eval_output:
        build_confusion(args.eval_output)
    if not CONF.is_file():
        raise FileNotFoundError(f"Chưa có {CONF.relative_to(ROOT)}: chạy với --eval-output trước")
    fig_confusion()
    fig_very_high()
    fig_events()
    fig_h()
    fig_d()
