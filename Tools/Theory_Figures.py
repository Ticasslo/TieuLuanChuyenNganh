# Sinh hình minh họa cho ghi chú lý thuyết Document/06_Theory/ (SVG, chỉ dùng thư viện chuẩn).
# Chạy: python Tools/Theory_Figures.py (từ gốc repo). Hình CDF dùng số tự đặt để minh họa; hình thước NSE dùng số thật của bài.
#
# Mục lục:
#   Phần 1 — Hằng số và hàm vẽ SVG dùng chung
#   Phần 2 — Hình của Phần 2 BiasCast: đường CDF bậc thang 5 lưu vực, so sánh hai mô hình 451 lưu vực,
#            thước NSE trung vị giữa mốc dưới và mốc trên (số thật từ Bảng F1 của bài),
#            ba kiểu sai cùng NSE (lệch đều, biên độ thấp, chậm nhịp) và persistence

from pathlib import Path

# %% Phần 1 — Hằng số và hàm vẽ SVG dùng chung
OUT_DIR = Path(__file__).resolve().parent.parent / "Document" / "06_Theory" / "Figures"
W, H = 640, 420                      # kích thước hình (px)
LEFT, RIGHT, TOP, BOTTOM = 70, 30, 40, 60
FONT = "font-family='Arial, sans-serif'"


def vn(v, fmt):
    """Định dạng số theo kiểu Việt Nam (dấu phẩy thập phân, dấu trừ thật) — chỉ dùng cho chữ hiển thị."""
    return format(v, fmt).replace(".", ",").replace("-", "−")


def make_axes(x_min, x_max, x_ticks, title, x_label, y_label):
    """Trả về (danh sách phần tử SVG, hàm đổi tọa độ dữ liệu sang pixel) cho trục x, y (y từ 0 tới 1)."""
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM

    def to_px(x, y):
        return LEFT + (x - x_min) / (x_max - x_min) * pw, TOP + (1 - y) * ph

    el = [f"<rect width='{W}' height='{H}' fill='white'/>",
          f"<text x='{W / 2}' y='22' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>{title}</text>"]
    for y in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        _, py = to_px(x_min, y)
        el.append(f"<line x1='{LEFT}' y1='{py}' x2='{W - RIGHT}' y2='{py}' stroke='#e6e6e6'/>")
        el.append(f"<text x='{LEFT - 8}' y='{py + 4}' text-anchor='end' font-size='12' {FONT}>{vn(y, '.1f')}</text>")
    for x in x_ticks:
        px, _ = to_px(x, 0)
        el.append(f"<line x1='{px}' y1='{TOP}' x2='{px}' y2='{H - BOTTOM}' stroke='#e6e6e6'/>")
        el.append(f"<text x='{px}' y='{H - BOTTOM + 18}' text-anchor='middle' font-size='12' {FONT}>{vn(x, 'g')}</text>")
    el.append(f"<line x1='{LEFT}' y1='{H - BOTTOM}' x2='{W - RIGHT}' y2='{H - BOTTOM}' stroke='black'/>")
    el.append(f"<line x1='{LEFT}' y1='{TOP}' x2='{LEFT}' y2='{H - BOTTOM}' stroke='black'/>")
    el.append(f"<text x='{LEFT + pw / 2}' y='{H - 15}' text-anchor='middle' font-size='13' {FONT}>{x_label}</text>")
    el.append(f"<text x='18' y='{TOP + ph / 2}' text-anchor='middle' font-size='13' {FONT} "
              f"transform='rotate(-90 18 {TOP + ph / 2})'>{y_label}</text>")
    return el, to_px


def step_path(xs, to_px, x_min):
    """Đường bậc thang của CDF thực nghiệm: tại mỗi giá trị đã sắp xếp, y nhảy thêm 1/n."""
    n, pts, y = len(xs), [], 0.0
    pts.append(to_px(x_min, 0))
    for i, x in enumerate(xs):
        pts.append(to_px(x, y))
        y = (i + 1) / n
        pts.append(to_px(x, y))
    return "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)


def save(name, elements, height=H, width=W):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>\n"
           + "\n".join(elements) + "\n</svg>\n")
    (OUT_DIR / name).write_text(svg, encoding="utf-8")
    print("Đã ghi", OUT_DIR / name)


# %% Phần 2 — Hình của Phần 2 BiasCast
def fig_cdf_steps():
    """5 lưu vực, NSE −3,00; 0,55; 0,60; 0,65; 0,70: mỗi lưu vực làm đường nhảy lên 0,2."""
    nse = sorted([0.70, 0.65, 0.60, 0.55, -3.00])
    x_min = -3.5
    el, to_px = make_axes(x_min, 1.0, [-3, -2, -1, 0, 0.5, 1], "Đường CDF của 5 lưu vực (minh họa)",
                          "NSE của lưu vực (trục x)", "Tỷ lệ lưu vực có NSE ≤ x (trục y)")
    el.append(f"<path d='{step_path(nse, to_px, x_min)}' fill='none' stroke='#1f77b4' stroke-width='2.5'/>")
    for i, x in enumerate(nse):
        px, py = to_px(x, (i + 1) / len(nse))
        el.append(f"<circle cx='{px}' cy='{py}' r='4' fill='#1f77b4'/>")
    labels = {-3.00: "lưu vực E: −3,00 → y = 0,2", 0.60: "trung vị 0,60"}
    px, py = to_px(-3.0, 0.2)
    el.append(f"<text x='{px + 8}' y='{py - 8}' font-size='12' {FONT}>{labels[-3.00]}</text>")
    mx, _ = to_px(0.60, 0.5)
    _, y0 = to_px(0, 0.5)
    el.append(f"<line x1='{LEFT}' y1='{y0}' x2='{mx}' y2='{y0}' stroke='#d62728' stroke-dasharray='5,4'/>")
    el.append(f"<line x1='{mx}' y1='{y0}' x2='{mx}' y2='{H - BOTTOM}' stroke='#d62728' stroke-dasharray='5,4'/>")
    el.append(f"<text x='{mx - 6}' y='{y0 - 8}' text-anchor='end' font-size='12' fill='#d62728' {FONT}>y = 0,5 cắt đường tại NSE = 0,60 (trung vị)</text>")
    save("BiasCast2_CDF_Steps.svg", el)


def fig_cdf_compare():
    """Hai mô hình trên 451 lưu vực; NSE tự đặt bằng hàm của thứ hạng p, giá trị < −1 vẽ dồn ở mép trái."""
    n, x_min = 451, -1.0
    ps = [(i + 0.5) / n for i in range(n)]
    good = sorted(max(x_min, 0.80 - 1.4 * (1 - p) ** 4) for p in ps)
    weak = sorted(max(x_min, 0.60 - 2.0 * (1 - p) ** 3) for p in ps)
    el, to_px = make_axes(x_min, 1.0, [-1, -0.5, 0, 0.5, 1], "So sánh hai mô hình trên 451 lưu vực (minh họa)",
                          "NSE (giá trị &lt; −1 vẽ dồn ở mép trái)", "Tỷ lệ lưu vực có NSE ≤ x")
    for xs, color, name, ly in [(weak, "#9e9e9e", "Mô hình B (kém hơn)", 70), (good, "#ff7f0e", "Mô hình A (tốt hơn)", 50)]:
        el.append(f"<path d='{step_path(xs, to_px, x_min)}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        el.append(f"<line x1='{LEFT + 15}' y1='{ly}' x2='{LEFT + 45}' y2='{ly}' stroke='{color}' stroke-width='3'/>")
        el.append(f"<text x='{LEFT + 52}' y='{ly + 4}' font-size='12' {FONT}>{name}</text>")
        med = xs[n // 2]
        mx, my = to_px(med, 0.5)
        el.append(f"<circle cx='{mx}' cy='{my}' r='5' fill='{color}'/>")
        el.append(f"<text x='{mx + 8}' y='{my + 16}' font-size='12' {FONT}>trung vị {vn(med, '.2f')}</text>")
    _, y0 = to_px(0, 0.5)
    el.append(f"<line x1='{LEFT}' y1='{y0}' x2='{W - RIGHT}' y2='{y0}' stroke='#d62728' stroke-dasharray='5,4'/>")
    tx, ty = to_px(-0.95, 0.12)
    el.append(f"<text x='{tx}' y='{ty}' font-size='12' {FONT}>đuôi trái: lưu vực tệ nhất</text>")
    ax, ay = to_px(0.05, 0.75)
    el.append(f"<text x='{ax}' y='{ay}' font-size='12' {FONT}>đường càng về bên phải → càng tốt</text>")
    save("BiasCast2_CDF_Compare.svg", el)


def fig_benchmark_ruler():
    """Thước NSE trung vị: mốc dưới, mốc trên và các thí nghiệm chính (BiasCast Bảng F1, PDF trang 27)."""
    items = [  # (tên, NSE trung vị, màu, độ lệch dọc của nhãn so với thước, căn chữ)
        ("Mốc dưới: chỉ dự báo (𝒟FC)", 0.39, "#757575", 45, "start"),
        ("Học chuyển giao, mọi trọng số", 0.41, "#1f77b4", -75, "start"),
        ("Encoder–Decoder LSTM", 0.57, "#9467bd", 80, "middle"),
        ("Sequential Forecast LSTM", 0.63, "#ff7f0e", -40, "end"),
        ("Mốc trên: tái phân tích (𝒟RA)", 0.69, "#424242", 45, "end"),
        ("Sequential + lưu lượng quá khứ", 0.71, "#d62728", -110, "end"),
    ]
    x_min, x_max, h, yb = 0.35, 0.75, 340, 170
    pw = W - LEFT - RIGHT

    def px(x):
        return LEFT + (x - x_min) / (x_max - x_min) * pw

    el = [f"<rect width='{W}' height='{h}' fill='white'/>",
          f"<text x='{W / 2}' y='24' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>NSE trung vị trên 451 lưu vực (BiasCast, Bảng F1)</text>",
          f"<rect x='{px(0.39)}' y='{yb - 8}' width='{px(0.69) - px(0.39)}' height='16' fill='#fff2cc'/>",
          f"<line x1='{LEFT}' y1='{yb}' x2='{W - RIGHT}' y2='{yb}' stroke='black' stroke-width='1.5'/>"]
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        el.append(f"<line x1='{px(t)}' y1='{yb - 4}' x2='{px(t)}' y2='{yb + 4}' stroke='black'/>")
        el.append(f"<text x='{px(t)}' y='{h - 15}' text-anchor='middle' font-size='11' fill='#555' {FONT}>{vn(t, '.2f')}</text>")
    for name, v, color, dy, anchor in items:
        x = px(v)
        ly = yb + dy
        el.append(f"<line x1='{x}' y1='{yb}' x2='{x}' y2='{ly + (6 if dy < 0 else -14)}' stroke='{color}'/>")
        el.append(f"<circle cx='{x}' cy='{yb}' r='6' fill='{color}'/>")
        tx = x - 4 if anchor == "end" else (x + 4 if anchor == "start" else x)
        el.append(f"<text x='{tx}' y='{ly}' text-anchor='{anchor}' font-size='12' fill='{color}' {FONT}>{name}: {vn(v, '.2f')}</text>")
    el.append(f"<text x='{(px(0.39) + px(0.69)) / 2}' y='{yb + 115}' text-anchor='middle' font-size='11' fill='#8a6d00' {FONT}>vùng vàng: khoảng cách giữa mốc dưới và mốc trên</text>")
    save("BiasCast2_Benchmark_Ruler.svg", el, height=h)


OBS_10 = [2, 2, 3, 8, 12, 7, 4, 3, 2, 2]     # chuỗi qmax đo 10 ngày dùng chung cho Mục 2.6 (minh họa)


def series_panels(name, title, panels, y_max=15):
    """Vẽ các ô nhỏ cạnh nhau; mỗi ô: đường đo thật (đen) và một đường dự báo (màu), kèm chú thích dưới ô."""
    pw, ph, gap, top, left = 250, 190, 40, 55, 40
    content = left + len(panels) * pw + (len(panels) - 1) * gap + 20
    width = max(content, 460)  # đủ chỗ cho tiêu đề khi chỉ có 1 ô; ô được canh giữa
    left += (width - content) / 2
    height = top + ph + 95
    el = [f"<rect width='{width}' height='{height}' fill='white'/>",
          f"<text x='{width / 2}' y='24' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>{title}</text>"]
    for k, (sub, sim, color, notes) in enumerate(panels):
        x0 = left + k * (pw + gap)

        def px(i, v):
            return x0 + i / (len(OBS_10) - 1) * pw, top + (1 - v / y_max) * ph

        el.append(f"<rect x='{x0}' y='{top}' width='{pw}' height='{ph}' fill='none' stroke='#bbbbbb'/>")
        for v in (0, 5, 10, 15):
            _, y = px(0, v)
            el.append(f"<text x='{x0 - 5}' y='{y + 4}' text-anchor='end' font-size='10' fill='#666' {FONT}>{v}</text>")
        for i in range(len(OBS_10)):
            x, _ = px(i, 0)
            el.append(f"<text x='{x}' y='{top + ph + 14}' text-anchor='middle' font-size='10' fill='#666' {FONT}>{i + 1}</text>")
        for data, col, wd in ((OBS_10, "black", 2.5), (sim, color, 2.5)):
            pts = " ".join(f"{px(i, v)[0]:.1f},{px(i, v)[1]:.1f}" for i, v in enumerate(data))
            el.append(f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='{wd}'/>")
        el.append(f"<text x='{x0 + pw / 2}' y='{top - 8}' text-anchor='middle' font-size='13' font-weight='bold' fill='{color}' {FONT}>{sub}</text>")
        for j, line in enumerate(notes):
            el.append(f"<text x='{x0 + pw / 2}' y='{top + ph + 34 + 16 * j}' text-anchor='middle' font-size='12' {FONT}>{line}</text>")
    el.append(f"<text x='{width / 2}' y='{height - 6}' text-anchor='middle' font-size='11' fill='#555' {FONT}>đen: qmax đo thật; trục ngang: ngày 1–10; trục dọc: mm/ngày</text>")
    save(name, el, height=height, width=width)


def fig_three_errors():
    """Ba dự báo có NSE gần bằng nhau nhưng sai theo ba kiểu khác nhau; KGE tách được kiểu sai."""
    shift = [v + 2.2 for v in OBS_10]
    flat = [4.5 + 0.3 * (v - 4.5) for v in OBS_10]
    late = [0.2 * v + 0.8 * p for v, p in zip(OBS_10, [2] + OBS_10[:-1])]
    series_panels("BiasCast2_Three_Errors.svg", "Ba dự báo cùng NSE khoảng 0,5 nhưng sai theo ba kiểu (minh họa)", [
        ("Lệch đều: luôn cao hơn 2,2", shift, "#1f77b4", ["NSE 0,54", "r = 1,00   α = 1,00   β = 1,49"]),
        ("Biên độ thấp: quá phẳng", flat, "#ff7f0e", ["NSE 0,51", "r = 1,00   α = 0,30   β = 1,00"]),
        ("Chậm nhịp: đỉnh trễ 1 ngày", late, "#2ca02c", ["NSE 0,52", "r = 0,75   α = 0,94   β = 1,00"]),
    ])


def fig_persistence():
    """Persistence: lấy qmax đo hôm trước làm dự báo cho hôm nay (ngày 1 lấy 2)."""
    pers = [2] + OBS_10[:-1]
    series_panels("BiasCast2_Persistence.svg", "Persistence: ngày mai bằng hôm nay (minh họa)", [
        ("Persistence", pers, "#d62728", ["NSE 0,25", "đường đỏ là đường đen dịch phải 1 ngày"]),
    ])


if __name__ == "__main__":
    fig_cdf_steps()
    fig_cdf_compare()
    fig_benchmark_ruler()
    fig_three_errors()
    fig_persistence()
