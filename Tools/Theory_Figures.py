# Sinh hình minh họa cho ghi chú lý thuyết Document/06_Theory/ (SVG, chỉ dùng thư viện chuẩn).
# Chạy: python Tools/Theory_Figures.py (từ gốc repo). Hình CDF dùng số tự đặt để minh họa; hình thước NSE dùng số thật của bài.
# Quy ước: mọi chữ hiển thị trong hình viết tiếng Anh (số dùng dấu chấm thập phân); chú thích hình trong ghi chú .md viết tiếng Việt.
#
# Mục lục:
#   Phần 1 - Hằng số và hàm vẽ SVG dùng chung
#   Phần 2 - Hình của Phần 2 BiasCast: đường CDF bậc thang 5 lưu vực, so sánh hai mô hình 451 lưu vực,
#            thước NSE trung vị giữa mốc dưới và mốc trên (số thật từ Bảng F1 của bài),
#            ba kiểu sai cùng NSE (lệch đều, biên độ thấp, chậm nhịp) và persistence
#   Phần 3 - Hình của Phần 3 BiasCast: cách dựng đường duy trì lưu lượng từ 10 ngày, bốn mức lưu lượng
#            trên đường duy trì của một lưu vực, ba chỉ số %BiasFHV, %BiasFMS, %BiasFLV,
#            ngưỡng chu kỳ lặp lại từ đỉnh năm (Gumbel, L-moments), ghép sự kiện lũ trong cửa sổ ±1 ngày,
#            bảng 2x2 chấm theo trận và ví dụ cửa sổ ±1/±2 ngày, trục năm của các kỳ dữ liệu
#   Phần 4 - Hình của Phần 4 BiasCast: ghép biến và đổi trục trên một lô thu nhỏ,
#            thuộc tính nhúng một lần rồi lặp theo ngày, kích thước tensor của một lô thật
#   Phần 5 - Hình của Phần 5 BiasCast: dữ liệu từng thí nghiệm đọc trong khung 365 ngày (một pha hay hai pha),
#            Sequential Forecast LSTM so với Encoder-Decoder LSTM (cách nối pha quá khứ và ngày t)

import math
from pathlib import Path
from statistics import NormalDist

# %% Phần 1 - Hằng số và hàm vẽ SVG dùng chung
OUT_DIR = Path(__file__).resolve().parent.parent / "Document" / "06_Theory" / "Figures"
W, H = 640, 420                      # kích thước hình (px)
LEFT, RIGHT, TOP, BOTTOM = 70, 30, 40, 60
FONT = "font-family='Arial, sans-serif'"


def num(v, fmt):
    # Định dạng số hiển thị trong hình: dấu chấm thập phân, dấu trừ thật.
    return format(v, fmt).replace("-", "−")


def make_axes(x_min, x_max, x_ticks, title, x_label, y_label):
    # Trả về (danh sách phần tử SVG, hàm đổi tọa độ dữ liệu sang pixel) cho trục x, y (y từ 0 tới 1).
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM

    def to_px(x, y):
        return LEFT + (x - x_min) / (x_max - x_min) * pw, TOP + (1 - y) * ph

    el = [f"<rect width='{W}' height='{H}' fill='white'/>",
          f"<text x='{W / 2}' y='22' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>{title}</text>"]
    for y in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        _, py = to_px(x_min, y)
        el.append(f"<line x1='{LEFT}' y1='{py}' x2='{W - RIGHT}' y2='{py}' stroke='#e6e6e6'/>")
        el.append(f"<text x='{LEFT - 8}' y='{py + 4}' text-anchor='end' font-size='12' {FONT}>{num(y, '.1f')}</text>")
    for x in x_ticks:
        px, _ = to_px(x, 0)
        el.append(f"<line x1='{px}' y1='{TOP}' x2='{px}' y2='{H - BOTTOM}' stroke='#e6e6e6'/>")
        el.append(f"<text x='{px}' y='{H - BOTTOM + 18}' text-anchor='middle' font-size='12' {FONT}>{num(x, 'g')}</text>")
    el.append(f"<line x1='{LEFT}' y1='{H - BOTTOM}' x2='{W - RIGHT}' y2='{H - BOTTOM}' stroke='black'/>")
    el.append(f"<line x1='{LEFT}' y1='{TOP}' x2='{LEFT}' y2='{H - BOTTOM}' stroke='black'/>")
    el.append(f"<text x='{LEFT + pw / 2}' y='{H - 15}' text-anchor='middle' font-size='13' {FONT}>{x_label}</text>")
    el.append(f"<text x='18' y='{TOP + ph / 2}' text-anchor='middle' font-size='13' {FONT} "
              f"transform='rotate(-90 18 {TOP + ph / 2})'>{y_label}</text>")
    return el, to_px


def step_path(xs, to_px, x_min):
    # Đường bậc thang của CDF thực nghiệm: tại mỗi giá trị đã sắp xếp, y nhảy thêm 1/n.
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


# %% Phần 2 - Hình của Phần 2 BiasCast
def fig_cdf_steps():
    # 5 lưu vực, NSE −3,00; 0,55; 0,60; 0,65; 0,70: mỗi lưu vực làm đường nhảy lên 0,2.
    nse = sorted([0.70, 0.65, 0.60, 0.55, -3.00])
    x_min = -3.5
    el, to_px = make_axes(x_min, 1.0, [-3, -2, -1, 0, 0.5, 1], "CDF of 5 basins (illustration)",
                          "Basin NSE (x-axis)", "Fraction of basins with NSE ≤ x (y-axis)")
    el.append(f"<path d='{step_path(nse, to_px, x_min)}' fill='none' stroke='#1f77b4' stroke-width='2.5'/>")
    for i, x in enumerate(nse):
        px, py = to_px(x, (i + 1) / len(nse))
        el.append(f"<circle cx='{px}' cy='{py}' r='4' fill='#1f77b4'/>")
    labels = {-3.00: "basin E: −3.00 → y = 0.2", 0.60: "median 0.60"}
    px, py = to_px(-3.0, 0.2)
    el.append(f"<text x='{px + 8}' y='{py - 8}' font-size='12' {FONT}>{labels[-3.00]}</text>")
    mx, _ = to_px(0.60, 0.5)
    _, y0 = to_px(0, 0.5)
    el.append(f"<line x1='{LEFT}' y1='{y0}' x2='{mx}' y2='{y0}' stroke='#d62728' stroke-dasharray='5,4'/>")
    el.append(f"<line x1='{mx}' y1='{y0}' x2='{mx}' y2='{H - BOTTOM}' stroke='#d62728' stroke-dasharray='5,4'/>")
    el.append(f"<text x='{mx - 6}' y='{y0 - 8}' text-anchor='end' font-size='12' fill='#d62728' {FONT}>y = 0.5 crosses the curve at NSE = 0.60 (median)</text>")
    save("BiasCast2_CDF_Steps.svg", el)


def fig_cdf_compare():
    # Hai mô hình trên 451 lưu vực; NSE tự đặt bằng hàm của thứ hạng p, giá trị < −1 vẽ dồn ở mép trái.
    n, x_min = 451, -1.0
    ps = [(i + 0.5) / n for i in range(n)]
    good = sorted(max(x_min, 0.80 - 1.4 * (1 - p) ** 4) for p in ps)
    weak = sorted(max(x_min, 0.60 - 2.0 * (1 - p) ** 3) for p in ps)
    el, to_px = make_axes(x_min, 1.0, [-1, -0.5, 0, 0.5, 1], "Two models over 451 basins (illustration)",
                          "NSE (values &lt; −1 drawn at the left edge)", "Fraction of basins with NSE ≤ x")
    for xs, color, name, ly in [(weak, "#9e9e9e", "Model B (weaker)", 70), (good, "#ff7f0e", "Model A (better)", 50)]:
        el.append(f"<path d='{step_path(xs, to_px, x_min)}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        el.append(f"<line x1='{LEFT + 15}' y1='{ly}' x2='{LEFT + 45}' y2='{ly}' stroke='{color}' stroke-width='3'/>")
        el.append(f"<text x='{LEFT + 52}' y='{ly + 4}' font-size='12' {FONT}>{name}</text>")
        med = xs[n // 2]
        mx, my = to_px(med, 0.5)
        el.append(f"<circle cx='{mx}' cy='{my}' r='5' fill='{color}'/>")
        el.append(f"<text x='{mx + 8}' y='{my + 16}' font-size='12' {FONT}>median {num(med, '.2f')}</text>")
    _, y0 = to_px(0, 0.5)
    el.append(f"<line x1='{LEFT}' y1='{y0}' x2='{W - RIGHT}' y2='{y0}' stroke='#d62728' stroke-dasharray='5,4'/>")
    tx, ty = to_px(-0.95, 0.12)
    el.append(f"<text x='{tx}' y='{ty}' font-size='12' {FONT}>left tail: worst basins</text>")
    ax, ay = to_px(0.05, 0.75)
    el.append(f"<text x='{ax}' y='{ay}' font-size='12' {FONT}>further right → better</text>")
    save("BiasCast2_CDF_Compare.svg", el)


def fig_benchmark_ruler():
    # Thước NSE trung vị: mốc dưới, mốc trên và các thí nghiệm chính (BiasCast Bảng F1, PDF trang 27).
    items = [  # (tên, NSE trung vị, màu, độ lệch dọc của nhãn so với thước, căn chữ)
        ("Lower benchmark: forecast only (𝒟FC)", 0.39, "#757575", 45, "start"),
        ("Transfer learning, all weights", 0.41, "#1f77b4", -75, "start"),
        ("Transfer learning, embedding only", 0.44, "#17becf", -145, "start"),
        ("Encoder–Decoder LSTM", 0.57, "#9467bd", 80, "middle"),
        ("Sequential Forecast LSTM", 0.63, "#ff7f0e", -40, "end"),
        ("Upper benchmark: reanalysis (𝒟RA)", 0.69, "#424242", 110, "end"),
        ("Sequential + past discharge", 0.71, "#d62728", -110, "end"),
    ]
    x_min, x_max, h, yb = 0.35, 0.75, 375, 205
    pw = W - LEFT - RIGHT

    def px(x):
        return LEFT + (x - x_min) / (x_max - x_min) * pw

    el = [f"<rect width='{W}' height='{h}' fill='white'/>",
          f"<text x='{W / 2}' y='24' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>Median NSE over 451 basins (BiasCast, Table F1)</text>",
          f"<rect x='{px(0.39)}' y='{yb - 8}' width='{px(0.69) - px(0.39)}' height='16' fill='#fff2cc'/>",
          f"<line x1='{LEFT}' y1='{yb}' x2='{W - RIGHT}' y2='{yb}' stroke='black' stroke-width='1.5'/>"]
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        el.append(f"<line x1='{px(t)}' y1='{yb - 4}' x2='{px(t)}' y2='{yb + 4}' stroke='black'/>")
        el.append(f"<text x='{px(t)}' y='{h - 15}' text-anchor='middle' font-size='11' fill='#555' {FONT}>{num(t, '.2f')}</text>")
    for name, v, color, dy, anchor in items:
        x = px(v)
        ly = yb + dy
        el.append(f"<line x1='{x}' y1='{yb}' x2='{x}' y2='{ly + (6 if dy < 0 else -14)}' stroke='{color}'/>")
        el.append(f"<circle cx='{x}' cy='{yb}' r='6' fill='{color}'/>")
        tx = x - 4 if anchor == "end" else (x + 4 if anchor == "start" else x)
        el.append(f"<text x='{tx}' y='{ly}' text-anchor='{anchor}' font-size='12' fill='{color}' {FONT}>{name}: {num(v, '.2f')}</text>")
    el.append(f"<text x='{(px(0.39) + px(0.69)) / 2}' y='{yb + 140}' text-anchor='middle' font-size='11' fill='#8a6d00' {FONT}>yellow band: gap between lower and upper benchmark</text>")
    save("BiasCast2_Benchmark_Ruler.svg", el, height=h)


OBS_10 = [2, 2, 3, 8, 12, 7, 4, 3, 2, 2]     # chuỗi qmax đo 10 ngày dùng chung cho Mục 2.6 (minh họa)


def series_panels(name, title, panels, y_max=15):
    # Vẽ các ô nhỏ cạnh nhau; mỗi ô: đường đo thật (đen) và một đường dự báo (màu), kèm chú thích dưới ô.
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
    el.append(f"<text x='{width / 2}' y='{height - 6}' text-anchor='middle' font-size='11' fill='#555' {FONT}>black: observed qmax; x-axis: day 1–10; y-axis: mm/day</text>")
    save(name, el, height=height, width=width)


def fig_three_errors():
    # Ba dự báo có NSE gần bằng nhau nhưng sai theo ba kiểu khác nhau; KGE tách được kiểu sai.
    shift = [v + 2.2 for v in OBS_10]
    flat = [4.5 + 0.3 * (v - 4.5) for v in OBS_10]
    late = [0.2 * v + 0.8 * p for v, p in zip(OBS_10, [2] + OBS_10[:-1])]
    series_panels("BiasCast2_Three_Errors.svg", "Three forecasts with NSE ≈ 0.5 but three kinds of error (illustration)", [
        ("Constant offset: always +2.2", shift, "#1f77b4", ["NSE 0.54", "r = 1.00   α = 1.00   β = 1.49"]),
        ("Low amplitude: too flat", flat, "#ff7f0e", ["NSE 0.51", "r = 1.00   α = 0.30   β = 1.00"]),
        ("Lagged timing: peak 1 day late", late, "#2ca02c", ["NSE 0.52", "r = 0.75   α = 0.94   β = 1.00"]),
    ])


def fig_persistence():
    # Persistence: lấy qmax đo hôm trước làm dự báo cho hôm nay (ngày 1 lấy 2).
    pers = [2] + OBS_10[:-1]
    series_panels("BiasCast2_Persistence.svg", "Persistence: tomorrow equals today (illustration)", [
        ("Persistence", pers, "#d62728", ["NSE 0.25", "red = black shifted right by 1 day"]),
    ])


# %% Phần 3 - Hình của Phần 3 BiasCast
QMAX_10 = [1.2, 0.8, 5.0, 2.1, 1.0, 0.9, 3.4, 1.5, 0.7, 1.1]   # 10 ngày qmax của ví dụ Mục 3.5 (mm/ngày)
LEVELS = [  # (tên, p đầu, p cuối, màu nền, số ngày trung bình mỗi năm) - bốn mức của 03_Pipeline.md Mục 6.2
    ("Very high", 0.0, 0.02, "#f4cccc", 7),
    ("High", 0.02, 0.2, "#fce5cd", 66),
    ("Medium", 0.2, 0.7, "#d9ead3", 183),
    ("Low", 0.7, 1.0, "#cfe2f3", 110),
]
ARROW = ("<defs><marker id='arr' markerWidth='10' markerHeight='8' refX='9' refY='4' orient='auto'>"
         "<path d='M0,0 L10,4 L0,8 z' fill='#444'/></marker></defs>")


def rank_color(rank, n):
    # màu theo thứ hạng: lớn nhất xanh đậm, nhỏ nhất xanh nhạt; một ngày giữ cùng màu ở cả hai ô
    t = rank / (n - 1)
    return f"rgb({int(20 + t * 170)},{int(60 + t * 150)},{int(140 + t * 100)})"


def lognormal_fdc(mu, sigma, n=1000):
    # đường duy trì tự đặt (log lưu lượng phân phối chuẩn): trả về (danh sách p, danh sách q giảm dần)
    ps = [(i + 0.5) / n for i in range(n)]
    return ps, [math.exp(mu + sigma * NormalDist().inv_cdf(1 - p)) for p in ps]


def make_fdc_axes(width, height, top, y_min, y_max, y_ticks, title, y_label):
    # trục x là xác suất vượt 0-1 (tuyến tính), trục y là lưu lượng theo thang log
    pw, ph = width - LEFT - RIGHT, height - top - BOTTOM
    span = math.log10(y_max) - math.log10(y_min)

    def to_px(p, q):
        return LEFT + p * pw, top + (math.log10(y_max) - math.log10(q)) / span * ph

    el = [f"<rect width='{width}' height='{height}' fill='white'/>", ARROW,
          f"<text x='{width / 2}' y='22' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>{title}</text>"]
    for q in y_ticks:
        _, py = to_px(0, q)
        el.append(f"<line x1='{LEFT}' y1='{py}' x2='{width - RIGHT}' y2='{py}' stroke='#e6e6e6'/>")
        el.append(f"<text x='{LEFT - 8}' y='{py + 4}' text-anchor='end' font-size='12' {FONT}>{num(q, 'g')}</text>")
    for i in range(11):
        px, _ = to_px(i / 10, y_min)
        el.append(f"<text x='{px}' y='{height - BOTTOM + 18}' text-anchor='middle' font-size='12' {FONT}>{num(i / 10, 'g')}</text>")
    el.append(f"<rect x='{LEFT}' y='{top}' width='{pw}' height='{ph}' fill='none' stroke='black'/>")
    el.append(f"<text x='{LEFT + pw / 2}' y='{height - 15}' text-anchor='middle' font-size='13' {FONT}>"
              f"Exceedance probability p (fraction of days with flow ≥ y)</text>")
    el.append(f"<text x='18' y='{top + ph / 2}' text-anchor='middle' font-size='13' {FONT} "
              f"transform='rotate(-90 18 {top + ph / 2})'>{y_label}</text>")
    return el, to_px


def curve_path(ps, qs, to_px):
    return "M " + " L ".join(f"{to_px(p, q)[0]:.1f},{to_px(p, q)[1]:.1f}" for p, q in zip(ps, qs))


def fig_fdc_build():
    # Ô trái: 10 ngày theo thời gian; ô phải: cùng 10 cột xếp từ lớn tới nhỏ, dưới cột ghi ngày gốc và p = i / n
    n, y_max = len(QMAX_10), 6.0
    order = sorted(range(n), key=lambda d: -QMAX_10[d])          # order[i] = ngày đứng thứ i + 1
    rank = {d: i for i, d in enumerate(order)}
    width, height, top, ph, pw, gap, left = 820, 420, 75, 220, 320, 110, 60
    el = [f"<rect width='{width}' height='{height}' fill='white'/>", ARROW,
          f"<text x='{width / 2}' y='24' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>"
          f"Building a flow duration curve from 10 days (illustration)</text>"]
    panels = [("Step 1: daily qmax in time order", list(range(n))),
              ("Step 2: sort from largest to smallest", order)]
    bw = pw / n
    for k, (sub, days) in enumerate(panels):
        x0 = left + k * (pw + gap)
        el.append(f"<text x='{x0 + pw / 2}' y='{top - 22}' text-anchor='middle' font-size='13' font-weight='bold' {FONT}>{sub}</text>")
        el.append(f"<line x1='{x0}' y1='{top + ph}' x2='{x0 + pw}' y2='{top + ph}' stroke='black'/>")
        el.append(f"<line x1='{x0}' y1='{top}' x2='{x0}' y2='{top + ph}' stroke='black'/>")
        for v in (0, 2, 4, 6):
            y = top + (1 - v / y_max) * ph
            el.append(f"<text x='{x0 - 6}' y='{y + 4}' text-anchor='end' font-size='11' fill='#555' {FONT}>{v}</text>")
        tops = []
        for i, d in enumerate(days):
            v = QMAX_10[d]
            bx, by = x0 + i * bw, top + (1 - v / y_max) * ph
            tops.append((bx + bw / 2, by))
            el.append(f"<rect x='{bx + 3}' y='{by}' width='{bw - 6}' height='{top + ph - by}' fill='{rank_color(rank[d], n)}'/>")
            # ô phải ghi số trong cột (cột đậm chữ trắng, cột nhạt chữ đen) để đường đứt nét đi qua đỉnh cột không đè lên số
            ty, fill = (by + 13, "white" if rank[d] < 5 else "#111") if k == 1 else (by - 4, "black")
            el.append(f"<text x='{bx + bw / 2}' y='{ty}' text-anchor='middle' font-size='10' fill='{fill}' {FONT}>{num(v, '.1f')}</text>")
            el.append(f"<text x='{bx + bw / 2}' y='{top + ph + 15}' text-anchor='middle' font-size='10' {FONT}>d{d + 1}</text>")
            if k == 1:
                el.append(f"<text x='{bx + bw / 2}' y='{top + ph + 31}' text-anchor='middle' font-size='10' fill='#c00000' {FONT}>{num((i + 1) / n, '.1f')}</text>")
        if k == 1:
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in tops)
            el.append(f"<polyline points='{pts}' fill='none' stroke='#c00000' stroke-width='2' stroke-dasharray='5,3'/>")
            el.append(f"<text x='{x0 - 6}' y='{top + ph + 15}' text-anchor='end' font-size='10' fill='#555' {FONT}>day</text>")
            el.append(f"<text x='{x0 - 6}' y='{top + ph + 31}' text-anchor='end' font-size='10' fill='#c00000' {FONT}>p = i/n</text>")
            hx, hy = tops[2]
            el.append(f"<line x1='{hx + 60}' y1='{hy - 40}' x2='{hx + 4}' y2='{hy - 6}' stroke='#444' marker-end='url(#arr)'/>")
            el.append(f"<text x='{hx + 62}' y='{hy - 44}' font-size='11' {FONT}>rank 3, p = 0.3:</text>")
            el.append(f"<text x='{hx + 62}' y='{hy - 30}' font-size='11' {FONT}>qmax ≥ 2.1 on 30% of days</text>")
        else:
            el.append(f"<text x='{x0 - 6}' y='{top + ph + 15}' text-anchor='end' font-size='10' fill='#555' {FONT}>day</text>")
    ax0, ay = left + pw + 20, top + ph / 2
    el.append(f"<line x1='{ax0}' y1='{ay}' x2='{ax0 + gap - 40}' y2='{ay}' stroke='#444' stroke-width='2' marker-end='url(#arr)'/>")
    el.append(f"<text x='{ax0 + (gap - 40) / 2}' y='{ay - 8}' text-anchor='middle' font-size='12' {FONT}>sort</text>")
    for j, line in enumerate(["y-axis: qmax (mm/day). Same color = same day in both panels.",
                              "Red dashed line through the bar tops = the flow duration curve; the order of days is lost."]):
        el.append(f"<text x='{width / 2}' y='{height - 34 + 16 * j}' text-anchor='middle' font-size='12' fill='#333' {FONT}>{line}</text>")
    save("BiasCast3_FDC_Build.svg", el, height=height, width=width)


def fig_fdc_levels():
    # Đường duy trì tự đặt của một lưu vực (thang log), tô bốn mức theo xác suất vượt và kẻ ba ngưỡng
    width, height, top = 680, 460, 95
    ps, qs = lognormal_fdc(0.0, 0.8)
    el, to_px = make_fdc_axes(width, height, top, 0.05, 20, [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
                              "Four flow levels on one basin's duration curve (illustration)", "qmax (mm/day, log scale)")
    for name, p0, p1, color, days in LEVELS:
        x0, y0 = to_px(p0, 20)
        x1, y1 = to_px(p1, 0.05)
        el.insert(3, f"<rect x='{x0}' y='{y0}' width='{x1 - x0}' height='{y1 - y0}' fill='{color}'/>")
        if name == "Very high":
            el.append(f"<line x1='{x0 + 6}' y1='{top - 4}' x2='{x0 + 6}' y2='{top - 26}' stroke='#990000'/>")
            el.append(f"<text x='{x0 + 2}' y='{top - 30}' font-size='12' fill='#990000' {FONT}>Very high: p 0-0.02, ≈ 7 days/year</text>")
        else:
            cx = (x0 + x1) / 2
            el.append(f"<text x='{cx}' y='{y1 - 44}' text-anchor='middle' font-size='13' font-weight='bold' {FONT}>{name}</text>")
            el.append(f"<text x='{cx}' y='{y1 - 28}' text-anchor='middle' font-size='11' {FONT}>p {num(p0, 'g')}-{num(p1, 'g')}</text>")
            el.append(f"<text x='{cx}' y='{y1 - 13}' text-anchor='middle' font-size='11' {FONT}>≈ {days} days/year</text>")
    el.append(f"<path d='{curve_path(ps, qs, to_px)}' fill='none' stroke='#1f4e79' stroke-width='2.5'/>")
    for p in (0.02, 0.2, 0.7):
        q = math.exp(0.8 * NormalDist().inv_cdf(1 - p))
        px, py = to_px(p, q)
        el.append(f"<line x1='{LEFT}' y1='{py}' x2='{px}' y2='{py}' stroke='#c00000' stroke-dasharray='5,4'/>")
        el.append(f"<circle cx='{px}' cy='{py}' r='4' fill='#c00000'/>")
        el.append(f"<text x='{px + 7}' y='{py - 6}' font-size='12' fill='#c00000' {FONT}>threshold at p = {num(p, 'g')}: {num(q, '.2f')}</text>")
    el.append(f"<text x='{width / 2}' y='{top - 52}' text-anchor='middle' font-size='11' fill='#555' {FONT}>"
              f"Thresholds from observed qmax 1981-2013 of this basin only; a forecast is put into a level with the same thresholds</text>")
    save("BiasCast3_FDC_Levels.svg", el, height=height, width=width)


def fdc_metrics(obs, sim):
    # ba chỉ số Yilmaz theo đúng cách tính của NeuralHydrology metrics.py (hai chuỗi đã sắp giảm dần, đủ dài)
    n = len(obs)
    h = round(0.02 * n)
    fhv = sum(s - o for s, o in zip(sim[:h], obs[:h])) / sum(obs[:h]) * 100
    lo, up = round(0.2 * n), round(0.7 * n)
    fms = ((math.log(sim[lo]) - math.log(sim[up])) - (math.log(obs[lo]) - math.log(obs[up]))) \
        / (math.log(obs[lo]) - math.log(obs[up])) * 100
    lw = round(0.3 * n)
    lo_obs, lo_sim = [math.log(v) for v in obs[-lw:]], [math.log(v) for v in sim[-lw:]]
    qol, qsl = sum(v - min(lo_obs) for v in lo_obs), sum(v - min(lo_sim) for v in lo_sim)
    return fhv, fms, -1 * (qsl - qol) / qol * 100


def fig_fdc_metrics():
    # Đường duy trì quan trắc và mô hình (tự đặt): mô hình hụt đỉnh, đoạn giữa thoải, đoạn thấp không xuống đủ thấp
    width, height, top = 680, 470, 60
    ps, obs = lognormal_fdc(0.0, 0.8)
    _, sim = lognormal_fdc(0.05, 0.55)
    fhv, fms, flv = fdc_metrics(obs, sim)
    el, to_px = make_fdc_axes(width, height, top, 0.05, 20, [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
                              "Comparing two duration curves: %BiasFHV, %BiasFMS, %BiasFLV (illustration)",
                              "qmax (mm/day, log scale)")
    for p0, p1, color in ((0.0, 0.02, "#f4cccc"), (0.2, 0.7, "#d9ead3"), (0.7, 1.0, "#cfe2f3")):
        x0, y0 = to_px(p0, 20)
        x1, y1 = to_px(p1, 0.05)
        el.insert(3, f"<rect x='{x0}' y='{y0}' width='{x1 - x0}' height='{y1 - y0}' fill='{color}'/>")
    # FMS: đoạn giữa p 0,2-0,7 tô xanh lá, nhưng chỉ số chỉ đọc hai mép => kẻ hai đường dọc đánh dấu hai điểm đó
    for p in (0.2, 0.7):
        x, y_top = to_px(p, 20)
        _, y_bot = to_px(p, 0.05)
        el.append(f"<line x1='{x}' y1='{y_top}' x2='{x}' y2='{y_bot}' stroke='#38761d' stroke-width='1.5' stroke-dasharray='6,4'/>")
        el.append(f"<text x='{x + 4}' y='{y_bot - 6}' font-size='11' fill='#38761d' {FONT}>FMS point p = {num(p, 'g')}</text>")
    for j, line in enumerate(["p 0.02-0.2:", "not measured by", "these 3 metrics"]):
        x, y = to_px(0.03, 0.42)
        el.append(f"<text x='{x}' y='{y + 14 * j}' font-size='11' fill='#777' {FONT}>{line}</text>")
    for data, color, dash in ((obs, "black", ""), (sim, "#ff7f0e", " stroke-dasharray='7,4'")):
        el.append(f"<path d='{curve_path(ps, data, to_px)}' fill='none' stroke='{color}' stroke-width='2.5'{dash}/>")
    n = len(ps)
    for data, color in ((obs, "black"), (sim, "#ff7f0e")):
        (ax, ay), (bx, by) = to_px(ps[round(0.2 * n)], data[round(0.2 * n)]), to_px(ps[round(0.7 * n)], data[round(0.7 * n)])
        el.append(f"<line x1='{ax}' y1='{ay}' x2='{bx}' y2='{by}' stroke='{color}' stroke-width='1.5' stroke-dasharray='2,3'/>")
        el.append(f"<circle cx='{ax}' cy='{ay}' r='4.5' fill='{color}'/><circle cx='{bx}' cy='{by}' r='4.5' fill='{color}'/>")
    notes = [  # (p, q, căn chữ, dòng chữ) - vị trí chữ chú thích theo tọa độ dữ liệu
        (0.035, 13, "start", [f"%BiasFHV = {num(fhv, '.0f')}%", "top 2% of days (red band):", "model peaks too low"]),
        (0.30, 4.6, "start", [f"%BiasFMS = {num(fms, '.0f')}%", "middle segment (green band),", "uses only its 2 edge dots:", "model curve is flatter"]),
        (0.715, 2.6, "start", [f"%BiasFLV = {num(flv, '.0f')}%", "lowest 30% of days", "(blue band, log scale):", "model does not go low enough"]),
    ]
    for p, q, anchor, lines in notes:
        x, y = to_px(p, q)
        for j, line in enumerate(lines):
            weight = " font-weight='bold'" if j == 0 else ""
            el.append(f"<text x='{x}' y='{y + 15 * j}' text-anchor='{anchor}' font-size='12'{weight} {FONT}>{line}</text>")
    for j, (color, name, dash) in enumerate((("black", "observed", ""), ("#ff7f0e", "model", " stroke-dasharray='7,4'"))):
        y = top + 20 + 18 * j
        el.append(f"<line x1='{width - RIGHT - 120}' y1='{y}' x2='{width - RIGHT - 90}' y2='{y}' stroke='{color}' stroke-width='2.5'{dash}/>")
        el.append(f"<text x='{width - RIGHT - 84}' y='{y + 4}' font-size='12' {FONT}>{name}</text>")
    el.append(f"<text x='{width / 2}' y='42' text-anchor='middle' font-size='11' fill='#555' {FONT}>"
              f"Each series is sorted on its own before comparing, so the days do not have to match</text>")
    save("BiasCast3_FDC_Metrics.svg", el, height=height, width=width)
    return fhv, fms, flv


ANNUAL_MAX_10 = [8, 12, 9, 15, 10, 22, 11, 9, 13, 18]   # đỉnh năm của 10 năm, ví dụ Mục 3.6 (mm/ngày)
EULER = 0.5772156649                                    # hằng số Euler-Mascheroni trong công thức Gumbel


def gumbel_lmom(xs):
    # tham số Gumbel theo L-moments: l1 = trung bình, l2 = nửa trung bình |xi - xj| mọi cặp; alpha = l2 / ln 2, xi = l1 - 0,5772 alpha
    n = len(xs)
    l1 = sum(xs) / n
    l2 = sum(abs(a - b) for i, a in enumerate(xs) for b in xs[i + 1:]) / (n * (n - 1) / 2) / 2
    alpha = l2 / math.log(2)
    return l1 - EULER * alpha, alpha


def gumbel_level(xi, alpha, t):
    # ngưỡng ứng với chu kỳ lặp lại T năm: F = 1 - 1/T, x = xi - alpha ln(-ln F)
    return xi - alpha * math.log(-math.log(1 - 1 / t))


def fig_gumbel_fit():
    # Đỉnh năm của 10 năm (chấm) theo chu kỳ lặp lại thực nghiệm, đường Gumbel khớp bằng L-moments, bốn ngưỡng 1.5/2/5/10 năm của đề tài
    width, height, top = 680, 460, 70
    xi, alpha = gumbel_lmom(ANNUAL_MAX_10)
    t_min, t_max, y_max = 1.0, 20.0, 25.0
    pw, ph = width - LEFT - RIGHT, height - top - BOTTOM

    def to_px(t, q):
        return LEFT + math.log10(t / t_min) / math.log10(t_max / t_min) * pw, top + (1 - q / y_max) * ph

    el = [f"<rect width='{width}' height='{height}' fill='white'/>",
          f"<text x='{width / 2}' y='22' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>"
          f"Return period thresholds from 10 annual maxima (illustration)</text>",
          f"<text x='{width / 2}' y='42' text-anchor='middle' font-size='11' fill='#555' {FONT}>"
          f"Gumbel fitted with L-moments: xi = {num(xi, '.2f')}, alpha = {num(alpha, '.2f')}</text>"]
    for q in range(0, 26, 5):
        _, py = to_px(1, q)
        el.append(f"<line x1='{LEFT}' y1='{py}' x2='{width - RIGHT}' y2='{py}' stroke='#e6e6e6'/>")
        el.append(f"<text x='{LEFT - 8}' y='{py + 4}' text-anchor='end' font-size='12' {FONT}>{q}</text>")
    for t in (1, 1.5, 2, 5, 10, 20):
        px, _ = to_px(t, 0)
        el.append(f"<line x1='{px}' y1='{top}' x2='{px}' y2='{top + ph}' stroke='#e6e6e6'/>")
        el.append(f"<text x='{px}' y='{top + ph + 18}' text-anchor='middle' font-size='12' {FONT}>{num(t, 'g')}</text>")
    el.append(f"<rect x='{LEFT}' y='{top}' width='{pw}' height='{ph}' fill='none' stroke='black'/>")
    el.append(f"<text x='{LEFT + pw / 2}' y='{height - 15}' text-anchor='middle' font-size='13' {FONT}>Return period T (years, log scale)</text>")
    el.append(f"<text x='18' y='{top + ph / 2}' text-anchor='middle' font-size='13' {FONT} "
              f"transform='rotate(-90 18 {top + ph / 2})'>Annual maximum qmax (mm/day)</text>")
    ts = [1.01 * (t_max / 1.01) ** (i / 200) for i in range(201)]
    pts = " ".join(f"{to_px(t, gumbel_level(xi, alpha, t))[0]:.1f},{to_px(t, gumbel_level(xi, alpha, t))[1]:.1f}" for t in ts)
    el.append(f"<polyline points='{pts}' fill='none' stroke='#1f4e79' stroke-width='2.5'/>")
    n = len(ANNUAL_MAX_10)
    for rank, v in enumerate(sorted(ANNUAL_MAX_10, reverse=True), start=1):
        px, py = to_px((n + 1) / rank, v)   # vị trí vẽ Weibull T = (n + 1) / thứ hạng, chỉ để minh họa
        el.append(f"<circle cx='{px}' cy='{py}' r='5' fill='black'/>")
    for t, color in ((1.5, "#7f6000"), (2, "#38761d"), (5, "#e69138"), (10, "#c00000")):
        q = gumbel_level(xi, alpha, t)
        px, py = to_px(t, q)
        el.append(f"<line x1='{LEFT}' y1='{py}' x2='{px}' y2='{py}' stroke='{color}' stroke-dasharray='5,4'/>")
        el.append(f"<circle cx='{px}' cy='{py}' r='5' fill='white' stroke='{color}' stroke-width='2.5'/>")
        el.append(f"<text x='{px + 8}' y='{py + 16}' font-size='12' fill='{color}' {FONT}>T = {num(t, 'g')}: {num(q, '.1f')}</text>")
    lx, ly = width - RIGHT - 210, top + ph - 52
    el.append(f"<circle cx='{lx}' cy='{ly}' r='5' fill='black'/>")
    el.append(f"<text x='{lx + 10}' y='{ly + 4}' font-size='12' {FONT}>annual maximum (10 years)</text>")
    el.append(f"<line x1='{lx - 12}' y1='{ly + 20}' x2='{lx + 4}' y2='{ly + 20}' stroke='#1f4e79' stroke-width='2.5'/>")
    el.append(f"<text x='{lx + 10}' y='{ly + 24}' font-size='12' {FONT}>fitted Gumbel curve</text>")
    save("BiasCast3_Gumbel_Fit.svg", el, height=height, width=width)
    return xi, alpha


OBS_15 = [5, 6, 8, 13, 15, 11, 8, 7, 6, 9, 14, 10, 7, 6, 5]     # 15 ngày quan trắc, ví dụ ghép sự kiện Mục 3.6
SIM_15 = [5, 6, 7, 10, 13, 12, 9, 7, 6, 9, 11, 9, 8, 13, 6]     # 15 ngày dự báo
EVENT_THRESHOLD = 12


def up_crossings(xs, thr):
    # ngày bắt đầu một sự kiện: hôm trước dưới ngưỡng, hôm nay từ ngưỡng trở lên (như mã Nearing 2024)
    return [i for i in range(1, len(xs)) if xs[i - 1] < thr <= xs[i]]


def fig_event_match():
    # Hai chuỗi 15 ngày, một ngưỡng; đánh dấu ngày bắt đầu sự kiện và kết quả ghép trong cửa sổ ±1 ngày
    width, height, top = 680, 430, 60
    pw, ph, y_max = width - LEFT - RIGHT, 260, 18.0
    n = len(OBS_15)

    def to_px(i, v):
        return LEFT + i / (n - 1) * pw, top + (1 - v / y_max) * ph

    el = [f"<rect width='{width}' height='{height}' fill='white'/>",
          f"<text x='{width / 2}' y='22' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>"
          f"Matching flood events within ±1 day (illustration)</text>",
          f"<text x='{width / 2}' y='42' text-anchor='middle' font-size='11' fill='#555' {FONT}>"
          f"An event starts on the day the series crosses the threshold from below; a 2-day high counts once</text>"]
    for v in (0, 5, 10, 15):
        _, py = to_px(0, v)
        el.append(f"<line x1='{LEFT}' y1='{py}' x2='{width - RIGHT}' y2='{py}' stroke='#e6e6e6'/>")
        el.append(f"<text x='{LEFT - 8}' y='{py + 4}' text-anchor='end' font-size='12' {FONT}>{v}</text>")
    for i in range(n):
        px, _ = to_px(i, 0)
        el.append(f"<text x='{px}' y='{top + ph + 18}' text-anchor='middle' font-size='11' {FONT}>{i + 1}</text>")
    el.append(f"<rect x='{LEFT}' y='{top}' width='{pw}' height='{ph}' fill='none' stroke='black'/>")
    el.append(f"<text x='{LEFT + pw / 2}' y='{top + ph + 38}' text-anchor='middle' font-size='13' {FONT}>Day</text>")
    el.append(f"<text x='18' y='{top + ph / 2}' text-anchor='middle' font-size='13' {FONT} "
              f"transform='rotate(-90 18 {top + ph / 2})'>qmax (mm/day)</text>")
    _, ty = to_px(0, EVENT_THRESHOLD)
    el.append(f"<line x1='{LEFT}' y1='{ty}' x2='{width - RIGHT}' y2='{ty}' stroke='#c00000' stroke-width='1.5' stroke-dasharray='6,4'/>")
    el.append(f"<text x='{LEFT + 6}' y='{ty - 6}' font-size='12' fill='#c00000' {FONT}>threshold {EVENT_THRESHOLD}</text>")
    for data, color, dash in ((OBS_15, "black", ""), (SIM_15, "#ff7f0e", " stroke-dasharray='7,4'")):
        pts = " ".join(f"{to_px(i, v)[0]:.1f},{to_px(i, v)[1]:.1f}" for i, v in enumerate(data))
        el.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2.5'{dash}/>")
        for i in up_crossings(data, EVENT_THRESHOLD):
            px, py = to_px(i, data[i])
            el.append(f"<circle cx='{px}' cy='{py}' r='6' fill='{color}'/>")
    notes = [  # (ngày, mức chữ, căn chữ, màu, hai dòng chữ)
        (3.4, 17.2, "middle", "#38761d", ["hit", "observed day 4, model day 5"]),
        (10.0, 17.2, "middle", "#c00000", ["miss", "observed day 11, no model event"]),
        (13.9, 17.2, "middle", "#b45f06", ["false alarm", "model day 14"]),
    ]
    for i, v, anchor, color, lines in notes:
        px, py = to_px(i, v)
        for j, text in enumerate(lines):
            weight = " font-weight='bold'" if j == 0 else ""
            el.append(f"<text x='{px}' y='{py + 14 * j}' text-anchor='{anchor}' font-size='12'{weight} fill='{color}' {FONT}>{text}</text>")
    ly = top + ph + 62
    for j, (color, name, dash) in enumerate((("black", "observed", ""), ("#ff7f0e", "model", " stroke-dasharray='7,4'"))):
        x = LEFT + 120 + 200 * j
        el.append(f"<line x1='{x}' y1='{ly}' x2='{x + 30}' y2='{ly}' stroke='{color}' stroke-width='2.5'{dash}/>")
        el.append(f"<text x='{x + 36}' y='{ly + 4}' font-size='12' {FONT}>{name}</text>")
    el.append(f"<text x='{width / 2}' y='{ly + 28}' text-anchor='middle' font-size='12' {FONT}>"
              f"recall = 1 / 2 observed events caught; precision = 1 / 2 model events correct</text>")
    save("BiasCast3_Event_Match.svg", el, height=height, width=width)


OBS_DAYS, SIM_DAYS = (3, 9), (5, 9)   # ngày bắt đầu trận ở câu hỏi 5 Mục 3.6


def match_events(obs, sim, window):
    # Ghép trận theo cửa sổ ±window ngày => (B bắt được, S bỏ sót, N báo nhầm)
    hits = sum(any(abs(o - m) <= window for m in sim) for o in obs)
    false = sum(not any(abs(o - m) <= window for o in obs) for m in sim)
    return hits, len(obs) - hits, false


def fig_event_table():
    # Bảng 2x2 của chấm theo trận và hai trục ngày cho câu hỏi 5 (cửa sổ ±1 và ±2 ngày)
    width, height = 680, 560
    el = [f"<rect width='{width}' height='{height}' fill='white'/>",
          f"<text x='{width / 2}' y='24' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>"
          f"Event scoring is a 2x2 table (illustration)</text>"]
    x0, y0, cw, ch, hh = 210, 50, 210, 62, 34
    el.append(f"<text x='{x0 + cw}' y='{y0 + 14}' text-anchor='middle' font-size='12' font-weight='bold' {FONT}>Model</text>")
    for j, head in enumerate(("warns (event)", "silent (no event)")):
        el.append(f"<text x='{x0 + cw * j + cw / 2}' y='{y0 + hh - 4}' text-anchor='middle' font-size='12' {FONT}>{head}</text>")
    cells = [[("#d9ead3", "B = hit", "caught"), ("#f4cccc", "S = miss", "flood not warned")],
             [("#fce5cd", "N = false alarm", "warned, no flood"), ("#eeeeee", "quiet days", "not counted")]]
    for i, head in enumerate(("Observed: flood", "Observed: no flood")):
        cy = y0 + hh + ch * i
        el.append(f"<text x='{x0 - 10}' y='{cy + ch / 2 + 4}' text-anchor='end' font-size='12' {FONT}>{head}</text>")
        for j, (fill, big, small) in enumerate(cells[i]):
            cx = x0 + cw * j
            el.append(f"<rect x='{cx}' y='{cy}' width='{cw}' height='{ch}' fill='{fill}' stroke='black'/>")
            el.append(f"<text x='{cx + cw / 2}' y='{cy + 27}' text-anchor='middle' font-size='14' font-weight='bold' {FONT}>{big}</text>")
            el.append(f"<text x='{cx + cw / 2}' y='{cy + 46}' text-anchor='middle' font-size='11' fill='#444' {FONT}>{small}</text>")
    fy = y0 + hh + 2 * ch + 26
    for k, line in enumerate(("POD (recall) = B / (B + S)  |  precision = B / (B + N)",
                              "FAR = N / (B + N)  |  CSI = B / (B + S + N)")):
        el.append(f"<text x='{width / 2}' y='{fy + 18 * k}' text-anchor='middle' font-size='13' {FONT}>{line}</text>")
    d0, d1, days = 150, width - 40, 11

    def dx(d):
        return d0 + (d - 1) / (days - 1) * (d1 - d0)

    for k, w in enumerate((1, 2)):
        by = fy + 80 + 140 * k
        el.append(f"<text x='20' y='{by - 22}' font-size='13' font-weight='bold' {FONT}>Window ±{w} day{'s' if w > 1 else ''}</text>")
        for o in OBS_DAYS:
            el.append(f"<rect x='{dx(o - w)}' y='{by - 12}' width='{dx(o + w) - dx(o - w)}' height='54' fill='#d9ead3' opacity='0.7'/>")
        for row, (name, evs, color) in enumerate((("observed", OBS_DAYS, "black"), ("model", SIM_DAYS, "#ff7f0e"))):
            y = by + 30 * row
            el.append(f"<text x='{d0 - 14}' y='{y + 4}' text-anchor='end' font-size='12' {FONT}>{name}</text>")
            el.append(f"<line x1='{d0}' y1='{y}' x2='{d1}' y2='{y}' stroke='#999'/>")
            for e in evs:
                el.append(f"<circle cx='{dx(e)}' cy='{y}' r='7' fill='{color}'/>")
        for d in range(1, days + 1):
            el.append(f"<text x='{dx(d)}' y='{by + 58}' text-anchor='middle' font-size='11' {FONT}>{d}</text>")
        b, sm, n = match_events(OBS_DAYS, SIM_DAYS, w)
        el.append(f"<text x='{d0}' y='{by + 80}' font-size='12' {FONT}>B = {b}, S = {sm}, N = {n} =&gt; "
                  f"POD = {num(b / (b + sm), '.2f')}, precision = {num(b / (b + n), '.2f')}, CSI = {num(b / (b + sm + n), '.2f')}</text>")
    el.append(f"<text x='{width / 2}' y='{height - 10}' text-anchor='middle' font-size='11' fill='#555' {FONT}>"
              f"Green band = days around an observed event start where a model start counts as a hit</text>")
    save("BiasCast3_Event_Table.svg", el, height=height, width=width)


def fig_period_timeline():
    # Trục năm 1980-2018 có lưới từng năm; mỗi hàng một thanh kèm một dòng giải thích:
    # số đo qmax, phần dùng tính ngưỡng, ba kỳ train/validation/test, lưới ECMWF HRES; hai trận lũ lớn
    width, height = 900, 525
    x0, x1, y_first, y_last = 150, width - 20, 1980, 2018
    top, row_h = 50, 92

    def yx(year):
        return x0 + (year - y_first) / (y_last - y_first) * (x1 - x0)

    rows = [  # (nhãn hàng 2 dòng, [(năm bắt đầu, năm kết thúc, màu, chữ trong thanh)], dòng giải thích dưới thanh)
        (["observed qmax", "(gauge data)"], [(1981, 2018, "#3d85c6", "measured every day 1981-2017 (37 years)")],
         ["Daily maximum discharge measured at the gauge. The test years 2014-2017 are measured too (0.23% missing):",
          "this is the 'answer key' the forecasts are scored against."]),
        (["used for", "thresholds"], [(1981, 2014, "#cfe2f3", "only 1981-2013 (33 years)")],
         ["Flood thresholds (1.5/2/5/10-year) and the four flow levels are computed only from these years,",
          "deliberately stopping before the test years so the thresholds never see the data being scored."]),
        (["model", "periods"], [(2003, 2010, "#38761d", "train|2003-2009"), (2010, 2014, "#e69138", "val|2010-2013"),
                               (2014, 2018, "#c00000", "test|2014-2017")],
         ["train: the model learns its weights.  val: used to pick the best epoch.  test: final score, model vs observed qmax.",
          "Years before 2003 are never targets; they only serve as the 364-day input history."]),
        (["ECMWF HRES", "model grid"], [(2002, 2006.08, "#eeeeee", "~40 km|2002-2006"), (2006.08, 2010.07, "#d9d9d9", "25 km|2006-2010"),
                                        (2010.07, 2016.19, "#b7b7b7", "16 km|Jan 2010 - Mar 2016"),
                                        (2016.19, 2018, "#7f7f7f", "9 km")],
         ["Archived operational forecasts: each year was made by the model version running that year (grid refined 1 Feb 2006, Jan 2010, 8 Mar 2016).",
          "Train sees only the coarse grids (~40 then 25 km); about half of test uses 9 km. The data is not adjusted for these jumps."]),
    ]
    bottom = top + row_h * len(rows)
    el = [f"<rect width='{width}' height='{height}' fill='white'/>",
          f"<text x='{width / 2}' y='24' text-anchor='middle' font-size='16' font-weight='bold' {FONT}>"
          f"Which years are used for what (BiasCast data, this project's thresholds)</text>"]
    for year in range(y_first, y_last + 1):   # lưới dọc mỗi năm, đậm hơn mỗi 5 năm
        el.append(f"<line x1='{yx(year):.1f}' y1='{top}' x2='{yx(year):.1f}' y2='{bottom}' "
                  f"stroke='{'#bdbdbd' if year % 5 == 0 else '#ececec'}'/>")
    under_rows = len(el)   # đường lũ chèn vào đây => nằm dưới thanh và nền chữ
    for k, (name, bars, notes) in enumerate(rows):
        y = top + 6 + row_h * k
        for m, line in enumerate(name):
            el.append(f"<text x='{x0 - 10}' y='{y + 14 + 14 * m}' text-anchor='end' font-size='12' font-weight='bold' {FONT}>{line}</text>")
        for a, b, color, text in bars:
            el.append(f"<rect x='{yx(a):.1f}' y='{y}' width='{yx(b) - yx(a):.1f}' height='36' fill='{color}' stroke='white'/>")
            fill = "white" if color in ("#3d85c6", "#38761d", "#e69138", "#c00000", "#7f7f7f") else "black"
            lines = text.split("|")
            for m, line in enumerate(lines):   # chữ trong thanh, 1-3 dòng, căn giữa theo chiều dọc
                ty = y + 18 + 11 * (m - (len(lines) - 1) / 2) + 4
                el.append(f"<text x='{(yx(a) + yx(b)) / 2:.1f}' y='{ty:.1f}' text-anchor='middle' font-size='10' "
                          f"font-weight='{'bold' if m == 0 else 'normal'}' fill='{fill}' {FONT}>{line}</text>")
        el.append(f"<rect x='{x0 - 4}' y='{y + 41}' width='{x1 - x0 + 4}' height='{13 * len(notes) + 6}' fill='white'/>")  # nền trắng che lưới sau chữ
        for m, line in enumerate(notes):
            el.append(f"<text x='{x0}' y='{y + 52 + 13 * m}' font-size='11' fill='#444' {FONT}>{line}</text>")
    for year, label, note in ((2002.6, "Aug 2002 flood", "before train: not a target"),
                              (2013.45, "Jun 2013 flood", "in val and in thresholds")):
        x = yx(year)
        el.insert(under_rows, f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{bottom + 4}' stroke='#1f4e79' stroke-width='1.5' stroke-dasharray='4,3'/>")
        el.append(f"<text x='{x:.1f}' y='{bottom + 16}' text-anchor='middle' font-size='11' font-weight='bold' fill='#1f4e79' {FONT}>{label}</text>")
        el.append(f"<text x='{x:.1f}' y='{bottom + 29}' text-anchor='middle' font-size='10' fill='#1f4e79' {FONT}>{note}</text>")
    ay = bottom + 40
    el.append(f"<line x1='{x0}' y1='{ay}' x2='{x1}' y2='{ay}' stroke='black'/>")
    for year in range(y_first, y_last + 1):   # nhãn mỗi năm, xoay dọc, đặt giữa cột của năm
        el.append(f"<line x1='{yx(year):.1f}' y1='{ay}' x2='{yx(year):.1f}' y2='{ay + 4}' stroke='black'/>")
        if year < y_last:
            x = yx(year + 0.5)
            el.append(f"<text x='{x:.1f}' y='{ay + 8}' font-size='10' text-anchor='end' dy='3' {FONT} "
                      f"transform='rotate(-90 {x:.1f} {ay + 8})'>{year}</text>")
    el.append(f"<text x='{width / 2}' y='{height - 10}' text-anchor='middle' font-size='11' fill='#555' {FONT}>"
              f"One grid column = one year (darker line every 5 years); each bar starts on 1 January of its first year</text>")
    save("BiasCast3_Periods.svg", el, height=height, width=width)

# %% Phần 4 - Hình của Phần 4 BiasCast
RAIN = {"A": [5, 0, 12], "B": [2, 8, 1]}     # lô thu nhỏ: 2 mẫu x 3 ngày, biến mưa (số tự đặt)
TEMP = {"A": [10, 11, 9], "B": [3, 4, 6]}    # biến nhiệt độ của cùng lô
VAR_FILL = {"rain": "#dbe9f6", "temp": "#fde4cf", "e1": "#e2efda", "e2": "#e2efda", "s1": "#ead1dc"}


def grid(el, x, y, rows, cols, vals, title=None, cw=52, ch=26, lw=62):
    # Vẽ một bảng nhỏ: nhãn hàng bên trái, nhãn cột ở trên, ô tô màu theo tên cột; trả về đáy bảng.
    if title:
        el.append(f"<text x='{x + lw + cw * len(cols) / 2}' y='{y - 6}' text-anchor='middle' font-size='12' font-weight='bold' {FONT}>{title}</text>")
    for j, c in enumerate(cols):
        el.append(f"<text x='{x + lw + cw * j + cw / 2}' y='{y + 16}' text-anchor='middle' font-size='11' fill='#444' {FONT}>{c}</text>")
    for i, r in enumerate(rows):
        ry = y + 22 + ch * i
        el.append(f"<text x='{x + lw - 8}' y='{ry + 17}' text-anchor='end' font-size='11' fill='#444' {FONT}>{r}</text>")
        for j, c in enumerate(cols):
            el.append(f"<rect x='{x + lw + cw * j}' y='{ry}' width='{cw}' height='{ch}' fill='{VAR_FILL.get(c, '#f2f2f2')}' stroke='#666'/>")
            el.append(f"<text x='{x + lw + cw * j + cw / 2}' y='{ry + 17}' text-anchor='middle' font-size='12' {FONT}>{vals[i][j]}</text>")
    return y + 22 + ch * len(rows)


def section(el, y, text, sub):
    # Tiêu đề một bước (đậm) kèm một dòng giải thích.
    el.append(f"<text x='20' y='{y}' font-size='14' font-weight='bold' {FONT}>{text}</text>")
    el.append(f"<text x='20' y='{y + 17}' font-size='11' fill='#555' {FONT}>{sub}</text>")


def arrow(el, x1, y1, x2, y2, label=None):
    # Mũi tên thẳng, nhãn (nếu có) đặt phía trên giữa mũi tên.
    el.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#444' stroke-width='1.5' marker-end='url(#arr)'/>")
    if label:
        el.append(f"<text x='{(x1 + x2) / 2}' y='{min(y1, y2) - 9}' text-anchor='middle' font-size='11' fill='#444' {FONT}>{label}</text>")


def fig_tensor_concat():
    # Ba bước của lớp đầu vào trên một lô thu nhỏ: mỗi biến một tensor, ghép theo trục cuối, đổi trục lô/ngày.
    width, height = 860, 640
    days = ["day 1", "day 2", "day 3"]
    el = [f"<rect width='{width}' height='{height}' fill='white'/>", ARROW,
          f"<text x='{width / 2}' y='24' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>"
          f"From one tensor per variable to LSTM input (toy batch: 2 samples, 3 days, 2 variables)</text>"]
    section(el, 56, "Step 1. Loader output: one tensor per variable, shape (samples, days, 1) = (2, 3, 1)",
            "Each variable is a single column of numbers, so its last axis has size 1")
    for k, (name, data) in enumerate((("rain", RAIN), ("temp", TEMP))):
        grid(el, 60 + 400 * k, 100, ["sample A", "sample B"], days, [data["A"], data["B"]], title=f"{name}: (2, 3, 1)")
    section(el, 210, "Step 2. torch.cat(dim=-1): put the variables side by side, (2, 3, 1) + (2, 3, 1) = (2, 3, 2)",
            "Each day of each sample is now one row of 2 numbers [rain, temp]; in BiasCast 32 columns of size 1 give 32")
    for k, sm in enumerate(("A", "B")):
        rows = [[RAIN[sm][d], TEMP[sm][d]] for d in range(3)]
        grid(el, 60 + 400 * k, 254, days, ["rain", "temp"], rows, title=f"sample {sm}: (3, 2)")
    section(el, 390, "Step 3. transpose(0, 1): swap the sample and day axes, (2, 3, 2) -> (3, 2, 2)",
            "Same numbers, regrouped by day: the LSTM reads day 1 for all samples, then day 2, then day 3")
    for d in range(3):
        rows = [[RAIN[sm][d], TEMP[sm][d]] for sm in ("A", "B")]
        x = 40 + 280 * d
        bottom = grid(el, x, 434, ["sample A", "sample B"], ["rain", "temp"], rows, title=f"{days[d]}: (2, 2)")
        if d < 2:
            mid = (434 + 22 + bottom) / 2
            arrow(el, x + 176, mid, x + 266, mid, "LSTM next step")
    for k, line in enumerate(("Real BiasCast batch: 32 tensors (256, 364, 1) -> cat -> (256, 364, 32) -> transpose -> (364, 256, 32)",
                              "ECMWF HRES: 5 tensors (256, 1, 1) -> cat -> (256, 1, 5) -> transpose -> (1, 256, 5)")):
        el.append(f"<text x='{width / 2}' y='{565 + 22 * k}' text-anchor='middle' font-size='13' {FONT}>{line}</text>")
    el.append(f"<text x='{width / 2}' y='{height - 14}' text-anchor='middle' font-size='11' fill='#555' {FONT}>"
              f"Numbers are made up; colours mark the variable (blue = rain, orange = temperature)</text>")
    save("BiasCast4_Tensor_Concat.svg", el, height=height, width=width)


def chain(el, x, y, boxes, fill, bw=190, bh=40, gap=26):
    # Một chuỗi hộp nối bằng mũi tên theo hàng ngang; mỗi hộp hai dòng chữ; trả về mép phải hộp cuối.
    for k, (top, bottom) in enumerate(boxes):
        bx = x + (bw + gap) * k
        el.append(f"<rect x='{bx}' y='{y}' width='{bw}' height='{bh}' rx='5' fill='{fill}' stroke='#666'/>")
        el.append(f"<text x='{bx + bw / 2}' y='{y + 16}' text-anchor='middle' font-size='11' font-weight='bold' {FONT}>{top}</text>")
        el.append(f"<text x='{bx + bw / 2}' y='{y + 31}' text-anchor='middle' font-size='11' {FONT}>{bottom}</text>")
        if k < len(boxes) - 1:
            arrow(el, bx + bw, y + bh / 2, bx + bw + gap - 2, y + bh / 2)
    return x + (bw + gap) * len(boxes) - gap


def fig_tensor_static():
    # Thuộc tính nhúng một lần rồi lặp cho mọi ngày, nối với nhúng động (ví dụ số của Mục 4.2), và kích thước thật của một lô.
    width, height = 900, 700
    days = ["day 1", "day 2", "day 3"]
    dyn = [[0.9, -0.53], [0.4, 0.1], [-0.2, 0.7]]   # nhúng động 2 chiều của mẫu A (ngày 1 là ví dụ Mục 4.2)
    stat = 0.15                                     # nhúng thuộc tính 1 chiều của mẫu A (ví dụ Mục 4.2)

    def fmt(v):
        return num(v, "g")

    el = [f"<rect width='{width}' height='{height}' fill='white'/>", ARROW,
          f"<text x='{width / 2}' y='24' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>"
          f"Catchment attributes: embedded once, repeated for every day, then concatenated (sample A)</text>"]
    y0 = 80
    b = grid(el, 20, y0, days, ["e1", "e2"], [[fmt(v) for v in r] for r in dyn], title="dynamic embedding (3, 2)")
    el.append(f"<text x='{20 + 62 + 52}' y='{b + 18}' text-anchor='middle' font-size='11' fill='#555' {FONT}>changes every day</text>")
    grid(el, 200, y0 + 26, ["once"], ["s1"], [[fmt(stat)]], title="attribute embedding (1)")
    el.append(f"<text x='{200 + 62 + 26}' y='{y0 + 92}' text-anchor='middle' font-size='11' fill='#555' {FONT}>no day axis</text>")
    arrow(el, 322, y0 + 61, 414, y0 + 61, "repeat")
    grid(el, 400, y0, days, ["s1"], [[fmt(stat)]] * 3, title="repeated (3, 1)")
    arrow(el, 522, y0 + 61, 614, y0 + 61, "cat")
    grid(el, 600, y0, days, ["e1", "e2", "s1"], [[fmt(r[0]), fmt(r[1]), fmt(stat)] for r in dyn], title="LSTM input (3, 3)")
    section(el, 250, "Real model, one batch of 256 samples: hindcast branch (364 days) and forecast branch (day t)",
            "Each branch has its own dynamic and attribute embedding; both end with 16 + 16 = 32 numbers per day")
    branches = (("hindcast", 290, "32 inputs", 32, 364, "#dbe9f6"), ("forecast", 400, "5 ECMWF HRES", 5, 1, "#e2efda"))
    for name, y, label, nv, nd, fill in branches:
        # Hàng trên: biến động của nhánh; hàng dưới: thuộc tính tĩnh nhúng riêng cho nhánh đó; hộp phải: ghép.
        right = chain(el, 20, y, [(label, f"{nv} x (256, {nd}, 1)"), ("cat + transpose", f"({nd}, 256, {nv})"),
                                  (f"embed {nv} -> 16", f"({nd}, 256, 16)")], fill, bw=180)
        chain(el, 20, y + 50, [("33 attributes", "(256, 33)"), ("embed 33 -> 16", "(256, 16)"),
                               ("repeat each day", f"({nd}, 256, 16)")], "#ead1dc", bw=180)
        arrow(el, right, y + 20, right + 40, y + 40)
        arrow(el, right, y + 70, right + 40, y + 50)
        el.append(f"<rect x='{right + 42}' y='{y + 25}' width='190' height='40' rx='5' fill='#f2f2f2' stroke='#666'/>")
        el.append(f"<text x='{right + 137}' y='{y + 41}' text-anchor='middle' font-size='11' font-weight='bold' {FONT}>{name}: 16 + 16 = 32</text>")
        el.append(f"<text x='{right + 137}' y='{y + 56}' text-anchor='middle' font-size='11' {FONT}>({nd}, 256, 32)</text>")
    el.append(f"<text x='{width / 2}' y='522' text-anchor='middle' font-size='12' font-weight='bold' {FONT}>"
              f"cat along the day axis: 364 + 1 = 365 steps -> one shared LSTM (hidden size 128)</text>")
    chain(el, 20, 545, [("LSTM output", "(364, 256, 128) + (1, 256, 128)"), ("transpose back + cat", "(256, 365, 128)"),
                        ("dropout, linear 128 -> 1", "prediction (256, 365, 1)")], "#e2efda", bw=240)
    chain(el, 20, 615, [("qmax label", "(256, 365, 1), not transposed"), ("loss", "last day only (predict_last_n = 1)")],
          "#fde4cf", bw=240)
    el.append(f"<text x='{width / 2}' y='{height - 14}' text-anchor='middle' font-size='11' fill='#555' {FONT}>"
              f"Two different 32s: 32 input variables before the embedding; 16 + 16 embedding dimensions entering the LSTM</text>")
    save("BiasCast4_Tensor_Static.svg", el, height=height, width=width)


# %% Phần 5 - Hình của Phần 5 BiasCast
SRC_FILL = {"FC": "#a9d18e", "RA": "#9dc3e6"}   # xanh lá: ECMWF HRES; xanh dương: tái phân tích, quan trắc


def source_box(el, x, y, w, h, srcs, text):
    # Một ô dữ liệu: tô một màu, hoặc chia đôi trên/dưới khi có hai nguồn; chữ ở giữa ("|" tách dòng cho từng nửa ô).
    for k, src in enumerate(srcs):
        hh = h / len(srcs)
        el.append(f"<rect x='{x}' y='{y + hh * k}' width='{w}' height='{hh}' fill='{SRC_FILL[src]}'/>")
    el.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='none' stroke='#555'/>")
    lines = text.split("|")   # ô hai nguồn: mỗi dòng chữ nằm gọn trong nửa ô của nguồn đó
    for k, line in enumerate(lines):
        cy = y + h * (2 * k + 1) / (2 * len(lines))
        el.append(f"<text x='{x + w / 2}' y='{cy + 4}' text-anchor='middle' font-size='11' {FONT}>{line}</text>")


def fig_input_layout():
    # Mỗi thí nghiệm: 364 ngày quá khứ và ngày t nhận nguồn nào, một pha hay hai pha, NSE trung vị (Bảng F1 của bài).
    width, height = 1110, 470
    rows = [  # tên, vai trò, nguồn quá khứ, nguồn ngày t, chữ quá khứ, chữ ngày t, hai pha?, mô hình, NSE, vận hành
        ("Baseline D_FC", "forecast only, lower benchmark", ["FC"], ["FC"], "ECMWF HRES forecast (5 vars)",
         "ECMWF HRES forecast", False, "1 LSTM, 1 embedding", "0.39", "yes"),
        ("Baseline D_RA", "reanalysis only, upper benchmark", ["RA"], ["RA"], "reanalysis (31 vars)",
         "reanalysis of day t", False, "1 LSTM, 1 embedding", "0.69", "no"),
        ("Baseline D_FCRA", "reanalysis + forecast", ["RA", "FC"], ["RA", "FC"], "reanalysis (31 vars)|+ ECMWF HRES forecast (5 vars)",
         "reanalysis of day t|+ ECMWF HRES forecast", False, "1 LSTM, 1 embedding", "0.68", "no"),
        ("Encoder-Decoder LSTM", "two LSTMs, two phases", ["RA"], ["FC"], "reanalysis (31 vars)",
         "ECMWF HRES forecast", True, "2 LSTMs + handoff, 2 embeddings", "0.57", "yes"),
        ("Sequential Forecast LSTM", "one LSTM, two phases", ["RA"], ["FC"], "reanalysis (31 vars), + qmean in the 0.71 run",
         "ECMWF HRES forecast", True, "1 LSTM, 2 embeddings", "0.63 / 0.71*", "yes"),
    ]
    x_past, w_past, w_t, gap = 220, 290, 160, 14
    el = [f"<rect width='{width}' height='{height}' fill='white'/>",
          f"<text x='{width / 2}' y='26' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>"
          f"What each experiment reads in its 365-day window</text>"]
    head_y = 62
    for x, text in ((x_past + w_past / 2, "past 364 days (t-364 ... t-1)"), (x_past + w_past + gap + w_t / 2, "day t (day being forecast)"),
                    (820, "architecture"), (960, "median|NSE"), (1045, "usable in|operation")):
        parts = text.split("|")   # tiêu đề cột hẹp viết hai dòng để không chạm cột bên cạnh
        for k, part in enumerate(parts):
            el.append(f"<text x='{x}' y='{head_y - 14 * (len(parts) - 1 - k)}' text-anchor='middle' font-size='12' "
                      f"font-weight='bold' {FONT}>{part}</text>")
    for i, (name, role, past, day_t, past_txt, t_txt, two, model, nse, op) in enumerate(rows):
        y = 80 + i * 66
        if i == 3:
            el.append(f"<line x1='20' y1='{y - 9}' x2='{width - 20}' y2='{y - 9}' stroke='#999' stroke-dasharray='4,3'/>")
        el.append(f"<text x='20' y='{y + 18}' font-size='13' font-weight='bold' {FONT}>{name}</text>")
        el.append(f"<text x='20' y='{y + 35}' font-size='11' fill='#555' {FONT}>{role}</text>")
        x_t = x_past + w_past + (gap if two else 0)
        source_box(el, x_past, y, w_past, 44, past, past_txt)
        source_box(el, x_t, y, w_t, 44, day_t, t_txt)
        if two:
            el.append(f"<line x1='{x_past + w_past}' y1='{y + 22}' x2='{x_t}' y2='{y + 22}' stroke='#c00' stroke-width='2'/>")
        el.append(f"<text x='820' y='{y + 26}' text-anchor='middle' font-size='11' {FONT}>{model}</text>")
        el.append(f"<text x='960' y='{y + 26}' text-anchor='middle' font-size='12' {FONT}>{nse}</text>")
        color = "#2e7d32" if op == "yes" else "#c62828"
        el.append(f"<text x='1045' y='{y + 26}' text-anchor='middle' font-size='12' font-weight='bold' fill='{color}' {FONT}>{op}</text>")
    ly = 80 + len(rows) * 66 + 6
    for k, (src, text) in enumerate((("RA", "reanalysis / observations (ERA5-Land, E-OBS, MSWEP, GLEAM)"),
                                     ("FC", "ECMWF HRES forecast, 1-day lead"))):
        lx = 20 + k * 470
        el.append(f"<rect x='{lx}' y='{ly}' width='16' height='12' fill='{SRC_FILL[src]}' stroke='#555'/>")
        el.append(f"<text x='{lx + 22}' y='{ly + 11}' font-size='11' {FONT}>{text}</text>")
    for k, line in enumerate(("Single-phase rows (above the dashed line): all 365 days get the same variables through one embedding; "
                              "red link = two phases with separate embeddings.",
                              "Usable in operation = needs no reanalysis of day t (day t has not happened yet when the forecast is made). "
                              "* 0.71 with past discharge qmean in the hindcast. NSE: Table F1 of the paper.")):
        el.append(f"<text x='20' y='{ly + 36 + 18 * k}' font-size='11' fill='#555' {FONT}>{line}</text>")
    save("BiasCast5_Input_Layout.svg", el, height=height, width=width)


def fig_seq_vs_encdec():
    # Hai cách nối pha quá khứ và ngày t: một LSTM chạy liền (Sequential) và hai LSTM nối bằng mạng chuyển trạng thái (Encoder-Decoder).
    width, height = 1000, 480
    bw, gap = 175, 22
    blue, green, grey, red = "#dbe9f6", "#dff0d8", "#eeeeee", "#f8d7da"
    el = [f"<rect width='{width}' height='{height}' fill='white'/>", ARROW,
          f"<text x='{width / 2}' y='24' text-anchor='middle' font-size='15' font-weight='bold' {FONT}>"
          f"How the past (days t-364 ... t-1) is handed to day t: one LSTM vs two LSTMs</text>"]

    def box(x, y, top, bottom, fill):
        el.append(f"<rect x='{x}' y='{y}' width='{bw}' height='40' rx='5' fill='{fill}' stroke='#666'/>")
        el.append(f"<text x='{x + bw / 2}' y='{y + 16}' text-anchor='middle' font-size='11' font-weight='bold' {FONT}>{top}</text>")
        el.append(f"<text x='{x + bw / 2}' y='{y + 31}' text-anchor='middle' font-size='11' {FONT}>{bottom}</text>")

    def row(y0, title, sub, boxes, fc_col):
        el.append(f"<text x='18' y='{y0}' font-size='14' font-weight='bold' {FONT}>{title}</text>")
        el.append(f"<text x='18' y='{y0 + 17}' font-size='11' fill='#555' {FONT}>{sub}</text>")
        y = y0 + 100
        for k, (top, bottom, fill) in enumerate(boxes):
            x = 18 + (bw + gap) * k
            box(x, y, top, bottom, fill)
            if k < len(boxes) - 1:
                arrow(el, x + bw, y + 20, x + bw + gap - 2, y + 20)
        fx = 18 + (bw + gap) * fc_col
        box(fx, y0 + 32, "Forecast embedding", "5 ECMWF HRES vars -> 16 (+16)", green)
        arrow(el, fx + bw / 2, y0 + 72, fx + bw / 2, y - 2)

    row(50, "Sequential Forecast LSTM (best model of the paper): 84,769 parameters (simple embedding, no qmean)",
        "The same LSTM keeps running: its memory (h, c) at the end of day t-1 goes straight into day t",
        [("Hindcast embedding", "31 reanalysis vars -> 16 (+16)", blue),
         ("LSTM, days t-364 ... t-1", "hidden size 128", blue),
         ("Same LSTM, day t", "starts from (h, c) of day t-1", red),
         ("Linear 128 -> 1", "qmax of day t", grey)], 2)
    row(250, "Encoder-Decoder LSTM: 233,762 parameters (own count, simple embedding, no qmean)",
        "Two separate LSTMs: the memory of the first passes through a handoff network and becomes the starting memory of the second",
        [("Hindcast embedding", "31 reanalysis vars -> 16 (+16)", blue),
         ("LSTM 1, days t-364 ... t-1", "hidden size 128", blue),
         ("Handoff network", "(h, c) 256 -> 128 -> 256, linear", red),
         ("LSTM 2, day t only", "starts from handoff output", red),
         ("Linear 128 -> 1", "qmax of day t", grey)], 3)
    for k, line in enumerate(("(+16) = 16-dim embedding of the 33 basin attributes, appended at every day. Blue = reanalysis input, green = ECMWF HRES forecast of day t, "
                              "red = where the two models differ.",
                              "Median NSE (Table F1): Sequential 0.63 (simple embedding), Encoder-Decoder 0.57 (complex) / 0.59 (simple); "
                              "with past discharge qmean: 0.71 vs 0.66.")):
        el.append(f"<text x='18' y='{445 + 18 * k}' font-size='11' fill='#555' {FONT}>{line}</text>")
    save("BiasCast5_Seq_vs_EncDec.svg", el, height=height, width=width)


if __name__ == "__main__":
    fig_cdf_steps()
    fig_cdf_compare()
    fig_benchmark_ruler()
    fig_three_errors()
    fig_persistence()
    fig_fdc_build()
    fig_fdc_levels()
    print("FHV, FMS, FLV =", fig_fdc_metrics())
    print("Gumbel xi, alpha =", fig_gumbel_fit())
    fig_event_match()
    fig_event_table()
    fig_period_timeline()
    fig_tensor_concat()
    fig_tensor_static()
    fig_input_layout()
    fig_seq_vs_encdec()
