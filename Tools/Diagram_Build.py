# Sinh sơ đồ đề tài Diagrams/ProjectDiagram.drawio từ nội dung khai báo bên dưới.
# Chạy: python Tools/Diagram_Build.py (từ gốc repo). Nội dung khớp Document/01_Plan/03_Pipeline.md.
#
# Mục lục:
#   Phần 1 — Màu theo bước và kiểu hình
#   Phần 2 — Nội dung: trình tự, kiến trúc (bài gốc / sửa đổi), thứ tự cắt và khóa luận
#   Phần 3 — Bố cục tọa độ và ghi tệp

from html import escape
from pathlib import Path

# %% Phần 1 — Màu theo bước và kiểu hình
COLORS = {
    "keep": ("#ffffff", "#666666"),
    "0": ("#eeeeee", "#666666"),
    "A": ("#fff2cc", "#d6b656"),
    "B": ("#d5e8d4", "#82b366"),
    "C": ("#e1d5e7", "#9673a6"),
    "F": ("#ffe6cc", "#d79b00"),
    "EV": ("#f8cecc", "#b85450"),
    "G": ("#dae8fc", "#6c8ebf"),
}
BOX = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};fontSize={fs};align=center;spacingLeft=6;spacingRight=6;{extra}"
BAND = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#f7f7f7;strokeColor=#999999;fontSize=14;align=left;fontStyle=1;verticalAlign=top;spacingLeft=10;spacingTop=4;"
TEXT = "text;html=1;align={a};verticalAlign=middle;fontSize={fs};fontStyle=1;"
EDGE = "endArrow=block;endFill=1;html=1;rounded=1;edgeStyle=orthogonalEdgeStyle;strokeColor=#333333;{dash}strokeWidth={w};exitX={ex};exitY={ey};exitDx=0;exitDy=0;entryX={nx};entryY={ny};entryDx=0;entryDy=0;"

# %% Phần 2 — Nội dung
SEQUENCE = [
    ("0", "0. Hạ tầng: fork riêng, mamba-ssm trên T4, kiểm thử đơn vị, đo thời gian"),
    ("A", "A. Tái lập, persistence, PNSE, DLinear, nhiễu 3 hạt giống (6 lần)"),
    ("B", "B. LSTM cải tiến: Q, masked mean theo nguồn, che dữ liệu (12 lần)"),
    ("C", "C. 5 lõi cùng tham số, tinh chỉnh 8 cấu hình, 3 hạt giống (tối đa 70 lần)"),
    ("C", "D–H. 730 ngày, dự báo 1–7 ngày, CMAL, biến thể, kiểm chứng lấy mẫu lũ (42 lần)"),
    ("EV", "Đánh giá: mức, sự kiện lũ, kịch bản vận hành, kiểm định"),
    ("G", "G. Giải thích mô hình, dò trạng thái ẩn, ngưỡng lũ, demo"),
    ("keep", "BÁO CÁO"),
]
LEGEND = [
    ("keep", "Giữ nguyên"), ("A", "A: tái lập, mốc"), ("B", "B: LSTM cải tiến"),
    ("C", "C, D, E, H: lõi, chuỗi dài, biến thể, nhiều ngày"), ("F", "F: xác suất"),
    ("EV", "Đánh giá thêm"), ("G", "G: giải thích, ngưỡng lũ, demo"),
]
# Mỗi tầng: (tên, [(khối bài gốc, [(bước, khối sửa đổi), ...]), ...])
BANDS = [
    ("1. Dữ liệu đầu vào", [
        ("Hindcast 364 ngày × 31 biến (ERA5-Land, E-OBS, MSWEP, GLEAM), một khối chung",
         [("B", "B2. Chia nhóm theo nguồn dữ liệu"), ("C", "D. Mở rộng 730 ngày (LSTM, Mamba, S4D)")]),
        ("Q quá khứ: qmean theo m³/s (nhãn mm/ngày), thiếu ngày → mất mẫu",
         [("B", "B1. Q cùng đơn vị mm/ngày với nhãn, thêm qmax quá khứ; B2: nhóm Q riêng"),
          ("B", "B3. Che dữ liệu khi huấn luyện (0,1/0,12 và 0,05/0,05)")]),
        ("Forecast ngày t: 5 biến ECMWF HRES",
         [("C", "H. Dự báo 1–7 ngày: ngày đầu ECMWF thật, các ngày sau để trống; so với chỉ quá khứ và dự báo hoàn hảo")]),
        ("33 thuộc tính tĩnh · nhãn qmax ngày t (mm/ngày theo diện tích)", []),
    ]),
    ("2. Tiền xử lý", [
        ("Chia tập: huấn luyện 2003–2009, kiểm định 2010–2013, kiểm tra 2014–2017 · z-score theo tập huấn luyện",
         [("A", "A1. Ngưỡng mức và ngưỡng lũ tính trên 1981–2013 (trước kỳ kiểm tra)")]),
        ("Lọc: thiếu 1 biến ở 1 ngày là loại cả mẫu 365 ngày",
         [("B", "B2. Sửa bộ lọc: giữ mẫu thiếu đầu vào, chỉ loại khi thiếu nhãn")]),
        ("Lô 256, trộn giữa các lưu vực", []),
    ]),
    ("3. Mô hình (~85 nghìn tham số)", [
        ("Nhúng tuyến tính, ghép 32 chiều/bước",
         [("B", "B2. Nhúng riêng từng nguồn, masked mean các nguồn có dữ liệu")]),
        ("LSTM 32→128: 364 bước hindcast + 1 bước forecast, trạng thái chạy liên tục",
         [("C", "C. Mô hình chung nhận lõi bất kỳ: LSTM, GRU, Transformer, S4D, Mamba (cùng tham số, lệch ≤1%)"),
          ("C", "E. Mamba hai chiều, Transformer patch")]),
        ("Dropout 0,3, tuyến tính 128→1 → qmax",
         [("F", "F. Đầu CMAL → xác suất vượt ngưỡng lũ"), ("C", "H. Đầu ra 7 bước: qmax ngày t … t+6")]),
    ]),
    ("4. Huấn luyện", [
        ("Hàm mất mát NSE*: (ŷ−y)² / (σ lưu vực + 0,1)²", []),
        ("Adam 10⁻³, cosine annealing, ≤30 epoch, dừng sớm theo NSE kiểm định",
         [("B", "B1. Sửa lưu mô hình tốt nhất; ghi NSE kiểm định và kiểm tra mỗi epoch")]),
        ("1 hạt giống · ẩn 128, lô 256, dropout 0,3",
         [("A", "A4. 3 hạt giống mọi cấu hình, đo nhiễu hạt giống"),
          ("C", "C. Tinh chỉnh 8 cấu hình mỗi lõi: dung lượng × độ sâu × learning rate")]),
    ]),
    ("5. Đánh giá và sản phẩm", [
        ("Kiểm tra 2014–2017: NSE, KGE, CDF (tốt nhất NSE≈0,705)",
         [("A", "A3. Persistence (NSE≈0,35–0,37), PNSE, KGE, DLinear"),
          ("EV", "Wilcoxon theo lưu vực + Cohen's d, hiệu chỉnh Holm")]),
        ("Mốc: chỉ dự báo (0,387) so tái phân tích (≈0,69)",
         [("EV", "Theo 4 mức lưu lượng; sự kiện lũ chu kỳ 1, 2, 5, 10 năm (POD, FAR, F1)"),
          ("EV", "Kịch bản vận hành: mất Q, độ trễ thực của từng nguồn")]),
        ("ΔNSE theo lưu vực, tương quan thuộc tính",
         [("EV", "ΔNSE SSM so LSTM theo lưu vực: tuyết, động lực chậm, lũ nhanh"),
          ("EV", "Chi phí: tham số, thời gian mỗi epoch, bộ nhớ")]),
        ("Chưa có giải thích mô hình, ngưỡng lũ, ứng dụng",
         [("G", "G. Integrated Gradients, dò trạng thái ẩn, ngưỡng Gumbel L-moments, demo trên trang web của nhóm (VPS): bản đồ 451 lưu vực, đường Q 7 ngày; chạy CPU: Kaggle GPU → Kaggle CPU → laptop → VPS")]),
    ]),
]
CUT_ORDER = ("Thứ tự cắt khi thiếu thời gian: (1) biến thể dự phòng bước E → (2) kiểm chứng lấy mẫu lũ"
             " → (3) CMAL bước F → (4) giải thích mô hình → (5) biến thể chính bước E → (6) 730 ngày cho S4D"
             " → (7) cách dự báo hoàn hảo ở bước H → (8) tinh chỉnh 8 → 4 cấu hình mỗi lõi")
THESIS = ("Khóa luận: đồ thị mạng sông, tiền huấn luyện trên tái phân tích 1981–2002, so với khung mã nguồn mở của Google, dữ liệu giờ,"
          " dự báo nhiều ngày với dự báo thời tiết nhiều hạn thật, hàm mất mát ưu tiên đỉnh,"
          " biến thể lai Mamba–Transformer, xLSTM, phần mềm ứng dụng")

# %% Phần 3 — Bố cục tọa độ và ghi tệp
cells = []


def vertex(cid, text, style, x, y, w, h):
    cells.append(f'<mxCell id="{cid}" parent="1" style="{style}" value="{escape(text)}" vertex="1">'
                 f'<mxGeometry height="{h}" width="{w}" x="{x}" y="{y}" as="geometry" /></mxCell>')


def box(cid, step, text, x, y, w, h, fs=12, extra=""):
    f, s = COLORS[step]
    vertex(cid, text, BOX.format(f=f, s=s, fs=fs, extra=extra), x, y, w, h)


def edge(cid, src, dst, dashed, down=False):
    ex, ey, nx, ny = (0.5, 1, 0.5, 0) if down else (1, 0.5, 0, 0.5)
    style = EDGE.format(dash="dashed=1;" if dashed else "", w=1.2 if dashed else 1.8, ex=ex, ey=ey, nx=nx, ny=ny)
    cells.append(f'<mxCell id="{cid}" edge="1" parent="1" source="{src}" style="{style}" target="{dst}">'
                 f'<mxGeometry relative="1" as="geometry" /></mxCell>')


LEFT, WIDTH, ROW_H, ROW_STEP, BAND_GAP = 20, 1460, 66, 78, 22
vertex("t1", "SƠ ĐỒ ĐỀ TÀI", TEXT.format(a="center", fs=19), LEFT, 40, WIDTH, 32)
vertex("secI", "I. TRÌNH TỰ", BAND, LEFT, 82, WIDTH, 150)
step_w = (WIDTH - 40 - 22 * (len(SEQUENCE) - 1)) / len(SEQUENCE)
for i, (step, text) in enumerate(SEQUENCE):
    box(f"s{i}", step, text, 40 + i * (step_w + 22), 124, round(step_w, 2), 88,
        extra="fontStyle=1;" if text == "BÁO CÁO" else "")
    if i:
        edge(f"es{i}", f"s{i - 1}", f"s{i}", dashed=False)

vertex("lgt", "II. KIẾN TRÚC", TEXT.format(a="left", fs=14), LEFT, 240, WIDTH, 22)
x, y = 40, 268
for i, (step, text) in enumerate(LEGEND):
    w = 14 + 8 * len(text)
    if x + w > LEFT + WIDTH - 20:
        x, y = 40, y + 32
    box(f"lg{i}", step, text, x, y, w, 26, fs=11)
    x += w + 10
y += 36
vertex("h1", "BÀI GỐC - Sequential Forecast LSTM có Q quan trắc", TEXT.format(a="center", fs=14), 40, y, 660, 26)
vertex("h2", "SỬA ĐỔI (mũi tên nét đứt)", TEXT.format(a="center", fs=14), 780, y, 680, 26)
y += 36

prev_last = None
for b, (title, rows) in enumerate(BANDS):
    band_h = 34 + ROW_STEP * len(rows) + 4
    vertex(f"band{b}", title, BAND, LEFT, y, WIDTH, band_h)
    for r, (orig, mods) in enumerate(rows):
        ry = y + 34 + r * ROW_STEP
        oid = f"a{b}_{r}"
        box(oid, "keep", orig, 40, ry, 660, ROW_H)
        if r == 0 and prev_last:
            edge(f"eb{b}", prev_last, oid, dashed=False, down=True)
        prev = oid
        for m, (step, text) in enumerate(mods):
            mid = f"o{b}_{r}_{m}"
            wide = len(mods) == 1 and step == "G"
            box(mid, step, text, 780 + m * 350, ry, 680 if wide else 330, ROW_H)
            edge(f"e{mid}", prev, mid, dashed=True)
            prev = mid
        prev_last = oid
    y += band_h + BAND_GAP

vertex("secIII", "III. THỨ TỰ CẮT VÀ KHÓA LUẬN", BAND, LEFT, y, WIDTH, 34 + 2 * ROW_STEP + 4)
box("cut", "keep", CUT_ORDER, 40, y + 34, 1420, ROW_H)
box("thesis", "G", THESIS, 40, y + 34 + ROW_STEP, 1420, ROW_H)
page_h = y + 34 + 2 * ROW_STEP + 40

xml = ('<mxfile host="Electron">\n  <diagram id="sododetai" name="Sơ đồ đề tài">\n'
       f'    <mxGraphModel dx="1030" dy="626" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
       f'fold="1" page="1" pageScale="1" pageWidth="1500" pageHeight="{page_h}" math="0" shadow="0">\n'
       '      <root>\n        <mxCell id="0" />\n        <mxCell id="1" parent="0" />\n'
       + "".join(f"        {c}\n" for c in cells)
       + "      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n")
out = Path(__file__).resolve().parent.parent / "Diagrams" / "ProjectDiagram.drawio"
out.write_text(xml, encoding="utf-8")
print(f"Đã ghi {out} ({len(cells)} phần tử, cao {page_h}px)")
