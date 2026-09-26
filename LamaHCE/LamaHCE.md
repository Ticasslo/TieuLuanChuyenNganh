# LamaH-CE — Ghi chú thực thi

> Tài liệu ghi các quyết định và kết quả thực thi với bộ dữ liệu LamaH-CE cho bài cơ sở Kirschstein & Sun (ICML 2024), *The Merit of River Network Topology for Neural Flood Forecasting*. Mã: mỗi notebook Kaggle một tệp `.py` tự đủ — `LamaHCE_Download_Core.py` (notebook 1), `LamaHCE_Download_Extra.py` (notebook 2). Thông tin tổng quan về bộ dữ liệu: `KhaoSat/Dataset.md` Mục 2. Trích dẫn bài cơ sở: PMLR 235:24713–24725, https://proceedings.mlr.press/v235/kirschstein24a.html. Trích dẫn dữ liệu: Klingler và cs. (2021), ESSD 13:4529–4565, DOI 10.5194/essd-13-4529-2021; dữ liệu DOI 10.5281/zenodo.4525244.

## 1. Nguồn dữ liệu

| Mục | Nội dung |
|---|---|
| Bản dùng | v1.0, Zenodo `10.5281/zenodo.5153305`, tệp `1_LamaH-CE_daily_hourly.tar.gz` |
| Dung lượng | 14,8 GB nén; khoảng 70 GB giải nén (theo trang Zenodo) |
| Giấy phép | CC BY-SA 4.0 — ghi nguồn theo thư mục `Info`; dữ liệu phái sinh chia sẻ công khai phải dùng cùng giấy phép |
| Cấu trúc | 7 phần (bài báo ESSD 2021): lưu vực A (toàn thượng nguồn), B (trung gian), C (trung gian, ít tác động), trạm đo D, mạng sông, mô hình thủy văn COSERO, thư mục Info. Mỗi phần A–D gồm thuộc tính, chuỗi thời gian (ngày và giờ) và shapefile |

## 2. Dữ liệu bài cơ sở sử dụng

Kết quả đọc toàn bộ mã của Kirschstein & Sun (`dataset.py`, `functions.py`, 8 script huấn luyện/kiểm thử, 8 notebook):

| Tệp | Cột dùng |
|---|---|
| `B_basins_intermediate_all/1_attributes/Stream_dist.csv` | `ID`, `NEXTDOWNID`, `dist_hdn`, `elev_diff` (`strm_slope` được tính lại) |
| `B_basins_intermediate_all/2_timeseries/hourly/ID_*.csv` | `YYYY`, `prec`, `volsw_123`, `2m_temp`, `surf_press` |
| `D_gauges/2_timeseries/hourly/ID_*.csv` | `YYYY`, `qobs` |

- Mọi thí nghiệm (kể cả mạng con với trạm gốc 71, 211, 387, 532) đọc cùng ba nguồn trên; notebook chỉ đọc tệp kết quả và tệp `adjacency_*.csv` do mã sinh ra.
- Trạm bị loại nếu lưu lượng ≤ 0 tại bất kỳ giờ nào trong toàn chuỗi (giá trị thiếu là -999), vì vậy phải giữ đủ mọi năm của `qobs`.
- Hàm tải gốc lưu nguyên tệp 14,8 GB rồi giải nén ba thư mục (gồm cả chuỗi ngày), cần khoảng 50–55 GB ổ đĩa (ước tính) — vượt dung lượng trống 20,9 GB của phiên Kaggle đã đo.

## 3. Cách tải và lưu trữ

- **Nền tảng:** notebook Kaggle chạy CPU (không tốn quota GPU), bật Internet, chạy nền bằng *Save Version → Save & Run All* (tối đa 12 giờ).
- **Cách tải:** đọc tệp nén theo luồng, không lưu tệp gốc; chỉ giữ các tệp cần và ghi thẳng vào một tệp ZIP trong `/kaggle/working`. Lý do dùng một tệp ZIP: output notebook chỉ lưu tối đa 20 GB, và có báo cáo người dùng trên diễn đàn Kaggle rằng output chỉ giữ khoảng 500 tệp (chưa có xác nhận chính thức) trong khi dữ liệu có hơn 1.700 tệp.
- **Hai gói dữ liệu:**

| Gói | Nội dung | Mục đích |
|---|---|---|
| `lamah_ce_core.zip` (`LamaHCE_Download_Core.py`) | Toàn bộ B và D trừ chuỗi ngày; thư mục Info | Đủ dữ liệu tái lập bài cơ sở, đủ mọi năm 1981–2017 và mọi cột |
| `lamah_ce_extra.zip` (`LamaHCE_Download_Extra.py`) | A (trừ chuỗi ngày), thuộc tính và shapefile của C, mạng sông, COSERO, Info | Baseline không đồ thị dùng khí tượng toàn thượng nguồn; thuộc tính tĩnh (LOAN, EA-LSTM); bản đồ demo; baseline mô hình vật lý |

- Không tải chuỗi theo ngày (tính lại được từ chuỗi giờ; bản chỉ có chuỗi ngày nặng 1,5 GB, tải riêng khi cần) và chuỗi thời gian của C.
- **Lưu trữ lâu dài:** tạo Kaggle Dataset (Private) từ output của mỗi notebook; các notebook sau gắn bằng *Add Input*. Dataset riêng tư có thể chia sẻ cho cộng tác viên.
- Mã nguồn và 957 checkpoint (1,9 GB; 162 cho thí nghiệm chính ở Bảng 2 = 18 cấu hình đồ thị × 3 kiến trúc × 3 cách chia, 3 cho MLP, 108 cho ablation, 684 cho mạng con) của bài cơ sở lấy bằng `git clone` repo `github.com/nkirschi/neural-flood-forecasting`, không đưa vào dataset.

## 4. Kiểm thử trước khi chạy

Chạy thử trên tệp nén giả mô phỏng đủ 7 phần: mỗi tệp chạy độc lập trong môi trường mới; lọc đúng thư mục; hơn 1.200 tệp nằm trong một ZIP, kiểm tra CRC không lỗi; pandas đọc trực tiếp CSV trong ZIP; giải nén cho đúng cấu trúc đường dẫn mà mã của Kirschstein & Sun đọc; ngưỡng dung lượng bỏ qua phần vượt và in cảnh báo. Chưa kiểm được trên dữ liệu thật: dung lượng thực tế, tốc độ Zenodo → Kaggle, trường hợp rớt mạng (phải chạy lại từ đầu).

## 5. Kết quả chạy

Chưa có — cập nhật sau khi chạy trên Kaggle.
