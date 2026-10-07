# LamaH-CE — Ghi chú thực thi

> Tài liệu ghi các quyết định và kết quả thực thi với bộ dữ liệu LamaH-CE. Dữ liệu chính của đề tài là Extended LamaH-CE theo ngày của bài cơ sở BiasCast (Mục 6). Mục 1–5 ghi gói LamaH-CE gốc theo giờ (tải ban đầu theo mã của Kirschstein & Sun, ICML 2024), giữ cho các hướng khóa luận: dữ liệu giờ, mạng sông. Mã: mỗi notebook Kaggle một tệp `.py` tự đủ — `02_LamaHCE_Download_Core.py` (notebook 1), `03_LamaHCE_Download_Extra.py` (notebook 2). Thông tin tổng quan về bộ dữ liệu: `Document/05_Survey/01_Datasets.md` Mục 2. Trích dẫn mã tải gốc: Kirschstein & Sun, PMLR 235:24713–24725, https://proceedings.mlr.press/v235/kirschstein24a.html. Trích dẫn dữ liệu: Klingler và cs. (2021), ESSD 13:4529–4565, DOI 10.5194/essd-13-4529-2021; dữ liệu DOI 10.5281/zenodo.4525244.

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
| `lamah_ce_core.zip` (`02_LamaHCE_Download_Core.py`) | Toàn bộ B và D trừ chuỗi ngày; thư mục Info | Đủ dữ liệu tái lập bài cơ sở, đủ mọi năm 1981–2017 và mọi cột |
| `lamah_ce_extra.zip` (`03_LamaHCE_Download_Extra.py`) | A (trừ chuỗi ngày), thuộc tính và shapefile của C, mạng sông, COSERO, Info | Baseline không đồ thị dùng khí tượng toàn thượng nguồn; thuộc tính tĩnh (LOAN, EA-LSTM); bản đồ demo; baseline mô hình vật lý |

- Không tải chuỗi theo ngày (tính lại được từ chuỗi giờ; bản chỉ có chuỗi ngày nặng 1,5 GB, tải riêng khi cần) và chuỗi thời gian của C.
- **Lưu trữ lâu dài:** tạo Kaggle Dataset (Private) từ output của mỗi notebook; các notebook sau gắn bằng *Add Input*. Dataset riêng tư có thể chia sẻ cho cộng tác viên.
- Mã nguồn và 957 checkpoint (1,9 GB; 162 cho thí nghiệm chính ở Bảng 2 = 18 cấu hình đồ thị × 3 kiến trúc × 3 cách chia, 3 cho MLP, 108 cho ablation, 684 cho mạng con) của bài cơ sở lấy bằng `git clone` repo `github.com/nkirschi/neural-flood-forecasting`, không đưa vào dataset.

## 4. Kiểm thử trước khi chạy

Chạy thử trên tệp nén giả mô phỏng đủ 7 phần: mỗi tệp chạy độc lập trong môi trường mới; lọc đúng thư mục; hơn 1.200 tệp nằm trong một ZIP, kiểm tra CRC không lỗi; pandas đọc trực tiếp CSV trong ZIP; giải nén cho đúng cấu trúc đường dẫn mà mã của Kirschstein & Sun đọc; ngưỡng dung lượng bỏ qua phần vượt và in cảnh báo. Chưa kiểm được trên dữ liệu thật: dung lượng thực tế, tốc độ Zenodo → Kaggle, trường hợp rớt mạng (phải chạy lại từ đầu).

## 5. Kết quả chạy

| Gói | Kích thước output trên Kaggle | Dòng "XONG" (thời gian, GB nén đã đọc, số file) | Dòng "Kiểm tra ZIP" (số mục, lỗi CRC) | Cảnh báo vượt dung lượng |
|---|---|---|---|---|
| `lamah_ce_core.zip` | 6,82 GB | 183,4 phút; 14,85 GB nén; duyệt 7.366, giữ 2.000 file (31,43 GB gốc) | 2.001 mục (2.000 file + `manifest.csv`), lỗi CRC `None` | Không |
| `lamah_ce_extra.zip` | 6,32 GB | 98,8 phút; 14,85 GB nén; duyệt 7.366, giữ 1.866 file (26,50 GB gốc) | 1.867 mục (1.866 file + `manifest.csv`), lỗi CRC `None` | Không |

Tổng 13,14 GB, dưới giới hạn 20 GB của output notebook; mỗi gói dưới ngưỡng bỏ qua 18 GB của mã. Cả hai gói đầy đủ: có dòng "Kiểm tra ZIP" với lỗi CRC `None`, không có dòng "CẢNH BÁO".

Số file theo thư mục:

| Gói | Thư mục | Số file |
|---|---|---|
| Lõi | `B_basins_intermediate_all`: thuộc tính / chuỗi giờ / shapefile | 4 / 859 / 7 |
| Lõi | `D_gauges`: thuộc tính / chuỗi giờ / tệp khoảng trống (`gaps`) / shapefile | 3 / 882 / 230 / 7 |
| Lõi | `Info_deutsch`, `Info_english` | 4, 4 |
| Phụ | `A_basins_total_upstrm`: thuộc tính / chuỗi giờ / shapefile | 3 / 859 / 7 |
| Phụ | `C_basins_intermediate_lowimp`: thuộc tính / shapefile | 3 / 7 |
| Phụ | `E_stream_network` | 17 |
| Phụ | `F_hydrol_model` (COSERO): gốc / đầu vào / chuỗi / shapefile / đầu ra | 1 / 13 / 859 / 11 / 3 |
| Phụ | `G_appendix`: thuộc tính / mã Python / mã R / shapefile / hình / CORINE | 1 / 8 / 17 / 44 / 3 / 2 |
| Phụ | `Info_deutsch`, `Info_english` | 4, 4 |

## 6. Extended LamaH-CE (BiasCast, HESS 2026)

Dữ liệu chính của đề tài. Notebook `01_LamaHCEExt_Download.py` tải dữ liệu và kết quả thí nghiệm của tác giả BiasCast.

| Mục | Nội dung |
|---|---|
| Dữ liệu | Extended LamaH-CE theo ngày, Zenodo 17119635, tệp `Extended_LamaH-CE_daily.tar.gz` 0,95 GB, CC BY-NC 4.0; thư mục gốc `LamaH_extended/` gồm `A_basins_total_upstrm` và `D_gauges` (mỗi phần có thuộc tính, chuỗi thời gian, shapefile); tệp trạm ngày có cột `YYYY;MM;DD;qmin;qmean;qmax` (m³/s) |
| Kết quả tác giả | Zenodo 17292895, tệp `Experiments.tar.gz` 0,23 GB: cấu hình, trọng số, scaler, NSE/KGE theo lưu vực của 24 cấu hình |
| Cách tải | Đọc theo luồng, ghi vào `lamah_ce_ext.zip` và `biascast_experiments.zip`; lưu thành Kaggle Dataset (Private) `lamah-ce-ext` |
| Kaggle Dataset `lamah-ce-ext` (bản 1, 27/9/2026) | Kaggle tự giải nén hai tệp ZIP khi tạo dataset: thư mục `lamah_ce_ext/LamaH_extended/` (`A_basins_total_upstrm`, `D_gauges`), `biascast_experiments/Experiments/` (5 nhóm thí nghiệm và `basins_filtered.txt`), cùng `biascast_data_quality.csv`, `biascast_persistence_cdf.png`; 2.116 tệp, 3,45 GB. Notebook sau đọc thẳng các thư mục này, không đọc ZIP |
| Kiểm tra chất lượng và persistence | % thiếu `qmax` và `qmean` theo kỳ train 2003–2009 / validation 2010–2013 / test 2014–2017 của 451 lưu vực; % thiếu từng biến khí tượng 2003–2017 và ngày có ECMWF đầu tiên; persistence trên test (qmax(t−1) và qmean(t−1) làm dự báo cho qmax(t)) so với NSE theo lưu vực của 5 cấu hình tác giả; hình CDF. Mã: persistence tính trong notebook bước A (`Workspace/03_Baseline/`), % thiếu trong notebook khám phá (`Workspace/02_Exploration/`) |
| Kết quả | Đã chạy trên Kaggle (26/9/2026), số liệu ở Mục 6.1; `lamah_ce_ext.zip` 1.991 file (2,81 GB gốc → 0,95 GB, 10,0 phút), `biascast_experiments.zip` 123 file (0,64 GB gốc → 0,23 GB, 2,2 phút), lỗi CRC `None` |

### 6.1. Kết quả kiểm tra (Kaggle, 26/9/2026)

**Tệp.** `lamah_ce_ext.zip`: `A_basins_total_upstrm` (3 thuộc tính, 859 chuỗi ngày, 7 shapefile), `D_gauges` (3 thuộc tính, 1.112 tệp chuỗi, 7 shapefile). `biascast_experiments.zip`: 11 nhóm thí nghiệm (Baseline 3 nhóm, CrossDomain 2, Encoder–Decoder và Sequential mỗi loại có/không Q, học chuyển giao 2). Kết quả tác giả có 451 lưu vực.

**Tệp khí tượng lưu vực** có 44 cột: `DOY`; 21 biến ERA5-Land; 8 biến ECMWF (`t2m`, `d2m`, `msl`, `ssrd`, `tp`, `lsp`, `cp`, `e`); 8 biến E-OBS (`tg`, `tn`, `tx`, `rr`, `pp`, `fg`, `hu`, `qq`); `MSWEP_RR`; `GLEAM_ETA`, `GLEAM_ETP`. Cấu hình của tác giả không dùng `ECMWF_msl`, `ECMWF_lsp`, `ECMWF_cp`, `EOBS_hu`. Lưu lượng (`qmean`, `qmax`) nằm ở tệp trạm `D_gauges`, không nằm trong tệp khí tượng.

**Lưu lượng thiếu (451 lưu vực).** Không có ô mang mã −999. `qmax` và `qmean` thiếu cùng ngày.

| Kỳ | Ngày-lưu vực lý thuyết (451 × số ngày) | Dòng có trong tệp | Ô trống trong các dòng có | Tổng thiếu | Tỷ lệ |
|---|---|---|---|---|---|
| Train 2003–2009 | 1.153.207 | 1.115.347 | 3.040 | 40.900 | 3,55% |
| Validation 2010–2013 | 658.911 | 658.581 | 2.300 | 2.630 | 0,40% |
| Test 2014–2017 | 658.911 | 658.911 | 1.545 | 1.545 | 0,23% |

Tỉ lệ thiếu tính trên đủ mọi ngày của kỳ. Lưu vực có thiếu: train 45 (29 thiếu trên 10%), validation 10 (5), test 10 (3).

Phần thiếu tập trung ở train, chủ yếu do **27 lưu vực bắt đầu đo sau 01/01/2003** (tệp không có dòng trước ngày bắt đầu): 104 (04/06/2003), 839, 876 (01/01/2004), 110 (30/04/2004), 341 (20/08/2004), 822, 823, 824 (01/11/2004), 329 (04/08/2005), 327 (21/10/2005), 298 (28/10/2005), 92 (01/01/2006), 485, 622 (01/01/2007), 590 (10/05/2007), 577 (07/08/2007), 608 (14/09/2007), 309 (28/10/2007), 561 (01/01/2008), 391 (22/01/2008), 791 (01/03/2008), 296 (17/12/2008), 295 (01/01/2009), 800 (24/11/2009), 623 (01/01/2010), 600 (23/03/2010), 324 (07/09/2010). Ba lưu vực cuối (623, 600, 324) không có Q trong cả kỳ train nhưng vẫn được đánh giá ở test. Với mô hình có Q trong hindcast 365 ngày, bộ lọc mẫu của NeuralHydrology còn bỏ thêm năm đầu sau ngày bắt đầu đo của các lưu vực này. Ngày cuối có `qmax`: 26/11/2017 – 31/12/2017.

**Khí tượng 2003–2017** (2.471.029 ngày-lưu vực = 451 × 5.479, đủ dòng): gần như không thiếu; cột thiếu nhiều nhất `EOBS_hu` 0,36% (không dùng trong cấu hình), mọi cột khác 0,00%. Dự báo ECMWF bắt đầu 01/01/2002 (notebook giới thiệu, Phần 1). Tệp khí tượng bắt đầu 01/01/1981 ở mọi lưu vực, nên hindcast 730 ngày (cần khí tượng từ 2001) không mất mẫu train năm 2003.

**NSE tập test 2014–2017 theo lưu vực.**

| Mô hình | Số lưu vực | P10 | P25 | Trung vị | P75 | P90 |
|---|---|---|---|---|---|---|
| Sequential LSTM có Q (nhúng đơn giản) | 450 | 0,419 | 0,575 | 0,705 | 0,810 | 0,876 |
| Sequential LSTM có Q (nhúng phức tạp) | 450 | 0,416 | 0,588 | 0,705 | 0,805 | 0,864 |
| Sequential LSTM không Q | 451 | 0,346 | 0,495 | 0,628 | 0,716 | 0,788 |
| Encoder–Decoder có Q (nhúng đơn giản) | 450 | 0,397 | 0,566 | 0,672 | 0,772 | 0,848 |
| Baseline chỉ dự báo | 451 | 0,023 | 0,191 | 0,387 | 0,532 | 0,680 |
| Persistence qmax(t−1) | 451 | −0,081 | 0,129 | 0,346 | 0,655 | 0,800 |
| Persistence qmean(t−1) | 451 | 0,147 | 0,232 | 0,368 | 0,614 | 0,765 |

| So với persistence | ΔNSE trung vị so với qmax(t−1) | Tỷ lệ lưu vực mô hình hơn | ΔNSE trung vị so với qmean(t−1) | Tỷ lệ lưu vực mô hình hơn |
|---|---|---|---|---|
| Sequential LSTM có Q (đơn giản) | +0,341 | 93,3% | +0,261 | 94,0% |
| Sequential LSTM có Q (phức tạp) | +0,330 | 92,2% | +0,261 | 93,1% |
| Sequential LSTM không Q | +0,244 | 80,3% | +0,189 | 82,5% |
| Encoder–Decoder có Q (đơn giản) | +0,306 | 88,4% | +0,235 | 90,9% |
| Baseline chỉ dự báo | +0,042 | 55,9% | +0,000 | 50,1% |

**Nhận định.**
- Trung vị tự tính khớp bài (0,705; 0,628; 0,387).
- Mô hình tốt nhất hơn persistence rõ ràng (trung vị +0,26 đến +0,34, hơn ở hơn 92% lưu vực) — bài chưa báo cáo mốc này. Mô hình chỉ dùng dự báo thời tiết (0,387) ngang persistence, tức không hơn cách lấy lưu lượng hôm qua.
- Ở nhóm lưu vực dễ (P90), persistence qmax(t−1) đạt 0,800, sát mô hình (0,876): phần cải thiện của mô hình tập trung ở lưu vực khó.
- Dữ liệu thiếu ít (test 0,23%, khí tượng gần như đủ; train 3,55% chủ yếu do 27 trạm bắt đầu đo muộn). Tuy vậy bộ lọc mẫu của bài loại mẫu khi bất kỳ cột nào thiếu trong cửa sổ 365 ngày, kể cả cột ECMWF; ECMWF có từ 2002 nên năm 2003 không bị loại vì ECMWF; mẫu bị loại chủ yếu ở năm đầu của các trạm đo muộn và các ngày thiếu Q. Sửa bộ lọc (bước B1) lấy lại phần này; masked mean còn có giá trị chịu dữ liệu mất khi vận hành (`Document/04_Ideas/01_Ideas.md` Y1, Y2).

### 6.2. Khám phá dữ liệu

GVHD yêu cầu phần giới thiệu dữ liệu chỉ liệt kê các dạng cơ bản, không đi sâu vào số mẫu, số lượng chi tiết; phân tích sâu phục vụ quyết định khi dựng bộ nạp dữ liệu và chương kết quả. Mã khám phá đặt ở `Workspace/02_Exploration/` (chưa viết), chạy trên Kaggle (CPU), gắn dataset `lamah-ce-ext`.

Kết quả đọc toàn bộ tệp chuỗi ngày (27/9/2026): 859 lưu vực có chuỗi khí tượng, 882 trạm có chuỗi lưu lượng ngày, 1981–2017; 40 biến khí tượng (ERA5-Land 21, ECMWF HRES 8, E-OBS 8, MSWEP 1, GLEAM 2); 62 thuộc tính tĩnh cho 859 lưu vực; dự báo ECMWF bắt đầu 01/01/2002 với độ phủ năm 2002 là 100%, các nguồn khác bắt đầu 01/01/1981; độ phủ lưu lượng năm 1981 là 67,3%, tăng dần và gần đủ từ 2003; trạm lấy từ `Gauges.shp` (EPSG:3035), ranh giới từ `Basins_A.shp` (859 đa giác).

Điểm đã biết (Mục 6.1, trên 451 lưu vực): Q thiếu train 3,55% do 27 trạm bắt đầu đo muộn, test 0,23%; khí tượng gần như đủ, có từ 1981; ECMWF có từ 2002.
