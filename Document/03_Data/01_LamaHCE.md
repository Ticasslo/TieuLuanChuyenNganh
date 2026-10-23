# LamaH-CE — Ghi chú thực thi

> Tài liệu ghi các quyết định và kết quả thực thi với bộ dữ liệu LamaH-CE. Dữ liệu chính của đề tài là Extended LamaH-CE theo ngày của bài cơ sở BiasCast (Mục 6). Mục 1–5 ghi gói LamaH-CE gốc theo giờ (tải ban đầu theo mã của Kirschstein & Sun, ICML 2024), giữ cho các hướng khóa luận: dữ liệu giờ, mạng sông. Mã: mỗi notebook Kaggle một tệp `.py` tự đủ — `Workspace/01_Download/02_LamaHCE_Download_Core.py`, `03_LamaHCE_Download_Extra.py`. Thông tin tổng quan về bộ dữ liệu: `Document/05_Survey/01_Datasets.md` Mục 2. Trích dẫn mã tải gốc: Kirschstein & Sun, PMLR 235:24713–24725, https://proceedings.mlr.press/v235/kirschstein24a.html. Trích dẫn dữ liệu: Klingler và cs. (2021), ESSD 13:4529–4565, DOI 10.5194/essd-13-4529-2021; dữ liệu DOI 10.5281/zenodo.4525244.

## 1. Nguồn dữ liệu

| Mục | Nội dung |
|---|---|
| Bản dùng | v1.0, Zenodo `10.5281/zenodo.5153305`, tệp `1_LamaH-CE_daily_hourly.tar.gz` |
| Dung lượng | 14,8 GB nén; khoảng 70 GB giải nén (theo trang Zenodo) |
| Giấy phép | CC BY-SA 4.0 — ghi nguồn theo thư mục `Info`; dữ liệu phái sinh chia sẻ công khai phải dùng cùng giấy phép |
| Cấu trúc | 7 phần (bài báo ESSD 2021): lưu vực A (toàn thượng nguồn), B (trung gian), C (trung gian, ít tác động), trạm đo D, mạng sông, mô hình thủy văn COSERO, thư mục Info. Mỗi phần A–D gồm thuộc tính, chuỗi thời gian (ngày và giờ) và shapefile |

## 2. Dữ liệu Kirschstein & Sun sử dụng

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
| `lamah_ce_core.zip` (`02_LamaHCE_Download_Core.py`) | Toàn bộ B và D trừ chuỗi ngày; thư mục Info | Đủ dữ liệu tái lập Kirschstein & Sun, đủ mọi năm 1981–2017 và mọi cột |
| `lamah_ce_extra.zip` (`03_LamaHCE_Download_Extra.py`) | A (trừ chuỗi ngày), thuộc tính và shapefile của C, mạng sông, COSERO, Info | Baseline không đồ thị dùng khí tượng toàn thượng nguồn; thuộc tính tĩnh (LOAN, EA-LSTM); bản đồ demo; baseline mô hình vật lý |

- Không tải chuỗi theo ngày (tính lại được từ chuỗi giờ; bản chỉ có chuỗi ngày nặng 1,5 GB, tải riêng khi cần) và chuỗi thời gian của C.
- **Lưu trữ lâu dài:** tạo Kaggle Dataset (Private) từ output của mỗi notebook; các notebook sau gắn bằng *Add Input*. Dataset riêng tư có thể chia sẻ cho cộng tác viên.
- Mã nguồn và 957 checkpoint (1,9 GB; 162 cho thí nghiệm chính ở Bảng 2 = 18 cấu hình đồ thị × 3 kiến trúc × 3 cách chia, 3 cho MLP, 108 cho ablation, 684 cho mạng con) của Kirschstein & Sun lấy bằng `git clone` repo `github.com/nkirschi/neural-flood-forecasting`, không đưa vào dataset.

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
| Kaggle Dataset `lamah-ce-ext` (bản 1, 27/9/2026) | Kaggle tự giải nén hai tệp ZIP khi tạo dataset: thư mục `lamah_ce_ext/LamaH_extended/` (`A_basins_total_upstrm`, `D_gauges`), `biascast_experiments/Experiments/` (5 thư mục cấp một `Baseline`, `CrossDomain`, `Encoder_Decoder_LSTM`, `Sequential_Forecast_LSTM`, `TransferLearning`, gồm 11 nhóm thí nghiệm con, và `basins_filtered.txt`), cùng `biascast_data_quality.csv`, `biascast_persistence_cdf.png`; 2.116 tệp, 3,45 GB. Notebook sau đọc thẳng các thư mục này, không đọc ZIP |
| Kiểm tra chất lượng và persistence | % thiếu `qmax` và `qmean` theo kỳ train 2003–2009 / validation 2010–2013 / test 2014–2017 của 451 lưu vực; % thiếu từng biến khí tượng 2003–2017 và ngày có ECMWF đầu tiên; persistence trên test (qmax(t−1) và qmean(t−1) làm dự báo cho qmax(t)) so với NSE theo lưu vực của 5 cấu hình tác giả; hình CDF. Mã: persistence tính trong notebook bước A (`Workspace/03_Baseline/`), % thiếu trong notebook khám phá (`Workspace/02_Exploration/`) |
| Kết quả | Đã chạy trên Kaggle (26/9/2026), số liệu ở Mục 6.1; `lamah_ce_ext.zip` 1.991 file (2,81 GB gốc → 0,95 GB, 10,0 phút), `biascast_experiments.zip` 123 file (0,64 GB gốc → 0,23 GB, 2,2 phút), lỗi CRC `None` |

### 6.1. Kết quả kiểm tra (Kaggle, 26/9/2026)

**Tệp.** `lamah_ce_ext.zip`: `A_basins_total_upstrm` (3 tệp thuộc tính, 859 tệp chuỗi ngày, 7 tệp phục vụ bản đồ), `D_gauges` (3 tệp thuộc tính, 882 tệp chuỗi ngày, 230 tệp `gaps`, 7 tệp phục vụ bản đồ). Tổng 1.112 tệp trong `D_gauges/2_timeseries` gồm 882 tệp `daily` và 230 tệp `gaps`, không tương ứng với 1.112 trạm. `biascast_experiments.zip`: 5 thư mục cấp một chia thành 11 nhóm thí nghiệm con (Baseline 3 nhóm, CrossDomain 2, Encoder–Decoder và Sequential mỗi loại có/không Q, học chuyển giao 2). Kết quả tác giả có 451 lưu vực.

**Tệp khí tượng lưu vực** có 44 cột: `DOY`; 21 biến ERA5-Land; 8 biến ECMWF HRES (`t2m`, `d2m`, `msl`, `ssrd`, `tp`, `lsp`, `cp`, `e`); 8 biến E-OBS (`tg`, `tn`, `tx`, `rr`, `pp`, `fg`, `hu`, `qq`); `MSWEP_RR`; `GLEAM_ETA`, `GLEAM_ETP`. Cấu hình của tác giả không dùng `ECMWF_msl`, `ECMWF_lsp`, `ECMWF_cp`, `EOBS_hu`. Lưu lượng (`qmean`, `qmax`) nằm ở tệp trạm `D_gauges`, không nằm trong tệp khí tượng.

**Lưu lượng thiếu (451 lưu vực).** Không có ô mang mã −999. `qmax` và `qmean` thiếu cùng ngày.

| Kỳ | Ngày-lưu vực lý thuyết (451 × số ngày) | Dòng có trong tệp | Ô trống trong các dòng có | Tổng thiếu | Tỷ lệ |
|---|---|---|---|---|---|
| Train 2003–2009 | 1.153.207 | 1.115.347 | 3.040 | 40.900 | 3,55% |
| Validation 2010–2013 | 658.911 | 658.581 | 2.300 | 2.630 | 0,40% |
| Test 2014–2017 | 658.911 | 658.911 | 1.545 | 1.545 | 0,23% |

Tỉ lệ thiếu tính trên đủ mọi ngày của kỳ. Lưu vực có thiếu: train 45 (29 thiếu trên 10%), validation 10 (5), test 10 (3).

Phần thiếu tập trung ở train, chủ yếu do **27 lưu vực bắt đầu đo sau 01/01/2003** (tệp không có dòng trước ngày bắt đầu): 104 (04/06/2003), 839, 876 (01/01/2004), 110 (30/04/2004), 341 (20/08/2004), 822, 823, 824 (01/11/2004), 329 (04/08/2005), 327 (21/10/2005), 298 (28/10/2005), 92 (01/01/2006), 485, 622 (01/01/2007), 590 (10/05/2007), 577 (07/08/2007), 608 (14/09/2007), 309 (28/10/2007), 561 (01/01/2008), 391 (22/01/2008), 791 (01/03/2008), 296 (17/12/2008), 295 (01/01/2009), 800 (24/11/2009), 623 (01/01/2010), 600 (23/03/2010), 324 (07/09/2010). Ba lưu vực cuối (623, 600, 324) không có Q trong cả kỳ train nhưng vẫn được đánh giá ở test. Với mô hình có Q trong hindcast 365 ngày, bộ lọc mẫu của NeuralHydrology còn bỏ thêm năm đầu sau ngày bắt đầu đo của các lưu vực này. Ngày cuối có `qmax`: 26/11/2017 – 31/12/2017.

**Khí tượng 2003–2017** (2.471.029 ngày-lưu vực = 451 × 5.479, đủ dòng): gần như không thiếu; cột thiếu nhiều nhất `EOBS_hu` 0,36% (không dùng trong cấu hình), mọi cột khác 0,00%. Dự báo ECMWF HRES bắt đầu 01/01/2002 (notebook giới thiệu, Phần 1). Tệp khí tượng bắt đầu 01/01/1981 ở mọi lưu vực; trước 01/01/2002 các ô ECMWF HRES để trống. Với hindcast 730 ngày, mẫu ngày t cần quá khứ từ t − 729, nên mẫu trước 31/12/2003 có khung bắt đầu trước 01/01/2002: khí tượng tái phân tích đủ, nhưng bộ lọc mẫu gốc kiểm mọi cột trên cả khung, kể cả các ô ECMWF HRES mô hình không dùng, nên gần hết mẫu train năm 2003 bị bỏ; với bộ lọc đã sửa ở bước B2 (chỉ yêu cầu nhãn ngày t và thuộc tính) thì không mất mẫu.

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
- Dữ liệu thiếu ít (test 0,23%, khí tượng gần như đủ; train 3,55% chủ yếu do 27 trạm bắt đầu đo muộn). Tuy vậy bộ lọc mẫu của bài loại mẫu khi bất kỳ cột nào thiếu trong cửa sổ 365 ngày, kể cả cột ECMWF; ECMWF có từ 2002 nên năm 2003 không bị loại vì ECMWF; mẫu bị loại chủ yếu ở năm đầu của các trạm đo muộn và các ngày thiếu Q. Sửa bộ lọc (bước B2) lấy lại phần này; masked mean còn có giá trị chịu dữ liệu mất khi vận hành (`Document/04_Ideas/01_Ideas.md` Y1, Y2).

### 6.2. Khám phá dữ liệu

Theo góp ý của GVHD (`Document/02_Meetings/Meeting_2026-09-27.md` Mục 3.1): phần liệt kê mỗi bộ dữ liệu một dòng gồm kích thước, cách phát hành, thống kê cơ bản nhất; thống kê sâu (độ lệch chuẩn, tứ phân vị, khác biệt phân phối train/test) viết thành đoạn thảo luận để biện luận kết quả, đồng thời phục vụ quyết định khi dựng bộ nạp dữ liệu. Mã khám phá đặt ở `Workspace/02_Exploration/01_A1_LamaHCEExt_Explore.py` (bước A1, báo cáo đầy đủ ở `01_A1_LamaHCEExt_Explore.md` cùng thư mục), chạy trên Kaggle (CPU), gắn dataset `lamah-ce-ext`.

Extended LamaH-CE gồm 859 lưu vực có chuỗi khí tượng, 882 trạm có chuỗi lưu lượng ngày; 40 biến khí tượng (ERA5-Land 21, ECMWF HRES 8, E-OBS 8, MSWEP 1, GLEAM 2) và 62 thuộc tính tĩnh cho 859 lưu vực. Mọi tệp khí tượng có đủ dòng ngày từ 01/01/1981 đến 31/12/2019. Trạm lấy từ `Gauges.shp` (EPSG:3035), ranh giới từ `Basins_A.shp` (859 đa giác).

Kiểm GIS tại A1 cell 3.5 xác nhận Basins_A.shp có 859 Polygon/64 cột, Gauges.shp có 882 Point/30 cột, cùng CRS EPSG:3035 với đơn vị mét. Cả hai không có hình học thiếu, rỗng hoặc không hợp lệ trong tập không rỗng; các cận tọa độ hữu hạn. Khóa ID duy nhất và khớp đúng tập CSV A/D tương ứng, chứa đủ 451 ID được chọn. Những phép kiểm này cho phép ghép bản đồ bằng ID, chưa xác nhận vị trí trạm so với đa giác cùng ID hoặc diện tích hình học. Bản đồ dùng geometry của lớp; không suy lon/lat là độ chỉ từ tên cột. Lưu vực mức A có thể lồng nhau, nên không cộng diện tích để suy tổng diện tích vùng không trùng.

Bản đồ A1 cell 3.6 ghép được 859 đa giác với 859 trạm thuộc A; tập chọn có 349 trạm Áo, 88 Đức, 9 Thụy Sĩ và 5 Séc, tổng 451. Tập A còn có một trạm Liechtenstein nhưng không nằm trong danh sách chọn. Áo chiếm 77,38% số ID được chọn; số ID theo quốc gia không phải tỷ trọng diện tích lãnh thổ. Hình gồm tập 859/451, diện tích trên thang log, độ cao trung bình lưu vực và quốc gia đặt trạm; đã kiểm chữ/chú giải trên PNG thực tế. Xem [bộ bản đồ A1](../../Workspace/02_Exploration/Figures/A1_P3_basin_maps.png) và thuyết minh A1 Mục 17.13.

Phân phối diện tích ở A1 cell 3.7 khác nhau giữa toàn bộ A và tập chọn: area_calc của 859 có trung vị 178,421 km², p99 36289,86232 km², max 131247,215 km²; của 451 tương ứng 101,465, 1760,49450 và 15025,609 km². Trung vị tập chọn thấp hơn khoảng 43,13%; đây là mô tả quan sát, chưa chứng minh tiêu chí tác giả lọc. Scatter area_calc–area_gov sát đường 1:1 trả lời độ khớp hai diện tích của cùng ID, không chứng minh hai tập có cùng phân phối diện tích. CSV histogram có đủ 180 dòng và sáu nhóm cộng đúng số lưu vực/100%; đã lưu cùng [hình diện tích](../../Workspace/02_Exploration/Figures/A1_P3_area_distributions.png) trong đầu ra A1 của repo. Ô log thể hiện rõ phân bố hơn ô tuyến tính có bin đầu chứa 93,94645% A và 99,55654% tập chọn; chưa đổi đầu vào huấn luyện từ phép vẽ này.

A1 cell 3.8 có bốn hình phân phối đủ 33 thuộc tính: 32 trường số và lc_dom phân loại. Các trung vị và tần số mã lớp khớp lần kiểm trước; chênh tỷ trọng lớp lớn nhất giữa A và tập chọn khoảng 2,423537 điểm phần trăm ở mã 211. Trong 451, bare_fra và lake_fra có trung vị 0; tỷ lệ ở bin đầu không phải tỷ lệ số 0. Một số histogram cần đọc cùng mức số thập phân và cách chia bin; mvert_dist trên trục tuyến tính bị dồn đầu. Phép kiểm 3.9 bổ sung ECDF, đếm số 0/lưới thập phân, quan hệ trong dòng, Pearson/Spearman, scaler tĩnh và quan hệ hình học, với kết quả dưới đây. Không tự bỏ biến tương quan, cắt GVF hoặc sửa scaler từ biểu đồ phân phối. Xem thuyết minh A1 Mục 17.15–17.17.

Output 3.9 xác nhận số 0 thật trong 451: bare_fra 270/451=59,866962%, lake_fra 382/451=84,700665%, urban_fra 63/451=13,968958%; sáu quan hệ nội bộ không gắn cờ hàng nào ở A/451. Pearson/Spearman của elev_mean–elev_med là 0,998055/0,995125, lai_max–lai_diff 0,990917/0,984007, còn gvf_max–gvf_diff 0,402956/0,282500. 23/24 scaler YAML đọc được có đủ 33 mean/std khớp 451 hàng trong dung sai rtol=10⁻⁶, atol=10⁻⁸, std dùng ddof=1; run TL_AllWeights còn thiếu YAML ở đường dẫn kiểm. Các z cực trị gồm area_calc +16,142492 tại 612, lake_fra +13,125663 tại 697, urban_fra +10,476958 tại 363; ID 55 cực tiểu ở NDVI/GVF, chuyển hồ sơ Phần 9 nhưng chưa gán nguyên nhân cho dự báo NaN. Cả 859 trạm tương ứng A được đa giác cùng ID covers, gồm cả biên; max chênh diện tích hình học–area_calc là 0,007759% ở A và 0,007299% ở 451. Chưa chứng minh độ chính xác ranh giới ngoài thực địa.

Phần 3 đã chốt sau output 3.10. lon/lat của CSV và shapefile bằng đúng tọa độ x/y EPSG:3035, sai khác tối đa 0 m ở 882/451 trạm. Mạng chọn có 322/451=71,396896% đầu nguồn và 129 lồng nhau, mọi đường NEXTDOWNID được giải; số HIERARCHY gốc=1 là 316. Kết quả 71,4% khác 72,5% trong BiasCast, nguyên nhân chưa xác nhận. Tổng năm tỷ phần có max 1,001, vượt 1 ở 16 ID của A và 10 ID được chọn; mức vượt nhỏ phù hợp về độ lớn với giả thuyết làm tròn nhưng chưa chứng minh nguyên nhân, chưa tự chuẩn hóa lại. Run TL_AllWeights_RA_Complex_Embedding không tìm được scaler cục bộ hoặc base_run_dir khai báo để đối chiếu; kết luận vẫn là 23/24 run đã kiểm. Mục 17.19 thuyết minh A1 tổng hợp chi tiết và bàn giao ba giới hạn này; chúng chưa buộc thay tập 451 hay hợp đồng 33 thuộc tính.

Cell 4.1 kiểm 40 biến khí tượng trong 451 lưu vực trên lịch đầy đủ của train/validation/test và xác nhận chỉ EOBS_hu thiếu: 1980/1153207=0,171695% ở train, 3049/658911=0,462733% ở validation, 3910/658911=0,593403% ở test. Tổng 8939 ô, đều rỗng có dòng; không có thiếu dòng, −999, nội dung không chuyển số hay vô hạn trong phạm vi đã kiểm. 39 biến còn lại có 0 ô thiếu; số âm hữu hạn khác −999 vẫn được tính hợp lệ. Tỷ lệ của cả tám biến E-OBS nhỏ hơn tám lần do mẫu số lớn hơn, không thay tỷ lệ riêng EOBS_hu. Hình đã nhận và lưu; thời gian quét 189,8 giây. Chưa suy số mẫu cửa sổ bị loại từ số ô thiếu. Cell 4.2: Kết quả thực tế: cả ba kỳ khớp các số đếm ở Phần 2. Toàn bộ thiếu qmax trong ba kỳ là không có dòng hoặc ô rỗng; không có qmax âm, −999, nội dung không chuyển số hay vô hạn. Số 0 vẫn là nhãn hợp lệ. Mẫu số từng năm là 451×365=164615, hoặc 451×366=165066 khi nhuận. Từ 1981 đến 2017, số lưu vực có ít nhất một ngày hợp lệ tăng từ 312 lên 451, nhưng không tăng đơn điệu: năm 2014 chỉ có 450. Năm 2010 cũng chỉ có 450, dù ngày Q hợp lệ đầu tiên muộn nhất của tập chọn là 07/09/2010. Do đó, không thể mô tả số lưu vực có Q bằng một đường chỉ cộng dồn ngày bắt đầu. Hai năm này có ít nhất một lưu vực thiếu cả năm; ID cụ thể và vị trí đoạn thiếu cần được truy bằng bảng chi tiết, chưa đọc từ vệt màu hoặc đoán nguyên nhân. Danh sách bắt đầu sau 01/01/2003 vẫn gồm 27 ID. Ba ID 623, 600, 324 lần lượt có ngày Q đầu tiên 01/01/2010, 23/03/2010, 07/09/2010 và đều có 0 nhãn train. ID 800 bắt đầu 24/11/2009, chỉ có 38 nhãn train hợp lệ. Đây là số nhãn, chưa phải số cửa sổ có thể huấn luyện.

Cell 4.3: Kết quả thực tế: có 362 đoạn thiếu với tổng 616264 ngày–lưu vực trong lịch 1981–2017. Không có chuỗi hoàn toàn không có Q hợp lệ. Các loại vị trí cộng đúng tổng, nhưng số lưu vực giữa các vị trí có thể chồng nhau. Các tổng tự đối chiếu: 601693+14571=616264; 491+14045+35=14571 ô rỗng; 451×13514−5493121=601693 ngày không có dòng. Không có giá trị không hợp lệ khác trong các đoạn. Tất cả 220 đoạn bên trong là ô rỗng có dòng, không có đoạn thiếu dòng bên trong. Đây là mô tả trạng thái dữ liệu, chưa biết lý do mất quan trắc.

Cell 4.4: Kết quả thực tế: 8939 ô EOBS_hu thiếu được chia thành 7288 đoạn, xuất hiện ở 412/451 lưu vực (91,352550%). Có 39 lưu vực không thiếu EOBS_hu trong toàn lịch 2003–2017. Tỷ lệ ô thiếu thấp không đồng nghĩa chỉ một nhóm nhỏ lưu vực bị ảnh hưởng. Năm 2014 có tỷ lệ thiếu cao nhất, 1852/164615=1,125049%, với 330 lưu vực có ít nhất một ô thiếu. Diễn biến không tăng đơn điệu: sau 2014, số thiếu giảm tới 2017. Không dùng ba tỷ lệ gộp train/validation/test để kết luận thiếu tăng đều từng năm. Số ID có thiếu cũng không nhất thiết đồng biến với số ô: năm 2010 thiếu 624 ô ở 91 ID, năm 2011 thiếu 595 ô ở 144 ID.

Tháng 11 và 12 có tỷ lệ gộp cao nhất. Chúng đóng góp 3619/8939=40,485513% tổng ô thiếu. Đây là phân bố theo tháng trên toàn bộ 15 năm; chưa chứng minh từng năm hoặc từng lưu vực đều có cùng hình thái mùa, và chưa xác định nguyên nhân. Mẫu số tháng 2 là 451×(15×28+4)=191224, tính bốn ngày nhuận 2004/2008/2012/2016. Không so mùa chỉ bằng số đếm chưa hiệu chỉnh độ dài tháng.

ID 55 đóng góp 33/41 đoạn và 79,007147% số ngày của cụm này. Do đó, cụm dài không đại diện đồng đều cho toàn bộ 451 lưu vực. Các khoảng lặp ở ID 55 thường nối mùa thu năm trước tới mùa xuân năm sau; ví dụ 01/11/2003–29/04/2004 và 02/11/2016–29/04/2017. Tuy nhiên, ID 551 chỉ có sáu đoạn, nằm trong 2001–2007; ID 203 chỉ có một đoạn 2009–2010. Không suy cả ba đều thiếu mùa đông gần như mọi năm. Dạng hai cụm trên histogram cũng chưa tự xác nhận phân phối hai đỉnh về mặt thống kê.

ID 55 có đoạn 537 ngày từ 01/05/2000 đến 19/10/2001, nên phát biểu “không có đoạn Q hợp lệ đủ 365 ngày trong toàn lịch sử” không đúng. Trên giai đoạn 2003–2017, đoạn dài nhất chỉ 225 ngày; ba kỳ riêng lần lượt 185/218/225. Kết quả củng cố việc cần kiểm các cửa sổ có Q, nhưng chưa xác nhận nguyên nhân dự báo NaN: tính mẫu phải xét qmean thực sự làm đầu vào, vị trí ngày cần Q trong cửa sổ, khởi động trước kỳ và bộ lọc gốc/sửa. Phần 7 kiểm số mẫu; Phần 9 tổng hợp hồ sơ ID 55. Không thay số NaN bằng một kết luận nhân quả từ biểu đồ.

Kiểm kê thuộc tính A xác nhận `Catchment_attributes.csv` có 859 dòng và 63 cột, gồm ID và 62 thuộc tính; `Stream_dist.csv` có ID và `dist_hup`; `Water_balance.csv` có ID và chín trường P, PCRPS, PERA5, PMSW, PET, ET0, ETA, Q1, Q2. Cả ba tệp có khóa ID duy nhất, hợp lệ và khớp đúng tập A, chứa đủ 451 ID được chọn. Cả 24 cấu hình có cùng 33 tên và cùng thứ tự khai báo `static_attributes`, tất cả lấy từ Catchment_attributes; 29 trường không được yêu cầu gồm hai trường mùa, `glac_fra`, 10 trường đất và 16 trường địa chất. Sự hiện diện của tên chưa xác nhận giá trị đầy đủ/hợp lý. Tài liệu giải thích chi tiết, nguồn và ví dụ nằm tại [thuyết minh A1](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore.md).

Phép quét khí tượng phân biệt ngày có dòng với ngày có giá trị của từng biến: không phát hiện ô −999 hay cặp (lưu vực, biến) rỗng hoàn toàn. ERA5-Land, E-OBS, MSWEP và GLEAM có giá trị ở hai đầu 01/01/1981 và 31/12/2019 cho mọi cặp (lưu vực, biến), nhưng điều này không loại trừ ô trống ở giữa. ECMWF HRES bắt đầu có giá trị ngày 01/01/2002 cho cả tám biến tại 859 lưu vực. Các biến `cp`, `e`, `lsp`, `ssrd`, `tp` kết thúc ngày 31/12/2017; `d2m`, `msl`, `t2m` kết thúc ngày 31/12/2019, đồng nhất giữa các lưu vực. Trong năm biến forecast mô hình dùng, chỉ `t2m` và `d2m` có đến 2019; input hiện tại chưa đủ để chạy nguyên cấu hình đó trong 2018–2019. Mã khảo sát lưu bảng chi tiết ở `Tables/A1_P2_forcing_coverage.csv` trong thư mục đầu ra notebook.

Toàn bộ 882 tệp Q ngày không thiếu dòng bên trong khoảng đầu–cuối của từng tệp, nhưng có ô rỗng. Trong 451 lưu vực, qmean và qmax đều có 5478550 giá trị hữu hạn không âm, 14571 ô rỗng và không có số âm; qmin có thêm 43 ô âm ở sáu ID 631, 644, 645, 672, 699, 720. Không phát hiện −999, nội dung không phải số trong ô không rỗng, vô hạn hoặc vi phạm thứ tự qmin ≤ qmean ≤ qmax trên các cặp hữu hạn trong toàn bộ D. Sự nhất quán nội bộ chưa xác nhận chất lượng quan trắc hoặc tính đúng của cực trị so với chuỗi giờ.

Trong 451 lưu vực có 2270 ngày qmax=0 tại 11 ID, chiếm 0,041434% ngày qmax hợp lệ toàn chuỗi. Train, validation và test lần lượt có 148, 133 và 256 ngày bằng 0. ID 758 có nhiều nhất, 839 ngày và một đoạn liên tiếp 260 ngày trên toàn chuỗi. Chưa xác định số 0 là dòng cạn thật hay vấn đề quan trắc; không tự đổi thành thiếu. Bản tổng hợp tại Mục 2–16 thuyết minh A1 bao gồm số đo theo kỳ, tập ID và liên hệ với Phần 3–7.

Kiểm chất lượng thuộc tính xác nhận cả 33 trường đầu vào đều đầy đủ, không có −999/vô hạn/nội dung không phải số hoặc cột có độ lệch chuẩn bằng 0, ở cả tập 859 và tập 451. Trong toàn bộ 62 trường chỉ có một ô rỗng ở geol_perme tại ID 651, thuộc tập chọn nhưng không phải đầu vào cấu hình. Trong 451, area_calc có trung vị 101,465 km² và khoảng 3,972–15025,609 km². Mã lc_dom có 12 lớp ở cả hai tập; số lưu vực theo lớp không phải tỷ lệ diện tích. Các giá trị GVF ngoài 0–1 được giữ nguyên theo quy trình nguồn.

Cả 859 ID của A có area_calc và area_gov dương, hữu hạn; area_ratio metadata khớp area_calc/area_gov với sai khác tối đa 0,0005. Trong 451, trung vị và p99 của chênh diện tích tuyệt đối trên area_gov là 0,684452% và 8,685852%; lớn nhất tại ID 205, +33,971246%. Mẫu số đổi nhãn Q trong luồng LamaH hiện tại vẫn là area_gov: hàm đọc dùng normalize_discharge=False nhưng lớp dữ liệu gọi _normalize_discharge ngay sau đó (`lamah.py` dòng 125–133). Công thức elon_ratio tính từ area_calc/mvert_dist có sai khác trung vị 0,000932 trong 451, tối đa 0,012241; có 21/451 ID có giá trị lưu trữ lớn hơn 1. Chưa xác nhận nguyên nhân sai khác hoặc thay thuộc tính/chuẩn hóa dựa trên các trường hợp này.

Điểm đã biết (Mục 6.1, trên 451 lưu vực): Q thiếu train 3,55% do 27 trạm bắt đầu đo muộn, test 0,23%; khí tượng gần như đủ, có từ 1981; ECMWF có từ 2002.

Đối chiếu khóa ID của thuộc tính trạm xác nhận `Gauge_attributes.csv` chứa đúng 882 ID của D, còn hai bảng `Hydro_indices_1981_2017.csv` và `Hydro_indices_1989_2009.csv` chứa đúng 859 ID của A. Cả ba bảng đều chứa đủ 451 ID được chọn. Toàn bộ 23 trạm có D nhưng không có A mang mã `degimpact=x`; mã này thuộc nhóm không xét trong phân định lưu vực theo [Bảng 2 LamaH-CE](https://essd.copernicus.org/articles/13/4529/2021/essd-13-4529-2021-t02.xlsx), chưa xác định lý do riêng của từng trạm. Thuộc tính `gaps_pre` và `gaps_post` dùng đơn vị ‰ cho chuỗi giờ trước và sau xử lý, theo [Bảng A1 LamaH-CE](https://essd.copernicus.org/articles/13/4529/2021/essd-13-4529-2021-t03.xlsx).

Điều kiện `degimpact` thuộc `u/l` trong tập A cho 454 ID, gồm toàn bộ 451 ID trong `basins_filtered.txt` và ba ID 319, 410, 592. Vì vậy, chỉ riêng điều kiện không/ít tác động con người chưa tái tạo được danh sách thí nghiệm. [BiasCast Mục 2.1](https://hess.copernicus.org/articles/30/5067/2026/index.html#section2) mô tả nhóm 451 theo mức tác động và cấp tổng hợp A; lý do loại thêm ba ID trên chưa được xác nhận.

Ba trạm này đều ở Áo và mang mã `degimpact=l`. Ngày có qmax hợp lệ đầu tiên của 319, 410 và 592 lần lượt là 20/11/2012, 01/01/2013 và 01/01/2012; ngày cuối đều là 31/12/2017. Cả ba không có dòng Q trong kỳ train 2003–2009. Validation lần lượt có 407, 224 và 731 ngày qmax hợp lệ trên 1461 ngày; riêng 410 có 141 ô trống trong 365 dòng năm 2013. Cả ba có đủ 1461 ngày test 2014–2017. Điều kiện có nhãn train không phải điều kiện bắt buộc để nằm trong danh sách 451, vì 324, 600 và 623 vẫn được chọn dù không có nhãn train. Khác biệt về năm bắt đầu và độ phủ validation là quan sát, chưa chứng minh tiêu chí loại của tác giả.

Danh sách 451 trùng chính xác với tập lưu vực `u/l` ở mức A có `obsbeg_day` không muộn hơn 2010; đây là mô tả tái hiện, chưa phải tiêu chí lọc do tác giả xác nhận. Trong nhóm `u/l`, không có trạm bắt đầu năm 2011: nhóm được chọn có năm bắt đầu muộn nhất 2010, còn ba trạm ngoài danh sách bắt đầu 2012–2013. Vì vậy, ranh giới ≤2011 cũng tái hiện cùng danh sách và dữ liệu chưa phân biệt được ngưỡng ban đầu. Không thể thay cách diễn đạt này bằng yêu cầu có số liệu ngay từ đầu validation: 600 và 324 bắt đầu lần lượt ngày 23/03/2010 và 07/09/2010, sau ngày 01/01/2010.

Trong nhóm `u/l` của A, bảy ID 32, 94, 822, 823, 824, 839 và 876 có năm `obsbeg_day` khác năm qmax hợp lệ đầu tiên trong tệp Extended. Theo [BiasCast Mục 2.1](https://hess.copernicus.org/articles/30/5067/2026/index.html#section2), qmin và qmax được trích từ dữ liệu giờ. Sáu trạm có năm qmax bắt đầu trùng `obsbeg_hr`: 32 (Áo, 1990), 822, 823, 824, 839 và 876 (Séc, 2004). `obsbeg_day` của 839 là 1993, của bốn trạm Séc còn lại là 1981. Cả ba biến qmin, qmean, qmax bắt đầu cùng ngày ở từng trạm trong bảy ca; chưa xác nhận được việc bản Extended cắt chuỗi qmean hay dựng lại qmean từ chuỗi giờ.

Trong 27 lưu vực bắt đầu có qmax sau 01/01/2003, 22 có năm bắt đầu trùng `obsbeg_day`; năm lưu vực Séc trên có năm bắt đầu trùng `obsbeg_hr`. Đây là đối chiếu ở mức năm, chưa xác nhận nguyên nhân hoặc ngày bắt đầu bằng chuỗi giờ gốc. [LamaH-CE Mục 4.1](https://essd.copernicus.org/articles/13/4529/2021/#section4) mô tả chuỗi giờ liên tục của Séc bắt đầu sau 2005, trong khi năm trạm được kiểm có metadata `obsbeg_hr=2004` và các biến Q trong Extended bắt đầu năm 2004. Chênh lệch này cần được giữ rõ khi trích dẫn, chưa xác định nguyên nhân; không thay mốc quan sát bằng mốc mô tả khái quát trong bài.

Trạm 94 (Áo) có `obsbeg_day=1981`, `obsbeg_hr=1976`, tệp Extended có dòng từ 01/01/1981 nhưng cả ba biến Q có giá trị hợp lệ đầu tiên ngày 01/01/1982. Đọc trực tiếp năm 1981 xác nhận đủ 365 dòng và mỗi biến có 365 ô rỗng, không có số âm, −999, nội dung không phải số hoặc vô hạn. Hai thuộc tính bắt đầu chưa giải thích được mốc này; nguyên nhân thiếu cần đối chiếu dữ liệu gốc.

Quét gaps ở 4.5 xác nhận 230 tệp, 780671 mốc giờ duy nhất, không trùng và đều sắp tăng; 215 đoạn ≤6 giờ có 542 mốc, 983 đoạn >6 giờ có 780129 mốc. Trong tập chọn có 97 tệp với 356866 mốc. Theo [LamaH-CE Mục 4.1](https://essd.copernicus.org/articles/13/4529/2021/#section4), đây là danh sách thiếu trước xử lý, các đoạn ≤6 giờ được nội suy; phân loại từ danh sách chưa xác nhận giá trị nội suy trong bản Extended. Trong đúng 451×2003–2017, cả 6885 ô qmax rỗng đều nằm trên ngày có đủ 24 mốc gaps; không có ngày Q hợp lệ chứa đủ 24 mốc. Có 309 ngày–lưu vực Q hợp lệ chứa 1–23 mốc (94/31/184 theo kỳ), là mức đồng xuất hiện chứ chưa phải số ngày qmax sai hoặc đỉnh bị hụt.

ID 410 có 3421 mốc, toàn bộ thuộc hai đoạn >6 giờ: 141 ngày Q rỗng có đúng 3384 mốc (24/ngày), còn 37 mốc nằm trên ba ngày hợp lệ 01/08/2013, 09/10/2013, 30/12/2013 với 12/16/9 mốc và qmax 1,76/2,26/3,00 m³/s. Nếu mẫu số là 43824 giờ, gaps_pre/post suy ra 3421,99704/3420,989088 giờ; mẫu số tác giả dùng chưa xác nhận. Sai số làm tròn gần nhất đến 0,001‰ chỉ tương ứng tối đa 0,021912 giờ, không giải thích được riêng chênh 0,99704 giờ với gaps_pre. Không thêm một mốc giả; xác nhận nguyên nhân sai khác và tổng hợp Q vẫn cần chuỗi giờ. Với ID 94, đoạn 8761 giờ thuộc 31/12/1990 23:00–31/12/1991 23:00, còn năm 1981 có đúng 8760 giờ.

Phần 4 hoàn thành khảo sát thiếu dữ liệu qua 4.1–4.5, kèm tổng kết tại Mục 18.6 thuyết minh A1. Tám CSV gaps nguồn đã nhận được giữ nguyên; bảng giao chi tiết tách không có dòng/ô rỗng là bảng dẫn xuất riêng.

Cell 5.1 đã đo phân phối qmax trên toàn bản ghi của 451 ID: 5478550 ngày hợp lệ, gồm 2270 ngày bằng 0 (0,041434%). Theo m³/s, trung vị/p99/max là 1,62/64,75/1335; đổi bằng area_gov sang mm/ngày cho 1,179497/24,106667/644,119760. Độ lệch phân phối mẫu giảm từ 15,360516 xuống 9,688783, nhưng cả hai vẫn lệch phải rõ. Đây là thống kê gộp theo ngày–lưu vực, mỗi bản ghi dài góp nhiều ngày hơn; chưa phải phép kiểm dạng đuôi hoặc scaler train. Giá trị qmax đổi sang mm/ngày là lưu lượng cực đại quy đổi theo diện tích, không phải tổng thể tích dòng chảy thực tế của cả ngày.

Mười cực đại theo lưu vực ở 5.1 đều thuộc lưu vực nhỏ (area_gov 3,9–41,7 km²), chưa đủ để kết luận các lưu vực nhỏ chi phối toàn bộ đuôi phân phối. Cột ngày chỉ ghi lần đầu đạt cực đại; không có ngày đầu đạt cực đại trong test chưa loại trừ giá trị bằng cực đại lặp lại ở test. Chưa gán lỗi, cắt ngưỡng hoặc loại quan sát. Hình 5.1 đã nhận được giữ nguyên; bốn CSV và cache do Kaggle xuất chưa được nhận cục bộ.

Cell 5.2 đã đọc 24 cấu hình: tất cả dùng lamah_a, nhãn qmax, không khai báo normalize_discharge hoặc custom_normalization riêng cho qmax. Lời gọi helper với normalize_discharge=False chỉ đọc Q thô; lớp LamaH vẫn gọi bước đổi đơn vị bằng area_gov sau đó. Đọc được scaler qmax của 23 run; 22 run giải được cùng phạm vi 451 ID, 2003–2009, khởi động 364 ngày, 1112307 nhãn hữu hạn. Cả 22 đều báo no_candidate_match với tham chiếu float32 và dung sai rtol=10⁻⁵, atol=10⁻⁷: area_gov cho mean/std 2,572048/5,260822, trong khi scaler hiển thị 2,572082/5,262434, lệch khoảng 0,0013%/0,0306%. Chưa tái hiện chính xác scaler, không làm tròn hoặc nới dung sai để nhận khớp. Run EncDec RA không Q Simple dùng train_data_file hoặc periods riêng chưa hỗ trợ; TL_AllWeights vẫn thiếu YAML. Hai run này không được gán kết quả từ các run còn lại.

Cell 5.3 đã phân biệt sai khác số học: 22/22 run tái dựng được cùng phạm vi train đều khớp riêng ứng viên mm/ngày theo area_gov, giữ nguyên rtol=10⁻⁵, atol=10⁻⁷. NumPy float64 cho mean/std 2,572081505851/5,262434050208; ép lưu trữ sang float32 rồi tích lũy float64 vẫn khớp. NumPy float32 và xarray khi tắt tăng tốc cùng cho 2,572082281113/5,262434482574 và khớp cả 22; chỉ xarray float32 với tùy chọn tăng tốc hiện tại cho kết quả lệch ở 5.2. Môi trường có Bottleneck 1.4.2, không có numbagg; kết quả xác nhận tác động của đường tính hiện tại, chưa truy ngược backend huấn luyện tác giả. Có hai cặp scaler qmax khác nhau: 5 tệp chứa 2,572081506154/5,262434054634, 18 tệp chứa 2,572082281113/5,262434482574; trong 18 tệp sau chỉ 17 run được kiểm phạm vi. Max toàn lịch có z≈121,9108 theo các scaler train đã so, không dùng làm ngưỡng gán lỗi hoặc cắt đỉnh.

Run EncDec RA không Q Simple khai báo train_data_file trỏ tới đường dẫn gốc kết thúc bằng /train_data, per_basin_train_periods_file=None; chưa tái dựng nguồn dữ liệu train lưu sẵn. Run này và TL_AllWeights thiếu YAML vẫn not_evaluated. Đối chiếu đơn vị/phạm vi 22 run đã chốt; không sao chép scaler sang hai run còn mở.

Cell 5.4 đã đếm toàn bộ ngày vượt p99 gộp trên bản ghi 1981–2017: ngưỡng 64,75 m³/s hoặc 24,106667 mm/ngày, mỗi đơn vị có 54775 ngày vượt (0,999808% ngày hợp lệ), giữ 2270 số 0 trong phân vị/mẫu số. Số ngày đúng bằng ngưỡng là 15 và 32; hai tập đuôi không được suy là cùng ngày–ID chỉ vì số đếm bằng nhau. Theo m³/s, nhóm area_gov lớn nhất (221,5–15131 km², 113 ID) góp 96,182565% đuôi. Theo mm/ngày, nhóm nhỏ nhất (3,9–51,4 km², 113 ID) góp 61,845733% đuôi nhưng chỉ 23,681010% ngày hợp lệ, tỷ số hai tỷ phần 2,611617. Hai nhóm nhỏ hơn cộng lại góp 83,534459% đuôi mm/ngày so với 48,509295% ngày hợp lệ. Đây là mô tả phụ thuộc đơn vị/ngưỡng trên toàn lịch, không phải quan hệ nhân quả hoặc số sự kiện lũ độc lập.

Top 12 theo số ngày đuôi mm/ngày góp 20147 ngày (36,781378%), đứng đầu 265 (3618), 205 (2554), 531 (2184), 55 (1912). ID 10 đứng đầu max 5.1 không thuộc top 12 đếm ngày này. Với 55, tỷ lệ vượt 27,678054% có mẫu số 6908 ngày quan trắc hợp lệ; Phần 4 đã thấy thiếu theo mùa nên không suy tỷ lệ tương ứng trên mọi ngày lịch. Hai hình 5.4 đã nhận được sao chép nguyên bản, bốn CSV chi tiết chưa được nhận.

Cell 5.5 đã khảo sát mùa trong bảy năm thủy văn 2011–2017 (01/10/2010–30/09/2017), nằm trọn trong khoảng first/last chung 07/09/2010–26/11/2017. Có 2557 ngày, 1153207 ngày–lưu vực kỳ vọng, 1149918 hợp lệ, 3289 thiếu (0,285205%); 436/451 ID đủ cả khoảng. Bốn nhóm elev_mean thực tế là 221–590/593–794/798–1292/1296–3122 m, số ID 113/113/112/113 và số đủ khoảng 110/112/107/107. Trung vị ngang các mean qmax ngày theo tháng đạt mức lớn nhất ở E1 tháng 1 (1,256506 mm/ngày), E2 tháng 6 (1,800817, gần tháng 1 là 1,763132), E3 tháng 6 (3,538713), E4 tháng 6 (8,654398); E4 tháng 2 thấp nhất 1,020396, tỷ số tháng 6/tháng 2 khoảng 8,48. Đây là trung bình cực đại ngày, không phải qmean hoặc tổng dòng chảy tháng; chưa xác nhận cơ chế vật lý từ độ cao.

Đường trung vị từ 436 ID đủ khoảng gần đường dùng mọi quan trắc có mặt: chênh tuyệt đối lớn nhất của các cặp được in là 0,223203 mm/ngày ở E4 tháng 9. Tuy vậy 15 ID thiếu vẫn cần giữ giới hạn riêng. Năm ID 55/683/573/675/682 góp 3128/3289 ngày thiếu (95,104895%); riêng 55 thiếu 1193/2557 ngày (46,656238%). Hai đường nhóm gần nhau chưa chứng minh mỗi ID không bị thiên lệch quan trắc theo mùa. Hai hình 5.5 được giữ nguyên, năm CSV chi tiết chưa được nhận. Cell 5.6 đã đo đường phân vị qmax trong cùng khoảng 2557 ngày, dùng 436 ID đủ khoảng cho đường nhóm và mọi quan trắc có mặt cho các hồ sơ. ID 55 chỉ góp 1364 ngày nên đường của nó vẫn chịu giới hạn thiếu theo mùa. Q tại các mức vượt 1/50/99% của 742 là 0,360224/0,045969/0,009409 mm/ngày; của 758 là 0,556891/0,023204/0,001092, có 11 ngày bằng 0 trong khoảng. Đây là phân vị nội suy, không bảo đảm đúng tỷ lệ ngày vượt nghiêm ngặt khi có giá trị trùng.

Trên toàn lịch 1981–2017, có 49 đoạn bằng 0 ≥10 ngày ở 7 ID (1690 ngày, trung vị 20, dài nhất 260) và 3416 đoạn dương ở 328 ID (51397 ngày, trung vị 12, dài nhất 136). Tổng 53087 ngày tương đương 0,968997% số ngày hợp lệ; hai tập ID có thể giao nhau. Dài nhất bằng 0 là 758, 22/01–08/10/1998; dài nhất dương là 112, 0,099 m³/s, 13/12/2011–26/04/2012. Các đoạn được xác định bằng đúng giá trị lưu trữ và ngày liên tiếp; NaN ngắt đoạn. Đây là cờ đọc lại, chưa phân loại nguyên nhân hoặc loại quan trắc.

Ở test, 742/758 có mean 0,035177/0,015273 mm/ngày, variance ddof=0 là 0,000301/0,000158 (mm/ngày)² nhưng CV 0,493538/0,821864 và 78/57 mức giá trị. Cần gọi là thang Q thấp, phương sai tuyệt đối nhỏ và có đoạn lặp; không gọi cả chuỗi gần như hằng số về biến thiên tương đối. 55 có đoạn bằng nhau dài nhất 4/5/9 ngày ở train/validation/test, chưa kiểm cửa sổ qmean đủ 365 ngày hoặc nguyên nhân NaN. 800 chỉ có 38 nhãn train với 36 mức, không dùng mean/std của giai đoạn ngắn ấy đại diện cả mùa. Hai ảnh 5.6 được giữ nguyên; bảy CSV chi tiết chưa được nhận.

Cell 5.7 đã đọc lại toàn lịch 1981–2017 của 451 ID: mỗi biến có 6094814 ngày–lưu vực kỳ vọng, 5493121 dòng, 601693 dòng thiếu và 14571 ô rỗng; qmax/qmean có 5478550 số hữu hạn không âm, qmin thêm 43 số âm. qmax khớp chính xác cache 5.1. Cả ba phép so thứ tự qmin/qmean/qmax có 0 vi phạm trên 5478550 cặp; số âm vẫn có cờ riêng, không vì đạt thứ tự mà được chứng nhận đúng về vật lý.

Có 2392 ngày qmean=0, gồm 2270 ngày qmax=0 và 122 ngày qmax>0 (10 ID). Tổng 165 dòng có cờ bằng 43 qmin âm + 122 mẫu số 0, không giao nhau. Hai mươi ví dụ được in có qmax 0,001–0,007 m³/s, phù hợp một giả thuyết làm tròn khi chỉ một trong 24 giờ dương; chưa kiểm cơ chế trên toàn bộ 122 ngày hoặc đối chiếu số giờ có mặt. Không thay qmean=0 bằng epsilon. Tỷ số trên 5476158 ngày có qmean>0 có median/p95/p99/max 1,072246/1,655340/2,663639/28; tại một ngày đầu đạt max của từng ID, 451 tỷ số cho 1,824584/6,681692/9,602294/11,133176. Bảng liệt kê ID/ngày tỷ số lớn nhất có mã nhưng chưa có output, chưa xác định ID cho tỷ số 28.

Mười ID có cực đại mm/ngày lớn nhất đều chỉ đạt đúng một ngày: 10/25/23/20/24 trong train, 101/102 trong validation, 30/261/797 trước train, không có ngày đồng cực đại nào ở test. Chưa suy test không có Q rất cao tại các ID khác hoặc theo ngưỡng riêng; phần đó thuộc Phần 8. Sáu cửa sổ Q quanh các đỉnh dùng 10/101/30/261/797/15; qmean tăng cùng qmax là bối cảnh trong cùng nguồn, chưa xác nhận đỉnh từ dữ liệu giờ độc lập. Hai ảnh thật được giữ nguyên; bảy CSV đã xuất trong output chưa nhận cục bộ, thời gian 38,8 giây. Mã hiện hành xuất thêm bảng q_ratio_extremes và dùng log cho ECDF tỷ số dương; chưa có output thật của bảng/ảnh tạo lại đó.

Cell 5.8 đã đo toàn lịch khí tượng 01/01/1981–31/12/2019, 14244 ngày × 451 ID = 6424044 ngày–lưu vực trên mỗi biến. Các loại ô và số ngày từng ID/biến khớp 2.8. Không có thiếu dòng A, -999, nội dung không phải số hoặc vô hạn. ERA5L/MSWEP/GLEAM và sáu cột EOBS pp/qq/rr/tg/tn/tx đủ toàn bộ lịch; EOBS_fg có 31 ô rỗng, hu có 14184. Vì 4.1 không có fg thiếu trong 2003–2017 và có 8939 ô hu thiếu, 31 ô fg và 5245 ô hu còn lại nằm ngoài ba kỳ; ngày/ID và phần giao trước đầu train được định vị ở kết quả 5.9 bên dưới. Các ô ECMWF rỗng khớp phạm vi nguồn 2002–2017 hoặc 2002–2019, không đổi thành số 0.

Tỷ lệ số 0 của ERA5L_prec/ECMWF_tp/EOBS_rr/MSWEP_RR là 20,241689%/18,067349%/56,983218%/31,482334%. Các nguồn không cùng kỳ và ngưỡng lưu số chưa được đối chiếu, chưa xếp hạng độ đúng của ngày khô. Tỷ lệ ET âm ERA5L_total_et/ECMWF_e/GLEAM_ETA/GLEAM_ETP là 3,752636%/2,120544%/0,372974%/0,371137%; giữ nguyên dấu. SWE có median 0,13, p99 3233,87, max 6968,38 theo số lưu trữ (tham chiếu mm), chưa gán lỗi hoặc diễn giải thành độ dày tuyết. ECMWF_cp có 91,275529% số 0, p95=p99 0,01, max 0,06 và mean 0,000963, rất khác thang lsp/tp; chưa gán đơn vị CSV từ độ lớn.

GLEAM_ETA/ETP có 6424044 cặp: 1767201 bằng nhau chính xác (27,509167%), 4656843 ETA<ETP, 0 ETA>ETP. Dung sai 1e-6 không thêm cặp gần bằng. Chênh ETA−ETP có min/median/max −13,3/−0,16/0; trị tuyệt đối lớn nhất tại ID 348 ngày 10/08/1992, ETA 2,49 và ETP 15,79. Hai cột không trùng toàn bộ; chưa suy nguyên nhân các ngày bằng nhau hoặc đồng nhất chúng với ngày ET=0.

Bốn cột LAI cao/thấp và ẩm đất trên/sâu đều có 211068 tháng-năm đủ dữ liệu; tỷ lệ tháng-năm hằng chính xác 5,106885%/62,708227%/1,725984%/32,817860%. Median ngang mean tháng của LAI cao lớn nhất tháng 7 (3,661290), nhỏ nhất tháng 2 (1,436240); LAI thấp 2,2 ở tháng 6–8 và 1,9 tháng 2. Đất lớp trên có max median tháng 3 là 0,382945, min tháng 8 là 0,352630; lớp sâu max tháng 4 0,399325, min tháng 9 0,380667. Không suy các đoạn hằng chứng minh dữ liệu tháng hoặc số được điền.

Sáu ảnh 5.8 và văn bản bảng (2)–(6) được giữ nguyên; chín CSV chi tiết chưa nhận về repo. Thời gian thực tế 312,7 giây; tệp nhị phân tạm được báo đã đóng/xóa, 2,11 GB là giới hạn dự kiến, chưa đo dung lượng đỉnh. Bảng A1 cung cấp tham chiếu cho 36 cột; ECMWF_ssrd vẫn còn bất nhất nhãn W/m²/J/m² và các phép xử lý CSV chưa được xác minh đầy đủ.

Cell 5.9 xác nhận đúng tập ngày–ID E-OBS thiếu với 4.1 ở cả sáu phép so. EOBS_fg có 31 ô rỗng đều trước 2003: 29 ở 55, một ở 536 ngày 30/01/1982, một ở 813 ngày 13/04/1996; không có trong 364 ngày trước train. EOBS_hu trước/train/validation/test/sau kỳ thiếu 4005/1980/3049/3910/1240, tổng 14184. Cửa sổ 02/01/2002–31/12/2002 có 514 ô hu thiếu tại 145 ID, là một phần trong 4005, chưa phải số mẫu mất và chưa xác nhận biến đó có trong đầu vào từng run.

Trên 2635644 bộ ba mưa ECMWF, tp/lsp/cp đều ở lưới 0,01; không có tp−lsp<−0,02. Có 1414144 bộ ba cp=0 nhưng tp−lsp>0,02. Số dư tp−lsp−cp có min/median/max 0/0,19/61,28, mean 1,444995; khi thử cp×1000 là −5/0,07/5, mean 0,482809. Cùng dung sai 0,02, số khớp 991554/992463. Biên làm tròn giả định 0,015/5,010 cho 936280/2635644 bộ ba trong biên; hai biên khác nhau không xác nhận đơn vị. Kết quả phù hợp giả thuyết khác thang kèm làm tròn thô, chưa chứng minh quy trình xử lý CSV; không sửa cp.

Cell 5.10 xác nhận 432 ID không đạt 1000, 13 ID luôn ≥1000, 6 ID có ngày dưới/trên ngưỡng theo thang SWE lưu trữ. Cả 451 đủ 14244 ngày: 6424044 ô, không thiếu; 206964 ô ≥1000 gồm 185172 ở nhóm luôn cao và 21792 ở nhóm qua ngưỡng. Mười ID đầu có min=p01=p05, ngày đầu đạt min đều trong 1981; ID 56 có min/median/max 6701,71/6755,550/6968,38, ID 55 là 2641,28/2766,585/3249,58. Mức cao kéo dài đã xác nhận bằng số lưu trữ, chưa suy nguyên nhân vật lý hoặc kiểm đủ chuyển đổi CSV. Ca 800 có 53 ngày ≥1000, 3165 ngày 0; 604 không đạt ngưỡng, 7394 ngày 0. Bốn đối chiếu với 5.9 đều True; thời gian 35,3 giây. Hình thật được giữ nguyên, năm CSV chi tiết chưa nhận; không bổ sung cell SWE để lặp các cực trị.

Nguồn giờ 2.5 đã tải đủ 452 ID và báo download_verified_md5, gồm 451 ID mô hình và 410. Mốc log cuối 14,80 GB nén ở 122,8 phút, chưa phải thời gian hoặc dung lượng riêng cache Q. 5.9 báo csv_found tại /kaggle/working/Sources/LamaH_CE_v1/D_gauges/2_timeseries/hourly, 452 CSV, đủ 451 ID chọn; header YYYY/MM/DD/hh/mm/qobs/ckhs/qceq/qcol. Ba dòng đầu ID 10 có qobs=1,02 tại 00/01/02 giờ 01/01/1981, cờ 1/0/0; chưa giải nghĩa cờ từ ba dòng. Các số E-OBS/mưa/SWE giữ nguyên; 5.9 lần này 54,1 giây. Hai log gốc đã nhận và giữ nguyên; chưa tổng hợp Q giờ.

Cell 5.11 đã đối chiếu 452 ID trong 1981–2017, 462,4 giây: 131878728 dòng giờ trên 146599872 kỳ vọng, 14721144 giờ không có dòng, 360096 giá trị −999, 111 số âm khác −999; không rỗng/chữ/vô hạn. Mọi số −999 nằm ở mốc gaps đoạn >6h, còn 191 mốc ngắn đều có số; tổng 360287=356866+3421 của 410. 77 đoạn ngắn có đủ biên; 75 nằm trong 0,0005 m³/s, max 0,000666666667, hai ca còn lại cần đọc token trước khi diễn giải làm tròn.

Có 5479642 ngày đủ 24 Q giờ, 593 ngày có 1–23 số, 628093 ngày không có số; cả qmin/qmean/qmax đều có cùng vị trí có/thiếu như Q giờ. qmin/qmax sai khác 0 ở toàn bộ 5480235 cặp, qmean có tất cả cặp trong 0,0005 m³/s, 630278 cặp trong 10⁻⁹. Không có ngày chỉ có Q lưu hoặc chỉ có Q tái dựng. Trừ 410 còn đúng 451 ID: 5477960 ngày đủ giờ, 590 ngày một phần, tổng 5478550 Q có số, 14571 ô rỗng và 601693 dòng thiếu, khớp các phần trước. Mean phù hợp làm tròn 0,001 nhưng chưa xác nhận quy tắc điểm giữa. Q trên ngày thiếu giờ chỉ phản ánh các giờ còn có số, chưa bảo đảm không bỏ lỡ đỉnh thật.

Mười cực đại top mm/ngày đều có đủ 24 Q giờ và khớp max nguồn; tại 10 ngày 22/08/2005, 249 m³/s xuất hiện lúc 22:00. Mười lăm đoạn hằng dài nhất có 1617 ngày đều đủ 24 giờ và khớp; 11 đoạn có mọi Q giờ cùng một mức. 814/818 thực sự hằng theo giờ ở các đoạn đầu 2017, 819 hằng 0,151 ở cả hai đoạn; chưa suy nguyên nhân. Khớp nguồn không xác nhận độc lập quan trắc vật lý.

ID 410 có 1682 ngày đủ giờ, 3 ngày một phần, 141 ngày rỗng trong tệp 1826 ngày. Ba ngày 01/08, 09/10, 30/12/2013 có lần lượt 12/8/15 Q giờ hữu hạn, mean 1,730833/2,233750/2,871333 và qmax 1,76/2,26/3,00, tái tạo các giá trị ngày trong dung sai. Các giờ −999 còn lại là 12/16/9, khớp 37 mốc gaps. Chưa xác nhận nguyên nhân chênh metadata pre/post.

Cờ nguồn có ckhs=0/1 là 9219232/122659496 giờ, qceq=0/1 là 108636295/23242433, qcol rỗng/0/1 là 360202/131508782/9744. Bảng A1 tr.4549 định nghĩa ckhs là đã kiểm cơ quan thủy văn (không mô tả phạm vi), qceq bằng nhau ≥10 bước, qcol ngoại lai tự động có thể là Q thật. Chưa dùng cờ lọc; chín CSV 5.11 chưa nhận. Kết quả cờ/giờ âm và hai ca nội suy được ghi tại 5.12 dưới đây.

Cell 5.12 đọc 27 ID trong 27,1 giây. Riêng 451 ID, 5477960 ngày đủ 24 số giờ và 590 ngày có một phần cho tổng 5478550 cặp; 616264 ngày hai phía thiếu, không có ngày lệch vị trí có số. qmin/qmax đều khớp ở 10⁻⁹, qmean có 630134 cặp ở 10⁻⁹ và toàn bộ trong 0,0005 m³/s. Sáu CSV đã xuất Kaggle nhưng chưa nhận cục bộ.

Hai đoạn ngắn vượt 0,0005 là 622 ngày 01/08/2012 17:00–21:00 (5h, biên 2,872/3,000) và 726 ngày 25/10/2015 15:00–16:00 (2h, biên 0,946/0,989). Max số dư 0,000666667, đều nằm trong biên 0,001 giả định. Bảy điểm giữa đều bằng đường thẳng qua biên lưu làm tròn lên tới 0,001, đã kiểm bằng phân số chính xác; đây là phép khớp trên hai ca, chưa chứng minh thuật toán nguồn hoặc sửa số/dung sai chung. Các giờ giữa đều ckhs=0; không tự đồng nhất cờ với lỗi.

Mười giờ đỉnh đều ckhs=1, qceq=0, mỗi ngày chỉ một giờ đạt max. qcol=1 ở ID 10 (22/08/2005 22:00, 249 m³/s) và 797 (18/07/1987 08:00, 79,475 m³/s), tám đỉnh còn lại qcol=0. Hai giá trị có cờ vẫn hiện diện trong Q giờ/ngày; cờ ngoại lai không chứng minh sai, cờ đã kiểm không bảo đảm hết sai số vật lý. Trong 15 đoạn hằng, 38808 giờ có 38773 qceq=1; không qcol=1/rỗng. Năm đoạn 758 trước 1999 đều ckhs=0 (16680h); các đoạn còn lại đều ckhs=1 (22128h), gồm 814/818/819. Chưa suy nguyên nhân đoạn hằng hoặc trạng thái toàn kỳ từ các ca này.

111 giờ âm thuộc 43 ngày ở đúng sáu ID: 631 (64h/12 ngày/min −0,168), 644 (8/7/−0,045), 645 (4/1/−0,034), 672 (1/1/−0,036), 699 (32/21/−0,265), 720 (2/1/−0,374). Cùng phép min giờ khớp qmin tại 5.11, số âm có từ nguồn giờ, không phát sinh ở tổng hợp ngày. Chưa suy nguyên nhân hoặc tự loại qmean/qmax cùng dòng; trạng thái cờ của toàn bộ 111 giờ chưa có CSV, chỉ 20 dòng đầu được in.

**Phần 5 hoàn thành khảo sát theo kế hoạch**, giữ các giới hạn quan trắc vật lý và nhãn trên ngày thiếu giờ. Đơn vị/dấu/tích lũy khí tượng sang Phần 6; số mẫu/NaN sang Phần 7; lệch kỳ/đỉnh sang Phần 8; hồ sơ trạm sang Phần 9. Phần 6 hoàn thành khảo sát định lượng đến 6.4; kết quả và giới hạn ngữ nghĩa được ghi dưới đây. Phần 7.1 xác định 24 config/8 hợp đồng hiệu lực, 23 config sẵn sàng đếm CSV và một cache cần kiểm; 7.2 đã có số cửa sổ CSV trên 451 ID; 7.3 xác nhận các tệp cohort ứng viên đều khớp 451, 7.4 chốt phạm vi từng kỳ và tổng kết Phần 7, giữ data_dir/cache lịch sử chưa xác nhận.

**Phần 6.1 — phạm vi và mean năm cặp khí tượng:** 451 ID, 62,6 giây, hình đã nhận; năm CSV chi tiết chưa nhận cục bộ. Nhiệt độ/điểm sương có 2964874 cặp trong 2002–2019, bức xạ/mưa/ET có 2635644 trong 2002–2017. Mỗi cặp có đủ 100% lịch ở cả train (1153207), validation/test (658911 mỗi kỳ), không có forecast-only hoặc cả hai thiếu trên toàn lịch; phần trước/sau forecast là reference-only. Năm thường có 164615, năm nhuận 165066 cặp. Không biến mean các năm không có forecast thành 0.

Hiệu forecast−đối chiếu tính từ mean in (xấp xỉ) ở train/validation/test: nhiệt độ +0,532371/+0,299938/+0,141663; điểm sương +0,133687/−0,163255/−0,243635; mưa +0,198588/+0,371429/+0,243896; ET +0,152071/+0,141209/+0,002721; bức xạ +32,102839/+36,515574/+31,042678 theo giá trị lưu trữ. Chưa gọi các hiệu là bias vật lý hoặc nguyên nhân thay đổi theo năm. Cặp net ERA5L/downward ECMWF và nhãn W/m²/J/m² vẫn chưa thống nhất; mã xử lý chuyển đổi CSV chưa tìm thấy trong thư mục Data của mã tác giả (README chỉ trỏ Zenodo), không suy ra từ độ lớn histogram.

Trong từng kỳ, tỷ lệ MSWEP=0 là 31,616353/35,667639/35,830787%, ECMWF_tp=0 là 19,236096/16,455030/18,418269%. ET có 2,406073/2,895991/2,316094% cặp trái dấu khác 0, hai cột vẫn chủ yếu dương; chưa đảo dấu hay chọn ngưỡng mưa. Mean ET test gần nhau chưa cho biết sai khác từng ngày/phân phối triệt tiêu hay không. Kết quả 6.2 được ghi dưới đây, giữ nguyên thang raw và giới hạn ngữ nghĩa nguồn. Chi tiết bảng và hình ở Mục 20 của thuyết minh A1.

**Phần 6.2 — phân phối ghép theo ID/kỳ:** 451 ID, 68,5 giây; cả 15 tổ hợp cặp×kỳ đủ 451 ID, không thiếu cặp/r không xác định, W1 hai cách và bất đẳng thức đều đạt kiểm. Hình gốc đã nhận, năm CSV chi tiết chưa nhận cục bộ. Các số sau là median ngang chỉ số từng ID, không phải chỉ số ngày gộp; Pearson r không có đơn vị, khoảng cách còn trên thang raw.

| Cặp | W1 train / validation / test | MAE train / validation / test | r train / validation / test |
|---|---|---|---|
| Nhiệt độ | 0,515729 / 0,313196 / 0,273285 | 0,865092 / 0,698323 / 0,646304 | 0,992032 / 0,994477 / 0,994271 |
| Điểm sương | 0,421893 / 0,348549 / 0,320856 | 0,810583 / 0,716674 / 0,656010 | 0,991814 / 0,993915 / 0,993829 |
| Mưa | 0,459116 / 0,579295 / 0,610623 | 1,712260 / 1,781711 / 1,752786 | 0,805085 / 0,828564 / 0,833226 |
| ET | 0,165436 / 0,191923 / 0,201834 | 0,446132 / 0,429555 / 0,432272 | 0,900324 / 0,910356 / 0,899123 |
| Bức xạ | 27,488608 / 31,157673 / 26,163005 | 29,918631 / 32,037810 / 27,473566 | 0,959826 / 0,971678 / 0,976999 |

Mưa có r tăng đồng thời W1 tăng, cần phân biệt diễn biến ngày và phân phối biên. ET test có mean gộp gần nhau ở 6.1 nhưng MAE median vẫn 0,432272; median mean_delta −0,008029 khác dấu mean gộp +0,002721 vì khác phép gộp. Nhiệt độ/điểm sương r≈0,99 chưa tách mức tái hiện mùa khỏi biến thiên bất thường. Không suy hệ thống dự báo cải tiến từ khác kỳ hoặc xếp hạng chéo đơn vị.

Ba ca W1 bức xạ test lớn là 56/47/112 với W1 130,322272/113,032149/110,394606, cùng tập ba SWE cực đại cao nhất 5.10; chưa suy cơ chế hay quan hệ nhân quả. Mean_delta=MAE=W1 bằng nhau tới độ chính xác in; phép đếm 6.3 xác nhận cả 1461 ngày test của từng ID đều có forecast lớn hơn đối chiếu. Cặp net/downward và nhãn đơn vị vẫn chưa giải quyết. Kết quả 6.3 được ghi dưới đây, giữ nguyên giới hạn đại lượng/đơn vị. Toàn bộ bảng (1)–(4), hình và giới hạn ở Mục 20.2 thuyết minh A1.

**Phần 6.3 — chênh tháng và dấu các ca W1 lớn:** thời gian 113,0 giây; hình và nguyên văn output đã nhận, năm CSV chi tiết chưa nhận cục bộ. Cả 15 tổ hợp cặp×kỳ đủ 451 ID, không tháng rỗng/r không xác định; số ngày và mean/MAE/RMSE gộp trọng số tháng khớp 6.2, sai khác lớn nhất 5,684342×10⁻¹⁴ raw. Mọi tháng test đủ cặp, gồm 50963 cặp tháng 2 có ngày nhuận 2016.

ET test có median chênh forecast−đối chiếu âm tháng 11–4, dương tháng 5–10: tháng 12 −0,306129, tháng 7/8 +0,357258/+0,359194. Không cộng các median tháng thành mean toàn mạng. ET r tháng ở test chỉ 0,316470–0,798720 so với 0,899123 cả kỳ; chưa lấy hiệu này làm tỷ lệ ảnh hưởng mùa. Mưa test chênh median âm tháng 6–8, dương chín tháng khác; W1 median lớn nhất tháng 5 (1,079516), MAE lớn nhất tháng 7 (2,953468), r thấp nhất tháng 6 (0,759745). Điểm sương test chênh median âm mọi tháng, từ −0,515968 (12) đến −0,061083 (6).

Bức xạ 56/47/112 được xác nhận **1461/1461 ngày test có forecast lớn hơn cột đối chiếu**, không ngày âm/bằng nhau; chưa suy nguyên nhân hoặc giải quyết net/downward/đơn vị. W1=|mean_delta| không bảo đảm cùng ngày một dấu: ID 57 điểm sương vẫn có 176 ngày dương; 265 nhiệt độ có 94 ngày âm. Mưa 478 có 782 ngày dương/511 âm/168 bằng nhưng mean_delta âm, nên tần suất và độ lớn chênh phải đọc riêng.

Chi tiết bảng (1)–(4), hình và bản văn gốc ở Mục 20.3 thuyết minh A1. Cell 6.4 chuẩn bị so tương quan trên cùng ngày trước/sau trừ mean tháng của mỗi nguồn, **chỉ fit mean trên cặp train**; tháng train thiếu giữ NaN, không dùng validation/test điền. Đây là phép chẩn đoán, không đổi dữ liệu đầu vào/scaler hoặc xác nhận kỹ năng dự báo độc lập; kết quả 6.4 được ghi dưới đây.

**Phần 6.4 — r sau trừ mean tháng train:** 451 ID, 62,3 giây; hình và output nguyên văn đã nhận, năm CSV chi tiết chưa nhận cục bộ. Mỗi cặp có 5412 ID×tháng train, dùng 198–217 cặp mỗi tháng, không thiếu mean. Cả ba kỳ giữ nguyên toàn bộ ngày/451 ID và mọi r trước/sau đều xác định; nguồn 6.2 và mean train 6.3 khớp.

| Cặp | r median raw → centered ở test | Median delta_r từng ID | ID giảm / tăng |
|---|---|---|---|
| Bức xạ | 0,976999 → 0,934359 | −0,037656 | 451 / 0 |
| Nhiệt độ | 0,994271 → 0,974278 | −0,020438 | 451 / 0 |
| Điểm sương | 0,993829 → 0,967819 | −0,025444 | 451 / 0 |
| Mưa | 0,833226 → 0,827278 | −0,004908 | 430 / 21 |
| ET | 0,899123 → 0,615688 | −0,283952 | 451 / 0 |

ET centered r median ở train/validation là 0,592640/0,636095, đều giảm ở 451 ID. Trừ mean tháng chỉ loại một thành phần mùa, không suy tỷ phần phương sai hay kỹ năng dự báo độc lập; r còn chứa biến thiên trong tháng/xu thế/tín hiệu chung. Median hiệu không bằng hiệu hai median nói chung. ET 391/390/398 có centered r test 0,315074/0,340275/0,382067; giữ để đọc hồ sơ, không lọc ID hoặc chọn mô hình.

**Phần 6 hoàn thành khảo sát định lượng 6.1–6.4**, giữ các câu hỏi chuyển đổi CSV/quy ước ngày và net/downward W/m²/J/m² ở mục hạn chế. Không tự đảo ET, đổi cp, trừ nền SWE hoặc chọn hiệu chỉnh theo test. Cell đầu Phần 7 kiểm cấu hình thật, biến từng pha, lịch khởi động và ngữ nghĩa path trước khi đếm mẫu 365/730. Với forecast=1 và overlap=0, đây là tổng khung gồm 364/729 ngày hindcast + một ngày forecast; số cửa sổ CSV đã có tại 7.2, giữ riêng giới hạn cohort/cache lịch sử. Chi tiết ở Mục 20.4–21.3 thuyết minh A1.

**Phần 7.1 — hợp đồng cửa sổ:** đã nhận bảng (1)–(5), nguyên văn lưu tại Tables/A1_P7_sample_contract_parsed_output.txt. Có 24 config/8 nhóm hiệu lực, gồm 14 cudalstm, 5 handoff_forecast_lstm, 5 sequential_forecast_lstm; tất cả lamah_a, seq_length=365, predict_last_n=1, overlap=0. Có 10 run hai pha forecast=1, bốn run dùng qmean dynamic. Cả 24 geometry_ready; 23 ready_for_daily_count và một review_required. Bảng vai trò có 24 dòng, lịch 48 dòng; đây chưa là số mẫu hợp lệ.

Trong 120 trường path, 119 không hoạt động sau Config parse; đúng chuỗi "None"/YAML null thành None, các path thật được giữ. Nguồn ngoài duy nhất là train_data_file của Encoder_Decoder_LSTM_RA_without_q_Simple_Embedding, trỏ /home/ok2907/Documents/BiasCast/Experiments/Encoder_Decoder_LSTM/Encoder_Decoder_LSTM_without_q/Encoder_Decoder_LSTM_RA_without_q_Simple_Embedding/train_data. Chưa xác nhận tệp/nội dung cache trong Input. Không thay nó bằng CSV rồi gọi là số mẫu train lịch sử của run.

Hai pha dùng 31 đối chiếu (+qmean ở bốn run) đến t−1 và năm ECMWF ở t. Riêng hai config FCRA dùng cả năm ECMWF ở hindcast, nên phải kiểm cả lịch sử của chúng. Bốn nhóm một pha dùng toàn khung. **EOBS_hu không nằm trong dynamic của các nhóm này**; tỷ lệ thiếu hu không tự là số mẫu mô hình mất. L=365/730 là tổng khung: khởi động train lần lượt 02/01/2002 và 02/01/2001; L=730 là kịch bản, không đổi config nguồn.

**Phần 7.2 — số cửa sổ CSV đã đo:** 451 ID, bảy nhóm ready/23 config, lịch đọc 02/01/2001–31/12/2017, 33 thuộc tính tĩnh không thiếu/không hữu hạn/sentinel; chạy 120,3 giây. Các kiểm nhãn so 4.2, cộng năm, phân hoạch lý do và quan hệ tập con đều đạt. Hình và nguyên văn bảng (1)–(6) đã lưu, bảy CSV chi tiết chưa nhận cục bộ. Config đọc từ YAML nguồn và đối chiếu chữ ký 7.1; area_gov lấy từ gauge, không dùng area_calc thay thế.

Train có 1153207 ngày–ID lịch và 1112307 ngày có nhãn. L=365 không Q giữ 1112307; có Q giữ 1094272, ít hơn 18035 ngày có nhãn do cửa sổ đầu vào thiếu; kiểm theo pha không tăng giao có nhãn. L=730 ở nhóm hai pha 31 đối chiếu / 5 forecast tăng từ 958231 lên 1112307, phục hồi 154076 mẫu ở 425 ID; nhóm có qmean tăng 928114 lên 1079046, phục hồi 150932 ở 417 ID. Toàn bộ phục hồi nằm trong năm 2003. Các nhóm thật sự dùng ECMWF lịch sử (một pha forecast/FCRA hoặc FCRA hai pha) không phục hồi; nhóm chỉ đối chiếu giữ 1112307 ở cả hai L. Đây là chẩn đoán mặt nạ CSV, chưa chạy bộ lọc sửa hoặc mô hình.

Validation/test luôn có 658911 lookup gốc mỗi kỳ. Không Q có giao đầy đủ–nhãn 656281/657366 ở cả hai L. Có Q: L=365 là 652489/652636; L=730 là 648779/649787. Hai quy tắc cho cùng giao nhãn ở evaluation; đủ dữ liệu không đồng nghĩa dự báo hữu hạn và chưa mô phỏng masked mean.

324/600/623 không có nhãn train. Có Q/L=365, thêm **55 và 800** có 0 giao cửa sổ train dù có 1292 và 38 ngày nhãn; hai ID vẫn có 0 ở L=730. Không Q thì 800 giữ 38 mẫu. Có Q/L=730 có tổng 9 ID train không có cửa sổ, bốn ID bổ sung là 295/296/391/791 theo 7.3; ID 55 là trường hợp evaluation không có giao cửa sổ mỗi kỳ có Q, cả hai L. 742/758 có đủ 2557 mẫu train ở L=365 kể cả có Q, nên không dùng thiếu mẫu để giải thích NSE âm của hai ID trong kịch bản này.

**Phần 7.3 — cohort/nguồn và hồ sơ cửa sổ đã đo:** chạy 1,5 giây; 24 config × ba kỳ = 72 danh sách tìm theo phần sau Experiments đều có 451 ID, không trùng, cohort_key 702fe55b22be, bằng selected_ids. Không có danh sách khác tập 451 hoặc chưa đọc được. Đây là các tệp ứng viên tại Input hiện tại, chưa xác nhận tệp lịch sử. Bảng được dán trực tiếp trong chat; năm CSV chi tiết/tệp log riêng chưa nhận cục bộ.

20 config không Q khai báo /home/ok2907/Documents/Data/LamaH_expanded; bốn config có Q khai báo LamaH_expanded_q_input. Các path máy tác giả không tồn tại như khai báo và khác tên cuối của LamaH_extended hiện tại, không tự chứng minh CSV hiện tại thiếu hoặc sai. Cache train duy nhất của Encoder_Decoder_LSTM_RA_without_q_Simple_Embedding chưa tìm được dưới Experiments hiện tại: 0 ứng viên, kích thước NaN, chưa đọc nội dung.

Không Q chỉ có 324/600/623 không có mẫu train vì không nhãn. Có Q/L=365 thêm 55/800 có nhãn nhưng không cửa sổ; L=730 thêm 295/296/391/791, tổng chín ID train không có giao cửa sổ. ID 55 là trường hợp duy nhất không có giao cửa sổ evaluation có Q ở cả hai kỳ/L. Danh sách giống nhau giữa hai quy tắc mặt nạ; không tự là lý do lọc ID hoặc xác nhận cơ chế NaN.

**Phần 7.4 — phạm vi CSV cuối đã đo:** chạy 1,6 giây; nhóm cache W8404b4046b9d khớp mặt nạ W6f046c6c21f2. Giữ hai dòng train L=365/730 của một config là train_cache_content_unknown, không gán số từ nhóm CSV tương đương. Bốn dòng validation/test dùng số CSV của W6f046c6c21f2 sau kiểm target, biến động/tĩnh, offset và lịch giống hoàn toàn. Nhóm cache có lookup 658911 mỗi kỳ, giao đầu vào đầy đủ–nhãn validation 656281 và test 657366, ở cả hai L.

Tổng 144 dòng config/kỳ/L gồm 138 số CSV riêng, bốn số evaluation từ mặt nạ tương đương và hai dòng train còn NaN. Như vậy mỗi L có số train CSV cho 23 config, evaluation CSV cho đủ 24 config. Đây là số trên Input hiện tại/cohort ứng viên, không phải số mẫu huấn luyện lịch sử đã xác nhận; cache/scaler/data_dir lịch sử vẫn có giới hạn nguồn. Nguyên văn output 7.4 đã lưu ở Tables/A1_P7_sample_final_scope_output.txt; ba CSV cuối chưa nhận cục bộ.

**Phần 7 hoàn thành khảo sát cửa sổ CSV**, tổng kết ở Mục 21.5 thuyết minh A1. Giữ nguyên 451 ID, nhãn thiếu và hai dòng cache chưa biết; chưa chạy loader/mô hình, thay bộ lọc hoặc xác nhận cơ chế NaN/NSE. Mặt nạ theo pha chỉ bỏ điều kiện ở ô mô hình không dùng, không bỏ ECMWF lịch sử ở nhóm FCRA thực sự tiêu thụ nó.

**Phần 8.1 — phân phối qmax giữa kỳ đã đo:** chạy 2,6 giây; giữ ngày có nhãn của cả 451, không lọc theo cửa sổ mô hình. Train có 448 ID/1112307 ngày, validation 451/656281, test 451/657366; số qmax=0 tương ứng 148/133/256. Hình thật đã lưu nguyên bản; bảng dán trực tiếp trong chat, năm CSV chi tiết/tệp log nguyên bản riêng chưa nhận.

Các phép so train dùng 448 ID: trung vị tỷ số mean validation/test là 1,056465/0,995390; std là 1,063381/0,901536; W1 là 0,247428/0,290672 mm/ngày. Thống kê riêng kỳ validation/test vẫn dùng 451, không lấy tỷ số hai trung vị thay cho trung vị tỷ số cùng ID. 324/600/623 không có train nên tỷ số/W1/ngưỡng train giữ NaN.

Ngưỡng p95/p99 chỉ fit từ train, so > và báo ngày bằng ngưỡng riêng. Mẫu số đủ điều kiện train/validation/test là 1112307/652230/652983, khác toàn bộ nhãn validation/test vì ba ID không có train góp 4051/4383 ngày. Tỷ lệ gộp p95 là 4,969941%/5,842571%/4,928765%; p99 là 1,009344%/1,476013%/1,180123%. Trung vị tỷ lệ từng ID test lần lượt 4,483231%/0,889802%: cách gộp ngày và trọng số đều theo ID cho kết quả khác nhau, chưa kết luận test ít hay nhiều đỉnh hơn trên mọi lưu vực.

800 chỉ có 38 ngày train, mean/std 1,591469/0,164561; tỷ số mean validation/test 6,009626/5,333317, std 103,024895/87,495430, W1 8,366351/7,283518. Các cực trị so train này phải đọc cùng mẫu train ngắn; không tự gán cơ chế hoặc fit lại ngưỡng bằng held-out. 742/758 có mean test/train 0,372395/0,211303 và std 0,156114/0,100242, thang biến thiên nhỏ nhưng chưa ghép sai số/NSE theo cùng ngày.

**Phần 8.2 — đóng góp đuôi/độ phủ train đã đo:** chạy 3,6 giây; hình và output nguyên văn giữ nguyên byte, sáu CSV chi tiết chưa nhận. Cộng tháng/năm khớp 8.1. ID 800 có đúng 38 ngày train từ 24/11 đến 31/12/2009, chỉ hai loại tháng; p95/p99 1,844900/1,977811 mm/ngày không đại diện đủ mùa. 324/600/623 không có ngưỡng; 295/296 có 365/380 ngày, 791/590 có 671/967 ngày dù đã đủ 12 loại tháng. Đủ loại tháng chưa bảo đảm ngưỡng ổn định.

800 góp 861/32184=2,675242% ngày test vượt p95 và 844/7706=10,952505% ngày vượt p99. Các ID tiếp theo ở p99: 697/590/563/593 có 97/67/66/65 ngày. Phạm vi đối chiếu không có 800 có tỷ lệ p95 test 4,807666% (31323/651522), p99 1,053226% (6862/651522), so với train tương ứng 4,969931%/1,009288%. Trung vị tỷ lệ p99 từng ID test vẫn 0,889802% ở cả hai phạm vi. 800 ảnh hưởng mạnh lên tổng ngày nhưng không giải thích toàn bộ chênh kỳ; không đổi danh sách 451/ngưỡng hoặc coi ngày vượt là sự kiện độc lập.

Theo năm, p95 train 2009 là 8,562663%, p99 validation 2013 là 1,880470%; mẫu số chỉ ngày quan trắc có ngưỡng train xác định, chưa xác nhận cơ chế khí hậu. Số tham chiếu test p99 theo tỷ lệ train thực nghiệm từng ID là 6618,328384 so với 7706 ngày thật; không có 800 là 6579,881015 so với 6862. Hiệu là đối chiếu mô tả, không phải kỳ vọng nhân quả. Chi tiết Mục 22.2 thuyết minh A1.

**Phần 8.3 — kiểm kê NSE đã đo:** chạy 76,3 giây; output nguyên văn giữ nguyên byte, bốn CSV chi tiết chưa nhận. Không có environment.json/export A2 dưới /kaggle/input và /kaggle/working của phiên A1 hiện tại. Lần tái lập A2 đã ghi tại Mục 6.3 là bằng chứng riêng; không dùng kết quả tác giả thay nhãn nguồn chạy lại.

Sáu CSV tác giả đều đúng canonical test/best_model/test_metrics.csv, mỗi run chỉ một candidate; schema basin/NSE/KGE, mỗi tệp đúng 451 ID, không trùng/ngoài/thiếu ID hoặc NSE không phải số/vô hạn. Config qmax và ngày test khớp A1, chưa xác nhận tập ngày/quan trắc thật chấm NSE. NSE hữu hạn theo SeqLSTM_Q/noQ, EncDec_Q/noQ, D_FC, D_RA là 450/451/450/451/451/451; số âm 4/3/13/20/39/3; median 0,705284/0,627573/0,662180/0,568854/0,386676/0,692254.

Hai run có Q có NSE NaN tại 55. 742/758 âm ở cả sáu run; SeqLSTM_Q là −43,940674/−78,295601, EncDec_Q −51,199471/−131,419769; EncDec_noQ −63,028679/−157,913528. 800 dương ở cả sáu run (0,180123–0,624186), dù train riêng ID ngắn và không có cửa sổ train nhóm Q theo CSV hiện tại. Mô hình chung nhiều lưu vực có thể dự báo ID không có mẫu train riêng; chưa quy nguyên nhân sai số từ hồ sơ độ phủ/phương sai. Bảng đủ sáu run và bốn ID ở Mục 22.3 thuyết minh A1.

**Phần 8.4 — ghép mô tả NSE tác giả đã đo:** chạy 1,3 giây; hình/log giữ nguyên byte, bốn CSV chi tiết chưa nhận. 450/451 cặp cho đặc điểm không cần train; 447/448 cho đặc điểm cần train do 324/600/623 không có train và 55 NaN NSE ở hai run có Q. Không cắt NSE âm. Spearman NSE–std test theo sáu run 0,134738/0,374418/0,187813/0,472534/0,562259/0,357729; Pearson tương ứng chỉ 0,073922/0,153573/0,072219/0,072416/0,131294/0,071974.

Không có 742/758 trong phạm vi đối chiếu, Pearson NSE–std test tăng thành 0,138146/0,281682/0,178350/0,306779/0,424646/0,232980; Spearman giảm khoảng 0,005876–0,011640. Không có 800, Pearson NSE–tỷ số std test/train tăng khoảng 0,144053–0,175335, còn Spearman chỉ tăng 0,001307–0,006444. Đây là độ nhạy của hệ số với hồ sơ đã biết, không phải tiêu chí loại ID. W1 thô có Spearman dương 0,147796–0,326982 với NSE, chưa suy lệch phân phối có lợi; W1 phụ thuộc thang và NSE chuẩn hóa phương sai.

324/600/623 không có nhãn train riêng nhưng vẫn có NSE tác giả hữu hạn; có cả dương/âm theo run. Tỷ số/W1/ngưỡng cần train giữ NaN. **Phần 8 hoàn thành khảo sát mô tả**, tổng kết tại Mục 22.5 giữ nguồn NSE tác giả riêng export A2 và giới hạn ngày/quan trắc thực chấm chưa đối chiếu. Không p-value giả định ID độc lập, chọn mô hình/lọc lưu vực bằng test hoặc đổi danh sách 451. Chi tiết Mục 22.4–22.5 thuyết minh A1.

**Phần 9.1 — chuỗi tác giả bốn hồ sơ đã đối chiếu:** chạy 25,2 giây; hình/log nguyên bản giữ nguyên byte, ba CSV chi tiết chưa nhận. Sáu archive đọc được, mỗi tệp 451 khóa; kiểm sâu 24 hồ sơ 55/742/758/800, một time_step, đủ dòng lịch test, không dòng ngoài kỳ. Mặt nạ obs–CSV trùng từng hồ sơ; 55 có 807 ngày obs, ba ID còn lại 1461. Sai khác max 55/742/758 lần lượt 7,902447×10⁻⁶/1,199719×10⁻⁷/1,186967×10⁻⁷ mm/ngày. 800 có một ngày/run vượt 10⁻⁵, max 1,068512×10⁻⁵; không coi mọi obs khớp dung sai hoặc tự nâng dung sai.

Cả sáu config chặn qmax âm khi chấm; giữ sim gốc và bản theo config. 24 trạng thái NSE hữu hạn/NaN khớp, 22 NSE hữu hạn tính lại float64 lệch CSV tối đa 8,661331×10⁻⁶. SeqLSTM_noQ tại 742/758 có 397/581 sim âm trên đủ 1461 cặp; NSE gốc −6,523761/−13,830682 thành −4,263644/−8,714366 theo chính sách tác giả. Đây không phải hiệu chỉnh mới do A1 chọn.

742/758 có std obs 0,017361/0,012552. SeqLSTM_Q mean error 0,096242/0,092153 và RMSE 0,116386/0,111776 mm/ngày; phần mean error bình phương lớn. SeqLSTM_noQ mean error −0,001009/0,010126, RMSE 0,039831/0,039123; phần phương sai sai số chiếm ưu thế. EncDec_noQ có Var(error)/Var(obs) in bằng 1 ở cả hai ID; 9.2 xác nhận hai chuỗi sim hằng và bằng nhau chính xác. Phân rã MSE mô tả đại số chỉ số, không xác nhận nguyên nhân mô hình hoặc quan trắc vật lý.

55 có 0 sim hữu hạn ở hai run Q dù 807 obs; bốn run khác có đủ 1461 sim nhưng chỉ 807 cặp chấm. 800 có 1461 cặp, mean error âm ở cả sáu run, NSE 0,180123–0,624186; train riêng 38 ngày chưa quy được nguyên nhân sai số. Phạm vi xác nhận chỉ bốn hồ sơ, chưa mở rộng sang 451. Chi tiết Mục 23.1 thuyết minh A1.

**Phần 9.2 — độ biến thiên dự báo đã đo:** chạy 0,7 giây; hình/log nguyên bản giữ nguyên byte, ba CSV chi tiết chưa nhận. EncDec_noQ dự báo đúng một giá trị hiển thị 0,173010 mm/ngày cho cả 742/758 trên 1461 ngày; hai chuỗi gốc và sau chặn bằng nhau chính xác từng ngày, std=0, tương quan giữ NaN. Điều này giải thích Var(error)/Var(obs)=1 trong phép phân rã NSE, chưa xác nhận nguyên nhân bên trong mô hình. EncDec_Q vẫn biến thiên (1046/1020 mức, std 0,010477/0,009026); hai ca 55 toàn NaN ở run Q không bị coi là hằng bằng 0.

Tỷ phần MSE từ bình phương mean error ở 742/758 là 68,3797/67,9708% cho SeqLSTM_Q; 0,0642/6,6989% cho SeqLSTM_noQ; 97,7214/98,8298% EncDec_Q; 98,4382/99,3707% EncDec_noQ. Không dùng thành phần sai số như tỷ phần nguyên nhân vật lý hoặc đề xuất hiệu chỉnh bằng test. Tại 55, 113 sim âm SeqLSTM_noQ đều ngoài ngày có obs nên chặn không đổi NSE; D_FC có 47 sim âm trên cặp chấm/96 ngoài obs, NSE tăng 0,000043 theo chính sách khai báo.

Năm run có biến thiên ở 742/758 có r sim_raw 0,974435–0,999120, chưa suy trùng dữ liệu hoặc cơ chế chung. Các ngày obs–CSV vượt 10⁻⁵ tại 800 là 31/08/2017 ở bốn LSTM/EncDec và 09/08/2017 ở D_FC/D_RA. Chênh so với CSV ép float32 trực tiếp vẫn khoảng 8×10⁻⁶; chưa tái tạo đầy đủ chuẩn hóa/khôi phục nhãn hoặc đổi dung sai. **Phần 9 hoàn thành khảo sát bốn hồ sơ**, giữ nguyên nhân mô hình/quan trắc và phạm vi toàn bộ 451 chưa xác nhận; tổng kết Mục 23.3 thuyết minh A1.

**Phần 10.1 — bốn mức từ quan trắc 1981–2013 đã dựng:** chạy 2,2 giây; hình thật giữ nguyên byte, bảng dán trong chat; ba CSV chi tiết/log nguyên bản riêng chưa nhận. 451/451 ID có ba ngưỡng phân biệt q30/q80/q98, không ngưỡng thiếu/trùng. Reference có 4821184 nhãn/5435903 ngày–ID lịch, thiếu 614719, qmax=0 có 2014 ngày. Thấp q<q30, trung bình q30≤q<q80, cao q80≤q<q98, rất cao q≥q98; giữ ties/0, không ép tỷ lệ danh nghĩa.

Tỷ lệ gộp thấp/trung bình/cao/rất cao reference là 29,802223/50,098233/18,079833/2,019711%; test 29,840454/51,963746/16,389500/1,806300%. Test có 11874 ngày rất cao tại 441 ID; 10 ID không có ngày rất cao vẫn giữ trong tập 451. Median tỷ lệ rất cao từng ID test 1,779603%, khác cách gộp ngày. Tổng bốn mức cộng khớp nhãn và nhãn+thiếu khớp lịch từng kỳ; không có ngày hữu hạn thiếu ngưỡng. Reference chứa train/validation, không cộng lặp các phạm vi; chưa gọi rất cao là sự kiện lũ độc lập hoặc ngưỡng chu kỳ lặp lại.

324/600/623 có 1212/1380/1459 ngày reference từ 2010–2013, nên có ngưỡng trước test dù không có nhãn train. 800 có 1499 ngày từ 24/11/2009, q30/q80/q98=1,309533/14,020534/55,540751 mm/ngày. 55 có 6101 nhãn và 5952 ngày thiếu dù cả 33 năm/12 loại tháng có số; chưa là 33 năm đầy đủ. 742/758 đủ 12053 ngày, 758 có 828 ngày 0 và vẫn có ba ngưỡng phân biệt. Chi tiết Mục 24.1 thuyết minh A1. Mã chỉ chỉnh trục x hình ngưỡng từ 0 để bỏ miền âm trống; hình nhận hiện tại vẫn bản gốc trước chỉnh hiển thị.

**Phần 10.2 — độ nhạy hai nửa đã đo:** chạy 2,6 giây; hình thật giữ nguyên byte, bảng dán trong chat, sáu CSV/log nguyên bản riêng chưa nhận. 413 ID có số ở cả 1981–1997 và 1998–2013, 38 chỉ late. Tổng nhãn/thiếu cộng khớp reference 10.1. Median tỷ số late/early q30/q80/q98 là 1,013158/1,020000/1,058653; p05–p95 q98 0,840961–1,462652. Các phân vị ngang ID không phải khoảng tin cậy; chưa quy cho xu thế vật lý.

Trên cùng 4693525 ngày reference của 413 ID, 416579 ngày đổi mức (8,875611% gộp; median từng ID 6,977516%); 29128 ngày đổi trạng thái rất cao. Không là lỗi dự báo hoặc kiểm định ngoài mẫu. 38 ID thiếu early giữ NaN ở phép so, không gán ổn định. Ngưỡng chính và danh sách 451 giữ nguyên.

55 chỉ có 7 loại tháng có số early so với 12 late; q98 tăng 58,658021→84,884211 mm/ngày (tỷ số 1,447103), q30 giảm. Thiếu mùa và biến thiên quan trắc cùng có thể ảnh hưởng, chưa xác nhận cơ chế. 742/758 đủ cả lịch hai nửa nhưng các ngưỡng đổi khác hướng; 758 q30 có tỷ số 2,024390, q80 0,857143, q98 1,115789. 324/600/623/800 chỉ late, ngưỡng late bằng official do cùng tập số liệu chứ chưa chứng minh ổn định. Chi tiết Mục 24.2 thuyết minh A1.

**Phần 10.3 — độ phủ năm thủy văn đã kiểm:** chạy 6,2 giây; hình thật giữ nguyên byte, bảng dán trong chat, bốn CSV/log nguyên bản riêng chưa nhận. 34 năm thủy văn giao reference 1981–2013, 32 năm nằm trọn (nhãn 1982–2013); ngày có nhãn 4821184 và thiếu trong reference 614719 cộng khớp 10.1. Giữ mẫu số 365/366 cả năm, tách ngày ngoài reference.

Theo quy tắc ≥50% lịch năm, 451 ID có 13191 đỉnh năm ứng viên; thêm điều kiện năm nằm trọn reference còn 12881. Chênh 310 là ID có năm biên 1981 đủ điều kiện. Cả hai cách đều có 427 ID ≥10 năm, 24 ID chỉ 3–9 năm, không ID nào không có năm đủ điều kiện; median n tương ứng 33/32. Năm thủy văn 2014 chỉ có 92 ngày cuối 2013, không qua quy tắc và không đọc test để hoàn thành năm đó.

55 có 14 năm qua quy tắc nhưng chỉ một năm đủ mọi quan trắc, 13 năm còn lại thiếu nguyên tháng; chưa gọi là 14 đỉnh thật quan trắc đầy đủ. 324 có 3 năm; 600/623/800 có 4. 742/758 có 33 năm theo mẫu chính, 32 năm trọn; ties vẫn là các năm riêng. Ngưỡng 10 năm là cờ báo cáo, chưa dùng loại ID hoặc xác nhận độ khớp phân phối. Chi tiết Mục 24.3 thuyết minh A1.

**Phần 10.4 — Gumbel L-moments đã fit:** chạy 5,8 giây; hình thật giữ nguyên byte, bảng dán trong chat, bảy CSV/log nguyên bản riêng chưa nhận. Cả 451 ID có fit hữu hạn ở cả mẫu chính nửa lịch năm và mẫu phụ trọn reference; 427 có ≥10 năm, 24 có 3–9 năm (median 6), không có ngưỡng âm trong mẫu chính tại bốn T yêu cầu. Median ngưỡng chính T=1,5/2/5/10 là 14,742515/19,247298/29,935596/36,029289 mm/ngày. Đây là phân vị ngang ID, không là khoảng tin cậy; fit xác định chưa bảo đảm phù hợp phân phối.

310 ID đổi mẫu khi bỏ năm biên; median tỷ số chính/full-window lần lượt 1,004081/1,000653/0,995079/0,993445, nhưng max chênh tuyệt đối ở T=10 là 17,551206 mm/ngày. Giữ mẫu chính theo giao thức, không chọn lại theo chênh. Mẫu chính có 455 năm–ID thiếu nguyên tháng, trong đó 310 là năm biên; 145 còn lại là năm trọn reference nhưng thiếu tháng.

55 có n=14 nhưng chỉ một năm đủ mọi quan trắc. 324 có ba đỉnh, 600/623/800 có bốn; T=5 lớn hơn n ở bốn ID này và T=10 lớn hơn n ở toàn bộ 24 mẫu ngắn. 800 có ngưỡng 1,5 năm cao nhất 124,010896 mm/ngày từ bốn đỉnh, chưa coi là ngưỡng đã kiểm chứng. 15 ID có Q1,5<q98 ngày; median tỷ số Q1,5/q98 là 1,744736. 55/742/758 có tỷ số 0,985369/0,543445/0,446109; hai ngưỡng khác mẫu và ý nghĩa, không áp điều kiện thứ tự hoặc chặn sửa. Chi tiết Mục 24.4 thuyết minh A1.

**Phần 10.5 — độ nhạy bỏ năm đã đo:** chạy 8,7 giây; hình và tệp output thật giữ nguyên byte, bốn CSV chi tiết chưa nhận. Mỗi T có 13191 fit bỏ năm xác định (52764 dòng ngưỡng ở bốn T), toàn bộ 451 ID có đủ n lần; không có ngưỡng âm hoặc baseline không dương ở phép kiểm này. Median mức thay đổi tuyệt đối lớn nhất so ngưỡng gốc ở T=1,5/2/5/10 lần lượt 3,059859/4,230001/7,163202/8,283694% cho 427 ID n≥10 và 12,940379/12,329141/15,003291/17,440408% cho 24 ID n<10. p95 tại T=10 là 22,735924/35,394873%; các phân vị ngang ID không là khoảng tin cậy.

Mẫu dài vẫn có thể nhạy: 338 có n=33, mức đổi lớn nhất T=10 là 36,916733%; 197 có n=29 là 34,735286%. 391 (n=6) cao nhất 48,527003%. 600 có miền ngưỡng 10 năm 13,283658–23,065162 mm/ngày so gốc 20,223715; mức đổi lớn nhất 34,316430%. 55 đổi nhiều nhất khi bỏ 2012, năm đó quan trắc chưa hoàn chỉnh. 324/800 có năm ảnh hưởng nhất không phải max mẫu; giữ mọi lần bỏ, không chỉ bỏ đỉnh lớn nhất. 742/758 cùng n=33 nhưng mức nhạy 13,919609/5,814266%. Không dùng độ nhạy loại ID/bỏ năm/đổi phân phối; chi tiết Mục 24.5 thuyết minh A1.

**Phần 10.6 — độ khớp mô tả đã đo:** chạy 3,0 giây; hình và tệp output thật giữ nguyên byte, bốn CSV chi tiết chưa nhận. 427 ID n≥10 dùng 13033 đỉnh, 24 ID n<10 dùng 158, tổng 13191 khớp 10.3–10.5. Median D là 0,132249/0,212231, p95 0,212309/0,336517; đây là chênh CDF trên chính mẫu đã fit, không có p-value hoặc tiêu chí chấp nhận. Median RMSE phân vị/scale là 0,343798/0,375853; không phải lỗi dự báo theo ngày.

Gumbel của 742/758 đặt khoảng 4,656443/8,673815% xác suất dưới 0; max F(0) nhóm n≥10 là 18,8673% (ID 322). Đây là miền hỗ trợ của phân phối đã fit, không phải Q âm quan trắc; bốn ngưỡng dương ở 10.4 không làm F(0) bằng 0. Giữ giới hạn này, không tự chặn/đổi phân phối hoặc ngưỡng. 161 (n=33) có D cao nhất nhóm ≥10 là 0,363198; 800 (n=4) cao nhất nhóm ngắn 0,446926. 338 có chênh điểm thứ tự lớn nhất/scale 8,161611, cùng hồ sơ có độ nhạy bỏ năm 36,916733% ở 10.5; cùng dữ liệu nên không là kiểm chứng độc lập. Chi tiết Mục 24.6 thuyết minh A1.

**Phần 10.7 — ngày vượt ngưỡng cố định đã đếm:** chạy 7,6 giây; hình và tệp output thật giữ nguyên byte, năm CSV chi tiết chưa nhận. Phủ nhãn khớp 8.1/reference 10.1, mọi ngày có nhãn đều có ngưỡng, không có ties trong các số đã in. Test tại T=1,5/2/5/10 có 3509/1773/498/220 ngày–ID vượt, tương ứng 410/368/214/131 ID có vượt và 41/83/237/320 ID có quan trắc nhưng không vượt. Tỷ lệ gộp test 0,533797/0,269713/0,075757/0,033467%; median từng ID 0,342231/0,136893/0/0%. Chưa là số trận độc lập hoặc số ID đủ điều kiện chấm dự báo.

Ngưỡng 10 năm: train có 438 ngày vượt, validation 394, test 220; các kỳ khác độ dài và phủ nhãn nên không suy trực tiếp thay đổi tần suất trận từ số đếm. Test phân theo năm 2014/2015/2016/2017 có 115/24/56/25 ngày vượt T=10. Reference có 2352 ngày vượt T=10 nhưng chồng train/validation; các T lồng nhau, không cộng như các sự kiện khác nhau. 1/T không là tỷ lệ ngày vượt cần đạt.

55 test chỉ có 807/1461 ngày có số, 654 thiếu; 28 ngày vượt T=1,5 tương ứng 3,469641% trên phần quan trắc, không gọi là tỷ lệ toàn lịch. 742/758 đủ 1461 ngày nhưng không vượt bốn T, khác ba ID 324/600/623 không có nhãn train (tỷ lệ NaN). 24 ID có mẫu fit <10 năm lại đủ mọi 35064 ngày test, đóng góp 17 ngày vượt T=10 ở 8 ID; nhóm ≥10 có 203 ngày ở 123 ID. Độ phủ test không khắc phục mẫu fit ngắn. Chi tiết Mục 24.7 thuyết minh A1.

**Phần 10.8 — đoạn vượt liên tiếp đã tách:** chạy 11,4 giây; hình và tệp output thật giữ nguyên byte, bốn CSV chi tiết chưa nhận. Tổng độ dài khớp 10.7. Test có 2093/1157/366/184 đoạn ở T=1,5/2/5/10, tương ứng 3509/1773/498/220 ngày vượt. Median độ dài đều một ngày, p95 3/3/2/2 ngày, max 120/61/22/9 ngày. Chưa coi các đoạn là trận độc lập.

Ở test, T=1,5 có 2091 đoạn biết cả hai phía và hai đoạn một ngày có đầu quan trắc nhưng cuối chạm biên; ba T sau đều biết cả hai phía. Không đoạn test nào kề ngay ngày thiếu, nhưng khoảng dài giữa các đoạn vẫn có thể thiếu, và 55 vẫn thiếu 654 ngày test. Reference T=1,5/T=2 có một đoạn kề ngày thiếu mỗi mức; train có 8/7/1/0 đoạn chạm kỳ và validation có một đoạn chạm kỳ T=1,5. Không giả ngày ngoài kỳ/ngày thiếu là ngày không vượt.

55 có 28 ngày T=1,5 thành 14 đoạn (max tám ngày); 600 có bốn ngày T=10 thành bốn đoạn một ngày, vẫn cần kiểm tiêu chí gộp. 697 có đoạn 120 ngày 11/06–08/10/2016 tại T=1,5, cùng peak 3,253889 mm/ngày ở 05–06/08 với các đoạn T=2 dài 61 ngày/T=5 dài 22 ngày. 563 có đoạn 63 ngày tại T=1,5 và 46 ngày tại T=2 cùng peak 24/02/2014. Không cộng các T như trận riêng hoặc suy một trận vật lý dài 120 ngày. Chi tiết Mục 24.8 thuyết minh A1.

**Phần 10.9 đã có output thật:** quy tắc USWRC bảo toàn ngày vượt; test giảm 2093→1869, 1157→1073, 366→345 và 184→178 cụm ở T=1,5/2/5/10, tương ứng 10,702341%/7,260156%/5,737705%/3,260870%. Giữ 4/2/0/0 ranh giới thiếu chưa xác định; median mức giảm từng ID có đoạn đều 0%. Diện tích area_gov của tập 451 có min/median/max 3,9/101/15131 km², khoảng cách sau ceiling 6/9/14 ngày. Hình/output giữ nguyên byte; bảy CSV chi tiết chưa nhận cục bộ; thời gian 83,1 giây. Cụm theo quy tắc chưa chứng minh độc lập vật lý/thống kê, chưa ghép/chấm dự báo. Chi tiết Mục 24.9 và tổng kết 24.10 thuyết minh A1.

**A1 đã chốt khảo sát đến 10.9; Phần 11 chỉ nằm trong thuyết minh MD (Mục 25):** bàn giao tập 451, đơn vị/mặt nạ, thiếu nhãn train ở 324/600/623, train ngắn của 800, hồ sơ 55/742/758, giới hạn nguồn khí tượng/cache/scaler và 24 mẫu Gumbel ngắn. A3 kiểm nguồn dự báo và đánh giá theo giao thức cố định; B2 kiểm bộ lọc đúng ô dùng theo pha. Chưa tự sửa dữ liệu, lọc ID hoặc chọn mô hình bằng test. Ca vượt T=10 riêng tháng 6/2013 chưa có bảng kết quả chuyên biệt. Notebook giữ phép tính và kiểm tra, tổng hợp/diễn giải tập trung tại [Phần 11 của báo cáo A1](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore.md#sec-25).

### 6.3. Tái lập bằng trọng số tác giả (bước A2, Kaggle 11/10/2026)

Notebook `Workspace/03_Baseline/01_A2_LamaHCEExt_Reproduce.py` (Kaggle `01_A2_LamaHCEExt_Reproduce`, GPU T4, Internet bật, Persistence Files only, dataset `lamah-ce-ext`). Kết quả từng phần ghi dần tại đây.

**Môi trường (Phần 1).** Thư viện cài từ fork `Ticasslo/neuralhydrology` commit `dda45fc` (pip ghi đúng mã commit trong `direct_url.json`). Kaggle: Python 3.13.15, NumPy 2.1.3, pandas 2.3.3, xarray 2025.12.0, torch 2.11.0+cu128, SciPy 1.16.3, numba 0.61.2, ruamel.yaml 0.19.1, matplotlib 3.10.0; GPU Tesla T4. Cài thư viện không làm đổi phiên bản gói có sẵn.

**Kết quả của tác giả (Phần 2).** Thư mục `Experiments/` có 24 run (thư mục có `config.yml`) và `basins_filtered.txt` (451 lưu vực). Mọi run dùng `seed` 111, `seq_length` 365, nhãn `qmax`. Hai biến thể lớp nhúng: `Simple_Embedding` (một lớp tuyến tính) và `Complex_Embedding` (30-20-64). Pha quá khứ của mô hình hai pha có 31 biến tái phân tích (RA), 32 khi thêm `qmean`, hoặc 36 khi thêm 5 biến dự báo lưu trữ (FCRA); ngày t luôn là 5 biến ECMWF HRES. Bốn run học chuyển giao thiếu tệp (ba run `PreRun` chỉ dựng cấu trúc nên không có `test_metrics.csv`; `TL_AllWeights` không có tệp chuẩn hóa); các run khác đủ `best_model.pt`, `train_data/train_data_scaler.yml`, `test/best_model/test_metrics.csv`.

| Mô hình dùng ở A2 | Thư mục run (dưới `Experiments/`) |
|---|---|
| Sequential Forecast LSTM có lưu lượng quá khứ | `Sequential_Forecast_LSTM/Sequential_Forecast_LSTM_with_q/Sequential_Forecast_LSTM_RA_with_q_Simple_Embedding` |
| Sequential Forecast LSTM không lưu lượng | `Sequential_Forecast_LSTM/Sequential_Forecast_LSTM_without_q/Sequential_Forecast_LSTM_RA_without_q_Simple_Embedding` |
| Encoder–Decoder LSTM có lưu lượng quá khứ | `Encoder_Decoder_LSTM/Encoder_Decoder_LSTM_with_q/Encoder_Decoder_LSTM_with_q_Complex_Embedding` |
| Encoder–Decoder LSTM không lưu lượng | `Encoder_Decoder_LSTM/Encoder_Decoder_LSTM_without_q/Encoder_Decoder_LSTM_RA_without_q_Complex_Embedding` |
| 𝒟FC (mốc dưới) | `Baseline/Forecast` |
| 𝒟RA (mốc trên) | `Baseline/Reanalysis/Reanalysis_Simple_Embedding` |

**Lý do chọn 6 trên 24 run.** Bước A2 nhằm kiểm môi trường của đề tài (thư viện cài từ fork, thư mục Q tự dựng, cách đánh giá) cho ra đúng số của tác giả, và chuẩn bị các mốc so sánh mà các bước sau dùng. Sáu run được chọn theo bốn tiêu chí:

1. **Lấy từ các run tác giả dùng cho kết luận chính về lưu lượng quá khứ:** mục 3.4 của bài (*The role of integrating past discharge in the training domain*) có Hình 7 gồm sáu đường CDF, khai báo trong tệp cấu hình vẽ hình của mã tác giả (`Inspect_Experiments/config/Role_of_integrating_discharge.yml`): 𝒟FC, 𝒟RA, Sequential Forecast LSTM không Q và có Q (Simple), Encoder–Decoder LSTM không Q và có Q (Complex); sáu run trên đúng là sáu đường đó; các tệp cấu hình hình vẽ khác cũng dùng `Reanalysis_Simple_Embedding` làm 𝒟RA. NSE trung vị của hai bản 𝒟RA trong kết quả tác giả đều làm tròn thành 0,69 (Simple 0,6923, Complex 0,6884), nên bản dùng được xác định qua tệp cấu hình hình vẽ, không qua số trong bài.
2. **Bao trùm mọi thành phần cần kiểm:** đủ ba kiến trúc (LSTM thường, Sequential Forecast LSTM, Encoder–Decoder LSTM), có và không có `qmean` (kiểm thư mục Q tự dựng), hai loại lớp nhúng (Simple, Complex), hai nguồn dữ liệu (dự báo ECMWF HRES, tái phân tích). Các run còn lại chỉ lặp lại một tổ hợp đã có: bản lớp nhúng khác của cùng mô hình, biến thể có dự báo lưu trữ trong pha quá khứ (FCRA, cùng thư mục dữ liệu gốc), CrossDomain (LSTM thường 5 biến).
3. **Là các mốc các bước sau cần:** Sequential Forecast LSTM có Q là mô hình tốt nhất của bài (0,71) và là điểm xuất phát của bước B, C; 𝒟FC và 𝒟RA là mốc dưới, mốc trên; Sequential Forecast LSTM không Q và hai run Encoder–Decoder LSTM (có, không Q) là đối chứng cho vai trò của Q và của kiến trúc. Đề tài không phát triển tiếp hướng học chuyển giao và FCRA.
4. **Chạy lại trọn được:** bốn run học chuyển giao thiếu tệp (ba run `PreRun` không có `test_metrics.csv`, `TL_AllWeights` không có tệp chuẩn hóa).

Mỗi run đánh giá mất 2–6 phút GPU (Phần 6: 136–311 giây); tái lập thêm khoảng 20 run còn chạy được tốn khoảng 1,5 giờ GPU, không cung cấp kiểm tra mới và không phục vụ các bước sau, nên chỉ làm khi cần chứng minh toàn bộ Bảng F1 tái lập được.

**Đối chiếu và chuẩn bị run (Phần 5).** NSE trung vị test trong kết quả có sẵn của tác giả khớp số trong bài ở cả sáu run (lệch ≤ 0,005 do bài làm tròn hai chữ số):

| Run | NSE trung vị (tệp tác giả) | Trong bài | Số lưu vực có NSE | Số tham số |
|---|---|---|---|---|
| Sequential Forecast LSTM có Q | 0,7053 | 0,71 (mục 3.4) | 450 | 84.785 |
| Sequential Forecast LSTM không Q | 0,6276 | 0,63 (mục 3.2) | 451 | 84.769 |
| Encoder–Decoder LSTM có Q | 0,6622 | 0,66 (mục 3.4) | 450 | 341.436 |
| Encoder–Decoder LSTM không Q | 0,569 | 0,57 (mục 3.2, 3.4) | 451 | |
| 𝒟FC | 0,3867 | 0,39 (Bảng F1) | 451 | 83.713 |
| 𝒟RA | 0,6923 | 0,69 (mục 3.4, Bảng F1) | 451 | 84.129 |

Số trong bài lấy từ Bảng F1 (Phụ lục F, trang 5093 của tạp chí, trang 27 của tệp PDF) và phần chữ mục 3.2 (trang 5076) và 3.4 (trang 5078). Trong bản PDF, tên dòng của Bảng F1 in lệch so với hàng số, nên dòng tương ứng của mỗi run được xác định qua cột Min: giá trị Min trùng NSE của lưu vực 758 trong kết quả tác giả (Sequential Forecast LSTM có Q −78,3; không Q −8,71; Encoder–Decoder LSTM có Q −131,42; 𝒟FC −33,29; 𝒟RA −85,12). Encoder–Decoder LSTM không Q khớp 0,57 ở phần chữ mục 3.2 và 3.4 (NSE trung vị lấy từ hình CDF của Phần 7, làm tròn ba chữ số). Hai run có Q chỉ có NSE ở 450 lưu vực. Sáu run được chép sang thư mục làm việc (bỏ kết quả test cũ, trọng số từng epoch và trạng thái optimizer; 0,3–1,4 MB mỗi run), đổi sáu trường đường dẫn, `device` sang GPU và `num_workers` từ 20 xuống 4 (số CPU của Kaggle); không run nào dùng mã hóa lưu vực hay tệp đặc trưng thêm. Cả sáu run dựng lại được mô hình bằng thư viện và nạp `best_model.pt` không lỗi.

**Đánh giá lại trên kỳ test (Phần 6).** Chạy `eval_run` với `best_model.pt` của tác giả (không huấn luyện) trên T4, 451 lưu vực, kỳ 2014–2017:

| Run | Thời gian | Lưu vực có NSE | NSE trung vị chạy lại | NSE trung vị tác giả | Chênh |
|---|---|---|---|---|---|
| Sequential Forecast LSTM có Q | 288 giây | 450 | 0,7053 | 0,7053 | 0,0000 |
| Sequential Forecast LSTM không Q | 283 giây | 451 | 0,6276 | 0,6276 | 0,0000 |
| Encoder–Decoder LSTM có Q | 287 giây | 450 | 0,6622 | 0,6622 | 0,0000 |
| Encoder–Decoder LSTM không Q | 311 giây | 451 | 0,569 | 0,569 | dưới 0,001 |
| 𝒟FC | 136 giây | 451 | 0,3864 | 0,3867 | −0,0002 |
| 𝒟RA | 233 giây | 451 | 0,6923 | 0,6923 | +0,0001 |

Ở hai run có Q, lưu vực 55 không có giá trị dự báo nào trong kỳ test (thư viện báo mọi giá trị dự báo là NaN), nên chỉ còn 450 lưu vực; kết quả của tác giả cũng có 450 lưu vực. Lưu vực 55 chỉ có `qmean` ở 1025 trên 1825 ngày của kỳ test cộng 364 ngày khởi động, nên cửa sổ 365 ngày của mọi mẫu đều chứa ô `qmean` trống; ở kỳ test mẫu không bị lọc mà vẫn chạy, nhưng đầu vào có ô trống nên dự báo ra NaN.

**So từng lưu vực và từng ngày (Phần 7).** So với `test_metrics.csv` và `test_results.p` của tác giả (có trên Zenodo):

- Quan trắc `qmax_obs` trùng hoàn toàn ở cả sáu run (lệch lớn nhất 0, vị trí ô trống giống nhau).
- Dự báo `qmax_sim` lệch rất nhỏ: trung vị 0,0001–0,0003 mm/ngày, mức 99,9% là 0,011–0,028 mm/ngày, lớn nhất 0,09–0,22 mm/ngày trên khoảng 650 nghìn ngày-lưu vực mỗi run; mức này phù hợp với sai khác số học giữa hai GPU và hai môi trường thư viện, không phải sai khác dữ liệu.
- Lệch NSE trung bình theo lưu vực 0,00004–0,0003. Chỉ hai lưu vực có lệch NSE trên 0,01 là 742 và 758 (tới 0,085 ở lưu vực 758, run Sequential Forecast LSTM có Q); cả hai có phương sai `qmax` quan trắc kỳ test chỉ 0,0002–0,0003 (mm/ngày)² và NSE từ −4 tới −131 ở mọi run. Vì NSE = 1 − SSE / (n × phương sai quan trắc), phương sai gần 0 khuếch đại lệch dự báo khoảng 0,001 mm/ngày thành lệch NSE vài phần trăm. Hai lưu vực này cần xem lại ở khảo sát dữ liệu (A1).
- Biểu đồ phân tán NSE từng lưu vực nằm trên đường y = x và đường CDF chạy lại trùng đường CDF của tác giả ở cả sáu run (hình lưu ở `Workspace/03_Baseline/Figures/A2_Scatter.png`, `A2_CDF.png`, `A2_Basin758.png`). Riêng run Encoder–Decoder LSTM không Q, hàm tính KGE báo hệ số tương quan không xác định (một chuỗi không đổi) ở ít nhất một lưu vực; lưu vực đó xác định ở bước A3.

Kết luận tái lập: đạt. Tiêu chí xét trên dự báo thay vì trên NSE từng lưu vực (vì NSE khuếch đại lệch ở lưu vực có phương sai quan trắc gần 0), gồm bốn điều kiện, đều thỏa ở cả sáu run: quan trắc trùng tác giả; lệch NSE trung vị ≤ 0,005; lệch dự báo ở mức 99,9% ≤ 0,05 mm/ngày; mọi lưu vực lệch NSE trên 0,01 (742, 758) đều có NSE dưới −1.

Bản lưu chính thức là Version 2 của notebook trên Kaggle (Save & Run All, GPU T4, 2106 giây, output 1,55 GB, kết luận `REPRO_OK = True`, lưu vực lệch NSE trên 0,01 chỉ có 742 và 758). Ở chế độ chạy toàn bộ, Kaggle chạy trong phiên mới, không thấy tệp do Persistence giữ ở phiên tương tác: thư mục Q được dựng lại (khoảng 270 giây) và sáu run được đánh giá lại (2,5–5,4 phút mỗi run, theo mốc thời gian trong nhật ký).

Config ghi cứng đường dẫn máy tác giả ở 6 trường: `data_dir` (`/home/ok2907/Documents/Data/LamaH_expanded`; run có Q dùng `LamaH_expanded_q_input`), `run_dir`, `train_dir`, `train_basin_file`, `validation_basin_file`, `test_basin_file`. Tệp khí tượng mức A có 40 biến (ERA5-Land 21, ECMWF HRES 8, E-OBS 8, MSWEP 1, GLEAM 2) và `DOY`, từ 1981 tới 2019; tệp trạm có `qmin`, `qmean`, `qmax` (m³/s), từ 1981 tới 2017.

**Đơn vị `qmean` tác giả đưa vào mô hình (Phần 3).** Thư viện chuẩn hóa mỗi biến bằng trung bình và độ lệch chuẩn tính trên kỳ train, lưu ở `train_data/train_data_scaler.yml` (`xarray_feature_center`, `xarray_feature_scale`). Cách tính trong mã (`basedataset.py` dòng 437–449, 762–765): gộp mọi lưu vực và mọi ngày, bỏ ô trống, độ lệch chuẩn dạng tổng thể; biến đầu vào tính từ ngày bắt đầu train trừ 364 ngày khởi động (02/01/2002) tới 31/12/2009, nhãn bị xóa trong các ngày khởi động nên chỉ tính từ 01/01/2003. Tính lại từ tệp trạm của 451 lưu vực theo hai đơn vị và so với tệp chuẩn hóa của run Sequential Forecast LSTM có lưu lượng quá khứ:

| Biến | Nguồn | Trung bình | Độ lệch chuẩn | Lệch so với tác giả |
|---|---|---|---|---|
| `qmean` | tệp chuẩn hóa của tác giả | 4,5778 | 16,1461 | |
| `qmean` | tính lại, m³/s | 4,5778 | 16,1461 | 0,00% / 0,00% |
| `qmean` | tính lại, mm/ngày | 2,0872 | 3,5224 | 54,41% / 78,18% |
| `qmax` (đối chứng) | tệp chuẩn hóa của tác giả | 2,5721 | 5,2624 | |
| `qmax` (đối chứng) | tính lại, m³/s | 5,2147 | 17,5222 | 102,74% / 232,97% |
| `qmax` (đối chứng) | tính lại, mm/ngày | 2,5721 | 5,2624 | 0,00% / 0,00% |

Dòng đối chứng `qmax` khớp ở mm/ngày theo MATCH_TOL=0.01 (1%) của cell A2, phù hợp mã đổi nhãn sang mm/ngày. Bảng hiển thị bốn chữ số thập phân; riêng bảng A2 không xác nhận bằng nhau ở dung sai 10⁻⁵ hoặc giống hệt phép giảm số học của thư viện. A1 5.3 sau đó đã xác nhận mean/std qmax mm/ngày theo area_gov ở dung sai 10⁻⁵ cho 22 run tái dựng được phạm vi, đồng thời phân biệt sai khác xarray float32 tăng tốc trong môi trường kiểm hiện tại. Kết luận trong phạm vi phép kiểm đơn vị A2:

- Tác giả đưa `qmean` vào mô hình theo **m³/s**, trong khi nhãn `qmax` theo mm/ngày.
- Giá trị `qmean` lấy nguyên từ tệp trạm `D_gauges`, ngày thiếu để trống, không điền: thống kê khớp tới chữ số thứ tư. Hàm đọc tệp trạm của thư viện đổi số âm (mã ghi chú -999 là giá trị thiếu) thành ô trống (`lamah.py` dòng 228–229), còn hàm đọc tệp khí tượng không đổi, nên khi chép `qmean` sang tệp khí tượng cũng đổi số âm thành ô trống; ở 3 lưu vực kiểm ngẫu nhiên (788, 330, 698) tệp trạm không có -999, ngày thiếu là ô trống.
- Thư mục dựng lại (451 tệp, 1,35 GB): `qmean` đọc bằng hàm đọc tệp khí tượng của thư viện có trung bình 4,5778, độ lệch chuẩn 16,1461 trên 02/01/2002–31/12/2009, khớp tệp chuẩn hóa của tác giả; ô trống 3,89% số ngày-lưu vực trong khoảng này; lưu vực 324, 600, 623 không có `qmean` cả khoảng. Trong các lưu vực có dữ liệu, lưu vực 800 thiếu nhiều nhất: 2883 ngày, chỉ có `qmean` từ 24/11/2009 (khớp ngày bắt đầu đo ở Mục 6.1), nên ở run có Q lưu vực này không có mẫu train nào (bộ lọc của thư viện bỏ mẫu có ô trống trong cửa sổ 365 ngày, `basedataset.py` dòng 861–919, mà kỳ train hết 31/12/2009) (hình `Workspace/03_Baseline/Figures/A2_QmeanBasin800.png`).
- Thư mục dữ liệu có Q của tác giả tên `LamaH_expanded_q_input`.
- Bước A2 và A4 dựng thư mục Q theo đúng cách này; bước B1 thử đổi `qmean` sang mm/ngày cho cùng đơn vị với nhãn (`03_Pipeline.md` Mục 3.3, 7).
