# So sánh chi tiết hai ứng viên bài cơ sở: #104 Kirschstein & Sun (ICML 2024) và #121 BiasCast (HESS 2026)

> Tài liệu phục vụ quyết định chọn bài cơ sở. Số `#N` trỏ về `RESEARCHING.md`. Đánh giá theo tiêu chí ở `CHECKPDF.md` Mục 4.1, 5.12, 8.4c, 8.6; đọc mã ở `CHECKCODE.md` Mục 1 và 15.

## Tóm tắt

Hai bài cùng dùng LamaH-CE (Trung Âu) và cùng là bài toán **dự báo** (có Q quan trắc quá khứ, dự báo Q sau thời điểm phát hành), nhưng khác hẳn nhau về quy mô và cách đặt bài toán:

- **#104** dự báo **Q theo giờ, trước 6 giờ**, cho **358 trạm cùng lúc** trên một đồ thị mạng sông, chỉ từ **24 giờ quá khứ × 5 biến**, không có thuộc tính tĩnh, không có dự báo thời tiết. Mô hình là một lớp affine, 19 lớp GNN và một lớp affine giải mã (khoảng 0,33–0,65 triệu tham số).
- **#121** dự báo **lưu lượng cực đại ngày (qmax) của ngày t**, cho **451 lưu vực độc lập**, từ **364 ngày quá khứ × 31–32 biến** (tái phân tích nhiều nguồn và Q), **dự báo thời tiết ECMWF thật của ngày t (5 biến)** và **33 thuộc tính tĩnh**. Mô hình là các biến thể LSTM dự báo kiểu Google, khoảng 0,08–0,14 triệu tham số.

Về độ tin cậy số liệu công bố, #121 hơn rõ: số tự tính lại khớp bài. #104 có công thức NSE sai, và nhiễu giữa các lần chạy lớn cỡ toàn bộ chênh lệch trong bảng kết quả. Ngược lại, #104 hơn về nơi công bố (ICML, ngành AI) và số trích dẫn (21 so với 3).

## 1. Phạm vi đã đọc

| | #104 Kirschstein & Sun | #121 BiasCast |
|---|---|---|
| Toàn văn | Toàn bộ 13 trang + Phụ lục A.1–A.8 | Toàn bộ Mục 1–4, Phụ lục A–G |
| Mã | Toàn bộ repo (~880 dòng): `dataset.py`, `models.py`, `functions.py`, 8 script train/test, 5 notebook | Repo phân tích (~1.550 dòng); fork NeuralHydrology: 1 commit sửa 7 tệp, cùng mô hình `sequential_forecast_lstm.py`, `inputlayer.py`, `fc.py` và bộ cắt cửa sổ `basedataset.py` |
| Kết quả của tác giả | 957 checkpoint: đọc `hparams`, lịch sử loss, kích thước trọng số (đếm tham số trực tiếp) | 24 cấu hình, scaler, `test_metrics.csv` theo lưu vực (tự tính lại trung vị) |
| Chạy thử | Chưa (dữ liệu giờ đang ở Kaggle) | Chưa |

## 2. Thông tin xuất bản

| Khía cạnh | #104 Kirschstein & Sun | #121 BiasCast |
|---|---|---|
| Tên bài | The Merit of River Network Topology for Neural Flood Forecasting | BiasCast: learning and adjusting real time biases from meteorological forecasts to enhance runoff predictions |
| Tác giả | Nikolas Kirschstein, Yixuan Sun (Oxford, TU München) | Konold, Feigl, Podest, Klingler, Schulz (BOKU Wien) |
| Nơi công bố | ICML 2024, PMLR 235, tr. 24713–24725 | Hydrology and Earth System Sciences 30:5067–5096, 2026 (bản thảo HESS Discussions 2025) |
| Ngành | Học máy (CNTT/AI) | Thủy văn (Khoa học Trái đất) |
| Xếp hạng | CORE A* | Scimago 2025 Q1, SJR 2,035, H-index 194 |
| Trích dẫn (Google Scholar) | 21 (25/9/2026) | 3 (26/9/2026, bản Discussions); một trong số đó là AIFL của ECMWF (J. Hydrol. 2026) |
| Phản biện | OpenReview, ICML Poster | 2 phản biện (Gwyneth Matthews và 1 người ẩn danh); biên tập Micha Werner |
| Mã, giấy phép | GitHub, không có tệp LICENSE | Mã phân tích CC BY 4.0 (Zenodo 17293199); fork NeuralHydrology (BSD-3 theo upstream) |
| Dữ liệu, giấy phép | LamaH-CE gốc, CC BY-SA 4.0 | Extended LamaH-CE, CC BY-NC 4.0 (phi thương mại); bản 1.0 |

## 3. Bài toán — minh họa bằng một mẫu cụ thể

**#104.** Một mẫu là **một "ảnh chụp" của cả mạng 358 trạm**:
- Đầu vào: 24 giờ liên tiếp, ví dụ 00:00–23:00 ngày 14/05/2016 (theo mốc giờ của tệp LamaH-CE). Mỗi trạm có 5 biến theo giờ: Q quan trắc, mưa, độ ẩm đất lớp 0–100 cm, nhiệt độ 2 m, áp suất bề mặt.
- Đầu ra: Q của **cả 358 trạm lúc 05:00 ngày 15/05/2016**, tức 6 giờ sau giờ đầu vào cuối cùng (mã: nhãn ở chỉ số `offset + W + L − 1`).
- Mô hình không nhận thông tin gì về khoảng từ 23:00 tới 05:00, kể cả dự báo mưa.

**#121.** Một mẫu là **một lưu vực tại một ngày**:
- Hindcast: 364 ngày từ 17/05/2015 tới 14/05/2016. Mỗi ngày có 31 biến tái phân tích / quan trắc lưới gộp theo lưu vực, cộng `qmean` (lưu lượng trung bình ngày quan trắc) ở biến thể có Q.
- Forecast: ngày 15/05/2016, 5 biến dự báo ECMWF HRES. Theo bài, đây là trung bình 8 giá trị 3 giờ tính từ lần phát hành 00 UTC.
- Tĩnh: 33 thuộc tính của lưu vực.
- Đầu ra: **lưu lượng cực đại trong ngày 15/05/2016 (qmax)**. Tác giả gọi đây là "lead 1 ngày"; thực chất là đỉnh Q xảy ra trong 0–24 giờ sau lần phát hành dự báo.
- Căn chỉnh giữa ngày UTC của HRES và ngày của tệp LamaH-CE: chưa xác nhận (mã tạo dữ liệu không công khai).

| Khía cạnh | #104 | #121 |
|---|---|---|
| Loại bài toán | Dự báo (có Q quá khứ) | Dự báo (có hoặc không có Q quá khứ) |
| Tầm dự báo | Đúng 6 giờ (một thời điểm) | Đỉnh trong ngày t, 0–24 giờ |
| Dùng dự báo thời tiết tương lai | Không | Có (ECMWF HRES thật, lưu trữ) |
| Không gian | 358 trạm dự báo đồng thời, liên kết bằng đồ thị | 451 lưu vực độc lập, một mô hình chung |
| Q đích | Q theo giờ (m³/s) | qmax ngày (tệp `D_gauges` ghi m³/s, trích từ dữ liệu giờ), bộ nạp đổi sang **mm/ngày** chia theo diện tích `area_gov` trước khi huấn luyện |

## 4. Dữ liệu

| Khía cạnh | #104 | #121 |
|---|---|---|
| Bộ dữ liệu | LamaH-CE gốc (Klingler và cs. 2021), bản giờ | Extended LamaH-CE (Konold và cs. 2025), bản ngày: LamaH-CE + E-OBS, MSWEP, GLEAM, ECMWF HRES + qmin/qmax trích từ dữ liệu giờ |
| Độ phân giải | Giờ | Ngày |
| Đơn vị không gian của khí tượng | **Mức B — vùng trung gian** giữa hai trạm (`B_basins_intermediate_all`) | **Mức A — toàn bộ lưu vực thượng nguồn** của trạm (tương đương CAMELS) |
| Chọn trạm / lưu vực | Thành phần liên thông "Danube A" (608 trạm, gốc là trạm 399). Giữ trạm có Q > 0 và đủ 157.800 giờ 2000–2017 → **358 trạm**. Nối tắt cạnh khi xóa trạm; đồ thị còn 357 cạnh (dạng cây) | Lưu vực ít hoặc không chịu tác động con người → **451 lưu vực** (72,5% đầu nguồn, 27,5% lồng nhau). Diện tích trung bình 252 km², độ lệch chuẩn 915 km² (scaler của tác giả) |
| Kỳ dữ liệu dùng | 2000–2017 (18 năm) | 2003–2017 (15 năm), cộng 364 ngày khởi động trước 2003 |
| Chia tập | Test 2016–2017. Train 8 năm theo 3 cách (năm chẵn 2000–2014, năm lẻ 2001–2015, liên tục 2008–2015). Validation = 1/5 cửa sổ train chọn ngẫu nhiên (chồng lấn với train) | Train 2003–2009, validation 2010–2013, test 2014–2017 (theo thời gian, không chồng lấn) |
| Dung lượng tải | Tệp nén 14,8 GB (giải nén ~70 GB); phần mã đọc ≈ 1,1 GB float32 (358 × 157.800 × 5). Đã tải lên Kaggle thành 2 gói ZIP: lõi 6,82 GB + phụ 6,32 GB | Tệp nén 0,95 GB; kết quả thí nghiệm 0,23 GB |
| Đơn vị lưu lượng | Q vào và ra đều m³/s (trước chuẩn hóa) | Nhãn qmax đổi sang mm/ngày theo diện tích; Q đầu vào `qmean` đọc từ thư mục khí tượng riêng của tác giả (`LamaH_expanded_q_input`, không công khai), bộ nạp không đổi đơn vị. Scaler của tác giả: `qmean` trung bình 4,58, độ lệch chuẩn 16,15; `qmax` trung bình 2,57, độ lệch chuẩn 5,26 — `qmean` lớn hơn `qmax` cho thấy `qmean` đầu vào vẫn ở m³/s (suy luận, chưa kiểm được vì thư mục không công khai) |
| Chuẩn hóa | z-score **theo từng trạm**, thống kê 2000–2015 cho mọi cách chia (gồm cả những năm không thuộc tập train của cách chia đó) | z-score **toàn cục** (chung mọi lưu vực) theo kỳ train; thuộc tính tĩnh chuẩn hóa theo trung bình / độ lệch chuẩn qua 451 lưu vực |
| Mạng sông | Có: `Stream_dist.csv` (khoảng cách, chênh cao, độ dốc) | Không dùng; có sẵn trong LamaH-CE gốc nếu muốn thêm |

## 5. Đầu vào

| Khía cạnh | #104 | #121 (mô hình tốt nhất: Sequential Forecast LSTM có Q) |
|---|---|---|
| Biến động | 5: `qobs`, `prec`, `volsw_123`, `2m_temp`, `surf_press` | Hindcast 32: ERA5-Land 21 (nhiệt độ max/mean/min, điểm sương max/mean/min, gió u/v, albedo, LAI 2 loại, tuyết SWE, bức xạ sóng ngắn max/mean, bức xạ nhiệt max/mean, áp suất, bốc thoát hơi, mưa, độ ẩm đất 2 lớp); E-OBS 7 (nhiệt độ mean/min/max, mưa, áp suất mực biển, gió, bức xạ); MSWEP 1 (mưa); GLEAM 2 (bốc hơi thực, tiềm năng); `qmean`. Forecast 5 ECMWF: `t2m`, `d2m`, `ssrd`, `tp`, `e` |
| Thuộc tính tĩnh | Không | 33: địa hình (diện tích, cao độ ×4, độ dốc, hình dạng ×3, mật độ sông), khí hậu (mưa, bốc hơi ×2, chỉ số khô hạn ×2, mùa mưa, tỷ lệ tuyết, tần suất và thời gian mưa lớn / khô ×4), lớp phủ (loại chủ đạo, tỷ lệ nông nghiệp / trơ / rừng / hồ / đô thị, LAI ×2, NDVI ×2, GVF ×2) |
| Độ dài chuỗi | 24 bước giờ = 1 ngày | 364 bước ngày hindcast + 1 bước forecast |
| Kích thước một mẫu | 358 trạm × 24 × 5 = **42.960 giá trị** (mỗi trạm 120) | 364 × 32 + 1 × 5 + 33 = **11.686 giá trị** (không Q: 11.322) |
| Kích thước một batch | 64 đồ thị × 358 trạm = 22.912 chuỗi trạm, ≈ 2,75 triệu giá trị | 256 chuỗi lưu vực, ≈ 2,99 triệu giá trị |
| Số mẫu train | ≈ 69.900 ảnh chụp mỗi cách chia (69.944 / 69.848 / 69.896; cửa sổ cắt trong từng năm, mỗi năm mất 29 giờ); 4/5 dùng để train ≈ 55.955, tức ≈ 20 triệu mẫu trạm–giờ | Tối đa 2.557 ngày × 451 = **1.153.207** mẫu lưu vực–ngày (trước khi loại mẫu thiếu nhãn) |
| Số mẫu test | 17.486 ảnh chụp × 358 = 6,26 triệu mẫu trạm–giờ | Tối đa 1.461 × 451 = 658.911 mẫu lưu vực–ngày |
| Cách mô hình phân biệt trạm | Không có thuộc tính: MLP dùng chung trọng số cho mọi trạm; GNN chỉ phân biệt qua vị trí trên đồ thị | Qua 33 thuộc tính tĩnh |

## 6. Đầu ra

| Khía cạnh | #104 | #121 |
|---|---|---|
| Số giá trị mỗi mẫu | 358 (một Q cho mỗi trạm) | 1 (qmax của lưu vực trong ngày t) |
| Dạng | Hồi quy điểm, Q đã chuẩn hóa theo trạm, khử chuẩn hóa về m³/s khi đánh giá | Hồi quy điểm (head `regression`, kích hoạt tuyến tính), đơn vị mm/ngày; đầu ra âm được cắt về 0 (`clip_targets_to_zero`). NSE theo lưu vực không đổi khi đổi đơn vị, nên con số so sánh được; muốn ra m³/s phải nhân lại diện tích |
| Bất định | Không | Không (có head CMAL/GMM trong NeuralHydrology nhưng bài không dùng) |
| Nhiều lead time | Không (Phụ lục A.4 chạy riêng lead 1–12 giờ) | Không (dự báo nhiều ngày là "nghiên cứu sắp tới" theo Mục 3.7) |

## 7. Mô hình

| Khía cạnh | #104 | #121 |
|---|---|---|
| Kiến trúc chính | Làm phẳng 24 × 5 → affine 120→128 → 19 lớp GNN (bằng đường đi dài nhất của đồ thị) → affine 128→1 | Nhúng động FC 32→16 (hindcast), 5→16 (forecast); nhúng tĩnh FC 33→16; LSTM 1 lớp, 128 đơn vị, chạy liên tục qua 364 bước hindcast rồi 1 bước forecast; dropout 0,3; head tuyến tính 128→1 |
| Bộ mã hóa thời gian | **Không có**: 24 giờ bị làm phẳng thành 120 số | LSTM qua 365 bước |
| Các biến thể so sánh | 3 loại lớp (ResGCN, GCNII, ResGAT) × 6 định nghĩa cạnh (cô lập, nhị phân, chiều dài sông, chênh cao, độ dốc, học được / tất cả) × 3 hướng; MLP 2 lớp × 512 | CUDA LSTM (3 baseline dữ liệu), CrossDomain, Encoder–Decoder LSTM (handoff, Nearing và cs. 2024), Sequential Forecast LSTM, học chuyển giao (toàn bộ trọng số / chỉ lớp nhúng); nhúng đơn giản (16) và phức tạp (30-20-64) |
| Số tham số | Đếm từ checkpoint: GCNII 327.270 (gồm 357 trọng số cạnh học được); ResGCN 329.345; ResGAT 645.505; MLP 587.777 | Tính từ mã và cấu hình: Sequential Forecast LSTM nhúng đơn giản 84.785 (LSTM 82.944); nhúng phức tạp 143.291 |
| Sai khác mã – bài | "19-layer MLP" (Mục 4.2) và "wide 2-layer MLP" (Bảng 2); ResGAT bỏ qua trọng số cạnh | Nhúng đơn giản mô tả có tanh (Mục 2.2.6) nhưng mã `FC` với một lớp là tuyến tính thuần (Mục 3.5 của bài cũng gọi là "linear"); phương trình (7), (10) ghi Q quá khứ là `q_max`, mã dùng `qmean` |

## 8. Huấn luyện

| Khía cạnh | #104 | #121 |
|---|---|---|
| Optimizer | Adam, lr 1e-4, L2 1e-5, cố định | Adam, lr 1e-3, CosineAnnealing (T_max 30, lr min 1e-5), cắt gradient 1 |
| Epoch, batch | 100 epoch, batch 64 đồ thị (≈ 875 bước/epoch, ≈ 87.500 bước) | Tối đa 30 epoch, dừng sớm (patience 5, min_delta 0,005 NSE validation), batch 256 (≤ 4.505 bước/epoch) |
| Hàm mất mát | MSE × "relevancy score" của cửa sổ (1000 × (đạo hàm trung bình)² × tích phân Q/μ; mã lấy trung bình đạo hàm trên cả batch) | NSE* (Kratzert 2019): MSE chia (độ lệch chuẩn lưu vực + 0,1)² |
| Chọn mô hình | Epoch có loss validation nhỏ nhất (validation ngẫu nhiên, chồng lấn train). Ví dụ checkpoint: GCNII tốt nhất ở epoch 97/100, MLP ở epoch 88/100 — loss còn giảm gần cuối | Epoch có NSE validation tốt nhất (validation theo thời gian) |
| Siêu tham số | Cố định cho mọi cấu hình, không báo cáo tìm kiếm | Tối ưu Bayes 100 lần thử trên validation (hidden 64/128/256, dropout 0,1–0,3, nhiễu nhãn 0,001–0,1, batch 64/128/256) |
| Hạt giống | 42, một lần chạy mỗi cấu hình và cách chia | 111, một lần chạy mỗi cấu hình (Phụ lục G) |
| Số lần huấn luyện công bố | 957 (162 chính + 3 MLP + 108 ablation + 684 mạng con) | 24 cấu hình trên Zenodo (20 dòng ở Bảng F1) |

## 9. Đánh giá và kết quả

| Khía cạnh | #104 | #121 |
|---|---|---|
| Chỉ số | NSE có trọng số relevancy theo trạm, trung bình qua 358 trạm, ghi dạng % (NSE × 100); mẫu số dùng trung bình Q kỳ 2000–2015 | NSE và KGE chuẩn theo từng lưu vực trên 4 năm test; báo cáo trung vị, trung bình, độ lệch chuẩn, min, max, phân vị 10/25/75/90 (Bảng F1); đường CDF qua 451 lưu vực |
| Kết quả chính | Mọi tổ hợp 80,2–85,6%; MLP 85,37 ± 1,64; GCNII học được hai chiều 85,56 ± 1,41 (± là qua 3 cách chia năm) | Trung vị NSE: chỉ dự báo 0,39 · CrossDomain 0,58 → 0,33 · học chuyển giao 0,41–0,44 · Encoder–Decoder 0,57 / có Q 0,66–0,67 · Sequential 0,61–0,63 / có Q 0,70–0,71 · tái phân tích (thời tiết hoàn hảo) 0,69 |
| Kết luận của tác giả | Đồ thị sông không cải thiện so với MLP | Sequential Forecast LSTM giảm sai lệch dự báo thời tiết tốt nhất; thêm Q quá khứ tăng rõ và vượt baseline tái phân tích; học chuyển giao kém ổn định |
| Baseline đơn giản | Không có persistence | Không có persistence; có baseline chỉ dự báo, tái phân tích, kết hợp |
| Độ tin cậy con số | Công thức NSE sai (thổi phồng mẫu số theo μ²); nhiễu giữa hai cấu hình tương đương tới 6,67 điểm % — cỡ toàn bộ khoảng 80,2–85,6% | Tự tính lại từ tệp của tác giả khớp bài; chênh lệch cỡ 0,01–0,02 giữa các biến thể chưa tách được khỏi nhiễu (1 hạt giống) |

## 10. Tính toán

| Khía cạnh | #104 | #121 |
|---|---|---|
| Phần cứng, thời gian | Bài không báo cáo | RTX 4090; vài phút tới khoảng 1 giờ mỗi lần chạy (Phụ lục G) |
| Bộ nhớ dữ liệu khi huấn luyện | Nạp toàn bộ ≈ 1,1 GB vào RAM | Nhỏ hơn; ≈ 0,1 tỷ giá trị cho 451 lưu vực × 16 năm × ~38 cột |
| Khả thi trên Colab / Kaggle miễn phí | Được với mô hình gốc; cửa sổ dài cần viết lại bộ nạp (cắt theo năm) | Mô hình nhỏ (dưới 0,15 triệu tham số), dữ liệu nhẹ; chưa đo thời gian trên T4 / P100 |

## 11. Tái lập và mở rộng

| Khía cạnh | #104 | #121 |
|---|---|---|
| Khung mã | Tự viết (PyTorch Geometric), ~880 dòng | NeuralHydrology — khung chuẩn của cộng đồng; có sẵn LSTM, GRU, Transformer, EA-LSTM, MTS-LSTM, handoff |
| Tái lập số liệu | 957 checkpoint: tính lại được mà không cần huấn luyện, nhưng phải sửa hàm NSE | Có `best_model.pt` và `test_metrics.csv`. Biến thể có Q cần tự tạo thư mục `LamaH_expanded_q_input` (không có trên Zenodo) |
| Chèn Mamba | Thay lớp affine bằng Mamba mã hóa chuỗi thời gian từng trạm, trước GNN | Thay LSTM trong Sequential Forecast LSTM (thêm vào `modelzoo`); lớp Mamba có sẵn của NeuralHydrology quét sai trục (`CHECKCODE.md` Mục 13) |
| Khoảng trống tác giả tự nêu | Cửa sổ dài bị giới hạn vì lớp affine; GNN bỏ lỡ đỉnh đột ngột | Chỉ thử kiến trúc LSTM; chỉ dữ liệu ngày; chỉ lead 1 ngày; chỉ Trung Âu; độ trễ công bố tái phân tích (Mục 3.7) |
| Mở rộng sang đồ thị | Có sẵn | Làm được với `Stream_dist.csv` (27,5% lưu vực lồng nhau) |

## 12. Lỗi và điểm yếu đã xác nhận

**#104** (`CHECKCODE.md` Mục 1):
- NSE sai công thức: lấy μ (m³/s) trừ nhãn đã chuẩn hóa.
- Nhiễu giữa các lần chạy cỡ toàn bảng kết quả.
- Validation ngẫu nhiên trên cửa sổ chồng lấn với train.
- Relevancy tính trên cả batch.
- ResGAT bỏ trọng số cạnh.
- `strict=False` khi nạp checkpoint.
- Cửa sổ không vượt ranh giới năm.
- Mâu thuẫn MLP 19 lớp / 2 lớp.
- Không có baseline persistence; không có thuộc tính tĩnh.

**#121** (`CHECKCODE.md` Mục 15):
- 1 hạt giống.
- Baseline tái phân tích dùng thời tiết của chính ngày nhãn (mốc trên, không vận hành được).
- Không lùi đầu vào theo độ trễ công bố (tác giả thừa nhận ở Mục 3.7).
- Thiếu thư mục `q_input` trên Zenodo.
- `_is_best_model` bỏ qua `min_delta`.
- Phương trình ghi `q_max` nhưng mã dùng `qmean`.
- Nhãn qmax ở mm/ngày còn `qmean` đầu vào nhiều khả năng ở m³/s; với z-score toàn cục, `qmean` của lưu vực nhỏ bị nén sát một giá trị (bài không nêu).
- Mô tả nhúng đơn giản không khớp mã.
- Không có baseline persistence.
- Chuỗi 364 ngày nhưng dự báo chỉ 1 bước.

## 13. Phản biện công khai

| | #104 | #121 |
|---|---|---|
| Có công khai không | Không: OpenReview của ICML 2024 chỉ có tóm tắt và PDF | Có: 2 phản biện, 2 vòng, trả lời của tác giả, bản đánh dấu sửa đổi (HESS Discussions egusphere-2025-4978) |
| Đánh giá chung | Không biết | Tích cực ở cả hai phản biện; vòng 2 xác nhận mọi ý đã được giải quyết |
| Ý phê bình chính | Không biết | Chỉ lead 1 ngày; cách đặt vấn đề "bias" trong khi Q quan trắc chủ yếu giảm bất định trạng thái ban đầu; thiếu phân tích theo lưu vực (đã bổ sung Mục 2.3, 3.1.1, 3.6); tanh ở lớp nhúng; đề nghị so với persistence — tác giả không thêm, chỉ lập luận rằng qmax ngày tự tương quan thấp |
| Ý nghĩa cho tiểu luận | Không có | Persistence và dự báo nhiều ngày là hai điểm mở mà người phản biện đã chỉ ra — làm được và có căn cứ để nêu trước hội đồng; persistence đã tính: NSE trung vị 0,35–0,37, mô hình tốt nhất hơn ở trên 92% lưu vực (`LamaHCE/LamaHCE.md` Mục 6.1) |

Chi tiết: `CHECKPDF.md` Mục 5.12.

## 14. Tổng hợp

| Khía cạnh | Bên hơn | Lý do ngắn |
|---|---|---|
| Nơi công bố, trích dẫn | #104 | ICML A*, ngành AI, 21 trích dẫn so với HESS Q1 thủy văn, 3 trích dẫn |
| Độ tin cậy số liệu gốc | #121 | Không có lỗi làm sai số công bố; #104 phải tính lại NSE |
| Phương pháp đánh giá | #121 | Chia theo thời gian, tối ưu siêu tham số trên validation, thống kê đầy đủ theo lưu vực |
| Sát vận hành | #121 | Dự báo thời tiết thật + Q quan trắc; #104 không có thông tin tương lai |
| Độ phong phú đầu vào | #121 | 31–32 biến + 5 biến dự báo + 33 thuộc tính so với 5 biến, không thuộc tính |
| Mô hình mốc | #121 | Dòng LSTM dự báo đang dùng vận hành (Google); #104 không có bộ mã hóa thời gian |
| Độ phân giải thời gian | #104 | Giờ, phù hợp chuỗi dài; #121 theo ngày |
| Quy mô không gian, đồ thị | #104 | Mạng 358 trạm có cấu trúc sông |
| Dữ liệu nhẹ | #121 | 0,95 GB so với 14,8 GB |
| Khung mã | #121 | NeuralHydrology, sẵn nhiều baseline |
| Minh bạch phản biện | #121 | Phản biện công khai, tích cực; #104 không công khai |

Việc còn phải xác nhận trước khi chốt:
- Yêu cầu của GVHD về số trích dẫn tối thiểu cho bài cơ sở.
- Căn chỉnh múi giờ giữa HRES và ngày LamaH-CE ở #121.
- Tỷ lệ mẫu thiếu nhãn thực tế của hai bộ.
