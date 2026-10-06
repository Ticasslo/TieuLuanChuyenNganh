# Giới thiệu bài cơ sở — BiasCast (HESS 2026)

> Tài liệu trình bày bài báo được chọn làm bài cơ sở cho đề tài *Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy*: nội dung bài, mô hình, chỉ số đánh giá, kết quả, lý do chọn, hạn chế phát hiện khi đọc mã nguồn và phản biện, hướng cải tiến. Pipeline của đề tài: `Document/1-KeHoach/KienTrucPipeline.md`.

## 1. Thông tin bài báo

| Mục | Nội dung |
|---|---|
| Tên bài | *BiasCast: learning and adjusting real time biases from meteorological forecasts to enhance runoff predictions* |
| Tác giả | Oliver Konold, Moritz Feigl, Patrick Podest, Christoph Klingler, Karsten Schulz (BOKU University, Vienna; baseflow AI solutions, Vienna; Johannes Kepler University Linz) |
| Nơi công bố | Hydrology and Earth System Sciences (HESS), 30, 5067–5096, 2026; bản thảo đăng HESS Discussions 27/11/2025 |
| Xếp hạng | Scimago 2025: Q1 (Earth and Planetary Sciences; Water Science and Technology), SJR 2,035, H-index 194 |
| Trích dẫn | 3 (Google Scholar, 26/9/2026), trong đó có AIFL — mô hình dự báo lưu lượng toàn cầu của ECMWF (J. Hydrol. 2026) |
| Trang chính thức | https://hess.copernicus.org/articles/30/5067/2026/ |
| Phản biện | Công khai: https://egusphere.copernicus.org/preprints/2025/egusphere-2025-4978/ |
| Mã nguồn | https://github.com/conestone/neuralhydrology — thư viện NeuralHydrology do tác giả chỉnh sửa: mô hình (`modelzoo`), bộ đọc dữ liệu (`datasetzoo`), vòng huấn luyện (`training`), đánh giá (`evaluation`), điều khiển bằng tệp cấu hình `config.yml`; https://github.com/conestone/biascast — notebook gọi thư viện để chạy thí nghiệm, mã phân tích kết quả và vẽ hình trong bài (không chứa mô hình) |
| Kết quả, trọng số | https://doi.org/10.5281/zenodo.17241922 |
| DOI | 10.5194/hess-30-5067-2026 |

Trích dẫn: Konold, O., Feigl, M., Podest, P., Klingler, C., & Schulz, K. (2026). BiasCast: learning and adjusting real time biases from meteorological forecasts to enhance runoff predictions. *Hydrology and Earth System Sciences*, 30, 5067–5096. https://doi.org/10.5194/hess-30-5067-2026

## 2. Tóm tắt

Mô hình học sâu thủy văn thường được huấn luyện trên dữ liệu tái phân tích (khí tượng quá khứ đã hiệu chỉnh, chất lượng cao), nhưng khi vận hành phải chạy với dự báo thời tiết — kém chính xác hơn. Bài đo mức suy giảm này và so sánh các cách huấn luyện, các kiến trúc LSTM để mô hình tự học và bù sai lệch của dự báo thời tiết, khi dự báo lưu lượng cực đại ngày cho 451 lưu vực Trung Âu.

Kết quả chính: mô hình học trên tái phân tích rồi chạy với dự báo thật giảm NSE trung vị từ 0,58 xuống 0,33; kiến trúc Sequential Forecast LSTM đạt 0,63; thêm lưu lượng quan trắc gần nhất đạt 0,71, vượt cả mô hình dùng thời tiết "hoàn hảo" (0,69).

Điểm quan trọng đối với đề tài: tác giả chỉ thử các kiến trúc LSTM và tự nêu đây là hạn chế (Mục 3.7). Đề tài thay lõi LSTM bằng Mamba và so sánh với GRU, Transformer, S4D trong cùng khung.

## 3. Bài toán

Dự báo cho từng lưu vực, một mô hình chung cho cả 451 lưu vực.

- **Đầu vào:** 364 ngày quá khứ (hindcast) gồm 31 biến khí tượng tái phân tích, có hoặc không có lưu lượng trung bình ngày quan trắc; dự báo thời tiết ECMWF HRES cho ngày cần dự báo (5 biến); 33 thuộc tính tĩnh của lưu vực.
- **Đầu ra:** lưu lượng cực đại trong ngày t (qmax) — đỉnh lưu lượng trong 24 giờ sau lần phát hành dự báo lúc 00 UTC.

Ví dụ một mẫu: dự báo cho ngày 15/05/2016 dùng khí tượng và lưu lượng từ 17/05/2015 tới 14/05/2016, dự báo ECMWF phát hành 00 UTC ngày 15/05/2016, và cho ra lưu lượng lớn nhất trong ngày 15/05/2016.

Đây là bài toán **dự báo**: dùng dữ liệu tới thời điểm phát hành, có lưu lượng quan trắc quá khứ và dự báo thời tiết thật cho tương lai. Bài toán **mô phỏng** chỉ dùng khí tượng để ước lượng lưu lượng của chính thời điểm đó, không dùng lưu lượng quan trắc. Dự báo khớp với tên đề tài.

Lưu lượng cực đại ngày được chọn vì gần với cảnh báo lũ hơn lưu lượng trung bình ngày. Trong mã, nhãn được đổi sang đơn vị mm/ngày (chia cho diện tích lưu vực) — vẫn là lưu lượng, đổi ngược về m³/s bằng cách nhân diện tích.

## 4. Dữ liệu và cách chia tập

- **Extended LamaH-CE** (Konold và cs., Zenodo 10.5281/zenodo.17119635, bản 1.0, giấy phép CC BY-NC 4.0, 0,95 GB): bản mở rộng của LamaH-CE (Klingler và cs., ESSD 2021) cho 859 lưu vực Trung Âu, theo ngày, thêm bốn nguồn khí tượng (E-OBS, MSWEP, GLEAM, dự báo ECMWF HRES) gộp theo lưu vực, và lưu lượng cực tiểu, trung bình, cực đại ngày trích từ dữ liệu giờ.
- **Lưu vực:** 451 lưu vực ít hoặc không chịu tác động của con người, mức A (toàn bộ lưu vực thượng nguồn của trạm, tương đương cách gộp của CAMELS); 72,5% là lưu vực đầu nguồn, 27,5% lồng nhau; từ vùng núi cao Alps tới đồng bằng.
- **Biến đầu vào** (Phụ lục A): ERA5-Land 21 biến (nhiệt độ, điểm sương, gió, albedo, chỉ số lá, tuyết, bức xạ, áp suất, bốc thoát hơi, mưa, độ ẩm đất hai lớp); E-OBS 7; MSWEP 1 (mưa); GLEAM 2 (bốc hơi thực, tiềm năng); ECMWF HRES 5 (nhiệt độ, điểm sương, bức xạ, mưa, bốc hơi) — trung bình 8 giá trị 3 giờ tính từ lần phát hành 00 UTC; 33 thuộc tính địa hình, khí hậu, lớp phủ.
- **Chia tập theo thời gian:** train 2003–2009, validation 2010–2013, test 2014–2017; NeuralHydrology nạp thêm giai đoạn khởi động trước mỗi kỳ nên validation và test đủ 4 năm.
- **Quy mô:** tối đa 1,15 triệu mẫu lưu vực–ngày cho train, 0,66 triệu cho test.

Vị trí trong bài: Mục 2.1, 2.2; Phụ lục A (Bảng A1, A2).

## 5. Mô hình

Mọi mô hình dùng thư viện NeuralHydrology, gồm mạng nhúng đầu vào → LSTM → lớp hồi quy.

| Mô hình | Cách hoạt động |
|---|---|
| CUDA LSTM (baseline) | LSTM chuẩn; ba biến thể dữ liệu: chỉ dự báo (cận dưới), chỉ tái phân tích với thời tiết của chính ngày dự báo (cận trên, không vận hành được), tái phân tích + dự báo |
| CrossDomain | Huấn luyện trên 5 biến tái phân tích, chạy với 5 biến dự báo tương ứng — mô phỏng cách làm truyền thống |
| Học chuyển giao | Huấn luyện trên tái phân tích, tinh chỉnh trên dự báo: toàn bộ trọng số hoặc chỉ mạng nhúng |
| Encoder–Decoder LSTM | Hai LSTM: một cho hindcast, một cho forecast, nối bằng mạng handoff — kiến trúc của mô hình dự báo lũ Google (Nearing và cs., Nature 2024) |
| Sequential Forecast LSTM | Một LSTM chạy liên tục qua 364 ngày hindcast rồi ngày forecast, mỗi pha có mạng nhúng riêng |

- Thêm hai trục so sánh: có hoặc không có lưu lượng quan trắc trong hindcast; mạng nhúng đơn giản (16 chiều) hoặc phức tạp (30-20-64).
- Mô hình tốt nhất (Sequential Forecast LSTM có Q, nhúng đơn giản): LSTM 128 chiều, dropout đầu ra 0,3, khoảng 85 nghìn tham số (tính từ mã và cấu hình).

Siêu tham số (Mục 2.2, Phụ lục D, cấu hình trên Zenodo):

| Nhóm | Giá trị |
|---|---|
| Dữ liệu | Chuỗi 365 ngày (364 hindcast + 1 forecast) |
| Huấn luyện | Adam, learning rate 10⁻³, CosineAnnealing, tối đa 30 epoch, batch 256, cắt gradient 1 |
| Dừng sớm, chọn mô hình | Theo NSE validation (patience 5, cải thiện tối thiểu 0,005) |
| Tìm siêu tham số | Tối ưu Bayes 100 lần thử trên validation: số chiều ẩn 64/128/256, dropout 0,1–0,3, nhiễu nhãn, batch 64/128/256 |
| Hạt giống | 1 (Phụ lục G) |
| Phần cứng | RTX 4090; vài phút tới khoảng 1 giờ mỗi lần chạy |

## 6. Hàm mất mát và chỉ số đánh giá

**Hàm mất mát NSE\*** (Kratzert và cs. 2019): sai số bình phương của mỗi lưu vực chia cho bình phương (độ lệch chuẩn lưu lượng của lưu vực + 0,1), rồi lấy trung bình. Cách chia này giúp lưu vực lớn và nhỏ đóng góp ngang nhau.

**Chỉ số đánh giá:** Nash–Sutcliffe Efficiency (NSE) và Kling–Gupta Efficiency (KGE) theo từng lưu vực trên 4 năm test.

- NSE = 1 − Σ(Q dự báo − Q thực)² / Σ(Q trung bình − Q thực)². NSE = 1: hoàn hảo; 0: ngang đoán bằng trung bình; âm: tệ hơn.
- Báo cáo trung vị, trung bình, độ lệch chuẩn, min, max, phân vị 10/25/75/90 qua 451 lưu vực (Bảng F1) và đường phân phối tích lũy (CDF) của NSE.

**Các phân tích bổ sung:**

- Khoảng cách Wasserstein giữa phân phối tái phân tích và dự báo của từng lưu vực, tương quan với thuộc tính lưu vực (Mục 3.1.1): lưu vực núi cao, dốc, nhiều tuyết có sai lệch dự báo lớn nhất.
- Chênh lệch NSE theo lưu vực so với baseline chỉ dự báo, tương quan với thuộc tính (Mục 3.6): lưu vực khô hạn được lợi nhiều nhất.

Vị trí trong bài: Mục 2.2, 2.3; Phụ lục C.

## 7. Kết quả của bài

NSE trung vị trên 451 lưu vực, test 2014–2017 (Bảng F1):

| Mô hình / cấu hình | NSE trung vị |
|---|---|
| CrossDomain: học trên tái phân tích → chạy với dự báo | 0,58 → 0,33 |
| Baseline chỉ dự báo (cận dưới) | 0,39 |
| Học chuyển giao | 0,41 – 0,44 |
| Encoder–Decoder LSTM | 0,57 |
| Sequential Forecast LSTM | 0,63 |
| Encoder–Decoder LSTM có Q quan trắc | 0,66 – 0,67 |
| Baseline tái phân tích với thời tiết hoàn hảo (cận trên) | 0,69 |
| **Sequential Forecast LSTM có Q quan trắc** | **0,70 – 0,71** |

- Thay tái phân tích bằng dự báo thời tiết làm mô hình suy giảm rõ — không thể huấn luyện trên tái phân tích rồi chạy với dự báo.
- Sequential Forecast LSTM bù sai lệch tốt nhất và ổn định nhất; thêm lưu lượng quan trắc vượt cả cận trên.
- Đưa dự báo lưu trữ vào pha hindcast và mạng nhúng phức tạp không giúp ổn định.

Vị trí trong bài: Mục 3.1–3.6, Hình 3–10, Phụ lục F.

## 8. Lý do chọn bài

**Quá trình chọn:** tra 137 bài báo từ 2024 có huấn luyện mô hình học sâu dự báo lưu lượng trên dữ liệu công khai dài năm; đọc mã nguồn 35 bài; xếp hạng 13 ứng viên; đọc toàn văn, mã nguồn và phản biện của các ứng viên đứng đầu; tính lại kết quả của tác giả từ tệp công bố.

**Lý do:**

1. **Đúng bài toán dự báo, sát vận hành nhất trong các ứng viên:** dùng cả lưu lượng quan trắc và dự báo thời tiết thật, dự báo đỉnh lưu lượng ngày.
2. **Số liệu tin được:** NSE theo lưu vực tính lại từ tệp kết quả của tác giả khớp với bài (ví dụ 0,705 cho mô hình tốt nhất); mã huấn luyện là thư viện chuẩn NeuralHydrology, bản fork chỉ sửa bộ nạp dữ liệu, dừng sớm và lập lịch learning rate.
3. **Phương pháp chặt chẽ:** chia tập theo thời gian tách hẳn train, validation, test; tối ưu siêu tham số trên validation; kiểm tra trong mã không rò rỉ nhãn.
4. **Mô hình mốc mạnh:** các kiến trúc hindcast–forecast cùng dòng với mô hình dự báo lũ vận hành của Google.
5. **Khoảng trống do chính tác giả nêu:** chỉ thử kiến trúc LSTM, chỉ dữ liệu ngày, chỉ dự báo 1 ngày — đúng chỗ đề tài đưa Mamba vào; người phản biện đề nghị so với persistence nhưng bài chưa làm.
6. **Phản biện công khai, đánh giá tích cực:** hai người phản biện, hai vòng.
7. **Tái lập được, dữ liệu nhẹ:** cấu hình, trọng số và kết quả theo lưu vực của mọi thí nghiệm công khai; dữ liệu 0,95 GB; chạy được trên GPU miễn phí (Kaggle, Google Colab).
8. **Hỗ trợ demo:** 451 lưu vực có ranh giới, tọa độ; dữ liệu trạm có từ 1981 để tính ngưỡng lũ.

**So với Kirschstein & Sun (ICML 2024), ứng viên đã cân nhắc:** Kirschstein & Sun hơn về uy tín nơi công bố (hội nghị A* ngành học máy, 21 trích dẫn) và có dữ liệu theo giờ, mạng sông; nhưng công thức NSE trong mã tính sai, nhiễu giữa các lần huấn luyện lớn cỡ toàn bộ chênh lệch trong bảng kết quả, validation lấy lẫn trong train, mô hình không có thành phần học chuỗi thời gian và bài toán không dùng dự báo thời tiết. Chi tiết: `Document/6-ChonBaiCoSo/CHECKPDF.md` Mục 8.4c, 8.6.

## 9. Hạn chế phát hiện khi đọc mã nguồn và phản biện

- **Một hạt giống:** mỗi cấu hình chạy 1 lần; chênh lệch nhỏ (khoảng 0,01–0,02 NSE) giữa các biến thể chưa tách được khỏi nhiễu.
- **Không có baseline persistence:** người phản biện 1 đề nghị; tác giả chỉ lập luận rằng lưu lượng cực đại ngày ít tự tương quan, chưa kiểm bằng số. Nhóm đã tự tính trên tập test: persistence đạt NSE trung vị 0,35–0,37; mô hình tốt nhất (0,705) hơn persistence ở trên 92% lưu vực, còn mô hình chỉ dùng dự báo thời tiết (0,387) chỉ ngang persistence.
- **Baseline tái phân tích dùng thời tiết của chính ngày dự báo** — là mốc tham chiếu trên, không phải cấu hình vận hành được.
- **Không lùi đầu vào theo độ trễ công bố thật** của tái phân tích (tác giả tự nêu ở Mục 3.7).
- **Thiếu dữ liệu cho biến thể có Q:** mã đọc thư mục `LamaH_expanded_q_input` không có trên Zenodo; phải tự dựng.
- **Đơn vị lưu lượng không đồng nhất:** nhãn qmax được đổi sang mm/ngày, còn lưu lượng quan trắc đầu vào nhiều khả năng giữ m³/s (suy từ thống kê chuẩn hóa của tác giả, chưa kiểm được vì thư mục trên không công khai).
- **Sai khác nhỏ giữa bài và mã:** mạng nhúng đơn giản được mô tả có hàm kích hoạt tanh nhưng trong mã là một lớp tuyến tính; phương trình (7), (10) ghi Q quá khứ là q_max trong khi mã dùng lưu lượng trung bình ngày; hàm lưu mô hình tốt nhất không tính ngưỡng cải thiện tối thiểu của dừng sớm.
- **Phạm vi hẹp do tác giả tự nêu:** chỉ Trung Âu, chỉ dự báo 1 ngày, chỉ kiến trúc LSTM, chỉ dữ liệu ngày.
- **Bộ dữ liệu bản 1.0:** trang Zenodo ghi chưa phải bản sửa cuối; giấy phép phi thương mại.

Các hạn chế trên không làm sai số liệu công bố, và phần lớn là chỗ đề tài bổ sung: persistence, nhiều hạt giống, thống nhất đơn vị lưu lượng, so sánh nhiều kiến trúc.

## 10. Hướng cải tiến

**Câu hỏi nghiên cứu:** trong khung dự báo hindcast–forecast có dự báo thời tiết thật, lõi thời gian dạng SSM (Mamba) có cải thiện so với Sequential Forecast LSTM và các kiến trúc khác không, và chuỗi quá khứ dài hơn có giúp không. Kết quả khẳng định hay phủ định đều là câu trả lời hợp lệ; đề tài đặt vấn đề là nghiên cứu so sánh, không mặc định Mamba phải thắng.

**Phần cốt lõi:**

1. Tái lập bài gốc bằng trọng số của tác giả; bổ sung baseline persistence; đo nhiễu giữa các hạt giống.
2. Cải tiến LSTM của bài (tầng 1): giữ mẫu có dữ liệu thiếu bằng masked mean thay vì bỏ cả mẫu; hoàn thiện đầu vào lưu lượng quan trắc (cùng đơn vị với nhãn, Q các ngày trước, xử lý Q thiếu); huấn luyện có bỏ ngẫu nhiên lưu lượng quan trắc để một mô hình dùng được cả khi trạm mất số liệu (hướng tiếp theo do chính tác giả nêu); tổ hợp nhiều hạt giống.
3. Thay lõi LSTM bằng GRU, Transformer, S4D và Mamba ở dạng thuần trên quy trình đã cải tiến (tầng 2), cùng dữ liệu, cùng chia tập, cùng ngân sách tinh chỉnh, nhiều hạt giống; sau đó một biến thể cải tiến cho Mamba và một cho Transformer, mỗi biến thể so với bản thuần của chính nó. S4D là đối chứng để phân biệt "Mamba không phù hợp" với "SSM nói chung không phù hợp".
4. Thử chuỗi quá khứ 730 ngày cho LSTM và Mamba — nơi Mamba được kỳ vọng có lợi thế chi phí.
5. Đầu ra xác suất (CMAL) trên cấu hình tốt nhất để có xác suất vượt ngưỡng lũ.
6. Đánh giá sát vận hành và theo loại lưu vực: kịch bản mất lưu lượng quan trắc 1–7 ngày và tái phân tích đến trễ 5 ngày (ERA5 công bố trễ 5 ngày); so sánh Mamba với LSTM theo từng lưu vực, tương quan với thuộc tính lưu vực để biết Mamba hơn ở loại lưu vực nào.

**Phần đáp ứng yêu cầu tiểu luận:**

7. Giải thích mô hình (XAI) bằng Integrated Gradients (thư viện Captum): ngày nào trong quá khứ và nhóm biến nào (tái phân tích, dự báo thời tiết, lưu lượng quan trắc) ảnh hưởng tới dự báo.
8. Demo bản đồ dự báo theo ngày cho 451 lưu vực, tô màu theo ngưỡng chu kỳ lặp lại (return period) của lưu lượng, xem chuỗi lưu lượng và giải thích XAI từng lưu vực.

**Giai đoạn khóa luận:** dự báo nhiều ngày; kết hợp dữ liệu giờ của LamaH-CE gốc; ưu tiên đỉnh lũ trong hàm mất mát; đồ thị mạng sông giữa các lưu vực lồng nhau (ý tưởng từ Kirschstein & Sun); thêm các biến thể lai; nâng demo thành phần mềm ứng dụng. Phạm vi chi tiết: `Document/1-KeHoach/KienTrucPipeline.md` Mục 9.

## 11. Pipeline và kiến trúc chi tiết của bài (đọc từ mã)

Nguồn: bản fork NeuralHydrology của tác giả (`datasetzoo/basedataset.py`, `datasetzoo/lamah.py`, `modelzoo/inputlayer.py`, `modelzoo/sequential_forecast_lstm.py`, `modelzoo/handoff_forecast_lstm.py`, `training/loss.py`, `training/basetrainer.py`, `training/early_stopping.py`, `training/logger.py`, `evaluation/tester.py`), notebook `Run_Experiment.ipynb` và 24 tệp `config.yml` trên Zenodo. Toàn bộ pipeline chạy bằng một lệnh `start_run(config.yml)`, sau đó `eval_run(run_dir, "test")`.

### 11.1. Các bước xử lý

| Bước | Việc | Chi tiết trong mã |
|---|---|---|
| 1. Đọc dữ liệu | Mỗi lưu vực một tệp CSV ngày | Khí tượng mức A (31 biến tái phân tích + 5 biến ECMWF; `qmean` ở biến thể có Q đọc từ thư mục riêng của tác giả); nhãn `qmax` đọc từ tệp trạm (m³/s) và chia theo diện tích `area_gov` thành mm/ngày; 33 thuộc tính tĩnh |
| 2. Cắt theo kỳ | Train 2003–2009, validation 2010–2013, test 2014–2017 | Mỗi kỳ nạp thêm 364 ngày khởi động trước ngày bắt đầu; nhãn trong giai đoạn khởi động bị đặt thành rỗng |
| 3. Thống kê chuẩn hóa | Tính trên kỳ train, lưu `train_data_scaler.yml` | z-score toàn cục mỗi biến (trung bình, độ lệch chuẩn qua mọi lưu vực và ngày); thuộc tính tĩnh chuẩn hóa qua 451 lưu vực; riêng hàm mất mát cần độ lệch chuẩn `qmax` của từng lưu vực, tính trên dữ liệu chưa chuẩn hóa |
| 4. Lọc mẫu | Mỗi mẫu = một lưu vực × một ngày t | Khi huấn luyện, mẫu bị loại nếu **bất kỳ biến động nào trong 37 cột có giá trị thiếu ở bất kỳ ngày nào của cửa sổ 365 ngày** (kể cả các cột ECMWF ở những ngày mô hình không dùng tới), hoặc nhãn thiếu, hoặc thuộc tính thiếu; khi validation và test, giữ mọi mẫu đủ lịch sử |
| 5. Lấy mẫu | Trộn ngẫu nhiên mẫu của mọi lưu vực | Batch 256 mẫu |
| 6. Lan truyền thuận | Xem Mục 11.2 | Đầu ra cho cả 365 bước, chỉ bước cuối (ngày t) được dùng (`predict_last_n: 1`) |
| 7. Hàm mất mát | NSE* | Trung bình của (ŷ − y)² / (σ_lưu vực + 0,1)² trên các mẫu có nhãn; ŷ, y ở dạng đã chuẩn hóa |
| 8. Tối ưu | Adam, learning rate 10⁻³ | CosineAnnealing (T_max 30, nhỏ nhất 10⁻⁵), cắt gradient 1, tối đa 30 epoch |
| 9. Validation mỗi epoch | Dự báo toàn bộ 2010–2013 cho 451 lưu vực | Chỉ số theo dõi = **trung vị** NSE qua các lưu vực; dừng sớm khi 5 epoch liên tiếp không tăng quá 0,005; lưu `best_model.pt` |
| 10. Test | Nạp `best_model.pt`, dự báo từng ngày 2014–2017 | Khử chuẩn hóa về mm/ngày, bỏ ngày thiếu quan trắc, tính NSE và KGE cho từng lưu vực → `test_metrics.csv` |

### 11.2. Kiến trúc Sequential Forecast LSTM (mô hình tốt nhất) theo kích thước tensor

B là kích thước batch (256).

| Khối | Đầu vào | Đầu ra | Tham số |
|---|---|---|---|
| Nhúng hindcast | 364 ngày × 32 biến (31 tái phân tích + `qmean`) | 364 × 16 | Lớp tuyến tính 32 → 16 |
| Nhúng forecast | 1 ngày × 5 biến ECMWF | 1 × 16 | Lớp tuyến tính 5 → 16 |
| Nhúng thuộc tính | 33 thuộc tính | 16, lặp lại cho mọi bước và nối vào sau nhúng động → mỗi bước 32 chiều | Hai lớp tuyến tính 33 → 16 (một cho hindcast, một cho forecast) |
| LSTM pha hindcast | [364, B, 32] | Chuỗi đầu ra [B, 364, 128] và trạng thái (h, c) | LSTM 32 → 128, khởi tạo forget bias 3 |
| LSTM pha forecast | [1, B, 32], khởi tạo bằng (h, c) của pha hindcast | [B, 1, 128] | Cùng một LSTM với pha hindcast |
| Đầu ra | Nối hai pha [B, 365, 128] → dropout 0,3 | [B, 365, 1]; lấy bước cuối là qmax ngày t | Lớp tuyến tính 128 → 1 |

Tổng khoảng 85 nghìn tham số (LSTM chiếm 82.944). Với mạng nhúng phức tạp (30-20-64, tanh giữa các lớp), LSTM nhận 128 chiều mỗi bước, tổng khoảng 143 nghìn tham số.

**Encoder–Decoder LSTM (handoff):** LSTM hindcast 128 chạy 364 ngày → nối (h, c) thành 256 chiều → lớp tuyến tính 256 → 128 → lớp tuyến tính 128 → 256 → tách thành trạng thái ban đầu (h, c) của LSTM forecast 128 → chạy ngày t → lớp đầu ra riêng. **Baseline CUDA LSTM:** một LSTM chạy 365 ngày trên một nhóm biến duy nhất (dự báo, tái phân tích hoặc cả hai). **Học chuyển giao:** chạy mô hình mới 1 epoch để dựng cấu trúc, chép trọng số từ mô hình tái phân tích, rồi tinh chỉnh 20 epoch (chỉ mạng nhúng động hoặc toàn bộ).

### 11.3. Siêu tham số thực tế trong 24 cấu hình

| Nhóm cấu hình | Số chiều ẩn | Batch | Dropout đầu ra | Nhiễu nhãn |
|---|---|---|---|---|
| Sequential Forecast LSTM, Encoder–Decoder LSTM (mọi biến thể) | 128 | 256 | 0,3 | Không |
| Baseline chỉ dự báo | 128 | 256 | 0,3 | 0,1 |
| Baseline tái phân tích | 128 | 128 hoặc 256 | 0,2–0,3 | 0,001–0,1 |
| Baseline tái phân tích + dự báo | 128 hoặc 256 | 256 | 0,2–0,4 | 0,001 |
| CrossDomain | 256 | 256 | 0,3–0,4 | 0,01 ở pha huấn luyện |

Mọi cấu hình dùng hạt giống 111.

### 11.4. Các điểm mã khác với mô tả trong bài

- Mạng handoff của Encoder–Decoder LSTM gồm hai lớp tuyến tính liên tiếp không có hàm kích hoạt (lớp `FC` của NeuralHydrology không áp kích hoạt cho lớp cuối), trong khi bài mô tả là mạng handoff phi tuyến; tác giả xác nhận khi trả lời phản biện là "một lớp 128".
- Mạng nhúng "đơn giản" là một lớp tuyến tính, không có tanh như Mục 2.2.6 mô tả.
- Mọi cấu hình Sequential Forecast LSTM và Encoder–Decoder LSTM dùng cùng một bộ siêu tham số (128, 256, 0,3), trong khi các baseline có bộ khác nhau; bài không nêu rõ tối ưu Bayes được chạy riêng cho từng kiến trúc hay không.
- Quy tắc lọc mẫu ở bước 4 khiến mẫu huấn luyện bị loại khi thiếu `qmean` hoặc thiếu dữ liệu ECMWF ở bất kỳ ngày nào trong 365 ngày trước đó — dữ liệu ECMWF có từ năm 2002 (`Document/3-DuLieu/LamaHCE.md` Mục 6.1), nên năm 2003 không bị loại vì ECMWF; mẫu huấn luyện bị loại chủ yếu ở năm đầu của 27 trạm bắt đầu đo muộn và các ngày thiếu `qmean`.
