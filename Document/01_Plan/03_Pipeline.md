# Kiến trúc và pipeline của đề tài

> Tài liệu gốc của đề tài: câu hỏi nghiên cứu, đóng góp và tính mới, dữ liệu và bộ nạp, kiến trúc mô hình, giao thức huấn luyện và so sánh công bằng, bộ đánh giá, ma trận thí nghiệm theo bước, ngân sách tính toán, rủi ro, phạm vi tiểu luận và khóa luận. Bài cơ sở: BiasCast (Konold và cs., HESS 2026) — `Document/01_Plan/02_BasePaper.md`. Dữ liệu: `Document/03_Data/01_LamaHCE.md`. Ý tưởng và căn cứ: `Document/04_Ideas/01_Ideas.md`. Ghi chú từng tài liệu: `Document/05_Survey/03_PaperNotes.md`. Các mục ghi "chưa chốt" chờ quyết định. Mọi nhận định "chưa có công trình nào" dựa trên tra cứu tới ngày 7/10/2026 và được ghi kèm công trình gần nhất đã biết.

---

## 1. Câu hỏi nghiên cứu, đóng góp và tính mới

### 1.1. Câu hỏi nghiên cứu

| Mã | Câu hỏi | Trả lời bằng |
|---|---|---|
| RQ1 | Kết quả BiasCast có tái lập được không, nhiễu giữa các hạt giống lớn cỡ nào, và mô hình thật sự hơn các mốc không học (persistence) và mốc tuyến tính bao nhiêu? | Bước A |
| RQ2 | Các cải tiến quy trình không phụ thuộc kiến trúc (sửa điểm yếu trong mã, thống nhất đơn vị Q, giữ mẫu thiếu dữ liệu bằng masked mean theo nguồn, huấn luyện có che dữ liệu) cải thiện Sequential Forecast LSTM bao nhiêu và giúp mô hình chịu mất dữ liệu ra sao? | Bước B |
| RQ3 | Trong bài toán dự báo có dự báo thời tiết thật và Q quan trắc, khi cùng số tham số, cùng ngân sách tinh chỉnh, cùng hạt giống, lõi SSM (S4D, Mamba) có hơn, ngang hay kém LSTM, GRU, Transformer? Hơn ở dải lưu lượng nào, loại lưu vực nào? | Bước C, E; đánh giá Mục 6 |
| RQ4 | Khi dữ liệu đến trễ đúng như thực tế vận hành (mỗi nguồn một độ trễ) hoặc mất Q, lõi nào suy giảm ít hơn? | Đánh giá kịch bản vận hành (Mục 6.3) |
| RQ5 | Chuỗi quá khứ dài hơn (730 ngày) có giúp không, với chi phí tính toán bao nhiêu? | Bước D |
| RQ6 | Khi dự báo Q 1–7 ngày mà chỉ ngày đầu có dự báo thời tiết thật, kỹ năng của LSTM và lõi SSM suy giảm theo hạn ra sao, và dự báo thời tiết đóng góp bao nhiêu ở từng hạn? | Bước H |

### 1.2. Đóng góp dự kiến

| Mã | Đóng góp | Loại |
|---|---|---|
| C1 | So sánh có kiểm soát (cùng tham số, cùng ngân sách tinh chỉnh, 3 hạt giống, kiểm định thống kê) giữa LSTM, GRU, Transformer, S4D, Mamba trong khung hindcast–forecast dùng dự báo ECMWF HRES lưu trữ thật và Q quan trắc, trên 451 lưu vực Trung Âu | Thực nghiệm, tính mới chính |
| C2 | Bộ đánh giá vận hành: kịch bản độ trễ theo từng nguồn dữ liệu (ERA5-Land, E-OBS, MSWEP, GLEAM, Q) và mất Q, áp dụng cho mọi lõi | Phương pháp đánh giá |
| C3 | Tái lập BiasCast, đo nhiễu hạt giống, bổ sung mốc persistence, PNSE, mốc tuyến tính; phát hiện và sửa điểm yếu trong mã (bộ lọc mẫu loại cả mẫu khi thiếu một giá trị, hàm lưu mô hình tốt nhất); kiểm đơn vị Q đầu vào mà tác giả dùng | Tái lập, chuẩn hóa đánh giá |
| C4 | Đánh giá theo mức lưu lượng và theo sự kiện lũ (chu kỳ lặp lại), phân tích theo lưu vực kiểm giả thuyết "SSM có lợi ở lưu vực động lực chậm, bất lợi ở lũ nhanh" | Phân tích |
| C5 | Giải thích mô hình (Integrated Gradients, dò trạng thái ẩn) so sánh độ dài nhớ hiệu dụng của LSTM và Mamba; demo bản đồ cảnh báo 451 lưu vực với đường Q 7 ngày | Ứng dụng |
| C6 | Mở rộng BiasCast từ dự báo 1 ngày sang 1–7 ngày bằng dữ liệu sẵn có, so ba cách (chỉ quá khứ, lai, dự báo hoàn hảo) cho LSTM và lõi SSM | Thực nghiệm |

### 1.3. Tính mới so với công trình gần nhất

| Công trình gần nhất | Đã làm | Khác biệt của đề tài |
|---|---|---|
| BiasCast (Konold và cs., HESS 2026) | Khung hindcast–forecast với ECMWF HRES thật và Q quan trắc, chỉ LSTM, 1 hạt giống, không persistence | Thêm các lõi khác dưới giao thức công bằng; nhiều hạt giống; mốc persistence; sửa lỗi mã |
| Shahriar (preprint SSRN, 10/2026) | Mamba-3 và LSTM cùng tham số, cùng độ sâu, 671 lưu vực CAMELS; LSTM thắng mọi cặp (NSE 0,752 so với 0,727); Mamba quá khớp sau khoảng 10 epoch | Bài đó là mô phỏng (không dự báo thời tiết, không Q quá khứ); đề tài là dự báo vận hành với dự báo thật, Q quan trắc, đánh giá theo mức và kịch bản trễ |
| Zhang và cs. (J. Hydrol. 2026) | LSTM, Transformer, PatchTST, Mamba, DLinear theo giờ, CAMELS; Mamba gần nhưng hơi kém LSTM | Đề tài dùng dự báo thời tiết lưu trữ thật, dữ liệu Trung Âu, thêm S4D, kiểm định thống kê theo lưu vực, kịch bản vận hành |
| Demiray & Demir (preprint EarthArXiv) | Mamba so với LSTM, GRU, Transformer, 125 lưu vực Iowa, theo giờ, có SHAP | Khác bài toán (không ECMWF lưu trữ, không giao thức cùng tham số); đề tài thêm S4D, kịch bản vận hành, đánh giá theo mức |
| RiverMamba (NeurIPS 2025) | Mamba quét không gian trên lưới 0,05°, nhãn là tái phân tích GloFAS | Đề tài theo lưu vực, nhãn là Q quan trắc, so sánh có kiểm soát với các lõi khác |
| Gauch và cs. (HESS 2025); MF²LSTM (preprint EGUsphere 2026) | Masked mean cho dữ liệu khí tượng thiếu (mô phỏng, CAMELS); MF²LSTM che Q để đồng hóa, theo giờ, một mô hình LSTM | Đề tài áp dụng cho bài toán ngày có dự báo thật, cho mọi lõi, nhóm theo nguồn dữ liệu, và kiểm tra bằng độ trễ thực tế của từng nguồn |
| Dubey và cs. (arXiv 2025) | Kịch bản trễ và mất dữ liệu cho LSTM mã hóa – giải mã, CAMELS-US, HYSETS, CAMELS-IND, có HRES | Đề tài: độ trễ riêng của từng nguồn (ERA5-Land 5 ngày, GLEAM tới hơn 1 năm…), Q quan trắc, so sánh nhiều lõi |
| HydroDiffusion (Wang và cs., Water Resources Research 2026; số liệu theo bản arXiv 2512.12183) | Mô hình khuếch tán xác suất với lõi SSM (S4D-FT) so với lõi LSTM, dự báo 0–7 ngày bằng dự báo tái lập GEFSv12, 531 lưu vực CAMELS, không dùng Q quan trắc làm đầu vào; SSM nhỉnh hơn LSTM (NSE ngày 0 trung vị 0,75 so với 0,73) | Đề tài dùng Mamba (SSM chọn lọc) bên cạnh S4D, dự báo ECMWF HRES lưu trữ, Q quan trắc, Trung Âu; so sánh 5 lõi cùng tham số có kiểm định; kịch bản vận hành |
| Saint-Fleur và cs. (HESS 30:3497–3527, 2026) | Ba cách đưa Q quan trắc vào dự báo 1–7 ngày (MLP điều phối, LSTM của Kratzert 2019, SAC-SMA), CAMELS-US, CAMELS-FR, có lưu trữ dự báo ECMWF; với LSTM, Q chủ yếu giúp ở lưu vực LSTM vốn kém | Chỉ LSTM; đề tài so nhiều lõi và đo mức phụ thuộc vào Q qua kịch bản mất Q |

Cách nói trước hội đồng: đề tài không khẳng định Mamba tốt hơn; đóng góp là trả lời có kiểm soát câu hỏi "lõi SSM có đáng thay LSTM trong dự báo vận hành không, ở đâu và với điều kiện nào", kèm bộ đánh giá sát vận hành. Kết quả Mamba kém LSTM vẫn là kết quả có giá trị (cùng chiều với Shahriar 2026, Zhang 2026) nếu giao thức chặt.

### 1.4. Nguyên tắc thiết kế

1. **So sánh công bằng:** cùng dữ liệu, chia tập, hàm mất mát, bộ nạp, khung hindcast–forecast, ngân sách tinh chỉnh, số hạt giống; chỉ khác lõi thời gian. Báo cáo kết quả cả ở mức cùng tham số và ở cấu hình tốt nhất sau tinh chỉnh.
2. **Mốc mạnh trước khi so:** tầng 1 cải tiến LSTM trước, rồi mọi lõi dùng chung quy trình đã cải tiến.
3. **Thay đổi từng thứ một:** mỗi cải tiến là một thí nghiệm riêng so với cấu hình liền trước.
4. **Chống rò rỉ:** checkpoint chọn theo validation; NSE test mỗi epoch chỉ để theo dõi; ngưỡng mức lưu lượng tính trên dữ liệu trước kỳ test.
5. **Kiểm thử trước khi huấn luyện:** mọi mô-đun mới có kiểm thử đơn vị (Mục 9.2).
6. **Kết quả nào cũng là câu trả lời:** báo cáo cả kết quả âm tính.

---

## 2. Tổng quan các giai đoạn

| Giai đoạn | Việc chính | Đầu ra | Trạng thái |
|---|---|---|---|
| 0. Hạ tầng | Fork NeuralHydrology riêng, cài `mamba-ssm` trên Kaggle T4, kiểm thử đơn vị, đo thời gian | Môi trường chạy được mọi lõi; bảng thời gian mỗi epoch | Đã lấy 3 bản sửa từ thư viện gốc vào nhánh `research` (commit `dda45fc`); đo thời gian ở đầu A4; `mamba-ssm` và kiểm thử đơn vị ở bước C |
| 1. Thu thập dữ liệu | Tải Extended LamaH-CE và kết quả của tác giả | Kaggle Dataset `lamah-ce-ext` | Xong (`01_LamaHCE.md` Mục 6) |
| 2. Khảo sát dữ liệu | Theo khuôn GVHD: dạng cơ bản + thống kê sâu; ngưỡng mức lưu lượng; bảng độ trễ nguồn | Báo cáo khảo sát, tệp ngưỡng | Một phần số liệu đã có; notebook viết lại |
| 3. Tái lập và mốc | Chạy lại trọng số tác giả; persistence; DLinear; nhiễu hạt giống | Bảng mốc | Persistence xong (NSE trung vị 0,35–0,37, `01_LamaHCE.md` Mục 6.1); chạy lại trọng số tác giả (A2) khớp NSE trung vị cả 6 cấu hình (`01_LamaHCE.md` Mục 6.3) |
| 4. Bộ nạp | Sửa bộ lọc mẫu, nhóm biến theo nguồn, Q cùng đơn vị | Cấu hình dữ liệu dùng chung | Chưa làm |
| 5. Mô hình | Khung hindcast–forecast với lõi thay được | Mô-đun trong fork | Chưa làm |
| 6. Huấn luyện | Bước B → H (Mục 7) | Trọng số, nhật ký | Chưa làm |
| 7. Đánh giá | Bộ đánh giá Mục 6 | Bảng, hình, kiểm định | Chưa làm |
| 8. Giải thích | Integrated Gradients, dò trạng thái ẩn | Hình mức ảnh hưởng | Chưa làm |
| 9. Ngưỡng lũ | Gumbel L-moments trên qmax năm | Ngưỡng 1, 2, 5, 10 năm mỗi trạm | Chưa làm |
| 10. Demo | Một mục trong trang web nhiều mục của nhóm, chạy liên tục trên VPS Oracle Cloud miễn phí: bản đồ cảnh báo phát lại kỳ test, đường Q 7 ngày, mức ảnh hưởng của các nhóm nguồn theo lưu vực; mô hình chạy thật trên CPU cho lưu vực và ngày được chọn, có thử "nếu… thì…" với dự báo ECMWF HRES ngày t; huấn luyện trên GPU, demo chạy CPU, làm trên laptop trước rồi đưa lên VPS (Mục 9.6) | Ứng dụng | Chưa làm |

---

## 3. Dữ liệu, bộ nạp và tiền xử lý

### 3.1. Giữ nguyên như bài gốc

- 451 lưu vực mức A (`basins_filtered.txt`), train 2003–2009, validation 2010–2013, test 2014–2017; NeuralHydrology nạp thêm giai đoạn khởi động trước mỗi kỳ (thư trả lời phản biện 1 của BiasCast), dài 364 ngày = `seq_length` − `predict_last_n` (`basedataset.py`).
- Nhãn `qmax` ngày t, bộ nạp đổi sang mm/ngày theo diện tích (`area_gov`); demo đổi ngược về m³/s.
- Hindcast 364 ngày (31 biến tái phân tích), forecast ngày t (5 biến ECMWF HRES), 33 thuộc tính tĩnh.
- z-score toàn cục tính trên train; hàm mất mát NSE* dùng độ lệch chuẩn qmax từng lưu vực.

### 3.2. Nhóm biến theo nguồn dữ liệu

Hindcast chia thành các nhóm theo nguồn, mỗi nhóm một mạng nhúng; masked mean lấy trung bình các nhóm có dữ liệu ở mỗi bước (Gauch và cs., HESS 2025). Nhóm theo nguồn vì dữ liệu mất khi vận hành theo cả nguồn (một nhà cung cấp chậm), không theo từng biến.

| Nhóm | Biến | Độ trễ khi vận hành (nguồn tra) | Ghi chú |
|---|---|---|---|
| ERA5-Land | 21 biến | Khoảng 5 ngày (Copernicus, trang Climate reanalysis; `04_Ideas/01_Ideas.md` Mục 1.1) | |
| E-OBS | 7 biến | Bản cập nhật hằng ngày trễ 1 ngày; bản tháng tạm thời; bản đầy đủ 2 lần mỗi năm (Copernicus, trang *Daily vs Full E-OBS updates*) | Bản hằng ngày dùng mạng trạm khác và kiểm tra chất lượng đơn giản hơn bản đầy đủ mà dữ liệu dùng — lệch miền không mô phỏng được với dữ liệu hiện có |
| MSWEP | Mưa | Bản NRT trễ khoảng 2–3 giờ (UCAR Climate Data Guide, trang MSWEP; GloH2O) | Bản NRT khác bản lịch sử có hiệu chỉnh trạm — lệch miền không mô phỏng được với dữ liệu hiện có |
| GLEAM | Bốc hơi thực, tiềm năng | Phát hành mỗi năm một lần, kéo dài tới hết năm trước (ICDC Hamburg: bản 3/2021 thêm 2019–2020, bản 8/2024 thêm 2023; GLEAM4 — Miralles và cs., Scientific Data 2025) | Khi vận hành có thể thiếu từ vài tháng tới hơn một năm gần nhất |
| Q quan trắc | `qmean`, `qmax` quá khứ (cùng đơn vị mm/ngày với nhãn) | eHYD (Áo) khoảng 2 giờ (BiasCast Mục 3.7) | Nhóm riêng như MF²LSTM |
| Forecast | 5 biến ECMWF HRES ngày t | Phát hành 00 UTC ngày t (BiasCast Mục 2.1) | Mạng nhúng riêng cho pha forecast |

### 3.3. Sửa bộ nạp

| Việc | Lý do | Cách làm |
|---|---|---|
| Sửa `_validate_samples` để giữ mẫu có đầu vào thiếu khi dùng masked mean | Bản gốc loại mẫu nếu bất kỳ cột nào thiếu ở bất kỳ ngày nào trong 365 ngày (`basedataset.py` dòng 861–919), kể cả khi bật `nan_handling_method` | Khi có `nan_handling_method`, chỉ yêu cầu nhãn và thuộc tính tĩnh hợp lệ; vẫn loại mẫu không đủ lịch sử |
| Dựng thư mục có Q (thay `LamaH_expanded_q_input` không công bố) | Biến thể có Q của bài đọc thư mục này | Bộ nạp đọc `qmean` thẳng từ tệp khí tượng (`datasetzoo/lamah.py` dòng 107–135; chỉ nhãn `qmax` được đọc từ `D_gauges` và đổi sang mm/ngày), nên chép `qmean` từ `D_gauges` vào tệp khí tượng mức A (thêm `qmax` cho B1). Tác giả dùng `qmean` theo m³/s, ngày thiếu để trống: thống kê chuẩn hóa tính lại từ `D_gauges` khớp tệp chuẩn hóa của tác giả tới chữ số thứ tư, đối chứng `qmax` khớp ở mm/ngày (A2 Phần 3, `03_Data/01_LamaHCE.md` Mục 6.3); A2 và A4 dựng đúng như vậy, B1 đổi sang mm/ngày theo `area_gov` như nhãn |
| Báo cáo số mẫu hợp lệ từng cấu hình | Biết cải tiến lấy lại bao nhiêu mẫu | Ghi số mẫu train trước và sau sửa bộ lọc, với hindcast 365 và 730 ngày |

### 3.4. Che dữ liệu khi huấn luyện

Bản fork đã có `nan_step_probability`, `nan_sequence_probability` (`basedataset.py` dòng 180, 207–229): khi huấn luyện, mỗi nhóm biến bị đặt NaN ngẫu nhiên từng bước và cả chuỗi, không bao giờ che hết mọi nhóm. Xác suất dùng chung cho mọi nhóm; muốn xác suất riêng cho nhóm Q phải sửa nhỏ. Hai giá trị cần thử: 0,1/0,12 (Gauch và cs. 2025, che khí tượng) và 0,05/0,05 (MF²LSTM, che Q).

---

## 4. Kiến trúc mô hình

### 4.1. Khung chung

| Khối | Đầu vào → đầu ra | Ghi chú |
|---|---|---|
| Nhúng hindcast | Mỗi nhóm nguồn → d chiều; masked mean qua các nhóm; nối nhúng thuộc tính tĩnh | Lớp tuyến tính như cấu hình tốt nhất của bài (`02_BasePaper.md` Mục 11; 16 chiều + 16 chiều tĩnh) |
| Nhúng forecast | 5 biến ECMWF → d chiều; nối nhúng tĩnh | Cùng số chiều với hindcast |
| Lõi thời gian | Chuỗi 365 bước (364 hindcast + 1 forecast) → biểu diễn bước cuối | Thay được: LSTM, GRU, Transformer, S4D, Mamba |
| Đầu ra | Bước cuối → qmax ngày t | Hồi quy hoặc CMAL (bước F) |

Đề tài viết một mô hình chung "Sequential Forecast" nhận lõi bất kỳ: nối nhúng hindcast và forecast theo thời gian rồi chạy lõi một lần. Với lõi LSTM, cách này tương đương đúng Sequential Forecast LSTM của bài (LSTM chạy hết hindcast rồi tiếp tục trạng thái sang forecast); kiểm bằng cách nạp trọng số của tác giả và so đầu ra (Mục 9.2). Bài cũng cho thấy trạng thái chạy liên tục bền hơn mạng handoff (435 so với 361 lưu vực cải thiện), nên không dùng Encoder–Decoder cho các lõi mới.

### 4.2. Các lõi thời gian

| Lõi | Cài đặt | Vai trò |
|---|---|---|
| LSTM | `nn.LSTM`, forget bias 3 như bài (`02_BasePaper.md` Mục 11.2) | Mốc chính |
| GRU | `nn.GRU` | Mốc hồi quy nhẹ hơn |
| Transformer | Bộ mã hóa pre-LayerNorm, mặt nạ nhân quả, mã hóa vị trí sin, lấy đầu ra bước cuối; khởi điểm từ `transformer.py` của NeuralHydrology | Mốc attention; Liu và cs. (2024) cho thấy bản cơ bản thua LSTM |
| S4D | Khối S4D (Gu và cs., NeurIPS 2022) có kết nối tắt và LayerNorm; siêu tham số khởi điểm theo S4D-FT (d_state, min_dt 0,01, max_dt 0,1, lr riêng cho tham số SSM) | Đối chứng SSM bất biến thời gian — tách "Mamba không hợp" khỏi "SSM không hợp" |
| Mamba | Khối `mamba_ssm.Mamba` có kết nối tắt và LayerNorm, đầu vào dạng (batch, length, dim); dự phòng `mambapy` (PyTorch thuần) nếu không cài được kernel CUDA | Lõi chính của đề tài |
| DLinear | Đúng bản gốc (Zeng và cs., AAAI 2023; mã `cure-lab/LTSF-Linear`, `models/DLinear.py`): mỗi biến tự dự báo từ quá khứ của chính nó, không trộn biến, nên không dùng được thời tiết. Đầu vào `qmean` 365 ngày t − 365 … t − 1 (dịch lùi 1 ngày bằng `lagged_features` để không rò rỉ `qmean` ngày t), trung bình trượt 25 ngày tách xu hướng – dao động, mỗi phần một lớp tuyến tính 365 → 1, cộng lại ra `qmax` ngày t | Mốc tuyến tính trên lưu lượng quá khứ (giữa persistence và học sâu), không phải lõi so sánh |

Lớp `mamba.py` có sẵn của NeuralHydrology quét sai trục (`02_BasePaper.md` Mục 11.5) nên không dùng.

### 4.3. Biến thể cải tiến (bước E)

| Biến thể | Mô tả | Căn cứ | So với |
|---|---|---|---|
| Mamba hai chiều | Thêm nhánh quét ngược trên toàn cửa sổ 365 bước rồi gộp với nhánh xuôi; đầu ra chỉ ở bước cuối, mọi đầu vào đều ≤ ngày t nên không rò rỉ | ResBi-Mamba Plus (AISE 2026); S-Mamba, CMMamba dùng Mamba hai chiều | Mamba thuần cùng tham số |
| Transformer patch | Chia hindcast thành đoạn (patch) làm token như PatchTST | Nie và cs. (ICLR 2023); Zhang và cs. (2026): khi có Q quá khứ, PatchTST tốt nhất ở hạn từ 3 giờ | Transformer thuần cùng tham số |
| Dự phòng: Transformer nhúng tích chập nhân quả | Hai lớp Conv1D nhân quả có kết nối tắt trước attention | Liu và cs. (J. Hydrol. 2024) | Transformer thuần |
| Dự phòng: S5D | Thêm Conv1D, LayerNorm, Softsign vào S4D | Jing và cs. (ESWA 2026) | S4D thuần |

### 4.4. Dự báo nhiều ngày (bước H)

Mô hình chung dự báo qmax ngày t đến t+6: tham số `forecast_seq_length` có sẵn trong thư viện (`basedataset.py`); hàm mất mát NSE* lấy trung bình trên 7 hạn là lựa chọn của đề tài. Dữ liệu chỉ có dự báo ECMWF HRES cho ngày t, nên so ba cách cấp đầu vào cho pha dự báo:

| Cách | Đầu vào pha dự báo | Vai trò |
|---|---|---|
| (a) Chỉ quá khứ | Không có nhóm dự báo | Mốc vận hành không có dự báo thời tiết |
| (b) Lai | Ngày t: 5 biến ECMWF HRES thật; ngày t+1 đến t+6: để trống, xử lý bằng masked mean (nhóm dự báo bị che) | Cách vận hành được với dữ liệu sẵn có |
| (c) Dự báo hoàn hảo | 5 biến tái phân tích tương ứng (như CrossDomain) cho cả 7 ngày | Cận trên, không vận hành được |

Bộ nạp phải bảo đảm cách (b) không đưa dự báo hạn 1 ngày của các ngày sau vào mô hình (tránh dùng thông tin tốt hơn thực tế); kiểm bằng kiểm thử đơn vị. Ở những bước pha dự báo không có nhóm nào (cách a; cách b từ ngày t+1), masked mean của thư viện điền một giá trị cố định thay vì NaN (`modelzoo/inputlayer.py`, `_masked_mean_embedding`); bật `timestep_counter` có sẵn (`datasetzoo/basedataset.py`) để mô hình biết đang ở hạn thứ mấy.

### 4.5. Đầu ra xác suất (bước F)

Đầu CMAL (có sẵn `head: cmal`), hàm mất mát log-likelihood; dự báo điểm lấy trung vị phân phối; xác suất vượt ngưỡng lũ cho demo. Căn cứ: Klotz và cs. (HESS 2022) — CMAL tốt nhất trong bốn cách.

---

## 5. Huấn luyện và giao thức so sánh

### 5.1. Cấu hình chung (giữ như bài)

Theo cấu hình của tác giả (BiasCast Mục 2.2; `02_BasePaper.md` Mục 11): NSE*, Adam lr 10⁻³, CosineAnnealing 30 epoch (nhỏ nhất 10⁻⁵), batch 256, cắt gradient 1, dropout đầu ra 0,3, dừng sớm theo trung vị NSE validation (patience 5, cải thiện tối thiểu 0,005, đã sửa lỗi lưu mô hình tốt nhất). Mỗi epoch ghi NSE trung vị validation và test (yêu cầu GVHD); checkpoint chọn theo validation.

### 5.2. Giao thức so sánh lõi

| Bước | Cách làm |
|---|---|
| Mức cùng tham số | Mỗi lõi chỉnh số chiều để tổng tham số lệch không quá 1% so với LSTM cải tiến (khoảng 85 nghìn tham số với 128 chiều) — tránh nhiễu do khác dung lượng; Shahriar (2026) giữ chênh lệch dưới 0,5%; ngưỡng 1% là lựa chọn của đề tài vì số chiều chỉnh theo bước nguyên |
| Tinh chỉnh cùng ngân sách | Lựa chọn của đề tài để mỗi lõi có cùng ngân sách (bài gốc tối ưu Bayes 100 lần cho LSTM — Phụ lục D, G; không khả thi cho 5 lõi): mỗi lõi 8 cấu hình, 1 hạt giống, chọn theo validation: 2 mức dung lượng (cùng tham số, gấp khoảng 3 lần) × 2 độ sâu (1, 2 lớp) × 2 learning rate (10⁻³, 5×10⁻⁴) |
| Chạy chính thức | Cấu hình cùng tham số và cấu hình tốt nhất của mỗi lõi chạy 3 hạt giống |
| Báo cáo | Từng hạt giống, trung bình ± độ lệch chuẩn, tổ hợp (trung bình dự báo 3 hạt giống); số tham số, thời gian mỗi epoch, bộ nhớ GPU |

### 5.3. Hạt giống và tổ hợp

3 hạt giống cho mọi cấu hình chính thức. Nhiễu giữa hạt giống đo ở bước A dùng làm thước đo: chênh lệch nhỏ hơn nhiễu này không được kết luận.

---

## 6. Đánh giá

### 6.1. Chỉ số chính và kiểm định

- **Theo lưu vực trên test 2014–2017:** NSE, KGE (và ba thành phần r, α, β), **PNSE** (NSE lấy qmax ngày t−1 quan sát làm mốc, tức hệ số persistence của Kitanidis & Bras, WRR 1980; MF²LSTM dùng tên PNSE).
- **Tổng hợp:** trung vị, trung bình, P10/P25/P75/P90, đường CDF, số lưu vực tốt lên / kém đi.
- **Kiểm định:** Wilcoxon signed-rank hai phía, ghép cặp theo lưu vực, kèm cỡ hiệu ứng Cohen's d dạng ghép cặp (trung bình chênh lệch chia độ lệch chuẩn của chênh lệch) — cách của Kratzert và cs. (2019, 2024), Gauch và cs. (2025), Nearing và cs. (2024); hiệu chỉnh Holm khi so nhiều cặp.

### 6.2. Theo mức lưu lượng và theo sự kiện (yêu cầu GVHD)

| Bước | Nội dung |
|---|---|
| Ngưỡng mức | Riêng từng lưu vực, tính trên qmax 1981–2013 (trước kỳ test): bốn mức theo xác suất vượt của đường duy trì lưu lượng — rất cao (0–0,02), cao (0,02–0,2), trung bình (0,2–0,7), thấp (0,7–1). Yilmaz, Gupta, Wagener (WRR 2008) chia ba đoạn: cao 0–0,02, giữa 0,2–0,7, thấp 0,7–1; khoảng 0,02–0,2 nằm giữa hai đoạn của Yilmaz, đề tài tách thành mức riêng để bốn mức phủ hết đường duy trì |
| Chỉ số theo mức | Ma trận nhầm lẫn 4 mức, độ chính xác, recall từng mức; %BiasFHV, %BiasFMS, %BiasFLV |
| Ngưỡng lũ | Chu kỳ lặp lại 1, 2, 5, 10 năm (như Nearing và cs., Nature 2024), Gumbel fit bằng L-moments (Hosking, JRSS B 1990) trên qmax năm 1981–2013, như AIFL. Số năm có Q khác nhau giữa các trạm (3 trạm chỉ có Q từ 2010 — `01_LamaHCE.md` Mục 6.1), nên ghi số năm dùng để fit cho từng lưu vực; lưu vực dưới 10 năm báo cáo riêng (ngưỡng 10 năm là lựa chọn của đề tài) |
| Sự kiện | Một sự kiện tính là phát hiện đúng nếu dự báo và quan trắc cùng vượt ngưỡng trong cửa sổ ±1 ngày — lựa chọn của đề tài cho hạn 1 ngày; Nearing và cs. (2024) tính đúng khi hai bên vượt ngưỡng "within two days of each other" với dự báo nhiều ngày; bước H dùng ±2 ngày như Nearing |
| Chỉ số sự kiện | Tỉ lệ phát hiện (POD/recall), tỉ lệ báo động nhầm (FAR), precision, F1, CSI; sai số thời điểm đỉnh |
| Ngưỡng kép (phân tích độ nhạy) | Ngưỡng của mô hình tính trên chính chuỗi dự báo của mô hình (Nearing 2024, AIFL); vì mô hình chỉ có dự báo 2003–2017 (gồm cả kỳ train), chỉ dùng làm phân tích phụ, nêu rõ hạn chế |
| Cách đọc | Ưu tiên bỏ sót ít ở mức lũ và mức rất thấp, chấp nhận báo động nhầm nhiều hơn (GVHD); so các lõi ở từng mức |

### 6.3. Kịch bản vận hành (chỉ suy luận)

| Kịch bản | Mô tả |
|---|---|
| S0 | Đủ dữ liệu (như bài) |
| S1 | Mất Q 1, 3, 7 ngày cuối |
| S2 | Mất Q toàn cửa sổ (trạm hỏng dài hạn) |
| S3 | Độ trễ thực tế (Mục 3.2): ERA5-Land thiếu 5 ngày cuối; GLEAM thiếu toàn cửa sổ (giả định xấu nhất); E-OBS, MSWEP đủ tới t−1 (giả định bản hằng ngày, bản NRT tương đương bản lịch sử); Q đủ tới t−1 |
| S4 | S3 + mất Q 1 ngày cuối |

Các kịch bản chạy trên cả kỳ validation (dùng để chọn cấu hình ở bước B*) và kỳ test (chỉ để báo cáo). Mô hình không có masked mean không chạy được với giá trị thiếu: dùng cách điền đơn giản (giá trị gần nhất, khí hậu ngày) làm mốc so sánh. Báo cáo NSE theo từng kịch bản và mức suy giảm so với S0 cho mỗi lõi.

### 6.4. Theo lưu vực

ΔNSE giữa Mamba (và S4D) với LSTM cải tiến; bản đồ; tương quan Spearman với 33 thuộc tính và khoảng cách Wasserstein tái phân tích – dự báo (mã phân tích của tác giả). Kiểm giả thuyết từ Wang và cs. (WRR 2025), Jing và cs. (ESWA 2026), Zhang và cs. (2026): SSM có lợi ở lưu vực tuyết, động lực chậm, bất lợi ở lũ nhanh — dùng tỷ lệ tuyết, độ cao, chỉ số khô hạn, diện tích. Tách nhóm đầu nguồn (72,5%) và lồng nhau (27,5%) theo số liệu của BiasCast (thư trả lời phản biện 1, `PeerReview/ResponseToReferee1.pdf`).

### 6.5. Chi phí tính toán

Số tham số, thời gian mỗi epoch, bộ nhớ GPU, thời gian suy luận cho 365 và 730 ngày — kiểm lập luận "Mamba tuyến tính theo độ dài chuỗi" trên bài toán thật.

### 6.6. Theo hạn dự báo (bước H)

Với từng hạn 1–7 ngày: NSE, KGE; PNSE lấy qmax ngày t−1 quan sát làm mốc cho mọi hạn (persistence cùng hạn); chỉ số sự kiện lũ với cửa sổ ±2 ngày như Nearing và cs. (2024); so ba cách (a), (b), (c) để tách phần đóng góp của dự báo thời tiết thật ở ngày đầu và mức suy giảm khi không có dự báo.

### 6.7. Giải thích mô hình

Dò trạng thái ẩn (ý tưởng Y4): hồi quy tuyến tính từ trạng thái của LSTM và Mamba sang lượng nước tuyết `ERA5L_swe` và ẩm đất, chỉ cần suy luận (cách của Hu và cs., Environmental Modelling & Software 2025, cho LSTM). Tuyết và ẩm đất cũng là biến đầu vào nên kết quả dò một phần là hiển nhiên; báo cáo nêu rõ hạn chế này. Integrated Gradients (Captum) cho LSTM cải tiến và Mamba tốt nhất: mức ảnh hưởng theo ngày trong cửa sổ (độ dài nhớ hiệu dụng của từng lõi) và theo nhóm nguồn (tái phân tích, dự báo ECMWF, Q). Khác Demiray & Demir (SHAP, Iowa): tách đóng góp của dự báo thời tiết thật và Q quan trắc, so sánh độ dài nhớ giữa hai lõi.

---

## 7. Ma trận thí nghiệm theo bước

Số lần huấn luyện ghi dạng "cấu hình × hạt giống". Mỗi bước có điều kiện chuyển bước.

| Bước | Thí nghiệm | Số lần | Điều kiện chuyển bước |
|---|---|---|---|
| 0 | Hạ tầng: fork riêng, wheel `mamba-ssm`, kiểm thử đơn vị (Mục 9.2), đo thời gian 1 epoch mỗi lõi | 0 (chạy thử ngắn) | Mọi kiểm thử qua; có bảng thời gian |
| A1 | Khảo sát dữ liệu; ngưỡng mức và ngưỡng lũ tính trên 1981–2013 | 0 | Có tệp ngưỡng |
| A2 | Kiểm đơn vị `qmean` và dựng thư mục Q (Mục 3.3); chạy lại trọng số tác giả trên kỳ test cho 6 cấu hình: Sequential Forecast LSTM và Encoder–Decoder LSTM, mỗi loại có và không có lưu lượng quá khứ, mốc dưới 𝒟FC, mốc trên 𝒟RA (đúng sáu run của Hình 7, mục 3.4 của bài, bao trùm 3 kiến trúc, có/không Q, hai loại nhúng, hai nguồn dữ liệu; lý do chọn ở `03_Data/01_LamaHCE.md` Mục 6.3); lưu dự báo cho A3 | 0 | NSE trung vị lệch số của tác giả không quá 0,005 |
| A3 | Mốc so sánh, sau A1: persistence (tính lại), PNSE, KGE (r, α, β) cho 6 cấu hình của A2 và persistence, tính trên dự báo đã lưu ở A2; DLinear bản gốc trên `qmean` quá khứ (Mục 4.2), cách chuẩn hóa `qmean` (chung mọi lưu vực hay riêng từng lưu vực) và xử lý ô trống trong 365 ngày quá khứ quyết theo kết quả khảo sát của A1 | 1 × 3 | |
| A4 | Đo thời gian 1 epoch trên T4 (phần đo thời gian của bước 0) rồi huấn luyện lại Sequential Forecast LSTM có lưu lượng quá khứ theo cấu hình tác giả, giữ nguyên cả lỗi lưu mô hình tốt nhất (sửa ở B1), hạt giống 111 (của tác giả), 222, 333 — đo nhiễu hạt giống (B0) | 1 × 3 | Có độ lệch chuẩn giữa hạt giống; NSE trung vị gần 0,705 của bài |
| B1 | B0 + sửa lỗi lưu mô hình tốt nhất + Q cùng đơn vị mm/ngày với nhãn (nếu A2 xác nhận tác giả dùng đơn vị khác) + thêm `qmax` quá khứ vào nhóm Q | 1 × 3 | |
| B2 | B1 + nhúng theo nguồn, masked mean + sửa bộ lọc mẫu | 1 × 3 | |
| B3 | B2 + che dữ liệu khi huấn luyện, hai mức xác suất (0,1/0,12 và 0,05/0,05) | 2 × 3 | |
| B* | Chọn "LSTM cải tiến" bằng luật cố định, chỉ dùng kỳ validation: trong các cấu hình B1–B3 có NSE trung vị S0 kém cấu hình tốt nhất không quá độ lệch chuẩn giữa hạt giống của NSE trung vị validation (đo ở A4), chọn cấu hình có NSE trung vị trung bình qua S1–S4 cao nhất; hòa thì chọn cấu hình đơn giản hơn. Kết quả trên test của mọi cấu hình B vẫn báo cáo đủ | 0 | Luật chọn ghi trước khi chạy để tránh chọn theo test |
| C1 | Tinh chỉnh 5 lõi (LSTM, GRU, Transformer, S4D, Mamba) trên quy trình B* | 5 × 8 × 1 | |
| C2 | Chạy chính thức: cấu hình cùng tham số và cấu hình tốt nhất của mỗi lõi | tối đa 5 × 2 × 3 | |
| D | Hindcast 730 ngày cho LSTM, Mamba, S4D (cấu hình tốt nhất) | 3 × 3 | |
| E | Mamba hai chiều; Transformer patch (dự phòng: Transformer tích chập, S5D) | 2 × 3 (+2 × 3) | Chỉ làm nếu C xong |
| F | CMAL cho LSTM cải tiến và lõi SSM tốt nhất | 2 × 3 | |
| H | Dự báo 1–7 ngày cho LSTM cải tiến và lõi SSM tốt nhất: cách (a), (b), (c) | 2 × 3 × 3 | Làm ngay sau C; ưu tiên hơn E |
| GV | Thí nghiệm kiểm chứng gợi ý GVHD: lấy mẫu nhiều ngày mức cao (Y7) trên mô hình tốt nhất, đánh giá bằng Mục 6.2 | 1 × 3 | Bằng chứng trái chiều (Martel 2025) nên trình bày là kiểm chứng |
| Đánh giá | Mục 6.1–6.6 cho mọi cấu hình chính thức | 0 | |
| G | Giải thích mô hình, ngưỡng lũ, demo | 0 | |

Tổng tối đa 130 lần huấn luyện (A: 6; B: 12; C: 40 + 30; D, E, F, H, GV: 9 + 6 + 6 + 18 + 3), ít hơn nếu cấu hình tốt nhất của một lõi trùng cấu hình cùng tham số, thêm 6 nếu làm biến thể dự phòng.

---

## 8. Ngân sách tính toán, rủi ro và thứ tự cắt

### 8.1. Ngân sách

- Từ 3 tài khoản Kaggle trở lên, mỗi tài khoản 30 giờ GPU/tuần, tối đa 12 giờ/phiên (hạn mức Kaggle hiển thị trong tài khoản) — khoảng 90 giờ/tuần. Chỉ dùng GPU T4: tệp `setup.py` của `state-spaces/mamba` và `causal-conv1d` chỉ biên dịch nhân CUDA cho kiến trúc từ sm_75 trở lên (T4 là sm_75, P100 là sm_60).
- Thời gian mỗi lần trên T4 chưa đo (bài: vài phút tới khoảng 1 giờ trên RTX 4090). Đo thật ở đầu bước A4 rồi lập lịch; ví dụ trung bình 45 phút/lần thì 130 lần cần khoảng 98 giờ GPU, tức hơn một tuần với 3 tài khoản (khoảng 90 giờ/tuần).
- Mỗi tài khoản chạy một nhóm thí nghiệm; trọng số và nhật ký lưu thành Kaggle Dataset để đánh giá tập trung.

### 8.2. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Không cài được `mamba-ssm` trên Kaggle | Có gói wheel dựng sẵn cho Kaggle chia sẻ công khai (Kaggle Dataset `nctuan/mamba-ssm`: `mamba_ssm` 2.3.2, `causal_conv1d` 1.6.2, Python 3.12) và cách cài trong issue #668 của `state-spaces/mamba`; wheel phải khớp phiên bản torch, CUDA của Kaggle nên kiểm ở bước 0, hoặc tự build một lần rồi lưu Kaggle Dataset; dự phòng `mambapy` (PyTorch thuần, quét song song, chậm hơn); môi trường Kaggle đo ngày 11/10/2026 là Python 3.13.15, torch 2.11.0+cu128, NumPy 2.1.3 (A2 Phần 1), nên wheel dựng cho Python 3.12 có thể không cài được — kiểm ở bước C |
| Mamba quá khớp sớm (như Shahriar 2026) | Dừng sớm theo validation, dropout, theo dõi NSE validation và test mỗi epoch |
| Không đạt chênh lệch tham số ≤1% vì số chiều chỉ chỉnh theo bước nguyên (mô hình nhỏ, khoảng 85 nghìn tham số) | Đo ở bước 0; nếu không đạt, chỉnh thêm số chiều lớp nhúng hoặc đầu ra, ghi rõ chênh lệch thực tế |
| Lõi mới học kém do siêu tham số của LSTM | Tinh chỉnh cùng ngân sách có learning rate (Mục 5.2) |
| Masked mean, che dữ liệu làm giảm NSE ở S0 | Vẫn báo cáo; chọn B* theo validation; lợi ích ở kịch bản S1–S4 |
| Ngưỡng lũ 10 năm có ít sự kiện trong 4 năm test | Báo cáo số sự kiện; kết luận chính dựa trên 1 và 2 năm |
| Thời gian chạy vượt dự kiến | Cắt theo Mục 8.3 |
| PyTorch không chạy được trên CPU ARM của VPS, hoặc Mamba bằng `mambapy` trên CPU lệch kết quả hay quá chậm | Kiểm ở bước D2 trên Kaggle, D4 trên VPS (Mục 9.6); Mamba không đạt thì phần chạy trực tiếp chỉ dùng LSTM cải tiến, các lõi khác đọc kết quả tính sẵn; VPS không đạt thì dùng bản laptop (D3) |
| VPS Oracle Always Free bị thu hồi khi rảnh (7 ngày dưới 20% CPU, mạng, bộ nhớ — tài liệu Oracle, cập nhật 12/6/2026) | Đóng gói demo bằng Docker để dựng lại nhanh; giữ bản chạy trên laptop (Mục 9.6, bước D3) cho ngày bảo vệ |

### 8.3. Thứ tự cắt nếu thiếu thời gian

Cắt từ trên xuống; RQ1–RQ4 giữ nguyên, RQ5 và RQ6 chỉ thu hẹp: (1) biến thể dự phòng ở bước E; (2) thí nghiệm GV; (3) bước F (CMAL; demo khi đó tô màu theo dự báo điểm so với ngưỡng, không có xác suất vượt ngưỡng); (4) giải thích mô hình; (5) biến thể chính ở bước E; (6) 730 ngày cho S4D; (7) cách (c) ở bước H; (8) tinh chỉnh giảm từ 8 xuống 4 cấu hình mỗi lõi.

---

## 9. Hạ tầng và tổ chức mã

### 9.1. Khung và nơi chạy

- **Thư viện và cấu hình tách riêng.** NeuralHydrology là thư viện Python; thí nghiệm khai báo bằng tệp `config.yml` (chọn mô hình, số chiều, hạt giống, xác suất che…). Cấu hình chỉ chọn được khả năng thư viện đã có, nên các phần chưa có (bộ lọc mẫu mới, mô hình chung nhận lõi bất kỳ, S4D, Mamba đúng trục, xác suất che riêng cho Q, lưu mô hình tốt nhất) được viết vào thư viện; mọi thí nghiệm còn lại chỉ là tệp cấu hình.
- **Fork thư viện:** `Ticasslo/neuralhydrology`, fork từ bản của tác giả BiasCast (`conestone/neuralhydrology`). Bản của tác giả là thư viện gốc `neuralhydrology/neuralhydrology` tại commit `f00cf47` cộng đúng một commit `9d94908` sửa 7 tệp (bộ nạp `datasetzoo/lamah.py`, `training/basetrainer.py`, `training/early_stopping.py`, `training/logger.py`, `training/__init__.py`, `evaluation/tester.py`, `utils/config.py`); Sequential Forecast LSTM có sẵn trong thư viện gốc (commit `feb4d45`, #199). Fork từ bản của tác giả để cấu hình BiasCast chạy được ngay và tái lập đúng. Nhánh `master` giữ nguyên mã tác giả (dùng cho bước A); nhánh `research` chứa phần đề tài viết thêm, mỗi thay đổi một commit để so sánh được khối giữ và khối thêm.
- **Lấy bản sửa từ thư viện gốc:** thư viện gốc đã có thêm 19 commit sau `f00cf47`. Ở bước 0, nhánh `research` lấy riêng (cherry-pick, mỗi commit một lần) các commit cần cho đề tài: `68fbddb` (#280, tương thích Numpy 2.4 — môi trường Kaggle), `1d5716d` (#264, sửa tên tham số dropout của Transformer — bước C), `2f374a4` (#295, chỉ số đánh giá dự báo xác suất — bước F). Lõi xLSTM (`843ed59`, #259) để cho khóa luận. Không lấy `9528d1d` (#303, sửa thuộc tính `dynamic_learning_rate` của `utils/config.py`): bản fork của tác giả đã viết lại phần lập lịch learning rate và dừng sớm, không có thuộc tính này; `ea94a40` (#302) chưa cần. Nhánh `master` giữ nguyên mã tác giả. Bước A chạy trên `research` sau khi lấy ba commit trên, vì ba commit không đổi kết quả của mô hình cho đầu ra điểm: #280 chỉ đổi `int(f)` thành `f.item()`, #264 chỉ chạm Transformer, #295 chỉ khác khi đầu ra có chiều `samples`; cả ba áp được lên `9d94908` không xung đột (kiểm bằng `git apply --check`). Kết quả: nhánh `research` tại commit `dda45fc` (`9d94908` + 3 commit trên, giữ thông tin commit gốc bằng `cherry-pick -x`); notebook cài thư viện bằng mã commit này.
- **Mốc cố định cho từng nhóm thí nghiệm:** thư viện được sửa chủ yếu ở bước 0 → B; trước bước C gắn tag `freeze-C`, toàn bộ bước C và D chạy trên mốc này để mọi lõi dùng cùng mã. Sau mốc, chỉ được thêm mã mới (lõi, biến thể ở bước E) mà không đổi mã dùng chung; mỗi lần thêm phải kiểm chứng bằng cách nạp cùng trọng số, chạy lại một cấu hình cũ và nhận đầu ra trùng với mốc trước, rồi gắn mốc mới (`freeze-E`…). Nếu phát hiện lỗi ở mã dùng chung (bộ nạp, vòng huấn luyện) sau khi đã chạy, sửa lỗi, gắn mốc mới và chạy lại mọi thí nghiệm bị ảnh hưởng; không trộn kết quả của hai mốc trong cùng một bảng so sánh; báo cáo ghi rõ mỗi bảng chạy trên mốc nào.
- **Thư viện trong repo đề tài:** fork được gắn làm submodule ở thư mục `NeuralHydrology/`, theo nhánh `research` (tạo từ commit `9d94908`). Submodule ghim một commit cụ thể, chỉ đổi khi chủ động cập nhật: sửa thư viện thì commit và đẩy trong fork, rồi commit lại con trỏ submodule trong repo đề tài.
- **Cài theo mã commit:** notebook Kaggle cài `pip install git+https://github.com/Ticasslo/neuralhydrology@<mã commit>`, không cài theo tên nhánh, để mỗi kết quả chạy lại được đúng phiên bản mã.
- **Tệp cấu hình** đặt trong repo đề tài (thư mục huấn luyện ở `Workspace/`), mỗi thí nghiệm một tệp; mỗi lần huấn luyện lưu kèm tệp cấu hình đã dùng và mã commit thư viện.
- Bản sao mã tác giả trong `PaperResearch/PaperResearchCode/` giữ nguyên để đối chiếu.
- Kaggle chính, Colab dự phòng; mã chạy được trên cả hai.

### 9.2. Kiểm thử đơn vị trước khi huấn luyện

| Kiểm thử | Mục đích |
|---|---|
| Mô hình chung với lõi LSTM nạp trọng số tác giả cho đầu ra trùng Sequential Forecast LSTM (sai khác dưới 10⁻⁵) | Bảo đảm khung chung đúng |
| Mamba nhận tensor (batch, length, dim) | Tránh lỗi trục như lớp có sẵn |
| Không rò rỉ: đổi đầu vào sau ngày t không đổi dự báo | Kiểm mặt nạ nhân quả, quét hai chiều |
| Bộ lọc mẫu mới giữ mẫu có NaN đầu vào, loại mẫu thiếu nhãn | Kiểm sửa `_validate_samples` |
| Masked mean với một nhóm bị che cho kết quả hữu hạn | Kiểm đường đi NaN |
| Số tham số các lõi ở mức cùng tham số lệch không quá 1% | Kiểm giao thức Mục 5.2 |
| Bước H cách (b): ngày t+1 đến t+6 không nhận dự báo ECMWF; bước không có nhóm nào vẫn cho đầu ra hữu hạn | Tránh dùng thông tin tốt hơn thực tế ở bước H |

### 9.3. Phân công

| Thành viên | Mảng | Việc chính |
|---|---|---|
| Người 1 | Dữ liệu và đánh giá | Khảo sát dữ liệu, ngưỡng mức và ngưỡng lũ, bộ nạp (nhóm theo nguồn, sửa bộ lọc, Q, cách b của bước H), bộ đánh giá Mục 6, demo |
| Người 2 | Mô hình và huấn luyện | Fork thư viện, mô hình chung và các lõi, kiểm thử đơn vị, cài `mamba-ssm`, tệp cấu hình, huấn luyện trên Kaggle, giải thích mô hình |

Điểm nối giữa hai mảng: bộ nạp (người 1) phải xong trước bước B2; đầu ra dự báo của người 2 lưu theo một định dạng chung để bộ đánh giá đọc được.

### 9.4. Thư mục mã

Mã đặt trong `Workspace/`, mỗi thư mục con một việc; mỗi notebook một tệp `.py` tự đủ, có mục lục, chia cell theo "Phần". Tên tệp (cũng là tên notebook Kaggle) dạng `NN_<Bước>_<BộDữLiệu>_<Việc>.py`: `NN` là thứ tự chạy trong thư mục, `<Bước>` là mã bước ở Mục 7 (VD `03_Baseline/01_A2_LamaHCEExt_Reproduce.py`); tệp không thuộc bước nào (tải dữ liệu) bỏ phần `<Bước>`. Thứ tự thực hiện bước A: A2 → A1 → A3 → A4, xong trước bước B. A2 chạy trước để kiểm môi trường; A1 trước A3 vì DLinear trong A3 cần A1 để quyết cách chuẩn hóa và xử lý ô trống của `qmean`, và persistence cần biết trước các lưu vực bất thường; B cần số liệu thiếu và số mẫu hợp lệ của A1.

| Thư mục | Việc | Trạng thái |
|---|---|---|
| `01_Download/` | Tải dữ liệu | Xong |
| `02_Exploration/` | Khảo sát dữ liệu (A1), ngưỡng | Có nháp Phần 1 của `01_A1_LamaHCEExt_Explore.py` |
| `03_Baseline/` | A2 tái lập (`01_A2_LamaHCEExt_Reproduce.py`), A3 mốc so sánh gồm persistence, PNSE, KGE, DLinear (`02_A3_LamaHCEExt_Benchmarks.py`), A4 huấn luyện lại và nhiễu hạt giống (`03_A4_LamaHCEExt_Retrain.py`) | A2 đã chạy, tái lập đạt (`03_Data/01_LamaHCE.md` Mục 6.3) |
| `04_Setup/` | Cài môi trường, wheel `mamba-ssm`, kiểm thử đơn vị, đo thời gian | Chưa viết |
| `05_Training/` | Notebook huấn luyện theo bước B–H | Chưa viết |
| `06_Evaluation/` | Bộ đánh giá Mục 6 | Chưa viết |
| `07_Demo/` | Demo (mục dự án trong trang web của nhóm) | Chưa viết |

Tên thư mục từ `04_` trở đi là đề xuất, đặt khi bắt đầu viết mã.

### 9.5. Đầu ra cho đánh giá và demo

Bộ đánh giá chỉ đọc dự báo đã lưu, không gọi mô hình. Demo dùng các tệp này cho bản đồ và đường Q, còn phần chạy mô hình trực tiếp theo Mục 9.6. Mỗi cấu hình chính thức xuất một tệp dự báo kỳ test theo định dạng chung: mã lưu vực, ngày phát hành, hạn (1–7), qmax quan trắc, qmax dự báo (mm/ngày và m³/s), phân vị khi dùng CMAL, mã cấu hình và mã commit thư viện. Kết quả giải thích mô hình (Mục 6.7) lưu riêng, tổng hợp theo lưu vực, để demo hiển thị. Định dạng tệp cụ thể chọn khi viết bộ đánh giá. Trang demo công khai phải ghi nguồn dữ liệu (Extended LamaH-CE, LamaH-CE, BiasCast) kèm giấy phép, ghi rõ đây là phát lại kỳ test để minh họa, không phải cảnh báo chính thức (điều kiện CHMI về số liệu lưu lượng Séc, `01_OverallPlan.md` Mục 9), và không thu phí, không quảng cáo (giấy phép phi thương mại).

### 9.6. Đưa mô hình vào demo

Demo có hai phần. Phần phát lại đọc tệp dự báo ở Mục 9.5 (bản đồ 451 lưu vực, đường Q, ngưỡng lũ, mức ảnh hưởng của các nhóm nguồn). Phần chạy mô hình trực tiếp nhận lưu vực và ngày t trong kỳ test, lấy cửa sổ hindcast, dự báo ECMWF HRES ngày t và thuộc tính lưu vực từ dữ liệu đã lưu, chuẩn hóa bằng thông số chuẩn hóa lúc huấn luyện, chạy mô hình rồi đổi kết quả sang m³/s; phần thử "nếu… thì…" cho người xem nhân lượng mưa hoặc cộng nhiệt độ của ECMWF HRES ngày t rồi chạy lại. Dữ liệu Extended LamaH-CE chỉ tới 2017 nên đây vẫn là phát lại, không phải dự báo thời gian thực (Mục 11).

Huấn luyện chạy trên GPU (Kaggle); demo chạy trên CPU. Mỗi lần dự báo một lưu vực, một ngày là một lượt chạy xuôi của mô hình khoảng 85 nghìn tham số trên chuỗi 365 bước, nên CPU đủ dùng. GPU miễn phí để phục vụ web không phù hợp: Hugging Face ZeroGPU tính hạn mức GPU theo tài khoản người xem (tài khoản miễn phí 5 phút mỗi ngày) và chỉ chạy Gradio; Space Gradio hoặc Docker trên CPU Basic cần gói trả phí mới tạo được (tài liệu Hugging Face *Spaces ZeroGPU*, *Using GPU Spaces*). Giao diện chạy trong notebook Kaggle không dùng làm demo vì chỉ sống trong phiên notebook (tối đa 12 giờ).

`mamba-ssm` chỉ chạy trên GPU CUDA, nên khi chạy trên CPU, khối Mamba thay bằng `mambapy` (PyTorch thuần). README của `mambapy` ghi cách cài đặt tương đương số học với bản gốc và có hàm nạp trọng số `state-spaces/mamba-130m`; việc ánh xạ trọng số cho khối Mamba của đề tài chưa kiểm. LSTM, GRU, Transformer chạy CPU bằng nguyên mã; S4D chạy CPU nếu cài đặt không dùng kernel CUDA riêng (chọn khi viết lõi ở bước C).

Thứ tự thực hiện, bước sau chỉ làm khi bước trước đạt:

| Bước | Nơi chạy | Thiết bị | Việc | Điều kiện đạt |
|---|---|---|---|---|
| D1 | Kaggle | GPU | Viết hàm dự báo tách khỏi mã huấn luyện: nạp trọng số và thông số chuẩn hóa, nhận mã lưu vực và ngày t, trả qmax; chạy lại trên toàn kỳ test | Trùng tệp dự báo của bộ đánh giá (Mục 9.5), chỉ lệch ở mức làm tròn số thực; ngưỡng sai khác đặt khi viết mã |
| D2 | Kaggle | CPU | Chạy cùng hàm trên CPU; Mamba dùng `mambapy` | Trùng kết quả D1; đo thời gian mỗi lần dự báo và bộ nhớ |
| D3 | Laptop | CPU | Tải trọng số, thông số chuẩn hóa và dữ liệu đầu vào từ Kaggle; dựng giao diện demo (phần phát lại và phần chạy trực tiếp) | Trùng kết quả D2; giao diện chạy đủ hai phần |
| D4 | VPS Oracle (CPU ARM) | CPU | Chép đúng mã D3 lên VPS, cài PyTorch bản CPU cho ARM, đóng gói Docker (Mục 8.2), đưa ra Internet | Trùng kết quả D3; có đường dẫn công khai |

Bản trên laptop (D3) giữ làm dự phòng cho ngày bảo vệ. Nếu Mamba trên CPU không đạt ở D2 (lệch kết quả hoặc quá chậm), phần chạy trực tiếp chỉ dùng LSTM cải tiến, các lõi khác hiển thị kết quả tính sẵn.

---

## 10. Đối chiếu với pipeline của BiasCast

| Giai đoạn | BiasCast | Đề tài | Giống / khác |
|---|---|---|---|
| Dữ liệu | ERA5-Land, E-OBS, MSWEP, GLEAM, ECMWF HRES theo lưu vực | Cùng dữ liệu Zenodo | Giống |
| Bộ nạp | Loại mẫu có bất kỳ giá trị thiếu nào; Q đầu vào `qmean` theo m³/s, khác đơn vị nhãn mm/ngày (A2 Phần 3) | Sửa bộ lọc; nhóm theo nguồn; Q cùng đơn vị với nhãn | Sửa, mở rộng |
| Mô hình | Chỉ LSTM (tác giả tự nêu hạn chế, Mục 3.7) | LSTM, GRU, Transformer, S4D, Mamba trong khung chung | Khoảng trống chính |
| Huấn luyện | NSE*, tối ưu Bayes, 1 hạt giống | Giữ NSE* và lịch học; tinh chỉnh cùng ngân sách; 3 hạt giống | Khác số hạt giống, giao thức tinh chỉnh |
| Đánh giá | NSE, KGE; cận trên, cận dưới | Thêm persistence, PNSE, theo mức, theo sự kiện lũ, kịch bản vận hành, kiểm định | Mở rộng |
| Hạn dự báo | 1 ngày | 1 ngày (bước C) và 1–7 ngày (bước H) | Mở rộng |
| Tái lập, XAI, ngưỡng lũ, demo | Không có | Có | Chỉ có ở đề tài |

---

## 11. Khóa luận

| Hướng | Lý do để sau |
|---|---|
| Ghép dữ liệu giờ vào quá khứ gần (MF-LSTM) | Cần bộ nạp hai độ phân giải; dữ liệu giờ đã có (`lamah-ce-core`, `lamah-ce-extra`) |
| Dự báo nhiều ngày với dự báo thời tiết nhiều hạn thật (tiểu luận chỉ có ngày đầu, bước H) | Extended LamaH-CE chỉ có dự báo ECMWF HRES hạn 1 ngày: tệp khí tượng có đúng 8 cột ECMWF, mỗi biến một giá trị mỗi ngày, lấy trung bình từ lần phát hành 00 UTC (BiasCast Mục 2.1, 2.3; `03_Data/01_LamaHCE.md` Mục 6.1). Nguồn ứng viên: dự báo tái lập GEFSv12 (NOAA, 2000–2019, 16 ngày, miễn phí trên AWS), kho lưu trữ ECMWF HRES; tác giả BiasCast thông báo đang làm nghiên cứu dự báo nhiều ngày (BiasCast Mục 4) |
| Hàm mất mát ưu tiên đỉnh lũ | Có đánh đổi với kỹ năng chung (Baste 2025; Talbot & Davenport, preprint 2026) |
| Đồ thị mạng sông (Y18) | Đồ thị thô không tự cải thiện (Kirschstein & Sun 2024); GAT, reachability có bằng chứng (Mosaffa và cs., HESS 2026; Wang và cs., npj Natural Hazards 2025); chưa tìm thấy công trình kết hợp GNN với Mamba cho dòng chảy |
| Tiền huấn luyện trên tái phân tích 1981–2002 rồi tinh chỉnh trên dự báo (Y19) | AIFL (J. Hydrol. 2026) thấy hai giai đoạn tốt hơn, BiasCast thấy học chuyển giao kém; kiểm trên cùng dữ liệu để giải thích mâu thuẫn |
| So với khung mã nguồn mở của Google (Y20) | Google công bố mã 3/6/2026 (Apache 2.0); cần kiểm có trọng số sẵn không và dữ liệu huấn luyện Caravan có chứa lưu vực LamaH-CE không |
| Biến thể lai Mamba–Transformer (SST), LOAN, xLSTM | Sau khi có kết quả dạng thuần |
| Phần mềm ứng dụng | Yêu cầu của khóa luận (mức "có tính ứng dụng" cần hỏi GVHD). Nâng trang demo: tìm kiếm lưu vực theo tên, sông, vị trí; trang từng lưu vực (Q 1–7 ngày, ngưỡng, xác suất vượt ngưỡng); so sánh mô hình và giải thích mô hình theo lưu vực; tài khoản, cảnh báo qua email; API. Tham khảo Google Flood Hub (trang Google Research *Flood Forecasting*, Flood Forecasting API). Dữ liệu Extended LamaH-CE chỉ tới 2017 nên vẫn là phát lại; chạy gần thời gian thực cần nguồn trực tiếp (eHYD, dữ liệu mở ECMWF, ERA5-Land), chưa kiểm giấy phép và cách tải |

---

## 12. Việc chưa chốt

| Việc | Ghi chú |
|---|---|
| Tên nội bộ tiểu luận | `01_OverallPlan.md` Mục 1 |
| Trang web và demo: công nghệ giao diện; cách đưa ra Internet (Cloudflare Tunnel cần tên miền quản lý trên Cloudflare; mở cổng và reverse proxy); chạy liên tục khi VPS Always Free có thể bị thu hồi lúc rảnh (Mục 8.2) | Tra và quyết khi có bản demo chạy thử; mã đề tài xuất dự báo theo định dạng ở Mục 9.5 và hàm dự báo theo các bước ở Mục 9.6 |

---

## 13. Căn cứ chính cho các quyết định

| Quyết định | Căn cứ |
|---|---|
| Masked mean theo nhóm nguồn, che dữ liệu khi huấn luyện | Gauch và cs., HESS 29:6221–6235 (2025); mức che 0,05 lấy theo MF²LSTM (preprint EGUsphere 2026); mã fork (`basedataset.py`, `inputlayer.py`) |
| Độ trễ từng nguồn | Copernicus (ERA5-Land khoảng 5 ngày; E-OBS hằng ngày trễ 1 ngày, bản đầy đủ 2 lần/năm); UCAR Climate Data Guide (MSWEP NRT 2–3 giờ); ICDC Hamburg (GLEAM mỗi năm); BiasCast Mục 3.7 (eHYD khoảng 2 giờ) |
| Giao thức cùng tham số | Shahriar (SSRN 2026); Liu và cs. (J. Hydrol. 2024) |
| PNSE | Kitanidis & Bras, WRR 16:1034–1044 (1980); MF²LSTM (preprint EGUsphere 2026) |
| Chu kỳ lặp lại 1, 2, 5, 10 năm; precision, recall, F1; cửa sổ thời gian | Nearing và cs., Nature 627 (2024) |
| Gumbel L-moments | Hosking, JRSS B 52:105–124 (1990); AIFL (Taccari và cs., J. Hydrol. 678, 2026) |
| Đoạn đường duy trì lưu lượng | Yilmaz, Gupta, Wagener, WRR 44 (2008) |
| Wilcoxon signed-rank + Cohen's d | Kratzert và cs. (2019, 2024); Gauch và cs. (2025); Nearing và cs. (2024) |
| Siêu tham số khởi điểm S4D | Wang và cs., WRR 2025, phụ lục S4D-FT |
| PatchTST, DLinear | Nie và cs. (ICLR 2023); Zhang và cs. (J. Hydrol. 2026) |
| Trạng thái chạy liên tục thay vì handoff | BiasCast Mục 3.6 |

---

## 14. Tài liệu tham khảo

Bài ghi "preprint" chưa qua phản biện; chỉ dùng làm công trình liên quan để xác định tính mới hoặc làm nguồn phụ, mọi quyết định phương pháp có thêm nguồn đã công bố. Tệp PDF của các bài có ghi chú nằm trong `PaperResearch/PaperResearchPDF/` (tóm tắt từng bài: `Document/05_Survey/03_PaperNotes.md`).

**Bài cơ sở và dữ liệu**
- Konold, O., Feigl, M., Podest, P., Klingler, C., Schulz, K. (2026). BiasCast. *Hydrology and Earth System Sciences* 30, 5067–5096. https://doi.org/10.5194/hess-30-5067-2026 — mã `github.com/conestone/biascast`, `github.com/conestone/neuralhydrology`; thư trả lời phản biện: `BasePaper/PeerReview/`.
- Klingler, C., Schulz, K., Herrnegger, M. (2021). LamaH-CE. *Earth System Science Data* 13, 4529–4565. https://doi.org/10.5194/essd-13-4529-2021

**Phương pháp huấn luyện và đánh giá**
- Kratzert, F. và cs. (2019). NSE*, mô hình nhiều lưu vực. *HESS* 23, 5089–5110. https://doi.org/10.5194/hess-23-5089-2019
- Kratzert, F. và cs. (2024). Không huấn luyện trên một lưu vực. *HESS* 28, 4187. https://doi.org/10.5194/hess-28-4187-2024
- Gauch, M. và cs. (2025). Đầu vào thiếu, masked mean. *HESS* 29, 6221. https://doi.org/10.5194/hess-29-6221-2025
- Nearing, G. và cs. (2022). Đồng hóa và tự hồi quy với Q quan trắc. *HESS* 26, 5493–5513. https://doi.org/10.5194/hess-26-5493-2022
- Nearing, G. và cs. (2024). Dự báo lũ toàn cầu. *Nature* 627, 559. https://doi.org/10.1038/s41586-024-07145-1
- Klotz, D. và cs. (2022). Bất định, CMAL. *HESS* 26, 1673. https://doi.org/10.5194/hess-26-1673-2022
- Kitanidis, P. K., Bras, R. L. (1980). Real-time forecasting with a conceptual hydrologic model, 2. Applications and results. *Water Resources Research* 16(6), 1034–1044 — hệ số persistence.
- Hosking, J. R. M. (1990). L-moments. *Journal of the Royal Statistical Society, Series B* 52(1), 105–124.
- Yilmaz, K. K., Gupta, H. V., Wagener, T. (2008). Đoạn đường duy trì lưu lượng. *Water Resources Research* 44, W09417. https://doi.org/10.1029/2007WR006716
- Martel, J.-L. và cs. (2025). Lấy mẫu nhiều đỉnh. *HESS* 29, 4951–4968. https://doi.org/10.5194/hess-29-4951-2025
- Baste, S. và cs. (2025). Giới hạn đỉnh của LSTM. *HESS* 29. https://doi.org/10.5194/hess-29-5871-2025
- Taccari, M. L. và cs. (2026). AIFL. *Journal of Hydrology* 678, 136064. https://doi.org/10.1016/j.jhydrol.2026.136064 (tệp `Related/Taccari2026_arXiv_AIFL.pdf` là bản arXiv 2602.16579).
- Acuña Espinoza, E. và cs. (2026). MF²LSTM. Preprint EGUsphere, đang phản biện ở *HESS*. https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4885/
- Saint-Fleur, B. E. và cs. (2026). Chiến lược đồng hóa Q. *HESS* 30, 3497–3527. https://doi.org/10.5194/hess-30-3497-2026
- Hu, Y. và cs. (2025). Dò khái niệm thủy văn trong trạng thái LSTM. *Environmental Modelling & Software* 192, 106527. https://doi.org/10.1016/j.envsoft.2025.106527
- Talbot và Davenport (2026). *No Free Lunch? Improving LSTM Flood Predictions with Minimal Loss…* Preprint ESS Open Archive. https://doi.org/10.22541/essoar.177075740.01674581

**Kiến trúc**
- Gu, A., Dao, T. (2024). Mamba. *COLM 2024*, arXiv 2312.00752.
- Gu, A. và cs. (2022). S4D. *NeurIPS 2022*, arXiv 2206.11893.
- Wang và cs. (2025). S4D-FT. *Water Resources Research*. https://doi.org/10.1029/2025WR039888
- Jing và cs. (2026). S4D, S5D. *Expert Systems with Applications*. https://doi.org/10.1016/j.eswa.2026.133040
- Zhang và cs. (2026). LSTM, Transformer, PatchTST, Mamba, DLinear. *Journal of Hydrology*. https://doi.org/10.1016/j.jhydrol.2026.135727
- Liu và cs. (2024). Giới hạn của Transformer. *Journal of Hydrology* 637, 131389. https://doi.org/10.1016/j.jhydrol.2024.131389
- Nie, Y. và cs. (2023). PatchTST. *ICLR 2023*, arXiv 2211.14730.
- Zeng, A. và cs. (2023). DLinear. *AAAI 2023*, arXiv 2205.13504.
- Sheng và cs. (2026). ResBi-Mamba Plus. *AISE 2026*. https://doi.org/10.23919/AISE.2026.000010
- Wang và cs. (2025). S-Mamba, *Is Mamba Effective for Time Series Forecasting?* *Neurocomputing*. https://doi.org/10.1016/j.neucom.2024.129178
- Li và cs. (2024). CMMamba. *Journal of Big Data*. https://doi.org/10.1186/s40537-024-01001-9
- Shams Eddin, M. H. và cs. (2025). RiverMamba. *NeurIPS 2025*. https://doi.org/10.52202/085713-4446

**Công trình so sánh cho tính mới**
- Shahriar, M. (2026). Mamba và LSTM cùng tham số, 671 lưu vực CAMELS. Preprint SSRN. https://doi.org/10.2139/ssrn.7552815
- Demiray & Demir (2025). Mamba cho dự báo dòng chảy dài hạn, Iowa. Preprint EarthArXiv. https://doi.org/10.31223/X5B164
- Dubey và cs. (2025). Kịch bản trễ và mất dữ liệu. arXiv 2510.18535.
- Wang, Y. và cs. (2026). HydroDiffusion. *Water Resources Research*. https://doi.org/10.1029/2025WR043158 (bản arXiv 2512.12183).

**Hướng khóa luận**
- Kirschstein, N., Sun, Y. (2024). The Merit of River Network Topology for Neural Flood Forecasting. *ICML 2024*, PMLR 235, 24713–24725.
- Mosaffa, H. và cs. (2026). LSTM + GNN định tuyến trên LamaH-CE. *HESS* 30, 2079. https://hess.copernicus.org/articles/30/2079/2026/
- Wang, H. và cs. (2025). Đồ thị reachability. *npj Natural Hazards*. https://doi.org/10.1038/s44304-025-00083-6

**Nguồn dữ liệu vận hành và hạ tầng**
- Copernicus Climate Change Service: *Daily vs Full E-OBS updates*, https://surfobs.climate.copernicus.eu/userguidance/daily_vs_full_eobs.php ; trang Climate reanalysis (độ trễ ERA5-Land).
- UCAR Climate Data Guide: MSWEP, https://climatedataguide.ucar.edu/climate-data/global-high-resolution-precipitation-mswep
- ICDC Hamburg: cập nhật GLEAM v3.7 → v4.1 (8/2024), https://cen.uni-hamburg.de/en/icdc/about-icdc/news/2024/20240823-upgrade-of-gleam-evaporation-parameters.html ; Miralles và cs. (2025), GLEAM4, *Scientific Data*. https://doi.org/10.1038/s41597-025-04610-y
- Caravan MultiMet (arXiv 2411.09459) — dự báo IFS HRES 10 ngày cho lưu vực Caravan.
- NOAA GEFSv12 reforecast, https://registry.opendata.aws/noaa-gefs-reforecast
- `mamba-ssm`: https://github.com/state-spaces/mamba (`setup.py`); dự phòng `mambapy`: https://github.com/alxndrTL/mamba.py
- Oracle Cloud, *Always Free Resources* (cập nhật 12/6/2026), https://docs.oracle.com/en-us/iaas/Content/FreeTier/resourceref.htm

