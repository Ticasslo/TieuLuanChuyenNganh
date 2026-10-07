# Kiến trúc và pipeline của đề tài

> Tài liệu gốc của đề tài: câu hỏi nghiên cứu, đóng góp và tính mới, dữ liệu và bộ nạp, kiến trúc mô hình, giao thức huấn luyện và so sánh công bằng, bộ đánh giá, ma trận thí nghiệm theo bước, ngân sách tính toán, rủi ro, phạm vi tiểu luận và khóa luận. Bài cơ sở: BiasCast (Konold và cs., HESS 2026) — `Document/01_Plan/02_BasePaper.md`. Dữ liệu: `Document/03_Data/01_LamaHCE.md`. Ý tưởng và căn cứ: `Document/04_Ideas/01_Ideas.md`. Ghi chú từng tài liệu: `Document/05_Survey/03_PaperNotes.md`. Các mục ghi "chưa chốt" chờ quyết định. Mọi nhận định "chưa có công trình nào" dựa trên tra cứu tới ngày 7/10/2026 và được ghi kèm công trình gần nhất đã biết.

---

## 1. Câu hỏi nghiên cứu, đóng góp và tính mới

### 1.1. Câu hỏi nghiên cứu

| Mã | Câu hỏi | Trả lời bằng |
|---|---|---|
| RQ1 | Kết quả BiasCast có tái lập được không, nhiễu giữa các hạt giống lớn cỡ nào, và mô hình thật sự hơn các mốc không học (persistence) và mốc tuyến tính bao nhiêu? | Bước A |
| RQ2 | Các cải tiến quy trình không phụ thuộc kiến trúc (sửa lỗi mã, thống nhất đơn vị Q, giữ mẫu thiếu dữ liệu bằng masked mean theo nguồn, huấn luyện có che dữ liệu) cải thiện Sequential Forecast LSTM bao nhiêu và giúp mô hình chịu mất dữ liệu ra sao? | Bước B |
| RQ3 | Trong bài toán dự báo có dự báo thời tiết thật và Q quan trắc, khi cùng số tham số, cùng ngân sách tinh chỉnh, cùng hạt giống, lõi SSM (S4D, Mamba) có hơn, ngang hay kém LSTM, GRU, Transformer? Hơn ở dải lưu lượng nào, loại lưu vực nào? | Bước C, E; đánh giá Mục 6 |
| RQ4 | Khi dữ liệu đến trễ đúng như thực tế vận hành (mỗi nguồn một độ trễ) hoặc mất Q, lõi nào suy giảm ít hơn? | Đánh giá kịch bản vận hành (Mục 6.3) |
| RQ5 | Chuỗi quá khứ dài hơn (730 ngày) có giúp không, với chi phí tính toán bao nhiêu? | Bước D |

### 1.2. Đóng góp dự kiến

| Mã | Đóng góp | Loại |
|---|---|---|
| C1 | So sánh có kiểm soát (cùng tham số, cùng ngân sách tinh chỉnh, 3 hạt giống, kiểm định thống kê) giữa LSTM, GRU, Transformer, S4D, Mamba trong khung hindcast–forecast dùng dự báo ECMWF HRES lưu trữ thật và Q quan trắc, trên 451 lưu vực Trung Âu | Thực nghiệm, tính mới chính |
| C2 | Bộ đánh giá vận hành: kịch bản độ trễ theo từng nguồn dữ liệu (ERA5-Land, E-OBS, MSWEP, GLEAM, Q) và mất Q, áp dụng cho mọi lõi | Phương pháp đánh giá |
| C3 | Tái lập BiasCast, đo nhiễu hạt giống, bổ sung mốc persistence, PNSE, mốc tuyến tính; phát hiện và sửa lỗi trong mã (bộ lọc mẫu, đơn vị Q, lưu mô hình tốt nhất) | Tái lập, chuẩn hóa đánh giá |
| C4 | Đánh giá theo mức lưu lượng và theo sự kiện lũ (chu kỳ lặp lại), phân tích theo lưu vực kiểm giả thuyết "SSM có lợi ở lưu vực động lực chậm, bất lợi ở lũ nhanh" | Phân tích |
| C5 | Giải thích mô hình (Integrated Gradients) so sánh độ dài nhớ hiệu dụng của LSTM và Mamba; demo bản đồ cảnh báo 451 lưu vực | Ứng dụng |

### 1.3. Tính mới so với công trình gần nhất

| Công trình gần nhất | Đã làm | Khác biệt của đề tài |
|---|---|---|
| BiasCast (Konold và cs., HESS 2026) | Khung hindcast–forecast với ECMWF HRES thật và Q quan trắc, chỉ LSTM, 1 hạt giống, không persistence | Thêm các lõi khác dưới giao thức công bằng; nhiều hạt giống; mốc persistence; sửa lỗi mã |
| Shahriar (preprint SSRN, 10/2026) | Mamba và LSTM cùng tham số, cùng độ sâu, 671 lưu vực CAMELS; LSTM thắng mọi cặp (NSE 0,752 so với 0,727); Mamba quá khớp sau khoảng 10 epoch | Bài đó là mô phỏng (không dự báo thời tiết, không Q quá khứ); đề tài là dự báo vận hành với dự báo thật, Q quan trắc, đánh giá theo mức và kịch bản trễ |
| Zhang và cs. (J. Hydrol. 2026) | LSTM, Transformer, PatchTST, Mamba, DLinear theo giờ, CAMELS; Mamba gần nhưng hơi kém LSTM | Đề tài dùng dự báo thời tiết lưu trữ thật, dữ liệu Trung Âu, thêm S4D, kiểm định thống kê theo lưu vực, kịch bản vận hành |
| Demiray & Demir (preprint EarthArXiv) | Mamba so với LSTM, GRU, Transformer, 125 lưu vực Iowa, theo giờ, có SHAP | Khác bài toán (không ECMWF lưu trữ, không giao thức cùng tham số); đề tài thêm S4D, kịch bản vận hành, đánh giá theo mức |
| RiverMamba (NeurIPS 2025) | Mamba quét không gian trên lưới 0,05°, nhãn là tái phân tích GloFAS | Đề tài theo lưu vực, nhãn là Q quan trắc, so sánh có kiểm soát với các lõi khác |
| Gauch và cs. (HESS 2025); MF²LSTM (preprint EGUsphere 2026) | Masked mean cho dữ liệu khí tượng thiếu (mô phỏng, CAMELS); MF²LSTM che Q để đồng hóa, theo giờ, một mô hình LSTM | Đề tài áp dụng cho bài toán ngày có dự báo thật, cho mọi lõi, nhóm theo nguồn dữ liệu, và kiểm tra bằng độ trễ thực tế của từng nguồn |
| Dubey và cs. (arXiv 2025) | Kịch bản trễ và mất dữ liệu cho LSTM mã hóa – giải mã, CAMELS-US, HYSETS, CAMELS-IND, có HRES | Đề tài: độ trễ riêng của từng nguồn (E-OBS 6 tháng, GLEAM 1 năm…), Q quan trắc, so sánh nhiều lõi |

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
| 0. Hạ tầng | Fork NeuralHydrology riêng, cài `mamba-ssm` trên Kaggle T4, kiểm thử đơn vị, đo thời gian | Môi trường chạy được mọi lõi; bảng thời gian mỗi epoch | Chưa làm |
| 1. Thu thập dữ liệu | Tải Extended LamaH-CE và kết quả của tác giả | Kaggle Dataset `lamah-ce-ext` | Xong (`01_LamaHCE.md` Mục 6) |
| 2. Khảo sát dữ liệu | Theo khuôn GVHD: dạng cơ bản + thống kê sâu; ngưỡng mức lưu lượng; bảng độ trễ nguồn | Báo cáo khảo sát, tệp ngưỡng | Một phần số liệu đã có; notebook viết lại |
| 3. Tái lập và mốc | Chạy lại trọng số tác giả; persistence; DLinear; nhiễu hạt giống | Bảng mốc | Persistence xong (0,35–0,37) |
| 4. Bộ nạp | Sửa bộ lọc mẫu, nhóm biến theo nguồn, Q cùng đơn vị | Cấu hình dữ liệu dùng chung | Chưa làm |
| 5. Mô hình | Khung hindcast–forecast với lõi thay được | Mô-đun trong fork | Chưa làm |
| 6. Huấn luyện | Bước B → F (Mục 7) | Trọng số, nhật ký | Chưa làm |
| 7. Đánh giá | Bộ đánh giá Mục 6 | Bảng, hình, kiểm định | Chưa làm |
| 8. Giải thích | Integrated Gradients | Hình mức ảnh hưởng | Chưa làm |
| 9. Ngưỡng lũ | Gumbel L-moments trên qmax năm | Ngưỡng 1, 2, 5, 10 năm mỗi trạm | Chưa làm |
| 10. Demo | Bản đồ cảnh báo phát lại kỳ test | Ứng dụng | Chưa làm |

---

## 3. Dữ liệu, bộ nạp và tiền xử lý

### 3.1. Giữ nguyên như bài gốc

- 451 lưu vực mức A (`basins_filtered.txt`), train 2003–2009, validation 2010–2013, test 2014–2017; NeuralHydrology nạp thêm giai đoạn khởi động trước mỗi kỳ.
- Nhãn `qmax` ngày t, bộ nạp đổi sang mm/ngày theo diện tích (`area_gov`); demo đổi ngược về m³/s.
- Hindcast 364 ngày (31 biến tái phân tích), forecast ngày t (5 biến ECMWF HRES), 33 thuộc tính tĩnh.
- z-score toàn cục tính trên train; hàm mất mát NSE* dùng độ lệch chuẩn qmax từng lưu vực.

### 3.2. Nhóm biến theo nguồn dữ liệu

Hindcast chia thành các nhóm theo nguồn, mỗi nhóm một mạng nhúng; masked mean lấy trung bình các nhóm có dữ liệu ở mỗi bước (Gauch và cs., HESS 2025). Nhóm theo nguồn vì dữ liệu mất khi vận hành theo cả nguồn (một nhà cung cấp chậm), không theo từng biến.

| Nhóm | Biến | Độ trễ khi vận hành (nguồn tra) | Ghi chú |
|---|---|---|---|
| ERA5-Land | 21 biến | Khoảng 5 ngày (Copernicus) | |
| E-OBS | 7 biến | Phiên bản mới mỗi 6 tháng (Copernicus CDS) | Có thể thiếu tới vài tháng gần nhất |
| MSWEP | Mưa | Bản NRT trễ khoảng 2–3 giờ (GloH2O) | Bản NRT khác bản lịch sử có hiệu chỉnh trạm — lệch miền không mô phỏng được với dữ liệu hiện có |
| GLEAM | Bốc hơi thực, tiềm năng | Cập nhật mỗi năm (tháng 3–4), kéo dài tới hết năm trước (GLEAM4, Scientific Data 2025) | Khi vận hành có thể thiếu tới khoảng 15 tháng gần nhất |
| Q quan trắc | `qmean`, `qmax` quá khứ (cùng đơn vị mm/ngày với nhãn) | eHYD (Áo) khoảng 2 giờ (BiasCast Mục 3.7) | Nhóm riêng như MF²LSTM |
| Forecast | 5 biến ECMWF HRES ngày t | Phát hành 00 UTC ngày t | Mạng nhúng riêng cho pha forecast |

### 3.3. Sửa bộ nạp

| Việc | Lý do | Cách làm |
|---|---|---|
| Sửa `_validate_samples` để giữ mẫu có đầu vào thiếu khi dùng masked mean | Bản gốc loại mẫu nếu bất kỳ cột nào thiếu ở bất kỳ ngày nào trong 365 ngày (`basedataset.py` dòng 861–919), kể cả khi bật `nan_handling_method` | Khi có `nan_handling_method`, chỉ yêu cầu nhãn và thuộc tính tĩnh hợp lệ; vẫn loại mẫu không đủ lịch sử |
| Dựng thư mục có Q (thay `LamaH_expanded_q_input` không công bố) | Biến thể có Q của bài đọc thư mục này | Chép `qmean`, `qmax` từ `D_gauges` vào tệp khí tượng mức A; đổi sang mm/ngày theo `area_gov` như nhãn |
| Báo cáo số mẫu hợp lệ từng cấu hình | Biết cải tiến lấy lại bao nhiêu mẫu | Ghi số mẫu train trước và sau sửa bộ lọc, với hindcast 365 và 730 ngày |

### 3.4. Che dữ liệu khi huấn luyện

Bản fork đã có `nan_step_probability`, `nan_sequence_probability` (`basedataset.py` dòng 180, 207–229): khi huấn luyện, mỗi nhóm biến bị đặt NaN ngẫu nhiên từng bước và cả chuỗi, không bao giờ che hết mọi nhóm. Xác suất dùng chung cho mọi nhóm; muốn xác suất riêng cho nhóm Q phải sửa nhỏ. Hai giá trị cần thử: 0,1/0,12 (Gauch và cs. 2025, che khí tượng) và 0,05/0,05 (MF²LSTM, che Q).

---

## 4. Kiến trúc mô hình

### 4.1. Khung chung

| Khối | Đầu vào → đầu ra | Ghi chú |
|---|---|---|
| Nhúng hindcast | Mỗi nhóm nguồn → d chiều; masked mean qua các nhóm; nối nhúng thuộc tính tĩnh | Lớp tuyến tính như cấu hình tốt nhất của bài (16 chiều + 16 chiều tĩnh) |
| Nhúng forecast | 5 biến ECMWF → d chiều; nối nhúng tĩnh | Cùng số chiều với hindcast |
| Lõi thời gian | Chuỗi 365 bước (364 hindcast + 1 forecast) → biểu diễn bước cuối | Thay được: LSTM, GRU, Transformer, S4D, Mamba |
| Đầu ra | Bước cuối → qmax ngày t | Hồi quy hoặc CMAL (bước F) |

Đề tài viết một mô hình chung "Sequential Forecast" nhận lõi bất kỳ: nối nhúng hindcast và forecast theo thời gian rồi chạy lõi một lần. Với lõi LSTM, cách này tương đương đúng Sequential Forecast LSTM của bài (LSTM chạy hết hindcast rồi tiếp tục trạng thái sang forecast); kiểm bằng cách nạp trọng số của tác giả và so đầu ra (Mục 9.2). Bài cũng cho thấy trạng thái chạy liên tục bền hơn mạng handoff (435 so với 361 lưu vực cải thiện), nên không dùng Encoder–Decoder cho các lõi mới.

### 4.2. Các lõi thời gian

| Lõi | Cài đặt | Vai trò |
|---|---|---|
| LSTM | `nn.LSTM`, forget bias 3 như bài | Mốc chính |
| GRU | `nn.GRU` | Mốc hồi quy nhẹ hơn |
| Transformer | Bộ mã hóa pre-LayerNorm, mặt nạ nhân quả, mã hóa vị trí sin, lấy đầu ra bước cuối; khởi điểm từ `transformer.py` của NeuralHydrology | Mốc attention; Liu và cs. (2024) cho thấy bản cơ bản thua LSTM |
| S4D | Khối S4D (Gu và cs., NeurIPS 2022) có kết nối tắt và LayerNorm; siêu tham số khởi điểm theo S4D-FT (d_state, min_dt 0,01, max_dt 0,1, lr riêng cho tham số SSM) | Đối chứng SSM bất biến thời gian — tách "Mamba không hợp" khỏi "SSM không hợp" |
| Mamba | Khối `mamba_ssm.Mamba` có kết nối tắt và LayerNorm, đầu vào dạng (batch, length, dim); dự phòng `mambapy` (PyTorch thuần) nếu không cài được kernel CUDA | Lõi chính của đề tài |
| DLinear | Tách xu hướng – dao động trên cửa sổ, hai lớp tuyến tính (Zeng và cs., AAAI 2023) | Mốc tuyến tính, không phải lõi so sánh |

Lớp `mamba.py` có sẵn của NeuralHydrology quét sai trục (`02_BasePaper.md` Mục 11.5) nên không dùng.

### 4.3. Biến thể cải tiến (bước E)

| Biến thể | Mô tả | Căn cứ | So với |
|---|---|---|---|
| Mamba hai chiều | Thêm nhánh quét ngược trên toàn cửa sổ 365 bước rồi gộp với nhánh xuôi; đầu ra chỉ ở bước cuối, mọi đầu vào đều ≤ ngày t nên không rò rỉ | ResBi-Mamba Plus (AISE 2026); S-Mamba, CMMamba dùng Mamba hai chiều | Mamba thuần cùng tham số |
| Transformer patch | Chia hindcast thành đoạn (patch) làm token như PatchTST | Nie và cs. (ICLR 2023); Zhang và cs. (2026): PatchTST tốt nhất khi có Q | Transformer thuần cùng tham số |
| Dự phòng: Transformer nhúng tích chập nhân quả | Hai lớp Conv1D nhân quả có kết nối tắt trước attention | Liu và cs. (J. Hydrol. 2024) | Transformer thuần |
| Dự phòng: S5D | Thêm Conv1D, LayerNorm, Softsign vào S4D | Jing và cs. (ESWA 2026) | S4D thuần |

### 4.4. Đầu ra xác suất (bước F)

Đầu CMAL (có sẵn `head: cmal`), hàm mất mát log-likelihood; dự báo điểm lấy trung vị phân phối; xác suất vượt ngưỡng lũ cho demo. Căn cứ: Klotz và cs. (HESS 2022) — CMAL tốt nhất trong bốn cách.

---

## 5. Huấn luyện và giao thức so sánh

### 5.1. Cấu hình chung (giữ như bài)

NSE*, Adam lr 10⁻³, CosineAnnealing 30 epoch (nhỏ nhất 10⁻⁵), batch 256, cắt gradient 1, dropout đầu ra 0,3, dừng sớm theo trung vị NSE validation (patience 5, cải thiện tối thiểu 0,005, đã sửa lỗi lưu mô hình tốt nhất). Mỗi epoch ghi NSE trung vị validation và test (yêu cầu GVHD); checkpoint chọn theo validation.

### 5.2. Giao thức so sánh lõi

| Bước | Cách làm |
|---|---|
| Mức cùng tham số | Mỗi lõi chỉnh số chiều để tổng tham số lệch không quá 5% so với LSTM cải tiến (khoảng 85 nghìn tham số với 128 chiều) — tránh nhiễu do khác dung lượng, như Shahriar (2026) |
| Tinh chỉnh cùng ngân sách | Mỗi lõi 8 cấu hình, 1 hạt giống, chọn theo validation: 2 mức dung lượng (cùng tham số, gấp khoảng 3 lần) × 2 độ sâu (1, 2 lớp) × 2 learning rate (10⁻³, 5×10⁻⁴) |
| Chạy chính thức | Cấu hình cùng tham số và cấu hình tốt nhất của mỗi lõi chạy 3 hạt giống |
| Báo cáo | Từng hạt giống, trung bình ± độ lệch chuẩn, tổ hợp (trung bình dự báo 3 hạt giống); số tham số, thời gian mỗi epoch, bộ nhớ GPU |

### 5.3. Hạt giống và tổ hợp

3 hạt giống cho mọi cấu hình chính thức. Nhiễu giữa hạt giống đo ở bước A dùng làm thước đo: chênh lệch nhỏ hơn nhiễu này không được kết luận.

---

## 6. Đánh giá

### 6.1. Chỉ số chính và kiểm định

- **Theo lưu vực trên test 2014–2017:** NSE, KGE (và ba thành phần r, α, β), **PNSE** (NSE lấy qmax ngày t−1 quan sát làm mốc, MF²LSTM; tương đương kỹ năng so với persistence).
- **Tổng hợp:** trung vị, trung bình, P10/P25/P75/P90, đường CDF, số lưu vực tốt lên / kém đi.
- **Kiểm định:** Wilcoxon signed-rank hai phía, ghép cặp theo lưu vực, kèm cỡ hiệu ứng Cohen's d — cách của Kratzert và cs. (2019, 2024), Gauch và cs. (2025), Nearing và cs. (2024); hiệu chỉnh Holm khi so nhiều cặp.

### 6.2. Theo mức lưu lượng và theo sự kiện (yêu cầu GVHD)

| Bước | Nội dung |
|---|---|
| Ngưỡng mức | Riêng từng lưu vực, tính trên qmax 1981–2013 (trước kỳ test): bốn mức theo xác suất vượt của đường duy trì lưu lượng — rất cao (0–0,02), cao (0,02–0,2), trung bình (0,2–0,7), thấp (0,7–1) (đoạn của Yilmaz, Gupta, Wagener, WRR 2008) |
| Chỉ số theo mức | Ma trận nhầm lẫn 4 mức, độ chính xác, recall từng mức; %BiasFHV, %BiasFMS, %BiasFLV |
| Ngưỡng lũ | Chu kỳ lặp lại 1, 2, 5, 10 năm (như Nearing và cs., Nature 2024), Gumbel fit bằng L-moments trên qmax năm 1981–2013 (như AIFL) |
| Sự kiện | Một sự kiện tính là phát hiện đúng nếu dự báo và quan trắc cùng vượt ngưỡng trong cửa sổ ±1 ngày (Nearing dùng 2 ngày với dữ liệu nhiều ngày lead) |
| Chỉ số sự kiện | Tỉ lệ phát hiện (POD/recall), tỉ lệ báo động nhầm (FAR), precision, F1, CSI; sai số thời điểm đỉnh |
| Ngưỡng kép (phân tích độ nhạy) | Ngưỡng của mô hình tính trên chính chuỗi dự báo của mô hình (Nearing 2024, AIFL); vì mô hình chỉ có dự báo 2003–2017 (gồm cả kỳ train), chỉ dùng làm phân tích phụ, nêu rõ hạn chế |
| Cách đọc | Ưu tiên bỏ sót ít ở mức lũ và mức rất thấp, chấp nhận báo động nhầm nhiều hơn (GVHD); so các lõi ở từng mức |

### 6.3. Kịch bản vận hành (chỉ suy luận)

| Kịch bản | Mô tả |
|---|---|
| S0 | Đủ dữ liệu (như bài) |
| S1 | Mất Q 1, 3, 7 ngày cuối |
| S2 | Mất Q toàn cửa sổ (trạm hỏng dài hạn) |
| S3 | Độ trễ thực tế: ERA5-Land thiếu 5 ngày cuối; E-OBS thiếu 180 ngày cuối; GLEAM thiếu toàn cửa sổ; MSWEP đủ (giả định bản NRT tương đương bản lịch sử); Q đủ tới t−1 |
| S4 | S3 + mất Q 1 ngày cuối |

Mô hình không có masked mean không chạy được với giá trị thiếu: dùng cách điền đơn giản (giá trị gần nhất, khí hậu ngày) làm mốc so sánh. Báo cáo NSE theo từng kịch bản và mức suy giảm so với S0 cho mỗi lõi.

### 6.4. Theo lưu vực

ΔNSE giữa Mamba (và S4D) với LSTM cải tiến; bản đồ; tương quan Spearman với 33 thuộc tính và khoảng cách Wasserstein tái phân tích – dự báo (mã phân tích của tác giả). Kiểm giả thuyết từ Wang và cs. (WRR 2025), Jing và cs. (ESWA 2026), Zhang và cs. (2026): SSM có lợi ở lưu vực tuyết, động lực chậm, bất lợi ở lũ nhanh — dùng tỷ lệ tuyết, độ cao, chỉ số khô hạn, diện tích. Tách nhóm đầu nguồn (72,5%) và lồng nhau (27,5%).

### 6.5. Chi phí tính toán

Số tham số, thời gian mỗi epoch, bộ nhớ GPU, thời gian suy luận cho 365 và 730 ngày — kiểm lập luận "Mamba tuyến tính theo độ dài chuỗi" trên bài toán thật.

### 6.6. Giải thích mô hình

Integrated Gradients (Captum) cho LSTM cải tiến và Mamba tốt nhất: mức ảnh hưởng theo ngày trong cửa sổ (độ dài nhớ hiệu dụng của từng lõi) và theo nhóm nguồn (tái phân tích, dự báo ECMWF, Q). Khác Demiray & Demir (SHAP, Iowa): tách đóng góp của dự báo thời tiết thật và Q quan trắc, so sánh độ dài nhớ giữa hai lõi.

---

## 7. Ma trận thí nghiệm theo bước

Số lần huấn luyện ghi dạng "cấu hình × hạt giống". Mỗi bước có điều kiện chuyển bước.

| Bước | Thí nghiệm | Số lần | Điều kiện chuyển bước |
|---|---|---|---|
| 0 | Hạ tầng: fork riêng, wheel `mamba-ssm`, kiểm thử đơn vị (Mục 9.2), đo thời gian 1 epoch mỗi lõi | 0 (chạy thử ngắn) | Mọi kiểm thử qua; có bảng thời gian |
| A1 | Khảo sát dữ liệu; ngưỡng mức và ngưỡng lũ tính trên 1981–2013 | 0 | Có tệp ngưỡng |
| A2 | Chạy lại trọng số tác giả (Sequential có/không Q, Encoder–Decoder có Q, baseline chỉ dự báo) | 0 | NSE trung vị lệch số của tác giả không quá 0,005 |
| A3 | Persistence (xong), PNSE, DLinear | DLinear 1 × 3 | |
| A4 | Huấn luyện lại Sequential LSTM có Q theo cấu hình tác giả — đo nhiễu hạt giống (B0) | 1 × 3 | Có độ lệch chuẩn giữa hạt giống |
| B1 | B0 + sửa lỗi lưu mô hình tốt nhất + Q cùng đơn vị mm/ngày + thêm `qmax` quá khứ vào nhóm Q | 1 × 3 | |
| B2 | B1 + nhúng theo nguồn, masked mean + sửa bộ lọc mẫu | 1 × 3 | |
| B3 | B2 + che dữ liệu khi huấn luyện, hai mức xác suất (0,1/0,12 và 0,05/0,05) | 2 × 3 | Chọn mức tốt hơn theo validation và theo S1–S4 |
| B* | Chọn "LSTM cải tiến" theo validation; nếu B2, B3 không hơn B1 trên validation thì giữ cấu hình đơn giản hơn, nhưng vẫn báo cáo độ bền ở S1–S4 | 0 | |
| C1 | Tinh chỉnh 5 lõi (LSTM, GRU, Transformer, S4D, Mamba) trên quy trình B* | 5 × 8 × 1 | |
| C2 | Chạy chính thức: cấu hình cùng tham số và cấu hình tốt nhất của mỗi lõi | tối đa 5 × 2 × 3 | |
| D | Hindcast 730 ngày cho LSTM, Mamba, S4D (cấu hình tốt nhất) | 3 × 3 | |
| E | Mamba hai chiều; Transformer patch (dự phòng: Transformer tích chập, S5D) | 2 × 3 (+2 × 3) | Chỉ làm nếu C xong |
| F | CMAL cho LSTM cải tiến và lõi SSM tốt nhất | 2 × 3 | |
| GV | Thí nghiệm kiểm chứng gợi ý GVHD: lấy mẫu nhiều ngày mức cao (Y7) trên mô hình tốt nhất, đánh giá bằng Mục 6.2 | 1 × 3 | Bằng chứng trái chiều (Martel 2025) nên trình bày là kiểm chứng |
| Đánh giá | Mục 6.1–6.5 cho mọi cấu hình chính thức | 0 | |
| G | Giải thích mô hình, ngưỡng lũ, demo | 0 | |

Tổng tối đa 112 lần huấn luyện (A: 6; B: 12; C: 40 + 30; D, E, F, GV: 9 + 6 + 6 + 3), ít hơn nếu cấu hình tốt nhất của một lõi trùng cấu hình cùng tham số, thêm 12 nếu làm biến thể dự phòng.

---

## 8. Ngân sách tính toán, rủi ro và thứ tự cắt

### 8.1. Ngân sách

- Từ 3 tài khoản Kaggle trở lên, mỗi tài khoản 30 giờ GPU/tuần, tối đa 12 giờ/phiên — khoảng 90 giờ/tuần. Chỉ dùng GPU T4 (`mamba-ssm` cần sm_75 trở lên; P100 không chạy được).
- Thời gian mỗi lần trên T4 chưa đo (bài: vài phút tới khoảng 1 giờ trên RTX 4090). Bước 0 đo thật rồi lập lịch; ví dụ trung bình 45 phút/lần thì 112 lần cần khoảng 84 giờ GPU, tức khoảng một tuần với 3 tài khoản.
- Mỗi tài khoản chạy một nhóm thí nghiệm; trọng số và nhật ký lưu thành Kaggle Dataset để đánh giá tập trung.

### 8.2. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Không cài được `mamba-ssm` trên Kaggle | Đã có notebook công khai build `mamba_ssm` 2.3.1 cho Python 3.12 trên Kaggle; dự phòng `mambapy` (PyTorch thuần, quét song song, chậm hơn) |
| Mamba quá khớp sớm (như Shahriar 2026) | Dừng sớm theo validation, dropout, theo dõi NSE validation và test mỗi epoch |
| Lõi mới học kém do siêu tham số của LSTM | Tinh chỉnh cùng ngân sách có learning rate (Mục 5.2) |
| Masked mean, che dữ liệu làm giảm NSE ở S0 | Vẫn báo cáo; chọn B* theo validation; lợi ích ở kịch bản S1–S4 |
| Ngưỡng lũ 10 năm có ít sự kiện trong 4 năm test | Báo cáo số sự kiện; kết luận chính dựa trên 1 và 2 năm |
| Thời gian chạy vượt dự kiến | Cắt theo Mục 8.3 |

### 8.3. Thứ tự cắt nếu thiếu thời gian

Cắt từ trên xuống, không ảnh hưởng RQ2–RQ4: (1) biến thể dự phòng ở bước E; (2) thí nghiệm GV; (3) giải thích mô hình; (4) biến thể chính ở bước E; (5) 730 ngày cho S4D; (6) tinh chỉnh giảm từ 8 xuống 4 cấu hình mỗi lõi.

---

## 9. Hạ tầng và tổ chức mã

### 9.1. Khung và nơi chạy

- Fork riêng của bản fork NeuralHydrology của tác giả (`conestone/neuralhydrology`), sửa trên nhánh riêng; notebook Kaggle cài bằng `pip install git+…`. Bản gốc trong `PaperResearch/PaperResearchCode/` giữ nguyên để đối chiếu. Tên tài khoản và nhánh: chưa chốt.
- Kaggle chính, Colab dự phòng; mã chạy được trên cả hai.

### 9.2. Kiểm thử đơn vị trước khi huấn luyện

| Kiểm thử | Mục đích |
|---|---|
| Mô hình chung với lõi LSTM nạp trọng số tác giả cho đầu ra trùng Sequential Forecast LSTM (sai khác dưới 10⁻⁵) | Bảo đảm khung chung đúng |
| Mamba nhận tensor (batch, length, dim) | Tránh lỗi trục như lớp có sẵn |
| Không rò rỉ: đổi đầu vào sau ngày t không đổi dự báo | Kiểm mặt nạ nhân quả, quét hai chiều |
| Bộ lọc mẫu mới giữ mẫu có NaN đầu vào, loại mẫu thiếu nhãn | Kiểm sửa `_validate_samples` |
| Masked mean với một nhóm bị che cho kết quả hữu hạn | Kiểm đường đi NaN |
| Số tham số các lõi ở mức cùng tham số lệch không quá 5% | Kiểm giao thức Mục 5.2 |

### 9.3. Thư mục mã

Mã đặt trong `Workspace/`, mỗi thư mục con một việc, tệp đánh số theo thứ tự chạy; mỗi notebook một tệp `.py` tự đủ, có mục lục, chia cell theo "Phần".

| Thư mục | Việc | Trạng thái |
|---|---|---|
| `01_Download/` | Tải dữ liệu | Xong |
| `02_Exploration/` | Khảo sát dữ liệu, ngưỡng | Chưa viết |
| `03_Baseline/` | Tái lập, persistence, DLinear, nhiễu hạt giống | Chưa viết |
| `04_Setup/` | Cài môi trường, wheel `mamba-ssm`, kiểm thử đơn vị, đo thời gian | Chưa viết |
| `05_Training/` | Notebook huấn luyện theo bước B–F | Chưa viết |
| `06_Evaluation/` | Bộ đánh giá Mục 6 | Chưa viết |
| `07_Demo/` | Demo | Chưa viết |

Tên thư mục từ `04_` trở đi là đề xuất, đặt khi bắt đầu viết mã.

---

## 10. Đối chiếu với pipeline của BiasCast

| Giai đoạn | BiasCast | Đề tài | Giống / khác |
|---|---|---|---|
| Dữ liệu | ERA5-Land, E-OBS, MSWEP, GLEAM, ECMWF HRES theo lưu vực | Cùng dữ liệu Zenodo | Giống |
| Bộ nạp | Loại mẫu có bất kỳ giá trị thiếu nào; Q đầu vào khác đơn vị nhãn | Sửa bộ lọc; nhóm theo nguồn; Q cùng đơn vị | Sửa lỗi, mở rộng |
| Mô hình | Chỉ LSTM (tác giả tự nêu hạn chế, Mục 3.7) | LSTM, GRU, Transformer, S4D, Mamba trong khung chung | Khoảng trống chính |
| Huấn luyện | NSE*, tối ưu Bayes, 1 hạt giống | Giữ NSE* và lịch học; tinh chỉnh cùng ngân sách; 3 hạt giống | Khác số hạt giống, giao thức tinh chỉnh |
| Đánh giá | NSE, KGE; cận trên, cận dưới | Thêm persistence, PNSE, theo mức, theo sự kiện lũ, kịch bản vận hành, kiểm định | Mở rộng |
| Tái lập, XAI, ngưỡng lũ, demo | Không có | Có | Chỉ có ở đề tài |

---

## 11. Khóa luận

| Hướng | Lý do để sau |
|---|---|
| Ghép dữ liệu giờ vào quá khứ gần (MF-LSTM) | Cần bộ nạp hai độ phân giải; dữ liệu giờ đã có (`lamah-ce-core`, `lamah-ce-extra`) |
| Dự báo nhiều ngày | Extended LamaH-CE chỉ có dự báo cho ngày t (chưa xác nhận trên dữ liệu); cần nguồn dự báo nhiều lead |
| Hàm mất mát ưu tiên đỉnh lũ | Có đánh đổi với kỹ năng chung (Baste 2025; Talbot & Davenport, preprint 2026) |
| Đồ thị mạng sông | Đồ thị thô không tự cải thiện (Kirschstein & Sun 2024); GAT, reachability có bằng chứng (Mosaffa 2026, Wang 2025) |
| Biến thể lai Mamba–Transformer (SST), LOAN, xLSTM | Sau khi có kết quả dạng thuần |
| Phần mềm ứng dụng | Yêu cầu của khóa luận |

---

## 12. Việc chưa chốt

| Việc | Ghi chú |
|---|---|
| Tài khoản và nhánh của fork NeuralHydrology | Cần trước bước 0 |
| Công nghệ và nơi triển khai demo | Cần trước bước G |
| Tên nội bộ tiểu luận | `01_OverallPlan.md` Mục 1 |

---

## 13. Căn cứ chính cho các quyết định

| Quyết định | Căn cứ |
|---|---|
| Masked mean theo nhóm nguồn, che dữ liệu khi huấn luyện | Gauch và cs., HESS 29:6221–6235 (2025); MF²LSTM (EGUsphere 2026); mã fork (`basedataset.py`, `inputlayer.py`) |
| Độ trễ từng nguồn | Copernicus (ERA5, E-OBS mỗi 6 tháng); GloH2O (MSWEP NRT 2–3 giờ); GLEAM4 (Scientific Data 2025, cập nhật mỗi năm); BiasCast Mục 3.7 (eHYD 2 giờ) |
| Giao thức cùng tham số | Shahriar (SSRN 2026); Liu và cs. (J. Hydrol. 2024) |
| PNSE | MF²LSTM (EGUsphere 2026) |
| Chu kỳ lặp lại 1, 2, 5, 10 năm; precision, recall, F1; cửa sổ thời gian | Nearing và cs., Nature 627 (2024) |
| Gumbel L-moments | AIFL (Taccari và cs., J. Hydrol. 2026) |
| Đoạn đường duy trì lưu lượng | Yilmaz, Gupta, Wagener, WRR 44 (2008) |
| Wilcoxon signed-rank + Cohen's d | Kratzert và cs. (2019, 2024); Gauch và cs. (2025); Nearing và cs. (2024) |
| Siêu tham số khởi điểm S4D | Wang và cs., WRR 2025, phụ lục S4D-FT |
| PatchTST, DLinear | Nie và cs. (ICLR 2023); Zhang và cs. (J. Hydrol. 2026) |
| Trạng thái chạy liên tục thay vì handoff | BiasCast Mục 3.6 |
