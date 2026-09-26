# Báo cáo đánh giá các bài báo ứng viên làm bài cơ sở

> Ngày lập: 26/9/2026. Tài liệu này tổng hợp kết quả đọc toàn văn, đối chiếu mã nguồn, chạy thử và tra cứu các công trình trích dẫn đối với các bài báo ứng viên làm bài cơ sở cho đề tài *"Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy"*. Chi tiết đọc mã nguồn nằm ở `CHECKCODE.md`; số liệu chi tiết các ứng viên khác ở Phụ lục A; danh sách đầy đủ các bài ở `RESEARCHING.md` (số `#N`). Tệp PDF lưu tại thư mục `PDF/` (không đưa lên git).

## Tóm tắt

- **Xếp hạng:** 1. **#104** (Kirschstein & Sun, ICML 2024) · 2. **#67** (Acuña Espinoza và cs., HESS 2025) và **#22** (Wang và cs., WRR 2025) đồng hạng · 4. #103 · 5. #71/#54 · 6. #53 · 7. #42 · 8. #6 · 9. #36 · 10. #135 · 11. #38. Các bài #122, #123, #124 không có mã nguồn, chỉ dùng để trích dẫn; MTPre chưa xuất bản (đánh giá ở Mục 5.10).
- **Bài cơ sở đã chốt (26/9/2026): #104** — bài gốc là bài toán dự báo, mô hình chưa có thành phần học chuỗi thời gian (chỗ cải tiến bằng Mamba rõ ràng), cho phép xây dựng kiến trúc không gian–thời gian (Mamba + GNN), có dữ liệu mạng sông để làm XAI theo không gian và demo bản đồ.

---

## 1. Mục đích, phạm vi và phương pháp

**Mục đích.** Chọn một bài báo đã công bố làm cơ sở để cải tiến kiến trúc (thêm/thay bằng Mamba) và so sánh với các kiến trúc học sâu khác, đồng thời xác định rõ phạm vi công việc cho tiểu luận.

**Phạm vi.** Các ứng viên được lọc từ `RESEARCHING.md` (137 bài từ 2024): bài có mã nguồn, dữ liệu công khai, huấn luyện mô hình học sâu cho lưu lượng. Ngoài danh sách ban đầu (lọc theo CAMELS-US ngày), đã lọc lại theo dữ liệu **theo giờ** và bổ sung hai bài #67, #103.

**Phương pháp.**
1. Đọc toàn văn từng bài theo cấu trúc thống nhất: thông tin xuất bản, bài toán, dữ liệu và giao thức, mô hình, huấn luyện và đánh giá, kết quả, hạn chế (tác giả tự nêu và nhận xét của nhóm), hướng cải tiến. Số liệu bảng đối chiếu với ảnh trang PDF.
2. Đọc mã nguồn, đối chiếu với mô tả trong bài (`CHECKCODE.md`).
3. Chạy thử mã trên CPU trong môi trường riêng (PyTorch 2.14 CPU) với dữ liệu giả hoặc đầu vào ngẫu nhiên để kiểm tra khả năng chạy.
4. Tra các công trình trích dẫn qua OpenAlex và Semantic Scholar để kiểm tra tính mới của hướng cải tiến.
5. Chấm điểm theo bộ tiêu chí ở Mục 2 và xếp hạng.

---

## 2. Tiêu chí đánh giá

| Nhóm | Tiêu chí | Nội dung kiểm tra |
|---|---|---|
| **A — quyết định** | A1. Phù hợp đề tài | Là bài toán **dự báo** lưu lượng (dùng dữ liệu tới thời điểm t, có Q quá khứ, dự báo Q sau t) bằng học sâu thuần. Bài mô phỏng hoặc mô hình lai bị trừ điểm |
| | A2. Cải tiến được | Có chỗ chèn Mamba tự nhiên; có động cơ lấy từ hạn chế do chính bài nêu; chưa có công trình làm; đo được cải thiện trong khung so sánh công bằng; có hướng cải tiến phụ |
| **B — điều kiện cần** | B1. Mã nguồn, tái lập | Mã công khai, chạy được, số liệu tái lập được |
| | B2. Dữ liệu | Công khai, miễn phí, đủ lớn cho học sâu |
| | B3. Tài nguyên tính toán | Chạy được trên Google Colab (người dùng có Colab Pro) hoặc Kaggle (GPU T4; 12 giờ/phiên, 30 giờ/tuần theo tài liệu Kaggle). `mamba-ssm` biên dịch cho kiến trúc sm_75 trở lên nên chạy được trên T4, không chạy trên P100 (sm_60) |
| | B4. Độ dài dữ liệu và chuỗi | Dữ liệu dài nhiều năm, chuỗi đầu vào đủ dài để thể hiện ưu thế của Mamba |
| **C — điểm cộng** | C1. Độ tin cậy phương pháp | Chia tập theo thời gian, có validation độc lập, NSE tính theo trạm, nhiều hạt giống |
| | C2. Uy tín | Nơi công bố, số trích dẫn |
| | C3. Hỗ trợ demo | TLCN bắt buộc có demo; KLTN phải thành phần mềm ứng dụng |
| | C4. Khung so sánh nhiều kiến trúc | Dễ đặt Mamba cạnh LSTM/GRU/Transformer trong cùng điều kiện |
| | C5. Mở rộng sang bộ dữ liệu khác | Cách làm chuyển được sang bộ dữ liệu thứ hai |
| | C6. Rủi ro cải tiến không thắng | Baseline gốc mạnh tới đâu, còn dư địa hay không |

Venue CNTT/AI là điểm cộng đối với bài cơ sở (bắt buộc đối với phần khảo sát tài liệu). Độ khó của mã nguồn không được xem là tiêu chí.

---

## 3. Tài liệu đã sử dụng

| # | Bài | Tài liệu | Mức độ đọc |
|---|---|---|---|
| #104 | Kirschstein & Sun — ICML 2024 | `PDF/104_ICML2024_RiverTopology.pdf` | Toàn văn + toàn bộ mã + kết quả tác giả nộp |
| #67 | Acuña Espinoza và cs. — HESS 2025 | `PDF/67_HESS2025_MF-LSTM.pdf` | Toàn văn + mã bản Zenodo của bài + thư viện Hy2DL |
| #22 | Wang và cs. — WRR 2025 | `PDF/22_S4D_FT_ChinhThuc.pdf`, bản arXiv v1, Supporting Information | Toàn văn + SI + toàn bộ mã |
| #103 | Wang, Chen, Zheng, Song — npj Natural Hazards 2025 | Toàn văn HTML truy cập mở (nature.com) | Toàn văn + mã |
| #54 | McEachran và cs. — WRR 2025 | `PDF/54_WRR2025_FHNN.pdf` | Toàn văn |
| #71 | Ghosh và cs. — IEEE ICDM 2025 | `PDF/71_ICDM2025_arXiv-v2.pdf` (arXiv 2407.20152v2) | Toàn văn bản arXiv + mã |
| #53 | Cheng và cs. — Machine Learning: Earth 2026 | `PDF/53_ML_Earth.pdf` | Toàn văn + mã |
| #42 | Zhang và cs. — IEEE TGRS 2025 | `PDF/42_TGRS2025_TFRN.pdf` (GVHD tải) | Toàn văn + mã |
| #6 | Liu, Shen và cs. — HESS 2025 | `PDF/06_HESS2025_RNNs-to-Transformers.pdf` | Toàn văn + mã |
| #36 | Jing và cs. — ESWA 2026 | `PDF/36_ESWA2026_SSM.pdf` (GVHD tải) | Toàn văn + các tệp mã bài dùng |
| #135 | Sun & Sun — Machine Learning: Earth 2026 | `PDF/135_ML_Earth.pdf` | Toàn văn + mã |
| #38 | Ouyang, Deng, Ni — WRR 2026 | `PDF/38_WRR_2026.pdf` | Toàn văn + mã |
| #124 | Sheng và cs. — Neural Networks 2026 | `PDF/124_NN2026_Columbia.pdf` (GVHD tải) | Toàn văn (không có mã) |
| #122 | Zhou và cs. — EAAI 2026 | `PDF/122_EAAI2026_LamaH.pdf` (GVHD tải) | Toàn văn (không có mã) |
| #123 | Wang và cs. — ESWA 2025 | `PDF/123_ESWA2025_Iowa.pdf` (GVHD tải) | Toàn văn (không có mã) |
| #120 | Mosaffa và cs. — HESS 2026 | `PDF/Mosaffa_HESS2026_LSTM-GNN.pdf` | Phương pháp và kết quả |
| — | HydroDiffusion — arXiv 2512.12183 | `PDF/HydroDiffusion_arXiv2512.12183.pdf` | Phương pháp |
| — | MTPre (Song, Chae, Chung) | Chưa xuất bản; chỉ có mã trên Zenodo `19367140` | README + mã |

Chưa tải: Supplement của #6, #38, #54; bảng bổ sung S1–S5 của #135; SI của #36 (không tải được).

---

## 4. Ba ứng viên chính

### 4.1. #104 — The Merit of River Network Topology for Neural Flood Forecasting

**Thông tin xuất bản.** Nikolas Kirschstein, Yixuan Sun (University of Oxford; Technical University of Munich). Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, tr. 24713–24725 — hội nghị CORE A*, ngành CNTT/AI. Mã nguồn `github.com/nkirschi/neural-flood-forecasting` (đẩy lần cuối 31/03/2025, không có tệp LICENSE).

**Bài toán.** Dự báo lưu lượng theo giờ đồng thời cho toàn bộ mạng trạm đo (hồi quy trên nút đồ thị): từ W giờ gần nhất (lưu lượng + 4 biến khí tượng) của mọi trạm, dự báo lưu lượng sau L giờ. Câu hỏi nghiên cứu: đưa cấu trúc mạng sông vào mô hình bằng mạng nơ-ron đồ thị (GNN) có cải thiện dự báo không.

**Dữ liệu và giao thức.**
- LamaH-CE theo giờ (CC BY 4.0); thành phần liên thông "Danube A" (608/859 trạm), lọc trạm có dữ liệu đầy đủ 2000–2017 còn **358 trạm**; nối lại cạnh khi xóa trạm để giữ liên thông.
- Biến khí tượng: mưa, độ ẩm đất lớp mặt, nhiệt độ không khí, áp suất bề mặt. Chuẩn hóa z-score theo trạm với thống kê 2000–2015.
- Tập kiểm tra 2016–2017; ba cách chọn 8 năm huấn luyện (năm chẵn 2000–2015, năm lẻ 2000–2015, liên tục 2008–2015). Validation là 1/5 số cửa sổ huấn luyện chọn ngẫu nhiên.
- Cửa sổ W = 24 giờ, lead time L = 6 giờ. Quy mô: 358 trạm × 157.800 giờ ≈ 56,5 triệu bước trạm–giờ (≈ 1,1 GB float32).

**Mô hình.** Kiến trúc "bánh kẹp": làm phẳng cửa sổ W giờ × 5 biến rồi đưa qua một lớp affine → N = 19 lớp GNN (bằng đường đi dài nhất của đồ thị) → lớp affine giải mã. Ba loại lớp ResGCN, GCNII, ResGAT; sáu định nghĩa ma trận kề (cô lập, nhị phân, chiều dài sông, chênh cao, độ dốc, học được); ba hướng cạnh.

**Huấn luyện và đánh giá.** Adam, lr 1e-4, 100 epoch, batch 64, L2 1e-5; hàm mất mát MSE nhân "điểm liên quan" (ưu tiên cửa sổ biến động mạnh); chọn trạng thái có loss validation nhỏ nhất; mỗi cấu hình 1 hạt giống × 3 cách chia năm. Chỉ số: NSE có trọng số, trung bình qua các trạm.

**Kết quả.** NSE có trọng số của mọi tổ hợp nằm trong 80,2%–85,6%; không tổ hợp đồ thị nào vượt MLP 2 lớp (85,37% ± 1,64%); bỏ hết cạnh không làm giảm hiệu năng quá độ lệch chuẩn; trọng số cạnh học được không tương quan với trọng số vật lý. Trạm kém nhất có các đỉnh đột ngột, hẹp mà mô hình bỏ lỡ.

**Kết quả kiểm chứng mã nguồn và chạy thử.**
- **Lỗi công thức NSE:** hàm `evaluate_nse` lấy trung bình lưu lượng ở đơn vị gốc (m³/s) trừ cho nhãn đã chuẩn hóa. Mẫu số NSE bị thổi phồng xấp xỉ μ² lần ở trạm có lưu lượng trung bình lớn, đẩy NSE sát 1 (28% trạm có NSE > 0,99 trong tệp kết quả tác giả nộp). Giá trị NSE tuyệt đối của bài vì thế không dùng được. Do mẫu số của mỗi trạm như nhau với mọi mô hình, **thứ tự giữa các mô hình trên cùng trạm được giữ nguyên**, nên kết luận "đồ thị không hơn MLP" nhiều khả năng vẫn đúng.
- Validation chọn ngẫu nhiên trên các cửa sổ chồng lấn nên không độc lập với tập huấn luyện.
- Bộ nạp dữ liệu cắt cửa sổ **bên trong từng năm**: cửa sổ không vượt ranh giới năm, mỗi năm mất W + L mẫu; cửa sổ dài từ 1 năm trở lên không chạy được với cấu trúc hiện tại.
- Kèm theo 162 checkpoint (1,9 GB), đủ để tính lại NSE đúng mà không cần huấn luyện lại. Việc tính lại cần tải 14,8 GB dữ liệu giờ; tốc độ tải trên máy hiện tại (0,37–0,75 MB/s) không phù hợp, nên thực hiện trên Colab.
- Mã gồm khoảng 880 dòng Python và 8 notebook phân tích.

**Tính mới của hướng cải tiến.** 15 công trình trích dẫn (Semantic Scholar) và các công trình trích dẫn #103 không có công trình nào dùng Mamba làm bộ mã hóa thời gian trên LamaH-CE. Công trình gần nhất là Mosaffa và cs. (HESS 2026, #120) ghép LSTM với GNN trên LamaH-CE **theo ngày**, không dùng Q quá khứ (Mục 6).

**Hạn chế.** Tác giả tự nêu: dữ liệu không có nhãn sự kiện lũ và thời gian truyền lũ; GNN khó dự báo đỉnh đột ngột. Nhận xét của nhóm: bộ mã hóa thời gian chỉ là một lớp affine; cửa sổ 24 giờ ngắn (tác giả thừa nhận do giới hạn tính toán); tập kiểm tra 2 năm; chỉ số NSE có trọng số khác chuẩn; lỗi NSE và validation nêu trên.

**Hướng cải tiến cho tiểu luận.**
1. Tính lại NSE đúng từ checkpoint của tác giả để có baseline chính xác.
2. Thay lớp affine bằng Mamba làm bộ mã hóa thời gian cho từng trạm (Mamba → GNN → giải mã), kéo cửa sổ từ 24 giờ lên vài ngày–vài tuần.
3. So sánh Mamba có và không có đồ thị để trả lời lại câu hỏi của bài: khi có bộ mã hóa thời gian tốt, cấu trúc mạng sông có giúp không.
4. Báo cáo thêm NSE chuẩn (không trọng số), FHV để so được với các bài khác; chuyển validation sang chia theo thời gian.

### 4.2. #67 — Technical note: An approach for handling multiple temporal frequencies with different input dimensions using a single LSTM cell (MF-LSTM)

**Thông tin xuất bản.** Eduardo Acuña Espinoza, Frederik Kratzert, Daniel Klotz, Martin Gauch, Manuel Álvarez Chaves, Ralf Loritz, Uwe Ehret (KIT; Google Research). Hydrology and Earth System Sciences 29:1749–1758 (2025), DOI `10.5194/hess-29-1749-2025`, Scimago Q1, CC BY 4.0. Mã nguồn: thư viện Hy2DL (`github.com/eduardoAcunaEspinoza/Hy2DL`, BSD-3, cập nhật 03/09/2026) và bản dùng cho bài trên Zenodo `10.5281/zenodo.14780059`.

**Bài toán.** **Mô phỏng** lưu lượng theo giờ ("simulating hourly discharges") cho 516 lưu vực CAMELS-US; mục tiêu lâu dài của tác giả là hệ thống dự báo vận hành theo giờ.

**Dữ liệu và giao thức.** 11 biến NLDAS theo giờ + 26 thuộc tính tĩnh; lưu lượng giờ USGS. Dữ liệu giờ công khai, CC BY 4.0 (Zenodo `4072701`: NLDAS 17 GB CSV + USGS 1 GB, hoặc tệp NetCDF 19,5 GB). Train 10/1990–9/2003, validation 10/2003–9/2008, test 10/2008–9/2018 (28 năm).

**Mô hình.** Một năm dữ liệu giờ là 8.760 bước; MF-LSTM gộp 351 ngày đầu thành bước ngày, giữ 336 giờ cuối, còn 687 bước đưa vào một ô LSTM, dùng lớp nhúng riêng cho từng tần suất để nhận số biến khác nhau. So sánh với MTS-LSTM, sMTS-LSTM (Gauch 2021) và LSTM chạy thuần theo giờ với 4.320 bước.

**Huấn luyện và đánh giá.** Hidden 128, batch 256, 30 epoch, loss NSE* (Kratzert), tổ hợp 10 hạt giống, test bằng epoch cuối; NSE theo lưu vực, báo cáo trung vị.

**Kết quả.** NSE trung vị 0,75 cho MF-LSTM, MTS-LSTM, sMTS-LSTM và LSTM thuần theo giờ (thí nghiệm cùng số biến); 0,81 cho MF-LSTM và MTS-LSTM khi số biến khác nhau giữa hai tần suất (báo trên kỳ validation). MF-LSTM nhanh hơn LSTM thuần theo giờ khoảng 5 lần; mất ~7 giờ/lần trên V100.

**Kết quả kiểm chứng mã nguồn và chạy thử.**
- Bản mã trên Zenodo (đọc từ xa, chỉ tải các tệp cần) khớp bài: giai đoạn chia tập, `seq_length = 365×24`, `predict_last_n = 24`, siêu tham số, loss NSE*, test epoch cuối, NSE theo lưu vực lấy trung vị. Có đủ 10 hạt giống cho hai thí nghiệm chính.
- Nhật ký huấn luyện (hạt giống 110): 28.608 s ≈ 7,9 giờ/lần; NSE validation cao nhất 0,743 ở epoch 16, giảm còn 0,729 ở epoch 28 (quá khớp nhẹ).
- Thư viện Hy2DL: scaler chỉ tính trên kỳ train; hỗ trợ lưu dữ liệu dạng zarr trên đĩa; loss NSE*, WeightedMSE, NLL, CRPS; thêm mô hình bằng cách đăng ký lớp trong `modelzoo/factory.py`.
- Thư viện có mô hình dự báo `MF2LSTM` (hindcast đa tần suất + dự báo tuần tự, teacher-forcing khi huấn luyện, tự hồi quy khi suy luận, đồng hóa Q quan trắc). Đã chạy thử trên dữ liệu giả đúng định dạng CAMELS-US giờ (2 lưu vực): huấn luyện 1 epoch (loss 49,9 → 22,5), đánh giá ra dự báo lead time 1–24 giờ và tính được NSE, RMSE, PNSE theo lưu vực và lead time. Ràng buộc: MF2LSTM yêu cầu `custom_seq_processing` và `nan_handling_method: masked_mean`; hàm đánh giá cần `forecast_mode=True`. Chế độ dự báo chưa có bài báo công bố số liệu.

**Tính mới.** Các công trình trích dẫn (OpenAlex) không dùng Mamba/SSM.

**Hạn chế.** Tác giả tự nêu: siêu tham số lấy từ Gauch 2021, không chỉnh lại. Nhận xét của nhóm: bài gốc là mô phỏng; chỉ so các biến thể LSTM; LSTM thuần 4.320 giờ đã cho NSE bằng MF-LSTM nên dư địa tăng NSE chưa rõ; dữ liệu lớn (~20 GB); là technical note.

**Hướng cải tiến cho tiểu luận.** Thêm Mamba vào `modelzoo` và chạy trực tiếp chuỗi 8.760 bước giờ, so với MF-LSTM (gộp bước) và LSTM thuần; chuyển sang chế độ dự báo bằng MF2LSTM làm baseline; mở rộng sang các bộ CAMELS khác mà thư viện hỗ trợ.

### 4.3. #22 — A Deep State Space Model for Rainfall-Runoff Simulations (S4D-FT)

**Thông tin xuất bản.** Yihan Wang, Lujun Zhang, Annan Yu, N. Benjamin Erichson, Tiantian Yang (University of Oklahoma; Cornell University; Lawrence Berkeley National Laboratory). Water Resources Research 61(12), e2025WR039888, 2025, DOI `10.1029/2025WR039888`, Scimago Q1, CC BY-NC-ND. Mã nguồn `github.com/Pandas-Paws/S4D_rainfall_runoff_simulations` (commit cuối 09/12/2025, không có LICENSE), Zenodo `10.5281/zenodo.16988503`.

**Bài toán.** **Mô phỏng** lưu lượng ngày theo cách chuỗi-đến-một: từ 365 ngày khí tượng (tính cả ngày hiện tại) và thuộc tính tĩnh, ước lượng lưu lượng của ngày cuối cửa sổ; không dùng lưu lượng quá khứ.

**Dữ liệu và giao thức.** 531 lưu vực CAMELS-US; 5 biến NLDAS + 27 thuộc tính tĩnh. Thiết lập global: train 10/1999–9/2008, test 10/1989–9/1999. Thiết lập PUB: 12 phần lưu vực. Validation: 10% cửa sổ huấn luyện chọn ngẫu nhiên.

**Mô hình.** Phép chiếu đầu vào → chồng lớp S4D (H hệ tuyến tính bất biến theo thời gian, ma trận A đường chéo, Δt học riêng từng kênh, tính bằng tích chập qua FFT) → phép chiếu đầu ra. S4D-FT tỉ lệ lại phần thực và phần ảo của A khi khởi tạo. Siêu tham số (Bảng S3): d_model 128, d_state 128, 6 lớp, α_r = α_i = 10, dropout 0,12, 50 epoch, batch 128, lr 4e-4, lr_min 4e-5, lr_dt 1e-3, Δt ∈ [0,01; 0,1], weight decay 0,03, wd 0,02.

**Huấn luyện và đánh giá.** Adam; tổ hợp 8 thành viên. Chi phí trên GPU L40S: S4D-FT ~10 giờ/hạt giống (4,8 GB), LSTM ~12,5 giờ/hạt giống; PUB gấp 12 lần. Chỉ số NSE, KGE, Pearson-r, FHV, FLV, PBias.

**Kết quả (trung vị 531 lưu vực).**

| Thiết lập | Mô hình | NSE | KGE | Pearson-r | FHV (%) | FLV (%) | PBias (%) |
|---|---|---|---|---|---|---|---|
| Global | LSTM | 0,72 | 0,74 | 0,86 | −17,51 | 10,63 | 5,42 |
| Global | S4D | 0,72 | 0,72 | 0,86 | −18,07 | 16,18 | 5,92 |
| Global | **S4D-FT** | **0,74** | **0,75** | **0,87** | **−16,98** | 20,17 | 5,87 |
| PUB | LSTM | 0,62 | 0,61 | 0,83 | −20,82 | 8,20 | 6,74 |
| PUB | **S4D-FT** | **0,66** | **0,63** | **0,85** | −21,14 | 30,22 | 9,08 |

S4D-FT tốt ở lưu vực khô hạn, tuyết chi phối, ít "flashy"; kém ở lưu vực mưa–tuyết hỗn hợp có đỉnh lũ lớn (Appalachia, Tây Bắc Thái Bình Dương). Supporting Information: với 1 hạt giống, đổi một siêu tham số làm NSE dao động 0,01–0,06 và KGE tới 0,15 — lớn hơn mức hơn LSTM của S4D-FT (NSE +0,02).

**Kết quả kiểm chứng mã nguồn và chạy thử.**
- Script `train_val.sh` khớp Bảng S3 ở d_model, d_state, số lớp, dropout, α, epoch, batch, lr, weight decay; khoảng Δt, lr tham số SSM, lr_dt, wd chạy theo giá trị mặc định khác bảng. Chạy nguyên trạng nhiều khả năng cho kết quả thấp hơn bài.
- `datautils.reshape_data` gán nhãn là Q của ngày cuối cửa sổ (xác nhận bài toán mô phỏng). Chuyển sang dự báo cần sửa hai chỗ: thêm cột `QObs(mm/d)` vào đầu vào trong `datasets.py::_load_data` và dời nhãn sang ngày t+h.
- Dựng mô hình `HOPE` đúng Bảng S3, chạy forward/backward trên CPU với đầu vào 16 × 365 × 32: đầu ra (16, 1), 402.177 tham số. Repo thiếu khai báo `torchvision` và `einops`.

**Tính mới.** Chưa có công trình thay S4D-FT bằng Mamba trên CAMELS. Hai công trình liên quan của cùng nhóm (Mục 6): HydroDiffusion (đã chuyển S4D-FT sang dự báo xác suất 7 ngày) và Block-Biased Mamba (NeurIPS 2025, chỉ ra Mamba kém S4D ở tác vụ phụ thuộc xa, đề xuất B2S6).

**Hạn chế.** Tác giả tự nêu: nhân tích chập toàn cục ưu tiên tần số thấp, làm suy giảm đỉnh lũ ngắn; khó áp ràng buộc bảo toàn khối lượng. Nhận xét của nhóm: cải thiện nhỏ (NSE +0,02), các baseline trích từ bài khác; validation không độc lập; bài toán mô phỏng; SSM bất biến theo thời gian khác với Mamba chọn lọc.

**Hướng cải tiến cho tiểu luận.** So sánh S4D-FT với Mamba và B2S6 trong cùng khung HOPE; đánh giá riêng đỉnh lũ (FHV, các vùng S4D-FT yếu); tách validation theo thời gian; chuyển sang dự báo tham khảo mã HydroDiffusion.

---

## 5. Các ứng viên khác

### 5.1. #103 — Accelerating flood warnings by 10 hours (FloodGNNs)

Wang, Chen, Zheng, Song. npj Natural Hazards 2:45 (2025), DOI `10.1038/s44304-025-00083-6`; mã `github.com/Dreamzz5/FloodGNNs` (MIT). Dự báo nhiều bước (24 giờ quá khứ → 24 giờ tới) trên 358 trạm LamaH-CE (tiền xử lý theo #104), train 2010–2015, test 2016–2017. Đóng góp: giải thích việc đồ thị không giúp trong #104 bằng hiện tượng over-squashing trên cây sông và đề xuất **đồ thị dense theo khả năng tới được** (RBF trên chiều dài sông). Kết quả chỉ trình bày dạng hình: đồ thị dense hơn topo/cô lập và EA-LSTM, rõ hơn ở tầm xa và dòng lớn; NSE 24 giờ của GNN xấp xỉ NSE 14 giờ của EA-LSTM.

Đối chiếu mã: cấu hình import `GCN_Point` nhưng module chỉ có `FloodGNN` (lỗi import khi chạy nguyên trạng); loss MSE trong khi bài ghi MAE; NSE cộng gộp mọi trạm, không trọng số, trong khi bài ghi NSE có trọng số; không có mã tiền xử lý (dữ liệu đã xử lý tải từ Google Drive) và mã EA-LSTM. Mô hình cũng không có thành phần học chuỗi thời gian. **Vai trò:** baseline đồ thị dense trong tiểu luận nếu chọn #104.

### 5.2. #54 và #71 — FHNN

**#54:** McEachran và cs., Water Resources Research 61(11), e2024WR039064 (2025), CC BY-NC. **#71:** Ghosh và cs., IEEE ICDM 2025 (CORE A*), arXiv 2407.20152. Mã chung `github.com/arvindrenga96/FHNN`.

Bài toán dự báo 1–7 ngày có tích hợp lưu lượng quan trắc gần nhất. Bộ mã hóa gồm 3 BiLSTM theo 3 thang thời gian, bộ giải mã LSTM chạy trên khí tượng tương lai (quan trắc). Trên CAMELS 531 lưu vực (#54, train 1985–1993 / val 1993–1995 / test 1995–2005), FHNN hơn LSTM-AR (NSE trung bình ngày 1: 0,792 so với 0,708); trên bản #71 (train 1989–2001 / test 2001–2009), FHNN hơn LSTM-AR, RR-Former, ExoTST và TFT (NSE trung bình 1 bước 0,77). Phần vận hành: FHNN hơn dự báo NWS ở 30/46 trận lũ; 82% dự báo đỉnh của FHNN thấp hơn thực tế.

Hạn chế chính: repo thiếu toàn bộ tiền xử lý (thư mục `DATA/`); hàm cắt cửa sổ trong mã cắt 1 cửa sổ/năm thủy văn, khác mô tả "bước trượt 1 ngày"; batch gán cứng 256; thang thời gian trong mã (14/84 ngày) khác bài (7/30 ngày); hai bài dùng hai giao thức và số liệu khác nhau; trong mã, tầng chậm không nhận [h_m; h_f] và không có MLP như công thức (2). Không tái lập được số liệu. **Vai trò:** tham khảo thiết kế bài toán dự báo (bộ mã hóa quá khứ có Q, bộ giải mã chạy trên khí tượng tương lai).

### 5.3. #53 — HydroTFT

Cheng, Fan, Jia, Xie, Lu. Machine Learning: Earth 2, 025014 (2026), CC BY 4.0; mã `github.com/qxc101/HydroTFT`. Dự báo 1, 7, 14 ngày chỉ dùng khí tượng quá khứ, đặc trưng kỹ thuật và thuộc tính tĩnh; CAMELS 531, giao thức Kratzert. Ở tầm 1 ngày HydroTFT không hơn LSTM có ý nghĩa thống kê (ΔNSE +0,014, p = 0,36); hơn có ý nghĩa ở 7–14 ngày nhưng NSE tuyệt đối thấp (0,17 ở ngày +7).

Đối chiếu mã: chọn epoch có NSE cao nhất trên chính kỳ test; tầm 1 ngày chạy chế độ nowcast với kiến trúc giản lược `VanillaTFT`; ε và chuẩn hóa khác bài; ablation chỉ trên 50 lưu vực. **Vai trò:** tái dùng các đặc trưng kỹ thuật (tổng mưa 90 ngày, độ-ngày, sin/cos).

### 5.4. #42 — TFRN

Zhang và cs., IEEE Transactions on Geoscience and Remote Sensing 63, 4211314 (2025); mã `github.com/redtea-code/TFRN` (MIT). Dự báo 1–7 ngày trên CAMELS (671) và CAMELS-AUS (561), train 1980–1995 / val 1995–2000 / test 2000–2014. Mô hình gồm cổng chọn biến, khối tích chập miền thời gian và tần số, Transformer có AdaLN điều kiện hóa bằng chuỗi phụ cùng kỳ năm trước, bộ mã hóa–giải mã LSTM; cửa sổ chính 15 ngày. CAMELS: NSE 0,853 so với TFT 0,840 và LSTM 0,827.

Đối chiếu mã: NSE tính gộp mọi lưu vực; cấu hình (độ dài chuỗi, tháng của chuỗi phụ) chọn theo số liệu tập test, mỗi bộ dữ liệu một cấu hình; vòng lặp 3 khối DiT chỉ khối cuối có tác dụng; số liệu cùng mô hình lệch nhẹ giữa các bảng. **Vai trò:** ý tưởng chuỗi phụ cùng kỳ + AdaLN có thể ghép với Mamba.

### 5.5. #6 — From RNNs to Transformers

Liu, Shen và cs., Hydrology and Earth System Sciences 29:6811–6828 (2025), CC BY 4.0. Khung thống nhất so sánh LSTM với 12 mô hình (11 Transformer + DLinear) trên 5 loại bài toán. LSTM tốt nhất ở hồi quy (KGE 0,80 trên CAMELS) và dự báo 1 ngày; Transformer vượt ở tự hồi quy tầm xa nhưng mức tuyệt đối thấp. Không có SSM/Mamba. Trong mã, chỉ LSTM dùng loss kiểu NSE, các mô hình attention dùng MSE, lr khác, không cắt gradient. **Vai trò:** khung chạy baseline (kèm dữ liệu CAMELS đã xử lý).

### 5.6. #36 — Benchmarking structured state space models for differentiable parameter learning

Jing, Luo, Yang, Zuo. Expert Systems with Applications 329 (2026) 133040 (Q1, ngành CNTT/AI); mã chính thức `github.com/chooron/dmg-research/tree/master/project/bettermodel`. Mô hình lai dPL: bộ mã hóa học sâu sinh tham số động cho HBV, HBV tính ra Q (mô phỏng). S4D (mã kế thừa từ #22) cho NSE 0,756 so với LSTM 0,742 (531 lưu vực); đề xuất S5D. Báo cáo 5 hạt giống trong repo: NSE trung vị của 7 bộ mã hóa chỉ nằm trong 0,7416–0,7514, vì tầng HBV làm mờ khác biệt giữa các kiến trúc; khi chạy end-to-end (không HBV), S4D 0,751 so với LSTM 0,722. **Vai trò:** tham khảo; không phù hợp làm bài cơ sở vì là mô hình lai, mô phỏng.

### 5.7. #135 — Zero-shot forecasting of streamflow using time series foundation models

Sun & Sun, Machine Learning: Earth 2, 010501 (2026), CC BY 4.0; mã `github.com/dialuser/tsfm_study`. Đánh giá 4 mô hình nền tảng chuỗi thời gian (MOIRAI, Chronos-Bolt, TTM, Sundial) so với LSTM trên CAMELS 531 (ngày, 3 giờ). LSTM tốt nhất ở dự báo 1 ngày (NSE trung vị 0,593), Sundial zero-shot gần bằng. TTM tinh chỉnh đạt NSE trung vị 0,753 ở chế độ nowcast. Mã công bố thiếu phần đa biến và tinh chỉnh. **Vai trò:** bài benchmark, không có kiến trúc để cải tiến.

### 5.8. #38 — A Hierarchical Deep Learning Framework for Runoff Prediction Using Raster-Based Spatial Representations

Ouyang, Deng, Ni. Water Resources Research 62, e2026WR043815 (2026), CC BY. Mô phỏng lưu lượng ngày, câu hỏi nghiên cứu là cách biểu diễn thuộc tính tĩnh (raster + CNN) chứ không phải kiến trúc thời gian. MID-CNN-LSTM NSE 0,73 so với Attr-LSTM 0,70 (16 thuộc tính); lợi thế biến mất ở PUB. Một hạt giống; khác biệt ≤ 0,03 NSE. **Vai trò:** tham khảo giao thức thống kê (bootstrap cặp, Wilcoxon + Benjamini–Hochberg).

### 5.9. Các bài không có mã nguồn

- **#124 STResBiGRU** (Sheng và cs., Neural Networks 205, 2026): dự báo 2–24 giờ cho một chuỗi lưu lượng xả sông Columbia (2016–2019), không có validation, không nêu siêu tham số. Có baseline MambaFormer: tốt nhất ở 2–4 giờ nhưng giảm còn NSE 0,797 ở 24 giờ. Cột R² của Transformer và MambaFormer trùng tuyệt đối với cột NSE ở cả 5 lead time.
- **#122 HMGSTN** (Zhou và cs., EAAI 177, 2026): đồ thị dị thể khí tượng–thủy văn; dữ liệu chính không công khai; trên LamaH-CE (thí nghiệm phụ, 27 trạm) ngang GraphWaveNet. Cột PCC của bảng ablation trùng tuyệt đối với cột PCC của 5 baseline.
- **#123 CTB** (Wang và cs., ESWA 281, 2025): CNN + TCN + BiGRU dự báo 1–120 giờ trên WaterBench-Iowa; làm trơn chuỗi bằng trung bình trượt trước khi chia tập; thua S2S ở 96–120 giờ.

### 5.10. MTPre

Song, Chae, Chung — Mamba-Transformer lai, CAMELS-US 674 lưu vực, mã MIT trên Zenodo `19367140`; bài đang phản biện, chưa xuất bản; chi tiết mã ở `CHECKCODE.md` Mục 12. Đánh giá sơ bộ (chỉ dựa trên README và mã, chưa có toàn văn): điểm mạnh là đã có Mamba, bài toán dự báo 7 ngày, mã MIT; điểm yếu là chưa qua phản biện, pipeline chính dùng EMD tính trên toàn chuỗi trước khi cắt cửa sổ (rò rỉ tương lai, số liệu công bố bị thổi phồng), dùng khí tượng tương lai quan trắc làm đầu vào dự báo, cửa sổ chỉ 15 ngày quá khứ (không khai thác chuỗi dài), và Mamba đã có sẵn nên hướng cải tiến chủ yếu là sửa phương pháp. Nếu được phép dùng bài chưa xuất bản, MTPre xếp khoảng hạng 7 (sau #53, trên #42); nếu không, chỉ dùng làm tham khảo.

---

## 6. Công trình liên quan phát hiện trong quá trình tra cứu

| Công trình | Nội dung | Liên hệ với ứng viên |
|---|---|---|
| **Mosaffa và cs.**, HESS 30:2079–2092 (2026), DOI `10.5194/hess-30-2079-2026`; mã `github.com/hmosaffa/GNN_flow_routing` (chỉ tệp mô hình) | LSTM (sinh dòng chảy từng tiểu lưu vực) + GNN (định tuyến theo mạng sông) trên LamaH-CE ngày, 530 tiểu lưu vực, 1987–2017; cửa sổ 180 ngày, dự báo Q ngày hôm sau không dùng Q quá khứ; chia 70/15 ngẫu nhiên + 15% cuối làm test. NSE trung bình LSTM-GAT 0,61 so với LSTM 0,46 | Hướng "bộ mã hóa thời gian + GNN" trên LamaH-CE đã có với LSTM theo ngày; là baseline cho #104 |
| **HydroDiffusion** (Wang, Yu, Zhang, Varadharajan, Erichson), arXiv 2512.12183 (12/2025, preprint); mã `github.com/yhwang08/HydroDiffusion` | Mô hình khuếch tán dự báo xác suất 7 ngày, backbone S4D-FT, 531 lưu vực CAMELS, so với DRUM và 2 biến thể LSTM | Cùng nhóm #22; đã chuyển S4D-FT sang dự báo |
| **Block-Biased Mamba** (Yu & Erichson), NeurIPS 2025, arXiv 2505.09022; mã `github.com/AnnanYu/mamba-b2s6` | Chỉ ra Mamba kém S4D ở tác vụ phụ thuộc xa theo ba lập luận: khả năng biểu diễn, thiên kiến quy nạp, ổn định huấn luyện; đề xuất B2S6 vượt S4/S4D. Thí nghiệm chỉ gồm Long-Range Arena (ảnh làm phẳng, văn bản, ListOps, Pathfinder) và mô hình hóa ngôn ngữ, không có chuỗi thời gian hồi quy. Lập luận thiên kiến quy nạp phụ thuộc loại dữ liệu (bài giải thích cơ chế chọn lọc theo nội dung khó hợp lý với chuỗi điểm ảnh); hai lập luận còn lại không phụ thuộc loại dữ liệu. Miralles-González và cs. (arXiv 2501.14850) chỉ ra phần lớn kết quả trên LRA đến từ phụ thuộc gần. Repo công khai chỉ có mã huấn luyện mô hình ngôn ngữ (WikiText103, SlimPajama) | Rủi ro chung cho mọi ứng viên dùng Mamba với cửa sổ dài, gồm cả #104; mức áp dụng cho dữ liệu dòng chảy chưa được kiểm chứng |
| **FlowGATFormer**, Journal of Hydrology (2026) | GAT + Informer, 11 trạm CAMELS-CH, dự báo 1–24 ngày | Cùng ý "đồ thị + mô hình chuỗi thời gian", không dùng Mamba |
| **NeuralHydrology — lớp Mamba** (PR #163, 02/2024) | `InputLayer` trả tensor [thời gian, batch, đặc trưng] nhưng `mamba_ssm.Mamba` yêu cầu [batch, thời gian, đặc trưng] → Mamba quét theo trục batch; người đóng góp ghi đạt khoảng NSE 0,4 | Không dùng nguyên lớp này làm baseline; phải chuyển trục trước khi gọi Mamba |
| **#35** (Journal of Hydrology 2026) | S-Mamba trong benchmark theo giờ quét theo biến, không theo thời gian | Không dùng làm bằng chứng "Mamba theo thời gian kém LSTM" |
| Zhang và cs., Journal of Hydrology 662 (2025), DOI `10.1016/j.jhydrol.2025.134145` | Gọi S4D là "Mamba-type" khi mô phỏng xả hồ chứa | Khi viết cần phân biệt S4D (bất biến) với Mamba (chọn lọc) |

---

## 7. Tổng hợp đánh giá và xếp hạng

**7.1. Bảng chấm ba ứng viên chính.**

| Tiêu chí | #104 | #67 | #22 |
|---|---|---|---|
| A1. Phù hợp đề tài | Tốt — bài gốc là dự báo 6 giờ | Khá — bài gốc mô phỏng; thư viện có chế độ dự báo đã chạy thử | Khá — bài gốc mô phỏng; chuyển sang dự báo cần sửa 2 hàm |
| A2. Cải tiến được | Tốt — bộ mã hóa thời gian chưa có; Mamba + GNN + dữ liệu giờ chưa ai làm; rủi ro từ Block-Biased Mamba khi kéo dài cửa sổ | Tốt — tác giả nêu LSTM gặp khó ở chuỗi dài; rủi ro từ Block-Biased Mamba | Tốt — S4D-FT vs Mamba vs B2S6 có cơ sở lý thuyết; phần chuyển sang dự báo đã có |
| B1. Mã, tái lập | Khá — mã đủ, có checkpoint; phải sửa NSE | Tốt — mã bài khớp, thư viện bảo trì | Khá — phải sửa cấu hình về Bảng S3 |
| B2. Dữ liệu | LamaH-CE, CC BY 4.0 | CAMELS-US giờ, CC BY 4.0 (~20 GB) | CAMELS-US |
| B3. Tài nguyên | Nhẹ ở cửa sổ 24 giờ | ~7,9 giờ/hạt giống (V100) | ~10 giờ/hạt giống (L40S) |
| B4. Độ dài | 18 năm giờ; cửa sổ kéo được tới dưới 1 năm | 28 năm giờ, chuỗi 8.760 bước | Dữ liệu ngày, chuỗi 365 bước |
| C1. Phương pháp | Yếu — validation ngẫu nhiên, NSE sai, 1 hạt giống | Tốt — chia theo thời gian, NSE theo lưu vực, 10 hạt giống | Trung bình — validation ngẫu nhiên |
| C2. Uy tín | ICML (CORE A*) | HESS Q1, nhóm Kratzert | WRR Q1 |
| C3. Demo | Bản đồ 358 trạm theo giờ | Theo lưu vực, theo giờ | Theo lưu vực, theo ngày |
| C4. So sánh nhiều kiến trúc | Mã nhỏ, dễ thêm LSTM/GRU/Transformer/Mamba | Registry mô hình, có LSTM/MF-LSTM/MF2LSTM | Có LSTM và S4D |
| C5. Bộ dữ liệu khác | Phần đồ thị cần topo; nhánh không đồ thị chuyển được | Hỗ trợ CAMELS-US/GB/DE/CH/PL, Caravan | Chỉ CAMELS-US |
| C6. Rủi ro | MLP 2 lớp đã ngang mọi GNN | LSTM thuần đã ngang MF-LSTM | S4D-FT chỉ hơn LSTM 0,02 |

**7.2. Xếp hạng.**

| Hạng | Bài | Lý do chính |
|---|---|---|
| **1** | **#104** | Duy nhất đạt tốt cả hai tiêu chí quyết định; các nhược điểm (NSE, validation) sửa được và không đảo kết luận chính |
| **2** | **#67** (đồng hạng) | Phương pháp chuẩn nhất, chuỗi dài nhất; bài gốc mô phỏng, lợi thế chuỗi dài bị đặt dấu hỏi |
| **2** | **#22** (đồng hạng) | So sánh SSM bất biến ↔ chọn lọc có cơ sở lý thuyết (B2S6); bài gốc mô phỏng, chuỗi vừa phải |
| 4 | #103 | Hợp đề tài nhưng mã lỗi, không có bảng số; làm baseline đồ thị cho #104 |
| 5 | #71/#54 | Dự báo 1–7 ngày nhưng không tái lập được số liệu |
| 6 | #53 | Dự báo, cải tiến được; chọn epoch trên tập test |
| 7 | #42 | Dự báo, cải tiến được; NSE gộp, chọn cấu hình trên tập test |
| 8 | #6 | Bài benchmark, không có kiến trúc cụ thể để cải tiến |
| 9 | #36 | Mô hình lai, mô phỏng; HBV làm mờ khác biệt kiến trúc |
| 10 | #135 | Bài benchmark mô hình nền tảng |
| 11 | #38 | Mô phỏng; trọng tâm là biểu diễn thuộc tính tĩnh |

---

## 8. Nhận định và đề xuất cho tiểu luận

**8.1. Bài cơ sở đã chốt: #104** (26/9/2026). Lý do:
1. Bài gốc là dự báo lưu lượng theo giờ, khớp tên đề tài nhất.
2. Cải tiến tạo ra kiến trúc không gian–thời gian: Mamba mã hóa chuỗi thời gian từng trạm, GNN truyền thông tin dọc mạng sông.
3. Câu hỏi nghiên cứu rõ: bài gốc kết luận đồ thị không hơn MLP nhưng mô hình không có thành phần học chuỗi thời gian — khi có bộ mã hóa thời gian tốt và cửa sổ dài hơn, đồ thị có giúp dự báo không.
4. Có đóng góp phụ: chỉ ra và sửa lỗi tính NSE, tính lại bằng checkpoint của tác giả.
5. Demo bản đồ 358 trạm trực quan; có dữ liệu mạng sông để làm XAI theo không gian.
6. So sánh được với nhiều công trình trên cùng dữ liệu: MLP/GNN của #104, đồ thị dense của #103, LSTM + GNN của Mosaffa và cs. 2026 (#120).

**8.2. Phạm vi tiểu luận đề xuất.**

Câu hỏi nghiên cứu đề xuất: với dự báo lưu lượng theo giờ trên mạng sông, bộ mã hóa thời gian dạng SSM (Mamba) với cửa sổ dài hơn có cải thiện so với lớp affine của bài gốc hay không, và đồ thị mạng sông còn đóng góp gì khi đã có bộ mã hóa thời gian tốt. Kết quả khẳng định hay phủ định đều là câu trả lời hợp lệ; nên thống nhất cách đặt vấn đề này với GVHD từ đầu để không phụ thuộc vào việc Mamba hay đồ thị phải thắng.

Phần cốt lõi (ưu tiên):
1. Chạy lại #104, sửa công thức NSE, tính lại bằng 162 checkpoint (baseline chính xác).
2. Thay lớp affine bằng Mamba làm bộ mã hóa thời gian, kéo cửa sổ 24 giờ lên vài ngày–vài tuần.
3. So sánh MLP, GNN gốc, LSTM, GRU, Transformer, S4D, Mamba (có và không có đồ thị) trong cùng điều kiện, validation chia theo thời gian. S4D (SSM bất biến theo thời gian; mã S4D của #22 đã chạy thử được trên CPU, không cần kernel CUDA riêng) là đối chứng để phân biệt "Mamba không hợp" với "SSM nói chung không hợp" nếu Mamba không vượt baseline.

Phần đáp ứng yêu cầu (làm ở mức cần thiết):
4. XAI bằng Captum (Integrated Gradients) theo thời gian (giờ nào quan trọng) và theo không gian (trạm thượng nguồn nào quan trọng). Demiray & Demir đã làm Mamba + SHAP trên WaterBench-Iowa, nên phần XAI cần khác biệt ở chiều không gian; hướng XAI trên đồ thị sông chưa được tra cứu.
5. Demo bản đồ dự báo theo giờ, tô màu theo ngưỡng return period, xem chuỗi Q và giải thích XAI của từng trạm (demo là yêu cầu bắt buộc của tiểu luận).

Để dành cho khóa luận: đồ thị dense, biến thể B2S6 (repo công khai chỉ có mã mô hình ngôn ngữ, cần tự tích hợp), cửa sổ rất dài (viết lại bộ nạp dữ liệu), bộ dữ liệu thứ hai, phần mềm ứng dụng.

**8.3. Rủi ro và cách xử lý.**

| Rủi ro | Cách xử lý |
|---|---|
| Mamba không vượt MLP, hoặc thua S4D ở cửa sổ dài (Block-Biased Mamba) | Đề tài được định vị là nghiên cứu so sánh; có đối chứng S4D trong phần cốt lõi; B2S6 để dành cho khóa luận |
| Đồ thị không đóng góp thêm khi đã có bộ mã hóa thời gian tốt | Là một câu trả lời hợp lệ của câu hỏi nghiên cứu; demo bản đồ trạm vẫn dùng được |
| Khối lượng lớn so với một học kỳ | Ưu tiên phần cốt lõi 1–3; XAI và demo làm ở mức đáp ứng yêu cầu |
| Cửa sổ dài cần viết lại bộ nạp dữ liệu | Viết bộ nạp theo chuỗi liên tục, dùng cách chia năm liên tục 2008–2015 |
| Mức thổi phồng NSE chưa đo | Bước đầu tiên trên Colab: tải LamaH-CE, tính lại NSE từ checkpoint |

**8.4. So sánh với hai ứng viên đồng hạng 2.**
- **#22** phù hợp khi muốn một nghiên cứu so sánh SSM thuần (S4D-FT, Mamba, B2S6); hạn chế: bài gốc mô phỏng, phần chuyển sang dự báo đã được nhóm tác giả thực hiện, cải tiến chỉ là thay khối, CAMELS-US không có mạng sông để thêm GNN hoặc XAI không gian.
- **#67** phù hợp khi ưu tiên chuỗi giờ dài, phương pháp chuẩn và nhiều bộ dữ liệu (giai đoạn khóa luận); hạn chế: bài gốc mô phỏng, chế độ dự báo chưa có số liệu công bố, dữ liệu lớn, không có mạng sông.

---

## 9. Giới hạn của báo cáo

- Chưa chạy mã nào trên dữ liệu thật; Mamba (`mamba-ssm`) cần GPU CUDA nên chưa chạy được trên máy hiện tại.
- Chưa đo mức thổi phồng NSE thực tế của #104 (cần tải 14,8 GB dữ liệu).
- Tra cứu tính mới dựa trên OpenAlex và Semantic Scholar; không tra được Google Scholar.
- Việc Mamba có vượt baseline hay không là câu hỏi nghiên cứu, không xác định được trước khi thực nghiệm.

---

## Phụ lục A. Chi tiết đọc toàn văn các ứng viên khác

### A.1. #22 — chi tiết bổ sung

- **Kết quả theo vùng:** S4D-FT tốt ở Tây Nam Thái Bình Dương, Trung Nam (Kansas, Missouri, Arkansas, Texas); kém ở bờ Đông (New Jersey, Virginia, West Virginia), Ngũ Đại Hồ. Ở PUB, lợi thế lan rộng hơn (Appalachia, Rockies, Tây Bắc).
- **Quy nguyên:** kỹ năng NSE tương quan chủ yếu với cải thiện Pearson-r (r = 0,61); kỹ năng KGE tương quan chủ yếu với cải thiện FHV (r = 0,51 và 0,60). S4D-FT cải thiện Pearson-r ở 75% lưu vực nhưng FHV chỉ ở 50%.
- **Baseline trích từ bài khác (Bảng 3):** Sac-SMA NSE 0,65; MC-LSTM 0,72; Transformer KGE 0,71; Modified Transformer KGE 0,74.
- **Supporting Information:** Bảng S2 (LSTM) hidden 256, batch 256, forget bias 3, lr 1e-3/5e-4/1e-4 theo epoch 1–10/11–25/26–30, dropout 0,4, 30 epoch. Hình S1/S2 (độ nhạy, 1 hạt giống): giá trị tối ưu NSE ≈ 0,74, KGE ≈ 0,765; max_dt = 1 cho NSE ≈ 0,72, KGE ≈ 0,667; min_dt = 0,001 cho KGE ≈ 0,715; batch 256 cho NSE ≈ 0,705; 30 epoch cho KGE ≈ 0,675; lr 2e-4 cho KGE ≈ 0,62. Đây là các giá trị mà `train_val.sh` (max_dt 1, min_dt 0,001 mặc định) và `train_pub.sh` (batch 256, 30 epoch) đang dùng. SI không ghi phân tích độ nhạy tính trên kỳ nào. Bảng S5: S4D-FT cải thiện NSE ở 68,7%, KGE 54,4%, Pearson-r 75,0%, FHV 49,5% lưu vực. Hình S4: ở lưu vực ẩm (AI < 1), S4D-FT có sai số tương đối lớn hơn LSTM trên 2% dòng lớn nhất.

### A.2. #54 và #71 — FHNN

**Thông tin xuất bản.** #54: Zac McEachran, Rahul Ghosh, Arvind Renganathan, Somya Sharma, Kelly Lindsay, Michael Steinbach, John Nieber, Christopher Duffy, Vipin Kumar (University of Minnesota; Pennsylvania State University), Water Resources Research 61(11), e2024WR039064 (2025), DOI `10.1029/2024WR039064`, CC BY-NC. Dữ liệu dự báo các trận lũ trên HydroShare `10.4211/hs.bdc496c09e394875accc732e94daab3b`. #71: Ghosh và cs., IEEE ICDM 2025 (tr. 1224–1233), arXiv 2407.20152v2. README repo `arvindrenga96/FHNN` ghi repo là mã của bài ICDM; repo có mã mô hình và 4 baseline (CT-LSTM, AR-LSTM, RRFormer, TFT).

**Dữ liệu và giao thức.** CAMELS (#54): bước ngày, train 1985–1993, validation 1993–1995, test 1995–2005; mô hình global cho 531 lưu vực; hai chế độ tối ưu (7 ngày và 1 ngày); giai đoạn dự báo dùng khí tượng quan trắc. Vận hành NWS: bước 6 giờ, mô hình riêng từng lưu vực, train tới 2009, validation 2010–2011, test 2012–2019; mưa dự báo lưu trữ của NCRFC chỉ 24–48 giờ, sau đó gán 0.

**Mô hình.** Bộ mã hóa 3 BiLSTM theo thang nhanh/trung gian/chậm (CAMELS: 1/7/30 ngày theo bài), trạng thái ghép lại để khởi tạo bộ giải mã LSTM một chiều chạy trên khí tượng tương lai. Kích thước: LSTM 256 → FHNN 255 = 3 × 85. Hàm mục tiêu RMSE.

**Kết quả (#54, Bảng 2, NSE trung bình / trung vị).**

| Mô hình (chế độ) | Ngày 1 | Ngày 3 | Ngày 7 | Gộp 7 ngày |
|---|---|---|---|---|
| LSTM-AR (7 ngày) | 0,708 / 0,816 | 0,653 / 0,796 | 0,576 / 0,780 | 0,634 / 0,792 |
| FHNN (7 ngày) | 0,792 / 0,853 | 0,722 / 0,808 | 0,678 / 0,789 | 0,718 / 0,807 |
| LSTM-AR (1 ngày) | 0,766 / 0,857 | — | — | — |
| FHNN (1 ngày) | 0,806 / 0,861 | — | — | — |

Vận hành: FHNN có NSE tổng trận cao hơn NWS ở 30/46 trận (65%); NWS tốt hơn trong 12–18 giờ đầu; FHNN tốt hơn từ ngày 2 đến ngày 3–4. Trong vòng 1 ngày trước đỉnh, RMSE mực nước NWS 0,8 ft, FHNN 1,6 ft; 82% dự báo đỉnh của FHNN thấp hơn thực tế.

**Kết quả (#71, Bảng 5, train 1989–2001 / test 2001–2009, 5 lần khởi tạo, NSE trung bình / trung vị).**

| Mô hình | 1 bước | 3 bước | 7 bước |
|---|---|---|---|
| LSTM-AR | 0,66 / 0,80 | 0,65 / 0,79 | 0,63 / 0,78 |
| RR-Former | 0,65 / 0,78 | 0,59 / 0,75 | 0,53 / 0,72 |
| ExoTST | 0,73 / 0,79 | 0,67 / 0,75 | 0,64 / 0,74 |
| TFT | 0,75 / 0,83 | 0,72 / 0,81 | 0,71 / 0,80 |
| FHNN | 0,77 / 0,85 | 0,73 / 0,82 | 0,72 / 0,81 |

**Nhận xét của nhóm.** Phần CAMELS của #54 chỉ so với 1 baseline, không có khoảng tin cậy; hai bài dùng hai giao thức và số liệu khác nhau; trong bản arXiv, chữ và Bảng 4 lệch nhau ở FHNN; NSE của mô hình học máy là trung bình các bản dự báo 7 ngày còn NSE của NWS là NSE tổng; phần CAMELS không nêu tập validation. Mã: bộ mã hóa chỉ nhận chuỗi trước cửa sổ dự báo (không rò rỉ nhãn), chọn checkpoint theo loss validation; `UTILS.per_node_R2` tính NSE theo lưu vực (đúng); `UTILS.stride_array` cắt mỗi năm thủy văn đúng 1 cửa sổ 365 ngày; `batch_size` gán cứng 256; thiếu tiền xử lý.

### A.3. #53 — HydroTFT

**Thông tin xuất bản.** Qi Cheng, Yingda Fan, Xiaowei Jia (Rutgers University), Yiqun Xie (University of Maryland), Dan Lu (Oak Ridge National Laboratory). Machine Learning: Earth 2, 025014 (2026), DOI `10.1088/3049-4753/aea164`, CC BY 4.0.

**Giao thức.** CAMELS 531, forcing Maurer 5 biến + 5 đặc trưng kỹ thuật (tổng mưa trượt 90 ngày, độ-ngày tan băng 7 ngày, số ngày mưa > 1 mm trong 7 ngày, sin/cos ngày trong năm), 27 thuộc tính tĩnh; train 10/1999–9/2008, test 10/1989–9/1999; bài không mô tả tập validation.

**Mô hình.** Temporal Fusion Transformer: mạng chọn biến tĩnh/động, LSTM encoder khởi tạo từ ngữ cảnh tĩnh, attention đa đầu, đầu tuyến tính ra p giá trị; d = 256, ~3,4 triệu tham số. Loss NSE theo lưu vực; 3 hạt giống.

**Kết quả (tầm 1 ngày).**

| Mô hình | NSE trung bình | NSE trung vị | KGE | RMSE (mm/ngày) |
|---|---|---|---|---|
| EA-LSTM | 0,661 | 0,702 | 0,665 | 1,58 |
| LSTM | 0,677 | 0,724 | 0,691 | 1,52 |
| iTransformer | 0,404 | 0,510 | 0,481 | 2,05 |
| PatchTST | 0,341 | 0,398 | 0,372 | 2,22 |
| HydroTFT | 0,691 | 0,727 | 0,655 | 1,49 |

Tầm 7–14 ngày: HydroTFT tốt nhất và khác biệt có ý nghĩa (ΔNSE +0,020 đến +0,042), nhưng NSE trung bình ngày +7 chỉ 0,170, ngày +14 là 0,154. Trọng số attention trung bình dao động hẹp quanh mức đều 1/365.

### A.4. #42 — TFRN

**Thông tin xuất bản.** Yongquan Zhang, Yuanhao Chen, Haoqi Yu, Jingrong Dai, Bo Liu, Nannan Li, Changmiao Wang, Ahmed Elazab. IEEE TGRS 63 (2025) 4211314, DOI `10.1109/TGRS.2025.3605332`; ngành viễn thám/kỹ thuật điện.

**Chi tiết.** Theo khung RR-Former: 5 biến khí tượng + thuộc tính tĩnh dạng số (52 cho CAMELS, 108 cho AUS). Quantile loss, Adam lr 1e-3, batch 512, 100 epoch, chọn epoch theo loss validation; 1 lần chạy. Bảng II: CAMELS — TFRN MAE 0,307 / NSE 0,853 / RMSE 1,403 / KGE 0,884; TFT 0,313 / 0,840 / 1,427 / 0,874; RR-Former 0,327 / 0,842 / 1,424 / 0,870; LSTM 0,340 / 0,827 / 1,489 / 0,851. CAMELS-AUS — TFRN 0,167 / 0,805 / 1,500 / 0,844; LSTM 0,195 / 0,750 / 1,707 / 0,761. Ablation: riêng ED-LSTM đã đạt NSE 0,828; ba mô-đun cộng lại thêm +0,025. Bảng VII, VIII (chọn cấu hình) báo cáo trên tập test và dòng được chọn trùng Bảng II. Bài mô tả "chuẩn hóa độc lập từng tập" nhưng mã chuẩn hóa val/test bằng thống kê tập train. Bảng X: hidden 16, 0,0402 triệu tham số.

### A.5. #6 — From RNNs to Transformers

**Thông tin xuất bản.** Jiangtao Liu, Chaopeng Shen, Fearghal O'Donncha, Yalan Song, Wei Zhi, Hylke E. Beck, Tadd Bindas, Nicholas Kraabel, Kathryn Lawson. HESS 29:6811–6828 (2025), DOI `10.5194/hess-29-6811-2025`.

**Dữ liệu.** CAMELS 531 (train 1999–2008 / val 1980–1989 / test 1989–1999), Global Streamflow 3.434 lưu vực, ISMN độ ẩm đất, SNOTEL SWE, CAMELS-Chem DO.

**Kết quả.** Hồi quy: LSTM KGE 0,80 (CAMELS), 0,75 (toàn cầu), hơn Transformer tốt nhất 0,07 và 0,11. Dự báo (khí tượng tương lai hoàn hảo + Q quá khứ): lead 1 ngày LSTM KGE 0,89 > ETSformer 0,81; thu hẹp còn 0,01 ở lead 30 ngày. Tự hồi quy: lead ≥ 7 ngày Pyraformer 0,32 > LSTM 0,15. Zero-shot (7 lưu vực): TimeGPT 0,68 > LSTM 0,50. Crossformer phát thải CO₂ gấp ~13 lần LSTM.

### A.6. #36 — S4D/S5D trong dPL

**Thông tin xuất bản.** Xin Jing, Jungang Luo, Xue Yang, Ganggang Zuo (Xi'an University of Technology). ESWA 329 (2026) 133040, DOI `10.1016/j.eswa.2026.133040`.

**Giao thức.** CAMELS-US 531 và 671 lưu vực; mưa, Tmean, PET; 35 thuộc tính tĩnh; train 10/1980–9/1995, test 10/1995–9/2010; không có validation, đánh giá ở epoch 100; mẫu 730 ngày (365 khởi động + 365 tính loss); loss NSE (ε = 0,1); Adadelta lr 1,0, batch 100.

**Kết quả (Bảng 2, NSE / KGE).**

| Bộ mã hóa | 531 lưu vực | 671 lưu vực |
|---|---|---|
| δMG-LSTM | 0,742 / 0,755 | 0,735 / 0,745 |
| δMG-S4D | 0,756 / 0,778 | 0,752 / 0,770 |
| δMG-TCN | 0,732 / 0,759 | 0,738 / 0,767 |
| δMG-TimeMixer | 0,742 / 0,748 | 0,746 / 0,757 |
| δMG-Transformer | 0,741 / 0,742 | 0,743 / 0,743 |
| δMG-S5Dv2 | — | 0,763 / 0,771 |
| LSTM thuần | — | 0,722 / 0,769 |
| S4D thuần | — | 0,751 / 0,756 |

PUB: S4D hơn LSTM ở cả 10 phần (NSE50 0,70 so với 0,66). S4D có trường tiếp nhận tới ~300 ngày (LSTM ~30–50 ngày) nhưng tham số dao động mạnh. Báo cáo 5 hạt giống trong repo: S5Dv2 0,7514, S5Dv1 0,7512, S4D 0,7496, LSTM 0,7416 — S5Dv1 và S5Dv2 gần như bằng nhau.

### A.7. #135 — Mô hình nền tảng chuỗi thời gian

**Thông tin xuất bản.** Alexander Y. Sun, Albert A. Sun (UT Austin). Machine Learning: Earth 2, 010501 (2026), DOI `10.1088/3049-4753/ae4982`.

**Kết quả đơn biến, dự báo 1 ngày (NSE trung vị / trung bình):** LSTM 0,593 / 0,589; MOIRAI 0,258 / −1,448; Chronos 0,495 / 0,490; TTM 0,437 / 0,470; Sundial 0,564 / 0,564. Kéo dài lên 7 ngày: NSE trung vị các mô hình nền tảng còn 0,19–0,23. Dữ liệu 3 giờ: LSTM tốt nhất ở bước 3 giờ (NSE 0,984). Đa biến nowcast: TTM zero-shot 0,469; TTM tinh chỉnh 0,739 (không thuộc tính tĩnh) và 0,753 (có thuộc tính tĩnh). Kỳ test 1995–2005 khác giao thức Kratzert.

### A.8. #38 — Biểu diễn raster

**Thông tin xuất bản.** De-Hui Ouyang, E Deng, Yi-Qing Ni (The Hong Kong Polytechnic University). WRR 62, e2026WR043815 (2026), DOI `10.1029/2026WR043815`.

**Kết quả (Bảng 2, trung vị 531 lưu vực).**

| Giao thức | Mô hình | NSE | KGE′ | FHV₀.₀₂ |
|---|---|---|---|---|
| Thời gian | Attr-LSTM | 0,70 | 0,75 | −27,16 |
| Thời gian | RasterMean-LSTM | 0,71 | 0,73 | −28,45 |
| Thời gian | MID-CNN-LSTM | 0,73 | 0,76 | −27,53 |
| Thời gian | HIGH-CNN-LSTM | 0,71 | 0,73 | −28,26 |
| PUB | Attr-LSTM | 0,61 | 0,64 | −34,15 |
| PUB | RasterMean-LSTM | 0,63 | 0,67 | −31,08 |
| PUB | MID-CNN-LSTM | 0,60 | 0,62 | −34,36 |

LSTM 1 lớp 256 nút, 30 epoch, 1 hạt giống; chi phí GPU mỗi epoch: MID ≈ 2 lần, HIGH ≈ 14 lần Attr-LSTM. Mã dùng hằng số chuẩn hóa chép từ repo khác, có lỗi Tmax = Tmin.

### A.9. #124 — STResBiGRU

Ziyu Sheng, Yuting Cao, Yin Yang, Shiping Wen. Neural Networks 205 (2027) 109416, DOI `10.1016/j.neunet.2026.109416`. Dữ liệu Columbia River DART (`wqm_hourly`), 35.040 mẫu giờ 2016–2019, bài không nêu tên trạm; chia 75/25 theo thời gian, không validation. BiGRU có kết nối tắt thời gian, kiến trúc ResNet Plus, chú ý hai đường (SENet theo thời gian và theo đặc trưng), snapshot ensemble.

| Lead (h) | STResBiGRU MAE / NSE | Baseline tốt nhất theo MAE | Baseline tốt nhất theo NSE |
|---|---|---|---|
| 2 | 3,48 / 0,952 | MambaFormer 3,61 | MambaFormer 0,953 |
| 4 | 4,61 / 0,918 | MambaFormer 4,58 | MambaFormer 0,934 |
| 8 | 5,17 / 0,909 | MambaFormer 5,70 | Transformer, MambaFormer 0,902 |
| 12 | 5,67 / 0,901 | BiLSTM 6,28 | BiLSTM, BiGRU 0,883 |
| 24 | 6,92 / 0,866 | Transformer 7,25 | Transformer 0,856 |

Mức cải thiện "khoảng 20%" chỉ đúng khi so với baseline yếu nhất; so với baseline mạnh nhất, MAE giảm 3,6–9,7%, và ở lead 4 giờ thua MambaFormer.

### A.10. #122 — HMGSTN

Xuerui Zhou, Baowei Yan, Jun Zhang, Jianbo Chang, Dongxu Yang. EAAI 177 (2026) 114967, DOI `10.1016/j.engappai.2026.114967`. Dữ liệu chính 18 trạm sông Nhã Lung (2016–2021, bước 6 giờ, không công khai), chia 70/30, không validation; LamaH-CE là thí nghiệm phụ (27 trạm, đầu vào 216 giờ, 1 hạt giống). Bộ mã hóa thời gian Linear → TCN giãn → cổng Fourier → chú ý thời gian; GCN cho đồ thị khí tượng và thủy văn, cổng hợp nhất chéo, ma trận kề động; loss MSE + trọng số 3 cho điểm vượt phân vị 95%. Yalong: NSE 0,916 (h4) → 0,695 (h24), GraphWaveNet 0,914 → 0,693 (Wilcoxon không có ý nghĩa ở h12, h24). LamaH-CE h24: HMGSTN 0,661 = GraphWaveNet 0,661. Có thể tham khảo hàm mất mát có trọng số đỉnh lũ và quy trình hiệu chỉnh khoảng bất định.

### A.11. #123 — CTB

Deguang Wang, Qian Li, Shijun Liu, Li Pan, Jun Li. ESWA 281 (2025) 127387, DOI `10.1016/j.eswa.2025.127387`. WaterBench-Iowa 125 trạm (10/2011–9/2018); đầu vào 72 giờ + mưa 120 giờ tới coi như biết trước; chia 7:2:1 (không rõ cách chia); mọi baseline dùng chung siêu tham số. NSE trung vị: 0,90 (1 giờ), 0,73 (24 giờ), 0,52 (72 giờ), 0,38 (120 giờ); S2S hơn CTB ở 96–120 giờ.
