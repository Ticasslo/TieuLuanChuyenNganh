> **Phạm vi (cập nhật 25/9/2026):** ban đầu chỉ lấy bài train trên CAMELS; sau đó người dùng mở rộng — lưu vực nào cũng được, miễn dữ liệu free + dài năm, từ 2024 trở đi. Bài không dùng CAMELS gắn nhãn 🟡 "Không phải CAMELS". Đã gộp 3 cặp nhập trùng (#6, #33, #35); xóa bài hydro-climatic/land parameter (dự báo đa biến, không rõ lưu lượng là biến chính — người dùng quyết định); đánh số lại.
>
> **Đã kiểm tra lại toàn bộ 25/9/2026** qua trang gốc từng bài + OpenAlex/Crossref/Semantic Scholar: đã xóa bài không dùng CAMELS, sửa dataset/model/code/năm sai, thêm 5 bài (#42–#46). Đã loại thêm: bài RFFA vùng thiếu dữ liệu (Ấn Độ/Mỹ — không xác nhận được có dùng CAMELS) và bài zero-shot foundation model (không train model trên CAMELS). Số "Cited by" giữ theo Google Scholar (Claude không vào được Scholar); ô ghi "(OA)"/"(S2)" là số của OpenAlex/Semantic Scholar, thường thấp hơn Scholar.

#1 — Temporal Fusion Transformers for streamflow prediction: Value of combining attention with recurrence
Mục
Nội dung
1. Năm
2024
2. Cited by
139
3. Kiến trúc model
Temporal Fusion Transformer (TFT) — kiến trúc lai kết hợp LSTM encoder-decoder + cơ chế multi-head attention + gating network, so sánh trực tiếp với LSTM chuẩn. Train 10 lần với seed ngẫu nhiên khác nhau cho mỗi model.
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — thuộc ngành Tài nguyên nước/Khoa học Trái Đất (Water Resources), KHÔNG thuộc CNTT/AI thuần túy nhưng là tạp chí hàng đầu về ứng dụng ML trong thủy văn. — DOI 10.1016/j.jhydrol.2024.131301 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1 (Water Science and Technology, theo SJR)
6. Dataset
🟡 Reference — CAMELS-US (531 lưu vực) là 1 trong 5 tập con hợp thành Caravan (2.610 lưu vực), không train riêng lẻ CAMELS-US
7. Code free
Đã tra thêm — không tìm thấy repo GitHub/Zenodo riêng cho bài này (chỉ có bài liên quan không cùng tác giả). Coi như không có code free.

#2 — HESS Opinions: Never train a Long Short-Term Memory (LSTM) network on a single basin
Mục
Nội dung
1. Năm
2024
2. Cited by
294
3. Kiến trúc model
LSTM chuẩn (không phải kiến trúc mới) — đây là bài "position paper", họ train và so sánh LSTM huấn luyện đơn lưu vực (single-basin) vs đa lưu vực (multi-basin) để chứng minh luận điểm, không đề xuất kiến trúc mới.
4. Tạp chí & ngành
Hydrology and Earth System Sciences (HESS, Copernicus) — ngành Tài nguyên nước/Thủy văn, không thuộc CNTT/AI thuần túy. — DOI 10.5194/hess-28-4187-2024 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1 (Water Science and Technology, theo SJR)
6. Dataset
🟢 Core — Có nhắc và dùng CAMELS-US cùng với Caravan , Global Runoff Data Center — làm ví dụ minh họa cho luận điểm đa lưu vực
7. Code free
Có — link "Model code and software" công khai trên trang HESS

Lưu ý: bài này là bài quan điểm (opinion article), không phải bài đề xuất kiến trúc mới — dùng được cho phần lý luận "vì sao nên train đa lưu vực", nhưng không có architecture cụ thể mới để bạn phát triển tiếp. Bạn cân nhắc có tính vào không.

#3 — Unveiling the limits of deep learning models in hydrological extrapolation tasks
Mục
Nội dung
1. Năm
2025
2. Cited by
57
3. Kiến trúc model
LSTM đơn thuần (stand-alone), so sánh với hybrid model (kết hợp physics-based + data-driven). Trọng tâm bài là khả năng ngoại suy (extrapolation) khi gặp sự kiện mưa cực đoan tổng hợp vượt ngoài phạm vi dữ liệu train.
4. Tạp chí & ngành
Hydrology and Earth System Sciences (HESS) — ngành Tài nguyên nước/Thủy văn, không thuộc CNTT/AI — DOI 10.5194/hess-29-5871-2025 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟡 Reference — train chính trên CAMELS-CH (196 lưu vực Thụy Sĩ), CAMELS-US chỉ được nhắc làm tài liệu tham khảo trong phần Data sets, không phải dataset train chính
7. Code free
Có — link Zenodo công khai (doi:10.5281/zenodo.14771377)

#4 — Probing the limit of hydrologic predictability with the Transformer network (Liu, Bian, Lawson, Shen)
Mục
Nội dung
1. Năm
2024
2. Cited by
115
3. Kiến trúc model
vanilla Transformer encoder + biến thể "recurrence-free" Transformer, so với LSTM baseline
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — ngành Thủy văn/Khoa học Trái đất, không phải CNTT/AI — DOI 10.1016/j.jhydrol.2024.131389 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (671 lưu vực CONUS; bài dạng Short communication)
7. Code free
Có — github.com/mhpi/hydroDL + Zenodo

[Đọc code 26/9/2026: hydroDL (mhpi) KHÔNG chứa code Transformer (grep toàn repo không có) → code mô hình của bài không nằm trong repo này; khung Transformer của cùng nhóm Shen có ở #6]

#5 — Multi-step ahead forecasting of daily streamflow based on transform-based DL model (He, Xu, Wu, Kang, Huang)
Mục
Nội dung
1. Năm
2025
2. Cited by
41
3. Kiến trúc model
Rel-Informer (Informer cải tiến relative position encoding) so với Informer, Transformer, LSTM
4. Tạp chí & ngành
Scientific Reports (Nature Portfolio) — đa ngành, bài này thuộc Thủy văn/Kỹ thuật môi trường — DOI 10.1038/s41598-025-89837-w (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1/Q2 (tùy category)
6. Dataset
🟢 Core — CAMELS-US (596/671 lưu vực)
7. Code free
Không — không có repo riêng; chỉ dùng thư viện Informer công khai (github.com/zhouhaoyi/Informer2020)

Gốc kiến trúc: Informer — Zhou et al., AAAI 2021 (README repo chính thức ghi "AAAI'21 Best Paper"), code github.com/zhouhaoyi/Informer2020 — tức kiến trúc gốc thuộc venue CNTT/AI dù bài này đăng ở Scientific Reports

#6 — From RNNs to Transformers: benchmarking deep learning architectures for hydrologic prediction
Mục
Nội dung
1. Năm
2025
2. Cited by
46 / 24 (Crossref, 25/9/2026)
3. Kiến trúc model
Khung benchmark: LSTM so với 11 kiến trúc Transformer (Informer, iTransformer, PatchTST, Crossformer, Pyraformer, Reformer, ETSformer, Non-stationary, CARDformer, vanilla…) + DLinear, TimesNet, LLM/TSAM zero-shot. Kết quả: LSTM tốt nhất ở hồi quy (KGE trung vị 0,75 toàn cầu, hơn Transformer tốt nhất 0,11); Transformer vượt LSTM khi bài toán phức tạp hơn (forecast, autoregression, zero-shot). Tác giả: Liu, Shen, O'Donncha, Song, Zhi, Beck, Bindas, Kraabel, Lawson
4. Tạp chí & ngành
HESS 29(23):6811–6828, DOI 10.5194/hess-29-6811-2025 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS + 3.434 lưu vực toàn cầu (Beck et al. 2020); thêm độ ẩm đất, SWE, DO
7. Code free
Có — code thật ở Zenodo doi:10.5281/zenodo.17863071 (bản mới nhất, concept DOI 10.5281/zenodo.15852144): HESS_paper.zip 327 MB, giấy phép CC BY-NC 4.0, pipeline PyTorch đọc NetCDF. Bản zenodo.15852145 ghi trong bài chỉ là trang trỏ sang bản mới (kiểm 25/9/2026)

[Đọc code 25/9/2026: zip kèm sẵn dữ liệu đã xử lý `examples/data_preparation/data/CAMELS_processed/CAMELS.nc` (1.098 MB giải nén) — 531 lưu vực, 15 forcing (NLDAS/Maurer/Daymet), 26 thuộc tính tĩnh, giao thức Kratzert; chuẩn hóa chỉ theo thống kê tập train (`cal_statistics`); 14 mô hình chạy chung 1 khung; README khuyên Python 3.12, có hướng dẫn cắm dataset riêng]

[Gộp 25/9/2026: bài này từng bị nhập trùng thành #101]

#7 — Interpretable ML on large samples for runoff estimation in ungauged basins (Xu, Lin, Hu, Wang, Wu, Zhang, Xiao, Luo)
Mục
Nội dung
1. Năm
2024 (Vol. 639, Article 131598)
2. Cited by
51-52
3. Kiến trúc model
XGBoost (interpretable, dùng SHAP) cho parameter regionalization; so sánh với deep transfer learning LSTM và Transformer
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — ngành Thủy văn, không phải CNTT/AI — DOI 10.1016/j.jhydrol.2024.131598 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟡 Reference — dùng Caravan (chỉ 482/6.830 lưu vực gốc từ CAMELS-US, qua forcing ERA5-Land khác gốc)
7. Code free
không thấy đề cập trong phần preview tôi đọc được (không có Data/Code availability statement rõ ràng)

#8 — Are deep learning models in hydrology entity aware?
Mục
Nội dung
1. Năm
2025
2. Cited by
39
3. Kiến trúc model
LSTM (entity-aware, so sánh có/không static features)
4. Tạp chí & ngành
Geophysical Research Letters (AGU) — ngành Thủy văn/Khoa học Trái đất, không phải CNTT/AI — DOI 10.1029/2024gl113036 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (531 lưu vực)
7. Code free
Có — code công khai trên Zenodo (Heudorfer & Acuña Espinoza 2025), theo mục Open Research trên AGU

#9 — Uncovering the dynamic drivers of floods through interpretable deep learning
Mục
Nội dung
1. Năm
2024
2. Cited by
36
3. Kiến trúc model
peak-sensitive Informer so với Transformer, LSTM
4. Tạp chí & ngành
Earth's Future (AGU) — Thủy văn/Khoa học Trái đất — DOI 10.1029/2024ef004751 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟡 Reference — Caravan (482 lưu vực Mỹ, theo Kratzert et al. 2024), không phải CAMELS-US gốc
7. Code free
Không có repo riêng — chỉ dùng thư viện công khai Informer (github.com/zhouhaoyi/Informer2020) và SHAP (github.com/shap/shap)

#10 — Probabilistic Diffusion Models Advance Extreme Flood Forecasting (DRUM) — Ou, Nai, Pan, Zheng, Shen et al., DOI 10.1029/2025GL115705
Mục
Nội dung
1. Năm
2025
2. Cited by
22
3. Kiến trúc model
DRUM (Diffusion-based Runoff Model) — mô hình khuếch tán xác suất (denoising diffusion probabilistic model) sinh dự báo runoff dạng ensemble từ nhiễu, có điều kiện hóa theo dữ liệu khí tượng quá khứ/dự báo + thuộc tính lưu vực tĩnh. So sánh với LSTM-based benchmarks (state-of-the-art). Đây là kiến trúc rất mới/hiếm trong thủy văn (generative AI/diffusion).
4. Tạp chí & ngành
Geophysical Research Letters (AGU) — Thủy văn/Khoa học Trái đất, không thuộc CNTT/AI thuần túy nhưng dùng kỹ thuật Generative AI — DOI 10.1029/2025gl115705 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US chuẩn (531 lưu vực CONUS); so với LSTM tất định (LSTM-d) và LSTM xác suất (LSTM-p)
7. Code free
Có — github.com/ozg2021/DRUM + Zenodo doi:10.5281/zenodo.15846601

Đồng tác giả có Chaopeng Shen — cùng nhóm với 2 bài đã có trong danh sách (Probing the limit, From RNNs to Transformers) — độ tin cậy học thuật cao.

#11 — Analyzing the generalization capabilities of a hybrid hydrological model for extrapolation to extreme events
Mục
Nội dung
1. Năm
2025
2. Cited by
38
3. Kiến trúc model
Hybrid model (process-based conceptual + LSTM) so với LSTM chuẩn và mô hình concептual thuần túy, tập trung khả năng ngoại suy sự kiện cực đoan
4. Tạp chí & ngành
HESS — Thủy văn, không CNTT/AI — DOI 10.5194/hess-29-1277-2025 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (531 lưu vực); so LSTM, HBV và hybrid
7. Code free
Có — Zenodo doi:10.5281/zenodo.14191623 + github.com/eduardoAcunaEspinoza/hybrid_extrapolation

#12 — QDeepGR4J
Mục
Nội dung
1. Năm
2025
2. Cited by
11
3. Kiến trúc model
Hybrid GR4J (conceptual) + CNN/LSTM/RNN, mở rộng bằng quantile regression để định lượng uncertainty, dự báo multi-step streamflow
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2025.134434 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟡 Reference — CAMELS-AUS (Úc)
7. Code free
Không thấy — trang ScienceDirect và bản arXiv 2510.05453 đều không có link code (chỉ ghi dùng CAMELS-Aus)

#13 — To bucket or not to bucket?
Mục
Nội dung
1. Năm
2024
2. Cited by
48
3. Kiến trúc model
Hybrid model — dynamic parameterization của conceptual model bằng LSTM, so sánh với LSTM chuẩn và conceptual model tĩnh
4. Tạp chí & ngành
HESS — Thủy văn, không CNTT/AI — DOI 10.5194/hess-28-2705-2024 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟡 Reference — CAMELS-GB (Anh)
7. Code free
Có — KIT-HYD/Hy2DL, Zenodo doi:10.5281/zenodo.11103634

#14 — Multi-step regional rainfall-runoff modeling using pyramidal transformer
Mục
Nội dung
1. Năm
2025
2. Cited by
20
3. Kiến trúc model
Pyramidal Transformer (PT) — kiến trúc attention đa phân giải thời gian (pyramidal attention), so với RR-Former (Transformer gốc) và LSTM
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2025.132935 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (448 lưu vực)
7. Code free
Không thấy — trang ScienceDirect không có link GitHub/Zenodo (trang ScienceDirect chỉ cho xem tóm tắt + mục Data availability; tìm GitHub/Zenodo theo tên bài không ra)

#15 — Investigating the streamflow simulation capability of MC-LSTM across CONUS
Mục
Nội dung
1. Năm
2025
2. Cited by
23
3. Kiến trúc model
MC-LSTM (Mass-Conserving LSTM) — biến thể LSTM tuân thủ định luật bảo toàn khối lượng, so với LSTM chuẩn
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2025.133161 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — chỉ **8 lưu vực** chọn từ CAMELS (không phải toàn bộ CONUS); so MC-LSTM với LSTM và Sac-SMA
7. Code free
Có — github.com/Pandas-Paws/MC-LSTM (README ghi "preliminary version"; bài J. Hydrol., DOI 10.1016/j.jhydrol.2025.133161; chỉ 8 lưu vực). Tìm qua rà GitHub 25/9/2026

#16 — A mass conservation relaxed (MCR) LSTM model for streamflow simulation across CONUS
Mục
Nội dung
1. Năm
2025
2. Cited by
13
3. Kiến trúc model
MCR-LSTM — biến thể "nới lỏng" ràng buộc bảo toàn khối lượng so với MC-LSTM gốc, so với LSTM chuẩn và MC-LSTM
4. Tạp chí & ngành
Water Resources Research (AGU) — Thủy văn, không CNTT/AI — DOI 10.1029/2024wr039131 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (ghi rõ CONUS trong tiêu đề, cùng nhóm tác giả với MC-LSTM CONUS ở trang 7)
7. Code free
Có — Zenodo doi:10.5281/zenodo.15420018

[Đọc code 26/9/2026: Zenodo MCR-LSTM.zip (0,1 MB): dựa khung EA-LSTM của Kratzert, 531 lưu vực, train 1999–2008, val/test 1989–1999, chuỗi 365; cùng nhóm với #22]

#17 — Improving differentiable hydrologic modeling with interpretable forcing fusion
Mục
Nội dung
1. Năm
2025
2. Cited by
15
3. Kiến trúc model
Mô hình differentiable (δ model) — neural network học cách kết hợp trọng số 3 nguồn forcing (Daymet, NLDAS, Maurer) đưa vào mô hình vật lý để dự đoán streamflow
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2025.133320 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (CONUS, dùng đúng 3 forcing chuẩn CAMELS-US) — nhóm tác giả Chaopeng Shen (Penn State, MHPI)
7. Code free
Có khả năng — mục Data availability có câu "The differentiable model code used in this work was..." nhưng bị cắt ở bản xem được; chưa đọc được link

#18 — CH-RUN: a deep-learning-based spatially contiguous runoff reconstruction for Switzerland
Mục
Nội dung
1. Năm
2025
2. Cited by
21
3. Kiến trúc model
LSTM và TCN (temporal convolutional network) tái tạo runoff không gian liên tục toàn Thụy Sĩ (1962-2023)
4. Tạp chí & ngành
HESS — Thủy văn, không CNTT/AI — DOI 10.5194/hess-29-1061-2025 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟡 Reference — CAMELS-CH (98 lưu vực chọn từ 267 trạm)
7. Code free
Có — github.com/bask0/mach-flow

#19 — Associations between deep learning runoff predictions and hydrogeological conditions in Australia
Mục
Nội dung
1. Năm
2024
2. Cited by
18
3. Kiến trúc model
LSTM tích hợp catchment attributes (địa chất thủy văn), kết hợp unsupervised learning để phân loại lưu vực
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2024.132569 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟡 Reference — CAMELS-AUS (Úc, hàng trăm lưu vực)
7. Code free
Không — "Data will be made available on request"

#20 — HydroQuantum: A new quantum-driven Python package for hydrological simulation
Mục
Nội dung
1. Năm
2026 (Jan 2026, Vol 195)
2. Cited by
11
3. Kiến trúc model
QLSTM (fully Quantum LSTM), VQC (Variational Quantum Circuits), Hybrid Quantum-Classical LSTM — so với LSTM cổ điển. Kiến trúc rất hiếm (quantum computing áp dụng vào thủy văn)
4. Tạp chí & ngành
Environmental Modelling & Software (Elsevier) — có thể xem là giao thoa Thủy văn + Khoa học máy tính lượng tử, không hẳn CNTT/AI cổ điển — DOI 10.1016/j.envsoft.2025.106736 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS (18 vùng HUC2, 671 lưu vực), streamflow + nhiệt độ nước, train 2000-2014 test 2015-2022
7. Code free
Có — github.com/Clemson-Hydroinformatics-Lab/HydroQuantum- (link từ trang lab tác giả Vidya Samadi)

#21 — Probabilistic physics-guided deep neural networks with recurrence and attention mechanisms for interpretable daily streamflow simulation
Mục
Nội dung
1. Năm
2025
2. Cited by
8
3. Kiến trúc model
DeepAR (Deep Autoregressive Recurrent) và TFT (Temporal Fusion Transformer), có phiên bản "physics-guided" tích hợp catchment attributes, kèm quantile regression để định lượng uncertainty
4. Tạp chí & ngành
Water Resources Research (AGU) — Thủy văn, không CNTT/AI — DOI 10.1029/2025wr040173 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (531/671 lưu vực, 18 vùng HUC2 trên toàn CONUS)
7. Code free
Không — code "available after publication from the corresponding author upon request"

#22 — A deep state space model for rainfall-runoff simulations (S4D-FT)
Mục
Nội dung
1. Năm
2025
2. Cited by
15
3. Kiến trúc model
S4D-FT (Frequency Tuned Diagonal State Space Sequence) — kiến trúc State Space Model (họ hàng với Mamba), so với LSTM chuẩn và mô hình vật lý Sacramento Soil Moisture Accounting. Tóm tắt bài: S4D-FT vượt LSTM trên nhiều vùng (tóm tắt không nêu số cụ thể)
4. Tạp chí & ngành
Water Resources Research 61(12), e2025WR039888 (AGU, 12/2025), DOI 10.1029/2025WR039888 — Thủy văn, không CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (531 lưu vực CONUS chuẩn)
7. Code free
Có — github.com/Pandas-Paws/S4D_rainfall_runoff_simulations (không có file LICENSE; code dựa trên kratzert/lstm_for_pub; train S4D, LSTM, MC-LSTM trên 531 lưu vực CAMELS, forcing NLDAS bản cập nhật từ HydroShare; có cả thiết lập PUB) (kiểm repo qua GitHub API 25/9/2026)

#23 — A process-driven deep learning hydrological model for daily rainfall-runoff simulation (PRNN-EA-LSTM)
Mục
Nội dung
1. Năm
2024
2. Cited by
44 (OA) / 46 (S2)
3. Kiến trúc model
PRNN-EA-LSTM — nhúng mô hình conceptual EXP-HYDRO vào một RNN cell làm "process driver", kết hợp Entity-Aware LSTM (EA-LSTM) làm post-processor
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2024.131434 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (671 lưu vực CONUS); so PRNN-EA-LSTM với LSTM, EA-LSTM, EXP-HYDRO
7. Code free
Không thấy — trang ScienceDirect không có link code (trang ScienceDirect chỉ cho xem tóm tắt + mục Data availability; tìm GitHub/Zenodo theo tên bài không ra)

#24 — Meta-LSTM in hydrology: Advancing runoff predictions through model-agnostic meta-learning
Mục
Nội dung
1. Năm
2024
2. Cited by
40
3. Kiến trúc model
Meta-LSTM dựa trên khung Model-Agnostic Meta-Learning (MAML), fine-tune động tham số để thích ứng các kịch bản runoff hiếm gặp, cải thiện dự đoán cực trị (FLV +2.73%, FHV +11.04%)
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2024.131521 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS (US) + CAMELS-AUS
7. Code free
Không thấy — mục Data availability chỉ nói CAMELS và CAMELS-AUS tải tự do, không có link code

#25 — Deciphering the mechanism of better predictions of regional LSTM models in ungauged basins
Mục
Nội dung
1. Năm
2024
2. Cited by
37
3. Kiến trúc model
LSTM khu vực (regional LSTM), so sánh chiến lược huấn luyện truyền thống (có/không static attributes) với chiến lược huấn luyện theo phân loại nhóm lưu vực (classification-based)
4. Tạp chí & ngành
Water Resources Research (AGU) — Thủy văn, không CNTT/AI — DOI 10.1029/2023wr035876 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (363/671 lưu vực, chia 7 vùng CONUS)
7. Code free
Có — dùng mã nguồn mở của Kratzert et al. 2019 (neuralHydrology)

#26 — A differentiable, physics-based hydrological model and its evaluation for data-limited basins
Mục
Nội dung
1. Năm
2025
2. Cited by
23
3. Kiến trúc model
dXAJ (differentiable Xin'anjiang model — cấu trúc XAJ + LSTM học tham số) và dXAJ_nn (thay module bốc thoát hơi bằng neural network)
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2024.132471 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core (một phần) — 5 lưu vực Tam Hiệp (Trung Quốc) + 8 lưu vực từ CAMELS
7. Code free
Có — github.com/OuyangWenyu/HydroDHM (code của bài J. Hydrol., DOI 10.1016/j.jhydrol.2024.132471; có LSTM, DPL-XAJ, hỗ trợ CAMELS). Tìm qua rà GitHub 25/9/2026

#27 — Effects of Multi-Step-Ahead Prediction Strategies on LSTM-Based Runoff Prediction
Mục
Nội dung
1. Năm
2025
2. Cited by
13
3. Kiến trúc model
LSTM chuẩn, so sánh 3 chiến lược dự đoán đa bước: multi-output, direct, recursive, qua nhiều lead-time (1-7 ngày)
4. Tạp chí & ngành
Water Resources Management (Springer) — Thủy văn, không CNTT/AI — DOI 10.1007/s11269-025-04332-1 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS (48 lưu vực chọn ngẫu nhiên)
7. Code free
Không — "data ... available from the corresponding author upon reasonable request"

#28 — Investigate the rainfall-runoff relationship and hydrological concepts inside LSTM
Mục
Nội dung
1. Năm
2025
2. Cited by
14
3. Kiến trúc model
LSTM chuẩn, phân tích nội tại bằng linear probes để khảo sát cách LSTM hình thành khái niệm thủy văn (soil water, snow water equivalent) qua cell state
4. Tạp chí & ngành
Environmental Modelling & Software (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.envsoft.2025.106527 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (281 lưu vực) + Caravan (bổ sung soil water/snow)
7. Code free
Có — github.com/Yq-H47/Hydrological-concepts-inside-LSTM

#29 — Dive into transfer-learning for daily rainfall-runoff modeling in data-limited basins
Mục
Nội dung
1. Năm
2025
2. Cited by
16
3. Kiến trúc model
LSTM huấn luyện trên CAMELS làm source model, sau đó transfer learning sang 2 lưu vực Trung Quốc dữ liệu hạn chế (Duoyingping, Fujiangqiao)
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2025.133063 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS (523 lưu vực) làm source model transfer learning
7. Code free
Không thấy — mục Data availability chỉ cảm ơn bộ CAMELS, không có link code

#30 — Synergizing Intuitive Physics and Big Data in Deep Learning (DPL)
Mục
Nội dung
1. Năm
2024
2. Cited by
16
3. Kiến trúc model
Deep Process Learning (DPL) — 6 biến thể DL submodel được dẫn dắt bởi trực giác vật lý (physics-guided), tích hợp hiểu biết quá trình địa vật lý vào neural network
4. Tạp chí & ngành
Water Resources Research (AGU) — Thủy văn, không CNTT/AI — DOI 10.1029/2024wr037582 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (531 lưu vực); so 6 biến thể DPL-H với LSTM và MC-LSTM
7. Code free
Có — Zenodo zenodo.org/records/13335553

#31 — Fine-tuning long short-term memory models for seamless transition in hydrological modelling
Mục
Nội dung
1. Năm
2025
2. Cited by
18
3. Kiến trúc model
LSTM pre-trained trên Caravan (forcing ERA5-Land), fine-tune để chuyển sang dùng dữ liệu vệ tinh mưa gần thời gian thực (near-real-time satellite precipitation)
4. Tạp chí & ngành
Environmental Modelling & Software (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.envsoft.2025.106350 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟡 Reference — Caravan (forcing ERA5-Land); con số "472 lưu vực CAMELS-US" ghi trước đây không tìm thấy trong bài
7. Code free
Có — bài ghi "I have shared the link to my data and code at the Software and Data availability section" (link nằm trong toàn văn, bản xem được không hiện)

#32 — Developing an explainable deep learning module based on the LSTM framework for flood prediction
Mục
Nội dung
1. Năm
2025
2. Cited by
14
3. Kiến trúc model
LSTM + gated module đơn giản hóa chèn giữa input và LSTM để tăng khả năng giải thích, nhóm thông tin gate thành 4 nhóm (ngắn/dài hạn của precip/temp)
4. Tạp chí & ngành
Frontiers in Water — Thủy văn/Khoa học Trái đất, không CNTT/AI — DOI 10.3389/frwa.2025.1562842 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1 — SJR 0,943; ngành Water Science and Technology (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟢 Core — CAMELS-US (531 lưu vực CONUS)
7. Code free
Không — không có link code trong bài

#33 — Assessing the transferability of LSTM-based streamflow models under varying source basin diversity and target data availability (Mangla Basin, Pakistan)
Mục
Nội dung
1. Năm
2026
2. Cited by
1
3. Kiến trúc model
LSTM tiền huấn luyện trên nguồn đa dạng (CAMELS-US, Caravan) rồi chuyển sang lưu vực Mangla thiếu dữ liệu. Tác giả: Adnan, Ouyang, Ye, Khan, Chai, Ma
4. Tạp chí & ngành
Journal of Hydrology: Regional Studies 64:103329 (Elsevier), DOI 10.1016/j.ejrh.2026.103329 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1 (theo #85, #92)
6. Dataset
🟢 Core (một phần) — 531 lưu vực CAMELS-US + ~5.000 lưu vực Caravan làm nguồn; lưu vực đích Mangla (Pakistan) không công khai
7. Code free
Không — "Data will be made available on request"

[Gộp 25/9/2026: bài này từng bị nhập trùng thành #115]

#34 — Is smart sampling worth it? Impact of training data selection on the performance of LSTMs in streamflow prediction
Mục
Nội dung
1. Năm
2026
2. Cited by
2
3. Kiến trúc model
LSTM chuẩn, ablation study 4 nhóm chiến lược lấy mẫu dữ liệu train (sampling strategies)
4. Tạp chí & ngành
Hydrology Research (IWA Publishing) — Thủy văn, không CNTT/AI — DOI 10.2166/nh.2026.119 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1 — SJR 0,812; ngành Water Science and Technology (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟢 Core — CAMELS-US
7. Code free
Có — Zenodo zenodo.org/records/17816198

#35 — Temporal inductive biases in hourly flood forecasting: a comparative analysis of recurrent, attention-based, and state-space neural networks
Mục
Nội dung
1. Năm
2026
2. Cited by
1
3. Kiến trúc model
Benchmark dự báo lưu lượng theo giờ 1–24 giờ: mô hình tuyến tính/lớp đơn giản, LSTM, Transformer chuyên chuỗi thời gian (Informer, Autoformer, Patch Transformer/PatchTST) và SSM (S-Mamba). Hai chế độ: có dùng lưu lượng quá khứ làm đầu vào (discharge-assisted) và thuần mưa–dòng chảy. Kết quả: Patch Transformer tốt nhất; LSTM vẫn vững; Mamba cân bằng tốt giữa độ chính xác và chi phí tính toán. Giải thích bằng attention + Integrated Gradients. Tác giả: Zhang, Ouyang, Xu, Liu, Wang, Tang (cùng nhóm #108)
4. Tạp chí & ngành
Journal of Hydrology 677 Part A:135727 (Elsevier, 9/2026), DOI 10.1016/j.jhydrol.2026.135727 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 516 lưu vực CAMELS-US theo giờ
7. Code free
Có — github.com/binbinlan/RF-Bench. File LICENSE là Apache-2.0 nhưng huy hiệu README ghi CC BY-NC-SA (mâu thuẫn); có link demo Colab. models/ có S_Mamba, PatchTST, Informer, Autoformer, Transformer, LSTM, DLinear. S_Mamba dùng mamba-ssm 1.2.0 + torch 1.11, kiểu "S-Mamba" (Wang et al. 2024): mỗi BIẾN là 1 token, 2 khối Mamba quét qua các biến — không quét theo thời gian. Dữ liệu CAMELS theo giờ (cột QObs(mm/h)) không kèm trong repo, phải tự tải (kiểm repo qua GitHub API 25/9/2026)

[Đọc code 25/9/2026 — lỗi phương pháp trong code công khai: (1) `preprocess_data.py` giữ `.head(10000)` → chỉ 10.000 giờ (~1,1 năm) đầu mỗi lưu vực, không phải chuỗi dài năm; (2) `Dataset_Runoff` nối mọi lưu vực thành 1 bảng rồi chia 70/10/20 theo khối → cửa sổ trượt cắt ngang ranh giới 2 lưu vực; (3) dòng fit StandardScaler trên tập train bị comment, thay bằng fit trên chính tập đang dùng (val/test tự chuẩn hóa bằng thống kê của mình → rò rỉ); (4) `add_attributes` hard-code đường dẫn `D:/new_models/...`. Chưa xác nhận bài báo có dùng đúng thiết lập này không]

[Gộp 25/9/2026: bài này từng bị nhập trùng thành #121]

#36 — Benchmarking structured state space models for differentiable parameter learning: A large-scale evaluation on accuracy and physical consistency
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (OA/S2)
3. Kiến trúc model
S4D và S5D (structured state space, KHÔNG phải Mamba) cho differentiable parameter learning, so với LSTM, TCN, Transformer, TimeMixer — NSE S4D 0,756 vs LSTM 0,742
4. Tạp chí & ngành
Expert Systems with Applications 329 (11/2026), bài 133040, DOI 10.1016/j.eswa.2026.133040 (Elsevier; tác giả Jing, Luo, Yang, Zuo) — ngành CNTT/AI (Q1, Artificial Intelligence + Computer Science Applications)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US 671 lưu vực, streamflow từ trạm USGS
7. Code free
Có — **chính thức**: mục "Data and software availability" của bài ghi `github.com/chooron/dmg-research/tree/master/project/bettermodel` (đọc toàn văn 26/9/2026).

[Đọc code 25/9/2026: dmg-research có báo cáo `project/bettermodel/ablation/report/multiseed_7model_comparison_report.md` — 7 mô hình sinh tham số dHBV trên camels_671, test 1995–2010, 5 seed, S5D/S4D đứng đầu (NSE trung bình ~0,689/0,688 vs LSTM 0,672) → khớp nội dung bài nhưng số khác bài (0,756/0,742) và dùng TSMixer thay TimeMixer; repo 475 MB khi clone, không LICENSE. Repo `chooron/HydroModelsPaperCode` là code của bài HydroModels.jl — KHÔNG phải bài này]

#37 — HydEquivNet: bridging classical unit hydrograph theory and data-driven hydrology via scale-equivariant learning
Mục
Nội dung
1. Năm
2026
2. Cited by
Mới (đã có bài khác trích dẫn)
3. Kiến trúc model
HydEquivNet (HEN) — mạng scale-equivariant kết hợp lý thuyết unit hydrograph cổ điển với học sâu, dùng phép tích chập nhân quả 1D; so sánh với LSTM và Mamba (selective state-space model)
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2026.136146 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS-US
7. Code free
Không thấy (trang ScienceDirect chỉ cho xem tóm tắt + mục Data availability; tìm GitHub/Zenodo theo tên bài không ra)

[Rà GitHub 26/9/2026: có repo github.com/Zafkiellll/HydEquivNet nhưng RỖNG (chỉ README 1 dòng, cập nhật 5/2026). Đoạn trích Google Scholar (người dùng tra 26/9/2026) cho thấy bài có dùng khối Mamba selective state-space — chưa đọc được toàn văn để xác nhận vai trò]

#38 — A Hierarchical Deep Learning Framework for Runoff Prediction Using Raster-Based Spatial Representations
Mục
Nội dung
1. Năm
2026
2. Cited by
Mới
3. Kiến trúc model
Framework phân cấp mã hóa thuộc tính lưu vực tĩnh dạng raster bằng CNN (fixed-grid và adaptive multi-patch), so với bảng thuộc tính truyền thống
4. Tạp chí & ngành
Water Resources Research (AGU) — Thủy văn, không CNTT/AI — DOI 10.1029/2026wr043815 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS-US (bản chính thức WRR 2026, DOI 10.1029/2026WR043815)
7. Code free
Có — github.com/Dehui-nb/CNN-LSTM-for-run-off-simulation + Zenodo

[Đọc code 26/9/2026: github Dehui-nb/CNN-LSTM-for-run-off-simulation: giao thức Kratzert (train 1999–2008, đánh giá 1989–1999, NLDAS), nhiều lead time, không early stopping — cố định epoch 30, không chọn checkpoint theo tập đánh giá; raster thuộc tính tĩnh phải tự tạo]

#39 — Diffusion-Based Probabilistic Modeling for Hourly Streamflow Prediction and Assimilation (h-Diffusion)
Mục
Nội dung
1. Năm
2026
2. Cited by
3
3. Kiến trúc model
h-Diffusion — denoising diffusion probabilistic model với backbone MTS-LSTM, dùng kỹ thuật inpainting (RePaint) để đồng hóa dữ liệu quan trắc theo giờ
4. Tạp chí & ngành
Water Resources Research (AGU) — Thủy văn, không CNTT/AI — DOI 10.1029/2025wr042720 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US hourly (516 lưu vực), nhóm Chaopeng Shen (Penn State)
7. Code free
Có — Zenodo (models + code)

[Đọc code 26/9/2026: Zenodo 21416183 (code 0,1 MB) + 21414935 (mô hình 9 MB, dự báo 12 GB): cấu hình dataset_config_mforcing: train 1990–2003, val = test = 2003–2008 (TRÙNG — chưa xác nhận val có dùng để chọn mô hình)]

#40 — Physics-informed, Differentiable hydrologic models for capturing unseen extreme events
Mục
Nội dung
1. Năm
2026
2. Cited by
12
3. Kiến trúc model
Mô hình differentiable physics-informed, tập trung khả năng dự đoán sự kiện cực đoan chưa từng thấy trong dữ liệu train
4. Tạp chí & ngành
Water Resources Research (AGU) — Thủy văn, không CNTT/AI — DOI 10.1029/2025wr040414 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (tập con 498/671 lưu vực), mô hình δHBV1.0/δHBV1.1p, nhóm Chaopeng Shen (Penn State); bản chính thức WRR 2026, DOI 10.1029/2025WR040414
7. Code free
Có — Zenodo doi:10.5281/zenodo.15978036 (δHBV1.1p), 10.5281/zenodo.6326394, 10.5281/zenodo.7314083

#41 — Learning geometry-aware streamflow forecast models from a large-scale river graph dataset
Mục
Nội dung
1. Năm
2026
2. Cited by
Mới
3. Kiến trúc model
GNN message-passing (MGN, MGN-LSTM, GWN) học trên CAMELS-G (dataset đồ thị mới: >500 mạng lưới sông Mỹ, ~100.000 node sông, ~6 tỷ điểm dữ liệu, dựa trên hydrofabric + NWM)
4. Tạp chí & ngành
Journal of Hydrology (Elsevier) — Thủy văn, không CNTT/AI — DOI 10.1016/j.jhydrol.2026.136059 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core (biến thể đồ thị) — CAMELS-G, xây trên nền dữ liệu CAMELS-US
7. Code free
Dữ liệu: Có — Zenodo doi:10.5281/zenodo.19657857 (CAMELS-G Graph Dataset); code: không thấy link (trang ScienceDirect chỉ cho xem tóm tắt + mục Data availability; tìm GitHub/Zenodo theo tên bài không ra)

#42 — Frequency-Domain Convolutional Network With Historical Data Fusion Module for Regional Streamflow Prediction (TFRN)
Mục
Nội dung
1. Năm
2025 (online 2/9/2025)
2. Cited by
1 (theo trang IEEE Xplore)
3. Kiến trúc model
TFRN (Temporal Fusion Runoff Network) — CNN trích đặc trưng ở cả miền thời gian và miền tần số, kết hợp LSTM + Transformer với module adaptive fusion để đưa thông tin lịch sử xa vào LSTM; dự báo runoff 7 ngày
4. Tạp chí & ngành
IEEE Transactions on Geoscience and Remote Sensing, vol. 63, art. 4211314, DOI 10.1109/TGRS.2025.3605332 — venue IEEE (Viễn thám + Kỹ thuật điện/điện tử), không phải CNTT/AI thuần
5. Xếp hạng Q
Q1 (Scimago: Earth & Planetary Sciences + Electrical & Electronic Engineering)
6. Dataset
🟢 Core — CAMELS 671 + CAMELS-AUS 561 (đọc toàn văn 26/9/2026, CHECKPDF Mục 5.4: NSE trong bài là NSE gộp mọi lưu vực, cấu hình chọn theo bảng kỳ test)
7. Code free
Có — github.com/redtea-code/TFRN

#43 — Foundation-Scale Satellite Embeddings Reframe Hydrological Generalization as a Representation Problem
Mục
Nội dung
1. Năm
2026
2. Cited by
Mới
3. Kiến trúc model
LSTM decoder có điều kiện theo bộ mã hóa đặc trưng lưu vực; dùng satellite embeddings 64 chiều từ Google AlphaEarth Foundations thay/bổ sung thuộc tính tĩnh để tăng khả năng tổng quát hóa sang lưu vực không có trạm đo
4. Tạp chí & ngành
Geophysical Research Letters (AGU) — Thủy văn/Khoa học Trái đất, không CNTT/AI — DOI 10.1029/2025gl121604 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟡 Reference — CAMELS-AUS v2 (455 lưu vực), không dùng CAMELS-US
7. Code free
Có — Zenodo doi:10.5281/zenodo.19234398

#44 — A variational approach at uncertainty estimation in data-driven rainfall-runoff modeling
Mục
Nội dung
1. Năm
2026
2. Cited by
Mới
3. Kiến trúc model
Variational LSTM (vLSTM) — dùng variational inference cho dự báo xác suất phi tham số, decoder lan truyền bất định; so với LSTM đầu ra GMM và CMAL. Tác giả: Álvarez Chaves, Acuña Espinoza, Klotz, Gupta, Ehret, Guthke
4. Tạp chí & ngành
Machine Learning: Earth (IOP Publishing), vol 2, 025009, DOI 10.1088/3049-4753/ae89ba — giao thoa ML + Khoa học Trái đất
5. Xếp hạng Q
Chưa có hạng — tạp chí mới, chưa có trên Scimago (người dùng tra 25/9/2026)
6. Dataset
🟢 Core — CAMELS-US (531 lưu vực)
7. Code free
Có — DaRUS doi:10.18419/DARUS-5118

#45 — Transfer Learning Using Inaccurate Physics Rule for Streamflow Prediction
Mục
Nội dung
1. Năm
2024
2. Cited by
0 (Crossref, 25/9/2026 — Crossref thường đếm thấp hơn Scholar với kỷ yếu hội nghị)
3. Kiến trúc model
Meta-transfer learning tận dụng luật vật lý không chính xác để dự báo lưu lượng ở lưu vực không có trạm đo (prediction in ungauged basins). Tác giả: Tianshu Bao, Taylor T. Johnson, Xiaowei Jia
4. Tạp chí & ngành
Hội nghị IJCAI 2024 (33rd International Joint Conference on Artificial Intelligence), track AI for Good, tr. 7170–7178, DOI 10.24963/ijcai.2024/793 — ✅ ngành CNTT/AI
5. Xếp hạng Q
Hội nghị — CORE A*
6. Dataset
🟢 Core — CAMELS
7. Code free
Không thấy

#46 — Testing discharge assimilation strategies to enhance short-range AI-based operational rainfall–runoff forecasts
Mục
Nội dung
1. Năm
2026
2. Cited by
Mới
3. Kiến trúc model
MLP điều phối 3 chiến lược đồng hóa lưu lượng quan trắc (train 20 lần với seed khác nhau), trên nền LSTM đã train sẵn (Kratzert et al. 2019) và SAC-SMA. Tác giả: Saint-Fleur, Gaume, Surmont, Akil, Theriez
4. Tạp chí & ngành
Hydrology and Earth System Sciences (HESS), vol 30, tr. 3497–3527, DOI 10.5194/hess-30-3497-2026 — Thủy văn, không CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS-US (531 lưu vực; 56 cho đánh giá ensemble) + CAMELS-FR (338 lưu vực)
7. Code free
Có — Zenodo doi:10.5281/zenodo.20415493

[Đọc code 26/9/2026: Zenodo 20415493 (4,6 MB, CC BY 4.0): khung đồng hóa lưu lượng cho dự báo vận hành, CAMELS-US + Pháp]

#47 — ZeroDiff: Zero-Shot Time Series Reconstruction via Informed-Prior Diffusion
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Google Scholar chưa hiện số trích dẫn, tra 25/9/2026)
3. Kiến trúc model
ZeroDiff — prior có thông tin (Conditional VAE ước lượng moment + LSTM học động lực) dựng từ biến ngoại sinh, sau đó mô hình diffusion (encoder hai chiều + module wave fusion) hiệu chỉnh sai số tái tạo. Train trên lưu vực có trạm, suy ra chuỗi lưu lượng ở lưu vực chưa có trạm (zero-shot). So sánh 4 kiến trúc nền trên CAMELS, có Mamba (theo trích đoạn Google Scholar: Mamba đạt NSE riêng lẻ cao nhất 0,557). Tác giả: Yingda Fan, Dan Lu, Xiaowei Jia
4. Tạp chí & ngành
Hội nghị ICML 2026 (Forty-third International Conference on Machine Learning), OpenReview id 5EKo3JdcR5 — ✅ ngành CNTT/AI
5. Xếp hạng Q
Hội nghị — CORE A*
6. Dataset
🟢 Core — CAMELS (tập con benchmark 531 lưu vực; 200 lưu vực cho phần minh họa, theo README repo). Bài còn dùng dữ liệu năng lượng mặt trời, nhiệt độ nước, phát thải methane
7. Code free
Có — github.com/YingdaFan/ZeroDiff-ICML2026

[Đọc code 26/9/2026: Repo kèm file CAMELS parquet; bài toán tái tạo chuỗi zero-shot (không dự báo); bảng so sánh có Mamba (NSE 0,557 cao nhất khi đứng riêng)]

#48 — A dynamic-gated Mixture-of-Experts framework improves and interprets daily streamflow simulation (HydroMoE)
Mục
Nội dung
1. Năm
2026
2. Cited by
1
3. Kiến trúc model
HydroMoE — Mixture-of-Experts cổng động: mỗi quá trình thủy văn (tuyết tan, bốc thoát hơi, sinh dòng chảy, thoát nước) có 1 chuyên gia vật lý + 1 chuyên gia mạng nơ-ron, cổng theo khí tượng chọn trọng số; so với LSTM vùng (ensemble) và mô hình khả vi. Tác giả: Yuan, Hu, Zhan, Lin
4. Tạp chí & ngành
Communications Earth & Environment (Nature Portfolio) 7:614, DOI 10.1038/s43247-026-03799-z — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟡 Reference — 550 lưu vực CAMELS-US nhưng lấy từ bản Caravan (forcing khác CAMELS gốc)
7. Code free
Có — github.com/RaySpaniare/HydroMoE + Zenodo doi:10.5281/zenodo.19804505

#49 — Improving deep learning-based streamflow forecasting under trend varying conditions through evaluation of new wavelet preprocessing technique
Mục
Nội dung
1. Năm
2024
2. Cited by
10
3. Kiến trúc model
LSTM, RNN, GRU kết hợp tiền xử lý wavelet MODWET; chọn biến bằng genetic programming (GP) và partial correlation (PCI). Tác giả: Behbahani, Mazarei, Bagtzoglou
4. Tạp chí & ngành
Stochastic Environmental Research and Risk Assessment 38:3963–3984 (Springer), DOI 10.1007/s00477-024-02788-y — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — chỉ 3 lưu vực CAMELS-US, dòng chảy theo tháng
7. Code free
Không — "No datasets were generated or analysed", không có link code

#50 — An interpretable and lightweight dynamic framework for streamflow–meteorology networks to enhance daily streamflow forecasting (IHRC-SMSE)
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Scholar chưa hiện)
3. Kiến trúc model
IHRC-SMSE — reservoir computing có suy luận cấu trúc mạng thủy văn + similarity embedding; so với TCN, GRU, Seq2Seq, LSTM, HoGRC. Tác giả: Qian, Zhang, Zhao, Wang, Tang
4. Tạp chí & ngành
Water Resources Research 62, DOI 10.1029/2025WR043249 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 50 lưu vực CAMELS chia 8 nhóm
7. Code free
Không thấy — mục Open Research chỉ dẫn link dữ liệu CAMELS

#51 — Exploring hydrological variable interconnections and enhancing predictions for data-limited basins through multi-task learning
Mục
Nội dung
1. Năm
2025
2. Cited by
9
3. Kiến trúc model
LSTM đa nhiệm (multi-task: dòng chảy + độ ẩm đất SMAP), probing biến ẩn. Tác giả: Ouyang, Gu, Ye, Liu, Zhang
4. Tạp chí & ngành
Water Resources Research 61, DOI 10.1029/2023WR036593 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 591 lưu vực CAMELS-US (forcing NLDAS-II, thuộc tính CAMELS)
7. Code free
Có — Zenodo doi:10.5281/zenodo.10024011 (code) + 10.5281/zenodo.12165586 (dữ liệu đã xử lý)

#52 — Hierarchical deep learning for consistent multi-timescale hydrological forecasting
Mục
Nội dung
1. Năm
2025
2. Cited by
10
3. Kiến trúc model
HDL — LSTM + temporal hierarchical reconciliation (THR) bằng mạng nơ-ron, dự báo nhất quán ngày/tuần (7 ngày). Tác giả: Jahangir, Quilty
4. Tạp chí & ngành
Water Resources Research 61, DOI 10.1029/2024WR038105 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — hơn 400 lưu vực CAMELS-US (có dùng thêm Caravan)
7. Code free
Có một phần — Zenodo 15596532 chỉ có dữ liệu (CSV), không có code

[Đọc 26/9/2026: Zenodo 10.5281/zenodo.15596532 chỉ chứa DỮ LIỆU (4 file CSV train/test 421 lưu vực, bản Daymet + ERA5, 383 MB) — KHÔNG có code → sửa mục Code: "Có một phần — chỉ dữ liệu"]

#53 — HydroTFT: a cross-basin attention model for multi-horizon rainfall–runoff prediction
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Scholar chưa hiện)
3. Kiến trúc model
HydroTFT — dựa trên Temporal Fusion Transformer: variable-selection network động/tĩnh, LSTM encoder, multi-head attention diễn giải được; 1 mô hình vùng cho toàn bộ lưu vực. Tác giả: Cheng, Fan, Jia, Xie, Lu (Khoa Khoa học máy tính Rutgers + ORNL)
4. Tạp chí & ngành
Machine Learning: Earth 2(2):025014 (IOP), DOI 10.1088/3049-4753/aea164 — tạp chí ML cho khoa học Trái đất
5. Xếp hạng Q
Chưa có hạng — tạp chí mới, chưa có trên Scimago (người dùng tra 25/9/2026)
6. Dataset
🟢 Core — 531 lưu vực CAMELS
7. Code free
Có — github.com/qxc101/HydroTFT

#54 — Knowledge-guided machine learning for operational flood forecasting (FHNN)
Mục
Nội dung
1. Năm
2025
2. Cited by
7
3. Kiến trúc model
FHNN (Factorized Hierarchical Neural Network) — mô hình nghịch suy ra trạng thái lưu vực + mô hình thuận dự báo dòng chảy; so với LSTM tự hồi quy và dự báo của NWS. Tác giả: McEachran, Ghosh, Renganathan, …, Vipin Kumar (Khoa CS&E Đại học Minnesota)
4. Tạp chí & ngành
Water Resources Research 61(11), DOI 10.1029/2024WR039064 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS-US
7. Code free
Có — github.com/arvindrenga96/FHNN + HydroShare (dữ liệu)

[Đọc code 26/9/2026: Repo THIẾU thư mục DATA (README nhắc DATA/preprocessData.ipynb nhưng không có), tiền xử lý ra mảng numpy làm ngoài repo, đường dẫn cứng /home/kumarv/...; config: train 1985–1993, val 1993–1995, test 1995–2005, cửa sổ 365, dự báo 1 ngày; baseline RRFormer, TFT, AR-LSTM, CT-LSTM]

#55 — An argument for parsimony in differentiable hydrologic models
Mục
Nội dung
1. Năm
2026
2. Cited by
1
3. Kiến trúc model
Mô hình khả vi HBV: MLP (tham số tĩnh) vs LSTM (tham số động), đơn/ensemble tham số; giải thích bằng Integrated Gradients. Tác giả: Poudel, Steinschneider
4. Tạp chí & ngành
HESS 30:5173–5193, DOI 10.5194/hess-30-5173-2026 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS-US (425 train, 106 giữ lại để thử lưu vực không trạm)
7. Code free
Có — github.com/snpoudel/diff-hydro

#56 — How to deal w___ missing input data
Mục
Nội dung
1. Năm
2025
2. Cited by
10
3. Kiến trúc model
LSTM với 3 cơ chế xử lý thiếu forcing: input replacing, masked mean, attention. Tác giả: Gauch, Kratzert, Klotz, Nearing, Cohen, Gilon
4. Tạp chí & ngành
HESS 29:6221, DOI 10.5194/hess-29-6221-2025 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS-US (3 forcing Daymet/Maurer/NLDAS)
7. Code free
Có — github.com/gauchm/missing-inputs + Zenodo doi:10.5281/zenodo.17362593

[Đọc code 26/9/2026: Repo chỉ gồm notebook vẽ hình + bản vá cho neuralhydrology; trọng số/dự báo trên Zenodo]

#57 — Improving adaptive runoff forecasts in data-scarce watersheds through personalized federated learning (FedHydroDSW)
Mục
Nội dung
1. Năm
2024 (kỷ yếu in 2025)
2. Cited by
0 (Scholar chưa hiện)
3. Kiến trúc model
FedHydroDSW — học liên kết cá nhân hóa (personalized federated learning) cho dự báo dòng chảy ở lưu vực thiếu dữ liệu. Tác giả: Xie, Zhang, Wang, Jie, Fang, Cai (Đại học Hà Hải)
4. Tạp chí & ngành
Hội nghị ICPR 2024 (International Conference on Pattern Recognition), LNCS, tr. 180–198, DOI 10.1007/978-3-031-78183-4_12 — ✅ ngành CNTT/AI
5. Xếp hạng Q
Hội nghị — CORE B (ICORE2026, người dùng tra portal.core.edu.au 25/9/2026)
6. Dataset
🟢 Core — CAMELS (số lưu vực chưa xác nhận: trang chỉ ghi "nhiều lưu vực thiếu dữ liệu")
7. Code free
Có — github.com/xyjie37/FedHydroDSW

[Đọc code 26/9/2026: Repo có sẵn 1 lưu vực mẫu + 2 file .pt; thiết kế cho 8 lưu vực (7 giàu dữ liệu + 1 thiếu), PyTorch 1.11]

#58 — A data-centric perspective on the information needed for hydrological uncertainty predictions
Mục
Nội dung
1. Năm
2024
2. Cited by
21
3. Kiến trúc model
HopCPT (conformal prediction với Modern Hopfield Network) cho khoảng bất định dòng chảy; so với CMAL, Bluecat, kNN. Tác giả: Auer, Gauch, Kratzert, Nearing, Hochreiter, Klotz
4. Tạp chí & ngành
HESS 28:4099–4126, DOI 10.5194/hess-28-4099-2024 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS
7. Code free
Có — Zenodo doi:10.5281/zenodo.10674231 (code) + 3 bộ Zenodo mô hình/kết quả

#59 — Exploring Kolmogorov-Arnold neural networks for hybrid and transparent hydrological modeling
Mục
Nội dung
1. Năm
2025
2. Cited by
4
3. Kiến trúc model
KAN (Kolmogorov-Arnold Network) trong mô hình lai thủy văn, so với LSTM và MLP. Tác giả: Jing, Yang, Luo, Zuo
4. Tạp chí & ngành
Environmental Modelling & Software 193:106648 (Elsevier), DOI 10.1016/j.envsoft.2025.106648 — Mô hình hóa môi trường, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 569 lưu vực CAMELS
7. Code free
Có — github.com/chooron/Hydro-KAN-V1

#60 — Soil Water Accounting Network (SWAN): a novel neural network for modeling conceptual hydrological processes
Mục
Nội dung
1. Năm
2025
2. Cited by
5
3. Kiến trúc model
SWAN — mạng nơ-ron mô phỏng các quá trình khái niệm (tích trữ nước trong đất); so với LSTM, EA-LSTM, MC-LSTM. Tác giả: Fang, Qu, Li, Li, Shi, Sun, Yang, …
4. Tạp chí & ngành
Journal of Hydrology 661:133562 (Elsevier), DOI 10.1016/j.jhydrol.2025.133562 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 374 lưu vực CAMELS
7. Code free
Không thấy — trang ScienceDirect không có link code (chỉ xem được tóm tắt)

#61 — Monte-Carlo-assisted endo-exo temporal transformer for high-confidence interval forecasting of daily runoff (ETT)
Mục
Nội dung
1. Năm
2026
2. Cited by
8
3. Kiến trúc model
ETT (Endo-Exo Temporal Transformer) — module trộn đặc trưng nội sinh/ngoại sinh + LSTM + Efficient Additive Attention trên Transformer; khoảng tin cậy bằng Monte Carlo; so với 7 mô hình (Transformer, LSTM…). Tác giả: Hu, Xu, Wang, Wang, Li
4. Tạp chí & ngành
Stochastic Environmental Research and Risk Assessment 40(4):82 (Springer), DOI 10.1007/s00477-026-03206-1 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — chỉ 4 trạm CAMELS-US
7. Code free
Có — github.com/wangwen1621/ETT

[Đọc code 26/9/2026: Repo chỉ có README (83 KB, không có main.py như README nói)]

#62 — Using physics-encoded GeoAI to improve the physical realism of deep learning's rainfall-runoff responses under climate change
Mục
Nội dung
1. Năm
2024
2. Cited by
8
3. Kiến trúc model
dNN — mạng nơ-ron mã hóa vật lý (physics-encoded) so với LSTM vùng, đánh giá độ hợp lý vật lý khi khí hậu thay đổi. Tác giả: Li, Hu, Zhang, Shen, Xu, Chen, …
4. Tạp chí & ngành
International Journal of Applied Earth Observation and Geoinformation 133:104101 (Elsevier), DOI 10.1016/j.jag.2024.104101 — Viễn thám/GIS, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — chỉ 29 lưu vực CAMELS ở California
7. Code free
Có một phần — bài dẫn trang GitHub tác giả github.com/lhmygis (chưa xác định repo cụ thể)

#63 — Rainfall-runoff modeling using transductive long short-term memory and hyperparameter optimization
Mục
Nội dung
1. Năm
2025
2. Cited by
1
3. Kiến trúc model
TLSTM — LSTM transductive có spatial attention, tối ưu siêu tham số bằng differential evolution (DE). Tác giả: Zhao Chen, Bao Xie
4. Tạp chí & ngành
Intelligent Decision Technologies 19(6):3707–3743 (SAGE), DOI 10.1177/18724981251389504 — ✅ tạp chí ngành AI/hệ ra quyết định thông minh
5. Xếp hạng Q
Q3 (Scimago SJR 2025 = 0,257; ngành Computer Science – Artificial Intelligence)
6. Dataset
🟢 Core — CAMELS (số lưu vực chưa xác nhận)
7. Code free
Có — github.com/ZhaoChenchina/RRM

[Đọc code 26/9/2026: Repo chỉ có 1 file RRM-TLSTM-DE (193 dòng)]

#64 — Update hydrological states or meteorological forcings? Comparing data assimilation methods for differentiable hydrologic models
Mục
Nội dung
1. Năm
2025
2. Cited by
6
3. Kiến trúc model
So sánh các cách đồng hóa dữ liệu (cập nhật trạng thái vs cập nhật forcing) cho mô hình khả vi δHBV1.1p, đối chiếu LSTM. Tác giả: Jamaat, Song, Rahmani, Liu, Lawson, … (nhóm Chaopeng Shen)
4. Tạp chí & ngành
Journal of Hydrology 663:134137 (Elsevier), DOI 10.1016/j.jhydrol.2025.134137 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531/671 lưu vực CAMELS
7. Code free
Không thấy — trang ScienceDirect không có link code (chỉ xem được tóm tắt)

#65 — Deep learning foundation and pattern models: Challenges in hydrological time series
Mục
Nội dung
1. Năm
2026 (online 2025)
2. Cited by
3
3. Kiến trúc model
LSTM và các mô hình nền tảng chuỗi thời gian (foundation models) cho chuỗi thủy văn. Tác giả: He, Chen, Jafari, Idamekorala, Geoffrey Fox
4. Tạp chí & ngành
The International Journal of High Performance Computing Applications 40(1) (SAGE), DOI 10.1177/10943420251380008 — ✅ tạp chí ngành CNTT (tính toán hiệu năng cao)
5. Xếp hạng Q
Q2 (Scimago SJR 2025 = 0,638; ngành Computer Science)
6. Dataset
🟡 Reference — CAMELS + Caravan (số lưu vực chưa xác nhận)
7. Code free
Có một phần — bài ghi mã "fully open-source" trên Google Colab, trang xem được không hiện link

#66 — Regionalizing hydrologic information for runoff predictions beyond continental boundaries using machine learning
Mục
Nội dung
1. Năm
2025
2. Cited by
2
3. Kiến trúc model
RF, SVR, XGBoost, hồi quy tuyến tính/logistic để chuyển thông tin thủy văn cho lưu vực không có trạm, vượt ranh giới châu lục. Tác giả: Fathi, Awadallah
4. Tạp chí & ngành
Advances in Water Resources 206:105162 (Elsevier), DOI 10.1016/j.advwatres.2025.105162 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core (một phần) — CAMELS-US + CAMELS-GB (số lưu vực chưa xác nhận)
7. Code free
Không thấy — chỉ có link dữ liệu CAMELS

#67 — Technical note: An approach for handling multiple temporal frequencies with different input dimensions using a single LSTM cell (MF-LSTM)
Mục
Nội dung
1. Năm
2025
2. Cited by
19
3. Kiến trúc model
MF-LSTM — 1 ô LSTM nhận input nhiều tần suất thời gian (ngày/giờ) để dự báo theo giờ, giảm 5 lần thời gian xử lý; so với MTS-LSTM, sMTS-LSTM, ODE-LSTM. Tác giả: Acuña Espinoza, Kratzert, Klotz, Gauch, Álvarez Chaves, Loritz, Ehret
4. Tạp chí & ngành
HESS 29:1749–1758, DOI 10.5194/hess-29-1749-2025 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 516 lưu vực CAMELS-US
7. Code free
Có — github.com/eduardoAcunaEspinoza/Hy2DL + Zenodo doi:10.5281/zenodo.14780059

[Đọc code 26/9/2026: Hy2DL: thư viện BSD-3, CAMELS-US/GB/DE/CH/Caravan, có CudaLSTM, MF-LSTM, HBV/SHM lai; dữ liệu tự tải — dùng được làm khung baseline]

[Đọc toàn văn 26/9/2026 (CHECKPDF Mục 4.2): bài là **mô phỏng** theo giờ (không phải dự báo); dữ liệu CAMELS-US **theo giờ** (NLDAS + USGS, Zenodo 4072701, CC BY 4.0, ~20 GB); chuỗi 8.760 giờ, tác giả tự nêu LSTM tốn tính toán và khó học chuỗi dài; train 1990–2003 / val 2003–2008 / test 2008–2018; NSE trung vị 0,75. Hy2DL có chế độ dự báo (`ForecastTester`)]

#68 — Using machine learning to discover parsimonious and physically-interpretable representations of catchment-scale rainfall-runoff dynamics
Mục
Nội dung
1. Năm
2025
2. Cited by
9
3. Kiến trúc model
Mạng dựng từ nút MCP (mass-conserving perceptron) — diễn giải được theo thiết kế, so với LSTM. Tác giả: Yuan-Heng Wang, Hoshin Gupta
4. Tạp chí & ngành
Water Resources Research 61(12), DOI 10.1029/2025WR040178 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 513 lưu vực CAMELS-US
7. Code free
Có — Zenodo (Wang & Gupta, 2025), theo mục Open Research

#69 — Enhancing runoff prediction accuracy of deep learning model using baseflow separation method and timestamp information
Mục
Nội dung
1. Năm
2025
2. Cited by
8
3. Kiến trúc model
Mô hình encoder-decoder (RR-Former, FBS-RR-Former) thêm tách dòng chảy ngầm (baseflow separation) + thông tin thời điểm; so với TimeSter. Tác giả: Jin, Xu, Chen, Cai, Meng
4. Tạp chí & ngành
Journal of Hydrology 662:134044 (Elsevier), DOI 10.1016/j.jhydrol.2025.134044 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core (một phần) — CAMELS + CAMELS-AUS (số lưu vực chưa thống nhất: trang ghi 671, trích đoạn Scholar ghi 150)
7. Code free
Không thấy — trang ScienceDirect không có link code (chỉ xem được tóm tắt)

#70 — Enhancing runoff prediction with causal lag-aware attention and multi-scale fusion in transformer models (CLAMF-Former)
Mục
Nội dung
1. Năm
2026 (online 2025)
2. Cited by
9
3. Kiến trúc model
CLAMF-Former — Transformer có attention nhận biết độ trễ nhân quả + trộn đa tỷ lệ; so với LSTM-MSV-S2S, RR-Former, DTSW-transformer. Tác giả: Yuan, Yan
4. Tạp chí & ngành
Journal of Hydrology 664:134369 (Elsevier), DOI 10.1016/j.jhydrol.2025.134369 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS (trang ghi 674 lưu vực)
7. Code free
Không thấy — chỉ ghi dữ liệu CAMELS tải tự do, không có link code

#71 — Hierarchically disentangled recurrent network for factorizing system dynamics of multi-scale systems: an application on hydrological systems
Mục
Nội dung
1. Năm
2025
2. Cited by
1
3. Kiến trúc model
HDRN/FHNN — mạng hồi quy phân tầng tách động lực đa tỷ lệ thời gian (mô hình nghịch + thuận); so với mô hình vật lý và Transformer. Tác giả: Ghosh, Renganathan, McEachran, …, Vipin Kumar (Đại học Minnesota). Cùng nhóm với #54
4. Tạp chí & ngành
Hội nghị IEEE ICDM 2025 (International Conference on Data Mining), DOI 10.1109/ICDM65498.2025.00131 — ✅ ngành CNTT/AI
5. Xếp hạng Q
Hội nghị — CORE A* (ICORE2026, người dùng tra portal.core.edu.au 25/9/2026)
6. Dataset
🟢 Core (một phần) — thí nghiệm chính trên lưu vực NWS North Central RFC, CAMELS dùng để kiểm chứng thêm — 531 lưu vực (đã xác nhận từ bản arXiv v2, 26/9/2026)
7. Code free
Có (một phần) — README repo `github.com/arvindrenga96/FHNN` ghi repo là mã của bài ICDM này ("ML Model (FHNN — this repo)"); thiếu bước tiền xử lý dữ liệu. CAMELS 531 lưu vực, train 1989–2001 / test 2001–2009 (bản arXiv v2)

#72 — Improving streamflow predictions across CONUS by integrating advanced machine learning models and diverse data
Mục
Nội dung
1. Năm
2024
2. Cited by
14
3. Kiến trúc model
ViT-LSTM — Vision Transformer trích đặc trưng không gian + LSTM; so với LSTM, Geo-LSTM, Geo-ViT-LSTM; thêm dữ liệu viễn thám. Tác giả: Tayal, Renganathan, Dan Lu (ORNL)
4. Tạp chí & ngành
Environmental Research Letters 19(10):104009 (IOP), DOI 10.1088/1748-9326/ad6fb7 — Khoa học môi trường, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS
7. Code free
Không — bài ghi "We plan to host our model code on github", chưa có link

#73 — An evaluation of random forest based input variable selection methods for one month ahead streamflow forecasting
Mục
Nội dung
1. Năm
2024
2. Cited by
35
3. Kiến trúc model
5 phương pháp chọn biến dựa trên Random Forest, ghép với RF, SVR, Gaussian Process, LSTM để dự báo dòng chảy trước 1 tháng. Tác giả: Fang, Ren, Liu, Shang, Jia, Jiang, Zhang
4. Tạp chí & ngành
Scientific Reports 14:29766 (Nature Portfolio) — đa ngành, bài thuộc Thủy văn — DOI 10.1038/s41598-024-81502-y (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — chỉ 3 lưu vực CAMELS, dữ liệu theo tháng
7. Code free
Không thấy — chỉ có link dữ liệu CAMELS và chỉ số khí hậu

#74 — Synergizing machine learning and modified physical models for hydrology modeling: A case study of modified SIMHYD and TANK models
Mục
Nội dung
1. Năm
2026 (online 2025)
2. Cited by
2
3. Kiến trúc model
Mô hình lai: SIMHYD/TANK (và bản sửa GCR) làm lớp vật lý bên trong kiến trúc RNN, mạng nơ-ron làm xương sống. Tác giả: Lei, Cheng, Zhang, Liu
4. Tạp chí & ngành
Journal of Hydrology 664:134452 (Elsevier), DOI 10.1016/j.jhydrol.2025.134452 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 569 lưu vực CAMELS-US
7. Code free
Không thấy — trang ScienceDirect không có link code (chỉ xem được tóm tắt)

#75 — WRO-water: A hydrology-guided multi-head attention transformer for runoff prediction and water quality early warning
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Scholar chưa hiện)
3. Kiến trúc model
WRO-Water — Transformer multi-head attention có dẫn hướng thủy văn (điểm attention theo thời gian phụ thuộc mưa, dòng chảy trước đó) cho dự báo dòng chảy + cảnh báo chất lượng nước. Tác giả: Abbas, Almakki, Albathan, Obidallah, …
4. Tạp chí & ngành
Journal of Environmental Management 415:130545 (Elsevier), DOI 10.1016/j.jenvman.2026.130545 — Quản lý môi trường, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — CAMELS (671 lưu vực Mỹ)
7. Code free
Có một phần — github.com/qaisar256/WRO-Water chỉ có README + 1 file zip, không có LICENSE, cả repo 17 KB dù README ghi có kèm trọng số mô hình (không khớp); tạo 9/7/2026 (kiểm repo qua GitHub API 25/9/2026)

#76 — PatchX: A dual-branch embedding framework with cross-attention for streamflow forecasting
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Scholar chưa hiện)
3. Kiến trúc model
PatchX — embedding 2 nhánh (patch theo kênh + theo biến) nối bằng cross-attention; so với iTransformer, PatchTST, LSTM. Tác giả: Jiang, Wang, Wu
4. Tạp chí & ngành
Environmental Modelling & Software 206:107176 (Elsevier), DOI 10.1016/j.envsoft.2026.107176 — Mô hình hóa môi trường, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — chỉ 6 lưu vực CAMELS
7. Code free
Có — Zenodo doi:10.5281/zenodo.19510951

#77 — Deep learning approaches for streamflow flash drought prediction across the contiguous United States
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Scholar chưa hiện)
3. Kiến trúc model
TFT (Temporal Fusion Transformer) và LSTM có/không thuộc tính tĩnh, dự báo dòng chảy để nhận diện hạn chớp nhoáng (flash drought). Tác giả: Bakar, Nguyen, Kim, Laskhmi
4. Tạp chí & ngành
Journal of Hydrology 677:135956 (Elsevier), DOI 10.1016/j.jhydrol.2026.135956 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 671 lưu vực CAMELS (forcing Daymet)
7. Code free
Không thấy — chỉ có link dữ liệu CAMELS, USGS, Daymet

#78 — VMDI-LSTM-ED: A novel enhanced decomposition ensemble model incorporating data integration for accurate non-stationary daily streamflow forecasting
Mục
Nội dung
1. Năm
2025
2. Cited by
27
3. Kiến trúc model
VMDI-LSTM-ED — phân rã VMD + LSTM encoder-decoder có attention, tích hợp thuộc tính lưu vực; so với VMD-LSTM, Transformer, LSTM-ED. Tác giả: Liu, Xu, Lu
4. Tạp chí & ngành
Journal of Hydrology 653:132769 (Elsevier), DOI 10.1016/j.jhydrol.2025.132769 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — chỉ 6 lưu vực CAMELS-US
7. Code free
Không thấy — trang ScienceDirect không có link code (chỉ xem được tóm tắt)

#79 — Enhancing multi-step ahead daily runoff prediction via HydMoE model with local-global hybrid attention
Mục
Nội dung
1. Năm
2026
2. Cited by
1
3. Kiến trúc model
HydMoE — Mixture of Experts + Time2Vec + Local-Global Hybrid Attention, dự báo 1–7 ngày (NSE 0,762 ở 1 ngày theo tóm tắt); so với Attention+LSTM, TCN+Transformer. Tác giả: Yang, Chen (Tsinghua)
4. Tạp chí & ngành
Water Resources Management 40:156 (Springer), DOI 10.1007/s11269-026-04502-9 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — tập con CAMELS (số lưu vực chưa xác nhận)
7. Code free
Không thấy — trang Springer không có mục code

#80 — Machine learning for a heterogeneous water modeling framework
Mục
Nội dung
1. Năm
2025
2. Cited by
9
3. Kiến trúc model
LSTM và mô hình khái niệm khả vi (δ conceptual, dCFE) tích hợp vào khung NextGen của National Water Model Mỹ: đánh giá lưu vực không trạm, chọn mô hình theo chẩn đoán, vùng hóa theo thuộc tính lưu vực (bài dạng technical note). Tác giả: Frame, Araki, Bhuiyan, Bindas, …, Ogden
4. Tạp chí & ngành
JAWRA Journal of the American Water Resources Association 61(1):e70000 (Wiley), DOI 10.1111/1752-1688.70000 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Chưa xác nhận — người dùng tra Scimago 25/9/2026 không ra tên tạp chí
6. Dataset
🟢 Core — 495 lưu vực CAMELS
7. Code free
Có — github.com/NOAA-OWP/lstm + các repo NWC-CUAHSI-Summer-Institute (dCFE, cfe_py, attribute_pca, model-selection)

#81 — Deep learning-based approach for enhancing streamflow prediction in watersheds with aggregated and intermittent observations
Mục
Nội dung
1. Năm
2025
2. Cited by
36
3. Kiến trúc model
LSTM (khung hydroDL) học từ quan trắc dòng chảy gộp theo thời đoạn và bị gián đoạn, dùng cho tách chuỗi/lấp khoảng trống. Tác giả: Mangukiya, Sharma (IIT Roorkee)
4. Tạp chí & ngành
Water Resources Research 61(1):e2024WR037331, DOI 10.1029/2024WR037331 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core (một phần) — CAMELS-US (671 lưu vực) + 159 lưu vực CAMELS-IND
7. Code free
Có — dùng hydroDL: github.com/mhpi/hydroDL + Zenodo doi:10.5281/zenodo.3993880

#82 — A novel hybrid fine-tuning method for supercharging deep learning model development for hydrological prediction
Mục
Nội dung
1. Năm
2026
2. Cited by
4
3. Kiến trúc model
Phương pháp fine-tuning lai LSTM + Random Forest giúp phát triển mô hình DL dự báo dòng chảy nhanh và tốt hơn. Tác giả: Jahangir, Quilty, Shen, Scott, Steinschneider, Adamowski
4. Tạp chí & ngành
Environmental Modelling & Software 201:106978 (Elsevier), DOI 10.1016/j.envsoft.2026.106978 — Mô hình hóa môi trường, không phải CNTT/AI (bản preprint trước đó: EGUsphere, DOI 10.5194/egusphere-2025-846)
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 421 lưu vực CAMELS-US
7. Code free
Có — Zenodo doi:10.5281/zenodo.19423883

#83 — Evaluation of streamflow predictions from LSTM models in water- and energy-limited regions in the United States
Mục
Nội dung
1. Năm
2024
2. Cited by
25
3. Kiến trúc model
LSTM 1 lớp và 3 lớp, so sánh vùng giới hạn bởi nước (Great Basin) và bởi năng lượng (New England). Tác giả: Khand, Senay (USGS)
4. Tạp chí & ngành
Machine Learning with Applications 16:100551 (Elsevier), DOI 10.1016/j.mlwa.2024.100551 — ✅ tạp chí ngành ML
5. Xếp hạng Q
Q1 — SJR 1,225; ngành Computer Science – Artificial Intelligence (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟢 Core — 45 lưu vực CAMELS (27 New England + 18 Great Basin)
7. Code free
Không thấy — trang không có mục code

#84 — Streamflow prediction in ungauged catchments through use of catchment classification and deep learning
Mục
Nội dung
1. Năm
2024
2. Cited by
52
3. Kiến trúc model
Phân loại lưu vực (Random Forest) rồi huấn luyện LSTM theo nhóm để dự báo lưu vực không trạm. Tác giả: He, Jiang, Ren, Cui, Qin, Du, Zhu, …
4. Tạp chí & ngành
Journal of Hydrology 639:131638 (Elsevier), DOI 10.1016/j.jhydrol.2024.131638 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 671 lưu vực CAMELS
7. Code free
Không thấy — trang ScienceDirect không có link code

#85 — A comparative assessment of a hybrid approach against conventional and machine-learning daily streamflow prediction in ungauged basins
Mục
Nội dung
1. Năm
2025
2. Cited by
2
3. Kiến trúc model
So sánh LSTM vùng, mô hình lai dPL + HBV và HBV vùng hóa truyền thống cho lưu vực không trạm. Tác giả: Lee, Kim
4. Tạp chí & ngành
Journal of Hydrology: Regional Studies 62:102854 (Elsevier), DOI 10.1016/j.ejrh.2025.102854 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1 — SJR 1,415 (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟢 Core — 671 lưu vực CAMELS
7. Code free
Không thấy — trang ScienceDirect không có link code

#86 — Streamflow prediction in human-regulated catchments using multiscale deep learning modeling with anthropogenic similarities
Mục
Nội dung
1. Năm
2024
2. Cited by
28
3. Kiến trúc model
LSTM và mô hình khả vi DPL dùng bộ thuộc tính đa tỷ lệ có yếu tố con người (hồ chứa, tưới…). Tác giả: Tursun, Xie, Wang, Peng, Liu, Zheng, Wu, Nie
4. Tạp chí & ngành
Water Resources Research 60(9):e2023WR036853, DOI 10.1029/2023WR036853 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core (một phần) — 95 lưu vực CAMELS + 24 lưu vực sông Hoàng Hà (Trung Quốc)
7. Code free
Có — Zenodo zenodo.org/records/11112700

#87 — Combining global precipitation data and machine learning to predict flood peaks in ungauged areas with similar climate
Mục
Nội dung
1. Năm
2024
2. Cited by
14
3. Kiến trúc model
Mô hình ML huấn luyện trên lưu vực Mỹ (giàu dữ liệu) + mưa toàn cầu, dự báo đỉnh lũ cho vùng khí hậu tương tự (thử ở Úc, Brazil, Chile, Thụy Sĩ, Anh). Tác giả: Rasheed, Aravamudan, Zhang, …
4. Tạp chí & ngành
Advances in Water Resources 192:104781 (Elsevier), DOI 10.1016/j.advwatres.2024.104781 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core (một phần) — huấn luyện trên CAMELS-US, kiểm tra trên CAMELS-AUS/BR/CL/CH/GB (số lưu vực chưa xác nhận)
7. Code free
Có một phần — bài ghi đã nộp link code kèm bài, trang xem được không hiện link

#88 — Improving annual streamflow estimates using Budyko parameterization driven by catchment attributes
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Scholar chưa hiện)
3. Kiến trúc model
Tham số hóa phương trình Budyko bằng ML từ thuộc tính lưu vực, ước tính dòng chảy và bốc thoát hơi theo năm. Tác giả: Tamiru, Mekonnen, Kumar
4. Tạp chí & ngành
Journal of Hydrology X 30:100216 (Elsevier), DOI 10.1016/j.hydroa.2026.100216 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1 — SJR 1,042 (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟢 Core — 671 lưu vực CAMELS (dòng chảy năm, không phải ngày)
7. Code free
Không thấy — trang ScienceDirect không có link code

#89 — Exploring the performance and interpretability of hybrid hydrologic model coupling physical mechanisms and deep learning
Mục
Nội dung
1. Năm
2025
2. Cited by
47
3. Kiến trúc model
12 mô hình lai vùng hóa dựa trên dPL + HBV, so với LSTM vùng. Tác giả: He, Jiang, Ren, Cui, Du, Zhu, Qin, …
4. Tạp chí & ngành
Journal of Hydrology 649:132440 (Elsevier), DOI 10.1016/j.jhydrol.2024.132440 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 512/671 lưu vực CAMELS
7. Code free
Không thấy — trang ScienceDirect không có link code

#90 — Comparing three machine learning algorithms with existing methods for natural streamflow estimation
Mục
Nội dung
1. Năm
2024 (online 2023)
2. Cited by
10
3. Kiến trúc model
ANN, Random Forest, LightGBM ước lượng đồng thời tham số mô hình GR4J–CemaNeige để tái tạo dòng chảy tự nhiên; so với phương pháp tỷ lệ diện tích và chuyển theo khoảng cách. Tác giả: Mehrvand, Boucher, Kornelsen, Amani
4. Tạp chí & ngành
Hydrological Sciences Journal 69(1) (Taylor & Francis), DOI 10.1080/02626667.2023.2273402 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1 — SJR 0,797 (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟢 Core (một phần) — 675 lưu vực Mỹ + Québec (lưu vực Mỹ lấy từ CAMELS theo trích đoạn Scholar)
7. Code free
Chưa xác nhận — trang Taylor & Francis bị chặn (paywall)

#91 — Key drivers of physics-embedded machine learning performance for streamflow prediction
Mục
Nội dung
1. Năm
2026
2. Cited by
3
3. Kiến trúc model
Mô hình ML nhúng vật lý: mạng hồi quy nhận biết vật lý (PRNN từ EXP-HYDRO, HBV, FLEX, WBM) ghép 1D-CNN theo 2 cách nối. Tác giả: Adombi, Chesnaux
4. Tạp chí & ngành
Environmental Earth Sciences 85:197 (Springer), DOI 10.1007/s12665-026-12919-z — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1 — SJR 0,696 (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟢 Core — 500 lưu vực CAMELS-US
7. Code free
Không — "Codes used in this study will be provided by the authors upon reasonable request"

#92 — Information and disinformation in hydrological data across space: the case of streamflow predictions using machine learning
Mục
Nội dung
1. Năm
2024
2. Cited by
13
3. Kiến trúc model
LSTM đo giá trị thông tin của dữ liệu lưu vực cho (donor) khi dự báo lưu vực đích; thử 1–128 lưu vực cho. Tác giả: A. Gupta
4. Tạp chí & ngành
Journal of Hydrology: Regional Studies 51:101607 (Elsevier), DOI 10.1016/j.ejrh.2023.101607 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1 — SJR 1,415 (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟢 Core — 461 lưu vực CAMELS
7. Code free
Không thấy — trang ScienceDirect không có link code

#93 — Ensembling differentiable process-based and data-driven models with diverse meteorological forcing datasets to advance streamflow simulation
Mục
Nội dung
1. Năm
2025
2. Cited by
9
3. Kiến trúc model
Ghép tổ hợp (ensemble) LSTM và mô hình khả vi δHBV với nhiều bộ forcing khí tượng; thử theo thời gian và không gian. Tác giả: Li, Song, Pan, Lawson, … (nhóm Chaopeng Shen)
4. Tạp chí & ngành
HESS 29:6829, DOI 10.5194/hess-29-6829-2025 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS
7. Code free
Có — Zenodo doi:10.5281/zenodo.7943626 (δHBV) + doi:10.5281/zenodo.6326394 (LSTM)

#94 — Enhancing large-scale streamflow prediction by coupling structural inductive bias with meta-learning
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Scholar chưa hiện)
3. Kiến trúc model
Kết hợp inductive bias cấu trúc (EA-LSTM) với meta-learning MAML; so với LSTM, EA-LSTM, MAML-LSTM, SAC-SMA. Tác giả: Tang, Zhao, Yu, Yu, Qin, Li
4. Tạp chí & ngành
Journal of Hydrology 675:135577 (Elsevier), DOI 10.1016/j.jhydrol.2026.135577 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 468 lưu vực CAMELS
7. Code free
Không thấy — trang ScienceDirect không có link code

#95 — Modeling runoff with incomplete data: a comparison of hydrological, deep learning, and hybrid approaches
Mục
Nội dung
1. Năm
2026
2. Cited by
3
3. Kiến trúc model
So sánh mô hình thủy văn XAJ, LSTM và mô hình lai (META, Guided-LSTM) khi dữ liệu thiếu. Tác giả: Wu, Zorn, Zhao, Kløve, Liu, Guo, …
4. Tạp chí & ngành
Journal of Hydrology 669:135132 (Elsevier), DOI 10.1016/j.jhydrol.2026.135132 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core (một phần) — 40 lưu vực tổng cộng từ CAMELS-US + Nam Trung Quốc, Đức, Pháp
7. Code free
Không thấy — trang ScienceDirect không có link code

#96 — A novel strategy for automatic selection of cross-basin data to improve local machine learning-based runoff models
Mục
Nội dung
1. Năm
2024
2. Cited by
19
3. Kiến trúc model
LSTM với chiến lược tự động chọn dữ liệu lưu vực khác (basin affinity) để huấn luyện mô hình cho từng lưu vực. Tác giả: Nai, Liu, Tang, Liu, Sun, Gaffney
4. Tạp chí & ngành
Water Resources Research 60(5):e2023WR035051, DOI 10.1029/2023WR035051 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS
7. Code free
Có — Zenodo doi:10.5281/zenodo.8308207

#97 — Streamflow regime-based classification and hydrologic similarity analysis of catchment behavior using differentiable modeling with multiphysics outputs
Mục
Nội dung
1. Năm
2025
2. Cited by
5
3. Kiến trúc model
Mô hình khả vi (differentiable) có nhiều đầu ra vật lý, dùng để phân loại lưu vực theo chế độ dòng chảy và phân tích tương đồng thủy văn; so với LSTM. Tác giả: Hu, Li, Zhang, Xu, Chu, Shen, Li
4. Tạp chí & ngành
Journal of Hydrology 653:132766 (Elsevier), DOI 10.1016/j.jhydrol.2025.132766 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 531 lưu vực CAMELS
7. Code free
Không thấy — trang ScienceDirect không có link code

#98 — Improving streamflow simulation through machine learning-powered data integration and its potential for forecasting in the Western U.S.
Mục
Nội dung
1. Năm
2025
2. Cited by
9 (OpenAlex, 25/9/2026)
3. Kiến trúc model
LSTM và DI-LSTM (data integration: đưa lưu lượng quan trắc trễ 1 ngày vào input); KGE trung vị 0,80 → 0,96. Tác giả: Yang, Pan, Feng, Xiao, Dixon, Hartman, Shen, Song, Sengupta, Delle Monache, Ralph
4. Tạp chí & ngành
HESS 29(20):5453–5476, DOI 10.5194/hess-29-5453-2025 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟡 Không phải CAMELS — 646 lưu vực miền Tây Mỹ chọn từ GAGES-II (USGS, public domain); forcing CW3E 1 km (CC-BY-SA 4.0, reachhydro.org), lưu lượng USGS NWIS, SWE Đại học Arizona; train 1983–2002, test 2003–2022
7. Code free
Có một phần — chỉ trích thư viện hydroDL chung (Zenodo doi:10.5281/zenodo.5015120), không có repo riêng tái tạo bài

#99 — Daily runoff prediction based on lightweight Mamba with partial normalization (LightMamba)
Mục
Nội dung
1. Năm
2024
2. Cited by
11 (Crossref, 25/9/2026)
3. Kiến trúc model
LightMamba — biến thể Mamba gọn nhẹ + partial normalization + MPM; so với mô hình thống kê, ML, RNN, attention. Tác giả: Jia, Li, Huang, Chen
4. Tạp chí & ngành
Hydrology Research 55(12):1182–1196 (IWA), DOI 10.2166/nh.2024.063 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1 — SJR 0,812; ngành Water Science and Technology (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟡 Không phải CAMELS — 3 trạm sông Mississippi (dữ liệu USGS, free), 1983–2023 (~40 năm)
7. Code free
Không thấy — trang bài không nêu code

#100 — Mamba-enhanced multi-scale state space model for robust runoff prediction under data-scarce conditions across climatic zones (MSM)
Mục
Nội dung
1. Năm
2026
2. Cited by
1 (Crossref, 25/9/2026)
3. Kiến trúc model
MSM — khối tích chập đa tỷ lệ + attention, Residual-Enhanced Mamba, temporal attention; so với Transformer, TCN, GRU, LSTM, Mamba gốc. Tác giả: Wang, Zhang, Xu
4. Tạp chí & ngành
Journal of Hydrology 676:135670 (Elsevier), DOI 10.1016/j.jhydrol.2026.135670 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟡 Không phải CAMELS — 4 trạm: Hongshan, Yingluoxia, Tangnaihai (Trung Quốc) + Pawcatuck (Mỹ, USGS). Bài ghi "publicly available" nhưng 3 trạm Trung Quốc chưa xác nhận tải được; số năm chưa xác nhận
7. Code free
Không thấy — trang ScienceDirect không nêu code

#101 — Towards generalized hydrological forecasting using transformer models for 120 h streamflow prediction
Mục
Nội dung
1. Năm
2026
2. Cited by
1 (Crossref, 25/9/2026)
3. Kiến trúc model
Transformer dự báo 120 giờ, so với LSTM, GRU, Seq2Seq, Persistence. Tác giả: Demiray, Demir
4. Tạp chí & ngành
Machine Learning: Earth 2(2):025006 (IOP), DOI 10.1088/3049-4753/ae81b8 — ✅ tạp chí ML cho Khoa học Trái đất
5. Xếp hạng Q
Chưa có hạng — tạp chí mới, chưa có trên Scimago (người dùng tra 25/9/2026)
6. Dataset
🟡 Không phải CAMELS — WaterBench-Iowa (1 trong 4 bộ dữ liệu của dự án), 125 trạm Iowa, theo giờ, 10/2011–9/2018 (7 năm)
7. Code free
Có một phần — repo github.com/uihilab/WaterBench chỉ có dữ liệu + notebook LSTM/GRU/Seq2Seq/Ridge, KHÔNG có mô hình Transformer của bài này (kiểm repo qua GitHub API 25/9/2026)

#102 — Unified multi-task learning for hydrological processes using a shared transformer framework (HydroMTL)
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Crossref, 25/9/2026)
3. Kiến trúc model
Transformer encoder dùng chung + decoder riêng cho 2 nhiệm vụ: dự báo lưu lượng 24 giờ và tăng độ phân giải thời gian của mưa; so với Persistence, GRU, LSTM, Transformer. Tác giả: Demiray, Demir
4. Tạp chí & ngành
Scientific Reports 16:26768 (Nature Portfolio), DOI 10.1038/s41598-026-55130-7 — đa ngành
5. Xếp hạng Q
Q1 (theo #5, tùy category)
6. Dataset
🟡 Không phải CAMELS — WaterBench-Iowa, chỉ 1 trạm (USGS 06817000), 2011–2018; mưa IowaRain 2016–2019
7. Code free
Có — github.com/uihilab/HydroMTL-RainStream

[Đọc code 26/9/2026: Repo 6 file (MTL-train.py, hybrid_model.py…), 1 trạm Clarinda]

#103 — Accelerating flood warnings by 10 hours: the power of river network topology in AI-enhanced flood forecasting (FloodGNNs)
Mục
Nội dung
1. Năm
2025
2. Cited by
15 (Crossref, 25/9/2026)
3. Kiến trúc model
6 GNN (ChebNet, GAT, GraphSAGE, GCNII, GCN, GIN) trên đồ thị dày dựa theo reachability (giảm over-squashing) so với EA-LSTM; GNN dự báo 24 giờ ngang EA-LSTM 14 giờ. Tác giả: Wang, Chen, Zheng, Song
4. Tạp chí & ngành
npj Natural Hazards 2:45 (Nature Portfolio), DOI 10.1038/s44304-025-00083-6 — Thiên tai/Khoa học Trái đất
5. Xếp hạng Q
Chưa xác nhận — chưa tra được (25/9/2026)
6. Dataset
🟡 Không phải CAMELS — LamaH-CE (1 trong 4 bộ dữ liệu của dự án), 358 trạm sau lọc, theo giờ, 2000–2017 (18 năm)
7. Code free
Có — github.com/Dreamzz5/FloodGNNs

[Đọc toàn văn + mã 26/9/2026 (CHECKPDF Mục 5.1, CHECKCODE Mục 4): dự báo 24 h → 24 h; kết quả chỉ dạng hình; mã lỗi import `GCN_Point`, loss MSE (bài ghi MAE), NSE gộp mọi trạm không trọng số (bài ghi có trọng số); dữ liệu đã xử lý tải từ Google Drive, không có mã tiền xử lý/EA-LSTM]

#104 — The Merit of River Network Topology for Neural Flood Forecasting
Mục
Nội dung
1. Năm
2024
2. Cited by
21 (Google Scholar, người dùng tra 25/9/2026)
3. Kiến trúc model
GNN (GCN, GAT…) mô hình hóa lưu lượng trên mạng lưới trạm, so các cách định nghĩa ma trận kề; kết quả: đồ thị thô KHÔNG cải thiện. Tác giả: Kirschstein, Sun
4. Tạp chí & ngành
ICML 2024 (PMLR, tr. 24713–24725), arXiv 2405.19836 — ✅ Hội nghị ML — DOI 10.5555/3692070.3693060 (Crossref, 26/9/2026)
5. Xếp hạng Q
Hội nghị — CORE A* (ICML, chưa tra lại portal CORE)
6. Dataset
🟡 Không phải CAMELS — LamaH-CE (1 trong 4 bộ dữ liệu của dự án)
7. Code free
Có — github.com/nkirschi/neural-flood-forecasting

[Đọc code 26/9/2026: Đọc code 26/9/2026: dataset.py tự tải LamaH-CE từ Zenodo 5153305, dữ liệu theo GIỜ 2000–2017, chuẩn hóa bằng thống kê chỉ tính trên năm train, test 2016–2017, cửa sổ 24 h, lead 6 h, 3 fold, GCN/GAT/ResGAT + MLP baseline; repo 2,2 GB (kèm checkpoint), không chạy bằng biến môi trường — sửa DATASET_PATH trong script]

#105 — HydroGAT: Distributed Heterogeneous Graph Attention Transformer for Spatiotemporal Flood Prediction
Mục
Nội dung
1. Năm
2025
2. Cited by
8 (Google Scholar, người dùng tra 25/9/2026)
3. Kiến trúc model
HydroGAT — đồ thị dị thể mức điểm ảnh (đất + sông), GAT không gian + attention thời gian; NSE tới 0,97 dự báo lưu lượng theo giờ; huấn luyện phân tán tới 64 GPU A100. Tác giả: Sarkar, Hakimi, Chen, Huang, Lu, Demir, Jannesari
4. Tạp chí & ngành
ACM SIGSPATIAL 2025, DOI 10.1145/3748636.3764172 — ✅ Hội nghị CNTT (GIS)
5. Xếp hạng Q
Hội nghị — CORE A (ICORE2026, người dùng tra portal.core.edu.au 25/9/2026)
6. Dataset
🟡 Không phải CAMELS — 2 lưu vực vùng Trung Tây Mỹ (Cedar River…), dữ liệu kiểu WaterBench, chỉ tháng 5–9 các năm 2012–2018
7. Code free
Có — github.com/swapp-lab/HydroGAT (cần DGL + GPU CUDA)

[Đọc code 26/9/2026: Repo có preprocess + src (7 MB), 2 lưu vực Iowa mức điểm ảnh]

#106 — Multi-Source Adaptive-Fusion Transfer Learning for Streamflow Forecasting in Data-Scarce Catchments (MSAF-TL)
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Crossref, 25/9/2026)
3. Kiến trúc model
MSAF-TL — bộ trích đặc trưng động (LSTM) + mô-đun tương đồng tĩnh + hợp nhất thích ứng bằng cross-attention, chuyển tri thức từ nhiều lưu vực nguồn sang lưu vực thiếu dữ liệu. Tác giả: Gao, Liang, Borgomeo, Cho (Cambridge)
4. Tạp chí & ngành
Water Resources Research 62(6):e2025WR042495, DOI 10.1029/2025WR042495 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟡 Không phải CAMELS — CAMELS-GB (free, UKCEH), 669 lưu vực Anh, 1970–2015 (45 năm)
7. Code free
Có — github.com/YuxuanGaoYG/MSAF-TL (dựa trên neuralhydrology)

#107 — Federated learning based water streamflow forecasting via multi-sensor data fusion (FedMSF)
Mục
Nội dung
1. Năm
2026
2. Cited by
4 (Crossref, 25/9/2026)
3. Kiến trúc model
FedMSF — học liên kết (federated learning) hợp nhất nhiều trạm, không gom dữ liệu tập trung. Tác giả: Xia, Feng, Ge, Rodrigues, Gadekallu, Fang
4. Tạp chí & ngành
Information Fusion 129:104020 (Elsevier), DOI 10.1016/j.inffus.2025.104020 — ✅ tạp chí CNTT/AI
5. Xếp hạng Q
Q1 — SJR 4,197; ngành Computer Science (Information Systems, Software…) (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟡 Không phải CAMELS — CAMELS-GB (free), 671 lưu vực, 1970–2015
7. Code free
Không thấy — trang không có link code

#108 — Deep learning for cross-region streamflow and flood forecasting at a global scale (ED-DLSTM)
Mục
Nội dung
1. Năm
2024
2. Cited by
46 (Crossref, 25/9/2026)
3. Kiến trúc model
ED-DLSTM — encoder-decoder LSTM 2 lớp + mô-đun mã hóa thuộc tính không gian; NSE trung bình 0,75 trên >2.000 lưu vực, thử lưu vực không trạm ở Chile. Tác giả: Zhang, Ouyang, Cui, … (CAS)
4. Tạp chí & ngành
The Innovation 5(3):100617 (Cell Press), DOI 10.1016/j.xinn.2024.100617 — đa ngành
5. Xếp hạng Q
Chưa xác nhận — chưa tra được (25/9/2026)
6. Dataset
🟡 Không phải CAMELS — Hơn 2.000 lưu vực Mỹ, Canada, Trung Âu, Anh (tài liệu tham khảo dẫn CAMELS, HYSETS, LamaH-CE, CAMELS-GB, Caravan — đều free) + 160 lưu vực Chile
7. Code free
Chưa xác nhận — trang Cell không hiện mục code

#109 — Global prediction of extreme floods in ungauged watersheds
Mục
Nội dung
1. Năm
2024
2. Cited by
340 (Crossref, 25/9/2026)
3. Kiến trúc model
LSTM encoder–decoder (1 LSTM cho khí tượng quá khứ, 1 LSTM cho dự báo khí tượng) dự báo lũ tới 5 ngày cho lưu vực không trạm, ngang/tốt hơn GloFAS; mô hình đang chạy thật trong Google Flood Hub. Tác giả: Nearing, …, Kratzert, …, Matias (Google)
4. Tạp chí & ngành
Nature 627:559–563, DOI 10.1038/s41586-024-07145-1 — đa ngành
5. Xếp hạng Q
Q1
6. Dataset
🟡 Không phải CAMELS — 5.680 trạm GRDC, 1984–2021 (lưu lượng GRDC chỉ cấp theo yêu cầu, không phân phối lại; một phần có trong Caravan/GRDC-Caravan)
7. Code free
Có một phần — code phân tích github.com/google-research-datasets/global_streamflow_model_paper + mô hình/kết quả Zenodo doi:10.5281/zenodo.10397664 + thư viện neuralhydrology

[Đọc code 26/9/2026: Repo chỉ tái tạo hình/thống kê từ dự báo đã lưu (Zenodo 10397664) — KHÔNG có code mô hình để huấn luyện]

#110 — Investigating Deep Learning Knowledge Transfer in Streamflow Prediction From Global to Local Catchment
Mục
Nội dung
1. Năm
2026
2. Cited by
7 (Crossref, 25/9/2026)
3. Kiến trúc model
LSTM tiền huấn luyện trên vùng giàu dữ liệu rồi chuyển sang Trung Á; so LSTM cục bộ và LSTM vùng. Tác giả: Ougahi, Rowan
4. Tạp chí & ngành
Water Resources Research 62(2):e2025WR041194, DOI 10.1029/2025WR041194 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟡 Không phải CAMELS — 441 lưu vực nguồn (CAMELS-GB, CAMELS-CH, British Columbia; 1980–2015) + 36 lưu vực Trung Á (GRDC, 1950–1990); forcing ERA5-Land qua Caravan
7. Code free
Có một phần — chỉ dùng thư viện neuralhydrology, không có repo riêng

#111 — Transfer learning using the global Caravan dataset for developing a local river streamflow prediction model
Mục
Nội dung
1. Năm
2025
2. Cited by
3 (Crossref, 25/9/2026)
3. Kiến trúc model
LSTM tiền huấn luyện trên Caravan rồi tinh chỉnh cho sông cục bộ; so GR4J và LSTM chỉ học dữ liệu cục bộ. Tác giả: Alzhanov, Nugumanova, Moreido
4. Tạp chí & ngành
Environmental Modelling & Software 194:106691 (Elsevier), DOI 10.1016/j.envsoft.2025.106691 — Mô hình hóa môi trường, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟡 Không phải CAMELS — Caravan (free, ~6.830 lưu vực toàn cầu, ~40 năm)
7. Code free
Có một phần — chỉ dùng thư viện neuralhydrology, không có repo riêng

#112 — Deep learning for monthly rainfall–runoff modelling: a large-sample comparison with conceptual models across Australia
Mục
Nội dung
1. Năm
2024
2. Cited by
47 (Crossref, 25/9/2026)
3. Kiến trúc model
LSTM dự báo dòng chảy tháng so với mô hình khái niệm WAPABA; LSTM bằng hoặc hơn WAPABA ở 69% lưu vực. Tác giả: Clark, Lerat, Perraud, Fitch (CSIRO)
4. Tạp chí & ngành
HESS 28(5):1191–1213, DOI 10.5194/hess-28-1191-2024 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟡 Không phải CAMELS — 496 lưu vực Úc (dữ liệu BoM, free), 1950–2020 (70 năm), bước thời gian tháng
7. Code free
Có — csiro-hydroinformatics.github.io/monthly-lstm-runoff

#113 — Deep learning for the probabilistic prediction of semi-continuous hydrological variables – An application to streamflow prediction across CONUS
Mục
Nội dung
1. Năm
2026
2. Cited by
3 (Crossref, 25/9/2026)
3. Kiến trúc model
LSTM gắn lớp đầu ra xác suất Rectified Gaussian hoặc hurdle (xử lý lưu lượng bằng 0/gián đoạn), so với LSTM tất định. Tác giả: Quilty, Jahangir
4. Tạp chí & ngành
Journal of Hydrology 668:134986 (Elsevier), DOI 10.1016/j.jhydrol.2026.134986 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟢 Core — 421 lưu vực CAMELS-US (+ Caravan), 6/1984–1/2012
7. Code free
Có — Zenodo doi:10.5281/zenodo.18358063

[Đọc code 26/9/2026: Code MATLAB (không phải Python); dữ liệu .mat 421 lưu vực kèm Zenodo]

#114 — Using Deep Learning in Ensemble Streamflow Forecasting: Exploring the Predictive Value of Explicit Snowpack Information
Mục
Nội dung
1. Năm
2025
2. Cited by
6 (Crossref, 25/9/2026)
3. Kiến trúc model
LSTM kết hợp Ensemble Streamflow Prediction cho dự báo mùa, thử có/không thông tin tuyết (snowpack). Tác giả: Modi, Jennings, Kasprzyk, Small, Wobus, Livneh
4. Tạp chí & ngành
Journal of Advances in Modeling Earth Systems 17(3):e2024MS004582 (AGU), DOI 10.1029/2024MS004582 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1 — SJR 2,154; ngành Earth and Planetary Sciences / Global and Planetary Change (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
🟡 Không phải CAMELS — 664 lưu vực GAGES-II + 76 lưu vực miền Tây Mỹ; train WY1983–2000, test đến WY2020
7. Code free
Có — Zenodo zenodo.org/records/14213154

#115 — Using tide for rainfall runoff simulation with feature projection and reversible instance normalization (RR-TiDE)
Mục
Nội dung
1. Năm
2025
2. Cited by
2 (OpenAlex, 25/9/2026)
3. Kiến trúc model
RR-TiDE — Time-series Dense Encoder (thuần MLP) + lớp chiếu đặc trưng + RevIN; vượt Transformer và LSTM ở dự báo 7 ngày, NSE trung vị 0,82 ở 51 lưu vực thiếu dữ liệu. Tác giả: Fang, Qu, Yang, Li, Shi, Xu, Yu
4. Tạp chí & ngành
Scientific Reports 15 (Nature Portfolio), DOI 10.1038/s41598-025-91219-1 — đa ngành
5. Xếp hạng Q
Q1 (theo #5, tùy category)
6. Dataset
🟢 Core — CAMELS-US (số lưu vực tổng chưa xác nhận; 51 lưu vực cho thử nghiệm thiếu dữ liệu)
7. Code free
Có một phần — bài chỉ dẫn code TiDE gốc github.com/lich99/TiDE, không có repo riêng của RR-TiDE

[Đọc code 26/9/2026: Repo lich99/TiDE ghi 'Unofficial Implement' của TiDE tổng quát — không phải code của bài]

#116 — ResBi-Mamba Plus: Deep Bidirectional Mamba with Spatiotemporal Attention for Robust Interpretable Hourly Runoff Forecasting
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (OpenAlex, 25/9/2026)
3. Kiến trúc model
ResBi-Mamba Plus — Bi-Mamba dựa trên Mamba-2 (nhánh xuôi + nhánh ngược, chỉ quét trong cửa sổ lịch sử, không dùng dữ liệu tương lai) + attention không gian–thời gian STA (kiểu SENet, chiến lược CD/CI) + khung ResNet Plus 2 nhánh, sâu tới 6 khối; dự báo 2–48 giờ, loss MSE. So với TCN, Bi-LSTM, Transformer, Informer, iTransformer, Mamba-2, MambaFormer; ở 48 giờ sai số thấp hơn iTransformer 9,9%. Tác giả: Sheng, Wen, Feng, Shi
4. Tạp chí & ngành
Artificial Intelligence Science and Engineering 2(2):165–180 (IEEE Xplore, NXB Southwest University), DOI 10.23919/AISE.2026.000010 — ✅ tạp chí AI, Open Access
5. Xếp hạng Q
Chưa có hạng — tạp chí mới, người dùng tra Scimago không ra (25/9/2026)
6. Dataset
🟡 Không phải CAMELS — Columbia Basin Research (CBR, 1 trong 4 bộ dữ liệu của dự án), dữ liệu theo giờ 1/2016–12/2019 (4 năm, 35.040 mẫu), chia 75/25; đơn vị kcfs
7. Code free
Không thấy — bài không nêu link code (người dùng gửi toàn văn 25/9/2026)

#117 — RMC: advancing daily runoff forecasting with a unified cross-scale deep learning approach
Mục
Nội dung
1. Năm
2026 (online 2025)
2. Cited by
4 (ScienceDirect, 25/9/2026)
3. Kiến trúc model
RMC (Res-Mamba-Causal) — tích chập dư đa tỷ lệ (dilated) + lõi Mamba + causal self-attention trong cửa sổ trượt; dự báo 1 ngày; SHAP giải thích. Vượt LSTM, Transformer, CNN, Mamba (VD KGE 0,992 ở Fish River). Tác giả: Xu, Zeng, Wang, Zhang, Zang
4. Tạp chí & ngành
Journal of Hydrology 665:134722 (Elsevier, 2/2026), DOI 10.1016/j.jhydrol.2025.134722 — Thủy văn/Khoa học Trái đất, không phải CNTT/AI
5. Xếp hạng Q
Q1
6. Dataset
🟡 Chưa xác nhận là CAMELS — 4 trạm USGS (Fish River, Redstone, Johnson Creek, McKenzie River), theo ngày; 8 biến đầu vào (dayl, prcp, srad, swe, tmax, tmin, vp, runoff) trùng đúng bộ forcing Daymet của CAMELS nên rất có thể lấy từ CAMELS; số năm chưa xác nhận
7. Code free
Không thấy — bản xem trước ScienceDirect không có link code

#118 — Innovative daily runoff prediction model integrating black-winged kite algorithm and Mamba2–Transformer architecture (HFOVPM)
Mục
Nội dung
1. Năm
2026
2. Cited by
12 (Google Scholar, người dùng tra 25/9/2026)
3. Kiến trúc model
HFOVPM — lai Mamba2 + Transformer, tối ưu siêu tham số bằng thuật toán Black-Winged Kite. Tác giả: Xu, …, Zang
4. Tạp chí & ngành
Ecological Informatics 93:103565 (Elsevier), Open Access — Sinh thái/Tin học môi trường — DOI 10.1016/j.ecoinf.2025.103565 (Crossref, 26/9/2026)
5. Xếp hạng Q
Q1 — SJR 1,770; có ngành Computer Science Applications (Scimago SJR 2025, người dùng tra 25/9/2026)
6. Dataset
Chưa xác nhận — trang ScienceDirect bị Cloudflare chặn, chưa đọc được phần dữ liệu
7. Code free
Không (thực tế) — bài dẫn github.com/wangwen1621/HFOVPM nhưng repo chỉ có README.md, ghi "sẽ công bố code sau" (kiểm tra 25/9/2026)

[Tóm tắt qua OpenAlex 26/9/2026 (bài Open Access, 4 trích dẫn): VMD tối ưu tham số bằng thuật toán Black-winged Kite → Mamba2 → Transformer + hiệu chỉnh sai số; dữ liệu ngày của 3 trạm (1 lưu vực tuyết tan núi cao, 1 cận nhiệt đới mưa, 1 hỗn hợp) — tên trạm, nguồn dữ liệu và việc VMD có tách trước khi chia tập hay không CHƯA xác nhận (ScienceDirect trả 403)]

#119 — A hybrid statistical-dynamical forecast of seasonal streamflow for a catchment in the Upper Columbia River basin in Canada
Mục
Nội dung
1. Năm
2025
2. Cited by
0 (OpenAlex, 25/9/2026)
3. Kiến trúc model
LSTM dự báo tổng lượng dòng chảy mùa, huấn luyện với ERA5 rồi chạy với dự báo mùa SEAS5 (ECMWF); so bước thời gian tháng và ngày (tháng tốt hơn)
4. Tạp chí & ngành
Frontiers in Water (Frontiers Media), DOI 10.3389/frwa.2025.1595898 — Thủy văn, không phải CNTT/AI
5. Xếp hạng Q
Q1 — SJR 0,943 (theo #32)
6. Dataset
🟡 Không phải CAMELS — 1 lưu vực hồ chứa thượng lưu sông Columbia (Canada); ERA5 + SEAS5 (free, Copernicus CDS), lưu lượng Bonneville Power Administration "2020 Level Modified Streamflow 1928–2018" (công khai); không phải bộ CBR/DART của dự án
7. Code free
Không thấy — mục Data availability chỉ nêu nguồn dữ liệu, không có link code

#120 — A GNN routing module is all you need for LSTM Rainfall–Runoff models (Mosaffa, Pappenberger, Prudhomme, Chantry, Rüdiger, Cloke)
Mục
Nội dung
1. Năm
2026
2. Cited by
5 (Crossref)
3. Kiến trúc model
LSTM 2 lớp (128) sinh dòng chảy cho từng tiểu lưu vực + module GNN định tuyến trên đồ thị mạng sông có trọng số thời gian truyền; thử GCN, GAT, GraphSAGE, ChebNet. LSTM-GAT tốt nhất: NSE trung bình 0,61 so với LSTM 0,46; 78% trạm cải thiện, mạnh nhất ở hạ lưu. Dự báo 1 ngày, cửa sổ 180 ngày
4. Tạp chí & ngành
HESS 30:2079, DOI 10.5194/hess-30-2079-2026 — Thủy văn, không phải CNTT/AI
5. Xếp hạng Q
Q1 (HESS, theo các mục HESS khác)
6. Dataset
🟡 Không phải CAMELS — LamaH-CE, 530 tiểu lưu vực thượng Danube, theo ngày 1987–2017 (31 năm)
7. Code free
Có một phần — github.com/hmosaffa/GNN_flow_routing: chỉ 4 file mô hình (lstm_gat/gcn/graphsage/chebnet.py, ~800 dòng), KHÔNG có code tiền xử lý/huấn luyện/đánh giá; README ghi MIT nhưng không có file LICENSE; bài ghi dữ liệu "cung cấp khi yêu cầu" (LamaH-CE gốc vẫn free)

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

[Đọc code 26/9/2026: chỉ 4 file mô hình (~800 dòng), không có dữ liệu/huấn luyện/đánh giá. Đọc toàn văn: đầu vào chỉ gồm mưa, độ ẩm đất, nhiệt độ + 59 thuộc tính tĩnh, **không dùng Q quá khứ**; chia 70/15 ngẫu nhiên + 15% cuối làm test; tăng cường 2,5% đỉnh lũ ×4 trong tập train; GPU A100]

#121 — BiasCast: learning and adjusting real time biases from meteorological forecasts to enhance runoff predictions (Konold, Feigl, Podest, Klingler …)
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Crossref)
3. Kiến trúc model
LSTM (Sequential Forecast LSTM, Encoder–Decoder LSTM, học chuyển giao) học sai lệch của dự báo khí tượng ECMWF-HRES để dự báo lưu lượng cực đại ngày trước 24 giờ; NSE trung vị 0,63 → 0,71 khi thêm lưu lượng quan trắc
4. Tạp chí & ngành
HESS 30:5067, DOI 10.5194/hess-30-5067-2026 — Thủy văn, không phải CNTT/AI
5. Xếp hạng Q
Q1 (HESS)
6. Dataset
🟡 Không phải CAMELS — LamaH-CE mở rộng, 451 lưu vực, 2003–2017 (train 2003–2009, val 2010–2013, test 2014–2017) + dự báo ECMWF-HRES (chưa xác nhận free)
7. Code free
Có một phần — dùng thư viện NeuralHydrology, trang bài không thấy link repo riêng

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#122 — A heterogeneous multi-graph spatio-temporal network for runoff forecasting (Zhou, Yan, Zhang, Chang, Yang)
Mục
Nội dung
1. Năm
2026
2. Cited by
1 (Crossref)
3. Kiến trúc model
HMGSTN — đồ thị khí tượng + đồ thị thủy văn (hướng thượng–hạ lưu) + cổng hợp nhất chéo, ma trận kề động theo mẫu, TCN + cổng Fourier + chú ý thời gian, dự báo phần gia tăng, loss trọng số đỉnh lũ, MC Dropout có hiệu chỉnh; so với LSTM, T-Transformer, STGCN, GraphWaveNet, ST-Transformer (đọc toàn văn 26/9/2026, CHECKPDF Phụ lục A.10)
4. Tạp chí & ngành
Engineering Applications of Artificial Intelligence, DOI 10.1016/j.engappai.2026.114967 — ngành CNTT/AI
5. Xếp hạng Q
Chưa tra Scimago
6. Dataset
🟡 Không phải CAMELS — dữ liệu chính: 18 trạm sông Nhã Lung (Yalong, Trung Quốc) 2016–2021 bước 6 giờ, KHÔNG công khai, chia 70/30 không validation; LamaH-CE chỉ là thí nghiệm phụ (27 trạm, theo giờ, 1 hạt giống, không nêu kỳ chia)
7. Code free
Không — toàn văn ghi "Data will be made available on request", không có mã (đọc 26/9/2026)

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#123 — Long-term river flow forecasting: An integrated deep learning model with multi-scale feature extraction (Wang, Li, Liu, Pan, Li)
Mục
Nội dung
1. Năm
2025
2. Cited by
10 (OpenAlex) / 9 (Crossref)
3. Kiến trúc model
CTB = CNN 1D → TCN 3 tầng → BiGRU; đầu vào 72 giờ + mưa 120 giờ tới coi như biết trước, dự báo 1–120 giờ; so với GRU, LSTM, S2S, Informer, iTransformer (đọc toàn văn 26/9/2026, CHECKPDF Phụ lục A.11: NSE trung vị 0,90 ở 1 h → 0,38 ở 120 h, thua S2S ở 96–120 h; làm trơn chuỗi bằng trung bình trượt, chia 7:2:1 không rõ cách, baseline không tinh chỉnh)
4. Tạp chí & ngành
Expert Systems with Applications, DOI 10.1016/j.eswa.2025.127387 (7/2025) — ngành CNTT/AI
5. Xếp hạng Q
Q1 (ESWA, theo #36)
6. Dataset
🟡 Không phải CAMELS — WaterBench-Iowa (125 lưu vực Iowa, 10/2011–9/2018)
7. Code free
Không — toàn văn chỉ công bố liên kết dữ liệu WaterBench, không có mã (đọc 26/9/2026)

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#124 — Residual bidirectional gated recurrent unit with spatiotemporal shortcuts and dual-path attention for robust interpretable runoff forecasting (STResBiGRU — Sheng, Cao, Yang, Wen)
Mục
Nội dung
1. Năm
2026 (số 1/2027)
2. Cited by
0 (Crossref)
3. Kiến trúc model
STResBiGRU — BiGRU có kết nối tắt thời gian giữa các ô GRU (h_t = GRU(...) + h_{t-1}), kiến trúc ResNet Plus 2 đường khối dư + kết nối tắt dày, attention 2 đường DPA (SENet theo thời gian và theo đặc trưng), snapshot ensemble; dự báo Q theo giờ lead 2–24 h, loss MSE. Input: lưu lượng xả quá khứ + nhiệt độ nước + biến thời gian (năm/tháng/ngày/thứ/giờ) — không có mưa. So với LSTM, BiLSTM, GRU, BiGRU, TCN, Transformer, **MambaFormer**: MambaFormer tốt nhất ở lead 2–4 h (NSE 0,953/0,934) nhưng tụt ở lead 24 h (0,797, kém cả LSTM 0,842); STResBiGRU NSE 0,866 ở 24 h. Cùng nhóm tác giả #116
4. Tạp chí & ngành
Neural Networks, DOI 10.1016/j.neunet.2026.109416 — ngành CNTT/AI
5. Xếp hạng Q
Chưa tra Scimago
6. Dataset
🟡 Không phải CAMELS — **Columbia River DART (CBR, Univ. Washington)**, dữ liệu chất lượng nước theo giờ, 1/2016–12/2019 (35.040 giờ, 4 năm); chia 75/25 theo thời gian (train 2016–2018, test 2019), KHÔNG có tập validation
7. Code free
Không thấy — người dùng tra Google + mở toàn văn 26/9/2026: bài không có mục code/data availability, không có repo

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#125 — A novel hydrological signature-informed framework for enhancing streamflow prediction using multi-task learning (Wang, Li, Wei, Zhang …)
Mục
Nội dung
1. Năm
2026
2. Cited by
5 (OpenAlex)
3. Kiến trúc model
Mô hình học sâu (LSTM-based, chưa xác nhận) học đa nhiệm: dự báo Q + các chữ ký thủy văn (HS); NSE trung vị 0,739 so với 0,666; lợi nhất ở lưu vực dòng chảy nền, hạn, lũ; dự báo đến 30 ngày; giải thích bằng SHAP
4. Tạp chí & ngành
Water Resources Research, DOI 10.1029/2025WR041485 — Thủy văn, không phải CNTT/AI
5. Xếp hạng Q
Q1 (WRR)
6. Dataset
🟡 Caravan (6.830 lưu vực, gồm CAMELS-US và LamaH-CE)
7. Code free
Có — github.com/wangzili98/Hydrological-Signatures-Aware-Framework (Zenodo 10.5281/zenodo.15867114, CC BY 4.0, người dùng tìm 26/9/2026): LSTM seq2seq, Transformer, CNN-LSTM + SHAP + GradNorm; đọc Caravan CSV, đường dẫn cứng D:\data\Caravan; mốc thời gian chia tập không có trong config (chưa xác nhận)

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#126 — A Multitask Transfer Learning Framework for LSTM-Based Streamflow Forecasting (Alzhanov, Nugumanova, Demir)
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (OpenAlex)
3. Kiến trúc model
LSTM học chuyển giao + đa nhiệm (dự báo thêm độ dày nước tuyết SWE) cho lưu vực chịu ảnh hưởng tuyết; so 4 cấu hình, chuỗi 30–365 ngày
4. Tạp chí & ngành
IEEE Access, DOI 10.1109/ACCESS.2026.3705788 — đa ngành (Kỹ thuật/CS)
5. Xếp hạng Q
Chưa tra Scimago
6. Dataset
🟢 Core — pretrain CAMELS-US, đánh giá 30 lưu vực có tuyết giữ lại + 1 lưu vực Kazakhstan
7. Code free
Chưa xác nhận

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#127 — AI-Empowered Latent Four-dimensional Variational Data Assimilation for River Discharge Forecasting (Wang, Bertoli, Cheng, Schröter, Caporali, Piggott, Wang, Arcucci)
Mục
Nội dung
1. Năm
2025
2. Cited by
6 (OpenAlex) / 7 (Crossref)
3. Kiến trúc model
Latent 4D-Var: autoencoder tích chập (CAE) nén trường trạng thái + LSTM làm mô hình thay thế trong không gian ẩn, tối ưu hàm giá 4D-Var không cần mô hình tiếp tuyến/liên hợp
4. Tạp chí & ngành
IEEE JSTARS, DOI 10.1109/JSTARS.2025.3611136 — Viễn thám/Khoa học Trái đất
5. Xếp hạng Q
Chưa tra Scimago
6. Dataset
🟡 Không phải CAMELS — trường trạng thái từ EFAS, quan trắc từ LamaH-CE
7. Code free
Không thấy

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#128 — Aligning and Assimilating Multi-source Data for Flood Forecasting (Wang, Bertoli, Cheng, Schröter …)
Mục
Nội dung
1. Năm
2026
2. Cited by
0 (Crossref)
3. Kiến trúc model
Đồng hóa dữ liệu đa nguồn (khí tượng + thủy văn) cho dự báo lũ bằng học máy; cùng nhóm với bài JSTARS latent 4D-Var
4. Tạp chí & ngành
ICCS 2026, Lecture Notes in Computer Science, DOI 10.1007/978-3-032-29924-6_20 — hội nghị CNTT (khoa học tính toán)
5. Xếp hạng Q
Chưa tra CORE
6. Dataset
🟡 Không phải CAMELS — LamaH-CE (trạm) + dữ liệu khác, chưa xác nhận
7. Code free
Chưa xác nhận

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#129 — Latent Three-Dimensional Variational Data Assimilation with Convolutional Autoencoder and LSTM for Flood Forecasting (Wang, Bertoli, Schröter, Caporali …)
Mục
Nội dung
1. Năm
2025
2. Cited by
2 (Crossref)
3. Kiến trúc model
Latent 3D-Var: CAE + LSTM làm mô hình thay thế trong không gian ẩn
4. Tạp chí & ngành
ICCS 2025, Lecture Notes in Computer Science, DOI 10.1007/978-3-031-97567-7_4 — hội nghị CNTT (khoa học tính toán)
5. Xếp hạng Q
Chưa tra CORE
6. Dataset
🟡 Không phải CAMELS — LamaH-CE + EFAS (theo đoạn trích)
7. Code free
Chưa xác nhận

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#130 — Integrating Transformer Encoders with Graph Attention Networks for River Discharge Prediction (Laakissi, Ennayri, Laakissi …)
Mục
Nội dung
1. Năm
2025
2. Cited by
0 (Crossref)
3. Kiến trúc model
Transformer-GAT: Transformer encoder trích phụ thuộc thời gian dài, GAT mô hình hóa phụ thuộc không gian trên mạng sông; dự báo nhiều bước; NSE 0,8447
4. Tạp chí & ngành
ICCIT 2025 (International Conference on Cognitive and Intelligent Computing…, IEEE), DOI 10.1109/ICCIT68389.2025.11453422 — hội nghị CNTT
5. Xếp hạng Q
Chưa tra CORE (hội nghị nhỏ)
6. Dataset
🟡 Không phải CAMELS — LamaH-CE (theo đoạn trích), đánh giá trên các trận lũ
7. Code free
Không thấy

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#131 — Spatio-Temporal Physics-Informed Graph Transformer with Dynamic Flow Routing for Flood Forecasting (ST-PIGT — Liu, Hamza, Long, Zia)
Mục
Nội dung
1. Năm
2025
2. Cited by
0 (Crossref)
3. Kiến trúc model
ST-PIGT: graph attention động (trọng số cạnh RBF) + Transformer thời gian có attention dẫn bởi mưa + PINN đa nhiệm (liên tục, động lượng); NSE 0,9591 trên 100 trạm
4. Tạp chí & ngành
RICAI 2025 (7th Int. Conf. on Robotics, Intelligent Control and AI, IEEE), DOI 10.1109/RICAI68060.2025.11385300 — hội nghị CNTT
5. Xếp hạng Q
Chưa tra CORE (hội nghị nhỏ)
6. Dataset
🟡 Không phải CAMELS — tập con 100 trạm LamaH-CE
7. Code free
Không thấy

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#132 — CauSTream: Causal Spatio-Temporal Representation Learning for Streamflow Forecasting (Wan, Shah, Sabo, Liu …)
Mục
Nội dung
1. Năm
2025
2. Cited by
1 (Crossref)
3. Kiến trúc model
CauSTream: học đồng thời đồ thị nhân quả dòng chảy (giữa các biến khí tượng) và đồ thị định tuyến giữa các trạm; so ConvLSTM, STGCN, MTGNN, TCDF, CSF (không có Transformer/Mamba); dự báo 1/3/7 ngày
4. Tạp chí & ngành
IEEE BigData 2025, DOI 10.1109/BigData66926.2025.11402305 (arXiv 2512.16046) — hội nghị CNTT
5. Xếp hạng Q
Chưa tra CORE
6. Dataset
🟡 Không phải CAMELS — 3 lưu vực Mỹ (Brazos 73 trạm, Colorado 10, thượng Mississippi 8), USGS + Livneh, theo ngày; chỉ 10/1973–9/1977 (4 năm)
7. Code free
Không thấy trong bài

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#133 — Spatio-temporal Causal Learning for Streamflow Forecasting (CSF)
Mục
Nội dung
1. Năm
2024
2. Cited by
5 (OpenAlex)
3. Kiến trúc model
CSF: dùng đồ thị dòng sông làm tri thức tiên nghiệm để học cấu trúc nhân quả, rồi dự báo dòng chảy tại trạm đích; vượt STGNN thường
4. Tạp chí & ngành
IEEE BigData 2024, DOI 10.1109/BigData62323.2024.10826120 (arXiv 2411.17937) — hội nghị CNTT
5. Xếp hạng Q
Chưa tra CORE
6. Dataset
🟡 Không phải CAMELS — lưu vực Brazos (Texas), USGS; giai đoạn chưa xác nhận (bài sau #CauSTream dùng 1973–1977)
7. Code free
Không thấy

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#134 — Global Flood Projection and Socioeconomic Implications Under a Deep Learning Framework (Kang, Yin, Slater, Liu, Sun …)
Mục
Nội dung
1. Năm
2025
2. Cited by
14 (OpenAlex)
3. Kiến trúc model
Mô hình lai học sâu – thủy văn có ràng buộc vật lý, học dòng chảy ngày rồi chiếu tương lai với 20 GCM CMIP6 × 4 SSP để đánh giá rủi ro lũ và thiệt hại kinh tế – xã hội
4. Tạp chí & ngành
Water Resources Research, DOI 10.1029/2024WR037139 — Thủy văn
5. Xếp hạng Q
Q1 (WRR)
6. Dataset
🟡 Không phải CAMELS — lưu vực toàn cầu (có trích LamaH-CE), chi tiết chưa xác nhận
7. Code free
Chưa xác nhận

[Nguồn: người dùng tra Google Scholar 26/9/2026; kiểm bằng Crossref/OpenAlex/trang bài]

#135 — Zero-shot forecasting of streamflow using time series foundation models: are we there yet? (Alexander Y. Sun, Albert A. Sun)
Mục
Nội dung
1. Năm
2026
2. Cited by
Chưa tra
3. Kiến trúc model
So 4 mô hình nền tảng chuỗi thời gian (TSFM) zero-shot — Sundial, Chronos, TTM (TinyTimeMixer), MOIRAI — với LSTM chuyên biệt; dự báo trước 1 ngày (dữ liệu ngày) và 3 giờ (dữ liệu 3H). NSE trung vị ngày: LSTM 0,593 > Sundial 0,564 > Chronos 0,495 > TTM 0,437 > MOIRAI 0,258 → TSFM zero-shot chưa bằng LSTM
4. Tạp chí & ngành
Machine Learning: Earth 2(1), IOP, 20/3/2026, DOI 10.1088/3049-4753/ae4982 — tạp chí học máy (như #53, #101)
5. Xếp hạng Q
Chưa có hạng (tạp chí mới, như #53)
6. Dataset
🟢 Core — 531 lưu vực CAMELS-US; ngày 1980–2010, 3 giờ 2006–2020 (NLDAS mở rộng)
7. Code free
Có — github.com/dialuser/tsfm_study (không LICENSE, đẩy code 2/2026): LSTM + script chạy 4 TSFM; train 1980–1992, val 1992–1995, test 1995–2005, chuỗi 365 ngày dự báo 1 ngày; thống kê chuẩn hóa chỉ tính trên tập train; kèm kết quả 5 seed. Đường dẫn cứng /home/suna/...

[Nguồn: tra mạng 26/9/2026 (vòng cuối); thông tin bài từ trang IOPscience, code đã clone và đọc]

#136 — Novel Deep Learning Transformer Model for Short to Sub-Seasonal Streamflow Forecast (FutureTST — Ambika, Tayal, Mishra, Lu)
Mục
Nội dung
1. Năm
2025
2. Cited by
13 (OpenAlex)
3. Kiến trúc model
FutureTST (Future Time Series Transformer) — xử lý riêng khí tượng + lưu lượng quá khứ và đưa thêm thời tiết dự báo tương lai; NSE trung bình 0,82 → 0,67 cho dự báo 1 → 30 ngày; thêm lưu lượng thượng lưu cải thiện tới 10%; so với mô hình thủy văn vật lý (tin cậy tới ~4 ngày)
4. Tạp chí & ngành
Geophysical Research Letters 52(14), DOI 10.1029/2025GL116707 — Khoa học Trái đất (ORNL + IIT Gandhinagar)
5. Xếp hạng Q
Q1 (GRL)
6. Dataset
Chưa xác nhận — trang Wiley/OSTI không đọc được; có tác giả Vimal Mishra (IIT Gandhinagar) nên có thể là lưu vực Ấn Độ (dữ liệu CWC không free) — cần xem toàn văn
7. Code free
Có một phần (không chính thức) — không thấy repo của bài; có code FutureTST ở repo khác cùng nhóm/người liên quan: qxc101/StreamForcasting (models/futureTST.py, cùng tác giả repo HydroTFT #53), YingdaFan/futuretst-forecast và demiludan/FutureTST_TVA (demo dự báo giờ lưu vực Tennessee, 8/2026)

[Nguồn: tra mạng vòng 1 (Transformer), 26/9/2026]

#137 — Multi-source heterogeneous data-driven interpretable model based on transformer and Kolmogorov-Arnold networks (Ding, Wang, Tan, Zhang)
Mục
Nội dung
1. Năm
2026
2. Cited by
12 (OpenAlex)
3. Kiến trúc model
Transformer + KAN, dữ liệu đa nguồn, có diễn giải (chi tiết chưa đọc được)
4. Tạp chí & ngành
Expert Systems with Applications, DOI 10.1016/j.eswa.2026.132862 — ngành CNTT/AI
5. Xếp hạng Q
Q1 (ESWA, theo #36)
6. Dataset
Chưa xác nhận — ScienceDirect chặn, OpenAlex không có tóm tắt; nhóm tác giả (Zhaocai Wang) thường dùng trạm Trung Quốc (bài EMS 2024 của nhóm dùng 11 trạm sông Gia Lăng)
7. Code free
Chưa xác nhận

[Nguồn: tra mạng vòng 3 (KAN), 26/9/2026]
