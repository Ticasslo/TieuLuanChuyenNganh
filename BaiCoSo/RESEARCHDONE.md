# Tổng hợp các bài báo quan trọng (lọc từ 137 bài trong `RESEARCHING.md`)

> Ngày lập: 25–26/9/2026. Số `#N` là số thứ tự trong `RESEARCHING.md`. Tài liệu phục vụ hai yêu cầu của GVHD: **chọn bài cơ sở để cải tiến** và **viết phần khảo sát tài liệu**. Phân tích chi tiết các ứng viên bài cơ sở nằm ở `CHECKPDF.md` (toàn văn) và `CHECKCODE.md` (mã nguồn).

**Tiêu chí lọc.** Xuất bản từ 2024; dữ liệu công khai, dài nhiều năm; có huấn luyện mô hình học sâu cho lưu lượng Q; ưu tiên có mã nguồn công khai, kiến trúc SSM/Mamba và Transformer, venue Q1 hoặc CNTT/AI. Loại: MDPI, preprint/luận văn, bài tổng quan, bài dữ liệu, bài không huấn luyện mô hình, bài dự báo mực nước.

**Tổng quan kho 137 bài.**
- **Mã nguồn:** phần lớn các bài có mã riêng; một số chỉ có mã một phần (thư viện chung, repo chưa chính thức hoặc rỗng); số còn lại không có hoặc không tìm thấy (đã rà GitHub theo tên bài và tác giả).
- **Venue CNTT/AI:** 15 bài (Mục 4).
- **Mamba/SSM cho dòng chảy:** 10 bài (Mục 2).

---

## 0. Kết luận chọn bài cơ sở

**Xếp hạng** (chi tiết và bộ tiêu chí ở `CHECKPDF.md` Mục 2 và Mục 7):

| Hạng | # | Bài | Venue | Dữ liệu | Nhận xét chính |
|---|---|---|---|---|---|
| **1** | #104 | Kirschstein & Sun — *The Merit of River Network Topology for Neural Flood Forecasting* | ICML 2024 (CORE A*) | LamaH-CE theo giờ 2000–2017, 358 trạm | Dự báo 6 giờ; bộ mã hóa thời gian chỉ là một lớp affine → chỗ cải tiến bằng Mamba rõ nhất; có 957 checkpoint (162 cho thí nghiệm chính); mã tính NSE sai công thức (sửa được bằng checkpoint; ảnh hưởng lên kết luận so sánh cần tính lại) |
| **2** | #67 | Acuña Espinoza và cs. — MF-LSTM | HESS 2025 (Q1) | CAMELS-US theo giờ 1990–2018, 516 lưu vực | Chuỗi dài nhất (8.760 bước), phương pháp chuẩn nhất, thư viện Hy2DL có chế độ dự báo; nhẹ hơn #22 trên GPU miễn phí; bài gốc là mô phỏng |
| **3** | #22 | Wang và cs. — S4D-FT | WRR 2025 (Q1) | CAMELS-US ngày, 531 lưu vực | So sánh S4D-FT với Mamba và B2S6 có cơ sở lý thuyết; bài gốc là mô phỏng, phần chuyển sang dự báo đã có (HydroDiffusion); chi phí huấn luyện lớn nhất |
| 4 | #103 | Wang, Chen, Zheng, Song — FloodGNNs | npj Natural Hazards 2025 | LamaH-CE theo giờ | Hợp đề tài; mã lỗi import, không có checkpoint, kết quả chỉ dạng hình → baseline đồ thị dense cho #104 |
| 5 | #71 (+#54) | FHNN | ICDM 2025 (CORE A*) + WRR 2025 | CAMELS-US ngày | Dự báo 1–7 ngày; không tái lập được số liệu (thiếu tiền xử lý) |
| 6 | #53 | HydroTFT | Machine Learning: Earth 2026 | CAMELS-US ngày | Chọn epoch trên tập test |
| 7 | #42 | TFRN | IEEE TGRS 2025 | CAMELS + CAMELS-AUS | NSE gộp mọi lưu vực; cấu hình chọn trên tập test |
| 8 | #6 | Liu, Shen và cs. — From RNNs to Transformers | HESS 2025 (Q1) | CAMELS + toàn cầu | Bài benchmark; hợp làm khung chạy baseline |
| 9 | #36 | S4D/S5D trong dPL | ESWA 2026 (Q1, CNTT/AI) | CAMELS-US | Mô hình lai, mô phỏng; HBV làm mờ khác biệt kiến trúc |
| 10 | #135 | Sun & Sun — TSFM zero-shot | Machine Learning: Earth 2026 | CAMELS-US | Bài benchmark mô hình nền tảng |
| 11 | #38 | Ouyang và cs. — Raster CNN-LSTM | WRR 2026 (Q1) | CAMELS-US | Mô phỏng; trọng tâm là biểu diễn thuộc tính tĩnh |

**Bài cơ sở đã chốt (26/9/2026): #104** — bài gốc là dự báo, cải tiến tạo kiến trúc không gian–thời gian (Mamba mã hóa thời gian từng trạm + GNN theo mạng sông), câu hỏi nghiên cứu rõ (có bộ mã hóa thời gian tốt thì đồ thị có giúp không), có dữ liệu mạng sông để làm XAI theo không gian và demo bản đồ. Phạm vi tiểu luận đề xuất ở `CHECKPDF.md` Mục 8.

**Khung chạy baseline:** #6 (kèm dữ liệu CAMELS đã xử lý), Hy2DL (#67) hoặc NeuralHydrology. Lớp Mamba có sẵn trong NeuralHydrology đưa tensor sai thứ trục vào `mamba_ssm`, cần chuyển trục trước khi dùng (`CHECKCODE.md` Mục 13).

**Bài chưa xuất bản:** MTPre (Song, Chae, Chung — Mamba-Transformer lai, CAMELS-US 674, mã MIT trên Zenodo `19367140`) chưa xuất bản. Nếu được phép dùng bài chưa xuất bản thì chỉ dùng nhánh không EMD, vì nhánh EMD tách chuỗi trên toàn bộ dữ liệu trước khi cắt cửa sổ (rò rỉ tương lai).

---

## 1. Các bài đã đọc nhưng không phù hợp làm bài cơ sở

> Đã đọc mã nguồn của 35 bài. Không đọc: #8 (gói 3,5 GB, bài phân tích LSTM), #52 (Zenodo chỉ có dữ liệu), #58 (ước lượng bất định), #76 (6 lưu vực), #98 (dùng lại hydroDL), #114 (dự báo mùa).

**1.1. Dùng được nhưng có điểm yếu.**

| # | Bài | Điểm yếu tìm thấy trong mã |
|---|---|---|
| #120 | GNN routing (HESS 2026), LamaH-CE ngày 530 tiểu lưu vực | Repo chỉ có 4 tệp mô hình, không có mã dữ liệu/huấn luyện; không dùng Q quá khứ; chia tập ngẫu nhiên |
| #125 | HS-informed (WRR 2026), Caravan 6.830 lưu vực | Mốc chia tập không có trong cấu hình, đường dẫn cứng |
| #16 | MCR-LSTM (WRR 2025), CAMELS 531 | Biến thể LSTM của cùng nhóm #22 — hợp làm baseline |
| #39 | h-Diffusion (WRR), CAMELS giờ | Tập validation trùng tập test 2003–2008 |
| #10 | DRUM (GRL 2025), CAMELS 530 | Mô phỏng cùng ngày, không dự báo trước; 400 epoch |
| #106 | MSAF-TL (WRR 2026), CAMELS-GB | Ngoài 4 bộ dữ liệu dự án |
| #48 | HydroMoE | Không kèm dữ liệu; chọn trạm theo kết quả tập test |
| #35 | RF-Bench | ~1,1 năm/lưu vực, scaler fit trên test, cửa sổ cắt ngang ranh giới lưu vực |
| #75 | WRO-Water | Chỉ có dữ liệu giả |

**1.2. Không phù hợp.** #15 MC-LSTM và #57 FedHydroDSW (8 lưu vực) · #61 ETT (repo chỉ có README) · #63 TLSTM (1 tệp mã) · #109 Nearing (repo chỉ vẽ hình) · #115 RR-TiDE (repo TiDE tổng quát, không phải mã của bài) · #47 ZeroDiff (bài toán tái tạo chuỗi) · #102 HydroMTL (1 trạm) · #105 HydroGAT (2 lưu vực mức điểm ảnh, cần GPU mạnh) · #56 (bản vá NeuralHydrology) · #113 (mã MATLAB) · #46 (đồng hóa lưu lượng vận hành).

**1.3. Các bài venue CNTT/AI không có mã nguồn:** #122 EAAI 2026 (LamaH-CE), #123 ESWA 2025 (WaterBench-Iowa), #124 Neural Networks 2026 (Columbia DART theo giờ, 2016–2019). #4: repo hydroDL không chứa mã Transformer của bài.

**1.4. Kết quả rà soát bổ sung.**
- Rà lại toàn bộ 137 bài theo tiêu chí có mã nguồn + bài toán dự báo + dữ liệu miễn phí dài năm + venue uy tín (26/9/2026): không có bài nào vượt Kirschstein & Sun (#104). Các bài venue CNTT/AI còn lại hoặc không có mã (#107, #122–#124, #130–#133, #137), hoặc khác bài toán (#47 tái tạo chuỗi, #45 lưu vực không trạm, #127–#129 đồng hóa dữ liệu), hoặc cần tài nguyên vượt khả năng (#105).
- Rà GitHub cho 18 bài có đóng góp kiến trúc nhưng ghi "không có mã" (#1, #5, #9, #14, #21, #23, #24, #32, #37, #41, #69, #70, #72, #77, #79, #94, #108, #117): chỉ #37 HydEquivNet có repo nhưng repo rỗng.
- Tra mạng theo các hướng Mamba/SSM, xLSTM, Caravan, TSFM, GNN, dữ liệu giờ, LamaH-CE, WaterBench, Columbia, Transformer, TCN/KAN, venue CNTT/AI, và tra ngược GitHub: bổ sung #135, #136, #137; ngoài RiverMamba không có repo Mamba nào cho dòng chảy gắn với bài đã xuất bản.
- Loại khi tra: xLSTM 430 lưu vực Canada (SSRN), học bán giám sát CAMELS-DE (HESS Discussions), Benford Loss, DL vs National Water Model (1 trạm), Mamba_Runoff_demo (repo trống), Vischer và cs. HESS 2025 (chỉ có mã tiền xử lý), Izadi và cs. ESWA 2024, Wang và cs. EMS 2024, Liu và cs. ESWA 2025, BS-Former 2024, PatchTST-LSTM WRM 2026, TiDE/N-HiTS/PatchTST JAWRA 2026, KAN-LSTM HESS 2026, FlowNet (arXiv), Yu và cs. HESS 2024; các mô hình đồ thị nhân quả không gian–thời gian CSF (#133, IEEE BigData 2024: 73 trạm Brazos, dữ liệu ngày 1967–1977, không có mã), CauSTream (#132, IEEE BigData 2025: không có mã), STREAMS (mã công khai nhưng xuất bản 2023, repo không kèm dữ liệu), TC-GTN (workshop Climate Change AI tại NeurIPS 2025).

**1.5. Về venue.** Trong các ứng viên, #104 (ICML) và #71 (ICDM) thuộc venue CNTT/AI; #36 ở ESWA (CNTT/AI) nhưng xếp thấp vì là mô hình lai. Các bài còn lại ở tạp chí thủy văn Q1. Có thể lập luận: kiến trúc gốc thuộc venue AI (Mamba — COLM 2024; PatchTST — ICLR 2023; Informer — AAAI 2021), bài cơ sở là bài ứng dụng.

---

## 2. Mamba / SSM cho dòng chảy — bằng chứng cho phần khảo sát

| # | Bài | Venue | Dữ liệu | Code | Kết quả chính |
|---|---|---|---|---|---|
| #22 | S4D-FT | WRR 2025, Q1 | CAMELS-US 531 | Có | S4D-FT vượt LSTM trên nhiều vùng |
| #35 | RF-Bench (S-Mamba) | J. Hydrol. 2026, Q1 | CAMELS-US 516 theo giờ | Có | Mamba **không thắng** PatchTST; cân bằng độ chính xác–chi phí. Code: S-Mamba quét qua các **biến**, không quét theo thời gian; code chỉ dùng ~1,1 năm/lưu vực + scaler fit trên test |
| #36 | S4D/S5D cho dPL | ESWA 2026, Q1 (CNTT/AI) | CAMELS-US 531/671 | Có (chính thức) | NSE S4D 0,756 so với LSTM 0,742 |
| #37 | HydEquivNet (lý thuyết đường đơn vị + học đẳng biến theo tỷ lệ, có khối Mamba selective SSM theo đoạn trích Scholar) | J. Hydrol. 2026, Q1 | CAMELS-US 531 | Không (repo rỗng) | Chưa đọc được toàn văn |
| #124 | STResBiGRU (so với **MambaFormer**) | Neural Networks 2026 (CNTT/AI) | Columbia DART theo giờ 2016–2019 | Không | MambaFormer tốt nhất ở lead 2–4 h nhưng tụt mạnh ở 24 h (NSE 0,797 < LSTM 0,842) — bằng chứng Mamba lai không bền theo lead time |
| #99 | LightMamba | Hydrology Research 2024, Q1 | 3 trạm Mississippi (USGS), 1983–2023 | Không | Mamba gọn nhẹ + partial normalization vượt RNN/attention |
| #100 | MSM (Mamba đa tỷ lệ) | J. Hydrol. 2026, Q1 | 4 trạm (3 Trung Quốc + 1 Mỹ) | Không | Vượt Transformer, TCN, GRU, LSTM, Mamba gốc |
| #116 | ResBi-Mamba Plus | AI Sci. & Eng. 2026 (IEEE, chưa có hạng) | **Columbia Basin (CBR)** theo giờ 2016–2019 | Không | Bi-Mamba-2 + attention không gian–thời gian; 48 giờ sai số thấp hơn iTransformer 9,9%. Quét 2 chiều chỉ trong cửa sổ lịch sử → không rò rỉ tương lai |
| #117 | RMC (Res-Mamba-Causal) | J. Hydrol. 2026, Q1 | 4 trạm USGS, biến trùng forcing Daymet/CAMELS | Không | CNN dư + Mamba + causal attention vượt LSTM, Transformer, Mamba |
| #118 | HFOVPM (VMD + Mamba2 + Transformer) | Ecological Informatics 2026, Q1 | 3 trạm ngày (tuyết tan / mưa cận nhiệt / hỗn hợp), tên trạm chưa xác nhận | Không (repo chỉ có README) | Lai Mamba2–Transformer, tối ưu siêu tham số bằng metaheuristic |

**Nhận định:** bằng chứng Mamba cho dòng chảy **chưa thống nhất**. Các bài thắng (#99, #100, #117) dùng ít trạm, không công khai code. Benchmark lớn nhất (#35, 516 lưu vực) cho thấy Mamba chưa vượt PatchTST — nhưng mã công khai của #35 có lỗi phương pháp (Mục 1.1) nên kết luận này cần dè dặt → đóng khung đề tài là **nghiên cứu so sánh**, khớp nhận định đã chốt trong `CLAUDE.md`.

**Chưa xuất bản (theo dõi, không đưa vào kho):**
- **HydroDiffusion** (Wang, Yu, Zhang, Varadharajan, Erichson; arXiv 2512.12183, 12/2025; mã `github.com/yhwang08/HydroDiffusion`): mô hình khuếch tán dự báo xác suất 7 ngày, backbone S4D-FT, 531 lưu vực CAMELS, vượt DRUM; cùng nhóm tác giả #22, #15, #16.
- **Demiray & Demir** — Mamba + XAI trên WaterBench (EarthArXiv/SSRN 2025).
- **MTPre** (Song, Chae, Chung; mã Zenodo `19367140`, MIT): Mamba encoder + Transformer decoder, 674 lưu vực CAMELS-US, train 1980–1995 / val 1995–1999 / test 1999–2014, cửa sổ 15 ngày quá khứ + 7 ngày dự báo, so với ED-Transformer, Encoder-only, iTransformer, LSTM; chuyển giao sang CAMELS-GB. Nhánh có EMD tách chuỗi Q trên toàn bộ chuỗi trước khi cắt cửa sổ nên rò rỉ thông tin tương lai; nhánh không EMD sạch. README ghi "HESS 2026" nhưng chưa có tập/DOI.
- **Block-Biased Mamba** (Yu & Erichson, NeurIPS 2025; không phải bài dòng chảy): chỉ ra Mamba kém S4D ở tác vụ phụ thuộc xa, đề xuất B2S6 — liên quan trực tiếp tới lập luận chọn Mamba; thí nghiệm chỉ trên Long-Range Arena và mô hình ngôn ngữ, chưa kiểm chứng trên dữ liệu dòng chảy.

---

## 3. Bài trên 4 bộ dữ liệu của dự án

| Bộ dữ liệu | Bài quan trọng |
|---|---|
| **CAMELS-US** | #35 RF-Bench, #22 S4D-FT, #6 RNNs→Transformers, #36, #53, #75, #4 (Transformer, nhóm Shen), #67 MF-LSTM (theo giờ, code Hy2DL), #10 DRUM (diffusion, GRL) |
| **LamaH-CE** | #103 FloodGNNs (đồ thị reachability, có code), #104 Kirschstein & Sun — ICML 2024 (CORE A*, có code; đồ thị thô **không** cải thiện) |
| **WaterBench-Iowa** | #101 Demiray & Demir — Transformer 120 giờ (ML: Earth 2026; repo WaterBench chỉ có dữ liệu + baseline LSTM/GRU/Seq2Seq), #102 HydroMTL (Sci. Rep. 2026, 1 trạm, có code), #105 HydroGAT (SIGSPATIAL 2025, CORE A, có code, cần GPU mạnh) |
| **Columbia Basin (CBR/DART)** | #116 ResBi-Mamba Plus (không code). #119 dùng lưu lượng BPA thượng lưu Columbia, không phải CBR |

---

## 4. Bài ở venue CNTT/AI (đáp ứng yêu cầu khảo sát của GVHD)

| # | Bài | Venue | Hạng | Code |
|---|---|---|---|---|
| #104 | The Merit of River Network Topology for Neural Flood Forecasting | ICML 2024 | CORE A* | Có |
| #47 | ZeroDiff (tái tạo chuỗi zero-shot bằng diffusion, có train trên CAMELS) | ICML 2026 | CORE A* | Có |
| #45 | Transfer Learning Using Inaccurate Physics Rule for Streamflow Prediction | IJCAI 2024 | CORE A* | Không |
| #71 | Hierarchically disentangled recurrent network (HDRN) | IEEE ICDM 2025 | CORE A* | Một phần (repo FHNN, thiếu tiền xử lý) |
| #105 | HydroGAT | ACM SIGSPATIAL 2025 | CORE A | Có |
| #57 | FedHydroDSW (học liên kết cho lưu vực thiếu dữ liệu) | ICPR 2024 | CORE B | Có |
| #36 | S4D/S5D cho dPL | Expert Systems with Applications 2026 | Q1 | Có |
| #107 | FedMSF (học liên kết, CAMELS-GB) | Information Fusion 2026 | Q1 (SJR 4,197) | Không |
| #83 | LSTM vùng giới hạn nước/năng lượng | Machine Learning with Applications 2024 | Q1 | Không |
| #53 | HydroTFT | Machine Learning: Earth 2026 | chưa có hạng | Có |
| #135 | TSFM zero-shot vs LSTM (CAMELS) | Machine Learning: Earth 2026 | chưa có hạng | Có |
| #101 | Transformer 120 giờ trên WaterBench | Machine Learning: Earth 2026 | chưa có hạng | Một phần (repo WaterBench chỉ có dữ liệu + LSTM/GRU/Seq2Seq) |
| #63 | TLSTM (LSTM transductive + spatial attention) | Intelligent Decision Technologies 2025 | Q3 (ngành AI) | Có |
| #65 | Deep learning foundation and pattern models | Int. J. High Performance Computing Applications 2026 | Q2 (ngành CS) | Một phần |
| #116 | ResBi-Mamba Plus | Artificial Intelligence Science and Engineering 2026 (IEEE Xplore) | chưa có hạng | Không |

Tổng: 15 bài (6 hội nghị: #104, #47, #45, #71, #105, #57; 9 tạp chí: #36, #107, #83, #53, #101, #63, #65, #116, #135), 7 bài có code riêng. Giáp ranh, không tính: #118 (*Ecological Informatics* — Scimago có xếp thêm ngành Computer Science Applications nhưng là tạp chí sinh thái), #42 (IEEE TGRS — ngành viễn thám/kỹ thuật điện).

---

## 5. Nền tảng / baseline hay trích dẫn

| # | Bài | Vai trò |
|---|---|---|
| #109 | Nearing et al. — *Global prediction of extreme floods in ungauged watersheds* (Nature 2024, 340 trích dẫn) | Mô hình LSTM của Google Flood Hub — mốc "state of the art vận hành" |
| #2 | *HESS Opinions: Never train an LSTM on a single basin* (HESS 2024, 294 trích dẫn) | Lý luận vì sao train đa lưu vực (bài quan điểm, không kiến trúc mới) |
| #1 | Koya & Roy — TFT cho dòng chảy (J. Hydrol. 2024, 139 trích dẫn) | Kết hợp attention + recurrence; không có code |
| #4 | Liu et al. — *Probing the limit of hydrologic predictability with the Transformer* (J. Hydrol. 2024, 115 trích dẫn) | Transformer thuần không vượt LSTM trên CAMELS; code hydroDL |
| #5 | Rel-Informer (Sci. Rep. 2025) | Hướng Informer; gốc Informer — AAAI 2021 Best Paper |
| #98 | Yang et al. — DI-LSTM (HESS 2025) | Ý tưởng đưa lưu lượng trễ vào input (KGE 0,80 → 0,96), dùng được trong pipeline |
| #54 | FHNN — *Knowledge-guided machine learning for operational flood forecasting* (WRR 2025, Q1, CAMELS-US 531, code `github.com/arvindrenga96/FHNN`) | Mô hình hồi quy phân tầng (suy trạng thái lưu vực + dự báo) cho dự báo vận hành; nhóm Vipin Kumar (cùng nhóm #71 ICDM) |
| #82 | Jahangir et al. — *A novel hybrid fine-tuning method for supercharging deep learning model development* (EMS 2026, Q1, CAMELS-US 421, code Zenodo) | Kỹ thuật fine-tuning lai LSTM + Random Forest — mẹo huấn luyện, không phải kiến trúc mới |
| #115 | RR-TiDE (Sci. Rep. 2025) | Mô hình thuần MLP vượt Transformer/LSTM trên CAMELS — baseline đơn giản đáng thử |
