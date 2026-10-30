---
paths:
  - "Document/**"
  - "PaperResearch/**"
---

# Sự thật về bài tham khảo và cách giữ tài liệu (Document/, PaperResearch/)

## Yêu cầu của GVHD khi chọn bài khảo sát

- Không MDPI, không chỉ preprint, trích dẫn ≥ 5, ưu tiên Scimago Q1/Q2; ưu tiên venue CNTT/AI là điểm cộng, không bắt buộc.
- Số trích dẫn tra bằng Google Scholar, Scopus: công cụ của Claude thường không tra được "Cited by" => nói ngay và nhờ người dùng tra.

## Thư mục PDF

- `PaperResearch/PaperResearchPDF/` chỉ giữ PDF cần cho đề tài, tên `AuthorYear_Venue_Topic.pdf`:
  - `BasePaper/`: BiasCast PDF, bản chữ `Konold2026_HESS_BiasCast_FullText.md`, `PeerReview/` (thư trả lời phản biện và biên tập).
  - `Evidence/`: căn cứ cho pipeline — Kratzert 2019 (NSE*), Kratzert 2024, Nearing 2022, Nearing 2024, Klotz 2022 (CMAL), Gauch 2025, Martel 2025, Baste 2025, Yang 2025, Yilmaz 2008, Klingler 2021 (bài dữ liệu LamaH-CE), Liu 2024 (giới hạn Transformer).
  - `MambaSSM/`: Mamba gốc, S4D gốc, RiverMamba, ResBi-Mamba Plus, S4D-FT, HydroDiffusion (bản arXiv), S4D/S5D, PatchTST, Zhang 2026 (so sánh LSTM/PatchTST/Mamba).
  - `Survey/`: bài của `02_LiteratureReview.md` (`MambaTimeSeries/` Mục 1.1, `StreamflowAsia/` Mục 2.1).
  - `Related/`: công trình trích dẫn BiasCast (AIFL, MF²LSTM — `01_BasePaper.md` Mục 12).
  - `Thesis/`: hướng khóa luận (Kirschstein & Sun, Mosaffa, Wang FloodGNNs — đồ thị trên LamaH; MF-LSTM — dữ liệu giờ).
- Bài chỉ nhắc qua trong ghi chú (Frame 2022, xLSTM, Zhou 2026, Sun 2026…) không giữ PDF. Thêm PDF mới => thêm dòng vào bảng tương ứng ở `05_Survey/03_PaperNotes.md`.
- Mã: `PaperResearch/PaperResearchCode/BiasCast` (mã tác giả), `BiasCast_NH` (fork NeuralHydrology của tác giả) — 2 submodule trỏ `conestone/biascast`, `conestone/neuralhydrology`, chỉ đọc.

## BiasCast

- Konold và cs., HESS 30:5067–5096, 2026; 1 hạt giống; 3 trích dẫn Google Scholar (khi chọn). Đọc fork + 24 cấu hình + kết quả: không có lỗi làm sai số liệu, trung vị tự tính khớp bài (`01_BasePaper.md` Mục 9, 11).
- Persistence NSE trung vị 0,35–0,37 so với mô hình tốt nhất 0,705; bài không có baseline persistence (phản biện 1 đề nghị).

## Kirschstein & Sun (ICML 2024)

- Không chọn làm bài cơ sở; tham khảo cho hướng đồ thị ở khóa luận. Lỗi mã đã xác nhận (`git show 247769a:Document/6-ChonBaiCoSo/CHECKCODE.md` Mục 1): NSE sai mẫu số; nhiễu giữa các lần huấn luyện tới 6,67 điểm % trong khi Bảng 2 chỉ trải 80,2–85,6%; ResGAT bỏ trọng số cạnh; relevancy tính trên cả batch.
- Đồ thị mạng sông thô không tự cải thiện (ICML 2024); chỉ các cách có bằng chứng (GAT, đồ thị reachability, đồ thị mức ô lưới) mới cải thiện (`01_Datasets.md` 7.2).

## OpenHydroNet (Google FloodHub)

- `github.com/google-research/flood-forecasting`, Nature 2024 + HESS 2025; đánh giá 26/9/2026: hạng 3, không thay bài cơ sở — không dùng Q quan trắc, không có mốc tái lập được, dự báo thời tiết thật trong dữ liệu công khai chỉ 2012–2022 (`git show 247769a:Document/6-ChonBaiCoSo/CHECKPDF.md` 5.11, 7.3, 8.4b; `CHECKCODE.md` Mục 14).

## RiverMamba (chỉ tham khảo, không chạy, không fine-tune)

- Shams Eddin, Zhang, Kollet, Gall — NeurIPS 2025 Poster, DOI `10.52202/085713-4446`, arXiv 2505.22535, mã `github.com/HakamShams/RiverMamba_code`. Ghi chú đọc mã: `06_Theory/02_RiverMamba.md`.
- Checkpoint không dùng được: lớp đầu vào cố định theo 136 biến lưới 0,05° (4 GloFAS + 32 ERA5-Land + 1 CPC + 99 tĩnh), khác hẳn dữ liệu theo lưu vực.
- Loss (từ mã): Δ lưu lượng biến đổi `sign(x)*log1p(|x|)` trước MSE/L1; trọng số theo chu kỳ lặp lại (ưu tiên lũ hiếm) × theo hạn `exp(|i-T-1|*alpha)`. Đầu ra là Δ so với hiện tại, 7 đầu MLP cho 7 hạn.
- LOAN: `(X−μ)/σ + GELU(Linear(X_static))` — hợp với thuộc tính tĩnh lưu vực.
- Mamba hai chiều trong RiverMamba quét chuỗi **không gian** (đường cong lấp không gian). Với chuỗi thời gian: rò rỉ tương lai chỉ khi xuất đầu ra tại từng bước (seq-to-seq) => phải causal; quét hai chiều **trong cửa sổ lịch sử** rồi dự báo sau cửa sổ thì không rò rỉ (ResBi-Mamba Plus, AISE 2026).
- Nguyên tắc lùi đầu vào theo độ trễ công bố thật của nguồn (GloFAS/ERA5 t-1, CPC t-2) để chống rò rỉ tương lai — áp dụng ở kịch bản S0–S4.
- Lưới 0,05° toàn cầu = 3000×7200 ≈ 21,6 triệu ô (**tính cả biển**); đất liền 6.221.926 điểm; tập AIFAS 1.529.667 điểm (lưu lượng trung vị > 10 m³/s).

## Đính chính đã kiểm

- Kết quả "dự báo 24 giờ ngang EA-LSTM 14 giờ nhờ đồ thị reachability" thuộc Wang et al., npj Natural Hazards 2025 (DOI `10.1038/s44304-025-00083-6`, mô hình **GNN, không Mamba**) — không phải G-Mamba. G-Mamba thật (Chen & Tang, Neurocomputing 680, 2026, DOI `10.1016/j.neucom.2026.133280`) là mô hình không gian–thời gian tổng quát (giao thông, điện, khí tượng, tỷ giá, ETT), **không có dữ liệu dòng chảy**.
- Bài SST là **CIKM 2025** (không phải 2024).

## Công trình gần nhất (đóng khung tính mới, `02_Pipeline.md` Mục 1.3)

- Shahriar (SSRN 2026; so cùng tham số lệch < 0,5%), Zhang 2026, Demiray & Demir (preprint 2025, WaterBench-Iowa, Mamba + SHAP), Dubey 2025, MF²LSTM, HydroDiffusion (WRR 2026, arXiv 2512.12183: S4D-FT so với LSTM với GEFSv12, không Q), Saint-Fleur (HESS 2026).
- Bằng chứng Mamba cho dòng chảy chưa thống nhất (`01_Datasets.md` 7.1, `02_LiteratureReview.md` Mục 1).
