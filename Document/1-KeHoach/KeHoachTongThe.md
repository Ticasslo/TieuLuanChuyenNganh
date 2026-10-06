# Kế hoạch nghiên cứu: Dự báo lưu lượng dòng chảy bằng Mamba (State Space Model)

> Kế hoạch tổng thể cho tiểu luận chuyên ngành → khóa luận tốt nghiệp. Chi tiết pipeline, kiến trúc, lộ trình cải tiến và phạm vi: `Document/1-KeHoach/KienTrucPipeline.md`. Giới thiệu bài cơ sở: `Document/1-KeHoach/GioiThieuBaiCoSo.md`. Sơ đồ: `Diagrams/SoDoDT.drawio`. Các mục ghi "chưa chốt" chờ quyết định, chưa được điền.

---

## 1. Tên đề tài

| | Tên | Trạng thái |
|---|---|---|
| Tên đăng ký với GVHD | *"Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy"* | Giữ nguyên |
| Tên nội bộ tiểu luận | — | Chưa chốt |
| Tên nội bộ khóa luận | — | Chưa chốt |

Ràng buộc: tên không có chữ "lũ lụt" — đầu ra là lưu lượng Q, không phải vùng ngập hay mực nước. "Nguy cơ lũ" chỉ suy ra gián tiếp qua ngưỡng chu kỳ lặp lại trên Q.

---

## 2. Bài toán

- **Đầu ra:** lưu lượng lớn nhất ngày t (qmax) tại trạm cửa ra của từng lưu vực, dự báo trước 1 ngày. Q (m³/s) được bộ nạp NeuralHydrology đổi sang mm/ngày theo diện tích lưu vực: Q = q × diện tích (km²) / 86,4.
- **Đầu vào:** 364 ngày quá khứ (31 biến khí tượng tái phân tích và quan trắc lưới + Q quá khứ), 5 biến dự báo thời tiết ECMWF HRES cho ngày t, 33 thuộc tính tĩnh của lưu vực.
- **Q ≠ mực nước:** quy đổi cần đường quan hệ mực nước – lưu lượng riêng từng mặt cắt, ngoài phạm vi đề tài.
- **Cách học:** một mô hình chung cho 451 lưu vực, mỗi lưu vực dự báo độc lập.

---

## 3. Bài cơ sở và dữ liệu

**Bài cơ sở (người dùng chốt 6/10/2026):** Konold và cs., *BiasCast*, HESS 30:5067–5096 (2026). Mô hình tốt nhất (Sequential Forecast LSTM có Q quan trắc) đạt NSE trung vị 0,705 trên 451 lưu vực, test 2014–2017; persistence đạt 0,35–0,37 (tự tính). Lý do chọn: `Document/6-ChonBaiCoSo/CHECKPDF.md` Mục 8.4c, 8.6; `Document/1-KeHoach/GioiThieuBaiCoSo.md` Mục 8.

**Dữ liệu:** Extended LamaH-CE theo ngày (Zenodo 17119635, 0,95 GB, CC BY-NC 4.0): 859 lưu vực, 882 trạm, 1981–2017, 40 biến khí tượng từ ERA5-Land, E-OBS, MSWEP, GLEAM (từ 1981) và ECMWF HRES (từ 01/01/2002); bài cơ sở dùng 451 lưu vực. Kết quả kiểm tra và khảo sát: `Document/3-DuLieu/LamaHCE.md` Mục 6. Dữ liệu LamaH-CE gốc theo giờ và mạng sông đã có trên Kaggle cho các hướng khóa luận.

**Bộ dữ liệu khác** (CAMELS-US, Columbia Basin, WaterBench-Iowa) và công trình huấn luyện trên chúng: `Document/5-KhaoSat/Dataset.md`, dùng cho phần khảo sát và công trình liên quan.

---

## 4. RiverMamba — vai trò tham khảo

Shams Eddin, Zhang, Kollet, Gall — *RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting*, NeurIPS 2025 (Poster), DOI `10.52202/085713-4446`, arXiv 2505.22535, mã `github.com/HakamShams/RiverMamba_code`. Ghi chú đọc mã: `Document/7-LyThuyet/RiverMamba.md`; lý thuyết: `Document/7-LyThuyet/LyThuyetCauTruc.md` Mục 4.7–4.10.

Không dùng trực tiếp: checkpoint có lớp đầu vào cố định theo 136 biến lưới 0,05°, khác dữ liệu theo lưu vực. Ý tưởng tham khảo (đã xác nhận từ mã, cần kiểm chứng lại khi áp dụng):

| Thành phần | Mô tả |
|---|---|
| Biến đổi log1p có dấu | `sign(x)*log1p(|x|)` áp lên Δ lưu lượng trước MSE/L1 |
| Trọng số chu kỳ lặp lại trong hàm mất mát | Ưu tiên thời điểm vượt ngưỡng lũ hiếm |
| LOAN | `(X−μ)/σ + GELU(Linear(X_static))` — đưa thuộc tính tĩnh vào chuẩn hóa |
| Mamba hai chiều | Rò rỉ tương lai chỉ khi xuất dự báo tại từng bước của chuỗi; quét hai chiều trong cửa sổ quá khứ rồi dự báo sau cửa sổ thì không rò rỉ (ResBi-Mamba Plus, AISE 2026) — ứng viên biến thể Mamba ở bước E |
| Lùi đầu vào theo độ trễ công bố | Nguyên tắc chống rò rỉ tương lai khi dùng dữ liệu vận hành (kịch bản trễ dữ liệu ở bước đánh giá mở rộng) |

---

## 5. Bằng chứng về kiến trúc

- **Mamba/SSM cho dòng chảy chưa thống nhất:** S4D-FT, S4D/S5D vượt LSTM trên CAMELS-US; Mamba thuần (S-Mamba, quét theo biến) thua PatchTST và LSTM (Zhang và cs., J. Hydrology 2026); lớp Mamba có sẵn trong NeuralHydrology quét sai trục (`Document/6-ChonBaiCoSo/CHECKCODE.md` Mục 13); biến thể cải tiến (ResBi-Mamba Plus) cho kết quả tốt. → Đề tài là nghiên cứu so sánh, không mặc định Mamba thắng; mọi lợi ích của Mamba ghi là giả thuyết cần kiểm chứng.
- **Đồ thị mạng sông:** đồ thị thô không cải thiện (Kirschstein & Sun, ICML 2024); GAT, đồ thị reachability (Wang và cs., npj Natural Hazards 2025), đồ thị mức ô lưới (HydroGAT) cải thiện. → Chỉ dùng cách đã có bằng chứng; để cho khóa luận.
- **Đỉnh lũ là điểm yếu chung** của SSM và GNN → đánh giá riêng đỉnh lũ và theo mức lưu lượng (Mục 7).
- **G-Mamba** (Chen & Tang, Neurocomputing 680, 2026, DOI `10.1016/j.neucom.2026.133280`) là mô hình không gian – thời gian tổng quát, không thử trên dữ liệu dòng chảy; kết quả "cảnh báo sớm 10 giờ" thuộc Wang và cs. (npj Natural Hazards 2025, DOI `10.1038/s44304-025-00083-6`, mô hình GNN).
- **XAI:** Integrated Gradients/SHAP qua Captum; không dùng LLM viết báo cáo (thay bằng câu mẫu điền số). Demiray & Demir (preprint 2025, WaterBench-Iowa) đã làm Mamba + SHAP → cần đóng góp khác biệt.

---

## 6. Mô hình so sánh

| Mô hình | Vai trò |
|---|---|
| Persistence (qmax và qmean ngày t−1) | Mốc sàn (đã tính) |
| Mô hình của tác giả BiasCast (chạy lại trọng số) | Mốc tái lập |
| LSTM cải tiến (tầng 1) | Mốc mạnh cho tầng 2 |
| GRU, Transformer, S4D, Mamba trong cùng khung hindcast – forecast | So sánh lõi thời gian (tầng 2) |
| Biến thể Mamba (quét hai chiều) và Transformer (patch như PatchTST) | Kiểm chứng một ý cải tiến cho mỗi lõi |

Chi tiết và số lần huấn luyện: `Document/1-KeHoach/KienTrucPipeline.md` Mục 9.1.

---

## 7. Chỉ số đánh giá

- **Chính:** NSE, KGE theo từng lưu vực trên test 2014–2017; báo cáo trung vị, phân vị và đường CDF như bài gốc; số lưu vực tốt lên / kém đi.
- **Bổ sung:** FHV (đỉnh), FLV (dòng chảy kiệt), sai số thời điểm đỉnh.
- **Theo mức lưu lượng (yêu cầu GVHD):** chia qmax thành các mức theo ngưỡng riêng từng lưu vực, tính ma trận nhầm lẫn, tỉ lệ phát hiện và báo động nhầm ở mức lũ (`Document/1-KeHoach/KienTrucPipeline.md` Mục 7).
- **Độ tin cậy:** 3 hạt giống mỗi cấu hình, trung bình ± độ lệch chuẩn; chọn checkpoint theo validation, NSE test ghi mỗi epoch chỉ để theo dõi.

---

## 8. Hạ tầng

- **Tính toán:** Kaggle là chính (GPU 30 giờ/tuần, tối đa 12 giờ/phiên), Google Colab miễn phí dự phòng, Lightning AI khi cần. `mamba-ssm` cần GPU CUDA (kernel `selective_scan_cuda`, `causal_conv1d_cuda`); build wheel một lần rồi lưu Kaggle Dataset (`Document/7-LyThuyet/RiverMamba.md` Mục 0).
- **Dữ liệu:** Kaggle Dataset private `lamah-ce-ext` (dữ liệu chính và kết quả thí nghiệm của tác giả); `lamah-ce-core`, `lamah-ce-extra` (LamaH-CE gốc theo giờ, mạng sông) cho khóa luận.
- **Khung mã:** bản fork NeuralHydrology của tác giả BiasCast.
- **Demo (bắt buộc với tiểu luận):** công nghệ và nơi triển khai chưa chốt; VPS chỉ dùng nếu có thời gian.

---

## 9. Hạn chế và đạo đức

- Công cụ nghiên cứu, không thay cảnh báo chính thức; cần người kiểm tra.
- Mô hình học trên tái phân tích suy giảm khi chạy với dự báo thời tiết thật — BiasCast đo được NSE trung vị 0,58 → 0,33; đề tài dùng dự báo thật ngay từ đầu vào.
- Dữ liệu chỉ ở Trung Âu (Áo và vùng lân cận); khả năng áp dụng cho Việt Nam chưa được kiểm chứng.
- Extended LamaH-CE dùng giấy phép CC BY-NC 4.0 (không thương mại); bản 1.0 ghi chưa phải bản sửa cuối.

---

## 10. Hướng công bố (sau khóa luận)

SOICT (mời mở rộng sang *Multimedia Tools and Applications* / *Informatica*); chu kỳ 2027 trở đi.

---

## Tài liệu tham khảo chính

- **Bài cơ sở** — Konold, O., Feigl, M., Podest, P., Klingler, C., & Schulz, K. (2026). BiasCast: learning and adjusting real time biases from meteorological forecasts to enhance runoff predictions. *Hydrology and Earth System Sciences*, 30, 5067–5096. https://doi.org/10.5194/hess-30-5067-2026 (mã: `github.com/conestone/neuralhydrology`, `github.com/conestone/biascast`; dữ liệu: https://doi.org/10.5281/zenodo.17119635; trọng số: Zenodo 17292895).
- LamaH-CE — Klingler, C., Schulz, K., & Herrnegger, M. (2021). *Earth System Science Data*, 13, 4529–4565. https://doi.org/10.5194/essd-13-4529-2021.
- Kirschstein, N., & Sun, Y. (2024). The Merit of River Network Topology for Neural Flood Forecasting. *ICML 2024*, PMLR 235, 24713–24725 — tham khảo cho hướng đồ thị.
- RiverMamba — NeurIPS 2025, DOI `10.52202/085713-4446`, arXiv 2505.22535.
- Mamba — Gu & Dao, COLM 2024, arXiv 2312.00752.
- Yilmaz, K. K., Gupta, H. V., & Wagener, T. (2008). A process-based diagnostic approach to model evaluation. *Water Resources Research* — chia đoạn đường duy trì lưu lượng.
- Khảo sát tài liệu: `Document/5-KhaoSat/KhaoSatTaiLieu.md`; danh sách bài báo: `Document/6-ChonBaiCoSo/RESEARCHING.md`, `Document/6-ChonBaiCoSo/RESEARCHDONE.md`.
