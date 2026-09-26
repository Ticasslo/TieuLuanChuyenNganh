# Kế hoạch nghiên cứu: Dự báo lưu lượng dòng chảy bằng Mamba (State Space Model)

> Tài liệu kế hoạch cho tiểu luận chuyên ngành → khóa luận tốt nghiệp.

**Hướng đề tài (chốt 25/9/2026):** tự xây mô hình Mamba trên **4 bộ dữ liệu benchmark công khai** (`Dataset.md`). RiverMamba chỉ là **tài liệu tham khảo** về kiến trúc/loss.

**Các mục còn để trống/"chưa chốt"** là cố ý — đang chờ yêu cầu mới của người dùng, không tự điền.

---

## 1. Tên đề tài

| | Tên | Trạng thái |
|---|---|---|
| Tên đăng ký với GVHD | *"Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy"* | ✅ Giữ nguyên — tên rộng, vẫn khớp hướng mới (không nêu lưu vực/kiến trúc cụ thể) |
| Tên nội bộ tiểu luận | — | ⏳ Chưa chốt |
| Tên nội bộ khóa luận | — | ⏳ Chưa chốt |

**Ràng buộc tên (chốt 19/8/2026):** không có chữ "lũ lụt" — output là lưu lượng Q (m³/s), không phải vùng ngập/mực nước. "Nguy cơ lũ" chỉ suy ra gián tiếp qua ngưỡng return period trên Q.

---

## 2. Bài toán

- **Đầu ra:** lưu lượng dòng chảy Q (m³/s, hoặc mm/ngày khi chuẩn hóa theo diện tích lưu vực) tại trạm đo.
- **Q ≠ mực nước:** quy đổi cần rating curve riêng từng mặt cắt — ngoài phạm vi đề tài.
- **Dạng dữ liệu:** cả 4 bộ dữ liệu là **chuỗi thời gian theo từng lưu vực/trạm** (forcing khí tượng đã lấy trung bình theo lưu vực + thuộc tính tĩnh + lưu lượng quan trắc), không phải lưới không gian như RiverMamba. Hệ quả: Mamba xử lý **chuỗi thời gian dài** tại mỗi lưu vực (VD cửa sổ 365 ngày như HydroDiffusion, `Dataset.md` 6.2.2) — đúng chỗ lợi thế độ phức tạp tuyến tính của Mamba phát huy (RiverMamba gốc chỉ nhìn lại 4 ngày).
- **Ground truth:** lưu lượng **quan trắc thật** tại trạm (USGS, cơ quan thủy văn Trung Âu...).

---

## 3. Bốn bộ dữ liệu

Chi tiết đầy đủ (quy mô, biến, topology, công trình đã huấn luyện, số liệu mốc): **`Dataset.md`**. Tóm tắt:

| Bộ dữ liệu | Khu vực | Quy mô | Độ phân giải | Topology sông | Điểm mạnh chính |
|---|---|---|---|---|---|
| LamaH-CE | Trung Âu | 859 lưu vực, 1981–2017 | Ngày + giờ | ✅ Có sẵn | Topology + nhiều biến (15 forcing, >60 thuộc tính) |
| CAMELS-US | Mỹ | 671 lưu vực, 1980–2014 | Ngày (giờ: bản mở rộng 516 lưu vực) | ❌ | Nhiều mốc SSM/Mamba nhất để đối chiếu |
| Columbia Basin (CBR/DART) | Mỹ | 35+ điểm đo | Ngày (chất lượng nước: giờ) | Không có tệp | Dữ liệu vận hành hồ chứa thật |
| WaterBench-Iowa | Mỹ | 125 lưu vực, 2011–2018 | Giờ | ✅ Dựng được (HydroGAT) | Đủ mốc Mamba/Transformer/Graph/XAI, có code |

**Vai trò từng bộ (dataset chính/phụ, dùng để làm gì):** ⏳ chưa chốt. Các phương án đã phân tích ở `Dataset.md` Mục 9.2 (A/B/C) được viết khi xét từng bộ riêng lẻ — cần xét lại khi chốt dùng cả 4.

**Rủi ro dữ liệu đã biết:**
- CAMELS-US ~15 GB nén / **~130 GB giải nén** — có thể vượt dung lượng đĩa Colab nếu giải nén hết (dung lượng đĩa hiện tại của gói Colab Pro chưa kiểm tra); cần tải/giải nén từng phần (chưa xác nhận cách làm).
- Columbia Basin không đóng gói sẵn, không có forcing khí tượng, không có thuộc tính lưu vực — phải tự dựng pipeline truy vấn DART + tự ghép forcing từ nguồn khác.
- HydroGAT (WaterBench) không công bố dữ liệu đã xử lý — muốn tái lập phải tự dựng lại từ Stage IV/DEM/USGS.

---

## 4. RiverMamba — vai trò tham khảo

Paper: Shams Eddin, Zhang, Kollet, Gall — *"RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting"*, **NeurIPS 2025 (Poster)**, DOI proceedings `10.52202/085713-4446`, arXiv 2505.22535, code `github.com/HakamShams/RiverMamba_code`. Ghi chú đọc code: `RiverMamba.md`. Lý thuyết cơ chế: `LyThuyetCauTruc.md` Mục 4.7–4.10.

**Không dùng được trực tiếp cho hướng mới:** checkpoint pretrained (lớp input cố định theo 136 biến lưới 0,05°, không khớp dữ liệu theo lưu vực), pipeline dữ liệu, serialization space-filling curve (dành cho lưới không gian — với dữ liệu theo lưu vực, quan hệ không gian nên biểu diễn bằng đồ thị mạng lưới sông, xem Mục 5).

**Ý tưởng có thể mang sang (cần kiểm chứng lại khi áp dụng, không mặc định hiệu quả trên dữ liệu mới):**

| Thành phần | Mô tả (đã xác nhận từ code repo) |
|---|---|
| Biến đổi log1p có dấu | `sign(x)*log1p(|x|)` áp lên Δ lưu lượng trước khi tính MSE/L1 |
| Trọng số return period trong loss | Ưu tiên thời điểm vượt ngưỡng lũ hiếm → cải thiện F1 (Bảng 2a paper) |
| Trọng số lead time | `exp(|i-T-1|*alpha)` — ưu tiên ngày dự báo gần |
| Dự báo Δ thay vì giá trị tuyệt đối | Output là thay đổi so với hiện tại, 7 đầu MLP riêng cho 7 lead time |
| LOAN | `LOAN(X) = (X−μ)/σ + GELU(Linear(X_static))` — tiêm thuộc tính tĩnh lưu vực vào chuẩn hóa. Khớp tự nhiên với thuộc tính lưu vực của LamaH-CE/CAMELS |
| Bidirectional Mamba | Rò rỉ tương lai chỉ xảy ra khi mô hình xuất dự báo **tại từng bước** của chuỗi (kiểu mô phỏng seq-to-seq, output bước t mà nhánh ngược đã thấy bước > t) → trường hợp này phải dùng Mamba causal. Nếu chỉ quét 2 chiều **trong cửa sổ lịch sử** rồi dự báo các bước **sau** cửa sổ thì không rò rỉ — ResBi-Mamba Plus (AISE 2026, dữ liệu CBR theo giờ, RESEARCHING #116) làm đúng cách này. (Sửa 25/9/2026: bản cũ ghi "chuỗi thời gian thuần phải dùng causal" là quá tuyệt đối.) |
| Độ trễ dữ liệu khi ghép input | Lùi input theo độ trễ công bố thật của nguồn — nguyên tắc chống rò rỉ tương lai, áp dụng lại được |

Số liệu tham chiếu của RiverMamba (R² 0,873 vs GloFAS; 0,506 vs GRDC) **không so trực tiếp được** với kết quả trên 4 bộ dữ liệu mới (khác dữ liệu, khác chỉ số chính).

---

## 5. Kiến trúc — hướng đi và bằng chứng

⏳ **Kiến trúc cụ thể cho tiểu luận/khóa luận chưa chốt.** **Bài cơ sở đã chốt (26/9/2026): #104** — Kirschstein & Sun, *The Merit of River Network Topology for Neural Flood Forecasting*, ICML 2024, LamaH-CE theo giờ (lý do và phạm vi tiểu luận đề xuất: `CHECKPDF.md` Mục 7–8; xếp hạng các ứng viên: `RESEARCHDONE.md` Mục 0). Bằng chứng literature đã tổng hợp (`Dataset.md` Mục 7):

- **Mamba/SSM cho dòng chảy: kết quả chưa thống nhất** — S4D-FT/S4D/S5D vượt LSTM trên CAMELS-US, nhưng Mamba thuần thua PatchTST và LSTM (Zhang et al., J. Hydrology 2026 — đó là S-Mamba quét theo biến, không theo thời gian; thêm nữa lớp Mamba có sẵn trong NeuralHydrology quét sai trục, `CHECKCODE.md` Mục 13); biến thể có cải tiến (ResBi-Mamba Plus) cho kết quả tốt. → Nên đóng khung đề tài là **nghiên cứu so sánh**, không mặc định Mamba thắng.
- **Đồ thị mạng lưới sông: kết luận trái chiều** — đồ thị thô không cải thiện (ICML 2024, LamaH-CE); GAT cải thiện (HESS 2026, LamaH-CE); đồ thị làm dày theo reachability cải thiện rõ (Wang et al., npj Natural Hazards 2025); đồ thị mức ô lưới cải thiện (HydroGAT). → Nếu dùng đồ thị, phải dùng cách đã có bằng chứng, không dùng đồ thị thô.
- **Điểm yếu chung tại đỉnh lũ:** cả SSM (S4D-FT) lẫn GNN (ICML 2024) đều yếu ở dòng chảy biến động nhanh/đỉnh hẹp → đánh giá riêng đỉnh lũ (FHV), dùng loss có trọng số return period.

**Về G-Mamba và kết quả "sớm 10 giờ" (kiểm tra trang gốc 25/9/2026):**
- **G-Mamba** (Xiaojian Chen, Qiusheng Tang — *"Graph-enhanced Mamba: Efficient spatiotemporal sequence modeling with selective state space and graph neural networks"*, **Neurocomputing** 680, 2026, DOI `10.1016/j.neucom.2026.133280`) là mô hình dự báo chuỗi không gian–thời gian **tổng quát** (thử trên giao thông, điện, khí tượng, tỷ giá, nhiệt độ máy biến áp — không có dữ liệu dòng chảy); cơ chế tiêm ngữ cảnh láng giềng từ đồ thị vào bước cập nhật trạng thái của selective SSM.
- Kết quả **dự báo 24 giờ ngang EA-LSTM 14 giờ** thuộc về Wang, Chen, Zheng, Song — *"Accelerating flood warnings by 10 hours…"*, **npj Natural Hazards** 2025, DOI `10.1038/s44304-025-00083-6` (#103) — mô hình GNN, không dùng Mamba.
- Do đó chưa có công trình kết hợp GNN với Mamba **cho dự báo lưu lượng** (`Dataset.md` 7.4); G-Mamba là tham khảo kiến trúc cho hướng Graph + Mamba.

**XAI (quyết định 23/8/2026):** nếu làm XAI thì dùng feature attribution (Integrated Gradients/SHAP qua Captum), **không** fine-tune LLM viết báo cáo — thay bằng template câu điền số liệu. Lưu ý: Demiray & Demir (preprint 2025, WaterBench-Iowa) đã làm Mamba + SHAP → nếu làm XAI cần đóng góp khác biệt so với công trình này.

---

## 6. Baseline

⏳ Danh sách chính thức chưa chốt. Gợi ý rút ra từ literature (`Dataset.md` 7.5 + RiverMamba):

| Baseline | Lý do |
|---|---|
| Persistence, Climatology | Mốc sàn chuẩn |
| LSTM (và/hoặc EA-LSTM) | Chuẩn ngành trong mô hình mưa–dòng chảy học sâu (thư viện neuralhydrology có sẵn) |
| GRU | Baseline song song LSTM, rẻ hơn |
| Transformer / **PatchTST** | PatchTST thắng Mamba thuần trên CAMELS-US theo giờ (Zhang et al. 2026) |
| Random Forest / XGBoost | Ở quy mô ít lưu vực, ML cổ điển có thể vượt học sâu (Shanko & Melesse 2026) |
| Mô hình đồ thị (GAT...) | Nếu hướng có dùng đồ thị |

---

## 7. Chỉ số đánh giá

**Regression:** NSE (>0,5 đạt), KGE, RSR (<0,7 đạt), PBIAS, cộng thêm **FHV** (sai lệch đỉnh lũ) và **FLV** (sai lệch dòng chảy kiệt) theo Sun & Sun (2026).

**Classification:** F1-score theo ngưỡng return period (RiverMamba báo cáo 1,5/2/5/10/20 năm) — ngưỡng tự fit Gumbel/L-moments trên lưu lượng lớn nhất năm của từng trạm.

**Lưu ý đa lưu vực (cả 4 bộ đều nhiều lưu vực):** NSE/KGE nhạy với đặc tính dòng chảy từng lưu vực → báo cáo **trung vị/phân phối theo lưu vực**, bổ sung NRMSE/PBIAS khi so giữa vùng. Huấn luyện nhiều seed, báo cáo phân phối kết quả (Koya & Roy).

---

## 8. Hạn chế & đạo đức

- Công cụ nghiên cứu, không thay cảnh báo chính thức; cần human-in-the-loop.
- Mô hình huấn luyện trên dữ liệu tái phân tích/quan trắc lịch sử sẽ **suy giảm khi chạy với dự báo khí tượng thực** — BiasCast định lượng NSE trung vị 0,58 → 0,33 trên LamaH-CE; cần nêu rõ.
- CAMELS-US và bản C của LamaH-CE chọn lưu vực **ít chịu tác động con người** → kết quả chưa đại diện cho lưu vực có hồ chứa điều tiết.
- Cả 4 bộ đều ở Mỹ/châu Âu — khả năng áp dụng cho Việt Nam chưa được kiểm chứng trong đề tài.

---

## 9. Hạ tầng

- **Compute train:** Google Colab Pro là chính, Kaggle dự phòng (GPU 30 giờ/tuần, tối đa 12 giờ/phiên), Lightning AI khi cần. Thư viện `mamba-ssm` cần GPU CUDA (kernel `selective_scan_cuda`, `causal_conv1d_cuda`); build wheel 1 lần rồi lưu Drive/Kaggle Dataset (`RiverMamba.md` Mục 0).
- **Demo (yêu cầu bắt buộc của TLCN — `02_literature_review.md` Mục 3):** ⏳ chưa chốt; đề xuất hiện tại ở `CHECKPDF.md` Mục 8.2. Lưu ý: 4 bộ dữ liệu là dữ liệu lịch sử, không có luồng cập nhật hằng ngày. VPS Oracle Cloud Always Free (2 OCPU/12 GB từ 15/6/2026) vẫn dùng được để host demo.

---

## 10. Hướng công bố (sau khóa luận)

SOICT (mời mở rộng sang *Multimedia Tools and Applications* / *Informatica*). SOICT 2026 đã qua hạn (abstract 9/9, full paper 16/9/2026) → nhắm chu kỳ 2027 trở đi.

---

## 11. Checklist

⏳ Chờ yêu cầu mới của người dùng.

---

## Tài liệu tham khảo chính

- Danh sách bộ dữ liệu + 19 công trình huấn luyện trên chúng: `Dataset.md` Mục 6 (nguồn tham khảo ở Mục 10).
- RiverMamba — NeurIPS 2025, DOI `10.52202/085713-4446` · [arXiv 2505.22535](https://arxiv.org/abs/2505.22535) · [code](https://github.com/HakamShams/RiverMamba_code).
- G-Mamba — Chen & Tang, Neurocomputing 680 (2026), DOI [10.1016/j.neucom.2026.133280](https://doi.org/10.1016/j.neucom.2026.133280) (tổng quát, không phải thủy văn — xem Mục 5).
- Wang et al., npj Natural Hazards 2025, DOI [10.1038/s44304-025-00083-6](https://doi.org/10.1038/s44304-025-00083-6) (GNN + đồ thị reachability, kết quả "10 tiếng").
- Mamba gốc — Gu & Dao, COLM 2024, arXiv 2312.00752.
- Khảo sát Mamba vs Transformer + 6 bài dự báo Q ở venue CNTT/AI: `02_literature_review.md`.
