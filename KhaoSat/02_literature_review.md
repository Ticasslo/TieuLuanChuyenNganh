# Khảo sát tài liệu — Mamba và bài toán dự báo lưu lượng dòng chảy

> Tài liệu trình bày kết quả khảo sát tính hợp lý của đề tài theo hai yêu cầu đánh giá: **(1) yêu cầu riêng cho nhóm nghiên cứu** — đánh giá Mamba có cải thiện đáng kể trong các công trình đã công bố hay không (Mục 1), và tìm ít nhất 5 bài báo trong 5 năm gần đây giải quyết bài toán dự báo lưu lượng theo tiêu chí uy tín: không MDPI, không chỉ là preprint, trích dẫn ≥ 5, ưu tiên Scimago Q1/Q2, ưu tiên Việt Nam → Đông Nam Á → châu Á, **bắt buộc đăng ở tạp chí/hội nghị ngành CNTT hoặc AI** (Mục 2); **(2) yêu cầu chung** — tiểu luận phải ra được demo/phần mềm (Mục 3). Số trích dẫn và hạng Scimago lấy từ Google Scholar và Scimago tại thời điểm tra (08–09/2026).
>
> Hướng đề tài hiện tại: tự xây dựng và so sánh mô hình trên 4 bộ dữ liệu benchmark công khai (`Dataset.md`, `flood-forecasting-research.md`); RiverMamba là tài liệu tham khảo về kiến trúc.

---

## 1. Mamba có thật sự cải thiện đáng kể không?

**1.1. Các bài đánh giá Mamba cho dự báo chuỗi thời gian.**

| # | Bài | Nơi công bố | Trích dẫn | Kết luận chính |
|---|---|---|---|---|
| 1 | Wang, Kong, Feng, Wang, Yang, Zhao, Wang, Zhang — *"Is Mamba Effective for Time Series Forecasting?"*, DOI [10.1016/j.neucom.2024.129178](https://doi.org/10.1016/j.neucom.2024.129178) | Neurocomputing (Q1, SJR ~1,47), 2024/2025 | 463 | Mamba ngang Transformer, không vượt trội rõ ràng |
| 2 | *"SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting"*, DOI [10.1145/3746252.3761394](https://doi.org/10.1145/3746252.3761394) | CIKM 2025 (hội nghị ACM) | 68 | Đề xuất lai Mamba–Transformer; hai kiến trúc có thế mạnh khác nhau |
| 3 | Li, Qin, Cui, Sun, Wang — *"CMMamba: channel mixing Mamba for time series forecasting"*, DOI [10.1186/s40537-024-01001-9](https://doi.org/10.1186/s40537-024-01001-9) | Journal of Big Data (Q1, SJR 1,979), 2024 | 32 | Bidirectional Mamba + trộn kênh đạt/vượt PatchTST trên 7 bộ dữ liệu công khai, nhanh hơn, ít bộ nhớ hơn |
| 4 | Wu, Gong, Zhang, Li — *"DTMamba: Dual Twin Mamba for Time Series Forecasting"*, DOI [10.26599/TST.2024.9010143](https://doi.org/10.26599/TST.2024.9010143) | Tsinghua Science and Technology (Q1, Computer Science), 2026 | 25 | Vượt 11 mô hình SOTA (đa số Transformer) trên 8 bộ dữ liệu |

**1.2. Kết luận.** Hai bài đầu cho thấy các công trình hiện tại **không chứng minh Mamba vượt trội rõ ràng** so với Transformer nói chung. Hai bài sau cho thấy kiến trúc Mamba được cải tiến phù hợp (trộn kênh, khối dual twin) có thể đạt hoặc vượt Transformer SOTA. Kết luận hợp lý nhất: hiệu năng phụ thuộc thiết kế cụ thể, không kiến trúc nào áp đảo cố hữu.
- **Mamba:** độ phức tạp gần tuyến tính theo độ dài chuỗi, nắm tốt phụ thuộc dài hạn, nhưng nén lịch sử vào trạng thái kích thước cố định nên có thể mất thông tin chi tiết.
- **Transformer:** nhìn được toàn bộ chuỗi, mạnh ở động lực ngắn hạn, nhưng độ phức tạp bậc hai.
- Xu hướng 2025–2026 nghiêng về kiến trúc lai Mamba–Transformer (SST, DeMa, FLDmamba, UniMamba).
- Block-Biased Mamba (Yu & Erichson, NeurIPS 2025) chỉ ra Mamba kém S4D ở tác vụ phụ thuộc xa và đề xuất biến thể B2S6. Thí nghiệm của bài chỉ gồm Long-Range Arena và mô hình hóa ngôn ngữ, chưa có chuỗi thời gian thủy văn; bản thân LRA bị chỉ ra là chủ yếu đo phụ thuộc gần (Miralles-González và cs., arXiv 2501.14850). Mức áp dụng cho dự báo lưu lượng vì vậy là câu hỏi thực nghiệm, cần đối chứng Mamba với S4D.

**1.3. Ý nghĩa cho đề tài.** Không lập luận "chọn Mamba vì Mamba tốt hơn Transformer". Đề tài được định vị là **nghiên cứu so sánh**, với các lý do: (1) hiệu quả tính toán với chuỗi dài, phù hợp tài nguyên sinh viên; (2) Mamba còn ít được áp dụng cho bài toán dự báo lưu lượng; (3) RiverMamba (NeurIPS 2025) cho thấy Mamba cải thiện so với GloFAS và LSTM trên dữ liệu GRDC toàn cầu.

**1.4. Mamba và SSM cho bài toán lưu lượng.**

Công trình dùng **Mamba** cho đúng bài toán lưu lượng/dòng chảy:

| Bài | Nơi công bố | Venue CNTT/AI | Ghi chú |
|---|---|---|---|
| RiverMamba (Shams Eddin, Zhang, Kollet, Gall) | NeurIPS 2025 | Có | R² so với GloFAS 0,289 → 0,506 (+75%), so với LSTM 0,462 → 0,506 (+10%) trên GRDC |
| ResBi-Mamba Plus (Sheng, Wen, Feng, Shi), DOI [10.23919/AISE.2026.000010](https://doi.org/10.23919/AISE.2026.000010) | Artificial Intelligence Science and Engineering (IEEE Xplore), 6/2026 | Có | Dự báo giờ trên Columbia River (CBR 2016–2019); tạp chí mới, chưa có hạng Scimago, bài chưa đủ 5 trích dẫn |
| RMC (Res-Mamba-Causal) | Journal of Hydrology 665, 2026 | Không | Dự báo ngày tại McKenzie River, NSE 0,9804 |
| MamGA, Daily runoff Mamba (Hydrology Research), D2Mamba | Tạp chí thủy văn | Không | — |
| Demiray & Demir — *"Advancing Long-Horizon Hydrological Forecasting: A Mamba-based Approach with Explainable AI…"*, DOI [10.31223/X5B164](https://doi.org/10.31223/X5B164) | EarthArXiv 9/2025 (preprint) | — | 125 trạm Iowa, dự báo 120 giờ; Mamba tương đương, một số khía cạnh nhỉnh hơn Transformer |

Công trình dùng **họ SSM (S4/S4D)** chứ không phải Mamba: Jing và cs., *Expert Systems with Applications* 2026 (S4D/S5D trong mô hình lai, 671 lưu vực CAMELS-US, venue CNTT/AI); Wang và cs., *Water Resources Research* 2025 (S4D-FT); Zhang và cs., *Journal of Hydrology* 2025 (gọi S4D là "Mamba-type", bài toán xả hồ chứa).

Công trình liên quan khác: dự báo hạn hán SPEI bằng Mamba (Tang và cs., IEEE TGRS 2025); FloodMamba-Net giám sát hình thái sông bằng ảnh vệ tinh (Journal of Hydrology 2025) — khác bài toán. Tra arXiv với "mamba AND hydrology" chỉ ra RiverMamba. Nhóm tác giả ResBi-Mamba Plus có chuỗi công trình dự báo dòng chảy ở venue CNTT/AI (IEEE TNNLS 2024, Neural Networks 2026, ESWA 2024, IEEE IoT 2023) dùng TCN/GRU, chỉ chuyển sang Mamba ở bài 2026.

**Tổng hợp theo mức độ tiêu chí:** Mamba + đúng bài toán Q + venue CNTT/AI: 2 bài (RiverMamba, ResBi-Mamba Plus). Mở rộng sang họ SSM: thêm 1 bài (Jing và cs., ESWA 2026). Phạm vi tra cứu: Google Scholar (nhiều biến thể từ khóa, khu vực, ngôn ngữ, "Related articles"), Semantic Scholar, dblp, arXiv API, OpenAlex, Crossref, OpenReview, IEEE Xplore, ACM DL, GitHub.

Lưu ý khi tra cứu: "Jamba" trùng tên tạp chí *Jàmbá*; "Hyena" trùng "Spotted Hyena Optimizer"; "StreamFlow" trong thị giác máy tính là thuật toán optical flow; "Mamba (B7H015)" là tên trạm đo ở Nam Phi.

### Câu hỏi chuẩn bị bảo vệ

**Mamba có thật sự cải thiện đáng kể không?** So với Transformer nói chung thì không có bằng chứng vượt trội tuyệt đối. Trên bài toán lưu lượng, RiverMamba cải thiện so với GloFAS (+75% R²) và LSTM (+10% R²); các kiến trúc Mamba được cải tiến phù hợp có thể vượt Transformer SOTA. Đề tài vì vậy được thiết kế là nghiên cứu so sánh có kiểm soát.

**Vì sao dùng Mamba, không phải Transformer?**
1. Đầu vào là chuỗi dài; Mamba xử lý tuyến tính theo độ dài chuỗi, Transformer bậc hai.
2. Mamba còn rất mới cho bài toán dự báo lưu lượng (Mục 1.4). Transformer đã được thử ở lưu vực Mekong và thua LSTM (bài #2, Mục 2.1) — Mamba với cơ chế nén trạng thái tuyến tính là một hướng khác để kiểm chứng.

**Nếu có công trình Transformer tương đương thì sao?** Đó là tiến bộ khoa học bình thường, không phủ nhận giá trị kết quả so sánh tại thời điểm thực hiện.

**Mamba có nhớ kém hơn Transformer không?** Có một đánh đổi thật: Mamba nén lịch sử vào trạng thái kích thước cố định; trên bài kiểm tra "needle-in-a-haystack", Transformer nhớ chi tiết đơn lẻ chính xác hơn. Tuy nhiên dự báo lưu lượng cần thông tin tổng hợp và xu hướng tích lũy (độ ẩm đất, giai đoạn mùa mưa) hơn là chi tiết đơn lẻ, nên phù hợp với cách nén của Mamba (chi tiết ở `LyThuyetCauTruc.md` Mục 4.2).

---

## 2. Bài báo giải quyết bài toán dự báo lưu lượng và mực nước

Bài toán của đề tài là dự báo **lưu lượng dòng chảy (Q)**; nguy cơ lũ được đánh giá gián tiếp qua ngưỡng return period trên Q. Mục 2.1 là nhánh chính (bắt buộc); Mục 2.2 là nhánh mực nước, chỉ để tham khảo.

### 2.1. Dự báo lưu lượng dòng chảy

Tiêu chí: không MDPI, không chỉ preprint, trích dẫn ≥ 5, ưu tiên Scimago Q1/Q2, trong giai đoạn 2021–2026, bắt buộc ngành CNTT/AI (Scimago Subject Category có Computer Science/Artificial Intelligence, hoặc venue IEEE/ACM).

**Việt Nam**

| # | Bài báo | Tạp chí | Năm | Trích dẫn | Vùng/Trạm | Thuật toán | DOI |
|---|---|---|---|---|---|---|---|
| 1 | Comparison of Deep Learning Techniques for River Streamflow Forecasting | IEEE Access (Q1) | 2021 | 185 | Sông Hồng — trạm Sơn Tây | FFNN, CNN, LSTM, GRU, Stacked LSTM, BiLSTM | [10.1109/ACCESS.2021.3077703](https://doi.org/10.1109/ACCESS.2021.3077703) |
| 2 | Streamflow Prediction in the Mekong River Basin Using Deep Neural Networks | IEEE Access (Q1) | 2023 | 21 | Mekong — 7 trạm dòng chính (Chiang Saen, Luang Prabang, Nong Khai, Nakhon Phanom, Mukdahan, Pakse, Kratie) | MLP, CNN, LSTM, Transformer — LSTM tốt nhất mọi lead time; Transformer kém nhất, mùa khô dài hạn tại Kratie NSE 0,38. Tác giả: Nguyen T-T-H, Vu D-Q, Mai S.T., Dang T.D. | [10.1109/ACCESS.2023.3301153](https://doi.org/10.1109/ACCESS.2023.3301153) |

**Đông Nam Á**

| # | Bài báo | Tạp chí | Năm | Trích dẫn | Vùng/Trạm | Thuật toán | DOI |
|---|---|---|---|---|---|---|---|
| 3 | Comparative analysis of artificial intelligence methods for streamflow forecasting | IEEE Access (Q1) | 2024 | 27 | Sông Johor, Malaysia (1977–2005, bước tháng) | ANN (MLP) và CNN + wavelet + bản Bayesian đo bất định | [10.1109/ACCESS.2024.3351754](https://doi.org/10.1109/ACCESS.2024.3351754) |

**Châu Á**

| # | Bài báo | Tạp chí | Năm | Trích dẫn | Vùng/Trạm | Thuật toán | DOI |
|---|---|---|---|---|---|---|---|
| 4 | Improved Transformer Model for Enhanced Monthly Streamflow Predictions of the Yangtze River | IEEE Access (Q1) | 2022 | 87 | Dương Tử — trạm Hán Khẩu | Transformer 2 encoder (lưu lượng + ENSO) + cross-attention + VMD; R² > 0,91 trong 21 năm, RMSE 2.579 m³/tháng. Tác giả: Liu, Liu, Mu | [10.1109/ACCESS.2022.3178521](https://doi.org/10.1109/ACCESS.2022.3178521) |
| 5 | Study on runoff forecasting and error correction driven by atmosphere–ocean-land dataset | Expert Systems with Applications (Q1, AI) | 2024/2025 | 21 | Yalong — trạm Lianghekou, 1958–2018 | GPR, LSTM, SVM + P-XGBoost-SHAP + hiệu chỉnh sai số EEMD-AR; NSE ~0,93 | [10.1016/j.eswa.2024.125744](https://doi.org/10.1016/j.eswa.2024.125744) |
| 6 | Forecasting Floods Using Deep Learning Models: A Longitudinal Case Study of Chenab River, Pakistan | IEEE Access (Q1) | 2024 | 9 | Sông Chenab, Pakistan (trạm cụ thể chưa xác nhận do trang chặn truy cập) | LSTM (R² 0,91, r 0,96), ML-GMDH (R² 0,88). Tác giả: Aatif, Fahiem, Tahir | [10.1109/ACCESS.2024.3445586](https://doi.org/10.1109/ACCESS.2024.3445586) |

**Nhận xét.** 6 bài đạt yêu cầu tối thiểu 5 bài, đúng thứ tự ưu tiên vùng (2 Việt Nam, 1 Đông Nam Á, 3 châu Á); 5 bài IEEE Access và 1 bài ESWA, đều thuộc ngành CNTT/AI. Các thuật toán đã được thử: LSTM, GRU, CNN, FFNN, BiLSTM, Transformer, GPR, SVM, ML-GMDH — chưa bài nào dùng Mamba/SSM.
- **Bài #1 (sông Hồng):** mô hình phức tạp hơn (Stacked LSTM 3 lớp, BiLSTM) không vượt LSTM/GRU 1 lớp; với đỉnh lũ 23/7/2014 (6.740 m³/s), LSTM sai 3,0% còn GRU 5,7% dù GRU có NSE tổng thể nhỉnh hơn ở dự báo 2 ngày — NSE cao không đồng nghĩa dự báo đúng đỉnh lũ.
- **Bài #2 (Mekong):** Transformer thua LSTM ở cả ngắn hạn và dài hạn; bài không giải thích cơ chế.
- **Bài #3 (Johor):** wavelet-ANN RMSE 119,25 so với ANN 126,88; CNN ổn định hơn nhưng quá khớp ở thành phần tần số thấp. Không so Transformer/Mamba; có giá trị tham khảo cho phần đánh giá bất định.
- **Bài #4 (Dương Tử):** Transformer tùy biến thắng ARIMA, TCN, LSTM và Transformer thường; năm lũ 1998/2016 R² > 0,95. Đối chiếu với bài #2 cho thấy cách tùy biến kiến trúc theo bài toán quyết định kết quả hơn là bản thân kiến trúc gốc.

### 2.2. Dự báo mực nước (tham khảo)

| # | Bài báo | Tạp chí | Năm | Trích dẫn | Vùng | Thuật toán | DOI |
|---|---|---|---|---|---|---|---|
| 2.2.1 | Water Level Prediction Model Based on GRU and CNN | IEEE Access (Q1) | 2020 | 264 | Dương Tử và nhiều sông khác, Trung Quốc | GRU + CNN qua IoT. Tác giả: Pan, Zhou, Cao, Liu, Hao, Li, Chen | [10.1109/ACCESS.2020.2982433](https://doi.org/10.1109/ACCESS.2020.2982433) |

Bài 2.2.1 lệch mốc 5 năm (2020) nhưng được giữ vì số trích dẫn cao và dùng học sâu. Đã loại: Water Level Prediction at TICH-BUI river (IEEE ICMLC 2019, SVR); Multi-input LSTM for water level forecasting in Black River (IEEE ICMLANT 2021, 1 trích dẫn); "Accurate discharge and water level forecasting… Red River + Dakbla" (Scientific Reports, ngành đa lĩnh vực).

---

## 3. Tính hợp lý của tiểu luận — khả năng ra phần mềm/demo

Yêu cầu: *"Phải ra được phần mềm/demo. Trường hợp là demo thì sang KLTN phải nâng lên thành phần mềm có tính ứng dụng."*

- **Tiểu luận:** 4 bộ dữ liệu benchmark là dữ liệu lịch sử, không có luồng cập nhật hằng ngày, nên demo sẽ chạy trên dữ liệu lịch sử (phát lại). Dạng demo cụ thể chưa chốt; đề xuất hiện tại (bản đồ trạm dự báo theo giờ, tô màu theo ngưỡng return period, kèm giải thích XAI) nằm ở `BaiCoSo/CHECKPDF.md` Mục 8.
- **Khóa luận:** cần làm rõ mức "phần mềm có tính ứng dụng" (tài khoản người dùng, chọn nhiều lưu vực, cảnh báo qua email/SMS…) khi tới giai đoạn khóa luận.

---

## 4. Ý nghĩa tổng hợp cho đề tài

- Đạt yêu cầu khảo sát (6 bài, đúng thứ tự ưu tiên vùng, đều thuộc ngành CNTT/AI); chưa bài nào dùng Mamba/SSM, cho thấy tính mới của hướng Mamba.
- Mamba không có bằng chứng vượt trội tuyệt đối so với Transformer; đề tài lập luận theo hiệu quả tính toán, tính mới và so sánh có kiểm soát.
- Bài #2 và #4 cho thấy kết quả phụ thuộc cách tùy biến kiến trúc theo bài toán — cơ sở để thiết kế thí nghiệm so sánh công bằng giữa các kiến trúc.
- Bài #1 cho thấy cần đánh giá riêng đỉnh lũ, không chỉ dựa vào NSE tổng thể — phù hợp với việc dùng trọng số return period trong hàm mất mát.
