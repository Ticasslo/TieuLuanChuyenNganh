# Khảo sát tài liệu (Literature Review) — Mamba & Dự báo lưu lượng dòng chảy

> File này ghi lại kết quả khảo sát tính hợp lý/đúng đắn của đề tài theo đúng 2 yêu cầu đánh giá đề tài: **(1) Yêu cầu riêng cho nhóm nghiên cứu** — trước hết đánh giá Mamba trong literature có cải thiện đáng kể không (Mục 1), rồi mới tìm ít nhất 5 bài báo **trong 5 năm trở lại đây** giải quyết bài toán dự báo lưu lượng, kiểm tra theo tiêu chí uy tín (không MDPI, không preprint, trích dẫn ≥5, ưu tiên Scimago Q1/Q2, ưu tiên VN → Đông Nam Á → châu Á, **bắt buộc đăng ở tạp chí/hội nghị thuộc ngành CNTT hoặc AI** — yêu cầu trực tiếp từ GVHD, bổ sung 18/8/2026) (Mục 2); **(2) Yêu cầu chung** — tính hợp lý TLCN phải ra được demo/phần mềm (Mục 3). Toàn bộ số liệu trích dẫn/quartile lấy từ Google Scholar và Scimago thật, verify qua WebSearch/WebFetch — không suy đoán.

---

## 1. Mamba có thật sự cải thiện đáng kể không?

**Bài 1 (chính):** Zihan Wang, Fanheng Kong, Shi Feng, Ming Wang, Xiaocui Yang, Han Zhao, Daling Wang, Yifei Zhang — *"Is Mamba Effective for Time Series Forecasting?"*, **Neurocomputing** (Elsevier, **Q1**, SJR ~1,47), 2024/2025, DOI [10.1016/j.neucom.2024.129178](https://doi.org/10.1016/j.neucom.2024.129178), **463 trích dẫn** (Google Scholar, tự tra trực tiếp 19/8/2026 — cập nhật từ 454). Không phải MDPI. Có bản arxiv ([2403.11144](https://arxiv.org/abs/2403.11144)) nhưng đã chính thức publish, không phải chỉ preprint.

**Bài 2 (bổ sung):** *"SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting"*, công bố tại **CIKM 2024** (Proceedings of the 34th ACM International Conference on Information and Knowledge Management), DOI [10.1145/3746252.3761394](https://doi.org/10.1145/3746252.3761394) — hội nghị uy tín cao ngành CS/AI, không phải preprint, không phải MDPI. Lưu ý: đây là hội nghị (conference), không phải tạp chí, nên không có Scimago quartile như bài journal. **68 trích dẫn** (Google Scholar, tự tra 19/8/2026 — cập nhật từ 67).

**Bài 3 (bổ sung, 15/8/2026):** *"CMMamba: channel mixing Mamba for time series forecasting"*, Li, Qin, Cui, Sun, Wang — **Journal of Big Data** (Springer/SpringerOpen, Scimago **Q1**, SJR 1,979), 2024, DOI [10.1186/s40537-024-01001-9](https://doi.org/10.1186/s40537-024-01001-9) — Open Access, không phải MDPI, không phải preprint. **32 trích dẫn** (Google Scholar, tự tra trực tiếp 15/8/2026 — chính xác hơn số 31 lấy từ metric Springer trước đó). Dùng Bidirectional Mamba (đúng cơ chế đã học ở `LyThuyetCauTruc.md` 4.7) làm lõi, thêm Channel Mixing (trộn thông tin giữa các biến có độ tương quan cao) — kết quả: **đạt/vượt hiệu năng Transformer SOTA (PatchTST)** trên 7 dataset công khai (ETT, Weather, Exchange, ILI), đồng thời nhanh hơn, tốn ít bộ nhớ hơn. Đây là bằng chứng có lợi cho Mamba (khác 2 bài trên vốn kết luận "ngang nhau") — nhưng chỉ là 1 kiến trúc Mamba **cải tiến riêng**, không phải Mamba gốc, nên không phủ nhận kết luận chung ở dưới, chỉ làm rõ thêm "Mamba cải tiến đúng cách có thể thắng". ⚠️ Không liên quan trực tiếp tới RiverMamba về kiến trúc (CMMamba trộn theo BIẾN/kênh, RiverMamba trộn theo ĐIỂM KHÔNG GIAN qua serialization+LOAN) — chỉ dùng làm bằng chứng chung cho câu hỏi "Mamba vs Transformer", không phải bài liên quan trực tiếp tới phương pháp đề tài.

**Bài 4 (bổ sung, 19/8/2026):** *"DTMamba: Dual Twin Mamba for Time Series Forecasting"*, Wu Z, Gong Y, Zhang A, Li B — **Tsinghua Science and Technology** (Tsinghua University Press, Scimago **Q1**, category Computer Science/Software Engineering), 2026 (vol 31, issue 2, tr. 1124-1136), DOI [10.26599/TST.2024.9010143](https://doi.org/10.26599/TST.2024.9010143) — không phải MDPI, không phải preprint (đã publish chính thức). **25 trích dẫn** (Google Scholar, tự tra 19/8/2026 — lưu ý: Crossref/Web of Science riêng của tạp chí chỉ ghi 3/1 trích dẫn vì bản journal chính thức mới đăng 2026, trong khi Google Scholar gộp cả trích dẫn tới bản arxiv gốc đăng từ 5/2024 — dùng Google Scholar để nhất quán với toàn bộ phương pháp luận file này). Dùng "dual twin Mamba block" để bắt phụ thuộc dài hạn + residual network — kết quả: **vượt 11 model SOTA khác** (đa số dựa trên Transformer) trên 8 dataset công khai. Thêm 1 bằng chứng có lợi cho Mamba, cùng hướng với CMMamba (Bài 3).

**Kết luận — cần nói thẳng vì đây là rủi ro thật cho việc bảo vệ đề tài:** 2 bài đầu (Wang et al., SST) đồng thuận — literature hiện tại **không cho thấy Mamba vượt trội rõ ràng** so với Transformer nói chung. Bài 3 và 4 (CMMamba, DTMamba) bổ sung góc nhìn cân bằng hơn: kiến trúc Mamba **cải tiến đúng cách** (thêm channel-mixing, hoặc dual twin block) CÓ THỂ đạt/vượt Transformer SOTA — nghĩa là kết luận đúng nhất không phải "Mamba thua/ngang Transformer tuyệt đối", mà là "hiệu năng phụ thuộc cách thiết kế cụ thể, không có bên nào áp đảo cố hữu". Hai kiến trúc có thế mạnh khác nhau:
- Mamba: độ phức tạp tính toán gần tuyến tính (nhanh hơn, rẻ hơn), nắm tốt phụ thuộc dài hạn, nhưng nén lịch sử vào 1 trạng thái kích thước cố định → có thể mất thông tin.
- Transformer: nhìn được toàn bộ chuỗi (kể cả tương lai nếu cho phép), mạnh về động lực ngắn hạn, nhưng độ phức tạp bậc hai (chậm, tốn tài nguyên hơn).

Xu hướng nghiên cứu mới nhất (2025-2026) nghiêng về **hybrid Mamba-Transformer** thay vì chọn hẳn 1 bên — nhiều paper mới (SST, DeMa, FLDmamba, UniMamba...) đều là dạng lai.

**Ý nghĩa cho đề tài:** Không nên lập luận "chọn Mamba vì nó tốt hơn Transformer nói chung" — không có bằng chứng đủ mạnh cho khẳng định này. Nên lập luận theo hướng: (1) hiệu quả tính toán (huấn luyện được trên GPU free, phù hợp ngân sách sinh viên), (2) đây là kiến trúc mới, ít người áp dụng cho bài toán thủy văn Việt Nam — có giá trị khám phá, (3) RiverMamba (paper gốc đề tài dựa vào) đã chứng minh được hiệu quả thật trên GRDC toàn cầu so với GloFAS/LSTM (Mục 2.5 file kế hoạch chính).

### Q&A chuẩn bị bảo vệ — câu hỏi "vì sao Mamba, có cải thiện thật không"

**Q: Mamba có thật sự cải thiện đáng kể không?**
A: Có điều kiện — không phải so với Transformer nói chung (ngang nhau, không bên nào áp đảo). Nhưng so với cách làm cũ trong đúng bài toán đề tài (RiverMamba, số liệu thật trên GRDC):

| So với | R² | Cải thiện tương đối |
|---|---|---|
| GloFAS (mô hình vật lý đang vận hành thật) | 0,289 → 0,506 | +75% |
| LSTM | 0,462 → 0,506 | +10% |

**Q: Vì sao dùng Mamba, không phải Transformer hay kiến trúc khác?**
A: 3 lý do, không phải vì "Mamba tốt nhất":
1. Input là chuỗi rất dài (nhiều ngày × nhiều điểm sông) — Mamba xử lý tuyến tính, không phải cắt bớt dữ liệu như Transformer (bậc hai).
2. Đã có sẵn checkpoint pretrained + code công khai (RiverMamba, GitHub) — dùng Transformer nghĩa là train từ đầu, không đủ nguồn lực 1 tiểu luận sinh viên (đã tra: hiện chưa có hệ thống Transformer nào tương đương — pretrain toàn cầu + benchmark GRDC + công khai checkpoint — để kế thừa).
3. Tính mới: khảo sát bài báo dự báo lưu lượng ở Việt Nam (Mục 2) — chưa ai dùng Mamba (SSM) cho dự báo lưu lượng ở VN. ⚠️ Riêng Transformer thì ĐÃ có người thử (bài #2 Mục 2, Mekong, IEEE Access 2023) — kết luận chính bài đó: *"Transformer is not recommended for long-term prediction"*, LSTM thắng. Đây thực ra là bằng chứng có lợi cho Mamba: Mamba (SSM) giải quyết đúng điểm yếu khiến Transformer thua LSTM ở bài toán này (nén trạng thái tuyến tính thay vì attention bậc hai) — nhưng câu "chưa ai dùng Transformer ở VN" là SAI, đã sửa lại ở đây (phát hiện 18/8/2026).

**Q: Lỡ có người xây "RiverTransformer" tương đương thì sao?**
A: Hiện tại (12/8/2026, đã tra) chưa có — bài Transformer gần nhất chỉ test ở quy mô nhỏ (125 điểm, Iowa Mỹ), không toàn cầu, không rõ checkpoint công khai. Nếu sau này có, đó là tiến bộ khoa học bình thường, không phủ nhận giá trị kết quả đã làm tại thời điểm này — nghiên cứu không cần là "phương pháp duy nhất tồn tại" mới có ý nghĩa.

⚠️ **Bổ sung 15/8/2026 — trước đó chỉ mô tả sơ, chưa có tên/nguồn cụ thể, đã tra rõ để không bị hỏi mà không trả lời được:** bài này là *"Towards Generalized Hydrological Forecasting using Transformer Models for 120-Hour Streamflow Prediction"*, tác giả **Bekir Demiray, Ibrahim Demir** — arxiv [2406.07484](https://arxiv.org/abs/2406.07484), cũng có bản trên SSRN — **chưa tìm thấy bản journal chính thức đã publish, có thể chỉ đang ở dạng preprint**. Test trên 125 trạm thuộc bộ dữ liệu WaterBench-Iowa (Mỹ), so Transformer với LSTM/GRU/Seq2Seq/Persistence, dự báo 120 giờ. **Chỉ dùng làm bằng chứng phụ trong câu trả lời Q&A này** (không đưa vào danh sách bài chính Mục 2 vì có thể là preprint, phạm vi hẹp — chỉ 1 bang nước Mỹ, không phải toàn cầu như RiverMamba).

⚠️ **Cập nhật quan trọng (19/8/2026):** cùng 2 tác giả này vừa ra bản MỚI, lần này đánh giá thẳng **Mamba** (không chỉ Transformer nữa) — *"Advancing Long-Horizon Hydrological Forecasting: A Mamba-based Approach with Explainable AI for Generalized Streamflow Prediction"*, EarthArXiv, 9/2025, DOI [10.31223/X5B164](https://doi.org/10.31223/X5B164) (cũng có bản SSRN). Cùng 125 trạm Iowa, dự báo 120 giờ — kết luận: **Mamba đạt độ chính xác tương đương, vài khía cạnh nhỉnh hơn Transformer**, cả 2 đều vượt trội các phương pháp khác. **Vẫn chỉ là preprint EarthArXiv** (xác nhận trực tiếp: chưa qua peer review, chưa đăng tạp chí nào — có 1 nguồn tự nhận "published in Environmental Modelling and Software" nhưng đã tự kiểm tra và xác định SAI, DOI 10.31223 là của EarthArXiv chứ không phải tạp chí đó). Không đủ điều kiện đưa vào Mục 1/2 chính thức, nhưng **cần theo dõi** — đây là bằng chứng Mamba-cho-streamflow thứ 2 ngoài RiverMamba, phạm vi vẫn hẹp (1 bang Mỹ). Nếu bài này được publish chính thức trước khi bảo vệ, cần thêm lại vào Mục 1.

**Q: Vậy có phải chỉ vì tình cờ tìm được bài RiverMamba, không phải vì Mamba thật sự hợp?**
A: Đúng là khởi nguồn từ việc đọc được bài RiverMamba — nhưng quyết định theo tiếp dựa trên bằng chứng đã tự kiểm chứng (bảng R² ở trên, checkpoint/code có thật, khảo sát bài báo VN ở Mục 2 xác nhận tính mới), không phải chọn mù quáng vì tiện có sẵn.

**Q: Mamba có nhớ kém hơn Transformer không — tính toán ít hơn thì có đánh đổi gì không?** *(bổ sung 12/8/2026, sau khi tra kỹ cơ chế, xem chi tiết đầy đủ ở `LyThuyetCauTruc.md` Mục 4.2)*
A: Có 1 đánh đổi thật, cần nói thẳng khi bảo vệ, không né tránh: Mamba nén toàn bộ lịch sử vào 1 trạng thái kích thước cố định (khác Transformer giữ nguyên, không nén, mọi vị trí quá khứ). Trên benchmark "needle-in-the-haystack" (tìm đúng 1 chi tiết nhỏ giấu trong chuỗi rất dài), **Transformer nhớ chính xác hơn Mamba rõ rệt** — đây là hạn chế thật, đã được literature Mamba tự thừa nhận, không phải ưu điểm tuyệt đối. Tuy nhiên dự báo lưu lượng là bài toán cần **thông tin tổng hợp/xu hướng** (đất bão hòa tới đâu, mùa mưa giai đoạn nào) chứ không phải "nhớ chính xác 1 chi tiết nhỏ xa xưa" — nên đây là dạng bài toán **hợp với cách nén của Mamba**, không phải dạng bài toán mà điểm yếu này gây hại. Kết luận an toàn khi trả lời hội đồng: *"Mamba có đánh đổi thật về khả năng nhớ chi tiết đơn lẻ so với Transformer, nhưng dạng thông tin bài toán này cần (xu hướng tích lũy) phù hợp với cách nén của Mamba hơn, cộng thêm lợi thế tính toán trong ngân sách GPU free."*

---

## 2. Bài báo giải quyết bài toán dự báo lưu lượng và mực nước

Bài toán lớn của đề tài là dự báo **lưu lượng dòng chảy (Q)** — phục vụ cảnh báo nguy cơ lũ gián tiếp qua ngưỡng return period, không phải dự báo trực tiếp "lũ lụt"/vùng ngập (xem `CLAUDE.md` mục "Vài quyết định kỹ thuật đã chốt"). Chia làm 2 nhánh theo đại lượng dự báo: **lưu lượng dòng chảy (Q)** (Mục 2.1 — nhánh chính, bắt buộc theo yêu cầu chung) và **mực nước** (Mục 2.2 — nhánh phụ, bổ sung tham khảo). 2 đại lượng khác đơn vị/bản chất, không so trực tiếp được với RiverMamba (model chỉ ra Q) — nên mực nước không tính vào yêu cầu tối thiểu "≥5 bài", chỉ dùng làm tài liệu tham khảo mở rộng.

### 2.1 Dự báo lưu lượng dòng chảy (Q) — nhánh chính, bắt buộc theo yêu cầu chung

Tiêu chí lọc: không MDPI, không chỉ preprint, trích dẫn ≥5 (Google Scholar thật), ưu tiên Scimago Q1/Q2, ưu tiên tuyệt đối Việt Nam, trong 5 năm trở lại đây (2021-2026), **bắt buộc đăng ở tạp chí/hội nghị thuộc ngành CNTT hoặc AI** (Scimago Subject Category có "Computer Science"/"Artificial Intelligence", hoặc venue IEEE/ACM — yêu cầu trực tiếp từ GVHD).

**Nhóm chính — 7 bài, vượt tiêu chí tối thiểu (19/8/2026), tách theo vùng ưu tiên**

**Việt Nam (3 bài)**

| # | Bài báo | Tạp chí (Scimago) | Năm | Trích dẫn | Vùng/Trạm | Thuật toán chính | DOI |
|---|---|---|---|---|---|---|---|
| 1 | Comparison of Deep Learning Techniques for River Streamflow Forecasting | *IEEE Access* (**Q1** — category: Computer Science, Engineering) | 2021 | **185** | Sông Hồng — **trạm Sơn Tây** | FFNN, CNN, LSTM, GRU, Stacked LSTM, BiLSTM (so 6 model) | [10.1109/ACCESS.2021.3077703](https://doi.org/10.1109/ACCESS.2021.3077703) |
| 2 | Streamflow Prediction in the Mekong River Basin Using Deep Neural Networks | *IEEE Access* (**Q1**) | 2023 | **21** | Mekong — ⚠️ chưa xác nhận trạm cụ thể (dữ liệu khí hậu/khí tượng rộng cả lưu vực) | MLP, CNN, LSTM, **Transformer** — kết luận: LSTM tốt nhất, Transformer không hợp dự báo dài hạn. Tác giả: Nguyen T-T-H, Vu D-Q, Mai S.T., Dang T.D. (Việt Nam) | [10.1109/ACCESS.2023.3301153](https://doi.org/10.1109/ACCESS.2023.3301153) |
| 3⚠️ | Hybrid model to improve the river streamflow forecasting utilizing multi-layer perceptron-based intelligent water drop optimization algorithm | *Soft Computing* (Springer, **Q2** — category: Software, Theoretical Computer Science) | **2020** | **60** | **Nông Sơn + Thành Mỹ, Vu Gia-Thu Bồn** (đúng lưu vực đề tài, đúng 2 trạm đang xin KTTV) | MLP + Intelligent Water Drop optimization. Tác giả: Pham QB, Afan HA, Mohammadi B, Ahmed AN, Linh NTT, Vo ND, Moazenzadeh R, Yu PS, El-Shafie A | [10.1007/s00500-020-05058-5](https://doi.org/10.1007/s00500-020-05058-5) |

⚠️ **Bài #3 ngoài mốc "5 năm trở lại đây"** (2020, lệch 1 năm) — quyết định có chủ đích đưa vào nhóm chính (19/8/2026): đã có đủ 6 bài khác đạt mọi tiêu chí kể cả năm, nên bài này là bổ sung giá trị cao (đúng ngay lưu vực đề tài) chứ không phải "chữa cháy" thiếu số lượng. Cần nói rõ điểm lệch này nếu hội đồng hỏi.

**Đông Nam Á (1 bài)**

| # | Bài báo | Tạp chí (Scimago) | Năm | Trích dẫn | Vùng/Trạm | Thuật toán chính | DOI |
|---|---|---|---|---|---|---|---|
| 4 | Comparative analysis of artificial intelligence methods for streamflow forecasting | *IEEE Access* (**Q1**) | 2024 | **27** | Sông Johor, Malaysia — ⚠️ chưa xác nhận tên trạm cụ thể | So nhiều mô hình ML/DL + wavelet transform | [10.1109/ACCESS.2024.3351754](https://doi.org/10.1109/ACCESS.2024.3351754) |

**Châu Á (3 bài: Trung Quốc x2, Pakistan x1)**

| # | Bài báo | Tạp chí (Scimago) | Năm | Trích dẫn | Vùng/Trạm | Thuật toán chính | DOI |
|---|---|---|---|---|---|---|---|
| 5 | Improved Transformer Model for Enhanced Monthly Streamflow Predictions of the Yangtze River | *IEEE Access* (**Q1**) | 2022 | **87** | Dương Tử — **trạm Hán Khẩu (Hankou)** | Transformer (double-encoder, cross-attention) | [10.1109/ACCESS.2022.3178521](https://doi.org/10.1109/ACCESS.2022.3178521) |
| 6 | Study on runoff forecasting and error correction driven by atmosphere–ocean-land dataset | *Expert Systems with Applications* (**Q1** — category: AI, Computer Science Applications) | 2024 (online 11/2024) / 2025 (số in chính thức, theo CrossRef) | **21** | Yalong — **trạm Lianghekou** | GPR, LSTM, SVM + P-XGBoost-SHAP feature selection + EEMD-AR error correction | [10.1016/j.eswa.2024.125744](https://doi.org/10.1016/j.eswa.2024.125744) |
| 7 | Forecasting Floods Using Deep Learning Models: A Longitudinal Case Study of Chenab River, Pakistan | *IEEE Access* (**Q1**) | 2024 | **9** | Sông Chenab, Pakistan — ⚠️ chưa xác nhận trạm nào trong 4 trạm phổ biến (Marala/Khanki/Qadirabad/Trimmu) | LSTM, ML-GMDH — dự báo lưu lượng ngày (R²=0,91). Tác giả: Aatif K., Fahiem M.A., Tahir F. — cả 3 đều thuộc khoa Công nghệ thông tin (Lahore College for Women University) | [10.1109/ACCESS.2024.3445586](https://doi.org/10.1109/ACCESS.2024.3445586) |

✅ **Đạt và vượt yêu cầu tối thiểu "≥5 bài trong 5 năm trở lại đây"** (7 bài, 6/7 đạt đúng mốc năm), đúng thứ tự ưu tiên vùng (VN → Đông Nam Á → châu Á): 3 bài VN, 1 bài Đông Nam Á, 3 bài châu Á. 5/7 bài là IEEE Access (#1,2,4,5,7), 1 bài Expert Systems with Applications (#6), 1 bài Soft Computing (#3) — toàn bộ đều Scimago category CNTT/AI. Đã tra rất rộng (>50 câu truy vấn, nhiều vòng, nhiều tạp chí CNTT/AI khác ngoài IEEE) cho cả 3 vùng trước khi chốt — không tìm thêm được bài nào khác đạt tiêu chí.

⚠️ **Bài #2 quan trọng cho lập luận Mamba:** đây là bằng chứng cho thấy Transformer ĐÃ được thử ở đúng vùng Mekong/VN và THUA LSTM — củng cố lý do chọn Mamba (SSM giải quyết đúng điểm yếu này) thay vì Transformer thuần. Xem sửa lại ở Mục 1.

### 2.2 Dự báo mực nước — nhánh phụ, bổ sung tham khảo

Cùng tiêu chí CNTT/AI + ≥5 trích dẫn như Mục 2.1, chỉ khác đại lượng dự báo (mực nước thay vì Q). Không tính vào yêu cầu tối thiểu "≥5 bài" của đề tài (yêu cầu đó chỉ áp dụng cho lưu lượng, xem đầu Mục 2) — dùng làm tài liệu tham khảo bổ sung, không trích dẫn như bằng chứng "cùng bài toán Q" với RiverMamba.

| # | Bài báo | Tạp chí (Scimago) | Năm | Trích dẫn | Vùng/Trạm | Thuật toán chính | DOI |
|---|---|---|---|---|---|---|---|
| 2.2.1⚠️ | Water Level Prediction Model Based on GRU and CNN | *IEEE Access* (**Q1**) | **2020** | **264** | Dương Tử + nhiều sông khác, Trung Quốc — nhiều trạm qua IoT, không phải 1 trạm đơn | GRU + CNN (kết hợp không gian-thời gian qua IoT). Tác giả: Pan M, Zhou H, Cao J, Liu Y, Hao J, Li S, Chen CH | [10.1109/ACCESS.2020.2982433](https://doi.org/10.1109/ACCESS.2020.2982433) |

⚠️ Ngoài mốc "5 năm trở lại đây" (2020, lệch 1 năm) — giữ lại vì 264 trích dẫn (mạnh nhất toàn file, kể cả so với Mục 2.1) và dùng GRU+CNN (deep learning thật).

📝 **Đã loại khỏi Mục 2.2:** Water Level Prediction at TICH-BUI river in Vietnam Using Support Vector Regression (IEEE ICMLC 2019, Nguyen TT, Le HTT) — bỏ vì **vừa 2019 (lệch mốc năm) vừa chỉ dùng SVR** (không phải deep learning) — không đủ giá trị bù cho việc lệch năm, khác với GRU-CNN (lệch năm nhưng bù bằng 264 trích dẫn + đúng deep learning). Multi-input LSTM for water level forecasting in Black River at the border of Vietnam-China (IEEE ICMLANT 2021, tác giả Truong Thi Hai Yen et al.) — chỉ **1 trích dẫn**, không đạt ≥5 dù đã nới tiêu chí bài toán. Cũng đã loại (sai ngành tạp chí, dù đúng bài toán mực nước): "Accurate discharge and water level forecasting... Red River + Dakbla, Việt Nam" (Scientific Reports — Scimago category Multidisciplinary, không có Computer Science).

---

## 3. Tính hợp lý của TLCN — có ra được phần mềm/demo không?

Yêu cầu gốc: *"Phải ra được phần mềm/demo. Trường hợp là demo thì sang KLTN phải nâng lên thành phần mềm có tính ứng dụng."*

Đối chiếu với `flood-forecasting-research.md`:
- **Tiểu luận:** đã có kế hoạch demo thật — Mục 8.1 (VPS Oracle Cloud, host inference + dashboard) và Mục 10 (Đầu ra: "Báo cáo tiểu luận + demo VPS"). Hệ thống cập nhật dự báo 1 lần/ngày, hiển thị qua dashboard — đúng dạng "demo" theo yêu cầu đề ra, không chỉ là báo cáo/notebook.
- **Khóa luận:** kế hoạch hiện tại (Mục 10) mới ghi "vẫn 1 hệ thống, chỉ đổi lõi bên trong" — **chưa ghi rõ bước "nâng demo thành phần mềm có tính ứng dụng"** như yêu cầu đề ra cho KLTN. Đây là điểm cần bổ sung vào kế hoạch khi làm khóa luận (VD: thêm giao diện người dùng đầy đủ, xử lý được nhiều người dùng cùng lúc, không chỉ là dashboard xem kết quả) — ghi nhận ở đây để không quên, chưa cần làm ngay.

✅ **Kết luận:** phần "tiểu luận phải ra demo" đã có sẵn trong kế hoạch, không thiếu. Phần "khóa luận phải nâng cấp thành phần mềm ứng dụng" chưa được ghi cụ thể — cần làm rõ thêm "phần mềm có tính ứng dụng" nghĩa là mức nào (có tài khoản người dùng? Nhiều lưu vực chọn được? Cảnh báo qua email/SMS?) khi tới giai đoạn khóa luận.

---

## 4. Ý nghĩa tổng hợp cho đề tài

- **Đủ và vượt yêu cầu tối thiểu (7 bài, tối thiểu là 5), đúng thứ tự ưu tiên vùng** (3 VN, 1 Đông Nam Á, 3 châu Á) — toàn bộ Scimago category CNTT/AI (5 IEEE Access + 1 Expert Systems with Applications + 1 Soft Computing). Đã tra rất kỹ cả 3 vùng ưu tiên (nhiều vòng, hơn 50 câu truy vấn, nhiều tạp chí CNTT/AI khác ngoài IEEE) trước khi chốt danh sách này.
- **Thuật toán đã thử ở 7 bài:** LSTM, GRU, CNN, FFNN, BiLSTM, Transformer (2 bài, cả VN lẫn Trung Quốc), GPR, SVM, ML-GMDH, MLP+Intelligent Water Drop — **chưa bài nào dùng Mamba/SSM**, củng cố tính mới của đề tài.
- **Bằng chứng quan trọng cho lập luận chọn Mamba:** bài #2 (Mekong, VN) cho thấy Transformer thua LSTM ở đúng bài toán/vùng gần đề tài nhất — Mamba (SSM) giải quyết đúng điểm yếu này (nén trạng thái tuyến tính thay vì attention bậc hai), là lý do kỹ thuật cụ thể chứ không chỉ "kiến trúc mới nên thử".
- **Bài #3 đúng ngay lưu vực đề tài:** Pham et al. 2020, Vu Gia-Thu Bồn, trạm Nông Sơn/Thành Mỹ — duy nhất trong 7 bài lệch mốc năm (2020), đưa vào nhóm chính có chủ đích vì giá trị vùng nghiên cứu, cần nói rõ điểm lệch này nếu hội đồng hỏi.
- **Vẫn còn đúng, không đổi:** Mamba không có bằng chứng vượt trội tuyệt đối so với Transformer nói chung (Mục 1) — chuẩn bị sẵn câu trả lời theo hướng hiệu quả tính toán + tính mới + bằng chứng cụ thể (bài #2), không phải "Mamba luôn thắng".
- **Nhánh phụ mực nước (Mục 2.2):** 1 bài tham khảo (GRU-CNN, Trung Quốc, 2020) — không tính vào yêu cầu tối thiểu, lệch mốc năm nhưng giữ lại vì 264 trích dẫn (mạnh nhất file) + đúng deep learning. Dùng khi cần mở rộng góc nhìn "lũ lụt nói chung", không dùng thay thế bằng chứng cho bài toán Q.
