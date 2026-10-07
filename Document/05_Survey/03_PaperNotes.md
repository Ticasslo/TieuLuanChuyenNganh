# Ghi chú đọc tài liệu tham khảo

> Tóm tắt từng tệp PDF trong `PaperResearch/PaperResearchPDF/`: nội dung chính, điều rút ra và chỗ áp dụng cho đề tài. Mỗi mục đọc từ phần tóm tắt, kết luận và các đoạn liên quan trong bài; số liệu trích nguyên từ bài. Ý tưởng có thể áp dụng được gom ở Mục 7 và ghi vào `Document/04_Ideas/01_Ideas.md` Mục 3; ý tưởng chỉ đưa vào pipeline khi được duyệt.

---

## 1. Bài cơ sở và phản biện (`BasePaper/`)

| Tệp | Nội dung | Rút ra, học được | Áp dụng cho đề tài |
|---|---|---|---|
| `Konold2026_HESS_BiasCast.pdf` (+ bản chữ `_FullText.md`) | Dự báo qmax ngày t trên 451 lưu vực Extended LamaH-CE bằng dự báo thời tiết ECMWF HRES thật; so baseline, cross-domain, Encoder–Decoder LSTM, Sequential Forecast LSTM, học chuyển giao, có/không Q quá khứ | Học trên tái phân tích rồi chạy với dự báo làm NSE trung vị giảm 0,58 → 0,33; Sequential Forecast LSTM có Q đạt 0,705, vượt cả baseline tái phân tích 0,69; học chuyển giao chỉ 0,41–0,44 | Bài cơ sở; chi tiết ở `Document/01_Plan/02_BasePaper.md` |
| `PeerReview/ResponseToReferee1.pdf` | Trả lời 24 ý của phản biện 1 | Validation và test dùng đủ 4 năm vì NeuralHydrology nạp thêm giai đoạn khởi động trước mỗi kỳ (thư nói cửa sổ 365 ngày; trong mã là 364 ngày = `seq_length` − `predict_last_n`); 27,5% trong 451 lưu vực là lưu vực lồng nhau (xác định qua `NEXTDOWNID`); mạng bàn giao trạng thái của Encoder–Decoder là 1 lớp kết nối đầy đủ 128 nút, không tinh chỉnh; khi học chuyển giao chỉ cập nhật lớp nhúng động, lớp nhúng tĩnh đóng băng; NeuralHydrology lặng lẽ bỏ lưu vực không tạo được mẫu hợp lệ; lưu vực kém nhất (758) nghi do tác động của con người; tác giả cho rằng persistence ít ý nghĩa với qmax vì tự tương quan ngày thấp | Persistence tự tính đạt 0,35–0,37, gần mô hình chỉ dùng dự báo (0,39) — số liệu phản biện lại lập luận của tác giả, nên giữ mốc persistence. Lưu vực lồng nhau: khi phân tích theo lưu vực (Y3) tách nhóm đầu nguồn và nhóm lồng nhau |
| `PeerReview/ResponseToReferee2.pdf` | Trả lời phản biện 2 | Phản biện chỉ ra: chỉ dự báo 1 ngày; lớp nhúng dùng tanh có thể bão hòa. Tác giả xác nhận tanh là mặc định của NeuralHydrology, chưa thử ReLU, dropout trong lớp nhúng bằng 0. Năm biến của thí nghiệm CrossDomain: `ERA5L_2m_temp_mean`, `ERA5L_2m_dp_temp_mean`, `ERA5L_surf_net_solar_rad_mean`, `MSWEP_RR`, `GLEAM_ETA` (ghép với 5 biến ECMWF). Ở quy mô quốc gia có thể thay tái phân tích bằng dữ liệu phân tích gần thời gian thực (INCA của GeoSphere Austria) | Thử lớp nhúng ReLU và dropout (Y14). Cặp bức xạ của tác giả là bức xạ thuần ERA5-Land với bức xạ tới ECMWF `ssrd` — hai đại lượng khác nhau; khi tự so sánh tái phân tích và dự báo cần ghi rõ điểm này |
| `PeerReview/RefereeReport_Round2.pdf` | Phản biện vòng 2 (ảnh chụp, đọc bằng mắt) | Phản biện đánh giá cao phát hiện lưu vực có độ lệch phân bố lớn nhất lại được lợi ít nhất từ mọi kỹ thuật; còn 3 ý kỹ thuật nhỏ | Dẫn chứng điểm mạnh của bài khi trình bày với GVHD |
| `PeerReview/ResponseToReferee_Round2.pdf` | Trả lời vòng 2 | Câu "trạng thái ô nhớ giữ tín hiệu mùa" chỉ áp dụng cho lưu vực núi cao, có tuyết; baseline chỉ dự báo dùng dự báo trước 1 ngày cho cả chuỗi 365 ngày; chọn MSWEP cho mưa vì nhiều đánh giá cho thấy tốt nhất | Hiểu đúng thiết kế đầu vào khi tái lập bước A |
| `PeerReview/ResponseToEditor_Round1.pdf` | Thư gửi biên tập, liệt kê thay đổi | Mục 2.3, 3.1.1 (khoảng cách Wasserstein theo lưu vực), 3.6 (tương quan ΔNSE với 33 thuộc tính), Phụ lục F (bảng thống kê) được thêm theo phản biện; sửa ΔNSE trên Hình 5 từ 0,3 thành 0,24 | Biết phần nào của bài là bổ sung sau phản biện |
| `PeerReview/ResponseToEditor_TechnicalCorrections.pdf` | Sửa kỹ thuật cuối | Sửa lỗi chữ, thêm tên nước vào Hình 1, giải thích ký hiệu ở Phụ lục C | Không có nội dung mới |

## 2. Căn cứ cho pipeline (`Evidence/`)

| Tệp | Nội dung | Rút ra, học được | Áp dụng cho đề tài |
|---|---|---|---|
| `Baste2025_HESS_LSTMPeakLimit.pdf` | LSTM 196 lưu vực Thụy Sĩ chạy với mưa thiết kế cực đoan, so với mô hình lai | LSTM không dự báo vượt giới hạn lý thuyết 73 mm/ngày dù dữ liệu huấn luyện có 183 mm/ngày; hệ số dòng chảy giảm khi mưa tăng; nguyên nhân chính là cổng LSTM chặn thông tin mới, không chỉ bão hòa trạng thái; tăng số nút ẩn và dữ liệu đa dạng giúp một phần | Lý do đánh giá riêng đỉnh lũ (FHV, mức lũ) và so Mamba với LSTM ở dải lưu lượng cao |
| `Gauch2025_HESS_MissingInputs.pdf` | Ba cách xử lý thiếu đầu vào khí tượng: thay giá trị, masked mean, attention | Masked mean tốt nhất với cách biệt nhỏ; attention học lại gần đúng masked mean nên chưa cần; thử trên CAMELS-US 531 lưu vực | Căn cứ bước B (masked mean, Y1) |
| `Klingler2021_ESSD_LamaHCE.pdf` | Bài dữ liệu LamaH-CE: 859 lưu vực, 9 nước, ~170.000 km², >60 thuộc tính, chuỗi ngày và giờ >35 năm | Có thuộc tính về tác động con người và chất lượng chuỗi (`gaps_pre`, `gaps_post` — khoảng trống trước và sau nội suy tuyến tính); có lưu vực trung gian để dựng mạng sông; kèm kết quả mô hình khái niệm COSERO | Trích dẫn dữ liệu; mục dataset theo khuôn GVHD; thuộc tính chất lượng dùng khi giải thích lưu vực kém |
| `Klotz2022_HESS_UncertaintyCMAL.pdf` | So bốn cách ước lượng bất định bằng học sâu (GMM, CMAL, UMAL, MC dropout) | CMAL tốt nhất về độ tin cậy, độ sắc và dự báo điểm; phân bố bất đối xứng hợp với lưu lượng; MC dropout kém nhất | Căn cứ bước F (đầu CMAL) |
| `Kratzert2019_HESS_NSEStarLoss.pdf` | Một LSTM/EA-LSTM cho 531 lưu vực CAMELS; đề xuất NSE* | NSE* (NSE trung bình theo lưu vực) cải thiện chủ yếu các lưu vực kém: EA-LSTM NSE trung bình 0,63 → 0,67, LSTM không thuộc tính tĩnh 0,23 → 0,39; trung vị gần như không đổi | Giữ NSE* như bài cơ sở; khi báo cáo nên kèm cả trung bình và phân vị thấp, không chỉ trung vị |
| `Kratzert2024_HESS_NeverTrainSingleBasin.pdf` | Bài quan điểm: không huấn luyện LSTM trên một lưu vực | Mô hình nhiều lưu vực luôn tốt hơn; cấu hình chuẩn: 256 nút ẩn, dropout 0,4, 30 epoch, lr 1e-3 → 5e-4 → 1e-4, chuỗi 365 ngày | Biện luận dùng mô hình chung 451 lưu vực; phản biện các bài một lưu vực trong khảo sát (Mục 5) |
| `Liu2024_JHydrol_TransformerLimit.pdf` | Transformer cơ bản so với LSTM trên CAMELS | Transformer cơ bản thua LSTM, nhất là dòng chảy cao; Transformer thêm lớp nhúng tích chập nhân quả (2 lớp Conv1D có kết nối tắt, nhân hướng về quá khứ) ngang LSTM (KGE 0,74 so với 0,73) nhưng dao động giữa các lần chạy lớn hơn; tác giả cho rằng kiến trúc chỉ còn cải thiện cỡ 0,02 KGE trên CAMELS | Transformer ở bước C phải có cải tiến mới công bằng; nhúng tích chập nhân quả là ứng viên biến thể Transformer (Y12) |
| `Martel2025_HESS_PeakOversampling.pdf` | Sáu cách cải thiện đỉnh lũ của LSTM, 88 lưu vực Quebec | Lấy mẫu nhiều đỉnh lũ là mô hình kém nhất về đỉnh; attention đa đầu ít lợi; đưa mô phỏng của mô hình vật lý làm đầu vào tốt nhất | Bằng chứng trái chiều cho Y7 |
| `Nearing2022_HESS_Autoregressive.pdf` | So tự hồi quy và đồng hóa dữ liệu khi đưa Q quan trắc vào LSTM | Đưa Q làm đầu vào chính xác và rẻ hơn đồng hóa biến phân; cần huấn luyện có che bớt Q để chịu thiếu dữ liệu | Căn cứ dùng Q quá khứ và Y1 |
| `Nearing2024_Nature_GlobalFloods.pdf` | Mô hình LSTM của Google dự báo lũ toàn cầu ở lưu vực không có trạm | Đánh giá bằng precision, recall, F1 theo chu kỳ lặp lại; ngưỡng tính riêng cho chuỗi mô hình và chuỗi quan trắc; một sự kiện được tính đúng nếu cả hai vượt ngưỡng trong vòng 2 ngày | Cách đánh giá mức lũ ở `03_Pipeline.md` Mục 6.2 nên theo khung này (Y10) |
| `Yang2025_HESS_DataIntegrationLSTM.pdf` | Đưa Q và tuyết quan trắc có độ trễ vào LSTM, 646 lưu vực miền Tây Hoa Kỳ | KGE trung vị 0,80 → 0,96 khi thêm Q trễ 1 ngày, vẫn 0,89 khi trễ 10 ngày; SWE chỉ có lợi ở bước tháng | Q trễ vài ngày vẫn hữu ích — căn cứ cho kịch bản mất Q, trễ Q (Y1, Y2) |
| `Yilmaz2008_WRR_FlowDurationSegments.pdf` | Chỉ số đặc trưng thủy văn theo đường duy trì lưu lượng | Đoạn cao: xác suất vượt 0–0,02; đoạn giữa 0,2–0,7; đoạn thấp 0,7–1,0; các chỉ số %BiasFHV, %BiasFMS, %BiasFLV | Định nghĩa FHV, FLV; cách chia mức ở `03_Pipeline.md` Mục 6.2 (thêm mức 0,02–0,2 nằm giữa hai đoạn của Yilmaz) |

## 3. Mamba, SSM và kiến trúc so sánh (`MambaSSM/`)

| Tệp | Nội dung | Rút ra, học được | Áp dụng cho đề tài |
|---|---|---|---|
| `Gu2022_NeurIPS_S4D.pdf` | S4D: SSM ma trận chéo | Khởi tạo quyết định hiệu năng; nhân chập tính bằng vài dòng mã; ngang S4 | Lý thuyết và cách cài S4D ở bước C |
| `Gu2024_COLM_Mamba.pdf` | Mamba: SSM chọn lọc, tham số phụ thuộc đầu vào, quét song song theo phần cứng | Tuyến tính theo độ dài chuỗi; mạnh với chuỗi dài | Lý thuyết lõi Mamba (`06_Theory/01_ArchitectureTheory.md`) |
| `Jing2026_ESWA_S4D-S5D.pdf` | S4D làm bộ mã hóa trong mô hình lai học tham số vi phân (dPL), CAMELS-US | NSE trung vị 0,742 (LSTM) → 0,756 (S4D) trên 531 lưu vực, lợi nhiều ở lưu vực tuyết, núi; biến thể S5D thêm Conv1D, LayerNorm, Softsign, bản tốt nhất đạt 0,763 (bảng 671 lưu vực) | Giả thuyết cho Y3: SSM hơn ở lưu vực tuyết; S5D là ứng viên biến thể (Y15). Lưu ý: kết quả trong khung lai, không phải mô hình thuần |
| `Nie2023_ICLR_PatchTST.pdf` | PatchTST: chia chuỗi thành đoạn (patch), mỗi kênh độc lập | Patch giữ ngữ nghĩa cục bộ, giảm chi phí attention, nhìn được chuỗi dài hơn | Ứng viên biến thể Transformer ở bước E |
| `ShamsEddin2025_NeurIPS_RiverMamba.pdf` | Mamba không gian – thời gian dự báo lưu lượng toàn cầu lưới 0,05°, tới 7 ngày, dùng ECMWF HRES | Mamba quét chuỗi không gian; hàm mất mát có trọng số theo chu kỳ lặp lại | Tham khảo (`06_Theory/02_RiverMamba.md`) |
| `Sheng2026_AISE_ResBiMambaPlus.pdf` | Mamba hai chiều + attention không gian – thời gian + ResNet, lưu lượng giờ sông Columbia | Quét hai chiều trong cửa sổ quá khứ; tới 48 giờ; chỉ dùng dữ liệu, chưa có ràng buộc vật lý | Căn cứ biến thể Mamba hai chiều ở bước E |
| `Wang2025_WRR_S4D-FT.pdf` | S4D-FT so với LSTM và SAC-SMA, 531 lưu vực CAMELS | S4D-FT hơn LSTM ở lưu vực tuyết và dòng chảy gián đoạn, kém ở lưu vực lũ nhanh, lưu lượng lớn — nhân chập toàn cục thiên về động lực chậm | Giả thuyết cho Y3 và đánh giá theo mức: SSM có thể yếu ở đỉnh lũ nhanh |
| `Wang2025_WRR_S4D-FT_Supplement.pdf` | Phụ lục S4D-FT | Siêu tham số: d_model 128, d_state 128, 6 lớp, dropout 0,12, min_dt 0,01, max_dt 0,1, 50 epoch, batch 128, lr 4e-4, lr riêng cho tham số SSM 4e-5; thứ tự tinh chỉnh: dung lượng → rời rạc hóa → tối ưu | Điểm xuất phát cấu hình S4D ở bước C |
| `Zhang2026_JHydrol_TemporalInductiveBiases.pdf` | So LSTM, Transformer, PatchTST, Mamba, DLinear dự báo theo giờ, 516 lưu vực CAMELS | Có Q quá khứ: DLinear tốt nhất ở 1 giờ, PatchTST tốt nhất khi tầm xa hơn; không có Q hoặc lưu vực ẩm, tuyết: LSTM và Mamba ổn định hơn; Mamba gần nhưng hơi kém LSTM | Bằng chứng không mặc định Mamba thắng; DLinear là mốc đơn giản đáng thêm (Y13) |

## 4. Công trình trích dẫn BiasCast (`Related/`)

| Tệp | Nội dung | Rút ra, học được | Áp dụng cho đề tài |
|---|---|---|---|
| `Taccari2026_arXiv_AIFL.pdf` | AIFL (ECMWF; bản arXiv của bài đăng *Journal of Hydrology* 678, 136064, 2026): LSTM toàn cầu 18.588 lưu vực, huấn luyện trước trên ERA5-Land rồi tinh chỉnh trên dự báo IFS | Học chuyển giao hai giai đoạn hơn huấn luyện một giai đoạn — ngược kết quả của BiasCast; ngưỡng lũ Gumbel bằng L-moments; khung ngưỡng kép (ngưỡng mô hình từ chuỗi mô phỏng, ngưỡng quan trắc từ chuỗi đo); precision cao, recall thấp | Thảo luận về học chuyển giao; khung ngưỡng kép cho `03_Pipeline.md` Mục 6.2 (Y10) |
| `AcunaEspinoza2026_EGUsphere_MF2LSTM.pdf` | MF²LSTM: LSTM đa tần số, masked mean, đồng hóa Q, dự báo giờ (preprint) | Q trễ (t−1, t−2, t−3) nhúng thành nhóm riêng rồi masked mean với khí tượng; che ngẫu nhiên khi huấn luyện với p_step = p_seq = 0,05 sau thử nghiệm; đề xuất chỉ số PNSE — NSE lấy Q quan sát cuối cùng làm mốc thay cho trung bình, vì NSE thường đánh giá quá cao mô hình có Q đầu vào | PNSE cho mọi cấu hình có Q (Y9); xác suất che Q ở Y1 nên thử cả 0,05 (Y11) |

## 5. Khảo sát (`Survey/`)

### 5.1. Mamba cho chuỗi thời gian tổng quát (`MambaTimeSeries/`)

| Tệp | Nội dung | Rút ra, học được | Áp dụng cho đề tài |
|---|---|---|---|
| `Li2024_JBigData_CMMamba.pdf` | Mamba hai chiều + trộn kênh, dữ liệu ETT, thời tiết | Tác giả tự nêu Mamba còn yếu với chuỗi ngắn | Biện luận chọn độ dài chuỗi 365/730 ngày ở bước D |
| `Wang2025_Neurocomputing_IsMambaEffective.pdf` | S-Mamba: Mamba hai chiều quét theo biến, FFN theo thời gian | Mamba ngang Transformer, chi phí thấp; lợi thế chủ yếu khi nhiều biến | Khảo sát Mục 1; ý quét theo biến để cho khóa luận |
| `Wu2026_TST_DTMamba.pdf` | DTMamba: khối Mamba đôi + residual, kênh độc lập | Lợi ở chuỗi ít chiều, dự báo dài | Khảo sát Mục 1 |
| `Xu2025_CIKM_SST.pdf` | SST: Mamba cho mẫu dài hạn độ phân giải thô, Transformer cho biến động ngắn hạn độ phân giải mịn | Xếp chồng Mamba và Transformer đơn giản bị nhiễu thông tin; tách theo thang thời gian thì tốt | Hướng biến thể lai ở khóa luận |

### 5.2. Dự báo lưu lượng ở châu Á, venue CNTT (`StreamflowAsia/`)

| Tệp | Nội dung | Rút ra, học được | Áp dụng cho đề tài |
|---|---|---|---|
| `Aatif2024_IEEEAccess_Chenab.pdf` | Curve number + LSTM, ML-GMDH, sông Chenab (Pakistan) | LSTM R² 0,91; một lưu vực | Khảo sát Mục 2 |
| `Chang2025_ESWA_Yalong.pdf` | Chọn nhân tố bằng XGBoost-SHAP, dự báo GPR/LSTM/SVM, hiệu chỉnh sai số EEMD-AR, trạm Lianghekou | NSE ~0,93; tác giả nêu rõ nguy cơ rò rỉ thông tin khi phân rã chuỗi và dùng dự báo cuốn chiếu để tránh | Khảo sát Mục 2; nhắc kiểm tra rò rỉ khi đọc bài có phân rã chuỗi |
| `Le2021_IEEEAccess_RedRiver.pdf` | FFNN, CNN, LSTM, GRU, StackedLSTM, BiLSTM, sông Hồng | LSTM, GRU một lớp đủ tốt; mô hình phức tạp hơn không hơn | Khảo sát Mục 2; căn cứ đưa GRU vào bước C |
| `Liu2022_IEEEAccess_Yangtze.pdf` | Transformer hai bộ mã hóa + VMD + ENSO, lưu lượng tháng sông Dương Tử | Cross-attention giữa hai chuỗi | Khảo sát Mục 2 |
| `Nguyen2023_IEEEAccess_Mekong.pdf` | MLP, CNN, LSTM, Transformer, sông Mekong | LSTM NSE ≥ 0,8; Transformer không nên dùng cho dự báo dài, nhất là mùa khô | Khảo sát Mục 2 |
| `Wei2024_IEEEAccess_Johor.pdf` | ANN, CNN, wavelet, mạng nơ-ron Bayes, sông Johor (Malaysia) | Wavelet giảm RMSE 6%; ước lượng bất định bằng MCMC và bootstrap | Khảo sát Mục 2 |

Các bài ở Mục 5.2 đều huấn luyện trên một lưu vực, trái với khuyến nghị của Kratzert và cs. (2024); đây là điểm khác biệt của đề tài khi trình bày (một mô hình chung 451 lưu vực).

## 6. Hướng khóa luận (`Thesis/`)

| Tệp | Nội dung | Rút ra, học được | Áp dụng cho đề tài |
|---|---|---|---|
| `AcunaEspinoza2025_HESS_MF-LSTM.pdf` | MF-LSTM: một ô LSTM xử lý nhiều tần số (dữ liệu cũ thô, gần đây mịn) | Giữ hiệu năng như MTS-LSTM, nhanh hơn 5 lần so với chỉ dùng dữ liệu giờ; lớp nhúng đưa số đầu vào khác nhau về cùng chiều | Hướng dữ liệu giờ ở khóa luận |
| `Kirschstein2024_ICML_RiverTopology.pdf` | GNN trên mạng trạm LamaH-CE theo giờ | Bỏ hết cạnh vẫn cho kết quả như có đồ thị; trọng số cạnh học được không theo quy luật; khó dự báo đỉnh nhọn | Lý do không làm đồ thị ở tiểu luận |
| `Mosaffa2026_HESS_LSTM-GNN.pdf` | LSTM sinh dòng chảy tại chỗ + GNN truyền lũ, 530 tiểu lưu vực Danube trên LamaH-CE | LSTM-GAT tốt nhất (NSE trung bình 0,61, RMSE giảm ~35%), lợi nhiều ở trạm hạ lưu có nhiều nhánh | Hướng đồ thị ở khóa luận |
| `Wang2025_npjNH_FloodGNNs.pdf` | Đồ thị reachability dày cho GNN trên LamaH-CE | Mạng sông dạng cây gây oversquashing; đồ thị reachability giảm khoảng cách điện trở, dự báo 24 giờ ngang EA-LSTM 14 giờ | Hướng đồ thị ở khóa luận |

## 7. Ý tưởng rút ra

| Mã | Ý tưởng | Nguồn |
|---|---|---|
| Y9 | Báo cáo thêm PNSE (NSE lấy Q quan sát ngày trước làm mốc) cho mọi cấu hình có Q đầu vào | MF²LSTM |
| Y10 | Đánh giá mức lũ theo khung ngưỡng kép (ngưỡng tính riêng trên chuỗi dự báo và chuỗi quan trắc), cho lệch thời gian vài ngày, ngưỡng Gumbel L-moments | Nearing 2024, AIFL |
| Y11 | Nhóm Q trễ nhiều ngày làm nhóm nhúng riêng trong masked mean; thử xác suất che Q 0,05 bên cạnh 0,1/0,12 | MF²LSTM, Gauch 2025 |
| Y12 | Biến thể Transformer có lớp nhúng tích chập nhân quả | Liu 2024 |
| Y13 | Thêm DLinear làm mốc đơn giản | Zhang 2026 |
| Y14 | Lớp nhúng ReLU + dropout thay tanh | Phản biện 2 của BiasCast |
| Y15 | Biến thể S5D (Conv1D, LayerNorm, Softsign) cho lõi S4D | Jing 2026 |

Chi tiết căn cứ, chi phí, vị trí trong pipeline: `Document/04_Ideas/01_Ideas.md` Mục 3.

## 8. Đánh giá danh mục PDF

44 tệp đều có vai trò: căn cứ cho pipeline, lý thuyết, khảo sát theo yêu cầu GVHD hoặc hướng khóa luận. Không có bài nào cần bỏ.

## 9. Công trình liên quan chưa có PDF

| Công trình | Nội dung | Rút ra | Áp dụng |
|---|---|---|---|
| *HydroDiffusion*, Water Resources Research, đăng 6/10/2026, DOI `10.1029/2025WR043158` (ghi chú đọc từ bản arXiv 2512.12183) | Mô hình khuếch tán xác suất với lõi SSM S4D-FT, khử nhiễu cả quỹ đạo 7 ngày một lần; 531 lưu vực CAMELS; dự báo vận hành bằng dự báo tái lập GEFSv12; không dùng Q quan trắc làm đầu vào | Lõi SSM nhỉnh hơn lõi LSTM (NSE ngày 0 trung vị 0,75 so với 0,73); công trình gần nhất về SSM trong dự báo có dự báo thời tiết thật | `03_Pipeline.md` Mục 1.3 |
| Saint-Fleur và cs., HESS 30:3497–3527 (2026), DOI `10.5194/hess-30-3497-2026` | Ba chiến lược đồng hóa Q (đưa Q quan trắc, thêm dự báo của mô hình nền, hậu xử lý sai số) cho dự báo 1–7 ngày; CAMELS-US, CAMELS-FR; dự báo hoàn hảo và dự báo tổ hợp ECMWF | Với LSTM, Q quan trắc giúp chủ yếu ở lưu vực LSTM vốn kém; tổ hợp của LSTM bị thiếu độ phân tán | `03_Pipeline.md` Mục 1.3; đối chiếu khi phân tích lợi ích của Q theo lưu vực (Mục 6.4) |
