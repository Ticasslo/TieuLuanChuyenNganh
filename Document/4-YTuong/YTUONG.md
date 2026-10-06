# Ý tưởng cho đề tài — đọc sâu BiasCast và tra cứu bổ sung

Tệp này tập hợp các ý tưởng mới tìm được khi đọc lại toàn văn BiasCast (Konold và cs., HESS 30:5067–5096, 2026) và tra cứu thêm, chưa có trong lộ trình ở `Document/1-KeHoach/KienTrucPipeline.md` Mục 8–9. Mỗi ý tưởng ghi căn cứ, chi phí, vị trí đề xuất trong lộ trình. Y1, Y2, Y3 đã đưa vào pipeline (`Document/1-KeHoach/KienTrucPipeline.md` Mục 8.2, 9.1 và sơ đồ `Diagrams/SoDoDT.drawio`, 27/9/2026; người dùng giao Claude chọn). Y6 đã đưa vào pipeline (`Document/1-KeHoach/KienTrucPipeline.md` Mục 7, 9.1; yêu cầu của GVHD, 6/10/2026).

## 1. Đợt tra 27/9/2026

### 1.1. Nguồn đã đọc

| Nguồn | Nội dung dùng được |
|---|---|
| BiasCast, toàn văn bản HESS (Mục 2.3, 3.1–3.7, 4, Phụ lục C–G) | Hướng tương lai tác giả tự nêu; phân tích theo lưu vực; tối ưu siêu tham số |
| Mã phân tích của tác giả `PaperResearchCode/BiasCast/Inspect_Experiments/code/` | `nse_delta_static_attributes.py` (ΔNSE theo lưu vực tương quan với 33 thuộc tính), `wasserstein_distance_static_attributes.py` |
| Cấu hình `Sequential_Forecast_LSTM_FCRA_without_q_Simple_Embedding/config.yml` (Zenodo 17292895) | Hindcast có `ERA5L_swe` (tuyết) và `ERA5L_volsw_123`, `ERA5L_volsw_4` (ẩm đất) |
| Gauch và cs., HESS 29:6221–6235 (2025) | Khi huấn luyện, bỏ ngẫu nhiên từng bước thời gian (xác suất 0,1) và cả chuỗi của một nguồn (xác suất 0,1); chỉ áp dụng cho khí tượng, không cho lưu lượng |
| Copernicus, trang Climate reanalysis | ERA5 và ERA5-Land cập nhật hằng ngày, trễ 5 ngày so với thời gian thực |
| Caravan MultiMet (Shalev & Kratzert, arXiv 2411.09459 v1) | IFS HRES 01/01/2016–30/09/2024; GraphCast 02/01/2016–21/12/2023 (khởi tạo từ HRES), chỉ 4 biến (nhiệt độ 2 m, mưa, gió u, v); dữ liệu theo UTC-0; CC BY 4.0 |
| Wang và cs., ICCS 2026 (LamaH-CE + EFAS, đồng hóa 3D-Var) | Không dùng Extended LamaH-CE, không trích BiasCast; không áp dụng |

### 1.2. Sự thật mới từ bài

- **Hướng tương lai tác giả nêu (Mục 4):** kiến trúc thích ứng dùng Q khi có và vẫn chạy được khi không có Q, tính tới hỏng cảm biến và áp dụng cách huấn luyện của Gauch và cs. (2025); kết hợp nhiều nguồn dự báo thời tiết; dự báo nhiều ngày (bài riêng sắp tới — chưa tìm thấy bản công bố).
- **Học chuyển giao để bỏ Q thất bại (Mục 3.4):** học có Q rồi tinh chỉnh không Q đều kém hơn mô hình không Q huấn luyện trực tiếp.
- **Độ trễ dữ liệu (Mục 3.7):** tác giả thừa nhận tái phân tích ở hindcast bị trễ khi vận hành; Q của Áo trên eHYD trễ khoảng 2 giờ.
- **Sequential bền hơn Encoder–Decoder theo lưu vực (Mục 3.6):** không có Q, Sequential Forecast LSTM cải thiện 435/451 lưu vực so với baseline chỉ dự báo, Encoder–Decoder 361/451 (làm kém 90); có Q: 445/451 so với làm kém 28. Tác giả quy cho trạng thái chạy liên tục từ hindcast sang forecast, không qua mạng handoff.
- **Nghịch lý núi cao (Mục 3.6):** lưu vực núi cao, nhiều tuyết có khác biệt tái phân tích – dự báo lớn nhất (Wasserstein) nhưng cải thiện ít nhất; lưu vực khô cải thiện nhiều nhất (tương quan ΔNSE với chỉ số khô hạn r = 0,31–0,50). Tác giả giải thích: trạng thái ô nhớ của LSTM giữ tín hiệu tuyết tan theo mùa — giả thuyết, chưa kiểm chứng trong bài.
- **Tối ưu siêu tham số (Phụ lục D, G):** Bayes 100 lần thử, không gian số chiều ẩn 64/128/256, dropout 0,1/0,2/0,3, nhiễu nhãn 0,001/0,01/0,1, batch 64/128/256; mỗi lần 30 epoch, cosine annealing; chỉ 1 hạt giống do giới hạn tính toán. Khoảng tinh chỉnh 64/128/256 của đề tài (Mục 9 bước C) khớp không gian này.
- **Ngoại lệ do con người (Mục 3.3):** lưu vực 758 dự báo cao hơn thực đo suốt kỳ test, nghi do lấy nước hoặc công trình điều tiết.

### 1.3. Ý tưởng

| # | Ý tưởng | Căn cứ | Mới so với bài | Chi phí | Đề xuất |
|---|---|---|---|---|---|
| Y1 | **Mô hình chịu mất Q:** huấn luyện có bỏ ngẫu nhiên Q (từng ngày và cả chuỗi) theo cách của Gauch và cs.; khi test chạy các kịch bản mất Q 1, 3, 7 ngày cuối và mất hẳn — một mô hình dùng được cho cả trạm có và không có Q | Hướng tương lai tác giả tự nêu; học chuyển giao của bài thất bại; Gauch và cs. chỉ thử với khí tượng; dữ liệu thiếu ít (Q test 0,23%, khí tượng gần như đủ — `Document/3-DuLieu/LamaHCE.md` Mục 6.1); phần mẫu train bị bộ lọc loại chủ yếu ở 27 trạm đo muộn và ngày thiếu Q, được lấy lại một phần bằng sửa bộ lọc ở B1; còn masked mean + bỏ ngẫu nhiên Q cho khả năng chịu mất dữ liệu khi vận hành | Chưa tìm thấy công trình làm với Q trong khung hindcast–forecast có dự báo thật (tra 27/9/2026, chưa tra hết) | Thấp: nằm trên cải tiến masked mean (bước B1); thêm 1 cấu hình × 3 hạt giống; các kịch bản chỉ là suy luận | Tiểu luận, gộp vào tầng 1 (bước B) |
| Y2 | **Kịch bản vận hành thật:** khi test, bỏ tái phân tích 5 ngày cuối của hindcast (ERA5 trễ 5 ngày), giữ Q ngày t−1 và dự báo HRES — đo NSE khi dữ liệu đến trễ như thực tế | Copernicus: ERA5-Land trễ 5 ngày; bài thừa nhận hạn chế độ trễ nhưng không đo | Bài giả định mọi dữ liệu có ngay | Rất thấp nếu đã có Y1 (cùng cơ chế mặt nạ): chỉ suy luận. Độ trễ của E-OBS, MSWEP, GLEAM chưa tra — kịch bản thận trọng coi mọi nguồn tái phân tích cùng trễ 5 ngày | Tiểu luận, bước đánh giá; dùng cho demo |
| Y3 | **Mamba thắng ở đâu:** ΔNSE theo lưu vực giữa Mamba và LSTM cải tiến (bản đồ nhiệt kiểu Hình 9, đếm số lưu vực tốt lên/kém đi), tương quan Spearman với 33 thuộc tính và khoảng cách Wasserstein; kiểm giả thuyết "trạng thái nhớ bù sai lệch ở lưu vực tuyết" của bài cho lõi SSM | Mục 3.6 của bài; mã phân tích của tác giả dùng lại được (đổi đường dẫn và mốc so sánh) | Bài chỉ phân tích các biến thể LSTM | Không huấn luyện thêm | Tiểu luận, bước đánh giá (C) |
| Y4 | **Dò trạng thái ẩn (probing):** hồi quy tuyến tính từ trạng thái ô nhớ LSTM và trạng thái SSM của Mamba sang lượng nước tuyết `ERA5L_swe` và ẩm đất `ERA5L_volsw_*` | Bài #28 trong `RESEARCHING.md` (Environ. Model. Softw. 2025, dò khái niệm thủy văn trong LSTM); giả thuyết của BiasCast Mục 3.6 | Chưa tìm thấy công trình dò trạng thái Mamba trong thủy văn (chưa tra hết) | Thấp (chỉ suy luận + hồi quy); hạn chế: tuyết và ẩm đất đang là biến đầu vào nên kết quả dò một phần hiển nhiên — muốn chặt phải huấn luyện thêm bản bỏ các biến này | Tùy chọn cho phần XAI (bước G); bản chặt để khóa luận |
| Y5 | **Nhiều nguồn dự báo (HRES + GraphCast):** kiểm giả thuyết của tác giả rằng ghép nhiều hệ dự báo giúp mô hình | BiasCast Mục 3.7, 4; Caravan MultiMet có GraphCast cho lưu vực LamaH | Tác giả chỉ nêu giả thuyết | Cao: GraphCast chỉ từ 2016, trong khi Q của LamaH-CE hết 2017 → trùng 2016–2017, không đủ huấn luyện theo cách chia của bài; GraphCast chỉ 4 biến; khác múi giờ (UTC-0) | Không cho tiểu luận; khóa luận chỉ ở dạng thử đổi nguồn dự báo khi test |

### 1.4. Căn cứ bổ sung cho thiết kế đã chốt

- Mamba nối hindcast và forecast thành một chuỗi liên tục (`KienTrucPipeline.md` Mục 3.2) đúng với phát hiện của bài: trạng thái chạy liên tục bền hơn mạng handoff (435 so với 361 lưu vực cải thiện).
- Q ngày t−1 dùng được trong vận hành: eHYD (Áo) công bố Q trễ khoảng 2 giờ — lập luận cho phần bảo vệ.
- Báo cáo nên kèm số lưu vực tốt lên/kém đi và trung vị, vì trung bình NSE bị vài lưu vực ngoại lệ (như 758) kéo lệch.

### 1.5. Chưa xác nhận

- Bài dự báo nhiều ngày của nhóm tác giả: chưa tìm thấy bản công bố (tra 27/9/2026).
- `Document/6-ChonBaiCoSo/CHECKPDF.md` ghi dữ liệu dự báo của OpenHydroNet phủ HRES 2012–2020 và GraphCast 2015–2022, khác mốc 2016 trong bài Caravan MultiMet v1 — có thể do bản dữ liệu mới hơn; cần mở tệp zarr để kiểm nếu dùng Y5.
- Độ trễ công bố của E-OBS, MSWEP, GLEAM.

## 2. Đợt 6/10/2026 — từ góp ý của GVHD (họp 27/9/2026)

Nguồn: `Document/2-HopGVHD/HopNhom_2026-09-27.md` Mục 2; tra bổ sung Martel và cs. (2025), *Hydrology and Earth System Sciences* 29:4951–4968, DOI `10.5194/hess-29-4951-2025`.

### 2.1. Ý tưởng

| # | Ý tưởng | Căn cứ | Mới so với bài | Chi phí | Đề xuất |
|---|---|---|---|---|---|
| Y6 | **Đánh giá theo mức lưu lượng:** mô hình vẫn dự báo Q; sau khi dự báo, chia cả Q quan trắc và Q dự báo thành các mức (thấp, trung bình, cao, lũ) theo ngưỡng riêng từng lưu vực, rồi tính độ chính xác từng mức, ma trận nhầm lẫn, tỉ lệ phát hiện lũ và tỉ lệ báo động nhầm; kèm sai lệch đoạn cao (2% lớn nhất) và đoạn thấp (30% nhỏ nhất) của đường duy trì lưu lượng | Yêu cầu của GVHD; Yilmaz, Gupta, Wagener (2008, *Water Resources Research*) cho cách chia đoạn đường duy trì; FHV, FLV có sẵn trong NeuralHydrology | BiasCast chỉ báo cáo NSE, KGE | Không huấn luyện thêm | Tiểu luận, bước đánh giá — đã đưa vào pipeline |
| Y7 | **Lấy mẫu nhiều ngày lưu lượng cao khi huấn luyện:** bộ nạp chọn mẫu có ngày t thuộc mức cao/lũ với xác suất lớn hơn | GVHD gợi ý; Martel và cs. (2025, 88 lưu vực Québec, LSTM, kiểm tra lưu vực không đo) thử lấy mẫu nhiều đỉnh và kết quả đỉnh **kém đi**; thêm dữ liệu từ nhiều lưu vực khác thì tốt lên | Chưa thấy công trình thử trong khung hindcast–forecast có Q quan trắc (chưa tra hết) | 1 cấu hình × 3 hạt giống; phải xong Y6 để biết mức nào yếu | Chưa chốt. Bằng chứng trái chiều nên chỉ nên làm như thí nghiệm kiểm chứng, so với LSTM cải tiến trên cả NSE và chỉ số theo mức |
| Y8 | **Đóng băng có chọn lọc theo giải thích mô hình:** tìm lớp hoặc nút ảnh hưởng nhiều tới mức đang yếu, chỉ huấn luyện tiếp các lớp đó | GVHD gợi ý, thầy nói chưa từng làm với LSTM | Chưa tra công trình trong thủy văn | Cao, chưa có quy trình chuẩn cho mô hình hồi quy chuỗi | Không cho tiểu luận; khóa luận nếu Y6, Y7 cho thấy một mức yếu rõ |
