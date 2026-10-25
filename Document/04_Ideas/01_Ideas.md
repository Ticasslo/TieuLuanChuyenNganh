# Ý tưởng cho đề tài — đọc sâu BiasCast và tra cứu bổ sung

Tệp này tập hợp các ý tưởng mới tìm được khi đọc lại toàn văn BiasCast (Konold và cs., HESS 30:5067–5096, 2026) và tra cứu thêm, kèm căn cứ, chi phí, vị trí đề xuất trong lộ trình. Trạng thái từng ý tưởng trong pipeline (`Document/01_Plan/03_Pipeline.md`): Mục 4 của tệp này.

## 1. Ý tưởng từ toàn văn BiasCast

### 1.1. Nguồn đã đọc

| Nguồn | Nội dung dùng được |
|---|---|
| BiasCast, toàn văn bản HESS (Mục 2.3, 3.1–3.7, 4, Phụ lục C–G) | Hướng tương lai tác giả tự nêu; phân tích theo lưu vực; tối ưu siêu tham số |
| Mã phân tích của tác giả `PaperResearch/PaperResearchCode/BiasCast/Inspect_Experiments/code/` | `nse_delta_static_attributes.py` (ΔNSE theo lưu vực tương quan với 33 thuộc tính), `wasserstein_distance_static_attributes.py` |
| Cấu hình `Sequential_Forecast_LSTM_FCRA_without_q_Simple_Embedding/config.yml` (Zenodo 17292895) | Hindcast có `ERA5L_swe` (tuyết) và `ERA5L_volsw_123`, `ERA5L_volsw_4` (ẩm đất) |
| Gauch và cs., HESS 29:6221–6235 (2025) | Khi huấn luyện, bỏ ngẫu nhiên từng bước thời gian (xác suất 0,1) và cả chuỗi của một nguồn (xác suất 0,12); chỉ áp dụng cho khí tượng, không cho lưu lượng |
| Copernicus, trang Climate reanalysis | ERA5 và ERA5-Land cập nhật hằng ngày, trễ 5 ngày so với thời gian thực |
| Caravan MultiMet (Shalev & Kratzert, arXiv 2411.09459 v1) | IFS HRES 01/01/2016–30/09/2024; GraphCast 02/01/2016–21/12/2023 (khởi tạo từ HRES), chỉ 4 biến (nhiệt độ 2 m, mưa, gió u, v); dữ liệu theo UTC-0; CC BY 4.0 |
| Wang và cs., ICCS 2026 (LamaH-CE + EFAS, đồng hóa 3D-Var) | Không dùng Extended LamaH-CE, không trích BiasCast; không áp dụng |

### 1.2. Sự thật mới từ bài

- **Hướng tương lai tác giả nêu (Mục 4):** kiến trúc thích ứng dùng Q khi có và vẫn chạy được khi không có Q, tính tới hỏng cảm biến và áp dụng cách huấn luyện của Gauch và cs. (2025); kết hợp nhiều nguồn dự báo thời tiết; dự báo nhiều ngày (bài riêng sắp tới — chưa tìm thấy bản công bố).
- **Học chuyển giao để bỏ Q thất bại (Mục 3.4):** học có Q rồi tinh chỉnh không Q đều kém hơn mô hình không Q huấn luyện trực tiếp.
- **Độ trễ dữ liệu (Mục 3.7):** tác giả thừa nhận tái phân tích ở hindcast bị trễ khi vận hành; Q của Áo trên eHYD trễ khoảng 2 giờ (Mục 3.4).
- **Sequential bền hơn Encoder–Decoder theo lưu vực (Mục 3.6):** không có Q, Sequential Forecast LSTM cải thiện 435/451 lưu vực so với baseline chỉ dự báo, Encoder–Decoder 361/451 (làm kém 90); có Q: 445/451 so với làm kém 28. Tác giả quy cho trạng thái chạy liên tục từ hindcast sang forecast, không qua mạng handoff.
- **Nghịch lý núi cao (Mục 3.6):** lưu vực núi cao, nhiều tuyết có khác biệt tái phân tích – dự báo lớn nhất (Wasserstein) nhưng cải thiện ít nhất; lưu vực khô cải thiện nhiều nhất (tương quan ΔNSE với chỉ số khô hạn r = 0,31–0,50). Tác giả giải thích: trạng thái ô nhớ của LSTM giữ tín hiệu tuyết tan theo mùa — giả thuyết, chưa kiểm chứng trong bài.
- **Tối ưu siêu tham số (Phụ lục D, G):** Bayes 100 lần thử, không gian số chiều ẩn 64/128/256, dropout 0,1/0,2/0,3, nhiễu nhãn 0,001/0,01/0,1, batch 64/128/256; mỗi lần 30 epoch, cosine annealing; chỉ 1 hạt giống do giới hạn tính toán. Đề tài không lặp lại tối ưu Bayes mà tinh chỉnh cùng ngân sách cho mọi lõi (`03_Pipeline.md` Mục 5.2).
- **Ngoại lệ do con người (Mục 3.3):** lưu vực 758 dự báo cao hơn thực đo suốt kỳ test, nghi do lấy nước hoặc công trình điều tiết.

### 1.3. Ý tưởng

| # | Ý tưởng | Căn cứ | Mới so với bài | Chi phí | Đề xuất |
|---|---|---|---|---|---|
| Y1 | **Mô hình chịu mất Q:** huấn luyện có bỏ ngẫu nhiên Q (từng ngày và cả chuỗi) theo cách của Gauch và cs.; khi test chạy các kịch bản mất Q 1, 3, 7 ngày cuối và mất hẳn — một mô hình dùng được cho cả trạm có và không có Q | Hướng tương lai tác giả tự nêu; học chuyển giao của bài thất bại; Gauch và cs. chỉ thử với khí tượng; dữ liệu thiếu ít (Q test 0,23%, khí tượng gần như đủ — `Document/03_Data/01_LamaHCE.md` Mục 6.1); phần mẫu train bị bộ lọc loại chủ yếu ở 27 trạm đo muộn và ngày thiếu Q, được lấy lại một phần bằng sửa bộ lọc ở B2; còn masked mean + bỏ ngẫu nhiên Q cho khả năng chịu mất dữ liệu khi vận hành | Chưa tìm thấy công trình làm với Q trong khung hindcast–forecast có dự báo thật (tra 27/9/2026, chưa tra hết) | Thấp: nằm trên cải tiến masked mean (bước B2); thêm 1 cấu hình × 3 hạt giống; các kịch bản chỉ là suy luận | Tiểu luận, gộp vào tầng 1 (bước B) |
| Y2 | **Kịch bản vận hành thật:** khi test, bỏ tái phân tích 5 ngày cuối của hindcast (ERA5 trễ 5 ngày), giữ Q ngày t−1 và dự báo HRES — đo NSE khi dữ liệu đến trễ như thực tế | Copernicus: ERA5-Land trễ 5 ngày; bài thừa nhận hạn chế độ trễ nhưng không đo | Bài giả định mọi dữ liệu có ngay | Rất thấp nếu đã có Y1 (cùng cơ chế mặt nạ): chỉ suy luận. Độ trễ riêng của E-OBS, MSWEP, GLEAM và nguồn tra: `03_Pipeline.md` Mục 3.2 | Tiểu luận, bước đánh giá; dùng cho demo |
| Y3 | **Mamba thắng ở đâu:** ΔNSE theo lưu vực giữa Mamba và LSTM cải tiến (bản đồ nhiệt kiểu Hình 9, đếm số lưu vực tốt lên/kém đi), tương quan Spearman với 33 thuộc tính và khoảng cách Wasserstein; kiểm giả thuyết "trạng thái nhớ bù sai lệch ở lưu vực tuyết" của bài cho lõi SSM | Mục 3.6 của bài; mã phân tích của tác giả dùng lại được (đổi đường dẫn và mốc so sánh) | Bài chỉ phân tích các biến thể LSTM | Không huấn luyện thêm | Tiểu luận, bước đánh giá (C) |
| Y4 | **Dò trạng thái ẩn (probing):** hồi quy tuyến tính từ trạng thái ô nhớ LSTM và trạng thái SSM của Mamba sang lượng nước tuyết `ERA5L_swe` và ẩm đất `ERA5L_volsw_*` | Hu và cs., *Investigate the rainfall-runoff relationship and hydrological concepts inside LSTM*, Environmental Modelling & Software 192:106527 (2025), DOI `10.1016/j.envsoft.2025.106527`: dò lượng nước trong đất, tuyết trong trạng thái ô nhớ LSTM bằng hồi quy tuyến tính; giả thuyết của BiasCast Mục 3.6 | Chưa tìm thấy công trình dò trạng thái Mamba trong thủy văn (chưa tra hết) | Thấp (chỉ suy luận + hồi quy); hạn chế: tuyết và ẩm đất đang là biến đầu vào nên kết quả dò một phần hiển nhiên — muốn chặt phải huấn luyện thêm bản bỏ các biến này | Bước G (`03_Pipeline.md` Mục 6.7); bản chặt để khóa luận |
| Y5 | **Nhiều nguồn dự báo (HRES + GraphCast):** kiểm giả thuyết của tác giả rằng ghép nhiều hệ dự báo giúp mô hình | BiasCast Mục 3.7, 4; Caravan MultiMet có GraphCast cho lưu vực LamaH | Tác giả chỉ nêu giả thuyết | Cao: GraphCast chỉ từ 2016, trong khi Q của LamaH-CE hết 2017 → trùng 2016–2017, không đủ huấn luyện theo cách chia của bài; GraphCast chỉ 4 biến; khác múi giờ (UTC-0) | Không cho tiểu luận; khóa luận chỉ ở dạng thử đổi nguồn dự báo khi test |

### 1.4. Căn cứ bổ sung cho thiết kế đã chốt

- Mamba nối hindcast và forecast thành một chuỗi liên tục (`03_Pipeline.md` Mục 4.1) đúng với phát hiện của bài: trạng thái chạy liên tục bền hơn mạng handoff (435 so với 361 lưu vực cải thiện).
- Q ngày t−1 dùng được trong vận hành: eHYD (Áo) công bố Q trễ khoảng 2 giờ — lập luận cho phần bảo vệ.
- Báo cáo nên kèm số lưu vực tốt lên/kém đi và trung vị, vì trung bình NSE bị vài lưu vực ngoại lệ (như 758) kéo lệch.

### 1.5. Chưa xác nhận

- Bài dự báo nhiều ngày của nhóm tác giả: chưa tìm thấy bản công bố (tra 27/9/2026).
- Tài liệu dữ liệu của OpenHydroNet (Google FloodHub) ghi dữ liệu dự báo phủ HRES 2012–2020 và GraphCast 2015–2022, khác mốc 2016 trong bài Caravan MultiMet v1 — có thể do bản dữ liệu mới hơn; cần mở tệp zarr để kiểm nếu dùng Y5.

## 2. Ý tưởng từ góp ý của GVHD (họp 27/9/2026)

Nguồn: `Document/02_Meetings/Meeting_2026-09-27.md` Mục 2; tra bổ sung Martel và cs. (2025), *Hydrology and Earth System Sciences* 29:4951–4968, DOI `10.5194/hess-29-4951-2025`.

### 2.1. Ý tưởng

| # | Ý tưởng | Căn cứ | Mới so với bài | Chi phí | Đề xuất |
|---|---|---|---|---|---|
| Y6 | **Đánh giá theo mức lưu lượng:** mô hình vẫn dự báo Q; sau khi dự báo, chia cả Q quan trắc và Q dự báo thành các mức (thấp, trung bình, cao, lũ) theo ngưỡng riêng từng lưu vực, rồi tính độ chính xác từng mức, ma trận nhầm lẫn, tỉ lệ phát hiện lũ và tỉ lệ báo động nhầm; kèm sai lệch đoạn cao (2% lớn nhất) và đoạn thấp (30% nhỏ nhất) của đường duy trì lưu lượng | Yêu cầu của GVHD; Yilmaz, Gupta, Wagener (2008, *Water Resources Research*) cho cách chia đoạn đường duy trì; FHV, FLV có sẵn trong NeuralHydrology | BiasCast chỉ báo cáo NSE, KGE | Không huấn luyện thêm | Tiểu luận, bước đánh giá — đã đưa vào pipeline |
| Y7 | **Lấy mẫu nhiều ngày lưu lượng cao khi huấn luyện:** bộ nạp chọn mẫu có ngày t thuộc mức cao/lũ với xác suất lớn hơn | GVHD gợi ý; Martel và cs. (2025, 88 lưu vực Québec, LSTM, kiểm tra lưu vực không đo) thử lấy mẫu nhiều đỉnh và kết quả đỉnh **kém đi**; thêm dữ liệu từ nhiều lưu vực khác thì tốt lên | Chưa thấy công trình thử trong khung hindcast–forecast có Q quan trắc (chưa tra hết) | 1 cấu hình × 3 hạt giống; phải xong Y6 để biết mức nào yếu | Thí nghiệm kiểm chứng ở bước GV (bằng chứng trái chiều), so với mô hình tốt nhất trên cả NSE và chỉ số theo mức |
| Y8 | **Đóng băng có chọn lọc theo giải thích mô hình:** tìm lớp hoặc nút ảnh hưởng nhiều tới mức đang yếu, chỉ huấn luyện tiếp các lớp đó | GVHD gợi ý, thầy nói chưa từng làm với LSTM | Chính BiasCast (Mục 2.2.5, 3.2) đã thử đóng băng cố định (không theo XAI): giữ LSTM và đầu ra học từ tái phân tích, chỉ huấn luyện tiếp lớp nhúng trên dữ liệu dự báo (TL EmbeddingNet) — NSE trung vị 0,44, hơn huấn luyện tiếp toàn bộ nhưng kém xa Sequential Forecast LSTM (0,63), chỉ 275/451 lưu vực tốt lên. Chưa tra công trình chọn lớp đóng băng theo XAI trong thủy văn | Cao, chưa có quy trình chuẩn cho mô hình hồi quy chuỗi | Không cho tiểu luận; khóa luận nếu Y6, Y7 cho thấy một mức yếu rõ |

## 3. Ý tưởng từ đọc lại toàn bộ tài liệu tham khảo

Nguồn: 44 tệp trong `PaperResearch/PaperResearchPDF/`, ghi chú từng bài ở `Document/05_Survey/03_PaperNotes.md`.

### 3.1. Ý tưởng

| # | Ý tưởng | Căn cứ | Mới so với bài | Chi phí | Đề xuất |
|---|---|---|---|---|---|
| Y9 | **Chỉ số PNSE:** NSE lấy Q quan sát ngày t−1 làm mốc thay cho trung bình; báo cáo cho mọi cấu hình có Q đầu vào, bên cạnh NSE | Hệ số persistence — Kitanidis & Bras, WRR 16:1034–1044 (1980). Acuña Espinoza và cs. (MF²LSTM, preprint EGUsphere 2026): NSE đánh giá quá cao mô hình có Q đầu vào vì Q đã đặt dự báo vào đúng mức; PNSE là mốc chặt hơn ở tầm ngắn. Persistence của đề tài đã tính sẵn | BiasCast chỉ báo cáo NSE, KGE | Không huấn luyện thêm; tính từ dự báo đã có | Đánh giá (cùng chỗ với persistence, bước A và đánh giá chính) |
| Y10 | **Khung ngưỡng kép cho đánh giá mức lũ:** ngưỡng mức lũ tính riêng trên chuỗi dự báo và chuỗi quan trắc; một sự kiện tính là đúng nếu cả hai vượt ngưỡng trong cùng cửa sổ thời gian (Nearing dùng 2 ngày); ngưỡng chu kỳ lặp lại fit Gumbel bằng L-moments | Nearing và cs. (Nature 2024); Taccari và cs. (AIFL, Journal of Hydrology 678, 2026) | BiasCast không đánh giá theo sự kiện | Không huấn luyện thêm | Bổ sung vào `03_Pipeline.md` Mục 6.2 (đánh giá theo mức lưu lượng) |
| Y11 | **Nhóm Q riêng trong masked mean và xác suất che Q:** nhúng Q trễ (t−1, t−2, t−3) thành nhóm riêng, gộp với các nhóm khí tượng bằng masked mean; thử xác suất che 0,05 bên cạnh 0,1/0,12 | MF²LSTM chọn p_step = p_seq = 0,05 sau thử nghiệm để cân bằng giữa có Q và mất Q; Gauch và cs. (2025) dùng 0,1/0,12 cho khí tượng | Bài đưa qmean thẳng vào đầu vào hindcast | +1–2 lần huấn luyện ở bước B | Gộp vào Y1 ở bước B |
| Y12 | **Transformer có lớp nhúng tích chập nhân quả:** hai lớp Conv1D có kết nối tắt, nhân chỉ nhìn về quá khứ, trước lớp attention | Liu và cs. (J. Hydrol. 2024): Transformer cơ bản thua LSTM, bản có nhúng tích chập ngang LSTM trên CAMELS | Bài không có Transformer | +1 lần huấn luyện | Ứng viên biến thể Transformer ở bước E, thay hoặc bên cạnh PatchTST |
| Y13 | **Mốc DLinear:** mô hình tuyến tính đơn giản trên cửa sổ quá khứ | Zhang và cs. (J. Hydrol. 2026): khi có Q quá khứ, DLinear tốt nhất ở tầm 1 giờ | Bài không có mốc tuyến tính | +1 lần huấn luyện, rất nhẹ | Bước A3, làm mốc giữa persistence và học sâu |
| Y14 | **Lớp nhúng ReLU + dropout:** thay tanh trong lớp nhúng, thêm dropout | Phản biện 2 của BiasCast: tanh trong lớp nhúng cộng với tanh trong LSTM có thể bão hòa; tác giả xác nhận chưa thử, dropout lớp nhúng bằng 0; Baste và cs. (2025) chỉ ra giới hạn đỉnh của LSTM | Tác giả nêu là hướng tương lai | +1–2 lần huấn luyện ở bước B | Bước B, nếu còn ngân sách |
| Y15 | **Biến thể S5D cho lõi S4D:** thêm Conv1D (thiên hướng cục bộ), LayerNorm, Softsign | Jing và cs. (ESWA 2026), khung lai học tham số trên CAMELS-US: S4D 0,756 so với LSTM 0,742 (531 lưu vực); biến thể S5D tốt nhất 0,763 (bảng 671 lưu vực) | Bài không có SSM | +1 lần huấn luyện | Ứng viên biến thể ở bước E nếu S4D tốt hơn Mamba ở bước C |
| Y16 | **FiLM cho thuộc tính lưu vực:** thuộc tính tĩnh sinh hệ số γ, β để điều chỉnh đặc trưng ẩn theo dạng γ·h + β thay cho nối thuộc tính vào đầu vào mỗi bước | GVHD giới thiệu ở buổi họp 27/9/2026 (biên bản Mục 3.2); FiLM — Perez và cs., AAAI 2018, arXiv 1709.07871. Bằng chứng gần nhất trong thủy văn: EA-LSTM dùng thuộc tính tĩnh điều khiển cổng vào, NSE trung vị tổ hợp 0,742 so với 0,758 của LSTM nối thuộc tính (Kratzert và cs., HESS 2019, Bảng 3) — điều chỉnh bằng thuộc tính tĩnh chưa cho thấy lợi thế | BiasCast nối nhúng thuộc tính tĩnh vào mỗi bước | +3 lần huấn luyện | Chưa đưa vào; ứng viên bước B nếu còn ngân sách |
| Y17 | **Dự báo 1–7 ngày bằng dữ liệu sẵn có:** dự báo qmax ngày t đến t+6; so ba cách cấp đầu vào pha dự báo: chỉ quá khứ, lai (ngày đầu dùng ECMWF HRES thật, các ngày sau để trống xử lý bằng masked mean), dự báo hoàn hảo bằng tái phân tích làm cận trên | Phản biện 2 của BiasCast chỉ ra hạn chế chỉ dự báo 1 ngày; tác giả nêu dự báo nhiều ngày là nghiên cứu sắp tới (BiasCast Mục 4); Extended LamaH-CE chỉ có ECMWF hạn 1 ngày (8 cột, một giá trị mỗi ngày); thư viện có sẵn `forecast_seq_length` | Bài chỉ dự báo 1 ngày | +18 lần huấn luyện | Bước H |

### 3.2. Giả thuyết cho phân tích theo lưu vực (Y3)

Ba nguồn độc lập cùng chỉ ra SSM có lợi ở lưu vực có động lực chậm (tuyết, dòng chảy gián đoạn) và kém ở lưu vực lũ nhanh, lưu lượng lớn: Wang và cs. (WRR 2025, S4D-FT), Jing và cs. (ESWA 2026, S4D), Zhang và cs. (J. Hydrol. 2026, Mamba ổn định ở lưu vực ẩm và tuyết). Khi làm Y3, kiểm tra giả thuyết này bằng tương quan ΔNSE Mamba–LSTM với tỷ lệ tuyết, độ cao, chỉ số khô hạn. Tách thêm lưu vực đầu nguồn và lưu vực lồng nhau (27,5% trong 451 lưu vực theo thư trả lời phản biện 1 của BiasCast).

## 4. Trạng thái các ý tưởng trong pipeline

| Ý tưởng | Trạng thái | Vị trí trong `03_Pipeline.md` |
|---|---|---|
| Y1, Y11 | Đưa vào | Bước B2–B3 (nhóm Q riêng, che dữ liệu hai mức xác suất); kịch bản S1, S2 |
| Y2 | Đưa vào, thiết kế lại theo độ trễ riêng từng nguồn | Mục 6.3, kịch bản S3, S4 |
| Y3 | Đưa vào | Mục 6.4 |
| Y4 | Đưa vào | Mục 6.7, bước G |
| Y5 | Không cho tiểu luận | Mục 11 |
| Y6 | Đưa vào | Mục 6.2 |
| Y7 | Đưa vào dạng thí nghiệm kiểm chứng | Bước GV |
| Y8 | Chưa đưa vào | Cần kết quả giải thích mô hình trước |
| Y9 | Đưa vào | Mục 6.1, bước A3 |
| Y10 | Đưa vào; ngưỡng kép chỉ là phân tích phụ | Mục 6.2 |
| Y12, Y15 | Dự phòng | Bước E |
| Y13 | Đưa vào | Bước A3 |
| Y14 | Chưa đưa vào; mã cho thấy lớp nhúng đơn giản (16 chiều) chỉ là một lớp tuyến tính, tanh chỉ có ở lớp nhúng phức 3 lớp, nên Y14 chỉ liên quan khi dùng nhúng phức | Ứng viên bước B nếu còn ngân sách |
| Y16 | Chưa đưa vào; bằng chứng gần nhất (EA-LSTM) không hơn cách nối thuộc tính | Ứng viên bước B nếu còn ngân sách |
| Y17 | Đưa vào | Bước H, Mục 4.4, 6.6 |
| Y18, Y19, Y20 | Ứng viên khóa luận (Mục 5 của tệp này) | Mục 11 |
| Y21 | Chưa đưa vào; mốc phụ nếu còn thời gian | Bước A3 (nếu đưa vào) |
| Y22 | Chưa đưa vào; bước C cần cùng cách khởi tạo cho mọi lõi | Mục 8 của tệp này |

## 5. Ứng viên cho khóa luận

Nguồn: tra cứu tới ngày 8/10/2026. Chọn hướng sau khi có kết quả tiểu luận.

| # | Ý tưởng | Căn cứ | Mới so với tiểu luận | Chi phí | Đề xuất |
|---|---|---|---|---|---|
| Y18 | **Đồ thị mạng sông + lõi tốt nhất:** lõi thời gian (Mamba hoặc LSTM) cho từng lưu vực, GAT truyền thông tin theo quan hệ thượng – hạ nguồn giữa các lưu vực lồng nhau | Đồ thị thô không cải thiện (Kirschstein & Sun, ICML 2024); GAT cải thiện trên LamaH-CE (Mosaffa và cs., HESS 2026); đồ thị reachability (Wang và cs., npj Natural Hazards 2025); 27,5% trong 451 lưu vực là lưu vực lồng nhau (thư trả lời phản biện 1 của BiasCast); chưa tìm thấy công trình kết hợp GNN với Mamba cho dòng chảy (`Document/05_Survey/01_Datasets.md` Mục 7.4, tra lại 8/10/2026, chưa tra hết) | Thêm chiều không gian; tiểu luận coi mỗi lưu vực độc lập | Cao: bộ nạp theo đồ thị, `NEXTDOWNID` có sẵn | Khóa luận |
| Y19 | **Tiền huấn luyện trên tái phân tích dài rồi tinh chỉnh trên dự báo:** dùng tái phân tích 1981–2002 (có trong Extended LamaH-CE, trước khi có ECMWF) để tiền huấn luyện, rồi tinh chỉnh trên giai đoạn có dự báo; có thể kết hợp đóng băng có chọn lọc (Y8) | AIFL (Taccari và cs., Journal of Hydrology 678:136064, 2026): tiền huấn luyện ERA5-Land 1980–2019 rồi tinh chỉnh trên IFS tốt hơn chỉ dùng IFS; BiasCast (Mục 3.2) thấy học chuyển giao kém Sequential Forecast LSTM (0,44 so với 0,63), nhưng chỉ dùng kỳ 2003–2009 | Giải thích mâu thuẫn giữa hai bài trên cùng một bộ dữ liệu | Thấp về dữ liệu (đã có sẵn); thêm vài lần huấn luyện | Khóa luận; nếu tiểu luận còn ngân sách có thể thử sớm |
| Y20 | **So với khung mã nguồn mở của Google:** dùng mô hình dự báo lũ Google công bố mã (3/6/2026, Apache 2.0; dữ liệu Caravan; đầu vào GraphCast, IFS, IMERG, CPC; cho phép huấn luyện, tinh chỉnh trên lưu vực riêng) làm mốc mạnh | Google Research blog *Open sourcing Google's hydrology framework* (3/6/2026) | Mốc vận hành của Google thay vì chỉ mốc của BiasCast | Trung bình đến cao; chưa kiểm có trọng số huấn luyện sẵn không và Caravan có chứa các lưu vực LamaH-CE không (nếu có thì mô hình Google đã thấy dữ liệu kỳ test) | Khóa luận, tùy kết quả kiểm |

## 6. Mô hình mới nổi và mô hình nền tảng

Nguồn: tra cứu tới ngày 17/10/2026.

**Mô hình nền tảng cho chuỗi thời gian** (time series foundation model): mô hình lớn (hàng trăm triệu tham số) được huấn luyện sẵn một lần trên rất nhiều chuỗi thời gian thuộc nhiều lĩnh vực (điện, bán lẻ, giao thông, thời tiết, cả chuỗi tổng hợp), rồi đem dùng cho chuỗi mới **không cần huấn luyện lại** (zero-shot): đưa vào đoạn quá khứ, mô hình trả ra dự báo. Có thể tinh chỉnh thêm trên dữ liệu riêng. Khác mô hình của đề tài ở chỗ mô hình của đề tài (khoảng 85 nghìn tham số) được huấn luyện riêng trên 451 lưu vực.

| # | Ý tưởng | Căn cứ | Mới so với bài | Chi phí | Đề xuất |
|---|---|---|---|---|---|
| Y21 | **Mốc mô hình nền tảng chạy không huấn luyện:** chạy TimesFM-3 hoặc Chronos-2 trên kỳ test, đầu vào là `qmean` quá khứ và các biến khí tượng làm biến phụ (covariate), ghi NSE cạnh persistence và DLinear; nếu còn thời gian, thêm bản tinh chỉnh | TimesFM-3 (Google Research blog, 31/8/2026): 330 triệu tham số, nhận nhiều chuỗi và biến phụ, các bản trước chỉ nhận một chuỗi. Chronos-2 (Amazon; bản thảo *A foundation model for multivariate time series forecasting*, Hình 7, nơi đăng chưa xác nhận): trên 1 582 lưu vực CAMELS-DE, không huấn luyện, có biến khí tượng đạt NSE trung vị 0,68, chỉ dùng chuỗi lưu lượng đạt 0,15, nhóm LSTM chuyên dụng đạt 0,90; tinh chỉnh thì ngang LSTM. Đó là bài toán mô phỏng, không phải dự báo với ECMWF HRES; chưa tìm thấy kết quả của TimesFM trên lưu lượng | BiasCast không có mốc mô hình nền tảng; trả lời câu hỏi "vì sao không dùng mô hình nền tảng" | Không huấn luyện (chỉ chạy dự báo, GPU Kaggle); thời gian chạy cho 451 lưu vực × 1 461 ngày chưa ước tính | Mốc phụ ở bước A3 nếu còn thời gian; không đặt vào so sánh cùng tham số ở bước C (330 triệu so với 85 nghìn tham số) và không dùng cho demo chạy CPU |

**Đã xem, không áp dụng — Jev (TypeSafe AI).** Jev là mô hình "System One" do TypeSafe AI công bố ngày 15/9/2026, đang ở giai đoạn truy cập sớm qua API trả phí. Đầu vào là trạng thái dạng chữ hoặc dữ liệu có cấu trúc; đầu ra là **quyết định có kiểu định sẵn kèm xác suất** (phân loại, chọn nhánh, chấm điểm, tối đa 255 lựa chọn), không sinh chữ, trả lời trong 70–500 ms (TypeSafe AI blog *Introducing System One Models & Jev*, 15/9/2026). Không áp dụng cho đề tài vì: (1) Jev không dự báo giá trị số theo chuỗi thời gian, còn bài toán là hồi quy qmax từ chuỗi khí tượng và lưu lượng; (2) đây là mô hình đóng chạy qua API, cùng nhóm với mô hình ngôn ngữ lớn — đề tài không dùng mô hình ngôn ngữ lớn theo góp ý của GVHD (buổi họp 27/9/2026); (3) chỗ cần "ra quyết định" duy nhất là đổi dự báo Q thành mức cảnh báo, việc này đã làm bằng ngưỡng chu kỳ lặp lại (`01_Plan/03_Pipeline.md` Mục 6.2), minh bạch và kiểm được hơn; (4) chưa có đánh giá độc lập, số liệu hiện có do chính công ty công bố.

## 7. Công trình mới liên quan

Nguồn: tra cứu tới ngày 18/10/2026. Không có ý tưởng mới cần đưa vào pipeline tiểu luận; các mục dưới bổ sung căn cứ cho khóa luận và cho phần tính mới.

| Công trình | Nội dung chính | Liên quan tới đề tài |
|---|---|---|
| Yu, Tolson, Lin, *Exploring xLSTM for large-sample hourly rainfall–runoff modelling*, Environmental Modelling & Software 207:107193 (trực tuyến 6/10/2026), DOI `10.1016/j.envsoft.2026.107193` | MF-xLSTM so với sMTS-LSTM, MF-LSTM trên 430 lưu vực Canada, theo giờ (2011–2023). Kiểm tra theo thời gian (344 lưu vực): NSE trung vị 0,835 so với 0,842 của MF-LSTM — ngang hoặc hơi kém; kiểm tra lưu vực không có trạm (86 lưu vực): 0,758 so với 0,743 — nhỉnh hơn. Siêu tham số chung (256 chiều ẩn, batch 128, 30 epoch) tinh chỉnh trên sMTS-LSTM rồi áp cho cả ba mô hình, không cân số tham số | Căn cứ cho hướng xLSTM ở khóa luận (`03_Pipeline.md` Mục 11). Cách so sánh của bài (siêu tham số tinh chỉnh cho LSTM rồi áp cho mô hình khác, không cùng tham số) khác giao thức bước C của đề tài (cùng tham số, cùng ngân sách tinh chỉnh cho mọi lõi) — có thể nêu khi bàn tính mới |
| Yang, Ji, Lonzarich, Song, Shen, *Diffusion-Based Probabilistic Modeling for Hourly Streamflow Prediction and Assimilation*, Water Resources Research 2026, DOI `10.1029/2025WR042720` (bản arXiv 2510.08488, đọc toàn văn) | h-Diffusion: mô hình khuếch tán khử nhiễu, mạng khử nhiễu là **MTS-LSTM** (nhánh LSTM ngày + nhánh LSTM giờ), điều kiện là khí tượng ngày và giờ, thuộc tính lưu vực và lưu lượng ngày (quan trắc hoặc mô phỏng) — tức là chia nhỏ lưu lượng ngày thành giờ, không phải dự báo bằng dự báo thời tiết. CAMELS theo giờ, 516 lưu vực, huấn luyện 1990–2003. NSE trung vị (một nguồn khí tượng): MTS-LSTM 0,763, MF-LSTM 0,756, h-Diffusion 0,774; thêm đồng hóa 5 giờ quan trắc gần nhất bằng kỹ thuật vẽ bù (RePaint) lên 0,827. Với nhiều nguồn khí tượng: 0,812 / 0,805 / 0,804 / 0,844; ở lũ hiếm (top 1%) h-Diffusion kém MTS-LSTM (CRPSS −0,10). Mã: bài hứa công bố | Thêm bằng chứng rằng **đưa quan trắc lưu lượng gần nhất vào** tăng mạnh kỹ năng (+0,04 tới +0,05 NSE), cùng chiều kết quả `qmean` của BiasCast (0,63 → 0,71) và nhóm Q ở bước B của đề tài. Lõi vẫn là LSTM. Căn cứ cho hướng dự báo xác suất, dữ liệu giờ ở khóa luận |
| Zhang, Yue, Basirifard, Cao, Yang, *A Mamba-type of deep state space model for reservoir release simulation with a large-scale verification over 441 dams across CONUS*, Journal of Hydrology 662:134145, 2025, DOI `10.1016/j.jhydrol.2025.134145` (đọc tóm tắt; toàn văn trả phí) | Dù tên bài nói "Mamba-type", mô hình thực ra là **S4D** (SSM bất biến thời gian, không có cơ chế chọn lọc của Mamba). Mô phỏng lượng xả của 441 hồ chứa Mỹ; S4D tốt hơn LSTM và XGBoost ở phần lớn hồ chứa; có SHAP. Số liệu chi tiết chưa đọc | Khác bài toán (hồ chứa có điều tiết, không phải dòng chảy tự nhiên), nhưng thêm một bằng chứng S4D hơn LSTM trong thủy văn, cùng chiều S4D-FT (Wang và cs., WRR 2025) — củng cố lý do giữ S4D ở bước C để tách "SSM" khỏi "cơ chế chọn lọc của Mamba". Cũng là ví dụ về việc gọi S4D là "Mamba" — báo cáo của đề tài phải phân biệt rõ hai tên |
| Nearing, Daňhelka, bài blog EGU Hydrological Sciences *Why We Open-Sourced Google's Hydrology Framework* (8/10/2026) | OpenHydroNet là bản fork sửa nhiều của NeuralHydrology, hỗ trợ dự báo thật với dữ liệu dự báo thời tiết, có trọng số tiền huấn luyện trên Caravan MultiMet, khuyến nghị tinh chỉnh trên dữ liệu địa phương khi vận hành; mã `github.com/google-research/flood-forecasting` | Bổ sung cho Y20 (Mục 5): mô hình của Google xây trên cùng thư viện với đề tài, nên dùng làm mốc ở khóa luận ít tốn công chuyển đổi hơn dự kiến |

## 8. Ý tưởng từ góp ý của GVHD (họp 11/10/2026)

Nguồn: `Document/02_Meetings/Meeting_2026-10-11.md` Mục 2.2.

| # | Ý tưởng | Căn cứ | Mới so với bài | Chi phí | Đề xuất |
|---|---|---|---|---|---|
| Y22 | **Khởi tạo ưu tiên đặc trưng quan trọng:** thay khởi tạo ngẫu nhiên, cho mô-đun gắn với trọng số đặc trưng (attention theo biến, mô hình đa chuyên gia) ưu tiên từ đầu các đầu vào quan trọng theo lý thuyết thủy văn hoặc theo kết quả giải thích mô hình | GVHD gợi ý (biên bản 11/10/2026 Mục 2.2). Một lượt tra (25/10/2026) chưa tìm thấy công trình dự báo dòng chảy khởi tạo theo mức quan trọng của đặc trưng, chưa tra hết. Cách khởi tạo không ngẫu nhiên có bằng chứng gần nhất là tiền huấn luyện rồi tinh chỉnh (Y19: AIFL, Journal of Hydrology 678:136064, 2026) | BiasCast khởi tạo mặc định của thư viện; lớp nhúng tự học trọng số từng đầu vào | Cần mô-đun attention theo biến (Sequential Forecast LSTM chưa có) và một cách đặt trọng số ban đầu; vài lần huấn luyện | Không cho tiểu luận: bước C so các lõi phải cùng cách khởi tạo, cùng giao thức. Khóa luận, gộp với Y19 hoặc dùng kết quả bước G làm căn cứ chọn đặc trưng |
