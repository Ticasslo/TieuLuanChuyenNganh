# Kịch bản trình bày tiến độ ngày 25/10/2026 cho hai thành viên

Tài liệu dùng để tập nói và điều khiển màn hình khi báo cáo. Những đoạn trích dẫn là lời có thể đọc trực tiếp; phần **Mở**, **Chỉ vào** và **Chuyển màn hình** là hướng dẫn thao tác, không đọc thành lời. Các số liệu lấy từ kết quả đã kiểm trong repo, cập nhật ngày 25/10/2026. Bản thuyết minh gửi giảng viên là [báo cáo tiến độ](04_ProgressReport_2026-10-23.md).

Thông điệp xuyên suốt: **nhóm đã hiểu dữ liệu, tái lập suy luận, tự huấn luyện các mốc và xây dựng bộ đánh giá; LSTM có kỹ năng chuỗi khá tốt nhưng khả năng bắt đúng ngày khởi phát lũ còn hạn chế. Đặc điểm lưu vực có liên hệ với mức cải thiện và các xu hướng chính giữ qua ba seed.** Kết quả hiện có là nền thực nghiệm cho bước cải tiến và so sánh các lõi, chưa phải kết quả chứng minh lợi ích của Mamba.

## 1. Phân vai và đường đi của buổi trình bày

### 1.1. Bốn lượt trình bày

| Lượt | Người nói | Nội dung, màn hình chính | Thời lượng tập dượt |
|---|---|---|---|
| 1 | Người 1 — Huỳnh Thanh Nhân | Mở đầu, bài toán; A1: dữ liệu, độ thiếu, ngưỡng; A3: persistence, DLinear, kết quả và chuỗi dự báo | 9–10 phút |
| 2 | Người 2 — Huỳnh Ngọc Thắng | A2: tái lập từ trọng số tác giả; A4: ba LSTM huấn luyện từ đầu, thời gian, biến thiên giữa seed | 6–7 phút |
| 3 | Người 1 | Bộ đánh giá: phạm vi chung, CDF, bốn mức và sự kiện lũ | 6–7 phút |
| 4 | Người 2 | Phân tích theo lưu vực: nhóm thuộc tính, tương quan, biến thiên seed và kết luận | 7–8 phút |

Kịch bản đầy đủ dự kiến khoảng 30–35 phút, chưa tính hỏi đáp. [Mục 10](#10-lộ-trình-rút-gọn-15-phút) chỉ rõ những điểm dừng giữ lại khi có 15 phút. Số phút là lịch tập, không phải thời gian chạy notebook.

Phân vai này là phân công **trình bày**, không thay đổi người thực hiện hay chủ tài khoản của các notebook. Khi Người 2 trình bày phân tích lưu vực, vẫn ghi nguồn notebook thuộc tài khoản `hunhthanhnhn`.

### 1.2. Các tab cần mở sẵn

| Nhãn tab khi tập | Notebook, tài khoản | Nội dung cần mở sẵn |
|---|---|---|
| A — A3 | `02_A3_LamaHCEExt_Benchmarks`, `hunhthanhnhn` | Cell 9.5–9.7; output cell 10.7; hình ở cell 11.10 |
| B — A4 | `04_A4_LamaHCEExt_Compare`, `hunhngcthng`, Version 2 | Output Phần 2, 3 và 4 |
| C — Evaluation | `01_Evaluation_LamaHCEExt`, `hunhthanhnhn`, **Version 4** | Bản chấm **14 run**; output cell 11.1 và thư mục `evaluation_output/` |
| D — ByBasin | `02_Evaluation_LamaHCEExt_ByBasin`, `hunhthanhnhn`, **Version 2** | Bản có **18 phép so**, `A4=True`; output cell 9.1 và `basin_analysis_output/Figures/` |
| E — dự phòng huấn luyện | `03_A4_LamaHCEExt_Retrain_Seed111`, `hunhngcthng`, Version 1 | Log huấn luyện và kết quả checkpoint; dùng khi thầy muốn xem bằng chứng chạy thật |
| F — A1 và A2 | `01_A1_LamaHCEExt_Explore` Version 2; `01_A2_LamaHCEExt_Reproduce`, `hunhngcthng`, Version 2 | Hình bản đồ, thiếu qmax, bốn mức cho P1-01b; hình A2_CDF cho P2-00 |

Các tab A–D có thể đặt trên cùng một máy. Người đang cầm chuột chuyển màn hình theo lời người nói; tập trước cả thao tác chuyển tab và cuộn đến đầu ra. Nếu dùng hai máy, kiểm trước quyền mở notebook của nhau.

### 1.3. Xác nhận đúng phiên bản trước buổi báo cáo

Trong tab C, output kết luận phải có `runs: 14`, `A4_contracts_matched: true` và `common_day_IDs: 652628`. Trong tab D phải có `runs: 14`, `contrasts: 18`, `A4_included: true` và `association_rows: 1674`.

Bản chính thức là **Evaluation Version 4** và **ByBasin Version 2**; `.ipynb` trong repo là đúng hai bản này, mã trùng `.py`. `Tables/` và `Figures/` trong repo là Output nguyên byte của hai bản đó (một bộ 14 run duy nhất), nên mở repo hay mở Kaggle đều cùng số.

Các kết quả đã có sẵn để trình bày; mở bản lưu có output. Không bấm chạy lại cell huấn luyện trong buổi báo cáo. Huấn luyện LSTM mất khoảng 2,5–3,2 giờ một lần; dùng log và checkpoint đã lưu để chứng minh kết quả.

## 2. Cách tìm đúng cell và kết quả thật

### 2.1. Đánh dấu theo đề mục, không theo số thứ tự cell của giao diện

Tìm theo tên như **9.5. Huấn luyện và báo cáo seed 111**, không dựa vào số đếm cell tuyệt đối: số đếm có thể đổi khi notebook được chia lại Markdown/code. Khi tập, ghi nhớ tiêu đề và phần output ngay dưới cell đó.

| Notebook | Cell có kết quả cần chỉ | Cell định nghĩa hàm liên quan |
|---|---|---|
| A3 | 9.5, 9.6, 9.7 có log huấn luyện; 10.7 có bảng; 11.10 có hình chuỗi | Một số cell 9.1–9.3 và 10.1–10.6 chỉ chuẩn bị hàm/bảng |
| Evaluation | **11.1** điều phối, in kiểm nguồn/phạm vi/kết luận, ghi CSV và hình | 3.1, 4.1, 5.1, 6.1, 7.1, 8.4, 10.1–10.3 chủ yếu định nghĩa hàm |
| ByBasin | **9.1** điều phối, in nguồn/số run/số phép so/kết luận, ghi bảng và năm hình | Phần 4–7 định nghĩa phép phân tích và hàm vẽ |
| A4 Compare | Các cell code Phần **2, 3, 4** có bảng và hình trực tiếp | Phần 1 kiểm nguồn; Phần 5–6 kiểm/xuất dự báo |

Ở Evaluation, tiêu đề output **“Phần 8.4 - Ba hợp đồng LSTM A4”** xuất hiện trong kết quả của cell 11.1 vì hàm điều phối gọi bộ đọc A4. Không cần tìm một output riêng dưới cell định nghĩa hàm 8.4.

Ở ByBasin, **Phần 7.1 định nghĩa hàm vẽ**; năm PNG được tạo khi chạy cell 9.1 và lưu trong Output. Muốn trình bày hình, mở tệp PNG tương ứng trong `basin_analysis_output/Figures/`.

### 2.2. Những tệp Output cần mở trước

| Tab | Đường dẫn tính từ thư mục Output của notebook | Dùng ở điểm dừng |
|---|---|---|
| C | `evaluation_output/Evaluation_CDF_test_S0_lead1.png` | P1-07 |
| C | `evaluation_output/Evaluation_levels_pooled.csv` | P1-08 |
| C | `evaluation_output/Evaluation_events_pooled.csv` | P1-09 |
| C | `evaluation_output/Evaluation_events_pooled_tolerance1.csv` | P1-10 |
| C | `evaluation_output/Evaluation_summary.csv` | Tra cứu số trung vị theo run |
| D | `basin_analysis_output/Figures/BasinAnalysis_quartile_delta_NSE.png` | P2-06 |
| D | `basin_analysis_output/Figures/BasinAnalysis_static_delta_NSE_rho.png` | P2-07 |
| D | `basin_analysis_output/Figures/BasinAnalysis_quartile_flood_POD.png` | P2-08 |
| D | `basin_analysis_output/Figures/BasinAnalysis_seed_spread.png` | P2-09 |
| D | `basin_analysis_output/Figures/BasinAnalysis_priority_scatter.png` | Dự phòng khi hỏi về từng điểm/lưu vực |

Mở các CSV trước buổi báo cáo và xác định đúng hàng cần đọc. Với bảng sự kiện, chọn **test / S0 / lead 1 / T=2 / USWRC / cohort=all**; bảng chính có dung sai 0. Với bảng tổng hợp chỉ số, chọn **scope=common_source_all**. Không lấy số từ các scope khác nhau để lập một bảng so sánh.

Các bảng gọn dưới đây có thể dùng ngay khi tập nói. Nếu giao diện xem CSV khó đọc, dùng bảng tương ứng trong [thuyết minh Evaluation](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md#10-kết-quả-thực-nghiệm-trên-kaggle) và [thuyết minh ByBasin](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md#11-kết-quả-thực-nghiệm-và-kiểm-chứng); giữ tab Kaggle để chỉ nguồn và quá trình chạy.

## 3. Lượt 1 — Người 1: mở đầu, A1 và A3

### P1-01. Mở đầu: nhóm đang làm gì và đã đến đâu?

**Mở:** đầu [báo cáo tiến độ, Mục 1–2](04_ProgressReport_2026-10-23.md), sau đó chuyển sang tab A. Chưa cuộn vào mã.

**Chỉ vào:** dòng tên bài toán, đầu ra qmax, số 451 lưu vực và ba kỳ dữ liệu.

**Lời nói có thể đọc trực tiếp:**

> “Em xin trình bày tiến độ thực nghiệm của nhóm. Bài toán là dự báo lưu lượng cực đại của ngày kế tiếp cho 451 lưu vực ở Trung Âu, sử dụng bộ Extended LamaH-CE của bài cơ sở BiasCast.
>
> Lưu lượng là lượng nước đi qua một mặt cắt sông trong một giây, đơn vị m³/s. Trong dữ liệu, qmean là lưu lượng trung bình ngày, còn qmax là cực đại từ các số đo giờ trong ngày. Nhóm dự báo qmax, tức tập trung vào mức lưu lượng cực đại của ngày đích.
>
> Dữ liệu huấn luyện là 2003–2009, validation là 2010–2013 và test là 2014–2017. Validation được dùng để chọn mô hình; test dùng để báo cáo kết quả.
>
> Nhóm đã khảo sát dữ liệu ở A1, chạy lại sáu cấu hình từ trọng số tác giả ở A2, rồi tự huấn luyện ba DLinear ở A3 và ba LSTM ở A4. Sau đó, nhóm chấm các dự báo bằng cùng một bộ đánh giá và phân tích kết quả theo đặc điểm lưu vực.”

**Nếu cần giải thích thời điểm dự báo, thêm:**

> “Ví dụ cần dự báo qmax ngày 1/1/2014, quan trắc lưu lượng đầu vào chỉ được kết thúc ở 31/12/2013. Mô hình LSTM còn dùng dự báo thời tiết cho ngày đích; đó là thông tin dự báo, khác số đo của ngày đích sau khi ngày đó đã xảy ra.”

**Chốt:**

> “Hôm nay nhóm trình bày những gì đã chạy được, các mốc so sánh và những hạn chế nhìn thấy từ kết quả. Phần so sánh với Mamba là bước tiếp theo.”

Nguồn: [báo cáo tiến độ, Mục 1–2](04_ProgressReport_2026-10-23.md).

### P1-01b. A1: dữ liệu gồm những gì, thiếu ở đâu, ngưỡng lấy từ đâu?

**Mở theo thứ tự:** tab F (notebook A1) hoặc các hình cục bộ:
1. [Bản đồ lưu vực](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/Figures/A1_P3_basin_maps.png).
2. [Tổng quan thiếu qmax](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/Figures/A1_P4_qmax_missing_overview.png).
3. [Ngưỡng bốn mức](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/Figures/A1_P10_qmax_four_levels.png).
4. [Đỉnh năm và ngưỡng Gumbel](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/Figures/A1_P10_gumbel_return_levels.png), chỉ mở khi được hỏi.

**Chỉ vào:** ô (a) và (d) của bản đồ; bảng tỷ lệ thiếu theo kỳ; đường q30/q80/q98 của hình bốn mức.

**Lời nói — nguồn dữ liệu:**

> “Bộ dữ liệu gốc là LamaH-CE, công bố năm 2021 trên tạp chí Earth System Science Data, gồm các lưu vực ở Trung Âu. Nhóm tác giả BiasCast công bố bản mở rộng Extended LamaH-CE theo ngày cùng bài báo năm 2026, thêm ba nguồn khí tượng E-OBS, MSWEP, GLEAM và dự báo thời tiết ECMWF HRES. Nhóm dùng đúng 451 lưu vực ít hoặc không chịu tác động của con người mà bài cơ sở chọn.
>
> Trên bản đồ, nền xám là toàn bộ lưu vực của bộ gốc, màu xanh là 451 lưu vực được chọn. Trạm đặt chủ yếu ở Áo với 349 trạm và Đức 88 trạm; Thụy Sĩ có 9 và Séc có 5. Diện tích từ khoảng 4 đến hơn 15 nghìn km², trung vị khoảng 101 km², nên phần lớn là lưu vực nhỏ.
>
> Mỗi lưu vực có 33 thuộc tính tĩnh về địa hình, khí hậu, lớp phủ và thực vật, 31 biến khí tượng lịch sử từ bốn nguồn ERA5-Land, E-OBS, MSWEP, GLEAM, và 5 biến dự báo ECMWF HRES. Dự báo ECMWF HRES chỉ có từ 01/01/2002; train là 2003–2009.”

**Lời nói — độ thiếu và số mẫu:**

> “Trước khi huấn luyện, nhóm đo dữ liệu thiếu ở đâu. Nhãn qmax thiếu 3,55% ở train, 0,40% ở validation và 0,23% ở test. Phần thiếu tập trung ở một số trạm: lưu vực 324, 600 và 623 không có nhãn train nhưng có nhãn khi đánh giá; lưu vực 800 chỉ có 38 ngày nhãn train; lưu vực 55 thiếu lịch sử Q nên không tạo được cửa sổ đầu vào cho các mô hình có Q.
>
> Có nhãn chưa đồng nghĩa có mẫu huấn luyện, vì mỗi mẫu cần cả cửa sổ lịch sử. Với cửa sổ 730 ngày có Q, nếu đòi đủ mọi cột trên toàn cửa sổ thì có khoảng 928 nghìn mẫu; nếu chỉ kiểm các ô thật sự được dùng ở từng pha thì khoảng 1,08 triệu mẫu, nhiều hơn khoảng 151 nghìn. Kết quả này là căn cứ để sửa bộ lọc mẫu ở bộ nạp B2.”

| Kỳ | Ngày–lưu vực theo lịch | Nhãn thiếu | Tỷ lệ thiếu |
|---|---:|---:|---:|
| Train 2003–2009 | 1.153.207 | 40.900 | 3,55% |
| Validation 2010–2013 | 658.911 | 2.630 | 0,40% |
| Test 2014–2017 | 658.911 | 1.545 | 0,23% |

**Lời nói — ngưỡng đánh giá:**

> “A1 cũng dựng các ngưỡng cho phần đánh giá. Với mỗi lưu vực, nhóm lấy q30, q80, q98 của qmax quan trắc 1981–2013, tức hoàn toàn trước test, để chia bốn mức thấp, trung bình, cao và rất cao. Ngưỡng tính riêng từng lưu vực, nên cùng một trị số có thể là rất cao ở sông nhỏ nhưng bình thường ở sông lớn.
>
> Ngưỡng lũ 1,5; 2; 5 và 10 năm được ước lượng từ đỉnh qmax mỗi năm thủy văn, từ 1/10 đến 30/9, bằng phân phối Gumbel. Cả 451 lưu vực đều có ngưỡng; 427 lưu vực có ít nhất 10 năm đỉnh, 24 lưu vực chỉ có 3–9 năm nên kết quả của nhóm này được báo riêng.”

**Thảo luận thống kê nếu thầy hỏi:**
- Tỷ lệ bốn mức trên giai đoạn tham chiếu là 29,80%; 50,10%; 18,08% và 2,02%, khớp q30/q80/q98 theo cách dựng.
- Ở validation, mức rất cao chiếm 2,23%. Tỷ lệ ở test không bắt buộc đúng 2%, xem Q9.
- Mẫu đỉnh năm ngắn làm ngưỡng 5 và 10 năm kém chắc chắn hơn. Ví dụ lưu vực 800 chỉ có bốn năm đỉnh, nên ngưỡng của nó không được trình bày như ngưỡng đã kiểm chứng.

**Chốt:**

> “Phần khảo sát này cho ba thứ dùng ở các bước sau: danh sách lưu vực cần đọc riêng khi xem kết quả, căn cứ sửa bộ nạp, và các ngưỡng cố định để đánh giá phân mức và sự kiện lũ.”

Nguồn:
- [Báo cáo tiến độ, Mục 2.3 và 4](04_ProgressReport_2026-10-23.md#4-a1--dữ-liệu-cho-biết-điều-gì-trước-khi-học-mô-hình).
- Bản đồ và số trạm theo quốc gia: [A1 Mục 17.13](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/01_A1_LamaHCEExt_Explore.md#1713-bộ-bản-đồ-và-số-lưu-vực-theo-quốc-gia--36).
- Bốn mức: [A1 Mục 24.1](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/01_A1_LamaHCEExt_Explore.md#241-cell-101--bốn-mức-từ-qmax-quan-trắc-trước-test).
- Bài dữ liệu LamaH-CE: [Klingler và cs., ESSD 2021](https://doi.org/10.5194/essd-13-4529-2021).

### P1-02. Vì sao cần persistence và DLinear?

**Mở:** tab A, đầu Phần 9. Có thể trỏ thêm phần cấu hình DLinear 7.11 nếu thầy hỏi cách cài đặt; điểm trình bày chính vẫn là log và kết quả.

**Chỉ vào:** tiêu đề “Huấn luyện DLinear với ba seed”.

**Lời nói:**

> “Trước khi đánh giá mô hình học sâu, nhóm cần các mốc đơn giản. Mốc đầu tiên là persistence, có thể hiểu là dự báo giữ nguyên số đo gần nhất. Trong thí nghiệm này, dự báo qmax ngày mai bằng qmax hôm nay. Ví dụ hôm nay đo được 100 m³/s thì persistence dự báo ngày mai cũng 100 m³/s. Mốc này không cần huấn luyện.
>
> Mốc thứ hai là DLinear, một mô hình tuyến tính cho chuỗi thời gian có bước phân tách chuỗi. Nó nhận 365 ngày qmean quá khứ và học để dự báo qmax ngày tiếp theo.
>
> DLinear dùng trung bình trượt 25 ngày để tạo phần xu hướng, rồi lấy chuỗi ban đầu trừ xu hướng để có phần dao động. Mỗi phần đi qua một lớp tuyến tính, hai đầu ra được cộng lại. Hai nhánh được học chung bằng sai số dự báo cuối cùng.
>
> Persistence cho biết kỹ năng chỉ từ việc giữ nguyên giá trị gần nhất; DLinear cho biết một mô hình tuyến tính học từ cả lịch sử làm được đến đâu. Đây là các mốc để đọc lợi ích của mô hình phức tạp hơn.”

**Khi giải thích trung bình trượt, có thể dùng ví dụ giả định:**

> “Tại một vị trí, nếu giá trị gốc là 14 và trung bình cửa sổ là 10, thì xu hướng tại đó là 10, phần dao động là 4. Đây chỉ là ví dụ giải thích phép tách, không phải số liệu của một lưu vực cụ thể.”

**Cần phân biệt:** 25 ngày là cửa sổ làm mượt; 365 ngày là chiều dài lịch sử DLinear nhận. LSTM A4 có hợp đồng đầu vào khác, gồm lịch sử khí tượng/Q và dự báo ECMWF HRES; không nói hai mô hình dùng cùng toàn bộ đầu vào.

Nguồn: [A3, Phần 7–8](../../Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.md#phần-7-dữ-liệu-và-tiền-xử-lý-dlinear), [mã DLinear gốc](https://github.com/cure-lab/LTSF-Linear/blob/7f7a25b6d32fa0df77ce4c20d4ffeb849b5a7e47/models/DLinear.py).

### P1-03. Cell 9.5–9.7: chứng minh đã huấn luyện ba lần

**Mở:** output cell **9.5 — Huấn luyện và báo cáo seed 111**. Mở bảng lịch sử epoch và bảng “Kết quả từ checkpoint tốt nhất”. Sau đó lướt nhanh output 9.6 và 9.7 để chỉ có đủ seed 222, 333.

**Chỉ vào theo thứ tự:** số seed → log epoch → cột `validation_median_NSE` → `selected_epoch` → NSE test. Không đọc mọi cột cấu hình hoặc hash.

**Bảng để tập đọc:**

| Seed | Số epoch đã chạy | Epoch chọn | NSE trung vị validation tại checkpoint chọn | NSE trung vị test |
|---|---:|---:|---:|---:|
| 111 | 6 | 4 | 0,4247 | 0,3828 |
| 222 | 6 | 2 | 0,4240 | 0,3781 |
| 333 | 10 | 5 | 0,4282 | 0,3796 |

**Lời nói:**

> “Đây là log huấn luyện thật của seed 111. Mỗi dòng là một epoch, tức một lượt đi qua tập mẫu huấn luyện. Mô hình học trên train, sau đó được chấm trên validation.
>
> Seed là hạt giống của các phép ngẫu nhiên. Nhóm giữ nguyên cấu hình nhưng khởi tạo và chạy ba seed khác nhau để xem kết quả có phụ thuộc mạnh vào một lần chạy hay không.
>
> Với seed 111, mô hình chạy sáu epoch nhưng chọn checkpoint ở epoch 4 vì có NSE trung vị validation tốt nhất. Hai seed còn lại chọn epoch 2 và 5. Nhóm không chọn checkpoint bằng điểm test.
>
> Ba NSE trung vị test lần lượt khoảng 0,383; 0,378 và 0,380. Trung bình các NSE trung vị là khoảng 0,3802, độ lệch chuẩn mẫu khoảng 0,0024.”

**Nếu được hỏi `train_loss` có phải NSE không:**

> “Không ạ. Loss là hàm để tối ưu khi huấn luyện, ở đây theo quy ước NSE* của NeuralHydrology. NSE trong bảng đánh giá được tính riêng trên dự báo đã đổi về đơn vị vật lý. Hai cột đó không đọc như cùng một đại lượng.”

**Lưu ý cho người nói:** trung bình ba NSE trung vị không phải điểm của dự báo ensemble. Notebook giữ riêng ba chuỗi dự báo, không lấy trung bình ba chuỗi rồi gọi là ensemble.

Nguồn: output cell 9.5–9.7 và [A3, Phần 9](../../Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.md#phần-9-huấn-luyện-dlinear-với-ba-seed).

### P1-04. Cell 10.7: mô hình có hơn các mốc đơn giản không?

**Mở:** output **10.7 — Bảng kết quả và kỹ năng hơn persistence**. Cuộn đến bảng so sánh có các dòng `SeqLSTM_Q`, ba `DLinear_seed_*` và `Persistence_qmax`.

**Chỉ vào:** cột `NSE_median`, sau đó `KGE_median`; dùng bảng thuộc giao chung, không dùng dòng native riêng của từng run.

| Bộ dự báo | NSE trung vị trên giao chung | Ý nghĩa trong phần trình bày |
|---|---:|---|
| Persistence qmax | 0,3500 | Giữ nguyên số đo gần nhất |
| DLinear 111 / 222 / 333 | 0,3828 / 0,3781 / 0,3796 | Học quan hệ tuyến tính từ lịch sử Q |
| SeqLSTM_Q tác giả | 0,7053 | Trọng số LSTM tác giả, chạy lại ở A2 |

**Lời nói:**

> “Bảng này trả lời mô hình học được hơn mốc đơn giản bao nhiêu. Các bộ dự báo được chấm trên cùng ngày và cùng quan trắc.
>
> Mỗi lưu vực cho một điểm NSE, rồi nhóm lấy trung vị qua các lưu vực có điểm hợp lệ. NSE bằng 1 là hoàn hảo; bằng 0 là sai số bình phương ngang mốc trung bình quan trắc trong kỳ chấm; âm là sai số còn lớn hơn mốc đó. NSE 0,705 không phải tỷ lệ dự báo đúng 70,5%.
>
> Persistence đạt khoảng 0,350, DLinear đạt khoảng 0,380, còn LSTM từ trọng số tác giả đạt khoảng 0,705. DLinear có cải thiện so với giữ nguyên số đo gần nhất, nhưng vẫn thấp hơn LSTM trong các cấu hình đã chạy.
>
> Tuy nhiên, theo KGE, DLinear chỉ khoảng 0,51–0,52, thấp hơn persistence khoảng 0,675. KGE đọc thêm tương quan, biên độ dao động và mức trung bình, nên một chỉ số tốt lên chưa bảo đảm mọi khía cạnh đều tốt lên.”

**Chốt:**

> “Kết quả này xác lập được các mốc kỹ năng. Vì đầu vào của DLinear và LSTM khác nhau, nhóm chưa quy toàn bộ chênh lệch cho kiến trúc.”

Nguồn: output cell 10.7, [A3, Phần 10](../../Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.md#phần-10-so-sánh-các-mô-hình-trên-giao-ngày-chung).

### P1-05. Cell 11.10 và 11.12: mô hình sai ở đâu, vì sao cần đánh giá thêm?

**Mục đích của điểm dừng này:** giải thích những gì bảng NSE chưa thể hiện hết. Lưu vực 800 minh họa việc mô hình theo được biến động chung nhưng vẫn hụt đỉnh; 742/758 minh họa việc sai lệch mức nền gặp quan trắc ít biến động có thể làm NSE âm sâu. Đây là cầu nối từ bảng chỉ số sang đánh giá ngày lưu lượng cao và sự kiện lũ, không phải phần giới thiệu thêm một mô hình.

#### A. Cell 11.10 — nhìn trực tiếp sai sót trên chuỗi dự báo

**Mở:** hình ở output **11.10 — Chuỗi dự báo ở 742, 758, 363 và 800**. [Mở hình cục bộ](../../Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/Figures/A3_profile_hydrographs.png).

**Chỉ vào:** hàng lưu vực 800 trước, rồi hàng 742 hoặc 758. Xác nhận màu theo chú giải: đường đen liền là quan trắc qmax; nét xám là persistence; các đường còn lại là dự báo mô hình. Không dùng độ cao đường để so trực tiếp giữa hai hàng có thang trục khác nhau.

**Lời nói:**

> “Sau khi xem bảng NSE, nhóm kiểm tra trực tiếp chuỗi dự báo để biết mô hình còn sai ở đâu. Trục ngang là thời gian, trục dọc là qmax quy đổi theo diện tích, đơn vị mm/ngày. Đường đen là quan trắc, các đường màu là dự báo.
>
> Trước tiên, ở lưu vực 800, các đường dự báo theo được một phần biến động chung nhưng nhiều đỉnh cao của đường đen vẫn bị dự báo thấp hơn. Như vậy, có kỹ năng trên cả chuỗi chưa có nghĩa đã dự báo tốt những ngày lưu lượng cao. Đó là lý do nhóm đánh giá thêm theo mức lưu lượng và theo sự kiện.
>
> Tiếp theo, ở 742 và 758, lưu lượng quan trắc ít biến động nhưng nhiều dự báo bị lệch khỏi mức nền đó. Vì NSE so sai số với độ biến thiên quan trắc, sai lệch này có thể làm điểm âm rất sâu. Nhóm giữ các lưu vực này và kiểm nguyên nhân của điểm số.”

**Câu chuyển sang hình tiếp theo:**

> “Để phân biệt sai lệch mức nền với phần sai số thay đổi theo ngày, nhóm phân rã sai số ở hình tiếp theo.”

#### B. Cell 11.12 — mở hình và xác định các thành phần trước khi diễn giải

**Mở:** output **11.12 — Thành phần sai số chuẩn hóa**, hình `A3_profile_error_components.png`. [Mở hình cục bộ](../../Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/Figures/A3_profile_error_components.png).

**Hình này trả lời:** với mỗi bộ dự báo tại một lưu vực, sai số tổng thể có bao nhiêu giá trị đến từ **lệch trung bình**, và bao nhiêu đến từ **sai số thay đổi quanh mức lệch trung bình đó**?

**Giải thích bố cục theo đúng thứ tự chỉ tay:**

1. Bốn ô là bốn lưu vực: **742 trên trái, 758 trên phải, 363 dưới trái, 800 dưới phải**.
2. Trục ngang là các run. Mỗi cột là một bộ dự báo tại lưu vực đang xét. Hình A3 này gồm sáu run A2, ba DLinear và hai persistence; **các cột DLinear 111/222/333 không phải ba LSTM A4**.
3. Mỗi cột chồng hai phần: **cam** ở dưới, **xanh** ở trên.
4. Trục tung là **sai số bình phương trung bình đã chuẩn hóa bằng phương sai quan trắc**. Đây là tỷ số không có đơn vị, không phải qmax hay mm/ngày.
5. Đường ngang nét đứt tại **1** là mốc **NSE = 0**. Đọc vị trí đỉnh cột so với đường này.

**Lời nói mở hình:**

> “Mỗi ô vẫn là một lưu vực, nhưng ở đây mỗi mô hình được biểu diễn bằng một cột sai số. Nhóm chia cột thành hai phần để xem mô hình bị lệch mức trung bình nhiều hay sai số thay đổi nhiều theo ngày. Cột thấp hơn nghĩa sai số chuẩn hóa nhỏ hơn.”

#### C. Màu cam: sai lệch mức trung bình

Gọi sai số ngày t là **e(t) = dự báo(t) − quan trắc(t)**. Trung bình sai số cũng bằng trung bình dự báo trừ trung bình quan trắc.

**Phần cam = [trung bình sai số]² / phương sai quan trắc.** Nhãn tiếng Anh là `Squared mean error / obs variance`.

**Cách hiểu:** dự báo có thể thường xuyên cao hơn hoặc thấp hơn mức quan trắc, tạo sai lệch trung bình kéo dài. Bình phương làm phần đóng góp này không âm. Vì đã bình phương, **chỉ nhìn phần cam không biết dự báo lệch cao hay lệch thấp**; cần đối chiếu chuỗi ở cell 11.10 hoặc dấu của `mean_error` trong bảng chẩn đoán.

**Lời nói:**

> “Phần cam đo sai lệch mức trung bình. Ví dụ, nếu dự báo thường xuyên cao hơn quan trắc một lượng gần cố định, trung bình sai số sẽ khác 0 và phần cam tăng. Nếu thường xuyên thấp hơn, phần cam cũng tăng vì sai lệch được bình phương. Muốn biết lệch theo hướng nào thì phải xem lại chuỗi hoặc sai số có dấu.”

Không nói “cam là sai số của xu hướng DLinear”: đây là thành phần sai số đánh giá cho **mọi run**, không phải nhánh trend trong kiến trúc DLinear.

#### D. Màu xanh: sai số thay đổi quanh mức lệch trung bình

**Phần xanh = phương sai của sai số / phương sai quan trắc.** Nhãn tiếng Anh là `Error variance / obs variance`.

**Cách hiểu:** sau khi trừ mức sai lệch trung bình, sai số còn thay đổi nhiều hay ít giữa các ngày. Sai số có thể tăng mạnh ở một số đỉnh, giảm ở ngày khác, hoặc đổi dấu giữa dự báo thừa và thiếu. Phần xanh lớn cho biết có độ phân tán sai số lớn; riêng hình này chưa phân biệt đầy đủ lệch thời điểm, hụt biên độ hay các kiểu sai khác.

**Lời nói:**

> “Phần xanh đo mức thay đổi của sai số quanh mức lệch trung bình. Một mô hình có thể trung bình không lệch nhiều, nhưng hôm thì dự báo thừa, hôm thì thiếu, hoặc sai nhiều ở một số đỉnh. Những sai số đó có thể triệt tiêu khi lấy trung bình, nhưng vẫn xuất hiện trong phần xanh.”

**Điểm dễ nhầm:** màu xanh là phương sai **sai số**, không phải phương sai **dự báo**. Một dự báo hằng vẫn có phần xanh vì quan trắc thay đổi làm sai số thay đổi. Cũng không gọi phần xanh là toàn bộ “nhiễu ngẫu nhiên”: nó có thể chứa sai sót có cấu trúc của mô hình.

#### E. Tổng hai phần và đường mốc 1 liên hệ với NSE thế nào?

Với cùng ngày chấm, dùng phương sai theo ddof=0 như mã A3:

**MSE = [trung bình sai số]² + phương sai sai số.**

Chia hai vế cho phương sai quan trắc dương:

**Tổng giá trị cam + xanh = MSE / phương sai quan trắc = 1 − NSE.**

Do đó, **đỉnh cột là giá trị 1 − NSE**, không phải NSE. Cột càng cao thì NSE càng thấp.

| Giá trị đọc tại đỉnh cột — ví dụ minh họa | NSE tương ứng | Cách diễn giải |
|---|---:|---|
| 0 | 1 | Dự báo hoàn hảo trên các ngày chấm |
| 0,2 | 0,8 | Sai số bình phương nhỏ hơn mốc trung bình quan trắc |
| 1 | 0 | Sai số bình phương ngang mốc trung bình quan trắc |
| 5 | −4 | Sai số bình phương lớn hơn mốc đó |

**Lời nói:**

> “Tổng hai phần bằng 1 trừ NSE. Vì vậy đường nét đứt ở mức 1 tương ứng NSE bằng 0. Cột nằm dưới đường này cho NSE dương; cột cao hơn cho NSE âm. Ví dụ đỉnh cột bằng 5 thì NSE bằng 1 trừ 5, tức âm 4.”

**Lưu ý trục:** hình dùng symlog: vùng từ 0 đến 1 là tuyến tính, phía trên 1 được nén theo thang log. Cần đọc các mốc số trên từng ô; không so trực tiếp chiều cao cột giữa hai ô có thang hiển thị khác nhau. Cũng không ước lượng “cam chiếm bao nhiêu phần trăm sai số” bằng tỷ lệ chiều dài hai màu trên vùng log; muốn tỷ lệ phải dùng giá trị số trong bảng.

#### F. Ví dụ số để phân biệt cam và xanh khi tập

Ví dụ sau chỉ giải thích công thức, không phải số liệu của bốn lưu vực. Giả sử phương sai quan trắc bằng 1 trên bốn ngày:

| Sai số của một dự báo qua bốn ngày | Trung bình sai số | Phương sai sai số | Cam | Xanh | Tổng | NSE |
|---|---:|---:|---:|---:|---:|---:|
| +2, +2, +2, +2 | 2 | 0 | 4 | 0 | 4 | −3 |
| +2, −2, +2, −2 | 0 | 4 | 0 | 4 | 4 | −3 |

Hai dự báo có cùng MSE và NSE, nhưng kiểu sai khác nhau: trường hợp đầu luôn lệch cao một lượng cố định; trường hợp sau sai số đổi dấu nên trung bình bằng 0 mà sai số bình phương vẫn lớn. Đây là lý do xem hai màu thay vì chỉ một điểm NSE.

#### G. Đọc bốn ô của chính hình kết quả

| Ô hình | Chỉ vào đâu | Kết luận được phép rút ra |
|---|---|---|
| 742 — trên trái | Nhiều cột mô hình cao hơn đường 1; phần cam lớn ở nhiều run; hai persistence nằm dưới 1 | Nhiều run có NSE âm ở lưu vực này; sai lệch trung bình đóng góp lớn ở nhiều run. Không khẳng định mọi run đều cùng kiểu sai: SeqLSTM_noQ có phần sai số biến thiên nổi bật |
| 758 — trên phải | Phần cam rất lớn ở nhiều run, đặc biệt EncDec; DLinear 111 cũng cao hơn 1; persistence rất thấp | Mức trung bình dự báo lệch so với quan trắc, cùng phương sai quan trắc nhỏ, làm sai số chuẩn hóa lớn. Mô hình phức tạp không bảo đảm hơn persistence ở mọi lưu vực |
| 363 — dưới trái | Các DLinear hơi cao hơn 1; persistence qmax còn cao hơn; phần xanh chiếm phần lớn giá trị sai số ở nhiều run | DLinear có NSE hơi âm ở đây nhưng vẫn có thể hơn persistence, tức PNSE dương. NSE và PNSE dùng hai mốc khác nhau |
| 800 — dưới phải | Các cột dưới 1, phần xanh đóng góp lớn hơn phần cam | NSE dương nhưng vẫn còn sai số thay đổi theo ngày. Đối chiếu cell 11.10 thấy những đỉnh bị dự báo hụt; hình phân rã riêng lẻ chưa chứng minh mọi phần xanh đều do hụt đỉnh |

**Ví dụ thật dễ đọc:** tại lưu vực **758, DLinear seed 111**, giá trị cam xấp xỉ **12,516**, xanh **0,048**, tổng **12,564**. Vì tổng bằng 1−NSE nên NSE xấp xỉ **−11,564**. Trường hợp này có đóng góp sai lệch trung bình rất lớn; không cần đọc toàn bộ cột khác để giải thích cơ chế của điểm âm.

Các số làm tròn trên được đối chiếu từ [chỉ số cùng giao ngày](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_paired_contrasts.csv) và [trung bình/std quan trắc](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_attributes_and_diagnostics.csv). Trong Output A3, bảng dữ liệu của hình là `A3_figure_error_components.csv`.

#### H. Lời thoại liền mạch khoảng 1–2 phút cho cell 11.12

**Thao tác:** chỉ chú giải hai màu → đường mốc 1 → ô 758/DLinear 111 → ô 800 → chốt mục đích.

> “Hình này giúp nhóm giải thích vì sao các mô hình có điểm NSE như ở bảng trước. Mỗi cột là một bộ dự báo tại một lưu vực, gồm hai phần sai số.
>
> Phần cam là bình phương sai lệch trung bình: dự báo thường xuyên cao hơn hoặc thấp hơn mức quan trắc sẽ làm phần này tăng. Phần xanh là độ biến thiên của sai số quanh mức lệch trung bình: có thể trung bình không lệch nhiều nhưng từng ngày vẫn sai lớn.
>
> Hai phần đều được chia cho phương sai quan trắc, tổng bằng 1 trừ NSE. Vì thế cột thấp hơn đường nét đứt 1 cho NSE dương, cao hơn cho NSE âm.
>
> Chẳng hạn ở lưu vực 758, DLinear seed 111 có tổng khoảng 12,56 nên NSE khoảng âm 11,56; đóng góp lớn nằm ở sai lệch trung bình. Quan trắc ở đây ít biến động nên sai lệch nền trở nên rất lớn so với độ biến thiên của quan trắc.
>
> Còn ở lưu vực 800, các cột đều dưới 1, tức NSE dương, nhưng vẫn có phần sai số biến thiên đáng kể. Khi quay lại hình chuỗi, nhóm thấy nhiều đỉnh quan trắc bị dự báo hụt.
>
> Như vậy nhóm vừa kiểm các ca NSE âm sâu, vừa nhận ra có NSE dương chưa đủ để kết luận dự báo tốt các đỉnh. Đó là lý do phần đánh giá tiếp theo kiểm riêng mức lưu lượng cao và sự kiện lũ.”

**Nếu chỉ còn 30 giây, dùng bản ngắn:**

> “Cam là sai lệch trung bình, xanh là độ biến thiên của sai số; tổng bằng 1 trừ NSE. Ở 742/758, nhiều dự báo lệch mức nền trên chuỗi ít biến động nên NSE âm sâu. Ở 800, NSE dương nhưng vẫn còn sai số ở các biến động và đỉnh. Vì vậy nhóm cần xem cả chuỗi và đánh giá thêm theo mức, theo sự kiện.”

**Giới hạn cần giữ:** đây là đồng nhất thức mô tả sai số trên các ngày chấm. Nó không tự xác định vì sao mạng học sai, không chứng minh một thuộc tính hay một nguồn dữ liệu là nguyên nhân. Việc 363 được chọn để chẩn đoán sau khi đọc test không làm thay đổi tập ID, cấu hình hoặc chọn checkpoint.

Nguồn công thức và cách vẽ: [mã A3](../../Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.py), hàm chẩn đoán ở cell 11.3 và phần vẽ ở cell 11.12; [thuyết minh A3, Phần 11](../../Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.md#phần-11-hồ-sơ-bất-thường-và-trực-quan-hóa-kết-quả).

**Chuyển lời sang Người 2:**

> “Phần vừa rồi là các mốc A3 và chẩn đoán chuỗi. Tiếp theo, bạn Thắng trình bày A2, chạy lại từ trọng số tác giả, và A4, ba lần tự huấn luyện lại LSTM.”

## 4. Lượt 2 — Người 2: A2, A4 và biến thiên giữa ba LSTM

### P2-00. A2: chạy lại sáu cấu hình từ trọng số tác giả

**Mở:** tab F (notebook A2) hoặc [hình A2_CDF cục bộ](../../Workspace/03_Baseline/01_A2_LamaHCEExt_Reproduce/Figures/A2_CDF.png).

**Chỉ vào:** từng cặp đường liền (nhóm chạy lại) và nét đứt (tác giả); vị trí trục dọc 0,5.

**Lời nói:**

> “Trước khi tự huấn luyện, nhóm kiểm xem có chạy đúng quy trình của tác giả không. A2 nạp trọng số tốt nhất và thống kê chuẩn hóa tác giả đã công bố, rồi chạy suy luận trên test 2014–2017 cho sáu cấu hình. Sáu cấu hình gồm Sequential Forecast LSTM và Encoder–Decoder LSTM, mỗi loại có hoặc không có Q quá khứ, cùng hai mốc khí tượng: D_FC dùng dự báo, D_RA dùng tái phân tích.
>
> Tiêu chí tái lập được đặt trước khi chạy: quan trắc trùng tệp của tác giả, NSE trung vị lệch không quá 0,005, và sai khác dự báo ở phân vị 99,9% không quá 0,05 mm/ngày. Cả sáu cấu hình đều đạt. Trên hình, mỗi cặp đường gần như trùng nhau.
>
> Cấu hình tốt nhất là Sequential Forecast LSTM có Q, NSE trung vị 0,7053, khớp đúng số của bài. Đây là mốc mà các bước sau phải so.”

| Cấu hình | NSE trung vị tác giả | NSE trung vị nhóm chạy lại |
|---|---:|---:|
| Sequential Forecast LSTM có Q | 0,7053 | 0,7053 |
| Sequential Forecast LSTM không Q | 0,6276 | 0,6276 |
| Encoder–Decoder LSTM có Q | 0,6622 | 0,6622 |
| Encoder–Decoder LSTM không Q | ≈0,569 | ≈0,569 |
| D_FC | 0,3867 | 0,3864 |
| D_RA | 0,6923 | 0,6923 |

**Nếu được hỏi lưu vực lệch:**

> “Chỉ hai lưu vực 742 và 758 lệch NSE trên 0,01. Cả hai có NSE dưới −1 vì qmax ở kỳ test gần như không đổi, nên sai số rất nhỏ cũng làm NSE thay đổi nhiều. Lưu vực 55 không có dự báo ở hai cấu hình có Q do thiếu lịch sử Q; kết quả của tác giả cũng vậy.”

**Cần nói đúng:** A2 là tái lập **suy luận**: nạp trọng số có sẵn, không huấn luyện, nên không tính vào sáu lần huấn luyện mới. Mã thư viện cố định ở commit `dda45fc`.

Nguồn: [báo cáo tiến độ, Mục 5](04_ProgressReport_2026-10-23.md#5-a2--chạy-lại-từ-trọng-số-tác-giả), [báo cáo A2](../../Workspace/03_Baseline/01_A2_LamaHCEExt_Reproduce/01_A2_LamaHCEExt_Reproduce.md).

### P2-01. Phần 2: phân biệt A2 với A4

**Mở:** tab B, output cell code **Phần 2 — Bảng 3 hạt giống và tác giả, xét điều kiện A4**. Tên bảng bắt đầu **“Test 2014-2017: 3 hạt giống và tác giả (best_model.pt của mỗi run)”**.

**Chỉ vào:** ba dòng seed và dòng tác giả, các cột epoch, thời gian và NSE test.

**Lời nói:**

> “A2 vừa rồi chỉ nạp trọng số có sẵn. A4 là bước khác: nhóm khởi tạo và huấn luyện lại Sequential Forecast LSTM có Q từ đầu với ba seed.
>
> A4 giữ cấu hình mô hình nền của tác giả. Mục đích là kiểm khả năng tự huấn luyện lại và đo biến thiên giữa các lần chạy, làm mốc cho những cải tiến tiếp theo.”

| Seed | Epoch chọn | Số epoch chạy | Thời gian huấn luyện trên T4 | NSE trung vị validation | NSE trung vị test |
|---|---:|---:|---:|---:|---:|
| 111 | 11 | 16 | 2,50 giờ | 0,7411 | 0,6952 |
| 222 | 15 | 20 | 3,17 giờ | 0,7541 | 0,7038 |
| 333 | 14 | 19 | 3,08 giờ | 0,7574 | 0,7059 |

**Nói tiếp:**

> “Ba NSE trung vị test là 0,6952; 0,7038 và 0,7059. Giá trị từ trọng số tác giả khoảng 0,7053, nằm trong khoảng ba lần chạy. Trung bình và độ lệch chuẩn mẫu của ba NSE trung vị test là khoảng 0,7016 cộng trừ 0,0056.
>
> Mỗi lần huấn luyện mất khoảng 2,5–3,2 giờ trên T4. Tính cả ba DLinear và ba LSTM, nhóm đã có sáu lần huấn luyện mới từ đầu. Những lần nạp trọng số tác giả ở A2 không tính là huấn luyện mới.”

**Cần nói đúng:** A4 đạt điều kiện tái lập đã đặt; khoảng ba seed không phải khoảng tin cậy thống kê. Không chọn seed 333 để gọi là “mô hình tốt nhất” chỉ vì đã thấy test của seed đó cao nhất.

Nguồn: [báo cáo A4, kết quả Phần 2](../../Workspace/03_Baseline/03_A4_LamaHCEExt_Retrain/03_A4_LamaHCEExt_Retrain.md#p2).

### P2-02. Phần 4: nhìn đường học và checkpoint được chọn

**Mở:** output **Phần 4 — Hình**, hình **A4_Seeds.png**. [Mở hình cục bộ](../../Workspace/03_Baseline/04_A4_LamaHCEExt_Compare/Figures/A4_Seeds.png).

**Chỉ vào:** ô (a), ba đường validation và ngôi sao; sau đó ô (b), ba điểm test và đường mốc tác giả.

**Lời nói:**

> “Ô bên trái là NSE trung vị validation theo epoch. Mỗi đường ứng với một seed; ngôi sao đánh dấu epoch được lưu làm checkpoint tốt nhất. Ba checkpoint là epoch 11, 15 và 14.
>
> Có thể thấy điểm validation tăng nhanh lúc đầu, rồi dao động. Vì vậy nhóm dùng validation và quy tắc dừng sớm, thay vì mặc định lấy epoch cuối.
>
> Ô bên phải đối chiếu ba NSE trung vị test với mốc tác giả. Các điểm tương đối gần nhau nhưng không trùng nhau, cho thấy một lần chạy không đại diện đầy đủ cho biến thiên của quá trình huấn luyện.”

**Nếu được hỏi cùng seed 111 sao khác tác giả:**

> “Cùng seed chưa bảo đảm cùng toàn bộ phép tính. Môi trường thư viện và phần cứng khác nhau; A4 còn giữ cách chạy GPU không tất định của quy trình nền. Vì thế nhóm đo kết quả nhiều seed, không yêu cầu dự báo trùng từng giá trị với tác giả.”

**Nếu được hỏi lỗi lưu best model:**

> “Ở A4, nhóm giữ quy trình gốc để đo mốc nền. Riêng ba lần A4 này, nhóm so trọng số `best_model.pt` với trọng số từng epoch: epoch được lưu trùng epoch có NSE validation cao nhất, nên lỗi không ảnh hưởng kết quả A4. Bản sửa đã đưa vào thư viện của nhóm ở bước B1, kèm kiểm thử tự động; ba lần huấn luyện B1 đã hoàn tất; NSE trung vị test lần lượt là 0,7274; 0,7133; 0,7217, còn bộ đánh giá 14 run dưới đây chưa gồm B1.”

Giải thích lỗi chi tiết ở Q19 (Mục 8).

Nguồn: [báo cáo A4, Phần 4](../../Workspace/03_Baseline/03_A4_LamaHCEExt_Retrain/03_A4_LamaHCEExt_Retrain.md#p4).

### P2-03. Phần 3 hoặc hình A4_BasinNoise: một lưu vực có thể nhạy với seed

**Mở:** output Phần 3, bảng **“Nhiễu hạt giống theo mức NSE của lưu vực”**; hoặc hình thứ hai ở Phần 4, **A4_BasinNoise.png**. [Mở hình cục bộ](../../Workspace/03_Baseline/04_A4_LamaHCEExt_Compare/Figures/A4_BasinNoise.png).

**Chỉ vào:** trung vị std NSE theo lưu vực khoảng 0,0207; ô phân tán có trục tung là std qua ba seed. Giải thích mỗi chấm là một lưu vực.

**Lời nói:**

> “Trung vị toàn tập tương đối gần nhau chưa có nghĩa từng lưu vực đều ổn định. Nhóm tính độ lệch chuẩn NSE qua ba seed tại từng lưu vực; trung vị của các độ lệch chuẩn này khoảng 0,0207.
>
> Hình bên phải cho thấy có những lưu vực nhạy với seed hơn hẳn. Điều này quan trọng vì ở các bước sau, một mức cải thiện nhỏ của mô hình mới có thể lẫn với biến thiên giữa các lần huấn luyện.”

**Nếu dùng ô CDF của A4_BasinNoise:** nói rõ hình A4 này giới hạn trục và không hiển thị toàn bộ các điểm âm rất sâu. Hình CDF của bộ đánh giá ở lượt tiếp theo giữ toàn bộ điểm hữu hạn, nên dùng hình đó khi bàn phần đuôi âm.

**Bảng kiểm định Phần 3 chỉ trình bày nếu được hỏi:**

> “Hai cặp seed 111–222 và 111–333 có p sau hiệu chỉnh 0,008 và 0,013, dưới 0,05, dù median hiệu NSE chỉ khoảng 0,004. Do đó, nhóm đọc cùng độ lớn hiệu và biến thiên seed, không lấy p nhỏ một mình để kết luận cải tiến có giá trị thực tế.”

**Không trộn các họ p:** Phần A4 Compare có các họ so sánh riêng giữa seed và với tác giả. Bộ đánh giá 14 run hiệu chỉnh theo 78 cặp mỗi chỉ số; xem câu hỏi Q11 ở Mục 8 nếu thầy đối chiếu hai bảng.

**Chuyển lời sang Người 1:**

> “A4 cho thấy nhóm đã tự huấn luyện được mô hình nền và có mốc biến thiên giữa seed. Sau đây bạn Nhân trình bày cách chấm chung các bộ dự báo và vì sao ngoài NSE cần xem cả phân mức và sự kiện lũ.”

## 5. Lượt 3 — Người 1: bộ đánh giá chung

### P1-06. Cell 11.1: đang chấm những gì, trên dữ liệu nào?

**Mở:** tab C, output **11.1 — Input, chạy kiểm thử và đánh giá**. Tìm hai bảng **“Phần 8.4 - Manifest A4 và nguồn A3”** và **“Phần 8.4 - Ba hợp đồng LSTM A4”**, dòng **“14 run, giữ 451 ID”**, rồi JSON kết luận cuối cell.

**Chỉ vào:** `A3_reference_matched`, `A4_contracts_matched`, số run và phần coverage. Chỉ cần nói kiểm nguồn đã đạt, không đọc từng hash.

**Lời nói:**

> “Notebook này không huấn luyện thêm mô hình. Nó đọc các dự báo đã lưu từ A2, A3 và A4, rồi áp cùng cách tính chỉ số, cùng ngưỡng và cùng ngày chấm.
>
> Có 14 bộ dự báo: sáu run A2, ba DLinear, ba LSTM A4 và hai mốc persistence. Ở đây một run là một bộ kết quả cụ thể; ba seed được giữ thành ba run, nên 14 run không phải 14 kiến trúc khác nhau.
>
> Đợt thực nghiệm hiện tại là test, kịch bản S0, hạn một ngày. S0 nghĩa là không che thêm đầu vào theo kịch bản; dữ liệu nguồn vẫn có thể thiếu. Hạn một ngày nghĩa là dự báo ngày kế tiếp sau ngày cuối của lịch sử quan trắc.
>
> Để so công bằng, nhóm dùng giao chung: chỉ giữ những ngày tại từng lưu vực mà quan trắc, persistence và toàn bộ dự báo đều hữu hạn. Giao này có 652.628 ngày–lưu vực, thuộc 450 lưu vực có ngày được chấm.”

**Giải thích ngày–lưu vực:**

> “Ngày 1/1 ở lưu vực A là một cặp ngày–lưu vực, cùng ngày ở lưu vực B là một cặp khác. Vì vậy số này đếm cả chiều thời gian và không gian.”

**Nếu hỏi vì sao 451 mà chấm 450:**

> “ID 55 không có ngày hợp lệ trong giao chung: cửa sổ 365 ngày của các run có Q luôn chứa ô qmean trống nên các run đó không có dự báo ở lưu vực này, kể cả kết quả của tác giả. Nhóm vẫn giữ ID 55 với trạng thái không có ngày chấm; không tự đổi danh sách thành 450 lưu vực.”

Nguồn: [Evaluation, Mục 10](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md#10-kết-quả-thực-nghiệm-trên-kaggle).

### P1-07. Hình CDF: cả tập lưu vực được dự báo như thế nào?

**Mở:** `evaluation_output/Evaluation_CDF_test_S0_lead1.png`. [Hình 14 run cục bộ](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/Figures/Evaluation_CDF_test_S0_lead1.png).

**Chỉ vào theo thứ tự:** ô NSE → trục tung 0,5 → chú giải các LSTM/DLinear/persistence → ô PNSE → ô KGE.

**Lời nói:**

> “Mỗi lưu vực có một điểm, nên nhóm dùng đường phân phối để nhìn toàn bộ tập lưu vực thay vì chỉ một con số trung bình. Trục ngang là điểm chỉ số, trục dọc là tỷ lệ lưu vực có điểm không vượt mức đó.
>
> Ví dụ ở mức trục dọc 0,5, giao với một đường cho biết trung vị: khoảng một nửa lưu vực có điểm không vượt giá trị đó. Với NSE, PNSE và KGE, điểm lớn hơn là tốt hơn, nên đường dịch sang phải nhìn chung biểu thị phân phối điểm cao hơn. Nếu các đường cắt nhau thì cần đọc thêm từng vùng và phép so ghép cặp.
>
> Ô PNSE so trực tiếp với persistence. PNSE dương nghĩa sai số bình phương nhỏ hơn persistence trên cùng ngày; âm là lớn hơn. Đường persistence qmax nằm ở PNSE bằng 0 theo định nghĩa khi mẫu số hợp lệ.
>
> Ô KGE bổ sung góc nhìn về tương quan, độ dao động và mức trung bình. Các mô hình có thể tốt hơn theo NSE nhưng kém hơn theo KGE; trường hợp DLinear so persistence là một ví dụ.”

**Bảng tra nhanh khi nói:**

| Run | NSE trung vị | PNSE trung vị | KGE trung vị |
|---|---:|---:|---:|
| LSTM tác giả | 0,7053 | 0,4956 | 0,7671 |
| A4 111 | 0,6952 | 0,4954 | 0,7581 |
| A4 222 | 0,7038 | 0,5056 | 0,7910 |
| A4 333 | 0,7059 | 0,5018 | 0,7860 |
| Persistence qmax | 0,3500 | 0 | 0,6749 |

**Chốt:**

> “Các LSTM có kỹ năng chuỗi cao hơn các mốc đơn giản trong thí nghiệm này, nhưng vẫn có phần đuôi lưu vực dự báo kém. Tiếp theo nhóm kiểm riêng các ngày lưu lượng cao và các trận.”

**Nếu được hỏi về trục âm:** trục symlog có vùng tuyến tính gần 0, vùng log ở xa; nhờ đó giữ được cả điểm âm rất sâu. Không đọc khoảng cách ngang như thang tuyến tính trên toàn trục.

Nguồn số: [Evaluation_summary.csv của 14 run](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/Tables/Evaluation_summary.csv).

### P1-08. Bốn mức: bắt đúng những ngày rất cao được bao nhiêu?

**Mở:** `evaluation_output/Evaluation_levels_pooled.csv`. Chọn test/S0/lead 1/common_source_all; tìm các dòng `level=very_high` của `SeqLSTM_Q`, `DLinear_seed_111` và `Persistence_qmax`. Bảng gọn đủ 14 run nằm tại [Evaluation, Mục 10.2](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md#102-kết-quả-liên-tục-và-bốn-mức).

**Chỉ vào:** các cột `observed_days`, `correct`, `recall`, `precision`. Chưa mở toàn bộ ma trận nhầm lẫn nếu thời gian ngắn.

**Lời nói:**

> “Phần trước chấm sai số trên chuỗi số liên tục. Ở đây nhóm chuyển cả quan trắc và dự báo sang bốn mức: thấp, trung bình, cao và rất cao. Ngưỡng là q30, q80, q98 của từng lưu vực, đã tính ở A1 từ dữ liệu trước test.
>
> Recall mức rất cao trả lời: trong những ngày thật sự rất cao, mô hình bắt đúng được bao nhiêu. Precision trả lời: trong những ngày mô hình báo rất cao, bao nhiêu ngày thật sự rất cao.
>
> Trên giao chấm có 11.759 ngày–lưu vực mức rất cao. LSTM tác giả bắt đúng 5.595 ngày, recall khoảng 47,58%; persistence bắt đúng 4.716 ngày, khoảng 40,11%; DLinear seed 111 khoảng 22,18%. Precision của ba run này lần lượt khoảng 67,58%; 40,28% và 53,33%. Ba LSTM nhóm tự huấn luyện bắt đúng 5.155, 5.690 và 5.620 ngày, recall 43,84–48,39%, quanh mức trọng số tác giả.”

| Run | Recall rất cao | Precision rất cao |
|---|---:|---:|
| SeqLSTM_Q tác giả | 47,58% | 67,58% |
| A4 111 / 222 / 333 | 43,84% / 48,39% / 47,79% | 69,63% / 66,53% / 65,68% |
| DLinear 111 | 22,18% | 53,33% |
| Persistence qmax | 40,11% | 40,28% |

**Chốt:**

> “DLinear có NSE nhỉnh hơn persistence nhưng recall ngày rất cao thấp hơn. Như vậy, cải thiện kỹ năng cả chuỗi chưa bảo đảm cải thiện ở dải lưu lượng mà mình quan tâm.”

**Khi hỏi accuracy:**

> “Accuracy chung của persistence cao hơn LSTM tác giả trong lần chấm này, nhưng accuracy chịu ảnh hưởng các mức phổ biến. Nhóm vẫn cần đọc riêng recall và precision của mức rất cao, không dùng accuracy một mình để đánh giá cảnh báo.”

Nguồn: [bảng mức cộng dồn đã kiểm](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/Tables/Evaluation_levels_pooled.csv).

### P1-09. Sự kiện chính: có bắt đúng ngày khởi phát trận không?

**Mở:** `evaluation_output/Evaluation_events_pooled.csv`, các dòng **test/S0/lead 1/T=2/USWRC/cohort=all**, dung sai chính bằng 0. Bảng gọn 14 run ở [Evaluation, Mục 10.3](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md#103-sự-kiện-lũ-chu-kỳ-hai-năm).

**Giải thích POD trước khi đọc số:** POD viết đầy đủ là **Probability of Detection**, trong phần này gọi là **tỷ lệ phát hiện được các trận thật**. Nó trả lời: **“Trong tất cả trận thực sự xảy ra và đủ điều kiện chấm, mô hình bắt được bao nhiêu trận?”**

**POD = số trận bắt được / (số trận bắt được + số trận bỏ sót).**

Ví dụ minh họa: có 100 trận thật, mô hình bắt được 20 và bỏ sót 80 thì POD = 20/(20+80) = 0,20 = **20%**. Có thể nói “mô hình bắt được 20 trong 100 trận thật”. Trên cùng tập trận, POD càng cao nghĩa là bỏ sót càng ít. Đây không phải tỷ lệ đúng của mọi ngày và chưa cho biết có bao nhiêu báo động nhầm.

Trong bảng chính của đề tài, “bắt được” còn phải thỏa quy tắc **trận dự báo bắt đầu cùng ngày trận thật**. Vì vậy, khi đọc POD phải đi kèm ngưỡng, cách gộp trận và dung sai thời gian.

**Đọc các cột liên quan:** `hit` = số trận bắt được; `missed` = số trận thật bị bỏ sót; `false_alarm` = số trận dự báo không ghép được với trận thật. **FAR = False Alarm Ratio**, tức tỷ lệ báo nhầm trong số trận được dự báo: **FAR = báo nhầm / (bắt được + báo nhầm)**. POD đo mức bỏ sót, FAR đo mức báo nhầm; cần xem cả hai.

**Chỉ vào:** `hit`, `missed`, `false_alarm`, `POD`, `FAR`.

**Lời nói:**

> “Phân mức vừa rồi kiểm từng ngày. Phần này kiểm sự kiện vượt ngưỡng lũ theo chu kỳ lặp lại. Các ngày vượt ngưỡng được trích thành sự kiện; tiêu chí USWRC dùng để gộp các đỉnh chưa độc lập thành cùng một cụm.
>
> Trong bảng chính, trận dự báo phải bắt đầu cùng ngày với trận thật. POD là tỷ lệ trận thật được bắt đúng, bằng số bắt được chia số bắt được cộng số bỏ sót. FAR là tỷ lệ báo nhầm trong các trận được dự báo, bằng báo nhầm chia bắt được cộng báo nhầm.
>
> Ở ngưỡng hai năm có 1.066 trận thật đủ điều kiện trên 366 lưu vực. Ba LSTM A4 bắt đúng ngày 180, 216 và 191 trận; POD khoảng 16,89%, 20,26% và 17,92%. FAR vẫn khoảng 67–70%.”

| Run, T=2 năm, USWRC, dung sai 0 | Bắt được | Bỏ sót | Báo nhầm | POD | FAR |
|---|---:|---:|---:|---:|---:|
| LSTM tác giả | 208 | 858 | 431 | 19,51% | 67,45% |
| A4 111 | 180 | 886 | 372 | 16,89% | 67,39% |
| A4 222 | 216 | 850 | 446 | 20,26% | 67,37% |
| A4 333 | 191 | 875 | 446 | 17,92% | 70,02% |
| DLinear 111 | 2 | 1.064 | 327 | 0,19% | 99,39% |
| Persistence qmax | 0 | 1.066 | 1.066 | 0% | 100% |

**Chốt quan trọng:**

> “LSTM có NSE trung vị khoảng 0,70 nhưng vẫn bỏ sót nhiều ngày khởi phát trận theo quy tắc này. Seed 333 có NSE cao nhất, còn seed 222 có POD cao nhất. Do đó xếp theo NSE chưa quyết định thứ tự về bắt lũ.”

**Giải thích 366 nếu được hỏi:**

> “450 lưu vực có ngày chấm chỉ số chuỗi, nhưng chỉ 366 có trận thật đủ điều kiện ở ngưỡng hai năm. Lưu vực không có trận thật không được gán POD bằng 0 hay bằng 1 để đưa vào tương quan.”

**Giải thích ngưỡng hai năm nếu được hỏi:**

> “Đó là ngưỡng lưu lượng ước tính theo phân phối cực đại năm, ứng với chu kỳ lặp lại hai năm trong mô hình thống kê. Nó không có nghĩa cứ đúng hai năm sẽ xuất hiện một trận. Nhóm còn báo các ngưỡng 1,5; 5 và 10 năm riêng.”

Nguồn: [bảng sự kiện 14 run](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/Tables/Evaluation_events_pooled.csv).

### P1-10. Bảng phụ ±1 ngày: tại sao persistence lại có POD 100%?

**Mở:** `Evaluation_events_pooled_tolerance1.csv`, cùng T=2/USWRC/cohort=all. Chỉ vào dòng persistence và một hoặc ba dòng A4.

**Lời nói:**

> “Nhóm giữ thêm bảng phụ cho phép lệch ngày bắt đầu tối đa một ngày để xem kết quả nhạy với dung sai thời gian như thế nào.
>
> Với dung sai này, persistence đạt POD 100% ở ngưỡng hai năm vì chuỗi dự báo của nó là quan trắc dịch sang ngày hôm sau. Trận bị trễ một ngày vẫn được ghép với trận thật trong bảng phụ.
>
> Điều đó không có nghĩa persistence cảnh báo đúng ngày. Vì vậy, bảng chính của thí nghiệm hạn một ngày dùng dung sai 0; bảng ±1 chỉ dùng để đọc độ nhạy thời gian. POD của ba A4 ở bảng phụ là khoảng 28,71%; 34,05% và 31,24%.”

**Chuyển lời sang Người 2:**

> “Bộ đánh giá cho thấy kỹ năng chuỗi và khả năng bắt trận là hai góc nhìn khác nhau. Phần cuối, bạn Thắng sẽ trình bày mô hình cải thiện nhiều hoặc ít ở những loại lưu vực nào, và các xu hướng đó có giữ qua ba seed hay không.”

Nguồn: [bảng phụ 14 run](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/Tables/Evaluation_events_pooled_tolerance1.csv).

## 6. Lượt 4 — Người 2: phân tích lưu vực và kết luận

### P2-04. Cell 9.1: câu hỏi của phân tích thuộc tính

**Mở:** tab D, output **9.1 — Kiểm đầu vào → tính bảng → vẽ → ghi nguồn**. Tìm dòng **14 run; 18 phép so; A4=True** và JSON kết luận.

**Chỉ vào:** `contrasts: 18`, `static_predictors: 31`, `association_rows: 1674`, `models_retrained: false`.

**Lời nói:**

> “Sau khi có dự báo và chỉ số từng lưu vực, nhóm đặt câu hỏi: mức cải thiện của mô hình có thay đổi theo diện tích, độ cao, lượng tuyết và độ khô hạn hay không?
>
> Notebook này không huấn luyện thêm. Nó ghép thuộc tính với điểm đánh giá của cùng lưu vực, rồi phân tích 18 phép so trên 31 thuộc tính dạng số.
>
> Đại lượng thường dùng là hiệu NSE của mô hình trừ NSE persistence tại cùng lưu vực. Ví dụ mô hình có NSE 0,6 và persistence 0,4 thì hiệu bằng 0,2 điểm. Đây không phải mức tăng 20% hoặc NSE tuyệt đối của mô hình.”

**Tóm tắt 18 phép so nếu cần:** sáu A2 so persistence; ba DLinear so persistence; hai cặp có/không Q; một cặp D_RA/D_FC; ba A4 so persistence; ba A4 so tác giả. Tổng 18. Ba outcome là ΔNSE, ΔPNSE, ΔPOD; 31 × 18 × 3 = 1.674 hàng hệ số.

**Câu giữ đúng phạm vi:**

> “Phân tích này xác định liên hệ giữa đặc điểm lưu vực và mức cải thiện. Nó chưa xác định trực tiếp mô hình đang sử dụng đầu vào nào hoặc đóng góp nhân quả của từng biến.”

Nguồn: [ByBasin, Mục 5 và 11](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md#11-kết-quả-thực-nghiệm-và-kiểm-chứng).

### P2-05. Giải thích Q1–Q4 trước khi đọc hình

**Mở:** hình `BasinAnalysis_quartile_delta_NSE.png`, chưa đọc số ngay. [Mở hình cục bộ](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Figures/BasinAnalysis_quartile_delta_NSE.png).

**Chỉ vào:** trục ngang Q1–Q4 và tiêu đề bốn cột.

**Lời nói:**

> “Nhóm chia lưu vực thành các nhóm theo thuộc tính. Q1 là nhóm có thuộc tính thấp nhất, Q4 là nhóm cao nhất. Ví dụ với độ cao, Q1 là các lưu vực thấp hơn và Q4 là các lưu vực cao hơn.
>
> Đây là chia theo thuộc tính, không phải chia theo NSE. Mọi mô hình dùng cùng biên nhóm. Trục dọc của hình là trung vị hiệu NSE so persistence trong nhóm, nên đường cao hơn nghĩa mức cải thiện điển hình lớn hơn.”

**Lưu ý để nói chính xác:** các biên tứ phân vị được tính từ 451 giá trị thuộc tính; số ID có outcome hữu hạn được báo riêng. Ties có thể làm giảm số nhóm với một số thuộc tính. Bốn thuộc tính ưu tiên hiện có bốn nhóm và được giữ cố định khi so các run.

### P2-06. Hình nhóm ΔNSE: ba seed giữ xu hướng nào?

**Mở tiếp:** vẫn hình trên, tập trung **hàng dưới — Author and LSTM A4**, cột **Mean elevation**; sau đó cột **Snow fraction** và **Aridity**. Hàng trên dùng để đối chiếu DLinear và các mốc tác giả.

**Chỉ vào:** Q1 và Q4 ở ô độ cao, ba nét A4 trong chú giải; sau đó đường khô hạn.

**Lời nói:**

> “Ở hàng dưới là trọng số tác giả và ba LSTM nhóm tự huấn luyện. Mỗi kiểu nét giữ riêng một seed.
>
> Với độ cao, nhóm thấp nhất có median cải thiện NSE khoảng 0,425–0,430 điểm, trong khi nhóm cao nhất khoảng 0,126–0,132 điểm. Cả ba seed đều cho thấy mức cải thiện thấp hơn ở nhóm cao nhất.
>
> Với tỷ phần tuyết, mức cải thiện cũng thấp hơn ở nhóm tuyết cao nhất. Với chỉ số khô hạn, median hiệu ở Q1 khoảng 0,238–0,245, còn Q4 khoảng 0,371–0,392.
>
> Các nhóm giữa không phải lúc nào cũng tăng hoặc giảm đều. Điều nhóm quan sát được là xu hướng tổng thể và sự tương đồng qua seed; không kết luận cứ tăng độ cao một chút thì điểm sẽ giảm một lượng cố định.”

**Chốt:**

> “Mô hình vẫn cải thiện theo median ở nhóm cao nhất, nhưng lợi ích so persistence nhỏ hơn. Điều này khác với nói mọi lưu vực cao đều có NSE thấp.”

Nguồn: [bảng nhóm](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_grouped_skill.csv), [biên nhóm](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_quartile_definitions.csv).

### P2-07. Heatmap: định lượng liên hệ, tránh chỉ nhìn đường nhóm

**Mở:** `BasinAnalysis_static_delta_NSE_rho.png`. [Mở hình cục bộ](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Figures/BasinAnalysis_static_delta_NSE_rho.png).

**Chỉ vào:** thang màu; ba cột **LSTM A4 111/222/333 minus Persistence**; ba hàng `elev_mean`, `frac_snow`, `arid_1`. Không bắt đầu bằng đọc đủ 31 tên thuộc tính.

**Lời nói:**

> “Hình này tổng hợp tương quan thứ hạng Spearman giữa thuộc tính và hiệu NSE. Mỗi hàng là một thuộc tính, mỗi cột là một phép so mô hình. Màu xanh là liên hệ âm, màu đỏ là liên hệ dương; màu không phải giá trị NSE.
>
> Liên hệ âm nghĩa là lưu vực có thuộc tính lớn hơn thường đi cùng mức cải thiện thấp hơn. Liên hệ dương nghĩa là thường đi cùng mức cải thiện cao hơn.
>
> Với ba LSTM A4 so persistence, hệ số độ cao nằm khoảng −0,42, tỷ phần tuyết khoảng −0,27, còn độ khô hạn khoảng 0,18–0,21. Ba seed giữ cùng hướng của các liên hệ này.”

| Thuộc tính, outcome ΔNSE so persistence | A4 111 | A4 222 | A4 333 |
|---|---:|---:|---:|
| Diện tích | −0,092 | −0,101 | −0,105 |
| Độ cao trung bình | −0,415 | −0,423 | −0,417 |
| Tỷ phần tuyết | −0,261 | −0,273 | −0,267 |
| Chỉ số khô hạn | 0,183 | 0,201 | 0,212 |

**Câu cần nói ngay sau bảng:**

> “Các hệ số này mô tả sự khác nhau giữa lưu vực. Độ cao, tuyết, khí hậu và thực vật có thể biến đổi cùng nhau; một hệ số riêng lẻ chưa chứng minh thuộc tính đó gây ra kết quả. Nhóm dùng kết quả để xác định các nhóm cần kiểm chứng tiếp bằng thí nghiệm che nguồn hoặc giải thích mô hình.”

**Nếu hỏi có ổn định qua ba seed không:**

> “Với ΔNSE, cả 31 thuộc tính giữ dấu của hệ số qua ba seed. Tuy nhiên, cùng dấu không có nghĩa liên hệ mạnh: chẳng hạn hệ số diện tích chỉ khoảng −0,1. Nhóm báo cả độ lớn hệ số, số ID và phạm vi biến thiên, không chỉ đếm số hệ số cùng dấu.”

Nguồn: [hệ số](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_static_associations.csv), [độ ổn định](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_seed_association_stability.csv).

### P2-08. Hình POD theo nhóm: phát hiện trận không giống tối ưu NSE

**Mở:** `BasinAnalysis_quartile_flood_POD.png`, hàng dưới, cột **Catchment area**. [Mở hình cục bộ](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Figures/BasinAnalysis_quartile_flood_POD.png).

**Nhắc lại trước khi chỉ hình:** POD là tỷ lệ trận thật được phát hiện theo quy tắc ghép đang dùng; POD 0,2 trên trục tung nghĩa là **20% số trận thật được bắt**, không phải NSE bằng 0,2. Bảng này cộng số trận trong nhóm trước khi chia, không lấy trung bình POD của các lưu vực.

**Chỉ vào:** Q1 và Q4 diện tích; nhắc trục dọc đã đổi thành POD, không còn là ΔNSE.

**Lời nói:**

> “Ở hình này trục dọc là POD cộng dồn trận thật ở ngưỡng hai năm, yêu cầu bắt đầu đúng ngày. Nhóm cộng số trận bắt được trước rồi chia tổng số trận thật trong từng nhóm.
>
> Ở nhóm diện tích nhỏ nhất, ba LSTM có POD khoảng 13–16%. Ở nhóm lớn nhất, POD khoảng 25–30%. Ví dụ seed 222 tăng từ 42 trên 267 trận, khoảng 15,73%, lên 64 trên 216 trận, khoảng 29,63%.
>
> Trong khi đó, hình hiệu NSE cho mức cải thiện ở nhóm diện tích lớn nhất thấp hơn nhóm nhỏ nhất. Hai kết quả không mâu thuẫn: một bên là hiệu kỹ năng chuỗi theo lưu vực, bên kia là tỷ lệ bắt trận cộng dồn. Chúng đo hai khía cạnh và có trọng số khác nhau.”

**Chốt:**

> “Vì mục tiêu có liên quan các mức lưu lượng cao, nhóm cần nhìn cả chuỗi và sự kiện, không dùng chỉ một chỉ số để xác định nhóm lưu vực phù hợp.”

Nguồn: [bảng nhóm và số trận](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_grouped_skill.csv).

### P2-09. Hình seed spread: ổn định ở cấp tập khác ổn định từng lưu vực

**Mở:** `BasinAnalysis_seed_spread.png`. [Mở hình cục bộ](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Figures/BasinAnalysis_seed_spread.png).

**Chỉ vào:** ô ΔNSE bên trái; hai hộp DLinear và LSTM_A4; sau đó ô ΔPOD bên phải. Trục tung là độ lệch chuẩn mẫu qua ba seed tại từng lưu vực, không phải NSE.

**Lời nói:**

> “Dù hướng liên hệ thuộc tính tương đối ổn định, từng lưu vực vẫn có thể thay đổi đáng kể giữa các seed. Với mỗi lưu vực, nhóm có ba hiệu NSE so persistence và tính độ lệch chuẩn của chúng.
>
> Đường giữa hộp biểu diễn trung vị của các độ lệch chuẩn theo lưu vực. Giá trị này khoảng 0,02065 với LSTM và 0,00383 với DLinear. Các chấm ngoài hộp là những lưu vực có độ biến thiên khác biệt lớn; nhóm vẫn giữ chúng trong hình.
>
> Đây là mô tả hai cấu hình đã chạy. DLinear và LSTM có đầu vào khác nhau nên nhóm không quy toàn bộ chênh lệch độ phân tán cho kiến trúc.”

**Chỉ tiếp ô POD và nói:**

> “DLinear gần như nằm ở độ phân tán POD bằng 0 vì các seed gần như đều không bắt được trận đúng ngày. Kết quả ổn định quanh mức kém không phải lợi thế. Với LSTM, POD thay đổi giữa seed ở 123 trên 366 lưu vực có trận thật, nên biến thiên này cũng cần được đọc khi đánh giá khả năng cảnh báo.”

**Nếu được hỏi n=450 và n=366:** NSE/PNSE có 450 ID đủ điểm; POD chỉ có 366 ID có trận thật đủ điều kiện. Ba seed không tạo thành ba lưu vực độc lập.

Nguồn: [phân tán LSTM](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_LSTM_A4_seed_spread.csv), [phân tán DLinear](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_DLinear_seed_spread.csv).

### P2-10. Hình scatter — dự phòng khi muốn xem từng lưu vực

**Mở khi được hỏi:** `BasinAnalysis_priority_scatter.png`. [Mở hình cục bộ](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Figures/BasinAnalysis_priority_scatter.png).

**Chỉ vào:** hàng đầu, ô độ cao; đường ngang 0 và các chấm hai phía. Chú ý hình cố định ba phép so: LSTM tác giả so persistence; LSTM tác giả có Q so không Q; DLinear seed 111 so persistence. Đây không phải hình hiển thị ba seed A4.

**Lời nói:**

> “Mỗi chấm là một lưu vực. Chấm ở trên đường 0 nghĩa mô hình bên trái có NSE cao hơn mốc so ở lưu vực đó; chấm dưới 0 nghĩa kém hơn. Có thể thấy các điểm phân tán khá rộng, nên một hệ số tương quan không có nghĩa mọi lưu vực đều đi theo cùng một quy luật.”

### P2-11. Kết luận và hướng tiếp theo

**Mở:** [báo cáo tiến độ, Mục 8](04_ProgressReport_2026-10-23.md#8-công-việc-tiếp-theo-và-sản-phẩm-bàn-giao), hoặc giữ hình cuối và nói kết luận.

**Lời nói kết thúc:**

> “Tóm lại, nhóm đã khảo sát dữ liệu, tái lập suy luận từ trọng số tác giả, tự huấn luyện ba DLinear và ba LSTM, rồi xây dựng bộ đánh giá chung. LSTM đạt NSE trung vị khoảng 0,70, sát mốc tác giả và cao hơn các mốc đơn giản trong cấu hình đã chạy.
>
> Khi đánh giá sâu hơn, nhóm thấy kỹ năng cả chuỗi chưa đồng nghĩa bắt tốt ngày khởi phát lũ. Phân tích thuộc tính cho biết mức cải thiện còn thay đổi theo loại lưu vực; các liên hệ chính với độ cao, tuyết và độ khô hạn giữ qua ba seed.
>
> B1 gồm ba thay đổi trên LSTM nền: sửa lưu checkpoint tốt nhất, đưa Q quá khứ về cùng đơn vị mm/ngày với nhãn, và thêm qmax quá khứ làm đầu vào. Bản sửa checkpoint đã có trong thư viện của nhóm kèm kiểm thử; B1 đã hoàn tất ba seed, có NSE trung vị test 0,7274; 0,7133; 0,7217 theo output chính thức. Bảng 14 run trong buổi trình bày này chưa gồm B1; bước tiếp theo là đưa dự báo B1 qua cùng bộ đánh giá để so với A4. Luồng bộ nạp B2 viết tiếp trên mã B1, tổ chức đầu vào theo nguồn và xử lý nhóm bị thiếu. Sau khi có khung dữ liệu và đánh giá thống nhất, nhóm mới so sánh các lõi LSTM, GRU, Transformer, S4D và Mamba.
>
> Các kết quả hôm nay xác lập mốc thực nghiệm và các câu hỏi cần kiểm chứng ở bước cải tiến.”

**Nếu cần phân công câu chốt:** Người 1 bổ sung một câu về việc tích hợp bộ nạp/đánh giá; Người 2 nói về B1 và luồng mô hình. Phân biệt kết quả huấn luyện B1 vừa có với bảng đánh giá chung 14 run chưa gồm B1; không nói B2 đã tích hợp hoặc Mamba đã được so sánh trên dataset này.

## 7. Tờ nhớ thuật ngữ ngay khi đứng trình bày

| Thuật ngữ | Cách nói ngắn, dễ hiểu | Điểm cần tránh nhầm |
|---|---|---|
| Lưu vực | Vùng đất có nước chảy về một cửa ra/trạm đo | Không phải một điểm dữ liệu theo ngày |
| Q, qmean, qmax | Lưu lượng; trung bình ngày; cực đại từ các số đo giờ trong ngày | Không phải mực nước hoặc diện tích ngập |
| mm/ngày theo diện tích | Quy đổi Q để đối chiếu quy mô lưu vực; q=Q×86,4/area_gov | Vẫn là biểu diễn lưu lượng, không phải lượng mưa đầu ra |
| Persistence | Giữ nguyên số đo gần nhất để dự báo ngày tiếp theo | Không huấn luyện |
| DLinear | Mô hình tuyến tính cho chuỗi thời gian có bước tách xu hướng/dao động | Tên mô hình cụ thể, không phải LSTM hoặc mô hình phi tuyến |
| Seed | Hạt giống của các phép ngẫu nhiên trong một lần chạy | Cùng seed không bảo đảm trùng từng bit giữa mọi môi trường |
| Checkpoint | Trọng số được lưu tại một thời điểm huấn luyện | Checkpoint chọn theo validation, không theo test |
| Run | Một bộ dự báo/kết quả của cấu hình, seed và phạm vi cụ thể | 14 run không phải 14 kiến trúc |
| S0 | Không che thêm đầu vào theo kịch bản | Dữ liệu gốc vẫn có thể thiếu |
| Hạn 1 | Dự báo ngày kế tiếp lịch sử quan trắc | Không phải dự báo đồng thời bảy ngày |
| Mặt nạ | Bảng chọn các ngày–lưu vực được chấm | Không phải tự ý bỏ các điểm xấu |
| Giao chung | Các vị trí đủ quan trắc và mọi dự báo để so công bằng | Có thể ít ngày hơn phạm vi riêng của mỗi run |
| NSE | So sai số bình phương với mốc trung bình quan trắc trong kỳ chấm | Không phải accuracy, không đổi 0,70 thành đúng 70% |
| PNSE | So sai số bình phương với persistence | 0 là ngang persistence khi mẫu số xác định |
| KGE | Kết hợp tương quan r, tỷ lệ std α, tỷ lệ trung bình β | Tính từng lưu vực rồi tổng hợp; không thay median r/α/β vào công thức |
| CDF | Tỷ lệ lưu vực có điểm không vượt giá trị trục ngang | CDF không phải đường dự báo theo thời gian |
| Median | Trung vị: giá trị ở giữa phân phối điểm | Khác mean, ít bị kéo bởi số cực đoan hơn |
| Recall mức rất cao | Tỷ lệ ngày rất cao thật được nhận diện đúng | Khác tỷ lệ trận lũ bắt được |
| POD sự kiện — Probability of Detection | Tỷ lệ phát hiện được các trận thật: bắt được / (bắt được + bỏ sót). Ví dụ bắt 20/100 trận thì POD 20% | Phụ thuộc quy tắc ghép; lưu vực không có trận thật có POD không xác định |
| FAR sự kiện — False Alarm Ratio | Tỷ lệ báo nhầm trong số trận được dự báo: báo nhầm / (bắt được + báo nhầm) | Không phải 1−POD; khác false positive rate dùng mẫu số khác |
| ΔNSE | NSE mô hình trừ NSE mốc tại cùng lưu vực/ngày | Đơn vị điểm, không phải phần trăm tăng |
| Spearman ρ | Hệ số liên hệ theo thứ hạng, trong khoảng [−1;1] | Liên hệ không phải đóng góp nhân quả |
| Q1–Q4 trên hình nhóm | Nhóm thuộc tính từ thấp đến cao | Không phải bốn mức lưu lượng; cũng không phải nhóm NSE |
| Std giữa seed | Độ biến thiên kết quả qua các lần huấn luyện | Không phải sai số chuẩn hoặc khoảng tin cậy |

Nguồn khái niệm: [báo cáo tiến độ, Mục 2–3](04_ProgressReport_2026-10-23.md), [hình lý thuyết đánh giá](../06_Theory/08_EvaluationFigures.md), [giao thức Evaluation](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md), [giao thức ByBasin](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md).

## 8. Câu hỏi có thể gặp và câu trả lời ngắn

### Q1. Tại sao chưa dùng Mamba mà đã chạy DLinear và LSTM?

**Người trả lời chính: Người 2.**

> “Nhóm cần một mốc huấn luyện lại được và một bộ đánh giá đáng tin trước khi thay lõi. Nếu mô hình mới hơn LSTM một chút nhưng mức hơn đó nằm trong dao động giữa seed, kết luận sẽ yếu. A3/A4 giúp xác lập các mốc này; phần Mamba là bước so sánh tiếp theo.”

### Q2. DLinear có phải linear regression không? Tại sao dùng tên riêng?

**Người trả lời chính: Người 1.**

> “Về bản chất, trước khi chặn đầu ra âm, DLinear vẫn là phép tuyến tính cộng bias trên lịch sử đầu vào. Tên Decomposition-Linear chỉ thiết kế cụ thể: tách chuỗi bằng trung bình trượt, học hai nhánh tuyến tính rồi cộng lại. Nhóm giữ tên DLinear để chỉ đúng mô hình gốc được triển khai.”

### Q3. Mô hình có học riêng cho mỗi lưu vực không?

**Người trả lời chính: Người 2.**

> “Các run ở đây dùng một bộ trọng số chung qua nhiều lưu vực, không huấn luyện 451 mô hình riêng. Mỗi lần inference vẫn dự báo cho từng chuỗi lưu vực; điểm đánh giá được tính riêng theo ID rồi tổng hợp. Đầu vào LSTM còn có thuộc tính tĩnh để mô tả khác biệt giữa các lưu vực.”

### Q4. Có dùng dữ liệu tương lai không?

**Người trả lời chính: Người 1; Người 2 bổ sung kiến trúc khi cần.**

> “Quan trắc Q đầu vào kết thúc trước ngày đích. DLinear dùng qmean quá khứ. LSTM SeqLSTM_Q dùng lịch sử quan trắc và dự báo ECMWF HRES ngày đích. Scaler và thống kê huấn luyện mới được fit từ train; checkpoint chọn theo validation. Riêng D_RA dùng tái phân tích ở ngày đích nên được ghi rõ là mốc tham chiếu về nguồn thông tin, không xem là cấu hình dự báo vận hành tương đương.”

Nếu hỏi trung bình trượt DLinear có lấy ngày sau vị trí đang xét: cửa sổ làm mượt là đối xứng bên trong **chuỗi lịch sử đã biết**, biên được đệm bằng giá trị đầu/cuối. Không lấy qmean của ngày đích từ bên ngoài cửa sổ quá khứ.

### Q5. Tại sao lại qmean đầu vào nhưng qmax đầu ra?

**Người trả lời chính: Người 1.**

> “Nhãn của bài cơ sở là qmax ngày đích, còn qmean quá khứ là thông tin quan trắc đầu vào. Hai đại lượng có liên hệ nhưng không giống nhau. Ở A3, nhóm dùng DLinear nhận qmean để tạo mốc lưu lượng quá khứ; không nói đây là dự báo qmean rồi đem chấm như qmax.”

### Q6. Vì sao chọn checkpoint theo validation mà vẫn in test từng epoch?

**Người trả lời chính: Người 1 với DLinear; Người 2 với A4/B1.**

> “Với DLinear ở A3, NSE test được ghi mỗi epoch để theo dõi theo góp ý của thầy. A4 giữ nguyên quy trình gốc của tác giả nên chưa ghi test từng epoch; từ B1, notebook chấm test cho mọi epoch sau khi huấn luyện xong và vẽ cùng đường validation. Trong mọi trường hợp test không tham gia chọn checkpoint hoặc dừng sớm; nhóm chọn bằng validation đã khai báo, không chọn epoch hoặc seed theo test.”

### Q7. NSE âm có phải code sai không?

**Người trả lời chính: Người 1.**

> “NSE âm là một giá trị hợp lệ: sai số mô hình lớn hơn mốc trung bình quan trắc trong kỳ chấm. Ở lưu vực có phương sai nhãn rất nhỏ, sai lệch nền có thể làm NSE âm rất lớn. Nhóm đã đối chiếu ngày, đơn vị, quan trắc và chỉ số trước khi phân tích các ca này.”

### Q8. NSE tăng nhưng KGE hoặc bắt lũ giảm thì chọn theo cái gì?

**Người trả lời chính: Người 1.**

> “Đề tài dùng NSE trung vị validation cho việc chọn mô hình theo giao thức. Khi báo cáo trên test, nhóm dùng nhiều chỉ số để nêu cả ưu và nhược điểm. Nếu NSE tăng nhưng kỹ năng ở mức cao hoặc sự kiện giảm, nhóm phải báo đúng sự đánh đổi đó, không gọi là tốt hơn toàn diện.”

### Q9. Ngưỡng rất cao hoặc ngưỡng lũ tính từ test không?

**Người trả lời chính: Người 1.**

> “Không ạ. Ngưỡng được tính ở A1 từ giai đoạn trước test 1981–2013 rồi áp cố định vào test. Đây là quy ước xây ngưỡng đánh giá của đề tài; khác với scaler mô hình, được fit từ train 2003–2009. Ngưỡng trước test có thể gồm giai đoạn validation nhưng không dùng test để đặt ngưỡng.”

Với q98, không nói luôn đúng 2% số ngày test phải nằm ở mức rất cao: tỷ lệ đó phụ thuộc phân phối test so với giai đoạn tính ngưỡng.

### Q10. Sao persistence có POD bằng 0 ở bảng này và bằng 1 ở bảng kia?

**Người trả lời chính: Người 1.**

> “Hai bảng có dung sai ngày bắt đầu khác nhau. Persistence dịch quan trắc một ngày; bảng chính đòi đúng ngày nên không nhận các trận bị trễ đó. Bảng phụ cho lệch một ngày nên nhận chúng. Đây là lý do nhóm giữ bảng chính và bảng phụ tách biệt.”

Các số 0 và 1 ở đây là kết quả thực nghiệm của ngưỡng/cách gộp đang trình bày, không khẳng định mọi dataset đều cho persistence đúng các giá trị đó.

### Q11. Sao A4 Compare báo hai cặp seed khác nhau có ý nghĩa, bộ đánh giá chỉ báo một?

**Người trả lời chính: Người 2, Người 1 bổ sung bộ đánh giá.**

> “Hai bảng hiệu chỉnh Holm theo hai họ phép thử khác nhau. A4 Compare chỉ xét ba cặp seed trong một họ; bộ đánh giá 14 run có 78 cặp cho mỗi chỉ số. Họ càng lớn thì p đã hiệu chỉnh càng tăng, nên cặp nằm sát ngưỡng có thể qua ở bảng này mà không qua ở bảng kia. Nhóm ghi rõ họ phép thử của từng bảng và không dùng p đã hiệu chỉnh của bảng này thay cho bảng kia.”

| NSE, cặp seed | p Holm, A4 Compare (3 cặp) | p Holm, Evaluation (78 cặp) |
|---|---:|---:|
| 111 so 222 | 0,0084 | 0,0251 |
| 111 so 333 | 0,0129 | 0,0515 |
| 222 so 333 | 0,189 | 0,531 |

Cả hai bảng đều cho thấy hai lần chạy chỉ khác seed vẫn có thể khác nhau có ý nghĩa thống kê dù trung vị hiệu chỉ khoảng 0,004 NSE. Đây là lý do nhóm đọc p cùng độ lớn hiệu và mốc nhiễu seed. Nêu bảng này khi thật sự được hỏi, không dùng làm phần mở đầu.

### Q12. Ba LSTM không khác tác giả có ý nghĩa thì đã chứng minh tương đương chưa?

**Người trả lời chính: Người 2.**

> “Chưa ạ. Không phát hiện khác biệt trong phép kiểm đang dùng không phải chứng minh tương đương. Kết luận của nhóm là huấn luyện lại đạt tiêu chí tái lập đã đặt và kết quả trung vị tương đối sát mốc tác giả, đồng thời báo cả khoảng và độ phân tán của ba lần chạy.”

### Q13. Như vậy đã tìm được thuộc tính quan trọng nhất chưa?

**Người trả lời chính: Người 2.**

> “Nhóm đã tìm được những thuộc tính có liên hệ ổn định với mức cải thiện, chẳng hạn độ cao, tuyết và khô hạn. Tuy nhiên đây là tương quan giữa các lưu vực, chưa phải mức đóng góp của từng đầu vào trong mô hình. Các thuộc tính có thể đồng biến; bước che nguồn và giải thích mô hình sẽ bổ sung bằng chứng về việc mô hình dùng thông tin nào.”

Nếu thầy nêu ví dụ chó–mèo: “Phân tích hiện tại cho biết mô hình có xu hướng làm tốt hơn ở nhóm dữ liệu có đặc điểm gì. Để nói riêng đặc điểm đó làm kết quả thay đổi, cần thí nghiệm kiểm soát hoặc giải thích mô hình, không chỉ nhìn tương quan.”

### Q14. Độ cao cao thì mô hình kém hơn phải không?

**Người trả lời chính: Người 2.**

> “Kết quả đang nói mức cải thiện so persistence có xu hướng thấp hơn ở nhóm cao hơn. Nó chưa đồng nhất với NSE tuyệt đối thấp hơn, vì persistence ở mỗi loại lưu vực cũng có kỹ năng khác nhau. Cần giữ rõ mốc so sánh khi diễn giải.”

### Q15. Tại sao p-value tương quan nhỏ mà vẫn phải ghi giới hạn?

**Người trả lời chính: Người 2.**

> “Phép hoán vị theo ID giả định các lưu vực có thể trao đổi được. Thực tế có lưu vực lồng nhau và cùng khí hậu vùng nên có phụ thuộc không gian. Nhóm ưu tiên hệ số, hướng và số ID; p hỗ trợ thăm dò. Đã có kiểm độ nhạy trên nhóm không có lưu vực được chọn khác ở thượng lưu, nhưng chưa gọi đó là mẫu độc lập hoàn toàn.”

### Q16. DLinear ít biến động hơn, sao không chọn luôn DLinear?

**Người trả lời chính: Người 2.**

> “Độ ổn định và chất lượng là hai đại lượng khác nhau. DLinear có độ phân tán nhỏ hơn trong cấu hình này nhưng NSE thấp hơn LSTM và gần như không bắt được ngày khởi phát trận theo quy tắc chính. Vì vậy không chọn mô hình chỉ vì các lần chạy gần nhau.”

### Q17. Kết quả này có áp dụng cho dataset khác không?

**Người trả lời chính: Người 2.**

> “Hiện kết luận thực nghiệm áp dụng cho Extended LamaH-CE và những cấu hình đã chạy. Muốn khẳng định trên dataset khác cần đánh giá trên dataset đó; mối liên hệ trong một dataset chưa chứng minh khả năng chuyển miền.”

### Q18. B2 đã xong chưa, vì sao chưa huấn luyện Mamba?

**Người trả lời chính: Người 1 về bộ nạp, Người 2 về khung mô hình.**

> “Chưa ạ. B1 đã hoàn tất ba seed; B2 có bản kiểm bộ nạp độc lập và cần tích hợp trên mã B1, rồi hai luồng rà soát chung trước khi huấn luyện B2. Khi khung dùng chung ổn định, nhóm mới so các lõi để tránh thay đồng thời nhiều yếu tố mà không biết lợi ích đến từ đâu.”

### Q19. Lỗi lưu mô hình tốt nhất là gì, sửa thế nào?

**Người trả lời chính: Người 2.**

> “Sau mỗi epoch, thư viện làm hai việc: kiểm có lưu `best_model.pt` không, và kiểm có dừng sớm không. Dừng sớm chỉ coi là cải thiện khi NSE validation tăng hơn 0,005, và chỉ khi đó mới cập nhật mức tốt nhất của nó. Hàm lưu mô hình lại so với chính mức tốt nhất đó, nên một epoch kém hơn epoch tốt nhất thật vẫn có thể ghi đè `best_model.pt`.
>
> Ví dụ NSE validation qua 8 epoch là 0,600; 0,650; 0,680; 0,684; 0,682; 0,683; 0,681; 0,6835. Epoch 4 tốt nhất (0,684), nhưng 0,684 chỉ hơn 0,680 có 0,004 < 0,005 nên dừng sớm vẫn giữ mức 0,680. Các epoch 5, 6, 7, 8 (0,682; 0,683; 0,681; 0,6835) đều lớn hơn 0,680 nên lần lượt ghi đè; cuối cùng `best_model.pt` là epoch 8 (0,6835) thay vì epoch 4.
>
> Bản sửa giữ một giá trị tốt nhất riêng cho việc lưu mô hình, không đụng quy tắc dừng sớm. Kiểm thử tự động chạy đúng dãy trên và yêu cầu `best_model.pt` là epoch 4; cùng dãy chạy với mã cũ cho epoch 8.”

Nguồn: [BasePaper Mục 11.5](02_BasePaper.md), [lý thuyết Mục 4.7](../06_Theory/06_BiasCast4_Model.md); kiểm thử `NeuralHydrology/test/test_best_model.py` ở fork `e794d7c`.

### Q20. Vì sao chỉ dùng 451 lưu vực, không dùng hết bộ gốc?

**Người trả lời chính: Người 1.**

> “Nhóm giữ đúng danh sách 451 lưu vực của bài cơ sở để so được với kết quả và trọng số tác giả. Bài mô tả đây là các lưu vực ít hoặc không chịu tác động của con người, tính trên toàn bộ vùng thượng nguồn của trạm. Khi đối chiếu, nhóm thấy danh sách trùng với nhóm lưu vực có mức tác động u hoặc l và bắt đầu đo không muộn hơn 2010. Đây là quy tắc nhóm tự tái hiện từ dữ liệu, tác giả chưa xác nhận là tiêu chí lọc.”

Nguồn: [A1 Mục 10](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/01_A1_LamaHCEExt_Explore.md#10-danh-sách-451-có-được-giải-thích-đầy-đủ-bằng-ul-không--217219), [BasePaper Mục 2](02_BasePaper.md).

### Q21. Train thiếu nhãn 3,55% có làm sai kết quả không?

**Người trả lời chính: Người 1.**

> “Ngày thiếu nhãn không được dùng để tính loss hay chỉ số; nhóm không tự điền giá trị vào đó. Ảnh hưởng chính là một số trạm có ít mẫu train, như 324, 600, 623 không có nhãn train và 800 chỉ có 38 ngày. Vì vậy, khi đọc kết quả theo từng trạm, nhóm đánh dấu riêng các trạm này thay vì loại khỏi bảng.”

## 9. Những câu diễn đạt cần sửa khi tập

| Cách nói dễ gây hiểu nhầm | Cách nói dùng trong báo cáo |
|---|---|
| “NSE 0,7 là đúng 70%” | “NSE trung vị khoảng 0,7; đây là chỉ số sai số so với một mốc, không phải accuracy” |
| “Nhóm có 14 mô hình khác nhau” | “Nhóm chấm 14 run, gồm các cấu hình, seed và mốc dự báo” |
| “DLinear tốt hơn persistence” mà không chỉ rõ chỉ số | “DLinear có NSE/PNSE trung vị cao hơn, nhưng KGE và recall rất cao thấp hơn trong lần chấm này” |
| “LSTM đã giải quyết tốt cảnh báo lũ” | “LSTM có kỹ năng chuỗi khá tốt nhưng POD đúng ngày ở ngưỡng hai năm mới khoảng 17–20%” |
| “Seed 333 là seed tốt nhất nên chọn nó” | “Báo cáo đủ ba seed; checkpoint chọn bằng validation, không chọn seed theo test” |
| “POD persistence 100% nên tốt nhất” | “Giá trị đó thuộc bảng phụ ±1 ngày chấp nhận độ trễ; bảng chính đòi đúng ngày” |
| “31 thuộc tính cùng dấu nên tất cả đều quan trọng” | “Dấu hệ số nhất quán qua seed; cần xem độ lớn và các đặc điểm đồng biến” |
| “Độ cao làm NSE giảm” | “Độ cao có liên hệ âm với mức cải thiện NSE so persistence” |
| “Ba seed cho thêm ba lần số lượng lưu vực” | “Ba seed là ba kết quả trên cùng tập lưu vực” |
| “Trung bình ba score là ensemble” | “Đó là tổng hợp chỉ số qua ba lần chạy; chưa tạo dự báo ensemble” |
| “Từ mô hình có Q cao hơn là chứng minh Q gây cải thiện đúng từng đó” | “Các run hiện có cho chênh lệch; tách riêng ảnh hưởng đầu vào còn cần kiểm soát cấu hình và nhiễu seed” |
| “Dataset có thuộc tính này nên chắc chắn qua dataset khác NSE thấp” | “Cần kiểm chứng chuyển miền trên dataset tương ứng” |

## 10. Lộ trình rút gọn 15 phút

| Thời gian | Người nói | Điểm dừng cần giữ | Phần có thể bỏ khi thiếu thời gian |
|---|---|---|---|
| 0–1 phút | Người 1 | P1-01: bài toán, kỳ dữ liệu, tiến độ | Ví dụ đơn vị quy đổi dài |
| 1–2,5 phút | Người 1 | P1-01b: bản đồ 451 lưu vực, bảng tỷ lệ thiếu, một câu về ngưỡng | Số mẫu 730 ngày, Gumbel, thảo luận thống kê |
| 2,5–4,5 phút | Người 1 | P1-02 ngắn; bảng 10.7 ở P1-04 | P1-03 chỉ nói một câu; bỏ P1-05 nếu không hỏi NSE âm |
| 4,5–7 phút | Người 2 | P2-00 một câu "cả sáu đạt, 0,7053"; P2-01 bảng ba seed; P2-02 hình A4_Seeds | Bảng sáu cấu hình A2; P2-03 và bảng kiểm định chi tiết |
| 7–11 phút | Người 1 | P1-06 phạm vi; P1-07 CDF; P1-09 bảng lũ, thêm một câu từ P1-10 | Bốn mức chỉ nói 1–2 câu; không đọc toàn bộ CSV |
| 11–14 phút | Người 2 | P2-04/05 định nghĩa ΔNSE/Q1–Q4; P2-06 hình nhóm; P2-07 ba hệ số | Bỏ scatter; POD theo nhóm và seed spread chỉ nói kết luận ngắn |
| 14–15 phút | Người 2 | P2-11: ba kết luận và bước B1/B2 tiếp theo | Không đọc lại tất cả số liệu |

Nếu chỉ có một hình cho phần thuộc tính, ưu tiên **BasinAnalysis_quartile_delta_NSE.png**, vì nhóm thấp/cao và ba seed dễ giải thích trước. Heatmap và seed spread giữ làm bằng chứng bổ sung.

## 11. Tập dượt và kiểm tra trước khi báo cáo

### 11.1. Buổi tập đầu

1. Mở sẵn tab A–D và các hình/bảng đã lưu, xác nhận tab C có 14 run và tab D có 18 phép so.
2. Người 1 nói P1-01, P1-01b rồi P1-02 đến P1-05, tự bấm đến đúng output; Người 2 theo dõi xem có hiểu người nói đang chứng minh điều gì không.
3. Người 2 nói P2-00 đến P2-03 và chuyển lời; Người 1 mở đúng tab C ngay sau câu chuyển.
4. Người 1 nói P1-06 đến P1-10; tập đọc POD và FAR đúng mẫu số.
5. Người 2 nói P2-04 đến P2-11; tập chỉ đúng hàng/cột A4 và phân biệt ΔNSE với NSE.
6. Mỗi người đặt ba câu hỏi từ Mục 8 cho người kia. Khi không nhớ số lẻ, dùng khoảng đã làm tròn trong kịch bản và mở bảng nguồn để tra.

### 11.2. Trước giờ báo cáo

- Mở đúng Evaluation Version 4 và ByBasin Version 2 (14 run, 18 phép so).
- Các cell huấn luyện có log/checkpoint đã lưu; không để notebook đang chờ chạy trong lúc trình bày.
- Mở sẵn các tệp CSV với đúng scope, kỳ, hạn, ngưỡng, grouping và cohort; tránh cuộn tìm hàng giữa lúc nói.
- Người 2 mở được tab ByBasin của Người 1 hoặc Người 1 giữ màn hình đó khi Người 2 thuyết trình.
- Hình và bảng cục bộ ở repo dùng làm bản trình bày dự phòng; không chép hình số liệu sang thư mục Document.
- Mở sẵn tab F (A1, A2) cho P1-01b và P2-00; notebook Seed111 để ở tab dự phòng, chỉ mở khi được hỏi.
- Bấm giờ một lần theo bản đầy đủ, một lần theo bản 15 phút.
- Nhớ bốn câu chốt: **đã huấn luyện sáu lần mới; LSTM gần mốc tác giả; NSE tốt chưa đủ cho bắt lũ; liên hệ thuộc tính ổn định chưa phải quan hệ nhân quả.**

## 12. Nguồn để tra khi tập nói

| Nội dung | Tài liệu/đầu ra đối chiếu |
|---|---|
| Toàn bộ tiến độ, dữ liệu, chỉ số cơ bản | [Báo cáo tiến độ](04_ProgressReport_2026-10-23.md) |
| Bài toán và kiến trúc LSTM nền | [Giới thiệu BiasCast, Mục 3–6 và 11.2](02_BasePaper.md) |
| Dữ liệu, độ thiếu, số mẫu, ngưỡng | [A1](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/01_A1_LamaHCEExt_Explore.md), Mục 17, 18, 21, 24 |
| Tái lập từ trọng số tác giả | [A2](../../Workspace/03_Baseline/01_A2_LamaHCEExt_Reproduce/01_A2_LamaHCEExt_Reproduce.md) |
| Huấn luyện DLinear, bảng và chẩn đoán | [A3](../../Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.md), Phần 7–11 |
| A4, thời gian, epoch, nhiễu seed | [Báo cáo A4](../../Workspace/03_Baseline/03_A4_LamaHCEExt_Retrain/03_A4_LamaHCEExt_Retrain.md), [notebook Compare chính thức](../../Workspace/03_Baseline/04_A4_LamaHCEExt_Compare/04_A4_LamaHCEExt_Compare.ipynb) |
| Giao thức và kết quả đánh giá chung | [Evaluation](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md), Mục 10 |
| Kết luận chạy Evaluation Version 4 | [Evaluation_summary.json](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/Tables/Evaluation_summary.json), [kiểm thử](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/Tables/Evaluation_checks.csv) |
| Thuộc tính, nhóm và độ ổn định A4 | [ByBasin, Mục 11](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md#11-kết-quả-thực-nghiệm-và-kiểm-chứng) |
| Kết luận chạy ByBasin Version 2 | [BasinAnalysis_summary.json](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_summary.json), [kiểm thử](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Tables/BasinAnalysis_checks.csv) |
| Học cách đọc CDF, KGE, phân mức, sự kiện, kiểm định | [Hình lý thuyết đánh giá](../06_Theory/08_EvaluationFigures.md), Mục 1–7 và 10; đây là số liệu giả minh họa |
| Bản thảo kết quả viết cho báo cáo | [Chương 6](../07_Report/06_ExperimentsAndResults.md) |

