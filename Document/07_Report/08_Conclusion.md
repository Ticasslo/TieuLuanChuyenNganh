# Chương 8. Kết luận

Đề tài dự báo lưu lượng đỉnh ngày tới (qmax) cho 451 lưu vực Trung Âu, lấy BiasCast (Konold và cs., HESS 2026) làm bài cơ sở. Đề tài tái lập bài gốc, cải tiến quy trình dữ liệu của LSTM, rồi so năm lõi thời gian (LSTM, GRU, Transformer, S4D, Mamba) dưới cùng số tham số, cùng ngân sách tinh chỉnh và 3 seed. Bài toán được mở rộng sang dự báo 1–7 ngày và cửa sổ quá khứ 2 năm, kèm giải thích mô hình và một demo công khai. Tổng cộng đề tài huấn luyện 120 lần trên GPU miễn phí. Chương này trả lời sáu câu hỏi nghiên cứu (Mục 8.1), nêu đóng góp (8.2), hạn chế (8.3) và hướng cho khóa luận (8.4).

Mọi so sánh dưới đây dùng **nhiễu seed 0,0079** làm thước đo: độ lệch chuẩn NSE trung vị validation giữa ba lần huấn luyện chỉ khác seed (B0, Chương 6 Mục 6.6.2). Hiệu nhỏ hơn mức này được coi là chưa phân biệt được.

## 8.1. Trả lời câu hỏi nghiên cứu

**RQ1. BiasCast có tái lập được không, nhiễu seed lớn cỡ nào, mô hình hơn các mốc đơn giản bao nhiêu?**

Có. Nạp trọng số của tác giả cho sáu cấu hình đều khớp NSE trung vị của bài; Sequential Forecast LSTM có Q đạt 0,7053 (A2). Huấn luyện lại từ đầu với 3 seed cho 0,6952 / 0,7038 / 0,7059, giá trị của bài nằm trong khoảng này (A4). Trên cùng tập ngày test, mô hình đạt NSE trung vị 0,705, so với 0,380 của DLinear (mô hình tuyến tính trên lưu lượng quá khứ) và 0,350 của persistence (lấy qmax hôm nay làm dự báo ngày mai) (Chương 6 Mục 6.5–6.6). Học sâu vì vậy thật sự hơn hẳn các mốc không dùng thời tiết. Persistence không bắt được trận lũ nào đúng ngày khởi phát, còn LSTM bắt được khoảng 19% (Mục 6.9.2). Nhiễu seed không nhỏ: 0,0079 NSE ở validation, cùng cỡ với nhiều khác biệt giữa các cấu hình ở Chương 6, nên so sánh chỉ một seed dễ kết luận sai.

**RQ2. Các cải tiến quy trình không phụ thuộc kiến trúc giúp LSTM bao nhiêu?**

Cải tiến rõ nhất là thêm qmax quá khứ vào đầu vào (B1b): NSE trung vị validation tăng 0,0188, bằng 2,4 lần nhiễu seed. Đổi Q quá khứ sang cùng đơn vị mm/ngày với nhãn (B1a) chỉ tăng 0,0045 (0,56 lần nhiễu), tập trung ở lưu vực nhỏ. Trên test, B1b đạt 0,7183 ± 0,0086 so với 0,7053 của tác giả (Mục 6.6.3). Nhúng theo nguồn với masked mean (B2) và che dữ liệu khi huấn luyện (B3) không làm đổi kỹ năng khi đủ dữ liệu, nhưng giúp mô hình chạy được khi một nguồn bị thiếu. Theo luật ghi trước, B3 với mức che 0,1/0,12 có NSE trung bình qua các kịch bản thiếu dữ liệu cao nhất (0,7023) nên được chọn làm LSTM cải tiến B\* (Mục 6.6.9). Đánh đổi: B\* thận trọng hơn ở đỉnh, bắt ít trận lũ hai năm hơn B1b (POD 15,1% so với 19,0%) nhưng cũng báo nhầm ít hơn khoảng 96 trận (Mục 6.9.5).

**RQ3. Khi cùng số tham số và cùng ngân sách tinh chỉnh, lõi SSM hơn, ngang hay kém LSTM, GRU, Transformer?**

Mamba **ngang** LSTM và GRU, còn S4D và Transformer **kém** hơn. Trên validation, GRU (0,7686), LSTM (0,7677) và hai cấu hình Mamba (0,7662 và 0,7630) đều nằm trong một lần nhiễu seed so với B\* (0,7669). S4D (0,740–0,742) và Transformer (0,739–0,740) kém khoảng 3,3 lần nhiễu (Mục 6.6.10). Trên test, Mamba 2 khối có NSE trung vị cao nhất (0,7209) nhưng chỉ hơn LSTM 0,0025 (Mục 6.6.11). Gấp 3 số tham số không giúp lõi nào vượt ngưỡng nhiễu. Về bắt lũ, LSTM và GRU bắt nhiều trận nhất (POD 15,4% và 15,2%), Mamba 13,5%, S4D 9,5% (Mục 6.9.5). Như vậy, ở quy mô khoảng 85 nghìn tham số và dữ liệu theo ngày, không có bằng chứng Mamba tốt hơn LSTM. Kết quả cùng chiều với Shahriar (SSRN 2026) và Zhang và cs. (J. Hydrol. 2026), và khác chiều với HydroDiffusion (Wang và cs., WRR 2026), nơi SSM nhỉnh hơn LSTM.

Theo lưu vực, mức hơn LSTM của Transformer, S4D và Mamba gấp 3 tăng theo tỉ lệ mưa dạng tuyết và giảm theo độ khô hạn, đồng nhất qua cả 3 seed (Mục 6.11.5). Điều này phù hợp một phần với giả thuyết "SSM có lợi ở lưu vực động lực chậm" (lưu vực tuyết tích nước nhiều tháng). Mamba mức cùng tham số thì không cho hướng ổn định qua các seed. Đây là tương quan mô tả, không phải bằng chứng nhân quả.

**RQ4. Khi dữ liệu đến trễ như thực tế hoặc mất Q, lõi nào suy giảm ít hơn?**

Độ trễ phát hành thật của các nguồn khí tượng (ERA5-Land thiếu 5 ngày cuối, GLEAM thiếu cả cửa sổ; kịch bản S3) hầu như không làm giảm kỹ năng ở mọi lõi: NSE trung vị test thay đổi từ −0,009 tới +0,006 so với khi đủ dữ liệu. Mất lưu lượng quan trắc mới là yếu tố quyết định. Mất Q 7 ngày cuối làm NSE giảm 0,10–0,15; mất Q cả cửa sổ (trạm hỏng dài hạn, S2) làm giảm 0,11 (LSTM) tới 0,19 (Transformer 2 lớp). Ở S2, LSTM giảm ít nhất (0,6032, giảm 0,1146), GRU và Mamba giảm 0,12–0,14, Transformer nhiều nhất (`Workspace/06_Evaluation/04_C2_LamaHCEExt_Scenarios/Archive/Version1/C2_scenarios_stats.csv`; Chương 6 Mục 6.6.12). Ý nghĩa thực tế: mô hình phụ thuộc mạnh vào số liệu trạm đo ngày hôm qua, nên độ tin cậy của đường truyền số liệu trạm quan trọng hơn độ trễ của dữ liệu khí tượng.

**RQ5. Cửa sổ quá khứ 730 ngày có giúp không, với chi phí bao nhiêu?**

Không đáng kể với LSTM và Mamba: NSE tổ hợp 3 seed đổi −0,0012 và −0,0003, đều dưới 0,16 lần nhiễu. S4D cải thiện nhẹ (+0,0036 tổ hợp; 266/451 lưu vực tốt lên), nhiều hơn ở lưu vực lớn và khô hạn, nhưng vẫn kém LSTM. Chi phí gần gấp đôi: thời gian huấn luyện mỗi epoch 1,95–1,96 lần, bộ nhớ GPU 1,69–1,75 lần; suy luận trên CPU 1,50 lần (LSTM) tới 2,42 lần (Mamba) (Mục 6.6.14). Lý do khả dĩ được bước G xác nhận: LSTM, GRU và Mamba dồn 90% ảnh hưởng của pha quá khứ vào 1–2 tháng gần nhất (Mục 6.12.2), nên năm thứ hai gần như không được dùng. Lợi thế lý thuyết "Mamba có độ phức tạp tuyến tính theo độ dài chuỗi" không chuyển thành lợi ích kỹ năng ở bài toán này.

**RQ6. Khi dự báo 1–7 ngày, kỹ năng suy giảm theo hạn ra sao và dự báo thời tiết đóng góp bao nhiêu?**

Kỹ năng giảm nhanh nếu thiếu thời tiết tương lai. NSE trung vị test của LSTM từ hạn 1 tới hạn 7 là 0,49 → 0,09 khi chỉ dùng quá khứ, 0,60 → 0,08 khi có ECMWF HRES ngày đầu (cách lai), 0,63 → 0,24 khi có dự báo GEFSv12 7 ngày, và 0,73 → 0,64 khi biết trước thời tiết thật (cận trên); Mamba tương tự (`Workspace/05_Training/15_H_LamaHCEExt_Compare/15_H_LamaHCEExt_Compare.md`; Chương 6 Mục 6.6.13). ECMWF HRES ngày đầu chỉ giúp hạn 1–2; từ hạn 2, GEFSv12 là phần đóng góp lớn nhất. Khoảng cách tới cận trên tăng theo hạn, nên phần lớn kỹ năng còn thiếu nằm ở chất lượng dự báo thời tiết chứ không ở mô hình thủy văn. Giải thích mô hình củng cố điều này: ở cách GEFS, tỉ phần ảnh hưởng của thông tin tương lai tăng từ 9% lên 30% theo hạn (LSTM), gần bằng cách hoàn hảo (Mục 6.12.3). Hai lõi cách nhau tối đa 2,5 lần nhiễu khi lấy trung bình 7 hạn; ở hạn 7, Mamba kém LSTM 2,9–4,2 lần nhiễu ở cả ba cách vận hành được.

## 8.2. Đóng góp

1. **So sánh có kiểm soát năm lõi** trong bài toán dự báo vận hành dùng dự báo thời tiết ECMWF HRES lưu trữ thật và lưu lượng quan trắc: cùng số tham số (lệch dưới 1%), cùng ngân sách tinh chỉnh, 3 seed, kết luận dựa trên độ lớn hiệu so với nhiễu seed chứ không chỉ p-value.
2. **Tái lập và chuẩn hóa đánh giá BiasCast:** đo nhiễu seed, bổ sung mốc persistence, PNSE và DLinear mà bài gốc không có; sửa lỗi lưu mô hình tốt nhất và bộ lọc mẫu trong mã; phát hiện Q đầu vào của tác giả theo m³/s trong khi nhãn theo mm/ngày.
3. **Bộ đánh giá sát vận hành:** kịch bản thiếu dữ liệu theo độ trễ thật của từng nguồn, đánh giá theo bốn mức lưu lượng như bài toán phân loại, và theo sự kiện lũ ở các ngưỡng chu kỳ lặp lại 1,5–10 năm, áp dụng cho mọi lõi.
4. **Mở rộng sang dự báo 1–7 ngày** với bốn cách cấp thời tiết tương lai, tách được đóng góp của dự báo ngày đầu, của dự báo nhiều ngày thật và khoảng cách còn lại tới dự báo hoàn hảo.
5. **Giải thích mô hình** bằng Integrated Gradients có kiểm completeness và deletion test: đo độ dài nhớ hiệu dụng của năm lõi và vai trò thông tin tương lai theo hạn.
6. **Demo công khai** tại `https://ticasslo.dev`: 21 đường dự báo cho 451 lưu vực, 7 mô hình chạy trực tiếp trên CPU, số trên trang được kiểm trùng với số đã chấm (Chương 7).

## 8.3. Hạn chế

- **Phạm vi dữ liệu:** chỉ 451 lưu vực Trung Âu theo ngày, kỳ test 2014–2017. Kết luận chưa được kiểm ở khu vực khác, kể cả Việt Nam, và ở độ phân giải giờ.
- **Quy mô mô hình:** chỉ khoảng 85 nghìn và 255 nghìn tham số. Kết luận "Mamba ngang LSTM" có thể đổi ở mô hình lớn hơn hoặc chuỗi dài hơn nhiều.
- **Ngân sách tinh chỉnh hạn chế:** 8 cấu hình một seed cho mỗi lõi, so với 100 lần tối ưu Bayes cho LSTM ở bài gốc. Một lõi có thể kém chỉ vì lưới chưa phủ vùng siêu tham số tốt của nó. Lưới tinh chỉnh như nhau giúp so sánh công bằng nhưng không bảo đảm mỗi lõi đạt tối ưu.
- **Bắt lũ còn yếu:** khi bắt buộc đúng ngày khởi phát, mọi mô hình chỉ bắt 9,5–19,0% trận lũ hai năm; mọi mô hình hụt đỉnh (%BiasFHV trung vị âm). Hàm mất mát NSE\* thưởng việc khớp cả chuỗi, không ưu tiên đỉnh lũ.
- **Dự báo nhiều ngày:** GEFSv12 reforecast lấy trung bình 5 thành viên, độ phân giải thô hơn ECMWF HRES. Ba cách vận hành được của bước H đạt đỉnh validation ở epoch 1–5 rồi quá khớp, nên giao thức 30 epoch chưa hợp với bài toán này.
- **Tái lập từng bit:** Transformer và Mamba không lặp lại từng bit trên GPU khi chạy lại cùng seed (lệch NSE tới 0,011 ở Mamba gấp 3), nên một phần khác biệt nhỏ giữa các lõi này còn chịu nhiễu cài đặt.
- **Bằng chứng mô tả:** tương quan thuộc tính lưu vực và Integrated Gradients cho biết mô hình nhạy với gì, không chứng minh quan hệ nhân quả.
- **Không phải hệ thống cảnh báo:** đầu ra là lưu lượng, không phải mực nước hay vùng ngập. Demo chỉ phát lại quá khứ. Dữ liệu dùng giấy phép phi thương mại, và số liệu lưu lượng của Cộng hòa Séc không được dùng cho cảnh báo vận hành theo điều kiện của CHMI (`02_Pipeline.md` Mục 1.6).

## 8.4. Hướng phát triển cho khóa luận

- **Dữ liệu giờ:** ghép lưu lượng và khí tượng theo giờ cho quá khứ gần (MF-LSTM); dữ liệu LamaH-CE theo giờ đã được tải sẵn.
- **Dự báo nhiều ngày có xác suất:** dùng từng thành viên GEFSv12 thay vì trung bình, cùng đầu ra xác suất CMAL (Klotz và cs., HESS 2022) để ước xác suất vượt ngưỡng lũ.
- **Hàm mất mát ưu tiên đỉnh lũ** và lấy mẫu nhiều ngày lưu lượng cao, có kiểm đánh đổi với kỹ năng chung (Baste 2025; Martel 2025).
- **Biến thể kiến trúc:** Mamba hai chiều trong cửa sổ quá khứ, Transformer chia đoạn kiểu PatchTST, mô hình lai Mamba–Transformer, xLSTM.
- **Đồ thị mạng sông:** nối các lưu vực theo quan hệ thượng–hạ lưu bằng GAT hoặc đồ thị reachability; chưa tìm thấy công trình kết hợp GNN với Mamba cho dòng chảy (`02_Pipeline.md` Mục 11).
- **Phần mềm ứng dụng:** nâng demo thành hệ thống chạy gần thời gian thực với nguồn trực tiếp (eHYD, dữ liệu mở ECMWF), trang từng lưu vực, cảnh báo và API.
