# Chương 1. Mở đầu

## 1.1. Bối cảnh

**Lưu lượng dòng chảy** (discharge, ký hiệu Q) là thể tích nước chảy qua một mặt cắt sông trong một đơn vị thời gian, đo bằng m³/s tại các trạm đo. Dự báo Q vài giờ tới vài ngày trước là đầu vào của cảnh báo lũ, vận hành hồ chứa và quản lý nguồn nước. Cách truyền thống dùng mô hình thủy văn khái niệm hoặc vật lý, mô tả quá trình mưa thấm vào đất, tan tuyết, chảy tràn và chảy trong sông bằng các phương trình có tham số hiệu chỉnh riêng cho từng lưu vực.

Từ cuối thập niên 2010, mạng học sâu cạnh tranh trực tiếp với cách truyền thống. Kratzert và cs. (HESS 2019) huấn luyện **một** mạng LSTM chung cho hàng trăm lưu vực, mỗi lưu vực được mô tả bằng thuộc tính tĩnh (diện tích, độ cao, khí hậu, đất…), và đạt kỹ năng cao hơn các mô hình thủy văn hiệu chỉnh riêng từng lưu vực. Hệ thống dự báo lũ toàn cầu của Google cũng dùng LSTM (Nearing và cs., Nature 2024). LSTM vì vậy trở thành mốc chuẩn của học sâu thủy văn.

Hai vấn đề còn mở là nền của đề tài:

- **Khoảng cách giữa nghiên cứu và vận hành.** Phần lớn nghiên cứu huấn luyện và đánh giá mô hình trên dữ liệu khí tượng tái phân tích, tức dữ liệu quá khứ đã được hiệu chỉnh kỹ. Khi vận hành, mô hình phải dùng dự báo thời tiết, kém chính xác hơn. BiasCast (Konold và cs., HESS 30:5067–5096, 2026) đo được mức suy giảm này trên 451 lưu vực Trung Âu: mô hình học trên tái phân tích rồi chạy với dự báo ECMWF HRES thật giảm NSE trung vị từ 0,58 xuống 0,33. BiasCast đề xuất huấn luyện trực tiếp với dự báo thật và lưu lượng quan trắc gần nhất, đạt 0,71. Tác giả chỉ thử các biến thể LSTM và tự nêu đây là hạn chế (BiasCast Mục 3.7).
- **Kiến trúc mới chưa được kiểm chứng đủ.** Mô hình không gian trạng thái (state space model, SSM) như S4D (Gu và cs., NeurIPS 2022) và Mamba (Gu & Dao, 2023) xử lý chuỗi dài với chi phí tuyến tính theo độ dài, và cạnh tranh với Transformer ở nhiều bài toán chuỗi. Với dòng chảy, bằng chứng chưa thống nhất: có công trình thấy SSM nhỉnh hơn LSTM (HydroDiffusion, Wang và cs., WRR 2026), có công trình thấy LSTM vẫn thắng khi so cùng số tham số (Shahriar, SSRN 2026; Zhang và cs., J. Hydrol. 2026) (Chương 3; `05_Survey/02_LiteratureReview.md` Mục 1). Các so sánh này đều ở bài toán mô phỏng hoặc không dùng dự báo thời tiết lưu trữ thật kèm lưu lượng quan trắc.

Đề tài nằm ở giao của hai vấn đề: trong đúng khung dự báo vận hành của BiasCast, thay lõi LSTM bằng các kiến trúc khác và so sánh có kiểm soát.

## 1.2. Bài toán

Với mỗi trạm đo trong 451 lưu vực của bộ dữ liệu Extended LamaH-CE (Áo và các vùng lân cận ở Đức, Thụy Sĩ, Séc…), dự báo **lưu lượng đỉnh ngày t** (qmax, giá trị lớn nhất trong ngày) vào đầu ngày t. Mô hình nhận:

- dữ liệu khí tượng và lưu lượng đo của 364 ngày trước;
- dự báo thời tiết ECMWF HRES cho ngày t, phát hành 00 UTC ngày t;
- 33 thuộc tính tĩnh của lưu vực.

Một mô hình chung được huấn luyện cho cả 451 lưu vực trên kỳ 2003–2009, chọn trên kỳ 2010–2013 và kiểm tra trên kỳ 2014–2017. Đề tài mở rộng bài toán theo hai hướng: dự báo qmax 1–7 ngày tới (bước H) và nhìn lại 730 ngày thay vì 365 (bước D).

Đầu ra là lưu lượng, không phải mực nước hay vùng ngập. "Nguy cơ lũ" chỉ được suy gián tiếp qua các ngưỡng lưu lượng theo chu kỳ lặp lại (return period) của từng trạm, ví dụ mức lưu lượng trung bình 2 năm mới bị vượt một lần (Chương 4 Mục 4.6.2). Quy đổi sang mực nước cần đường quan hệ mực nước – lưu lượng riêng từng mặt cắt sông, ngoài phạm vi đề tài.

## 1.3. Mục tiêu và câu hỏi nghiên cứu

Mục tiêu của đề tài là trả lời có kiểm soát câu hỏi: *trong dự báo dòng chảy vận hành, lõi SSM có đáng thay LSTM không, ở đâu và với điều kiện nào?* Đề tài không xuất phát từ giả định Mamba tốt hơn; kết quả Mamba ngang hoặc kém LSTM cũng là câu trả lời có giá trị nếu giao thức chặt.

| Mã | Câu hỏi | Bước |
|---|---|---|
| RQ1 | Kết quả BiasCast có tái lập được không, nhiễu giữa các seed lớn cỡ nào, và mô hình hơn các mốc đơn giản (persistence, mô hình tuyến tính) bao nhiêu? | A |
| RQ2 | Các cải tiến quy trình không phụ thuộc kiến trúc (thống nhất đơn vị Q, thêm qmax quá khứ, nhúng theo nguồn, che dữ liệu khi huấn luyện) cải thiện LSTM bao nhiêu và giúp mô hình chịu mất dữ liệu ra sao? | B |
| RQ3 | Khi cùng số tham số, cùng ngân sách tinh chỉnh, cùng seed, lõi SSM (S4D, Mamba) hơn, ngang hay kém LSTM, GRU, Transformer? Ở dải lưu lượng nào, loại lưu vực nào? | C |
| RQ4 | Khi dữ liệu đến trễ như thực tế vận hành hoặc mất lưu lượng quan trắc, lõi nào suy giảm ít hơn? | Kịch bản S0–S4 |
| RQ5 | Cửa sổ quá khứ 730 ngày có giúp không, với chi phí tính toán bao nhiêu? | D |
| RQ6 | Khi dự báo 1–7 ngày, kỹ năng của LSTM và Mamba suy giảm theo hạn ra sao, và dự báo thời tiết đóng góp bao nhiêu ở từng hạn? | H |

## 1.4. Đóng góp

1. **So sánh có kiểm soát** LSTM, GRU, Transformer, S4D và Mamba trong khung dự báo dùng dự báo ECMWF HRES lưu trữ thật và lưu lượng quan trắc, trên 451 lưu vực: số tham số lệch dưới 1%, cùng 8 cấu hình tinh chỉnh mỗi lõi, 3 seed, so độ lớn hiệu với nhiễu seed.
2. **Tái lập BiasCast và chuẩn hóa đánh giá:** đo nhiễu seed; bổ sung mốc persistence, PNSE và DLinear; phát hiện và sửa hai điểm yếu trong mã (bộ lọc loại cả mẫu khi thiếu một giá trị; hàm lưu mô hình tốt nhất); phát hiện lưu lượng đầu vào của tác giả ở đơn vị khác nhãn.
3. **Bộ đánh giá sát vận hành:** kịch bản dữ liệu thiếu theo độ trễ phát hành thật của từng nguồn và theo mất lưu lượng quan trắc; đánh giá theo bốn mức lưu lượng như bài toán phân loại; đánh giá theo sự kiện lũ ở ngưỡng chu kỳ lặp lại 1,5; 2; 5; 10 năm.
4. **Mở rộng sang dự báo 1–7 ngày**, so bốn cách cấp thời tiết tương lai (không có, chỉ ngày đầu, dự báo GEFSv12 7 ngày, thời tiết hoàn hảo).
5. **Giải thích mô hình** bằng Integrated Gradients và dò trạng thái ẩn: độ dài nhớ hiệu dụng của năm lõi, vai trò của từng nguồn dữ liệu và của thông tin tương lai theo hạn.
6. **Demo công khai** tại `https://ticasslo.dev`, phát lại dự báo của mọi mô hình cho 451 lưu vực và chạy trực tiếp mô hình trên CPU cho thử nghiệm "nếu… thì…".

## 1.5. Phạm vi

- Dữ liệu: Extended LamaH-CE theo ngày (Zenodo 17119635, giấy phép CC BY-NC 4.0), dùng cho nghiên cứu và giáo dục phi thương mại.
- Mô hình: khoảng 85 nghìn và 255 nghìn tham số, huấn luyện trên GPU T4 miễn phí của Kaggle; tổng 120 lần huấn luyện.
- Ngoài phạm vi tiểu luận: dữ liệu giờ, đầu ra xác suất, đồ thị mạng sông, biến thể cải tiến của Mamba và Transformer. Các hướng này để cho khóa luận (Chương 8 Mục 8.4).
- Demo là công cụ minh họa, phát lại kỳ test 2014–2017, không thay hệ thống cảnh báo chính thức.

## 1.6. Bố cục báo cáo

| Chương | Nội dung |
|---|---|
| 2. Cơ sở lý thuyết | Năm lõi thời gian (LSTM, GRU, Transformer, S4D, Mamba) và các chỉ số đánh giá |
| 3. Công trình liên quan | BiasCast và các công trình gần nhất về SSM, Mamba trong dự báo dòng chảy |
| 4. Dữ liệu | Extended LamaH-CE, khảo sát thống kê, ngưỡng mức lưu lượng và ngưỡng lũ |
| 5. Phương pháp | Khung dự báo, khối giữ và khối sửa, giao thức so sánh cùng tham số, các mở rộng |
| 6. Thí nghiệm và kết quả | Bước A, B, C, H, D; đánh giá theo chỉ số, mức, sự kiện, kịch bản, lưu vực; giải thích mô hình |
| 7. Demo | Kiến trúc, giao diện, kiểm đúng và triển khai trang web |
| 8. Kết luận | Trả lời RQ1–RQ6, đóng góp, hạn chế, hướng khóa luận |
