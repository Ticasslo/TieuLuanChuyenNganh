# Giới thiệu bài cơ sở — Kirschstein & Sun (ICML 2024)

> Phương án thay thế trong đề xuất bài cơ sở (`TongQuan/ChonBaiCoSo.md`); bài đề xuất chính là BiasCast (`TongQuan/GioiThieuBaiCoSo_BiasCast.md`). Tài liệu trình bày bài báo làm bài cơ sở cho đề tài *Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy*: nội dung bài, mô hình, chỉ số đánh giá, kết quả, lý do chọn, hạn chế phát hiện khi đọc mã nguồn và hướng cải tiến. Chi tiết đánh giá các ứng viên: `BaiCoSo/CHECKPDF.md`, `BaiCoSo/CHECKCODE.md`. Kiến trúc và pipeline của đề tài: `TongQuan/KienTrucPipeline.md`.

## 1. Thông tin bài báo

| Mục | Nội dung |
|---|---|
| Tên bài | *The Merit of River Network Topology for Neural Flood Forecasting* |
| Tác giả | Nikolas Kirschstein, Yixuan Sun (University of Oxford; Technical University of Munich) |
| Nơi công bố | Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, tr. 24713–24725 |
| Xếp hạng | ICML — hội nghị CORE A*, ngành học máy |
| Trích dẫn | 21 (Google Scholar, 25/9/2026) |
| Trang chính thức | https://proceedings.mlr.press/v235/kirschstein24a.html |
| Phản biện | OpenReview `QE6iC9s6vU`, ghi "ICML 2024 Poster" — https://openreview.net/forum?id=QE6iC9s6vU |
| arXiv | 2405.19836 |
| Mã nguồn | https://github.com/nkirschi/neural-flood-forecasting (kèm 957 checkpoint) |
| DOI | Không có DOI đăng ký; trích dẫn theo PMLR |

Trích dẫn: Kirschstein, N., & Sun, Y. (2024). The Merit of River Network Topology for Neural Flood Forecasting. *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 24713–24725.

## 2. Tóm tắt

Bài dự báo lưu lượng sông (Q, m³/s) theo giờ đồng thời cho 358 trạm đo trên lưu vực sông Danube (bộ dữ liệu LamaH-CE). Câu hỏi nghiên cứu: đưa cấu trúc mạng lưới sông vào mô hình bằng mạng nơ-ron đồ thị (GNN) có giúp dự báo tốt hơn không. Kết luận của tác giả: đồ thị gần như không giúp — mô hình bỏ hết cạnh hoặc một MLP đơn giản đạt kết quả ngang các GNN.

Điểm quan trọng đối với đề tài: phần xử lý chuỗi thời gian của mô hình chỉ là một lớp tuyến tính (affine) trên 24 giờ gần nhất. Đây là vị trí đề tài thay bằng mô hình học chuỗi (Mamba, đối chứng với LSTM, GRU, Transformer, S4D).

## 3. Bài toán

Hồi quy trên nút của đồ thị: mỗi nút là một trạm đo, mỗi cạnh nối một trạm với trạm hạ lưu kế tiếp.

- **Đầu vào:** W = 24 giờ gần nhất của mọi trạm, mỗi giờ 5 biến — lưu lượng Q quan trắc, lượng mưa, độ ẩm đất lớp mặt, nhiệt độ không khí, áp suất bề mặt.
- **Đầu ra:** lưu lượng Q của mọi trạm sau L = 6 giờ.

Đây là bài toán **dự báo**: dùng dữ liệu tới thời điểm t, có Q quan trắc quá khứ, ước lượng Q sau t. Bài toán **mô phỏng** thì chỉ dùng khí tượng (tính cả thời điểm hiện tại) để ước lượng Q của chính thời điểm đó, không dùng Q quan trắc; mô phỏng phục vụ lưu vực không có trạm đo và kịch bản khí hậu, còn dự báo phục vụ cảnh báo. Dự báo khớp với tên đề tài.

## 4. Dữ liệu và cách chia tập

- **LamaH-CE** (Klingler và cs., Earth System Science Data 13:4529–4565, 2021; DOI 10.5194/essd-13-4529-2021; dữ liệu Zenodo 10.5281/zenodo.5153305; giấy phép CC BY-SA 4.0): dữ liệu theo giờ của 859 trạm vùng Trung Âu, diện tích khoảng 170.000 km², có sẵn mạng lưới sông.
- LamaH-CE trải trên 9 quốc gia, từ vùng đồng bằng khí hậu lục địa tới vùng núi cao Alps, nhưng mạng sông của nó gồm 4 thành phần liên thông (Hình 2 và Mục 3.1 của bài). Bài chỉ lấy thành phần liên thông lớn nhất "Danube A" (608 trạm); loại trạm có đoạn thiếu dài hoặc không đủ dữ liệu 2000–2017, còn **358 trạm**. Khi xóa trạm, các trạm thượng và hạ lưu của nó được nối lại để mạng vẫn liên thông.
- Chuẩn hóa z-score riêng từng trạm.
- **Tập kiểm tra:** 2016–2017. **Tập huấn luyện:** 8 năm, thử 3 cách chọn — các năm chẵn 2000–2015, các năm lẻ 2000–2015, các năm liên tục 2008–2015. Kết quả báo cáo là trung bình ± độ lệch chuẩn qua 3 cách chia.
- **Validation:** 1/5 số cửa sổ huấn luyện, chọn ngẫu nhiên.

Vị trí trong bài: Mục 3.1 (tr. 2–3).

## 5. Mô hình

Kiến trúc "bánh kẹp" gồm ba khối nối tiếp:

| Khối | Chức năng | Chi tiết |
|---|---|---|
| Mã hóa | Lớp affine | Làm phẳng 24 giờ × 5 biến của từng trạm thành vector 128 chiều; không học quan hệ theo thời gian |
| Xử lý không gian | 19 lớp GNN, kích hoạt ReLU | Truyền thông tin giữa các trạm theo mạng sông; 19 lớp vì đường đi dài nhất của đồ thị có 19 cạnh |
| Giải mã | Lớp affine | Chiếu vector ẩn ra Q sau 6 giờ của từng trạm |

Các biến thể được so sánh:

- **3 loại lớp GNN:** ResGCN (GCN có nối tắt), GCNII, ResGAT (GAT có nối tắt, dùng cơ chế attention).
- **6 cách định nghĩa cạnh:** cô lập (bỏ hết cạnh), nhị phân (có/không nối), trọng số theo chiều dài dòng sông, theo chênh lệch cao độ, theo độ dốc trung bình, trọng số do mô hình tự học. Với ResGAT, trường hợp tự học được thay bằng việc đưa cả ba trọng số vật lý cùng lúc.
- **3 hướng truyền:** xuôi dòng, ngược dòng, hai chiều.
- Tổng cộng 18 cấu hình đồ thị × 3 loại lớp × 3 cách chia tập.
- **Mốc so sánh:** MLP 2 lớp × 512 chiều, không dùng đồ thị.

Siêu tham số (Bảng 1, tr. 5):

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Dữ liệu | Cửa sổ W / lead time L | 24 giờ / 6 giờ |
| Mô hình | Kiến trúc mặc định, số lớp, số chiều ẩn | GCNII, 19 lớp, d = 128 |
| Huấn luyện | Khởi tạo, tối ưu, epoch, batch, learning rate, điều chuẩn L2 | Kaiming, Adam, 100, 64, 10⁻⁴, 10⁻⁵ |
| Chọn mô hình | Epoch có loss validation nhỏ nhất | Validation 1/5 cửa sổ train, chọn ngẫu nhiên |

## 6. Hàm mất mát và chỉ số đánh giá

**Điểm liên quan (relevancy score).** Dữ liệu có nhiều giai đoạn lưu lượng gần như không đổi. Tác giả gán cho mỗi cửa sổ một trọng số dựa trên tốc độ thay đổi lưu lượng và tổng lượng nước trong cửa sổ so với lưu lượng trung bình của trạm (tốc độ thay đổi được tính nặng gấp đôi). Cửa sổ biến động mạnh, lưu lượng cao được ưu tiên.

**Hàm mất mát.** MSE nhân với điểm liên quan, cộng điều chuẩn L2.

**Chỉ số đánh giá: NSE có trọng số.** Nash–Sutcliffe Efficiency (NSE) là chỉ số chuẩn của ngành thủy văn, so sai số của mô hình với sai số của cách dự báo đơn giản nhất là luôn đoán bằng lưu lượng trung bình: NSE = 1 − Σ(Q dự báo − Q thực)² / Σ(Q trung bình − Q thực)².

- NSE = 1: dự báo hoàn hảo. NSE = 0: ngang đoán bằng trung bình. NSE < 0: tệ hơn đoán bằng trung bình.
- Bài nhân mỗi số hạng với điểm liên quan, tính NSE cho từng trạm trên lưu lượng đã khử chuẩn hóa, rồi lấy trung bình 358 trạm.
- Bài ghi NSE dạng phần trăm: 85,37% nghĩa là NSE = 0,8537.
- Chú thích 3 của bài: NSE không trọng số cho kết luận tương tự nhưng giá trị cao hơn, vì các cửa sổ ít biến động chiếm đa số.

Vị trí trong bài: Mục 3.2, các đoạn "Relevancy Score", "Optimisation Objective", "Testing Metric" (tr. 4).

**Các phân tích bổ sung:**

- Tương quan Pearson giữa trọng số cạnh mô hình tự học và trọng số vật lý (Bảng 3, tr. 7).
- Lặp lại thí nghiệm trên 4 mạng con nhỏ với số chiều ẩn 512 (Bảng A.4–A.7).
- Phân tích trạm dự báo kém nhất (Mục 4.5, tr. 7).
- Ảnh hưởng của độ dài cửa sổ (12–72 giờ) và lead time (1–12 giờ) (Bảng A.8, tr. 13).

## 7. Kết quả của bài

| Mô hình / cấu hình | NSE có trọng số |
|---|---|
| MLP 2 lớp (không đồ thị) | 85,37% ± 1,64% |
| GCNII, trọng số tự học, hai chiều (cao nhất) | 85,56% ± 1,41% |
| ResGCN, cô lập (bỏ hết cạnh) | 85,07% ± 0,66% |
| Khoảng của mọi tổ hợp đồ thị | 80,2% – 85,6% |

- Không tổ hợp đồ thị nào vượt MLP quá độ lệch chuẩn; bỏ hết cạnh cũng không làm giảm hiệu năng quá độ lệch chuẩn. Kết luận của tác giả: cấu trúc mạng sông gần như không giúp.
- Trọng số cạnh tự học không tương quan với trọng số vật lý, dấu tương quan còn đảo chiều giữa các kiến trúc.
- Mô hình bỏ lỡ các đỉnh lũ đột ngột, hẹp (trạm kém nhất).
- **Cửa sổ dài hơn giúp rõ rệt ở lead time dài** (Bảng A.8): lead 12 giờ tăng từ 63,16% (cửa sổ 12 giờ) lên 75,52% (cửa sổ 72 giờ); lead 6 giờ tăng từ 82,35% lên 87,59%. Tác giả nêu cửa sổ lớn hơn bị giới hạn bởi chi phí tính toán của lớp mã hóa affine (Mục 4.1).

Vị trí trong bài: Bảng 2 (tr. 6), Bảng 3 (tr. 7), Mục 4.5 (tr. 7), Bảng A.8 (tr. 13).

## 8. Lý do chọn bài (khi dùng làm bài cơ sở)

**Quá trình chọn:**

1. Tra 137 bài báo từ 2024 có huấn luyện mô hình học sâu dự báo lưu lượng trên dữ liệu công khai dài năm; loại MDPI, preprint, bài chỉ công bố dữ liệu và bài tổng quan.
2. Đọc mã nguồn 35 bài; xếp hạng 13 ứng viên theo 3 nhóm tiêu chí: quyết định (đúng bài toán dự báo, có chỗ cải tiến), điều kiện cần (mã chạy được, dữ liệu miễn phí, chạy được trên GPU miễn phí), điểm cộng (phương pháp chặt chẽ, uy tín nơi công bố, hỗ trợ demo).
3. Đọc toàn văn và mã của 5 ứng viên đứng đầu; chạy thử mã của MF-LSTM và S4D-FT trên dữ liệu giả; mã của bài cơ sở được đối chiếu với tệp kết quả và checkpoint tác giả công bố.

**Lý do:**

1. **Đúng bài toán dự báo lưu lượng theo giờ, có dùng lưu lượng quan trắc quá khứ.** BiasCast (Konold và cs., HESS 2026 — bài đề xuất chính) dự báo có cả lưu lượng quan trắc và dự báo thời tiết thật trên cùng LamaH-CE nhưng theo ngày, không có đồ thị; ứng viên hạng 3 (mô hình của Google FloodHub) dự báo theo ngày và không dùng lưu lượng quan trắc; hai ứng viên hạng 4 và 5 là bài toán mô phỏng.
2. **Chỗ cải tiến rõ và có động cơ từ chính bài.** Khối thời gian chỉ là lớp affine, trong khi Bảng A.8 cho thấy cửa sổ dài hơn cải thiện kết quả nhưng bị giới hạn chi phí. Mamba có chi phí tăng tuyến tính theo độ dài chuỗi, phù hợp để kéo dài cửa sổ. Chưa có công trình nào đưa Mamba vào khung này.
3. **Hai câu hỏi nghiên cứu thay vì một:** Mamba có hơn LSTM, GRU, Transformer ở phần thời gian không; và khi đã có bộ mã hóa thời gian tốt, đồ thị mạng sông có còn giúp không.
4. **Tái lập được:** mã công khai, kèm 957 checkpoint (162 cho thí nghiệm chính ở Bảng 2, 3 cho MLP, 108 cho ablation, 684 cho mạng con), tính lại kết quả mà không cần huấn luyện lại.
5. **Uy tín:** ICML là hội nghị hạng CORE A* ngành học máy.
6. **Hỗ trợ demo:** 358 trạm có tọa độ và mạng sông, làm được bản đồ dự báo theo giờ.
7. **Chạy được trên GPU miễn phí** (Kaggle, Google Colab).

**So với các ứng viên xếp sau:**

| Tiêu chí | Kirschstein & Sun, ICML 2024 | Google FloodHub — Nearing và cs., Nature 2024; Gauch và cs., HESS 2025 | MF-LSTM — Acuña Espinoza và cs., HESS 2025 | S4D-FT — Wang và cs., WRR 2025 |
|---|---|---|---|---|
| Bài toán | Dự báo 6 giờ | Dự báo 0–7 ngày bằng khí tượng dự báo, không dùng Q quan trắc | Mô phỏng theo giờ | Mô phỏng theo ngày |
| Mô hình gốc | GNN, MLP | LSTM hindcast–forecast | LSTM đa tần suất | SSM S4D |
| Dữ liệu | LamaH-CE, 358 trạm, có mạng sông | Caravan + MultiMet, theo ngày; dự báo thời tiết thật chỉ có 2012–2022 | CAMELS-US, 516 lưu vực, không có mạng sông | CAMELS-US, 531 lưu vực |
| Phương pháp đánh giá | Yếu (Mục 9) | Mã tốt; không có số liệu công bố tái lập được trên dữ liệu công khai | Chặt chẽ nhất: chia theo thời gian, NSE chuẩn, 10 hạt giống | Trung bình |
| Chi phí huấn luyện | Nhẹ | Chưa đo | ~7,9 giờ/lần trên V100 | ~10 giờ/lần trên L40S, tổ hợp 8 lần |
| Vai trò trong đề tài | Bài cơ sở | Ý tưởng đầu ra xác suất, tách hindcast/forecast; khung cho bộ dữ liệu thứ hai | Giao thức đánh giá; bộ dữ liệu thứ hai | Mô hình S4D làm đối chứng SSM |

Bài FloodGNNs (Wang, Chen, Zheng, Song — npj Natural Hazards 2025) dùng cùng 358 trạm và tiền xử lý theo bài cơ sở, nhưng mã không chạy nguyên trạng, không có bảng số liệu và không có checkpoint; được dùng làm baseline đồ thị dày.

## 9. Hạn chế phát hiện khi đọc mã nguồn

- **Lỗi công thức NSE.** Hàm `evaluate_nse` lấy lưu lượng trung bình ở đơn vị gốc (m³/s) trừ cho nhãn đã chuẩn hóa. Mẫu số NSE bị thổi phồng ở trạm có lưu lượng lớn, đẩy NSE sát 1: 28% trạm có NSE > 0,99 trong tệp kết quả tác giả công bố. Giá trị NSE tuyệt đối trong bài vì vậy không dùng được. Thứ tự giữa các mô hình trên cùng một trạm được giữ nguyên, nhưng thứ tự sau khi lấy trung bình qua các trạm có thể đổi, nên kết luận "đồ thị không hơn MLP" cần tính lại bằng checkpoint để xác nhận.
- **Nhiễu giữa các lần huấn luyện lớn cỡ toàn bộ chênh lệch trong Bảng 2.** Theo mã, cấu hình ResGAT "all" và ResGAT "binary" giống hệt nhau về chức năng (cùng tập cạnh, cùng khởi tạo, trọng số cạnh bị bỏ qua), nhưng NSE trung bình theo fold trong tệp kết quả của tác giả lệch nhau tới 6,67 điểm %, trong khi toàn bộ Bảng 2 chỉ trải trong 80,2%–85,6%. Mỗi cấu hình chỉ chạy 1 lần, nên chênh lệch giữa các dòng Bảng 2 không phân biệt được với nhiễu: kết luận chắc chắn của bài là "không phát hiện khác biệt", chưa đủ để khẳng định đồ thị không giúp.
- Mức thổi phồng NSE tăng theo lưu lượng trung bình của trạm, nên chênh lệch giữa các mô hình ở trạm sông lớn, hạ lưu — nơi thông tin thượng nguồn được kỳ vọng giúp nhiều nhất — bị nén gần về 0 khi lấy trung bình (suy luận từ công thức, cần tính lại để xác nhận). Trạm "kém nhất" ở Mục 4.5 cũng được chọn theo NSE bị lỗi này.
- Điểm liên quan trong mã được tính trung bình trên cả batch thay vì từng trạm như công thức trong bài.
- ResGAT không dùng trọng số cạnh vật lý (lớp `GATConv` không khai báo `edge_dim`), nên các dòng ResGAT có trọng số trong Bảng 2 thực chất gần như đồ thị nhị phân.
- Validation là 1/5 cửa sổ chọn ngẫu nhiên trong cùng kỳ huấn luyện; các cửa sổ chồng lấn nhau nên không độc lập với tập huấn luyện.
- Mỗi cấu hình chỉ chạy 1 hạt giống ngẫu nhiên.
- Bộ nạp dữ liệu cắt cửa sổ trong từng năm, nên cửa sổ từ 1 năm trở lên không chạy được.
- Mốc trung bình trong NSE là lưu lượng trung bình 2000–2015 của trạm, không phải trung bình kỳ kiểm tra như NSE chuẩn; bài ghi NSE nằm trong [0, 1] trong khi tệp kết quả của tác giả có trạm NSE âm (thấp nhất −1,98).
- Mục 4.2 gọi baseline là "MLP 19 lớp", chú thích Bảng 2 ghi "MLP 2 lớp"; mã dùng 2 lớp × 512 chiều.

Các hạn chế trên đều sửa được, và việc sửa là một phần đóng góp của đề tài: tính lại NSE đúng từ checkpoint của tác giả, dùng validation chia theo thời gian, chạy nhiều hạt giống, viết lại bộ nạp dữ liệu cho cửa sổ dài.

## 10. Hướng cải tiến

**Câu hỏi nghiên cứu:** với dự báo lưu lượng theo giờ trên mạng sông, bộ mã hóa thời gian dạng SSM (Mamba) với cửa sổ dài hơn có cải thiện so với lớp affine của bài gốc không, và đồ thị mạng sông còn đóng góp gì khi đã có bộ mã hóa thời gian tốt. Kết quả khẳng định hay phủ định đều là câu trả lời hợp lệ; đề tài đặt vấn đề là nghiên cứu so sánh, không mặc định Mamba phải thắng.

**Phần cốt lõi:**

1. Tái lập bài gốc, sửa công thức NSE và tính lại kết quả từ checkpoint để có mốc so sánh chính xác.
2. Thay lớp affine bằng Mamba làm bộ mã hóa thời gian, kéo cửa sổ từ 24 giờ lên dài hơn; giữ nguyên phần GNN.
3. So sánh MLP, GNN gốc, LSTM, GRU, Transformer, S4D và Mamba, mỗi mô hình chạy cả có và không có đồ thị, cùng điều kiện, validation chia theo thời gian, nhiều hạt giống. S4D là đối chứng để phân biệt "Mamba không phù hợp" với "SSM nói chung không phù hợp".

**Phần đáp ứng yêu cầu tiểu luận:**

4. Giải thích mô hình (XAI) bằng Integrated Gradients (thư viện Captum): giờ nào và trạm thượng nguồn nào ảnh hưởng tới dự báo.
5. Demo bản đồ dự báo theo giờ cho 358 trạm, tô màu theo ngưỡng chu kỳ lặp lại (return period) của lưu lượng, xem chuỗi Q và giải thích XAI từng trạm.

**Giai đoạn khóa luận:** đồ thị dày, biến thể Mamba cải tiến cho phụ thuộc xa, bộ dữ liệu thứ hai (CAMELS-US theo giờ), nâng demo thành phần mềm ứng dụng.
