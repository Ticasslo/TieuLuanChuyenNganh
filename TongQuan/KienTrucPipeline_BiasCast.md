# Kiến trúc và pipeline của đề tài — phương án bài cơ sở BiasCast

> Tài liệu mô tả pipeline khi dùng BiasCast (Konold và cs., HESS 2026) làm bài cơ sở: mỗi giai đoạn nhận gì, làm gì, cho ra gì, chạy ở đâu; kiến trúc mô hình dạng mô-đun; ma trận thí nghiệm; các việc chưa chốt. Đây là phương án đang cân nhắc song song với phương án Kirschstein & Sun (ICML 2024) ở `TongQuan/KienTrucPipeline.md`; việc chọn bài cơ sở chưa chốt. So sánh hai bài: `BaiCoSo/SOSANH.md`. Chi tiết tải dữ liệu: `LamaHCE/LamaHCE.md` Mục 6. Các mục ghi "chưa chốt" chờ quyết định, chưa được điền.

## 1. Tổng quan các giai đoạn

| Giai đoạn | Việc chính | Đầu ra | Nền tảng | Trạng thái |
|---|---|---|---|---|
| 1. Thu thập dữ liệu | Tải Extended LamaH-CE và kết quả thí nghiệm của tác giả từ Zenodo | Kaggle Dataset `lamah-ce-ext` | Kaggle, CPU | Đã chạy; chờ ghi log kiểm tra ZIP |
| 2. Khảo sát dữ liệu | % thiếu, phân phối, bản đồ 451 lưu vực, so sánh tái phân tích với dự báo | Báo cáo khảo sát kèm biểu đồ, bản đồ | Kaggle hoặc Colab, CPU | Một phần có mã (% thiếu, persistence) |
| 3. Tái lập bài gốc và baseline persistence | Nạp mô hình tác giả, chạy lại tập test; tính persistence | Bảng mốc: số tự chạy lại + persistence | Kaggle hoặc Colab, GPU | Persistence có mã; tái lập chưa làm |
| 4. Tiền xử lý và bộ nạp dữ liệu | Dựng bộ dữ liệu cho NeuralHydrology: hindcast, forecast, thuộc tính, nhãn | Thư mục dữ liệu và tệp cấu hình dùng chung | Kaggle hoặc Colab | Chưa làm |
| 5. Mô hình | Khung hindcast–forecast với lõi thời gian thay được | Các cấu hình mô hình cần so sánh | — | Chưa làm |
| 6. Huấn luyện | Huấn luyện mọi cấu hình, nhiều hạt giống | Trọng số, nhật ký huấn luyện | Kaggle hoặc Colab, GPU T4 | Chưa làm |
| 7. Đánh giá | Chỉ số theo lưu vực trên test 2014–2017, so với persistence và bài gốc | Bảng, đường CDF, bản đồ ΔNSE | Kaggle hoặc Colab | Chưa làm |
| 8. Giải thích mô hình (XAI) | Integrated Gradients theo ngày và theo nhóm biến | Biểu đồ mức ảnh hưởng | Kaggle hoặc Colab, GPU | Chưa làm |
| 9. Ngưỡng nguy cơ lũ | Fit phân phối cực trị trên qmax năm của từng trạm | Ngưỡng return period của mỗi trạm | CPU | Chưa làm |
| 10. Demo | Bản đồ 451 lưu vực theo ngày, chuỗi Q, giải thích XAI | Ứng dụng demo | Chưa chốt | Chưa làm |

Luồng dữ liệu: hai tệp nén trên Zenodo → (1) Kaggle Dataset → (2) báo cáo khảo sát → (3) mốc tự chạy lại và persistence → (4) thư mục dữ liệu NeuralHydrology → (5, 6) trọng số các mô hình → (7) bảng kết quả → (8, 9) mức ảnh hưởng và ngưỡng nguy cơ lũ → (10) demo.

## 2. Chi tiết từng giai đoạn

### 2.1. Thu thập dữ liệu

- **Đầu vào:** `Extended_LamaH-CE_daily.tar.gz` (0,95 GB, Zenodo 17119635, CC BY-NC 4.0) và `Experiments.tar.gz` (0,23 GB, Zenodo 17292895: cấu hình, trọng số, scaler, NSE/KGE theo lưu vực của 24 cấu hình).
- **Cách làm:** đọc theo luồng, ghi vào hai tệp ZIP, lưu thành Kaggle Dataset.
- **Dữ liệu bổ sung đã có:** LamaH-CE gốc theo giờ trên Kaggle (`lamah-ce-core`, `lamah-ce-extra`) — dùng cho mạng sông, shapefile và dữ liệu giờ nếu mở rộng.
- **Tệp mã:** `LamaHCE/LamaHCEExt_Download.py` (Phần 1–2).

### 2.2. Khảo sát dữ liệu

- **Cấu trúc tệp:** cột thực tế so với Bảng A1, A2 của bài (31 biến tái phân tích, 5 biến ECMWF, 33 thuộc tính); đơn vị; mã giá trị thiếu (-999).
- **Chất lượng:** % thiếu `qmax`, `qmean` và từng biến khí tượng theo kỳ train / validation / test; lưu vực thiếu dài; năm bắt đầu có dữ liệu ECMWF và tái phân tích (quyết định độ dài hindcast tối đa).
- **Phân phối:** histogram từng biến; so sánh phân phối tái phân tích và dự báo của 5 biến chung (Phụ lục B của bài); qmax so với qmean.
- **Không gian:** bản đồ 451 lưu vực (72,5% đầu nguồn, 27,5% lồng nhau), diện tích, độ cao.
- **Đỉnh lũ:** chuỗi qmax lớn nhất năm từng trạm (đầu vào cho giai đoạn 9).
- **Tệp mã:** `LamaHCE/LamaHCEExt_Download.py` Phần 3 (% thiếu, persistence); khảo sát đầy đủ ở tệp dự kiến `LamaHCE/LamaHCEExt_Explore.py`.

### 2.3. Tái lập bài gốc và baseline persistence

Mục đích: có mốc tự chạy lại và bổ sung baseline mà bài gốc không báo cáo.

- **Tái lập:** cài bản fork NeuralHydrology của tác giả; nạp `best_model.pt` của các cấu hình chính (Sequential Forecast LSTM, Encoder–Decoder LSTM, có và không có Q); chạy tập test 2014–2017; so NSE từng lưu vực với `test_metrics.csv` của tác giả.
- **Việc phải làm thêm:** biến thể có Q đọc thư mục `LamaH_expanded_q_input` không có trên Zenodo — tự dựng bằng cách chép cột `qmean` từ tệp trạm vào tệp khí tượng; kiểm đơn vị `qmean` (nhiều khả năng m³/s trong khi nhãn `qmax` được đổi sang mm/ngày).
- **Persistence:** qmax(t) ≈ qmax(t−1) và qmax(t) ≈ qmean(t−1). Người phản biện 1 của bài đề nghị mốc này, tác giả không thêm (chỉ lập luận qmax ngày tự tương quan thấp).
- **Nhiễu giữa các lần chạy:** bài chạy 1 hạt giống; huấn luyện lại cấu hình tốt nhất với vài hạt giống để biết mức chênh lệch nào là thật.
- **Đầu ra:** bảng mốc gồm số của tác giả, số tự chạy lại, persistence, độ lệch giữa các hạt giống.
- **Tệp mã:** persistence ở `LamaHCEExt_Download.py` Phần 3; tái lập ở tệp dự kiến `LamaHCE/LamaHCEExt_Reproduce.py`.

### 2.4. Tiền xử lý và bộ nạp dữ liệu

- **Khung:** bộ nạp `lamah_a` của NeuralHydrology (bản fork), dữ liệu mức A — toàn bộ lưu vực thượng nguồn của trạm.
- **Đầu vào của bài gốc:** hindcast 364 ngày × 31 biến tái phân tích (+ `qmean` ở biến thể có Q); forecast ngày t × 5 biến ECMWF HRES; 33 thuộc tính tĩnh.
- **Nhãn:** `qmax` của ngày t, bộ nạp đổi sang mm/ngày theo diện tích lưu vực; demo đổi ngược về m³/s.
- **Chuẩn hóa:** z-score toàn cục tính trên kỳ train (mặc định NeuralHydrology).
- **Chia tập:** giữ đúng bài gốc — train 2003–2009, validation 2010–2013, test 2014–2017.
- **Lọc mẫu:** NeuralHydrology loại mẫu huấn luyện khi bất kỳ biến động nào thiếu ở bất kỳ ngày nào trong cửa sổ (kể cả cột ECMWF ở ngày hindcast); kéo dài hindcast sẽ làm mất thêm mẫu, nên số mẫu hợp lệ phải được báo cáo cho từng độ dài hindcast. Pipeline chi tiết của bài gốc: `TongQuan/GioiThieuBaiCoSo_BiasCast.md` Mục 11.
- **Chưa chốt:** độ dài hindcast (giữ 365 ngày làm mốc; thử dài hơn để khai thác thế mạnh chuỗi dài của Mamba, phụ thuộc năm bắt đầu có dữ liệu); đưa `qmean` về cùng đơn vị mm/ngày với nhãn; có thêm dữ liệu giờ của LamaH-CE gốc cho phần quá khứ gần hay không (ý tưởng ghép ngày + giờ của MF-LSTM, Acuña Espinoza và cs., HESS 2025); lead time nhiều ngày (Extended LamaH-CE chỉ có dự báo cho ngày t — mở rộng cần nguồn dự báo nhiều lead, ví dụ Caravan MultiMet; chưa xác nhận).

### 2.5. Mô hình

Chi tiết ở Mục 3.

### 2.6. Huấn luyện

- **Cấu hình mốc (giữ như bài gốc để so sánh được):** hàm mất mát NSE* (chia sai số cho độ lệch chuẩn lưu lượng từng lưu vực), Adam lr 1e-3, CosineAnnealing 30 epoch, batch 256, cắt gradient 1, dropout đầu ra 0,3, dừng sớm theo NSE validation (patience 5).
- **Siêu tham số:** cùng ngân sách tìm kiếm cho mọi lõi thời gian; bài gốc tối ưu Bayes 100 lần thử (kích thước ẩn, dropout, nhiễu nhãn, batch) — mức tìm kiếm cho đề tài chưa chốt, phụ thuộc quota GPU.
- **Hạt giống:** nhiều hạt giống cho mỗi cấu hình (số lượng chưa chốt).
- **Ràng buộc phần cứng:** `mamba-ssm` cần GPU CUDA từ sm_75 (T4 chạy được, P100 không); wheel biên dịch một lần rồi lưu lại (`LyThuyet/RiverMamba.md` Mục 0). Bài gốc báo vài phút tới khoảng 1 giờ mỗi lần chạy trên RTX 4090; thời gian trên T4 chưa đo.

### 2.7. Đánh giá

- **Chỉ số chính:** NSE và KGE theo từng lưu vực trên test 2014–2017, báo cáo trung vị, trung bình, phân vị 10/25/75/90 và đường CDF như Bảng F1, các hình của bài gốc.
- **Chỉ số bổ sung:** FHV (sai lệch đỉnh), sai số thời điểm đỉnh, F1 theo ngưỡng return period.
- **So sánh:** với persistence, với số của tác giả và số tự chạy lại; ΔNSE theo lưu vực và tương quan với thuộc tính (như Mục 3.6 của bài); trung bình ± độ lệch chuẩn qua các hạt giống; kiểm định thống kê (phương pháp chưa chốt).
- **Mốc tham chiếu của bài:** baseline chỉ dự báo (cận dưới, NSE trung vị 0,39) và baseline tái phân tích với thời tiết hoàn hảo (cận trên, 0,69).

### 2.8. Giải thích mô hình (XAI)

- Integrated Gradients bằng Captum.
- Theo thời gian: ngày nào trong 364 ngày hindcast ảnh hưởng nhiều nhất.
- Theo nhóm biến: tỷ trọng đóng góp của tái phân tích, dự báo ECMWF và Q quan trắc — trả lời trực tiếp câu hỏi của bài gốc về vai trò của dự báo thời tiết và của Q.
- Khác biệt với Demiray & Demir (Mamba + SHAP trên WaterBench-Iowa): tách đóng góp của dự báo thời tiết thật so với tái phân tích.

### 2.9. Ngưỡng nguy cơ lũ

- Mô hình dự báo lưu lượng cực đại ngày, không dự báo mực nước hay vùng ngập; nguy cơ lũ đánh giá qua ngưỡng lưu lượng ứng với chu kỳ lặp lại.
- Ngưỡng tính riêng từng trạm bằng Gumbel hoặc L-moments trên chuỗi qmax lớn nhất năm; tệp trạm có dữ liệu từ 1981.
- **Chưa chốt:** các chu kỳ lặp lại dùng làm ngưỡng.

### 2.10. Demo

- Bản đồ 451 lưu vực tô màu theo mức vượt ngưỡng; chọn ngày để xem dự báo; bấm vào lưu vực để xem chuỗi qmax dự báo so với thực đo, dự báo thời tiết đầu vào và giải thích XAI.
- Dữ liệu lịch sử, demo phát lại theo thời gian trên tập test 2014–2017.
- **Chưa chốt:** công nghệ làm demo và nơi triển khai.

## 3. Kiến trúc mô hình

### 3.1. Khung chung

Giữ khung hindcast–forecast của bài gốc, chỉ thay lõi thời gian:

| Khối | Đầu vào → đầu ra | Vai trò |
|---|---|---|
| Nhúng đầu vào | Biến hindcast (31–32), biến forecast (5), thuộc tính (33) → vector cùng số chiều | Đưa hai nguồn khác số biến về cùng không gian |
| Lõi thời gian | 364 bước hindcast rồi 1 bước forecast → trạng thái ẩn | Học quan hệ theo thời gian, chuyển trạng thái từ quá khứ sang ngày dự báo |
| Đầu ra | Trạng thái ẩn bước cuối → qmax ngày t | Hồi quy |

### 3.2. Các lựa chọn cho lõi thời gian

| Lựa chọn | Cách ghép vào khung | Nguồn |
|---|---|---|
| Sequential Forecast LSTM | Một LSTM chạy liên tục qua hindcast rồi forecast | Bài gốc (mô hình tốt nhất) |
| Encoder–Decoder LSTM | Hai LSTM nối bằng mạng handoff | Bài gốc; Nearing và cs. (Nature 2024) |
| GRU | Thay LSTM trong khung sequential | Baseline hồi quy |
| Transformer | Chuỗi 365 bước đã nhúng, attention có mặt nạ nhân quả | Baseline attention |
| S4D | Thay LSTM trong khung sequential | Đối chứng SSM bất biến theo thời gian; mã từ S4D-FT (Wang và cs., WRR 2025) |
| Mamba | Nối nhúng hindcast và forecast thành một chuỗi 365 bước, lấy đầu ra bước cuối | Mô hình chính; phải tự thêm vào NeuralHydrology (lớp Mamba có sẵn quét sai trục, `BaiCoSo/CHECKCODE.md` Mục 13) |

- Quét hai chiều chỉ được dùng bên trong phần hindcast rồi dự báo sau cửa sổ; đầu ra lấy tại bước cuối nên không rò rỉ tương lai.
- Mạng nhúng: tuyến tính (16 chiều) như cấu hình tốt nhất của bài; có dùng mạng nhúng phức tạp hay LOAN của RiverMamba để đưa thuộc tính vào hay không: chưa chốt.

### 3.3. Ma trận thí nghiệm

- Mỗi lõi thời gian chạy hai biến thể dữ liệu: không có Q và có Q quan trắc trong hindcast (bài gốc cho thấy Q tăng NSE trung vị 0,63 → 0,71).
- **LSTM là baseline chính:** so sánh quan trọng nhất là Mamba với Sequential Forecast LSTM ở cùng dữ liệu, cùng ngân sách tham số và tìm kiếm siêu tham số.
- **Trục độ dài hindcast (chưa chốt):** 365 ngày (mốc bài gốc) và dài hơn — nơi Mamba được kỳ vọng có lợi thế về chi phí tính toán so với LSTM và Transformer.
- **Mốc so sánh:** persistence; số của tác giả; số tự chạy lại nhiều hạt giống; hai baseline tham chiếu của bài (chỉ dự báo, tái phân tích).
- Điều kiện công bằng: cùng dữ liệu, cùng chia tập, cùng hàm mất mát, cùng hạt giống, cùng ngân sách huấn luyện; cách khống chế số tham số chưa chốt (Sequential Forecast LSTM của bài có khoảng 85 nghìn tham số với nhúng đơn giản).

## 4. Hạ tầng và tổ chức mã

- **Tính toán:** Google Colab bản miễn phí là chính, Kaggle dự phòng (GPU 30 giờ/tuần, tối đa 12 giờ/phiên); mã chạy được trên cả hai.
- **Khung mã:** bản fork NeuralHydrology của tác giả (1 commit sửa 7 tệp so với bản gốc): thêm mô hình vào `modelzoo`, cấu hình bằng tệp YAML như 24 cấu hình của bài.
- **Dữ liệu:** Kaggle Dataset `lamah-ce-ext` (và `lamah-ce-core`, `lamah-ce-extra` nếu dùng dữ liệu giờ, mạng sông).
- **Tổ chức mã:** mỗi notebook một tệp `.py` tự đủ, có mục lục, chia cell theo "Phần"; ghi chú kết quả ở `LamaHCE/LamaHCE.md`.

| Tệp | Giai đoạn | Trạng thái |
|---|---|---|
| `LamaHCE/LamaHCEExt_Download.py` | 1, một phần 2 và 3 | Đã chạy |
| `LamaHCE/LamaHCEExt_Explore.py` | 2 | Tên dự kiến |
| `LamaHCE/LamaHCEExt_Reproduce.py` | 3 | Tên dự kiến |
| Tệp cho giai đoạn 4–10 | 4–10 | Chưa đặt tên |

## 5. Việc chưa chốt

| Việc | Giai đoạn |
|---|---|
| Chọn bài cơ sở (BiasCast đã gửi GVHD 26/9/2026, chờ duyệt; Kirschstein & Sun là phương án thay thế) | Toàn đề tài |
| Có đổi quy tắc lọc mẫu (chỉ kiểm cột ECMWF ở ngày forecast) hay giữ như bài gốc | 4 |
| Độ dài hindcast ngoài mốc 365 ngày | 4 |
| Đưa `qmean` về cùng đơn vị với nhãn | 4 |
| Có dùng dữ liệu giờ LamaH-CE gốc cho quá khứ gần hay không | 4, 5 |
| Có mở rộng sang dự báo nhiều ngày hay không (cần nguồn dự báo khác) | 4, 5 |
| Mạng nhúng và cách đưa thuộc tính tĩnh (nhúng thường hay LOAN) | 5 |
| Cách khống chế số tham số giữa các mô hình | 5 |
| Mức tìm kiếm siêu tham số và số hạt giống | 6 |
| Phương pháp kiểm định thống kê | 7 |
| Các chu kỳ lặp lại dùng làm ngưỡng | 9 |
| Công nghệ và nơi triển khai demo | 10 |
| Thêm đồ thị mạng sông giữa các lưu vực lồng nhau (tiểu luận hay khóa luận) | 5 |

## 6. Đối chiếu với pipeline của BiasCast

Pipeline giữ nguyên những phần quyết định khả năng so sánh (cùng 451 lưu vực, cùng chia tập, cùng đầu vào và nhãn, cùng hàm mất mát và khung NeuralHydrology) và mở rộng ở những chỗ bài gốc tự nêu hạn chế hoặc người phản biện chỉ ra. Pipeline của bài đọc từ mã (`github.com/conestone/biascast`, bản fork NeuralHydrology) và 24 cấu hình trên Zenodo.

| Giai đoạn | BiasCast | Đề tài | Giống / khác |
|---|---|---|---|
| Thu thập dữ liệu | Tự gộp ERA5-Land, E-OBS, MSWEP, GLEAM, ECMWF HRES theo lưu vực | Dùng bản đã công bố trên Zenodo | Cùng dữ liệu |
| Khảo sát | Phân phối năm các nguồn (Phụ lục B), khoảng cách Wasserstein tái phân tích – dự báo | Thêm % thiếu, bản đồ, đỉnh lũ | Đề tài mở rộng |
| Bộ nạp, chia tập | `lamah_a`, hindcast 364 + forecast 1 ngày, 2003–2009 / 2010–2013 / 2014–2017 | Giữ nguyên; thử hindcast dài hơn | Giống, thêm một trục thí nghiệm |
| Mô hình | Chỉ các kiến trúc LSTM (tác giả tự nêu ở Mục 3.7) | LSTM, GRU, Transformer, S4D, Mamba trong cùng khung | Đề tài mở rộng — khoảng trống chính |
| Huấn luyện | NSE*, tối ưu Bayes, 1 hạt giống | Giữ cấu hình mốc; nhiều hạt giống | Khác số hạt giống |
| Đánh giá | NSE, KGE theo lưu vực; cận trên, cận dưới | Thêm persistence, FHV, thời điểm đỉnh, F1 theo ngưỡng, kiểm định | Đề tài mở rộng |
| Tái lập | Không áp dụng | Chạy lại trọng số của tác giả, đo nhiễu giữa hạt giống | Chỉ có ở đề tài |
| XAI, ngưỡng nguy cơ lũ, demo | Không có | Có | Chỉ có ở đề tài |

## 7. Khác biệt với phương án Kirschstein & Sun

| Khía cạnh | Phương án Kirschstein & Sun | Phương án BiasCast |
|---|---|---|
| Bài toán | Q theo giờ trước 6 giờ, 358 trạm cùng lúc | qmax ngày t với dự báo ECMWF thật, 451 lưu vực |
| Giai đoạn 3 | Sửa và tính lại NSE từ 957 checkpoint | Chạy lại trọng số tác giả, thêm persistence |
| Lõi mô hình | Bộ mã hóa thời gian thay lớp affine, trước GNN | Lõi thời gian thay LSTM trong khung hindcast–forecast |
| Câu hỏi nghiên cứu | Mamba có hơn các bộ mã hóa khác; đồ thị có giúp không | Mamba có hơn LSTM dự báo kiểu Google khi dùng dự báo thời tiết thật; chuỗi dài hơn có giúp không |
| Thế mạnh chuỗi dài của Mamba | Rõ (dữ liệu giờ) | Phải kéo dài hindcast hoặc thêm dữ liệu giờ |
| Dữ liệu cần | 13 GB (đã có trên Kaggle) | 1,2 GB |
