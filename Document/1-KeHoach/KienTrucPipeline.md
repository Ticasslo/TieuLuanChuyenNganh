# Kiến trúc và pipeline của đề tài

> Tài liệu mô tả pipeline của đề tài trên bài cơ sở BiasCast (Konold và cs., HESS 2026): mỗi giai đoạn nhận gì, làm gì, cho ra gì, chạy ở đâu; kiến trúc mô hình dạng mô-đun; ma trận thí nghiệm; đánh giá theo mức lưu lượng; lộ trình cải tiến và phạm vi. Giới thiệu bài cơ sở: `Document/1-KeHoach/GioiThieuBaiCoSo.md`. Chi tiết tải dữ liệu: `Document/3-DuLieu/LamaHCE.md` Mục 6. Các mục ghi "chưa chốt" chờ quyết định, chưa được điền.

## 1. Tổng quan các giai đoạn

| Giai đoạn | Việc chính | Đầu ra | Nền tảng | Trạng thái |
|---|---|---|---|---|
| 1. Thu thập dữ liệu | Tải Extended LamaH-CE và kết quả thí nghiệm của tác giả từ Zenodo | Kaggle Dataset `lamah-ce-ext` | Kaggle, CPU | Xong (26/9/2026): 1.991 + 123 file, lỗi CRC không có (`Document/3-DuLieu/LamaHCE.md` Mục 6) |
| 2. Khảo sát dữ liệu | % thiếu, phân phối, bản đồ 451 lưu vực, so sánh tái phân tích với dự báo | Báo cáo khảo sát kèm biểu đồ, bản đồ | Kaggle hoặc Colab, CPU | % thiếu đã có (`Document/3-DuLieu/LamaHCE.md` Mục 6.1: Q thiếu train 3,55% do 27 trạm bắt đầu đo muộn, test 0,23%; khí tượng gần như đủ, có từ 1981); khảo sát đầy đủ chưa làm |
| 3. Tái lập bài gốc và baseline persistence | Nạp mô hình tác giả, chạy lại tập test; tính persistence | Bảng mốc: số tự chạy lại + persistence | Kaggle hoặc Colab, GPU | Persistence xong (NSE trung vị 0,35–0,37); tái lập chưa làm |
| 4. Tiền xử lý và bộ nạp dữ liệu | Dựng bộ dữ liệu cho NeuralHydrology: hindcast, forecast, thuộc tính, nhãn | Thư mục dữ liệu và tệp cấu hình dùng chung | Kaggle hoặc Colab | Chưa làm |
| 5. Mô hình | Khung hindcast–forecast với lõi thời gian thay được | Các cấu hình mô hình cần so sánh | — | Chưa làm |
| 6. Huấn luyện | Tầng 1 (cải tiến LSTM) rồi tầng 2 (các lõi), 3 hạt giống mỗi cấu hình (Mục 8–9) | Trọng số, nhật ký huấn luyện | Kaggle hoặc Colab, GPU T4 | Chưa làm |
| 7. Đánh giá | Chỉ số theo lưu vực trên test 2014–2017, so với persistence và bài gốc; đánh giá theo mức lưu lượng (Mục 7) | Bảng, đường CDF, bản đồ ΔNSE, ma trận nhầm lẫn theo mức | Kaggle hoặc Colab | Chưa làm |
| 8. Giải thích mô hình (XAI) | Integrated Gradients theo ngày và theo nhóm biến | Biểu đồ mức ảnh hưởng | Kaggle hoặc Colab, GPU | Chưa làm |
| 9. Ngưỡng nguy cơ lũ | Fit phân phối cực trị trên qmax năm của từng trạm | Ngưỡng return period của mỗi trạm | CPU | Chưa làm |
| 10. Demo | Bản đồ 451 lưu vực theo ngày, chuỗi Q, giải thích XAI | Ứng dụng demo | Chưa chốt | Chưa làm |

Luồng dữ liệu: hai tệp nén trên Zenodo → (1) Kaggle Dataset → (2) báo cáo khảo sát → (3) mốc tự chạy lại và persistence → (4) thư mục dữ liệu NeuralHydrology → (5, 6) trọng số các mô hình → (7) bảng kết quả → (8, 9) mức ảnh hưởng và ngưỡng nguy cơ lũ → (10) demo.

## 2. Chi tiết từng giai đoạn

### 2.1. Thu thập dữ liệu

- **Đầu vào:** `Extended_LamaH-CE_daily.tar.gz` (0,95 GB, Zenodo 17119635, CC BY-NC 4.0) và `Experiments.tar.gz` (0,23 GB, Zenodo 17292895: cấu hình, trọng số, scaler, NSE/KGE theo lưu vực của 24 cấu hình).
- **Cách làm:** đọc theo luồng, ghi vào hai tệp ZIP, lưu thành Kaggle Dataset.
- **Dữ liệu bổ sung đã có:** LamaH-CE gốc theo giờ trên Kaggle (`lamah-ce-core`, `lamah-ce-extra`) — dữ liệu giờ và mạng sông cho các hướng của khóa luận (Mục 9.2).
- **Tệp mã:** `LamaHCE/LamaHCEExt_Download.py` (Phần 1–2).

### 2.2. Khảo sát dữ liệu

- **Cấu trúc tệp:** cột thực tế so với Bảng A1, A2 của bài (31 biến tái phân tích, 5 biến ECMWF, 33 thuộc tính); đơn vị; mã giá trị thiếu (-999).
- **Chất lượng:** % thiếu `qmax`, `qmean` và từng biến khí tượng theo kỳ train / validation / test; lưu vực thiếu dài; năm bắt đầu có dữ liệu ECMWF và tái phân tích (quyết định độ dài hindcast tối đa).
- **Phân phối:** histogram từng biến; so sánh phân phối tái phân tích và dự báo của 5 biến chung (Phụ lục B của bài); qmax so với qmean.
- **Không gian:** bản đồ 451 lưu vực (72,5% đầu nguồn, 27,5% lồng nhau), diện tích, độ cao.
- **Đỉnh lũ:** chuỗi qmax lớn nhất năm từng trạm (đầu vào cho giai đoạn 9).
- **Tệp mã:** `LamaHCE/LamaHCEExt_Download.py` Phần 3 (% thiếu, persistence); khảo sát đầy đủ ở `LamaHCE/LamaHCEExt_Explore.py` (giới thiệu 4 phần) và `LamaHCE/LamaHCEExt_Analysis.py` (phân tích sâu), `Document/3-DuLieu/LamaHCE.md` Mục 6.2.

### 2.3. Tái lập bài gốc và baseline persistence

Mục đích: có mốc tự chạy lại và bổ sung baseline mà bài gốc không báo cáo.

- **Tái lập:** cài bản fork NeuralHydrology của tác giả; nạp `best_model.pt` của các cấu hình chính (Sequential Forecast LSTM, Encoder–Decoder LSTM, có và không có Q); chạy tập test 2014–2017; so NSE từng lưu vực với `test_metrics.csv` của tác giả.
- **Việc phải làm thêm:** biến thể có Q đọc thư mục `LamaH_expanded_q_input` không có trên Zenodo — tự dựng bằng cách chép cột `qmean` từ tệp trạm vào tệp khí tượng; kiểm đơn vị `qmean` (nhiều khả năng m³/s trong khi nhãn `qmax` được đổi sang mm/ngày).
- **Persistence:** qmax(t) ≈ qmax(t−1) và qmax(t) ≈ qmean(t−1). Người phản biện 1 của bài đề nghị mốc này, tác giả không thêm (chỉ lập luận qmax ngày tự tương quan thấp). Kết quả tự tính trên test 2014–2017: persistence đạt NSE trung vị 0,346 (qmax) và 0,368 (qmean); mô hình tốt nhất của tác giả (0,705) hơn persistence ở trên 92% lưu vực; mô hình chỉ dùng dự báo thời tiết (0,387) ngang persistence (`Document/3-DuLieu/LamaHCE.md` Mục 6.1).
- **Nhiễu giữa các lần chạy:** bài chạy 1 hạt giống; huấn luyện lại cấu hình tốt nhất với vài hạt giống để biết mức chênh lệch nào là thật.
- **Đầu ra:** bảng mốc gồm số của tác giả, số tự chạy lại, persistence, độ lệch giữa các hạt giống.
- **Tệp mã:** persistence ở `LamaHCEExt_Download.py` Phần 3; tái lập ở tệp dự kiến `LamaHCE/LamaHCEExt_Reproduce.py`.

### 2.4. Tiền xử lý và bộ nạp dữ liệu

- **Khung:** bộ nạp `lamah_a` của NeuralHydrology (bản fork), dữ liệu mức A — toàn bộ lưu vực thượng nguồn của trạm.
- **Đầu vào của bài gốc:** hindcast 364 ngày × 31 biến tái phân tích (+ `qmean` ở biến thể có Q); forecast ngày t × 5 biến ECMWF HRES; 33 thuộc tính tĩnh.
- **Nhãn:** `qmax` của ngày t, bộ nạp đổi sang mm/ngày theo diện tích lưu vực; demo đổi ngược về m³/s.
- **Chuẩn hóa:** z-score toàn cục tính trên kỳ train (mặc định NeuralHydrology).
- **Chia tập:** giữ đúng bài gốc — train 2003–2009, validation 2010–2013, test 2014–2017.
- **Lọc mẫu:** NeuralHydrology loại mẫu huấn luyện khi bất kỳ biến động nào thiếu ở bất kỳ ngày nào trong cửa sổ (kể cả cột ECMWF ở ngày hindcast); kéo dài hindcast sẽ làm mất thêm mẫu, nên số mẫu hợp lệ phải được báo cáo cho từng độ dài hindcast. Pipeline chi tiết của bài gốc: `Document/1-KeHoach/GioiThieuBaiCoSo.md` Mục 11.
- **Theo phạm vi Mục 9:** hindcast 365 ngày (mốc) và 730 ngày (khí tượng có từ 1981 nên đủ mẫu train từ 2003); đưa `qmean` về cùng đơn vị mm/ngày với nhãn và sửa bộ lọc mẫu (tầng 1, Mục 8.2). Để khóa luận: ghép dữ liệu giờ của LamaH-CE gốc cho phần quá khứ gần (ý tưởng MF-LSTM, Acuña Espinoza và cs., HESS 2025); lead time nhiều ngày (Extended LamaH-CE có vẻ chỉ có dự báo cho ngày t — cần nguồn dự báo nhiều lead, ví dụ Caravan MultiMet; chưa xác nhận trên dữ liệu).

### 2.5. Mô hình

Chi tiết ở Mục 3.

### 2.6. Huấn luyện

- **Cấu hình mốc (giữ như bài gốc để so sánh được):** hàm mất mát NSE* (chia sai số cho độ lệch chuẩn lưu lượng từng lưu vực), Adam lr 1e-3, CosineAnnealing 30 epoch, batch 256, cắt gradient 1, dropout đầu ra 0,3, dừng sớm theo NSE validation (patience 5).
- **Siêu tham số:** cùng ngân sách tìm kiếm cho mọi lõi thời gian; bài gốc tối ưu Bayes 100 lần thử (kích thước ẩn, dropout, nhiễu nhãn, batch). Đề tài tìm số chiều ẩn 64 / 128 / 256 trên validation cho mỗi lõi (Mục 9), giữ các siêu tham số khác như bài gốc; mở rộng nếu còn quota.
- **Hạt giống:** 3 hạt giống cho mỗi cấu hình (Mục 9); kết quả báo cáo cả từng hạt giống và tổ hợp (trung bình dự báo của 3 hạt giống).
- **Ràng buộc phần cứng:** `mamba-ssm` cần GPU CUDA từ sm_75 (T4 chạy được, P100 không); wheel biên dịch một lần rồi lưu lại (`Document/7-LyThuyet/RiverMamba.md` Mục 0). Bài gốc báo vài phút tới khoảng 1 giờ mỗi lần chạy trên RTX 4090; thời gian trên T4 chưa đo.

### 2.7. Đánh giá

- **Chỉ số chính:** NSE và KGE theo từng lưu vực trên test 2014–2017, báo cáo trung vị, trung bình, phân vị 10/25/75/90 và đường CDF như Bảng F1, các hình của bài gốc.
- **Chỉ số bổ sung:** FHV (sai lệch đỉnh), FLV (sai lệch dòng chảy kiệt), sai số thời điểm đỉnh; đánh giá theo mức lưu lượng (Mục 7).
- **Theo dõi khi huấn luyện:** sau mỗi epoch ghi NSE trung vị trên cả validation và test (yêu cầu GVHD, họp 27/9/2026) để phát hiện sớm khi validation không phản ánh test; chọn checkpoint theo validation như bài gốc, NSE test chỉ để theo dõi và báo cáo.
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
| Mamba | Nối nhúng hindcast và forecast thành một chuỗi 365 bước, lấy đầu ra bước cuối | Mô hình chính; phải tự thêm vào NeuralHydrology (lớp Mamba có sẵn quét sai trục, `Document/6-ChonBaiCoSo/CHECKCODE.md` Mục 13) |

- Quét hai chiều chỉ được dùng bên trong phần hindcast rồi dự báo sau cửa sổ; đầu ra lấy tại bước cuối nên không rò rỉ tương lai.
- Mạng nhúng: tuyến tính (16 chiều) như cấu hình tốt nhất của bài; có dùng mạng nhúng phức tạp hay LOAN của RiverMamba để đưa thuộc tính vào hay không: chưa chốt.

### 3.3. Ma trận thí nghiệm

- Mỗi lõi thời gian chạy hai biến thể dữ liệu: không có Q và có Q quan trắc trong hindcast (bài gốc cho thấy Q tăng NSE trung vị 0,63 → 0,71).
- **LSTM là baseline chính:** so sánh quan trọng nhất là Mamba với Sequential Forecast LSTM ở cùng dữ liệu, cùng ngân sách tham số và tìm kiếm siêu tham số.
- **Trục độ dài hindcast:** 365 ngày (mốc bài gốc) và 730 ngày cho LSTM và Mamba (Mục 9) — nơi Mamba được kỳ vọng có lợi thế về chi phí tính toán so với LSTM và Transformer.
- **Thứ tự:** tầng 1 cải tiến LSTM trước, rồi các lõi dạng thuần, rồi một biến thể cải tiến cho Mamba và một cho Transformer (Mục 8).
- **Mốc so sánh:** persistence; số của tác giả; số tự chạy lại nhiều hạt giống; hai baseline tham chiếu của bài (chỉ dự báo, tái phân tích).
- Điều kiện công bằng: cùng dữ liệu, cùng chia tập, cùng hàm mất mát, cùng hạt giống, cùng ngân sách huấn luyện; cách khống chế số tham số chưa chốt (Sequential Forecast LSTM của bài có khoảng 85 nghìn tham số với nhúng đơn giản).

## 4. Hạ tầng và tổ chức mã

- **Tính toán:** Kaggle là chính (GPU 30 giờ/tuần, tối đa 12 giờ/phiên), Google Colab bản miễn phí dự phòng; mã chạy được trên cả hai.
- **Khung mã:** bản fork NeuralHydrology của tác giả (1 commit sửa 7 tệp so với bản gốc): thêm mô hình vào `modelzoo`, cấu hình bằng tệp YAML như 24 cấu hình của bài.
- **Dữ liệu:** Kaggle Dataset `lamah-ce-ext` (và `lamah-ce-core`, `lamah-ce-extra` nếu dùng dữ liệu giờ, mạng sông).
- **Tổ chức mã:** mỗi notebook một tệp `.py` tự đủ, có mục lục, chia cell theo "Phần"; ghi chú kết quả ở `Document/3-DuLieu/LamaHCE.md`.

| Tệp | Giai đoạn | Trạng thái |
|---|---|---|
| `LamaHCE/LamaHCEExt_Download.py` | 1, một phần 2 và 3 | Đã chạy |
| `LamaHCE/LamaHCEExt_Explore.py` | 2 | Đang viết (từng cell) |
| `LamaHCE/LamaHCEExt_Reproduce.py` | 3 | Tên dự kiến |
| Tệp cho giai đoạn 4–10 | 4–10 | Chưa đặt tên |

## 5. Việc chưa chốt

Các việc đã có hướng theo phạm vi ở Mục 9: độ dài hindcast (365 và 730 ngày), đưa `qmean` về cùng đơn vị với nhãn và sửa bộ lọc mẫu (tầng 1), số hạt giống (3) và mức tinh chỉnh (số chiều ẩn 64 / 128 / 256), dữ liệu giờ, dự báo nhiều ngày và đồ thị (khóa luận). Còn lại:

| Việc | Giai đoạn |
|---|---|
| Biến thể cải tiến của Mamba và Transformer (ứng viên: quét hai chiều trong hindcast; patch như PatchTST) | 5 |
| Mạng nhúng và cách đưa thuộc tính tĩnh (nhúng thường hay LOAN) | 5 |
| Cách khống chế số tham số giữa các mô hình | 5 |
| Phương pháp kiểm định thống kê | 7 |
| Các chu kỳ lặp lại dùng làm ngưỡng | 9 |
| Công nghệ và nơi triển khai demo | 10 |

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

## 7. Đánh giá theo mức lưu lượng

Yêu cầu của GVHD (họp 27/9/2026, `Document/2-HopGVHD/HopNhom_2026-09-27.md` Mục 2.2): bài toán là hồi quy nên NSE không chỉ ra mô hình mạnh, yếu ở đâu; chia lưu lượng thành các mức và đo như bài toán phân loại. Mô hình không đổi, việc chia mức làm sau khi đã có dự báo, không cần huấn luyện thêm.

| Bước | Nội dung |
|---|---|
| Ngưỡng | Riêng từng lưu vực, tính trên qmax kỳ huấn luyện: các phân vị của đường duy trì lưu lượng (đoạn cao 2% lớn nhất, đoạn thấp 30% nhỏ nhất theo Yilmaz, Gupta, Wagener, 2008, *Water Resources Research*) và ngưỡng lũ theo chu kỳ lặp lại (giai đoạn 9). Số mức và ngưỡng cụ thể chưa chốt |
| Gán mức | Gán mức cho cả qmax quan trắc và qmax dự báo của từng ngày kiểm tra |
| Chỉ số | Ma trận nhầm lẫn giữa các mức; độ chính xác từng mức; với mức lũ: tỉ lệ phát hiện, tỉ lệ báo động nhầm, F1; FHV, FLV cho đoạn cao và thấp |
| Cách đọc | Ưu tiên bỏ sót ít ở mức lũ và mức rất thấp, chấp nhận báo động nhầm nhiều hơn (theo GVHD); so sánh các lõi ở từng mức để biết Mamba hơn hay kém LSTM ở dải lưu lượng nào |
| Dùng cho | Biện luận kết quả; chọn cải tiến tiếp theo (ý tưởng lấy mẫu nhiều ngày lưu lượng cao, `Document/4-YTuong/YTUONG.md` Y7) |

## 8. Lộ trình cải tiến hai tầng

Đề tài cải tiến theo hai tầng: tầng 1 cải tiến quy trình dữ liệu, đầu vào, huấn luyện ngay trên LSTM của bài gốc; tầng 2 thay lõi thời gian (Mamba, Transformer, GRU, S4D) và dùng lại toàn bộ cải tiến của tầng 1. Các cải tiến tầng 1 không phụ thuộc kiến trúc, nên làm trên LSTM trước (nhanh, đã có sẵn trong NeuralHydrology) rồi áp dụng cho mọi lõi — so sánh ở tầng 2 vì vậy công bằng và mốc LSTM đủ mạnh.

### 8.1. Việc bắt buộc trước khi cải tiến (chạy lại được bài gốc)

| Việc | Lý do |
|---|---|
| Sửa đường dẫn trong 24 tệp `config.yml` (`data_dir`, `run_dir`, `train_basin_file`) | Ghi cứng đường dẫn máy tác giả (`/home/ok2907/...`) |
| Dựng thư mục `LamaH_expanded_q_input` (chép `qmean` vào tệp khí tượng) | Biến thể có Q đọc thư mục này; không được công bố |
| Sửa `_is_best_model` cho khớp ngưỡng dừng sớm | Có thể lưu epoch kém hơn tối đa 0,005 NSE validation; áp dụng chung cho mọi mô hình |

### 8.2. Tầng 1 — cải tiến trên LSTM

Mỗi cải tiến chạy thành một thí nghiệm riêng (bật từng cái so với LSTM gốc), nhiều hạt giống, để biết đóng góp của từng cái.

| Cải tiến | Điểm yếu của bài gốc mà nó xử lý | Bằng chứng công bố | Có sẵn trong NeuralHydrology |
|---|---|---|---|
| Giữ mẫu có dữ liệu thiếu thay vì bỏ cả mẫu, dùng *masked mean* (nhúng riêng từng nguồn khí tượng rồi lấy trung bình các nguồn có dữ liệu) | Bộ nạp bỏ mọi mẫu huấn luyện có giá trị thiếu ở bất kỳ biến nào trong 365 ngày; bài dùng 5 nguồn (ERA5-Land, E-OBS, MSWEP, GLEAM, ECMWF) — đúng bối cảnh nhiều nguồn mà phương pháp được thiết kế | Gauch và cs., HESS 29:6221–6235 (2025): trong ba cách (input replacing, masked mean, attention), masked mean tốt nhất với cách biệt nhỏ; attention không hơn masked mean; mô hình vận hành của Google dùng masked mean | Có lớp nhúng (`nan_handling_method`); **phải sửa bộ lọc mẫu** — trong bản fork, `basedataset.py` vẫn loại mẫu có giá trị thiếu dù bật tùy chọn này |
| Hoàn thiện cách đưa lưu lượng quan trắc vào: cùng đơn vị với nhãn, thêm `qmax` các ngày trước, xử lý ngày thiếu Q | Bài đã đưa `qmean` quá khứ vào hindcast (NSE trung vị 0,63 → 0,71), nhưng `qmean` nhiều khả năng lệch đơn vị với nhãn và mẫu thiếu Q bị bỏ; phần cải thiện còn lại dự kiến nhỏ hơn nhiều so với mức tăng khi lần đầu thêm Q | Nearing và cs., HESS 26:5493–5513 (2022): tự hồi quy tăng NSE trung vị ~10% so với mô hình không dùng Q (0,796 → 0,879 trên 531 lưu vực CAMELS); khi huấn luyện, thay Q thiếu bằng giá trị mô hình dự báo để chịu được dữ liệu thiếu | Một phần: `lagged_features` (thêm Q trễ) dùng được cho mọi mô hình; `autoregressive_inputs` và cách thay Q thiếu chỉ có trong mô hình `arlstm` — với Sequential Forecast LSTM phải tự viết |
| Chịu mất lưu lượng quan trắc: khi huấn luyện bỏ ngẫu nhiên Q (từng ngày và cả chuỗi, xác suất 0,1 như Gauch và cs.) qua cơ chế masked mean | Mô hình có Q cần Q đủ 365 ngày; trạm mất số liệu thì phải dùng mô hình khác; tác giả thử học chuyển giao để bỏ Q nhưng thất bại (Mục 3.4 của bài) | Hướng tương lai do tác giả BiasCast nêu (Mục 4); Gauch và cs. 2025 áp dụng cách bỏ ngẫu nhiên cho khí tượng, chưa cho Q; dữ liệu thiếu thực tế ít (`Document/3-DuLieu/LamaHCE.md` Mục 6.1); mẫu train bị bộ lọc loại chủ yếu ở 27 trạm đo muộn và ngày thiếu Q — lấy lại một phần bằng sửa bộ lọc; masked mean chủ yếu cho khả năng chịu mất dữ liệu khi vận hành | Dùng lớp nhúng masked mean; phải tự thêm bước bỏ ngẫu nhiên Q khi huấn luyện |
| Tổ hợp nhiều hạt giống (ensemble) | Bài chạy 1 hạt giống | Cách làm chuẩn của các công trình NeuralHydrology (tổ hợp LSTM trong Kratzert và cs., WRR 2019; tổ hợp 8 hạt giống trong Gauch và cs., Environ. Model. Softw. 2021) | Chỉ cần chạy nhiều lần rồi lấy trung bình |
| Tinh chỉnh siêu tham số cho LSTM dự báo | Mọi cấu hình LSTM dự báo dùng chung 128 / 256 / 0,3 | Baste và cs., HESS 29:5871–5891 (2025): LSTM có giới hạn trên của lưu lượng dự báo được (73 mm/ngày, thấp hơn đỉnh 183 mm/ngày trong dữ liệu huấn luyện); tăng số chiều ẩn lên 256 kèm tập huấn luyện lớn hơn nâng giới hạn này lên 194 mm/ngày nhưng không bảo đảm dự báo đỉnh tốt hơn | Có (tối ưu Bayes như bài gốc) |
| Đầu ra xác suất (CMAL) | Bài chỉ dự báo một giá trị | Klotz và cs., HESS 26:1673–1693 (2022): mô hình hỗn hợp cho ước lượng bất định đáng tin cậy | Có (`head: cmal`) — cho xác suất vượt ngưỡng lũ, dùng trực tiếp cho demo |
| Ưu tiên đỉnh lũ trong huấn luyện (trọng số theo chu kỳ lặp lại, lấy mẫu thêm sự kiện lớn) | LSTM có xu hướng dự báo thấp đỉnh lũ | Baste và cs. 2025 khuyến nghị tăng trọng số sự kiện cực trị; Talbot & Davenport (preprint 2026) cho thấy có đánh đổi giữa kỹ năng đỉnh và kỹ năng chung; RiverMamba dùng trọng số theo chu kỳ lặp lại | Phải tự viết hàm mất mát |

Đề xuất cho tiểu luận: bốn cải tiến đầu (dữ liệu thiếu, lưu lượng quan trắc, chịu mất lưu lượng quan trắc, tổ hợp hạt giống) — xuất phát trực tiếp từ điểm yếu tìm thấy trong mã của bài, chi phí thấp. Tinh chỉnh siêu tham số làm cùng ngân sách cho mọi lõi ở tầng 2. Đầu ra xác suất làm ở bước F của tiểu luận (Mục 9); ưu tiên đỉnh lũ để cho khóa luận.

### 8.3. Tầng 2 — thay lõi thời gian

Lõi Mamba, Transformer, GRU, S4D thay LSTM trong cùng khung (Mục 3), dùng cấu hình dữ liệu và huấn luyện tốt nhất của tầng 1, cùng ngân sách tinh chỉnh, cùng số hạt giống.

Thứ tự trong tầng 2: chạy **dạng thuần** của mỗi lõi trước (Mamba thuần, Transformer thuần chỉ thay LSTM, giữ nguyên mọi thứ khác), rồi mới tới **biến thể lai hoặc cải tiến** (ví dụ Mamba quét hai chiều trong hindcast, Mamba kết hợp attention hoặc Transformer, đưa thuộc tính tĩnh bằng LOAN). Mỗi biến thể so với dạng thuần của chính nó với cùng số tham số, để biết phần cải thiện đến từ lõi hay từ thành phần thêm vào. Nhảy thẳng vào mô hình lai thì không tách được hai phần này. Thêm trục độ dài hindcast (365 ngày và dài hơn) — kết hợp với cải tiến xử lý dữ liệu thiếu để không mất mẫu khi kéo dài chuỗi.

### 8.4. Hướng mở rộng khác (ngoài đồ thị và XAI)

| Hướng | Mô tả | Căn cứ | Phù hợp với BiasCast | Đề xuất |
|---|---|---|---|---|
| Ghép dữ liệu giờ vào quá khứ gần | Giữ hindcast xa theo ngày, thêm vài ngày gần nhất theo giờ (khí tượng giờ mức A, Q giờ) | Ý tưởng MF-LSTM (Acuña Espinoza và cs., HESS 2025); tác giả BiasCast tự nêu hạn chế "chỉ dữ liệu ngày" | Nhãn qmax là đỉnh lấy từ dữ liệu giờ, nên thông tin giờ gần nhất có lý do để giúp; dữ liệu giờ đã có trên Kaggle | Khóa luận (Mục 9) |
| Dự báo xác suất | Đầu CMAL cho phân phối qmax, suy ra xác suất vượt ngưỡng lũ | Klotz và cs., HESS 2022 | Có sẵn trong NeuralHydrology; dùng trực tiếp cho demo cảnh báo | Tiểu luận, bước F (Mục 9) |
| Ưu tiên đỉnh lũ | Trọng số theo chu kỳ lặp lại, lấy mẫu thêm sự kiện lớn | Baste và cs. 2025; Talbot & Davenport (preprint 2026) — có đánh đổi với kỹ năng chung | Cần tự viết hàm mất mát | Khóa luận |
| Dự báo nhiều ngày | Lead 1–7 ngày | Tác giả BiasCast nêu là nghiên cứu tiếp theo của nhóm | Extended LamaH-CE chỉ có dự báo cho ngày t; cần nguồn dự báo khác (Caravan MultiMet) | Khóa luận |
| Mô hình lai vật lý (bảo toàn khối lượng) | MC-LSTM có sẵn trong NeuralHydrology | Bằng chứng trái chiều: Frame và cs. (HESS 2022) thấy MC-LSTM kém LSTM ở sự kiện cực trị; Baste và cs. 2025 thấy mô hình lai ngoại suy đỉnh tốt hơn | Rủi ro cao | Không đề xuất cho tiểu luận |
| xLSTM làm thêm một lõi | Biến thể LSTM mới (Beck và cs., 2024) | Nghiên cứu nhiều lưu vực mới ở dạng preprint (SSRN); các bài đăng tạp chí chủ yếu một lưu vực hoặc MDPI | Thay lõi trong cùng khung như Mamba | Tùy chọn, chưa chốt |
| Mô hình nền tảng chuỗi thời gian (dự báo không huấn luyện) | Làm mốc so sánh kiểu "AI tổng quát" | Có trong danh sách khảo sát (`Document/6-ChonBaiCoSo/CHECKPDF.md` Mục 5.7) | Chỉ làm mốc, không phải cải tiến | Tùy chọn |

### 8.5. Câu hỏi nghiên cứu theo hai tầng

1. Các cải tiến quy trình (xử lý dữ liệu thiếu, tự hồi quy, chịu mất lưu lượng quan trắc, tổ hợp) cải thiện LSTM của bài gốc bao nhiêu so với NSE trung vị 0,71 và so với persistence; mô hình giữ được bao nhiêu độ chính xác khi mất Q hoặc tái phân tích đến trễ.
2. Trên quy trình đã cải tiến, lõi Mamba có hơn LSTM và các lõi khác không, hơn ở loại lưu vực nào, và chuỗi quá khứ dài hơn có giúp không.

Lưu ý: hiệu quả của các cải tiến trên dữ liệu LamaH-CE chưa được kiểm; bằng chứng ở trên đến từ các bộ dữ liệu khác (chủ yếu CAMELS-US và Thụy Sĩ).

## 9. Phạm vi đề xuất cho tiểu luận và khóa luận

Người dùng giao chọn phạm vi (26/9/2026). Nguyên tắc: tiểu luận làm trọn một câu chuyện "tái lập → cải tiến LSTM → so sánh lõi → giải thích → demo" với số lần huấn luyện vừa quota GPU miễn phí; mọi hướng cần thêm dữ liệu hoặc nhiều mã mới để cho khóa luận.

### 9.1. Tiểu luận

| Bước | Nội dung | Số lần huấn luyện (ước tính) | Sản phẩm |
|---|---|---|---|
| A. Dữ liệu và mốc | Khảo sát dữ liệu; chạy lại trọng số của tác giả trên test; persistence | 0 (chỉ suy luận) | Bảng mốc: số tác giả, số chạy lại, persistence |
| B. Tầng 1 — cải tiến LSTM | LSTM gốc chạy lại; + xử lý dữ liệu thiếu (masked mean, sửa bộ lọc mẫu); + hoàn thiện đầu vào lưu lượng (cùng đơn vị, Q trễ, xử lý Q thiếu); + kết hợp cả hai; + huấn luyện bỏ ngẫu nhiên Q để chịu mất lưu lượng quan trắc. Mỗi cấu hình 3 hạt giống, trung bình 3 hạt giống là kết quả tổ hợp | 5 cấu hình × 3 = 15 | Bảng đóng góp từng cải tiến; "LSTM cải tiến" |
| C. Tầng 2 — lõi thuần | GRU, Transformer, S4D, Mamba thay LSTM trên quy trình đã cải tiến; cùng ngân sách tinh chỉnh nhỏ (số chiều ẩn 64 / 128 / 256 trên validation) cho mọi lõi | 4 lõi × 3 = 12, cộng tinh chỉnh | Bảng so sánh kiến trúc |
| D. Chuỗi dài | LSTM và Mamba với hindcast 365 và 730 ngày | 2 × 3 = 6 | Trả lời "chuỗi dài hơn có giúp không" |
| E. Biến thể cải tiến | Một biến thể Mamba (ứng viên: quét hai chiều trong hindcast) và một biến thể Transformer (ứng viên: chia chuỗi thành đoạn — patch — như PatchTST, ICLR 2023), mỗi biến thể so với bản thuần của chính nó cùng số tham số; có cả hai để so sánh giữa các lõi ở mức cải tiến vẫn công bằng | 2 × 3 = 6 | Kiểm chứng một ý cải tiến cho mỗi lõi chính |
| F. Dự báo xác suất | Đầu CMAL trên cấu hình tốt nhất | 3 | Xác suất vượt ngưỡng lũ cho demo |
| Đánh giá mở rộng (cho B–F) | Kịch bản vận hành: mất Q 1, 3, 7 ngày cuối và mất hẳn; tái phân tích 5 ngày cuối của hindcast không có (ERA5 công bố trễ 5 ngày). So sánh theo lưu vực: ΔNSE giữa Mamba và LSTM cải tiến, đếm số lưu vực tốt lên / kém đi, tương quan Spearman với 33 thuộc tính và khoảng cách Wasserstein tái phân tích – dự báo (dùng lại mã phân tích của tác giả) | 0 (chỉ suy luận) | Độ chính xác khi dữ liệu đến trễ hoặc mất; Mamba hơn ở loại lưu vực nào |
| Đánh giá theo mức lưu lượng (cho A–F) | Ma trận nhầm lẫn, tỉ lệ phát hiện và báo động nhầm ở mức lũ, FHV, FLV (Mục 7) | 0 (chỉ suy luận) | Mô hình mạnh, yếu ở dải lưu lượng nào |
| G. XAI, ngưỡng lũ, demo | Integrated Gradients theo ngày và nhóm biến cho LSTM cải tiến và Mamba; ngưỡng chu kỳ lặp lại từ qmax năm 1981–2017; bản đồ 451 lưu vực | 0 | Hình XAI, ngưỡng, demo |

Tổng khoảng 42 lần huấn luyện cộng tinh chỉnh. Thời gian mỗi lần trên T4/P100 chưa đo (bài báo: vài phút tới khoảng 1 giờ trên RTX 4090) — đo ở lần chạy đầu của bước B để chỉnh số hạt giống nếu vượt quota.

Thứ tự cắt nếu thiếu thời gian (cắt từ trên xuống, không ảnh hưởng hai câu hỏi nghiên cứu chính ở Mục 8.5): (1) giải thích mô hình XAI ở bước G; (2) biến thể cải tiến ở bước E; (3) chuỗi 730 ngày ở bước D. Tầng 1, tầng 2 dạng thuần, đánh giá mở rộng, đánh giá theo mức lưu lượng và demo giữ nguyên.

Kết quả nào cũng thành câu trả lời: cải tiến tầng 1 giúp bao nhiêu; mô hình giữ được bao nhiêu độ chính xác khi mất lưu lượng quan trắc hoặc dữ liệu đến trễ; Mamba hơn, ngang hay kém LSTM trên cùng quy trình và ở loại lưu vực nào; chuỗi dài có giúp không.

### 9.2. Khóa luận

| Hướng | Lý do để sau |
|---|---|
| Ghép dữ liệu giờ vào quá khứ gần | Cần dựng bộ nạp ghép hai độ phân giải; dữ liệu đã có |
| Dự báo nhiều ngày | Cần nguồn dự báo nhiều lead (Caravan MultiMet) |
| Ưu tiên đỉnh lũ trong hàm mất mát | Có đánh đổi với kỹ năng chung, cần thí nghiệm riêng |
| Đồ thị mạng sông giữa các lưu vực lồng nhau | Cần dựng đồ thị và mô-đun không gian |
| Thêm biến thể lai (Mamba + attention, Mamba–Transformer, LOAN), xLSTM | Sau khi có kết quả dạng thuần |
| Nâng demo thành phần mềm ứng dụng | Yêu cầu của khóa luận |
