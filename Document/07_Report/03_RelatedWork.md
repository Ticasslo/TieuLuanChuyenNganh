# Chương 3. Công trình liên quan

Chương này đặt đề tài trong các nhóm công trình gần nhất: học sâu cho dòng chảy với LSTM làm mốc (Mục 3.1), bài cơ sở BiasCast (3.2), xử lý dữ liệu thiếu khi vận hành (3.3), mô hình không gian trạng thái và Mamba (3.4), dự báo nhiều ngày (3.5). Mục 3.6 tóm tắt khác biệt của đề tài so với từng công trình. Ghi chú đọc chi tiết từng bài ở `05_Survey/03_PaperNotes.md`; khảo sát theo yêu cầu của GVHD ở `05_Survey/02_LiteratureReview.md`.

## 3.1. Học sâu cho dòng chảy: LSTM là mốc chuẩn

- **Kratzert và cs. (HESS 23:5089, 2019)** huấn luyện một LSTM chung cho 531 lưu vực CAMELS-US, mỗi lưu vực được mô tả bằng thuộc tính tĩnh. Mô hình chung tốt hơn các mô hình thủy văn hiệu chỉnh riêng từng lưu vực. Bài cũng đưa ra hàm mất mát NSE\*, chia sai số của mỗi lưu vực cho độ lệch chuẩn lưu lượng của nó, mà BiasCast và đề tài dùng lại.
- **Nearing và cs. (Nature 627, 2024)** dùng LSTM mã hóa – giải mã cho hệ thống dự báo lũ toàn cầu của Google, dự báo tới 7 ngày ở các lưu vực không có trạm đo. Kiến trúc mã hóa – giải mã này là một trong các mô hình BiasCast so sánh.
- **Kratzert và cs. (HESS 28:4187, 2024)** cho thấy một mô hình LSTM huấn luyện trên càng nhiều lưu vực càng tốt, kể cả cho từng lưu vực riêng. Đây là căn cứ để dùng một mô hình chung cho cả 451 lưu vực.
- **Thư viện NeuralHydrology** (Kratzert và cs., JOSS 2022) cài đặt sẵn các mô hình trên và là nền mã của BiasCast lẫn đề tài.

Về các kiến trúc khác LSTM: Gao và cs. (J. Hydrol. 589, 2020) thấy GRU ngang LSTM với ít tham số hơn; Liu và cs. (J. Hydrol. 2024) thấy Transformer cơ bản kém LSTM trên CAMELS, nhất là ở dòng chảy cao, và phải thêm lớp nhúng tích chập nhân quả mới ngang LSTM (KGE 0,74 so với 0,73). Các so sánh này không giữ cùng số tham số giữa các mô hình.

## 3.2. Bài cơ sở: BiasCast

**Konold và cs., Hydrology and Earth System Sciences 30:5067–5096, 2026** (Scimago Q1). BiasCast hỏi: mô hình học sâu thường được huấn luyện trên khí tượng tái phân tích (dữ liệu quá khứ đã hiệu chỉnh), nhưng khi vận hành lại chạy với dự báo thời tiết, kém chính xác hơn. Mô hình suy giảm bao nhiêu, và kiến trúc nào tự học bù được sai lệch của dự báo? Dữ liệu là Extended LamaH-CE, 451 lưu vực Trung Âu theo ngày, có dự báo ECMWF HRES lưu trữ cho ngày cần dự báo.

NSE trung vị trên test 2014–2017 (BiasCast Bảng F1):

| Mô hình | NSE trung vị |
|---|---|
| Học trên tái phân tích → chạy với dự báo thật | 0,58 → 0,33 |
| Chỉ dùng dự báo thời tiết (cận dưới) | 0,39 |
| Học chuyển giao tái phân tích → dự báo | 0,41–0,44 |
| Encoder–Decoder LSTM | 0,57 |
| Sequential Forecast LSTM | 0,63 |
| Encoder–Decoder LSTM có lưu lượng quan trắc | 0,66–0,67 |
| Tái phân tích với thời tiết hoàn hảo (cận trên, không vận hành được) | 0,69 |
| **Sequential Forecast LSTM có lưu lượng quan trắc** | **0,70–0,71** |

Kết luận chính của BiasCast: không thể huấn luyện trên tái phân tích rồi chạy với dự báo; Sequential Forecast LSTM (một LSTM chạy liên tục qua quá khứ và ngày dự báo) bù sai lệch tốt nhất; thêm lưu lượng quan trắc gần nhất vượt cả cận trên dùng thời tiết hoàn hảo.

**Lý do chọn làm bài cơ sở** (`01_BasePaper.md` Mục 8): đúng bài toán dự báo vận hành (có lưu lượng quan trắc và dự báo thời tiết thật); số liệu tính lại từ tệp kết quả của tác giả khớp bài; mã, cấu hình, trọng số và kết quả theo lưu vực đều công khai; dữ liệu 0,95 GB chạy được trên GPU miễn phí; phản biện công khai hai vòng; và khoảng trống do chính tác giả nêu (chỉ LSTM, chỉ dự báo 1 ngày, chỉ dữ liệu ngày) đúng chỗ đề tài mở rộng.

**Hạn chế phát hiện khi đọc mã và phản biện** (`01_BasePaper.md` Mục 9):

- Mỗi cấu hình chỉ chạy một seed, nên chênh lệch 0,01–0,02 NSE giữa các biến thể chưa tách được khỏi nhiễu.
- Không có mốc persistence, dù người phản biện 1 đề nghị. Đề tài tự tính: persistence đạt NSE trung vị 0,35–0,37.
- Lưu lượng quan trắc đầu vào giữ đơn vị m³/s trong khi nhãn ở mm/ngày; thư mục dữ liệu có lưu lượng mà mã đọc không được công bố.
- Hàm lưu mô hình tốt nhất không tính ngưỡng cải thiện tối thiểu của dừng sớm; bộ lọc mẫu loại cả mẫu khi thiếu một giá trị.
- Phân phối NSE của mô hình tốt nhất có đuôi trái nặng: trung vị 0,71 nhưng trung bình 0,40, nghĩa là vài lưu vực có NSE âm rất lớn.

Các hạn chế này không làm sai số liệu công bố; phần lớn là chỗ đề tài bổ sung.

**Kirschstein & Sun (ICML 2024)**, ứng viên bài cơ sở khác, dùng mạng nơ-ron đồ thị trên mạng sông của LamaH-CE theo giờ. Bài không được chọn vì công thức NSE trong mã tính sai, nhiễu giữa các lần huấn luyện lớn cỡ toàn bộ chênh lệch trong bảng kết quả, và bài toán không dùng dự báo thời tiết. Bài được giữ làm tham khảo cho hướng đồ thị ở khóa luận.

## 3.3. Dữ liệu thiếu khi vận hành

- **Gauch và cs. (HESS 2025)** đề xuất masked mean: mỗi nguồn khí tượng một lớp nhúng, ở mỗi bước lấy trung bình các nguồn có dữ liệu, kèm che ngẫu nhiên khi huấn luyện (xác suất 0,1 theo bước, 0,12 theo chuỗi). Bài làm trên bài toán mô phỏng, CAMELS-US. Đề tài áp dụng cách này cho bài toán dự báo có dự báo thời tiết thật, nhóm theo nhà cung cấp dữ liệu, cho mọi lõi.
- **MF²LSTM** (preprint EGUsphere 2026) che lưu lượng quan trắc khi huấn luyện để mô hình dùng được cả khi có và không có số liệu trạm, theo giờ, một mô hình LSTM. Mức che 0,05/0,05 của bài là một trong hai mức đề tài thử ở B3.
- **Dubey và cs. (arXiv 2025)** đánh giá LSTM mã hóa – giải mã khi dữ liệu đến trễ hoặc mất, trên CAMELS-US, HYSETS, CAMELS-IND. Đề tài khác ở chỗ dùng độ trễ phát hành riêng của từng nguồn (ERA5-Land khoảng 5 ngày, GLEAM mỗi năm một lần…) và so nhiều lõi.
- **Saint-Fleur và cs. (HESS 30:3497–3527, 2026)** so ba cách đưa lưu lượng quan trắc vào dự báo 1–7 ngày với LSTM trên CAMELS-US, CAMELS-FR; lưu lượng chủ yếu giúp ở lưu vực mà LSTM vốn kém. Đề tài đo mức phụ thuộc vào lưu lượng quan trắc của năm lõi bằng kịch bản mất Q.

## 3.4. Mô hình không gian trạng thái và Mamba

**Nền tảng.** S4 và biến thể đường chéo S4D (Gu và cs., NeurIPS 2022) biểu diễn chuỗi bằng một hệ tuyến tính có trạng thái ẩn, tham số không đổi theo thời gian, tính được như tích chập dài bằng FFT. Mamba (Gu & Dao, 2023) cho các tham số phụ thuộc đầu vào (cơ chế chọn lọc), để mô hình tự quyết giữ hay quên thông tin theo từng bước, với chi phí tuyến tính theo độ dài chuỗi. Lý thuyết ở Chương 2.

**Chuỗi thời gian nói chung.** Bằng chứng chưa thống nhất (`05_Survey/02_LiteratureReview.md` Mục 1.1). Wang và cs. (Neurocomputing 2025, *Is Mamba Effective for Time Series Forecasting?*) thấy Mamba ngang Transformer, không vượt trội rõ ràng. Các biến thể cải tiến như CMMamba (Journal of Big Data 2024) và DTMamba (Tsinghua Science and Technology 2026) vượt Transformer trên các bộ dữ liệu chuẩn. Xu hướng gần đây là lai Mamba–Transformer (SST, CIKM 2025). Yu & Erichson (NeurIPS 2025) chỉ ra Mamba kém S4D ở tác vụ phụ thuộc xa, nên đề tài đưa S4D vào làm đối chứng SSM không chọn lọc.

**Dòng chảy.** Các công trình dùng Mamba hoặc SSM cho lưu lượng:

| Công trình | Bài toán và dữ liệu | Kết quả |
|---|---|---|
| RiverMamba (Shams Eddin và cs., NeurIPS 2025) | Mamba quét không gian trên lưới 0,05° toàn cầu, nhãn là tái phân tích GloFAS, đánh giá trên trạm GRDC | R² so với GloFAS 0,289 → 0,506, so với LSTM 0,462 → 0,506 |
| Shahriar (preprint SSRN 2026) | Mamba-3 và LSTM cùng tham số (lệch dưới 0,5%), cùng độ sâu, 671 lưu vực CAMELS, mô phỏng | LSTM thắng mọi cặp (NSE 0,752 so với 0,727); Mamba quá khớp sau khoảng 10 epoch |
| Zhang và cs. (J. Hydrol. 2026) | LSTM, Transformer, PatchTST, Mamba, DLinear theo giờ, CAMELS | Mamba gần nhưng hơi kém LSTM; có Q quá khứ thì PatchTST tốt nhất từ hạn 3 giờ |
| HydroDiffusion (Wang và cs., WRR 2026) | Mô hình khuếch tán với lõi S4D-FT so với LSTM, dự báo 0–7 ngày bằng GEFSv12 reforecast, 531 lưu vực CAMELS, không dùng Q quan trắc | SSM nhỉnh hơn LSTM (NSE ngày 0 trung vị 0,75 so với 0,73) |
| Jing và cs. (ESWA 2026) | S4D làm bộ mã hóa trong mô hình lai học tham số vi phân, 531 lưu vực CAMELS-US | NSE trung vị 0,742 (LSTM) → 0,756 (S4D), lợi nhiều ở lưu vực tuyết, núi; kết quả trong khung lai, không phải mô hình thuần |
| Demiray & Demir (preprint EarthArXiv 2025) | Mamba so với LSTM, GRU, Transformer, 125 trạm Iowa theo giờ, có SHAP | Mamba tương đương, một số khía cạnh nhỉnh hơn Transformer |
| ResBi-Mamba Plus (AISE 2026) | Mamba hai chiều, dự báo theo giờ trên sông Columbia | Mamba hai chiều trong cửa sổ quá khứ không rò rỉ tương lai |

Như vậy, các so sánh cùng số tham số (Shahriar, Zhang) nghiêng về LSTM; các kết quả SSM tốt hơn đến từ biến thể đã cải tiến hoặc bài toán khác (lưới không gian, mô hình khuếch tán). Chưa có công trình nào so các lõi này trong bài toán dự báo dùng dự báo thời tiết lưu trữ thật kèm lưu lượng quan trắc, dưới cùng ngân sách tinh chỉnh và nhiều seed.

## 3.5. Dự báo nhiều ngày

Extended LamaH-CE chỉ có dự báo ECMWF HRES cho ngày đầu (BiasCast Mục 2.1); tác giả BiasCast nêu dự báo nhiều ngày là hướng tiếp theo (BiasCast Mục 4). Nearing và cs. (2024) dự báo tới 7 ngày với dự báo thời tiết của ECMWF. HydroDiffusion (Wang và cs., 2026) dùng GEFSv12 reforecast, bộ dự báo lại 2000–2019 của NOAA phát hằng ngày, tải tự do, cho dự báo 0–7 ngày. Đề tài dùng cùng nguồn GEFSv12 cho bước H, nhưng kèm ECMWF HRES ngày đầu và lưu lượng quan trắc, và so với ba cách cấp thời tiết khác để tách phần đóng góp của dự báo nhiều ngày.

## 3.6. Vị trí của đề tài

| Công trình gần nhất | Đề tài khác ở chỗ |
|---|---|
| BiasCast | Thêm 4 lõi dưới giao thức công bằng; 3 seed; mốc persistence, DLinear; sửa điểm yếu mã; dự báo 1–7 ngày; đánh giá theo mức và sự kiện lũ |
| Shahriar 2026; Zhang 2026 | Bài toán dự báo vận hành với dự báo thời tiết thật và Q quan trắc thay vì mô phỏng; thêm S4D làm đối chứng; kịch bản dữ liệu thiếu |
| HydroDiffusion 2026 | Dùng Mamba (SSM chọn lọc) cùng S4D; có Q quan trắc; dữ liệu Trung Âu; so 5 lõi cùng tham số có kiểm định |
| Demiray & Demir 2025 | Giao thức cùng tham số; S4D; giải thích tách đóng góp của dự báo thời tiết và Q theo hạn bằng Integrated Gradients, so độ dài nhớ của 5 lõi với nhiễu seed |
| Gauch 2025; MF²LSTM; Dubey 2025 | Masked mean và che dữ liệu cho bài toán dự báo theo ngày, cho mọi lõi; kịch bản theo độ trễ phát hành thật của từng nguồn |
| RiverMamba 2025 | Theo lưu vực thay vì lưới, nhãn là Q quan trắc, so sánh có kiểm soát với các lõi khác |

Đề tài vì vậy không khẳng định Mamba tốt hơn. Đóng góp là trả lời có kiểm soát câu hỏi: lõi SSM có đáng thay LSTM trong dự báo vận hành không, ở đâu và với điều kiện nào (`02_Pipeline.md` Mục 1.3). Nhận định "chưa có công trình nào" ở trên dựa trên tra cứu Google Scholar, Semantic Scholar, arXiv, OpenAlex, IEEE Xplore và ACM DL (`05_Survey/02_LiteratureReview.md` Mục 1.4).
