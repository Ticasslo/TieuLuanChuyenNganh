# Chương 5. Phương pháp

Chương này trình bày cách đề tài dựng mô hình và tổ chức thí nghiệm. Mục 5.1 mô tả bài toán và khung dự báo của BiasCast mà đề tài giữ lại. Mục 5.2 tách rõ khối giữ nguyên của bài gốc và khối đề tài thêm hoặc sửa. Mục 5.3 trình bày các thay đổi ở bộ nạp dữ liệu (bước B). Mục 5.4 trình bày mô hình chung nhận năm lõi thời gian (bước C). Mục 5.5 là giao thức huấn luyện và so sánh công bằng. Mục 5.6 và 5.7 mô tả hai mở rộng: dự báo 1–7 ngày (bước H) và cửa sổ quá khứ 730 ngày (bước D). Mục 5.8 là các kịch bản dữ liệu thiếu khi vận hành, Mục 5.9 là phương pháp giải thích mô hình (bước G). Lý thuyết của từng lõi và từng chỉ số ở Chương 2; số liệu dữ liệu ở Chương 4; kết quả ở Chương 6.

## 5.1. Bài toán và khung dự báo

**Bài toán.** Với mỗi trạm đo trong 451 lưu vực Extended LamaH-CE, dự báo lưu lượng đỉnh của ngày t (qmax, lưu lượng lớn nhất trong ngày) vào đầu ngày t, tức dự báo trước một ngày. Một mô hình chung được huấn luyện cho cả 451 lưu vực; đặc điểm từng lưu vực đưa vào qua 33 thuộc tính tĩnh (diện tích, độ cao, khí hậu, đất, thực vật…). Train là 2003–2009, validation 2010–2013, test 2014–2017 (BiasCast Mục 2.2; Chương 4).

**Khung hindcast–forecast.** Mỗi mẫu là một cửa sổ 365 ngày chia hai pha:

- **Pha quá khứ (hindcast), 364 ngày t−364 … t−1:** 31 biến khí tượng tái phân tích và quan trắc (ERA5-Land, E-OBS, MSWEP, GLEAM) cùng lưu lượng đo quá khứ. Đây là những gì đã biết tới hết ngày hôm qua.
- **Pha dự báo (forecast), ngày t:** 5 biến của dự báo thời tiết ECMWF HRES phát hành lúc 00 UTC ngày t (mưa, nhiệt độ, bức xạ…). Đây là thông tin về ngày cần dự báo mà người vận hành có thật vào sáng ngày t.

Ví dụ: để dự báo qmax ngày 15/06/2016, mô hình đọc dữ liệu khí tượng và lưu lượng từ 17/06/2015 tới 14/06/2016, cộng dự báo ECMWF HRES cho ngày 15/06/2016. Nhãn là qmax đo ngày 15/06/2016.

**Mô hình gốc của BiasCast** là Sequential Forecast LSTM: mỗi pha có một lớp nhúng tuyến tính riêng; hai chuỗi nhúng nối theo thời gian và đi qua **một** LSTM chạy liên tục, trạng thái cuối pha quá khứ chuyển thẳng sang bước dự báo; đầu ra ở bước cuối qua lớp tuyến tính cho qmax ngày t. BiasCast cho thấy cách chạy liên tục này bền hơn cách dùng hai mạng riêng cho hai pha (Encoder–Decoder): 435 so với 361 lưu vực cải thiện (BiasCast Mục 3.4; `01_BasePaper.md` Mục 11). Vì vậy đề tài giữ cách chạy liên tục cho mọi lõi.

**Hàm mất mát NSE\*** (Kratzert và cs., HESS 2019): sai số bình phương của từng mẫu chia cho (σ_b + ε)², với σ_b là độ lệch chuẩn qmax của lưu vực b trên kỳ train và ε = 0,1. Nhờ chia theo σ_b, lưu vực lớn (lưu lượng hàng trăm m³/s) không lấn át lưu vực nhỏ khi học; tối thiểu NSE\* tương đương tối đa NSE trung bình qua các lưu vực.

## 5.2. Khối giữ nguyên và khối thêm hoặc sửa

Theo góp ý của GVHD (biên bản họp 27/9/2026), kiến trúc tách rõ phần lấy từ bài gốc và phần đề tài làm. Sơ đồ đầy đủ ở `Diagrams/ProjectDiagram.drawio`, phần II: cột trái là khối của BiasCast đọc từ mã tác giả, cột phải là khối đề tài thay hoặc thêm, tô màu theo bước.

| Tầng | Giữ nguyên như BiasCast | Đề tài thêm hoặc sửa | Bước |
|---|---|---|---|
| Dữ liệu | 451 lưu vực, ba kỳ thời gian, 31 biến quá khứ, 5 biến ECMWF HRES, 33 thuộc tính tĩnh, z-score tính trên train | Q quá khứ đổi sang mm/ngày như nhãn; thêm qmax quá khứ; dựng lại thư mục dữ liệu có Q mà tác giả không công bố | B1a, B1b |
| Bộ nạp | Cách cắt cửa sổ, khởi động 364 ngày trước mỗi kỳ | Sửa bộ lọc mẫu để giữ mẫu thiếu một phần đầu vào; che dữ liệu ngẫu nhiên theo nguồn khi huấn luyện | B2, B3 |
| Nhúng | Lớp tuyến tính 16 chiều, nhúng tĩnh 16 chiều | Một lớp nhúng cho mỗi nhóm nguồn, gộp bằng masked mean | B2 |
| Lõi thời gian | LSTM một lớp, forget bias 3, chạy liên tục hai pha | Mô hình chung nhận 5 lõi: LSTM, GRU, Transformer, S4D, Mamba, cùng số tham số | C |
| Đầu ra | Lớp tuyến tính ở bước cuối, dropout 0,3 | Bước H: 7 đầu ra cho hạn 1–7, bộ đếm hạn | H |
| Huấn luyện | NSE\*, Adam lr 10⁻³, cosine 30 epoch, batch 256, cắt gradient 1 | Chạy đủ 30 epoch không dừng sớm; sửa lỗi lưu mô hình tốt nhất; 3 seed; tinh chỉnh cùng ngân sách cho mọi lõi | B0, C |
| Đánh giá | NSE, KGE, 𝒟FC, 𝒟RA trên test | Persistence, PNSE, DLinear, bốn mức lưu lượng, sự kiện lũ, kịch bản dữ liệu thiếu, kiểm định ghép cặp, giải thích mô hình | A, đánh giá, G |

Mọi thay đổi mã nằm trong bản fork NeuralHydrology của đề tài (`Ticasslo/neuralhydrology`, nhánh `research`); nhánh `master` giữ nguyên mã tác giả tại `9d94908`. Mỗi giai đoạn chạy trên một mốc cố định: `freeze-C` cho bước B, C, D; `freeze-H` và `freeze-H2` cho bước H (`02_Pipeline.md` Mục 9.1). Mọi mô-đun mới có kiểm thử đơn vị trước khi huấn luyện (285 kiểm thử ở `freeze-H`).

## 5.3. Cải tiến quy trình dữ liệu (bước B)

Bước B cải tiến Sequential Forecast LSTM trước khi so các lõi, để mọi lõi dùng chung một quy trình đã tốt nhất có thể (`02_Pipeline.md` Mục 1.4, nguyên tắc 2). Mỗi bước thêm đúng một thay đổi so với bước liền trước (ablation study), cùng 30 epoch và 3 seed, theo góp ý của GVHD (biên bản họp 25/10/2026).

| Bước | Thay đổi so với bước trước | Lý do |
|---|---|---|
| B0 | Mô hình gốc LSTM, sửa lỗi lưu mô hình tốt nhất, chạy đủ 30 epoch | Mốc của bước B; đo nhiễu giữa 3 seed dưới cùng giao thức |
| B1a | Lưu lượng trung bình quá khứ (`qmean`) đổi từ m³/s sang mm/ngày theo diện tích lưu vực | Nhãn đã ở mm/ngày; Q đầu vào theo m³/s làm lưu vực lớn có giá trị lớn hơn hẳn sau chuẩn hóa toàn cục |
| B1b | Thêm qmax quá khứ (mm/ngày) vào nhóm Q | Đỉnh hôm qua là thông tin gần nhất về đỉnh hôm nay |
| B2 | Mỗi nguồn dữ liệu một lớp nhúng, gộp bằng masked mean; sửa bộ lọc mẫu | Khi vận hành, dữ liệu mất theo cả nguồn (một nhà cung cấp chậm), mô hình phải chạy được khi thiếu một nguồn |
| B3 | Che dữ liệu ngẫu nhiên theo nguồn khi huấn luyện, hai mức | Dạy mô hình chịu được nguồn bị thiếu |

**Đơn vị của Q (B1a).** Đổi m³/s sang mm/ngày là chia thể tích nước chảy qua trạm trong một ngày cho diện tích lưu vực: Q [mm/ngày] = Q [m³/s] × 86.400 / (diện tích [km²] × 1.000). Ví dụ trạm Schruns (102 km²) với Q = 10 m³/s tương ứng 8,5 mm/ngày, nghĩa là lượng nước chảy ra trong ngày bằng một lớp nước dày 8,5 mm trải đều trên lưu vực. Sau khi đổi, lưu vực lớn và nhỏ có Q cùng thang đo với lượng mưa (cũng tính theo mm/ngày). Chương 6 Mục 6.6.3 cho thấy cải thiện tập trung ở lưu vực nhỏ.

**Nhóm theo nguồn và masked mean (B2).** Biến của pha quá khứ chia thành năm nhóm theo nhà cung cấp: ERA5-Land 21 biến, E-OBS 7, MSWEP 1, GLEAM 2, Q quan trắc 2; pha dự báo là nhóm ECMWF HRES 5 biến (`Workspace/05_Training/06_B2_LamaHCEExt_Train/06_B2_LamaHCEExt_Train.md`). Mỗi nhóm có lớp nhúng riêng ra 16 chiều. Ở mỗi ngày, vectơ nhúng là trung bình các nhóm có dữ liệu ngày đó; nhóm thiếu bị bỏ khỏi trung bình (masked mean, Gauch và cs., HESS 2025). Ví dụ: ngày ERA5-Land chưa phát hành, vectơ nhúng là trung bình bốn nhóm còn lại.

**Sửa bộ lọc mẫu (B2).** Mã gốc của NeuralHydrology loại cả mẫu nếu bất kỳ biến nào thiếu ở bất kỳ ngày nào trong 365 ngày, kể cả khi đã bật cách xử lý dữ liệu thiếu (`datasetzoo/basedataset.py`, hàm `_validate_samples`). Khi dùng masked mean, đề tài chỉ yêu cầu nhãn, thuộc tính tĩnh và đủ độ dài lịch sử hợp lệ; dữ liệu thiếu trong đầu vào do masked mean xử lý. Bộ lọc sửa được kiểm bằng 12 phép kiểm trên dữ liệu giả có lời giải biết trước (`Workspace/04_Setup/01_B2_LamaHCEExt_DataLoader/01_B2_LamaHCEExt_DataLoader.md` Mục 3.2).

**Che dữ liệu khi huấn luyện (B3).** Thư viện có sẵn hai xác suất: che từng ngày của một nhóm và che cả chuỗi của một nhóm (`basedataset.py`, `_add_nan_streaks`). Đề tài thử hai mức: 0,1/0,12 theo Gauch và cs. (2025), gọi là P10S12, và 0,05/0,05 theo MF²LSTM, gọi là P05S05.

**Chọn LSTM cải tiến B\*.** Luật chọn ghi trước khi chạy và chỉ dùng kỳ validation (`02_Pipeline.md` Mục 7): trong các cấu hình B1b, B2, B3 có NSE trung vị validation đủ dữ liệu (S0) kém cấu hình tốt nhất không quá nhiễu giữa seed của B0 (0,0079), chọn cấu hình có NSE trung vị trung bình cao nhất qua các kịch bản thiếu dữ liệu S1–S4 (Mục 5.8). Kết quả: **B\* = B3 P10S12**, 84.865 tham số (`Workspace/06_Evaluation/03_B_LamaHCEExt_Validation/03_B_LamaHCEExt_Validation.md` Mục 9.2). Luật ưu tiên khả năng chịu thiếu dữ liệu vì khi đủ dữ liệu, các cấu hình không phân biệt được.

## 5.4. Mô hình chung năm lõi (bước C)

Đề tài viết một mô hình chung "Sequential Forecast" trong fork, chọn lõi bằng khóa cấu hình `sf_core` (`modelzoo/sequential_forecast.py`). Mọi phần ngoài lõi giữ như B\*: lớp nhúng theo nguồn, masked mean, nhúng tĩnh, đầu ra, hàm mất mát. Chỉ lõi thời gian thay đổi.

| Lõi | Cài đặt | Vai trò trong so sánh |
|---|---|---|
| LSTM | `nn.LSTM` của PyTorch, forget bias 3 | Mốc chính. Với một lớp, mô hình chung trùng từng bit Sequential Forecast LSTM của tác giả cả khi dự báo bằng trọng số tác giả (451/451 lưu vực) và khi huấn luyện |
| GRU | `nn.GRU` | Hồi quy có ít cổng hơn LSTM: tách lợi thế của cấu trúc cổng LSTM khỏi lợi thế của họ mạng hồi quy |
| Transformer | Bộ mã hóa pre-LayerNorm, mặt nạ nhân quả, mã hóa vị trí sin, 4 đầu chú ý; lấy đầu ra bước cuối | Mốc dựa trên attention |
| S4D | Khối S4D (Gu và cs., NeurIPS 2022) có kết nối tắt và LayerNorm, N = 64 | SSM có tham số không đổi theo thời gian: tách câu hỏi "Mamba có hợp không" khỏi "SSM có hợp không" |
| Mamba | Khối `mamba_ssm.Mamba` (Gu & Dao, 2023) có kết nối tắt và LayerNorm, `d_state` 16, `d_conv` 4 | SSM chọn lọc, lõi chính được khảo sát |

Lớp Mamba có sẵn trong NeuralHydrology không dùng được vì quét theo trục đặc trưng thay vì trục thời gian (`01_BasePaper.md` Mục 11.5). Lõi Mamba của đề tài được kiểm là nhân quả (đổi dữ liệu tương lai không làm đổi đầu ra quá khứ) và quét đúng trục; bản PyTorch thuần `mambapy` nạp được cùng trọng số và cho cùng kết quả, dùng cho demo trên CPU (`Workspace/04_Setup/04_0_Cores_Check/04_0_Cores_Check.md`).

**Cùng số tham số.** Mỗi lõi chỉnh số chiều ẩn (và kích thước tầng trong của Transformer, Mamba) để tổng tham số của cả mô hình lệch không quá 1% so với LSTM của B\* (khoảng 85 nghìn). Nếu không làm vậy, một lõi thắng có thể chỉ vì nhiều tham số hơn. Bảng dưới là số chiều ứng với mốc LSTM 84.801 tham số của B1b (`Workspace/04_Setup/05_0_C_Configs_Check/05_0_C_Configs_Check.md` Phần 4); lệch lớn nhất 0,895%.

| Lõi | Độ sâu | Mức cùng tham số: chiều ẩn / tham số | Mức gấp 3: chiều ẩn / tham số |
|---|---|---|---|
| LSTM | 1 lớp | 128 / 84.801 | 235 / 254.824 |
| LSTM | 2 lớp | 77 / 84.042 | 139 / 253.736 |
| GRU | 1 lớp | 150 / 84.679 | 274 / 255.179 |
| GRU | 2 lớp | 90 / 84.439 | 162 / 255.583 |
| Transformer | 1 lớp | 80 / 84.864 | 144 / 254.438 |
| Transformer | 2 lớp | 56 / 84.759 | 100 / 254.361 |
| S4D | 2 khối | 111 / 84.757 | 216 / 254.017 |
| S4D | 4 khối | 72 / 84.385 | 146 / 255.769 |
| Mamba | 2 khối | 72 / 84.975 | 132 / 254.641 |
| Mamba | 4 khối | 49 / 84.525 | 91 / 254.613 |

Ở cùng số tham số, GRU có chiều ẩn lớn hơn LSTM (3 khối cổng so với 4), Mamba 4 khối có chiều ẩn nhỏ nhất vì mỗi khối tốn nhiều tham số.

## 5.5. Giao thức huấn luyện và so sánh

**Cấu hình huấn luyện chung** theo tác giả (BiasCast Mục 2.2): NSE\*, Adam learning rate 10⁻³, lịch cosine 30 epoch giảm tới 10⁻⁵, batch 256, cắt gradient ở chuẩn 1, dropout đầu ra 0,3.

**Chạy đủ 30 epoch, không dừng sớm.** BiasCast dừng khi NSE validation không tăng quá 0,005 trong 5 epoch, nên các lần chạy dừng ở số epoch khác nhau (lần huấn luyện lại A4: 16–20 epoch) và lịch cosine bị cắt giữa chừng. Đề tài chạy mọi lần huấn luyện đủ 30 epoch, tức hết đúng một chu kỳ cosine của tác giả, để các mô hình được so trên cùng số epoch (góp ý GVHD, biên bản họp 25/10/2026). Mô hình dùng để chấm (`best_model.pt`) là epoch có NSE trung vị validation cao nhất. NSE test được tính ở mọi epoch chỉ để theo dõi, không tham gia chọn.

**Tinh chỉnh cùng ngân sách (C1).** BiasCast tối ưu Bayes 100 lần cho LSTM (BiasCast phụ lục D), không khả thi cho 5 lõi trên GPU miễn phí. Đề tài cho mỗi lõi cùng 8 cấu hình, 1 seed:

- 2 mức dung lượng: cùng tham số (khoảng 85 nghìn) và gấp khoảng 3 lần (khoảng 254 nghìn);
- 2 độ sâu theo lõi: LSTM, GRU, Transformer 1 và 2 lớp; S4D, Mamba 2 và 4 khối. Độ sâu chọn theo lõi vì một lớp Transformer (attention cộng tầng truyền thẳng) tương đương hai khối Mamba về tham số (Gu & Dao, phụ lục), và các mô hình SSM trong thủy văn thường xếp 4–6 lớp (Wang và cs. 2025; Jing và cs. 2026);
- 2 learning rate: 10⁻³ và 5 × 10⁻⁴.

Siêu tham số ngoài lưới lấy theo mặc định mã tham chiếu của từng kiến trúc (dropout trong khối Transformer 0,1, S4D 0,1, Mamba 0). Cấu hình LSTM 1 lớp, cùng tham số, learning rate 10⁻³ chính là B\* nên không chạy lại; C1 gồm 39 lần chạy.

**Loại sớm hai vòng.** C1 chỉ để xếp hạng cấu hình, nên dùng cách loại sớm của tinh chỉnh siêu tham số (successive halving, Li và cs., JMLR 2018). Vòng 1 chạy cả 39 cấu hình 15 epoch đầu của lịch cosine 30 epoch; trong từng lõi, cấu hình cách cấu hình tốt nhất không quá nhiễu seed 0,0079 được giữ, và cấu hình tốt nhất ở mức cùng tham số luôn được giữ. Vòng 2 chạy tiếp các cấu hình được giữ tới epoch 30 bằng phần chạy tiếp của fork, khôi phục đủ trạng thái lịch learning rate và bộ sinh số ngẫu nhiên; chạy nửa rồi tiếp trùng từng bit với chạy liền ở LSTM (`test/test_resume.py`). Vì tương quan thứ hạng giữa epoch 15 và epoch 30 trên các cấu hình đã giữ thấp hơn mong đợi (Spearman 0,81), vòng 2 được mở rộng theo luật ghi sẵn cho 7 cấu hình bị loại nhưng có khả năng vượt (`02_Pipeline.md` Mục 5.2). Mọi cấu hình đi vào C2 đều đã chạy đủ 30 epoch.

**Chạy chính thức (C2).** Mỗi lõi chạy hai cấu hình với 3 seed (111, 222, 333): cấu hình tốt nhất ở mức cùng tham số và cấu hình tốt nhất cả lõi (trùng thì chỉ một). Kết quả là 8 cấu hình × 3 seed = 24 lần chạy (`Workspace/05_Training/13_C2_LamaHCEExt_Compare/13_C2_LamaHCEExt_Compare.md`):

| Lõi | Cấu hình C2 | Tham số |
|---|---|---|
| LSTM | 2 lớp, cùng tham số, lr 10⁻³ | 84.106 |
| GRU | 2 lớp, cùng tham số, lr 10⁻³ | 84.503 |
| Transformer | 2 lớp, cùng tham số, lr 10⁻³; 2 lớp, gấp 3, lr 5 × 10⁻⁴ | 84.823; 254.425 |
| S4D | 4 khối, cùng tham số, lr 10⁻³; 4 khối, gấp 3, lr 10⁻³ | 84.449; 255.833 |
| Mamba | 2 khối, cùng tham số, lr 10⁻³; 4 khối, gấp 3, lr 10⁻³ | 85.039; 254.677 |

Một lần chạy GRU (seed 222) dừng giữa chừng vì gradient tràn số float32 ở epoch 14. Lần chạy lại bật một chốt chặn: bước huấn luyện nào có gradient NaN hoặc vô cực thì bỏ qua, không cập nhật trọng số; chỉ 1 bước bị bỏ trong 30 epoch.

**Ngưỡng nhiễu seed.** Ba lần chạy chỉ khác seed vẫn cho NSE trung vị validation khác nhau: độ lệch chuẩn của B0 là 0,0079. Đề tài dùng con số này làm thước đo: chênh lệch giữa hai mô hình nhỏ hơn 0,0079 không được kết luận là hơn hay kém. Kiểm định Wilcoxon trên 451 lưu vực không đủ để kết luận, vì A4 cho thấy hai lần chạy chỉ khác seed vẫn có p sau hiệu chỉnh Holm dưới 0,05; do đó mọi so sánh ghi kèm độ lớn của hiệu và bội số so với nhiễu seed (`02_Pipeline.md` Mục 6.1).

**Ba tầng chỉ số.** Chỉ NSE trung vị validation được dùng để chọn (epoch, cấu hình, B\*, mô hình demo). Trên test, các lõi được so bằng nhiều chỉ số (NSE, KGE, PNSE, %BiasFHV, bốn mức lưu lượng, sự kiện lũ). Test không bao giờ dùng để chọn.

## 5.6. Dự báo 1–7 ngày (bước H)

Bước H mở rộng bài toán từ dự báo qmax ngày t sang ngày t tới t+6. Cửa sổ dài 371 ngày: 364 ngày quá khứ và 7 ngày dự báo; mô hình có 7 đầu ra, hàm mất mát là NSE\* lấy trung bình trên 7 hạn (lựa chọn của đề tài). Bộ đếm hạn có sẵn trong thư viện (`timestep_counter`) cho mô hình biết bước đang xét là hạn thứ mấy. Epoch tốt nhất chọn theo trung vị qua lưu vực của NSE trung bình 7 hạn trên validation (`NSE_lead_mean`).

Extended LamaH-CE chỉ có dự báo ECMWF HRES cho ngày t. Vì vậy đề tài so bốn cách cấp đầu vào cho 7 ngày của pha dự báo:

| Cách | Ngày t | Ngày t+1 … t+6 | Vai trò |
|---|---|---|---|
| Chỉ quá khứ | Không có dữ liệu thời tiết | Không có | Mốc khi không có dự báo thời tiết |
| Lai | ECMWF HRES thật | Để trống, masked mean xử lý | Vận hành được với dữ liệu sẵn có |
| GEFS | ECMWF HRES thật và GEFSv12 hạn 1 | GEFSv12 hạn k, cùng lần phát 00 UTC ngày t | Vận hành được với dự báo nhiều ngày thật |
| Hoàn hảo | 5 biến tái phân tích của chính ngày đó | Tái phân tích | Cận trên, không vận hành được (biết trước thời tiết thật) |

GEFSv12 reforecast của NOAA là bộ dự báo thời tiết nhiều ngày lưu trữ phủ đủ 2003–2017, phát hằng ngày, tải tự do. Đề tài tự tải và gộp về từng lưu vực (`03_Data/01_LamaHCE.md` Mục 7). Ở bước của hạn k, mô hình chỉ thấy nhóm GEFS hạn k của lần phát ngày t, không thấy dự báo phát sau ngày t; ràng buộc này được kiểm bằng kiểm thử đơn vị (`datautils/forecast_inputs.py`).

Hai lõi của bước H là LSTM `lstm_L2_x1_lr1e-3` và Mamba `mamba_L2_x1_lr1e-3`, cả hai ở mức cùng tham số, để khác biệt theo hạn chỉ do lõi, không lẫn dung lượng. Mamba 4 khối gấp 3 có NSE validation cao hơn 0,0032, chỉ bằng 0,40 lần nhiễu seed, nhưng đắt hơn 2,7 lần. Bước H gồm 2 lõi × 4 cách × 3 seed = 24 lần chạy. Bộ đếm hạn làm số tham số tăng nhẹ (LSTM 84.414, Mamba 85.111); cách GEFS thêm 7 lớp nhúng (LSTM 85.086, Mamba 85.783). Mọi so sánh trong bước H lệch tham số dưới 1%.

## 5.7. Cửa sổ quá khứ 730 ngày (bước D)

Bước D hỏi: mô hình có lợi khi nhìn lại 2 năm thay vì 1 năm không? Cửa sổ dài 730 ngày gồm 729 ngày quá khứ và ngày dự báo t; mọi thứ khác giữ như C2 (cấu hình, seed, 30 epoch). Ba lõi ở mức cùng tham số: LSTM, Mamba và S4D, cùng cấu hình C2 (`lstm_L2_x1_lr1e-3`, `mamba_L2_x1_lr1e-3`, `s4d_L4_x1_lr1e-3`), 3 seed, tổng 9 lần chạy. S4D có mặt vì là SSM không chọn lọc, đối chứng cho Mamba. Bản 365 ngày để so là 3 run C2 cùng cấu hình.

Hai độ dài dùng cùng bộ lọc mẫu đã sửa ở B2. Với bộ lọc gốc, cửa sổ 730 ngày bỏ gần hết mẫu train năm 2003 vì ECMWF HRES chỉ có từ 01/01/2002 (`03_Data/01_LamaHCE.md` Mục 6.1); bộ lọc đã sửa giữ các mẫu đó, nên hai bên học trên cùng tập mẫu và chỉ khác độ dài cửa sổ.

## 5.8. Kịch bản dữ liệu thiếu khi vận hành

Khi vận hành, mỗi nguồn dữ liệu đến trễ một khoảng khác nhau. Đề tài chạy lại mô hình đã huấn luyện (chỉ suy luận, không huấn luyện thêm) dưới các kịch bản xóa dữ liệu tương ứng (`02_Pipeline.md` Mục 3.2, 6.3):

| Kịch bản | Dữ liệu bị xóa | Tình huống thực tế |
|---|---|---|
| S0 | Không | Đủ dữ liệu, như bài gốc |
| S1 | Q của 1, 3 hoặc 7 ngày cuối | Trạm đo truyền số liệu chậm hoặc hỏng ngắn hạn |
| S2 | Q toàn cửa sổ | Trạm hỏng dài hạn |
| S3 | ERA5-Land 5 ngày cuối; GLEAM toàn cửa sổ | Độ trễ phát hành thật: ERA5-Land khoảng 5 ngày (Copernicus); GLEAM phát hành mỗi năm một lần (ICDC Hamburg), giả định xấu nhất là thiếu cả cửa sổ. E-OBS có bản hằng ngày trễ 1 ngày và MSWEP có bản gần thời gian thực trễ 2–3 giờ nên giữ đủ tới t−1 |
| S4 | S3 và thêm mất Q ngày cuối | Kết hợp |

Mô hình có masked mean (B2 trở đi) bỏ nhóm thiếu khỏi trung bình. Mô hình không có masked mean (B1b) được điền giá trị quá khứ gần nhất còn thấy, dự phòng bằng giá trị khí hậu cùng ngày trong năm tính trên train. Kịch bản chạy trên validation (để chọn B\*) và trên test (chỉ để báo cáo).

## 5.9. Giải thích mô hình (bước G)

Bước G giải thích các mô hình đã huấn luyện, không huấn luyện thêm: 15 run C2 mức cùng tham số (5 lõi × 3 seed) và 24 run H (riêng từng hạn 1–7).

- **Integrated Gradients** (Sundararajan và cs., ICML 2017) qua thư viện Captum 0.9.0: tính mức ảnh hưởng của từng ô đầu vào (một biến ở một ngày) lên dự báo, bằng cách tích phân gradient trên đường thẳng từ một đầu vào gốc tới đầu vào thật. Đầu vào gốc là trung bình kỳ train. Mỗi mẫu được kiểm completeness: tổng ảnh hưởng phải bằng chênh lệch dự báo giữa đầu vào thật và đầu vào gốc.
- **Kiểm độ tin cậy bằng xóa ô (deletion test):** xóa dần các ô có ảnh hưởng lớn nhất và so mức thay đổi dự báo với hai đối chứng xóa ngẫu nhiên. Ảnh hưởng đúng thì xóa theo ảnh hưởng phải làm dự báo đổi nhanh hơn xóa ngẫu nhiên.
- **Chỉ số rút ra:** tỉ phần ảnh hưởng theo ngày trong cửa sổ; độ dài nhớ hiệu dụng L50, L90 (số ngày gần nhất chứa 50%, 90% ảnh hưởng của pha quá khứ); tỉ phần theo nhóm nguồn; ở bước H thêm tỉ phần của thông tin tương lai theo từng hạn.
- **Dò trạng thái ẩn** (theo Hu và cs., Environmental Modelling & Software 2025): hồi quy ridge từ trạng thái ẩn của lõi ở ngày t−1 sang các đại lượng thủy văn không có trong đầu vào (mưa cộng dồn 30, 90, 180 ngày), so với hai đối chứng: đầu vào thô ngày t−1 và cùng mô hình khởi tạo ngẫu nhiên. Trạng thái ẩn chỉ được coi là "nhớ" một đại lượng khi dò tốt hơn cả hai đối chứng.

Mẫu test được lấy phân tầng theo bốn mức lưu lượng (Chương 4 Mục 4.6.1) để đủ ngày lũ. Các lõi được so bằng hiệu so với nhiễu 3 seed như Mục 5.5.

## 5.10. Hạ tầng tính toán

Mọi lần huấn luyện chạy trên GPU NVIDIA T4 miễn phí của Kaggle (tối đa 30 giờ mỗi tuần, 12 giờ mỗi phiên); `mamba-ssm` 2.3.2.post1 được tự biên dịch cho T4 và PyTorch 2.11 (`Workspace/04_Setup/01_0_MambaSSM_Build/01_0_MambaSSM_Build.md`). Toàn bộ dữ liệu train được đặt sẵn trên GPU (khóa `data_on_gpu` của fork): kết quả trùng từng bit với bộ nạp gốc, một epoch huấn luyện nhanh gần gấp đôi (7,3 xuống 3,9 phút) (`Workspace/04_Setup/02_0_DataLoader_Benchmark/02_0_DataLoader_Benchmark.md` Mục 5). Lần chạy vượt 12 giờ được chia qua hai phiên bằng phần chạy tiếp của fork. Tổng cộng đề tài huấn luyện 120 lần: A 6, B 18, C1 39, C2 24, H 24, D 9 (`02_Pipeline.md` Mục 7).
