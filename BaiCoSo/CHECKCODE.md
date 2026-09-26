# Báo cáo đọc mã nguồn các bài báo ứng viên làm bài cơ sở

> Ngày lập: 26/9/2026. Tài liệu kiểm tra mã nguồn thật của từng ứng viên về tính đúng đắn phương pháp (chia tập, chuẩn hóa, chọn mô hình, cách tính chỉ số), mức độ khớp với mô tả trong bài, khả năng tái lập trên Google Colab và vị trí chèn khối Mamba. Phân tích toàn văn và xếp hạng nằm ở `CHECKPDF.md`. Mã nguồn lưu tại thư mục `Code/` (không đưa lên git).
>
> Phạm vi đọc: toàn bộ đường ống dữ liệu → mô hình → huấn luyện → đánh giá → script chạy của mỗi repo. Không đọc từng dòng các thư viện bên thứ ba được chép nguyên vào repo mà đường ống không gọi tới (`22_S4D-FT/papercode/src`, `42_TFRN/utils/notebooks/backend`, phần nội tại các mô hình Time-Series-Library của #6, notebook vẽ hình của #38). Chạy thử thực hiện trên CPU (PyTorch 2.14) trong môi trường riêng, không nằm trong repo.

## 0. Tổng hợp

| # | Repo | Chia tập | Chuẩn hóa theo train | Chọn mô hình | Kèm dữ liệu | Vấn đề chính |
|---|---|---|---|---|---|---|
| #104 | `Code/104_ICML-RiverTopology` | Test 2016–2017, train theo năm | Có (2000–2015) | Loss nhỏ nhất trên 1/5 cửa sổ train ngẫu nhiên | Tự tải LamaH-CE; kèm 957 checkpoint | Công thức NSE sai; validation không độc lập; cửa sổ cắt trong từng năm |
| #121 | `Code/BiasCast` + fork NeuralHydrology + Zenodo 17292895 | 2003–2009 / 2010–2013 / 2014–2017 | Có (scaler lưu kèm) | Theo validation (NeuralHydrology) | Extended LamaH-CE 0,95 GB (Zenodo 17119635) | 1 hạt giống; chưa chạy thử |
| #109 + #56 | `Code/OpenHydroNet` (Google, Apache 2.0) | Cấu hình vận hành: train = validation = test 1982–2023; cấu hình CAMELS-US: validation trùng test 2021–2025 | Có | Theo cấu hình (NeuralHydrology) | Mẫu Caravan nhỏ; MultiMet zarr công khai; kèm 2 bộ trọng số (huấn luyện trên toàn bộ 1982–2023) | Không có mốc công bố tái lập được; chỉ dữ liệu ngày; không dùng Q quan trắc |
| #67 | Hy2DL + Zenodo `14780059` | 1990–2003 / 2003–2008 / 2008–2018 | Có | Epoch cuối | Tự tải CAMELS-US giờ | Không có vấn đề phương pháp; dữ liệu ~20 GB |
| #22 | `Code/22_S4D-FT` | Train 1999–2008 / test 1989–1999 | Có (hằng số) | Epoch cố định 49; validation 10% cửa sổ train ngẫu nhiên | Tự tải CAMELS + NLDAS mở rộng | Script không khớp Bảng S3; một số tiện ích hỏng |
| #103 | FloodGNNs | Theo tỉ lệ thời gian | Có | Theo validation | Dữ liệu đã xử lý trên Google Drive | Lỗi import; NSE gộp; loss khác bài |
| #71/#54 | FHNN (GitHub) | 1985–1993 / 1993–1995 / 1995–2005 | Có | Loss validation nhỏ nhất | Không (thiếu tiền xử lý) | Không tái lập được số liệu |
| #53 | `Code/53_HydroTFT` | Không có validation | Hằng số toàn cục | Chọn epoch theo NSE trên kỳ test | Không | Chọn mô hình trên tập test |
| #42 | `Code/42_TFRN` | 1980–1995 / 1995–2000 / 2000–2014 | Có | MSE validation nhỏ nhất | Không | Lỗi vòng lặp DiT; NSE gộp |
| #6 | `Code/06_RNNs-to-Transformers` | Có | Có | Epoch cuối | Có (`CAMELS.nc`) | Hàm mất mát khác nhau giữa LSTM và Transformer |
| #36 | dmg-research (GitHub) | 1980–1995 / 1995–2010 | Có | Epoch cố định 100 | Không | Cần gói `dmg`, PyTorch 2.9.1 |
| #135 | `Code/135_TSFM` | 1980–1992 / 1992–1995 / 1995–2005 | Có | MSE validation + dừng sớm | Không | Thiếu mã phần đa biến |
| #38 | `Code/38_CNN-LSTM` | Kratzert + PUB 12 phần | Có (hằng số) | Epoch cố định 30 | Không (raster tự tạo) | Cấu hình raster mặc định lệch bài |
| — | `Code/MTPre` | 1980–1995 / 1995–1999 / 1999–2014 | Có | NSE-like validation | Không | Nhánh EMD rò rỉ tương lai |

---

## 1. #104 — River Network Topology, ICML 2024

**Nguồn.** `github.com/nkirschi/neural-flood-forecasting`, đẩy lần cuối 31/03/2025, không có tệp LICENSE. Bản trong `Code/` (3,4 GB) kèm checkpoint (1,9 GB) và kết quả. Mã ~880 dòng Python (`dataset.py`, `functions.py`, `models.py`, 8 script `train_*`/`test_*`) và 8 notebook.

**Dữ liệu.** `LamaHDataset` (PyTorch Geometric) tự tải `1_LamaH-CE_daily_hourly.tar.gz` (Zenodo `5153305`, 14,8 GB). Bắt đầu từ trạm 399, duyệt ngược dòng lấy thành phần liên thông; trạm hợp lệ khi lưu lượng > 0 trên toàn chuỗi và đủ (18×365+5)×24 giờ trong 2000–2017; trạm không hợp lệ bị xóa và nối lại cạnh. Toàn bộ dữ liệu ≈ 358 × 157.800 × 5 float32 ≈ 1,1 GB.

**Chia tập và chuẩn hóa.** Test luôn là 2016–2017; train theo danh sách năm của từng fold. Thống kê z-score tính trên 2000–2015 cho từng trạm (với fold năm chẵn/lẻ, thống kê có gồm năm không dùng để train nhưng không gồm năm test). Cửa sổ cắt **bên trong từng năm** (`year_tensors`; số mẫu mỗi năm = (số giờ − (W + L)) / stride + 1): cửa sổ không vượt ranh giới năm, W ≥ 1 năm không chạy được.

**Mô hình.** `construct_model`: số kênh vào = `window_size × 5`; `BaseModel.forward` làm `x.flatten(1)` rồi qua `Linear` (bộ mã hóa), N lớp GNN, `Linear` giải mã. Không có lớp hồi quy, tích chập hay attention theo thời gian; `models.py` import `LSTM` nhưng không dùng.

**Huấn luyện.** `train`: `random_split` 1/5 cửa sổ làm holdout (cửa sổ chồng lấn cùng kỳ train), lưu `best_model_params` ở epoch có loss holdout nhỏ nhất. Loss = MSE × "interestingness score" (bình phương độ dốc trung bình × tích phân lưu lượng tương đối trong cửa sổ). Hạt giống 42.

**Đánh giá.** `evaluate_nse` lấy `mean = dataset.mean[:, [0]]` (lưu lượng trung bình **đơn vị gốc**) và tính `mse_loss(mean, data.y)` với `data.y` **đã chuẩn hóa**, rồi nhân σ² cho cả tử và mẫu. Mẫu số vì vậy là (μ_gốc − y_chuẩn_hóa)²·σ² thay vì (μ − y_gốc)², lệch với công thức "Testing Metric" của bài (tr. 4, mẫu số dùng μ là lưu lượng trung bình của trạm). Mọi thí nghiệm đặt `normalized: True` nên lỗi này áp dụng cho toàn bộ kết quả; tử số (sai số mô hình) vẫn đúng. Trong `results_mlp.csv`, MLP có NSE trung vị 0,937, 28% trạm có NSE > 0,99, 10% trạm ≥ 0,9998. Cách sửa đúng theo bài: mẫu số Σ w·(0 − y_chuẩn_hóa)²·σ², tương đương Σ w·(μ − y_gốc)²; khi so với bài khác thì báo cáo thêm NSE chuẩn dùng trung bình tập kiểm tra. Mẫu số của mỗi trạm giống nhau cho mọi mô hình nên thứ tự giữa các mô hình trên cùng trạm không đổi; riêng NSE tổng hợp (trung bình qua các trạm) có thể đổi thứ tự vì mỗi trạm bị thổi phồng một mức khác nhau — cần tính lại để xác nhận. Mức thổi phồng tăng theo lưu lượng trung bình của trạm (xấp xỉ 1 + μ²/E[y_chuẩn_hóa²]), nên chênh lệch giữa các mô hình ở trạm sông lớn, hạ lưu bị nén về gần 0 và gần như không đóng góp vào trung bình; trong khi đây là nhóm trạm mà thông tin thượng nguồn (đồ thị) được kỳ vọng giúp nhiều nhất (suy luận từ công thức, chưa kiểm bằng số). Việc chọn trạm kém nhất ở Mục 4.5 của bài cũng dựa trên NSE bị lỗi này.

**Trọng số "relevancy score" khác mô tả của bài.** `interestingness_score` (`functions.py` dòng 157–170) tính `torch.gradient(...)[0].mean()` không chỉ định chiều, nên đạo hàm được lấy trung bình trên **toàn bộ batch** (mọi trạm × mọi mẫu) thành một số duy nhất rồi mới bình phương; bài mô tả trọng số tính riêng cho từng trạm (tr. 4). Hệ quả: trọng số của một trạm phụ thuộc các trạm và mẫu khác cùng batch (batch 64 khi huấn luyện, 1 mẫu × 358 trạm khi đánh giá); đạo hàm có dấu nên đoạn lên và đoạn xuống triệt tiêu nhau. Trọng số này có mặt ở cả hàm mất mát lẫn NSE có trọng số.

**ResGAT bỏ qua trọng số cạnh vật lý.** `ResGAT` truyền trọng số cạnh vào `GATConv` nhưng không khai báo `edge_dim`; trong PyTorch Geometric 2.6.1 (`gat_conv.py` dòng 175–180 và 400), khi `edge_dim` là None thì `lin_edge` là None và `edge_attr` bị bỏ qua. Kiểm lại mã nguồn PyTorch Geometric 2.5.0 cho cùng kết quả (`requirements.txt` ghi 2.6.1, phát hành sau ICML 2024, nên phiên bản dùng khi chạy thí nghiệm chưa xác nhận). Trọng số chỉ còn tác dụng lọc cạnh có giá trị 0 (dòng 104–105 `models.py`). Vì vậy các cấu hình ResGAT "stream length", "elevation difference", "average slope", "all" ở Bảng 2(c) thực chất gần như đồ thị nhị phân; khác biệt giữa các dòng này chủ yếu do ngẫu nhiên khi huấn luyện.

**Nhiễu giữa các lần huấn luyện lớn cỡ toàn bộ chênh lệch trong Bảng 2.** ResGAT "all" (ma trận trọng số 3 cột, không lọc cạnh) và ResGAT "binary" (trọng số 1, không cạnh nào bị lọc) giống hệt nhau về chức năng theo mã: cùng tập cạnh, cùng khởi tạo (hạt giống 42), `edge_attr` bị bỏ qua. Tuy vậy checkpoint của hai cấu hình khác nhau ở toàn bộ 80 tensor, và NSE trung bình theo từng fold trong `results_full.csv` lệch nhau tới 6,67 điểm % (downstream, fold 0: 75,30% so với 81,97%); 9 cặp (3 hướng × 3 fold) lệch từ −1,40 đến +6,67 điểm %. Toàn bộ Bảng 2 chỉ trải trong 80,2%–85,6%. Nguyên nhân khả dĩ là phép cộng scatter trên GPU không tất định (`ensure_reproducibility` không bật `torch.use_deterministic_algorithms`); các cấu hình "isolated" ở ba hướng cho checkpoint trùng khớp từng bit vì không có cạnh nào để cộng dồn. Hệ quả: mỗi cấu hình chỉ chạy 1 lần nên chênh lệch giữa các dòng Bảng 2 không phân biệt được với nhiễu; kết luận đúng của bài là "không phát hiện khác biệt", chưa đủ để khẳng định đồ thị không giúp.

**Bài ghi NSE nằm trong [0, 1].** Mục 3.2 (tr. 4) mô tả NSE thuộc [0, 1]; `results_mlp.csv` có NSE âm ở một số trạm (thấp nhất −1,98) và các giá trị này vẫn được lấy trung bình.

**Nạp checkpoint không kiểm tra khớp tham số.** `load_model_and_dataset` gọi `load_state_dict(..., strict=False)`: nếu tên tham số lệch (ví dụ do khác phiên bản thư viện), mô hình vẫn chạy với trọng số khởi tạo ngẫu nhiên mà không báo lỗi. Khi tính lại NSE cần kiểm tra danh sách tham số thiếu/thừa.

**Lọc trạm chặt hơn mô tả.** Bài (tr. 3) loại trạm có khoảng thiếu dài hoặc thiếu dữ liệu 2000–2017; mã (`_has_valid_data`) loại trạm nếu lưu lượng ≤ 0 tại **bất kỳ giờ nào trong toàn bộ chuỗi** (kể cả trước năm 2000), nên cũng loại trạm từng có lưu lượng bằng 0 hoặc thiếu dữ liệu ngoài giai đoạn nghiên cứu.

**Mốc so sánh của NSE.** Công thức của bài và mã đều so với lưu lượng trung bình giai đoạn 2000–2015 của trạm (không phải trung bình tập kiểm tra như NSE chuẩn), nên con số không so trực tiếp được với NSE ở các bài khác.

**Sai khác nhỏ khác.** Bài mô tả chuẩn hóa trên toàn chuỗi, mã dùng 2000–2015 (mã đúng hơn); bài ghi trọng số cạnh âm bị cắt về 0, mã lấy trị tuyệt đối; phần chữ ghi "MLP 19 lớp" trong khi Bảng 2 và `train_mlp.py` dùng MLP 2 lớp 512 nút; `ResGAT` truyền trọng số cạnh vào `GATConv` không khai báo `edge_dim`.

**Tái lập.** Kèm 957 checkpoint (1,9 GB; 162 cho thí nghiệm chính ở Bảng 2 = 18 cấu hình đồ thị × 3 kiến trúc × 3 cách chia, 3 cho MLP, 108 cho ablation, 684 cho mạng con) để tính lại NSE không cần huấn luyện lại; cần PyTorch Geometric; đường dẫn đặt trong `DATASET_PATH`, `CHECKPOINT_PATH` đầu mỗi script.

**Điểm chèn Mamba.** Thay `self.encoder = Linear(...)` trong `BaseModel` bằng bộ mã hóa thời gian: đưa tensor (số trạm, W, 5) qua Mamba, lấy đầu ra bước cuối làm embedding d = 128 cho GNN; tăng `window_size`; so Mamba có/không có đồ thị và so với MLP.

---

## 2. #67 — MF-LSTM, thư viện Hy2DL

**Nguồn.** Thư viện `github.com/eduardoAcunaEspinoza/Hy2DL` (BSD-3, bản 2.0.1, cập nhật 03/09/2026, ~8.500 dòng, cấu trúc dựa theo NeuralHydrology) và bản mã dùng cho bài trên Zenodo `10.5281/zenodo.14780059` (7,4 GB, gồm mã, kết quả, mô hình; đọc từ xa bằng HTTP Range, chỉ tải các tệp cần).

**Bản mã của bài (Zenodo).** `experiments/Result_reproducibility.pdf` hướng dẫn 3 script (`mflstm.py`, `mflstm_differentinputsperfreq.py`, `mflstm_3freq.py`); tổ hợp bằng cách chạy nhiều hạt giống. `experiments/mflstm.py`: train 1990–2003, val 2003–2008, test 2008–2018, `seq_length = 365×24`, `predict_last_n = 24`, hidden 128, batch 256, 30 epoch, loss `nse_basin_averaged` (NSE*), test bằng epoch cuối; `functions_evaluation.nse` tính NSE chuẩn theo từng lưu vực (mẫu số dùng trung bình quan trắc kỳ test, bỏ NaN ở cả dự báo và quan trắc) rồi lấy trung vị. Nhãn val/test không chuẩn hóa (`standardize_data(standardize_output=False)`, dòng 231 và 375), dự báo khử chuẩn hóa bằng scaler kỳ train, nên NSE tính ở đơn vị gốc; mô hình test là mô hình sau epoch cuối, validation chỉ để theo dõi. Kết quả lưu cho 10 hạt giống (110, 111, 222, …, 999) ở cả hai thí nghiệm chính; thí nghiệm 3 tần suất có 1 hạt giống. `run_progress.txt` (hạt giống 110): ~890 s/epoch, tổng 28.608 s ≈ 7,9 giờ; NSE validation 0,705 (epoch 4) → 0,743 (epoch 16) → 0,729 (epoch 28).

**Thư viện hiện tại.** `datasetzoo/` (CAMELS-US/GB/DE/CH/PL, Caravan, `hourlycamelsus.py` đọc CSV NLDAS giờ + USGS giờ theo từng lưu vực), `modelzoo/` (CudaLSTM, MF2LSTM, HBV/SHM lai; đăng ký qua `factory.py`), `evaluation/` (`simulation_tester.py`, `forecast_tester.py`, `metrics.py` với NSE theo lưu vực và PNSE so với persistence), `training/` (loss NSEBasinAveraged, WeightedMSE, NLL, CRPS). Scaler chỉ tính trên kỳ train; kỳ val/test bắt buộc nạp `scaler.yml`. Hỗ trợ lưu dữ liệu dạng zarr trên đĩa khi RAM không đủ. Ví dụ `examples/lstm_rainfall_runoff.py` test bằng mô hình epoch cuối.

**Chế độ dự báo.** `MF2LSTM` (Multi-Frequency Masked-Forecasting LSTM): hindcast đa tần suất + dự báo tuần tự; teacher-forcing khi huấn luyện, tự hồi quy khi suy luận; đồng hóa Q quan trắc bằng masked-mean. Chưa có cấu hình mẫu dự báo cho CAMELS-US giờ và chưa có bài báo công bố số liệu.

**Chạy thử.** Tạo dữ liệu giả đúng định dạng CAMELS-US giờ (2 lưu vực, 01–08/2000; CSV NLDAS giờ, USGS giờ, tệp `camels_*.txt`); cấu hình `dataset: hourly_camels_us`, `model: mf2lstm`, hindcast 13 bước ngày + 24 bước giờ gồm nhóm khí tượng và nhóm Q quá khứ, dự báo 24 giờ với khí tượng quan trắc làm pseudo-forecast và Q tự hồi quy. Kết quả: huấn luyện 1 epoch chạy hết 79 batch (loss 49,9 → 22,5); đánh giá ra mảng (2 lưu vực × 1.489 giờ × 1 biến × 24 lead time); `calculate_metrics(..., forecast_mode=True)` trả về NSE, RMSE, PNSE theo lưu vực và lead time. Ràng buộc phát hiện: MF2LSTM yêu cầu `custom_seq_processing` và `nan_handling_method: masked_mean` với nhóm biến; bảng thuộc tính phải có cột `huc_02`.

**Điểm chèn Mamba.** Thêm một lớp mô hình vào `modelzoo/` và đăng ký trong `factory.py`, nhận cùng `sample` như CudaLSTM; có thể chạy thẳng chuỗi 8.760 bước giờ.

---

## 3. #22 — S4D-FT

**Nguồn.** `github.com/Pandas-Paws/S4D_rainfall_runoff_simulations`, commit cuối 09/12/2025, không có LICENSE (phần `datautils.py` kế thừa Kratzert, Apache-2.0). Cấu trúc kế thừa `kratzert/lstm_for_pub`: `main_with_train_val.py` (thiết lập global), `main_pub.py` (PUB 12 phần), `train_val.sh`, `train_pub.sh`, `run_global_parallel.py`, `papercode/SSM_test.py` (lớp `HOPE`), `papercode/models/s4/s4d.py` (lớp S4D), `lstm.py`, `mclstm_modifiedhydrology.py`, `datasets.py`, `datautils.py`, `nseloss.py`, `analysis/`.

**Dữ liệu.** Người dùng tự tải CAMELS v1.2 và thay forcing NLDAS bằng bản mở rộng (HydroShare); dữ liệu train đóng gói HDF5, thuộc tính tĩnh vào SQLite; loại các thuộc tính suy từ lưu lượng (`INVALID_ATTR`).

**Chia tập và chuẩn hóa.** Train 10/1999–9/2008, test 10/1989–9/1999; khai báo val 1984–1989 nhưng đoạn tạo tệp validation bị chú thích. Validation thực tế là `random_split` 10% cửa sổ train. Chuẩn hóa bằng hằng số `SCALER` kỳ train; trung bình/độ lệch của Tmax và Tmin trùng nhau (10,864/10,932), thống kê đầu ra chép từ bản Maurer.

**Mô hình.** `HOPE`: Linear(32 → d_model) → n lớp [S4D + Dropout + residual + BatchNorm1d] → trung bình theo thời gian → Linear(d_model → 1). S4D tính tích chập bằng FFT độ dài 2L rồi cắt L (nhân quả); `cfr`, `cfi` tỉ lệ lại phần thực và ảo của A. Nhãn là Q của ngày cuối cửa sổ (`datautils.reshape_data`: `y_new[i] = y[i + seq_length − 1]`).

**Huấn luyện và đánh giá.** Loss NSE*; lưu checkpoint mọi epoch; đánh giá tại epoch 49 cố định; 8 hạt giống (200–207) qua `run_global_parallel.py`. Chỉ số NSE (`analysis/performance_functions.py::nse`) là NSE chuẩn theo từng lưu vực: mẫu số dùng trung bình quan trắc của chính kỳ test, bỏ quan trắc âm (-999); nhãn test để ở đơn vị gốc (chỉ chuẩn hóa khi `is_train`), dự báo được khử chuẩn hóa và cắt giá trị âm về 0.

**Sai khác mã – bài.**
1. `train_val.sh` chỉ truyền lr, weight decay, batch, epoch, d_model, d_state, n_layers, dropout, cfr, cfi; các tham số còn lại lấy mặc định: `lr_min` 1e-3 (bảng 4e-5), `min_dt` 1e-3 (bảng 1e-2), `max_dt` 1 (bảng 1e-1), `lr_dt` 0, `wd` 0. Theo Hình S1/S2 của SI, các giá trị này nằm trong nhóm cấu hình làm KGE giảm 0,05–0,11.
2. `s4d.py` truyền `lr` vào vị trí weight decay khi đăng ký `log_A_real`, `A_imag`; không cài cơ chế `lr_dt` và `wd` mà SI mô tả.
3. `train_pub.sh` dùng 30 epoch, batch 256 (bài: dùng lại siêu tham số global); README kiểm tra PUB ở epoch 49.
4. LSTM baseline: `--batch_size` mặc định 64 đè giá trị 256; `--epochs` mặc định 50 (Bảng S2: 30). LSTM tự cài bằng vòng lặp Python (không cuDNN).
5. PUB: chuẩn hóa thuộc tính tĩnh lúc đánh giá dùng thống kê của lưu vực test.
6. Tiện ích hỏng: `run_global_parallel.py` gọi `main.py` không tồn tại; README gọi `train_val_global.sh`; MC-LSTM không chạy được từ script chính; `analysis/main_performance_ensemble_only.py` dùng `DataFrame.append` (bị bỏ từ pandas 2.0); không có bước gộp thành Bảng 3.

**Chạy thử.** Dựng `HOPE` đúng Bảng S3 (d_model 128, d_state 128, 6 lớp, cfr = cfi = 10) và chạy forward/backward trên CPU (PyTorch bản CPU, không cần kernel CUDA) với đầu vào ngẫu nhiên 16 × 365 × 32: đầu ra (16, 1), 402.177 tham số. Repo không khai báo phụ thuộc `torchvision`, `einops`.

**Chuyển sang dự báo.** Thêm cột `QObs(mm/d)` vào đầu vào trong `datasets.py::_load_data` và dời nhãn sang ngày t+h trong `reshape_data`; cần xử lý giá trị thiếu của Q quá khứ.

**Điểm chèn Mamba.** Thay `LTI(...)` trong `HOPE.__init__` bằng khối Mamba; Mamba nhận (B, L, D) còn S4D nhận (B, D, L) nên cần đổi trục; thử đọc đầu ra bằng trung bình theo thời gian và bằng bước cuối; khôi phục validation 1984–1989.

---

## 4. #103 — FloodGNNs

**Nguồn.** `github.com/Dreamzz5/FloodGNNs`, MIT, đẩy lần cuối 05/10/2024. Khung BasicTS rút gọn (`basicts/`), mô hình `baselines/GNNs/arch/GNNs.py::FloodGNN`, cấu hình `baselines/GNNs/config.py`, `experiments/train.py`. Không có mã tiền xử lý LamaH-CE và mã EA-LSTM.

**Dữ liệu.** `data.dat` (memmap), `desc.json`, `adj_mx.pkl` (dense), `topology_adj.pkl` tải từ Google Drive (thư mục `Flood`). Chia train/val/test theo tỉ lệ thời gian trong `desc.json`.

**Mô hình.** Conv2d 1×1 nhúng 3 kênh mỗi bước thời gian → 3 lớp GNN áp riêng từng bước (nối tắt x0) → làm phẳng (thời gian × 32 kênh) → Conv 1×1 ra 24 bước. Không có thành phần học chuỗi thời gian.

**Sai khác mã – bài.** `config.py` import `GCN_Point` nhưng `arch/__init__.py` chỉ export `FloodGNN` (lỗi import); loss `masked_mse` (bài: MAE); `masked_nse` cộng tử và mẫu trên toàn bộ trạm và bước, `if_score=False` (bài: NSE có trọng số); mốc trung bình trong `masked_nse` là `scaler.mean` của kỳ train (`base_tsf_runner.py` dòng 469, 521), cùng đơn vị với nhãn chỉ khi `RESCALE = True` — giá trị này đọc từ `desc.json` trên Google Drive, chưa xác nhận; `input_dim = 3` trong khi `FORWARD_FEATURES = [0..6]`; `train.py` ghi đè chính tệp cấu hình để đổi `conv_type`.

---

## 5. #54 và #71 — FHNN

**Nguồn.** `github.com/arvindrenga96/FHNN` (đọc trên GitHub): `config.py`, `MODEL.py`/`MODEL.ipynb`, `LSTM.py`, `UTILS.py`, `MODELS/` (notebook + script SLURM cho FHNN, AR-LSTM, RRFormer, TFT; `ctlstm.py`). README ghi mã là của bài ICDM 2025.

**Dữ liệu và cấu hình.** Thiếu `DATA/preprocessData.ipynb` và bước dựng mảng (lưu vực × thời gian × đặc trưng); `RAW_DIR` gõ cứng. `config.py`: 27 thuộc tính tĩnh + 5 biến khí tượng + lưu lượng; train 1985–1993, val 1993–1995, test 1995–2005 (giao thức ICDM để ở dạng chú thích); window = 365, `forecast = 1`; forward_code_dim 255, latent 85, dropout 0,4, batch 128, 50 epoch, lr 1e-3, `inits = 5`.

**Mô hình.** `lstm_hierarchical_enc_with_label_dec`: BiLSTM ngày → lấy mẫu thưa bước 14 → BiLSTM → lấy mẫu thưa bước 6 (84 ngày) → BiLSTM; ghép h, c ba tầng (3 × 85) khởi tạo LSTM giải mã chạy trên khí tượng tương lai. Thang 14/84 ngày khác 7/30 ngày trong bài WRR.

**Huấn luyện và đánh giá.** Bộ mã hóa dùng `x[:, :-forecast]` (không rò rỉ nhãn); MSE có mặt nạ giá trị thiếu; lưu checkpoint theo loss validation nhỏ nhất; khử chuẩn hóa bằng thống kê train. `UTILS.per_node_R2` tính NSE theo lưu vực. `UTILS.stride_array` cắt mỗi năm thủy văn đúng 1 cửa sổ 365 ngày từ 1/10. Notebook FHNN gán cứng `batch_size = 256`.

**Điểm chèn Mamba.** Thay 3 BiLSTM của bộ mã hóa bằng Mamba hai chiều (quét trong cửa sổ lịch sử), ánh xạ đầu ra bước cuối sang trạng thái khởi tạo bộ giải mã qua lớp tuyến tính.

---

## 6. #53 — HydroTFT

**Nguồn.** `github.com/qxc101/HydroTFT`, MIT; README ghi là bản cho phản biện, phải chép đè lên repo `kratzert/ealstm_regional_modeling`. Gồm `main.py`, `papercode/tft.py`, `datasets.py`, `datautils.py`, `evalutils.py`, `nseloss.py` (trọng số theo tầm dự báo `horizon_alpha`).

**Chia tập và chọn mô hình.** Train 1999–2008; `val_start/val_end` = 1989–1999 trùng kỳ test. Khi đánh giá, `--eval_last_n` (mặc định 10) chạy 10 checkpoint cuối trên kỳ 1989–1999, chọn epoch có NSE trung bình cao nhất và báo cáo kết quả của epoch đó trên cùng kỳ (`main.py` dòng ~713–750).

**Sai khác mã – bài.** Bài không nêu bước chọn epoch và khẳng định không rò rỉ; `pred_days = 0` (nowcast) gọi `VanillaTFT` — kiến trúc giản lược khác Bảng 1; ε của loss mặc định 0,1 (bài 1e-6); chuẩn hóa bằng `SCALER` toàn cục và hằng số xấp xỉ cho đặc trưng kỹ thuật (bài: theo từng lưu vực); tầm 7 ngày README chạy `--seq_length 270` (bài: 365); batch mặc định 512 (bài: 1024 cho tầm 1 ngày). Đặc trưng kỹ thuật tính bằng cửa sổ trượt quá khứ (không rò rỉ).

---

## 7. #42 — TFRN

**Nguồn.** `github.com/redtea-code/TFRN`, MIT. `configs/`, `data/` (`dataset.py` CAMELS-US, `dataset_aus.py` CAMELS-AUS; danh sách 448/671 và 561 lưu vực), `models/TFRN.py`, `models/block/` (DIT, FCB, MIXer, SAT), `pretrain.py`, `pretrain_test_global.py`, `utils/`. Cấu trúc giống RR-Former.

**Dữ liệu và chia tập.** Cấu hình mặc định là CAMELS-AUS (`static_size=52` gõ cứng trong `TFRN_base`). Train 1980–1995, val 1995–2000, test 2000–2014; chuẩn hóa val/test bằng `train_means.csv`/`train_stds.csv`. Bỏ dòng thiếu (`dropna`) trước khi cắt cửa sổ.

**Mô hình và huấn luyện.** Nạp chuỗi 365 ngày nhưng chỉ dùng 15 ngày lưu lượng gần nhất + 22 ngày khí tượng (gồm 7 ngày tương lai quan trắc) + đoạn 22 ngày ở vị trí `month × 30` (mặc định `month = 9`); `hidden_size = 16`. Loss `QuantileLoss(0.5)`, 100 epoch, batch 512, lr 1e-3; chọn checkpoint MSE validation nhỏ nhất.

**Lỗi và sai khác.** `for block in self.DIT: out, attn = block(seq, seq_p)` — 3 khối DiT nhận cùng đầu vào, kết quả bị ghi đè, chỉ khối cuối có tác dụng. `calc_nse` tính NSE trên toàn bộ mẫu test của mọi lưu vực gộp chung. Tập test tạo bằng `CamelsDatasetWithStatic_decompose` trong khi train dùng `CamelsDatasetWithStatic`. Các tùy chọn phân rã (MVMD, MEMD, SSA, STL) nếu bật đều phân rã trên toàn chuỗi (mặc định tắt).

---

## 8. #6 — From RNNs to Transformers

**Nguồn.** Zenodo `10.5281/zenodo.17863071` (`HESS_paper.zip`), CC BY-NC 4.0. Gói `rainflow/` (`config/`, `models/` với 15 tệp mô hình, `training/`, `utils/`); `examples/` có script cho 4 bài toán; kèm `CAMELS.nc` (1.098 MB: 531 lưu vực, 15 biến forcing, 26 thuộc tính tĩnh).

**Chia tập và huấn luyện.** `cal_statistics` chỉ ở chế độ train; script `LSTM.sh` dự báo: train 1999–2008, val 2008–2013 (bài ghi val 1980–1989), test 1989–1999. Huấn luyện đủ 30 epoch, dùng mô hình epoch cuối; không script nào bật `--do_eval`. "Dự báo" dùng `--DI` (khí tượng cùng ngày + lưu lượng trễ).

**Vấn đề.** Trong 52 script, chỉ `LSTM.sh` bật `--use_target_stds` (loss kiểu NSE), lr 5e-4, cắt gradient 1; 12 mô hình còn lại dùng MSE thuần, lr 1e-4, không cắt gradient. `load_model_resume` sắp xếp tên checkpoint theo chuỗi ký tự (lấy epoch 9 thay vì 30 khi chạy riêng `--do_test`). `LSTM_mask` gọi `torch._cudnn_rnn` và `self.cuda()` trong `__init__` (phụ thuộc phiên bản PyTorch/CUDA). FHV/FLV bị chú thích trong hàm `test`.

**Điểm chèn Mamba.** Thêm `rainflow/models/Mamba.py` theo giao diện Time-Series-Library, tạo script giống `LSTM.sh`; chạy mọi mô hình với cùng hàm mất mát.

---

## 9. #36 — S4D/S5D trong dPL

**Nguồn.** `github.com/chooron/dmg-research/tree/master/project/bettermodel` (mã chính thức theo mục "Data and software availability" của bài), đẩy lần cuối 14/9/2026, không có LICENSE. Phụ thuộc gói `dmg>=1.4.0`, `pytorch-tcn`, `pytorch-tsmixer`, Hydra, `uv`; PyTorch 2.9.1 + CUDA 12.8.

**Cấu hình.** `conf/config_dhbv_hopev1_531.yaml` (S4D) và `config_dhbv_lstm_531.yaml` chỉ khác phần mô hình: train 1980/10/01–1995/09/30, test 1995/10/01–2010/09/30, `rho` 365 + warm-up 365, HBV 16 thành phần, tham số động `parBETA, parK0, parBETAET`, 35 thuộc tính, NseBatchLoss, Adadelta lr 1,0, batch 100, 100 epoch, `test_epoch: 100`.

**Mô hình.** `layers/hope.py` là lớp S4D/HOPE của #22; `hope_mlp_v1.py` dùng 4 lớp, sigmoid đầu ra. `my_trainer.init_optimizer` tạo một nhóm tham số chung nên lỗi truyền `lr` vào weight decay kế thừa từ #22 không ảnh hưởng kết quả.

**Đa hạt giống.** `ablation/report/multiseed_7model_comparison_report.md`: 5 hạt giống, 671 lưu vực, Friedman + Wilcoxon + Holm; NSE trung vị S5Dv2 0,7514, S5Dv1 0,7512, S4D 0,7496, TCN 0,7423, Transformer 0,7418, TSMixer 0,7418, LSTM 0,7416.

**Điểm chèn Mamba.** Thêm `mamba_mlp.py` theo mẫu `hope_mlp_v1.py` (Mamba nhân quả) và một cấu hình sao từ cấu hình S4D.

---

## 10. #135 — TSFM zero-shot

**Nguồn.** `github.com/dialuser/tsfm_study`, đẩy 28/02/2026, không có LICENSE. LSTM ngày (`lstm_camelsdaily.py`), LSTM 3 giờ (`lstm_main.py`), TSFM (`tsfm/1d`, `tsfm/3H`), môi trường conda riêng cho từng TSFM.

**Chia tập và huấn luyện.** Train 1980–1992, val 1992–1995, test 1995–2005; thống kê chuẩn hóa tính ở chế độ train. LSTM hidden 128, chuỗi 365, đơn biến (`add_forcing: False`), 50 epoch tối đa, AdamW + cosine, MSE, EMA; lưu `bestmodel{seed}.pth` khi loss validation giảm, dừng sớm sau 5 epoch. Biến đổi `log10(√Q + 0,1)` có hoạt động (chú thích trong cấu hình đã lỗi thời).

**Vấn đề.** Repo chỉ có thí nghiệm đơn biến, không có mã TTM đa biến, tinh chỉnh TTM, tinh chỉnh Chronos (Bảng 4). TSFM dùng log1p và lọc lưu vực thiếu > 10%, LSTM không lọc.

---

## 11. #38 — Hierarchical CNN-LSTM

**Nguồn.** `github.com/Dehui-nb/CNN-LSTM-for-run-off-simulation`, không có LICENSE. Dùng các script trong `Codes_for_Models/` (bản cũ ở thư mục gốc dừng sớm theo NSE trên kỳ đánh giá).

**Chi tiết.** Train 1999–2008, đánh giá 1989–1999; `SCALER` NLDAS giống #22 (lỗi Tmax = Tmin); LSTM 1 lớp 256, dropout 0,4, forget bias 5; 30 epoch cố định, batch 64, lr 1e-3 → 5e-4 → 1e-4, hạt giống 200; 16 thuộc tính tĩnh (khớp bài). PUB: thuộc tính tĩnh chuẩn hóa chỉ bằng lưu vực train. Loss `MultiStepNSELoss` ε = 1e-6. Bỏ dòng thiếu trước khi cắt cửa sổ. Script raster: `MID_RESIZE_MODE = "keep_aspect_pad"`, giới hạn 20 mảnh, `RUN_ALL_SEEDS = True` (lệch bài).

---

## 12. MTPre (chưa xuất bản)

**Nguồn.** Zenodo `10.5281/zenodo.19367140` (01/04/2026), MIT; repo GitHub trả về 404. Gồm dữ liệu (`camels_rrformer_dataset_v4_static_scaled_raid.py`, `camels_dataset_emd.py`), mô hình (`rrformer_variants.py` với `MambaEncRRFormer`, `lstm_backbone.py`), huấn luyện (`train_full.py`), script `run_benchmark_suite.py` (không EMD), `run_emd_suite.py` (có EMD), `run_transfer_gb*.py`.

**Chi tiết.** CAMELS-US 674 lưu vực, forcing Daymet; train 1980–1995, val 1995–1999, test 1999–2014; cửa sổ 22 ngày (15 quá khứ + 7 dự báo, khí tượng tương lai quan trắc); đầu vào có Q quan trắc 15 ngày quá khứ, 7 ngày tương lai điền bằng Q ngày cuối (`__getitem__`, dòng 521–529) — bài toán dự báo; chọn checkpoint theo "NSE-like" trên validation, patience 15. README ghi mọi mô hình dùng chung pipeline EMD + nhúng CNN 1D; EMD tính trên toàn chuỗi trước khi cắt cửa sổ nên rò rỉ tương lai. Loại mọi ngày có cờ chất lượng khác "A" rồi `dropna`.

---

## 13. NeuralHydrology — lớp Mamba có sẵn

`neuralhydrology/modelzoo/mamba.py` (PR #163, 02/2024; sửa lần cuối 03/2025): 1 khối Mamba (`d_state` 16, `d_conv` 4, `expand` 2) sau `InputLayer`. `InputLayer.forward` trả [seq_length, batch_size, n_features] nhưng `mamba_ssm.Mamba` nhận (B, L, D), nên Mamba quét theo trục batch; `transpose(0, 1)` chỉ đặt ở đầu ra. PR ghi mô hình đạt khoảng NSE 0,4. Cách sửa: chuyển trục `x.transpose(0, 1)` trước khi gọi `self.mamba`.

---

## 14. OpenHydroNet — mã mô hình dự báo của Google FloodHub

**Nguồn.** `github.com/google-research/flood-forecasting` (Apache 2.0), bản trong `Code/OpenHydroNet` (clone 26/9/2026, commit 21/9/2026, 236 MB gồm 80 MB trọng số, ~19.300 dòng Python). Fork NeuralHydrology, gói `googlehydrology`: `datasetzoo/` (`caravan.py`, `multimet.py`), `modelzoo/` (`handoff_forecast_lstm.py`, `mean_embedding_forecast_lstm.py`, mỗi tệp ~560 dòng; `head.py` gồm đầu ra CMAL), `training/`, `evaluation/`, thư mục `test/`, notebook `tutorial/OpenHydroNet_Tutorial.ipynb` kèm mẫu Caravan nhỏ.

**Mô hình.** `MeanEmbeddingForecastLSTM`: mỗi nguồn khí tượng qua một mạng nhúng riêng, gộp bằng trung bình có mặt nạ (bỏ nguồn bị thiếu), rồi vào LSTM hindcast (365 bước) và LSTM forecast (lead time 7); trạng thái LSTM hindcast truyền sang LSTM forecast; thuộc tính tĩnh qua mạng nhúng riêng. Không có Mamba hay Transformer.

**Dữ liệu và cấu hình.** Chỉ hỗ trợ tần suất ngày (`multimet.py` dòng 196). Không có cơ chế đưa lưu lượng quan trắc vào đầu vào. Cấu hình vận hành (`floodhub-settings-config.yml`, `handoff-forecast-lstm-config.yml`) đặt train, validation, test cùng 01/01/1982–31/12/2023. Cấu hình CAMELS-US đặt validation trùng test (2021–2025) và đọc thư mục `camels-updated-2025` chưa rõ nguồn. Bản zarr công khai `gs://caravan-multimet/v1.1` đọc được theo từng lưu vực (khối 128 lưu vực): HRES 22.492 lưu vực × 3.196 ngày (01/01/2012–30/09/2020) × 10 lead time; GraphCast 2.911 ngày (01/01/2015–20/12/2022); ERA5-Land 27.333 ngày (tới 31/10/2024).

**Trọng số huấn luyện sẵn.** 2 bộ (`google-floodhub-settings-55-epochs`, bản lọc NSE > 0,5 — 85 epoch), huấn luyện trên 1982–2023; `Pretrained-Models-README.md` cấm dùng để đánh giá theo thời gian trong 1982–2023 vì rò rỉ dữ liệu, chỉ dùng cho tinh chỉnh, đánh giá lưu vực không tham gia huấn luyện, hoặc dự báo sau 2023.

**Pipeline (đọc `datasetzoo/multimet.py`, `modelzoo/mean_embedding_forecast_lstm.py`, `training/loss.py`, `datautils/scaler.py`, `evaluation/metrics.py`).**
- **Mẫu tại ngày phát hành t:** hindcast lấy các biến không có lead time trong [t−364, t]; sản phẩm dự báo dùng làm hindcast chỉ lấy lead 0 của các lần phát hành [t−365, t−1] (lùi một ngày để không dùng thông tin chưa có tại t); forecast lấy dự báo phát hành tại t với lead 1–7, cộng phần chồng lấn `forecast_overlap` (lead 0 của các ngày trước, 365 ngày ở cấu hình vận hành). Nhãn là chuỗi Q kết thúc tại t + 7; hàm mất mát chỉ tính trên 8 bước cuối (`predict_last_n: 8`, lead 0–7).
- **Mô hình:** mỗi nguồn khí tượng qua mạng nhúng riêng (có nối thuộc tính tĩnh đã nhúng), gộp bằng trung bình có mặt nạ để bỏ nguồn bị thiếu; LSTM hindcast chạy trước, đầu ra của nó đưa vào LSTM forecast; đầu CMAL (3 thành phần phân phối, lấy 7.500 mẫu, báo cáo trung bình mẫu, cắt giá trị âm về 0). Có biến đếm bước thời gian (`timestep_counter`).
- **Huấn luyện:** Adam, ReduceLROnPlateau, cắt gradient chuẩn 1, nhiễu nhãn 0,005, dropout đầu ra 0,4, forget bias khởi tạo 3; hàm mất mát có sẵn: MSE, RMSE, NSE* chia theo độ lệch chuẩn từng lưu vực, CMAL.
- **Chuẩn hóa và dữ liệu thiếu:** scaler tính trên kỳ train, lưu `scaler.nc`, dùng lại khi suy luận/tinh chỉnh; `validate_samples.py` đánh dấu mẫu hợp lệ (bỏ mẫu thiếu nhãn thay vì bỏ cả lưu vực); `union_features.py` lấp nguồn thiếu bằng nguồn thay thế.
- **Đánh giá:** `metrics.py` có NSE, MSE, RMSE, KGE, Alpha/Beta-NSE, Beta-KGE, Pearson r, FHV, FLV, FMS, Peak-Timing, Missed-Peaks, sai số % đỉnh; tính theo từng lead time.

**Điểm chèn Mamba.** Thay `self.hindcast_lstm` (và có thể `self.forecast_lstm`) trong `mean_embedding_forecast_lstm.py` bằng khối Mamba, giữ cơ chế truyền trạng thái bằng lớp chiếu; đăng ký mô hình mới trong `modelzoo/__init__.py`.

---

## 15. #121 BiasCast

**Nguồn.** `github.com/conestone/biascast` (bản trong `Code/BiasCast`, commit 03/08/2026, ~1.550 dòng Python + notebook): chỉ gồm notebook chạy thí nghiệm (`Experiments/_run/Run_Experiment.ipynb`, `Transfer_Weights.ipynb`) và mã phân tích, vẽ hình (`Inspect_Experiments/`). Huấn luyện dùng bản fork NeuralHydrology `github.com/conestone/neuralhydrology` (đọc ở đoạn dưới). Thư mục `Experiments/` và `Data/` để trống, phải thay bằng Zenodo 17292895 (cấu hình `config.yml`, `best_model.pt`, scaler, `test_metrics.csv` NSE/KGE theo lưu vực cho từng thí nghiệm) và Zenodo 17119635 (Extended LamaH-CE, 0,95 GB).

**Fork NeuralHydrology** (`Code/BiasCast_NH`, `github.com/conestone/neuralhydrology`): tách từ NeuralHydrology upstream tại commit `f00cf47`, thêm đúng 1 commit `9d94908` (16/09/2025) sửa 7 tệp: `datasetzoo/lamah.py` (cho phép nhãn `qmean`, `qmin`, `qmax`; chỉ nạp 1 cột lưu lượng — cột nhãn đầu tiên khớp), `evaluation/tester.py` (thêm đánh giá bằng `best_model.pt`), `training/early_stopping.py` (mới), `training/__init__.py` (bộ lập lịch learning rate), `training/basetrainer.py` (dừng sớm theo NSE validation, lưu mô hình tốt nhất), `training/logger.py`, `utils/config.py`. Mô hình (`sequential_forecast_lstm.py`, `handoff_forecast_lstm.py`, `cudalstm.py`) và hàm tính chỉ số là của upstream, không sửa.

**Cắt cửa sổ (kiểm chống rò rỉ).** `basedataset.py` dòng 158–164: với ngày nhãn t, hindcast lấy [t − 364, t − 1], forecast lấy ngày t; nên Q quan trắc (`qmean`) và khí tượng tái phân tích chỉ tới hôm trước, dự báo ECMWF cho ngày t, nhãn `qmax` của ngày t — không rò rỉ nhãn.

**Cấu hình thí nghiệm (Zenodo 17292895, 24 cấu hình, đọc bằng cách giải nén theo luồng, không lưu tệp nén).** 451 lưu vực LamaH-CE mức A; train 01/2003–12/2009, validation 2010–2013, test 2014–2017; `seq_length` 365, `forecast_seq_length` 1; LSTM 128, dropout đầu ra 0,3, Adam lr 1e-3, CosineAnnealing 30 epoch, cắt gradient 1, hàm mất mát NSE (chia theo độ lệch chuẩn từng lưu vực), dừng sớm theo NSE validation (patience 5, min_delta 0,005); mọi cấu hình cùng hạt giống 111 (1 lần chạy). Mô hình tốt nhất (Sequential Forecast LSTM có Q): hindcast 32 biến (ERA5-Land 21, E-OBS 7, MSWEP 1, GLEAM 2, `qmean`), forecast 5 biến ECMWF (t2m, d2m, ssrd, tp, e), 33 thuộc tính tĩnh, nhãn `qmax`.

**Tự tính lại từ `test_metrics.csv` của tác giả (trung vị NSE / KGE, 451 lưu vực).** CrossDomain tái phân tích 0,581 → chạy với dự báo 0,327; Sequential Forecast LSTM có Q 0,705 / 0,774 (Complex) và 0,705 / 0,767 (Simple); không Q 0,609–0,628; Encoder–Decoder LSTM có Q 0,662–0,672, không Q 0,568–0,591; học chuyển giao 0,388–0,440; baseline CUDA LSTM chỉ dự báo 0,387, chỉ tái phân tích 0,688–0,692, tái phân tích + dự báo 0,679–0,701. Khớp các số trong tóm tắt bài (0,58 → 0,33; 0,63; 0,71).

**Vấn đề phát hiện.**
- Chỉ 1 hạt giống cho mọi cấu hình; chênh lệch nhỏ (khoảng 0,01–0,02 NSE) giữa các biến thể không phân biệt được với nhiễu.
- Baseline CUDA LSTM "tái phân tích" và "tái phân tích + dự báo" nhận khí tượng tái phân tích của chính ngày nhãn (thời tiết hoàn hảo), nên là mốc tham chiếu trên, không phải cấu hình vận hành được.
- Đầu vào hindcast dùng tái phân tích tới hôm trước; thực tế ERA5-Land, E-OBS, GLEAM công bố trễ nhiều ngày — cấu hình không lùi đầu vào theo độ trễ công bố (tác giả tự nêu hạn chế này ở Mục 3.7).
- Thí nghiệm có Q đọc thư mục `LamaH_expanded_q_input` — không có trong bản Zenodo; phải tự chép cột `qmean` từ `D_gauges` vào tệp khí tượng mức A (do bộ nạp chỉ nạp 1 cột lưu lượng).
- Đơn vị lưu lượng: bộ nạp `datasetzoo/lamah.py` chỉ nạp 1 cột lưu lượng từ `D_gauges` (cột nhãn đầu tiên khớp, ở đây `qmax`) và luôn chia theo diện tích `area_gov` sang mm/ngày; tệp `D_gauges` của Extended LamaH-CE có cột `qmin;qmean;qmax` cùng đơn vị m³/s (kiểm tệp `ID_205.csv`). `qmean` đầu vào đọc từ thư mục khí tượng riêng, không đổi đơn vị; scaler của tác giả cho `qmean` trung bình 4,58 / độ lệch 16,15 so với `qmax` 2,57 / 5,26 — nhiều khả năng `qmean` ở m³/s còn nhãn ở mm/ngày (chưa kiểm được vì thư mục không công khai). NSE theo lưu vực không bị ảnh hưởng bởi đơn vị nhãn.
- Lọc mẫu huấn luyện (`basedataset.py`, `_validate_samples`): mẫu bị loại nếu bất kỳ cột nào trong 37 biến động có giá trị thiếu ở bất kỳ ngày nào của cửa sổ 365 ngày — kể cả cột ECMWF ở các ngày hindcast mà mô hình không dùng, và `qmean` ở biến thể có Q; validation và test giữ mọi mẫu. Nếu ECMWF bắt đầu từ 2003 thì phần lớn năm 2003 mất mẫu huấn luyện (năm bắt đầu chưa kiểm).
- Dừng sớm và chọn mô hình theo **trung vị** NSE validation qua các lưu vực (`logger.py` trả trung vị cho mọi chỉ số validation).
- Hàm mất mát NSE* dùng độ lệch chuẩn `qmax` từng lưu vực tính trên dữ liệu chưa chuẩn hóa (mm/ngày) làm trọng số cho sai số trên dữ liệu đã chuẩn hóa toàn cục — như NeuralHydrology gốc.
- Mạng handoff của Encoder–Decoder LSTM (`handoff_forecast_lstm.py`, `state_handoff_network: hiddens 128`): `FC` một lớp không có kích hoạt, tiếp theo `handoff_linear` tuyến tính → ánh xạ tuyến tính 256 → 128 → 256; bài mô tả là mạng phi tuyến.
- 24 cấu hình: mọi cấu hình Sequential Forecast LSTM và Encoder–Decoder LSTM cùng siêu tham số (hidden 128, batch 256, dropout 0,3, không nhiễu nhãn); các baseline có bộ khác nhau (hidden 128–256, dropout 0,2–0,4, nhiễu nhãn 0,001–0,1) — chưa rõ tối ưu Bayes chạy riêng cho từng kiến trúc hay không.
- `_is_best_model` so với `best_value` không tính `min_delta` trong khi `EarlyStopping` chỉ cập nhật `best_value` khi cải thiện > 0,005: `best_model.pt` có thể bị ghi đè bởi epoch kém hơn epoch đã lưu trước đó tối đa 0,005 NSE validation.
- Bài ghi dự báo ECMWF HRES lấy trung bình 8 giá trị 3 giờ từ lần phát hành 00 UTC của ngày t (Mục 2.1); mã tạo Extended LamaH-CE không công khai nên chưa kiểm được từ mã, căn chỉnh giữa ngày UTC và ngày của LamaH-CE chưa xác nhận.
- Nhúng "đơn giản" (`hiddens: [16]`) trong mã là một lớp `Linear` thuần: lớp `FC` của NeuralHydrology không áp hàm kích hoạt cho lớp cuối; Mục 2.2.6 của bài mô tả có tanh, Mục 3.5 gọi là nhúng tuyến tính. Số tham số tính từ mã: Sequential Forecast LSTM nhúng đơn giản 84.785, nhúng phức tạp (30-20-64) 143.291.

**Tái lập.** Theo README: cài fork NeuralHydrology, sửa đường dẫn trong `config.yml`, chạy notebook để huấn luyện hoặc nạp `best_model.pt` qua `checkpoint_path`. Yêu cầu Python 3.12.3 và GPU CUDA. Chưa chạy thử.

**Điểm chèn Mamba.** Thêm mô hình vào `modelzoo` của fork NeuralHydrology, thay LSTM trong Sequential Forecast LSTM; lưu ý lớp Mamba có sẵn của NeuralHydrology bị sai trục (Mục 13).

---

## 16. Kết luận từ việc đọc mã nguồn

1. **#104:** mã nhỏ, dễ chèn bộ mã hóa thời gian; lỗi NSE và validation sửa được; có checkpoint để tính lại.
2. **#121 BiasCast:** mã dựa trên NeuralHydrology, công khai đủ cấu hình, trọng số và kết quả test theo lưu vực; dữ liệu nhẹ.
3. **OpenHydroNet:** mã chất lượng nhất (Google duy trì, có kiểm thử, tài liệu); không có mốc công bố tái lập được trên dữ liệu công khai, trọng số sẵn không dùng làm mốc theo thời gian.
4. **#67:** mã của bài sạch nhất (NSE chuẩn, đơn vị gốc, scaler kỳ train, không chọn mô hình trên test); thư viện có sẵn chế độ dự báo đã chạy thử được.
5. **#22:** khung nhỏ, thay một dòng khởi tạo lớp SSM; NSE chuẩn; phải sửa cấu hình về Bảng S3.
6. **#6:** tái lập nhanh nhất để làm baseline, cần đồng nhất hàm mất mát.
7. **#53, #42:** chọn mô hình hoặc cấu hình dựa trên tập test.
8. **#71/#54, #103:** thiếu tiền xử lý hoặc có lỗi khiến không tái lập được số liệu.
9. Chưa chạy repo nào trên dữ liệu thật; Mamba cần GPU CUDA.
