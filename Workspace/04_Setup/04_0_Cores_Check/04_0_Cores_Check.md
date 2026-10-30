# Bước 0 — Mô hình chung nhận 5 lõi và kiểm trên GPU

Tài liệu đi cùng [mã Python](04_0_Cores_Check.py) của notebook Kaggle `04_0_Cores_Check`. Bước C so 5 lõi thời gian (LSTM, GRU, Transformer, S4D, Mamba) trong cùng một mô hình Sequential Forecast, cùng số tham số (`02_Pipeline.md` Mục 4.1, 4.2, 5). Phần mã thư viện của việc này nằm ở fork `Ticasslo/neuralhydrology`, commit `fed53dc` (nhánh `core-models`, nay cũng là đầu nhánh `research`); kiểm thử CPU chạy ở máy. Notebook kiểm phần chỉ chạy được trên GPU và dữ liệu thật.

## Mục lục

- [1. Câu hỏi cần trả lời](#1)
- [2. Mô hình chung: giữ gì, thêm gì](#2)
- [3. Cài đặt notebook](#3)
- [4. Các Phần và kết quả](#4)
- [5. Kết luận cho bước C](#5)
- [6. Nguồn](#6)

<a id="1"></a>
## 1. Câu hỏi cần trả lời

1. Lõi LSTM của mô hình chung có cho đúng kết quả của `SequentialForecastLSTM` (mô hình BiasCast) không, cả khi dự báo bằng trọng số tác giả lẫn khi huấn luyện?
2. Lõi Mamba có nhìn tương lai không, có quét đúng trục thời gian không, và trọng số có chạy được trên CPU bằng `mambapy` cho demo không?
3. Với mỗi lõi và số lớp, `hidden_size` (và kích thước trong) bao nhiêu thì số tham số lệch mô hình LSTM tác giả không quá 1%?
4. Trên T4, mỗi lõi tốn bao nhiêu thời gian một bước huấn luyện, một lô dự báo, bao nhiêu bộ nhớ GPU, và hai lần chạy cùng hạt giống có trùng từng bit không?

<a id="2"></a>
## 2. Mô hình chung: giữ gì, thêm gì

**Khối giữ của BiasCast.** Mạng nhúng quá khứ và mạng nhúng ngày t (`InputLayer`, mỗi mạng ra 16 chiều), dropout đầu ra, đầu dự báo (`head`), loss NSE*, optimizer Adam, lịch cosine, bộ nạp `data_on_gpu`. Hai chuỗi nhúng ghép theo thời gian (364 ngày quá khứ + ngày t = 365 bước) như `SequentialForecastLSTM`.

**Khối thêm.** Lõi thời gian chọn bằng khóa config `sf_core` (`modelzoo/sequential_forecast.py`, `modelzoo/sf_cores.py`). Mỗi lõi nhận chuỗi `[lô, 365, 32]` và trả chuỗi cùng độ dài; đầu ra bước k chỉ phụ thuộc đầu vào bước ≤ k (nhân quả). Đầu dự báo đọc bước cuối (ngày t).

| Lõi | Cách dựng | Nguồn mã |
|---|---|---|
| LSTM, GRU | `torch.nn.LSTM`/`GRU`, `sf_layers` lớp; chạy phần quá khứ rồi phần ngày t nối trạng thái, như `SequentialForecastLSTM` | PyTorch |
| Transformer | Chiếu tuyến tính → mã hóa vị trí hình sin của thư viện (cộng) → `sf_layers` lớp encoder chuẩn hóa trước (pre-LN) có mặt nạ nhân quả → LayerNorm cuối; tầng truyền thẳng `sf_transformer_dim_feedforward` (mặc định 4 × `hidden_size`) | PyTorch `nn.TransformerEncoderLayer` |
| S4D | Chiếu tuyến tính → `sf_layers` lần `x + Dropout(khối(LayerNorm(x)))` → LayerNorm cuối; khối S4D tính tích chập nhân SSM bằng FFT dài 2L, cắt L bước đầu (nhân quả); tham số SSM có tốc độ học riêng `sf_s4d_lr`, không weight decay | Chép từ `models/s4/s4d.py` của `state-spaces/s4` (Apache-2.0), chỉ đổi định dạng trục và thay `einops` |
| Mamba | Cùng khung dư như S4D, khối là `mamba_ssm.Mamba` 2.3.2.post1; kích thước trong `sf_mamba_d_inner` | `mamba_ssm` |

Các lõi đều là bản gốc (vanilla) của từng kiến trúc; khung dư chuẩn hóa trước cho S4D, Mamba theo cách các bài gốc xếp khối (Gu và cs. 2022; Gu & Dao 2023). Bản `modelzoo/mamba.py` có sẵn trong thư viện quét sai trục (`.claude/rules/library.md`) nên không dùng.

**Sửa ngoài `modelzoo`.** `training/__init__.py`: tham số mang thuộc tính `_optim` (tham số SSM của S4D) vào nhóm optimizer riêng, như `example.py` của `state-spaces/s4`; mô hình không có tham số này thì optimizer như cũ. `utils/config.py`: 13 khóa `sf_*`. Kiểm thử mới `test/test_sequential_forecast.py` (43 kiểm thử) và tài liệu `docs/`.

**Cùng số tham số.** `match_parameters` dò `hidden_size` (Transformer: bội số của số đầu chú ý) rồi dò kích thước trong (Transformer: `sf_transformer_dim_feedforward`; Mamba: `sf_mamba_d_inner`) cho số tham số gần mốc nhất, báo lỗi nếu lệch quá 1%. Kích thước trong cần thiết vì chỉ dò `hidden_size` thì Transformer và Mamba nhiều lớp không xuống dưới 1% được (bước nhảy số tham số giữa hai `hidden_size` liền nhau quá lớn). `mamba_ssm` tính `d_inner = int(expand × d_model)` (`mamba_simple.py`), nên thư viện đổi `sf_mamba_d_inner` thành `expand = (d_inner + 0,5) / d_model` để ra đúng `d_inner`.

<a id="3"></a>
## 3. Cài đặt notebook

| Mục | Giá trị | Lý do |
|---|---|---|
| Accelerator | GPU T4 x2 (dùng GPU 0) | Đúng GPU huấn luyện; `mamba_ssm` cần sm_75 trở lên |
| Internet | On | Clone fork từ GitHub |
| Input | Kaggle Dataset `hunhngcthng/lamah-ce-ext`; Output notebook `01_0_MambaSSM_Build` Version 1 | Dữ liệu và trọng số tác giả; wheel `mamba-ssm`, `causal-conv1d` dựng cho T4 |
| Thư viện | Fork `Ticasslo/neuralhydrology` commit `fed53dc` (nhánh `core-models` = `f229e02` + 2 commit mô hình chung) | Bản dùng ở bước C |

<a id="4"></a>
## 4. Các Phần và kết quả

Kết quả lấy từ bản lưu chính thức **Version 1** (Save & Run All, phiên sạch, 28/10/2026, 29,6 phút), notebook kèm output ở [04_0_Cores_Check.ipynb](04_0_Cores_Check.ipynb); mọi bảng "Điều kiện" của 7 Phần đều đạt. Môi trường: Python 3.13.15, torch 2.11.0+cu128, cuDNN 9.19, pandas 2.3.3, ảnh Kaggle dựng ngày 22/10/2026. Phiên tương tác chạy trước đó cho cùng kết luận ở mọi Phần; số đo thời gian chênh tới 18% giữa hai lần.

### Phần 1. Cài thư viện

Clone đúng commit `fed53dc`, cài `mamba-ssm` 2.3.2.post1 + `causal-conv1d` 1.6.2.post1 từ wheel tự dựng (chạy thử xuôi + ngược đạt); pip không đổi thư viện có sẵn của ảnh Kaggle; nạp được `sequential_forecast`, `sf_cores`, và `sf_cores` thấy `mamba_ssm`.

### Phần 2. Toàn bộ kiểm thử của fork trên GPU

**205 kiểm thử đạt, 0 hỏng** (13,9 phút), trong đó 43 kiểm thử của mô hình chung (`test_sequential_forecast.py`, 14,6 giây). 43 kiểm thử gồm: lõi LSTM 1 lớp trùng từng bit `SequentialForecastLSTM` khi dự báo và sau nhiều bước huấn luyện có dropout; 4 lõi không nhìn tương lai; hình dạng đầu ra; `match_parameters` cho mọi lõi và số lớp dùng ở Phần 5; nhóm optimizer với Adam, AdamW và lịch cosine; huấn luyện + đánh giá thật (dữ liệu kiểm thử CAMELS-US của thư viện) với từng lõi, bật và tắt `data_on_gpu`.

### Phần 3. Lõi LSTM so với mô hình tác giả

Hai run cùng config tác giả `Sequential_Forecast_LSTM_RA_with_q_Simple_Embedding`, cùng `best_model.pt` của tác giả (Zenodo 17292895), chỉ khác lớp mô hình: `ref` = `SequentialForecastLSTM`, `sf_lstm` = `SequentialForecast` với `sf_core: lstm` (trọng số đổi tên `lstm.` → `core.rnn.`). Thư mục Q dựng như A2 (`qmean` m³/s từ `D_gauges`, 4,1 phút; thống kê chuẩn hóa lệch tệp tác giả 3,1e-07).

| Run | Số tham số | NSE trung vị test | Bài BiasCast | Lưu vực trùng từng bit |
|---|---|---|---|---|
| `ref` | 84.785 | 0,705259 | 0,705284 | – |
| `sf_lstm` | 84.785 | 0,705259 | 0,705284 | 451/451 (lệch lớn nhất 0) |

NSE trung vị lệch bài 0,000025, như A2 (`Workspace/03_Baseline/01_A2_LamaHCEExt_Reproduce/01_A2_LamaHCEExt_Reproduce.md`). Lưu vực 55 không có `qmean` nên không có dự báo ở cả hai run, như tác giả.

### Phần 4. Lõi Mamba

Lõi Mamba 2 lớp, `hidden_size` 64, `sf_mamba_d_inner` 130, đầu vào ngẫu nhiên 8 mẫu × 365 bước × 32 chiều.

| Kiểm | Kết quả |
|---|---|
| Đổi mạnh đầu vào ở bước k (0, 100, 363, 364): đầu ra trước k đổi | 0 (đầu ra tại k đổi 3,9–5,4) |
| Hoán vị mẫu trong lô: đầu ra hoán vị theo | lệch 0 |
| Từng mẫu chạy riêng so với chạy cả lô | lệch 1,2e-06 |
| Công thức đếm tham số so với khối `mamba_ssm.Mamba` thật (6 cấu hình khối + cả mô hình) | trùng cả 7 |
| `mambapy` 1.2.0 nạp trọng số strict, so `mamba_ssm` trên GPU / CPU | sai số tương đối 1,7e-07 / 1,8e-07 |
| `mambapy` trên CPU Kaggle, 4 mẫu × 365 bước × 2 lớp, chỉ xuôi | 0,06 giây |

Hai phép thử trục cho thấy lõi quét theo thời gian, không theo lô. Công thức đếm tham số cho phép dò cùng số tham số trên máy không có `mamba_ssm`. `mambapy` cho cùng kết quả nên demo trên CPU (D2–D4, `02_Pipeline.md` Mục 9.6) dùng được trọng số huấn luyện bằng `mamba_ssm`.

### Phần 5. Cùng số tham số

Mốc: mô hình LSTM tác giả trên config này, 84.785 tham số (mốc chính thức ở bước C là LSTM của B*). Các khóa khác của lõi giữ mặc định: 4 đầu chú ý, S4D N = 64, Mamba `d_state` 16, `d_conv` 4. Bảng: [Tables/core_param_match.csv](Tables/core_param_match.csv).

| Lõi | Số lớp | `hidden_size` | Kích thước trong | Số tham số | Lệch |
|---|---|---|---|---|---|
| **LSTM** | **1** | **128** | – | **84.785** | **0** |
| LSTM | 2 | 77 | – | 84.026 | −0,895% |
| **GRU** | **1** | **150** | – | **84.663** | **−0,144%** |
| GRU | 2 | 90 | – | 84.423 | −0,427% |
| Transformer | 1 | 80 | 335 | 84.848 | +0,074% |
| **Transformer** | **2** | **56** | **243** | **84.743** | **−0,050%** |
| Transformer | 4 | 40 | 168 | 84.625 | −0,189% |
| **S4D** | **2** | **111** | – | **84.741** | **−0,052%** |
| S4D | 4 | 72 | – | 84.369 | −0,491% |
| S4D | 6 | 55 | – | 84.213 | −0,675% |
| **Mamba** | **2** | **72** | **143** | **84.959** | **+0,205%** |
| Mamba | 4 | 49 | 96 | 84.509 | −0,326% |
| Mamba | 6 | 38 | 77 | 84.387 | −0,469% |

Dòng đậm là cấu hình chính dùng ở Phần 6. Mọi 13 cấu hình đều lệch dưới 1%; các số lớp khác dành cho tinh chỉnh C1 (`02_Pipeline.md` Mục 5.2).

### Phần 6. Huấn luyện ngắn trên dữ liệu thật

Dataset train của config tác giả (451 lưu vực, 1.094.272 mẫu, 4.275 lô 256 — bằng một epoch của B0), `data_on_gpu`, dựng 1,7 phút. Mỗi lõi ở cấu hình chính chạy 200 bước huấn luyện đúng như vòng `_train_epoch` của thư viện (cùng hạt giống 111, cùng thứ tự lô, loss NSE*, cắt gradient), đo thời gian từ bước 20; rồi 50 lô dự báo ở chế độ eval. Mỗi lõi chạy 2 lần để xem có lặp lại từng bit không. S4D dùng tốc độ học tham số SSM bằng 1/10 tốc độ học chung. Bảng: [Tables/core_short_training.csv](Tables/core_short_training.csv).

| Lõi | ms/bước huấn luyện | So với lõi LSTM | Phút/epoch huấn luyện (ước tính) | ms/lô dự báo | GPU đỉnh (GB) | Hai lần chạy trùng từng bit |
|---|---|---|---|---|---|---|
| `SequentialForecastLSTM` (thư viện) | 38,0 | – | 2,71 | 15,8 | 1,58 | – |
| LSTM | 40,1 | 1,0× | 2,86 | 16,2 | 1,60 | có |
| GRU | 68,5 | 1,7× | 4,88 | 38,6 | 1,61 | có |
| Transformer | 105,6 | 2,6× | 7,52 | 65,4 | 0,98 | không (lệch 2,6e-03) |
| S4D | 114,5 | 2,9× | 8,16 | 51,2 | 1,41 | có |
| Mamba | 121,4 | 3,0× | 8,65 | 40,6 | 1,17 | không (lệch 1,7e-06) |

- **Lõi LSTM trùng từng bit `SequentialForecastLSTM` sau 200 bước huấn luyện trên GPU** (trọng số lệch 0). Cùng với Phần 3, B0 làm bằng lõi LSTM của mô hình chung sẽ cho đúng kết quả B0 đã có.
- Loss mọi lõi hữu hạn và giảm (20 bước cuối 0,027–0,035 so với 20 bước đầu 0,056–0,175). 200 bước chưa đủ để so độ chính xác giữa các lõi.
- Lõi LSTM chậm hơn mô hình thư viện 2,1 ms/bước ở Version 1 nhưng chỉ 0,3 ms ở phiên tương tác (33,9 so với 33,6), trong mức dao động giữa hai lần đo.
- **Transformer và Mamba không lặp lại từng bit** khi chạy hai lần cùng hạt giống (không bật chế độ tất định); LSTM, GRU, S4D thì có. Lệch của Mamba nhỏ (1,7e-06 sau 200 bước), lệch của Transformer lớn hơn (2,6e-03). Nhiễu giữa các lần chạy của hai lõi này vì vậy gồm cả phần không tất định của GPU, không chỉ do hạt giống.
- Thời gian là của cài đặt cụ thể trên T4: LSTM, GRU dùng nhân cuDNN; Mamba dùng nhân `mamba_ssm`; S4D, Transformer chạy PyTorch thường. Cùng số tham số không có nghĩa cùng lượng phép tính.

### Phần 7. Tóm tắt và dọn

Lưu `cores_summary.json` (số liệu chính của mọi Phần, [Tables/cores_summary.json](Tables/cores_summary.json)), `environment.json`, `manifest.csv` (sha256 10 tệp Output); xóa bản clone, thư mục Q, dataset tạm và hai run của Phần 3. Commit thư viện đúng `fed53dc`.

<a id="5"></a>
## 5. Kết luận cho bước C

- Mô hình chung chạy đúng trên GPU với cả 5 lõi: kiểm thử đạt hết, lõi LSTM trùng từng bit mô hình tác giả cả khi dự báo và khi huấn luyện, lõi Mamba nhân quả và quét đúng trục.
- Cả 13 cấu hình (5 lõi × số lớp) khớp số tham số trong 1%, nên tinh chỉnh C1 dùng được mọi số lớp đã định.
- Bước huấn luyện của các lõi mới chậm hơn lõi LSTM 1,7–3,0 lần. Với `data_on_gpu`, một epoch huấn luyện ước tính 2,9 phút (LSTM) tới 8,7 phút (Mamba), chưa tính validation và chấm test.
- Transformer và Mamba không lặp lại từng bit trên GPU; khi so với mốc nhiễu ở bước C, nhiễu của hai lõi này gồm cả phần không tất định của GPU.
- Trọng số Mamba chạy được bằng `mambapy` trên CPU với cùng kết quả, đủ cho demo.

<a id="6"></a>
## 6. Nguồn

- Konold và cs. (2026), BiasCast, HESS 30:5067–5096; config và trọng số tác giả Zenodo 17292895; dữ liệu Zenodo 17119635.
- Fork `Ticasslo/neuralhydrology` commit `fed53dc`: `modelzoo/sequential_forecast.py` (dòng 13–79), `modelzoo/sf_cores.py` (dòng 63–101 `RNNCore`, 104–142 `TransformerCore`, 145–175 `ResidualCore`, 178–256 `S4DBlock`, 264–293 `mamba_expand`, `mamba_block_parameters`, 338– `match_parameters`), `training/__init__.py` (dòng 14–55 `get_optimizer`), `test/test_sequential_forecast.py`.
- Gu, Gupta, Goel, Ré (2022), On the Parameterization and Initialization of Diagonal State Space Models, NeurIPS 2022; mã `github.com/state-spaces/s4`, `models/s4/s4d.py` và `example.py` (Apache-2.0).
- Gu & Dao (2023), Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arXiv 2312.00752; `mamba_ssm` 2.3.2.post1, `mamba_ssm/modules/mamba_simple.py` (`d_inner = int(expand * d_model)`).
- Vaswani và cs. (2017), Attention Is All You Need, NeurIPS 2017 (tầng truyền thẳng 4 × `d_model`); PyTorch 2.11 `nn.TransformerEncoderLayer` (`norm_first`).
- `mambapy` 1.2.0 (`github.com/alxndrTL/mamba.py`).
- Output notebook Version 1: `cores_summary.json`, `environment.json`, `manifest.csv`, `pytest_junit.xml`.
