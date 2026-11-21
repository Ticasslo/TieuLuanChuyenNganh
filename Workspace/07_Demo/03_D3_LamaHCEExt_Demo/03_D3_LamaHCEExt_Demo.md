# D3 • Demo phát lại Extended LamaH-CE

**Demo D3 đã kiểm phát lại, suy luận CPU và giải thích mô hình (XAI) trên laptop cho đủ bốn bước C2, H, D và G.** Gói hợp nhất chính thức `downloads/D3_CHDG_CPU_V13` (`169` tệp, `852,2 MB`) kết hợp `13_C2` V1, `15_H` V1, `17_D` V1, `01_Evaluation` V13 (bảng S0, H và D), `01_D1` V2, `02_D2` V2 và trọn bộ bước G V1 (`01_G_C2_XAI` + `02_G_H_XAI`): `451` lưu vực, `21` đường dự báo (`8` C2 + `8` H + `3` D + `2` mốc), `7` mô hình chạy trực tiếp CPU “Nếu… thì…” (5 lõi C2 + 2 checkpoint H cách lai), và trang Giải thích hiển thị trực tiếp theo từng lưu vực cho cả **H** (`demo/GH_demo_basin.npz`, 451 lưu vực × 7 hạn × 4 cách) lẫn **C2** (`demo/G_demo_basin.npz`, 451 lưu vực × 5 lõi kèm bản đồ nhiệt 0–30 ngày).

Trang Dự báo có bản đồ vùng nghiên cứu, 451 trạm, lịch test, các đường qmax và ngưỡng lũ theo m³/s. C2 giữ tám cấu hình thuộc năm lõi, mỗi cấu hình dùng một seed theo NSE validation. GRU seed 111 bật sẵn. Chức năng “nếu… thì…” dùng một cấu hình đại diện mỗi lõi C2, lấy cấu hình và seed có NSE validation cao nhất của lõi trong số đã kiểm CPU.

## Chuẩn bị trên laptop

D3 chạy bằng `prepare.py` và `app.py` trên laptop. Dùng lại các Output Kaggle đã lưu. Cài thư viện theo `requirements.txt`, gồm xarray để đọc dự báo A2. Để bật “nếu… thì…” cho C2, cài fork NeuralHydrology tại commit `db6dc1ffcd8c14994cd80be3ed8ff417002c01a1` cùng PyTorch CPU, ruamel.yaml, tensorboard và mambapy 1.2.0.

Giải nén các Output vào các thư mục con riêng trong `downloads/` cạnh `prepare.py` và giữ nguyên cấu trúc của từng Output. Bố cục đã kiểm dùng `downloads/D2_Version1/`, `downloads/Evaluation_V10/evaluation_output/` và bảng kịch bản trong `downloads/C2_Scenarios_V1/c2_scenarios/`. Git chứa mã, hướng dẫn và biên bản. Dữ liệu, checkpoint, Output đầy đủ và môi trường Python được chuẩn bị riêng trên từng máy.

| Vai trò | Input |
|---|---|
| Dữ liệu | Thư mục LamaH Extended từ `lamah-ce-ext`, gồm `A_basins_total_upstrm/3_shapefiles/Basins_A.shp` và `D_gauges/1_attributes/Gauge_attributes.csv` |
| Ngưỡng | `01_A1_LamaHCEExt_Explore` Version 3. Hai tệp `Tables/A1_P10_qmax_level_thresholds.csv` và `Tables/A1_P10_flood_return_levels.csv` |
| Mốc | `01_A2_LamaHCEExt_Reproduce` Version 2, giữ thư mục `a2_output/runs/`. `02_A3_LamaHCEExt_Benchmarks` Version 1, lấy `A3_persistence_predictions.npz` và `A3_Q_source_window.npz` trong `a3_output/` |
| C2 | `13_C2_LamaHCEExt_Compare` Version 1, gồm đủ 24 NPZ và manifest |
| Bản đồ NSE, cảnh báo lũ | Evaluation Version 10. Bản đồ và trang lũ dùng dữ liệu S0 từ bốn bảng `Evaluation_metrics.csv`, `Evaluation_events.csv`, `Evaluation_levels.csv` và `Evaluation_event_cases.csv` |
| Bảng S0–S4 của trang So sánh | `C2_scenarios_stats.csv` từ Output `04_C2_LamaHCEExt_Scenarios` Version 1 |
| D | Output `17_D_LamaHCEExt_Compare` sau khi có các run 730 bước; truyền thêm thư mục `D/` của Evaluation vào `--d-evaluation` (hoặc đặt cùng `--inputs`) để nối bảng mức lưu lượng và sự kiện lũ USWRC của D vào trang Cảnh báo lũ |
| H | Giải nén toàn bộ Output `15_H_LamaHCEExt_Compare` Version 1 vào `downloads/H_Compare_V1/`. Giữ nguyên thư mục `h_compare/`, gồm 168 NPZ, manifest và hai bảng bàn giao. Truyền thư mục `H/` của Evaluation Version 12 vào `--h-evaluation` để ghép đúng các bảng điểm và sự kiện. |
| CPU “nếu… thì…” C2 | Output `02_D2_LamaHCEExt_CPU` Version 1 đã kiểm và Output train Version 1 của năm cấu hình được chọn |
| CPU “nếu… thì…” H | Hai Output H Train Version 1 của LSTM lai seed 333 và Mamba lai seed 222. Giữ `h_output/` và ba tệp checkpoint, config, scaler của thư mục run tương ứng |
| Giải thích XAI (bước G) | Output `01_G_C2_XAI` Version 1 (`g_output/demo/G_demo_basin.npz` + `g_output/manifest.csv`) và `02_G_H_XAI` Version 1 (`g_output/demo/GH_demo_basin.npz` + `g_output/manifest.csv`, không cần giải nén 1,87 GB `ig/*.npz`). Các bảng tổng hợp (`G_run_metrics.csv`, `G_basin_L90.csv`, `G_lag_share.npz`, `GH_lead_summary.csv`, …) tự đọc từ Output G hoặc từ `Workspace/08_XAI/*/Tables/` trong repo |

| Lõi | Output train Version 1 cho chạy trực tiếp |
|---|---|
| LSTM | `12_C2_Train_lstm_L2_x1_lr1e-3_Seed333` |
| GRU | `12_C2_Train_gru_L2_x1_lr1e-3_Seed111` |
| Transformer | `12_C2_Train_transformer_L2_x1_lr1e-3_Seed222` |
| S4D | `12_C2_Train_s4d_L4_x1_lr1e-3_Seed222` |
| Mamba | `12_C2_Train_mamba_L4_x3_lr1e-3_Seed222_Cont` |

Mỗi Output train giữ `c_output/environment.json` và ba tệp trong thư mục run ghi ở cột `thư mục run` của [C2_handoff.csv](../../05_Training/13_C2_LamaHCEExt_Compare/Tables/C2_handoff.csv) gồm `config.yml`, `best_model.pt` và `train_data/train_data_scaler.yml`.

## Chạy trên Windows

Mở PowerShell trong thư mục chứa `prepare.py`. Tạo môi trường riêng và cài thư viện.

```powershell
python -m venv .venv
$demo_python = ".\.venv\Scripts\python.exe"
& $demo_python -m pip install -r requirements.txt
& $demo_python prepare.py --inputs .\downloads --check-inputs --with-live
```

Lệnh kiểm liệt kê các nguồn còn thiếu và in đường dẫn raw khi tìm đủ. Với ô “nếu… thì…”, cài thêm PyTorch CPU, mambapy và fork đúng commit.

```powershell
& $demo_python -m pip install "torch==2.11.0+cpu" --index-url https://download.pytorch.org/whl/cpu
& $demo_python -m pip install "mambapy==1.2.0" "git+https://github.com/Ticasslo/neuralhydrology.git@db6dc1ffcd8c14994cd80be3ed8ff417002c01a1"
& $demo_python prepare.py --inputs .\downloads --output .\downloads\D3_C2_live --with-live
```

Khi lệnh đóng gói báo thành công, truyền đường dẫn raw mà lệnh kiểm đã in. Ví dụ bên dưới dùng cấu trúc gốc của dataset.

```powershell
& $demo_python app.py --bundle .\downloads\D3_C2_live --raw .\downloads\lamah-ce-ext\lamah_ce_ext\LamaH_extended
```

Mở [demo trên laptop](http://127.0.0.1:8050). Nếu chạy phát lại trước, tạo gói mới bằng `prepare.py` với một đường dẫn `--output` khác, bỏ tùy chọn `--with-live`, rồi mở `app.py` với `--bundle` tương ứng. Dùng `--replay-only <lõi>` khi chuẩn bị gói chạy trực tiếp trên máy có lõi đã được xác định quá chậm.

### Phát lại H hạn 1–7

Trên laptop đã lưu Output `15_H_LamaHCEExt_Compare` Version 1 ở `downloads/H_Compare_V1/h_compare/` và bảng điểm H trong [Evaluation Version 12](../../06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md#kết-quả-kaggle-version-12). Các lệnh sau dùng để kiểm đầu vào và dựng gói mới từ thư mục D3. Với gói `downloads/D3_H_V12` đã dựng, dùng dòng cuối để mở lại [demo tại cổng 8052](http://127.0.0.1:8052/).

```powershell
$env:PYTHONUTF8 = "1"
$demo_python = ".\.venv\Scripts\python.exe"
$evaluationH = "..\..\06_Evaluation\01_Evaluation_LamaHCEExt\downloads\Evaluation_V12\evaluation_output\H"
& $demo_python prepare.py --inputs .\downloads --with-live --h-evaluation $evaluationH --check-inputs
& $demo_python prepare.py --inputs .\downloads --with-live --h-evaluation $evaluationH --output .\downloads\D3_H_V12
& $demo_python app.py --bundle .\downloads\D3_H_V12 --raw .\downloads\lamah-ce-ext\lamah_ce_ext\LamaH_extended --port 8052
```

Khi đóng gói, 191 mục của manifest H Compare khớp SHA-256 với bảng kiểm nguồn của Evaluation. Gói giữ tám run do H Compare chọn trên validation, mỗi cặp lõi và cách một run. Ngày đích của H nằm từ 07/01/2014 đến 25/12/2017.

**Kiểm thử trên laptop ngày 17/11/2026.** Toàn bộ 56 ma trận phát lại khớp NPZ sau quy đổi sang m³/s. Persistence dùng ngày đích trừ số ngày của hạn dự báo. Cả 42 yêu cầu dự báo khớp các cửa sổ tương ứng, gồm ngày nhuận và hai biên lịch. Tính lại 25.256 giá trị NSE theo lưu vực, run và hạn trên mặt nạ chung. Kết quả khớp Evaluation Version 12. So sánh chênh NSE qua API khớp cả 56 cặp run–hạn. Cảnh báo tại ID 6 đọc được bảng sự kiện và mức lưu lượng của cả bảy hạn. Trên trình duyệt, đã kiểm biểu đồ GEFS đi cùng persistence, bản đồ chênh NSE và bảng cảnh báo ở hạn 7. Các lượt kiểm này ghi nhận 0 lỗi và 0 cảnh báo trong console trình duyệt.

Cả 34 tệp dữ liệu và checkpoint C2 giữ nguyên SHA-256 so với gói đã kiểm trước đó. Trong gói kết hợp, API “Nếu… thì…” chạy lại được trên CPU với LSTM và Mamba của C2.

Dựng lại demo từ bản sao sạch của nhánh `main` với mã bàn giao. Cả 132 tệp trong gói dựng lại khớp SHA-256 với gói đã kiểm trên laptop. Kiểm lại API dự báo, chênh NSE và cảnh báo lũ ở hạn 7 đều đạt.

### Chạy trực tiếp H cách lai

Dùng Output Version 1 của `14_H_Train_lstm_L2_x1_lr1e-3_hybrid_Seed333` và `14_H_Train_mamba_L2_x1_lr1e-3_hybrid_Seed222`. Giải nén từng Output vào một thư mục riêng dưới `downloads/H_Train_V1/`. Mỗi Output cần `h_output/environment.json`, `h_output/manifest.csv` và ba tệp `config.yml`, `best_model.pt`, `train_data/train_data_scaler.yml` trong thư mục run ghi ở [H_handoff.csv](../../05_Training/15_H_LamaHCEExt_Compare/Tables/H_handoff.csv). Khâu đóng gói đối chiếu môi trường, epoch chọn và SHA-256 với H Compare trước khi nạp mô hình.

Hai checkpoint H dùng NeuralHydrology `freeze-H` tại `6f68f471bec255f67d446acd6c6556358ff9bfea`, PyTorch CPU 2.11.0 và mambapy 1.2.0. Tiến trình H chạy riêng theo đường dẫn `--h-python`. Ứng dụng và các mô hình C2 dùng môi trường `freeze-C` đã kiểm. Từ thư mục D3, chuẩn bị Python cho H và dựng gói mới bằng các lệnh sau. Để mở lại [demo tại cổng 8053](http://127.0.0.1:8053/) với gói `D3_H_CPU_V12` đã có, khai báo hai đường dẫn Python rồi chạy lệnh `app.py` ở dòng cuối.

```powershell
$env:PYTHONUTF8 = "1"
$demo_python = ".\.venv\Scripts\python.exe"
$h_python = ".\downloads\.venv-h\Scripts\python.exe"
$evaluationH = "..\..\06_Evaluation\01_Evaluation_LamaHCEExt\downloads\Evaluation_V12\evaluation_output\H"
& $demo_python -m venv .\downloads\.venv-h
& $h_python -m pip install -r requirements.txt
& $h_python -m pip install "torch==2.11.0+cpu" --index-url https://download.pytorch.org/whl/cpu
& $h_python -m pip install "mambapy==1.2.0" "git+https://github.com/Ticasslo/neuralhydrology.git@6f68f471bec255f67d446acd6c6556358ff9bfea"
& $h_python -m pip check
& $demo_python prepare.py --inputs .\downloads --with-live --h-evaluation $evaluationH --h-python $h_python --output .\downloads\D3_H_CPU_V12
& $demo_python app.py --bundle .\downloads\D3_H_CPU_V12 --raw .\downloads\lamah-ce-ext\lamah_ce_ext\LamaH_extended --h-python $h_python --port 8053
```

Trong **Dự báo → Nếu… thì…**, chọn LSTM hoặc Mamba có nhãn **ECMWF HRES ngày đầu**. Ngày phát hành nằm trong khoảng 07/01/2014–19/12/2017 để cả bảy ngày đích nằm trên lịch H đã đối chiếu. Hệ số mưa và mức cộng nhiệt áp dụng tại ngày phát hành. Sáu ngày khí tượng tiếp theo giữ giá trị thiếu theo giao thức H cách lai. Mỗi lần gọi API đối chiếu dự báo gốc với chuỗi phát lại ở cả bảy hạn trước khi trả bảng gốc và bảng sau thay đổi.

**Kiểm thử trên laptop ngày 17/11/2026.** Ở mỗi mô hình, kiểm 24 cửa sổ từ bốn lưu vực và sáu ngày phát hành, gồm ngày nhuận và hai biên. Cả 336 giá trị theo hạn khớp NPZ trước và sau chặn âm trong dung sai `2 × 10⁻⁵ + 10⁻⁶ × |giá trị NPZ|` mm/ngày. p95 đo trên yêu cầu gốc sau khi nạp lưu vực, qua tiến trình H với tối đa bốn luồng PyTorch. Lần đầu của LSTM gồm khởi tạo tiến trình, PyTorch và mô hình. Lần đầu của Mamba dùng tiến trình đã khởi tạo. [Bảng kiểm CPU](Archive/H_CPU_V12/H_CPU_check.csv).

| Mô hình | Epoch | Sai số tuyệt đối lớn nhất (mm/ngày) | Yêu cầu đầu (ms) | p95 sau nạp (ms) |
|---|---:|---:|---:|---:|
| LSTM lai seed 333 | 2 | 2,29 × 10⁻⁵ | 10.206 | 8,6 |
| Mamba lai seed 222 | 1 | 9,54 × 10⁻⁶ | 484 | 28,5 |

Trên trình duyệt, tại ID 6 ngày phát hành 15/06/2016, nhân mưa 1,5 và cộng nhiệt độ 2 °C cho bảng đủ bảy ngày đích từ 15/06 đến 21/06. Ở hạn 1, LSTM đổi từ 19,8888 sang 21,5528 m³/s, Mamba đổi từ 21,5690 sang 22,6395 m³/s. Kết quả là kịch bản minh họa tại một lưu vực và ngày. Console trong lượt kiểm không ghi lỗi hay cảnh báo.

Qua API, cả hai mô hình đã kiểm thay mưa riêng, thay nhiệt độ riêng, đặt mưa bằng 0 và gửi đồng thời các yêu cầu H. Kiểm lại phần “Nếu… thì…” của cả năm lõi C2 đạt. Trong gói hợp nhất, 131 tệp dữ liệu và checkpoint có sẵn khớp SHA-256 với gói phát lại trước đó.

#### Kiểm mở rộng trên toàn kỳ test

Kiểm API phát lại qua 3.297 yêu cầu bao phủ 451 lưu vực và bảy hạn. Đối chiếu 2.015.928 giá trị trên đường dự báo, 144 truy vấn bản đồ NSE và 360 truy vấn cảnh báo. Tất cả khớp dữ liệu trong gói. Tính lại số ca trúng, bỏ sót và báo nhầm từ 969.816 bản ghi sự kiện rồi đối chiếu 416.724 hàng tổng hợp. Kiểm thêm cùng số hàng mức lưu lượng. Mở đủ năm trang trên trình duyệt ẩn và kiểm lại LSTM ở ca từng lỗi, Mamba tại ngày phát hành cuối kỳ. [Báo cáo API](Archive/H_CPU_V12/D3_replay_API_summary.json) · [Báo cáo đối chiếu bảng cảnh báo](Archive/H_CPU_V12/D3_flood_tables_summary.json).

Trên mẫu ban đầu, 336 giá trị đạt dung sai tuyệt đối `2 × 10⁻⁵` mm/ngày cộng sai số tương đối đã ghi ở trên. Khi quét rộng, một số giá trị LSTM từ CPU khác NPZ chấm trên Kaggle vượt dung sai đó dù lịch và quan trắc khớp. Chạy lại CPU 64 bit trên các ca đại diện không loại bỏ chênh lệch, nên phạm vi này chưa xác định được nguồn chính xác của sai số. Demo dùng dung sai riêng cho H là `5 × 10⁻⁴ + 10⁻⁶ × |giá trị NPZ|` mm/ngày. Cả 20 ca từng vượt dung sai cũ trả đủ bảy hạn. Thử thay sai 1 m³/s trong bản sao tệp phát lại vẫn bị API từ chối. [Báo cáo kiểm lại dung sai](Archive/H_CPU_V12/H_CPU_numeric_guard_regression.json).

Lượt suy luận CPU toàn phần chạy trên 451 lưu vực × 1.449 ngày đích × 7 hạn cho từng checkpoint (tổng 9.148.986 điểm kiểm). LSTM kiểm đủ 4.574.493 giá trị trong 681,58 giây, đúng lịch và không thiếu dự báo, với sai số tuyệt đối lớn nhất 0,00013542 mm/ngày trước và sau chặn âm (11.919 điểm vượt dung sai cũ `2 × 10⁻⁵` mm/ngày của C2). Mamba hoàn tất đủ 451 lưu vực và 4.574.493 giá trị trong 12.453,70 giây (~3,46 giờ), không thiếu mục tiêu hay lệch quan trắc, sai số tuyệt đối lớn nhất 0,00003529 mm/ngày trước và sau chặn âm (3 điểm vượt dung sai cũ tại lưu vực 30 và 318). [Kết quả LSTM toàn kỳ](Archive/H_CPU_V12/H_CPU_LSTM_full_summary.json) · [Tổng hợp toàn kỳ](Archive/H_CPU_V12/H_CPU_full_summary.json).

`H_CPU_release_check.json` ghi nhận `complete=true`, `passed=true`, `coverage_ok=true`, `numeric_ok=true` cho cả 9.148.986 điểm với ngưỡng dung sai tuyệt đối 0,0005 mm/ngày cộng `10⁻⁶ × |giá trị NPZ|`. Nhật ký lỗi trống 0 byte. [Biên bản nghiệm thu H CPU](Archive/H_CPU_V12/H_CPU_release_check.json) (nhật ký chạy ngầm lưu cục bộ tại `downloads/D3_H_CPU_V12/verification/H_CPU_full.log`).

## Kiểm trước khi bàn giao

- Trang Dự báo hiện 451 trạm và lịch 2014–2017. Kiểm hai mốc BiasCast, persistence, tám cấu hình C2, GRU seed 111 bật sẵn và bốn ngưỡng lũ theo m³/s.
- Trang So sánh hiện bản đồ NSE và bảng S0–S4. Trang Cảnh báo lũ hiện bốn mức lưu lượng và các trận bắt được, bỏ sót, báo nhầm trên S0.
- Với năm lõi chạy trực tiếp, thử cùng một lưu vực và ngày có HRES. Máy chủ tự đối chiếu dự báo gốc với chuỗi phát lại trước khi trả kết quả “nếu… thì…”. Đo riêng lần đầu và lần tiếp theo ở laptop.
- Sau khi kiểm trên dữ liệu thật, bàn giao gói đã tạo trong `downloads/D3_C2_live`, mã `app.py`, thư mục `static/`, `requirements.txt`, mô-đun `DemoRuntime/forecast.py` và đường dẫn dataset cho Thắng. Giữ `DemoRuntime/` ngang cấp với thư mục D3 như trong repo để ứng dụng nạp được bộ dự báo. Thắng dùng gói này để kiểm lại trên VPS ở D4.

Các mục H/D và G hiện số liệu khi gói có nguồn tương ứng. Trang Giải thích nhận CSV `G_IntegratedGradients.csv` dự kiến với `basin_id,origin_date,core,group,day_offset,attribution` và vẽ theo từng lõi. Định dạng này cần đối chiếu với Output bước G khi bàn giao. Phiên bản localhost dành cho laptop. Việc triển khai VPS D4 thuộc bước tiếp theo.

## Biên bản kiểm thử D3 trên laptop ngày 16/11/2026

**Đạt trong phạm vi phát lại C2.** Giao diện được kiểm trong tab ẩn tại `http://127.0.0.1:8050/` trên phiên đang chạy với lệnh `python .\app.py --bundle .\downloads\D3_C2_live`. Các phép đối chiếu dữ liệu và phản hồi API chạy riêng qua bộ kiểm thử Flask trên cùng gói dữ liệu. Console của tab kiểm thử ghi nhận 0 lỗi và 0 cảnh báo trong các thao tác đã thực hiện.

| Trang | Kết quả quan sát trên phiên đang chạy |
|---|---|
| Dự báo | Mặc định là ID 2, ngày 01/01/2016, hạn 1 ngày và GRU seed 111. Bản đồ hiển thị 451 điểm. Đổi sang ID 6, ngày 15/06/2016 và bật `Persistence qmax` cập nhật biểu đồ với quan trắc, GRU, persistence và bốn ngưỡng lũ 1,5/2/5/10 năm theo m³/s. Tắt persistence làm đường này biến mất. Bấm điểm trên bản đồ tại ngày 01/01/2014 đổi lựa chọn và biểu đồ sang ID 130. Danh sách có tám cấu hình C2 và hai mốc BiasCast, persistence. |
| So sánh mô hình | Bản đồ NSE đổi theo lựa chọn mô hình và mốc tham chiếu. Thử Mamba L4 x3 so với persistence, bản đồ có 450 điểm hữu hạn trên 451 ID. Bảng C2 S0–S4 hiển thị 56 dòng, gồm ba biến thể S1. Bảng D2 CPU hiển thị 8 dòng. |
| Cảnh báo lũ | Đổi sang Mamba L4 x3 seed 222, ID 6, hạn 1. Hiển thị 4 chu kỳ lặp lại, 7 dòng sự kiện và 4 mức lưu lượng. Số liệu mẫu khớp Evaluation với 6 trường hợp bỏ sót và 1 trường hợp báo nhầm khi cộng qua các ngưỡng. |
| Giải thích | Trang mở được và hiển thị thông báo đang chờ Integrated Gradients của bước G. |
| Giới thiệu | Trang hiển thị mô tả dữ liệu, kiến trúc, cách đọc NSE và nguồn tham khảo. |

- So toàn bộ tám chuỗi C2 trong gói với NPZ từ `13_C2_LamaHCEExt_Compare` Version 1. Mỗi chuỗi có 451 × 1.461 ô ngày–lưu vực. Thứ tự ID, ngày, các giá trị thiếu và giá trị sau quy đổi từ mm/ngày sang m³/s đều khớp chính xác. Tại ID 6 ngày 15/06/2016, API trả GRU seed 111 là 18,506696 m³/s và Mamba L4 x3 seed 222 là 23,747490 m³/s. Cửa sổ biểu đồ tại ngày đầu và cuối kỳ đều có 31 ngày nằm trong kỳ test.
- Mức chênh NSE tại ID 6 giữa Mamba L4 x3 seed 222 và persistence là +0,247801, khớp phép trừ từ `Evaluation_metrics.csv` Version 10. Số dòng và loại sự kiện mẫu khớp `Evaluation_event_cases.csv` cùng phiên bản.
- SHA-256 của cả năm tệp `best_model.pt` trong gói khớp `D2_CPU_benchmark.csv` Version 1. Các yêu cầu gửi ID, ngày, hạn hoặc run ngoài phạm vi trả HTTP 400 kèm thông báo. API “nếu… thì…” cũng trả HTTP 400 trong chế độ phát lại, tương ứng với thông báo tính năng chưa bật trên giao diện.

Lần kiểm phát lại ở trên dùng `--bundle`. Các số CPU trên trang So sánh lấy từ D2 Version 1 chạy trên Kaggle. Kết quả đo trực tiếp trên laptop ở phần dưới.

### Suy luận CPU “nếu… thì…” trên laptop ngày 16/11/2026

**Đạt 5/5 lõi C2 qua API và giao diện.** Phiên local dùng `.venv` với Python 3.12.6, PyTorch `2.11.0+cpu`, NeuralHydrology commit `db6dc1ffcd8c14994cd80be3ed8ff417002c01a1`, mambapy `1.2.0`, NumPy `2.5.5`, pandas `2.3.3` và xarray `2026.9.0`. `pip check` đạt. Giao diện được kiểm trên cổng `8051`, mở riêng với dữ liệu gốc để bật suy luận CPU.

Kiểm tại lưu vực 6, ngày phát hành và ngày đích 15/06/2016, hạn 1. Mỗi lõi có 8 yêu cầu thành công. Bốn trường hợp gồm giữ nguyên HRES, nhân mưa ×1,5, cộng nhiệt độ +2 °C và áp dụng cả hai. Trường hợp áp dụng cả hai được lặp lại thêm bốn lần với kết quả giống nhau. Yêu cầu nhân mưa với hệ số âm bị từ chối với HTTP 422. Sai số tuyệt đối lớn nhất giữa dự báo CPU gốc và chuỗi NPZ phát lại là `3,58 × 10⁻⁵ m³/s`, trong ngưỡng đối chiếu của ứng dụng sau quy đổi đơn vị. Trên trình duyệt đã bấm **Chạy thử** cho cả năm lõi và xác nhận bảng kết quả cùng ngày, hạn và số liệu sau làm tròn. Console ghi nhận 0 lỗi và 0 cảnh báo trong các thao tác kiểm.

| Lõi và seed | Phát lại (m³/s) | CPU gốc (m³/s) | Mưa ×1,5 (m³/s) | Nhiệt +2 °C (m³/s) | Cả hai (m³/s) | Yêu cầu đầu (ms) | Trung vị sau nạp (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRU 111 | 18,5067 | 18,5067 | 18,9960 | 18,9683 | 19,5850 | 6.372 | 151 |
| Mamba 222 | 23,7475 | 23,7475 | 24,5484 | 24,4552 | 25,3211 | 517 | 182 |
| LSTM 333 | 18,7991 | 18,7991 | 19,5401 | 19,3375 | 20,4820 | 456 | 93 |
| Transformer 222 | 22,9251 | 22,9251 | 23,8845 | 23,3723 | 24,3846 | 473 | 95 |
| S4D 222 | 20,9075 | 20,9075 | 21,9263 | 21,7872 | 22,9266 | 554 | 198 |

Thời gian đo bằng Flask test client, tối đa bốn luồng PyTorch. Yêu cầu đầu của GRU gồm khởi tạo thư viện và nạp mô hình trong tiến trình. Các lõi tiếp theo dùng tiến trình đã khởi tạo. Cột trung vị lấy từ năm yêu cầu áp dụng cả hai thay đổi sau khi đã nạp mô hình và lưu vực. Đây là số đo tại một lưu vực và ngày trên laptop.

Nhật ký máy chủ ghi HTTP 404 cho `/favicon.ico`. Cả năm lần gửi biểu mẫu “nếu… thì…” từ trình duyệt trả HTTP 200.

Mở [phiên có “nếu… thì…”](http://127.0.0.1:8051/). Để mở lại sau khi tắt, chạy từ thư mục D3.

```powershell
.\.venv\Scripts\python.exe .\app.py --bundle .\downloads\D3_C2_live --raw .\downloads\lamah-ce-ext\lamah_ce_ext\LamaH_extended --port 8051
```

Phạm vi của lần kiểm ngày 16/11/2026 là C2. Kết quả phát lại H được ghi trong mục Phát lại H hạn 1–7.

### Kiểm thử đầy đủ gói D3 trên laptop ngày 16/11/2026

**Phát lại và suy luận C2 đạt các kiểm tra số liệu.** Lần kiểm này phát hiện lỗi giao diện khi đổi đầu vào lúc yêu cầu CPU đang chạy. Kết quả kiểm lại sau sửa được ghi ở cuối biên bản. Thao tác giao diện được thực hiện trên phiên thật ở cổng 8051. Kiểm tra dữ liệu và API dùng trực tiếp gói `D3_C2_live` bằng Flask test client trong môi trường đã ghi ở trên.

| Hạng mục | Phạm vi và kết quả |
|---|---|
| Kiểm đầu vào và đóng gói lại | `prepare.py --check-inputs --with-live` đạt. Đóng gói lại từ các nguồn gốc cho cùng nội dung catalog và 34 tệp còn lại khớp SHA-256. Thứ tự điểm và đa giác vùng nghiên cứu khớp. Shapely báo API biến đổi tọa độ cũ sắp bị loại bỏ, gói vẫn tạo thành công. |
| Nguồn dự báo và ngưỡng | Khớp toàn bộ 8 chuỗi C2, mốc BiasCast từ A2, persistence và quan trắc từ A3. Kiểm thứ tự 451 ID, lịch đủ 1.461 ngày và quy đổi đơn vị. Cả 1.804 ngưỡng lũ khớp bảng A1 sau quy đổi. Tám bảng CSV và năm checkpoint khớp SHA-256 nguồn. Danh sách seed C2 khớp lựa chọn theo NSE validation. |
| Trang Dự báo | 463 yêu cầu phát lại, bao phủ đủ 451 lưu vực và thử thêm ngày đầu, ngày nhuận 29/02/2016 và ngày cuối. Mỗi yêu cầu đối chiếu 10 chuỗi dự báo cùng quan trắc và ngưỡng. Trên giao diện, bật cả 10 chuỗi ở ID 876 ngày 31/12/2017 để kiểm 15 đường gồm các ngưỡng và quan trắc. Tắt cả 10, biểu đồ còn quan trắc và bốn ngưỡng. |
| Trang So sánh mô hình | 110 lượt kiểm NSE tuyệt đối và chênh NSE với mọi cặp trong 10 lựa chọn. Số liệu theo lưu vực và nội dung bốn bảng khớp nguồn. Giao diện lần lượt hiện 8 dòng hiệu năng C2, 8 dòng chi phí C2, 8 dòng D2 CPU và 56 dòng kịch bản C2. |
| Trang Cảnh báo lũ | 40 cặp mô hình–lưu vực từ 10 lựa chọn × 4 ID. Bảng ngưỡng, từng trận và bốn mức lưu lượng khớp Evaluation V10. Trong mẫu có 20 lần bắt được, 260 lần bỏ sót và 89 lần báo nhầm, cộng theo từng mô hình, lưu vực và ngưỡng. |
| Giải thích và Giới thiệu | Giải thích hiển thị đúng trạng thái chờ dữ liệu G. Giới thiệu mở được và hiện nội dung nguồn, dữ liệu, kiến trúc, cách đọc NSE. |
| Kiểm dữ liệu ngoài phạm vi | 10 yêu cầu GET với lưu vực, ngày, run hoặc hạn không hợp lệ và một yêu cầu POST trên bản phát lại trả HTTP 400. 8 ca đầu vào sai trên phiên CPU trả HTTP 400/422 theo trường hợp. Hệ số mưa bằng 0 được chấp nhận. |
| Suy luận CPU | 5 lõi × 4 lưu vực (2, 6, 130, 876) × 4 ngày (01/01/2014, 29/02/2016, 15/06/2016, 31/12/2017). Cả 80 yêu cầu giữ nguyên HRES và 80 yêu cầu nhân mưa ×1,5 cùng cộng nhiệt độ +2 °C trả HTTP 200. Dự báo gốc trong cả 80 trường hợp khớp NPZ theo ngưỡng đối chiếu. Giữ nguyên HRES cho cùng kết quả gốc. Đổi HRES cho giá trị thô khác ở cả 80 trường hợp. Sai số tuyệt đối lớn nhất của dự báo gốc là 3,58 × 10⁻⁵ m³/s. |
| H, D, G với dữ liệu giả lập | Kiểm API phát lại H đủ hạn 1–7, giữ trống 6 ngày ở mỗi biên, đọc so sánh và sự kiện ở hạn 7. Kiểm đọc NSE của D chuỗi 730 ngày và chênh NSE với C2 cùng seed. Kiểm G chọn đúng lưu vực/ngày và từ chối bảng thiếu cột. Dữ liệu kết quả thực tế H, D và G chưa có trong bộ đầu vào đã rà. H chạy trực tiếp và VPS D4 còn chờ kiểm. |

#### Lỗi giao diện đã tái hiện

1. Mở **Dự báo → Nếu… thì…**. Chọn GRU seed 111, ID 6, ngày 15/06/2016, nhiệt độ cộng bằng 0 và hệ số mưa bằng 1.
2. Bấm **Chạy thử** rồi đổi hệ số mưa sang 1,5 trong lúc nút còn bị vô hiệu hóa.
3. Khi yêu cầu cũ trả về, ô mưa đang hiện 1,5 nhưng bảng hiển thị `18,5067 → 18,5067 m³/s`, tức kết quả với hệ số 1. Bấm lại với hệ số 1,5 trả đúng `18,5067 → 18,9960 m³/s`.

Giao diện trước khi sửa chỉ đối chiếu lưu vực, ngày và mô hình trước khi hiển thị phản hồi. Hai giá trị mưa và nhiệt chưa được đối chiếu với yêu cầu đã gửi, nên phản hồi cũ có thể hiện dưới đầu vào mới.

#### Kiểm lại sau sửa và trạng thái bàn giao

`static/demo.js` lưu cả lưu vực, ngày, mô hình, hệ số mưa và mức cộng nhiệt tại lúc gửi yêu cầu. Kết quả chỉ hiện khi các giá trị này vẫn khớp với biểu mẫu. Đổi mưa hoặc nhiệt xóa bảng kết quả ngay khi sửa đầu vào.

| Ca kiểm lại trên cổng 8051 | Kết quả |
|---|---|
| Đổi hệ số mưa từ 1 sang 1,5 khi Mamba đang chạy, giữ nhiệt +0 °C | Phản hồi cũ bị bỏ qua. Lần gửi lại tại ID 6 ngày 15/06/2016 trả 23,7475 → 24,5484 m³/s. |
| Đổi nhiệt từ +0 sang +2 °C khi LSTM đang chạy, giữ mưa ×1 | Phản hồi cũ bị bỏ qua. Lần gửi lại tại cùng lưu vực và ngày trả 18,7991 → 19,3375 m³/s. |

Cả hai lần kiểm xác nhận nút **Chạy thử** trở lại trạng thái bấm được khi yêu cầu kết thúc. Kiểm cú pháp `demo.js` bằng Node đạt. Trình duyệt ghi nhận 0 lỗi và 0 cảnh báo trong hai ca kiểm lại.

Trước khi bàn giao mốc C2, dựng lại gói từ bản sao sạch của `main` với đúng các tệp mã D3 sẽ đưa lên Git. Catalog trùng nội dung và 34 tệp còn lại trùng SHA-256 với gói đã kiểm. Bản phát lại, bảng so sánh, bảng cảnh báo và API “Nếu… thì…” của cả năm lõi đều qua kiểm tra trên bản dựng này.

**Bản D3 đã qua kiểm thử phát lại và suy luận C2, H trên laptop.** Các đầu vào sau cần cho phần kết quả D và giải thích mô hình.

| Phần | Đầu vào và việc cần xác nhận |
|---|---|
| D chuỗi 730 ngày | Output `17_D_LamaHCEExt_Compare` (`d_summary.json`, `D_score_stats.csv`, `D_ensemble_scores.csv`, `D_cost.csv`, `D_basin_scores.csv`, `D_demo_runs.csv`, `D_handoff.csv`, `manifest.csv` và các NPZ kỳ test hạn 1) cùng thư mục `evaluation_output/D/` (`Evaluation_D_summary.json`, `Evaluation_D_input_check.csv`, `Evaluation_D_{summary,length_pairs,metrics,events,levels,event_cases}.csv`). `prepare.py --d-evaluation <thư_mục_D>` đối chiếu manifest và bảng bàn giao của D Compare với bảng kiểm nguồn Evaluation rồi nối trực tiếp vào trang Dự báo, So sánh và Cảnh báo lũ. |
| Giải thích G | Hỗ trợ chuẩn bàn giao chính thức bước G Version 1 (`demo/G_demo_basin.npz` cho C2 và `demo/GH_demo_basin.npz` cho H, kiểm SHA-256 từng mục với `g_output/manifest.csv`, kiểm đủ 451 ID, kích thước mảng, giá trị hữu hạn và tổng tỉ phần bằng 1) cùng toàn bộ bảng tổng hợp từ `Workspace/08_XAI/*/Tables/` (đồng thời vẫn tương thích ngược với `G_IntegratedGradients.csv` cũ). |
| Triển khai VPS D4 | Bàn giao cho Thắng gói đã kiểm (`D3_CHDG_CPU_V13`), `app.py`, `prepare.py`, `static/`, `requirements.txt`, `DemoRuntime/{audit.py,forecast.py,h_cpu.py}`, phiên bản hai môi trường Python (`freeze-C` và `freeze-H`) và bảng đối chiếu CPU x86_64 để Thắng kiểm lại suy luận CPU trên kiến trúc ARM64 (`linux/aarch64`, Oracle Cloud Ampere A1) trước khi công bố theo Pipeline Mục 9.6. |

### Tích hợp trang Giải thích (XAI bước G) và gói `D3_CHDG_CPU_V13` (cập nhật 21/11/2026)

- **Đóng gói và kiểm định (`prepare.py`):** Bổ sung `verify_g_manifest_entry`, `validate_g_c2_demo`, `validate_g_h_demo` và `package_xai`. Khâu đóng gói đối chiếu trực tiếp SHA-256 và dung lượng của `demo/G_demo_basin.npz` / `demo/GH_demo_basin.npz` với `g_output/manifest.csv` mà không đòi giải nén thư mục `ig/*.npz` (~1,87 GB), kiểm đủ thứ tự 451 lưu vực A3, tổng `phase_share`/`group_share`/`lag_share` bằng 1 và `sum(fore_day_share) = phase_share[..., 1]`, rồi xuất `G_C2_demo.npz` / `G_H_demo.npz` kèm 14 bảng tổng hợp XAI và trường `xai` trong `catalog.json`. Đồng thời sửa lỗi tìm `D_cost.csv` đọc thẳng từ thư mục `d_compare/` để không đụng gói `D3_*` cũ nằm chung trong `downloads/`.
- **API và giao diện (`app.py`, `static/{index.html,demo.js,style.css}`):** Nâng cấp `GET /api/explanation` hỗ trợ `scope=h` (451 lưu vực × 4 cách `gefs/hybrid/past/perfect` × 7 hạn: tỉ phần pha theo hạn, đóng góp 9 nhóm nguồn tại hạn chọn, bản đồ nhiệt `7 × 7` ngày dự báo LSTM vs. Mamba hoặc đường cong `L50/L90` với `past`), `scope=c2` (5 lõi C2: 7 nhóm nguồn, tích lũy độ trễ 0–60 ngày, bản đồ nhiệt 0–30 ngày từng lưu vực khi có `G_C2_demo.npz` hoặc so sánh `L90` lưu vực hiện tại với trung vị toàn miền từ `G_basin_L90.csv` khi chưa tải `G_demo_basin.npz`) và chế độ tương thích ngược `scope=legacy`. Nâng cấp `GET /api/health` trả thêm cờ `xai_h`, `xai_c2_basin`, `xai_c2_summary`.
- **Kiểm thử trên gói thật `downloads/D3_CHDG_CPU_V13` (`169` tệp, `852,2 MB`):**
  - Giải nén và kiểm SHA-256 trọn bộ tệp ngoài `ig/` từ cả hai notebook G Version 1: `02_G_H_XAI` V1 (`20` tệp, `demo/GH_demo_basin.npz`: `3.573.887` byte, SHA-256 `33158af069ca9ea3b374fdd2d2d50c5767f7ec2aaf7ca08029eefed44775bfec`) và `01_G_C2_XAI` V1 (`24` tệp, `demo/G_demo_basin.npz`: `17.046.888` byte, SHA-256 `1f69ae328b2baa83273590bb5b53d4bbd6ec46ada0648fafc8f0154e9b98a629`).
  - `prepare.py --check-inputs` và đóng gói hoàn chỉnh `D3_CHDG_CPU_V13` đạt (`451` lưu vực, `21` đường dự báo, `7` mô hình chạy trực tiếp CPU).
  - Kiểm thử tự động qua Flask `test_client()`: `/api/health` (`models=21`, `live_what_if=7`, `xai_h=true`, `xai_c2_basin=true`, `xai_c2_summary=true`), `/api/catalog` (`451` lưu vực, `5` run demo C2 + `8` run demo H), hồi quy `/api/forecast`, `/api/comparison` (D vs. C2 cùng seed), `/api/flood` (H hạn 7) và `/api/what-if` trực tiếp trên CPU cho cả C2 (GRU seed 111: `18,5067 → 19,5850 m³/s`) và H (LSTM lai seed 333: `19,8888 → 21,5528 m³/s` đủ 7 hạn).
  - Đối chiếu số liệu chính xác (`atol = 10⁻⁶`) cho **48 truy vấn H XAI** (`4` lưu vực `2, 6, 238, 876` × `4` cách × `3` hạn `1, 4, 7`) trùng khớp tuyệt đối với `GH_demo_basin.npz`, **20 truy vấn C2 XAI** (`4` lưu vực × `5` lõi gồm ma trận nhiệt `6 × 31`, `group_share`, `lag_cum_60`, `L50/L90`) trùng khớp tuyệt đối với `G_demo_basin.npz`, và **4 ca đầu vào lỗi** trả HTTP `400`. Kiểm cú pháp `static/demo.js` bằng `node --check` đạt `0` lỗi.

### Hoàn thiện giao diện công khai và cấu hình WSGI cho D4 (cập nhật 18/11/2026)

- **Thang màu bản đồ trang So sánh (`static/demo.js`):** Khóa dải màu hiển thị `[-1; 1]` quanh `0` cho NSE tuyệt đối và đối xứng `±P95(|ΔNSE|)` (giới hạn trong `[0,05; 1,0]`) cho hiệu NSE, tránh trường hợp một vài trạm phương sai nhỏ có NSE âm sâu (như ID 758 ≈ −13,22) kéo giãn toàn bộ thang màu làm nhạt 448 trạm còn lại. Hộp thoại rê chuột (`hover`) vẫn giữ nguyên giá trị thực chưa cắt.
- **Tuân thủ giấy phép và miễn trừ trên trang Giới thiệu (`static/index.html`):** Ghi đầy đủ ba nguồn LamaH-CE (CC BY 4.0), Extended LamaH-CE (CC BY-NC 4.0, DOI `10.5281/zenodo.17119635`) và BiasCast (bài báo HESS 2026, trọng số `10.5281/zenodo.17292895`, mã nguồn), nêu rõ giao diện chỉ phát lại kỳ kiểm tra 2014–2017 để minh họa nghiên cứu, không phải dự báo thời gian thực hay cảnh báo lũ chính thức (ghi nhận nguồn lưu lượng CHMI tại Cộng hòa Séc theo Pipeline Mục 1.6, 9.5) và phục vụ phi thương mại, không thu phí, không quảng cáo.
- **Điểm vào WSGI, kiểm sức khỏe và luồng CPU cho VPS ARM64 (`app.py`, `DemoRuntime/h_cpu.py`):** Bổ sung hàm `from_env()` đọc `DEMO_BUNDLE`, `DEMO_RAW`, `DEMO_H_PYTHON`, biến `DEMO_THREADS` giới hạn số luồng PyTorch, tham số `--host` (mặc định `127.0.0.1` trên laptop) và đường dẫn `GET /api/health`. Trên máy chủ ARM64 ít nhân vật lý (Oracle Ampere A1 Always Free, 2 OCPU / 12 GB RAM, không siêu phân luồng), cài wheel `torch==2.11.0` CPU `linux_aarch64` từ `https://download.pytorch.org/whl/cpu` và `mambapy==1.2.0`, đặt `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `OMP_WAIT_POLICY=PASSIVE`, `DEMO_THREADS=2`, chạy **1 tiến trình worker nhiều luồng** (`gunicorn --workers 1 --threads 4 --timeout 120 -b 0.0.0.0:8050 "app:from_env()"`) để chia sẻ bộ nhớ đệm `DemoForecaster` (`RLock`) và duy nhất một tiến trình con `HWorker` (`Lock`), giữ tổng RSS dưới 1,5 GB và tránh tranh chấp luồng OpenMP.

D4 là bước triển khai VPS sau bàn giao gói `D3_CHDG_CPU_V13`.

Nguồn [Pipeline Mục 9.6](../../../Document/01_Plan/02_Pipeline.md#96-đưa-mô-hình-vào-demo), [wheel PyTorch CPU](https://download.pytorch.org/whl/cpu/torch/), [Flask](https://flask.palletsprojects.com/en/stable/quickstart/) và [Plotly.js](https://plotly.com/javascript/plotlyjs-function-reference/).
