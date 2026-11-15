# D2 • Đối chiếu và đo CPU

Mở `02_D2_LamaHCEExt_CPU.ipynb` trên Kaggle, chọn **Accelerator = None** và **Internet = On**. Gắn 12 Input gồm dataset `lamah-ce-ext`, Output Version 1 của `02_A3_LamaHCEExt_Benchmarks`, `13_C2_LamaHCEExt_Compare`, `01_D1_LamaHCEExt_Inference` và tám Output train dưới đây. Dùng bản D1 đã lưu xong với `complete=true`. D2 ghim `freeze-C`, kiểm hash với D1 và chạy Mamba bằng `mambapy==1.2.0` trên PyTorch CPU. Lớp dự báo dùng chung với D1.

| Cấu hình | Notebook Output Version 1 |
|---|---|
| LSTM x1 | `12_C2_Train_lstm_L2_x1_lr1e-3_Seed333` |
| GRU x1 | `12_C2_Train_gru_L2_x1_lr1e-3_Seed111` |
| Transformer x1 | `12_C2_Train_transformer_L2_x1_lr1e-3_Seed222` |
| Transformer x3 | `12_C2_Train_transformer_L2_x3_lr5e-4_Seed333_Cont` |
| S4D x1 | `12_C2_Train_s4d_L4_x1_lr1e-3_Seed222` |
| S4D x3 | `12_C2_Train_s4d_L4_x3_lr1e-3_Seed333_Cont` |
| Mamba x1 | `12_C2_Train_mamba_L2_x1_lr1e-3_Seed222` |
| Mamba x3 | `12_C2_Train_mamba_L4_x3_lr1e-3_Seed222_Cont` |

Nhấn **Run All** sau khi gắn Input. Mỗi cấu hình lấy seed có NSE trung vị validation cao nhất. Đối chiếu 36 dự báo một ngày trên ba lưu vực theo diện tích và 12 ngày trải đều với NPZ mà D1 đã tái lập trên T4. Báo thời gian nạp, thời gian lần đầu, p50/p95 khi đã nạp dữ liệu lưu vực và RSS của tiến trình lấy mẫu trên máy Kaggle.

`complete=true` và `runs=8` xác nhận notebook đã xuất bảng đủ tám cấu hình. `numeric_passed=8` và `checks=36` ở mỗi dòng xác nhận cả 288 phép so nằm trong dung sai. Run lệch hoặc lỗi CPU có trạng thái `replay_only_numeric_or_runtime` và lý do trong cột `error`. Giữ lại bảng kể cả khi có run chỉ phát lại. Số đo p95 dùng để chốt độ trễ trên thiết bị đích.

Gửi số Version, `D2_summary.json`, `D2_CPU_benchmark.csv`, `environment.json`, `manifest.csv` trong `/kaggle/working/d2_output`.

Đối chiếu triển khai CPU theo [mambapy 1.2.0](https://pypi.org/project/mambapy/1.2.0/) và [Phần 8 của thử nghiệm Mamba](../../04_Setup/01_0_MambaSSM_Build/01_0_MambaSSM_Build.md).

## Kết quả Kaggle Version 1

D2 dùng 4 luồng CPU, PyTorch 2.11.0+cpu, mambapy 1.2.0 và `freeze-C`. Cả 8 cấu hình đạt 36 phép đối chiếu mỗi cấu hình. Dung sai cho từng dự báo là `2e-5 + 1e-6 × |giá trị NPZ|` mm/ngày. Sai số tuyệt đối lớn nhất báo cáo là 0,00003052 mm/ngày. Manifest D2 khớp cả ba tệp đầu ra. Tám SHA-256 checkpoint khớp `C2_handoff.csv`, còn SHA-256 manifest của C2 Compare khớp nguồn Evaluation Version 9.

| Cấu hình | Seed | p95 khi đã nạp lưu vực (ms) | Đại diện lõi cho D3 theo NSE validation |
|---|---:|---:|---|
| LSTM x1 | 333 | 15,83 | Có |
| GRU x1 | 111 | 37,89 | Có |
| Transformer x1 | 222 | 16,58 | Có |
| Transformer x3 | 333 | 18,07 | |
| S4D x1 | 222 | 79,64 | Có |
| S4D x3 | 333 | 180,15 | |
| Mamba x1 | 222 | 45,02 | |
| Mamba x3 | 222 | 99,26 | Có |

RSS đỉnh của tiến trình trong phiên Kaggle là 583,34 MiB. Lần dự báo đầu tiên của GRU, cấu hình chạy đầu notebook, mất 3,33 giây khi chưa nạp dữ liệu lưu vực, bên cạnh 4,58 giây đo riêng khi nạp mô hình. Năm cấu hình đại diện đủ điều kiện về sai số để đóng gói ô “nếu… thì…” cho D3. Đo lại độ trễ trên laptop và VPS trước khi chốt khả năng chạy trực tiếp trên từng máy.

[Bảng benchmark gốc](Archive/Version1/D2_CPU_benchmark.csv), [summary](Archive/Version1/D2_summary.json), [môi trường](Archive/Version1/environment.json) và [manifest](Archive/Version1/manifest.csv).
