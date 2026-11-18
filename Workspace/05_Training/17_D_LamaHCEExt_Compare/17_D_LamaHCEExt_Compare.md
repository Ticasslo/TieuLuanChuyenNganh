# D — So sánh độ dài 365 và 730 bước

Bước D kiểm lợi ích của kéo dài lịch sử trên ba cấu hình cùng mức tham số. Thứ tự chạy là LSTM, Mamba, S4D. Theo Mục 8.3 của Pipeline, khi thiếu thời gian, chạy đủ ba seed LSTM và ba seed Mamba trước, có thể bỏ toàn nhóm S4D.

| Cấu hình | Seed | Số tham số của C2 |
|---|---|---:|
| `lstm_L2_x1_lr1e-3` | 111, 222, 333 | 84.106 |
| `mamba_L2_x1_lr1e-3` | 111, 222, 333 | 85.039 |
| `s4d_L4_x1_lr1e-3` | 111, 222, 333 | 84.449 |

Số 365/730 là `seq_length` của NeuralHydrology. Mẫu gồm 364/729 bước quá khứ đến t−1 và một ngày ECMWF HRES tại ngày đích t. Hai phía dùng `freeze-C`, cùng phiên bản bộ lọc, cùng cấu hình, seed và ngân sách 30 epoch. Bộ chuẩn hóa được fit theo cùng quy trình train của thư viện. Vì khoảng warmup thay đổi theo seq_length, thống kê scaler có thể thay đổi, nên bảng đối chiếu ghi riêng hash scaler hai phía.

## 1. Input và thứ tự

Notebook CPU, Internet Off. Tên notebook `17_D_LamaHCEExt_Compare`, đặt sau `14_H_Train`, `15_H_LamaHCEExt_Compare` và `16_D_Train`.

Với LSTM và Mamba, gắn đủ 14 Output cho Compare, gồm hai nguồn chung, sáu Output C2 365 bước và sáu Output D 730 bước. Khi chạy thêm S4D, gắn đủ ba cặp S4D để tổng số nguồn là 20. Hai nguồn chung là `13_C2_LamaHCEExt_Compare` Version 1 và `02_A3_LamaHCEExt_Benchmarks` Version 1.

| Mô hình | Seed | Notebook C2 365 bước (Version 1) | Notebook D 730 bước (`16_D_Train/`) |
|---|---:|---|---|
| LSTM | 111 | `12_C2_Train_lstm_L2_x1_lr1e-3_Seed111` | `16_D_Train_lstm_L2_x1_lr1e-3_Seed111` (hoặc `_Cont`) |
| LSTM | 222 | `12_C2_Train_lstm_L2_x1_lr1e-3_Seed222` | `16_D_Train_lstm_L2_x1_lr1e-3_Seed222` (hoặc `_Cont`) |
| LSTM | 333 | `12_C2_Train_lstm_L2_x1_lr1e-3_Seed333` | `16_D_Train_lstm_L2_x1_lr1e-3_Seed333` (hoặc `_Cont`) |
| Mamba | 111 | `12_C2_Train_mamba_L2_x1_lr1e-3_Seed111` | `16_D_Train_mamba_L2_x1_lr1e-3_Seed111` (hoặc `_Cont`) |
| Mamba | 222 | `12_C2_Train_mamba_L2_x1_lr1e-3_Seed222` | `16_D_Train_mamba_L2_x1_lr1e-3_Seed222` (hoặc `_Cont`) |
| Mamba | 333 | `12_C2_Train_mamba_L2_x1_lr1e-3_Seed333` | `16_D_Train_mamba_L2_x1_lr1e-3_Seed333` (hoặc `_Cont`) |
| S4D | 111 | `12_C2_Train_s4d_L4_x1_lr1e-3_Seed111` | `16_D_Train_s4d_L4_x1_lr1e-3_Seed111` (hoặc `_Cont`) |
| S4D | 222 | `12_C2_Train_s4d_L4_x1_lr1e-3_Seed222` | `16_D_Train_s4d_L4_x1_lr1e-3_Seed222` (hoặc `_Cont`) |
| S4D | 333 | `12_C2_Train_s4d_L4_x1_lr1e-3_Seed333` | `16_D_Train_s4d_L4_x1_lr1e-3_Seed333` (hoặc `_Cont`) |

Với mỗi dòng được chọn, gắn Output hoàn thành (`status = "xong"`): nếu phiên 1 hoàn tất 30 epoch và chấm xong test thì gắn bản thường; nếu phiên 1 tạm dừng và chạy tiếp ở phiên 2 thì chỉ gắn bản `_Cont`. Các tên của nguồn 365 bước lấy từ [C2_handoff.csv](../13_C2_LamaHCEExt_Compare/Tables/C2_handoff.csv).

Compare dùng cấu trúc Output huấn luyện từ `Tools/C_Notebooks_Build.py d` (khuôn `Tools/C_Notebook_Template.py` giữ nguyên từng byte). Cổng đọc nhận `c_output` hoặc `d_output`, chứa `environment.json`, `summary.json` và `manifest.csv`, với `experiment_name = D_<cấu hình>_seed<N>`. Tệp dự báo gốc nằm ở `test/best_model/test_results.p` của thư mục run. Chốt chặn gradient mặc định tắt cho cả chín run D (trùng với chín run C2 đối chứng); nếu một run riêng lẻ phải bật `--grad-guard`, `D_config_check.csv`, `D_contract_check.csv` và `d_summary.json` ghi rõ cờ bật, số bước bỏ và trạng thái tương đương nghiêm ngặt (`strict_pair_equivalent`).

Đã đối chiếu chín NPZ 365 bước trong C2 Compare với ngữ cảnh A3 thật. Sai số lớn nhất giữa NSE trung vị ghi trong bàn giao và NSE tính từ NPZ là `3,56 × 10⁻⁸`. Các bản train 365 bước đang giữ trên laptop là bản rút gọn cho demo hoặc bản có hash môi trường khác C2 Compare. Khi chấm D trên Kaggle, gắn nguyên Notebook Output Version 1 theo bảng trên để kiểm cấu hình và manifest. Các lượt C2 đã lưu được dùng lại.

## 2. Đối chiếu và chấm

Phần 2 đối chiếu hai config đã lưu, loại các khóa đường dẫn và quản lý phiên, kiểm phần khác biệt khoa học đúng `seq_length` (`365` so với `730`). Mỗi run phải có checkpoint chọn bằng validation, đủ 30 epoch, bảng đường học `curve_<experiment>.csv` khớp `summary.json` và test hoàn thành. Hash scaler của từng run và thông tin `grad_guard` được giữ trong audit. Nếu Output huấn luyện có trường `train_samples`, kiểm số mẫu cùng mốc 1.112.307 theo audit B2; khi thiếu số từ Output, audit ghi rõ chưa xác nhận số mẫu từng run.

Phần 3 đọc kết quả đã kiểm byte, đối chiếu quan trắc A3, dựng NPZ theo hợp đồng Evaluation và tính lại NSE nguồn với quan trắc native. Phần 4 chấm NSE, PNSE và KGE trên giao ngày của các run D và C2 cùng cấu hình; kết quả giữ từng seed, từng lưu vực và tổ hợp 3 hạt giống (`D_ensemble_scores.csv`, chỉ tính trong notebook, không xuất NPZ), ghi riêng hiệu hai trung vị, trung vị hiệu ghép cặp và bội ngưỡng nhiễu B0 (`noise_sd`). Bảng bàn giao `D_handoff.csv` và bảng chọn mô hình demo theo validation `D_demo_runs.csv` ghi đủ mã băm SHA-256 của `best_model.pt`, `config.yml`, `train_data_scaler.yml` và NPZ. Hình `D_Length_Comparison.png` đặt chú giải ngoài vùng vẽ, hiển thị trực tiếp bằng `plt.show()` và in chú thích tiếng Việt từ số liệu thực ngay bên dưới. Chi phí gộp cả phiên 1 (`previous_session`) khi run chạy qua hai phiên (`_Cont`), kể cả khi phiên 2 chỉ chấm test (`SKIP_TRAIN`), rồi so trung vị phút/epoch train, phút validation/test, tổng giờ train và đỉnh bộ nhớ GPU theo cặp C2/D. Nếu GPU hai phía khác nhau, cần đọc cột GPU trước khi diễn giải tỷ số thời gian.

## 3. Đầu ra và nối tiếp

Thư mục `/kaggle/working/d_compare`. Hợp đồng dự báo tên `contracts/SF_<cấu hình>_D_seed<N>_test_S0_lead1.npz`. Output của notebook này được gắn tiếp vào `01_Evaluation_LamaHCEExt` để chấm riêng phân vùng `evaluation_output/D/` (sáu chỉ số thủy văn, mức lưu lượng và sự kiện lũ USWRC trên giao chung giữa D, C2 cùng cấu hình/seed và `Persistence_qmax`), sau đó nối vào trang **So sánh** và **Cảnh báo lũ** của `03_D3_LamaHCEExt_Demo`.

Gửi lại `d_summary.json`, `D_config_check.csv`, `D_contract_check.csv`, `D_seed_scores.csv`, `D_score_stats.csv`, `D_ensemble_scores.csv`, `D_cost.csv`, `D_cost_ratios.csv`, `D_demo_runs.csv`, `D_handoff.csv`, `D_Length_Comparison.png`, `manifest.csv`. Giữ `D_input_sources.csv`, `D_basin_scores.csv`, `D_common_mask.npz` và toàn bộ contracts trong Output.

Nguồn: [Pipeline Mục 7 và 8.3](../../../Document/01_Plan/02_Pipeline.md), [C2 Compare](../13_C2_LamaHCEExt_Compare/13_C2_LamaHCEExt_Compare.md).
