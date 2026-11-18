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

Phần 3 đọc kết quả đã kiểm byte, đối chiếu quan trắc A3, dựng NPZ theo hợp đồng Evaluation và tính lại NSE nguồn với quan trắc native. Phần 4 chấm NSE, PNSE và KGE trên giao ngày của các run D và C2 cùng cấu hình; kết quả giữ từng seed, từng lưu vực và tổ hợp 3 hạt giống (`D_ensemble_scores.csv`, chỉ tính trong notebook, không xuất NPZ), ghi riêng hiệu hai trung vị, trung vị hiệu ghép cặp và bội ngưỡng nhiễu B0 (`noise_sd`). Bảng bàn giao `D_handoff.csv` và bảng chọn mô hình demo theo validation `D_demo_runs.csv` ghi đủ mã băm SHA-256 của `environment.json`, `best_model.pt`, `config.yml`, `train_data_scaler.yml` và NPZ. Hình `D_Length_Comparison.png` đặt chú giải ngoài vùng vẽ, hiển thị trực tiếp bằng `plt.show()` và in chú thích tiếng Việt từ số liệu thực ngay bên dưới. Chi phí gộp cả phiên 1 (`previous_session`) khi run chạy qua hai phiên (`_Cont`), kể cả khi phiên 2 chỉ chấm test (`SKIP_TRAIN`), rồi so trung vị phút/epoch train, phút validation/test, tổng giờ train và đỉnh bộ nhớ GPU theo cặp C2/D. Nếu GPU hai phía khác nhau, cần đọc cột GPU trước khi diễn giải tỷ số thời gian.

## 3. Đầu ra và nối tiếp

Thư mục `/kaggle/working/d_compare`. Hợp đồng dự báo tên `contracts/SF_<cấu hình>_D_seed<N>_test_S0_lead1.npz`. Output của notebook này được gắn tiếp vào `01_Evaluation_LamaHCEExt` để chấm riêng phân vùng `evaluation_output/D/` (sáu chỉ số thủy văn, mức lưu lượng và sự kiện lũ USWRC trên giao chung giữa D, C2 cùng cấu hình/seed và `Persistence_qmax`), sau đó nối vào trang **So sánh** và **Cảnh báo lũ** của `03_D3_LamaHCEExt_Demo`.

Gửi lại `d_summary.json`, `D_config_check.csv`, `D_contract_check.csv`, `D_seed_scores.csv`, `D_score_stats.csv`, `D_ensemble_scores.csv`, `D_cost.csv`, `D_cost_ratios.csv`, `D_demo_runs.csv`, `D_handoff.csv`, `D_Length_Comparison.png`, `manifest.csv`. Giữ `D_input_sources.csv`, `D_basin_scores.csv`, `D_common_mask.npz` và toàn bộ contracts trong Output.

## 4. Kết quả đã xác minh: `lstm_L2_x1_lr1e-3` (Version 1, 3 hạt giống)

Ba lượt `16_D_Train_lstm_L2_x1_lr1e-3_Seed111/222/333` (Version 1, Tesla T4, `freeze-C = db6dc1f`, 84.106 tham số, `seq_length = 730`, tắt `grad_guard`, 0 bước cập nhật bị bỏ) đã hoàn tất đủ 30 epoch trong một phiên mỗi lượt. Toàn bộ 30/30 mục `manifest.csv` khớp SHA-256, `best_model.pt` trùng từng tensor với `model_epoch<saved_epoch>.pt`, cấu hình chỉ khác C2 đúng khóa `seq_length` (`365 → 730`) và `train_data_scaler.yml` trùng nhau ở cả ba hạt giống (`f6d8ffa6...`, khác bản 365 bước `baae107d...` do khoảng warmup đầu kỳ train thay đổi).

| Hạt giống | Epoch chọn (365 / 730) | NSE val (365 / 730) | NSE test giao chung (365 / 730) | KGE test giao chung (365 / 730) | Phút train/epoch (730) | Tổng giờ train (730) | VRAM đỉnh (MiB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 111 | 24 / 20 | 0,7679 / 0,7717 (+0,0039) | 0,7166 / 0,7161 (−0,0005) | 0,7801 / 0,7818 (+0,0017) | 5,90 | 3,71 | 4.083 |
| 222 | 20 / 16 | 0,7672 / 0,7699 (+0,0028) | 0,7188 / 0,7179 (−0,0009) | 0,7724 / 0,7699 (−0,0024) | 5,89 | 3,70 | 4.083 |
| 333 | 22 / 28 | 0,7681 / 0,7728 (+0,0047) | 0,7180 / 0,7148 (−0,0032) | 0,7627 / 0,7680 (+0,0053) | 6,75 | 4,17 | 4.083 |
| **TB ± SD** | — | **0,7677 ± 0,0005 / 0,7715 ± 0,0014** | **0,7178 ± 0,0011 / 0,7162 ± 0,0016** | **0,7717 ± 0,0087 / 0,7732 ± 0,0075** | **6,18 ± 0,49** | **3,86 ± 0,27** | **4.083** |

*Ghi chú phạm vi:* Số liệu test trên chấm sơ bộ trên giao chung 657.347 cặp ngày–lưu vực (451 lưu vực) của sáu run LSTM 365/730 bước; con số chính thức toàn bước D sẽ khóa trên giao chung của toàn bộ 6 hoặc 9 cặp khi `mamba_L2_x1_lr1e-3` (và `s4d_L4_x1_lr1e-3`) hoàn tất.

- **Độ chính xác và kiểm định:** NSE trung vị validation tăng nhẹ ở cả ba hạt giống (+0,0028 đến +0,0047, tương ứng +0,35 đến +0,59 lần ngưỡng nhiễu B0 0,0079), chọn hạt giống **333** (epoch 28, NSE val 0,7728) cho demo D. Trên tập test, chênh lệch NSE trung vị từng hạt giống (−0,0005 đến −0,0032) và tổ hợp 3 hạt giống (`0,7351` ở 365 bước so với `0,7338` ở 730 bước; `ΔNSE = −0,0012`, bằng −0,16 lần nhiễu B0) đều không có ý nghĩa thống kê sau hiệu chỉnh Holm (`p_Holm ≥ 0,075`). Tổ hợp 3 hạt giống đạt `KGE = 0,7799` ở 730 bước so với `0,7767` ở 365 bước (`+0,0031`).
- **Dòng chảy rất cao, sự kiện lũ và đặc trưng lưu vực:** Recall nhóm lưu lượng rất cao (`> Q98`, 11.871 ngày–lưu vực) tăng nhẹ ở cả ba hạt giống (từ `0,4471–0,4743` lên `0,4562–0,4797`), và sai số tuyệt đối trung bình thời điểm đỉnh lũ USWRC chu kỳ 2 năm giảm từ `0,37–0,57` ngày xuống `0,24–0,39` ngày. Xét hiệu `ΔNSE (730 − 365)` trung bình ba hạt giống theo 451 lưu vực (kiểm định hoán vị hai phía 9.999 lần), cửa sổ hai năm cải thiện có ý nghĩa thống kê tại các lưu vực có băng hà và tuyết tích lũy nhiều năm: `glac_fra` (`ρ = +0,1405`, `p = 0,0026`), `frac_snow` (`ρ = +0,1394`, `p = 0,0034`) và độ cao trung bình `elev_mean` (`ρ = +0,1057`, `p = 0,0251`).
- **Chi phí huấn luyện và suy luận CPU:** Trên Tesla T4, trung vị thời gian huấn luyện mỗi epoch tăng từ `3,20` phút (C2) lên `6,18` phút (`1,93×`, sát tỉ lệ tuyến tính `730 / 365 = 2,00×`), tổng giờ huấn luyện tăng từ `2,06` lên `3,86` giờ (`1,87×`) và đỉnh bộ nhớ GPU tăng từ `2.381` lên `4.083` MiB (`1,71×`). Kiểm thử suy luận CPU theo giao thức D2 (36 lượt trên ba lưu vực đại diện diện tích nhỏ/trung vị/lớn, 4 luồng CPU) cho hạt giống 333 đạt 36/36 lượt khớp NPZ trong dung sai `2 × 10⁻⁵ + 10⁻⁶|y|` mm/ngày (sai số tuyệt đối lớn nhất `2,19 × 10⁻⁵` mm/ngày), với độ trễ ấm `p50` tăng `2,03×` (`7,15 → 14,53` ms) và `p95 = 32,59` ms.

Nguồn: [Pipeline Mục 7 và 8.3](../../../Document/01_Plan/02_Pipeline.md), [C2 Compare](../13_C2_LamaHCEExt_Compare/13_C2_LamaHCEExt_Compare.md).
