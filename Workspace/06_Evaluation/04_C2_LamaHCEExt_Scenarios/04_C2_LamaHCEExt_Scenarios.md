# C2 — Kịch bản vận hành trên test

Suy luận 24 checkpoint đã chọn của C2 trên S0, S1_Q1, S1_Q3, S1_Q7, S2, S3, S4. Nguồn 24 notebook và SHA-256 của trọng số, config và scaler nằm trong [C2_handoff.csv](../../05_Training/13_C2_LamaHCEExt_Compare/Tables/C2_handoff.csv). Giữ nguyên `best_model.pt` và scaler train. S0 tái lập để đối chiếu dự báo nguồn trước khi xuất sáu kịch bản che thêm.

| Cài đặt | Giá trị |
|---|---|
| Kaggle | Tesla T4, Internet On |
| Môi trường wheel Mamba Version 1 | Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8 |
| Dataset | `lamah-ce-ext` |
| Notebook Output, Version 1 | `02_A3_LamaHCEExt_Benchmarks`, `13_C2_LamaHCEExt_Compare`, `01_0_MambaSSM_Build` |
| Train Output, Version 1 | 24 tên đúng cột `notebook` của C2_handoff. Cấu hình x3 dùng Output `_Cont` đã xong |
| Thư viện | `freeze-C`, `db6dc1f` |
| Output | `/kaggle/working/c2_scenarios` |

Dùng bản sao từ notebook có ảnh Kaggle cu128 đã chạy được D1, giữ **Pin to original environment** rồi nhập tệp `.ipynb` này. Bộ 28 Input giống D1, gồm dataset, ba Output nguồn và 24 Output train. Chạy **Save & Run All** với các Input Version 1 đã gắn. Log sau mỗi run ghi số thứ tự trên 24 và thời gian. Mỗi checkpoint chạy bảy kịch bản, nên thời gian D1 chỉ cho biết chi phí của một lượt tái lập S0.

Phần 1 kiểm manifest, môi trường nguồn và hash các tệp train. Phần 2 làm NaN đúng những ô thuộc kịch bản sau chuẩn hóa, giữ các biến dự báo ngày t. B1b không thuộc nhóm C2. Mọi checkpoint C2 đều dùng masked mean. Phần 3 chấm 24×7 trường hợp trên cùng giao ngày–lưu vực. S0 lấy lại chính dự báo C2 đã kiểm. Xuất 144 NPZ mới kiểu `contracts/SF_<cấu hình>_C2_seed<N>_test_<kịch bản>_lead1.npz`.

Để chấm thêm mức và sự kiện, gắn nguyên Output notebook này vào Evaluation cùng C2 Compare Version 1. Bộ đánh giá giữ chỉ số C2 S0 trên giao với các mốc A–B ở thư mục chính. Bảng S0–S4 của 24 run C2 dùng chung một mặt nạ riêng trong `evaluation_output/C2_scenarios/`.

Đạt khi `c2_scenarios_summary.json` có `complete=true`, `S0_reproduced=24`, `S0_basin_checks=10824` và `new_contracts=144`. Gửi lại `c2_scenarios_summary.json`, `C2_scenarios_S0_check.csv`, `C2_scenarios_sources.csv`, `C2_scenarios_seed_scores.csv`, `C2_scenarios_stats.csv`, `manifest.csv`. Giữ toàn bộ contracts và mặt nạ trong Output. Luồng cục bộ đã kiểm hợp đồng 168 trường hợp bằng dữ liệu giả. Thời gian chạy toàn bộ trên T4 cần đo từ lần chạy thật.

Nguồn: [C2 Compare](../../05_Training/13_C2_LamaHCEExt_Compare/13_C2_LamaHCEExt_Compare.md), [Pipeline Mục 6.3](../../../Document/01_Plan/02_Pipeline.md#63-kịch-bản-vận-hành-chỉ-suy-luận).

## Kết quả đã kiểm trên Kaggle

Cả 24 checkpoint tái lập S0 trên 451 lưu vực trong 1.461 ngày. Đủ 10.824 phép kiểm theo run–lưu vực. Sai số tuyệt đối lớn nhất của dự báo bằng 0 mm/ngày, sai số quan trắc lớn nhất là 0,00001128 mm/ngày. Tám tệp đầu ra đã tải khớp byte và SHA-256 của manifest. Evaluation xác nhận đủ 144 NPZ mới và tính lại được mặt nạ chung 657.347 ngày–lưu vực trên 451 ID.

Tính lại 168 dòng theo seed từ 75.768 dòng theo lưu vực cho kết quả khớp. Bảng cấu hình gồm 56 dòng và lấy trung bình, độ lệch chuẩn mẫu của ba trung vị theo lưu vực. Dùng phạm vi test này để mô tả độ nhạy với từng kịch bản. [Bảng tổng hợp theo cấu hình](Archive/Version1/C2_scenarios_stats.csv), [bảng điểm theo seed](Archive/Version1/C2_scenarios_seed_scores.csv), [summary](Archive/Version1/c2_scenarios_summary.json), [manifest của toàn bộ Output](Archive/Version1/manifest.csv).
