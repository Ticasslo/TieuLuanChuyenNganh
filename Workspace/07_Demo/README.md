# Kết quả C2, Evaluation và demo D1–D3

## Các phiên đã kiểm

| Notebook | Nguồn | Kết quả |
|---|---|---|
| [04_C2_LamaHCEExt_Scenarios](../06_Evaluation/04_C2_LamaHCEExt_Scenarios/04_C2_LamaHCEExt_Scenarios.md) | C2 Compare Version 1, 24 checkpoint và A3 Version 1 | Tái lập S0 đúng 24 run trên 451 lưu vực và 1.461 ngày. Cả 10.824 phép kiểm theo run–lưu vực đạt. Sai số dự báo lớn nhất bằng 0 mm/ngày. Xuất 144 NPZ mới cho S1_Q1, S1_Q3, S1_Q7, S2, S3 và S4. |
| [01_Evaluation_LamaHCEExt](../06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md#132-kịch-bản-c2) | Chín Input của Version 9 và Notebook Output 04_C2 | Bảng chính giữ 56 run S0 và 652.628 ngày–lưu vực trên 450 ID có ngày hợp lệ. Phần C2 chấm 168 trường hợp qua 7 kịch bản trên 657.347 ngày–lưu vực và 451 ID. 205 kiểm thử đạt. |
| [01_D1_LamaHCEExt_Inference](01_D1_LamaHCEExt_Inference/01_D1_LamaHCEExt_Inference.md) | Kaggle Version 1, T4 cu128, `freeze-C` | Đối chiếu 24 run và 10.824 cặp run–lưu vực. Sai số dự báo lớn nhất bằng 0 mm/ngày trước và sau chặn âm. |
| [02_D2_LamaHCEExt_CPU](02_D2_LamaHCEExt_CPU/02_D2_LamaHCEExt_CPU.md) | Kaggle Version 1, CPU 4 luồng, `mambapy==1.2.0`, cùng nguồn D1 | Tám cấu hình có 36 lượt kiểm mỗi cấu hình, đạt 288/288. p95 sau khi nạp lưu vực từ 15,83 đến 180,15 ms trên Kaggle. RSS đỉnh đo được 583,34 MiB. |

Phiên 04_C2 và hai phiên D1/D2 dùng NeuralHydrology `freeze-C` tại `db6dc1ffcd8c14994cd80be3ed8ff417002c01a1`. D1 và 04_C2 dùng PyTorch 2.11.0+cu128 trên Tesla T4. Evaluation đọc các Output dự báo và chấm trên CPU. D2 dùng PyTorch 2.11.0+cpu.

## Tệp đối chiếu và cách chạy lại

- [C2_handoff.csv](../05_Training/13_C2_LamaHCEExt_Compare/Tables/C2_handoff.csv) có tên chính xác của 24 Output train. Chín run x3 dùng Output cuối có hậu tố `_Cont`. D1 và 04_C2 dùng cùng bộ 28 Input. D2 lấy D1 Version 1 và tám checkpoint đứng đầu validation, mỗi cấu hình một seed.
- [04_C2](../06_Evaluation/04_C2_LamaHCEExt_Scenarios/Archive/Version1/) giữ summary, bảng nguồn, bảng điểm theo seed, thống kê và manifest của Output. Toàn bộ NPZ, mặt nạ và các bảng chi tiết nằm ở Kaggle.
- [Evaluation Version 10](../06_Evaluation/01_Evaluation_LamaHCEExt/Archive/Version10/) giữ bảng tổng hợp, coverage, biên bản đầu vào, các bảng phân mức và sự kiện lũ cộng dồn. Giao chung theo kịch bản C2 rộng hơn giao S0 của 56 run vì hai lần chấm lấy giao trên hai tập run khác nhau.
- [D2 Version 1](02_D2_LamaHCEExt_CPU/Archive/Version1/) giữ nguyên bốn tệp Output. So CPU với NPZ ở dung sai `2e-5 + 1e-6 × |giá trị NPZ|` mm/ngày.
- D1/D2 dùng chung mô-đun [DemoRuntime](DemoRuntime/). Từ gốc repo, chạy `python Tools/Demo_Notebooks_Build.py` để sinh lại hai notebook khi sửa mô-đun.

## D3 trên laptop

[Hướng dẫn dựng và biên bản D3/C2](03_D3_LamaHCEExt_Demo/03_D3_LamaHCEExt_Demo.md) ghi tên các nguồn, lệnh đóng gói và cách bật “Nếu… thì…”. Mốc này phát lại tám cấu hình C2, BiasCast và persistence trên 451 lưu vực kỳ test 2014–2017. Năm lõi C2 đã đối chiếu suy luận CPU với NPZ trên laptop. GRU seed 111 là lựa chọn mặc định. Kiểm lại trên giao diện xác nhận phản hồi cũ được bỏ qua khi đổi mưa hoặc nhiệt lúc đang chạy.

H hạn 1–7, D 730 bước và G Integrated Gradients tiếp tục theo các Output bàn giao tương ứng. D4 trên VPS là bước triển khai sau bàn giao.
