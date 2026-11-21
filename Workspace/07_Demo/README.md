# Kết quả C2, H và demo D1–D3

## Các phiên đã kiểm

| Notebook | Nguồn | Kết quả |
|---|---|---|
| [04_C2_LamaHCEExt_Scenarios](../06_Evaluation/04_C2_LamaHCEExt_Scenarios/04_C2_LamaHCEExt_Scenarios.md) | C2 Compare Version 1, 24 checkpoint và A3 Version 1 | Tái lập S0 đúng 24 run trên 451 lưu vực và 1.461 ngày. Cả 10.824 phép kiểm theo run–lưu vực đạt. Sai số dự báo lớn nhất bằng 0 mm/ngày. Xuất 144 NPZ mới cho S1_Q1, S1_Q3, S1_Q7, S2, S3 và S4. |
| [Evaluation Version 10](../06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md#132-kịch-bản-c2) | Chín Input của Version 9 và Notebook Output 04_C2 | Bảng chính giữ 56 run S0 và 652.628 ngày–lưu vực trên 450 ID có ngày hợp lệ. Phần C2 chấm 168 trường hợp qua 7 kịch bản trên 657.347 ngày–lưu vực và 451 ID. 205 kiểm thử đạt. |
| [Evaluation Version 12](../06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md#kết-quả-kaggle-version-12) | Mười Input của Version 10 và H Compare Version 1 | Chấm 168 NPZ H, đủ 24 run ở mỗi hạn, trên 651.841 cặp ngày–lưu vực và 451 lưu vực. 205 kiểm thử đạt. Giữ nguyên bốn bảng S0 dùng trong demo C2. |
| [Evaluation Version 13](../06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md#kết-quả-kaggle-version-13) | Mười một Input của Version 12 và `17_D_LamaHCEExt_Compare` Version 1 | Chấm phân vùng `D/` gồm 19 chuỗi (9 cặp D/C2 + persistence) trên 657.347 cặp ngày–lưu vực và 451 lưu vực. 212 kiểm thử đạt. Giữ nguyên SHA-256 bốn bảng S0 dùng trong demo C2. |
| [01_D1_LamaHCEExt_Inference](01_D1_LamaHCEExt_Inference/01_D1_LamaHCEExt_Inference.md) | Kaggle Version 1 & Version 2, T4 cu128, `freeze-C` | V1 đối chiếu 24 run C2 (10.824 cặp, sai số 0 mm/ngày). V2 bổ sung 3 run đại diện D 730 bước (1.353 cặp trong 16,77 phút, sai số 0 mm/ngày trước và sau chặn âm; tổng 144,24 phút). |
| [02_D2_LamaHCEExt_CPU](02_D2_LamaHCEExt_CPU/02_D2_LamaHCEExt_CPU.md) | Kaggle Version 1 & Version 2, CPU 4 luồng, `mambapy==1.2.0`, cùng nguồn D1 | V1 kiểm 8 cấu hình C2 (288/288). V2 bổ sung 3 run đại diện D 730 bước đạt trọn 108/108 (`D_numeric_passed=3`): p50 ấm LSTM 19,92 ms (`1,50×`), Mamba 61,72 ms (`2,42×`), S4D 127,96 ms (`1,83×`). |

Phiên 04_C2 và hai phiên D1/D2 dùng NeuralHydrology `freeze-C` tại `db6dc1ffcd8c14994cd80be3ed8ff417002c01a1`. D1 và 04_C2 dùng PyTorch 2.11.0+cu128 trên Tesla T4. Evaluation đọc các Output dự báo và chấm trên CPU. D2 dùng PyTorch 2.11.0+cpu.

## Tệp đối chiếu và cách chạy lại

- [C2_handoff.csv](../05_Training/13_C2_LamaHCEExt_Compare/Tables/C2_handoff.csv) có tên chính xác của 24 Output train. Chín run x3 dùng Output cuối có hậu tố `_Cont`. D1 và 04_C2 dùng cùng bộ 28 Input. D2 lấy D1 Version 1 và tám checkpoint đứng đầu validation, mỗi cấu hình một seed.
- [04_C2](../06_Evaluation/04_C2_LamaHCEExt_Scenarios/Archive/Version1/) giữ summary, bảng nguồn, bảng điểm theo seed, thống kê và manifest của Output. Toàn bộ NPZ, mặt nạ và các bảng chi tiết nằm ở Kaggle.
- [Evaluation Version 10](../06_Evaluation/01_Evaluation_LamaHCEExt/Archive/Version10/) giữ bảng tổng hợp, coverage, biên bản đầu vào, các bảng phân mức và sự kiện lũ cộng dồn. Giao chung theo kịch bản C2 rộng hơn giao S0 của 56 run vì hai lần chấm lấy giao trên hai tập run khác nhau.
- [Evaluation Version 12](../06_Evaluation/01_Evaluation_LamaHCEExt/Archive/Version12/) giữ tóm tắt, kiểm đầu vào, kết quả theo hạn, phân mức và sự kiện lũ cộng dồn. Để dựng demo H, tải hai Output đầy đủ đã lưu trên Kaggle là `01_Evaluation_LamaHCEExt` Version 12 và `15_H_LamaHCEExt_Compare` Version 1.
- [D2 Version 1](02_D2_LamaHCEExt_CPU/Archive/Version1/) giữ nguyên bốn tệp Output. So CPU với NPZ ở dung sai `2e-5 + 1e-6 × |giá trị NPZ|` mm/ngày.
- D1/D2 dùng chung mô-đun [DemoRuntime](DemoRuntime/). Từ gốc repo, chạy `python Tools/Demo_Notebooks_Build.py` để sinh lại hai notebook khi sửa mô-đun.

## D3 trên laptop

[Hướng dẫn dựng và biên bản D3](03_D3_LamaHCEExt_Demo/03_D3_LamaHCEExt_Demo.md) ghi nguồn và cách chạy. Demo phát lại đủ 21 đường dự báo trên 451 lưu vực kỳ test 2014–2017: 8 cấu hình C2, 8 run H (bốn cách, hạn 1–7), 3 run D (chuỗi 730 ngày), BiasCast A2 và persistence. Năm lõi C2 và hai checkpoint H cách lai (`14_H_Train_lstm_L2_x1_lr1e-3_hybrid_Seed333`, `14_H_Train_mamba_L2_x1_lr1e-3_hybrid_Seed222` tại `freeze-H = 6f68f471bec255f67d446acd6c6556358ff9bfea`) đã hoàn tất đối chiếu suy luận CPU toàn kỳ với NPZ trên laptop (đạt 9.148.986 điểm kiểm trong [H_CPU_release_check.json](03_D3_LamaHCEExt_Demo/Archive/H_CPU_V12/H_CPU_release_check.json)). GRU seed 111 là lựa chọn mặc định.

Gói hợp nhất mới nhất `downloads/D3_CHDG_CPU_V13` tích hợp đầy đủ C2, H, D, `Evaluation` V13 (gồm bảng mức lưu lượng và sự kiện lũ USWRC của D) và **trang Giải thích (XAI)** từ bước G Version 1 (`demo/GH_demo_basin.npz` kiểm SHA-256 theo manifest cho 451 lưu vực × 7 hạn × 4 cách cùng toàn bộ bảng tổng hợp và `L90` từng lưu vực của C2 XAI; tự động mở thêm bản đồ nhiệt 61 ngày theo lưu vực của C2 ngay khi bổ sung `demo/G_demo_basin.npz`). Giao diện đã tích hợp thang màu bản đồ NSE/ΔNSE chống ngoại lệ, nội dung giấy phép CC BY 4.0 / CC BY-NC 4.0 + miễn trừ CHMI trên trang Giới thiệu và hàm WSGI `from_env()` / `/api/health` sẵn sàng cho triển khai VPS ARM64 ở D4.
