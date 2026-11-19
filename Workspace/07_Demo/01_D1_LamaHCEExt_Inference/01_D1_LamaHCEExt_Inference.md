# D1 • Một hàm dự báo cho C2, H và D

`DemoRuntime/forecast.py` nạp config, checkpoint và scaler của từng run. Hàm `DemoForecaster.predict(basin, origin_date, hres=None)` trả ngày đích, hạn và qmax theo mm/ngày và m³/s. `origin_date` là ngày đầu của pha dự báo. Cửa sổ C2, H và D lần lượt gồm 364+1, 364+7 và 729+1 bước. H cách lai chỉ thay đầu vào ECMWF HRES ngày đầu. Giá trị thay trực tiếp dùng đơn vị gốc. Hệ số mưa tác động lên `ECMWF_tp` và mức cộng nhiệt độ tác động lên `ECMWF_t2m`.

## Kaggle

| Cài đặt | Giá trị |
|---|---|
| Notebook | `01_D1_LamaHCEExt_Inference` |
| Thiết bị | T4, Internet On, ảnh Python 3.13.15 / torch 2.11.0+cu128 / CUDA 12.8 khớp wheel nguồn |
| Dataset | `lamah-ce-ext` |
| Output Version 1 | `02_A3_LamaHCEExt_Benchmarks`, `13_C2_LamaHCEExt_Compare`, `01_0_MambaSSM_Build` |
| Output huấn luyện Version 1 | Đủ 24 tên ở cột `notebook` trong [C2_handoff.csv](../../05_Training/13_C2_LamaHCEExt_Compare/Tables/C2_handoff.csv). Các run x3 dùng Output cuối có hậu tố `_Cont` |
| Thư viện C2 | `freeze-C`, `db6dc1ffcd8c14994cd80be3ed8ff417002c01a1` |
| Output | `/kaggle/working/d1_output` |

Save & Run All. Điều kiện đạt là `complete=true`, `runs=24`, `checks=10824`, đủ 1.461 ngày mỗi run. So từng ô trước và sau chặn âm với NPZ C2, kiểm mặt nạ và quan trắc, sai số dự báo `atol=2e-5 mm/ngày`, `rtol=1e-6`. Ba điểm mỗi run đối chiếu thêm đường gọi một lưu vực, một ngày. GRU seed 111 là mô hình bật sẵn trên D3.

Nếu cổng wheel báo lệch phiên bản, đọc thông báo để xác định trường lệch trong Python, PyTorch, CUDA hoặc GPU. D1 đối chiếu trực tiếp với `wheels/build_info.json` của `01_0_MambaSSM_Build` Version 1. Trong **Settings → Session options → Environment**, chọn **Pin to original environment** nếu ảnh gốc của bản D1 là ảnh C2 cu128. Khi bản D1 bắt đầu trên ảnh mới hơn, dùng **Copy & Edit** từ Version 1 của `12_C2_Train_gru_L2_x1_lr1e-3_Seed111`. Notebook nguồn ghi ảnh Kaggle 31499, Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8 và hai Tesla T4. Giữ ảnh gốc, thay nội dung bản sao bằng D1 và kiểm lại môi trường trước khi gắn đủ Input. [Tài liệu môi trường notebook Kaggle](https://www.kaggle.com/docs/notebooks).

Gửi số Version và `D1_summary.json`, `D1_run_check.csv`, `D1_reproduction.csv`, `D1_examples.csv`, `environment.json`, `manifest.csv`. H/D dùng cùng lớp dự báo khi có checkpoint và NPZ. H cần chạy bằng `freeze-H`; D dùng `freeze-C`. Khi gắn thêm Output `17_D_LamaHCEExt_Compare` cùng các Output huấn luyện đại diện trong `D_demo_runs.csv` (hai hoặc ba cấu hình bước D đưa lên trang 1), D1 tự động đối chiếu toàn kỳ test 451 lưu vực × 1.461 ngày ở `seq_length=730` và xuất thêm `D1_D_run_check.csv`, `D1_D_reproduction.csv`, `D1_D_examples.csv` trong cùng `manifest.csv` mà không đổi 24 dòng C2 gốc.

Nếu bản D1 cũ dừng với lỗi `Config nằm ngoài giao thức suy luận C2/H/D`, cập nhật cell **Phần 2**, chạy lại cell đó, rồi chạy **Phần 3** trong phiên đã cài xong thư viện. NeuralHydrology đọc `additional_feature_files: None` từ config đã lưu thành danh sách rỗng. Kiểm tra thực hiện trên `Config` đã phân tích và vẫn từ chối đường dẫn bổ sung thật. Nếu kernel đã khởi động lại, chạy đủ ba phần theo thứ tự.

Nguồn triển khai [Pipeline Mục 9.6](../../../Document/01_Plan/02_Pipeline.md#96-đưa-mô-hình-vào-demo).

## Kết quả Kaggle Version 1

D1 hoàn tất 24 run, 451 lưu vực mỗi run và 1.461 ngày test. Cả 10.824 phép kiểm đạt, với sai số tuyệt đối lớn nhất bằng 0 mm/ngày cả trước và sau chặn âm. Phiên T4 dùng Python 3.13.15, PyTorch 2.11.0+cu128 và CUDA 12.8. C2 Compare Version 1 có SHA-256 manifest `50cb35873409a945a3f99bb41f6fb5a29553e2af2b6c387dcfd15b47ed094c08`. D2 nhận Output D1 đã lưu; SHA-256 của manifest D1 mà D2 kiểm là `484b1f8a03c418c88751a5b28ca7978b25c4d16b31658e3639348b0afb32f7c0`. [Kết quả D2 Version 1](../02_D2_LamaHCEExt_CPU/Archive/Version1/D2_summary.json).

## Kết quả Kaggle Version 2 (bổ sung bước D 730 bước)

Version 2 (32 Input trên Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, `freeze-C`) đối chiếu lại nguyên vẹn **24 run C2** (`10.824` phép kiểm trong `127,47` phút, `max_raw_abs_error = 0`, `max_clip_abs_error = 0` mm/ngày) và đối chiếu thêm **3 run đại diện bước D** (`1.353` phép kiểm trên `451` lưu vực × `1.461` ngày trong `16,77` phút, `D_max_raw_abs_error = 0`, `D_max_clip_abs_error = 0` mm/ngày; tổng thời gian `144,24` phút). Cả `8/8` tệp trong `manifest.csv` (`7c5b323403bb6e98c47109198f0a5983dc82722967b4e0f60e9a28e07313336c`) khớp SHA-256 và dung lượng; `D_compare_manifest_sha256` khớp `17_D_LamaHCEExt_Compare` Version 1 (`b1ff6006a08fd70d3ec4150902746966cba667ac9f9ebf5250962d7e9c9e9899`).

| Run D (`seq_length = 730`) | Epoch | Lưu vực | Ngày test | Thời gian T4 (s) | Kết quả |
|---|---:|---:|---:|---:|---|
| `SF_lstm_L2_x1_lr1e-3_D_seed333` | 28 | 451 | 1.461 | 229,2 | Đạt (`0` mm/ngày) |
| `SF_mamba_L2_x1_lr1e-3_D_seed333` | 20 | 451 | 1.461 | 337,3 | Đạt (`0` mm/ngày) |
| `SF_s4d_L4_x1_lr1e-3_D_seed111` | 19 | 451 | 1.461 | 440,0 | Đạt (`0` mm/ngày) |

[Tóm tắt Version 2](Archive/Version2/D1_summary.json) · [Kiểm run D](Archive/Version2/D1_D_run_check.csv) · [Kiểm run C2](Archive/Version2/D1_run_check.csv) · [Manifest](Archive/Version2/manifest.csv).
