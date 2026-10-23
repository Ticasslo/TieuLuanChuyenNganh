# Bước 0 — Dựng `mamba-ssm` cho Kaggle T4

Tài liệu đi cùng [mã Python](01_0_MambaSSM_Build.py) của notebook Kaggle `01_0_MambaSSM_Build`. Notebook biên dịch hai gói `causal-conv1d` và `mamba-ssm` cho đúng môi trường Kaggle, kiểm kết quả tính bằng bộ test chính thức của tác giả, rồi lưu hai tệp wheel vào Output để các notebook huấn luyện sau gắn làm Input và cài trong vài giây.

## Mục lục

- [1. Vì sao phải tự dựng](#1)
- [2. Cài đặt notebook](#2)
- [3. Các Phần và kết quả](#3)
- [4. Kết luận cho các bước sau](#4)
- [5. Cách dùng ở notebook khác](#5)
- [6. Nguồn](#6)

<a id="1"></a>
## 1. Vì sao phải tự dựng

Phép quét chọn lọc (selective scan) của Mamba và tích chập nhân quả đi kèm được viết bằng CUDA C++. Mã này phải biên dịch thành thư viện `.so` khớp đúng phiên bản torch, CUDA, Python và thế hệ GPU; wheel lệch phiên bản torch báo lỗi `undefined symbol` khi nạp (state-spaces/mamba, issue 217, 652, 760, 764).

Môi trường Kaggle (ảnh dựng ngày 01/10/2026): Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8, `nvcc` 12.8, GPU Tesla T4 (compute capability 7.5), 4 CPU, RAM 31,3 GB. Bản phát hành `mamba-ssm` 2.3.2.post1 và `causal-conv1d` 1.6.2.post1 (9/5/2026) chỉ có wheel dựng sẵn cho torch 2.6–2.10; `causal-conv1d` 1.7.0 (20/8/2026) cũng dừng ở torch 2.10. Không có wheel cho torch 2.11 nên phải tự biên dịch. Wheel Kaggle Dataset `nctuan/mamba-ssm` dựng cho Python 3.12 nên không dùng được.

`setup.py` của cả hai repo gán cứng danh sách kiến trúc GPU (sm_75, 80, 87, 90, và với CUDA 12.8 thêm 100, 120) và không có biến môi trường để giới hạn. Notebook chèn một dòng gán lại danh sách thành chỉ `sm_75` ngay sau khối gán đó, nên chỉ biên dịch cho T4.

<a id="2"></a>
## 2. Cài đặt notebook

| Mục | Giá trị | Lý do |
|---|---|---|
| Accelerator | GPU T4 x2 (dùng 1) | Cần torch CUDA và `nvcc`; kiểm trên đúng GPU huấn luyện |
| Internet | On | Tải mã nguồn GitHub, cài `mambapy` |
| Persistence | Files only | Giữ wheel trong `/kaggle/working` khi phiên tương tác khởi động lại |
| Environment | Pin to original environment | Giữ torch 2.11 cho notebook này; notebook khác vẫn có thể nhận ảnh mới, nên cell cài ở Phần 9 tự kiểm phiên bản torch |
| Phiên bản gói | `causal-conv1d` v1.6.2.post1, `mamba-ssm` v2.3.2.post1 | Cùng đợt phát hành; `mamba-ssm` gọi `causal_conv1d_fwd_function`, `causal_conv1d_bwd_function`, chữ ký giống nhau ở 1.6.2.post1 và 1.7.0 |

<a id="3"></a>
## 3. Các Phần và kết quả

Kết quả dưới đây lấy từ bản lưu chính thức (Save & Run All, phiên sạch, 23/10/2026): chạy hết Phần 1–9 trong 827 giây, mọi Phần đạt. Phiên tương tác chạy trước đó cùng ngày cho cùng kết luận ở mọi Phần; số đo thời gian và mức lệch đạo hàm chênh nhau chút ít giữa hai lần chạy.

### Phần 1. Kiểm môi trường

Đủ cả năm điều kiện biên dịch: có GPU compute capability ≥ 7.5, có `nvcc`, `nvcc` cùng phiên bản CUDA chính với torch (12), đủ thư viện (`triton` 3.6.0, `einops` 0.8.2, `ninja` 1.13.2, `transformers` 5.16.1), nạp được `wheel.bdist_wheel`.

### Phần 2. Tìm wheel dựng sẵn

Tên wheel dựng theo đúng công thức `get_wheel_url()` trong `setup.py` của hai repo: `cu12` (mọi CUDA 12.x), torch lấy major.minor, cờ cxx11 ABI, `cp313`, `linux_x86_64`. Wheel cần (`…+cu12torch2.11cxx11abiTRUE-cp313…`) không có trên release; release có wheel cp313 cu12 cho torch 2.6, 2.7, 2.8, 2.9, 2.10.

### Phần 3–4. Biên dịch

| Gói | Thời gian biên dịch | Dung lượng | Kiến trúc trong `.so` (`cuobjdump`) | Log có kiến trúc khác |
|---|---|---|---|---|
| `causal-conv1d` | 5,9 phút | 29,3 MB | sm_75 | không |
| `mamba-ssm` | 6,2 phút | 47,7 MB | sm_75 | không |

`mamba-ssm` có một extension `selective_scan_cuda` gồm 1 tệp `.cpp` và 9 tệp `.cu`. Biên dịch bằng `pip wheel --no-build-isolation` để dùng torch có sẵn của ảnh, `MAX_JOBS` = 4.

### Phần 5. Cài và nạp

Cài bằng `--no-deps`: pip không đụng torch và không kéo `tilelang`, `quack-kernels` (chỉ Mamba-3 dùng; mọi chỗ gọi đều nằm trong khối `try`). Torch trước và sau khi cài đều là 2.11.0+cu128. Nạp được `causal_conv1d_cuda`, `selective_scan_cuda`, `causal_conv1d`, `mamba_ssm` và `from mamba_ssm import Mamba`. Tệp `wheels/build_info.json` ghi Python, torch, CUDA, cờ ABI, `nvcc`, kiến trúc, tag hai repo và sha256 của hai wheel. Hai lần biên dịch cùng mã nguồn cho wheel khác sha256 (phiên tương tác và bản lưu), nên cell cài chỉ so wheel với `build_info.json` nằm cùng thư mục, không dùng lẫn wheel của hai lần dựng.

### Phần 6. Bộ test chính thức của tác giả

| Nhóm | Số test | Đạt | Lỗi | Bỏ qua | Thời gian |
|---|---|---|---|---|---|
| `mamba` `tests/ops/test_selective_scan.py` (`selective_scan_fn`, `mamba_inner_fn` so với bản PyTorch thuần, xuôi và ngược) | 20 | 20 | 0 | 0 | 17 giây |
| `causal-conv1d` `test_causal_conv1d`, fp32, xuôi và ngược | 3456 | 2160 | 0 | 1296 | 28 giây |
| `causal-conv1d` ba test tính tất định | 6 | 6 | 0 | 0 | 4 giây |

1296 test bỏ qua do chính tác giả đặt điều kiện ("Only channel_last support initial_states or return_final_states"). Ngưỡng fp32 của tác giả cho selective scan là rtol 6e-4, atol 2e-3. Test `test_causal_conv1d_bwd_default_mode_is_not_reproducible` đạt (không phải xfail): trên T4, đạo hàm ngược của `causal-conv1d` ở chế độ mặc định khác nhau giữa các lần chạy; bật `torch.use_deterministic_algorithms(True)` thì lặp lại y hệt.

### Phần 7. Khối Mamba ở kích thước của đề tài

Kích thước lấy theo Sequential Forecast LSTM của BiasCast: lô 256, 365 bước (364 ngày quá khứ và ngày t), 128 chiều.

| Kiểm | Kết quả |
|---|---|
| Đường nhanh (`mamba_inner_fn`) so với đường chậm (`causal_conv1d_fn` + `selective_scan_fn`), cùng trọng số | Đầu ra trùng; đạo hàm lệch tuyệt đối lớn nhất 2,7e-5 (`conv1d.bias`), sai số tương đối dưới 1e-6 |
| Không rò rỉ tương lai: đổi đầu vào từ bước 300 | Đầu ra trước bước 300 lệch đúng 0; từ bước 300 lệch 0,40 (cả hai đường) |
| Tính tất định, 3 lần chạy lại xuôi + ngược | Đầu ra trùng; đạo hàm lệch tới 3,1e-5 ở chế độ mặc định (5 tham số, lớn nhất ở `conv1d.bias`) và 1,0e-5 khi bật chế độ tất định (3 tham số, lớn nhất ở `D`) |
| Thời gian một lượt xuôi + ngược, trung vị 10 lần | Mamba đường nhanh 76,8 ms (116.480 tham số, bộ nhớ đỉnh 1,88 GB); đường chậm 80,0 ms (1,68 GB); `nn.LSTM` 128 → 128 33,7 ms (132.096 tham số, 2,13 GB) |

Đạo hàm vẫn lệch khi bật chế độ tất định vì nhân lan truyền ngược của selective scan cộng đạo hàm `dD`, `ddelta_bias`, `dA`, `dB`, `dC` bằng `gpuAtomicAdd` (`csrc/selective_scan/selective_scan_bwd_kernel.cuh` dòng 475–491 ở tag v2.3.2.post1); phép cộng nguyên tử không cố định thứ tự cộng, và giao diện Python không có tùy chọn tất định. Số đo thời gian chỉ để tham khảo, số đo chính thức làm ở bước đo thời gian một epoch mỗi lõi.

### Phần 8. `mambapy` dùng lại trọng số

`mambapy` 1.2.0 (PyPI, 31/7/2024, PyTorch thuần) có `MambaBlock` cùng tên tham số và cùng giá trị mặc định với `mamba_ssm.Mamba` (d_state 16, expand 2, d_conv 4, dt_rank = ⌈d_model/16⌉, không bias ở `in_proj`/`out_proj`, có bias ở `conv1d`). Nạp `state_dict` của khối Mamba ở Phần 7 với `strict=True`: không thiếu, không thừa khóa. Đầu ra trùng `mamba_ssm` cả trên GPU (32 mẫu) và CPU (4 mẫu), sai số bằng 0 ở sáu chữ số thập phân. Trên CPU 4 lõi của Kaggle, 4 mẫu × 365 bước chạy xuôi mất 0,20 giây.

### Phần 9. Cell cài dùng lại

Cell tự đủ, không dùng biến của Phần 1–8: tìm `build_info.json` dưới `/kaggle/input` (rồi `/kaggle/working/wheels`), dừng nếu Python, torch hoặc CUDA khác lúc dựng hoặc GPU không phải T4, kiểm sha256 hai wheel, cài `--no-deps`, kiểm torch trên đĩa không đổi, chạy thử một khối Mamba xuôi + ngược. Chạy trong phiên dựng (wheel ở `/kaggle/working/wheels`): đạt.

<a id="4"></a>
## 4. Kết luận cho các bước sau

- Lõi Mamba chạy được trên Kaggle T4 với torch 2.11; nhân CUDA tính đúng theo bộ test của tác giả.
- Huấn luyện Mamba trên GPU không lặp lại y hệt từng bit kể cả khi cùng hạt giống và bật chế độ tất định (mức lệch đạo hàm khoảng 1e-5). Khi đọc nhiễu giữa các hạt giống ở A4 và bước C, phần lệch này nằm trong nhiễu của lõi Mamba.
- Ở kích thước của đề tài, một lượt xuôi + ngược của khối Mamba chậm hơn `nn.LSTM` (cuDNN) khoảng 2,3 lần và dùng ít bộ nhớ hơn.
- Trọng số huấn luyện bằng `mamba_ssm` nạp thẳng vào `mambapy` cho demo chạy CPU (D2–D4, `03_Pipeline.md` Mục 9.6) và cho cùng kết quả.

<a id="5"></a>
## 5. Cách dùng ở notebook khác

Gắn Output của notebook `01_0_MambaSSM_Build` làm Input (Add Input → Notebook Output), chép nguyên cell Phần 9 làm cell đầu. Khi Kaggle đổi ảnh làm torch khác 2.11.0+cu128, cell dừng và báo phải dựng lại bằng notebook này.

<a id="6"></a>
## 6. Nguồn

- `state-spaces/mamba` tag v2.3.2.post1: `setup.py`, `mamba_ssm/modules/mamba_simple.py`, `mamba_ssm/ops/selective_scan_interface.py`, `csrc/selective_scan/selective_scan_bwd_kernel.cuh`, `tests/ops/test_selective_scan.py`. https://github.com/state-spaces/mamba
- `Dao-AILab/causal-conv1d` tag v1.6.2.post1: `setup.py`, `causal_conv1d/cpp_functions.py`, `tests/test_causal_conv1d.py`. https://github.com/Dao-AILab/causal-conv1d
- Danh sách wheel của các bản phát hành: GitHub API `repos/{repo}/releases`.
- PyTorch v2.11.0 `torch/utils/cpp_extension.py` (`_get_cuda_arch_flags` bỏ cờ kiến trúc mặc định khi tham số biên dịch đã có `arch`; ninja chạy với `-v`). https://github.com/pytorch/pytorch
- `mambapy` 1.2.0, `mambapy/mamba.py`, `mambapy/pscan.py`. https://pypi.org/project/mambapy/
- Kaggle Notebooks documentation (Save & Run All chạy phiên sạch). https://www.kaggle.com/docs/notebooks
