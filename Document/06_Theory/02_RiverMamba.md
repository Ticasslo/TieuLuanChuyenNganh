# RiverMamba — Ghi chú đọc mã nguồn (tài liệu tham khảo)

> Tài liệu tổng hợp kết quả đọc repo chính thức của RiverMamba (Shams Eddin, Zhang, Kollet, Gall — NeurIPS 2025; DOI proceedings `10.52202/085713-4446`; arXiv 2505.22535; mã nguồn `github.com/HakamShams/RiverMamba_code`, đọc ngày 26/8/2026). Đề tài không chạy hay fine-tune RiverMamba (đề tài dùng 4 bộ dữ liệu benchmark theo lưu vực, xem `Document/01_Plan/01_OverallPlan.md`); tài liệu này dùng để tham khảo kiến trúc, hàm mất mát, quy ước dữ liệu và kinh nghiệm cài đặt `mamba-ssm` trên Colab/Kaggle. Thư mục `docs/` của repo chỉ chứa ảnh và poster, không có tài liệu văn bản bổ sung.

---

## 0. Cài đặt `mamba-ssm` trên Colab/Kaggle

- Cả Colab và Kaggle đều là môi trường tạm: Colab ngắt khi không hoạt động khoảng 90 phút hoặc tối đa khoảng 12 giờ liên tục; Kaggle tối đa 12 giờ mỗi phiên GPU, 30 giờ GPU mỗi tuần. Môi trường cài đặt mất sau mỗi phiên trên cả hai nền tảng.
- Cách xử lý: build `mamba-ssm`, `causal-conv1d` (và `flash-attn` nếu cần) **một lần**, lưu file `.whl` đã biên dịch vào nơi lưu trữ bền (Google Drive với Colab, Kaggle Dataset với Kaggle). Các phiên sau chỉ cần cài từ file wheel trong vài giây thay vì biên dịch lại từ mã nguồn.
- `mamba-ssm` cần GPU CUDA kiến trúc sm_75 trở lên (T4 chạy được, P100 không): `setup.py` của `state-spaces/mamba` và `causal-conv1d` chỉ biên dịch cho `compute_75` trở lên.
- Mã của dự án viết dạng `.py` chia `# %% Phần X`, mỗi cell dán trực tiếp vào Colab/Kaggle — thuận tiện so sánh thay đổi qua git.

## 1. Tổng quan repo

- **Tác giả mã:** Mohamad Hakam Shams Eddin, Yikui Zhang (GitHub `HakamShams`, `yikuizh`).
- **Đơn vị:** Computer Vision Group, Institute of Computer Science III, University of Bonn.
- **Giấy phép:** mã BSD-3-Clause; Mamba đi kèm Apache-2.0; Flash-Attention BSD-3-Clause.
- **Ngôn ngữ:** Jupyter Notebook 86,4%, Python 13,1%, Shell 0,5%.
- **Môi trường đã kiểm thử:** PyTorch 1.12.1 và 2.5.1, Python 3.10.0, Ubuntu 20.04.5 LTS, GPU NVIDIA A100/RTX 3090.
- **Mức độ sử dụng:** 20 sao, 2 fork (thời điểm đọc) — cộng đồng nhỏ, ít lỗi đã được người khác giải quyết sẵn.

## 2. Cài đặt môi trường

- Cài qua `requirements.txt` (pip venv hoặc conda); PyTorch phải cập nhật cho khớp driver GPU.
- Cài Mamba: `pip install mamba-ssm[causal-conv1d]`, thêm cờ `--no-build-isolation` nếu lỗi.
- Flash-Attention cần `ninja` hoạt động đúng (`ninja --version` trả mã thoát 0). Không có `ninja`, việc biên dịch có thể mất tới 2 giờ (chỉ dùng 1 lõi); có `ninja` thì khoảng 3–5 phút trên máy 64 lõi. Thời gian trên Colab/Kaggle (ít lõi hơn) chưa đo.
- **Bước thủ công sau khi cài (dễ bỏ sót):** chép `mamba_simple.py`, `mamba2_simple.py` trong `models/vim/` vào `<env>/lib/<python_version>/site-packages/mamba_ssm/modules/`, và `selective_scan_interface.py` vào `.../mamba_ssm/ops/`.

## 3. Cấu trúc mã nguồn

| Đường dẫn | Nội dung |
|---|---|
| `config.py` | Tệp cấu hình duy nhất cho huấn luyện, kiểm thử và mô hình (tên tệp cấu hình là tên thí nghiệm) |
| `dataset/RiverMamba_dataset.py` | Nạp và tiền xử lý dữ liệu (Mục 5) |
| `train.py` | Huấn luyện 1 node, nhiều GPU kiểu DataParallel |
| `train_multinode.py` | Huấn luyện nhiều node bằng PyTorch DDP |
| `inference_aifas.py` | Suy luận trên tập điểm AIFAS (1.529.667 điểm) |
| `inference_full_map.py` | Suy luận trên toàn bộ lưới đất liền (6.221.926 điểm, lưới 3000×7200 ở 0,05°) |
| `models/build.py` | Dựng mô hình |
| `models/encoder/`, `models/decoder/` | Mỗi thư mục có 3 lựa chọn backbone: `FlashTransformer.py`, `Mamba.py`, `Mamba2.py` |
| `models/head/MLP.py` | Đầu ra MLP |
| `models/loan.py` | Module LOAN (lý thuyết ở `01_ArchitectureTheory.md` Mục 4.8) |
| `models/loss.py` | Hàm mất mát (log1p có dấu, trọng số return period và lead time) |
| `models/mha.py` | Multi-head attention |
| `models/vim/` | Mã Vision Mamba gốc, cần chép vào site-packages (Mục 2) |
| `preprocessing_grdc/` | Tiền xử lý dữ liệu trạm GRDC |
| `serialization/` | Dựng space-filling curve (Mục 7) |
| `scripts/` | Các tệp `.sh` tải và giải nén dữ liệu (Mục 9) |
| `vis/` | 7 script trực quan hóa cho từng loại dữ liệu (CPC, ERA5-Land, GloFAS reanalysis, GRDC, HRES, Reforecast, Static) |
| `lstm_repo/` | Baseline Encoder-Decoder LSTM (kiểu mô hình vận hành của Google) |

**Backbone hỗ trợ:** Transformer + Flash-Attention, Vision Mamba, Vision Mamba2. **Baseline có sẵn:** chỉ Encoder-Decoder LSTM; repo không có baseline GRU hay Transformer thuần.

## 4. Cấu hình và siêu tham số (`config.py`)

- **Đường dẫn dữ liệu:** 6 biến `root_glofas_reanalysis`, `root_era5_land_reanalysis`, `root_static`, `root_hres_forecast`, `root_cpc`, `root_obs` (GRDC).
- **Chọn backbone:** `encoder="Mamba"`, `decoder="Mamba"`, `head="MLP"` (mặc định); đổi sang `"FlashTransformer"` hoặc `"Mamba2"` chỉ cần sửa cấu hình.
- **Encoder:** `en_embed_dim=[192,192,192]`, `en_depths=[2,2,2]` (3 tầng, mỗi tầng 2 khối). Phần SSM: `en_d_state=[1,1,1]` (nhỏ hơn nhiều so với giá trị 16 thường dùng), `en_d_conv=[3,3,3]`, `en_expand=[1,1,1]`, `en_dt_min=0.001`, `en_dt_max=0.1`, `en_bi_ssm=True` (bidirectional, xem `01_ArchitectureTheory.md` Mục 4.7). Dropout mặc định 0.
- **Decoder:** 1 tầng (`de_embed_dim=[192]`, `de_depths=[1]`).
- **Embedding riêng từng nguồn:** `en_embed_glofas=48`, `en_embed_era5=128`, `en_embed_cpc=16`, `de_embed_hres=64` — mỗi nhóm đầu vào qua lớp embedding riêng trước khi ghép.
- **Huấn luyện:** `batch_size_train=1`, trong đó mỗi mẫu đã gồm `n_points=254945` điểm không gian (bản thân một mẫu đóng vai trò như một batch lớn). `n_epochs=100`, Adam, `lr=1e-3`, `weight_decay=1e-6`, `max_norm=10` (cắt gradient). Lịch học cosine, `lr_min=1e-5`, `lr_decay_step=20`, `lr_decay_rate=0.9`, không warmup.
- **Cửa sổ thời gian:** `delta_t=4` (4 ngày quá khứ làm hindcast), `delta_t_f=7` (dự báo 7 ngày).

**Phiên bản thư viện (`requirements.txt`, 67 gói, cố định phiên bản):** `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `torchaudio==2.4.1+cu121` (bản build cho CUDA 12.1); `mamba-ssm==2.2.4`, `causal-conv1d==1.5.0.post8`, `flash-attn==2.7.0.post2`. Runtime Colab/Kaggle có thể dùng phiên bản CUDA khác 12.1, nên cần kiểm tra `nvidia-smi`/`nvcc --version` trước khi cài và chọn bản torch tương ứng.

## 5. Nạp và tiền xử lý dữ liệu (`dataset/RiverMamba_dataset.py`)

- Đọc NetCDF qua `xarray` (engine `netcdf4`).
- **Tiền xử lý:**
  - Log1p có dấu `log1p(|x|) * sign(x)`, chỉ áp dụng cho các biến khai báo trong `variables_*_log1p`.
  - Chuẩn hóa Z-score `(x − mean) / (std + 1e-6)` cho biến động.
  - Biến tĩnh chuẩn hóa min-max về `[-1, 1]`.
  - Trọng số lead time `exp(|t − max_t| * alpha)`.
- **Mục tiêu được mã hóa dạng chênh lệch (delta):** dataset trừ giá trị hiện tại khỏi các giá trị dự báo trước khi đưa vào mô hình.
- `__getitem__` trả về dict gồm `glofas`, `era5`, `hres_forecast`, `cpc` (dạng `[T, P, V]`), biến tĩnh `[P, V]`, `glofas_target`/`obs_target` (dạng `[lead_time, P, 1]`) và metadata (`curves`, `weight`, `file_name`, `random_indices`).
- Tham số `delta_t_f` của lớp `RiverMamba_dataset` có giá trị mặc định 15, nhưng `config.py` truyền vào 7 nên giá trị thực tế là 7.

## 6. Suy luận (`inference_aifas.py`)

- Chạy `python inference_aifas.py`, không nhận tham số dòng lệnh; mọi cấu hình lấy từ `config.py` (`config_file.read_arguments(train=False, print=True, save=False)`).
- Checkpoint mặc định tại `./RiverMamba_aifas_reanalysis.pth` (gán vào `config.pretrained_model`); dùng checkpoint khác phải đổi tên tệp hoặc sửa đường dẫn trong cấu hình.
- Đầu ra: NetCDF, biến `dis24` (tên biến lưu lượng của GloFAS), chiều `[time, x]`, lưu trong `./inference/`, giá trị cắt về không âm, kiểu float32.
- Cố định seed (`utils.fix_seed()`), chọn GPU qua `config.gpu_id`, hỗ trợ nhiều GPU bằng `DataParallel`.
- Dữ liệu được chia dọc theo space-filling curve thành các đoạn bằng `n_points` lúc huấn luyện (với `n_points=254945`, tập AIFAS 1.529.667 điểm chia thành 6 đoạn, đoạn cuối có thể lẻ). Mặc định chỉ xuất kết quả tại vị trí có trạm GRDC.

## 7. Space-filling curve (`serialization/generate_curves.py`)

5 loại curve (lý thuyết ở `01_ArchitectureTheory.md` Mục 4.9):
- **Gilbert** — Hilbert curve tổng quát cho vùng chữ nhật có kích thước không phải lũy thừa của 2 (dùng implementation numpy từ `jakubcerveny/gilbert`).
- **Sweep_h / Sweep_v** — quét xoắn ốc cầu (loxodrome) theo chiều ngang/dọc.
- **Zigzag_h / Zigzag_v** — như Sweep nhưng bảo đảm các điểm liền kề trên curve cũng liền kề trong không gian.

## 8. Checkpoint và reforecast công bố

**Tập AIFAS (1.529.667 điểm):**

| Tên trong README | Dữ liệu mục tiêu | Giai đoạn huấn luyện |
|---|---|---|
| RiverMamba_aifas_reanalysis | GloFAS reanalysis | 1979–2018 |
| RiverMamba_aifas_reanalysis | GRDC observations | 1979–2018 |

README ghi cùng một tên cho hai checkpoint có mục tiêu khác nhau; khi tải cần kiểm tra đường dẫn tải thực tế thay vì dựa vào tên hiển thị.

**Toàn lưới (6.221.926 điểm):** 1 checkpoint, mục tiêu GloFAS reanalysis, huấn luyện 1979–2023.

**Reforecast (dùng làm baseline):**

| Tên | Mô hình | Số trạm | Mục tiêu | Huấn luyện | Reforecast |
|---|---|---|---|---|---|
| LSTM_glofas_reanalysis | ed-LSTM | 3.366 | GloFAS | 1979–2018 | 2019–2024 |
| LSTM_grdc_obs | ed-LSTM | 3.366 | GRDC | 1979–2018 | 2019–2024 |
| RiverMamba_glofas_reanalysis | RiverMamba | 3.366 | GloFAS | 1979–2018 | 2019–2024 |
| RiverMamba_grdc_obs | RiverMamba | 3.366 | GRDC | 1979–2018 | 2019–2024 |

Mỗi tệp NetCDF reforecast có 7 bước thời gian (lead time 7 ngày), tên tệp là ngày phát hành dự báo lúc 00:00 UTC; thông tin trạm tra tại `GRDC_Meta`, "index AIFAS" là vị trí trên trục x. Reforecast toàn lưới: `RiverMamba_glofas_reanalysis_full_map`, huấn luyện 1979–2023, reforecast năm 2024. Tất cả mô hình là tất định (deterministic), không có nowcasting; có bản sao trên Hugging Face.

## 9. Bộ dữ liệu gốc

- Nguồn: `https://doi.org/10.60507/FK2/T8QYWE` (khoảng 10 TB sau giải nén) — CPC 16 GB, ERA5-Land khoảng 7,6 TB, ECMWF-HRES khoảng 1,9 TB, GloFAS static khoảng 46 GB, GloFAS reanalysis khoảng 600 GB.
- Nén `.7z` (cần `p7zip-full`); tệp lớn chia theo năm, giải nén phần `.001` thì `7z` tự nối các phần tiếp theo.
- Script `scripts/download_*.sh` dùng `wget --continue` từ `bonndata.uni-bonn.de`, chỉ tải các tệp toàn cầu cố định, không giới hạn theo vùng.
- **Quy ước nhãn thời gian (right-labeled):** theo README, lưu lượng gán ngày 01/01/2024 là trung bình của ngày 31/12/2023. Khi ghép đầu vào và mục tiêu cần tính đến quy ước này để tránh lệch 1 ngày.
- **Độ trễ công bố:** RiverMamba lùi đầu vào theo độ trễ công bố thật của nguồn (GloFAS/ERA5 t−1, CPC t−2) để tránh rò rỉ tương lai.

## 10. Hướng dẫn trích dẫn

BibTeX trong README repo dùng bản arXiv (`arXiv:2505.22535`, 2025). Bản chính thức trong NeurIPS 2025 proceedings có DOI `10.52202/085713-4446` (`proceedings.neurips.cc`); ưu tiên DOI proceedings khi trích dẫn.

## 11. Các điểm tham khảo cho đề tài

- LOAN có trong mã (`models/loan.py`): `(X − μ)/σ + GELU(Linear(X_static))` — phù hợp với thuộc tính tĩnh lưu vực của các bộ dữ liệu benchmark.
- Hàm mất mát: biến đổi `sign(x)·log1p(|x|)` trên chênh lệch lưu lượng trước MSE/L1, trọng số theo return period (ưu tiên lũ hiếm) nhân trọng số lead time `exp(|t − max_t|·alpha)`; đầu ra là chênh lệch so với hiện tại, 7 đầu MLP cho 7 lead time.
- Bidirectional Mamba trong RiverMamba quét chuỗi **không gian** (space-filling curve), không phải chuỗi thời gian.
- Checkpoint không dùng lại được cho dữ liệu theo lưu vực: lớp đầu vào cố định theo 136 biến lưới 0,05° (4 GloFAS + 32 ERA5-Land + 1 CPC + 99 tĩnh).
- README xác nhận huấn luyện trên dữ liệu khác bài báo cần tự tinh chỉnh siêu tham số và trọng số hàm mất mát.
- Issue duy nhất của repo (#2, "Issues with HRES data download", mở 13/11/2025) vẫn chưa được trả lời tại thời điểm đọc.
