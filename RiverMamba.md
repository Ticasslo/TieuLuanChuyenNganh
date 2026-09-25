# RiverMamba — Hiểu code repo trước khi chạy (chuẩn bị cho Google Colab)

> File này chỉ ghi phần **hiểu repo/code có gì**, chưa viết code Colab thật. Đọc kỹ toàn bộ README chính thức (github.com/HakamShams/RiverMamba_code, đọc ngày 26/8/2026) + kiểm tra thư mục `docs/` (chỉ chứa ảnh/poster, không có tài liệu text thêm). Phần code Colab thật sẽ viết ở file riêng `RiverMamba_code.py` khi bắt đầu chạy, chia `# %% Cell N`, mỗi lần đúng 1 cell.

---

## 0. Quyết định nền tảng + format code (chốt 26/8/2026)

**Colab vẫn là chính, Kaggle dự phòng** — không đổi vì lo "mất cài đặt": cả 2 nền tảng đều ephemeral như nhau (Colab ~90 phút không hoạt động hoặc tối đa ~12h liên tục; Kaggle tối đa 9h/phiên GPU), không phải vấn đề riêng của Colab. Cách xử lý đúng cho cả 2: **build `mamba-ssm`/`causal-conv1d`/`flash-attn` 1 lần, lưu file `.whl` đã compile vào nơi lưu trữ bền** (Google Drive cho Colab, Kaggle Dataset cho Kaggle) — phiên sau chỉ `pip install <file>.whl` (vài giây) thay vì build lại từ source (flash-attn có thể mất tới 2 giờ nếu ninja không chạy đúng). Vì vậy **Cell đầu tiên của `RiverMamba_code.py` phải gồm cả bước lưu wheel vào Drive**, không chỉ cài rồi thôi.

**Format code:** `.py` với `# %% Cell N` (theo đúng quy ước đã chốt toàn dự án) — **không dùng `.ipynb`** — lý do: git diff được, và mỗi cell copy dán trực tiếp vào 1 cell thật trên Colab, đúng quy tắc "1 cell/lần, đợi xác nhận".

## 1. Tổng quan repo

- **Tác giả code:** Mohamad Hakam Shams Eddin, Yikui Zhang (2 contributor GitHub: `HakamShams`, `yikuizh`)
- **Đơn vị:** Computer Vision Group, Institute of Computer Science III, University of Bonn
- **License code:** BSD-3-Clause. Mamba đi kèm license Apache-2.0, Flash-Attention license BSD-3-Clause (khác nhau theo từng phần phụ thuộc)
- **Ngôn ngữ:** Python 13.1%, Shell 0.5%, còn lại 86.4% là Jupyter Notebook — ⚠️ đáng chú ý vì phần lớn code là notebook, khớp thuận lợi với việc chạy trên Colab (không cần viết lại từ script thuần)
- Đã test trên PyTorch (1.12.1 & 2.5.1), Python 3.10.0, Ubuntu 20.04.5 LTS, GPU NVIDIA A100/RTX 3090
- 20 stars, 2 forks — repo nhỏ, ít người dùng, nghĩa là ít khả năng có issue đã được người khác giải quyết sẵn trên GitHub Issues nếu gặp lỗi lạ

## 2. Cài đặt môi trường — vài điểm dễ vướng

- Cài qua `requirements.txt` (pip venv hoặc conda) — PyTorch phải tự cập nhật lại cho khớp driver GPU đang dùng (không cố định version)
- Cài Mamba: `pip install mamba-ssm[causal-conv1d]`, thêm cờ `--no-build-isolation` nếu lỗi
- Cài Flash-Attention (`pip install flash-attn --no-build-isolation`) cần `ninja` hoạt động đúng (`ninja --version` rồi `echo $?` phải ra 0) — nếu không, compile có thể mất tới **2 giờ** (chỉ dùng 1 core); có ninja đúng thì chỉ 3-5 phút trên máy 64-core. ⚠️ Trên Colab free (thường ít core hơn nhiều), thời gian compile thực tế cần thử mới biết, có thể lâu hơn ví dụ 64-core nêu trên.
- **Vision Mamba cần bước copy thủ công sau khi cài xong:** copy các file trong thư mục `models/vim/` (`mamba_simple.py`, `mamba2_simple.py`) vào `<env>/lib/<python_version>/site-packages/mamba_ssm/modules/`, và `selective_scan_interface.py` vào `.../mamba_ssm/ops/` — đây là bước **dễ bị bỏ sót** vì không tự động, phải tự nhớ làm sau `pip install`.

## 3. Cấu trúc code — bản đồ thư mục

```
config.py                  — 1 file config duy nhất cho train/test/model (tên file config = tên experiment)
dataset/RiverMamba_dataset.py
train.py                   — train 1 node, nhiều GPU kiểu DataParallel (đơn giản)
train_multinode.py         — train nhiều node/nhiều GPU, dùng PyTorch DDP (phức tạp hơn, không cần cho tiểu luận)
inference_aifas.py         — suy luận trên tập điểm AIFAS đã lọc (1.529.667 điểm)
inference_full_map.py      — suy luận trên toàn bộ lưới đất liền (6.221.926 điểm, 3000×7200 @0.05°)
models/
  build.py
  encoder/, decoder/       — mỗi thư mục có 3 lựa chọn: FlashTransformer.py, Mamba.py, Mamba2.py (đổi backbone qua config)
  head/MLP.py
  loan.py                  — module LOAN — khớp đúng phần đã học lý thuyết ở LyThuyetCauTruc.md Mục 4.6, xác nhận có thật trong code, không phải chỉ lý thuyết suông
  loss.py                  — hàm loss thật (log1p có dấu + trọng số return period/lead time, đã ghi trong CLAUDE.md)
  mha.py
  vim/                     — code Vision Mamba gốc, cần copy ra ngoài site-packages (Mục 2)
preprocessing_grdc/        — tiền xử lý dữ liệu GRDC (điểm trạm thật, liên quan tới bước fine-tune 3 của tiểu luận)
serialization/             — dựng space-filling curve (Mục 5 bên dưới)
scripts/                   — các file .sh tải/giải nén dữ liệu (Mục 6)
vis/                       — 7 script visualize riêng cho từng loại dữ liệu (CPC, ERA5-Land, GloFAS reanalysis, GRDC obs, HRES, Reforecast, Static)
lstm_repo/                 — code baseline Encoder-Decoder LSTM (mô hình gốc kiểu Google operational model)
```

**Backbone hỗ trợ (chọn qua config):** Transformer+Flash-Attention, Vision Mamba, Vision Mamba2.
**Baseline có sẵn:** chỉ 1 — Encoder-Decoder LSTM. ⚠️ Không có sẵn baseline GRU/Transformer thuần trong repo dù đề tài dự định so sánh với cả 3 (LSTM/GRU/Transformer) — cần tự viết thêm GRU/Transformer thuần nếu muốn so sánh đủ, không có sẵn.

## 3b. `config.py` — chi tiết hyperparameter thật (đọc trực tiếp từ code, 26/8/2026)

**Đường dẫn dữ liệu:** `root_glofas_reanalysis`, `root_era5_land_reanalysis`, `root_static`, `root_hres_forecast`, `root_cpc`, `root_obs` (GRDC) — 6 biến riêng, đúng khớp 5+1 nhóm dữ liệu đã biết.

**Chọn backbone qua config** (không phải sửa code): `encoder="Mamba"`, `decoder="Mamba"` (mặc định), `head="MLP"` — đổi sang `"FlashTransformer"` hoặc `"Mamba2"` chỉ cần đổi tên trong config, không cần sửa logic.

**Encoder:** `en_embed_dim=[192,192,192]`, `en_depths=[2,2,2]` (3 tầng, mỗi tầng 2 khối) — riêng phần SSM: `en_d_state=[1,1,1]` (nhỏ bất ngờ, thường Mamba khác dùng d_state=16), `en_d_conv=[3,3,3]`, `en_expand=[1,1,1]`, `en_dt_min=0.001`/`en_dt_max=0.1`, `en_bi_ssm=True` (bidirectional, khớp LyThuyetCauTruc.md Mục 4.7). Dropout mặc định = 0.

**Decoder:** nhẹ hơn hẳn encoder — chỉ 1 tầng (`de_embed_dim=[192]`, `de_depths=[1]`).

**Embedding riêng từng nguồn dữ liệu trước khi ghép:** `en_embed_glofas=48`, `en_embed_era5=128`, `en_embed_cpc=16`, `de_embed_hres=64` — xác nhận mỗi nhóm input có lớp embedding riêng, không nối thẳng raw values.

**Training:** `batch_size_train=1` ⚠️ — batch size chỉ bằng 1, vì mỗi "mẫu" đã là `n_points=254945` điểm cùng lúc (bản thân 1 mẫu đã rất lớn, đóng vai trò như batch) — cần hiểu đúng để ước lượng VRAM, không phải batch nhỏ theo nghĩa thường. `n_epochs=100`, optimizer Adam, `lr=1e-3`, `weight_decay=0.000001`, `max_norm=10` (gradient clipping). LR scheduler dạng cosine, `lr_min=1e-5`, `lr_decay_step=20`, `lr_decay_rate=0.9`, không warmup.

**Cửa sổ thời gian dữ liệu:** `delta_t=4` (nhìn lại 4 ngày quá khứ làm "hindcast"), `delta_t_f=7` (dự báo xa 7 ngày) — khớp đúng "7 ngày lead time" đã biết, thêm chi tiết mới là cửa sổ nhìn lại chỉ 4 ngày, không phải dài hơn.

## 3c. `requirements.txt` — version cụ thể, có rủi ro tương thích Colab

67 package, pin version chính xác. Đáng chú ý nhất:
- `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `torchaudio==2.4.1+cu121` — build riêng cho **CUDA 12.1** (hậu tố `+cu121`)
- `mamba-ssm==2.2.4`, `causal-conv1d==1.5.0.post8`, `flash-attn==2.7.0.post2` — pin rất chặt, không phải "bản mới nhất"

⚠️ **Rủi ro cần kiểm tra khi thật sự setup Colab:** Colab free tier thường có sẵn 1 version CUDA/PyTorch cố định theo runtime hiện tại, có thể khác 12.1 — nếu lệch, cài `torch==2.4.1+cu121` trực tiếp có thể xung đột driver GPU của Colab. Cần kiểm tra `nvidia-smi`/`nvcc --version` trên Colab trước khi cài, có thể phải đổi sang bản torch khác hoặc cài lại CUDA toolkit đúng bản.

## 3d. Issue đã biết, chưa giải quyết trên GitHub (đọc 26/8/2026)

Repo chỉ có **đúng 1 issue**, còn mở, chưa ai trả lời:
- **Issue #2 — "Issues with HRES data download"**, mở bởi `ZY-LIi` ngày 13/11/2025, vẫn **open**, không có issue đã đóng nào để tham khảo cách xử lý.

⚠️ Vì tiểu luận cũng cần HRES sau này (dự báo thật/deploy — không phải core train), đây là rủi ro đã biết trước: nếu gặp lỗi tải HRES, không có sẵn thread GitHub nào để tham khảo cách người khác đã giải quyết — phải tự debug từ đầu hoặc hỏi thẳng tác giả.

## 4. Training — lưu ý quan trọng cho tiểu luận

> *"Training on real-world different than the one in the paper, requires fine-tuning for the hyper-parameters and weighting."* — README tự xác nhận: **fine-tune cho vùng mới (Vu Gia-Thu Bồn) không phải chỉ chạy lại script, cần tự chỉnh lại hyperparameter và trọng số loss** — không có công thức sẵn cho từng vùng cụ thể, phải tự thử nghiệm.

## 4b. `dataset/RiverMamba_dataset.py` — cách load/tiền xử lý dữ liệu thật (đọc trực tiếp, 26/8/2026)

- Đọc **NetCDF thuần** qua `xarray` (engine `netcdf4`) — không dùng zarr, khớp toàn bộ dữ liệu đã tải (.nc)
- **Tiền xử lý xác nhận chính xác:**
  - Log1p có dấu: `log1p(x) * sign(x)` — đúng công thức đã ghi ở CLAUDE.md, giờ xác nhận qua code thật, chỉ áp dụng cho biến được đánh dấu trong `variables_*_log1p` (không phải mọi biến đều log1p)
  - Z-score: `(x - mean) / (std + 1e-6)` — công thức chuẩn hóa cụ thể lần đầu thấy
  - Static features: min-max về `[-1, 1]` riêng (khác cách chuẩn hóa Z-score của biến động)
  - Trọng số lead time: `exp(|t - max_t| * alpha)` — khớp 100% công thức đã ghi CLAUDE.md Mục "Loss function"
- **Dữ liệu output là delta-encoded** ("current step subtracted from forecasts") — nghĩa là bản thân dataset đã trừ đi giá trị hiện tại trước khi đưa vào model, không phải chỉ riêng trong loss function
- `__getitem__` trả về dict gồm: `glofas`/`era5`/`hres_forecast`/`cpc` (dạng `[T, P, V]`), static `[P, V]`, `glofas_target`/`obs_target` (dạng `[lead_time, P, 1]`), cộng metadata (`curves`, `weight`, `file_name`, `random_indices`)

⚠️ **Phát hiện lệch giá trị mặc định, cần cẩn thận:** class `RiverMamba_dataset` có tham số `delta_t_f` (lead time) với **default nội bộ = 15**, nhưng `config.py` thật sự đặt **`delta_t_f=7`**. Vì config truyền vào sẽ ghi đè default của class, giá trị thật sự dùng là **7** (khớp "7 ngày lead time" đã biết) — chỉ là lưu ý để không hoang mang nếu đọc code thấy số 15, đó chỉ là fallback nội bộ không được dùng thật.

## 4c. Chạy inference thật — chi tiết lệnh (đọc trực tiếp `inference_aifas.py`)

- Chạy đơn giản: `python inference_aifas.py` — **không nhận argument dòng lệnh**, mọi cấu hình lấy từ `config.py` (gọi `config_file.read_arguments(train=False, print=True, save=False)`)
- **Checkpoint phải đặt đúng tên/đường dẫn cố định:** script mặc định tìm file tại `./RiverMamba_aifas_reanalysis.pth` (gán vào `config.pretrained_model`) — muốn dùng checkpoint khác (VD bản GRDC fine-tune) phải **tự đổi tên file hoặc sửa path trong config**, không tự nhận diện
- Output: file **NetCDF**, biến tên **`dis24`** (khớp đúng tên biến GloFAS đã học ở `01_glofas.md`), chiều `[time, x]`, lưu trong thư mục `./inference/`, giá trị bị clip về không âm, kiểu float32
- Setup nội bộ: cố định seed (`utils.fix_seed()`), tự nhận GPU qua `config.gpu_id`, hỗ trợ multi-GPU bằng `DataParallel` nếu có nhiều GPU

## 5. Inference — cách chia batch theo curve

Script inference chia dữ liệu dọc theo space-filling curve, chạy tuần tự theo từng đoạn bằng đúng `n_points` mà model đã train (VD model train với `n_points=254945` thì AIFAS 1.529.667 điểm sẽ tự chia thành 6 đoạn, đoạn cuối có thể lẻ). `inference_aifas.py` mặc định chỉ xuất kết quả tại đúng vị trí có trạm GRDC (có thể sửa code để xuất hết nếu cần).

**5 loại serialization curve** (thư mục `serialization/`, chạy qua `generate_curves.py`) — khớp đúng phần đã học ở LyThuyetCauTruc.md Mục 4.9:
- **Gilbert** — Hilbert curve tổng quát cho vùng chữ nhật không phải lũy thừa 2 (dùng lại implementation numpy từ `jakubcerveny/gilbert`)
- **Sweep_h / Sweep_v** — quét kiểu xoắn ốc cầu (spherical helix/loxodrome) theo chiều ngang/dọc
- **Zigzag_h / Zigzag_v** — giống Sweep nhưng đảm bảo các điểm liền kề trên curve cũng liền kề ngoài không gian thật (Sweep không đảm bảo điều này)

## 6. Checkpoint có sẵn — bảng đầy đủ từ README

**AIFAS points (1.529.667 điểm):**

| Tên | Target data | Giai đoạn train |
|---|---|---|
| RiverMamba_aifas_reanalysis | GloFAS reanalysis | 1979-2018 |
| RiverMamba_aifas_reanalysis ⚠️ | GRDC observations | 1979-2018 |

⚠️ **Bất thường nhận thấy trong README:** cả 2 dòng cùng tên file `RiverMamba_aifas_reanalysis` dù target khác nhau (1 dùng GloFAS, 1 dùng GRDC thật) — nhiều khả năng là lỗi đánh máy trong README gốc (dòng 2 lẽ ra phải có tên khác, VD `..._grdc`), cần cẩn thận khi tải để không lấy nhầm checkpoint — kiểm tra kỹ link tải thực tế, không chỉ tin tên hiển thị.

**Full resolution (6.221.926 điểm, 3000×7200 @0.05°):** 1 checkpoint, target GloFAS reanalysis, train **1979-2023** (khác AIFAS train tới 2018 — bản full map train dài hơn 5 năm).

**Reforecasts có sẵn (để baseline, không phải để train):**

| Tên | Model | # trạm | Target | Train | Reforecast |
|---|---|---|---|---|---|
| LSTM_glofas_reanalysis | ed-LSTM | 3.366 | GloFAS | 1979-2018 | 2019-2024 |
| LSTM_grdc_obs | ed-LSTM | 3.366 | GRDC | 1979-2018 | 2019-2024 |
| RiverMamba_glofas_reanalysis | RiverMamba | 3.366 | GloFAS | 1979-2018 | 2019-2024 |
| RiverMamba_grdc_obs | RiverMamba | 3.366 | GRDC | 1979-2018 | 2019-2024 |

✅ **3.366 trạm khớp đúng** con số đã ghi trong `CLAUDE.md` (fine-tune GRDC). Mỗi file netCDF reforecast có 7 bước thời gian (lead time 7 ngày), tên file = ngày phát hành dự báo lúc 00:00 UTC. Thông tin trạm tra tại `GRDC_Meta`, "index AIFAS" = vị trí trên trục x của file.

Full-map reforecast: chỉ 1 bộ, `RiverMamba_glofas_reanalysis_full_map`, train 1979-2023, reforecast năm 2024, 6.221.926 điểm dọc trục x. **Tất cả model đều deterministic, không có nowcasting.** Cũng có bản trên Hugging Face.

## 7. Dataset gốc — quy mô và quy ước quan trọng

- Nguồn đầy đủ: `https://doi.org/10.60507/FK2/T8QYWE` (~10TB sau giải nén) — CPC 16GB, ERA5-Land ~7.6TB, ECMWF-HRES ~1.9TB, GloFAS static ~46GB, GloFAS-reanalysis ~600GB
- Nén dạng `.7z`, cần gói `p7zip-full` để giải nén; file lớn (ERA5-Land/HRES) chia nhỏ theo năm, giải nén phần `.001` thì `7z` tự tìm `.002/.003...` tiếp theo
- Script tải: `scripts/download_*.sh`, dùng `wget --continue` từ `bonndata.uni-bonn.de` — ⚠️ đã xác nhận trước đó (CLAUDE.md) script này **chỉ tải file toàn cầu cố định, không giới hạn vùng được** — không dùng được để tải riêng Việt Nam, phải tải trực tiếp từ nguồn gốc từng loại dữ liệu

⚠️ **Quy ước gán nhãn thời gian — chi tiết mới, quan trọng, chưa từng ghi trước đây:** *"river discharge on 01/01/2024 correspond to the mean river discharge for the day of 12/31/2023"* — dữ liệu gán nhãn "ngày X" thực ra là **trung bình của ngày X-1** (right-labeled). Đây là điểm phải làm đúng khi ghép input/target lúc code inference thật, liên quan trực tiếp tới quy tắc lùi ngày (t-1/t-2) đã ghi trong `CLAUDE.md`/`flood-forecasting-research.md` Mục 4.4 — cần đối chiếu lại kỹ khi implement để không lệch 1 ngày.

## 8. Trích dẫn chính thức từ tác giả (BibTeX trong README)

Tác giả tự ghi `journal={arXiv preprint arXiv:2505.22535}, year={2025}` trong BibTeX ở README repo code — đây là trích dẫn arXiv tác giả để sẵn cho người dùng code, không phải bằng chứng "không có DOI proceedings". ⚠️ **Sửa lại (30/8/2026):** đã xác nhận **CÓ** DOI proceedings NeurIPS chính thức `10.52202/085713-4446` (qua `proceedings.neurips.cc`, xem `flood-forecasting-research.md` mục "Tài liệu tham khảo chính") — nhận định cũ ở đây ("không có entry NeurIPS proceedings riêng") đã SAI, chỉ là do lúc đó trang proceedings có thể chưa lên hoặc README repo chưa cập nhật theo. Ưu tiên dùng DOI proceedings khi trích dẫn chính thức.

## 9. Điểm liên quan trực tiếp tới đề tài (đối chiếu lại)

- ✅ Xác nhận lại: pretrain 1979-2018 (AIFAS) hoặc 1979-2023 (full map), 3.366 trạm GRDC fine-tune, reforecast 2019-2024 — khớp hoàn toàn với những gì đã ghi trong `CLAUDE.md`
- ✅ LOAN có thật trong code (`models/loan.py`) — không chỉ là khái niệm lý thuyết
- ✅ Công thức loss/trọng số lead time `exp(|t-max_t|*alpha)` và log1p có dấu — xác nhận khớp 100% qua đọc trực tiếp `dataset/RiverMamba_dataset.py`, không chỉ suy ra từ paper
- ⚠️ Không có baseline GRU/Transformer thuần sẵn trong repo — phải tự viết nếu muốn đủ 3 baseline như kế hoạch
- ⚠️ Fine-tune cho vùng mới cần tự chỉnh hyperparameter/loss weighting, không có công thức sẵn — đúng như đã dự đoán trong kế hoạch (Phương án B, Mục 2.4), giờ có xác nhận từ chính tác giả
- ⚠️ Quy ước nhãn thời gian right-labeled — cần kiểm tra kỹ khi ghép dữ liệu inference thật, tránh lệch 1 ngày
- ⚠️ **Rủi ro CUDA 12.1 pin cứng** (`torch==2.4.1+cu121`) — cần kiểm tra `nvidia-smi` trên Colab trước khi cài, có thể phải đổi bản torch nếu Colab runtime dùng CUDA khác
- ⚠️ **1 issue GitHub còn mở chưa giải quyết** — lỗi tải HRES (#2, mở 13/11/2025) — không có thread tham khảo sẵn nếu gặp lỗi tương tự
- ⚠️ **Checkpoint phải đặt đúng tên file cố định** (`./RiverMamba_aifas_reanalysis.pth`) hoặc tự sửa path trong config — script không tự dò tìm
- ⚠️ `batch_size_train=1` trong config gốc — vì mỗi mẫu đã là 254.945 điểm cùng lúc, không phải batch nhỏ theo nghĩa thường — cần hiểu đúng khi ước lượng VRAM free tier Colab có đủ không
- 📝 Phần lớn code là Jupyter Notebook (86.4%) — thuận lợi cho việc chạy thử trên Colab, nhiều khả năng ít phải chuyển đổi định dạng

---

*Dừng ở đây — chưa viết code. Bước tiếp theo (thật sự setup Colab, chạy thử) sẽ làm ở file `RiverMamba_code.py` riêng, chia `# %% Cell N`, đúng 1 cell mỗi lần theo quy tắc đã chốt.*
