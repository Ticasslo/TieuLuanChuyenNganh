---
paths:
  - "NeuralHydrology/**"
  - "Workspace/**/*.py"
  - "PaperResearch/PaperResearchCode/**"
  - "Plans/**"
---

# Sự thật đã kiểm về mã (NeuralHydrology, BiasCast, mamba-ssm)

> Mọi mục dưới đây đã kiểm bằng mã hoặc output thật. Trước khi dựa vào một mục cho mã mới, mở lại tệp, dòng được nêu ở đúng commit đang dùng.

## Fork và nhánh

- Fork của tác giả BiasCast (`conestone/neuralhydrology`) = thư viện gốc tại `f00cf47` + đúng 1 commit `9d94908` sửa 7 tệp: `lamah.py`, `basetrainer`, `early_stopping`, `logger`, `training/__init__`, `tester`, `config`. Không sửa `modelzoo`; Sequential Forecast LSTM có sẵn ở thư viện gốc (#199).
- Fork của đề tài `Ticasslo/neuralhydrology` (submodule `NeuralHydrology/`): `master` = `9d94908`; nhánh `research` tạo 7/10/2026, cherry-pick #280 (NumPy 2.4), #264 (dropout Transformer), #295 (chỉ số xác suất) — áp không xung đột, không đổi đầu ra điểm => `dda45fc` (bước A). B1 thêm `e794d7c` (24/10/2026): sửa lỗi lưu mô hình tốt nhất + `test/test_best_model.py`. Không lấy #303 vì fork tác giả không có thuộc tính `dynamic_learning_rate`. Thư viện gốc có thêm 19 commit tới 10/10/2026.
- Mốc `freeze-C`, `freeze-E` theo `03_Pipeline.md` Mục 9.1. Notebook cài thư viện theo mã commit, không theo tên nhánh. Cấu hình thí nghiệm nằm trong repo đề tài.
- Mã tác giả (`PaperResearch/PaperResearchCode/BiasCast`, `BiasCast_NH`) là submodule chỉ đọc; phần sửa chỉ làm ở fork `Ticasslo`.
- Tác giả huấn luyện ở commit `91af6a0` không công bố; Zenodo không có `output.log`.

## Dữ liệu và bộ nạp

- Nhãn `qmax` được bộ nạp đổi sang mm/ngày theo diện tích (vẫn là lưu lượng): Q (m³/s) = q (mm/ngày) × diện tích (km²) / 86,4.
- Q đầu vào của tác giả: `qmean` theo **m³/s**, ngày thiếu để trống, thư mục `LamaH_expanded_q_input`; bộ nạp đọc `qmean` thẳng từ tệp khí tượng (`lamah.py` dòng 107–135). A2, A4 dựng đúng như vậy (chép `qmean` từ `D_gauges` vào tệp khí tượng); B1 đổi sang mm/ngày theo `area_gov` như nhãn và thêm `qmax` quá khứ.
- Thống kê chuẩn hóa: biến đầu vào tính cả 364 ngày khởi động, nhãn bị xóa trong ngày khởi động (`basedataset.py` dòng 437–449). Tính lại từ `D_gauges` khớp tệp chuẩn hóa tác giả (4,5778 / 16,1461).
- `basedataset.py` không tham chiếu `nan_handling_method` => khi huấn luyện vẫn bỏ mẫu có NaN, kiểm **mọi cột trên cả khung**, kể cả ô mô hình không dùng; validation, test không lọc, ra dự báo NaN. Vì vậy với khung 730 ngày, bộ lọc gốc kiểm cả ô ECMWF HRES trống trước 2002 => chỉ đủ mẫu năm 2003; B2 sửa `_validate_samples`.
- Masked mean của thư viện tự điền giá trị cố định khi mọi nhóm trống, có `timestep_counter`.
- `autoregressive_inputs` chỉ chạy với `arlstm` (không dùng được cho Sequential Forecast LSTM); `lagged_features` dùng được cho mọi mô hình.
- Fork đã có `nan_step_probability`, `nan_sequence_probability` cho B3; chỉ cần sửa nhỏ nếu muốn xác suất riêng cho Q.
- `basins_filtered.txt` có sẵn trong `Experiments/` trên Zenodo. Kỳ dữ liệu (cấu hình ghi `dd/mm/YYYY`): train 2003–2009, validation 2010–2013, test 2014–2017.
- ECMWF HRES có từ 01/01/2002, các nguồn khác từ 1981; dữ liệu chỉ có ECMWF HRES hạn 1 ngày (8 cột, một giá trị/ngày).
- Q thiếu kỳ train 3,55% do 27 trạm bắt đầu đo muộn (3 trạm không có Q cả kỳ train), test 0,23%.
- Lưu vực đặc biệt: 55 không có `qmean` => không có dự báo ở run có Q (giống tác giả) và không có cửa sổ DLinear; 742, 758 có `qmax` gần như hằng số (phương sai 0,0002–0,0003, NSE −4…−131).

## Mốc A3 (người 1, `Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.md`)

- Trung vị NSE/PNSE/KGE trên giao chung 450 lưu vực, 652.628 ngày–lưu vực: SeqLSTM có Q 0,705/0,496/0,767 (hơn persistence ở 420/450); 𝒟RA 0,693/0,485/0,737; DLinear 3 hạt giống 0,380±0,002/0,061/0,518 (hơn persistence khoảng 315/450, KGE thấp hơn vì α ≈ 0,70); persistence `qmax` 0,350/0/0,675.
- DLinear làm đúng bản gốc trên `qmean` quá khứ theo mm/ngày (diện tích `area_gov`), 365 ngày đủ, không điền ô thiếu.

## Huấn luyện và đánh giá

- Lỗi lưu mô hình tốt nhất của fork tác giả (`02_BasePaper.md` Mục 11.5): `_is_best_model` so với `best_value` không tính `min_delta`, còn `EarlyStopping` chỉ cập nhật `best_value` khi cải thiện trên 0,005 => `best_model.pt` có thể bị ghi đè bởi epoch kém hơn tối đa 0,005 NSE validation. A4 giữ nguyên lỗi, B1 sửa ở `e794d7c`: `_best_metric_value` riêng trong `basetrainer.py`, dừng sớm không đổi; kiểm thử 4 trường hợp đạt trên mã mới, hỏng trên mã cũ (`best_model` epoch 8 thay vì 4); chạy huấn luyện thật trên dữ liệu kiểm thử của thư viện: `best_model.pt` trùng epoch NSE validation cao nhất. `pip install git+…` không cài `test/`, muốn chạy kiểm thử phải `git clone`. Ở A4 cả 3 hạt giống có `best_model.pt` trùng epoch NSE validation cao nhất (epoch 11/15/14), dừng sớm ở 16/20/19.
- `continue_training` không khôi phục trạng thái dừng sớm và scheduler => mỗi lần huấn luyện chạy trọn 1 phiên Kaggle.
- Không bật chế độ tất định (các bước B, C chạy cùng cách nên nhiễu đo phải gồm phần không tất định của GPU); đạo hàm Mamba không tất định từng bit kể cả khi bật chế độ tất định (`gpuAtomicAdd`) => nhiễu hạt giống là một phần của kết quả, đo ở A4 và tính vào C.
- Nhiễu hạt giống A4 (SeqLSTM có Q): NSE trung vị test 0,6952/0,7038/0,7059 (trung bình 0,7016, độ lệch chuẩn 0,0056); NSE trung vị validation 0,7411/0,7541/0,7574 (độ lệch chuẩn 0,0086); độ lệch chuẩn NSE theo lưu vực trung vị 0,0207, lớn nhất ở lưu vực NSE thấp. Chi tiết: `Workspace/03_Baseline/03_A4_LamaHCEExt_Retrain/03_A4_LamaHCEExt_Retrain.md`.
- B1 (fork `e794d7c`, `qmean_mmd` + `qmax_mmd`, 3 hạt giống): NSE trung vị test 0,7274/0,7133/0,7217 (trung bình 0,7208, độ lệch chuẩn 0,0071), validation 0,7699/0,7712/0,7728 (độ lệch chuẩn 0,0015); +0,019 NSE test so với A4 = 3,4 độ lệch chuẩn hạt giống A4, khoảng không chồng; KGE +0,0014; độ lệch chuẩn NSE theo lưu vực trung vị 0,0148. Áp quy tắc lưu mã cũ lên đường validation B1 ra cùng epoch (10/8/3) => cải thiện do đầu vào. Số tham số 84.801 (A4 84.785); một epoch kể cả validation khoảng 9,6 phút như A4, dừng sớm ở epoch 10/13/8. Lưu vực 742, 758 làm trung bình hiệu và Cohen's dz vô nghĩa => đọc trung vị hiệu. Chi tiết: `Workspace/05_Training/01_B1_LamaHCEExt_Train/01_B1_LamaHCEExt_Train.md`.
- Tái lập A2: chạy lại trọng số tác giả trên T4 khớp NSE trung vị cả 6 run (SeqLSTM có Q 0,7053, không Q 0,6276, EncDec có Q bản Complex 0,6622, EncDec không Q bản Complex 0,569, 𝒟FC 0,3864/0,3867, 𝒟RA bản Simple 0,6923); quan trắc trùng, dự báo lệch < 0,03 mm/ngày ở 99,9% cặp ngày–lưu vực; 6 run = đúng 6 đường Hình 7 (mục 3.4) theo `Inspect_Experiments/config/Role_of_integrating_discharge.yml`. EncDec không Q có cảnh báo KGE r không xác định ở ít nhất 1 lưu vực.
- Dự báo âm trước khi cắt về 0: có (1.660–2.261 cặp ngày–lưu vực ở A4, 348–4.454 ở B1); thư viện chỉ chặn âm khi tính chỉ số (`evaluation/tester.py` dòng 319–322) theo `clip_targets_to_zero: [qmax]`; `test_results.p` giữ dự báo thô => NPZ phải chặn giống vậy để NSE tính lại khớp thư viện.

## mamba-ssm và môi trường

- Môi trường Kaggle (11/10/2026, còn đúng ở 24/10/2026): Python 3.13, torch 2.11.0+cu128, NumPy 2.1.3, GPU T4 (sm_75).
- Bản phát hành `mamba-ssm` chỉ có wheel tới torch 2.10 => tự biên dịch `mamba-ssm` 2.3.2.post1 + `causal-conv1d` 1.6.2.post1 chỉ cho sm_75 (`Workspace/04_Setup/01_0_MambaSSM_Build/01_0_MambaSSM_Build.py`, Version 1, 13 phút 59 giây); đạt test chính thức của tác giả thư viện.
- `mambapy` 1.2.0 (dự phòng, chạy CPU cho demo D2–D4) nạp trọng số `strict`, trùng kết quả GPU/CPU.
- NeuralHydrology có sẵn `modelzoo/mamba.py` (`Mamba_SSM`, PR #163) nhưng **quét sai trục**: `InputLayer` trả [seq, batch, feat] còn `mamba_ssm.Mamba` nhận (batch, length, dim), nên Mamba quét theo trục batch; sửa bằng `x.transpose(0, 1)` (`02_BasePaper.md` Mục 11.5). Đề tài tự thêm Mamba vào `modelzoo` ở bước C.
