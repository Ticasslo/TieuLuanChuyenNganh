# CLAUDE.md

> Hướng dẫn cho Claude Code trong repo này, cả hai thành viên dùng chung (không phải tài liệu cho hội đồng). "Người dùng" là người đang làm việc với Claude. Quy tắc chi tiết theo loại tệp nằm ở `.claude/rules/` và tự nạp khi Claude mở tệp tương ứng:
> - `notebooks.md` — viết notebook trong `Workspace/`, Kaggle, hình;
> - `docs.md` — viết tệp `.md`, vai trò từng tệp trong `Document/`;
> - `library.md` — sự thật đã kiểm về mã NeuralHydrology, BiasCast, `mamba-ssm`;
> - `references.md` — sự thật về các bài tham khảo, cách giữ PDF.

## 1. Đề tài

- Tiểu luận chuyên ngành → khóa luận tốt nghiệp: dự báo **lưu lượng dòng chảy** (Q) bằng **Mamba** (State Space Model), so sánh với các kiến trúc học sâu khác. Bằng chứng Mamba cho dòng chảy chưa thống nhất (`01_Datasets.md` 7.1, `02_LiteratureReview.md` Mục 1) => đóng khung là **nghiên cứu so sánh**, không lập luận "chọn Mamba vì Mamba luôn tốt hơn".
- **Bài cơ sở: BiasCast** (Konold và cs., HESS 30:5067–5096, 2026; chốt 6/10/2026 sau góp ý của GVHD ở buổi họp 27/9/2026): dự báo `qmax` ngày t (trước 1 ngày) cho 451 lưu vực **Extended LamaH-CE theo ngày**, dùng dự báo ECMWF HRES thật và Q quan trắc; một mô hình chung cho 451 lưu vực. Mã `github.com/conestone/biascast` + fork NeuralHydrology, trọng số Zenodo 17292895, dữ liệu Zenodo 17119635 (0,95 GB, CC BY-NC). Giới thiệu và lý do chọn: `01_BasePaper.md` (Mục 8, 9, 11).
- **Tài liệu gốc của đề tài: `Document/01_Plan/02_Pipeline.md`** (câu hỏi nghiên cứu, đóng góp, các bước, ma trận tối đa 136 lần huấn luyện, thứ tự cắt Mục 8.3, việc chưa chốt Mục 12). Đổi phạm vi hay quyết định => sửa cả `01_BasePaper.md`, `02_Pipeline.md` và sơ đồ.
- **Tên:** tiểu luận dùng tên đăng ký với GVHD *"Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy"* (chốt 19/8/2026, giữ 18/10/2026); tên khóa luận **chưa chốt**. Tên **không** có chữ "lũ lụt": đầu ra là Q, không phải vùng ngập hay mực nước; "nguy cơ lũ" chỉ suy gián tiếp qua ngưỡng chu kỳ lặp lại trên Q (Gumbel L-moments từng trạm), không quy ra mét (cần đường quan hệ mực nước–lưu lượng riêng từng mặt cắt).
- **Chỉ tham khảo, không chạy:** RiverMamba (kiến trúc, loss), Kirschstein & Sun ICML 2024 (hướng đồ thị cho khóa luận), OpenHydroNet — chi tiết ở `.claude/rules/references.md`.
- ⛔ **Không làm, không đề xuất, không nhắc lại bất cứ thứ gì liên quan tới lưu vực Vu Gia–Thu Bồn** (yêu cầu dứt khoát 25/9/2026).
- Mục "chưa chốt" trong tệp kế hoạch là cố ý: không tự điền, không tự quyết thay — hỏi người dùng.
- Đề tài linh hoạt: được đề xuất đổi hướng khi có cách tốt hơn, phải nêu lý do và nguồn.

## 2. Nhóm và phân công

- **Người 1: Huỳnh Thanh Nhân** (Kaggle `hunhthanhnhn`) — dữ liệu, bộ nạp, đánh giá, demo D1–D3.
- **Người 2: Huỳnh Ngọc Thắng** (Kaggle `hunhngcthng`, GitHub `Ticasslo`, chủ VPS) — thư viện, các lõi mô hình, huấn luyện, giải thích mô hình (XAI), demo D4 trên VPS.
- Việc theo từng bước, điểm bàn giao, quy ước chung: `ASSIGNMENT.md`; lịch hai luồng `02_Pipeline.md` Mục 9.7; dàn ý báo cáo Mục 9.8.
- Bộ nhớ Claude (`~/.claude/projects/...`) chỉ ở máy từng người; quy tắc chung của nhóm nằm ở tệp này và `.claude/rules/`.

## 3. Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `Document/01_Plan` … `07_Report` | Ghi chú `.md` đánh số theo thứ tự đọc: kế hoạch, biên bản họp, dữ liệu, ý tưởng, khảo sát, lý thuyết, bản thảo báo cáo (vai trò từng tệp: `.claude/rules/docs.md`) |
| `Workspace/<NN_Việc>/<notebook>/` | Notebook Kaggle, mỗi thư mục một việc, mỗi notebook một thư mục con cùng tên (`.py`, `.ipynb` chính thức có output, báo cáo `.md`, `Figures/`, `Tables/`): `01_Download`, `02_Exploration` (A1), `03_Baseline` (A2–A4), `04_Setup` (bước 0), `05_Training` (B0 trở đi), `06_Evaluation` (bộ đánh giá); `07_Demo` thêm khi bắt đầu |
| `Diagrams/ProjectDiagram.drawio` | Sơ đồ duy nhất của đề tài, sinh bằng `Tools/Diagram_Build.py` (sửa Phần 2 của script rồi chạy lại, không sửa tay tệp drawio) |
| `Tools/` | Công cụ phụ trợ: chép lời buổi họp, sinh sơ đồ, sinh hình lý thuyết |
| `NeuralHydrology/` | Submodule fork `Ticasslo/neuralhydrology`: `master` = mã tác giả `9d94908`, `research` = phần đề tài viết thêm (mốc `freeze-C`, `freeze-E` theo `02_Pipeline.md` Mục 9.1) |
| `PaperResearch/` | PDF tham khảo và mã tác giả BiasCast (2 submodule chỉ đọc) |
| `Plans/` | Kế hoạch từng bước của người 2, chỉ ở máy (loại bằng `.git/info/exclude`) |
| Gốc repo | `README.md`, `CLAUDE.md`, `ASSIGNMENT.md`, `LICENSE`, `.gitmodules` |

- Tên thư mục và tệp viết **tiếng Anh**, nội dung tiếng Việt. Tệp mới trong `Document/` đánh số tiếp theo thứ tự đọc (biên bản họp `Meeting_<ngày>.md`); PDF đặt tên `AuthorYear_Venue_Topic.pdf`.
- Trong ghi chú, tên tệp `.md` trần (VD `02_Pipeline.md` Mục 6.2) là tệp trong `Document/`; đường dẫn `PaperResearch/...` tính từ gốc repo.
- Quá trình chọn bài cơ sở (137 bài, `CHECKPDF.md`, `CHECKCODE.md`) đã xóa 7/10/2026; xem lại bằng `git show 247769a:Document/6-ChonBaiCoSo/<tên tệp>`.

## 4. Quy tắc làm việc — BẮT BUỘC

**Trung thực và nguồn**
1. **Tra kỹ trước khi nói, không bịa, không "tự cảm nhận".** Mọi khẳng định kỹ thuật, số liệu phải có nguồn kiểm được (bài báo, trang chính thức, mã thật, kết quả tự chạy). Chưa chắc thì nói "chưa xác nhận".
2. **Phát hiện mình từng nói sai thì nói thẳng là sai**, không lặng lẽ sửa, không đổ lỗi cho người dùng khi lỗi do mình suy đoán (tiền lệ: đoán sai thứ tự tham số `area` của cdsapi, người dùng làm đúng theo hình).
3. **Thứ tự tin cậy:** mã/output thật > tài liệu chính thức > suy luận > "chắc là vậy". Bằng chứng mới mâu thuẫn suy đoán cũ thì sửa ngay.
4. **Không tra được thì nói ngay** (Google Scholar "Cited by", Scopus, trang chặn bot, paywall…): bí ở bước nào, vì sao, người dùng cần tra ở đâu để đưa lại kết quả. Không báo "không tìm thấy" như thể đã tra hết.
5. **Tra mã nguồn, tài liệu đúng phiên bản trước khi viết code**; trong câu trả lời nêu ngắn đã kiểm gì (tệp, dòng, commit).

**Quy trình**
6. **Lập kế hoạch, chờ duyệt rồi mới làm:** trước mỗi bước, notebook hay sửa mã, trình bày kế hoạch (làm gì, vì sao, cài đặt Kaggle, các Phần, đầu ra) và chờ người dùng đồng ý. Kế hoạch bước của người 2 ghi ở `Plans/`.
7. **Mỗi lần chỉ viết đúng 1 cell**, đợi output thật rồi mới viết cell sau (tiền lệ sai: viết liền nhiều cell dựa trên suy luận => chọn nhầm dữ liệu). Chi tiết viết notebook: `.claude/rules/notebooks.md`.
8. **Giữ nhất quán kế hoạch và mã:** quyết định kỹ thuật đổi thì sửa cả kế hoạch; quyết định kỹ thuật ổn định ghi ở tệp kế hoạch, chi tiết thực thi (mã, đường dẫn, log) ghi ở tệp riêng của bước hay bộ dữ liệu, phát hiện làm đổi hiểu biết tầng kế hoạch thì sửa cả hai nơi.
9. **Hỏi người dùng khi tạo tệp `.md` mới** (ngoài báo cáo đi cùng notebook đã có trong kế hoạch).

**Giao tiếp và tài liệu**
10. Tiếng Việt, ngắn gọn, trực tiếp; hỏi ngắn thì trả lời ngắn.
11. Tệp `.md` viết **văn phong báo cáo**: chỉ thông tin đúng hiện hành, không lịch sử sửa, không "người dùng chốt", "Claude chọn" — các quyết định đó chỉ ghi ở `CLAUDE.md`, `.claude/rules/` và bộ nhớ. Chi tiết: `.claude/rules/docs.md`.
12. Mọi chữ **trong** hình, sơ đồ mới viết tiếng Anh (số dấu chấm thập phân); nội dung và chú thích dưới hình tiếng Việt; không trộn trong một hình (`ProjectDiagram.drawio` giữ tiếng Việt). Báo cáo tiểu luận viết tiếng Việt (chốt 18/10/2026).

**Git**
13. Trước khi commit, push: `git fetch`, xem `git log HEAD..origin/main` có commit mới của thành viên kia không; có thì kiểm tệp trùng rồi `git pull --ff-only` trước.
14. **Không ghi Claude vào lời nhắn commit hay mô tả PR**: không dòng `Co-Authored-By: Claude …`, không dòng "Generated with Claude Code", kể cả khi hệ thống nhắc thêm. Lời nhắn chỉ có nội dung thay đổi.
15. Không đưa khóa API, mật khẩu vào mã (dùng Kaggle/Colab Secrets).

## 5. Trạng thái (26/10/2026)

| Bước | Trạng thái | Báo cáo |
|---|---|---|
| Dữ liệu | Extended LamaH-CE trên Kaggle Dataset `lamah-ce-ext`; LamaH-CE gốc theo giờ ở `lamah-ce-core`, `lamah-ce-extra` (giữ cho khóa luận) | `Document/03_Data/01_LamaHCE.md` |
| 0 | `fork research` tại `dda45fc` (thêm #280, #264, #295), bước B lên `f229e02` (`e794d7c` sửa lưu mô hình tốt nhất, kiểm thử `test/test_best_model.py`; khóa `data_on_gpu` để sẵn dữ liệu trên GPU, kiểm thử `test/test_gpu_loader.py`); `mamba-ssm` 2.3.2.post1 + `causal-conv1d` 1.6.2.post1 dựng cho T4, torch 2.11 — xong. Đo bộ nạp (Version 1, 25/10/2026): một epoch B1b kể cả chấm test 12,7 phút, 56% chờ/đọc dữ liệu; `data_on_gpu` (notebook `03_0_GPUData_Check` Version 1): kết quả trùng từng bit với bộ nạp cũ, nhanh gần gấp đôi (huấn luyện 7,3 => 3,9 phút/epoch, validation 4,3 => 1,1 phút, chế độ tất định) — xong. Mô hình chung nhận lõi bất kỳ, các lõi, kiểm thử đơn vị — chưa làm | `Workspace/04_Setup/01_0_MambaSSM_Build/01_0_MambaSSM_Build.md`, `02_0_DataLoader_Benchmark/02_0_DataLoader_Benchmark.md` (Mục 5: kiểm GPU) |
| A2 | Tái lập bằng trọng số tác giả, khớp NSE trung vị cả 6 run (SeqLSTM có Q 0,7053) | `Workspace/03_Baseline/01_A2_LamaHCEExt_Reproduce/01_A2_LamaHCEExt_Reproduce.md` |
| A1 | Khảo sát dữ liệu, ngưỡng 4 mức và ngưỡng lũ (người 1) | `Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/01_A1_LamaHCEExt_Explore.md` |
| A3 | Persistence, PNSE, KGE, DLinear 3 hạt giống (người 1): persistence 0,350, DLinear 0,380, SeqLSTM có Q 0,705 trên giao chung 450 lưu vực | `Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.md` |
| A4 | Huấn luyện lại SeqLSTM có Q, 3 hạt giống: NSE trung vị test 0,6952/0,7038/0,7059 (bài 0,7053 nằm trong khoảng); độ lệch chuẩn NSE trung vị validation 0,0086 (giao thức dừng sớm của tác giả; luật B* dùng độ lệch chuẩn của B0 30 epoch); 2,5–3,2 giờ/lần trên T4; 3 tệp NPZ cho bộ đánh giá | `Workspace/03_Baseline/03_A4_LamaHCEExt_Retrain/03_A4_LamaHCEExt_Retrain.md` |
| B0 | SeqLSTM có Q 30 epoch, 3 hạt giống (Version 1 cả 4 notebook): `best_model.pt` epoch 25/15/14; NSE trung vị validation 0,7423/0,7541/0,7574 => **ngưỡng B\* 0,0079**; test 0,7056 ± 0,0017 (bài 0,7053); trọng số trùng từng bit A4 ở mọi epoch chung (nhiễu chỉ do hạt giống); chạy đủ 30 epoch hơn dừng sớm +0,0043 ở hạt giống 111, nhỏ hơn nhiễu hạt giống; 2,36–2,58 giờ/notebook; 3 NPZ `SeqLSTM_Q_B0_seed*` ở Output `02_B0_LamaHCEExt_Compare` Version 1 | `Workspace/05_Training/01_B0_LamaHCEExt_Train/01_B0_LamaHCEExt_Train.md` |
| Đánh giá | Bộ đánh giá Version 4 chấm 14 run (6 A2, 3 DLinear, 2 persistence, 3 A4) trên test/S0/hạn 1 ngày; phân tích theo đặc điểm lưu vực Version 2, 18 phép so (người 1) | `Workspace/06_Evaluation/01_…md`, `02_…md` |
| Tiếp theo | Mỗi người nhận trọn một bước (chốt 26/10/2026). Người 1: chấm 3 NPZ B0 (bộ đánh giá Version 5, mục B0 Chương 6); **B2 trọn gói** (bộ nạp + mô hình + huấn luyện + báo cáo, trên cấu hình B1b), rồi **B3**; chấm B1a, B1b. Người 2: **B1a, B1b** 30 epoch; phần còn lại bước 0 (mô hình chung nhận lõi bất kỳ, các lõi, kiểm thử đơn vị); B*, `freeze-C`. Họp GVHD kế tiếp Chủ nhật 8/11/2026: báo đo bộ nạp, B0 và phần xong tới đó | `ASSIGNMENT.md` |

## 6. Quyết định đã chốt (không hỏi lại, không tranh luận lại)

**Phạm vi tiểu luận** (`02_Pipeline.md` Mục 7; viết lại 7/10/2026 theo yêu cầu "pipeline tốt nhất, làm kĩ": ưu tiên chất lượng, nhấn mạnh tính mới)
- 0 hạ tầng → A tái lập + mốc + nhiễu 3 hạt giống → B LSTM cải tiến, mỗi bước thêm một thay đổi, 30 epoch (B0 nền đã sửa lưu mô hình tốt nhất; B1a Q cùng đơn vị mm/ngày; B1b thêm `qmax` quá khứ; B2 nhúng theo nguồn, masked mean, sửa `_validate_samples`; B3 che dữ liệu 0,1/0,12 và 0,05/0,05; B* chọn theo luật ghi sẵn) → C 5 lõi (LSTM, GRU, Transformer, S4D, Mamba) trong mô hình chung Sequential Forecast, số tham số lệch ≤ 1%, tinh chỉnh 8 cấu hình/lõi, chạy chính thức 3 hạt giống → **H dự báo 1–7 ngày** (chốt 7/10/2026, ưu tiên hơn E: LSTM + lõi SSM tốt nhất × 3 cách: chỉ quá khứ; lai — ngày đầu ECMWF HRES thật, ngày sau để trống xử lý bằng masked mean; dự báo hoàn hảo bằng tái phân tích làm cận trên) → D 730 ngày (LSTM, Mamba, S4D; so với 365 ngày, cả hai dùng bộ lọc đã sửa để cùng tập mẫu — chốt 18/10/2026) → E Mamba hai chiều, Transformer patch → F CMAL → GV kiểm chứng lấy mẫu nhiều ngày lũ (Y7) → đánh giá → G XAI (Integrated Gradients + dò trạng thái ẩn Y4, chốt 7/10/2026) và demo.
- **Giao thức huấn luyện (chốt 25/10/2026 theo góp ý GVHD họp 25/10, người dùng giao chọn cách tốt nhất):** mọi lần huấn luyện chạy đủ 30 epoch (hết chu kỳ cosine T_max 30 của tác giả), không dừng sớm, `best_model.pt` theo NSE trung vị validation (`early_stopping_patience: 30`; bỏ khóa thì thư viện không lưu `best_model.pt`). Thêm B0 (LSTM nền A4 trên `f229e02`, 30 epoch, thước đo nhiễu hạt giống cho B*), B1a (B0 + đơn vị Q, ablation), B1b (B1a + `qmax` quá khứ). A4 giữ làm kết quả tái lập; B* chọn trong B1b–B3. Bản B1 dừng sớm (đổi cùng lúc hai thay đổi) đã xóa khỏi repo 25/10/2026, xem lại ở commit `a06a224`. Test mọi epoch cho lần chạy chính thức, không cho tinh chỉnh C1.
- Tối đa 136 lần huấn luyện; cứ chạy theo kế hoạch, không cắt vì ngân sách (giao thức 30 epoch, `data_on_gpu`, số đo B0: khoảng 2,5 giờ/lần chính thức, 2,0 giờ/lần C1, khoảng 320 giờ GPU); thứ tự cắt khi thiếu thời gian: `02_Pipeline.md` Mục 8.3.
- Tính mới đóng khung trung thực so với Shahriar (SSRN 2026), Zhang 2026, Demiray, Dubey 2025, MF²LSTM, HydroDiffusion (WRR 2026), Saint-Fleur (HESS 2026) — `02_Pipeline.md` Mục 1.3.
- Khóa luận: dữ liệu giờ, dự báo nhiều ngày với dự báo thời tiết nhiều hạn thật (GEFSv12 hoặc kho ECMWF), loss ưu tiên đỉnh, đồ thị, biến thể lai, xLSTM.

**Đánh giá** (`02_Pipeline.md` Mục 6)
- Ba tầng chỉ số (đồng ý 18/10/2026): **chỉ NSE trung vị validation dùng để chọn** (checkpoint, cấu hình, B*, mô hình demo); so các lõi trên test bằng nhiều chỉ số; %BiasFHV báo cả trung vị có dấu và trung vị trị tuyệt đối.
- NSE, KGE, PNSE; Wilcoxon + Cohen's d + Holm. A4 cho thấy hai lần chạy chỉ khác hạt giống vẫn có p Holm < 0,05 => so mô hình phải kèm độ lớn hiệu, so với mốc nhiễu hạt giống, không dùng p một mình.
- 4 mức Yilmaz; sự kiện lũ chu kỳ lặp lại 1,5/2/5/10 năm, Gumbel L-moments trên đỉnh năm thủy văn 1/10–30/9, bỏ năm thiếu quá nửa (Claude chọn 17/10/2026 khi được giao: 1,5 thay 1,01 của Nearing vì ngưỡng 1,01 thấp hơn khoảng 99% đỉnh năm, trùng mức "rất cao").
- Ghép trận đúng ngày bắt đầu (cửa sổ 0); bước H ±⌊hạn/3⌋; bảng phụ ±1 (duyệt 23/10/2026 theo quy tắc ±1/3 hạn của NOAA CR-TA 98-03, vì ±1 cho persistence POD = 1); gộp trận theo tiêu chí độc lập USWRC 1976, bản không gộp báo cáo kèm; kết quả chính cộng dồn trận của 451 lưu vực, không dùng quy ước gán 1 của Nearing khi lưu vực không có trận.
- Kịch bản S0–S4 theo độ trễ thực của từng nguồn: E-OBS có bản hằng ngày trễ 1 ngày (Copernicus *Daily vs Full E-OBS*), GLEAM phát hành mỗi năm (ICDC Hamburg), MSWEP NRT 2–3 giờ (UCAR).

**Hạ tầng**
- **Kaggle là compute chính** (chốt 27/9/2026; GPU 30 giờ/tuần, tối đa 12 giờ/phiên, chỉ dùng T4 vì `mamba-ssm` cần sm_75 trở lên); Colab miễn phí dự phòng (Colab Pro đã hủy 26/9/2026); không dùng GPU trả phí trừ khi gói miễn phí không đủ. Người dùng tự chép notebook vào Kaggle (không cài Kaggle CLI).
- Bước A chạy trên nhánh `research` tại `dda45fc` (Claude chốt 7/10/2026, ba bản sửa áp không xung đột, không đổi đầu ra điểm); không lấy #303.
- DLinear làm đúng bản gốc trên `qmean` quá khứ (Claude chọn 11/10/2026 khi được giao).
- A4, B, C: không bật chế độ tất định; B0 cho thấy với LSTM trên T4 cùng hạt giống vẫn trùng từng bit (nhiễu chỉ do hạt giống), Mamba kiểm lại ở C; mỗi lần huấn luyện trọn 1 phiên Kaggle (chi tiết `.claude/rules/library.md`).

**Demo và XAI**
- Demo là một mục trong trang web nhiều mục của người 2 (giới thiệu, hồ sơ, dự án), chạy 24/7 trên VPS Oracle Cloud Always Free (từ 12/6/2026 hạn mức A1 còn 2 OCPU/12 GB; máy rảnh 7 ngày dưới 20% CPU/mạng/bộ nhớ có thể bị thu hồi — docs.oracle.com). Mức 2 (chọn 10/10/2026): mô hình chạy thật trên CPU, thử "nếu… thì…" với ECMWF HRES ngày t; 4 bước D1 Kaggle GPU → D2 Kaggle CPU (Mamba dùng `mambapy`) → D3 laptop → D4 VPS, bản laptop dự phòng ngày bảo vệ (`02_Pipeline.md` Mục 9.6). Không thuê GPU phục vụ web, không làm giao diện trên Kaggle; công nghệ giao diện, Cloudflare Tunnel, tên miền tra khi có demo chạy thử (Mục 12).
- XAI: feature attribution qua Captum (Integrated Gradients/SHAP), **không** fine-tune LLM viết báo cáo, thay bằng câu mẫu điền số (23/8/2026). Demiray & Demir (preprint 2025, WaterBench-Iowa) đã làm Mamba + SHAP => cần đóng góp khác biệt.

**Góp ý GVHD** (`Document/02_Meetings/Meeting_2026-09-27.md`, `Meeting_2026-10-11.md`, `Meeting_2026-10-25.md`)
- Mục dữ liệu: gạch đầu dòng theo năm phát hành + đoạn thảo luận thống kê để biện luận.
- Kiến trúc tách rõ khối giữ của bài gốc và khối thêm/sửa; không khẳng định điều không kiểm soát được; không dùng LLM.
- **Đánh giá theo mức lưu lượng** như bài toán phân loại, ưu tiên báo động nhầm hơn bỏ sót lũ (`02_Pipeline.md` Mục 6.2).
- Theo dõi chỉ số tập test mỗi epoch (đề tài vẫn chọn checkpoint theo validation); khuyến khích viết tiếng Anh.
- Y7 (lấy mẫu nhiều ngày lũ, bằng chứng trái chiều) đưa vào bước GV; Y8 (đóng băng có chọn lọc theo XAI) chưa đưa vào.
- Họp 11/10/2026 (`Meeting_2026-10-11.md`): thêm đoạn lý thuyết thuộc tính lưu vực vào Chương 4 (`02_Pipeline.md` Mục 9.8); khởi tạo ưu tiên đặc trưng ghi thành Y22, không đưa vào tiểu luận; vẽ training loss; không chọn seed theo kết quả.
- Họp 25/10/2026 (`Meeting_2026-10-25.md`): so sánh cùng số epoch (=> giao thức 30 epoch); ablation thêm từng thay đổi (=> B0, B1a, B1b); đo tốc độ bộ nạp chỉ lặp dữ liệu, không mô hình, rồi báo thầy (notebook `Workspace/04_Setup/02_0_DataLoader_Benchmark/`); test trong từng epoch để theo dõi; tự đề xuất hướng, không hỏi "làm gì tiếp".
- Khảo sát bài báo: không MDPI, không chỉ preprint, trích dẫn ≥ 5, ưu tiên Scimago Q1/Q2 và venue CNTT/AI (điểm cộng, không bắt buộc — đính chính 26/9/2026). Tiểu luận **phải có demo**; khóa luận nâng thành phần mềm có tính ứng dụng.
