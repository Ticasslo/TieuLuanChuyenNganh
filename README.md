# Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy

Tiểu luận chuyên ngành (định hướng khóa luận tốt nghiệp): dự báo lưu lượng dòng chảy bằng kiến trúc Mamba (State Space Model), so sánh với LSTM, GRU, Transformer, S4D. Bài cơ sở: BiasCast (Konold và cs., *Hydrology and Earth System Sciences* 30:5067–5096, 2026) — dự báo lưu lượng lớn nhất ngày trước 1 ngày trên 451 lưu vực Extended LamaH-CE, dùng dự báo thời tiết ECMWF HRES thật và lưu lượng quan trắc.

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `Document/01_Plan/` | Giới thiệu bài cơ sở BiasCast (`01_BasePaper.md`); tài liệu gốc của đề tài: tên, câu hỏi nghiên cứu, pipeline, kiến trúc, phạm vi (`02_Pipeline.md`) |
| `Document/02_Meetings/` | Biên bản họp với GVHD và việc cần làm |
| `Document/03_Data/` | Ghi chú thực thi dữ liệu LamaH-CE và Extended LamaH-CE: tải, kiểm tra, kết quả |
| `Document/04_Ideas/` | Ý tưởng cho đề tài: căn cứ, chi phí, vị trí trong lộ trình |
| `Document/05_Survey/` | Khảo sát bộ dữ liệu, khảo sát tài liệu, ghi chú đọc từng PDF tham khảo |
| `Document/06_Theory/` | Lý thuyết LSTM, Transformer, GRU, Mamba; ghi chú đọc mã RiverMamba (tham khảo); ghi chú học BiasCast từng phần; cách đọc hình các chỉ số đánh giá (`08_EvaluationFigures.md`) |
| `Document/07_Report/` | Bản thảo báo cáo theo chương: Chương 4 Dữ liệu và Chương 6 Thí nghiệm, kết quả A1–A4/B0–B3, test/S0 và validation S0–S4 |
| `Diagrams/` | Sơ đồ đề tài (`ProjectDiagram.drawio`, sinh bằng `Tools/Diagram_Build.py`): thứ tự thực hiện, kiến trúc chi tiết |
| `Workspace/` | Mã notebook Kaggle, mỗi thư mục một việc, mỗi notebook một thư mục con cùng tên (`.py`, `.ipynb` có output, báo cáo `.md`, `Figures/`, `Tables/`): `01_Download` (tải dữ liệu), `02_Exploration` (A1), `03_Baseline` (A2–A4 đã có kết quả), `04_Setup` (bước 0: dựng `mamba-ssm` cho Kaggle T4, đo tốc độ bộ nạp, kiểm mô hình chung 5 lõi, đo thời gian lưới tinh chỉnh C1), `05_Training` (huấn luyện từ B0), `06_Evaluation` (bộ đánh giá test/S0/hạn một ngày và validation S0–S4 của bước B) |
| `NeuralHydrology/` | Thư viện huấn luyện (submodule, fork `Ticasslo/neuralhydrology`, nhánh `research`) — nơi viết phần đề tài thêm vào thư viện |
| `Tools/` | Công cụ phụ trợ: chuyển giọng nói buổi họp thành văn bản, sinh sơ đồ đề tài, sinh hình minh họa cho ghi chú lý thuyết (`Theory_Figures.py`), sinh notebook huấn luyện bước C từ khuôn (`C_Notebook_Template.py`, `C_Notebooks_Build.py`) |
| `PaperResearch/` | PDF tham khảo (`PaperResearchPDF/`, ghi chú từng bài ở `Document/05_Survey/03_PaperNotes.md`) và mã tác giả BiasCast (`PaperResearchCode/`, hai submodule trỏ tới `conestone/biascast`, `conestone/neuralhydrology` — chỉ để đọc, không sửa) |

PDF trong `PaperResearch/PaperResearchPDF/` để nhóm tham khảo. Giấy phép MIT trong `LICENSE` chỉ áp dụng cho mã và tài liệu do nhóm viết; PDF thuộc bản quyền của nhà xuất bản, thư viện NeuralHydrology (`NeuralHydrology/`, `PaperResearch/PaperResearchCode/BiasCast_NH`) theo giấy phép BSD 3-Clause; kho `conestone/biascast` không có tệp giấy phép nên chỉ dùng để đọc, tham khảo; dữ liệu Extended LamaH-CE theo CC BY-NC 4.0.

## Làm việc chung

- **Phân công:** `ASSIGNMENT.md` — việc của từng người theo từng bước, điểm bàn giao, quy ước chung.
- **Đọc trước:** `Document/01_Plan/02_Pipeline.md` (tài liệu gốc: câu hỏi nghiên cứu, đóng góp, các bước, ma trận thí nghiệm), sau đó `01_BasePaper.md` và sơ đồ `Diagrams/ProjectDiagram.drawio` (mở bằng draw.io).
- **Tải repo kèm mã tác giả:** `git clone --recurse-submodules <link repo>`; nếu đã clone rồi thì chạy `git submodule update --init`.
- **Thư viện huấn luyện:** fork `github.com/Ticasslo/neuralhydrology` — nhánh `master` là mã tác giả, nhánh `research` là phần đề tài viết thêm; notebook cài theo mã commit. Sửa thư viện: làm trong `NeuralHydrology/`, commit và push trong thư mục đó (nhánh `research`), rồi commit lại thư mục `NeuralHydrology` ở repo đề tài để cập nhật con trỏ submodule.
- **Dữ liệu:** Kaggle Dataset `hunhngcthng/lamah-ce-ext` (công khai), tải bằng `Workspace/01_Download/01_LamaHCEExt_Download/01_LamaHCEExt_Download.py`.
- **Quy ước:** tên thư mục, tệp tiếng Anh, nội dung tiếng Việt; tài liệu trong `Document/` đánh số theo thứ tự đọc; mỗi thư mục `Workspace/` một việc; mỗi notebook có nguồn `.py` tự đủ, tên `NN_<Bước>_<BộDữLiệu>_<Việc>.py` (VD `01_A2_LamaHCEExt_Reproduce.py`, trùng tên notebook Kaggle), chia cell `# %% Phần X`. `.py` là mã nguồn để chạy tương tác; `.ipynb` cùng tên là bản chính thức tải từ phiên bản Save & Run All trên Kaggle, có output. Quy tắc làm việc chung của nhóm (cho Claude Code): `CLAUDE.md` và `.claude/rules/`; phân công: `ASSIGNMENT.md`. Không đưa khóa API vào mã (dùng Kaggle/Colab Secrets).
- **Sơ đồ:** sửa nội dung trong `Tools/Diagram_Build.py` rồi chạy lại, không sửa tay tệp `.drawio`.

## Trạng thái

Đã hoàn thành A1–A4 và B0–B3. Trên giao chung A3 gồm 450 lưu vực, SeqLSTM có Q đạt NSE trung vị 0,705259; persistence qmax đạt 0,350031; DLinear ba seed đạt 0,380152 ± 0,002415. [Evaluation Version 8](Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md#12-kết-quả-chấm-chung-version-8) theo giao thức 3 đã chấm 32 run trên test/S0/hạn một ngày, giao chung 652628 ngày–ID/450 ID; 205 kiểm thử và 152 kiểm chéo output đạt. Bảng lũ chính yêu cầu cùng ngày bắt đầu, bảng phụ dùng ±1 ngày. Kết quả gồm chỉ số liên tục, bốn mức, sự kiện lũ và kiểm định ghép cặp.

Bước 0 đã dựng xong `mamba-ssm` cho Kaggle T4 (torch 2.11), đạt bộ test chính thức của tác giả, và đo tốc độ bộ nạp: một epoch kể cả chấm test 12,7 phút, hơn một nửa là chờ hoặc đọc dữ liệu; bản sửa thư viện để sẵn dữ liệu trên GPU cho kết quả trùng từng bit với bộ nạp cũ và nhanh gần gấp đôi, dùng từ B0. A4 huấn luyện lại mô hình tốt nhất của bài với ba hạt giống: NSE trung vị test 0,6952–0,7059 (bài 0,7053), một lần huấn luyện 2,5–3,2 giờ trên T4. Mọi lần huấn luyện chạy đủ 30 epoch (không dừng sớm) và bước B thêm từng thay đổi một: B0 (LSTM nền 30 epoch, đã xong: NSE trung vị test 0,7056 ± 0,0017 qua ba hạt giống, ngưỡng chọn B* 0,0079, 2,4–2,6 giờ/notebook; trọng số trùng từng bit lần chạy A4 cùng hạt giống), B1a (Q quá khứ theo mm/ngày như nhãn: NSE trung vị validation 0,7558, hơn B0 +0,0045, trong ngưỡng nhiễu; cải thiện tập trung ở lưu vực nhỏ), B1b (thêm `qmax` quá khứ: validation 0,7745 ± 0,0041, hơn B1a +0,0188, vượt ngưỡng; test 0,7183 ± 0,0086), cả hai đã xong. Cho bước C, mô hình chung nhận năm lõi (LSTM, GRU, Transformer, S4D, Mamba) đã kiểm trên GPU: lõi LSTM trùng từng bit mô hình của tác giả, cả 20 cấu hình tinh chỉnh khớp số tham số trong 1%; một lần tinh chỉnh trên nền B* mất 2,1–15,6 giờ trên T4 (tổng khoảng 300 giờ), bốn cấu hình vượt 12 giờ một phiên Kaggle nên chạy qua hai phiên với kết quả như chạy liền. Nhân đã hoàn tất B2 (nhúng theo nguồn, masked mean, sửa bộ lọc mẫu) và B3 (hai mức che): chín run 30 epoch, train/Compare Version 1, chấm chung trong Evaluation Version 8. Validation S0–S4 Version 1 đã hoàn thành 12 checkpoint/84 NPZ, giao chung 652489 ngày–ID/450 ID; P10S12 qua cổng S0 và đứng đầu S1–S4 (0,70230 ± 0,00567). Thắng xác nhận B* theo luật ghi trước và gắn `freeze-C`; test/S1–S4, nhiều hạn và huấn luyện chính thức năm lõi C tiếp tục theo pipeline.

Mã bộ đánh giá hỗ trợ hạn 1–7 với dung sai chính ±⌊hạn/3⌋, đã kiểm bằng dữ liệu giả; kết quả test hiện có là S0/hạn một ngày; [validation bước B Version 1](Workspace/06_Evaluation/03_B_LamaHCEExt_Validation/03_B_LamaHCEExt_Validation.md#9-kết-quả-validation-version-1) đã chấm S0–S4/hạn một ngày. Phân tích theo lưu vực có **18 phép so, 31 thuộc tính và 1.674 hệ số** cho ΔNSE, ΔPNSE, ΔPOD hai năm, qua **41 kiểm thử Kaggle**. Ba seed LSTM giữ liên hệ âm của ΔNSE so persistence với độ cao (ρ từ −0,423 đến −0,415) và tuyết (−0,273 đến −0,261), dương với độ khô hạn (0,183 đến 0,212). Đây là liên hệ mô tả trong Extended LamaH-CE, chưa xác định đóng góp nhân quả của đầu vào. [Kết quả phân tích A4](Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md#11-kết-quả-thực-nghiệm-và-kiểm-chứng).

Bản tích hợp thư viện bàn giao cho C là `db6dc1ffcd8c14994cd80be3ed8ff417002c01a1`, kết hợp B2 và `research` tại `5844037`; các thí nghiệm B2/B3/validation đã lưu giữ pin 2ccaa14. `configure_lamah_b2` nhận mô hình chung `sequential_forecast`; B3 cần `preserve_masking=True` để giữ xác suất che, mặc định là 0 như B2. Có 15 kiểm mới đạt, phạm vi hồi quy ghi tại [biên bản tích hợp](Workspace/04_Setup/01_B2_LamaHCEExt_DataLoader/Tables/B2_C_integration_checks.json); bằng chứng T4 57 + 4 chỉ thuộc 2ccaa14. Khuôn C đã áp cấu hình B* (Phần 4 dựng như B3 với `preserve_masking=True`, so từng khóa với `config.yml` của B* hạt giống 111); kiểm trên T4 với `db6dc1f`: 247 kiểm thử của fork đạt, lõi LSTM chung chạy một epoch trùng từng bit `model_epoch001.pt` của B*; `freeze-C` = `db6dc1f`.

## Tài liệu báo cáo

- [Chương 4 — Dữ liệu](Document/07_Report/04_Data.md): nguồn theo năm, đơn vị, độ phủ, cửa sổ và ngưỡng.
- [Chương 6 — Thí nghiệm và kết quả](Document/07_Report/06_ExperimentsAndResults.md): giao thức, các kết quả đã chạy và diễn giải.
- [Bộ đánh giá và bằng chứng thực nghiệm](Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md): mã, notebook, bảng kết quả và kiểm tra.
- [Phân tích kết quả theo đặc điểm lưu vực](Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md): phương pháp, kết quả Kaggle, năm hình và kiểm thử Kaggle.
