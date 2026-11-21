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
| `Document/07_Report/` | Bản thảo báo cáo theo chương (đánh số theo dàn ý `02_Pipeline.md` Mục 9.8) |
| `Diagrams/` | Sơ đồ đề tài (`ProjectDiagram.drawio`, sinh bằng `Tools/Diagram_Build.py`): thứ tự thực hiện, kiến trúc chi tiết |
| `Workspace/` | Mã notebook Kaggle, mỗi thư mục một việc, mỗi notebook một thư mục con cùng tên (`.py`, `.ipynb` có output, báo cáo `.md`, `Figures/`, `Tables/`): `01_Download` (tải dữ liệu), `02_Exploration` (A1), `03_Baseline` (A2–A4 đã có kết quả), `04_Setup` (bước 0: dựng `mamba-ssm` cho Kaggle T4, đo tốc độ bộ nạp, kiểm mô hình chung 5 lõi, đo thời gian lưới tinh chỉnh C1), `05_Training` (huấn luyện B0–B3, C1, C2, H, D và các notebook so sánh), `06_Evaluation` (bộ đánh giá S0/kịch bản/nhiều hạn, phân tích theo lưu vực và validation bước B), `07_Demo` (hàm dự báo chung D1, đo CPU D2 và ứng dụng web năm trang D3), `08_XAI` (bước G: Integrated Gradients, dò trạng thái ẩn) |
| `NeuralHydrology/` | Thư viện huấn luyện (submodule, fork `Ticasslo/neuralhydrology`, nhánh `research`) — nơi viết phần đề tài thêm vào thư viện |
| `Tools/` | Công cụ phụ trợ: chuyển giọng nói buổi họp thành văn bản, sinh sơ đồ đề tài, sinh hình minh họa cho ghi chú lý thuyết (`Theory_Figures.py`), sinh notebook huấn luyện C1/C2/D (`C_Notebook_Template.py`, `C_Notebooks_Build.py`), bước H (`H_Notebook_Template.py`, `H_GEFS_Notebook_Template.py`, `H_Notebooks_Build.py`), notebook demo D1/D2 (`Demo_Notebooks_Build.py`) và kiểm nhanh Output D (`D_Check_Outputs.py`) |
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

Mọi bước thí nghiệm của tiểu luận đã xong (120 lần huấn luyện trên Kaggle T4); còn đưa demo lên VPS (D4). Kết quả chính:

- **A — tái lập và mốc:** chạy lại trọng số BiasCast khớp NSE trung vị cả 6 run ([A2](Workspace/03_Baseline/01_A2_LamaHCEExt_Reproduce/01_A2_LamaHCEExt_Reproduce.md)); trên giao chung 450 lưu vực, SeqLSTM có Q 0,705, DLinear 0,380 ± 0,002, persistence 0,350 ([A3](Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.md)); huấn luyện lại ba hạt giống 0,6952–0,7059 ([A4](Workspace/03_Baseline/03_A4_LamaHCEExt_Retrain/03_A4_LamaHCEExt_Retrain.md)).
- **B — cải tiến LSTM, 30 epoch, mỗi bước một thay đổi:** B0 cho ngưỡng nhiễu hạt giống 0,0079; Q cùng đơn vị (B1a) +0,0045, thêm `qmax` quá khứ (B1b) +0,0188 ở validation; B2 nhúng theo nguồn, masked mean, sửa bộ lọc mẫu; B3 che dữ liệu khi huấn luyện; **B\* = B3 P10S12** theo luật ghi trước (validation S0–S4).
- **C — năm lõi cùng tham số:** [C1](Workspace/05_Training/11_C1_LamaHCEExt_Compare/11_C1_LamaHCEExt_Compare.md) tinh chỉnh 39 cấu hình, [C2](Workspace/05_Training/13_C2_LamaHCEExt_Compare/13_C2_LamaHCEExt_Compare.md) 8 cấu hình × 3 hạt giống: NSE trung vị validation GRU 0,7686, LSTM 0,7677, Mamba 0,7630–0,7662 nằm trong một lần nhiễu; S4D, Transformer thấp hơn khoảng 3,3 lần nhiễu.
- **H — dự báo 1–7 ngày:** [LSTM, Mamba × bốn cách cấp thời tiết](Workspace/05_Training/15_H_LamaHCEExt_Compare/15_H_LamaHCEExt_Compare.md); NSE trung vị test hạn 1 => 7 của LSTM: chỉ quá khứ 0,49 => 0,09, lai 0,60 => 0,08, GEFSv12 0,63 => 0,24, dự báo hoàn hảo 0,73 => 0,64.
- **D — chuỗi quá khứ 730 ngày:** [không nâng NSE](Workspace/05_Training/17_D_LamaHCEExt_Compare/17_D_LamaHCEExt_Compare.md) của LSTM, Mamba (hiệu dưới 0,2 lần nhiễu), S4D +0,0036 ở tổ hợp; thời gian huấn luyện gần gấp đôi.
- **Đánh giá:** [Evaluation Version 13](Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md) chấm 56 run test/S0, C2 qua 7 kịch bản vận hành, 168 tệp H 7 hạn và 9 cặp D (chỉ số liên tục, 4 mức lưu lượng, sự kiện lũ, kiểm định ghép cặp); [phân tích theo lưu vực](Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md) Version 3.
- **G — giải thích mô hình:** [Integrated Gradients, dò trạng thái ẩn](Workspace/08_XAI/01_G_C2_XAI/01_G_C2_XAI.md) cho 15 run C2 và 24 run H; LSTM, GRU, Mamba dồn khoảng một nửa ảnh hưởng vào ngày t, t−1, Transformer, S4D trải gần cả cửa sổ; ở H, mô hình dùng GEFSv12 tăng dần theo hạn gần như cách dự báo hoàn hảo.
- **Demo:** [D1–D3](Workspace/07_Demo/README.md) — hàm dự báo chung trùng tệp dự báo trên GPU và CPU, ứng dụng 5 trang chạy trên laptop cho C2, H, D.

Thư viện: fork `Ticasslo/neuralhydrology`, mốc `freeze-C` = `db6dc1f` (bước C, D), `freeze-H` = `6f68f47`, `freeze-H2` = `840450f` (bước H, G).

## Tài liệu báo cáo

- [Chương 4 — Dữ liệu](Document/07_Report/04_Data.md): nguồn theo năm, đơn vị, độ phủ, cửa sổ và ngưỡng.
- [Chương 6 — Thí nghiệm và kết quả](Document/07_Report/06_ExperimentsAndResults.md): giao thức, các kết quả đã chạy và diễn giải.
- [Bộ đánh giá và bằng chứng thực nghiệm](Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md): mã, notebook, bảng kết quả và kiểm tra.
- [Phân tích kết quả theo đặc điểm lưu vực](Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md): phương pháp, kết quả Kaggle, năm hình và kiểm thử Kaggle.
