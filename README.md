# Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy

Tiểu luận chuyên ngành (định hướng khóa luận tốt nghiệp): dự báo lưu lượng dòng chảy bằng kiến trúc Mamba (State Space Model), so sánh với LSTM, GRU, Transformer, S4D. Bài cơ sở: BiasCast (Konold và cs., *Hydrology and Earth System Sciences* 30:5067–5096, 2026) — dự báo lưu lượng lớn nhất ngày trước 1 ngày trên 451 lưu vực Extended LamaH-CE, dùng dự báo thời tiết ECMWF HRES thật và lưu lượng quan trắc.

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `Document/01_Plan/` | Kế hoạch tổng thể (`01_OverallPlan.md`), giới thiệu bài cơ sở BiasCast (`02_BasePaper.md`), pipeline, kiến trúc, phạm vi (`03_Pipeline.md`) |
| `Document/02_Meetings/` | Biên bản họp với GVHD và việc cần làm |
| `Document/03_Data/` | Ghi chú thực thi dữ liệu LamaH-CE và Extended LamaH-CE: tải, kiểm tra, kết quả |
| `Document/04_Ideas/` | Ý tưởng cho đề tài: căn cứ, chi phí, vị trí trong lộ trình |
| `Document/05_Survey/` | Khảo sát bộ dữ liệu, khảo sát tài liệu, ghi chú đọc từng PDF tham khảo |
| `Document/06_Theory/` | Lý thuyết LSTM, Transformer, GRU, Mamba; ghi chú đọc mã RiverMamba (tham khảo); ghi chú học BiasCast từng phần; cách đọc hình các chỉ số đánh giá (`08_EvaluationFigures.md`) |
| `Document/07_Report/` | Bản thảo báo cáo theo chương: Chương 4 Dữ liệu và Chương 6 Thí nghiệm, kết quả từ A1/A2/A3 và đánh giá S0 |
| `Diagrams/` | Sơ đồ đề tài (`ProjectDiagram.drawio`, sinh bằng `Tools/Diagram_Build.py`): thứ tự thực hiện, kiến trúc chi tiết, thứ tự cắt |
| `Workspace/` | Mã notebook Kaggle, mỗi thư mục một việc: `01_Download` (tải dữ liệu), `02_Exploration` (A1), `03_Baseline` (A2–A4 đã có kết quả), `04_Setup` (bước 0: dựng `mamba-ssm` cho Kaggle T4), `06_Evaluation` (bộ đánh giá, bảng và hình S0/hạn một ngày) |
| `NeuralHydrology/` | Thư viện huấn luyện (submodule, fork `Ticasslo/neuralhydrology`, nhánh `research`) — nơi viết phần đề tài thêm vào thư viện |
| `Tools/` | Công cụ phụ trợ: chuyển giọng nói buổi họp thành văn bản, sinh sơ đồ đề tài, sinh hình minh họa cho ghi chú lý thuyết (`Theory_Figures.py`) |
| `PaperResearch/` | PDF tham khảo (`PaperResearchPDF/`, ghi chú từng bài ở `Document/05_Survey/03_PaperNotes.md`) và mã tác giả BiasCast (`PaperResearchCode/`, hai submodule trỏ tới `conestone/biascast`, `conestone/neuralhydrology` — chỉ để đọc, không sửa) |

PDF trong `PaperResearch/PaperResearchPDF/` để nhóm tham khảo. Giấy phép MIT trong `LICENSE` chỉ áp dụng cho mã và tài liệu do nhóm viết; PDF thuộc bản quyền của nhà xuất bản, thư viện NeuralHydrology (`NeuralHydrology/`, `PaperResearch/PaperResearchCode/BiasCast_NH`) theo giấy phép BSD 3-Clause; kho `conestone/biascast` không có tệp giấy phép nên chỉ dùng để đọc, tham khảo; dữ liệu Extended LamaH-CE theo CC BY-NC 4.0.

## Làm việc chung

- **Phân công:** `ASSIGNMENT.md` — việc của từng người theo từng bước, điểm bàn giao, quy ước chung.
- **Đọc trước:** `Document/01_Plan/03_Pipeline.md` (tài liệu gốc: câu hỏi nghiên cứu, đóng góp, các bước, ma trận thí nghiệm), sau đó `02_BasePaper.md` và sơ đồ `Diagrams/ProjectDiagram.drawio` (mở bằng draw.io).
- **Tải repo kèm mã tác giả:** `git clone --recurse-submodules <link repo>`; nếu đã clone rồi thì chạy `git submodule update --init`.
- **Thư viện huấn luyện:** fork `github.com/Ticasslo/neuralhydrology` — nhánh `master` là mã tác giả, nhánh `research` là phần đề tài viết thêm; notebook cài theo mã commit. Sửa thư viện: làm trong `NeuralHydrology/`, commit và push trong thư mục đó (nhánh `research`), rồi commit lại thư mục `NeuralHydrology` ở repo đề tài để cập nhật con trỏ submodule.
- **Dữ liệu:** Kaggle Dataset `hunhngcthng/lamah-ce-ext` (công khai), tải bằng `Workspace/01_Download/01_LamaHCEExt_Download.py`.
- **Quy ước:** tên thư mục, tệp tiếng Anh, nội dung tiếng Việt; tài liệu trong `Document/` đánh số theo thứ tự đọc; mỗi thư mục `Workspace/` một việc; mỗi notebook có nguồn `.py` tự đủ, tên `NN_<Bước>_<BộDữLiệu>_<Việc>.py` (VD `01_A2_LamaHCEExt_Reproduce.py`, trùng tên notebook Kaggle), chia cell `# %% Phần X`. `.py` là mã nguồn để chạy tương tác; `.ipynb` cùng tên là bản chính thức tải từ phiên bản Save & Run All trên Kaggle, có output. Quy tắc làm việc chung của nhóm (cho Claude Code): `CLAUDE.md` và `.claude/rules/`; phân công: `ASSIGNMENT.md`. Không đưa khóa API vào mã (dùng Kaggle/Colab Secrets).
- **Sơ đồ:** sửa nội dung trong `Tools/Diagram_Build.py` rồi chạy lại, không sửa tay tệp `.drawio`.

## Trạng thái

Đã hoàn thành A1, tái lập suy luận A2 và các mốc A3. Trên giao chung 450 lưu vực, SeqLSTM có Q đạt NSE trung vị 0,705259; persistence qmax đạt 0,350031; DLinear ba seed đạt 0,380152 ± 0,002415. Bộ đánh giá giao thức 2 đã chạy trên Kaggle cho test/S0/hạn một ngày, khớp A3, qua 60 kiểm thử và 57 kiểm tra CSV cục bộ. Bảng lũ chính yêu cầu cùng ngày bắt đầu; bảng phụ dùng ±1 ngày. Kết quả gồm chỉ số liên tục, bốn mức, sự kiện lũ và kiểm định ghép cặp.

Bước 0 đã dựng xong `mamba-ssm` cho Kaggle T4 (torch 2.11), đạt bộ test chính thức của tác giả. A4 huấn luyện lại mô hình tốt nhất của bài với ba hạt giống: NSE trung vị test 0,6952–0,7059 (bài 0,7053), một lần huấn luyện 2,5–3,2 giờ trên T4. Luồng mô hình tiếp tục B1; luồng dữ liệu/đánh giá chấm ba tệp dự báo của A4 và tích hợp bộ nạp B2 khi B1 sẵn sàng. Các kịch bản S1–S4, nhiều hạn và so sánh năm lõi được bổ sung khi có dự báo tương ứng.

Phân tích đặc điểm lưu vực đã chạy trên Kaggle: 12 phép so, 31 thuộc tính dạng số, 1.116 hệ số cho ΔNSE, ΔPNSE và ΔPOD lũ hai năm. Có 30 kiểm thử và 143 kiểm tra kết quả cục bộ đạt. Với SeqLSTM_Q so persistence, mức cải thiện NSE liên hệ âm với độ cao trung bình (ρ = −0,422); với SeqLSTM có Q so không Q, chênh lệch NSE liên hệ dương với diện tích (ρ = 0,274). Các hệ số mô tả liên hệ trong Extended LamaH-CE.

## Tài liệu báo cáo

- [Báo cáo tiến độ cho GVHD](Document/01_Plan/04_ProgressReport_2026-10-23.md): dữ liệu, tái lập, persistence/DLinear, phân mức và sự kiện, có ví dụ giải thích chỉ số.
- [Chương 4 — Dữ liệu](Document/07_Report/04_Data.md): nguồn theo năm, đơn vị, độ phủ, cửa sổ và ngưỡng.
- [Chương 6 — Thí nghiệm và kết quả](Document/07_Report/06_ExperimentsAndResults.md): giao thức, các kết quả đã chạy và diễn giải.
- [Bộ đánh giá và bằng chứng thực nghiệm](Workspace/06_Evaluation/01_Evaluation_LamaHCEExt.md): mã, notebook, bảng kết quả và kiểm tra.
- [Phân tích kết quả theo đặc điểm lưu vực](Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin.md): phương pháp, kết quả Kaggle, bốn hình và kiểm đối chiếu CSV.
