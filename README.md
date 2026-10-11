# Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy

Tiểu luận chuyên ngành (định hướng khóa luận tốt nghiệp): dự báo lưu lượng dòng chảy bằng kiến trúc Mamba (State Space Model), so sánh với LSTM, GRU, Transformer, S4D. Bài cơ sở: BiasCast (Konold và cs., *Hydrology and Earth System Sciences* 30:5067–5096, 2026) — dự báo lưu lượng lớn nhất ngày trước 1 ngày trên 451 lưu vực Extended LamaH-CE, dùng dự báo thời tiết ECMWF thật và lưu lượng quan trắc.

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `Document/01_Plan/` | Kế hoạch tổng thể (`01_OverallPlan.md`), giới thiệu bài cơ sở BiasCast (`02_BasePaper.md`), pipeline, kiến trúc, phạm vi (`03_Pipeline.md`) |
| `Document/02_Meetings/` | Biên bản họp với GVHD và việc cần làm |
| `Document/03_Data/` | Ghi chú thực thi dữ liệu LamaH-CE và Extended LamaH-CE: tải, kiểm tra, kết quả |
| `Document/04_Ideas/` | Ý tưởng cho đề tài: căn cứ, chi phí, vị trí trong lộ trình |
| `Document/05_Survey/` | Khảo sát bộ dữ liệu, khảo sát tài liệu, ghi chú đọc từng PDF tham khảo |
| `Document/06_Theory/` | Lý thuyết LSTM, Transformer, GRU, Mamba; ghi chú đọc mã RiverMamba (tham khảo); ghi chú học BiasCast từng phần |
| `Diagrams/` | Sơ đồ đề tài (`ProjectDiagram.drawio`, sinh bằng `Tools/Diagram_Build.py`): thứ tự thực hiện, kiến trúc chi tiết, thứ tự cắt |
| `Workspace/` | Mã notebook Kaggle, mỗi thư mục một việc: `01_Download` (tải dữ liệu), `02_Exploration` (khám phá dữ liệu, bước A1 — mới có nháp Phần 1), `03_Baseline` (tái lập và mốc so sánh, bước A2–A4 — A2 đã chạy, tái lập đạt) |
| `NeuralHydrology/` | Thư viện huấn luyện (submodule, fork `Ticasslo/neuralhydrology`, nhánh `research`) — nơi viết phần đề tài thêm vào thư viện |
| `Tools/` | Công cụ phụ trợ: chuyển giọng nói buổi họp thành văn bản, sinh sơ đồ đề tài |
| `PaperResearch/` | PDF tham khảo (`PaperResearchPDF/`, ghi chú từng bài ở `Document/05_Survey/03_PaperNotes.md`) và mã tác giả BiasCast (`PaperResearchCode/`, hai submodule trỏ tới `conestone/biascast`, `conestone/neuralhydrology` — chỉ để đọc, không sửa) |

PDF trong `PaperResearch/PaperResearchPDF/` để nhóm tham khảo. Giấy phép MIT trong `LICENSE` chỉ áp dụng cho mã và tài liệu do nhóm viết; PDF thuộc bản quyền của nhà xuất bản, thư viện NeuralHydrology (`NeuralHydrology/`, `PaperResearch/PaperResearchCode/BiasCast_NH`) theo giấy phép BSD 3-Clause; kho `conestone/biascast` không có tệp giấy phép nên chỉ dùng để đọc, tham khảo; dữ liệu Extended LamaH-CE theo CC BY-NC 4.0.

## Làm việc chung

- **Đọc trước:** `Document/01_Plan/03_Pipeline.md` (tài liệu gốc: câu hỏi nghiên cứu, đóng góp, các bước, ma trận thí nghiệm), sau đó `02_BasePaper.md` và sơ đồ `Diagrams/ProjectDiagram.drawio` (mở bằng draw.io).
- **Tải repo kèm mã tác giả:** `git clone --recurse-submodules <link repo>`; nếu đã clone rồi thì chạy `git submodule update --init`.
- **Thư viện huấn luyện:** fork `github.com/Ticasslo/neuralhydrology` — nhánh `master` là mã tác giả, nhánh `research` là phần đề tài viết thêm; notebook cài theo mã commit. Sửa thư viện: làm trong `NeuralHydrology/`, commit và push trong thư mục đó (nhánh `research`), rồi commit lại thư mục `NeuralHydrology` ở repo đề tài để cập nhật con trỏ submodule.
- **Dữ liệu:** Kaggle Dataset `lamah-ce-ext` (riêng tư, cần chia sẻ quyền truy cập cho thành viên), tải bằng `Workspace/01_Download/01_LamaHCEExt_Download.py`.
- **Quy ước:** tên thư mục, tệp tiếng Anh, nội dung tiếng Việt; tài liệu trong `Document/` đánh số theo thứ tự đọc; mỗi thư mục `Workspace/` một việc; mỗi notebook một tệp `.py` tự đủ, tên `NN_<Bước>_<BộDữLiệu>_<Việc>.py` (VD `01_A2_LamaHCEExt_Reproduce.py`, trùng tên notebook Kaggle), chia cell `# %% Phần X`; không đưa khóa API vào mã (dùng Kaggle/Colab Secrets).
- **Sơ đồ:** sửa nội dung trong `Tools/Diagram_Build.py` rồi chạy lại, không sửa tay tệp `.drawio`.

## Trạng thái

Đã chốt bài cơ sở BiasCast và pipeline; đã tải dữ liệu, kiểm tra % thiếu và tính persistence (NSE trung vị 0,35–0,37 so với 0,705 của mô hình tốt nhất). Việc tiếp theo: gửi kiến trúc cho GVHD; bước 0 (sửa thư viện trên nhánh `research` của fork, cài `mamba-ssm`, kiểm thử đơn vị); viết lại notebook khám phá dữ liệu; bước A (chạy lại trọng số của tác giả).
