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
| `Document/06_Theory/` | Lý thuyết LSTM, Transformer, GRU, Mamba; ghi chú đọc mã RiverMamba (tham khảo) |
| `Diagrams/` | Sơ đồ đề tài (`ProjectDiagram.drawio`, sinh bằng `Tools/Diagram_Build.py`): thứ tự thực hiện, kiến trúc chi tiết, thứ tự cắt |
| `Workspace/` | Mã notebook Kaggle, mỗi thư mục một việc: `01_Download` (tải dữ liệu), `02_Exploration` (khám phá, phân tích dữ liệu) |
| `Tools/` | Công cụ phụ trợ: chuyển giọng nói buổi họp thành văn bản, sinh sơ đồ đề tài |

PDF bài báo (`PaperResearch/PaperResearchPDF/`) và mã nguồn BiasCast (`PaperResearch/PaperResearchCode/`) chỉ lưu trên máy, không đưa lên repo (bản quyền, dung lượng).

## Trạng thái

Đã chốt bài cơ sở BiasCast; đã tải dữ liệu, kiểm tra % thiếu và tính persistence (NSE trung vị 0,35–0,37 so với 0,705 của mô hình tốt nhất); đã viết notebook giới thiệu và phân tích dữ liệu. Việc tiếp theo: gửi kiến trúc cho GVHD, chạy lại trọng số của tác giả (bước A).
