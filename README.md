# Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy

Tiểu luận chuyên ngành (định hướng khóa luận tốt nghiệp): dự báo lưu lượng dòng chảy bằng kiến trúc Mamba (State Space Model), so sánh với LSTM, GRU, Transformer, S4D. Bài cơ sở: BiasCast (Konold và cs., *Hydrology and Earth System Sciences* 30:5067–5096, 2026) — dự báo lưu lượng lớn nhất ngày trước 1 ngày trên 451 lưu vực Extended LamaH-CE, dùng dự báo thời tiết ECMWF thật và lưu lượng quan trắc.

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `Document/1-KeHoach/` | Kế hoạch tổng thể (`KeHoachTongThe.md`), giới thiệu bài cơ sở BiasCast (`GioiThieuBaiCoSo.md`), pipeline, kiến trúc, phạm vi (`KienTrucPipeline.md`) |
| `Document/2-HopGVHD/` | Biên bản họp với GVHD và việc cần làm |
| `Document/3-DuLieu/` | Ghi chú thực thi dữ liệu LamaH-CE và Extended LamaH-CE: tải, kiểm tra, kết quả |
| `Document/4-YTuong/` | Ý tưởng cho đề tài theo từng đợt tra cứu |
| `Document/5-KhaoSat/` | Khảo sát bộ dữ liệu và khảo sát tài liệu |
| `Document/6-LyThuyet/` | Lý thuyết LSTM, Transformer, GRU, Mamba; ghi chú đọc mã RiverMamba (tham khảo) |
| `Diagrams/` | Sơ đồ đề tài (`SoDoDT.drawio`): thứ tự thực hiện và kiến trúc chi tiết |
| `LamaHCE/` | Notebook Kaggle: tải, giới thiệu, phân tích dữ liệu |
| `Tools/` | Công cụ phụ trợ (chuyển giọng nói buổi họp thành văn bản) |

PDF bài báo (`PaperResearchPDF/`) và mã nguồn BiasCast (`PaperResearchCode/`) chỉ lưu trên máy, không đưa lên repo (bản quyền, dung lượng).

## Trạng thái

Đã chốt bài cơ sở BiasCast; đã tải dữ liệu, kiểm tra % thiếu và tính persistence (NSE trung vị 0,35–0,37 so với 0,705 của mô hình tốt nhất); đã viết notebook giới thiệu và phân tích dữ liệu. Việc tiếp theo: gửi kiến trúc cho GVHD, chạy lại trọng số của tác giả (bước A).
