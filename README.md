# Ứng dụng mô hình học sâu trong bài toán dự báo lưu lượng dòng chảy

Tiểu luận chuyên ngành (định hướng khóa luận tốt nghiệp): dự báo lưu lượng dòng chảy Q (m³/s) bằng kiến trúc Mamba (State Space Model), so sánh với các kiến trúc học sâu khác (LSTM, GRU, Transformer, mô hình đồ thị) trên các bộ dữ liệu benchmark công khai.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `flood-forecasting-research.md` | Kế hoạch tổng thể: bài toán, bộ dữ liệu, kiến trúc, baseline, chỉ số đánh giá, hạ tầng |
| `KhaoSat/Dataset.md` | Khảo sát 4 bộ dữ liệu (LamaH-CE, CAMELS-US, Columbia Basin, WaterBench-Iowa) và các công trình đã huấn luyện mô hình trên chúng |
| `KhaoSat/02_literature_review.md` | Khảo sát tài liệu: hiệu quả của Mamba, các bài dự báo lưu lượng ở venue CNTT/AI, yêu cầu demo |
| `BaiCoSo/RESEARCHING.md` | Danh sách 137 bài báo (từ 2024) có huấn luyện mô hình dự báo lưu lượng |
| `BaiCoSo/RESEARCHDONE.md` | Tổng hợp các bài quan trọng và xếp hạng ứng viên bài cơ sở |
| `BaiCoSo/CHECKPDF.md` | Báo cáo đọc toàn văn các ứng viên bài cơ sở, tiêu chí, xếp hạng và đề xuất |
| `BaiCoSo/CHECKCODE.md` | Báo cáo đọc mã nguồn các ứng viên bài cơ sở |
| `BaiCoSo/SOSANH.md` | So sánh chi tiết hai ứng viên Kirschstein & Sun (ICML 2024) và BiasCast (HESS 2026): dữ liệu, đầu vào, đầu ra, mô hình, huấn luyện, kết quả |
| `TongQuan/ChonBaiCoSo.md` | Đề xuất bài cơ sở gửi GVHD: BiasCast (chính), Kirschstein & Sun (thay thế) |
| `TongQuan/GioiThieuBaiCoSo_BiasCast.md` | Giới thiệu bài cơ sở BiasCast (HESS 2026) |
| `TongQuan/GioiThieuBaiCoSo.md` | Giới thiệu bài cơ sở Kirschstein & Sun (ICML 2024), phương án thay thế |
| `TongQuan/KienTrucPipeline.md` | Kiến trúc và pipeline của đề tài: các giai đoạn, mô-đun mô hình, chỉ số đánh giá, việc chưa chốt |
| `TongQuan/KienTrucPipeline_BiasCast.md` | Kiến trúc và pipeline theo phương án bài cơ sở BiasCast (HESS 2026) |
| `LamaHCE/` | Ghi chú và mã tải, xử lý bộ dữ liệu LamaH-CE cho bài cơ sở (chạy trên Kaggle) |
| `LyThuyet/LyThuyetCauTruc.md` | Ghi chú lý thuyết LSTM, Transformer, GRU, Mamba kèm ví dụ số |
| `LyThuyet/RiverMamba.md` | Ghi chú đọc mã nguồn RiverMamba (tài liệu tham khảo) |

PDF bài báo và mã nguồn của các công trình tham khảo không đưa lên repo (bản quyền, dung lượng).

## Trạng thái

Đã hoàn thành khảo sát dữ liệu, khảo sát tài liệu và đánh giá ứng viên bài cơ sở. Bài cơ sở: Kirschstein & Sun, *The Merit of River Network Topology for Neural Flood Forecasting* (ICML 2024), dữ liệu LamaH-CE theo giờ.
