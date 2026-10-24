---
paths:
  - "**/*.md"
  - "Document/**"
  - "Diagrams/**"
---

# Quy tắc viết tài liệu (.md, Document/, Diagrams/)

## Văn phong

- **Văn phong báo cáo**: chỉ ghi thông tin đúng hiện hành; không ghi lịch sử sửa ("⚠️ sửa lại", "SAI", "ghi cũ", "lần N", gạch ngang chữ); không ghi "người dùng chốt/giao", "Claude chọn", "đợt tra ngày…", "mã đã sửa". Quyết định và ngày chốt chỉ ghi ở `CLAUDE.md`, `.claude/rules/` và bộ nhớ. Đính chính chỉ nói trong chat, tệp thì sửa thẳng thành nội dung đúng. (Ngoại lệ: tệp `CLAUDE.md`, `.claude/rules/`, `ASSIGNMENT.md` là tài liệu làm việc, được ghi ngày chốt.)
- **Ghi nguồn mọi khẳng định**: mỗi sự thật kèm nguồn (bài báo + mục/bảng, trang chính thức, tệp mã + dòng, output notebook + Version). Lựa chọn của đề tài ghi rõ là lựa chọn và lý do, không trình bày như sự thật của bài gốc.
- **Mã và ghi chú tách tệp**: `.md` chỉ chứa lý do, nguồn, kết quả — không chứa khối mã; mã đặt ở `.py` tương ứng.
- Gọi bài bằng tên ("BiasCast", "Kirschstein & Sun"), không dùng `#N`. Dự báo thời tiết ghi "ECMWF HRES"; "ECMWF" trần chỉ là tổ chức.
- Số thập phân trong văn bản tiếng Việt dùng dấu phẩy (0,7053); chữ trong hình dùng dấu chấm.
- Thuật ngữ lần đầu xuất hiện ghi kèm tiếng Anh trong ngoặc khi cần (VD "chu kỳ lặp lại (return period)").

## Báo cáo gửi GVHD và bản thảo báo cáo

- Người đọc là dân IT chưa đọc bài: khái niệm thủy văn, chỉ số đánh giá giải thích kỹ kèm ví dụ số; thuật ngữ học máy dùng thẳng; không ghi lý do "đã chốt"; không dồn thông số vào một đoạn.
- Người dùng tự soạn docx từ `.md` trong `Document/01_Plan/` và `Document/07_Report/` — viết `.md`, không dùng Artifact.
- Góp ý của GVHD: mục dữ liệu viết gạch đầu dòng theo năm phát hành + đoạn thảo luận thống kê để biện luận; kiến trúc phân biệt rõ khối giữ của bài gốc và khối thêm/sửa; không khẳng định điều không kiểm soát được.
- Báo cáo tiểu luận viết tiếng Việt; dàn ý ở `03_Pipeline.md` Mục 9.8.
- Khi học lý thuyết: dạy từ khái niệm nền, từng phần nhỏ, có ví dụ số tự tính và câu hỏi tự kiểm tra rồi mới sang phần sau.

## Hình và sơ đồ

- Mọi chữ **trong** hình, sơ đồ mới viết toàn tiếng Anh (số dấu chấm thập phân); nội dung và chú thích dưới hình tiếng Việt; không trộn hai thứ tiếng trong một hình. `Diagrams/ProjectDiagram.drawio` giữ nguyên tiếng Việt.
- `ProjectDiagram.drawio` là sơ đồ duy nhất của đề tài (không dùng Mermaid/PlantUML): phần I thứ tự thực hiện 0 → A → B → C → H, D, E, F, GV → đánh giá → G → báo cáo; phần II kiến trúc 5 tầng, cột trái khối bài gốc BiasCast đọc từ mã, cột phải khối đề tài sửa/thay (màu theo bước). Sơ đồ chỉ chứa phạm vi tiểu luận: không đưa khóa luận, thứ tự cắt (xem `03_Pipeline.md` Mục 8.3). Sinh bằng `Tools/Diagram_Build.py`: đổi pipeline (`03_Pipeline.md` Mục 4–8) => sửa Phần 2 của script rồi chạy lại, không sửa tay tệp drawio.
- Hình lý thuyết ở `Document/06_Theory/Figures/`, sinh bằng `Tools/Theory_Figures.py` (sơ đồ, bảng: SVG Python thuần; hình số liệu: matplotlib, seaborn khi cần); sinh xong mở trong trình duyệt kiểm chữ đè.
- Hình kết quả notebook nằm ở `Workspace/<NN_Việc>/<notebook>/Figures/`; `.md` trỏ tới đó, không chép sang `Document/`.

## Tên và đường dẫn

- Tên tệp, thư mục tiếng Anh; trong `Document/` đánh số `01_`, `02_`… theo thứ tự đọc, tệp mới đánh số tiếp; biên bản họp `Meeting_<ngày>.md`; PDF `AuthorYear_Venue_Topic.pdf`.
- Tên tệp `.md` trần (VD `03_Pipeline.md` Mục 6.2) là tệp trong `Document/`; `PaperResearch/...` tính từ gốc repo.
- **Hỏi người dùng trước khi tạo tệp `.md` mới** (trừ báo cáo cạnh notebook đã có trong kế hoạch).

## Vai trò từng tệp

| Tệp | Vai trò |
|---|---|
| `01_Plan/01_OverallPlan.md` | Kế hoạch tổng thể: tên đề tài, bài toán, bài cơ sở và dữ liệu, RiverMamba tham khảo, bằng chứng kiến trúc, mô hình so sánh, chỉ số, hạ tầng (Mục 8), hạn chế, tài liệu tham khảo |
| `01_Plan/02_BasePaper.md` | Giới thiệu BiasCast: bài toán, dữ liệu, mô hình, kết quả, lý do chọn (Mục 8), kiểm mã (Mục 9), kiến trúc đọc từ mã (Mục 11, 11.5 mã còn dùng), công trình trích dẫn (Mục 12) |
| `01_Plan/03_Pipeline.md` | **Tài liệu gốc**: Mục 1 RQ1–RQ6, đóng góp C1–C6, tính mới (1.3); 2 giai đoạn; 3 dữ liệu, bộ nạp; 4 kiến trúc (4.4 dự báo 1–7 ngày); 5 giao thức cùng tham số; 6 đánh giá (6.1 ba tầng chỉ số, 6.2 theo mức/sự kiện, 6.3 kịch bản vận hành, 6.6 theo hạn, 6.7 XAI); 7 ma trận thí nghiệm; 8 ngân sách, rủi ro, thứ tự cắt (8.3); 9 hạ tầng (9.1 mốc fork, 9.3 phân công, 9.5 xuất dự báo, 9.6 demo, 9.7 lịch, 9.8 dàn ý); 11 khóa luận; 12 việc chưa chốt |
| `01_Plan/04_ProgressReport_2026-10-23.md` | Báo cáo tiến độ gửi GVHD |
| `01_Plan/05_PresentationGuide_2026-10-25.md` | Kịch bản trình bày tiến độ cho hai thành viên: phân vai, tab cần mở, lời nói từng điểm dừng (A1–A4, bộ đánh giá, phân tích lưu vực), câu hỏi có thể gặp, lộ trình 15 phút; tài liệu nội bộ, không gửi GVHD |
| `02_Meetings/Meeting_<ngày>.md` | Biên bản họp với GVHD (chép lời bằng Whisper trên Kaggle, `Tools/Meeting_Transcribe.py`): góp ý riêng nhóm, góp ý chung, bảng áp dụng, điểm cần nói chính xác |
| `03_Data/01_LamaHCE.md` | Thực thi dữ liệu: Extended LamaH-CE (Mục 6: tải, % thiếu, tóm tắt A2 ở 6.3) và LamaH-CE gốc theo giờ (Mục 1–5, cho khóa luận) |
| `04_Ideas/01_Ideas.md` | Nhật ký ý tưởng theo ngày: nguồn đã đọc, sự thật mới, bảng ý tưởng Y1… (căn cứ, chi phí, vị trí đề xuất), mục chưa xác nhận. Ý tưởng chỉ vào pipeline khi người dùng đồng ý; đợt sau thêm Mục mới, không tra lại nguồn đã ghi |
| `05_Survey/01_Datasets.md` | Khảo sát 4 bộ dữ liệu (Extended LamaH-CE, CAMELS-US, Columbia Basin, WaterBench-Iowa) + 19 công trình, phân tích tổng hợp (7.1 Mamba/SSM, 7.2 đồ thị) |
| `05_Survey/02_LiteratureReview.md` | Khảo sát tài liệu theo yêu cầu GVHD: Mamba có cải thiện thật không (Mục 1), dự báo Q ở venue CNTT/AI (ưu tiên VN → ĐNA → châu Á), Q&A bảo vệ, demo; Mục 5 công trình venue CNTT/AI, Mục 6 công trình nền tảng |
| `05_Survey/03_PaperNotes.md` | Ghi chú đọc 44 PDF: nội dung, rút ra, áp dụng; Mục 7 ý tưởng Y9–Y15. Thêm PDF mới => thêm dòng |
| `06_Theory/01_ArchitectureTheory.md` | Lý thuyết LSTM, Transformer, GRU, Mamba (SSM, rời rạc hóa, Selective Scan, hai chiều, LOAN, đường cong lấp không gian); công thức có ví dụ số tự tính. Mục 4.7, 4.9, 4.10 là cơ chế riêng RiverMamba cho dữ liệu lưới (tham khảo) |
| `06_Theory/02_RiverMamba.md` | Tham khảo: đọc mã RiverMamba (cấu hình, loss, dữ liệu, cài `mamba-ssm` trên Colab). Không chạy |
| `06_Theory/03_BiasCast1_Overview.md` … `07_BiasCast5_Experiments.md` | Học BiasCast từng phần: 1 tổng quan (trạm, lưu vực, đơn vị, nguồn dữ liệu, lệch miền); 2 chỉ số (NSE, trung vị/CDF, 𝒟FC, 𝒟RA, KGE, persistence, PNSE, Wilcoxon); 3 dữ liệu phía lưu lượng, bốn mức, chu kỳ lặp lại; 4 mô hình (tensor, nhúng, LSTM, NSE*, huấn luyện, dừng sớm, tham số); 5 thí nghiệm (baseline, Cross-Domain, Encoder–Decoder, ECMWF HRES pha quá khứ, `qmean`, 6 run A2). Phần sau mỗi phần một tệp |
| `06_Theory/08_EvaluationFigures.md` | Đọc 10 hình chỉ số đánh giá trên dữ liệu giả lập (hình `Metrics_*.svg`, Phần 6 của `Tools/Theory_Figures.py`), ba tầng chỉ số, câu hỏi ôn |
| `07_Report/` | Bản thảo các chương báo cáo (`04_Data.md`, `06_ExperimentsAndResults.md` của người 1) |

- Thay đổi phạm vi hay quyết định => sửa cả `02_BasePaper.md`, `03_Pipeline.md` và sơ đồ. Không dùng `#N` trong hai tệp này.
- Quyết định kỹ thuật ổn định ghi ở tệp kế hoạch; chi tiết thực thi theo bước, bộ dữ liệu ghi ở báo cáo cạnh notebook hoặc `03_Data/`; phát hiện làm đổi hiểu biết tầng kế hoạch => sửa cả hai nơi.
