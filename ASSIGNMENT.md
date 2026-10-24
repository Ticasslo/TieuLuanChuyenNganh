# Phân công công việc

> Việc của từng thành viên theo từng bước, từ nay tới hết tiểu luận. Nội dung kỹ thuật của mỗi bước, điều kiện chuyển bước và số lần huấn luyện: `Document/01_Plan/03_Pipeline.md` Mục 7; lịch hai luồng: Mục 9.7; dàn ý báo cáo: Mục 9.8. Người 1 và người 2 tương ứng với ai do nhóm tự điền.

| | Thành viên | Mảng |
|---|---|---|
| Người 1 | *(điền tên)* | Dữ liệu, bộ nạp, đánh giá, demo D1–D3 (hàm dự báo, chạy CPU, bản trên laptop) |
| Người 2 | *(điền tên)* | Thư viện, các lõi mô hình, huấn luyện, giải thích mô hình, đưa demo lên VPS (D4) |

## 1. Trạng thái hiện tại

| Đã xong | Nơi ghi |
|---|---|
| Tải Extended LamaH-CE và kết quả của tác giả (Kaggle Dataset `lamah-ce-ext`) | `Document/03_Data/01_LamaHCE.md` Mục 6 |
| Tỉ lệ thiếu dữ liệu, persistence (NSE trung vị 0,35–0,37) | `01_LamaHCE.md` Mục 6.1 |
| Fork thư viện `Ticasslo/neuralhydrology`, nhánh `research` tại `dda45fc` (bước 0a) | `03_Pipeline.md` Mục 9.1 |
| A2: chạy lại trọng số tác giả, khớp NSE trung vị cả 6 run | `01_LamaHCE.md` Mục 6.3 |
| A1: hoàn thành khảo sát và tính ngưỡng đánh giá | [Thuyết minh A1](Workspace/02_Exploration/01_A1_LamaHCEExt_Explore.md#sec-25) |
| A3: persistence, sáu run A2, DLinear 3 hạt giống và chẩn đoán | [Thuyết minh A3](Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks.md#report-part-10) |
| Báo cáo tiến độ A1/A2/A3, cập nhật 23/10/2026 | [Báo cáo gửi GVHD](Document/01_Plan/04_ProgressReport_2026-10-23.md) |
| Bộ đánh giá: đã chấm A2/DLinear trên test, S0, hạn một ngày; khớp A3 và qua đối chiếu CSV | [Kết quả bộ đánh giá](Workspace/06_Evaluation/01_Evaluation_LamaHCEExt.md#10-kết-quả-thực-nghiệm-trên-kaggle) |
| Phân tích thuộc tính: 12 phép so, 31 trường dạng số; kết quả Kaggle và 143 kiểm CSV cục bộ đạt | [Kết quả theo lưu vực](Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin.md#11-kết-quả-thực-nghiệm-và-kiểm-chứng) |
| Ghi chú học BiasCast Phần 1–5 | `Document/06_Theory/03_…` tới `07_…` |

## 2. Việc theo từng bước

Mã việc trong bảng được giữ cố định để đối chiếu. **Ưu tiên Người 1 từ 23/10/2026:** hoàn tất báo cáo tiến độ → bộ đánh giá (#5), thử trên dự báo A2 và DLinear; tiếp tục tích hợp bộ nạp B2 (#4) sau A4/B1 của Người 2. Bản kiểm bộ nạp B2 độc lập đã đạt các phép kiểm, còn bước tích hợp và rà soát chung. Cột "Cần trước" ghi đầu vào của từng việc.

### Người 1 — dữ liệu, bộ nạp, đánh giá, demo D1–D3

| # | Bước | Việc | Ra | Cần trước |
|---|---|---|---|---|
| 1 | A1 | Notebook `Workspace/02_Exploration/01_A1_LamaHCEExt_Explore.py`: cấu trúc tệp, −999 toàn chuỗi, tỉ lệ thiếu, số mẫu hợp lệ (365 và 730 ngày, bộ lọc gốc và bộ lọc sửa), phân phối, bản đồ, so ECMWF HRES với tái phân tích và bước nhảy theo năm, so các kỳ, lưu vực bất thường (55, 742, 758, 800), ngưỡng 4 mức và ngưỡng lũ 1,5/2/5/10 năm | Tệp ngưỡng; số liệu và hình cho chương Dữ liệu | Dataset `lamah-ce-ext` |
| 2 | Báo cáo tiến độ | Báo cáo gửi GVHD (khoảng 25/10/2026): tái lập A2, dữ liệu A1, persistence và DLinear; giải thích thủy văn, chỉ số bằng ví dụ cho người chưa đọc bài | [Báo cáo tiến độ](Document/01_Plan/04_ProgressReport_2026-10-23.md) | A1, A2, A3 |
| 3 | A3 | Notebook `Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks.py`: persistence, PNSE, KGE (r, α, β) cho 6 run A2; DLinear 3 hạt giống; lưu vực NSE âm lớn so với danh sách A1 | Bảng mốc | Dự báo đã lưu của A2, kết quả A1 |
| 4 | B2 (bộ nạp) | Nhóm biến theo nguồn, sửa `_validate_samples`, nhóm Q (`qmean`, `qmax` quá khứ theo mm/ngày), cách (b) của bước H (ngày t+1…t+6 không nhận ECMWF HRES); kiểm thử đơn vị phần bộ nạp | Mã trong fork, nhánh `research` | A1 (số mẫu), B1 của người 2 |
| 5 | Bộ đánh giá | Mục 6 của pipeline: NSE, KGE, PNSE, Wilcoxon + Cohen's d + Holm; 4 mức; sự kiện lũ; kịch bản S0–S4; theo lưu vực; theo hạn; đọc tệp dự báo định dạng chung (Mục 9.5). Đã chấm và đối chiếu A2/DLinear trên test/S0/hạn một ngày. Phân tích đặc điểm lưu vực đã chạy trên các mốc A2/DLinear. Chấm thêm validation, S1–S4 và nhiều hạn khi có dự báo | [Mã đánh giá](Workspace/06_Evaluation/01_Evaluation_LamaHCEExt.py), [thuyết minh](Workspace/06_Evaluation/01_Evaluation_LamaHCEExt.md), [phân tích theo lưu vực](Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin.md) | Dự báo A2/DLinear; tệp ngưỡng A1 cho phần mức/sự kiện; thuộc tính gốc và Output A1/A3 cho phân tích lưu vực; bổ sung dự báo S0–S4 và nhiều hạn khi có |
| 6 | Đánh giá | Chạy bộ đánh giá cho bước B (cả S0–S4 trên validation để chọn B*), bước C, các bước mở rộng | Bảng, hình kết quả | Dự báo do người 2 xuất |
| 7 | Demo D1–D3 | Hàm dự báo tách khỏi mã huấn luyện → chạy CPU → giao diện trên laptop, đóng gói thành một khối độc lập (mã, trọng số, dữ liệu, hướng dẫn chạy) để đưa lên VPS (Mục 9.6) | Khối demo chạy được trên laptop, dùng làm bản dự phòng ngày bảo vệ | Trọng số mô hình chính thức, kết quả XAI |
| 8 | Báo cáo | Chương Dữ liệu, Đánh giá, Demo; slide phần tương ứng. Đã có bản thảo dữ liệu và thí nghiệm từ A1/A2/A3, đánh giá test/S0/hạn một ngày; bổ sung các bước sau khi có kết quả | [Chương 4](Document/07_Report/04_Data.md), [Chương 6](Document/07_Report/06_ExperimentsAndResults.md) | Kết quả các bước |

### Người 2 — thư viện, mô hình, huấn luyện, giải thích mô hình, VPS

| # | Bước | Việc | Ra | Cần trước |
|---|---|---|---|---|
| 1 | Bước 0 | Cài `mamba-ssm` trên Kaggle T4 (dự phòng `mambapy`); mô hình chung "Sequential Forecast" nhận lõi bất kỳ; các lõi GRU, Transformer, S4D, Mamba; kiểm thử đơn vị Mục 9.2 (nạp trọng số tác giả cho đầu ra trùng, không rò rỉ, đúng trục, số tham số lệch ≤ 1%) | Mã trong fork, wheel ở Output notebook `01_0_MambaSSM_Build` (gắn làm Input) | Fork `research` |
| 2 | A4 | Notebook `Workspace/03_Baseline/03_A4_LamaHCEExt_Retrain.py`: đo thời gian 1 epoch trên T4; huấn luyện lại mô hình tốt nhất của bài với hạt giống 111, 222, 333 | Độ lệch chuẩn giữa hạt giống; thời gian mỗi lần chạy để gắn ngày cho lịch | Thư mục Q (A2 Phần 4) |
| 3 | B1, B3, B* | B1: sửa lưu mô hình tốt nhất, Q mm/ngày, qmax quá khứ; B2 phần mô hình (nhúng theo nguồn, masked mean); B3: che dữ liệu hai mức; B*: chọn theo luật ghi sẵn; gắn mốc `freeze-C` | Trọng số, nhật ký, tệp dự báo | A4; bộ nạp B2 của người 1 |
| 4 | C | C1 tinh chỉnh 5 lõi × 8 cấu hình; C2 chạy chính thức 3 hạt giống | Tệp dự báo mọi cấu hình chính thức | `freeze-C` |
| 5 | Mở rộng | H (dự báo 1–7 ngày) → D (730 ngày) → E (biến thể) → F (CMAL) → GV, cắt theo `03_Pipeline.md` Mục 8.3 khi thiếu thời gian | Tệp dự báo | C xong |
| 6 | G | Integrated Gradients (Captum), dò trạng thái ẩn; xuất kết quả theo lưu vực cho demo | Hình, tệp kết quả XAI | Mô hình chính thức |
| 7 | Demo D4 | Đưa khối demo D3 vào trang web trên VPS Oracle (CPU ARM): cài PyTorch bản CPU cho ARM, Docker, tên miền và cách đưa ra Internet (Mục 9.6, 12); giữ máy chạy liên tục (VPS rảnh 7 ngày có thể bị thu hồi) | Đường dẫn công khai của demo | Khối demo D3 của người 1 |
| 8 | Báo cáo | Chương Cơ sở lý thuyết, Phương pháp, Thí nghiệm; slide phần tương ứng | | Kết quả các bước |

### Cả hai

- Học BiasCast Phần 6 (theo lưu vực) và Phần 7 (hạn chế, phản biện), cùng các nội dung học trước từng bước theo bảng ở `03_Pipeline.md` Mục 9.7 (DLinear trước A3, masked mean trước B2, đọc hình các chỉ số, Cohen's d và Holm trước đánh giá — `Document/06_Theory/08_EvaluationFigures.md`, CMAL trước F, Integrated Gradients trước G).
- Chương Mở đầu, Công trình liên quan, Kết luận; slide và hỏi đáp bảo vệ (`Document/05_Survey/02_LiteratureReview.md`).
- Xem lại kết quả của nhau trước khi ghi vào báo cáo.
- Khi bước C và các bước mở rộng cần nhiều GPU, người 1 chạy một phần thí nghiệm trên tài khoản Kaggle của mình bằng tệp cấu hình và mã commit do người 2 cung cấp (mỗi tài khoản một nhóm thí nghiệm).

## 3. Điểm bàn giao

| Từ | Đến | Thứ bàn giao | Khi nào |
|---|---|---|---|
| Người 1 | Người 2 | Tệp ngưỡng, số mẫu hợp lệ, danh sách lưu vực bất thường (A1) | Sau A1 |
| Người 2 | Người 1 | Thời gian 1 lần huấn luyện (A4) | Sau A4 — cả nhóm gắn ngày cho lịch |
| Người 1 | Người 2 | Bộ nạp B2 đã qua kiểm thử | Trước B2 |
| Người 2 | Người 1 | Tệp dự báo theo định dạng chung (`03_Pipeline.md` Mục 9.5), kèm mã cấu hình và mã commit thư viện | Sau mỗi bước huấn luyện |
| Người 2 | Người 1 | Trọng số, thông số chuẩn hóa, kết quả XAI cho demo | Trước demo D1 |
| Người 1 | Người 2 | Khối demo D3 đã chạy trên laptop | Trước demo D4 |

## 4. Quy ước làm việc chung

- **Đọc trước:** `Document/01_Plan/03_Pipeline.md`, `Document/01_Plan/02_BasePaper.md`, `CLAUDE.md` (quy tắc làm việc).
- **Notebook:** mỗi notebook Kaggle là một tệp `.py` tự đủ trong `Workspace/<NN_Việc>/`, tên `NN_<Bước>_<BộDữLiệu>_<Việc>.py`, chia cell `# %% Phần X`, có mục lục ở đầu; hình tải về đặt ở `Workspace/<NN_Việc>/Figures/`, tên bắt đầu bằng mã bước; chữ trong hình tiếng Anh, chú thích tiếng Việt.
- **Thư viện:** sửa trong `NeuralHydrology/` (nhánh `research`), mỗi thay đổi một commit; notebook cài theo mã commit; sau mốc `freeze-C` không đổi mã dùng chung (`03_Pipeline.md` Mục 9.1).
- **Kết quả:** trọng số và nhật ký nằm ở Output của bản Save Version (Save & Run All); notebook sau gắn Output đó làm Input, ghi số phiên bản cho kết quả chính thức; chia sẻ notebook cho thành viên kia; mỗi lần huấn luyện lưu kèm tệp cấu hình và mã commit.
- **Ghi chú:** kết quả và quyết định ghi vào `.md` tương ứng trong `Document/`, văn phong báo cáo, có nguồn.
- **Không** đưa khóa API vào mã; không dùng GPU trả phí; mỗi tài khoản Kaggle chạy một nhóm thí nghiệm.
