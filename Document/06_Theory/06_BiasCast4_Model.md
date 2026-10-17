# BiasCast — Phần 4: Bên trong mô hình và cách huấn luyện

Phần 4 của ghi chú học BiasCast (Phần 1: `03_BiasCast1_Overview.md`, Phần 2: `04_BiasCast2_Metrics.md`, Phần 3: `05_BiasCast3_Data.md`). Phần 1–3 đã trả lời "mô hình nhận gì, trả gì, chấm thế nào, dữ liệu ra sao"; Phần 4 mở hộp mô hình: số liệu biến thành tensor thế nào, đi qua những lớp nào, hàm mất mát và vòng huấn luyện chạy ra sao, mô hình được chọn thế nào. Ví dụ số lấy từ mô hình tốt nhất của bài — Sequential Forecast LSTM có `qmean` (NSE trung vị 0,7053 khi tái lập ở bước A2, [LamaH] Mục 6.3); các khối ở Phần 4 là khối dùng chung của mọi mô hình trong bài, khác nhau giữa các kiến trúc để Phần 5.

Ký hiệu nguồn:

- **[Bài]**: bản chữ `PaperResearch/PaperResearchPDF/BasePaper/Konold2026_HESS_BiasCast_FullText.md`, kèm số dòng.
- **[Mã]**: mã NeuralHydrology trong bản fork của tác giả, thư mục `PaperResearch/PaperResearchCode/BiasCast_NH/neuralhydrology/`, ghi tên tệp và số dòng.
- **[Cơ sở]**: `01_Plan/02_BasePaper.md`, kèm mục.
- **[LamaH]**: `03_Data/01_LamaHCE.md`, kèm mục.
- **[Kratzert]**: Kratzert và cs., *Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets*, HESS 23, 5089–5110, 2019, DOI `10.5194/hess-23-5089-2019`; PDF ở `PaperResearch/PaperResearchPDF/Evidence/Kratzert2019_HESS_NSEStarLoss.pdf`, kèm trang tạp chí.
- **[Lý thuyết]**: `06_Theory/01_ArchitectureTheory.md` (công thức LSTM có ví dụ số), kèm mục.

Nội dung Phần 4: (4.1) từ bảng số tới tensor; (4.2) lớp nhúng; (4.3) lõi LSTM; (4.4) lớp đầu ra và dropout; (4.5) hàm mất mát NSE*; (4.6) vòng huấn luyện; (4.7) validation, dừng sớm và chọn siêu tham số; (4.8) đếm tham số và so sánh cùng tham số; (4.9) tóm tắt và câu hỏi ôn.

---

### 4.1. Từ bảng số tới tensor

Phần 1 (Mục 1.4, 1.9) đã nêu hai ý: mọi biến được chuẩn hóa z-score, và một mẫu gồm 364 ngày quá khứ × 32 biến, 1 ngày dự báo × 5 biến, 33 thuộc tính. Mục này đi theo đúng thứ tự mã chạy, từ tệp số liệu tới tensor vào mạng rồi ngược lại ra m³/s.

| Bước | Làm gì | Ở đâu trong mã |
|---|---|---|
| 1 | Cắt chuỗi theo kỳ, lùi thêm 364 ngày khởi động; nhãn trong các ngày khởi động bị xóa thành NaN | [Mã] `datasetzoo/basedataset.py` 437–449 |
| 2 | Tính thống kê chuẩn hóa (chỉ kỳ train), chuẩn hóa | 753–765 |
| 3 | Loại mẫu không hợp lệ | `_validate_samples`, 861–919 |
| 4 | Gom mẫu thành lô, chuyển trục, đưa vào mạng | `modelzoo/inputlayer.py` 215–245 |
| 5 | Đổi kết quả về đơn vị thật khi đánh giá | `evaluation/tester.py` 250–256 |

#### Ý chính

*Ô trống được lưu bằng NaN:* "ô trống" là khái niệm (ngày đó không có số liệu); NaN ("Not a Number") là một giá trị số thực đặc biệt theo chuẩn IEEE 754 mà mã dùng để đánh dấu ô trống. NaN khác `null`/`None` (không có đối tượng nào): cột số của NumPy chỉ chứa được số thực, nên ô trống phải là NaN. Trong dữ liệu của bài, ngày thiếu `qmax`, `qmean` là ô để trống hoặc thiếu cả dòng, không có −999 (Phần 3, Mục 3.4): pandas đọc ô trống thành NaN, ngày thiếu dòng thành NaN khi mã điền đủ trục ngày (bước 1). Bộ nạp còn đổi mọi giá trị âm của nhãn thành NaN ([Mã] `datasetzoo/lamah.py` 228–229) — quy ước −999 của bộ LamaH-CE gốc. Mục này gọi chung là ô trống.

**Bước 1 — Khởi động.** Muốn dự báo ngày 1/1/2003 (ngày đầu kỳ train) thì cần 364 ngày quá khứ, tức từ 2/1/2002. Mã lấy thêm đoạn này cho **đầu vào**, nhưng xóa **nhãn** của nó thành NaN (dòng 448–449), để không ngày nào ngoài kỳ train được dùng làm đáp án. Mã cũng điền đủ mọi ngày trong khoảng (dòng 446–447): ngày nào tệp không có thì thành ô trống chứ không bị nhảy cóc, nên "364 dòng" luôn đúng là "364 ngày liền nhau". Như vậy ô trống ở bước này đến từ hai nguồn khác nhau: nhãn bị **cố ý xóa** trong 364 ngày khởi động (đầu vào những ngày này vẫn đủ số), và ngày **thật sự không có số liệu** trong tệp (VD trạm chưa đo). Nhãn trống trong ngày khởi động không gây hại, vì hàm mất mát chỉ lấy nhãn ngày cuối (Mục 4.5); còn đầu vào trống thì làm cả mẫu bị loại (bước 3).

**Bước 2 — Chuẩn hóa.** Với mỗi biến động (31 biến tái phân tích, `qmean`, 5 biến ECMWF HRES, nhãn `qmax`): một trung bình và một độ lệch chuẩn chung cho **mọi lưu vực, mọi ngày** của kỳ train, bỏ qua ô trống (dòng 764–765); rồi z = (x − trung bình) / độ lệch chuẩn (dòng 758). Với biến đầu vào, thống kê tính trên đoạn kỳ train **kể cả** 364 ngày khởi động, tức 2/1/2002–31/12/2009 ([LamaH] Mục 6.3); với nhãn `qmax`, các ngày khởi động đã bị xóa ở bước 1 nên thống kê chỉ gồm 1/1/2003–31/12/2009. Với 33 thuộc tính tĩnh: trung bình và độ lệch chuẩn tính **qua 451 lưu vực** (mỗi lưu vực một giá trị, dòng 725–736).

Thống kê được lưu vào tệp `train_data_scaler.yml` trong thư mục của run (`_dump_scaler`, dòng 254). Khi chạy validation và test, mã **đọc lại** tệp này chứ không tính lại (cờ `_compute_scaler` chỉ bật ở kỳ train, dòng 105–107). Lý do: nếu tính thống kê trên kỳ test thì thông tin kỳ test (VD kỳ test ướt hơn nên trung bình cao hơn) lọt vào đầu vào — một dạng rò rỉ.

Ví dụ số với ba biến của một lưu vực, một ngày (số chuẩn hóa của `qmean` là số thật của tác giả, [LamaH] Mục 6.3; hai biến còn lại là số giả định để minh họa):

| Biến | Giá trị thật | Trung bình kỳ train | Độ lệch chuẩn kỳ train | z |
|---|---|---|---|---|
| Mưa (giả định) | 12 mm | 3 | 6 | (12 − 3) / 6 = 1,5 |
| Nhiệt độ (giả định) | −2 °C | 8 | 8 | (−2 − 8) / 8 = −1,25 |
| `qmean` | 20 m³/s | 4,5778 | 16,1461 | (20 − 4,5778) / 16,1461 ≈ 0,96 |

Sau bước này mọi biến xoay quanh 0, phần lớn nằm trong khoảng −3 tới 3, dù gốc là mm, °C hay m³/s. Mạng không phải tự học "mưa 12 và nhiệt độ −2 cái nào lớn hơn" theo đơn vị.

**Bước 3 — Loại mẫu (chỉ ở kỳ train).** Hàm `_validate_samples` đánh dấu một mẫu (một lưu vực, một ngày t) là **không hợp lệ** nếu ([Mã] `basedataset.py` 893–917):

- chưa đủ 365 ngày lịch sử;
- **có bất kỳ ô trống nào** ở bất kỳ cột đầu vào động nào trong cả 365 ngày — hàm kiểm mọi cột (31 biến tái phân tích, `qmean`, 5 biến ECMWF HRES) trên đủ 365 ngày, kể cả những ô mô hình không dùng: ECMWF HRES của 364 ngày quá khứ, tái phân tích và `qmean` của ngày t (`basedataset.py` 577, 588, 622; [Cơ sở] Mục 11.4);
- nhãn ngày t trống;
- có thuộc tính tĩnh trống.

Chỉ nhãn **ngày t** được kiểm; nhãn của 364 ngày trước đó không được xem, nên nhãn bị xóa trong ngày khởi động (bước 1) không làm mất mẫu nào. Mẫu không hợp lệ bị bỏ khỏi danh sách mẫu huấn luyện. Hệ quả: lưu vực thiếu `qmean` một ngày thì mọi mẫu có ngày đó trong cửa sổ — tức 365 mẫu liên tiếp — bị bỏ (ví dụ đầy đủ ở Phần 3, Mục 3.4).

Ở validation và test, việc lọc này **tắt** (`basedataset.py` 617–630, chỉ gọi kiểm NaN khi `is_train`): mọi mẫu đủ 365 ngày lịch sử đều được chạy; mẫu có đầu vào trống cho ra dự báo trống, rồi khi tính chỉ số chỉ giữ ngày có cả quan trắc lẫn dự báo (`evaluation/metrics.py` 38–45). Lưu vực 55 thiếu `qmean` nên ở run có `qmean` mọi dự báo của nó đều trống, không có ngày nào được chấm ([LamaH] Mục 6.3). Cách thay thế (điền theo nhóm nguồn, che có chủ đích) là việc của bước B2, B3 (`01_Plan/03_Pipeline.md` Mục 7).

**Bước 4 — Lô và trục.** Bộ nạp gom B = 256 mẫu thành một lô. Mỗi biến là một tensor riêng có trục (lô, ngày, 1); lớp đầu vào ghép 32 biến theo trục cuối rồi **đổi chỗ hai trục đầu** (dòng `.transpose(0, 1)` trong `inputlayer.py`), vì LSTM của PyTorch mặc định nhận (ngày, lô, biến):

| Thứ | Kích thước | Ghi chú |
|---|---|---|
| 32 biến quá khứ, mỗi biến | (256, 364, 1) | sau khi ghép: (256, 364, 32) → đổi trục: (364, 256, 32) |
| 5 biến ECMWF HRES, mỗi biến | (256, 1, 1) | sau khi ghép, đổi trục: (1, 256, 5) |
| 33 thuộc tính | (256, 33) | không có trục ngày |
| Nhãn | (256, 365, 1) | cả 365 ngày, nhưng hàm mất mát chỉ lấy ngày cuối (`predict_last_n` = 1, Mục 4.5) |

**Bước 5 — Đổi ngược.** Mạng xuất ŷ ở thang chuẩn hóa. Khi đánh giá, mã đổi cả quan trắc lẫn dự báo về đơn vị nhãn bằng ŷ × độ lệch chuẩn + trung bình (`tester.py` 250–256), rồi mới tính NSE. Đơn vị nhãn là mm/ngày (Phần 1, Mục 1.4); muốn ra m³/s thì nhân diện tích: Q = q × A / 86,4.

Ví dụ (thống kê của `qmax` ở đây là số giả định; số thật đọc từ `train_data_scaler.yml` ở bước A1): trung bình 2,5 mm/ngày, độ lệch chuẩn 4,0; mạng xuất ŷ = 1,2 → 1,2 × 4,0 + 2,5 = 7,3 mm/ngày → lưu vực 200 km²: 7,3 × 200 / 86,4 ≈ 16,9 m³/s.

**Tóm tắt ý chính:** lấy thêm 364 ngày khởi động (xóa nhãn) → chuẩn hóa bằng thống kê kỳ train, lưu ra tệp, kỳ sau đọc lại → kỳ train bỏ mọi mẫu có ô trống trong cửa sổ, kỳ test vẫn chạy nhưng ra dự báo trống → gom lô, đổi trục thành (ngày, lô, biến) → đầu ra đổi ngược về mm/ngày rồi mới chấm.

#### Chi tiết kỹ thuật

*(a) Một bộ thống kê chung cho mọi lưu vực.* Chuẩn hóa không theo từng lưu vực: lưu vực lớn vẫn có z lớn, lưu vực nhỏ vẫn có z nhỏ. Thông tin "lưu vực này to hay nhỏ, ướt hay khô" nhờ vậy còn nằm trong số liệu, và mô hình chung cho 451 lưu vực cần nó. Mặt trái là biến có thang lệch giữa các lưu vực như `qmean` theo m³/s bị dồn về một khoảng hẹp ở đa số lưu vực nhỏ (Phần 3, Mục 3.4, bảng hai lưu vực).

*(b) Nhãn cũng được chuẩn hóa.* Hàm mất mát so ŷ với nhãn **đã chuẩn hóa**; vì sao điều này không làm sai NSE* là việc của Mục 4.5.

*(c) Ô trống không bị điền.* Mã không điền ô trống bằng 0 hay trung bình; mẫu chứa ô trống bị bỏ cả mẫu (bước 3). Thư viện có tham số `nan_handling_method` để lớp đầu vào xử lý ô trống (VD trung bình có che), nhưng chỉ lớp đầu vào đọc tham số này (`modelzoo/inputlayer.py`), còn bộ nạp `basedataset.py` không đọc, nên mẫu có ô trống vẫn bị bỏ trước khi tới lớp đầu vào. Bài BiasCast giữ nguyên cách bỏ mẫu này và chỉ nêu ở phần thảo luận rằng khi vận hành cần tính tới trạm hỏng và nên áp dụng cách huấn luyện của Gauch và cs. (HESS 29, 6221–6235, 2025) ([Bài] dòng 350). Đề tài làm việc đó ở bước B2 (chia đầu vào quá khứ thành nhóm theo nguồn, mỗi nhóm một lớp nhúng, gộp bằng trung bình có che ô trống; sửa bộ lọc để chỉ yêu cầu nhãn ngày t và thuộc tính hợp lệ) và B3 (cố ý che ngẫu nhiên từng nguồn khi huấn luyện) (`01_Plan/03_Pipeline.md` Mục 3.2–3.4). Sau khi sửa: mẫu có đầu vào thiếu vẫn được học và vẫn cho ra dự báo, dựa trên các nguồn còn lại; mẫu thiếu nhãn ngày t vẫn bị bỏ khỏi huấn luyện vì không có đáp án để tính sai số; thiếu nguồn thì dự báo kém hơn, mức kém đi đo bằng kịch bản mất Q S1 (mất 1, 3, 7 ngày cuối) và S2 (mất cả cửa sổ) (`01_Plan/03_Pipeline.md` Mục 6.3); nếu mọi nhóm cùng trống ở một ngày, trung bình có che của thư viện điền một giá trị cố định cho ngày đó.

**Câu hỏi tự kiểm tra (Mục 4.1):**

1. Vì sao thống kê chuẩn hóa chỉ tính trên kỳ train rồi lưu ra tệp, mà không tính lại trên kỳ test?
2. Một ngày `qmean` = 50 m³/s. Dùng thống kê của tác giả (trung bình 4,5778, độ lệch chuẩn 16,1461), tính z. Một suối nhỏ có `qmean` = 1 m³/s thì z bằng bao nhiêu? Hai số z này nói gì về cách mô hình "thấy" suối nhỏ?
3. Một lưu vực thiếu `qmean` đúng một ngày, 10/3/2005. Những mẫu nào của lưu vực này bị loại? Khoảng bao nhiêu mẫu?
4. Mạng xuất ŷ = −0,3; thống kê `qmax` (giả định) trung bình 2,5, độ lệch chuẩn 4,0; lưu vực 500 km². Tính qmax theo mm/ngày và m³/s.
5. Vì sao lớp đầu vào phải đổi trục (256, 364, 32) thành (364, 256, 32)?

---

### 4.2. Lớp nhúng

Lớp nhúng (embedding) là bước đầu tiên bên trong mạng: nó biến các biến đầu vào đã chuẩn hóa thành một vector có số chiều cố định, rồi mới đưa vào LSTM. Trong Sequential Forecast LSTM có **bốn** lớp nhúng ([Mã] `modelzoo/sequential_forecast_lstm.py` 43–47; [Cơ sở] Mục 11.2):

| Lớp nhúng | Vào | Ra | Tham số |
|---|---|---|---|
| Biến động quá khứ (hindcast) | 32 biến | 16 | 32 × 16 + 16 = 528 |
| Biến động ngày t (forecast) | 5 biến ECMWF HRES | 16 | 5 × 16 + 16 = 96 |
| Thuộc tính, nhánh hindcast | 33 thuộc tính | 16 | 33 × 16 + 16 = 544 |
| Thuộc tính, nhánh forecast | 33 thuộc tính | 16 | 544 |

Mỗi ngày, vector động 16 chiều được **nối** với vector thuộc tính 16 chiều thành 32 chiều; vector thuộc tính giống nhau ở mọi ngày vì thuộc tính không đổi theo thời gian (`inputlayer.py`, đoạn `unsqueeze(0).repeat(...)` rồi `torch.cat`). Kết quả: pha quá khứ (364, 256, 32), pha ngày t (1, 256, 32) — **cùng 32 chiều**, nên cùng một LSTM chạy được qua cả hai pha. Mã bắt buộc điều này: hai lớp nhúng hindcast và forecast phải ra cùng kích thước, nếu không sẽ báo lỗi (dòng 46–47).

#### Ý chính

**Bước 1 — Lớp nhúng làm phép tính gì.** Lớp nhúng "đơn giản" của bài là **một lớp tuyến tính**: y = W·x + b. Ví dụ thu nhỏ với 3 biến động → 2 chiều và 2 thuộc tính → 1 chiều (số giả định):

- Biến động một ngày (đã chuẩn hóa): x = (mưa 1,2; nhiệt độ −0,5; `qmean` 0,3).
- W = [[0,5; 0,2; 1,0], [−0,3; 0,4; 0,1]], b = (0,1; 0).
- Chiều 1: 0,5 × 1,2 + 0,2 × (−0,5) + 1,0 × 0,3 + 0,1 = 0,6 − 0,1 + 0,3 + 0,1 = **0,9**.
- Chiều 2: −0,3 × 1,2 + 0,4 × (−0,5) + 0,1 × 0,3 + 0 = −0,36 − 0,2 + 0,03 = **−0,53**.
- Thuộc tính s = (0,8; −1,0), W_s = [0,5; 0,25], b_s = 0: 0,4 − 0,25 = **0,15**.
- Vector vào LSTM ngày này: (0,9; −0,53; 0,15). Ngày hôm sau hai số đầu đổi theo thời tiết, số cuối vẫn 0,15.

Mỗi chiều đầu ra là một **tổ hợp có trọng số** của mọi biến: chiều 1 ở ví dụ chủ yếu "đọc" mưa và lưu lượng, chiều 2 đọc mưa với dấu ngược. Các trọng số này được học cùng lúc với LSTM.

**Bước 2 — Vì sao cần lớp nhúng.** Lớp đầu tiên của LSTM tự nó đã là phép nhân ma trận với đầu vào, nên đặt thêm một lớp tuyến tính phía trước **không** làm mô hình biểu diễn được nhiều hơn (hai phép tuyến tính liên tiếp vẫn là một phép tuyến tính). Lý do nằm ở cấu trúc nhiều nguồn ([Bài] dòng 98; ý thứ hai là suy luận của ghi chú này từ cấu trúc mã):

- Pha quá khứ có 32 biến, pha ngày t chỉ có 5 biến ECMWF HRES. Một LSTM không nhận được hai kích thước khác nhau; hai lớp nhúng **riêng** đưa cả hai về cùng 16 chiều.
- Hai lớp nhúng riêng nghĩa là mỗi nguồn có bộ trọng số riêng: mưa ERA5-Land và mưa ECMWF HRES đi qua hai W khác nhau, nên mạng có chỗ để học rằng "mưa ECMWF HRES 10 mm" không hoàn toàn giống "mưa tái phân tích 10 mm" — đúng cái lệch miền bài muốn bù (Phần 1, Mục 1.11).
- Bài nêu học chuyển giao là một động cơ chính (*"a primary motivation"*): chiếu các nguồn có số biến khác nhau vào một không gian chung, nên có thể huấn luyện trước trên tái phân tích rồi chỉ tinh chỉnh lớp nhúng cho dự báo ([Bài] dòng 98; thí nghiệm học chuyển giao để Phần 5).

**Bước 3 — Nhúng đơn giản và nhúng phức tạp.** Bài so hai loại ([Bài] dòng 172):

| Loại | Cấu trúc theo bài | Cấu trúc theo mã | Dropout trong lớp nhúng |
|---|---|---|---|
| Đơn giản | một lớp 16 chiều, có tanh | một lớp tuyến tính 16 chiều, **không** tanh | 0 |
| Phức tạp | ba lớp 30, 20, 64, có tanh | tuyến tính 30 → tanh → tuyến tính 20 → tanh → tuyến tính 64 | 0 |

Lớp `FC` của thư viện chỉ chèn hàm kích hoạt **giữa** các lớp, không chèn sau lớp cuối ([Mã] `modelzoo/fc.py`: nhánh một lớp chỉ có `nn.Linear`, nhánh nhiều lớp thêm kích hoạt và dropout sau từng lớp ẩn rồi kết thúc bằng `nn.Linear`). Vì vậy nhúng "đơn giản" trong mã là tuyến tính thuần, khác câu "tanh activation" ở Mục 2.2.6 của bài; chính bài ở chỗ khác lại gọi nó là "simple (linear)" ([Bài] dòng 280 và chú thích Hình 8). Dropout trong lớp nhúng bằng 0 ở mọi thí nghiệm ([Bài] dòng 290).

Nếu thêm tanh vào ví dụ ở bước 1: tanh(0,9) ≈ 0,72, tanh(−0,53) ≈ −0,49 — số bị nén về khoảng (−1, 1), và phép biến đổi thành phi tuyến.

**Bước 4 — Kết quả của bài.** Ở baseline, Encoder–Decoder và Sequential Forecast LSTM, nhúng đơn giản cho NSE **nhỉnh hơn** nhúng phức tạp; riêng học chuyển giao thì ngược lại ([Bài] dòng 280, 288). Bài giải thích: nhúng phức tạp có nhiều tham số so với lượng dữ liệu nên dễ học thuộc nhiễu của kỳ train, lại không có dropout; còn khi học chuyển giao chỉ tinh chỉnh lớp nhúng, lớp nhúng phải gánh toàn bộ việc thích nghi miền nên cần phi tuyến ([Bài] dòng 290). Đây là giải thích của tác giả, chưa có thí nghiệm riêng kiểm chứng; với một hạt giống, chênh lệch "nhỉnh hơn" cũng chưa biết có vượt mức nhiễu giữa các lần huấn luyện hay không (bước A4 đo mức nhiễu này).

**Tóm tắt ý chính:** lớp nhúng = tổ hợp có trọng số của các biến → mỗi nguồn một lớp nhúng riêng, đưa 32 biến và 5 biến về cùng 16 chiều, nối thêm 16 chiều thuộc tính → cùng một LSTM chạy qua cả hai pha → trong bài, nhúng đơn giản (tuyến tính) cho NSE nhỉnh hơn nhúng phức tạp, trừ học chuyển giao.

#### Chi tiết kỹ thuật

*(a) Khởi tạo trọng số.* Lớp `FC` khởi tạo W theo phân phối đều trong khoảng (−√(3/n), √(3/n)) với n là số đầu vào, b = 0 ([Mã] `modelzoo/fc.py`, hàm `_reset_parameters`). Với lớp nhúng hindcast n = 32: khoảng (−0,31; 0,31).

*(b) Ý nghĩa cho đề tài.* Ở bước B2, lớp nhúng được tách theo **nguồn** (mỗi nguồn khí tượng một lớp nhúng, gộp bằng trung bình có che ô trống) để mẫu thiếu một nguồn vẫn dùng được; ở bước C, mọi lõi (LSTM, Mamba, S4D…) nhận cùng đầu ra lớp nhúng nên chênh lệch kết quả chỉ đến từ lõi (`01_Plan/03_Pipeline.md` Mục 3, 4).

**Câu hỏi tự kiểm tra (Mục 4.2):**

1. Tính lại ví dụ ở bước 1 với x = (mưa 0; nhiệt độ 1,0; `qmean` −0,2), cùng W, b. Vector vào LSTM ngày đó là gì (thuộc tính giữ nguyên)?
2. Vì sao Sequential Forecast LSTM cần hai lớp nhúng động riêng, không dùng chung một lớp cho cả pha quá khứ và pha ngày t?
3. Đếm tham số của lớp nhúng phức tạp cho 32 biến động (32 → 30 → 20 → 64). So với 528 của nhúng đơn giản thì gấp bao nhiêu lần?
4. Bài viết "simple embedding … 16 hidden units and tanh", mã lại không có tanh. Khi viết báo cáo, đề tài nên ghi theo bên nào, và vì sao?
5. "Nhúng đơn giản cho NSE nhỉnh hơn nhúng phức tạp" — cần thêm gì để kết luận này chắc chắn?
