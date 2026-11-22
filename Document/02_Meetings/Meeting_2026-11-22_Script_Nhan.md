# Kịch bản thuyết trình chi tiết 10 phút — Báo cáo tiến độ 2 tuần (07/11 – 21/11/2026)

> **Trước khi vào họp (chỉ cần chạy đúng 1 lệnh này để bật sẵn Demo D3 ở nền):**
> Mở PowerShell trong thư mục `C:\TLCN\TieuLuanChuyenNganh\Workspace\07_Demo\03_D3_LamaHCEExt_Demo`, dán lệnh sau rồi mở sẵn `http://127.0.0.1:8053/` trên trình duyệt:
> ```powershell
> $env:PYTHONUTF8 = "1"
> .\.venv\Scripts\python.exe .\app.py --bundle .\downloads\D3_CHDG_CPU_V13 --raw .\downloads\lamah-ce-ext\lamah_ce_ext\LamaH_extended --h-python .\downloads\.venv-h\Scripts\python.exe --port 8053
> ```

---

## Bước 1 (`00:00 – 01:00`) · Mở đầu: 2 tuần qua em đã làm trọn chuỗi việc gì?

- **Thao tác màn hình:** Chưa cần mở ảnh, nhìn thẳng màn hình/thầy và nói tự nhiên (có thể để sẵn trang [ASSIGNMENT.md](../../ASSIGNMENT.md) nếu thầy muốn xem phân công).
- **Lời nói chi tiết:**
  > "Dạ em chào thầy ạ. Hôm nay em xin báo cáo phần công việc em đã hoàn thành trong 2 tuần vừa qua, tính từ sau khi bạn Thắng chốt xong bước huấn luyện C2.
  >
  > Trong 2 tuần này, em tập trung khép kín **4 mảnh ghép liên tiếp nhau** của hệ thống:
  > - **Thứ nhất là phần Đánh giá chuẩn hóa:** em chấm toàn bộ các kịch bản thiếu dữ liệu thực tế cho 5 lõi C2, chấm dự báo nhiều hạn 1 đến 7 ngày của bước H, chấm mở rộng bước D và chạy phân tích tương quan theo 31 đặc trưng lưu vực.
  > - **Thứ hai là phần Thực nghiệm trọn gói bước D:** em trực tiếp huấn luyện đủ 9 mô hình với cửa sổ lịch sử dài gấp đôi — từ 1 năm lên 2 năm, tức 730 ngày — và dựng bộ biểu đồ so sánh đa tiêu chí.
  > - **Thứ ba là phần Kiểm chứng triển khai và Demo D3:** em tách hàm suy luận khỏi mã huấn luyện, kiểm tra độ chính xác trên cả GPU lẫn CPU, rồi dựng hoàn chỉnh ứng dụng web 5 trang chạy thật trên laptop, vừa ghép xong luôn cả trang Giải thích mô hình của bước G.
  > - **Cuối cùng là cập nhật trọn vẹn Chương 6 báo cáo**, gồm cả phần bước D và phần giải thích mô hình bước G.
  >
  > Sau đây em xin đi lần lượt từng phần, bắt đầu từ khâu đánh giá."

---

## Bước 2 (`01:00 – 02:45`) · Đánh giá kịch bản S0–S4 (C2), nhiều hạn (H) và phân tích theo lưu vực

- **Thao tác màn hình:**
  - Nếu thầy muốn xem danh mục notebook đánh giá, mở lướt [01_Evaluation_LamaHCEExt.md](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md).
  - **Đến đoạn nói về đặc trưng lưu vực (`01:55`), mở ngay ảnh sau:**
    - **Ảnh cần mở:** [BasinAnalysis_C2_vs_LSTM_static_delta_NSE_rho.png](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Archive/Version3/Figures/BasinAnalysis_C2_vs_LSTM_static_delta_NSE_rho.png)
    - **Thuộc notebook:** `02_Evaluation_LamaHCEExt_ByBasin` (**Version 3**, [mã/báo cáo](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md))
- **Lời nói chi tiết:**
  > "Ở mảng đầu tiên là **đánh giá**, mục tiêu của em là bảo đảm mọi mô hình của cả nhóm — từ C2, H cho tới D — đều được chấm trên **cùng một mặt nạ ngày, cùng đơn vị mm/ngày và cùng bộ ngưỡng lũ**, không có chuyện so lệch tập dữ liệu.
  >
  > Cụ thể, em đã chạy notebook **`04_C2_LamaHCEExt_Scenarios` Version 1**: kiểm tra lại đúng 24 run C2 ở điều kiện đầy đủ S0 khớp sai số **0 mm/ngày**, rồi tạo thêm **144 tệp dự báo** cho 6 kịch bản mất hoặc trễ dữ liệu ngoài thực tế — ví dụ trễ số đo lưu lượng 1, 3, 7 ngày hay mất dự báo thời tiết.
  >
  > Sau đó, em nâng cấp bộ đánh giá dùng chung **`01_Evaluation_LamaHCEExt` từ Version 10 lên Version 12 rồi Version 13**: chạy qua **212 trên 212 kiểm thử tự động**, chấm trọn **168 tệp dự báo nhiều hạn 1–7 ngày của bước H** trên hơn 651 nghìn cặp ngày–lưu vực và chấm luôn phân vùng **bước D** trên 451 lưu vực.
  >
  > *(Mở ảnh `BasinAnalysis_C2_vs_LSTM_static_delta_NSE_rho.png` — thuộc notebook `02_Evaluation_LamaHCEExt_ByBasin` Version 3)*
  >
  > Thưa thầy, thay vì chỉ nhìn một con số NSE trung vị cho cả 451 lưu vực, em chạy thêm notebook **`02_Evaluation_LamaHCEExt_ByBasin` Version 3** để kiểm tra xem mỗi kiến trúc mạnh ở kiểu lưu vực nào. Trên bản đồ nhiệt này, mỗi hàng là một đặc trưng tự nhiên của lưu vực — như độ cao, lượng tuyết, độ khô hạn — còn mỗi cột là mức chênh NSE của từng lõi so với LSTM chuẩn cùng hạt giống.
  >
  > *(Chỉ vào vùng màu xanh ở dòng `frac_snow` và vùng màu đỏ ở dòng `arid_1`)*
  >
  > Kết quả cho thấy rất nhất quán qua cả 3 hạt giống: các lõi **Transformer, S4D và Mamba bản mở rộng x3** có xu hướng cải thiện rõ hơn ở các **lưu vực có nhiều tuyết (`frac_snow` dương đồng nhất)**, trong khi lại kém ưu thế hơn LSTM ở các **lưu vực khô hạn (`arid_1` âm đồng nhất)**. Đây là cơ sở rất quan trọng để mình giải thích hành vi mô hình theo địa lý thủy văn chứ không chỉ so điểm trung bình."

---

## Bước 3 (`02:45 – 06:00`) · Trọng tâm: Bước D (tăng cửa sổ 365 → 730 ngày) — Vì sao làm, làm thế nào và kết quả

### 3.1. (`02:45 – 03:55`) Vì sao làm bước D, làm thế nào và kết quả tổng quan?

- **Thao tác màn hình:**
  - **Mở ảnh:** [D_Overview_Accuracy.png](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/Figures/D_Overview_Accuracy.png)
  - **Thuộc notebook:** `17_D_LamaHCEExt_Compare` (**Version 3**, [notebook chính thức](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/17_D_LamaHCEExt_Compare.ipynb) · [báo cáo](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/17_D_LamaHCEExt_Compare.md))
- **Lời nói chi tiết:**
  > "Sang mảng thứ hai là **bước D — thí nghiệm mở rộng độ dài chuỗi lịch sử từ 365 ngày lên 730 ngày**.
  >
  > **Trước hết, vì sao nhóm phải làm bước D?** Ở bước C2, mọi mô hình chỉ nhìn quá khứ **365 ngày (1 năm)** để dự báo ngày tiếp theo. Nhưng trong thủy văn, nhiều quá trình chậm như tích–tan tuyết ở vùng núi cao hay trữ lượng nước ngầm ở vùng khô hạn có thể kéo dài hơn một năm. Hơn nữa, Mamba và S4D thuộc họ **State Space Model**, vốn nổi tiếng với kỳ vọng xử lý chuỗi dài tốt hơn LSTM. Nếu không thử kéo dài cửa sổ lên **2 năm (730 ngày)** thì mình chưa trả lời được câu hỏi: *liệu cho nhìn xa gấp đôi, các mô hình SSM có vượt lên rõ rệt so với LSTM hay không?*
  >
  > **Về cách làm và nguồn gốc dữ liệu để so sánh công bằng:** Em chọn đúng **3 lõi đại diện ở mức cùng quy mô tham số (~85 nghìn tham số)** từ C2 là **LSTM, Mamba và S4D**. Em giữ nguyên 100% mã thư viện `freeze-C`, giữ nguyên siêu tham số, giữ nguyên 30 epoch và 3 hạt giống `111, 222, 333` — **chỉ đổi duy nhất `seq_length` từ 365 sang 730 ngày** (gồm 729 ngày quá khứ + 1 ngày dự báo). Đặc biệt, nhờ bộ lọc mẫu đã sửa từ bước B2, cả bản 365 ngày (lấy từ 9 run C2 của Thắng) và bản 730 ngày (9 run em huấn luyện mới trong thư mục `16_D_Train/`) đều được chấm trên **đúng cùng một tập `657.347` cặp ngày–lưu vực của cả 451 lưu vực**, tránh hoàn toàn sai lệch do khác tập mẫu. Vì Mamba và S4D chạy 730 ngày vượt giới hạn 12 giờ một phiên của Kaggle, em dùng cơ chế lưu và nối trạng thái huấn luyện (`_Cont`) để chạy liền mạch qua 2 phiên; cả **9/9 run** đều hoàn tất với `0` bước bị tràn gradient.
  >
  > **Bây giờ nhìn vào ảnh `D_Overview_Accuracy.png`** *(thuộc notebook `17_D_LamaHCEExt_Compare` Version 3 — đã kiểm khớp `34/34` tệp manifest)*:
  > - *(Chỉ vào khung bên trái)* Đây là **NSE trung vị trên tập test** của từng lõi khi đặt cặp 365 ngày (cột nhạt) cạnh 730 ngày (cột đậm), chấm tròn là từng hạt giống.
  > - *(Chỉ vào khung bên phải)* Đây là **mức chênh `730 − 365`** đặt cạnh **dải nhiễu hạt giống (`±0,0079`)** đo từ bước B0 để biết sự chênh lệch có lớn hơn nhiễu ngẫu nhiên hay không.
  >
  > Kết quả thực nghiệm rất rõ và trung thực: cả ba lõi đều có biến động NSE nằm **trong dải nhiễu hạt giống**:
  > - **LSTM** giảm nhẹ `−0,0016` (`0,7178 → 0,7162`).
  > - **Mamba** giảm nhẹ `−0,0023` (`0,7204 → 0,7180`).
  > - Riêng **S4D** là lõi **duy nhất cải thiện nhất quán** ở cả NSE (`+0,0053`, tức `0,6933 → 0,6986`, đạt `+0,67 ×` nhiễu) lẫn PNSE (`+0,0117`).
  >
  > **Vì sao lại có sự khác biệt này?** Điều này khớp độc lập với kết quả giải thích mô hình ở bước G: **LSTM và Mamba** có cơ chế chọn lọc nên đã dồn 90% sự chú ý vào **1–2 tháng gần nhất**, thành ra 365 ngày là đã bão hòa; còn **S4D** là mô hình không chọn lọc, trải bộ nhớ rộng tới **~200 ngày**, nên khi nới cửa sổ lên 730 ngày thì S4D là lõi tận dụng được thêm ngữ cảnh xa."

### 3.2. (`03:55 – 04:40`) Chi phí tính toán tăng bao nhiêu so với lợi ích thu về?

- **Thao tác màn hình:**
  - **Mở ảnh:** [D_Cost_Tradeoff.png](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/Figures/D_Cost_Tradeoff.png)
  - **Thuộc notebook:** `17_D_LamaHCEExt_Compare` (**Version 3**)
  - *(Ảnh dự phòng nếu thầy hỏi quá trình hội tụ qua 30 epoch: [D_Curves_365_vs_730.png](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/Figures/D_Curves_365_vs_730.png) — cùng notebook `17_D_LamaHCEExt_Compare` V3.)*
- **Lời nói chi tiết:**
  > "Tiếp theo, để biết có đáng đánh đổi tài nguyên hay không, em mời thầy xem ảnh **`D_Cost_Tradeoff.png`** *(cùng notebook `17_D_LamaHCEExt_Compare` Version 3)*.
  >
  > *(Chỉ trục ngang là tổng giờ huấn luyện GPU T4, trục dọc là NSE test, nhãn là VRAM đỉnh, và mũi tên nối từ 365 sang 730)*
  >
  > Toàn bộ các mũi tên đều **đi ngang sang phải rất dài nhưng hầu như không đi lên**:
  > - **LSTM** tăng thời gian từ `2,0` lên `3,9` giờ, VRAM từ `2,4` lên `4,1` GB.
  > - **Mamba** tăng từ `5,2` lên `10,0` giờ, VRAM từ `1,8` lên `3,1` GB.
  > - **S4D** tăng từ `6,5` lên `12,9` giờ, VRAM từ `2,4` lên `4,2` GB.
  >
  > **Ý nghĩa rút ra rất dứt khoát:** tăng độ dài chuỗi lên gấp đôi làm **chi phí huấn luyện và bộ nhớ tăng gần gấp đôi**, trong khi độ chính xác trung vị không vượt nổi nhiễu hạt giống. Do đó với bài toán dự báo hạn 1 ngày trên Extended LamaH-CE, **cửa sổ 365 ngày là lựa chọn tối ưu nhất để triển khai thực tế**."

### 3.3. (`04:40 – 05:25`) Mổ xẻ theo 451 lưu vực: vì sao trung vị đứng yên nhưng từng vùng lại tách 2 hướng?

- **Thao tác màn hình:**
  - **Mở ảnh:** [BasinAnalysis_D_static_delta_NSE_rho.png](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Archive/Version3/D/Figures/BasinAnalysis_D_static_delta_NSE_rho.png)
  - **Thuộc notebook:** `02_Evaluation_LamaHCEExt_ByBasin` (**Version 3**, phân vùng `D/`)
- **Lời nói chi tiết:**
  > "Nhưng nếu chỉ nhìn trung vị toàn tập thì có thể bỏ sót tác động theo vùng. Vì vậy em chạy thêm phân vùng `D/` trong notebook **`02_Evaluation_LamaHCEExt_ByBasin` Version 3**, tính hệ số tương quan hạng Spearman giữa hiệu `ΔNSE (730 − 365)` của từng lưu vực với 31 thuộc tính tĩnh trên cả 451 trạm.
  >
  > *(Chỉ cột `S4D` rồi so sang cột `LSTM` trên ảnh `BasinAnalysis_D_static_delta_NSE_rho.png` — màu xanh là 730 tốt hơn khi thuộc tính đó cao, màu đỏ là ngược lại)*
  >
  > Ta thấy 3 lõi **tách hẳn thành 2 hướng đối lập nhau**:
  > - **S4D** (giữ dấu đồng nhất trên cả 3 seed, và Mamba nghiêng nhẹ cùng chiều) cải thiện mạnh hơn ở các **lưu vực khô hạn, bốc thoát hơi cao và diện tích lớn** (`arid_1 ρ = +0,208`, `et0_mean +0,197`), nhưng giảm tương đối ở vùng núi cao/nhiều tuyết.
  > - Ngược lại, **LSTM** lại cải thiện rõ ở các **lưu vực có băng hà và tỷ phần tuyết cao** (`glac_fra +0,141`, `frac_snow +0,139`).
  >
  > Như vậy, cửa sổ 730 ngày không phải hoàn toàn vô tác dụng, mà nó **phân hóa theo chế độ thủy văn và cơ chế nhớ của từng kiến trúc**."

### 3.4. (`05:25 – 06:00`) Kiểm chứng trực quan bằng đường quá trình dòng chảy (Hydrograph)

- **Thao tác màn hình:**
  - **Mở ảnh:** [D_Hydrograph_Diagnostics.png](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/Figures/D_Hydrograph_Diagnostics.png)
  - **Thuộc notebook:** `17_D_LamaHCEExt_Compare` (**Version 3**)
- **Lời nói chi tiết:**
  > "Cuối cùng của phần bước D, em mở ảnh **`D_Hydrograph_Diagnostics.png`** *(cùng notebook `17_D_LamaHCEExt_Compare` Version 3)*. Ảnh này lấy thẳng các lưu vực cải thiện nhiều nhất và suy giảm nhiều nhất để vẽ chồng đường **lưu lượng quan trắc thật** với đường dự báo **365 ngày** và **730 ngày** theo từng ngày.
  >
  > *(Chỉ nhanh vào các đỉnh lũ và phần chân lũ)*
  >
  > Nhìn vào đây người nghe thấy ngay bằng mắt thường sự khác biệt giữa 365 và 730 ngày nằm ở những đợt đỉnh lũ cụ thể chứ không phải con số trừu tượng."

---

## Bước 4 (`06:00 – 09:00`) · Từ mô hình nghiên cứu ra Demo chạy thật (D1 → D2 → D3)

### 4.1. (`06:00 – 07:05`) Vì sao phải qua 2 bước kiểm chứng D1, D2 trước khi làm web D3? Làm bằng cách nào?

- **Thao tác màn hình:** Nói rõ mạch logic 3 bài toán kỹ thuật (nếu thầy hỏi số chi tiết thì mở nhanh [01_D1_LamaHCEExt_Inference.md](../../Workspace/07_Demo/01_D1_LamaHCEExt_Inference/01_D1_LamaHCEExt_Inference.md) và [02_D2_LamaHCEExt_CPU.md](../../Workspace/07_Demo/02_D2_LamaHCEExt_CPU/02_D2_LamaHCEExt_CPU.md)).
- **Lời nói chi tiết:**
  > "Sang mảng thứ ba là **xây dựng Demo D1–D3**. Thay vì làm web ngay từ đầu, em bắt buộc phải giải quyết tuần tự **3 bài toán kỹ thuật** để bảo đảm Demo chạy đúng số liệu thật và chạy mượt trên máy không có GPU:
  >
  > 1. **Bài toán 1 — Tách hàm dự báo khỏi mã huấn luyện (`01_D1_LamaHCEExt_Inference` V2 trên GPU T4):** Trong thư viện NeuralHydrology gốc, code dự báo dính chặt vào quy trình huấn luyện rất cồng kềnh. Em viết lại một mô-đun dự báo độc lập trong `DemoRuntime/forecast.py` dùng chung cho mọi lõi C2, H và D. Để chắc chắn không viết sai bước chuẩn hóa hay nạp dữ liệu, em chạy đối chiếu trên GPU T4 và xác nhận sai số tuyệt đối bằng **`0` mm/ngày** cho cả **24 run C2** (`10.824` cặp lưu vực) lẫn **3 run đại diện D** (`1.353` cặp).
  > 2. **Bài toán 2 — Đưa Mamba và S4D từ GPU xuống chạy trên CPU (`02_D2_LamaHCEExt_CPU` V2 trên CPU 4 luồng):** Máy chủ VPS và laptop không có GPU, trong khi thư viện `mamba-ssm` dùng lúc huấn luyện **bắt buộc cần CUDA GPU**. Cách em giải quyết là thay backend suy luận bằng **`mambapy==1.2.0`** (thuần PyTorch, nạp trực tiếp đúng file trọng số `best_model.pt` đã train từ GPU mà **không phải huấn luyện lại**), còn S4D dùng phép biến đổi FFT trên CPU. Kết quả kiểm thử đạt **288/288 lượt C2** và **108/108 lượt D**, độ trễ cho 1 lần dự báo chuỗi 730 bước chỉ **`19,9 ms` (LSTM), `61,7 ms` (Mamba) và `128,0 ms` (S4D)**. Riêng hai mô hình **bước H cách lai** em đã kiểm toàn bộ kỳ test trên laptop đủ **9.148.986 điểm** với `0` lỗi.
  > 3. **Bài toán 3 — Kiến trúc lai của Demo D3 (`prepare.py` + `app.py`):** Nếu người dùng bấm đổi trạm mà chạy cả 21 mô hình mỗi lần thì web sẽ rất chậm. Vì vậy em viết script **`prepare.py`** gom toàn bộ kết quả đã kiểm SHA-256 (từ C2, H, D, A1, A2, A3, Evaluation V13 và XAI bước G) đóng gói sẵn thành một thư mục độc lập **`downloads/D3_CHDG_CPU_V13`** (`169` tệp, `852 MB`) để **phát lại tức thì**; **chỉ riêng chức năng “Nếu… thì…”** (khi người dùng tự thay đổi lượng mưa/nhiệt độ chưa từng có) thì máy chủ mới nạp checkpoint `best_model.pt` cùng dữ liệu gốc `LamaH_extended` để **chạy PyTorch thật trên CPU**. Vì C2 dùng thư viện `freeze-C` còn H dùng `freeze-H`, em tách riêng tiến trình con `.venv-h` cho mô hình H để không xung đột phiên bản thư viện."

### 4.2. (`07:05 – 09:00`) Trình diễn trực tiếp 4 tính năng trên trình duyệt (`http://127.0.0.1:8053`)

- **Thao tác màn hình:** Chuyển sang cửa sổ trình duyệt đang mở `http://127.0.0.1:8053/` và thao tác tuần tự 4 tab:

1. **Tab `Dự báo` (`07:05 – 07:50` · 45 giây):**
   - **Thao tác:** Chỉ bản đồ 451 trạm → chọn **ID 6**, ngày **15/06/2016** → bật `LSTM D (730)` và `Persistence` cạnh `GRU C2` → kéo xuống **“Nếu… thì…”**, chọn **LSTM (ECMWF HRES ngày đầu — 7 hạn)**, nhập mưa **`1,5`**, nhiệt **`2`**, bấm **Chạy thử**.
   - **Lời nói:**
     > "Bây giờ em xin demo trực tiếp ứng dụng đang chạy trên laptop.
     > - Ở trang **Dự báo**, dữ liệu chuỗi được đóng gói từ 21 mô hình (C2, H, D, BiasCast A2 và persistence), còn **4 đường ngưỡng lũ 1,5 / 2 / 5 / 10 năm** lấy từ bảng ngưỡng `A1` và quy đổi theo diện tích lưu vực sang đơn vị **m³/s** cho trực quan. Em chọn **ID 6 ngày 15/06/2016** và bật so sánh đồng thời GRU C2, LSTM D 730 ngày và mốc persistence.
     > - Kéo xuống khung **“Nếu… thì…”**: đây là phần **chạy suy luận thật trên CPU**. Em chọn **LSTM dự báo 7 ngày**, giả sử lượng mưa tăng **1,5 lần** và nhiệt độ tăng **+2 °C**, bấm **Chạy thử**. *(Đợi 1 giây bảng hiện ra)*
     > - Trước khi in bảng này, máy chủ tự chạy lại dự báo gốc trên CPU, đối chiếu khớp với tệp NPZ gốc để bảo đảm không sai số, rồi mới tính kịch bản mưa/nhiệt mới cho đủ 7 ngày tới — thầy thấy ở hạn 1 lưu lượng tăng từ **`19,89` lên `21,55 m³/s`**."

2. **Tab `So sánh mô hình` (`07:50 – 08:10` · 20 giây):**
   - **Thao tác:** Bấm tab **So sánh mô hình** → chọn `LSTM D (730)` so với `C2 cùng seed`.
   - **Lời nói:**
     > "Trang **So sánh mô hình** đọc trực tiếp điểm NSE 451 lưu vực từ `Evaluation` V10–V13 và `D_basin_scores.csv`. Em đã khóa thang màu đối xứng `±P95` quanh 0 để vài trạm ngoại lệ có NSE âm sâu không làm nhạt màu 448 trạm còn lại; bên dưới có đủ bảng kịch bản S0–S4 và tốc độ CPU D2."

3. **Tab `Cảnh báo lũ` (`08:10 – 08:25` · 15 giây):**
   - **Thao tác:** Bấm tab **Cảnh báo lũ** → chọn `LSTM H GEFS`, **ID 6**, **hạn 7**.
   - **Lời nói:**
     > "Trang **Cảnh báo lũ** đọc các bảng sự kiện từ `Evaluation` V10/V12/V13, cho thấy theo từng trạm và từng hạn dự báo: phân bố 4 mức lưu lượng và từng trận lũ độc lập theo chuẩn USWRC — trận nào bắt trúng, bỏ sót hay báo nhầm."

4. **Tab `Giải thích` (`08:25 – 09:00` · 35 giây):**
   - **Thao tác:** Bấm tab **Giải thích** → giữ mặc định **H · GEFS · hạn 7** (rê chuột qua 3 biểu đồ) → đổi sang **C2 (hạn 1 ngày)** và chọn lõi **Mamba** (hoặc **LSTM**).
   - **Lời nói:**
     > "Cuối cùng là trang **Giải thích (XAI)** em vừa ghép trọn bộ tối qua, đọc trực tiếp 2 tệp tổng hợp theo lưu vực từ bước G Version 1 của Thắng (`GH_demo_basin.npz` cho bước H và `G_demo_basin.npz` cho bước C2, đều đã kiểm SHA-256 theo `manifest.csv`):
     > - Ở chế độ **bước H (GEFS, hạn 7)**, người xem thấy ngay với đúng lưu vực đang chọn: tỉ phần thông tin tương lai tăng dần theo hạn 1–7, đóng góp 9 nhóm nguồn, và **bản đồ nhiệt 7×7 giữa LSTM và Mamba** giải thích trực quan vì sao Mamba khai thác GEFS ít hơn LSTM.
     > - Khi chuyển sang **bước C2 (hạn 1 ngày)** *(đổi dropdown)*, giao diện vẽ 7 nhóm nguồn của 5 lõi, đường cong tích lũy độ trễ 0–60 ngày và **bản đồ nhiệt 0–30 ngày × 6 nhóm nguồn của riêng lưu vực đó** — thấy rõ mô hình dựa mạnh nhất vào ngày nào và nguồn dữ liệu nào.
     > - Còn trang **Giới thiệu** đã ghi đủ bản quyền LamaH-CE / BiasCast và miễn trừ nghiên cứu phi thương mại để sẵn sàng đưa lên VPS."

---

## Bước 5 (`09:00 – 10:00`) · Tổng kết Chương 6 và kế hoạch chốt tuần cuối

- **Thao tác màn hình:** Mở [06_ExperimentsAndResults.md](../../Document/07_Report/06_ExperimentsAndResults.md) và kéo lướt qua **Mục 6.10 (bước D)**, **Mục 6.11.5 (ByBasin V3)** và **Mục 6.12 (bước G)**.
- **Lời nói chi tiết:**
  > "Về phần báo cáo, em cũng đã cập nhật hoàn chỉnh **Chương 6** trong repo — bao gồm kết quả bước D ở Mục 6.10, phân tích lưu vực ByBasin V3 ở Mục 6.11.5, và vừa viết xong trọn **Mục 6.12 về Giải thích mô hình bước G** với đầy đủ bảng số liệu đối chiếu từ CSV và 6 hình minh họa.
  >
  > Như vậy đến hôm nay, toàn bộ phần thực nghiệm, đánh giá, giải thích và demo trên laptop của nhóm đã hoàn tất đồng bộ. Công việc còn lại trong 2 tuần tới trước hạn nộp là rất gọn:
  > 1. Nén gói hoàn chỉnh `D3_CHDG_CPU_V13` bàn giao cho bạn Thắng đưa lên máy chủ **VPS Oracle ARM64 (bước D4)** — phần mã WSGI, kiểm sức khỏe `/api/health` và giới hạn luồng CPU cho ARM64 em đã chuẩn bị sẵn.
  > 2. Hai bạn cùng rà soát toàn bộ báo cáo tiểu luận và làm slide bảo vệ.
  >
  > Em xin cảm ơn thầy đã lắng nghe ạ, rất mong nhận được góp ý của thầy!"

---

## Phần dự phòng: Trả lời nhanh nếu thầy hỏi thêm (30 giây/câu)

- **Thầy hỏi:** *"Tại sao tăng cửa sổ lên 730 ngày mà LSTM và Mamba không tăng NSE?"*
  - **Trả lời:** *"Dạ vì với bài toán dự báo hạn 1 ngày trên Extended LamaH-CE, lưu lượng hôm qua và thời tiết ngày dự báo đã mang phần lớn thông tin. Điều này khớp độc lập với kết quả XAI bước G: độ dài nhớ L90 của LSTM và Mamba chỉ khoảng 34–65 ngày, còn S4D trải bộ nhớ tới ~200 ngày nên S4D là lõi duy nhất hưởng lợi nhất quán (+0,0053 NSE, +0,0117 PNSE) khi tăng lên 730 ngày ạ."*
- **Thầy hỏi:** *"Mamba vốn dùng nhân CUDA `mamba-ssm` trên GPU, làm sao chạy được trên CPU của laptop và VPS?"*
  - **Trả lời:** *"Dạ ở bước D2 em đã tách riêng backend CPU dùng `mambapy==1.2.0` thuần PyTorch và kiểm đối chiếu trực tiếp với NPZ xuất từ GPU: sai số nhỏ hơn `2 × 10⁻⁵` mm/ngày, và độ trễ 1 lần suy luận chuỗi 730 bước trên CPU 4 luồng chỉ khoảng `62 ms` thôi ạ."*
- **Thầy hỏi:** *"Integrated Gradients ở trang Giải thích có chứng minh được nhân quả không?"*
  - **Trả lời:** *"Dạ không ạ — trong Chương 6 nhóm ghi rất rõ IG chỉ đo **độ nhạy** của mô hình theo đầu vào so với đường gốc trung bình kỳ train, và có thể chia trọng số giữa các nguồn mưa tương quan như MSWEP/E-OBS/ERA5L. Tuy nhiên phép kiểm chứng **deletion test** đã đạt 100% (360/360 trường hợp ở bước H), khẳng định IG chọn đúng cả ngày lẫn biến mà mô hình thực sự dựa vào."*
