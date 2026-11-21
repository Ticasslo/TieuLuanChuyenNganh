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

## Bước 3 (`02:45 – 06:00`) · Trọng tâm: Bước D (tăng cửa sổ 365 → 730 ngày) qua bộ biểu đồ V3

### 3.1. (`02:45 – 03:45`) Độ chính xác tổng quan: kéo dài 2 năm có tốt hơn 1 năm không?

- **Thao tác màn hình:**
  - **Mở ảnh:** [D_Overview_Accuracy.png](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/Figures/D_Overview_Accuracy.png)
  - **Thuộc notebook:** `17_D_LamaHCEExt_Compare` (**Version 3**, [notebook chính thức](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/17_D_LamaHCEExt_Compare.ipynb) · [báo cáo](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/17_D_LamaHCEExt_Compare.md))
- **Lời nói chi tiết:**
  > "Sang mảng thứ hai là **bước D**, em phụ trách trọn gói từ khâu huấn luyện 9 run trên Kaggle đến so sánh và đánh giá. Câu hỏi nghiên cứu ở đây rất rõ: *mô hình ở bước C2 đang nhìn quá khứ 365 ngày (tức 1 năm); nếu ta cho mô hình nhìn xa gấp đôi — **730 ngày (tức 2 năm)** — thì các lõi tuần tự như LSTM, Mamba và S4D có học được chu kỳ dài hơn để dự báo chính xác hơn không?*
  >
  > Em đã huấn luyện hoàn tất **9 trên 9 run** (3 lõi × 3 hạt giống, cùng quy mô tham số và cùng 30 epoch như C2, không bị tràn gradient ở bước nào), rồi tổng hợp trong notebook **`17_D_LamaHCEExt_Compare` Version 3** với bộ 9 biểu đồ đa tiêu chí.
  >
  > *(Chỉ vào hai khung biểu đồ trên ảnh `D_Overview_Accuracy.png`)*
  >
  > Nhìn vào ảnh tổng quan này, kết quả thực nghiệm rất trung thực: khi tăng từ 365 lên 730 ngày, điểm NSE trên tập kiểm tra thay đổi rất nhỏ, đều nằm trong biên nhiễu giữa các hạt giống (`0,0079`). Cụ thể:
  > - **LSTM** gần như giữ nguyên (`0,7178` sang `0,7162`, lệch `−0,0016`).
  > - **Mamba** cũng gần như đi ngang (`0,7204` sang `0,7180`, lệch `−0,0023`).
  > - Riêng **S4D** — là mô hình không gian trạng thái không chọn lọc — là lõi **duy nhất tăng nhất quán** trên cả NSE (`+0,0053`, tức `0,6933 → 0,6986`) lẫn PNSE (`+0,0117`).
  >
  > Điều này khớp rất đẹp với kết quả giải thích mô hình ở bước G sau này: LSTM và Mamba vốn dồn 90% sự chú ý vào 1–2 tháng gần nhất nên 1 năm lịch sử là đã đủ bão hòa; còn S4D trải bộ nhớ rộng ra khoảng 200 ngày nên khi nới cửa sổ lên 2 năm, S4D là lõi hưởng lợi rõ nhất."

### 3.2. (`03:45 – 04:30`) Đánh đổi chi phí GPU/VRAM so với lợi ích

- **Thao tác màn hình:**
  - **Mở ảnh:** [D_Cost_Tradeoff.png](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/Figures/D_Cost_Tradeoff.png)
  - **Thuộc notebook:** `17_D_LamaHCEExt_Compare` (**Version 3**)
  - *(Nếu thầy hỏi thêm về quá trình hội tụ 30 epoch, mở dự phòng ảnh [D_Curves_365_vs_730.png](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/Figures/D_Curves_365_vs_730.png) từ cùng notebook.)*
- **Lời nói chi tiết:**
  > "Bây giờ nếu đặt độ chính xác cạnh chi phí tính toán — như trên ảnh **`D_Cost_Tradeoff.png`** thuộc cùng notebook `17_D_Compare` Version 3 — ta thấy bức tranh rất rõ.
  >
  > *(Chỉ vào mũi tên dịch chuyển từ chấm 365 sang chấm 730 của từng lõi)*
  >
  > Khi tăng độ dài chuỗi gấp đôi, thời gian huấn luyện và bộ nhớ GPU gần như **tăng gấp đôi theo**:
  > - **LSTM** từ khoảng `2 giờ` lên `3,9 giờ`.
  > - **Mamba** từ `5,2 giờ` lên `10 giờ`.
  > - **S4D** từ `6,5 giờ` lên gần `13 giờ` (phải chia 2 phiên Kaggle nối tiếp nhau).
  >
  > Trong khi đó trục dọc — tức độ chính xác NSE — gần như đi ngang. Từ đây nhóm rút ra **khuyến nghị thực tiễn rất dứt khoát**: với bài toán dự báo 1 ngày trên bộ Extended LamaH-CE, **cửa sổ 365 ngày đã là điểm tối ưu về hiệu năng trên chi phí**; không cần tốn gấp đôi tài nguyên cho 730 ngày."

### 3.3. (`04:30 – 05:20`) Tại sao trung vị đi ngang nhưng từng nhóm lưu vực lại tách hai hướng?

- **Thao tác màn hình:**
  - **Mở ảnh:** [BasinAnalysis_D_static_delta_NSE_rho.png](../../Workspace/06_Evaluation/02_Evaluation_LamaHCEExt_ByBasin/Archive/Version3/D/Figures/BasinAnalysis_D_static_delta_NSE_rho.png)
  - **Thuộc notebook:** `02_Evaluation_LamaHCEExt_ByBasin` (**Version 3**, phân vùng `D/`)
- **Lời nói chi tiết:**
  > "Tuy nhiên, em không dừng ở điểm trung vị mà tiếp tục mổ xẻ hiệu `730 − 365` theo từng lưu vực trong phân vùng D của notebook **`02_Evaluation_LamaHCEExt_ByBasin` Version 3**.
  >
  > *(Chỉ cột `S4D` đối chiếu với cột `LSTM` trên ảnh `BasinAnalysis_D_static_delta_NSE_rho.png`)*
  >
  > Ảnh này cho thấy một phát hiện khá thú vị: khi tăng lên 2 năm, 3 lõi tách thành **hai hướng trái ngược nhau** theo đặc trưng thủy văn:
  > - **S4D** (và Mamba nghiêng nhẹ cùng chiều) lại cải thiện rõ hơn ở các **lưu vực khô hạn, bốc thoát hơi cao và diện tích lớn** (`arid_1` tương quan `+0,208`, `et0_mean +0,197`) — tức nơi cần nhớ cân bằng nước dài hạn.
  > - Ngược lại, **LSTM** lại cải thiện ở các **lưu vực có băng hà và tuyết** (`glac_fra +0,141`, `frac_snow +0,139`), nơi có chu kỳ tích–tan tuyết theo mùa dài.
  >
  > Nhờ phân tích này, mình hiểu được rằng cửa sổ 730 ngày không vô nghĩa hoàn toàn, mà tác động của nó phân hóa theo từng chế độ thủy văn."

### 3.4. (`05:20 – 06:00`) Nhìn trực quan vào đường quá trình dòng chảy (Hydrograph)

- **Thao tác màn hình:**
  - **Mở ảnh:** [D_Hydrograph_Diagnostics.png](../../Workspace/05_Training/17_D_LamaHCEExt_Compare/Figures/D_Hydrograph_Diagnostics.png)
  - **Thuộc notebook:** `17_D_LamaHCEExt_Compare` (**Version 3**)
- **Lời nói chi tiết:**
  > "Và để người đọc dễ hình dung sự khác biệt ngoài thực tế chứ không chỉ qua con số, trong bộ 9 hình của `17_D_Compare` Version 3 em vẽ thêm ảnh **`D_Hydrograph_Diagnostics.png`** — so sánh trực tiếp đường lưu lượng quan trắc (màu đen) với dự báo 365 ngày và 730 ngày tại những lưu vực cải thiện mạnh nhất và suy giảm mạnh nhất.
  >
  > *(Chỉ nhanh vào đỉnh lũ ở các ô trên cùng)*
  >
  > Nhìn vào đây người xem thấy ngay mô hình 730 ngày thay đổi cách bám đỉnh lũ và chân lũ như thế nào trên từng trạm cụ thể."

---

## Bước 4 (`06:00 – 09:00`) · Kiểm chứng suy luận D1–D2 và chạy trực tiếp Demo D3

### 4.1. (`06:00 – 06:40`) Kiểm chứng suy luận GPU/CPU trước khi làm Demo

- **Thao tác màn hình:** Nói ngắn gọn (chỉ cần mở [01_D1_LamaHCEExt_Inference.md](../../Workspace/07_Demo/01_D1_LamaHCEExt_Inference/01_D1_LamaHCEExt_Inference.md) hoặc [02_D2_LamaHCEExt_CPU.md](../../Workspace/07_Demo/02_D2_LamaHCEExt_CPU/02_D2_LamaHCEExt_CPU.md) nếu thầy muốn xem bảng số).
- **Lời nói chi tiết:**
  > "Sang mảng thứ ba là **Demo**, trước khi làm giao diện web, em đã kiểm chứng rất chặt khâu suy luận độc lập qua hai notebook:
  > - **`01_D1_LamaHCEExt_Inference` Version 2 (trên GPU T4):** hàm suy luận tách rời tái lập chính xác tuyệt đối **`0` mm/ngày** cho cả 24 run C2 lẫn 3 run đại diện bước D.
  > - **`02_D2_LamaHCEExt_CPU` Version 2 (trên CPU 4 luồng):** vì máy chủ web sau này chỉ có CPU chứ không có GPU, em chuyển Mamba sang thư viện `mambapy` thuần PyTorch và kiểm chứng đạt trọn **288/288 lượt C2** và **108/108 lượt D**, độ trễ cho 1 lần dự báo chuỗi 730 bước chỉ từ `20 ms` (LSTM) đến `62 ms` (Mamba) và `128 ms` (S4D). Riêng hai mô hình **bước H cách lai**, em đã kiểm toàn bộ kỳ test trên laptop đủ **9,15 triệu điểm** với `0` lỗi."

### 4.2. (`06:40 – 09:00`) Trình diễn trực tiếp Demo D3 trên trình duyệt (`http://127.0.0.1:8053`)

- **Thao tác màn hình:** Chuyển sang cửa sổ trình duyệt đang mở `http://127.0.0.1:8053/` (gói hoàn chỉnh `downloads/D3_CHDG_CPU_V13`: **451 lưu vực, 21 đường dự báo, 7 mô hình chạy trực tiếp CPU, trọn bộ XAI bước G**). Thực hiện tuần tự đúng **4 thao tác**:

1. **Tab `Dự báo` (`06:40 – 07:30` · 50 giây):**
   - **Thao tác:** Chỉ vào bản đồ 451 trạm → chọn **ID 6**, ngày **15/06/2016** → bật thêm `LSTM D (730)` và `Persistence` cạnh `GRU C2` → kéo xuống mục **“Nếu… thì…”**, chọn **LSTM (ECMWF HRES ngày đầu — 7 hạn)**, nhập mưa **`1,5`**, nhiệt **`2`**, bấm **Chạy thử**.
   - **Lời nói:**
     > "Đây là ứng dụng Demo D3 chạy trực tiếp trên laptop với gói hợp nhất mới nhất **`D3_CHDG_CPU_V13`**. Trên trang **Dự báo**, người dùng chọn bất kỳ trạm nào trong **451 lưu vực** — ví dụ ID 6 ngày 15/06/2016 — và bật so sánh cùng lúc các mô hình C2, H, D và mốc persistence kèm **4 ngưỡng lũ** theo chu kỳ lặp lại 1,5 / 2 / 5 / 10 năm quy đổi ra **m³/s**.
     >
     > Điểm đặc biệt là khung **“Nếu… thì…”** ở dưới đây: em chọn **LSTM dự báo 7 ngày**, giả sử lượng mưa tăng gấp **1,5 lần** và nhiệt độ tăng **+2 °C**, rồi bấm **Chạy thử**. *(Đợi 1 giây bảng hiện ra)*
     >
     > Máy vừa chạy **mô hình PyTorch thật trên CPU laptop**, tự đối chiếu kết quả gốc khớp với NPZ rồi mới trả bảng dự báo cho đủ 7 ngày tới — thầy thấy ở hạn 1 lưu lượng tăng ngay từ **`19,89` lên `21,55 m³/s`**."

2. **Tab `So sánh mô hình` (`07:30 – 07:55` · 25 giây):**
   - **Thao tác:** Bấm tab **So sánh mô hình** → chọn `LSTM D (730)` so với `C2 cùng seed` (hoặc `Mamba C2` so với `Persistence`).
   - **Lời nói:**
     > "Sang trang **So sánh mô hình**, bản đồ hiển thị trực quan mức chênh NSE giữa hai mô hình trên cả 451 trạm — em đã khóa thang màu đối xứng quanh 0 để không bị vài trạm ngoại lệ kéo nhạt màu bản đồ — kèm đầy đủ các bảng điểm kịch bản S0–S4 và tốc độ CPU ở bên dưới."

3. **Tab `Cảnh báo lũ` (`07:55 – 08:15` · 20 giây):**
   - **Thao tác:** Bấm tab **Cảnh báo lũ** → chọn `LSTM H GEFS`, **ID 6**, **hạn 7**.
   - **Lời nói:**
     > "Ở trang **Cảnh báo lũ**, ứng dụng nối trực tiếp kết quả từ `Evaluation` V13 để thống kê theo **4 mức lưu lượng** và các **trận lũ độc lập theo chuẩn USWRC**: trận nào bắt đúng, trận nào bỏ sót, trận nào báo động nhầm."

4. **Tab `Giải thích` (`08:15 – 09:00` · 45 giây):**
   - **Thao tác:** Bấm tab **Giải thích** → để mặc định **H · GEFS · hạn 7** (rê chuột qua 3 biểu đồ) → sau đó đổi mục chọn sang **C2 (hạn 1 ngày)** và đổi lõi sang **Mamba** hoặc **LSTM** để hiện bản đồ nhiệt 0–30 ngày.
   - **Lời nói:**
     > "Cuối cùng là trang **Giải thích (XAI)** mà em vừa hoàn thiện tối qua, ghép trọn bộ hai tệp kết quả bước G của Thắng (`GH_demo_basin.npz` và `G_demo_basin.npz`, đều kiểm SHA-256) trên cả **451 lưu vực**:
     > - Ở chế độ **bước H (GEFS, hạn 7)**, thầy thấy ngay tỉ phần đóng góp của thông tin thời tiết tương lai tăng dần từ hạn 1 lên hạn 7, phân rã theo 9 nhóm nguồn, và bên dưới là **bản đồ nhiệt 7×7 so sánh LSTM và Mamba** — thấy rất rõ Mamba dùng thông tin GEFS ít hơn LSTM.
     > - Khi chuyển sang **bước C2 (hạn 1 ngày)** *(chuyển dropdown)*, giao diện hiển thị 7 nhóm nguồn của 5 lõi, đường cong tích lũy độ trễ 0–60 ngày — thấy ngay LSTM/GRU/Mamba dồn vào vài ngày gần nhất còn S4D/Transformer trải dài — và ở dưới là **bản đồ nhiệt ngày trễ 0–30 × 6 nhóm nguồn trực tiếp của đúng lưu vực đang chọn**.
     > - Ngoài ra trang **Giới thiệu** đã ghi đầy đủ bản quyền dữ liệu và miễn trừ phục vụ nghiên cứu phi thương mại."

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
