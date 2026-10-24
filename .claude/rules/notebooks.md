---
paths:
  - "Workspace/**"
  - "Tools/**"
---

# Quy tắc viết notebook và mã (Workspace/, Tools/)

## Thư mục và tên tệp

- `Workspace/<NN_Việc>/`: **mỗi thư mục chỉ một loại việc** (`01_Download` chỉ tải, `02_Exploration` khám phá, `03_Baseline` tái lập mốc, `04_Setup` hạ tầng, `05_Training` huấn luyện từ B1, `06_Evaluation` bộ đánh giá); không để mã phân tích trong thư mục tải. Thư mục cho bước sau thêm khi bắt đầu.
- **Mỗi notebook Kaggle/Colab một tệp `.py` tự đủ mã**, chia `# %% Phần X`, tên `NN_<Bước>_<BộDữLiệu>_<Việc>.py` (NN là thứ tự chạy trong thư mục, `<Bước>` là mã bước ở `03_Pipeline.md` Mục 7 như `A2`; tệp tải dữ liệu bỏ `<Bước>`). Tên notebook Kaggle trùng tên tệp. **Mỗi notebook một thư mục con cùng tên** trong `Workspace/<NN_Việc>/`, chứa `.py`, `.ipynb`, báo cáo `.md` và `Figures/`, `Tables/` của riêng notebook đó (VD `Workspace/02_Exploration/01_A1_LamaHCEExt_Explore/`). Các biến thể của cùng một notebook (VD 3 hạt giống A4) chung một thư mục mang tên gốc (`03_A4_LamaHCEExt_Retrain/` chứa `_Seed111/222/333` và báo cáo); hình nằm ở thư mục của notebook sinh ra hình.
- **`.py` là mã nguồn** để chạy tương tác và thử; **`.ipynb` cùng tên là bản chính thức** — tải về từ đúng phiên bản Save & Run All trên Kaggle, có output, ghi số Version trong báo cáo. Sửa mã ở `.py`, chạy Save & Run All, tải `.ipynb` mới về thay bản cũ.
- **Báo cáo `.md` đặt cạnh notebook, cùng tên** (VD `03_A4_LamaHCEExt_Retrain.md`): thuyết minh theo Phần, kết quả, quyết định của bước. Ghi chú theo bộ dữ liệu đặt ở `Document/03_Data/`.
- **Hình tải về đặt cạnh notebook**: `Workspace/<NN_Việc>/<notebook>/Figures/`, tên bắt đầu bằng mã bước (VD `A4_Seeds.png`); tệp `.md` trỏ tới đường dẫn này, không chép hình sang `Document/`. Bảng nhỏ cần giữ đặt ở `Workspace/<NN_Việc>/<notebook>/Tables/`.

## Quy trình

- **Trình bày kế hoạch notebook trước, chờ duyệt rồi mới viết mã**: cài đặt Kaggle (tên, GPU hay CPU, Internet, Persistence, Input kèm số Version), các Phần, đầu ra, vì sao. Người 2 ghi kế hoạch ở `Plans/NN_<Bước>_….md`.
- Mỗi việc đi 2 giai đoạn: (1) hiểu cấu trúc, khái niệm — chưa viết mã; (2) viết mã khi đã hiểu chắc giai đoạn 1.
- **Tra trước khi viết từng cell**: đọc mã nguồn, tài liệu đúng phiên bản đang chạy (fork `research`: `dda45fc` cho bước A, `e794d7c` từ B1, phiên bản thư viện trên Kaggle), nêu ngắn đã kiểm gì.
- **Mỗi lần chỉ viết đúng 1 cell**, đợi người dùng chạy và gửi output thật rồi mới viết cell tiếp (output cell trước có thể đổi hướng cell sau).
- Mỗi lần đưa cell: nói tên notebook, Phần mấy, làm gì. Mỗi lần sửa notebook: nói rõ Phần nào mới, Phần nào sửa, cần chạy lại từ Phần nào.
- **Tệp chính thức chạy nguyên bằng Save & Run All**: người dùng chép nguyên tệp vào Kaggle, không sửa dòng nào — không công tắc, không hằng số phải đổi tay, không cell phải xóa. Nhiều biến thể (VD 3 hạt giống) => mỗi biến thể một tệp chạy được ngay (`..._Seed111.py`, `..._Seed222.py`, `..._Seed333.py`).
- **Test là cell tạm trong chat, không ghi vào tệp nào**: đo thời gian, chạy ngắn kiểm mã, kiểm nhanh => dán thẳng trong tin nhắn, ghi rõ "TEST – cell tạm" và chạy sau Phần nào; người dùng tạo 1 cell, chạy, gửi output, xóa. Không đánh số Phần cho test, không gọi phiên tương tác chạy thử là một bước của bản chính thức.
- Mã viết xong thì chạy thử ở máy bằng dữ liệu giả (thư mục scratchpad) trước khi đưa người dùng, nhất là phần hình và phần đọc tệp.

## Cách viết mã

- **Mục lục ở đầu tệp** dạng chú thích: chỉ tên, mục đích từng Phần, không viết trước nội dung cell chưa tới.
- **Đánh số cell gắn với Phần**: `# %% Phần X - <tên ngắn>` nếu Phần có 1 cell; chỉ dùng `X.1`, `X.2`… khi Phần có từ 2 cell (thêm cell thứ 2 thì đổi cell đầu thành `Phần X.1`). Không dùng `Cell N`. Mỗi cell có 1–2 dòng chú thích đầu nói làm gì.
- Chú thích chỉ dùng `#`, không docstring; gạch nối `-` (không `—`); suy ra viết `=>` (không `→`).
- **Chuyên nghiệp, dùng lại được**: hằng số và hàm khai báo một lần, không lặp giá trị qua nhiều cell; ưu tiên mã chạy được trên cả Kaggle và Colab.
- **Không ghi cứng khóa API, mật khẩu**: dùng Kaggle Secrets/Colab Secrets, có cách nhập tay khi chạy nơi khác.
- **Kiểm điều kiện thì `raise`**, không chỉ in cảnh báo: thiếu Input, sai sha256, số liệu không khớp => dừng với thông báo nói rõ thiếu gì.
- **Output đẹp, dễ đọc**: bảng dùng `display()` (Styler, ẩn chỉ số, làm tròn), chia khối có tiêu đề, rút gọn đường dẫn dài; chủ động vẽ khi hình trực quan hơn bảng.
- Khám phá dữ liệu phải làm sâu: cấu trúc tệp (cột thật so với tên trong bài, đơn vị, cờ chất lượng), % thiếu, phân phối từng biến, bản đồ khi có tọa độ, nguồn tham chiếu liên quan. Chỉ chuyển bước khi đã hiểu, không phải khi "chạy không lỗi".

## Hình trong notebook

- Mọi chữ **trong** hình tiếng Anh, số dấu chấm thập phân.
- Có tiêu đề chung `suptitle` nói hình gì; mỗi ô con có tiêu đề riêng `(a)`, `(b)`… nói ô đó vẽ gì.
- Chú giải đặt **ngoài** vùng vẽ (`loc="upper left", bbox_to_anchor=(1.01, 1)`), không che dữ liệu.
- Ngay dưới hình in **1 chú thích tiếng Việt** bằng hàm `caption()` (`print(f"Hình: ...")`): cách đọc và ý nghĩa; số liệu trong chú thích lấy từ dữ liệu, không gõ tay.
- Lưu `savefig(..., bbox_inches="tight")` vào thư mục output, tên bắt đầu bằng mã bước; người dùng tải về đặt ở `Workspace/<NN_Việc>/<notebook>/Figures/`.
- Nhãn, năm, kỳ dữ liệu lấy từ cấu hình hay dữ liệu, không ghi cứng (đã có tiền lệ ghi sai kỳ validation).
- Vẽ xong chạy thử bằng dữ liệu giả và **xem ảnh** để kiểm chữ đè, chú giải che trước khi đưa người dùng. Hình số liệu dùng matplotlib (seaborn khi cần).

## Kaggle

- Kaggle là nền tảng chính (GPU T4, 30 giờ/tuần, tối đa 12 giờ/phiên); Colab miễn phí dự phòng, Lightning AI khi cần; việc nặng viết notebook Kaggle, không chạy ở máy cá nhân. Người dùng tự chép notebook vào Kaggle (không dùng Kaggle CLI).
- **Kaggle Dataset chỉ cho dữ liệu tải từ ngoài**: `hunhngcthng/lamah-ce-ext` (công khai; Extended LamaH-CE + kết quả tác giả BiasCast, 3 GB), `lamah-ce-core`, `lamah-ce-extra` (Private, LamaH-CE gốc theo giờ, giữ cho khóa luận); tải bằng 3 notebook trong `Workspace/01_Download/` (`01_LamaHCEExt_Download`, `02_LamaHCE_Download_Core`, `03_LamaHCE_Download_Extra`). Collection "TieuLuanChuyenNganh".
- **Kết quả giữa các bước không chuyển thành Dataset**: gắn thẳng Output của notebook đã Save Version làm Input (VD `/kaggle/input/notebooks/hunhngcthng/01-a2-lamahceext-reproduce/a2_output`). Input luôn lấy phiên bản mới nhất => ghi số Version của nguồn trong báo cáo và `environment.json`, kiểm sha256 theo `manifest.csv` của nguồn.
- Mã tìm Input theo mẫu tên tệp dưới `/kaggle/input` thay vì ghi cứng tên người hay tên notebook (VD A3 tìm mọi `A1_*.csv`; lần chạy chính thức gắn `notebooks/hunhthanhnhn/a1-dataset`, 305 CSV, không có trong collection).
- **Collection có 16 mục (24/10/2026)**: 3 Dataset trên; notebook của người 2 (`hunhngcthng`): 3 notebook tải, `01_A2_LamaHCEExt_Reproduce`, `01_0_MambaSSM_Build`, `03_A4_LamaHCEExt_Retrain_Seed111/222/333`, `04_A4_LamaHCEExt_Compare`; của người 1 (`hunhthanhnhn`): `01_A1_LamaHCEExt_Explore`, `02_A3_LamaHCEExt_Benchmarks`, `01_Evaluation_LamaHCEExt`, `02_Evaluation_LamaHCEExt_ByBasin`.
- **Bản lưu chính thức (Version)**: A1 Version 2; bộ đánh giá `01_Evaluation_LamaHCEExt` Version 4 (14 run, Input A4 Compare Version 2), `02_Evaluation_LamaHCEExt_ByBasin` Version 2 (Input = Output bộ đánh giá Version 4); A2 Version 2 (Save & Run All 2106 giây, output 1,55 GB, 6 run); bước 0 `01_0_MambaSSM_Build` Version 1; A3 Version 1; A4 Seed111 Version 1, Seed222 Version 2, Seed333 Version 2; `04_A4_LamaHCEExt_Compare` Version 2 (CPU; Input = Output 3 notebook hạt giống + A3 + `lamah-ce-ext`).
- Wheel `mamba-ssm` lấy từ Output notebook `01_0_MambaSSM_Build`; notebook sau chép cell Phần 9 của notebook đó làm cell cài.
- **Save & Run All chạy phiên mới**: không thấy tệp tạo ở phiên tương tác => tính thời gian theo chạy lại toàn bộ. Mỗi lần huấn luyện chạy trọn trong 1 phiên (dưới 12 giờ) vì `continue_training` không khôi phục dừng sớm và scheduler.
- Mỗi notebook cuối cùng ghi `environment.json` (Input, Version nguồn, thư viện, commit NeuralHydrology) và `manifest.csv` (mọi tệp output kèm sha256).
- Thời gian đo được (A4, T4): khoảng 9,5 phút/epoch SeqLSTM (huấn luyện 5,6–5,9 phút, validation 3,4–3,6 phút), 2,5–3,2 giờ/lần; đánh giá một run 2–6 phút.

## Hợp đồng dữ liệu cho bộ đánh giá

- Mỗi mô hình xuất tệp NPZ theo hợp đồng `contracts/<Mô hình>_<Bước>_seed<N>_<kỳ>_<kịch bản>_lead<h>.npz` (VD `SeqLSTM_Q_A4_seed111_test_S0_lead1.npz`); đơn vị mm/ngày, cùng ngày, cùng thứ tự lưu vực với `A3_persistence_predictions.npz`. Hàm kiểm hợp đồng và cách dựng lại ngữ cảnh: `Workspace/03_Baseline/04_A4_LamaHCEExt_Compare/04_A4_LamaHCEExt_Compare.py` Phần 5; bộ đánh giá: `Workspace/06_Evaluation/`.
- Kiểm NSE tính lại từ NPZ khớp chỉ số thư viện trước khi bàn giao.
