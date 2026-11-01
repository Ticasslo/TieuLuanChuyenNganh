# Bộ đánh giá dự báo lưu lượng — Extended LamaH-CE

Tài liệu đi cùng [mã Python](01_Evaluation_LamaHCEExt.py) và [notebook](01_Evaluation_LamaHCEExt.ipynb) (bản chính thức: Version 4 của notebook Kaggle `01_Evaluation_LamaHCEExt`, Save & Run All trên CPU, có output), thực hiện việc #5 trong [phân công](../../../ASSIGNMENT.md) và [pipeline Mục 6](../../../Document/01_Plan/02_Pipeline.md#6-đánh-giá). Bộ đánh giá đọc dự báo đã lưu, tính chỉ số, đánh giá bốn mức và sự kiện lũ, so sánh theo cặp lưu vực, rồi xuất CSV/PNG/JSON trực tiếp cho Kaggle Save Version.

**S0** là kịch bản không che thêm đầu vào; dữ liệu gốc vẫn có thể thiếu. **Hạn một ngày** là dự báo ngày kế tiếp sau ngày cuối của lịch sử quan trắc. Một **ngày–lưu vực** là một ngày tại một ID; **mặt nạ** là bảng True/False chọn các cặp đó để chấm. **Giao chung** chỉ giữ ngày–lưu vực có quan trắc, persistence và dự báo của mọi run cùng hữu hạn.

**Version 4 chấm 14 run trên test 2014–2017, S0, hạn một ngày: sáu A2, ba DLinear, hai persistence và ba LSTM A4; 129 kiểm thử đạt, khớp A3 và ba hợp đồng A4.** Sáu chỉ số và số ngày chấm khớp A3 ở 4.961 cặp run–lưu vực; giao chung có 652.628 ngày–lưu vực trên 450 ID có cặp hợp lệ. Bảng lũ chính yêu cầu **cùng ngày bắt đầu**, bảng phụ cho phép **±1 ngày**, Holm chỉ đếm p hữu hạn. Ở ngưỡng lũ hai năm, SeqLSTM_Q có POD 19,51%, ba A4 16,89–20,26%, D_RA 20,54%, DLinear 0,19% và persistence qmax 0%. Kết quả chi tiết ở [Mục 10](#10-kết-quả-thực-nghiệm-trên-kaggle). Bộ đánh giá chỉ đọc dự báo đã lưu, không gọi mô hình hoặc huấn luyện lại.

Mã dùng **giao thức 3**: dung sai chính ±⌊hạn/3⌋ ngày cho từng hạn 1–7, bảng phụ ±1 ngày. Ở hạn một ngày, dung sai chính bằng 0. Phần nhiều hạn đã qua kiểm thử cục bộ và đường chạy dữ liệu giả đủ bảy hạn; kết quả thực nghiệm hiện chỉ có hạn một ngày.

## 1. Đầu vào và cách chạy

Mã trong `.py` và `.ipynb` hiện có đã đồng bộ để chuẩn bị **Version 5**, tự đọc **17 run**: sáu A2, ba DLinear, hai persistence, ba LSTM A4 và ba LSTM B0. Notebook `.ipynb` đã đồng bộ mã và định dạng tại chỗ; trạng thái thực thi/output cũ được xóa. Output phiên tương tác bổ sung B0 đã được kiểm, còn bản Save Version 5 chưa được cung cấp để xác nhận. Các bảng và hình trong repo vẫn là kết quả **Version 4**. Cần chốt bản lưu chính thức trước khi thay bộ bảng/hình theo Version 5.

Gắn năm nguồn Output đã Save Version: A1 Version 3, A2 Version 2, A3 Version 1, **`04_A4_LamaHCEExt_Compare` Version 2**, **`02_B0_LamaHCEExt_Compare` Version 1**. Dataset gốc và các Output huấn luyện riêng không bắt buộc. CPU, Internet không bắt buộc, chạy toàn bộ `.py` trong phiên sạch, sau đó Save & Run All thành Version 5. Không cần huấn luyện lại các mốc.

| Nguồn | Tệp cần dùng |
|---|---|
| A1 | `A1_P10_qmax_level_thresholds.csv`, `A1_P10_flood_return_levels.csv` |
| A2 | `environment.json`; dưới `runs/<run>/`: `config.yml` và `test_results.p` của sáu run |
| A3 | `A3_Q_source_window.npz`, `A3_persistence_predictions.npz`, `A3_all_metrics_per_basin.csv` |
| DLinear trong A3 | `dlinear/seed_111`, `seed_222`, `seed_333`, mỗi thư mục có `test_predictions.npz` |
| A4 Compare Version 2 | Toàn thư mục `a4_compare/`, gồm `a4_summary.json`, `environment.json`, `manifest.csv` và ba NPZ `contracts/SeqLSTM_Q_A4_seed{111,222,333}_test_S0_lead1.npz`; các tệp còn lại trong manifest cũng cần được giữ |
| B0 Compare Version 1 | Toàn thư mục `b0_compare/`, gồm `b0_summary.json`, `environment.json`, `manifest.csv` và ba NPZ `contracts/SeqLSTM_Q_B0_seed{111,222,333}_test_S0_lead1.npz`; giữ đủ mọi tệp được kê trong manifest |

Mặc định tìm nguồn dưới `/kaggle/input`, xuất vào `/kaggle/working/evaluation_output`. A2 được nhận diện bằng `environment.json` trong thư mục có đủ `runs/<sáu run>`, nên không nhầm với `environment.json` của A4. A4 được tìm qua `a4_summary.json`; thiếu hoặc trùng nguồn làm chương trình dừng, không tự bỏ seed hoặc chọn phiên bản. Script có thể chỉ rõ `--a1`, `--a2`, `--a3`, `--a4`, `--b0`; notebook chạy nguyên mã sau khi gắn đủ Input, không cần điền đường dẫn hay tên tài khoản.

Trên Kaggle, dùng CPU và Run All từ đầu sau khi nhập bản mã mới. Phần 8.1 nhận diện nguồn A2; Phần 8.4 đọc/kiểm ba NPZ A4 và ba NPZ B0; Phần 10.3 vẽ CDF có chú giải phân biệt các seed; Phần 11.1 điều phối chấm chung. Cần chạy lại các phần định nghĩa phía trước trong phiên sạch để tránh giữ hàm cũ.

Kiểm độc lập bằng lệnh `python Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.py --self-test` từ thư mục repo. Môi trường cần NumPy, pandas, SciPy, xarray, PyYAML; phần vẽ dùng matplotlib. Kaggle A1/A2/A3 đã có các thư viện này. Bộ đánh giá không thay phiên bản thư viện trong phiên chạy.

Mã `.py` dùng các cell `# %%` và cell Markdown `# %% [markdown]`: tiêu đề phần → đề mục → code đúng chức năng. Mã của 17 code cell trong `.ipynb` đã đồng bộ với `.py`; output phiên tương tác bổ sung B0 đã được đối chiếu, bản lưu Save Version 5 đang chờ chạy. Kiểm cục bộ bằng chính mã đọc từ `.ipynb` sau định dạng đạt 144 ca.

Kiểm cục bộ mã chuẩn bị Version 5: **144 kiểm thử đạt**; đường đọc → chấm → xuất bằng dữ liệu giả chạy đủ 17 run, 360 dòng kiểm định và hình CDF. Đây không phải kết quả chấm NPZ B0 thật.

Sau khi chạy Version 5, kiểm `Evaluation_summary.json`: `complete=true`, `B0_contracts_matched=true`, `A4_contracts_matched=true`, `A3_reference_matched=true`, coverage có 17 run ở test/S0/lead1. Có thêm `Evaluation_B0_input_check.csv` và `Evaluation_B0_contract_check.csv`. Các NPZ B0 phải khai báo commit đầy đủ `f229e02bf98cbd979835a4b6cd503d0e3adb8bc3`; ba notebook huấn luyện đều Version 1. `environment.json` của B0 lưu commit theo từ điển `B0`/`A4`, khác trường chuỗi của A4. Ngưỡng B* được kiểm lại bằng std mẫu (`ddof=1`) của ba NSE trung vị validation trong `b0_summary.json`.

Thêm B0 làm họ Holm có tối đa 120 cặp mỗi chỉ số (16 run chính, bỏ persistence qmean phụ), tổng 360 dòng kiểm định. Không thay p Holm của Version 4 trong báo cáo bằng p từ họ mới trước khi có output. Phép đối chiếu A3 vẫn tính riêng trên tập run gốc, tránh đổi mặt nạ tham chiếu.

Đầu vào cần thêm duy nhất so với Version 4 là Output `02_B0_LamaHCEExt_Compare` Version 1; giữ các Input A1/A2/A3/A4 đang dùng. Cập nhật mã trong cùng notebook Kaggle `01_Evaluation_LamaHCEExt`, không tạo notebook V5 riêng.

Khi Save & Run All Version 5 thành công, bản `.ipynb` có output và các kết quả tương ứng thay vào cùng thư mục notebook, `Tables/` và `Figures/`. File `.ipynb` hiện có đã sửa tại chỗ để chạy Save Version, không được ghi nhận như bản Version 5 đã chạy.

### 1.1. Kiểm output phiên tương tác trước Save Version 5

Đã đọc toàn bộ 26 tệp được cung cấp: 22 CSV, 3 JSON và 1 PNG. Có **84 kiểm chéo** về schema, danh tính run, seed/epoch/commit, số ngày, các phép tổng hợp và công thức; tất cả đạt. Sáu chỉ số liên tục và ba chữ ký được đối chiếu với bảng chi tiết; NSE/PNSE được tính lại từ các tổng bình phương, KGE từ r/alpha/beta. Bảng tổng hợp 459 dòng, 360 dòng kiểm định, 30.668 dòng phân mức, hai bảng sự kiện 61.336 dòng và 22.948 điểm CDF khớp dữ liệu chi tiết tương ứng.

Cả ba NPZ B0 có đúng seed 111/222/333, checkpoint epoch 25/15/14, notebook huấn luyện Version 1 và commit `f229e02`. Giao chung giữ **652.628 ngày–lưu vực, 450 ID có ngày hợp lệ**. NSE trung vị B0 lần lượt là **0,707234; 0,703778; 0,705890**. Ngưỡng B* trong summary/protocol cùng bằng **0,007932303438926078**.

Bảng kịch bản rỗng có header vì chỉ có S0. KGE của `EncDec_noQ` không xác định tại ID 742/758 do r không xác định, nên hình có n=448 cho đường đó; các đường NSE/PNSE tương ứng vẫn có n=450. Hai seed 222/333 của B0 khớp A4 ở toàn bộ bảng chỉ số, phù hợp checkpoint giữ nguyên. Hình CDF có đủ ba ô, chú giải ngoài vùng vẽ, giữ đuôi âm và n riêng theo từng chỉ số.

Kiểm hash cục bộ bao phủ 26 tệp output nhận được. Hash của NPZ/pickle nguồn được đối chiếu giữa các bảng manifest và kiểm đầu vào do Kaggle xuất; không tính lại trực tiếp vì các tệp nguồn đó không nằm trong lô output được cung cấp.

Phần định dạng chỉ thay Markdown/mục lục và bỏ dòng yêu cầu gửi lại output; AST tính toán giữ nguyên. File `.py` và 17 code cell của `.ipynb` đồng bộ, schema notebook hợp lệ; 144 kiểm thử từ chính mã notebook sau định dạng đạt.

## 2. Hợp đồng dữ liệu và phạm vi chấm

Mỗi dự báo có tên run phân biệt seed, kỳ, kịch bản, hạn; lịch ngày đích; danh sách ID; diện tích; obs canonical; sim trước/sau chính sách chặn âm; và persistence. Các ma trận đều có thứ tự `[ngày đích, lưu vực]`. Lịch phải liên tục và duy nhất; thiếu dự báo được giữ bằng NaN trên lịch đó. Danh sách chính thức giữ đủ 451 ID.

Các run cùng nhóm kỳ/kịch bản/hạn phải có cùng lịch, ID, diện tích, obs canonical và persistence. Bộ đọc A2 giữ thêm obs native. Bộ đọc DLinear kiểm ngày, seed, kỳ, diện tích, obs và chính sách chặn âm theo export A3. Nguồn Q của A3 được đổi lại từ m³/s theo area_gov để kiểm obs và persistence; ngày nguồn persistence kỳ test phải là ngày đích trừ một.

| Scope | Quan trắc | Ngày được chấm | Mục đích |
|---|---|---|---|
| `native` | Obs của chính A2; DLinear dùng obs nguồn | Obs/sim hữu hạn | Đọc độ phủ riêng; PNSE có trạng thái không cấp persistence |
| `paired_source` | qmax canonical | Obs/sim/persistence hữu hạn | So từng run với persistence trên cùng ngày |
| `common_source_all` | qmax canonical | Giao hữu hạn của toàn bộ run trong nhóm và persistence | Bảng so sánh chính, phân mức và sự kiện |

Thêm mô hình có thể làm giao chung thay đổi. Phép đối chiếu A3 luôn kiểm riêng tập run gốc A3, còn bảng mở rộng tính lại giao của tập mô hình được đánh giá. Điểm âm, ID không có ngày chấm và trạng thái không xác định được giữ trong CSV.

Với A4, ba NPZ phải có cùng ngày, thứ tự ID, diện tích, obs canonical và persistence với A3, bằng nhau từng giá trị kể cả vị trí NaN. Dự báo A4 trước chặn âm được giữ ở `sim_raw_mm_day`; bảng chấm dùng `sim_mm_day` theo chính sách clip đã khai báo. NPZ A4 chỉ có obs canonical, nên scope `native` của A4 là phạm vi obs canonical/sim hữu hạn; không được coi là đối chiếu từng bit với obs float32 riêng trong pickle huấn luyện.

## 3. Chỉ số liên tục và đường duy trì lưu lượng

NSE, PNSE, KGE dạng 2009 và r/α/β dùng float64, cùng mặt nạ trong một scope, độ lệch chuẩn ddof=0. PNSE lấy qmax hôm trước làm mốc cho dự báo một ngày. Mẫu số bằng 0 có giá trị NaN và trạng thái giải thích; các hàm không tự thêm epsilon hoặc thu hẹp mask đã truyền. [Công thức và ví dụ A3](../../03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.md#report-part-5).

Ba chỉ số đường duy trì được tính trên cùng cặp ngày nhưng xếp riêng obs và sim. Quy ước rời rạc của bộ đánh giá được cố định trước khi chạy:

| Chỉ số | Quy ước |
|---|---|
| %BiasFHV | Lấy ceil(0,02 × n), tối thiểu một ngày lớn nhất; 100 × (tổng sim − tổng obs) / tổng obs |
| %BiasFMS | So độ dốc log giữa phân vị 0,8 và 0,3, nội suy tuyến tính trên log lưu lượng đã xếp; 100 × (dốc sim − dốc obs) / dốc obs |
| %BiasFLV | Lấy ceil(0,3 × n) ngày thấp nhất; so tổng khoảng cách log tới log cực tiểu của từng chuỗi, với dấu âm theo quy ước FLV |

Chỉ phép log của FMS/FLV dùng sàn **10⁻⁶ mm/ngày**, đồng thời báo số giá trị nằm dưới sàn. Mẫu số bằng 0 vẫn trả NaN. Đây là phép xử lý riêng của chữ ký đường duy trì; NSE/KGE và chuỗi Q không bị thay đổi. Các lựa chọn nội suy/lấy số ngày này được ghi trong `Evaluation_protocol.json`; không xem chúng là đối chiếu từng bit với cách lấy hạng rời rạc của NeuralHydrology.

Mỗi chỉ số được tổng hợp qua lưu vực bằng trung vị, trung bình và P10/P25/P75/P90, kèm số ID hữu hạn. Chữ ký FDC có cả trung vị có dấu và trung vị trị tuyệt đối để tránh triệt tiêu hai hướng lệch.

## 4. Bốn mức lưu lượng

Dùng q30/q80/q98 A1 trên tham chiếu 1981–2013: thấp khi q < q30, trung bình khi q30 ≤ q < q80, cao khi q80 ≤ q < q98, rất cao khi q ≥ q98. Ba ngưỡng phải hữu hạn, tăng nghiêm ngặt và đúng danh sách ID.

Ma trận nhầm lẫn có hàng là mức quan trắc, cột là mức dự báo. Xuất số ngày của cả 16 ô, recall, precision, F1 từng mức và accuracy, cả theo lưu vực và cộng dồn. Mức không có ngày quan trắc có recall NaN; mức không có lần báo có precision NaN. Ví dụ, 10 ngày rất cao thật được báo đúng 7 ngày cho recall 0,7; nếu mô hình báo rất cao 14 ngày thì precision là 0,5.

## 5. Sự kiện lũ và cờ kiểm duyệt

Ngưỡng lấy từ `half_calendar_primary` của A1 cho T = 1,5; 2; 5; 10 năm, giữ số năm fit theo từng lưu vực. Bộ đánh giá áp cùng ngưỡng và quy tắc cho obs/sim; xuất cả đoạn liên tiếp và cụm USWRC. [Giao thức đã chốt](../../../Document/01_Plan/02_Pipeline.md#62-theo-mức-lưu-lượng-và-theo-sự-kiện-yêu-cầu-gvhd).

Một đoạn gồm các ngày liên tiếp có Q **lớn hơn** ngưỡng. Quy tắc gộp xét khoảng cách đỉnh tối thiểu ceil[5 + ln(A dặm vuông)] và yêu cầu đáy giữa hai đỉnh thấp hơn 0,75 lần đỉnh nhỏ hơn; chưa đủ một trong hai điều kiện thì gộp. Sau gộp, kiểm lại cụm kề. Đồng đỉnh dùng ngày đỉnh cuối bên trái và đầu bên phải để tính khoảng cách gần nhất. [HEC-USACE giải thích khoảng cách theo diện tích](https://www.hec.usace.army.mil/confluence/sspdocs/sspum/latest/peaks-over-threshold-analysis/peaks-over-threshold-filtering).

Ngày thiếu ngăn việc nối cụm. Đoạn chạm biên kỳ hoặc có ngày trống ngay trước/sau giữ cờ onset/ending không quan sát được. Điểm sự kiện chính chỉ dùng cụm có cả hai biên được quan sát và đủ ngày hợp lệ trong bán kính ghép quanh onset; số cụm bị kiểm duyệt được báo riêng. Đây là quy ước đánh giá trên lịch có thiếu của đề tài; dữ liệu mất quanh một trận có thể khiến trận đó nằm ngoài phạm vi chấm.

Ghép sự kiện theo **ngày bắt đầu (onset)**, một–một, tối đa hóa số cặp trước, rồi tối thiểu hóa tổng sai lệch ngày. **Bảng chính dùng ±⌊hạn đang chấm/3⌋ ngày; bảng phụ luôn dùng ±1 ngày**, xuất riêng với hậu tố `_tolerance1`. Dung sai áp dụng như nhau cho mọi run cùng hạn. `forecast_horizon` chỉ kiểm hợp đồng tổng số hạn, không thay dung sai của hạn đang chấm: hạn 1 trong mô hình bảy hạn vẫn yêu cầu bắt đầu cùng ngày. Sai số thời điểm đỉnh dùng ngày đầu trong trường hợp đồng đỉnh, sau khi cặp sự kiện đã được xác định.

| Hạn dự báo (ngày) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dung sai chính (±ngày) | 0 | 0 | 1 | 1 | 1 | 2 | 2 |
| Dung sai phụ (±ngày) | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

Ở hạn 3–5, hai bảng có cùng dung sai nhưng vẫn xuất riêng theo vai trò và không nhân đôi số trận. Ví dụ, trận thật bắt đầu ngày 10, dự báo bắt đầu ngày 12: không khớp ở hạn 1–5; khớp ở hạn 6–7 nếu hai trận đủ điều kiện chấm. Cửa sổ đối xứng cũng chấp nhận lệch sớm cùng độ lớn.

[NOAA CR TA 98-03, Mục III.B](https://repository.library.noaa.gov/view/noaa/30340/noaa_30340_DS1.pdf) dùng cửa sổ ±1/3 thời gian từ phát hành đến thời điểm vượt ngưỡng dự báo; 24 giờ tương ứng ±8 giờ. Yêu cầu cùng ngày là cách áp dụng cho lịch ngày của đề tài, không phải kiểm chứng sai số trong tám giờ. [Nearing và cs. (2024), Methods–Metrics](https://www.nature.com/articles/s41586-024-07145-1) dùng cửa sổ **±2 ngày**; bảng phụ ±1 phục vụ phân tích độ nhạy khi đối chiếu, không phải tái lập nguyên giao thức Nearing.

Mỗi bảng áp bán kính kiểm duyệt tương ứng với dung sai của mình. Vì vậy số trận đủ điều kiện có thể khác giữa bảng chính và phụ; mỗi bảng giữ riêng số bắt được, bỏ sót, báo nhầm và số bị kiểm duyệt. Ở hạn một và hai ngày, trận trễ một ngày không được tính là bắt đúng trong bảng chính, kể cả khi hai trận còn chồng lấn.

Quy tắc lấy phần nguyên ±⌊hạn/3⌋ là cách rời rạc hóa trên lịch ngày của đề tài, thống nhất với kế hoạch bước H. Đây không phải xác minh sai số theo giờ của NOAA; hạn một ngày tương ứng ±8 giờ ở quy tắc thời gian liên tục, còn dữ liệu hiện tại chỉ cho phép phân biệt ngày lịch.

Từ số bắt được h, bỏ sót m, báo nhầm f, tính POD = h/(h+m), FAR = f/(h+f), precision = h/(h+f), F1 = 2h/(2h+m+f), CSI = h/(h+m+f). Khi mẫu số bằng 0, kết quả là NaN. Bảng chính cộng số trận trước khi chia, tách riêng từng T và cách gộp; đồng thời có kết quả cho tất cả ID, nhóm fit ≥10 năm và nhóm fit <10 năm. Các nhóm này là các cách trình bày song song, không cộng với nhau.

## 6. Kiểm định và kịch bản

Wilcoxon signed-rank hai phía ghép theo ID trên điểm `common_source_all`, kèm Cohen's dz = trung bình hiệu / std mẫu của hiệu. Dấu hiệu là **run bên trái trừ run bên phải**. Các ID có một trong hai chỉ số không xác định được loại khỏi đúng phép kiểm đó và được phản ánh trong số cặp. Nếu mọi hiệu bằng 0, p = 1 theo quy ước; std hiệu bằng 0 làm dz không xác định. [Tài liệu SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).

Họ Holm mặc định gồm mọi cặp run không thứ tự, bỏ mốc qmean phụ, riêng theo chỉ số NSE/PNSE/KGE và từng nhóm kỳ/kịch bản/hạn. Với sáu A2, ba DLinear và persistence qmax, mỗi họ dự kiến có 45 phép so. **Chỉ p-value hữu hạn được tính vào kích thước họ hiệu chỉnh** (`family_size`); p = NaN được giữ NaN và không tăng hệ số nhân. `planned_family_size` ghi số phép so dự kiến. Ví dụ, p = [0,01; NaN; 0,03] cho p sau Holm = [0,02; NaN; 0,03]. Báo p gốc, p hiệu chỉnh, số cặp hữu hạn, số lưu vực tốt lên/kém đi/hòa và cỡ hiệu ứng. Các lưu vực lồng nhau có thể phụ thuộc không gian, nên p cần được đọc cùng cỡ hiệu ứng và phân phối chênh lệch.

Bộ đánh giá nhận dự báo S0–S4 đã được bên mô hình tạo ra; không tạo kịch bản bằng cách làm nhiễu dự báo. Chênh lệch với S0 được **tính lại trên giao ngày của S0 và đúng kịch bản**, thay vì trừ hai median khác phạm vi. Từng hạn được chấm riêng. Với hạn k, persistence giữ qmax ở origin−1 cho mọi hạn, không lấy số đo đã xuất hiện sau origin. Persistence trong các kịch bản mất Q là mốc quan trắc để diễn giải điểm, không được xem là mô hình vận hành có số đo sẵn.

Khi thêm ba A4, nhóm test/S0/hạn một ngày có 14 run; bỏ qmean phụ còn 13 run, tương ứng **78 cặp dự kiến mỗi chỉ số và 234 hàng kiểm định cho ba chỉ số**. Các p Holm được tính lại theo họ mở rộng, không chép p đã điều chỉnh từ họ 45 cặp. Tập ID vẫn giữ đủ 451; số cặp có chỉ số hợp lệ được ghi riêng.

## 7. Hợp đồng NPZ cho dự báo bước A4/B/C/H

Mỗi NPZ mở rộng chứa một run/seed/kỳ/kịch bản/hạn. Truyền tệp bằng `--extra-contract`; có thể lặp tham số cho nhiều tệp. Tên run cần chứa seed để phân biệt ba lần chạy. Mỗi nhóm phải có cùng lịch và thứ tự 451 ID.

| Trường NPZ | Nội dung |
|---|---|
| `dates` | Ngày đích, lịch liên tục |
| `origin_dates` | Ngày đích đầu của đợt dự báo: dates − (lead−1) |
| `persistence_source_dates` | dates − lead, tức origin−1 |
| `basin_ids`, `area_gov_km2` | Thứ tự lưu vực và diện tích trạm |
| `obs_mm_day`, `persistence_mm_day` | Quan trắc canonical và mốc persistence từ nguồn Q |
| `sim_raw_mm_day`, `sim_mm_day` | Dự báo trước/sau chính sách chặn âm đã khai báo |
| `metadata_json` | Chuỗi JSON: run, period, scenario, lead, forecast_horizon, seed, unit, config_sha256, library_commit, clip_to_zero |

Đơn vị bắt buộc là `mm/day_area_gov`; m³/s khôi phục bằng q × area_gov / 86,4. Metadata có thể lưu thêm ngày phát hành thực tế và đường dẫn nguồn. `origin_dates` là quy ước cửa sổ của mô hình, không tự xác nhận giờ/ngày sản phẩm khí tượng được phát hành. Bên xuất chịu trách nhiệm lấy persistence đúng giá trị quan trắc ở ngày nguồn; bên đánh giá kiểm quan hệ ngày và sự nhất quán giữa các run.

### 7.1. Nhận tự động ba hợp đồng A4

Phần 8.4 đọc `manifest.csv` của `a4_compare` với các cột `tệp`, `sha256`, kiểm mọi tệp được liệt kê, từ chối đường dẫn ra ngoài thư mục và tên trùng. Cần đủ ba NPZ, summary và environment; ba seed không được bỏ qua khi thiếu tệp. Hash hai nguồn `A3_persistence_predictions.npz`, `A3_Q_source_window.npz` trong metadata A4 phải khớp đúng Input A3 đang dùng.

Các trường run, seed, kỳ, hạn, đơn vị, config SHA-256, commit thư viện, chính sách clip và epoch được kiểm trước khi gọi `check_same_context`. Epoch phải trùng `saved_epoch` trong summary. Version nguồn huấn luyện được đối chiếu với bản bàn giao: seed 111 Version 1, seed 222/333 Version 2, A3 Version 1. Version 2 của notebook Compare là phiên bản yêu cầu theo [báo cáo A4](../../03_Baseline/03_A4_LamaHCEExt_Retrain/03_A4_LamaHCEExt_Retrain.md#p5); giá trị băm ghi danh tính tệp thực tế, không gọi đây là truy vấn xác nhận Version trực tiếp từ Kaggle.

Ba run có tên `SeqLSTM_Q_A4_seed111`, `SeqLSTM_Q_A4_seed222`, `SeqLSTM_Q_A4_seed333`. Các tệp này được nhận tự động, không cần truyền `--extra-contract` cho A4. Tham số đó vẫn dùng cho dự báo mở rộng khác; trùng tên run/kỳ/kịch bản/hạn làm chương trình dừng.

## 8. Các phép kiểm đã thực hiện cục bộ

Phép kiểm độc lập bao gồm ví dụ NSE/PNSE/KGE và tham chiếu Pearson từ SciPy; chuỗi hằng, phương sai nhỏ, zero-SSE, NaN/mask/vô cực; tính bất biến đổi đơn vị; ngưỡng bằng biên; USWRC với đáy bằng 0,75 và khoảng thiếu; ghép sự kiện không tái sử dụng; persistence trễ một ngày bị loại ở bảng chính hạn 1 nhưng được bắt ở bảng phụ; Holm bỏ NaN và dz với số biết trước; kịch bản cùng ngày chấm. Kiểm nhiều hạn bao phủ cả bảy hạn, biên lệch sớm/trễ trong và ngoài dung sai, kiểm duyệt quanh ngày thiếu/biên lịch, tách bảng chính/phụ khi cùng dung sai, cùng kiểm ngày nguồn persistence tại origin−1.

**129 kiểm thử độc lập của giao thức 3 đều đạt.** Đầu vào hạn, tổng số hạn và dung sai sai kiểu hoặc ngoài phạm vi được từ chối; bool không được xem là số nguyên. Môi trường kiểm cục bộ: Python 3.12.6, NumPy 1.26.4, pandas 2.3.3, SciPy 1.16.3 và xarray 2025.12.0. Kiểm hồi quy thêm 400 ca sự kiện hạn 1, gồm hai cách gộp, dung sai 0/1 và ngày thiếu, khớp chính xác cách ghép của giao thức 2. Đã đối chiếu thêm 120 ca chỉ số với hàm A3 giữ nguyên, 60 ca gộp cụm với hàm A1 giữ nguyên và 80 ca ghép sự kiện với lời giải duyệt hết các cách ghép; tất cả đều khớp.

Fixture I/O dùng hai ID và lịch test đầy đủ, tạo tệp đúng cấu trúc CSV A1, pickle xarray A2, NPZ DLinear và hợp đồng nhiều hạn; sau đó đi qua bộ đọc, chấm, cộng dồn, ghép bảng tham chiếu và kiểm ngày nguồn persistence. Mọi số liệu fixture được tạo trong thư mục tạm và không được xuất thành kết quả LamaH-CE trong repo.

Các kiểm nhiều hạn bao phủ cả hạn 1–7, gồm phân luồng hai bảng chính/phụ khi cùng dung sai và giữ schema bảng ghép rỗng. Hàm chạy nhận `expected_ids` riêng cho fixture; cách chạy chính thức mặc định yêu cầu 451 ID. Trạng thái hoàn thành của fixture chỉ xác nhận đường chạy dữ liệu giả.

**36 kiểm tích hợp A4 cục bộ đạt**, bổ sung cho 129 kiểm hồi quy của giao thức. Fixture gồm ba ID và đủ lịch test, có ID không được chấm ở giao chung, các tên `environment.json` ở cả A2/A4 và một ngày A4 thiếu thêm. Đã kiểm các trường hợp thiếu seed, sai hash, nguồn A3 khác, sai Version/epoch/seed/đơn vị/commit, sai chính sách clip, đảo ID, sai ngày persistence, trùng nguồn và đường dẫn manifest ra ngoài thư mục.

Đường chạy tự đọc A4 chấm đủ 14 run và xuất **24 tệp** cho test/S0/hạn một ngày, gồm hai bảng kiểm A4 mới. Toàn bộ 16 bảng tính toán đã có khớp cách truyền cùng ba NPZ bằng bộ đọc hợp đồng trước đó. Fixture giảm giao chung đúng một ngày do A4 thiếu thêm, trong khi bảy phép đối chiếu A3 vẫn đạt trên giao riêng của 11 run gốc. Họ Holm có 78 cặp dự kiến mỗi chỉ số, tổng 234 hàng; bảng lũ giữ tolerance chính 0 và phụ 1. Cờ `complete` được đặt false trước khi kiểm nguồn, nên một lần chạy dừng không để lại cờ hoàn thành của lần cũ.

Khi chạy trên Input thật, bảy kiểm đối chiếu A3 gồm sáu chỉ số và số ngày cho từng run–ID ở giao chung, với rtol/atol = 10⁻¹⁰ cho giá trị số; NaN phải cùng vị trí. Sai khác làm chương trình dừng trước cờ hoàn thành. Đây là đối chiếu float64 A3; số KGE CSV A2 float32 được diễn giải trong thuyết minh A3.

## 9. Đầu ra cần giữ và gửi lại

| Nhóm | Tệp chính |
|---|---|
| Kiểm và nguồn | `Evaluation_checks.csv`, `Evaluation_input_manifest.csv`, `Evaluation_A3_reference_check.csv` |
| Bàn giao A4 | `Evaluation_A4_input_check.csv` (manifest và hai nguồn A3), `Evaluation_A4_contract_check.csv` (ba seed, epoch, độ phủ, số dự báo âm và ngữ cảnh) |
| Chỉ số và độ phủ | `Evaluation_metrics.csv`, `Evaluation_summary.csv`, `Evaluation_coverage.csv`, `Evaluation_paired_tests.csv` |
| Bốn mức | `Evaluation_confusion.csv`, `Evaluation_levels.csv`, `Evaluation_levels_pooled.csv` |
| Sự kiện chính, ±⌊hạn/3⌋ ngày | `Evaluation_events.csv`, `Evaluation_event_matches.csv`, `Evaluation_events_pooled.csv` |
| Sự kiện phụ, ±1 ngày | `Evaluation_events_tolerance1.csv`, `Evaluation_event_matches_tolerance1.csv`, `Evaluation_events_pooled_tolerance1.csv` |
| Kịch bản và hình | `Evaluation_scenario_changes.csv`, `Evaluation_CDF_points.csv`, PNG CDF từng nhóm |
| Quy ước và kết luận | `Evaluation_protocol.json`, `Evaluation_run_index.json`, `Evaluation_summary.json` |

Với đầu vào A2/DLinear/A4, phạm vi là test/S0/hạn một ngày. JSON kết luận có `A4_contracts_matched`, danh sách `A4_runs` và `A4_compare_version_expected`; chỉ đặt hoàn thành sau khi chấm và đối chiếu A3 thành công.

JSON ghi `protocol_version = 3`, tên chính sách `event_tolerance` và `event_tolerance_days_by_lead` cho các hạn thực sự được chấm. Chấm đúng một hạn giữ thêm `event_tolerance_days` là số nguyên; chấm nhiều hạn để trường scalar này là null và dùng bản đồ theo hạn. `matching_tolerance_days` trong từng dòng sự kiện và cặp ghép ghi dung sai đã áp dụng.

[Notebook phân tích thuộc tính 02](../02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md) đọc Output Version 4 này, chạy 18 phép so trên 14 run. Khi bổ sung dự báo của bước B/C/H, giữ cùng các tệp đầu ra để đối chiếu thêm theo kịch bản và hạn. Bảng chênh lệch kịch bản có đầy đủ tên cột ngay cả khi chưa có dòng dữ liệu ngoài S0.

## 10. Kết quả thực nghiệm trên Kaggle

### 10.1. Phạm vi và tính nhất quán

Kết quả dưới đây là Output của **Version 4**, Save & Run All trên CPU (72 giây), giao thức 3, test/S0/hạn một ngày. Input là Output A1 `01-a1-lamahceext-explore`, A2 `01-a2-lamahceext-reproduce`, A3 `02-a3-lamahceext-benchmarks` và A4 `04-a4-lamahceext-compare` Version 2. Môi trường Kaggle: Python 3.13.15, NumPy 2.1.3, pandas 2.3.3, SciPy 1.16.3. [Kết luận chạy](Tables/Evaluation_summary.json), [nguồn đầu vào](Tables/Evaluation_input_manifest.csv).

| Kiểm tra | Kết quả |
|---|---|
| Kiểm thử trước khi chấm | 129/129 đạt |
| Đối chiếu A3 | 7/7 kiểm đạt; mỗi kiểm gồm 4.961 cặp run–ID: NSE, PNSE, KGE, r, α, β và n |
| Nguồn A4 | Manifest 11 tệp và hai nguồn ngữ cảnh A3 đạt kiểm hash; ba hợp đồng có cùng lịch, ID, diện tích, obs canonical và persistence với A3 |
| Giao ngày chung | 652.628 ngày–ID; 450 ID có ngày chấm, giữ ID 55 với n = 0 |
| Chỉ số liên tục | 18.942 dòng = 14 run × 451 ID × 3 scope |
| Ma trận bốn mức | 101.024 dòng = 14 run × 451 ID × 16 ô; tổng mỗi ma trận khớp số ngày chấm |
| Sự kiện chính, dung sai 0 | 50.512 dòng theo run/ID/T/cách gộp; 10.823 cặp ghép một–một |
| Sự kiện phụ, ±1 ngày | 50.512 dòng; 30.074 cặp ghép một–một |
| Kiểm định | 234 phép so = 78 cặp × 3 chỉ số; Holm tách theo từng chỉ số, mọi p hữu hạn |

NSE/PNSE có 450 ID hữu hạn trên giao chung cho mọi run. KGE/r của EncDec_noQ có 448 ID hữu hạn do đầu ra hằng tại 742 và 758. ID 55 có trạng thái không có cặp hợp lệ ở mọi run trên giao chung. [Kiểm thử](Tables/Evaluation_checks.csv), [đối chiếu A3](Tables/Evaluation_A3_reference_check.csv), [kiểm nguồn A4](Tables/Evaluation_A4_input_check.csv).

| Run A4 | Epoch chọn | Ngày–ID có dự báo | Ngày–ID có obs/sim hữu hạn | Dự báo âm trước clip |
|---|---:|---:|---:|---:|
| Seed 111 | 11 | 652.644 | 652.636 | 1.660 |
| Seed 222 | 15 | 652.644 | 652.636 | 2.261 |
| Seed 333 | 14 | 652.644 | 652.636 | 1.978 |

Mỗi hợp đồng A4 giữ 451 ID trên 1.461 ngày đích. Số 652.644 mô tả các dự báo hữu hạn trước khi yêu cầu có obs; 652.636 là số cặp obs/sim của riêng mỗi A4, còn 652.628 là giao để so toàn bộ 14 run. Thêm A4 không làm giảm giao chung so với giao của 11 run gốc. [Bảng hợp đồng A4](Tables/Evaluation_A4_contract_check.csv).

![Phân phối NSE, PNSE và KGE trên giao ngày chung](Figures/Evaluation_CDF_test_S0_lead1.png)

*Hình 1. Mỗi điểm là chỉ số của một lưu vực; giữ toàn bộ giá trị hữu hạn, kể cả phần đuôi âm. Các trục dùng symlog, tuyến tính trong [−1, 1]. Ba đường A4 màu xanh dùng kiểu nét khác nhau cho từng seed; đường SeqLSTM_Q liền là trọng số tác giả. Số n trong chú giải là số ID có chỉ số hữu hạn. Đường persistence PNSE nằm tại 0 theo định nghĩa.*

### 10.2. Kết quả liên tục và bốn mức

| Run | NSE trung vị | PNSE trung vị | KGE trung vị | %BiasFHV trung vị |
|---|---:|---:|---:|---:|
| SeqLSTM_Q, trọng số tác giả | 0,705259 | 0,495584 | 0,767074 | −13,01 |
| SeqLSTM_noQ | 0,622922 | 0,375311 | 0,688235 | −17,88 |
| EncDec_Q | 0,662155 | 0,435417 | 0,698093 | −17,31 |
| EncDec_noQ | 0,567347 | 0,285036 | 0,640148 | −17,13 |
| D_FC | 0,385892 | 0,076250 | 0,533138 | −22,63 |
| D_RA | 0,692511 | 0,485222 | 0,737006 | −14,08 |
| DLinear 111 | 0,382799 | 0,066592 | 0,519476 | −31,35 |
| DLinear 222 | 0,378069 | 0,055647 | 0,520644 | −30,78 |
| DLinear 333 | 0,379588 | 0,059714 | 0,513177 | −31,91 |
| LSTM A4 seed 111 | 0,695242 | 0,495352 | 0,758090 | −15,70 |
| LSTM A4 seed 222 | 0,703778 | 0,505573 | 0,790977 | −11,71 |
| LSTM A4 seed 333 | 0,705890 | 0,501788 | 0,785984 | −11,56 |
| Persistence qmax | 0,350031 | 0 | 0,674939 | 0,00 |
| Persistence qmean, phụ | 0,366131 | 0,034721 | 0,457145 | −37,15 |

Giá trị A2, DLinear và persistence khớp A3. KGE trung vị của ba DLinear khoảng 0,513–0,521, thấp hơn persistence 0,674939. [Bảng đầy đủ](Tables/Evaluation_summary.csv).

Trung bình ± độ lệch chuẩn mẫu của ba NSE trung vị A4 là **0,701636 ± 0,005638**; khoảng [0,695242; 0,705890] bao quanh mốc tác giả 0,705259. Đây là biến thiên của ba lần huấn luyện, không phải khoảng tin cậy hay chỉ số của một ensemble. Ba seed đều giữ riêng; không chọn seed theo test. NSE của A4 tương đối sát mốc tác giả nhưng KGE thay đổi theo seed: 111 thấp hơn mốc về trung vị, 222 và 333 cao hơn. Với PNSE, cả ba median dương nhưng mean lần lượt **−1,189; −0,508; −0,355**; cần giữ phần đuôi các lưu vực mà persistence mạnh khi đọc kết quả.

Trên giao chung có **194.703 ngày–ID mức thấp** và **11.759 ngày–ID mức rất cao**. Các tỷ lệ dưới đây cộng ô đếm trước khi chia, không phải trung vị chỉ số theo lưu vực.

| Run | Recall mức thấp (%) | Recall mức rất cao (%) | Precision mức rất cao (%) |
|---|---:|---:|---:|
| SeqLSTM_Q, trọng số tác giả | 66,93 | 47,58 | 67,58 |
| SeqLSTM_noQ | 65,79 | 36,17 | 63,73 |
| EncDec_Q | 53,67 | 41,31 | 62,22 |
| EncDec_noQ | 54,95 | 38,50 | 51,30 |
| D_FC | 47,23 | 21,49 | 41,75 |
| D_RA | 65,06 | 47,27 | 63,78 |
| DLinear 111 | 61,83 | 22,18 | 53,33 |
| DLinear 222 | 67,61 | 22,30 | 53,72 |
| DLinear 333 | 69,52 | 21,80 | 54,86 |
| LSTM A4 seed 111 | 71,53 | 43,84 | 69,63 |
| LSTM A4 seed 222 | 76,29 | 48,39 | 66,53 |
| LSTM A4 seed 333 | 73,17 | 47,79 | 65,68 |
| Persistence qmax | 88,07 | 40,11 | 40,28 |
| Persistence qmean, phụ | 96,27 | 18,40 | 61,53 |

SeqLSTM_Q bắt được 5.595/11.759 ngày rất cao, nhiều hơn persistence qmax 4.716 ngày; precision cũng cao hơn. Ba A4 bắt được 5.155/5.690/5.620 ngày rất cao, quanh mức của trọng số tác giả. DLinear đạt khoảng 22% recall mức rất cao, phù hợp với độ co biên độ và hụt dải cao đã thấy ở A3. Accuracy tổng thể của persistence qmax là 82,90%, cao hơn SeqLSTM_Q 77,62% và ba A4 79,34/80,54/79,77%; tỷ lệ đúng chung chịu ảnh hưởng lớn của các mức phổ biến, vì vậy phải đọc riêng recall ở hai đầu dải lưu lượng. [Bảng phân mức](Tables/Evaluation_levels_pooled.csv).

%BiasFHV âm ở mọi mô hình học, thể hiện hụt tổng của dải 2% lưu lượng cao nhất theo định nghĩa FHV; DLinear hụt nhiều nhất, khoảng −31%. FLV của DLinear có trị tuyệt đối trung vị lớn, khoảng 597–754%; chỉ số này đo hình dạng đuôi log, không phải phần trăm thiếu thể tích nước. Trên giao chấm, ba seed DLinear có 9.197/12.134/13.218 giá trị sim dưới sàn log, chủ yếu do chặn các dự báo âm về 0. Khi đọc FLV cần giữ cùng sàn 10⁻⁶ mm/ngày và số ngày chịu phép log này.

### 10.3. Sự kiện lũ chu kỳ hai năm

#### 10.3.1. Bảng chính: cùng ngày bắt đầu

Theo quy tắc USWRC, có **1.066 cụm quan trắc đủ điều kiện trên 366 lưu vực**. Bảng cộng số trận trước khi tính tỷ lệ, gồm cả nhóm ngưỡng fit dưới 10 năm; CSV có bảng tách riêng nhóm đó.

| Run | Bắt được | Bỏ sót | Báo nhầm | POD (%) | FAR (%) |
|---|---:|---:|---:|---:|---:|
| SeqLSTM_Q, trọng số tác giả | 208 | 858 | 431 | 19,51 | 67,45 |
| SeqLSTM_noQ | 188 | 878 | 304 | 17,64 | 61,79 |
| EncDec_Q | 126 | 940 | 289 | 11,82 | 69,64 |
| EncDec_noQ | 174 | 892 | 445 | 16,32 | 71,89 |
| D_FC | 63 | 1.003 | 286 | 5,91 | 81,95 |
| D_RA | 219 | 847 | 377 | 20,54 | 63,26 |
| DLinear 111 | 2 | 1.064 | 327 | 0,19 | 99,39 |
| DLinear 222 | 2 | 1.064 | 340 | 0,19 | 99,42 |
| DLinear 333 | 2 | 1.064 | 322 | 0,19 | 99,38 |
| LSTM A4 seed 111 | 180 | 886 | 372 | 16,89 | 67,39 |
| LSTM A4 seed 222 | 216 | 850 | 446 | 20,26 | 67,37 |
| LSTM A4 seed 333 | 191 | 875 | 446 | 17,92 | 70,02 |
| Persistence qmax | 0 | 1.066 | 1.066 | 0,00 | 100,00 |
| Persistence qmean, phụ | 0 | 1.066 | 201 | 0,00 | 100,00 |

Nguồn: [bảng chính](Tables/Evaluation_events_pooled.csv). D_RA dùng tái phân tích ở ngày đích nên là mốc tham chiếu về nguồn thông tin. SeqLSTM_Q bắt được nhiều trận hơn các cấu hình DLinear, nhưng vẫn bỏ sót 858/1.066 trận. Ba seed DLinear chỉ bắt đúng ngày khởi phát ở hai trận dù NSE trung vị nhỉnh hơn persistence; cải thiện sai số cả chuỗi không bảo đảm cải thiện thời điểm khởi phát lũ. Seed 333 có NSE trung vị cao nhất trong ba A4, còn seed 222 bắt đúng ngày nhiều trận nhất, nên xếp theo NSE không tự xác định thứ tự về bắt lũ. Cả ba A4 bỏ sót 850–886/1.066 trận, FAR khoảng 67–70%. Cùng ngày bắt đầu vẫn có thể khác ngày đỉnh, nên sai số thời điểm đỉnh được báo riêng.

#### 10.3.2. Bảng phụ: ±1 ngày

Giữ cùng ngưỡng và quy tắc gộp, cho phép onset lệch tối đa một ngày. Ở T = 2 năm, mẫu số cũng là 1.066 cụm trên 366 lưu vực; tại ngưỡng khác cần đọc số kiểm duyệt của từng bảng.

| Run | Bắt được | Bỏ sót | Báo nhầm | POD (%) | FAR (%) |
|---|---:|---:|---:|---:|---:|
| SeqLSTM_Q, trọng số tác giả | 348 | 718 | 291 | 32,65 | 45,54 |
| SeqLSTM_noQ | 256 | 810 | 236 | 24,02 | 47,97 |
| EncDec_Q | 218 | 848 | 197 | 20,45 | 47,47 |
| EncDec_noQ | 226 | 840 | 393 | 21,20 | 63,49 |
| D_FC | 94 | 972 | 255 | 8,82 | 73,07 |
| D_RA | 325 | 741 | 271 | 30,49 | 45,47 |
| DLinear 111 | 222 | 844 | 107 | 20,83 | 32,52 |
| DLinear 222 | 237 | 829 | 105 | 22,23 | 30,70 |
| DLinear 333 | 222 | 844 | 102 | 20,83 | 31,48 |
| LSTM A4 seed 111 | 306 | 760 | 246 | 28,71 | 44,57 |
| LSTM A4 seed 222 | 363 | 703 | 299 | 34,05 | 45,17 |
| LSTM A4 seed 333 | 333 | 733 | 304 | 31,24 | 47,72 |
| Persistence qmax | 1.066 | 0 | 0 | 100,00 | 0,00 |
| Persistence qmean, phụ | 117 | 949 | 84 | 10,98 | 41,79 |

**POD = 1 của persistence phản ánh dung sai ghép ±1 ngày trong bảng phụ.** Chuỗi chép qmax hôm qua dịch sự kiện sang hôm sau; độ lệch onset một ngày vẫn được tính là bắt được ở bảng này. Sai số tuyệt đối thời điểm đỉnh trung bình là đúng **1 ngày**. Bảng chính cùng ngày bắt đầu loại các cặp trễ như vậy. Bảng phân mức đúng ngày cho recall mức rất cao của persistence là 40,11%.

Ở bảng phụ, persistence cũng đạt POD = 1 tại T = 5 và 10 năm; T = 1,5 năm đạt 1.849/1.850 ≈ 99,95%. Ở bảng chính, persistence không có cặp đúng ngày tại cả bốn ngưỡng và cả hai cách gộp. Các mẫu số và nhóm fit ngắn được giữ trong [bảng phụ](Tables/Evaluation_events_pooled_tolerance1.csv).

### 10.4. So sánh ghép cặp

Mỗi chỉ số có một họ Holm gồm 78 cặp của 13 run (bỏ mốc qmean phụ). Trên NSE, DLinear 111/222/333 hơn persistence ở **318/450, 309/450, 317/450** ID; median chênh lệch tương ứng **0,035019/0,031839/0,031465**. p sau Holm lần lượt khoảng 8,85×10⁻²⁹; 4,27×10⁻²⁶; 3,22×10⁻²⁷. Cỡ hiệu ứng dz lần lượt **0,0506/0,4480/0,3933**; khác biệt giữa seed cần được đọc cùng các lưu vực ở phần đuôi âm.

Trên KGE, mỗi seed DLinear thấp hơn persistence ở **449/450** ID. Trên PNSE, median của ba seed dương nhưng mean âm; một số lưu vực có persistence rất mạnh làm tỷ số sai số lớn. Vì vậy, p nhỏ hoặc đa số lưu vực tốt lên không thay thế việc kiểm mức thiệt hại ở các lưu vực dự báo kém.

| A4 trừ SeqLSTM_Q tác giả, trên NSE | ID A4 tốt hơn | Median hiệu | p sau Holm |
|---|---:|---:|---:|
| Seed 111 | 209/450 | −0,002327 | 0,481862 |
| Seed 222 | 249/450 | 0,003656 | 0,288877 |
| Seed 333 | 254/450 | 0,004786 | 0,530678 |

Hướng hiệu trong bảng được đổi từ CSV để thống nhất **A4 trừ tác giả**; p hai phía giữ nguyên. Không phát hiện khác biệt NSE ở mức 0,05 trong họ 78 cặp này; kết quả đó không phải phép thử tương đương. So persistence, ba A4 tốt hơn về NSE ở **418/424/428 trên 450 ID**, với median hiệu **0,327901/0,339101/0,329561**.

NSE seed 111 so 222 có p Holm **0,025099**, seed 111 so 333 có p Holm **0,051454**, trong khi median hiệu chỉ khoảng −0,00384 và −0,00444. Họ kiểm ở đây có 78 cặp mỗi chỉ số, khác họ ba cặp seed trong báo cáo A4; vì vậy không dùng hoán đổi p đã điều chỉnh của hai bảng. Quyết định về một cải tiến cần đọc cả độ lớn hiệu, biến thiên nhiều seed và phạm vi dữ liệu, không chỉ p < 0,05. [Toàn bộ 234 phép so](Tables/Evaluation_paired_tests.csv).

### 10.5. Phạm vi xác nhận và tệp lưu

Lần chạy xác nhận bộ đánh giá trên dự báo thực nghiệm **test/S0/hạn một ngày**, gồm chỉ số liên tục, phân mức, sự kiện và kiểm định. Các kịch bản S1–S4, nhiều hạn và tập validation sẽ được chấm khi có dự báo tương ứng từ luồng mô hình; hai cờ `operational_scenarios_run` và `multi_lead_run` trong JSON giữ false đúng phạm vi này.

`Tables/` và `Figures/` giữ nguyên byte 15 trong 24 tệp Output Version 4: các bảng kiểm, bảng tổng hợp, JSON và hình CDF. Chín bảng chi tiết theo từng ID hoặc từng cặp ghép (`Evaluation_metrics.csv`, `Evaluation_confusion.csv`, `Evaluation_levels.csv`, `Evaluation_events.csv`, `Evaluation_event_matches.csv`, hai bản `_tolerance1`, `Evaluation_CDF_points.csv`, `Evaluation_scenario_changes.csv`; khoảng 36 MB) nằm trong Output Kaggle; notebook 02 và các bước sau gắn thẳng Output đó làm Input.
