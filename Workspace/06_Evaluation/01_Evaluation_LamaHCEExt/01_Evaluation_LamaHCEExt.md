# Bộ đánh giá dự báo lưu lượng — Extended LamaH-CE

Tài liệu đi cùng [mã Python](01_Evaluation_LamaHCEExt.py) và [notebook](01_Evaluation_LamaHCEExt.ipynb) (bản chính thức: Version 2 của notebook Kaggle `01_Evaluation_LamaHCEExt`, Save & Run All trên CPU, có output), thực hiện việc #5 trong [phân công](../../../ASSIGNMENT.md) và [pipeline Mục 6](../../../Document/01_Plan/03_Pipeline.md#6-đánh-giá). Bộ đánh giá đọc dự báo đã lưu, tính chỉ số, đánh giá bốn mức và sự kiện lũ, so sánh theo cặp lưu vực, rồi xuất CSV/PNG/JSON trực tiếp cho Kaggle Save Version.

**S0** là kịch bản không che thêm đầu vào; dữ liệu gốc vẫn có thể thiếu. **Hạn một ngày** là dự báo ngày kế tiếp sau ngày cuối của lịch sử quan trắc. Một **ngày–lưu vực** là một ngày tại một ID; **mặt nạ** là bảng True/False chọn các cặp đó để chấm. **Giao chung** chỉ giữ ngày–lưu vực có quan trắc, persistence và dự báo của mọi run cùng hữu hạn.

**Bộ đánh giá giao thức 2 đã chạy trên Output A1/A2/A3: 11 run, test 2014–2017, S0, hạn một ngày; 60 kiểm thử Kaggle và 57 kiểm tra CSV cục bộ đạt.** Sáu chỉ số và số ngày chấm khớp A3 ở 4.961 cặp run–lưu vực; giao chung có 652.628 ngày–lưu vực trên 450 ID có cặp hợp lệ. Giao thức dùng **cùng ngày bắt đầu cho bảng lũ chính, ±1 ngày cho bảng phụ và Holm chỉ đếm p hữu hạn**. Ở ngưỡng lũ hai năm, SeqLSTM_Q có POD 19,51%, D_RA 20,54%, DLinear 0,19% và persistence qmax 0%. Kết quả chi tiết ở [Mục 10](#10-kết-quả-thực-nghiệm-trên-kaggle). Phần này triển khai độc lập với B1/B2, không gọi mô hình hoặc huấn luyện lại.

Mã hiện tại mở rộng thành **giao thức 3**, dùng dung sai chính ±⌊hạn/3⌋ ngày cho từng hạn 1–7 và giữ bảng phụ ±1 ngày. Phần mở rộng đã qua kiểm thử cục bộ và đường chạy dữ liệu giả đủ bảy hạn. Các kết quả thực nghiệm ở Mục 10 và `Tables/` vẫn thuộc lần chạy giao thức 2; ở hạn một ngày, hai giao thức cùng dùng dung sai chính 0.

## 1. Đầu vào và cách chạy

**Tích hợp A4 đã chạy thành công trên Kaggle: 14 run, 129 kiểm thử đạt, A3 và hợp đồng A4 đều khớp; giao chung có 652.628 ngày–lưu vực trên 450 ID có ngày chấm.** [Mục 11](#11-xác-nhận-tích-hợp-a4-trên-kaggle) ghi độ phủ và kiểm nguồn của lần chạy này; Mục 10 giữ riêng kết quả điểm số của lần chấm 11 run.

Notebook tự đọc **14 run**: sáu A2, ba DLinear, hai persistence và ba LSTM A4. Gắn bốn nguồn Output đã Save Version: A1, A2, A3 và **`04_A4_LamaHCEExt_Compare` Version 2**. Dataset LamaH-CE gốc và ba Output huấn luyện A4 riêng không bắt buộc, vì A3/A4 Compare đã xuất đủ ngữ cảnh và dự báo. Chỉ chạy lại bộ đánh giá, không huấn luyện lại A1–A4.

| Nguồn | Tệp cần dùng |
|---|---|
| A1 | `A1_P10_qmax_level_thresholds.csv`, `A1_P10_flood_return_levels.csv` |
| A2 | `environment.json`; dưới `runs/<run>/`: `config.yml` và `test_results.p` của sáu run |
| A3 | `A3_Q_source_window.npz`, `A3_persistence_predictions.npz`, `A3_all_metrics_per_basin.csv` |
| DLinear trong A3 | `dlinear/seed_111`, `seed_222`, `seed_333`, mỗi thư mục có `test_predictions.npz` |
| A4 Compare Version 2 | Toàn thư mục `a4_compare/`, gồm `a4_summary.json`, `environment.json`, `manifest.csv` và ba NPZ `contracts/SeqLSTM_Q_A4_seed{111,222,333}_test_S0_lead1.npz`; các tệp còn lại trong manifest cũng cần được giữ |

Mặc định tìm nguồn dưới `/kaggle/input`, xuất vào `/kaggle/working/evaluation_output`. A2 được nhận diện bằng `environment.json` trong thư mục có đủ `runs/<sáu run>`, nên không nhầm với `environment.json` của A4. A4 được tìm qua `a4_summary.json`; thiếu hoặc trùng nguồn làm chương trình dừng, không tự bỏ seed hoặc chọn phiên bản. Script có thể chỉ rõ `--a1`, `--a2`, `--a3`, `--a4`; notebook chạy nguyên mã sau khi gắn đủ Input, không cần điền đường dẫn hay tên tài khoản.

Trên Kaggle, dùng CPU và Run All từ đầu sau khi nhập bản mã mới. Phần 8.1 nhận diện nguồn A2; Phần 8.4 đọc/kiểm ba NPZ A4; Phần 10.3 vẽ CDF có chú giải phân biệt các seed; Phần 11.1 điều phối chấm chung. Cần chạy lại các phần định nghĩa phía trước trong phiên sạch để tránh giữ hàm cũ.

Kiểm độc lập bằng lệnh `python Workspace/06_Evaluation/01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.py --self-test` từ thư mục repo. Môi trường cần NumPy, pandas, SciPy, xarray, PyYAML; phần vẽ dùng matplotlib. Kaggle A1/A2/A3 đã có các thư viện này. Bộ đánh giá không thay phiên bản thư viện trong phiên chạy.

Mã `.py` dùng các cell `# %%` và cell Markdown `# %% [markdown]`: tiêu đề phần → đề mục → code đúng chức năng. Repo giữ `.ipynb` chính thức Version 2 có output của lần chấm 11 run. Nguồn `.py` hiện bao gồm A4 và kết quả 14 run đã được kiểm ở Mục 11; tệp `.ipynb` tải từ phiên Save & Run All tương ứng với lần chấm này được bổ sung khi có, không thay bản chính thức bằng notebook trống output.

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

Ngưỡng lấy từ `half_calendar_primary` của A1 cho T = 1,5; 2; 5; 10 năm, giữ số năm fit theo từng lưu vực. Bộ đánh giá áp cùng ngưỡng và quy tắc cho obs/sim; xuất cả đoạn liên tiếp và cụm USWRC. [Giao thức đã chốt](../../../Document/01_Plan/03_Pipeline.md#62-theo-mức-lưu-lượng-và-theo-sự-kiện-yêu-cầu-gvhd).

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

Các trường run, seed, kỳ, hạn, đơn vị, config SHA-256, commit thư viện, chính sách clip và epoch được kiểm trước khi gọi `check_same_context`. Epoch phải trùng `saved_epoch` trong summary. Version nguồn huấn luyện được đối chiếu với bản bàn giao: seed 111 Version 1, seed 222/333 Version 2, A3 Version 1. Version 2 của notebook Compare là phiên bản yêu cầu theo [báo cáo A4](https://github.com/Ticasslo/TieuLuanChuyenNganh/blob/b98ad95db47a7ca363740d6696b6b040e5aeabf7/Workspace/03_Baseline/03_A4_LamaHCEExt_Retrain.md#p5); giá trị băm ghi danh tính tệp thực tế, không gọi đây là truy vấn xác nhận Version trực tiếp từ Kaggle.

Ba run có tên `SeqLSTM_Q_A4_seed111`, `SeqLSTM_Q_A4_seed222`, `SeqLSTM_Q_A4_seed333`. Các tệp này được nhận tự động, không cần truyền `--extra-contract` cho A4. Tham số đó vẫn dùng cho dự báo mở rộng khác; trùng tên run/kỳ/kịch bản/hạn làm chương trình dừng.

## 8. Các phép kiểm đã thực hiện cục bộ

Phép kiểm độc lập bao gồm ví dụ NSE/PNSE/KGE và tham chiếu Pearson từ SciPy; chuỗi hằng, phương sai nhỏ, zero-SSE, NaN/mask/vô cực; tính bất biến đổi đơn vị; ngưỡng bằng biên; USWRC với đáy bằng 0,75 và khoảng thiếu; ghép sự kiện không tái sử dụng; persistence trễ một ngày bị loại ở bảng chính hạn 1 nhưng được bắt ở bảng phụ; Holm bỏ NaN và dz với số biết trước; kịch bản cùng ngày chấm. Kiểm nhiều hạn bao phủ cả bảy hạn, biên lệch sớm/trễ trong và ngoài dung sai, kiểm duyệt quanh ngày thiếu/biên lịch, tách bảng chính/phụ khi cùng dung sai, cùng kiểm ngày nguồn persistence tại origin−1.

**129 kiểm thử độc lập của giao thức 3 đều đạt.** Đầu vào hạn, tổng số hạn và dung sai sai kiểu hoặc ngoài phạm vi được từ chối; bool không được xem là số nguyên. Môi trường kiểm cục bộ: Python 3.12.6, NumPy 1.26.4, pandas 2.3.3, SciPy 1.16.3 và xarray 2025.12.0. Kiểm hồi quy thêm 400 ca sự kiện hạn 1, gồm hai cách gộp, dung sai 0/1 và ngày thiếu, khớp chính xác mã giao thức 2. Trước lần sửa dung sai và Holm, đã đối chiếu thêm 120 ca chỉ số với hàm A3 giữ nguyên, 60 ca gộp cụm với hàm A1 giữ nguyên và 80 ca ghép sự kiện với lời giải duyệt hết các cách ghép; tất cả đều khớp.

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

Với đầu vào A2/DLinear/A4, phạm vi là test/S0/hạn một ngày. JSON kết luận có `A4_contracts_matched`, danh sách `A4_runs` và `A4_compare_version_expected`; chỉ đặt hoàn thành sau khi chấm và đối chiếu A3 thành công. Cần gửi hai bảng Phần 8.4, số run/giao ngày ở Phần 3–6, kết luận Phần 11 và các tệp `Evaluation_A4_input_check.csv`, `Evaluation_A4_contract_check.csv`, `Evaluation_coverage.csv`, `Evaluation_summary.csv`, `Evaluation_paired_tests.csv`, `Evaluation_summary.json` cùng hình CDF. Mọi cảnh báo/traceback được giữ để đối chiếu.

Các bảng và hình trong Mục 10 thuộc lần chấm 11 run; Mục 11 trình bày lần chấm 14 run từ các CSV thực nghiệm đã nhận. [Notebook phân tích thuộc tính 02](../02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md) đã chạy trên 14 run với 18 phép so và 1.674 hệ số; ba seed A4 giữ cùng hướng liên hệ ΔNSE với độ cao, tuyết và độ khô hạn. [Kết quả và kiểm chứng theo lưu vực](../02_Evaluation_LamaHCEExt_ByBasin/02_Evaluation_LamaHCEExt_ByBasin.md#12-kết-quả-phân-tích-14-run-có-a4).

Khi bổ sung dự báo của bước B/C/H, giữ cùng các tệp đầu ra để đối chiếu thêm theo kịch bản và hạn. Bảng chênh lệch kịch bản có đầy đủ tên cột ngay cả khi chưa có dòng dữ liệu ngoài S0.

Lần chạy Kaggle đã nhận có `protocol_version = 2`, `event_tolerance_days = 0` và `supplementary_event_tolerance_days = 1` trong JSON. Các bảng tổng hợp trong `Tables/` giữ đúng xuất xứ lần chạy này; ba CSV chi tiết của mỗi dung sai được giữ trong Output Kaggle. Tệp tổng hợp phụ không nằm trong 21 tệp được cung cấp, nên `Evaluation_events_pooled_tolerance1.csv` được tính lại từ bảng chi tiết đầy đủ `Evaluation_events_tolerance1.csv`, có ghi nguồn trong biên nhận kiểm tra.

Lần chạy bằng mã hiện tại xuất `protocol_version = 3`, tên chính sách `event_tolerance` và `event_tolerance_days_by_lead` cho các hạn thực sự được chấm. Chấm đúng một hạn giữ thêm `event_tolerance_days` là số nguyên; chấm nhiều hạn để trường scalar này là null và dùng bản đồ theo hạn. `matching_tolerance_days` trong từng dòng sự kiện và cặp ghép ghi dung sai đã áp dụng. Kết quả Kaggle hạn 1 đã nhận vẫn dùng được, không cần chạy lại chỉ để mở rộng mã cho hạn 2–7.

## 10. Kết quả thực nghiệm trên Kaggle

### 10.1. Phạm vi và tính nhất quán

Các bảng dưới dùng giao thức 2: cùng ngày bắt đầu ở bảng chính và ±1 ngày ở bảng phụ. Chỉ số liên tục, phân mức và kiểm định không đổi so với lần chạy trước; các tệp metrics, điểm CDF và PNG CDF có cùng SHA-256.

Nguồn thực nghiệm là Output A1 `01-a1-lamahceext-explore`, A2 `01-a2-lamahceext-reproduce` và A3 `02-a3-lamahceext-benchmarks`. Môi trường Kaggle: Python 3.13.15, NumPy 2.1.3, pandas 2.3.3, SciPy 1.16.3. [Kết luận chạy](Tables/Evaluation_summary.json), [nguồn đầu vào](Tables/Evaluation_input_manifest.csv).

| Kiểm tra | Kết quả |
|---|---|
| Kiểm thử trước khi chấm | 60/60 đạt |
| Đối chiếu A3 | 7/7 kiểm đạt; mỗi kiểm gồm 4.961 cặp run–ID: NSE, PNSE, KGE, r, α, β và n |
| Giao ngày chung | 652.628 ngày–ID; 450 ID có ngày chấm, giữ ID 55 với n = 0 |
| Chỉ số liên tục | 14.883 dòng, gồm 11 run × 451 ID × 3 scope |
| Ma trận bốn mức | 79.376 ô đếm; tổng mỗi ma trận khớp số ngày chấm |
| Sự kiện chính, tolerance = 0 | 39.688 dòng theo run/ID/T/cách gộp; 6.801 cặp ghép một–một |
| Sự kiện phụ, ±1 ngày | 39.688 dòng; 23.221 cặp ghép một–một, khớp lần chạy trước |
| Kiểm định | 135 phép so, gồm 45 cặp × 3 chỉ số; Holm tách theo từng chỉ số |
| Kiểm tra CSV cục bộ | 57/57 kiểm đạt: công thức, tổng hợp, ghép sự kiện, kiểm định và dữ liệu CDF |

NSE/PNSE có 450 ID hữu hạn trên giao chung cho mọi run. KGE/r của EncDec_noQ có 448 ID hữu hạn do đầu ra hằng tại 742 và 758. ID 55 có trạng thái không có cặp hợp lệ ở mọi run trên giao chung. [Bảng đối chiếu A3](Tables/Evaluation_A3_reference_check.csv), [kiểm tra CSV](Tables/Evaluation_output_audit.csv).

![Phân phối NSE, PNSE và KGE trên giao ngày chung](Figures/Evaluation_CDF_test_S0_lead1.png)

*Hình 1. Mỗi điểm là chỉ số của một lưu vực; giữ toàn bộ giá trị hữu hạn, kể cả phần đuôi âm. Các trục dùng symlog, tuyến tính trong [−1, 1]. NSE/PNSE của DLinear cao hơn persistence ở trung vị, còn KGE thấp hơn. Đường persistence PNSE nằm tại 0 theo định nghĩa.*

### 10.2. Kết quả liên tục và bốn mức

NSE trung vị SeqLSTM_Q là **0,705259**, persistence qmax **0,350031**; DLinear 111/222/333 lần lượt **0,382799/0,378069/0,379588**. Các giá trị khớp A3. KGE trung vị của ba DLinear khoảng 0,513–0,521, thấp hơn persistence 0,674939. [Bảng đầy đủ](Tables/Evaluation_summary.csv).

Trên giao chung có **194.703 ngày–ID mức thấp** và **11.759 ngày–ID mức rất cao**. Các tỷ lệ dưới đây cộng ô đếm trước khi chia, không phải trung vị chỉ số theo lưu vực.

| Run | Recall mức thấp (%) | Recall mức rất cao (%) | Precision mức rất cao (%) |
|---|---:|---:|---:|
| SeqLSTM_Q | 66,93 | 47,58 | 67,58 |
| SeqLSTM_noQ | 65,79 | 36,17 | 63,73 |
| EncDec_Q | 53,67 | 41,31 | 62,22 |
| EncDec_noQ | 54,95 | 38,50 | 51,30 |
| D_FC | 47,23 | 21,49 | 41,75 |
| D_RA | 65,06 | 47,27 | 63,78 |
| DLinear 111 | 61,83 | 22,18 | 53,33 |
| DLinear 222 | 67,61 | 22,30 | 53,72 |
| DLinear 333 | 69,52 | 21,80 | 54,86 |
| Persistence qmax | 88,07 | 40,11 | 40,28 |
| Persistence qmean, phụ | 96,27 | 18,40 | 61,53 |

SeqLSTM_Q bắt được 5.595/11.759 ngày rất cao, nhiều hơn persistence qmax 4.716 ngày; precision cũng cao hơn. DLinear đạt khoảng 22% recall mức rất cao, phù hợp với độ co biên độ và hụt dải cao đã thấy ở A3. Accuracy tổng thể của persistence qmax là 82,90%, cao hơn SeqLSTM_Q 77,62%; tỷ lệ đúng chung chịu ảnh hưởng lớn của các mức phổ biến, vì vậy phải đọc riêng recall ở hai đầu dải lưu lượng. [Bảng phân mức](Tables/Evaluation_levels_pooled.csv).

Sai lệch FHV trung vị của SeqLSTM_Q là −13,01%; DLinear 111/222/333 là −31,35/−30,78/−31,91%, thể hiện hụt tổng của dải 2% lưu lượng cao nhất theo định nghĩa FHV. FLV của DLinear có trị tuyệt đối trung vị lớn, khoảng 597–754%; chỉ số này đo hình dạng đuôi log, không phải phần trăm thiếu thể tích nước. Trên giao chấm, ba seed có 9.197/12.134/13.218 giá trị sim dưới sàn log, chủ yếu do chặn các dự báo âm về 0. Khi đọc FLV cần giữ cùng sàn 10⁻⁶ mm/ngày và số ngày chịu phép log này.

### 10.3. Sự kiện lũ chu kỳ hai năm

#### 10.3.1. Bảng chính: cùng ngày bắt đầu

Theo quy tắc USWRC, có **1.066 cụm quan trắc đủ điều kiện trên 366 lưu vực**. Bảng cộng số trận trước khi tính tỷ lệ, gồm cả nhóm ngưỡng fit dưới 10 năm; CSV có bảng tách riêng nhóm đó.

| Run | Bắt được | Bỏ sót | Báo nhầm | POD (%) | FAR (%) |
|---|---:|---:|---:|---:|---:|
| SeqLSTM_Q | 208 | 858 | 431 | 19,51 | 67,45 |
| SeqLSTM_noQ | 188 | 878 | 304 | 17,64 | 61,79 |
| EncDec_Q | 126 | 940 | 289 | 11,82 | 69,64 |
| EncDec_noQ | 174 | 892 | 445 | 16,32 | 71,89 |
| D_FC | 63 | 1.003 | 286 | 5,91 | 81,95 |
| D_RA | 219 | 847 | 377 | 20,54 | 63,25 |
| DLinear 111 | 2 | 1.064 | 327 | 0,19 | 99,39 |
| DLinear 222 | 2 | 1.064 | 340 | 0,19 | 99,42 |
| DLinear 333 | 2 | 1.064 | 322 | 0,19 | 99,38 |
| Persistence qmax | 0 | 1.066 | 1.066 | 0,00 | 100,00 |
| Persistence qmean, phụ | 0 | 1.066 | 201 | 0,00 | 100,00 |

Nguồn: [bảng chính](Tables/Evaluation_events_pooled.csv). D_RA dùng tái phân tích ở ngày đích nên là mốc tham chiếu về nguồn thông tin. SeqLSTM_Q bắt được nhiều trận hơn các cấu hình DLinear, nhưng vẫn bỏ sót 858/1.066 trận. Ba seed DLinear chỉ bắt đúng ngày khởi phát ở hai trận dù NSE trung vị nhỉnh hơn persistence; cải thiện sai số cả chuỗi không bảo đảm cải thiện thời điểm khởi phát lũ. Cùng ngày bắt đầu vẫn có thể khác ngày đỉnh, nên sai số thời điểm đỉnh được báo riêng.

#### 10.3.2. Bảng phụ: ±1 ngày

Giữ cùng ngưỡng và quy tắc gộp, cho phép onset lệch tối đa một ngày. Ở T = 2 năm, mẫu số cũng là 1.066 cụm trên 366 lưu vực; tại ngưỡng khác cần đọc số kiểm duyệt của từng bảng.

| Run | Bắt được | Bỏ sót | Báo nhầm | POD (%) | FAR (%) |
|---|---:|---:|---:|---:|---:|
| SeqLSTM_Q | 348 | 718 | 291 | 32,65 | 45,54 |
| SeqLSTM_noQ | 256 | 810 | 236 | 24,02 | 47,97 |
| EncDec_Q | 218 | 848 | 197 | 20,45 | 47,47 |
| EncDec_noQ | 226 | 840 | 393 | 21,20 | 63,49 |
| D_FC | 94 | 972 | 255 | 8,82 | 73,07 |
| D_RA | 325 | 741 | 271 | 30,49 | 45,47 |
| DLinear 111 | 222 | 844 | 107 | 20,83 | 32,52 |
| DLinear 222 | 237 | 829 | 105 | 22,23 | 30,70 |
| DLinear 333 | 222 | 844 | 102 | 20,83 | 31,48 |
| Persistence qmax | 1.066 | 0 | 0 | 100,00 | 0,00 |
| Persistence qmean, phụ | 117 | 949 | 84 | 10,98 | 41,79 |

**POD = 1 của persistence phản ánh dung sai ghép ±1 ngày trong bảng phụ.** Chuỗi chép qmax hôm qua dịch sự kiện sang hôm sau; độ lệch onset một ngày vẫn được tính là bắt được ở bảng này. Sai số tuyệt đối thời điểm đỉnh trung bình là đúng **1 ngày**. Bảng chính cùng ngày bắt đầu loại các cặp trễ như vậy. Bảng phân mức đúng ngày cho recall mức rất cao của persistence là 40,11%.

Ở bảng phụ, persistence cũng đạt POD = 1 tại T = 5 và 10 năm; T = 1,5 năm đạt 1.849/1.850 ≈ 99,95%. Ở bảng chính, persistence không có cặp đúng ngày tại cả bốn ngưỡng và cả hai cách gộp. Các mẫu số và nhóm fit ngắn được giữ trong [bảng phụ](Tables/Evaluation_events_pooled_tolerance1.csv).

### 10.4. So sánh ghép cặp

Trên NSE, DLinear 111/222/333 hơn persistence ở **318/450, 309/450, 317/450** ID; median chênh lệch tương ứng **0,035019/0,031839/0,031465**. p sau Holm lần lượt khoảng 5,79×10⁻²⁹; 2,67×10⁻²⁶; 2,06×10⁻²⁷. Cỡ hiệu ứng dz lần lượt **0,0506/0,4480/0,3933**; khác biệt giữa seed cần được đọc cùng các lưu vực ở phần đuôi âm.

Trên KGE, mỗi seed DLinear thấp hơn persistence ở **449/450** ID. Trên PNSE, median của ba seed dương nhưng mean âm; một số lưu vực có persistence rất mạnh làm tỷ số sai số lớn. Vì vậy, p nhỏ hoặc đa số lưu vực tốt lên không thay thế việc kiểm mức thiệt hại ở các lưu vực dự báo kém. [Toàn bộ 135 phép so](Tables/Evaluation_paired_tests.csv).

### 10.5. Phạm vi xác nhận và bằng chứng lưu

Lần chạy xác nhận bộ đánh giá trên dự báo thực nghiệm **test/S0/hạn một ngày**, gồm chỉ số liên tục, phân mức, sự kiện và kiểm định. Các kịch bản S1–S4, nhiều hạn và tập validation sẽ được chấm khi có dự báo tương ứng từ luồng mô hình; hai cờ trong JSON hiện giữ false đúng phạm vi này.

Đã kiểm 21 tệp kết quả được cung cấp và tính lại một bảng tổng hợp phụ từ CSV chi tiết; các bảng tổng hợp và hình được giữ trong `Tables`/`Figures`. [Manifest tệp nhận](Tables/Evaluation_received_manifest.csv) ghi SHA-256; [biên nhận kiểm tra](Tables/Evaluation_output_verification.json) ghi 57 phép kiểm cục bộ đạt và nguồn của bảng phụ được tính lại. Các phép kiểm đối chiếu số đếm, ngày ghép, sai số đỉnh, Wilcoxon, dz, Holm và CDF từ CSV; chuỗi dự báo thô và 20 đầu vào Kaggle tiếp tục nằm ở Output/Input Kaggle.

## 11. Xác nhận tích hợp A4 trên Kaggle

Lần chấm test/S0/hạn một ngày gồm đủ **14 run và 451 ID**. Có **129 kiểm thử đạt**, `A3_reference_matched=true` và `A4_contracts_matched=true`. Manifest A4 có 11 tệp, cùng hai nguồn ngữ cảnh A3, đều đạt kiểm hash. Ba hợp đồng LSTM có cùng lịch, ID, diện tích, obs canonical và persistence với A3. Nguồn: [biên nhận nhật ký thực thi](Tables/Evaluation_A4_execution_receipt.json).

| Run A4 | Epoch chọn | Ngày–ID có dự báo | Ngày–ID có obs/sim hữu hạn | Dự báo âm trước clip |
|---|---:|---:|---:|---:|
| Seed 111 | 11 | 652.644 | 652.636 | 1.660 |
| Seed 222 | 15 | 652.644 | 652.636 | 2.261 |
| Seed 333 | 14 | 652.644 | 652.636 | 1.978 |

Mỗi hợp đồng giữ 451 ID trên 1.461 ngày đích. Giao chung của cả 14 run có **652.628 ngày–lưu vực trên 450 ID có ngày chấm**, bằng số lượng của lần chấm 11 run. Thêm A4 không làm giảm số ngày trong giao này. Số 652.644 mô tả các dự báo hữu hạn trước khi yêu cầu có obs; 652.636 là số cặp obs/sim của riêng mỗi A4, còn 652.628 là giao để so toàn bộ mô hình.

Giao thức 3 giữ dung sai lũ chính **0 ngày ở hạn một ngày**, bảng phụ **±1 ngày**. Lần chấm này chỉ gồm S0 và hạn 1; chưa có kết quả vận hành S1–S4 hoặc nhiều hạn. Môi trường là Python 3.13.15, NumPy 2.1.3, pandas 2.3.3 và SciPy 1.16.3.

Các bảng tổng hợp thực nghiệm của lần chấm 14 run được lưu riêng tại [Tables/A4_Integrated](Tables/A4_Integrated/Evaluation_received_manifest.csv). Bốn CSV có 378 hàng tổng hợp chỉ số, 234 hàng kiểm định và 336 hàng sự kiện cho mỗi dung sai. **49 kiểm cục bộ đạt**: điểm số và bảng sự kiện của 11 run cũ khớp hoàn toàn; thống kê ghép cặp/p gốc của 135 phép so cũ giữ nguyên; tỷ lệ sự kiện khớp số đếm, các nhóm fit bảo toàn tổng và Holm được tính lại trên 78 cặp mỗi chỉ số. [Bảng kiểm](Tables/A4_Integrated/Evaluation_output_audit.csv), [phạm vi kiểm chứng](Tables/A4_Integrated/Evaluation_output_verification.json). Chưa có CSV chỉ số từng ID và NPZ tại máy cho lần này, nên kiểm tổng hợp không được gọi là tính lại Wilcoxon, dz hoặc chạy lại dự báo.

### 11.1. Chỉ số liên tục trên cùng giao ngày

| Run | NSE trung vị | PNSE trung vị | KGE trung vị |
|---|---:|---:|---:|
| SeqLSTM_Q, trọng số tác giả | 0,705259 | 0,495584 | 0,767074 |
| LSTM A4 seed 111 | 0,695242 | 0,495352 | 0,758090 |
| LSTM A4 seed 222 | 0,703778 | 0,505573 | 0,790977 |
| LSTM A4 seed 333 | 0,705890 | 0,501788 | 0,785984 |
| Persistence qmax | 0,350031 | 0 | 0,674939 |

Mỗi chỉ số trong bảng có 450 ID hữu hạn. Trung bình ± độ lệch chuẩn mẫu của ba NSE trung vị A4 là **0,701636 ± 0,005638**; khoảng [0,695242; 0,705890] bao quanh mốc tác giả 0,705259. Đây là biến thiên của ba lần huấn luyện, không phải khoảng tin cậy hay chỉ số của một ensemble. Ba seed đều giữ riêng; không chọn seed theo test. [Bảng chỉ số đầy đủ](Tables/A4_Integrated/Evaluation_summary.csv).

NSE của A4 tương đối sát mốc tác giả nhưng KGE thay đổi theo seed: 111 thấp hơn mốc về trung vị, 222 và 333 cao hơn. FHV trung vị của ba seed là **−15,70%; −11,71%; −11,56%**, cho thấy tổng dải lưu lượng cao theo định nghĩa FHV vẫn bị dự báo hụt. Với PNSE, cả ba median dương nhưng mean lần lượt **−1,189; −0,508; −0,355**; cần giữ phần đuôi các lưu vực mà persistence mạnh khi đọc kết quả.

![CDF trên giao chung của 14 run](Figures/Evaluation_CDF_test_S0_lead1_with_A4.png)

Hình 2. Ba đường A4 màu xanh dùng kiểu nét khác nhau cho từng seed; đường SeqLSTM_Q liền là trọng số tác giả. Số n trong chú giải là số ID có chỉ số hữu hạn; EncDec_noQ có 448 KGE, các run còn lại có 450. Giữ toàn bộ điểm âm.

### 11.2. Khởi phát lũ hai năm: đúng ngày và bảng phụ

Ngưỡng hai năm, cụm USWRC và nhóm tất cả ID có **1.066 trận thật đủ điều kiện trên 366 lưu vực**. Bảng chính yêu cầu cùng ngày bắt đầu.

| Run | Bắt được | Bỏ sót | Báo nhầm | POD (%) | FAR (%) |
|---|---:|---:|---:|---:|---:|
| SeqLSTM_Q, trọng số tác giả | 208 | 858 | 431 | 19,51 | 67,45 |
| A4 seed 111 | 180 | 886 | 372 | 16,89 | 67,39 |
| A4 seed 222 | 216 | 850 | 446 | 20,26 | 67,37 |
| A4 seed 333 | 191 | 875 | 446 | 17,92 | 70,02 |
| Persistence qmax | 0 | 1.066 | 1.066 | 0,00 | 100,00 |

Nguồn: [sự kiện chính](Tables/A4_Integrated/Evaluation_events_pooled.csv). Seed 333 có NSE trung vị cao nhất trong ba seed, còn seed 222 bắt đúng ngày nhiều trận nhất. Vì vậy, xếp theo NSE không tự xác định thứ tự về bắt lũ. Cả ba A4 vẫn bỏ sót 850–886/1.066 trận và có FAR khoảng 67–70%; tái lập NSE không có nghĩa đã giải quyết tốt cảnh báo khởi phát lũ.

Ở bảng phụ ±1 ngày, A4 111/222/333 bắt được **306/363/333 trận**, POD **28,71/34,05/31,24%** và FAR **44,57/45,17/47,72%**. Persistence qmax đạt POD 100% ở bảng phụ do chấp nhận trễ một ngày, trong khi POD ở bảng chính bằng 0. [Bảng phụ](Tables/A4_Integrated/Evaluation_events_pooled_tolerance1.csv).

### 11.3. Kiểm định ghép cặp và ảnh hưởng của họ Holm

| A4 trừ SeqLSTM_Q tác giả, trên NSE | ID A4 tốt hơn | Median hiệu | p sau Holm |
|---|---:|---:|---:|
| Seed 111 | 209/450 | −0,002327 | 0,481862 |
| Seed 222 | 249/450 | 0,003656 | 0,288877 |
| Seed 333 | 254/450 | 0,004786 | 0,530678 |

Hướng hiệu trong bảng được đổi từ CSV để thống nhất **A4 trừ tác giả**; p hai phía giữ nguyên. Không phát hiện khác biệt NSE ở mức 0,05 trong họ 78 cặp này; kết quả đó không phải phép thử tương đương. So persistence, ba A4 tốt hơn về NSE ở **418/424/428 trên 450 ID**, với median hiệu **0,327901/0,339101/0,329561**. [234 phép so đầy đủ](Tables/A4_Integrated/Evaluation_paired_tests.csv).

NSE seed 111 so 222 có p Holm **0,025099**, seed 111 so 333 có p Holm **0,051454**, trong khi median hiệu chỉ khoảng −0,00384 và −0,00444. Họ kiểm ở đây có 78 cặp mỗi chỉ số, khác họ ba cặp seed trong báo cáo A4; vì vậy không dùng hoán đổi p đã điều chỉnh của hai bảng. Quyết định về một cải tiến cần đọc cả độ lớn hiệu, biến thiên nhiều seed và phạm vi dữ liệu, không chỉ p < 0,05.
