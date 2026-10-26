# B2 — Bộ nạp và mô hình Extended LamaH-CE

Notebook kiểm source, bộ nạp và mô hình B2 trên nền `f229e02`, trước khi huấn luyện ba seed. Output Kaggle đã nhận tới hết Phần 9.2: **47/47 kiểm đạt**, `B2_LOADER_INDEPENDENT_OK=True`, `B2_READY_FOR_TRAINING=False`. Phần kiểm độc lập đã đạt theo các bảng/log được cung cấp; đang chờ bàn giao B1b và commit thư viện được duyệt trước huấn luyện chính thức.

## Mã B2 theo phân công hiện hành

Người 1 thực hiện trọn B2 và B3 theo `ASSIGNMENT.md`, gồm bộ nạp, mô hình, huấn luyện, so sánh và đánh giá. Thư viện B2 được chia sẻ trên nhánh [`codex/b2-source-masked-mean`](https://github.com/Ticasslo/neuralhydrology/tree/codex/b2-source-masked-mean), commit [`cac96c54379853af4f3f430fcd5ec0df55e5fb30`](https://github.com/Ticasslo/neuralhydrology/commit/cac96c54379853af4f3f430fcd5ec0df55e5fb30), từ nền `f229e02bf98cbd979835a4b6cd503d0e3adb8bc3`; chưa gộp `research`.

Các thay đổi ở `NeuralHydrology/` gồm flatten nhóm biến LamaH; giữ mẫu thiếu đầu vào khi bật masked mean nhưng vẫn kiểm nhãn, lịch sử và thuộc tính; dùng chung quy tắc counter/che forecast cho mẫu thường và batch vector hóa. `configure_lamah_b2` trong `neuralhydrology/datautils/source_groups.py` tạo cấu hình B2 từ B1b, giữ kích thước nhúng, hidden size, head, loss và chuẩn hóa của B1b. Nhóm hindcast là ERA5-Land 21, E-OBS 7, MSWEP 1, GLEAM 2, Q 2; forecast gồm ECMWF HRES 5. Mô hình dùng `SequentialForecastLSTM` với masked mean của `InputLayer`. Cách (b) bước H dùng `forecast_input_mode: first_day`, counter bật và zero overlap; các bản HRES phát hành sau ngày đầu bị che trên bản sao dữ liệu.

Kiểm cục bộ mã thư viện mới: 13 ca B2 và 13 ca hồi quy đạt trên CPU; một ca CUDA chưa chạy vì máy không có CUDA. Mã kiểm thử đơn vị nằm trong `NeuralHydrology/test/test_b2_source_inputs.py`, theo yêu cầu kiểm thử thư viện. Các kiểm thử này không xác nhận dữ liệu thật, tích hợp B1b hay huấn luyện chính thức.

Notebook hiện có 31 code cell, mã `.py` và `.ipynb` đã đồng bộ tới hết Phần 9.2. Bản ứng viên được tạo từ patch B2 đóng băng, có kiểm nguồn, giữ nguyên chuẩn hóa và nạp trên bản sao package. Người dùng tự chạy Kaggle; notebook không chạy epoch hay optimizer. Điểm dừng là bàn giao B1b và commit thư viện đã kiểm trước huấn luyện chính thức.

Phiên tương tác đang có đủ biến tới hết 4.3 có thể tiếp tục từ **4.4**. Khi dùng Save & Run All, chạy từ đầu trong phiên mới để tránh module NeuralHydrology cũ còn nạp. Input giữ như các Phần 1–4 đã kiểm: dataset `lamah-ce-ext`, Output A1 Version 3; Output A2 dùng đối chiếu bản sao config nếu có. CPU đủ cho notebook kiểm, GPU là tùy chọn. Output nằm trong `/kaggle/working/b2_output`.

### Hợp đồng tích hợp đang áp dụng

- B1b cung cấp hai cột forcing `qmean_mmd`, `qmax_mmd`, đổi từ `D_gauges` bằng cùng phép đổi đơn vị của nhãn. B2 đọc hai cột đó và chỉ dùng tới t−1; không đổi đơn vị lần hai hoặc dùng alias thử nghiệm `qmax_past` trong cấu hình chính thức.
- Giữ cách fit scaler và std nhãn của `f229e02`/B1b, kể cả quy ước ngữ cảnh của mã nền. Không đưa `b2_normalize` và `b2_train_stds` của prototype cũ vào ablation B2.
- Nhúng theo nguồn và masked mean dùng `InputLayer` cùng `SequentialForecastLSTM`; không đổi lõi sang mã bước C. Bộ nạp thường và `data_on_gpu` cần cùng nhóm biến, mặt nạ và counter.
- Kiểm thử thư viện trên nhánh riêng trước khi gộp `research`; notebook huấn luyện phải pin commit cuối đã kiểm. Huấn luyện ba seed B2 chỉ bắt đầu sau khi nhận và đối chiếu bàn giao B1b.

Các mục trên đã đối chiếu với Phần 3–4 của notebook benchmark bộ nạp, `LamaH._load_basin_data`, chuẩn hóa/cắt warmup của `BaseDataset`, `InputLayer._masked_mean_embedding` và `SequentialForecastLSTM` tại mã nền tương ứng.

## Cách chạy và dữ liệu cần gắn

Gắn dataset lamah-ce-ext và output notebook A1 đã lưu. Dataset chứa raw và các config tác giả; nếu không tìm được config, có thể gắn thêm A2 hoặc đặt đường dẫn B2_CONFIG. A3 không bắt buộc. CPU đủ cho kiểm thử; cần Internet khi cài thư viện. Chạy tuần tự các đề mục; output được ghi trực tiếp trong b2_output, không đóng ZIP.

Tập mô hình giữ 451 lưu vực, train 2003–2009, validation 2010–2013, test 2014–2017. Đếm raw trên toàn bộ 451 ID; chỉ khởi tạo dataset/mô hình ở tập hồ sơ nhỏ để giảm RAM. Không suy đã khởi tạo và huấn luyện mô hình trên toàn bộ 451 ID từ phép đếm raw.

## Phần 1. Môi trường, mã nền và Input

Xác nhận đúng mã thư viện và tìm nguồn dữ liệu trước khi mở các bảng lớn. Dataset lamah-ce-ext và output A1 là đầu vào chính; A2 có thể bổ sung config, A3 không bắt buộc.

### 1.1. Thiết lập chạy và các đường dẫn

Mặc định chạy kiểm trên CPU, không cần GPU. Có thể đặt đường dẫn qua biến môi trường; giữ các mặc định khi Input đã gắn đúng. B1_COMMIT/config chỉ là thông tin nhận bàn giao, không tự thay nền đã pin.

### 1.2. Chuẩn bị thư viện và xác nhận mã nền

Thư viện nền được nhận diện bằng commit đầy đủ trong metadata PEP 610 và SHA-256 văn bản chuẩn LF của sáu tệp: LamaH, BaseDataset, InputLayer, đọc scaler, Config và GpuBatchLoader. Nếu bản đã cài khác commit, chỉ cài lại NeuralHydrology bằng `--no-deps`; các gói đã có phải giữ nguyên phiên bản. Nếu mã nguồn khác hash hoặc phiên còn module NeuralHydrology đã nạp, cell dừng trước khi áp patch. Cài gói cần Internet; chạy trong phiên sạch.

Kiểm cục bộ bằng chính code Phần 1.1–1.2 trong `.ipynb`: nguồn `f229e02` khớp cả sáu hash; tệp nguồn bị sửa bị từ chối; module nạp trước bị từ chối. Hai nhánh đã cài đúng commit và đã cài sai commit được kiểm bằng giả lập metadata/lệnh pip, không cài gói qua mạng trong phép thử. Cả 31 code cell của `.ipynb` vẫn khớp `.py`; đây chỉ là kiểm đồng bộ, không phải đã chạy toàn bộ 31 cell mới trên Kaggle.

### 1.3. Tìm raw, danh sách ID, config và bảng A1

Nguồn raw được kiểm theo cấu trúc timeseries và attributes; khi có nhiều thư mục dữ liệu, ưu tiên nguồn gốc `LamaH_extended` duy nhất. Danh sách `basins_filtered.txt` phải có đúng 451 ID dương, không trùng sau chuẩn hóa và cùng tập ID giữa các bản sao. Giữ thứ tự của nguồn `Experiments` gốc; kiểm đủ 451 tệp forcing và 451 tệp gauge, chưa đọc toàn bộ chuỗi ngày.

Cell lập bảng config theo nguồn, loại mô hình và các tên cột Q thực tế. Chưa chọn cấu hình khoa học ở bước này, chưa xác nhận B1b hoàn tất và chưa fit scaler. Hai bảng A1 bắt buộc là `A1_P7_sample_contract_groups.csv` và `A1_P7_sample_counts.csv`; kiểm header và từ chối các bản cùng tên nhưng khác hash. Version A1 mong đợi là 3; chỉ tên và nội dung các CSV không tự xác nhận được số Version Kaggle.

Output gồm bốn bảng: Input tìm được, danh sách lưu vực, cấu hình tìm được và hai bảng A1 bắt buộc. Các receipt được ghi vào `B2_input_sources.json`, `B2_input_configs.csv`, `B2_basin_list_inventory.csv`, `B2_A1_inventory.csv`. Receipt chỉ đánh dấu hoàn tất kiểm Input; các cờ tích hợp B1b và huấn luyện vẫn false.

Đã kiểm cục bộ bằng chính cell `.ipynb` trên cây Input giả đủ 451 ID: **12 ca I/O đạt**, gồm giữ thứ tự gốc, chấp nhận bản A1 giống byte, từ chối bản A1 khác hash, ID trùng sau chuẩn hóa, tập ID khác nhau, thiếu gauge/A1, header A1 sai và config không phải ánh xạ. Đã kiểm bảng Styler render HTML, schema notebook và hash nguồn; 31 code cell đồng bộ với `.py`. Đây chưa phải kết quả dữ liệu thật của Phần 1.3.

Output Kaggle Phần 1.1–1.2 đã được cung cấp: Python 3.13.15, torch 2.11.0+cpu, NumPy 2.1.3, pandas 2.3.3; cả sáu tệp nguồn khớp SHA-256 `f229e02`. Output Kaggle Phần 1.3 cũng đã nhận: nguồn `lamah_ce_ext/LamaH_extended`, đủ 451 ID/902 tệp, 36 config và 21 CSV A1. Danh sách lưu vực có SHA-256 `45c3d648b3868d5bb27ef50b3cff2c23658bd09ace001141380d82ce1ded2feb`; hai bảng A1 bắt buộc được nhận diện đúng. Các config có Q trong inventory đều dùng `qmean`, chưa có cấu hình B1b dùng cả `qmean_mmd` và `qmax_mmd`.

## Phần 2. Hợp đồng nguồn, biến, pha và đơn vị

Hợp đồng B2 dùng hai cột Q của B1b là `qmean_mmd` và `qmax_mmd`, chỉ thuộc hindcast; nhãn `qmax` giữ riêng để không xóa Q quá khứ trong warmup. Cấu hình cụ thể được chọn/đối chiếu ở Phần 2.2; Phần 2.1 chỉ xác lập quy tắc.

### 2.1. Hàm nhóm nguồn và kiểm hợp đồng cấu hình

Một nhóm nguồn thiếu khi có bất kỳ biến thành phần NaN tại bước đó, theo `InputLayer._masked_mean_embedding`. Các hàm nhóm giữ thứ tự nguồn ERA5L/EOBS/MSWEP/GLEAM/Q/ECMWF và thứ tự biến trong từng nhóm. Chỉ nhận Q đầu vào theo tên B1b; không suy đơn vị từ tên `qmean` cũ hoặc dùng nhãn `qmax` như đầu vào.

`b2_contract` kiểm hợp các biến động bằng đúng hai pha, không có biến trùng/thừa/thiếu, Q chỉ thuộc hindcast, forecast chỉ ECMWF HRES, zero overlap, hình học cửa sổ hợp lệ và một tần suất ngày. Cách (b) dùng `forecast_input_mode: first_day`, masked mean và counter. Với dữ liệu chỉ có HRES hạn một ngày, hợp đồng kiểm nhiều hạn từ chối chế độ dùng các bản HRES phát hành ở các ngày tương lai. Các khóa prototype `b2_data_contract`, `b2_hybrid_forecast` bị từ chối.

Đã chạy chính cell `.ipynb` qua **52 kiểm tra cục bộ**, gồm nhóm nguồn đối chiếu hàm thư viện trên nhiều thứ tự biến, cấu hình một/bảy hạn, đọc `Config` thật không thay dữ liệu cấu hình và từ chối các vi phạm pha/nhãn/biến/counter. Hai bảng hiển thị là hợp đồng kỳ vọng, không phải số biến thực tế đã xác nhận ở config được chọn. File `B2_expected_contract.json` giữ `selected_config_verified=false` và `B1b_integration_verified=false`.

### 2.2. Chọn config nền và tạo hợp đồng kiểm B2

Nguồn chuẩn là `Sequential_Forecast_LSTM_RA_with_q_Simple_Embedding` trong Experiments đi cùng danh sách lưu vực. Kiểm mốc 32/5/33 biến và các thiết lập đã đối chiếu ở B0: 128 nút ẩn, batch 256, loss NSE, Adam, 30 epoch, hai embedding FC [16]/tanh/dropout 0, cosine T_max 30/eta_min 10⁻⁵. Mốc tác giả có dừng sớm 5 epoch; template giao thức B đặt patience 30 và `data_on_gpu=true`.

So toàn bộ thiết lập khoa học với các bản `SeqLSTM_Q` từ A2, chuẩn hóa biểu diễn ngày trước khi so. Các đường dẫn thực thi, thiết bị, số tiến trình, tên run và metadata phần mềm được bỏ khỏi phép so theo đúng các thay đổi trong notebook A2. Các khóa đường dẫn còn lại được diễn giải theo `Config._parse_config` tại `f229e02`: chuỗi `"None"` tương đương `None`, danh sách giữ từng đường dẫn. Cấu hình nhúng Complex không được dùng thay thế; bất kỳ khác biệt khoa học giữa các bản simple đều làm cell dừng.

Giữ `BASE_CONFIG`/`BASE_HINDCAST` nguyên mốc tác giả cho đối chiếu A1. Tạo `B1B_EXPECTED_CONFIG` riêng bằng cách thay `qmean` đúng vị trí thành `qmean_mmd`, `qmax_mmd`; sau đó tạo `B2_CONFIG_TEMPLATE` bằng nhóm nguồn và masked mean. Template giữ nguyên nhúng 16 chiều, lõi LSTM, loss, tối ưu hóa, scheduler và chuẩn hóa. Cấu hình B2 dự kiến có 33 biến hindcast, 5 forecast, 33 thuộc tính và 38 biến động.

Các JSON xuất ra là template khoa học, chưa chuẩn bị đường dẫn thực thi. Không coi config suy ra là bàn giao B1b hoặc các cột Q đã kiểm; các cờ `B1b_handoff_verified`, `Q_columns_verified_against_raw`, `runtime_paths_prepared`, `official_training_started` giữ false. Bảng biến ghi tên cột Q mm/ngày và cột nguồn `D_gauges` tương ứng; không đổi giá trị Q ở cell này.

Đã chạy chính cell `.ipynb` với các cây config giả: **45 kiểm tra đạt**, gồm các bản sao A2 đổi đường dẫn/thiết bị, ngày biểu diễn khác nhưng tương đương, chuỗi `"None"`/YAML null ở các khóa đường dẫn, từ chối bản sao đổi tham số/kỳ/seed, nhúng Complex, thiếu config gốc và cấu trúc biến sai. Đã so cách diễn giải đường dẫn với `Config._parse_config` trên giá trị rỗng, chuỗi, danh sách và đối tượng `Path`; tệp đặc trưng hoặc kỳ riêng thực sự khai báo vẫn bị từ chối. Chuỗi `"None"` ở khóa không phải đường dẫn không được tự coi là rỗng. Template sau khi nạp bằng `Config` khớp hàm `configure_lamah_b2` trong nhánh thư viện; config đầu vào không bị thay. Các bảng Styler render và 31 code cell vẫn đồng bộ. Đây chưa phải kiểm config thật của Phần 2.2 trên Kaggle.

Output 2.1 đã nhận đúng sáu nhóm kỳ vọng và quy tắc Q/forecast/chuẩn hóa. Lần chạy đầu của 2.2 dừng ở `additional_feature_files`: điều kiện cũ kiểm trực tiếp giá trị YAML, nhận nhầm chuỗi `"None"` là có tệp. Đã sửa điều kiện và chữ ký so bản sao theo cách đọc của thư viện, giữ nguyên config gốc.

Output chạy lại 2.2 người dùng cung cấp đã đạt: **24/24 kiểm mốc tác giả đúng**, cả config tác giả và hai bản sao A2 có `scientific_match=True`; nguồn được chọn là config tác giả Simple. SHA-256 nguồn tác giả là `76f58f7cbb57b3c857aa59882e2bdf3630c03c42cbcc244a479227e25680d128`; hai bản sao A2 cùng hash `415e417818bd6c97d51d9787793c4c919bf960e35c3abd33a52df8eac2e7b5af`. Sáu nhóm có số biến 21/7/1/2/2 và 5; bảng hợp đồng có 38 biến động, Q chỉ hindcast tới t-1. Kết luận này xác nhận template suy ra từ nguồn đã chọn; chưa xác nhận hai cột Q trong dữ liệu thật hoặc bàn giao B1b.

### 2.3. Diện tích, thuộc tính và đường dẫn của 451 ID

Đọc `Catchment_attributes.csv` mức A và `Gauge_attributes.csv`, kiểm header, ID nguyên dương, ID/cột trùng và đủ tập ID trong từng bảng trước khi ghép. Giữ nguyên thứ tự 451 ID của nguồn; yêu cầu `area_gov` (km²) dương/hữu hạn và 33 trường theo config là số hữu hạn, không có mã -999. Cột ngoài config có thể chứa chữ hoặc thiếu; không dùng chúng để loại tập lưu vực đã chọn. Đối chiếu chính xác giá trị diện tích/thuộc tính và thứ tự ID với `load_lamah_attributes` của mã nền. Thứ tự cột bảng kiểm theo config; không thay quy tắc sắp thuộc tính khi bộ nạp tạo tensor.

Lập chỉ mục 451 tệp forcing và 451 tệp gauge, từ chối ID trùng hoặc tên tệp không tương thích `ID_<ID>.csv` của thư viện. Lưu năm tệp: `B2_attribute_sources.csv`, `B2_static_field_check.csv`, `B2_basin_attributes.csv`, `B2_data_file_index.csv`, `B2_attribute_contract.json`. Chỉ hai bảng thuộc tính được băm SHA-256 tại cell này; chỉ mục chuỗi ngày ghi đường dẫn và xác nhận sự tồn tại, chưa xác nhận nội dung hay ngày của từng tệp.

Công thức tham chiếu cho Q là `Q / (area_gov * 1e6) * 1000 * 86400`, cùng thứ tự phép tính của `_normalize_discharge` trong `f229e02`. Cell 2.3 chưa đọc chuỗi Q, đổi đơn vị, điền dữ liệu hoặc fit scaler; các cờ liên quan dữ liệu ngày, bàn giao B1b và huấn luyện giữ false.

Đã chạy chính cell `.ipynb` qua **36 kiểm tra cục bộ** trên cây raw giả đủ 451 ID, có ID ngoài tập và cột ngoài config chứa chữ/thiếu. Đã kiểm giữ thứ tự, khớp bộ đọc LamaH thật, năm tệp output và ba bảng HTML; các trường hợp ID/header trùng, ID sai, thiếu trường/tệp/ID, diện tích không hợp lệ, thuộc tính thiếu/vô hạn/-999 và khác giá trị/thứ tự so với thư viện đều bị từ chối. Các số liệu giả không dùng làm kết quả LamaH thật.

Output Kaggle 2.3 đã nhận: **451 ID × 33 thuộc tính**, mỗi trường đủ 451 giá trị hữu hạn, không có mã -999; `library_values_matched=True`. `area_gov` có min/trung vị/max **3,9 / 101 / 15131 km²**, đủ 451 tệp forcing và 451 tệp gauge. Hash nguồn Catchment là `aaa06cbfc315a618d8d60289cc02e97ca9837d7e211492c3b159a16007257dd7`, Gauge là `a6f0404d68845ff63dd7b4dd8b25b5955eebd9e0f7db049e147335f028ceba24`. Các khoảng NDVI/GVF trong output được giữ nguyên theo nguồn và giao thức B1b; kiểm hữu hạn ở đây không áp ngưỡng [0,1] cho mọi thuộc tính. Chưa đọc chuỗi ngày, đổi Q hoặc fit scaler.

## Phần 3. Bộ lọc gốc và các ví dụ biết trước

Chạy thân hàm bộ lọc từ mã nền cố định trên dữ liệu giả. Những giá trị ở phần này là kỳ vọng toán học của ví dụ, không phải số mẫu hoặc kỹ năng dự báo trên LamaH-CE.

### 3.1. Hàm lọc gốc và quy tắc đối chiếu

Kiểm lại SHA-256 `basedataset.py` từ Phần 1.2; tách đúng một hàm `_validate_samples` ở cấp module, xác nhận sáu tham số và giữ nguyên thân hàm. Chỉ bỏ decorator JIT, chạy trong namespace riêng với `np`, `prange=range` và `List`; không lấy hàm đang được vá trong module nạp vào bộ nhớ. Lưu hash cả tệp và hàm vào `B2_filter_reference.json`, quy tắc theo chế độ vào `B2_filter_rules.csv`.

Train gốc loại mẫu khi bất kỳ NaN nào trong đầu vào động của toàn cửa sổ; bộ lọc yêu cầu ít nhất một nhãn trong cửa sổ `predict_last_n`, thuộc tính biến thiên theo thời gian hợp lệ ở ngày cuối mẫu và đủ lịch sử. Wrapper đối chiếu B2 truyền `x_d=None` khi bật masked mean, giữ nguyên điều kiện nhãn, thuộc tính và lịch sử. Với một nhãn/một ngày, điều kiện nhãn tương đương qmax ngày đích không phải NaN; nhiều ngày dự báo có thể giữ cửa sổ thiếu một phần nhãn theo đúng giao thức mã nền. Thuộc tính cố định toàn lưu vực được kiểm riêng bởi `BaseDataset`; đối chiếu raw ở 2.3 không thay bước đó.

Validation/test truyền `x_d`, `x_s`, `y` bằng `None` khi lập lookup nên chỉ yêu cầu đủ lịch sử. Sự hiện diện của một mục lookup không tự xác nhận dự báo hoặc chỉ số đánh giá hữu hạn. Cell 3.1 định nghĩa quy tắc và wrapper cho fixture một tần suất/một nhãn; chưa áp patch B2 hoặc xác nhận lookup dữ liệu thật.

Kiểm cục bộ chính cell 3.1: **32 kiểm tra đạt**, gồm so nguyên thân hàm, đối chiếu với hàm đúng commit `f229e02` biên dịch bằng Numba trên 24 cấu hình ngẫu nhiên, đủ/thiếu lịch sử, đầu vào thiếu, nhãn thiếu toàn bộ/một phần và thuộc tính biến thiên thiếu. Wrapper giữ nguyên các mảng đầu vào, từ chối hình dạng/tham số sai; nguồn đổi hash bị từ chối. Hai bảng đã render HTML, các tệp quy tắc/metadata và đồng bộ 31 code cell đã kiểm.

Output Kaggle 3.1 đã nhận đúng commit `f229e02`, hash tệp `9da162ecc1cff00783fbc66e85ace386193c3c9f0f7d272429f8c5521720a494`, hash hàm `c4051cdb3848127ad6fa51b0e6439e5553efa2b663c8d75a0d002994fe8a7139`, `body_preserved=True`. Ba chế độ lookup trong output đúng quy tắc đã đối chiếu. Kết quả này xác nhận nguồn và định nghĩa wrapper; chưa phải thực thi ví dụ 3.2 hoặc lookup LamaH thật.

### 3.2. Kiểm các mặt nạ giữ và loại mẫu biết trước

Ví dụ giả 7 ngày, tổng khung 3 ngày: đầu vào có NaN tại index 3, nhãn thiếu tại index 5 và thuộc tính biến thiên theo thời gian thiếu tại index 6. Mặt nạ kỳ vọng, theo index 0–6, là `0010001` cho train gốc, `0011101` cho masked mean, `0011100` khi thêm kiểm thuộc tính biến thiên và `0011111` cho evaluation. Số mẫu giữ tương ứng là 2/4/3/5; đây chỉ là số trong fixture, không phải số liệu LamaH.

Cell kiểm **11 mặt nạ đầy đủ**, gồm bốn ca trên, toàn bộ đầu vào thiếu ở train gốc/masked mean, toàn bộ nhãn thiếu ở train, toàn bộ dữ liệu thiếu ở evaluation, không đủ lịch sử, nhãn thiếu một phần trong hai ngày dự báo và thuộc tính biến thiên được kiểm tại ngày cuối mẫu. Thêm một kiểm mảng đầu vào giữ nguyên, tổng cộng **12 kiểm**. Thuộc tính biến thiên trong fixture không thay kiểm thuộc tính cố định toàn lưu vực của bộ nạp.

Kỳ vọng được viết sẵn độc lập với kết quả hàm. Lưu `B2_filter_fixture_days.csv`, `B2_filter_fixture_masks.csv`, `B2_checks.csv` và `B2_filter_fixture_summary.json`; chỉ đánh dấu summary hoàn thành sau khi tất cả kiểm đạt. Các cờ áp patch, lookup thật và huấn luyện giữ false. Chạy lại cell thay kết quả nhóm fixture hiện tại, giữ các nhóm kiểm khác.

Kiểm cục bộ bằng chính cell `.ipynb`: **12/12 kiểm đạt, 11 mặt nạ khớp**. Đã xác nhận bộ kiểm từ chối tám biến thể cố tình làm sai: bỏ qua chế độ masked mean, bỏ kiểm nhãn, bỏ kiểm thuộc tính biến thiên, dùng quy tắc train cho evaluation, sửa mảng nguồn, bỏ kiểm lịch sử, dịch mặt nạ nhưng giữ số mẫu và đòi mọi nhãn nhiều ngày đều có dữ liệu. Đã kiểm CSV/JSON, hai bảng HTML, chạy lại và đồng bộ 31 code cell.

Output Kaggle 3.2 đã nhận: cả **12/12 kiểm đều True**, 11 mặt nạ đúng kỳ vọng và các mảng fixture không bị sửa. Bảng 7 ngày cho số mẫu giữ 2/4/3/5 đúng các chế độ đã nêu; evaluation giữ ngày thiếu nhãn tại index 5 khi đủ lịch sử. Kết quả này xác nhận fixture/wrapper, chưa xác nhận patch B2 hoặc lookup dữ liệu thật.

## Phần 4. Bản sửa bộ nạp và kiểm nguồn mã

Tạo bản sao package và một patch có thể xem lại. Bộ lọc vẫn dùng hàm Numba của thư viện; khi masked mean được bật, điểm gọi không truyền đầu vào động vào bước loại mẫu do NaN. Nhãn, tĩnh và đủ lịch sử vẫn được kiểm.

### 4.1. Đọc ngày và kiểm Q tham chiếu trên một ID

`b2_read_daily` kiểm header, cột yêu cầu, thành phần ngày nguyên/hữu hạn, ngày lịch hợp lệ/không trùng/tăng dần và giá trị số không vô hạn. Giữ khoảng trống lịch và NaN để xử lý theo hợp đồng ở bước sau; giá trị khí tượng âm được giữ nguyên. Chỉ sắp cột theo danh sách yêu cầu, không tự sắp ngày, nội suy hoặc xóa hàng.

`b2_q_reference` chỉ đọc `qmean`, `qmax` m³/s từ `D_gauges`. Đối chiếu Q âm thành NaN với `load_lamah_discharge` và gọi `_normalize_discharge` đúng `f229e02`, `area_gov` và tần suất ngày. Kết quả có `qmean_mmd`, `qmax_mmd` tham chiếu và nhãn `qmax` riêng cùng giá trị qmax_mmd trước xử lý warmup. Hàm này không đọc hai cột forcing *_mmd làm nguồn đổi đơn vị. Bộ nạp B2 chính thức dùng cột đã chuẩn bị bởi B1b; hàm tham chiếu phục vụ kiểm độc lập từ raw.

Cell đọc **một ID đầu tiên theo thứ tự nguồn**, 36 biến khí tượng và Q từ gauge, rồi đối chiếu Q lên lịch forcing bằng ngày. Ngày gauge không có trong forcing không được thêm vào khung; ngày forcing thiếu gauge giữ NaN. Ghi `B2_daily_probe_sources.csv`, `B2_daily_probe_Q_check.csv`, `B2_daily_reference_summary.json`; hai bảng output mô tả phạm vi nguồn và Q. Chưa ghi tệp forcing mới, xác nhận bàn giao/cột forcing B1b, áp patch, fit scaler hoặc lập lookup. Mã `b2_lamah_load` dùng alias prototype đã được bỏ khỏi cell 4.1; Phần 4.4 tạo bản patch tích hợp từ các thay đổi đã kiểm trên nhánh B2.

Kiểm cục bộ chính cell `.ipynb`: **33 kiểm đạt**, dùng fixture hai ID để xác nhận chỉ đọc ID đầu nguồn. Q tham chiếu khớp giá trị tính tay, bảo toàn Q bằng 0, chuyển cả -999 và Q âm khác thành NaN, giữ nhãn riêng và không đổi các cột forcing mm/ngày lần hai. Đã kiểm ghép ngày với khoảng trống gauge, NaN/khí tượng âm, thứ tự biến, header/ngày sai, giá trị vô hạn, diện tích sai, tệp gốc giữ nguyên, CSV/JSON và hai bảng HTML. Hai hàm thư viện được so AST với đúng commit `f229e02`.

Output Kaggle 4.1 đã nhận trên **ID 2**: forcing 14244 dòng từ 01/01/1981 đến 31/12/2019, SHA-256 `48b1905115a644365cda5f71bea921825f0f882897996e124d29ea3aa871ea13`; gauge 13514 dòng từ 01/01/1981 đến 31/12/2017, hash `85789248b7a4cc91c8ae2f6300baf4e4994011bb7d3760f317941535f119a92d`. Hai nguồn Q không có giá trị âm/rỗng trong tệp này; cả qmean_mmd, qmax_mmd và nhãn qmax tham chiếu có **13514 giá trị hữu hạn**, khớp quy tắc thư viện và số hữu hạn sau ghép lịch. Q ở phần lịch forcing 2018–2019 giữ NaN. Kết quả chỉ xác nhận một ID và Q từ raw, chưa xác nhận các cột forcing B1b, patch hoặc lookup thật.

### 4.2. Khóa chuẩn hóa và std nhãn theo mã nền B1b

Đã bỏ `b2_normalize` và `b2_train_stds` của prototype khỏi cell này. Giữ nguyên năm phương thức `BaseDataset`: `_load_or_create_xarray_dataset`, `_load_combined_attributes`, `_setup_normalization`, `_calculate_per_basin_std`, `_load_data`. Kiểm hash nguồn và AST các phương thức đang nạp; định nghĩa `b2_require_normalization_preserved` để kiểm bản source ứng viên ở bước tạo patch. Guard so AST, cho phép thay định dạng nhưng từ chối thay thân/chữ ký, thiếu hoặc trùng phương thức.

Mã nền tính mean/std động trên xarray train có warmup, bỏ qua NaN với ddof=0; qmax nhãn trước ngày bắt đầu kỳ đã thành NaN. Cột Q đầu vào riêng giữ dữ liệu quá khứ. Thuộc tính giữ thứ tự alphabet trong tensor và mean/sample std ddof=1. Validation/test dùng scaler đặc trưng được truyền từ train, còn `_calculate_per_basin_std` vẫn tính std nhãn của kỳ đang nạp theo `f229e02`. B2 giữ nguyên hành vi này; không thêm scaler/std nhãn riêng của prototype.

So 21 khóa chuẩn hóa/phạm vi của template B2 với `B1B_EXPECTED_CONFIG`, gồm biến đích, thuộc tính, tùy chỉnh chuẩn hóa, loss, độ dài cửa sổ, kỳ ngày và các khóa cache/tệp kỳ riêng. Đây là so template suy ra từ nguồn đã chọn, chưa phải xác nhận config B1b bàn giao. Lưu `B2_normalization_methods.csv`, `B2_normalization_config_check.csv`, `B2_normalization_contract.json`; hai bảng hiển thị nguồn phương thức và quy tắc. Cell chỉ kiểm nguồn/template, chưa fit scaler raw hoặc áp patch.

Kiểm cục bộ chính cell `.ipynb`: **39 kiểm đạt**, gồm năm phương thức gốc/đang nạp, 21 khóa template, định dạng source tương đương, dạng null đường dẫn và từ chối các thay đổi phạm vi warmup, ddof, thứ tự thuộc tính, xử lý nhãn warmup hoặc thứ tự nạp dữ liệu. Đã kiểm phương thức bị thay trong bộ nhớ, hash nguồn sai, output CSV/JSON, hai bảng HTML và đồng bộ 31 code cell. Kiểm thử thư viện `test_masked_mean_preserves_b1b_normalization` cũng đạt trên CPU: scaler động, scaler thuộc tính và std nhãn train giữa hai cấu hình giống từng giá trị trên dữ liệu giả.

Output Kaggle 4.2 đã nhận: cả **năm phương thức có `loaded_method_matched=True`**, hành động `keep unchanged`, và 21 khóa template khớp. Bảng quy tắc giữ warmup đầu vào, xóa nhãn warmup riêng, thứ tự thuộc tính, scaler đặc trưng train dùng lại khi evaluation và std nhãn theo kỳ đang nạp. Đây là kiểm nguồn/template; chưa fit scaler raw hoặc xác nhận chuẩn hóa dữ liệu B1b/B2 thật.

### 4.3. Hàm chung che forecast và chuyển counter

`prepare_forecast_inputs(data, cfg)` dùng chung cho mẫu `[time, feature]` và batch `[batch, time, feature]`. Với masked mean/counter, sao hai counter từ `x_d` vào pha tương ứng. Với `forecast_input_mode: first_day`, yêu cầu masked mean, counter, zero overlap, có hindcast và forecast chỉ ECMWF; clone từng tensor forecast rồi gán NaN tại `[..., 1:, :]`. Trục thời gian được xử lý giống nhau ở cả hai dạng tensor. Các dictionary của mẫu/batch được chuẩn bị cho mô hình, tensor nguồn và các lát cắt chồng nhau giữ nguyên.

Hàm thay `b2_prepare_sample` và khóa prototype `b2_hybrid_forecast`. Khi tích hợp `InputLayer`, counter được nối sau phép nhúng để không biến nhóm nguồn đang thiếu thành nhóm có dữ liệu. Template B2 một ngày vẫn `forecast_input_mode: all`, counter tắt; chế độ `first_day`/7 hạn ở cell này là fixture kiểm helper, chưa phải thay cấu hình huấn luyện chính thức.

Cell có **9 kiểm tensor giả**: giữ ngày đầu, che ngày sau, đúng trục thời gian batch, hindcast giữ nguyên, mẫu/batch khớp, counter đúng giá trị và được clone, tensor nguồn giữ nguyên, chế độ all/counter tắt không thay dữ liệu và template B2 giữ nguyên. Hai bảng hiển thị ví dụ một biến ECMWF ở 7 hạn và kết quả kiểm. Lưu `B2_forecast_helper_days.csv`, `B2_checks.csv`, `B2_forecast_helper_summary.json`. Chưa nối helper vào loader/GpuBatchLoader/InputLayer trong notebook hoặc xác nhận cửa sổ dữ liệu thật.

Kiểm cục bộ: **9/9 kiểm trong cell và 27 kiểm bổ sung đạt**. AST helper trùng chính xác hàm trên nhánh thư viện B2. Đã so chức năng cho mẫu/batch, hạn 1/7, all/first_day, counter bật/tắt; kiểm từ chối hợp đồng sai hoặc thiếu pha/counter, gradient chỉ đi qua HRES ngày đầu, nguồn forecast toàn NaN giữ thiếu dù có counter, các CSV/JSON/bảng HTML và chạy lại cell. **Sáu ca kiểm thư viện CPU đạt**, gồm batch thường/vector hóa bằng nhau, mô hình forward/backward, tính bất biến khi sửa HRES phát hành tương lai và từ chối cấu hình sai. Những ca này xác nhận mã trên nhánh thư viện; cell notebook vẫn chưa áp patch hoặc kiểm tích hợp model. Output Kaggle 4.3 đã nhận đủ **9/9 kiểm True**: mẫu/batch khớp, giữ ngày đầu HRES và tensor nguồn, template B2 không đổi. Chưa áp patch ở bước này.

### 4.4. Tạo bản ứng viên đúng patch B2 trên f229e02

Patch đóng băng gồm năm tệp thay đổi và hai module mới trên nhánh `codex/b2-source-masked-mean`; SHA-256 `d801c7ec70fb3db032f8795aeefa6684f6a97146f2e4fe8db94132daa5108945`. Notebook áp từng hunk vào nguồn đã kiểm hash, chỉ cho phép các đường dẫn được khai báo. Bản sao package được đặt dưới thư mục mang tiền tố hash patch. Năm phương thức chuẩn hóa được bảo vệ bằng AST; `datautils/utils.py` và định dạng scaler giữ nguyên.

LamaH chỉ đổi cách flatten danh sách biến nhóm, giữ nhãn `qmax` trước đầu vào. Điểm gọi bộ lọc bỏ kiểm NaN động khi dùng masked mean nhưng giữ nhãn, lịch sử, thuộc tính biến thiên và kiểm thuộc tính cố định. Loader thường và vector hóa gọi cùng helper forecast. `InputLayer` nối counter sau phép nhúng khi masked mean, giữ nhánh cũ khi counter tắt. Config bổ sung `forecast_input_mode`; module `source_groups` tạo B2 từ cấu hình B1b. Source cài đặt và submodule không bị ghi đè trong Kaggle.

### 4.5. Nạp bản ứng viên và kiểm builder

Nạp lại NeuralHydrology từ cùng một thư mục ứng viên, kiểm vị trí tám class/hàm và hash source. `configure_lamah_b2(Config(B1B_EXPECTED_CONFIG))` phải khớp template đã đối chiếu. Đây là kiểm builder trên config kỳ vọng; không đánh dấu bàn giao B1b đã xác nhận.

## Phần 5. Kiểm tích hợp bằng dữ liệu giả

### 5.1. Fixture hai lưu vực theo schema LamaH

Hai ID giả có diện tích 43,2 và 86,4 km², dữ liệu ngày liên tục và các ngày Q thiếu biết trước. Forcing chứa `qmean_mmd`, `qmax_mmd`; gauge giữ Q m³/s. Khởi tạo LamaH/BaseDataset thật cho train và evaluation, giữ nhãn tách khỏi Q quá khứ. Cấu hình nhỏ chỉ phục vụ kiểm mã.

### 5.2. Q, warmup và chuẩn hóa theo mã nền

Kiểm đổi Q đúng một lần bằng hệ số độc lập bằng 2 tại area=43,2. Nhãn warmup NaN nhưng đầu vào qmax_mmd còn dữ liệu. Tính kỳ vọng mean/std từ CSV float32 với warmup cho biến động, bỏ warmup của nhãn; đối chiếu std nhãn theo từng kỳ train/validation. Không thêm std train vào scaler riêng.

### 5.3. Lookup, nhãn thiếu và nhánh cũ

So toàn bộ cặp ngày–ID với tập kỳ vọng độc lập. Mẫu train thiếu đầu vào được giữ khi có nhãn; tắt masked mean phải khớp bộ lọc complete-case gốc. Evaluation giữ lịch đủ ngữ cảnh, dùng lại feature scaler và lấy mẫu chồng nhau không làm đổi nguồn.

### 5.4. Nhúng, counter, gradient và vector hóa

Kiểm nhóm thiếu/mọi nhóm thiếu, counter không làm nguồn thiếu thành nguồn có dữ liệu, config không bị sửa, forward/backward hữu hạn và cùng trọng số cho cùng đầu ra khi counter tắt. Batch thường và `GpuBatchLoader` chạy trên CPU phải khớp từng giá trị, kể cả NaN và ngày. Khi có CUDA, thêm kiểm batch CUDA; thiếu CUDA ghi `not_run_no_CUDA`, không giả định đã kiểm GPU.

### 5.5. Hợp đồng sai và ranh giới fit

Từ chối Q trong forecast, nhãn lẫn đầu vào, nguồn lạ, biến trùng và diện tích sai. Can thiệp dữ liệu ngoài phạm vi train **cộng warmup** phải không làm đổi feature scaler. Warmup vẫn thuộc phạm vi fit biến động theo giao thức gốc.

## Phần 6. Đếm raw 451 ID và đối chiếu A1

### 6.1. Bộ đếm cửa sổ độc lập

Tính mặt nạ bằng tổng tích lũy boolean, đối chiếu với duyệt từng cửa sổ trên dữ liệu giả và ngày nhuận. Không coi hai hàng kề nhau là hai ngày liên tiếp khi lịch đã bỏ ngày.

### 6.2. Đếm 451 ID và scaler gốc cho 365/730

Mỗi ID được đọc để kiểm Q và đếm theo kỳ/năm. Tách sáu loại mặt nạ: A2 complete-case toàn khung/theo pha, B1b complete-case toàn khung/theo pha, masked mean B2 và lịch evaluation. Giữ `qmean` raw cho đối chiếu hợp đồng A1; B2 dùng hai cột mm/ngày.

Một mảng float32 ánh xạ trên đĩa chứa train và warmup tối đa. Với mỗi tổng khung 365/730, dựng xarray theo đúng thứ tự lưu vực của NH, xóa nhãn warmup và gọi nguyên `BaseDataset._setup_normalization`. Thuộc tính dùng mean/std mẫu và thứ tự alphabet như mã nền. Lưu hai scaler YAML bằng `_dump_scaler` gốc; tệp mảng tạm được xóa sau khi tính để giảm dung lượng Output. Không dùng phép gộp thống kê hoặc cơ chế scaler riêng của prototype.

Chỉ các ID probe được dựng thêm forcing tham chiếu dưới `_scratch/reference_input`. Các cột khí tượng cũ giữ nội dung; lịch ngày được bổ sung ô trống khi cần, hai cột Q thêm theo hàm gốc và 17 chữ số có nghĩa. Kiểm float32 sau ghi/đọc phải khớp raw tham chiếu. Dữ liệu phụ này không phải bàn giao B1b hoặc nguồn huấn luyện chính thức. Có log tiến độ mỗi 50 ID.

### 6.3. Đối chiếu hợp đồng A1

Chỉ so nhóm có cùng tập biến/pha/nhãn, thuộc tính, kỳ ngày và hình học. Các chỉ số đối chiếu gồm số nhãn hữu hạn, complete-case toàn khung và theo pha cho hai độ dài trên ba kỳ. Khớp số lượng không chứng minh mọi tệp nguồn giống byte; thiếu hợp đồng ghi not_run và chênh số ghi review_required. Khi chưa đạt, tổng kết không đánh dấu kiểm độc lập đạt.

### 6.4. Bất biến và hồ sơ thiếu nhãn

Kiểm đủ 451 ID cho mỗi kỳ/độ dài, quan hệ tập con/tập cha và số ngày evaluation suy từ lịch. Giữ hồ sơ các ID đặc biệt để tách thiếu nhãn train khỏi thiếu đầu vào; không tạo thêm mẫu train cho ID không có nhãn.

## Phần 7. Lookup và mô hình trên dữ liệu thật tham chiếu

### 7.1. Cấu hình probe

Giữ template khoa học của B2, chuyển đường dẫn về dữ liệu tham chiếu và dùng hidden_size=16 cho mô hình kiểm ngẫu nhiên. Template huấn luyện vẫn hidden_size=128. Cấu hình probe không có epoch hay optimizer; không dùng để chọn mô hình bằng test.

### 7.2. Đối chiếu lookup 365/730

Khởi tạo LamaH cho các ID probe bằng scaler toàn tập tương ứng từng độ dài. So đầy đủ tập ngày với bộ đếm độc lập trên train/validation/test; lưu cả CSV từng ID và ví dụ cửa sổ. Chỉ đếm raw đủ 451 ID, không tự nhận đã khởi tạo loader cho toàn bộ 451 ID.

### 7.3. Batch và loss của thư viện

Batch thường/vector hóa trên dữ liệu tham chiếu phải khớp. Chạy forward/backward qua SequentialForecastLSTM và `get_loss_obj` của NH với loss NSE; kiểm dự báo, loss và gradient hữu hạn. Giá trị loss của trọng số ngẫu nhiên không phải điểm kỹ năng; số bước optimizer luôn bằng 0.

## Phần 8. Forecast nhiều hạn và kiểm thông tin tương lai

### 8.1. Hình học cửa sổ

Fixture giữ bốn bước hindcast và bảy forecast, tổng khung 11. Ngày phát hành là bước forecast đầu, Q chỉ tới ngày trước đó. Tổng khung tăng cùng số hạn, không rút ngắn lịch sử ngoài ý muốn.

### 8.2. Che dữ liệu và counter trong hai loader

Kiểm mask/counter được áp khi lấy mẫu, tensor lưu chung còn nguyên, lấy mẫu chồng nhau ổn định và mô hình bảy hạn hữu hạn. Đối chiếu thêm batch vector hóa với batch collate thường.

### 8.3. Can thiệp dữ liệu tương lai

Thay Q/tái phân tích từ ngày phát hành và HRES phát hành sau ngày đầu phải thay nhãn nhưng giữ đầu vào được phép cùng dự báo. Thay HRES đúng ngày phát hành là đối chứng dương: đầu vào forecast ngày đầu phải đổi. Không sửa tệp Input, chỉ can thiệp hàm đọc trong phạm vi kiểm.

### 8.4. Scaler YAML và probe nhiều hạn thật

Đọc lại scaler bằng hàm NH gốc, kiểm feature center/scale và tạo evaluation dataset. Không thêm hoặc đòi std train trong YAML. Trên một ID thật tham chiếu, kiểm 364 bước hindcast, bảy forecast, Q tới t−1, mask HRES, batch vector hóa và dự báo hữu hạn. Đây là probe dữ liệu, chưa là thí nghiệm nhiều hạn đã huấn luyện.

## Phần 9. Tổng kết và điểm chờ

### 9.1. Trạng thái bằng chứng

`B2_LOADER_INDEPENDENT_OK` chỉ True khi các kiểm giả, đếm raw, đối chiếu A1, lookup, mô hình và probe hybrid đều đạt. `B2_READY_FOR_TRAINING=False` cho tới khi bàn giao B1b, cấu hình/dữ liệu được đối chiếu và commit thư viện được duyệt. Chỉ có mặt config chưa đủ xác nhận bàn giao. Chưa merge research hoặc huấn luyện ba seed trong notebook này.

### 9.2. Output và manifest

Lưu `B2_summary.json`, `B2_checks.csv`, bảng trạng thái, `environment.json`, source ứng viên và patch. `manifest.csv`/`B2_output_manifest.csv` chứa hash bằng chứng của lần chạy, gồm summary và scaler YAML; bỏ qua dữ liệu phụ/tệp fixture trong scratch. Chạy Save & Run All kết thúc tại đây, không tự bắt đầu huấn luyện.

## Kiểm cục bộ của bản hiện tại

Toàn bộ **31 code cell** đã chạy bằng IPython trên cây Input giả đủ 451 ID; **47/47 kiểm đạt**, 18 dòng đối chiếu A1 giả khớp, 43 bảng HTML được tạo và 68 mục manifest được kiểm hash. Source ứng viên khớp AST với cả bảy tệp thay đổi/module mới trên nhánh B2. Dữ liệu giả không dùng để kết luận số mẫu hoặc kỹ năng LamaH thật. Môi trường cục bộ: Python 3.12, NumPy 1.26.4, pandas 2.3.3, PyTorch 2.9.1 CPU.

Kiểm thư viện: **13 ca B2 và 13 ca hồi quy đạt trên CPU**; một ca CUDA bỏ qua vì môi trường không có CUDA. Các kiểm huấn luyện dài của bộ hồi quy không được chạy lại trong lượt này. Output Kaggle 4.4–9.2 của bản hiện tại đã nhận và được đối chiếu bên dưới.

## Kết quả Kaggle bản f229e02 theo output đã nhận

Nguồn bằng chứng là các bảng/log người dùng gửi, chưa phải tệp `.ipynb` tải từ một Save Version đã xác định. Đã đối chiếu **8/8 hash source** của Phần 4.4 với source ứng viên cục bộ; cùng patch `d801c7ec70fb3db032f8795aeefa6684f6a97146f2e4fe8db94132daa5108945`. Cả tám class/hàm tại 4.5 được nạp từ cùng thư mục ứng viên. `datautils/utils.py` giữ nguyên hash nền, các phương thức chuẩn hóa được bảo toàn.

Fixture có 22 mẫu train, 16 validation. Cả chín kiểm loader, năm kiểm model CPU, ba kiểm hợp đồng và kiểm bộ đếm cửa sổ đều đạt. Đã đếm đủ **451 ID**, đối chiếu A1 theo hợp đồng `Wb741da88055e`: **18/18 dòng chênh lệch bằng 0**. Điều này xác nhận số đếm trên hợp đồng chung, không tự xác nhận mọi tệp raw lịch sử giống byte.

| Tổng khung | Train complete-case toàn khung | Train complete-case theo pha | Train masked mean | Giữ thêm so với toàn khung |
|---:|---:|---:|---:|---:|
| 365 | 1094272 | 1094272 | 1112307 | 18035 |
| 730 | 928114 | 1079046 | 1112307 | 184193 |

Ở khung 730, 150932 mẫu được giữ thêm khi chỉ kiểm đúng pha và 33261 mẫu thêm khi cho phép đầu vào thiếu bằng masked mean. Các số trên là số mẫu đủ điều kiện, chưa phải bằng chứng NSE tăng. Các ID 324/600/623 không có nhãn train, giữ 0 mẫu train; ID 55 và 800 có lần lượt 1292 và 38 mẫu train masked mean ở khung 365. Giữ đủ 451 ID trong hợp đồng và báo riêng thiếu nhãn/thiếu đầu vào.

Lookup thực trên **10 ID** khớp toàn bộ tập ngày ở cả 365/730: mỗi độ dài có 11923 mục train, 14610 validation, 14610 test. Cửa sổ giữ đúng 364/729 ngày hindcast, kết thúc t−1; các mốc ngày nhuận trong output phù hợp. Không diễn giải thành đã khởi tạo loader trên toàn bộ 451 ID.

Cả **sáu kiểm hybrid** đạt, gồm mẫu/batch vector hóa bằng nhau và can thiệp dữ liệu tương lai giữ nguyên đầu vào được phép/dự báo. YAML scaler đọc lại đúng định dạng gốc. Probe nhiều hạn trên ID 2 phát hành 01/01/2010, hindcast kết thúc 31/12/2009, dự báo tới 07/01/2010: 364 bước hindcast, 7 forecast, `HRES_from_future_releases_used=False`, dự báo hữu hạn. Bảng 9.1 báo probe gradient và batch thật đạt; bảng chi tiết 7.3 không có trong phần dán nên chưa ghi giá trị loss hoặc hình dạng dự báo cụ thể của lần chạy này.

Tổng kết 9.2 có **47/47 kiểm đạt, 451 ID đếm raw, 10 ID probe, 0 bước optimizer**; kiểm độc lập True và sẵn sàng huấn luyện False. CUDA ghi `not_run_no_CUDA`; chưa xác nhận GPU. B1b chưa được bàn giao/đối chiếu, nhánh thư viện chưa merge, ba seed B2 chưa được huấn luyện.

Các SyntaxWarning về `\d` trong `datautils/utils.py` và `\s` trong `datasetzoo/camelsus.py` thuộc chuỗi regex của mã nền. Chúng không phải traceback và không làm các kiểm nhận được thất bại; giữ source pin/hash hiện tại. Cảnh báo thiếu std tại 324/600/623 phù hợp với 0 nhãn train; không điền std bằng nhãn evaluation.

Phần dán lặp bảng 7.2 và thiếu ba ô cuối hàng test/730 của bảng 6.2. Chưa nhận CSV/JSON gốc để kiểm riêng nội dung/hashes; không ghi các ô suy ra như số đo trực tiếp. Các hạn chế này không tạo thêm lỗi thuật toán trong output đã cung cấp. Mã `.py` không thay đổi sau đợt rà này; chưa cần chạy lại chỉ vì các cảnh báo trên. Bản `.ipynb` cục bộ chỉ cập nhật receipt/diễn giải, không giả lập output thực thi. Số Save Version chưa được xác nhận.

## Output cần đối chiếu sau khi chạy

- Bảng/tóm tắt Phần 6.2–6.4, 7.2–7.3, 8.4 và 9.1–9.2; mọi traceback nếu có.
- `B2_summary.json`, `B2_checks.csv`, `B2_A1_count_comparison.csv`, `B2_real_lookup_audit.csv`, `B2_candidate_source.json`, `manifest.csv`.
- Giữ patch và các scaler YAML trong Output Save Version để đối chiếu tiếp với B1b.

## Phụ lục: bằng chứng prototype cũ, không phải bản B2 hiện tại

Các kết quả dưới đây thuộc lần chạy ngày 22–23/10/2026 trên nền `dda45fc`. Prototype có tên Q và cách fit scaler khác; giữ để truy vết, không dùng thay xác nhận bản `f229e02` hiện tại hoặc bàn giao B1b.

### Kết quả thực tế trên Kaggle

Các kết quả dưới đây lấy từ output Kaggle được cung cấp ngày 23/10/2026, không phải số của bộ dữ liệu giả cục bộ. Output xác nhận `B2_LOADER_INDEPENDENT_OK=True`, `B2_READY_FOR_TRAINING=False` và không có traceback thất bại. CSV/JSON/patch trong Kaggle Output chưa được tải về và kiểm hash riêng tại máy; nội dung ngoài các bảng/log đã cung cấp không được coi là đã kiểm trực tiếp.

#### Môi trường và nguồn đã nhận diện

| Thành phần | Giá trị thực tế |
|---|---|
| Python | 3.13.15 |
| NumPy / pandas / xarray | 2.1.3 / 2.3.3 / 2025.12.0 |
| PyTorch | 2.11.0+cpu |
| SciPy / Numba / ruamel.yaml / tqdm | 1.16.3 / 0.61.2 / 0.19.1 / 4.67.3 |
| Raw | /kaggle/input/datasets/hunhngcthng/lamah-ce-ext/lamah_ce_ext/LamaH_extended |
| Danh sách lưu vực / config tìm thấy / CSV A1 Phần 7 | 451 / 36 / 21 |
| Thuộc tính tĩnh | 33 |
| area_gov nhỏ nhất / trung vị / lớn nhất | 3,9 / 101 / 15131 km² |

Nhóm nguồn khớp hợp đồng: hindcast có ERA5-Land 21, E-OBS 7, MSWEP 1, GLEAM 2 và Q 2 biến; forecast có 5 ECMWF. Phần 4 đã tạo patch và nạp các lớp từ library_candidate. Việc hoàn tất các cell sau các điều kiện hash mã nền xác nhận notebook đã đi qua bước kiểm nguồn; chưa có mã commit tích hợp B1.

#### Kiểm số học, đọc dữ liệu và đường NaN

| Nhóm kiểm | Số ca đạt | Phạm vi |
|---|---:|---|
| fixture_filter | 5 | Mặt nạ biết trước; giữ đầu vào thiếu, loại nhãn/tĩnh thiếu, đủ lịch sử |
| synthetic_loader | 9 | Đổi đơn vị, alias, warmup, scaler train-only, std train, lookup và lấy mẫu lặp |
| synthetic_model | 4 | Nhóm thiếu, fallback, counter, gradient và đầu ra khi counter tắt |
| input_contract | 3 | Hợp đồng sai, diện tích lỗi và can thiệp ngoài kỳ train |
| count_oracle | 1 | Rolling so với duyệt từng cửa sổ, lịch ngày và ngày nhuận |
| real_counts | 1 | Quan hệ mặt nạ và số ngày/ID trên raw 451 lưu vực |
| hybrid_forecast | 5 | Che HRES, counter, mẫu chồng lấn, bảy bước và can thiệp tương lai trên fixture |
| scaler_roundtrip | 1 | YAML được đọc lại bằng load_scaler và còn std train |
| Tổng | 29 | Mọi ca được in trong output đều passed=True |

Dataset giả hai lưu vực qua loader có 22 mẫu train và 16 mẫu validation; đây chỉ là ca kiểm, không đưa vào thống kê LamaH thật. Kiểm can thiệp tương lai trên fixture đã giữ nguyên input hợp lệ và đầu ra deterministic, trong khi nhãn tương lai thực sự thay đổi. Kết quả này đi cùng kiểm mặt nạ và cửa sổ raw, không thay thế thí nghiệm dự báo nhiều hạn chính thức.

#### Số mẫu train và phần được giữ thêm

| Tổng khung | Bộ lọc toàn khung A2 | Complete-case theo pha A2 | Masked mean B2 | Giữ thêm so với bộ lọc toàn khung | Tăng theo số mẫu gốc |
|---:|---:|---:|---:|---:|---:|
| 365 | 1094272 | 1094272 | 1112307 | 18035 | 1,65% |
| 730 | 928114 | 1079046 | 1112307 | 184193 | 19,85% |

Ở khung 730, chuyển từ kiểm toàn khung sang chỉ kiểm các ô được dùng theo pha giữ thêm **150932** mẫu; cho phép thiếu đầu vào bằng masked mean giữ thêm **33261** mẫu so với mốc complete-case theo pha. Hai phần cộng thành **184193**. Ở khung 365, hai cách complete-case có cùng số mẫu; masked mean giữ thêm **18035** mẫu. Đây là mức thay đổi tập mẫu đủ điều kiện, chưa chứng minh kỹ năng dự báo tăng.

Với dữ liệu và hợp đồng hiện tại, thêm qmax_past vào đầu vào chưa làm thay các số complete-case so với cấu hình A2 có qmean: original_B1_inputs và phase_complete_B1_inputs bằng các mốc A2 tương ứng. Không suy điều này đúng với mọi nguồn dữ liệu hoặc kịch bản mất dữ liệu. Cột B1_inputs chỉ là hợp đồng đầu vào ứng viên, không phải kết quả huấn luyện B1.

Hai tổng khung có cùng **1112307** mẫu train đủ điều kiện theo masked mean, bằng số nhãn train hữu hạn trong phạm vi đã kiểm. Vẫn giữ lịch 1153207 ngày–ID và danh sách 451 lưu vực; nhãn thiếu không trở thành mẫu train.

#### Validation/test: điều kiện chấm khác với lịch lookup

| Kỳ | Tổng khung | Complete-case A2 có nhãn | Ngày–ID có nhãn, masked mean B2 | Chênh so với complete-case | Lookup theo lịch |
|---|---:|---:|---:|---:|---:|
| Validation | 365 | 652489 | 656281 | 3792 | 658911 |
| Test | 365 | 652636 | 657366 | 4730 | 658911 |
| Validation | 730 | 648779 | 656281 | 7502 | 658911 |
| Test | 730 | 649787 | 657366 | 7579 | 658911 |

Lookup evaluation vẫn là **658911** mục mỗi kỳ ở cả hai độ dài; mã nền vốn đã giữ lịch evaluation mà không bắt buộc đầu vào hoặc nhãn hữu hạn khi lập lookup. Vì vậy các chênh lệch trong bảng không phải số ngày lịch mới được tạo hoặc số dự báo hữu hạn đã đo của mô hình B2 được huấn luyện. Khi đánh giá mô hình chính thức còn phải lập giao obs/sim thực sự hữu hạn.

Đối chiếu A1 có **18/18** dòng chênh lệch bằng 0 ở hợp đồng `Wb741da88055e`, bao gồm số nhãn và hai mốc complete-case trên ba kỳ, hai độ dài. Trạng thái `counts_match_source_identity_not_proven` được giữ: khớp số đếm không tự chứng minh mọi tệp raw/cache lịch sử giống byte.

#### Các hồ sơ train cần giữ

| ID | Nhãn train hữu hạn | Complete-case toàn khung 365 | Masked mean B2 |
|---:|---:|---:|---:|
| 55 | 1292 | 0 | 1292 |
| 295 | 365 | 1 | 365 |
| 324 | 0 | 0 | 0 |
| 600 | 0 | 0 | 0 |
| 623 | 0 | 0 | 0 |
| 742 | 2557 | 2557 | 2557 |
| 758 | 2557 | 2557 | 2557 |
| 800 | 38 | 0 | 38 |

ID 55 và 800 có thể tham gia train theo điều kiện mới khi có nhãn, dù thiếu cửa sổ đầu vào complete-case. Điều này không chứng minh mọi mẫu có đầy đủ thông tin hoặc mô hình sẽ dự báo tốt. ID 324/600/623 vẫn không có nhãn train, không tạo mẫu train và không lấy evaluation để điền std train.

ID 295 còn một mẫu complete-case ở khung NH 365 vì pha hindcast có 364 ngày. DLinear A3 đòi 365 ngày qmean quá khứ và từng có 0 mẫu train tại ID này; hai con số thuộc hai cửa sổ khác nhau.

#### Lookup thật, probe mô hình và dữ liệu nhiều hạn

Đã khởi tạo LamaH/BaseDataset trên **10 ID kiểm**, so cả tập ngày lookup với bộ đếm độc lập. Với mỗi độ dài 365 và 730, train có **11923** mục, validation và test đều **14610**; tất cả dòng tổng hợp có all_dates_match=True. Không diễn giải probe 10 ID thành đã khởi tạo loader hay huấn luyện trên toàn bộ 451 ID.

Các cửa sổ ngày đầu đúng hình học: train 365 từ 02/01/2002 đến 01/01/2003, hindcast kết thúc 31/12/2002; train 730 bắt đầu 02/01/2001 và có cùng ngày đích. Validation/test giữ ranh giới tương ứng, kể cả ngày nhuận trong khung 730.

Probe ngẫu nhiên trên batch thật có batch_size=8, hidden_size=16, prediction_shape=[8,1,1], loss NSE* hiển thị **0,001107** và gradient hữu hạn trên CPU. Số bước optimizer bằng **0**. Loss này chỉ kiểm đường tính toán; không phải NSE đánh giá hoặc kết quả huấn luyện B2.

Probe cách (b) trên raw tại **ID 2** phát hành 01/01/2010, chỉ dùng Q đến 31/12/2009, giữ 364 bước hindcast và 7 bước forecast đến 07/01/2010. Cờ HRES_from_future_releases_used=False và prediction_finite=True. Thí nghiệm H nhiều hạn vẫn chưa được huấn luyện hoặc đánh giá.

Mốc tiến độ cuối Phần 6.2 là **415,4 giây kể từ STARTED ở Phần 1.1**. Con số này gồm các bước chuẩn bị/kiểm trước đó, không phải thời gian đọc raw riêng hoặc tổng thời gian hoàn tất notebook; output hiện nhận chưa có elapsed_seconds của B2_summary.json.

#### Cảnh báo và trạng thái bàn giao

Các SyntaxWarning về escape `\d` trong datautils/utils.py và `\s` trong datasetzoo/camelsus.py thuộc chuỗi regex của mã nền NeuralHydrology. Package datasetzoo import cả lớp CamelsUS khi nạp các dataset, dù notebook dùng LamaH. Chúng là cảnh báo khi biên dịch/import, không phải traceback thất bại trong lần chạy được cung cấp. Giữ nguyên nguồn pin và hash; chưa thêm thay đổi thuật toán hoặc yêu cầu chạy lại chỉ vì các cảnh báo này.

Phần kiểm độc lập trên Kaggle đã đạt. Phần còn lại là rà patch cùng Người 2 và tích hợp lên mã/cấu hình B1 cụ thể; sau đó kiểm lại phần tích hợp bị ảnh hưởng và bàn giao commit thư viện. `B2_READY_FOR_TRAINING=False` phản ánh việc này, không phải bộ lọc hoặc phép đếm đang lỗi. Huấn luyện chính thức ba seed B2 thuộc Người 1 theo phân công hiện hành, chưa bắt đầu trong notebook này.

### Trạng thái kiểm chứng và giới hạn

Kiểm cục bộ ngày 22/10/2026: 31 cell đã chạy qua IPython với bộ LamaH giả 451 ID, 29 ca kiểm đạt. Bộ giả có lịch, config, thuộc tính và bảng đối chứng giả được ghi rõ nguồn; các số đếm của nó không đưa vào kết quả LamaH thật. Đã kiểm đường đọc CSV, lookup ngày ở 365/730, sinh patch từ cell notebook, scaler YAML, forward/backward và can thiệp tương lai. Môi trường kiểm cục bộ là Python 3.12.6, NumPy 1.26.4, pandas 2.3.3, xarray 2025.12.0, PyTorch 2.9.1 trên CPU; môi trường Kaggle có thể khác và sẽ được notebook ghi lại.

Mã nền được cố định tại dda45fcf19ee7ffd481b9d192778539df46fb074. Bản ứng viên chưa tích hợp B1; B2_READY_FOR_TRAINING vì vậy giữ False trong kết quả hiện có. Lần chạy Kaggle đã nhận đủ CSV A1 để đối chiếu, không có trạng thái not_run cho phần này. Bộ nạp dùng một tần suất ngày, một kỳ train chung và Sequential Forecast overlap=0; cấu hình nhiều tần suất/per-basin periods hoặc kiến trúc khác cần kiểm riêng.

Không yêu cầu huấn luyện ba seed để xác nhận đường dữ liệu, không gọi kiểm source/ZIP là tái huấn luyện, không lấy điểm test để chọn giải pháp. Việc tích hợp B1, cấu hình huấn luyện chính thức và freeze-C phải được bàn giao ở bước tương ứng.

### Nguồn đối chiếu

- [Phân công Người 1](../../../ASSIGNMENT.md), bộ nạp B2 và điểm bàn giao.
- [Pipeline](../../../Document/01_Plan/02_Pipeline.md), Mục 3, 4.4, 7 và 9.
- [A1](../02_Exploration/01_A1_LamaHCEExt_Explore.md), hợp đồng Phần 7 và bàn giao Mục 25.
- [A3](../03_Baseline/02_A3_LamaHCEExt_Benchmarks.md), Q nguồn/diện tích/đơn vị và giới hạn nhãn train.
- [Mã LamaH nền](../../NeuralHydrology/neuralhydrology/datasetzoo/lamah.py), [BaseDataset](../../NeuralHydrology/neuralhydrology/datasetzoo/basedataset.py), [InputLayer](../../NeuralHydrology/neuralhydrology/modelzoo/inputlayer.py) và [đọc scaler](../../NeuralHydrology/neuralhydrology/datautils/utils.py).
