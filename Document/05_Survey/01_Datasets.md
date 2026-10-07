# Khảo sát bộ dữ liệu và công trình liên quan cho bài toán dự báo lưu lượng dòng chảy

> Tài liệu tổng hợp **4 bộ dữ liệu công khai**: **LamaH-CE**, **CAMELS-US**, **Columbia Basin (CBR/DART)**, **WaterBench-Iowa**, cùng các công trình **có huấn luyện mô hình trên chính bộ dữ liệu đó**.
> Ngày tra cứu: **13/9/2026**. Thông tin công trình được đối chiếu với abstract gốc, DOI hoặc cơ sở dữ liệu học thuật OpenAlex. Lưu ý về độ tin cậy số liệu: xem **Mục 8**.

## Mục lục

1. So sánh nhanh 4 bộ dữ liệu
2. LamaH-CE
3. CAMELS-US
4. Columbia Basin (CBR/DART)
5. WaterBench-Iowa
6. Công trình huấn luyện mô hình trên từng bộ dữ liệu
7. Phân tích tổng hợp
8. Lưu ý về số liệu
9. Thế mạnh của từng bộ dữ liệu
10. Nguồn tham khảo

---

## 1. So sánh nhanh 4 bộ dữ liệu

| Tiêu chí | LamaH-CE | CAMELS-US | Columbia Basin (CBR/DART) | WaterBench-Iowa |
|---|---|---|---|---|
| Loại dữ liệu | Large-sample hydrology, đóng gói sẵn | Large-sample hydrology, đóng gói sẵn | Cổng dữ liệu vận hành, truy vấn theo nhu cầu | **Benchmark thiết kế riêng cho học máy** (nguyên tắc FAIR) |
| Khu vực | Trung Âu (thượng lưu Danube, Áo) | Hoa Kỳ lục địa | Lưu vực sông Columbia (Mỹ) | Bang Iowa, Mỹ |
| Quy mô | **859** lưu vực có trạm đo | **671** lưu vực | 35+ điểm đo (đập, trạm USGS) | **125** lưu vực |
| Thời gian | **1981–2017** | **1980–2014** | Lưu lượng từ **1878**; chất lượng nước theo giờ từ **1995** | **10/2011 – 9/2018** |
| Độ phân giải | **Ngày + giờ** | **Ngày** (bản gốc) · **giờ** ở bản mở rộng 516 lưu vực (Gauch et al., 2021) | **Ngày** cho lưu lượng và vận hành đập · **giờ** chỉ cho chất lượng nước | **Chỉ giờ** |
| Biến đầu vào | **15** biến khí tượng (ERA5-Land) | **5** biến × 3 bộ forcing | Không có forcing khí tượng đóng gói | **7** đặc trưng |
| Thuộc tính lưu vực | **>60** thuộc tính | 6 nhóm thuộc tính | Không có | Có (gộp trong 7 đặc trưng) |
| Topology mạng lưới sông | ✅ **Có sẵn** (`HIERARCHY`, `NEXTUPID`, `NEXTDOWNID`) | ❌ Không (lưu vực độc lập) | Có quan hệ thực tế nhưng không có tệp topology | ✅ Dựng được từ DEM + hướng dòng chảy (mã nguồn HydroGAT) |
| **Số công trình huấn luyện mô hình** | **4** | **11** | **1** | **3** |
| Có công trình Mamba/SSM | ❌ | ✅ (S4D-FT, S4D/S5D, HydroDiffusion; Mamba trong Zhang et al. và Demiray & Demir) | ✅ (ResBi-Mamba Plus) | ✅ (Mamba của Demiray & Demir, bản preprint) |
| Có công trình mô hình đồ thị | ✅ 2 công trình, **kết luận trái ngược** | ❌ | ❌ | ✅ (HydroGAT) |
| Venue CNTT hạng cao nhất | **ICML 2024 (CORE A\*)** | Expert Systems with Applications | AI Science and Engineering (IEEE) | **ACM SIGSPATIAL 2025 (CORE A)** |
| Kết quả mốc để đối chiếu | LSTM NSE 0,46 → LSTM-GAT 0,61 | S4D NSE 0,756 · S5D 0,763 | — | NSE trung vị 0,74 · KGE trung vị 0,79 (benchmark gốc, 120 giờ) |
| Mã nguồn công khai | ✅ Mô hình GNN của ICML 2024 + bộ đọc dữ liệu (neuralhydrology) | Bộ đọc dữ liệu (thư viện neuralhydrology, gồm `camelsus` và `hourlycamelsus`) | — | ✅ Dataset + mô hình HydroGAT |
| Giấy phép | CC-BY 4.0 | Công khai (NCAR/UCAR) | Công khai (cơ quan chính phủ Mỹ) | Công khai (FAIR) |
| Dung lượng | 1,5 GB (ngày) · 14,8 GB (ngày + giờ) | ~15 GB nén · ~130 GB giải nén | Theo truy vấn (CSV) | Repo GitHub ~7,6 GB (gồm dữ liệu thô và dữ liệu đã xử lý) |
| DOI dữ liệu | [10.5281/zenodo.4525244](https://doi.org/10.5281/zenodo.4525244) | [10.5065/D6MW2F4D](https://dx.doi.org/10.5065/D6MW2F4D) · [10.5065/D6G73C3Q](https://doi.org/10.5065/D6G73C3Q) | cbr.washington.edu/dart | [10.5194/essd-14-5605-2022](https://doi.org/10.5194/essd-14-5605-2022) |

---

## 2. LamaH-CE

### 2.1 Thông tin xuất bản

| Mục | Nội dung |
|---|---|
| Tên đầy đủ | **LA**rge-Sa**M**ple D**A**ta for **H**ydrology and Environmental Sciences for **C**entral **E**urope |
| Tác giả | Christoph Klingler, Karsten Schulz, Mathew Herrnegger (BOKU Vienna) |
| Tạp chí | Earth System Science Data (ESSD), 2021 |
| DOI bài báo | [10.5194/essd-13-4529-2021](https://doi.org/10.5194/essd-13-4529-2021) |
| DOI dữ liệu | [10.5281/zenodo.4525244](https://doi.org/10.5281/zenodo.4525244), bản v1.0 tại [zenodo.org/records/5153305](https://zenodo.org/records/5153305) |
| Giấy phép | CC BY-SA 4.0 (Attribution-ShareAlike, theo mô tả bản v1.0 trên Zenodo): được dùng và sửa đổi, kể cả thương mại, nhưng phải ghi nguồn và dữ liệu phái sinh khi chia sẻ phải dùng cùng giấy phép |

### 2.2 Phạm vi

- **859 lưu vực có trạm đo**, tổng diện tích khoảng **170.000 km²**
- **9 quốc gia**: Áo, Đức, Séc, Thụy Sĩ, Slovakia, Ý, Liechtenstein, Slovenia, Hungary
- Độ cao từ **130 đến 4.049 m**, gồm cả đồng bằng lẫn vùng núi Alps
- Thời gian **1981–2017**; dữ liệu khí tượng liên tục, không khuyết

### 2.3 Ba kiểu phân định lưu vực

| Kiểu | Thư mục | Ý nghĩa | Số lưu vực |
|---|---|---|---|
| **A** | `A_basins_total_upstrm` | Toàn bộ diện tích thượng nguồn của trạm (giống CAMELS) | 859 |
| **B** | `B_basins_intermediate_all` | **Lưu vực trung gian**: phần diện tích giữa trạm đang xét và các trạm ngay phía trên, tạo thành mạng lưới liên kết | 859 |
| **C** | `C_basins_intermediate_lowimp` | Như kiểu B nhưng chỉ giữ trạm ít chịu tác động của con người | 454 |

Kiểu B phù hợp để xây dựng đồ thị vì các lưu vực trung gian không chồng lấn diện tích. Ở kiểu A, diện tích lưu vực hạ nguồn bao trùm cả lưu vực thượng nguồn.

### 2.4 Topology mạng lưới sông

| Trường | Ý nghĩa |
|---|---|
| `HIERARCHY` | Bậc của trạm trong mạng lưới |
| `NEXTUPID` | Mã trạm ngay phía thượng nguồn |
| `NEXTDOWNID` | Mã trạm ngay phía hạ nguồn |

- Mạng lưới là **đồ thị có hướng không chu trình (DAG)**: nút là trạm đo, cạnh là hướng dòng chảy
- Có tham chiếu chéo sang EU-Hydro và RiverATLAS
- Thành phần liên thông lớn nhất ("Danube A") gồm **608/859 trạm**. Công trình ICML 2024 (Mục 6.1.1) lọc tiếp còn **358 trạm** có dữ liệu giờ liên tục giai đoạn 2000–2017 để xây dựng đồ thị

### 2.5 Dữ liệu khí tượng

- **15 biến** từ ECMWF ERA5-Land, có cả bản ngày và giờ, đã tính trung bình theo lưu vực
- Gồm các nhóm: giáng thủy, nhiệt độ, gió, bức xạ, độ ẩm, độ dày tuyết, độ ẩm đất, bốc thoát hơi. Danh sách chi tiết ở Bảng A2 của bài báo gốc

### 2.6 Thuộc tính lưu vực (hơn 60 thuộc tính)

| Nhóm | Số lượng |
|---|---|
| Địa hình | 10 |
| Khí hậu | 12 |
| Đặc trưng thủy văn | 14 |
| Lớp phủ đất | 6 |
| Thảm thực vật | 6 |
| Đất | 10 |
| Địa chất | 16 |
| Tác động của con người | 4 |

### 2.7 Cấu trúc tệp

| Đường dẫn | Nội dung |
|---|---|
| `A_basins_total_upstrm/1_attributes/Catchment_attributes.csv` | Thuộc tính lưu vực |
| `A_basins_total_upstrm/2_timeseries/daily/ID_{id}.csv` | Forcing khí tượng theo ngày (tương tự `hourly/`) |
| `B_basins_intermediate_all/...`, `C_basins_intermediate_lowimp/...` | Cấu trúc tương tự cho kiểu B và C |
| `D_gauges/1_attributes/Gauge_attributes.csv` | Thuộc tính trạm, gồm topology và diện tích `area_gov` |
| `D_gauges/2_timeseries/daily/ID_{id}.csv` | Lưu lượng quan trắc theo ngày (tương tự `hourly/`) |

- Cột lưu lượng: **`qobs`**, đơn vị **m³/s**; cột thời gian `YYYY`, `MM`, `DD` (bản giờ có thêm `hh`, `mm`); có cờ chất lượng
- Quy đổi sang mm/ngày: `qobs / (area_gov × 10⁶) × 1000 × 86400`

### 2.8 Tải về

| Tệp | Nén | Giải nén |
|---|---|---|
| `1_LamaH-CE_daily_hourly.tar.gz` | 14,8 GB | ~70 GB |
| `2_LamaH-CE_daily.tar.gz` | 1,5 GB | ~5 GB |

### 2.9 Ưu điểm và hạn chế

**Ưu điểm:** có sẵn topology mạng lưới sông; nhiều biến khí tượng (15) và thuộc tính (>60); có cả ngày và giờ; có nhóm thuộc tính mô tả tác động của con người (hồ chứa); bản theo ngày nhẹ.

**Hạn chế:** chưa có công trình Mamba hoặc Transformer để làm mốc đối chiếu; khi xây dựng đồ thị cần lọc bớt trạm (công trình ICML 2024 chỉ giữ 358/859 trạm sau khi lọc theo tính liên thông và độ đầy đủ dữ liệu).

---

## 3. CAMELS-US

### 3.1 Thông tin xuất bản

| Mục | Nội dung |
|---|---|
| Tên đầy đủ | **C**atchment **A**ttributes and **ME**teorology for **L**arge-sample **S**tudies |
| Bài báo | Newman et al. 2015, chuỗi thời gian ([10.5194/hess-19-209-2015](https://doi.org/10.5194/hess-19-209-2015)) · Addor et al. 2017, thuộc tính ([10.5194/hess-21-5293-2017](https://doi.org/10.5194/hess-21-5293-2017)) |
| Tạp chí | Hydrology and Earth System Sciences (HESS) |
| DOI dữ liệu | [10.5065/D6MW2F4D](https://dx.doi.org/10.5065/D6MW2F4D) (chuỗi) · [10.5065/D6G73C3Q](https://doi.org/10.5065/D6G73C3Q) (thuộc tính) |
| Đơn vị quản lý | NCAR/UCAR; chuỗi v1.2 (3/2016), thuộc tính v2.0 (10/2017) |

### 3.2 Phạm vi

- **671 lưu vực** trên Hoa Kỳ lục địa (25–50°N, 66–125°W)
- Diện tích **4 – 25.000 km²**, trung vị **336 km²**
- Tiêu chí chọn: lưu vực **ít chịu tác động của con người**. Điều kiện này khác với các lưu vực có hồ chứa điều tiết

### 3.3 Thời gian

| Bộ forcing | Thời gian |
|---|---|
| Daymet | 1/1/1980 – 31/12/2014 |
| NLDAS | 1/1/1980 – 31/12/2014 |
| Maurer | 1/1/1980 – 31/12/2008 |
| Lưu lượng (USGS) | 1980–2014 |

**Bản mở rộng theo giờ:** bản CAMELS-US gốc chỉ có dữ liệu theo ngày. Gauch et al. (2021, *Rainfall–runoff prediction at multiple timescales with a single Long Short-Term Memory network*, HESS 25:2045) xây dựng thêm bản theo giờ cho **516/531 lưu vực** có số đo lưu lượng theo giờ từ USGS, dùng forcing theo giờ từ **NLDAS**. Thư viện neuralhydrology có bộ đọc riêng cho bản này (`hourlycamelsus`).

### 3.4 Dữ liệu khí tượng

Có **3 bộ forcing độc lập** cho cùng một lưu vực (Daymet, Maurer, NLDAS), nhờ đó kiểm tra được độ nhạy của mô hình với nguồn dữ liệu đầu vào.

| Biến | Tên | Đơn vị |
|---|---|---|
| `PRCP` | Tổng giáng thủy ngày | mm/ngày |
| `SRAD` | Bức xạ sóng ngắn trung bình | W/m² |
| `Tmax` | Nhiệt độ cao nhất | °C |
| `Tmin` | Nhiệt độ thấp nhất | °C |
| `Vp` | Áp suất hơi nước | Pa |

Riêng bộ Daymet có thêm `dayl` (độ dài ngày).

### 3.5 Lưu lượng

Lấy từ **USGS National Water Information System**, là số đo thực tế tại trạm.

### 3.6 Thuộc tính lưu vực

Gồm 6 nhóm: địa hình, khí hậu, đặc trưng dòng chảy, lớp phủ đất, đất, địa chất.

### 3.7 Tải về

Dữ liệu ở dạng tệp ASCII, khoảng **15 GB nén / 130 GB giải nén**; tệp thuộc tính khoảng 1 MB; kèm shapefile ranh giới lưu vực. Tải tại [ral.ucar.edu/solutions/products/camels](https://ral.ucar.edu/solutions/products/camels).

### 3.8 Ưu điểm và hạn chế

**Ưu điểm:** là benchmark phổ biến nhất cho mô hình mưa–dòng chảy dùng học sâu; có nhiều công trình SSM, Transformer và foundation model để đối chiếu; lưu lượng là số đo thực tế chất lượng cao.

**Hạn chế:** không có topology mạng lưới sông; bản gốc chỉ có độ phân giải ngày (bản theo giờ là bản mở rộng cho 516 lưu vực); ít biến khí tượng (5); lưu vực ít chịu tác động của con người; dung lượng lớn.

---

## 4. Columbia Basin (CBR/DART)

### 4.1 Bản chất dữ liệu

Đây **không phải bộ dữ liệu đóng gói** mà là **cổng dữ liệu vận hành DART** (Data Access in Real Time) do **Columbia Basin Research, Đại học Washington** duy trì. Người dùng tự truy vấn rồi tải về dạng CSV, nên phải tự xây dựng quy trình lấy và làm sạch dữ liệu.

### 4.2 Nguồn dữ liệu

| Cơ quan | Đóng góp |
|---|---|
| U.S. Army Corps of Engineers (USACE) | Dữ liệu vận hành đập |
| U.S. Geological Survey (USGS) | Lưu lượng tại trạm sông |
| NOAA | Dữ liệu biển, dự báo nguồn nước |
| Các đơn vị điện lực địa phương (PUD) | Dữ liệu từng đập |

### 4.3 Biến thủy văn

| Biến | Ý nghĩa | Đơn vị |
|---|---|---|
| Inflow / Outflow | Lưu lượng vào/ra đập, trung bình 24 giờ | kcfs |
| Spill / Spill percent | Lưu lượng xả tràn và tỷ lệ xả tràn | kcfs / % |
| Elevation | Cao trình mực nước hồ | feet |
| Temperature | Nhiệt độ nước | °C |
| Dissolved Gas | Khí hòa tan | mmHg / % |
| Turbidity | Độ đục | feet |
| Streamflow | Lưu lượng tại trạm USGS | cfs |
| Stage | Mực nước | ft |

Điểm riêng của bộ dữ liệu này là có **dữ liệu vận hành hồ chứa thực tế** (lưu lượng ra, xả tràn, cao trình hồ).

### 4.4 Thời gian

| Loại | Thời gian | Độ phân giải |
|---|---|---|
| Lưu lượng | 1878 – nay | Ngày |
| Dữ liệu vận hành đập | Chủ yếu từ thập niên 1960 | Ngày |
| Chất lượng nước | 1995 – nay | Giờ |

### 4.5 Trạm và đập chính

| Sông | Đập |
|---|---|
| Columbia | Bonneville, The Dalles, John Day, McNary, Priest Rapids, Wanapum, Rock Island, Rocky Reach, Wells, Grand Coulee |
| Snake | Ice Harbor, Lower Monumental, Little Goose, Lower Granite |
| Nhánh khác | Dworshak, Libby |

### 4.6 Topology

Quan hệ thượng–hạ nguồn tồn tại trong thực tế, ví dụ trên sông Snake theo thứ tự Lower Granite → Little Goose → Lower Monumental → Ice Harbor. Tuy nhiên hệ thống **không cung cấp tệp topology**, nên muốn xây dựng đồ thị phải tự dựng từ vị trí trạm và thứ tự các đập.

### 4.7 Truy cập

Có công cụ truy vấn trên web, xuất bảng HTML/CSV, và có API đơn giản để lấy dữ liệu tự động.

### 4.8 Ưu điểm và hạn chế

**Ưu điểm:** có dữ liệu vận hành hồ chứa; có dữ liệu chất lượng nước theo giờ (lưu lượng và vận hành đập công bố theo ngày); chuỗi lưu lượng rất dài; miễn phí.

**Hạn chế:** phải tự xử lý nhiều; không có forcing khí tượng và thuộc tính lưu vực; không có tệp topology; ít công trình để đối chiếu.

---

## 5. WaterBench-Iowa

### 5.1 Thông tin xuất bản

| Mục | Nội dung |
|---|---|
| Tên đầy đủ | WaterBench-Iowa: a large-scale benchmark dataset for data-driven streamflow forecasting |
| Tác giả | Ibrahim Demir, Zhongrun Xiang, Bekir Demiray, Muhammed Sit |
| Tạp chí | Earth System Science Data (ESSD), 2022, tập 14, tr. 5605–5616 |
| DOI | [10.5194/essd-14-5605-2022](https://doi.org/10.5194/essd-14-5605-2022) |
| Mã nguồn và dữ liệu | [github.com/uihilab/WaterBench](https://github.com/uihilab/WaterBench) |

### 5.2 Đặc điểm

Đây là bộ dữ liệu **được thiết kế ngay từ đầu cho học máy** theo nguyên tắc FAIR (Findability, Accessibility, Interoperability, Reuse). Bài báo gốc đã cung cấp sẵn kết quả của một số kiến trúc học sâu làm mốc so sánh.

### 5.3 Phạm vi và nội dung

| Mục | Nội dung |
|---|---|
| Số lưu vực | 125 lưu vực tại bang Iowa |
| Thời gian | 1/10/2011 – 30/9/2018 |
| Độ phân giải | Giờ |
| Đặc trưng | 7 đặc trưng, trong đó có giáng thủy, lưu lượng, bốc thoát hơi, diện tích lưu vực, độ dốc, loại đất |
| Nguồn | NASA, NOAA, USGS, Iowa Flood Center |
| Bài toán chuẩn | Dự báo lưu lượng theo giờ cho 120 giờ tới |

### 5.4 Kết quả mốc trong bài báo gốc

Bài báo gốc đánh giá hồi quy tuyến tính, LSTM, GRU và Seq2Seq cho bài toán dự báo 120 giờ trên 125 lưu vực, với kết quả được báo cáo là **NSE trung vị 0,74** và **KGE trung vị 0,79**.

### 5.5 Ưu điểm và hạn chế

**Ưu điểm:** là bộ dữ liệu duy nhất trong 4 bộ có đủ công trình Mamba, Transformer, mô hình đồ thị và XAI để đối chiếu (công trình Mamba + XAI hiện là preprint); có mã nguồn công khai cho cả dữ liệu lẫn mô hình đồ thị (HydroGAT); độ phân giải giờ; tải trực tiếp từ GitHub (repo khoảng 7,6 GB).

**Hạn chế:** chuỗi thời gian ngắn (7 năm); ít đặc trưng (7); chỉ gồm một bang với địa hình tương đối đồng nhất.

---

## 6. Công trình huấn luyện mô hình trên từng bộ dữ liệu

**Tiêu chí đưa vào:** chỉ gồm các công trình **có huấn luyện mô hình trên chính bộ dữ liệu tương ứng**. Các công trình chỉ công bố dữ liệu, chỉ dùng dữ liệu làm số liệu quan trắc phụ, hoặc chưa xác nhận được việc huấn luyện đều không được liệt kê.

**Ký hiệu:** cột "CNTT/AI" được đánh dấu ✅ nếu công trình đăng ở venue thuộc Computer Science/Artificial Intelligence hoặc venue IEEE/ACM. Hạng Q có dấu \* cần đối chiếu SCImago (xem Mục 8).

### 6.0 Danh sách tóm tắt (19 công trình)

| # | Công trình | Bộ dữ liệu | Mô hình | Venue | Năm | Hạng | CNTT/AI | Trích dẫn |
|---|---|---|---|---|---|---|---|---|
| 1 | The Merit of River Network Topology for Neural Flood Forecasting | LamaH-CE | GNN | **ICML** | 2024 | **CORE A\*** | ✅ | 1 (bản arXiv) |
| 2 | A GNN routing module is all you need for LSTM rainfall–runoff models | LamaH-CE | LSTM + GNN | HESS | 2026 | Q1\* | ❌ | 4 |
| 3 | Global Daily Discharge Estimation Based on Grid LSTM and River Routing | LamaH-CE (một phần) | Grid LSTM | Water Resources Research | 2025 | Q1\* | ❌ | 49 |
| 4 | BiasCast | LamaH-CE | LSTM | HESS | 2026 | Q1\* | ❌ | 3 (bản preprint) |
| 5 | A Deep State Space Model for Rainfall-Runoff Simulations | CAMELS-US | **S4D-FT** | Water Resources Research | 2025 | Q1\* | ❌ | 9 |
| 6 | HydroDiffusion | CAMELS-US | Diffusion + **S4D-FT** | arXiv | 2025 | Preprint | — | — |
| 7 | Benchmarking structured state space models for differentiable parameter learning | CAMELS-US | **S4D, S5D** | **Expert Systems with Applications** | 2026 | Q1\* | ✅ | 0 |
| 8 | Temporal inductive biases in hourly flood forecasting | CAMELS-US | LSTM, PatchTST, **Mamba** | Journal of Hydrology | 2026 | Q1\* | ❌ | 1 |
| 9 | Zero-shot forecasting of streamflow using time series foundation models | CAMELS-US | MOIRAI, Chronos, TTM, Sundial | Machine Learning: Earth | 2026 | Chưa xếp hạng | — | 1 |
| 10 | Efficacy of temporal fusion transformers for runoff simulation | CAMELS-US | TFT, LSTM | arXiv | 2025 | Preprint | — | 2 |
| 11 | Multi-Task Learning as a Step Toward General-Purpose Hydrological Forecasting | CAMELS-US (qua Caravan) | Transformer, **Mamba** | EarthArXiv | 2025 | Preprint | — | — |
| 12 | Testing discharge assimilation strategies | CAMELS-US | MLP (LSTM đã huấn luyện sẵn) | HESS | 2026 | Q1\* | ❌ | — |
| 13 | Is smart sampling worth it? | CAMELS-US | LSTM | Hydrology Research | 2026 | \* | ❌ | — |
| 14 | Assessing the transferability of LSTM-based streamflow models | CAMELS-US (nguồn transfer) | LSTM | Journal of Hydrology: Regional Studies | 2026 | \* | ❌ | — |
| 15 | Comparative Analysis of LSTM and Random Forest | CAMELS-US (16 lưu vực) | LSTM, Random Forest | Water (MDPI) | 2026 | \* | ❌ | — |
| 16 | **ResBi-Mamba Plus** | Columbia Basin | Bi-Mamba + attention | **AI Science and Engineering (IEEE)** | 2026 | Chưa xếp hạng | ✅ | 0 |
| 17 | **HydroGAT** | WaterBench-Iowa | Graph attention + Transformer | **ACM SIGSPATIAL** | 2025 | **CORE A** | ✅ | 1 |
| 18 | Advancing Long-Horizon Hydrological Forecasting: A Mamba-based Approach with XAI | WaterBench-Iowa | **Mamba** | SSRN / EarthArXiv | 2025 | Preprint | — | 1 (bản SSRN) |
| 19 | Towards generalized hydrological forecasting using transformer models for 120 h | WaterBench-Iowa | Transformer | Machine Learning: Earth | 2026 | Chưa xếp hạng | — | — |

---

### 6.1 LamaH-CE (4 công trình)

#### 6.1.1 The Merit of River Network Topology for Neural Flood Forecasting

| Mục | Nội dung |
|---|---|
| Tác giả | Nikolas Kirschstein, Yixuan Sun |
| Venue | **ICML 2024**, main track, PMLR tập 235, tr. 24713–24725 (**CORE A\***) |
| Link | [proceedings.mlr.press/v235/kirschstein24a.html](https://proceedings.mlr.press/v235/kirschstein24a.html) · [arXiv 2405.19836](https://arxiv.org/abs/2405.19836) |
| Mã nguồn | ✅ [github.com/nkirschi/neural-flood-forecasting](https://github.com/nkirschi/neural-flood-forecasting) |
| Dữ liệu | LamaH-CE **theo giờ, 2000–2017**. Từ 859 trạm → **608 trạm** thuộc thành phần liên thông lớn nhất ("Danube A") → **358 trạm** sau khi bỏ trạm có khoảng thiếu dữ liệu dài hơn 6 giờ hoặc không phủ đủ 2000–2017. Lặp lại thí nghiệm trên **4 mạng con nhỏ** để loại trừ khả năng do độ sâu mạng |
| Đầu vào mỗi trạm | Cửa sổ **24 giờ** gồm lưu lượng và 4 biến khí tượng: giáng thủy, độ ẩm đất tầng mặt, nhiệt độ không khí, áp suất bề mặt; chuẩn hóa z-score theo từng trạm |
| Đầu ra | Lưu lượng tại mỗi trạm sau **6 giờ** |
| Mô hình | Kiến trúc "sandwich": **Encoder** (1 lớp affine đưa cửa sổ đầu vào lên không gian ẩn 128 chiều) → **19 lớp GNN** + ReLU (19 = độ dài đường đi dài nhất trong đồ thị) → **Decoder** (1 lớp affine ra 1 giá trị/trạm). Thử 3 loại lớp GNN: **ResGCN, GCNII, ResGAT**. Baseline: MLP 19 lớp không dùng đồ thị |
| Định nghĩa cạnh | 4 nhóm: **cô lập** (không cạnh), **nhị phân**, **trọng số vật lý** (chiều dài dòng, chênh lệch độ cao, độ dốc trung bình), **học được**. Mỗi loại thử 3 hướng: **xuôi dòng**, **ngược dòng**, **hai chiều** |
| Huấn luyện | Hàm mất mát: sai số bình phương **có trọng số "relevancy"** — ưu tiên cửa sổ có tốc độ thay đổi lưu lượng nhanh (trọng số gấp đôi) và lưu lượng cao so với trung bình. Adam, learning rate 10⁻⁴, L2 10⁻⁵ |
| Chia dữ liệu | Test luôn là **2016–2017**. Train 8 năm trong 2000–2015 theo 3 fold: các năm chẵn / các năm lẻ / 2008–2015 liên tục |
| Kết quả (GCNII, NSE) | Cô lập: 84,12% ± 1,88<br>Nhị phân: xuôi 84,09% ± 1,11 · ngược 85,16% ± 1,74 · hai chiều 84,81% ± 0,53<br>Học được: xuôi 84,91% ± 1,97 · ngược 85,00% ± 2,11 · **hai chiều 85,56% ± 1,41** (tốt nhất)<br>**MLP không đồ thị: 85,37% ± 1,64** |
| Kết quả trọng số vật lý (xuôi dòng) | ResGCN: chiều dài dòng 81,64% ± 1,45 · chênh lệch độ cao 82,16% ± 1,85<br>ResGAT: chiều dài dòng 80,21% ± 4,85 · chênh lệch độ cao 80,58% ± 5,00 |
| Kết luận | Mọi cấu hình đồ thị đều **không vượt rõ** MLP không dùng đồ thị; chênh lệch nằm trong khoảng dao động giữa các fold |
| Kết luận | *"the model fails to benefit from the river network topology information, both on the entire network and small subgraphs"*. Trọng số cạnh học được không tương quan với các định nghĩa tĩnh; GNN gặp khó khi dự báo **đỉnh lưu lượng đột ngột và hẹp** |
| Ý nghĩa với đề tài | Là bằng chứng ở venue hạng cao cho thấy đồ thị mạng lưới sông **dạng thô** không tự động cải thiện dự báo. Cần tính đến khi thiết kế mô hình kết hợp đồ thị (xem Mục 7.2) |

#### 6.1.2 A GNN routing module is all you need for LSTM rainfall–runoff models

| Mục | Nội dung |
|---|---|
| Tác giả | Hamidreza Mosaffa, Florian Pappenberger, Christel Prudhomme, Matthew Chantry, Christoph Rüdiger, Hannah Cloke |
| Venue | Hydrology and Earth System Sciences, 2026 |
| Link | [hess.copernicus.org/articles/30/2079/2026](https://hess.copernicus.org/articles/30/2079/2026/) |
| Dữ liệu | LamaH-CE, thượng lưu Danube, **530 lưu vực con**, 1/1/1987 – 31/12/2017 |
| Chia dữ liệu | Train 70% / Validation 15% / Test 15% (theo thời gian) |
| Mô hình | LSTM sinh dòng chảy cục bộ + GNN định tuyến dòng chảy; so sánh GCN, GAT, GraphSAGE, ChebNet |
| Chỉ số | NSE, KGE, CC, RMSE |
| Kết quả | **LSTM-GAT: NSE 0,61 · KGE 0,65 · CC 0,84 · RMSE 13,77 m³/s**<br>LSTM-GraphSAGE: NSE 0,60 · KGE 0,60<br>LSTM-ChebNet: NSE 0,59 · KGE 0,58<br>LSTM-GCN: NSE 0,48 · KGE 0,50<br>**LSTM: NSE 0,46 · KGE 0,49 · RMSE 21,24 m³/s** |
| Ý nghĩa với đề tài | Mô hình đồ thị có cơ chế attention (GAT) giảm RMSE khoảng 35% so với LSTM, cải thiện rõ nhất ở trạm hạ nguồn có mức kết nối cao. Đây là đối trọng với công trình 6.1.1 trên cùng bộ dữ liệu |

#### 6.1.3 Global Daily Discharge Estimation Based on Grid LSTM and River Routing

| Mục | Nội dung |
|---|---|
| Tác giả | Yang et al. |
| Venue | Water Resources Research, 2025 |
| DOI | [10.1029/2024WR039764](https://doi.org/10.1029/2024WR039764) |
| Dữ liệu | Tập lưu vực huấn luyện toàn cầu, trong đó có LamaH (*"Training basins included data from multiple sources including LamaH"*) |
| Mô hình | Grid LSTM kết hợp mô hình định tuyến RAPID |
| Ý nghĩa với đề tài | Minh chứng cho hướng mô hình hóa trên lưới kết hợp định tuyến. Không dùng làm mốc so sánh vì không có kết quả riêng cho LamaH-CE |

#### 6.1.4 BiasCast: learning and adjusting real time biases from meteorological forecasts to enhance runoff predictions

| Mục | Nội dung |
|---|---|
| Tác giả | Oliver Konold, Moritz Feigl, Patrick Podest, Christoph Klingler, Karsten Schulz |
| Venue | Hydrology and Earth System Sciences, 2026 |
| Link | [hess.copernicus.org/articles/30/5067/2026](https://hess.copernicus.org/articles/30/5067/2026/) |
| Dữ liệu | Extended LamaH-CE, **451 lưu vực** ít chịu tác động của con người |
| Chia dữ liệu | Train 2003–2009 / Validation 2010–2013 / Test 2014–2017 |
| Mô hình | Sequential Forecast LSTM, Encoder-Decoder LSTM, transfer learning với mạng embedding riêng cho đầu vào tĩnh và động |
| Chỉ số | NSE trung bình theo lưu vực |
| Kết quả | Khi chuyển từ dữ liệu tái phân tích sang dữ liệu dự báo khí tượng thực, **NSE trung vị giảm từ 0,58 xuống 0,33**. Sequential Forecast LSTM đạt NSE trung vị **0,71** khi có thêm lưu lượng quan trắc |
| Ý nghĩa với đề tài | Định lượng mức suy giảm khi mô hình huấn luyện bằng dữ liệu tái phân tích được chạy với dữ liệu dự báo thực. Đây là hạn chế cần nêu trong báo cáo |

---

### 6.2 CAMELS-US (11 công trình)

#### 6.2.1 A Deep State Space Model for Rainfall-Runoff Simulations

| Mục | Nội dung |
|---|---|
| Tác giả | Yihan Wang, Lujun Zhang, Annan Yu, N. Benjamin Erichson, Tiantian Yang |
| Venue | Water Resources Research, 2025 |
| DOI | [10.1029/2025WR039888](https://doi.org/10.1029/2025WR039888) · [arXiv 2501.14980](https://arxiv.org/abs/2501.14980) |
| Dữ liệu | 531 lưu vực tại Hoa Kỳ lục địa |
| Mô hình | **S4D-FT** (Frequency Tuned Diagonal state space model) |
| Baseline | LSTM, Sacramento Soil Moisture Accounting (SAC-SMA) |
| Kết quả | S4D-FT **vượt LSTM**, rõ nhất ở lưu vực chịu chi phối của tuyết tan hoặc dòng chảy gián đoạn; **kém hơn ở chế độ dòng chảy biến động nhanh, biên độ lớn** |
| Ý nghĩa với đề tài | Bằng chứng mô hình họ SSM có thể vượt LSTM. Đồng thời cho thấy điểm yếu ở đúng loại dòng chảy liên quan tới lũ (xem Mục 7.3) |

#### 6.2.2 HydroDiffusion: Diffusion-Based Probabilistic Streamflow Forecasting with a State Space Backbone

| Mục | Nội dung |
|---|---|
| Tác giả | Yihan Wang, Annan Yu, Lujun Zhang, Charuleka Varadharajan, N. Benjamin Erichson |
| Venue | arXiv (preprint) |
| Link | [arXiv 2512.12183](https://arxiv.org/abs/2512.12183) |
| Dữ liệu | CAMELS, 531 lưu vực, theo ngày |
| Mô hình | Diffusion model với backbone **S4D-FT** |
| Cấu hình | Cửa sổ quá khứ 365 ngày → 8 ngày đầu ra (ngày hiện tại + 7 ngày dự báo) |
| Chia dữ liệu | Train 10/1980–9/1990 / Validation 1990–1995 / Test 1995–2005 |
| Baseline | Diffusion model dùng backbone LSTM, DRUM |
| Kết quả | Vượt DRUM ở cả dự báo tất định và dự báo xác suất |
| Ý nghĩa với đề tài | Mở ra hướng **dự báo xác suất** dùng SSM, phù hợp với nhu cầu cảnh báo lũ (cần khoảng tin cậy) |

#### 6.2.3 Benchmarking structured state space models for differentiable parameter learning: A large-scale evaluation on accuracy and physical consistency

| Mục | Nội dung |
|---|---|
| Tác giả | Xin Jing, Jungang Luo, Xue Yang, Ganggang Zuo |
| Venue | **Expert Systems with Applications**, tập 329, bài 133040, 2026 |
| DOI | [10.1016/j.eswa.2026.133040](https://doi.org/10.1016/j.eswa.2026.133040) |
| Dữ liệu | 671 lưu vực CAMELS-US |
| Mô hình | **S4D, S5D** trong khung differentiable parameter learning |
| Kết quả | NSE: **0,742** (baseline) → **0,756** (S4D) → **0,763** (S5D) |
| Ý nghĩa với đề tài | Công trình SSM ở tạp chí thuộc lĩnh vực AI. Các giá trị NSE dùng làm mốc kiểm tra khi cài đặt mô hình |

#### 6.2.4 Temporal inductive biases in hourly flood forecasting: a comparative analysis of recurrent, attention-based, and state-space neural networks

| Mục | Nội dung |
|---|---|
| Tác giả | Binlan Zhang, Chaojun Ouyang, Qingsong Xu, Yanlin Liu, Zi Wang, Quan Tang |
| Venue | Journal of Hydrology, 2026 |
| DOI | [10.1016/j.jhydrol.2026.135727](https://doi.org/10.1016/j.jhydrol.2026.135727) |
| Dữ liệu | 516 lưu vực CAMELS-US, theo giờ (khớp với bản CAMELS-US theo giờ của Gauch et al., 2021, xem Mục 3.3) |
| Mô hình | LSTM, GRU, Transformer, Informer, Autoformer, RRformer, PatchTST, DLinear, Mamba (S-Mamba); chuỗi vào 96 giờ, dự báo 1–24 giờ, có và không có Q quan trắc |
| Kết quả | Lead 1 giờ: DLinear tốt nhất ở 76,7% lưu vực. Lead 3–24 giờ: **PatchTST tốt nhất** (53–91% lưu vực), **LSTM đứng thứ hai**; Mamba được đánh giá cân bằng giữa độ chính xác và chi phí; các biến thể Transformer khác kém cả LSTM và Mamba. Khi không có Q quan trắc (chỉ khí tượng), lợi thế của PatchTST giảm, LSTM và Mamba bền hơn. "Mamba" ở bài này là S-Mamba, trích tương quan giữa các **biến** trước khối Mamba |
| Ý nghĩa với đề tài | Mamba thuần có thể kém PatchTST và LSTM ở bài toán có Q quan trắc; PatchTST là ứng viên biến thể Transformer (bước E). Chuỗi vào chỉ 96 giờ, ngắn hơn nhiều so với hindcast 364 ngày của BiasCast |

#### 6.2.5 Zero-shot forecasting of streamflow using time series foundation models: are we there yet?

| Mục | Nội dung |
|---|---|
| Tác giả | Alexander Y. Sun, Albert A. Sun |
| Venue | Machine Learning: Earth (IOP Publishing), 2026 |
| DOI | [10.1088/3049-4753/ae4982](https://iopscience.iop.org/article/10.1088/3049-4753/ae4982) |
| Dữ liệu | 531 lưu vực CAMELS (diện tích < 2.000 km²); 1980–2010 theo ngày, 2006–2020 theo 3 giờ |
| Mô hình | MOIRAI, Chronos, TTM, Sundial (zero-shot); **fine-tune** Chronos và TTM; LSTM làm baseline |
| Chỉ số | NSE, KGE, FHV (sai lệch đỉnh lũ), FLV (sai lệch dòng chảy kiệt) |
| Kết quả | Zero-shot theo ngày: Sundial NSE trung vị **0,564**, LSTM **0,593**<br>Zero-shot theo 3 giờ: Sundial NSE trung vị **0,967**<br>Sau fine-tune: TTM NSE trung vị **0,753** (có thuộc tính tĩnh) |
| Ý nghĩa với đề tài | Foundation model sau fine-tune đạt mức tương đương SSM. FHV và FLV là các chỉ số nên dùng thêm để đánh giá riêng đỉnh lũ và dòng chảy kiệt |

#### 6.2.6 Efficacy of temporal fusion transformers for runoff simulation

| Mục | Nội dung |
|---|---|
| Tác giả | S. R. Koya, T. Roy |
| Venue | arXiv (preprint), 2025 |
| Link | [arXiv 2506.20831](https://arxiv.org/abs/2506.20831) |
| Dữ liệu | 531 lưu vực CAMELS-US và 5 tập con Caravan (Mỹ, Úc, Brazil, Anh, Chile) |
| Mô hình | Temporal Fusion Transformer, LSTM; mỗi kiến trúc huấn luyện **10 lần với khởi tạo ngẫu nhiên khác nhau** |
| Ý nghĩa với đề tài | Gợi ý về phương pháp: huấn luyện nhiều lần và báo cáo phân phối kết quả thay vì một lần chạy |

#### 6.2.7 Multi-Task Learning as a Step Toward Building General-Purpose Hydrological Forecasting Systems

| Mục | Nội dung |
|---|---|
| Tác giả | Bekir Zahit Demiray, Ibrahim Demir |
| Venue | EarthArXiv (preprint), 10/2025 |
| Link | [eartharxiv.org/repository/view/10562](https://eartharxiv.org/repository/view/10562/) |
| Dữ liệu | CAMELS trong kho Caravan, hơn 600 lưu vực tại Mỹ |
| Mô hình | Hai kiến trúc đa nhiệm: Transformer và **Mamba** |
| Bài toán | Dự báo đồng thời lưu lượng và độ ẩm đất, hạn 7 ngày |
| Kết quả | Mô hình đa nhiệm đạt độ chính xác tương đương hoặc cao hơn một chút so với mô hình đơn nhiệm |

#### 6.2.8 Testing discharge assimilation strategies to enhance short-range AI-based operational rainfall–runoff forecasts

| Mục | Nội dung |
|---|---|
| Tác giả | B. E. Saint-Fleur, E. Gaume, F. Surmont, N. Akil, D. Theriez |
| Venue | Hydrology and Earth System Sciences, tập 30, tr. 3497–3527, 2026 |
| Link | [hess.copernicus.org/articles/30/3497/2026](https://hess.copernicus.org/articles/30/3497/2026/) |
| Dữ liệu | CAMELS-US và CAMELS-FR |
| Mô hình | **MLP** điều phối 3 chiến lược đồng hóa lưu lượng (huấn luyện 20 lần với các seed khác nhau). Mô hình nền là LSTM **đã huấn luyện sẵn** của Kratzert et al. (2019) và SAC-SMA |
| Ý nghĩa với đề tài | Nhóm tác giả chỉ ra phần lớn nghiên cứu học sâu cho mô hình mưa–dòng chảy chưa tích hợp bước đồng hóa lưu lượng, trong khi bước này thường giúp tăng độ chính xác dự báo hạn ngắn trong vận hành. Là hướng mở rộng khi đưa mô hình vào ứng dụng thực tế |

#### 6.2.9 Is smart sampling worth it? Impact of training data selection on the performance of LSTMs in streamflow prediction

| Mục | Nội dung |
|---|---|
| Tác giả | Benedikt Heudorfer, Ralf Loritz |
| Venue | Hydrology Research, tập 57, số 3, tr. 501–517, 3/2026 |
| Link | [iwaponline.com/hr/article/57/3/501/111226](https://iwaponline.com/hr/article/57/3/501/111226) |
| Dữ liệu | CAMELS-US, cắt giảm dữ liệu huấn luyện theo 4 nhóm chiến lược (cực trị thủy văn, độ hiếm sự kiện, ngữ cảnh thời gian, tính đại diện không gian) |
| Mô hình | LSTM |
| Kết quả | **Lấy mẫu ngẫu nhiên cho kết quả tốt hơn các chiến lược lấy mẫu có chủ đích** |
| Lưu ý | Tác giả nêu kết luận có thể không áp dụng cho bối cảnh một lưu vực hoặc chuỗi dữ liệu ngắn |

#### 6.2.10 Assessing the transferability of LSTM-based streamflow models under varying source basin diversity and target data availability (Mangla Basin, Pakistan)

| Mục | Nội dung |
|---|---|
| Tác giả | M. Adnan, W. Ouyang, L. Ye, M. A. Khan, Y. Chai, H. Ma |
| Venue | Journal of Hydrology: Regional Studies, 4/2026 |
| Link | [sciencedirect.com/science/article/pii/S2214581826002272](https://www.sciencedirect.com/science/article/pii/S2214581826002272) |
| Dữ liệu | CAMELS-US và Caravan làm **dữ liệu huấn luyện nguồn**, chuyển giao sang lưu vực Mangla (Pakistan) thiếu dữ liệu |
| Mô hình | LSTM với transfer learning |
| Kết quả | Với 80% độ dài dữ liệu huấn luyện, NSE validation đạt **0,89** (nguồn CAMELS-US) và **0,87** (nguồn Caravan), cao hơn mô hình chỉ huấn luyện tại chỗ. Lợi thế giảm dần khi dữ liệu địa phương tăng |
| Ý nghĩa với đề tài | Minh chứng cho transfer learning: huấn luyện trên bộ dữ liệu lớn rồi chuyển giao sang lưu vực thiếu dữ liệu |

#### 6.2.11 Comparative Analysis of LSTM and Random Forest Algorithms for Streamflow Prediction: A Case Study of Diverse River Basins in the United States

| Mục | Nội dung |
|---|---|
| Tác giả | A. Dula Shanko, A. Melesse |
| Venue | Water (MDPI), tập 18, số 14, bài 1768, 7/2026 |
| Link | [mdpi.com/2073-4441/18/14/1768](https://www.mdpi.com/2073-4441/18/14/1768) |
| Dữ liệu | **16 lưu vực** CAMELS-US |
| Đầu vào | 5 biến khí hậu + lưu lượng trễ 7, 14, 30 ngày |
| Mô hình | LSTM, Random Forest |
| Kết quả | **Random Forest NSE trung bình 0,755; LSTM 0,632**. Cả hai tốt ở lưu vực tuyết tan, kém nhất ở vùng ẩm có dòng chảy biến động nhanh |
| Ý nghĩa với đề tài | Ở quy mô ít lưu vực, mô hình học máy cổ điển vẫn có thể vượt học sâu, nên cần đưa Random Forest hoặc XGBoost vào nhóm baseline |

---

### 6.3 Columbia Basin (1 công trình)

#### 6.3.1 ResBi-Mamba Plus: Deep Bidirectional Mamba with Spatiotemporal Attention for Robust Interpretable Hourly Runoff Forecasting

| Mục | Nội dung |
|---|---|
| Tác giả | Ziyu Sheng, Shiping Wen, Zhongkai Feng, Kaibo Shi |
| Venue | **Artificial Intelligence Science and Engineering** (IEEE Xplore), 2026 |
| DOI | [10.23919/AISE.2026.000010](https://doi.org/10.23919/AISE.2026.000010) |
| Dữ liệu | *"hourly runoff dataset from the Columbia River"* |
| Mô hình | **Bi-Mamba** (Mamba hai chiều) + **Spatiotemporal Attention** + **ResNet Plus** (hai nhánh residual song song và kết nối tắt dày đặc) |
| Kết quả | Vượt các baseline trong thí nghiệm ablation và thí nghiệm so sánh |
| Ý nghĩa với đề tài | Là công trình Mamba lai cho dự báo dòng chảy ở venue CNTT, cho thấy Mamba kết hợp attention và residual đạt kết quả tốt |

---

### 6.4 WaterBench-Iowa (3 công trình)

#### 6.4.1 HydroGAT: Distributed Heterogeneous Graph Attention Transformer for Spatiotemporal Flood Prediction

| Mục | Nội dung |
|---|---|
| Tác giả | Aishwarya Sarkar, Autrin Hakimi, Xiaoqiong Chen, Hai Huang, Chaoqun Lu, Ibrahim Demir, Ali Jannesari |
| Venue | **ACM SIGSPATIAL 2025** (33rd International Conference on Advances in Geographic Information Systems), **CORE A** |
| DOI | [10.1145/3748636.3764172](https://doi.org/10.1145/3748636.3764172) · [arXiv 2509.02481](https://arxiv.org/abs/2509.02481) |
| Mã nguồn | [github.com/swapp-lab/HydroGAT](https://github.com/swapp-lab/HydroGAT) |
| Dữ liệu | Lưu vực Cedar River (CRB, 18 trạm) và Des Moines River (DSMRB, 33 trạm); lưu lượng theo giờ và quan hệ lưu vực từ WaterBench; giáng thủy NCEP/EMC Stage IV; DEM từ USGS 3D Elevation Program. **Dữ liệu đã xử lý không được công bố**, phải tự xây dựng lại từ các nguồn gốc |
| Khác biệt với benchmark gốc | Thiết lập **khác** benchmark của WaterBench-Iowa (125 lưu vực, dự báo 120 giờ): HydroGAT dùng 2 lưu vực, cửa sổ 72 giờ, chỉ tháng 5–9 → không so sánh trực tiếp với NSE 0,74 của benchmark gốc |
| Thời gian | Tháng 5–9 các năm 2012–2018, theo giờ |
| Chia dữ liệu | Train 2012–2016 / Validation 2017 / Test 2017–2018 |
| Đầu vào | Giáng thủy và lưu lượng lịch sử tại các trạm có đo |
| Mô hình | Đồ thị lưu vực không đồng nhất (mỗi ô lưới đất và ô lưới sông là một nút) + graph attention + Transformer; cửa sổ 72 giờ vào → 72 giờ ra; 32 hidden features, 2 attention heads |
| Baseline | DCRNN, GraphWaveNet, RGCN, GCRNN, STGCN-WAVE |
| Kết quả | **CRB: NSE 0,97 · KGE 0,96 · PBIAS −1,10%**<br>**DSMRB: NSE 0,97 · KGE 0,92 · PBIAS +4,70%** |
| Hạ tầng | Siêu máy tính **NERSC Perlmutter** (mỗi node 4× NVIDIA A100); mở rộng từ 4 lên 64 GPU tăng tốc tới 15 lần, rút ngắn từ nhiều ngày trên 1 GPU xuống vài giờ |
| Ý nghĩa với đề tài | Kiến trúc đồ thị + Transformer ở venue CNTT hạng A, có mã nguồn công khai. Có thể dùng làm nền để thay khối Transformer bằng Mamba. Kết quả rất cao một phần vì chỉ đánh giá mùa mưa và dùng lưu lượng lịch sử làm đầu vào, nên so sánh cần cùng điều kiện |

#### 6.4.2 Advancing Long-Horizon Hydrological Forecasting: A Mamba-based Approach with Explainable AI for Generalized Streamflow Prediction

| Mục | Nội dung |
|---|---|
| Tác giả | Bekir Zahit Demiray, Ibrahim Demir |
| Venue | **Preprint**: SSRN ([10.2139/ssrn.5638262](https://doi.org/10.2139/ssrn.5638262)) và EarthArXiv ([10.31223/X5B164](https://doi.org/10.31223/X5B164)), 2025 |
| Dữ liệu | WaterBench-Iowa, 125 lưu vực |
| Bài toán | 72 giờ đầu vào → dự báo 120 giờ |
| Mô hình | **Mamba** |
| Baseline | Persistence, LSTM, GRU, Seq2Seq, Transformer |
| Chỉ số | NSE, KGE, Pearson's r, NRMSE |
| Kết quả | Mamba đạt độ chính xác **tương đương, một số mặt nhỉnh hơn nhẹ** so với Transformer; cả hai vượt rõ các phương pháp còn lại |
| XAI | Dùng SHAP để phân tích mức đóng góp của đặc trưng đầu vào |
| Ý nghĩa với đề tài | Là công trình gần nhất với hướng Mamba + XAI. Hướng nghiên cứu mới cần có đóng góp khác biệt so với công trình này |

#### 6.4.3 Towards generalized hydrological forecasting using transformer models for 120 h streamflow prediction

| Mục | Nội dung |
|---|---|
| Tác giả | Bekir Z. Demiray, Ibrahim Demir |
| Venue | Machine Learning: Earth, 2026 (bản preprint: [arXiv 2406.07484](https://arxiv.org/abs/2406.07484)) |
| Dữ liệu | 125 lưu vực tại Iowa |
| Bài toán | 72 giờ đầu vào (giáng thủy, bốc thoát hơi, lưu lượng) → dự báo 120 giờ |
| Mô hình | Transformer; baseline LSTM, GRU, Seq2Seq, Persistence |
| Chỉ số | NSE, KGE, Pearson's r, NRMSE |
| Kết quả | Transformer có NSE và KGE trung vị cao nhất, NRMSE thấp nhất |

---

## 7. Phân tích tổng hợp

### 7.1 Bằng chứng về Mamba/SSM

| Công trình | Bộ dữ liệu | Kết quả |
|---|---|---|
| S4D-FT (WRR 2025) | CAMELS-US | SSM vượt LSTM, nhưng kém hơn ở dòng chảy biến động nhanh, biên độ lớn |
| Benchmarking SSM (ESWA 2026) | CAMELS-US | S4D, S5D vượt baseline |
| Temporal inductive biases (J. Hydrology 2026) | CAMELS-US | PatchTST tốt nhất, LSTM thứ hai; Mamba cân bằng độ chính xác – chi phí |
| ResBi-Mamba Plus (AISE 2026) | Columbia | Mamba lai vượt các baseline |
| Mamba + XAI (preprint 2025) | WaterBench-Iowa | Mamba tương đương Transformer |

**Nhận xét:** kết quả về Mamba/SSM chưa thống nhất. Mô hình thuần cho kết quả không ổn định giữa các nghiên cứu. Các biến thể có cải tiến (điều chỉnh tần số, hai chiều, attention, residual) cho kết quả tốt hơn. Vì vậy nên xây dựng đề tài theo hướng **nghiên cứu so sánh**, không mặc định Mamba sẽ vượt các kiến trúc khác.

### 7.2 Mô hình đồ thị mạng lưới sông có cải thiện dự báo không?

| Công trình | Bộ dữ liệu | Kết luận |
|---|---|---|
| Kirschstein & Sun (ICML 2024) | LamaH-CE | **Không cải thiện** |
| Mosaffa et al. (HESS 2026) | LamaH-CE | **Có cải thiện** (GAT: NSE 0,46 → 0,61) |
| HydroGAT (ACM SIGSPATIAL 2025) | WaterBench-Iowa | **Có cải thiện** (NSE 0,97) |

**Công trình giải thích mâu thuẫn** (không thuộc 4 bộ dữ liệu trên): *Accelerating flood warnings by 10 hours: the power of river network topology in AI-enhanced flood forecasting* — Hongjun Wang, Jiyuan Chen, Yinqiang Zheng, Xuan Song — npj Natural Hazards, 2025, DOI [10.1038/s44304-025-00083-6](https://doi.org/10.1038/s44304-025-00083-6).

- **Nguyên nhân:** mạng lưới sông có cấu trúc dạng cây, khoảng cách giữa các nút lớn, gây ra hiện tượng **over-squashing** (thông tin bị nén mất khi truyền qua nhiều bước), nên GNN không khai thác được quan hệ giữa các nút xa nhau
- **Cách khắc phục:** biến đổi đồ thị theo **khả năng tới được (reachability)** để làm dày kết nối
- **Kết quả:** dự báo mực nước 24 giờ đạt độ chính xác tương đương EA-LSTM ở 14 giờ, tức **kéo dài tầm dự báo thêm 10 giờ (71%)**

| Cách xây dựng đồ thị | Kết quả | Bằng chứng |
|---|---|---|
| Đồ thị thô (chỉ nối nút kề trực tiếp) | Không cải thiện | ICML 2024 |
| Đồ thị có attention học trọng số cạnh (GAT) | Cải thiện | HESS 2026 |
| Đồ thị làm dày theo reachability | Cải thiện rõ, đặc biệt với lũ cực đoan | npj Natural Hazards 2025 |
| Đồ thị mức ô lưới (pixel-level) | Cải thiện | ACM SIGSPATIAL 2025 |

**Nhận xét:** nếu kết hợp đồ thị vào mô hình, cần dùng một trong các cách đã có bằng chứng hiệu quả, không dùng đồ thị thô.

### 7.3 Điểm yếu chung tại đỉnh lũ

- S4D-FT kém hơn LSTM ở chế độ **dòng chảy biến động nhanh, biên độ lớn**
- GNN (ICML 2024) gặp khó với **đỉnh lưu lượng đột ngột và hẹp**

Cả SSM và GNN đều yếu ở loại sự kiện quan trọng nhất với cảnh báo lũ. Đề xuất:
- Đánh giá riêng đỉnh lũ bằng **FHV**, không chỉ dùng NSE tổng thể
- Dùng hàm mất mát có **trọng số theo chu kỳ lặp lại** (return period), như RiverMamba

### 7.4 Khoảng trống nghiên cứu

| Công trình | Cách xử lý thông tin không gian | Có GNN | Có Mamba |
|---|---|---|---|
| RiverMamba (NeurIPS 2025, DOI [10.52202/085713-4446](https://doi.org/10.52202/085713-4446)) | Tuần tự hóa không gian bằng space-filling curve | ❌ | ✅ |
| ResBi-Mamba Plus (AISE 2026) | Spatiotemporal attention | ❌ | ✅ |
| HydroGAT (ACM SIGSPATIAL 2025) | Graph attention | ✅ | ❌ (Transformer) |
| Mamba + XAI (preprint 2025) | Không xử lý không gian | ❌ | ✅ |

Trong phạm vi tra cứu, **chưa tìm thấy công trình nào kết hợp GNN với Mamba** cho dự báo lưu lượng dòng chảy.

Ở bài toán tổng quát, có kiến trúc kết hợp GNN + Mamba ở bài toán **tổng quát** — G-Mamba (Xiaojian Chen, Qiusheng Tang, *"Graph-enhanced Mamba: Efficient spatiotemporal sequence modeling with selective state space and graph neural networks"*, Neurocomputing vol 680, 2026, DOI [10.1016/j.neucom.2026.133280](https://doi.org/10.1016/j.neucom.2026.133280)): tiêm ngữ cảnh láng giềng từ đồ thị (tĩnh + động) vào bước cập nhật trạng thái của selective SSM, thử trên benchmark giao thông, điện, khí tượng, tỷ giá, nhiệt độ máy biến áp — **không có dữ liệu dòng chảy** (đã đọc trang abstract + phần mở đầu). Vì vậy nhận định trên vẫn đúng với riêng bài toán dòng chảy; G-Mamba dùng làm tham khảo kiến trúc.

### 7.5 Gợi ý phương pháp rút ra

| Gợi ý | Nguồn |
|---|---|
| Huấn luyện nhiều lần với seed khác nhau, báo cáo phân phối kết quả | Koya & Roy (TFT) |
| Dùng thêm FHV/FLV bên cạnh NSE, KGE | Sun & Sun (foundation models) |
| Nêu rõ mức suy giảm khi chuyển từ dữ liệu tái phân tích sang dữ liệu dự báo | BiasCast |
| Đưa mô hình học máy cổ điển (Random Forest/XGBoost) vào baseline | Shanko & Melesse |
| Cân nhắc transfer learning từ bộ dữ liệu lớn sang lưu vực thiếu dữ liệu | Adnan et al. |
| Đưa PatchTST vào baseline Transformer | Zhang et al. |

---

## 8. Lưu ý về số liệu

- **Hạng Q** có dấu \* được ghi theo tra cứu sơ bộ, **cần đối chiếu lại trên SCImago** trước khi dùng chính thức. Hạng CORE của ICML (A\*) và ACM SIGSPATIAL (A) đã được tra cứu.
- **Số trích dẫn** lấy từ Google Scholar hoặc OpenAlex tại thời điểm 13/9/2026. Hai nguồn có thể cho số khác nhau và số liệu thay đổi theo thời gian. Ô "—" là chưa tra cứu.
- **Đối chiếu số liệu với toàn văn:** NSE 0,742/0,756/0,763 (Jing et al., ESWA) đúng với bài, trong đó 0,742/0,756 trên 531 lưu vực và 0,763 (S5Dv2) trên 671 lưu vực; chạy 5 hạt giống trong repo cho NSE trung vị S4D 0,750, S5D 0,751, LSTM 0,742. Thứ tự PatchTST tốt nhất, LSTM thứ hai của Zhang et al. đúng với toàn văn (516 lưu vực CAMELS-US theo giờ). Chưa đối chiếu: cấu hình thí nghiệm chi tiết của ResBi-Mamba Plus; mô hình cụ thể đạt NSE 0,74 / KGE 0,79 trong benchmark gốc của WaterBench-Iowa.
- Công trình 6.4.3 không ghi tên "WaterBench" trong abstract. Việc xếp vào WaterBench-Iowa dựa trên việc công trình dùng đúng 125 lưu vực tại Iowa và có cùng nhóm tác giả với bộ dữ liệu.
- Các công trình **preprint** (6.2.2, 6.2.6, 6.2.7, 6.4.2) chưa qua bình duyệt, chỉ nên trích dẫn như công trình liên quan.
- Danh sách công trình phản ánh kết quả tra cứu tới ngày 13/9/2026, **không khẳng định đã bao quát toàn bộ**.

---

## 9. Thế mạnh của từng bộ dữ liệu

Đề tài dùng Extended LamaH-CE theo bài cơ sở BiasCast; ba bộ còn lại dùng cho phần khảo sát và công trình liên quan.

| Nhu cầu | Bộ dữ liệu phù hợp |
|---|---|
| Nhiều công trình SSM/Mamba để đối chiếu | CAMELS-US |
| Đủ công trình Mamba, Transformer, đồ thị và XAI để đối chiếu, có mã nguồn công khai | WaterBench-Iowa |
| Topology mạng lưới sông có sẵn, chuỗi dài, nhiều biến | LamaH-CE |
| Dữ liệu vận hành hồ chứa thực tế | Columbia Basin |
| Lưu lượng theo giờ | WaterBench-Iowa, LamaH-CE, CAMELS-US (bản mở rộng 516 lưu vực) |

**Ghi chú:**

Các bộ dữ liệu trên phục vụ hướng **tự xây dựng mô hình**. Chúng không thay thế trực tiếp dữ liệu đầu vào của RiverMamba, vốn yêu cầu 136 biến trên lưới 0,05°.

---

## 10. Nguồn tham khảo

**Bộ dữ liệu**
- LamaH-CE: [essd.copernicus.org/articles/13/4529/2021](https://essd.copernicus.org/articles/13/4529/2021/) · [zenodo.org/records/5153305](https://zenodo.org/records/5153305)
- CAMELS-US: [hess.copernicus.org/articles/21/5293/2017](https://hess.copernicus.org/articles/21/5293/2017/) · [hess.copernicus.org/articles/19/209/2015](https://hess.copernicus.org/articles/19/209/2015/) · [ral.ucar.edu/solutions/products/camels](https://ral.ucar.edu/solutions/products/camels) · bản theo giờ (Gauch et al., 2021): [hess.copernicus.org/articles/25/2045/2021](https://hess.copernicus.org/articles/25/2045/2021/)
- Columbia Basin DART: [cbr.washington.edu/dart/overview](https://www.cbr.washington.edu/dart/overview) · [cbr.washington.edu/dart/metadata/river](https://www.cbr.washington.edu/dart/metadata/river)
- WaterBench-Iowa: [essd.copernicus.org/articles/14/5605/2022](https://essd.copernicus.org/articles/14/5605/2022/) · [github.com/uihilab/WaterBench](https://github.com/uihilab/WaterBench)

**Công trình huấn luyện mô hình**
- Kirschstein & Sun (ICML 2024): [proceedings.mlr.press/v235/kirschstein24a.html](https://proceedings.mlr.press/v235/kirschstein24a.html) · mã nguồn [github.com/nkirschi/neural-flood-forecasting](https://github.com/nkirschi/neural-flood-forecasting)
- Mosaffa et al. (HESS 2026): [hess.copernicus.org/articles/30/2079/2026](https://hess.copernicus.org/articles/30/2079/2026/)
- Yang et al. (WRR 2025): [doi.org/10.1029/2024WR039764](https://doi.org/10.1029/2024WR039764)
- Konold et al. (HESS 2026): [hess.copernicus.org/articles/30/5067/2026](https://hess.copernicus.org/articles/30/5067/2026/)
- Wang et al., S4D-FT (WRR 2025): [doi.org/10.1029/2025WR039888](https://doi.org/10.1029/2025WR039888)
- Wang et al., HydroDiffusion: [arxiv.org/abs/2512.12183](https://arxiv.org/abs/2512.12183)
- Jing et al. (ESWA 2026): [doi.org/10.1016/j.eswa.2026.133040](https://doi.org/10.1016/j.eswa.2026.133040)
- Zhang et al. (J. Hydrology 2026): [doi.org/10.1016/j.jhydrol.2026.135727](https://doi.org/10.1016/j.jhydrol.2026.135727)
- Sun & Sun (Machine Learning: Earth 2026): [iopscience.iop.org/article/10.1088/3049-4753/ae4982](https://iopscience.iop.org/article/10.1088/3049-4753/ae4982)
- Koya & Roy (arXiv 2025): [arxiv.org/abs/2506.20831](https://arxiv.org/abs/2506.20831)
- Demiray & Demir, Multi-task (EarthArXiv 2025): [eartharxiv.org/repository/view/10562](https://eartharxiv.org/repository/view/10562/)
- Saint-Fleur et al. (HESS 2026): [hess.copernicus.org/articles/30/3497/2026](https://hess.copernicus.org/articles/30/3497/2026/)
- Heudorfer & Loritz (Hydrology Research 2026): [iwaponline.com/hr/article/57/3/501/111226](https://iwaponline.com/hr/article/57/3/501/111226)
- Adnan et al. (J. Hydrology: Regional Studies 2026): [sciencedirect.com/science/article/pii/S2214581826002272](https://www.sciencedirect.com/science/article/pii/S2214581826002272)
- Shanko & Melesse (Water 2026): [mdpi.com/2073-4441/18/14/1768](https://www.mdpi.com/2073-4441/18/14/1768)
- Sheng et al., ResBi-Mamba Plus (AISE 2026): [doi.org/10.23919/AISE.2026.000010](https://doi.org/10.23919/AISE.2026.000010)
- Sarkar et al., HydroGAT (ACM SIGSPATIAL 2025): [doi.org/10.1145/3748636.3764172](https://doi.org/10.1145/3748636.3764172) · [github.com/swapp-lab/HydroGAT](https://github.com/swapp-lab/HydroGAT)
- Demiray & Demir, Mamba + XAI (preprint 2025): [eartharxiv.org/repository/view/10109](https://eartharxiv.org/repository/view/10109/)
- Demiray & Demir, Transformer 120 h (Machine Learning: Earth 2026): [arxiv.org/abs/2406.07484](https://arxiv.org/abs/2406.07484)

**Công trình liên quan khác**
- Wang et al. (npj Natural Hazards 2025): [doi.org/10.1038/s44304-025-00083-6](https://doi.org/10.1038/s44304-025-00083-6)
- Shams Eddin et al., RiverMamba (NeurIPS 2025): [doi.org/10.52202/085713-4446](https://doi.org/10.52202/085713-4446)
