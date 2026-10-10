# BiasCast — Phần 1: BiasCast nói về gì

Ghi chú học bài cơ sở BiasCast (Konold và cs., HESS 30:5067–5096, 2026, DOI `10.5194/hess-30-5067-2026`) theo thứ tự từ khái niệm nền tới nâng cao. Mỗi phần giải thích khái niệm trước, sau đó mới đọc đoạn tương ứng của bài; ví dụ số tự đặt ra để minh họa được ghi rõ "minh họa".

Ký hiệu nguồn dùng trong tệp:

- **[Bài]**: bản chữ `PaperResearch/PaperResearchPDF/BasePaper/Konold2026_HESS_BiasCast_FullText.md`, kèm số dòng.
- **[Mã]**: mã nạp dữ liệu của tác giả `PaperResearch/PaperResearchCode/BiasCast_NH/neuralhydrology/datasetzoo/lamah.py`, kèm số dòng; các tệp mã khác trong cùng thư mục `neuralhydrology/` ghi tên tệp đầy đủ.
- **[LamaH]**: `03_Data/01_LamaHCE.md` (ghi chú dữ liệu của đề tài), kèm số dòng.
- **[Cơ sở]**: `01_Plan/02_BasePaper.md` (giới thiệu bài cơ sở), kèm mục.

Ghi chú học chia thành bảy phần, mỗi phần một tệp trong `06_Theory/`, đánh số tiếp theo thứ tự đọc: (1) BiasCast nói về gì — tệp này; (2) đo chất lượng dự báo — NSE và độ suy giảm khi đổi miền dữ liệu; (3) dữ liệu Extended LamaH-CE; (4) LSTM và hàm mất mát NSE*; (5) ba kiến trúc; (6) thêm lưu lượng quan trắc, dự báo lưu trữ, mạng nhúng; (7) kết quả theo lưu vực và hạn chế.

---

## Phần 1. BiasCast nói về gì

### 1.1. Đọc tên bài

Tên gốc ([Bài] dòng 43): *BiasCast: learning and adjusting real time biases from meteorological forecasts to enhance runoff predictions*.

| Cụm | Nghĩa |
|---|---|
| *meteorological forecasts* | dự báo thời tiết |
| *biases* | sai lệch có hệ thống |
| *learning and adjusting … biases* | học ra sai lệch rồi tự bù lại |
| *runoff predictions* | dự báo dòng chảy |

Nghĩa cả câu: học và tự bù sai lệch của dự báo thời tiết để dự báo dòng chảy tốt hơn. Tên bài chứa đủ ba khái niệm chính: dòng chảy, dự báo thời tiết, sai lệch.

**Dự báo thời tiết trong bài là gì.** Là ECMWF HRES — *European Centre for Medium-Range Weather Forecasts – High Resolution Forecast*, bản dự báo độ phân giải cao của Trung tâm Dự báo Thời tiết Hạn vừa châu Âu — lần chạy 00 UTC, dùng 5 biến (Mục 1.6). Phần mở đầu của bài có nêu các mô hình dự báo khác (DWD-ICON của Đức, NOAA-GFS của Mỹ) chỉ để minh họa rằng sai lệch khác nhau theo mô hình ([Bài] dòng 64); dữ liệu và thí nghiệm chỉ dùng ECMWF HRES ([Bài] dòng 80).

**Sai lệch có hệ thống (bias) là gì.** Sai số của một dự báo là hiệu giữa giá trị dự báo và giá trị thật. Sai số có hai loại:

| Loại | Đặc điểm | Ví dụ (minh họa) | Bù được không |
|---|---|---|---|
| Sai số ngẫu nhiên | lúc dư lúc thiếu, không theo quy luật; lấy trung bình nhiều ngày thì gần bằng 0 | mưa thật 10 mm: hôm dự báo 12, hôm 8, hôm 11 | không đoán trước được từng lần |
| Sai lệch có hệ thống | lệch đều về một phía hoặc theo một quy luật; lấy trung bình nhiều ngày vẫn khác 0 | ở một vùng núi, dự báo mưa gần như lúc nào cũng chỉ bằng khoảng 70% mưa thật | có — biết quy luật thì sửa được, như đồng hồ chạy chậm đều 5 phút mỗi ngày |

Chữ "Bias" trong tên bài là loại thứ hai: vì có quy luật nên mô hình có thể học ra để bù. Nguồn gốc của sai lệch trong dự báo thời tiết ở Mục 1.11; cách bài xử lý ở Mục 1.12.

### 1.2. Bài toán viết dưới dạng một hàm

Toàn bộ bài đi tìm một hàm f sao cho:

qmax(trạm, ngày t) = f(thời tiết quá khứ của lưu vực, dự báo thời tiết của lưu vực cho ngày t, thuộc tính lưu vực)

Mô tả trong phần này theo mô hình tốt nhất của bài (Sequential Forecast LSTM).

| Thành phần | Cụ thể | Nguồn |
|---|---|---|
| Thời tiết quá khứ của lưu vực | 364 ngày liền trước ngày t (từ t − 364 tới t − 1); mỗi ngày 31 biến từ ERA5-Land, E-OBS, MSWEP, GLEAM (danh sách ở Mục 1.5), không có ECMWF HRES; biến thể có lưu lượng thêm `qmean` đo tại trạm thành 32 biến | [Cơ sở] Mục 11.2; [Bài] dòng 354 |
| Dự báo thời tiết của lưu vực cho ngày t | 5 biến ECMWF HRES của lần chạy 00 UTC ngày t (mốc đầu ngày t theo giờ UTC; tệp dự báo phát ra sau đó vài giờ, Mục 1.9) | [Bài] dòng 80; [Cơ sở] Mục 4 |
| Thuộc tính lưu vực | 33 con số không đổi theo thời gian của cả lưu vực mà trạm đo (Mục 1.3), không phải của điểm đặt trạm; VD độ cao trung bình, độ dốc trung bình | [Bài] dòng 182, 214, 222 |
| Đầu ra qmax | lưu lượng lớn nhất trong ngày t đo tại trạm, mm/ngày | Mục 1.4 |

Các nguồn dữ liệu thời tiết, mỗi nguồn một dòng (chi tiết và nguồn tra ở Mục 1.5):

| Nguồn | Là gì | Làm bằng cách nào | Loại |
|---|---|---|---|
| ERA5-Land | dữ liệu mặt đất dựng lại cho quá khứ; lưới 9 km; 21 biến (nhiệt độ, mưa, tuyết, độ ẩm đất…) | chạy mô hình mặt đất CHTESSEL, lấy các biến khí quyển (nhiệt độ, độ ẩm không khí…) của tái phân tích khí quyển ERA5 làm đầu vào và hiệu chỉnh theo chênh lệch độ cao; bản thân ERA5 dùng mô hình IFS kết hợp số đo bằng đồng hóa 4D-Var. Số đo không đưa thẳng vào ERA5-Land mà đi gián tiếp qua ERA5. Do tổ chức ECMWF làm trong Dịch vụ Biến đổi Khí hậu Copernicus — cùng tổ chức làm ra bản dự báo ECMWF HRES nhưng là sản phẩm khác, không chứa dự báo | tái phân tích |
| E-OBS | số đo thật của trạm khí tượng các nước châu Âu (dự án ECA&D); 7 biến dùng (nhiệt độ, mưa, áp suất, gió, bức xạ) | nội suy thống kê số đo trạm ra lưới: tạo 20 phương án nội suy (mô phỏng có điều kiện với trường ngẫu nhiên tương quan không gian) rồi lấy trung bình làm bản "best-guess"; gió dùng hồi quy kriging; riêng bức xạ dùng thêm ảnh vệ tinh CERES để nội suy giữa các trạm | quan trắc |
| MSWEP | lượng mưa; 1 biến | trộn ba loại ước lượng mưa: số đo trạm (GSOD, GHCN-D, GPCC…), vệ tinh (IMERG, GSMaP, PERSIANN-CCS-CDR) và tái phân tích ERA5; bản hiện hành dùng học máy để trộn và hiệu chỉnh sai lệch | trộn, có cả tái phân tích |
| GLEAM | lượng bốc thoát hơi thực tế và tiềm năng; 2 biến | bộ thuật toán ước tính bốc hơi chủ yếu từ dữ liệu vệ tinh (bức xạ, nhiệt độ, gió, độ ẩm đất, thảm thực vật; bản 4.3a dùng thêm tái phân tích): tính bốc hơi tiềm năng bằng phương trình Penman rồi nhân với hệ số "căng thẳng" theo độ ẩm đất và thực vật để ra bốc hơi thực tế; độ ẩm đất vệ tinh được đồng hóa vào mô hình | mô hình dựa trên vệ tinh |
| ECMWF HRES | dự báo thời tiết cho tương lai, chỉ dùng cho ngày t; 5 biến | mô hình khí quyển IFS của tổ chức ECMWF chạy tới từ trạng thái hiện tại (Mục 1.6). Trong bài, chữ "ECMWF" và các cột `ECMWF_…` chỉ riêng sản phẩm dự báo này; cột tái phân tích mang tiền tố `ERA5L_` | dự báo số trị |

Cả năm nguồn đều là dữ liệu dạng lưới (E-OBS đi từ số đo điểm của trạm nhưng đã nội suy thành lưới); bài lấy chúng ở dạng raster rồi gộp theo lưu vực ([Bài] dòng 80, cách gộp ở Mục 1.3). Ô lưới theo các bản mô tả trong nguồn: ERA5-Land 0,1° (gốc 9 km), E-OBS 0,1° hoặc 0,25°, MSWEP 0,1°, GLEAM 0,25° (bản 3) hoặc 0,1° (bản 4), ECMWF HRES 9 km từ 08/03/2016 và thô hơn trước đó (Mục 1.5). Chỉ lưu lượng là số đo tại một điểm (trạm).

Nguồn mô tả cách làm: trang ERA5-Land (Copernicus Climate Data Store), Muñoz-Sabater và cs. (2021) Mục 2.4, Hersbach và cs. (2020) cho ERA5; trang E-OBS của ECA&D, Cornes và cs. (2018); trang GloH2O và tài liệu MSWEP V3; trang gleam.eu. Mô tả GLEAM theo phiên bản 4 hiện hành; bài BiasCast không ghi phiên bản E-OBS, MSWEP, GLEAM đã dùng, nên chi tiết thuật toán của bản tác giả dùng có thể khác.

Ghi chú cho từng thành phần:

- **Vì sao pha quá khứ không có ECMWF HRES:** bài theo thiết kế của Nearing và cs. (2024) — pha quá khứ dùng tái phân tích, pha dự báo dùng dự báo thời tiết ([Bài] dòng 132). Bài có thử thêm dự báo ECMWF HRES lưu trữ vào pha quá khứ nhưng không cải thiện và làm mô hình kém ổn định hơn ([Bài] dòng 244–258; chi tiết ở Phần 6).
- **Thí nghiệm có đủ dữ liệu tới hôm qua, vận hành thật thì không:** trong thí nghiệm, pha quá khứ có tái phân tích sát tới ngày t − 1; khi vận hành, tái phân tích trễ khoảng 5 ngày nên mấy ngày gần nhất chưa có. Bài chỉ thảo luận, không thí nghiệm tình huống này (Mục 1.5).
- **Mốc giờ lần chạy:** 00 UTC là 1 giờ sáng giờ Trung Âu mùa đông, 2 giờ sáng mùa hè, 7 giờ sáng giờ Việt Nam. Nhiệt độ, điểm sương (và áp suất mực biển, không dùng) là trung bình 8 giá trị cách nhau 3 giờ tính từ 00 UTC; mưa, bốc hơi là tổng ngày ([Bài] dòng 80).
- **Căn ngày:** cách căn ngày t của dữ liệu lưu lượng (giờ địa phương hay UTC) với ngày UTC của bản dự báo chưa kiểm được vì mã tạo bộ dữ liệu không công khai ([Cơ sở] Mục 3).

Cách mô hình dùng các thành phần trên:

- f là mạng LSTM (học ở Phần 4). Mỗi lần tính f cho ra một giá trị: một trạm, một ngày.
- Một mô hình f (một bộ trọng số) dùng chung cho 451 lưu vực Trung Âu ít hoặc không chịu tác động của con người ([Bài] dòng 84–86; [Cơ sở] Mục 3).
- **365 bước, không phải 365 lớp.** Mỗi mẫu đưa vào là cả bảng 364 ngày × 32 biến cộng 1 ngày dự báo. Một lớp LSTM duy nhất (một bộ trọng số) đọc lần lượt từng ngày: mỗi bước nhận dữ liệu của một ngày và trạng thái nhớ của bước trước; sau 364 bước quá khứ là bước thứ 365 đọc dự báo ECMWF HRES ngày t, và chỉ bước cuối này cho ra qmax ([Cơ sở] Mục 11.2). Mô hình không lấy dự báo của chính nó làm đầu vào cho bước sau. Cơ chế trạng thái nhớ học ở Phần 4.
- **Xử lý trước khi vào mạng:** đọc tệp, đổi −999 thành NaN, đổi nhãn sang mm/ngày; cắt theo kỳ; chuẩn hóa z-score mọi biến và thuộc tính bằng trung bình, độ lệch chuẩn của kỳ train; cắt cửa sổ và loại mẫu thiếu dữ liệu khi huấn luyện. Bên trong mạng, lớp nhúng đổi 32 biến quá khứ, 5 biến dự báo và 33 thuộc tính thành các vectơ 16 chiều trước khi vào LSTM ([Cơ sở] Mục 11.1 bước 1–4, Mục 11.2). Chi tiết ở Phần 3 và Phần 4.

Mục 1.3 giải thích trạm và lưu vực; Mục 1.4 giải thích giá trị trả về; Mục 1.5–1.7 giải thích từng đầu vào.

### 1.3. Trạm đo và lưu vực — đầu vào lấy ở đâu

**Trạm đo dòng chảy** là điểm đặt trên sông để đo lượng nước chảy qua. **Lưu vực của trạm** là toàn bộ vùng đất mà nước mưa rơi xuống đều chảy về trạm đó. Ranh giới lưu vực là đường sống núi chia nước, không phải vòng tròn quanh trạm: mưa rơi bên trong ranh giới chảy xuống suối rồi qua trạm; mưa rơi bên ngoài chảy sang thung lũng khác; đất nằm phía hạ lưu trạm không thuộc lưu vực vì nước ở đó không chảy ngược lên trạm. Vị trí đặt trạm quyết định lưu vực: dời trạm đi chỗ khác thì vùng thượng nguồn đổ về nó cũng khác.

LamaH chia lưu vực theo ba cách, gọi là mức A, B, C (Klingler và cs., ESSD 2021, tr. 4531, Hình 2 tr. 4533):

| Mức | Vùng gộp dữ liệu của một trạm | Chồng lấn | Thư mục |
|---|---|---|---|
| **A** | toàn bộ vùng thượng nguồn đổ về trạm (lưu vực địa hình), giống cách của bộ CAMELS | có: lưu vực trạm hạ lưu bao trọn lưu vực trạm thượng lưu | `A_basins_total_upstrm` |
| B | chỉ phần diện tích chênh lệch giữa lưu vực của trạm và lưu vực của các trạm ngay phía trên (lưu vực trung gian) | không; cần biết thứ bậc thượng – hạ lưu để ghép lại | `B_basins_intermediate_all` |
| C | như B nhưng chỉ giữ 454 lưu vực không hoặc ít chịu tác động của con người | không | `C_basins_intermediate_lowimp` |

Ví dụ: trạm X ở thượng lưu, trạm Y ở hạ lưu. Mức A: vùng của Y gồm cả vùng của X, nên mỗi trạm tự đủ dữ liệu, dự báo độc lập được. Mức B: vùng của Y chỉ còn phần đất giữa X và Y, nên muốn dự báo Y phải dùng thêm nước từ X chảy xuống (mô hình mạng sông). Bài dùng **mức A** ([Bài] dòng 86): mọi tính toán gộp thành một đơn vị cho mỗi trạm, "lumped for each gauge" ([Bài] dòng 84). Ranh giới 859 lưu vực mức A nằm trong tệp `Basins_A.shp` ([LamaH] dòng 133); diện tích trung vị 178 km², nhỏ nhất 4 km², lớn nhất 131 247 km² (Klingler và cs. 2021, tr. 4531).

**Đầu vào nằm ở đâu:**

| Thành phần | Lấy ở đâu | Nguồn |
|---|---|---|
| Thời tiết quá khứ và dự báo ECMWF HRES | trung bình trên cả lưu vực: dữ liệu dạng lưới được gộp theo đa giác lưu vực | [Bài] dòng 80: *"obtained as raster data and subsequently aggregated to the LamaH basins"* |
| 33 thuộc tính | của cả lưu vực (VD độ cao trung bình, độ dốc trung bình) | [Bài] dòng 214, 222 |
| Lưu lượng quá khứ (đầu vào) và qmax (đầu ra) | đo tại trạm | tệp trạm `D_gauges` ([Mã] dòng 225–226) |

**Ví dụ một trạm (minh họa).** Dữ liệu thời tiết không có sẵn theo trạm mà theo ô lưới; mỗi ô một giá trị (ERA5-Land có ô 0,1°, độ phân giải gốc 9 km — trang ERA5-Land, Copernicus Climate Data Store). Lấy biến `ERA5L_prec` (tổng lượng mưa trong ngày của ERA5-Land, đơn vị mm; BiasCast Bảng A1) ngày 14/05/2016, đặt lưới lên lưu vực của trạm X; mỗi số trong bảng là lượng mưa (mm) của một ô; ô **in đậm** nằm trong ranh giới lưu vực X, trạm X nằm ở hàng 5, cột 3, ngay dưới chỗ nước thoát ra khỏi lưu vực:

| `ERA5L_prec` (mm) | Cột 1 | Cột 2 | Cột 3 | Cột 4 | Cột 5 | Cột 6 |
|---|---|---|---|---|---|---|
| Hàng 1 | 2 | 3 | 5 | 6 | 4 | 1 |
| Hàng 2 | 3 | **8** | **12** | **10** | 5 | 2 |
| Hàng 3 | 2 | **7** | **15** | **11** | **6** | 1 |
| Hàng 4 | 1 | **4** | **9** | **5** | 3 | 0 |
| Hàng 5 | 0 | 1 | ⊙ trạm X | 2 | 1 | 0 |

- Gộp 10 ô trong ranh giới: (8 + 12 + 10 + 7 + 15 + 11 + 6 + 4 + 9 + 5) / 10 = 87 / 10 = **8,7 mm**. Đây là lượng mưa ngày 14/05/2016 của lưu vực X — cả lưu vực chỉ còn một con số cho biến này trong ngày này. Ví dụ dùng trung bình cộng cho dễ thấy; bài không mô tả chi tiết cách gộp.
- Các ô ngoài ranh giới đổ nước sang lưu vực khác nên không tính cho X.
- Lặp lại cho mọi biến (mỗi biến một lưới) và mọi ngày từ 1981 tới 2017 (riêng các cột ECMWF HRES có từ 01/01/2002, [LamaH] dòng 101) thì được tệp thời tiết của trạm X (giá trị minh họa):

| Ngày | `ERA5L_prec` — mưa ERA5-Land (mm) | `ERA5L_2m_temp_mean` — nhiệt độ trung bình ERA5-Land (°C) | … | `ECMWF_tp` — mưa dự báo ECMWF HRES (mm) | … |
|---|---|---|---|---|---|
| 13/05/2016 | 0,0 | 12,1 | … | 0,3 | … |
| 14/05/2016 | 8,7 | 10,4 | … | 6,2 | … |
| 15/05/2016 | 21,3 | 9,8 | … | 14,0 | … |

- Muốn dự báo qmax ngày 15/05/2016 tại X, mô hình đọc tệp thời tiết của X (364 ngày trước và dự báo ECMWF HRES của ngày 15/05), thuộc tính lưu vực X và lưu lượng quá khứ đo tại X, rồi so kết quả với qmax đo tại X.

**Nhiều trạm — tệp nằm ở đâu.** Mỗi trạm có một mã số; cùng mã số đó dùng để tìm dữ liệu của trạm ở hai thư mục và hai tệp thuộc tính ([Mã] dòng 116–133, 180–181, 225–226, 265–269):

| Dữ liệu | Đường dẫn trong bộ dữ liệu | Số tệp |
|---|---|---|
| Thời tiết trung bình lưu vực (mức A) | `A_basins_total_upstrm/2_timeseries/daily/ID_<số>.csv` — một tệp cho mỗi lưu vực | 859 lưu vực ([LamaH] dòng 133) |
| Lưu lượng đo tại trạm | `D_gauges/2_timeseries/daily/ID_<số>.csv` — một tệp cho mỗi trạm | 882 trạm ([LamaH] dòng 133) |
| Thuộc tính lưu vực | `A_basins_total_upstrm/1_attributes/Catchment_attributes.csv` — một tệp chung, mỗi dòng một lưu vực | 1 |
| Thuộc tính trạm (có diện tích `area_gov`) | `D_gauges/1_attributes/Gauge_attributes.csv` — một tệp chung, mỗi dòng một trạm | 1 |

Bộ dữ liệu có 859 lưu vực; bài chỉ dùng 451 lưu vực "no and low anthropogenic influence" ở mức A ([Bài] dòng 86). LamaH-CE xếp mỗi trạm vào một mức tác động `degimpact` (Klingler và cs. 2021, Bảng 2, tr. 17): u — không tác động (không có loại tác động nào, trạm nằm trên khu dân cư, lấy hoặc dẫn nước làm đổi diện tích lưu vực dưới 1%); l — tác động thấp (trạm trong hoặc dưới khu đô thị, hồ có cửa ra tự nhiên, thực vật gần trạm, lấy hoặc dẫn nước 1–3%); m — vừa (hồ chứa cắt lũ, hồ có cửa điều tiết, đo lũ không chuẩn…, 3–10%); s — mạnh (hồ chứa có nước quanh năm); x — loại bỏ. Bài giữ u và l; bài không nêu lý do, nhưng mô hình không có thông tin về vận hành hồ chứa hay lấy nước nên lưu lượng bị con người điều tiết không học được từ thời tiết (suy luận). Bài cũng không ghi cách lọc ra đúng con số 451; tác giả công bố sẵn danh sách mã số `basins_filtered.txt` ở gốc thư mục `Experiments/` trên Zenodo, khớp 451 lưu vực trong kết quả test của tác giả ([Bài] dòng 86; [Cơ sở] Mục 11). Với trạm số 100 (VD), mã đọc `A_basins_total_upstrm/2_timeseries/daily/ID_100.csv` và `D_gauges/2_timeseries/daily/ID_100.csv`, ghép hai bảng theo ngày, rồi lấy dòng của trạm 100 trong hai tệp thuộc tính. Mô hình dùng chung một bộ trọng số, nhưng mỗi mẫu chỉ chứa dữ liệu của đúng một trạm, không nhìn sang trạm khác. 72,5% lưu vực là lưu vực đầu nguồn, 27,5% là lưu vực lồng nhau — có ít nhất một trạm khác trong tập nằm ở thượng nguồn ([Bài] dòng 86). Khi trạm X nằm ở thượng nguồn trạm Y (như ví dụ mức A ở trên), lưu vực Y bao trọn lưu vực X: mưa trên vùng X được tính vào trung bình của X và một lần nữa vào trung bình của Y, nhưng hai lần dự báo tách biệt nhau.

**Sai số do cách lấy đầu vào:**

- Gộp trung bình làm mất phân bố không gian: mưa ở đầu nguồn xa và mưa sát trạm cho cùng một con số trung bình, dù nước từ hai nơi về tới trạm ở thời điểm khác nhau. Bài nêu thí nghiệm với dữ liệu phân bố theo không gian có thể cho kết quả khác ([Bài] dòng 334: *"fully distributed settings"*).
- Lưu vực chỉ phủ vài ô lưới thì trung bình phụ thuộc vào rất ít ô (suy luận; bài không phân tích riêng).
- Lưu lượng tại trạm không đo trực tiếp mà suy từ mực nước qua đường quan hệ mực nước – lưu lượng, sai số lớn khi lũ cực đoan hoặc cảm biến hỏng ([Bài] dòng 68). Sai số này nằm ở cả lưu lượng đầu vào lẫn nhãn.
- Quan hệ thượng – hạ lưu giữa các trạm lồng nhau không được dùng.

### 1.4. Giá trị trả về: qmax

**Lưu lượng Q** là thể tích nước chảy qua mặt cắt sông tại trạm trong một giây, đơn vị m³/s. Q = 50 m³/s nghĩa là mỗi giây có 50 m³ nước đi qua. Lưu lượng khác mực nước (mét); mô hình chỉ dự báo lưu lượng.

**Tệp lưu lượng.** Mỗi trạm có một tệp CSV phân cách bằng dấu chấm phẩy ở thư mục `D_gauges/2_timeseries/daily/ID_<số>.csv`, tách riêng với tệp thời tiết ([Mã] dòng 225–226). Ngày ghi ở ba cột `YYYY`, `MM`, `DD` và được ghép thành chỉ mục ngày khi đọc ([Mã] dòng 285–286). Bảng dưới minh họa cấu trúc (giá trị minh họa):

| YYYY | MM | DD | qmean | qmin | qmax | Ghi chú |
|---|---|---|---|---|---|---|
| 2016 | 05 | 14 | 12,3 | 10,1 | 15,8 | ngày bình thường |
| 2016 | 05 | 15 | 30,7 | 14,2 | 61,5 | ngày mưa lớn, đỉnh vọt lên |
| 2016 | 05 | 16 | −999 | −999 | −999 | trạm không có số đo |

- Giá trị thiếu ghi −999; khi đọc, mọi giá trị âm của cột nhãn được đổi thành NaN ([Mã] dòng 229). Nếu giữ −999, mô hình sẽ học một "lưu lượng âm" không có thật.
- `qmean`, `qmin`, `qmax` là lưu lượng trung bình, nhỏ nhất, lớn nhất trong ngày. Bản LamaH-CE theo ngày gốc chỉ có lưu lượng trung bình; tác giả trích cực tiểu và cực đại ngày từ dữ liệu theo giờ của LamaH-CE ([Bài] dòng 80). Mã thực hiện bước trích này không công khai; về ý tưởng, mỗi ngày lấy giá trị nhỏ nhất và lớn nhất trong các giá trị giờ của ngày đó. Cách căn ranh giới ngày (giờ địa phương hay UTC) chưa kiểm được ([Cơ sở] Mục 3).
- **Vì sao dự báo qmax:** lũ gây thiệt hại tại thời điểm đỉnh; trung bình ngày làm phẳng đỉnh. Trong ví dụ trên, ngày 15/05 có qmean 30,7 nhưng qmax 61,5 m³/s.

**Đổi đơn vị.** Khi đọc nhãn từ tệp trạm vào bộ nhớ (chưa phải đưa vào mô hình), mã chia lưu lượng cho diện tích lưu vực `area_gov` ([Mã] dòng 131–133, 303–308; chú thích ở dòng 32–34 ghi lưu lượng gốc của LamaH tính bằng m³/s); hệ số 10⁶ trong công thức cho thấy diện tích tính bằng km²: q = Q / (A × 10⁶) × 1000 × 86400.

| Phép tính | Ý nghĩa | Đơn vị sau phép tính |
|---|---|---|
| A × 10⁶ | đổi km² ra m² | m² |
| Q / (A × 10⁶) | chia thể tích nước mỗi giây cho diện tích | m/s (chiều cao lớp nước) |
| × 1000 | đổi m ra mm | mm/s |
| × 86400 | số giây trong một ngày | mm/ngày |

Ví dụ: Q = 50 m³/s, A = 500 km² → 50 / (5 × 10⁸) × 1000 × 86400 = 8,64 mm/ngày, tức lượng nước chảy qua trạm trong một ngày bằng một lớp nước dày 8,64 mm phủ đều cả lưu vực. Đây vẫn là lưu lượng tại trạm, không phải lượng mưa. Đổi ngược: Q (m³/s) = q (mm/ngày) × A (km²) / 86,4.

**Đường đi của nhãn** (mã NeuralHydrology trong bản fork của tác giả):

| Bước | Việc | Đơn vị của nhãn | Nguồn |
|---|---|---|---|
| 1 | Đọc `qmax` từ tệp trạm, đổi giá trị âm thành NaN, chia theo diện tích | m³/s → mm/ngày | [Mã] dòng 131–133, 229, 303–308 |
| 2 | Chuẩn hóa z-score: (y − trung bình) / độ lệch chuẩn, hai thống kê tính trên kỳ train; đầu vào cũng được chuẩn hóa cùng cách nhưng là dòng dữ liệu riêng | không thứ nguyên | `datasetzoo/basedataset.py` dòng 758, 762–765 |
| 3 | Huấn luyện: mô hình nhận đầu vào, xuất ŷ ở thang chuẩn hóa; hàm mất mát so ŷ với nhãn đã chuẩn hóa | không thứ nguyên | [Cơ sở] Mục 11.1 bước 7 |
| 4 | Test: đổi ngược ŷ và nhãn bằng × độ lệch chuẩn + trung bình, rồi tính NSE, KGE | mm/ngày | `evaluation/tester.py` dòng 250–256 |

**Đầu ra của mô hình:** số mạng trực tiếp xuất ra ở lớp cuối nằm ở thang chuẩn hóa; kết quả tác giả lưu và báo cáo là **qmax của ngày t tại từng trạm, đơn vị mm/ngày** — mỗi ngày 451 giá trị. Mã không đổi kết quả về m³/s; khi cần (VD hiển thị trên trang demo) nhân theo công thức đổi ngược ở trên. NSE không phụ thuộc đơn vị vì mỗi lưu vực chỉ chịu một phép nhân hằng số dương và cộng hằng số (Phần 2); đơn vị chủ yếu ảnh hưởng tới việc học của mô hình chung.

Lý do đổi: lưu lượng tính bằng m³/s phụ thuộc mạnh vào diện tích, lưu vực lớn có lưu lượng lớn hơn lưu vực nhỏ nhiều lần; chia theo diện tích đưa mọi lưu vực về cùng thang đo với lượng mưa (mm), nhờ đó một mô hình chung học được cho cả 451 lưu vực.

### 1.5. Đầu vào thứ nhất: thời tiết quá khứ — quan trắc và tái phân tích

**Tệp thời tiết.** Mỗi lưu vực có một tệp `A_basins_total_upstrm/2_timeseries/daily/ID_<số>.csv` ([Mã] dòng 180–181) gồm 44 cột: `DOY`, 21 biến ERA5-Land, 8 biến ECMWF HRES, 8 biến E-OBS, `MSWEP_RR`, `GLEAM_ETA`, `GLEAM_ETP` ([LamaH] dòng 87), từ 01/01/1981 tới 31/12/2017 ([LamaH] dòng 101, 133). Mọi giá trị là trung bình trên lưu vực (Mục 1.3). Năm nguồn thuộc ba loại dữ liệu khác nhau:

| Nguồn | Số biến | Loại dữ liệu | Nguồn mô tả |
|---|---|---|---|
| ERA5-Land | 21 | tái phân tích | Muñoz-Sabater và cs., 2021; [Bài] dòng 80 |
| E-OBS | 8 (bài dùng 7, bỏ `EOBS_hu`) | quan trắc trạm khí tượng nội suy thành lưới | Cornes và cs., 2018; [Bài] dòng 80 |
| MSWEP | 1 (mưa) | trộn số đo trạm, vệ tinh và tái phân tích | Beck và cs., 2019 |
| GLEAM | 2 (bốc hơi thực, bốc hơi tiềm năng) | mô hình ước tính bốc hơi từ dữ liệu vệ tinh | Miralles và cs., 2011 |
| ECMWF HRES | 8 (bài dùng 5, bỏ `ECMWF_msl`, `ECMWF_lsp`, `ECMWF_cp`) | dự báo thời tiết số trị (Mục 1.6) | [Bài] dòng 80 |

**Năm nguồn dữ liệu là gì:**

| Nguồn | Ai làm, làm thế nào | Lưới, thời gian | Nguồn mô tả |
|---|---|---|---|
| **ERA5-Land** | Dịch vụ Biến đổi Khí hậu Copernicus (C3S) do ECMWF vận hành. Chạy lại riêng phần mặt đất của tái phân tích ERA5 ở lưới mịn hơn, lấy nhiệt độ, độ ẩm… của ERA5 làm đầu vào điều khiển; số đo chỉ ảnh hưởng gián tiếp qua ERA5 (chi tiết ở đoạn "Tái phân tích" bên dưới) | 0,1° (gốc 9 km), theo giờ, toàn cầu, từ 1950; cập nhật hằng ngày | trang ERA5-Land, Copernicus Climate Data Store; Muñoz-Sabater và cs., 2021 |
| **E-OBS** | Dự án ECA&D (European Climate Assessment & Dataset) gom số đo trạm khí tượng do các cơ quan khí tượng thủy văn quốc gia châu Âu cung cấp, rồi nội suy thành lưới; chỉ phủ đất liền châu Âu. Bản dùng phổ biến là trung bình của 20 thành viên nội suy ("best-guess"). Bức xạ dùng thêm sản phẩm vệ tinh CERES để nội suy giữa các trạm. Khoảng 24 giờ của "một ngày" có thể khác nhau giữa các vùng tùy nơi đo nửa đêm–nửa đêm hay sáng–sáng | 0,1° và 0,25°, theo ngày, từ 1950 (gió từ 1980); bản mới phát hành mỗi 6 tháng | trang E-OBS, Copernicus Climate Data Store; trang E-OBS của ECA&D; Cornes và cs., 2018 |
| **MSWEP** | Multi-Source Weighted-Ensemble Precipitation (GloH2O): trộn ba loại ước lượng mưa — số đo trạm, vệ tinh và mô hình/tái phân tích — để tận dụng điểm mạnh của từng loại ở vùng nhiều trạm, vùng mưa đối lưu, vùng mưa front; chỉ có biến mưa | bản V2: 3 giờ, 0,1°; bản hiện hành: theo giờ, 0,1°, toàn cầu, từ 1979, trễ khoảng 2–3 giờ | trang GloH2O; Climate Data Guide (UCAR); Beck và cs., 2019 |
| **GLEAM** | Global Land Evaporation Amsterdam Model: tập thuật toán ước tính lượng bốc hơi từ mặt đất dựa trên dữ liệu vệ tinh; cho bốc thoát hơi thực tế và tiềm năng | bản 3: theo ngày, 0,25°; bản 4: 0,1° | trang gleam.eu; Climate Data Guide (UCAR); Miralles và cs., 2011 |
| **ECMWF HRES** | Bản dự báo tất định (một phương án duy nhất) độ phân giải cao nhất của ECMWF, chạy bằng hệ thống IFS; nay ECMWF gọi là "IFS Medium-range Control forecast (formerly HRES)". Chi tiết ở Mục 1.6 | 9 km từ 08/03/2016 (chu kỳ mô hình 41r2), trước đó thô hơn; mỗi ngày 4 lần chạy 00/06/12/18 UTC, dữ liệu của bài dùng lần 00 UTC | trang ECMWF Set I và Operational archive; ECMWF Newsletter 147; [Bài] dòng 80 |

Bài không ghi phiên bản cụ thể của E-OBS, MSWEP, GLEAM được dùng; độ phân giải ở bảng là của các bản được mô tả trong nguồn, chưa chắc trùng với bản tác giả dùng.

**31 biến thời tiết quá khứ** (Bảng A1 của bài, PDF trang 20–21; tên biến đúng như tên cột trong tệp). Ở mô hình tốt nhất, phần quá khứ **không chứa ECMWF HRES**: chú thích Bảng A1 ghi mọi biến tái phân tích là đầu vào động ở tất cả thí nghiệm, trừ thí nghiệm CrossDomain (chỉ 5 biến có biến tương ứng trong ECMWF HRES) và Baseline Forecast (chỉ 5 biến ECMWF HRES); thí nghiệm thêm dự báo lưu trữ vào phần quá khứ học ở Phần 6 ([Bài] dòng 248, 354).

| Nguồn | Biến | Nghĩa | Đơn vị |
|---|---|---|---|
| ERA5-Land | `ERA5L_2m_temp_max`, `_mean`, `_min` | nhiệt độ không khí ở 2 m: lớn nhất, trung bình, nhỏ nhất | °C |
| ERA5-Land | `ERA5L_2m_dp_temp_max`, `_mean`, `_min` | nhiệt độ điểm sương ở 2 m (đại diện độ ẩm không khí): lớn nhất, trung bình, nhỏ nhất | °C |
| ERA5-Land | `ERA5L_10m_wind_u`, `ERA5L_10m_wind_v` | gió ở 10 m, thành phần hướng đông và hướng bắc | m/s |
| ERA5-Land | `ERA5L_fcst_alb` | albedo (tỷ lệ ánh sáng mặt đất phản xạ lại) | — |
| ERA5-Land | `ERA5L_lai_high_veg`, `ERA5L_lai_low_veg` | chỉ số diện tích lá của thảm thực vật cao, thấp | m²/m² |
| ERA5-Land | `ERA5L_swe` | lượng nước trong lớp tuyết (đương lượng nước của tuyết) | mm |
| ERA5-Land | `ERA5L_surf_net_solar_rad_max`, `_mean` | bức xạ mặt trời thuần ở mặt đất (tới trừ phản xạ): lớn nhất, trung bình | W/m² |
| ERA5-Land | `ERA5L_surf_net_therm_rad_max`, `_mean` | bức xạ nhiệt thuần ở mặt đất: lớn nhất, trung bình | W/m² |
| ERA5-Land | `ERA5L_surf_press` | áp suất mặt đất | Pa |
| ERA5-Land | `ERA5L_total_et` | tổng bốc thoát hơi | mm |
| ERA5-Land | `ERA5L_prec` | tổng lượng mưa | mm |
| ERA5-Land | `ERA5L_volsw_123`, `ERA5L_volsw_4` | độ ẩm đất: lớp 0–100 cm, lớp 100–289 cm | m³/m³ |
| E-OBS | `EOBS_tg`, `EOBS_tn`, `EOBS_tx` | nhiệt độ không khí ngày: trung bình, nhỏ nhất, lớn nhất | °C |
| E-OBS | `EOBS_rr` | tổng lượng mưa | mm |
| E-OBS | `EOBS_pp` | áp suất mực biển trung bình | hPa |
| E-OBS | `EOBS_fg` | tốc độ gió trung bình ở 10 m | m/s |
| E-OBS | `EOBS_qq` | bức xạ mặt trời ở mặt đất | W/m² |
| MSWEP | `MSWEP_RR` | tổng lượng mưa | mm |
| GLEAM | `GLEAM_ETA`, `GLEAM_ETP` | bốc thoát hơi thực tế, bốc thoát hơi tiềm năng | mm |

Đếm: ERA5-Land 21, E-OBS 7, MSWEP 1, GLEAM 2 — cộng 31. Lượng mưa có ba nguồn độc lập (ERA5-Land, E-OBS, MSWEP), nhiệt độ có hai nguồn (ERA5-Land, E-OBS); chú thích Bảng 1 của bài nêu việc lấy biến từ nhiều sản phẩm nhằm kết hợp điểm mạnh của từng nguồn và tránh để sai lệch hệ thống của một nguồn lan ra mọi biến ([Bài] dòng 124).

Bài gọi chung dữ liệu quá khứ là "reanalysis" (VD [Bài] dòng 248: *"solely reanalysis data in the hindcast phase"*); theo bảng trên chỉ ERA5-Land là tái phân tích đúng nghĩa, cách gọi của bài mang nghĩa "dữ liệu quá khứ chất lượng cao, đã có đủ".

**Quan trắc** là giá trị do dụng cụ thật đo: thùng đo mưa, nhiệt kế ở trạm khí tượng (khác với trạm đo dòng chảy ở Mục 1.3).

- Thưa: chỉ có giá trị tại vị trí đặt trạm; muốn có giá trị cho cả vùng phải nội suy từ các trạm xung quanh. E-OBS là kết quả nội suy số đo trạm khí tượng châu Âu thành lưới đều.
- Có sai số đo: thùng đo mưa hứng thiếu khi có gió, do gió thổi hạt mưa bay qua miệng thùng ([Bài] dòng 68, dẫn Yang và cs., 1999).
- Chỉ có sau khi sự việc xảy ra: không có số đo cho ngày mai.

**Tái phân tích** là chạy lại một mô hình vật lý khí quyển cho các năm đã qua, kết hợp với số đo trên toàn cầu theo các định luật vật lý, để tạo ra bộ dữ liệu đầy đủ (mọi ô lưới, mọi ngày) và nhất quán (*"globally complete and consistent dataset"* — trang ERA5-Land, Copernicus Climate Data Store). Kỹ thuật kết hợp số đo vào mô hình gọi là **đồng hóa số liệu**; ERA5 dùng phương pháp 4D-Var (Hersbach và cs., 2020, QJRMS).

- **ERA5** là tái phân tích khí quyển, trực tiếp đồng hóa số đo.
- **ERA5-Land** chạy lại riêng phần mặt đất của ERA5 ở độ phân giải cao hơn (9 km), dùng nhiệt độ, độ ẩm không khí… của ERA5 làm đầu vào điều khiển. Số đo không được dùng trực tiếp khi tạo ERA5-Land mà ảnh hưởng gián tiếp qua đầu vào từ ERA5; nhiệt độ, độ ẩm, áp suất được hiệu chỉnh theo chênh lệch độ cao giữa hai lưới (trang ERA5-Land, Copernicus Climate Data Store).
- Trong dữ liệu của đề tài, các cột khí tượng (mọi nguồn) gần như không thiếu giá trị — khí tượng 2003–2017 của 451 lưu vực thiếu nhiều nhất 0,36% ở cột `EOBS_hu` không dùng trong cấu hình, mọi cột khác 0,00% ([LamaH] dòng 101).
- Chỉ làm được sau khi số đo đã về: ERA5 và bản sơ bộ ERA5-Land-T được cập nhật hằng ngày với độ trễ khoảng 5 ngày so với hiện tại (trang ERA5, Copernicus Climate Data Store; thông báo phát hành ERA5-Land-T, diễn đàn ECMWF). Khi vận hành không có tái phân tích cho ngày cần dự báo.

**Thí nghiệm của bài so với vận hành thật.** Thí nghiệm chạy trên dữ liệu lịch sử đã đầy đủ: phần 364 ngày quá khứ có tái phân tích sát tới ngày t − 1, và khi huấn luyện mẫu có ô thiếu bị loại ([Cơ sở] Mục 11.1 bước 4). Khi vận hành, sáng ngày t chỉ có ERA5-Land tới khoảng 5 ngày trước; mấy ngày gần nhất — những ngày ảnh hưởng mạnh nhất tới lưu lượng ngày t — chưa có. Bài không làm thí nghiệm cho tình huống này; Mục 3.7 (Hạn chế) chỉ thảo luận ([Bài] dòng 338):

- độ trễ của tái phân tích ở pha quá khứ là ràng buộc, thường quan trọng khi vận hành; tái phân tích gần thời gian thực trong tương lai có thể giảm bớt;
- ở quy mô quốc gia có thể thay pha quá khứ bằng phân tích thời gian thực, nowcast hoặc số đo trạm, VD sản phẩm INCA của GeoSphere Austria (Haiden và cs., 2011);
- ở quy mô nhiều lưu vực như bài, dùng số đo trạm gặp khó vì số trạm mỗi lưu vực rất khác nhau (lưu vực lớn hàng chục trạm, lưu vực nhỏ một trạm), số chiều đầu vào khác nhau giữa các lưu vực; mạng nhúng có thể giúp nhưng ảnh hưởng tới chất lượng và độ ổn định là câu hỏi mở;
- giảm phụ thuộc vào dữ liệu trễ hoặc không có được nêu là hướng cần làm ([Bài] dòng 340).

Mỗi nguồn trễ khác nhau (bản hằng ngày của E-OBS khoảng 1 ngày, MSWEP gần thời gian thực khoảng 2–3 giờ, GLEAM cập nhật mỗi năm một lần — bản 4.3b phát hành 21/09/2026 phủ tới hết 2025, nên khi vận hành không có GLEAM cho năm đang chạy (trang gleam.eu)); đề tài đo tác động của độ trễ thật bằng các kịch bản S0–S4 (`01_Plan/03_Pipeline.md` Mục 6.3).

So sánh bằng hình ảnh kỳ thi: quan trắc là điểm thật nhưng chỉ biết của vài người; tái phân tích là chấm lại cả lớp sau khi có đáp án — đầy đủ, sát thực tế hơn dự báo nhưng vẫn có sai số, và chỉ làm được sau kỳ thi; dự báo là đoán điểm trước khi thi — có ngay nhưng sai nhiều hơn.

### 1.6. Đầu vào thứ hai: dự báo thời tiết số trị cho ngày t

**Dự báo thời tiết số trị** dùng cùng loại mô hình vật lý khí quyển, bắt đầu từ trạng thái khí quyển hiện tại rồi tính tới tương lai. Khác với tái phân tích, không có số đo của tương lai để kết hợp vào, nên sai số tích lũy theo thời gian: dự báo kém chính xác hơn dữ liệu quan trắc và tái phân tích, và sai số tăng theo hạn dự báo ([Bài] dòng 62, dẫn Lavers và cs., 2021; Nester và cs., 2012).

Trong dữ liệu:

- **ECMWF HRES**: bản dự báo tất định độ phân giải cao của Trung tâm Dự báo Thời tiết Hạn vừa châu Âu (ECMWF); "tất định" nghĩa là chỉ một phương án dự báo, khác với dự báo tổ hợp (ENS) chạy nhiều phương án để ước lượng độ bất định. Dữ liệu của bài dùng bản phát lúc 00 UTC; dữ liệu lưới được gộp theo lưu vực như mọi nguồn khác ([Bài] dòng 80).
- Độ phân giải lên 9 km từ 08/03/2016; trước đó ô lưới thô khoảng gấp đôi, cỡ 16 km (ECMWF Newsletter 147: độ phân giải lưới HRES "roughly doubled to 9 km"). Chuỗi ECMWF HRES trong dữ liệu (2002–2017) vì vậy đến từ nhiều phiên bản mô hình khác nhau, nên sai lệch của dự báo có thể thay đổi theo thời gian (suy luận; bài không phân tích điểm này).
- Nhiệt độ, điểm sương, áp suất mực biển là trung bình 8 giá trị cách nhau 3 giờ tính từ lần phát hành 00 UTC; với mọi biến, bài ghi chung là trung bình ngày (VD nhiệt độ) hoặc tổng ngày (VD mưa) ([Bài] dòng 80). Năm biến bài dùng ([Cơ sở] Mục 4; tên, nghĩa, đơn vị theo Bảng A1 của bài, PDF trang 21):

| Biến | Nghĩa | Đơn vị |
|---|---|---|
| `ECMWF_t2m` | nhiệt độ không khí trung bình ở 2 m (dự báo) | °C |
| `ECMWF_d2m` | nhiệt độ điểm sương trung bình ở 2 m (dự báo) | °C |
| `ECMWF_ssrd` | bức xạ mặt trời tới mặt đất (dự báo) | W/m² |
| `ECMWF_tp` | tổng lượng mưa (dự báo) | mm |
| `ECMWF_e` | tổng bốc thoát hơi thực tế (dự báo) | mm |

Ba cột ECMWF HRES còn lại trong tệp (`ECMWF_msl` áp suất mực biển, `ECMWF_lsp` mưa diện rộng, `ECMWF_cp` mưa đối lưu) không dùng trong cấu hình.
- Mỗi ngày một giá trị cho hạn 1 ngày; có từ 01/01/2002 ([LamaH] dòng 101).

### 1.7. Đầu vào thứ ba: thuộc tính lưu vực

**Thuộc tính tĩnh** là các con số mô tả lưu vực, không đổi theo ngày: địa hình, khí hậu trung bình nhiều năm, lớp phủ đất, thực vật. Tệp `A_basins_total_upstrm/1_attributes/Catchment_attributes.csv` có 62 thuộc tính cho 859 lưu vực ([LamaH] dòng 133); cấu hình của bài dùng 33, liệt kê ở Bảng A2 của bài (PDF trang 22; [Bài] dòng 182).

**Vì sao cần.** Một mô hình dùng chung cho 451 lưu vực. Cùng 20 mm mưa, lưu vực núi cao nhiều tuyết, dốc, ít rừng phản ứng khác hẳn lưu vực đồng bằng nhiều rừng. Nếu chỉ nhận chuỗi thời tiết, mô hình không biết đang dự báo cho loại lưu vực nào; thuộc tính tĩnh cho nó biết điều đó.

**33 thuộc tính** (tên cột, nghĩa, đơn vị theo Bảng A2):

| Nhóm | Thuộc tính | Nghĩa | Đơn vị |
|---|---|---|---|
| Địa hình (10) | `area_calc` | diện tích lưu vực tính từ ranh giới | km² |
| | `elev_mean`, `elev_med` | độ cao trung bình, trung vị | m trên mực biển |
| | `elev_std`, `elev_ran` | độ lệch chuẩn, khoảng chênh (cao nhất − thấp nhất) của độ cao | m |
| | `slope_mean` | độ dốc trung bình | m/km |
| | `mvert_dist` | khoảng cách ngang từ điểm xa nhất của lưu vực tới trạm (trục dài) | km (Bảng A2 ghi km², có lẽ lỗi in) |
| | `mvert_ang` | góc giữa hướng bắc và trục dài | độ |
| | `elon_ratio` | tỷ số kéo dài: lưu vực tròn hay dài | — |
| | `strm_dens` | mật độ sông suối | km/km² |
| Khí hậu (11) | `p_mean` | lượng mưa trung bình ngày | mm/ngày |
| | `et0_mean`, `eta_mean` | bốc thoát hơi tham chiếu, thực tế trung bình ngày | mm/ngày |
| | `arid_1`, `arid_2` | chỉ số khô hạn (et0_mean / p_mean) và nghịch đảo của nó | — |
| | `p_season` | tính mùa và thời điểm của mưa trong năm | — |
| | `frac_snow` | tỷ lệ lượng mưa rơi dưới dạng tuyết | — |
| | `hi_prec_fr`, `hi_prec_du` | số ngày mưa lớn mỗi năm, độ dài trung bình đợt mưa lớn | ngày/năm, ngày |
| | `lo_prec_fr`, `lo_prec_du` | số ngày khô mỗi năm, độ dài trung bình đợt khô | ngày/năm, ngày |
| Lớp phủ đất (6) | `lc_dom` | mã ba chữ số của loại lớp phủ chiếm ưu thế | — |
| | `agr_fra`, `bare_fra`, `forest_fra`, `lake_fra`, `urban_fra` | tỷ lệ diện tích nông nghiệp, đất trống, rừng, mặt nước, đô thị | — |
| Thực vật (6) | `lai_max`, `lai_diff` | chỉ số diện tích lá: tháng cao nhất, chênh giữa tháng cao nhất và thấp nhất | m²/m² |
| | `ndvi_max`, `ndvi_min` | chỉ số thực vật NDVI: tháng cao nhất, thấp nhất | — |
| | `gvf_max`, `gvf_diff` | tỷ lệ thực vật xanh: tháng cao nhất, chênh cao nhất − thấp nhất | — |

Đếm: 10 + 11 + 6 + 6 = 33.

**Cách mô hình dùng.** Thuộc tính tĩnh được chuẩn hóa z-score như các đầu vào khác, rồi **ghép vào từng bước thời gian**: ở mỗi bước trong 365 bước, vectơ đầu vào gồm các biến thời tiết của ngày đó nối thêm 33 thuộc tính của lưu vực (lớp đầu vào của NeuralHydrology mặc định "concatenate the static inputs to each dynamic time step", `modelzoo/inputlayer.py` dòng 200). Ví dụ lưu vực X có `elev_mean` = 1500 m: con số 1500 (sau chuẩn hóa) xuất hiện lặp lại ở cả 365 bước.

**Ghi chú.**
- Thuộc tính khí hậu (`p_mean`, `frac_snow`…) là trung bình nhiều năm, tính sẵn trong bộ dữ liệu; kỳ năm dùng để tính chưa kiểm (cần xem tài liệu LamaH-CE, Klingler và cs., 2021) — nếu kỳ đó chồng lên kỳ test 2014–2017 thì thuộc tính chứa một phần thông tin của kỳ test.
- Bài dùng thuộc tính địa hình để giải thích sai lệch do độ phân giải: khoảng cách Wasserstein giữa ECMWF HRES và tái phân tích tương quan với độ cao trung bình (r = 0,19–0,67) và với độ dốc (r = 0,63 cho lượng mưa) ([Bài] dòng 214, 222, 224; Mục 1.11).

### 1.8. Biến và ngày — hai chiều của dữ liệu

Tệp thời tiết của một lưu vực là một bảng hai chiều: **mỗi dòng là một ngày** (1981–2017, 13.514 dòng), **mỗi cột là một biến** (đại lượng đo mỗi ngày: nhiệt độ, mưa…). "Dùng 31 biến" là nói số cột, không phải số ngày.

Số biến bài dùng — cấu hình bỏ 4 cột `ECMWF_msl`, `ECMWF_lsp`, `ECMWF_cp`, `EOBS_hu` ([LamaH] dòng 87). Bài viết đã mở rộng LamaH thêm "15 further variables from five sources" ([Bài] dòng 80), khớp với số biến ngoài ERA5-Land được dùng (7 + 1 + 2 + 5); tệp thực tế có 19 cột ngoài ERA5-Land.

| Nhóm | Có trong tệp | Dùng | Vai trò |
|---|---|---|---|
| ERA5-Land | 21 | 21 | quá khứ |
| E-OBS | 8 | 7 | quá khứ |
| MSWEP | 1 | 1 | quá khứ |
| GLEAM | 2 | 2 | quá khứ |
| Cộng quá khứ | 32 | 31 | |
| ECMWF HRES | 8 | 5 | ngày t |
| `qmean` (lưu lượng trung bình ngày đo tại trạm) | tệp trạm | 1 | quá khứ, chỉ ở biến thể có lưu lượng |
| Thuộc tính tĩnh | 62 | 33 | không đổi theo ngày |

**Cách dùng các ngày: cửa sổ trượt.** Mô hình không nhận cả chuỗi 1981–2017 một lần. Với mỗi lưu vực và mỗi ngày t trong kỳ huấn luyện, lấy 364 ngày ngay trước t làm phần quá khứ, dự báo ECMWF HRES của ngày t, thuộc tính lưu vực và nhãn qmax ngày t; mỗi bộ như vậy là một mẫu. Các cửa sổ chồng lên nhau, nên một ngày quá khứ xuất hiện trong nhiều mẫu.

Ví dụ ba mẫu liên tiếp của lưu vực X (ngày tính bằng Python, `t − 364` tới `t − 1`):

| Mẫu | Ngày t (nhãn qmax, ECMWF HRES) | Phần quá khứ 364 ngày |
|---|---|---|
| 1 | 01/01/2005 | 03/01/2004 – 31/12/2004 |
| 2 | 02/01/2005 | 04/01/2004 – 01/01/2005 |
| 3 | 03/01/2005 | 05/01/2004 – 02/01/2005 |

Cửa sổ trượt mỗi lần một ngày; ngày 01/01/2005 là ngày t của mẫu 1 nhưng là ngày quá khứ cuối cùng của mẫu 2.

Hình dạng một mẫu (biến thể không có lưu lượng; có lưu lượng thì phần quá khứ thêm `qmean` thành 32 cột):

| Phần | Kích thước | Ý nghĩa |
|---|---|---|
| Quá khứ | 364 × 31 | 364 dòng ngày, 31 cột biến |
| Ngày t | 1 × 5 | 5 biến ECMWF HRES |
| Thuộc tính | 33 | ghép vào từng bước (Mục 1.7) |
| Nhãn | 1 | qmax ngày t |

Khi huấn luyện, nhiều mẫu được xếp thành một lô (batch) B mẫu, nên phần quá khứ của cả lô là khối B × 364 × 31.

- Kỳ huấn luyện 2003–2009 có 2.557 ngày; 451 × 2.557 ≈ 1,15 triệu mẫu lưu vực–ngày ([Cơ sở] Mục 4), trước khi lọc mẫu thiếu dữ liệu.
- Chia kỳ: train 2003–2009, validation 2010–2013, test 2014–2017 ([Bài] dòng 98).
- Bài không nêu lý do bắt đầu từ 2003. Mốc này khớp với dữ liệu: ECMWF HRES chỉ có từ 01/01/2002 ([LamaH] dòng 101), mỗi mẫu cần ECMWF HRES của ngày t, và bộ lọc mẫu của mã loại mẫu có giá trị thiếu ở bất kỳ cột nào trong cửa sổ ([Cơ sở] Mục 11.1 bước 4). Các năm trước 2003 chỉ đóng vai phần quá khứ của cửa sổ.

### 1.9. Một mẫu cụ thể

Mẫu cho trạm X, ngày t = 15/05/2016 ([Cơ sở] Mục 3):

| Thành phần | Nội dung | Kích thước (mô hình tốt nhất) |
|---|---|---|
| Quá khứ | thời tiết trung bình lưu vực X và `qmean` đo tại X, từ 17/05/2015 tới 14/05/2016 | 364 ngày × 32 biến |
| Dự báo | ECMWF HRES phát 00 UTC ngày 15/05/2016, trung bình lưu vực X | 1 ngày × 5 biến |
| Thuộc tính | thuộc tính của lưu vực X | 33 |
| Nhãn | qmax ngày 15/05/2016 đo tại X, mm/ngày | 1 |

Theo lô B mẫu (B = 256), kích thước tensor là (B, 364, 32), (B, 1, 5), (B, 33) ([Cơ sở] Mục 11.2); tensor thuộc tính (B, 33) được lớp đầu vào nhân bản theo trục thời gian rồi ghép vào từng bước (Mục 1.7). Đơn vị của `qmean` đầu vào chưa kiểm được vì thư mục dữ liệu có lưu lượng của tác giả không công khai; thống kê chuẩn hóa của tác giả gợi ý vẫn là m³/s, khác đơn vị mm/ngày của nhãn ([Cơ sở] Mục 9); kiểm ở bước A4 (`01_Plan/03_Pipeline.md` Mục 7).

Ranh giới thời gian: mọi dữ liệu quá khứ dừng ở hôm trước ngày t; thông tin duy nhất về chính ngày t là bản dự báo của lần chạy 00 UTC, nhìn trước ngày đó. Lần chạy 00 UTC mang mốc giờ 00 UTC nhưng tệp dự báo được phát ra sau đó vài giờ; thời điểm phát thực tế chưa kiểm, cần khi tính kịch bản vận hành.

### 1.10. Đầu vào và nhãn

Trong học có giám sát, mô hình nhận đầu vào X, cho ra dự đoán ŷ, và hàm mất mát so ŷ với **nhãn** y — đáp án đúng; gradient của hàm mất mát dùng để sửa trọng số.

| Thành phần | Vai trò |
|---|---|
| 31 biến thời tiết quá khứ | đầu vào |
| `qmean` các ngày quá khứ | đầu vào (đã đo xong trước ngày t) |
| 5 biến ECMWF HRES ngày t | đầu vào |
| 33 thuộc tính | đầu vào |
| qmax ngày t | **nhãn duy nhất** |

Lưu lượng có mặt ở cả hai phía; phân biệt bằng thời điểm: lưu lượng các ngày trước t là đầu vào, qmax của ngày t là nhãn.

**Rò rỉ dữ liệu (data leakage)** là khi đầu vào chứa thông tin mà lúc dự báo thật chưa thể có. VD đưa `qmean` hoặc tái phân tích của chính ngày t vào đầu vào: lúc dự báo (đầu ngày t) các số này chưa tồn tại, `qmean` ngày t lại gần như chứa sẵn đáp án vì qmax và qmean cùng ngày tương quan mạnh. Mô hình rò rỉ cho điểm test rất cao nhưng không dùng được khi vận hành. Vì vậy mọi đầu vào quá khứ dừng ở t − 1, còn ngày t chỉ có dự báo ECMWF HRES.

### 1.11. Sai lệch có hệ thống và lệch miền

**Hai loại sai số.** Sai số = dự báo − thực tế.

- Sai ngẫu nhiên: lúc dư lúc thiếu, không theo quy luật, trung bình gần 0.
- Sai lệch có hệ thống (bias): lệch đều về một phía hoặc theo quy luật.

Ví dụ lượng mưa 5 ngày ở một lưu vực núi (số minh họa, mm):

| Ngày | 1 | 2 | 3 | 4 | 5 | Sai số trung bình |
|---|---|---|---|---|---|---|
| Thực tế | 10 | 12 | 8 | 15 | 20 | |
| Dự báo A | 12 | 10 | 9 | 13 | 19 | |
| Sai số A | +2 | −2 | +1 | −2 | −1 | −0,4 → ngẫu nhiên |
| Dự báo B | 7 | 9 | 6 | 11 | 14 | |
| Sai số B | −3 | −3 | −2 | −4 | −6 | −3,6 → có hệ thống: luôn thiếu, mưa càng lớn thiếu càng nhiều |

Sai số B có quy luật nên về nguyên tắc học ra được: thấy dự báo B nói 14 mm thì hiểu thực tế khoảng 20 mm. Đây là chữ "Bias" trong tên BiasCast.

**Nguồn gốc bias của dự báo thời tiết** ([Bài] dòng 64, dẫn Haiden và cs., 2024):

- độ phân giải mô hình: một ô lưới gộp cả đỉnh núi lẫn thung lũng;
- kỹ thuật đồng hóa số liệu tạo trạng thái ban đầu;
- hiệu ứng địa hình.

Bias khác nhau theo mô hình dự báo, theo biến và theo vùng.

**Ví dụ: độ phân giải và lượng mưa vùng núi.** Mô hình thời tiết gán cho mỗi ô lưới một độ cao trung bình, nên địa hình bị làm phẳng:

- mưa địa hình (mây bị sườn núi đẩy lên, lạnh đi và đổ mưa) bị tính sai lượng và vị trí;
- nhiệt độ, vốn giảm theo độ cao, lệch ở những nơi cao hoặc thấp hơn độ cao trung bình của ô (ERA5-Land cũng phải hiệu chỉnh nhiệt độ, độ ẩm, áp suất theo chênh lệch độ cao — Mục 1.5).

Bài nêu hiệu ứng địa hình ở vùng núi phức tạp khó giải quyết ở độ phân giải của mô hình dự báo ([Bài] dòng 214, dẫn Haiden và cs., 2024; Lavers và cs., 2021) và đo được điều này.

**Cách bài đo: khoảng cách Wasserstein.** Với từng lưu vực, bài so phân phối giá trị của một biến trong kỳ test giữa tái phân tích và ECMWF HRES ([Bài] dòng 176–182). Khoảng cách Wasserstein bậc 1 đo hai phân phối khác nhau bao nhiêu: với hai mẫu cùng số phần tử, sắp xếp mỗi mẫu tăng dần rồi lấy trung bình chênh lệch tuyệt đối từng cặp. Ví dụ mưa tái phân tích {0, 2, 10}, mưa ECMWF HRES {0, 1, 6}: (|0 − 0| + |2 − 1| + |10 − 6|) / 3 = 5 / 3 ≈ 1,67 mm. Bằng 0 khi hai phân phối trùng nhau; càng lớn càng lệch.

Kết quả của bài:

| Phát hiện | Nguồn |
|---|---|
| Khoảng cách Wasserstein tương quan với độ cao trung bình, từ r = 0,19 (điểm sương) tới r = 0,67 (mưa) | [Bài] dòng 214, 222 |
| Với mưa còn tương quan với độ dốc, r = 0,63 | [Bài] dòng 222 |
| Lưu vực càng cao, càng dốc thì ECMWF HRES càng lệch so với tái phân tích | [Bài] dòng 224 |

Bài không hiệu chỉnh riêng sai lệch do độ phân giải; mô hình học ngầm phần bù (Mục 1.12), có độ cao và độ dốc trong 33 thuộc tính đầu vào (Mục 1.7). Kết quả: lưu vực núi cao, nhiều tuyết lệch nhiều nhất nhưng cải thiện ít; lưu vực khô hạn cải thiện nhiều nhất. Tác giả giải thích (dùng chữ *likely*, chưa kiểm chứng) là trạng thái nhớ của LSTM giữ tín hiệu tuyết tan theo mùa nên mô hình mốc vốn đã tốt ở vùng núi ([Bài] dòng 51, 322, 328). Chi tiết ở Phần 7.

**Lệch miền (domain shift).** Mô hình thường được huấn luyện bằng dữ liệu quá khứ chất lượng cao, nhưng khi vận hành chỉ có dự báo thời tiết cho ngày cần dự báo. Dữ liệu lúc chạy thật có phân phối khác dữ liệu lúc huấn luyện, nên chất lượng giảm ([Bài] dòng 51: *"performance degradation when transitioning from high quality reanalysis to meteorological forecast data with lower accuracy"*).

Ví dụ (minh họa): mô hình học trên tái phân tích rằng 20 mm mưa ở lưu vực núi X cho qmax khoảng 8 mm/ngày. Khi vận hành, ECMWF HRES ở X thường báo thiếu (như dự báo B ở trên), trận mưa thật 20 mm chỉ được báo 14 mm, mô hình vẫn áp quy tắc cũ nên dự báo qmax thấp hơn thực tế. Mức suy giảm đo bằng NSE học ở Phần 2.

### 1.12. Cách tiếp cận truyền thống và cách của BiasCast

**Cách truyền thống** gồm hai bước: (1) hiệu chỉnh dự báo thời tiết, ví dụ nhân lượng mưa với một hệ số tính từ số liệu dài hạn (Lenderink và cs., 2007) hoặc dùng học máy để sửa dự báo mưa (XGBoost, LSTM, CNN — [Bài] dòng 66); (2) đưa dự báo đã hiệu chỉnh vào mô hình thủy văn. Bước (1) cần một "mưa đúng" làm nhãn, thường là số đo trạm hoặc tái phân tích, mà chính số đo mưa cũng có sai số (hứng thiếu do gió, sai số cảm biến); bài cho rằng sai số này có thể truyền tiếp vào dự báo dòng chảy ([Bài] dòng 68: *"it can be assumed… a source of uncertainty with potential error propagation"*). Đây là lập luận của tác giả để biện minh hướng đi; các thí nghiệm của bài (liệt kê ở [Bài] dòng 51) không gồm phương án hiệu chỉnh mưa rồi đưa vào mô hình, nên điểm này không được kiểm chứng trực tiếp.

**Cách của BiasCast:** bỏ bước hiệu chỉnh mưa riêng; mô hình nhận thẳng dự báo ECMWF HRES và được huấn luyện với nhãn là lưu lượng đo ở trạm. Lưu lượng quan trắc được xem là đáng tin hơn lượng mưa quan trắc vì là phản ứng tổng hợp của cả lưu vực và được đo liên tục tại trạm cố định, ít chịu sai số đại diện không gian hơn lượng mưa vốn phải nội suy từ mạng điểm đo ([Bài] dòng 68). Muốn dự đoán khớp lưu lượng đo, trọng số của mô hình phải tự gánh phần bù sai lệch của dự báo — mô hình học ngầm phần hiệu chỉnh, không có bước nào xuất ra "mưa đã sửa".

**Ví dụ số (minh họa).** Một lưu vực giả định với ba quy luật: lưu lượng = 0,5 × mưa thật; dự báo mưa luôn thiếu 30% (mưa dự báo = 0,7 × mưa thật); tái phân tích xem như bằng mưa thật. Mô hình một trọng số: lưu lượng dự đoán = w × mưa đầu vào.

| Ngày | Mưa thật | Mưa tái phân tích | Mưa dự báo | Lưu lượng đo (nhãn) |
|---|---|---|---|---|
| 1 | 10 | 10 | 7 | 5 |
| 2 | 20 | 20 | 14 | 10 |
| 3 | 0 | 0 | 0 | 0 |
| 4 | 30 | 30 | 21 | 15 |

- **Huấn luyện bằng tái phân tích, chạy bằng dự báo (lệch miền; bài dựng lại tình huống này ở thí nghiệm Cross-Domain, [Bài] dòng 116, 200).** Huấn luyện cho w = 0,5. Khi vận hành chỉ có mưa dự báo: ngày 1 dự đoán 0,5 × 7 = 3,5 trong khi thật là 5; ngày 4 dự đoán 10,5 trong khi thật là 15 — thiếu đều 30%, đúng bằng sai lệch của dự báo.
- **Huấn luyện thẳng bằng dự báo, nhãn là lưu lượng đo (cách của BiasCast).** w tối ưu theo bình phương tối thiểu qua gốc tọa độ: w = Σxy / Σx² = (7·5 + 14·10 + 0·0 + 21·15) / (49 + 196 + 0 + 441) = 490 / 686 ≈ 0,714. Chạy với mưa dự báo: ngày 1 cho 0,714 × 7 = 5,0; ngày 4 cho 0,714 × 21 = 15,0 — khớp. Phân tích: 0,714 = 0,5 × (1 / 0,7), tức w gộp cả quy luật của sông (0,5) và phần bù sai lệch thiếu 30% của dự báo (1 / 0,7), dù mô hình không được cho biết dự báo thiếu bao nhiêu.

**Từ ví dụ tới bài thật.** Trong bài, sai lệch không phải một hệ số cố định mà thay đổi theo vùng, theo mùa, theo biến ([Bài] dòng 64); mô hình là LSTM khoảng 85 nghìn tham số ([Cơ sở] Mục 11.2) nhận thêm 33 thuộc tính lưu vực và 364 ngày quá khứ, nên về nguyên tắc có thể học cách bù khác nhau cho từng nơi và từng mùa. Mức bù thực tế đạt được ở từng loại lưu vực là kết quả của bài (Phần 7). Cách này đòi hỏi kho dự báo lưu trữ đủ nhiều năm để huấn luyện; bộ dữ liệu có ECMWF HRES từ 2002 ([LamaH] dòng 101).

### 1.13. Tóm tắt Phần 1

| Thành phần | Nội dung |
|---|---|
| Đơn vị dự báo | một trạm đo dòng chảy; lưu vực mức A là toàn bộ vùng thượng nguồn của trạm |
| Đầu vào | trung bình trên lưu vực: 364 ngày thời tiết quá khứ (31 biến từ tái phân tích, quan trắc và sản phẩm trộn) và dự báo ECMWF HRES 5 biến cho ngày t; 33 thuộc tính lưu vực; có hoặc không có `qmean` quá khứ đo tại trạm |
| Đầu ra | qmax ngày t tại từng trạm, mm/ngày (đổi từ m³/s theo diện tích lưu vực); mỗi ngày 451 giá trị |
| Mô hình | LSTM dùng chung một bộ trọng số cho 451 lưu vực Trung Âu (Extended LamaH-CE), mỗi mẫu là một trạm; train 2003–2009, validation 2010–2013, test 2014–2017 |
| Vấn đề | huấn luyện trên dữ liệu chất lượng cao, vận hành với dự báo có sai lệch → lệch miền → dự báo kém đi |
| Cách tiếp cận | huấn luyện đầu–cuối với dự báo thật và nhãn lưu lượng đo để mô hình tự học phần bù sai lệch |

### 1.14. Câu hỏi tự kiểm tra

1. Lượng mưa đưa vào mô hình để dự báo trạm X là mưa tại điểm đặt trạm X hay mưa của vùng nào?
   Đáp án: trung bình các ô lưới nằm trong lưu vực của X, tức toàn bộ vùng thượng nguồn đổ về X (Mục 1.3).
2. Trạm X nằm ở thượng nguồn trạm Y. Mưa trên lưu vực X có góp vào đầu vào của Y không?
   Đáp án: có, vì ở mức A lưu vực Y bao trọn lưu vực X; nhưng hai lần dự báo cho X và Y tách biệt (Mục 1.3).
3. Khi vận hành, muốn dự báo qmax ngày mai, trong ba loại dữ liệu thời tiết loại nào có giá trị cho ngày mai?
   Đáp án: chỉ dự báo số trị. Quan trắc chỉ có sau khi sự việc xảy ra; tái phân tích trễ khoảng 5 ngày (Mục 1.5).
4. Trạm ghi qmax = −999 cho một ngày. Mã xử lý thế nào?
   Đáp án: đổi thành NaN ([Mã] dòng 229). Mẫu có nhãn NaN không được dùng để tính hàm mất mát.
5. Lưu vực 200 km² có qmax 20 m³/s. Bằng bao nhiêu mm/ngày?
   Đáp án: 20 / (2 × 10⁸) × 1000 × 86400 = 8,64 mm/ngày.
6. Trong ví dụ Mục 1.12, nếu dự báo mưa thừa 20% (mưa dự báo = 1,2 × mưa thật), w học được bằng bao nhiêu?
   Đáp án: 0,5 / 1,2 ≈ 0,417.
7. Trong cách của BiasCast có bước nào xuất ra lượng mưa đã hiệu chỉnh không?
   Đáp án: không. Phần bù sai lệch nằm ẩn trong trọng số; chỉ lưu lượng được so với nhãn.
