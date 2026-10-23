# Báo cáo tiến độ: ứng dụng mô hình học sâu trong dự báo lưu lượng dòng chảy

**Thời điểm cập nhật: 23/10/2026. Đợt báo cáo dự kiến: 25/10/2026.**

## 1. Kết quả chính và vị trí hiện tại của đề tài

Đề tài đã hoàn thành khảo sát dữ liệu A1, chạy lại suy luận từ trọng số tác giả ở A2 và xây dựng các mốc so sánh A3, gồm persistence và DLinear với ba hạt giống. Các kết quả tạo cơ sở để đánh giá những mô hình tiếp theo trên cùng dữ liệu, đơn vị và ngày dự báo.

| Hạng mục | Kết quả đã đạt |
|---|---|
| A1 — dữ liệu | Khảo sát 451 lưu vực; xác định độ phủ nhãn, đơn vị, cửa sổ đầu vào; tính ngưỡng bốn mức lưu lượng và ngưỡng lũ theo chu kỳ lặp lại |
| A2 — tái lập | Chạy lại sáu cấu hình từ trọng số tác giả; cả sáu đạt tiêu chí tái lập, NSE trung vị cao nhất là 0,7053 |
| A3 — mốc so sánh | Tính lại NSE, PNSE, KGE và các thành phần; persistence qmax có NSE trung vị 0,3500; DLinear đạt 0,3802 ± 0,0024 giữa ba hạt giống |
| Chẩn đoán | Giải thích các trường hợp NSE âm lớn bằng độ phủ, phương sai quan trắc và sai số dự báo; giữ đầy đủ các lưu vực trong danh sách đánh giá |
| Bộ đánh giá | Chấm A2/DLinear trên test/S0/hạn một ngày, khớp A3; có bốn mức, sự kiện lũ và kiểm định ghép cặp |

Trong các cấu hình đã đánh giá, Sequential Forecast LSTM có lưu lượng quá khứ đạt NSE trung vị cao nhất. DLinear cải thiện so với persistence theo NSE và PNSE, nhưng có KGE thấp hơn. Vì vậy, kết quả cần được đọc theo cả sai số tổng thể, tương quan, biên độ và mức lệch trung bình.

**Bộ đánh giá dùng chung** tại `Workspace/06_Evaluation/` đã chạy thành công trên dự báo A2 và DLinear với test/S0/hạn một ngày. Các kịch bản mất dữ liệu và nhiều hạn được bổ sung khi luồng mô hình xuất dự báo tương ứng. Phần tích hợp bộ nạp B2 được thực hiện sau A4 và B1. So sánh chính thức LSTM, GRU, Transformer, S4D và Mamba thuộc giai đoạn tiếp theo.

## 2. Bài toán thủy văn và dữ liệu đầu vào

### 2.1. Cần dự báo đại lượng nào?

Lưu vực là vùng đất có nước chảy về một cửa ra, nơi đặt trạm đo. Nước mưa có thể chảy trực tiếp vào sông, thấm vào đất, tích trữ dưới đất hoặc dưới dạng tuyết rồi giải phóng về sau; một phần trở lại khí quyển do bốc hơi và thoát hơi nước. Do đó, lưu lượng hôm nay phụ thuộc cả thời tiết hôm nay lẫn trạng thái tích trữ từ các ngày trước.

**Lưu lượng Q**, đơn vị m³/s, là thể tích nước đi qua mặt cắt sông trong một giây. Dữ liệu có `qmean` là lưu lượng trung bình ngày và `qmax` là lưu lượng lớn nhất trong các số đo giờ của ngày. Đầu ra cần dự báo là **qmax của ngày t**. Chọn qmax giúp bài toán gắn với đỉnh dòng chảy và đánh giá nguy cơ lũ.

Để so sánh lưu vực lớn và nhỏ, lưu lượng được quy đổi theo diện tích trạm:

**q (mm/ngày) = Q (m³/s) × 86,4 / area_gov (km²).**

Ví dụ minh họa, Q = 3 m³/s trên lưu vực 43,2 km² tương ứng q = 6 mm/ngày. Đại lượng qmax sau quy đổi vẫn biểu diễn tốc độ dòng chảy cực đại theo diện tích; nó không phải tổng lượng nước thực tế của cả ngày.

### 2.2. Mô hình được biết gì tại thời điểm dự báo?

Mô hình sử dụng lịch sử khí tượng và lưu lượng đến ngày t−1, thuộc tính lưu vực, cùng dự báo thời tiết cho ngày t. Ví dụ, để dự báo qmax ngày 01/01/2014, các quan trắc Q đầu vào chỉ được kết thúc ở 31/12/2013.

Cần phân biệt **tái phân tích khí tượng** và **dự báo khí tượng**. Tái phân tích tái dựng thời tiết đã xảy ra bằng mô hình và quan trắc; dự báo là thông tin có thể dùng trước ngày đích. Khi đánh giá dự báo vận hành, hai loại thông tin này có vai trò khác nhau. Bài cơ sở BiasCast nghiên cứu cách kết hợp lịch sử tái phân tích với dự báo ECMWF HRES và lưu lượng quan trắc gần nhất. [BiasCast, HESS 2026](https://hess.copernicus.org/articles/30/5067/2026/).

Khung Sequential Forecast LSTM của A2 gồm 364 ngày lịch sử và một bước dự báo ngày t, tổng cộng 365 bước. Một mô hình chung học từ nhiều lưu vực; các thuộc tính tĩnh giúp mô tả sự khác biệt giữa chúng.

### 2.3. Phạm vi dữ liệu và cách chia thời gian

Đề tài dùng Extended LamaH-CE và danh sách 451 lưu vực của bài cơ sở. Các thuộc tính, chuỗi khí tượng và Q được ghép bằng mã lưu vực và ngày. Chi tiết dữ liệu được trình bày trong [báo cáo A1](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore.md).

| Thành phần | Phạm vi sử dụng |
|---|---|
| Không gian | 451 lưu vực Trung Âu; diện tích trạm từ 3,9 đến 15.131 km², trung vị 101 km² |
| Thuộc tính tĩnh | 33 thuộc tính về địa hình, khí hậu và đặc điểm lưu vực |
| Khí tượng lịch sử | 31 biến thuộc ERA5-Land, E-OBS, MSWEP và GLEAM |
| Dự báo khí tượng | 5 biến ECMWF HRES trong các cấu hình đang xét; nguồn dự báo có từ 01/01/2002 |
| Huấn luyện | 2003–2009: học trọng số và fit các thống kê huấn luyện |
| Validation | 2010–2013: chọn checkpoint và cấu hình |
| Test | 2014–2017: báo cáo kết quả sau khi cố định quy trình |

Ngưỡng phục vụ đánh giá mức và sự kiện dùng giai đoạn tham chiếu 1981–2013, tức trước test. Giai đoạn tham chiếu này có vai trò khác với tập dùng fit scaler của mô hình.

## 3. Các chỉ số được hiểu và tính như thế nào?

### 3.1. Persistence — dự báo bằng số đo hôm qua

Persistence qmax dự báo **qmax(t) bằng qmax(t−1)**. Đây là mốc đơn giản nhưng có ý nghĩa thủy văn: nếu dòng chảy thay đổi chậm, số đo hôm qua đã cung cấp nhiều thông tin về hôm nay. Mô hình học cần chứng minh phần cải thiện so với mốc này.

Độ trễ được tính theo ngày lịch. Nếu hôm qua thiếu quan trắc thì persistence hôm nay thiếu; không lấy ngày có số đo gần nhất để thay. Ngày đầu test sử dụng Q ngày 31/12/2013. Trên toàn bộ test có 19 ngày–lưu vực có nhãn hôm nay nhưng thiếu qmax hôm trước.

### 3.2. NSE và PNSE — sai số được so với mốc nào?

Gọi oᵢ là quan trắc, sᵢ là dự báo, pᵢ là persistence và ō là trung bình quan trắc trên các ngày được chấm:

**NSE = 1 − Σ(sᵢ − oᵢ)² / Σ(oᵢ − ō)².**

NSE = 1 là dự báo hoàn hảo; NSE = 0 có tổng sai số bình phương ngang việc dùng hằng số ō; NSE âm nghĩa là sai số lớn hơn mốc đó. Trung bình ō ở đây là mốc dùng để diễn giải chỉ số trong kỳ chấm.

**PNSE = 1 − Σ(sᵢ − oᵢ)² / Σ(pᵢ − oᵢ)².**

PNSE dương nghĩa là mô hình hơn persistence; bằng 0 là ngang; âm là kém hơn. Ví dụ PNSE = 0,25 tương ứng tổng sai số bình phương giảm 25% so với persistence trên cùng những ngày đó.

**Ví dụ số, không phải dữ liệu thực nghiệm:** giả sử bốn ngày có quan trắc [1; 2; 3; 4], dự báo [1; 2; 2; 4] và persistence [1; 1; 2; 3], cùng đơn vị mm/ngày. Quan trắc ngày trước chuỗi là 1, nên dãy persistence đúng quy tắc trễ một ngày. Sai số bình phương của mô hình bằng 1, của persistence bằng 3; tổng bình phương độ lệch quanh trung bình quan trắc 2,5 bằng 5. Khi đó NSE = 1 − 1/5 = **0,8** và PNSE = 1 − 1/3 ≈ **0,667**.

### 3.3. KGE — dự báo sai ở nhịp, biên độ hay trung bình?

KGE dùng trong báo cáo là dạng có ba thành phần:

**KGE = 1 − √[(r − 1)² + (α − 1)² + (β − 1)²].**

| Thành phần | Cách tính | Cách đọc |
|---|---|---|
| r | Tương quan Pearson giữa dự báo và quan trắc | Cho biết hai chuỗi lên xuống đồng nhịp đến mức nào; lý tưởng bằng 1 |
| α | Độ lệch chuẩn dự báo / độ lệch chuẩn quan trắc | Nhỏ hơn 1: dao động bị co; lớn hơn 1: dao động quá mạnh; lý tưởng bằng 1 |
| β | Trung bình dự báo / trung bình quan trắc | Nhỏ hơn 1: thấp hơn trung bình thực; lớn hơn 1: cao hơn; lý tưởng bằng 1 |

Ở ví dụ bốn ngày trên, r ≈ 0,9234; α ≈ 0,9747; β = 0,9; KGE ≈ **0,8715**. Nếu dự báo luôn gấp đôi quan trắc, r vẫn bằng 1 nhưng α = β = 2 và KGE ≈ −0,4142. Tương quan cao vì vậy cần được đọc cùng sai lệch biên độ và trung bình. Với nhãn qmax, β phản ánh mức trung bình của cực đại ngày; không diễn giải trực tiếp thành sai số tổng thể tích dòng chảy.

### 3.4. Tổng hợp qua nhiều lưu vực

Mỗi lưu vực có một bộ chỉ số riêng trên kỳ chấm; sau đó lấy trung vị, trung bình và phân vị qua các lưu vực. Cách này giúp lưu vực lớn không lấn át lưu vực nhỏ chỉ vì Q tính bằng m³/s lớn hơn. Trung vị mô tả vị trí giữa của phân phối; trung bình nhạy với các trường hợp NSE âm rất lớn. Báo cáo giữ cả hai cùng đường phân phối và các hồ sơ chẩn đoán.

NSE, PNSE, KGE và các thành phần dùng chung mặt nạ ngày trong từng phép chấm. So trực tiếp các mô hình cần cùng quan trắc và cùng ngày hữu hạn. Trường hợp công thức không xác định, như quan trắc hằng đối với NSE hoặc dự báo hằng đối với r, được giữ bằng NaN và trạng thái giải thích; số lưu vực hữu hạn được báo kèm.

## 4. A1 — dữ liệu cho biết điều gì trước khi học mô hình?

### 4.1. Độ phủ và khả năng tạo mẫu

Nhãn qmax có độ phủ cao trên toàn bộ tập, nhưng thiếu tập trung ở một số trạm và giai đoạn. Một ngày–lưu vực là một mã lưu vực ở một ngày lịch.

| Kỳ | Ngày–lưu vực theo lịch | Nhãn hữu hạn | Nhãn thiếu | Tỷ lệ thiếu |
|---|---:|---:|---:|---:|
| Train | 1.153.207 | 1.112.307 | 40.900 | 3,55% |
| Validation | 658.911 | 656.281 | 2.630 | 0,40% |
| Test | 658.911 | 657.366 | 1.545 | 0,23% |

Các lưu vực 324, 600 và 623 không có nhãn trong train nhưng có nhãn ở kỳ đánh giá; lưu vực 800 chỉ có 38 ngày nhãn train. Lưu vực 55 có nhãn ở một phần test nhưng thiếu lịch sử Q cần cho cửa sổ đầu vào. Đây là những khác biệt cần giữ khi đọc kết quả theo trạm.

**Có nhãn chưa đồng nghĩa có mẫu huấn luyện.** Một mẫu còn cần lịch sử và những biến mà mô hình sử dụng. Với khung 730 ngày có Q, phép đếm A1 cho 928.114 mẫu nếu yêu cầu đủ mọi cột trên toàn khung, và 1.079.046 mẫu nếu chỉ kiểm các ô thực sự được dùng ở từng pha, tăng 150.932 mẫu. Nguyên nhân quan trọng là ECMWF HRES bắt đầu năm 2002, trong khi pha quá khứ có thể không dùng các ô ECMWF đó. Kết quả này là căn cứ kiểm bộ nạp; tác động tới kỹ năng phải được đo bằng huấn luyện theo giao thức bước B.

### 4.2. Ngưỡng bốn mức và ngưỡng lũ

A1 đã tính **q30, q80, q98 riêng cho 451 lưu vực** trên qmax tham chiếu 1981–2013. Bốn mức lần lượt là thấp (q < q30), trung bình (q30 ≤ q < q80), cao (q80 ≤ q < q98) và rất cao (q ≥ q98). Nhờ ngưỡng riêng, một trị số có thể là rất cao ở sông nhỏ nhưng bình thường ở sông lớn.

Ví dụ minh họa, nếu q30 = 0,5; q80 = 2; q98 = 5 mm/ngày, thì ngày có qmax = 6 thuộc mức rất cao. Dự báo 1,5 sẽ bị xếp mức trung bình: mô hình bỏ sót mức rất cao dù sai số liên tục chỉ là một con số. Bộ đánh giá sẽ dùng ma trận nhầm lẫn, recall và precision để thể hiện điều này.

Ngưỡng lũ **1,5; 2; 5; 10 năm** được ước lượng từ đỉnh qmax của năm thủy văn 01/10–30/09, fit Gumbel bằng L-moments. Năm có dưới một nửa số ngày quan trắc được loại khỏi mẫu đỉnh năm. Các đoạn vượt ngưỡng được tổng hợp thành cụm theo quy tắc USWRC để chuẩn bị đánh giá sự kiện. [Phương pháp và kết quả ngưỡng A1, Phần 10](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore.md#sec-24).

Chu kỳ T năm tương ứng xác suất vượt trong một năm khoảng 1/T theo mô hình thống kê, chẳng hạn mức 10 năm tương ứng khoảng 10% mỗi năm; không có nghĩa đúng mười năm mới xảy ra một lần. Ngưỡng tính được ở cả 451 lưu vực; **427 lưu vực có ít nhất 10 năm đỉnh, 24 lưu vực chỉ có 3–9 năm**, nên kết quả nhóm mẫu ngắn sẽ được báo riêng. Bốn mức theo phân vị ngày và ngưỡng theo đỉnh năm là hai cách đánh giá bổ sung nhau.

## 5. A2 — chạy lại từ trọng số tác giả

### 5.1. Phạm vi tái lập

A2 nạp trọng số tốt nhất và thống kê chuẩn hóa đã công bố của tác giả, rồi chạy suy luận trên test 2014–2017. Sáu cấu hình bao gồm hai kiến trúc xử lý quá khứ–dự báo, mỗi kiến trúc có hoặc không có Q quá khứ, và hai mốc khí tượng. Đây là tái lập suy luận; thí nghiệm huấn luyện lại ba hạt giống thuộc A4.

Sequential Forecast LSTM duy trì trạng thái khi chuyển từ lịch sử sang ngày dự báo. Encoder–Decoder dùng hai mạng cho hai pha. D_FC là mốc dùng chuỗi dự báo khí tượng; D_RA là mốc dùng tái phân tích, kể cả ở ngày đích. D_RA giúp đối chiếu tác động của nguồn thông tin, còn dự báo vận hành phải tuân theo thông tin sẵn có tại lúc phát hành.

### 5.2. Kết quả

| Cấu hình | NSE trung vị tác giả | NSE trung vị chạy lại | Lưu vực có NSE |
|---|---:|---:|---:|
| Sequential Forecast LSTM có Q | 0,7053 | 0,7053 | 450 |
| Sequential Forecast LSTM không Q | 0,6276 | 0,6276 | 451 |
| Encoder–Decoder LSTM có Q | 0,6622 | 0,6622 | 450 |
| Encoder–Decoder LSTM không Q | ≈0,569 | ≈0,569 | 451 |
| D_FC | 0,3867 | 0,3864 | 451 |
| D_RA | 0,6923 | 0,6923 | 451 |

**Cả sáu cấu hình đạt tiêu chí tái lập.** Quan trắc trùng tệp tác giả; chênh NSE trung vị không quá 0,005; sai khác dự báo ở phân vị 99,9% không quá 0,05 mm/ngày. Hai lưu vực lệch NSE trên 0,01 là 742 và 758, đều có NSE dưới −1 và phương sai quan trắc nhỏ. Mã thư viện được cố định tại commit `dda45fcf19ee7ffd481b9d192778539df46fb074` để truy lại môi trường tính toán.

![Phân phối NSE: chạy lại so với tác giả](../../Workspace/03_Baseline/Figures/A2_CDF.png)

*Hình 1. Đường liền là kết quả chạy lại, nét đứt là kết quả tác giả. Các cặp đường gần như trùng nhau. Trục dọc biểu diễn tỷ lệ lưu vực có NSE nhỏ hơn hoặc bằng giá trị trên trục ngang; vị trí 0,5 cho trung vị. Hình A2 hiển thị miền NSE khoảng −0,5 đến 1; các điểm âm xa hơn được phân tích đầy đủ trong A3.*

Lưu vực 55 không có dự báo hữu hạn ở hai cấu hình có Q do lịch sử Q thiếu; danh sách đầu vào vẫn giữ 451 lưu vực. Hồ sơ tái lập và đối chiếu dự báo từng ngày được trình bày tại [ghi chú dữ liệu, Mục 6.3](../03_Data/01_LamaHCE.md#63-tái-lập-bằng-trọng-số-tác-giả-bước-a2-kaggle-11102026).

## 6. A3 — persistence, DLinear và so sánh trên cùng ngày

### 6.1. Cách bảo đảm so sánh công bằng

A3 đọc lại dự báo sáu run A2, kiểm lịch và đối chiếu quan trắc với qmax gốc đã đổi đơn vị. Ngoài phạm vi riêng của từng run, A3 lập giao chung của sáu A2, persistence qmax và ba DLinear: **652.628 ngày–lưu vực trên 450 lưu vực có cặp hữu hạn**. Lưu vực 55 vẫn nằm trong danh sách 451 và có trạng thái không có ngày chấm ở giao này.

Các số trong bảng A3 dưới đây dùng cùng qmax nguồn, cùng ngày cho mọi mô hình. Vì phạm vi ngày khác bảng A2, một số trung vị thay đổi nhẹ, chẳng hạn Sequential Forecast LSTM không Q từ 0,6276 ở phạm vi riêng sang 0,6229 ở giao chung. Số âm được giữ; dự báo được chặn dưới ở 0 sau khi đổi ngược đơn vị theo cấu hình đã cố định, đồng thời lưu bản trước chặn để chẩn đoán.

### 6.2. DLinear đã được huấn luyện như thế nào?

DLinear phân tách chuỗi thành xu hướng bằng trung bình trượt 25 ngày và phần dao động còn lại. Hai lớp tuyến tính dự báo hai phần rồi cộng lại. A3 giữ cấu trúc từ [mã DLinear gốc](https://github.com/cure-lab/LTSF-Linear/blob/7f7a25b6d32fa0df77ce4c20d4ffeb849b5a7e47/models/DLinear.py); đây là mốc đơn giản của họ mô hình được công bố bởi [Zeng và cộng sự, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26317).

Mô hình A3 nhận một biến qmean trong 365 ngày t−365 đến t−1, dự báo qmax ngày t và có 732 tham số. Cả đầu vào và đầu ra dùng mm/ngày; scaler chung chỉ fit từ train. Các cửa sổ thiếu lịch sử được giữ dự báo NaN. Tập train có 1.094.229 mẫu từ 445 lưu vực.

Ba hạt giống 111, 222, 333 dùng cùng cấu hình Adam, batch 256, tối đa 30 epoch và dừng sớm. Checkpoint được lưu theo mỗi cải thiện nghiêm ngặt của NSE trung vị validation; quy tắc patience có ngưỡng cải thiện 0,005. Kết quả test từng epoch chỉ dùng theo dõi. Các phép kiểm kiến trúc, loss, gradient, chỉ mục ngày và nạp lại checkpoint đều đạt trước khi chạy chính thức.

| Hạt giống | Epoch đã chạy | Epoch được chọn bằng validation | NSE trung vị validation | NSE trung vị test trên giao chung |
|---|---:|---:|---:|---:|
| 111 | 6 | 4 | 0,424701 | 0,382799 |
| 222 | 6 | 2 | 0,423997 | 0,378069 |
| 333 | 10 | 5 | 0,428187 | 0,379588 |

Trung bình ± độ lệch chuẩn mẫu của ba NSE trung vị test là **0,380152 ± 0,002415**. Đây là thống kê giữa ba lần chạy, không phải điểm của dự báo tổ hợp. DLinear dùng một chuỗi lưu lượng, trong khi các A2 dùng thêm thông tin khí tượng và có cấu trúc đầu vào khác; so sánh này xác lập mốc kỹ năng, còn tác động riêng của kiến trúc sẽ được kiểm trong thí nghiệm có cùng đầu vào.

### 6.3. Kết quả chính trên giao chung

| Run | NSE trung vị | NSE trung bình | PNSE trung vị | KGE trung vị |
|---|---:|---:|---:|---:|
| Sequential Forecast LSTM có Q | **0,705259** | 0,401554 | **0,495584** | **0,767074** |
| Sequential Forecast LSTM không Q | 0,622922 | 0,565503 | 0,375311 | 0,688235 |
| Encoder–Decoder LSTM có Q | 0,662155 | 0,207468 | 0,435417 | 0,698093 |
| Encoder–Decoder LSTM không Q | 0,567347 | 0,013414 | 0,285036 | 0,640148 |
| D_FC | 0,385892 | 0,251283 | 0,076250 | 0,533138 |
| D_RA | 0,692511 | 0,387618 | 0,485222 | 0,737006 |
| DLinear 111 | 0,382799 | 0,391289 | 0,066592 | 0,519476 |
| DLinear 222 | 0,378069 | 0,424958 | 0,055647 | 0,520644 |
| DLinear 333 | 0,379588 | 0,423480 | 0,059714 | 0,513177 |
| Persistence qmax | 0,350031 | 0,356946 | 0 | 0,674939 |
| Persistence qmean, mốc phụ | 0,366131 | 0,420600 | 0,034721 | 0,457145 |

Mỗi dòng có 450 NSE hữu hạn. Riêng KGE của Encoder–Decoder không Q có 448 lưu vực hữu hạn vì dự báo hằng tại 742 và 758. Mốc qmean dùng qmean(t−1) để dự báo qmax(t), nên được giữ làm đối chiếu phụ; mốc chính của PNSE là qmax(t−1).

Sequential Forecast LSTM có Q hơn persistence theo PNSE ở **420/450 lưu vực** (93,33%). DLinear hơn ở 318, 309 và 317 lưu vực tùy hạt giống, tương ứng khoảng 69–71%. PNSE trung vị của Sequential Forecast LSTM có Q gần 0,50 cho thấy mức giảm SSE đáng kể so với số đo hôm qua tại lưu vực ở giữa phân phối PNSE.

DLinear có KGE trung vị khoảng **0,518**, thấp hơn persistence **0,675**. Trung vị α của DLinear khoảng 0,70 và r khoảng 0,63, cho thấy cần kiểm sự co biên độ và tương quan khi diễn giải kết quả. Trung vị của từng thành phần mô tả phân phối riêng; KGE phải được tính từ các thành phần của từng lưu vực trước khi tổng hợp.

### 6.4. Vì sao một số lưu vực có NSE âm rất lớn?

NSE chia sai số cho độ biến thiên quan trắc. Ở lưu vực ít biến thiên, một sai lệch trung bình tương đối nhỏ cũng có thể tạo NSE âm lớn. A1 đã ghi nhận 742 và 758 có phương sai test nhỏ; A3 cho thấy nhiều mô hình lệch nền ở hai lưu vực này, trong khi persistence bám tốt chuỗi thay đổi chậm.

Ví dụ thực nghiệm tại **758**, DLinear seed 111 có r ≈ **0,9837** nhưng NSE ≈ **−11,5639**, với β ≈ **3,9076**: chuỗi lên xuống khá đồng nhịp nhưng mức trung bình quá cao. Encoder–Decoder không Q cho dự báo hằng tại cả 742 và 758, làm r và KGE không xác định. Các trường hợp này giải thích một phần khoảng cách giữa NSE trung bình và trung vị, chẳng hạn 0,4016 so với 0,7053 của Sequential Forecast LSTM có Q.

Ở lưu vực 363, DLinear có NSE hơi âm nhưng PNSE dương do persistence còn có sai số lớn hơn. Ca này minh họa hai chỉ số đang dùng hai mốc khác nhau. Lưu vực 363 được bổ sung sau khi đọc test để chẩn đoán; không dùng để đổi cấu hình hoặc chọn checkpoint.

### 6.5. Bổ sung đánh giá theo mức và sự kiện

Bộ đánh giá dùng ngưỡng A1 đã chấm cùng giao ngày của A3. Trên 11.759 ngày–lưu vực quan trắc ở mức rất cao, SeqLSTM_Q bắt đúng 5.595 ngày, tương ứng recall **47,58%**, precision **67,58%**; persistence qmax đạt recall **40,11%**, precision **40,28%**. Ba seed DLinear có recall khoảng **21,80–22,30%**, cho thấy hạn chế ở dải lưu lượng cao dù NSE trung vị nhỉnh hơn persistence.

Với ngưỡng lũ chu kỳ hai năm và quy tắc USWRC, có **1.066 cụm quan trắc đủ điều kiện trên 366 lưu vực**. SeqLSTM_Q bắt 348 cụm, bỏ sót 718 và báo nhầm 291: POD **32,65%**, FAR **45,54%**. DLinear có POD **20,83–22,23%** và FAR **30,70–32,52%**, thể hiện đánh đổi giữa bắt được nhiều trận và báo nhầm.

Persistence đạt POD = 1 trong phép ghép sự kiện này vì chuỗi chép hôm qua làm trận xuất hiện trễ đúng một ngày, vẫn nằm trong dung sai ±1 ngày đã định. Sai số thời điểm đỉnh trung bình tuyệt đối là một ngày. Kết quả đó phải được đọc cùng phân mức đúng ngày; nó không tương đương dự báo đúng ngày hay cảnh báo sớm hoàn hảo. Giao thức được giữ cố định khi so mô hình.

Các kiểm định gồm 45 cặp run × 3 chỉ số, hiệu chỉnh Holm riêng theo từng chỉ số. Chênh lệch NSE của DLinear so với persistence có median dương ở cả ba seed; KGE thấp hơn persistence ở 449/450 lưu vực cho mỗi seed. [Bảng mức, sự kiện và kiểm định](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt.md#10-kết-quả-thực-nghiệm-trên-kaggle).

## 7. Đã trả lời được gì về thuộc tính ảnh hưởng đến dự báo?

Các kết quả hiện tại cung cấp ba loại bằng chứng. Thứ nhất, độ phủ và chiều dài lịch sử quyết định một trạm có mẫu hợp lệ hay dự báo hữu hạn hay không. Thứ hai, độ biến thiên của nhãn và khả năng dự báo bằng hôm qua ảnh hưởng trực tiếp đến cách đọc NSE, PNSE. Thứ ba, các cấu hình có Q quá khứ đạt NSE trung vị cao hơn các cấu hình tương ứng không Q trong các run đã khảo sát.

Để trả lời sâu hơn câu hỏi “đặc điểm nào khiến mô hình dự báo tốt ở lưu vực này?”, kế hoạch đánh giá sẽ liên hệ **chênh lệch kỹ năng từng lưu vực** với 33 thuộc tính, ưu tiên diện tích, độ cao, tỷ lệ tuyết và chỉ số khô hạn. Ví dụ, cần kiểm liệu lợi ích của một lõi có tập trung ở lưu vực tích trữ lâu hay phản ứng nhanh với mưa. Phân tích này gắn với [pipeline Mục 6.4 và 6.7](03_Pipeline.md#64-theo-lưu-vực), gồm tương quan theo lưu vực và giải thích mô hình ở các bước sau.

Để xác định phần đóng góp của nhóm nguồn, các kịch bản mất Q và độ trễ khí tượng S0–S4 sẽ đo mức suy giảm khi thông tin sẵn có thay đổi. So sánh lõi dùng cùng đầu vào, ngân sách tinh chỉnh và hạt giống giúp phân biệt lợi ích của dữ liệu với lợi ích của kiến trúc. Bằng chứng hiện có áp dụng cho Extended LamaH-CE; kết luận chuyển sang dataset khác cần kiểm chứng riêng.

## 8. Công việc tiếp theo và sản phẩm bàn giao

### 8.1. Bộ đánh giá dùng chung và phạm vi mở rộng

Bộ đánh giá đọc dự báo đã lưu, không cần huấn luyện hoặc chờ B1. Dự báo A2 và DLinear là dữ liệu thử đầu tiên; kết quả A3 là đối chiếu cho các chỉ số liên tục.

Đã có [mã bộ đánh giá](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt.py) và [thuyết minh](../../Workspace/06_Evaluation/01_Evaluation_LamaHCEExt.md). Lần chạy Kaggle trên Output thật A1/A2/A3 qua 49 kiểm thử và khớp A3 ở sáu chỉ số cùng số ngày trên 4.961 cặp run–lưu vực. Các CSV kết quả qua 39 kiểm tra cục bộ về công thức, tổng hợp, ghép sự kiện và kiểm định. Phạm vi đã xác nhận là test/S0/hạn một ngày.

| Nhóm công việc | Sản phẩm cần có |
|---|---|
| Hợp đồng dự báo | Mã lưu vực, ngày phát hành, ngày đích, hạn, cấu hình, seed, kịch bản, obs/sim và đơn vị; đọc được export A2/DLinear |
| Chỉ số liên tục | NSE, KGE, PNSE và r/α/β; phạm vi riêng và giao chung; bảng độ phủ, trạng thái và phân phối |
| So sánh mô hình | Ghép cặp theo lưu vực, Wilcoxon hai phía, Cohen's d ghép cặp và hiệu chỉnh Holm theo họ phép so sánh đã định |
| Bốn mức | Dùng q30/q80/q98 A1; ma trận nhầm lẫn, recall và precision từng mức; sai lệch các đoạn đường duy trì lưu lượng |
| Sự kiện lũ | Dùng ngưỡng Gumbel A1, gộp cụm theo quy tắc cố định và ghép một–một; POD, FAR, precision, F1, CSI và sai số thời điểm đỉnh |
| Theo kịch bản và hạn | Nhận dự báo S0–S4 và hạn 1–7 khi có; thống kê mức suy giảm so với S0 và kết quả từng hạn |

Các bảng ngưỡng và cụm A1, dự báo cùng thông tin cấu hình A2/DLinear được bàn giao bằng Kaggle Dataset từ Output đã lưu. Tệp nén không phải điều kiện của quy trình; các notebook giữ đầu ra trực tiếp để dùng qua Save Version và Add Input. Bước chuẩn hóa dự báo và chỉ số liên tục có thể triển khai trước trong lúc hoàn tất chia sẻ các tệp ngưỡng.

### 8.2. Phối hợp với luồng mô hình

Người 2 thực hiện A4 để đo nhiễu hạt giống của mô hình nền và B1 để hoàn thiện quy trình lưu checkpoint, thống nhất Q và bổ sung qmax quá khứ. B2 đã có bản kiểm bộ nạp độc lập đạt các phép kiểm và đếm mẫu; việc tích hợp, rà soát chung và huấn luyện B2 được tiếp tục khi A4/B1 sẵn sàng. [Phân công chi tiết](../../ASSIGNMENT.md).

Sau đó, bộ đánh giá được dùng lại cho bước B, các lõi ở bước C và các phần mở rộng. Kết quả validation phục vụ lựa chọn theo quy tắc đã định; test phục vụ báo cáo. Các phân tích sự kiện và thuộc tính sẽ bổ sung góc nhìn về bỏ sót lũ và điều kiện mà từng mô hình hoạt động tốt.

## 9. Tài liệu và kết quả đối chiếu

| Nội dung | Nguồn |
|---|---|
| Bài cơ sở, thiết kế dự báo | [Konold và cộng sự, BiasCast, HESS 2026](https://hess.copernicus.org/articles/30/5067/2026/) |
| Dữ liệu và trọng số tác giả | [Extended LamaH-CE](https://doi.org/10.5281/zenodo.17119635), [kết quả thực nghiệm BiasCast](https://doi.org/10.5281/zenodo.17292895) |
| Khảo sát, độ phủ, ngưỡng và cụm | [Thuyết minh A1](../../Workspace/02_Exploration/01_A1_LamaHCEExt_Explore.md#sec-25) |
| Đối chiếu chạy lại trọng số tác giả | [LamaH-CE, Mục 6.3](../03_Data/01_LamaHCE.md#63-tái-lập-bằng-trọng-số-tác-giả-bước-a2-kaggle-11102026), [mã A2](../../Workspace/03_Baseline/01_A2_LamaHCEExt_Reproduce.py) |
| Mốc so sánh, huấn luyện và chẩn đoán | [Thuyết minh A3](../../Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks.md#report-part-10), [mã A3](../../Workspace/03_Baseline/02_A3_LamaHCEExt_Benchmarks.py) |
| Giao thức đánh giá và kế hoạch tiếp theo | [Pipeline](03_Pipeline.md), [giải thích cách đọc chỉ số](../06_Theory/08_EvaluationFigures.md) |
| Bản thảo báo cáo theo chương | [Chương 4 — Dữ liệu](../07_Report/04_Data.md), [Chương 6 — Thí nghiệm và kết quả](../07_Report/06_ExperimentsAndResults.md) |

Các số liệu trong báo cáo là kết quả đã chạy của A1, A2, A3 và bộ đánh giá trên test/S0/hạn một ngày. Các dãy số nhỏ ghi rõ “ví dụ minh họa” chỉ dùng giải thích đơn vị và công thức. Kịch bản thiếu dữ liệu, nhiều hạn và so sánh các lõi được bổ sung theo tiến độ thực nghiệm.
