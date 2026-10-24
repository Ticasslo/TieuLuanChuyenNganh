# Quan hệ giữa đặc điểm lưu vực và kết quả dự báo — Extended LamaH-CE

Tài liệu đi cùng [mã Python](02_Evaluation_LamaHCEExt_ByBasin.py) và [notebook Kaggle](02_Evaluation_LamaHCEExt_ByBasin.ipynb), mở rộng phân tích theo lưu vực ở [pipeline Mục 6.4](../../../Document/01_Plan/03_Pipeline.md#64-theo-lưu-vực). Phân tích sử dụng các dự báo A2, DLinear, LSTM A4 và persistence đã lưu để trả lời: **mức cải thiện dự báo thay đổi thế nào theo diện tích, độ cao, tỷ phần tuyết, độ khô hạn và mức đầy đủ dữ liệu; các liên hệ có ổn định qua ba seed LSTM hay không?**

Bản phân tích đã chạy trên Kaggle với **14 run, 18 phép so**, giữ 451 ID và giao chung 652.628 ngày–lưu vực. **Ba seed LSTM A4 giữ liên hệ âm của ΔNSE với độ cao và tuyết, dương với độ khô hạn**, đồng thời cho thấy biến thiên đáng kể ở một số lưu vực. Phần mở rộng qua 41 kiểm thử Kaggle và 492 kiểm CSV cục bộ. [Kết quả A4](#12-kết-quả-phân-tích-14-run-có-a4); Mục 11.1–11.7 trình bày bộ 11 run gốc.

Trên 450 lưu vực có ngày chấm, **mức cải thiện NSE của SeqLSTM_Q so persistence có liên hệ âm với độ cao trung bình (ρ = −0,422) và tỷ phần tuyết (ρ = −0,263), liên hệ dương với chỉ số khô hạn (ρ = 0,211)**. Diện tích có liên hệ yếu trong phép so này (ρ = −0,075), nhưng liên hệ dương rõ hơn với chênh lệch SeqLSTM có Q so không Q (ρ = 0,274). Kết luận phụ thuộc vào mô hình, mốc tham chiếu và chỉ số đang xét. [Kết quả thực nghiệm](#11-kết-quả-thực-nghiệm-và-kiểm-chứng) trình bày hệ số, phân nhóm, số trận và độ nhạy.

Các hệ số mô tả mối liên hệ giữa các lưu vực. Những thuộc tính như độ cao, tuyết và thực vật có thể cùng biến đổi theo vùng, nên hệ số riêng lẻ chưa xác định đóng góp nhân quả của một đầu vào. Thí nghiệm che nguồn và giải thích mô hình cung cấp bằng chứng bổ sung ở các bước tiếp theo.

## 1. Phạm vi và thuật ngữ

**S0** là kịch bản không che thêm đầu vào; dữ liệu nguồn vẫn có thể thiếu. **Hạn một ngày** là dự báo ngày kế tiếp sau ngày cuối của lịch sử quan trắc. Một **ngày–lưu vực** là một ngày tại một ID. **Mặt nạ** là ma trận True/False chọn các ngày–lưu vực được chấm; **giao chung** chỉ giữ những vị trí có quan trắc, persistence và dự báo của mọi run cùng hữu hạn.

Bảng chính dùng test 2014–2017, S0, hạn một ngày và phạm vi `common_source_all` của bộ đánh giá. Tập chính thức giữ đủ 451 ID, kể cả ID không có ngày chấm hoặc chỉ số không xác định. Với nguồn A3 đã kiểm, giao chung có 652.628 ngày–lưu vực trên 450 ID có ngày hợp lệ. Khi chạy, các số này được đối chiếu với đầu vào thực tế thay vì dùng thay cho phép kiểm.

Mỗi hàng tương quan sử dụng các ID có đủ thuộc tính và outcome tương ứng, kèm số lượng được dùng. Không loại NSE âm, không chọn riêng các lưu vực được cải thiện cho bảng chính và không đổi tập ID để làm hệ số lớn hơn.

## 2. Bốn nguồn Input và cách chạy

Notebook chạy bằng CPU, đọc đầu vào dưới `/kaggle/input` và ghi trực tiếp vào `/kaggle/working/basin_analysis_output`. Gắn dataset gốc và Output của các notebook đã Save Version làm Input:

| Nguồn | Nội dung dùng trong phân tích |
|---|---|
| Dataset `lamah-ce-ext` | `A_basins_total_upstrm/1_attributes/Catchment_attributes.csv` và `D_gauges/1_attributes/Gauge_attributes.csv` |
| Output A1 | `A1_P3_static_attribute_sources.csv`; bảng topology nếu có |
| Output A3 | `A3_DLinear_data_audit.npz`, `A3_final_evaluation_masks.npz`, `A3_DLinear_loss_stats.csv` |
| Output bộ đánh giá | `Evaluation_summary.json`, `Evaluation_protocol.json`, `Evaluation_metrics.csv`, `Evaluation_events.csv`; với A4 cần thêm `Evaluation_A4_input_check.csv` và `Evaluation_A4_contract_check.csv` |

Output A1 xác nhận ánh xạ 33 thuộc tính đã dùng; giá trị thuộc tính được đọc từ bảng gốc. Output A3 cung cấp lịch, ID, diện tích, Q số thô và mặt nạ đánh giá. Output bộ đánh giá cung cấp chỉ số theo lưu vực và số trận đã chấm bằng giao thức cùng ngày bắt đầu. Không cần gắn lại Output A2/A4 Compare hoặc chạy lại mô hình.

Để phân tích A4, dùng Output bộ đánh giá **đã Save Version sau lần chấm 14 run** và cập nhật phiên bản Input đó trong notebook 02. Các CSV tổng hợp như `Evaluation_summary.csv` hoặc `Evaluation_events_pooled.csv` không thay được bảng từng lưu vực `Evaluation_metrics.csv` và `Evaluation_events.csv`. Phần 2 in số run, số phép so và cờ A4; lần phân tích mở rộng phải báo **14 run, 18 phép so, A4=True**.

Nguồn 11 run gốc được nhận từ giao thức đánh giá 2; nguồn 14 run cần giao thức từ 3, xác nhận đủ ba hợp đồng A4 Compare Version 2 và các hàng kiểm nguồn đều đạt. Metadata seed trong chỉ số/sự kiện phải khớp bảng hợp đồng. Thiếu một seed hoặc có run chưa khai báo làm chương trình dừng. Cả hai chế độ đều dùng test/S0/hạn một ngày, cùng giao chung và dung sai khớp trận chính bằng 0. Nguồn trùng tên cần được chỉ rõ; không tự chọn một phiên bản giữa nhiều ứng viên.

Mã `.py` là nguồn chỉnh sửa chính. Repo giữ `.ipynb` chính thức Version 1 có output của lần phân tích 11 run; nguồn `.py` và kết quả CSV tại Mục 12 đã bao gồm A4. Tệp `.ipynb` tải từ phiên Save & Run All tương ứng với lần phân tích 14 run được bổ sung khi có. Notebook trình bày theo phần, đề mục Markdown và cell code theo chức năng; không thay bản chính thức bằng notebook trống output. Kết quả ghi trực tiếp, không đóng ZIP.

## 3. Ghép dữ liệu và kiểm tính nhất quán

Các bảng ghép theo khóa ID, không theo vị trí dòng. Danh sách ID, lịch ngày và thứ tự các trục trong NPZ phải khớp trước khi tính thống kê. Khóa trùng, ID thiếu/thừa, dữ liệu vô cực hoặc schema không rõ được báo ở bước kiểm nguồn.

Mặt nạ `common_source_all` đọc từ A3 được dùng để đối chiếu số ngày đánh giá của từng run. Các hàng NSE/PNSE dùng đúng cùng phạm vi; không lấy NSE native của hai run có ngày chấm khác nhau rồi trừ. Các run phải có cùng quan trắc canonical, số ngày và mẫu số đánh giá trong phép so. Lần chấm 14 run hiện giữ nguyên giao A3; nếu một nguồn mới làm giảm giao, phép kiểm sẽ dừng để cập nhật mặt nạ chẩn đoán, không âm thầm dùng lại thống kê trên mặt nạ cũ.

Ngưỡng lũ và sự kiện được giữ theo Output bộ đánh giá. Phân tích này không fit lại ngưỡng, không đổi quy tắc gộp cụm và không nới dung sai để tăng số trận bắt được. Bảng chính chỉ lấy cụm USWRC, chu kỳ lặp lại hai năm và dung sai 0.

## 4. Thuộc tính tĩnh và đơn vị

Danh sách 33 trường là hợp đồng đầu vào đã kiểm tại [A1 Mục 17.4–17.6](../../02_Exploration/01_A1_LamaHCEExt_Explore/01_A1_LamaHCEExt_Explore.md#174-xác-nhận-33-thuộc-tính-từ-24-cấu-hình). Toàn bộ giá trị lấy từ `Catchment_attributes.csv`:

| Nhóm | Thuộc tính |
|---|---|
| Địa hình và hình dạng, 10 trường | area_calc, elev_mean, elev_med, elev_std, elev_ran, slope_mean, mvert_dist, mvert_ang, elon_ratio, strm_dens |
| Khí hậu, 11 trường | p_mean, et0_mean, eta_mean, arid_1, arid_2, p_season, frac_snow, hi_prec_fr, hi_prec_du, lo_prec_fr, lo_prec_du |
| Lớp phủ, 6 trường | lc_dom, agr_fra, bare_fra, forest_fra, lake_fra, urban_fra |
| Thực vật, 6 trường | lai_max, lai_diff, ndvi_max, ndvi_min, gvf_max, gvf_diff |

Bảng tương quan chính dùng **31 trường**, loại mã phân loại `lc_dom` và góc vòng `mvert_ang` khỏi phép tương quan theo thứ tự số thông thường. Đây là lựa chọn phương pháp phân tích; hợp đồng 33 đầu vào của các mô hình tái lập được giữ nguyên.

| Trường | Ý nghĩa và đơn vị |
|---|---|
| area_calc | Diện tích tính từ ranh giới lưu vực, km² |
| area_gov | Diện tích cơ quan cung cấp tại trạm, km²; lấy từ bảng Gauge, dùng đổi Q |
| elev_mean, elev_med, elev_std, elev_ran | Trung bình, trung vị, độ lệch chuẩn và khoảng độ cao, m |
| slope_mean | Độ dốc trung bình, m/km |
| mvert_dist | Chiều dài trục từ điểm xa nhất tới trạm, km |
| mvert_ang | Hướng trục dài lưu vực so với Bắc, độ |
| elon_ratio, strm_dens | Chỉ số hình dạng không thứ nguyên; mật độ mạng sông km/km² |
| p_mean, et0_mean, eta_mean | Lượng mưa, bốc thoát hơi tham chiếu và bốc thoát hơi thực tế trung bình, mm/ngày |
| arid_1 | ET0 từ GCD v2 chia mưa ERA5-Land, không thứ nguyên |
| arid_2 | Nghịch đảo chỉ số khô hạn từ GCD v2, không thứ nguyên |
| p_season | Chỉ số mùa mưa không thứ nguyên; dấu mang ý nghĩa mùa |
| frac_snow | Tỷ phần lượng mưa rơi dưới dạng tuyết, không phải tỷ lệ ngày có tuyết |
| hi_prec_fr, lo_prec_fr | Tần số ngày mưa lớn/ngày khô, ngày/năm |
| hi_prec_du, lo_prec_du | Thời lượng trung bình các đợt tương ứng, ngày |
| Các trường hậu tố fra thuộc lớp phủ | Tỷ phần diện tích, không phải phần trăm |
| Các chỉ số LAI, NDVI và GVF | Chỉ số thực vật không thứ nguyên từ các trung bình tháng dài hạn |

`area_calc` và `area_gov` có nguồn và mục đích khác nhau. Q số thô được đổi theo q(mm/ngày) = Q(m³/s) × 86,4 / area_gov(km²), cùng quy ước A3. Không thay diện tích đổi đơn vị bằng diện tích dùng làm predictor.

Theo [LamaH-CE Mục 5](https://essd.copernicus.org/articles/13/4529/2021/#section5), nhiều chỉ số khí hậu được tính trên 1989–2009, ET0 từ GCD v2 có kỳ 1970–2000 và các chỉ số thực vật dùng kỳ nguồn dài hạn có thể tới 2020. Đây là metadata có sẵn của dataset, không phải toàn bộ thống kê chỉ fit từ train của thí nghiệm. Phạm vi nguồn được giữ khi diễn giải; các diagnostics mới từ chuỗi Q được tách riêng theo kỳ.

## 5. Các phép so sánh và ba outcome

Mỗi phép so có hướng cố định: **run bên trái trừ run tham chiếu**. Giá trị dương biểu thị cải thiện theo outcome đang xét.

| Nhóm | Các phép so |
|---|---|
| Sáu run A2 so persistence qmax | SeqLSTM_Q, SeqLSTM_noQ, EncDec_Q, EncDec_noQ, D_FC và D_RA; mỗi run trừ Persistence_qmax |
| Ba seed DLinear so persistence qmax | DLinear_seed_111, DLinear_seed_222 và DLinear_seed_333; mỗi run trừ Persistence_qmax |
| Có/không Q trong cùng kiến trúc | SeqLSTM_Q − SeqLSTM_noQ; EncDec_Q − EncDec_noQ |
| Hai miền khí tượng tham chiếu | D_RA − D_FC |
| Ba seed LSTM A4 so persistence qmax | SeqLSTM_Q_A4_seed111/222/333, mỗi run trừ Persistence_qmax |
| Ba seed LSTM A4 so trọng số tác giả | Mỗi run A4 trừ SeqLSTM_Q |

Nguồn 11 run có 12 phép so; nguồn 14 run có **18 phép so**, tương ứng **1.674 hàng tương quan = 31 thuộc tính × 18 phép so × 3 outcome**. Mười hai phép so gốc giữ nguyên cách tính, seed hoán vị và họ Holm. Thêm A4 không làm đổi p của các họ liên hệ thuộc tính cũ khi dữ liệu của các phép so đó giữ nguyên. Đây là các họ 31 thuộc tính, khác họ 78 cặp mô hình trong notebook đánh giá 01.

D_RA dùng tái phân tích ở ngày đích, nên đây là mốc tham chiếu về nguồn thông tin. Các cặp A2 có/không Q cũng chỉ mô tả những run đã có, chưa tách riêng tác động của đầu vào khỏi nhiễu hạt giống và những khác biệt cấu hình đi kèm.

**ΔNSE** là NSE của run trừ NSE tham chiếu trên cùng ngày. Ví dụ NSE 0,60 so 0,40 cho ΔNSE = 0,20. Đây là chênh lệch điểm, không phải cải thiện 20%.

**ΔPNSE** là PNSE của run trừ PNSE tham chiếu. PNSE = 1 − SSE mô hình / SSE persistence, trong đó SSE là tổng bình phương sai số trên mặt nạ đã cố định. PNSE của persistence qmax bằng 0 khi mẫu số dương, nên trong các phép so persistence, ΔPNSE bằng PNSE của run. Các phép so giữa hai mô hình vẫn lấy chênh hai PNSE. Mẫu số bằng 0 cho trạng thái không xác định, không thêm epsilon.

**ΔPOD hai năm** là POD của run trừ POD tham chiếu ở ngưỡng lũ hai năm, cụm USWRC và dung sai 0. POD = h/(h+m), với h là số trận bắt đúng và m là số trận bỏ sót. Phép so kiểm cùng số trận thật đủ điều kiện trước khi trừ. Lưu vực không có trận thật đủ điều kiện có POD không xác định, không được gán POD bằng 0 để đưa vào tương quan.

Tại hạn một ngày, dung sai 0 yêu cầu trận dự báo bắt đầu cùng ngày trận thật. Bảng ±1 ngày của bộ đánh giá là phân tích phụ riêng. Không trộn hai mức dung sai vào cùng outcome.

## 6. Tương quan, permutation và Holm

Spearman đo xu hướng đơn điệu giữa một thuộc tính và một outcome theo ID. Hệ số ρ gần 1 biểu thị hai đại lượng thường tăng cùng nhau; gần −1 biểu thị xu hướng ngược chiều. Hệ số gần 0 không loại trừ quan hệ phi tuyến hoặc tác động có điều kiện.

Tất cả điểm hữu hạn được giữ, kể cả NSE âm lớn và mức cải thiện âm. Mỗi cặp thuộc tính–outcome có số ID, số giá trị khác nhau và trạng thái tương ứng. Chuỗi hằng hoặc không đủ dữ liệu không có hệ số/p-value hợp lệ.

Kiểm định permutation dùng 9.999 lần hoán vị, seed 20261023. Trong mỗi lần, outcome được ghép lại ngẫu nhiên với các ID, giữ nguyên giá trị thuộc tính. P-value hai phía theo trị tuyệt đối là (1 + số lần |ρ hoán vị| ≥ |ρ quan sát|) / (9.999 + 1). Quy ước cộng 1 giữ p-value Monte Carlo dương; đây là kiểm định theo |ρ|, không mặc định đồng nhất với mọi quy ước p hai phía của thư viện.

[SciPy về Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) lưu ý độ chính xác của p-value xấp xỉ khi mẫu nhỏ; [tài liệu permutation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) mô tả hoán vị cách ghép cặp cho các phép tương quan. Với 451 lưu vực, phép hoán vị cung cấp tham chiếu bổ sung cho hệ số, trong khi hạn chế phụ thuộc không gian vẫn cần được giữ khi diễn giải.

Holm điều chỉnh trong từng **contrast × outcome**, gồm 31 phép thử thuộc tính tĩnh đã định trước. Chỉ p-value hữu hạn tham gia số lượng phép thử; hàng NaN giữ trạng thái và không làm tăng m. Bảng xuất ghi family, m và p đã điều chỉnh. Các family khác nhau được trình bày riêng; mức ý nghĩa trong một family không có nghĩa đã kiểm soát đồng thời mọi kết luận của toàn notebook.

Các lưu vực có thể lồng nhau và cùng chịu điều kiện khí hậu vùng. Hoán vị độc lập theo ID không mô hình hóa cấu trúc phụ thuộc này. **ρ, hướng liên hệ và số ID là kết quả mô tả chính; p-value là bằng chứng thăm dò theo giả định trao đổi được giữa ID.** Không dùng thứ hạng p-value làm thứ hạng đóng góp nhân quả của đầu vào.

## 7. Diagnostics từ chuỗi Q và độ phủ train

NPZ audit A3 cung cấp qmean/qmax số thô trên lịch đầy đủ. Các thống kê train được tính trong 2003–2009, không gồm năm ngữ cảnh 2002. Giá trị Q được đổi sang mm/ngày bằng area_gov; số 0 và các khoảng thiếu được giữ đúng lịch.

Nhóm diagnostics gồm độ phủ nhãn và lịch sử đầu vào, số cửa sổ DLinear đủ điều kiện, phân phối Q và mức thay đổi giữa train/test. Số ngày có nhãn, số cửa sổ hoàn chỉnh và số ngày chấm test là ba đại lượng khác nhau; lưu vực có nhãn train không nhất thiết có đủ 365 ngày qmean liên tục trước nhãn để tham gia DLinear.

Các thống kê từ Q có thể gồm trung bình, độ lệch chuẩn, biến thiên tương đối, tỷ lệ bằng 0 và tương quan trễ một ngày. Tương quan trễ chỉ dùng cặp ngày liền nhau trên lịch gốc; không nối hai phía khoảng thiếu sau khi bỏ NaN. Thống kê có mẫu số bằng 0 hoặc không đủ mẫu giữ trạng thái không xác định.

Diagnostics test và chênh train→test giúp đọc các trường hợp như ít nhãn train hoặc phương sai test rất nhỏ. Chúng được tính để mô tả kết quả, không dùng fit scaler, chọn checkpoint, sửa cấu hình hoặc đổi ngưỡng. Riêng liên hệ giữa NSE và phương sai Q test còn có thành phần toán học vì phương sai nằm trong mẫu số NSE.

Bảng liên hệ với diagnostics chỉ báo ρ và số ID, **không thêm p-value** vào family 31 thuộc tính tĩnh. Bảng độ lệch chuẩn loss A3 và số cửa sổ đã lưu được dùng để kiểm lại các phép tính có cùng định nghĩa.

## 8. Nhóm thuộc tính, biến phân loại và nhiễu seed

Bốn thuộc tính ưu tiên cho hình và bảng nhóm là `area_calc`, `elev_mean`, `frac_snow`, `arid_1`. Các mốc tứ phân vị được tính một lần trên toàn bộ 451 giá trị thuộc tính trước khi ghép outcome. Ties có thể làm các mốc trùng; khi đó số khoảng thực tế giảm và được ghi rõ. Các run dùng cùng biên nhóm, kèm số ID có outcome hữu hạn trong từng nhóm.

`lc_dom` được mô tả theo lớp CORINE, giữ mã và số lượng từng lớp. `mvert_ang` được chia thành bốn cung [0°, 90°), [90°, 180°), [180°, 270°), [270°, 360°); 360° cùng nhóm với 0°. Góc thiếu hoặc ngoài [0°, 360°] có nhóm báo riêng. Các bảng này mô tả phân phối outcome trong nhóm; không áp thứ tự lớn–nhỏ vật lý cho mã lớp hoặc bỏ qua tính vòng của góc.

Ba seed của mỗi họ DLinear và LSTM A4 được giữ thành các run riêng. Hai bảng độ phân tán tính trung bình, độ lệch chuẩn mẫu, min và max của các hiệu chỉ số so persistence tại cùng lưu vực, chỉ tổng hợp khi đủ ba giá trị hợp lệ. Ba seed không tạo thêm ba lưu vực độc lập. Trung bình ba chỉ số cũng không phải chỉ số của một dự báo ensemble; notebook không tạo chuỗi dự báo ensemble từ các score. Độ phân tán giữa DLinear và LSTM mô tả hai cấu hình/đầu vào đã chạy, không cô lập ảnh hưởng của kiến trúc.

Bảng `BasinAnalysis_seed_association_stability.csv` đối chiếu ρ của từng thuộc tính–outcome qua ba seed so persistence, tách hai họ mô hình. Với đủ 14 run, bảng có **186 hàng = 2 họ × 31 thuộc tính × 3 outcome**. Mỗi hàng giữ riêng ρ từng seed, số ID min/max, trung bình/std mẫu và min/max của ρ; trạng thái phân biệt cùng dương, cùng âm, cùng bằng 0, lẫn dấu/chạm 0 và seed không xác định. Cùng dấu không đồng nghĩa hiệu ứng lớn hoặc có ý nghĩa thống kê. Bảng này không gộp p, không tạo thêm phép thử từ ba seed và không coi range hệ số là khoảng tin cậy.

## 9. Kiểm độ nhạy với lưu vực lồng nhau

Nếu Output A1 có `A1_P3_final_selected_topology.csv`, phân tích đọc cột `has_known_selected_upstream` sau khi kiểm ID, kiểu boolean và trạng thái đường đi mạng sông. Giá trị thiếu không được tự chuyển thành False.

Nhóm đầu nguồn trong tập chọn gồm các ID không có lưu vực được chọn khác ở thượng lưu. Định nghĩa này khác `HIERARCHY = 1` trên toàn bộ mạng gốc. A1 đã kiểm 322 ID đầu nguồn theo tập chọn; con số khi chạy được đối chiếu với bảng nguồn, không ép bằng giá trị đã biết.

Phân tích độ nhạy báo lại ρ và số ID của bốn thuộc tính ưu tiên trên nhóm này. Việc giảm lồng nhau giúp đánh giá mức ổn định của xu hướng, nhưng các lưu vực còn có thể cùng vùng khí hậu và gần nhau trong không gian. Bảng độ nhạy không được diễn giải là mẫu hoàn toàn độc lập.

## 10. Output và cách trình bày kết quả

Các bảng dùng tiền tố `BasinAnalysis_`; hình nằm trong `Figures/` với cùng tiền tố. Output gồm các nhóm:

| Nhóm đầu ra | Nội dung |
|---|---|
| Nguồn và kiểm thử | Thông tin nguồn, kiểm ID/lịch/mặt nạ, kiểm thử ví dụ số và trạng thái thực thi |
| Thuộc tính và diagnostics | Giá trị 451 ID, đơn vị, độ phủ, thống kê train/test và cờ không xác định |
| Contrast | Các chênh lệch theo lưu vực, số ngày và số trận dùng trong phép so |
| Tương quan tĩnh | 31 thuộc tính × 18 contrast × 3 outcome khi có A4; nguồn 11 run dùng 12 contrast; kèm n, ρ, permutation p và Holm |
| Mô tả nhóm | Biên tứ phân vị, số lượng và phân phối outcome; lớp CORINE và cung góc |
| Diagnostics và độ nhạy | Tương quan mô tả, độ phân tán ba seed và nhóm đầu nguồn nếu nguồn hợp lệ |
| Hình | Heatmap hệ số, biểu đồ bốn thuộc tính ưu tiên và phân phối theo nhóm |

Các tệp CSV cần giữ gồm `BasinAnalysis_checks.csv`, `BasinAnalysis_attributes_and_diagnostics.csv`, `BasinAnalysis_attribute_quality.csv`, `BasinAnalysis_paired_contrasts.csv`, `BasinAnalysis_static_associations.csv`, `BasinAnalysis_diagnostic_associations.csv`, `BasinAnalysis_quartile_definitions.csv`, `BasinAnalysis_grouped_skill.csv`, `BasinAnalysis_categorical_summary.csv`, `BasinAnalysis_DLinear_seed_spread.csv`, `BasinAnalysis_topology_sensitivity.csv` và `BasinAnalysis_input_manifest.csv`. Hai JSON là `BasinAnalysis_protocol.json` và `BasinAnalysis_summary.json`.

Phần độ ổn định bổ sung `BasinAnalysis_seed_association_stability.csv` và, khi có A4, `BasinAnalysis_LSTM_A4_seed_spread.csv`. Lần chấm 14 run xuất **14 CSV, hai JSON và năm PNG**. Cờ `complete` chuyển false từ đầu mỗi lần chạy và chỉ thành true sau khi các bảng/hình được xuất đầy đủ.

| Hình PNG trong Figures | Nội dung |
|---|---|
| BasinAnalysis_static_delta_NSE_rho.png | Hệ số Spearman của 31 thuộc tính với ΔNSE ở đủ 18 phép so khi có A4 |
| BasinAnalysis_priority_scatter.png | Bốn thuộc tính ưu tiên và ΔNSE, giữ mọi điểm âm; ba cặp hiển thị được chọn trước |
| BasinAnalysis_quartile_delta_NSE.png | ΔNSE trung vị theo cùng nhóm thuộc tính, tách hàng DLinear và LSTM A4, giữ riêng từng seed |
| BasinAnalysis_quartile_flood_POD.png | POD cộng dồn số trận theo cùng nhóm thuộc tính; đọc cùng mẫu số trong CSV |
| BasinAnalysis_seed_spread.png | Phân phối std mẫu của ba hiệu chỉ số tại từng ID, so hai họ DLinear/LSTM A4; giữ ngoại lệ |

Khi gửi kết quả mới, giữ hai JSON, bảng checks, static_associations, grouped_skill, quartile_definitions, hai bảng seed_spread, seed_association_stability, attribute_quality, topology_sensitivity và năm hình. Các tệp trực tiếp trong `Tables/BasinAnalysis_*` giữ bộ 11 run gốc; bộ 14 run được lưu riêng ở [Tables/BasinAnalysis_A4_Integrated](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_received_manifest.csv) và `Figures/BasinAnalysis_A4_Integrated/`.

Khi trình bày kết quả cần nêu tên contrast, outcome, hướng và độ lớn hệ số, số ID, khoảng đặc điểm và độ ổn định giữa các seed/nhóm. Chẳng hạn, “mức cải thiện NSE của SeqLSTM_Q so persistence có xu hướng lớn hơn ở các lưu vực có chỉ số khô hạn cao hơn” mô tả đúng đại lượng đã đo. Khẳng định “độ khô hạn làm NSE cao” đòi hỏi bằng chứng khác.

## 11. Kết quả thực nghiệm và kiểm chứng

### 11.1. Phạm vi thực nghiệm

Các kết quả trong Mục 11.1–11.7 thuộc nguồn **11 run gốc**, với Output bộ đánh giá giao thức 2, A1, A3 và dataset gốc trên Kaggle. Có **30 kiểm thử đạt**, giữ 451 ID, trong đó 450 ID có **652.628 ngày–lưu vực** trên giao chung test/S0/hạn một ngày. ID 55 vẫn có mặt với số ngày chấm bằng 0. Kiểm train, mặt nạ và mẫu số quan trắc đạt trước khi tính tương quan.

Phân tích có **1.116 hàng = 31 thuộc tính × 12 phép so × 3 outcome**; toàn bộ 1.116 hệ số chính hữu hạn. ΔNSE và ΔPNSE dùng 450 ID; ΔPOD dùng 366 ID có trận thật đủ điều kiện. Cả 33 thuộc tính đều đủ giá trị trên 451 ID, với 31 trường dùng cho tương quan. Thời gian chạy Kaggle khoảng 75,6 giây. [Thông tin lần chạy](Tables/BasinAnalysis_summary.json), [giao thức](Tables/BasinAnalysis_protocol.json), [độ phủ thuộc tính](Tables/BasinAnalysis_attribute_quality.csv).

### 11.2. Đặc điểm liên hệ với mức cải thiện NSE

Các hệ số dưới đây đều là Spearman giữa thuộc tính và **NSE bên trái trừ NSE bên phải**, mỗi hệ số dùng 450 lưu vực.

| Phép so | Diện tích | Độ cao trung bình | Tỷ phần tuyết | Chỉ số khô hạn arid_1 |
|---|---:|---:|---:|---:|
| SeqLSTM_Q − persistence qmax | −0,075 | −0,422 | −0,263 | 0,211 |
| EncDec_Q − persistence qmax | −0,092 | −0,331 | −0,209 | 0,111 |
| DLinear 111 − persistence qmax | −0,297 | −0,246 | −0,133 | 0,226 |
| DLinear 222 − persistence qmax | −0,290 | −0,262 | −0,157 | 0,238 |
| DLinear 333 − persistence qmax | −0,287 | −0,263 | −0,159 | 0,236 |
| SeqLSTM_Q − SeqLSTM_noQ | 0,274 | 0,096 | 0,008 | 0,105 |
| EncDec_Q − EncDec_noQ | 0,194 | 0,170 | 0,053 | −0,058 |
| D_RA − D_FC | 0,184 | −0,426 | −0,453 | 0,491 |

SeqLSTM_Q và EncDec_Q có mức cải thiện so persistence thấp hơn ở các lưu vực cao hơn hoặc có tỷ phần tuyết lớn hơn. Riêng lợi ích của run có Q so run không Q tăng theo diện tích: ρ = 0,274 với SeqLSTM và 0,194 với EncDec. Đổi mốc tham chiếu làm thay đổi câu hỏi và hướng liên hệ; vì vậy không thể gán một dấu chung cho “ảnh hưởng của diện tích”. Các run A2 có/không Q hiện có một seed, nên phép so còn chứa biến thiên giữa các lần huấn luyện.

Trong 31 thuộc tính, `ndvi_max` có |ρ| lớn nhất với ΔNSE của SeqLSTM_Q so persistence, **ρ = 0,500**, tiếp theo là `et0_mean` **0,474** và `eta_mean` **0,434**. Đây là các mối liên hệ đơn biến với mức cải thiện, không phải thứ hạng đóng góp của đầu vào trong mô hình. Với D_RA − D_FC, `bare_fra` có ρ = −0,520 và `arid_1` có ρ = 0,491; D_RA dùng tái phân tích tại ngày đích nên chỉ dùng làm mốc tham chiếu về nguồn thông tin.

Các p-value của những liên hệ mạnh thường chạm sàn Monte Carlo 0,0001; p sau Holm được lưu theo từng family. Những giá trị bằng nhau ở sàn không cho phép xếp thứ hạng độ mạnh bằng p. [Toàn bộ hệ số, n và p](Tables/BasinAnalysis_static_associations.csv).

![Liên hệ giữa thuộc tính và chênh lệch NSE](Figures/BasinAnalysis_static_delta_NSE_rho.png)

Màu thể hiện hướng và độ lớn hệ số. Mỗi cột là một phép so cụ thể; toàn bộ các mức cải thiện âm vẫn được giữ.

### 11.3. Khoảng đặc điểm và mức cải thiện theo nhóm

Biên nhóm tính trên thuộc tính của cả 451 ID trước khi ghép điểm số. Q1 là khoảng thấp nhất, Q4 là khoảng cao nhất; số ID giữa các nhóm có thể khác nhau do giá trị bằng nhau ở biên. Bảng dưới chỉ trình bày Q1/Q4; CSV giữ đủ bốn nhóm, số ID, khoảng tứ phân vị và ba seed.

| Thuộc tính và khoảng | ID có ΔNSE | Median ΔNSE SeqLSTM_Q − persistence | Median ΔNSE DLinear 111 − persistence |
|---|---:|---:|---:|
| Diện tích Q1: [3,972; 51,4725) km² | 112 | 0,3819 | 0,0567 |
| Diện tích Q4: [219,95; 15.025,609] km² | 113 | 0,2666 | 0,0027 |
| Độ cao Q1: [221; 591,5) m | 113 | 0,4420 | 0,1037 |
| Độ cao Q4: [1.294; 3.122] m | 112 | 0,1302 | 0,0110 |
| Tỷ phần tuyết Q1: [0,05; 0,09) | 106 | 0,3455 | 0,0274 |
| Tỷ phần tuyết Q4: [0,25; 0,61] | 117 | 0,1403 | 0,0137 |
| Khô hạn Q1: [0,37; 0,49) | 106 | 0,2392 | 0,0244 |
| Khô hạn Q4: [0,81; 1,51] | 118 | 0,3977 | 0,0949 |

Ví dụ, ở nhóm cao từ 1.294 m, SeqLSTM_Q vẫn có median cải thiện dương 0,1302 điểm so persistence, nhưng thấp hơn 0,4420 ở nhóm thấp dưới 591,5 m. Các nhóm trung gian không nhất thiết tạo xu hướng đơn điệu: ΔNSE của SeqLSTM_Q theo nhóm tuyết lần lượt 0,3455; 0,4194; 0,4217; 0,1403. Hệ số âm tổng thể không có nghĩa mỗi lần tăng nhóm tuyết đều làm điểm giảm.

Trong phép so SeqLSTM có Q − không Q, median ΔNSE theo bốn nhóm diện tích là **0,0417; 0,0533; 0,0912; 0,0876**. Đây là bằng chứng cụ thể cho mức chênh lớn hơn ở nhóm diện tích lớn trong các run đã có. [Biên nhóm](Tables/BasinAnalysis_quartile_definitions.csv), [bảng nhóm đầy đủ](Tables/BasinAnalysis_grouped_skill.csv).

![Mức cải thiện NSE theo nhóm thuộc tính](Figures/BasinAnalysis_quartile_delta_NSE.png)

Ba seed DLinear có đường mô tả gần nhau ở cấp nhóm; từng lưu vực vẫn có thể biến thiên lớn.

![Các cặp thuộc tính và chênh lệch theo từng lưu vực](Figures/BasinAnalysis_priority_scatter.png)

Biểu đồ giữ các điểm âm lớn và dùng cùng 450 ID ở mỗi cặp. Độ phân tán cho thấy một thuộc tính riêng lẻ chưa giải thích toàn bộ khác biệt giữa các lưu vực.

### 11.4. Kết quả bắt đúng ngày khởi phát lũ

Với lũ hai năm, cụm USWRC và dung sai 0, POD cộng dồn của SeqLSTM_Q theo bốn nhóm diện tích tăng từ **14,23% lên 28,70%**. Các mẫu số là số trận, khác với số ID dùng để tính median ΔNSE.

| Nhóm diện tích | Trận thật đủ điều kiện | SeqLSTM_Q: bắt đúng / POD | EncDec_Q: bắt đúng / POD |
|---|---:|---:|---:|
| Q1 | 267 | 38 / 14,23% | 23 / 8,61% |
| Q2 | 303 | 55 / 18,15% | 37 / 12,21% |
| Q3 | 280 | 53 / 18,93% | 27 / 9,64% |
| Q4 | 216 | 62 / 28,70% | 39 / 18,06% |

Trong khi median ΔNSE của SeqLSTM_Q so persistence thấp hơn ở Q4 diện tích so Q1, POD khởi phát lũ lại cao hơn ở Q4. Hai chỉ số mô tả hai khía cạnh khác nhau của dự báo. Với nhóm độ cao Q4, POD của SeqLSTM_Q là **36/289 = 12,46%**, thấp hơn **81/306 = 26,47%** ở Q3; quan hệ theo độ cao cũng không đơn điệu qua mọi nhóm.

Mỗi seed DLinear chỉ có hai trận khớp đúng ngày trong toàn bộ 1.066 trận, nên POD theo nhóm gần 0. Số lượng này quá ít để dùng riêng hình nhóm nhằm khẳng định một loại lưu vực nào phù hợp cho cảnh báo bằng DLinear. [Số đếm và tỷ lệ từng nhóm](Tables/BasinAnalysis_grouped_skill.csv).

![POD cộng dồn theo nhóm thuộc tính](Figures/BasinAnalysis_quartile_flood_POD.png)

### 11.5. Độ nhạy theo topology, seed và độ phủ dữ liệu

Nhóm không có lưu vực được chọn khác ở thượng lưu gồm **322 ID**, trong đó **321 ID có ΔNSE**. Với SeqLSTM_Q so persistence, các hệ số độ cao, tuyết và khô hạn lần lượt là **−0,437; −0,313; 0,266**, cùng hướng với kết quả toàn bộ 450 ID. Hệ số diện tích của phép so có Q − không Q là **0,325** với SeqLSTM và **0,293** với EncDec. Các xu hướng này còn hiện diện sau khi giảm lồng nhau trong tập chọn; các nhóm vẫn có thể phụ thuộc khí hậu và không gian. Bảng topology có tám hàng outcome hằng nên ρ không xác định, giữ đúng trạng thái. [Độ nhạy topology](Tables/BasinAnalysis_topology_sensitivity.csv).

Ba seed DLinear giữ cùng hướng liên hệ với bốn thuộc tính ưu tiên. Median độ lệch chuẩn mẫu của ΔNSE qua ba seed theo lưu vực là **0,00383 điểm**, nhưng tại 742 và 758 lần lượt là **2,753 và 6,477 điểm**. Cần đọc cả phần đuôi này thay vì dùng sự gần nhau của các đường nhóm để kết luận mọi lưu vực ổn định. [Độ phân tán seed](Tables/BasinAnalysis_DLinear_seed_spread.csv).

Tỷ lệ thiếu nhãn train có liên hệ rất yếu với ΔNSE của SeqLSTM_Q so persistence, **ρ = −0,048**, và số cửa sổ train DLinear có **ρ = 0,037**. Vì vậy, trong tập đã chấm, hai thống kê độ phủ này chưa giải thích rõ thứ hạng cải thiện của SeqLSTM_Q. Kiểm độ phủ vẫn cần để xác định các ngày và ID được phép chấm. [Liên hệ diagnostics](Tables/BasinAnalysis_diagnostic_associations.csv).

### 11.6. Kết luận dùng khi trình bày với giảng viên

**Đặc điểm lưu vực có liên hệ với mức cải thiện, và mối liên hệ thay đổi theo phép so.** Trên Extended LamaH-CE, SeqLSTM_Q cải thiện so persistence ít hơn ở nhóm độ cao và tỷ phần tuyết cao; lợi ích của run có Q so không Q có xu hướng lớn hơn ở lưu vực diện tích lớn. Các xu hướng chính còn giữ khi chỉ xét nhóm không có lưu vực được chọn ở thượng lưu. Riêng khả năng bắt đúng ngày khởi phát lũ phải đọc bằng POD và số trận, vì kết luận từ NSE không tự chuyển sang kết luận về sự kiện.

Những kết quả này xác định được các nhóm cần phân tích và kiểm chứng tiếp. Để xác định mô hình thực sự sử dụng đầu vào nào cần kết hợp che nguồn, giải thích mô hình và thí nghiệm có kiểm soát. Khẳng định hiệu quả trên dataset khác cần đánh giá trên dataset đó; tương quan trong một dataset chưa cung cấp bằng chứng chuyển miền.

### 11.7. Bằng chứng kiểm kết quả

Đã giữ nguyên **18 tệp thực nghiệm** gồm 12 CSV, hai JSON và bốn PNG, tổng 4.408.277 byte. **143 kiểm cục bộ đạt**, gồm đối chiếu hai phía của toàn bộ 5.412 hàng contrast với CSV bộ đánh giá, tính lại 1.116 hệ số bằng SciPy, 1.116 p-value với 9.999 hoán vị mỗi phép thử và Holm, 180 hệ số diagnostics, biên nhóm, 240 hàng thống kê nhóm, bảng phân loại và 1.353 hàng phân tán seed. Sai khác |ρ| tối đa khoảng 4,44×10⁻¹⁶; p hoán vị khớp hoàn toàn. Đọc CSV bằng chế độ round-trip để giữ nguyên các số thực dùng xác định thứ hạng và ties.

[Manifest tệp nhận](Tables/BasinAnalysis_received_manifest.csv), [bảng kiểm](Tables/BasinAnalysis_output_audit.csv) và [biên nhận kiểm tra](Tables/BasinAnalysis_output_verification.json) ghi phạm vi đối chiếu. SHA-256 của bốn đầu vào bộ đánh giá được kiểm bằng các tệp cục bộ tương ứng. Giá trị thuộc tính gốc, NPZ A3 và thành viên nhóm topology được kiểm trong lần chạy Kaggle; kiểm cục bộ sử dụng CSV đã xuất, không nạp lại các nguồn đầy đủ đó hoặc huấn luyện lại mô hình.

### 11.8. Kiểm phần mở rộng A4

Mã mở rộng qua **41 kiểm thử độc lập**, cùng đường chạy dữ liệu giả cho cả 11 và 14 run. Mười bảng tính toán gốc khớp hoàn toàn bản trước; khi thêm A4, mọi hàng của 12 phép so cũ giữ nguyên, gồm ρ, p hoán vị và Holm. Đường chạy mới xuất đủ 18 phép so, 1.674 hàng tương quan, 186 hàng độ ổn định và năm hình.

Đã kiểm 12 trường hợp từ chối hoặc bảo vệ trạng thái: thiếu từng seed A4, metadata summary sai, cờ ngữ cảnh/hash không đạt, seed chỉ số không khớp, khác mẫu số sự kiện, bảng nguồn có lỗi và chạy lại thất bại sau một lần thành công. Hình được kiểm bố cục trên dữ liệu giả. Những kiểm tra này xác nhận mã và đường dữ liệu; kết quả thực nghiệm A4 trên 451 lưu vực được trình bày ở Mục 12.

## 12. Kết quả phân tích 14 run có A4

### 12.1. Phạm vi và đối chiếu

Phân tích trên Output đánh giá giao thức 3 hoàn thành trong khoảng **40,7 giây** trên Kaggle CPU, với **41 kiểm thử đạt**, 14 run, 18 phép so và 31 thuộc tính dạng số. Giữ đủ 451 ID; 450 ID có 652.628 ngày–lưu vực chấm chung. Có **1.674 hệ số hữu hạn** và **186 hàng độ ổn định giữa seed**. [Thông tin lần chạy](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_summary.json).

Đã lưu nguyên **21 tệp** gồm 14 CSV, hai JSON và năm PNG, tổng 6.614.708 byte. **492 kiểm cục bộ đạt**: tính lại 1.674 hệ số bằng SciPy, 1.674 p-value với 9.999 hoán vị mỗi phép thử và Holm, 270 hệ số diagnostics, 360 hàng bảng nhóm, hai bảng phân tán gồm 2.706 hàng và 186 hàng độ ổn định hệ số. Sai khác |ρ| tối đa **4,44×10⁻¹⁶**, p hoán vị khớp hoàn toàn; kết quả của 12 phép so gốc giữ nguyên.

Các phía lặp lại trong bảng contrast được kiểm nhất quán theo run–ID; 78 hàng tổng hợp sáu chỉ số và 13 hàng sự kiện khớp bộ đánh giá 14 run. Kiểm cục bộ dựa trên CSV đã xuất; không nạp lại thuộc tính gốc, NPZ A3, dự báo ngày hoặc bảng chỉ số/sự kiện từng ID gốc của bộ đánh giá trong lần này. Thành viên nhóm topology và danh tính đầu vào đầy đủ được kiểm tại Kaggle. [Manifest](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_received_manifest.csv), [bảng kiểm](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_output_audit.csv), [phạm vi kiểm chứng](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_output_verification.json).

### 12.2. Liên hệ với thuộc tính được giữ qua ba seed LSTM

| Thuộc tính, outcome ΔNSE so persistence | Seed 111 | Seed 222 | Seed 333 |
|---|---:|---:|---:|
| Diện tích area_calc | −0,092 | −0,101 | −0,105 |
| Độ cao trung bình | −0,415 | −0,423 | −0,417 |
| Tỷ phần tuyết trong lượng mưa | −0,261 | −0,273 | −0,267 |
| Chỉ số khô hạn ET0/lượng mưa | 0,183 | 0,201 | 0,212 |

Mỗi hệ số dùng 450 lưu vực. Các liên hệ độ cao, tuyết và khô hạn giữ hướng qua ba seed; cùng kết luận mô tả đã thấy với trọng số tác giả. Diện tích có liên hệ âm yếu, với p Holm **0,3992 / 0,2744 / 0,1856** trong các họ 31 thuộc tính tương ứng; ba thuộc tính còn lại có p Holm 0,0031. Các p này dựa giả định trao đổi ID, không loại trừ phụ thuộc không gian.

Toàn bộ **31/31 thuộc tính có ρ cùng dấu qua ba seed đối với ΔNSE**. Con số tương ứng là 29/31 cho ΔPNSE và 27/31 cho ΔPOD hai năm; cùng dấu chưa bảo đảm hệ số lớn. Kết quả củng cố độ ổn định của hướng liên hệ với mức cải thiện, nhưng không biến tương quan thành đóng góp nhân quả của đầu vào. [Bảng hệ số](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_static_associations.csv), [độ ổn định](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_seed_association_stability.csv).

![Mức cải thiện NSE theo nhóm, gồm ba seed A4](Figures/BasinAnalysis_A4_Integrated/BasinAnalysis_quartile_delta_NSE.png)

*Hình 5. Các run dùng cùng biên nhóm thuộc tính. Hàng dưới giữ riêng ba seed A4 và trọng số tác giả, trên cùng thang tung độ với hàng DLinear. Đường nhóm mô tả median hiệu NSE, không phải phần trăm cải thiện.*

### 12.3. Nhóm độ cao và diện tích cho hai góc nhìn kỹ năng

Ở nhóm độ cao thấp nhất [221; 591,5) m, median ΔNSE của A4 111/222/333 là **0,4281 / 0,4248 / 0,4303**; ở nhóm cao nhất [1.294; 3.122] m là **0,1317 / 0,1319 / 0,1258**. Cả ba lần chạy vẫn cải thiện theo median ở nhóm cao nhất, nhưng mức cải thiện thấp hơn rõ rệt. Với chỉ số khô hạn, median hiệu tăng từ khoảng **0,238–0,245** ở Q1 lên **0,371–0,392** ở Q4; các nhóm giữa không hoàn toàn đơn điệu.

POD đúng ngày ở nhóm diện tích nhỏ nhất là **13,11 / 15,73 / 14,98%**, tương ứng 35/267, 42/267, 40/267 trận. Nhóm diện tích lớn nhất đạt **25,46 / 29,63 / 26,85%**, tương ứng 55/216, 64/216, 58/216 trận. Dù median ΔNSE thấp hơn ở Q4 diện tích so Q1, POD lại cao hơn ở Q4. Phép so hiệu NSE theo lưu vực và tỷ lệ bắt trận cộng dồn có đại lượng cũng như trọng số khác nhau. [Biên nhóm](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_quartile_definitions.csv), [bảng nhóm](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_grouped_skill.csv).

![POD theo nhóm, gồm ba seed A4](Figures/BasinAnalysis_A4_Integrated/BasinAnalysis_quartile_flood_POD.png)

*Hình 6. POD lũ hai năm dùng cùng ngày bắt đầu, cộng số trận trước khi chia. Hàng A4 cho thấy độ biến thiên theo seed và thuộc tính; cần đọc kèm số trận trong từng nhóm.*

### 12.4. Ổn định hướng liên hệ và biến thiên từng lưu vực

Median std mẫu của ΔNSE qua ba seed tại từng ID là **0,02065** với LSTM A4 và **0,00383** với DLinear. Với ΔPNSE, hai giá trị tương ứng là **0,03661** và **0,00591**. Đây là độ phân tán của hiệu chỉ số tại cùng ID, không phải độ lệch chuẩn của ba NSE trung vị toàn tập. Hai mô hình dùng đầu vào và cấu hình khác nhau nên chưa thể quy toàn bộ chênh lệch này cho kiến trúc.

Tại 742 và 758, std mẫu ΔNSE của A4 là **1,818** và **3,912**; phương sai quan trắc nhỏ làm chỉ số nhạy với sai lệch dự báo. Hướng tương quan ổn định ở cấp tập không bảo đảm từng lưu vực ít biến động giữa seed.

Có **126/366 ID có trận thật** mà POD A4 khác nhau giữa ba seed; DLinear chỉ có 2/366 ID như vậy. Median std POD bằng 0 ở cả hai họ, nên cần đọc cả phân phối: DLinear gần như không bắt được trận đúng ngày, tạo độ phân tán nhỏ quanh kết quả kém. Độ ổn định không thay cho chất lượng dự báo. [Phân tán A4](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_LSTM_A4_seed_spread.csv), [phân tán DLinear](Tables/BasinAnalysis_A4_Integrated/BasinAnalysis_DLinear_seed_spread.csv).

![Độ phân tán qua ba seed tại từng lưu vực](Figures/BasinAnalysis_A4_Integrated/BasinAnalysis_seed_spread.png)

*Hình 7. Std mẫu của ba hiệu chỉ số tại mỗi lưu vực; n = 450 cho NSE/PNSE và 366 cho POD. Trục symlog giữ cả giá trị 0 và các ngoại lệ lớn; các seed không được tính thành lưu vực độc lập.*

## 13. Nguồn tham khảo

- [A1 — cấu trúc, ý nghĩa và kiểm thuộc tính](../../02_Exploration/01_A1_LamaHCEExt_Explore/01_A1_LamaHCEExt_Explore.md), đặc biệt Mục 17.4–17.6, 17.17–17.19.
- [A3 — dữ liệu DLinear, mặt nạ và chẩn đoán sai số](../../03_Baseline/02_A3_LamaHCEExt_Benchmarks/02_A3_LamaHCEExt_Benchmarks.md).
- [Bộ đánh giá — chỉ số, mức lưu lượng và sự kiện](../01_Evaluation_LamaHCEExt/01_Evaluation_LamaHCEExt.md).
- [Klingler và cs. (2021), LamaH-CE](https://essd.copernicus.org/articles/13/4529/2021/), Mục 5 và Phụ lục A: định nghĩa, đơn vị và nguồn thuộc tính.
- [Konold và cs. (2026), BiasCast](https://hess.copernicus.org/articles/30/5067/2026/index.html), phân tích theo lưu vực và liên hệ đặc điểm với mức cải thiện; phần phân tích chọn riêng ΔNSE dương của bài được phân biệt với tập đầy đủ dùng ở đây.
- [SciPy — Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) và [permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html): hệ số thứ hạng và phép thử bằng hoán vị.
