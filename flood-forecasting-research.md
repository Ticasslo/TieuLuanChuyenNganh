# Đề tài: Dự báo lũ lụt/sạt lở miền Trung Việt Nam bằng Mamba (State Space Model)

> Tài liệu kế hoạch nghiên cứu cho tiểu luận chuyên ngành → khóa luận tốt nghiệp.

**Tên đề tài cụ thể (tự đặt):**
- 🎓 Tiểu luận: *"Ứng dụng mô hình Mamba (State Space Model) trong dự báo lưu lượng dòng chảy lũ lụt lưu vực sông Vu Gia – Thu Bồn"*
- 🎓🎓 Khóa luận: *"Dự báo lũ lụt đa lưu vực miền Trung Việt Nam bằng kiến trúc Mamba tích hợp đồ thị mạng lưới sông (Graph-enhanced Mamba)"*

---

## 📌 Tóm tắt phạm vi — đọc mục này trước tiên

| | 🎓 TIỂU LUẬN CHUYÊN NGÀNH (làm ngay) | 🎓🎓 KHÓA LUẬN TỐT NGHIỆP (làm sau) |
|---|---|---|
| **Kiến trúc** | Mamba **thuần** | Mamba **+ Graph** (nâng cấp lõi, vẫn 1 model) |
| **Vùng** | 1 lưu vực — Vu Gia–Thu Bồn | Nhiều lưu vực, có thể mở rộng toàn VN |
| **Dữ liệu** | GloFAS (historical + forecast) + code/checkpoint RiverMamba | Thêm: `ldd` GloFAS (dựng Graph), IBTrACS (bão), DEM (địa hình) |
| **Kiến trúc hệ thống** | 1 hệ thống dự báo, cập nhật 1 lần/ngày | Vẫn 1 hệ thống, chỉ đổi lõi bên trong |
| **So sánh** | vs LSTM/GRU/Transformer baseline (target GloFAS) + vs thực tế (GRDC trạm Nông Sơn, bắt buộc — Mục 2.5) | vs paper GNN-Transformer 2026 đã có |
| **Đầu ra** | Báo cáo tiểu luận + demo VPS | Khóa luận + có thể công bố SOICT |

Nguyên tắc xuyên suốt file: mọi nội dung không có nhãn 🎓🎓 là việc cần làm ngay cho tiểu luận. Nhãn 🎓🎓 Khóa luận đánh dấu phần để dành cho sau, chưa cần làm ở giai đoạn hiện tại.

---

## 📖 Thuật ngữ cần biết trước

Mục này dành cho người đọc lần đầu, chưa quen với đề tài. Nếu đã nắm các khái niệm dưới đây, có thể bỏ qua và đọc thẳng từ Mục 1.

**Nhóm AI / Machine Learning**

| Thuật ngữ | Giải thích |
|---|---|
| Mamba / State Space Model (SSM) | Một kiểu kiến trúc mạng neural chuyên xử lý dữ liệu dạng chuỗi theo thời gian, cùng nhóm bài toán với LSTM, GRU, Transformer, nhưng xử lý chuỗi dài nhanh hơn hẳn. Đây là kiến trúc chính mà đề tài sử dụng. |
| Checkpoint | File lưu lại "bộ não" (trọng số) của một model sau khi đã train. Tải checkpoint có sẵn nghĩa là dùng lại model người khác đã train, thay vì phải train từ đầu. |
| Pretrain / Fine-tune | Pretrain là train model lần đầu trên một tập dữ liệu lớn, tổng quát. Fine-tune là train tiếp theo — nhẹ hơn, ít bước hơn — trên một tập dữ liệu nhỏ, riêng biệt, để model "chuyên" hơn cho đúng bài toán/vùng cần, không phải train lại từ số 0. |
| Transfer learning | Dùng một model đã học từ một bài toán/vùng có nhiều dữ liệu làm điểm khởi đầu cho một bài toán/vùng khác thường ít dữ liệu hơn — thường nhanh hơn và cho kết quả tốt hơn train từ đầu. |
| Baseline | Model hoặc phương pháp cũ, đơn giản hơn, dùng làm mốc so sánh để biết model chính (Mamba) có thật sự tốt hơn hay không. |
| Encoder / Decoder | Trong kiến trúc encoder-decoder, encoder là phần "đọc hiểu" dữ liệu đầu vào (thường là quá khứ), decoder là phần "tạo ra" kết quả dự báo (thường là tương lai). Một model có thể gồm cả hai phần nối tiếp nhau. |
| Epoch | Một lần model học qua hết toàn bộ dữ liệu train. "60 epoch" nghĩa là lặp lại toàn bộ quá trình học 60 lần trên cùng một bộ dữ liệu. |
| LOAN, D_state, D_conv, Dropout... | Các thành phần/tham số kỹ thuật sâu hơn của kiến trúc Mamba — giải thích riêng khi xuất hiện lần đầu ở Mục 4.2, không cần nắm trước. |

**Nhóm thủy văn (hydrology)**

| Thuật ngữ | Giải thích |
|---|---|
| Lưu lượng (discharge) | Lượng nước chảy qua một điểm trên sông trong một đơn vị thời gian, đo bằng m³/s. Đây là đại lượng chính model dự báo — khác với mực nước. |
| Mực nước (water level) | Độ cao mặt nước sông, đo bằng mét. Khác với lưu lượng — muốn quy đổi qua lại cần một công thức riêng gọi là rating curve. |
| Rating curve | Công thức (hoặc đường cong) quy đổi giữa lưu lượng và mực nước, khác nhau ở từng mặt cắt sông cụ thể — không có công thức chung cho mọi nơi. |
| Return period (chu kỳ lặp lại) | Cách đo độ hiếm của một trận lũ. "Lũ 20 năm" không nghĩa là đúng 20 năm mới xảy ra 1 lần, mà là mức lũ có xác suất khoảng 5% xảy ra trong bất kỳ năm nào (trung bình lặp lại mỗi 20 năm). |
| Reanalysis / Forecast / Reforecast | Ba cách một hệ thống mô phỏng thời tiết/thủy văn có thể vận hành. Reanalysis mô phỏng lại quá khứ bằng thời tiết thật đã biết (chính xác nhất, dùng để train). Forecast dự báo thật cho tương lai, dùng lúc vận hành thật. Reforecast "giả vờ dự báo" một ngày quá khứ để đo độ tin cậy của Forecast, dùng làm baseline đánh giá. |
| Naturalized flow (dòng chảy tự nhiên) | Dòng chảy sông giả định không chịu can thiệp của con người (đập, hồ chứa) — khác dòng chảy thật ở vùng có đập. |
| NSE, KGE, RSR, PBIAS, R², F1-score | Các chỉ số đo độ chính xác của model — chi tiết ở Mục 6. |

**Các từ viết tắt hay gặp**

| Viết tắt | Giải thích |
|---|---|
| GloFAS | Global Flood Awareness System — hệ thống mô phỏng lưu lượng sông toàn cầu do Copernicus (châu Âu) quản lý, nguồn dữ liệu chính của đề tài. Chi tiết: `01_glofas.md`. |
| GRDC | Global Runoff Data Centre — kho dữ liệu lưu lượng sông đo thật (không mô phỏng) từ các trạm toàn thế giới, miễn phí cho nghiên cứu. |
| KTTV | Khí tượng Thủy văn — ngành và cơ quan nhà nước Việt Nam (Tổng cục KTTV) quản lý dữ liệu thời tiết, thủy văn cấp quốc gia. |
| ECMWF | European Centre for Medium-Range Weather Forecasts — trung tâm dự báo thời tiết châu Âu, vận hành GloFAS, ERA5, HRES. |
| CDS / EWDS | Climate Data Store / Earth Watch Data Store — cổng tải dữ liệu của ECMWF/Copernicus. |
| LISFLOOD | Mô hình thủy văn vật lý (không phải AI) mà GloFAS chạy bên trong để mô phỏng nước chảy. |
| ERA5 / CPC / HRES | Ba nguồn dữ liệu thời tiết làm input cho model — ERA5 (tái phân tích khí tượng, ECMWF), CPC (dữ liệu mưa, NOAA/Mỹ), HRES (dự báo thời tiết thật, ECMWF). |
| DONRE / PCTT | Sở/Ban Tài nguyên Môi trường và Ban Chỉ huy Phòng chống thiên tai — cơ quan cấp tỉnh (VD Quảng Nam), khác KTTV vốn ở cấp quốc gia. |

---

## 1. Tổng quan đề tài

| Mục | Nội dung |
|---|---|
| Kiến trúc AI | Mamba / State Space Model (SSM). Ở khóa luận, kiến trúc này được nâng cấp thành Mamba+Graph (🎓🎓). |
| Bài toán | Dự báo lưu lượng sông (discharge) và nguy cơ lũ lụt. |
| Vùng nghiên cứu | Miền Trung Việt Nam. Tiểu luận tập trung vào 1 lưu vực cụ thể là Vu Gia–Thu Bồn (Quảng Nam-Đà Nẵng, khoảng 10.350 km²), nơi hai con sông Vu Gia và Thu Bồn hợp lưu tại Đại Lộc thành một hệ thống chung. Đây chỉ là một trong nhiều lưu vực của miền Trung (còn có Trà Khúc, Kôn, Ba, Thạch Hãn, Hương...), nên phạm vi tiểu luận chưa phủ hết miền Trung. Điều này không phải vì GloFAS thiếu dữ liệu — GloFAS phủ toàn cầu, đã có sẵn dữ liệu cho toàn bộ Việt Nam, chỉ cần đổi bounding box là lấy được vùng khác. Việc chọn 1 lưu vực đơn thuần là quyết định về phạm vi (scope) để tiểu luận chắc chắn hoàn thành đúng hạn; khi khóa luận mở rộng thêm, vẫn dùng lại đúng nguồn GloFAS này và chỉ cần đổi bounding box (xem checklist ở Mục 11). |
| Ý nghĩa thực tế | Tính đến 4/12/2025, thiên tai trong năm 2025 đã làm 419 người chết/mất tích và gây thiệt hại khoảng 3,97 tỷ USD trên toàn quốc (nguồn: VietNamNet). Riêng đợt lũ miền Trung tháng 10-12/2025 làm 219 người chết/mất tích, thiệt hại 1,61 tỷ USD (nguồn: Wikipedia "2025 Central Vietnam floods", đã đối chiếu khớp với nhiều nguồn tin quốc tế khác). |
| Điểm tổng đánh giá | 10/10 — cao nhất trong số 32 đề tài đã khảo sát. |

---

## 2. Cơ sở khoa học (nền tảng chung cho cả 2 giai đoạn)

### 2.1 Kiến trúc nền và paper tham chiếu

Mamba (Gu & Dao, 2023) xử lý được chuỗi thời gian rất dài với độ phức tạp tính toán tuyến tính, khác với Transformer vốn có độ phức tạp bậc hai, nên không cần cắt bớt dữ liệu đầu vào.

Paper mẫu chính của đề tài là RiverMamba (NeurIPS 2025, arXiv 2505.22535) — công trình dùng Mamba để dự báo lưu lượng sông trên quy mô toàn cầu. Khi được đánh giá bằng số đo thật từ GRDC, RiverMamba vượt trội cả mô hình vật lý (GloFAS) lẫn LSTM (chi tiết xem Mục 2.5).

Code và checkpoint của RiverMamba đã được công khai tại `github.com/HakamShams/RiverMamba_code`, bao gồm code gốc và một script tải sẵn trọng số đã train (`donwload_pretrained_models.sh`). Đây là tài nguyên quan trọng, giúp giảm đáng kể rủi ro kỹ thuật cho tiểu luận.

### 2.2 RiverMamba gốc train thế nào

Paper gốc train theo 2 giai đoạn, và cả hai giai đoạn đều ở quy mô toàn cầu — không có giai đoạn nào chạy ở quy mô 1 lưu vực nhỏ. Input ở cả 2 giai đoạn giống hệt nhau (gồm GloFAS, ERA5-Land, CPC, và các biến tĩnh); điểm khác nhau nằm ở target (đáp án dùng để sửa trọng số) và ở việc điểm nào trên lưới được tính vào loss:

| | Giai đoạn 1: Pretrain | Giai đoạn 2: Fine-tune GRDC |
|---|---|---|
| Target (đáp án) | `dis24` của chính GloFAS (mô phỏng) | Số đo thật từ **3.366 trạm GRDC** — *"compute the loss only on points where GRDC observations are available, without considering reanalysis data"* (đúng nơi có trạm, bỏ hẳn số GloFAS) |
| Số điểm tính loss | ~1,5 triệu điểm sông đã lọc (chi tiết lọc: Mục 2.3) | Chỉ ~3.366 điểm đã khớp với trạm |
| Train | 1979–2018 | 1979–2018 (y hệt) |
| Validation | 2019–2020 | 2019–2020 (y hệt) |
| Test | 2021–2024 | 2021–2023 |
| Epochs | 60 | 20 (fine-tune "chỉnh nhẹ", không học lại từ đầu) |
| Đánh giá kết quả so với | Chính GloFAS (Bảng 1 paper) | GRDC thật + benchmark thêm GloFAS-reforecast (Bảng 3 paper) |

### 2.3 Checkpoint và độ phủ GRDC tại Việt Nam

Có 2 checkpoint riêng biệt có thể tải về, cùng tên file (`RiverMamba_aifas_reanalysis`) nhưng khác trọng số: một bản chỉ pretrain, chưa fine-tune trên GRDC; và một bản đã fine-tune trên GRDC, cùng kiến trúc và cùng 1.529.667 điểm "AIFAS" nhưng khác trọng số.

Tập điểm "AIFAS" này được lọc theo trình tự sau: xuất phát từ ~21 triệu điểm lưới gốc (độ phân giải 0.05°) phủ toàn cầu, kể cả biển; lọc bỏ biển còn lại 6.221.926 điểm đất liền; rồi lọc tiếp, chỉ giữ những điểm có lưu lượng trung vị lớn hơn 10 m³/s, còn lại 1.529.667 điểm. Ngoài ra còn có một checkpoint khác train trên toàn bộ 6.221.926 điểm đất liền, không lọc theo lưu lượng — mục đích là để model có thể tạo ra bản đồ lưu lượng dày đặc ở mọi điểm đất liền lúc suy luận, không chỉ tại các điểm có sông. Checkpoint này không cần thiết cho tiểu luận, vì tiểu luận chỉ quan tâm đến những điểm có sông thật trong lưu vực Vu Gia-Thu Bồn — dùng đúng logic lọc AIFAS là đủ và tiết kiệm compute hơn.

Checkpoint được dùng làm điểm khởi đầu cho Phương án B (Mục 2.4) là bản đã fine-tune trên GRDC, không phải bản pretrain-only. Đây là cách làm chuẩn trong transfer learning: luôn bắt đầu từ trọng số tinh chỉnh nhất hiện có, trừ khi có bằng chứng cụ thể cho thấy nó sẽ gây hại — và hiện chưa có bằng chứng như vậy.

GRDC có tổng cộng 27 trạm tại Việt Nam, xác nhận trực tiếp trên `portal.grdc.bafg.de` bằng cách lọc Country = Viet Nam. Trong số đó, chỉ đúng 1 trạm nằm trong lưu vực Vu Gia-Thu Bồn: trạm `2371300 NONG SON`, trên sông Song Tranh (một nhánh của Thu Bồn), tọa độ 15,7167°N / 108,0167°E. 26 trạm còn lại (thuộc các lưu vực Trà Khúc, Vệ, Hương, Đà Rằng, Lam, Hiếu, Kontum, Đắk Lắk, Mekong Delta...) không dùng cho tiểu luận, nhưng được giữ lại cho khóa luận khi mở rộng ra nhiều lưu vực (Mục 10).

Một câu hỏi tự nhiên là làm sao chỉ 1 trạm thật lại có thể cải thiện được cả một lưới hàng triệu điểm chưa từng có trạm đo. Lý do là model dùng chung một bộ trọng số cho mọi điểm trên lưới, không phải một bảng tra cứu riêng cho từng điểm. Khi trọng số được sửa dựa trên sai lệch tại 1 trạm thật — ví dụ nhận ra rằng "mưa lớn 3 ngày liên tiếp ở địa hình dốc khiến GloFAS đoán thấp hơn thực tế khoảng 30%" — model học được một quy luật chung, và quy luật đó áp dụng lại cho mọi điểm khác có đặc điểm đầu vào tương tự (địa hình, khí hậu), kể cả những điểm chưa từng có trạm đo. Đây chính là lý do chỉ 3.366 trên hàng triệu điểm có đáp án thật vẫn cải thiện được toàn bộ lưới.

### 2.4 Hướng tiếp cận model cho tiểu luận — 2 phương án

Phương án chính, gọi là Phương án B, là tải checkpoint đã pretrain và fine-tune GRDC toàn cầu, sau đó fine-tune tiếp cho Vu Gia-Thu Bồn qua 2 bước nối tiếp: bước 2 với target là GloFAS, rồi bước 3 với target là GRDC thật tại Nông Sơn (bắt buộc). Đây là hướng đi có quy mô nhỏ, dùng transfer learning, thời gian nhanh và rủi ro thấp. Cần lưu ý rằng cả 2 bước fine-tune này là phần đề tài tự thêm vào, không phải điều mà paper RiverMamba gốc đã làm, nên cần ghi rõ điều này trong báo cáo để tránh gây hiểu nhầm là đang "reproduce y hệt paper".

Phương án bổ sung, gọi là Phương án A, chỉ làm nếu còn thời gian: train lại từ đầu (from scratch) trên dữ liệu Việt Nam, rồi so sánh kết quả A với B. Có cơ sở khoa học ủng hộ việc chọn hướng B: nghiên cứu của Ougahi và cộng sự, *"Investigating Deep Learning Knowledge Transfer in Streamflow Prediction From Global to Local Catchment"* (Water Resources Research, 2026), đạt NSE=0,85 và KGE=0,80 khi áp dụng transfer learning từ vùng nhiều dữ liệu sang vùng khan hiếm dữ liệu — đúng tình huống của đề tài này.

**Pipeline 3 bước fine-tune thật sự cho tiểu luận**

```
Bước 1: Checkpoint GRDC toàn cầu (RiverMamba tự làm sẵn, chỉ cần tải)
              │
              ▼
Bước 2 (CHÍNH): Fine-tune cho Vu Gia-Thu Bồn, target = GloFAS
        (dùng GloFAS chứ không dùng thẳng GRDC ở bước chính này, vì GRDC duy
        nhất tìm được trong vùng — trạm Nông Sơn — chỉ có dữ liệu 1978-1990,
        trước đập Sông Tranh 2 (2011), không đáng tin cho việc train hướng
        tới dự báo hiện tại — chi tiết ở bước 3. Hệ quả: kết quả "vs GloFAS"
        ở bước này chỉ có ý nghĩa yếu, xem lý do đầy đủ ở Mục 2.5)
              │
              ▼
Bước 3 (BẮT BUỘC): Fine-tune thêm 1 lớp nhẹ, target = GRDC thật,
        chỉ trạm Nông Sơn (2371300) — trạm DUY NHẤT trong 27 trạm GRDC
        Việt Nam nằm đúng trong Vu Gia-Thu Bồn. Bắt buộc vì: (1) chi phí
        làm rất thấp — dùng lại y hệt pipeline bước 2, chỉ đổi target,
        dữ liệu free có sẵn; (2) không làm thì tiểu luận không có bất kỳ
        kết quả nào "so với thực tế" — chỉ toàn so với mô phỏng (GloFAS)
        hoặc so kiến trúc (LSTM/GRU) — xem lý do đầy đủ ở Mục 2.5
```

Dữ liệu Nông Sơn từ GRDC có một hạn chế đáng lưu ý: nó chỉ có giai đoạn 1978–1990, thiếu 56,3% số ngày trong khoảng đó. Đập thủy điện Sông Tranh 2 hoàn thành vào tháng 8/2011, tức là sau toàn bộ giai đoạn dữ liệu này — nghĩa là dữ liệu Nông Sơn phản ánh dòng chảy tự nhiên, chưa bị điều tiết, khác hẳn dòng chảy hiện tại vốn đã bị đập kiểm soát. Dùng dữ liệu này để hiệu chỉnh model rồi áp dụng cho hiện tại có rủi ro dạy sai quy luật.

Tuy vậy, đây không phải lý do để bỏ qua bước 3. Chính paper RiverMamba gốc cũng gặp y hệt vấn đề này ở quy mô toàn cầu và không xử lý gì — trích nguyên văn: *"GloFAS reanalysis dataset only simulates the naturalized flow without considering realistic human interventions such as dams... this can be a major source of bias"*. Khi lọc 3.366 trạm GRDC, paper cũng không đặt tiêu chí nào về độ mới của dữ liệu — trạm cũ dừng từ 1980-1990 vẫn được giữ, miễn là trùng khoảng thời gian 1979-2018. Vì vậy, việc dùng dữ liệu Nông Sơn 1978-1990 là theo đúng chuẩn mà paper đã làm, chỉ cần ghi rõ hạn chế này trong báo cáo — rằng dữ liệu này phản ánh độ khớp lịch sử, không phải độ chính xác dự báo hiện tại.

Về lâu dài, giải pháp tốt hơn là nộp đơn mua dữ liệu Nông Sơn hiện tại qua KTTV Việt Nam (Mục 5, #10) — cách này giải quyết đúng gốc rễ vấn đề. Việc này không vội, vì tiểu luận dự kiến hoàn thành khoảng 11-12/2026, nhưng nên nộp đơn sớm: dùng GRDC cũ tạm thời trong lúc chờ, và khi có dữ liệu mới thì chuyển sang dùng làm chính, còn dữ liệu GRDC cũ vẫn giữ giá trị tham khảo hoặc kiểm tra chéo GloFAS ở thời kỳ trước đập.

Về phạm vi, đề tài quyết định giữ nguyên 1 lưu vực Vu Gia-Thu Bồn cho tiểu luận, không mở rộng ra toàn miền Trung dù có nhiều trạm GRDC hơn ở những nơi khác. Lý do gồm ba điểm: khớp đúng với câu chuyện thực tế (đợt lũ miền Trung 2025), có thể so sánh được với paper GNN-Transformer 2026 (vốn cũng chỉ làm trên lưu vực này), và làm sâu 1 lưu vực có giá trị hơn dàn trải trên nhiều lưu vực một cách sơ sài. Việc mở rộng vùng sẽ là công việc của khóa luận (Mục 10).

### 2.5 Vì sao bước 3 (GRDC thật) là bắt buộc, không thể chỉ dùng bước 2 (GloFAS)

Ở Bảng 3 của paper gốc, việc đánh giá dùng số đo thật từ GRDC làm đáp án — cả RiverMamba lẫn GloFAS-reforecast đều bị so sánh với đáp án độc lập đó. Đây là phép so sánh công bằng, vì có một "giám khảo" độc lập mà không bên nào được học theo trước. Chính vì vậy, khi RiverMamba thắng GloFAS trong phép so sánh này, kết luận đó mới có ý nghĩa thật.

Ngược lại, bước 2 của tiểu luận (fine-tune với target là GloFAS) không thể tạo ra một kết quả "so với thực tế" tương tự chỉ bằng cách đem so kết quả Mamba với GloFAS. Lý do là "đáp án" dùng để chấm điểm ở bước này chính là GloFAS: model được dạy suốt quá trình để bắt chước đúng cách GloFAS mô phỏng. Nếu lấy kết quả khớp cao đó làm bằng chứng cho việc "Mamba dự báo lũ chính xác", đây sẽ là một lập luận vòng — khớp với một mô phỏng không chứng minh được là khớp với thực tế, vì bản thân mô phỏng đó (GloFAS) vốn đã có sai số so với thực tế. Số liệu thật của paper xác nhận đúng điều này: cùng là RiverMamba, R² so với GloFAS (Bảng 1, Mục 4) đạt 0,873, nhưng R² so với GRDC thật (Bảng 3, Mục 4) chỉ còn 0,506 — hai con số này đo hai thứ khác nhau, không thể dùng thay cho nhau.

Đây chính là lý do bước 3 là bắt buộc: nó là cách duy nhất để tiểu luận có được một kết quả so với thực tế mà không rơi vào lập luận vòng nói trên. Chi phí thực hiện thấp vì dùng lại pipeline của bước 2, chỉ cần đổi target, còn dữ liệu thì đã có sẵn miễn phí. Lợi ích lại lớn, vì tiểu luận sẽ có sẵn câu trả lời khi hội đồng hỏi "so với thực tế thì sao".

### 2.6 Baseline so sánh

| Baseline | Paper gốc có dùng không? | Ghi chú |
|---|---|---|
| Climatology (trung bình lịch sử) | ✅ Có | Baseline yếu nhất, làm mốc sàn |
| Persistence (giữ nguyên giá trị hôm nay) | ✅ Có | |
| Encoder-Decoder LSTM | ✅ Có | Kiến trúc cụ thể từ hệ thống dự báo lũ vận hành thật của Google — không phải LSTM thường |
| **GloFAS** (bản reforecast) | ✅ Có | Baseline quan trọng nhất |
| GRU thuần | ❌ Không | Tự thêm riêng cho tiểu luận, không có trong paper gốc |
| Transformer thuần | ❌ Không | Tự thêm riêng cho tiểu luận, không có trong paper gốc |

Khi trích dẫn rằng "paper cũng so sánh với LSTM", cần ghi rõ rằng GRU và Transformer là phần tự thêm cho tiểu luận, để tránh gây hiểu nhầm là chúng thuộc đúng thiết kế thí nghiệm gốc.

Một điểm dễ nhầm lẫn là GloFAS thực ra có 3 sản phẩm khác nhau, và baseline nói ở trên khác với GloFAS được dùng làm target train (chi tiết khái niệm nằm ở `01_glofas.md`):

Sản phẩm Historical/Reanalysis (`cems-glofas-historical`) chạy LISFLOOD bằng thời tiết thật đã biết (ERA5). Đây chính là "GloFAS" được dùng làm target train ở bước 2 và ở giai đoạn pretrain (Mục 2.2), không phải một baseline.

Sản phẩm Forecast (`cems-glofas-forecast`) chạy bằng dự báo thời tiết thật của ECMWF (HRES), cập nhật hàng ngày. Sản phẩm này dùng cho hệ thống dự báo thật lúc deploy (Mục 4.1), cũng không phải một baseline đánh giá.

Sản phẩm Reforecast (`cems-glofas-reforecast`) chạy bằng dự báo tổ hợp lịch sử (ECMWF-ENS, 11 thành viên), phát hành 2 lần mỗi tuần, nhìn xa 46 ngày, phủ giai đoạn 2003-2022. Mục đích gốc của sản phẩm này là đo độ tin cậy của hệ thống Forecast, bằng cách đứng ở một mốc quá khứ rồi dự báo rồi so với thực tế đã biết. Đây chính là baseline "GloFAS" xuất hiện trong Bảng 3 (Mục 4.3) — cả RiverMamba lẫn GloFAS Reforecast cùng được so với GRDC thật, và phép so sánh này công bằng vì cả hai đều đang "dự báo", không phải mô phỏng với thời tiết thật đã có sẵn như Historical.

Có một điểm chưa xác nhận được: cách paper hòa giải cấu trúc của Reforecast (phát hành 2 lần/tuần, nhìn xa 46 ngày) với cách đánh giá hàng ngày trong khoảng 1-7 ngày ở Bảng 3. Phần đã tra được của paper chỉ ghi tham chiếu tới cách xử lý của 2 nguồn khác, chưa mô tả rõ cách làm cụ thể. Cần tải thử dataset `cems-glofas-reforecast` và tự kiểm tra cấu trúc trước khi code lại baseline này (dataset chưa có sẵn trong Mục 5, cần bổ sung khi tới lượt implement baseline).

Lý do GloFAS là baseline quan trọng nhất nằm ở bản chất của nó. GloFAS chạy trên nền LISFLOOD, mô phỏng nước chảy bằng các phương trình vật lý/thủy lực thật (thấm đất, diễn toán dòng chảy qua từng đoạn kênh...), không học từ dữ liệu như AI mà chỉ dùng vài tham số vật lý (độ nhám, độ dốc...), khác hẳn hàng triệu trọng số của một mạng neural. Cả `cems-glofas-historical` lẫn `cems-glofas-forecast` dùng để train Mamba thực chất là output từ mô phỏng vật lý này, không phải số đo cảm biến 100% thật. Nói cách khác, đề tài về bản chất đang kiểm chứng: AI (Mamba) có dự báo chính xác hơn một model vật lý (GloFAS/LISFLOOD) hay không.

### 2.7 Baseline nâng cao và research gap (🎓🎓 khóa luận)

Paper GNN-Transformer 2026 (Nguyen et al., *River* journal, DOI 10.1002/rvr2.70046), tựa đề *"Hybrid GNN and temporal encoder models for flood forecasting in the Vu Gia–Thu Bon River Basin"*, dùng dữ liệu thật từ 9 trạm quan trắc mực nước trong lưu vực (Ái Nghĩa, Cầu Lâu, Giao Thủy, Hiệp Đức, Hội An, Hội Khách, Nông Sơn, Tam Kỳ, Thành Mỹ), với hơn 19.831 bản ghi mỗi trạm theo giờ (11/2017–12/2024), gồm mực nước, lượng mưa, và dữ liệu khí tượng — không dùng lưu lượng. Model GNN-Transformer trong paper này giảm 20–30% RMSE/MAPE/sMAPE so với LSTM/Transformer thuần. Vì paper chỉ làm trên Vu Gia-Thu Bồn, nếu khóa luận sau này mở rộng ra nhiều lưu vực, việc so sánh trực tiếp chỉ hợp lệ cho riêng kết quả của lưu vực này — cần tránh nói chung chung kiểu "G-Mamba thắng GNN-Transformer 2026".

Nguồn dữ liệu của paper này không phải GRDC, mà lấy trực tiếp từ Ban Chỉ huy Phòng chống thiên tai tỉnh Quảng Nam (Quang Nam Provincial Steering Committee for Natural Disaster Prevention and Control), không qua cổng GRDC. Paper tự ghi rõ dữ liệu không công khai: *"The data... are available on request from the corresponding author. The data are not publicly available due to privacy or ethical restrictions."*

Một phát hiện đáng chú ý là trong 9 trạm của paper này, có 4 trạm trùng với danh sách đang cần cho tiểu luận: Nông Sơn, Thành Mỹ (2 trạm ưu tiên chính, Mục 5 #10), Câu Lâu (xin thêm, nhiều khả năng cũng có lưu lượng), và Ái Nghĩa (xin thêm, không có lưu lượng). Tuy nhiên dữ liệu của paper này là theo giờ và chỉ có mực nước (giai đoạn 2017-2024), không phải lưu lượng cần cho bước 3 (chi tiết đơn vị xem ghi chú ở Mục 5). Nguồn này (Ban Chỉ huy PCTT tỉnh Quảng Nam) đã được cân nhắc nhưng không theo đuổi thêm, vì chỉ có giá trị phụ — làm case study trực quan hoặc đối chiếu với thang Báo động 1/2/3 (Mục 6) — không giúp được cho bước 3, nên không đáng công liên hệ thêm một kênh nữa ngoài KTTV.

Có 2 điểm cần cẩn thận nếu sau này trích dẫn số liệu hoặc sơ đồ của paper GNN-Transformer 2026 — đây là lỗi/mâu thuẫn nội bộ trong chính paper, không phải hiểu nhầm từ phía đề tài này:

Thứ nhất, sơ đồ mạng lưới trạm ở Mục 2.1 của paper đó xếp sai nhánh: paper liệt kê "Vu Gia branch: Hiep Duc → Thanh My → Nong Son → Ai Nghia", xếp Nông Sơn vào nhánh Vu Gia. Nhưng tra chéo Wikipedia tiếng Việt (mục Sông Thu Bồn) cho thấy: *"Bắt đầu khi đi qua địa phận Nông Sơn, Duy Xuyên, sông mới bắt đầu được gọi là Thu Bồn"* — nghĩa là Nông Sơn chính là điểm sông Tranh đổi tên thành Thu Bồn, tức nằm trên nhánh Thu Bồn. Điều này khớp đúng với dữ liệu GRDC đã tra ở Mục 2.3 ("Nông Sơn, sông Song Tranh, nhánh Thu Bồn"). Vì vậy, không nên copy nguyên sơ đồ "Vu Gia branch" của paper này nếu trích dẫn về sau.

Thứ hai, đơn vị mực nước trong paper không nhất quán: phần văn xuôi (Mục 4.1.1 của paper) ghi mực nước tính bằng "meters", nhưng Bảng 2 và Bảng 6 lại ghi cột "WL (mm)", và các giá trị thực tế (ví dụ Ái Nghĩa 610,4; hay MAE dự báo khoảng 90) chỉ hợp lý nếu đơn vị là milimét — nếu là mét thì vô lý, vì không sông nào cao 600m hay có sai số dự báo tới 90m. Nhiều khả năng câu "in meters" trong paper là một lỗi đánh máy, còn đơn vị thật sự là milimét.

Về research gap: paper 2026 nói trên cần một mạng cảm biến nội bộ tốn kém và khó tiếp cận. Đề tài này thử nghiệm theo hướng ngược lại — chỉ dùng dữ liệu vệ tinh/toàn cầu miễn phí (GloFAS), để xem có đạt được độ chính xác gần bằng cách làm cũ hay không, theo câu chuyện "dân chủ hóa" việc tiếp cận công nghệ dự báo lũ.

Nghiên cứu "Accelerating flood warnings by 10 hours" (npj Natural Hazards) chứng minh rằng việc mô hình hóa cấu trúc mạng lưới sông cải thiện đáng kể thời gian cảnh báo sớm — đây là nền tảng cho hướng đi ở Mục 3.

Cuối cùng, một related work khác không áp dụng trực tiếp nhưng đáng biết khi hội đồng hỏi: FM-Mamba (Scientific Reports, 2026) cũng dùng Mamba cho bài toán "flood mapping", nhưng là bài toán khác hẳn — phân đoạn ảnh SAR vệ tinh để vẽ ranh giới vùng ngập tại một thời điểm, không phải dự báo chuỗi thời gian lưu lượng như đề tài này.

---

## 3. 🎓🎓 Hướng cải tiến kiến trúc nâng cao — chỉ dành cho khóa luận

> ⛔ Mục này không áp dụng cho tiểu luận chuyên ngành. Đọc lướt để biết trước, chưa cần làm ở giai đoạn hiện tại.

### 3.1 Mamba + Graph — mô hình hóa mạng lưới sông

Một dòng sông không phải một đường thẳng, mà là một mạng lưới gồm nhiều nhánh phụ đổ vào nhánh chính. Ý tưởng của hướng cải tiến này là kết hợp Mamba (xử lý chuỗi thời gian tại mỗi điểm đo) với Graph Transformer (hiểu quan hệ thượng nguồn–hạ nguồn giữa các điểm đo/nhánh sông). Kiến trúc mẫu cho hướng này là G-Mamba (Graph-enhanced Mamba), vốn "tiêm" thông tin từ các nhánh sông lân cận vào quá trình xử lý chuỗi thời gian.

Bằng chứng cho hướng này đến từ paper G-Mamba gốc: kiến trúc này đạt độ chính xác dự báo mực nước 24 giờ ngang bằng với baseline EA-LSTM vốn chỉ dự báo được 14 giờ — chênh lệch 10 tiếng, tương đương cải thiện 71% tầm dự báo dài hạn. Kỹ thuật cốt lõi nằm ở chỗ graph mạng lưới sông vốn có dạng cây (tree-like), điều này gây ra hiện tượng "over-squashing" trong GNN (thông tin bị nén mất khi truyền qua nhiều bước). Paper giải quyết vấn đề này bằng cách biến đổi graph topology thành "dense reachability-based graph" để giảm resistance distance giữa các node — đây là kỹ thuật cụ thể có thể tham khảo khi implement Graph cho khóa luận.

Hướng này phù hợp vì bốn lý do: có bằng chứng đo lường cụ thể (10 tiếng), không mơ hồ; tận dụng được Graph Transformer, một kiến trúc đã tìm hiểu trước đó trong quá trình khảo sát các đề tài khác (gian lận, giao thông) khi chọn đề tài, nên không phải học từ số 0; dữ liệu cần thiết đã có sẵn trong kế hoạch (Mục 5 #11 là nguồn chính, #3 dùng phụ trợ); và độ khó ở mức vừa phải, không đòi hỏi kiến thức PDE/vật lý chuyên sâu.

Hướng này được để dành cho khóa luận thay vì làm ngay ở tiểu luận, vì tiểu luận cần giữ phạm vi nhỏ để chắc chắn hoàn thành đúng hạn (1 lưu vực, Mamba thuần so với baseline). Thêm Graph ngay từ đầu sẽ làm tăng rủi ro trễ hạn. Để dành cho khóa luận vừa an toàn hơn, vừa tạo được một câu chuyện tăng dần rõ ràng để trình bày với hội đồng.

Về bản chất kỹ thuật khi implement, cần lưu ý rằng Mamba+Graph không phải là hai model chạy nối tiếp nhau. Graph được nhúng thẳng vào bên trong quá trình xử lý của Mamba — ở khóa luận, lõi "Mamba thuần" sẽ được thay bằng lõi "Mamba+Graph", nhưng vẫn là một model duy nhất.

Cũng cần làm rõ rằng RiverMamba gốc không dùng graph/GNN thật. Cách paper gốc xử lý quan hệ không gian là "duỗi" các điểm lưới thành một chuỗi 1D theo space-filling curve (một đường quét cố định đi qua các điểm), rồi cho Mamba xử lý hai chiều trên chuỗi đó (chi tiết ở Mục 4). Đây chỉ là một cách xấp xỉ quan hệ không gian, không phải một graph tường minh với node và edge thật. Vì vậy, việc làm G-Mamba ở khóa luận không phải là "thay một module Graph có sẵn", mà là thêm mới hoàn toàn một cơ chế graph (node là điểm đo, edge là quan hệ thượng–hạ nguồn) mà kiến trúc gốc chưa hề có. Đây là một đóng góp kỹ thuật thật sự mới, cần trình bày đúng như vậy trong khóa luận, không nên nói là "cải tiến module graph của RiverMamba".

Về nguồn dữ liệu dựng Graph, nên dùng `ldd` của chính GloFAS thay vì HydroBASINS hay Hydroviet. GloFAS/LISFLOOD tự có sẵn biến `ldd` (local drainage direction) — bản đồ hướng dòng chảy nội bộ mà chính LISFLOOD dùng để nối các ô lưới khi mô phỏng nước chảy, tải cùng nguồn với GloFAS tại JRC Data Catalogue. Đây là lựa chọn khớp nhất để dựng Graph vì bốn lý do: khớp chính xác 100% với đúng lưới GloFAS mà model đang dùng, không cần bước "snap"/khớp tọa độ giữa hai định dạng dữ liệu khác nhau; trong khi đó HydroBASINS (dạng polygon lưu vực) và HydroRIVERS/Hydroviet (dạng đường sông vector) đều khác định dạng với lưới GloFAS, dùng được nhưng phức tạp hơn và dễ lệch nếu khớp sai; cách dùng chính ô lưới làm node, nối theo flow-direction, là cách làm chuẩn cho model dạng lưới, có tiền lệ nghiên cứu thật (HydroGAT: node là raster cell, edge theo flow-direction); và HydroBASINS/HydroRIVERS vẫn giữ vai trò phụ — xác định ranh giới lưu vực (đã dùng ở tiểu luận, Mục 5 #3) hoặc đối chiếu/kiểm tra lại graph dựng từ `ldd`.

Khi implement Graph, nên kết hợp cả fine-tune và train from-scratch theo từng phần, thay vì chọn hẳn một trong hai. Đây là khuyến nghị dựa trên thực hành ML chuẩn, vì hiện chưa có paper nào làm đúng y hệt trường hợp này để trích dẫn trực tiếp. Cụ thể, phần Mamba nên giữ trọng số làm điểm khởi đầu (warm-start), không train lại từ 0 — bắt đầu từ checkpoint bước 1 (bản fine-tune GRDC toàn cầu, chưa qua bước 2/3 riêng cho Vu Gia-Thu Bồn), không dùng checkpoint sau bước 3 của tiểu luận, vì checkpoint đó đã "học lệch" quá riêng cho 1 lưu vực, không phù hợp làm điểm khởi đầu cho một model cần chạy trên nhiều vùng — bước 2/3 của tiểu luận coi như một "nhánh cụt", chỉ phục vụ tiểu luận có kết quả, khóa luận không kế thừa trực tiếp từ đó. Ngược lại, phần Graph hoàn toàn mới, checkpoint chưa từng thấy, nên phải khởi tạo ngẫu nhiên và bắt buộc train from scratch. Toàn bộ model (Mamba cũ cộng Graph mới) sẽ được train cùng lúc, end-to-end, trên dữ liệu nhiều lưu vực Việt Nam. Lý do không thể chỉ "tiếp tục fine-tune" như Phương án B là vì khi chèn Graph vào, đầu vào của các khối Mamba thay đổi (nhận thêm thông tin đã qua Graph), khác hẳn lúc pretrain khi Mamba chỉ thấy chuỗi space-filling curve thuần túy — nên phần Graph mới bắt buộc phải tự học từ đầu.

Có một phương pháp bổ sung, không bắt buộc nhưng được khuyến khích, đúng theo cách Ougahi và cộng sự (2026) đã làm: sau khi train chung, làm thêm một bước "fine-tune riêng từng lưu vực" (regionalization). Bước này gồm hai phần. Đầu tiên là train chung (pooled): train Mamba+Graph một lần trên tất cả lưu vực Việt Nam cùng lúc, với graph gồm nhiều cụm node riêng theo từng lưu vực (không nối giữa các lưu vực với nhau), để model học được đặc trưng thủy văn chung. Sau đó là fine-tune riêng từng lưu vực: fine-tune nhẹ (ít epoch, learning rate nhỏ) riêng cho từng lưu vực, để chỉnh thêm cho đặc điểm riêng (loại đất, độ dốc, vận hành hồ chứa khác nhau) mà bước train chung có thể bỏ sót. Cách làm này có cơ sở khoa học rõ ràng: chính Ougahi và cộng sự đạt NSE=0,85 và KGE=0,80 khi train trên vùng nhiều dữ liệu rồi fine-tune riêng cho từng vùng ít dữ liệu.

Việc thêm cột input mới, ví dụ IBTrACS hay DEM (Mục 5 #5–#6), cũng có thể thực hiện, theo cùng nguyên tắc "warm-start một phần, train mới một phần" như với Graph, chứ không phải chỉ fine-tune đơn thuần. Checkpoint của RiverMamba có lớp input cố định theo đúng số cột đã train (input động cộng 99 biến tĩnh, xem Mục 4), nên không thể nạp checkpoint rồi nhét thêm cột mới vào thẳng. Cách làm đúng là mở rộng lớp input để nhận thêm cột mới: phần trọng số ứng với cột cũ giữ nguyên từ checkpoint, còn phần ứng với cột mới được khởi tạo ngẫu nhiên (bắt buộc train mới), rồi fine-tune lại ít nhất ở lớp input và vài lớp đầu, hoặc cả model nếu cần. Đây là một hướng độc lập với Graph, và có thể làm cả hai cùng lúc nếu muốn.

Cần kiểm tra một điều trước khi tải DEM riêng: LISFLOOD Static Features (96 biến, Mục 4) nhiều khả năng đã có sẵn biến địa hình (elevation/slope) trong số 96 biến tĩnh đó, vì bản thân GloFAS/LISFLOOD cần dữ liệu địa hình để mô phỏng dòng chảy. Paper chỉ nói chung chung "terrain, soil, land use" mà chưa liệt kê hết từng biến, nên cần đối chiếu danh sách 96 biến thật (từ JRC Data Catalogue) khi tới lượt làm file riêng cho nhóm dữ liệu LISFLOOD Static Features. Nếu danh sách đó đã đủ elevation/slope, có thể bỏ nguồn DEM rời (#6), tiết kiệm một bước tải dữ liệu không cần thiết.

---

## 4. Kiến trúc hệ thống — 1 hệ thống dự báo, chỉ đổi lõi bên trong

### 4.1 Cách xây model (🎓 tiểu luận)

Cách làm không phải là "fine-tune một model chung cho mọi việc", mà là dùng thư viện Mamba (`state-spaces/mamba` hoặc bản PyTorch thuần) làm khối xây dựng, rồi tự thiết kế kiến trúc nhận input (mưa, lưu lượng) và cho ra output (dự báo) qua khối Mamba.

Cách làm chính là tải checkpoint RiverMamba đã train toàn cầu, rồi fine-tune cho Vu Gia-Thu Bồn theo hướng transfer learning, đúng theo pipeline 3 bước đã trình bày ở Mục 2.4 (bước 2 và bước 3 đều bắt buộc, lý do đầy đủ ở Mục 2.5).

**Bảng tổng hợp input/output** (theo đúng schema RiverMamba, chỉ đổi vùng địa lý sang Vu Gia-Thu Bồn, không đổi loại cột):

| Loại | Nhóm | Số cột | Đổi theo ngày? | Nguồn |
|---|---|---|---|---|
| Input động (dynamic) | GloFAS reanalysis | 4 | ✅ Có | CDS/EWDS |
| Input động | ERA5-Land | 32 | ✅ Có | CDS |
| Input động | CPC Precipitation | 1 | ✅ Có | NOAA |
| Input tĩnh (static) | LISFLOOD Static Features | 96 | ❌ Không, cố định | JRC |
| Input tĩnh | Tọa độ Cartesian x/y/z | 3 | ❌ Không, cố định | tự tính từ lat/lon, không tải |
| **→ Tổng dùng để train/fine-tune** | | **136** | | |
| Input phụ (optional) | HRES Forecast | 7 | ✅ Có | ECMWF MARS — không nằm trong 136 ở trên, chỉ cộng thêm khi dự báo thật/validation sau 2010 |
| **Output** (không phải input) | Δ discharge — thay đổi lưu lượng, dự báo 1-7 ngày tới | 7 số | — | model tự tạo ra, không tải |

Paper không tự nêu con số tổng 136 này ở bất kỳ đâu — con số này được cộng lại từ Bảng 4-7 và Phụ lục A.1-A.6, đã đối chiếu 2 lần độc lập để xác nhận. Có 2 điểm cần lưu ý khi tra cứu lại. Thứ nhất, 3 biến tọa độ Cartesian (x, y, z) được mô tả ở Phụ lục A.6 — *"we add the Cartesian coordinates for the points on the WGS-84 ellipsoid to enhance the positional encoding"* — không nằm trong 96 biến JRC, mà được tự tính từ lat/lon theo công thức WGS-84, không cần tải. Vậy số biến tĩnh thật sự là 96+3 = 99. Thứ hai, HRES (7 biến) tách riêng, chỉ dùng bổ sung lúc dự báo hoặc validation sau năm 2010, không dùng lúc train chính — paper thay bằng ERA5-Land cho giai đoạn trước 2010.

**Chi tiết từng biến** (để trích dẫn hoặc tra cứu lại khi cần):

| Nhóm | Số biến | Chi tiết |
|---|---|---|
| GloFAS reanalysis | 4 | discharge (`dis24`), runoff (`acc_rod24`), snow depth (`sd`), soil wetness (`swi`) |
| ERA5-Land | 32 | 14 biến tức thời + 18 biến tích lũy theo ngày: nhiệt độ, độ ẩm, tuyết, độ ẩm đất, bốc thoát... (DOI 10.24381/cds.e2161bac) |
| LISFLOOD Static Features | 96 | 7 nhóm nhỏ: địa hình/mạng sông (12), lưới (2), land use (6), thực vật (45), đất (14), nhu cầu nước (3), tham số hiệu chỉnh GloFASv4.0 (14) — cộng lại đúng 96 |
| HRES Forecast | 7 | `e`, `sf`, `sp`, `ssr`, `str`, `t2m`, `tp` |

Năm nhóm dữ liệu trên có độ phân giải lưới khác nhau, không tự khớp sẵn với nhau: GloFAS ở 0.05°, ERA5-Land ở 0.1°, CPC ở 0.5°, HRES theo độ phân giải IFS riêng, còn LISFLOOD Static Features đã ở 0.05° nên khớp sẵn. Paper quy hết về lưới 0.05° của GloFAS bằng các phép nội suy khác nhau cho từng nhóm — bilinear cho ERA5-Land, nearest-point cho CPC, và regrid cho HRES. Bước resample này không có sẵn trong repo RiverMamba, cần tự code lại; chi tiết từng nhóm dữ liệu được ghi trong file riêng, ví dụ `01_glofas.md`.

Output của model không phải mực nước tuyệt đối, mà là thay đổi lưu lượng (Δ discharge) so với ngày trước đó, cho 1 đến 7 ngày tới — giá trị Δ này được cộng dồn vào giá trị hiện tại để ra dự báo tuyệt đối. Ví dụ, nếu hôm nay lưu lượng là 100 m³/s, model chạy một lần sẽ ra luôn 7 số Δ (không phải chạy riêng cho từng ngày): mai Δ=+5 nghĩa là 105 m³/s, mốt Δ=+7 nghĩa là 112 m³/s, và tiếp tục như vậy. Đây gọi là "rolling forecast" — mỗi ngày chạy lại một lần, và mỗi lần luôn nhìn 7 ngày tới kể từ ngày hiện tại.

Lưu lượng (m³/s) và mực nước (mét) là hai đại lượng khác nhau — model chỉ dự báo lưu lượng, đúng theo cách GloFAS và RiverMamba làm. Muốn có mực nước thật cần thêm một bước chuyển đổi qua "rating curve" (quan hệ lưu lượng và mực nước), và công thức này riêng cho từng mặt cắt sông cụ thể, cần khảo sát thực địa, hiện chưa có sẵn miễn phí cho Vu Gia-Thu Bồn. Vì vậy, tiểu luận đánh giá "nguy cơ lũ" dựa trên việc lưu lượng vượt ngưỡng theo return period (chuẩn ngành, RiverMamba cũng làm theo cách này — xem Mục 6), không quy đổi ra mét.

Cần phân biệt "1 lần/ngày" với "chỉ dự báo được 1 ngày": tần suất cập nhật (bao lâu chạy lại model một lần, ở đây là 1 lần/ngày) khác với tầm dự báo (mỗi lần chạy dự báo xa bao nhiêu, ở đây là 1-7 ngày tới). Mỗi lần chạy, model dự báo cả 7 ngày tới cùng một lúc. Việc không real-time được không phải do đề tài tự giới hạn, mà do chính nguồn dữ liệu: `cems-glofas-forecast` bản thân chỉ tự cập nhật 1 lần mỗi ngày, không có dữ liệu mới hơn tần suất đó để lấy — đây cũng là cách các cơ quan dự báo lũ thật vận hành, kể cả GloFAS gốc.

### 4.2 Kiến trúc RiverMamba bên trong — encoder-decoder, không chỉ 1 khối Mamba

RiverMamba có cấu trúc encoder-decoder rõ ràng, không phải một khối Mamba đơn lẻ nhận input rồi ra thẳng output.

**Bảng tra nhanh thông số:**

| Thành phần | Số lượng/thông số | Ghi chú |
|---|---|---|
| Encoder (Hindcast layers) | 3 lớp | Mamba 2 chiều, nén T=4 ngày còn T=1 |
| Decoder (Forecast layers) | 7 khối (L=7) | Mỗi khối lo đúng 1 ngày dự báo |
| Hidden dimension (encoder) | K=192 | |
| Hidden dimension (decoder) | 192+64=256 | Cộng thêm 64 chiều từ HRES |
| D_state (khối Mamba) | 16 | |
| D_conv (khối Mamba) | 4 | |
| Dropout lúc pretrain | 0,2 | |
| Dropout lúc fine-tune GRDC | 0,4 | Tăng vì dữ liệu ít hơn, chống overfit |
| Đầu ra (regression heads) | 7 MLP riêng biệt | Mỗi đầu ra đúng 1 số Δ discharge/ngày |
| Số tham số (parameter count) | Paper không công bố | |

Phần encoder, gọi là "Hindcast layers", có nhiệm vụ nén 4 ngày dữ liệu lịch sử thành 1 bản tóm tắt. Phần này gồm 3 lớp, mỗi lớp là 1 khối Mamba hai chiều (bidirectional — đọc xuôi rồi đọc ngược, sau đó gộp lại), khác với Mamba nguyên bản vốn chỉ đọc xuôi vì thiết kế ban đầu cho văn bản. Lớp đầu tiên xử lý đủ 4 ngày input (T=4); mỗi lớp sau đó nén thời gian còn một nửa, nên qua 3 lớp thì còn lại đúng 1 bản tóm tắt (T=1). Hidden dimension của phần này là K=192.

Phần decoder, gọi là "Forecast layers", có nhiệm vụ tạo ra 7 ngày dự báo. Phần này gồm 7 khối riêng biệt (đúng bằng L=7 ngày dự báo), mỗi khối chuyên trách 1 ngày trong tương lai, không dùng chung 1 khối cho cả 7 ngày. Mỗi khối nhận bản tóm tắt từ encoder, cộng thêm dữ liệu dự báo thời tiết HRES nếu có (chỉ áp dụng cho giai đoạn sau 2010) — 64 chiều từ HRES được nối thêm vào 192 chiều cũ, thành 256. Đầu ra cuối cùng gồm 7 "đầu" MLP riêng biệt, mỗi đầu dự báo đúng 1 số Δ discharge cho 1 ngày.

LOAN (Location-Aware Adaptive Normalization) là cách mà 99 biến tĩnh được "tiêm" vào model, không phải ghép thẳng vào input như các biến động:

```
LOAN(X) = chuẩn_hóa(X) + GELU(Linear(biến_tĩnh))
```

Nói cách khác, dữ liệu động (mưa, độ ẩm...) vẫn được chuẩn hóa như bình thường, nhưng model cộng thêm một "độ lệch" tính riêng từ đặc điểm địa hình/đất tại đúng điểm đó. Nhờ vậy, hai điểm cùng lượng mưa nhưng địa hình khác nhau (dốc so với bằng phẳng) sẽ được xử lý khác nhau ngay từ bước này. Cơ chế LOAN xuất hiện ở cả encoder lẫn decoder.

Về cách "duỗi" lưới 2D thành chuỗi 1D, paper dùng một kết hợp cụ thể chứ không chọn ngẫu nhiên một loại đường quét: Sweep (quét ngang rồi quét dọc) được dùng ở 2 lớp encoder đầu, còn Gilbert curve (một biến thể của Hilbert curve) được dùng ở lớp sau. Các ngày liên tiếp được nối đuôi nhau — điểm cuối của chuỗi ngày t chính là điểm đầu của chuỗi ngày t+1.

### 4.3 Kết quả thật paper báo cáo — số liệu tham chiếu khi làm thí nghiệm

Paper báo cáo kết quả bằng chỉ số R² (coefficient of determination), không phải NSE như Mục 6 khuyến nghị dùng cho tiểu luận. Hai chỉ số này có liên quan nhưng công thức khác nhau, nên cần phân biệt rõ khi so trực tiếp số của mình với số của paper.

**Bảng 1 — target = GloFAS, test 2021-2024:**

| Model | R² | KGE | F1-score |
|---|---|---|---|
| Climatology | 0,135 | 0,245 | – |
| Persistence | 0,683 | 0,841 | 0,322 |
| LSTM | 0,849 | 0,892 | 0,358 |
| **RiverMamba** | **0,873** | **0,913** | **0,459** |

**Bảng 3 — target = GRDC thật, test 2021-2023:**

| Model | R² | KGE | F1-score |
|---|---|---|---|
| Climatology | -0,001 | 0,087 | – |
| Persistence | 0,066 | 0,392 | 0,146 |
| GloFAS Reforecast | 0,289 | 0,494 | 0,204 |
| LSTM | 0,462 | 0,614 | 0,148 |
| **RiverMamba** | **0,506** | **0,661** | **0,243** |

Mọi chỉ số ở Bảng 3 (so với GRDC thật) đều thấp hơn hẳn so với Bảng 1 (so với GloFAS), kể cả với chính RiverMamba — R² giảm từ 0,87 xuống còn 0,51. Đây là bằng chứng số liệu thật cho thấy việc "so với thực tế" luôn khó hơn hẳn "so với mô phỏng" (lý do đầy đủ đã trình bày ở Mục 2.5).

Về tài nguyên tính toán thật đã dùng: giai đoạn pretrain mất khoảng 3 ngày trên 16 GPU (A100 80GB/48GB); giai đoạn fine-tune GRDC mất khoảng 4 giờ trên 16 GPU. Cách dự báo của paper là kiểu deterministic (ra 1 giá trị mỗi lần), không phải ensemble. Paper không công bố tổng số tham số của model. Ý nghĩa cho tiểu luận là: quy mô 16 GPU chạy nhiều ngày nói trên áp dụng cho việc train toàn cầu (1,5 triệu điểm) — còn việc fine-tune cho 1 lưu vực nhỏ (chỉ vài trăm điểm) sẽ nhẹ hơn rất nhiều, càng củng cố nhận định ở Mục 8.2 rằng 1 GPU miễn phí (Kaggle/Colab) là đủ dùng.

### 4.4 Chi tiết training thật — loss function và độ trễ dữ liệu

Chi tiết dưới đây được xác nhận trực tiếp từ code thật trong repo (`models/loss.py`, `train.py`, `dataset/RiverMamba_dataset.py`), không chỉ dựa vào mô tả bằng lời của paper.

Trước khi tính loss, giá trị được biến đổi bằng công thức log1p có dấu:

```
forward:  log1p_transform(x)     = sign(x) * log1p(|x|)
inverse:  log1p_inv_transform(x) = sign(x) * expm1(|x|)
```

Phép biến đổi này được áp dụng lên Δ discharge (target) trước khi đưa vào loss, vì lưu lượng có dải giá trị rất rộng — từ vài m³/s ở sông nhỏ tới hàng nghìn m³/s ở sông lớn. Việc nén bằng log giúp model không bị chi phối bởi các điểm sông lớn, trong khi thành phần `sign(x)` vẫn giữ lại được chiều tăng/giảm, vì Δ có thể âm. Loss chính là MSE (hoặc L1) tính trên giá trị đã biến đổi này, sau đó nhân thêm một trọng số theo từng điểm — không dùng MSE trần trên giá trị gốc.

Trọng số trong loss gồm 2 phần nhân với nhau, không phải một trọng số đều cho mọi điểm. Phần thứ nhất tính theo return period: điểm nào tại thời điểm đó vượt 1 trong 9 ngưỡng return period (1,5/2/5/10/20/50/100/200/500 năm, cùng bộ ngưỡng đã dùng ở Mục 6) sẽ được nhân một hệ số ưu tiên cao hơn hẳn mức thường, để đảm bảo model không "lười" chỉ tối ưu cho đa số điểm bình thường mà bỏ qua đúng lúc lũ hiếm hoặc nghiêm trọng xảy ra. Đây chính là cơ chế đứng sau nhận định ở Bảng 2a của paper, rằng trọng số này giúp cải thiện F1-score đáng kể. Phần thứ hai tính theo lead time, tức ngày dự báo xa bao nhiêu: trọng số giảm dần theo công thức `exp(|i - T - 1| * alpha)`, khiến ngày dự báo gần (ví dụ ngày mai) được ưu tiên trọng số cao hơn hẳn ngày xa (ví dụ ngày thứ 7) — điều này phản ánh việc dự báo gần cần chính xác hơn dự báo xa, vốn đã khó và có sai số tự nhiên lớn hơn.

Một chi tiết quan trọng khác là độ trễ dữ liệu input khi ghép ngày "hiện tại", nhằm tránh rò rỉ dữ liệu tương lai. Paper dịch lùi input theo đúng độ trễ công bố thật của từng nguồn, không dùng thẳng dữ liệu của ngày t: GloFAS reanalysis và ERA5-Land được lùi 1 ngày (t-1), còn CPC được lùi 2 ngày (t-2). Trích nguyên văn: *"To ensure that we do not include any data after 00:00 UTC and thus mimic a realistic deployment of RiverMamba, we shift X_GloFAS and X_ERA5 by −1 day and X_CPC by −2 days."* CPC cần độ trễ dài hơn vì đây là dữ liệu dựa trên trạm đo mưa mặt đất, cần thời gian kiểm tra chất lượng trước khi công bố (điều này khớp với ghi chú về Consolidated/Intermediate ở `01_glofas.md`).

Ý nghĩa cho tiểu luận là: đây là chi tiết bắt buộc phải làm đúng khi code inference thật, không chỉ lúc train/test offline trên dữ liệu lịch sử. Nếu ghép input ngày "hôm nay" bằng đúng dữ liệu GloFAS/ERA5-Land/CPC của chính ngày hôm nay, thay vì lùi đúng 1-2 ngày theo độ trễ công bố thật, sẽ vô tình dùng dữ liệu chưa từng có thật tại thời điểm dự báo, không phản ánh đúng cách hệ thống sẽ vận hành trong thực tế.

Về HydroRIVERS, cần làm rõ một điểm dễ hiểu nhầm: paper có một phụ lục riêng thử nghiệm dùng HydroRIVERS (dữ liệu thuộc tính sông phổ biến) làm nguồn biến tĩnh thay thế, so với LISFLOOD Static Features. Đây là một so sánh về nguồn dữ liệu tĩnh, không phải thử nghiệm dựng graph/GNN — paper vẫn không dùng graph thật, đúng như đã ghi ở Mục 3.1. Không tìm thấy code công khai cho phần này trong repo, chỉ có mô tả trong bài viết. Hiện chưa cần hành động gì cho tiểu luận hay khóa luận, chỉ ghi lại để biết khi cần đối chiếu thêm về sau.

### 4.5 Sơ đồ hệ thống

**🎓 Giai đoạn tiểu luận:**

```
┌─────────────────────────────────────────────────────────┐
│ HỆ THỐNG DỰ BÁO (nghiên cứu chính)                        │
│ Lõi model: Mamba thuần                                    │
│ Input: cems-glofas-forecast → dự báo lưu lượng 1-7 ngày  │
│ Cập nhật: 1 lần/ngày (đúng chuẩn ngành thủy văn)         │
└─────────────────────────────────────────────────────────┘
```

**🎓🎓 Giai đoạn khóa luận — chỉ đổi lõi, kiến trúc tổng thể giữ nguyên:**

```
┌──────────────────────────────────────────────────────────┐
│ HỆ THỐNG DỰ BÁO (nâng cấp)                                │
│ Lõi model: Mamba + Graph (G-Mamba) — VẪN LÀ 1 MODEL       │
│ Graph dựng từ ldd (local drainage direction) của GloFAS   │
│ — khớp đúng lưới model, giúp model "nhìn" được quan hệ    │
│ thượng nguồn/hạ nguồn giữa các điểm cùng lúc với           │
│ chuỗi thời gian                                            │
│ Input: cems-glofas-forecast (như cũ)                      │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Bảng đầy đủ Dataset & API

| # | Nguồn | Giai đoạn | Vai trò | Free? | Link/Cách lấy |
|---|---|---|---|---|---|
| 1 | `cems-glofas-historical` | 🎓 Tiểu luận | Huấn luyện model (1979–nay) | ✅ Free, cần tài khoản ECMWF | `ewds.climate.copernicus.eu/datasets/cems-glofas-historical` |
| 2 | `cems-glofas-forecast` | 🎓 Tiểu luận | Dự báo thật hàng ngày | ✅ Free, cùng tài khoản | `ewds.climate.copernicus.eu/datasets/cems-glofas-forecast` |
| 3 | HydroBASINS (HydroSHEDS/WWF) | 🎓 Tiểu luận (xác định ranh giới lưu vực) → 🎓🎓 Khóa luận (đối chiếu/kiểm tra Graph, vai trò phụ trợ — #11) | Cần ở tiểu luận để biết đúng ô lưới GloFAS nào thuộc Vu Gia-Thu Bồn | ✅ Free | hydrosheds.org/products/hydrobasins |
| 4 | Hydroviet/shapefiles (GitHub) | 🎓 Tiểu luận → 🎓🎓 Khóa luận (tương tự #3) | Repo có shapefile khu vực Mekong (xác nhận có `vnreservoirs.zip` — hồ chứa VN; chưa xác nhận chắc có shapefile sông cụ thể, cần tự kiểm tra danh sách file khi tải). Nếu không có shapefile sông, dùng HydroBASINS (#3) là đủ | ✅ Free | github.com/Hydroviet/shapefiles |
| 5 | IBTrACS (NOAA) | 🎓🎓 Khóa luận | Mở rộng: đường đi bão lịch sử | ✅ Free | data.humdata.org/dataset/ibtracs |
| 6 | SRTM / Copernicus DEM | 🎓🎓 Khóa luận | Mở rộng: địa hình/độ dốc — nghi ngờ trùng lặp với LISFLOOD Static Features (Mục 4), cần đối chiếu danh sách 96 biến thật trước khi quyết định có cần tải riêng không (Mục 3.1) | ✅ Free | thư viện Python `elevation` |
| 7 | GRanD / Global Dam Watch | 🎓🎓 Khóa luận | Mở rộng: vị trí hồ thủy điện (không có xả lũ real-time) | ✅ Free (vị trí), ❌ (xả lũ real-time cần xin EVN) | globaldamwatch.org/data |
| 8 | RiverMamba code + pretrained checkpoint | 🎓 Tiểu luận | Code gốc để học kiến trúc + trọng số đã train sẵn để fine-tune (Phương án B) | ✅ Free | github.com/HakamShams/RiverMamba_code |
| 9 | Sentinel-1 SAR + Copernicus Global Flood Monitoring (GFM) | 🎓 Tiểu luận (bonus, khuyến khích — paper gốc có tiền lệ trực tiếp: phụ lục riêng làm 8 case study lũ thật có bản đồ trực quan — Tây Âu 2021, Đông/Trung Âu 2024, Tây Ban Nha, Đức, Kenya-Tanzania, California, Trung Quốc) | Bản đồ vùng ngập thực tế từ ảnh vệ tinh radar, gần thời gian thực — dùng đối chiếu trực quan cảnh báo của model với ảnh vệ tinh thật tại đợt lũ miền Trung 2025 (đúng theo cách paper trình bày case study), không cần trạm đo | ✅ Free — xem trên trình duyệt (Copernicus Browser) không cần đăng ký, nhưng tải qua API cần tài khoản Copernicus miễn phí (giống tài khoản EWDS) | global-flood.emergency.copernicus.eu |
| 10 | Trạm thủy văn **Nông Sơn + Thành Mỹ** (ưu tiên cao nhất, dữ liệu dài hạn đã kiểm chứng) + **Câu Lâu** (xin thêm, khả năng cao có Q nhưng chưa rõ từ năm nào) + Ái Nghĩa (xin thêm, không có Q) — dữ liệu hiện tại, chính thức KTTV VN | 🎓 Tiểu luận (khuyến khích mạnh — #12 là bản free thay thế tạm cho Nông Sơn, cách duy nhất có ground truth thật không dính vấn đề đập, Mục 2.5) | Số liệu quan trắc thật, cập nhật gần đây. Sau khi đọc ảnh chụp thật bảng Phụ lục I (289/QĐ-TTg) và đối chiếu paper Nguyen et al. 2024 (khảo sát thực địa, xem ghi chú bên dưới): Nông Sơn (Thu Bồn) và Thành Mỹ (thượng Vu Gia) chắc chắn có lưu lượng (Q) dài hạn, phủ đủ 2 nhánh hợp lưu. Câu Lâu nhiều khả năng cũng có Q (cột "Ghi chú" trong văn bản chỉ đánh dấu phù sa R là yếu tố mới, không phải Q) nhưng chưa rõ dữ liệu bắt đầu từ năm nào, nên đáng xin thêm và cần hỏi KTTV xác nhận độ dài dữ liệu. Hiệp Đức thì có Q là yếu tố mới đang thêm (2021-2030), khả năng cao chưa có lịch sử dài. Ái Nghĩa không có Q — chỉ mực nước/mưa — vẫn xin thêm được (gần vùng lũ thật, có giá trị case study) nhưng không nên kỳ vọng có lưu lượng. Việc này không vội vì dự kiến hoàn thành tiểu luận khoảng 11-12/2026, nhưng nên nộp đơn ngay | ⚠️ Chưa xác nhận lại được con số ~280.000 VNĐ/trạm/năm. Trang thủ tục thật chỉ ghi "thông báo mức phí đối với trường hợp phải trả phí" (báo phí sau khi nộp hồ sơ, không nêu số cụ thể) — con số 280k trước đây lấy từ Thông tư 197/2016/TT-BTC (văn bản phí riêng), có thể vẫn đúng nhưng chưa tái xác nhận được từ chính trang thủ tục. Cá nhân/sinh viên nộp đơn được, xử lý ~1 ngày làm việc + 1 ngày sau khi thanh toán | `dichvucong.monre.gov.vn/pages/ChiTietDichVuTrucTuyen.aspx?dv=352` (mẫu đơn Phụ lục 4, Nghị định 38/2016/NĐ-CP) — đã tự vào kiểm tra, hoạt động đúng, khớp nội dung. Link cũ `dichvucong.gov.vn/...ma_thu_tuc=1.001149` và `dichvucong.mae.gov.vn` đều đã lỗi, không dùng nữa — domain đúng là `dichvucong.monre.gov.vn` |
| 11 | LISFLOOD static maps for GloFAS — biến `ldd` (local drainage direction) | 🎓🎓 Khóa luận | Nguồn chính để dựng Graph (node=ô lưới, edge=hướng dòng chảy) — khớp chính xác lưới GloFAS, tốt hơn dùng HydroBASINS/HydroRIVERS vì không cần bước khớp tọa độ giữa 2 định dạng khác nhau | ✅ Free | data.jrc.ec.europa.eu/dataset/68050d73-9c06-499c-a441-dc5053cb0c86 |
| 12 | **GRDC (Global Runoff Data Centre)** — trạm `2371300 NONG SON`, sông Song Tranh (nhánh Thu Bồn), 15,7167°N/108,0167°E | 🎓 Tiểu luận (bắt buộc cho bước 3, Mục 2.5) | Bản miễn phí của cùng vị trí Nông Sơn, nhưng chỉ có 1978–1990 (thiếu 56,3% ngày), trước đập Sông Tranh 2 (hoàn thành 8/2011) — dùng làm nguồn chính cho bước 3 trong lúc chờ #10, hoặc làm kiểm tra chéo độ chính xác GloFAS thời kỳ trước đập. Tổng VN có 27 trạm GRDC, 26 trạm còn lại ở lưu vực khác (Trà Khúc, Vệ, Hương, Đà Rằng, Lam, Hiếu, Kontum, Đắk Lắk, Mekong Delta...), giữ cho khóa luận | ✅ Free (điều khoản: chỉ dùng nghiên cứu, không thương mại, không phân phối lại) | portal.grdc.bafg.de → "Download by Station" → lọc Country = Viet Nam |
| 13 | `cems-glofas-reforecast` | 🎓 Tiểu luận (cần cho baseline "GloFAS Reforecast", Mục 2.6) | Dự báo tổ hợp lịch sử (ECMWF-ENS, 11 thành viên), phát hành 2 lần/tuần, nhìn xa 46 ngày, phủ 2003-2022 — dùng làm baseline khi đánh giá so với GRDC thật (bước 3), khác với #1 (Historical, dùng làm target train). ⚠️ Chưa xác nhận cách hòa giải cấu trúc 46 ngày/2 lần-tuần với đánh giá hàng ngày 1-7 ngày như paper làm — cần tự kiểm tra cấu trúc dữ liệu thật khi tải | ✅ Free, cùng tài khoản EWDS | `ewds.climate.copernicus.eu/datasets/cems-glofas-reforecast` |

⚠️ **Lưu ý:** link CDS cũ (`cds.climate.copernicus.eu`) đã ngừng host dữ liệu CEMS — chỉ dùng link EWDS ở trên.

⚠️ **Lưu ý lịch:** Global Flood Monitoring bảo trì tạm ngưng 31/7 – 4/8/2026 — tránh tải dữ liệu trong khung này.

📝 **Ghi chú:** DAHITI không dùng, vì không phủ đúng vùng — không xác nhận được trạm nào tại Vu Gia-Thu Bồn.

📝 **Kênh liên hệ tác giả bổ sung, chạy song song với đơn KTTV:** có một paper khoa học khác — đánh giá bồi lắng/biến đổi hình thái lưu vực Vu Gia-Thu Bồn (Nguyen et al., PMC/ScienceDirect, PMC11168294) — trích dẫn *"continuous daily discharge and suspended sediment concentrations (SSC) data spanned the 1996–2020 period were obtained at two stations located in the upstream region of the VGTB River basin, namely, Thanh My and Nong Son"*. Đây là lưu lượng thật (không phải mực nước) tại 2 trạm, phủ cả 2 nhánh: Thanh Mỹ (nhánh Vu Gia) và Nông Sơn (nhánh Thu Bồn), kéo dài đến 2020, tức sau đập Sông Tranh 2, nên phản ánh dòng chảy đã điều tiết — tốt hơn hẳn so với chỉ có 1 trạm. Paper không nêu tên cơ quan cấp dữ liệu, chỉ ghi *"available from the corresponding author upon reasonable request"*, tức không có link tải công khai và không phải một kênh mới để tự tải. Tuy vậy, đáng thử liên hệ tác giả (Binh Quang Nguyen, ĐH Bách khoa Đà Nẵng, `nqbinh@dut.udn.vn`) song song với đơn KTTV, vì đây đúng định dạng lưu lượng cần cho bước 3 và phủ đủ 2 nhánh — khác với paper GNN-Transformer trước đó vốn chỉ có mực nước, và cũng khác với đơn KTTV hiện đang xin đúng Nông Sơn+Ái Nghĩa chứ không phải Thanh Mỹ+Nông Sơn.

📝 **Trạm nào có lưu lượng (Q) trong Vu Gia-Thu Bồn** — đọc lại từ ảnh chụp thật bảng Phụ lục I (289/QĐ-TTg), trong đó có cột "Ghi chú" nêu rõ yếu tố nào mới được thêm vào, đối chiếu thêm với paper Nguyen et al. 2024 (Heliyon, PMC11168294, khảo sát thực địa 2021-2022): *"These are the only two stations in the basin that monitor discharge and SSC"* (dữ liệu paper này tính tới 2020):

| Trạm | Sông | Trạng thái Q | Vì sao |
|---|---|---|---|
| Nông Sơn | Thu Bồn | ✅ Chắc chắn, ổn định lâu dài | Không có ghi chú thay đổi; khớp thêm paper Nguyen 2024 |
| Thành Mỹ | Cái (thượng Vu Gia) | ✅ Chắc chắn, ổn định lâu dài | Không có ghi chú thay đổi; khớp thêm paper Nguyen 2024 |
| Câu Lâu | Thu Bồn | 🟡 Nhiều khả năng có, chưa rõ từ năm nào | Ghi chú chỉ ghi "Tăng yếu tố đo R" — nghĩa là Q đã có sẵn trước quy hoạch, chỉ R là mới |
| Sông Thanh, Hà Tân, Thác Cạn, Tiên Phước | Nhánh phụ nhỏ | 🟡 Nhiều khả năng có, chưa rõ từ năm nào | Không có ghi chú "mới thêm Q" |
| Sông Hàn | Hàn (Đà Nẵng, cửa sông) | 🟡 Nhiều khả năng có, chưa rõ từ năm nào | Không có ghi chú "mới thêm Q" |
| Hiệp Đức | Thu Bồn | 🟠 Q là yếu tố MỚI (2021-2030) | Ghi chú ghi rõ "Tăng yếu tố đo Q" — khả năng cao chưa có lịch sử dài |
| Ái Nghĩa | Yên (Vu Gia) | ❌ Không có Q | Chỉ mực nước/mưa |
| Hội Khách | Vu Gia | ❌ Không có Q | Chỉ mực nước/mưa, dù tên đúng "sông Vu Gia" |
| Tam Kỳ | Tam Kỳ | ❌ Không có Q, khác lưu vực | Xác nhận không thuộc Vu Gia-Thu Bồn — validate lỗi paper GNN-Transformer 2026 (Mục 2.7) |

Ý nghĩa cho việc chọn trạm xin KTTV: Nông Sơn và Thành Mỹ vẫn là 2 trạm chắc chắn nhất, vì có dữ liệu dài và đã được kiểm chứng độc lập. Câu Lâu, Sông Thanh, Hà Tân, Thác Cạn, Tiên Phước, Sông Hàn nhiều khả năng đã có Q sẵn nên đáng xin thêm nếu cần mở rộng, dù quy mô tiểu lưu vực nhỏ khiến giá trị đại diện thấp hơn Nông Sơn/Thành Mỹ/Câu Lâu, và cần hỏi KTTV xác nhận độ dài dữ liệu trước. Riêng Hiệp Đức là trường hợp khác hẳn, vì Q được đánh dấu rõ là yếu tố mới đang thêm, khả năng cao chưa có lịch sử dài, nên cần hỏi riêng.

📝 **#10 vs #12 — không thay thế nhau:** #10 (trả phí) có dữ liệu hiện tại, dùng để đánh giá model dự báo bây giờ. #12 (free) chỉ có dữ liệu 1978-1990, trước đập, và là nguồn dùng cho bước 3 bắt buộc (Mục 2.5) khi chưa có #10; giá trị phụ thêm của #12 là kiểm tra chéo lịch sử, nhưng không thay thế được #10 cho mục đích đánh giá dự báo hiện tại. Nên làm cả 2: dùng #12 ngay vì đã có sẵn và miễn phí, bắt buộc phải có ít nhất bản này, đồng thời nộp đơn #10 song song mà không cần đợi kết quả #12 xong mới nộp.

---

## 6. Chỉ số đánh giá model (metrics chuẩn ngành thủy văn) — dùng từ tiểu luận

Bài toán của đề tài có 2 tầng, nên cần đánh giá theo cả 2 kiểu — regression và classification — chứ không chỉ 1 kiểu. Tầng thứ nhất, model (Mamba) làm regression, dự báo ra một số liên tục là Δ lưu lượng (m³/s). Tầng thứ hai, kết quả đó được so với ngưỡng "return period" để ra một kết quả classification — có xảy ra lũ ở mức đó hay không. Ngưỡng này được tính bằng cách fit phân phối Gumbel (extreme value distribution) trên lưu lượng lớn nhất mỗi năm trong lịch sử, dùng phương pháp L-moments (RiverMamba dùng dữ liệu 1979–2022 để tính); ví dụ, "lũ 2 năm" nghĩa là mức lưu lượng có 50% khả năng bị vượt trong một năm bất kỳ. Copernicus đã tính sẵn ngưỡng này cho GloFAS trên toàn cầu, nên không cần tự tính lại từ đầu.

Về việc chia mấy cấp, RiverMamba tính threshold ở 9 mức (1,5 / 2 / 5 / 10 / 20 / 50 / 100 / 200 / 500 năm), nhưng khi báo cáo F1-score chính, paper chỉ dùng 5 mức thấp — 1,5/2/5/10/20 năm (Bảng 1 và Hình 5c của paper) — không dùng các mức 50-500 năm. Lý do nhiều khả năng là vì sự kiện càng hiếm càng khó đánh giá tin cậy khi dữ liệu lịch sử/test có hạn (1979–2022, khoảng 44 năm), tuy điều này chưa được xác nhận trực tiếp từ paper. Khuyến nghị cho tiểu luận là dùng đúng 5 mức 1,5/2/5/10/20 năm như RiverMamba báo cáo chính.

Việt Nam có một hệ thống chính thức riêng là thang Báo động lũ cấp 1/2/3 (Quyết định 05/2020/QĐ-TTg, 18/2021/QĐ-TTg), dễ hiểu hơn với hội đồng, nhưng thang này tính theo mực nước (mét) tại từng trạm, không phải lưu lượng. Vì model chỉ ra lưu lượng, nên không áp dụng trực tiếp thang này được — cần một rating curve, hiện chưa có sẵn miễn phí. Vì vậy, nên giữ Báo động 1/2/3 làm điểm liên hệ thực tế khi viết báo cáo, và chỉ dùng thật được nếu sau này có mực nước thật từ trạm Nông Sơn/Ái Nghĩa (Mục 5 #10).

Cũng có nghiên cứu quốc tế thật làm multi-class severity classification cho lũ, không chỉ dừng ở nhị phân — cơ quan khí tượng Mỹ (NWS) dùng thang 4 cấp, và nghiên cứu tại Bangladesh cũng làm phân loại đa cấp bằng ML. Vậy hướng phân nhiều cấp là chuẩn phổ biến trong ngành, không phải điều tự nghĩ ra riêng cho đề tài này.

Ngành thủy văn không chỉ dùng RMSE/MAE thông thường mà có bộ chỉ số riêng, chia làm 2 nhóm.

**Nhóm regression (đánh giá độ khớp số liệu):**

| Chỉ số | Ý nghĩa | Ngưỡng "đạt" |
|---|---|---|
| **NSE** (Nash–Sutcliffe Efficiency) | Đo mức khớp giữa dự báo và thực tế | NSE > 0,5 = đạt |
| **KGE** (Kling–Gupta Efficiency) | Tổng hợp tương quan + biến động + độ lệch | Càng gần 1 càng tốt |
| **RSR** | Tỷ lệ RMSE/độ lệch chuẩn | RSR < 0,7 = đạt |
| **PBIAS** | % sai lệch thiên vị (dự báo cao/thấp hơn thực tế) | Càng gần 0 càng tốt |

**Nhóm classification (đánh giá có bắt được lũ hay không — quan trọng không kém, RiverMamba paper cũng dùng):**

| Chỉ số | Ý nghĩa |
|---|---|
| **F1-score** (theo từng mức return period 1,5–20 năm) | So dự báo vượt ngưỡng có khớp với thực tế vượt ngưỡng không — đo khả năng "cảnh báo lũ đúng lúc", cái NSE/KGE không đo được trực tiếp |

Ở tiểu luận (🎓), nên dùng cả 4 chỉ số regression cộng thêm F1-score để so sánh Mamba với LSTM/GRU/Transformer baseline trên 1 lưu vực — nếu thiếu F1-score sẽ không đánh giá được đúng trọng tâm "dự báo lũ", mà chỉ đo được mức "khớp số liệu chung chung".

Ở khóa luận (🎓🎓), có thêm một lưu ý phương pháp luận: NSE/KGE nhạy với đặc tính dòng chảy hơn là hiệu năng thật của model, nên khi so sánh nhiều lưu vực sông khác nhau, nên dùng thêm RMSE chuẩn hóa (NRMSE) hoặc percent bias để so sánh công bằng hơn giữa các vùng.

---

## 7. Hạn chế & lưu ý đạo đức (áp dụng chung cả 2 giai đoạn — đưa vào phần "Hạn chế của đề tài")

Về cảnh báo giả (false alarm): một hệ thống AI dự báo sai có thể làm người dân "nhờn" cảnh báo, giảm tin tưởng về lâu dài. Cần nêu rõ trong báo cáo rằng đây là một công cụ hỗ trợ, không thay thế cảnh báo chính thức của cơ quan nhà nước.

Về trách nhiệm giải trình, nên có một tuyên bố rõ ràng trong báo cáo rằng hệ thống mang tính nghiên cứu/demo, không dùng để ra quyết định sơ tán thực tế nếu chưa qua kiểm định của cơ quan chuyên môn.

Về thiên lệch dữ liệu, GloFAS được hiệu chỉnh tốt hơn ở những lưu vực lớn có nhiều trạm quan trắc, nên cần nêu rõ hạn chế này khi áp dụng cho một lưu vực nhỏ như ở miền Trung. Hạn chế này có thể giảm bớt nếu xin được dữ liệu lưu lượng tại trạm Nông Sơn/Thành Mỹ (Mục 5 #10).

Một hạn chế cấu trúc quan trọng là GloFAS không mô phỏng đập/hồ chứa — hạn chế này chính paper gốc cũng tự thừa nhận và có thể trích dẫn thẳng: *"GloFAS reanalysis dataset only simulates the naturalized flow without considering realistic human interventions such as dams, reservoirs, diversions, irrigation withdrawals... this can be a major source of bias in GloFAS compared to the GRDC data"*, và ở phần Kết luận: *"observational data are affected by human interventions like dams and there is a need to integrate such interventions in the model"*. Vu Gia-Thu Bồn có nhiều đập thủy điện (Sông Tranh 2, A Vương, Đắk Mi 4, Sông Bung...), nên model sẽ không "biết" lịch vận hành đập. Đây là một hạn chế cấu trúc, không thể sửa được chỉ bằng cách train nhiều hơn hay dùng dữ liệu tốt hơn, đúng như paper gốc tự nhận rằng *"this introduces biases that models cannot learn"*. Bản thân RiverMamba gốc cũng không xử lý được vấn đề này — đây là hạn chế chung của cả hướng tiếp cận, không riêng gì đề tài này.

Cuối cùng, hệ thống cần có con người giám sát (human-in-the-loop), và không nên được trình bày như một hệ thống "tự động ra quyết định hoàn toàn".

---

## 8. Hạ tầng triển khai (làm 1 lần, dùng chung cho cả 2 giai đoạn)

### 8.1 VPS deploy (hosting inference + dashboard, không dùng để train)

VPS sử dụng là Oracle Cloud Always Free, với home region là Singapore, và Phoenix/Chicago làm phương án dự phòng — ưu tiên region có 3 Availability Domain để dễ săn capacity.

Cấu hình free hiện tại là 2 OCPU/12GB RAM. Oracle đã âm thầm cắt giảm từ mức cũ 4 OCPU/24GB xuống mức này, có hiệu lực từ 15/6/2026, và quota theo tháng cũng giảm theo mà không có thông báo chính thức. Dù vậy, cấu hình 2 OCPU/12GB vẫn đủ để host dashboard và inference, vì việc Mamba suy luận rất nhẹ, không phải training — nên không cần trả thêm phí cho phần này.

Khi đăng ký tài khoản Oracle, cần đặt tên tenancy cẩn thận vì không đổi được về sau, tắt VPN lúc đăng ký, và dùng thẻ thật, không dùng thẻ ảo/prepaid.

### 8.2 Compute để train/fine-tune model (VPS Oracle không làm được việc này)

RiverMamba gốc được test trên GPU A100/RTX 3090 (24GB VRAM), nhưng đó là để train ở quy mô toàn cầu (6,2 triệu điểm lưới, khoảng 10TB dữ liệu). Việc fine-tune chỉ cho 1 lưu vực nhỏ như Vu Gia–Thu Bồn sẽ nhẹ hơn rất nhiều, và nhiều khả năng chạy vừa trên một GPU miễn phí 16GB.

Lý do bắt buộc cần GPU, không thể chạy bằng CPU, là vì thư viện Mamba chuẩn (`state-spaces/mamba`) dùng các CUDA kernel riêng (`selective_scan_cuda`, `causal_conv1d_cuda`) để đạt được tốc độ tuyến tính. Việc train chỉ bằng CPU là không khả thi, nên VPS Oracle (kiến trúc ARM, không có GPU) chỉ dùng để host, không thể dùng để train.

Có nhiều lựa chọn miễn phí, và kết hợp lại là đủ dùng, không cần trả phí:

| Nền tảng | GPU | Quota | Ghi chú |
|---|---|---|---|
| **Kaggle Notebooks** (khuyên dùng chính) | T4/P100 16GB | 30 giờ/tuần cố định, mỗi phiên tối đa 12 giờ | Đáng tin cậy hơn Colab vì quota rõ ràng |
| Google Colab free | T4 16GB | Dao động 15–30 giờ/tuần, phiên tối đa 12 giờ | Quota không cố định |
| Lightning AI | — | 80 giờ/tháng | UX kiểu VS Code, lưu trữ liên tục giữa các phiên — dùng dự phòng khi Kaggle/Colab hết quota |
| Amazon SageMaker Studio Lab | T4 | 4 giờ/phiên, 4 giờ/24h | ⚠️ AWS đóng đăng ký tài khoản mới từ 30/7/2026 — cần đăng ký trước ngày đó nếu muốn dùng; tài khoản cũ vẫn dùng được vô thời hạn |

Tổng cộng, 3 nguồn free (Kaggle + Colab + Lightning) cộng lại cho khoảng 45–60 giờ GPU/tuần trở lên, đủ dư cho việc fine-tune 1 lưu vực nhỏ mà không cần đụng tới phương án trả phí.

Riêng cho khóa luận, các con số ở trên chỉ tính cho việc fine-tune 1 lưu vực nhỏ. Khóa luận train Mamba+Graph trên nhiều lưu vực toàn Việt Nam (Mục 3), với quy mô data và model lớn hơn hẳn, nên free tier 16GB có thể không còn đủ VRAM hoặc đủ giờ mỗi tuần. Cần đánh giá lại nhu cầu compute khi khóa luận đã rõ phạm vi cụ thể; nếu free không đủ, có thể cân nhắc thuê GPU trả phí (Vast.ai/RunPod, khoảng $0,03–0,24/giờ).

---

## 9. 🎓🎓 Hướng công bố (chỉ tính tới sau khi có kết quả khóa luận)

SOICT là hội nghị CNTT của Việt Nam, nơi các bài chọn lọc được mời gửi mở rộng cho tạp chí Q1 *Multimedia Tools and Applications* hoặc *Informatica*. Câu chuyện "Mamba+Graph cải thiện X% so với paper GNN-Transformer 2026 đã có" là một góc phù hợp để công bố.

Về mặt thời gian, cần lưu ý rằng SOICT 2026 có hạn nộp abstract vào 9/9/2026 và full paper vào 16/9/2026 — quá gần, không kịp vì khóa luận làm sau tiểu luận. Mục tiêu thực tế hơn là nhắm tới chu kỳ SOICT năm sau (2027 trở đi), và theo dõi CFP mới khi khóa luận gần hoàn thành.

---

## 10. Lộ trình Tiểu luận chuyên ngành → Khóa luận tốt nghiệp

| Giai đoạn | Phạm vi chi tiết |
|---|---|
| 🎓 **Tiểu luận (làm ngay)** | Train Mamba thuần trên GloFAS cho 1 lưu vực (Vu Gia–Thu Bồn), so sánh NSE/KGE với LSTM/GRU/Transformer baseline (bước 2) + fine-tune bắt buộc thêm bằng GRDC thật tại trạm Nông Sơn để có 1 kết quả so với thực tế (bước 3, Mục 2.5), deploy hệ thống dự báo trên VPS Oracle |
| 🎓🎓 **Khóa luận (làm sau)** | Nâng cấp lõi thành Mamba+Graph (mô hình hóa mạng lưới sông, vẫn 1 model); mở rộng nhiều lưu vực toàn VN (transfer learning, vẫn dùng train/test split — có thể tận dụng thêm 26 trạm GRDC Việt Nam ngoài Vu Gia-Thu Bồn, Mục 2.3/Mục 5 #12, dù không bắt buộc); thêm dữ liệu bão (IBTrACS) + địa hình (DEM); so sánh trực tiếp với paper GNN-Transformer 2026; mục tiêu công bố SOICT |

---

## 11. Checklist việc cần làm

### 🎓 Phần Tiểu luận chuyên ngành (làm ngay)

- [ ] Đăng ký tài khoản ECMWF/EWDS

**Nguồn ground truth cho bước 3 (bắt buộc) — quy tắc quyết định rõ ràng, chỉ 1 nguồn chính tại 1 thời điểm:**

| Tình huống | Nguồn dùng làm chính | Trạng thái hiện tại |
|---|---|---|
| Chưa có dữ liệu mua (bây giờ) | GRDC free — trạm Nông Sơn, 1978-1990 (Mục 5 #12) | ✅ Free, tải được ngay |
| Sau khi có dữ liệu mua (Nông Sơn + Thành Mỹ ưu tiên, Câu Lâu/Ái Nghĩa xin thêm) | Chuyển hẳn sang bản mua (Mục 5 #10) làm chính | ⏳ Đã gửi email hỏi trước 28/7/2026 (hỏi Nông Sơn+Ái Nghĩa) — **đơn chính thức nộp sau cần cập nhật thêm Thành Mỹ**, xem ghi chú xác nhận trạm có Q ở Mục 5, đang chờ phản hồi email |
| Nếu xin không được | Vẫn dùng GRDC free, ghi rõ hạn chế (dữ liệu cũ, trước đập) trong báo cáo | Phương án dự phòng |

- [x] **Đã gửi email hỏi trước** (28/7/2026) tới `kttv@mae.gov.vn` (Tổng cục Khí tượng thủy văn) — hỏi xác nhận đúng thủ tục `dichvucong.monre.gov.vn/pages/ChiTietDichVuTrucTuyen.aspx?dv=352` (dv=352), mức phí hiện tại cho 2 trạm Nông Sơn + Ái Nghĩa, và khoảng thời gian dữ liệu có thể cung cấp — đang chờ phản hồi
- [ ] **Sau khi có phản hồi email:** nộp hồ sơ chính thức xin dữ liệu lưu lượng — **ưu tiên Nông Sơn + Thành Mỹ** (2 trạm chắc chắn có Q dài hạn, đối chiếu Quyết định 289/QĐ-TTg lẫn paper Nguyen et al. 2024, Mục 5 #10), xin thêm **Câu Lâu** (nhiều khả năng cũng có Q, chưa rõ từ năm nào — hỏi KTTV xác nhận) và Ái Nghĩa (không có Q, chỉ giá trị phụ) — theo đúng hướng dẫn nhận được từ email
- [ ] *(Kênh song song, ưu tiên cao — gửi ngay, không cần chờ KTTV)* Liên hệ Binh Quang Nguyen (`nqbinh@dut.udn.vn`, ĐH Bách khoa Đà Nẵng) — tác giả paper đánh giá bồi lắng Vu Gia-Thu Bồn (PMC11168294, Mục 5) — có dữ liệu **lưu lượng thật** (m³/s, đúng định dạng cần) tại **2 trạm Thanh Mỹ (Vu Gia) + Nông Sơn (Thu Bồn)**, 1996-2020, sau đập Sông Tranh 2 — phủ cả 2 nhánh. Không có link tải công khai (chỉ "available upon reasonable request"), nhưng đáng thử vì dài hơn GRDC và phủ đủ 2 nhánh
- [ ] **Ngay bây giờ, trong lúc chờ:** tải GRDC free trạm Nông Sơn qua `portal.grdc.bafg.de` (Mục 5 #12) — dùng theo đúng bảng quyết định ở trên
- [ ] Tạo VPS Oracle Cloud (Singapore → dự phòng Phoenix/Chicago) — cấu hình free hiện tại là 2 OCPU/12GB, chỉ dùng để host, không train
- [ ] Đăng ký Kaggle + Google Colab (GPU free) để train/fine-tune — cộng lại ~45-60 giờ/tuần, đủ dùng, không cần trả phí. *(Tùy chọn: Lightning AI, và SageMaker Studio Lab trước 30/7/2026 nếu còn kịp)*
- [ ] Tải shapefile Vu Gia–Thu Bồn từ Hydroviet/shapefiles hoặc HydroBASINS (chỉ để xác định ranh giới, chưa cần dựng đồ thị)
- [ ] Clone code RiverMamba (GitHub), đọc kỹ để hiểu kiến trúc + tải sẵn checkpoint pretrained
- [ ] **Tải đủ cả 5 nhóm input** (không chỉ GloFAS): GloFAS reanalysis (4 biến) + ERA5-Land (32 biến) + CPC Precipitation (1 biến) + LISFLOOD Static Features (96 biến JRC + 3 tọa độ Cartesian tự tính = 99 biến tĩnh) — tổng dùng để pretrain/fine-tune = 136 biến (Mục 4). HRES Forecast (7 biến) chỉ cần cho dự báo thật/validation sau 2010, có thể tạm bỏ qua cho tiểu luận. **Không dùng script `scripts/download_*.sh` có sẵn trong repo RiverMamba** — chỉ tải file `.7z` toàn cầu cố định (bonndata.uni-bonn.de), không hỗ trợ giới hạn vùng. Phải tải trực tiếp từ nguồn gốc mỗi loại: CDS API cho GloFAS/ERA5-Land (hỗ trợ tham số `area`), OPeNDAP cho CPC (đọc từ xa bằng `xarray` rồi cắt tọa độ), JRC Data Catalogue cho LISFLOOD Static Features (không cần đăng ký — "Anybody can directly and anonymously access the data"), MARS cho HRES (cần tài khoản riêng, chưa xác nhận điều kiện cho sinh viên VN, có thể bỏ qua nhóm này cho tiểu luận). Chi tiết từng bước xem file riêng từng nhóm dữ liệu (VD `01_glofas.md`)
- [ ] Trích xuất dữ liệu Vu Gia-Thu Bồn, fine-tune checkpoint RiverMamba từ bản GRDC toàn cầu, target = GloFAS (Phương án B — bước 2, Mục 2.4)
- [ ] **Bắt buộc:** fine-tune thêm 1 lớp nhẹ nữa, target = GRDC thật tại trạm Nông Sơn (bước 3, Mục 2.5) — cách duy nhất có 1 kết quả "so với thực tế". Chi phí thấp (dùng lại pipeline bước 2), lợi ích lớn (tránh bị hỏi "so với thực tế thì sao")
- [ ] *(Bonus nếu còn thời gian)* Train from scratch (Phương án A), so sánh với bản fine-tune
- [ ] *(Bonus nếu còn thời gian)* Ablation: fine-tune Vu Gia-Thu Bồn từ cả 2 checkpoint gốc (pretrain-only vs đã fine-tune GRDC), so sánh xem checkpoint nào cho kết quả tốt hơn
- [ ] Xây baseline LSTM/GRU/Transformer để so sánh
- [ ] Tải thử `cems-glofas-reforecast` (Mục 5 #13), tự kiểm tra cấu trúc dữ liệu thật (46 ngày, 2 lần/tuần, 11 ensemble member) trước khi code baseline "GloFAS Reforecast" cho bước 3 — cách hòa giải với đánh giá hàng ngày 1-7 ngày chưa xác nhận được từ paper, cần tự quyết định cách làm hợp lý
- [ ] **Dùng đúng loss function + trọng số + độ trễ dữ liệu như paper gốc** (xác nhận từ code thật, Mục 4.4): biến đổi Δ discharge bằng log1p có dấu trước khi tính MSE/L1; nhân trọng số theo return period (ưu tiên điểm lũ hiếm/nghiêm trọng) và theo lead time (ưu tiên ngày dự báo gần); dịch lùi input GloFAS/ERA5-Land 1 ngày, CPC 2 ngày khi ghép dữ liệu ngày "hiện tại" để không rò rỉ tương lai lúc suy luận thật
- [ ] Cài đặt NSE/KGE/RSR/PBIAS làm chỉ số đánh giá cho bước 2 (target GloFAS) — bước này tự nó không cần GRDC. GRDC dùng riêng ở bước 3 (bắt buộc, đã ghi ở mục trên)
- [ ] Lấy ngưỡng return period (đã tính sẵn từ Copernicus cho GloFAS, hoặc tự fit Gumbel/L-moments nếu cần) → tính F1-score đánh giá khả năng bắt lũ đúng lúc
- [ ] Xây dashboard hiển thị kết quả dự báo (cập nhật 1 lần/ngày) trên VPS
- [ ] *(Bonus, không bắt buộc)* Đăng ký tài khoản Copernicus, thử tải vài ảnh Sentinel-1/GFM cho đợt lũ 2025 để đối chiếu trực quan với cảnh báo của model

### 🎓🎓 Phần Khóa luận tốt nghiệp (làm sau, chưa cần bây giờ)

- [ ] Đọc paper GNN-Transformer 2026 (Vu Gia-Thu Bồn) — baseline nâng cao
- [ ] Đọc paper G-Mamba và "Accelerating flood warnings by 10 hours" để chuẩn bị nâng cấp lõi model
- [ ] Dựng đồ thị mạng lưới sông từ `ldd` (local drainage direction) của GloFAS — node = ô lưới, edge = hướng dòng chảy; HydroBASINS (#3) dùng phụ trợ đối chiếu — tích hợp vào lõi Mamba (không tạo model/lớp riêng)
- [ ] Nếu mở rộng toàn VN: trích xuất lại GloFAS cho toàn bộ VN (đổi bounding box, không cần tải dataset mới), vẫn đánh giá bằng train/test split
- [ ] Thêm dữ liệu IBTrACS (bão) + DEM (địa hình) — trước khi tải DEM riêng, đối chiếu danh sách 96 biến LISFLOOD Static Features thật (JRC) xem đã có elevation/slope chưa (Mục 3.1, Mục 5 #6) — tránh tải trùng dữ liệu đã có sẵn
- [ ] Chuẩn bị bài công bố SOICT nếu kết quả tốt

---

## Tài liệu tham khảo chính

- RiverMamba (NeurIPS 2025) — https://arxiv.org/abs/2505.22535
- G-Mamba (Graph-enhanced Mamba) — 🎓🎓 khóa luận — https://www.sciencedirect.com/science/article/abs/pii/S0925231226006776
- Accelerating flood warnings by 10 hours — 🎓🎓 khóa luận — https://www.nature.com/articles/s44304-025-00083-6
- Paper GNN-Transformer 2026 (Vu Gia-Thu Bồn) — 🎓🎓 khóa luận — https://onlinelibrary.wiley.com/doi/10.1002/rvr2.70046
- GloFAS-ERA5 (ESSD, 2020) — 🎓 tiểu luận — https://essd.copernicus.org/articles/12/2043/2020/
- EWDS (Copernicus) — 🎓 tiểu luận — https://ewds.climate.copernicus.eu/
- Ougahi et al., Global-to-local transfer learning cho streamflow (Water Resources Research, 2026) — 🎓 tiểu luận (căn cứ cho Phương án B) — https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025WR041194
- Copernicus Global Flood Monitoring (GFM) — 🎓 tiểu luận (bonus, đối chiếu trực quan) — https://global-flood.emergency.copernicus.eu/
- LISFLOOD static maps for GloFAS (`ldd`) — 🎓🎓 khóa luận (nguồn dựng Graph) — https://data.jrc.ec.europa.eu/dataset/68050d73-9c06-499c-a441-dc5053cb0c86
- Đánh giá bồi lắng/biến đổi hình thái Vu Gia-Thu Bồn — 🎓 tiểu luận (lead liên hệ xin lưu lượng thật Thanh Mỹ + Nông Sơn 1996-2020, Mục 5) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11168294/
- Quyết định 289/QĐ-TTg (8/4/2024) — Phê duyệt Quy hoạch mạng lưới trạm khí tượng thủy văn quốc gia 2021-2030 — 🎓 tiểu luận (nguồn chính thức xác nhận trạm nào có lưu lượng, Mục 5 #10) — https://vanban.chinhphu.vn/?pageid=27160&docid=210066

---

*Tài liệu kế hoạch nghiên cứu — các thông tin kỹ thuật, dữ liệu, và trích dẫn trong tài liệu đã được đối chiếu với nguồn gốc (paper, tài liệu chính thức của nhà cung cấp dữ liệu).*
