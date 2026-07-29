# GloFAS — Hiểu biết nền tảng (trước khi tải dữ liệu)

> File này chỉ ghi phần **hiểu khái niệm**, dừng lại ngay trước bước tải dữ liệu thật. Đi chậm, hiểu chắc từng phần rồi mới sang phần tải/code — không viết code hay chi tiết form download ở đây.

**Nguồn chính:** trang Overview chính thức `ewds.climate.copernicus.eu/datasets/cems-glofas-historical`, truy cập 27/7/2026.

---

## 1. GloFAS là gì

**GloFAS** (Global Flood Awareness System) là 1 hệ thống **mô phỏng lưu lượng sông toàn cầu bằng máy tính** — không phải đo nước thật bằng cảm biến. Do Joint Research Centre (JRC, Ủy ban châu Âu) quản lý, thuộc Copernicus Emergency Management Service (CEMS).

**Vì sao mô phỏng chứ không đo thật:** đặt cảm biến vật lý ở từng khúc sông trên khắp thế giới là không khả thi (quá tốn kém, nhiều nơi không tiếp cận được). Mô phỏng thì tính được ở **mọi điểm trên lưới toàn cầu**, kể cả nơi chưa từng có trạm đo — đánh đổi là độ chính xác thấp hơn đo thật.

## 2. Cách hoạt động — pipeline 2 bước

```
Bước 1: Dữ liệu thời tiết          Bước 2: Mô hình thủy văn
(mưa, nhiệt độ, gió...)    ──────▶  LISFLOOD                ──────▶  Lưu lượng sông
= ERA5 (của ECMWF)                 (tính nước từ trời rơi        (river discharge)
                                     xuống → chảy vào đất →
                                     chảy ra sông → chảy
                                     xuôi dòng qua từng
                                     đoạn sông)
```

- **ERA5** = dữ liệu tái phân tích khí tượng của ECMWF — "trời hôm đó mưa/nắng thế nào", kết hợp mô hình khí quyển với số liệu đo thật (vệ tinh, trạm mặt đất, tàu biển, bóng thám không...) để dựng lại bức tranh thời tiết quá khứ chính xác nhất có thể.
- **LISFLOOD** = mô hình thủy văn mã nguồn mở — "bộ máy tính nước từ mưa chảy ra sông ra sao" (thấm đất, diễn toán dòng chảy qua từng đoạn kênh...). Đây là mô hình **vật lý** (phương trình thủy lực thật), không phải AI học từ dữ liệu — chỉ vài tham số vật lý (độ nhám, độ dốc...), không phải hàng triệu trọng số.
- Cho LISFLOOD "ăn" dữ liệu ERA5 mỗi ngày → nhả ra kết quả: lưu lượng sông tại mọi điểm trên lưới.

**Ý nghĩa cho đề tài:** cả `cems-glofas-historical` đang dùng **thực chất là output từ mô phỏng vật lý này**, không phải số đo cảm biến 100% thật. Đề tài về bản chất đang kiểm chứng: AI (Mamba) có dự báo chính xác hơn model vật lý (GloFAS/LISFLOOD) này không.

## 3. GloFAS chia thành mấy loại

### 3.1 Theo version (phiên bản mô hình)

Giống app có v1, v2, v3 — mỗi version LISFLOOD được hiệu chỉnh lại, độ chi tiết bản đồ (độ phân giải lưới) khác nhau:

| Version | Độ phân giải lưới |
|---|---|
| Legacy (2.1, 3.1 trở về trước) | 0.1° × 0.1° |
| **Operational (4.0, mới nhất)** | **0.05° × 0.05°** — mịn gấp đôi |

**→ Đề tài dùng Version 4.0**, vì đây là bản có độ phân giải khớp với dữ liệu paper RiverMamba dùng (lưới 0.05°, 6.221.926 điểm đất liền toàn cầu — đã lọc bỏ biển; chi tiết cách lọc xem `flood-forecasting-research.md` Mục 2.3).

### 3.2 Theo loại sản phẩm (nhìn về quá khứ hay tương lai)

Đây **không phải 4 model khác nhau** — chỉ 1 bộ máy LISFLOOD duy nhất, khác nhau ở **cho nó ăn dữ liệu đầu vào nào** và **mục đích dùng để làm gì**:

```
Quá khứ xa ◀──────────────── HÔM NAY ────────────────▶ Tương lai xa
                                 │
   [Historical/Reanalysis]       │      [Forecast]        [Seasonal]
   [Reforecast]                  │
```

| Loại | Input | Mục đích | Dùng cho đề tài? |
|---|---|---|---|
| **Historical / Reanalysis** | ERA5 (đã biết chắc, vì là quá khứ) | Chép lại quá khứ chuẩn nhất có thể (1979→gần nay) — dùng để nghiên cứu/train | ✅ Đang dùng — để train Mamba |
| **Forecast** | Dự báo thời tiết thật của ECMWF (HRES), không dùng ERA5 được vì tương lai chưa xảy ra | Dự báo lũ thật, chạy mỗi ngày, nhìn 30 ngày tới — đây là thứ các cơ quan phòng chống thiên tai thật sự dùng | Sẽ cần — để demo/deploy hệ thống dự báo thật (không dùng để train) |
| **Reforecast** | Dự báo tổ hợp lịch sử — chạy bằng ECMWF-ENS (11 thành viên), phát hành 2 lần/tuần (thứ 2 và thứ 5), nhìn xa 46 ngày, phủ 2003-2022. Đứng ở 1 ngày quá khứ, dùng dự báo tổ hợp thời đó (không phải thời tiết thật đã biết), chạy y hệt quy trình Forecast | Đo độ tin cậy của hệ thống Forecast (so dự báo giả định với thực tế đã biết) | ✅ Cần dùng — đây chính là baseline "GloFAS Reforecast" trong paper RiverMamba (Bảng 3), tiểu luận cần tải để tái tạo baseline này khi đánh giá so với GRDC thật (xem `flood-forecasting-research.md` Mục 2.6, Mục 5 #13) |
| **Seasonal** | Dự báo khí hậu dài hạn (tháng, không phải ngày) | Xu hướng vài tháng tới (VD "mùa mưa năm nay nhiều/ít hơn bình thường") — độ chắc chắn thấp vì thời tiết chỉ dự báo chính xác được tối đa ~2 tuần | ❌ Không dùng |

### 3.3 Trong Historical: Consolidated vs Intermediate

| | Intermediate | Consolidated |
|---|---|---|
| Input khí tượng | ERA5T (bản tạm, near-real-time) | ERA5 (bản chính thức đã kiểm tra) |
| Độ trễ tới hiện tại | ~2–5 ngày | ~2–3 tháng |
| Độ chính xác | Thấp hơn | Cao hơn |

Cả 2 vẫn là dữ liệu **theo ngày** — khác nhau ở tốc độ/độ tin cậy publish, không phải độ chi tiết thời gian. **→ Đề tài ưu tiên Consolidated** (chính xác hơn); nếu tháng cần chưa có Consolidated do độ trễ, tạm dùng Intermediate cho khoảng đó.

**Vì sao có ERA5 và ERA5T:** ERA5 chạy theo chu kỳ 12 giờ/lần, lấy dự báo ngắn hạn trộn với số liệu đo mới về. **ERA5T** (T=tạm thời) ra sau 2-5 ngày, dùng số liệu đã có trong tay lúc đó, bỏ qua nguồn gửi trễ (tàu biển, trạm vùng xa...). **ERA5** (chính thức) ra sau 2-3 tháng, đợi đủ gần hết nguồn toàn cầu + kiểm tra loại bỏ số liệu bất thường, rồi chạy lại 1 lần nữa cho chuẩn. (Nguồn: ECMWF Confluence Wiki, Climate Data Guide UCAR.)

## 4. Bên trong dataset có gì — hình dung như 1 chồng ảnh

Coi dataset (Historical, v4.0, Consolidated — bản đề tài dùng) như **1 chồng ảnh, mỗi ngày 1 tấm**:

```
Ngày 1  ──▶  [ảnh N hàng × M cột ô vuông, mỗi ô 1 bộ số]
Ngày 2  ──▶  [ảnh N hàng × M cột ô vuông, mỗi ô 1 bộ số]
...
```

- **Mỗi "tấm ảnh"** = 1 ngày, phủ đúng vùng địa lý đã chọn tải.
- **Mỗi "ô vuông"** = 1 điểm trên bản đồ thật (tọa độ lat/lon riêng), rộng khoảng 5,5km × 5,5km ngoài đời (ứng với 0.05°).
- Khác ảnh thường (mỗi ô có 3 số R/G/B): dataset này **mỗi ô có 4 số**, không phải màu sắc mà là 4 đại lượng vật lý:

| Biến | Tên trong paper RiverMamba | Ý nghĩa dễ hiểu | Đơn vị |
|---|---|---|---|
| River discharge in the last 24 hours | `dis24` | Lưu lượng sông — nước chảy qua nhanh cỡ nào | m³/s |
| Runoff water equivalent | `acc_rod24` | Nước sinh ra từ mưa (mặt + ngầm), trước khi gộp vào sông | kg/m² |
| Snow depth water equivalent | `sd` | Tuyết (VN gần như luôn ~0) | kg/m² |
| Soil wetness index (root zone) | `swi` | Đất đang "ướt" cỡ nào, tầng rễ cây | 0–1 (không thứ nguyên) |

⚠️ **Lưu ý quan trọng đã xác nhận bằng paper RiverMamba (Bảng 4/Phụ lục A.1), không phải chỉ dùng `dis24`:** cả 4 biến trên đều là input của model, không chỉ lưu lượng — kể cả `sd` (tuyết) dù VN không có tuyết vẫn phải tải, vì checkpoint pretrained cố định theo đúng 4 cột này, thiếu cột sẽ lệch input schema (giá trị thực tế sẽ gần 0 cho vùng VN, không phải lỗi).

**Vì đây là lưới toàn vùng, không chỉ trên sông:** đa số ô (đất khô, đồi núi) sẽ có `dis24` gần 0 hoặc rất nhỏ, chỉ số ít ô nằm đúng trên dòng sông mới có số lớn — sông luôn nổi bật hẳn lên giữa 1 vùng đa số số nhỏ/gần 0.

## 5. Thông số kỹ thuật tóm tắt

| Thông số | Giá trị |
|---|---|
| Kiểu dữ liệu | Gridded (lưới), không phải điểm trạm |
| Hệ tọa độ | Regular latitude-longitude |
| Phạm vi không gian | Toàn cầu trừ Nam Cực (90N–60S, 180W–180E) |
| Độ phân giải không gian | 0.05° × 0.05° (Version 4.0) |
| Phạm vi thời gian | 1/1/1979 đến gần thời gian thực |
| Độ phân giải thời gian | Theo ngày (daily) |
| Định dạng file | GRIB2 và NetCDF-4 (khuyến nghị NetCDF-4, dễ đọc bằng `xarray`) |
| Phiên bản vận hành | GloFAS v4.0, phát hành 26/7/2023 |

⚠️ **Lưu ý vận hành** (ghi trên đầu trang Overview, cập nhật 1/2/2024): *"Please note that accessing this dataset via CDS for time-critical operation is not advised or supported"* — không khuyến nghị dùng cho tác vụ cần độ trễ thấp/khẩn cấp thật. Không ảnh hưởng đề tài (chỉ dùng để train trên dữ liệu lịch sử).

## 6. Giấy phép & trích dẫn

- **Giấy phép:** CEMS-FLOODS datasets licence
- **DOI:** [10.24381/cds.a4fdd6b9](https://doi.org/10.24381/cds.a4fdd6b9)
- **Ngày xuất bản:** 5/11/2019 | **Cập nhật lần cuối:** 26/7/2026

---

*Dừng ở đây — phần tiếp theo (form download, chọn vùng địa lý, code tải, kết quả test) sẽ viết ở file riêng khi thật sự bắt đầu tải, không viết trước khi chưa hiểu chắc phần trên.*
