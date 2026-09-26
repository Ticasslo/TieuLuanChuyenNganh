# Đề xuất bài cơ sở

> Bài 1 đã gửi GVHD ngày 26/9/2026; bài 2 chỉ gửi nếu GVHD không đồng ý bài 1. Hai bài báo được đề xuất theo thứ tự ưu tiên: bài 1 là lựa chọn chính; bài 2 là phương án thay thế nếu bài 1 không đáp ứng yêu cầu của GVHD (ví dụ số trích dẫn tối thiểu). Giới thiệu chi tiết: `TongQuan/GioiThieuBaiCoSo_BiasCast.md`, `TongQuan/GioiThieuBaiCoSo.md`. Pipeline: `TongQuan/KienTrucPipeline_BiasCast.md`, `TongQuan/KienTrucPipeline.md`. So sánh chi tiết hai bài: `BaiCoSo/SOSANH.md`.

## Bài 1 (đề xuất chính)

**BiasCast: learning and adjusting real time biases from meteorological forecasts to enhance runoff predictions**
Oliver Konold, Moritz Feigl, Patrick Podest, Christoph Klingler, Karsten Schulz. Hydrology and Earth System Sciences (HESS), 30, 5067–5096, 2026. DOI: 10.5194/hess-30-5067-2026. Tạp chí Scimago Q1 (SJR 2025: 2,035).

- Trang bài chính thức (HESS): https://hess.copernicus.org/articles/30/5067/2026/
- File PDF: https://hess.copernicus.org/articles/30/5067/2026/hess-30-5067-2026.pdf
- Trang phản biện công khai (2 người phản biện, 2 vòng, kèm trả lời của tác giả): https://egusphere.copernicus.org/preprints/2025/egusphere-2025-4978/
- Mã huấn luyện (thư viện NeuralHydrology do tác giả chỉnh sửa, chứa mô hình, bộ đọc dữ liệu, vòng huấn luyện và đánh giá): https://github.com/conestone/neuralhydrology
- Mã chạy thí nghiệm và phân tích kết quả của tác giả: https://github.com/conestone/biascast (bản lưu trữ: https://doi.org/10.5281/zenodo.17293199)
- Cấu hình, trọng số mô hình và kết quả của mọi thí nghiệm: https://doi.org/10.5281/zenodo.17241922

Bộ dữ liệu sử dụng: Extended LamaH-CE (859 lưu vực Trung Âu, dữ liệu theo ngày; LamaH-CE bổ sung các nguồn khí tượng E-OBS, MSWEP, GLEAM, dự báo thời tiết ECMWF HRES và lưu lượng cực tiểu, trung bình, cực đại ngày; giấy phép CC BY-NC 4.0). Bài dùng 451 lưu vực ít chịu tác động của con người, kỳ 2003–2017.

- Dữ liệu (Zenodo): https://doi.org/10.5281/zenodo.17119635
- Bài báo mô tả bộ dữ liệu gốc LamaH-CE (ESSD 2021): https://doi.org/10.5194/essd-13-4529-2021

## Bài 2 (phương án thay thế)

**The Merit of River Network Topology for Neural Flood Forecasting**
Nikolas Kirschstein, Yixuan Sun. Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, tr. 24713–24725. Hội nghị hạng CORE A*.

- Trang bài chính thức (kỷ yếu ICML 2024, PMLR): https://proceedings.mlr.press/v235/kirschstein24a.html
- File PDF: https://raw.githubusercontent.com/mlresearch/v235/main/assets/kirschstein24a/kirschstein24a.pdf
- Trang phản biện OpenReview (trạng thái "ICML 2024 Poster"): https://openreview.net/forum?id=QE6iC9s6vU
- Mã nguồn của tác giả (kèm checkpoint mô hình): https://github.com/nkirschi/neural-flood-forecasting

Bộ dữ liệu sử dụng: LamaH-CE (859 lưu vực Trung Âu, dữ liệu theo giờ 1981–2017, giấy phép CC BY-SA 4.0)

- Bài báo mô tả dữ liệu (ESSD 2021): https://doi.org/10.5194/essd-13-4529-2021
- Dữ liệu (Zenodo): https://doi.org/10.5281/zenodo.4525244

## Lý do xếp BiasCast trước

| Tiêu chí | BiasCast (bài 1) | Kirschstein & Sun (bài 2) |
|---|---|---|
| Bài toán | Dự báo lưu lượng cực đại ngày với dự báo thời tiết ECMWF thật và lưu lượng quan trắc — sát vận hành | Dự báo lưu lượng sau 6 giờ từ 24 giờ quá khứ, không dùng dự báo thời tiết |
| Độ tin cậy số liệu | Kết quả theo lưu vực của tác giả được tính lại, khớp bài | Công thức NSE trong mã tính sai; nhiễu giữa các lần huấn luyện lớn cỡ toàn bộ chênh lệch trong bảng kết quả |
| Chia dữ liệu | Train 2003–2009, validation 2010–2013, test 2014–2017, tách theo thời gian | Validation rút ngẫu nhiên từ các cửa sổ chồng lấn với train; test 2 năm |
| Mô hình mốc | Họ LSTM dự báo hindcast–forecast, cùng dòng với mô hình vận hành của Google (Nature 2024) | Lớp affine + mạng nơ-ron đồ thị, không có thành phần học chuỗi thời gian |
| Phản biện | Công khai, hai người phản biện đánh giá tích cực | Không công khai |
| Khoảng trống để cải tiến | Tác giả tự nêu chỉ thử kiến trúc LSTM, chỉ dữ liệu ngày, chỉ lead 1 ngày; người phản biện đề nghị so với persistence nhưng bài chưa làm | Bộ mã hóa thời gian chỉ là lớp affine; cửa sổ 24 giờ |
| Nơi công bố, trích dẫn | Tạp chí Q1 ngành thủy văn; 3 trích dẫn (xuất bản 2026) | Hội nghị A* ngành học máy; 21 trích dẫn |
| Dữ liệu | 0,95 GB, theo ngày | 14,8 GB nén, theo giờ |

BiasCast được xếp trước vì vững hơn ở các khía cạnh phương pháp: độ tin cậy số liệu, cách chia dữ liệu, mô hình mốc và tính thực tế của bài toán. Kirschstein & Sun hơn ở uy tín nơi công bố, số trích dẫn và dữ liệu theo giờ (hợp với thế mạnh chuỗi dài của Mamba); được giữ làm phương án thay thế khi yêu cầu về số trích dẫn hoặc nơi công bố ngành CNTT/AI được đặt lên trước.
