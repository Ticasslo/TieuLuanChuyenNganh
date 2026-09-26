# Kiến trúc và pipeline của đề tài — phương án bài cơ sở Kirschstein & Sun

> Tài liệu mô tả các giai đoạn thực hiện đề tài khi dùng Kirschstein & Sun (ICML 2024) làm bài cơ sở, trên bộ dữ liệu LamaH-CE theo giờ. Phương án song song dùng BiasCast (HESS 2026): `TongQuan/KienTrucPipeline_BiasCast.md`; so sánh hai bài: `BaiCoSo/SOSANH.md`; BiasCast đã được gửi GVHD làm đề xuất chính (26/9/2026); tài liệu này là phương án thay thế nếu GVHD không đồng ý. Nội dung gồm: mỗi giai đoạn nhận gì, làm gì, cho ra gì, chạy ở đâu; kiến trúc mô hình dạng mô-đun; chỉ số đánh giá; các việc chưa chốt. Giới thiệu bài cơ sở: `TongQuan/GioiThieuBaiCoSo.md`. Chi tiết tải dữ liệu: `LamaHCE/LamaHCE.md`. Phạm vi đề xuất: `BaiCoSo/CHECKPDF.md` Mục 8.2. Các mục ghi "chưa chốt" chờ quyết định, chưa được điền.

## 1. Tổng quan các giai đoạn

| Giai đoạn | Việc chính | Đầu ra | Nền tảng | Trạng thái |
|---|---|---|---|---|
| 1. Thu thập dữ liệu | Tải LamaH-CE từ Zenodo theo luồng, chỉ giữ phần cần | Kaggle Dataset `lamah-ce-core`, `lamah-ce-extra` | Kaggle, CPU | Đã chạy (ZIP 6,82 GB + 6,32 GB); chờ xác nhận log kiểm tra ZIP |
| 2. Khảo sát dữ liệu | Hiểu cấu trúc, chất lượng, phân phối, không gian của dữ liệu | Báo cáo khảo sát kèm biểu đồ, bản đồ | Kaggle hoặc Colab, CPU | Chưa làm |
| 3. Tái lập và tính lại NSE | Nạp checkpoint của tác giả, tính lại NSE đúng | Bảng mốc so sánh chính xác | Kaggle hoặc Colab, GPU | Chưa làm |
| 4. Tiền xử lý và bộ nạp dữ liệu | Lọc trạm, dựng đồ thị, chuẩn hóa, cắt cửa sổ, chia tập theo thời gian | Bộ nạp dữ liệu dùng chung cho mọi mô hình | Kaggle hoặc Colab | Chưa làm |
| 5. Mô hình | Khung mô-đun: bộ mã hóa thời gian, mô-đun không gian, bộ giải mã | Các cấu hình mô hình cần so sánh | — | Chưa làm |
| 6. Huấn luyện | Huấn luyện mọi cấu hình, nhiều hạt giống | Checkpoint, nhật ký huấn luyện | Kaggle hoặc Colab, GPU T4 | Chưa làm |
| 7. Đánh giá | Tính chỉ số trên tập kiểm tra, so sánh có kiểm định | Bảng và biểu đồ kết quả | Kaggle hoặc Colab | Chưa làm |
| 8. Giải thích mô hình (XAI) | Integrated Gradients theo thời gian và theo trạm | Bản đồ, biểu đồ mức ảnh hưởng | Kaggle hoặc Colab, GPU | Chưa làm |
| 9. Ngưỡng nguy cơ lũ | Fit phân phối cực trị lưu lượng từng trạm | Ngưỡng return period của mỗi trạm | CPU | Chưa làm |
| 10. Demo | Bản đồ dự báo theo giờ, chuỗi Q, giải thích XAI | Ứng dụng demo | Chưa chốt | Chưa làm |

Luồng dữ liệu giữa các giai đoạn: tệp nén LamaH-CE trên Zenodo → (1) hai Kaggle Dataset → (2) báo cáo khảo sát và danh sách trạm, đồ thị đã kiểm tra → (3) mốc NSE chính xác của bài gốc → (4) cửa sổ dữ liệu và ma trận kề → (5, 6) checkpoint các mô hình → (7) bảng kết quả → (8, 9) mức ảnh hưởng và ngưỡng nguy cơ lũ → (10) demo.

## 2. Chi tiết từng giai đoạn

### 2.1. Thu thập dữ liệu

- **Đầu vào:** tệp `1_LamaH-CE_daily_hourly.tar.gz` (14,8 GB nén, khoảng 70 GB giải nén) trên Zenodo `10.5281/zenodo.5153305`.
- **Cách làm:** đọc tệp nén theo luồng, không lưu tệp gốc, chỉ giữ các tệp cần và ghi vào một tệp ZIP. Lý do và kiểm thử: `LamaHCE/LamaHCE.md` Mục 3–4.
- **Đầu ra:** gói lõi (lưu vực trung gian B, trạm đo D theo giờ, thư mục Info — đủ tái lập bài gốc) và gói bổ sung (lưu vực toàn thượng nguồn A theo giờ, thuộc tính tĩnh, shapefile, mạng sông, kết quả mô hình thủy văn COSERO).
- **Tệp mã:** `LamaHCE/LamaHCE_Download_Core.py`, `LamaHCE/LamaHCE_Download_Extra.py`.

### 2.2. Khảo sát dữ liệu

Mục đích: hiểu chắc dữ liệu trước khi xây mô hình. Các câu hỏi phải trả lời:

- **Cấu trúc tệp:** cột thực tế trong tệp so với tên biến trong bài cơ sở và bài công bố dữ liệu (ESSD 2021); đơn vị; cờ chất lượng; mã giá trị thiếu (-999).
- **Chất lượng:** tỉ lệ thiếu theo trạm và theo năm; đoạn thiếu dài; kiểm tra lại quy tắc lọc 358 trạm của tác giả.
- **Phân phối:** histogram từng biến (lưu lượng, mưa, độ ẩm đất, nhiệt độ, áp suất); phân bố lưu lượng trung bình và cực đại giữa các trạm; tính mùa vụ.
- **Không gian:** bản đồ 358 trạm và ranh giới lưu vực; mạng lưới sông và đồ thị trạm (độ sâu, bậc vào, đường đi dài nhất 19 cạnh); trọng số cạnh vật lý (chiều dài, chênh cao, độ dốc).
- **Đỉnh lũ:** chuỗi lưu lượng lớn nhất năm từng trạm (đầu vào cho giai đoạn 9); các trạm có đỉnh đột ngột, hẹp.
- **Nguồn tham chiếu:** thuộc tính tĩnh có sẵn, kết quả COSERO, tài liệu gốc trong thư mục Info.
- **Đầu ra:** báo cáo khảo sát trong `LamaHCE/LamaHCE.md`, biểu đồ và bản đồ.
- **Tệp mã dự kiến:** `LamaHCE/LamaHCE_Explore.py`.

### 2.3. Tái lập và tính lại NSE

Mục đích: có mốc so sánh chính xác trước khi cải tiến, vì NSE trong mã của tác giả sai công thức (`TongQuan/GioiThieuBaiCoSo.md` Mục 9).

- **Đầu vào:** gói dữ liệu lõi; mã nguồn và checkpoint của tác giả (lấy bằng `git clone`).
- **Các bước:**
  1. Nạp checkpoint với kiểm tra khớp đủ tham số (mã gốc dùng `strict=False`, không báo lỗi khi lệch).
  2. Dự báo trên tập kiểm tra 2016–2017.
  3. Tính ba phiên bản NSE: theo mã tác giả (để khớp lại số trong bài), NSE có trọng số đúng công thức của bài (điểm liên quan tính theo từng trạm), NSE chuẩn không trọng số theo từng trạm (mốc là trung bình kỳ kiểm tra).
  4. So sánh thứ tự các cấu hình; kiểm tra kết luận "đồ thị không hơn MLP" còn đúng không.
  5. Ước lượng nhiễu giữa các lần huấn luyện: so các cặp cấu hình giống nhau về chức năng (ResGAT "all" và "binary" lệch tới 6,67 điểm % trong kết quả tác giả); nếu còn quota, huấn luyện lại vài cấu hình với nhiều hạt giống.
- **Phạm vi checkpoint:** 162 checkpoint của thí nghiệm chính (Bảng 2) và 3 checkpoint MLP; 108 checkpoint ablation và 684 checkpoint mạng con là tùy chọn.
- **Đầu ra:** bảng NSE đúng của bài gốc, dùng làm mốc cho giai đoạn 7.
- **Tệp mã dự kiến:** `LamaHCE/LamaHCE_RecomputeNSE.py`.

### 2.4. Tiền xử lý và bộ nạp dữ liệu

- **Lọc trạm và đồ thị:** giữ đúng quy tắc của tác giả (358 trạm, nối lại cạnh khi xóa trạm) để so sánh được với bài gốc; ma trận kề đọc từ `Stream_dist.csv`.
- **Chuẩn hóa:** z-score theo từng trạm, thống kê chỉ tính trên kỳ huấn luyện.
- **Cắt cửa sổ:** viết lại để cửa sổ được vượt ranh giới năm (mã gốc cắt trong từng năm nên không chạy được cửa sổ từ 1 năm trở lên).
- **Chia tập:** giữ tập kiểm tra 2016–2017 như bài gốc; validation chia theo thời gian thay vì ngẫu nhiên.
- **Đầu vào của bài gốc:** mỗi trạm, mỗi giờ 5 biến (Q, mưa, độ ẩm đất lớp 1–3, nhiệt độ 2 m, áp suất bề mặt; `dataset.py`), không có thuộc tính tĩnh. LamaH-CE còn các biến khí tượng theo giờ khác và hơn 60 thuộc tính lưu vực (bài công bố dữ liệu ESSD 2021); danh sách cột chính xác xác nhận ở giai đoạn khảo sát.
- **Chưa chốt:** năm dùng cho validation; độ dài cửa sổ W; các lead time L ngoài 6 giờ; có dùng thuộc tính tĩnh hay không; có thêm biến khí tượng ngoài 4 biến của bài gốc hay không (nếu thêm, phải tách riêng thí nghiệm để không lẫn với tác động của kiến trúc).

### 2.5. Mô hình

Chi tiết ở Mục 3.

### 2.6. Huấn luyện

- **Hàm mất mát mặc định:** MSE nhân điểm liên quan như bài gốc, để so sánh được.
- **Hàm mất mát thử nghiệm (chưa chốt):** trọng số theo ngưỡng return period như RiverMamba, ưu tiên các thời điểm lũ hiếm.
- **Chọn mô hình:** epoch có loss validation nhỏ nhất trên validation chia theo thời gian.
- **Hạt giống:** nhiều hạt giống cho mỗi cấu hình (số lượng chưa chốt, phụ thuộc quota GPU), vì nhiễu giữa các lần chạy của bài gốc lớn cỡ toàn bộ chênh lệch giữa các mô hình; ghi rõ có bật chế độ tính toán tất định trên GPU hay không.
- **Lưu trữ:** checkpoint lưu vào Kaggle Dataset hoặc Google Drive.
- **Ràng buộc phần cứng:** `mamba-ssm` cần GPU CUDA từ kiến trúc sm_75 (T4 chạy được, P100 không); wheel biên dịch một lần rồi lưu lại (`LyThuyet/RiverMamba.md` Mục 0).

### 2.7. Đánh giá

Danh sách chỉ số theo kế hoạch tổng thể (`flood-forecasting-research.md` Mục 7):

- **Hồi quy:** NSE chuẩn theo từng trạm (báo cáo trung vị và phân phối qua các trạm), NSE có trọng số (để so với bài gốc), KGE, RSR, PBIAS, FHV (sai lệch đỉnh lũ), FLV (sai lệch dòng chảy kiệt).
- **Phân loại:** F1 theo ngưỡng return period (vượt ngưỡng hay không).
- **So sánh:** trung bình ± độ lệch chuẩn qua các hạt giống và cách chia; kiểm định thống kê giữa các mô hình (phương pháp chưa chốt); phân tích theo nhóm trạm (thượng lưu và hạ lưu, kích thước lưu vực).

### 2.8. Giải thích mô hình (XAI)

- Integrated Gradients bằng thư viện Captum.
- Theo thời gian: giờ nào trong cửa sổ đầu vào ảnh hưởng nhiều nhất tới dự báo.
- Theo không gian: trạm thượng nguồn nào ảnh hưởng nhiều nhất tới dự báo của một trạm (chỉ với mô hình có đồ thị).
- Công trình của Demiray & Demir (WaterBench-Iowa) đã làm Mamba + SHAP, nên phần khác biệt của đề tài nằm ở chiều không gian.

### 2.9. Ngưỡng nguy cơ lũ

- Mô hình chỉ dự báo lưu lượng Q (m³/s), không dự báo mực nước hay vùng ngập. Nguy cơ lũ được đánh giá gián tiếp: Q dự báo vượt ngưỡng lưu lượng ứng với một chu kỳ lặp lại (return period).
- Ngưỡng tính riêng từng trạm bằng cách fit phân phối Gumbel hoặc L-moments trên chuỗi lưu lượng lớn nhất năm.
- **Chưa chốt:** các chu kỳ lặp lại dùng làm ngưỡng.

### 2.10. Demo

- Bản đồ 358 trạm tô màu theo mức vượt ngưỡng; chọn thời điểm để xem dự báo; bấm vào trạm để xem chuỗi Q dự báo so với thực đo và giải thích XAI.
- LamaH-CE là dữ liệu lịch sử, không có luồng cập nhật hằng ngày, nên demo phát lại theo thời gian trên tập kiểm tra.
- **Chưa chốt:** công nghệ làm demo và nơi triển khai (kế hoạch tổng thể ghi VPS Oracle Cloud Always Free dùng được để host).

## 3. Kiến trúc mô hình

### 3.1. Khung chung

Mọi mô hình đi qua ba khối, giống cấu trúc "bánh kẹp" của bài gốc nhưng thay khối đầu:

| Khối | Đầu vào → đầu ra | Vai trò |
|---|---|---|
| Bộ mã hóa thời gian | Chuỗi W giờ × 5 biến của từng trạm → vector ẩn của trạm | Học quan hệ theo thời gian trong cửa sổ |
| Mô-đun không gian (tùy chọn) | Vector ẩn của mọi trạm + ma trận kề → vector ẩn mới | Truyền thông tin giữa các trạm theo mạng sông |
| Bộ giải mã | Vector ẩn → Q sau L giờ của từng trạm | Chiếu ra giá trị dự báo |

### 3.2. Các lựa chọn cho từng khối

| Khối | Lựa chọn | Nguồn |
|---|---|---|
| Bộ mã hóa thời gian | Affine | Bài gốc |
| | LSTM, GRU | Baseline hồi quy chuẩn trong thủy văn |
| | Transformer | Baseline attention |
| | S4D | Đối chứng SSM bất biến theo thời gian; mã từ S4D-FT (Wang và cs., WRR 2025) |
| | Mamba | Mô hình chính: SSM chọn lọc, chi phí tuyến tính theo độ dài chuỗi |
| Mô-đun không gian | Không dùng (mỗi trạm dự báo riêng) | Tương ứng MLP và trường hợp cô lập của bài gốc |
| | GCNII, ResGCN, ResGAT | Bài gốc |
| | Đồ thị dày theo khả năng tới được | FloodGNNs (giai đoạn khóa luận) |
| Bộ giải mã | Affine | Bài gốc |
| Thuộc tính tĩnh (chưa chốt) | LOAN: cộng thuộc tính lưu vực vào bước chuẩn hóa | RiverMamba |

### 3.3. Ma trận thí nghiệm

- Mỗi bộ mã hóa thời gian chạy hai kiểu: không có mô-đun không gian và có GNN (loại GNN chưa chốt; ứng viên là cấu hình tốt nhất sau khi tính lại NSE ở giai đoạn 3).
- **LSTM là baseline chính:** là chuẩn của ngành thủy văn (Kratzert và cs. 2018, 2019) và là lõi của mô hình vận hành của Google. So sánh quan trọng nhất của đề tài là Mamba với LSTM ở cùng cửa sổ, cùng dữ liệu, cùng ngân sách tham số, có và không có đồ thị. Bài gốc không có LSTM. Với cửa sổ rất dài theo giờ (hàng nghìn bước), LSTM chạy chậm; dùng thêm MF-LSTM (Acuña Espinoza và cs., HESS 2025: gộp quá khứ xa thành bước ngày, giữ phần gần theo giờ) làm baseline LSTM hiệu quả. Siêu tham số LSTM tham khảo cấu hình của NeuralHydrology và OpenHydroNet (forget bias khởi tạo 3, dropout đầu ra 0,4, cắt gradient).
- So sánh với hai mốc của bài gốc (MLP và GNN gốc), dùng số liệu tính lại, và với baseline persistence (Q dự báo = Q quan trắc gần nhất) mà bài gốc không báo cáo — với lead 6 giờ có Q quá khứ, phần lớn kỹ năng có thể đến từ quán tính dòng chảy.
- Điều kiện công bằng: cùng dữ liệu, cùng W và L, cùng cách chia tập, cùng hạt giống, cùng ngân sách huấn luyện. Cách khống chế số tham số giữa các mô hình chưa chốt.
- Hai câu hỏi được trả lời từ ma trận: so theo hàng (cùng kiểu không gian, khác bộ mã hóa) cho biết Mamba có hơn các bộ mã hóa khác không; so theo cột (cùng bộ mã hóa, có và không có đồ thị) cho biết đồ thị có giúp không.

## 4. Hạ tầng và tổ chức mã

- **Tính toán:** Google Colab bản miễn phí là chính, Kaggle dự phòng (GPU 30 giờ/tuần, tối đa 12 giờ/phiên); mã chạy được trên cả hai nền tảng.
- **Dữ liệu:** Kaggle Dataset private, gắn vào notebook bằng Add Input; Colab lấy qua Kaggle API với token trong Colab Secrets.
- **Tổ chức mã:** mỗi notebook là một tệp `.py` tự đủ, đặt tên `LamaHCE_<Việc>.py`, có mục lục đầu tệp và chia cell theo "Phần". Ghi chú kết quả đặt ở `LamaHCE/LamaHCE.md`, không để trong mã.

| Tệp | Giai đoạn | Trạng thái |
|---|---|---|
| `LamaHCE_Download_Core.py` | 1 | Đã chạy |
| `LamaHCE_Download_Extra.py` | 1 | Đã chạy |
| `LamaHCE_Explore.py` | 2 | Tên dự kiến |
| `LamaHCE_RecomputeNSE.py` | 3 | Tên dự kiến |
| Tệp cho giai đoạn 4–10 | 4–10 | Chưa đặt tên |

## 5. Việc chưa chốt

| Việc | Giai đoạn |
|---|---|
| Năm dùng cho validation | 4 |
| Độ dài cửa sổ W và các lead time L | 4 |
| Có dùng thuộc tính tĩnh (LOAN) hay không | 4, 5 |
| Có thêm biến khí tượng ngoài 4 biến của bài gốc hay không | 4 |
| Loại GNN trong ma trận thí nghiệm | 5 |
| Cách khống chế số tham số giữa các mô hình | 5 |
| Hàm mất mát có trọng số return period | 6 |
| Số hạt giống cho mỗi cấu hình | 6 |
| Phương pháp kiểm định thống kê | 7 |
| Các chu kỳ lặp lại dùng làm ngưỡng | 9 |
| Công nghệ và nơi triển khai demo | 10 |
| Đưa CAMELS-US theo giờ vào tiểu luận hay để dành cho khóa luận | Toàn đề tài |

## 6. Đối chiếu với pipeline của bài cơ sở

Pipeline của đề tài giữ nguyên khung của bài cơ sở ở những phần quyết định khả năng so sánh (cùng tập trạm, cùng đồ thị, cùng tập kiểm tra, cùng cửa sổ và lead time mốc, cùng hàm mất mát mặc định), và thay đổi ở những phần bài cơ sở còn yếu. Pipeline của bài cơ sở đọc từ mã nguồn: `dataset.py`, `functions.py`, `models.py`, các script `train_*`/`test_*` và 8 notebook.

| Giai đoạn | Bài cơ sở | Đề tài | Giống / khác |
|---|---|---|---|
| Thu thập dữ liệu | Tải nguyên tệp 14,8 GB rồi giải nén 3 thư mục (ước tính cần khoảng 50–55 GB ổ đĩa) | Đọc theo luồng, chỉ giữ tệp cần, lưu thành Kaggle Dataset dùng lại | Cùng nguồn và cùng tệp dữ liệu; khác cách tải |
| Khảo sát dữ liệu | Notebook 6 cell: vẽ đồ thị trạm, tính đường đi dài nhất | Khảo sát sâu: cột, đơn vị, % thiếu, phân phối, bản đồ, đỉnh lũ | Đề tài mở rộng |
| Lọc trạm, dựng đồ thị | Duyệt ngược dòng từ trạm 399, lọc trạm, nối lại cạnh, còn 358 trạm | Giữ nguyên quy tắc | Giống |
| Chuẩn hóa | z-score theo trạm, thống kê 2000–2015 | z-score theo trạm, thống kê chỉ trên kỳ huấn luyện | Khác nhỏ |
| Cắt cửa sổ | Trong từng năm, W = 24 giờ, L = 6 giờ | Vượt ranh giới năm, W dài hơn (chưa chốt), giữ W = 24, L = 6 làm mốc | Khác |
| Chia tập | Test 2016–2017; 3 cách chọn năm train; validation 1/5 cửa sổ ngẫu nhiên | Giữ test 2016–2017; validation chia theo thời gian | Giống phần test, khác phần validation |
| Mô hình | Affine → 19 lớp GNN → affine; MLP làm mốc | Bộ mã hóa thời gian (6 lựa chọn) → GNN hoặc không → affine | Giữ khung, thay khối mã hóa |
| Huấn luyện | Adam, 100 epoch, MSE × điểm liên quan, 1 hạt giống | Giữ cấu hình mốc; nhiều hạt giống | Khác số hạt giống |
| Đánh giá | Một chỉ số: NSE có trọng số (công thức trong mã bị lỗi) | NSE đã sửa, NSE chuẩn, KGE, FHV, F1 theo ngưỡng; kiểm định thống kê | Đề tài mở rộng và sửa lỗi |
| Tính lại kết quả bài gốc | Không có | Tính lại NSE từ checkpoint của tác giả | Chỉ có ở đề tài |
| Phân tích thêm | Tương quan trọng số cạnh học được; 4 mạng con; trạm kém nhất; lưới cửa sổ 12–72 giờ × lead 1–12 giờ | Nhiễu giữa các lần chạy; phân tích theo nhóm trạm | Khác trọng tâm |
| XAI, ngưỡng nguy cơ lũ, demo | Không có | Có | Chỉ có ở đề tài |

## 7. Ý tưởng tham khảo từ OpenHydroNet (Google FloodHub)

Rút ra từ việc đọc mã OpenHydroNet (`BaiCoSo/CHECKCODE.md` Mục 14). Tất cả là lựa chọn chưa chốt.

| Ý tưởng | Trong OpenHydroNet | Cách áp dụng vào đề tài | Giai đoạn |
|---|---|---|---|
| Dự báo nhiều lead time cùng lúc | Đầu ra 8 bước, lead 0–7 ngày | Dự báo Q cho nhiều bước giờ tới (ví dụ 1–6 giờ hoặc dài hơn) thay vì chỉ 1 giá trị tại t + 6 giờ; báo cáo chỉ số theo từng lead time | 5, 7 |
| Đầu ra xác suất | Đầu CMAL, 3 thành phần | Cho ra xác suất Q vượt ngưỡng return period, dùng trực tiếp cho demo cảnh báo | 5, 9, 10 |
| Hàm mất mát NSE* | Chia sai số cho độ lệch chuẩn lưu lượng của từng lưu vực | Đối chứng với hàm mất mát có trọng số điểm liên quan của bài gốc | 6 |
| Thuộc tính tĩnh qua mạng nhúng | Nhúng thuộc tính rồi nối vào mọi bước | Dùng thuộc tính lưu vực của LamaH-CE (gói bổ sung) | 4, 5 |
| Bỏ mẫu thiếu thay vì bỏ trạm | `validate_samples.py` | Giữ được nhiều trạm hơn 358 nếu chỉ bỏ các mẫu có đoạn thiếu | 4 |
| Trung bình có mặt nạ theo nguồn | Gộp nhúng của các nguồn khí tượng, bỏ nguồn thiếu | Chỉ cần khi có nhiều nguồn khí tượng; LamaH-CE chỉ có một nguồn | Khóa luận |
| Lùi đầu vào theo thời điểm có dữ liệu | Dự báo dùng làm hindcast lấy lead 0 của lần phát hành hôm trước | Nguyên tắc chống rò rỉ tương lai khi dùng dữ liệu vận hành | Khóa luận |
| Bộ chỉ số đánh giá | `metrics.py` (Apache 2.0): NSE, KGE, FHV, FLV, Peak-Timing, Missed-Peaks… | Dùng lại để tính chỉ số chuẩn, tránh tự viết lại công thức NSE | 3, 7 |
| Scaler lưu kèm mô hình | `scaler.nc` tính trên kỳ train | Lưu thống kê chuẩn hóa cùng checkpoint | 4, 6 |
| Kỹ thuật huấn luyện | Cắt gradient, nhiễu nhãn, dropout đầu ra, forget bias 3, ReduceLROnPlateau | Cấu hình mặc định cho LSTM và các baseline | 6 |
| Dự báo nhiều ngày trên LamaH-CE | 859 lưu vực LamaH-CE có trong Caravan, dự báo HRES trùng 2012–2017 | Mở rộng sang dự báo 1–7 ngày theo ngày, cùng vùng và cùng kỳ test 2016–2017 | Khóa luận |

Lưu ý: trọng số huấn luyện sẵn của Google đã học cả 859 lưu vực LamaH-CE trong 1982–2023, nên không dùng được để đánh giá trên kỳ 2016–2017.

## 8. Hướng mở rộng từ BiasCast (Konold và cs., HESS 2026)

BiasCast dự báo lưu lượng cực đại ngày trước 24 giờ trên 451 lưu vực LamaH-CE, dùng dự báo thời tiết ECMWF HRES thật và lưu lượng quan trắc; mã, trọng số và dữ liệu công khai (`BaiCoSo/CHECKPDF.md` Mục 5.12).

| Ý tưởng | Cách áp dụng | Giai đoạn |
|---|---|---|
| Dự báo thời tiết thật cho LamaH-CE | Extended LamaH-CE (0,95 GB, theo ngày) có dự báo HRES gộp theo lưu vực — cho phép đưa khí tượng tương lai vào mô hình mà không phải tự tải dữ liệu ECMWF | Khóa luận |
| Baseline mạnh theo ngày | Sequential Forecast LSTM có Q quan trắc (NSE trung vị 0,71) làm mốc khi so Mamba ở bài toán theo ngày | Khóa luận |
| Tách ảnh hưởng của sai số dự báo thời tiết | So mô hình chạy với tái phân tích và với dự báo thật (bài cho thấy NSE trung vị giảm 0,58 → 0,33) | Khóa luận |
| Cùng vùng, cùng khung đánh giá | Cùng LamaH-CE, test 2014–2017 bao trùm kỳ test 2016–2017 của bài cơ sở | Khóa luận |

Pipeline đầy đủ khi dùng BiasCast làm bài cơ sở: `TongQuan/KienTrucPipeline_BiasCast.md`.

Lưu ý: Extended LamaH-CE dùng giấy phép CC BY-NC 4.0 (không dùng cho mục đích thương mại); bản 1.0 ghi chưa phải bản sửa cuối.
