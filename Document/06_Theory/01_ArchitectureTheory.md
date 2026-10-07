# Lý thuyết các kiến trúc Deep Learning — ghi chú học tập

> File cá nhân để học/ôn lại cơ chế hoạt động của từng kiến trúc deep learning liên quan tới đề tài (LSTM, Transformer, Mamba...). Mỗi phần đi từ công thức chuẩn → giải nghĩa từng ký hiệu → ví dụ số cụ thể (dùng bối cảnh dự báo lưu lượng sông cho dễ hình dung, số liệu là minh họa để hiểu cơ chế, không phải số từ model đã train thật).

> *Phạm vi áp dụng:* đề tài dùng dữ liệu theo lưu vực Extended LamaH-CE (`03_Data/01_LamaHCE.md`). Mục 4.7 (Bidirectional), 4.9 (Space-filling curve) và 4.10 là cơ chế riêng của RiverMamba cho **dữ liệu lưới không gian** — có giá trị tham khảo; phần lõi Mamba (4.1–4.6) và LOAN (4.8) vẫn áp dụng trực tiếp.

---

## Phần 1 — LSTM (Long Short-Term Memory)

### 1.1 Vấn đề LSTM giải quyết

Dự báo theo chuỗi thời gian cần "nhớ" chuyện đã xảy ra trước đó (VD: hôm qua mưa to thì hôm nay nước vẫn còn cao dù trời tạnh). LSTM có 2 loại trí nhớ tách biệt:
- **C_t (cell state)** — trí nhớ **dài hạn**, cộng dồn qua nhiều bước.
- **h_t (hidden state)** — câu trả lời/output **tại đúng bước hiện tại**, cũng được đưa sang bước sau làm "trí nhớ ngắn hạn".

### 1.2 Giải nghĩa ký hiệu

| Ký hiệu | Nghĩa là gì |
|---|---|
| `x_t` | Input tại bước t (VD: lượng mưa hôm nay) |
| `h_t` | "Câu trả lời tạm" của model tại bước t |
| `h_(t-1)` | Câu trả lời tạm của bước trước — trí nhớ ngắn hạn |
| `C_t` | Trí nhớ dài hạn — thiết kế để giữ lâu hơn h_t |
| `W_f, W_i, W_o, W_C` | Ma trận trọng số — model **tự học** qua training (random lúc đầu, tự chỉnh dần) |
| `b_f, b_i, b_o, b_C` | Hệ số bias — cũng tự học, cộng thêm vào (công thức đầy đủ có phần này, nhiều tài liệu tóm tắt hay lược bỏ cho gọn) |
| `sigmoid(x)` | Hàm ép số về khoảng **0 đến 1** — dùng làm "công tắc mờ" (0=đóng hẳn, 1=mở hẳn) |
| `tanh(x)` | Hàm ép số về khoảng **-1 đến 1** — dùng tạo giá trị có dấu |
| `[h_(t-1), x_t]` | Ghép 2 dãy số thành 1 dãy dài hơn (concatenate) |

### 1.3 Công thức đầy đủ, 4 cổng

| Cổng | Công thức | Vai trò |
|---|---|---|
| Cổng quên | `forget_gate = sigmoid(W_f · [h_(t-1), x_t] + b_f)` | Giữ lại bao nhiêu % trí nhớ cũ |
| Cổng nạp | `input_gate = sigmoid(W_i · [h_(t-1), x_t] + b_i)` | Cho vào bao nhiêu % thông tin mới |
| Thông tin mới | `candidate = tanh(W_C · [h_(t-1), x_t] + b_C)` | Nội dung cụ thể của thông tin mới |
| Cổng output | `output_gate = sigmoid(W_o · [h_(t-1), x_t] + b_o)` | Công bố bao nhiêu % trí nhớ ra ngoài |

**2 bước tổng hợp:**
```
C_t = forget_gate × C_(t-1)  +  input_gate × candidate     ← cập nhật trí nhớ dài hạn
h_t = output_gate × tanh(C_t)                                ← công bố output bước này
```

### 1.4 Ví dụ đầy đủ — dự báo lưu lượng tại 1 trạm từ mưa, 3 ngày

**Khởi đầu:** `C_0 = 0`, `h_0 = 0` (trí nhớ trống, chưa biết gì).

| | Ngày 1 | Ngày 2 | Ngày 3 |
|---|---|---|---|
| Mưa (x_t) | 5mm | 80mm | 2mm |
| Cổng quên | 0,9 | 0,9 | 0,7 |
| Cổng nạp | 0,5 | 0,95 | 0,2 |
| Thông tin mới | 0,2 | 0,9 | 0,1 |
| **C_t** = quên×C_(t-1) + nạp×mới | 0,9×0 + 0,5×0,2 = **0,10** | 0,9×0,10 + 0,95×0,9 = **0,945** | 0,7×0,945 + 0,2×0,1 = **0,6815** |
| Cổng output | 0,6 | 0,9 | 0,8 |
| tanh(C_t) | 0,0997 | 0,7377 | 0,5923 |
| **h_t** = output × tanh(C_t) | 0,6×0,0997 = **0,06** | 0,9×0,7377 = **0,66** | 0,8×0,5923 = **0,47** |

### 1.5 Từ h_t ra số m³/s thật — 1 lớp cuối cùng

`h_t` luôn nằm trong (-1, 1) — chưa phải đơn vị thật. Cần 1 phép biến đổi tuyến tính cuối, `a` và `b` cũng tự học:
```
Dự_báo (m³/s) = a × h_t + b
```
Minh họa với `a=300, b=100`:

| Ngày | h_t | Dự báo (m³/s) |
|---|---|---|
| 1 | 0,06 | 300×0,06+100 = **118** |
| 2 | 0,66 | 300×0,66+100 = **298** |
| 3 | 0,47 | 300×0,47+100 = **241** |

### 1.6 Điểm mấu chốt

Ngày 3 mưa rất ít (2mm, gần bằng ngày 1) nhưng dự báo (241) vẫn cao hơn nhiều so với ngày 1 (118) — vì `C_t` **cộng dồn** qua các ngày, cổng quên (0,7) chỉ xóa 1 phần trí nhớ "hôm qua mưa to", không xóa sạch. Đây chính là cách LSTM mô phỏng độ trễ lũ thật ngoài đời (nước lũ chưa rút kịp dù trời đã tạnh mưa).

### 1.6b `C_(t-1)` có phải chỉ là "hôm qua" không — cơ chế chuỗi đệ quy

Về công thức, `C_t` chỉ nhìn thấy đúng 1 bước trước (`C_(t-1)`) — nhưng vì `C_(t-1)` bản thân nó cũng được tính từ `C_(t-2)`, và `C_(t-2)` từ `C_(t-3)`... nên `C_t` **gián tiếp mang theo dấu vết của TOÀN BỘ lịch sử từ đầu chuỗi**, không chỉ đúng 1 ngày trước. Giống như viết nhật ký kiểu "tóm tắt tới hôm nay" — mỗi ngày chỉ đọc lại bản tóm tắt hôm qua rồi viết thêm, nhưng vì bản hôm qua đã được xây từ bản hôm kia (cứ thế lùi về), nên đọc đúng 1 bản gần nhất vẫn chứa dư âm của rất lâu về trước (dư âm mờ dần theo cổng quên). Đây là tính chất chung của **mọi kiến trúc dạng RNN** (LSTM, GRU...), không riêng gì LSTM.

### 1.7 Kiểm tra lại

- Công thức khớp đúng chuẩn LSTM gốc (Hochreiter & Schmidhuber 1997, cách trình bày phổ biến nhất).
- Đã tự tính tay lại toàn bộ số liệu ở bảng 1.4/1.5 (kể cả giá trị tanh) — khớp đúng, không có lỗi tính toán.
- Có lược bỏ hệ số bias (b_f, b_i, b_o, b_C) trong ví dụ số cho gọn — không đổi bản chất cơ chế, chỉ là chi tiết kỹ thuật bị bớt.

---

## Phần 2 — Transformer (Self-Attention)

### 2.1 Vấn đề Transformer giải quyết, khác LSTM ở đâu

LSTM đọc tuần tự từng ngày, dựa vào trí nhớ cộng dồn `C_t`. Transformer **nhìn toàn bộ chuỗi cùng lúc**, để mỗi vị trí tự tính "nên chú ý bao nhiêu % tới từng vị trí khác trong chuỗi", rồi trộn thông tin theo đúng % đó — không cần đọc tuần tự.

### 2.2 Q, K, V là gì — gốc thuật ngữ từ dictionary/hash map

```python
my_dict = {"key1": "value1", "key2": "value2"}
result = my_dict["key2"]   # Query = "key2", so khớp Key, lấy đúng Value
```

- **K (Key)** — nhãn gắn với mỗi vị trí, dùng để so khớp.
- **V (Value)** — nội dung thật gắn với Key đó, cái thực sự muốn lấy ra.
- **Q (Query)** — thứ đem đi so với các Key.

**Khác dict thường:** dict phải khớp *chính xác 100%* 1 key, chỉ lấy đúng 1 value. Attention so **độ giống nhau %** với **tất cả** key, rồi lấy **trung bình có trọng số** của tất cả value theo đúng % đó — không chọn hẳn 1 cái, mà pha trộn.

**Công thức tạo ra Q, K, V — 3 phép chiếu tuyến tính độc lập, cùng 1 input:**
```
Q_t = x_t · W_Q
K_t = x_t · W_K
V_t = x_t · W_V
```
`W_Q, W_K, W_V` là 3 bộ trọng số **hoàn toàn riêng, không share nhau khi tạo ra Q/K/V** — nhưng dùng **chung cho mọi vị trí trong chuỗi** (giống LSTM dùng chung W_f/W_i/W_o cho mọi bước thời gian, không phải mỗi ngày 1 bộ trọng số riêng).

**Vì sao cần chiếu qua W thay vì so trực tiếp x:** so trực tiếp giá trị thô (VD 5mm vs 80mm) thì bị khóa cứng vào đúng thang đo đó. Chiếu qua W_Q/W_K (tự học) cho phép model tự định nghĩa "độ tương đồng" theo đúng cái cần cho bài toán, không nhất thiết là "số mm gần nhau".

**Quan hệ giữa 3 cái, đường đi đầy đủ:**
```
x → ×W_Q → Q  ─┐
                ├──→ Q·K → score → softmax → % (tỷ lệ)
x → ×W_K → K  ─┘                                │
                                                   │
x → ×W_V → V  ─────────────────────────────────  ┼──→ % × V → output
```
Q và K **hợp tác tính ra %**; V **đứng riêng**, chỉ xuất hiện đúng 1 lần ở bước cuối — bị trộn theo đúng % vừa tính.

### 2.3 Ví dụ số — bản đơn giản (Q,K,V là 1 số), 2 ngày A (5mm) và B (80mm, hôm nay)

**Bước 1 — tính Q, K, V (W_Q=0,025, W_K=0,02, W_V=0,075 — coi như đã học được):**

| Ngày | x (mưa) | Q | K | V |
|---|---|---|---|---|
| A | 5mm | — (không cần hỏi) | 0,1 | 0,375 |
| B | 80mm | 2 | 1,6 | 6 |

**Bước 2 — so Q_B với từng K (score = Q×K):**
```
score(B,A) = 2 × 0,1  = 0,2
score(B,B) = 2 × 1,6  = 3,2
```

**Bước 3 — softmax ra %:**
```
e^0,2=1,221, e^3,2=24,532, tổng=25,753
% chú ý A = 1,221/25,753 = 4,7%
% chú ý B = 24,532/25,753 = 95,3%
```

**Bước 4 — trộn V theo đúng %:**
```
output_B = 0,047×0,375 + 0,953×6 = 0,018 + 5,718 = 5,736
```

**Bước 5 — lớp cuối ra m³/s (a=50, b=50):**
```
Dự_báo = 50×5,736 + 50 = 336,8 m³/s
```

### 2.4 Vì sao cần chia cho √d_k (scaling) — vấn đề thật khi tính full ma trận

Nếu tính đủ ma trận 2×2 (Q của cả A lẫn B, so với K của cả A lẫn B), ô `(B,B)` sẽ rất lớn (VD 64,0 trong 1 ví dụ minh họa) so với các ô còn lại — vì K_B tự nhân với chính Q_B (đều xuất phát từ 80mm, số lớn). Đưa thẳng số lớn này vào softmax sẽ ra gần 100% dồn vào đúng 1 ô, các ô khác gần 0% — model "cứng đờ". Công thức thật luôn có `÷ √d_k` (d_k = số chiều của Q/K, thực tế 64 hoặc hơn) để giữ điểm số ở mức vừa phải trước khi vào softmax, tránh 1 giá trị áp đảo hoàn toàn. Ví dụ trên dùng d_k=1 nên chưa thấy rõ vấn đề này, nhưng d_k thật (64+) thì bước chia này bắt buộc phải có.

### 2.5 Ví dụ số — bản vector thật (Q,K,V là vector 2 chiều), vẫn 2 ngày A/B

Thực tế Q/K/V không phải 1 số mà là **vector** (thật sự 64 hoặc 512 chiều — ở đây dùng 2 chiều để tính tay được). W giờ là **ma trận**, không phải 1 số.

**Bước 1 — W_Q=[0,02 , 0,01], W_K=[0,015 , 0,02], W_V=[0,03 , 0,05]:**
```
Q_A = 5×[0,02,0,01]  = [0,10 , 0,05]        Q_B = 80×[0,02,0,01] = [1,6 , 0,8]
K_A = 5×[0,015,0,02] = [0,075 , 0,10]       K_B = 80×[0,015,0,02] = [1,2 , 1,6]
V_A = 5×[0,03,0,05]  = [0,15 , 0,25]        V_B = 80×[0,03,0,05] = [2,4 , 4,0]
```

**Bước 2 — score giờ là tích vô hướng (dot product) = nhân từng cặp thành phần rồi CỘNG lại:**
```
score(B,A) = (1,6×0,075)+(0,8×0,10) = 0,12+0,08 = 0,20
score(B,B) = (1,6×1,2)+(0,8×1,6)   = 1,92+1,28 = 3,20
```
*(2 chiều nên cộng 2 số hạng; thực tế 64 chiều thì cộng 64 số hạng — cơ chế y hệt, chỉ dài hơn)*

**Bước 3 — softmax (giống hệt cách tính trước): 4,7% / 95,3%**

**Bước 4 — trộn V (giờ là trộn cả vector, từng chiều 1):**
```
output_B = 0,047×[0,15,0,25] + 0,953×[2,4,4,0]
         = [0,00705,0,01175] + [2,2872,3,812]
         = [2,294 , 3,824]
```
→ Output giờ là **1 vector** (2 số), không phải 1 số như bản đơn giản — mang nhiều thông tin hơn.

**Bước 5 — lớp cuối, W_final=[30,20], b=50:**
```
Dự_báo = 30×2,294 + 20×3,824 + 50 = 68,82+76,48+50 = 195,3 m³/s
```

### 2.6 Multi-head — vì sao cần nhiều "đầu" attention

1 head chỉ học được 1 kiểu quan hệ. Nhiều head (mỗi head có bộ W_Q/W_K/W_V **riêng**) chạy song song, mỗi head tự học 1 góc nhìn khác nhau.

**Head 1** = đúng kết quả mục 2.5: `output_B = [2,294 , 3,824]` (gần như chỉ nhìn chính ngày B — 95,3%)

**Head 2** — bộ trọng số khác: `W_Q2=[0,01,-0,005], W_K2=[0,008,0,01], W_V2=[0,02,-0,01]`
```
Q_B2=[0,8,-0,4]   K_A2=[0,04,0,05]   K_B2=[0,64,0,8]   V_A2=[0,1,-0,05]   V_B2=[1,6,-0,8]

score(B,A)2 = 0,8×0,04+(-0,4)×0,05 = 0,032-0,02 = 0,012
score(B,B)2 = 0,8×0,64+(-0,4)×0,8  = 0,512-0,32 = 0,192

softmax: e^0,012=1,012, e^0,192=1,212, tổng=2,224
% A = 45,5%,  % B = 54,5%     ← CÂN BẰNG hơn hẳn Head 1 (4,7%/95,3%)

output_B (head 2) = 0,455×[0,1,-0,05] + 0,545×[1,6,-0,8] = [0,918 , -0,459]
```
→ **Head 2 nhìn khác hẳn Head 1** dù cùng input — đúng mục đích multi-head: mỗi head tự học 1 kiểu chú ý riêng.

**Ghép 2 head lại, nén qua W_O (ma trận 4→2, tự học):**
```
Ghép: [2,294, 3,824, 0,918, -0,459]
W_O row1=[0,3,0,2,0,1,0,1] → 0,3×2,294+0,2×3,824+0,1×0,918+0,1×(-0,459) = 1,499
W_O row2=[0,1,0,3,0,2,0,2] → 0,1×2,294+0,3×3,824+0,2×0,918+0,2×(-0,459) = 1,468

→ Kết quả sau Multi-head Attention: [1,499 , 1,468]
```

### 2.7 Feed-forward — vì sao cần thêm, không phải dư thừa

**Phân công rõ ràng, không trùng việc với Attention:**

| Khối | Làm việc gì |
|---|---|
| Attention (Q,K,V,W_O) | **Trộn thông tin GIỮA các ngày** |
| Feed-forward | **Xử lý sâu thêm thông tin TRONG từng ngày riêng lẻ**, không nhìn ngày khác nữa |

Input = `[1,499 , 1,468]` (kết quả attention ngày B).

**Lớp phình to (2→4 chiều), qua ReLU:**
```
W_1 = [[0,1,0,2],[0,3,0,1],[0,2,0,2],[0,1,0,3]]
layer1 = W_1 × [1,499,1,468] = [0,4435 , 0,5965 , 0,5934 , 0,5903]
ReLU(x)=max(0,x) → toàn số dương nên không đổi gì ở ví dụ này
```

**Lớp nén nhỏ lại (4→2 chiều):**
```
W_2 = [[0,2,0,1,0,2,0,1],[0,1,0,2,0,1,0,2]]
layer2 = W_2 × layer1 = [0,326 , 0,341]
```
→ Đây là kết quả cuối của **1 lớp Transformer đầy đủ** cho ngày B, sẵn sàng đưa sang lớp tiếp theo (nếu xếp chồng nhiều lớp) hoặc W_final nếu là lớp cuối.

**Lưu ý quan trọng — W_1, W_2 KHÔNG phải riêng của ngày B:** giống hệt W_Q/W_K/W_V/W_O, `W_1` và `W_2` là **1 bộ quy tắc chung duy nhất, dùng cho MỌI ngày**, không đổi theo ngày. Cái riêng của ngày B chỉ là **input** đưa vào (`[1,499, 1,468]`, kết quả attention riêng của B). Nếu tính cho ngày A (giả sử attention output của A là `[0,8, 1,1]`, khác B), dùng **đúng W_1, W_2 y hệt trên**:

```
layer1_A = W_1 × [0,8,1,1] = [0,30 , 0,35 , 0,38 , 0,41]   (ReLU không đổi gì, toàn dương)
layer2_A = W_2 × [0,30,0,35,0,38,0,41] = [0,212 , 0,220]
```

→ Cùng `W_1, W_2`, nhưng vì **input khác nhau** (A khác B) nên **kết quả khác nhau** (`[0,212,0,220]` vs `[0,326,0,341]`). Trọng số không đổi theo ngày, chỉ có dữ liệu chạy qua là khác — đúng nguyên tắc "học 1 quy tắc chung, áp dụng lặp lại cho mọi vị trí" giống CNN (1 filter quét mọi vùng ảnh) và LSTM (1 bộ cổng cho mọi bước thời gian).

### 2.8 Vì sao bắt buộc cần ReLU (hàm phi tuyến) — không có thì hỏng gì

**Chứng minh: xếp nhiều lớp tuyến tính (không ReLU) luôn gộp được thành đúng 1 lớp duy nhất:**
```
Lớp 1: y = 2x
Lớp 2: z = 3y
→ z = 3×(2x) = 6x   ← y hệt chỉ có 1 lớp z=6x, xếp 2 lớp không thêm sức mạnh gì
```

**Thêm ReLU vào giữa thì khác hẳn:**
```
x=-5: y=2×(-5)=-10 → ReLU(-10)=0 → z=3×0=0
x=5:  y=2×5=10     → ReLU(10)=10 → z=3×10=30
```
Nếu vẫn là đường thẳng z=6x thì x=-5 phải ra -30, nhưng thực tế ra 0 — **không còn là đường thẳng**, hàm số "gãy" tại x=0.

**Vì sao "biến âm thành 0" lại tạo được đường cong — ví dụ 3 neuron, mỗi neuron gãy ở 1 ngưỡng mưa khác nhau:**
```
Neuron1 = ReLU(mưa-2)     Neuron2 = ReLU(mưa-5)     Neuron3 = ReLU(mưa-8)
Output = Neuron1+Neuron2+Neuron3
```

| Mưa | Neuron1 | Neuron2 | Neuron3 | Tổng |
|---|---|---|---|---|
| 1mm | 0 | 0 | 0 | **0** |
| 3mm | 1 | 0 | 0 | **1** |
| 6mm | 4 | 1 | 0 | **5** |
| 10mm | 8 | 5 | 2 | **15** |

Độ dốc tăng dần (không cố định) — đường **cong**, đúng hiện tượng thật (mưa ít gần như không ảnh hưởng, mưa nhiều ảnh hưởng tăng vọt do đất bão hòa).

**Nếu bỏ ReLU (không chặn âm), cùng 3 neuron:**

| Mưa | Tổng (không ReLU) = 3×mưa-15 |
|---|---|
| 1mm | **-12** ← vô lý, lưu lượng sông không âm được |
| 3mm | -6 |
| 6mm | 3 |
| 10mm | 15 |

Độ dốc **luôn = 3** mọi đoạn — đường thẳng tuyệt đối, mất khả năng mô phỏng ngưỡng; và ra được giá trị âm vô nghĩa vật lý vì không có gì chặn các số âm cộng dồn tự do.

**Tóm 1 câu:** 1 ReLU chỉ gãy 1 chỗ, nhưng hàng nghìn neuron ReLU với ngưỡng khác nhau cộng lại → xấp xỉ được đường cong bất kỳ (piecewise linear approximation) — sức mạnh nằm ở SỐ LƯỢNG neuron đa dạng ngưỡng, không phải 1 ReLU đơn lẻ.

### 2.9 Positional Encoding — vì sao cần thêm

Attention chỉ tính "độ liên quan" giữa các cặp vị trí — **không tự biết thứ tự thời gian** (khác LSTM, đọc tuần tự nên tự nhiên biết thứ tự). Đảo ngược thứ tự chuỗi, attention thô sẽ ra kết quả y hệt — vô lý cho bài toán có thứ tự thời gian quan trọng như dự báo lũ. Giải quyết bằng cách **cộng thêm 1 vector "vị trí"** vào input trước khi tính Q/K/V (thực tế dùng công thức sin/cos, nhưng ý tưởng là mỗi vị trí có 1 "dấu vân tay" số riêng để model phân biệt được thứ tự).

### 2.10 Toàn bộ pipeline, từ đầu tới cuối

```
Input (mưa mỗi ngày)
  → Embedding (biến số mưa thành vector)
  → + Positional Encoding (cộng thông tin vị trí/thứ tự)
  → Multi-head Self-Attention (nhiều head tính song song → ghép → W_O)
  → Feed-forward (phình to qua ReLU, nén nhỏ lại — xử lý riêng từng vị trí)
  → (lặp lại toàn bộ khối Attention+Feed-forward này N lớp, mỗi lớp có bộ trọng số riêng)
  → W_final → Dự báo m³/s thật
```

### 2.11 Tổng số ma trận trọng số thật (không phải chỉ 4 như ví dụ đơn giản)

| Thành phần | Có trong ví dụ mục 2.3-2.8 chưa? |
|---|---|
| W_Q, W_K, W_V (mỗi head) | ✅ |
| W_O (gộp multi-head) | ✅ (mục 2.6) |
| W_1, W_2 (feed-forward) | ✅ (mục 2.7) |
| W_final (lớp output cuối) | ✅ (mục 2.3/2.5) |

Tổng ước lượng cho model thật: `(6 ma trận × số head × số lớp) + 1 W_final` — VD 8 head × 6 lớp ≈ 288 ma trận, mỗi ma trận chứa hàng nghìn/triệu con số (không phải 1 số như ví dụ minh họa — xem 2.5 để thấy bản vector thật gần hơn quy mô thật, thực tế còn lớn hơn nữa: d_model thật thường 512, không phải 2).

### 2.12 So sánh cốt lõi với LSTM

| | LSTM | Transformer |
|---|---|---|
| Cách "nhớ" | Đọc tuần tự, cộng dồn vào `C_t` | Nhìn toàn bộ chuỗi cùng lúc, tính độ liên quan trực tiếp |
| Biết thứ tự thời gian | Tự nhiên có (đọc từng bước) | Phải thêm Positional Encoding |
| Độ phức tạp tính toán | Tuyến tính theo độ dài chuỗi, nhưng phải tuần tự (không song song hóa được) | O(n²) theo độ dài chuỗi, nhưng tính song song được toàn bộ |
| Ví dụ minh họa trong file | 3 ngày (1,2,3) — ngày 3 mưa ít vẫn dự báo cao vì `C_t` giữ lại phần lớn trí nhớ "ngày 2 mưa to" qua cổng quên | 2 ngày (A mưa ít, B mưa to = hôm nay) — B tự tính ra tỷ lệ % chú ý tới A và tới chính mình, rồi trộn V theo đúng % đó |

*Lưu ý: 2 ví dụ dùng cấu trúc khác nhau (3 ngày vs 2 ngày, vị trí ngày mưa to khác nhau) — không phải cùng 1 bộ số liệu chạy qua 2 kiến trúc để so trực tiếp, chỉ minh họa riêng từng cơ chế.*

### 2.13 Kiểm tra lại

- Công thức khớp chuẩn gốc "Attention Is All You Need" (Vaswani et al. 2017): `Attention(Q,K,V) = softmax(QK^T/√d_k)·V`, có Multi-head, Feed-forward, Positional Encoding — đủ đúng 5 thành phần chính.
- Đã tự tính tay lại **toàn bộ** số liệu ở mục 2.3, 2.5, 2.6, 2.7, 2.8 (kể cả các phép softmax, dot product, ReLU) — khớp đúng, không có lỗi tính toán.
- Toàn bộ số liệu trong file là **ví dụ minh họa tự chọn để hiểu cơ chế**, không phải số từ model đã train thật — quy mô thật (d_model, số head, số lớp) lớn hơn rất nhiều lần so với ví dụ.

---

## Phần 3 — GRU (Gated Recurrent Unit)

### 3.1 Vấn đề GRU giải quyết

Dự báo chuỗi thời gian cần 1 "bộ nhớ" theo dõi diễn biến qua các ngày, và tại mỗi ngày mới phải tự quyết định: giữ nguyên bộ nhớ cũ, hay cập nhật theo thông tin hôm nay — giữ/cập nhật bao nhiêu %. GRU dùng **đúng 1 trạng thái duy nhất** `h_t` (vừa là bộ nhớ, vừa là output), khác LSTM có 2 trạng thái tách biệt.

### 3.2 Giải nghĩa ký hiệu

| Ký hiệu | Nghĩa |
|---|---|
| `x_t` | Input ngày t (mưa) |
| `h_t` | Trạng thái/bộ nhớ tại ngày t — cũng là output |
| `h_(t-1)` | Bộ nhớ ngày trước |
| `W_r, W_z, W_h` | Ma trận trọng số — tự học |
| `r_t` | Cổng reset (0-1) |
| `z_t` | Cổng update (0-1) |
| `h̃_t` | Candidate — nội dung mới đề xuất |

### 3.3 Công thức đầy đủ, giải thích lý do từng bước

**Bước 1 — Cổng reset: "tạo nội dung mới thì dùng bao nhiêu % bộ nhớ cũ?"**
```
r_t = sigmoid(W_r · [h_(t-1), x_t])
```

**Bước 2 — Tạo nội dung mới, dùng bộ nhớ cũ ĐÃ qua lọc bởi cổng reset:**
```
h̃_t = tanh(W_h · [r_t × h_(t-1), x_t])
```

**Bước 3 — Cổng update: "cuối cùng, giữ bao nhiêu % bộ nhớ cũ NGUYÊN VẸN, lấy bao nhiêu % nội dung mới?"**
```
z_t = sigmoid(W_z · [h_(t-1), x_t])
```

**Bước 4 — Trộn theo tỷ lệ cổng update (2 phần luôn cộng = 100%):**
```
h_t = (1 - z_t) × h_(t-1)  +  z_t × h̃_t
```

### 3.4 Ví dụ số đầy đủ — mưa 3 ngày tại 1 trạm (5mm, 80mm, 2mm)

**Khởi đầu:** `h_0 = 0`

| | Ngày 1 (5mm) | Ngày 2 (80mm) | Ngày 3 (2mm) |
|---|---|---|---|
| r_t | 0,5 | 0,8 | 0,7 |
| z_t | 0,6 | 0,9 | 0,3 |
| h̃_t | 0,15 | 0,85 | 0,10 |
| **h_t** = (1-z)×h_(t-1)+z×h̃_t | 0,4×0+0,6×0,15=**0,09** | 0,1×0,09+0,9×0,85=0,009+0,765=**0,774** | 0,7×0,774+0,3×0,10=0,5418+0,03=**0,5718** |

### 3.5 Ra số m³/s thật

```
Dự_báo = a×h_t + b,  a=300, b=100
```

| Ngày | h_t | Dự báo (m³/s) |
|---|---|---|
| 1 | 0,09 | 300×0,09+100 = **127** |
| 2 | 0,774 | 300×0,774+100 = **332,2** |
| 3 | 0,5718 | 300×0,5718+100 = **271,5** |

### 3.6 Điểm mấu chốt

Ngày 3 mưa ít nhưng dự báo vẫn cao (271,5) — vì `z_3=0,3` nghĩa là giữ lại **70%** giá trị `h_2` (đang cao vì mưa to hôm trước).

### 3.7 GRU có nhớ dài hạn không

**Có.** Nếu `z_t` nhỏ liên tục qua nhiều ngày → `h_t` gần như giữ nguyên `h_(t-1)` mỗi bước → thông tin từ rất lâu trước vẫn còn sót lại. Không phải "không có" trí nhớ dài hạn, chỉ là lưu bằng 1 trạng thái duy nhất thay vì 2 như LSTM.

### 3.8 `h_(t-1)` có phải chỉ là "hôm qua" không — chuỗi đệ quy

Về công thức thì đúng chỉ là 1 bước trước, nhưng vì `h_(t-1)` được xây đệ quy từ `h_(t-2)`, `h_(t-2)` từ `h_(t-3)`... nên nó **gián tiếp mang theo cả lịch sử từ đầu chuỗi**, dù công thức mỗi bước chỉ viết đúng 1 biến "bước trước" (giống hệt cơ chế đã nói ở mục 1.6b cho LSTM — tính chất chung của mọi kiến trúc dạng RNN).

### 3.9 So sánh cơ chế với LSTM — khác nhau ở đâu

| | LSTM | GRU |
|---|---|---|
| Số trạng thái | 2 (`C_t` dài hạn + `h_t` output riêng) | **1** (`h_t` làm cả 2 việc) |
| Số cổng | 4 (quên, nạp, output, +candidate) | **2** (reset, update) |
| Cách quên/nạp | **Độc lập** — 2 cổng riêng, tổng % không nhất thiết = 100% | **Gộp chung** — 1 cổng update, `(1-z)` và `z` luôn cộng = 100% |
| Lọc bộ nhớ cũ trước khi tạo nội dung mới | Không có bước này — candidate tính thẳng từ `h_(t-1)` nguyên vẹn | **Có** — cổng reset lọc bớt `h_(t-1)` trước khi đưa vào tính candidate |
| Số ma trận trọng số | 4 (`W_f,W_i,W_o,W_C`) | **3** (`W_r,W_z,W_h`) — ít hơn 25% |
| Chuỗi đệ quy mang lịch sử | Có (qua `C_t`) | Có (qua `h_t`) — cùng nguyên lý |
| Tốc độ train | Chậm hơn (nhiều tham số hơn) | Nhanh hơn |
| Độ chính xác thực tế | Gần tương đương nhau, tùy bài toán | Gần tương đương nhau, tùy bài toán — không cái nào áp đảo hẳn (đúng như nhiều bài trong `02_LiteratureReview.md` dùng cả LSTM lẫn GRU làm baseline song song) |

### 3.10 Kiểm tra lại

- Công thức khớp chuẩn GRU gốc (Cho et al. 2014, "Learning Phrase Representations using RNN Encoder-Decoder").
- Đã tự tính tay lại toàn bộ số liệu ở bảng 3.4/3.5 — khớp đúng, không có lỗi tính toán.
- Số liệu là ví dụ minh họa để hiểu cơ chế, không phải số từ model đã train thật.

---

## Phần 4 — Mamba (State Space Model)

> Phần 4 gồm: động lực (4.1–4.2), SSM (4.3), rời rạc hóa (4.4), Selective SSM (4.5), Selective Scan (4.6), Bidirectional Mamba (4.7), LOAN (4.8), Space-filling curve serialization (4.9) và ví dụ tổng hợp một lớp Hindcast (4.10).

### 4.1 Vấn đề Mamba giải quyết — 3 mục tiêu cùng lúc

Đã biết 2 kiến trúc trước:
- **LSTM/GRU:** đọc **tuần tự từng bước** — nhớ tốt, nhưng **không tính song song được** (phải xong bước 1 mới tính bước 2) → chuỗi dài thì train rất chậm.
- **Transformer:** nhìn **toàn bộ chuỗi cùng lúc**, tính song song được (nhanh) — nhưng độ phức tạp **O(n²)**, chuỗi càng dài càng tốn RAM/thời gian theo cấp số nhân.

**Mamba muốn có cả 3 cái tốt cùng lúc:**
1. Tính **song song được** (nhanh như Transformer, không phải đọc tuần tự như LSTM).
2. Độ phức tạp **tuyến tính O(n)** (rẻ như LSTM, không phải O(n²) như Transformer).
3. Vẫn **nhớ tốt** theo thời gian (như LSTM/GRU).

### 4.2 Tính toán ít hơn Transformer thì có kém chính xác hơn không

**Trả lời ngắn: KHÔNG rõ rệt** — đã verify thật ở file `02_LiteratureReview.md`, Mục 1 (khảo sát Mamba — *khác Phần 1 của chính file này, Phần 1 ở đây là LSTM*): 2 bài (463 và 68 trích dẫn, số tra 19/8/2026) đều kết luận Mamba và Transformer **cạnh tranh ngang nhau** về độ chính xác, dù Mamba tính ít hơn hẳn (O(n) vs O(n²)).

**Vì sao "tính ít hơn" ≠ "kém chính xác hơn":** Độ phức tạp (Big-O) đo **lượng phép tính**, không đo **độ thông minh của thuật toán**. Thuật toán rẻ hơn hoàn toàn có thể đạt kết quả ngang bằng nếu nó chọn việc để làm thông minh hơn, thay vì tính kiểu "vét cạn".

**Ví dụ đối chiếu — thuật toán sắp xếp:**
- Bubble sort: O(n²) — so sánh **mọi cặp** phần tử (vét cạn), rất tốn.
- Merge sort: O(n log n) — nhanh hơn nhiều, nhưng **kết quả đúng y hệt** (mảng sắp xếp hoàn hảo).
- Không ai nói merge sort "kém chính xác hơn" vì nó tính ít hơn — nó chỉ khôn hơn trong cách làm.

**Áp vào Mamba vs Transformer:**
- Transformer: so sánh **mọi cặp vị trí** với nhau (kiểu vét cạn, giống bubble sort) → O(n²).
- Mamba: dùng cơ chế "chọn lọc" (Selective — sẽ học chi tiết cơ chế toán ở phần sau) để tự quyết định thông tin nào đáng giữ, không cần so sánh vét cạn mọi cặp → O(n), nhưng vẫn hiệu quả tương đương nhờ chọn lọc thông minh.

### 4.2b Đánh đổi thật sự tồn tại — không phải "miễn phí hoàn toàn"

Mamba **nén toàn bộ lịch sử vào 1 trạng thái kích thước CỐ ĐỊNH** (giống LSTM có `C_t` kích thước cố định). Transformer **giữ nguyên K, V của TỪNG vị trí** trong quá khứ, không nén — khi cần, tra lại chính xác đúng vị trí đó.

→ Về lý thuyết (nếu 2 bên xử lý **cùng độ dài chuỗi**): Transformer tra lại chi tiết xa xưa chính xác hơn, vì Mamba có thể mất mát chút ít do bị nén.

### 4.2c Nhưng thực tế lại khác — vì sao Mamba vẫn hợp lý cho chuỗi dài

**Vấn đề của Transformer trong thực tế:** O(n²) khiến nó **không xử lý được chuỗi thật sự dài**. Khi cần chuỗi dài, Transformer thường bị **buộc phải cắt bớt context** (VD chỉ nhìn 100 ngày gần nhất) — dữ liệu xa hơn **bị cắt bỏ hoàn toàn**, không phải "nhớ kém" mà là "không được thấy luôn". Lợi thế "nhớ chi tiết xa" của Transformer vì vậy **không có cơ hội phát huy** trong thực tế nếu chuỗi quá dài.

**Mamba nhờ rẻ (O(n))** mà xử lý được chuỗi dài hơn nhiều trong cùng ngân sách tính toán — VD minh họa (không phải số thật của RiverMamba, chỉ để dễ hình dung tỷ lệ): có thể "thấy" 1000 ngày, trong khi Transformer chỉ đủ sức thấy 100 ngày trong cùng ngân sách GPU.

**→ Nghịch lý:** dù Mamba nén thông tin (mất chút chi tiết), nhưng vì thấy được **nhiều dữ liệu lịch sử hơn hẳn**, tổng thể có khi Mamba lại nắm nhiều thông tin hữu ích hơn Transformer bị giới hạn cửa sổ nhìn ngắn.

**Với đúng bài toán dự báo lưu lượng — "chi tiết siêu nhỏ xa xưa" có thật sự cần không?** Dự báo lũ cần biết **xu hướng tích lũy** (đất đã bão hòa chưa, mùa mưa đang giai đoạn nào) — thông tin dạng **tổng hợp/thống kê**, không phải "nhớ chính xác đúng con số mưa ngày thứ 47 cách đây rất lâu". Kiểu thông tin tổng hợp này hợp với cách nén của Mamba hơn — khác bài toán kiểu "tìm đúng 1 câu nói cụ thể ở đầu văn bản rất dài" (mới thật sự cần Transformer nhớ chính xác từng chi tiết).

**Liên hệ ràng buộc thực tế của đề tài:** chạy trên GPU free, ngân sách hạn chế — dùng Transformer với chuỗi dài (nhiều ngày lịch sử) có thể **vượt quá khả năng GPU free** vì chi phí O(n²). Mamba cho phép dùng context dài hơn mà vẫn nằm trong ngân sách — lý do thực tế, không chỉ lý thuyết, để chọn Mamba.

### 4.2d Bằng chứng từ tài liệu

Tra thêm literature xác nhận đúng cả 2 chiều, cần nói thẳng cả ưu lẫn nhược, không chỉ nói phần có lợi cho Mamba:

- **Xác nhận đúng phần "nén mất chi tiết":** SSM (họ của Mamba) nén toàn bộ quá khứ vào 1 trạng thái kích thước cố định — đây là hạn chế **thật, đã được chính literature Mamba thừa nhận**, không phải suy diễn riêng của tôi.
- **Phát hiện quan trọng cần biết:** có 1 loại benchmark riêng gọi là **"needle-in-the-haystack"** (tìm đúng 1 chi tiết cụ thể được giấu ở đâu đó trong 1 chuỗi rất dài) — trên loại benchmark này, **Transformer có khả năng nhớ chính xác cao hơn Mamba rõ rệt**, vì Transformer tra được đúng vị trí gốc (không nén), còn Mamba phải dựa vào bản đã nén (có thể đã mất chi tiết đó).
- **Kết luận đúng, cân bằng:** Mamba **không phải lúc nào cũng thắng** Transformer về khả năng nhớ — cụ thể là **thua** ở bài toán cần nhớ chính xác 1 chi tiết nhỏ giữa chuỗi cực dài. Mamba chỉ có lợi thế rõ trong bối cảnh cần xử lý chuỗi dài hơn mức Transformer có thể kham nổi (do ràng buộc tính toán), và cho bài toán cần thông tin **tổng hợp/xu hướng** hơn là **chi tiết đơn lẻ chính xác**.
- **Ý nghĩa cho đề tài — cần nói thật khi bảo vệ:** dự báo lưu lượng gần với dạng "thông tin tổng hợp" (xu hướng mưa, độ bão hòa đất) hơn là "tìm đúng 1 chi tiết nhỏ" — nên đây là lý do hợp lý để dùng Mamba, nhưng **không nên khẳng định tuyệt đối** "Mamba nhớ tốt hơn Transformer" — chỉ nên nói "phù hợp hơn cho đúng dạng bài toán và ràng buộc tính toán của đề tài này".

---

### 4.3 State Space Model (SSM) — nền tảng toán của Mamba

#### 4.3.1 Nguồn gốc

State Space Model không phải phát minh riêng cho AI — là công cụ có sẵn từ **lý thuyết điều khiển tín hiệu** (kỹ thuật điện, tự động hóa), mô tả 1 hệ thống vật lý thay đổi theo thời gian. Mamba mượn lại đúng công cụ này.

#### 4.3.1b Quy ước ký hiệu — Phần 4 dùng 2 kiểu viết cho cùng 1 khái niệm

Toàn bộ Phần 4 (Mamba) dùng lẫn lộn 2 cách viết trạng thái — **cả 2 đều chỉ cùng 1 thứ**, không phải 2 khái niệm khác nhau:

| Vai trò | Kiểu ngoặc đơn (dùng ở 4.3–4.5) | Kiểu chỉ số dưới (dùng ở 4.4 cuối, 4.6, và LSTM/GRU) |
|---|---|---|
| Trạng thái CŨ (đầu vào) | `h(t)` | `h_(t-1)` |
| Trạng thái MỚI (kết quả) | `h(t+1)` | `h_t` |
| Trạng thái tại 1 ngày cụ thể (VD ngày 2) | `h(2)` | `h_2` |

**Lý do lẫn lộn:** Mục 4.3 dựng công thức theo kiểu toán liên tục (dùng `h(t)/h(t+1)`, khớp với cách trình bày SSM gốc trong sách điều khiển tín hiệu). Mục 4.6 và LSTM/GRU dùng kiểu chỉ số dưới quen thuộc hơn với dân deep learning. **Quy đổi:** `h_t ≡ h(t+1)`, `h_(t-1) ≡ h(t)` — chỉ khác cách đặt tên mốc thời gian, không đổi bản chất.

#### 4.3.2 Ý tưởng cốt lõi — ví dụ bồn nước

**Lưu ý trước khi đọc tiếp:** ví dụ bồn nước dưới đây chỉ dùng để hiểu **công thức toán** — không phải cách Mamba hoạt động thật trong AI. Xem khác biệt quan trọng ở mục 4.3.2b ngay sau đây.

- **Trạng thái (state)** = mực nước trong bồn ngay lúc này.
- **Input** = tốc độ bơm nước vào mỗi giây.
- **Quy luật:** mực nước thay đổi dựa trên **mực nước hiện tại** (bồn càng đầy càng tự rút nhanh qua khe hở) **cộng thêm** ảnh hưởng của việc đang bơm vào.

#### 4.3.2b Khác biệt quan trọng nhất — bồn nước thật (đo được) vs Mamba/AI thật (không đo được)

Trong ví dụ bồn nước, `h(t)` = mực nước — **đo lại được thật** bằng thước đo, độc lập mỗi ngày, không cần dữ liệu hôm qua.

**Nhưng trong Mamba/AI thật (dự báo lưu lượng), `h(t)` KHÔNG PHẢI đại lượng vật lý cụ thể nào cả** — nó là 1 vector số trừu tượng (VD RiverMamba dùng 192 con số, K=192) mà model **tự học ra** để tóm tắt "những gì cần nhớ" — không map vào 1 đại lượng đo được nào (không phải "mực nước", không phải "độ ẩm đất"...). **Không có cảm biến nào đo được `h(t)` trực tiếp.**

| | Bồn nước (ví dụ dạy toán) | Mamba/AI thật (dự báo lưu lượng) |
|---|---|---|
| `h(t)` là gì | Mực nước — **đo lại được** bằng thước đo thật | Vector số trừu tượng — **KHÔNG đo được**, chỉ tính ra |
| `x(t)` là gì | Tốc độ bơm | Mưa thật — **đo được** bằng trạm đo mưa mỗi ngày |
| Nếu không lưu trạng thái | Đo lại từ đầu được, không sao | **Mất trí nhớ hoàn toàn** — không có cách lấy lại |

**Ví dụ chứng minh vì sao chỉ "đo lại mưa mỗi ngày" là chưa đủ:** Ngày 2 mưa to (80mm), ngày 3 mưa ít (2mm). Nếu model chỉ nhìn đúng mưa ngày 3 (bỏ qua hoàn toàn "trí nhớ" từ ngày 2) → dự báo THẤP, sai — vì bỏ qua việc nước lũ ngày 2 chưa rút hết. Phải mang `h(3)` (đã "nhớ" ngày 2, tính từ trước) qua thì mới dự báo đúng.

Đây chính là lý do mục 4.3.7 (ngay sau đây) — cần tính và lưu `h(t+1)` — lại quan trọng tới vậy.

#### 4.3.3 Mục tiêu — nối thẳng với LSTM đã biết

LSTM tính: **trạng thái CŨ (`C_(t-1)`) + input MỚI (`x_t`) → trạng thái MỚI (`C_t`)**. SSM muốn làm đúng y hệt việc này, chỉ khác cách đặt tên mốc thời gian:

| | LSTM | SSM |
|---|---|---|
| Trạng thái CŨ (đầu vào) | `C_(t-1)` | `h(t)` |
| Input hiện tại | `x_t` | `x(t)` |
| Trạng thái MỚI (kết quả) | `C_t` | `h(t+1)` |

Không có gì khác về ý tưởng — chỉ khác tên gọi mốc thời gian.

#### 4.3.4 Bảng ký hiệu đầy đủ

| Ký hiệu | Nghĩa |
|---|---|
| `x(t)` | Input tại thời điểm t (VD: bơm đang chạy) |
| `h(t)` | Trạng thái CŨ, tại thời điểm t (VD: mực nước hiện tại) |
| `h(t+1)` | Trạng thái MỚI — **không phải kết quả dùng ngay, mà là "trí nhớ" chuẩn bị cho bước sau** (xem 4.3.7) |
| `h'(t)` | Tốc độ thay đổi ngay lúc t — chỉ là BƯỚC TRUNG GIAN để tính `h(t+1)`, không phải kết quả cuối. Giống đồng hồ tốc độ (km/h) khác vị trí xe (km) |
| `A` | Hằng số: trạng thái tự ảnh hưởng chính nó ra sao (VD: bồn càng đầy tự rút càng nhanh) |
| `B` | Hằng số: input ảnh hưởng tới trạng thái mạnh cỡ nào (VD: bơm mạnh cỡ nào) |
| `C` | Hằng số: cách đọc kết quả quan sát được từ trạng thái |
| `y(t)` | Output quan sát/đo được **của riêng bước t**, suy ra từ `h(t)` |

#### 4.3.5 Công thức đầy đủ

```
h'(t) = A·h(t) + B·x(t)              ← Bước trung gian: tốc độ thay đổi
h(t+1) = h(t) + h'(t)×(thời gian)    ← Trạng thái mới, CHUYỂN TIẾP cho bước sau
y(t)  = C·h(t)                        ← Output của riêng bước t, dùng h(t) (không cần h(t+1))
```

#### 4.3.6 Ví dụ số đầy đủ — bồn nước

**Thông số hệ thống:** `A=-0,1`, `B=0,3`, `C=1,5`. **Trạng thái cũ:** `h(t)=2` (mét). **Input:** `x(t)=1`. **Thời gian:** 1 giây.

```
Bước 1 (có sẵn):     x(t)=1,  h(t)=2
Bước 2 (trung gian): h'(t) = (-0,1×2)+(0,3×1) = -0,2+0,3 = 0,1
Bước 3 (chuyển tiếp):h(t+1) = 2 + 0,1×1 = 2,1
Bước 4 (output hôm nay): y(t) = 1,5×2 = 3
```

**Đối chiếu — nếu tắt bơm (`x(t)=0`):** `h'(t) = -0,2` → `h(t+1) = 2+(-0,2)×1 = 1,8` (mực nước giảm).

#### 4.3.7 Điểm quan trọng nhất — vì sao cần tính `h(t+1)` dù `y(t)` không cần tới nó

`y(t) = C·h(t)` chỉ dùng `h(t)` — không hề cần `h(t+1)`. Vậy tính `h(t+1)` để làm gì? **Có đủ 2 lý do, cộng lại:**

**Lý do 1 — `h(t+1)` là "trí nhớ" chuẩn bị sẵn cho NGÀY MAI dùng:**

```
Hôm nay (bước t):     y(t) = C·h(t)         ← dùng h(t), ra kết quả hôm nay
                       h(t+1) = h(t)+h'(t)   ← tính thêm, CHƯA dùng hôm nay

Ngày mai (bước t+1):  y(t+1) = C·h(t+1)      ← ngày mai cần TỚI h(t+1) này!
```

**Đối chiếu đúng LSTM:** LSTM tính `C_t` xong, `C_t` không dùng ngay cho gì trong ngày t (chỉ `h_t=output_gate×tanh(C_t)` mới là kết quả dùng ngay) — nhưng `C_t` bắt buộc phải tính, vì **ngày mai** nó đóng vai trò `C_(t-1)` để tính tiếp. `h(t+1)` trong SSM đóng đúng vai trò này.

**Lý do 2 (quan trọng hơn, dễ nhầm nhất) — không có cách nào khác để có được `h(t+1)`, vì trạng thái không đo lại được (xem 4.3.2b):**

Khác với `x(t)` (mưa — đo lại được thật mỗi ngày), `h(t)` trong Mamba/AI thật **không phải đại lượng đo được** — nó chỉ tồn tại nếu được **tính ra** từ bước trước. Ngày mai muốn có "trạng thái cũ" để bắt đầu tính tiếp, **không có lựa chọn nào khác** ngoài việc dùng đúng con số đã tính hôm nay — không có "cảm biến trạng thái" nào để đo lại thay thế.

```
Nếu KHÔNG tính h(t+1) hôm nay
        ↓
Ngày mai không có "trạng thái cũ" nào để bắt đầu tính
        ↓
Vì h(t) không đo lại được (khác x(t) — đo lại được thật)
        ↓
→ Model "mất trí nhớ", coi như bắt đầu lại từ số 0 mỗi ngày
        ↓
→ Mất khả năng "nhớ" ảnh hưởng của những ngày trước (VD mưa to ngày 2 vẫn ảnh hưởng dự báo ngày 3)
```

**Tóm bảng — mỗi bước có 2 việc tách biệt:**

| | Dùng để làm gì |
|---|---|
| `y(t)` (dùng `h(t)`) | Kết quả/output **của riêng ngày hôm nay** |
| `h(t+1)` | Không phải kết quả — là "trí nhớ" chuyển tiếp, chuẩn bị sẵn cho **ngày mai**, và là cách DUY NHẤT để trí nhớ tồn tại được (vì không đo lại được) |

Nếu chuỗi chỉ có đúng 1 ngày (không có ngày mai) thì không cần tính `h(t+1)`. Nhưng bài toán dự báo nhiều ngày liên tiếp thì mỗi ngày đều phải chuẩn bị sẵn `h(t+1)` để ngày sau dùng tiếp — đây là cách "trí nhớ" được truyền từ ngày này sang ngày khác, và là cách duy nhất vì không có cảm biến nào đo lại được trạng thái nội bộ.

#### 4.3.8 Vì sao cần bước "rời rạc hóa" tiếp theo

Máy tính không tính được `h'(t)` (đạo hàm liên tục) trực tiếp — Bước 2+3 ở 4.3.6 thực chất là cách "giả lập" thủ công. Bước **rời rạc hóa** (sắp học) sẽ gộp Bước 2+3 lại thành đúng 1 công thức chuẩn hóa duy nhất, dạng `h_t = Ā·h_(t-1) + B̄·x_t` — tính thẳng ra trạng thái mới trong 1 bước, giống hệt cách viết LSTM/GRU.

---

### 4.4 Rời rạc hóa — gộp `h'(t)` và `h(t+1)` thành 1 công thức duy nhất

#### 4.4.1 Ý tưởng — gộp 2 bước ở mục 4.3.6 lại

Nhắc lại 2 bước cũ:
```
h'(t) = A·h(t) + B·x(t)
h(t+1) = h(t) + h'(t)×Δt
```
Thay thẳng `h'(t)` vào công thức thứ 2:
```
h(t+1) = h(t) + [A·h(t)+B·x(t)]×Δt = (1+A·Δt)·h(t) + (B·Δt)·x(t)
```
Đặt `Ā = 1+A·Δt`, `B̄ = B·Δt`:
```
h(t+1) = Ā·h(t) + B̄·x(t)
```
Không còn `h'(t)` nữa — chỉ 1 công thức, đúng dạng `h_t=Ā·h_(t-1)+B̄·x_t` giống LSTM/GRU.

#### 4.4.2 Kiểm chứng — dùng lại đúng số ví dụ bồn nước (`A=-0,1`, `B=0,3`, `Δt=1`, `h(t)=2`, `x(t)=1`)

```
Ā = 1+(-0,1×1) = 0,9        B̄ = 0,3×1 = 0,3
h(t+1) = 0,9×2 + 0,3×1 = 1,8+0,3 = 2,1
```
Khớp chính xác kết quả tính 2 bước ở mục 4.3.6 (`h(t+1)=2,1`) — chứng minh rời rạc hóa chỉ là gộp lại bằng đại số, không đổi kết quả.

#### 4.4.3 Lưu ý kỹ thuật — đã tra xác nhận lại (14/8/2026, không suy đoán)

Cách `Ā=1+A·Δt` ở trên (xấp xỉ Euler) là bản đơn giản để hiểu. Công thức Mamba/SSM thật dùng `Ā=exp(A·Δt)` ("Zero-Order Hold" — ZOH) — ổn định hơn về số học khi `Δt` hoặc `A` lớn.

**Còn `B̄` thì sao — chi tiết dễ hiểu nhầm, đã tra rõ:** công thức ZOH chính xác cho `B̄` thật ra phức tạp hơn nhiều: `B̄ = A⁻¹(exp(A·Δt)−I)·B` (không đơn giản như `Ā`). Nhưng **Mamba KHÔNG dùng công thức chính xác này cho `B̄`** — code gốc của Mamba **cố tình dùng hỗn hợp**: `Ā` theo đúng ZOH (`exp(AΔt)`), còn `B̄` vẫn giữ kiểu Euler đơn giản (`B̄=Δt·B`, đúng như đã học ở 4.4.1). Đây là lựa chọn **có chủ đích** của tác giả Mamba (dễ cài đặt hơn hẳn, không cần nghịch đảo ma trận `A⁻¹` tốn kém) — đã xác nhận qua nhiều phân tích kỹ thuật: cách làm này **không ảnh hưởng đáng kể tới hiệu năng thực nghiệm**, vì về bản chất Euler là 1 xấp xỉ hợp lý của ZOH khi `Δt` nhỏ (`exp(x)≈1+x`).

Tóm lại: `Ā,B̄` trong toàn bộ ví dụ số ở file này (4.4-4.6) dùng đúng kiểu Euler đơn giản cho cả 2 — khớp đúng cách Mamba thật cài đặt cho `B̄` (dù không khớp hoàn toàn cách Mamba thật tính `Ā`, vốn dùng `exp()`). Nguồn: phân tích kỹ thuật Mamba (bài "Mamba: SSM, Theory and Implementation") và paper Mamba-3 (ICLR 2026, arxiv 2603.15569) — chính thức gọi tên cách làm này là "exponential-Euler discretization".

---

### 4.5 Selective SSM — cơ chế "chọn lọc" làm nên Mamba

#### 4.5.1 Vấn đề của SSM cổ điển (chưa chọn lọc)

`Ā, B̄` ở mục 4.4 là **hằng số cố định** — dùng y hệt mọi ngày, bất kể input là gì. Mưa 1mm hay 100mm vẫn nhân đúng 1 hệ số — model không tự "quyết định" được input nào quan trọng hơn.

#### 4.5.2 LSTM đã giải quyết vấn đề này ra sao — SSM cổ điển còn thiếu gì

LSTM tính cổng **từ chính input**: `forget_gate=sigmoid(W_f·[h_(t-1),x_t])` — mỗi ngày ra cổng khác nhau tùy input. SSM cổ điển không có cơ chế này.

#### 4.5.3 Ý tưởng Mamba — làm `B, C, Δ` tính từ input mỗi bước, `A` vẫn giữ cố định

```
B_t = w_B × x_t
C_t = w_C × x_t
Δ_t = softplus(w_Δ × x_t)     ← softplus(z)=ln(1+e^z), luôn dương
```
`w_B, w_C, w_Δ` là hằng số cố định (tự học) — nhưng vì nhân với `x_t` (khác nhau từng ngày), `B_t, C_t, Δ_t` ra số khác nhau mỗi ngày.

#### 4.5.4 Ví dụ số đầy đủ — 3 ngày mưa (5mm, 80mm, 2mm), `w_B=0,001, w_C=0,02, w_Δ=0,01`, `A=-0,1`

| Ngày | x | B_t=w_B×x | C_t=w_C×x | Δ_t=softplus(w_Δ×x) |
|---|---|---|---|---|
| 1 | 5mm | 0,005 | 0,1 | softplus(0,05)=0,719 |
| 2 | 80mm | 0,08 | 1,6 | softplus(0,8)=1,171 |
| 3 | 2mm | 0,002 | 0,04 | softplus(0,02)=0,703 |

**Rời rạc hóa, `Ā_t=1+A×Δ_t`, `B̄_t=B_t×Δ_t`:**
```
Ā_1=1+(-0,1×0,719)=0,928    B̄_1=0,005×0,719=0,0036
Ā_2=1+(-0,1×1,171)=0,883    B̄_2=0,08×1,171=0,0937
Ā_3=1+(-0,1×0,703)=0,930
```

**Tính trạng thái (`h(1)=0`):**
```
h(2) = Ā_1×h(1)+B̄_1×x_1 = 0,928×0+0,0036×5 = 0,018
```

**Tính output quan sát được, dùng `C_t`:**
```
y(2) = C_2×h(2) = 1,6×0,018 = 0,0288
```

**Lưu ý:** input mưa thô (mm) chưa chuẩn hóa nên số có thể lớn nhanh khi mưa to — thực tế Mamba xử lý dữ liệu đã qua embedding/chuẩn hóa trước, không dùng thẳng số mm thô như ví dụ minh họa.

#### 4.5.5 Vì sao `B_t=w_B×x_t` KHÁC hẳn `B` cố định trong SSM gốc — không phải đổi tên suông

**SSM gốc:** `x_t` xuất hiện **1 lần** — nhân với hệ số `B` cố định. Ảnh hưởng tỷ lệ **thẳng** với `x_t`.

**Selective SSM:** `x_t` xuất hiện **2 lần** — 1 lần để tính `B_t`, 1 lần nữa khi `B_t` nhân lại với `x_t`. Ảnh hưởng cuối cùng tỷ lệ với **`x_t²`**.

**Chứng minh bằng số — mưa tăng 16 lần (5mm→80mm):**

| | `B` cố định=0,01 | `B_t` chọn lọc, `w_B=0,001` |
|---|---|---|
| Mưa 5mm | 0,01×5=0,05 | (0,001×5)×5=0,025 |
| Mưa 80mm | 0,01×80=0,8 | (0,001×80)×80=6,4 |
| **Tỷ lệ tăng** | **16 lần** (đúng tỷ lệ mưa) | **256 lần** (=16², vì x_t nhân 2 lần) |

→ Mưa càng to, model "để ý" theo cấp số nhân, không chỉ tỷ lệ thường — đúng đặc tính lũ thật (mưa vượt ngưỡng thì ảnh hưởng bùng nổ).

#### 4.5.6 Vì sao `A` KHÔNG được làm chọn lọc trực tiếp — đã tra xác nhận, không suy đoán

Theo bài gốc: `A` giữ cố định **để đơn giản hóa tính toán**, nhưng ảnh hưởng của nó vẫn thay đổi theo input **gián tiếp qua `Δ_t`** (vì `Ā_t = 1+A×Δ_t`, và `Δ_t` đã chọn lọc). Không cần làm `A` tự thay đổi trực tiếp (phức tạp thêm không cần thiết) — `Δ` chọn lọc là đủ để "mượn" luôn tính chọn lọc gián tiếp cho `A`.

**Chứng minh bằng đúng số đã tính ở 4.5.4:** `A=-0,1` không đổi suốt 3 ngày, nhưng `Ā_t` (giá trị thật dùng để nhân `h(t)`) ra 3 số khác nhau: `0,928 / 0,883 / 0,930` — vì `Δ_t` khác nhau mỗi ngày. Ngày mưa to (`Δ_2` lớn nhất) → `Ā_2` nhỏ nhất → giữ lại ít trạng thái cũ hơn, hợp lý vì ngày mưa to nên nghiêng nhiều về thông tin mới.

#### 4.5.7 Bảng tổng kết — chọn lọc trực tiếp vs gián tiếp

| Tham số | Cách chọn lọc | Công thức |
|---|---|---|
| `B` | Trực tiếp | `B_t = w_B×x_t` |
| `C` | Trực tiếp | `C_t = w_C×x_t` |
| `Δ` | Trực tiếp | `Δ_t = softplus(w_Δ×x_t)` |
| `A` | Gián tiếp (qua `Δ_t`) | `Ā_t = 1+A×Δ_t` |

#### 4.5.8 Tóm lại

SSM cổ điển: bộ lọc cố định, áp dụng y hệt mọi input. Mamba (Selective): bộ lọc tự điều chỉnh theo từng input — input quan trọng thì được khuếch đại mạnh hơn hẳn (theo cấp số nhân qua `B_t`, và gián tiếp qua `A` nhờ `Δ_t`) — đúng khả năng LSTM có qua cổng, giờ SSM có thêm mà vẫn giữ cách tính nhanh.

---

### 4.6 Selective Scan — vì sao vẫn tính nhanh/song song được

#### 4.6.0 `h_0` là gì — định nghĩa trước khi dùng

`h_0` = **trạng thái khởi đầu, TRƯỚC KHI có input nào cả** — "trí nhớ trống, chưa biết gì" — đúng y hệt khái niệm `C_0=0, h_0=0` đã dùng ở LSTM (mục 1.4) và GRU (mục 3.4), quy ước bằng 0. Theo bảng quy đổi ký hiệu ở 4.3.1b (`h_(t-1) ≡ h(t)`, áp với `t=1`): `h_0 ≡ h(1)` — chính là giá trị đã dùng ở 4.5.4 (`h(1)=0`). Cùng 1 con số, chỉ khác cách viết.

#### 4.6.1 Vì sao SSM cổ điển (hệ số cố định) tính nhanh được — chứng minh mẹo "tích chập"

Công thức: `h_t = Ā·h_(t-1) + B̄·x_t`, `Ā, B̄` **cố định**. Bắt đầu `h_0=0` (xem 4.6.0). **Tự thay lần lượt vào nhau (thuần đại số):**
```
h_1 = Ā·h_0 + B̄·x_1 = B̄·x_1
h_2 = Ā·h_1 + B̄·x_2 = Ā·B̄·x_1 + B̄·x_2
h_3 = Ā·h_2 + B̄·x_3 = Ā²·B̄·x_1 + Ā·B̄·x_2 + B̄·x_3
```
**Quy luật:** `h_t = Σ (Ā^(t-k) × B̄ × x_k)`, k chạy 1→t. Đây chính là **định nghĩa tích chập (convolution)** — 1 kernel cố định `K[j]=Ā^j×B̄` áp dụng cho MỌI `h_t` (chỉ trượt theo t), giống hệt CNN dùng 1 filter cố định quét qua ảnh.

**Ví dụ số kiểm chứng — `Ā=0,9, B̄=0,5, x_1=2,x_2=3,x_3=4`:**

Cách tuần tự: `h_1=1, h_2=2,4, h_3=4,16`.

Cách kernel cố định `K[0]=0,5, K[1]=0,45, K[2]=0,405`:
```
h_3 = K[0]×x_3+K[1]×x_2+K[2]×x_1 = 0,5×4+0,45×3+0,405×2 = 2+1,35+0,81 = 4,16
```
Khớp đúng — chứng minh dùng **1 kernel cố định**, tính thẳng `h_3` **không cần biết `h_1,h_2` trước** → tính song song được mọi `h_t`.

#### 4.6.2 Vì sao Selective (Ā_t đổi theo t) làm hỏng mẹo này

Nếu `Ā` đổi theo từng bước, mở lại `h_3`:
```
h_3 = Ā_3·Ā_2·(B̄_1·x_1) + Ā_3·B̄_2·x_2 + B̄_3·x_3
```
Trọng số của `x_1` giờ là `Ā_3×Ā_2×B̄_1` — phụ thuộc **đúng chuỗi hệ số cụ thể ở giữa**, không còn "cách đây 2 bước luôn nhân `Ā²`" nữa. **Không tạo được 1 kernel cố định dùng chung cho mọi `h_t`** — mỗi vị trí cần tổ hợp trọng số riêng.

#### 4.6.3 Giải pháp — gộp 2 bước liên tiếp thành 1 bước duy nhất (Parallel/Selective Scan)

Dùng ký hiệu gọn `h_t = a_t·h_(t-1)+b_t`. **Thay bước 1 vào bước 2:**
```
h_2 = a_2·(a_1·h_0+b_1)+b_2 = (a_2·a_1)·h_0 + (a_2·b_1+b_2)
```
→ 2 bước gộp thành 1 bước, hệ số: `a_gộp=a_2×a_1`, `b_gộp=a_2×b_1+b_2`. **Điểm mấu chốt:** công thức gộp chỉ cần `a_1,b_1,a_2,b_2` — **không cần biết `h_0`** — tính trước được, độc lập.

**Ví dụ số — `a_1=0,9,b_1=1; a_2=0,8,b_2=2; a_3=0,7,b_3=0,5; h_0=0`:**

Tuần tự: `h_1=1, h_2=2,8, h_3=2,46`.

Gộp (2,3) trước, không cần biết `h_1`:
```
a_gộp(2,3)=0,7×0,8=0,56;  b_gộp(2,3)=0,7×2+0,5=1,9
Kiểm chứng: h_3=0,56×h_1+1,9=0,56×1+1,9=2,46 ✓
```

Gộp tiếp với bước 1 — ra 1 công thức DUY NHẤT cho cả 3 bước:
```
a_gộp(1,2,3)=0,56×0,9=0,504;  b_gộp(1,2,3)=0,56×1+1,9=2,46
Kiểm chứng: h_3=0,504×h_0+2,46=0,504×0+2,46=2,46 ✓
```

#### 4.6.3b Vì sao `a_gộp` vẫn quan trọng dù bị nhân với `h_0=0`

`h_0=0` (đúng quy ước, cố định, xem 4.6.0) khiến phép nhân CUỐI CÙNG (`a_gộp×h_0`) ra 0 — nhưng `a_gộp` **vẫn được dùng ngay trong lúc tính `b_gộp`** (phần không hề bị triệt tiêu, chính là kết quả cuối `h_t` khi `h_0=0`).

**Kiểm chứng bằng đúng số ở 4.6.3:** `b_gộp(1,2,3) = a_gộp(2,3) × b_1 + b_gộp(2,3) = 0,56×1+1,9 = 2,46`. Số `0,56` (chính là `a_gộp(2,3)`) **nằm ngay trong phép tính ra `2,46`** — nếu dùng sai `a` (VD lỡ dùng `a=1` thay vì `0,56`) sẽ ra `1×1+1,9=2,9` — sai. Vậy `a_gộp` không hề vô dụng — nó vẫn quyết định đúng `b_gộp`, chỉ là phần nhân trực tiếp với `h_0` mới bằng 0 do đúng ví dụ này bắt đầu từ ngày 1 (`h_0=0`).

#### 4.6.3c So sánh với LSTM/GRU — vì sao CHÚNG không dùng được mẹo gộp này (đã tra xác nhận, không chỉ suy luận riêng)

Nhắc lại mẹo gộp ở 4.6.3: `h_t=a_t·h_(t-1)+b_t`, gộp được vì `a_t, b_t` **tính trước được chỉ từ `x_t`, không cần biết `h_(t-1)`**.

**Ở LSTM, hệ số đóng vai trò `a_t` chính là `forget_gate`:** `forget_gate=sigmoid(W_f·[h_(t-1),x_t])` (đã học ở 1.3). Nhìn kỹ: `forget_gate` **cần biết `h_(t-1)` mới tính được** (không chỉ `x_t`) — mà `h_(t-1)` lại chính là thứ đang muốn "gộp trước khi biết". Đây là vòng lặp logic khiến LSTM **không thể tính trước hệ số**, không áp dụng được mẹo gộp/vòng. GRU tương tự: `z_t=sigmoid(W_z·[h_(t-1),x_t])` (3.3) — cũng cần `h_(t-1)`.

**Đã tra web xác nhận lại (không chỉ suy luận riêng):** *"vanilla GRU and LSTM cells include mixing among hidden state components, which causes their Jacobians to be dense in general"* — về mặt toán học, phép biến đổi trạng thái của LSTM/GRU là **phi tuyến/phức tạp** theo `h_(t-1)` (không đơn giản dạng affine `a×h_(t-1)+b` như Mamba) — không thỏa điều kiện cần cho phép "gộp kết hợp được" (associative, đã học ở 4.6.9 — Blelloch scan). **Đây là lý do gốc rễ** vì sao chỉ Mamba (và họ SSM tuyến tính) song song hóa được theo kiểu "vòng", còn LSTM/GRU thì không — dù công thức bề ngoài (`h_t` phụ thuộc `h_(t-1)`) trông khá giống nhau.

#### 4.6.4 "Vòng" (round) là gì — analogy giải đấu loại trực tiếp

8 đội bóng thi đấu loại trực tiếp, tìm 1 đội vô địch:
```
Vòng 1 (tứ kết):    Đội1-Đội2   Đội3-Đội4   Đội5-Đội6   Đội7-Đội8   ← 4 trận, đá CÙNG LÚC
Vòng 2 (bán kết):   Thắng(1,2)-Thắng(3,4)    Thắng(5,6)-Thắng(7,8)  ← 2 trận, đá CÙNG LÚC
Vòng 3 (chung kết): 2 đội thắng bán kết đấu nhau                    ← 1 trận
```
**"1 vòng" = 1 đợt, trong đó nhiều trận ĐỘC LẬP diễn ra cùng lúc** — nhưng phải xong HẾT 1 vòng mới sang vòng sau (vì vòng sau cần biết ai thắng vòng trước, không nhảy cóc được).

Phép "gộp" 2 công thức affine liền kề (đã học ở 4.6.3) có đúng tính chất này: gộp bước 1-2 và gộp bước 3-4 là **2 việc không liên quan nhau** → làm cùng lúc được (1 vòng). Nhưng gộp "khối 1-2" với "khối 3-4" thì **phải đợi cả 2 khối đó gộp xong trước** → là vòng kế tiếp.

#### 4.6.5 "Khối" và "mốc tham chiếu" — theo dõi cơ chế qua ví dụ 4 bước

Thêm 1 bước vào ví dụ cũ: `a_4=0,6, b_4=1,5`. Tuần tự: `h_1=1, h_2=2,8, h_3=2,46, h_4=0,6×2,46+1,5=2,976`.

Gọi **"khối"** = 1 nhóm bước liên tiếp đã gộp chung thành 1 công thức; **"mốc tham chiếu"** = trạng thái NGAY TRƯỚC bước đầu tiên của khối đó (khối đang viết công thức "so với" trạng thái nào).

```
Trước khi gộp — mỗi vị trí là 1 khối riêng, chỉ chứa đúng 1 bước:
  Khối {2}: h_2 = 0,8×h_1+2        ← mốc tham chiếu = h_1
  Khối {4}: h_4 = 0,6×h_3+1,5      ← mốc tham chiếu = h_3

Vòng 1 — gộp khối{1} với {2} → khối{1,2}; gộp khối{3} với {4} → khối{3,4} (2 phép ĐỘC LẬP, cùng lúc):
  Khối{1,2}: h_2 = 0,72×h_0+2,8    ← mốc LÙI từ h_1 sang h_0 (0,8×0,9=0,72; 0,8×1+2=2,8)
  Khối{3,4}: h_4 = 0,42×h_2+1,8    ← mốc LÙI từ h_3 sang h_2 (0,6×0,7=0,42; 0,6×0,5+1,5=1,8)

Vòng 2 — gộp khối{1,2} với {3,4} → khối{1,2,3,4}:
  h_4 = 0,3024×h_0+2,976           ← mốc LÙI từ h_2 sang h_0 (0,42×0,72=0,3024; 0,42×2,8+1,8=2,976)
```
Cắm `h_0=0`: `h_4=0,3024×0+2,976=2,976` ✓ khớp tuần tự.

#### 4.6.6 Quy luật tổng quát — 2 cách nhìn cùng 1 hiện tượng

- **Số khối giảm 1 nửa mỗi vòng:** `4 khối → 2 khối → 1 khối`.
- **Mốc tham chiếu lùi xa gấp đôi mỗi vòng:** khối chứa `1 bước → 2 bước → 4 bước`.

Đây là 1 hiện tượng nhìn từ 2 góc: số khối giảm 1 nửa ⟺ mỗi khối còn lại to gấp đôi ⟺ mốc tham chiếu lùi xa gấp đôi. **Số vòng cần = số lần chia đôi được cho tới khi còn 1 khối = `⌈log₂(n)⌉`.** Chuỗi 3-4 bước cần 2 vòng; chuỗi 1000 bước chỉ cần khoảng 10 vòng (thay vì 1000 bước chờ tuần tự).

#### 4.6.7 Mở rộng — tính ĐỦ mọi `h_t`, không chỉ 1 điểm cuối (ví dụ đầy đủ 6 bước)

Ví dụ 4.6.5 chỉ cho ra `h_4` (điểm cuối cùng) — nhưng Mamba cần **TẤT CẢ** `h_1,...,h_n` (vì `y_t=C_t·h_t` phải tính ở MỌI vị trí, không chỉ vị trí cuối). Cách làm: thay vì chỉ ghép cặp "khối chẵn", cho **MỌI vị trí** đều tự gộp với vị trí cách nó `d` bước mỗi vòng (`d` tăng gấp đôi mỗi vòng: `1,2,4,...`) — kỹ thuật cụ thể này gọi là **Hillis-Steele scan**.

**Mở rộng ví dụ ra 6 bước** (thêm `a_5=0,85,b_5=0,8`; `a_6=0,75,b_6=1,0`). Tuần tự (để đối chiếu): `h_1=1, h_2=2,8, h_3=2,46, h_4=2,976, h_5=0,85×2,976+0,8=3,3296, h_6=0,75×3,3296+1,0=3,4972`.

**Vòng 1 (d=1) — mọi vị trí gộp với vị trí liền trước, cùng lúc:**
| Vị trí | Gộp với | a mới | b mới |
|---|---|---|---|
| 1 | (không có) | 0,9 | 1 |
| 2 | 1 | 0,8×0,9=0,72 | 0,8×1+2=2,8 |
| 3 | 2 | 0,7×0,8=0,56 | 0,7×2+0,5=1,9 |
| 4 | 3 | 0,6×0,7=0,42 | 0,6×0,5+1,5=1,8 |
| 5 | 4 | 0,85×0,6=0,51 | 0,85×1,5+0,8=2,075 |
| 6 | 5 | 0,75×0,85=0,6375 | 0,75×0,8+1,0=1,6 |

**Vòng 2 (d=2) — mọi vị trí ≥3 gộp với vị trí cách 2, cùng lúc:**
| Vị trí | Gộp với | a mới | b mới |
|---|---|---|---|
| 1, 2 | giữ nguyên | — | — |
| 3 | 1 | 0,56×0,9=0,504 | 0,56×1+1,9=2,46 |
| 4 | 2 | 0,42×0,72=0,3024 | 0,42×2,8+1,8=2,976 |
| 5 | 3 | 0,51×0,56=0,2856 | 0,51×1,9+2,075=3,044 |
| 6 | 4 | 0,6375×0,42=0,26775 | 0,6375×1,8+1,6=2,7475 |

**Vòng 3 (d=4) — mọi vị trí ≥5 gộp với vị trí cách 4, cùng lúc:**
| Vị trí | Gộp với | a mới | b mới |
|---|---|---|---|
| 1-4 | giữ nguyên | — | — |
| 5 | 1 | 0,2856×0,9=0,25704 | 0,2856×1+3,044=3,3296 |
| 6 | 2 | 0,26775×0,72=0,19278 | 0,26775×2,8+2,7475=3,4972 |

**Cắm `h_0=0` vào cả 6 vị trí cùng lúc:**
```
h_1=1   h_2=2,8   h_3=2,46   h_4=2,976   h_5=3,3296   h_6=3,4972
```
✓ Khớp đúng 100% kết quả tuần tự — chỉ tốn **3 vòng** (`⌈log₂6⌉=3`) để ra ĐỦ cả 6 giá trị cùng lúc, thay vì 6 bước chờ nhau tuần tự.

#### 4.6.8 Liên hệ cài đặt thực tế

Cài đặt bằng **CUDA kernel chuyên dụng** (`selective_scan_cuda`, `causal_conv1d_cuda` — đúng tên đã ghi trong `Document/01_Plan/01_OverallPlan.md` Mục 8, lý do bắt buộc cần GPU) — tối ưu phần cứng: giữ tính toán trong bộ nhớ nhanh trên chip (SRAM) thay vì đọc/ghi liên tục ra bộ nhớ GPU chậm hơn (HBM) — lý do Mamba chạy nhanh thật trên GPU, không chỉ nhanh trên giấy.

#### 4.6.9 Kiểm tra lại và nguồn

- Đã tự tính tay toàn bộ số liệu ở 4.6.1, 4.6.3, 4.6.5, 4.6.7 (cả tuần tự lẫn gộp theo vòng) — khớp đúng nhau 100%, không có lỗi tính toán.
- **Đã tra web xác nhận lại nguồn gốc kỹ thuật** (không chỉ suy luận riêng): kỹ thuật "parallel scan" Mamba dùng chính là **Blelloch parallel scan (1990)** — nguyên văn tìm được: *"Mamba can be computed in parallel via the Blelloch parallel scan algorithm... enables efficient computation of all prefix values of a sequence, obtained by repeatedly applying an associative binary operator... arrays storing all hidden states can be computed in parallel in O(log t) time."* — khớp đúng những gì đã học ở đây (tính chất kết hợp được/associative, số vòng theo log). Nguồn: paper gốc "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arxiv 2312.00752) và các bài phân tích kỹ thuật liên quan.
- Kỹ thuật cụ thể minh họa ở 4.6.7 (mọi vị trí tự cập nhật mỗi vòng) là dạng "Hillis-Steele scan" — 1 cách cài đặt cụ thể của họ thuật toán parallel/associative scan nói chung mà Mamba dùng.

---

### 4.7 Bidirectional Mamba — cơ chế đặc thù RiverMamba

**Nguồn:** `arxiv.org/html/2505.22535v3` (bản đầy đủ RiverMamba, có appendix) — đọc trực tiếp, trích nguyên văn.

#### 4.7.1 Vấn đề — vì sao Mamba gốc (1 chiều) không đủ cho bài toán này

Mamba gốc tính `h_t = Ā_t·h_(t-1) + B̄_t·x_t` — trạng thái tại bước `t` chỉ phụ thuộc **quá khứ** (`h_(t-1)` và các bước trước), không bao giờ nhìn thấy bước sau. Đây gọi là tính chất **causal** (nhân quả) — đúng bản chất cho dữ liệu **thời gian thật** (hôm nay không thể biết mai mưa bao nhiêu).

Nhưng RiverMamba còn xử lý theo **KHÔNG GIAN** (hàng triệu điểm sông trải khắp bản đồ, ghép thành 1 chuỗi qua "space-filling curve" — học ở 4.9). Thứ tự trong chuỗi không gian này **không phải thời gian thật** — chỉ là thứ tự sắp xếp. Nếu chỉ quét 1 chiều (causal), điểm sông ở cuối chuỗi sẽ không bao giờ "biết" thông tin từ điểm ở đầu chuỗi, dù 2 điểm đó có thể **ở sát nhau ngoài đời thực** (chỉ bị đẩy xa nhau do cách sắp xếp thành chuỗi 1D). → Cần nhìn **cả 2 hướng**, không có lý do giữ tính "nhân quả" khi thứ tự không phải thời gian.

#### 4.7.1b Ví dụ cụ thể — lưới 4×4 điểm sông, thấy rõ vấn đề "bị đẩy xa nhau"

Giả lập 1 mảnh nhỏ bản đồ, 16 điểm sông xếp lưới:
```
(1,1) (1,2) (1,3) (1,4)
(2,1) (2,2) (2,3) (2,4)
(3,1) (3,2) (3,3) (3,4)
(4,1) (4,2) (4,3) (4,4)
```
Điểm `(1,4)` và `(2,4)` **sát nhau ngoài đời thực** (cùng cột, khác hàng liền kề — cách nhau đúng 1 ô lưới, VD 0,05°).

**Dùng "Sweep curve" (kiểu đơn giản nhất — quét từng hàng, trái→phải, hết hàng xuống hàng dưới, giống đọc sách) để xếp thành chuỗi 1D:**
```
Vị trí 1=(1,1)  Vị trí 2=(1,2)  Vị trí 3=(1,3)  Vị trí 4=(1,4)
Vị trí 5=(2,1)  Vị trí 6=(2,2)  Vị trí 7=(2,3)  Vị trí 8=(2,4)
```
`(1,4)` rơi vào **vị trí 4**, `(2,4)` rơi vào **vị trí 8** — cách nhau **4 vị trí** trong chuỗi (bị chen bởi `(2,1),(2,2),(2,3)` — 3 điểm không liên quan gì về sông ngòi, chỉ tình cờ cùng hàng).

**Đây là lỗi hệ thống, không phải ca đặc biệt:** MỌI cặp điểm thẳng cột, khác hàng liền kề đều cách nhau đúng bằng độ rộng hàng (ở đây =4). Lưới càng rộng (VD 1000 cột thay vì 4) thì khoảng cách trong chuỗi càng lớn dù ngoài đời vẫn chỉ cách 1 ô.

**Nếu Mamba chỉ quét 1 chiều (causal):** vị trí 8 (`(2,4)`) CÓ THỂ nhận thông tin từ vị trí 4 (`(1,4)`, vì 4 đứng "trước" 8 trong chuỗi) — nhưng vị trí 4 **KHÔNG BAO GIỜ** nhận được thông tin từ vị trí 8 (vì 8 đứng "sau" 4). Trong khi thực tế, `(1,4)` và `(2,4)` là hàng xóm sông thật — ảnh hưởng qua lại 2 chiều, không có "chiều nào xảy ra trước" như thời gian. Đây chính là lỗi Bidirectional Mamba phải vá.

#### 4.7.2 Cơ chế — trích nguyên văn paper

*"We use a bi-directional approach that converts **x** into **x'_o** using a forward and a backward 1-D causal convolution, where `o∈{f,b}` denotes the forward or backward pass."*

Giải nghĩa:
- `x` = chuỗi input (đã sắp xếp theo space-filling curve).
- `o∈{f,b}` — chạy **2 lần riêng biệt**: 1 lần xuôi (`f`, thứ tự 1→2→...→n), 1 lần ngược (`b`, thứ tự n→n-1→...→1).
- Mỗi lượt (xuôi/ngược) là **1 khối Mamba đầy đủ riêng** (conv1d riêng, `Ā,B̄,C` riêng tự học) — không share tính toán, chỉ chung kiến trúc, khác trọng số.

**Kết hợp lại — trích nguyên văn:** *"The final output **y** is obtained by gating **y_forward** and **y_backward** via SiLU(**z**) and adding them up."*

- `y_forward` = kết quả quét xuôi (mỗi điểm chỉ biết các điểm "trước" nó trong chuỗi).
- `y_backward` = kết quả quét ngược (mỗi điểm chỉ biết các điểm "sau" nó trong chuỗi).
- `z` = nhánh cổng có sẵn trong khối Mamba gốc (2 nhánh song song: 1 nhánh qua SSM ra `y`, 1 nhánh `z` riêng qua Linear rồi SiLU — giống vai trò "cổng output" của LSTM).
- `SiLU(x) = x×sigmoid(x)` — hàm kích hoạt mượt (không góc gãy cứng như ReLU).
- **Công thức tổng:** `y = SiLU(z)⊙y_forward + SiLU(z)⊙y_backward` (`⊙` = nhân từng phần tử) — mỗi điểm sông cuối cùng có thông tin từ **cả 2 hướng**, đã qua "cổng" lọc lại trước khi cộng.

#### 4.7.2b Ví dụ số đầy đủ — 4 điểm sông, thấy rõ output dựa trên gì

4 điểm theo đúng thứ tự chuỗi, input (đã gộp đặc trưng thành 1 số để dễ tính tay — thực tế là cả vector):
```
Vị trí:  1    2    3    4
Input x: 2    5    1    4
```
Forward và Backward là **2 khối HOÀN TOÀN RIÊNG** (trọng số khác nhau, đúng 4.7.2). Dùng bản SSM đơn giản để dễ theo dõi: `Āf=0,8, B̄f=0,5, Cf=0,9` (xuôi); `Āb=0,7, B̄b=0,4, Cb=1,1` (ngược).

**Quét XUÔI (1→2→3→4), `h_0^f=0`** (vị trí 1 không có gì "phía trước" nó):
```
h_1^f=0,8×0+0,5×2=1,0        h_2^f=0,8×1,0+0,5×5=3,3
h_3^f=0,8×3,3+0,5×1=3,14     h_4^f=0,8×3,14+0,5×4=4,512
y_forward = Cf×h^f:  0,9 / 2,97 / 2,826 / 4,0608
```
`y_forward_1=0,9` **CHỈ dựa vào `x_1`** — vì `h_0^f=0`, vị trí 1 "mù" hoàn toàn với vị trí 2,3,4.

**Quét NGƯỢC (4→3→2→1), `h_5^b=0`** (vị trí 4 không có gì "phía sau" nó):
```
h_4^b=0,7×0+0,4×4=1,6        h_3^b=0,7×1,6+0,4×1=1,52
h_2^b=0,7×1,52+0,4×5=3,064   h_1^b=0,7×3,064+0,4×2=2,9448
y_backward = Cb×h^b:  3,23928 / 3,3704 / 1,672 / 1,76   (theo thứ tự vị trí 1,2,3,4)
```
`y_backward_1=3,239` — con số này được tính từ chuỗi BẮT ĐẦU ở vị trí 4 (dùng `x_4`), truyền qua `x_3`, `x_2`, rồi mới tới vị trí 1 → `h_1^b` đã "gom" thông tin từ CẢ `x_2,x_3,x_4`.

**Cổng `z_t=w_z×x_t` (`w_z=0,3`), `SiLU(z)=z×sigmoid(z)`:**
```
z: 0,6 / 1,5 / 0,3 / 1,2
SiLU(z): 0,3874 / 1,2264 / 0,1723 / 0,9222
```

**Ghép cuối: `y_t = SiLU(z_t)×(y_forward_t+y_backward_t)`:**
```
y_1 = 0,3874×(0,9+3,239)   = 1,60
y_2 = 1,2264×(2,97+3,370)  = 7,78
y_3 = 0,1723×(2,826+1,672) = 0,78
y_4 = 0,9222×(4,061+1,76)  = 5,37
```

**So sánh — đúng điểm mấu chốt:** vị trí 1 nếu CHỈ quét xuôi thì output `=0,9` (chỉ biết `x_1`). Sau khi có cả 2 chiều, output cuối `=1,60` — đã "hấp thụ" thêm thông tin từ `x_2,x_3,x_4` qua đường quét ngược. Đây chính là cách vá lỗi `(1,4)`/`(2,4)` ở 4.7.1b — điểm đứng đầu chuỗi (dễ "mù" thông tin phía sau nếu chỉ quét xuôi) giờ vẫn nhận được thông tin từ các điểm phía sau nhờ nhánh backward.

**Lưu ý:** `y` ở đây (dù xuôi, ngược, hay ghép) đều là **vector đặc trưng trung gian** (giống `h_t` — trừu tượng, không đo được), KHÔNG phải số dự báo m³/s cuối cùng. Nếu còn khối Mamba khác phía sau (paper dùng nhiều khối xếp chồng, mỗi khối 1 curve khác nhau) thì `y` này lại làm input cho khối tiếp; nếu là lớp cuối thì mới qua 1 lớp tuyến tính cuối (giống `a×h_t+b` ở LSTM 1.5) để ra số m³/s.

#### 4.7.3 Dùng ở đâu trong RiverMamba

Đã xác nhận từ paper: **cả khối Hindcast (xử lý quá khứ: GloFAS, ERA5-Land, CPC, static features) VÀ khối Forecast (thêm HRES) đều dùng Bidirectional Mamba** — không phải chỉ 1 khối, vì cả 2 đều xử lý chuỗi không gian (mạng lưới sông), lý do ở 4.7.1 áp dụng cho cả 2.

#### 4.7.4 Đối chiếu lại LSTM đã học

LSTM cũng có biến thể "Bidirectional LSTM" (BiLSTM) — ý tưởng y hệt: chạy 1 LSTM xuôi + 1 LSTM ngược, ghép kết quả. RiverMamba áp dụng đúng ý tưởng đó vào Mamba, chỉ khác cách ghép: BiLSTM thường ghép bằng concatenate đơn giản (nối 2 vector lại), RiverMamba ghép bằng **gating** `SiLU(z)` — cho phép model tự học "nên tin bao nhiêu % vào mỗi hướng" thay vì luôn cộng/nối cố định.

#### 4.7.5 Kiểm tra lại

- Toàn bộ nội dung 4.7.2/4.7.3 trích nguyên văn từ bản đầy đủ paper (`arxiv.org/html/2505.22535v3`, có appendix) — không suy đoán.
- Ký hiệu `x'_o`, `o∈{f,b}`, `y_forward/y_backward`, `SiLU(z)` giữ nguyên đúng tên paper dùng, không đổi ký hiệu riêng để tránh sai lệch khi đối chiếu lại paper sau này.
- Đã tự tính tay lại độc lập toàn bộ số liệu ở 4.7.2b (quét xuôi, quét ngược, `SiLU(z)`, ghép cuối) — khớp đúng, không có lỗi. Ví dụ 4.7.1b (lưới 4×4, Sweep curve) và 4.7.2b (4 điểm, forward/backward) là ví dụ tự xây dựng để minh họa cơ chế — không phải số liệu lấy trực tiếp từ paper (paper không công bố trọng số/số liệu cụ thể).

---

### 4.8 LOAN (Location-Aware Adaptive Normalization)

**Nguồn:** `arxiv.org/html/2505.22535v3` (Equation 2, Section 3, Algorithm 1 Appendix D, Table 2b) — trích nguyên văn, không suy đoán trừ khi ghi rõ.

#### 4.8.1 Nhắc lại — "chuẩn hóa" (normalization) là gì, vì sao mạng neural cần nó

Dữ liệu đầu vào có thang đo rất khác nhau (mưa 0-200mm, độ cao 0-3000m, diện tích lưu vực hàng nghìn km²) — đưa thẳng vào model sẽ làm kênh có số lớn "át" hẳn kênh có số nhỏ. **Chuẩn hóa** = đưa mọi kênh về cùng 1 thang đo chung, thường bằng `(X-μ)/σ` (trừ trung bình, chia độ lệch chuẩn) — làm mọi kênh có cùng "tầm cỡ" số liệu, dễ học hơn.

#### 4.8.2 Vấn đề — chuẩn hóa thường XÓA MẤT đặc điểm riêng của từng điểm sông

Tưởng tượng 2 điểm sông: **Điểm A** — vùng núi dốc (thượng nguồn), lưu vực nhỏ, nước lên/rút rất nhanh. **Điểm B** — vùng đồng bằng (gần cửa sông), lưu vực rất lớn, nước lên/rút chậm, kéo dài. Cùng 1 lượng mưa, phản ứng dòng chảy 2 điểm này khác hẳn ngoài đời.

**Nhưng chuẩn hóa thường (`(X-μ)/σ`, tính riêng từng điểm/ngày) không hề biết điểm đó là A hay B** — chỉ nhìn vào chính dãy số của ngày đó rồi trừ-chia. Nếu A và B tình cờ có cùng lượng mưa hôm đó, sau chuẩn hóa dữ liệu của chúng **trông giống hệt nhau** — mất sạch thông tin "đây là điểm nào, địa hình ra sao", dù đó là thông tin quan trọng để dự báo đúng.

#### 4.8.3 Ý tưởng LOAN — giữ nguyên chuẩn hóa, cộng thêm "thẻ tên" riêng cho từng điểm

LOAN vẫn làm đúng `(X-μ)/σ` — KHÔNG bỏ bước này. Nhưng **cộng thêm** 1 lượng riêng cho từng điểm, tính từ **đặc trưng tĩnh** của chính điểm đó (99 biến LISFLOOD: độ cao, độ dốc, diện tích lưu vực... — không đổi theo ngày, chỉ phụ thuộc "đây là điểm nào"). Sau LOAN, dữ liệu của A và B dù mưa giống hệt nhau vẫn sẽ khác nhau — model không còn "mù" về địa hình.

#### 4.8.4 Công thức đầy đủ (Equation 2, trích nguyên văn)

```
LOAN(X) = (X−μ)/σ + GELU(Linear(X_static))
```

**Giải nghĩa ký hiệu (đã tra đúng shape từ paper):**
- `X` — tensor đặc trưng ĐỘNG (mưa, ERA5...), kích thước `(B,T,P,K)`: B=batch, T=số ngày, P=số điểm sông, K=số kênh (K=192 thật).
- `μ, σ` — trung bình/độ lệch chuẩn, tính dọc theo **chiều K** (giống LayerNorm chuẩn — riêng cho từng điểm, từng ngày).
- `X_static` — đặc trưng TĨNH của từng điểm (99 biến LISFLOOD), kích thước `(B,1,P,V_s)` — **không có chiều T** (vì static, không đổi theo ngày).
- `Linear` — 1 lớp tuyến tính **CHUNG** (cùng trọng số cho mọi điểm), chiếu `X_static` (V_s chiều) → K chiều, rồi nhân bản theo thời gian để khớp shape với `X`.
- `GELU(x)=x×Φ(x)` (`Φ`=hàm phân phối chuẩn tích lũy, không tính tay chính xác được) — bản xấp xỉ tính tay: `GELU(x)≈x×sigmoid(1,702x)`.

#### 4.8.5 Ví dụ số đầy đủ — 2 điểm A (núi) và B (đồng bằng), CÙNG lượng mưa hôm nay (rút gọn K=4, V_s=2 để tính tay)

**Giả sử `X` giống hệt nhau cho cả A và B** (tình cờ cùng lượng mưa) → sau chuẩn hóa, CẢ 2 RA CÙNG 1 SỐ:
```
X=[10,20,15,25] → μ=17,5, σ=5,59 → (X-μ)/σ = [-1,342, 0,447, -0,447, 1,342]   (giống nhau cho cả A và B)
```

**Nhưng đặc trưng tĩnh của A và B khác nhau:**
```
Điểm A (núi dốc, lưu vực nhỏ):    X_static_A = [0,3 ; 0,1]   (độ dốc cao, diện tích nhỏ)
Điểm B (đồng bằng, lưu vực lớn):  X_static_B = [0,05 ; 0,9]  (độ dốc thấp, diện tích lớn)
```

**Cùng 1 lớp `Linear` (trọng số CHUNG `W`, áp cho cả 2 điểm):**
```
W = [[0,5,0,2,-0,1,0,3],[0,1,-0,2,0,4,0,2]]

Linear(X_static_A) = [0,3×0,5+0,1×0,1, 0,3×0,2+0,1×(-0,2), 0,3×(-0,1)+0,1×0,4, 0,3×0,3+0,1×0,2] = [0,16, 0,04, 0,01, 0,11]
Linear(X_static_B) = [0,05×0,5+0,9×0,1, 0,05×0,2+0,9×(-0,2), 0,05×(-0,1)+0,9×0,4, 0,05×0,3+0,9×0,2] = [0,115, -0,17, 0,355, 0,195]
```

**Qua GELU (xấp xỉ `x×sigmoid(1,702x)`):**
```
GELU(Linear_A) = [0,0908, 0,0207, 0,0050, 0,0601]
GELU(Linear_B) = [0,0631, -0,0728, 0,2295, 0,1135]
```

**LOAN cuối cùng — cộng vào đúng phần chuẩn hóa (giống nhau) ở trên:**
```
LOAN(A) = [-1,342,0,447,-0,447,1,342] + [0,0908,0,0207,0,0050,0,0601]  = [-1,251, 0,468, -0,442, 1,402]
LOAN(B) = [-1,342,0,447,-0,447,1,342] + [0,0631,-0,0728,0,2295,0,1135] = [-1,279, 0,374, -0,218, 1,456]
```

**Điểm mấu chốt:** `LOAN(A)` và `LOAN(B)` là **2 vector khác nhau rõ rệt** (VD kênh 3: `-0,442` vs `-0,218`), **dù đầu vào động `X` giống hệt nhau 100%** — bằng chứng số học cho thấy LOAN khôi phục lại được sự khác biệt giữa điểm núi và điểm đồng bằng mà chuẩn hóa thường đã xóa mất.

#### 4.8.6 Chỉnh lại chính xác — KHÔNG PHẢI "mỗi điểm 1 LOAN riêng", mà CÙNG 1 công thức, khác INPUT

Công thức LOAN và ma trận `W` — **CHỈ CÓ 1 BỘ DUY NHẤT, DÙNG CHUNG CHO MỌI ĐIỂM SÔNG** (không phải điểm A có `W_A` riêng, điểm B có `W_B` riêng). Khác nhau giữa các điểm chỉ là **INPUT** đưa vào (`X` và `X_static` riêng của từng điểm) — đưa vào ĐÚNG 1 công thức chung.

**Đối chiếu nguyên tắc đã học ở Transformer (2.7):** *"W_1, W_2 KHÔNG phải riêng của ngày B — là 1 bộ quy tắc chung duy nhất, dùng cho MỌI ngày. Cái riêng của ngày B chỉ là INPUT đưa vào."* — LOAN đúng y hệt nguyên tắc này, chỉ đổi "ngày" thành "điểm sông": **1 LOAN duy nhất, chạy hàng triệu lần (1 lần/điểm/ngày)**, mỗi lần nhận input khác nhau nên cho kết quả khác nhau — không phải hàng triệu LOAN riêng biệt.

#### 4.8.7 Dùng ở đâu trong kiến trúc

Đã tra Algorithm 1 (Appendix D): LOAN xuất hiện **2 LẦN mỗi khối Hindcast** — 1 lần ngay sau serialization (trước Mamba), 1 lần sau khối Mamba (trước MLP). Với 3 lớp Hindcast → LOAN chạy **6 lần** trong encoding. Khối Forecast cũng dùng LOAN (nhắc ở caption Figure 4, không chi tiết trong bài chính).

#### 4.8.8 Vì sao GELU — nói thật: paper KHÔNG giải thích lý do chọn

Đã tra kỹ — paper không đưa lý do so sánh vì sao chọn GELU thay vì hàm khác, chỉ dùng thẳng. Có ablation về hàm kích hoạt (Table 2b, nhắc ở Section F.6) nhưng số liệu nằm ở phụ lục, chưa truy cập được.

#### 4.8.9 Vì sao CỘNG (không nhân) — suy luận riêng, KHÔNG phải trích paper

*Lưu ý:* paper không giải thích trực tiếp — suy luận từ cấu trúc công thức: cộng thêm (thay vì nhân) nghĩa là mỗi điểm có 1 "độ lệch nền" (baseline offset) riêng, không phụ thuộc độ lớn giá trị đã chuẩn hóa — giống vai trò bias trong hồi quy tuyến tính. Nếu nhân, độ lệch sẽ tỷ lệ với giá trị gốc — không hợp lý bằng cộng khi muốn biểu diễn "đặc điểm địa hình luôn dịch lưu lượng lên/xuống 1 lượng cố định", bất kể giá trị đo đang cao hay thấp.

#### 4.8.10 So với LayerNorm/BatchNorm đã biết

| | Tính `μ,σ` theo chiều nào | Có thêm gì khác |
|---|---|---|
| LayerNorm (chuẩn) | Theo K (kênh), riêng từng sample | Không |
| BatchNorm (chuẩn) | Theo Batch, riêng từng kênh | Không |
| LOAN | Theo K (giống LayerNorm) | **CÓ THÊM** độ lệch từ đặc trưng tĩnh địa điểm |

→ LOAN = LayerNorm chuẩn + thêm 1 "địa chỉ nhà" riêng cho từng điểm sông.

#### 4.8.11 Kết quả ablation (Table 2b, "Location Embedding") — nói thật cả kết quả lạ

| Cấu hình | KGE | F1 |
|---|---|---|
| Chỉ nối thẳng static features (không LOAN) | 0,9183 | 0,2790 |
| LOAN chỉ ở Hindcast | **0,8862** (thấp hơn cả không dùng!) | 0,2030 |
| LOAN chỉ ở Forecast | 0,9136 | 0,2593 |
| LOAN ở CẢ 2 (bản paper chọn) | **0,9205** (cao nhất) | **0,2875** (cao nhất) |

**Nhận xét:** dùng LOAN CHỈ ở Hindcast thực ra **tệ hơn** không dùng LOAN gì cả (0,8862 vs 0,9183) — chỉ khi dùng ở CẢ 2 khối mới tốt hơn baseline. Paper không giải thích sâu vì sao, chỉ kết luận dùng cả 2 "balances the metrics".

#### 4.8.12 Kiểm tra lại và nguồn

- Toàn bộ công thức, shape, trích dẫn Table 2b lấy từ `arxiv.org/html/2505.22535v3` (Equation 2, Algorithm 1 Appendix D) — không suy đoán.
- Đã tự tính tay lại độc lập toàn bộ số liệu ở 4.8.5 (chuẩn hóa, Linear, GELU xấp xỉ, cộng cuối cho cả A và B) — khớp đúng, không lỗi.
- Ví dụ "điểm A/điểm B" và số liệu (`X`, `X_static`, `W`) là tự xây dựng để minh họa cơ chế — không phải số liệu thật từ paper (paper không công bố trọng số/ví dụ số cụ thể).
- Mục 4.8.8 (vì sao GELU) và 4.8.9 (vì sao cộng) đã ghi rõ ràng: mục trước là thành thật paper không giải thích, mục sau là suy luận riêng — không lẫn lộn 2 loại thông tin này.

---

### 4.9 Space-filling curve serialization — chuyển mạng lưới sông (2D) thành chuỗi (1D) cho Mamba

**Nguồn:** `arxiv.org/html/2505.22535v3` (trích nguyên văn) + `jakubcerveny.github.io/gilbert` (thuật toán Gilbert curve gốc, tác giả chính chủ) — đã tra kỹ, không suy đoán.

#### 4.9.1 Vấn đề — vì sao cần serialization

Mamba chỉ xử lý được **chuỗi 1D** (thứ tự rõ ràng bước 1→2→3..., `h_t` phụ thuộc `h_(t-1)`). Nhưng mạng lưới sông là dữ liệu **không gian 2D** (hàng triệu điểm trải trên bản đồ, không có "thứ tự" tự nhiên). Cần 1 quy tắc "duỗi thẳng" lưới 2D thành 1 hàng 1D — gọi là **serialization**.

**Ví dụ lưới nhỏ dùng xuyên suốt phần này** (16 điểm sông, giá trị = lượng mưa mm, chỉ 1 biến để dễ tính tay — thực tế mỗi điểm có ~136 biến):
```
(1,1)=3  (1,2)=7  (1,3)=2  (1,4)=9
(2,1)=5  (2,2)=4  (2,3)=6  (2,4)=8
(3,1)=1  (3,2)=3  (3,3)=5  (3,4)=2
(4,1)=4  (4,2)=6  (4,3)=1  (4,4)=3
```

#### 4.9.2 Vì sao không được xếp bừa (ngẫu nhiên)

Mamba coi **2 vị trí liền chuỗi là "có liên quan mật thiết"** (nối trực tiếp qua `h_t=Ā·h_(t-1)+B̄·x_t`). Nếu xếp bừa (VD `Vị trí 1=(1,1), Vị trí 2=(3,4)...`), 2 vị trí liền chuỗi có thể là 2 điểm cách xa nhau cả bản đồ (`(1,1)` và `(3,4)` cách 2 hàng+3 cột) — Mamba lãng phí công học mối liên hệ vô nghĩa, trong khi 2 điểm sát nhau ngoài đời (VD `(1,1)`-`(1,2)`) có thể bị xếp cách xa trong chuỗi. → Cần quy tắc xếp **có chủ đích**, giữ tính chất **locality**: gần ngoài đời → gần trong chuỗi.

#### 4.9.3 Hàm song ánh `Φ:ℤ³→ℕ`

`Φ` nhận vào **3 con số**, trả ra **1 số vị trí duy nhất**. Đã tra xác nhận: 3 con số input là **(hàng, cột, NGÀY)** — không gian + thời gian gộp chung (paper gọi là *"spatio-temporal scanning path"*), không phải x,y,z không gian tĩnh.

**Ví dụ công thức Sweep-ngang** (16 điểm/ngày): `Φ(hàng,cột,ngày) = (ngày-1)×16 + (hàng-1)×4 + cột`
```
Φ(1,4,1) = 0+0+4 = 4      Φ(2,4,1) = 0+4+4 = 8      Φ(1,1,2) = 16+0+1 = 17
```

#### 4.9.4 Sweep curve — ngang và dọc, ví dụ số đầy đủ

**Sweep ngang** (quét trái→phải từng hàng, giống đọc sách):
```
Vị trí:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
Dãy x:   3  7  2  9  5  4  6  8  1  3  5  2  4  6  1  3
```

**Sweep dọc** (quét từng cột, trên xuống dưới):
```
Vị trí:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
Dãy x:   3  5  1  4  7  4  3  6  2  6  5  1  9  8  2  3
```

#### 4.9.5 2 loại lỗi của Sweep — ví dụ số cụ thể (dùng đúng Sweep ngang)

**Lỗi A — điểm gần ngoài đời bị xếp XA trong chuỗi:** `(1,4)` và `(2,4)` — sát nhau ngoài đời (cùng cột, hàng liền kề, cách đúng 1 ô). Trong chuỗi Sweep ngang: `(1,4)`=vị trí 4, `(2,4)`=vị trí 8 → **cách 4 vị trí**, bị 3 điểm không liên quan chen giữa.

**Lỗi B — điểm liền chuỗi lại XA ngoài đời (ngược lại Lỗi A):** vị trí 4=`(1,4)` và vị trí 5=`(2,1)` — **liền nhau trong chuỗi** (Mamba nối trực tiếp). Khoảng cách thật (Manhattan): `|1-2|+|4-1|=1+3=4 ô` — gấp 4 lần hàng xóm thật, dù Mamba xử lý như đang liền kề.

#### 4.9.6 Đổi hướng Sweep — lỗi KHÔNG biến mất, chỉ chuyển sang cặp khác

Với Sweep dọc (4.9.4): `(1,4)`=vị trí 13, `(2,4)`=vị trí 14 → **cách đúng 1, Lỗi A của cặp này hết!** Nhưng cặp khác lại lỗi: `(1,3)`(vị trí 9) và `(1,4)`(vị trí 13) — sát nhau ngoài đời nhưng giờ cách 4 vị trí. **Kết luận:** không có hướng Sweep nào hoàn hảo cho MỌI cặp cùng lúc — chỉ "chuyển lỗi từ cặp này sang cặp khác".

#### 4.9.7 Cần curve KHÔNG BAO GIỜ mắc Lỗi B — Hilbert/Gilbert curve

Lỗi B nghiêm trọng hơn Lỗi A (khiến Mamba **tin nhầm** 2 điểm không liên quan là liên quan). Cần cách xếp đảm bảo: **2 vị trí liền chuỗi LUÔN LUÔN là 2 ô liền kề thật, không ngoại lệ**.

**Hilbert curve** (toán học, từ 1891) — đường cong chỉ bước sang ô LIỀN KỀ mỗi lần. Ví dụ tối giản, lưới 2×2 (an toàn để kiểm chứng tay):
```
Vị trí 1=(1,1) → Vị trí 2=(2,1) → Vị trí 3=(2,2) → Vị trí 4=(1,2)
```
Kiểm tra: `(1,1)-(2,1)` cách 1 ✓; `(2,1)-(2,2)` cách 1 ✓; `(2,2)-(1,2)` cách 1 ✓ — **không cú nhảy nào**.

**Giới hạn của Hilbert gốc:** chỉ chạy được trên lưới **VUÔNG, kích thước lũy thừa 2** (2×2, 4×4, 8×8...) — vì cơ chế chia luôn bổ thành 4 hình vuông con bằng nhau. Mạng sông thật không vuông vắn như vậy.

**Gilbert curve** (Jakub Červený, xác nhận qua `jakubcerveny.github.io/gilbert`) = bản **tổng quát hóa** Hilbert — chạy được trên hình chữ nhật **kích thước bất kỳ**, vẫn giữ tính chất "không cú nhảy".

#### 4.9.8 Cơ chế Gilbert curve — KHÔNG ép thành hình vuông

Trả lời trực tiếp câu hỏi "hcn chia thành hình vuông hay sao": **KHÔNG.** Đây là khác biệt cốt lõi so với Hilbert gốc:

- **Hilbert gốc:** mọi lần chia đều bổ hình vuông thành **4 hình vuông con bằng nhau** — bắt buộc vuông, lũy thừa 2.
- **Gilbert:** chia hình chữ nhật thành **3 vùng — "up" (trên), "right" (phải), "down" (dưới)** (trích nguyên văn: *"A general rectangle... is split into three regions ('up','right','down'), for which the function calls itself recursively, until a trivial path can be produced"*).

**Vì sao chia 3 vùng lại tổng quát hóa được:** hình gần vuông → 3 vùng vẫn khá vuông vắn, giống Hilbert. Hình dài/hẹp (VD 8×2) → không ép vuông (sẽ ra mảnh quá mỏng, vô lý) mà làm **"bước dài" theo đúng hướng dài** (*"long right step... two recursions only in the same direction"*), chỉ chia 3 vùng khi cần rẽ.

**Xử lý số lẻ (để không hở đường đi):** trích nguyên văn: *"if the width of a rectangle is odd and its height is even, it is impossible to generate a continuous path"* — thuật toán tự động làm tròn 1 kích thước lên số chẵn khi cần (*"increments the length 'b/2' by one if it happens to be odd"*) để đảm bảo luôn nối liền, không hở.

**Điểm dừng đệ quy:** chia nhỏ dần tới khi vùng còn lại "trivial" (đủ nhỏ, VD 1 hàng/1 cột) — vẽ thẳng đường đơn giản, không chia tiếp.

**Không tính tay dãy số Gilbert 4×4 cụ thể** — thuật toán đệ quy trên phức tạp, tính tay dễ sai; muốn có dãy số thật cần chạy code tham khảo (`github.com/jakubcerveny/gilbert`).

#### 4.9.9 4 curve CHÍNH THỨC dùng trong RiverMamba + ý nghĩa "xoay"

Trích nguyên văn: *"we sweep in the first block over the horizontal direction. The second block then sweeps over the vertical direction and we continue with the Gilbert curve and its transposed. These four space-filling curves are iterated."*

```
Khối Hindcast 1: Sweep ngang
Khối Hindcast 2: Sweep dọc
Khối Hindcast 3: Gilbert curve
Khối Hindcast 4: Gilbert curve (xoay/transposed)
→ Lặp lại chu kỳ 4 curve này cho các khối tiếp theo
```

**"Xoay/transpose" nghĩa là gì — chứng minh bằng số thật:** transpose = đổi chỗ hàng↔cột, điểm `(hàng,cột)` thành `(cột,hàng)`. Lấy lưới mưa, transpose rồi áp **đúng thuật toán Sweep ngang** lên lưới đã transpose:
```
Lưới đã transpose:  (1,1)=3 (1,2)=5 (1,3)=1 (1,4)=4 / (2,1)=7 (2,2)=4 (2,3)=3 (2,4)=6 / ...
Sweep ngang trên lưới này → dãy: 3,5,1,4,7,4,3,6,2,6,5,1,9,8,2,3
```
**So với dãy Sweep DỌC gốc đã tính ở 4.9.4:** `3,5,1,4,7,4,3,6,2,6,5,1,9,8,2,3` — **KHỚP TUYỆT ĐỐI**. Đã chứng minh: **"Sweep dọc" = "Sweep ngang chạy trên lưới đã transpose"** — đây chính là nghĩa của "xoay": đổi vai trò hàng↔cột TRƯỚC KHI áp thuật toán, áp dụng y hệt nguyên lý này cho Gilbert → "Gilbert xoay" = Gilbert curve chạy trên lưới đã transpose, nhấn mạnh hướng đường chéo khác Gilbert gốc.

**Lý do dùng 4 curve thay vì 1 (trích nguyên văn):** *"enabling RiverMamba to capture different contextual features"* — mỗi curve "nhìn" mạng lưới theo góc khác, tránh thiên vị 1 hướng.

#### 4.9.10 Nối nhiều ngày lại thành 1 chuỗi

Trích nguyên văn: *"The spatial curves are connected over time by continuing the last point of the curve at t with the first point of the curve at t+1."*

**Ví dụ:** ngày 1 (Sweep ngang, 16 điểm) kết thúc ở vị trí 16 (`(4,4)`). Ngày 2 nối tiếp ngay, bắt đầu ở vị trí 17 (`(1,1)`, theo đúng công thức `Φ` ở 4.9.3). 2 vị trí này liền chuỗi, nhưng `(4,4)` ngày 1 và `(1,1)` ngày 2 là 2 góc đối diện bản đồ — **cùng kiểu lỗi B, xảy ra ở ranh giới giữa 2 NGÀY** thay vì 2 hàng. Paper không bàn sâu chỗ này, chỉ nêu quy tắc nối.

#### 4.9.11 Output của serialization là gì — nối với Bidirectional Mamba (4.7)

Output = **1 dãy số đã sắp lại theo `Φ`** (VD dãy Sweep ngang ở 4.9.4: `3,7,2,9,5,4,6,8,...`) — **không tính toán gì cả**, chỉ đổi thứ tự. Dãy này chính là `x_1,x_2,...` đưa thẳng vào Bidirectional Mamba (đã học ở 4.7.2, 4.7.2b).

Sau khi Mamba xử lý ra `y_1,...,y_16`, cần hàm **ngược** `Φ⁻¹:ℕ→ℤ³` (deserialization, đã tra xác nhận có tồn tại) để map kết quả về đúng tọa độ ban đầu: VD `y_4` (kết quả tại vị trí 4) → `Φ⁻¹(4)=(1,4)` → đây là kết quả dành cho đúng điểm `(1,4)` trên bản đồ. Không có bước map ngược này thì có `y_4` cũng vô nghĩa.

#### 4.9.12 Minh họa thêm — Zigzag curve (CHỈ để hiểu rõ hơn, KHÔNG phải 1 trong 4 curve chính thức)

*Lưu ý:* Zigzag không nằm trong danh sách 4 curve RiverMamba thực sự dùng (4.9.9) — paper có nhắc tới việc thử nghiệm Zigzag trong ablation, nhưng KHÔNG chọn dùng chính thức. Dùng ở đây chỉ để minh họa 1 tính chất, an toàn tính tay (khác Gilbert, không rủi ro tính sai).

**Zigzag ngang** (quét trái→phải hàng 1, rồi PHẢI→TRÁI hàng 2, lặp lại):
```
Hàng 1 (trái→phải): vị trí 1-4:   3  7  2  9
Hàng 2 (phải→trái): vị trí 5-8:   8  6  4  5
Hàng 3 (trái→phải): vị trí 9-12:  1  3  5  2
Hàng 4 (phải→trái): vị trí 13-16: 3  1  6  4
```
Kiểm tra ranh giới hàng: vị trí 4=`(1,4)`, vị trí 5=`(2,4)` — cùng cột, cách đúng 1 → **Lỗi B hết ở MỌI ranh giới hàng**. Nhưng sinh lỗi mới: `(1,1)`(vị trí 1) và `(2,1)`(vị trí 8) — sát nhau ngoài đời nhưng giờ cách **7 vị trí**, tệ hơn cả Sweep thường. → Đúng quy luật đã thấy: sửa cặp này thì hỏng cặp khác.

**Bảng so sánh tổng quan:**

| Curve | Lỗi B (nhảy giả liền kề) | Lỗi A (điểm gần bị đẩy xa) |
|---|---|---|
| Sweep ngang | Có (VD vị trí 4→5) | Có (VD `(1,4)`-`(2,4)` cách 4) |
| Sweep dọc | Có (dạng khác) | Đỡ hơn theo cột, tệ hơn theo hàng |
| Zigzag (minh họa, không chính thức) | Hết ở ranh giới hàng | Vẫn còn (VD `(1,1)`-`(2,1)` cách 7) |
| Gilbert (chính thức) | Hết hoàn toàn (đã chứng minh nguyên lý ở 4.9.7) | Tốt nhất trong các curve (theo paper) |

#### 4.9.13 Kết quả thực nghiệm

Trích nguyên văn: *"a combination of Sweep and Gilbert curves performs best"* — số liệu ablation chi tiết nằm ở phụ lục (Section F.4), chưa truy cập được bản đầy đủ, chỉ xác nhận được kết luận tổng quát này.

#### 4.9.14 Kiểm tra lại và nguồn

- Toàn bộ trích dẫn nguyên văn lấy từ `arxiv.org/html/2505.22535v3` (bản đầy đủ, có appendix) — không suy đoán.
- Cơ chế Gilbert curve (4.9.8) tra từ đúng trang tác giả gốc `jakubcerveny.github.io/gilbert` — không suy đoán.
- Đã tự tính tay lại toàn bộ số liệu ví dụ (Sweep ngang, Sweep dọc, Zigzag, Φ, transpose) — khớp đúng nhau, đặc biệt đã CHỨNG MINH bằng số thật "Sweep dọc = Sweep ngang trên lưới transpose" (4.9.9) — không chỉ khẳng định suông.
- Lưới 4×4 và toàn bộ số liệu mưa là ví dụ tự xây dựng để minh họa — không phải số liệu thật từ paper (paper không công bố ví dụ số cụ thể kiểu này).
- Cố tình KHÔNG tự bịa dãy số Gilbert 4×4 cụ thể (4.9.8) vì thuật toán đệ quy phức tạp, tính tay rủi ro sai — đây là giới hạn thành thật của phần ghi chú này, cần chạy code thật nếu cần số liệu chính xác.

---

### 4.10 Ví dụ minh họa TOÀN BỘ PIPELINE 1 lớp Hindcast — nối Serialization + LOAN + Selective SSM/Scan + Bidirectional lại với nhau

#### 4.10.1 RiverMamba có mấy lớp Hindcast, mỗi lớp làm gì

Đã tra xác nhận (Section 3 + Table 8, Appendix): *"we defined 3 layers to encode the input"*, *"Number of hindcast layers: 3"*. **3 lớp KHÔNG phải 3 loại khác nhau về vai trò** — là **3 lớp Mamba xếp CHỒNG lên nhau** (giống xếp nhiều lớp Transformer), mỗi lớp làm đúng 1 việc: nén dần chiều **THỜI GIAN** xuống 1 nửa. Trích nguyên văn: *"Except for the 1st layer which processes the full temporal resolution, the temporal resolution is down-sampled by a factor of 2 with a linear layer at the beginning of each hindcast layer, such that the output of the last hindcast layer... has a temporal resolution of T=1"*.

```
Lớp 1: xử lý ĐỦ số ngày gốc (không nén) — dùng Sweep ngang
Lớp 2: T giảm 1 nửa (qua 1 lớp Linear ở ĐẦU lớp) — dùng Sweep dọc
Lớp 3: T giảm tiếp 1 nửa, còn T=1 — dùng Gilbert curve
```
Cơ chế downsample cụ thể (paper chỉ nói *"a linear layer"*, không cho công thức) — **chưa xác nhận chi tiết**, không suy đoán thêm.

**Về 4 curve đã học ở 4.9.9** (Sweep ngang, Sweep dọc, Gilbert, Gilbert xoay): đó là **QUY LUẬT chung** paper mô tả cho trường hợp có nhiều lớp — nhưng vì model thật chỉ có **3 lớp**, nên trên thực tế chỉ dùng đúng 3 curve đầu của chu kỳ (Sweep ngang, Sweep dọc, Gilbert). Curve thứ 4 ("Gilbert xoay") chỉ được dùng NẾU có lớp thứ 4 trở lên — với config 3 lớp thật, nó không được gọi tới.

**Phạm vi ví dụ dưới đây: CHỈ làm đủ LỚP 1** (Sweep ngang, không cần downsample vì downsample xảy ra ở ĐẦU lớp 2). Lớp 2, Lớp 3, khối Forecast, và Loss function KHÔNG nằm trong ví dụ này (xem bảng giới hạn ở 4.10.7).

#### 4.10.2 Bước 1 — Input (T=2 ngày, P=2 điểm, K=3 kênh)

Dữ liệu gốc có dạng lưới `(B,T,P,K)`: `B` là batch (nhiều mẫu train cùng lúc), `T` là số ngày, `P` là số điểm sông trải trên bản đồ, `K=192` là số kênh đặc trưng thật. Đây là dữ liệu 2 chiều không gian (`P`) cộng thêm 1 chiều thời gian (`T`) — CHƯA có bất kỳ thứ tự tuyến tính nào để đưa thẳng vào Mamba, vì bản đồ sông là 1 mạng lưới, không phải 1 hàng.

**Ví dụ này rút gọn `K=192` còn `K=3`, gán ý nghĩa vật lý cụ thể cho từng kênh (tự chọn 3 biến quen thuộc để minh họa, không phải 3 kênh đầu tiên thật trong `X_embed` của paper):**
```
Kênh 1 = lượng mưa (mm)
Kênh 2 = nhiệt độ (°C, quy mô đã thu nhỏ để hợp với 2 kênh kia — thực tế cần chuẩn hóa riêng trước)
Kênh 3 = độ ẩm đất (đơn vị tương đối)
```
```
Ngày1-A=(1,1)=[10,4,7]   Ngày1-B=(1,2)=[5,15,20]
Ngày2-A=(1,1)=[8,6,5]    Ngày2-B=(1,2)=[12,10,15]
X_static_A=[0,2]  X_static_B=[0,7]  (không đổi theo ngày — VD độ dốc địa hình chuẩn hóa)
```
**Phần lược bỏ:** input thật phải qua **Embedding** (số vật lý thô → K=192 chiều, qua các lớp chiếu tuyến tính riêng cho từng loại biến) trước — ví dụ này bỏ qua bước đó, coi như đã có sẵn 3 kênh kể trên.

#### 4.10.3 Bước 2 — Serialization (4.9)

Toàn bộ cơ chế Mamba xây dựng từ Phần 4.3 tới giờ đều dựa trên 1 giả định nền tảng: có 1 chuỗi các bước `1,2,3,...`, và trạng thái `h_t` tính từ `h_(t-1)` theo đúng thứ tự đó — bản chất của mọi kiến trúc dạng đệ quy (RNN, LSTM, GRU, Mamba), không có "thứ tự" thì không có gì để đệ quy cả. Nhưng mạng lưới sông là 1 tấm lưới 2 chiều, không tự nhiên có thứ tự "bước 1, bước 2". Vì vậy trước khi vào Mamba, bắt buộc phải biến lưới 2D thành chuỗi 1D — qua hàm song ánh `Φ:ℤ³→ℕ` (3 chiều đầu vào là hàng, cột, ngày — gộp cả không gian lẫn thời gian). Lớp Hindcast đầu tiên dùng Sweep ngang — quét từng hàng, trái sang phải.
```
Φ(1,1,1)=1 → Vị trí 1 = Ngày1-A
Φ(1,2,1)=2 → Vị trí 2 = Ngày1-B
Φ(1,1,2)=3 → Vị trí 3 = Ngày2-A
Φ(1,2,2)=4 → Vị trí 4 = Ngày2-B
```
**Phần lược bỏ:** công thức `Φ` cụ thể là tự viết minh họa, paper không cho công thức toán tường minh này.

#### 4.10.4 Bước 3 — LOAN₁ (4.8)

Trước khi vào mạng neural, luôn cần chuẩn hóa (`(X-μ)/σ`) để các kênh có cùng thang đo. Nhưng phép chuẩn hóa này tính riêng từng điểm/ngày, hoàn toàn không biết điểm đó nằm ở đâu trên bản đồ — nếu 2 điểm địa hình khác hẳn nhau (núi dốc vs đồng bằng) tình cờ cùng lượng mưa hôm đó, sau chuẩn hóa chúng ra CÙNG 1 con số, model mất khả năng phân biệt (đã chứng minh cụ thể ở 4.8.2/4.8.5). LOAN giữ nguyên phép chuẩn hóa, cộng thêm 1 lượng riêng theo từng điểm, tính từ đặc trưng tĩnh (địa hình) qua 1 lớp `Linear` CHUNG cho mọi điểm rồi qua `GELU`.
```
A ngày1 (kênh mưa/nhiệt độ/độ ẩm=[10,4,7]): chuẩn hóa=[1,225;-1,225;0] + GELU(Linear(X_static_A))=[0,066;-0,029;0,043]
  → LOAN₁(1)=[1,291;-1,253;0,043]
B ngày1 ([5,15,20]): chuẩn hóa=[-1,336;0,267;1,069] + GELU(Linear(X_static_B))=[0,282;-0,086;0,173]
  → LOAN₁(2)=[-1,054;0,181;1,242]
A ngày2 ([8,6,5], μ=6,33,σ=1,25): chuẩn hóa=[1,333;-0,267;-1,067] + cùng GELU(A)
  → LOAN₁(3)=[1,399;-0,296;-1,024]
B ngày2 ([12,10,15], μ=12,33,σ=2,05): chuẩn hóa=[-0,163;-1,138;1,301] + cùng GELU(B)
  → LOAN₁(4)=[0,119;-1,224;1,474]
```

#### 4.10.5 Bước 4 — Khối Mamba: Selective SSM (4.5) + Selective Scan (4.6) + Bidirectional (4.7)

**Selective SSM:** SSM cổ điển dùng `Ā,B̄` cố định, áp dụng y hệt mọi ngày bất kể input — không tự "quyết định" ngày nào quan trọng hơn. Mamba sửa bằng cách làm `B_t,C_t,Δ_t` tính trực tiếp từ input mỗi bước (`B_t=w_B×x_t`, `Δ_t=softplus(w_Δ×x_t)`), khiến `Ā_t=1+A×Δ_t` biến đổi gián tiếp theo input dù `A` vẫn cố định.

**Selective Scan:** vì hệ số giờ đổi theo từng bước, mẹo tích chập cũ (dùng được khi `Ā,B̄` cố định) hỏng. Giải pháp: gộp 2 bước affine liền kề thành 1 công thức (`a_gộp=a_2×a_1`, `b_gộp=a_2×b_1+b_2`), lặp theo "vòng" — mỗi vòng khoảng cách gộp tăng gấp đôi — chỉ cần `⌈log₂n⌉` vòng ra ĐỦ mọi `h_t` cùng lúc. Làm được vì `Ā_t` chỉ cần biết `x_t`, KHÔNG cần `h_(t-1)` — khác LSTM/GRU (`forget_gate` bắt buộc cần `h_(t-1)`, đã chứng minh có nguồn ở 4.6.3c: Jacobian LSTM/GRU dày đặc, không thỏa tính chất kết hợp được).

**Bidirectional:** thứ tự chuỗi ở Bước 2 là do MÌNH sắp xếp theo không gian, không phải quan hệ nhân-quả thời gian — 2 điểm sông sát nhau ngoài đời có thể bị đẩy xa nhau trong chuỗi (chứng minh cụ thể bằng lưới `(1,4)`/`(2,4)` ở 4.7.1b). Quét 1 chiều sẽ làm điểm đứng đầu chuỗi "mù" thông tin từ các điểm phía sau. Giải pháp: chạy đúng cặp (Selective SSM+Scan) ở trên **2 lần, trọng số hoàn toàn riêng** (xuôi + ngược), rồi gộp bằng `y=SiLU(z)⊙y_forward+SiLU(z)⊙y_backward`.

Trọng số dùng cho ví dụ (tự chọn để tính tay, KHÔNG phải trọng số đã train thật): xuôi `A=-0,15,w_B=0,4,w_Δ=0,5,Cf=0,9`; ngược `A_b=-0,2,w_Bb=0,35,w_Δb=0,45,Cb=1,1`; gate `w_z=0,3`.

**Kênh 1 = mưa** (`x: 1,291;-1,054;1,399;0,119`):
```
Xuôi: h_1^f=0,712  h_2^f=0,868  h_3^f=1,595  h_4^f=1,425
Ngược: h_4^b=0,004  h_3^b=0,720  h_2^b=0,838  h_1^b=1,264
Đọc(Cf=0,9,Cb=1,1): y_fwd=[0,641;0,781;1,436;1,283]  y_bwd=[1,390;0,922;0,792;0,004]
Gate SiLU(z)=[0,255;-0,117;0,282;0,018]
y_kênh_mưa = [0,519 ; -0,199 ; 0,628 ; 0,024]
```

**Kênh 2 = nhiệt độ** (`x: -1,253;0,181;-0,296;-1,224`):
```
Xuôi: h_1^f=0,269  h_2^f=0,249  h_3^f=0,248  h_4^f=0,491
Ngược: h_4^b=0,239  h_3^b=0,228  h_2^b=0,203  h_1^b=0,433
Đọc: y_fwd=[0,242;0,224;0,223;0,442]  y_bwd=[0,476;0,223;0,251;0,263]
Gate=[-0,130;0,028;-0,041;-0,128]
y_kênh_nhiệtđộ = [-0,093 ; 0,013 ; -0,019 ; -0,090]
```

**Kênh 3 = độ ẩm đất** (`x: 0,043;1,242;-1,024;1,474`):
```
Xuôi: h_1^f=0,0005  h_2^f=0,649  h_3^f=0,801  h_4^f=1,646
Ngược: h_4^b=0,820  h_3^b=0,920  h_2^b=1,280  h_1^b=1,101
Đọc: y_fwd=[0,0005;0,585;0,721;1,481]  y_bwd=[1,211;1,408;1,012;0,902]
Gate=[0,007;0,244;-0,114;0,301]
y_kênh_độẩm = [0,008 ; 0,485 ; -0,198 ; 0,716]
```

**→ Output Mamba đủ 4 vị trí, đủ 3 kênh (mưa, nhiệt độ, độ ẩm — theo đúng thứ tự):**
```
Ngày1-A=[0,519;-0,093;0,008]    Ngày1-B=[-0,199;0,013;0,485]
Ngày2-A=[0,628;-0,019;-0,198]   Ngày2-B=[0,024;-0,090;0,716]
```
**Phần lược bỏ:** output trên chưa qua `Linear_in` (đầu khối Mamba), `conv1d+SiLU` (trộn cục bộ vài bước liền kề trước SSM), và `Linear_out` (cuối khối Mamba) — 3 lớp bọc quanh SSM mà khối Mamba thật có (đã nêu ở 4.7.2), ví dụ này chỉ tính đúng phần lõi SSM+Scan+Bidirectional.

#### 4.10.6 Bước 5 — LOAN₂ (4.8)

Sau khi qua khối Mamba, dữ liệu đã bị trộn thông tin qua lại giữa các vị trí (đúng mục đích của Selective Scan/Bidirectional) — nhưng vì bị trộn, "thẻ tên địa điểm" gắn ở LOAN₁ có thể đã bị pha loãng. LOAN₂ lặp lại ĐÚNG công thức Equation 2 — chuẩn hóa lại output Mamba, cộng lại đúng `GELU(Linear(X_static))` (không đổi vì static không theo ngày) — đảm bảo thông tin địa điểm vẫn rõ trước bước tiếp theo. Xác nhận từ Algorithm 1: LOAN chạy đúng 2 lần mỗi lớp.
```
Ngày1-A: μ=0,103,σ=0,193 → chuẩn hóa=[1,397;-0,882;-0,515] + GELU(A) → LOAN₂(Ngày1-A)=[1,463;-0,916;-0,467]
Ngày1-B: μ=0,060,σ=0,169 → chuẩn hóa=[-1,036;-0,315;1,352] + GELU(B) → LOAN₂(Ngày1-B)=[-0,763;-0,389;1,521]
Ngày2-A: μ=0,137,σ=0,355 → chuẩn hóa=[1,384;-0,440;-0,944] + GELU(A) → LOAN₂(Ngày2-A)=[1,450;-0,469;-0,901]
Ngày2-B: μ=0,217,σ=0,356 → chuẩn hóa=[-0,541;-0,861;1,402] + GELU(B) → LOAN₂(Ngày2-B)=[-0,259;-0,947;1,575]
```
**Đây là output của ĐÚNG 1 LỚP HINDCAST (T=2, chưa downsample)** — chưa phải số m³/s cuối.

#### 4.10.7 Bước 6 — Downsample `T=2→T=1` (mô tả, không tính đủ số)

RiverMamba đưa vào nhiều ngày lịch sử nhưng chỉ cần ra 1 dự báo cho hiện tại — cần tóm tắt dần các ngày lại. Cơ chế: ở ĐẦU mỗi lớp Hindcast (trừ lớp 1), 1 lớp Linear gộp từng cặp ngày liền kề thành 1 ngày mới, giảm `T` đi 1 nửa mỗi lần — cùng nguyên lý "giảm 1 nửa mỗi vòng" đã học ở Selective Scan (4.6.6), chỉ khác trục: đây gộp theo THỜI GIAN giữa các LỚP, còn 4.6 gộp theo VỊ TRÍ trong 1 lớp.
```
Ngày_gộp-A (kênh mưa, trọng số 0,5/0,5 tự chọn minh họa) = 0,5×0,519+0,5×0,628 = 0,574
```
**Phần lược bỏ:** paper chỉ nói *"a linear layer"*, không cho công thức/trọng số cụ thể — số `0,5/0,5` trên là tự chọn minh họa nguyên lý, không phải cách tính thật.

#### 4.10.8 Bước 7 — Lặp lại Bước 2-6 thêm 2 lần (Lớp 2, Lớp 3) — chỉ mô tả

Toàn bộ chuỗi Serialization→LOAN₁→Mamba→LOAN₂→Downsample chạy lại 2 lần nữa, đổi curve theo chu kỳ đã học (4.9.9): Lớp 2 dùng Sweep dọc (lý do: Sweep ngang và Sweep dọc mắc lỗi ở 2 cặp điểm KHÁC nhau — 4.9.6 — đổi hướng giúp "vá" cặp mà lớp trước bỏ sót), Lớp 3 dùng Gilbert (không có cú nhảy nào, tốt nhất trong các curve). Sau Lớp 3, `T` đã giảm hết còn `T=1`.
**Phần lược bỏ:** không tính số cụ thể cho Lớp 2, Lớp 3 — Gilbert không tự bịa số được (4.9.8); chưa rõ trọng số Mamba có share giữa 3 lớp hay không (paper không nói rõ).

#### 4.10.9 Bước 8 — MLP → dự báo m³/s (mô tả, không tính đủ số)

Vector tóm tắt cuối cùng (sau `T=1`, sau cả 3 lớp) qua 1 lớp tuyến tính cuối (`Dự_báo=a×h+b`, đúng nguyên tắc đã học ở LSTM 1.5) — ra đúng 1 số dự báo lưu lượng (m³/s) cho từng điểm sông.
**Phần lược bỏ:** khối Forecast (dùng thêm HRES), Embedding đầu vào, và Loss function (biến đổi `log1p` có dấu + trọng số return-period/lead-time, xem `02_RiverMamba.md` Mục 11) — không nằm trong phạm vi ví dụ 4.10.

#### 4.10.10 Bảng tổng hợp phạm vi — đã làm ĐỦ gì, còn thiếu gì (không giấu)

| Đã làm ĐỦ, có công thức/nguồn | Chưa làm — ngoài phạm vi ví dụ này |
|---|---|
| Serialization (Sweep ngang) — công thức `Φ` tự viết minh họa | Sweep dọc (Lớp 2), Gilbert (Lớp 3) — Gilbert cố tình không bịa số |
| LOAN₁, LOAN₂ — đúng công thức Equation 2, đủ 4 vị trí | Downsample T=2→1 — paper không cho công thức cụ thể, trọng số tự chọn minh họa |
| Selective SSM (4.5) + Selective Scan (4.6) + Bidirectional (4.7) — đúng công thức, đủ 3 kênh (mưa/nhiệt độ/độ ẩm) × 4 vị trí | `Linear_in`, `conv1d+SiLU`, `Linear_out` bọc quanh khối Mamba thật |
| Lớp 1 Hindcast hoàn chỉnh, ý nghĩa vật lý từng kênh rõ ràng | Lớp 2, Lớp 3, khối Forecast (dùng HRES), Embedding đầu vào, Loss function |
| Mọi số liệu tự tính tay verify khớp | Mọi trọng số (`A,w_B,W`...) là tự bịa minh họa — KHÔNG phải trọng số đã train thật |

#### 4.10.11 Kiểm tra lại

- Đã tự tính tay lại toàn bộ số liệu ở 4.10.4-4.10.6 (đủ 4 vị trí × 3 kênh × 2 chiều) — khớp đúng nhau, không lỗi tính toán.
- Số liệu là ví dụ tự xây dựng để minh họa cách các cơ chế (4.5-4.9) GHÉP LẠI với nhau trong 1 pipeline thật — không phải số liệu/trọng số thật từ paper.
- Mục đích của 4.10 là nối các mảnh lý thuyết rời rạc (4.5-4.9) thành 1 luồng liền mạch, dễ hình dung — không thay thế cho việc đọc code thật khi cần triển khai chính xác.
