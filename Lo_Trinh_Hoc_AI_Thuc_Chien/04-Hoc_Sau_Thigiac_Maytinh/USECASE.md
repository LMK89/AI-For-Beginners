# 🧭 USECASE: CNN Dùng Khi Nào? — Học Qua 30 Bức Ảnh 8×8 Thật

> **Dành cho người chưa biết gì:** Bài này dùng 30 "bức ảnh" tí hon 8×8 pixel (vạch dọc vs vạch ngang) — đủ nhỏ để bạn hình dung từng pixel trong đầu, đủ thật để chạy code ra số liệu. Ta sẽ thấy tận mắt: mạng phẳng đọc pixel thô **học vẹt vị trí** (100% tập học nhưng **0/6** trên ảnh mới!), còn phép tích chập + pooling thì **bất biến với dịch chuyển** (6/6). Và hiểu luôn padding, pooling, kernel sinh ra để làm gì — qua số đo thật, không qua định nghĩa suông. Code cuối file.

---

## 1. TL;DR — Tóm tắt 30 giây

| Câu hỏi | Trả lời ngắn gọn |
| :--- | :--- |
| CNN làm gì? | Thay vì "nhìn" cả bức ảnh như một dãy số dài, nó **trượt một kính lúp nhỏ (kernel 3×3)** khắp ảnh để tìm hoa văn (cạnh dọc, cạnh ngang...) — hoa văn nằm đâu cũng bắt được. |
| Vì sao không dùng mạng phẳng (MLP) cho ảnh? | Hai lý do đo được: (1) ảnh 1000×1000 nối vào 1000 neuron = **1 TỶ trọng số** (CNN: 288); (2) mạng phẳng học thuộc **từng vị trí pixel** — vật thể dịch sang trái 2 pixel là nó mù (số thật: 0/6 bên dưới). |
| Pooling để làm gì? | Trả lời câu "có vạch dọc **ở đâu đó** không?" thay vì "pixel số 13 có sáng không?". Số thật: bỏ pooling đi, đúng probe tụt từ 6/6 xuống 3/6. |
| Padding để làm gì? | Không đệm viền ảnh thì kernel không trượt ra được tới mép → **vật thể sát mép ảnh trở nên vô hình** (số thật: đáp ứng [0, 0] — như ảnh trống). |

---

## 2. CNN Dùng Cho Những Dữ Liệu Gì?

### ✅ Sân nhà — dữ liệu có CẤU TRÚC KHÔNG GIAN (pixel cạnh nhau liên quan nhau)

| Bài toán | Ứng dụng doanh nghiệp (khớp README chương này) |
| :--- | :--- |
| Kiểm thử ngoại quan sản phẩm | Soi lỗi bề mặt linh kiện trên dây chuyền — lỗi xuất hiện ở góc nào của khung hình cũng phải bắt được (bất biến dịch chuyển!) |
| OCR, đọc chữ số viết tay | Chữ số nằm lệch trái/phải vẫn là chữ số đó |
| Nhận diện khuôn mặt, xe cộ, y tế (X-quang) | Khối u ở vị trí bất kỳ trong phim chụp |

### ❌ Khi nào đừng dùng

| Tình huống | Vì sao | Dùng gì |
| :--- | :--- | :--- |
| Dữ liệu bảng (duyệt vay, churn) | Cột "thu nhập" đứng cạnh cột "nợ" chẳng có nghĩa "không gian" gì — trộn thứ tự cột mô hình vẫn phải chạy y nguyên | XGBoost (Chương 2) |
| Văn bản, chuỗi dài | Quan hệ tầm xa giữa các từ | Transformer (Chương 6) |
| Ảnh nhưng chỉ cần đặc điểm tổng thể (độ sáng trung bình, histogram màu) | Đếm thống kê là đủ, đừng nướng GPU | Đặc trưng thủ công + mô hình thường |

---

## 3. Bài Toán: Phân Loại Vạch Dọc vs Vạch Ngang — Và Hai Đấu Thủ

Mỗi ảnh 8×8 = 64 pixel (0 = đen, 1 = sáng). Nhãn: **│ dọc** hay **─ ngang**. Hai đấu thủ:

1. **Mạng phẳng:** Logistic Regression đọc thẳng 64 pixel — mỗi pixel một trọng số riêng gắn chặt vào **vị trí** đó.
2. **CNN mini:** đúng 2 kernel 3×3 cầm tay (một bắt cạnh dọc, một bắt cạnh ngang — dạng bộ lọc Sobel trong README), trượt khắp ảnh (có padding), qua ReLU, rồi **global max pooling** → mỗi ảnh chỉ còn 2 con số: `[đáp ứng dọc mạnh nhất, đáp ứng ngang mạnh nhất]`.

Kernel bắt vạch dọc (giống Sobel trong README của chương):

```
-1  2 -1        Gặp cột sáng kẹp giữa 2 cột tối → điểm cao
-1  2 -1        Gặp vùng phẳng → điểm 0
-1  2 -1        Gặp vạch ngang → điểm thấp
```

---

## 4. Bộ Dữ Liệu 30 Ảnh — Ảnh Nào Dễ, Ảnh Nào Gài Bẫy?

Cột **[dọc, ngang]** = 2 đặc trưng mà CNN mini trích ra được (số chạy thật) — chưa cần mô hình, nhìn 2 số này là tự phân loại được bằng mắt.

### 🟢 Nhóm A (ảnh 1–10): Vạch chuẩn ở vị trí cột/hàng 1–3

| # | Mô tả ảnh | Nhãn | [dọc, ngang] | Giải thích |
|---|:---|:---:|:---:|:---|
| 1–5 | Vạch dọc ở cột 1, 2, 3 (+2 ảnh lặp) | │ | [6.0, 1.0] | Đáp ứng dọc 6.0 vs ngang 1.0 — một trời một vực |
| 6–10 | Vạch ngang ở hàng 1, 2, 3 (+2 ảnh lặp) | ─ | [1.0, 6.0] | Đối xứng hoàn hảo |

### 🟢 Nhóm B (ảnh 11–16): Vạch ở vị trí 4–6 — với CNN chẳng có gì mới

| # | Mô tả ảnh | Nhãn | [dọc, ngang] | Giải thích |
|---|:---|:---:|:---:|:---|
| 11–13 | Vạch dọc cột 4, 5, 6 | │ | [6.0, 1.0] | **Y HỆT nhóm A!** Kernel trượt nên vạch nằm đâu, đáp ứng max vẫn 6.0. Với mạng phẳng thì đây là 64 pixel hoàn toàn khác — phải học lại từ đầu |
| 14–16 | Vạch ngang hàng 4, 5, 6 | ─ | [1.0, 6.0] | Tương tự |

**Đây là linh hồn của CNN:** cùng MỘT bộ trọng số 3×3 dùng lại cho mọi vị trí (weight sharing) → vật thể dịch đi đâu, đặc trưng không đổi. Mạng phẳng phải "học riêng từng chỗ đứng" của vạch.

### 🟡 Nhóm C (ảnh 17–22): Nhiễu muối tiêu — kernel vẫn điềm tĩnh

| # | Mô tả ảnh | Nhãn | [dọc, ngang] | Giải thích |
|---|:---|:---:|:---:|:---|
| 17–18 | Vạch dọc + 5 pixel nhiễu rải rác | │ | [6.0, 2.0] | Nhiễu lẻ tẻ chỉ đẩy đáp ứng ngang từ 1.0 lên 2.0 — vẫn thua xa 6.0 |
| 19–20 | Vạch ngang + 5 pixel nhiễu | ─ | [2.0, 6.0] | — |
| 21–22 | Vạch + 8 pixel nhiễu | │/─ | tương tự | Nhiễu phải nhiều đến mức tự xếp thành... một vạch thì mới lừa được kernel |

### 🟡 Nhóm D (ảnh 23–26): Vạch đứt quãng — suy giảm có kiểm soát

| # | Mô tả ảnh | Nhãn | [dọc, ngang] | Giải thích |
|---|:---|:---:|:---:|:---|
| 23 | Vạch dọc đứt 2 pixel | │ | [4.0, 1.0] | Đáp ứng giảm 6.0 → 4.0 nhưng vẫn phân loại ngon — sản phẩm trầy xước trên dây chuyền vẫn nhận ra |
| 24–26 | Vạch đứt 2–3 pixel khác | │/─ | [4.0, 1.0]... | Mô hình pixel thô thì mất trắng các pixel đó, dễ lung lay hơn |

### 🔴 Nhóm E (ảnh 27–28): Đường CHÉO — không thuộc lớp nào nhưng bị ép nhãn

| # | Mô tả ảnh | Nhãn (ép) | [dọc, ngang] | Giải thích |
|---|:---|:---:|:---:|:---|
| 27 | Đường chéo ↘ | │ (❓) | [1.0, 1.0] | Chéo = nửa dọc nửa ngang — đáp ứng 2 kernel **bằng nhau chằn chặn**. Đặc trưng đang hét lên "tôi không thuộc lớp nào cả!" nhưng nhãn ép nó phải chọn |
| 28 | Đường chéo ↗ | ─ (❓) | [1.0, 1.0] | Tương tự. Bài học thực chiến: khi ảnh ngoài phạm vi các lớp đã định nghĩa (out-of-distribution), mô hình vẫn bị ép trả lời — hệ thống tử tế phải có ngưỡng "không chắc chắn → chuyển người duyệt" |

### ⚫ Nhóm F (ảnh 29–30): Ảnh trống trơn, nhãn ngược nhau — quy luật bất biến quen thuộc

| # | Mô tả ảnh | Nhãn | [dọc, ngang] | Giải thích |
|---|:---|:---:|:---:|:---|
| 29 | Ảnh đen tuyền (64 pixel = 0) | │ | [0.0, 0.0] | Hai ảnh giống hệt nhau từng pixel... |
| 30 | Ảnh đen tuyền | ─ | [0.0, 0.0] | ...không kiến trúc nào phân biệt nổi. Y hệt nhóm F của mọi chương trước |

---

## 5. Trận Đấu Chính: Ảnh Mới Có Vạch Ở Vị Trí CHƯA TỪNG THẤY

Tập học chỉ có vạch ở cột/hàng 1–6. Đem 6 ảnh mới có vạch ở **cột/hàng 0 và 7** (mép ảnh, chưa từng xuất hiện) ra kiểm tra. Kết quả chạy thật:

| Mô hình | Tập học (28 ảnh) | 6 ảnh vị trí mới | Chuyện gì xảy ra |
| :--- | :---: | :---: | :--- |
| Mạng phẳng (64 pixel thô) | **100%** | **0/6** 💥 | Trọng số dính chặt vào pixel cột 1–6; vạch ở cột 0/7 rơi vào các pixel có trọng số ≈ 0 → đoán sai **TẤT CẢ**. Học vẹt vị trí hoàn hảo: điểm train tuyệt đối, gặp dịch chuyển là sụp đổ |
| **CNN mini (conv + padding + max pool)** | 96% | **6/6** ✅ | Kernel trượt tới đâu bắt vạch tới đó — cột 0 hay cột 7 cũng chỉ là "một chỗ khác để trượt qua". 4% train bị mất là 2 ảnh trống bị ép nhãn (bất khả kháng) |
| CNN nhưng **BỎ pooling** (chỉ đọc đáp ứng tại tâm ảnh) | 54% | 3/6 | Không gom đáp ứng toàn ảnh → chỉ thấy vạch nào đi qua đúng tâm. Pooling chính là thứ biến "phát hiện tại chỗ" thành "phát hiện ở bất kỳ đâu" |

**Và một phát hiện xém bị bỏ qua — bài học về padding:** phiên bản đầu tiên của thí nghiệm này **không có padding**, và vạch ở cột 0 cho đáp ứng **[0, 0]** — y hệt ảnh trống! Kernel 3×3 cần tâm của nó đặt lên vạch, nhưng không có viền đệm thì tâm kernel không bao giờ trượt ra được tới mép ảnh. Đệm một viền số 0 quanh ảnh (zero padding) → đáp ứng bật lên [6.0, 1.0]. Đó là lý do mọi CNN thật đều pad: **không pad = mù vùng mép ảnh**.

---

## 6. Các Config Ảnh Hưởng Cái Gì?

| Config | Ý nghĩa đời thường + số đo trong bài |
| :--- | :--- |
| **Kernel size** (3×3, 5×5) | Cỡ kính lúp. 3×3 đủ bắt cạnh; hoa văn to hơn thì chồng nhiều lớp conv (2 lớp 3×3 nhìn được vùng 5×5) thay vì phóng to kernel — ít tham số hơn |
| **Padding** | Không pad → vật ở mép vô hình (đo được: [0,0]); pad → thấy ([6,1]). Ngoài ra pad giữ kích thước ảnh không teo dần qua các lớp |
| **Pooling** | Đổi câu hỏi "ở pixel này có gì?" thành "trong vùng này có gì?". Bỏ đi: probe tụt 6/6 → 3/6. Trả giá: mất thông tin vị trí chính xác (bài toán cần định vị chính xác — segmentation — phải dùng kiến trúc khác) |
| **Số kernel (channels)** | Bài này 2 kernel tự chọn là đủ vì chỉ có 2 hoa văn. CNN thật để 32–64 kernel/lớp và **để mạng tự học** hình dạng kernel từ dữ liệu — backpropagation (Chương 3) sẽ tự mài ra bộ lọc Sobel và hơn thế |
| **Stride** | Bước nhảy của kính lúp. Stride 2 = quét thưa gấp đôi → nhanh hơn, ảnh ra nhỏ đi một nửa, đổi lấy khả năng bỏ sót chi tiết mảnh |
| **Số trọng số** | Ảnh 1000×1000 nối MLP 1000 neuron = **1.000.000.000 trọng số**; CNN 32 kernel 3×3 = **288**. Chênh nhau 3,5 triệu lần — lý do MLP "quá tải bộ nhớ" trong README không phải nói quá |

---

## 7. Checklist: Bài Toán Ảnh Của Bạn Cần Gì?

- [ ] Vật thể có thể xuất hiện ở **vị trí bất kỳ** trong khung hình? → bắt buộc conv + pooling (bất biến dịch chuyển).
- [ ] Vật thể hay nằm **sát mép ảnh**? → kiểm tra padding.
- [ ] Ảnh có thể **ngoài phạm vi các lớp** đã định nghĩa (nhóm E)? → thêm ngưỡng từ chối "không chắc → người duyệt".
- [ ] Cần biết vạch/lỗi nằm **chính xác ở đâu** (không chỉ có/không)? → cẩn thận với pooling, nó xóa thông tin vị trí.
- [ ] Dữ liệu ảnh ít? → đừng train CNN từ đầu; fine-tune mô hình pretrain (ResNet, EfficientNet) — cùng triết lý với lời khuyên ở Chương 6.

---

## 8. Code Chạy Lại Toàn Bộ (copy vào notebook là chạy)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.RandomState(0)

def v_bar(col, noise=0):                      # ảnh vạch dọc 8x8
    im = np.zeros((8,8)); im[:, col] = 1
    for _ in range(noise): im[rng.randint(8), rng.randint(8)] = 1
    return im
def h_bar(row, noise=0):                      # ảnh vạch ngang
    im = np.zeros((8,8)); im[row, :] = 1
    for _ in range(noise): im[rng.randint(8), rng.randint(8)] = 1
    return im

# Tập học: vạch ở cột/hàng 1-6 (+ ảnh nhiễu), nhãn 1=dọc 0=ngang
imgs = [v_bar(c) for c in range(1,7)] + [h_bar(r) for r in range(1,7)] \
     + [v_bar(c, noise=5) for c in [2,5]] + [h_bar(r, noise=5) for r in [2,5]]
y = np.array([1]*6 + [0]*6 + [1]*2 + [0]*2)

# CNN mini: kernel Sobel-dọc + xoay 90 độ, padding, ReLU, global max pooling
Kv = np.array([[-1,2,-1]]*3, float); Kh = Kv.T
def conv_feat(im):
    im = np.pad(im, 1)                        # <- thử xóa dòng này xem probe mép ảnh!
    fv = fh = 0.0
    for i in range(8):
        for j in range(8):
            p = im[i:i+3, j:j+3]
            fv = max(fv, (p*Kv).sum()); fh = max(fh, (p*Kh).sum())
    return [fv, fh]

X_raw = np.array([im.ravel() for im in imgs])
X_cnn = np.array([conv_feat(im) for im in imgs])

# Probe: vạch ở mép ảnh (cột/hàng 0 và 7) — vị trí CHƯA TỪNG THẤY
probes = [v_bar(0), v_bar(7), h_bar(0), h_bar(7), v_bar(7,noise=4), h_bar(0,noise=4)]
yp = np.array([1,1,0,0,1,0])

for ten, Xtr, Xpr in [("Mạng phẳng 64 pixel", X_raw, np.array([p.ravel() for p in probes])),
                      ("CNN mini",            X_cnn, np.array([conv_feat(p) for p in probes]))]:
    m = LogisticRegression(max_iter=5000).fit(Xtr, y)
    print(f"{ten}: train={(m.predict(Xtr)==y).mean():.0%}"
          f" | ảnh vị trí mới: {(m.predict(Xpr)==yp).sum()}/6")

# Bài tập tự nghịch:
# 1. Xóa dòng np.pad -> probe mép ảnh còn đúng mấy câu? Vì sao? (mục 5)
# 2. In conv_feat của ảnh đường chéo np.eye(8) -> hai đáp ứng bằng nhau!
# 3. Thay max pooling bằng average pooling -> ảnh vạch đứt (mất pixel) đổi thế nào?
```

---

## 9. Kết Nối

| Bạn muốn | Mở |
| :--- | :--- |
| Công thức tích chập, Sobel, kiến trúc CNN đầy đủ | `README.md` của chương này |
| Bài tập code | `04-Bai_Tap_Thuc_Hanh_Computer_Vision_CNN.ipynb` |
| Hiểu backpropagation tự học kernel thế nào | `../03-Neural_Network_Stochastic_Gradient_Descent/USECASE.md` |
| Ảnh cắt thành patch cho Transformer (ViT) | `../../Khang_lession/Transformer_SelfAttention/README.md` |

> 🌟 **Câu chốt:** CNN không "thông minh hơn" mạng phẳng — nó chỉ được **cài sẵn một niềm tin đúng đắn về ảnh**: hoa văn ở đâu cũng là hoa văn đó (weight sharing + pooling), và pixel chỉ có nghĩa khi đứng cạnh nhau (kernel cục bộ). Cài đúng niềm tin → cần ít dữ liệu hơn 3,5 triệu lần tham số để học cùng một việc.
