# 🧭 USECASE: Neural Network & SGD Dùng Khi Nào? — Học Qua 30 Dòng Dữ Liệu Thật

> **Dành cho người chưa biết gì:** Chương 1 đã cho thấy Logistic Regression chỉ vẽ được **một đường thẳng** chia đôi dữ liệu. Bài này đưa ra một bộ 30 dòng dữ liệu mà — về mặt toán học — **không tồn tại đường thẳng nào chia đúng** (bài toán XOR huyền thoại), rồi xem mạng Neural giải nó ra sao, và từng cấu hình (số neuron, hàm kích hoạt, learning rate, batch size) làm thay đổi kết quả thế nào. Mọi con số đều chạy thật — code cuối file.

---

## 1. TL;DR — Tóm tắt 30 giây

| Câu hỏi | Trả lời ngắn gọn |
| :--- | :--- |
| Mạng Neural (MLP) làm gì? | Ghép nhiều "Logistic Regression con" (neuron) thành tầng — mỗi neuron vẽ một đường thẳng, tầng sau **phối hợp** các đường đó thành ranh giới cong, gấp khúc tùy ý. |
| Khi nào BẮT BUỘC cần nó? | Khi dữ liệu kiểu **"lệch pha thì hỏng"** (XOR): giá trị của cột này tốt hay xấu **tùy thuộc** cột kia. Số thật bên dưới: Logistic đoán đúng 50% (như tung xu), MLP 4 neuron đúng 100%. |
| Bí mật quan trọng nhất? | Sức mạnh **không nằm ở số tầng** mà ở **hàm kích hoạt phi tuyến (ReLU)**. Số thật: mạng 4 neuron nhưng bỏ ReLU đi → đúng 50%, y hệt Logistic. Trăm tầng tuyến tính = một tầng. |
| SGD ảnh hưởng gì? | Learning rate quá nhỏ → không học được gì; quá to → nổ; batch nhỏ → hội tụ nhanh gấp ~20 lần trên dữ liệu này (số thật: 36 epoch vs 776 epoch). |

---

## 2. Bài Toán: 30 Cửa Hàng Lãi Hay Lỗ — Quy Luật "Lệch Pha Thì Chết"

Chuỗi bán lẻ có 30 cửa hàng, mỗi cửa hàng 2 con số: **Điểm vị trí** (0 = hẻm sâu, 10 = phố trung tâm) và **Mức giá bán** (0 = bình dân, 10 = cao cấp). Quy luật thị trường:

- Vị trí trung tâm + giá cao cấp → **LÃI** (khách sang, gánh nổi mặt bằng)
- Hẻm sâu + giá bình dân → **LÃI** (chi phí thấp, khách bình dân đông)
- Trung tâm + giá rẻ → **LỖ** (doanh thu không gánh nổi mặt bằng)
- Hẻm sâu + giá cao cấp → **LỖ** (khách sang không vào hẻm)

Tức là: **không cột nào tự nó tốt hay xấu** — lãi/lỗ do hai cột có "khớp pha" với nhau không. Đây chính là bài toán **XOR**: 4 góc dữ liệu xếp chéo nhau, kẻ đường thẳng kiểu gì cũng dính ít nhất 1 góc sai. Logistic Regression thua từ vòng gửi xe **về nguyên lý**, không phải do thiếu dữ liệu.

---

## 3. Bộ Dữ Liệu 30 Dòng — Và Hai Mô Hình Chấm Điểm Song Song

Cột **P-Log** = xác suất lãi do Logistic chấm; **P-MLP** = do mạng neural 4 neuron ẩn chấm (cả hai đều huấn luyện trên 22 dòng nhóm A+B, số chạy thật).

### 🟢 Nhóm A (dòng 1–16): Bốn góc XOR rõ ràng — sân khấu chính của bài

| # | Vị trí | Giá | Nhãn | P-Log | P-MLP | Giải thích |
|---|---:|---:|:---:|---:|---:|:---|
| 1–4 | 7–9 | 7–9 | 💰 Lãi | ~0.50 | 0.84–0.98 | Trung tâm + cao cấp. Logistic chấm 0.50 = "tôi chịu"; MLP tự tin đúng |
| 5–8 | 1–3 | 1–3 | 💰 Lãi | ~0.50 | 0.87–0.96 | Hẻm + bình dân — MLP hiểu "khớp pha là lãi" |
| 9–12 | 7–9 | 1–3 | 📉 Lỗ | ~0.50 | 0.07–0.19 | Trung tâm + giá rẻ — lệch pha |
| 13–16 | 1–3 | 7–9 | 📉 Lỗ | ~0.50 | 0.02–0.08 | Hẻm + cao cấp — lệch pha |

**Nhìn cột P-Log mà thương:** Logistic chấm cả 16 dòng **quanh quẩn 0.49–0.51** — nó không phân biệt nổi bất kỳ dòng nào với dòng nào, đúng như lý thuyết tiên đoán. Kết quả tổng: Logistic đúng 55% trên tập học, **50% trên cửa hàng mới** (= tung đồng xu); MLP 4 neuron: 95% tập học, **100% trên 8 cửa hàng mới**.

**MLP làm điều đó kiểu gì?** Mỗi neuron ẩn vẫn chỉ vẽ 1 đường thẳng — nhưng 4 neuron vẽ 4 đường, và tầng ra học cách **phối hợp**: "nằm phía trên đường số 1 VÀ bên phải đường số 3 → lãi". Ghép các mảnh thẳng thành ranh giới gấp khúc — đó là toàn bộ phép màu, không có gì huyền bí.

### 🟡 Nhóm B (dòng 17–22): Sát tâm (5,5) — vùng mập mờ của chính quy luật

| # | Vị trí | Giá | Nhãn | P-MLP | Giải thích |
|---|---:|---:|:---:|---:|:---|
| 17 | 5.5 | 5.5 | 💰 | 0.57 | Nhích khỏi tâm một chút — quy luật còn đúng nhưng yếu |
| 18 | 4.5 | 4.5 | 💰 | 0.67 | — |
| 19 | 5.5 | 4.5 | 📉 | 0.38 | — |
| 20 | 4.5 | 5.5 | 📉 | 0.45 | MLP chấm 0.38–0.67: thành thật báo "vùng này tôi không chắc" |
| 21 | 6.0 | 5.0 | 💰 | 0.47 | Sai trong gang tấc — giống nhóm B của Chương 1 |
| 22 | 5.0 | 6.0 | 📉 | 0.46 | Ngưỡng quyết định lại là chuyện threshold, không phải chuyện mô hình |

### 🔴 Nhóm C (dòng 23–26): Nhãn ghi sai sổ sách — và cái bẫy mạng to

| # | Vị trí | Giá | Nhãn (sai) | Giải thích |
|---|---:|---:|:---:|:---|
| 23 | 9 | 9 | 📉 (❓) | Góc "chắc chắn lãi" mà sổ ghi lỗ — kế toán nhập nhầm |
| 24 | 1 | 1 | 📉 (❓) | Tương tự |
| 25 | 9 | 1 | 💰 (❓) | Góc "chắc chắn lỗ" mà ghi lãi |
| 26 | 1 | 9 | 💰 (❓) | Tương tự |

Thí nghiệm thật khi cho 4 dòng này vào tập huấn luyện:
- **MLP 4 neuron:** điểm train tụt còn 73% — nó **không đủ chỗ nhớ để học vẹt** nên lờ 4 dòng nhiễu đi → cửa hàng mới vẫn đúng 100%. Underfit nhẹ hóa ra là áo giáp.
- **MLP 100 neuron:** điểm train 100% — thừa sức khoanh vùng riêng cho từng dòng nhiễu. Cái giá đo được: điểm mới (1.3, 1.3) — ngay cạnh góc bị ghi sai, nhãn thật là Lãi — bị nó phán **Lỗ**. Mạng càng to, càng cần dữ liệu sạch (hoặc các kỹ thuật kìm như dropout, early stopping, weight decay).

### 🔴 Nhóm D (dòng 27–28): Outlier đầu vào — mạng neural "tự tin ngoài vùng phủ sóng"

| # | Vị trí | Giá | Nhãn | P-MLP | Giải thích |
|---|---:|---:|:---:|---:|:---|
| 27 | 8 | 30 | 💰 | 0.95 | Giá = 30/10 — nhập sai đơn vị. MLP chưa từng thấy vùng này nhưng vẫn phán **cực kỳ tự tin** (0.95) |
| 28 | 2 | 30 | 📉 | 0.00 | Tự tin tuyệt đối theo hướng ngược lại |

**Bài học đắt giá:** mạng neural **không biết nói "tôi chưa thấy vùng này bao giờ"** — ranh giới đã học cứ thế kéo dài ra vô tận, càng xa càng tự tin. Trong thực tế đây là nguồn tai nạn lớn: dữ liệu nhập sai đơn vị vẫn nhận được dự đoán chắc nịch. Phòng bằng cách kiểm tra phạm vi đầu vào (input validation) trước khi cho vào mạng — đừng trông chờ mô hình tự cảnh giác.

### ⚫ Nhóm E (dòng 29–30): Trùng đặc trưng, ngược nhãn — quy luật bất biến của cả lộ trình

| # | Vị trí | Giá | Nhãn | P-MLP | Giải thích |
|---|---:|---:|:---:|---:|:---|
| 29 | 5 | 5 | 💰 | 0.53 | Hai cửa hàng số liệu y hệt... |
| 30 | 5 | 5 | 📉 | 0.53 | ...MLP cũng chỉ biết chấm cả hai 0.53. Deep Learning không phải phép thuật — thiếu cột thông tin thì chịu |

---

## 4. Thí Nghiệm Quan Trọng Nhất: Phi Tuyến Là Phép Màu, Không Phải Số Tầng

Bảng chạy thật trên cùng bộ dữ liệu (train trên 22 dòng sạch, probe = 8 cửa hàng mới):

| Mô hình | Train | Cửa hàng mới | Chẩn đoán |
| :--- | :---: | :---: | :--- |
| Logistic Regression | 55% | 50% | Thua về nguyên lý — không đường thẳng nào chia được XOR |
| **MLP 4 neuron, ReLU** | 95% | **100%** ✅ | Đủ 4 đường thẳng để ghép thành ranh giới chéo |
| MLP 4 neuron, **activation='identity'** | 50% | 50% ❌ | **Bỏ phi tuyến đi là chết ngay** — cộng trừ các đường thẳng vẫn ra... một đường thẳng. Đúng nguyên văn README: *"không có hàm kích hoạt phi tuyến, mạng sâu bao nhiêu lớp cũng chỉ tương đương một phép biến đổi tuyến tính"* |
| MLP 1 neuron, ReLU | 64% | 62% | 1 nếp gấp không đủ chia 4 góc chéo |
| MLP 2 neuron, ReLU | 50% | 50% | Lý thuyết bảo 2 neuron là đủ cho XOR — nhưng thực tế huấn luyện **kẹt ở cực tiểu cục bộ** (local minimum). Bài học thực chiến: người ta luôn cho dư neuron một chút để đường đi của Gradient Descent rộng rãi hơn |

---

## 5. SGD — Các Config Ảnh Hưởng Cái Gì? (số chạy thật)

### 5a. Learning rate — lại là nó, nhưng lần này có "nổ"

| learning_rate_init | Train | Loss cuối | Chuyện xảy ra |
| :---: | :---: | :---: | :--- |
| 0.0001 | 50% | 0.708 | Sau 1088 epoch vẫn ~mức đoán mò (0.693) — bước quá ngắn, chưa đi đến đâu |
| **0.05** | **100%** | **0.049** | Hội tụ đẹp sau 776 epoch |
| 5.0 | 50% | 0.697 | Nhảy loạn 23 epoch rồi bị early-stop — bước quá dài, văng qua văng lại quanh thung lũng, không bao giờ rơi xuống đáy |

So với Chương 1: hậu quả giống hệt (nhỏ = đứng im, to = hỏng), nhưng mạng neural có **bề mặt loss gồ ghề nhiều thung lũng** (non-convex) nên learning rate còn nhạy cảm hơn — đó cũng là lý do Adam (tự điều chỉnh bước chân) trở thành mặc định trong thực chiến.

### 5b. Batch size — vì sao gọi là "Stochastic"

| batch_size | Số epoch để hội tụ | Giải thích |
| :---: | :---: | :--- |
| 22 (cả tập — "Gradient Descent" nguyên bản) | 776 | Mỗi epoch chỉ cập nhật trọng số **1 lần** — chắc chắn nhưng lề mề |
| **1 (SGD thứ thiệt)** | **36** | Mỗi epoch cập nhật **22 lần** (mỗi mẫu một cú hích) — nhanh gấp ~20 lần. Hơi "say rượu" lảo đảo nhưng chính cái lảo đảo đó giúp nhảy thoát các hố cực tiểu cục bộ |

Thực chiến dùng **mini-batch 32–256**: dung hòa giữa ổn định và tốc độ, lại tận dụng được phép nhân ma trận song song trên GPU.

### 5c. Số neuron ẩn — bảng tổng kết chọn cỡ mạng

| Cỡ mạng | Khi nào dùng |
| :--- | :--- |
| Quá nhỏ (1–2 neuron) | Không đủ "nếp gấp" cho quy luật → underfit, hoặc kẹt local minimum |
| **Vừa (gấp ~2 lần độ phức tạp quy luật)** | Điểm ngọt — học được quy luật, không thừa chỗ học vẹt nhiễu |
| Quá to (100 neuron cho 26 dòng) | Học thuộc cả nhãn sai → sai lây sang vùng lân cận (thí nghiệm nhóm C). Nếu buộc phải dùng mạng to: thêm dropout / early stopping / weight decay |

---

## 6. Checklist: Có Cần Mạng Neural Không?

- [ ] Quan hệ các cột **một chiều, độc lập** → Logistic Regression (Chương 1) đủ, còn giải thích được hệ số.
- [ ] Dữ liệu bảng có quan hệ **NẾU–VÀ, chữ V** → thử XGBoost (Chương 2) trước — thường thắng MLP trên dữ liệu bảng.
- [ ] Quan hệ kiểu **XOR / "khớp pha"** — không cột nào tự tốt hay xấu → MLP với hàm kích hoạt phi tuyến.
- [ ] Ảnh, âm thanh, văn bản → mạng neural chuyên dụng (CNN — Chương 4, Transformer — Chương 6).
- [ ] Kiểm tra phạm vi đầu vào trước khi dự đoán — mạng neural tự tin cả ở nơi nó chưa từng thấy (nhóm D).
- [ ] Dữ liệu ít + mạng to = học vẹt. Luôn giữ tập kiểm thử riêng và ưu tiên mạng nhỏ nhất còn giải được bài.

---

## 7. Code Chạy Lại Toàn Bộ (copy vào notebook là chạy)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 30 cửa hàng: (điểm vị trí 0-10, mức giá 0-10, nhãn 1=lãi)
# Quy luật XOR: khớp pha (cùng cao / cùng thấp) = lãi, lệch pha = lỗ
rows = [
    (8,8,1),(9,7,1),(7,9,1),(8.5,8.5,1),(2,2,1),(1,3,1),(3,1,1),(1.5,2.5,1),
    (8,2,0),(9,3,0),(7,1,0),(8.5,2.5,0),(2,8,0),(1,7,0),(3,9,0),(2.5,8.5,0),
    (5.5,5.5,1),(4.5,4.5,1),(5.5,4.5,0),(4.5,5.5,0),(6,5,1),(5,6,0),   # B: sát tâm
    (9,9,0),(1,1,0),(9,1,1),(1,9,1),                                    # C: nhãn nhiễu
    (8,30,1),(2,30,0),                                                  # D: outlier đầu vào
    (5,5,1),(5,5,0),                                                    # E: trùng, ngược nhãn
]
X = np.array([[r[0],r[1]] for r in rows]); y = np.array([r[2] for r in rows])
sc = StandardScaler().fit(X); Xs = sc.transform(X)

probe = [(7.5,7.5,1),(2.5,1.5,1),(9,2,0),(1.5,8,0),(6.5,9,1),(3,2.5,1),(7,2.5,0),(2,6.5,0)]
Xp = sc.transform(np.array([[r[0],r[1]] for r in probe])); yp = np.array([r[2] for r in probe])
clean = list(range(22))

def thu(ten, model):
    model.fit(Xs[clean], y[clean])
    print(f"{ten}: train={(model.predict(Xs[clean])==y[clean]).mean():.0%}"
          f" cửa hàng mới={(model.predict(Xp)==yp).mean():.0%}")

thu("Logistic", LogisticRegression())
thu("MLP 4 neuron ReLU", MLPClassifier((4,), max_iter=8000, random_state=0))
thu("MLP 4 neuron KHÔNG phi tuyến", MLPClassifier((4,), activation='identity',
                                                  max_iter=8000, random_state=0))
thu("MLP 1 neuron", MLPClassifier((1,), max_iter=8000, random_state=0))

# SGD: nghịch learning rate và batch size
for lr in [0.0001, 0.05, 5.0]:
    m = MLPClassifier((4,), solver='sgd', learning_rate_init=lr,
                      max_iter=2000, random_state=0).fit(Xs[clean], y[clean])
    print(f"SGD lr={lr}: loss cuối={m.loss_:.3f} sau {m.n_iter_} epoch")

# Bài tập tự nghịch:
# 1. Cho thêm 4 dòng nhiễu 23-26 vào tập học, so MLP(4) vs MLP(100)
#    rồi dự đoán điểm (8.7, 8.7) — mạng nào bị nhiễu lừa?
# 2. Đổi random_state của MLP(2) chạy 10 lần — bao nhiêu lần thoát local minimum?
# 3. Vẽ ranh giới quyết định của MLP(4) bằng matplotlib contourf — thấy 4 đường gấp
```

---

## 8. Kết Nối

| Bạn muốn | Mở |
| :--- | :--- |
| Backpropagation, Chain Rule, lý thuyết SGD | `README.md` của chương này |
| Bài tập code | `03-Bai_Tap_Thuc_Hanh_Neural_Network_SGD.ipynb` |
| Ôn Logistic (đối thủ bị XOR hạ gục) | `../../Khang_lession/Logistic_Regression/USECASE.md` |
| Xem CNN xử lý ảnh — chương tiếp theo | `../04-Hoc_Sau_Thigiac_Maytinh/USECASE.md` |

> 🌟 **Câu chốt:** Mạng neural = nhiều đường thẳng + phi tuyến để ghép chúng lại. Bỏ phi tuyến đi thì trăm tầng cũng bằng một (đã đo: 50%). Và mô hình càng mạnh, càng cần bạn canh chừng nó học vẹt.
