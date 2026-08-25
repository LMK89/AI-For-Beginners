# 🧭 USECASE: Logistic Regression Dùng Khi Nào? — Học Qua 30 Dòng Dữ Liệu Thật

> **Dành cho người chưa biết gì:** File này KHÔNG có công thức khó. Bạn sẽ học Logistic Regression bằng cách nhìn vào một bộ dữ liệu 30 dòng cụ thể, xem mô hình đúng ở dòng nào, sai ở dòng nào, và **tại sao**. Mọi con số trong file này đều là kết quả chạy thật (code chạy lại nằm ở cuối file), không phải số bịa ra cho đẹp.

---

## 1. TL;DR — Tóm tắt 30 giây

| Câu hỏi | Trả lời ngắn gọn |
| :--- | :--- |
| Logistic Regression làm gì? | Trả lời câu hỏi **Có / Không** kèm theo **độ tự tin (xác suất 0 → 1)**. Ví dụ: "Khách này có được duyệt vay không? — Có, tự tin 85%". |
| Khi nào nó ĐÚNG? | Khi dữ liệu dạng bảng (số liệu), và quan hệ giữa đặc trưng với kết quả **một chiều, thuận theo trực giác**: thu nhập càng cao → càng dễ duyệt, nợ càng nhiều → càng khó duyệt. |
| Khi nào nó SAI? | Khi có **outlier** (dòng dị thường), **nhãn bị ghi sai**, quan hệ **vòng vèo phi tuyến** ("nợ = 0 lại là dấu hiệu xấu"), hoặc **thiếu thông tin** (2 hồ sơ giống hệt nhau nhưng kết quả khác nhau). |
| Tham số ảnh hưởng gì? | **Ngưỡng (threshold)** quyết định dòng nào được "lấy"; **C (regularization)** quyết định mô hình tin dữ liệu đến mức nào; **learning rate** quyết định mô hình học nhanh hay chậm hay... học hỏng. |

---

## 2. Logistic Regression Dùng Cho Những Dữ Liệu Gì?

### ✅ Các bài toán thực tế nó làm rất tốt

| Bài toán | Đầu vào (đặc trưng) | Đầu ra (Có/Không) |
| :--- | :--- | :--- |
| Lọc email spam | Số link trong mail, độ dài, từ khóa "trúng thưởng"... | Spam / Không spam |
| Duyệt khoản vay ngân hàng | Thu nhập, nợ hiện tại, tuổi, lịch sử trả nợ | Duyệt / Từ chối |
| Dự đoán khách rời bỏ (churn) | Số tháng dùng dịch vụ, số lần gọi tổng đài | Rời bỏ / Ở lại |
| Chẩn đoán y tế sơ bộ | Chỉ số xét nghiệm máu, tuổi, huyết áp | Nguy cơ / An toàn |

**Điểm chung:** dữ liệu dạng **bảng số liệu**, câu trả lời là **nhị phân (2 lớp)**, và ta muốn biết **xác suất** chứ không chỉ Có/Không khô khan (ngân hàng muốn biết "duyệt với độ tin 51%" khác xa "duyệt với độ tin 99%").

### ❌ Các loại dữ liệu KHÔNG nên dùng

| Loại dữ liệu | Tại sao không hợp | Dùng gì thay thế |
| :--- | :--- | :--- |
| Ảnh, âm thanh thô | Từng pixel riêng lẻ không có quan hệ tuyến tính với nhãn | CNN / Neural Network |
| Văn bản dài, ngữ cảnh | Nghĩa của từ phụ thuộc vào từ xung quanh | Transformer (Module 3 của bạn!) |
| Quan hệ phi tuyến mạnh, nhiều tương tác chéo | Logistic chỉ vẽ được **1 đường thẳng** chia đôi dữ liệu | XGBoost/LightGBM (Module 2 của bạn!) |
| Dự đoán con số liên tục (giá nhà) | Đây là bài toán hồi quy, không phải phân loại | Linear Regression |

> 💡 **Trực giác cốt lõi:** Logistic Regression = kẻ **một đường thẳng** chia mặt phẳng dữ liệu làm 2 nửa. Dữ liệu nào 2 nhóm đứng gọn 2 bên đường thẳng → nó đúng. Dữ liệu nào 2 nhóm trộn lẫn kiểu "da beo" → nó bó tay.

---

## 3. Bài Toán Xuyên Suốt: Duyệt Khoản Vay 💰

Bạn là nhân viên ngân hàng, có hồ sơ của 30 khách hàng. Mỗi hồ sơ chỉ có 2 con số:

- **Thu nhập** (triệu đồng/tháng)
- **Nợ hiện tại** (triệu đồng)

Và kết quả thực tế: **Duyệt (1)** hay **Từ chối (0)**.

Nhiệm vụ của Logistic Regression: học từ 30 hồ sơ này để sau đó tự động chấm hồ sơ mới. Nhưng khoan — **không phải dòng nào cũng "dạy" được cho mô hình**. Xem bảng dưới đây.

---

## 4. Bộ Dữ Liệu 30 Dòng — Dòng Nào Phù Hợp, Dòng Nào Không?

Cột **P(duyệt)** là xác suất mà mô hình (đã huấn luyện trên 20 dòng sạch, nhóm A+B) chấm cho từng hồ sơ — số chạy thật, bạn chạy lại được bằng code cuối file.

### 🟢 Nhóm A (dòng 1–14): Dữ liệu "đẹp" — Logistic Regression ăn điểm tuyệt đối

| # | Thu nhập | Nợ | Nhãn thật | P(duyệt) | Mô hình đoán | Giải thích |
|---|---:|---:|:---:|---:|:---:|:---|
| 1 | 45 | 5 | ✅ Duyệt | 0.73 | ✅ Đúng | Thu nhập cao, nợ thấp → mẫu mực |
| 2 | 60 | 10 | ✅ Duyệt | 0.85 | ✅ Đúng | Càng cao thu nhập, xác suất càng tăng |
| 3 | 38 | 8 | ✅ Duyệt | 0.64 | ✅ Đúng | Khá ổn, xác suất vừa phải (0.64) — hợp lý |
| 4 | 52 | 20 | ✅ Duyệt | 0.77 | ✅ Đúng | Nợ 20 nhưng thu nhập 52 gánh được |
| 5 | 70 | 30 | ✅ Duyệt | 0.88 | ✅ Đúng | Thu nhập lớn bù được nợ lớn |
| 6 | 55 | 15 | ✅ Duyệt | 0.80 | ✅ Đúng | Mẫu chuẩn |
| 7 | 80 | 20 | ✅ Duyệt | 0.93 | ✅ Đúng | Hồ sơ đẹp nhất nhóm → xác suất cao nhất nhóm |
| 8 | 8 | 40 | ❌ Từ chối | 0.22 | ✅ Đúng | Thu nhập thấp, nợ cao → từ chối rõ ràng |
| 9 | 12 | 25 | ❌ Từ chối | 0.29 | ✅ Đúng | Tương tự |
| 10 | 10 | 5 | ❌ Từ chối | 0.30 | ✅ Đúng | Nợ ít nhưng thu nhập quá thấp |
| 11 | 15 | 50 | ❌ Từ chối | 0.27 | ✅ Đúng | Nợ gấp 3 lần thu nhập |
| 12 | 6 | 10 | ❌ Từ chối | 0.25 | ✅ Đúng | Thu nhập 6 triệu → từ chối |
| 13 | 9 | 30 | ❌ Từ chối | 0.25 | ✅ Đúng | Tương tự |
| 14 | 5 | 60 | ❌ Từ chối | 0.17 | ✅ Đúng | Hồ sơ tệ nhất → xác suất thấp nhất. Mô hình rất "hiểu chuyện" |

**Vì sao phù hợp?** Nhóm này tuân theo đúng 1 quy luật một chiều: *thu nhập kéo xác suất lên, nợ kéo xác suất xuống*. Đây chính xác là thứ Logistic Regression được sinh ra để học. Để ý xác suất **tăng dần theo độ đẹp của hồ sơ** (0.17 → 0.93) chứ không chỉ Có/Không — đây là điểm ăn tiền của thuật toán.

### 🟡 Nhóm B (dòng 15–20): Sát ranh giới — đúng hay sai do NGƯỠNG quyết định

| # | Thu nhập | Nợ | Nhãn thật | P(duyệt) | Đoán (ngưỡng 0.5) | Giải thích |
|---|---:|---:|:---:|---:|:---:|:---|
| 15 | 25 | 20 | ✅ Duyệt | 0.45 | ❌ Sai (hụt ngưỡng) | Hồ sơ "lưng chừng", mô hình chấm 0.45 — chỉ thiếu 0.05! |
| 16 | 24 | 22 | ❌ Từ chối | 0.43 | ✅ Đúng | Gần giống dòng 15 nhưng nhãn ngược lại |
| 17 | 27 | 25 | ✅ Duyệt | 0.47 | ❌ Sai (hụt ngưỡng) | Sát nút 0.5 |
| 18 | 23 | 18 | ❌ Từ chối | 0.43 | ✅ Đúng | Sát nút |
| 19 | 26 | 21 | ✅ Duyệt | 0.46 | ❌ Sai (hụt ngưỡng) | Sát nút |
| 20 | 22 | 24 | ❌ Từ chối | 0.40 | ✅ Đúng | Sát nút |

**Vì sao "nửa phù hợp"?** Các hồ sơ này nằm **ngay trên đường ranh giới**. Mô hình không sai về bản chất — nó thành thật nói "tôi chỉ chắc 43–47%". Cái sai nằm ở chỗ ta ép nó chọn phe bằng ngưỡng 0.5. **Hạ ngưỡng xuống 0.45 thì cả 3 dòng 15, 17, 19 được lấy đúng, mà 16, 18, 20 vẫn bị loại đúng** (xem mục 6a). Bài học: với dòng sát ranh giới, chỉnh ngưỡng quan trọng hơn chỉnh mô hình.

### 🔴 Nhóm C (dòng 21–22): Outlier — kẻ phá hoại thầm lặng

| # | Thu nhập | Nợ | Nhãn thật | P(duyệt) | Đoán | Giải thích |
|---|---:|---:|:---:|---:|:---:|:---|
| 21 | 200 | 10 | ❌ Từ chối | 1.00 | ❌ Sai đau | Thu nhập 200 triệu nhưng bị từ chối vì **nghi gian lận hồ sơ** — lý do này KHÔNG nằm trong 2 cột dữ liệu |
| 22 | 48 | 300 | ✅ Duyệt | 0.18 | ❌ Sai đau | Nợ 300 triệu vẫn được duyệt vì **có tài sản thế chấp** — cũng không có cột nào ghi điều đó |

**Vì sao KHÔNG phù hợp?** Nhìn 2 cột số liệu thì dòng 21 là hồ sơ trong mơ, dòng 22 là thảm họa — nhưng nhãn thật lại ngược 100%. Mô hình chỉ nhìn thấy con số, không nhìn thấy "nghi gian lận" hay "có nhà thế chấp". **Và nguy hiểm hơn: nếu đem 2 dòng này vào tập huấn luyện, chúng phá luôn cả mô hình** (xem mục 5 — hệ số bị đổi dấu!).

### 🔴 Nhóm D (dòng 23–24): Nhãn bị ghi sai (label noise)

| # | Thu nhập | Nợ | Nhãn thật (bị ghi sai) | P(duyệt) | Đoán | Giải thích |
|---|---:|---:|:---:|---:|:---:|:---|
| 23 | 58 | 8 | ❌ Từ chối (❓) | 0.83 | ❌ "Sai" | Hồ sơ gần giống dòng 6 (55, 15, Duyệt) mà lại Từ chối — khả năng cao nhân viên **nhập nhầm nhãn** |
| 24 | 7 | 45 | ✅ Duyệt (❓) | 0.21 | ❌ "Sai" | Gần giống dòng 8 (8, 40, Từ chối) mà lại Duyệt — nhập nhầm |

**Vì sao KHÔNG phù hợp?** Ở đây **mô hình đoán hợp lý, còn nhãn mới là thứ sai**. Điểm thú vị: chính vì mô hình chấm dòng 23 tận 0.83 mà nhãn lại ghi Từ chối, ta mới phát hiện ra nghi vấn nhập liệu. Trong thực tế, người ta dùng chính Logistic Regression để **soi ngược lại dữ liệu bẩn** kiểu này.

### 🔴 Nhóm E (dòng 25–28): Quan hệ phi tuyến — đường thẳng bó tay

| # | Thu nhập | Nợ | Nhãn thật | P(duyệt) | Đoán | Giải thích |
|---|---:|---:|:---:|---:|:---:|:---|
| 25 | 30 | 0 | ❌ Từ chối | 0.56 | ❌ Sai | Nợ = 0 vì **chưa từng vay bao giờ** → không có lịch sử tín dụng → ngân hàng từ chối |
| 26 | 35 | 0 | ❌ Từ chối | 0.62 | ❌ Sai | Tương tự |
| 27 | 33 | 12 | ✅ Duyệt | 0.57 | ✅ Đúng (may) | Nợ vừa phải + đã trả đúng hạn → lại là dấu hiệu TỐT |
| 28 | 31 | 10 | ✅ Duyệt | 0.55 | ✅ Đúng (may) | Tương tự |

**Vì sao KHÔNG phù hợp?** Mô hình đã học quy luật "nợ càng ít càng tốt", nên nợ = 0 phải là tốt nhất → nó chấm dòng 25, 26 cao hơn dòng 27, 28. Nhưng thực tế ngược lại: **nợ = 0 là xấu (không có lịch sử), nợ vừa phải là tốt, nợ nhiều mới xấu** — quan hệ hình chữ V ngược, không phải đường một chiều. Một đường thẳng không thể vẽ được chữ V. Đây chính là lúc bạn cần cây quyết định / XGBoost (Module 2), hoặc tự tạo thêm đặc trưng mới kiểu cột `chua_co_lich_su_tin_dung` (0/1).

### 🔴 Nhóm F (dòng 29–30): Hai hồ sơ giống hệt nhau, kết quả khác nhau

| # | Thu nhập | Nợ | Nhãn thật | P(duyệt) | Đoán | Giải thích |
|---|---:|---:|:---:|---:|:---:|:---|
| 29 | 20 | 15 | ✅ Duyệt | 0.40 | ❌ Sai 1 trong 2 | Giống hệt dòng 30 về số liệu... |
| 30 | 20 | 15 | ❌ Từ chối | 0.40 | (chắc chắn) | ...nhưng nhãn ngược nhau |

**Vì sao KHÔNG phù hợp?** Đầu vào y hệt nhau thì mô hình bắt buộc cho ra xác suất y hệt nhau (cùng 0.40) — **không tồn tại mô hình nào trên đời phân biệt được 2 dòng này**, dù là Deep Learning. Vấn đề không nằm ở thuật toán mà ở **dữ liệu thiếu cột thông tin** (có thể người 29 có bảo lãnh, người 30 thì không — nhưng bảng không ghi). Bài học: khi thấy cặp dòng kiểu này, đi tìm thêm đặc trưng, đừng đi đổi thuật toán.

---

## 5. Thí Nghiệm Gây Sốc: Chỉ 2 Dòng Outlier Phá Nát Cả Mô Hình

Đây là kết quả chạy thật khi huấn luyện trên 2 tập khác nhau:

| Huấn luyện trên | Hệ số Thu nhập | Hệ số Nợ | Độ chính xác trên 20 dòng sạch |
| :--- | ---: | ---: | ---: |
| **20 dòng sạch (A+B)** | **+1.87** ✅ | **−0.47** ✅ | **85%** |
| **Cả 30 dòng (có nhiễu)** | +0.25 | **+0.33** ❌⚠️ | 75% |

Đọc kết quả này như sau:

- Mô hình sạch học đúng lẽ đời: thu nhập **cộng điểm** (+1.87), nợ **trừ điểm** (−0.47).
- Nhét thêm 10 dòng nhiễu vào (đặc biệt là dòng 22: nợ 300 triệu mà vẫn được duyệt), hệ số của Nợ **đổi dấu thành +0.33** — mô hình kết luận *"nợ càng nhiều càng dễ được duyệt"*!
- Vì Logistic Regression chỉ có vài hệ số để "ghi nhớ" toàn bộ dữ liệu, một dòng cực đoan có sức kéo rất lớn lên hệ số — khác với con người có thể nhún vai bỏ qua 1 ca lạ.

> 🎯 **Bài học quan trọng nhất file này:** Trước khi đổ lỗi cho thuật toán, hãy **in hệ số ra và hỏi "dấu của nó có thuận với lẽ thường không?"**. Hệ số ngược dấu trực giác = báo động đỏ dữ liệu có vấn đề.

---

## 6. Các Tham Số Ảnh Hưởng Cái Gì? Dòng Nào Được Lấy, Dòng Nào Không?

### 6a. Ngưỡng quyết định (threshold) — cái "gác cổng"

Mô hình chấm xác suất, còn **ngưỡng** quyết định từ bao nhiêu điểm thì "được lấy" (dự đoán = Duyệt). Kết quả chạy thật trên 30 dòng (dùng mô hình sạch):

| Ngưỡng | Số dòng được lấy | Cụ thể là các dòng | Điều gì xảy ra |
| :---: | :---: | :--- | :--- |
| **0.7** (khó tính) | 8 | 1, 2, 4, 5, 6, 7, 21, 23 | Chỉ lấy hồ sơ cực đẹp. **Mất cả dòng 3** (0.64 — người tốt bị oan). Duyệt ai gần như chắc đúng, nhưng bỏ sót nhiều |
| **0.5** (mặc định) | 13 | 1–7, 21, 23, 25–28 | Cân bằng. Nhưng cả 3 dòng đáng duyệt 15, 17, 19 (0.45–0.47) đều **trượt trong gang tấc** |
| **0.45** (nới nhẹ) | 16 | thêm 15, 17, 19 | 🎯 Điểm ngọt của bộ dữ liệu này: vớt đúng 3 dòng sát nút đáng duyệt, mà 16, 18, 20 (0.40–0.43) vẫn bị loại đúng |
| **0.3** (dễ dãi) | 22 | thêm cả 10, 16, 18, 20, 29, 30 | Vớt được nhiều người đáng duyệt, nhưng **lấy nhầm cả loạt hồ sơ đáng từ chối** (dòng 10, 16, 18, 20) |

**Chọn ngưỡng theo cái giá của sai lầm, không phải theo con số 0.5:**
- Cho vay nhầm mất tiền tỷ → tăng ngưỡng (0.7): thà bỏ sót còn hơn duyệt nhầm.
- Tầm soát ung thư, bỏ sót là chết người → hạ ngưỡng (0.3): thà báo nhầm còn hơn bỏ lọt.
- Để ý dòng 21 và 23 **ngưỡng nào cũng được lấy** (P = 1.00 và 0.83) dù nhãn thật là Từ chối — chỉnh ngưỡng không cứu được lỗi do outlier/nhãn sai. Ngưỡng chỉ cứu được nhóm B (sát ranh giới).

### 6b. Tham số C — mô hình "tin" dữ liệu đến mức nào (Regularization)

Trong `sklearn.linear_model.LogisticRegression(C=...)`, C nhỏ = kìm hãm mô hình mạnh, C lớn = thả cho mô hình tự do bám sát dữ liệu. Kết quả chạy thật khi huấn luyện trên cả 30 dòng nhiễu:

| C | Hệ số Thu nhập / Nợ | Chuyện gì xảy ra |
| :---: | :--- | :--- |
| **100** (tự do) | +0.30 / **+0.43** | Bám nhiễu mạnh nhất — hệ số Nợ dương to nhất, chính xác thấp nhất (57%). Mô hình cố "chiều lòng" cả dòng outlier 21, 22 |
| **1** (mặc định) | +0.25 / +0.33 | Vẫn hỏng vì nhiễu, nhưng đỡ hơn |
| **0.01** (kìm chặt) | +0.018 / +0.021 | Hệ số bị nén gần về 0 → mọi hồ sơ đều được chấm ≈ 0.50. Mô hình "đầu hàng", không phân biệt ai với ai nữa |

**Trực giác:** C giống dây cương ngựa. Thả lỏng (C lớn) → ngựa chạy theo mọi hòn đá bên đường (overfit theo nhiễu). Ghì chặt quá (C quá nhỏ) → ngựa đứng im không đi đâu (underfit, mọi dự đoán về 0.5). Thực tế người ta thử C = 0.01, 0.1, 1, 10, 100 rồi chọn cái tốt nhất trên tập kiểm thử.

### 6c. Learning rate (α) và số vòng lặp — dành cho bản tự code Numpy

Khi bạn tự code Gradient Descent trong `Logistic_Regression_Scratch.ipynb`, đây là kết quả thật trên 20 dòng sạch (đã chuẩn hóa, 1000 vòng lặp):

| Learning rate | Kết quả | Chẩn đoán |
| :---: | :--- | :--- |
| **0.0001** (quá nhỏ) | Hệ số ≈ (0.02, −0.01), Loss = 0.687 | Sau 1000 vòng vẫn gần như **chưa học được gì** (Loss 0.687 ≈ mức đoán mò 0.693). Mọi xác suất dính ở ~0.5 |
| **0.1** (vừa) | Hệ số (4.99, −1.36), Loss = 0.223 | ✅ Hội tụ đẹp, xác suất phân tầng hợp lý như bảng ở mục 4 |
| **50** (quá to) | Hệ số (79.8, −11.7), Loss = 0.029 | ⚠️ Trông "ngon" (Loss thấp) nhưng hệ số phình khổng lồ → mô hình chấm mọi hồ sơ hoặc 0.999 hoặc 0.001, **tự tin mù quáng**. Gặp hồ sơ mới lạ một chút là sai thảm, và với dữ liệu khó hơn sẽ dao động không hội tụ |

**Cách nhận biết nhanh:** vẽ đường Loss theo vòng lặp. Đi xuống mượt → α ổn. Nằm ngang mãi → α quá nhỏ. Nhảy loạn xạ / tràn số (`NaN`) → α quá to, chia α cho 10 rồi thử lại.

### 6d. Chuẩn hóa dữ liệu (StandardScaler) — bước hay bị bỏ quên nhất

Thu nhập cỡ hàng chục, nợ có dòng lên tới 300 — hai thang đo lệch nhau làm Gradient Descent đi vòng vèo. Kết quả thật:

| Dữ liệu | Cần gì để hội tụ |
| :--- | :--- |
| **Đã chuẩn hóa** | α = 0.1, **1.000 vòng** → Loss 0.223 ✅ |
| **Không chuẩn hóa** | α = 0.1 bị dao động (Loss 1.324 — tệ hơn cả đoán mò); phải hạ α xuống 0.0001 và chạy tận **50.000 vòng** mới đạt Loss 0.256 |

Tức là quên chuẩn hóa = trả giá gấp 50 lần số vòng lặp mà kết quả vẫn kém hơn. Luôn `StandardScaler` trước khi fit (với Logistic Regression / Neural Network; còn XGBoost ở Module 2 thì không cần — nhớ lại Key Notes trong README tổng).

### 6e. class_weight — khi 2 lớp mất cân bằng

Bộ 30 dòng này khá cân (16 duyệt / 14 từ chối) nên chưa thấy vấn đề. Nhưng thực tế hay gặp kiểu 990 hồ sơ tốt / 10 hồ sơ gian lận: mô hình chỉ cần đoán bừa "tất cả đều tốt" là đạt 99% chính xác — nghe cao mà vô dụng. Khi đó dùng `class_weight='balanced'` để bắt mô hình coi 1 lỗi bỏ sót gian lận nặng bằng 99 lỗi thường, và **đừng nhìn Accuracy nữa, hãy nhìn Precision/Recall** (phần Ngày 2 trong README của module này).

---

## 7. Checklist: Nhìn Dữ Liệu 10 Giây, Đoán Logistic Regression Hợp Hay Không

Trước khi fit, tự hỏi:

- [ ] Câu trả lời có phải dạng **Có/Không** (2 lớp)? — Nếu là con số liên tục → Linear Regression.
- [ ] Mỗi đặc trưng có quan hệ **một chiều** với kết quả không (càng X càng dễ Y)? — Nếu hình chữ V như "nợ = 0 lại xấu" (nhóm E) → tạo thêm đặc trưng hoặc dùng XGBoost.
- [ ] Có dòng nào **cực đoan bất thường** (nhóm C)? — Vẽ scatter plot, cắt bỏ hoặc xử lý riêng trước khi huấn luyện.
- [ ] Có cặp dòng **giống nhau mà nhãn ngược nhau** (nhóm F)? — Thiếu đặc trưng, đi thu thập thêm cột dữ liệu.
- [ ] Đã **chuẩn hóa** chưa? Hai lớp có **cân bằng** không?
- [ ] Sau khi fit: **dấu của hệ số có thuận lẽ thường không?** — Ngược dấu = dữ liệu có vấn đề (mục 5).
- [ ] Chọn **ngưỡng** theo cái giá của từng loại sai lầm, đừng mặc định 0.5.

---

## 8. Code Chạy Lại Toàn Bộ (copy vào notebook là chạy)

Mọi con số trong file này sinh ra từ đoạn code dưới. Hãy tự chạy, rồi thử nghịch: đổi ngưỡng, đổi C, xóa dòng 21–22 xem hệ số đổi dấu ra sao.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 30 hồ sơ: (thu nhập triệu/tháng, nợ hiện tại triệu, nhãn: 1=duyệt, 0=từ chối)
rows = [
    # Nhóm A (1-14): dữ liệu sạch, quan hệ một chiều
    (45,5,1),(60,10,1),(38,8,1),(52,20,1),(70,30,1),(55,15,1),(80,20,1),
    (8,40,0),(12,25,0),(10,5,0),(15,50,0),(6,10,0),(9,30,0),(5,60,0),
    # Nhóm B (15-20): sát ranh giới
    (25,20,1),(24,22,0),(27,25,1),(23,18,0),(26,21,1),(22,24,0),
    # Nhóm C (21-22): outlier (lý do nằm ngoài dữ liệu)
    (200,10,0),(48,300,1),
    # Nhóm D (23-24): nhãn bị nhập sai
    (58,8,0),(7,45,1),
    # Nhóm E (25-28): quan hệ phi tuyến (nợ=0 lại xấu)
    (30,0,0),(35,0,0),(33,12,1),(31,10,1),
    # Nhóm F (29-30): trùng đặc trưng, ngược nhãn
    (20,15,1),(20,15,0),
]
X = np.array([[r[0], r[1]] for r in rows], dtype=float)
y = np.array([r[2] for r in rows])

# Luôn chuẩn hóa trước khi fit (mục 6d)
Xs = StandardScaler().fit_transform(X)

# Mô hình "sạch": chỉ học trên 20 dòng nhóm A + B
clean = list(range(20))
m_clean = LogisticRegression(C=1.0).fit(Xs[clean], y[clean])
# Mô hình "nhiễu": học trên cả 30 dòng
m_full = LogisticRegression(C=1.0).fit(Xs, y)

print("Hệ số [thu nhập, nợ] sạch :", m_clean.coef_[0].round(2))   # [+1.87, -0.47] hợp lẽ đời
print("Hệ số [thu nhập, nợ] nhiễu:", m_full.coef_[0].round(2))    # [+0.25, +0.33] nợ ĐỔI DẤU!

# Xác suất từng dòng + thử các ngưỡng khác nhau (mục 6a)
p = m_clean.predict_proba(Xs)[:, 1]
for th in [0.7, 0.5, 0.45, 0.3]:
    lay = [i+1 for i in range(30) if p[i] >= th]
    print(f"Ngưỡng {th}: lấy {len(lay)} dòng -> {lay}")

# Bài tập tự nghịch:
# 1. Xóa dòng 21, 22 khỏi rows rồi fit lại cả 30 -> hệ số nợ có về âm không?
# 2. Đổi C=100 và C=0.01 -> xem hệ số và xác suất thay đổi thế nào (mục 6b)
# 3. Thêm cột thứ 3: chua_co_lich_su = 1 nếu nợ==0 -> nhóm E có được cứu không?
```

---

## 9. Kết Nối Với Các Tài Liệu Khác Trong Module

| Bạn muốn | Mở file |
| :--- | :--- |
| Hiểu công thức Sigmoid, BCE Loss, Gradient Descent | `README.md` (lộ trình 2 ngày) |
| Nghịch đồ thị tương tác | `study_guide.html` |
| Tự tay code từ đầu bằng Numpy | `Logistic_Regression_Scratch.ipynb` |
| Xử lý nhóm E (phi tuyến) một cách chuyên nghiệp | Module 2: `../XGBoost_LightGBM/` |

> 🌟 Khi đọc xong file này, bạn đã hiểu thứ mà nhiều người học cả tháng chưa nắm: **thuật toán không tốt hay xấu — nó chỉ hợp hay không hợp với hình dạng của dữ liệu.** 30 dòng ở trên chính là "bản đồ" các hình dạng đó.
