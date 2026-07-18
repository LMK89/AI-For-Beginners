# 🧭 USECASE: XGBoost/LightGBM Dùng Khi Nào? — Học Qua 30 Dòng Dữ Liệu Thật

> **Dành cho người chưa biết gì:** File này tiếp nối câu chuyện **duyệt khoản vay** trong `../Logistic_Regression/USECASE.md`. Ở bài trước, Logistic Regression bó tay với nhóm dữ liệu "phi tuyến" (nợ = 0 lại là xấu). Bài này ta nâng cấp bộ dữ liệu lên 30 dòng khó hơn, cho Logistic Regression và XGBoost **đấu tay đôi trên từng dòng**, và xem các tham số (`max_depth`, `learning_rate`, `min_child_weight`...) thay đổi kết quả ra sao. Mọi con số đều chạy thật — code ở cuối file.

---

## 1. TL;DR — Tóm tắt 30 giây

| Câu hỏi | Trả lời ngắn gọn |
| :--- | :--- |
| XGBoost/LightGBM làm gì? | Xây **hàng trăm cây quyết định nối tiếp nhau**, cây sau sửa lỗi cây trước. Kết quả: mô hình mạnh nhất hiện nay cho **dữ liệu dạng bảng**. |
| Khi nào nó THẮNG Logistic Regression? | Khi quy luật có dạng **NẾU... VÀ... THÌ...** (tương tác chéo), quan hệ **vòng vèo chữ V**, hoặc dữ liệu có **ô trống (NaN)**. |
| Khi nào nó cũng THUA? | Dữ liệu quá ít (nó **học vẹt** thay vì học quy luật), nhãn bị ghi sai (nó học thuộc luôn cả cái sai), và 2 dòng giống hệt nhau nhãn ngược nhau thì thần thánh cũng chịu. |
| Tham số ảnh hưởng gì? | `max_depth` = độ thông minh tối đa của mỗi cây (sâu quá → học vẹt); `learning_rate` + `n_estimators` = đi từng bước nhỏ nhiều bước hay bước to ít bước; `min_child_weight` = "bỏ qua chi tiết vụn vặt" (chống nhiễu). |

---

## 2. XGBoost/LightGBM Dùng Cho Những Dữ Liệu Gì?

### ✅ Sân nhà của nó — dữ liệu bảng (Tabular)

| Bài toán | Vì sao cây thắng |
| :--- | :--- |
| Chấm điểm tín dụng, duyệt vay | Quy luật ngân hàng vốn là các luật NẾU–THÌ chồng nhau — đúng hình dạng của cây |
| Dự đoán churn, gian lận thẻ | Nhiều tương tác chéo: "gọi tổng đài nhiều **VÀ** mới dùng dưới 3 tháng" mới là tín hiệu xấu |
| Xếp hạng, gợi ý sản phẩm, dự báo doanh số | Dữ liệu SQL/Excel hàng triệu dòng, trăm cột trộn số + danh mục |
| Các cuộc thi Kaggle dữ liệu bảng | Gần như mọi giải nhất tabular một thập kỷ qua đều là XGBoost/LightGBM |

**Ba đặc quyền mà Logistic Regression không có:**
1. **Không cần chuẩn hóa** (StandardScaler) — cây chỉ so sánh ngưỡng (`Nợ > 100?`), không nhân ma trận.
2. **Ăn được ô trống (NaN) trực tiếp** — mỗi nút chia nhánh tự học "hồ sơ thiếu thông tin thì rẽ trái hay rẽ phải có lợi hơn".
3. **Tự bắt được tương tác chéo và quan hệ phi tuyến** — không cần bạn ngồi chế thêm đặc trưng.

### ❌ Khi nào đừng dùng

| Tình huống | Vì sao | Dùng gì |
| :--- | :--- | :--- |
| Ảnh, âm thanh, văn bản | Cây không hiểu cấu trúc không gian/ngữ cảnh | CNN / Transformer (Module 3) |
| Dữ liệu cực ít (vài chục dòng) | Cây đủ sức **học thuộc lòng** toàn bộ → điểm train ảo cao, gặp dữ liệu mới là sập (chính là thí nghiệm mục 5!) | Logistic Regression đơn giản, hoặc đi thu thập thêm dữ liệu |
| Cần ngoại suy ra ngoài vùng đã thấy | Cây chỉ trả về giá trị của các lá đã học — thu nhập 500 triệu nó vẫn chấm như 80 triệu | Mô hình tuyến tính |
| Cần giải thích từng quyết định cho pháp lý | Trăm cây cộng lại khó cãi trước tòa hơn 1 phương trình | Logistic Regression + hệ số |

---

## 3. Bài Toán: Duyệt Vay Phiên Bản Khó — Quy Luật Thật Của Ngân Hàng

Lần này mỗi hồ sơ có **3 cột**: Thu nhập (triệu/tháng), Nợ hiện tại (triệu), Số năm làm việc. Ngân hàng chấm theo 3 luật ngầm (mô hình KHÔNG được cho biết, phải tự học ra):

1. **Nợ = 0 → Từ chối** (chưa từng vay = không có lịch sử tín dụng) — quan hệ chữ V, bài trước Logistic đã thua.
2. **Nợ > 100 → Từ chối** (nợ ngập đầu).
3. Còn lại: **Duyệt nếu thu nhập ≥ 30**, HOẶC **thu nhập ≥ 20 VÀ đã đi làm ≥ 5 năm** (thu nhập vừa nhưng ổn định lâu năm vẫn được tin) — đây là luật **tương tác chéo** NẾU–VÀ–THÌ.

Trong 30 dòng dưới, có nhóm tuân luật, có nhóm bị cài nhãn sai, có nhóm khuyết dữ liệu. Cột **P(duyệt)** là xác suất chạy thật của XGBoost (`max_depth=3`); cột **LR / XGB** là dự đoán của Logistic Regression và XGBoost — ✅ đúng nhãn, ❌ sai nhãn.

---

## 4. Bộ Dữ Liệu 30 Dòng — Ai Thắng Ở Dòng Nào?

### 🟢 Nhóm A (dòng 1–8): Dữ liệu đơn giản — hai bên hòa nhau

| # | Thu nhập | Nợ | Năm LV | Nhãn | P(duyệt) XGB | LR | XGB | Giải thích |
|---|---:|---:|---:|:---:|---:|:---:|:---:|:---|
| 1 | 50 | 10 | 4 | ✅ | 0.81 | ✅ | ✅ | Thu nhập cao, nợ ít — ai cũng đoán được |
| 2 | 65 | 20 | 6 | ✅ | 0.88 | ✅ | ✅ | Tương tự |
| 3 | 45 | 15 | 3 | ✅ | 0.83 | ❌ | ✅ | ⚠️ LR sai vì bị 2 dòng nhãn nhiễu 25–26 (hồ sơ gần giống nhưng ghi Từ chối) kéo lệch hệ số |
| 4 | 70 | 30 | 8 | ✅ | 0.90 | ✅ | ✅ | Hồ sơ đẹp |
| 5 | 8 | 40 | 2 | ❌ | 0.34 | ✅ | ✅ | Thu nhập thấp → từ chối |
| 6 | 12 | 25 | 1 | ❌ | 0.07 | ✅ | ✅ | Tương tự |
| 7 | 10 | 50 | 3 | ❌ | 0.33 | ✅ | ✅ | Tương tự |
| 8 | 6 | 15 | 1 | ❌ | 0.16 | ✅ | ✅ | Tương tự |

**Bài học:** với dữ liệu một chiều đơn giản, XGBoost không cho bạn thêm gì nhiều — nó chỉ đáng giá khi dữ liệu bắt đầu "xoắn".

### 🟩 Nhóm B (dòng 9–14): Chữ V — nơi Logistic Regression gãy, cây thì không

| # | Thu nhập | Nợ | Năm LV | Nhãn | P(duyệt) XGB | LR | XGB | Giải thích |
|---|---:|---:|---:|:---:|---:|:---:|:---:|:---|
| 9 | 32 | 0 | 4 | ❌ | 0.20 | ❌ | ✅ | Nợ = 0 → không lịch sử tín dụng. LR vẫn tưởng "nợ ít là tốt" nên duyệt nhầm |
| 10 | 40 | 0 | 6 | ❌ | 0.15 | ❌ | ✅ | Tương tự — LR duyệt nhầm |
| 11 | 33 | 12 | 4 | ✅ | 0.90 | ✅ | ✅ | Nợ vừa phải + trả tốt → duyệt |
| 12 | 36 | 18 | 5 | ✅ | 0.79 | ✅ | ✅ | Tương tự |
| 13 | 34 | 150 | 4 | ❌ | 0.10 | ✅ | ✅ | Nợ 150 → từ chối |
| 14 | 38 | 200 | 6 | ❌ | 0.05 | ✅ | ✅ | Nợ 200 → từ chối |

**Vì sao cây làm được?** Cây chỉ cần 2 nhát chém trên cùng 1 cột: `Nợ < 6?` → Từ chối; `Nợ > 100?` → Từ chối; ở giữa → xét tiếp. Đường thẳng của Logistic chỉ chém được **1 nhát một chiều** ("nợ càng nhiều càng xấu") nên vĩnh viễn không vẽ nổi chữ V. Đây chính là nhóm E "vô vọng" của bài trước — sang tay XGBoost thì thành chuyện nhỏ.

### 🟩 Nhóm C (dòng 15–20): Tương tác chéo — luật NẾU–VÀ–THÌ

| # | Thu nhập | Nợ | Năm LV | Nhãn | P(duyệt) XGB | LR | XGB | Giải thích |
|---|---:|---:|---:|:---:|---:|:---:|:---:|:---|
| 15 | 24 | 20 | 7 | ✅ | 0.97 | ✅ | ✅ | Thu nhập vừa NHƯNG 7 năm ổn định → duyệt |
| 16 | 22 | 15 | 6 | ✅ | 0.90 | ✅ | ✅ | Tương tự |
| 17 | 25 | 10 | 8 | ✅ | 0.81 | ✅ | ✅ | Tương tự |
| 18 | 24 | 20 | 1 | ❌ | 0.09 | ✅ | ✅ | Y hệt dòng 15 về thu nhập & nợ, chỉ khác **mới đi làm 1 năm** → từ chối |
| 19 | 22 | 15 | 2 | ❌ | 0.07 | ✅ | ✅ | So với dòng 16: khác mỗi số năm |
| 20 | 28 | 10 | 1 | ❌ | 0.02 | ✅ | ✅ | Tương tự |

**Điểm mấu chốt:** so cặp dòng 15 vs 18 — thu nhập, nợ **giống hệt nhau**, chỉ khác số năm làm việc mà kết quả ngược nhau. Giá trị của cột "năm" **phụ thuộc vào** cột "thu nhập" (chỉ quan trọng khi thu nhập 20–30). Cây biểu diễn điều này tự nhiên bằng nhánh lồng nhánh. Trên 30 dòng này LR "gồng" theo kịp, nhưng sang hồ sơ mới là lộ (mục 5). Còn `max_depth=1` (cây gốc cụt, chỉ 1 câu hỏi mỗi cây) **về nguyên tắc không thể** biểu diễn luật VÀ — xem mục 6a.

### 🟩 Nhóm D (dòng 21–24): Ô trống (NaN) — đặc sản của XGBoost

| # | Thu nhập | Nợ | Năm LV | Nhãn | P(duyệt) XGB | LR | XGB | Giải thích |
|---|---:|---:|---:|:---:|---:|:---:|:---:|:---|
| 21 | 55 | 10 | **(trống)** | ✅ | 0.89 | 🚫 | ✅ | Khách quên khai năm làm việc |
| 22 | **(trống)** | 20 | 6 | ✅ | 0.94 | 🚫 | ✅ | Thiếu thu nhập nhưng nợ ổn + 6 năm ổn định |
| 23 | 9 | **(trống)** | 1 | ❌ | 0.02 | 🚫 | ✅ | Thu nhập 9 triệu — thiếu cột nợ vẫn đủ kết luận |
| 24 | **(trống)** | 180 | 2 | ❌ | 0.06 | 🚫 | ✅ | Nợ 180 — khỏi cần biết thu nhập |

**🚫 = Logistic Regression không chạy nổi:** gặp NaN là `sklearn` văng lỗi ngay, phải tự đi điền giá trị thay thế (imputation) — điền dở là méo dữ liệu. Trong thí nghiệm này, tui phải **vứt cả 4 dòng** khỏi tập huấn luyện của LR. XGBoost thì nuốt NaN trực tiếp: tại mỗi nút chia, nó học luôn "hồ sơ thiếu cột này thì đi nhánh nào có lợi nhất". Dữ liệu doanh nghiệp thật lúc nào cũng lỗ chỗ ô trống — đây là lý do số 1 dân trong nghề mê XGBoost.

### 🔴 Nhóm E (dòng 25–28): Nhãn bị ghi sai — thuốc độc cho mọi mô hình

| # | Thu nhập | Nợ | Năm LV | Nhãn (ghi sai) | P(duyệt) XGB | Giải thích |
|---|---:|---:|---:|:---:|---:|:---|
| 25 | 48 | 12 | 5 | ❌ (❓) | 0.21 | Hồ sơ đẹp mà ghi Từ chối — XGBoost depth=3 **học thuộc luôn cái sai** (P chỉ 0.21) |
| 26 | 52 | 18 | 6 | ❌ (❓) | 0.40 | Tương tự |
| 27 | 7 | 30 | 1 | ✅ (❓) | 0.70 | Hồ sơ tệ mà ghi Duyệt — cây cũng chiều theo (P = 0.70) |
| 28 | 11 | 45 | 2 | ✅ (❓) | 0.47 | Tương tự |

**Vì sao nguy hiểm hơn cả với cây?** Logistic Regression chỉ có 3 hệ số nên nhiễu làm nó *lệch* (như làm sai dòng 3). Còn cây đủ sâu thì **khoanh vùng riêng cho từng dòng nhiễu để học thuộc** — điểm train tăng, nhưng nó đã ghi nhớ quy luật ma: *"thu nhập 7, nợ 30 → duyệt"*. Hậu quả đo được ở mục 5: gặp hồ sơ mới (9, 35, 2 năm) — đáng từ chối — cây depth=3 lại **duyệt**, vì quá giống dòng nhiễu 27–28 nó đã thuộc lòng. Muốn cây "lì" trước nhiễu → `min_child_weight` (mục 6c).

### 🔴 Nhóm F (dòng 29–30): Trùng đặc trưng, ngược nhãn — vô phương với mọi thuật toán

| # | Thu nhập | Nợ | Năm LV | Nhãn | P(duyệt) XGB | Giải thích |
|---|---:|---:|---:|:---:|---:|:---|
| 29 | 20 | 15 | 3 | ✅ | 0.48 | Giống hệt dòng 30... |
| 30 | 20 | 15 | 3 | ❌ | 0.48 | ...XGBoost cũng chỉ biết chấm cả hai đúng 0.48 — "tôi chịu, 50/50" |

Bài trước Logistic chấm cặp song sinh 0.40/0.40, bài này XGBoost chấm 0.48/0.48 — **đổi thuật toán không cứu được dữ liệu thiếu cột thông tin.** Bao giờ cũng vậy.

---

## 5. Thí Nghiệm Quan Trọng Nhất: Học Quy Luật Hay Học Vẹt?

Điểm số trên 30 dòng đã học (train) là **điểm ảo** — cây đủ sâu học thuộc được hết. Muốn biết mô hình thật sự hiểu quy luật, phải đem **8 hồ sơ MỚI** (chấm đúng theo 3 luật ngân hàng, mô hình chưa từng thấy) ra kiểm tra. Kết quả chạy thật:

| Mô hình | Đúng trên 30 dòng train | Đúng trên 8 hồ sơ MỚI | Chẩn đoán |
| :--- | :---: | :---: | :--- |
| Logistic Regression | 69% | 62% | Không học nổi chữ V và luật VÀ |
| XGBoost `max_depth=1` | 90% | 62% | Cây cụt 1 câu hỏi — không diễn đạt nổi luật "thu nhập ≥ 20 **VÀ** năm ≥ 5" |
| **XGBoost `max_depth=3`** | 93% | **75%** ✅ | Điểm ngọt — đủ sâu để học luật, chưa đủ sâu để học vẹt hết |
| XGBoost `max_depth=10` | 93% | 62% ⚠️ | **Điểm train Y HỆT depth=3 nhưng điểm hồ sơ mới rớt** — phần "thông minh thêm" chỉ dùng để học thuộc nhiễu |
| XGBoost `learning_rate=0.01, n=10 cây` | 53% | 50% | Chưa kịp học gì — như tung đồng xu |

**Hai kết luận xương máu:**
1. **Không bao giờ tin điểm train.** depth=3 và depth=10 cùng 93% train nhưng một đứa hiểu bài, một đứa học vẹt. Luôn phải giữ riêng một tập kiểm thử (validation set) mà mô hình chưa từng thấy.
2. **Ngay cả depth=3 cũng bị nhóm E đầu độc:** trong 2 hồ sơ mới nó sai, có hồ sơ (thu nhập 9, nợ 35, 2 năm — rõ ràng phải từ chối) bị nó **duyệt**, vì quá giống 2 dòng nhãn nhiễu 27–28 mà nó trót học. Dữ liệu sạch quan trọng hơn thuật toán xịn.

---

## 6. Các Tham Số Ảnh Hưởng Cái Gì? Dòng Nào Được Học, Dòng Nào Bị Bỏ Qua?

### 6a. `max_depth` — mỗi cây được hỏi bao nhiêu câu liên tiếp?

| max_depth | Chuyện xảy ra với từng nhóm dòng (số liệu thật) |
| :---: | :--- |
| **1** | Mỗi cây = đúng 1 câu hỏi. Nhờ boosting nhiều cây cộng lại vẫn xử được chữ V nhóm B trên train, nhưng luật VÀ của nhóm C thì đến hồ sơ mới là lộ tẩy (62%). Còn tệ hơn: nó đoán sai cả dòng 30 trên train — bó tay không đủ chỗ nhớ |
| **3** | Một cây hỏi được 3 câu liên tiếp: `Nợ < 6?` → `Thu nhập ≥ 30?` → `Năm ≥ 5?` — vừa khít 3 luật của ngân hàng. Đó là lý do nó thắng (75% hồ sơ mới) |
| **10** | Thừa sức khoanh vùng từng dòng nhiễu 25–28 để học thuộc → hồ sơ mới rớt về 62%. **Dữ liệu càng ít, càng phải để depth thấp** |

### 6b. `learning_rate` (η) + `n_estimators` — cặp đôi không tách rời

Khác với Gradient Descent của Logistic, ở đây `learning_rate` là **mức độ tin vào mỗi cây mới**: dự đoán = cây cũ + η × cây mới.

- `η = 0.01` với chỉ 10 cây → cộng lại mới "học" được 10% quy luật → 53% train, như chưa học (số thật ở bảng mục 5).
- `η = 0.3` với 200 cây (mặc định) → ổn với bài này.
- Kinh nghiệm thực chiến: **giảm η xuống 0.01–0.1 và tăng n_estimators lên hàng nghìn** + bật early stopping — đi chậm mà chắc hầu như luôn tổng quát hóa tốt hơn đi nhanh. Hai tham số này phải chỉnh **cùng nhau**: giảm η mà quên tăng số cây = underfit.

### 6c. `min_child_weight` (XGBoost) / `min_data_in_leaf` (LightGBM) — "đừng học chi tiết vụn vặt"

Tham số chống nhiễu: mỗi lá của cây phải chứa đủ "trọng lượng" mẫu, không được mở lá riêng cho 1–2 dòng cá biệt. Kết quả thật khi tăng từ 1 → 3 trên bộ 30 dòng:

- **Điều hay:** cây **từ chối học theo nhãn nhiễu** — dòng 25, 26 (hồ sơ đẹp bị ghi Từ chối) giờ được nó dự đoán **Duyệt**, tức là nó đoán theo lẽ thường thay vì học vẹt cái nhãn sai. Đây chính là tác dụng chống nhiễu trong sách vở.
- **Điều dở (bất ngờ!):** với vỏn vẹn 30 dòng, luật thật cũng chỉ có 2–3 dòng đại diện mỗi vùng — nên nó **bỏ qua luôn cả luật thật**: các dòng 10, 13, 14 (nhóm chữ V) bị đoán sai theo, điểm hồ sơ mới rớt còn 50%.

**Bài học:** `min_child_weight` là con dao hai lưỡi trên dữ liệu nhỏ. Nó phát huy đúng khi bạn có hàng nghìn dòng — lúc đó "vùng có 2 dòng cá biệt" gần như chắc chắn là nhiễu. Với 30 dòng thì đừng đụng vào.

### 6d. Các tham số đáng biết còn lại (khái niệm nhanh)

| Tham số | Ý nghĩa đời thường |
| :--- | :--- |
| `subsample`, `colsample_bytree` | Mỗi cây chỉ được xem ngẫu nhiên 70–80% số dòng/số cột → các cây đa dạng hơn, khó học vẹt hơn (giống tinh thần Random Forest) |
| `reg_alpha` (L1), `reg_lambda` (L2) | Phạt cây phức tạp ngay trong hàm lỗi — regularization "chính chủ" của XGBoost |
| `num_leaves` (LightGBM) | LightGBM mọc cây theo **lá** (leaf-wise) chứ không theo tầng — nhanh và mạnh hơn nhưng cực dễ overfit trên dữ liệu nhỏ; luôn chỉnh `num_leaves` đi kèm `min_data_in_leaf` |
| `scale_pos_weight` | Bản sao của `class_weight` bên Logistic — dùng khi 2 lớp mất cân bằng nặng |

### 6e. Feature Importance — món quà cho sếp

XGBoost depth=3 tự chấm độ quan trọng của từng cột (số thật): **Năm làm việc 0.51 — Nợ 0.29 — Thu nhập 0.20**. Bất ngờ không? Thu nhập bét bảng! Vì bản thân thu nhập cao chưa quyết định gì — nó phải **kết hợp** với năm làm việc (luật VÀ) mới ra kết quả, và chính cột "năm" là chìa khóa tách các cặp dòng sinh đôi 15/18, 16/19. Đây là loại insight bạn đem đi trình bày với quản lý được: *"muốn duyệt vay chính xác hơn, hãy thu thập kỹ số năm làm việc — nó quan trọng hơn cả thu nhập."*

---

## 7. Checklist: Chọn Logistic Regression Hay XGBoost?

- [ ] Dữ liệu dạng bảng, quan hệ một chiều đơn giản, cần giải thích từng hệ số → **Logistic Regression đủ dùng** (nhóm A).
- [ ] Có quan hệ chữ V / ngưỡng bậc thang (nhóm B), luật NẾU–VÀ–THÌ (nhóm C), hoặc nhiều ô trống NaN (nhóm D) → **XGBoost/LightGBM**.
- [ ] Dữ liệu dưới vài trăm dòng → cẩn thận tối đa: `max_depth ≤ 3`, đừng đụng `min_child_weight`, và bắt buộc có tập kiểm thử riêng.
- [ ] Nghi ngờ nhãn nhiễu (nhóm E)? → soi các dòng mà mô hình chấm xác suất ngược hẳn với nhãn; cân nhắc `min_child_weight`/`min_data_in_leaf` **nếu** dữ liệu đủ lớn.
- [ ] Sau khi train: nhìn **Feature Importance** để kể chuyện với sếp, và **không bao giờ** báo cáo điểm train.

---

## 8. Code Chạy Lại Toàn Bộ (copy vào notebook là chạy)

```python
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

nan = float('nan')
# 30 hồ sơ: (thu nhập, nợ, năm làm việc, nhãn 1=duyệt)
# Quy luật thật: nợ=0 -> 0; nợ>100 -> 0; còn lại: thu nhập>=30 -> 1, hoặc (>=20 và năm>=5) -> 1
rows = [
    # A (1-8) đơn giản
    (50,10,4,1),(65,20,6,1),(45,15,3,1),(70,30,8,1),
    (8,40,2,0),(12,25,1,0),(10,50,3,0),(6,15,1,0),
    # B (9-14) chữ V theo cột nợ
    (32,0,4,0),(40,0,6,0),(33,12,4,1),(36,18,5,1),(34,150,4,0),(38,200,6,0),
    # C (15-20) tương tác thu nhập x năm làm việc
    (24,20,7,1),(22,15,6,1),(25,10,8,1),(24,20,1,0),(22,15,2,0),(28,10,1,0),
    # D (21-24) khuyết thiếu NaN
    (55,10,nan,1),(nan,20,6,1),(9,nan,1,0),(nan,180,2,0),
    # E (25-28) nhãn ghi sai
    (48,12,5,0),(52,18,6,0),(7,30,1,1),(11,45,2,1),
    # F (29-30) trùng đặc trưng ngược nhãn
    (20,15,3,1),(20,15,3,0),
]
X = np.array([[r[0],r[1],r[2]] for r in rows]); y = np.array([r[3] for r in rows])

# 8 hồ sơ MỚI chấm đúng theo quy luật thật — thước đo "hiểu bài hay học vẹt"
probe = [(42,14,3,1),(58,25,7,1),(35,0,5,0),(37,160,5,0),
         (23,18,6,1),(26,12,1,0),(9,35,2,0),(31,22,4,1)]
Xp = np.array([[r[0],r[1],r[2]] for r in probe]); yp = np.array([r[3] for r in probe])

# Logistic Regression: phải vứt 4 dòng NaN + chuẩn hóa
mask = ~np.isnan(X).any(axis=1)
sc = StandardScaler().fit(X[mask])
lr = LogisticRegression(max_iter=5000).fit(sc.transform(X[mask]), y[mask])
print("LR  probe:", (lr.predict(sc.transform(Xp))==yp).mean())

# XGBoost: ăn NaN trực tiếp, không cần chuẩn hóa
for depth in [1, 3, 10]:
    m = xgb.XGBClassifier(max_depth=depth, learning_rate=0.3,
                          n_estimators=200, eval_metric='logloss').fit(X, y)
    print(f"XGB depth={depth}: train={(m.predict(X)==y).mean():.2f}"
          f" probe={(m.predict(Xp)==yp).mean():.2f}")

m3 = xgb.XGBClassifier(max_depth=3, learning_rate=0.3, n_estimators=200,
                       eval_metric='logloss').fit(X, y)
print("Feature importance:", dict(zip(['thu_nhap','no','nam'],
                                      m3.feature_importances_.round(2))))

# Bài tập tự nghịch:
# 1. Xóa 4 dòng nhiễu 25-28 rồi train lại depth=3 -> điểm probe lên bao nhiêu?
# 2. Thử min_child_weight=3 -> dòng 25, 26 được dự đoán thế nào? (mục 6c)
# 3. Thử LGBMClassifier với num_leaves=31 (mặc định) vs num_leaves=4 trên bộ này
```

---

## 9. Kết Nối Với Các Tài Liệu Khác

| Bạn muốn | Mở file |
| :--- | :--- |
| Ôn lại vì sao Logistic thua nhóm B | `../Logistic_Regression/USECASE.md` (nhóm E của bài đó) |
| Hiểu lý thuyết Gini, Boosting, GOSS/EFB | `README.md` (lộ trình 3–5 ngày) |
| Nghịch đồ thị tương tác | `study_guide.html` |
| Tự tay tuning và vẽ Feature Importance | `XGBoost_LightGBM_Practice.ipynb` |

> 🌟 **Câu chốt của bài:** XGBoost không thắng vì "thông minh hơn" — nó thắng vì **hình dạng của cây (NẾU–VÀ–THÌ, chém nhiều nhát) trùng với hình dạng quy luật của dữ liệu bảng**. Và dù thuật toán nào đi nữa: điểm train là ảo, dữ liệu sạch quý hơn mô hình xịn.
