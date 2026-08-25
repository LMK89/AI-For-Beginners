# 🌲 Lộ trình Học Tập & Thực Hành: XGBoost & LightGBM (3 - 5 Ngày)

> **"XGBoost và LightGBM chính là những vị 'vua thống trị' không thể chối cãi của dữ liệu bảng (Tabular Data) trong thế giới thực tế của doanh nghiệp."**

Chào Khang! Trong các doanh nghiệp thực tế (như tài chính, ngân hàng, thương mại điện tử, logistics), hơn **80% dữ liệu** tồn tại dưới dạng bảng (hàng và cột từ SQL, Excel). Đối với dạng dữ liệu này, các mô hình Deep Learning phức tạp thường không những hoạt động kém hiệu quả mà còn ngốn rất nhiều chi phí tính toán so với các mô hình dựa trên cây quyết định (Tree-based models), đặc biệt là **Gradient Boosting**.

Dưới đây là lộ trình chi tiết giúp bạn làm chủ hai công cụ mạnh mẽ nhất hiện nay trong 3 - 5 ngày.

> 🧭 **Thấy lý thuyết khô khan?** Hãy mở file [USECASE.md](./USECASE.md) trước: XGBoost đấu tay đôi với Logistic Regression trên bộ dữ liệu duyệt vay 30 dòng — xem cây thắng ở dòng nào (chữ V, tương tác chéo, NaN), thua ở dòng nào (học vẹt, nhãn nhiễu), và `max_depth`, `learning_rate`, `min_child_weight` thay đổi kết quả ra sao. Toàn bộ số liệu chạy thật, kèm code.

---

## 🎯 Mục Tiêu Cốt Lõi Cần Đạt Được
Sau bài học này, bạn cần nắm vững:
1. **Sự tiến hóa của thuật toán:** Hiểu rõ con đường phát triển từ **Decision Tree (Cây quyết định)** $\rightarrow$ **Ensemble Learning (Học kết hợp: Bagging & Boosting)** $\rightarrow$ **Gradient Boosting**.
2. **Nghệ thuật Tinh chỉnh Siêu tham số (Hyperparameter Tuning):** Hiểu rõ ý nghĩa và cách tinh chỉnh `learning_rate`, `max_depth`, `subsample`, `min_child_weight` để kiểm soát tốt ranh giới giữa Underfitting và Overfitting.
3. **Giá trị thực tiễn trong doanh nghiệp:** Cách mô hình xử lý dữ liệu khuyết thiếu (missing values) và tính toán độ quan trọng của các tính năng (**Feature Importance**) để giải thích mô hình cho các nhà quản lý đưa ra quyết định kinh doanh.

---

## 🗓️ Lộ Trình Chi Tiết (3 - 5 Ngày)

### 📅 Ngày 1: Nền Tảng - Decision Tree & Ensemble Learning
Trước khi học các thuật toán nâng cao, bạn bắt buộc phải hiểu các viên gạch nền móng đầu tiên.

*   **Decision Tree (Cây quyết định):**
    *   Cách một cây quyết định phân đôi dữ liệu tại mỗi nút (node).
    *   Độ đo độ tinh khiết: **Gini Impurity** và **Entropy** (Độ hỗn loạn).
    *   **Information Gain (Độ lợi thông tin):** Mục tiêu chọn đặc trưng chia nhánh sao cho giảm thiểu độ hỗn loạn nhiều nhất.
*   **Ensemble Learning (Học kết hợp):**
    *   **Bagging (Bootstrap Aggregating):** Xây dựng song song nhiều cây độc lập rồi lấy trung bình kết quả (Ví dụ kinh điển: **Random Forest**). Giúp giảm phương sai (Variance) - hạn chế Overfitting.
    *   **Boosting:** Xây dựng các cây **tuần tự (sequential)**. Cây sau tập trung sửa chữa sai số (residual error) của cây trước. Giúp giảm độ chệch (Bias).

---

### 📅 Ngày 2: Bản chất của Gradient Boosting & Sự Ra Đời của XGBoost
Hôm nay chúng ta sẽ đi sâu vào cơ chế cốt lõi của "Boosting dựa trên độ dốc".

*   **Gradient Boosting Machine (GBM):**
    *   Mô hình khởi đầu bằng một dự đoán hằng số đơn giản (ví dụ: trung bình của nhãn).
    *   Tính toán phần dư sai số (residual error): $e_i = y_i - \hat{y}_i$.
    *   Huấn luyện một cây quyết định mới để **dự đoán phần dư sai số** đó chứ không dự đoán nhãn trực tiếp.
    *   Cập nhật dự đoán mới bằng cách cộng thêm phần dự đoán lỗi nhân với hệ số co lại (Learning Rate $\eta$):
        $$\hat{y}_{mới} = \hat{y}_{cũ} + \eta \cdot T_j(x)$$
*   **XGBoost (Extreme Gradient Boosting):**
    *   Là phiên bản nâng cấp phần cứng và tối ưu toán học cực mạnh của GBM.
    *   Sử dụng khai triển Taylor bậc hai (đạo hàm bậc 1 và bậc 2) của hàm lỗi để tính toán độ dốc tối ưu nhanh hơn.
    *   Tích hợp sẵn các kỹ thuật **Regularization** ($L_1$ và $L_2$) giúp kiểm soát sự phức tạp của cây trực tiếp trong hàm lỗi, giảm Overfitting tối đa.
    *   Cơ chế tìm điểm chia nhánh song song (Approximate Split Finding).

---

### 📅 Ngày 3: LightGBM - Kẻ Thách Thức Siêu Tốc Độ
Khi dữ liệu tăng lên hàng triệu dòng, XGBoost thường bắt đầu chạy chậm lại. Đó là lúc LightGBM xuất hiện.

*   **Sự khác biệt về cấu trúc tăng trưởng:**
    *   **XGBoost (Level-wise growth):** Phát triển cây theo chiều rộng, chia đều các nhánh ở cùng một cấp độ. Cân bằng hơn nhưng đôi khi thừa thãi.
    *   **LightGBM (Leaf-wise growth):** Phát triển cây theo chiều sâu. Nó tìm nút lá có độ lợi thông tin lớn nhất để tiếp tục chia đôi bất kể cấp độ. Nhanh hơn và giảm sai số tốt hơn nhưng rất dễ Overfitting trên tập dữ liệu nhỏ.
*   **Các thuật toán cốt lõi của LightGBM:**
    *   **GOSS (Gradient-based One-Side Sampling):** Giữ lại các mẫu có gradient lớn (lỗi nhiều) và lấy mẫu ngẫu nhiên các mẫu có gradient nhỏ để giảm kích thước dữ liệu huấn luyện mà không mất thông tin quan trọng.
    *   **EFB (Exclusive Feature Bundling):** Gộp các đặc trưng thưa thớt (sparse features) ít khi xuất hiện cùng nhau thành một đặc trưng duy nhất để giảm số lượng chiều dữ liệu.

---

### 📅 Ngày 4: Bí Thuật Tinh Chỉnh Siêu Tham Số (Hyperparameter Tuning)
Trọng tâm hôm nay là thực hành kiểm soát hành vi huấn luyện của mô hình.

| Siêu tham số (Hyperparameter) | Mô tả & Cách tinh chỉnh để tránh Overfitting |
| :--- | :--- |
| **`learning_rate` / `eta`** | Tốc độ học. Nhỏ hơn (0.01 - 0.1) giúp mô hình hội tụ tốt hơn nhưng cần nhiều cây hơn (`n_estimators`). |
| **`max_depth`** | Chiều sâu tối đa của cây. Càng lớn cây càng phức tạp và dễ overfitting. Thường để từ 3 - 8. |
| **`min_child_weight` (XGB) / `min_data_in_leaf` (LGB)** | Số lượng mẫu tối thiểu trong một nút lá. Tăng giá trị này lên giúp hạn chế việc chia nhánh quá nhỏ (tránh overfitting). |
| **`subsample` / `bagging_fraction`** | Tỷ lệ số lượng dòng dữ liệu được chọn ngẫu nhiên để huấn luyện mỗi cây. Thường để 0.7 - 0.9. |
| **`colsample_bytree` / `feature_fraction`** | Tỷ lệ số lượng cột đặc trưng được chọn ngẫu nhiên cho mỗi cây. Giúp tạo sự đa dạng giữa các cây. |

---

### 📅 Ngày 5: Thực Hành - Xử Lý Thiếu Hụt & Feature Importance
Bạn sẽ áp dụng mô hình vào một dự án thực tế: Phân loại rủi ro tín dụng hoặc Dự đoán khách hàng rời bỏ dịch vụ (Churn Prediction).

*   **Xử lý dữ liệu khuyết thiếu (Missing Values):**
    *   XGBoost và LightGBM có cơ chế tự động tìm hướng đi tối ưu cho các giá trị khuyết thiếu trong quá trình chia nhánh mà không cần chúng ta phải điền giá trị trung bình hay trung vị thủ công.
*   **Feature Importance (Độ quan trọng của đặc trưng):**
    *   Trích xuất thuộc tính `.feature_importances_` để trực quan hóa xem những biến số nào (ví dụ: Số dư tài khoản, Độ tuổi, Thu nhập) ảnh hưởng lớn nhất tới quyết định phê duyệt khoản vay.
    *   Đây là vũ khí tối thượng giúp bạn giải thích mô hình (Explainable AI) cho các cấp lãnh đạo trong doanh nghiệp.

---

## 🛠️ Bài Tập & Bộ Tài Liệu Tương Tác Của Bạn
Tôi đã chuẩn bị sẵn các tài nguyên học tập thực quan chất lượng cao dành riêng cho bạn:
1. **Trang web Tương tác Visual**:
   👉 `Khang_lession/XGBoost_LightGBM/study_guide.html` (Nơi trực quan hóa thuật toán và tinh chỉnh thử các hyperparameter trên điện thoại/máy tính).
2. **File Jupyter Notebook Thực Hành**:
   👉 `Khang_lession/XGBoost_LightGBM/XGBoost_LightGBM_Practice.ipynb` (Thực hành huấn luyện XGBoost và LightGBM trên tập dữ liệu bảng thực tế từ Kaggle).

*Chúc Khang chinh phục thành công đỉnh cao XGBoost & LightGBM! Hãy mở file HTML trực quan lên và khám phá sự kỳ diệu nhé! 🚀*
