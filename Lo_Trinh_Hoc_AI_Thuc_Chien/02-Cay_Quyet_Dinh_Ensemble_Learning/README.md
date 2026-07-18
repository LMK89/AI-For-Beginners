# Chương 2: Cây Quyết Định & Các Mô Hình Ensemble (XGBoost & LightGBM)

Chào mừng bạn đến với Chương 2! Trong thế giới doanh nghiệp thực tế, hơn 80% dữ liệu tồn tại dưới dạng bảng (tabular data từ SQL, Excel). Đây là nơi các mô hình dựa trên Cây Quyết Định (Tree-based models) như XGBoost và LightGBM thống trị tuyệt đối.
> 🧭 **Học qua ví dụ trước, lý thuyết sau?** Mở [USECASE về XGBoost/LightGBM](../../Khang_lession/XGBoost_LightGBM/USECASE.md): XGBoost đấu tay đôi Logistic Regression trên 30 hồ sơ vay — cây thắng ở đâu (chữ V, tương tác chéo, NaN), thua ở đâu (học vẹt, nhãn nhiễu), và `max_depth`, `learning_rate`, `min_child_weight` đổi kết quả thế nào. Số liệu chạy thật, kèm code.


---

## 📐 1. Bản Chất Toán Học (Math Foundations)

Để quyết định chia nhánh một cây quyết định tại một thuộc tính sao cho tối ưu nhất, chúng ta sử dụng các thước đo độ hỗn loạn thông tin.

### A. Chỉ số Entropy (Độ hỗn loạn):
$$H(X) = -\sum_{i=1}^{c} P(x_i) \log_2 P(x_i)$$
- Khi dữ liệu hoàn toàn đồng nhất (chỉ toàn nhãn 1 hoặc chỉ toàn nhãn 0), Entropy = 0.
- Khi dữ liệu phân tán hỗn loạn 50/50, Entropy đạt giá trị cực đại = 1.

### B. Chỉ số Gini Impurity (Độ không thuần khiết):
$$\text{Gini} = 1 - \sum_{i=1}^{c} (P(x_i))^2$$
- Chỉ số Gini càng nhỏ, tập dữ liệu tại nút đó càng thuần khiết.

---

## 🤝 2. Liên Kết Mô Hình Machine Learning (ML Connection)

Chúng ta đi qua con đường tiến hóa đầy ngoạn mục của các thuật toán cây:
1. **Decision Tree (Cây quyết định đơn lẻ):** Dễ hiểu nhưng cực kỳ dễ bị Overfitting (học tủ dữ liệu).
2. **Ensemble Learning (Học kết hợp):** Gộp nhiều cây lại để đưa ra kết quả chính xác hơn.
   - **Bagging (Random Forest):** Huấn luyện song song nhiều cây độc lập, lấy trung bình kết quả. Giúp giảm **Variance (Phương sai)**, tức là giảm Overfitting.
   - **Boosting (AdaBoost, Gradient Boosting Machine - GBM):** Huấn luyện các cây tuần tự, cây sau tập trung sửa sai lỗi (residuals) của cây trước. Giúp giảm **Bias (Độ chệch)**.
3. **XGBoost & LightGBM:** Bản nâng cấp thuật toán tối ưu phần cứng cực nhanh của GBM giúp xử lý hàng triệu dòng dữ liệu lớn của doanh nghiệp.

---

## 💼 3. Công Việc Thực Tế Của Data Analyst / Data Engineer

Trong doanh nghiệp, XGBoost và LightGBM là tiêu chuẩn vàng:

1. **Data Analyst / Business Analyst:**
   - Sử dụng thuộc tính `.feature_importances_` để vẽ biểu đồ tìm xem biến số nào quan trọng nhất (Ví dụ: Số dư tài khoản, tuổi tác, thu nhập ảnh hưởng ra sao đến hồ sơ vay vốn).
   - Giải thích hành vi mô hình cho ban giám đốc (Explainable AI).

2. **Data Engineer:**
   - Do thuật toán cây không cần tính toán nhân ma trận như Neural Network nên **không bắt buộc** phải chuẩn hóa dữ liệu (Feature Scaling). Điều này giúp tiết kiệm thời gian thiết kế pipeline dữ liệu thô cực lớn.
   - Mô hình cây tự động xử lý tốt dữ liệu khuyết thiếu (missing values) mà không cần điền tay thủ công.

---

## 💻 4. Lập Trình Python Thực Chiến

Dưới đây là cách sử dụng thư viện LightGBM cực nhanh trong thực tế:

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Khởi tạo mô hình LightGBM Classifier
model = lgb.LGBMClassifier(
    learning_rate=0.05,
    max_depth=5,
    n_estimators=100,
    random_state=42
)

# 2. Huấn luyện mô hình
model.fit(X_train, y_train)

# 3. Dự đoán kết quả
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác mô hình: {accuracy:.4f}")

# 4. Trích xuất độ quan trọng của đặc trưng
importances = model.feature_importances_
```

---

## 📊 5. Trực Quan Hóa (Visualization)

Xem các đồ thị trực quan hóa cơ chế chia nhánh cây tại:
👉 `lo_trinh_hoc_tap.html` (Phần 2)
