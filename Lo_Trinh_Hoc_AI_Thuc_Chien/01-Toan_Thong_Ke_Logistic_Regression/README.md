# Chương 1: Toán Thống Kê & Hồi Quy Logistic (Logistic Regression)

Chào mừng bạn đến với phần học đầu tiên của lộ trình AI thực chiến! Phần học này được thiết kế theo hình thức học xen kẽ (mixed): từ toán học -> liên kết mô hình AI -> ứng dụng thực tiễn của Data Analyst / Data Engineer -> code Python thuần -> trực quan hóa công thức.
> 🧭 **Học qua ví dụ trước, lý thuyết sau?** Mở [USECASE về Logistic Regression](../../Khang_lession/Logistic_Regression/USECASE.md): bộ dữ liệu duyệt vay 30 dòng — dòng nào mô hình đúng/sai, tại sao, và ngưỡng, C, learning rate ảnh hưởng ra sao. Toàn bộ số liệu chạy thật, kèm code.


---

## 📐 1. Bản Chất Toán Học (Math Foundations)

Khi bắt đầu học AI phân loại, chúng ta cần tìm một cách để chuyển đổi một giá trị số thực bất kỳ $z = W^T X + b$ (là kết quả của phương trình tuyến tính) thành một giá trị xác suất nằm trong khoảng $[0, 1]$.

Để làm được điều này, chúng ta sử dụng **Hàm kích hoạt Sigmoid** (hay còn gọi là hàm Logistic):
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Đặc điểm của Hàm Sigmoid:
- Nếu $z$ là một số dương cực kỳ lớn ($z \to +\infty$), thì $e^{-z} \to 0$, dẫn đến $\sigma(z) \to 1$.
- Nếu $z$ là một số âm cực kỳ lớn ($z \to -\infty$), thì $e^{-z} \to +\infty$, dẫn đến $\sigma(z) \to 0$.
- Nếu $z = 0$, thì $e^{0} = 1$, dẫn đến $\sigma(z) = 0.5$. Đây chính là **ngưỡng phân loại mặc định** (Decision Threshold).

---

## 🤝 2. Liên Kết Mô Hình Machine Learning (ML Connection)

Hồi quy Logistic thực chất là việc kết hợp phương trình đường thẳng tuyến tính với hàm kích hoạt Sigmoid để đưa ra dự đoán nhị phân (0 hoặc 1).

### Hàm Mất Mát (Loss Function): Binary Cross-Entropy Loss
Để đo lường độ sai lệch giữa dự đoán $\hat{y}$ và thực tế $y \in \{0, 1\}$, chúng ta không sử dụng Mean Squared Error (MSE) vì sẽ tạo ra hàm không lồi. Thay vào đó, chúng ta dùng hàm **Binary Cross-Entropy Loss**:
$$J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### Cơ chế phạt trực quan của Log Loss:
- Nếu nhãn thực tế $y = 1$ và mô hình dự đoán $\hat{y} \to 1$, thì Loss tiến về $0$.
- Nếu nhãn thực tế $y = 1$ nhưng mô hình dự đoán $\hat{y} \to 0$, thì $-\log(\hat{y}) \to +\infty$ (mô hình bị phạt cực kỳ nặng!).

---

## 💼 3. Công Việc Thực Tế Của Data Analyst / Data Engineer

Trong các doanh nghiệp công nghệ và tài chính, mô hình này là một vũ khí cực kỳ mạnh mẽ vì tính chất dễ giải thích:

1. **Data Analyst (Phân tích dữ liệu):**
   - Dự đoán khả năng khách hàng rời bỏ dịch vụ viễn thông (Customer Churn Rate).
   - Dự đoán khả năng khách hàng bấm vào một quảng cáo (Click-Through Rate - CTR).
   - Phân tích rủi ro tín dụng ngân hàng (Xác định khách hàng có khả năng nợ xấu).

2. **Data Engineer (Kỹ sư dữ liệu):**
   - Thiết kế các Pipeline dữ liệu tự động làm sạch và chuẩn hóa dữ liệu thô từ hệ thống SQL.
   - Thực hiện chuẩn hóa đặc trưng (Feature Scaling như `StandardScaler` hoặc `MinMaxScaler`) vì Gradient Descent trong hồi quy Logistic rất nhạy cảm với thang đo của dữ liệu.

---

## 💻 4. Lập Trình Python Thuần (Numpy Scratch)

Dưới đây là cách chúng ta tự viết toàn bộ mô hình Logistic Regression từ số không mà không cần thư viện `scikit-learn`:

```python
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # 1. Lan truyền xuôi (Forward Pass)
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            # 2. Tính toán đạo hàm Gradient
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            # 3. Cập nhật trọng số (Gradient Descent Step)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Lưu lại Loss lịch sử để vẽ đồ thị
            loss = - (1/m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            self.loss_history.append(loss)

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

---

## 📊 5. Trực Quan Hóa (Visualization)

Để trực quan hóa trực tiếp thuật toán này một cách sinh động, hãy mở file tương tác:
👉 `lo_trinh_hoc_tap.html` (Phần 1) hoặc file notebook thực hành kèm theo trong thư mục này.
