# 📘 Lộ trình Học Tập & Thực Hành: Logistic Regression (1 - 2 Ngày)

> **"Logistic Regression chính là 'cánh cửa' đưa bạn bước vào thế giới Phân loại (Classification) và Mạng Neural sâu sắc."**

Chào Khang! Nhiều người khi mới học AI thường vội vàng nhảy ngay vào Deep Learning đồ sộ mà bỏ qua các mô hình tuyến tính. Tuy nhiên, **Logistic Regression (Hồi quy Logistic)** thực chất chính là một **"Perceptron đơn giản"** – đơn vị cấu thành cơ bản nhất của mọi mạng neural (Neural Network). Hiểu sâu sắc thuật toán này sẽ giúp bạn có một nền tảng toán học và tư duy cực kỳ vững chắc.

Dưới đây là lộ trình chi tiết và tài liệu học tập được thiết kế riêng cho bạn trong 1 - 2 ngày tới.

> 🧭 **Thấy lý thuyết khô khan?** Hãy mở file [USECASE.md](./USECASE.md) trước: học qua bộ dữ liệu 30 dòng cụ thể (bài toán duyệt khoản vay) — xem Logistic Regression đúng ở dòng nào, sai ở dòng nào, tại sao, và các tham số (ngưỡng, C, learning rate) ảnh hưởng ra sao. Toàn bộ số liệu đều chạy thật, có code kèm theo để tự nghịch.

---

## 🎯 Mục Tiêu Cốt Lõi Cần Đạt Được
Sau bài học này, bạn cần phải nắm vững và tự tay giải thích/lập trình được:
1. **Tư duy Phân loại (Classification):** Cách biến đổi một đầu ra tuyến tính thành xác suất nằm trong khoảng $(0, 1)$ bằng hàm kích hoạt **Sigmoid**.
2. **Hàm mất mát (Loss Function):** Hiểu rõ bản chất hàm **Binary Cross-Entropy** (hoặc Softmax cho phân loại nhiều lớp) để đo lường độ sai lệch giữa dự đoán và thực tế.
3. **Tối ưu hóa (Optimization):** Cách thuật toán **Gradient Descent** tính toán độ dốc (gradient) và cập nhật trọng số ($\theta$) theo từng bước để giảm thiểu sai số.

---

## 🗓️ Lộ Trình Chi Tiết (1 - 2 Ngày)

### 📅 Ngày 1: Nền Tảng Lý Thuyết & Toán Học (Theory & Math)
Hôm nay, bạn hãy tập trung hiểu bản chất toán học đằng sau mô hình. Đừng sợ công thức! Hãy cố gắng hiểu ý nghĩa vật lý/hình học của chúng.

#### 1. Từ Tuyến Tính sang Xác Suất (Hàm Sigmoid)
*   Trong Hồi quy Tuyến tính (Linear Regression), đầu ra là một số thực bất kỳ: $z = \theta^T X = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ...$
*   Để phân loại nhị phân (Binary Classification - chỉ có 2 nhãn 0 hoặc 1), chúng ta cần ánh xạ $z$ về khoảng $(0, 1)$ biểu thị cho xác suất $P(y=1|X)$.
*   Hàm kích hoạt **Sigmoid** (hay hàm Logistic) thực hiện điều này cực kỳ xuất sắc:
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
*   **Đặc điểm:** 
    *   Nếu $z \to +\infty \Rightarrow \sigma(z) \to 1$
    *   Nếu $z \to -\infty \Rightarrow \sigma(z) \to 0$
    *   Nếu $z = 0 \Rightarrow \sigma(z) = 0.5$ (ngưỡng phân loại mặc định)

#### 2. Hàm Mất Mát: Binary Cross-Entropy Loss
*   Tại sao không dùng Mean Squared Error (MSE)? Vì khi kết hợp MSE với Sigmoid, hàm mục tiêu sẽ **không lồi (non-convex)**, dẫn đến rất nhiều cực tiểu cục bộ (local minima), khiến Gradient Descent hoạt động cực kỳ tệ.
*   Chúng ta sử dụng **Binary Cross-Entropy Loss** (còn gọi là Log Loss):
    $$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$
    *Trong đó:*
    *   $m$ là số lượng mẫu dữ liệu.
    *   $y^{(i)} \in \{0, 1\}$ là nhãn thực tế của mẫu thứ $i$.
    *   $\hat{y}^{(i)} = \sigma(\theta^T X^{(i)})$ là xác suất dự đoán của mô hình.
*   **Ý nghĩa:** 
    *   Nếu $y = 1$ và $\hat{y} \to 1 \Rightarrow \text{Loss} \to 0$. Nếu $\hat{y} \to 0 \Rightarrow \text{Loss} \to +\infty$ (phạt cực nặng!).
    *   Nếu $y = 0$ và $\hat{y} \to 0 \Rightarrow \text{Loss} \to 0$. Nếu $\hat{y} \to 1 \Rightarrow \text{Loss} \to +\infty$.

#### 3. Tối Ưu Hóa: Gradient Descent (Đạo Hàm)
*   Để tìm bộ trọng số $\theta$ tối ưu nhất giúp giảm thiểu $J(\theta)$, ta tính đạo hàm riêng (Gradient) của $J(\theta)$ theo từng tham số $\theta_j$:
    $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}$$
    *(Công thức đạo hàm này trông cực kỳ tối giản và giống hệt Linear Regression nhờ tính chất toán học vi diệu của hàm Sigmoid!)*
*   Quy tắc cập nhật trọng số (Learning Rate $\alpha$):
    $$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

---

### 📅 Ngày 2: Lập Trình Từ Đầu (Coding From Scratch) & Đánh Giá
Hôm nay bạn sẽ chuyển hóa lý thuyết thành code Python thuần túy (chỉ sử dụng `numpy`). Việc tự code từ đầu giúp bạn hiểu tường tận mọi ngóc ngách của thuật toán mà không phụ thuộc vào thư viện `scikit-learn`.

#### 1. Triển khai code các thành phần cốt lõi:
*   Định nghĩa hàm `sigmoid(z)`
*   Định nghĩa hàm tính toán Loss `compute_loss(y, y_pred)`
*   Viết vòng lặp huấn luyện (Training loop) tính gradient và cập nhật trọng số.
*   Đóng gói thành một class `LogisticRegression` chuẩn chỉ với các hàm `.fit(X, y)` và `.predict(X)`.

#### 2. Thử nghiệm trên tập dữ liệu thực tế:
*   Sử dụng tập dữ liệu phân loại kinh điển như **Iris Dataset** (lọc lấy 2 lớp để phân loại nhị phân) hoặc **Breast Cancer Dataset**.
*   Chuẩn hóa dữ liệu (Feature Scaling) vì Gradient Descent chạy rất nhạy cảm với thang đo của các đặc trưng.

#### 3. Đo lường hiệu năng (Evaluation Metrics):
*   **Accuracy (Độ chính xác):** Tỷ lệ dự đoán đúng trên tổng số.
*   **Confusion Matrix (Ma trận nhầm lẫn):** Hiểu rõ True Positive, False Positive, True Negative, False Negative.
*   **Precision & Recall & F1-Score:** Tại sao Accuracy đôi khi lại "lừa dối" chúng ta (nhất là trong bài toán dữ liệu mất cân bằng)?

---

## 🛠️ Bài Tập Thực Hành Dành Riêng Cho Khang
Tôi đã chuẩn bị sẵn cho bạn một file Jupyter Notebook mẫu có cấu trúc hoàn chỉnh tại:
👉 `Khang_lession/Logistic_Regression/Logistic_Regression_Scratch.ipynb`

**Nhiệm vụ của bạn:**
1. Mở file Notebook đó lên.
2. Đọc kỹ các phần hướng dẫn lý thuyết ngắn gọn.
3. Tự điền code vào các phần trống được đánh dấu `# YOUR CODE HERE`.
4. Chạy thực nghiệm, thay đổi tham số học tập (learning rate, số vòng lặp `epochs`) và quan sát đồ thị hàm Loss giảm dần theo thời gian.

---

## 📚 Tài Liệu Tham Khảo Thêm (Cực Hay)
*   **Video trực quan:** [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) (Kênh giải thích toán học bằng hình ảnh trực quan nhất thế giới).
*   **Bài viết tiếng Việt:** [Blog Machine Learning Cơ Bản - Bài 10: Logistic Regression](https://machinelearningcoban.com/2017/01/27/logisticregression/) của anh Vũ Hữu Tiệp (Cực kỳ chi tiết về toán và code).
*   **Khóa học Kinh điển:** Lớp Machine Learning của thầy Andrew Ng trên Coursera (DeepLearning.AI).

---

*Chúc Khang học tập thật tốt! Hãy mở file Notebook lên và bắt đầu hành trình chinh phục Logistic Regression nhé! 🚀*
