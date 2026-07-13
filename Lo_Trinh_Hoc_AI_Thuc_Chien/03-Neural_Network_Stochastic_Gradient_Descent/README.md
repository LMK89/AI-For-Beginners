# Chương 3: Mạng Neural Nhân Tạo & Tối Ưu Hóa (Stochastic Gradient Descent)

Chào mừng bạn đến với Chương 3! Đây là bước nhảy vọt đưa bạn từ Machine Learning truyền thống bước vào thế giới Học Sâu (Deep Learning) kỳ vĩ. Chúng ta sẽ cùng học cách liên kết toán học đạo hàm giải tích để tối ưu hóa cả một mạng lưới neuron.

---

## 📐 1. Bản Chất Toán Học (Math Foundations)

Trái tim của việc huấn luyện mạng Neural chính là **Thuật toán lan truyền ngược (Backpropagation)**. Để tìm cách điều chỉnh các trọng số $W$ ở các lớp ẩn phía trước sao cho giảm thiểu sai số ở đầu ra, chúng ta sử dụng **Đạo hàm chuỗi (Chain Rule)** trong giải tích toán học.

Giả sử ta có hàm lỗi Loss, giá trị dự đoán đầu ra $a$, đầu vào tuyến tính $z = W a_{trước} + b$. Đạo hàm riêng của Loss theo trọng số $W$ được tính bằng tích các đạo hàm riêng trung gian:
$$\frac{\partial Loss}{\partial W} = \frac{\partial Loss}{\partial a} \times \frac{\partial a}{\partial z} \times \frac{\partial z}{\partial W}$$

---

## 🤝 2. Liên Kết Mô Hình Deep Learning (DL Connection)

1. **Perceptron:** Đơn vị sơ khai nhất (chỉ gồm 1 neuron tuyến tính). Nếu không có hàm kích hoạt phi tuyến, cả một mạng neural sâu dù bao nhiêu lớp cũng chỉ tương đương với một phép biến đổi tuyến tính đơn giản.
2. **Mạng Neural Đa Tầng (Multi-Layer Perceptron - MLP):** Liên kết nhiều neuron xếp thành các lớp ẩn (Hidden Layers).
3. **Hàm Kích Hoạt Phi Tuyến (ReLU, LeakyReLU):** Giúp mạng học được các ranh giới phân loại phi tuyến tính phức tạp trong thế giới thực.
4. **Stochastic Gradient Descent (SGD):** Thay vì tính toán đạo hàm trên toàn bộ tập dữ liệu cực kỳ chậm, SGD tính đạo hàm và cập nhật trọng số liên tục dựa trên từng mẫu dữ liệu đơn lẻ (hoặc nhóm nhỏ - Mini-batch), giúp mô hình hội tụ nhanh hơn gấp nhiều lần.

---

## 💼 3. Công Việc Thực Tế Của Data Analyst / Data Engineer

Trong các bài toán dự báo phức tạp, MLP thường được áp dụng rộng rãi:

1. **Data Analyst / Data Scientist:**
   - Xây dựng mạng MLP dự báo chuỗi thời gian phi tuyến tính phức tạp (như dự báo phụ tải hệ thống điện lưới quốc gia, dự toán doanh số bán hàng mùa cao điểm).
   - Phân tích và phát hiện các mẫu hành vi gian lận tài chính (Fraud Detection) ẩn sâu trong các lớp giao dịch ngân hàng.

2. **Data Engineer:**
   - Thiết kế các Data Pipeline nạp dữ liệu liên tục theo dòng (Streaming Data) như Kafka để đưa vào mạng Neural dự báo thời gian thực.
   - Quản lý và theo dõi quá trình huấn luyện mô hình bằng các công cụ như MLflow.

---

## 💻 4. Lập Trình Python Thuần (Numpy Scratch)

Dưới đây là một đoạn code mẫu cực kỳ gọn mô phỏng quá trình Forward và Backward của 1 Neuron:

```python
import numpy as np

# Định nghĩa hàm kích hoạt ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Quá trình lan truyền xuôi và ngược của 1 nốt đơn giản
inputs = np.array([2.0, 3.0])
weights = np.array([0.5, -0.2])
bias = 0.1

# 1. Forward Pass
z = np.dot(inputs, weights) + bias
output = relu(z)

# 2. Giả sử sai số đạo hàm từ lớp sau truyền về là d_loss_out = 1.0
d_loss_out = 1.0
d_z = d_loss_out * relu_derivative(z)

# 3. Tính Gradient theo trọng số và bias
d_weights = d_z * inputs
d_bias = d_z

# 4. Cập nhật trọng số
learning_rate = 0.01
weights -= learning_rate * d_weights
bias -= learning_rate * d_bias
```

---

## 📊 5. Trực Quan Hóa (Visualization)

Quan sát trực quan hóa cấu trúc kết nối của mạng Neural tại:
👉 `lo_trinh_hoc_tap.html` (Phần 3)
