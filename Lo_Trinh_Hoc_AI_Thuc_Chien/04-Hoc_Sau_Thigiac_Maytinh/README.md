# Chương 4: Học Sâu Cho Thị Giác Máy Tính (Computer Vision)

Chào mừng bạn đến với Chương 4! Ở chương này, chúng ta sẽ liên kết trực tiếp phép toán ma trận trong đại số tuyến tính với cấu trúc điểm ảnh của một bức ảnh 2D, giải thích cách máy tính có thể tự động "nhìn" và phân loại hình ảnh.
> 🧭 **Học qua ví dụ trước, lý thuyết sau?** Mở [USECASE.md](./USECASE.md): 30 bức ảnh 8×8 — mạng phẳng đạt 100% tập học nhưng 0/6 trên ảnh vạch dịch chỗ, còn tích chập + padding + pooling đúng 6/6. Hiểu kernel, padding, pooling qua số đo thật, kèm code.


---

## 📐 1. Bản Chất Toán Học (Math Foundations)

Bản chất của Học sâu cho Thị giác Máy tính là **Phép toán Tích chập (Convolution)** trên ma trận 2D.

Một bức ảnh số thực chất là một ma trận hai chiều (2D), trong đó mỗi phần tử là giá trị độ sáng của điểm ảnh (pixel) từ 0 đến 255. Phép tích chập là việc nhân quét một ma trận bộ lọc nhỏ (gọi là **Kernel** hoặc **Filter**, ví dụ kích thước $3 \times 3$) trượt trên ma trận ảnh để tính tích vô hướng tại từng vị trí:
$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m, n)$$

### Ví dụ về Kernel phát hiện cạnh dọc (Sobel Filter):
$$\text{Kernel} = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$
Khi trượt bộ lọc này qua ảnh, nếu có sự thay đổi đột ngột về độ sáng giữa bên trái và bên phải điểm ảnh, kết quả tích chập sẽ rất lớn, giúp phát hiện ra đường biên cạnh dọc của vật thể.

---

## 🤝 2. Liên Kết Mô Hình Deep Learning (DL Connection)

Tại sao không dùng mạng phẳng MLP cho ảnh? Vì ảnh có kích thước lớn (ví dụ $1000 \times 1000$ pixels) sẽ tạo ra hàng triệu tham số kết nối, gây quá tải bộ nhớ và làm mất đi thông tin cấu trúc không gian (mối quan hệ giữa các pixel đứng cạnh nhau).

**Mạng Tích Chập (Convolutional Neural Network - CNN)** ra đời để giải quyết bài toán này:
1. **Lớp Tích chập (Convolutional Layer):** Sử dụng các bộ lọc trượt để tự động trích xuất đặc trưng biên cạnh, hình khối đơn giản đến phức tạp.
2. **Lớp Pooling (Max Pooling / Average Pooling):** Thu nhỏ kích thước ma trận ảnh giúp giảm lượng tính toán và làm cho mô hình bất biến trước các dịch chuyển nhỏ của vật thể trong ảnh.
3. **Lớp Kết nối đầy đủ (Fully Connected Layer):** Phẳng hóa các đặc trưng học được để đưa ra dự đoán lớp ảnh (ví dụ: Chó, Mèo, Chữ số).

---

## 💼 3. Công Việc Thực Tế Của Computer Vision Engineer

Công nghệ CNN được ứng dụng rộng rãi trong đời sống doanh nghiệp:

1. **Kiểm thử chất lượng sản phẩm (Visual Inspection):**
   - Lắp đặt camera trên băng chuyền nhà máy tự động, sử dụng mô hình CNN để phát hiện ra các vết nứt, sản phẩm lỗi (Anomalies) thời gian thực.

2. **Chăm sóc sức khỏe & Y tế (Medical Imaging):**
   - Phân tích ảnh chụp X-quang, MRI để tự động khoanh vùng khối u hoặc phát hiện sớm các dấu hiệu bệnh phổi.

3. **Giao thông thông minh:**
   - Hệ thống camera nhận diện biển số xe tự động (ANPR), phát hiện xe vượt đèn đỏ để phạt nguội.

---

## 💻 4. Lập Trình Thực Chiến PyTorch

Dưới đây là cách chúng ta định nghĩa một mạng CNN đơn giản bằng thư viện PyTorch:

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Lớp tích chập nhận ảnh 1 kênh màu (Grayscale), xuất ra 16 kênh đặc trưng
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Lớp Pooling giảm kích thước ma trận đi một nửa
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Lớp phân loại phẳng hóa đặc trưng
        self.fc = nn.Linear(16 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Phẳng hóa ma trận 2D thành vector 1D
        x = self.fc(x)
        return x
```

---

## 📊 5. Trực Quan Hóa (Visualization)

Xem ảnh động trực quan hóa cơ chế trượt của bộ lọc tích chập tại:
👉 `lo_trinh_hoc_tap.html` (Phần 4)
