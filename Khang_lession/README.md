# 🎓 Khang's Custom AI Learning Workspace

> **"Học máy không phải là học thuộc lòng công thức, mà là hiểu rõ trực quan và tự tay lập trình để giải quyết các bài toán thế giới thực."**

Chào Khang! Đây là thư mục làm việc và lộ trình học tập được thiết kế riêng biệt dựa trên các yêu cầu của bạn. File **README này đóng vai trò là Dashboard Trung tâm** – nơi ghi nhận lại toàn bộ yêu cầu, các ghi chú cốt lõi của từng bài học và cung cấp một Checklist tổng thể giúp bạn theo dõi sát sao tiến độ học tập hàng ngày.

---

## 📌 Lịch Sử Yêu Cầu Học Tập Của Khang (User Requests Log)

Dưới đây là 3 Module học tập trọng tâm bạn đã yêu cầu xây dựng:

1. **Module 1: Hồi quy Logistic (Logistic Regression)**
   * **Thời lượng:** 1 - 2 Ngày
   * **Mục tiêu:** Cánh cửa phân loại, hàm kích hoạt Sigmoid đưa đầu ra tuyến tính về xác suất $(0, 1)$, hàm mất mát Binary Cross-Entropy (BCE) Loss và tối ưu hóa bằng Gradient Descent.
2. **Module 2: XGBoost & LightGBM**
   * **Thời lượng:** 3 - 5 Ngày
   * **Mục tiêu:** Vua thống trị dữ liệu bảng (Tabular Data). Hiểu từ Cây quyết định $\rightarrow$ Bagging & Random Forest $\rightarrow$ Gradient Boosting. Học cách tinh chỉnh tham số chống overfitting, xử lý dữ liệu khuyết thiếu và vẽ Feature Importance đưa ra quyết định kinh doanh.
3. **Module 3: Transformer & Self-Attention**
   * **Thời lượng:** 1 - 2 Tuần
   * **Mục tiêu:** Chìa khóa của kỷ nguyên Generative AI. Hiểu sâu cơ chế Self-Attention thông qua bộ ba Query (Q), Key (K), Value (V). Nắm vững hạn chế chi phí bộ nhớ bậc hai $O(T^2)$ và giải pháp tối ưu tuyến tính $O(T)$ của Linformer. Mở rộng đa phương thức với Vision Transformer (ViT) cắt ảnh thành các patch.

---

## 📁 Sơ Đồ Hệ Thống Tài Nguyên Đã Thiết Kế (Workspace Resources)

Tất cả các tài liệu lý thuyết, trang web tương tác trực quan (Responsive di động iPhone 11) và file bài tập lập trình đã được chuẩn bị đầy đủ tại các thư mục tương ứng:

| Module Bài Học | 📘 Hướng Dẫn Lý Thuyết | 🎨 Trang Web Tương Tác Visual | 💻 File Bài Tập Lập Trình |
| :--- | :--- | :--- | :--- |
| **1. Logistic Regression** | [Lý thuyết & Lộ trình](file:///Users/lekhang/Documents/AI-For-Beginners/Khang_lession/Logistic_Regression/README.md) | [Mô phỏng Sigmoid & BCE Loss](file:///Users/lekhang/Documents/AI-For-Beginners/Khang_lession/Logistic_Regression/study_guide.html) | [Tự code bằng Numpy Scratch](file:///Users/lekhang/Documents/AI-For-Beginners/Khang_lession/Logistic_Regression/Logistic_Regression_Scratch.ipynb) |
| **2. XGBoost & LightGBM** | [Lý thuyết & Lộ trình](file:///Users/lekhang/Documents/AI-For-Beginners/Khang_lession/XGBoost_LightGBM/README.md) | [Mô phỏng Gini & Loss Curves](file:///Users/lekhang/Documents/AI-For-Beginners/Khang_lession/XGBoost_LightGBM/study_guide.html) | [Tuning & Feature Importance](file:///Users/lekhang/Documents/AI-For-Beginners/Khang_lession/XGBoost_LightGBM/XGBoost_LightGBM_Practice.ipynb) |
| **3. Transformer & Self-Attention** | [Lý thuyết & Lộ trình](file:///Users/lekhang/Documents/AI-For-Beginners/Khang_lession/Transformer_SelfAttention/README.md) | [Attention Map & O(T²) vs Linformer](file:///Users/lekhang/Documents/AI-For-Beginners/Khang_lession/Transformer_SelfAttention/study_guide.html) | [Lập trình MHA bằng PyTorch](file:///Users/lekhang/Documents/AI-For-Beginners/Khang_lession/Transformer_SelfAttention/Transformer_SelfAttention_Practice.ipynb) |

---

## 🗒️ Các Ghi Chú Cốt Lõi Quan Trọng (Key Notes)

### 📐 1. Logistic Regression
*   **Hàm Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$. Ranh giới phân loại mặc định tại $z = 0 \Rightarrow \sigma(z) = 0.5$.
*   **Binary Cross-Entropy Loss:** Phạt cực nặng bằng hàm Logarit khi mô hình tự tin sai nhãn. Ví dụ, nhãn thực tế $y=1$ mà dự đoán $\hat{y} \to 0$ thì Loss tiến về $+\infty$.
*   **Gradient Descent:** Đạo hàm riêng của BCE Loss đối với Hồi quy Logistic có dạng toán học tương tự Hồi quy Tuyến tính nhờ tính chất tuyệt vời của hàm Sigmoid.

### 🌲 2. XGBoost & LightGBM
*   **Bagging (Random Forest):** Huấn luyện nhiều cây song song độc lập. Giúp giảm **Variance (Phương sai)**, tức là giảm hiện tượng Overfitting.
*   **Boosting (GBM, XGBoost, LightGBM):** Huấn luyện các cây nông tuần tự, cây sau tập trung sửa sai (residual error) của cây trước. Giúp giảm **Bias (Độ chệch)**.
*   **Overfitting:** Cây quá sâu (`max_depth` lớn) và tốc độ học quá vội vã (`learning_rate` lớn) sẽ khiến sai số tập huấn luyện giảm về 0 nhưng sai số tập kiểm thử tăng mạnh. Hãy tinh chỉnh tăng `min_child_weight` (hoặc `min_data_in_leaf`) và giảm `learning_rate` để khắc phục.
*   **Chuẩn hóa dữ liệu:** Thuật toán cây phân nhánh dựa trên việc so sánh ngưỡng (ví dụ: $Age > 30$) chứ không nhân ma trận như Neural Network, do đó **không bắt buộc** phải thực hiện chuẩn hóa dữ liệu (`StandardScaler`).

### 🤖 3. Transformer & Self-Attention
*   **Query, Key, Value:** Tương tự hệ thống YouTube. Query ($Q$) là từ khóa bạn gõ, Key ($K$) là tiêu đề các video để so sánh độ khớp (dot product), Value ($V$) là nội dung video trích xuất ra dựa trên trọng số attention phù hợp.
*   **Độ phức tạp bậc hai $O(T^2)$:** Self-Attention chuẩn yêu cầu mỗi từ phải so khớp tương quan với mọi từ khác, khiến bộ nhớ phình to theo bình phương độ dài văn bản $T$. Gây sập bộ nhớ VRAM của GPU (Out of Memory - OOM) khi văn bản quá dài.
*   **Linformer:** Sử dụng ma trận chiếu tuyến tính $E, F$ để nén Key và Value từ chiều $T \times d$ xuống không gian hạng thấp $k \times d$ (với $k \ll T$). Giúp giảm độ phức tạp về tuyến tính **$O(T \cdot k)$**, xử lý văn bản cực dài cực kỳ nhẹ nhàng.
*   **Vision Transformer (ViT):** Chia bức ảnh 2D thành các miếng **Patch** vuông phẳng kích thước $16 \times 16$ pixel để đưa vào Transformer Encoder tương tự như các token từ ngữ của văn bản.

---

## 🗹 Master Checklist Theo Dõi Tiến Độ Học Tập

Khang hãy tích chọn các mục dưới đây (thay đổi `[ ]` thành `[x]`) trực tiếp trong trình soạn thảo code của bạn mỗi khi hoàn thành từng phần nhé!

### 🟩 Hồi quy Logistic (Lộ trình 1 - 2 ngày)
- [ ] **Lý thuyết 1.1**: Đọc hiểu bản chất ánh xạ tuyến tính sang xác suất của hàm Sigmoid.
- [ ] **Lý thuyết 1.2**: Hiểu cơ chế hoạt động phạt của hàm mất mát Binary Cross-Entropy Loss.
- [ ] **Lý thuyết 1.3**: Nắm vững nguyên lý tính Gradient Descent để cập nhật trọng số $\theta$.
- [ ] **Tương tác trực quan**: Mở `Logistic_Regression/study_guide.html` trên máy tính hoặc điện thoại kéo thử slider và xem hoạt động.
- [ ] **Bài tập Code**: Hoàn thành lập trình Numpy từ con số không trong file `Logistic_Regression_Scratch.ipynb` và pass qua toàn bộ các bộ test tự động.

### 🟩 XGBoost & LightGBM (Lộ trình 3 - 5 ngày)
- [ ] **Lý thuyết 2.1**: Nắm vững cơ chế Decision Tree và cách tính độ không thuần khiết Gini, Entropy.
- [ ] **Lý thuyết 2.2**: Phân biệt rõ cơ chế Ensemble Learning: Bagging (Random Forest) song song vs Boosting tuần tự sửa sai.
- [ ] **Lý thuyết 2.3**: Nắm vững sự khác biệt giữa mọc cây theo chiều rộng Level-wise (XGBoost) và mọc cây theo chiều sâu Leaf-wise (LightGBM).
- [ ] **Tương tác trực quan**: Mở `XGBoost_LightGBM/study_guide.html` kéo thử simulator tính toán Gini và quan sát đường cong Overfitting thay đổi theo tham số.
- [ ] **Bài tập Code**: Thực hành lập trình trên dữ liệu bảng dự đoán Churn tín dụng trong file `XGBoost_LightGBM_Practice.ipynb`. Tinh chỉnh siêu tham số và vẽ biểu đồ Feature Importance.

### 🟩 Transformer & Self-Attention (Lộ trình 1 - 2 tuần)
- [ ] **Lý thuyết 3.1**: Hiểu rõ điểm yếu của RNN/LSTM tuần tự và lý do Transformer thống trị kỷ nguyên song song hóa.
- [ ] **Lý thuyết 3.2**: Thuộc lòng ý nghĩa và cách tính toán Scaled Dot-Product Attention từ ma trận Q, K, V.
- [ ] **Lý thuyết 3.3**: Giải thích được bài toán nghẽn bộ nhớ $O(T^2)$ và cải tiến tối ưu tuyến tính $O(T)$ của Linformer.
- [ ] **Lý thuyết 3.4**: Hiểu cơ chế phân nhỏ ảnh thành Patch trong Vision Transformer (ViT).
- [ ] **Tương tác trực quan**: Mở `Transformer_SelfAttention/study_guide.html` tương tác chọn từ xem luồng chú ý và biểu đồ VRAM sập khi T quá lớn.
- [ ] **Bài tập Code**: Thực hành viết PyTorch cơ chế Scaled Dot-Product Attention và Multi-Head Attention từ con số không trong file `Transformer_SelfAttention_Practice.ipynb`. Vẽ biểu đồ nhiệt Attention Heatmap.

---

*Chúc Khang học tập thật xuất sắc! Không ngừng học hỏi, không ngừng nâng cấp bản thân nhé! 🚀*
