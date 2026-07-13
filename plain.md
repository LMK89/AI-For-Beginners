# Lộ Trình Học Tập AI Thực Chiến Cho Người Trái Ngành (Xen Kẽ & Thực Hành)

Chào bạn! Lộ trình này được thiết kế dành riêng cho người mới bắt đầu hoặc người trái ngành muốn nắm vững Trí tuệ Nhân tạo một cách thực tế nhất. Thay vì học lý thuyết khô khan suốt một chương dài, lộ trình của chúng ta sẽ học theo hình thức **xen kẽ (mixed)**:
> **Toán học -> Liên kết với mô hình ML/DL -> Công việc của Data Analyst / Data Engineer -> Code Python -> Trực quan hóa (Visual) công thức.**

Học tới đâu, xài được tới đó ngay lập tức!

---

## 🗓️ Lịch Trình Chi Tiết Các Phần Học

### 📅 Phần 1: Toán Thống Kê & Hồi Quy Logistic (Logistic Regression)
*   **Từ Khóa (Keywords):** `Hàm Sigmoid`, `Binary Cross-Entropy Loss`, `Gradient Descent`, `Phân loại Nhị phân (Binary Classification)`.
*   **Mối Liên Quan Lẫn Nhau:**
    *   Học **Toán thống kê** về xác suất và logarit.
    *   Liên kết trực tiếp tới mô hình **Machine Learning**: Hồi quy Logistic sử dụng hàm Sigmoid để biến đổi một giá trị tuyến tính bất kỳ ($Wx + b$) về xác suất nằm trong khoảng $(0, 1)$ nhằm phân loại đối tượng.
    *   Tối ưu hóa trọng số nhờ giải thuật **Gradient Descent** (đạo hàm riêng của hàm mất mát Cross-Entropy).
*   **Study Case Doanh Nghiệp (Thực Chiến):**
    *   **Data Analyst:** Sử dụng mô hình để phân tích tỷ lệ khách hàng rời bỏ dịch vụ viễn thông (Customer Churn Prediction) dựa trên hóa đơn và lịch sử gọi điện.
    *   **Data Engineer:** Tạo pipeline nạp dữ liệu sạch từ SQL database và chuẩn hóa các trường đặc trưng (Feature Scaling) giúp mô hình hội tụ nhanh hơn.
*   **Lập Trình Thực Chiến:** Tự viết Class `LogisticRegressionScratch` hoàn chỉnh bằng thư viện `numpy` từ số không. Vẽ đồ thị hàm Loss giảm dần qua từng vòng lặp (epochs).

---

### 📅 Phần 2: Cây Quyết Định & Các Mô Hình Ensemble (XGBoost & LightGBM)
*   **Từ Khóa (Keywords):** `Cây quyết định (Decision Tree)`, `Entropy & Gini`, `Ensemble Learning`, `Bagging (Random Forest)`, `Gradient Boosting (XGBoost/LightGBM)`, `Feature Importance`.
*   **Mối Liên Quan Lẫn Nhau:**
    *   **Toán học:** Sử dụng công thức tính độ hỗn loạn Entropy và độ không thuần khiết Gini của Toán xác suất để quyết định điểm chia nhánh tối ưu nhất.
    *   **Liên kết ML:** Sự phát triển từ một Cây quyết định đơn lẻ dễ bị Overfit -> Học kết hợp song song Bagging (Random Forest) -> Học tuần tự Boosting sửa sai lỗi (GBM) -> XGBoost & LightGBM siêu tốc độ.
*   **Study Case Doanh Nghiệp (Thực Chiến):**
    *   **Data Analyst:** Tìm kiếm các thuộc tính quan trọng nhất ảnh hưởng đến việc phê duyệt hồ sơ vay vốn ngân hàng (Credit Scoring) thông qua thuộc tính `.feature_importances_`.
    *   **Data Engineer:** Tạo pipeline tự động hóa huấn luyện mô hình dự toán giá nhà đất lớn hàng triệu dòng bằng LightGBM cực nhanh.
*   **Lập Trình Thực Chiến:** Sử dụng các thư viện `xgboost` và `lightgbm` để giải quyết bài toán tabular thực tế, vẽ biểu đồ Feature Importance và tinh chỉnh siêu tham số chống Overfitting (`max_depth`, `learning_rate`).

---

### 📅 Phần 3: Mạng Neural Nhân Tạo & Tối Ưu Hóa (SGD)
*   **Từ Khóa (Keywords):** `Perceptron`, `Mạng Neural Đa Tầng (MLP)`, `Lan Truyền Ngược (Backpropagation)`, `Đạo hàm chuỗi (Chain Rule)`, `Stochastic Gradient Descent (SGD)`.
*   **Mối Liên Quan Lẫn Nhau:**
    *   **Toán học:** Sử dụng phép toán giải tích đạo hàm chuỗi (Chain Rule) để tính độ dốc sai số và lan truyền ngược về phía trước điều chỉnh trọng số.
    *   **Liên kết DL:** Chuyển dịch từ thuật toán Perceptron một nốt tuyến tính đơn giản sang mạng neural sâu có khả năng học các mối quan hệ phi tuyến phức tạp nhờ các hàm kích hoạt phi tuyến (ReLU, Sigmoid).
*   **Study Case Doanh Nghiệp (Thực Chiến):**
    *   **Data Analyst / Machine Learning Engineer:** Dự báo nhu cầu chuỗi cung ứng dài hạn hoặc nhận diện mẫu hành vi gian lận thẻ tín dụng tinh vi trong các giao dịch trực tuyến.
*   **Lập Trình Thực Chiến:** Viết một mạng Neural Network 2 lớp từ số không bằng Python, lập trình quá trình Forward Pass, Backward Pass và tối ưu hóa trọng số.

---

### 📅 Phần 4: Học Sâu Cho Thị Giác Máy Tính (Computer Vision)
*   **Từ Khóa (Keywords):** `Mạng Tích Chập (CNN)`, `Phép toán tích chập (Convolution)`, `Bộ lọc (Kernel/Filter)`, `Pooling`, `Phân loại ảnh (Image Classification)`.
*   **Mối Liên Quan Lẫn Nhau:**
    *   **Toán học:** Phép toán tích chập bản chất là phép nhân ma trận chập 2D giữa bộ lọc nhỏ (Kernel) với ma trận điểm ảnh đầu vào để trích xuất các đặc trưng biên cạnh.
    *   **Liên kết Deep Learning & Vision:** Sử dụng mạng CNN thay vì mạng MLP phẳng thông thường để bảo toàn cấu trúc không gian 2D của bức ảnh, giúp máy tính tự động "nhìn" được ảnh.
*   **Study Case Doanh Nghiệp (Thực Chiến):**
    *   **Computer Vision Engineer:** Xây dựng hệ thống camera thông minh tự động phát hiện lỗi sản phẩm trên dây chuyền sản xuất nhà máy thông minh, điểm danh khuôn mặt tự động.
*   **Lập Trình Thực Chiến:** Thiết kế và huấn luyện một mạng CNN nhận dạng chữ số viết tay (MNIST) hoặc phân loại chó/mèo bằng thư viện hiện đại PyTorch.

---

### 📅 Phần 5: Xử Lý Ngôn Ngữ Tự Nhiên (NLP)
*   **Từ Khóa (Keywords):** `Word Embeddings`, `Khoảng cách Cosine (Cosine Similarity)`, `Mạng Tuần Tự (RNN/LSTM)`, `Tokenizer`.
*   **Mối Liên Quan Lẫn Nhau:**
    *   **Toán học:** Sử dụng độ đo góc giữa hai vectơ (Cosine Similarity) trong không gian nhiều chiều để đo lường mức độ tương đồng ngữ nghĩa giữa hai từ ngữ.
    *   **Liên kết NLP:** Kỹ thuật chuyển đổi từ ngôn ngữ con người (text) sang dạng số học (Word Embeddings) để đưa vào mạng tuần tự RNN/LSTM để phân tích chuỗi thời gian của từ ngữ.
*   **Study Case Doanh Nghiệp (Thực Chiến):**
    *   **NLP Engineer / Data Analyst:** Phân tích sắc thái (Sentiment Analysis) các bình luận của người dùng trên mạng xã hội hoặc xây dựng hệ thống hỏi đáp tự động thông minh (FAQ Bot).
*   **Lập Trình Thực Chiến:** Sử dụng thư viện `gensim` hoặc Hugging Face để tải mô hình Word2Vec, thực hiện các phép cộng trừ vectơ ngôn ngữ nổi tiếng (ví dụ: "King" - "Man" + "Woman" = "Queen") và đo khoảng cách Cosine.

---

### 📅 Phần 6: Transformers & Kỷ Nguyên Generative AI (GenAI)
*   **Từ Khóa (Keywords):** `Self-Attention`, `Query, Key, Value (Q, K, V)`, `Hạn chế O(T²)`, `Linformer (Tối ưu tuyến tính)`, `Vision Transformer (ViT)`.
*   **Mối Liên Quan Lẫn Nhau:**
    *   **Toán học:** Phép nhân ma trận và hàm Softmax để tính ma trận trọng số chú ý tương quan giữa các từ.
    *   **Liên kết GenAI:** Trái tim của các kiến trúc lớn như ChatGPT. Khắc phục hạn chế của RNN (không song song hóa được trên GPU) bằng cơ chế tự chú ý giúp học song song và xử lý ngữ cảnh cực kỳ dài.
*   **Study Case Doanh Nghiệp (Thực Chiến):**
    *   **Generative AI Engineer:** Xây dựng hệ thống tra cứu văn bản quy chế nội bộ doanh nghiệp thông minh (hệ thống RAG - Retrieval-Augmented Generation) kết hợp với các Large Language Models.
*   **Lập Trình Thực Chiến:** Viết cơ chế Scaled Dot-Product Attention và Multi-Head Attention bằng PyTorch từ đầu. Vẽ biểu đồ nhiệt (Heatmap) thể hiện mức độ chú ý giữa các từ ngữ trong câu.

---

## 🚀 Trực Quan Hóa Mối Liên Kết
Hãy mở trang web tương tác để xem biểu đồ, flowchart và hiệu ứng chuyển động trực quan về mối quan hệ giữa các bài học tại:
👉 `lo_trinh_hoc_tap.html`
