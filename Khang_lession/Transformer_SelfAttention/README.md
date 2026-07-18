# 🤖 Lộ trình Học Tập & Thực Hành: Transformer & Self-Attention (1 - 2 Tuần)

> **"Self-Attention chính là lực hấp dẫn liên kết các từ trong thế giới ngôn ngữ, mở ra cánh cổng kỷ nguyên Generative AI."**

Chào Khang! Kể từ bài báo lịch sử **"Attention Is All You Need"** (2017) của Google, kiến trúc **Transformer** đã thay thế hoàn toàn các mạng tuần tự cũ như RNN, LSTM để trở thành nền móng vững chắc cho ChatGPT (OpenAI), Claude (Anthropic), Midjourney, hay các hệ thống dự báo chuỗi thời gian tiên tiến nhất. Nếu không hiểu sâu sắc cơ chế **Self-Attention**, bạn sẽ hoàn toàn bị "mất gốc" và không thể đọc hiểu bất kỳ kiến trúc AI hiện đại nào.

Dưới đây là lộ trình học tập toàn diện được thiết kế trong 1 - 2 tuần giúp bạn làm chủ "bộ não" của GenAI.

> 🧭 **Thấy lý thuyết khô khan?** Hãy mở file [USECASE.md](./USECASE.md) trước: học qua 30 câu review thật — xem câu nào chỉ cần "đếm từ" là đoán được, câu nào (phủ định, đảo thứ tự, mỉa mai) khiến mô hình kiểu cũ chết đứng và vì sao phải có Self-Attention, kèm cái giá bộ nhớ O(T²) tính bằng số thật. Toàn bộ thí nghiệm chạy thật, kèm code.

---

## 🎯 Mục Tiêu Cốt Lõi Cần Đạt Được
Sau bài học này, bạn cần phải hiểu và làm chủ:
1. **Cơ chế Self-Attention:** Cách mô hình tự động tìm mối liên kết ngữ nghĩa giữa các từ trong chuỗi thông qua việc tính toán 3 ma trận: **Query (Q)**, **Key (K)**, và **Value (V)**.
2. **Hạn chế và Cải tiến độ phức tạp ($O(T^2)$ vs Linformer $O(T)$):** Hiểu rõ tại sao Self-Attention chuẩn lại ngốn tài nguyên RAM khủng khiếp khi chuỗi dài ra, và cách mô hình **Linformer** giải quyết bài toán này bằng phép chiếu ma trận hạng thấp (Low-rank Projection).
3. **Mở rộng đa phương thức (Multi-modal):** Cách Transformer xử lý hình ảnh bằng việc cắt ảnh thành các ô vuông nhỏ (**Vision Transformer - ViT**) và ứng dụng của nó trong các mô hình khuếch tán tạo ảnh (Stable Diffusion).

---

## 🗓️ Lộ Trình Học Tập Chi Tiết (1 - 2 Tuần)

### 📅 Tuần 1: Trọng Tâm Cốt Lõi - Bản Chất Toán Học Của Self-Attention
Hãy dành tuần này để hiểu thật sâu từng phép nhân ma trận đằng sau cơ chế Attention.

#### 1. Tại sao các mạng RNN/LSTM cũ lại thất bại?
*   RNN xử lý tuần tự từng từ một $\Rightarrow$ Không thể huấn luyện song song (Parallelization) trên GPU $\Rightarrow$ Không thể mở rộng quy mô (Scale up).
*   RNN bị mất trí nhớ ngắn hạn (Vanishing Gradient) $\Rightarrow$ Không thể liên kết các từ đứng quá xa nhau trong văn bản dài.

#### 2. Cơ chế Bộ Ba Q, K, V (Query, Key, Value)
Hãy tưởng tượng bạn đang tìm kiếm video trên YouTube:
*   **Query (Q):** Từ khóa bạn gõ vào ô tìm kiếm (Cái bạn đang muốn truy vấn).
*   **Key (K):** Tiêu đề và thẻ tag của các video hiện có trên hệ thống (Cái để đối sánh độ khớp).
*   **Value (V):** Nội dung thực tế của các video đó (Thông tin bạn sẽ nhận lại).

Công thức **Scaled Dot-Product Attention** kinh điển:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

*Trong đó:*
*   $Q K^T$: Tích vô hướng đo lường mức độ liên quan giữa từ này với toàn bộ các từ khác (độ tương đồng).
*   $\sqrt{d_k}$: Hệ số co giãn (Scale factor) giúp phân tán giá trị trước khi đưa vào hàm Softmax, tránh hiện tượng gradient bị triệt tiêu khi chiều đặc trưng $d_k$ quá lớn.
*   $\text{Softmax}$: Biến đổi các điểm số tương đồng thành phân phối xác suất (tổng bằng 1).
*   Nhân với $V$: Trích xuất lượng thông tin cần thiết dựa trên tỷ lệ chú ý tương ứng.

#### 3. Multi-Head Attention (Chú ý đa đầu)
Thay vì chỉ chú ý theo một góc nhìn duy nhất, Multi-Head Attention chia nhỏ các Query, Key, Value thành nhiều nhánh nhỏ chạy song song. Mỗi đầu (Head) sẽ tự học cách chú ý vào các khía cạnh ngữ nghĩa khác nhau (Ví dụ: Head 1 tập trung vào mối quan hệ Chủ ngữ - Động từ, Head 2 tập trung vào Đại từ thay thế, v.v.).

---

### 📅 Tuần 2: Nâng Cao - Linformer, ViT & Kỹ Thuật Thực Tế

#### 1. Bài Toán Độ Phức Tạp Bậc Hai $O(T^2)$
*   Trong Self-Attention chuẩn, ta phải tính tích $Q K^T$ với kích thước $T \times T$ (với $T$ là độ dài chuỗi văn bản đầu vào).
*   Nếu độ dài chuỗi $T = 1000 \Rightarrow$ Ma trận attention có $1,000,000$ phần tử.
*   Nếu $T = 100,000 \Rightarrow$ Cần tính toán và lưu trữ ma trận có **10 tỷ phần tử**! Lượng RAM GPU sẽ bị quá tải lập tức (Out of Memory - OOM).
*   Độ phức tạp tính toán và bộ nhớ là **Bậc hai $O(T^2)$**.

#### 2. Giải Pháp Linformer: Độ Phức Tạp Tuyến Tính $O(T)$
*(Phần này được nhấn mạnh trong tài liệu slide CS431 - Bài 10 trang 34-38 của bạn)*
*   **Ý tưởng cốt lõi:** Ma trận Attention thực chất có **hạng thấp (low-rank)**, nghĩa là có rất nhiều thông tin dư thừa và ta không cần một ma trận $T \times T$ khổng lồ.
*   **Giải pháp:** Linformer chiếu ma trận Key ($K$) và Value ($V$) từ chiều $T \times d$ xuống chiều thấp hơn $k \times d$ (với $k \ll T$) bằng hai ma trận chiếu tuyến tính $E$ và $F$.
*   Lúc này, ma trận Attention chỉ có kích thước $T \times k$ thay vì $T \times T$.
*   Độ phức tạp giảm từ bậc hai $O(T^2)$ xuống **tuyến tính $O(T \cdot k)$**, cho phép xử lý các văn bản cực kỳ dài mà không sợ cháy GPU!

#### 3. Vision Transformer (ViT) & Đa Phương Thức (Multi-modal)
*   Làm sao đưa hình ảnh vào Transformer khi ảnh là ma trận 2D?
*   **Giải pháp:** ViT cắt bức ảnh 2D thành các ô vuông nhỏ cố định (**Patches**, ví dụ kích thước $16 \times 16$ pixel).
*   Mỗi ô vuông patch được làm phẳng (flatten) và chiếu tuyến tính thành một vector (giống như một "từ" trong văn bản).
*   Đưa các vector này cùng với Positional Encoding vào Transformer Encoder chuẩn để phân loại ảnh hoặc tái tạo ảnh.

---

## 🛠️ Bài Tập & Tài Nguyên Trực Quan Dành Riêng Cho Khang
Tôi đã thiết kế sẵn các công cụ học tập cực chất cho bài học này:
1. **Bộ mô phỏng trực quan tương tác**:
   👉 `Khang_lession/Transformer_SelfAttention/study_guide.html` (Mô phỏng cơ chế tính toán Q, K, V trực quan và So sánh đồ thị OOM $O(T^2)$ với Linformer $O(T)$).
2. **File Jupyter Notebook Thực Hành**:
   👉 `Khang_lession/Transformer_SelfAttention/Transformer_SelfAttention_Practice.ipynb` (Tự tay code cơ chế Scaled Dot-Product Attention bằng PyTorch từ đầu!).

*Chúc Khang học tập thật tốt! Hãy mở file HTML trực quan lên và cùng bước vào thế giới kỳ diệu của GenAI nhé! 🚀*
