# Chương 6: Transformers & Kỷ Nguyên Generative AI (GenAI)

Chào mừng bạn đến với Chương 6 - đỉnh cao của lộ trình học tập AI thực chiến! Đây là nơi chúng ta gộp toàn bộ các mảnh ghép toán học ma trận, hàm kích hoạt Softmax, và vector embedding để xây dựng nên cấu trúc "bộ não" đứng sau ChatGPT - cơ chế **Self-Attention**.

> 🧭 **Học qua ví dụ trước, lý thuyết sau?** Mở [USECASE về Transformer & Self-Attention](../../Khang_lession/Transformer_SelfAttention/USECASE.md): 30 câu review — chứng minh mô hình đếm từ bắt buộc sai với cặp câu đảo thứ tự, vì sao cần Attention, và cái giá bộ nhớ O(T²) bằng số thật (T=131k → 68,7 GB). Kèm code.


---

## 📐 1. Bản Chất Toán Học (Math Foundations)

Trái tim của Transformer là cơ chế **Scaled Dot-Product Attention**. Bản chất toán học của nó là phép nhân ma trận và hàm xác suất Softmax.

Chúng ta có 3 ma trận biểu thị thông tin:
- **Query (Q):** Từ khóa chúng ta đang đi tìm kiếm thông tin.
- **Key (K):** Tiêu đề của các nguồn thông tin để đối sánh độ khớp.
- **Value (V):** Nội dung thực tế chúng ta trích xuất được.

Công thức tính toán:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### Chi tiết các bước toán học:
1. Phép nhân ma trận $Q K^T$: Đo độ tương đồng ngữ nghĩa giữa mọi cặp từ với nhau trong câu văn.
2. Chia cho $\sqrt{d_k}$: Hệ số co giãn (scale) để tránh hiện tượng đạo hàm bị triệt tiêu khi kích thước vector quá lớn.
3. Hàm Softmax: Ánh xạ điểm tương đồng về dạng phân phối xác suất chú ý (tổng bằng 1).
4. Nhân với ma trận Value $V$: Trích xuất lượng thông tin quan trọng của từ ngữ dựa trên trọng số attention tương ứng.

---

## 🤝 2. Liên Kết Mô Hình Deep Learning (DL Connection)

1. **Hạn chế của RNN/LSTM cũ:** Xử lý tuần tự từng từ một nên không thể tận dụng sức mạnh tính toán song song (Parallelization) của GPU, làm chậm tốc độ huấn luyện trên dữ liệu lớn.
2. **Transformer:** Loại bỏ hoàn toàn tính tuần tự. Sử dụng cơ chế Self-Attention kết hợp với **Positional Encoding** (mã hóa vị trí) giúp mô hình có thể nhìn thấy toàn bộ câu văn cùng lúc, xử lý song song siêu tốc trên GPU và hiểu ngữ cảnh cực kỳ dài.
3. **Bài toán độ phức tạp bậc hai $O(T^2)$:** Trong cơ chế Attention chuẩn, ta phải tính ma trận $T \times T$ (với $T$ là độ dài chuỗi). Nếu $T = 100,000$, ma trận sẽ chứa 10 tỷ phần tử gây sập bộ nhớ GPU (Out Of Memory - OOM).
4. **Giải pháp Linformer:** Chiếu ma trận Key ($K$) và Value ($V$) từ không gian $T \times d$ xuống không gian hạng thấp (low-rank) $k \times d$ (với $k \ll T$). Giúp giảm độ phức tạp tính toán xuống tuyến tính **$O(T)$**, xử lý văn bản cực dài cực kỳ nhẹ nhàng.

---

## 💼 3. Công Việc Thực Tế Của Generative AI Engineer

Generative AI đang thay đổi cách vận hành của mọi doanh nghiệp toàn cầu:

1. **Hệ thống RAG (Retrieval-Augmented Generation):**
   - Xây dựng Chatbot thông minh cho doanh nghiệp bằng cách kết nối mô hình ngôn ngữ lớn (như GPT-4, Llama-3) với kho tài liệu quy trình, chính sách nội bộ dạng PDF để Chatbot trả lời chính xác, không bị "bịa đặt" (hallucination) thông tin.

2. **Dịch thuật đa ngôn ngữ thời gian thực:**
   - Xây dựng các cổng dịch thuật tài liệu pháp lý, hợp đồng thương mại quốc tế chất lượng cao.

3. **Hệ thống tóm tắt văn bản và chăm sóc khách hàng tự động.**

---

## 💻 4. Lập Trình Python Thực Chiến PyTorch

Dưới đây là cách lập trình cơ chế toán học Scaled Dot-Product Attention bằng PyTorch từ đầu:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    # Kích thước chiều của ma trận Key
    d_k = Q.size(-1)

    # 1. Nhân ma trận đo độ tương đồng
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # 2. Co giãn điểm số (Scale)
    scores = scores / math.sqrt(d_k)

    # 3. Áp dụng Softmax để lấy trọng số xác suất chú ý
    attention_weights = F.softmax(scores, dim=-1)

    # 4. Nhân với ma trận Value để trích xuất thông tin
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

---

## 📊 5. Trực Quan Hóa (Visualization)

Xem biểu đồ ma trận nhiệt độ chú ý (Attention Heatmap) và cơ chế Linformer tại:
👉 `lo_trinh_hoc_tap.html` (Phần 6)
