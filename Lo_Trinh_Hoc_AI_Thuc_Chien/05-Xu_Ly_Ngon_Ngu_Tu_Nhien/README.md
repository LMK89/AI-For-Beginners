# Chương 5: Xử Lý Ngôn Ngữ Tự Nhiên (Natural Language Processing)

Chào mừng bạn đến với Chương 5! Làm thế nào để máy tính - vốn chỉ hiểu được các con số 0 và 1 - lại có thể đọc hiểu và cảm nhận được ý nghĩa của các từ ngữ phức tạp mà con người nói hàng ngày? Hãy cùng khám phá mối liên kết toán học hình học đằng sau quá trình này.

> 🧭 **Học qua ví dụ trước, lý thuyết sau?** Mở [USECASE.md](./USECASE.md): tự train word embedding trên kho 38 câu rồi soi 30 từ — đồng nghĩa tụ đàn (cos(ngon, tuyệt) = 1.00), nhưng trái nghĩa cũng dính nhau (cos(ngon, dở) = 0.99!), từ đa nghĩa kẹt giữa hai cụm, từ hiếm thành vector rác. Số liệu chạy thật, kèm code.


---

## 📐 1. Bản Chất Toán Học (Math Foundations)

Bản chất của NLP hiện đại là **Không Gian Vectơ Hình Học**. Chúng ta biến đổi mỗi từ ngữ thành một điểm tọa độ trong không gian nhiều chiều (gọi là **Word Embedding**).

Để đo lường xem hai từ ngữ có mang ý nghĩa giống nhau hay không, chúng ta sử dụng công thức **Độ tương đồng Cosine (Cosine Similarity)** trong toán học hình học giải tích để đo góc giữa 2 vectơ:
$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|} = \frac{\sum u_i v_i}{\sqrt{\sum u_i^2} \sqrt{\sum v_i^2}}$$

### Đặc điểm của Cosine Similarity:
- Nếu góc giữa hai vectơ bằng $0^\circ$ (hai từ hoàn toàn đồng nghĩa như "học máy" và "machine learning"), Cosine Similarity = 1.
- Nếu góc giữa hai vectơ bằng $90^\circ$ (hai từ hoàn toàn độc lập, không liên quan), Cosine Similarity = 0.
- Nếu góc giữa hai vectơ bằng $180^\circ$ (hai từ trái ngược nghĩa hoàn toàn), Cosine Similarity = -1.

---

## 🤝 2. Liên Kết Mô Hình Deep Learning (DL Connection)

1. **Bag of Words / TF-IDF:** Cách tiếp cận đếm từ thô sơ, không giữ được thứ tự trước sau của câu.
2. **Word Embeddings (Word2Vec, GloVe):** Học không gian vectơ ngữ nghĩa. Tạo ra hiện tượng toán học kỳ diệu:
   $$\vec{v}_{\text{"Vua"}} - \vec{v}_{\text{"Nam"}} + \vec{v}_{\text{"Nữ"}} \approx \vec{v}_{\text{"Hoàng hậu"}}$$
3. **Mạng Recurrent Neural Network (RNN / LSTM):** Do ngôn ngữ có tính tuần tự (từ đứng trước bổ nghĩa cho từ đứng sau), mạng RNN được thiết kế để truyền trạng thái ẩn (Hidden State) qua từng bước thời gian, giúp mạng "nhớ" được ngữ cảnh của câu văn.

---

## 💼 3. Công Việc Thực Tế Của NLP Engineer / Data Analyst

Trong doanh nghiệp, NLP giúp xử lý khối lượng lớn văn bản không cấu trúc:

1. **Phân tích sắc thái khách hàng (Sentiment Analysis):**
   - Quét toàn bộ bình luận của khách hàng trên Fanpage hoặc sàn TMĐT, tự động phân loại xem khách hàng đang Hài lòng (Tích cực) hay Phẫn nộ (Tiêu cực) để chăm sóc kịp thời.

2. **Hệ thống tìm kiếm thông minh (Semantic Search):**
   - Thay vì tìm kiếm chính xác từng từ khóa (keyword search), hệ thống sử dụng vector embedding để tìm kiếm theo ý nghĩa câu hỏi của người dùng (Ví dụ: Tìm "cách kết nối wifi" ra kết quả "hướng dẫn truy cập mạng không dây").

3. **Phân loại văn bản tự động:**
   - Tự động gắn thẻ, phân loại email khiếu nại của khách hàng gửi về đúng phòng ban xử lý (Tài chính, Kỹ thuật, Bảo hành).

---

## 💻 4. Lập Trình Python Thực Chiến

Dưới đây là cách sử dụng thư viện `SentenceTransformers` hiện đại để tính toán độ tương đồng giữa các câu văn bằng tiếng Việt:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Tải mô hình embedding hỗ trợ đa ngôn ngữ
model = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli')

# 2. Định nghĩa các câu văn cần so khớp
sentences = [
    "Học trí tuệ nhân tạo rất thú vị và có nhiều cơ hội việc làm.",
    "Lập trình học máy thực sự rất cuốn hút.",
    "Hôm nay trời mưa to quá, tôi không đi chơi được."
]

# 3. Chuyển đổi câu văn thành các vector tọa độ
embeddings = model.encode(sentences)

# 4. Tính toán độ tương đồng Cosine giữa câu 0 và câu 1
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

sim_01 = cosine_similarity(embeddings[0], embeddings[1])
sim_02 = cosine_similarity(embeddings[0], embeddings[2])

print(f"Độ tương đồng ngữ nghĩa giữa câu 0 và câu 1: {sim_01:.4f}")
print(f"Độ tương đồng ngữ nghĩa giữa câu 0 và câu 2: {sim_02:.4f}")
```

---

## 📊 5. Trực Quan Hóa (Visualization)

Xem không gian vectơ ngữ nghĩa 3D trực quan tại:
👉 `lo_trinh_hoc_tap.html` (Phần 5)
