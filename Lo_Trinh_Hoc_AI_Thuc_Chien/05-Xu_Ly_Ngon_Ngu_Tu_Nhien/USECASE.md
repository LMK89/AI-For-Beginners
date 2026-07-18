# 🧭 USECASE: Word Embeddings Dùng Khi Nào? — Học Qua 30 Từ Với Số Đo Thật

> **Dành cho người chưa biết gì:** Bài này trả lời câu hỏi trong README chương 5 — *"làm sao máy tính hiểu nghĩa của từ?"* — bằng cách **tự tay train một bộ word embedding thật** trên kho 38 câu tiếng Việt, rồi soi 30 từ dưới kính hiển vi: từ nào máy "hiểu" đúng, từ nào máy hiểu sai, và **vì sao**. Mọi con số cosine similarity đều chạy thật bằng đúng phương pháp trong sách (đếm đồng xuất hiện → PPMI → SVD, tổ tiên của Word2Vec/GloVe). Code cuối file.

---

## 1. TL;DR — Tóm tắt 30 giây

| Câu hỏi | Trả lời ngắn gọn |
| :--- | :--- |
| Word Embedding làm gì? | Biến mỗi từ thành một **tọa độ trong không gian** sao cho từ gần nghĩa nằm gần nhau. Số thật: cos("ngon", "tuyệt") = **1.00**; cos("ngon", "bóng") = **0.00**. |
| So với one-hot (mỗi từ một ô riêng)? | One-hot: **mọi cặp từ đều có cosine = 0** — "ngon" với "tuyệt" xa lạ y như "ngon" với "bóng". Máy không có khái niệm "gần nghĩa". |
| Nó học nghĩa từ đâu? | Từ **ngữ cảnh**: "bạn hãy cho tôi biết bạn của từ là ai, tôi sẽ nói từ đó nghĩa là gì". Từ nào hay đứng cạnh những từ giống nhau → vector gần nhau. |
| Cạm bẫy chết người? | Chính vì học từ ngữ cảnh: **trái nghĩa cũng đứng gần nhau** (số thật: cos("ngon", "dở") = **0.99**!) vì "món này ngon quá" và "món này dở quá" có ngữ cảnh y hệt. Làm sentiment analysis bằng embedding thô là dính bẫy này ngay. |

---

## 2. Bài Toán: Dạy Máy "Hiểu" 57 Từ Từ Kho 38 Câu

Kho văn bản: 38 câu ngắn kiểu *"món này ngon quá"*, *"phim này cuốn lắm"*, *"cầu thủ đá bóng vào lưới"*, *"cho thêm đá vào ly trà"*... Quy trình đúng như README:

1. Đếm ma trận **đồng xuất hiện**: từ nào đứng cạnh từ nào (cửa sổ ±2 từ), bao nhiêu lần.
2. Chuẩn hóa bằng **PPMI** (ưu tiên các cặp đứng cạnh nhau "bất thường nhiều" so với ngẫu nhiên).
3. Nén bằng **SVD** xuống còn 6 chiều → mỗi từ một vector 6 số.
4. Đo **Cosine Similarity** giữa các vector (đúng công thức trong README).

Word2Vec/GloVe làm tinh vi hơn (mạng neural, negative sampling) nhưng **cùng nguyên lý phân phối**: nghĩa của từ = tập ngữ cảnh nó xuất hiện. Mọi hiện tượng dưới đây đều xảy ra y hệt với Word2Vec thật.

---

## 3. Bảng 30 Từ — Máy Hiểu Đúng Từ Nào, Sai Từ Nào?

Cột **Hàng xóm gần nhất** = 2 từ có cosine cao nhất với từ đó (kết quả chạy thật). Cột **SL** = số lần từ xuất hiện trong kho.

### 🟢 Nhóm A (từ 1–5): Từ khen — đồng nghĩa tự động tụ đàn

| # | Từ | SL | Hàng xóm gần nhất | Nhận xét |
|---|:---|---:|:---|:---|
| 1 | ngon | 5 | tuyệt (1.00), dở (0.99) | Bắt được "tuyệt" là anh em — điều one-hot vĩnh viễn không làm nổi |
| 2 | tuyệt | 3 | ngon (1.00), dở (0.98) | — |
| 3 | hấp_dẫn | 2 | hay (1.00), nhạt (1.00) | Nhận đúng họ hàng "hay" |
| 4 | hay | 4 | hấp_dẫn (1.00), bài (0.99) | — |
| 5 | cuốn | 3 | chán (1.00), nhạt (1.00) | Khoan — sao hàng xóm của "cuốn" lại là "chán"?! Xem nhóm B |

**Điều kỳ diệu:** không ai dạy máy "ngon nghĩa giống tuyệt" — nó tự suy ra vì cả hai cùng hay đứng trong khung *"món này ___ quá"*. Với one-hot, cos(ngon, tuyệt) = 0 — tri thức này không tồn tại.

### ⚠️ Nhóm B (từ 6–9): Từ chê — và CẠM BẪY TRÁI NGHĨA nổi tiếng

| # | Từ | SL | Hàng xóm gần nhất | Nhận xét |
|---|:---|---:|:---|:---|
| 6 | dở | 4 | này (0.99), **ngon (0.99)** | Trái nghĩa mà cosine 0.99! |
| 7 | tệ | 3 | bài (0.99), **hay (0.99)** | Tương tự |
| 8 | chán | 3 | cuốn (1.00), nhạt (1.00) | Dính chặt với từ khen "cuốn" |
| 9 | nhạt | 2 | cuốn (1.00), chán (1.00) | — |

**Đây là phát hiện quan trọng nhất bài:** "món này **ngon** quá" và "món này **dở** quá" — ngữ cảnh giống nhau từng chữ, chỉ khác đúng từ đang xét. Mà nguyên lý của embedding là *"ngữ cảnh giống nhau → vector gần nhau"* → **trái nghĩa bị hút vào nhau**. Word2Vec/GloVe thật trên tỷ câu cũng dính ("good" và "bad" nổi tiếng là gần nhau). Hệ quả thực chiến: đừng làm phân tích cảm xúc chỉ bằng khoảng cách embedding thô — cần lớp phân loại học thêm chiều "sắc thái", hoặc dùng mô hình ngữ cảnh (Chương 6).

### 🟢 Nhóm C (từ 10–20): Danh từ chủ đề — bản đồ tự vẽ thành cụm

| # | Từ | SL | Hàng xóm gần nhất | Nhận xét |
|---|:---|---:|:---|:---|
| 10 | món | 6 | lắm (1.00), bánh (1.00) | Cụm ẩm thực |
| 11 | cơm | 5 | quán (0.98), phở (0.96) | — |
| 12 | phở | 3 | đắt (1.00), rẻ (1.00) | Dính với giá cả — vì câu "cơm này dở nhưng rẻ" |
| 13 | bánh | 3 | ghê (1.00), thiệt (1.00) | — |
| 14 | phim | 5 | truyện (1.00), hát (1.00) | Cụm giải trí |
| 15 | truyện | 4 | phim (1.00), hát (1.00) | — |
| 16 | nhạc | 2 | hát (1.00), phim (1.00) | — |
| 17 | trà | 3 | ít (0.99), thôi (0.99) | Cụm đồ uống |
| 18 | bóng | 2 | cú (1.00), có (0.99) | Cụm bóng đá |
| 19 | cầu_thủ | 2 | đẹp (0.99), lưới (0.99) | — |
| 20 | lưới | 1 | đẹp (1.00), cầu_thủ (0.99) | — |

Số đo giữa các cụm: cos(phim, truyện) = **1.00** (cùng cụm), cos(món, cơm) = **0.85**, nhưng cos(trà, bóng) = **0.13** và cos(ngon, bóng) = **0.00** — các chủ đề không liên quan **tự động tách xa nhau**. Đây là nền của mọi hệ tìm kiếm ngữ nghĩa (semantic search): gõ "quán phở ngon" tìm ra được bài viết chứa "tiệm hủ tiếu tuyệt hảo" dù không trùng từ nào.

### 🔴 Nhóm D (từ 21): Từ đa nghĩa — một vector không thể ngồi hai ghế

| # | Từ | SL | Hàng xóm gần nhất | Nhận xét |
|---|:---|---:|:---|:---|
| 21 | đá | 6 | vào (1.00), trực_tiếp (0.99) | "nước **đá**" với "**đá** bóng" là MỘT vector! |

Số đo thật: cos(đá, trà) = **0.74** và cos(đá, bóng) = **0.75** — vector "đá" bị kéo căng **kẹt chính giữa hai nghĩa**, không thuộc hẳn cụm nào. Word2Vec/GloVe cấp mỗi từ đúng MỘT vector cố định, bất kể câu nào. Đây chính là lỗ hổng mà **contextual embedding** (BERT, Transformer — Chương 6) sinh ra để vá: cùng chữ "đá" nhưng trong câu trà đá thì vector ngả về đồ uống, trong câu bóng đá thì ngả về thể thao.

### 🔴 Nhóm E (từ 22–24): Từ hiếm — xuất hiện 1 lần, vector là rác

| # | Từ | SL | Hàng xóm gần nhất | Nhận xét |
|---|:---|---:|:---|:---|
| 22 | tuyệt_vời | 1 | ông (1.00), mặt_trời (1.00) | Chỉ xuất hiện trong câu "tuyệt vời ông mặt trời" |
| 23 | ông | 1 | tuyệt_vời (1.00), mặt_trời (1.00) | Cả 3 từ ôm cứng lấy nhau |
| 24 | mặt_trời | 1 | tuyệt_vời (1.00), ông (1.00) | — |

Nguy hiểm ở chỗ: "tuyệt_vời" đáng lẽ phải đồng nghĩa với "tuyệt", nhưng số đo thật cos(tuyệt_vời, ngon) = **0.00** — vì nó chỉ có đúng 1 ngữ cảnh (một câu cảm thán về mặt trời), vector của nó là **kỷ niệm về câu duy nhất đó** chứ không phải "nghĩa". Đây là lý do Word2Vec có tham số `min_count` để loại từ hiếm — thà không có vector còn hơn vector rác.

### 🟡 Nhóm F (từ 25–30): Từ chức năng — đứng cạnh mọi thứ nên chẳng nghĩa là gì

| # | Từ | SL | Hàng xóm gần nhất | Nhận xét |
|---|:---|---:|:---|:---|
| 25 | này | 26 | dở (0.99), ngon (0.98) | Từ phổ biến nhất kho — dính với cả khen lẫn chê |
| 26 | quá | 10 | hát (0.99), nhạc (0.99) | — |
| 27 | lắm | 8 | món (1.00), ghê (0.99) | — |
| 28 | thiệt | 6 | ghê (1.00), bánh (1.00) | — |
| 29 | nhưng | 3 | rẻ (1.00), quán (0.99) | — |
| 30 | vào | 2 | trực_tiếp (1.00), đá (1.00) | — |

"Này" xuất hiện 26 lần, cạnh đủ loại từ → vector của nó là món "lẩu thập cẩm" không mang nghĩa riêng (cos(này, ngon) = 0.98 mà cos(này, dở) cũng = 0.99). NLP cổ điển gọi đây là **stopwords** và thường lọc bỏ trước khi phân tích; PPMI cũng đã tự giảm sức nặng của chúng một phần.

**➕ Từ thứ 31 không có ghế — OOV (Out-of-Vocabulary):** từ "xuất_sắc" không xuất hiện trong kho → **không có vector nào cả**. Hệ thống gặp từ mới là câm. Các đời sau vá bằng subword (FastText: ghép vector từ các mảnh chữ) và tokenizer BPE của Transformer.

---

## 4. Các Config Ảnh Hưởng Cái Gì? (số chạy thật)

### 4a. Số chiều embedding `d`

| d | cos(ngon, tuyệt) | cos(hay, cuốn) | Chẩn đoán |
| :---: | :---: | :---: | :--- |
| 2 | 1.00 | 1.00 | Chật quá — MỌI cặp từ đều ≈ 1.00, cả "ngon" với "dở" lẫn "ngon" với "này". Không gian 2 chiều không đủ chỗ để tách 57 từ |
| **6** | **1.00** | **0.99** | Vừa cho kho 38 câu — cụm tách cụm, đồng nghĩa vẫn dính nhau |
| 20 | 0.73 | 0.81 | Loãng — kho bé không đủ tín hiệu lấp 20 chiều, các chiều thừa chứa nhiễu làm mọi cosine tụt xuống |

Kinh nghiệm: kho càng lớn càng gánh được nhiều chiều (Word2Vec chuẩn dùng 100–300 chiều cho kho tỷ từ). Kho bé + d to = nhiễu.

### 4b. Cửa sổ ngữ cảnh `window`

Lý thuyết: window nhỏ (±1, ±2) → embedding thiên về **cú pháp/loại từ** (từ thay thế được cho nhau); window lớn (±5, ±10) → thiên về **chủ đề** (từ cùng đề tài). Số đo trên kho này: window 1 và 5 cho kết quả gần như nhau (cos(ngon, tuyệt) 1.00 vs 0.99) — **kho 38 câu quá ngắn và rập khuôn để thấy khác biệt**; hiệu ứng này cần kho cỡ Wikipedia mới lộ rõ. (Ghi chú thẳng thắn: usecase này cố tình dùng kho tí hon để bạn soi được từng con số — đổi lại có những hiện tượng chỉ hiện ra ở quy mô lớn.)

### 4c. `min_count` — ngưỡng loại từ hiếm

Nhóm E chính là lời giải thích: từ xuất hiện 1–2 lần cho vector rác. Đặt `min_count=2` là "tuyệt_vời, ông, mặt_trời" bị loại — mất từ nhưng sạch không gian. Đánh đổi kinh điển giữa **độ phủ** và **độ tin cậy**.

### 4d. Nấc thang tiến hóa — chọn công nghệ nào cho bài toán nào

| Công nghệ | Biết gì | Điểm mù | Dùng khi |
| :--- | :--- | :--- | :--- |
| One-hot / BoW | Từ nào xuất hiện | Mọi cặp từ đều xa lạ như nhau | Bài toán từ khóa đơn giản, cần nhanh-rẻ |
| **Word Embedding (bài này)** | Từ nào **gần nghĩa** từ nào | Trái nghĩa dính nhau (nhóm B), đa nghĩa (D), từ hiếm (E), OOV | Tìm kiếm ngữ nghĩa, gom cụm chủ đề, làm đầu vào cho mô hình khác |
| Contextual (BERT/Transformer) | Nghĩa của từ **trong câu cụ thể** | Nặng, cần GPU, cần pretrain | Sentiment thật, phân tích câu phức tạp — xem Chương 6 |

---

## 5. Checklist: Dùng Embedding Sao Cho Khỏi Sập Bẫy

- [ ] Cần "gần nghĩa" (search, gợi ý, gom cụm)? → embedding thắng one-hot tuyệt đối.
- [ ] Làm sentiment? → **đừng** dùng khoảng cách embedding thô — nhớ cos(ngon, dở) = 0.99 (nhóm B).
- [ ] Ngôn ngữ có từ đa nghĩa nặng (tiếng Việt: "đá", "bạc", "đường")? → cân nhắc contextual embedding (nhóm D).
- [ ] Kho văn bản của bạn bé? → giảm số chiều `d`, tăng `min_count`, và tốt nhất: dùng embedding **pretrain sẵn** (PhoW2V, fastText tiếng Việt) thay vì tự train.
- [ ] Sản phẩm sẽ gặp từ mới liên tục (tên món, tiếng lóng)? → chọn giải pháp subword (FastText/BPE) để không câm trước OOV.

---

## 6. Code Chạy Lại Toàn Bộ (numpy thuần, copy vào notebook là chạy)

```python
import numpy as np

corpus = """mon nay ngon qua
mon nay tuyet qua
com me nau ngon lam
com me nau tuyet lam
pho quan nay ngon thiet
pho quan nay tuyet thiet
banh nay ngon ghe
banh nay hap_dan ghe
mon nay hap_dan qua
phim nay hay qua
phim nay cuon qua
truyen nay hay lam
truyen nay cuon lam
nhac nay hay thiet
bai hat nay cuon thiet
mon nay do qua
mon nay te qua
com quan do lam
com quan te lam
pho nay do thiet
banh nay te ghe
phim nay chan qua
phim nay nhat qua
truyen nay chan lam
truyen nay nhat lam
nhac nay chan thiet
cho them da vao ly tra
ly tra nay nhieu da qua
uong tra it da thoi
cau_thu da bong vao luoi
cau_thu da phat truc_tiep
tran bong co cu da dep
mon nay ngon nhung dat
phim nay hay nhung dai
com nay do nhung re
mua ban o cho som
me mua rau o cho
tuyet_voi ong mat_troi"""
sents = [s.split() for s in corpus.strip().split("\n")]
vocab = sorted({t for s in sents for t in s})
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# 1. Ma trận đồng xuất hiện (cửa sổ ±2)
C = np.zeros((V, V))
for s in sents:
    for i, w in enumerate(s):
        for j in range(max(0, i-2), min(len(s), i+3)):
            if j != i: C[w2i[w], w2i[s[j]]] += 1

# 2. PPMI  3. SVD xuống 6 chiều
tong = C.sum(); hang = C.sum(1, keepdims=True); cot = C.sum(0, keepdims=True)
with np.errstate(divide='ignore', invalid='ignore'):
    pmi = np.log(C * tong / (hang * cot))
ppmi = np.nan_to_num(np.maximum(pmi, 0))
U, S, _ = np.linalg.svd(ppmi)
E = U[:, :6] * S[:6]                      # mỗi từ = vector 6 số

# 4. Cosine similarity (đúng công thức README chương 5)
def cos(a, b):
    u, v = E[w2i[a]], E[w2i[b]]
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return 0.0 if n == 0 else u @ v / n

print("Đồng nghĩa :", f"cos(ngon, tuyet) = {cos('ngon','tuyet'):.2f}")
print("Cạm bẫy    :", f"cos(ngon, do)    = {cos('ngon','do'):.2f}  <- trái nghĩa mà!")
print("Khác chủ đề:", f"cos(ngon, bong)  = {cos('ngon','bong'):.2f}")
print("Đa nghĩa   :", f"cos(da, tra) = {cos('da','tra'):.2f}, cos(da, bong) = {cos('da','bong'):.2f}")

# Bài tập tự nghịch:
# 1. Đổi số chiều 6 -> 2 và 20, đo lại 4 dòng trên (mục 4a)
# 2. Thêm 5 câu chứa "tuyet_voi" trong ngữ cảnh khen món ăn
#    -> cos(tuyet_voi, ngon) có thoát khỏi 0.00 không? (nhóm E)
# 3. Vẽ 57 từ lên mặt phẳng bằng E[:, :2] + matplotlib để THẤY các cụm chủ đề
```

---

## 7. Kết Nối

| Bạn muốn | Mở |
| :--- | :--- |
| Công thức Cosine, Word2Vec, RNN/LSTM | `README.md` của chương này |
| Bài tập code | `05-Bai_Tap_Thuc_Hanh_NLP_Embeddings.ipynb` |
| Xem contextual embedding vá nhóm B & D thế nào | `../06-Transformers_Generative_AI/` + `../../Khang_lession/Transformer_SelfAttention/USECASE.md` |

> 🌟 **Câu chốt:** Embedding là bước máy tính lần đầu có "trực giác" về nghĩa — nhưng trực giác đó học từ ngữ cảnh, nên nó mang đúng điểm mù của ngữ cảnh: trái nghĩa dính nhau, đa nghĩa kẹt giữa, từ hiếm thành rác. Biết điểm mù ở đâu quan trọng không kém biết nó mạnh chỗ nào.
