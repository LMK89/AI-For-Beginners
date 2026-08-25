# 🧭 USECASE: Transformer & Self-Attention Dùng Khi Nào? — Học Qua 30 Câu Văn Thật

> **Dành cho người chưa biết gì:** Hai bài trước (`../Logistic_Regression/USECASE.md`, `../XGBoost_LightGBM/USECASE.md`) làm việc với **dữ liệu bảng**. Bài này đổi sang **văn bản** — nơi Transformer thống trị. Cách học vẫn y nguyên: một bộ 30 câu đánh giá phim/sản phẩm (nhãn: 👍 tích cực / 👎 tiêu cực), xem mô hình kiểu cũ chết ở câu nào, **chứng minh bằng toán** chứ không nói suông, rồi mới hiểu Attention sinh ra để giải quyết đúng cái chỗ chết đó. Mọi con số đều chạy thật — code cuối file.

---

## 1. TL;DR — Tóm tắt 30 giây

| Câu hỏi | Trả lời ngắn gọn |
| :--- | :--- |
| Transformer làm gì? | Xử lý **chuỗi** (văn bản, code, chuỗi thời gian, ảnh cắt patch) bằng cách cho **mỗi từ tự tìm xem nó liên quan đến từ nào khác** trong câu (Self-Attention), bất kể đứng gần hay xa. |
| Khi nào BẮT BUỘC cần nó? | Khi **thứ tự từ và ngữ cảnh đổi nghĩa cả câu**: "đầu chán sau hay" ≠ "đầu hay sau chán", "không hay" ≠ "hay". Mô hình đếm từ kiểu cũ bị toán học kết án là không phân biệt nổi. |
| Khi nào KHÔNG cần? | Dữ liệu bảng (dùng XGBoost — Module 2), bài văn bản dễ ăn theo từ khóa (lọc spam đơn giản — đếm từ là đủ), và **dữ liệu ít** (30 câu không train nổi Transformer từ đầu — phải đi mượn mô hình đã học sẵn). |
| Tham số ảnh hưởng gì? | **Độ dài ngữ cảnh T** quyết định câu dài có bị cắt cụt không — và bộ nhớ phình theo **T²** (số thật: T = 131k tốn 68,7 GB cho MỘT ma trận attention!); **positional encoding** là thứ duy nhất giữ lại thứ tự từ; **num_heads** = số "góc nhìn" quan hệ song song. |

---

## 2. Transformer Dùng Cho Những Dữ Liệu Gì?

### ✅ Sân nhà — mọi thứ có dạng CHUỖI mà thứ tự mang nghĩa

| Bài toán | Vì sao cần Attention |
| :--- | :--- |
| Chatbot, dịch máy, tóm tắt (ChatGPT, Claude) | Nghĩa mỗi từ phụ thuộc từ đứng quanh nó, có khi cách cả nghìn từ |
| Phân loại cảm xúc, phân tích phản hồi khách hàng | "không tệ" là khen, "không tốt" là chê — phải nối được "không" với từ nó bổ nghĩa |
| Code (Copilot, Claude Code) | Biến khai báo dòng 5, dùng ở dòng 500 — quan hệ tầm xa |
| Ảnh (ViT), âm thanh, protein (AlphaFold) | Cắt thành chuỗi patch/token rồi cũng xử như văn bản |

### ❌ Khi nào đừng dùng

| Tình huống | Vì sao | Dùng gì |
| :--- | :--- | :--- |
| Dữ liệu bảng (duyệt vay 2 bài trước) | Các cột không có "thứ tự" hay "ngữ cảnh" — Attention không có gì để chú ý | XGBoost/LightGBM |
| Văn bản dễ, ăn theo từ khóa | "khuyến mãi 100%", "trúng thưởng" → spam. Đếm từ là xong, rẻ hơn nghìn lần | BoW/TF-IDF + Logistic Regression |
| Chỉ có vài chục / vài trăm mẫu | Transformer có hàng triệu tham số — 30 câu chỉ đủ cho nó học vẹt | Fine-tune mô hình pretrain, hoặc mô hình nhỏ |
| Cần chạy trên thiết bị yếu, độ trễ thấp | O(T²) ngốn RAM và thời gian (bảng mục 6a) | Mô hình nhỏ / distill / Linformer |

---

## 3. Bài Toán: Phân Loại Cảm Xúc 30 Câu Review — Và "Đối Thủ" Để So Sánh

Trước Transformer, cách phổ biến để máy "đọc" văn bản là **Bag-of-Words (BoW — túi từ)**: đếm mỗi từ xuất hiện mấy lần, xếp thành vector, rồi đưa cho... chính Logistic Regression của Module 1! Nhược điểm chí mạng: **vứt bỏ hoàn toàn thứ tự từ** — câu bị xé thành túi từ lộn xộn.

Ta sẽ cho BoW + Logistic Regression đọc 30 câu dưới đây, xem nó gãy ở đâu. Đó chính là bản đồ chỉ ra **Attention sinh ra để làm gì**.

---

## 4. Bộ Dữ Liệu 30 Câu — Câu Nào Cần Attention, Câu Nào Không?

Cột **BoW** = mô hình túi từ đoán đúng (✅) hay sai (❌) — kết quả chạy thật.

### 🟢 Nhóm A (câu 1–10): Ăn theo từ khóa — KHÔNG cần Transformer

| # | Câu | Nhãn | BoW | Giải thích |
|---|:---|:---:|:---:|:---|
| 1 | "Phim này hay tuyệt vời" | 👍 | ✅ | Thấy "hay", "tuyệt vời" → khen. Đếm từ là đủ |
| 2 | "Sản phẩm rất tốt và đáng tiền" | 👍 | ✅ | Từ khóa "tốt", "đáng tiền" |
| 3 | "Dịch vụ chu đáo, nhân viên thân thiện" | 👍 | ✅ | — |
| 4 | "Món ăn ngon, sẽ quay lại" | 👍 | ✅ | — |
| 5 | "Giao hàng nhanh, đóng gói đẹp" | 👍 | ✅ | — |
| 6 | "Phim dở quá, phí tiền vé" | 👎 | ✅ | "dở", "phí tiền" |
| 7 | "Sản phẩm tệ, dùng hai ngày đã hỏng" | 👎 | ✅ | — |
| 8 | "Thái độ phục vụ kém" | 👎 | ✅ | — |
| 9 | "Món ăn nhạt nhẽo và nguội ngắt" | 👎 | ✅ | — |
| 10 | "Giao hàng chậm, hộp móp méo" | 👎 | ✅ | — |

**Bài học:** phần lớn dữ liệu thực tế thuộc nhóm này. Nếu toàn bộ bài toán của bạn giống nhóm A — **đừng đốt tiền GPU cho Transformer**, một mô hình đếm từ chạy trong 1 giây là xong.

### 🟡 Nhóm B (câu 11–16): Phủ định — một chữ "không" lật ngược cả câu

| # | Câu | Nhãn | BoW | Giải thích |
|---|:---|:---:|:---:|:---|
| 11 | "Phim này không hay" | 👎 | ✅* | Chứa từ "hay" (tích cực!) nhưng nghĩa là chê |
| 12 | "Chất lượng không tốt như quảng cáo" | 👎 | ✅* | Chứa "tốt" mà là chê |
| 13 | "Diễn xuất không hề thuyết phục" | 👎 | ✅* | — |
| 14 | "Sản phẩm không tệ chút nào" | 👍 | ✅* | Chứa "tệ" (tiêu cực!) mà là khen |
| 15 | "Không hề thất vọng, rất đáng mua" | 👍 | ✅* | Chứa "thất vọng" mà là khen |
| 16 | "Không có gì để chê" | 👍 | ✅* | Chứa "chê" mà là khen |

**Dấu ✅\* nghĩa là gì?** Trên 30 câu đã học, BoW đoán đúng cả 6 — nhưng là **học vẹt**, không phải hiểu phủ định. Bằng chứng chạy thật: đem 4 câu phủ định **mới** ("Món ăn không ngon", "Dịch vụ không thân thiện"...) ra kiểm tra, BoW đoán **sai cả 4/4** — nó thấy "ngon", "thân thiện" là phán 👍 ngay, chữ "không" đứng cạnh như vô hình. Chi tiết ở mục 5.

### 🔴 Nhóm C (câu 17–20): Cùng một túi từ, thứ tự khác — toán học kết án BoW

| # | Câu | Nhãn | BoW | Giải thích |
|---|:---|:---:|:---:|:---|
| 17 | "Đầu phim chán nhưng càng về sau càng hay" | 👍 | ❌ | Cặp 17–18 dùng **chính xác cùng một bộ từ** |
| 18 | "Đầu phim hay nhưng càng về sau càng chán" | 👎 | ✅ | chỉ đổi chỗ "hay" ↔ "chán" |
| 19 | "Tưởng dở mà hóa ra hay" | 👍 | ❌ | Cặp 19–20 cũng vậy |
| 20 | "Tưởng hay mà hóa ra dở" | 👎 | ✅ | — |

**Đây là chỗ hay nhất file:** tui đã in vector túi từ của câu 17 và 18 ra so — **giống nhau đến từng con số** (chạy thật: `giong het = True`). Nghĩa là với BoW, câu 17 và 18 là MỘT — không tồn tại cách nào để nó phân biệt, giống cặp hồ sơ song sinh 29/30 của hai bài trước. Nó bắt buộc phải đoán cả hai câu cùng một nhãn → **chắc chắn sai ít nhất 1 câu mỗi cặp**, dù bạn train kiểu gì. Đây không phải "mô hình yếu" — đây là **thông tin thứ tự đã bị vứt đi từ lúc biểu diễn dữ liệu**. Muốn cứu phải đổi cách biểu diễn: đó là việc của positional encoding + Attention.

### 🟡 Nhóm D (câu 21–24): Câu dài trộn khen chê — phải biết CHÚ Ý vào đâu

| # | Câu | Nhãn | BoW | Giải thích |
|---|:---|:---:|:---:|:---|
| 21 | "Bộ phim tôi chờ suốt ba năm, xếp hàng mua vé từ sáng sớm, rốt cuộc chỉ là nỗi thất vọng" | 👎 | ✅ | 18 từ kể lể, ý quyết định nằm **cuối câu** |
| 22 | "Rạp hơi cũ, ghế hơi cứng, âm thanh thường thôi, nhưng bộ phim thì xứng đáng từng phút" | 👍 | ✅ | 3 ý chê + 1 ý khen — nhưng ý khen mới là chủ đề chính |
| 23 | "Nhân viên dễ thương, quán trang trí đẹp, nhạc hay, mỗi tội đồ ăn thì dở tệ" | 👎 | ✅ | 3 khen + 1 chê — nhưng đi ăn thì đồ ăn là thứ quyết định |
| 24 | "Đặt hàng hơi rắc rối, chờ hơi lâu, nhưng chất lượng món hàng vượt xa mong đợi" | 👍 | ✅ | Tương tự |

**Vì sao vẫn ✅?** Trên 30 câu, BoW gỡ được nhờ đếm tỉ số từ khen/chê và học vẹt. Nhưng để ý bản chất: câu 22 có tới 3 cụm chê và 1 cụm khen mà nhãn là 👍 — quy luật thật là *"ý sau chữ **nhưng**, gắn với **chủ đề chính**, mới quyết định"*. Đếm từ không chứa quy luật đó, chỉ là trùng hợp thắng trên mẫu nhỏ. Attention thì mô hình hóa đúng bản chất: token cuối câu bắn Query tìm về "nhưng", "bộ phim", "xứng đáng" và **phân bổ trọng số chú ý** nghiêng hẳn về vế chính — đây cũng chính là lý do RNN đời cũ chết (đọc tuần tự đến cuối câu 21 thì đã "quên" đầu câu — vanishing gradient, xem README module).

### 🔴 Nhóm E (câu 25–28): Mỉa mai & tri thức đời thường — Transformer nhỏ cũng khóc

| # | Câu | Nhãn | BoW | Giải thích |
|---|:---|:---:|:---:|:---|
| 25 | "Hay lắm, xem mười phút là ngủ luôn" | 👎 | ✅* | "Hay lắm" là **mỉa mai** — nghĩa đen ngược nghĩa thật |
| 26 | "Tuyệt vời, chờ hai tiếng mới được phục vụ" | 👎 | ✅* | Tương tự |
| 27 | "Pin tụt năm mươi phần trăm sau một giờ dùng" | 👎 | ✅* | **Không có từ cảm xúc nào** — phải BIẾT pin tụt nhanh là tệ |
| 28 | "Dùng ba ngày mới phải sạc pin một lần" | 👍 | ✅* | Phải biết 3 ngày 1 lần sạc là trâu |

**Vì sao xếp "khó nhất"?** (✅\* lại là học vẹt trên mẫu nhỏ.) Các câu này đòi **tri thức về thế giới**, không phải kỹ thuật xử lý chuỗi: "ngủ khi xem phim = phim chán", "pin tụt 50%/giờ = tệ". Một Transformer nhỏ train từ đầu trên vài nghìn review không có nguồn nào để biết điều đó. Đây là lý do các mô hình lớn phải **pretrain trên hàng nghìn tỷ từ** — tri thức đời thường thấm vào trọng số từ dữ liệu khổng lồ, rồi mới fine-tune cho việc cụ thể. Bài học chọn công nghệ: gặp nhiều câu nhóm E → đừng tự train, hãy fine-tune mô hình pretrain (PhoBERT cho tiếng Việt, hoặc gọi API LLM).

### ⚫ Nhóm F (câu 29–30): Câu giống hệt nhau, nhãn ngược nhau — bất khả kháng quen thuộc

| # | Câu | Nhãn | BoW | Giải thích |
|---|:---|:---:|:---:|:---|
| 29 | "Ừ thì cũng được đấy" | 👍 | ❌ | Người dễ tính nói → khen thật |
| 30 | "Ừ thì cũng được đấy" | 👎 | ✅ | Người khó tính nói kèm cái nhún vai → chê |

Y hệt nhóm F của hai bài trước: **đầu vào giống hệt nhau thì mọi mô hình — kể cả GPT — bắt buộc cho cùng một đầu ra.** Thông tin phân biệt (giọng điệu, người nói, ngữ cảnh hội thoại) không nằm trong câu chữ. Muốn giải phải đưa thêm ngữ cảnh vào input (các câu trước đó, lịch sử người viết), không phải đổi mô hình. Quy luật này xuyên suốt cả 3 module — hãy khắc cốt ghi tâm.

---

## 5. Kết Quả Chạy Thật: Vá Từng Lỗ Và Thấy Vì Sao Phải Có Attention

| Mô hình | Đúng trên 30 câu | Sai ở câu | Đúng trên 4 câu phủ định MỚI |
| :--- | :---: | :--- | :---: |
| BoW đơn từ (unigram) | 90% | 17, 19, 29 — đúng các câu bị "toán học kết án" | **0/4** ❌ |
| BoW cặp từ liền kề (bigram) | 97% | chỉ còn 29 (bất khả kháng) | **0/4** ❌ |

Đọc bảng này thấy cả một câu chuyện:

1. **Unigram sai đúng những câu lý thuyết bảo phải sai** (mỗi cặp C dính 1 câu, cặp F dính 1 câu) — không phải xui, là tất yếu.
2. **Bigram (đếm thêm cặp từ liền kề: "không_hay", "càng_hay"...) vá được cặp C** → chứng tỏ chỉ cần nhét **một chút thông tin thứ tự** vào biểu diễn là cứu được cả nhóm câu. Đây chính là tinh thần của positional encoding.
3. **Nhưng cả hai đều 0/4 với câu phủ định mới** ("Món ăn không ngon" → cả hai phán 👍!). Vì sao? Với bigram, "không_ngon" là một chiều vector **chưa từng xuất hiện** khi học — nó đã học "không_hay" mà không rút ra được quy luật chung *"KHÔNG + từ khen = chê"*. Mỗi cụm từ là một ô nhớ riêng, **kiến thức không lan sang cụm mới**.

**Và đây là lúc Transformer bước vào.** Nó sửa tận gốc bằng 3 tầng (nối với lý thuyết Q–K–V YouTube trong README module):

- **Embedding:** mỗi từ thành một vector nghĩa — "ngon", "hay", "tốt" nằm **gần nhau** trong không gian. Kiến thức về "hay" tự động lan sang "ngon".
- **Positional encoding:** đóng dấu vị trí vào từng vector — thứ tự từ không còn bị vứt đi (chỗ chết của nhóm C).
- **Self-Attention:** từ "không" bắn **Query** đi hỏi cả câu "tôi đang bổ nghĩa cho ai?", **Key** của "ngon" khớp mạnh nhất → attention weight dồn vào đó → **Value** của "ngon" bị kéo về phía nghĩa đảo ngược. Quy luật phủ định được học **một lần, áp dụng cho mọi từ, ở mọi khoảng cách** — kể cả "không" và từ nó bổ nghĩa cách nhau 20 từ (nhóm D).

---

## 6. Các Config/Tham Số Ảnh Hưởng Cái Gì? Câu Nào Sống, Câu Nào Chết?

### 6a. Độ dài ngữ cảnh T (context length) — và cái giá O(T²)

Mỗi từ phải so khớp với mọi từ khác → ma trận attention T×T. Số liệu thật (float32, **một** head, **một** layer):

| T (số token) | Bộ nhớ ma trận attention | Linformer (k=256) | Giảm |
| ---: | ---: | ---: | ---: |
| 512 | 0,001 GB | 0,0005 GB | 2× |
| 4.096 | 0,067 GB | 0,004 GB | 16× |
| 32.768 | **4,3 GB** | 0,034 GB | 128× |
| 131.072 | **68,7 GB** 💥 | 0,134 GB | 512× |

Mà mô hình thật có nhiều head, nhiều layer: T = 32.768 với 12 layer × 12 head = **618 GB** chỉ riêng các ma trận attention — sập cả cụm GPU. Đây là lý do:
- Mọi mô hình đều phải **chốt cứng T** (context window). Văn bản dài hơn T → **bị cắt cụt**. Hãy nhìn lại câu 21: ý quyết định "nỗi thất vọng" nằm **cuối câu** — nếu T nhỏ và phần đuôi bị cắt, mô hình chỉ thấy "chờ suốt ba năm, xếp hàng mua vé" và phán 👍 sai bét. **Config T quyết định trực tiếp nhóm D sống hay chết.**
- **Linformer** (trọng tâm Tuần 2 của module): nén K, V từ T chiều xuống k = 256 chiều bằng phép chiếu hạng thấp → chi phí tuyến tính. Đánh đổi: attention chỉ còn "bản nén" — với T = 131k nó nén 512 lần, các chi tiết li ti có thể mất. Bài văn cần soi từng chữ (hợp đồng pháp lý) thì cẩn trọng.

### 6b. Positional Encoding — công tắc bật/tắt "thứ tự từ"

Thí nghiệm tưởng tượng được toán bảo chứng: **tắt positional encoding đi**, Self-Attention đối xử mọi vị trí như nhau → cả mô hình trở thành một cỗ máy túi-từ cao cấp → cặp câu 17/18 lại có biểu diễn **giống hệt nhau** → quay về đúng cái chết của nhóm C (mục 4 đã chứng minh với BoW). Một dòng cộng vector nhỏ bé nhưng gánh toàn bộ khả năng phân biệt thứ tự của Transformer. Khi tự code MHA trong notebook của module, hãy thử bỏ dòng cộng positional encoding và xem loss trên các cặp kiểu 17/18 — vĩnh viễn không giảm.

### 6c. `num_heads` — bao nhiêu "góc nhìn" chạy song song?

| Cấu hình | Chuyện gì xảy ra |
| :--- | :--- |
| 1 head | Một bộ Q–K–V duy nhất phải ôm mọi loại quan hệ: phủ định (câu 11–16), liên kết vế sau "nhưng" (câu 22–24), chủ đề chính (câu 23)... các quan hệ giẫm chân nhau |
| 8–12 head | Mỗi head tự chuyên môn hóa: head này chuyên bắt cặp "không → từ được bổ nghĩa", head kia chuyên nối "nhưng" với vế chính. Câu 23 cần **đồng thời** head chủ-đề (đồ ăn là chính) và head cảm xúc (dở tệ) |
| Quá nhiều head | d_model bị chia nhỏ cho từng head (d_k = d_model/heads) → mỗi head "cận thị"; thêm head không thêm phép màu |

### 6d. Các lựa chọn còn lại (nhanh)

| Config | Ý nghĩa đời thường |
| :--- | :--- |
| `d_model` | Độ "đậm đặc" của vector nghĩa mỗi từ. Nhỏ quá → "không tệ chút nào" và "tệ" ép chung một chỗ, mất sắc thái. Nhớ chia $\sqrt{d_k}$ khi tính attention (README giải thích vì sao) |
| Causal mask (GPT) vs 2 chiều (BERT) | GPT chỉ được nhìn từ phía **trước** (để sinh chữ tiếp theo); BERT nhìn cả **hai phía**. Phân loại cảm xúc như bài này → kiểu BERT lợi hơn: từ "hay" ở câu 17 cần thấy cả "càng về sau" đứng TRƯỚC nó |
| Train từ đầu vs fine-tune | 30 câu (hay cả 30 nghìn câu) không đủ train Transformer từ đầu — nhóm E cần tri thức từ pretrain. Thực chiến tiếng Việt: fine-tune PhoBERT, hoặc few-shot với LLM API. **Đây cũng là lời thú nhận của chính file này:** thí nghiệm mục 5 dùng BoW làm "đối chứng" vì train Transformer tử tế trên 30 câu là bất khả — bản thân điều đó là bài học về chọn công cụ theo cỡ dữ liệu |

---

## 7. Checklist: Văn Bản Này Có Cần Transformer Không?

- [ ] Nhãn đoán được bằng **từ khóa đơn lẻ** (nhóm A)? → BoW/TF-IDF + Logistic Regression, xong việc trong 1 giây.
- [ ] Có **phủ định, đảo thứ tự, câu dài trộn khen chê** (nhóm B, C, D)? → cần mô hình hiểu ngữ cảnh: Transformer.
- [ ] Có **mỉa mai / cần tri thức đời thường** (nhóm E)? → không chỉ Transformer, mà phải là Transformer **đã pretrain** (PhoBERT, LLM API).
- [ ] Văn bản dài cỡ nào? → chọn context length T đủ chứa, và nhớ chi phí T² (mục 6a); rất dài → kiến trúc tuyến tính kiểu Linformer.
- [ ] Có cặp mẫu **giống hệt nhau nhãn ngược nhau** (nhóm F)? → đi tìm thêm ngữ cảnh cho input, đổi mô hình vô ích.
- [ ] Dữ liệu gắn nhãn dưới vài nghìn mẫu? → đừng train từ đầu; fine-tune hoặc few-shot.

---

## 8. Code Chạy Lại Toàn Bộ (copy vào notebook là chạy)

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# 30 câu, nhãn 1=tích cực, 0=tiêu cực (đầy đủ 6 nhóm A-F như bảng mục 4)
data = [
    ("Phim này hay tuyệt vời",1),("Sản phẩm rất tốt và đáng tiền",1),
    ("Dịch vụ chu đáo, nhân viên thân thiện",1),("Món ăn ngon, sẽ quay lại",1),
    ("Giao hàng nhanh, đóng gói đẹp",1),("Phim dở quá, phí tiền vé",0),
    ("Sản phẩm tệ, dùng hai ngày đã hỏng",0),("Thái độ phục vụ kém",0),
    ("Món ăn nhạt nhẽo và nguội ngắt",0),("Giao hàng chậm, hộp móp méo",0),
    ("Phim này không hay",0),("Chất lượng không tốt như quảng cáo",0),
    ("Diễn xuất không hề thuyết phục",0),("Sản phẩm không tệ chút nào",1),
    ("Không hề thất vọng, rất đáng mua",1),("Không có gì để chê",1),
    ("Đầu phim chán nhưng càng về sau càng hay",1),
    ("Đầu phim hay nhưng càng về sau càng chán",0),
    ("Tưởng dở mà hóa ra hay",1),("Tưởng hay mà hóa ra dở",0),
    ("Bộ phim tôi chờ suốt ba năm, xếp hàng mua vé từ sáng sớm, rốt cuộc chỉ là nỗi thất vọng",0),
    ("Rạp hơi cũ, ghế hơi cứng, âm thanh thường thôi, nhưng bộ phim thì xứng đáng từng phút",1),
    ("Nhân viên dễ thương, quán trang trí đẹp, nhạc hay, mỗi tội đồ ăn thì dở tệ",0),
    ("Đặt hàng hơi rắc rối, chờ hơi lâu, nhưng chất lượng món hàng vượt xa mong đợi",1),
    ("Hay lắm, xem mười phút là ngủ luôn",0),
    ("Tuyệt vời, chờ hai tiếng mới được phục vụ",0),
    ("Pin tụt năm mươi phần trăm sau một giờ dùng",0),
    ("Dùng ba ngày mới phải sạc pin một lần",1),
    ("Ừ thì cũng được đấy",1),("Ừ thì cũng được đấy",0),
]
texts = [t for t,_ in data]; y = np.array([l for _,l in data])

for name, vec in [("unigram", CountVectorizer()),
                  ("bigram", CountVectorizer(ngram_range=(1,2)))]:
    Xb = vec.fit_transform(texts)
    m = LogisticRegression(max_iter=5000).fit(Xb, y)
    sai = [i+1 for i in range(30) if m.predict(Xb)[i] != y[i]]
    print(f"{name}: đúng {(m.predict(Xb)==y).mean():.0%} — sai câu {sai}")
    # Câu phủ định MỚI: cả hai đều sai sạch!
    probe = ["Món ăn không ngon","Phim không dở chút nào",
             "Dịch vụ không thân thiện","Nhân viên không hề chu đáo"]
    print(f"  probe phủ định mới (nhãn thật 0,1,0,0):", m.predict(vec.transform(probe)))

# Chứng minh cặp 17/18 là MỘT với túi từ
v = CountVectorizer(); Xb = v.fit_transform(texts).toarray()
print("Vector câu 17 == câu 18:", np.array_equal(Xb[16], Xb[17]))

# Cái giá O(T^2): bộ nhớ ma trận attention (float32, 1 head 1 layer)
for T in [512, 4096, 32768, 131072]:
    print(f"T={T}: full={T*T*4/1e9:.3f} GB | Linformer k=256: {T*256*4/1e9:.4f} GB")

# Bài tập tự nghịch:
# 1. Thêm 5 câu phủ định vào tập học -> probe có khá lên không? Vì sao vẫn không bền?
# 2. Thử ngram_range=(1,3). Số chiều vector phình lên bao nhiêu? (lời nguyền tổ hợp)
# 3. Khi làm notebook MHA của module: bỏ positional encoding, cho model học cặp câu
#    17/18 -> quan sát loss không thể giảm về 0.
```

---

## 9. Kết Nối Với Các Tài Liệu Khác

| Bạn muốn | Mở file |
| :--- | :--- |
| Hiểu công thức Q–K–V, Softmax, √d_k, Linformer, ViT | `README.md` (lộ trình 1–2 tuần) |
| Nghịch Attention Map tương tác | `study_guide.html` |
| Tự code Multi-Head Attention bằng PyTorch | `Transformer_SelfAttention_Practice.ipynb` |
| Ôn hai người anh em dữ liệu bảng | `../Logistic_Regression/USECASE.md`, `../XGBoost_LightGBM/USECASE.md` |

> 🌟 **Câu chốt của cả bộ ba USECASE:** Logistic Regression chết vì dữ liệu cong (chữ V), XGBoost cứu; BoW chết vì thứ tự từ bị vứt đi, Attention cứu. Nhưng cả ba bài đều chung một chân lý: **mẫu giống hệt nhau mà nhãn ngược nhau thì không thuật toán nào cứu nổi — lúc đó thứ cần nâng cấp là dữ liệu, không phải mô hình.**
