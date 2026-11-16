# Báo cáo Day 7

## 1. Dữ liệu và benchmark
### Corpus, nguồn và pipeline từ Day 06
Sử dụng tài liệu **Basics Networking** của Sobur Setiaman, truy cập công khai theo URL trong `data/manifest.json` (Academia.edu). PDF có 65 trang; phạm vi được chỉ mục là **40 trang vật lý 7–46**, tương ứng 9 module, khoảng 31.779 ký tự chuẩn hóa. Quyền phân phối lại toàn văn chưa được xác minh, vì vậy PDF và bản full trace chứa văn bản nguồn được giữ local, không đẩy công khai.

Tái sử dụng PostgreSQL 16 + pgvector (`public.day06_chunks`) và embedding `sentence-transformers/all-MiniLM-L6-v2`, vector 384 chiều, tìm cosine. Day 06 có A = 83 fixed-size chunks và B = **84 logical chunks**. Day 07 sử dụng cùng 84 chunks B cho tất cả phép so sánh; không tạo lại embedding tài liệu. `starter/bridge_day6.py` truy vấn database local, `starter/data/converted/chunks.json` chứa dữ liệu trung gian phục vụ TF-IDF.

### Benchmark và nguyên tắc chia dữ liệu
`benchmark/questions.json`: **24 câu**, trong đó dev **15 answerable + 2 unanswerable**; test **5 answerable mới + 2 unanswerable**. Hai mươi câu answerable đã được kiểm tra gold theo nguồn/trang/chunks. Kết quả validator: `84 chunks B`, `dev = 15+2`, `test = 5+2` và `20/20 gold evidence`; đây là kiểm tra dữ liệu, không phải chấm chất lượng trả lời Agent. Năm câu test q16–q20 của Day 06 không được coi là held-out test Day 07 vì đã được xem trước, chỉ dùng làm diagnostic.

Hai câu unanswerable DEV hỏi số đo packet loss live Wi-Fi và mật khẩu WPA3 cụ thể; hai câu unanswerable TEST hỏi kết quả đo TCP retransmission qua Wireshark và SNMP community string của production switch. **Đã có đợt audit trực tiếp PDF vật lý trang 7–46**: kiểm tra toàn bộ text và hình ảnh 40 trang, đối chiếu các trang 12, 27–28, 41–46 và trang có nội dung switch. Không tìm thấy bốn **giá trị đo/cấu hình cụ thể** được hỏi; các đoạn nói về packet loss, Wireshark, WPA3, retransmission chỉ mang tính khái niệm. Vì vậy dữ liệu nguồn hỗ trợ `answerable=false` cho cả bốn câu.

Kiểm tra độc lập trên PDF gốc cho thấy **20/20 gold evidence answerable khớp trực tiếp trang vật lý được khai báo**, đồng thời SHA256 `questions.json` khớp mã benchmark đã đóng băng trước TEST. Kết quả kiểm tra máy đọc được lưu ở `artifacts/benchmark_pdf_crosscheck.json`.

---

## 2. Retrieval và thử nghiệm
### Bảng tổng hợp kết quả thực nghiệm (Dev & Test)

*Ghi chú:* Retrieval-only chấm 15 câu answerable DEV và 5 câu answerable TEST (loại hai câu unanswerable ở mỗi split). Mỗi câu dùng truy vấn gốc một lần, không gọi Gemini; chi phí **API LLM = 0 USD**, không tính CPU, điện và hạ tầng local. Độ trễ retrieval tính bằng `median_ms` theo `summary.json`, không bao gồm khởi tạo model.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** | Dense (MiniLM + pgvector cosine, chiến lược B), top-k=3, candidate-k=12, filter: Off, rerank: Không, budget: 800 ký tự | Thiết lập mốc cơ sở ban đầu dựa trên độ tương đồng ngữ nghĩa vector trên chiến lược chunking B | dev | 10/15 (66,7%) | 12/15 (80,0%) | 10/15 (66,7%) | **15,257 ms** / $0 (0 LLM calls) |
| **Phương pháp 2** | TF-IDF (trên 84 chunks B), top-k=3, candidate-k=12, filter: Off, rerank: Không, budget: 800 ký tự | Khớp từ vựng trực tiếp giúp bắt chính xác thuật ngữ kỹ thuật chuyên ngành tốt hơn embedding tổng quát | dev | 12/15 (80,0%) | 14/15 (93,3%) | 12/15 (80,0%) | **0,295 ms** / $0 (0 LLM calls) |
| **Phương pháp 3** | Hybrid RRF (Dense + TF-IDF, k=60), top-k=3, candidate-k=12, filter: Off, fusion: RRF; không có reranker riêng, budget: 800 ký tự | Kết hợp thứ hạng ngữ nghĩa (Dense) và từ khóa (TF-IDF) qua Reciprocal Rank Fusion để tận dụng ưu thế cả hai | dev | 11/15 (73,3%) | 14/15 (93,3%) | 11/15 (73,3%) | **15,469 ms** / $0 (0 LLM calls) |
| **Cải thiện 1 (I1)** | TF-IDF (đối chiếu cùng Dense & Hybrid), top-k=3, candidate-k=12, filter: Off, rerank: Không, budget: **1.600 ký tự** | Mở rộng context budget từ 800 lên 1.600 ký tự giúp các chunk bằng chứng ở rank 2–3 không bị cắt bỏ khi đóng gói ngữ cảnh | dev | 12/15 (80,0%) | 14/15 (93,3%) | **14/15 (93,3%)** | **0,430 ms** / $0 (0 LLM calls) |
| **Cải thiện 2 (I2)** | Dense (đối chiếu cùng TF-IDF & Hybrid), top-k=3, candidate-k=12, **filter: On (section)**, rerank: Không, budget: 1.600 ký tự | Bật lọc metadata theo phân mục (section) cho các câu hỏi có chỉ định mục nhằm thu hẹp không gian tìm kiếm và giảm nhiễu | dev | 11/15 (73,3%) | 12/15 (80,0%) | 11/15 (73,3%) | **12,687 ms** / $0 (0 LLM calls) |
| **Cấu hình đã chọn** | **TF-IDF**, top-k=3, candidate-k=12, filter: Off, rerank: Không, budget: **1.600 ký tự** | Khóa cấu hình tốt nhất sau đánh giá tập dev để kiểm thử độc lập trên tập test | test | 3/5 (60,0%) | 5/5 (100,0%) | 4/5 (80,0%) | **0,237 ms** / $0 (0 LLM calls) |
| **Baseline** | Dense (MiniLM + pgvector cosine, chiến lược B), top-k=3, candidate-k=12, filter: Off, rerank: Không, budget: 800 ký tự | Chạy đối chiếu cấu hình cơ sở trên cùng tập test để so sánh khách quan với cấu hình đã chốt | test | **5/5 (100,0%)** | 5/5 (100,0%) | **5/5 (100,0%)** | **33,304 ms** / $0 (0 LLM calls) |

*Kết quả đối chứng I1 (DEV, budget 1.600, filter Off):* Dense đạt Hit@1/Hit@3/Context lần lượt `10/15`, `12/15`, `10/15`; Hybrid đạt `11/15`, `14/15`, `14/15`. *Kết quả đối chứng I2 (DEV, budget 1.600, filter On):* TF-IDF giữ nguyên `12/15`, `14/15`, `14/15`; Hybrid giữ nguyên `11/15`, `14/15`, `14/15`.


**Đã đối chiếu thời gian retrieval:** Các giá trị `median_ms` trong bảng được điền theo 7 `summary.json` cụ thể máy chạy thí nghiệm. Đối chứng bổ sung trên DEV: Dense/1600 (filter Off) **13,201 ms**; Hybrid/1600 (filter Off) **15,276 ms**; TF-IDF/1600 (filter On) **0,264 ms**; Hybrid/1600 (filter On) **15,421 ms**. Một phép chạy riêng `day06-dev-tfidf = 0,232 ms` không sử dụng thay cho thí nghiệm so sánh Day 07. Các số này là retrieval-only, **không bao gồm thời gian khởi tạo mô hình hay gọi Gemini**.

### Phân tích thử nghiệm và quyết định lựa chọn cấu hình
#### 1. Giải thích cấu hình đồng thời là phương pháp mới và cải thiện
Cấu hình **TF-IDF / 1.600 ký tự** mang tính chất kép: vừa là **phương pháp truy xuất mới** thay thế cho Baseline Dense (chuyển từ tìm kiếm tương đồng vector trên MiniLM sang khớp từ vựng trực tiếp bằng TF-IDF trên 84 chunks của chiến lược B), vừa tích hợp bước **cải thiện I1** (tăng ngưỡng ngân sách ngữ cảnh từ 800 lên 1.600 ký tự). 

Về mặt phương pháp luận thực nghiệm, việc so sánh trực tiếp **Baseline (Dense/800)** với **TF-IDF/1600** làm thay đổi đồng thời hai biến số độc lập (thuật toán retriever và context budget). Do đó, mức tăng hiệu năng tổng hợp trên tập dev không thể quy hoàn toàn cho riêng thuật toán TF-IDF mà phải được bóc tách qua các thí nghiệm trung gian:

* Chuyển đổi thuật toán giữ nguyên budget (`Dense/800` → `TF-IDF/800`) giúp tăng `Hit@1` từ `10/15` lên `12/15` (+13,3 điểm phần trăm) và `Hit@3` từ `12/15` lên `14/15` (+13,3 điểm phần trăm).

* Giữ nguyên thuật toán và tăng budget (`TF-IDF/800` → `TF-IDF/1600`) giữ nguyên thứ hạng tìm kiếm (`Hit@1` và `Hit@3` không đổi) nhưng nâng `Context Evidence Rate` từ `12/15` lên `14/15` (+13,3 điểm phần trăm).

#### 2. Hai nhóm lỗi tiêu biểu trên tập dev, nguyên nhân và hướng thử nghiệm
* **Lỗi 1 — Mất bằng chứng khi đóng gói context vượt ngân sách:**

  * **Biểu hiện:** Ở mức ngân sách 800 ký tự, nhiều truy vấn đã tìm đúng tài liệu chứa bằng chứng trong top-3 (`Evidence Hit@3` đạt `14/15` với TF-IDF/Hybrid và `12/15` với Dense), nhưng bằng chứng không xuất hiện trong ngữ cảnh cuối cùng đưa cho mô hình (`Context Evidence Rate` chỉ đạt `12/15` với TF-IDF, `11/15` với Hybrid và `10/15` với Dense).

  * **Nguyên nhân:** Các chunk chứa bằng chứng nằm ở vị trí rank 2 hoặc rank 3 bị loại bỏ trong quá trình ghép chuỗi do chunk ở rank 1 đã chiếm phần lớn giới hạn 800 ký tự.

  * **Cách thử (Cải thiện I1):** Tăng context budget từ 800 lên 1.600 ký tự. Kết quả giúp `Context Evidence Rate` của TF-IDF tăng từ `12/15 (80,0%)` lên `14/15 (93,3%)` và Hybrid tăng từ `11/15 (73,3%)` lên `14/15 (93,3%)`, chạm mức trần của `Evidence Hit@3` (đánh đổi bằng độ dài prompt đầu vào lớn hơn). Riêng Dense không đổi (`10/15`) vì tăng budget không sửa được thứ hạng các chunk chưa truy xuất vào top-3.

* **Lỗi 2 — Nhiễu thứ hạng chéo phân mục (Cross-Section Semantic Noise):**

  * **Biểu hiện:** Với Dense retrieval, các câu hỏi có đề cập rõ tên phân mục (section) vẫn bị xếp nhầm các đoạn văn bản từ section khác lên vị trí rank 1 (`Hit@1` chỉ đạt `10/15 – 66,7%` ngay cả khi đã tăng budget lên 1.600 ký tự).

  * **Nguyên nhân:** Vector embedding tổng quát (MiniLM) tính toán độ tương đồng ngữ nghĩa trên toàn bộ tài liệu mà không có cơ chế ràng buộc cứng theo cấu trúc phân mục.

  * **Cách thử (Cải thiện I2):** Bật bộ lọc metadata theo section (`query filter: On`) cho các câu hỏi có thông tin phân mục tại mức budget 1.600 ký tự. Thử nghiệm giúp Dense cải thiện `Hit@1` và `Context Evidence Rate` từ `10/15 (66,7%)` lên `11/15 (73,3%)`, trong khi TF-IDF và Hybrid không thay đổi kết quả. Tuy nhiên, số lượng câu hỏi có metadata filter trong tập dev quá ít nên tác động chưa đủ lớn để tổng quát hóa.

#### 3. Lý do chọn cấu hình và phân tích so sánh trên tập test sau khi chốt
* **Lý do chọn cấu hình (trước khi mở tập test):** Cấu hình **TF-IDF / 1.600 ký tự (filter: Off)** được chốt làm phương án chính thức vì đạt hiệu suất cao nhất và cân bằng nhất trên tập dev: dẫn đầu về `Hit@1` với `12/15 (80,0%)`, đồng thời đạt mức cao nhất ở cả `Evidence Hit@3` và `Context Evidence Rate` với `14/15 (93,3%)`. Phương án này vượt trội hoàn toàn so với Baseline Dense/800 (`66,7%` / `80,0%` / `66,7%`), nhỉnh hơn Hybrid RRF/1600 ở độ chính xác top-1 (`80,0%` so với `73,3%`), đồng thời giữ kiến trúc đơn giản, không phụ thuộc vào metadata filter vốn có độ phủ thấp.

* **So sánh trên tập test sau khi đóng băng cấu hình:** Khi chạy đánh giá trên 5 câu hỏi answerable của tập `test` (`starter/compare_day07.py`), thứ hạng giữa hai cấu hình bị đảo chiều so với tập dev:

  * **Baseline (Dense / 800 ký tự):** đạt **5/5 (100,0%)** trên cả ba chỉ số `Hit@1`, `Evidence Hit@3` và `Context Evidence Rate`.

  * **Cấu hình đã chọn (TF-IDF / 1.600 ký tự):** đạt **3/5 (60,0%)** ở `Hit@1`, **5/5 (100,0%)** ở `Evidence Hit@3` và **4/5 (80,0%)** ở `Context Evidence Rate` (thấp hơn Baseline 2/5 câu ở `Hit@1` và 1/5 câu ở `Context Evidence Rate`).

* **Kết luận thực nghiệm:** Bất chấp kết quả kém thuận lợi hơn trên tập test, lựa chọn **TF-IDF / 1.600 ký tự** vẫn được giữ nguyên theo đúng kỷ luật thực nghiệm (cấu hình đã được chốt trước khi mở tập test, tuyệt đối không tinh chỉnh ngược tham số hay loại bỏ kết quả test bất lợi). Bên cạnh đó, quy mô tập test chỉ gồm **5 câu answerable** (mỗi câu tương ứng **20,0 điểm phần trăm**), nên chênh lệch 1–2 câu chưa đủ ý nghĩa thống kê để kết luận khả năng tổng quát hóa vượt trội của Dense. Ngoài ra, cần tách bạch báo cáo chỉ số retrieval-only (đánh giá truy vấn gốc một lần) với chỉ số Agent TEST ở phần sau (vốn bao gồm truy vấn do LLM viết lại, nhiều lượt gọi search và bộ dựng ngữ cảnh cuối khác nhau, điển hình như trường hợp lỗi đa giai đoạn `d7_t02`).

---

## 3. Agent và đánh giá
### 3.1. Kiến trúc điều phối, tool và cơ chế xử lý lỗi
- **Mô hình và điều phối (`Controller`):** Sử dụng **Gemini `gemini-3.5-flash-lite`** qua API với cơ chế **native function calling**. `Controller` tiếp nhận câu hỏi cùng lịch sử hội thoại để quyết định gọi công cụ truy xuất, tiếp tục tìm kiếm bổ sung, hoặc tổng hợp câu trả lời kèm citation.

- **Công cụ truy xuất (`search_day06`):** Thực thi `Retriever` trên corpus B (84 chunks từ 40 trang tài liệu), lấy `top-k = 3` (`candidate-k = 12`) và tắt bộ lọc truy vấn (`use-query-filter = Off`).

- **Số lượt gọi và điều kiện dừng:** Giới hạn tối đa **2 lượt tìm kiếm** (`max-searches = 2`) cho mỗi câu hỏi. Vòng lặp dừng khi mô hình chủ động ngừng gọi tool để viết câu trả lời; nếu mô hình tiếp tục yêu cầu gọi tool sau hai lượt search, hệ thống ghi nhận trạng thái `search_limit` và tự động trả về **abstention** (từ chối trả lời).

- **Cách chọn và đóng gói ngữ cảnh (`Context builder`):** Loại bỏ trùng lặp (deduplicate) theo chunk ID và ghép các chunks vào `final_context` trong giới hạn ngân sách **800 ký tự** (baseline) hoặc **1.600 ký tự** (cấu hình chọn). Khi tìm kiếm nhiều lượt, `Context builder` giữ nguyên chuỗi chunks đã chọn từ lượt trước trong phạm vi ngân sách.

- **Ghi vết và kiểm tra hậu kiểm:** Toàn bộ `events`, `retrieval_call_audit`, `final_context`, câu trả lời, citation, số lượng token và thời gian thực thi được lưu trong local traces; số lỗi từ nhà cung cấp (`provider_error`) bằng **0** trên cả hai tập `dev` và `test`. Hệ thống tự động kiểm tra ID của citation có tồn tại trong `final_context`, còn việc nội dung citation có thực sự hỗ trợ câu trả lời hay không được chấm riêng ở bước review.

---

### 3.2. Bảng tổng hợp kết quả Agent (DEV và TEST)
| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Baseline: Dense/800** (`dev` — 17 câu) | 17/17 | 24 retrieval calls / 41 LLM calls (tối đa 2 search/câu) | `5 / 0 / 10` — TB: **1,33/2** **(15 câu answerable)** | 10/10 | 10/10 | 2/2 | 5/15 |
| **Cấu hình chọn: TF-IDF/1600** (`dev` — 17 câu) | 17/17 | 20 retrieval calls / 37 LLM calls (tối đa 2 search/câu) | `1 / 0 / 14` — TB: **1,87/2** **(15 câu answerable)** | 14/14 | 14/14 | 2/2 | 1/15 |
| **Baseline: Dense/800** (`test` — 7 câu, frozen) | 7/7 | 12 retrieval calls / 19 LLM calls **(tối đa 2 lượt tìm/câu)** | `1 / 0 / 4` — TB: **1,60/2** **(5 câu answerable)** | 4/4 | 4/4 | 2/2 | 1/5 |
| **Cấu hình chọn: TF-IDF/1600** (`test` — 7 câu, frozen) | 7/7 | 10 retrieval calls / 17 LLM calls **(tối đa 2 lượt tìm/câu)** | `1 / 0 / 4` — TB: **1,60/2** **(5 câu answerable)** | 4/4 | 4/4 | 2/2 | 1/5 |

**Chấm Context Quality 0/1/2 trên 5 câu answerable TEST** (độc lập với điểm câu trả lời):

| Cấu hình | Điểm 0 | Điểm 1 | Điểm 2 | Trung bình /2 |
|---|---:|---:|---:|---:|
| Dense/800 | 0 | 0 | 5 | 2,00 |
| TF-IDF/1600 | 1 | 0 | 4 | 1,60 |

Trong trường hợp `d7_t02`, Dense có đủ bằng chứng (Context Quality = 2) nhưng Agent từ chối trả lời (Grounded Answer = 0); TF-IDF không giữ được bằng chứng trong `final_context` (Context Quality = 0).

---

### 3.3. Phân tích các truy vấn tiêu biểu (DEV và TEST)
| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **d7_t01** (`test`) | Answerable **(single-mode fiber)** | Cả Dense/800 và TF-IDF/1600 đều gọi tool và đưa được gold evidence ở **trang 15** vào `final_context`. | Cả hai cấu hình đều sinh câu trả lời kèm citation hợp lệ trỏ tới bằng chứng trong context. | **`grounded_answer = 2`**, **`Context Quality = 2`** (cả hai cấu hình) do gold evidence ở trang 15 nằm đầy đủ trong `final_context`. |
| **d7_t02** (`test`) | Answerable **(địa chỉ IPv6)** | **Dense/800:** query `IPv6 address bits Module 4` lấy được chunk trang 20 chứa `Uses 128-bit addresses` vào `final_context` (513 ký tự), nhưng vẫn xin gọi search lần 3.<br>**TF-IDF/1600:** lượt 1 lấy nhầm chunk IPv4 32-bit (trang 20); lượt 2 tìm `IPv6 128 bit Module 4` thấy chunk 128-bit nhưng bị loại khỏi `final_context` do hết budget giữ chuỗi cũ, sau đó gọi search lần 3. | Cả hai cấu hình đều chạm ngưỡng `search_limit` và tự động **abstain** (không phát sinh câu trả lời hay citation). | **`grounded_answer = 0`**, **`false_abstention`** (cả hai).<br>• **Dense (`Context Quality = 2`):** lỗi chính sách dừng/tổng hợp dù đã đủ bằng chứng.<br>• **TF-IDF (`Context Quality = 0`):** lỗi đóng gói context giữa nhiều lượt search kết hợp chạm `search_limit`. |
| **d7_t03** (`test`) | Answerable **(WPA3 thay PSK)** | Cả Dense/800 và TF-IDF/1600 đều truy xuất được bằng chứng tại **trang 27** (về SAE) vào `final_context`. | Cả hai cấu hình sinh câu trả lời kèm citation hợp lệ. | **`grounded_answer = 2`**, **`Context Quality = 2`** (cả hai cấu hình); đủ bằng chứng trang 27 (SAE) trong `final_context`. |
| **d7_t04** (`test`) | Answerable **(Remote Access VPN)** | Cả Dense/800 và TF-IDF/1600 đều truy xuất được bằng chứng tại **trang 37** (remote access vào mạng riêng) vào `final_context`. | Cả hai cấu hình sinh câu trả lời kèm citation hợp lệ. | **`grounded_answer = 2`**, **`Context Quality = 2`** (cả hai cấu hình); đủ bằng chứng trang 37 trong `final_context`. |
| **d7_t05** (`test`) | Answerable **(network hardening)** | Cả Dense/800 và TF-IDF/1600 đều truy xuất được bằng chứng tại **trang 40** (giảm vulnerability/phòng attack) vào `final_context`. | Cả hai cấu hình sinh câu trả lời kèm citation hợp lệ. | **`grounded_answer = 2`**, **`Context Quality = 2`** (cả hai cấu hình); đủ bằng chứng trang 40 trong `final_context`. |
| **d7_u_test01** (`test`) | Unanswerable **(phần trăm TCP retransmission trong live Wireshark)** | Cả hai cấu hình gọi tool tìm kiếm trên corpus nhưng không tìm thấy số liệu đo phần trăm (PDF chỉ nêu khả năng Wireshark và hiện tượng retransmission). | Cả hai cấu hình đều thực hiện **abstention** (không đưa ra citation bịa đặt). | **`abstention_correct = true`**. |
| **d7_u_test02** (`test`) | Unanswerable **(SNMP community string của production switch)** | Cả hai cấu hình gọi tool tìm kiếm trên corpus nhưng không có thông tin cấu hình production switch hay community string trong 40 trang chỉ mục. | Cả hai cấu hình đều thực hiện **abstention** (không đưa ra citation bịa đặt). | **`abstention_correct = true`**. |
| **q15** (`dev`) | Answerable **(bốn bước DHCP DORA)** | TF-IDF/1600 truy xuất được 3 bước đầu vào `final_context`, nhưng bước thứ 4 (**Acknowledge**) nằm ở chunk phía sau và chạm giới hạn vòng lặp tìm kiếm. | Từ chối trả lời (**abstain**) do thiếu đủ cả 4 bước trong ngữ cảnh cuối. | **`grounded_answer = 0`**, **`false_abstention`** (1/15 trên TF-IDF/1600 dev) do lỗi bằng chứng trải nhiều chunk (**multi-chunk evidence**) kết hợp giới hạn `max-searches = 2`. |
| **q04, q05, q06, q09** (`dev`) | Answerable **(4 câu dev)** | Dense/800 không đưa đủ bằng chứng vào context dẫn đến từ chối nhầm; TF-IDF/1600 truy xuất và đóng gói đủ bằng chứng vào `final_context`. | TF-IDF/1600 sinh câu trả lời có citation hợp lệ thay vì abstain như Dense/800. | Sửa được lỗi **`false_abstention`** trên Dense/800 để đạt **`grounded_answer = 2`** trên TF-IDF/1600. |

---

### 3.4. Trace, quy trình tái lập, thời gian và chi phí
#### 3.4.1. Lưu trữ dữ liệu, trace và nhãn đánh giá
- **Cấu hình đóng băng & dữ liệu nền:** Cấu hình chốt trước khi mở tập TEST được lưu tại `artifacts/frozen_selection.json` với mã băm SHA256 của `benchmark/questions.json` là `79fd71b13eee1c3cd507b6c7ab98732d121d6cbbed73408c97e53d6ce835600a`. Metadata nguồn lưu tại `data/manifest.json`, `data/sections.json` và corpus B lưu tại `starter/data/converted/chunks.json`.

- **Traces & tệp nhãn nội bộ:** Nhật ký chạy Agent thực tế nằm tại `starter/results/gemini-{dev,test}-{dense800,tfidf1600}/traces.json`; các tệp gắn nhãn và rà soát nằm tại `starter/results/gemini-test-*/reviews.json` (nhãn **AI-assisted draft** đối chiếu PDF), `context_audit.csv`, `agent_metrics.json` và `auto_summary.json`.

- **Minh chứng tổng hợp công khai (không chứa toàn văn):** Gồm `artifacts/day07_final_metrics.json`, `artifacts/test_agent_comparison.json` và `artifacts/test_per_question_audit.csv`.

#### 3.4.2. Quy trình lệnh chạy tái lập và tổng hợp chỉ số
**Lệnh chạy và đánh giá (từ thư mục gốc repo):** Cài dependency trong `requirements.txt`, cấu hình `GEMINI_API_KEY` qua biến môi trường và bảo đảm PostgreSQL Day 06 chạy. Không commit `.env` hoặc API key.

```powershell
# Retrieval-only TEST của cấu hình đã chọn
python starter/compare_day07.py --chunks starter/data/converted/chunks.json --questions benchmark/questions.json --split test --enable-test --methods tfidf --budget 1600 --top-k 3 --candidate-k 12 --output starter/results/retrieval-test-tfidf1600
# Agent TEST, sử dụng cấu hình đã đóng băng
python starter/run_day07_agent.py --provider gemini --model gemini-3.5-flash-lite --method tfidf --split test --enable-test --budget 1600 --top-k 3 --candidate-k 12 --max-searches 2 --output starter/results/gemini-test-tfidf1600/traces.json
```

**Trình tự thí nghiệm đã áp dụng:** chạy DEV và chấm DEV → khóa cấu hình → chạy hai cấu hình trên TEST → chấm TEST. Không chạy lại TEST sau khi xem kết quả để thay thế điểm độc lập. Nếu cần tái lập từ đầu, vẫn phải tuân thủ thứ tự đó.

- **Bước 1 — Đóng băng cấu hình trước TEST:** Chạy `python starter/freeze_day07.py` để ghi nhận cấu hình chọn (`TF-IDF/1600`), cấu hình đối chiếu (`Dense/800`) và mã SHA256 của bộ câu hỏi benchmark vào `artifacts/frozen_selection.json` trước khi chạy thực nghiệm trên tập TEST (mã nguồn Agent nằm tại `starter/run_day07_agent.py`, `starter/agentic.py`, `starter/retrieval_compare.py`).

- **Bước 2 — Tính lại chỉ số Agent TEST cho cấu hình Baseline (`Dense/800`):** Script `starter/agent_metrics.py` đọc bộ câu hỏi (`--cases`), tệp dấu vết chạy (`--traces`) và tệp nhãn kiểm duyệt (`--reviews`) của cấu hình `gemini-test-dense800` để tính lại toàn bộ chỉ số đánh giá (không gọi API) và xuất ra `agent_metrics.json`:

```powershell
python starter/agent_metrics.py `
  --cases benchmark/questions.json `
  --traces starter/results/gemini-test-dense800/traces.json `
  --reviews starter/results/gemini-test-dense800/reviews.json `
  --split test `
  --output starter/results/gemini-test-dense800/agent_metrics.json
```

- **Bước 3 — Tính lại chỉ số Agent TEST cho cấu hình đã chọn (`TF-IDF/1600`):** Thực hiện tương tự trên dữ liệu chạy của cấu hình `gemini-test-tfidf1600` để xuất tệp `agent_metrics.json` phục vụ đối chiếu ngang hàng với Baseline:

```powershell
python starter/agent_metrics.py `
  --cases benchmark/questions.json `
  --traces starter/results/gemini-test-tfidf1600/traces.json `
  --reviews starter/results/gemini-test-tfidf1600/reviews.json `
  --split test `
  --output starter/results/gemini-test-tfidf1600/agent_metrics.json
```

- **Bước 4 — Xuất tệp tổng hợp metrics công khai:** Chạy `python starter/export_final_day07_metrics.py` để kiểm tra tính toàn vẹn của benchmark hash, tổng hợp kết quả từ hai tệp `agent_metrics.json` ở trên và xuất ra `artifacts/day07_final_metrics.json` (loại bỏ toàn bộ văn bản gốc và traces nhạy cảm).

#### 3.4.3. Thời gian thực thi, lượng token và chi phí
- **Thời gian thực thi (median end-to-end):**

  - Trên tập **DEV (17 câu):** Dense/800 đạt **9,63 s**; TF-IDF/1600 đạt **5,23 s**.

  - Trên tập **TEST (7 câu):** Dense/800 đạt **19,57 s**; TF-IDF/1600 đạt **3,20 s**. **(Chênh lệch độ trễ qua API trên mẫu nhỏ chịu ảnh hưởng bởi biến động mạng/máy chủ, không khẳng định tốc độ hệ thống tổng quát).**

- **Tiêu thụ token và chi phí API:**

  - Trên tập **DEV:** Dense/800 dùng **23.308** input tokens / **1.807** output tokens; TF-IDF/1600 dùng **26.684** input tokens / **2.056** output tokens.

  - Trên tập **TEST:** Dense/800 dùng **11.255** prompt tokens / **755** completion tokens; TF-IDF/1600 dùng **13.264** prompt tokens / **715** completion tokens.

  - **Chi phí (`service_cost_usd`):** Trong trace ghi `service_cost_usd = null` (chưa xác định chi phí). Tài khoản sử dụng Google AI Studio Free Tier.

---

### 3.5. Kết quả trước/sau, lỗi, đánh đổi và giới hạn
- **So sánh trước/sau (DEV so với TEST):**

  - Trên **DEV**, cấu hình chọn `TF-IDF/1600` vượt trội rõ rệt so với baseline `Dense/800`: `Grounded answer` tăng từ `10/15` (`1,33/2`) lên `14/15` (`1,87/2`), giảm `False abstention` từ `5/15` xuống `1/15` (khắc phục được `q04, q05, q06, q09`) và giảm median latency từ `9,63 s` xuống `5,23 s`.

  - Trên **TEST** (đã khóa cấu hình, không chỉnh sửa hậu kiểm), không có bằng chứng `TF-IDF/1600` thắng về chất lượng trả lời: hai cấu hình **đồng hạng** về `Grounded answer` (`4/5`, TB `1,60/2`), `Citation validity` (`4/4`), `Abstention đúng` (`2/2`) và `False abstention` (`1/5`). Xét riêng `Context Quality` thủ công trên 5 câu answerable TEST, `Dense/800` đạt **5/5** (TB `2,00/2`) cao hơn mức **4/5** (TB `1,60/2`) của `TF-IDF/1600`, trong khi `TF-IDF/1600` tốn ít lượt gọi tool/LLM hơn (`10/17` so với `12/19`) và có median latency thấp hơn (`3,20 s` so với `19,57 s`).

- **Ví dụ lỗi điển hình:**

  - **Lỗi phân mảnh bằng chứng đa chunk (`q15` - DEV):** Cần đủ 4 bước DHCP DORA nhưng context chỉ gom được 3 bước đầu, bước **Acknowledge** nằm ở chunk sau và chạm trần `max-searches = 2`.

  - **Lỗi điều phối dừng (`d7_t02` - TEST, Dense/800):** Bằng chứng `Uses 128-bit addresses` (trang 20) đã nằm trong `final_context` 513 ký tự ngay lượt đầu, nhưng mô hình vẫn yêu cầu tìm lần 3, chạm `search_limit` và abstain.

  - **Lỗi đóng gói context đa lượt (`d7_t02` - TEST, TF-IDF/1600):** Lượt 1 lấy nhầm chunk IPv4 32-bit; lượt 2 tìm thấy chunk IPv6 128-bit nhưng bị loại khỏi `final_context` vì `context builder` giữ nguyên chuỗi chunk cũ trong budget, sau đó chạm `search_limit`.

- **Đánh đổi và giới hạn:**

  - **Đánh đổi:** Ngân sách 1.600 ký tự giúp giảm nguy cơ cắt cụt bằng chứng nhưng làm tăng tổng lượng prompt tokens đầu vào trên cả `dev` và `test`.

  - **Giới hạn:** (1) So sánh `Dense/800` với `TF-IDF/1600` thay đổi đồng thời cả thuật toán retrieval lẫn context budget nên là đánh giá cấp **cấu hình**, không cô lập từng biến; (2) Cơ chế giữ chunk cũ của `context builder` và việc thiếu bước ép tổng hợp câu trả lời cuối (**final synthesis** tắt tool) khi chạm `max-searches = 2` dễ dẫn đến từ chối nhầm; (3) Quy mô corpus nhỏ (1 tài liệu 40 trang, 84 chunks) và tập TEST chỉ có 5 câu answerable (mỗi câu tương ứng 20 điểm phần trăm) nên chưa đủ cơ sở khẳng định khả năng tổng quát hóa; (4) `Citation validity` chỉ tính trên các câu có phát sinh citation (không tính câu abstain)

## 4. Kết luận và phạm vi nộp bài

Cấu hình **TF-IDF/1600** được chọn trên DEV và đóng băng trước khi mở TEST. Trên DEV, cấu hình này cải thiện tỷ lệ câu answerable đạt 2/2 từ 10/15 lên 14/15, nhưng trên TEST cả hai cùng đạt 4/5. Riêng retrieval-only TEST, Dense/800 đạt Hit@1 = 5/5 và Context Evidence Rate = 5/5, cao hơn TF-IDF/1600 (3/5 và 4/5). Không dùng kết quả TEST để sửa lại cấu hình hoặc lựa chọn mô hình.

**Tình trạng đánh giá:** 24/24 unit tests thành công; bộ benchmark có 24 câu (20 answerable, 4 unanswerable). Đã kiểm tra trực tiếp 20 gold excerpts trên PDF (**20/20**) và rà soát độc lập 40 trang nguồn cho bốn câu unanswerable, lưu kết quả tại `artifacts/unanswerable_audit.md` và `artifacts/benchmark_pdf_crosscheck.json`. Các nhãn chấm câu trả lời/citation có AI hỗ trợ đánh giá.