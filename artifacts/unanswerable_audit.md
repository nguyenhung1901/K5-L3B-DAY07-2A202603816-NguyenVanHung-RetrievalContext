# Unanswerable Benchmark Audit — Day 07


## 1. Phạm vi nguồn và tính toàn vẹn

- PDF: `Basics_Networking.pdf`, 65 trang vật lý; phạm vi index Day 07 là **trang vật lý 7–46 (40 trang)**, bao gồm Modules 1–9.
- SHA256 PDF kiểm tra: `108140d34d891b58afaf2f6138bc4799432e369e5f65ccd191c8430c75cdee59`.
- SHA256 benchmark `questions.json`: `79fd71b13eee1c3cd507b6c7ab98732d121d6cbbed73408c97e53d6ce835600a` — **khớp cấu hình frozen** đã chốt trước TEST.
- Benchmark gồm 24 câu (DEV 15+2; TEST 5+2). **Cả 20 gold evidence answerable đều khớp trực tiếp trang PDF được khai báo (20/20)** sau chuẩn hóa Unicode và khoảng trắng/dấu câu; đây là xác minh gold nguồn, không phải điểm Agent.

## 2. Phương pháp xác minh

1. Trích xuất văn bản từ từng trang vật lý **7–46**, không giới hạn vào các chunk mà Retriever tìm thấy.
2. Tìm các thuật ngữ liên quan, gồm `packet loss`, `Wireshark`, `retransmission`, `WPA3`, `password`, `SNMP`, `community`, và đối chiếu ngữ cảnh xuất hiện.
3. Kiểm tra ảnh kết xuất của **toàn bộ 40 trang** để nhận diện hình, bảng hoặc thông tin không phản ánh đầy đủ trong text extraction; rà soát kỹ trang thuộc phần Wi-Fi, switching và monitoring.
4. Phân biệt tài liệu **có nhắc khái niệm** với **có cung cấp giá trị đo/cấu hình cụ thể mà câu hỏi yêu cầu**.

## 3. Kết quả theo từng truy vấn

| ID | Trang PDF có nội dung gần chủ đề | Nội dung tài liệu hỗ trợ | Kết luận |
|---|---|---|---|
| `d7_u_dev01` | 42 | Trang 42 giải thích ping, packet loss và latency, không công bố tỷ lệ đo của mạng Wi-Fi văn phòng cụ thể. Không thấy bảng/số liệu live-test tương ứng ở các trang còn lại. | **Không đủ bằng chứng** (`answerable=false`) |
| `d7_u_dev02` | 12, 27, 28, 41 | Trang 12 mô tả AP; 27 giải thích WPA3; 28 đề cập sai mật khẩu; 41 khuyến nghị mật khẩu mạnh. Không đưa mật khẩu của một AP cụ thể hay bài lab WPA3. | **Không đủ bằng chứng** (`answerable=false`) |
| `d7_u_test01` | 43, 44, 46 | Trang 43 nói về Wireshark; 44 và 46 nói tới retransmission như hiện tượng/công cụ kiểm tra. Không có kết quả đo tỷ lệ phần trăm từ live experiment. | **Không đủ bằng chứng** (`answerable=false`) |
| `d7_u_test02` | 11, 30, 31, 45 | Có giải thích switch/MAC/VLAN và kiểm tra thiết bị nhưng không có nội dung SNMP, community string, cấu hình production switch hoặc một chuỗi cấu hình cụ thể trong phạm vi chỉ mục. | **Không đủ bằng chứng** (`answerable=false`) |

### Tần suất thuật ngữ trong 40 trang được chỉ mục

| Từ khóa | Các trang PDF vật lý tìm thấy |
|---|---|
| `packet loss` | 42 |
| `wireshark` | 43, 46 |
| `retransmission` | 44, 46 |
| `wpa3` | 12, 27, 28, 41 |
| `password` | 28, 39, 40, 41 |
| `snmp` | Không tìm thấy |
| `community string` | Không tìm thấy |

**Diễn giải:** Tìm thấy các thuật ngữ như packet loss, retransmission hoặc WPA3 **không làm cho câu hỏi answerable** nếu tài liệu thiếu tỷ lệ đo hay mật khẩu cụ thể. Ngược lại, không tìm thấy `SNMP`/`community` trong văn bản là một dấu hiệu hỗ trợ, nhưng kết luận còn dựa trên rà soát trực quan các trang liên quan.

## 4. Quyết định nhãn và giới hạn

Kết quả rà soát nguồn PDF 7–46 **ủng hộ `answerable=false` cho cả bốn câu**. Nếu Agent từ chối ở bốn câu này thì quyết định **phù hợp với phạm vi tài liệu**; tuy nhiên, điểm `abstention_correct` trong file `reviews.json` cần gắn với *trace thực tế* của từng cấu hình, không suy ra chỉ từ benchmark.


## 5. Lưu trữ và tái lập

- Bản tóm tắt kiểm tra máy đọc được: `artifacts/benchmark_pdf_crosscheck.json`.
- Benchmark chính thức không thay đổi: `benchmark/questions.json`.
- Chỉ số Agent DEV/TEST: `starter/results/gemini-*/agent_metrics.json` và bản tổng hợp `artifacts/day07_final_metrics.json`.
- PDF gốc và full traces có nội dung nguồn giữ ở local; việc PDF được công bố trên mạng không tự động xác nhận quyền tái phân phối toàn văn.
