# Day 07 — Nhận xét thực nghiệm trên dev

**Nguồn:** Số liệu từ ba lần chạy thật trên máy học viên, ngày 16/11/2026; tổng hợp trong `dev_retrieval_summary.json`. Đây là log *sau* các phép đo, không phải nhật ký giả thuyết được đóng dấu trước thí nghiệm.

| Cấu hình | Giả thuyết thử | Kết quả dev | Diễn giải |
|---|---|---|---|
| B0: 800 ký tự, không filter | Baseline chung | Dense 10/12/10; TF-IDF 12/14/12; Hybrid 11/14/11 | Định dạng số là Hit@1 / Evidence Hit@3 / Context Evidence trên 15 câu. |
| I1: 1.600 ký tự, không filter | Context rộng hơn giữ được evidence | Dense 10/12/10; TF-IDF 12/14/14; Hybrid 11/14/14 | Có cải thiện context cho hai retriever. Không thay đổi ranking. |
| I2: 1.600 ký tự, filter khi query có section | Giảm nhiễu các câu hỏi có scope rõ | Dense 11/12/11; TF-IDF 12/14/14; Hybrid 11/14/14 | Dense tăng Hit@1/Context 1 câu; không cải thiện chỉ số chung của hai phương pháp còn lại. |

**Không được suy ra** Hybrid tốt hơn TF-IDF, hoặc việc tăng budget sẽ tăng accuracy end-to-end. Các số này là kiểm tra tự động, cần audit thủ công.

## Hai trường hợp cần minh chứng trong báo cáo

- **q15, DHCP DORA:** Với chunking B hiện có, gold quote nằm qua hai chunks. Trong TF-IDF @1600, không có single-chunk hit và context evidence tự động cũng không đạt. Cần đọc full trace trước khi kết luận lỗi là do chunking, retrieval hay pack context.
- **q05, Access Point:** TF-IDF @1600 không có Hit@1 nhưng có Hit@3 và context hit. Giải thích vì sao top-1 chưa chứa đủ bằng chứng, dù kết quả top-3 đủ để trả lời; dùng trace thật.

## Quyết định đề xuất (sau dev retrieval, trước agent/test)

- **Baseline agent:** Dense / top-3 / context 800 / không query filter.
- **Ứng viên chính:** TF-IDF / top-3 / context 1600 / không query filter.
- **RRF Hybrid:** giữ làm so sánh phương pháp thứ ba; chưa có lợi thế so với TF-IDF trên 15 dev.
- **Đánh giá agent thật:** cần chạy ít nhất baseline và ứng viên trên dev để kiểm tra việc lựa chọn có còn hợp lý khi thêm answer/citation/abstention; chỉ sau đó mới chốt bằng `freeze_day07.py` và chạy test mới.

**Phương pháp/giới hạn:** Dev q01–q15 kế thừa Day06, test cũ q16–q20 chỉ là bộ diagnostic (không dùng làm test mới). Các trường hợp unanswerable phải được người thực hiện kiểm tra toàn bộ 40 trang. Không xem đáp án hoặc kết quả test mới để tối ưu mô hình. Không có kết quả agent thật ở thời điểm tạo file này.
