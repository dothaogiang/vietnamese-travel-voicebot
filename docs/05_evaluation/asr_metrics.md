# ASR evaluation protocol vi-v1

Protocol cố định trong `configs/evaluation.yaml` và hiện thực tại `src/vietnamese_travel_voicebot/data/normalize.py` và `src/vietnamese_travel_voicebot/evaluation/metrics.py`.

## Chuẩn hóa transcript

Áp dụng giống nhau cho reference và hypothesis: Unicode NFC → chữ thường → thay từng ký tự Unicode thuộc nhóm punctuation bằng dấu cách → gộp mọi whitespace thành một dấu cách và bỏ khoảng trắng đầu/cuối. Giữ dấu tiếng Việt, chữ số và các ký hiệu không thuộc punctuation. Không verbalize số. Reference rỗng sau chuẩn hóa là lỗi dữ liệu; hypothesis rỗng được tính là xóa toàn bộ nội dung.

Ví dụ: `"Đà Nẵng, 2026!"` → `"đà nẵng 2026"`. Mọi thay đổi quy tắc phải tạo protocol mới và chạy lại cả hai model.

## Metric

- **CER chính:** tổng Levenshtein character errors / tổng ký tự reference sau chuẩn hóa; tính cả dấu cách.
- **WER phụ:** tổng Levenshtein token errors / tổng token reference, token được tách bằng whitespace. Đây là word-like WER theo khoảng trắng tiếng Việt, không phải đánh giá ranh giới từ ngữ nghĩa.
- Cả hai là corpus micro average, tính edit distance riêng từng câu rồi cộng tử số và mẫu số. Không nối các câu thành một chuỗi; không lấy trung bình tỉ lệ từng câu.
- Ghi kèm `sample_count`, tổng ký tự/token reference, protocol version, model revision, dataset version và split. CER/WER có thể trên 1 nếu có nhiều insertion.

Hiện thực edit distance dùng Python standard library để có thể chạy offline; không phụ thuộc JiWER. VIVOS test bản gốc đã được tải và khóa manifest; kết quả baseline hai model trên CPU nằm trong [hồ sơ baseline](../experiments/baseline.md). Cần tiếp tục nghe kiểm định nhóm lỗi tên riêng, dấu thanh, số, tiếng ồn, giọng vùng miền và omission. Không đổi test split hoặc protocol sau khi xem kết quả; thay đổi methodology phải tạo phiên bản mới.
