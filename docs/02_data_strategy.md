# Chiến lược dữ liệu

Trước khi dùng bất kỳ dataset nào, lập hồ sơ gồm tên, nguồn, license, mục đích, kích thước, sample rate/định dạng, transcript, speaker, phân chia và giới hạn sử dụng. Xác minh quyền sử dụng cho mục đích dự kiến.

Bắt đầu bằng subset nhỏ để kiểm tra audio/transcript và chạy baseline trước khi mở rộng. Giữ split test cố định; không chỉnh preprocessing hoặc metric sau khi xem test để cải thiện con số. Audio, dữ liệu cá nhân và model/checkpoint không được commit vào Git.

Đã chọn VIVOS bản gốc làm benchmark nghiên cứu có điều kiện trong [ADR 002](decisions/002_dataset_selection.md). VIVOS đã tải, kiểm định kỹ thuật và chạy full test cho hai ASR model; license CC BY-NC-SA 4.0 giới hạn mục đích phi thương mại. VIVOS là tiếng đọc sạch, vì vậy không thay thế tập đánh giá hội thoại du lịch thực tế. Đường chuẩn bị [Bud500 test keyword slice](05_evaluation/bud500_protocol.md) đã có, nhưng chưa tải/chạy dữ liệu thật trong agent sandbox và chỉ là đánh giá bổ sung nhiều chủ đề. Xem `data/README.md` để biết quy ước lưu trữ.
