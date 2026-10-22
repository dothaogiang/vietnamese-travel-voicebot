# Hướng dẫn dự án

## Mục tiêu và luồng mặc định

Xây dựng **Vietnamese Travel Planning & Destination Information Voicebot**: Voice → VAD → ASR → Intent/Entity → RAG/Tools → LLM → TTS → Voice. Giữ domain du lịch; không huấn luyện foundation model từ đầu. Dữ kiện truy xuất được phải dựa trên knowledge base hoặc tools/API, không để LLM tự bịa.

## Ranh giới kỹ thuật

- Tách API, business logic, model và tools để thay thế từng thành phần.
- Notebook chỉ dành cho exploration; logic production nằm trong `src/` hoặc `api/`.
- Tham số thí nghiệm nằm trong `configs/`; kết quả và metadata được lưu theo từng experiment.
- Thay đổi kiến trúc, dataset, model, evaluation hoặc deployment cần cập nhật tài liệu liên quan.

## Dữ liệu và model

- Trước khi dùng dataset, ghi tên, nguồn, license, mục đích, kích thước, định dạng, phân chia và giới hạn; kiểm tra quyền sử dụng, audio, transcript và speaker.
- Bắt đầu bằng subset nhỏ, kiểm định dữ liệu và baseline; không tải dataset/model lớn hoặc tạo checkpoint hàng loạt khi chưa cần.
- Không commit audio, dữ liệu riêng tư, checkpoint hoặc model lớn. Dùng storage phù hợp và chỉ commit code, cấu hình, tài liệu, metadata và kết quả đánh giá nhẹ.
- Không đổi test set, metric hay preprocessing để làm đẹp kết quả; thay methodology cần tạo phiên bản experiment mới.

## Chất lượng và kiểm thử

Ưu tiên correctness, reproducibility, data/model evaluation và hiệu quả tài nguyên. `unittest` đã chạy cho protocol ASR; khi thêm hành vi, bổ sung kiểm thử tập trung, ghi lệnh chạy và kiểm tra các thành phần bị ảnh hưởng. Lệnh chạy nằm trong README.

## Bảo mật

Không commit `.env`, token, API key hoặc thông tin xác thực. `.env.example` chỉ chứa tên biến và giá trị mẫu không nhạy cảm.
