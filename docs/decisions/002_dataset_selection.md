# ADR 002: Chọn dữ liệu ASR baseline

- Trạng thái: Chấp nhận có điều kiện cho giai đoạn 1 (2026-10-14).
- Mục đích: benchmark nghiên cứu cục bộ, không dùng làm dữ liệu sản phẩm thương mại.

## Quyết định

Chọn **VIVOS bản gốc**, test split do tác giả phát hành, làm benchmark cố định ban đầu. [Bản ghi Zenodo chính thức](https://zenodo.org/records/7068130) mô tả khoảng 15 giờ tiếng nói tiếng Việt đọc, test khoảng 760 câu/45 phút với 19 người nói; archive khoảng 1,5 GB. Ghi metadata chi tiết, checksum archive, phiên bản và danh sách file sau khi tải và kiểm tra thực tế ở giai đoạn 2. Chưa tạo manifest giả hoặc tải dữ liệu trong giai đoạn 1.

License theo bản gốc là **CC BY-NC-SA 4.0** và nguồn ghi rõ mục đích học thuật. Phải kiểm tra lại điều kiện dùng trước khi tải, phân phối, huấn luyện hoặc đưa vào sản phẩm. Nếu mục tiêu triển khai không phù hợp, chỉ dùng VIVOS cho đánh giá được phép và chọn tập đánh giá có quyền sử dụng khác cho sản phẩm.

## Giới hạn và protocol

VIVOS là tiếng đọc trong môi trường tương đối yên tĩnh, không đại diện đầy đủ cho hội thoại du lịch, tên địa danh hiếm, giọng vùng miền hay tiếng ồn ngoài thực địa. Giữ nguyên test split; không chọn mẫu theo kết quả. Nếu cần subset để kiểm tra smoke, tách khỏi kết quả benchmark chính và ghi seed/danh sách ID. Kiểm tra sample rate, kênh, định dạng audio, transcript và speaker sau khi có dữ liệu, không suy đoán từ bản mirror. Travel-domain evaluation set sẽ được thiết kế riêng ở giai đoạn 5.
