# Knowledge base

Lưu tài liệu du lịch Markdown có nguồn, ngày cập nhật, chủ đề và quyền sử dụng. Dữ kiện realtime hoặc giao dịch đi qua tools/API.

Mỗi file `.md` cần YAML-style front matter với đủ `title`, `source_url`, `updated_at`, `license`; `category` là tùy chọn, mặc định lấy tên thư mục. Nội dung đặt sau front matter:

```markdown
---
title: Tên tài liệu
source_url: https://example.org/source
updated_at: 2026-10-22
license: CC BY 4.0
category: destinations
---

Thông tin được nguồn dẫn xác nhận.
```

Lát cắt đầu chỉ truy xuất từ file local, không tự tải URL. Mọi trả lời đều trích URL và metadata nguồn; khi không có đoạn phù hợp, hệ thống nói chưa có dữ liệu. `destinations/hue.md` là ví dụ ngắn dựa trên [UNESCO](https://whc.unesco.org/en/list/678/); điều kiện tái sử dụng câu diễn đạt cần được xác nhận trước khi phát hành.

Chạy hỏi đáp văn bản từ thư mục gốc:

```bash
python scripts/ask_travel.py "Quần thể di tích Huế được ghi danh năm nào?"
```

Lưu ý: retrieval hiện dùng trùng lặp từ khóa, chưa dùng embedding hoặc LLM; intent/entity là luật đơn giản và chưa bao phủ mọi cách nói.
