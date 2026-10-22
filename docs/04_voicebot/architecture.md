# Voicebot architecture

Luồng mục tiêu: Voice → VAD → ASR → Intent/Entity → RAG/Tools → LLM → TTS → Voice. API, business logic, model và tools phải có ranh giới thay thế được.

## Lát cắt văn bản hiện tại

`src/vietnamese_travel_voicebot/voicebot/pipeline.py` hiện chạy câu hỏi → intent/entity → retrieval từ Markdown local → trả lời bằng đúng đoạn chứng cứ và metadata nguồn. Khi không có kết quả đủ liên quan, pipeline trả thông báo thiếu dữ liệu. Chạy bằng `python scripts/ask_travel.py "..."`.

Knowledge Markdown cần metadata `title`, `source_url`, `updated_at`, `license`; mỗi chunk giữ các trường này qua retrieval. Retrieval hiện dùng lexical token overlap có tham số trong `configs/rag.yaml`, chưa dùng embedding, vector store hoặc LLM. Intent/entity dùng luật và danh sách điểm đến hữu hạn, nên đây là baseline kiểm tra luồng, chưa phải dialogue system.

## Bước tiếp theo

Trước khi nối audio, mở rộng bộ kiểm tra truy vấn du lịch và luồng text cho câu hỏi → intent/entity → RAG cho kiến thức có nguồn hoặc tool cho dữ liệu thời gian thực → LLM → câu trả lời kèm nguồn. Sau đó nối VAD/ASR phía vào và TTS phía ra; giữ transcript để người dùng kiểm tra nhận dạng. Ghi trạng thái, thời gian và lỗi của từng lớp để truy ra bước gây trễ hoặc trả lời sai. Cách chạy các model trên một RTX 3050 6 GB được quyết định từ phép đo VRAM/latency.
