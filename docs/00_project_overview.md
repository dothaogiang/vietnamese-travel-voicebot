# Tổng quan dự án

Vietnamese Travel Planning & Destination Voicebot giúp người Việt hỏi bằng giọng nói về điểm đến, phương tiện, hoạt động và lịch trình. Mục tiêu nghiên cứu là Vietnamese ASR và thích nghi miền du lịch dựa trên pretrained model; mục tiêu sản phẩm là hệ thống có thể thay thế ASR, retrieval, tools, LLM và TTS độc lập.

Luồng mục tiêu: Voice → VAD → ASR → Intent/Entity → RAG/Tools → LLM → TTS → Voice. Giai đoạn 1 đã có package, lựa chọn ASR baseline và protocol CER/WER; pipeline giọng nói vẫn chưa chạy thật. Trạng thái và bước tiếp theo nằm trong [kế hoạch triển khai](implementation_plan.md).
