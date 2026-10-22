# ASR fundamentals

ASR chuyển audio tiếng Việt thành transcript; lỗi dấu thanh, tên địa danh, số/ngày giờ và tiếng ồn có thể làm sai intent hoặc lịch trình phía sau. Baseline đã chọn PhoWhisper-small và Whisper-small làm đối chứng trong [ADR 001](../decisions/001_model_selection.md). Chất lượng sẽ được đo bằng [CER/WER vi-v1](../05_evaluation/asr_metrics.md), không suy ra từ kết quả công bố của model khác protocol.
