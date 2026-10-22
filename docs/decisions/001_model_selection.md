# ADR 001: Chọn ASR baseline tiếng Việt

- Trạng thái: Chấp nhận cho giai đoạn 1 (2026-10-14).
- Phạm vi: baseline inference cục bộ; chưa quyết định mô hình production hoặc fine-tuning.

## Bối cảnh

Máy mục tiêu có một RTX 3050 6 GB. Bắt đầu bằng mô hình pretrained nhỏ, batch size 1, audio 16 kHz mono. Chưa xác nhận CUDA có thể truy cập từ tiến trình trong sandbox, nên khả năng chạy và latency là giả thuyết cần đo ở giai đoạn 2.

## Quyết định

| Vai trò | Model và revision cố định | Nguồn/license | Lý do |
|---|---|---|---|
| Chính | `vinai/PhoWhisper-small` @ `a86b604c346caf7148c37512eafe783a16420adb` | [Model card](https://huggingface.co/vinai/PhoWhisper-small), BSD-3-Clause | Huấn luyện cho tiếng Việt; 244M tham số theo [repository tác giả](https://github.com/VinAIResearch/PhoWhisper). |
| Đối chứng | `openai/whisper-small` @ `973afd24965f72e36ca33b3055d56a652f456b4d` | [Model card](https://huggingface.co/openai/whisper-small), Apache-2.0 | Cùng cỡ 244M, đa ngôn ngữ, để đo tác dụng của thích nghi tiếng Việt trên cùng protocol. |

Dùng adapter Transformers ở giai đoạn 2; mã tải/inference chưa được viết. Revision, ngôn ngữ, task, sample rate, batch size và dtype dự kiến nằm trong `configs/asr.yaml`. Trước khi chạy, phải kiểm tra VRAM thực tế, tốc độ, phiên bản runtime và tương thích dtype; không tự động chuyển qua CPU rồi ghi kết quả như chạy GPU.

Không chọn upstream [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) cho baseline vì danh sách ngôn ngữ công bố không có tiếng Việt. File adapter `parakeet.py` chỉ còn là placeholder theo cây code ban đầu.

## Hệ quả

Chưa có số CER/WER của dự án. Số công bố của tác giả không thể thay cho đo trên split và normalization đã khóa tại đây. Việc tải trọng số gần 1 GB của từng model được hoãn tới lúc kiểm tra quyền dữ liệu, dung lượng đĩa và CUDA ở giai đoạn 2.
