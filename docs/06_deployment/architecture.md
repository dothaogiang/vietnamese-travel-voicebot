# Deployment architecture

Chưa chọn phương án deployment. Quyết định cần dựa trên streaming/latency, phần cứng, quyền riêng tư, chi phí và khả năng vận hành; cập nhật tài liệu khi chốt.

Lúc có luồng chạy thật, tối thiểu ghi request ID, thời gian và trạng thái lỗi theo VAD/ASR/RAG-Tools/LLM/TTS. Tách metric aggregate khỏi audio/transcript của người dùng; chỉ lưu nội dung giọng nói theo chính sách đồng ý, thời hạn lưu và xóa được xác định trước. Bắt đầu bằng log/metric cục bộ có cấu trúc; chọn backend quan sát như OpenTelemetry/Loki/Grafana sau khi đo nhu cầu vận hành. [OpenTelemetry](https://opentelemetry.io/docs/concepts/) mô tả ba loại tín hiệu logs, metrics và traces, nhưng không bắt buộc một backend cụ thể.
