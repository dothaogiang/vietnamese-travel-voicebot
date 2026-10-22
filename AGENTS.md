# Repository Guidelines

## Mục tiêu và phạm vi

Xây dựng **Vietnamese Travel Planning & Destination Voicebot** để người Việt hỏi thông tin điểm đến, phương tiện, hoạt động và lập lịch trình bằng tiếng Việt qua giọng nói. Trọng tâm nghiên cứu là Vietnamese ASR và travel-domain adaptation từ pretrained model; trọng tâm kỹ thuật là hệ thống ASR, LLM, RAG, tools, TTS và API có thể thay thế từng thành phần. Không huấn luyện foundation model từ đầu, và không chuyển sản phẩm thành chatbot tổng quát hoặc chỉ là demo ASR.

## Kiến trúc và cấu trúc

Luồng mặc định: **Voice → VAD → ASR → Intent/Entity → RAG/Tools → LLM → TTS → Voice**. Có thể thay cách triển khai một thành phần, nhưng ghi lại lý do nếu thay đổi kiến trúc hoặc loại bỏ layer. LLM dùng knowledge hoặc tools cho dữ kiện thực tế; không bịa thông tin có thể truy xuất. Dùng RAG cho kiến thức du lịch, tools/API cho dữ liệu realtime hoặc transaction. Tách API/business logic khỏi model và tools.

Repository đã có scaffold: `configs/` cho cấu hình, `src/vietnamese_travel_voicebot/` cho code package, `api/` cho HTTP, `scripts/` cho tác vụ, `tests/` cho kiểm thử và `experiments/` cho hồ sơ thí nghiệm. Các adapter ASR, API và pipeline voicebot vẫn là placeholder. Notebook dành cho exploration và phân tích, không chứa production logic. Chạy starter bằng `python3 main.py`; xem `docs/implementation_plan.md` để biết ranh giới từng giai đoạn.

## Nguyên tắc dữ liệu, model và thí nghiệm

- Ưu tiên pretrained model, baseline nhỏ, fine-tuning/domain adaptation có kiểm soát; cân nhắc PEFT/LoRA và mixed precision khi phù hợp.
- Trước khi dùng dataset, ghi nhận tên, nguồn, license, mục đích, kích thước, định dạng audio/transcript, phân chia và giới hạn. Kiểm tra quyền sử dụng, chất lượng audio, transcript và speaker; dữ liệu công khai không mặc nhiên được phép cho mọi mục đích.
- Với GPU/storage giới hạn, bắt đầu bằng subset nhỏ, kiểm định dữ liệu, chạy baseline rồi mới mở rộng. Không tải dataset/model lớn, tạo checkpoint hoặc nhân bản audio hàng loạt khi chưa cần.
- Đặt tham số thí nghiệm trong cấu hình, không hard-code trong source. Lưu metadata và kết quả từng experiment; giữ cả kết quả không tốt. Không đổi test set, metric, preprocessing hay quy trình đánh giá để làm đẹp kết quả; thay đổi methodology phải tạo phiên bản thí nghiệm mới.
- Không commit dữ liệu/audio hay checkpoint/model lớn (`*.wav`, `*.mp3`, `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`) vào Git. Quản lý qua storage phù hợp; repository giữ code, cấu hình, tài liệu, metadata và kết quả đánh giá.

## Giao diện và kiểm thử

Ưu tiên UI hiện đại, tối giản, thân thiện, tập trung vào voice interaction và lập kế hoạch du lịch. Hiển thị rõ các trạng thái `Idle`, `Listening`, `Processing`, `Thinking`, `Speaking`, `Error`; cho phép xem transcript để kiểm tra ASR. Chỉ thêm UI phục vụ trực tiếp use case.

Đã có `unittest` cho normalization và CER/WER: `.venv/bin/python -m unittest discover -s tests -v` sau khi cài editable theo README. Khi thêm hoặc đổi hành vi, viết kiểm thử tập trung, ghi lại lệnh chạy và kiểm tra các thành phần bị ảnh hưởng. Ưu tiên theo thứ tự: correctness, reproducibility, data/model evaluation, resource efficiency, maintainability, performance, convenience.

## Quy trình thay đổi

Trước thay đổi lớn, đọc tài liệu liên quan, kiểm tra implementation và đánh giá tác động tới experiment. Thay đổi ảnh hưởng kiến trúc, dataset, model, evaluation hoặc deployment phải cập nhật tài liệu tương ứng. Không đổi domain mặc định **Vietnamese Travel Planning & Destination Information** nếu chưa đánh giá lại use case, dữ liệu, knowledge base, evaluation và architecture. Giữ secrets ngoài source control.
