# Kế hoạch dựng code base

## Mục tiêu

Từ starter `main.py`, xây dựng nền tảng có thể phát triển thành voicebot du lịch tiếng Việt theo luồng Voice → VAD → ASR → Intent/Entity → RAG/Tools → LLM → TTS → Voice. Giai đoạn này chỉ dựng cấu trúc, tài liệu và hợp đồng module tối thiểu; chưa tích hợp model, dữ liệu thật, dịch vụ ngoài hoặc deployment chạy production.

## Quyết định nền

- Dùng package `src/vietnamese_travel_voicebot/` theo miền chức năng; API gọi vào business logic thay vì chứa logic model.
- Giữ entry point `python3 main.py` chạy được trong môi trường Python tiêu chuẩn.
- Chưa thêm framework/dependency cho đến khi chọn implementation có căn cứ.
- Cấu hình tách theo concern trong YAML; mỗi experiment lưu config, metadata, metric và giới hạn.
- Dataset/audio/checkpoint nằm ngoài Git. Chưa tạo split manifests rỗng để tránh nhầm là dữ liệu đã được chuẩn bị.
- RAG phục vụ tri thức du lịch có nguồn; tools/API phục vụ dữ liệu realtime hoặc giao dịch.

## Các giai đoạn

### Giai đoạn 0 — Scaffold (hoàn thành)

- [x] Tạo cấu trúc thư mục và module Python khung.
- [x] Tạo README, hướng dẫn dự án, cấu hình mẫu và tài liệu kiến trúc ban đầu.
- [x] Thêm quy tắc loại trừ secrets, dữ liệu audio và model khỏi Git.
- [x] Xác nhận hướng tổ chức và các giả định stack trước khi triển khai chức năng; ghi trong ADR 003.

**Kiểm tra:** `python3 main.py`; kiểm tra các file cấu hình và cây thư mục. Giai đoạn 1 đã bổ sung `unittest`.

### Giai đoạn 1 — Đặc tả và chọn baseline ASR (hoàn thành về code/tài liệu)

- [x] Ghi máy mục tiêu RTX 3050 6 GB, tài nguyên cục bộ và mục tiêu RTF ≤ 1 cần đo; user xác minh trên host ngày 2026-10-16: CUDA hoạt động, 1 GPU, 5.66 GiB VRAM. Tool sandbox vẫn không thấy GPU.
- [x] Chọn PhoWhisper-small và Whisper-small đối chứng, khóa revision; loại Parakeet v3 khỏi baseline tiếng Việt.
- [x] Khóa protocol `vi-v1`: normalization, corpus micro CER/WER, nhóm lỗi và VIVOS test split.
- [x] Ghi ADR 001/002/003, cấu hình và hồ sơ baseline; package import được qua editable install.

**Checkpoint:** có model card/nguồn, cấu hình baseline, protocol đánh giá và kế hoạch subset dữ liệu trước khi tải model hoặc dataset lớn. Xác minh CUDA từ host và quyền dùng VIVOS là điều kiện đầu giai đoạn 2. Chưa có điểm ASR.

### Giai đoạn 2 — Nền dữ liệu và ASR inference

Đọc [báo cáo kiến thức cần nắm](03_asr/phase_2_learning_guide.md) trước khi thiết kế manifest, audio preprocessing và adapter inference.

- [x] Git đã có commit và remote ngày 2026-10-22; run mới ghi commit, dirty flag, hash mã nguồn và interpreter Miniconda. Các run CPU cũ giữ nguyên metadata gốc, không gán hồi tố Git revision.
- [x] Pin phiên bản dependency inference; mỗi run lưu model revision, config, manifest/archive checksum tùy cung cấp và môi trường.
- [x] Xác minh CUDA trên host và quyền nghiên cứu phi thương mại cho VIVOS.
- [x] Tải VIVOS từ Zenodo ngày 2026-10-19; MD5 `72972a8b14050f3f11ea7c3debabd7af`, SHA-256 `147477f7a7702cbafc2ee3808d1c142989d0dbc8d9fce8e07d5f329d5119e4ca`; metadata lưu tại `data/raw/vivos/vivos_metadata.json`.
- [x] Định nghĩa manifest và preprocessing có thể lặp lại: parser VIVOS, kiểm tra file/transcript/speaker leakage, metadata WAV, PCM decode/downmix/resample và JSONL ổn định.
- [x] Thiết kế interface ASR provider; hiện thực adapter Transformers Whisper/PhoWhisper với revision pin, tiếng Việt, `transcribe`, CUDA rõ ràng và metadata latency/VRAM.
- [x] Thêm kiểm thử cho normalization, manifest, audio preprocessing, inference contract và evaluator trên fixture tổng hợp.
- [x] Kiểm định bộ dữ liệu VIVOS và tạo manifest ngày 2026-10-19: 11.660 train, 760 test; 12.420 WAV, 16 kHz, mono, PCM 16-bit.
- [x] Chạy smoke 8 mẫu bằng CPU ngày 2026-10-19: CER 0.013850, WER 0.035714, RTF 0.5124, 0 lỗi. Run đầu có cảnh báo attention mask; rerun sau sửa đã sạch cảnh báo.
- [x] Chạy full test VIVOS cho PhoWhisper-small bằng CPU ngày 2026-10-19: 760/760, 0 lỗi, CER 0.027463, WER 0.062808, RTF 0.4801.
- [x] Chạy full test VIVOS cho Whisper-small bằng CPU ngày 2026-10-19: 760/760, 0 lỗi, CER 0.145556, WER 0.266770, RTF 0.4736.
- [x] Phân tích lỗi bước đầu theo câu và speaker trên cùng 760 mẫu; xem `docs/05_evaluation/error_analysis.md`.
- [x] Bud500 đã tải hai shard test, tìm được 34 mẫu khớp bộ keyword đã khóa; pilot 15 mẫu đầu được giữ riêng. Version 3 đánh giá số mẫu thực có và tránh lặp lại inference; mục tiêu 100 vẫn được theo dõi là thiếu.
- [x] Thêm bộ kiểm chứng summary/predictions/manifest, tính lại CER/WER và bộ nghe audit 10 câu lỗi + 3 câu đúng. Script `complete_phase2.py` điều phối phần còn thiếu và sinh báo cáo; không tự đánh dấu đã nghe.
- [ ] Nghe kiểm định các câu lỗi nặng; bổ sung bộ câu du lịch có quyền sử dụng, tách khỏi benchmark VIVOS.
- [ ] Đo bộ nhớ và tốc độ ASR trên GPU khi workload vLLM nhả VRAM; hiện kết quả chính thức chỉ có CPU.

**Checkpoint:** full VIVOS và toàn bộ 34 mẫu keyword Bud500 đều đã chạy hai model trên CPU, đối chiếu predictions/manifest; Bud500 thiếu mục tiêu 100 dù đã xét hết hai test shard. Chưa nghe kiểm định lỗi nặng hoặc đo VRAM/GPU; đây chưa phải giai đoạn 2 hoàn tất. Cần thêm dữ liệu đánh giá độc lập để đạt mục tiêu domain. Trong Miniconda, dùng `python scripts/complete_phase2.py --run-gpu` khi GPU đủ VRAM, sau đó điền audit và chạy lại không có flags để kiểm tra. [Báo cáo trạng thái](05_evaluation/phase2_status.md) theo dõi số mẫu và việc chấm riêng, không chạy lại model để bù số mẫu thiếu.

### Giai đoạn 3 — Voicebot dọc tối thiểu

- [ ] Xác định schema API request/response và trạng thái Idle, Listening, Processing, Thinking, Speaking, Error.
- [x] Tạo local Markdown ingestion/chunking/retrieval có citation/source metadata; giữ tools realtime tách khỏi RAG.
- [x] Chạy luồng văn bản đầu tiên `câu hỏi → intent/entity → retrieval → câu trả lời kèm nguồn`; có từ chối khi thiếu bằng chứng và CLI smoke cho một truy vấn Huế. Retrieval hiện là từ khóa; chưa có LLM, tools hoặc bộ đánh giá câu hỏi cố định.
- [ ] Nối audio input → VAD → ASR → luồng văn bản → TTS qua interface thay thế được; xử lý lỗi và thiếu dữ kiện ở từng lớp.
- [ ] Đo bộ nhớ và thời gian từng thành phần trên một GPU 6 GB; chốt cách chạy tuần tự hoặc phân bổ thiết bị dựa trên phép đo.

**Checkpoint:** luồng văn bản và một luồng hỏi đáp/lập lịch trình bằng giọng nói chạy end-to-end trên dữ liệu mẫu được cấp phép; hiển thị transcript và nguồn trả lời, xử lý được lỗi ở từng lớp.

### Giai đoạn 4 — API, UI và triển khai

- [ ] Chốt API và deployment architecture dựa trên latency, streaming và môi trường mục tiêu.
- [ ] Tạo API health, ASR và voicebot; giữ business logic ở `src/vietnamese_travel_voicebot/voicebot`.
- [ ] Xây UI tối giản tập trung voice, transcript và trạng thái hội thoại.
- [ ] Container hóa sau khi dependency và lệnh chạy đã ổn định; tài liệu hóa secrets, logging và vận hành.
- [ ] Ghi request ID, thời gian và lỗi theo từng bước VAD/ASR/RAG-Tools/LLM/TTS; quy định lưu, ẩn và xóa audio/transcript người dùng trước khi bật logging thực tế.

**Checkpoint:** kiểm thử API, luồng lỗi, cấu hình môi trường và hướng dẫn chạy cục bộ/container.

### Giai đoạn 5 — Domain adaptation và đánh giá

- [ ] Tạo travel-domain speech/evaluation set có quyền sử dụng rõ ràng, không trộn test vào train.
- [ ] Chạy domain adaptation/finetuning có kiểm soát, lưu từng experiment và cả kết quả không tốt.
- [ ] Phân tích lỗi theo vùng miền, tên riêng, số liệu, tiếng ồn và loại intent.
- [ ] So sánh bằng cùng protocol; chỉ thay methodology trong experiment version mới.
- [ ] Thiết kế vòng phản hồi có đồng ý của người dùng, kiểm định annotation và tách dữ liệu phản hồi khỏi test set đã khóa.

**Checkpoint:** báo cáo cải thiện và tradeoff tài nguyên/latency, có thể tái lập từ metadata.

## Tiêu chí hoàn thành scaffold

- `python3 main.py` chạy được mà không cần cài package bổ sung.
- Cấu trúc phản ánh các lớp data, ASR, training, evaluation, voicebot, knowledge và API.
- Tài liệu nêu rõ domain, kiến trúc, ranh giới dữ liệu và bước tiếp theo.
- Git ignore ngăn commit `.env`, audio, dataset local và checkpoint/model.
- Không có dữ liệu/model thật hoặc tuyên bố rằng pipeline đã được triển khai.

## Rủi ro cần giải quyết ở giai đoạn kế tiếp

| Rủi ro | Cách xử lý |
|---|---|
| Chọn model/dataset không phù hợp license hoặc phần cứng | Đánh giá nguồn, license, chi phí và subset trước khi tải |
| Metric thay đổi giữa các lần chạy | Chốt normalization, split, metric và lưu cấu hình theo experiment |
| LLM trả lời sai thông tin du lịch | Bắt buộc truy xuất nguồn cho dữ kiện và dùng tool cho realtime |
| Scaffold tạo cảm giác hệ thống đã hoạt động | Giữ module trống có TODO, không thêm fallback giả hoặc dữ liệu demo không được gắn nhãn |

## Áp dụng hai sơ đồ lộ trình tham khảo

Hai hình ở thư mục gốc là bản đồ kỹ năng và kiến trúc tham khảo, không phải danh sách dịch vụ phải triển khai. Với một GPU 6 GB và baseline nghiên cứu cục bộ, ưu tiên **dữ liệu có nguồn → inference có thể lặp lại → đánh giá → luồng văn bản du lịch → luồng giọng nói → vận hành/feedback**. Luồng văn bản ở giai đoạn 3 là bước tích hợp để kiểm tra RAG/tools/LLM; sản phẩm cuối vẫn là voicebot du lịch.

- **Áp dụng sớm:** Git/version ID, kiểm định dữ liệu, kiểm thử, metadata thí nghiệm, đo latency/VRAM và log lỗi theo lớp. Bộ hồ sơ experiment hiện có đủ cho baseline đầu tiên; [MLflow Tracking](https://mlflow.org/docs/latest/tracking/) có thể bổ sung khi số run hoặc nhu cầu so sánh tăng.
- **Bổ sung theo nhu cầu chia sẻ dữ liệu:** [DVC](https://doc.dvc.org/start) khi cần đồng bộ nhiều phiên bản dataset/model qua Git và storage riêng. Trước mắt checksum, manifest và đường dẫn storage đã đáp ứng mục tiêu kiểm định một benchmark.
- **Để sau khi có tải vận hành thực:** Airflow/MinIO/model registry, Kubernetes, Prometheus/Grafana/Loki/Tempo và serving phân tán. [Airflow](https://airflow.apache.org/docs/apache-airflow/2.10.5/) phục vụ workflow batch theo lịch; [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/latest/) hướng tới serving LLM ở quy mô lớn. Quyết định triển khai chúng cần số liệu tải, latency, dung lượng và nhu cầu nhiều người dùng.
- **Ngoài phạm vi hiện tại:** OCR, diarization, DPO/RLHF và huấn luyện foundation model. Chỉ mở rộng khi use case du lịch thật sự cần, có dữ liệu/quyền sử dụng và tiêu chí đánh giá riêng.
