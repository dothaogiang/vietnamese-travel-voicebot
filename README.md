# Vietnamese Travel Planning & Destination Voicebot

Trợ lý giọng nói tiếng Việt hỗ trợ hỏi thông tin điểm đến, phương tiện, hoạt động và lập lịch trình du lịch. Giai đoạn 1 đã chọn ASR baseline và khóa protocol đánh giá. Phase 2 đã chuẩn bị VIVOS và chạy full benchmark CPU cho PhoWhisper-small cùng Whisper-small. Phase 3 có lát cắt hỏi đáp văn bản với local lexical RAG; tập đánh giá du lịch, LLM, hội thoại giọng nói và API còn đang triển khai.

## Bắt đầu

```bash
python3 main.py
```

Lệnh này chạy starter nhẹ, không tải model/dữ liệu và chưa gọi dịch vụ bên ngoài.

## Cài môi trường Miniconda cho ASR GPU

Môi trường Conda chỉ chứa Python 3.13 và pip; các thư viện Python được cài bằng pip. CUDA runtime đi kèm wheel PyTorch, vì vậy không cần cài CUDA Toolkit vào Conda. Cài PyTorch 2.9.1 + CUDA 12.8 theo lệnh Linux chính thức của PyTorch, rồi cài package ASR:

```bash
conda env create -f environment.yml
conda activate vietnamese-travel-voicebot
python -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[asr]'
python scripts/check_environment.py --require-cuda
```

Lệnh cuối phải báo `cuda_available=True` và nhận diện RTX 3050. `pip install -e '.[asr]'` giữ các phiên bản ASR trong `pyproject.toml`; PyTorch cùng phiên bản CUDA được cài trước theo chỉ dẫn chính thức để pip nhận đúng wheel CUDA.

Để cài package nhẹ cho chỉnh sửa tài liệu/chạy phần không dùng ASR, dùng `python -m pip install -e .` trong environment này. Không cài `torchvision`, notebook hoặc thư viện training nếu chưa cần.

## Phase 2: dữ liệu và ASR

Môi trường chạy là **Miniconda `vietnamese-travel-voicebot`**. Để tiếp tục từ kết quả VIVOS đã có, tạo báo cáo kiểm tra và bộ nghe lỗi:

```bash
conda activate vietnamese-travel-voicebot
python scripts/complete_phase2.py
```

Mở `experiments/001_asr_baseline/outputs/phase2_audit/review.html`, nghe 10 câu lỗi nặng và 3 câu đúng được chọn. Điền `review.csv` cùng thư mục: `review_status=reviewed`, một `error_group` có trong hướng dẫn HTML và ghi nhận vào `notes`. Script giữ nguyên các ghi chú khi chạy lại. Bộ nghe dùng audio gốc, không nhân bản dataset.

Sau khi tài khoản được cấp quyền Bud500 và GPU còn ít nhất 2,5 GiB VRAM trống, chạy phần còn thiếu:

```bash
python scripts/complete_phase2.py --run-bud500 --run-gpu
```

Lệnh này chạy hai model tuần tự trên Bud500 và full VIVOS GPU/FP16; mỗi lần ghi vào thư mục mới. Báo cáo ở [trạng thái Phase 2](docs/05_evaluation/phase2_status.md), cấu hình tại `configs/phase2.yaml`. Exit code `2` nghĩa là còn điều kiện chưa đạt, xem các dòng `PENDING`; chỉ báo hoàn tất khi có đủ benchmark, phép đo GPU và review audio. Các run mới ghi Git commit, hash source, interpreter và Conda prefix. Nếu GPU hoặc quyền dữ liệu chưa sẵn sàng, báo cáo vẫn được sinh.

Từ thư mục gốc repository và sau khi activate Conda environment, chạy toàn bộ baseline và đối chứng trên host bằng một lệnh. Script dùng Python của Conda đang active, tải/kiểm MD5 VIVOS, tạo manifest, rồi chạy smoke và full test split tuần tự cho hai model:

```bash
bash scripts/run_phase2.sh
```

Nếu đã tải archive chính thức, dùng `bash scripts/run_phase2.sh --archive /path/to/vivos.tar.gz`. Kết quả mỗi lần chạy được lưu vào thư mục có timestamp dưới `experiments/001_asr_baseline/outputs/`.

Để chạy từng bước thủ công:

```bash
set -e
python scripts/check_environment.py --require-cuda
python scripts/download_data.py
python scripts/prepare_data.py --dataset-root data/raw/vivos/VIVOS
DATASET_SHA256="$(sha256sum data/raw/vivos/downloads/vivos.tar.gz | awk '{print $1}')"
python scripts/run_asr.py --manifest data/manifests/test.jsonl --dataset-root data/raw/vivos/VIVOS --dataset-sha256 "$DATASET_SHA256"
```

Lệnh ASR mặc định chạy 8 mẫu smoke; chỉ thêm `--full` sau khi smoke hợp lệ để chạy cả split. Tải VIVOS chính thức (1,5 GB) có kiểm tra MD5 trước khi giải nén; chỉ dùng theo CC BY-NC-SA 4.0 cho mục đích học thuật phi thương mại. Kết quả ở `experiments/001_asr_baseline/outputs/`, thư mục này và dữ liệu/model đều được loại khỏi Git. Xem [hướng dẫn Phase 2](docs/03_asr/phase_2_learning_guide.md) để biết điều kiện GPU, quyền dùng dữ liệu, lệnh kiểm thử và giới hạn hiện tại.

## Bud500: đánh giá thêm trên tiếng Việt nhiều chủ đề

Kho [Bud500 của VietAI](https://huggingface.co/datasets/linhtran92/viet_bud500) là dữ liệu tiếng Việt có audio và transcription, có nội dung du lịch xen trong nhiều chủ đề. Sau khi đăng nhập Hugging Face và chấp nhận điều kiện truy cập trên trang dataset, cài dependency nhỏ cho Parquet và chạy một lệnh trên máy host:

```bash
python -m pip install -e '.[bud500]'
bash scripts/run_bud500.sh
```

Shard test đầu (306 MB) cho 15 mẫu khớp từ khóa; hai model đã chạy thành công. Sau khi xét **cả hai shard test** (khoảng 608 MB tổng cộng), có 34 mẫu khớp keyword. Script đánh giá cả 34 mẫu và báo rõ mục tiêu 100 chưa đạt. Khi đã có kết quả hợp lệ của hai model trên cùng manifest, chạy lại lệnh sẽ bỏ qua inference. Không tải thêm train hoặc đổi keyword để làm đạt checklist; muốn đạt 100 cần thêm dữ liệu đánh giá độc lập.

Chạy tiếp bằng `python scripts/complete_phase2.py --run-bud500`. Cache cũ 306 MB được tái dùng; phần tải thêm tối đa khoảng 302 MB. Slice kết hợp version 3 nằm tại `data/raw/bud500/travel_test_v3/`, kết quả tại `experiments/002_bud500_eval/outputs/test_keyword_v3/`; bộ 15 mẫu và run cũ được giữ nguyên. Nếu tải bằng trình duyệt, có thể cung cấp cả hai file qua `bash scripts/run_bud500.sh --parquet /duong/dan/test-00000-....parquet --parquet /duong/dan/test-00001-....parquet`. Xem [protocol Bud500](docs/05_evaluation/bud500_protocol.md). Tập lọc keyword chưa đại diện cho hội thoại du lịch tự nhiên.

## Phase 3: hỏi đáp du lịch bằng văn bản

Luồng text đầu tiên đã nối intent/entity cơ bản với retrieval trên Markdown local và trả lời bằng đoạn nguồn, hoặc báo thiếu dữ kiện. Cài PyYAML rồi hỏi thử:

```bash
python -m pip install -e '.[voicebot]'
python scripts/ask_travel.py "Quần thể di tích Huế được ghi danh năm nào?"
```

Hiện retrieval dựa trên trùng từ khóa, chưa có embedding/LLM, dữ liệu mẫu còn rất nhỏ. Xem [Phase 3](docs/04_voicebot/architecture.md) và [quy ước nguồn dữ liệu](knowledge_base/README.md).

## Bản đồ repository

- `configs/`: cấu hình nền, ASR, RAG, voicebot và đánh giá.
- `src/vietnamese_travel_voicebot/`: package chứa data, ASR, training, evaluation, voicebot, knowledge và utilities.
- `api/`: lớp HTTP, schemas và routes; tách khỏi business logic.
- `data/`, `models/`: vùng local được bảo vệ khỏi Git; không lưu audio/checkpoint trong repository.
- `knowledge_base/`: tài liệu du lịch có nguồn và metadata.
- `docs/`: định nghĩa bài toán, kiến trúc, chiến lược dữ liệu, đánh giá và quyết định.
- `experiments/`: hồ sơ cấu hình, metadata và kết quả từng lần chạy.
- `notebooks/`: exploration/phân tích; không chứa production logic.

## Trạng thái

Host đã xác nhận RTX 3050 5.66 GiB và CUDA hoạt động. VIVOS đã tải, checksum hợp lệ và manifest train/test đã tạo. Hai model đã chạy full 760 mẫu VIVOS test trên CPU; PhoWhisper-small CER 0.027463, Whisper-small CER 0.145556, cả hai không có mẫu lỗi. Luồng hỏi đáp văn bản có thể chạy trên một nguồn mẫu Huế; Bud500 có pilot hợp lệ 15 mẫu, còn thiếu mục tiêu 100 mẫu và phép đo GPU. Xem [kế hoạch triển khai](docs/implementation_plan.md), [báo cáo Phase 2](docs/03_asr/phase_2_learning_guide.md), [hồ sơ baseline](docs/experiments/baseline.md) và [hướng dẫn dự án](PROJECT_GUIDELINES.md).
