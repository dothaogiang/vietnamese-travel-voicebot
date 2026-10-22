# Baseline experiment

## Hồ sơ giai đoạn 1 (2026-10-14)

**Trạng thái tại thời điểm giai đoạn 1 (2026-10-14):** protocol và ứng viên đã khóa; lúc đó chưa tải dataset/model và chưa có CER/WER. Kết quả full run hiện tại nằm ở các mục bên dưới.

| Mục | Giá trị |
|---|---|
| Máy mục tiêu | Ubuntu 24.04.5 LTS; một NVIDIA GeForce RTX 3050 6 GB (GA107 theo `lspci`) |
| Driver quan sát được | `/proc/driver/nvidia/version`: 595.91.07 |
| Runtime cục bộ | Python 3.13.9; PyTorch 2.9.1+cu128 đã có trong môi trường hệ thống |
| GPU trong tool sandbox | `torch.cuda.is_available()` trả `False`; `nvidia-smi` không liên lạc được driver. |
| GPU trên host người dùng (xác minh 2026-10-16) | `torch.cuda.is_available()=True`; 1 × NVIDIA GeForce RTX 3050; PyTorch báo 5.66 GiB VRAM. |
| Tài nguyên lúc khảo sát | RAM 31 GiB, khoảng 15 GiB available; swap 8 GiB, khoảng 4,3 GiB free; workspace còn khoảng 401 GiB |
| Model chính | `vinai/PhoWhisper-small` @ `a86b604c346caf7148c37512eafe783a16420adb` |
| Model đối chứng | `openai/whisper-small` @ `973afd24965f72e36ca33b3055d56a652f456b4d` |
| Dataset/split | VIVOS bản gốc, test split; CC BY-NC-SA 4.0, chỉ dùng theo quyền được cấp |
| Audio/inference dự kiến | 16 kHz mono, tiếng Việt, `transcribe`, batch 1, CUDA float16; host GPU đã xác minh, mức dùng VRAM thực tế chưa đo |
| Metric | `vi-v1`: corpus micro CER chính, whitespace-token WER phụ |

Khóa cấu hình tại `configs/asr.yaml` và `configs/evaluation.yaml`. Các nguồn và giới hạn được nêu trong [ADR 001](../decisions/001_model_selection.md), [ADR 002](../decisions/002_dataset_selection.md) và [protocol](../05_evaluation/asr_metrics.md). Mục tiêu RTF ≤ 1 chỉ là tiêu chí cần đo cho voicebot tương lai, chưa được chứng minh trên GPU này.

## Điều kiện trước lần chạy giai đoạn 2

1. CUDA trên host người dùng đã xác minh; tool sandbox vẫn không truy cập được GPU.
2. Quyền dùng VIVOS cho nghiên cứu phi thương mại đã được chấp thuận; archive chính thức đã tải và checksum khớp metadata Zenodo. Manifest đã kiểm định: 11.660 train, 760 test.
3. Cài dependency inference theo phiên bản đã kiểm tra; tải từng revision model đã khóa, tránh lưu trọng số vào Git.
4. Chạy smoke subset đã xác định trước; đo VRAM, RTF và lỗi; sau đó chạy toàn bộ test split với cùng preprocessing và protocol.
5. Lưu lệnh, cấu hình snapshot, seed, môi trường, kết quả từng model, nhóm lỗi và cả lần chạy thất bại trong `experiments/001_asr_baseline/`.

Không thay test set, normalization, metric hoặc preprocessing theo kết quả đã xem. Nếu cần đổi methodology, mở experiment version mới.

## Trạng thái triển khai pipeline (2026-10-19)

Đã hiện thực và kiểm thử bằng dữ liệu WAV tổng hợp: `scripts/prepare_data.py` tạo JSONL manifests; `WhisperASRProvider` là adapter chung cho model Whisper/PhoWhisper; `scripts/run_asr.py` chạy smoke 8 mẫu mặc định hoặc full split khi có `--full`. Artifact gồm `predictions.jsonl` và `summary.json`, có model revision, hash manifest, cấu hình, runtime, CER/WER, RTF và peak VRAM allocator.

Đã bổ sung `bash scripts/run_phase2.sh`; script hiện dùng Python trong Miniconda environment đang active, xác minh CUDA/VRAM còn trống, tải/kiểm archive, tạo manifest rồi chạy primary/comparator theo thứ tự smoke → full bằng CUDA/FP16; mỗi lần chạy ghi ra một thư mục timestamp riêng. Script dừng khi thiếu dữ liệu hoặc GPU, không tải model trước khi manifest đã tồn tại.

## Smoke run đầu tiên (2026-10-19)

PhoWhisper-small chạy được trên **CPU** với 8 mẫu VIVOS test: CER `0.013850`, WER `0.035714`, mean RTF `0.5124`, 0 mẫu lỗi. Run đầu có cảnh báo attention mask và chỉ được giữ làm lần thử ban đầu. Sau khi adapter được sửa để truyền mask và dùng tham số `dtype` mới, smoke rerun hoàn thành sạch cảnh báo. Artifact được chốt tại `experiments/001_asr_baseline/outputs/primary_smoke_rerun/summary.json` và `predictions.jsonl`. `peak_gpu_memory_bytes=null` xác nhận run này không dùng GPU.

GPU host lúc đó được xác nhận CUDA hoạt động, nhưng hai tiến trình `VLLM::EngineCore` giữ khoảng 4,27 GiB trên RTX 3050; run CUDA trước đó thất bại khi nạp model. Vì vậy hai full run bên dưới dùng CPU. Agent sandbox không truy cập CUDA. Metadata Git chưa đọc được tại thời điểm run nên hai full run cũ không có commit/version ID. Git đã được khôi phục ngày 2026-10-22; run mới ghi commit, dirty flag và hash source.

## Full test PhoWhisper-small (2026-10-19)

Full manifest VIVOS test đã chạy xong trên **CPU**: 760/760 mẫu, 0 lỗi, CER `0.027463`, WER `0.062808`, mean RTF `0.4801`. `evaluation_complete=true`, `full_manifest_evaluated=true`; có 760 dòng predictions trong `experiments/001_asr_baseline/outputs/primary/`. RTF dưới 1 nghĩa là inference trung bình nhanh hơn thời lượng audio trên runtime CPU này; không dùng làm kết luận về GPU hoặc độ trễ dịch vụ đồng thời.

Lệnh run không truyền `--dataset-sha256`, nên trường `dataset_sha256` trong `summary.json` là `null`; archive checksum vẫn được lưu riêng ở `data/raw/vivos/vivos_metadata.json`.

## So sánh hai model trên full VIVOS test (2026-10-19)

| Model | Thiết bị | Mẫu/lỗi | CER | WER | Mean RTF |
|---|---|---:|---:|---:|---:|
| PhoWhisper-small | CPU float32 | 760 / 0 | 0.027463 | 0.062808 | 0.4801 |
| Whisper-small | CPU float32 | 760 / 0 | 0.145556 | 0.266770 | 0.4736 |

Whisper-small có 760 prediction ở `experiments/001_asr_baseline/outputs/comparator_full_cpu/`; `evaluation_complete=true` và `full_manifest_evaluated=true`. Hai run có cùng manifest SHA-256 `fa94c08440db9b8d77b3b42ef584a6276286b9060c55e00b94beacd0fcdfecdb`, protocol `vi-v1`, số ký tự/token reference và cấu hình audio/inference; chỉ model/revision và role khác nhau. Cả hai run không truyền `--dataset-sha256`, nên summary có giá trị `null` tại trường đó; đối chiếu archive qua `data/raw/vivos/vivos_metadata.json`.

PhoWhisper giảm CER 11,81 điểm phần trăm và WER 20,40 điểm phần trăm so với Whisper-small trên VIVOS test. Mean RTF gần nhau, không nên kết luận model nào nhanh hơn từ một lần chạy CPU. Kết quả này không chứng minh chất lượng trên câu hỏi du lịch; cần tập domain riêng. [Phân tích lỗi bước đầu](../05_evaluation/error_analysis.md) ghi thống kê theo câu và speaker, không suy đoán nguyên nhân khi chưa nghe audio.

## Kiểm tra lại trong Miniconda (2026-10-22)

Đã dựng lại manifests vào thư mục tạm để kiểm định 12.420 WAV mà không ghi đè benchmark. Manifest train 11.660 mẫu và test 760 mẫu khớp byte-for-byte với bản đã khóa. SHA-256 archive được tính lại là `147477f7a7702cbafc2ee3808d1c142989d0dbc8d9fce8e07d5f329d5119e4ca`, khớp hồ sơ nguồn. `complete_phase2.py` tính lại CER/WER từ hai predictions và đối chiếu ID/reference/hash manifest, đều đạt.

Đã chạy smoke PhoWhisper thật từ cache, CPU float32, hai hàng đầu test: CER `0.024390`, WER `0.111111`, 0 lỗi; đây chỉ là kiểm tra runtime, không thay điểm full benchmark hoặc smoke 8 mẫu trước đó. Artifact: `experiments/001_asr_baseline/outputs/phase2_verification_20261022T072332Z/`. Interpreter ghi trong summary là `/home/giang/miniconda3/envs/vietnamese-travel-voicebot/bin/python` (Python 3.13.16), có Conda prefix, Git commit và hash source; dirty flag là `true`. Bộ kiểm thử tại thời điểm này đạt 26/26, gồm export Bud500 trên WAV/Parquet tự tạo và kiểm chứng điều kiện hoàn tất; chưa phải kiểm thử Bud500 thật.

Bộ nghe 13 mẫu đã được chuẩn bị. Nghe/ghi chú, Bud500 sau cấp quyền và đo GPU vẫn còn thiếu; xem [trạng thái Phase 2](../05_evaluation/phase2_status.md).

## Bud500 thí điểm: shard test đầu (2026-10-22)

Đã tải shard 306 MB và chạy 15 câu khớp keyword trong 3.750 hàng. Artifacts tại `experiments/002_bud500_eval/outputs/20261022T081230Z-2934158/`, manifest thuộc `data/raw/bud500/travel_test_v1/`. Hai run đã được đối chiếu lại ID/reference/manifest hash và tính lại metric, đều hợp lệ.

| Model | Mẫu/lỗi | CER | WER | Mean RTF |
|---|---:|---:|---:|---:|
| PhoWhisper-small | 15 / 0 | 0.088696 | 0.124088 | 0.5777 |
| Whisper-small | 15 / 0 | 0.271304 | 0.364964 | 0.5712 |

Đây là thí điểm nhỏ, chưa đạt mục tiêu 100 mẫu trong cấu hình Phase 2. Không suy rộng điểm 15 mẫu thành chất lượng toàn Bud500 hoặc hệ thống du lịch. Chuẩn bị version 3 mở rộng có giới hạn sang shard test thứ hai, giữ nguyên keyword, metric và run cũ; nếu toàn bộ nguồn này vẫn thiếu, tiếp tục báo thiếu dữ liệu thay vì nới tiêu chí.

## Bud500 version 3: toàn bộ 34 mẫu khớp keyword trong hai shard test (2026-10-22)

Hai shard đã được xét hết, cho 34 mẫu, tổng audio 88,63 giây. Đã chạy thực tế cả hai model trong Miniconda trên CPU float32, dùng model cache offline. Artifacts: `experiments/002_bud500_eval/outputs/test_keyword_v3/20261022T085657Z-2/`; manifest: `data/raw/bud500/travel_test_v3/manifest.jsonl`. Metric đã tính lại từ predictions và đối chiếu ID/reference/hash manifest cùng fingerprint inventory nguồn, đều hợp lệ.

| Model | Mẫu/lỗi | CER | WER | Mean RTF |
|---|---:|---:|---:|---:|
| PhoWhisper-small | 34 / 0 | 0.082886 | 0.116129 | 0.6281 |
| Whisper-small | 34 / 0 | 0.218726 | 0.303226 | 0.6223 |

`bud500_evaluated=true` xác nhận đã chấm đủ manifest hiện có. Mục tiêu 100 mẫu vẫn chưa đạt; không hạ ngưỡng hay nhân bản mẫu để đánh dấu hoàn thành. Lần gọi lại `complete_phase2.py --run-bud500` đã được kiểm tra: dùng lại kết quả hợp lệ và không tạo thêm run inference. 28/28 unittest đạt trong Miniconda. Đây là đánh giá mô tả trên một tập keyword nhỏ, không phải bằng chứng cho toàn bộ domain du lịch.
