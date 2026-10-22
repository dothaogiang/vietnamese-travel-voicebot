# Giai đoạn 2: kiến thức cần nắm để dựng dữ liệu và ASR inference

**Cập nhật ngày 2026-10-19:** Đã tải VIVOS và xác minh checksum; manifest có 11.660 train, 760 test (12.420 WAV). Cả PhoWhisper-small và Whisper-small đã chạy full 760 mẫu trên CPU: CER lần lượt `0.027463` và `0.145556`, không mẫu lỗi. **Giai đoạn 2 chưa hoàn tất:** chưa có tập đánh giá giọng nói du lịch hoặc phép đo GPU; hai tiến trình vLLM đang giữ VRAM. Xem [hồ sơ baseline](../experiments/baseline.md) và [phân tích lỗi](../05_evaluation/error_analysis.md).

## 1. Bản đồ công việc và ranh giới quyết định

| Việc trong giai đoạn 2 | Cần hiểu trước | Sản phẩm cần có |
|---|---|---|
| Dữ liệu | quyền dùng, split, speaker, audio, checksum | hồ sơ nguồn và manifest có thể kiểm tra |
| Preprocessing | sample rate, channel, resampling, cắt đoạn | audio đầu vào nhất quán, có metadata |
| Inference | Whisper, processor, generation, thiết bị | adapter ASR chạy model đã khóa revision |
| Đánh giá | chuẩn hóa, CER/WER, RTF, lỗi | kết quả tái lập trên test split cố định |

**Repo đã chốt:** `vinai/PhoWhisper-small` là baseline chính, `openai/whisper-small` là đối chứng; revision và tham số mục tiêu trong [`configs/asr.yaml`](../../configs/asr.yaml) và [ADR 001](../decisions/001_model_selection.md). VIVOS **bản gốc/test split** là benchmark có điều kiện về quyền dùng theo [ADR 002](../decisions/002_dataset_selection.md). Protocol `vi-v1` ở [`configs/evaluation.yaml`](../../configs/evaluation.yaml) và [mô tả metric](../05_evaluation/asr_metrics.md). Đây là lựa chọn của dự án, không phải quy tắc chung cho mọi hệ ASR. Theo [model card PhoWhisper](https://huggingface.co/vinai/PhoWhisper-small) và [model card Whisper-small](https://huggingface.co/openai/whisper-small), phải đọc riêng điều kiện sử dụng từng model trước khi triển khai.

**Đã xác minh từ terminal host của người dùng:** CUDA khả dụng, 1 GPU `NVIDIA GeForce RTX 3050`, 5.66 GiB VRAM. **Agent sandbox vẫn không thấy CUDA.** Chưa đo VRAM model, thời gian nạp, RTF, CER/WER hay segmentation; GPU sẵn sàng không chứng minh model đã chạy vừa VRAM. [Hồ sơ baseline](../experiments/baseline.md) ghi tách biệt hai môi trường.

## 2. ASR và Whisper: từ sóng âm đến văn bản

ASR nhận tín hiệu âm thanh và dự đoán chuỗi văn bản. Cần phân biệt **inference** (dùng model pretrained để suy luận), **fine-tuning** (cập nhật trọng số trên dữ liệu mới) và **domain adaptation** (điều chỉnh cho lời nói du lịch); giai đoạn 2 chỉ làm inference baseline. Nghiên cứu gốc trình bày Whisper như mô hình Transformer encoder–decoder đa nhiệm: đầu vào là đặc trưng log-Mel, decoder sinh token văn bản/timestamp theo task và ngôn ngữ. [Bài báo Whisper của OpenAI](https://arxiv.org/abs/2212.04356), [tài liệu Whisper trong Transformers](https://huggingface.co/docs/transformers/model_doc/whisper).

Những khái niệm cần tự giải thích được:

1. **Waveform → log-Mel:** waveform là dãy biên độ theo thời gian; đặc trưng thời gian–tần số là đầu vào model. Mã Whisper gốc đặt sample rate 16 kHz, cửa sổ FFT 400 mẫu, hop 160 mẫu và cửa sổ 30 giây; đây là đặc tính triển khai Whisper, không có nghĩa mọi audio dài hơn 30 giây phải bị cắt bỏ. [Mã âm thanh Whisper gốc](https://github.com/openai/whisper/blob/main/whisper/audio.py), [hướng dẫn long-form trong Transformers](https://huggingface.co/docs/transformers/model_doc/whisper).
2. **Encoder–decoder và sinh chuỗi:** encoder mã hóa âm thanh; decoder sinh token từng bước. Vì vậy thời gian inference phụ thuộc cả độ dài audio lẫn số token được sinh và cấu hình decoding. Kiểm tra `language="vi"` và `task="transcribe"` để đo nhận dạng tiếng Việt, tránh vô tình dùng nhận diện ngôn ngữ hoặc dịch. [Bài báo Whisper](https://arxiv.org/abs/2212.04356), [model card Whisper-small](https://huggingface.co/openai/whisper-small).
3. **PhoWhisper so với Whisper-small:** PhoWhisper được tác giả xây dựng cho tiếng Việt từ nền Whisper; so sánh phải dùng cùng split và protocol của repo, không lấy số công bố làm điểm của dự án. [Repository tác giả PhoWhisper](https://github.com/VinAIResearch/PhoWhisper), [model card](https://huggingface.co/vinai/PhoWhisper-small).
4. **Đoạn dài và lời nói ngắt quãng:** hiểu sự khác nhau giữa utterance có sẵn, VAD cắt theo lời nói, và chiến lược long-form có timestamp. Cắt sai biên có thể mất âm tiết hoặc ghép sai câu; lưu thời điểm cắt và version của chiến lược. Transformers mô tả cơ chế long-form của Whisper; mã pipeline cũng cảnh báo về `chunk_length_s` tùy cách dùng. [Tài liệu Whisper](https://huggingface.co/docs/transformers/model_doc/whisper), [mã pipeline ASR của Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py).

**Câu hỏi tự kiểm:** Tại sao cùng model nhưng `translate` và `transcribe` cho kết quả khác? Vì sao không được mặc định cắt mọi file ở giây thứ 30? Số CER tác giả công bố có thay cho CER `vi-v1` được không?

## 3. Audio: sample rate, mono, resampling và segmentation

Sample rate là số mẫu mỗi giây; **đổi nhãn metadata 48 kHz thành 16 kHz không phải resampling**. Khi giảm sample rate phải lọc để hạn chế aliasing rồi tạo dãy mẫu mới. Tài liệu torchaudio giải thích resampling và ảnh hưởng của bộ lọc/rolloff. [Hướng dẫn resampling của PyTorch Audio](https://docs.pytorch.org/audio/master/tutorials/audio_resampling_tutorial.html). Với Whisper, processor nên được truyền `sampling_rate` thật để tránh đầu vào sai âm thầm. [WhisperFeatureExtractor](https://huggingface.co/docs/transformers/model_doc/whisper).

Trong dự án, đích preprocessing hiện là **16 kHz, một kênh** theo [`configs/asr.yaml`](../../configs/asr.yaml). Khi đọc mỗi tệp, cần xác định sample rate gốc, số kênh, số mẫu, thời lượng (`samples / sample_rate`), định dạng và khả năng decode. Nếu có stereo, quy tắc downmix phải cố định và ghi version; không giả định kênh thứ nhất luôn là tiếng nói. Ghi cả thông số gốc và thông số sau biến đổi để kiểm tra. Mã Whisper gốc đọc audio thành mono waveform và resample về 16 kHz. [Mã `load_audio` của OpenAI](https://github.com/openai/whisper/blob/main/whisper/audio.py).

Đối với VIVOS, trước tiên giữ **một utterance gốc → một cặp audio/reference** để đánh giá split chính. Nếu sau này thêm VAD, silence trimming hoặc chunking, hãy giữ mapping từ segment về file, mốc thời gian và transcript gốc, rồi tạo preprocessing/experiment version khác. Đây là quy tắc tái lập của repo theo [kế hoạch](../implementation_plan.md), không phải kết luận rằng VAD luôn cải thiện ASR.

**Câu hỏi tự kiểm:** Nếu một clip có 48.000 mẫu ở 48 kHz, nó dài bao lâu và sau resample 16 kHz có xấp xỉ bao nhiêu mẫu? Nếu hai kênh ghi khác nhau, downmix sẽ ảnh hưởng thế nào?

## 4. VIVOS, quyền dùng và manifest

[Bản ghi VIVOS gốc trên Zenodo](https://zenodo.org/records/7068130) ghi phiên bản 1.0, khoảng 15 giờ tiếng nói đọc trong môi trường yên tĩnh; train có 46 speaker/11.660 utterance, test có 19 speaker/760 utterance và khoảng 45 phút. Trang gốc ghi **CC BY-NC-SA 4.0**, mục đích học thuật, archive khoảng 1,5 GB và MD5 được Zenodo công bố. Điều kiện này phải được kiểm tra theo mục đích dùng thực tế trước khi tải, chia sẻ, huấn luyện hoặc tích hợp sản phẩm; bản mirror có thể gắn nhãn khác nên không dùng để thay nguồn gốc. [Nguồn và license gốc](https://zenodo.org/records/7068130).

Sau khi có quyền dùng và tải từ nguồn gốc, cần lập hồ sơ: URL/DOI, version, ngày truy cập, tên và kích thước archive, checksum công bố và checksum tính lại, license, cấu trúc thư mục, số file, format/sample rate/channels/duration, bảng speaker và split. Checksum giúp phát hiện file tải lỗi hoặc thay đổi nội dung, không chứng minh dữ liệu được cấp phép. Zenodo công bố MD5 archive; có thể ghi thêm SHA-256 nội bộ, nhưng không tự suy ra chúng từ tên file. [Metadata VIVOS trên Zenodo](https://zenodo.org/records/7068130).

**Manifest** là bảng ánh xạ có thể kiểm tra. Parser hiện thực trong `src/vietnamese_travel_voicebot/data/manifest.py` ghi mỗi dòng JSONL với `utterance_id`, đường dẫn tương đối `audio_path`, `transcript`, `split`, `speaker_id`, `duration_s`, `sample_rate_hz`, `channels` và `sample_width_bytes`. Parser yêu cầu WAV PCM không rỗng, transcript hợp lệ, ID duy nhất, audio tồn tại, speaker không giao giữa train/test; nó không chứng minh transcript khớp audio và chưa kiểm định dữ liệu VIVOS thật. Checksum archive được lưu trong summary khi truyền `--dataset-sha256`; checksum không được suy ra từ manifest. [Thống kê split gốc của VIVOS](https://zenodo.org/records/7068130).

`data/` hiện chưa chứa dataset hay manifest thực tế. Một **smoke subset** lấy theo thứ tự ID ổn định trong manifest và được ghi `run_kind=smoke_subset`; không được báo như benchmark toàn test split. Dùng `--full` chỉ sau khi xác nhận manifest là VIVOS test đã khóa; đây là nguyên tắc repo trong [ADR 002](../decisions/002_dataset_selection.md).

## 5. Adapter inference với Transformers và GPU 6 GB

Transformers cung cấp `AutoProcessor`, mô hình speech seq2seq và pipeline ASR; processor biến waveform thành feature và decode token thành text. Đọc ví dụ chính thức để hiểu `sampling_rate`, `generate`, `batch_decode(skip_special_tokens=True)` và cách truyền task/ngôn ngữ. [Tài liệu Whisper](https://huggingface.co/docs/transformers/model_doc/whisper), [model card Whisper-small](https://huggingface.co/openai/whisper-small). Khi hiện thực, **pin phiên bản thư viện** bên cạnh revision model để tránh hành vi API trôi theo thời gian.

Interface `ASRProvider` nhận `AudioInput`: mono float32 đã decode, sample rate 16 kHz; trả `ASRResult` gồm transcript, model ID/revision, thời gian inference và peak allocator VRAM nếu chạy CUDA. `WhisperASRProvider` nạp đúng revision từ Transformers, đặt `eval()`, dùng `torch.inference_mode()`, ép `language=vi` và `task=transcribe`. Yêu cầu CUDA không tự chuyển CPU nếu CUDA không khả dụng. Bộ đọc WAV chỉ hỗ trợ PCM; compressed WAV và format khác cần chuyển đổi rõ ràng sang PCM trước.

Với target RTX 3050 6 GB, bắt đầu batch 1 và dtype `float16` **nếu** CUDA, model và toán tử tương thích. `float16` là giả thuyết tiết kiệm VRAM cần đo; khả năng fit không thể kết luận chỉ từ số tham số hoặc kích thước checkpoint vì còn activation, cache khi generate, framework và bộ nhớ nền. PyTorch phân biệt `max_memory_allocated` (tensor do allocator theo dõi) với bộ nhớ GPU tổng thể; ghi cả số đó và VRAM từ công cụ hệ thống. [Tài liệu CUDA PyTorch](https://docs.pytorch.org/docs/cuda.html), [max_memory_allocated](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html).

Quy trình runtime cần hiểu: kiểm tra `torch.cuda.is_available()` **trên host nơi inference chạy**; tải đúng model revision; `model.eval()` và `torch.inference_mode()` khi không cần gradient; kiểm tra dtype và lỗi OOM; không ghi một lần chạy CPU như kết quả GPU. `inference_mode()` không tự gọi `eval()`. [Tài liệu inference_mode của PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html), [CUDA availability](https://docs.pytorch.org/docs/cuda.html). `autocast` là cơ chế riêng và phải được kiểm tra tính tương thích, không xem mọi phép toán đều chuyển sang `float16`. [Ví dụ AMP chính thức](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

## 6. Đánh giá: CER, WER, tốc độ và phân tích lỗi

Protocol `vi-v1` đã khóa: Unicode NFC → lowercase → dấu câu thành khoảng trắng → gộp whitespace; giữ dấu tiếng Việt và chữ số. Áp dụng giống nhau cho reference/hypothesis. **CER** là tổng lỗi Levenshtein trên ký tự (gồm khoảng trắng) chia tổng ký tự reference; **WER** là tổng lỗi trên token tách bằng whitespace chia tổng token reference. Tính lỗi **từng utterance rồi cộng** trước khi chia (corpus micro), không lấy trung bình phần trăm từng câu. Công thức và hiện thực nằm tại [protocol repo](../05_evaluation/asr_metrics.md) và `src/vietnamese_travel_voicebot/evaluation/metrics.py`. Đây là định nghĩa của dự án; token tách bằng khoảng trắng trong tiếng Việt không nhất thiết trùng từ ngữ nghĩa.

Ví dụ reference `"đà nẵng"`, hypothesis `"đà nang"`: một thay thế ký tự tạo CER `1/7` theo quy tắc đếm cả khoảng trắng, một token sai tạo WER `1/2`. Nếu reference rỗng sau chuẩn hóa, dữ liệu lỗi; nếu hypothesis rỗng, toàn bộ reference là deletion. Cần đọc ba thao tác edit distance: **substitution, deletion, insertion**. Khi phân tích lỗi, lưu cả text gốc và text chuẩn hóa để nhận ra lỗi do model hay do quy tắc normalization. [Protocol `vi-v1`](../05_evaluation/asr_metrics.md).

**RTF (real-time factor)** trong báo cáo này định nghĩa là `thời gian xử lý / thời lượng audio`; RTF < 1 nghĩa là lần chạy đó xử lý nhanh hơn thời lượng clip. Evaluator đo thời gian giải mã/resample + inference cho từng mẫu; model load và warm-up được ghi riêng/loại khỏi RTF. Hai full run CPU đạt mean RTF lần lượt `0.4801` và `0.4736` trên cùng manifest, batch và dtype. GPU xử lý bất đồng bộ, nên phải đồng bộ trước/sau vùng đo hoặc dùng CUDA events để tránh báo thiếu thời gian. [PyTorch CUDA synchronize](https://docs.pytorch.org/docs/cuda.html), [CUDA events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.streams.Event.html). RTF CPU dưới 1 chưa chứng minh độ trễ hội thoại toàn hệ thống hoặc hiệu năng GPU.

Kết quả hiện tại cho phép phân nhóm lỗi để chẩn đoán: tên địa danh/tên riêng, dấu thanh, chữ số/ngày giờ, phát âm vùng miền, nhiễu, từ bị bỏ sót và câu dài. [Phân tích bước đầu](../05_evaluation/error_analysis.md) mới xác định speaker/câu có lỗi cao; cần nghe kiểm định trước khi gán nguyên nhân. VIVOS là tiếng đọc yên tĩnh theo [mô tả nguồn gốc](https://zenodo.org/records/7068130), nên benchmark này chưa đo đủ speech du lịch thực địa. Lưu sample ID, audio provenance, reference, hypothesis, thời lượng, group và cả ca thất bại. Không chọn lại test set hoặc chỉnh normalization theo lỗi đã xem; nếu đổi phương pháp thì tạo experiment version mới theo [kế hoạch](../implementation_plan.md).

## 7. Trình tự đọc và thực hành đề xuất

1. **60 phút — Bối cảnh:** đọc [kế hoạch](../implementation_plan.md), [ADR 001](../decisions/001_model_selection.md), [ADR 002](../decisions/002_dataset_selection.md) và [VIVOS gốc](https://zenodo.org/records/7068130). Viết lại bằng lời của mình: mục đích dữ liệu, giới hạn license, split cố định.
2. **90 phút — ASR/Whisper:** đọc phần mô hình và inference trong [bài báo Whisper](https://arxiv.org/abs/2212.04356), [docs Whisper](https://huggingface.co/docs/transformers/model_doc/whisper), [PhoWhisper](https://github.com/VinAIResearch/PhoWhisper). Vẽ đường waveform → log-Mel → encoder → decoder → transcript; giải thích `transcribe` so với `translate`.
3. **45 phút — Audio:** đọc [mã audio Whisper](https://github.com/openai/whisper/blob/main/whisper/audio.py) và [resampling tutorial](https://docs.pytorch.org/audio/master/tutorials/audio_resampling_tutorial.html). Tính thử số mẫu/thời lượng; định nghĩa các metadata phải lưu.
4. **60 phút — Runtime:** đọc [CUDA](https://docs.pytorch.org/docs/cuda.html), [inference_mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html), [Transformers ASR](https://huggingface.co/docs/transformers/tasks/asr). Viết contract input/output/error dự kiến trước khi tải model.
5. **45 phút — Đánh giá:** đọc [protocol `vi-v1`](../05_evaluation/asr_metrics.md), tự tính CER/WER cho hai câu và trình bày cách đo RTF/VRAM. Đọc lại [hồ sơ baseline](../experiments/baseline.md) để biết điều gì vẫn chưa được đo.

Thời lượng trên là **gợi ý học tập**, không phải ước tính thời gian hoàn thành giai đoạn 2.

## 8. Khi nào đã sẵn sàng triển khai giai đoạn 2?

- Giải thích được từ raw audio đến transcript, vì sao Whisper cần waveform đúng sample rate, và cách xử lý clip dài mà không làm mất lời nói.
- CUDA/VRAM đã được xác minh từ terminal host; cần chạy inference cũng trên host đó để đo peak VRAM và phân biệt OOM với lỗi decode audio. Tool sandbox hiện không có CUDA; xem [hồ sơ baseline](../experiments/baseline.md).
- Xác nhận quyền dùng VIVOS cho mục đích cụ thể; biết kiểm tra checksum, split, speaker và chất lượng bản ghi trước khi tạo manifest. Dữ liệu đã tải và qua kiểm định kỹ thuật; kiểm định nghe thủ công mẫu lỗi nặng vẫn còn.
- Contract audio, manifest và adapter đã có kiểm thử fixture; chưa xác nhận với VIVOS/model thật.
- Có thể tự tính CER/WER `vi-v1`, thiết kế phép đo RTF, lưu revision/config/runtime/sample IDs và phân tích lỗi mà không thay test split sau khi nhìn kết quả.

### Cách chạy code Phase 2

Lệnh sau tải VIVOS chính thức (đã được chấp thuận cho nghiên cứu phi thương mại), kiểm MD5, tạo manifest và chạy smoke/full lần lượt cho PhoWhisper-small và Whisper-small trên host GPU:

```bash
bash scripts/run_phase2.sh
```

Nếu đã có archive chính thức: `bash scripts/run_phase2.sh --archive /path/to/vivos.tar.gz`. Script giữ riêng mỗi run dưới `experiments/001_asr_baseline/outputs/<timestamp>/`.

Để chạy từng bước, dùng block dưới đây. Downloader lưu URL/version/license/kích thước/checksum và giải nén sau khi MD5 khớp Zenodo. `prepare_data.py` kiểm tra `train/` và `test/` chứa `prompts.txt` cùng `waves/<speaker>/<utterance_id>.wav`:

```bash
set -e
python scripts/check_environment.py --require-cuda
python scripts/download_data.py
python scripts/prepare_data.py --dataset-root data/raw/vivos/VIVOS
DATASET_SHA256="$(sha256sum data/raw/vivos/downloads/vivos.tar.gz | awk '{print $1}')"
python scripts/run_asr.py --manifest data/manifests/test.jsonl --dataset-root data/raw/vivos/VIVOS --dataset-sha256 "$DATASET_SHA256"
```

Lệnh ASR ở trên chạy smoke 8 hàng đầu theo ID đã sắp xếp; bước `check_environment.py` phải chạy trước và quy trình dừng nếu một lệnh lỗi (`set -e`). Script tải archive chính thức 1,5 GB, xác minh MD5 Zenodo rồi mới giải nén; VIVOS được cấp phép CC BY-NC-SA 4.0 cho mục đích học thuật phi thương mại. Sau smoke thành công, chạy toàn bộ test split bằng cùng lệnh và thêm `--full`.

Kết quả được ghi vào `experiments/001_asr_baseline/outputs/primary/`. Khi kiểm tra smoke thành công, benchmark toàn split bằng cách thêm `--full`; kết quả và output phải nằm ngoài Git. Đối chứng dùng `--role comparator` và thư mục output riêng. Không chạy lệnh inference cho đến khi model card/license, quyền dữ liệu và CUDA đã được xác minh.

Kiểm thử code không cần tải model/dataset:

```bash
python -m unittest discover -s tests -v
```

Các kiểm thử hiện dùng fixture tổng hợp và fake model trên CPU; chúng chứng minh contract/pipeline được nối, không chứng minh checkpoint thật tương thích, chạy được trên RTX 3050 hay đạt metric nào.

Giai đoạn 2 chỉ hoàn tất sau khi smoke và toàn bộ VIVOS test split chạy trên host GPU, metadata/checksum được xác nhận, và CER/WER/RTF/VRAM cùng các lỗi được lưu vào `experiments/001_asr_baseline/` theo [checkpoint](../implementation_plan.md).
