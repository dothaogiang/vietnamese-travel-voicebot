# Bud500: tập đánh giá bổ sung

## Nguồn và quyền dùng

[Trang Bud500 gốc trên Hugging Face](https://huggingface.co/datasets/linhtran92/viet_bud500) mô tả corpus ASR tiếng Việt khoảng 500 giờ, nhiều chủ đề (có du lịch), các split train/validation/test, mỗi hàng có `audio` và `transcription`. Repo **gated**: tài khoản phải chấp nhận điều kiện truy cập. Nhãn license trên Hub là `cc-by-nc-sa-4.0`, trong thân dataset card lại có đoạn Apache-2.0; hai thông tin này chưa nhất quán. Chỉ dùng cho nghiên cứu phi thương mại trong dự án này và xác minh quyền cụ thể trước khi phân phối/đưa vào sản phẩm. Không đưa audio, transcript hoặc parquet lên Git.

## Chọn mẫu và tái lập

Thí điểm ban đầu chỉ xét `data/test-00000-of-00002-531c1d81edb57297.parquet` (306 MB, 3.750 hàng), có **15** mẫu khớp bộ keyword đã đặt. Mục tiêu 100 mẫu chưa đạt dù cả hai model đã chạy đúng trên 15 mẫu; không đổi threshold hoặc keyword sau khi xem kết quả.

Phiên bản 3 của `scripts/prepare_bud500.py` xét thêm `data/test-00001-of-00002-1b2cd3ea16a2c5b1.parquet` (~302 MB), pin cùng revision trong `configs/bud500.yaml`. Script đọc cột transcript trước, chọn tối đa 100 hàng đầu theo thứ tự hai shard; keyword có ranh giới whitespace sau chuẩn hóa để tránh chọn “thuế” chỉ vì có chuỗi “huế”. Mở rộng phạm vi nguồn được ghi thành version mới; dữ liệu/điểm thí điểm cũ giữ nguyên. Nếu hết hai shard vẫn thiếu 100, báo thiếu dữ liệu và cần thêm tập đánh giá độc lập; không tự lấy thêm train hoặc nới keyword.

Đã xét hết hai shard và tìm được 34 mẫu. Đây là toàn bộ số mẫu khớp bộ keyword trong phạm vi test đã cấu hình, không phải lỗi download. Evaluator chạy cả 34 mẫu để ghi bằng chứng thực tế; `bud500_evaluated` theo dõi việc chấm hợp lệ, còn gate Phase 2 vẫn yêu cầu mục tiêu 100. Khi hai model đã được chấm, lệnh chạy lại không tạo thêm run chỉ vì số mẫu dưới mục tiêu.

Việc chọn diễn ra trước inference trên slice mới và không dựa vào lỗi model. Script giải mã các hàng được chọn thành PCM 16 kHz mono. Audio trên 30 giây bị báo lỗi để tránh chấm trên đầu vào bị cắt ngầm; muốn xử lý cần protocol long-form riêng.

Preparation version `bud500-keyword-v3` lưu tên/hash từng shard, thứ tự chọn, chỉ số hàng, hash manifest và từng WAV. `dataset_sha256` là hash inventory nguồn có thứ tự, không phải hash một archive. Dữ liệu v2 một shard vẫn đọc được. Audio cũ được liên kết bằng hardlink sau khi đối chiếu selection/hash, tránh nhân bản dữ liệu. Root mới là `data/raw/bud500/travel_test_v3/`; VIVOS và protocol metric `vi-v1` giữ nguyên.

```bash
conda activate vietnamese-travel-voicebot
python -m pip install -e '.[bud500]'
bash scripts/run_bud500.sh
```

Nếu gặp `GatedRepoError`/403 “not in the authorized list”, mở [trang dataset](https://huggingface.co/datasets/linhtran92/viet_bud500) khi đã đăng nhập, chọn **Request access/Agree** và chờ chủ dataset cấp quyền. Đăng nhập lại không tự cấp quyền; `hf auth login` chỉ cần khi CLI đang dùng tài khoản khác. Không dán token vào source hoặc chat. Sau khi được duyệt, chạy lại `bash scripts/run_bud500.sh`. Nếu mạng host không tải được, tải **đúng shard test** bằng trình duyệt của tài khoản đã có quyền rồi chạy `bash scripts/run_bud500.sh --parquet /duong/dan/test-....parquet`.

## Cách hiểu kết quả

Đây là **Bud500 test keyword slice**, không phải toàn bộ Bud500 test và không phải benchmark travel voicebot chính thức. Từ khóa trong transcript có thể bắt gặp câu đọc hoặc lời kể, không bảo đảm là yêu cầu lập lịch trình. Không có speaker ID trong schema được dataset card mô tả; manifest dùng `unknown`, vì vậy chưa phân tích theo speaker hoặc xác nhận speaker overlap với dữ liệu train model. Để đo chất lượng tương tác du lịch, vẫn cần một tập câu hỏi nói tự nhiên với transcript chuẩn, nhiều người và điều kiện ghi âm, được cấp quyền rõ ràng.
