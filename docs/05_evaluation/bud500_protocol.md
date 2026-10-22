# Bud500: tập đánh giá bổ sung

## Nguồn và quyền dùng

[Trang Bud500 gốc trên Hugging Face](https://huggingface.co/datasets/linhtran92/viet_bud500) mô tả corpus ASR tiếng Việt khoảng 500 giờ, nhiều chủ đề (có du lịch), các split train/validation/test, mỗi hàng có `audio` và `transcription`. Repo **gated**: tài khoản phải chấp nhận điều kiện truy cập. Nhãn license trên Hub là `cc-by-nc-sa-4.0`, trong thân dataset card lại có đoạn Apache-2.0; hai thông tin này chưa nhất quán. Chỉ dùng cho nghiên cứu phi thương mại trong dự án này và xác minh quyền cụ thể trước khi phân phối/đưa vào sản phẩm. Không đưa audio, transcript hoặc parquet lên Git.

## Chọn mẫu và tái lập

`scripts/prepare_bud500.py` chỉ tải `data/test-00000-of-00002-531c1d81edb57297.parquet` (shard đầu của test, khoảng 306 MB). Script đọc cột transcript trước, chọn tối đa 100 hàng đầu chứa một trong các từ khóa du lịch cố định trong `bud500.py`. Việc chọn này diễn ra trước ASR, không dựa trên dự đoán hoặc CER/WER. Sau đó script mới giải mã audio của các hàng được chọn bằng ffmpeg thành PCM 16 kHz mono. Manifest ghi chỉ số hàng gốc trong ID; metadata ghi shard SHA-256, revision nếu tải từ Hub, bộ từ khóa và toàn bộ chỉ số được chọn. Giữ nguyên VIVOS test và protocol `vi-v1`; Bud500 được báo cáo riêng.

```bash
conda activate vietnamese-travel-voicebot
python -m pip install -e '.[bud500]'
bash scripts/run_bud500.sh
```

Nếu gặp `GatedRepoError`/403 “not in the authorized list”, mở [trang dataset](https://huggingface.co/datasets/linhtran92/viet_bud500) khi đã đăng nhập, chọn **Request access/Agree** và chờ chủ dataset cấp quyền. Đăng nhập lại không tự cấp quyền; `hf auth login` chỉ cần khi CLI đang dùng tài khoản khác. Không dán token vào source hoặc chat. Sau khi được duyệt, chạy lại `bash scripts/run_bud500.sh`. Nếu mạng host không tải được, tải **đúng shard test** bằng trình duyệt của tài khoản đã có quyền rồi chạy `bash scripts/run_bud500.sh --parquet /duong/dan/test-....parquet`.

## Cách hiểu kết quả

Đây là **Bud500 test keyword slice**, không phải toàn bộ Bud500 test và không phải benchmark travel voicebot chính thức. Từ khóa trong transcript có thể bắt gặp câu đọc hoặc lời kể, không bảo đảm là yêu cầu lập lịch trình. Không có speaker ID trong schema được dataset card mô tả; manifest dùng `unknown`, vì vậy chưa phân tích theo speaker hoặc xác nhận speaker overlap với dữ liệu train model. Để đo chất lượng tương tác du lịch, vẫn cần một tập câu hỏi nói tự nhiên với transcript chuẩn, nhiều người và điều kiện ghi âm, được cấp quyền rõ ràng.
