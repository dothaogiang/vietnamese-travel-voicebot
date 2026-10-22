# Data storage

Các thư mục con chỉ là vị trí local dự kiến. Không đặt audio/dataset thật vào Git. Trước khi sử dụng VIVOS, ghi lại URL/DOI, version, ngày tải, license và mục đích nghiên cứu phi thương mại đã được chấp thuận; so sánh kích thước/checksum archive với metadata nguồn, rồi kiểm tra cấu trúc, định dạng audio, transcript, speaker và split.

VIVOS đã được cài trong `data/raw/vivos/VIVOS` và manifest đã tạo. Có thể tái tạo manifest bằng:

```bash
python scripts/prepare_data.py --dataset-root data/raw/vivos/VIVOS
```

Manifest JSONL trong `data/manifests/` giữ transcript, đường dẫn tương đối và metadata WAV; các file JSONL được Git ignore vì chứa nội dung dataset có license. Script kiểm tra file/transcript, ID trùng, WAV PCM không rỗng và speaker leakage train/test. Nó chưa thay cho rà soát transcript nghe thủ công hoặc hồ sơ license. Không dùng file rỗng làm bằng chứng có split; manifest, audio và model không được commit.

Bud500 bổ sung được chuẩn bị riêng tại `data/raw/bud500/travel_test_v1/`. Manifest, metadata, audio trích xuất và shard nguồn đều nằm ngoài Git. Xem [protocol Bud500](../docs/05_evaluation/bud500_protocol.md); không trộn Bud500 keyword slice với VIVOS test hoặc gọi đó là tập hội thoại du lịch thực tế.
