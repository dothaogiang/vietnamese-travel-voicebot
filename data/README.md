# Data storage

Các thư mục con chỉ là vị trí local dự kiến. Không đặt audio/dataset thật vào Git. Trước khi sử dụng VIVOS, ghi lại URL/DOI, version, ngày tải, license và mục đích nghiên cứu phi thương mại đã được chấp thuận; so sánh kích thước/checksum archive với metadata nguồn, rồi kiểm tra cấu trúc, định dạng audio, transcript, speaker và split.

VIVOS đã được cài trong `data/raw/vivos/VIVOS` và manifest đã tạo. Có thể tái tạo manifest bằng:

```bash
python scripts/prepare_data.py --dataset-root data/raw/vivos/VIVOS
```

Manifest JSONL trong `data/manifests/` giữ transcript, đường dẫn tương đối và metadata WAV; các file JSONL được Git ignore vì chứa nội dung dataset có license. Script kiểm tra file/transcript, ID trùng, WAV PCM không rỗng và speaker leakage train/test. Nó chưa thay cho rà soát transcript nghe thủ công hoặc hồ sơ license. Không dùng file rỗng làm bằng chứng có split; manifest, audio và model không được commit.

Bud500 pilot 15 mẫu giữ tại `data/raw/bud500/travel_test_v1/`; version 3 xét tối đa hai shard test được chuẩn bị riêng tại `data/raw/bud500/travel_test_v3/`. Audio cũ được tái dùng qua hardlink sau kiểm tra hash. Manifest, metadata, audio và shard nguồn đều nằm ngoài Git. Xem [protocol Bud500](../docs/05_evaluation/bud500_protocol.md); không trộn slice với VIVOS test hoặc coi đó là tập hội thoại du lịch thực tế.
