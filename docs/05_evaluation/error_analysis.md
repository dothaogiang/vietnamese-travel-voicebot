# Error analysis

## VIVOS test, hai full run CPU ngày 2026-10-19

Tính token edit distance trên các transcript đã chuẩn hóa `vi-v1`, ghép hai file prediction bằng `utterance_id` và kiểm tra reference trùng nhau. Đây là phân tích mô tả trên 760 câu VIVOS test, không thay đổi protocol CER/WER.

| Chỉ báo | PhoWhisper-small | Whisper-small |
|---|---:|---:|
| Câu khớp hoàn toàn sau chuẩn hóa | 477/760 | 113/760 |
| Số câu có ít token lỗi hơn model kia | 563 | 18 |
| Số câu hai model cùng mức token lỗi | 179 | 179 |

Trong PhoWhisper-small, speaker `VIVOSDEV10` có WER theo speaker `0.249` trên 40 câu; tiếp theo `VIVOSDEV04` `0.124` và `VIVOSDEV02` `0.119`. Ở Whisper-small, speaker `VIVOSDEV18` có WER `0.625`, `VIVOSDEV10` `0.551`, `VIVOSDEV02` `0.370`. Các số này chỉ xác định nhóm cần nghe kiểm định; chưa đủ chứng cứ gán nguyên nhân cho vùng miền, giọng, tiếng ồn hoặc nhãn sai.

Các câu PhoWhisper có nhiều token lỗi nhất gồm `VIVOSDEV10_262`, `VIVOSDEV10_239`, `VIVOSDEV10_140`, `VIVOSDEV10_072`, `VIVOSDEV10_030`, `VIVOSDEV10_017`, `VIVOSDEV08_129`. Cần nghe audio đối chiếu transcript gốc trước khi phân loại thành lỗi dấu thanh, số, omission, phát âm hoặc annotation. Không đưa transcript/audio có giấy phép vào tài liệu Git.

## Việc còn lại

Nghe kiểm định một tập nhỏ các lỗi nặng và mẫu đúng; ghi loại lỗi cùng `utterance_id`, split, model revision. Tạo tập evaluation du lịch riêng có quyền sử dụng, gồm địa danh/tên riêng, phương tiện, lịch trình, thời gian, số tiền, điều kiện thu âm và giọng nói phù hợp; tính CER/WER và độ chính xác entity quan trọng trên tập này. Không trộn tập VIVOS test vào dữ liệu fine-tuning.

## Bộ nghe kiểm định local

Trong Miniconda, chạy `python scripts/complete_phase2.py` để tạo `experiments/001_asr_baseline/outputs/phase2_audit/review.html` cùng `review.csv`. Mở HTML để nghe audio gốc; CSV cần `review_status=reviewed`, `error_group` và `notes` ở từng hàng. Nhóm cho phép: `correct`, `proper_nouns`, `tones`, `numbers`, `noise`, `omissions`, `substitutions`, `annotation`, `uncertain`. Chỉ gán nguyên nhân khi nghe thấy bằng chứng; chưa xác định thì dùng `uncertain` và ghi lý do. Không sửa reference/test manifest từ worksheet.

Selection gồm 10 câu có nhiều token lỗi nhất của PhoWhisper, ưu tiên WER cao khi cùng số lỗi, cùng 3 mẫu khớp hoàn toàn theo ID để đối chiếu. Hash predictions/manifest khóa selection; đổi dữ liệu cần tạo audit khác. Các file chứa transcript/audio links nằm ngoài Git. Trạng thái nghe chỉ đạt sau khi tất cả mẫu đã có ghi nhận thủ công; code không tự nhận đã kiểm định tai nghe.
