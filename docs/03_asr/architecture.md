# ASR architecture

ASR được gọi qua interface provider tại `src/vietnamese_travel_voicebot/asr/base.py`; preprocessing và inference tách riêng để model không nằm trong API. Giai đoạn 1 đã chốt hai ứng viên trong [ADR 001](../decisions/001_model_selection.md), nhưng adapter vẫn là placeholder. Giai đoạn 2 sẽ chốt contract nhận audio 16 kHz mono và trả transcript/metadata, triển khai adapter Transformers rồi đo VRAM, thời gian xử lý và CER/WER trên cùng split.
