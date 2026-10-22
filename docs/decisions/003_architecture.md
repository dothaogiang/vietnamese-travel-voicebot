# ADR 003: Architecture

- Trạng thái: Chấp nhận làm kiến trúc mục tiêu ban đầu.
- Quyết định: Tách Voice/VAD/ASR/Intent/RAG-Tools/LLM/TTS và API/business logic theo interface có thể thay thế. RAG dùng knowledge tĩnh có nguồn; tools/API dùng dữ liệu realtime hoặc transaction.
- Hệ quả: Module hiện là khung; cần chốt provider và giao thức ở các giai đoạn tiếp theo.

## Bổ sung 2026-10-14: Python package

Source code nằm trong `src/vietnamese_travel_voicebot/`, với các subpackage `data`, `asr`, `evaluation`, `voicebot`, `knowledge`, `training`, `utils`. `pyproject.toml` khai báo `src` layout chuẩn để cài editable và import bằng tên package ổn định, tránh import package chung tên `src`. Các file `api/` và `scripts/` vẫn là entrypoint bên ngoài; logic triển khai thuộc package. Thay đổi này không đổi luồng voicebot mục tiêu.
