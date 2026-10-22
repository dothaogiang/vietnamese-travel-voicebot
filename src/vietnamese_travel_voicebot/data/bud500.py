"""Build a small, reproducible Bud500 ASR evaluation slice from one Parquet shard."""

from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from dataclasses import asdict
from pathlib import Path

from vietnamese_travel_voicebot.data.manifest import ManifestRecord
from vietnamese_travel_voicebot.data.normalize import normalize_transcript

TRAVEL_TERMS = (
    "du lịch", "khách sạn", "đặt phòng", "tham quan", "địa điểm", "điểm đến",
    "bãi biển", "sân bay", "vé máy bay", "xe khách", "tàu hỏa", "hành trình",
    "lịch trình", "đà lạt", "nha trang", "hội an", "phú quốc", "vịnh hạ long",
    "sa pa", "sapa", "huế", "đà nẵng", "mũi né", "vũng tàu", "quy nhơn",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_bud500_shard(
    parquet_path: str | Path,
    output_root: str | Path,
    *,
    limit: int = 100,
    filter_mode: str = "travel",
    source_revision: str | None = None,
) -> tuple[Path, Path]:
    """Select rows by transcript before inference and export only those rows to PCM WAV."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Bud500 preparation needs pyarrow: python -m pip install '.[bud500]'") from exc
    if limit <= 0:
        raise ValueError("limit must be positive")
    if filter_mode not in ("travel", "all"):
        raise ValueError("filter_mode must be travel or all")
    source = Path(parquet_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Bud500 Parquet shard not found: {source}")
    output = Path(output_root).expanduser().resolve()
    manifest_path = output / "manifest.jsonl"
    metadata_path = output / "metadata.json"
    if manifest_path.exists():
        raise FileExistsError(f"output already exists; choose a new output root: {manifest_path}")
    parquet = pq.ParquetFile(source)
    if not {"audio", "transcription"}.issubset(parquet.schema_arrow.names):
        raise ValueError("Bud500 shard must contain audio and transcription columns")

    selected: dict[int, str] = {}
    row_index = 0
    for batch in parquet.iter_batches(batch_size=512, columns=["transcription"]):
        for raw_text in batch.column(0).to_pylist():
            if isinstance(raw_text, str) and raw_text.strip():
                normalized = normalize_transcript(raw_text)
                if filter_mode == "all" or any(term in normalized for term in TRAVEL_TERMS):
                    selected[row_index] = raw_text.strip()
                    if len(selected) >= limit:
                        break
            row_index += 1
        if len(selected) >= limit:
            break
    if not selected:
        raise ValueError("no matching transcripts in this shard; try --filter all or another shard")

    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[ManifestRecord] = []
    row_index = 0
    for batch in parquet.iter_batches(batch_size=16, columns=["audio"]):
        for audio in batch.column(0).to_pylist():
            if row_index in selected:
                if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
                    raise ValueError(f"row {row_index} has no embedded audio bytes")
                sample_id = f"bud500_test_{row_index:06d}"
                wav_path = audio_dir / f"{sample_id}.wav"
                completed = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", "pipe:0",
                     "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
                    input=audio["bytes"], capture_output=True, check=False,
                )
                if completed.returncode or not wav_path.is_file():
                    raise ValueError(f"could not decode audio row {row_index}: {completed.stderr.decode(errors='replace')[:300]}")
                with wave.open(str(wav_path), "rb") as wav:
                    frames = wav.getnframes()
                    if frames <= 0:
                        raise ValueError(f"empty audio row {row_index}")
                    records.append(ManifestRecord(
                        utterance_id=sample_id,
                        audio_path=wav_path.relative_to(output).as_posix(),
                        transcript=selected[row_index],
                        split="test",
                        speaker_id="unknown",
                        duration_s=frames / wav.getframerate(),
                        sample_rate_hz=wav.getframerate(),
                        channels=wav.getnchannels(),
                        sample_width_bytes=wav.getsampwidth(),
                    ))
            row_index += 1
        if row_index > max(selected):
            break

    if len(records) != len(selected):
        raise ValueError(f"exported {len(records)} of {len(selected)} selected rows")
    metadata_path.write_text(json.dumps({
        "dataset": "Bud500",
        "source": "linhtran92/viet_bud500",
        "source_revision": source_revision,
        "source_shard": source.name,
        "source_shard_sha256": _sha256(source),
        "filter_mode": filter_mode,
        "travel_terms": list(TRAVEL_TERMS) if filter_mode == "travel" else [],
        "selection": "first matching rows in source order, before model inference",
        "selected_row_indices": list(selected),
        "sample_count": len(records),
        "speaker_metadata": "not provided by source card; unknown in this slice",
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        "".join(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest_path, metadata_path
