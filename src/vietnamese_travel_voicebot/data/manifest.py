"""VIVOS prompt parsing and deterministic JSONL manifest generation."""

from __future__ import annotations

import json
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ManifestRecord:
    utterance_id: str
    audio_path: str
    transcript: str
    split: str
    speaker_id: str
    duration_s: float
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int


def _read_split(root: Path, split: str) -> list[ManifestRecord]:
    split_root = root / split
    prompts = split_root / "prompts.txt"
    if not prompts.is_file():
        raise FileNotFoundError(
            f"missing VIVOS prompt file: {prompts}. VIVOS is not installed at this path; "
            "run `python scripts/download_data.py` first or pass the extracted VIVOS root."
        )

    records: list[ManifestRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(prompts.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(f"invalid prompt at {prompts}:{line_number}; expected ID and transcript")
        utterance_id, transcript = parts[0], parts[1].strip()
        if utterance_id in seen:
            raise ValueError(f"duplicate utterance ID {utterance_id!r} in {prompts}")
        seen.add(utterance_id)
        speaker_id = utterance_id.split("_", 1)[0]
        audio_path = split_root / "waves" / speaker_id / f"{utterance_id}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(f"missing audio for {utterance_id}: {audio_path}")
        try:
            with wave.open(str(audio_path), "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                if frames <= 0 or rate <= 0 or channels <= 0 or width not in (1, 2, 3, 4):
                    raise ValueError("unsupported or empty PCM WAV")
                if wav.getcomptype() != "NONE":
                    raise ValueError("compressed WAV is not supported")
        except (wave.Error, EOFError) as exc:
            raise ValueError(f"invalid WAV for {utterance_id}: {audio_path}: {exc}") from exc
        records.append(
            ManifestRecord(
                utterance_id=utterance_id,
                audio_path=audio_path.relative_to(root).as_posix(),
                transcript=transcript,
                split=split,
                speaker_id=speaker_id,
                duration_s=frames / rate,
                sample_rate_hz=rate,
                channels=channels,
                sample_width_bytes=width,
            )
        )
    if not records:
        raise ValueError(f"no records found in {prompts}")
    return sorted(records, key=lambda record: record.utterance_id)


def build_vivos_manifests(dataset_root: str | Path, output_dir: str | Path) -> Mapping[str, Path]:
    """Validate VIVOS train/test audio and write sorted, root-relative JSONL manifests."""
    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_dir).expanduser()
    split_records = {split: _read_split(root, split) for split in ("train", "test")}
    train_ids = {record.utterance_id for record in split_records["train"]}
    test_ids = {record.utterance_id for record in split_records["test"]}
    if train_ids & test_ids:
        raise ValueError(f"utterance IDs appear in both train and test: {sorted(train_ids & test_ids)[:5]}")
    train_speakers = {record.speaker_id for record in split_records["train"]}
    test_speakers = {record.speaker_id for record in split_records["test"]}
    if train_speakers & test_speakers:
        raise ValueError(f"speakers appear in both train and test: {sorted(train_speakers & test_speakers)[:5]}")
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, records in split_records.items():
        path = destination / f"{split}.jsonl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        temporary.replace(path)
        paths[split] = path
    return paths


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    """Read JSONL records and reject malformed rows with line context."""
    records: list[ManifestRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(ManifestRecord(**json.loads(line)))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError(f"invalid manifest row at {path}:{line_number}: {exc}") from exc
    return records
