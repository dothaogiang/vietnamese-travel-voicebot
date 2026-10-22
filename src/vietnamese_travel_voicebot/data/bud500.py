"""Build versioned, reproducible Bud500 keyword slices from bounded test shards."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from vietnamese_travel_voicebot.data.manifest import ManifestRecord, read_manifest
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


def select_bud500_rows(source: Path, limit: int, terms: list[str], filter_mode: str = "travel") -> dict[int, str]:
    """Select by transcript only, before decoding audio or observing predictions."""
    import pyarrow.parquet as pq
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected = {}
    row_index = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=512, columns=["transcription"]):
        for raw_text in batch.column(0).to_pylist():
            if isinstance(raw_text, str) and raw_text.strip():
                normalized = normalize_transcript(raw_text)
                if normalized and (filter_mode == "all" or any(f" {term} " in f" {normalized} " for term in terms)):
                    selected[row_index] = raw_text.strip()
                    if len(selected) >= limit:
                        return selected
            row_index += 1
    return selected


def prepare_bud500_shard(
    parquet_path: str | Path,
    output_root: str | Path,
    *,
    limit: int = 100,
    filter_mode: str = "travel",
    source_revision: str | None = None,
    travel_terms: tuple[str, ...] | list[str] = TRAVEL_TERMS,
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
    terms = [normalize_transcript(term) for term in travel_terms if term.strip()]
    if filter_mode == "travel" and not terms:
        raise ValueError("travel filter requires non-empty travel_terms")
    source = Path(parquet_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Bud500 Parquet shard not found: {source}")
    output = Path(output_root).expanduser().resolve()
    manifest_path = output / "manifest.jsonl"
    metadata_path = output / "metadata.json"
    if manifest_path.exists():
        raise FileExistsError(f"output already exists; choose a new output root: {manifest_path}")
    source_sha256 = _sha256(source)
    parquet = pq.ParquetFile(source)
    if not {"audio", "transcription"}.issubset(parquet.schema_arrow.names):
        raise ValueError("Bud500 shard must contain audio and transcription columns")

    selected = select_bud500_rows(source, limit, terms, filter_mode)
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
                sample_id = f"bud500_test_{source_sha256[:12]}_{row_index:06d}"
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
                    if frames <= 0 or len(wav.readframes(frames)) != frames * wav.getnchannels() * wav.getsampwidth():
                        raise ValueError(f"empty audio row {row_index}")
                    if frames / wav.getframerate() > 30:
                        raise ValueError(f"row {row_index} exceeds the supported 30-second utterance window; define a long-form protocol before evaluating it")
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
    manifest_content = "".join(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    metadata_path.write_text(json.dumps({
        "preparation_version": "bud500-keyword-v2",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Bud500",
        "source": "linhtran92/viet_bud500",
        "source_revision": source_revision,
        "source_shard": source.name,
        "source_shard_sha256": source_sha256,
        "filter_mode": filter_mode,
        "travel_terms": terms if filter_mode == "travel" else [],
        "selection": "first matching rows in source order, before model inference",
        "selected_row_indices": list(selected),
        "sample_count": len(records),
        "manifest_sha256": hashlib.sha256(manifest_content.encode("utf-8")).hexdigest(),
        "audio_sha256": {record.audio_path: _sha256(output / record.audio_path) for record in records},
        "speaker_metadata": "not provided by source card; unknown in this slice",
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    temporary_manifest.write_text(manifest_content, encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest_path, metadata_path


def _inventory_fingerprint(shards: list[dict]) -> str:
    inventory = [{"name": shard["name"], "sha256": shard["sha256"]} for shard in shards]
    return hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def bud500_source_sha256(metadata: dict) -> str:
    """Use the ordered source-inventory digest for multi-shard datasets."""
    return metadata["dataset_sha256"] if "dataset_sha256" in metadata else metadata["source_shard_sha256"]


def prepare_bud500_testset(
    parquet_paths: list[Path], output_root: str | Path, *, limit: int = 100,
    source_revision: str | None = None, travel_terms: tuple[str, ...] | list[str] = TRAVEL_TERMS,
    reusable_slices: list[Path] | tuple = (), filter_mode: str = "travel",
) -> tuple[Path, Path]:
    """Combine matching rows from bounded test shards into a new, versioned slice."""
    if not parquet_paths or limit <= 0:
        raise ValueError("provide test shards and a positive limit")
    terms = [normalize_transcript(term) for term in travel_terms if term.strip()]
    if filter_mode not in ("travel", "all") or (filter_mode == "travel" and not terms):
        raise ValueError("invalid filter or empty travel terms")
    root = Path(output_root).resolve()
    if (root / "manifest.jsonl").exists():
        raise FileExistsError("slice already exists; choose a new output root")
    (root / "audio").mkdir(parents=True, exist_ok=True)
    reusable = []
    for candidate in reusable_slices:
        if (Path(candidate) / "manifest.jsonl").is_file():
            reusable.append((Path(candidate), validate_bud500_slice(candidate)))
    records, shards = [], []
    with tempfile.TemporaryDirectory(prefix="bud500-export-", dir=root.parent) as temporary:
        for source in map(Path, parquet_paths):
            if len(records) >= limit:
                break
            source_hash = _sha256(source)
            if any(shard["sha256"] == source_hash for shard in shards):
                raise ValueError("duplicate source shard; cannot count the same data twice")
            wanted = select_bud500_rows(source, limit - len(records), terms, filter_mode)
            shards.append({"name": source.name, "sha256": source_hash, "selected_row_indices": list(wanted)})
            if not wanted:
                continue
            cached = next((path for path, metadata in reusable
                           if metadata.get("source_shard_sha256") == source_hash
                           and metadata.get("filter_mode") == filter_mode
                           and metadata.get("travel_terms") == (terms if filter_mode == "travel" else [])
                           and metadata.get("selected_row_indices", [])[:len(wanted)] == list(wanted)), None)
            if cached is None:
                cached = Path(temporary) / source_hash
                prepare_bud500_shard(source, cached, limit=limit - len(records), source_revision=source_revision,
                                     travel_terms=terms, filter_mode=filter_mode)
            candidates = read_manifest(cached / "manifest.jsonl")[:len(wanted)]
            if [record.transcript for record in candidates] != list(wanted.values()):
                raise ValueError("cached transcripts do not match source selection")
            for record in candidates:
                target = root / "audio" / f"{record.utterance_id}.wav"
                original = cached / record.audio_path
                if target.exists():
                    if _sha256(target) != _sha256(original):
                        raise ValueError(f"existing audio differs: {target.name}")
                else:
                    target.hardlink_to(original)
                records.append(replace(record, audio_path=target.relative_to(root).as_posix()))
    if not records:
        raise ValueError("no travel keyword matches in supplied test shards")
    content = "".join(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    metadata = {
        "preparation_version": "bud500-keyword-v3", "prepared_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Bud500", "source": "linhtran92/viet_bud500", "source_revision": source_revision,
        "source_shards": shards, "dataset_sha256": _inventory_fingerprint(shards),
        "filter_mode": filter_mode, "travel_terms": terms if filter_mode == "travel" else [],
        "selection": "first matching rows in configured test-shard order, before model inference",
        "sample_count": len(records), "requested_sample_count": limit,
        "manifest_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "audio_sha256": {record.audio_path: _sha256(root / record.audio_path) for record in records},
        "speaker_metadata": "not provided by source card; unknown in this slice",
    }
    meta_path, manifest = root / "metadata.json", root / "manifest.jsonl"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary_manifest = manifest.with_suffix(".jsonl.tmp")
    temporary_manifest.write_text(content, encoding="utf-8")
    temporary_manifest.replace(manifest)
    return manifest, meta_path


def validate_bud500_slice(output_root: str | Path) -> dict:
    """Verify prepared manifest and audio checksums before reusing a cached slice."""
    root = Path(output_root).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    manifest = root / "manifest.jsonl"
    if metadata.get("preparation_version") not in ("bud500-keyword-v2", "bud500-keyword-v3") or metadata.get("manifest_sha256") != _sha256(manifest):
        raise ValueError("Bud500 manifest/provenance is missing, changed or uses an old preparation version; prepare a new slice")
    if metadata.get("preparation_version") == "bud500-keyword-v3" and metadata.get("dataset_sha256") != _inventory_fingerprint(metadata["source_shards"]):
        raise ValueError("Bud500 source inventory hash is invalid")
    records = read_manifest(manifest)
    if not records or len(records) != metadata.get("sample_count") or len({record.utterance_id for record in records}) != len(records):
        raise ValueError("Bud500 manifest count/IDs are invalid")
    for record in records:
        path = (root / record.audio_path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or metadata.get("audio_sha256", {}).get(record.audio_path) != _sha256(path):
            raise ValueError(f"Bud500 audio missing or changed: {record.utterance_id}")
        if record.split != "test" or record.sample_rate_hz != 16000 or record.channels != 1 or record.sample_width_bytes != 2:
            raise ValueError(f"invalid Bud500 audio contract: {record.utterance_id}")
    return metadata
