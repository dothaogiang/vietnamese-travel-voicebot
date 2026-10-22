"""Reproducible manifest-based ASR evaluation and prediction artifact writing."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from vietnamese_travel_voicebot.asr.base import ASRProvider
from vietnamese_travel_voicebot.data.manifest import read_manifest
from vietnamese_travel_voicebot.data.normalize import normalize_transcript
from vietnamese_travel_voicebot.evaluation.metrics import calculate_asr_metrics
from vietnamese_travel_voicebot.utils.audio import load_audio
from vietnamese_travel_voicebot.utils.versioning import code_version


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def evaluate_manifest(
    provider: ASRProvider,
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    target_sample_rate_hz: int = 16000,
    limit: int | None = None,
    dataset_sha256: str | None = None,
    dataset_name: str = "VIVOS",
    run_metadata: dict[str, Any] | None = None,
    warmup: bool = True,
) -> dict[str, Any]:
    """Run fixed-order inference, then save predictions and corpus micro metrics."""
    manifest = Path(manifest_path).resolve()
    root = Path(dataset_root).resolve()
    output = Path(output_dir)
    records = read_manifest(manifest)
    manifest_sample_count = len(records)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when supplied")
        records = records[:limit]
    if not records:
        raise ValueError(f"manifest contains no evaluation records: {manifest}")

    warmup_seconds = 0.0
    warmup_error: dict[str, str] | None = None
    if warmup:
        warmup_record = records[0]
        warmup_started = time.perf_counter()
        try:
            relative_audio = Path(warmup_record.audio_path)
            warmup_path = (root / relative_audio).resolve()
            if relative_audio.is_absolute() or not warmup_path.is_relative_to(root):
                raise ValueError(f"manifest audio path escapes dataset root: {warmup_record.audio_path}")
            provider.transcribe(load_audio(warmup_path, target_sample_rate_hz=target_sample_rate_hz))
        except Exception as exc:
            warmup_error = {"error_type": type(exc).__name__, "error": str(exc)}
        warmup_seconds = time.perf_counter() - warmup_started

    prediction_path = output / "predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    references: list[str] = []
    hypotheses: list[str] = []
    durations: list[float] = []
    processing_durations: list[float] = []
    evaluated_audio_seconds: list[float] = []
    peaks: list[int] = []
    failures: list[dict[str, str]] = []
    model_id: str | None = None
    model_revision: str | None = None
    with prediction_path.open("w", encoding="utf-8") as predictions:
        for record in records:
            processing_started = time.perf_counter()
            try:
                relative_audio = Path(record.audio_path)
                audio_path = (root / relative_audio).resolve()
                if relative_audio.is_absolute() or not audio_path.is_relative_to(root):
                    raise ValueError(f"manifest audio path escapes dataset root: {record.audio_path}")
                audio = load_audio(audio_path, target_sample_rate_hz=target_sample_rate_hz)
                result = provider.transcribe(audio)
            except Exception as exc:
                failure = {
                    "utterance_id": record.utterance_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                predictions.write(json.dumps({"status": "error", **failure}, ensure_ascii=False, sort_keys=True) + "\n")
                predictions.flush()
                continue
            references.append(record.transcript)
            hypotheses.append(result.transcript)
            durations.append(result.inference_seconds)
            processing_seconds = time.perf_counter() - processing_started
            processing_durations.append(processing_seconds)
            evaluated_audio_seconds.append(record.duration_s)
            model_id, model_revision = result.model_id, result.revision
            if result.peak_gpu_memory_bytes is not None:
                peaks.append(result.peak_gpu_memory_bytes)
            reference_norm = normalize_transcript(record.transcript)
            hypothesis_norm = normalize_transcript(result.transcript)
            row = {
                "status": "ok",
                "utterance_id": record.utterance_id,
                "speaker_id": record.speaker_id,
                "reference": record.transcript,
                "hypothesis": result.transcript,
                "reference_normalized": reference_norm,
                "hypothesis_normalized": hypothesis_norm,
                "inference_seconds": result.inference_seconds,
                "processing_seconds": processing_seconds,
                "audio_duration_seconds": record.duration_s,
                "rtf": processing_seconds / record.duration_s,
            }
            predictions.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    metrics = calculate_asr_metrics(references, hypotheses) if references else None
    total_audio_seconds = sum(evaluated_audio_seconds)
    try:
        import torch
        torch_version = torch.__version__
        cuda_build = torch.version.cuda
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_version, cuda_build, cuda_available = None, None, False
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    summary: dict[str, Any] = {
        "protocol_version": "vi-v1",
        "dataset": dataset_name,
        "split": records[0].split,
        "run_kind": "smoke_subset" if limit is not None else "full_manifest",
        "requested_sample_count": len(records),
        "sample_count": metrics.sample_count if metrics else 0,
        "failed_count": len(failures),
        "evaluation_complete": not failures,
        "full_manifest_evaluated": limit is None and len(records) == manifest_sample_count,
        "warmup_enabled": warmup,
        "warmup_seconds": warmup_seconds,
        "warmup_error": warmup_error,
        "failures": failures,
        "cer": metrics.cer if metrics else None,
        "wer": metrics.wer if metrics else None,
        "reference_char_count": metrics.reference_char_count if metrics else 0,
        "reference_token_count": metrics.reference_token_count if metrics else 0,
        "normalization_version": metrics.normalization_version if metrics else "vi-v1",
        "model_id": model_id,
        "model_revision": model_revision,
        "manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "dataset_sha256": dataset_sha256,
        "run_metadata": run_metadata or {},
        "total_audio_seconds": total_audio_seconds,
        "total_inference_seconds": sum(durations),
        "total_processing_seconds": sum(processing_durations),
        "mean_inference_seconds": sum(durations) / len(durations) if durations else None,
        "mean_rtf": sum(processing_durations) / total_audio_seconds if total_audio_seconds else None,
        "peak_gpu_memory_bytes": max(peaks) if peaks else None,
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "conda_prefix": sys.prefix if (Path(sys.prefix) / "conda-meta").is_dir() else None,
            "platform": platform.platform(),
            "torch": torch_version,
            "torch_cuda_build": cuda_build,
            "cuda_available": cuda_available,
            "transformers": transformers_version,
        },
        "created_unix_seconds": time.time(),
        "code_version": code_version(),
    }
    _atomic_json(output / "summary.json", summary)
    return summary
