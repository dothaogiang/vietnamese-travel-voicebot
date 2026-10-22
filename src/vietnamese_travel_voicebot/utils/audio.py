"""Small PCM WAV reader for reproducible local ASR preprocessing."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioInput:
    """Mono float32 audio in [-1, 1], with its current sample rate."""

    samples: np.ndarray
    sample_rate_hz: int


def _decode_pcm(frames: bytes, sample_width_bytes: int) -> np.ndarray:
    if sample_width_bytes == 1:
        return (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width_bytes == 2:
        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width_bytes == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        values = (values ^ 0x800000) - 0x800000
        return values.astype(np.float32) / 8388608.0
    if sample_width_bytes == 4:
        return np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width_bytes} bytes")


def load_audio(path: str | Path, target_sample_rate_hz: int = 16000) -> AudioInput:
    """Decode PCM WAV, average channels, and resample using band-limited TorchAudio."""
    if target_sample_rate_hz <= 0:
        raise ValueError("target_sample_rate_hz must be positive")
    try:
        with wave.open(str(path), "rb") as wav:
            channels, rate, width = wav.getnchannels(), wav.getframerate(), wav.getsampwidth()
            if wav.getcomptype() != "NONE":
                raise ValueError(f"compressed WAV is not supported: {path}")
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"invalid PCM WAV {path}: {exc}") from exc
    if channels <= 0 or rate <= 0 or not frames:
        raise ValueError(f"audio is empty or has invalid metadata: {path}")
    samples = _decode_pcm(frames, width)
    if samples.size % channels:
        raise ValueError(f"truncated interleaved audio frames: {path}")
    samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    if not np.isfinite(samples).all():
        raise ValueError(f"audio contains non-finite samples: {path}")
    if rate != target_sample_rate_hz:
        try:
            import torch
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("Torch and TorchAudio are required to resample audio") from exc
        waveform = torch.from_numpy(samples.copy()).unsqueeze(0)
        samples = torchaudio.functional.resample(waveform, rate, target_sample_rate_hz).squeeze(0).numpy()
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise ValueError(f"audio became empty after preprocessing: {path}")
    return AudioInput(samples=samples, sample_rate_hz=target_sample_rate_hz)
