"""Stable contracts shared by replaceable ASR backends."""

from dataclasses import dataclass
from typing import Protocol

from vietnamese_travel_voicebot.utils.audio import AudioInput


@dataclass(frozen=True)
class ASRResult:
    transcript: str
    model_id: str
    revision: str
    inference_seconds: float
    peak_gpu_memory_bytes: int | None = None


class ASRProvider(Protocol):
    def transcribe(self, audio: AudioInput) -> ASRResult:
        """Transcribe mono float32 audio sampled at the provider's expected rate."""
        ...
