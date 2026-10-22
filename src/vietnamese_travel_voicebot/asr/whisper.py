"""Transformers Whisper/PhoWhisper adapter with explicit device behavior."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from vietnamese_travel_voicebot.asr.base import ASRResult
from vietnamese_travel_voicebot.utils.audio import AudioInput


class WhisperASRProvider:
    """Load one pinned Transformers Whisper checkpoint and transcribe audio."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        device: str = "cuda:0",
        dtype: str = "float16",
        language: str = "vi",
        task: str = "transcribe",
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(device)
        self.language = language
        self.task = task
        if (processor is None) != (model is None):
            raise ValueError("processor and model must be provided together")
        if model is None:
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    f"requested device {device}, but CUDA is unavailable; no CPU fallback is enabled"
                )
            if self.device.type == "cuda" and self.device.index is not None:
                if self.device.index >= torch.cuda.device_count():
                    raise RuntimeError(f"requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible")
            try:
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
            except ImportError as exc:
                raise RuntimeError("install the ASR dependencies to load Transformers Whisper") from exc
            torch_dtype = getattr(torch, dtype, None)
            if not isinstance(torch_dtype, torch.dtype):
                raise ValueError(f"unsupported torch dtype: {dtype}")
            processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id, revision=revision, dtype=torch_dtype
            ).to(self.device)
        self.processor = processor
        self.model = model.to(self.device) if hasattr(model, "to") else model
        self.model.eval()

    def transcribe(self, audio: AudioInput) -> ASRResult:
        samples = np.asarray(audio.samples)
        if audio.sample_rate_hz != 16000:
            raise ValueError("WhisperASRProvider requires audio sampled at 16000 Hz")
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("audio samples must be a non-empty mono waveform")
        if not np.issubdtype(samples.dtype, np.number) or not np.isfinite(samples).all():
            raise ValueError("audio samples must contain only finite numeric values")
        processor_args = {
            "sampling_rate": 16000,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }
        waveform = samples.astype(np.float32, copy=False)
        try:
            inputs = self.processor(waveform, **processor_args)
        except TypeError as exc:
            # Keep small injected/test processors compatible when they do not
            # expose Whisper's optional attention-mask argument.
            if "return_attention_mask" not in str(exc):
                raise
            processor_args.pop("return_attention_mask")
            inputs = self.processor(waveform, **processor_args)
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            tokens = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                language=self.language,
                task=self.task,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_seconds = time.perf_counter() - started
        transcript = self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
        return ASRResult(
            transcript=transcript,
            model_id=self.model_id,
            revision=self.revision,
            inference_seconds=inference_seconds,
            peak_gpu_memory_bytes=peak,
        )
