"""ASR provider contract and CPU-only inference behavior tests."""

import unittest
import tempfile
import json
import importlib.util
import wave
from pathlib import Path

import numpy as np
if importlib.util.find_spec("torch") is not None:
    import torch
else:  # Keep the lightweight core install testable without ASR extras.
    torch = None  # type: ignore[assignment]

from vietnamese_travel_voicebot.asr.base import ASRResult
if torch is not None:
    from vietnamese_travel_voicebot.asr.whisper import WhisperASRProvider
else:
    WhisperASRProvider = None  # type: ignore[assignment,misc]
from vietnamese_travel_voicebot.utils.audio import AudioInput
from vietnamese_travel_voicebot.evaluation.evaluate_asr import evaluate_manifest


class FakeProcessor:
    def __call__(self, audio, sampling_rate, return_tensors):
        if sampling_rate != 16000 or return_tensors != "pt":
            raise AssertionError("unexpected processor arguments")
        return {"input_features": torch.zeros((1, 80, 10))}

    def batch_decode(self, tokens, skip_special_tokens):
        return ["  Đà Nẵng!  "]


class FakeModel:
    def __init__(self):
        self.eval_called = False
        self.generate_kwargs = None

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, input_features, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3]])


@unittest.skipIf(torch is None, "install the asr extra to test model adapter")
class WhisperProviderTests(unittest.TestCase):
    def test_injected_model_transcribes_with_locked_language_task(self):
        model = FakeModel()
        provider = WhisperASRProvider(
            model_id="test/model",
            revision="abc123",
            device="cpu",
            processor=FakeProcessor(),
            model=model,
        )
        result = provider.transcribe(AudioInput(np.zeros(1600, dtype=np.float32), 16000))
        self.assertIsInstance(result, ASRResult)
        self.assertEqual(result.transcript, "Đà Nẵng!")
        self.assertEqual(result.model_id, "test/model")
        self.assertEqual(result.revision, "abc123")
        self.assertTrue(model.eval_called)
        self.assertEqual(model.generate_kwargs["language"], "vi")
        self.assertEqual(model.generate_kwargs["task"], "transcribe")

    def test_rejects_wrong_sample_rate_and_nonfinite_input(self):
        provider = WhisperASRProvider(
            model_id="test/model", revision="abc123", device="cpu",
            processor=FakeProcessor(), model=FakeModel(),
        )
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            provider.transcribe(AudioInput(np.zeros(160, dtype=np.float32), 8000))
        with self.assertRaisesRegex(ValueError, "finite"):
            provider.transcribe(AudioInput(np.array([np.nan], dtype=np.float32), 16000))


class EchoProvider:
    def transcribe(self, audio):
        return ASRResult("Đà Nẵng đẹp", "fake/model", "fake-revision", 0.01, None)


class BrokenProvider:
    def transcribe(self, audio):
        raise RuntimeError("synthetic inference failure")


class ASREvaluationTests(unittest.TestCase):
    def _fixture(self, root):
        (root / "test" / "waves" / "speaker1").mkdir(parents=True)
        wav_path = root / "test" / "waves" / "speaker1" / "speaker1_0001.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 160)
        manifest = root / "test.jsonl"
        manifest.write_text(json.dumps({
            "utterance_id": "speaker1_0001", "audio_path": "test/waves/speaker1/speaker1_0001.wav",
            "transcript": "Đà Nẵng đẹp", "split": "test", "speaker_id": "speaker1",
            "duration_s": 0.01, "sample_rate_hz": 16000, "channels": 1, "sample_width_bytes": 2,
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        return manifest

    def test_writes_per_utterance_predictions_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._fixture(root)
            summary = evaluate_manifest(EchoProvider(), manifest, root, root / "run")
            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(summary["cer"], 0.0)
            self.assertTrue(summary["warmup_enabled"])
            self.assertTrue((root / "run" / "predictions.jsonl").is_file())
            self.assertTrue((root / "run" / "summary.json").is_file())

    def test_records_failed_utterance_without_reporting_it_as_a_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._fixture(root)
            summary = evaluate_manifest(BrokenProvider(), manifest, root, root / "failed-run")
            self.assertEqual(summary["sample_count"], 0)
            self.assertEqual(summary["failed_count"], 1)
            self.assertFalse(summary["evaluation_complete"])
            self.assertIsNone(summary["cer"])
            self.assertEqual(summary["warmup_error"]["error_type"], "RuntimeError")
