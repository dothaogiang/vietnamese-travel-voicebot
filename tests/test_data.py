"""Normalization examples that lock ASR evaluation protocol vi-v1."""

import unicodedata
import json
import importlib.util
import tempfile
import unittest
import wave
from pathlib import Path

from vietnamese_travel_voicebot.data.normalize import normalize_transcript
from vietnamese_travel_voicebot.data.manifest import build_vivos_manifests, read_manifest
from vietnamese_travel_voicebot.utils.audio import load_audio


class TranscriptNormalizationTests(unittest.TestCase):
    def test_preserves_vietnamese_tones_and_digits(self) -> None:
        self.assertEqual(normalize_transcript("ĐÀ   NẴNG, 2026!"), "đà nẵng 2026")

    def test_composes_unicode_and_separates_punctuation(self) -> None:
        decomposed = unicodedata.normalize("NFD", "Huế")
        self.assertEqual(normalize_transcript(f"{decomposed}-Đà Nẵng"), "huế đà nẵng")

    def test_rejects_non_text(self) -> None:
        with self.assertRaisesRegex(TypeError, "transcript must be a string"):
            normalize_transcript(None)  # type: ignore[arg-type]


class VivosManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "vivos"
        for split, items in {
            "train": [("speaker1_0001", "Xin chào Huế")],
            "test": [("speaker2_0001", "Đà Nẵng đẹp")],
        }.items():
            split_dir = self.root / split
            (split_dir / "waves" / items[0][0].split("_", 1)[0]).mkdir(parents=True)
            (split_dir / "prompts.txt").write_text(
                f"{items[0][0]} {items[0][1]}\n", encoding="utf-8"
            )
            with wave.open(
                str(split_dir / "waves" / items[0][0].split("_", 1)[0] / f"{items[0][0]}.wav"),
                "wb",
            ) as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(b"\x00\x00" * 80)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_builds_deterministic_manifests_and_audio_metadata(self) -> None:
        output = Path(self.temp.name) / "manifests"
        paths = build_vivos_manifests(self.root, output)
        rows = read_manifest(paths["test"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].utterance_id, "speaker2_0001")
        self.assertEqual(rows[0].transcript, "Đà Nẵng đẹp")
        self.assertEqual(rows[0].sample_rate_hz, 8000)
        self.assertEqual(rows[0].channels, 1)
        self.assertEqual(json.loads(paths["test"].read_text().splitlines()[0])["split"], "test")

    def test_fails_if_prompt_audio_is_missing(self) -> None:
        (self.root / "test" / "waves" / "speaker2" / "speaker2_0001.wav").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "speaker2_0001"):
            build_vivos_manifests(self.root, Path(self.temp.name) / "out")

    def test_fails_if_speaker_leaks_across_splits(self) -> None:
        test_dir = self.root / "test"
        (test_dir / "waves" / "speaker2" / "speaker2_0001.wav").unlink()
        (test_dir / "waves" / "speaker1").mkdir()
        (test_dir / "prompts.txt").write_text("speaker1_0002 Câu kiểm tra\n", encoding="utf-8")
        with wave.open(str(test_dir / "waves" / "speaker1" / "speaker1_0002.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\x00\x00" * 80)
        with self.assertRaisesRegex(ValueError, "speakers appear in both"):
            build_vivos_manifests(self.root, Path(self.temp.name) / "out")

    @unittest.skipUnless(importlib.util.find_spec("torchaudio"), "install the asr extra to test resampling")
    def test_audio_loader_downmixes_and_resamples_to_requested_rate(self) -> None:
        path = self.root / "test" / "waves" / "speaker2" / "speaker2_0001.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            frame = (1000).to_bytes(2, "little", signed=True) + (0).to_bytes(2, "little", signed=True)
            wav.writeframes(frame * 80)
        audio = load_audio(path, target_sample_rate_hz=16000)
        self.assertEqual(audio.sample_rate_hz, 16000)
        self.assertEqual(audio.samples.shape, (160,))
        self.assertGreater(float(audio.samples.max()), 0.0)
        self.assertAlmostEqual(float(audio.samples.mean()), 500 / 32768, places=3)
